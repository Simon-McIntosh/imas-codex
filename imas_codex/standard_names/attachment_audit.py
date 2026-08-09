"""Retroactive re-validation of source→name attachments already in the graph.

``workers._is_attachment_consistent`` decides whether a DD source path may
realize a standard name. It is evaluated at COMPOSE time only, so an attachment
written before a rule existed — or written by one of the paths that never
consulted it (legacy source migration, derived-parent seeders, and provenance
rebuilders that MERGE edges directly) — is never revisited. Current migration
paths guard a nonempty explicit cohort before their compare-and-set write, but
historical pairings still need this retroactive pass. A decision cached
durably and never re-evaluated when the deciding logic improves is permanently
wrong: a strike-point source stayed attached to a camera-orientation name on an
ACCEPTED standard name even though the guard rejects that pair today.

This module re-asks the guard's question of every attachment in the graph and
acts on the inconsistent ones, which makes the guard's reach retroactive and
covers every writer at once rather than one call site at a time.

**What happens to a rejected edge.** The attachment is detached — the
``PRODUCED_NAME`` edge, its ``produced_sn_id`` scalar mirror, the DD-side
``HAS_STANDARD_NAME`` projection and the name's ``source_paths`` entry — and the
freed source is returned to ``status='extracted'`` so the generate pool composes
a correct name for it. Nothing is deleted beyond those projections: the source
node, the name node, its ``REFINED_FROM`` chain and its ``DocsRevision``
snapshots all stay, and every detachment is recorded as a
``StandardNameChange`` (``operation='detach_inconsistent_attachment'``) linked
from the name, so the event is auditable and the history is intact.

**Compose semantics for the pairwise rule.** Most of the guard's rules are
order-independent: they look at the one path, the one name and the two units.
One is not — the distinct-vector rule consults ``existing_sources``, and at
compose time (``_process_attachments_core``) that list accumulates only the
sources ALREADY ACCEPTED for the name, so of a mutually-conflicting group the
first is kept and the rest rejected: one representative survives. The audit
reproduces that by walking each name's attachments in a fixed order and
evaluating every one against the accumulated ACCEPTED siblings, never against
the name's whole source set. Handing each member the full set instead would
reject all N of a conflicting group — including the representative compose would
have kept — and strip the name of its entire justification. The order must be
deterministic (paths sorted) because the read returns rows in Neo4j's order:
without it, two passes would detach different members of the same group.
Evaluating against the accumulated set needs no rule taxonomy — an
order-independent rule ignores the sibling list, so its verdict is unchanged.

**Accepted names.** An accepted name losing a source is a real event, not
routine hygiene: the name is catalog-authoritative and may already be published.
Following the precedent of ``sn run --reset-to`` / ``sn prune``, attachments on
``name_stage='accepted'`` names are reported but NOT detached unless the caller
passes ``include_accepted=True``.

**A whole-name wipeout is a NAME defect.** When EVERY attachment of a name is
rejected by the SAME rule, the sources agree with each other and with the DD and
it is the NAME that is the outlier — e.g. an ``…_of_ion_state`` name whose every
source is a species-level ``…/ion/element/atoms_n`` path claims a state
resolution none of its sources has. Detaching them would strip an accepted name
of its justification and rewind every source for paid re-composition to repair
what one ``imas-codex sn edit <name> --rename … --reason …`` fixes. Those names
are classified into ``names_misnamed`` and their attachments are EXCLUDED from
the detach set, so the audit hands the operator a rename worklist instead of
destroying the evidence. Two judgements are baked in:

* **Same rule, not any rule.** A uniform failure is a consensus of sources
  against one name, which is exactly what a rename repairs. Attachments failing
  for MIXED reasons are an incoherent source set — no single rename addresses
  them — so they stay on the ordinary detach-and-recompose path.
* **At least two attachments.** One source is no corroboration: with a single
  attachment there is no evidence about which side is wrong, the recompose costs
  one call, and that lone case is precisely what the detach path was written for
  (a strike-point source on a camera-orientation name). Because the pairwise
  rule now keeps a representative, a wipeout can only be caused by an
  order-independent rule — the uniform-rule reading is well defined.

``names_orphaned`` stays, and is not a duplicate of ``names_misnamed``: it is
measured on the graph AFTER the detach and catches the residue the classifier
deliberately lets through — chiefly the mixed-rule wipeout — so a name that
ended up with no source at all is still surfaced for a human decision rather
than auto-superseded. The two sets cannot overlap, since a misnamed name's
attachments are never detached.

Idempotent: a detached attachment no longer matches the selector, and an
attachment the guard accepts is never touched, so a second pass acts on nothing.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.models import NameStage
from imas_codex.standard_names.source_authority import (
    lock_participants,
    normalize_manifest_hash_binding,
    payload_hash,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AttachmentPairingGuardResult",
    "AttachmentVerdict",
    "AttachmentAuditResult",
    "NameLevelDefect",
    "audit_attachments",
    "guard_source_pairings",
    "recover_terminal_attachment",
    "recover_terminal_attachments",
    "reconcile_attachment_consistency",
]

_TERMINAL_RECOVERY_STAGES: frozenset[str] = frozenset(
    {
        NameStage.superseded.value,
        NameStage.exhausted.value,
        NameStage.contested.value,
    }
)

_TERMINAL_RECOVERY_SOURCE_STATUSES: frozenset[str] = frozenset({"composed", "attached"})

#: Name stages whose attachments are catalog-authoritative. Detaching a source
#: from one of these is gated behind ``include_accepted``, mirroring how the
#: destructive ``sn`` operations gate the same state.
_PROTECTED_NAME_STAGES: frozenset[str] = frozenset({"accepted"})

#: Terminal stages whose edges are the deliberate provenance record. A supersede
#: leaves the predecessor's ``PRODUCED_NAME`` / ``HAS_STANDARD_NAME`` edges intact
#: on purpose (see ``fold_name_into``) so the deprecation stub resolves to the
#: live successor; those edges assert history, not a live claim that the source
#: realizes that name. Detaching them would delete provenance and, worse, would
#: reset a source that has already produced a correct live name. Reported, never
#: acted on.
_HISTORICAL_NAME_STAGES: frozenset[str] = frozenset({"superseded"})

#: How many attachments a name needs before a uniform rejection is read as a
#: NAME-level defect rather than an attachment defect. Two is the smallest
#: corroboration: one source rejected on its own says nothing about which side is
#: wrong, and rewinding it costs a single re-compose.
_MIN_WIPEOUT_ATTACHMENTS = 2

#: Every attachment the graph asserts, with the unit context the dimensionality
#: rule needs on both sides. Sibling source paths are NOT collected here: the
#: pairwise rule must see only the siblings already ACCEPTED (compose semantics,
#: see the module docstring), which the audit accumulates from these very rows —
#: every attachment of a name is one of them.
#:
#: ``other_live_names`` counts the OTHER names this source still produces. A
#: source that also backs a live name has already been re-composed correctly;
#: only the stale extra edge should go, and the source must NOT be rewound to
#: ``extracted`` — that would strand the good name it already produced.
_ATTACHMENTS_QUERY = """
MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
{scope}
MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
OPTIONAL MATCH (dd)-[:HAS_UNIT]->(du:Unit)
WITH src, sn, dd, collect(DISTINCT du.id) AS dd_relationship_units
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(nu:Unit)
WITH src, sn, dd, dd_relationship_units,
     collect(DISTINCT nu.id) AS sn_relationship_units
OPTIONAL MATCH (dd)-[:IN_IDS]->(ids:IDS)
WITH src, sn, dd, dd_relationship_units, sn_relationship_units,
     collect(DISTINCT ids.dd_version) AS dd_versions
OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(other:StandardName)
WHERE other.id <> sn.id
  AND NOT (coalesce(other.name_stage, '') IN $historical)
WITH src, sn, dd, dd_relationship_units, sn_relationship_units, dd_versions,
     count(DISTINCT other) AS other_live_names
RETURN src.id            AS source_node_id,
       dd.id             AS dd_path,
       sn.id             AS sn_id,
       sn.name_stage     AS name_stage,
       sn.origin         AS origin,
       CASE size(dd_relationship_units)
         WHEN 0 THEN dd.unit
         WHEN 1 THEN dd_relationship_units[0]
         ELSE null
       END AS dd_unit,
       dd.unit           AS dd_declared_unit,
       dd_relationship_units,
       CASE size(dd_versions)
         WHEN 1 THEN dd_versions[0]
         ELSE null
       END AS dd_version,
       CASE size(sn_relationship_units)
         WHEN 0 THEN sn.unit
         WHEN 1 THEN sn_relationship_units[0]
         ELSE null
       END AS sn_unit,
       sn_relationship_units,
       other_live_names
"""

#: Scope clause narrowing the read to one name. Used by the write-time gate on
#: paths that migrate a historical source set onto a NEW name, where a full
#: corpus audit would be far too expensive for a hot path.
_ONE_NAME_SCOPE = "WHERE sn.id = $sn_id"

_PAIRING_GUARD_QUERY = """
MATCH (sn:StandardName {id: $sn_id})
OPTIONAL MATCH (bound:StandardNameSource)-[:PRODUCED_NAME]->(sn)
OPTIONAL MATCH (bound)-[:FROM_DD_PATH]->(bound_dd:IMASNode)
WITH sn, collect(DISTINCT bound_dd.id) AS existing_dd_paths
UNWIND $source_ids AS source_id
OPTIONAL MATCH (source:StandardNameSource {id: source_id})
OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
OPTIONAL MATCH (dd)-[:HAS_UNIT]->(du:Unit)
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(nu:Unit)
RETURN source_id,
       source.source_type AS source_type,
       dd.id AS dd_path,
       coalesce(du.id, dd.unit) AS dd_unit,
       coalesce(nu.id, sn.unit) AS sn_unit,
       EXISTS { (source)-[:PRODUCED_NAME]->(sn) } AS already_bound,
       existing_dd_paths,
       sn.name_stage AS name_stage
"""

#: Detach one attachment: the provenance edge, the DD-side projection, and the
#: name's ``source_paths`` entry.
#:
#: The source is rewound to ``extracted`` — with its attempt budget reset, since
#: the attempts it spent produced an attachment the guard rejects — ONLY when
#: this was its last live name (``item.reroute``). A source that still produces
#: another live name has already been composed correctly; rewinding it would
#: strand that good name, so only the stale edge is removed and its status is
#: left alone. The ``produced_sn_id`` mirror is re-pointed at whatever live name
#: remains, or cleared when none does, so the scalar never names a detached edge.
_DETACH_QUERY = """
UNWIND $items AS item
MATCH (src:StandardNameSource {id: item.source_node_id})
MATCH (sn:StandardName {id: item.sn_id})
OPTIONAL MATCH (src)-[pn:PRODUCED_NAME]->(sn)
DELETE pn
WITH src, sn, item
OPTIONAL MATCH (dd:IMASNode {id: item.dd_path})-[hsn:HAS_STANDARD_NAME]->(sn)
DELETE hsn
WITH src, sn, item
SET sn.source_paths = [
      p IN coalesce(sn.source_paths, [])
      WHERE NOT (p = 'dd:' + item.dd_path OR p = item.dd_path)
    ]
WITH src, item
OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(remaining:StandardName)
WHERE NOT (coalesce(remaining.name_stage, '') IN $historical)
WITH src, item, collect(remaining.id) AS remaining_ids
SET src.produced_sn_id = CASE WHEN size(remaining_ids) > 0
                              THEN remaining_ids[0] ELSE null END,
    src.status = CASE WHEN item.reroute THEN 'extracted' ELSE src.status END,
    src.composed_at = CASE WHEN item.reroute THEN null ELSE src.composed_at END,
    src.attempt_count = CASE WHEN item.reroute THEN 0 ELSE src.attempt_count END,
    src.claimed_at = CASE WHEN item.reroute THEN null ELSE src.claimed_at END,
    src.claim_token = CASE WHEN item.reroute THEN null ELSE src.claim_token END
RETURN count(*) AS detached
"""


#: Remove a DD-side realization that has no provenance node behind it. There is
#: no source to rewind — the projection and the name's ``source_paths`` entry are
#: the whole of the assertion, and both go.
_DETACH_PROJECTION_QUERY = """
MATCH (dd:IMASNode {id: $dd_path})-[hsn:HAS_STANDARD_NAME]->(sn:StandardName {id: $sn_id})
DELETE hsn
WITH sn
SET sn.source_paths = [
      p IN coalesce(sn.source_paths, [])
      WHERE NOT (p = 'dd:' + $dd_path OR p = $dd_path)
    ]
RETURN count(*) AS detached
"""


_TERMINAL_RECOVERY_ELIGIBILITY_QUERY = """
MATCH (src:StandardNameSource {id: $source_node_id})
      -[:FROM_DD_PATH]->(dd:IMASNode {id: $dd_path})
MATCH (src)-[:PRODUCED_NAME]->(sn:StandardName {id: $sn_id})
MATCH (dd)-[:HAS_STANDARD_NAME]->(sn)
WHERE src.source_type = 'dd'
  AND src.source_id = $dd_path
  AND src.status IN $source_statuses
  AND src.claimed_at IS NULL
  AND src.claim_token IS NULL
  AND src.produced_sn_id = sn.id
  AND sn.name_stage IN $terminal_stages
  AND COUNT { (src)-[:PRODUCED_NAME]->(:StandardName) } = 1
  AND (
    'dd:' + $dd_path IN coalesce(sn.source_paths, [])
    OR $dd_path IN coalesce(sn.source_paths, [])
    OR (
      size(coalesce(sn.source_paths, [])) = 0
      AND COUNT { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) } = 1
      AND COUNT { (:IMASNode)-[:HAS_STANDARD_NAME]->(sn) } = 1
    )
  )
RETURN src.id AS source_node_id,
       src.status AS source_status,
       coalesce(src.attempt_count, 0) AS attempt_count,
       src.last_error AS last_error,
       sn.name_stage AS name_stage
"""


_TERMINAL_RECOVERY_QUERY = """
MATCH (src:StandardNameSource {id: $source_node_id})
      -[:FROM_DD_PATH]->(dd:IMASNode {id: $dd_path})
MATCH (src)-[pn:PRODUCED_NAME]->(sn:StandardName {id: $sn_id})
MATCH (dd)-[hsn:HAS_STANDARD_NAME]->(sn)
WHERE src.source_type = 'dd'
  AND src.source_id = $dd_path
  AND src.status IN $source_statuses
  AND src.claimed_at IS NULL
  AND src.claim_token IS NULL
  AND src.produced_sn_id = sn.id
  AND sn.name_stage IN $terminal_stages
  AND COUNT { (src)-[:PRODUCED_NAME]->(:StandardName) } = 1
  AND (
    'dd:' + $dd_path IN coalesce(sn.source_paths, [])
    OR $dd_path IN coalesce(sn.source_paths, [])
    OR (
      size(coalesce(sn.source_paths, [])) = 0
      AND COUNT { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) } = 1
      AND COUNT { (:IMASNode)-[:HAS_STANDARD_NAME]->(sn) } = 1
    )
  )
WITH src, dd, sn, pn, hsn,
     src.status AS previous_status,
     coalesce(src.attempt_count, 0) AS previous_attempt_count,
     src.last_error AS previous_error,
     sn.name_stage AS terminal_stage
DELETE pn, hsn
SET sn.source_paths = [
      path IN coalesce(sn.source_paths, [])
      WHERE NOT (path = 'dd:' + $dd_path OR path = $dd_path)
    ]
CREATE (retry:StandardNameSourceRetry {id: $retry_event_id})
SET retry.source_id = src.id,
    retry.previous_status = previous_status,
    retry.previous_attempt_count = previous_attempt_count,
    retry.previous_error = previous_error,
    retry.reason = $reason + ' [terminal target "' + sn.id +
                   '" at name_stage "' + terminal_stage + '"]',
    retry.retried_at = datetime()
MERGE (src)-[:HAS_RETRY_EVENT]->(retry)
SET src.retry_events = coalesce(src.retry_events, []) + retry.id,
    src.status = 'extracted',
    src.produced_sn_id = null,
    src.composed_at = null,
    src.attempt_count = 0,
    src.claimed_at = null,
    src.claim_token = null,
    src.failed_at = null,
    src.last_error = null
CREATE (change:StandardNameChange {id: $change_event_id})
SET change.from_name = sn.id,
    change.operation = 'recover_terminal_source_binding',
    change.reason = $reason + ' [terminal target "' + sn.id +
                    '" at name_stage "' + terminal_stage + '"]',
    change.origin = 'terminal_binding_recovery',
    change.changed_at = datetime(),
    change.internal = true
MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(change)
RETURN src.id AS source_node_id,
       sn.id AS sn_id,
       terminal_stage AS name_stage,
       retry.id AS retry_event_id,
       change.id AS change_event_id
"""


_TERMINAL_RECOVERY_MANIFEST_SCHEMA = "imas-codex.terminal-attachment-recovery-manifest"
_TERMINAL_RECOVERY_RECEIPT_SCHEMA = "imas-codex.terminal-attachment-recovery-receipt"
_TERMINAL_RECOVERY_OPERATION = "recover_terminal_attachment"
_TERMINAL_RECOVERY_MUTABLE_SOURCE_FIELDS = frozenset(
    {
        "attempt_count",
        "claimed_at",
        "claim_token",
        "composed_at",
        "failed_at",
        "last_error",
        "produced_sn_id",
        "retry_events",
        "status",
    }
)

_TERMINAL_RECOVERY_SOURCES_QUERY = """
// TERMINAL_ATTACHMENT_RECOVERY_SOURCES
UNWIND $pairs AS item
OPTIONAL MATCH (source:StandardNameSource {id: item.source_id})
RETURN item.source_id AS source_id,
  [candidate IN collect(DISTINCT source) WHERE candidate IS NOT NULL | {
    element_id: elementId(candidate), labels: labels(candidate),
    properties: properties(candidate),
    relationships: [(candidate)-[relationship]-(other) | {
      element_id: elementId(relationship), type: type(relationship),
      direction: CASE WHEN startNode(relationship) = candidate THEN 'out' ELSE 'in' END,
      properties: properties(relationship), other_element_id: elementId(other),
      other_labels: labels(other), other_id: other.id,
      other_properties: properties(other)
    }]
  }] AS participants
ORDER BY source_id
"""

_TERMINAL_RECOVERY_NODES_QUERY = """
// TERMINAL_ATTACHMENT_RECOVERY_NODES
UNWIND $pairs AS item
OPTIONAL MATCH (node:IMASNode {id: item.dd_path})
RETURN item.dd_path AS participant_id,
  [candidate IN collect(DISTINCT node) WHERE candidate IS NOT NULL | {
    element_id: elementId(candidate), labels: labels(candidate),
    properties: properties(candidate),
    relationships: [(candidate)-[relationship]-(other) | {
      element_id: elementId(relationship), type: type(relationship),
      direction: CASE WHEN startNode(relationship) = candidate THEN 'out' ELSE 'in' END,
      properties: properties(relationship), other_element_id: elementId(other),
      other_labels: labels(other), other_id: other.id,
      other_properties: properties(other)
    }]
  }] AS participants
ORDER BY participant_id
"""

_TERMINAL_RECOVERY_NAMES_QUERY = """
// TERMINAL_ATTACHMENT_RECOVERY_NAMES
UNWIND $sn_ids AS sn_id
OPTIONAL MATCH (name:StandardName {id: sn_id})
RETURN sn_id AS participant_id,
  [candidate IN collect(DISTINCT name) WHERE candidate IS NOT NULL | {
    element_id: elementId(candidate), labels: labels(candidate),
    properties: properties(candidate),
    relationships: [(candidate)-[relationship]-(other) | {
      element_id: elementId(relationship), type: type(relationship),
      direction: CASE WHEN startNode(relationship) = candidate THEN 'out' ELSE 'in' END,
      properties: properties(relationship), other_element_id: elementId(other),
      other_labels: labels(other), other_id: other.id,
      other_properties: properties(other)
    }]
  }] AS participants
ORDER BY participant_id
"""

_TERMINAL_RECOVERY_RELATIONSHIP_LOCK_QUERY = """
// TERMINAL_ATTACHMENT_RECOVERY_RELATIONSHIP_LOCK
UNWIND $relationships AS item
CALL (item) {
  MATCH (owner)
  WHERE elementId(owner) = item.owner_element_id
  CALL (item, owner) {
    WITH item, owner
    WHERE item.direction = 'out'
    MATCH (owner)-[relationship]->(other)
    WHERE elementId(relationship) = item.element_id
      AND type(relationship) = item.type
      AND elementId(other) = item.other_element_id
    RETURN relationship
    UNION ALL
    WITH item, owner
    WHERE item.direction = 'in'
    MATCH (owner)<-[relationship]-(other)
    WHERE elementId(relationship) = item.element_id
      AND type(relationship) = item.type
      AND elementId(other) = item.other_element_id
    RETURN relationship
  }
  RETURN relationship
}
SET relationship._terminal_attachment_recovery_lock = true
REMOVE relationship._terminal_attachment_recovery_lock
RETURN count(relationship) AS locked
"""

_TERMINAL_RECOVERY_BATCH_QUERY = """
// TERMINAL_ATTACHMENT_RECOVERY_APPLY
UNWIND $items AS item
MATCH (source:StandardNameSource {id: item.source_id})
MATCH (node:IMASNode {id: item.dd_path})
MATCH (name:StandardName {id: item.sn_id})
MATCH (source)-[binding:PRODUCED_NAME]->(name)
MATCH (node)-[projection:HAS_STANDARD_NAME]->(name)
WHERE elementId(source) = item.source_element_id
  AND elementId(node) = item.node_element_id
  AND elementId(name) = item.name_element_id
  AND elementId(binding) = item.binding_element_id
  AND elementId(projection) = item.projection_element_id
  AND source.source_type = 'dd'
  AND source.source_id = item.dd_path
  AND source.status = item.previous_status
  AND source.status IN $source_statuses
  AND source.claimed_at IS NULL
  AND source.claim_token IS NULL
  AND source.produced_sn_id = name.id
  AND name.name_stage = item.terminal_stage
  AND name.name_stage IN $terminal_stages
WITH item, source, node, name, binding, projection,
     source.status AS previous_status,
     coalesce(source.attempt_count, 0) AS previous_attempt_count,
     source.last_error AS previous_error
DELETE binding, projection
SET name.source_paths = [
      path IN coalesce(name.source_paths, [])
      WHERE NOT (path = 'dd:' + item.dd_path OR path = item.dd_path)
    ]
CREATE (retry:StandardNameSourceRetry)
SET retry = item.retry_event,
    retry.retried_at = datetime(item.retry_event.retried_at),
    retry.previous_status = previous_status,
    retry.previous_attempt_count = previous_attempt_count,
    retry.previous_error = previous_error
CREATE (source)-[:HAS_RETRY_EVENT]->(retry)
SET source.retry_events = coalesce(source.retry_events, []) + retry.id,
    source.status = 'extracted',
    source.produced_sn_id = null,
    source.composed_at = null,
    source.attempt_count = 0,
    source.claimed_at = null,
    source.claim_token = null,
    source.failed_at = null,
    source.last_error = null
CREATE (change:StandardNameChange)
SET change = item.change_event,
    change.changed_at = datetime(item.change_event.changed_at)
CREATE (name)-[:HAS_INTERNAL_CHANGE]->(change)
RETURN count(*) AS applied,
       collect(source.id) AS source_ids,
       collect(retry.id) AS retry_event_ids,
       collect(change.id) AS change_event_ids
"""


class TerminalAttachmentRecoveryConflict(RuntimeError):
    """The exact manifest-bound terminal attachment closure changed."""


@dataclass(frozen=True)
class TerminalAttachmentRecoveryManifest:
    """One exact terminal-binding recovery cohort."""

    path: Path
    manifest_hash: str
    rows: tuple[dict[str, Any], ...]
    source_ids: tuple[str, ...]
    pairs: tuple[dict[str, str], ...]
    allowlist_hash: str


@dataclass(frozen=True)
class AttachmentVerdict:
    """One attachment the guard rejects."""

    source_node_id: str
    dd_path: str
    sn_id: str
    name_stage: str | None
    reason: str
    other_live_names: int = 0

    @property
    def rule(self) -> str:
        """The guard rule that rejected it, taken from the reason's prefix."""
        return self.reason.split(":", 1)[0] if ":" in self.reason else "unclassified"

    @property
    def protected(self) -> bool:
        """Catalog-authoritative — detaching needs ``include_accepted``."""
        return (self.name_stage or "") in _PROTECTED_NAME_STAGES

    @property
    def historical(self) -> bool:
        """A deliberate provenance edge on a superseded name — never detached."""
        return (self.name_stage or "") in _HISTORICAL_NAME_STAGES

    @property
    def reroute(self) -> bool:
        """Whether detaching leaves the source with no live name to realize."""
        return self.other_live_names == 0


@dataclass(frozen=True)
class AttachmentPairingGuardResult:
    """Fresh source pairings admitted and rejected by the write-time guard."""

    accepted_source_ids: tuple[str, ...]
    rejected: tuple[AttachmentVerdict, ...]


def guard_source_pairings(
    gc: Any, sn_id: str, source_ids: list[str]
) -> AttachmentPairingGuardResult:
    """Preflight fresh source-to-name pairings on the caller's query handle.

    Existing bindings are preserved: corpus reconciliation owns historical
    defects, while this boundary prevents a writer from introducing a new one.
    Signal and derived sources have no DD path to validate and pass unchanged.
    DD candidates are evaluated in deterministic path order with the same
    compose semantics as :func:`audit_attachments`.
    """
    from imas_codex.standard_names.workers import _is_attachment_consistent

    requested = sorted(set(source_ids))
    if not sn_id or not requested:
        return AttachmentPairingGuardResult((), ())

    rows = list(
        gc.query(
            _PAIRING_GUARD_QUERY,
            sn_id=sn_id,
            source_ids=requested,
        )
    )
    by_id = {row["source_id"]: row for row in rows}
    existing_paths = sorted(
        {path for row in rows for path in (row.get("existing_dd_paths") or []) if path}
    )
    accepted: list[str] = []
    rejected: list[AttachmentVerdict] = []
    for source_id in requested:
        row = by_id.get(source_id)
        if row is None or row.get("source_type") is None:
            rejected.append(
                AttachmentVerdict(
                    source_node_id=source_id,
                    dd_path="",
                    sn_id=sn_id,
                    name_stage=None,
                    reason="missing semantic source: source node does not exist",
                )
            )
            continue
        if row.get("already_bound") or not row.get("dd_path"):
            accepted.append(source_id)
            continue
        dd_path = row["dd_path"]
        ok, reason = _is_attachment_consistent(
            dd_path,
            sn_id,
            existing_sources=tuple(existing_paths),
            dd_unit=row.get("dd_unit"),
            sn_unit=row.get("sn_unit"),
        )
        if ok:
            accepted.append(source_id)
            existing_paths.append(dd_path)
        else:
            rejected.append(
                AttachmentVerdict(
                    source_node_id=source_id,
                    dd_path=dd_path,
                    sn_id=sn_id,
                    name_stage=row.get("name_stage"),
                    reason=reason,
                )
            )
    return AttachmentPairingGuardResult(tuple(accepted), tuple(rejected))


@dataclass(frozen=True)
class NameLevelDefect:
    """A name whose every attachment is rejected by one rule — rename it.

    Carries what the operator needs to act without re-querying: the name, the
    rule its whole source set trips, how many sources say so, and one example
    path to read the physics off. The repair is
    ``imas-codex sn edit <sn_id> --rename … --reason …``.
    """

    sn_id: str
    rule: str
    attachment_count: int
    example_dd_path: str
    name_stage: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sn_id": self.sn_id,
            "rule": self.rule,
            "attachments": self.attachment_count,
            "example_dd_path": self.example_dd_path,
            "name_stage": self.name_stage,
        }


@dataclass
class AttachmentAuditResult:
    """Outcome of one audit / reconcile pass."""

    checked: int = 0
    rejected: list[AttachmentVerdict] = field(default_factory=list)
    detached: int = 0
    sources_rerouted: int = 0
    skipped_protected: int = 0
    skipped_historical: int = 0
    skipped_misnamed: int = 0
    names_misnamed: list[NameLevelDefect] = field(default_factory=list)
    names_orphaned: list[str] = field(default_factory=list)
    dd_gap_evidence: list[dict[str, Any]] = field(default_factory=list)

    def by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.rejected:
            counts[v.rule] = counts.get(v.rule, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "rejected": len(self.rejected),
            "detached": self.detached,
            "sources_rerouted": self.sources_rerouted,
            "skipped_protected": self.skipped_protected,
            "skipped_historical": self.skipped_historical,
            "skipped_misnamed": self.skipped_misnamed,
            "names_misnamed": len(self.names_misnamed),
            # The rename worklist itself, not just its size — it is the whole
            # point of the classification and the operator acts on it directly.
            "misnamed": [d.as_dict() for d in self.names_misnamed],
            "names_orphaned": len(self.names_orphaned),
            "dd_gaps": len(self.dd_gap_evidence),
            "by_rule": self.by_rule(),
        }


def _attachment_dd_gap_evidence(rows: list[dict]) -> list[dict[str, Any]]:
    """Derive DD evidence only from fetched scalar-versus-edge contradictions."""
    from imas_codex.standard_names.sources.dd import (
        _unit_declaration_conflict_reports,
    )

    reports_by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        relationship_units = sorted(
            {
                str(unit).strip()
                for unit in row.get("dd_relationship_units") or []
                if str(unit).strip()
            }
        )
        if len(relationship_units) > 1:
            path = str(row.get("dd_path") or "").strip()
            if path:
                reports_by_path[path] = {
                    "path": path,
                    "kind": "self_contradiction",
                    "reason": (
                        "The DD node has multiple authoritative HAS_UNIT "
                        "relationships, so no unique unit declaration exists."
                    ),
                    "observed_dd_version": row.get("dd_version"),
                    "observed_value": str(row.get("dd_declared_unit") or ""),
                    "expected_value": ",".join(relationship_units),
                    "evidence_rule": "unit_relationship_is_unique",
                    "reporter": "attachment-audit",
                }
            continue
        unit_row = {
            "path": row.get("dd_path"),
            "unit": row.get("dd_declared_unit"),
            "unit_from_rel": relationship_units[0] if relationship_units else None,
        }
        for report in _unit_declaration_conflict_reports(
            [unit_row], row.get("dd_version")
        ):
            report["reporter"] = "attachment-audit"
            reports_by_path.setdefault(report["path"], report)
    return list(reports_by_path.values())


def audit_attachments(
    gc: Any | None = None, *, sn_id: str | None = None
) -> AttachmentAuditResult:
    """Re-validate graph attachments against the full guard suite. Read-only.

    Returns the verdicts without touching the graph, so the same predicate backs
    both the reporting path and the reconcile.

    Args:
        gc: An open ``GraphClient``, or None to open and close one.
        sn_id: Restrict the read to one name's attachments. The whole-name
            wipeout classification then sees exactly that name's source set,
            which is what it needs — it is a per-name rule. Used by the
            write-time gate; omit for the corpus-wide pass.
    """
    from imas_codex.standard_names.workers import _is_attachment_consistent

    own = gc is None
    if own:
        from imas_codex.graph.client import GraphClient

        client: Any = GraphClient()
    else:
        client = gc
    try:
        params: dict[str, Any] = {"historical": sorted(_HISTORICAL_NAME_STAGES)}
        if sn_id is not None:
            params["sn_id"] = sn_id
        rows = list(
            client.query(
                _ATTACHMENTS_QUERY.format(
                    scope="" if sn_id is None else _ONE_NAME_SCOPE
                ),
                **params,
            )
        )
    finally:
        if own:
            client.close()

    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["sn_id"], []).append(r)

    result = AttachmentAuditResult(
        checked=len(rows),
        dd_gap_evidence=_attachment_dd_gap_evidence(rows),
    )
    for sn_id in sorted(by_name):
        # Sorted by path so the pairwise rule keeps the SAME representative on
        # every pass: the read returns rows in Neo4j's order, and an
        # order-dependent rule applied to an unstable order would detach a
        # different member of a conflicting group each time.
        group = sorted(
            by_name[sn_id],
            key=lambda r: (r["dd_path"] or "", r["source_node_id"] or ""),
        )
        accepted_paths: list[str] = []
        group_rejected: list[AttachmentVerdict] = []
        for r in group:
            ok, reason = _is_attachment_consistent(
                r["dd_path"],
                sn_id,
                existing_sources=tuple(accepted_paths),
                dd_unit=r["dd_unit"],
                sn_unit=r["sn_unit"],
            )
            if ok:
                accepted_paths.append(r["dd_path"])
                continue
            group_rejected.append(
                AttachmentVerdict(
                    source_node_id=r["source_node_id"],
                    dd_path=r["dd_path"],
                    sn_id=sn_id,
                    name_stage=r["name_stage"],
                    reason=reason,
                    other_live_names=int(r.get("other_live_names") or 0),
                )
            )
        result.rejected.extend(group_rejected)
        defect = _name_level_defect(sn_id, group, group_rejected)
        if defect is not None:
            result.names_misnamed.append(defect)
    return result


def _name_level_defect(
    sn_id: str, group: list[dict], rejected: list[AttachmentVerdict]
) -> NameLevelDefect | None:
    """Classify a name whose whole source set trips one rule. See the docstring.

    None unless the name has corroborating sources (``_MIN_WIPEOUT_ATTACHMENTS``),
    every one of them is rejected, and they all trip the SAME rule. A superseded
    name is excluded: its edges are the deliberate provenance record, so there is
    no live claim to rename.
    """
    if len(group) < _MIN_WIPEOUT_ATTACHMENTS or len(rejected) != len(group):
        return None
    if (group[0].get("name_stage") or "") in _HISTORICAL_NAME_STAGES:
        return None
    rules = {v.rule for v in rejected}
    if len(rules) != 1:
        return None
    return NameLevelDefect(
        sn_id=sn_id,
        rule=next(iter(rules)),
        attachment_count=len(group),
        example_dd_path=rejected[0].dd_path,
        name_stage=group[0].get("name_stage"),
    )


def reconcile_attachment_consistency(
    gc: Any | None = None,
    *,
    include_accepted: bool = False,
    dry_run: bool = False,
    sn_id: str | None = None,
) -> AttachmentAuditResult:
    """Detach every inconsistent attachment and reroute its source to compose.

    Args:
        gc: An open ``GraphClient``, or None to open and close one.
        include_accepted: Also detach attachments on ``name_stage='accepted'``
            names. Off by default: an accepted name is catalog-authoritative, so
            losing a source is a decision, not hygiene.
        dry_run: Report the verdicts without writing.
        sn_id: Restrict the pass to one name (see :func:`audit_attachments`).

    Returns the :class:`AttachmentAuditResult`, whose ``by_rule()`` breaks the
    rejections down by which guard rule fired.
    """
    own = gc is None
    if own:
        from imas_codex.graph.client import GraphClient

        client: Any = GraphClient()
    else:
        client = gc
    try:
        result = audit_attachments(client, sn_id=sn_id)
        if not dry_run and result.dd_gap_evidence:
            try:
                from imas_codex.standard_names.dd_gaps import write_dd_gaps

                write_dd_gaps(result.dd_gap_evidence)
            except Exception:
                logger.warning(
                    "Attachment DD-gap evidence persistence failed without "
                    "changing reconciliation",
                    exc_info=True,
                )
        misnamed = {d.sn_id for d in result.names_misnamed}
        # Disjoint buckets, most-binding first: a provenance edge is never
        # touched; a name-level defect is repaired by renaming the name, not by
        # stripping its sources; only then does the accepted-name gate apply.
        actionable: list[AttachmentVerdict] = []
        for v in result.rejected:
            if v.historical:
                result.skipped_historical += 1
            elif v.sn_id in misnamed:
                result.skipped_misnamed += 1
            elif v.protected and not include_accepted:
                result.skipped_protected += 1
            else:
                actionable.append(v)

        if result.rejected:
            logger.warning(
                "reconcile_attachment_consistency: %d of %d attachment(s) fail "
                "the consistency guard (by rule: %s)",
                len(result.rejected),
                result.checked,
                result.by_rule(),
            )
        if result.skipped_protected:
            # Visible, not silent: these are wrong attachments the operator has
            # chosen not to break yet, and each one is on a published name.
            logger.warning(
                "reconcile_attachment_consistency: %d inconsistent attachment(s) "
                "left in place on accepted names (pass include_accepted to detach). "
                "First few: %s",
                result.skipped_protected,
                "; ".join(
                    f"{v.dd_path} → {v.sn_id} ({v.reason})"
                    for v in result.rejected
                    if v.protected and v.sn_id not in misnamed
                )[:1200],
            )
        if result.names_misnamed:
            # Loud: this is a worklist the operator must act on, and the audit
            # deliberately declines to "fix" it by destroying the evidence.
            logger.warning(
                "reconcile_attachment_consistency: %d name(s) are rejected by "
                "EVERY one of their sources under a single rule — the NAME is "
                "wrong, not the attachments (%d attachment(s) left in place). "
                "Repair with: imas-codex sn edit <name> --rename <new> --reason "
                "<why>. %s",
                len(result.names_misnamed),
                result.skipped_misnamed,
                "; ".join(
                    f"{d.sn_id} [{d.rule}, {d.attachment_count} src, "
                    f"e.g. {d.example_dd_path}]"
                    for d in result.names_misnamed
                )[:2000],
            )
        if result.skipped_historical:
            logger.info(
                "reconcile_attachment_consistency: %d inconsistent attachment(s) on "
                "superseded names left as the provenance record",
                result.skipped_historical,
            )
        if dry_run or not actionable:
            return result

        items = [
            {
                "source_node_id": v.source_node_id,
                "dd_path": v.dd_path,
                "sn_id": v.sn_id,
                "reroute": v.reroute,
            }
            for v in actionable
        ]
        rows = client.query(
            _DETACH_QUERY, items=items, historical=sorted(_HISTORICAL_NAME_STAGES)
        )
        result.detached = int(rows[0]["detached"]) if rows else 0
        result.sources_rerouted = len(
            {v.source_node_id for v in actionable if v.reroute}
        )

        _record_detachments(client, actionable)
        result.names_orphaned = _find_orphaned_names(
            client, sorted({v.sn_id for v in actionable})
        )
    finally:
        if own:
            client.close()

    logger.info(
        "reconcile_attachment_consistency: detached %d inconsistent attachment(s), "
        "returned %d source(s) to 'extracted' for re-composition",
        result.detached,
        result.sources_rerouted,
    )
    if result.names_orphaned:
        # A name whose only justification was a wrong attachment. Surfaced, not
        # auto-superseded: whether the concept survives is a human decision.
        logger.warning(
            "reconcile_attachment_consistency: %d name(s) now carry no source at "
            "all — every attachment they had was inconsistent: %s",
            len(result.names_orphaned),
            ", ".join(result.names_orphaned[:20]),
        )
    return result


def _require_terminal_recovery_sha(value: Any, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be exactly one SHA-256 hex digest")
    return normalized


def load_terminal_attachment_recovery_manifest(
    path: str | Path,
) -> TerminalAttachmentRecoveryManifest:
    """Load one exact homogeneous terminal-binding recovery manifest."""
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"terminal attachment recovery manifest is not valid JSON: {manifest_path}"
        ) from exc
    expected_top_level = {"schema", "schema_version", "operation", "rows"}
    if not isinstance(payload, dict) or set(payload) != expected_top_level:
        raise ValueError(
            "terminal attachment recovery manifest must contain only schema, "
            "schema_version, operation, and rows"
        )
    if (
        payload.get("schema") != _TERMINAL_RECOVERY_MANIFEST_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("operation") != _TERMINAL_RECOVERY_OPERATION
    ):
        raise ValueError("terminal attachment recovery manifest schema is unsupported")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("terminal attachment recovery manifest requires rows")
    expected_fields = {
        "operation",
        "source_id",
        "dd_path",
        "sn_id",
        "expected_source_status",
        "expected_name_stage",
        "expected_closure_hash",
        "expected_preserved_state_hash",
        "expected_participant_ids_hash",
        "expected_relationship_ids_hash",
        "west_intersection",
        "test_intersection",
    }
    normalized_rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_paths: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            missing = sorted(
                expected_fields - set(row) if isinstance(row, dict) else []
            )
            extra = sorted(set(row) - expected_fields if isinstance(row, dict) else [])
            raise ValueError(
                "terminal recovery row fields are not exact; "
                f"missing={missing}, extra={extra}"
            )
        if row["operation"] != _TERMINAL_RECOVERY_OPERATION:
            raise ValueError("terminal recovery manifest must be homogeneous")
        source_id = str(row["source_id"])
        dd_path = str(row["dd_path"])
        sn_id = str(row["sn_id"])
        if source_id != f"dd:{dd_path}" or not dd_path or not sn_id:
            raise ValueError("each row requires one exact dd:{path} source binding")
        pair = (dd_path, sn_id)
        if source_id in seen_sources or dd_path in seen_paths or pair in seen_pairs:
            raise ValueError(
                "terminal recovery manifest contains duplicate or overlapping rows"
            )
        seen_sources.add(source_id)
        seen_paths.add(dd_path)
        seen_pairs.add(pair)
        if row["expected_source_status"] not in _TERMINAL_RECOVERY_SOURCE_STATUSES:
            raise ValueError("terminal recovery source status is not eligible")
        if row["expected_name_stage"] not in _TERMINAL_RECOVERY_STAGES:
            raise ValueError("terminal recovery target stage is not terminal")
        for field_name in (
            "expected_closure_hash",
            "expected_preserved_state_hash",
            "expected_participant_ids_hash",
            "expected_relationship_ids_hash",
        ):
            _require_terminal_recovery_sha(row[field_name], field_name)
        if row["test_intersection"] != 0:
            raise ValueError("test intersection must be exactly zero")
        normalized_rows.append(copy.deepcopy(row))
    normalized_rows.sort(key=lambda item: item["source_id"])
    source_ids = tuple(str(row["source_id"]) for row in normalized_rows)
    pairs = tuple(
        {
            "source_id": str(row["source_id"]),
            "dd_path": str(row["dd_path"]),
            "sn_id": str(row["sn_id"]),
        }
        for row in normalized_rows
    )
    return TerminalAttachmentRecoveryManifest(
        path=manifest_path,
        manifest_hash=hashlib.sha256(raw).hexdigest(),
        rows=tuple(normalized_rows),
        source_ids=source_ids,
        pairs=pairs,
        allowlist_hash=payload_hash(source_ids),
    )


def _terminal_recovery_participant_ids(row: dict[str, Any]) -> tuple[str, ...]:
    ids: set[str] = set()
    for key in ("sources", "nodes", "names"):
        for participant in row.get(key) or []:
            if participant.get("element_id"):
                ids.add(str(participant["element_id"]))
            ids.update(
                str(relationship["other_element_id"])
                for relationship in participant.get("relationships") or []
                if relationship.get("other_element_id")
            )
    return tuple(sorted(ids))


def _terminal_recovery_relationship_ids(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        descriptor["element_id"]
        for descriptor in _terminal_recovery_relationship_descriptors(row)
    )


def _terminal_recovery_relationship_descriptors(
    row: dict[str, Any],
) -> tuple[dict[str, str], ...]:
    """Return one participant-anchored descriptor per exact relationship."""
    candidates: dict[str, list[dict[str, str]]] = {}
    for key in ("sources", "nodes", "names"):
        for participant in row.get(key) or []:
            owner_element_id = participant.get("element_id")
            if not owner_element_id:
                continue
            for relationship in participant.get("relationships") or []:
                element_id = relationship.get("element_id")
                other_element_id = relationship.get("other_element_id")
                relationship_type = relationship.get("type")
                direction = relationship.get("direction")
                if not all(
                    (element_id, other_element_id, relationship_type, direction)
                ):
                    continue
                candidates.setdefault(str(element_id), []).append(
                    {
                        "element_id": str(element_id),
                        "owner_element_id": str(owner_element_id),
                        "other_element_id": str(other_element_id),
                        "type": str(relationship_type),
                        "direction": str(direction),
                    }
                )
    return tuple(
        min(
            descriptors,
            key=lambda item: (
                item["owner_element_id"],
                item["other_element_id"],
                item["direction"],
            ),
        )
        for _, descriptors in sorted(candidates.items())
    )


def _terminal_recovery_protected_reasons(row: dict[str, Any]) -> list[str]:
    """Refuse recovery into immutable closures.

    Persistent test fixtures are immutable.  Facility batch membership is
    ordinary repairable state and yields no reason.
    """
    fixture = False

    def visit(value: Any, key: str = "") -> None:
        nonlocal fixture
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
            return
        if isinstance(value, list | tuple):
            for child in value:
                visit(child, key)
            return
        if not isinstance(value, str):
            return
        normalized = value.casefold()
        normalized_key = key.casefold()
        if normalized_key in {"id", "other_id", "source_id"}:
            fixture = fixture or normalized.startswith(
                ("fixture:", "test:", "signals:test:")
            )
        if normalized_key in {"origin", "source_type"} and normalized in {
            "fixture",
            "test",
        }:
            fixture = True

    visit(row)
    return ["current graph closure intersects test fixtures"] if fixture else []


def _terminal_recovery_preserved_payload(
    row: dict[str, Any],
    *,
    retry_event_id: str,
    change_event_id: str,
    cohort_context: dict[str, dict[str, set[str]]] | None = None,
) -> dict[str, Any]:
    source_id = str(row["source_id"])
    dd_path = str(row["dd_path"])
    sn_id = str(row["sn_id"])
    target_context = (cohort_context or {}).get(sn_id) or {
        "source_ids": {source_id},
        "dd_paths": {dd_path},
        "retry_event_ids": {retry_event_id},
        "change_event_ids": {change_event_id},
    }

    def normalized_relationships(
        participant: dict[str, Any], *, owner: str
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for relationship in copy.deepcopy(participant.get("relationships") or []):
            relationship_type = relationship.get("type")
            other_id = relationship.get("other_id")
            direction = relationship.get("direction")
            if owner == "source" and (
                relationship_type == "PRODUCED_NAME"
                and direction == "out"
                and other_id == sn_id
                or relationship_type == "HAS_RETRY_EVENT"
                and direction == "out"
                and other_id == retry_event_id
            ):
                continue
            if owner == "node" and (
                relationship_type == "HAS_STANDARD_NAME"
                and direction == "out"
                and other_id == sn_id
            ):
                continue
            if owner == "name" and (
                relationship_type == "PRODUCED_NAME"
                and direction == "in"
                and other_id in target_context["source_ids"]
                or relationship_type == "HAS_STANDARD_NAME"
                and direction == "in"
                and other_id in target_context["dd_paths"]
                or relationship_type == "HAS_INTERNAL_CHANGE"
                and direction == "out"
                and other_id in target_context["change_event_ids"]
            ):
                continue
            if other_id == source_id:
                relationship["other_properties"] = {
                    key: value
                    for key, value in (
                        relationship.get("other_properties") or {}
                    ).items()
                    if key not in _TERMINAL_RECOVERY_MUTABLE_SOURCE_FIELDS
                }
            normalized.append(relationship)
        return normalized

    source = copy.deepcopy((row.get("sources") or [{}])[0])
    node = copy.deepcopy((row.get("nodes") or [{}])[0])
    name = copy.deepcopy((row.get("names") or [{}])[0])
    source_properties = {
        key: value
        for key, value in (source.get("properties") or {}).items()
        if key not in _TERMINAL_RECOVERY_MUTABLE_SOURCE_FIELDS
    }
    name_properties = copy.deepcopy(name.get("properties") or {})
    name_properties["source_paths"] = [
        path
        for path in name_properties.get("source_paths") or []
        if path not in target_context["dd_paths"]
        and path not in target_context["source_ids"]
    ]
    return {
        "source": {
            "element_id": source.get("element_id"),
            "labels": source.get("labels") or [],
            "properties": source_properties,
            "relationships": normalized_relationships(source, owner="source"),
        },
        "node": {
            "element_id": node.get("element_id"),
            "labels": node.get("labels") or [],
            "properties": node.get("properties") or {},
            "relationships": normalized_relationships(node, owner="node"),
        },
        "name": {
            "element_id": name.get("element_id"),
            "labels": name.get("labels") or [],
            "properties": name_properties,
            "relationships": normalized_relationships(name, owner="name"),
        },
    }


def _terminal_recovery_event_ids(
    manifest: TerminalAttachmentRecoveryManifest, manifest_row: dict[str, Any]
) -> tuple[str, str]:
    identity_hash = payload_hash(
        {
            "manifest_hash": manifest.manifest_hash,
            "source_id": manifest_row["source_id"],
            "dd_path": manifest_row["dd_path"],
            "sn_id": manifest_row["sn_id"],
            "before_closure_hash": manifest_row["expected_closure_hash"],
        }
    )
    return (
        f"source-retry:terminal-attachment:{identity_hash}",
        f"sn-change:terminal-attachment:{identity_hash}",
    )


def _terminal_recovery_cohort_context(
    manifest: TerminalAttachmentRecoveryManifest,
) -> dict[str, dict[str, set[str]]]:
    """Group every authorized mutation by its shared terminal target."""
    context: dict[str, dict[str, set[str]]] = {}
    for manifest_row in manifest.rows:
        retry_event_id, change_event_id = _terminal_recovery_event_ids(
            manifest, manifest_row
        )
        target = context.setdefault(
            str(manifest_row["sn_id"]),
            {
                "source_ids": set(),
                "dd_paths": set(),
                "retry_event_ids": set(),
                "change_event_ids": set(),
            },
        )
        target["source_ids"].add(str(manifest_row["source_id"]))
        target["dd_paths"].add(str(manifest_row["dd_path"]))
        target["retry_event_ids"].add(retry_event_id)
        target["change_event_ids"].add(change_event_id)
    return context


def _terminal_recovery_snapshot_hashes(
    row: dict[str, Any],
    *,
    retry_event_id: str,
    change_event_id: str,
    cohort_context: dict[str, dict[str, set[str]]] | None = None,
) -> dict[str, str]:
    participant_ids = _terminal_recovery_participant_ids(row)
    relationship_ids = _terminal_recovery_relationship_ids(row)
    return {
        "closure_hash": payload_hash(row),
        "preserved_state_hash": payload_hash(
            _terminal_recovery_preserved_payload(
                row,
                retry_event_id=retry_event_id,
                change_event_id=change_event_id,
                cohort_context=cohort_context,
            )
        ),
        "participant_ids_hash": payload_hash(participant_ids),
        "relationship_ids_hash": payload_hash(relationship_ids),
    }


def _terminal_recovery_relationships(
    participant: dict[str, Any],
    relationship_type: str,
    *,
    direction: str,
    other_id: str,
) -> list[dict[str, Any]]:
    return [
        relationship
        for relationship in participant.get("relationships") or []
        if relationship.get("type") == relationship_type
        and relationship.get("direction") == direction
        and relationship.get("other_id") == other_id
    ]


def _terminal_recovery_is_already_current(
    row: dict[str, Any],
    manifest_row: dict[str, Any],
    *,
    retry_event_id: str,
    change_event_id: str,
) -> bool:
    if not (
        len(row.get("sources") or []) == 1
        and len(row.get("nodes") or []) == 1
        and len(row.get("names") or []) == 1
    ):
        return False
    source = row["sources"][0]
    node = row["nodes"][0]
    name = row["names"][0]
    source_properties = source.get("properties") or {}
    name_properties = name.get("properties") or {}
    retry_links = _terminal_recovery_relationships(
        source,
        "HAS_RETRY_EVENT",
        direction="out",
        other_id=retry_event_id,
    )
    change_links = _terminal_recovery_relationships(
        name,
        "HAS_INTERNAL_CHANGE",
        direction="out",
        other_id=change_event_id,
    )
    retry = (
        (retry_links[0].get("other_properties") or {}) if len(retry_links) == 1 else {}
    )
    change = (
        (change_links[0].get("other_properties") or {})
        if len(change_links) == 1
        else {}
    )
    return all(
        (
            source_properties.get("status") == "extracted",
            source_properties.get("produced_sn_id") is None,
            source_properties.get("attempt_count") in {None, 0},
            source_properties.get("claimed_at") is None,
            source_properties.get("claim_token") is None,
            not _terminal_recovery_relationships(
                source,
                "PRODUCED_NAME",
                direction="out",
                other_id=str(manifest_row["sn_id"]),
            ),
            not _terminal_recovery_relationships(
                node,
                "HAS_STANDARD_NAME",
                direction="out",
                other_id=str(manifest_row["sn_id"]),
            ),
            str(manifest_row["dd_path"])
            not in (name_properties.get("source_paths") or []),
            str(manifest_row["source_id"])
            not in (name_properties.get("source_paths") or []),
            retry.get("before_closure_hash") == manifest_row["expected_closure_hash"],
            retry.get("preserved_state_hash")
            == manifest_row["expected_preserved_state_hash"],
            retry.get("previous_status") == manifest_row["expected_source_status"],
            retry.get("terminal_sn_id") == manifest_row["sn_id"],
            change.get("before_closure_hash") == manifest_row["expected_closure_hash"],
            change.get("preserved_state_hash")
            == manifest_row["expected_preserved_state_hash"],
            change.get("operation") == "recover_terminal_source_binding",
            change.get("source_id") == manifest_row["source_id"],
        )
    )


def _terminal_recovery_plan_row(
    row: dict[str, Any],
    manifest: TerminalAttachmentRecoveryManifest,
    manifest_row: dict[str, Any],
    *,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
    cohort_context: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, Any] | None, list[str]]:
    retry_event_id, change_event_id = _terminal_recovery_event_ids(
        manifest, manifest_row
    )
    reasons = _terminal_recovery_protected_reasons(row)
    if not (
        len(row.get("sources") or []) == 1
        and len(row.get("nodes") or []) == 1
        and len(row.get("names") or []) == 1
    ):
        reasons.append("source, DD path, or terminal target is not unique")
        return None, reasons
    source = row["sources"][0]
    node = row["nodes"][0]
    name = row["names"][0]
    source_properties = source.get("properties") or {}
    node_properties = node.get("properties") or {}
    name_properties = name.get("properties") or {}
    current_hashes = _terminal_recovery_snapshot_hashes(
        row,
        retry_event_id=retry_event_id,
        change_event_id=change_event_id,
        cohort_context=cohort_context,
    )
    if _terminal_recovery_is_already_current(
        row,
        manifest_row,
        retry_event_id=retry_event_id,
        change_event_id=change_event_id,
    ):
        if (
            current_hashes["preserved_state_hash"]
            != manifest_row["expected_preserved_state_hash"]
        ):
            reasons.append("preserved terminal graph state drifted after recovery")
        return (
            {
                "source_id": manifest_row["source_id"],
                "dd_path": manifest_row["dd_path"],
                "sn_id": manifest_row["sn_id"],
                "status": "already_current",
                "precondition_hash": current_hashes["closure_hash"],
                "preserved_state_hash": current_hashes["preserved_state_hash"],
                "participant_ids": list(_terminal_recovery_participant_ids(row)),
                "relationship_ids": list(_terminal_recovery_relationship_ids(row)),
                "relationship_descriptors": list(
                    _terminal_recovery_relationship_descriptors(row)
                ),
                "retry_event_id": retry_event_id,
                "change_event_id": change_event_id,
            },
            reasons,
        )

    from_dd = _terminal_recovery_relationships(
        source,
        "FROM_DD_PATH",
        direction="out",
        other_id=str(manifest_row["dd_path"]),
    )
    bindings = _terminal_recovery_relationships(
        source,
        "PRODUCED_NAME",
        direction="out",
        other_id=str(manifest_row["sn_id"]),
    )
    projections = _terminal_recovery_relationships(
        node,
        "HAS_STANDARD_NAME",
        direction="out",
        other_id=str(manifest_row["sn_id"]),
    )
    all_outputs = [
        relationship
        for relationship in source.get("relationships") or []
        if relationship.get("type") == "PRODUCED_NAME"
        and relationship.get("direction") == "out"
    ]
    target_sources = [
        relationship
        for relationship in name.get("relationships") or []
        if relationship.get("type") == "PRODUCED_NAME"
        and relationship.get("direction") == "in"
    ]
    target_projections = [
        relationship
        for relationship in name.get("relationships") or []
        if relationship.get("type") == "HAS_STANDARD_NAME"
        and relationship.get("direction") == "in"
    ]
    source_paths = name_properties.get("source_paths") or []
    mirror_exact = (
        manifest_row["dd_path"] in source_paths
        or manifest_row["source_id"] in source_paths
    )
    isolated_empty_mirror = (
        not source_paths and len(target_sources) == 1 and len(target_projections) == 1
    )
    expected_values = {
        "source stable identity": (
            source_properties.get("id"),
            manifest_row["source_id"],
        ),
        "source type": (source_properties.get("source_type"), "dd"),
        "source DD scalar": (
            source_properties.get("source_id"),
            manifest_row["dd_path"],
        ),
        "source status": (
            source_properties.get("status"),
            manifest_row["expected_source_status"],
        ),
        "source target scalar": (
            source_properties.get("produced_sn_id"),
            manifest_row["sn_id"],
        ),
        "DD identity": (node_properties.get("id"), manifest_row["dd_path"]),
        "target identity": (name_properties.get("id"), manifest_row["sn_id"]),
        "terminal target stage": (
            name_properties.get("name_stage"),
            manifest_row["expected_name_stage"],
        ),
    }
    reasons.extend(
        f"{label} changed"
        for label, (actual, expected) in expected_values.items()
        if actual != expected
    )
    claim_fields = (
        "claimed_at",
        "claim_token",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
    )
    if any(
        source_properties.get(field_name) is not None for field_name in claim_fields
    ):
        reasons.append("source has an active claim")
    if any(name_properties.get(field_name) is not None for field_name in claim_fields):
        reasons.append("terminal target has an active claim")
    if len(from_dd) != 1:
        reasons.append("source does not have one exact FROM_DD_PATH edge")
    if len(bindings) != 1 or len(all_outputs) != 1:
        reasons.append("source does not have one exact terminal binding")
    if len(projections) != 1:
        reasons.append("DD path does not have one exact terminal projection")
    if not (mirror_exact or isolated_empty_mirror):
        reasons.append("target source-path mirror is not exact or isolated-empty")
    for field_name, current in current_hashes.items():
        expected_field = f"expected_{field_name}"
        if manifest_row[expected_field] != current:
            reasons.append(f"manifest {expected_field} drifted")
    event_reason = (
        f'{reason} [terminal target "{manifest_row["sn_id"]}" at name_stage '
        f'"{manifest_row["expected_name_stage"]}"]'
    )
    retry_event = {
        "id": retry_event_id,
        "source_id": manifest_row["source_id"],
        "terminal_sn_id": manifest_row["sn_id"],
        "terminal_stage": manifest_row["expected_name_stage"],
        "before_closure_hash": manifest_row["expected_closure_hash"],
        "preserved_state_hash": manifest_row["expected_preserved_state_hash"],
        "manifest_hash": manifest.manifest_hash,
        "run_id": run_id,
        "reason": event_reason,
        "retried_at": changed_at,
    }
    change_event = {
        "id": change_event_id,
        "from_name": manifest_row["sn_id"],
        "source_id": manifest_row["source_id"],
        "dd_path": manifest_row["dd_path"],
        "operation": "recover_terminal_source_binding",
        "origin": "terminal_binding_recovery",
        "internal": True,
        "before_closure_hash": manifest_row["expected_closure_hash"],
        "preserved_state_hash": manifest_row["expected_preserved_state_hash"],
        "manifest_hash": manifest.manifest_hash,
        "run_id": run_id,
        "reason": event_reason,
        "changed_at": changed_at,
    }
    plan = {
        "source_id": manifest_row["source_id"],
        "dd_path": manifest_row["dd_path"],
        "sn_id": manifest_row["sn_id"],
        "status": "planned",
        "precondition_hash": current_hashes["closure_hash"],
        "preserved_state_hash": current_hashes["preserved_state_hash"],
        "participant_ids": list(_terminal_recovery_participant_ids(row)),
        "relationship_ids": list(_terminal_recovery_relationship_ids(row)),
        "relationship_descriptors": list(
            _terminal_recovery_relationship_descriptors(row)
        ),
        "source_element_id": source.get("element_id"),
        "node_element_id": node.get("element_id"),
        "name_element_id": name.get("element_id"),
        "binding_element_id": bindings[0].get("element_id")
        if len(bindings) == 1
        else None,
        "projection_element_id": (
            projections[0].get("element_id") if len(projections) == 1 else None
        ),
        "previous_status": manifest_row["expected_source_status"],
        "terminal_stage": manifest_row["expected_name_stage"],
        "retry_event": retry_event,
        "change_event": change_event,
        "retry_event_id": retry_event_id,
        "change_event_id": change_event_id,
    }
    return plan, sorted(set(reasons))


def _read_terminal_recovery_rows(
    transaction: Any,
    pairs: tuple[dict[str, str], ...],
) -> list[dict[str, Any]]:
    """Read each source/DD participant once and each shared target once."""
    source_rows = [
        dict(row)
        for row in transaction.run(_TERMINAL_RECOVERY_SOURCES_QUERY, pairs=list(pairs))
    ]
    node_rows = [
        dict(row)
        for row in transaction.run(_TERMINAL_RECOVERY_NODES_QUERY, pairs=list(pairs))
    ]
    source_ids = sorted(str(pair["source_id"]) for pair in pairs)
    dd_paths = sorted(str(pair["dd_path"]) for pair in pairs)
    sn_ids = sorted({str(pair["sn_id"]) for pair in pairs})
    name_rows = [
        dict(row)
        for row in transaction.run(_TERMINAL_RECOVERY_NAMES_QUERY, sn_ids=sn_ids)
    ]
    if [row.get("source_id") for row in source_rows] != source_ids:
        raise TerminalAttachmentRecoveryConflict(
            "terminal recovery source closure did not return the exact allowlist"
        )
    if [row.get("participant_id") for row in node_rows] != dd_paths:
        raise TerminalAttachmentRecoveryConflict(
            "terminal recovery DD closure did not return the exact allowlist"
        )
    if [row.get("participant_id") for row in name_rows] != sn_ids:
        raise TerminalAttachmentRecoveryConflict(
            "terminal recovery target closure did not return the exact allowlist"
        )
    sources_by_id = {
        str(row["source_id"]): row.get("participants") or [] for row in source_rows
    }
    nodes_by_id = {
        str(row["participant_id"]): row.get("participants") or [] for row in node_rows
    }
    names_by_id = {
        str(row["participant_id"]): row.get("participants") or [] for row in name_rows
    }
    return [
        {
            "source_id": pair["source_id"],
            "dd_path": pair["dd_path"],
            "sn_id": pair["sn_id"],
            "sources": sources_by_id[str(pair["source_id"])],
            "nodes": nodes_by_id[str(pair["dd_path"])],
            "names": names_by_id[str(pair["sn_id"])],
        }
        for pair in pairs
    ]


def _read_terminal_recovery_plan(
    transaction: Any,
    manifest: TerminalAttachmentRecoveryManifest,
    *,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_terminal_recovery_rows(transaction, manifest.pairs)
    manifest_by_source = {row["source_id"]: row for row in manifest.rows}
    cohort_context = _terminal_recovery_cohort_context(manifest)
    plans: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row["source_id"])
        plan, reasons = _terminal_recovery_plan_row(
            row,
            manifest,
            manifest_by_source[source_id],
            reason=reason,
            run_id=run_id,
            changed_at=changed_at,
            cohort_context=cohort_context,
        )
        if reasons or plan is None:
            refusals.append(
                {
                    "source_id": source_id,
                    "reasons": sorted(set(reasons or ["terminal recovery refused"])),
                }
            )
        else:
            plans.append(plan)
    return plans, refusals


def _terminal_recovery_receipt(
    manifest: TerminalAttachmentRecoveryManifest,
    plans: list[dict[str, Any]],
    refusals: list[dict[str, Any]],
    *,
    apply: bool,
    run_id: str | None,
) -> dict[str, Any]:
    counts = Counter(plan["status"] for plan in plans)
    if refusals:
        mode = "refused"
    elif plans and counts["already_current"] == len(plans):
        mode = "already_current"
    else:
        mode = "applied" if apply else "dry_run"
    receipt = {
        "schema": _TERMINAL_RECOVERY_RECEIPT_SCHEMA,
        "schema_version": 1,
        "mode": mode,
        "operation": _TERMINAL_RECOVERY_OPERATION,
        "manifest_path": str(manifest.path),
        "manifest_hash": manifest.manifest_hash,
        "allowlist_hash": manifest.allowlist_hash,
        "run_id": run_id if apply else None,
        "counts": {
            "allowlisted": len(manifest.rows),
            "planned": counts["planned"],
            "already_current": counts["already_current"],
            "applied": counts["planned"] if mode == "applied" else 0,
            "refused": len(refusals),
        },
        "rows": [
            {
                **{
                    key: plan[key]
                    for key in (
                        "source_id",
                        "dd_path",
                        "sn_id",
                        "precondition_hash",
                        "preserved_state_hash",
                        "retry_event_id",
                        "change_event_id",
                    )
                },
                "status": (
                    "applied"
                    if mode == "applied" and plan["status"] == "planned"
                    else plan["status"]
                ),
            }
            for plan in sorted(plans, key=lambda item: item["source_id"])
        ],
        "refusals": sorted(refusals, key=lambda item: item["source_id"]),
    }
    receipt["receipt_hash"] = payload_hash(receipt)
    return receipt


def _lock_terminal_recovery_relationships(
    transaction: Any, relationships: list[dict[str, str]]
) -> tuple[str, ...]:
    by_id = {relationship["element_id"]: relationship for relationship in relationships}
    exact_relationships = [by_id[element_id] for element_id in sorted(by_id)]
    exact_ids = tuple(
        relationship["element_id"] for relationship in exact_relationships
    )
    rows = list(
        transaction.run(
            _TERMINAL_RECOVERY_RELATIONSHIP_LOCK_QUERY,
            relationships=exact_relationships,
        )
    )
    locked = int(dict(rows[0]).get("locked") or 0) if rows else 0
    if locked != len(exact_ids):
        raise TerminalAttachmentRecoveryConflict(
            "terminal recovery relationship set changed before locking"
        )
    return exact_ids


@retry_on_deadlock()
def recover_terminal_attachments(
    manifest_path: str | Path,
    *,
    reason: str,
    apply: bool = False,
    expected_manifest_hash: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Plan or atomically recover one exact terminal-binding cohort."""
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("a terminal attachment recovery reason is required")
    normalized_hash = normalize_manifest_hash_binding(
        expected_manifest_hash, apply=apply
    )
    manifest = load_terminal_attachment_recovery_manifest(manifest_path)
    if normalized_hash is not None and not hmac.compare_digest(
        manifest.manifest_hash, normalized_hash
    ):
        raise ValueError("manifest SHA-256 does not match the exact parsed bytes")
    base_run_id = run_id or (
        f"terminal-attachment-recovery:{uuid.uuid4()}" if apply else None
    )
    invocation_run_id = (
        f"{base_run_id}:manifest:{manifest.manifest_hash}"
        if base_run_id is not None
        else None
    )
    changed_at = datetime.now(UTC).isoformat() if apply else None
    own = gc is None
    if own:
        from imas_codex.graph.client import GraphClient

        client: Any = GraphClient()
    else:
        client = gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                plans, refusals = _read_terminal_recovery_plan(
                    transaction,
                    manifest,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if refusals:
                    transaction.rollback()
                    return _terminal_recovery_receipt(
                        manifest,
                        plans,
                        refusals,
                        apply=apply,
                        run_id=invocation_run_id,
                    )
                pending = [plan for plan in plans if plan["status"] == "planned"]
                current = [
                    plan for plan in plans if plan["status"] == "already_current"
                ]
                if pending and current:
                    transaction.rollback()
                    return _terminal_recovery_receipt(
                        manifest,
                        plans,
                        [
                            {
                                "source_id": "<allowlist>",
                                "reasons": [
                                    "mixed pending and recovered rows cannot prove one atomic cohort"
                                ],
                            }
                        ],
                        apply=apply,
                        run_id=invocation_run_id,
                    )
                if not apply or not pending:
                    transaction.rollback()
                    return _terminal_recovery_receipt(
                        manifest,
                        plans,
                        [],
                        apply=apply,
                        run_id=invocation_run_id,
                    )
                lock_participants(
                    transaction,
                    {
                        participant
                        for plan in pending
                        for participant in plan["participant_ids"]
                    },
                    conflict_type=TerminalAttachmentRecoveryConflict,
                    message="terminal recovery participant set changed before locking",
                )
                _lock_terminal_recovery_relationships(
                    transaction,
                    [
                        relationship
                        for plan in pending
                        for relationship in plan["relationship_descriptors"]
                    ],
                )
                locked_plans, locked_refusals = _read_terminal_recovery_plan(
                    transaction,
                    manifest,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if locked_refusals or [
                    plan["precondition_hash"] for plan in locked_plans
                ] != [plan["precondition_hash"] for plan in plans]:
                    raise TerminalAttachmentRecoveryConflict(
                        "terminal source, target, authority, or relationship state changed after locks"
                    )
                mutation_rows = list(
                    transaction.run(
                        _TERMINAL_RECOVERY_BATCH_QUERY,
                        items=pending,
                        source_statuses=sorted(_TERMINAL_RECOVERY_SOURCE_STATUSES),
                        terminal_stages=sorted(_TERMINAL_RECOVERY_STAGES),
                    )
                )
                if len(mutation_rows) != 1:
                    raise TerminalAttachmentRecoveryConflict(
                        "terminal recovery mutation cardinality changed"
                    )
                mutation = dict(mutation_rows[0])
                expected_sources = {plan["source_id"] for plan in pending}
                if (
                    int(mutation.get("applied") or 0) != len(pending)
                    or set(mutation.get("source_ids") or []) != expected_sources
                    or set(mutation.get("retry_event_ids") or [])
                    != {plan["retry_event_id"] for plan in pending}
                    or set(mutation.get("change_event_ids") or [])
                    != {plan["change_event_id"] for plan in pending}
                ):
                    raise TerminalAttachmentRecoveryConflict(
                        "terminal recovery mutation cardinality changed"
                    )
                post_plans, post_refusals = _read_terminal_recovery_plan(
                    transaction,
                    manifest,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if post_refusals or any(
                    plan["status"] != "already_current" for plan in post_plans
                ):
                    raise TerminalAttachmentRecoveryConflict(
                        "terminal recovery postflight proof did not hold"
                    )
                before = {plan["source_id"]: plan for plan in pending}
                if any(
                    plan["preserved_state_hash"]
                    != before[plan["source_id"]]["preserved_state_hash"]
                    for plan in post_plans
                ):
                    raise TerminalAttachmentRecoveryConflict(
                        "terminal recovery changed protected graph state"
                    )
                transaction.commit()
                return _terminal_recovery_receipt(
                    manifest,
                    pending,
                    [],
                    apply=True,
                    run_id=invocation_run_id,
                )
            except BaseException:
                try:
                    transaction.rollback()
                except Exception:
                    pass
                raise
    finally:
        if own:
            client.close()


def recover_terminal_attachment(
    dd_path: str,
    sn_id: str,
    *,
    reason: str,
    gc: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Recover one source finalized against a terminal name collision.

    This is deliberately narrower than ordinary semantic detachment. The
    source must be the exact DD provenance node for *dd_path*, be finalized as
    ``composed`` or ``attached`` and unclaimed, with both realization edges and
    scalar mirror pointing only to *sn_id*, and the target must still be
    terminal. Any mismatch refuses the operation without mutation.

    The live write is one transaction: it removes both realization edges and
    the target's source-path projection, returns the source to a fresh
    composition attempt, and records both a source retry event and a name
    change event carrying the terminal identity, stage, and operator reason.
    """
    reason = reason.strip()
    if not reason:
        raise ValueError("a non-empty recovery reason is required")

    own = gc is None
    if own:
        from imas_codex.graph.client import GraphClient

        client: Any = GraphClient()
    else:
        client = gc
    source_node_id = f"dd:{dd_path}"
    terminal_stages = sorted(_TERMINAL_RECOVERY_STAGES)
    result: dict[str, Any] = {
        "ok": False,
        "dd_path": dd_path,
        "sn_id": sn_id,
        "source_node_id": source_node_id,
        "dry_run": dry_run,
    }
    try:
        if dry_run:
            rows = list(
                client.query(
                    _TERMINAL_RECOVERY_ELIGIBILITY_QUERY,
                    source_node_id=source_node_id,
                    dd_path=dd_path,
                    sn_id=sn_id,
                    terminal_stages=terminal_stages,
                    source_statuses=sorted(_TERMINAL_RECOVERY_SOURCE_STATUSES),
                )
            )
            if not rows:
                return {
                    **result,
                    "reason": "exact terminal source binding is not recoverable",
                }
            row = rows[0]
            return {
                **result,
                "ok": True,
                "name_stage": row["name_stage"],
                "previous_attempt_count": row["attempt_count"],
            }

        with client.session() as session:
            tx = session.begin_transaction()
            try:
                rows = [
                    dict(row)
                    for row in tx.run(
                        _TERMINAL_RECOVERY_QUERY,
                        source_node_id=source_node_id,
                        dd_path=dd_path,
                        sn_id=sn_id,
                        terminal_stages=terminal_stages,
                        source_statuses=sorted(_TERMINAL_RECOVERY_SOURCE_STATUSES),
                        reason=reason,
                        retry_event_id=f"source-retry:{uuid.uuid4()}",
                        change_event_id=f"sn-change:{uuid.uuid4()}",
                    )
                ]
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
        if not rows:
            return {
                **result,
                "reason": "exact terminal source binding changed or is ineligible",
            }
        row = rows[0]
        result.update(
            {
                "ok": True,
                "name_stage": row["name_stage"],
                "retry_event_id": row["retry_event_id"],
                "change_event_id": row["change_event_id"],
            }
        )
        logger.info(
            "recover_terminal_attachment: returned %s to composition after "
            "detaching terminal target %s at %s (%s)",
            source_node_id,
            sn_id,
            row["name_stage"],
            reason,
        )
        return result
    finally:
        if own:
            client.close()


def detach_one_attachment(
    dd_path: str,
    sn_id: str,
    *,
    reason: str,
    gc: Any | None = None,
    dry_run: bool = False,
    terminal_recovery: bool = False,
) -> dict[str, Any]:
    """Detach ONE source→name realization the consistency guard cannot judge.

    The guard decides mechanical questions — dimensionality, locus/device, vector
    families. It is silent on a *semantic* mis-share: two names that are each
    valid, denote different quantities, and both got attached to one DD path, so
    the exported catalog lists that path under both. Deciding which one the path
    realizes is a physics judgement, and once made it needs an instrument.

    Removes the realization edges (``PRODUCED_NAME``, the DD-side
    ``HAS_STANDARD_NAME``, and the name's ``source_paths`` entry) and rewinds the
    source to ``extracted`` for re-composition ONLY when this was its last live
    name — reusing the reconcile's own detach semantics so a targeted repair and
    the retroactive sweep leave the graph in the same shape. Writes a
    ``StandardNameChange`` so the judgement and its reason survive in the ledger.

    Refuses when the pairing does not exist, or when it is the name's ONLY
    remaining attachment: a name whose every source is wrong is a NAME defect to
    repair with ``sn edit --rename``, never something to orphan by detaching.
    Exception: a derived parent with live children survives losing its last
    realization — its identity is anchored by the ``HAS_PARENT`` structure, and
    accepted derived parents routinely carry no realization at all, so the
    detach returns it to that designed state instead of orphaning it.

    ``terminal_recovery=True`` selects the stricter transactional recovery for
    a source finalized against a terminal target. It bypasses the ordinary
    would-orphan refusal only after proving the exact corrupted binding and
    writes the retry/change ledgers atomically.

    Returns ``{"ok": bool, ...}``; never raises on a refusal.
    """
    if terminal_recovery:
        return recover_terminal_attachment(
            dd_path,
            sn_id,
            reason=reason,
            gc=gc,
            dry_run=dry_run,
        )

    own = gc is None
    if own:
        from imas_codex.graph.client import GraphClient

        client: Any = GraphClient()
    else:
        client = gc
    try:
        # A pairing can exist as the DD-side projection alone: HAS_STANDARD_NAME
        # without the PRODUCED_NAME provenance behind it. The export reads the
        # projection, so a projection-only pairing is exactly the kind that
        # reaches the catalog, and it must be reachable here. Existence is
        # therefore either edge, and the would-orphan guard counts REALIZATIONS
        # (what a consumer sees), not provenance rows.
        rows = client.query(
            """
            MATCH (dd:IMASNode {id: $dd_path})
            MATCH (sn:StandardName {id: $sn_id})
            OPTIONAL MATCH (src:StandardNameSource)-[:FROM_DD_PATH]->(dd)
            WHERE (src)-[:PRODUCED_NAME]->(sn)
            WITH dd, sn, collect(src)[0] AS src
            OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(other:StandardName)
            WHERE other.id <> $sn_id
              AND NOT coalesce(other.name_stage, '') IN $historical
            WITH dd, sn, src, count(DISTINCT other) AS other_live
            RETURN src.id AS source_node_id,
                   other_live AS other_live_names,
                   EXISTS { (dd)-[:HAS_STANDARD_NAME]->(sn) } AS projected,
                   COUNT { (:IMASNode)-[:HAS_STANDARD_NAME]->(sn) } AS name_attachments,
                   (sn.origin = 'derived' AND EXISTS {
                        (child:StandardName)-[:HAS_PARENT]->(sn)
                        WHERE NOT coalesce(child.name_stage, '') IN $historical
                   }) AS structural_parent
            """,
            dd_path=dd_path,
            sn_id=sn_id,
            historical=sorted(_HISTORICAL_NAME_STAGES),
        )
        row = rows[0] if rows else None
        if not row or not (row.get("source_node_id") or row.get("projected")):
            return {
                "ok": False,
                "reason": f"{dd_path!r} does not realize {sn_id!r}",
            }
        # A derived parent with live children is anchored by its HAS_PARENT
        # structure, not by realizations — accepted derived parents routinely
        # carry none. Removing its last realization returns it to that
        # designed state rather than orphaning it, so the would-orphan
        # refusal does not apply.
        structural_parent = bool(row.get("structural_parent"))
        if int(row["name_attachments"]) <= 1 and not structural_parent:
            return {
                "ok": False,
                "reason": (
                    f"{sn_id!r} has only this one attachment — a name rejected by "
                    "its whole source set is a NAME defect; repair it with "
                    "sn edit --rename rather than orphaning it"
                ),
            }

        # With no provenance node behind the projection there is no source to
        # rewind — only the dangling projection to remove.
        has_source = bool(row["source_node_id"])
        reroute = has_source and int(row["other_live_names"] or 0) == 0
        result = {
            "ok": True,
            "dd_path": dd_path,
            "sn_id": sn_id,
            "source_node_id": row["source_node_id"],
            "source_rewound": reroute,
            "projection_only": not has_source,
            "structural_parent": structural_parent,
            "dry_run": dry_run,
        }
        if dry_run:
            return result

        if has_source:
            client.query(
                _DETACH_QUERY,
                items=[
                    {
                        "source_node_id": row["source_node_id"],
                        "dd_path": dd_path,
                        "sn_id": sn_id,
                        "reroute": reroute,
                    }
                ],
                historical=sorted(_HISTORICAL_NAME_STAGES),
            )
        else:
            client.query(_DETACH_PROJECTION_QUERY, dd_path=dd_path, sn_id=sn_id)
        _record_detachments(
            client,
            [
                AttachmentVerdict(
                    source_node_id=row["source_node_id"],
                    dd_path=dd_path,
                    sn_id=sn_id,
                    name_stage=None,
                    reason=f"semantic mis-share: {reason}",
                    other_live_names=int(row["other_live_names"] or 0),
                )
            ],
        )
    finally:
        if own:
            client.close()

    logger.info(
        "detach_one_attachment: %s no longer realizes %s (%s) — %s",
        dd_path,
        sn_id,
        "source rewound to compose" if reroute else "source keeps another live name",
        reason,
    )
    return result


def gate_migrated_attachments(
    gc: Any | None = None, *, sn_id: str
) -> AttachmentAuditResult:
    """Re-validate one name's attachments right after a source set migrated onto it.

    A path that moves a predecessor's whole ``PRODUCED_NAME`` / ``HAS_STANDARD_NAME``
    set onto a DIFFERENT name creates a new *pairing* out of a historical source
    set, and a new pairing is exactly what the consistency guard exists to judge.
    Without this, a rename can launder an edge the guard would have refused at
    compose time — which is how conductor-geometry vertices reached optical
    ``line_of_sight`` names.

    Scoped to the one name because this runs on a hot path: a corpus-wide audit
    per refine is not affordable. ``include_accepted`` stays off — a freshly
    migrated successor is never accepted yet, and an accepted name losing a
    source must remain a deliberate decision.

    Never raises: a gate that can fail the write it protects would turn a
    recoverable bad edge into a lost rename. Rejections are logged and, when the
    whole source set trips one rule, reported as a NAME defect rather than
    detached.
    """
    try:
        return reconcile_attachment_consistency(gc, sn_id=sn_id)
    except Exception:  # pragma: no cover - the rename must survive a gate failure
        logger.warning(
            "gate_migrated_attachments: could not re-validate attachments "
            "migrated onto %s; the corpus-wide audit will catch them",
            sn_id,
            exc_info=True,
        )
        return AttachmentAuditResult()


def _record_detachments(gc: Any, verdicts: list[AttachmentVerdict]) -> None:
    """Write one ``StandardNameChange`` per detachment so history survives."""
    from imas_codex.standard_names.provenance_lifecycle import (
        record_standard_name_change,
    )

    for v in verdicts:
        try:
            record_standard_name_change(
                gc,
                v.dd_path,
                v.sn_id,
                operation="detach_inconsistent_attachment",
                reason=v.reason,
                origin="attachment_consistency_reconcile",
            )
        except Exception:  # pragma: no cover - audit crumb must not block the fix
            logger.debug("Failed to record detachment of %s", v.dd_path, exc_info=True)


def _find_orphaned_names(gc: Any, sn_ids: list[str]) -> list[str]:
    """Names among *sn_ids* left with no producing source after detachment."""
    if not sn_ids:
        return []
    rows = gc.query(
        """
        UNWIND $ids AS sn_id
        MATCH (sn:StandardName {id: sn_id})
        WHERE NOT (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        RETURN sn.id AS id
        """,
        ids=sn_ids,
    )
    return sorted(r["id"] for r in rows)

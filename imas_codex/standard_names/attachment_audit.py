"""Retroactive re-validation of source→name attachments already in the graph.

``workers._is_attachment_consistent`` decides whether a DD source path may
realize a standard name. It is evaluated at COMPOSE time only, so an attachment
written before a rule existed — or written by one of the many paths that never
consult it (``persist_refined_name`` migrates a predecessor's whole source set
onto the successor, ``rebind_sources`` and ``bind_sources_exclusively`` move
sources on an edit cascade, the derived-parent seeders and the provenance
rebuilders MERGE edges directly) — is never revisited. A decision cached
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

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from imas_codex.graph.models import NameStage

logger = logging.getLogger(__name__)

__all__ = [
    "AttachmentPairingGuardResult",
    "AttachmentVerdict",
    "AttachmentAuditResult",
    "NameLevelDefect",
    "audit_attachments",
    "guard_source_pairings",
    "recover_terminal_attachment",
    "reconcile_attachment_consistency",
]

_TERMINAL_RECOVERY_STAGES: frozenset[str] = frozenset(
    {
        NameStage.superseded.value,
        NameStage.exhausted.value,
        NameStage.contested.value,
    }
)

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
OPTIONAL MATCH (dd)-[:IN_IDS]->(ids:IDS)
OPTIONAL MATCH (dd)-[:HAS_UNIT]->(du:Unit)
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(nu:Unit)
OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(other:StandardName)
WHERE other.id <> sn.id
  AND NOT (coalesce(other.name_stage, '') IN $historical)
RETURN src.id            AS source_node_id,
       dd.id             AS dd_path,
       sn.id             AS sn_id,
       sn.name_stage     AS name_stage,
       sn.origin         AS origin,
       CASE size(collect(DISTINCT du.id))
         WHEN 0 THEN dd.unit
         WHEN 1 THEN head(collect(DISTINCT du.id))
         ELSE null
       END AS dd_unit,
       dd.unit           AS dd_declared_unit,
       collect(DISTINCT du.id) AS dd_relationship_units,
       ids.dd_version    AS dd_version,
       coalesce(head(collect(DISTINCT nu.id)), sn.unit) AS sn_unit,
       count(DISTINCT other) AS other_live_names
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
  AND src.status = 'composed'
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
  AND src.status = 'composed'
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
    ``composed`` with both realization edges and scalar mirror pointing only to
    *sn_id*, and the target must still be terminal. Any mismatch refuses the
    operation without mutation.

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

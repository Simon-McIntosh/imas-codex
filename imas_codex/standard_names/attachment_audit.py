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

**Accepted names.** An accepted name losing a source is a real event, not
routine hygiene: the name is catalog-authoritative and may already be published.
Following the precedent of ``sn run --reset-to`` / ``sn prune``, attachments on
``name_stage='accepted'`` names are reported but NOT detached unless the caller
passes ``include_accepted=True``. A name left with no source at all is flagged
in the result as ``names_orphaned`` — that is a name whose entire justification
was a wrong attachment and it needs a human decision, not an automatic
supersede.

Idempotent: a detached attachment no longer matches the selector, and an
attachment the guard accepts is never touched, so a second pass acts on nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AttachmentVerdict",
    "AttachmentAuditResult",
    "audit_attachments",
    "reconcile_attachment_consistency",
]

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

#: Every attachment the graph asserts, with the unit context the dimensionality
#: rule needs on both sides. Sibling source paths of the same name are collected
#: so the distinct-vector rule sees what the name already carries.
#:
#: ``other_live_names`` counts the OTHER names this source still produces. A
#: source that also backs a live name has already been re-composed correctly;
#: only the stale extra edge should go, and the source must NOT be rewound to
#: ``extracted`` — that would strand the good name it already produced.
_ATTACHMENTS_QUERY = """
MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
OPTIONAL MATCH (dd)-[:HAS_UNIT]->(du:Unit)
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(nu:Unit)
OPTIONAL MATCH (sibling:StandardNameSource)-[:PRODUCED_NAME]->(sn)
OPTIONAL MATCH (sibling)-[:FROM_DD_PATH]->(sib_dd:IMASNode)
WITH src, sn, dd, du, nu,
     [p IN collect(DISTINCT sib_dd.id) WHERE p IS NOT NULL AND p <> dd.id] AS siblings
OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(other:StandardName)
WHERE other.id <> sn.id
  AND NOT (coalesce(other.name_stage, '') IN $historical)
RETURN src.id            AS source_node_id,
       dd.id             AS dd_path,
       sn.id             AS sn_id,
       sn.name_stage     AS name_stage,
       sn.origin         AS origin,
       coalesce(du.id, dd.unit) AS dd_unit,
       coalesce(nu.id, sn.unit) AS sn_unit,
       siblings          AS siblings,
       count(DISTINCT other) AS other_live_names
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


@dataclass
class AttachmentAuditResult:
    """Outcome of one audit / reconcile pass."""

    checked: int = 0
    rejected: list[AttachmentVerdict] = field(default_factory=list)
    detached: int = 0
    sources_rerouted: int = 0
    skipped_protected: int = 0
    skipped_historical: int = 0
    names_orphaned: list[str] = field(default_factory=list)

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
            "names_orphaned": len(self.names_orphaned),
            "by_rule": self.by_rule(),
        }


def audit_attachments(gc: Any | None = None) -> AttachmentAuditResult:
    """Re-validate every graph attachment against the full guard suite. Read-only.

    Returns the verdicts without touching the graph, so the same predicate backs
    both the reporting path and the reconcile.
    """
    from imas_codex.standard_names.workers import _is_attachment_consistent

    own = gc is None
    if own:
        from imas_codex.graph.client import GraphClient

        client: Any = GraphClient()
    else:
        client = gc
    try:
        rows = list(
            client.query(_ATTACHMENTS_QUERY, historical=sorted(_HISTORICAL_NAME_STAGES))
        )
    finally:
        if own:
            client.close()

    result = AttachmentAuditResult(checked=len(rows))
    for r in rows:
        ok, reason = _is_attachment_consistent(
            r["dd_path"],
            r["sn_id"],
            existing_sources=r["siblings"] or (),
            dd_unit=r["dd_unit"],
            sn_unit=r["sn_unit"],
        )
        if ok:
            continue
        result.rejected.append(
            AttachmentVerdict(
                source_node_id=r["source_node_id"],
                dd_path=r["dd_path"],
                sn_id=r["sn_id"],
                name_stage=r["name_stage"],
                reason=reason,
                other_live_names=int(r.get("other_live_names") or 0),
            )
        )
    return result


def reconcile_attachment_consistency(
    gc: Any | None = None, *, include_accepted: bool = False, dry_run: bool = False
) -> AttachmentAuditResult:
    """Detach every inconsistent attachment and reroute its source to compose.

    Args:
        gc: An open ``GraphClient``, or None to open and close one.
        include_accepted: Also detach attachments on ``name_stage='accepted'``
            names. Off by default: an accepted name is catalog-authoritative, so
            losing a source is a decision, not hygiene.
        dry_run: Report the verdicts without writing.

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
        result = audit_attachments(client)
        result.skipped_historical = sum(1 for v in result.rejected if v.historical)
        result.skipped_protected = sum(
            1 for v in result.rejected if v.protected and not include_accepted
        )
        actionable = [
            v
            for v in result.rejected
            if not v.historical and (include_accepted or not v.protected)
        ]

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
                    if v.protected
                )[:1200],
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

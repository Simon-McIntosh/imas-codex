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
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AttachmentVerdict",
    "AttachmentAuditResult",
    "NameLevelDefect",
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
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(nu:Unit)
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
       count(DISTINCT other) AS other_live_names
"""

#: Scope clause narrowing the read to one name. Used by the write-time gate on
#: paths that migrate a historical source set onto a NEW name, where a full
#: corpus audit would be far too expensive for a hot path.
_ONE_NAME_SCOPE = "WHERE sn.id = $sn_id"

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
            "by_rule": self.by_rule(),
        }


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

    result = AttachmentAuditResult(checked=len(rows))
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


def detach_one_attachment(
    dd_path: str,
    sn_id: str,
    *,
    reason: str,
    gc: Any | None = None,
    dry_run: bool = False,
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

    Returns ``{"ok": bool, ...}``; never raises on a refusal.
    """
    own = gc is None
    if own:
        from imas_codex.graph.client import GraphClient

        client: Any = GraphClient()
    else:
        client = gc
    try:
        rows = client.query(
            """
            MATCH (src:StandardNameSource)-[:FROM_DD_PATH]->(dd:IMASNode {id: $dd_path})
            MATCH (src)-[:PRODUCED_NAME]->(sn:StandardName {id: $sn_id})
            OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(other:StandardName)
            WHERE other.id <> $sn_id
              AND NOT coalesce(other.name_stage, '') IN $historical
            WITH src, sn, count(DISTINCT other) AS other_live
            OPTIONAL MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            RETURN src.id AS source_node_id, other_live AS other_live_names,
                   count(*) AS name_attachments
            """,
            dd_path=dd_path,
            sn_id=sn_id,
            historical=sorted(_HISTORICAL_NAME_STAGES),
        )
        row = rows[0] if rows else None
        if not row or not row.get("source_node_id"):
            return {
                "ok": False,
                "reason": f"{dd_path!r} does not realize {sn_id!r}",
            }
        if int(row["name_attachments"]) <= 1:
            return {
                "ok": False,
                "reason": (
                    f"{sn_id!r} has only this one attachment — a name rejected by "
                    "its whole source set is a NAME defect; repair it with "
                    "sn edit --rename rather than orphaning it"
                ),
            }

        reroute = int(row["other_live_names"] or 0) == 0
        result = {
            "ok": True,
            "dd_path": dd_path,
            "sn_id": sn_id,
            "source_node_id": row["source_node_id"],
            "source_rewound": reroute,
            "dry_run": dry_run,
        }
        if dry_run:
            return result

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

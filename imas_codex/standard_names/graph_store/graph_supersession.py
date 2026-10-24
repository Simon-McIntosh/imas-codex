"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

from . import graph_foundation as foundation
from .graph_foundation import _TransactionQuery


@retry_on_deadlock()
def supersede_prior_source_names(
    pairs: list[dict[str, str]],
    *,
    gc: Any | None = None,
) -> int:
    """Supersede stale pipeline names left on a source by ``--force``/regen.

    Enforces the invariant **one source → at most one non-superseded
    pipeline-origin name**.  When ``--force`` (or a regen pass) composes a
    *new* name for a DD source that already carries an accepted/in-flight
    pipeline name, the new ``StandardName`` node and its ``HAS_STANDARD_NAME``
    edge are created without retiring the predecessor — leaving two live names
    competing for one source (the Class-A duplicate).  This helper, called
    immediately after persisting the regenerated batch, retires those
    predecessors.

    For each ``{"new_name": <id>, "source_id": <dd path>}`` pair, any *other*
    ``StandardName`` linked to that source via ``HAS_STANDARD_NAME`` is
    superseded when **all** of:

    - its id differs from ``new_name`` (byte-identical regen is a no-op MERGE —
      the same node is reused, so nothing is superseded);
    - it is not already ``superseded``/``exhausted``;
    - it is not published catalog content: neither ``name_stage='approved'``
      nor carrying a ``catalog_pr_number``. Both conditions are required
      because they are set by the same merge write, so either one alone would
      let a half-written or hand-repaired row through. A name that reached
      ``approved`` was published by a merged catalog pull request; retiring it
      from an automatic compose pass would revoke governed content with no
      human in the loop. Withdrawing published content is a governance act and
      goes through the catalog, never through this dedup;
    - its ``origin`` is not ``derived`` — a structural parent is owned by the
      admission gate, not by any one source, so a source-keyed dedup must not
      retire it. ``catalog_edit`` alone is NOT exempt: unpublished names
      carrying that origin are one bulk import of this pipeline's own earlier
      output, sitting unreviewed, and the graph plus the review pipeline is
      the source of truth. Exempting them let an imported name keep competing
      for a source that had already recomposed, which is how one coordinate
      axis resolved to a single name while another stayed split across two.
      Publication, not origin, is what makes catalog content untouchable.

    **Only a source a composer already bound to the successor carries its
    provenance across.** A standard name is a stable semantic identity and a
    source is provenance for it, so the only evidence that a source realizes
    the successor is that something judged the pairing: the recomposed source
    itself, and any source already producing the successor. Every other source
    still realizing the predecessor is enumerated and left exactly where it is.
    Migrating them wholesale — retargeting a predecessor's entire population
    because one of its sources recomposed — stamps one source's new identity
    onto sources no composer and no reviewer ever judged against it, and is how
    a diagnostic line-of-sight name came to carry a conductor cross-section and
    a field-map grid axis. Unit and dimension agreement cannot catch it: both
    sides are metres.

    Before changing lifecycle, lineage, or provenance, the helper discovers the
    judged source set for every predecessor-to-successor pair and requires the
    canonical attachment guard to admit every pairing.  It then retargets
    exactly that set and compare-and-sets the predecessor state before marking
    it superseded.  A missing source, guard rejection, partial move, or
    concurrent drift raises so the caller-owned transaction rolls the generated
    successor and every associated mutation back together.

    A predecessor the judged set does not drain is **not** superseded. It stays
    live, carrying exactly the sources that still realize it, each of which
    still holds one live name — so the ``one source → at most one
    non-superseded pipeline name`` invariant holds for every source including
    the recomposed one, and no row is stranded. The refusal is recorded as a
    ``supersession_deferred`` entry in the change ledger naming both names and
    the retained count, so a split awaiting convergence is enumerable rather
    than silent. Convergence is then an explicit act: recompose the retained
    sources, or fold the names with :func:`tombstone_supersede_into`.

    For each superseded predecessor the helper marks it
    ``name_stage='superseded'``, clears its claim, and creates a
    ``(new)-[:REFINED_FROM]->(old)`` edge so chain history and the SPA's
    lineage widget stay intact. Historical relationships remain represented by
    the change ledger; the live provenance binding is moved exclusively to the
    successor.

    Returns the number of predecessor names superseded.
    """
    pairs = sorted(
        {
            (p["new_name"], p["source_id"])
            for p in pairs
            if p.get("new_name") and p.get("source_id")
        }
    )
    if not pairs:
        return 0

    normalized_pairs = [
        {"new_name": new_name, "source_id": source_id} for new_name, source_id in pairs
    ]

    def _apply(query_handle: Any) -> int:
        preflight_rows = list(
            query_handle.query(
                foundation.services._GENERATED_SUPERSESSION_PREFLIGHT_QUERY,
                pairs=normalized_pairs,
            )
            or []
        )

        returned_pairs = {
            (row.get("new_name"), row.get("requested_source_id"))
            for row in preflight_rows
        }
        expected_pairs = {
            (pair["new_name"], pair["source_id"]) for pair in normalized_pairs
        }
        if returned_pairs != expected_pairs:
            raise RuntimeError(
                "generated-name supersession preflight changed before it could "
                "enumerate every requested source"
            )
        missing_requested = [
            row["requested_source_id"]
            for row in preflight_rows
            if not row.get("requested_source_exists")
        ]
        missing_successors = [
            row["new_name"] for row in preflight_rows if not row.get("successor_exists")
        ]
        if missing_requested or missing_successors:
            details = []
            if missing_requested:
                details.append(
                    "missing DD sources " + ", ".join(sorted(set(missing_requested)))
                )
            if missing_successors:
                details.append(
                    "missing successors " + ", ".join(sorted(set(missing_successors)))
                )
            raise RuntimeError(
                "generated-name supersession preflight failed: " + "; ".join(details)
            )

        plans_by_pair: dict[tuple[str, str], dict[str, foundation.services.Any]] = {}
        winners_by_old: dict[str, set[str]] = {}
        for row in preflight_rows:
            old_name = row.get("old_name")
            if not old_name:
                continue
            new_name = row["new_name"]
            source_ids = sorted(set(row.get("judged_source_ids") or []))
            trigger_source_id = row.get("trigger_source_id")
            if not trigger_source_id or trigger_source_id not in source_ids:
                # The recomposed source is the one pairing this pass produced
                # evidence for. Without it there is nothing to carry across and
                # the predecessor binding it still holds would go unresolved.
                raise RuntimeError(
                    "generated-name supersession preflight found no successor "
                    f"binding for the recomposed source of {old_name!r} -> "
                    f"{new_name!r}"
                )
            key = (old_name, new_name)
            plan = plans_by_pair.setdefault(
                key,
                {
                    "old_name": old_name,
                    "new_name": new_name,
                    "old_stage": row.get("old_stage"),
                    "source_ids": [],
                    "retained_source_ids": [],
                },
            )
            if plan["old_stage"] != row.get("old_stage"):
                raise RuntimeError(
                    "generated-name supersession predecessor stage changed "
                    f"during preflight for {old_name!r}"
                )
            plan["source_ids"] = sorted(set(plan["source_ids"]) | set(source_ids))
            plan["retained_source_ids"] = sorted(
                set(plan["retained_source_ids"])
                | set(row.get("retained_source_ids") or [])
            )
            winners_by_old.setdefault(old_name, set()).add(new_name)

        conflicting = {
            old_name: sorted(winners)
            for old_name, winners in winners_by_old.items()
            if len(winners) != 1
        }
        if conflicting:
            raise RuntimeError(
                "generated-name supersession cannot route one predecessor to "
                f"multiple successors: {conflicting}"
            )

        plans = [plans_by_pair[key] for key in sorted(plans_by_pair)]
        if not plans:
            return 0

        from imas_codex.standard_names.attachment_audit import guard_source_pairings

        for plan in plans:
            guarded = guard_source_pairings(
                query_handle, plan["new_name"], plan["source_ids"]
            )
            expected = set(plan["source_ids"])
            accepted = set(guarded.accepted_source_ids)
            if guarded.rejected or accepted != expected:
                rejected = (
                    ", ".join(
                        f"{item.source_node_id}: {item.reason}"
                        for item in guarded.rejected
                    )
                    or "the pairing guard did not admit every source"
                )
                raise ValueError(
                    "generated-name attachment rejected; supersession rolled "
                    f"back for {plan['old_name']} -> {plan['new_name']}: {rejected}"
                )

        from imas_codex.standard_names.provenance_lifecycle import (
            record_standard_name_change,
            retarget_standard_name_sources,
        )

        for plan in plans:
            moved = retarget_standard_name_sources(
                query_handle,
                plan["old_name"],
                plan["new_name"],
                operation="regenerate",
                enforce_consistency=False,
                source_ids=plan["source_ids"],
                expected_current_bindings=dict.fromkeys(
                    plan["source_ids"], plan["old_name"]
                ),
            )
            expected = len(plan["source_ids"])
            if moved != expected:
                raise RuntimeError(
                    "semantic source set changed during generated-name "
                    f"supersession: expected {expected}, moved {moved} for "
                    f"{plan['old_name']} -> {plan['new_name']}"
                )

        deferred = [plan for plan in plans if plan["retained_source_ids"]]
        drained = [plan for plan in plans if not plan["retained_source_ids"]]
        if deferred:
            # The retarget rebuilds the successor's path projection from the
            # set it moved and empties the predecessor's. A predecessor that
            # stays live must get its own projection back, rebuilt from the
            # bindings it still holds.
            query_handle.query(
                """
                UNWIND $names AS old_name
                MATCH (old:StandardName {id: old_name})
                OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(old)
                OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
                OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
                WITH old, collect(DISTINCT CASE
                       WHEN source IS NULL THEN null
                       WHEN dd IS NOT NULL THEN 'dd:' + dd.id
                       WHEN signal IS NOT NULL THEN signal.id
                       WHEN source.source_type = 'derived'
                        AND source.source_id STARTS WITH 'derived:'
                       THEN source.source_id
                       ELSE source.id END) AS paths
                SET old.source_paths = [p IN paths WHERE p IS NOT NULL], old.updated_at = datetime()
                """,
                names=sorted({plan["old_name"] for plan in deferred}),
            )
            for plan in deferred:
                retained = plan["retained_source_ids"]
                record_standard_name_change(
                    query_handle,
                    plan["old_name"],
                    plan["new_name"],
                    operation="supersession_deferred",
                    reason=(
                        f"{len(retained)} source(s) still realize "
                        f"{plan['old_name']} and were never judged against "
                        f"{plan['new_name']}; the predecessor stays live rather "
                        "than carrying them across"
                    ),
                    origin="generated_name_supersession",
                )
                foundation.services.logger.warning(
                    "supersede_prior_source_names: %s kept live — %d source(s) "
                    "not judged against %s (%s)",
                    plan["old_name"],
                    len(retained),
                    plan["new_name"],
                    ", ".join(retained[:5]),
                )
        if not drained:
            return 0

        rows = list(
            query_handle.query(
                """
                // GENERATED_SUPERSESSION_FINALIZE
                UNWIND $plans AS plan
                MATCH (old:StandardName {id: plan.old_name}),
                      (new:StandardName {id: plan.new_name})
                WHERE coalesce(old.name_stage, '') =
                      coalesce(plan.old_stage, '')
                  AND NOT coalesce(old.name_stage, '') IN
                      ['superseded', 'exhausted', 'contested', 'approved']
                  AND old.catalog_pr_number IS NULL
                  AND coalesce(old.origin, 'pipeline') <> 'derived'
                  AND new.id <> old.id
                  AND NOT (old)-[:REFINED_FROM*1..]->(new)
                OPTIONAL MATCH (remaining:StandardNameSource)
                WHERE (remaining)-[:PRODUCED_NAME]->(old)
                   OR remaining.produced_sn_id = old.id
                WITH plan, old, new,
                     count(DISTINCT remaining) AS remaining_sources
                WHERE remaining_sources = 0
                UNWIND plan.source_ids AS expected_source_id
                MATCH (moved:StandardNameSource {id: expected_source_id})
                      -[:PRODUCED_NAME]->(new)
                WHERE moved.produced_sn_id = new.id
                WITH plan, old, new,
                     count(DISTINCT moved) AS moved_sources
                WHERE moved_sources = size(plan.source_ids)
                // A still-open steered edit (name-hint regeneration) must ride
                // the recomposed successor so its lifecycle can resolve at
                // review time; capture the predecessor's edit fields BEFORE any
                // mutation so `new` inherits the ORIGINAL 'open' status while
                // `old` is reconciled to 'applied' (its steering produced this
                // successor — it is no longer stuck 'open').
                WITH old, new,
                     (coalesce(old.edit_status, '') = 'open') AS carry_edit,
                     old.name_stage AS o_prior_stage,
                     old.edit_mode AS o_edit_mode,
                     old.name_hint AS o_name_hint,
                     old.docs_hint AS o_docs_hint,
                     old.edit_reason AS o_edit_reason,
                     old.edit_origin AS o_edit_origin,
                     old.edit_scope AS o_edit_scope,
                     old.edit_requested_at AS o_edit_requested_at,
                     old.edit_override_edits AS o_override_edits,
                     old.edit_include_accepted AS o_include_accepted,
                     old.edit_status AS o_edit_status
                SET old.updated_at = datetime(), new.updated_at = datetime(), old.name_stage = 'superseded',
                    // Capture the predecessor's live name-axis stage before the
                    // flip so the export boundary can emit a deprecation stub
                    // only for names that had reached 'accepted'. Unlike the
                    // refine path, this path supersedes a node still carrying
                    // its real stage, so it is recorded verbatim.
                    old.superseded_from_stage =
                        coalesce(old.superseded_from_stage, o_prior_stage),
                    old.claim_token = null,
                    old.claimed_at = null,
                    old.edit_status = CASE WHEN carry_edit THEN 'applied'
                                          ELSE old.edit_status END,
                    new.edit_mode = CASE WHEN carry_edit
                        THEN coalesce(new.edit_mode, o_edit_mode)
                        ELSE new.edit_mode END,
                    new.name_hint = CASE WHEN carry_edit
                        THEN coalesce(new.name_hint, o_name_hint)
                        ELSE new.name_hint END,
                    new.docs_hint = CASE WHEN carry_edit
                        THEN coalesce(new.docs_hint, o_docs_hint)
                        ELSE new.docs_hint END,
                    new.edit_reason = CASE WHEN carry_edit
                        THEN coalesce(new.edit_reason, o_edit_reason)
                        ELSE new.edit_reason END,
                    new.edit_origin = CASE WHEN carry_edit
                        THEN coalesce(new.edit_origin, o_edit_origin)
                        ELSE new.edit_origin END,
                    new.edit_scope = CASE WHEN carry_edit
                        THEN coalesce(new.edit_scope, o_edit_scope)
                        ELSE new.edit_scope END,
                    new.edit_requested_at = CASE WHEN carry_edit
                        THEN coalesce(new.edit_requested_at, o_edit_requested_at)
                        ELSE new.edit_requested_at END,
                    new.edit_override_edits = CASE WHEN carry_edit
                        THEN coalesce(new.edit_override_edits, o_override_edits)
                        ELSE new.edit_override_edits END,
                    new.edit_include_accepted = CASE WHEN carry_edit
                        THEN coalesce(new.edit_include_accepted, o_include_accepted)
                        ELSE new.edit_include_accepted END,
                    new.edit_status = CASE WHEN carry_edit
                        THEN coalesce(new.edit_status, o_edit_status)
                        ELSE new.edit_status END
                MERGE (new)-[:REFINED_FROM]->(old)
                RETURN old.id AS old_name, new.id AS new_name
                """,
                plans=[
                    {
                        "old_name": plan["old_name"],
                        "new_name": plan["new_name"],
                        "old_stage": plan["old_stage"],
                        "source_ids": plan["source_ids"],
                    }
                    for plan in drained
                ],
            )
            or []
        )
        expected_results = {(plan["old_name"], plan["new_name"]) for plan in drained}
        actual_results = {(row.get("old_name"), row.get("new_name")) for row in rows}
        if actual_results != expected_results or len(rows) != len(drained):
            raise RuntimeError(
                "generated-name supersession state changed before lifecycle "
                f"finalization: expected {len(drained)}, finalized {len(rows)}"
            )
        for row in rows:
            foundation.services.logger.info(
                "supersede_prior_source_names: %s superseded by %s (same source)",
                row.get("old_name"),
                row.get("new_name"),
            )
        return len(rows)

    if isinstance(gc, foundation.services._TransactionQuery):
        return _apply(gc)

    with (
        foundation.services.nullcontext(gc)
        if gc is not None
        else foundation.services.GraphClient() as graph
    ):
        with graph.session() as session:
            tx = session.begin_transaction()
            try:
                superseded = _apply(foundation.services._TransactionQuery(tx))
                tx.commit()
                return superseded
            except BaseException:
                with foundation.services.suppress(Exception):
                    tx.rollback()
                raise


@retry_on_deadlock()
def tombstone_supersede_into(
    old_id: str,
    into_id: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Supersede ``old_id`` INTO an already-existing accepted name ``into_id``.

    Generalises the one-off pair-supersede into a supported operation. Two
    honest repairs are otherwise blocked because ``sn edit --rename`` refuses a
    rename onto an existing id and :func:`supersede_prior_source_names` keys on
    a shared source:

    * folding a name into an already-existing canonical name;
    * re-pointing a name back onto a tombstoned id (after that id is restored
      to accepted).

    Mirrors the supersede semantics: stamps ``old.name_stage='superseded'``,
    ``old.superseded_from_stage = coalesce(old.superseded_from_stage,
    'accepted')`` (so catalog export emits a ``status: deprecated`` stub for it),
    clears its claim, and MERGEs ``(into)-[:REFINED_FROM]->(old)`` so the stub
    resolves to the live successor. Immutable source provenance remains on the
    source node; its live produced identity moves exclusively to ``into``.

    **The fold also carries ``old``'s sources onto ``into``.** A fold asserts
    that the two names denote one quantity, so every source realizing ``old``
    realizes ``into``; without the migration a source whose only name was
    ``old`` is left with no live name at all — silently un-named, and rewound to
    paid re-composition by the next drain. The explicit source cohort and its
    expected predecessor binding are guarded and migrated in the same
    transaction as the lifecycle write.

    Refuses (returns ``{"ok": False, "reason": ...}`` without writing) when:

    * ``old_id == into_id`` (nothing to fold);
    * ``old_id`` does not exist;
    * ``into_id`` does not exist;
    * ``into_id`` is neither ``name_stage='accepted'`` nor ``'approved'`` —
      the successor must be a live canonical name so the emitted deprecation
      stub points somewhere real (an unresolvable stub would be a dangling
      breaking-change pointer);
    * folding would create a ``REFINED_FROM`` cycle (``old`` already descends
      from ``into`` — threading ``into``→``old`` would close a loop).

    Idempotent: a second call re-stamps the same values and the MERGE is a
    no-op, so re-running reports ``ok`` with ``already_superseded=True``.
    """
    if old_id == into_id:
        return {"ok": False, "reason": "old and target are the same name"}

    with foundation.services.GraphClient() as gc:
        with gc.session() as session:
            transaction = session.begin_transaction()
            query_handle = foundation.services._TransactionQuery(transaction)
            try:
                rows = query_handle.query(
                    """
            OPTIONAL MATCH (old:StandardName {id: $old_id})
            OPTIONAL MATCH (into:StandardName {id: $into_id})
            RETURN old.id AS old_id,
                   old.name_stage AS old_stage,
                   old.superseded_from_stage AS old_sfs,
                   into.id AS into_id,
                   into.name_stage AS into_stage,
                   EXISTS { MATCH (old)-[:REFINED_FROM*1..]->(into) } AS cycle
            """,
                    old_id=old_id,
                    into_id=into_id,
                )
                row = rows[0] if rows else {}
                refusal: str | None = None
                if not row.get("old_id"):
                    refusal = f"name {old_id!r} not found"
                elif not row.get("into_id"):
                    refusal = f"target {into_id!r} not found"
                elif row.get("into_stage") not in ("accepted", "approved"):
                    refusal = (
                        f"target {into_id!r} is name_stage="
                        f"{row.get('into_stage')!r}, not 'accepted' or 'approved' — "
                        "supersede target must be the live canonical name"
                    )
                elif row.get("cycle"):
                    refusal = (
                        f"{old_id!r} already descends from {into_id!r} "
                        "(REFINED_FROM cycle) — cannot fold"
                    )
                if refusal:
                    transaction.rollback()
                    return {"ok": False, "reason": refusal}

                already = row.get("old_stage") == "superseded"
                carry_rows = query_handle.query(
                    """
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(old:StandardName {id: $old_id})
            OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(other:StandardName)
            WHERE other.id <> $old_id
              AND NOT coalesce(other.name_stage, '') IN ['superseded', 'exhausted']
            WITH src, count(other) AS live_elsewhere
            RETURN collect(src.id) AS source_ids,
                   count(src) AS sources,
                   sum(CASE WHEN live_elsewhere = 0 THEN 1 ELSE 0 END) AS would_strand
            """,
                    old_id=old_id,
                )
                carry = carry_rows[0] if carry_rows else {}
                source_ids = sorted(set(carry.get("source_ids") or []))
                result = {
                    "ok": True,
                    "old_id": old_id,
                    "into_id": into_id,
                    "old_prior_stage": row.get("old_stage"),
                    "already_superseded": already,
                    "sources_carried": len(source_ids),
                    "sources_would_strand": int(carry.get("would_strand") or 0),
                    "dry_run": dry_run,
                }
                if dry_run:
                    transaction.rollback()
                    return result

                if source_ids:
                    from imas_codex.standard_names.provenance_lifecycle import (
                        retarget_standard_name_sources,
                    )

                    retarget_standard_name_sources(
                        query_handle,
                        old_id,
                        into_id,
                        operation="fold",
                        source_ids=source_ids,
                        expected_current_bindings=dict.fromkeys(source_ids, old_id),
                    )

                finalized = query_handle.query(
                    """
            MATCH (old:StandardName {id: $old_id}),
                  (into:StandardName {id: $into_id})
            WHERE into.name_stage IN ['accepted', 'approved']
              AND NOT EXISTS { MATCH (old)-[:REFINED_FROM*1..]->(into) }
            SET old.updated_at = datetime(), old.name_stage = 'superseded',
                old.superseded_from_stage =
                    coalesce(old.superseded_from_stage, 'accepted'),
                old.claim_token = null,
                old.claimed_at = null,
                // Close any open edit on the folded predecessor. Once folded
                // into a live canonical name it is terminal and unreviewable;
                // a lingering 'open' edit would be orphaned forever.
                old.edit_status = CASE
                    WHEN coalesce(old.edit_status, '') = 'open'
                    THEN 'applied' ELSE old.edit_status END
            MERGE (into)-[:REFINED_FROM]->(old)
            RETURN old.id AS old_id, into.id AS into_id
            """,
                    old_id=old_id,
                    into_id=into_id,
                )
                if len(finalized) != 1:
                    raise RuntimeError("fold lifecycle compare-and-set changed")
                transaction.commit()
            except BaseException:
                with foundation.services.suppress(Exception):
                    transaction.rollback()
                raise

    result["attachments_rejected"] = 0
    result["attachments_detached"] = 0
    foundation.services.logger.info(
        "tombstone_supersede_into: %s superseded into %s (sfs=accepted, "
        "REFINED_FROM lineage merged)",
        old_id,
        into_id,
    )
    return result


SCALAR_SELECTED_DEDUP_MANIFEST_SCHEMA = (
    "imas-codex.standard-name-source-dedup-manifest.v1"
)


SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA = (
    "imas-codex.standard-name-source-dedup-receipt.v1"
)


class ScalarSelectedDedupConflict(RuntimeError):
    """The signed source-binding closure no longer matches graph authority."""


def _scalar_selected_dedup_authority(
    query_handle: _TransactionQuery,
    source_ids: list[str],
    reason: str,
    excluded_source_reasons: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = query_handle.query(
        """
        UNWIND $source_ids AS requested_id
        OPTIONAL MATCH (source:StandardNameSource {id: requested_id})
        RETURN requested_id AS requested_id,
               properties(source) AS source_properties,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[binding:PRODUCED_NAME]->(target:StandardName) |
                   {element_id: elementId(binding),
                    properties: properties(binding),
                    target_element_id: elementId(target),
                    target_id: target.id,
                    target_properties: properties(target)}]
               END AS bindings,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[origin:FROM_DD_PATH|FROM_SIGNAL]->(backing) |
                   {element_id: elementId(backing),
                    labels: labels(backing),
                    properties: properties(backing),
                    origin_element_id: elementId(origin),
                    origin_type: type(origin),
                    origin_properties: properties(origin),
                    projections: [(backing)-[projection:HAS_STANDARD_NAME]->
                      (projected:StandardName) |
                      {element_id: elementId(projection),
                       properties: properties(projection),
                       target_id: projected.id,
                       target_properties: properties(projected)}]}]
               END AS backings
        ORDER BY requested_id
        """,
        source_ids=source_ids,
    )
    participants: list[dict[str, foundation.services.Any]] = []
    actions: list[dict[str, foundation.services.Any]] = []
    refusals: list[dict[str, foundation.services.Any]] = []
    for raw in rows:
        row = dict(raw)
        bindings = sorted(
            (dict(binding) for binding in row.get("bindings") or []),
            key=lambda binding: (binding["target_id"], binding["element_id"]),
        )
        backings = sorted(
            (
                {
                    **dict(backing),
                    "labels": sorted(backing.get("labels") or []),
                    "projections": sorted(
                        (
                            dict(projection)
                            for projection in backing.get("projections") or []
                        ),
                        key=lambda projection: (
                            projection["target_id"],
                            projection["element_id"],
                        ),
                    ),
                }
                for backing in row.get("backings") or []
            ),
            key=lambda backing: backing["element_id"],
        )
        participant = {
            "source_id": row["requested_id"],
            "source_properties": row.get("source_properties"),
            "bindings": bindings,
            "backings": backings,
        }
        participants.append(participant)
        properties = row.get("source_properties")
        refusal: str | None = None
        if properties is None:
            refusal = "source does not exist"
        elif (
            properties.get("claimed_at") is not None
            or properties.get("claim_token") is not None
        ):
            refusal = "source has an active claim"
        elif properties.get("status") == "stale":
            refusal = "stale source is not deduplication authority"

        live_bindings = [
            binding
            for binding in bindings
            if (binding.get("target_properties") or {}).get("name_stage")
            not in ("superseded", "exhausted")
        ]
        live_target_ids = [binding["target_id"] for binding in live_bindings]
        scalar = properties.get("produced_sn_id") if properties else None
        if refusal is None and (
            len(live_bindings) < 2
            or len(set(live_target_ids)) != len(live_target_ids)
            or live_target_ids.count(scalar) != 1
        ):
            refusal = "produced_sn_id does not select exactly one live binding"

        remove_bindings = [
            binding for binding in live_bindings if binding["target_id"] != scalar
        ]
        projection_by_target: dict[str, list[dict[str, foundation.services.Any]]] = {}
        for backing in backings:
            for projection in backing["projections"]:
                projection_by_target.setdefault(projection["target_id"], []).append(
                    {
                        **projection,
                        "backing_element_id": backing["element_id"],
                    }
                )
        remove_projections: list[dict[str, foundation.services.Any]] = []
        if refusal is None:
            for binding in remove_bindings:
                matches = projection_by_target.get(binding["target_id"], [])
                if not matches:
                    refusal = "non-selected binding has no signed backing projection"
                    break
                if len(matches) > 1:
                    refusal = (
                        "non-selected binding has duplicate signed backing projections"
                    )
                    break
                remove_projections.append(matches[0])

        exclusion_reason = excluded_source_reasons.get(row["requested_id"])
        if exclusion_reason is not None:
            if refusal == "non-selected binding has no signed backing projection":
                refusals.append(
                    {
                        "source_id": row["requested_id"],
                        "reason": exclusion_reason,
                        "classification": "explicit_exclusion",
                        "authority_reason": refusal,
                    }
                )
            else:
                refusals.append(
                    {
                        "source_id": row["requested_id"],
                        "reason": (
                            refusal
                            or "explicit exclusion no longer matches a zero-projection row"
                        ),
                    }
                )
            continue
        if refusal is not None:
            refusals.append({"source_id": row["requested_id"], "reason": refusal})
            continue
        removals = sorted(
            (
                {
                    "target_id": binding["target_id"],
                    "binding_element_id": binding["element_id"],
                    "projection_element_id": projection["element_id"],
                    "backing_element_id": projection["backing_element_id"],
                }
                for binding, projection in zip(
                    remove_bindings, remove_projections, strict=True
                )
            ),
            key=lambda removal: (
                removal["target_id"],
                removal["binding_element_id"],
            ),
        )
        actions.append(
            {
                "source_id": row["requested_id"],
                "keep_target_id": scalar,
                "removals": removals,
            }
        )

    manifest = {
        "schema": foundation.services.SCALAR_SELECTED_DEDUP_MANIFEST_SCHEMA,
        "operation": "deduplicate_scalar_selected_sources",
        "reason": reason,
        "source_ids": source_ids,
        "excluded_source_reasons": excluded_source_reasons,
        "participants": participants,
        "actions": actions,
        "refusals": refusals,
    }
    return manifest, actions, refusals


def _lock_scalar_selected_dedup_authority(
    query_handle: _TransactionQuery, manifest: dict[str, Any]
) -> None:
    participants = manifest["participants"]
    source_ids = [row["source_id"] for row in participants]
    node_element_ids = sorted(
        {
            element_id
            for row in participants
            for element_id in [
                *(backing["element_id"] for backing in row["backings"]),
                *(binding["target_element_id"] for binding in row["bindings"]),
            ]
        }
    )
    relationship_element_ids = sorted(
        {
            element_id
            for row in participants
            for element_id in [
                *(binding["element_id"] for binding in row["bindings"]),
                *(backing["origin_element_id"] for backing in row["backings"]),
                *(
                    projection["element_id"]
                    for backing in row["backings"]
                    for projection in backing["projections"]
                ),
            ]
        }
    )
    locked_sources = query_handle.query(
        """
        UNWIND $source_ids AS source_id
        MATCH (source:StandardNameSource {id: source_id})
        SET source.produced_sn_id = source.produced_sn_id
        RETURN collect(source.id) AS ids
        """,
        source_ids=source_ids,
    )
    actual_sources = (
        sorted(locked_sources[0].get("ids") or []) if locked_sources else []
    )
    if actual_sources != source_ids:
        raise foundation.services.ScalarSelectedDedupConflict(
            "signed source set changed while locking"
        )
    if node_element_ids:
        locked_nodes = query_handle.query(
            """
            UNWIND $element_ids AS expected
            MATCH (node) WHERE elementId(node) = expected
            SET node += {}
            RETURN collect(elementId(node)) AS ids
            """,
            element_ids=node_element_ids,
        )
        actual_nodes = sorted(locked_nodes[0].get("ids") or []) if locked_nodes else []
        if actual_nodes != node_element_ids:
            raise foundation.services.ScalarSelectedDedupConflict(
                "signed backing set changed while locking"
            )
    if relationship_element_ids:
        locked_relationships = query_handle.query(
            """
            UNWIND $element_ids AS expected
            MATCH ()-[relationship]->() WHERE elementId(relationship) = expected
            SET relationship += {}
            RETURN collect(elementId(relationship)) AS ids
            """,
            element_ids=relationship_element_ids,
        )
        actual_relationships = (
            sorted(locked_relationships[0].get("ids") or [])
            if locked_relationships
            else []
        )
        if actual_relationships != relationship_element_ids:
            raise foundation.services.ScalarSelectedDedupConflict(
                "signed relationship set changed while locking"
            )


def _scalar_selected_dedup_counts(
    source_ids: list[str],
    actions: list[dict[str, Any]],
    refusals: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "requested": len(source_ids),
        "admitted": len(actions),
        "refused": len(refusals),
        "bindings_to_remove": sum(len(action["removals"]) for action in actions),
        "projections_to_remove": sum(len(action["removals"]) for action in actions),
    }


@retry_on_deadlock()
def deduplicate_scalar_selected_sources(
    source_ids: list[str],
    *,
    reason: str,
    excluded_source_reasons: dict[str, str] | None = None,
    apply: bool = False,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Remove non-scalar live bindings through a signed exact manifest."""
    requested = sorted(set(source_ids))
    if not requested:
        raise ValueError("source deduplication requires at least one source id")
    if len(requested) != len(source_ids):
        raise ValueError("source deduplication requires unique source ids")
    if not reason.strip():
        raise ValueError("source deduplication requires a non-empty reason")
    exclusions = dict(sorted((excluded_source_reasons or {}).items()))
    if not set(exclusions).issubset(requested):
        raise ValueError("excluded source ids must belong to the requested cohort")
    if any(not exclusion_reason.strip() for exclusion_reason in exclusions.values()):
        raise ValueError("every excluded source requires a non-empty reason")
    if apply and manifest_sha256 is None:
        raise ValueError("apply requires manifest_sha256")
    if manifest_sha256 is not None and not foundation.services._SHA256_RE.fullmatch(
        manifest_sha256
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")

    own = gc is None
    client: foundation.services.Any = foundation.services.GraphClient() if own else gc
    event_id = (
        f"sn-change:scalar-selected-dedup:{manifest_sha256}"
        if manifest_sha256
        else None
    )
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            query_handle = foundation.services._TransactionQuery(transaction)
            try:
                if apply:
                    replay_rows = query_handle.query(
                        """
                        OPTIONAL MATCH (change:StandardNameChange {id: $event_id})
                        RETURN change.to_name AS disposition_json
                        """,
                        event_id=event_id,
                    )
                    disposition_json = (
                        replay_rows[0].get("disposition_json") if replay_rows else None
                    )
                    if disposition_json is not None:
                        dispositions = foundation.services.json.loads(disposition_json)
                        if sorted(dispositions) != requested:
                            raise foundation.services.ScalarSelectedDedupConflict(
                                "recorded deduplication covers a different source set"
                            )
                        admitted_ids = sorted(
                            source_id
                            for source_id, target_id in dispositions.items()
                            if target_id is not None
                        )
                        postcondition = query_handle.query(
                            """
                            UNWIND $source_ids AS source_id
                            MATCH (source:StandardNameSource {id: source_id})
                            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(name:StandardName)
                            WHERE NOT coalesce(name.name_stage, '') IN
                              ['superseded', 'exhausted']
                            WITH source_id, source, collect(DISTINCT name.id) AS ids
                            RETURN collect({source_id: source_id,
                              scalar: source.produced_sn_id, ids: ids}) AS rows
                            """,
                            source_ids=admitted_ids,
                        )
                        current = postcondition[0].get("rows") if postcondition else []
                        if any(
                            row["ids"] != [row["scalar"]]
                            or dispositions.get(row["source_id"]) != row["scalar"]
                            for row in current
                        ) or len(current) != len(admitted_ids):
                            raise foundation.services.ScalarSelectedDedupConflict(
                                "recorded deduplication has lost its postcondition"
                            )
                        excluded_ids = sorted(set(requested) - set(admitted_ids))
                        if excluded_ids:
                            if excluded_ids != sorted(exclusions):
                                raise foundation.services.ScalarSelectedDedupConflict(
                                    "replay requires the signed exclusion reasons"
                                )
                            (
                                _,
                                excluded_actions,
                                excluded_refusals,
                            ) = foundation.services._scalar_selected_dedup_authority(
                                query_handle,
                                excluded_ids,
                                reason,
                                {
                                    source_id: exclusions[source_id]
                                    for source_id in excluded_ids
                                },
                            )
                            if excluded_actions or any(
                                refusal.get("classification") != "explicit_exclusion"
                                for refusal in excluded_refusals
                            ):
                                raise foundation.services.ScalarSelectedDedupConflict(
                                    "recorded exclusions have lost their authority"
                                )
                        transaction.rollback()
                        return {
                            "schema": foundation.services.SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA,
                            "outcome": "already_applied",
                            "dry_run": False,
                            "changed": 0,
                            "manifest_sha256": manifest_sha256,
                        }

                manifest, actions, refusals = (
                    foundation.services._scalar_selected_dedup_authority(
                        query_handle, requested, reason, exclusions
                    )
                )
                computed_hash = foundation.services._authority_payload_hash(manifest)
                counts = foundation.services._scalar_selected_dedup_counts(
                    requested, actions, refusals
                )
                if apply and computed_hash != manifest_sha256:
                    raise foundation.services.ScalarSelectedDedupConflict(
                        "fresh source deduplication manifest does not match signed hash"
                    )
                blocking_refusals = [
                    refusal
                    for refusal in refusals
                    if refusal.get("classification") != "explicit_exclusion"
                ]
                if blocking_refusals:
                    transaction.rollback()
                    return {
                        "schema": foundation.services.SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA,
                        "outcome": "refused",
                        "dry_run": not apply,
                        "changed": 0,
                        "would_change": 0,
                        "counts": counts,
                        "refusals": refusals,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }
                if not apply:
                    transaction.rollback()
                    return {
                        "schema": foundation.services.SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA,
                        "outcome": "would_apply",
                        "dry_run": True,
                        "changed": 0,
                        "would_change": 1,
                        "counts": counts,
                        "refusals": refusals,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }

                foundation.services._lock_scalar_selected_dedup_authority(
                    query_handle, manifest
                )
                locked_manifest, locked_actions, locked_refusals = (
                    foundation.services._scalar_selected_dedup_authority(
                        query_handle, requested, reason, exclusions
                    )
                )
                if (
                    any(
                        refusal.get("classification") != "explicit_exclusion"
                        for refusal in locked_refusals
                    )
                    or foundation.services._authority_payload_hash(locked_manifest)
                    != computed_hash
                ):
                    raise foundation.services.ScalarSelectedDedupConflict(
                        "source authority changed while acquiring participant locks"
                    )
                actions = locked_actions
                mutation_rows = [
                    {
                        "source_id": action["source_id"],
                        "keep_target_id": action["keep_target_id"],
                        "remove_target_id": removal["target_id"],
                        "binding_element_id": removal["binding_element_id"],
                        "projection_element_id": removal["projection_element_id"],
                        "backing_element_id": removal["backing_element_id"],
                    }
                    for action in actions
                    for removal in action["removals"]
                ]
                mutated = query_handle.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (source:StandardNameSource {id: expected.source_id})
                    MATCH (source)-[binding:PRODUCED_NAME]->
                          (removed:StandardName {id: expected.remove_target_id})
                    WHERE elementId(binding) = expected.binding_element_id
                      AND source.produced_sn_id = expected.keep_target_id
                      AND source.claimed_at IS NULL
                      AND source.claim_token IS NULL
                      AND EXISTS {
                        (source)-[:PRODUCED_NAME]->
                          (:StandardName {id: expected.keep_target_id})
                      }
                    MATCH (backing)-[projection:HAS_STANDARD_NAME]->(removed)
                    WHERE elementId(backing) = expected.backing_element_id
                      AND elementId(projection) = expected.projection_element_id
                    DELETE binding, projection
                    RETURN collect({source_id: expected.source_id,
                      binding_element_id: expected.binding_element_id,
                      projection_element_id: expected.projection_element_id}) AS rows
                    """,
                    rows=mutation_rows,
                )
                actual_mutations = mutated[0].get("rows") if mutated else []
                if len(actual_mutations) != len(mutation_rows):
                    raise foundation.services.ScalarSelectedDedupConflict(
                        "signed binding closure changed during deletion"
                    )

                target_ids = sorted(
                    {
                        target_id
                        for action in actions
                        for target_id in [
                            action["keep_target_id"],
                            *(removal["target_id"] for removal in action["removals"]),
                        ]
                    }
                )
                query_handle.query(
                    """
                    UNWIND $target_ids AS target_id
                    MATCH (target:StandardName {id: target_id})
                    OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(target)
                    OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
                    OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
                    WITH target, collect(DISTINCT CASE
                      WHEN source IS NULL THEN null
                      WHEN dd IS NOT NULL THEN 'dd:' + dd.id
                      WHEN signal IS NOT NULL THEN signal.id
                      WHEN source.source_type = 'derived'
                        AND source.source_id STARTS WITH 'derived:'
                      THEN source.source_id ELSE source.id END) AS paths
                    SET target.updated_at = datetime(), target.source_paths = [path IN paths WHERE path IS NOT NULL]
                    """,
                    target_ids=target_ids,
                )
                dispositions = {
                    action["source_id"]: action["keep_target_id"] for action in actions
                }
                dispositions.update(dict.fromkeys(exclusions))
                change_rows = query_handle.query(
                    """
                    MERGE (change:StandardNameChange {id: $event_id})
                    ON CREATE SET change.from_name = $removed_json,
                                  change.to_name = $disposition_json,
                                  change.operation =
                                    'deduplicate_scalar_selected_sources',
                                  change.reason = $reason,
                                  change.origin = 'semantic_source_reconciliation',
                                  change.run_id = $run_id,
                                  change.changed_at = datetime(),
                                  change.internal = true,
                                  change.manifest_sha256 = $manifest_sha256
                    WITH change
                    UNWIND $keep_target_ids AS keep_target_id
                    MATCH (kept:StandardName {id: keep_target_id})
                    MERGE (kept)-[:HAS_INTERNAL_CHANGE]->(change)
                    RETURN DISTINCT change.id AS change_id
                    """,
                    event_id=f"sn-change:scalar-selected-dedup:{computed_hash}",
                    removed_json=foundation.services.json.dumps(
                        {
                            action["source_id"]: [
                                removal["target_id"] for removal in action["removals"]
                            ]
                            for action in actions
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    disposition_json=foundation.services.json.dumps(
                        dispositions, sort_keys=True, separators=(",", ":")
                    ),
                    reason=reason,
                    run_id=run_id,
                    manifest_sha256=computed_hash,
                    keep_target_ids=sorted(
                        target_id
                        for target_id in set(dispositions.values())
                        if target_id is not None
                    ),
                )
                if len(change_rows) != 1:
                    raise foundation.services.ScalarSelectedDedupConflict(
                        "deduplication change receipt was not written exactly once"
                    )
                transaction.commit()
                return {
                    "schema": foundation.services.SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA,
                    "outcome": "applied",
                    "dry_run": False,
                    "changed": 1,
                    "sources_deduplicated": len(actions),
                    "bindings_removed": len(mutation_rows),
                    "projections_removed": len(mutation_rows),
                    "counts": counts,
                    "refusals": refusals,
                    "manifest": manifest,
                    "manifest_sha256": computed_hash,
                    "change_id": change_rows[0]["change_id"],
                }
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own:
            client.close()


SIGNED_SOURCE_DISPOSITION_ADJUDICATION_SCHEMA = (
    "imas-codex.catalog-edit-dual-binding-adjudication.v1"
)


SIGNED_SOURCE_DISPOSITION_MANIFEST_SCHEMA = (
    "imas-codex.signed-source-disposition-manifest.v2"
)


SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA = (
    "imas-codex.signed-source-disposition-receipt.v1"
)


_SIGNED_SOURCE_DISPOSITIONS = frozenset(
    {
        "retain_scalar_target",
        "retarget_scalar_target",
        "select_missing_scalar",
    }
)


_STRUCTURAL_LEGITIMACY_AUTHORITY_SCHEMA = (
    "imas-codex.refused-target-orphan-adjudication.v2"
)


_STRUCTURAL_LEGITIMACY_DISPOSITION = "preserve_as_structural_identity"


_ORPHAN_DISPOSITIONS = frozenset(
    {
        _STRUCTURAL_LEGITIMACY_DISPOSITION,
        "re_source_from_existing_dd_path",
        "retain_competing_binding",
        "retire_under_orphan_policy",
    }
)


class SignedSourceDispositionConflict(RuntimeError):
    """The adjudicated source closure no longer matches graph authority."""


def _catalog_edit_adjudication_signature_hash(value: Any) -> str:
    # ASCII-escaped JSON is required because signed rows carry non-ASCII characters.
    payload = foundation.services.json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return foundation.services.hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_signed_source_adjudication(
    adjudication: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    if (
        adjudication.get("schema")
        != foundation.services.SIGNED_SOURCE_DISPOSITION_ADJUDICATION_SCHEMA
    ):
        raise ValueError("unsupported source-disposition adjudication schema")
    signature = adjudication.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "sha256":
        raise ValueError("source-disposition adjudication requires a SHA-256 signature")
    declared_hash = signature.get("payload_sha256")
    if not isinstance(
        declared_hash, str
    ) or not foundation.services._SHA256_RE.fullmatch(declared_hash):
        raise ValueError(
            "adjudication payload_sha256 must be a lowercase SHA-256 digest"
        )
    payload = {key: value for key, value in adjudication.items() if key != "signature"}
    if (
        foundation.services._catalog_edit_adjudication_signature_hash(payload)
        != declared_hash
    ):
        raise ValueError("source-disposition adjudication signature does not match")
    raw_rows = adjudication.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("source-disposition adjudication requires at least one row")
    rows: list[dict[str, foundation.services.Any]] = []
    seen_sources: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("every source disposition must be an object")
        row = dict(raw_row)
        row_signature = row.get("row_signature_sha256")
        if not isinstance(
            row_signature, str
        ) or not foundation.services._SHA256_RE.fullmatch(row_signature):
            raise ValueError("every source disposition requires a row SHA-256")
        unsigned_row = {
            key: value for key, value in row.items() if key != "row_signature_sha256"
        }
        if (
            foundation.services._catalog_edit_adjudication_signature_hash(unsigned_row)
            != row_signature
        ):
            raise ValueError("source disposition row signature does not match")
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.startswith("dd:"):
            raise ValueError("source dispositions require exact DD source ids")
        if source_id in seen_sources:
            raise ValueError("source disposition ids must be unique")
        seen_sources.add(source_id)
        disposition = row.get("disposition")
        if disposition not in foundation.services._SIGNED_SOURCE_DISPOSITIONS:
            raise ValueError("source disposition is outside the closed set")
        candidates = row.get("candidate_live_targets")
        removed = row.get("removed_targets")
        survivor = row.get("surviving_target")
        prior_scalar = row.get("prior_scalar_target")
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(
                not isinstance(candidate, str) or not candidate
                for candidate in candidates
            )
            or len(candidates) != len(set(candidates))
            or candidates != sorted(candidates)
        ):
            raise ValueError("candidate live targets must be unique and sorted")
        if not isinstance(survivor, str) or survivor not in candidates:
            raise ValueError("surviving target must belong to the candidate set")
        expected_removed = sorted(set(candidates) - {survivor})
        if removed != expected_removed or not expected_removed:
            raise ValueError(
                "removed targets must be the exact non-surviving candidates"
            )
        if disposition == "retain_scalar_target" and prior_scalar != survivor:
            raise ValueError("retain disposition must preserve the prior scalar")
        if disposition == "retarget_scalar_target" and (
            prior_scalar is None or prior_scalar == survivor
        ):
            raise ValueError("retarget disposition must replace a different scalar")
        if disposition == "select_missing_scalar" and prior_scalar is not None:
            raise ValueError("missing-scalar disposition requires a null prior scalar")
        rows.append(row)
    rows = sorted(rows, key=lambda row: row["source_id"])
    summary = adjudication.get("summary")
    if summary is not None:
        disposition_counts = {
            disposition: sum(row["disposition"] == disposition for row in rows)
            for disposition in sorted(foundation.services._SIGNED_SOURCE_DISPOSITIONS)
        }
        if (
            not isinstance(summary, dict)
            or summary.get("adjudicated_rows") != len(rows)
            or summary.get("unique_source_rows") != len(rows)
            or summary.get("single_survivor_rows") != len(rows)
            or summary.get("disposition_sum") != len(rows)
            or summary.get("disposition_counts") != disposition_counts
        ):
            raise ValueError("source-disposition summary does not match signed rows")
    row_set_sha256 = foundation.services._catalog_edit_adjudication_signature_hash(
        {
            "schema": adjudication["schema"],
            "rows": rows,
        }
    )
    return declared_hash, row_set_sha256, rows


def _validate_structural_legitimacy_authority(
    authority: dict[str, Any] | None,
    authority_sha256: str | None,
    adjudication_rows: list[dict[str, Any]],
) -> tuple[str | None, frozenset[str]]:
    if authority is None and authority_sha256 is None:
        return None, frozenset()
    if authority is None or authority_sha256 is None:
        raise ValueError(
            "structural legitimacy authority and its SHA-256 are both required"
        )
    if not foundation.services._SHA256_RE.fullmatch(authority_sha256):
        raise ValueError(
            "structural legitimacy authority SHA-256 must be a lowercase digest"
        )
    if (
        foundation.services._catalog_edit_adjudication_signature_hash(authority)
        != authority_sha256
    ):
        raise ValueError("structural legitimacy authority signature does not match")
    if (
        authority.get("schema")
        != foundation.services._STRUCTURAL_LEGITIMACY_AUTHORITY_SCHEMA
    ):
        raise ValueError("unsupported structural legitimacy authority schema")
    if authority.get("read_only") is not True:
        raise ValueError("structural legitimacy authority must be read-only evidence")

    raw_rows = authority.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("structural legitimacy authority requires target rows")
    adjudication_by_source = {row["source_id"]: row for row in adjudication_rows}
    disposition_counts = {
        disposition: sum(
            isinstance(row, dict) and row.get("disposition") == disposition
            for row in raw_rows
        )
        for disposition in sorted(foundation.services._ORPHAN_DISPOSITIONS)
    }
    if any(
        not isinstance(row, dict)
        or row.get("disposition") not in foundation.services._ORPHAN_DISPOSITIONS
        for row in raw_rows
    ):
        raise ValueError("structural legitimacy authority has an unknown disposition")
    summary = authority.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("targets") != len(raw_rows)
        or summary.get("disposition_sum") != len(raw_rows)
        or summary.get("disposition_counts") != disposition_counts
    ):
        raise ValueError("structural legitimacy authority summary does not match rows")

    structural_targets: set[str] = set()
    for row in raw_rows:
        if row["disposition"] != foundation.services._STRUCTURAL_LEGITIMACY_DISPOSITION:
            continue
        target_id = row.get("name")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("structural legitimacy target requires an exact name")
        if target_id in structural_targets:
            raise ValueError("structural legitimacy targets must be unique")
        if row.get("mutation_authority") != "classification_only":
            raise ValueError(
                "structural legitimacy evidence cannot grant lifecycle authority"
            )
        closure = row.get("structural_closure")
        signed_children = (
            closure.get("live_has_parent_children")
            if isinstance(closure, dict)
            else None
        )
        if (
            not isinstance(closure, dict)
            or closure.get("classification")
            != "structurally_legitimate_without_producing_source"
            or closure.get("has_live_has_parent_child") is not True
            or not isinstance(signed_children, list)
            or not signed_children
            or any(
                not isinstance(child, dict)
                or not isinstance(child.get("name"), str)
                or not child.get("name")
                or not isinstance(child.get("relationship_id"), str)
                or not child.get("relationship_id")
                for child in signed_children
            )
        ):
            raise ValueError(
                "structural legitimacy target requires a signed direct child"
            )
        removed_bindings = row.get("current_removed_bindings")
        if not isinstance(removed_bindings, list) or not removed_bindings:
            raise ValueError(
                "structural legitimacy target requires signed removed bindings"
            )
        for binding in removed_bindings:
            if not isinstance(binding, dict):
                raise ValueError("structural removed binding must be an object")
            signed_row = adjudication_by_source.get(binding.get("source_id"))
            if (
                signed_row is None
                or binding.get("signed_row_sha256")
                != signed_row["row_signature_sha256"]
                or binding.get("signed_surviving_target")
                != signed_row["surviving_target"]
                or target_id not in signed_row["removed_targets"]
            ):
                raise ValueError(
                    "structural target is not bound to its signed source disposition"
                )
        structural_targets.add(target_id)
    return authority_sha256, frozenset(structural_targets)


def _is_live_structural_child(child: dict[str, Any]) -> bool:
    properties = child.get("child_properties") or {}
    return properties.get("name_stage") != "superseded" and properties.get(
        "status"
    ) not in {"deprecated", "superseded"}


def _signed_source_disposition_authority(
    query_handle: _TransactionQuery,
    rows: list[dict[str, Any]],
    adjudication_sha256: str,
    adjudication_row_set_sha256: str,
    reason: str,
    structural_authority_sha256: str | None,
    structural_target_ids: frozenset[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_ids = [row["source_id"] for row in rows]
    removed_target_ids = sorted(
        {target_id for row in rows for target_id in row["removed_targets"]}
    )
    graph_rows = query_handle.query(
        """
        UNWIND $source_ids AS requested_id
        OPTIONAL MATCH (source:StandardNameSource {id: requested_id})
        RETURN requested_id,
               elementId(source) AS source_element_id,
               properties(source) AS source_properties,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[binding:PRODUCED_NAME]->(target:StandardName) |
                   {element_id: elementId(binding),
                    properties: properties(binding),
                    target_element_id: elementId(target),
                    target_id: target.id,
                    target_properties: properties(target)}]
               END AS bindings,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[origin:FROM_DD_PATH|FROM_SIGNAL]->(backing) |
                   {element_id: elementId(backing),
                    labels: labels(backing),
                    properties: properties(backing),
                    origin_element_id: elementId(origin),
                    origin_type: type(origin),
                    origin_properties: properties(origin),
                    projections: [(backing)-[projection:HAS_STANDARD_NAME]->
                      (projected:StandardName) |
                      {element_id: elementId(projection),
                       properties: properties(projection),
                       target_element_id: elementId(projected),
                       target_id: projected.id,
                       target_properties: properties(projected)}]}]
               END AS backings
        ORDER BY requested_id
        """,
        source_ids=source_ids,
    )
    removed_target_rows = query_handle.query(
        """
        UNWIND $target_ids AS requested_id
        OPTIONAL MATCH (target:StandardName {id: requested_id})
        RETURN requested_id,
               elementId(target) AS target_element_id,
               properties(target) AS target_properties,
               CASE WHEN target IS NULL THEN [] ELSE
                 [(source:StandardNameSource)
                    -[binding:PRODUCED_NAME]->(target) |
                   {binding_element_id: elementId(binding),
                    binding_properties: properties(binding),
                    source_element_id: elementId(source),
                    source_id: source.id,
                    source_properties: properties(source)}]
               END AS incoming_bindings
               ,CASE WHEN target IS NULL THEN [] ELSE
                 [(child:StandardName)-[relationship:HAS_PARENT]->(target) |
                   {relationship_element_id: elementId(relationship),
                    relationship_properties: properties(relationship),
                    child_element_id: elementId(child),
                    child_id: child.id,
                    child_properties: properties(child)}]
               END AS structural_children
        ORDER BY requested_id
        """,
        target_ids=removed_target_ids,
    )
    removed_target_closures = [
        {
            "target_id": raw["requested_id"],
            "target_element_id": raw.get("target_element_id"),
            "target_properties": raw.get("target_properties"),
            "incoming_bindings": sorted(
                (dict(binding) for binding in raw.get("incoming_bindings") or []),
                key=lambda binding: (
                    binding["source_id"],
                    binding["binding_element_id"],
                ),
            ),
            "structural_children": sorted(
                (dict(child) for child in raw.get("structural_children") or []),
                key=lambda child: (
                    child["child_id"],
                    child["relationship_element_id"],
                ),
            ),
        }
        for raw in removed_target_rows
    ]
    expected_by_source = {row["source_id"]: row for row in rows}
    participants: list[dict[str, foundation.services.Any]] = []
    actions: list[dict[str, foundation.services.Any]] = []
    refusals: list[dict[str, foundation.services.Any]] = []
    for raw in graph_rows:
        expected = expected_by_source[raw["requested_id"]]
        bindings = sorted(
            (dict(binding) for binding in raw.get("bindings") or []),
            key=lambda binding: (binding["target_id"], binding["element_id"]),
        )
        backings = sorted(
            (
                {
                    **dict(backing),
                    "labels": sorted(backing.get("labels") or []),
                    "projections": sorted(
                        (dict(item) for item in backing.get("projections") or []),
                        key=lambda item: (item["target_id"], item["element_id"]),
                    ),
                }
                for backing in raw.get("backings") or []
            ),
            key=lambda backing: backing["element_id"],
        )
        participant = {
            "source_id": expected["source_id"],
            "adjudication_row_sha256": expected["row_signature_sha256"],
            "source_element_id": raw.get("source_element_id"),
            "source_properties": raw.get("source_properties"),
            "bindings": bindings,
            "backings": backings,
        }
        participants.append(participant)
        properties = raw.get("source_properties")
        refusal: str | None = None
        if properties is None:
            refusal = "source does not exist"
        elif (
            properties.get("claimed_at") is not None
            or properties.get("claim_token") is not None
        ):
            refusal = "source has an active claim"
        elif properties.get("status") == "stale":
            refusal = "stale source is not disposition authority"
        elif properties.get("produced_sn_id") != expected["prior_scalar_target"]:
            refusal = "source scalar changed from the adjudicated value"

        live_bindings = [
            binding
            for binding in bindings
            if (binding.get("target_properties") or {}).get("name_stage")
            not in foundation.services._TERMINAL_BINDING_NAME_STAGES
        ]
        live_target_ids = [binding["target_id"] for binding in live_bindings]
        expected_targets = expected["candidate_live_targets"]
        if refusal is None and (
            live_target_ids != expected_targets
            or len(live_target_ids) != len(set(live_target_ids))
        ):
            refusal = "live binding set changed from the adjudicated candidates"
        if refusal is None and not any(
            (binding.get("target_properties") or {}).get("origin") == "catalog_edit"
            for binding in live_bindings
        ):
            refusal = "catalog-edit participant is no longer present"
        if refusal is None and (
            len(backings) != 1
            or backings[0]["origin_type"] != "FROM_DD_PATH"
            or backings[0]["properties"].get("id") != expected["source_id"][3:]
        ):
            refusal = "source does not have one exact DD backing"

        projection_by_target: dict[str, list[dict[str, foundation.services.Any]]] = {}
        for backing in backings:
            for projection in backing["projections"]:
                if (projection.get("target_properties") or {}).get(
                    "name_stage"
                ) not in foundation.services._TERMINAL_BINDING_NAME_STAGES:
                    projection_by_target.setdefault(projection["target_id"], []).append(
                        {**projection, "backing_element_id": backing["element_id"]}
                    )
        if refusal is None:
            projected_targets = sorted(
                target_id
                for target_id, projections in projection_by_target.items()
                for _ in projections
            )
            if projected_targets != expected_targets:
                refusal = "live backing projection set changed from adjudication"

        if refusal is not None:
            refusals.append({"source_id": expected["source_id"], "reason": refusal})
            continue
        binding_by_target = {item["target_id"]: item for item in live_bindings}
        removals = [
            {
                "target_id": target_id,
                "binding_element_id": binding_by_target[target_id]["element_id"],
                "projection_element_id": projection_by_target[target_id][0][
                    "element_id"
                ],
                "backing_element_id": projection_by_target[target_id][0][
                    "backing_element_id"
                ],
            }
            for target_id in expected["removed_targets"]
        ]
        actions.append(
            {
                "source_id": expected["source_id"],
                "source_element_id": raw["source_element_id"],
                "prior_scalar_target": expected["prior_scalar_target"],
                "keep_target_id": expected["surviving_target"],
                "disposition": expected["disposition"],
                "removals": removals,
            }
        )

    if structural_authority_sha256 is not None:
        structurally_authorized_actions: list[dict[str, foundation.services.Any]] = []
        for action in actions:
            unauthorized_targets = sorted(
                removal["target_id"]
                for removal in action["removals"]
                if removal["target_id"] not in structural_target_ids
            )
            if unauthorized_targets:
                refusals.append(
                    {
                        "source_id": action["source_id"],
                        "target_ids": unauthorized_targets,
                        "reason": (
                            "removed target is outside signed structural "
                            "legitimacy authority"
                        ),
                    }
                )
            else:
                structurally_authorized_actions.append(action)
        actions = structurally_authorized_actions

    scheduled_bindings_by_target: dict[str, set[str]] = {}
    for action in actions:
        for removal in action["removals"]:
            scheduled_bindings_by_target.setdefault(removal["target_id"], set()).add(
                removal["binding_element_id"]
            )
    structural_exemptions: list[dict[str, foundation.services.Any]] = []
    orphaning_targets: set[str] = set()
    for closure in removed_target_closures:
        target_id = closure["target_id"]
        scheduled_bindings = scheduled_bindings_by_target.get(target_id)
        if not scheduled_bindings:
            continue
        remaining_live_bindings = {
            binding["binding_element_id"]
            for binding in closure["incoming_bindings"]
            if (binding.get("source_properties") or {}).get("status") != "stale"
        } - scheduled_bindings
        if remaining_live_bindings:
            continue
        live_children = [
            child
            for child in closure["structural_children"]
            if foundation.services._is_live_structural_child(child)
        ]
        if target_id in structural_target_ids and live_children:
            structural_exemptions.append(
                {
                    "target_id": target_id,
                    "live_direct_children": live_children,
                }
            )
        else:
            orphaning_targets.add(target_id)
    if orphaning_targets:
        admitted_actions: list[dict[str, foundation.services.Any]] = []
        for action in actions:
            orphaned_targets = sorted(
                removal["target_id"]
                for removal in action["removals"]
                if removal["target_id"] in orphaning_targets
            )
            if orphaned_targets:
                refusals.append(
                    {
                        "source_id": action["source_id"],
                        "target_ids": orphaned_targets,
                        "reason": (
                            "removal would leave target with zero live producing sources"
                        ),
                    }
                )
            else:
                admitted_actions.append(action)
        actions = admitted_actions
    refusals = sorted(
        refusals,
        key=lambda refusal: (
            refusal["source_id"],
            refusal["reason"],
            refusal.get("target_ids") or [],
        ),
    )
    manifest = {
        "schema": foundation.services.SIGNED_SOURCE_DISPOSITION_MANIFEST_SCHEMA,
        "operation": "apply_adjudicated_source_dispositions",
        "reason": reason,
        "adjudication_sha256": adjudication_sha256,
        "adjudication_row_set_sha256": adjudication_row_set_sha256,
        "source_ids": source_ids,
        "participants": participants,
        "removed_target_closures": removed_target_closures,
        "structural_exemptions": sorted(
            structural_exemptions, key=lambda exemption: exemption["target_id"]
        ),
        "actions": actions,
        "refusals": refusals,
    }
    if structural_authority_sha256 is not None:
        manifest["structural_legitimacy_authority"] = {
            "schema": foundation.services._STRUCTURAL_LEGITIMACY_AUTHORITY_SCHEMA,
            "payload_sha256": structural_authority_sha256,
            "target_ids": sorted(structural_target_ids),
        }
    return manifest, actions, refusals


def _lock_signed_source_disposition_authority(
    query_handle: _TransactionQuery, manifest: dict[str, Any]
) -> None:
    participants = manifest["participants"]
    removed_target_closures = manifest["removed_target_closures"]
    source_element_ids = sorted(
        {row["source_element_id"] for row in participants}
        | {
            binding["source_element_id"]
            for closure in removed_target_closures
            for binding in closure["incoming_bindings"]
        }
    )
    node_element_ids = sorted(
        {
            element_id
            for row in participants
            for element_id in [
                *(backing["element_id"] for backing in row["backings"]),
                *(binding["target_element_id"] for binding in row["bindings"]),
            ]
        }
        | {closure["target_element_id"] for closure in removed_target_closures}
        | {
            child["child_element_id"]
            for closure in removed_target_closures
            for child in closure["structural_children"]
        }
    )
    relationship_element_ids = sorted(
        {
            element_id
            for row in participants
            for element_id in [
                *(binding["element_id"] for binding in row["bindings"]),
                *(backing["origin_element_id"] for backing in row["backings"]),
                *(
                    projection["element_id"]
                    for backing in row["backings"]
                    for projection in backing["projections"]
                ),
            ]
        }
        | {
            binding["binding_element_id"]
            for closure in removed_target_closures
            for binding in closure["incoming_bindings"]
        }
        | {
            child["relationship_element_id"]
            for closure in removed_target_closures
            for child in closure["structural_children"]
        }
    )
    locked_sources = query_handle.query(
        """
        UNWIND $element_ids AS expected
        MATCH (source:StandardNameSource) WHERE elementId(source) = expected
        SET source.produced_sn_id = source.produced_sn_id
        RETURN collect(elementId(source)) AS ids
        """,
        element_ids=source_element_ids,
    )
    actual_sources = sorted(locked_sources[0].get("ids") or [])
    if actual_sources != source_element_ids:
        raise foundation.services.SignedSourceDispositionConflict(
            "signed source set changed while locking"
        )
    if node_element_ids:
        locked_nodes = query_handle.query(
            """
            UNWIND $element_ids AS expected
            MATCH (node) WHERE elementId(node) = expected
            SET node += {}
            RETURN collect(elementId(node)) AS ids
            """,
            element_ids=node_element_ids,
        )
        if sorted(locked_nodes[0].get("ids") or []) != node_element_ids:
            raise foundation.services.SignedSourceDispositionConflict(
                "signed participant nodes changed while locking"
            )
    if relationship_element_ids:
        locked_relationships = query_handle.query(
            """
            UNWIND $element_ids AS expected
            MATCH ()-[relationship]->() WHERE elementId(relationship) = expected
            SET relationship += {}
            RETURN collect(elementId(relationship)) AS ids
            """,
            element_ids=relationship_element_ids,
        )
        if sorted(locked_relationships[0].get("ids") or []) != relationship_element_ids:
            raise foundation.services.SignedSourceDispositionConflict(
                "signed participant relationships changed while locking"
            )


def _signed_source_disposition_counts(
    rows: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    refusals: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "requested": len(rows),
        "admitted": len(actions),
        "refused": len({refusal["source_id"] for refusal in refusals}),
        "bindings_to_remove": sum(len(action["removals"]) for action in actions),
        "projections_to_remove": sum(len(action["removals"]) for action in actions),
        "scalars_to_change": sum(
            action["prior_scalar_target"] != action["keep_target_id"]
            for action in actions
        ),
    }


def _signed_source_disposition_execution_authority(
    query_handle: _TransactionQuery,
    rows: list[dict[str, Any]],
    adjudication_sha256: str,
    adjudication_row_set_sha256: str,
    reason: str,
    *,
    admitted_subset: bool,
    structural_authority_sha256: str | None,
    structural_target_ids: frozenset[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not admitted_subset:
        manifest, actions, refusals = (
            foundation.services._signed_source_disposition_authority(
                query_handle,
                rows,
                adjudication_sha256,
                adjudication_row_set_sha256,
                reason,
                structural_authority_sha256,
                structural_target_ids,
            )
        )
        return rows, manifest, actions, refusals

    parent_manifest, parent_actions, parent_refusals = (
        foundation.services._signed_source_disposition_authority(
            query_handle,
            rows,
            adjudication_sha256,
            adjudication_row_set_sha256,
            reason,
            structural_authority_sha256,
            structural_target_ids,
        )
    )
    admitted_source_ids = {action["source_id"] for action in parent_actions}
    selected_rows = [row for row in rows if row["source_id"] in admitted_source_ids]
    manifest, actions, refusals = (
        foundation.services._signed_source_disposition_authority(
            query_handle,
            selected_rows,
            adjudication_sha256,
            adjudication_row_set_sha256,
            reason,
            structural_authority_sha256,
            structural_target_ids,
        )
    )
    excluded_source_ids = sorted(
        {row["source_id"] for row in rows} - admitted_source_ids
    )
    manifest["subset_selection"] = {
        "strategy": "admitted_rows_from_complete_signed_adjudication",
        "parent_manifest_sha256": foundation.services._authority_payload_hash(
            parent_manifest
        ),
        "parent_counts": foundation.services._signed_source_disposition_counts(
            rows, parent_actions, parent_refusals
        ),
        "adjudication_source_ids": [row["source_id"] for row in rows],
        "selected_source_ids": sorted(admitted_source_ids),
        "excluded_source_ids": excluded_source_ids,
        "excluded_refusals": parent_refusals,
    }
    return selected_rows, manifest, actions, refusals


_catalog_source_disposition_apply = functools.partial(
    apply_signed_manifest,
    authority_adapter="catalog-disposition",
    mutation_kind="select-survivor-and-release-bindings",
    guard_set=(
        "signed-adjudication",
        "last-producing-source",
        "structural-legitimacy",
        "out-of-allowlist-immutability",
    ),
)


_catalog_source_disposition_signature = inspect.signature(
    _catalog_source_disposition_apply
)


_catalog_source_disposition_parameters = (
    _catalog_source_disposition_signature.parameters
)


_catalog_source_disposition_apply.__signature__ = inspect.Signature(
    [
        _catalog_source_disposition_parameters["authority_path"].replace(
            name="adjudication", annotation="dict[str, Any]"
        ),
        _catalog_source_disposition_parameters["reason"],
        _catalog_source_disposition_parameters["apply"],
        _catalog_source_disposition_parameters["manifest_sha256"],
        _catalog_source_disposition_parameters["run_id"],
        _catalog_source_disposition_parameters["admitted_subset"],
        _catalog_source_disposition_parameters["structural_authority"],
        _catalog_source_disposition_parameters["structural_authority_sha256"],
        _catalog_source_disposition_parameters["gc"],
    ],
    return_annotation=_catalog_source_disposition_signature.return_annotation,
)


SEMANTIC_MIRROR_REPAIR_MANIFEST_SCHEMA = (
    "imas-codex.semantic-source-mirror-repair-manifest"
)


SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA = (
    "imas-codex.semantic-source-mirror-repair-receipt"
)


_PROJECTED_SEMANTIC_SOURCE_TYPES = frozenset({"dd", "signals"})


class SemanticMirrorRepairConflict(RuntimeError):
    """The signed scalar and backing-projection closure changed."""


def _semantic_mirror_repair_authority(
    query_handle: _TransactionQuery,
    source_ids: list[str],
    reason: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows = query_handle.query(
        """
        UNWIND $source_ids AS requested_id
        OPTIONAL MATCH (source:StandardNameSource {id: requested_id})
        RETURN requested_id,
               elementId(source) AS source_element_id,
               properties(source) AS source_properties,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[binding:PRODUCED_NAME]->(target:StandardName) |
                   {element_id: elementId(binding),
                    properties: properties(binding),
                    target_element_id: elementId(target),
                    target_id: target.id,
                    target_properties: properties(target)}]
               END AS bindings,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[origin:FROM_DD_PATH|FROM_SIGNAL]->(backing) |
                   {element_id: elementId(backing),
                    labels: labels(backing),
                    properties: properties(backing),
                    origin_element_id: elementId(origin),
                    origin_type: type(origin),
                    origin_properties: properties(origin),
                    projections: [(backing)-[projection:HAS_STANDARD_NAME]->
                      (projected:StandardName) |
                      {element_id: elementId(projection),
                       properties: properties(projection),
                       target_element_id: elementId(projected),
                       target_id: projected.id,
                       target_properties: properties(projected)}]}]
               END AS backings
        ORDER BY requested_id
        """,
        source_ids=source_ids,
    )
    participants: list[dict[str, foundation.services.Any]] = []
    actions: list[dict[str, foundation.services.Any]] = []
    refusals: list[dict[str, foundation.services.Any]] = []
    already_clean: list[dict[str, foundation.services.Any]] = []
    for raw in rows:
        bindings = sorted(
            (dict(binding) for binding in raw.get("bindings") or []),
            key=lambda binding: (binding["target_id"], binding["element_id"]),
        )
        backings = sorted(
            (
                {
                    **dict(backing),
                    "labels": sorted(backing.get("labels") or []),
                    "projections": sorted(
                        (dict(item) for item in backing.get("projections") or []),
                        key=lambda item: (item["target_id"], item["element_id"]),
                    ),
                }
                for backing in raw.get("backings") or []
            ),
            key=lambda backing: (backing["origin_type"], backing["element_id"]),
        )
        participant = {
            "source_id": raw["requested_id"],
            "source_element_id": raw.get("source_element_id"),
            "source_properties": raw.get("source_properties"),
            "bindings": bindings,
            "backings": backings,
        }
        participants.append(participant)
        properties = raw.get("source_properties")
        refusal: str | None = None
        if properties is None:
            refusal = "source does not exist"
        elif (
            properties.get("claimed_at") is not None
            or properties.get("claim_token") is not None
        ):
            refusal = "source has an active claim"
        elif properties.get("status") not in {"composed", "attached"}:
            refusal = "source is outside the live semantic-mirror lifecycle"

        live_bindings = [
            binding
            for binding in bindings
            if (binding.get("target_properties") or {}).get("name_stage")
            not in foundation.services._TERMINAL_BINDING_NAME_STAGES
        ]
        if refusal is None and len(live_bindings) != 1:
            refusal = "source does not have exactly one live target"

        source_type = properties.get("source_type") if properties else None
        expected_origin_type = {
            "dd": "FROM_DD_PATH",
            "signals": "FROM_SIGNAL",
        }.get(source_type)
        if (
            refusal is None
            and source_type in foundation.services._PROJECTED_SEMANTIC_SOURCE_TYPES
        ):
            if len(backings) != 1 or backings[0]["origin_type"] != expected_origin_type:
                refusal = "projected source does not have one exact upstream backing"
            elif properties.get("source_id") != backings[0]["properties"].get("id"):
                refusal = "source identity disagrees with its upstream backing"
        elif refusal is None and source_type != "derived":
            refusal = "source type has no declared semantic-mirror policy"

        if refusal is not None:
            refusals.append({"source_id": raw["requested_id"], "reason": refusal})
            continue

        target = live_bindings[0]
        target_id = target["target_id"]
        scalar_change = properties.get("produced_sn_id") != target_id
        projection_additions: list[dict[str, foundation.services.Any]] = []
        if source_type in foundation.services._PROJECTED_SEMANTIC_SOURCE_TYPES:
            backing = backings[0]
            matching_projections = [
                projection
                for projection in backing["projections"]
                if projection["target_id"] == target_id
            ]
            if len(matching_projections) > 1:
                refusals.append(
                    {
                        "source_id": raw["requested_id"],
                        "reason": "upstream backing has duplicate target projections",
                    }
                )
                continue
            if not matching_projections:
                projection_additions.append(
                    {
                        "backing_element_id": backing["element_id"],
                        "target_element_id": target["target_element_id"],
                        "target_id": target_id,
                    }
                )
        resolution = {
            "source_id": raw["requested_id"],
            "source_element_id": raw["source_element_id"],
            "source_type": source_type,
            "prior_scalar_target": properties.get("produced_sn_id"),
            "target_id": target_id,
            "target_element_id": target["target_element_id"],
            "scalar_change": scalar_change,
            "projection_additions": projection_additions,
        }
        if scalar_change or projection_additions:
            actions.append(resolution)
        else:
            already_clean.append(resolution)

    refusals = sorted(
        refusals, key=lambda refusal: (refusal["source_id"], refusal["reason"])
    )
    manifest = {
        "schema": foundation.services.SEMANTIC_MIRROR_REPAIR_MANIFEST_SCHEMA,
        "operation": "repair_scalar_projection_mismatches",
        "reason": reason,
        "source_ids": source_ids,
        "participants": participants,
        "actions": actions,
        "already_clean": already_clean,
        "refusals": refusals,
    }
    return manifest, actions, refusals, already_clean


def _lock_semantic_mirror_repair_authority(
    query_handle: _TransactionQuery, manifest: dict[str, Any]
) -> None:
    participants = manifest["participants"]
    source_element_ids = sorted(
        row["source_element_id"]
        for row in participants
        if row["source_element_id"] is not None
    )
    node_element_ids = sorted(
        {
            element_id
            for row in participants
            for element_id in [
                *(backing["element_id"] for backing in row["backings"]),
                *(binding["target_element_id"] for binding in row["bindings"]),
                *(
                    projection["target_element_id"]
                    for backing in row["backings"]
                    for projection in backing["projections"]
                ),
            ]
        }
    )
    relationship_element_ids = sorted(
        {
            element_id
            for row in participants
            for element_id in [
                *(binding["element_id"] for binding in row["bindings"]),
                *(backing["origin_element_id"] for backing in row["backings"]),
                *(
                    projection["element_id"]
                    for backing in row["backings"]
                    for projection in backing["projections"]
                ),
            ]
        }
    )
    locked_sources = query_handle.query(
        """
        UNWIND $element_ids AS expected
        MATCH (source:StandardNameSource) WHERE elementId(source) = expected
        SET source.produced_sn_id = source.produced_sn_id
        RETURN collect(elementId(source)) AS ids
        """,
        element_ids=source_element_ids,
    )
    if sorted(locked_sources[0].get("ids") or []) != source_element_ids:
        raise foundation.services.SemanticMirrorRepairConflict(
            "signed semantic source set changed while locking"
        )
    if node_element_ids:
        locked_nodes = query_handle.query(
            """
            UNWIND $element_ids AS expected
            MATCH (node) WHERE elementId(node) = expected
            SET node += {}
            RETURN collect(elementId(node)) AS ids
            """,
            element_ids=node_element_ids,
        )
        if sorted(locked_nodes[0].get("ids") or []) != node_element_ids:
            raise foundation.services.SemanticMirrorRepairConflict(
                "signed semantic participant nodes changed while locking"
            )
    if relationship_element_ids:
        locked_relationships = query_handle.query(
            """
            UNWIND $element_ids AS expected
            MATCH ()-[relationship]->() WHERE elementId(relationship) = expected
            SET relationship += {}
            RETURN collect(elementId(relationship)) AS ids
            """,
            element_ids=relationship_element_ids,
        )
        if sorted(locked_relationships[0].get("ids") or []) != relationship_element_ids:
            raise foundation.services.SemanticMirrorRepairConflict(
                "signed semantic relationships changed while locking"
            )


def _semantic_mirror_repair_counts(
    source_ids: list[str],
    actions: list[dict[str, Any]],
    refusals: list[dict[str, Any]],
    already_clean: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "requested": len(source_ids),
        "admitted": len(actions),
        "refused": len(refusals),
        "already_clean": len(already_clean),
        "scalars_to_change": sum(action["scalar_change"] for action in actions),
        "projections_to_add": sum(
            len(action["projection_additions"]) for action in actions
        ),
    }


_semantic_mirror_repair_apply = functools.partial(
    apply_signed_manifest,
    authority_adapter="semantic-mirror-repair",
    mutation_kind="restore-semantic-mirror",
    guard_set=(
        "sole-live-target-authority",
        "exact-upstream-backing",
        "out-of-allowlist-immutability",
    ),
)


_semantic_mirror_repair_signature = inspect.signature(_semantic_mirror_repair_apply)


_semantic_mirror_repair_parameters = _semantic_mirror_repair_signature.parameters


_semantic_mirror_repair_apply.__signature__ = inspect.Signature(
    [
        _semantic_mirror_repair_parameters["authority_path"].replace(
            name="source_ids", annotation="list[str]"
        ),
        _semantic_mirror_repair_parameters["reason"],
        _semantic_mirror_repair_parameters["apply"],
        _semantic_mirror_repair_parameters["manifest_sha256"],
        _semantic_mirror_repair_parameters["run_id"],
        _semantic_mirror_repair_parameters["gc"],
    ],
    return_annotation=_semantic_mirror_repair_signature.return_annotation,
)


_ineligible_source_retirement_apply = functools.partial(
    apply_signed_manifest,
    authority_adapter="ineligible-source-retirement",
    mutation_kind="release-ineligible-source-authority",
    guard_set=(
        "signed-lifecycle-and-claim",
        "permitted-orphan-hand-off",
        "out-of-allowlist-immutability",
    ),
)


_ineligible_source_retirement_signature = inspect.signature(
    _ineligible_source_retirement_apply
)


_ineligible_source_retirement_parameters = (
    _ineligible_source_retirement_signature.parameters
)


_ineligible_source_retirement_apply.__signature__ = inspect.Signature(
    [
        _ineligible_source_retirement_parameters["authority_path"].replace(
            name="source_ids", annotation="list[str]"
        ),
        _ineligible_source_retirement_parameters["reason"],
        _ineligible_source_retirement_parameters["apply"],
        _ineligible_source_retirement_parameters["manifest_sha256"],
        _ineligible_source_retirement_parameters["run_id"],
        _ineligible_source_retirement_parameters["gc"],
    ],
    return_annotation=_ineligible_source_retirement_signature.return_annotation,
)


_SIGNED_ORPHAN_RETIREMENT_DISPOSITION = "retire_under_orphan_policy"


def _validate_signed_orphan_retirement_authority(
    authority: dict[str, Any], authority_sha256: str
) -> list[dict[str, Any]]:
    if not foundation.services._SHA256_RE.fullmatch(authority_sha256):
        raise ValueError("orphan authority SHA-256 must be a lowercase digest")
    if (
        foundation.services._catalog_edit_adjudication_signature_hash(authority)
        != authority_sha256
    ):
        raise ValueError("orphan retirement authority signature does not match")
    if (
        authority.get("schema")
        != foundation.services._STRUCTURAL_LEGITIMACY_AUTHORITY_SCHEMA
    ):
        raise ValueError("unsupported orphan retirement authority schema")
    if authority.get("read_only") is not True:
        raise ValueError("orphan retirement authority must be read-only evidence")

    raw_rows = authority.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("orphan retirement authority requires target rows")
    if any(
        not isinstance(row, dict)
        or row.get("disposition") not in foundation.services._ORPHAN_DISPOSITIONS
        for row in raw_rows
    ):
        raise ValueError("orphan retirement authority has an unknown disposition")
    names = [row.get("name") for row in raw_rows]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every orphan disposition requires an exact name")
    if len(names) != len(set(names)):
        raise ValueError("orphan disposition names must be unique")

    disposition_counts = {
        disposition: sum(row["disposition"] == disposition for row in raw_rows)
        for disposition in sorted(foundation.services._ORPHAN_DISPOSITIONS)
    }
    summary = authority.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("targets") != len(raw_rows)
        or summary.get("disposition_sum") != len(raw_rows)
        or summary.get("disposition_counts") != disposition_counts
    ):
        raise ValueError("orphan retirement authority summary does not match rows")

    retirements: list[dict[str, foundation.services.Any]] = []
    for raw_row in raw_rows:
        if (
            raw_row["disposition"]
            != foundation.services._SIGNED_ORPHAN_RETIREMENT_DISPOSITION
        ):
            continue
        row = dict(raw_row)
        if row.get("mutation_authority") != "classification_only":
            raise ValueError(
                "signed orphan evidence cannot directly grant source mutation"
            )
        if row.get("name_stage") not in {"accepted", "reviewed", "drafted", "pending"}:
            raise ValueError("signed orphan target requires a live lifecycle stage")
        closure = row.get("structural_closure")
        if (
            not isinstance(closure, dict)
            or closure.get("classification") != "no_live_structural_descendant"
            or closure.get("has_live_has_parent_child") is not False
            or closure.get("has_live_refined_from_descendant") is not False
            or closure.get("live_has_parent_children") != []
            or closure.get("live_refined_from_descendants") != []
        ):
            raise ValueError(
                "signed orphan retirement requires an empty structural closure"
            )
        removed_bindings = row.get("current_removed_bindings")
        if not isinstance(removed_bindings, list) or not removed_bindings:
            raise ValueError(
                "signed orphan retirement requires name-specific binding evidence"
            )
        retirements.append(
            {
                **row,
                "authority_row_sha256": foundation.services._catalog_edit_adjudication_signature_hash(
                    row
                ),
            }
        )
    retirements.sort(key=lambda row: row["name"])
    if not retirements:
        raise ValueError("orphan authority contains no retirement dispositions")
    if summary.get("remaining_retirements") != len(retirements) or summary.get(
        "retirements_with_name_specific_evidence"
    ) != len(retirements):
        raise ValueError("orphan retirement summary does not match signed targets")
    return retirements


SIGNED_DUAL_AUTHORITY_RETIREMENT_MANIFEST_SCHEMA = (
    "imas-codex.signed-dual-authority-retirement-manifest"
)


SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA = (
    "imas-codex.signed-dual-authority-retirement-receipt"
)


class SignedDualAuthorityRetirementConflict(RuntimeError):
    """The joined source and lifecycle authority no longer matches the graph."""


def _validate_signed_dual_authority_retirement(
    source_adjudication: dict[str, Any],
    retirement_authority: dict[str, Any],
    retirement_authority_sha256: str,
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    source_sha256, source_row_set_sha256, source_rows = (
        foundation.services._validate_signed_source_adjudication(source_adjudication)
    )
    retirement_rows = foundation.services._validate_signed_orphan_retirement_authority(
        retirement_authority, retirement_authority_sha256
    )
    source_by_id = {row["source_id"]: row for row in source_rows}
    retirement_ids = {row["name"] for row in retirement_rows}
    signed_pairs: set[tuple[str, str]] = set()
    for retirement in retirement_rows:
        target_id = retirement["name"]
        for binding in retirement["current_removed_bindings"]:
            source_id = binding.get("source_id")
            source_row = source_by_id.get(source_id)
            if (
                source_row is None
                or binding.get("signed_row_sha256")
                != source_row["row_signature_sha256"]
                or binding.get("signed_surviving_target")
                != source_row["surviving_target"]
                or target_id not in source_row["removed_targets"]
            ):
                raise ValueError(
                    "retirement binding is not jointly authorized by its signed source row"
                )
            pair = (source_id, target_id)
            if pair in signed_pairs:
                raise ValueError("retirement binding authority must be unique")
            signed_pairs.add(pair)

    catalog_pairs = {
        (row["source_id"], target_id)
        for row in source_rows
        for target_id in row["removed_targets"]
        if target_id in retirement_ids
    }
    if signed_pairs != catalog_pairs:
        raise ValueError(
            "retirement bindings must be the exact intersection of both signed authorities"
        )
    covered_targets = {target_id for _, target_id in signed_pairs}
    if covered_targets != retirement_ids:
        raise ValueError(
            "every signed retirement target requires joint binding authority"
        )

    relevant_source_ids = {source_id for source_id, _ in signed_pairs}
    relevant_source_rows = [
        row for row in source_rows if row["source_id"] in relevant_source_ids
    ]
    return (
        source_sha256,
        source_row_set_sha256,
        relevant_source_rows,
        retirement_rows,
    )


def _signed_dual_authority_retirement_manifest(
    query_handle: _TransactionQuery,
    source_rows: list[dict[str, Any]],
    retirement_rows: list[dict[str, Any]],
    *,
    source_sha256: str,
    source_row_set_sha256: str,
    retirement_authority_sha256: str,
    reason: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_ids = [row["source_id"] for row in source_rows]
    target_ids = [row["name"] for row in retirement_rows]
    source_graph_rows = query_handle.query(
        """
        UNWIND $source_ids AS requested_id
        OPTIONAL MATCH (source:StandardNameSource {id: requested_id})
        RETURN requested_id,
               elementId(source) AS source_element_id,
               properties(source) AS source_properties,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[binding:PRODUCED_NAME]->(target:StandardName) |
                   {element_id: elementId(binding),
                    properties: properties(binding),
                    target_element_id: elementId(target),
                    target_id: target.id,
                    target_properties: properties(target)}]
               END AS bindings,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[origin:FROM_DD_PATH]->(backing) |
                   {element_id: elementId(backing),
                    properties: properties(backing),
                    origin_element_id: elementId(origin),
                    origin_properties: properties(origin),
                    projections: [(backing)-[projection:HAS_STANDARD_NAME]->
                      (projected:StandardName) |
                      {element_id: elementId(projection),
                       properties: properties(projection),
                       target_element_id: elementId(projected),
                       target_id: projected.id,
                       target_properties: properties(projected)}]}]
               END AS backings
        ORDER BY requested_id
        """,
        source_ids=source_ids,
    )
    target_graph_rows = query_handle.query(
        """
        UNWIND $target_ids AS requested_id
        OPTIONAL MATCH (target:StandardName {id: requested_id})
        RETURN requested_id,
               elementId(target) AS target_element_id,
               properties(target) AS target_properties,
               CASE WHEN target IS NULL THEN [] ELSE
                 [(source:StandardNameSource)-[binding:PRODUCED_NAME]->(target) |
                   {binding_element_id: elementId(binding),
                    binding_properties: properties(binding),
                    source_element_id: elementId(source),
                    source_id: source.id,
                    source_properties: properties(source)}]
               END AS incoming_bindings,
               CASE WHEN target IS NULL THEN [] ELSE
                 [(child:StandardName)-[relationship:HAS_PARENT]->(target) |
                   {relationship_element_id: elementId(relationship),
                    relationship_properties: properties(relationship),
                    child_element_id: elementId(child),
                    child_id: child.id,
                    child_properties: properties(child)}]
               END AS structural_children
        ORDER BY requested_id
        """,
        target_ids=target_ids,
    )

    retirement_by_name = {row["name"]: row for row in retirement_rows}
    expected_sources_by_target: dict[str, set[str]] = {
        target_id: set() for target_id in target_ids
    }
    expected_targets_by_source: dict[str, set[str]] = {
        source_id: set() for source_id in source_ids
    }
    for retirement in retirement_rows:
        for binding in retirement["current_removed_bindings"]:
            expected_sources_by_target[retirement["name"]].add(binding["source_id"])
            expected_targets_by_source[binding["source_id"]].add(retirement["name"])

    source_by_id = {row["source_id"]: row for row in source_rows}
    participants: list[dict[str, foundation.services.Any]] = []
    actions: list[dict[str, foundation.services.Any]] = []
    refusals: list[dict[str, foundation.services.Any]] = []
    for raw in source_graph_rows:
        expected = source_by_id[raw["requested_id"]]
        bindings = sorted(
            (dict(binding) for binding in raw.get("bindings") or []),
            key=lambda binding: (binding["target_id"], binding["element_id"]),
        )
        backings = sorted(
            (
                {
                    **dict(backing),
                    "projections": sorted(
                        (dict(item) for item in backing.get("projections") or []),
                        key=lambda item: (item["target_id"], item["element_id"]),
                    ),
                }
                for backing in raw.get("backings") or []
            ),
            key=lambda backing: backing["element_id"],
        )
        participant = {
            "source_id": expected["source_id"],
            "adjudication_row_sha256": expected["row_signature_sha256"],
            "source_element_id": raw.get("source_element_id"),
            "source_properties": raw.get("source_properties"),
            "bindings": bindings,
            "backings": backings,
        }
        participants.append(participant)
        properties = raw.get("source_properties")
        refusal: str | None = None
        if properties is None:
            refusal = "source does not exist"
        elif properties.get("status") == "stale":
            refusal = "stale source is not disposition authority"
        elif (
            properties.get("claimed_at") is not None
            or properties.get("claim_token") is not None
        ):
            refusal = "source has an active claim"
        elif properties.get("produced_sn_id") != expected["prior_scalar_target"]:
            refusal = "source scalar changed from signed authority"

        live_bindings = [
            binding
            for binding in bindings
            if (binding.get("target_properties") or {}).get("name_stage")
            not in foundation.services._TERMINAL_BINDING_NAME_STAGES
        ]
        if (
            refusal is None
            and [item["target_id"] for item in live_bindings]
            != expected["candidate_live_targets"]
        ):
            refusal = "live binding set changed from signed authority"
        if refusal is None and (
            len(backings) != 1
            or backings[0]["properties"].get("id") != expected["source_id"][3:]
        ):
            refusal = "source does not have one exact DD backing"
        projection_by_target: dict[str, list[dict[str, foundation.services.Any]]] = {}
        for backing in backings:
            for projection in backing["projections"]:
                if (projection.get("target_properties") or {}).get(
                    "name_stage"
                ) not in foundation.services._TERMINAL_BINDING_NAME_STAGES:
                    projection_by_target.setdefault(projection["target_id"], []).append(
                        {**projection, "backing_element_id": backing["element_id"]}
                    )
        projected_ids = sorted(
            target_id
            for target_id, projections in projection_by_target.items()
            for _ in projections
        )
        if refusal is None and projected_ids != expected["candidate_live_targets"]:
            refusal = "live backing projection set changed from signed authority"
        binding_by_target = {binding["target_id"]: binding for binding in live_bindings}
        scheduled_targets = sorted(expected_targets_by_source[expected["source_id"]])
        if refusal is None and any(
            len(projection_by_target.get(target_id, [])) != 1
            or target_id not in binding_by_target
            for target_id in scheduled_targets
        ):
            refusal = "signed retirement binding closure is not exact"
        if refusal is not None:
            refusals.append({"source_id": expected["source_id"], "reason": refusal})
            continue
        actions.append(
            {
                "source_id": expected["source_id"],
                "source_element_id": raw["source_element_id"],
                "prior_scalar_target": expected["prior_scalar_target"],
                "keep_target_id": expected["surviving_target"],
                "removals": [
                    {
                        "target_id": target_id,
                        "binding_element_id": binding_by_target[target_id][
                            "element_id"
                        ],
                        "projection_element_id": projection_by_target[target_id][0][
                            "element_id"
                        ],
                        "backing_element_id": projection_by_target[target_id][0][
                            "backing_element_id"
                        ],
                    }
                    for target_id in scheduled_targets
                ],
            }
        )

    target_closures: list[dict[str, foundation.services.Any]] = []
    for raw in target_graph_rows:
        signed = retirement_by_name[raw["requested_id"]]
        incoming = sorted(
            (dict(binding) for binding in raw.get("incoming_bindings") or []),
            key=lambda binding: (
                binding["source_id"],
                binding["binding_element_id"],
            ),
        )
        children = sorted(
            (dict(child) for child in raw.get("structural_children") or []),
            key=lambda child: (child["child_id"], child["relationship_element_id"]),
        )
        closure = {
            "target_id": signed["name"],
            "authority_row_sha256": signed["authority_row_sha256"],
            "target_element_id": raw.get("target_element_id"),
            "target_properties": raw.get("target_properties"),
            "incoming_bindings": incoming,
            "structural_children": children,
        }
        target_closures.append(closure)
        properties = raw.get("target_properties")
        live_incoming_sources = sorted(
            binding["source_id"]
            for binding in incoming
            if (binding.get("source_properties") or {}).get("status") != "stale"
        )
        refusal: str | None = None
        if properties is None:
            refusal = "target does not exist"
        elif properties.get("name_stage") != signed["name_stage"]:
            refusal = "target lifecycle changed from signed authority"
        elif (
            properties.get("claimed_at") is not None
            or properties.get("claim_token") is not None
        ):
            refusal = "target has an active claim"
        elif live_incoming_sources != sorted(
            expected_sources_by_target[signed["name"]]
        ):
            refusal = "target live producers differ from the jointly signed bindings"
        elif any(
            foundation.services._is_live_structural_child(child) for child in children
        ):
            refusal = "target has acquired a live HAS_PARENT child"
        if refusal is not None:
            refusals.append({"name_id": signed["name"], "reason": refusal})

    refusals.sort(
        key=lambda refusal: (
            refusal.get("source_id", ""),
            refusal.get("name_id", ""),
            refusal["reason"],
        )
    )
    manifest = {
        "schema": foundation.services.SIGNED_DUAL_AUTHORITY_RETIREMENT_MANIFEST_SCHEMA,
        "operation": "retire_signed_dual_authority_targets",
        "reason": reason,
        "source_authority": {
            "payload_sha256": source_sha256,
            "row_set_sha256": source_row_set_sha256,
            "source_ids": source_ids,
        },
        "retirement_authority": {
            "payload_sha256": retirement_authority_sha256,
            "target_ids": target_ids,
        },
        "participants": participants,
        "removed_target_closures": target_closures,
        "actions": actions,
        "refusals": refusals,
    }
    return manifest, actions, refusals


def _signed_dual_authority_change_id(manifest_sha256: str, name_id: str) -> str:
    name_digest = foundation.services.hashlib.sha256(
        name_id.encode("utf-8")
    ).hexdigest()[:20]
    return f"sn-change:dual-authority-retirement:{manifest_sha256}:{name_digest}"


_dual_authority_retirement_apply = functools.partial(
    apply_signed_manifest,
    authority_adapter="dual-authority-retirement",
    mutation_kind="release-and-supersede",
    guard_set=(
        "authority-join",
        "signed-lifecycle-and-claim",
        "no-live-structural-child",
        "out-of-allowlist-immutability",
    ),
)


_dual_authority_hidden_parameters = {
    "authority_file_sha256",
    "authority_payload_sha256",
    "authority_sha256",
    "authority_adapter",
    "mutation_kind",
    "guard_set",
    "name_ids",
    "client_factory",
}


_dual_authority_retirement_apply.__signature__ = inspect.Signature(
    parameter
    for name, parameter in inspect.signature(
        _dual_authority_retirement_apply
    ).parameters.items()
    if name not in _dual_authority_hidden_parameters
)


_refused_target_orphan_apply = functools.partial(
    apply_signed_manifest,
    authority_adapter="refused-target-orphan",
    mutation_kind="supersede",
    guard_set=(
        "signed-lifecycle-and-claim",
        "no-live-producing-source",
        "no-live-structural-child",
        "out-of-allowlist-immutability",
    ),
)


EXHAUSTED_ORPHAN_SUPERSESSION_MANIFEST_SCHEMA = (
    "imas-codex.exhausted-orphan-supersession-manifest.v1"
)


EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA = (
    "imas-codex.exhausted-orphan-supersession-receipt.v1"
)


class ExhaustedOrphanSupersessionConflict(RuntimeError):
    """The signed exhausted-orphan cohort no longer matches graph authority."""


def classify_sourceless_standard_names(gc: Any) -> list[dict[str, Any]]:
    """Classify every live name without a producer using its own source record.

    Recorded paths, resolution identities, source mirrors and detachment
    receipts belong to the name itself. Related names supply chain topology,
    but their source records do not establish that this name was ever bound.
    The independent count protects against an incomplete population read.
    """
    from imas_codex.standard_names.source_invariant import _derived_parent_cypher

    terminal = ["deprecated", "superseded"]

    def read(cypher: str, **params: Any) -> list[dict[str, Any]]:
        started = foundation.services.time.monotonic()
        rows = gc.query(cypher, **params)
        if foundation.services.time.monotonic() - started >= 10:
            raise RuntimeError("source audit query exceeded ten seconds")
        return rows

    count = read(
        """MATCH (sn:StandardName)
        WHERE NOT sn.status IN $terminal
          AND NOT EXISTS {
            (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
          }
        RETURN count(sn) AS total""",
        terminal=terminal,
    )[0]["total"]
    names = read(
        f"""MATCH (sn:StandardName)
        WHERE NOT sn.status IN $terminal
          AND NOT EXISTS {{
            (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
          }}
        RETURN sn.id AS id, sn.name_stage AS stage, sn.docs_stage AS docs_stage,
               sn.validation_status AS validation_status, sn.model AS model,
               sn.status AS status, sn.origin AS origin, sn.source_paths AS paths,
               sn.source_path AS path,
               sn.source_dd_resolution_converged_ids AS resolutions,
               ({_derived_parent_cypher()}) AS derived_parent
        ORDER BY id LIMIT 6000""",
        terminal=terminal,
    )
    if len(names) != count or len({row["id"] for row in names}) != count:
        raise RuntimeError(f"source audit population incomplete: {len(names)}/{count}")
    sources = read(
        """MATCH (source:StandardNameSource)
        RETURN source.id AS id, source.source_id AS source_id,
               source.produced_sn_id AS mirror,
               [(source)-[:PRODUCED_NAME]->(holder:StandardName)
                WHERE NOT holder.status IN $terminal | holder.id] AS holders
        ORDER BY id LIMIT 12000""",
        terminal=terminal,
    )
    source_total = read(
        "MATCH (source:StandardNameSource) RETURN count(source) AS total"
    )[0]["total"]
    if len(sources) != source_total:
        raise RuntimeError(
            f"source audit source read incomplete: {len(sources)}/{source_total}"
        )
    relations = read(
        """MATCH (a:StandardName)-[rel:REFINED_FROM|HAS_SUCCESSOR]->(b:StandardName)
        RETURN a.id AS a, type(rel) AS kind, b.id AS b LIMIT 12000"""
    )
    changes = read(
        """MATCH (change:StandardNameChange)
        RETURN change.from_name AS predecessor, change.to_name AS successor,
               change.operation AS operation LIMIT 20000"""
    )
    projections = read(
        """MATCH (dd:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
        RETURN dd.id AS path, sn.id AS name LIMIT 12000"""
    )
    all_name_ids = {
        row["id"]
        for row in read("MATCH (sn:StandardName) RETURN sn.id AS id LIMIT 6000")
    }
    inventory = read(
        """CALL { MATCH (sn:StandardName) RETURN count(sn) AS names }
        CALL { MATCH (:StandardName)-[rel:REFINED_FROM|HAS_SUCCESSOR]->(:StandardName)
               RETURN count(rel) AS relations }
        CALL { MATCH (change:StandardNameChange) RETURN count(change) AS changes }
        CALL { MATCH (:IMASNode)-[projection:HAS_STANDARD_NAME]->(:StandardName)
               RETURN count(projection) AS projections }
        RETURN names, relations, changes, projections"""
    )[0]
    if (
        len(all_name_ids) != inventory["names"]
        or len(relations) != inventory["relations"]
        or len(changes) != inventory["changes"]
        or len(projections) != inventory["projections"]
    ):
        raise RuntimeError("source audit supporting read incomplete")
    aliases: dict[str, set[str]] = foundation.services.defaultdict(set)
    mirrors: dict[str, set[str]] = foundation.services.defaultdict(set)
    holders: dict[str, set[str]] = {}
    for source in sources:
        source_id = source["id"]
        aliases[str(source_id)].add(source_id)
        if source["source_id"] is not None:
            aliases[str(source["source_id"])].add(source_id)
        if source["mirror"]:
            mirrors[source["mirror"]].add(source_id)
        holders[source_id] = set(source["holders"] or [])
    links: dict[str, set[str]] = foundation.services.defaultdict(set)
    for relation in relations:
        old, new = (
            (relation["b"], relation["a"])
            if relation["kind"] == "REFINED_FROM"
            else (relation["a"], relation["b"])
        )
        links[old].add(new)
        links[new].add(old)
    receipts: dict[str, set[str]] = foundation.services.defaultdict(set)
    detachment_paths: dict[str, set[str]] = foundation.services.defaultdict(set)
    for change in changes:
        old, new, operation = (
            change["predecessor"],
            change["successor"],
            change["operation"] or "",
        )
        # The graph supplies the operation vocabulary. An event connecting two
        # actual names records their identity chain, including future writers.
        if old in all_name_ids and new in all_name_ids and old != new:
            links[old].add(new)
            links[new].add(old)
        if operation.startswith("detach_") and isinstance(old, str) and "/" in old:
            detachment_paths[new].add("dd:" + old.removeprefix("dd:"))
        if operation in {
            "source_migration_manifest",
            "reattach_lapsed_detachment",
            "repair_semantic_source_binding",
            "recover_terminal_source_binding",
            "move_source_snapshot",
            "rebind_sources",
        }:
            receipts[new].add(operation)
    projection_paths: dict[str, set[str]] = foundation.services.defaultdict(set)
    for projection in projections:
        if projection["path"]:
            projection_paths[projection["name"]].add("dd:" + projection["path"])

    def chain(start: str) -> set[str]:
        seen: set[str] = set()
        pending = foundation.services.deque(links[start])
        while pending:
            target = pending.popleft()
            if target in seen:
                continue
            seen.add(target)
            pending.extend(links[target] - seen)
        return seen | {start}

    classified = []
    for name in names:
        identifier = name["id"]
        tokens = set(name["paths"] or []) | set(name["resolutions"] or [])
        if name["path"]:
            tokens.add(name["path"])
        tokens.update(projection_paths[identifier])
        tokens.update(detachment_paths[identifier])
        own_sources = set(mirrors[identifier])
        unresolved = set()
        for token in tokens:
            matches = set(aliases[str(token)])
            if not str(token).startswith(("dd:", "derived:")):
                matches.update(aliases["dd:" + str(token)])
            own_sources.update(matches)
            if not matches:
                unresolved.add(str(token))
        linked = chain(identifier)
        unheld = {source for source in own_sources if not holders[source]}
        unchained = {
            holder
            for source in own_sources
            for holder in holders[source]
            if holder not in linked
        }
        if name["derived_parent"]:
            population = "derived_parent"
        elif not tokens and not own_sources and not receipts[identifier]:
            population = "never_bound"
        elif unresolved or unheld or not own_sources:
            population = "dropped_edge"
        elif unchained:
            population = "unchained_holder"
        elif all(holders[source] & linked for source in own_sources):
            population = (
                "orphan_awaiting_retirement"
                if name["stage"] == "exhausted" and name["origin"] != "derived"
                else "held_in_chain"
            )
        else:
            population = "dropped_edge"
        classified.append(
            {
                "id": identifier,
                "name_stage": name["stage"],
                "docs_stage": name["docs_stage"],
                "validation_status": name["validation_status"],
                "model": name["model"],
                "status": name["status"],
                "origin": name["origin"],
                "population": population,
                "recorded_source_ids": sorted(own_sources),
                "unresolved_tokens": sorted(unresolved),
                "unchained_holders": sorted(unchained),
            }
        )
    return classified


def _exhausted_orphan_supersession_authority(
    query_handle: _TransactionQuery,
    name_ids: list[str],
    reason: str,
    eligible_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    rows = query_handle.query(
        """
        UNWIND $name_ids AS requested_id
        OPTIONAL MATCH (name:StandardName {id: requested_id})
        OPTIONAL MATCH (source:StandardNameSource)-[binding:PRODUCED_NAME]->(name)
        WITH requested_id, name,
             [producer IN collect(CASE WHEN source IS NULL THEN null ELSE
               {source_element_id: elementId(source),
                source_properties: properties(source),
                binding_element_id: elementId(binding),
                binding_properties: properties(binding)} END)
              WHERE producer IS NOT NULL] AS producers
        RETURN requested_id,
               elementId(name) AS name_element_id,
               properties(name) AS name_properties,
               producers
        ORDER BY requested_id
        """,
        name_ids=name_ids,
    )
    participants: list[dict[str, foundation.services.Any]] = []
    actions: list[dict[str, foundation.services.Any]] = []
    refusals: list[dict[str, str]] = []
    for raw in rows:
        participant = dict(raw)
        participant["producers"] = sorted(
            (dict(producer) for producer in participant.get("producers") or []),
            key=lambda producer: (
                producer["source_properties"].get("id", ""),
                producer["binding_element_id"],
            ),
        )
        participants.append(participant)
        properties = participant.get("name_properties")
        refusal: str | None = None
        if properties is None:
            refusal = "name does not exist"
        elif properties.get("name_stage") != "exhausted":
            refusal = f"name_stage is {properties.get('name_stage')!r}, not 'exhausted'"
        elif properties.get("origin") == "derived":
            refusal = "origin is derived"
        elif participant["producers"]:
            refusal = "name has a live producing source"
        elif raw["requested_id"] not in eligible_ids:
            refusal = (
                "source audit does not classify name as an orphan awaiting retirement"
            )
        if refusal is not None:
            refusals.append({"name_id": raw["requested_id"], "reason": refusal})
            continue
        actions.append(
            {
                "name_id": raw["requested_id"],
                "name_element_id": participant["name_element_id"],
                "expected_properties": properties,
            }
        )

    manifest = {
        "schema": foundation.services.EXHAUSTED_ORPHAN_SUPERSESSION_MANIFEST_SCHEMA,
        "operation": "supersede_exhausted_standard_name_orphans",
        "reason": reason,
        "name_ids": name_ids,
        "participants": participants,
        "actions": actions,
        "refusals": refusals,
    }
    return manifest, actions, refusals


def _exhausted_orphan_change_id(manifest_sha256: str, name_id: str) -> str:
    name_digest = foundation.services.hashlib.sha256(
        name_id.encode("utf-8")
    ).hexdigest()[:20]
    return f"sn-change:exhausted-orphan:{manifest_sha256}:{name_digest}"


@retry_on_deadlock()
def supersede_exhausted_standard_name_orphans(
    name_ids: list[str] | None = None,
    *,
    reason: str,
    apply: bool = False,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Supersede a signed cohort of non-derived exhausted source-less names.

    Omitting ``name_ids`` computes the fresh complete eligible cohort. Explicit
    ids retain refusal evidence for names outside that cohort. Apply requires
    the exact preview hash and writes one durable change row for every name.
    """
    if not reason.strip():
        raise ValueError("exhausted orphan supersession requires a non-empty reason")
    if name_ids is not None and len(set(name_ids)) != len(name_ids):
        raise ValueError("exhausted orphan supersession requires unique name ids")
    if apply and manifest_sha256 is None:
        raise ValueError("apply requires manifest_sha256")
    if manifest_sha256 is not None and not foundation.services._SHA256_RE.fullmatch(
        manifest_sha256
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")

    own = gc is None
    client: foundation.services.Any = foundation.services.GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            query_handle = foundation.services._TransactionQuery(transaction)
            try:
                eligible_ids = {
                    row["id"]
                    for row in foundation.services.classify_sourceless_standard_names(
                        query_handle
                    )
                    if row["population"] == "orphan_awaiting_retirement"
                }
                requested = (
                    sorted(name_ids) if name_ids is not None else sorted(eligible_ids)
                )
                if apply and requested:
                    expected_event_ids = [
                        foundation.services._exhausted_orphan_change_id(
                            manifest_sha256, name_id
                        )
                        for name_id in requested
                    ]
                    replay = query_handle.query(
                        """
                        UNWIND $rows AS expected
                        OPTIONAL MATCH (name:StandardName {id: expected.name_id})
                        OPTIONAL MATCH (change:StandardNameChange {
                          id: expected.event_id,
                          operation: 'supersede_exhausted_orphan',
                          manifest_sha256: $manifest_sha256
                        })
                        RETURN collect({name_id: expected.name_id,
                          stage: name.name_stage, status: name.status,
                          live_sources: COUNT {
                            (:StandardNameSource)-[:PRODUCED_NAME]->(name)
                          }, change_id: change.id}) AS rows
                        """,
                        rows=[
                            {"name_id": name_id, "event_id": event_id}
                            for name_id, event_id in zip(
                                requested, expected_event_ids, strict=True
                            )
                        ],
                        manifest_sha256=manifest_sha256,
                    )
                    replay_rows = replay[0].get("rows") if replay else []
                    present_events = [
                        row for row in replay_rows if row.get("change_id") is not None
                    ]
                    if present_events:
                        if len(present_events) != len(requested) or any(
                            row.get("stage") != "superseded"
                            or row.get("status") != "superseded"
                            or int(row.get("live_sources") or 0) != 0
                            for row in replay_rows
                        ):
                            raise foundation.services.ExhaustedOrphanSupersessionConflict(
                                "recorded orphan supersession has lost its postcondition"
                            )
                        transaction.rollback()
                        return {
                            "schema": foundation.services.EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA,
                            "outcome": "already_applied",
                            "dry_run": False,
                            "changed": 0,
                            "persistent_writes": 0,
                            "superseded": len(requested),
                            "ledger_rows": len(present_events),
                            "manifest_sha256": manifest_sha256,
                            "change_ids": sorted(expected_event_ids),
                        }

                manifest, actions, refusals = (
                    foundation.services._exhausted_orphan_supersession_authority(
                        query_handle, requested, reason, eligible_ids
                    )
                )
                computed_hash = foundation.services._authority_payload_hash(manifest)
                counts = {
                    "requested": len(requested),
                    "admitted": len(actions),
                    "refused": len(refusals),
                }
                if apply and computed_hash != manifest_sha256:
                    raise foundation.services.ExhaustedOrphanSupersessionConflict(
                        "fresh exhausted-orphan manifest does not match signed hash"
                    )
                if refusals:
                    transaction.rollback()
                    return {
                        "schema": foundation.services.EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA,
                        "outcome": "refused",
                        "dry_run": not apply,
                        "changed": 0,
                        "would_change": 0,
                        "counts": counts,
                        "refusals": refusals,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }
                if not apply:
                    transaction.rollback()
                    return {
                        "schema": foundation.services.EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA,
                        "outcome": "would_apply" if actions else "already_clean",
                        "dry_run": True,
                        "changed": 0,
                        "would_change": len(actions),
                        "counts": counts,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }

                locked = query_handle.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (name:StandardName {id: expected.name_id})
                    WHERE elementId(name) = expected.name_element_id
                    SET name.name_stage = name.name_stage
                    RETURN collect(name.id) AS ids
                    """,
                    rows=actions,
                )
                locked_ids = sorted(locked[0].get("ids") or []) if locked else []
                if locked_ids != requested:
                    raise foundation.services.ExhaustedOrphanSupersessionConflict(
                        "orphan participant set changed while acquiring locks"
                    )
                locked_manifest, locked_actions, locked_refusals = (
                    foundation.services._exhausted_orphan_supersession_authority(
                        query_handle, requested, reason, eligible_ids
                    )
                )
                if (
                    locked_refusals
                    or foundation.services._authority_payload_hash(locked_manifest)
                    != computed_hash
                ):
                    raise foundation.services.ExhaustedOrphanSupersessionConflict(
                        "orphan authority changed while acquiring participant locks"
                    )
                mutation_rows = [
                    {
                        **action,
                        "event_id": foundation.services._exhausted_orphan_change_id(
                            computed_hash, action["name_id"]
                        ),
                    }
                    for action in locked_actions
                ]
                changed_rows = query_handle.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (name:StandardName {id: expected.name_id})
                    WHERE elementId(name) = expected.name_element_id
                      AND name.name_stage = 'exhausted'
                      AND coalesce(name.origin, '') <> 'derived'
                      AND NOT EXISTS {
                        (:StandardNameSource)-[:PRODUCED_NAME]->(name)
                      }
                    SET name.updated_at = datetime(), name.name_stage = 'superseded',
                        name.status = 'superseded',
                        name.claimed_at = null,
                        name.claim_token = null
                    CREATE (change:StandardNameChange {
                      id: expected.event_id,
                      from_name: expected.name_id,
                      to_name: expected.name_id,
                      operation: 'supersede_exhausted_orphan',
                      reason: $reason,
                      origin: 'orphan_reconciliation',
                      run_id: $run_id,
                      changed_at: datetime(),
                      internal: true,
                      manifest_sha256: $manifest_sha256
                    })
                    CREATE (name)-[:HAS_INTERNAL_CHANGE]->(change)
                    RETURN name.id AS name_id, change.id AS change_id
                    ORDER BY name_id
                    """,
                    rows=mutation_rows,
                    reason=reason,
                    run_id=run_id,
                    manifest_sha256=computed_hash,
                )
                if [row["name_id"] for row in changed_rows] != requested:
                    raise foundation.services.ExhaustedOrphanSupersessionConflict(
                        "orphan lifecycle compare-and-set changed during supersession"
                    )
                transaction.commit()
                return {
                    "schema": foundation.services.EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA,
                    "outcome": "applied",
                    "dry_run": False,
                    "changed": len(changed_rows),
                    "persistent_writes": len(changed_rows) * 4,
                    "superseded": len(changed_rows),
                    "ledger_rows": len(changed_rows),
                    "counts": counts,
                    "manifest": manifest,
                    "manifest_sha256": computed_hash,
                    "change_ids": [row["change_id"] for row in changed_rows],
                }
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own:
            client.close()


ANCESTOR_SUPERSESSION_MANIFEST_SCHEMA = (
    "imas-codex.standard-name-ancestor-supersession-manifest.v1"
)


ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA = (
    "imas-codex.standard-name-ancestor-supersession-receipt.v1"
)


class SupersedeIntoAncestorConflict(RuntimeError):
    """The signed ancestor fold no longer matches its graph authority."""


def _ancestor_supersession_authority(
    query_handle: _TransactionQuery,
    old_id: str,
    ancestor_id: str,
    reason: str,
) -> tuple[dict[str, Any], list[str], list[str], list[str], list[str]]:
    rows = query_handle.query(
        """
        OPTIONAL MATCH (old:StandardName {id: $old_id})
        OPTIONAL MATCH (ancestor:StandardName {id: $ancestor_id})
        RETURN properties(old) AS old_properties,
               properties(ancestor) AS ancestor_properties,
               CASE WHEN old IS NULL OR ancestor IS NULL THEN false ELSE
                 EXISTS { MATCH (old)-[:REFINED_FROM*1..]->(ancestor) }
               END AS is_ancestor,
               CASE WHEN old IS NULL OR ancestor IS NULL THEN false ELSE
                 EXISTS { MATCH (ancestor)-[:REFINED_FROM*1..]->(old) }
               END AS reverse_reachable
        """,
        old_id=old_id,
        ancestor_id=ancestor_id,
    )
    row = rows[0] if rows else {}
    old_properties = row.get("old_properties")
    ancestor_properties = row.get("ancestor_properties")
    if old_properties is None:
        raise foundation.services.SupersedeIntoAncestorConflict(
            f"name {old_id!r} not found"
        )
    if ancestor_properties is None:
        raise foundation.services.SupersedeIntoAncestorConflict(
            f"target {ancestor_id!r} not found"
        )
    if not row.get("is_ancestor"):
        raise foundation.services.SupersedeIntoAncestorConflict(
            f"target {ancestor_id!r} is not an ancestor of {old_id!r}"
        )
    if row.get("reverse_reachable"):
        raise foundation.services.SupersedeIntoAncestorConflict(
            f"target {ancestor_id!r} participates in a directed cycle with {old_id!r}"
        )
    if ancestor_properties.get("name_stage") not in ("accepted", "approved"):
        raise foundation.services.SupersedeIntoAncestorConflict(
            f"ancestor {ancestor_id!r} is not an accepted live name"
        )
    if ancestor_properties.get("validation_status") != "valid":
        raise foundation.services.SupersedeIntoAncestorConflict(
            f"ancestor {ancestor_id!r} is not validation_status='valid'"
        )
    if old_properties.get("name_stage") == "superseded":
        raise foundation.services.SupersedeIntoAncestorConflict(
            f"name {old_id!r} is already superseded by a different manifest"
        )

    upstream_present = (
        foundation.services._standard_name_source_upstream_present_cypher("source")
    )
    source_rows = query_handle.query(
        """
        MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->
              (:StandardName {id: $old_id})
        RETURN source.id AS id,
               properties(source) AS properties,
        """
        + upstream_present
        + """ AS upstream_present,
               [(source)-[:PRODUCED_NAME]->(bound:StandardName) | bound.id]
                 AS bindings,
               [(source)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing) |
                 {element_id: elementId(backing), labels: labels(backing),
                  properties: properties(backing),
                  projections: [(backing)-[projection:HAS_STANDARD_NAME]->
                    (projected) |
                    {element_id: elementId(projection),
                     target_id: projected.id,
                     properties: properties(projection)}]}] AS backings
        ORDER BY source.id
        """,
        old_id=old_id,
    )
    sources: list[dict[str, foundation.services.Any]] = []
    retarget_ids: list[str] = []
    deduplicate_ids: list[str] = []
    stale_detach_ids: list[str] = []
    stale_refusal_ids: list[str] = []
    for raw in source_rows:
        source = dict(raw)
        source["bindings"] = sorted(set(source.get("bindings") or []))
        source["backings"] = sorted(
            (
                {
                    **dict(backing),
                    "labels": sorted(backing.get("labels") or []),
                    "projections": sorted(
                        (dict(item) for item in backing.get("projections") or []),
                        key=lambda item: (item["target_id"], item["element_id"]),
                    ),
                }
                for backing in source.get("backings") or []
            ),
            key=lambda backing: backing["element_id"],
        )
        source_id = source["id"]
        properties = source.get("properties") or {}
        scalar = properties.get("produced_sn_id")
        if properties.get("status") == "stale":
            if source.get("upstream_present"):
                stale_refusal_ids.append(source_id)
            else:
                allowed_targets = {old_id, ancestor_id}
                if not set(source["bindings"]).issubset(allowed_targets) or (
                    scalar is not None and scalar not in allowed_targets
                ):
                    raise foundation.services.SupersedeIntoAncestorConflict(
                        "stale source set cannot be detached exactly: "
                        f"{source_id!r} has bindings={source['bindings']!r}, "
                        f"scalar={scalar!r}"
                    )
                stale_detach_ids.append(source_id)
        elif source["bindings"] == [old_id] and scalar == old_id:
            retarget_ids.append(source_id)
        elif (
            source["bindings"] == sorted([old_id, ancestor_id])
            and scalar == ancestor_id
        ):
            deduplicate_ids.append(source_id)
        else:
            raise foundation.services.SupersedeIntoAncestorConflict(
                "source set cannot be migrated exactly: "
                f"{source_id!r} has bindings={source['bindings']!r}, scalar={scalar!r}"
            )
        sources.append(source)

    lineage_rows = query_handle.query(
        """
        MATCH (old:StandardName {id: $old_id})
        MATCH (member:StandardName)
        WHERE member = old OR EXISTS {
          MATCH (old)-[:REFINED_FROM*1..]-(member)
        }
        WITH collect(DISTINCT member.id) AS member_ids
        MATCH (successor:StandardName)-[edge:REFINED_FROM]->
              (predecessor:StandardName)
        WHERE successor.id IN member_ids AND predecessor.id IN member_ids
        RETURN elementId(edge) AS element_id,
               successor.id AS successor_id,
               predecessor.id AS predecessor_id,
               properties(edge) AS properties
        ORDER BY successor_id, predecessor_id, element_id
        """,
        old_id=old_id,
    )
    manifest = {
        "schema": foundation.services.ANCESTOR_SUPERSESSION_MANIFEST_SCHEMA,
        "operation": "supersede_into_ancestor",
        "old_id": old_id,
        "ancestor_id": ancestor_id,
        "reason": reason,
        "participants": {
            "old_properties": old_properties,
            "ancestor_properties": ancestor_properties,
            "sources": sources,
            "lineage_edges": [dict(item) for item in lineage_rows],
        },
        "actions": {
            "retarget_source_ids": retarget_ids,
            "deduplicate_source_ids": deduplicate_ids,
            "detach_stale_source_ids": stale_detach_ids,
            "refuse_present_upstream_stale_source_ids": stale_refusal_ids,
        },
    }
    return (
        manifest,
        retarget_ids,
        deduplicate_ids,
        stale_detach_ids,
        stale_refusal_ids,
    )


def _lock_ancestor_supersession_authority(
    query_handle: _TransactionQuery,
    manifest: dict[str, Any],
) -> None:
    """Lock every signed source, backing node, and backing projection."""
    sources = manifest["participants"]["sources"]
    source_ids = sorted(source["id"] for source in sources)
    backing_element_ids = sorted(
        {backing["element_id"] for source in sources for backing in source["backings"]}
    )
    projection_element_ids = sorted(
        {
            projection["element_id"]
            for source in sources
            for backing in source["backings"]
            for projection in backing["projections"]
        }
    )

    if source_ids:
        rows = query_handle.query(
            """
            UNWIND $source_ids AS source_id
            MATCH (source:StandardNameSource {id: source_id})
            SET source.produced_sn_id = source.produced_sn_id
            RETURN collect(source.id) AS locked_ids
            """,
            source_ids=source_ids,
        )
        locked = sorted(set(rows[0].get("locked_ids") or [])) if rows else []
        if locked != source_ids:
            raise foundation.services.SupersedeIntoAncestorConflict(
                "signed source set changed while acquiring participant locks"
            )

    if backing_element_ids:
        rows = query_handle.query(
            """
            UNWIND $element_ids AS expected_element_id
            MATCH (backing)
            WHERE elementId(backing) = expected_element_id
            SET backing.id = backing.id
            RETURN collect(elementId(backing)) AS locked_ids
            """,
            element_ids=backing_element_ids,
        )
        locked = sorted(set(rows[0].get("locked_ids") or [])) if rows else []
        if locked != backing_element_ids:
            raise foundation.services.SupersedeIntoAncestorConflict(
                "signed backing set changed while acquiring participant locks"
            )

    if projection_element_ids:
        rows = query_handle.query(
            """
            UNWIND $element_ids AS expected_element_id
            MATCH ()-[projection:HAS_STANDARD_NAME]->()
            WHERE elementId(projection) = expected_element_id
            SET projection += {}
            RETURN collect(elementId(projection)) AS locked_ids
            """,
            element_ids=projection_element_ids,
        )
        locked = sorted(set(rows[0].get("locked_ids") or [])) if rows else []
        if locked != projection_element_ids:
            raise foundation.services.SupersedeIntoAncestorConflict(
                "signed backing projection set changed while acquiring "
                "participant locks"
            )


def _detach_stale_ancestor_sources(
    query_handle: _TransactionQuery,
    manifest: dict[str, Any],
    stale_detach_ids: list[str],
) -> int:
    """Remove live-name bindings from signed upstream-stale sources."""
    if not stale_detach_ids:
        return 0
    stale_id_set = set(stale_detach_ids)
    rows = [
        {
            "source_id": source["id"],
            "expected_bindings": source["bindings"],
            "expected_scalar": (source.get("properties") or {}).get("produced_sn_id"),
            "projection_element_ids": sorted(
                projection["element_id"]
                for backing in source["backings"]
                for projection in backing["projections"]
                if projection["target_id"] in source["bindings"]
            ),
        }
        for source in manifest["participants"]["sources"]
        if source["id"] in stale_id_set
    ]
    detached = query_handle.query(
        """
        UNWIND $rows AS expected
        MATCH (source:StandardNameSource {id: expected.source_id})
        WHERE source.status = 'stale'
          AND source.claimed_at IS NULL
          AND source.claim_token IS NULL
          AND ((expected.expected_scalar IS NULL
                AND source.produced_sn_id IS NULL)
               OR source.produced_sn_id = expected.expected_scalar)
        MATCH (source)-[binding:PRODUCED_NAME]->(bound:StandardName)
        WITH expected, source, collect(binding) AS bindings,
             collect(DISTINCT bound.id) AS bound_ids
        WHERE size(bound_ids) = size(expected.expected_bindings)
          AND all(id IN bound_ids WHERE id IN expected.expected_bindings)
          AND all(id IN expected.expected_bindings WHERE id IN bound_ids)
        FOREACH (binding IN bindings | DELETE binding)
        SET source.produced_sn_id = null
        RETURN collect(DISTINCT source.id) AS source_ids
        """,
        rows=rows,
    )
    actual_ids = sorted(set(detached[0].get("source_ids") or []) if detached else set())
    if actual_ids != stale_detach_ids:
        raise foundation.services.SupersedeIntoAncestorConflict(
            "stale source binding set changed during ancestor fold"
        )

    projection_ids = sorted(
        projection_id for row in rows for projection_id in row["projection_element_ids"]
    )
    if projection_ids:
        deleted = query_handle.query(
            """
            UNWIND $projection_ids AS projection_id
            MATCH ()-[projection:HAS_STANDARD_NAME]->()
            WHERE elementId(projection) = projection_id
            DELETE projection
            RETURN collect(projection_id) AS projection_ids
            """,
            projection_ids=projection_ids,
        )
        actual_projection_ids = sorted(
            set(deleted[0].get("projection_ids") or []) if deleted else set()
        )
        if actual_projection_ids != projection_ids:
            raise foundation.services.SupersedeIntoAncestorConflict(
                "stale source projection set changed during ancestor fold"
            )
    return len(actual_ids)


@retry_on_deadlock()
def supersede_into_ancestor(
    old_id: str,
    ancestor_id: str,
    *,
    reason: str,
    apply: bool = False,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Fold a name into its accepted lineage ancestor without rewriting history.

    The preview signs the complete name, source, backing-projection, and lineage
    authority used by the fold. Apply is compare-and-set bound to that digest.
    Singly bound sources pass the attachment guard and the standard retarget
    primitive; sources already bound to the ancestor lose only their redundant
    descendant edge. No ``REFINED_FROM`` relationship is created or deleted.
    """
    if old_id == ancestor_id:
        raise foundation.services.SupersedeIntoAncestorConflict(
            "ancestor supersession requires two distinct names"
        )
    if not reason.strip():
        raise ValueError("ancestor supersession requires a non-empty reason")
    if apply and not manifest_sha256:
        raise ValueError("apply requires manifest_sha256")
    if manifest_sha256 is not None and not foundation.services._SHA256_RE.fullmatch(
        manifest_sha256
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")

    own = gc is None
    client: foundation.services.Any = foundation.services.GraphClient() if own else gc
    event_id = (
        f"sn-change:ancestor-supersession:{manifest_sha256}"
        if manifest_sha256
        else None
    )
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            query_handle = foundation.services._TransactionQuery(transaction)
            try:
                if apply:
                    replay_rows = query_handle.query(
                        """
                        OPTIONAL MATCH (event:StandardNameChange {id: $event_id})
                        OPTIONAL MATCH (old:StandardName {id: $old_id})
                        OPTIONAL MATCH (ancestor:StandardName {id: $ancestor_id})
                        RETURN event.id IS NOT NULL AS recorded,
                               old.name_stage AS old_stage,
                               CASE WHEN old IS NULL OR ancestor IS NULL THEN false ELSE
                                 EXISTS {
                                   MATCH (old)-[:REFINED_FROM*1..]->(ancestor)
                                 }
                                 AND NOT EXISTS {
                                   MATCH (ancestor)-[:REFINED_FROM*1..]->(old)
                                 }
                               END AS lineage_preserved,
                               CASE WHEN old IS NULL THEN 0 ELSE
                                 COUNT { (:StandardNameSource)-[:PRODUCED_NAME]->(old) }
                               END AS old_sources
                        """,
                        event_id=event_id,
                        old_id=old_id,
                        ancestor_id=ancestor_id,
                    )
                    replay = replay_rows[0] if replay_rows else {}
                    if replay.get("recorded"):
                        if not (
                            replay.get("old_stage") == "superseded"
                            and replay.get("lineage_preserved")
                            and int(replay.get("old_sources") or 0) == 0
                        ):
                            raise foundation.services.SupersedeIntoAncestorConflict(
                                "recorded ancestor supersession has lost its postcondition"
                            )
                        transaction.rollback()
                        return {
                            "schema": foundation.services.ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA,
                            "outcome": "already_applied",
                            "dry_run": False,
                            "changed": 0,
                            "manifest_sha256": manifest_sha256,
                        }
                    query_handle.query(
                        """
                        MATCH (old:StandardName {id: $old_id}),
                              (ancestor:StandardName {id: $ancestor_id})
                        SET old.claimed_at = old.claimed_at,
                            ancestor.claimed_at = ancestor.claimed_at
                        RETURN old.id AS old_id
                        """,
                        old_id=old_id,
                        ancestor_id=ancestor_id,
                    )

                (
                    manifest,
                    retarget_ids,
                    deduplicate_ids,
                    stale_detach_ids,
                    stale_refusal_ids,
                ) = foundation.services._ancestor_supersession_authority(
                    query_handle, old_id, ancestor_id, reason
                )
                computed_hash = foundation.services._authority_payload_hash(manifest)
                if apply and manifest_sha256 != computed_hash:
                    raise foundation.services.SupersedeIntoAncestorConflict(
                        "fresh ancestor supersession manifest does not match signed hash"
                    )
                all_source_ids = (
                    retarget_ids
                    + deduplicate_ids
                    + stale_detach_ids
                    + stale_refusal_ids
                )
                counts = {
                    "sources": len(all_source_ids),
                    "retarget": len(retarget_ids),
                    "deduplicate": len(deduplicate_ids),
                    "stale_detach": len(stale_detach_ids),
                    "stale_refused": len(stale_refusal_ids),
                }
                if stale_refusal_ids:
                    transaction.rollback()
                    return {
                        "schema": foundation.services.ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA,
                        "outcome": "refused",
                        "dry_run": not apply,
                        "changed": 0,
                        "would_change": 0,
                        "counts": counts,
                        "refusals": [
                            {
                                "source_id": source_id,
                                "reason": (
                                    "stale source has a present upstream entity"
                                ),
                            }
                            for source_id in stale_refusal_ids
                        ],
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }
                if apply:
                    foundation.services._lock_ancestor_supersession_authority(
                        query_handle, manifest
                    )
                    try:
                        (
                            locked_manifest,
                            locked_retarget,
                            locked_deduplicate,
                            locked_stale_detach,
                            locked_stale_refusal,
                        ) = foundation.services._ancestor_supersession_authority(
                            query_handle, old_id, ancestor_id, reason
                        )
                    except foundation.services.SupersedeIntoAncestorConflict as exc:
                        raise foundation.services.SupersedeIntoAncestorConflict(
                            "authority changed while acquiring participant locks"
                        ) from exc
                    if (
                        foundation.services._authority_payload_hash(locked_manifest)
                        != computed_hash
                    ):
                        raise foundation.services.SupersedeIntoAncestorConflict(
                            "authority changed while acquiring participant locks"
                        )
                    manifest = locked_manifest
                    retarget_ids = locked_retarget
                    deduplicate_ids = locked_deduplicate
                    stale_detach_ids = locked_stale_detach
                    stale_refusal_ids = locked_stale_refusal

                live_source_ids = retarget_ids + deduplicate_ids
                if live_source_ids:
                    from imas_codex.standard_names.attachment_audit import (
                        guard_source_pairings,
                    )

                    guarded = guard_source_pairings(
                        query_handle, ancestor_id, live_source_ids
                    )
                    if guarded.rejected or set(guarded.accepted_source_ids) != set(
                        live_source_ids
                    ):
                        details = (
                            ", ".join(
                                f"{item.source_node_id}: {item.reason}"
                                for item in guarded.rejected
                            )
                            or "guard did not admit the exact source cohort"
                        )
                        raise foundation.services.SupersedeIntoAncestorConflict(
                            f"source re-validation failed: {details}"
                        )

                if not apply:
                    transaction.rollback()
                    return {
                        "schema": foundation.services.ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA,
                        "outcome": "would_apply",
                        "dry_run": True,
                        "changed": 0,
                        "would_change": 1,
                        "counts": counts,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }

                stale_detached = foundation.services._detach_stale_ancestor_sources(
                    query_handle, manifest, stale_detach_ids
                )

                if retarget_ids:
                    from imas_codex.standard_names.provenance_lifecycle import (
                        retarget_standard_name_sources,
                    )

                    moved = retarget_standard_name_sources(
                        query_handle,
                        old_id,
                        ancestor_id,
                        operation="ancestor_fold",
                        reason=reason,
                        run_id=run_id,
                        record_change=False,
                        enforce_consistency=False,
                        source_ids=retarget_ids,
                        expected_current_bindings=dict.fromkeys(retarget_ids, old_id),
                        _transactional=True,
                    )
                    if moved != len(retarget_ids):
                        raise foundation.services.SupersedeIntoAncestorConflict(
                            "retargeted source count changed during ancestor fold"
                        )

                if deduplicate_ids:
                    deduplicated = query_handle.query(
                        """
                        UNWIND $source_ids AS source_id
                        MATCH (source:StandardNameSource {id: source_id}),
                              (old:StandardName {id: $old_id}),
                              (ancestor:StandardName {id: $ancestor_id})
                        WHERE source.produced_sn_id = $ancestor_id
                          AND COUNT {
                            (source)-[:PRODUCED_NAME]->(:StandardName)
                          } = 2
                          AND EXISTS { (source)-[:PRODUCED_NAME]->(old) }
                          AND EXISTS { (source)-[:PRODUCED_NAME]->(ancestor) }
                        MATCH (source)-[redundant:PRODUCED_NAME]->(old)
                        DELETE redundant
                        WITH source, old, ancestor
                        OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
                        OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
                        OPTIONAL MATCH (dd)-[dd_projection:HAS_STANDARD_NAME]->(old)
                        DELETE dd_projection
                        WITH source, old, ancestor, signal
                        OPTIONAL MATCH (signal)-[signal_projection:HAS_STANDARD_NAME]->(old)
                        DELETE signal_projection
                        RETURN collect(DISTINCT source.id) AS source_ids
                        """,
                        source_ids=deduplicate_ids,
                        old_id=old_id,
                        ancestor_id=ancestor_id,
                    )
                    actual = sorted(
                        set(deduplicated[0].get("source_ids") or [])
                        if deduplicated
                        else set()
                    )
                    if actual != deduplicate_ids:
                        raise foundation.services.SupersedeIntoAncestorConflict(
                            "dual-bound source set changed during ancestor fold"
                        )

                query_handle.query(
                    """
                    MATCH (old:StandardName {id: $old_id}),
                          (ancestor:StandardName {id: $ancestor_id})
                    OPTIONAL MATCH (old_source:StandardNameSource)
                      -[:PRODUCED_NAME]->(old)
                    OPTIONAL MATCH (old_source)-[:FROM_DD_PATH]->(old_dd:IMASNode)
                    OPTIONAL MATCH (old_source)-[:FROM_SIGNAL]->
                      (old_signal:FacilitySignal)
                    WITH old, ancestor,
                         collect(DISTINCT CASE
                           WHEN old_source IS NULL THEN null
                           WHEN old_dd IS NOT NULL THEN 'dd:' + old_dd.id
                           WHEN old_signal IS NOT NULL THEN old_signal.id
                           WHEN old_source.source_type = 'derived'
                             AND old_source.source_id STARTS WITH 'derived:'
                           THEN old_source.source_id ELSE old_source.id END)
                           AS old_paths
                    OPTIONAL MATCH (ancestor_source:StandardNameSource)
                      -[:PRODUCED_NAME]->(ancestor)
                    OPTIONAL MATCH (ancestor_source)-[:FROM_DD_PATH]->
                      (ancestor_dd:IMASNode)
                    OPTIONAL MATCH (ancestor_source)-[:FROM_SIGNAL]->
                      (ancestor_signal:FacilitySignal)
                    WITH old, ancestor, old_paths,
                         collect(DISTINCT CASE
                           WHEN ancestor_source IS NULL THEN null
                           WHEN ancestor_dd IS NOT NULL THEN 'dd:' + ancestor_dd.id
                           WHEN ancestor_signal IS NOT NULL THEN ancestor_signal.id
                           WHEN ancestor_source.source_type = 'derived'
                             AND ancestor_source.source_id STARTS WITH 'derived:'
                           THEN ancestor_source.source_id ELSE ancestor_source.id END)
                           AS ancestor_paths
                    SET old.source_paths = [path IN old_paths WHERE path IS NOT NULL],
                        ancestor.source_paths =
                          [path IN ancestor_paths WHERE path IS NOT NULL],
                        old.updated_at = datetime(),
                        ancestor.updated_at = datetime()
                    """,
                    old_id=old_id,
                    ancestor_id=ancestor_id,
                )

                finalized = query_handle.query(
                    """
                    MATCH (old:StandardName {id: $old_id}),
                          (ancestor:StandardName {id: $ancestor_id})
                    WHERE EXISTS { MATCH (old)-[:REFINED_FROM*1..]->(ancestor) }
                      AND NOT EXISTS {
                        MATCH (ancestor)-[:REFINED_FROM*1..]->(old)
                      }
                      AND ancestor.name_stage IN ['accepted', 'approved']
                      AND ancestor.validation_status = 'valid'
                      AND NOT EXISTS {
                        MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(old)
                      }
                    SET old.updated_at = datetime(), old.name_stage = 'superseded',
                        old.status = 'superseded',
                        old.superseded_from_stage =
                          coalesce(old.superseded_from_stage, 'accepted'),
                        old.claim_token = null,
                        old.claimed_at = null,
                        old.edit_status = CASE
                          WHEN coalesce(old.edit_status, '') = 'open'
                          THEN 'applied' ELSE old.edit_status END
                    MERGE (change:StandardNameChange {id: $event_id})
                    ON CREATE SET change.from_name = old.id,
                                  change.to_name = ancestor.id,
                                  change.operation = 'supersede_into_ancestor',
                                  change.reason = $reason,
                                  change.origin = 'lineage_reconciliation',
                                  change.run_id = $run_id,
                                  change.changed_at = datetime(),
                                  change.internal = true,
                                  change.manifest_sha256 = $manifest_sha256
                    MERGE (ancestor)-[:HAS_INTERNAL_CHANGE]->(change)
                    RETURN old.id AS old_id, change.id AS change_id
                    """,
                    old_id=old_id,
                    ancestor_id=ancestor_id,
                    event_id=f"sn-change:ancestor-supersession:{computed_hash}",
                    reason=reason,
                    run_id=run_id,
                    manifest_sha256=computed_hash,
                )
                if len(finalized) != 1:
                    raise foundation.services.SupersedeIntoAncestorConflict(
                        "ancestor fold lifecycle compare-and-set changed"
                    )

                post_lineage = query_handle.query(
                    """
                    MATCH (old:StandardName {id: $old_id})
                    MATCH (member:StandardName)
                    WHERE member = old OR EXISTS {
                      MATCH (old)-[:REFINED_FROM*1..]-(member)
                    }
                    WITH collect(DISTINCT member.id) AS member_ids
                    MATCH (successor:StandardName)-[edge:REFINED_FROM]->
                          (predecessor:StandardName)
                    WHERE successor.id IN member_ids AND predecessor.id IN member_ids
                    RETURN elementId(edge) AS element_id,
                           successor.id AS successor_id,
                           predecessor.id AS predecessor_id,
                           properties(edge) AS properties
                    ORDER BY successor_id, predecessor_id, element_id
                    """,
                    old_id=old_id,
                )
                if [dict(item) for item in post_lineage] != manifest["participants"][
                    "lineage_edges"
                ]:
                    raise foundation.services.SupersedeIntoAncestorConflict(
                        "REFINED_FROM history changed during ancestor fold"
                    )
                transaction.commit()
                return {
                    "schema": foundation.services.ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA,
                    "outcome": "applied",
                    "dry_run": False,
                    "changed": 1,
                    "sources_retargeted": len(retarget_ids),
                    "sources_deduplicated": len(deduplicate_ids),
                    "sources_stale_detached": stale_detached,
                    "counts": counts,
                    "manifest": manifest,
                    "manifest_sha256": computed_hash,
                    "change_id": finalized[0]["change_id"],
                }
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own:
            client.close()

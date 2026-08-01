"""Standard-name semantic provenance and internal change-history operations.

Semantic sources describe *which DD path or signal supports the current name*.
They are distinct from pipeline history (discarded candidates, reviews, edits,
and runs).  All name-changing routes use the retarget operation here so the
source ledger has one current target while lightweight change events can retain
an internal audit trail after unapproved candidates are compacted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

_DD_DOCS_ROOT = "https://imas-data-dictionary.readthedocs.io/en"

DELETION_OPERATIONS = frozenset(
    {
        "clear_selected_name",
        "clear_subsystem_name",
        "compact_unapproved_name",
        "cancel_staged_rename",
        "remove_provenance_orphan",
        "remove_derived_parent",
        "remove_skeleton_placeholder",
    }
)


def deletion_change_cypher(name_alias: str) -> str:
    """Return a Cypher clause that records a name deletion in its transaction."""
    if not name_alias.isidentifier():
        raise ValueError(f"invalid Cypher name alias: {name_alias!r}")
    return f"""
        CREATE (change:StandardNameChange {{
          id: 'sn-change:' + randomUUID(),
          from_name: {name_alias}.id,
          to_name: {name_alias}.id,
          operation: $deletion_operation,
          reason: $deletion_reason,
          origin: $deletion_origin,
          run_id: $deletion_run_id,
          changed_at: datetime(),
          internal: true
        }})
    """


def deletion_change_params(
    operation: str,
    *,
    reason: str,
    origin: str = "pipeline_cleanup",
    run_id: str | None = None,
) -> dict[str, str | None]:
    """Build validated parameters for an atomic deletion-ledger clause."""
    if operation not in DELETION_OPERATIONS:
        raise ValueError(f"unknown StandardName deletion operation: {operation!r}")
    return {
        "deletion_operation": operation,
        "deletion_reason": reason,
        "deletion_origin": origin,
        "deletion_run_id": run_id,
    }


def cancel_staged_rename(
    gc: Any,
    successor_id: str,
    *,
    reason: str,
    dry_run: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
) -> dict[str, Any]:
    """Cancel one unaccepted rename and restore its superseded predecessor.

    This is deliberately narrower than pruning: the successor must carry an
    open rename edit, remain unaccepted, and point directly to a superseded
    predecessor. Semantic-source and ownership edges move back to the
    predecessor in the same transaction that records and deletes the rejected
    successor.
    """
    if not reason.strip():
        raise ValueError("cancel_staged_rename requires a non-empty reason")
    rows = list(
        gc.query(
            """
            MATCH (successor:StandardName {id: $successor_id})
                  -[:REFINED_FROM]->(predecessor:StandardName)
            WHERE successor.edit_mode = 'rename'
              AND successor.edit_status = 'open'
              AND successor.name_stage IN ['drafted', 'reviewed', 'exhausted']
              AND predecessor.name_stage = 'superseded'
            RETURN successor.id AS successor,
                   successor.name_stage AS successor_stage,
                   predecessor.id AS predecessor,
                   CASE
                     WHEN predecessor.catalog_approved_at IS NOT NULL
                       OR predecessor.reviewer_score_name >= $min_score
                     THEN 'accepted'
                     WHEN predecessor.superseded_from_stage IN
                          ['pending', 'drafted', 'reviewed', 'exhausted']
                     THEN predecessor.superseded_from_stage
                     ELSE 'reviewed'
                   END AS predecessor_stage
            """,
            successor_id=successor_id,
            min_score=min_score,
        )
    )
    if len(rows) != 1:
        return {
            "ok": False,
            "successor": successor_id,
            "reason": "target is not one cancellable open rename successor",
            "dry_run": dry_run,
        }
    row = dict(rows[0])
    if dry_run:
        return {"ok": True, **row, "dry_run": True}

    deletion_clause = deletion_change_cypher("successor")
    result = list(
        gc.query(
            f"""
            MATCH (successor:StandardName {{id: $successor_id}})
                  -[:REFINED_FROM]->(predecessor:StandardName)
            WHERE successor.edit_mode = 'rename'
              AND successor.edit_status = 'open'
              AND successor.name_stage IN ['drafted', 'reviewed', 'exhausted']
              AND predecessor.name_stage = 'superseded'
            OPTIONAL MATCH (source:StandardNameSource)
                           -[produced:PRODUCED_NAME]->(successor)
            WITH successor, predecessor,
                 collect(DISTINCT source) AS sources,
                 collect(DISTINCT produced) AS produced_edges
            OPTIONAL MATCH (owner)-[owned:HAS_STANDARD_NAME]->(successor)
            WITH successor, predecessor, sources, produced_edges,
                 collect(DISTINCT owner) AS owners,
                 collect(DISTINCT owned) AS ownership_edges
            OPTIONAL MATCH (successor)-[:HAS_REVIEW]->(review:StandardNameReview)
            OPTIONAL MATCH (successor)-[:DOCS_REVISION_OF]->(revision:DocsRevision)
            WITH successor, predecessor, sources, produced_edges,
                 owners, ownership_edges,
                 collect(DISTINCT review) AS reviews,
                 collect(DISTINCT revision) AS revisions
            {deletion_clause}
            SET predecessor.name_stage = CASE
                    WHEN predecessor.catalog_approved_at IS NOT NULL
                      OR predecessor.reviewer_score_name >= $min_score
                    THEN 'accepted'
                    WHEN predecessor.superseded_from_stage IN
                         ['pending', 'drafted', 'reviewed', 'exhausted']
                    THEN predecessor.superseded_from_stage
                    ELSE 'reviewed'
                END,
                predecessor.superseded_from_stage = null,
                predecessor.claimed_at = null,
                predecessor.claim_token = null
            FOREACH (source IN sources |
              SET source.standard_name_id = predecessor.id,
                  source.claimed_at = null,
                  source.claim_token = null
              MERGE (source)-[:PRODUCED_NAME]->(predecessor))
            FOREACH (edge IN produced_edges | DELETE edge)
            FOREACH (owner IN owners |
              MERGE (owner)-[:HAS_STANDARD_NAME]->(predecessor))
            FOREACH (edge IN ownership_edges | DELETE edge)
            FOREACH (item IN reviews | DETACH DELETE item)
            FOREACH (item IN revisions | DETACH DELETE item)
            DETACH DELETE successor
            RETURN predecessor.id AS predecessor,
                   predecessor.name_stage AS restored_stage,
                   size(sources) AS sources_restored,
                   size(owners) AS owners_restored
            """,
            successor_id=successor_id,
            min_score=min_score,
            **deletion_change_params(
                "cancel_staged_rename",
                reason=reason,
                origin="operator_correction",
            ),
        )
    )
    if len(result) != 1:
        raise RuntimeError("staged rename no longer satisfies cancellation constraints")
    return {
        "ok": True,
        "successor": successor_id,
        **dict(result[0]),
        "dry_run": False,
    }


def official_dd_documentation_url(dd_version: str, dd_path: str) -> str:
    """Build the official version-pinned IDS reference URL for a DD path."""
    if not dd_version or not dd_path:
        raise ValueError("both dd_version and dd_path are required")
    ids = dd_path.split("/", 1)[0]
    version_part = quote(dd_version, safe=".-")
    ids_part = quote(ids, safe="_-")
    anchor = quote(dd_path.replace("/", "-"), safe="_-")
    return f"{_DD_DOCS_ROOT}/{version_part}/generated/ids/{ids_part}.html#{anchor}"


def fetch_public_semantic_sources(gc: Any, name: str) -> list[dict[str, Any]]:
    """Return graph-held DD/signal semantics without operational history.

    DD sources are pinned to the version recorded when the source was extracted.
    A missing version is an export error: callers must never infer or link to the
    latest DD. Authoritative raw DD content and non-authoritative enhancement
    context are separate public objects. Internal model/hash/cost/timestamps and
    edit history are never selected.
    """
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name})<-[:PRODUCED_NAME]-(src:StandardNameSource)
        OPTIONAL MATCH (src)-[:FROM_SIGNAL]->(signal:FacilitySignal)
        RETURN CASE WHEN src.source_type = 'dd' THEN src.source_id END AS dd_path,
               src.dd_version AS dd_version,
               src.dd_snapshot_pinned AS dd_snapshot_pinned,
               src.dd_documentation AS leaf_documentation,
               src.dd_parent_path AS parent_path,
               src.dd_parent_documentation AS parent_documentation,
               src.dd_data_type AS data_type,
               src.dd_unit AS unit,
               src.dd_coordinates AS coordinates,
               src.dd_lifecycle_status AS lifecycle_status,
               src.dd_lifecycle_version AS lifecycle_version,
               src.enhanced_description AS enhanced_description,
               src.enhancement_kind AS enhancement_kind,
               signal.id AS signal_id,
               src.provenance AS semantic_facet
        ORDER BY dd_path, signal_id
        """,
        name=name,
    )
    sources: list[dict[str, Any]] = []
    for row in rows or []:
        if row.get("dd_path"):
            dd_version = row.get("dd_version")
            if not dd_version:
                raise ValueError(
                    f"DD source {row['dd_path']!r} for {name!r} has no pinned "
                    "dd_version; refusing to infer the latest version"
                )
            if not row.get("dd_snapshot_pinned"):
                raise ValueError(
                    f"DD source {row['dd_path']!r} for {name!r} has no provable "
                    "immutable snapshot; refusing public projection"
                )
            source: dict[str, Any] = {
                "dd_path": row["dd_path"],
                "dd_version": dd_version,
                "dd_documentation_url": official_dd_documentation_url(
                    dd_version, row["dd_path"]
                ),
            }
            authoritative = {
                "leaf": row.get("leaf_documentation"),
                "parent_path": row.get("parent_path"),
                "parent": row.get("parent_documentation"),
                "data_type": row.get("data_type"),
                "unit": row.get("unit"),
                "coordinates": [
                    value for value in row.get("coordinates") or [] if value
                ],
                "lifecycle_status": row.get("lifecycle_status"),
                "lifecycle_version": row.get("lifecycle_version"),
            }
            source["dd_documentation"] = {
                key: value
                for key, value in authoritative.items()
                if value is not None and value != []
            }
            enhanced = {
                "description": row.get("enhanced_description"),
                "kind": row.get("enhancement_kind"),
            }
            enhanced = {key: value for key, value in enhanced.items() if value}
            if enhanced:
                source["enhanced_context"] = enhanced
            if row.get("semantic_facet") is not None:
                source["semantic_facet"] = row["semantic_facet"]
            sources.append(source)
        elif row.get("signal_id"):
            source = {"signal_id": row["signal_id"]}
            if row.get("semantic_facet") is not None:
                source["semantic_facet"] = row["semantic_facet"]
            sources.append(source)
    return sources


def retarget_standard_name_sources(
    gc: Any,
    old_name: str,
    new_name: str,
    *,
    operation: str = "refine",
    reason: str | None = None,
    origin: str | None = None,
    run_id: str | None = None,
    record_change: bool = True,
) -> int:
    """Move every semantic source from ``old_name`` to ``new_name``.

    ``FROM_DD_PATH`` / ``FROM_SIGNAL`` are never changed.  The operation makes
    ``PRODUCED_NAME``, its scalar mirror, upstream ``HAS_STANDARD_NAME`` and the
    successor's ``source_paths`` projection agree.  Competing historical source
    edges are removed; history belongs in ``StandardNameChange`` instead.
    """
    if not old_name or not new_name or old_name == new_name:
        return 0
    rows = gc.query(
        """
        MATCH (new:StandardName {id: $new_name})
        OPTIONAL MATCH (old:StandardName {id: $old_name})
        OPTIONAL MATCH (sns:StandardNameSource)
        // A successor edge is authoritative. Its scalar alone can outlive an
        // attachment-gate rejection and must not recreate the detached edge.
        WHERE (sns)-[:PRODUCED_NAME]->(old)
           OR sns.produced_sn_id = $old_name
           OR (sns)-[:PRODUCED_NAME]->(new)
        WITH new, old, collect(DISTINCT sns) AS sources
        SET new.source_paths = []
        WITH new, old, sources
        UNWIND sources AS source
        WITH new, old, source WHERE source IS NOT NULL
        OPTIONAL MATCH (source)-[prior:PRODUCED_NAME]->(:StandardName)
        DELETE prior
        MERGE (source)-[:PRODUCED_NAME]->(new)
        SET source.produced_sn_id = new.id
        WITH DISTINCT new, old, source
        OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
        OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
        OPTIONAL MATCH (dd)-[dd_old:HAS_STANDARD_NAME]->(:StandardName)
        DELETE dd_old
        WITH DISTINCT new, old, source, dd, signal
        OPTIONAL MATCH (signal)-[sig_old:HAS_STANDARD_NAME]->(:StandardName)
        DELETE sig_old
        FOREACH (_ IN CASE WHEN dd IS NULL THEN [] ELSE [1] END |
          MERGE (dd)-[:HAS_STANDARD_NAME]->(new))
        FOREACH (_ IN CASE WHEN signal IS NULL THEN [] ELSE [1] END |
          MERGE (signal)-[:HAS_STANDARD_NAME]->(new))
        WITH new, collect(DISTINCT source) AS moved,
             collect(DISTINCT CASE WHEN dd IS NULL THEN null ELSE 'dd:' + dd.id END) +
             collect(DISTINCT CASE WHEN signal IS NULL THEN null ELSE signal.id END) +
             collect(DISTINCT CASE
               WHEN dd IS NULL AND signal IS NULL
               THEN CASE
                 WHEN source.source_type = 'derived'
                  AND source.source_id STARTS WITH 'derived:'
                 THEN source.source_id
                 ELSE source.id
               END
               ELSE null
             END) AS authoritative_paths
        // Rebuild the cache from edge-bound sources. Existing caches can retain
        // paths rejected by the attachment gate and are not authoritative.
        WITH new, moved, [p IN authoritative_paths WHERE p IS NOT NULL] AS paths
        SET new.source_paths = reduce(acc = [], p IN paths |
          CASE WHEN p IN acc THEN acc ELSE acc + p END)
        RETURN size(moved) AS moved
        """,
        old_name=old_name,
        new_name=new_name,
    )
    moved = int(rows[0].get("moved", 0)) if rows else 0
    if record_change:
        record_standard_name_change(
            gc,
            old_name,
            new_name,
            operation=operation,
            reason=reason,
            origin=origin,
            run_id=run_id,
        )
    return moved


def bind_sources_exclusively(gc: Any, name: str, source_ids: list[str]) -> int:
    """Make listed source ids point exclusively at ``name`` and repair mirrors."""
    if not name or not source_ids:
        return 0
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name})
        UNWIND $source_ids AS source_id
        MATCH (source:StandardNameSource {id: source_id})
        OPTIONAL MATCH (source)-[prior:PRODUCED_NAME]->(:StandardName)
        DELETE prior
        MERGE (source)-[:PRODUCED_NAME]->(sn)
        SET source.produced_sn_id = sn.id
        WITH DISTINCT sn, source
        OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
        OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
        OPTIONAL MATCH (dd)-[dd_old:HAS_STANDARD_NAME]->(:StandardName)
        DELETE dd_old
        WITH DISTINCT sn, source, dd, signal
        OPTIONAL MATCH (signal)-[sig_old:HAS_STANDARD_NAME]->(:StandardName)
        DELETE sig_old
        FOREACH (_ IN CASE WHEN dd IS NULL THEN [] ELSE [1] END |
          MERGE (dd)-[:HAS_STANDARD_NAME]->(sn))
        FOREACH (_ IN CASE WHEN signal IS NULL THEN [] ELSE [1] END |
          MERGE (signal)-[:HAS_STANDARD_NAME]->(sn))
        WITH sn, collect(DISTINCT source) AS bound,
             collect(DISTINCT CASE WHEN dd IS NULL THEN null ELSE 'dd:' + dd.id END) +
             collect(DISTINCT CASE WHEN signal IS NULL THEN null ELSE signal.id END)
             AS paths
        SET sn.source_paths = [p IN paths WHERE p IS NOT NULL]
        RETURN size(bound) AS bound
        """,
        name=name,
        source_ids=sorted(set(source_ids)),
    )
    return int(rows[0]["bound"]) if rows else 0


def refresh_renamed_source_mirrors(gc: Any, renames: list[dict[str, str]]) -> int:
    """Repair scalar back-references after an in-place cascade id rename."""
    if not renames:
        return 0
    rows = gc.query(
        """
        UNWIND $renames AS rename
        MATCH (sn:StandardName {id: rename.to})
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        FOREACH (_ IN CASE WHEN source IS NULL THEN [] ELSE [1] END |
          SET source.produced_sn_id = sn.id)
        WITH sn, source
        OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
        FOREACH (_ IN CASE WHEN review IS NULL THEN [] ELSE [1] END |
          SET review.standard_name_id = sn.id)
        WITH sn, source
        OPTIONAL MATCH (sn)-[:DOCS_REVISION_OF]->(revision:DocsRevision)
        FOREACH (_ IN CASE WHEN revision IS NULL THEN [] ELSE [1] END |
          SET revision.sn_id = sn.id)
        RETURN count(DISTINCT source) AS refreshed
        """,
        renames=renames,
    )
    return int(rows[0]["refreshed"]) if rows else 0


def find_semantic_source_invariant_violations(gc: Any) -> list[dict[str, Any]]:
    """Find composed/attached sources whose current-target mirrors disagree."""
    rows = gc.query(
        """
        MATCH (source:StandardNameSource)
        WHERE source.status IN ['composed', 'attached']
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(sn:StandardName)
        WITH source, collect(DISTINCT sn) AS targets
        WITH source, targets,
             [target IN targets WHERE NOT target.name_stage IN
               ['superseded', 'exhausted']] AS live_targets
        OPTIONAL MATCH (source)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
        OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(mapped:StandardName)
        WITH source, targets, live_targets,
             collect(DISTINCT mapped.id) AS mapped_ids
        WHERE size(live_targets) <> 1
           OR source.produced_sn_id <> live_targets[0].id
           OR NOT live_targets[0].id IN mapped_ids
        RETURN source.id AS source_id,
               [target IN targets | target.id] AS produced_targets,
               [target IN live_targets | target.id] AS live_targets,
               source.produced_sn_id AS produced_sn_id,
               mapped_ids
        ORDER BY source.id
        """
    )
    return [dict(row) for row in rows or []]


_SEMANTIC_SOURCE_REPAIR_INSPECTION = """
UNWIND $source_ids AS source_id
MATCH (source:StandardNameSource {id: source_id})
OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
WITH source,
     [item IN collect(DISTINCT {
        id: target.id,
        stage: target.name_stage,
        validation_status: target.validation_status
      })
      WHERE item.id IS NOT NULL] AS targets
OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
WITH source, targets, collect(DISTINCT dd.id) AS dd_backings
OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
WITH source, targets, dd_backings,
     collect(DISTINCT signal.id) AS signal_backings
OPTIONAL MATCH (source)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(mapped:StandardName)
OPTIONAL MATCH (owner:StandardNameSource)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
RETURN source.id AS source_id,
       source.source_id AS semantic_id,
       source.source_type AS source_type,
       source.status AS status,
       source.produced_sn_id AS produced_sn_id,
       [target IN targets | target.id] AS produced_targets,
       targets AS target_states,
       [target IN targets
        WHERE NOT target.stage IN ['superseded', 'exhausted'] |
        target.id] AS live_targets,
       dd_backings,
       signal_backings,
       collect(DISTINCT mapped.id) AS mapped_ids,
       collect(DISTINCT owner.id) AS backing_owner_ids
ORDER BY source_id
"""


_SEMANTIC_SOURCE_REPAIR_MUTATION = """
MATCH (source:StandardNameSource {id: $source_id})
WHERE source.source_type = $source_type
  AND source.source_id = $semantic_id
  AND source.status = $status
  AND (source.produced_sn_id = $before_scalar
       OR (source.produced_sn_id IS NULL AND $before_scalar IS NULL))
OPTIONAL MATCH (source)-[produced:PRODUCED_NAME]->(current:StandardName)
WITH source, collect(DISTINCT current.id) AS current_targets
WHERE size(current_targets) = size($before_targets)
  AND all(id IN current_targets WHERE id IN $before_targets)
OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(state_target:StandardName)
WITH source, current_targets,
     [state IN collect(DISTINCT {
        id: state_target.id,
        stage: state_target.name_stage,
        validation_status: state_target.validation_status
      }) WHERE state.id IS NOT NULL] AS current_target_states
WHERE size(current_target_states) = size($before_target_states)
  AND all(state IN current_target_states WHERE any(
    expected IN $before_target_states
    WHERE expected.id = state.id
      AND coalesce(expected.stage, '') = coalesce(state.stage, '')
      AND coalesce(expected.validation_status, '') =
          coalesce(state.validation_status, '')
  ))
OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(current_live:StandardName)
WHERE NOT current_live.name_stage IN ['superseded', 'exhausted']
WITH source, current_targets, current_target_states,
     collect(DISTINCT current_live.id) AS current_live_targets
WHERE size(current_live_targets) = size($before_live_targets)
  AND all(id IN current_live_targets WHERE id IN $before_live_targets)
OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
WITH source, current_targets, current_target_states, current_live_targets,
     collect(DISTINCT dd.id) AS dd_backings
WHERE size(dd_backings) = size($dd_backings)
  AND all(id IN dd_backings WHERE id IN $dd_backings)
OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
WITH source, current_targets, current_target_states, current_live_targets, dd_backings,
     collect(DISTINCT signal.id) AS signal_backings
WHERE size(signal_backings) = size($signal_backings)
  AND all(id IN signal_backings WHERE id IN $signal_backings)
OPTIONAL MATCH (source)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
OPTIONAL MATCH (backing)-[projection:HAS_STANDARD_NAME]->(mapped:StandardName)
WITH source, current_targets, current_target_states, current_live_targets,
     dd_backings, signal_backings, backing,
     collect(DISTINCT mapped.id) AS mapped_ids,
     collect(DISTINCT projection) AS projections
WHERE size(mapped_ids) = size($before_mapped_ids)
  AND all(id IN mapped_ids WHERE id IN $before_mapped_ids)
OPTIONAL MATCH (owner:StandardNameSource)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
WITH source, current_targets, current_target_states, current_live_targets,
     dd_backings, signal_backings, backing, mapped_ids, projections,
     collect(DISTINCT owner.id) AS backing_owner_ids
WHERE size(backing_owner_ids) = size($backing_owner_ids)
  AND all(id IN backing_owner_ids WHERE id IN $backing_owner_ids)
MATCH (target:StandardName {id: $target})
OPTIONAL MATCH (source)-[stale:PRODUCED_NAME]->(old:StandardName)
WHERE old.id <> target.id
WITH source, target, backing, projections,
     collect(DISTINCT stale) AS stale_edges
FOREACH (edge IN stale_edges | DELETE edge)
MERGE (source)-[:PRODUCED_NAME]->(target)
SET source.produced_sn_id = target.id
FOREACH (edge IN projections | DELETE edge)
MERGE (backing)-[:HAS_STANDARD_NAME]->(target)
CREATE (change:StandardNameChange {
  id: $change_id,
  from_name: coalesce($before_scalar, $target),
  to_name: $target,
  operation: 'repair_semantic_source_binding',
  reason: $audit_reason,
  origin: $origin,
  run_id: $run_id,
  changed_at: datetime(),
  internal: true
})
MERGE (target)-[:HAS_INTERNAL_CHANGE]->(change)
RETURN source.id AS source_id, target.id AS target, change.id AS change_id
"""


_SEMANTIC_SOURCE_PATH_INSPECTION = """
UNWIND $name_ids AS name_id
MATCH (sn:StandardName {id: name_id})
OPTIONAL MATCH (imas:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
WITH sn, collect(DISTINCT 'dd:' + imas.id) AS hsn
OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
WHERE source.source_type <> 'derived' AND source.id IS NOT NULL
RETURN sn.id AS id,
       coalesce(sn.source_paths, []) AS current,
       [path IN hsn WHERE path IS NOT NULL AND path <> 'dd:'] AS hsn_paths,
       collect(DISTINCT source.id) AS produced_paths
ORDER BY id
"""


_SEMANTIC_SOURCE_PATH_WRITE = """
UNWIND $updates AS update
MATCH (sn:StandardName {id: update.id})
SET sn.source_paths = update.paths
RETURN sn.id AS id, sn.source_paths AS paths
ORDER BY id
"""


def _inspect_semantic_source_repairs(
    query: Any, source_ids: list[str]
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in query(_SEMANTIC_SOURCE_REPAIR_INSPECTION, source_ids=source_ids)
    ]


def _semantic_source_repair_plan(
    rows: list[dict[str, Any]],
    source_ids: list[str],
    authority_overrides: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["source_id"]): row for row in rows}
    missing = [source_id for source_id in source_ids if source_id not in by_id]
    if missing:
        raise ValueError("semantic sources do not exist: " + ", ".join(missing))

    unsupported = [
        source_id
        for source_id in source_ids
        if by_id[source_id].get("source_type") not in {"dd", "signals"}
    ]
    if unsupported:
        raise ValueError(
            "unsupported semantic source kinds: "
            + ", ".join(
                f"{source_id}={by_id[source_id].get('source_type')!r}"
                for source_id in unsupported
            )
        )

    ineligible = [
        source_id
        for source_id in source_ids
        if by_id[source_id].get("status") not in {"composed", "attached"}
    ]
    if ineligible:
        raise ValueError(
            "semantic sources are not composed or attached: " + ", ".join(ineligible)
        )

    backing_errors: list[str] = []
    for source_id in source_ids:
        row = by_id[source_id]
        semantic_id = row.get("semantic_id")
        dd_backings = sorted(row.get("dd_backings") or [])
        signal_backings = sorted(row.get("signal_backings") or [])
        backing_owner_ids = sorted(row.get("backing_owner_ids") or [])
        if row["source_type"] == "dd":
            valid = (
                semantic_id is not None
                and source_id == f"dd:{semantic_id}"
                and dd_backings == [semantic_id]
                and not signal_backings
            )
        else:
            valid = (
                semantic_id is not None
                and source_id == f"signals:{semantic_id}"
                and signal_backings == [semantic_id]
                and not dd_backings
            )
        exclusively_owned = backing_owner_ids == [source_id]
        if not valid or not exclusively_owned:
            backing_errors.append(
                f"{source_id}: semantic_id={semantic_id!r}, "
                f"dd={dd_backings!r}, signals={signal_backings!r}, "
                f"owners={backing_owner_ids!r}"
            )
    if backing_errors:
        raise ValueError(
            "semantic source backing identity or ownership is invalid: "
            + "; ".join(backing_errors)
        )

    planned: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    already_clean: list[dict[str, Any]] = []
    for source_id in source_ids:
        row = by_id[source_id]
        produced_targets = sorted(row.get("produced_targets") or [])
        live_targets = sorted(row.get("live_targets") or [])
        target_states = sorted(
            (dict(state) for state in row.get("target_states") or []),
            key=lambda state: str(state["id"]),
        )
        mapped_ids = sorted(row.get("mapped_ids") or [])
        state_by_id = {state["id"]: state for state in target_states}
        scalar = row.get("produced_sn_id")
        before = {
            "produced_targets": produced_targets,
            "live_targets": live_targets,
            "target_states": target_states,
            "produced_sn_id": scalar,
            "mapped_ids": mapped_ids,
        }
        base = {
            "source_id": source_id,
            "semantic_id": row["semantic_id"],
            "source_type": row["source_type"],
            "status": row["status"],
            "backing_id": (
                sorted(row.get("dd_backings") or [])[0]
                if row["source_type"] == "dd"
                else sorted(row.get("signal_backings") or [])[0]
            ),
            "backing_owner_ids": sorted(row.get("backing_owner_ids") or []),
            "before": before,
        }
        override = authority_overrides.get(source_id)
        if override is not None:
            if live_targets.count(override) != 1:
                raise ValueError(
                    "semantic source authority override is not exactly one current "
                    f"live target: {source_id}={override!r}"
                )
            target = override
            authority_basis = "explicit_authority_override"
        elif len(live_targets) == 1:
            target = live_targets[0]
            authority_basis = "sole_live_edge"
        elif len(live_targets) > 1 and scalar in live_targets:
            scalar_state = state_by_id.get(scalar, {})
            scalar_is_accepted_valid = (
                scalar_state.get("stage") == "accepted"
                and scalar_state.get("validation_status") == "valid"
            )
            accepted_valid_competitors = [
                name
                for name in live_targets
                if name != scalar
                and state_by_id.get(name, {}).get("stage") == "accepted"
                and state_by_id.get(name, {}).get("validation_status") == "valid"
            ]
            if accepted_valid_competitors and not scalar_is_accepted_valid:
                ambiguous.append(
                    {
                        **base,
                        "classification": "policy_conflict",
                        "reason": (
                            "produced_sn_id selects a lower-authority live target "
                            "while another live target is accepted and valid"
                        ),
                        "accepted_valid_competitors": accepted_valid_competitors,
                        "after": before,
                    }
                )
                continue
            target = str(scalar)
            authority_basis = "lifecycle_compatible_scalar"
        else:
            ambiguous.append(
                {
                    **base,
                    "classification": "ambiguous",
                    "reason": (
                        "no sole live edge and produced_sn_id does not select "
                        "exactly one live target"
                    ),
                    "after": before,
                }
            )
            continue

        after = {
            "produced_targets": [target],
            "live_targets": [target],
            "target_states": ([state_by_id[target]] if target in state_by_id else []),
            "produced_sn_id": target,
            "mapped_ids": [target],
        }
        item = {
            **base,
            "authoritative_target": target,
            "authority_basis": authority_basis,
            "removed_targets": [name for name in produced_targets if name != target],
            "after": after,
        }
        if before == after:
            already_clean.append({**item, "classification": "already_clean"})
        else:
            planned.append({**item, "classification": "planned"})
    return planned, ambiguous, already_clean


def repair_semantic_source_invariants(
    gc: Any,
    source_ids: list[str],
    *,
    reason: str,
    dry_run: bool = True,
    authority_overrides: Mapping[str, str] | None = None,
    origin: str = "semantic_source_repair",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Repair current-target mirrors for an explicit semantic-source allowlist.

    A sole live ``PRODUCED_NAME`` edge is authoritative. If several live edges
    exist, the scalar mirror may select one; otherwise the source is reported as
    ambiguous and remains untouched. Unsupported or structurally ambiguous
    backing projections reject the whole request before mutation.

    Applying re-reads and compare-checks every planned row inside one explicit
    transaction. Any drift or write failure rolls back the complete batch.
    ``source_paths`` is then rebuilt for every affected name from all surviving
    graph bindings, including sources outside the requested allowlist.
    """
    selected = sorted(set(source_ids))
    if not selected:
        return {
            "dry_run": dry_run,
            "source_ids": [],
            "planned": [],
            "repaired": [],
            "ambiguous": [],
            "already_clean": [],
        }
    if not reason.strip():
        raise ValueError("semantic source repair requires a non-empty reason")
    overrides = dict(authority_overrides or {})
    unexpected_overrides = sorted(set(overrides) - set(selected))
    if unexpected_overrides:
        raise ValueError(
            "semantic source authority overrides are outside the exact scope: "
            + ", ".join(unexpected_overrides)
        )

    if dry_run:
        rows = _inspect_semantic_source_repairs(gc.query, selected)
        planned, ambiguous, already_clean = _semantic_source_repair_plan(
            rows, selected, overrides
        )
        return {
            "dry_run": True,
            "source_ids": selected,
            "planned": planned,
            "repaired": [],
            "ambiguous": ambiguous,
            "already_clean": already_clean,
        }

    session_factory = getattr(gc, "session", None)
    if not callable(session_factory):
        raise TypeError("semantic source repair requires a transactional graph client")

    with session_factory() as session:
        transaction = session.begin_transaction()
        try:

            def tx_query(cypher: str, **params: Any) -> Any:
                return transaction.run(cypher, **params)

            rows = _inspect_semantic_source_repairs(tx_query, selected)
            planned, ambiguous, already_clean = _semantic_source_repair_plan(
                rows, selected, overrides
            )
            repaired: list[dict[str, Any]] = []
            affected_names: set[str] = set()
            for item in planned:
                before = item["before"]
                audit_detail = {
                    "source_id": item["source_id"],
                    "semantic_id": item["semantic_id"],
                    "backing_id": item["backing_id"],
                    "backing_owner_ids": item["backing_owner_ids"],
                    "before": before,
                    "after": item["after"],
                    "removed_targets": item["removed_targets"],
                    "authority_basis": item["authority_basis"],
                }
                change_id = f"sn-change:{uuid4()}"
                mutation_rows = [
                    dict(row)
                    for row in transaction.run(
                        _SEMANTIC_SOURCE_REPAIR_MUTATION,
                        source_id=item["source_id"],
                        semantic_id=item["semantic_id"],
                        source_type=item["source_type"],
                        status=item["status"],
                        before_scalar=before["produced_sn_id"],
                        before_targets=before["produced_targets"],
                        before_target_states=before["target_states"],
                        before_live_targets=before["live_targets"],
                        dd_backings=(
                            [item["backing_id"]] if item["source_type"] == "dd" else []
                        ),
                        signal_backings=(
                            [item["backing_id"]]
                            if item["source_type"] == "signals"
                            else []
                        ),
                        before_mapped_ids=before["mapped_ids"],
                        backing_owner_ids=item["backing_owner_ids"],
                        target=item["authoritative_target"],
                        change_id=change_id,
                        audit_reason=(
                            reason.strip()
                            + "; exact source binding repair "
                            + json.dumps(
                                audit_detail,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                        origin=origin,
                        run_id=run_id,
                    )
                ]
                if len(mutation_rows) != 1:
                    raise RuntimeError(
                        "semantic source changed during repair: " + item["source_id"]
                    )
                repaired.append(
                    {
                        **item,
                        "classification": "repaired",
                        "change_id": mutation_rows[0]["change_id"],
                    }
                )
                affected_names.update(before["produced_targets"])
                affected_names.add(item["authoritative_target"])

            if affected_names:
                path_rows = [
                    dict(row)
                    for row in transaction.run(
                        _SEMANTIC_SOURCE_PATH_INSPECTION,
                        name_ids=sorted(affected_names),
                    )
                ]
                updates = []
                for path_row in path_rows:
                    current = list(path_row.get("current") or [])
                    derived_keep = [
                        path for path in current if path.startswith("derived:")
                    ]
                    canonical_paths = sorted(
                        set(derived_keep)
                        | set(path_row.get("hsn_paths") or [])
                        | set(path_row.get("produced_paths") or [])
                    )
                    updates.append({"id": path_row["id"], "paths": canonical_paths})
                if len(updates) != len(affected_names):
                    raise RuntimeError(
                        "affected StandardName disappeared during repair"
                    )
                written_paths = [
                    dict(row)
                    for row in transaction.run(
                        _SEMANTIC_SOURCE_PATH_WRITE,
                        updates=updates,
                    )
                ]
                if written_paths != updates:
                    raise RuntimeError(
                        "affected StandardName source_paths changed during repair"
                    )
            after_rows = _inspect_semantic_source_repairs(tx_query, selected)
            after_planned, after_ambiguous, after_clean = _semantic_source_repair_plan(
                after_rows, selected, overrides
            )
            if after_planned:
                raise RuntimeError(
                    "semantic source repair did not converge: "
                    + ", ".join(item["source_id"] for item in after_planned)
                )
            if after_ambiguous != ambiguous:
                raise RuntimeError(
                    "ambiguous semantic source set changed during repair"
                )
            expected_clean = sorted(
                [item["source_id"] for item in repaired]
                + [item["source_id"] for item in already_clean]
            )
            if [item["source_id"] for item in after_clean] != expected_clean:
                raise RuntimeError("semantic source clean set changed during repair")
            after_clean_by_id = {item["source_id"]: item for item in after_clean}
            for clean_item in already_clean:
                if after_clean_by_id.get(clean_item["source_id"]) != clean_item:
                    raise RuntimeError(
                        "unmodified semantic source changed during repair: "
                        + clean_item["source_id"]
                    )
            transaction.commit()
        except BaseException:
            with suppress(Exception):
                transaction.rollback()
            raise

    return {
        "dry_run": False,
        "source_ids": selected,
        "planned": planned,
        "repaired": repaired,
        "ambiguous": after_ambiguous,
        "already_clean": already_clean,
    }


def record_standard_name_change(
    gc: Any,
    from_name: str,
    to_name: str,
    *,
    operation: str,
    reason: str | None = None,
    origin: str | None = None,
    run_id: str | None = None,
) -> str:
    """Persist a non-indexed internal event without making it a StandardName."""
    change_id = f"sn-change:{uuid4()}"
    gc.query(
        """
        CREATE (change:StandardNameChange {
          id: $id, from_name: $from_name, to_name: $to_name,
          operation: $operation, reason: $reason, origin: $origin,
          run_id: $run_id, changed_at: datetime($changed_at), internal: true
        })
        WITH change
        OPTIONAL MATCH (target:StandardName {id: $to_name})
        FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END |
          MERGE (target)-[:HAS_INTERNAL_CHANGE]->(change))
        """,
        id=change_id,
        from_name=from_name,
        to_name=to_name,
        operation=operation,
        reason=reason,
        origin=origin,
        run_id=run_id,
        changed_at=datetime.now(UTC).isoformat(),
    )
    return change_id


def retire_unrecoverable_provenance_orphans(
    gc: Any,
    name_ids: list[str],
    *,
    include_accepted: bool = False,
) -> list[str]:
    """Delete a reviewed set of source-less names with atomic ledger records.

    Every target must still have no ``PRODUCED_NAME`` source at mutation time.
    Accepted names require an explicit opt-in. The function is deliberately
    list-scoped: it cannot widen into an unbounded graph cleanup.
    """
    if not name_ids:
        return []
    rows = [
        dict(row)
        for row in gc.query(
            """
            UNWIND $ids AS id
            MATCH (sn:StandardName {id: id})
            WHERE NOT EXISTS {
              MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            }
            RETURN sn.id AS id, sn.name_stage AS stage
            ORDER BY id
            """,
            ids=sorted(set(name_ids)),
        )
    ]
    found = {row["id"] for row in rows}
    missing_or_sourced = sorted(set(name_ids) - found)
    if missing_or_sourced:
        raise ValueError(
            "targets are missing or no longer provenance orphans: "
            + ", ".join(missing_or_sourced)
        )
    accepted = sorted(row["id"] for row in rows if row.get("stage") == "accepted")
    if accepted and not include_accepted:
        raise ValueError(
            "accepted provenance orphans require include_accepted=True: "
            + ", ".join(accepted)
        )

    deletion_clause = deletion_change_cypher("sn")
    deletion_params = deletion_change_params(
        "remove_provenance_orphan",
        reason="no recoverable DD, signal, catalog, scalar, structural, or history source",
        origin="provenance_recovery",
    )
    retired: list[str] = []
    for name_id in sorted(found):
        result = list(
            gc.query(
                f"""
                MATCH (sn:StandardName {{id: $name_id}})
                WHERE NOT EXISTS {{
                  MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                }}
                OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
                OPTIONAL MATCH (sn)-[:DOCS_REVISION_OF]->(revision:DocsRevision)
                WITH sn, collect(DISTINCT review) AS reviews,
                     collect(DISTINCT revision) AS revisions
                {deletion_clause}
                FOREACH (item IN reviews | DETACH DELETE item)
                FOREACH (item IN revisions | DETACH DELETE item)
                DETACH DELETE sn
                RETURN $name_id AS id
                """,
                name_id=name_id,
                **deletion_params,
            )
        )
        if result:
            retired.append(name_id)
    return retired


def classify_missing_change_targets(gc: Any) -> dict[str, Any]:
    """Classify durable change rows whose target StandardName is absent.

    Deletion events intentionally outlive their target and are explained by a
    known mechanism operation. Older rows with a missing target and any other
    operation are retained as ``legacy_unexplained`` for investigation. This
    report is read-only and never removes history.
    """
    rows = [
        dict(row)
        for row in gc.query(
            """
            MATCH (change:StandardNameChange)
            WHERE change.to_name IS NOT NULL
              AND NOT EXISTS {
                MATCH (target:StandardName)
                WHERE target.id = change.to_name
              }
            RETURN change.id AS id,
                   change.from_name AS from_name,
                   change.to_name AS to_name,
                   change.operation AS operation,
                   CASE
                     WHEN change.operation IN $deletion_operations
                     THEN 'explained_deletion'
                     ELSE 'legacy_unexplained'
                   END AS classification
            ORDER BY change.id
            """,
            deletion_operations=sorted(DELETION_OPERATIONS),
        )
    ]
    return {
        "total": len(rows),
        "explained_deletion": sum(
            row["classification"] == "explained_deletion" for row in rows
        ),
        "legacy_unexplained": sum(
            row["classification"] == "legacy_unexplained" for row in rows
        ),
        "rows": rows,
    }


def compact_unapproved_superseded(
    gc: Any,
    *,
    names: list[str] | None = None,
    apply: bool = False,
) -> list[dict[str, Any]]:
    """Plan or compact safe unapproved superseded candidates.

    The default is a read-only manifest. Applying compacts only rows with one
    live tip: semantic sources are retargeted, a lightweight internal event is
    retained, then the obsolete StandardName and its owned review/doc snapshots
    are removed. Ambiguous/dead-end rows always remain for manual resolution.
    When *names* is provided, only those exact candidate ids are considered.
    """
    selected_names = list(dict.fromkeys(names)) if names else None
    rows = gc.query(
        """
        MATCH (old:StandardName)
        WHERE old.name_stage = 'superseded'
          AND old.catalog_approved_at IS NULL
          AND ($names IS NULL OR old.id IN $names)
        OPTIONAL MATCH (tip:StandardName)-[:REFINED_FROM*1..]->(old)
        WHERE NOT tip.name_stage IN ['superseded', 'exhausted']
        WITH old, collect(DISTINCT tip.id) AS tips
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(old)
        RETURN old.id AS name, old.superseded_from_stage AS prior_stage,
               tips, count(DISTINCT source) AS source_count,
               size(tips) = 1 AS safe_to_compact
        ORDER BY old.id
        """,
        names=selected_names,
    )
    manifest = [dict(row) for row in rows or []]
    if not apply:
        return manifest
    for item in manifest:
        tips = item.get("tips") or []
        if not item.get("safe_to_compact") or len(tips) != 1:
            continue
        old_name = item["name"]
        target = tips[0]
        retarget_standard_name_sources(
            gc,
            old_name,
            target,
            operation="retarget_compacted_name",
            record_change=False,
        )
        deletion_clause = deletion_change_cypher("old")
        deleted = gc.query(
            f"""MATCH (old:StandardName {{id: $old_name}})
            WHERE old.name_stage = 'superseded'
              AND old.catalog_approved_at IS NULL
            OPTIONAL MATCH (old)-[:HAS_REVIEW]->(review:StandardNameReview)
            OPTIONAL MATCH (old)-[:DOCS_REVISION_OF]->(revision:DocsRevision)
            WITH old, collect(DISTINCT review) AS reviews,
                 collect(DISTINCT revision) AS revisions
            {deletion_clause}
            FOREACH (item IN reviews | DETACH DELETE item)
            FOREACH (item IN revisions | DETACH DELETE item)
            DETACH DELETE old
            RETURN 1 AS deleted""",
            old_name=old_name,
            **deletion_change_params(
                "compact_unapproved_name",
                reason="unapproved superseded candidate compacted after source retarget",
            ),
        )
        item["compacted"] = bool(deleted)
    return manifest


def trace_standard_name_provenance(
    gc: Any,
    name: str,
    *,
    include_reviews: bool = False,
    max_depth: int = 10,
) -> dict[str, Any]:
    """Return explicitly requested semantic sources and internal history."""
    semantic_sources = fetch_public_semantic_sources(gc, name)
    change_rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name})-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
        RETURN change.from_name AS from_name, change.to_name AS to_name,
               change.operation AS operation, change.reason AS reason,
               change.origin AS origin, change.changed_at AS changed_at
        ORDER BY change.changed_at DESC LIMIT $limit
        """,
        name=name,
        limit=max(1, min(int(max_depth), 100)),
    )
    result: dict[str, Any] = {
        "name": name,
        "semantic_sources": semantic_sources,
        "internal_changes": [dict(row) for row in change_rows or []],
    }
    if include_reviews:
        reviews = gc.query(
            """
            MATCH (sn:StandardName {id: $name})-[:HAS_REVIEW]->(review:StandardNameReview)
            RETURN review.review_axis AS axis, review.score AS score,
                   review.tier AS tier, review.reviewed_at AS reviewed_at
            ORDER BY review.reviewed_at DESC LIMIT $limit
            """,
            name=name,
            limit=max(1, min(int(max_depth), 100)),
        )
        result["reviews"] = [dict(row) for row in reviews or []]
    return result

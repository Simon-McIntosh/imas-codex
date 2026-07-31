"""Standard-name semantic provenance and internal change-history operations.

Semantic sources describe *which DD path or signal supports the current name*.
They are distinct from pipeline history (discarded candidates, reviews, edits,
and runs).  All name-changing routes use the retarget operation here so the
source ledger has one current target while lightweight change events can retain
an internal audit trail after unapproved candidates are compacted.
"""

from __future__ import annotations

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
        raise RuntimeError("staged rename changed state before cancellation")
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

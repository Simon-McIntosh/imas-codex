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

_DD_DOCS_ROOT = "https://imas-data-dictionary.readthedocs.io/en"


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
        WHERE (sns)-[:PRODUCED_NAME]->(old) OR sns.produced_sn_id = $old_name
        WITH new, old, collect(DISTINCT sns) AS sources
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
        WITH new, old, collect(DISTINCT source) AS moved,
             collect(DISTINCT CASE WHEN dd IS NULL THEN null ELSE 'dd:' + dd.id END) +
             collect(DISTINCT CASE WHEN signal IS NULL THEN null ELSE signal.id END)
             AS derived_paths
        WITH new, moved,
             coalesce(new.source_paths, []) + coalesce(old.source_paths, []) +
             [p IN derived_paths WHERE p IS NOT NULL] AS paths
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
        OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
        FOREACH (_ IN CASE WHEN review IS NULL THEN [] ELSE [1] END |
          SET review.standard_name_id = sn.id)
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


def build_unapproved_cleanup_manifest(gc: Any) -> list[dict[str, Any]]:
    """Return the dry-run manifest for compacting unapproved dead candidates."""
    rows = gc.query(
        """
        MATCH (old:StandardName)
        WHERE old.name_stage = 'superseded'
          AND old.catalog_approved_at IS NULL
        OPTIONAL MATCH (tip:StandardName)-[:REFINED_FROM*1..]->(old)
        WHERE NOT tip.name_stage IN ['superseded', 'exhausted']
        WITH old, collect(DISTINCT tip.id) AS tips
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(old)
        RETURN old.id AS name, old.superseded_from_stage AS prior_stage,
               tips, count(DISTINCT source) AS source_count,
               size(tips) = 1 AS safe_to_compact
        ORDER BY old.id
        """
    )
    return [dict(row) for row in rows or []]


def compact_unapproved_superseded(
    gc: Any,
    *,
    apply: bool = False,
) -> list[dict[str, Any]]:
    """Plan or compact safe unapproved superseded candidates.

    The default is a read-only manifest. Applying compacts only rows with one
    live tip: semantic sources are retargeted, a lightweight internal event is
    retained, then the obsolete StandardName and its owned review/doc snapshots
    are removed. Ambiguous/dead-end rows always remain for manual resolution.
    """
    manifest = build_unapproved_cleanup_manifest(gc)
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
            operation="compact",
        )
        gc.query(
            """MATCH (:StandardName {id: $old_name})-[:HAS_REVIEW]->(review)
            DETACH DELETE review""",
            old_name=old_name,
        )
        gc.query(
            """MATCH (:StandardName {id: $old_name})-[:DOCS_REVISION_OF]->(revision)
            DETACH DELETE revision""",
            old_name=old_name,
        )
        gc.query(
            """MATCH (old:StandardName {id: $old_name})
            WHERE old.name_stage = 'superseded'
              AND old.catalog_approved_at IS NULL
            DETACH DELETE old""",
            old_name=old_name,
        )
        item["compacted"] = True
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

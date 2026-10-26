"""Graph storage operations for the standard-name pipeline."""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.models import SNRunStatus

from .graph_foundation import _CLAIM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def query_pipeline_buckets(
    threshold: float = 0.65, cap: int = 3, cache_ttl: float = 3.0, *, services: Any
) -> Any:
    """Query the 6 disjoint pipeline buckets from the graph.

    Uses a module-level time-based cache (TTL = *cache_ttl* seconds).

    Bucket definitions (disjoint by construction):

    - **A**: StandardNameSource nodes with ``status='extracted'``.
    - **B**: StandardName with ``name_stage='drafted'`` and no
      ``reviewer_score_name``.
    - **C**: StandardName reviewed/refining, below *threshold*, under
      *cap* → definite refine + re-review.
    - **D**: StandardName accepted, docs not yet drafted.
    - **E**: StandardName with ``docs_stage='drafted'``, no
      ``reviewer_score_docs``.
    - **F**: StandardName docs reviewed/refining, below *threshold*,
      under *cap* → definite docs refine.

    Args:
        threshold: Score threshold below which a name/docs enters
            the refine pool (default 0.65).
        cap: Maximum refine rotations before exhaustion (default 3).
        cache_ttl: Cache time-to-live in seconds (default 3.0).

    Returns:
        :class:`PipelineBuckets` with the 6 counts.
    """
    import time as _time

    from imas_codex.standard_names.cost_model import PipelineBuckets

    now = _time.time()
    if (
        services._pipeline_buckets_cache["value"] is not None
        and now - services._pipeline_buckets_cache["ts"] < cache_ttl
    ):
        return services._pipeline_buckets_cache["value"]

    with services.GraphClient() as gc:
        # Bucket A: pre-draft sources
        r_a = gc.query(
            "MATCH (sns:StandardNameSource {status: 'extracted'}) "
            "RETURN count(*) AS cnt"
        )
        a = r_a[0]["cnt"] if r_a else 0

        # Bucket B: drafted, not yet name-reviewed
        r_b = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage = 'drafted' "
            "AND sn.reviewer_score_name IS NULL "
            "RETURN count(*) AS cnt"
        )
        b = r_b[0]["cnt"] if r_b else 0

        # Bucket C: reviewed/refining, below threshold, under cap
        r_c = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage IN ['reviewed', 'refining'] "
            "AND coalesce(sn.reviewer_score_name, 0.0) < $threshold "
            "AND coalesce(sn.chain_length, 0) < $cap "
            "RETURN count(*) AS cnt",
            threshold=threshold,
            cap=cap,
        )
        c = r_c[0]["cnt"] if r_c else 0

        # Bucket D: name accepted AND name-reviewed, docs not yet drafted.
        # Mirrors the claim_generate_docs_batch hard gate — a name must carry a
        # real name-review score (reviewer_score_name) before docs are eligible,
        # so derived parents auto-accepted without review are excluded from the
        # generate_docs backlog until REVIEW_NAME has scored them.
        r_d = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage = 'accepted' "
            "AND sn.reviewer_score_name IS NOT NULL "
            "AND (sn.docs_stage IS NULL OR sn.docs_stage = 'pending') "
            "RETURN count(*) AS cnt"
        )
        d = r_d[0]["cnt"] if r_d else 0

        # Bucket E: docs drafted, not yet reviewed
        r_e = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.docs_stage = 'drafted' "
            "AND sn.reviewer_score_docs IS NULL "
            "RETURN count(*) AS cnt"
        )
        e = r_e[0]["cnt"] if r_e else 0

        # Bucket F: docs reviewed/refining, below threshold, under cap
        # Schema note: plan.md referenced refine_docs_count but actual
        # schema field is docs_chain_length.
        r_f = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.docs_stage IN ['reviewed', 'refining'] "
            "AND coalesce(sn.reviewer_score_docs, 0.0) < $threshold "
            "AND coalesce(sn.docs_chain_length, 0) < $cap "
            "RETURN count(*) AS cnt",
            threshold=threshold,
            cap=cap,
        )
        f = r_f[0]["cnt"] if r_f else 0

    result = PipelineBuckets(
        a_sources=a,
        b_drafted_unreviewed=b,
        c_refine_pending=c,
        d_accepted_no_docs=d,
        e_docs_unreviewed=e,
        f_refine_docs_pending=f,
    )
    services._pipeline_buckets_cache["value"] = result
    services._pipeline_buckets_cache["ts"] = now
    return result


def query_historical_cpi(
    model: str | None = None, *, services: Any
) -> dict[str, float]:
    """Average per-phase LLM cost across past StandardName nodes.

    Returns dict keyed by pool name (``generate_name``, ``review_name``,
    ``refine_name``, ``generate_docs``, ``review_docs``, ``refine_docs``).
    Missing keys = no historical data for that pool.

    Cached for the lifetime of the process (historical data is static
    within a run).

    Args:
        model: Optional LLM model filter. If provided, restricts to
            nodes composed with that model.
    """
    import time as _time

    # Process-lifetime cache (historical data is static within a run)
    if services._historical_cpi_cache["value"] is not None:
        return services._historical_cpi_cache["value"]

    result: dict[str, float] = {}
    # Pool → (cost_field, count_field) — direct 1:1 mapping.
    _pool_fields = {
        "generate_name": ("llm_cost_generate_name", "generate_name_count"),
        "review_name": ("llm_cost_review_name", "review_name_count"),
        "refine_name": ("llm_cost_refine_name", "refine_name_count"),
        "generate_docs": ("llm_cost_generate_docs", "generate_docs_count"),
        "review_docs": ("llm_cost_review_docs", "review_docs_count"),
        "refine_docs": ("llm_cost_refine_docs", "refine_docs_count"),
    }
    with services.GraphClient() as gc:
        for pool, (cost_f, count_f) in _pool_fields.items():
            rows = gc.query(
                f"MATCH (sn:StandardName) "
                f"WHERE sn.{cost_f} IS NOT NULL "
                f"AND sn.{cost_f} > 0 "
                f"AND coalesce(sn.{count_f}, 0) > 0 "
                f"RETURN avg(sn.{cost_f} / sn.{count_f}) AS cpi, "
                f"count(*) AS n"
            )
            if rows and rows[0]["n"] > 0 and rows[0]["cpi"] is not None:
                result[pool] = float(rows[0]["cpi"])

    services._historical_cpi_cache["value"] = result
    services._historical_cpi_cache["ts"] = _time.time()
    return result


def get_extraction_candidates_dd(
    ids_filter: str | None = None,
    domain_filter: str | None = None,
    limit: int = 500,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Query IMASNode paths grouped by semantic cluster.

    Returns dynamic leaf nodes that have been enriched (status=embedded),
    optionally filtered by IDS or physics domain.
    """
    with services.GraphClient() as gc:
        params: dict[str, Any] = {"limit": limit}
        where_clauses = [
            "n.node_type IN ['dynamic', 'constant']",
            "n.description IS NOT NULL",
            "n.description <> ''",
            # DD-version gate: nodes absent from the current DD never seed.
            "coalesce(n.lifecycle_status, '') <> 'removed'",
        ]

        if ids_filter:
            where_clauses.append("ids.id = $ids_filter")
            params["ids_filter"] = ids_filter
        if domain_filter:
            where_clauses.append("n.physics_domain = $domain_filter")
            params["domain_filter"] = domain_filter

        where = " AND ".join(where_clauses)
        results = gc.query(
            f"""
            MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
            WHERE {where}
            WITH n, ids
            OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            RETURN n.id AS path, n.description AS description,
                   n.unit AS unit, n.data_type AS data_type,
                   ids.id AS ids_name, c.label AS cluster_label
            ORDER BY ids.id, n.id
            LIMIT $limit
            """,
            **params,
        )
        from imas_codex.settings import get_dd_version
        from imas_codex.standard_names.dd_resolutions import resolve_dd_rows

        batch = resolve_dd_rows(results, dd_version=get_dd_version())
        if batch.refusals:
            logger.warning(
                "Extraction candidates omitted %d changed DD fields: %s",
                len(batch.refusals),
                ", ".join(
                    f"{refusal.path}:{refusal.field.value}"
                    for refusal in batch.refusals
                ),
            )
        return [
            {
                **results[resolved.row_index],
                **resolved.context.as_pipeline_item(),
            }
            for resolved in batch.resolved
        ]


def get_extraction_candidates_signals(
    facility: str, domain_filter: str | None = None, limit: int = 500, *, services: Any
) -> list[dict[str, Any]]:
    """Query FacilitySignal nodes for a given facility.

    Returns signals that have been enriched, optionally filtered by
    physics domain.
    """
    with services.GraphClient() as gc:
        params: dict[str, Any] = {"facility": facility, "limit": limit}
        where_clauses = ["s.status = 'enriched'"]

        if domain_filter:
            where_clauses.append("s.physics_domain = $domain_filter")
            params["domain_filter"] = domain_filter

        where = " AND ".join(where_clauses)
        results = gc.query(
            f"""
            MATCH (s:FacilitySignal)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE {where}
            WITH s
            OPTIONAL MATCH (s)-[:MAPS_TO]->(m:IMASNode)
            RETURN s.id AS signal_id, s.description AS description,
                   s.physics_domain AS physics_domain,
                   s.unit AS units,
                   m.id AS imas_path
            ORDER BY s.id
            LIMIT $limit
            """,
            **params,
        )
        return list(results)


def get_existing_standard_names(gc: Any | None = None, *, services: Any) -> set[str]:
    """Return the set of existing StandardName node IDs for deduplication.

    This is the single definition of the graph-wide all-name read: callers that
    would otherwise restate ``MATCH (sn:StandardName) RETURN sn.id AS id`` read
    it here instead. A caller already holding an open client passes it as *gc*
    so the read rides that session; with none, one is opened for this read.
    """
    if gc is not None:
        results = gc.query("MATCH (sn:StandardName) RETURN sn.id AS id")
        return {r["id"] for r in results}
    with services.GraphClient() as own:
        results = own.query("MATCH (sn:StandardName) RETURN sn.id AS id")
        return {r["id"] for r in results}


def get_named_source_ids(*, services: Any) -> set[str]:
    """Return source IDs already linked via HAS_STANDARD_NAME.

    Used for resumability: extract skips sources that already have
    a standard name unless --force is specified.
    """
    with services.GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN DISTINCT src.id AS source_id
        """)
        return {r["source_id"] for r in results}


def partition_focus_by_accepted(
    paths: list[str], *, gc: Any | None = None, services: Any
) -> tuple[list[str], list[str]]:
    """Split focus DD paths into ``(gap, accepted)`` by live accepted name.

    A path is *accepted* when its ``StandardNameSource`` (id ``dd:<path>``)
    has PRODUCED a ``StandardName`` whose ``name_stage`` is ``accepted`` or
    ``approved`` — i.e. it already carries a live, catalog-bound name. Gap
    paths lack any such name and are the ones focused mop-up should (re)stage;
    accepted paths are left untouched so a focus run never churns names the
    catalog already holds. Input order is preserved in both lists.

    *gc* may be an already-open graph client (reused for the query); when None a
    fresh :class:`GraphClient` is opened.
    """
    if not paths:
        return [], []
    ids = [f"dd:{p}" for p in paths]
    cypher = """
        UNWIND $ids AS sid
        MATCH (sns:StandardNameSource {id: sid})-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE sn.name_stage IN ['accepted', 'approved']
        RETURN DISTINCT sid AS sid
        """
    if gc is not None:
        rows = gc.query(cypher, ids=ids)
    else:
        with services.GraphClient() as _gc:
            rows = _gc.query(cypher, ids=ids)
    accepted_ids = {r["sid"] for r in (rows or [])}
    gap = [p for p in paths if f"dd:{p}" not in accepted_ids]
    accepted = [p for p in paths if f"dd:{p}" in accepted_ids]
    return gap, accepted


MANIFEST_DRAIN_DISPOSITIONS: tuple[str, ...] = (
    "accepted",
    "active_in_flight",
    "genuine_gap",
    "non_nameable",
    "ambiguous",
)


def canonicalize_manifest_drain_paths(paths: list[str], *, services: Any) -> list[str]:
    """Validate and de-duplicate exact DD paths without changing their order."""
    canonical: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str):
            raise ValueError("bounded drain paths must be strings")
        if raw.strip().startswith("dd:"):
            raise ValueError(f"bounded drain accepts DD paths, not source IDs: {raw!r}")
        if raw.strip().split(":", 1)[0] in {
            "catalog",
            "derived",
            "signal",
            "signals",
        }:
            raise ValueError(f"bounded drain accepts only canonical DD paths: {raw!r}")
        path = raw.strip().strip("/")
        if (
            not path
            or path != raw.strip()
            or "*" in path
            or "?" in path
            or any(char.isspace() for char in path)
            or ":" in path
        ):
            raise ValueError(f"bounded drain requires an exact DD path: {raw!r}")
        if path not in seen:
            seen.add(path)
            canonical.append(path)
    if not canonical:
        raise ValueError("bounded drain manifest resolved to zero exact DD paths")
    return canonical


def classify_manifest_drain_item(
    item: dict[str, Any], *, services: Any
) -> dict[str, Any]:
    """Return one immutable five-way disposition from a manifest plan row."""
    reasons: list[str] = []
    node = item.get("node") or {}
    source = item.get("source")
    live_names = item.get("live_names") or []
    all_name_ids = set(item.get("all_name_ids") or [])
    parents = item.get("parents") or []
    path = item["path"]
    dd_version = item["dd_version"]

    if not item.get("configured_version_present"):
        reasons.append("configured DD version is not the unique current version")
    if not node.get("id") or node.get("id") != path:
        reasons.append("exact IMAS identity is missing")
    if node.get("node_category") is None:
        reasons.append("IMAS node category is missing")

    if source:
        unit_ids = node.get("unit_ids") or []
        dd_parents = node.get("dd_parents") or []
        authoritative_unit = node.get("unit") or (unit_ids[0] if unit_ids else None)
        if (
            source.get("id") != f"dd:{path}"
            or source.get("source_type") != "dd"
            or source.get("source_id") != path
            or source.get("dd_version") != dd_version
            or source.get("dd_snapshot_pinned") is not True
            or sorted(source.get("dd_target_ids") or []) != [path]
        ):
            reasons.append("exact source identity or immutable DD pin is inconsistent")
        if (
            len(unit_ids) > 1
            or len(dd_parents) > 1
            or (
                node.get("unit") is not None
                and unit_ids
                and unit_ids != [node.get("unit")]
            )
        ):
            reasons.append("DD authority relationship cardinality is ambiguous")
        elif (
            source.get("description") != node.get("documentation")
            or source.get("dd_documentation") != node.get("documentation")
            or source.get("dd_data_type") != node.get("data_type")
            or source.get("dd_unit") != authoritative_unit
            or sorted(source.get("dd_coordinates") or [])
            != sorted(node.get("coordinate_ids") or [])
            or source.get("dd_lifecycle_status") != node.get("lifecycle_status")
            or source.get("dd_lifecycle_version") != node.get("lifecycle_version")
            or source.get("enhanced_description") != node.get("description")
            or source.get("enhancement_kind") != node.get("enrichment_source")
            or source.get("physics_domain") != node.get("physics_domain")
            or source.get("dd_parent_path")
            != (dd_parents[0].get("id") if dd_parents else None)
            or source.get("dd_parent_documentation")
            != (dd_parents[0].get("documentation") if dd_parents else None)
        ):
            reasons.append("immutable DD snapshot no longer matches authority")
        produced_sn_id = source.get("produced_sn_id")
        if produced_sn_id and produced_sn_id not in all_name_ids:
            reasons.append("produced_sn_id has no matching PRODUCED_NAME edge")
        if live_names and produced_sn_id != live_names[0].get("id"):
            reasons.append("produced_sn_id does not identify the live name binding")
        if source.get("scope_conflict"):
            reasons.append("source is held by a non-stale competing drain scope")

    if len(live_names) > 1:
        reasons.append("source has multiple live PRODUCED_NAME bindings")
    if any(name.get("scope_conflict") for name in live_names):
        reasons.append("live name is held by a non-stale competing drain scope")
    if any(parent.get("scope_conflict") for parent in parents):
        reasons.append("derived parent is held by a non-stale competing drain scope")
    if reasons:
        disposition = "ambiguous"
    else:
        from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES

        non_nameable = node.get("node_category") not in SN_SOURCE_CATEGORIES or (
            source and source.get("status") == "not_physical_quantity"
        )
        if non_nameable:
            disposition = "non_nameable"
        elif live_names:
            live = live_names[0]
            disposition = (
                "accepted"
                if live.get("validation_status") == "valid"
                and live.get("name_stage") in {"accepted", "approved"}
                and live.get("docs_stage") == "accepted"
                else "active_in_flight"
            )
        elif source and source.get("worker_claim_live"):
            disposition = "active_in_flight"
        else:
            disposition = "genuine_gap"

    source_status = source.get("status") if source else None
    drain_actionable = disposition == "genuine_gap" and (
        source is None
        or (
            source_status == "extracted"
            and int(source.get("attempt_count") or 0)
            < services._MAX_COMPOSE_CLAIM_ATTEMPTS
            and not all_name_ids
        )
    )
    return {
        **item,
        "disposition": disposition,
        "ambiguity_reasons": reasons,
        "drain_actionable": drain_actionable,
    }


_MANIFEST_DRAIN_PLAN_QUERY = """
UNWIND $paths AS path
OPTIONAL MATCH (configured_version:DDVersion {id: $dd_version})
WITH path, collect(configured_version) AS configured_versions
CALL () {
  MATCH (version:DDVersion)
  RETURN [entry IN collect(properties(version))] AS version_authority
}
WITH path, configured_versions, version_authority,
     size([version IN configured_versions
           WHERE version.is_current = true]) = 1
       AND size([version IN version_authority
                 WHERE version.is_current = true]) = 1 AS configured_version_present
OPTIONAL MATCH (node:IMASNode {id: path})
OPTIONAL MATCH (source:StandardNameSource {id: 'dd:' + path})
CALL (source) {
  OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(target:IMASNode)
  RETURN [id IN collect(DISTINCT target.id) WHERE id IS NOT NULL]
         AS dd_target_ids
}
CALL (source) {
  OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(all_name:StandardName)
  RETURN [id IN collect(DISTINCT all_name.id) WHERE id IS NOT NULL]
         AS all_name_ids
}
CALL (source) {
  OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(live:StandardName)
  WHERE NOT (coalesce(live.name_stage, '') IN $terminal_stages)
  RETURN [entry IN collect(DISTINCT CASE WHEN live IS NULL THEN null ELSE {
    id: live.id,
    name_stage: live.name_stage,
    docs_stage: live.docs_stage,
    validation_status: live.validation_status,
    claim_token: live.claim_token,
    claim_seq: live.claim_seq,
    drain_scope_id: live.drain_scope_id,
    drain_scope_claimed_at_epoch_ms:
      live.drain_scope_claimed_at.epochMillis,
    scope_conflict: live.drain_scope_id IS NOT NULL
      AND live.drain_scope_id <> $drain_scope_id
      AND live.drain_scope_claimed_at IS NOT NULL
      AND live.drain_scope_claimed_at >= datetime() - duration($scope_cutoff)
  } END) WHERE entry IS NOT NULL] AS live_names
}
CALL (source) {
  OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(live:StandardName)
  WHERE NOT (coalesce(live.name_stage, '') IN $terminal_stages)
  OPTIONAL MATCH (live)-[:HAS_PARENT*1..]->(parent:StandardName)
  WHERE parent.origin = 'derived'
  RETURN [entry IN collect(DISTINCT CASE WHEN parent IS NULL THEN null ELSE {
    id: parent.id,
    name_stage: parent.name_stage,
    docs_stage: parent.docs_stage,
    validation_status: parent.validation_status,
    claim_token: parent.claim_token,
    claim_seq: parent.claim_seq,
    drain_scope_id: parent.drain_scope_id,
    drain_scope_claimed_at_epoch_ms:
      parent.drain_scope_claimed_at.epochMillis,
    scope_conflict: parent.drain_scope_id IS NOT NULL
      AND parent.drain_scope_id <> $drain_scope_id
      AND parent.drain_scope_claimed_at IS NOT NULL
      AND parent.drain_scope_claimed_at >= datetime() - duration($scope_cutoff)
  } END) WHERE entry IS NOT NULL] AS parents
}
CALL (path, node) {
  OPTIONAL MATCH (gap:DDGap {path: path})
  OPTIONAL MATCH (node)-[owned:HAS_DD_GAP]->(gap)
  RETURN [entry IN collect(DISTINCT CASE WHEN gap IS NULL THEN null ELSE {
    id: gap.id, kind: gap.kind, status: gap.status, linked: owned IS NOT NULL
  } END) WHERE entry IS NOT NULL] AS dd_gaps
}
CALL (node) {
  OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit_node:Unit)
  RETURN [id IN collect(unit_node.id) WHERE id IS NOT NULL] AS unit_ids
}
CALL (node) {
  OPTIONAL MATCH (node)-[:HAS_COORDINATE]->(coordinate)
  RETURN [id IN collect(coordinate.id) WHERE id IS NOT NULL] AS coordinate_ids
}
CALL (node) {
  OPTIONAL MATCH (node)-[:HAS_PARENT]->(dd_parent:IMASNode)
  RETURN [entry IN collect(CASE WHEN dd_parent IS NULL THEN null ELSE {
    id: dd_parent.id, documentation: dd_parent.documentation
  } END) WHERE entry IS NOT NULL] AS dd_parents
}
RETURN path, $dd_version AS dd_version,
       configured_version_present,
       version_authority,
       CASE WHEN node IS NULL THEN null ELSE {
         id: node.id, node_category: node.node_category,
         lifecycle_status: node.lifecycle_status,
         lifecycle_version: node.lifecycle_version,
         documentation: node.documentation, data_type: node.data_type,
         physics_domain: node.physics_domain, unit: node.unit,
         description: node.description,
         enrichment_source: node.enrichment_source,
         unit_ids: unit_ids, coordinate_ids: coordinate_ids,
         dd_parents: dd_parents
       } END AS node,
       CASE WHEN source IS NULL THEN null ELSE {
         id: source.id, source_type: source.source_type,
         source_id: source.source_id, status: source.status,
         claim_token: source.claim_token, claim_seq: source.claim_seq,
         worker_claim_live: source.claim_token IS NOT NULL
           AND source.claimed_at IS NOT NULL
           AND source.claimed_at >= datetime() - duration($worker_cutoff),
         attempt_count: source.attempt_count,
         produced_sn_id: source.produced_sn_id,
         dd_version: source.dd_version,
         dd_snapshot_pinned: source.dd_snapshot_pinned,
         description: source.description,
         physics_domain: source.physics_domain,
         dd_documentation: source.dd_documentation,
         dd_parent_path: source.dd_parent_path,
         dd_parent_documentation: source.dd_parent_documentation,
         dd_data_type: source.dd_data_type,
         dd_unit: source.dd_unit,
         dd_coordinates: source.dd_coordinates,
         dd_lifecycle_status: source.dd_lifecycle_status,
         dd_lifecycle_version: source.dd_lifecycle_version,
         enhanced_description: source.enhanced_description,
         enhancement_kind: source.enhancement_kind,
         drain_scope_id: source.drain_scope_id,
         drain_scope_claimed_at_epoch_ms:
           source.drain_scope_claimed_at.epochMillis,
         scope_conflict: source.drain_scope_id IS NOT NULL
           AND source.drain_scope_id <> $drain_scope_id
           AND source.drain_scope_claimed_at IS NOT NULL
           AND source.drain_scope_claimed_at >= datetime() - duration($scope_cutoff),
         dd_target_ids: dd_target_ids
       } END AS source,
       all_name_ids, live_names, parents, dd_gaps
ORDER BY path
"""


def _resolved_manifest_drain_plan(
    rows: list[dict[str, Any]], dd_version: str, *, services: Any
) -> list[dict[str, Any]]:
    """Classify manifest rows only after resolving their DD node context."""
    from imas_codex.standard_names.dd_resolutions import resolve_dd_row

    resolved_rows = []
    for raw_row in rows:
        row = dict(raw_row)
        if row.get("node"):
            node = dict(row["node"])
            context = resolve_dd_row(
                {
                    **node,
                    "path": row["path"],
                    "coordinates": node.get("coordinate_ids") or (),
                },
                dd_version=dd_version,
            )
            node.update(context.as_pipeline_item())
            row["node"] = node
        resolved_rows.append(row)
    return [services.classify_manifest_drain_item(row) for row in resolved_rows]


def build_manifest_drain_plan(
    paths: list[str],
    *,
    dd_version: str,
    drain_scope_id: str = "dry-run",
    scope_timeout_seconds: int = 600,
    worker_timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    gc: Any | None = None,
    services: Any,
) -> list[dict[str, Any]]:
    """Read and classify every exact path without issuing any graph write."""
    exact_paths = services.canonicalize_manifest_drain_paths(paths)
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        rows = client.query(
            services._MANIFEST_DRAIN_PLAN_QUERY,
            paths=exact_paths,
            dd_version=dd_version,
            drain_scope_id=drain_scope_id,
            scope_cutoff=f"PT{scope_timeout_seconds}S",
            worker_cutoff=f"PT{worker_timeout_seconds}S",
            terminal_stages=sorted(services._TERMINAL_BINDING_NAME_STAGES),
        )
        return services._resolved_manifest_drain_plan(rows, dd_version)
    finally:
        if own:
            client.close()


class ManifestDrainConflict(RuntimeError):
    """The bounded manifest changed or contains an ambiguous identity."""


def _manifest_plan_fingerprint(plan: list[dict[str, Any]], *, services: Any) -> str:
    """Return a typed, order-stable snapshot of every projected authority."""
    import json

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            normalized = [normalize(item) for item in value]
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if hasattr(value, "iso_format"):
            return {"temporal": value.iso_format()}
        if hasattr(value, "isoformat"):
            return {"temporal": value.isoformat()}
        return {"typed_repr": repr(value)}

    return json.dumps(normalize(plan), sort_keys=True, separators=(",", ":"))


def _lock_manifest_drain_authority(gc: Any, paths: list[str], *, services: Any) -> None:
    """Take write locks on the exact DD authority and its current bindings."""
    gc.query(
        """
        MATCH (version:DDVersion)
        WITH collect(version) AS versions
        UNWIND $paths AS path
        MATCH (node:IMASNode {id: path})
        OPTIONAL MATCH (node)-[:HAS_UNIT|HAS_COORDINATE|HAS_PARENT]->(authority)
        OPTIONAL MATCH (source:StandardNameSource {id: 'dd:' + path})
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(name:StandardName)
        OPTIONAL MATCH (name)-[:HAS_PARENT*1..]->(parent:StandardName)
        WITH versions, collect(DISTINCT node) + collect(DISTINCT authority)
             + collect(DISTINCT source) + collect(DISTINCT name)
             + collect(DISTINCT parent) AS nodes
        UNWIND versions + [entry IN nodes WHERE entry IS NOT NULL] AS locked
        SET locked._drain_scope_lock = true
        REMOVE locked._drain_scope_lock
        """,
        paths=paths,
    )


@retry_on_deadlock()
def claim_manifest_drain_scope(
    paths: list[str],
    *,
    dd_version: str,
    drain_scope_id: str | None = None,
    scope_timeout_seconds: int = 600,
    gc: Any | None = None,
    services: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically create absent exact sources and CAS-stamp one bounded scope."""
    exact_paths = services.canonicalize_manifest_drain_paths(paths)
    scope_id = drain_scope_id or str(services.uuid.uuid4())
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        with client.session() as session:
            tx = session.begin_transaction()
            tx_gc = services._TransactionQuery(tx)
            try:
                rows = tx_gc.query(
                    services._MANIFEST_DRAIN_PLAN_QUERY,
                    paths=exact_paths,
                    dd_version=dd_version,
                    drain_scope_id=scope_id,
                    scope_cutoff=f"PT{scope_timeout_seconds}S",
                    worker_cutoff=f"PT{services._CLAIM_TIMEOUT_SECONDS}S",
                    terminal_stages=sorted(services._TERMINAL_BINDING_NAME_STAGES),
                )
                plan = services._resolved_manifest_drain_plan(rows, dd_version)
                initial_fingerprint = services._manifest_plan_fingerprint(plan)
                services._lock_manifest_drain_authority(tx_gc, exact_paths)
                locked_rows = tx_gc.query(
                    services._MANIFEST_DRAIN_PLAN_QUERY,
                    paths=exact_paths,
                    dd_version=dd_version,
                    drain_scope_id=scope_id,
                    scope_cutoff=f"PT{scope_timeout_seconds}S",
                    worker_cutoff=f"PT{services._CLAIM_TIMEOUT_SECONDS}S",
                    terminal_stages=sorted(services._TERMINAL_BINDING_NAME_STAGES),
                )
                plan = services._resolved_manifest_drain_plan(locked_rows, dd_version)
                if services._manifest_plan_fingerprint(plan) != initial_fingerprint:
                    raise services.ManifestDrainConflict(
                        "DD authority or binding state changed during manifest claim"
                    )
                ambiguous = [
                    item for item in plan if item["disposition"] == "ambiguous"
                ]
                if ambiguous:
                    details = "; ".join(
                        f"{item['path']}: {', '.join(item['ambiguity_reasons'])}"
                        for item in ambiguous[:10]
                    )
                    raise services.ManifestDrainConflict(
                        f"bounded manifest contains {len(ambiguous)} ambiguous path(s): "
                        + details
                    )

                absent = [
                    {
                        "id": f"dd:{item['path']}",
                        "source_type": "dd",
                        "source_id": item["path"],
                        "dd_path": item["path"],
                        "batch_key": "bounded-manifest",
                        "status": "extracted",
                        "description": item["node"].get("documentation"),
                        "physics_domain": item["node"].get("physics_domain"),
                    }
                    for item in plan
                    if item["source"] is None and item["disposition"] == "genuine_gap"
                ]
                prepared = services._pin_dd_source_snapshots(
                    tx_gc, absent, default_dd_version=dd_version
                )
                if prepared:
                    created = tx_gc.query(
                        """
                        UNWIND $sources AS src
                        MATCH (version:DDVersion {id: src.dd_version, is_current: true})
                        MATCH (node:IMASNode {id: src.dd_path})
                        WHERE node.node_category = src.expected_node_category
                          AND (node.documentation = src.expected_documentation OR
                               (node.documentation IS NULL AND
                                src.expected_documentation IS NULL))
                          AND (node.lifecycle_status = src.expected_lifecycle_status OR
                               (node.lifecycle_status IS NULL AND
                                src.expected_lifecycle_status IS NULL))
                        CREATE (source:StandardNameSource {
                          id: src.id, source_type: 'dd', source_id: src.source_id,
                          batch_key: src.batch_key, status: 'extracted',
                          description: src.description,
                          physics_domain: src.physics_domain,
                          dd_version: src.dd_version,
                          dd_documentation: src.dd_documentation,
                          dd_snapshot_pinned: true,
                          dd_parent_path: src.dd_parent_path,
                          dd_parent_documentation: src.dd_parent_documentation,
                          dd_data_type: src.dd_data_type,
                          dd_unit: src.dd_unit,
                          dd_coordinates: src.dd_coordinates,
                          dd_lifecycle_status: src.dd_lifecycle_status,
                          dd_lifecycle_version: src.dd_lifecycle_version,
                          enhanced_description: src.enhanced_description,
                          enhancement_kind: src.enhancement_kind,
                          attempt_count: 0,
                          drain_scope_id: $scope_id,
                          drain_scope_claimed_at: datetime(),
                          drain_scope_paths: [src.dd_path],
                          drain_scope_actionable: true
                        })
                        CREATE (source)-[:FROM_DD_PATH]->(node)
                        RETURN collect(source.id) AS ids
                        """,
                        sources=[
                            {
                                **source,
                                "expected_node_category": next(
                                    item["node"].get("node_category")
                                    for item in plan
                                    if item["path"] == source["dd_path"]
                                ),
                                "expected_documentation": next(
                                    item["node"].get("documentation")
                                    for item in plan
                                    if item["path"] == source["dd_path"]
                                ),
                                "expected_lifecycle_status": next(
                                    item["node"].get("lifecycle_status")
                                    for item in plan
                                    if item["path"] == source["dd_path"]
                                ),
                            }
                            for source in prepared
                        ],
                        scope_id=scope_id,
                    )
                    created_ids = set(created[0]["ids"] if created else [])
                    if created_ids != {source["id"] for source in prepared}:
                        raise services.ManifestDrainConflict(
                            "exact DD source creation changed during manifest stamp"
                        )

                existing = [
                    {
                        "id": item["source"]["id"],
                        "status": item["source"].get("status"),
                        "claim_token": item["source"].get("claim_token"),
                        "claim_seq": item["source"].get("claim_seq"),
                        "attempt_count": item["source"].get("attempt_count"),
                        "produced_sn_id": item["source"].get("produced_sn_id"),
                        "dd_version": item["source"].get("dd_version"),
                        "prior_scope_id": item["source"].get("drain_scope_id"),
                        "prior_scope_claimed_at_epoch_ms": item["source"].get(
                            "drain_scope_claimed_at_epoch_ms"
                        ),
                        "dd_target_ids": item["source"].get("dd_target_ids") or [],
                        "all_name_ids": item.get("all_name_ids") or [],
                        "actionable": (item["drain_actionable"]),
                        "path": item["path"],
                    }
                    for item in plan
                    if item["source"] is not None
                    and (
                        item["disposition"] == "active_in_flight"
                        or item["drain_actionable"]
                    )
                ]
                stamped_sources: set[str] = set()
                if existing:
                    stamped = tx_gc.query(
                        """
                        UNWIND $items AS item
                        MATCH (node:StandardNameSource {id: item.id})
                        SET node._drain_scope_lock = true
                        REMOVE node._drain_scope_lock
                        WITH node, item
                        OPTIONAL MATCH (node)-[:FROM_DD_PATH]->(dd:IMASNode)
                        WITH node, item,
                             [id IN collect(DISTINCT dd.id) WHERE id IS NOT NULL]
                               AS dd_target_ids
                        OPTIONAL MATCH (node)-[:PRODUCED_NAME]->(sn:StandardName)
                        WITH node, item, dd_target_ids,
                             [id IN collect(DISTINCT sn.id) WHERE id IS NOT NULL]
                               AS all_name_ids
                        WHERE node.status = item.status
                          AND (node.claim_token = item.claim_token OR
                               (node.claim_token IS NULL AND item.claim_token IS NULL))
                          AND (node.claim_seq = item.claim_seq OR
                               (node.claim_seq IS NULL AND item.claim_seq IS NULL))
                          AND (node.attempt_count = item.attempt_count OR
                               (node.attempt_count IS NULL AND item.attempt_count IS NULL))
                          AND (node.produced_sn_id = item.produced_sn_id OR
                               (node.produced_sn_id IS NULL AND
                                item.produced_sn_id IS NULL))
                          AND node.dd_version = item.dd_version
                          AND node.dd_snapshot_pinned = true
                          AND size(dd_target_ids) = size(item.dd_target_ids)
                          AND all(id IN dd_target_ids
                                  WHERE id IN item.dd_target_ids)
                          AND size(all_name_ids) = size(item.all_name_ids)
                          AND all(id IN all_name_ids
                                  WHERE id IN item.all_name_ids)
                          AND (node.drain_scope_id = item.prior_scope_id OR
                               (node.drain_scope_id IS NULL AND
                                item.prior_scope_id IS NULL))
                          AND (node.drain_scope_claimed_at.epochMillis =
                                 item.prior_scope_claimed_at_epoch_ms OR
                               (node.drain_scope_claimed_at IS NULL AND
                                item.prior_scope_claimed_at_epoch_ms IS NULL))
                        SET node.drain_scope_id = $scope_id,
                            node.drain_scope_claimed_at = datetime(),
                            node.drain_scope_paths = [item.path],
                            node.drain_scope_actionable = item.actionable
                        RETURN collect(node.id) AS ids
                        """,
                        items=existing,
                        scope_id=scope_id,
                    )
                    stamped_sources = set(stamped[0]["ids"] if stamped else [])
                    if stamped_sources != {item["id"] for item in existing}:
                        raise services.ManifestDrainConflict(
                            "source identity or lifecycle changed during manifest stamp"
                        )

                expected_names: dict[str, dict[str, Any]] = {}
                for item in plan:
                    if item["disposition"] != "active_in_flight":
                        continue
                    for name in [
                        *(item.get("live_names") or []),
                        *(item.get("parents") or []),
                    ]:
                        expected = expected_names.setdefault(
                            name["id"], {**name, "paths": []}
                        )
                        if item["path"] not in expected["paths"]:
                            expected["paths"].append(item["path"])
                if expected_names:
                    stamped = tx_gc.query(
                        """
                        UNWIND $items AS item
                        MATCH (node:StandardName {id: item.id})
                        SET node._drain_scope_lock = true
                        REMOVE node._drain_scope_lock
                        WITH node, item
                        WHERE (node.name_stage = item.name_stage OR
                               (node.name_stage IS NULL AND item.name_stage IS NULL))
                          AND (node.docs_stage = item.docs_stage OR
                               (node.docs_stage IS NULL AND item.docs_stage IS NULL))
                          AND (node.validation_status = item.validation_status OR
                               (node.validation_status IS NULL AND
                                item.validation_status IS NULL))
                          AND (node.claim_token = item.claim_token OR
                               (node.claim_token IS NULL AND item.claim_token IS NULL))
                          AND (node.claim_seq = item.claim_seq OR
                               (node.claim_seq IS NULL AND item.claim_seq IS NULL))
                          AND (node.drain_scope_id = item.drain_scope_id OR
                               (node.drain_scope_id IS NULL AND
                                item.drain_scope_id IS NULL))
                          AND (node.drain_scope_claimed_at.epochMillis =
                                 item.drain_scope_claimed_at_epoch_ms OR
                               (node.drain_scope_claimed_at IS NULL AND
                                item.drain_scope_claimed_at_epoch_ms IS NULL))
                        SET node.updated_at = datetime(), node.drain_scope_id = $scope_id,
                            node.drain_scope_claimed_at = datetime(),
                            node.drain_scope_paths = item.paths
                        RETURN collect(node.id) AS ids
                        """,
                        items=list(expected_names.values()),
                        scope_id=scope_id,
                    )
                    stamped_names = set(stamped[0]["ids"] if stamped else [])
                    if stamped_names != set(expected_names):
                        raise services.ManifestDrainConflict(
                            "name lifecycle or ownership changed during manifest stamp"
                        )
                tx.commit()
                return scope_id, plan
            except BaseException:
                with services.suppress(Exception):
                    tx.rollback()
                raise
    finally:
        if own:
            client.close()


def prepare_manifest_drain_scope(
    paths: list[str],
    *,
    dd_version: str,
    scope_timeout_seconds: int = 600,
    services: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Recover expired intersecting leases, then claim a freshly planned scope."""
    from imas_codex.standard_names.orphan_sweep import (
        find_expired_manifest_drain_scopes,
        recover_manifest_drain_scope,
    )

    exact_paths = services.canonicalize_manifest_drain_paths(paths)
    for stale_scope_id in find_expired_manifest_drain_scopes(
        exact_paths, scope_timeout_s=scope_timeout_seconds
    ):
        recover_manifest_drain_scope(
            stale_scope_id,
            scope_timeout_s=scope_timeout_seconds,
            worker_timeout_s=services._CLAIM_TIMEOUT_SECONDS,
            paths=exact_paths,
        )
    return services.claim_manifest_drain_scope(
        exact_paths,
        dd_version=dd_version,
        scope_timeout_seconds=scope_timeout_seconds,
    )


@retry_on_deadlock()
def clear_manifest_drain_scope(
    drain_scope_id: str, *, gc: Any | None = None, services: Any
) -> dict[str, int]:
    """Clear only one owned manifest scope without touching worker claims."""
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            CALL {
              MATCH (source:StandardNameSource {drain_scope_id: $scope_id})
              REMOVE source.drain_scope_id, source.drain_scope_claimed_at,
                     source.drain_scope_paths,
                     source.drain_scope_actionable,
                     source.drain_claim_scope_id
              RETURN count(source) AS sources
            }
            CALL {
              MATCH (name:StandardName {drain_scope_id: $scope_id})
              REMOVE name.drain_scope_id, name.drain_scope_claimed_at,
                     name.drain_scope_paths, name.drain_claim_scope_id
              RETURN count(name) AS names
            }
            RETURN sources, names
            """,
            scope_id=drain_scope_id,
        )
        return dict(rows[0]) if rows else {"sources": 0, "names": 0}
    finally:
        if own:
            client.close()


@retry_on_deadlock()
def release_manifest_drain_claims(
    drain_scope_id: str, *, gc: Any | None = None, services: Any
) -> dict[str, int]:
    """Release only claims atomically attributed to this bounded invocation."""
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            CALL {
              MATCH (source:StandardNameSource {
                drain_claim_scope_id: $scope_id,
                drain_scope_id: $scope_id
              })
              WHERE source.claim_token IS NOT NULL
              REMOVE source.claim_token, source.claimed_at
              RETURN count(source) AS sources
            }
            CALL {
              MATCH (name:StandardName {
                drain_claim_scope_id: $scope_id,
                drain_scope_id: $scope_id
              })
              WHERE name.claim_token IS NOT NULL
              SET name.updated_at = datetime(), name.name_stage = CASE
                    WHEN name.name_stage = 'refining'
                    THEN 'reviewed' ELSE name.name_stage END,
                  name.docs_stage = CASE
                    WHEN name.docs_stage = 'refining'
                    THEN 'reviewed' ELSE name.docs_stage END
              REMOVE name.claim_token, name.claimed_at
              RETURN count(name) AS names
            }
            RETURN sources, names
            """,
            scope_id=drain_scope_id,
        )
        return dict(rows[0]) if rows else {"sources": 0, "names": 0}
    finally:
        if own:
            client.close()


def finalize_manifest_drain_scope(
    drain_scope_id: str, *, paths: list[str], dd_version: str, services: Any
) -> list[dict[str, Any]]:
    """Release exact claims, report final state, and always clear the lease."""
    primary_error: BaseException | None = None
    try:
        services.release_manifest_drain_claims(drain_scope_id)
        return services.build_manifest_drain_plan(
            paths,
            dd_version=dd_version,
            drain_scope_id=drain_scope_id,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            services.clear_manifest_drain_scope(drain_scope_id)
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "bounded drain scope clear failed while preserving primary error"
            )


@retry_on_deadlock()
def scope_focus_names(
    sns_ids: list[str],
    run_id: str,
    *,
    reset: bool,
    gc: Any | None = None,
    services: Any,
) -> int:
    """Bind the names produced by focused sources to a scope ``run_id``.

    A ``--focus``/``--batch`` drain scopes its pools by ``run_id``; every name
    that should be worked this run must carry it. Two modes:

    - ``reset=False`` (default): bind ``run_id`` only, leaving each name in its
      current stage so it cycles from where it is (``drafted``→review,
      ``reviewed``→refine). This is the no-reset default — re-staging in-flight
      names to ``pending`` churns the hard tail (a genuinely-hard name is
      re-shot out of ``exhausted`` every pass and never settles).
    - ``reset=True`` (opt-in via ``--reseed``): re-stage each name to
      ``pending`` on both axes and clear its review scores/claims, then bind the
      ``run_id`` — the blunt reset-all. Targeted resets use ``--reset-to``.

    Returns the number of names bound.
    """
    if not sns_ids:
        return 0
    if reset:
        cypher = """
            UNWIND $ids AS sid
            MATCH (sns:StandardNameSource {id: sid})-[:PRODUCED_NAME]->(sn:StandardName)
            SET sn.name_stage = 'pending',
                sn.docs_stage = 'pending',
                sn.run_id = $run_id,
                sn.reviewed_name_at = null,
                sn.reviewed_docs_at = null,
                sn.reviewer_score_name = null,
                sn.reviewer_score_docs = null,
                sn.claim_token = null,
                sn.claimed_at = null,
                sn.updated_at = datetime()
            RETURN count(DISTINCT sn) AS n
            """
    else:
        cypher = """
            UNWIND $ids AS sid
            MATCH (sns:StandardNameSource {id: sid})-[:PRODUCED_NAME]->(sn:StandardName)
            SET sn.run_id = $run_id, sn.updated_at = datetime()
            RETURN count(DISTINCT sn) AS n
            """
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        rows = client.query(cypher, ids=sns_ids, run_id=run_id)
    finally:
        if own:
            client.close()
    return rows[0]["n"] if rows else 0


class ExactNameScopeConflict(ValueError):
    """An exact existing-name scope is unsafe or no longer exact."""


_EXACT_NAME_SCOPE_PREFLIGHT_QUERY = """
// EXACT_NAME_SCOPE_PREFLIGHT
UNWIND $names AS requested_name
OPTIONAL MATCH (requested:StandardName {id: requested_name})
WITH requested_name, collect(DISTINCT requested) AS matches
CALL (matches) {
  WITH matches
  UNWIND matches AS selected
  OPTIONAL MATCH (selected)-[:HAS_PARENT*0..]->(ancestor:StandardName)
  WITH selected, collect(DISTINCT ancestor) AS ancestors
  OPTIONAL MATCH (descendant:StandardName)-[:HAS_PARENT*1..]->(selected)
  WITH ancestors, collect(DISTINCT descendant) AS descendants
  WITH ancestors + descendants AS lineage
  UNWIND CASE WHEN lineage = [] THEN [null] ELSE lineage END AS member
  OPTIONAL MATCH (producer:StandardNameSource)-[:PRODUCED_NAME]->(member)
  RETURN collect(DISTINCT producer.id) AS producer_ids
}
RETURN requested_name,
       [candidate IN matches | {
         id: candidate.id,
         name_stage: candidate.name_stage,
         status: candidate.status,
         claimed_at: candidate.claimed_at,
         claim_token: candidate.claim_token,
         // Age is the fallback evidence, used only when the owning run cannot
         // be resolved; the run's own status is read separately and decides
         // every claim whose run_id resolves.
         claim_stale: candidate.claimed_at IS NOT NULL
                      AND candidate.claimed_at
                          < datetime() - duration({seconds: $claim_timeout_s}),
         run_id: candidate.run_id,
         drain_scope_id: candidate.drain_scope_id,
         drain_scope_claimed_at: candidate.drain_scope_claimed_at,
         drain_claim_scope_id: candidate.drain_claim_scope_id
       }] AS matches,
       [producer_id IN producer_ids
        WHERE producer_id STARTS WITH $fixture_source_id_prefix] AS fixture_producers
ORDER BY requested_name
"""


_EXACT_NAME_SCOPE_STAMP_QUERY = """
// EXACT_NAME_SCOPE_STAMP
UNWIND $names AS requested_name
MATCH (name:StandardName {id: requested_name})
WHERE NOT coalesce(name.name_stage, '') IN $terminal_name_stages
  AND NOT coalesce(name.status, '') IN $terminal_statuses
  AND (
    (name.claimed_at IS NULL AND name.claim_token IS NULL)
    OR name.id IN $released_ids
    OR name.claimed_at < datetime() - duration({seconds: $claim_timeout_s})
  )
  AND name.drain_scope_id IS NULL
  AND name.drain_scope_claimed_at IS NULL
  AND name.drain_claim_scope_id IS NULL
  AND NOT EXISTS {
    MATCH (name)-[:HAS_PARENT*0..]->(ancestor:StandardName)
    MATCH (fixture_source:StandardNameSource)-[:PRODUCED_NAME]->(ancestor)
    WHERE fixture_source.id STARTS WITH $fixture_source_id_prefix
  }
  AND NOT EXISTS {
    MATCH (descendant:StandardName)-[:HAS_PARENT*1..]->(name)
    MATCH (fixture_source:StandardNameSource)-[:PRODUCED_NAME]->(descendant)
    WHERE fixture_source.id STARTS WITH $fixture_source_id_prefix
  }
SET name.run_id = $run_id, name.updated_at = datetime(),
    name.claimed_at = null, name.claim_token = null
RETURN collect(name.id) AS stamped_ids
"""


_EXACT_NAME_SCOPE_RUN_STATUS_QUERY = """
// EXACT_NAME_SCOPE_RUN_STATUS
UNWIND $run_ids AS run_id
MATCH (owner:SNRun {id: run_id})
RETURN run_id, owner.status AS status
"""


_EXACT_SCOPE_TERMINAL_NAME_STAGES = frozenset({"superseded", "exhausted", "contested"})


_EXACT_SCOPE_TERMINAL_STATUSES = frozenset({"deprecated", "superseded"})


_TERMINAL_RUN_STATUSES = frozenset(
    {
        SNRunStatus.completed.value,
        SNRunStatus.interrupted.value,
        SNRunStatus.failed.value,
        SNRunStatus.degraded.value,
        SNRunStatus.stale.value,
    }
)


_CLAIM_ABSENT = "absent"


_CLAIM_HELD = "held"


_CLAIM_RELEASED = "released"


def _claim_disposition(
    candidate: dict[str, Any], run_statuses: dict[str, str], *, services: Any
) -> str:
    """Decide whether one preflight candidate's claim is held or abandoned.

    A claim is evidence that a worker holds the row. The direct evidence of
    abandonment is the owning run's terminal status: the claim died with the
    run, so the row is reclaimable at any age. Age is only a proxy for that,
    kept as the fallback for a claim whose owning run cannot be resolved — a
    legacy row with no ``run_id``, or a claim written by a process that never
    created a run. A token carrying no timestamp is never cleared by the orphan
    sweep either, so it stays held rather than becoming a row the reaper would
    leave behind.
    """
    claimed_at = candidate.get("claimed_at")
    if claimed_at is None and candidate.get("claim_token") is None:
        return services._CLAIM_ABSENT
    if claimed_at is None:
        return services._CLAIM_HELD
    status = run_statuses.get(str(candidate.get("run_id") or ""))
    if status is not None:
        return (
            services._CLAIM_RELEASED
            if status in services._TERMINAL_RUN_STATUSES
            else services._CLAIM_HELD
        )
    return (
        services._CLAIM_RELEASED
        if candidate.get("claim_stale", False)
        else services._CLAIM_HELD
    )


def _claimed_run_ids(rows: list[dict[str, Any]], *, services: Any) -> list[str]:
    """Collect the runs owning a claim in one preflight snapshot, bounded."""
    return sorted(
        {
            str(candidate["run_id"])
            for row in rows
            for candidate in (row.get("matches") or [])
            if candidate.get("claimed_at") is not None and candidate.get("run_id")
        }
    )


def _collect_run_statuses(
    run_ids: list[str], *, query: Any, services: Any
) -> dict[str, str]:
    """Resolve run rows that exist; a run absent from the graph stays unresolved.

    Callers only reach this when a claim is present, so the read is bounded by
    the size of the requested scope rather than by the claim population.
    """
    if not run_ids:
        return {}
    return {
        str(row["run_id"]): row.get("status")
        for row in query(services._EXACT_NAME_SCOPE_RUN_STATUS_QUERY, run_ids=run_ids)
    }


def _exact_name_scope_refusals(
    rows: list[dict[str, Any]],
    run_statuses: dict[str, str] | None = None,
    *,
    services: Any,
) -> list[str]:
    """Return deterministic refusal messages for one preflight snapshot.

    ``run_statuses`` maps the id of every run owning a claim in these rows to its
    ``SNRun`` status. A claim's own timestamp and token say a worker took the
    row; the owning run's status says whether that worker still exists, which is
    the fact age only approximates.
    """
    statuses = run_statuses or {}
    refusals: list[str] = []
    for row in rows:
        requested_name = str(row.get("requested_name") or "")
        matches = row.get("matches") or []
        if len(matches) != 1:
            state = "missing" if not matches else "ambiguous"
            refusals.append(f"{requested_name}: {state} StandardName identity")
            continue
        candidate = matches[0]
        if candidate.get(
            "name_stage"
        ) in services._EXACT_SCOPE_TERMINAL_NAME_STAGES or (
            candidate.get("status") in services._EXACT_SCOPE_TERMINAL_STATUSES
        ):
            refusals.append(f"{requested_name}: terminal StandardName lifecycle")
        if (
            candidate.get("claimed_at") is not None
            or candidate.get("claim_token") is not None
        ):
            # A claim is held while its owning run is unaccounted for: a run
            # still executing holds it at any age, and a run that reached a
            # terminal status released it the moment it stopped, so the row is
            # reclaimable however recently it was claimed. Age against the
            # orphan-sweep cutoff is the fallback for a claim whose run cannot
            # be resolved — a legacy row with no run_id, or a claim whose run
            # row is absent — and a token carrying no timestamp is never
            # cleared by the sweep either, so it stays held.
            if services._claim_disposition(candidate, statuses) == services._CLAIM_HELD:
                refusals.append(f"{requested_name}: current worker claim (live)")
        if any(
            candidate.get(field) is not None
            for field in (
                "drain_scope_id",
                "drain_scope_claimed_at",
                "drain_claim_scope_id",
            )
        ):
            refusals.append(f"{requested_name}: current drain scope")
        fixture_producers = sorted(set(row.get("fixture_producers") or []))
        if fixture_producers:
            refusals.append(
                f"{requested_name}: HAS_PARENT lineage is produced by fixture "
                f"source(s): {', '.join(fixture_producers)}"
            )
    return refusals


@retry_on_deadlock()
def scope_exact_standard_names(
    name_ids: list[str],
    run_id: str,
    *,
    dry_run: bool = False,
    gc: Any | None = None,
    services: Any,
) -> dict[str, Any]:
    """Preflight and bind one exact set of existing names to a fresh run scope.

    The read covers the complete requested set and both directions of its
    transitive ``HAS_PARENT`` lineage.  Any missing, ambiguous, terminal,
    claimed, drain-scoped, or fixture-lineage identity refuses the whole
    invocation.  A claim is refused while its owning run is unaccounted for: a
    run that is still executing refuses however old the claim is, while a run
    that reached a terminal status released its claim when it stopped, so the
    row is reclaimed at any age.  Age is the fallback for a claim whose owning
    run cannot be resolved, since it is then the only available evidence.
    ``run_id`` is durable provenance rather than a live lock, so a
    live invocation atomically replaces it on exactly the requested
    ``StandardName`` nodes after every live claim and drain field is clear.  It
    never touches ``StandardNameSource`` nodes.  Dry-run performs only the
    preflight read.
    """
    normalized = [str(name_id).strip() for name_id in name_ids]
    if not normalized or any(not name_id for name_id in normalized):
        raise services.ExactNameScopeConflict(
            "--name requires at least one non-empty identity"
        )
    duplicates = sorted(
        name_id for name_id in set(normalized) if normalized.count(name_id) > 1
    )
    if duplicates:
        raise services.ExactNameScopeConflict(
            "duplicate exact StandardName identity: " + ", ".join(duplicates)
        )

    from imas_codex.standard_names.fixture_sources import FIXTURE_SOURCE_ID_PREFIX

    requested = sorted(normalized)
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                rows = [
                    dict(row)
                    for row in transaction.run(
                        services._EXACT_NAME_SCOPE_PREFLIGHT_QUERY,
                        names=requested,
                        fixture_source_id_prefix=FIXTURE_SOURCE_ID_PREFIX,
                        claim_timeout_s=services.DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
                    )
                ]
                returned = [str(row.get("requested_name") or "") for row in rows]
                if returned != requested:
                    raise services.ExactNameScopeConflict(
                        "exact-name preflight did not return the complete requested set"
                    )
                run_statuses = services._collect_run_statuses(
                    services._claimed_run_ids(rows), query=transaction.run
                )
                refusals = services._exact_name_scope_refusals(rows, run_statuses)
                if refusals:
                    raise services.ExactNameScopeConflict("; ".join(refusals))

                released_ids = sorted(
                    {
                        str(candidate["id"])
                        for row in rows
                        for candidate in (row.get("matches") or [])
                        if services._claim_disposition(candidate, run_statuses)
                        == services._CLAIM_RELEASED
                    }
                )

                stamped_ids: list[str] = []
                if not dry_run:
                    stamp_rows = list(
                        transaction.run(
                            services._EXACT_NAME_SCOPE_STAMP_QUERY,
                            names=requested,
                            run_id=run_id,
                            terminal_name_stages=sorted(
                                services._EXACT_SCOPE_TERMINAL_NAME_STAGES
                            ),
                            terminal_statuses=sorted(
                                services._EXACT_SCOPE_TERMINAL_STATUSES
                            ),
                            fixture_source_id_prefix=FIXTURE_SOURCE_ID_PREFIX,
                            claim_timeout_s=services.DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
                            released_ids=released_ids,
                        )
                    )
                    if len(stamp_rows) != 1:
                        raise services.ExactNameScopeConflict(
                            "exact-name stamp did not return one cardinality result"
                        )
                    stamped_ids = sorted(dict(stamp_rows[0]).get("stamped_ids") or [])
                    if stamped_ids != requested:
                        raise services.ExactNameScopeConflict(
                            "exact-name state changed between preflight and stamp"
                        )
                    transaction.commit()
                else:
                    transaction.close()
                return {
                    "name_ids": requested,
                    "run_id": run_id,
                    "dry_run": dry_run,
                    "stamped": len(stamped_ids),
                }
            except BaseException:
                if not transaction.closed:
                    transaction.close()
                raise
    finally:
        if own:
            client.close()

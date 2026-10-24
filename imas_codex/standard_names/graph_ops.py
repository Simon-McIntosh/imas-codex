"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

# ruff: noqa: F401
import functools
import hashlib
import inspect
import json
import logging  # noqa: F401
import os
import re
import time
import uuid
from collections import Counter, defaultdict, deque  # noqa: F401
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import NameStage, RefineStopReason, SNRunStatus
from imas_codex.graph.schema import GraphSchema
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.doc_links import (
    find_name_links,
    find_name_references,
    find_prose_bracket_spans,
)
from imas_codex.standard_names.ledger import reattach_produced_name_edges  # noqa: F401
from imas_codex.standard_names.protection import (
    ProtectedDeletionError,
    automatic_deletion_protections,
    filter_automatic_deletion_candidates,
    refuse_protected_automatic_deletion,
)
from imas_codex.standard_names.provenance_lifecycle import (
    deletion_change_cypher,
    deletion_change_params,
)
from imas_codex.standard_names.review.holds import (
    docs_acceptance_set,  # noqa: F401
    docs_acceptance_where,  # noqa: F401
    docs_accepted,  # noqa: F401
    has_open_docs_hold_where,
    open_holds,
    sync_holds,  # noqa: F401
)
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

# Wire graph-domain calls through the entrypoint at runtime.
from .graph_store import graph_foundation as _graph_foundation


class _GraphServices:
    def __getattr__(self, name: str) -> Any:
        return globals()[name]


_graph_foundation.services = _GraphServices()


logger = import_module(".graph_store.graph_foundation", __package__).logger


_STABLE_BINDING_NAME_STAGES = import_module(
    ".graph_store.graph_foundation", __package__
)._STABLE_BINDING_NAME_STAGES
_TERMINAL_BINDING_NAME_STAGES = import_module(
    ".graph_store.graph_foundation", __package__
)._TERMINAL_BINDING_NAME_STAGES
_SETTLED_SOURCE_BINDING_STAGES = import_module(
    ".graph_store.graph_foundation", __package__
)._SETTLED_SOURCE_BINDING_STAGES
_CLAIM_TIMEOUT_SECONDS = import_module(
    ".graph_store.graph_foundation", __package__
)._CLAIM_TIMEOUT_SECONDS

# The lowest observed daily structural-reaper total is 74.  Eighty allows a
# normal-sized cleanup while refusing the 93-identity deletion that exposed the
# missing guard.
DERIVED_PARENT_CLEANUP_DELETION_LIMIT = import_module(
    ".graph_store.graph_foundation", __package__
).DERIVED_PARENT_CLEANUP_DELETION_LIMIT


DerivedParentCleanupRefusal = import_module(
    ".graph_store.graph_foundation", __package__
).DerivedParentCleanupRefusal


_TransactionQuery = import_module(
    ".graph_store.graph_foundation", __package__
)._TransactionQuery


def _ensure_json(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._ensure_json."""
    from .graph_store import graph_foundation

    return graph_foundation._ensure_json(*args, **kwargs)


def _ensure_list(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._ensure_list."""
    from .graph_store import graph_foundation

    return graph_foundation._ensure_list(*args, **kwargs)


def _scalar_domain(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._scalar_domain."""
    from .graph_store import graph_foundation

    return graph_foundation._scalar_domain(*args, **kwargs)


def reclassify_standard_name_domain(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation.reclassify_standard_name_domain."""
    from .graph_store import graph_foundation

    return graph_foundation.reclassify_standard_name_domain(*args, **kwargs)


#: Physics abbreviation expansions applied before name persistence.
#: Keys are case-sensitive abbreviations as they appear in LLM output;
#: values are their lowercase ISN-compliant expansions.
_PHYSICS_ABBREVIATIONS = import_module(
    ".graph_store.graph_foundation", __package__
)._PHYSICS_ABBREVIATIONS


def normalize_name_id(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation.normalize_name_id."""
    from .graph_store import graph_foundation

    return graph_foundation.normalize_name_id(*args, **kwargs)


def _public_segment_token_index(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._public_segment_token_index."""
    from .graph_store import graph_foundation

    return graph_foundation._public_segment_token_index(*args, **kwargs)


def _public_grammar_segment_order(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._public_grammar_segment_order."""
    from .graph_store import graph_foundation

    return graph_foundation._public_grammar_segment_order(*args, **kwargs)


def _segments_from_ir(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._segments_from_ir."""
    from .graph_store import graph_foundation

    return graph_foundation._segments_from_ir(*args, **kwargs)


def _parse_grammar(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._parse_grammar."""
    from .graph_store import graph_foundation

    return graph_foundation._parse_grammar(*args, **kwargs)


def _semantic_ir_expression(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._semantic_ir_expression."""
    from .graph_store import graph_foundation

    return graph_foundation._semantic_ir_expression(*args, **kwargs)


def strict_review_grammar_context(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation.strict_review_grammar_context."""
    from .graph_store import graph_foundation

    return graph_foundation.strict_review_grammar_context(*args, **kwargs)


def _compute_link_status(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._compute_link_status."""
    from .graph_store import graph_foundation

    return graph_foundation._compute_link_status(*args, **kwargs)


_LINK_RE = import_module(".graph_store.graph_foundation", __package__)._LINK_RE

_DD_LINK_RE = import_module(".graph_store.graph_foundation", __package__)._DD_LINK_RE


def _extract_links_from_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._extract_links_from_docs."""
    from .graph_store import graph_foundation

    return graph_foundation._extract_links_from_docs(*args, **kwargs)


# =============================================================================
# Pipeline cost model — bucket queries & historical CPI
# =============================================================================

# Module-level TTL cache for pipeline buckets.
_pipeline_buckets_cache = import_module(
    ".graph_store.graph_foundation", __package__
)._pipeline_buckets_cache

# Module-level cache for historical CPI (once per process).
_historical_cpi_cache = import_module(
    ".graph_store.graph_foundation", __package__
)._historical_cpi_cache


def query_pipeline_buckets(
    threshold: float = 0.65,
    cap: int = 3,
    cache_ttl: float = 3.0,
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
        _pipeline_buckets_cache["value"] is not None
        and now - _pipeline_buckets_cache["ts"] < cache_ttl
    ):
        return _pipeline_buckets_cache["value"]

    with GraphClient() as gc:
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
    _pipeline_buckets_cache["value"] = result
    _pipeline_buckets_cache["ts"] = now
    return result


def query_historical_cpi(model: str | None = None) -> dict[str, float]:
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
    if _historical_cpi_cache["value"] is not None:
        return _historical_cpi_cache["value"]

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
    with GraphClient() as gc:
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

    _historical_cpi_cache["value"] = result
    _historical_cpi_cache["ts"] = _time.time()
    return result


# =============================================================================
# Read helpers — extraction candidates
# =============================================================================


def get_extraction_candidates_dd(
    ids_filter: str | None = None,
    domain_filter: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Query IMASNode paths grouped by semantic cluster.

    Returns dynamic leaf nodes that have been enriched (status=embedded),
    optionally filtered by IDS or physics domain.
    """
    with GraphClient() as gc:
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
    facility: str,
    domain_filter: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Query FacilitySignal nodes for a given facility.

    Returns signals that have been enriched, optionally filtered by
    physics domain.
    """
    with GraphClient() as gc:
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


# =============================================================================
# Deduplication
# =============================================================================


def get_existing_standard_names(gc: Any | None = None) -> set[str]:
    """Return the set of existing StandardName node IDs for deduplication.

    This is the single definition of the graph-wide all-name read: callers that
    would otherwise restate ``MATCH (sn:StandardName) RETURN sn.id AS id`` read
    it here instead. A caller already holding an open client passes it as *gc*
    so the read rides that session; with none, one is opened for this read.
    """
    if gc is not None:
        results = gc.query("MATCH (sn:StandardName) RETURN sn.id AS id")
        return {r["id"] for r in results}
    with GraphClient() as own:
        results = own.query("MATCH (sn:StandardName) RETURN sn.id AS id")
        return {r["id"] for r in results}


def get_named_source_ids() -> set[str]:
    """Return source IDs already linked via HAS_STANDARD_NAME.

    Used for resumability: extract skips sources that already have
    a standard name unless --force is specified.
    """
    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN DISTINCT src.id AS source_id
        """)
        return {r["source_id"] for r in results}


def partition_focus_by_accepted(
    paths: list[str], *, gc: Any | None = None
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
        with GraphClient() as _gc:
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


def canonicalize_manifest_drain_paths(paths: list[str]) -> list[str]:
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


def classify_manifest_drain_item(item: dict[str, Any]) -> dict[str, Any]:
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
            and int(source.get("attempt_count") or 0) < _MAX_COMPOSE_CLAIM_ATTEMPTS
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
    rows: list[dict[str, Any]], dd_version: str
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
    return [classify_manifest_drain_item(row) for row in resolved_rows]


def build_manifest_drain_plan(
    paths: list[str],
    *,
    dd_version: str,
    drain_scope_id: str = "dry-run",
    scope_timeout_seconds: int = 600,
    worker_timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    gc: Any | None = None,
) -> list[dict[str, Any]]:
    """Read and classify every exact path without issuing any graph write."""
    exact_paths = canonicalize_manifest_drain_paths(paths)
    own = gc is None
    client = GraphClient() if own else gc
    try:
        rows = client.query(
            _MANIFEST_DRAIN_PLAN_QUERY,
            paths=exact_paths,
            dd_version=dd_version,
            drain_scope_id=drain_scope_id,
            scope_cutoff=f"PT{scope_timeout_seconds}S",
            worker_cutoff=f"PT{worker_timeout_seconds}S",
            terminal_stages=sorted(_TERMINAL_BINDING_NAME_STAGES),
        )
        return _resolved_manifest_drain_plan(rows, dd_version)
    finally:
        if own:
            client.close()


class ManifestDrainConflict(RuntimeError):
    """The bounded manifest changed or contains an ambiguous identity."""


def _manifest_plan_fingerprint(plan: list[dict[str, Any]]) -> str:
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


def _lock_manifest_drain_authority(gc: Any, paths: list[str]) -> None:
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
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically create absent exact sources and CAS-stamp one bounded scope."""
    exact_paths = canonicalize_manifest_drain_paths(paths)
    scope_id = drain_scope_id or str(uuid.uuid4())
    own = gc is None
    client = GraphClient() if own else gc
    try:
        with client.session() as session:
            tx = session.begin_transaction()
            tx_gc = _TransactionQuery(tx)
            try:
                rows = tx_gc.query(
                    _MANIFEST_DRAIN_PLAN_QUERY,
                    paths=exact_paths,
                    dd_version=dd_version,
                    drain_scope_id=scope_id,
                    scope_cutoff=f"PT{scope_timeout_seconds}S",
                    worker_cutoff=f"PT{_CLAIM_TIMEOUT_SECONDS}S",
                    terminal_stages=sorted(_TERMINAL_BINDING_NAME_STAGES),
                )
                plan = _resolved_manifest_drain_plan(rows, dd_version)
                initial_fingerprint = _manifest_plan_fingerprint(plan)
                _lock_manifest_drain_authority(tx_gc, exact_paths)
                locked_rows = tx_gc.query(
                    _MANIFEST_DRAIN_PLAN_QUERY,
                    paths=exact_paths,
                    dd_version=dd_version,
                    drain_scope_id=scope_id,
                    scope_cutoff=f"PT{scope_timeout_seconds}S",
                    worker_cutoff=f"PT{_CLAIM_TIMEOUT_SECONDS}S",
                    terminal_stages=sorted(_TERMINAL_BINDING_NAME_STAGES),
                )
                plan = _resolved_manifest_drain_plan(locked_rows, dd_version)
                if _manifest_plan_fingerprint(plan) != initial_fingerprint:
                    raise ManifestDrainConflict(
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
                    raise ManifestDrainConflict(
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
                prepared = _pin_dd_source_snapshots(
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
                        raise ManifestDrainConflict(
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
                        raise ManifestDrainConflict(
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
                        raise ManifestDrainConflict(
                            "name lifecycle or ownership changed during manifest stamp"
                        )
                tx.commit()
                return scope_id, plan
            except BaseException:
                with suppress(Exception):
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
) -> tuple[str, list[dict[str, Any]]]:
    """Recover expired intersecting leases, then claim a freshly planned scope."""
    from imas_codex.standard_names.orphan_sweep import (
        find_expired_manifest_drain_scopes,
        recover_manifest_drain_scope,
    )

    exact_paths = canonicalize_manifest_drain_paths(paths)
    for stale_scope_id in find_expired_manifest_drain_scopes(
        exact_paths, scope_timeout_s=scope_timeout_seconds
    ):
        recover_manifest_drain_scope(
            stale_scope_id,
            scope_timeout_s=scope_timeout_seconds,
            worker_timeout_s=_CLAIM_TIMEOUT_SECONDS,
            paths=exact_paths,
        )
    return claim_manifest_drain_scope(
        exact_paths,
        dd_version=dd_version,
        scope_timeout_seconds=scope_timeout_seconds,
    )


@retry_on_deadlock()
def clear_manifest_drain_scope(
    drain_scope_id: str, *, gc: Any | None = None
) -> dict[str, int]:
    """Clear only one owned manifest scope without touching worker claims."""
    own = gc is None
    client = GraphClient() if own else gc
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
    drain_scope_id: str, *, gc: Any | None = None
) -> dict[str, int]:
    """Release only claims atomically attributed to this bounded invocation."""
    own = gc is None
    client = GraphClient() if own else gc
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
    drain_scope_id: str,
    *,
    paths: list[str],
    dd_version: str,
) -> list[dict[str, Any]]:
    """Release exact claims, report final state, and always clear the lease."""
    primary_error: BaseException | None = None
    try:
        release_manifest_drain_claims(drain_scope_id)
        return build_manifest_drain_plan(
            paths,
            dd_version=dd_version,
            drain_scope_id=drain_scope_id,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            clear_manifest_drain_scope(drain_scope_id)
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "bounded drain scope clear failed while preserving primary error"
            )


@retry_on_deadlock()
def scope_focus_names(
    sns_ids: list[str], run_id: str, *, reset: bool, gc: Any | None = None
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
    client = GraphClient() if own else gc
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

#: ``SNRunStatus`` values that end a run's ownership of its claims. Every other
#: status — ``started``, a status written by a future release, or a run row that
#: cannot be read at all — leaves the claim held, so the reading here fails
#: closed: only a run known to have stopped releases its claim.
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


def _claim_disposition(candidate: dict[str, Any], run_statuses: dict[str, str]) -> str:
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
        return _CLAIM_ABSENT
    if claimed_at is None:
        return _CLAIM_HELD
    status = run_statuses.get(str(candidate.get("run_id") or ""))
    if status is not None:
        return _CLAIM_RELEASED if status in _TERMINAL_RUN_STATUSES else _CLAIM_HELD
    return _CLAIM_RELEASED if candidate.get("claim_stale", False) else _CLAIM_HELD


def _claimed_run_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Collect the runs owning a claim in one preflight snapshot, bounded."""
    return sorted(
        {
            str(candidate["run_id"])
            for row in rows
            for candidate in (row.get("matches") or [])
            if candidate.get("claimed_at") is not None and candidate.get("run_id")
        }
    )


def _collect_run_statuses(run_ids: list[str], *, query: Any) -> dict[str, str]:
    """Resolve run rows that exist; a run absent from the graph stays unresolved.

    Callers only reach this when a claim is present, so the read is bounded by
    the size of the requested scope rather than by the claim population.
    """
    if not run_ids:
        return {}
    return {
        str(row["run_id"]): row.get("status")
        for row in query(_EXACT_NAME_SCOPE_RUN_STATUS_QUERY, run_ids=run_ids)
    }


def _exact_name_scope_refusals(
    rows: list[dict[str, Any]], run_statuses: dict[str, str] | None = None
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
        if candidate.get("name_stage") in _EXACT_SCOPE_TERMINAL_NAME_STAGES or (
            candidate.get("status") in _EXACT_SCOPE_TERMINAL_STATUSES
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
            if _claim_disposition(candidate, statuses) == _CLAIM_HELD:
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
        raise ExactNameScopeConflict("--name requires at least one non-empty identity")
    duplicates = sorted(
        name_id for name_id in set(normalized) if normalized.count(name_id) > 1
    )
    if duplicates:
        raise ExactNameScopeConflict(
            "duplicate exact StandardName identity: " + ", ".join(duplicates)
        )

    from imas_codex.standard_names.fixture_sources import FIXTURE_SOURCE_ID_PREFIX

    requested = sorted(normalized)
    own = gc is None
    client = GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                rows = [
                    dict(row)
                    for row in transaction.run(
                        _EXACT_NAME_SCOPE_PREFLIGHT_QUERY,
                        names=requested,
                        fixture_source_id_prefix=FIXTURE_SOURCE_ID_PREFIX,
                        claim_timeout_s=DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
                    )
                ]
                returned = [str(row.get("requested_name") or "") for row in rows]
                if returned != requested:
                    raise ExactNameScopeConflict(
                        "exact-name preflight did not return the complete requested set"
                    )
                run_statuses = _collect_run_statuses(
                    _claimed_run_ids(rows), query=transaction.run
                )
                refusals = _exact_name_scope_refusals(rows, run_statuses)
                if refusals:
                    raise ExactNameScopeConflict("; ".join(refusals))

                released_ids = sorted(
                    {
                        str(candidate["id"])
                        for row in rows
                        for candidate in (row.get("matches") or [])
                        if _claim_disposition(candidate, run_statuses)
                        == _CLAIM_RELEASED
                    }
                )

                stamped_ids: list[str] = []
                if not dry_run:
                    stamp_rows = list(
                        transaction.run(
                            _EXACT_NAME_SCOPE_STAMP_QUERY,
                            names=requested,
                            run_id=run_id,
                            terminal_name_stages=sorted(
                                _EXACT_SCOPE_TERMINAL_NAME_STAGES
                            ),
                            terminal_statuses=sorted(_EXACT_SCOPE_TERMINAL_STATUSES),
                            fixture_source_id_prefix=FIXTURE_SOURCE_ID_PREFIX,
                            claim_timeout_s=DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
                            released_ids=released_ids,
                        )
                    )
                    if len(stamp_rows) != 1:
                        raise ExactNameScopeConflict(
                            "exact-name stamp did not return one cardinality result"
                        )
                    stamped_ids = sorted(dict(stamp_rows[0]).get("stamped_ids") or [])
                    if stamped_ids != requested:
                        raise ExactNameScopeConflict(
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


def get_source_name_mapping(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.get_source_name_mapping."""
    from .graph_store import graph_lineage

    return graph_lineage.get_source_name_mapping(
        *args, services=_graph_foundation.services, **kwargs
    )


def _get_rich_source_name_mapping(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._get_rich_source_name_mapping."""
    from .graph_store import graph_lineage

    return graph_lineage._get_rich_source_name_mapping(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Write helpers
# =============================================================================


def fetch_low_score_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_low_score_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_low_score_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_review_feedback_for_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_review_feedback_for_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_review_feedback_for_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_reviewer_history_for_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_reviewer_history_for_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_reviewer_history_for_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_docs_review_feedback_for_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_docs_review_feedback_for_standard_names."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_docs_review_feedback_for_standard_names(
        *args, services=_graph_foundation.services, **kwargs
    )


# Public rename. The canonical name is
# ``fetch_docs_review_feedback_for_standard_names`` (above); the legacy
# ``fetch_docs_review_feedback_for_sns`` alias is retained for one
# release with a DeprecationWarning. Package-internal callers have already
# moved to the canonical name.
def fetch_docs_review_feedback_for_sns(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_docs_review_feedback_for_sns."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_docs_review_feedback_for_sns(
        *args, services=_graph_foundation.services, **kwargs
    )


def _filter_admissible_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._filter_admissible_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._filter_admissible_parents(
        *args, services=_graph_foundation.services, **kwargs
    )


def _emit_magnitude_of_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._emit_magnitude_of_edges."""
    from .graph_store import graph_lineage

    return graph_lineage._emit_magnitude_of_edges(
        *args, services=_graph_foundation.services, **kwargs
    )


def _bootstrap_missing_derived_parent_targets(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._bootstrap_missing_derived_parent_targets."""
    from .graph_store import graph_lineage

    return graph_lineage._bootstrap_missing_derived_parent_targets(
        *args, services=_graph_foundation.services, **kwargs
    )


def _write_standard_name_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._write_standard_name_edges."""
    from .graph_store import graph_lineage

    return graph_lineage._write_standard_name_edges(
        *args, services=_graph_foundation.services, **kwargs
    )


_structural_edge_reconcile_apply = import_module(
    ".graph_store.graph_lineage", __package__
)._structural_edge_reconcile_apply

globals()["reconcile_structural_edges_for_" + "standard_names"] = (
    _structural_edge_reconcile_apply
)


def _clear_superseded_child_derived_parent_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._clear_superseded_child_derived_parent_edges."""
    from .graph_store import graph_lineage

    return graph_lineage._clear_superseded_child_derived_parent_edges(
        *args, services=_graph_foundation.services, **kwargs
    )


def rederive_structural_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.rederive_structural_edges."""
    from .graph_store import graph_lineage

    return graph_lineage.rederive_structural_edges(
        *args, services=_graph_foundation.services, **kwargs
    )


def _rewire_has_parent_off_superseded(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._rewire_has_parent_off_superseded."""
    from .graph_store import graph_lineage

    return graph_lineage._rewire_has_parent_off_superseded(
        *args, services=_graph_foundation.services, **kwargs
    )


def _parse_parent_grammar(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._parse_parent_grammar."""
    from .graph_store import graph_lineage

    return graph_lineage._parse_parent_grammar(
        *args, services=_graph_foundation.services, **kwargs
    )


def _query_seedable_derived_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_seedable_derived_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._query_seedable_derived_parents(
        *args, services=_graph_foundation.services, **kwargs
    )


def _query_legacy_repairable_derived_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_legacy_repairable_derived_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._query_legacy_repairable_derived_parents(
        *args, services=_graph_foundation.services, **kwargs
    )


def _query_derived_parents_for_admission_cleanup(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_derived_parents_for_admission_cleanup."""
    from .graph_store import graph_lineage

    return graph_lineage._query_derived_parents_for_admission_cleanup(
        *args, services=_graph_foundation.services, **kwargs
    )


def _delete_derived_parent_nodes(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._delete_derived_parent_nodes."""
    from .graph_store import graph_lineage

    return graph_lineage._delete_derived_parent_nodes(
        *args, services=_graph_foundation.services, **kwargs
    )


def sweep_orphaned_docs_revisions(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.sweep_orphaned_docs_revisions."""
    from .graph_store import graph_lineage

    return graph_lineage.sweep_orphaned_docs_revisions(
        *args, services=_graph_foundation.services, **kwargs
    )


def _derived_parent_source_metadata(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._derived_parent_source_metadata."""
    from .graph_store import graph_lineage

    return graph_lineage._derived_parent_source_metadata(
        *args, services=_graph_foundation.services, **kwargs
    )


def _validate_derived_parent_identity(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._validate_derived_parent_identity."""
    from .graph_store import graph_lineage

    return graph_lineage._validate_derived_parent_identity(
        *args, services=_graph_foundation.services, **kwargs
    )


def _eligible_derived_parent_unit_children(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._eligible_derived_parent_unit_children."""
    from .graph_store import graph_lineage

    return graph_lineage._eligible_derived_parent_unit_children(
        *args, services=_graph_foundation.services, **kwargs
    )


def _materialize_derived_parent_rows(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._materialize_derived_parent_rows."""
    from .graph_store import graph_lineage

    return graph_lineage._materialize_derived_parent_rows(
        *args, services=_graph_foundation.services, **kwargs
    )


_ParentMaterializationCapture = import_module(
    ".graph_store.graph_lineage", __package__
)._ParentMaterializationCapture


def _materialize_derived_parent_rows_batched(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._materialize_derived_parent_rows_batched."""
    from .graph_store import graph_lineage

    return graph_lineage._materialize_derived_parent_rows_batched(
        *args, services=_graph_foundation.services, **kwargs
    )


_normalization_peel_unit_repair_apply = import_module(
    ".graph_store.graph_lineage", __package__
)._normalization_peel_unit_repair_apply

globals()["repair_normalization_peel_parent_" + "units"] = (
    _normalization_peel_unit_repair_apply
)


def stamp_standard_name_units(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.stamp_standard_name_units."""
    from .graph_store import graph_lineage

    return graph_lineage.stamp_standard_name_units(
        *args, services=_graph_foundation.services, **kwargs
    )


def seed_parent_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.seed_parent_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.seed_parent_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def normalize_derived_parent_lifecycle(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.normalize_derived_parent_lifecycle."""
    from .graph_store import graph_lineage

    return graph_lineage.normalize_derived_parent_lifecycle(
        *args, services=_graph_foundation.services, **kwargs
    )


LIFECYCLELESS_STUB_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_lineage", __package__
).LIFECYCLELESS_STUB_MANIFEST_SCHEMA

LIFECYCLELESS_STUB_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_lineage", __package__
).LIFECYCLELESS_STUB_RECEIPT_SCHEMA

_LIFECYCLELESS_STUB_LOCK_QUERY = import_module(
    ".graph_store.graph_lineage", __package__
)._LIFECYCLELESS_STUB_LOCK_QUERY


LifecyclelessStubConflict = import_module(
    ".graph_store.graph_lineage", __package__
).LifecyclelessStubConflict


_READ_LIFECYCLELESS_STUBS_QUERY = import_module(
    ".graph_store.graph_lineage", __package__
)._READ_LIFECYCLELESS_STUBS_QUERY


def _canonicalize_lifecycleless_manifest_value(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._canonicalize_lifecycleless_manifest_value."""
    from .graph_store import graph_lineage

    return graph_lineage._canonicalize_lifecycleless_manifest_value(
        *args, services=_graph_foundation.services, **kwargs
    )


def _derived_parent_unit_from_eligible_children(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._derived_parent_unit_from_eligible_children."""
    from .graph_store import graph_lineage

    return graph_lineage._derived_parent_unit_from_eligible_children(
        *args, services=_graph_foundation.services, **kwargs
    )


def _spurious_stub_binding_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._spurious_stub_binding_authority."""
    from .graph_store import graph_lineage

    return graph_lineage._spurious_stub_binding_authority(
        *args, services=_graph_foundation.services, **kwargs
    )


def _partition_lifecycleless_stub_rows(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._partition_lifecycleless_stub_rows."""
    from .graph_store import graph_lineage

    return graph_lineage._partition_lifecycleless_stub_rows(
        *args, services=_graph_foundation.services, **kwargs
    )


def _lifecycleless_stub_manifest(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._lifecycleless_stub_manifest."""
    from .graph_store import graph_lineage

    return graph_lineage._lifecycleless_stub_manifest(
        *args, services=_graph_foundation.services, **kwargs
    )


def _reconcile_spurious_stub_source_scalars(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._reconcile_spurious_stub_source_scalars."""
    from .graph_store import graph_lineage

    return graph_lineage._reconcile_spurious_stub_source_scalars(
        *args, services=_graph_foundation.services, **kwargs
    )


_lifecycleless_stub_reconcile_apply = import_module(
    ".graph_store.graph_lineage", __package__
)._lifecycleless_stub_reconcile_apply

_lifecycleless_stub_reconcile_signature = import_module(
    ".graph_store.graph_lineage", __package__
)._lifecycleless_stub_reconcile_signature

_lifecycleless_stub_reconcile_parameters = import_module(
    ".graph_store.graph_lineage", __package__
)._lifecycleless_stub_reconcile_parameters

_lifecycleless_stub_reconcile_apply.__signature__ = inspect.Signature(
    [
        _lifecycleless_stub_reconcile_parameters["apply"],
        _lifecycleless_stub_reconcile_parameters["manifest_sha256"],
        _lifecycleless_stub_reconcile_parameters["gc"],
    ],
    return_annotation=_lifecycleless_stub_reconcile_signature.return_annotation,
)
globals()["reconcile_lifecycleless_" + "standard_name_stubs"] = (
    _lifecycleless_stub_reconcile_apply
)
del _lifecycleless_stub_reconcile_signature
del _lifecycleless_stub_reconcile_parameters


_SKELETON_PLACEHOLDER_PREDICATE = import_module(
    ".graph_store.graph_lineage", __package__
)._SKELETON_PLACEHOLDER_PREDICATE


def _query_skeleton_placeholders_for_cleanup(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_skeleton_placeholders_for_cleanup."""
    from .graph_store import graph_lineage

    return graph_lineage._query_skeleton_placeholders_for_cleanup(
        *args, services=_graph_foundation.services, **kwargs
    )


def write_standard_names(
    names: list[dict[str, Any]],
    *,
    override: bool = False,
    gc: Any | None = None,
    return_written_ids: bool = False,
) -> int | list[str]:
    """MERGE StandardName nodes with HAS_STANDARD_NAME relationships.

    Relationship direction: entity → concept
      (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
      (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)

    Each dict in *names* must have at least:
      - ``id``: the composed standard name string
      - ``source_types``: ["dd"] or ["signals"] etc.
      - ``source_id``: the originating path / signal ID

    Optional fields: ``unit``, ``description``,
    ``documentation``, ``kind``, ``links``, ``source_paths``,
    ``validity_domain``, ``constraints``, ``model``,
    ``generated_at``,
    ``review_tier``,
    ``vocab_gap_detail``, ``validation_issues``,
    ``validation_layer_summary``, ``cocos_transformation_type``, ``dd_version``,
    ``isn_version``, ``codex_version``,
    ``review_input_hash``.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection — write protected fields
        even on catalog-edited names.
    gc:
        Optional caller-owned graph query handle. All writes use this handle,
        allowing an explicit transaction to own the complete rich write.
    return_written_ids:
        Return the identities that survived validation and conflict checks.

    Performs conflict detection on ``unit``: if a StandardName already exists
    with a different unit value, that entry is skipped (not written)
    and a warning is logged.

    Returns the number of nodes written.
    """
    if not names:
        return [] if return_written_ids else 0

    # Pipeline protection — strip catalog-owned fields from catalog_edit items
    from imas_codex.standard_names.protection import filter_protected

    names, skipped = filter_protected(names, override=override)
    if skipped:
        logger.warning(
            "write_standard_names: stripped protected fields from %d catalog-edited name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    if not names:
        return [] if return_written_ids else 0

    # Strict grammar identity gate — reject names that ISN cannot parse and
    # losslessly recompose.  The legacy flat projection is not an authority:
    # it intentionally cannot represent every valid ordered operator tree.
    # This is a hard invariant: if a name fails grammar round-trip, it is
    # invalid by definition and must not enter the graph.
    try:
        from imas_standard_names import compose, parse
    except ImportError:
        parse = None  # type: ignore[assignment]

    if parse is not None:
        valid_names: list[dict[str, Any]] = []
        for n in names:
            try:
                parsed = parse(n["id"], strict=True)
                if compose(parsed.ir) != n["id"]:
                    raise ValueError("strict grammar round-trip changed the name")
                valid_names.append(n)
            except Exception as exc:
                logger.warning(
                    "write_standard_names: rejecting '%s' — grammar parse failed: %s",
                    n["id"],
                    str(exc)[:120],
                )
        if len(valid_names) < len(names):
            logger.info(
                "Grammar gate: %d/%d names passed parse validation",
                len(valid_names),
                len(names),
            )
        names = valid_names
        if not names:
            return [] if return_written_ids else 0

    # A COCOS-dependent name (cocos_transformation_type set) must carry the
    # integer cocos convention so the HAS_COCOS edge below is created. The
    # catalog follows a single convention — the current DD version's (DDv4 →
    # COCOS 17) — so default it from the current DD whenever compose/refine
    # didn't thread it through. COCOS-independent names stay unlinked by design.
    if any(
        n.get("cocos_transformation_type") and n.get("cocos") is None for n in names
    ):
        with nullcontext(gc) if gc is not None else GraphClient() as _conv_gc:
            _conv_row = _conv_gc.query(
                "MATCH (v:DDVersion {is_current: true}) RETURN v.cocos AS cocos"
            )
        _convention = (
            _conv_row[0]["cocos"]
            if _conv_row and _conv_row[0].get("cocos") is not None
            else None
        )
        if _convention is not None:
            for n in names:
                if n.get("cocos_transformation_type") and n.get("cocos") is None:
                    n["cocos"] = _convention

    # Guard: warn when cocos_transformation_type is set but cocos integer is
    # still missing (e.g. no current DD convention resolvable).
    for n in names:
        if n.get("cocos_transformation_type") and n.get("cocos") is None:
            logger.warning(
                "StandardName '%s' has cocos_transformation_type='%s' but no cocos "
                "integer — HAS_COCOS edge will not be created",
                n["id"],
                n["cocos_transformation_type"],
            )

    if gc is None and isinstance(GraphClient, type):
        with GraphClient() as client:
            return write_standard_names(
                names,
                override=override,
                gc=client,
                return_written_ids=return_written_ids,
            )
    if gc is not None and type(gc) is GraphClient:
        from imas_codex.standard_names.signed_manifest import _TransactionQuery

        with gc.session() as session:
            transaction = session.begin_transaction()
            try:
                result = write_standard_names(
                    names,
                    override=override,
                    gc=_TransactionQuery(transaction),
                    return_written_ids=return_written_ids,
                )
                transaction.commit()
                return result
            except BaseException:
                transaction.rollback()
                raise

    with nullcontext(gc) if gc is not None else GraphClient() as write_gc:
        # Conflict-detect on unit — same name with different unit is a data
        # integrity error.  Filter out conflicting entries rather than raising
        # so that non-conflicting entries can still proceed.
        unit_check_batch = [
            {"id": n["id"], "unit": n.get("unit")} for n in names if n.get("unit")
        ]
        if unit_check_batch:
            unit_conflicts = list(
                write_gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.id})
                    WHERE sn.unit IS NOT NULL AND b.unit IS NOT NULL
                      AND sn.unit <> b.unit
                    RETURN sn.id AS name,
                           sn.unit AS existing_unit,
                           b.unit AS incoming_unit
                    """,
                    batch=unit_check_batch,
                )
                or []
            )
            if unit_conflicts:
                conflict_details = "; ".join(
                    f"{c['name']}: {c['existing_unit']} vs {c['incoming_unit']}"
                    for c in unit_conflicts
                )
                logger.warning("Unit conflicts detected: %s", conflict_details)
                conflicting_ids = {c["name"] for c in unit_conflicts}
                names = [n for n in names if n["id"] not in conflicting_ids]
                if not names:
                    logger.warning("All entries had unit conflicts — nothing to write")
                    return [] if return_written_ids else 0

        # MERGE StandardName nodes with provenance — coalesce to preserve existing data
        write_gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.id})
            ON CREATE SET sn.status = 'draft', sn.updated_at = datetime()
            SET sn.updated_at = datetime(),
                // A null catalog status is outside the enumerated vocabulary
                // for a pipeline-minted identity; only fill the gap, never
                // overwrite a non-null status this write did not set.
                sn.status = coalesce(sn.status, 'draft'),
                sn.source_types = coalesce(b.source_types, sn.source_types),
                sn.description = coalesce(nullIf(b.description, ''), sn.description),
                sn.documentation = coalesce(b.documentation, sn.documentation),
                sn.kind = coalesce(b.kind, sn.kind),
                sn.links = coalesce(b.links, sn.links),
                sn.source_paths = coalesce(b.source_paths, sn.source_paths),
                sn.validity_domain = coalesce(b.validity_domain, sn.validity_domain),
                sn.constraints = coalesce(b.constraints, sn.constraints),
                sn.unit = coalesce(b.unit, sn.unit),
                sn.cocos_transformation_type = coalesce(b.cocos_transformation_type, sn.cocos_transformation_type),
                sn.cocos = coalesce(b.cocos, sn.cocos),
                sn.dd_version = coalesce(b.dd_version, sn.dd_version),
                sn.isn_version = coalesce(b.isn_version, sn.isn_version),
                sn.codex_version = coalesce(b.codex_version, sn.codex_version),
                sn.model = coalesce(b.model, sn.model),
                sn.generated_at = coalesce(b.generated_at, sn.generated_at),
                sn.review_tier = coalesce(b.review_tier, sn.review_tier),
                sn.vocab_gap_detail = coalesce(b.vocab_gap_detail, sn.vocab_gap_detail),
                sn.validation_issues = coalesce(b.validation_issues, sn.validation_issues),
                sn.validation_layer_summary = coalesce(b.validation_layer_summary, sn.validation_layer_summary),
                sn.validation_status = coalesce(b.validation_status, sn.validation_status),
                sn.validated_at = CASE WHEN b.validation_status IS NOT NULL
                                  THEN datetime()
                                  ELSE sn.validated_at END,
                sn.link_status = coalesce(b.link_status, sn.link_status),
                sn.review_input_hash = b.review_input_hash,
                sn.embedding = coalesce(b.embedding, sn.embedding),
                sn.embedded_at = coalesce(b.embedded_at, sn.embedded_at),
                sn.grammar_parse_version = b.grammar_parse_version,
                sn.validation_diagnostics_json = b.validation_diagnostics_json,
                sn.physical_base = b.physical_base,
                sn.geometric_base = b.geometric_base,
                sn.subject = b.subject,
                sn.component = b.component,
                sn.coordinate = b.coordinate,
                sn.transformation = b.transformation,
                sn.position = b.position,
                sn.process = b.process,
                sn.device = b.device,
                sn.region = b.region,
                sn.aggregation = b.aggregation,
                sn.orbit = b.orbit,
                sn.population = b.population,
                sn.object = b.object,
                sn.geometry = b.geometry,
                sn.llm_cost_refine_name = CASE WHEN sn.generate_name_count IS NOT NULL
                                             AND sn.generate_name_count > 0
                                             AND b.llm_cost IS NOT NULL
                                        THEN coalesce(sn.llm_cost_refine_name, 0.0) + b.llm_cost
                                        ELSE sn.llm_cost_refine_name END,
                sn.refine_name_count = CASE WHEN sn.generate_name_count IS NOT NULL
                                            AND sn.generate_name_count > 0
                                            AND b.llm_cost IS NOT NULL
                                       THEN coalesce(sn.refine_name_count, 0) + 1
                                       ELSE sn.refine_name_count END,
                sn.llm_cost_generate_name = CASE WHEN b.llm_cost IS NOT NULL
                                      THEN coalesce(sn.llm_cost_generate_name, 0.0) + b.llm_cost
                                      ELSE sn.llm_cost_generate_name END,
                sn.generate_name_count = CASE WHEN b.llm_cost IS NOT NULL
                                   THEN coalesce(sn.generate_name_count, 0) + 1
                                   ELSE sn.generate_name_count END,
                sn.regen_count = CASE WHEN b.regen_increment = true
                                 THEN coalesce(sn.regen_count, 0) + 1
                                 ELSE sn.regen_count END,
                sn.llm_cost = CASE WHEN b.llm_cost IS NOT NULL
                              THEN coalesce(sn.llm_cost, 0.0) + b.llm_cost
                              ELSE sn.llm_cost END,
                sn.llm_model = coalesce(b.llm_model, sn.llm_model),
                sn.llm_service = coalesce(b.llm_service, sn.llm_service),
                sn.llm_at = coalesce(b.llm_at, sn.llm_at),
                sn.llm_tokens_in = coalesce(b.llm_tokens_in, sn.llm_tokens_in),
                sn.llm_tokens_out = coalesce(b.llm_tokens_out, sn.llm_tokens_out),
                sn.llm_tokens_cached_read = coalesce(b.llm_tokens_cached_read, sn.llm_tokens_cached_read),
                sn.llm_tokens_cached_write = coalesce(b.llm_tokens_cached_write, sn.llm_tokens_cached_write),
                sn.created_at = coalesce(sn.created_at, datetime())
            """,
            batch=[
                {
                    "id": n["id"],
                    "source_types": n.get("source_types") or None,
                    "description": n.get("description"),
                    "documentation": n.get("documentation"),
                    "kind": n.get("kind"),
                    "links": n.get("links") or None,
                    "source_paths": n.get("source_paths") or None,
                    "validity_domain": n.get("validity_domain"),
                    "constraints": n.get("constraints") or None,
                    "unit": n.get("unit"),
                    "cocos_transformation_type": n.get("cocos_transformation_type"),
                    "cocos": n.get("cocos"),
                    "dd_version": n.get("dd_version"),
                    "isn_version": n.get("isn_version"),
                    "codex_version": n.get("codex_version"),
                    "model": n.get("model"),
                    "generated_at": n.get("generated_at"),
                    "review_tier": n.get("review_tier"),
                    "vocab_gap_detail": _ensure_json(n.get("vocab_gap_detail")),
                    "validation_issues": n.get("validation_issues") or None,
                    "validation_layer_summary": _ensure_json(
                        n.get("validation_layer_summary")
                    ),
                    "validation_status": n.get("validation_status"),
                    "link_status": _compute_link_status(n.get("links")),
                    "review_input_hash": n.get("review_input_hash"),
                    "embedding": n.get("embedding"),
                    "embedded_at": n.get("embedded_at"),
                    "llm_cost": n.get("llm_cost"),
                    "llm_model": n.get("llm_model"),
                    "llm_service": n.get("llm_service"),
                    "llm_at": n.get("llm_at"),
                    "llm_tokens_in": n.get("llm_tokens_in"),
                    "llm_tokens_out": n.get("llm_tokens_out"),
                    "llm_tokens_cached_read": n.get("llm_tokens_cached_read"),
                    "llm_tokens_cached_write": n.get("llm_tokens_cached_write"),
                    "regen_increment": n.get("regen_increment"),
                    **_parse_grammar(n["id"]),
                }
                for n in names
            ],
        )

        # Promote-on-higher-rank: read existing scalar primary +
        # source_domains, compute promotion in Python, write back.
        # Replaces the legacy list-append-with-dedupe semantics.
        from imas_codex.standard_names.domain_ranking import (
            maybe_promote_domain,
            merge_source_domains,
        )

        pd_inputs = []
        for n in names:
            sn_id = n.get("id")
            if not sn_id:
                continue
            candidate = _scalar_domain(n.get("physics_domain"))
            incoming_sources = _ensure_list(n.get("source_domains"))
            if candidate and candidate not in incoming_sources:
                incoming_sources = [*incoming_sources, candidate]
            if not candidate and not incoming_sources:
                continue
            pd_inputs.append(
                {
                    "id": sn_id,
                    "candidate": candidate,
                    "incoming_sources": incoming_sources,
                }
            )

        if pd_inputs:
            # Read existing values for this batch so we can compute
            # promotion in Python (rank tables are not encodable in Cypher).
            existing_rows = (
                write_gc.query(
                    """
                    UNWIND $ids AS sid
                    MATCH (sn:StandardName {id: sid})
                    RETURN sn.id AS id,
                           sn.physics_domain AS physics_domain,
                           sn.source_domains AS source_domains
                    """,
                    ids=[p["id"] for p in pd_inputs],
                )
                or []
            )
            existing_map = {
                row["id"]: row
                for row in existing_rows  # type: ignore[index]
            }

            pd_batch = []
            for p in pd_inputs:
                row = existing_map.get(p["id"], {})
                existing_primary = _scalar_domain(row.get("physics_domain"))
                existing_sources = _ensure_list(row.get("source_domains"))
                # Legacy nodes may have list-valued physics_domain — fold
                # those into source_domains too.
                legacy_list = (
                    row.get("physics_domain")
                    if isinstance(row.get("physics_domain"), list)
                    else None
                )
                if legacy_list:
                    existing_sources = list(
                        dict.fromkeys([*existing_sources, *legacy_list])
                    )

                merged_sources = merge_source_domains(
                    existing_sources, *p["incoming_sources"]
                )
                promoted = maybe_promote_domain(existing_primary, p["candidate"])
                pd_batch.append(
                    {
                        "id": p["id"],
                        "physics_domain": promoted,
                        "source_domains": merged_sources or None,
                    }
                )

            write_gc.query(
                """
                UNWIND $batch AS b
                MERGE (sn:StandardName {id: b.id})
                SET sn.updated_at = datetime(), sn.physics_domain = coalesce(b.physics_domain, sn.physics_domain),
                    sn.source_domains = coalesce(b.source_domains, sn.source_domains)
                """,
                batch=pd_batch,
            )

        # Create HAS_STANDARD_NAME relationships: entity → concept
        dd_names = [n for n in names if "dd" in (n.get("source_types") or [])]
        signal_names = [n for n in names if "signals" in (n.get("source_types") or [])]

        if dd_names:
            from imas_codex.standard_names.workers import _is_attachment_consistent

            pairing_rows = write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                OPTIONAL MATCH (src:IMASNode {id: b.source_id})
                OPTIONAL MATCH (src)-[:HAS_UNIT]->(du:Unit)
                OPTIONAL MATCH (bound:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
                RETURN b.id AS id, b.source_id AS source_id,
                       coalesce(du.id, src.unit) AS dd_unit,
                       sn.unit AS sn_unit,
                       collect(DISTINCT bound.id) AS existing_dd_paths
                """,
                batch=[
                    {"id": n["id"], "source_id": n["source_id"]}
                    for n in dd_names
                    if n.get("source_id")
                ],
            )
            accepted_by_name: dict[str, list[str]] = {}
            accepted_pairs: list[dict[str, str]] = []
            for row in sorted(
                pairing_rows, key=lambda item: (item["id"], item["source_id"])
            ):
                accepted_paths = accepted_by_name.setdefault(
                    row["id"], sorted(row.get("existing_dd_paths") or [])
                )
                accepted, reason = _is_attachment_consistent(
                    row["source_id"],
                    row["id"],
                    existing_sources=tuple(accepted_paths),
                    dd_unit=row.get("dd_unit"),
                    sn_unit=row.get("sn_unit"),
                )
                if accepted:
                    accepted_pairs.append(
                        {"id": row["id"], "source_id": row["source_id"]}
                    )
                    accepted_paths.append(row["source_id"])
                else:
                    logger.warning(
                        "write_standard_names: rejected DD pairing %s → %s: %s",
                        row["source_id"],
                        row["id"],
                        reason,
                    )
            write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                """,
                batch=accepted_pairs,
            )
        if signal_names:
            write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (src:FacilitySignal {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                """,
                batch=[
                    {"id": n["id"], "source_id": n["source_id"]}
                    for n in signal_names
                    if n.get("source_id")
                ],
            )

        # Create HAS_UNIT relationships: StandardName → Unit
        #
        # Self-healing: for each SN carrying a unit in this batch, drop ALL
        # existing HAS_UNIT edges before re-creating, so a unit correction
        # (e.g. m.sr → m^2.sr) replaces the edge rather than leaving a stale
        # one alongside the new one. An SN must have at most ONE HAS_UNIT edge,
        # matching its sn.unit property. Mirrors the IMASNode self-heal in
        # build_dd.py. Scoped to the batch's SN ids that carry a unit — an SN
        # not supplying a unit in this batch (e.g. a docs-only refine) keeps
        # its existing edge untouched.
        units_batch = [
            {"id": n["id"], "unit": n["unit"]} for n in names if n.get("unit")
        ]
        if units_batch:
            unit_sn_ids = [b["id"] for b in units_batch]
            write_gc.query(
                """
                UNWIND $ids AS id
                MATCH (sn:StandardName {id: id})-[r:HAS_UNIT]->()
                DELETE r
                """,
                ids=unit_sn_ids,
            )
            write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MERGE (u:Unit {id: b.unit})
                SET u.symbol = coalesce(u.symbol, b.unit)
                MERGE (sn)-[:HAS_UNIT]->(u)
                """,
                batch=units_batch,
            )

        # Create HAS_COCOS relationships: StandardName → COCOS
        # Use MATCH (not MERGE) — COCOS singleton nodes already exist.
        cocos_batch = [
            {"id": n["id"], "cocos": n["cocos"]}
            for n in names
            if n.get("cocos") is not None
        ]
        if cocos_batch:
            write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (c:COCOS {id: b.cocos})
                MERGE (sn)-[:HAS_COCOS]->(c)
                """,
                batch=cocos_batch,
            )

        # Create grammar decomposition: typed edges + per-segment columns
        token_miss_gaps = _write_grammar_decomposition(
            write_gc, [n["id"] for n in names]
        )

        # Emit structural edges: HAS_PARENT, HAS_ERROR, HAS_PREDECESSOR,
        # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN.
        # Tail pass — all nodes in this batch exist before edges are written.
        skeleton_candidate_ids = _write_standard_name_edges(write_gc, names)

        from imas_codex.standard_names.source_invariant import require_producing_sources

        require_producing_sources(write_gc, [n["id"] for n in names])

    # Persist token-miss gaps through the same caller-owned graph handle when
    # one is supplied, so transactional callers keep every rich-write effect.
    if token_miss_gaps:
        # Build sn_id → source mapping from the names list
        sn_source_map: dict[str, tuple[str, str]] = {}
        for n in names:
            sn_id = n["id"]
            source_id = n.get("source_id")
            source_type = "dd" if "dd" in (n.get("source_types") or []) else "signals"
            if source_id and sn_id not in sn_source_map:
                sn_source_map[sn_id] = (source_id, source_type)

        # Group gaps by source_type for write_vocab_gaps
        dd_gap_dicts: list[dict[str, str]] = []
        signal_gap_dicts: list[dict[str, str]] = []
        for gap in token_miss_gaps:
            mapping = sn_source_map.get(gap["sn_id"])
            if not mapping:
                continue
            source_id, source_type = mapping
            gap_dict = {
                "source_id": source_id,
                "segment": gap["segment"],
                "token": gap["token"],
                "reason": f"Token-miss during grammar edge writing for '{gap['sn_id']}'",
            }
            if source_type == "dd":
                dd_gap_dicts.append(gap_dict)
            else:
                signal_gap_dicts.append(gap_dict)

        if dd_gap_dicts:
            write_vocab_gaps(dd_gap_dicts, source_type="dd", gc=gc)
        if signal_gap_dicts:
            write_vocab_gaps(signal_gap_dicts, source_type="signals", gc=gc)

    # Sweep skeleton placeholders created by this write's relationship-side
    # MERGE on uncomposed targets (HAS_PARENT, HAS_ERROR, HAS_PREDECESSOR,
    # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN). A real StandardName
    # always has at least a created_at OR generated_at timestamp; pure
    # skeletons (id-only) are detached and deleted.
    # Reuse a caller-owned transaction when present; ordinary callers retain
    # the independent graph context used by the public writer. An empty
    # candidate set performs no query: it must never degrade into a global
    # placeholder search.
    swept_count = 0
    if skeleton_candidate_ids:
        with nullcontext(gc) if gc is not None else GraphClient() as sweep_gc:
            skeleton_placeholder_ids = _query_skeleton_placeholders_for_cleanup(
                sweep_gc, skeleton_candidate_ids
            )
            if skeleton_placeholder_ids:
                refuse_protected_automatic_deletion(
                    sweep_gc,
                    skeleton_placeholder_ids,
                    operation="skeleton placeholder cleanup",
                )
                deletion_clause = deletion_change_cypher("sn")
                swept = sweep_gc.query(
                    f"""
                    MATCH (sn:StandardName)
                    WHERE sn.id IN $candidate_ids
                      AND {_SKELETON_PLACEHOLDER_PREDICATE}
                    {deletion_clause}
                    DETACH DELETE sn
                    RETURN count(sn) AS swept
                    """,
                    candidate_ids=skeleton_placeholder_ids,
                    **deletion_change_params(
                        "remove_skeleton_placeholder",
                        reason="relationship placeholder never became a composed name",
                    ),
                )
                swept_count = int(swept[0]["swept"]) if swept else 0
    if swept_count:
        logger.info("Swept %d skeleton StandardName placeholder(s)", swept_count)

    written = len(names)
    logger.info("Wrote %d StandardName nodes", written)
    if return_written_ids:
        return [name["id"] for name in names]
    return written


class ReviewWriteResult(int):
    """Written review count with suggestion-clearing measurements."""

    suggestions_cleared_non_distinct: int
    suggestions_cleared_unparseable: int

    def __new__(
        cls,
        written: int,
        *,
        suggestions_cleared_non_distinct: int = 0,
        suggestions_cleared_unparseable: int = 0,
    ) -> ReviewWriteResult:
        result = super().__new__(cls, written)
        result.suggestions_cleared_non_distinct = suggestions_cleared_non_distinct
        result.suggestions_cleared_unparseable = suggestions_cleared_unparseable
        return result


def write_reviews(
    records: list[dict[str, Any]], *, skip_cost: bool = False
) -> ReviewWriteResult:
    """MERGE ``StandardNameReview`` nodes and ``HAS_REVIEW`` edges from StandardName.

    Each record must contain:

    - ``id`` (str) — composite key
      ``{standard_name_id}:{axis}:{review_group_id}:{cycle_index}``
    - ``standard_name_id`` (str) — parent StandardName
    - ``model`` (str), ``model_family`` (str), ``is_canonical`` (bool)
    - ``score`` (float 0-1), ``scores_json`` (str), ``tier`` (str)
    - ``reviewed_at`` (str ISO 8601)

    RD-quorum fields (required for new-style reviews):
    - ``review_axis`` (str) — "name" or "docs"
    - ``cycle_index`` (int) — 0, 1, or 2
    - ``review_group_id`` (str) — UUID
    - ``resolution_role`` (str) — "primary", "secondary", or "escalator"
    - ``resolution_method`` (str | None)

    Optional: ``comments`` (str), ``llm_cost`` (float),
    ``llm_tokens_in`` (int), ``llm_tokens_out`` (int),
    ``llm_model`` (str), ``llm_at`` (str), ``llm_service`` (str).

    Version provenance (auto-injected if not provided):
    ``codex_version`` (str), ``isn_version`` (str).

    MERGE-by-``id`` semantics make re-runs idempotent when the same
    model reviews the same name at the same timestamp.

    Parameters
    ----------
    records:
        Review record dicts.
    skip_cost:
        When ``True``, skip accumulating ``llm_cost_review_name`` /
        ``llm_cost_review_docs`` on StandardName nodes.  Use when StandardNameReview records were
        already persisted inline (crash-safety path) to avoid
        double-counting.

    Returns the number of StandardNameReview records written, with counts of
    suggestions cleared as non-distinct or unparseable on the integer result.
    """
    if not records:
        return ReviewWriteResult(0)
    # Guard: must attach to an existing StandardName.
    valid = [r for r in records if r.get("id") and r.get("standard_name_id")]
    if not valid:
        return ReviewWriteResult(0)

    from imas_standard_names import compose, parse

    from imas_codex.standard_names.models import suggestion_is_semantically_distinct

    suggestions_cleared_non_distinct = 0
    suggestions_cleared_unparseable = 0
    prepared: list[dict[str, Any]] = []
    for record in valid:
        proposed_spelling = record.get("suggested_name")
        if proposed_spelling and not suggestion_is_semantically_distinct(
            str(record["standard_name_id"]), str(proposed_spelling)
        ):
            try:
                compose(parse(str(proposed_spelling)).ir)
            except Exception:
                suggestions_cleared_unparseable += 1
            else:
                suggestions_cleared_non_distinct += 1
            record = {
                **record,
                "suggested_name": None,
                "suggestion_justification": None,
            }
        prepared.append(record)
    valid = prepared

    # Auto-inject codex and ISN versions (same pattern as persist_generated_name_batch)
    try:
        import imas_standard_names

        _isn_ver = imas_standard_names.__version__
    except (ImportError, AttributeError):
        _isn_ver = None
    try:
        import importlib.metadata

        _codex_ver = importlib.metadata.version("imas-codex")
    except Exception:
        _codex_ver = None
    for r in valid:
        r.setdefault("isn_version", _isn_ver)
        r.setdefault("codex_version", _codex_ver)

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (r:StandardNameReview {id: b.id})
            SET r.standard_name_id = b.standard_name_id,
                r.model = b.model,
                r.reviewer_model = b.reviewer_model,
                r.model_family = b.model_family,
                r.is_canonical = b.is_canonical,
                r.score = b.score,
                r.judge_kind = b.judge_kind,
                r.judgment_model = b.judgment_model,
                r.judge_answers_json = b.judge_answers_json,
                r.judge_probs_json = b.judge_probs_json,
                r.judge_confidence_json = b.judge_confidence_json,
                r.scores_json = b.scores_json,
                r.tier = b.tier,
                r.comments = b.comments,
                r.comments_per_dim_json = b.comments_per_dim_json,
                r.suggested_name = b.suggested_name,
                r.suggestion_justification = b.suggestion_justification,
                r.reviewed_at = CASE
                    WHEN b.reviewed_at IS NULL THEN NULL
                    ELSE datetime(b.reviewed_at)
                END,
                r.review_axis = b.review_axis,
                r.cycle_index = b.cycle_index,
                r.review_group_id = b.review_group_id,
                r.resolution_role = b.resolution_role,
                r.resolution_method = b.resolution_method,
                r.llm_model = b.llm_model,
                r.llm_cost = b.llm_cost,
                r.llm_tokens_in = b.llm_tokens_in,
                r.llm_tokens_out = b.llm_tokens_out,
                r.llm_tokens_cached_read = b.llm_tokens_cached_read,
                r.llm_tokens_cached_write = b.llm_tokens_cached_write,
                r.llm_at = CASE
                    WHEN b.llm_at IS NULL THEN NULL
                    ELSE datetime(b.llm_at)
                END,
                r.llm_service = b.llm_service,
                r.codex_version = b.codex_version,
                r.isn_version = b.isn_version
            WITH r, b
            MATCH (sn:StandardName {id: b.standard_name_id})
            MERGE (sn)-[:HAS_REVIEW]->(r)
            """,
            batch=[
                {
                    "id": r["id"],
                    "standard_name_id": r["standard_name_id"],
                    "model": r.get("model") or "",
                    # reviewer_model is the consumer-facing alias; fall back to model
                    "reviewer_model": r.get("reviewer_model") or r.get("model") or "",
                    "model_family": r.get("model_family") or "other",
                    "is_canonical": bool(r.get("is_canonical", False)),
                    "score": None
                    if r.get("judge_kind") == "jev"
                    else float(r.get("score") or 0.0),
                    "judge_kind": r.get("judge_kind"),
                    "judgment_model": r.get("judgment_model"),
                    "judge_answers_json": _ensure_json(r.get("judge_answers_json")),
                    "judge_probs_json": _ensure_json(r.get("judge_probs_json")),
                    "judge_confidence_json": _ensure_json(
                        r.get("judge_confidence_json")
                    ),
                    "scores_json": _ensure_json(r.get("scores_json") or "{}"),
                    "tier": r.get("tier") or "unknown",
                    "comments": r.get("comments") or "",
                    "comments_per_dim_json": _ensure_json(
                        r.get("comments_per_dim_json")
                    ),
                    "suggested_name": r.get("suggested_name") or "",
                    "suggestion_justification": r.get("suggestion_justification") or "",
                    "reviewed_at": r.get("reviewed_at"),
                    "review_axis": r.get("review_axis"),
                    "cycle_index": r.get("cycle_index"),
                    "review_group_id": r.get("review_group_id"),
                    "resolution_role": r.get("resolution_role"),
                    "resolution_method": r.get("resolution_method"),
                    "llm_model": r.get("llm_model"),
                    # Defensive: coalesce None → 0 so every Review has
                    # non-NULL cost/token fields (quarantine & cache-hit).
                    "llm_cost": r.get("llm_cost")
                    if r.get("llm_cost") is not None
                    else 0.0,
                    "llm_tokens_in": r.get("llm_tokens_in")
                    if r.get("llm_tokens_in") is not None
                    else 0,
                    "llm_tokens_out": r.get("llm_tokens_out")
                    if r.get("llm_tokens_out") is not None
                    else 0,
                    "llm_tokens_cached_read": r.get("llm_tokens_cached_read")
                    if r.get("llm_tokens_cached_read") is not None
                    else 0,
                    "llm_tokens_cached_write": r.get("llm_tokens_cached_write")
                    if r.get("llm_tokens_cached_write") is not None
                    else 0,
                    "llm_at": r.get("llm_at"),
                    "llm_service": r.get("llm_service"),
                    "codex_version": r.get("codex_version"),
                    "isn_version": r.get("isn_version"),
                }
                for r in valid
            ],
        )

        # --- Accumulate review cost on StandardName ---
        # Build per-SN cost totals from the batch, split by review axis.
        # Skipped when skip_cost=True (records already persisted inline).
        if not skip_cost:
            name_cost_map: dict[str, float] = {}
            docs_cost_map: dict[str, float] = {}
            for r in valid:
                sn_id = r.get("standard_name_id")
                cost = r.get("llm_cost")
                if sn_id and cost:
                    axis = r.get("review_axis", "name")
                    if axis == "docs":
                        docs_cost_map[sn_id] = docs_cost_map.get(sn_id, 0.0) + cost
                    else:
                        name_cost_map[sn_id] = name_cost_map.get(sn_id, 0.0) + cost
            if name_cost_map:
                gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.updated_at = datetime(), sn.llm_cost_review_name = coalesce(sn.llm_cost_review_name, 0.0) + b.cost,
                        sn.review_name_count = coalesce(sn.review_name_count, 0) + 1,
                        sn.llm_cost = coalesce(sn.llm_cost, 0.0) + b.cost
                    """,
                    batch=[
                        {"sn_id": sn_id, "cost": cost}
                        for sn_id, cost in name_cost_map.items()
                    ],
                )
            if docs_cost_map:
                gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.updated_at = datetime(), sn.llm_cost_review_docs = coalesce(sn.llm_cost_review_docs, 0.0) + b.cost,
                        sn.review_docs_count = coalesce(sn.review_docs_count, 0) + 1,
                        sn.llm_cost = coalesce(sn.llm_cost, 0.0) + b.cost
                    """,
                    batch=[
                        {"sn_id": sn_id, "cost": cost}
                        for sn_id, cost in docs_cost_map.items()
                    ],
                )

    logger.info(
        "Wrote %d StandardNameReview nodes; cleared suggestions: "
        "non-distinct=%d, unparseable=%d",
        len(valid),
        suggestions_cleared_non_distinct,
        suggestions_cleared_unparseable,
    )
    return ReviewWriteResult(
        len(valid),
        suggestions_cleared_non_distinct=suggestions_cleared_non_distinct,
        suggestions_cleared_unparseable=suggestions_cleared_unparseable,
    )


def update_review_aggregates(
    standard_name_ids: list[str],
    *,
    threshold: float = 0.2,
) -> int:
    """Recompute per-StandardName aggregates from attached StandardNameReview nodes.

    Sets ``review_count``, ``review_mean_score``, and ``review_disagreement``
    across attached reviews except non-canonical Jev shadow judgments.
    Axis-specific score and resolution-method mirrors are owned by their
    dedicated write paths.

    Returns the number of StandardName nodes updated.
    """
    if not standard_name_ids:
        return 0
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE NOT (coalesce(sn.name_stage, '') IN $frozen_name_stages)
            OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
            WHERE NOT (coalesce(r.judge_kind, '') = 'jev'
                       AND coalesce(r.is_canonical, false) = false)
            WITH sn, count(r) AS n, avg(r.score) AS mean,
                 CASE WHEN count(r) > 1 THEN max(r.score) - min(r.score) ELSE 0.0 END AS spread
            SET sn.updated_at = datetime(), sn.review_count = n,
                sn.review_mean_score = CASE WHEN n > 0 THEN mean ELSE null END,
                sn.review_disagreement = (n > 1 AND spread >= $threshold)
            RETURN sn.id AS id
            """,
            ids=standard_name_ids,
            threshold=float(threshold),
            frozen_name_stages=list(_PIPELINE_FROZEN_NAME_STAGES),
        )
        return len(list(rows or []))


# ─────────────────────────────────────────────────────────────────────────────
# Voiding one review dimension
# ─────────────────────────────────────────────────────────────────────────────

_DIMENSION_SCALE = 20.0


class ReviewDimensionVoidRefused(ValueError):
    """A dimension void was refused; nothing was written."""


def _review_tier(score: float) -> str:
    """Tier label for a normalised score, matching the review persist path."""
    if score >= 0.85:
        return "outstanding"
    if score >= 0.60:
        return "good"
    if score >= 0.40:
        return "inadequate"
    return "poor"


def plan_review_dimension_void(
    *,
    scores: dict[str, Any],
    void_records: list[dict[str, Any]] | None,
    dimension: str,
    actor: str,
    reason: str,
    grammar_signature: str | None,
    at: str | None = None,
) -> dict[str, Any]:
    """Decide one dimension void without touching the graph.

    The review score is a stored mirror of a derivation — ``sum(dimensions) /
    (len(dimensions) * 20)`` at review time — so a void is expressible as a
    recomputation over the surviving dimensions. That is the whole reason this
    route can be honoured by consumers that know nothing about it: every reader
    of ``r.score`` reads the recomputed number.

    Refusals are decided here so that a caller writing only on success cannot
    half-apply one. What justifies a void is the caller's business; what this
    enforces is that a void is attributable, reasoned, non-duplicating, and
    never the whole review.

    The original per-dimension scores are NOT modified. ``scores`` is evidence
    and is returned unchanged; the void record sits beside it.

    Raises :class:`ReviewDimensionVoidRefused` for the four refusals — an
    absent dimension, the last surviving dimension, an empty reason, and a
    dimension already voided.
    """
    if not str(reason or "").strip():
        raise ReviewDimensionVoidRefused(
            f"voiding {dimension!r} needs a reason: the reason is the record, "
            "and a void without one is indistinguishable from a lost score"
        )
    if not str(actor or "").strip():
        raise ReviewDimensionVoidRefused(
            f"voiding {dimension!r} needs an actor: an unattributed void "
            "cannot be adjudicated later"
        )

    numeric = {
        key: float(value)
        for key, value in (scores or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if dimension not in numeric:
        raise ReviewDimensionVoidRefused(
            f"{dimension!r} is not a scored dimension of this review "
            f"(scored: {', '.join(sorted(numeric)) or 'none'})"
        )

    records = [dict(record) for record in (void_records or [])]
    already = [
        record
        for record in records
        if record.get("dimension") == dimension and record.get("void_active", True)
    ]
    if already:
        raise ReviewDimensionVoidRefused(
            f"{dimension!r} is already voided on this review "
            f"({already[0].get('void_reason')!r}) — one dimension carries one "
            "record, so a second void would only duplicate it"
        )

    voided_names = {
        record["dimension"]
        for record in records
        if record.get("void_active", True) and record.get("dimension") in numeric
    }
    surviving = {
        key: value for key, value in numeric.items() if key not in voided_names
    }
    surviving.pop(dimension, None)
    if not surviving:
        raise ReviewDimensionVoidRefused(
            f"{dimension!r} is the last surviving dimension of this review — a "
            "review with every dimension void is not evidence of anything, and "
            "discarding it is a whole-review decision, not a dimension void"
        )

    record = {
        "dimension": dimension,
        "void_active": True,
        "voided_at": at or datetime.now(UTC).isoformat(),
        "void_actor": str(actor),
        "void_reason": str(reason),
        "void_grammar_signature": grammar_signature,
    }
    records.append(record)
    score = (sum(surviving.values()) / len(surviving)) / _DIMENSION_SCALE
    return {
        "void_records": records,
        "void_record": record,
        "surviving_dimensions": dict(sorted(surviving.items())),
        "score": score,
        "tier": _review_tier(score),
    }


_VOID_REVIEW_DIMENSION_READ = """
MATCH (r:StandardNameReview {id: $review_id})
RETURN r.id AS id,
       r.standard_name_id AS standard_name_id,
       r.review_axis AS review_axis,
       r.score AS score,
       r.scores_json AS scores_json,
       r.voided_dimensions_json AS voided_dimensions_json
"""

# scores_json is deliberately absent from this SET: the original per-dimension
# score is evidence and survives the void untouched.
_VOID_REVIEW_DIMENSION_WRITE = """
MATCH (r:StandardNameReview {id: $review_id})
SET r.score = $score,
    r.tier = $tier,
    r.voided_dimensions_json = $voided_dimensions_json
RETURN r.id AS id
"""


def void_review_dimension(
    review_id: str,
    dimension: str,
    *,
    actor: str,
    reason: str,
    grammar_signature: str | None = None,
    gc: Any | None = None,
    dry_run: bool = False,
    refresh_aggregates: bool = True,
) -> dict[str, Any]:
    """Void ONE scored dimension of a review and record why in the graph.

    A dimension can be discredited without the rest of the review being wrong.
    Deleting it would destroy the evidence and leave the surviving score
    unexplained, so the dimension's number stays exactly where it was and gains
    a void record beside it carrying the actor, the reason, the timestamp and
    the vocabulary signature observed at that moment.

    ``reason`` is the caller's free text and carries the entire justification;
    this route asserts nothing about what discredits a dimension, and must not
    be read as encoding one cause. The signature and timestamp are recorded as
    facts about the environment at void time rather than as evidence for the
    decision — ``isn_version`` is build-time metadata from the installed
    distribution and reports what was packaged, not which vocabulary or prompt
    the review ran under, so neither question can be answered after the fact
    without capturing the answer here.

    The review's ``score`` is recomputed over the surviving dimensions. Because
    ``score`` was always a derivation of the per-dimension scores, that single
    write is what makes ``update_review_aggregates``, the canonical-review
    projection, the reviewer benchmarks and every other reader of ``r.score``
    honour the void — a marker nothing consults would be worse than no marker.

    ``review_mean_score`` is the one reader that cannot follow on its own: it is
    a STORED average, refreshed only when ``update_review_aggregates`` runs. So
    the void refreshes it for the reviewed name, leaving no window in which the
    review says one thing and the name-level aggregate another. Pass
    ``refresh_aggregates=False`` when voiding in bulk and aggregating once at
    the end.

    Also writes a ``StandardNameChange``, the same ledger a detach judgement
    writes, so the decision survives independently of the review node.

    Refuses — raising :class:`ReviewDimensionVoidRefused` with nothing written —
    when the dimension is not scored on this review, when it is the last
    surviving one, when no reason is given, and when it is already voided.
    """
    own = gc is None
    client: Any = GraphClient() if own else gc
    try:
        rows = list(client.query(_VOID_REVIEW_DIMENSION_READ, review_id=review_id))
        if not rows:
            raise ReviewDimensionVoidRefused(
                f"no StandardNameReview {review_id!r} — nothing to void"
            )
        row = dict(rows[0])
        try:
            scores = json.loads(row.get("scores_json") or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ReviewDimensionVoidRefused(
                f"review {review_id!r} has unreadable per-dimension scores"
            ) from exc
        try:
            existing = json.loads(row.get("voided_dimensions_json") or "[]")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ReviewDimensionVoidRefused(
                f"review {review_id!r} has an unreadable void record"
            ) from exc

        if grammar_signature is None:
            with suppress(Exception):
                grammar_signature = isn_vocabulary_signature()

        plan = plan_review_dimension_void(
            scores=scores,
            void_records=existing if isinstance(existing, list) else [],
            dimension=dimension,
            actor=actor,
            reason=reason,
            grammar_signature=grammar_signature,
        )
        result = {
            "ok": True,
            "review_id": review_id,
            "standard_name_id": row.get("standard_name_id"),
            "review_axis": row.get("review_axis"),
            "dimension": dimension,
            "score_before": row.get("score"),
            "score": plan["score"],
            "tier": plan["tier"],
            "surviving_dimensions": plan["surviving_dimensions"],
            "void_record": plan["void_record"],
            "dry_run": dry_run,
        }
        if dry_run:
            return result

        client.query(
            _VOID_REVIEW_DIMENSION_WRITE,
            review_id=review_id,
            score=plan["score"],
            tier=plan["tier"],
            voided_dimensions_json=json.dumps(plan["void_records"]),
        )

        sn_id = row.get("standard_name_id")
        if sn_id:
            from imas_codex.standard_names.provenance_lifecycle import (
                record_standard_name_change,
            )

            result["change_event_id"] = record_standard_name_change(
                client,
                sn_id,
                sn_id,
                operation="void_review_dimension",
                reason=(
                    f"{dimension} voided on review {review_id} by {actor}: {reason}"
                ),
                origin="review_dimension_void",
            )
            if refresh_aggregates:
                result["aggregates_refreshed"] = bool(update_review_aggregates([sn_id]))
    finally:
        if own:
            client.close()

    logger.info(
        "void_review_dimension: %s voided on %s by %s — score %s to %.6f (%s)",
        dimension,
        review_id,
        actor,
        result.get("score_before"),
        result["score"],
        reason,
    )
    return result


@functools.lru_cache(maxsize=1)
def _winning_review_resolution_methods(
    schema_path: Path | None = None,
) -> frozenset[str]:
    """Return the review methods the LinkML schema marks as winning.

    The enum description is deliberately the admission authority: newly added
    enum values remain excluded until the schema explicitly names them as
    eligible for winning-group slot mirroring.
    """
    import yaml

    if schema_path is None:
        schema_path = Path(__file__).parents[1] / "schemas" / "standard_name.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    enum = schema["enums"]["ReviewResolutionMethod"]
    permissible = frozenset(enum["permissible_values"])
    eligibility_text, marker, _ = str(enum.get("description") or "").partition(
        "are eligible as winning groups"
    )
    if not marker:
        raise RuntimeError(
            "ReviewResolutionMethod schema does not declare winning-group eligibility"
        )
    referenced = frozenset(re.findall(r"``([a-z][a-z0-9_]*)``", eligibility_text))
    winning = permissible & referenced
    if not winning:
        raise RuntimeError(
            "ReviewResolutionMethod schema declares no winning resolution methods"
        )
    return winning


@functools.lru_cache(maxsize=1)
def _non_winning_review_resolution_methods() -> frozenset[str]:
    """Return schema enum values not admitted as winning methods."""
    import yaml

    schema_path = Path(__file__).parents[1] / "schemas" / "standard_name.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    permissible = frozenset(
        schema["enums"]["ReviewResolutionMethod"]["permissible_values"]
    )
    return permissible - _winning_review_resolution_methods()


def docs_review_eligibility_params() -> dict[str, list[str]]:
    """Return schema-derived parameters for docs-review eligibility."""
    winning = _winning_review_resolution_methods()
    non_winning = _non_winning_review_resolution_methods()
    return {
        "docs_review_winning_methods": sorted(winning),
        "docs_review_non_winning_methods": sorted(non_winning),
    }


def _docs_review_winner_query_body(name_variable: str = "sn") -> str:
    """Return the single graph traversal that selects a winning docs group."""
    if name_variable != "sn":
        raise ValueError("docs-review eligibility requires the StandardName alias 'sn'")
    return """
        MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
        WHERE review.review_axis = 'docs'
          AND review.review_group_id IS NOT NULL
          AND NOT EXISTS {
              MATCH (sn)-[:HAS_REVIEW]->(non_winner:StandardNameReview)
              WHERE non_winner.review_axis = 'docs'
                AND non_winner.resolution_method
                    IN $docs_review_non_winning_methods
          }
        WITH review.review_group_id AS review_group_id,
             max(CASE WHEN coalesce(review.is_canonical, false)
                 THEN 1 ELSE 0 END) AS canonical_group,
             max(review.reviewed_at) AS group_reviewed_at,
             [item IN collect({
                 review_id: review.id,
                 reviewed_at: review.reviewed_at,
                 resolution_method: review.resolution_method,
                 score: review.score
             }) WHERE item.resolution_method IN $docs_review_winning_methods]
                 AS winning_records
        WHERE size(winning_records) > 0
        UNWIND winning_records AS winning_record
        WITH review_group_id, canonical_group, group_reviewed_at, winning_record
        ORDER BY canonical_group DESC, group_reviewed_at DESC,
                 review_group_id DESC, winning_record.reviewed_at DESC,
                 winning_record.review_id DESC
        WITH collect({
            review_group_id: review_group_id,
            review_id: winning_record.review_id,
            resolution_method: winning_record.resolution_method,
            score: winning_record.score
        })[0] AS winner
        RETURN winner
    """


def docs_review_eligibility_where() -> str:
    """Return the shared predicate for a reachable winning docs review."""
    return f"EXISTS {{{_docs_review_winner_query_body()}}}"


def property_coverage(
    *,
    label: str,
    properties: tuple[str, ...],
    schema: GraphSchema,
    gc: Any | None = None,
) -> dict[str, int]:
    """Return candidate and property counts, failing closed on zero coverage."""
    declared = schema.get_all_slots(label)
    for property_name in properties:
        if property_name not in declared:
            raise ValueError(f"Unknown property '{property_name}' on {label}")
    if not properties:
        raise ValueError("At least one property is required for coverage")

    projections = ["count(node) AS candidates"]
    projections.extend(
        f"count(node.{property_name}) AS {property_name}_covered"
        for property_name in properties
    )
    query = f"MATCH (node:{label})\nRETURN " + ",\n       ".join(projections)
    if gc is None:
        with GraphClient() as graph:
            rows = graph.query(query)
    else:
        rows = gc.query(query)

    keys = ["candidates", *(f"{name}_covered" for name in properties)]
    coverage = {key: int((dict(rows[0]).get(key) if rows else 0) or 0) for key in keys}
    candidates = coverage["candidates"]
    for property_name in properties:
        coverage_key = f"{property_name}_covered"
        if candidates > 0 and coverage[coverage_key] == 0:
            raise RuntimeError(
                f"Property coverage failed for {label}.{property_name}: "
                f"candidates={candidates}, {coverage_key}=0"
            )
    return coverage


def docs_review_property_coverage(gc: Any | None = None) -> dict[str, int]:
    """Assert that every review property used by eligibility has live coverage.

    Cypher silently returns zero for a missing property. The coverage row makes
    a misspelled or removed slot a hard failure before the eligibility result is
    trusted, and also verifies the ``name``/``docs`` axis vocabulary.
    """
    query = """
        MATCH (review:StandardNameReview)
        RETURN count(review) AS reviews,
               count(review.review_axis) AS axis_covered,
               count(review.review_group_id) AS group_covered,
               count(review.resolution_method) AS method_covered,
               count(CASE WHEN review.review_axis IN ['name', 'docs']
                     THEN 1 END) AS known_axis,
               count(CASE WHEN review.review_axis = 'docs'
                     THEN 1 END) AS docs_axis
        """
    if gc is None:
        with GraphClient() as graph:
            rows = graph.query(query)
    else:
        rows = gc.query(query)
    if not rows:
        return {
            "reviews": 0,
            "axis_covered": 0,
            "group_covered": 0,
            "method_covered": 0,
            "known_axis": 0,
            "docs_axis": 0,
        }
    coverage = {key: int(value or 0) for key, value in dict(rows[0]).items()}
    reviews = coverage["reviews"]
    missing = [
        key
        for key in ("axis_covered", "group_covered", "method_covered")
        if reviews and coverage[key] == 0
    ]
    if coverage["axis_covered"] != coverage["known_axis"]:
        missing.append("known_axis")
    if reviews and coverage["docs_axis"] == 0:
        missing.append("docs_axis")
    if missing:
        raise RuntimeError(
            "docs-review eligibility property coverage failed: "
            + ", ".join(sorted(set(missing)))
        )
    return coverage


def write_name_review_results(
    entries: list[dict[str, Any]],
    *,
    stats: dict[str, Any] | None = None,
) -> int:
    """Write name-axis review scores to StandardName nodes.

    Writes ``reviewer_score_name``, ``reviewed_name_at``, and all
    name-axis rubric fields. Does **not** touch any shared aggregate slots
    (those have been removed from the schema).

    The in-memory ``reviewer_score`` dict key is mapped to the graph
    property ``reviewer_score_name`` here — there is no generic
    ``reviewer_score`` graph property.

    Parameters
    ----------
    entries:
        Dicts with at least ``id`` and ``reviewer_score`` keys.
        The ``reviewer_score`` key is the in-memory generic name used
        by the pipeline; it maps to ``sn.reviewer_score_name`` on the
        graph.
    stats:
        Optional mutable dict to accumulate write counters.

    Returns
    -------
    int
        Number of StandardName nodes written.
    """
    if not entries:
        return 0
    if stats is None:
        stats = {}

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.updated_at = datetime(), sn.reviewer_score_name = b.reviewer_score_name,
                sn.reviewed_name_at = b.reviewed_name_at,
                sn.reviewer_scores_name = coalesce(b.reviewer_scores_name, sn.reviewer_scores_name),
                sn.reviewer_comments_name = coalesce(b.reviewer_comments_name, sn.reviewer_comments_name),
                sn.reviewer_comments_per_dim_name = coalesce(b.reviewer_comments_per_dim_name, sn.reviewer_comments_per_dim_name),
                sn.reviewer_model_name = coalesce(b.reviewer_model_name, sn.reviewer_model_name),
                sn.review_tier = coalesce(b.review_tier, sn.review_tier),
                sn.review_input_hash = b.review_input_hash,
                sn.reviewer_suggested_name = coalesce(nullIf(b.reviewer_suggested_name, ''), sn.reviewer_suggested_name),
                sn.reviewer_suggestion_justification_name = coalesce(nullIf(b.reviewer_suggestion_justification_name, ''), sn.reviewer_suggestion_justification_name),
                sn.llm_cost_review_name = coalesce(sn.llm_cost_review_name, 0.0) + coalesce(b.llm_cost_review_name, 0.0),
                sn.review_name_count = coalesce(sn.review_name_count, 0) + 1,
                sn.llm_cost = coalesce(sn.llm_cost, 0.0) + coalesce(b.llm_cost_review_name, 0.0)
            """,
            batch=[
                {
                    "id": e.get("_original_id") or e["id"],
                    "reviewer_score_name": e.get("reviewer_score"),
                    "reviewed_name_at": e.get("reviewed_at"),
                    "reviewer_scores_name": _ensure_json(e.get("reviewer_scores")),
                    "reviewer_comments_name": e.get("reviewer_comments"),
                    "reviewer_comments_per_dim_name": _ensure_json(
                        e.get("reviewer_comments_per_dim")
                    ),
                    "reviewer_model_name": e.get("reviewer_model"),
                    "review_tier": e.get("review_tier"),
                    "review_input_hash": e.get("review_input_hash"),
                    "reviewer_suggested_name": e.get("_suggested_name") or "",
                    "reviewer_suggestion_justification_name": e.get(
                        "_suggestion_justification"
                    )
                    or "",
                    "llm_cost_review_name": e.get("llm_cost") or 0.0,
                }
                for e in entries
            ],
        )

    written = len(entries)
    logger.info("write_name_review_results: wrote %d names", written)
    return written


def write_docs_review_results(
    entries: list[dict[str, Any]],
    *,
    stats: dict[str, Any] | None = None,
) -> int:
    """Write docs-axis review scores to StandardName nodes.

    Gate: entries whose ``reviewed_name_at IS NULL`` in the graph are
    skipped (logged as ERROR, counted in
    ``stats['docs_skipped_missing_name']``).

    Does **not** touch any shared aggregate slots
    (those have been removed from the schema).

    The in-memory ``reviewer_score`` dict key is mapped to the graph
    property ``reviewer_score_docs`` here — there is no generic
    ``reviewer_score`` graph property.

    Parameters
    ----------
    entries:
        Dicts with at least ``id`` and ``reviewer_score`` keys.
        The ``reviewer_score`` key is the in-memory generic name used
        by the pipeline; it maps to ``sn.reviewer_score_docs`` on the
        graph.
    stats:
        Optional mutable dict to accumulate skip/write counters.

    Returns
    -------
    int
        Number of StandardName nodes written.
    """
    if not entries:
        return 0
    if stats is None:
        stats = {}

    # Gate check: query graph for reviewed_name_at on target names
    entry_ids = [e["id"] for e in entries if e.get("id")]
    if not entry_ids:
        return 0
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            RETURN sn.id AS id, sn.reviewed_name_at AS reviewed_name_at
            """,
            ids=entry_ids,
        )
        gated: dict[str, bool] = {}
        for r in rows or []:
            gated[r["id"]] = r.get("reviewed_name_at") is not None

    passed: list[dict[str, Any]] = []
    skipped = 0
    for e in entries:
        eid = e.get("id", "")
        if not gated.get(eid, False):
            logger.error(
                "write_docs_review_results: skipping %r — reviewed_name_at IS NULL",
                eid,
            )
            skipped += 1
            continue
        passed.append(e)
    stats["docs_skipped_missing_name"] = (
        stats.get("docs_skipped_missing_name", 0) + skipped
    )
    if not passed:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.updated_at = datetime(), sn.reviewer_score_docs = b.reviewer_score_docs,
                sn.reviewed_docs_at = b.reviewed_docs_at,
                sn.reviewer_scores_docs = coalesce(b.reviewer_scores_docs, sn.reviewer_scores_docs),
                sn.reviewer_comments_docs = coalesce(b.reviewer_comments_docs, sn.reviewer_comments_docs),
                sn.reviewer_comments_per_dim_docs = coalesce(b.reviewer_comments_per_dim_docs, sn.reviewer_comments_per_dim_docs),
                sn.reviewer_model_docs = coalesce(b.reviewer_model_docs, sn.reviewer_model_docs),
                sn.review_input_hash = b.review_input_hash,
                sn.llm_cost_review_docs = coalesce(sn.llm_cost_review_docs, 0.0) + coalesce(b.llm_cost_review_docs, 0.0),
                sn.review_docs_count = coalesce(sn.review_docs_count, 0) + 1,
                sn.llm_cost = coalesce(sn.llm_cost, 0.0) + coalesce(b.llm_cost_review_docs, 0.0)
            """,
            batch=[
                {
                    "id": e["id"],
                    "reviewer_score_docs": e.get("reviewer_score"),
                    "reviewed_docs_at": e.get("reviewed_at"),
                    "reviewer_scores_docs": _ensure_json(e.get("reviewer_scores")),
                    "reviewer_comments_docs": e.get("reviewer_comments"),
                    "reviewer_comments_per_dim_docs": _ensure_json(
                        e.get("reviewer_comments_per_dim")
                    ),
                    "reviewer_model_docs": e.get("reviewer_model"),
                    "review_input_hash": e.get("review_input_hash"),
                    "llm_cost_review_docs": e.get("llm_cost") or 0.0,
                }
                for e in passed
            ],
        )

    written = len(passed)
    logger.info("write_docs_review_results: wrote %d names", written)
    return written


def _resolve_grammar_token_version(gc: GraphClient, isn_version: str) -> str | None:
    """Find the best GrammarToken version for HAS_SEGMENT resolution.

    Prefers exact match with the ISN runtime version.  When no tokens
    exist for that version (e.g. ISN was upgraded but the grammar has
    not yet been re-synced — ``sn run`` auto-syncs at startup), falls
    back to the latest available version so that token-miss detection
    does not produce false-positive VocabGap nodes.

    Returns ``None`` when no GrammarToken nodes exist at all.
    """
    # Fast path: check exact version
    rows = list(
        gc.query(
            "MATCH (t:GrammarToken {version: $v}) RETURN t.version LIMIT 1",
            v=isn_version,
        )
        or []
    )
    if rows:
        return isn_version

    # Fallback: graph authority first, semantic package-version order second.
    from imas_codex.standard_names.grammar_query import select_grammar_version

    rows = list(
        gc.query(
            "MATCH (t:GrammarToken) "
            "WITH DISTINCT t.version AS version "
            "OPTIONAL MATCH (v:ISNGrammarVersion {version: version}) "
            "RETURN version, coalesce(v.active, false) AS active"
        )
        or []
    )
    fallback = select_grammar_version(rows)
    if fallback is not None:
        logger.warning(
            "No GrammarToken nodes for ISN %s — falling back to %s. "
            "`sn run` auto-syncs the grammar at startup; re-run to update.",
            isn_version,
            fallback,
        )
        return fallback

    return None


#: ISN grammar segments stored as bare-name columns on StandardName nodes.
#: Canonical render order places the single-token modifier segments
#: (aggregation, orbit, population) ahead of subject, mirroring the ISN
#: render order ``<aggregation>_<orbit>_<population>_<subject>_<base>``.
_GRAMMAR_SEGMENT_COLUMNS = (
    "physical_base",
    "aggregation",
    "orbit",
    "population",
    "subject",
    "state",
    "transformation",
    "component",
    "coordinate",
    "process",
    "position",
    "region",
    "device",
    "geometric_base",
    "object",
    "geometry",
)

#: Segment → typed-edge relationship type. Mirrors the GrammarToken edge
#: convention declared in ``standard_name.yaml`` (grammar_*_token slots).
#: Only segments that have a corresponding HAS_* typed edge appear here;
#: ``coordinate``/``object``/``geometry`` carry no typed edge today.
_SEGMENT_EDGE_TYPES: tuple[tuple[str, str], ...] = (
    ("physical_base", "HAS_PHYSICAL_BASE"),
    ("subject", "HAS_SUBJECT"),
    ("transformation", "HAS_TRANSFORMATION"),
    ("component", "HAS_COMPONENT"),
    ("coordinate", "HAS_COORDINATE"),
    ("process", "HAS_PROCESS"),
    ("position", "HAS_POSITION"),
    ("region", "HAS_REGION"),
    ("device", "HAS_DEVICE"),
    ("geometric_base", "HAS_GEOMETRIC_BASE"),
    ("aggregation", "HAS_AGGREGATION"),
    ("orbit", "HAS_ORBIT"),
    ("population", "HAS_POPULATION"),
)


def _coerce_segment_value(value: Any) -> str | None:
    """Coerce a parser segment value (str / Enum / tuple / None) to a graph-safe scalar.

    Multi-token segments (e.g. ``zone`` on ``lower_outer_squareness``) arrive
    from the ISN model as a tuple/list of tokens; an absent multi-token
    segment is an EMPTY tuple. Join the tokens with ``_`` (matching the
    surface form, ``lower_outer``) and map the empty case to ``None`` so an
    absent segment never persists as the literal string ``"()"``.
    """
    if value is None:
        return None
    if isinstance(value, tuple | list):
        parts = [p for p in (_coerce_segment_value(v) for v in value) if p]
        return "_".join(parts) if parts else None
    # Enum → its .value
    val = getattr(value, "value", value)
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _resolve_synced_segments(
    gc: GraphClient, token_version: str | None
) -> frozenset[str]:
    """Return grammar segments that have GrammarToken nodes in the graph.

    Segments with zero synced tokens (e.g. ``physical_base`` when the ISN
    graph spec excludes it) generate ``segment_edge_specs`` entries from
    the ISN parser but can never match a GrammarToken node. Filtering
    these out prevents false-positive token-miss warnings and VocabGap
    nodes.
    """
    if token_version is None:
        return frozenset()

    rows = list(
        gc.query(
            "MATCH (t:GrammarToken {version: $v}) RETURN DISTINCT t.segment AS seg",
            v=token_version,
        )
        or []
    )
    return frozenset(r["seg"] for r in rows)


def _write_grammar_decomposition(
    gc: GraphClient, name_ids: list[str]
) -> list[dict[str, str]]:
    """Write per-segment columns and typed grammar edges on StandardName nodes.

    Replaces ``_write_segment_edges``. Always populates
    compatibility per-segment columns (``sn.physical_base``, ``sn.subject``,
    …) from the strict ISN IR, regardless of whether a closed-vocabulary
    GrammarToken exists for the value. Conditionally writes typed edges
    only when a GrammarToken does exist.  The flat Pydantic projection is not a
    validity gate: ordered unary chains and binary trees remain valid and keep
    their IR-derived columns. On strict parser error all columns and edges are
    cleared (leaving the node unparseable). Such a name is quarantined at
    validate time
    (``validation_status='quarantined'``) — the single-pipeline gate — so
    no separate fallback flag is recorded.

    Idempotent: re-running on existing nodes overwrites columns and
    refreshes typed edges. Missing segments are written as ``null`` so
    re-writing a name whose grammar narrowed clears stale columns.

    Args:
        gc: Open GraphClient session.
        name_ids: List of StandardName.id values to process.

    Returns:
        List of token-miss gaps detected against the closed vocabulary,
        each a dict with keys ``sn_id``, ``segment``, ``token``.
    """
    all_gaps: list[dict[str, str]] = []

    try:
        from imas_standard_names import __version__ as isn_version
        from imas_standard_names.grammar import compose, parse
    except ImportError:
        logger.debug("ISN grammar not available — skipping grammar decomposition")
        return all_gaps

    # Token version is required for typed-edge writing only; column-write
    # path runs even when no GrammarToken nodes exist.
    token_version = _resolve_grammar_token_version(gc, isn_version)

    # Discover which segments actually have GrammarToken nodes in the graph.
    # Segments with 0 synced tokens (e.g. physical_base) generate edge specs
    # from ISN but can never match — exclude them from token-miss detection.
    synced_segments = _resolve_synced_segments(gc, token_version)

    # Build Cypher fragments once from the trusted module constants. Property
    # and relationship names come from a closed module-level tuple (never user
    # input), so direct interpolation is safe.
    edge_delete_types = "|".join(
        ["HAS_SEGMENT", *(rel for _seg, rel in _SEGMENT_EDGE_TYPES)]
    )
    columns_to_null = (
        ",\n                    ".join(
            f"sn.{col} = null" for col in _GRAMMAR_SEGMENT_COLUMNS
        )
        + ",\n                    sn.updated_at = datetime()"
    )
    columns_to_set = (
        ",\n                ".join(
            f"sn.{col} = ${col}" for col in _GRAMMAR_SEGMENT_COLUMNS
        )
        + ",\n                sn.updated_at = datetime()"
    )
    typed_edge_foreach = "\n                ".join(
        f"FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = '{seg}' "
        f"THEN [1] ELSE [] END |\n                    "
        f"MERGE (sn)-[:{rel}]->(t)\n                )"
        for seg, rel in _SEGMENT_EDGE_TYPES
    )

    for sn_id in name_ids:
        try:
            parsed = parse(sn_id, strict=True)
            if compose(parsed.ir) != sn_id:
                raise ValueError("strict grammar round-trip changed the name")
        except Exception as exc:
            logger.warning("Grammar parse failed for '%s': %s", sn_id, exc)
            # Clear columns and typed/segment edges for genuinely invalid names.
            gc.query(
                f"""
                MATCH (sn:StandardName {{id: $sn_id}})
                OPTIONAL MATCH (sn)-[r:{edge_delete_types}]->(:GrammarToken)
                DELETE r
                WITH sn
                SET {columns_to_null}
                """,
                sn_id=sn_id,
            )
            continue

        # ---- Always-on column write ------------------
        # Shared extraction authority — identical to the persist path
        # (_parse_grammar), so columns cannot diverge between the two.
        column_values = _segments_from_ir(parsed.ir)

        gc.query(
            f"""
            MATCH (sn:StandardName {{id: $sn_id}})
            SET {columns_to_set}
            """,
            sn_id=sn_id,
            **column_values,
        )

        # ---- Conditional typed-edge write (closed-vocab) ------------------
        if token_version is None:
            # No GrammarToken corpus — column write is enough; skip edges.
            continue

        # Idempotent: drop existing typed/segment edges before re-writing.
        gc.query(
            f"""
            MATCH (sn:StandardName {{id: $sn_id}})-[r:{edge_delete_types}]->(:GrammarToken)
            DELETE r
            """,
            sn_id=sn_id,
        )

        edges_param = [
            {
                "position": position,
                "segment": segment,
                "token": token,
            }
            for position, segment in enumerate(_public_grammar_segment_order())
            if segment in _GRAMMAR_SEGMENT_COLUMNS
            if (token := column_values.get(segment)) is not None
            and segment in synced_segments
        ]
        if not edges_param:
            continue

        results = list(
            gc.query(
                f"""
                MATCH (sn:StandardName {{id: $sn_id}})
                UNWIND $edges AS edge
                OPTIONAL MATCH (t:GrammarToken {{
                    value: edge.token,
                    segment: edge.segment,
                    version: $token_version
                }})
                WITH sn, edge, t
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (sn)-[r:HAS_SEGMENT]->(t)
                    SET r.position = edge.position,
                        r.segment = edge.segment
                )
                {typed_edge_foreach}
                RETURN edge.token AS token,
                       edge.segment AS segment,
                       t IS NOT NULL AS matched
                """,
                sn_id=sn_id,
                edges=edges_param,
                token_version=token_version,
            )
            or []
        )

        # Only report token misses for segments that have GrammarToken
        # nodes in the graph.  Segments like physical_base are intentionally
        # excluded from the graph sync and would always produce false
        # positives.
        missing = [
            f"{r['segment']}:{r['token']}"
            for r in results
            if not r.get("matched", True) and r.get("segment") in synced_segments
        ]
        if missing:
            logger.warning(
                "Token-miss for '%s': %s (ISN %s, tokens %s) — vocab gap",
                sn_id,
                ", ".join(missing),
                isn_version,
                token_version,
            )
            for r in results:
                if not r.get("matched", True) and r.get("segment") in synced_segments:
                    all_gaps.append(
                        {
                            "sn_id": sn_id,
                            "segment": r["segment"],
                            "token": r["token"],
                        }
                    )

    return all_gaps


# Deprecation alias — the old name is retained for one release. Callers
# inside the package have been migrated; external pipeline code may still
# import this symbol, so it is removed only in the release after that.
def _write_segment_edges(gc: GraphClient, name_ids: list[str]) -> list[dict[str, str]]:
    """Deprecated alias for :func:`_write_grammar_decomposition`."""
    import warnings

    warnings.warn(
        "_write_segment_edges is deprecated; use _write_grammar_decomposition.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _write_grammar_decomposition(gc, name_ids)


# =============================================================================
# Immediate-persist helpers — graph-state-machine generate_name
# =============================================================================

_GRAMMAR_FIELDS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
    "geometric_base",
    "object",
)

#: Units for which scalar quantities can safely default to ``one_like``
#: COCOS transformation type — these are sign-invariant under all COCOS
#: conventions.  Do not include ``Wb``, ``T``, ``A``, ``V.s``, ``T.m``
#: or any unit that may carry a COCOS-dependent sign.
SAFE_SCALAR_COCOS_UNITS: frozenset[str] = frozenset(
    {
        "1",
        "m",
        "m^2",
        "m^3",
        "eV",
        "Pa",
        "kg.m^-3",
        "s",
        "s^-1",
        "Hz",
        "m^-3",
        "m.s^-1",
        "A.m^-2",
    }
)


_RETIRE_FINALIZED_SOURCE_GAPS = """
                    WITH sns
                    OPTIONAL MATCH
                        (sns)-[source_gap:HAS_STANDARD_NAME_VOCAB_GAP]->(:VocabGap)
                    WITH sns, collect(DISTINCT source_gap) AS source_gap_edges
                    OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(dd:IMASNode)
                    OPTIONAL MATCH (sns)-[:FROM_SIGNAL]->(signal:FacilitySignal)
                    WITH sns, source_gap_edges,
                         collect(DISTINCT dd) + collect(DISTINCT signal)
                             AS backing_entities
                    FOREACH (edge IN source_gap_edges | DELETE edge)
                    WITH sns,
                         CASE
                           WHEN size(backing_entities) = 1
                            AND (
                              (sns.source_type = 'dd'
                               AND sns.id = 'dd:' + sns.source_id
                               AND backing_entities[0].id = sns.source_id
                               AND 'IMASNode' IN labels(backing_entities[0]))
                              OR
                              (sns.source_type = 'signals'
                               AND sns.id = 'signals:' + sns.source_id
                               AND backing_entities[0].id = sns.source_id
                               AND 'FacilitySignal' IN labels(backing_entities[0]))
                            )
                           THEN backing_entities ELSE []
                         END AS exact_backing
                    UNWIND CASE
                      WHEN exact_backing = [] THEN [null]
                      ELSE exact_backing
                    END AS source_entity
                    OPTIONAL MATCH
                        (source_entity)-[entity_gap:HAS_STANDARD_NAME_VOCAB_GAP]
                        ->(:VocabGap)
                    WITH sns, collect(DISTINCT entity_gap) AS entity_gap_edges
                    FOREACH (edge IN entity_gap_edges | DELETE edge)
                    RETURN sns.id AS id
"""


def persist_generated_name_winners(
    candidates: list[dict[str, Any]],
    *,
    compose_model: str,
    dd_version: str | None = None,
    cocos_version: int | None = None,
    run_id: str | None = None,
) -> list[str]:
    """Persist a single generate-name batch immediately to graph.

    Called from within ``compose_batch`` after LLM success.
    Enriches candidates with provenance metadata, embeds the standard-name
    string, and extracts grammar fields before writing.

    Embedding uses the standard-name string (``id``) — not the description,
    which is added later by the enrich pipeline.  If embedding fails for a
    candidate, ``embed_failed_at`` is set so the name can be retried later;
    the name is still written to the graph so it can advance through review.

    An exact token-and-sequence source CAS acquires the source write lock. The
    rich name writer, lifecycle transition, cluster backfill, and predecessor
    retirement then run through that transaction before the source claim is
    consumed. Any failure rolls every graph effect back together.

    Returns the finalized ``StandardNameSource`` identities.
    """
    from datetime import UTC

    if not candidates:
        return []

    now = datetime.now(UTC).isoformat()
    from imas_codex.standard_names.kind_derivation import derive_kind

    # Inject tool version provenance
    try:
        import imas_standard_names

        _isn_ver = imas_standard_names.__version__
    except (ImportError, AttributeError):
        _isn_ver = None
    try:
        import importlib.metadata

        _codex_ver = importlib.metadata.version("imas-codex")
    except Exception:
        _codex_ver = None

    for entry in candidates:
        entry.setdefault("model", compose_model)
        entry.setdefault("dd_version", dd_version)
        entry.setdefault("isn_version", _isn_ver)
        entry.setdefault("codex_version", _codex_ver)
        # validation_status is set upstream by audit logic (run_audits +
        # _is_quarantined) in the pool path or validate_worker in the
        # legacy linear path.  When neither has run (e.g. dry-run paths
        # that bypass audits, or callers that build candidates manually),
        # default to 'valid' here so the column is never NULL and
        # downstream filters keep working.  Embedding failures set
        # embed_failed_at for retry but do not change validation_status.
        entry.setdefault("validation_status", "valid")
        entry.setdefault("generated_at", now)
        # Strip private markers used only during in-batch attribution.
        entry.pop("_from_error_sibling", None)
        # Normalize name ID: expand physics abbreviations (e.g. ExB → e_cross_b)
        # and enforce lowercase before grammar validation.
        raw_name = entry.get("id") or ""
        if raw_name:
            entry["id"] = normalize_name_id(raw_name)
        # Derive kind from the name's own structure, overriding the LLM default.
        name = entry.get("id") or ""
        if name:
            entry["kind"] = derive_kind(name)

        # Default cocos_transformation_type to "one_like" for safe scalars
        # when the extractor / DD node did not already annotate one.
        if (
            not entry.get("cocos_transformation_type")
            and entry.get("kind") == "scalar"
            and (entry.get("unit") or "") in SAFE_SCALAR_COCOS_UNITS
        ):
            entry["cocos_transformation_type"] = "one_like"

    # --- Embedding deferred to shared discovery embed_description_worker ---
    # Clear embedding so the background embed worker picks these up.
    for entry in candidates:
        entry["embedding"] = None
        entry["embed_text_hash"] = None

    # Establish exact source identities before opening the transaction. Missing
    # fences are rejected by _complete_source_fences; stale fences do not bind
    # in the lock query below.
    write_batch: list[dict[str, Any]] = []
    candidate_by_sns: dict[str, dict[str, Any]] = {}
    for entry in candidates:
        if entry.get("model") == "deterministic:dd_error_modifier":
            continue
        raw_source_id = entry.get("source_id")
        source_types = entry.get("source_types") or []
        if not raw_source_id or not source_types:
            continue
        source_type = source_types[0]
        sns_id = f"{'dd' if source_type == 'dd' else 'signals'}:{raw_source_id}"
        write_batch.append(
            {
                "sns_id": sns_id,
                "source_id": raw_source_id,
                "source_type": source_type,
                "sn_id": entry["id"],
                "unit": entry.get("unit"),
                "model": compose_model,
                "claim_token": entry.get("source_claim_token"),
                "claim_seq": entry.get("source_claim_seq"),
            }
        )
        candidate_by_sns[sns_id] = entry

    write_batch = _complete_source_fences(write_batch)
    if not write_batch:
        return []

    finalized_ids: list[str] = []
    with GraphClient() as graph:
        with graph.session() as session:
            tx = session.begin_transaction()
            tx_gc = _TransactionQuery(tx)
            try:
                locked_rows = _lock_claimed_name_bindings(
                    tx_gc,
                    write_batch,
                    allow_missing=True,
                    allow_own_pending_reservation=True,
                    allow_dead_end_revival=True,
                )
                locked_ids = {
                    row["id"]
                    for row in locked_rows
                    if row.get("outcome", "winner") == "winner"
                }
                binding_kinds = {
                    row["id"]: (
                        row.get("binding_kind")
                        or (
                            "stable_reuse"
                            if row.get("target_stage") in _STABLE_BINDING_NAME_STAGES
                            else "owned_reservation"
                        )
                    )
                    for row in locked_rows
                    if row.get("outcome", "winner") == "winner"
                }
                locked_candidates = [
                    candidate_by_sns[sns_id]
                    for sns_id in candidate_by_sns
                    if sns_id in locked_ids
                ]
                if not locked_candidates:
                    tx.commit()
                    return []

                reserved_candidates = [
                    candidate_by_sns[sns_id]
                    for sns_id, kind in binding_kinds.items()
                    if kind == "owned_reservation"
                ]
                stable_candidates = [
                    candidate_by_sns[sns_id]
                    for sns_id, kind in binding_kinds.items()
                    if kind == "stable_reuse"
                ]
                written_name_ids = {entry["id"] for entry in stable_candidates}
                if reserved_candidates:
                    written_names = write_standard_names(
                        reserved_candidates,
                        gc=tx_gc,
                        return_written_ids=True,
                    )
                    if isinstance(written_names, int):
                        if written_names != len(reserved_candidates):
                            raise RuntimeError(
                                "rich standard-name write rejected an exact-claim "
                                "winner"
                            )
                        written_names = [entry["id"] for entry in reserved_candidates]
                    written_name_ids.update(written_names)
                finalize_batch = [
                    {**item, "binding_kind": binding_kinds[item["sns_id"]]}
                    for item in write_batch
                    if item["sns_id"] in locked_ids
                    and item["sn_id"] in written_name_ids
                ]
                if len(finalize_batch) != len(locked_candidates):
                    raise RuntimeError(
                        "rich standard-name write rejected an exact-claim winner"
                    )

                reserved_finalize_batch = [
                    item
                    for item in finalize_batch
                    if item["binding_kind"] == "owned_reservation"
                ]
                stable_finalize_batch = [
                    item
                    for item in finalize_batch
                    if item["binding_kind"] == "stable_reuse"
                ]
                finalized_rows: list[dict[str, Any]] = []
                if reserved_finalize_batch:
                    finalized_rows.extend(
                        tx_gc.query(
                            """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    MATCH (sns)-[reservation:PRODUCED_NAME]->(
                        sn:StandardName {id: b.sn_id})
                    WHERE sn.name_stage = $pending_stage
                      AND reservation.provisional = true
                      AND reservation.claim_token = b.claim_token
                      AND reservation.claim_seq = b.claim_seq
                    CALL (b) {
                      WITH b
                      WHERE b.source_type = 'dd'
                      MATCH (source:IMASNode {id: b.source_id})
                      RETURN source
                      UNION
                      WITH b
                      WHERE b.source_type = 'signals'
                      MATCH (source:FacilitySignal {id: b.source_id})
                      RETURN source
                    }
                    FOREACH (_ IN CASE
                      WHEN sns.compose_hint_status = 'open' THEN [1] ELSE [] END |
                      SET sns.compose_hint_status = 'consumed',
                          sns.compose_hint_consumed_at = datetime()
                    )
                    SET sn.name_stage = 'drafted',
                        sn.chain_length = coalesce(sn.chain_length, 0),
                        sn.docs_stage = coalesce(sn.docs_stage, 'pending'),
                        sn.docs_chain_length = coalesce(sn.docs_chain_length, 0),
                        sn.generated_at = coalesce(sn.generated_at, datetime()),
                        sn.model = coalesce(sn.model, b.model),
                        sn.run_id = coalesce(sns.run_id, sn.run_id),
                        sn.updated_at = datetime(),
                        sns.claim_token = null,
                        sns.claimed_at = null,
                        sns.status = 'composed',
                        sns.composed_at = datetime(),
                        sns.produced_sn_id = sn.id,
                        sns.last_error = null
                    REMOVE reservation.provisional,
                           reservation.claim_token,
                           reservation.claim_seq,
                           reservation.created_target
                    MERGE (source)-[:HAS_STANDARD_NAME]->(sn)
                    WITH sns, sn, b, b.source_type + ':' + b.source_id AS uri
                    SET sn.source_paths = CASE
                      WHEN uri IN coalesce(sn.source_paths, [])
                      THEN sn.source_paths
                      ELSE coalesce(sn.source_paths, []) + uri END,
                      sn.updated_at = datetime()
                    __RETIRE_FINALIZED_SOURCE_GAPS__
                    """.replace(
                                "__RETIRE_FINALIZED_SOURCE_GAPS__",
                                _RETIRE_FINALIZED_SOURCE_GAPS,
                            ),
                            batch=reserved_finalize_batch,
                            pending_stage=NameStage.pending.value,
                        )
                    )
                if stable_finalize_batch:
                    finalized_rows.extend(
                        tx_gc.query(
                            """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    MATCH (sn:StandardName {id: b.sn_id})
                    WHERE sn.name_stage IN $stable_stages
                      AND (b.unit IS NULL OR sn.unit IS NULL OR sn.unit = b.unit)
                    CALL (b) {
                      WITH b
                      WHERE b.source_type = 'dd'
                      MATCH (source:IMASNode {id: b.source_id})
                      RETURN source
                      UNION
                      WITH b
                      WHERE b.source_type = 'signals'
                      MATCH (source:FacilitySignal {id: b.source_id})
                      RETURN source
                    }
                    FOREACH (_ IN CASE
                      WHEN sns.compose_hint_status = 'open' THEN [1] ELSE [] END |
                      SET sns.compose_hint_status = 'consumed',
                          sns.compose_hint_consumed_at = datetime()
                    )
                    SET sns.claim_token = null,
                        sns.claimed_at = null,
                        sns.status = 'composed',
                        sns.composed_at = datetime(),
                        sns.produced_sn_id = sn.id,
                        sns.last_error = null
                    MERGE (sns)-[produced:PRODUCED_NAME]->(sn)
                    REMOVE produced.provisional,
                           produced.claim_token,
                           produced.claim_seq,
                           produced.created_target
                    MERGE (source)-[:HAS_STANDARD_NAME]->(sn)
                    WITH sns, sn, b, b.source_type + ':' + b.source_id AS uri
                    SET sn.source_paths = CASE
                      WHEN uri IN coalesce(sn.source_paths, [])
                      THEN sn.source_paths
                      ELSE coalesce(sn.source_paths, []) + uri END,
                      sn.updated_at = datetime()
                    __RETIRE_FINALIZED_SOURCE_GAPS__
                    """.replace(
                                "__RETIRE_FINALIZED_SOURCE_GAPS__",
                                _RETIRE_FINALIZED_SOURCE_GAPS,
                            ),
                            batch=stable_finalize_batch,
                            stable_stages=sorted(_STABLE_BINDING_NAME_STAGES),
                        )
                    )
                finalized_ids = [row["id"] for row in finalized_rows]
                if set(finalized_ids) != {item["sns_id"] for item in finalize_batch}:
                    raise RuntimeError(
                        "source claim changed during transactional name persistence"
                    )

                scoped_rows = tx_gc.query(
                    """
                    UNWIND $source_ids AS source_id
                    MATCH (source:StandardNameSource {id: source_id})
                    WHERE source.drain_scope_id IS NOT NULL
                    MATCH (source)-[:PRODUCED_NAME]->(name:StandardName)
                    OPTIONAL MATCH (name)-[:HAS_PARENT*1..]->(parent:StandardName)
                    WHERE parent.origin = 'derived'
                    WITH source, name,
                         [p IN collect(DISTINCT parent) WHERE p IS NOT NULL]
                           AS parents
                    WHERE (name.drain_scope_id IS NULL
                           OR name.drain_scope_id = source.drain_scope_id
                           OR name.drain_scope_claimed_at IS NULL
                           OR name.drain_scope_claimed_at < datetime()
                                - duration('PT600S'))
                      AND all(parent IN parents WHERE
                           parent.drain_scope_id IS NULL
                           OR parent.drain_scope_id = source.drain_scope_id
                           OR parent.drain_scope_claimed_at IS NULL
                           OR parent.drain_scope_claimed_at < datetime()
                                - duration('PT600S'))
                    SET name.updated_at = datetime(), name.drain_scope_id = source.drain_scope_id,
                        name.drain_scope_claimed_at =
                          source.drain_scope_claimed_at,
                        name.drain_scope_paths = source.drain_scope_paths
                    FOREACH (parent IN parents |
                      SET parent.updated_at = datetime(), parent.drain_scope_id = source.drain_scope_id,
                          parent.drain_scope_claimed_at =
                            source.drain_scope_claimed_at,
                          parent.drain_scope_paths = reduce(
                            paths = coalesce(parent.drain_scope_paths, []),
                            path IN source.drain_scope_paths |
                            CASE WHEN path IN paths THEN paths
                                 ELSE paths + path END))
                    RETURN source.id AS id
                    """,
                    source_ids=finalized_ids,
                )
                expected_scoped = {
                    row["id"]
                    for row in tx_gc.query(
                        """
                        UNWIND $source_ids AS source_id
                        MATCH (source:StandardNameSource {id: source_id})
                        WHERE source.drain_scope_id IS NOT NULL
                        RETURN source.id AS id
                        """,
                        source_ids=finalized_ids,
                    )
                }
                if {row["id"] for row in scoped_rows} != expected_scoped:
                    raise ManifestDrainConflict(
                        "generated name or derived parent has a competing drain lease"
                    )

                finalized_reserved_candidates = [
                    candidate_by_sns[sns_id]
                    for sns_id in finalized_ids
                    if binding_kinds[sns_id] == "owned_reservation"
                ]
                if finalized_reserved_candidates:
                    _backfill_cluster_from_sources(
                        finalized_reserved_candidates, gc=tx_gc, strict=True
                    )
                supersede_pairs = [
                    {"new_name": entry["id"], "source_id": entry["source_id"]}
                    for entry in finalized_reserved_candidates
                    if entry.get("id")
                    and entry.get("source_id")
                    and "dd" in (entry.get("source_types") or [])
                ]
                if supersede_pairs:
                    supersede_prior_source_names(supersede_pairs, gc=tx_gc)
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    if finalized_ids:
        bump_sn_run_counter(run_id, "names_composed", delta=len(finalized_ids))
    return finalized_ids


def persist_generated_name_batch(
    candidates: list[dict[str, Any]],
    *,
    compose_model: str,
    dd_version: str | None = None,
    cocos_version: int | None = None,
    run_id: str | None = None,
    return_winner_ids: bool = False,
) -> int | list[str]:
    """Persist a generate-name batch and report its finalized sources."""
    winner_ids = persist_generated_name_winners(
        candidates,
        compose_model=compose_model,
        dd_version=dd_version,
        cocos_version=cocos_version,
        run_id=run_id,
    )
    return winner_ids if return_winner_ids else len(winner_ids)


def _backfill_cluster_from_sources(
    candidates: list[dict[str, Any]],
    *,
    gc: Any | None = None,
    strict: bool = False,
) -> None:
    """Backfill ``primary_cluster_id`` and ``IN_CLUSTER`` edge from DD sources.

    For each candidate, query the graph for the best cluster via the
    HAS_STANDARD_NAME → IMASNode → IN_CLUSTER path, preferring
    IDS-scope > domain-scope > global-scope clusters.
    """
    sn_ids = [c["id"] for c in candidates if c.get("id")]
    if not sn_ids:
        return

    try:
        with nullcontext(gc) if gc is not None else GraphClient() as write_gc:
            results = write_gc.query(
                """
                UNWIND $sn_ids AS sid
                MATCH (sn:StandardName {id: sid})
                WHERE sn.primary_cluster_id IS NULL
                OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(imas:IMASNode)
                    -[:IN_CLUSTER]->(c:IMASSemanticCluster)
                WITH sn, c
                ORDER BY sn.id,
                         CASE c.scope
                             WHEN 'ids' THEN 0
                             WHEN 'domain' THEN 1
                             WHEN 'global' THEN 2
                             ELSE 3
                         END
                WITH sn, collect(c)[0] AS best_cluster
                WHERE best_cluster IS NOT NULL
                SET sn.updated_at = datetime(), sn.primary_cluster_id = best_cluster.id
                MERGE (sn)-[:IN_CLUSTER]->(best_cluster)
                RETURN sn.id AS sn_id, best_cluster.id AS cluster_id
                """,
                sn_ids=sn_ids,
            )
            if results:
                logger.debug(
                    "_backfill_cluster_from_sources: linked %d SNs to clusters",
                    len(results),
                )
    except Exception:
        if strict:
            raise
        logger.debug("_backfill_cluster_from_sources: failed", exc_info=True)


@retry_on_deadlock()
def _finalize_generated_name_stage(
    batch: list[dict[str, Any]],
) -> list[str]:
    """Set stage fields on new SNs and clear source claims — single transaction.

    Each item in *batch* must have ``sn_id`` (StandardName id), optionally
    ``sns_id`` (StandardNameSource id), and ``model``.

    In one transaction:
    - initialise missing/pending name lifecycle to ``drafted`` without
      demoting an existing reviewed, accepted, exhausted, or superseded name.
      A node carrying no description is parked at ``pending`` instead: name
      review requires prose, so drafting a prose-less name mints work nothing
      can claim — and the source is about to read ``composed``, which the
      compose pool excludes, so the pair would strand permanently.
      :func:`reconcile_reviewable_name_stage` re-opens such a pair.
    - fill missing docs/generation defaults without overwriting later state
    - Source: ``claim_token = null``, ``claimed_at = null``,
      ``status = 'composed'``, ``composed_at = datetime()``,
      ``produced_sn_id = sn.id``
    - Edge: ``(sns)-[:PRODUCED_NAME]->(sn)``
    """
    if not batch:
        return []

    winner_ids: list[str] = []
    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                rows = list(
                    tx.run(
                        """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE b.claim_token IS NOT NULL
                      AND b.claim_seq IS NOT NULL
                      AND sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.updated_at = datetime(), sn.name_stage = CASE
                            WHEN (sn.name_stage IS NULL
                                  OR sn.name_stage = 'pending')
                                 AND sn.description IS NOT NULL
                                 AND trim(sn.description) <> ''
                            THEN 'drafted'
                            WHEN sn.name_stage IS NULL THEN 'pending'
                            ELSE sn.name_stage END,
                        sn.chain_length = coalesce(sn.chain_length, 0),
                        sn.docs_stage = coalesce(sn.docs_stage, 'pending'),
                        sn.docs_chain_length = coalesce(sn.docs_chain_length, 0),
                        sn.generated_at = coalesce(sn.generated_at, datetime()),
                        sn.model = coalesce(sn.model, b.model)
                    WITH sn, b, sns
                    WHERE sns IS NOT NULL
                    SET sn.updated_at = datetime(), sns.claim_token  = null,
                        sns.claimed_at   = null,
                        sns.status       = 'composed',
                        sns.composed_at  = datetime(),
                        sns.produced_sn_id = sn.id,
                        sns.last_error   = null,
                        sn.run_id        = coalesce(sns.run_id, sn.run_id)
                    MERGE (sns)-[produced:PRODUCED_NAME]->(sn)
                    REMOVE produced.provisional,
                           produced.claim_token,
                           produced.claim_seq,
                           produced.created_target
                    __RETIRE_FINALIZED_SOURCE_GAPS__
                    """.replace(
                            "__RETIRE_FINALIZED_SOURCE_GAPS__",
                            _RETIRE_FINALIZED_SOURCE_GAPS,
                        ),
                        batch=batch,
                    )
                )
                winner_ids = [row["id"] for row in rows]
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    logger.debug(
        "_finalize_generated_name_stage: finalized %d SNs",
        len(winner_ids),
    )
    return winner_ids


def write_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.write_vocab_gaps."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.write_vocab_gaps(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_vocab_gap_adjudications(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.fetch_vocab_gap_adjudications."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.fetch_vocab_gap_adjudications(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Claim/mark/release — graph-state-machine workers
#
# Follows the battle-tested pattern from discovery/code/graph_ops.py:
#   1. claim: ORDER BY rand(), SET claimed_at + claim_token
#   2. verify: re-query by claim_token (prevents double-claim)
#   3. process (caller)
#   4. mark: SET result fields + clear claimed_at/claim_token (token-verified)
#   5. release (on error): clear claimed_at/claim_token (token-verified)
# =============================================================================

_CLAIM_TIMEOUT = import_module(
    ".graph_store.graph_catalog_reset", __package__
)._CLAIM_TIMEOUT


def claim_names_for_validation(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.claim_names_for_validation."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.claim_names_for_validation(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_names_validated(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_validated."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_validated(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_validation_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.release_validation_claims."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.release_validation_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_names_for_embedding(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.claim_names_for_embedding."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.claim_names_for_embedding(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_names_embedded(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_embedded."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_embedded(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_embedding_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.release_embedding_claims."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.release_embedding_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def get_validated_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.get_validated_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.get_validated_names(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_names_consolidated(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_consolidated."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_consolidated(
        *args, services=_graph_foundation.services, **kwargs
    )


def reset_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.reset_standard_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.reset_standard_names(
        *args, services=_graph_foundation.services, **kwargs
    )


def clear_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.clear_standard_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.clear_standard_names(
        *args, services=_graph_foundation.services, **kwargs
    )


def clear_sn_subsystem(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.clear_sn_subsystem."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.clear_sn_subsystem(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Link resolution
# =============================================================================

_MAX_LINK_RETRIES = import_module(
    ".graph_store.graph_catalog_reset", __package__
)._MAX_LINK_RETRIES


def claim_unresolved_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links.claim_unresolved_links."""
    from .graph_store import graph_doc_links

    return graph_doc_links.claim_unresolved_links(
        *args, services=_graph_foundation.services, **kwargs
    )


def resolve_links_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links.resolve_links_batch."""
    from .graph_store import graph_doc_links

    return graph_doc_links.resolve_links_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Documentation link resolution — post-drain cleanup
# =============================================================================


_NAME_LINK_RE = import_module(".graph_store.graph_doc_links", __package__)._NAME_LINK_RE


# When the LLM emits LaTeX with single backslashes inside its JSON output
# (e.g. ``$\theta$``, ``$\rho$``, ``\frac``), a lenient JSON decoder turns the
# ``\t``/``\r``/``\f``/``\b``/``\v`` prefix into the corresponding control
# character (TAB/CR/FF/BS/VT), corrupting the LaTeX (``$\rho$`` -> ``$<CR>ho$``).
# These control chars never appear legitimately in physics documentation, so map
# them back to the backslash form to reconstruct the command; any OTHER control
# char (NUL, ETX, …) is unrecoverable corruption and is stripped. Newline is kept.
_JSON_ESCAPE_CTRL = import_module(
    ".graph_store.graph_doc_links", __package__
)._JSON_ESCAPE_CTRL

_STRIP_CTRL_RE = import_module(
    ".graph_store.graph_doc_links", __package__
)._STRIP_CTRL_RE


def _sanitize_doc_text(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links._sanitize_doc_text."""
    from .graph_store import graph_doc_links

    return graph_doc_links._sanitize_doc_text(
        *args, services=_graph_foundation.services, **kwargs
    )


# This matcher parses an already-resolved Markdown link into both its visible
# label and its target so the accept gate can compare their meanings.
# ``find_name_references`` owns reference discovery instead; it intentionally
# returns only identity/span/syntax and does not expose a Markdown link's label.
_DOC_LINK_RE = import_module(".graph_store.graph_doc_links", __package__)._DOC_LINK_RE

_DOC_TEXT_UNSET = import_module(
    ".graph_store.graph_doc_links", __package__
)._DOC_TEXT_UNSET


def _doc_link_mismatches(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links._doc_link_mismatches."""
    from .graph_store import graph_doc_links

    return graph_doc_links._doc_link_mismatches(
        *args, services=_graph_foundation.services, **kwargs
    )


def _normalize_bare_doc_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links._normalize_bare_doc_links."""
    from .graph_store import graph_doc_links

    return graph_doc_links._normalize_bare_doc_links(
        *args, services=_graph_foundation.services, **kwargs
    )


def resolve_doc_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links.resolve_doc_links."""
    from .graph_store import graph_doc_links

    return graph_doc_links.resolve_doc_links(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Enrichment helpers — documentation iteration
# =============================================================================


def write_enrichment_results(
    results: list[dict[str, Any]],
    *,
    override: bool = False,
) -> int:
    """Write enrichment results back to graph.

    Only updates doc fields: description, documentation, links,
    validity_domain, constraints. Clears review_input_hash to invalidate
    stale reviews.

    Does NOT touch: id, kind, unit, model, grammar_parse_version, validation_diagnostics_json, etc.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection on catalog-edited names.

    Returns the number of nodes updated.
    """
    if not results:
        return 0

    # Pipeline protection
    from imas_codex.standard_names.protection import filter_protected

    results, skipped = filter_protected(results, override=override)
    if skipped:
        logger.warning(
            "write_enrichment_results: stripped protected fields from %d name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    if not results:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.updated_at = datetime(), sn.description = b.description,
                sn.documentation = b.documentation,
                sn.links = b.links,
                sn.validity_domain = b.validity_domain,
                sn.constraints = b.constraints,
                sn.link_status = b.link_status,
                sn.enriched_at = datetime(),
                sn.review_input_hash = null
            """,
            batch=[
                {
                    "id": r["id"],
                    "description": r.get("description") or "",
                    "documentation": r.get("documentation") or "",
                    "links": r.get("links") or None,
                    "validity_domain": r.get("validity_domain"),
                    "constraints": r.get("constraints") or None,
                    "link_status": _compute_link_status(r.get("links")),
                }
                for r in results
            ],
        )

    logger.info("Enriched %d StandardName nodes", len(results))
    return len(results)


# =============================================================================
# StandardNameSource CRUD
# =============================================================================

_VALID_PIPELINE_SOURCE_TYPES = {"dd", "signals"}
_MAX_SOURCE_COMPOSE_HINT_CHARS = 4000
_MAX_SOURCE_COMPOSE_HINT_REASON_CHARS = 2000


def _exact_dd_source_id(dd_path: str) -> tuple[str, str]:
    """Return canonical source id and bare path for one exact DD source."""
    raw = str(dd_path).strip()
    if raw.startswith("dd:"):
        raw = raw[3:]
    if (
        not raw
        or raw.startswith("signals:")
        or any(char in raw for char in "*?[]{}")
        or any(char.isspace() for char in raw)
    ):
        raise ValueError("source-hint requires one exact DD path without wildcards")
    return f"dd:{raw}", raw


def _source_compose_hint_decision(
    gc: Any,
    *,
    source_id: str,
    replace: bool,
) -> dict[str, Any]:
    """Classify whether an exact DD source can accept compose steering."""
    rows = gc.query(
        """
        OPTIONAL MATCH (sns:StandardNameSource {id: $source_id})
        WITH sns, CASE WHEN sns IS NULL THEN false ELSE EXISTS {
          MATCH (sns)-[:PRODUCED_NAME]->(live:StandardName)
          WHERE NOT (coalesce(live.name_stage, '') IN $terminal_stages)
        } END AS has_live_binding
        RETURN sns.id AS id,
               sns.source_id AS source_path,
               sns.source_type AS source_type,
               sns.status AS source_status,
               sns.claimed_at AS claimed_at,
               coalesce(sns.attempt_count, 0) AS attempt_count,
               sns.compose_hint_status AS compose_hint_status,
               CASE
                 WHEN sns IS NULL THEN 'missing_source'
                 WHEN sns.source_type <> 'dd' THEN 'not_dd_source'
                 WHEN sns.claimed_at IS NOT NULL THEN 'active_claim'
                 WHEN sns.status <> 'extracted' THEN 'source_not_extracted'
                 WHEN coalesce(sns.attempt_count, 0) >= $attempt_cap
                   THEN 'attempt_cap_reached'
                 WHEN has_live_binding THEN 'live_name_binding'
                 WHEN sns.compose_hint_status = 'open' AND NOT $replace
                   THEN 'open_hint_exists'
                 ELSE 'eligible'
               END AS decision
        """,
        source_id=source_id,
        replace=replace,
        attempt_cap=_MAX_COMPOSE_CLAIM_ATTEMPTS,
        terminal_stages=sorted(_TERMINAL_BINDING_NAME_STAGES),
    )
    return dict(rows[0]) if rows else {"decision": "missing_source", "id": None}


@retry_on_deadlock()
def set_source_compose_hint(
    dd_path: str,
    *,
    hint: str,
    reason: str,
    replace: bool = False,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Open durable composition steering for one exact DD source.

    Eligibility and the write enforce the same claim, lifecycle, attempt, and
    live-binding predicates. A claim turnover between classification and write
    therefore refuses the update rather than steering an in-flight worker.
    """
    source_id, source_path = _exact_dd_source_id(dd_path)
    hint = str(hint).strip()
    reason = str(reason).strip()
    if not hint:
        raise ValueError("source-hint requires a non-empty hint")
    if not reason:
        raise ValueError("source-hint requires a non-empty reason")
    if len(hint) > _MAX_SOURCE_COMPOSE_HINT_CHARS:
        raise ValueError(
            f"source hint exceeds {_MAX_SOURCE_COMPOSE_HINT_CHARS} characters"
        )
    if len(reason) > _MAX_SOURCE_COMPOSE_HINT_REASON_CHARS:
        raise ValueError(
            "source hint reason exceeds "
            f"{_MAX_SOURCE_COMPOSE_HINT_REASON_CHARS} characters"
        )

    own = gc is None
    client = GraphClient() if own else gc
    try:
        current = _source_compose_hint_decision(
            client,
            source_id=source_id,
            replace=replace,
        )
        result = {
            **current,
            "source_id": source_id,
            "source_path": source_path,
            "eligible": current.get("decision") == "eligible",
            "applied": False,
            "replaced": False,
            "dry_run": dry_run,
        }
        if current.get("decision") != "eligible":
            return result
        replacing_open = current.get("compose_hint_status") == "open"
        if dry_run:
            result["decision"] = "would_replace" if replacing_open else "would_set"
            result["replaced"] = replacing_open
            return result

        rows = client.query(
            """
            MATCH (sns:StandardNameSource {id: $source_id})
            WHERE sns.source_type = 'dd'
              AND sns.status = 'extracted'
              AND sns.claimed_at IS NULL
              AND coalesce(sns.attempt_count, 0) < $attempt_cap
              AND NOT EXISTS {
                MATCH (sns)-[:PRODUCED_NAME]->(live:StandardName)
                WHERE NOT (coalesce(live.name_stage, '') IN $terminal_stages)
              }
              AND ($replace OR coalesce(sns.compose_hint_status, '') <> 'open')
            WITH sns, sns.compose_hint_status = 'open' AS replaced
            SET sns.compose_hint = $hint,
                sns.compose_hint_reason = $reason,
                sns.compose_hint_status = 'open',
                sns.compose_hint_requested_at = datetime(),
                sns.compose_hint_consumed_at = null
            RETURN sns.id AS id, sns.source_id AS source_path, replaced
            """,
            source_id=source_id,
            hint=hint,
            reason=reason,
            replace=replace,
            attempt_cap=_MAX_COMPOSE_CLAIM_ATTEMPTS,
            terminal_stages=sorted(_TERMINAL_BINDING_NAME_STAGES),
        )
        if rows:
            result.update(
                applied=True,
                replaced=bool(rows[0].get("replaced")),
                decision="replaced" if rows[0].get("replaced") else "set",
            )
            return result

        latest = _source_compose_hint_decision(
            client,
            source_id=source_id,
            replace=replace,
        )
        latest_decision = latest.get("decision", "compare_and_set_lost")
        if latest_decision == "eligible":
            latest_decision = "compare_and_set_lost"
        result.update(
            latest,
            eligible=False,
            decision=latest_decision,
        )
        return result
    finally:
        if own:
            client.close()


def _pin_dd_source_snapshots(
    gc: GraphClient,
    sources: list[dict],
    *,
    default_dd_version: str | None = None,
) -> list[dict]:
    """Capture immutable DD semantics while the requested version is current.

    ``IMASNode`` is a mutable current-version projection, so it may only be
    sampled when its current ``DDVersion`` exactly matches the version supplied
    by extraction. Existing source nodes are never backfilled here: their
    original snapshot cannot be reconstructed safely after the fact.

    An already-pinned source is re-seedable without re-supplying its version.
    Re-seeding one (e.g. ``sn run --focus`` / ``--edits`` over a path whose
    source already exists) reuses the stored immutable snapshot: no version is
    required from the caller and ``IMASNode`` is not re-sampled — the ``MERGE
    ON MATCH`` clause preserves every ``dd_*`` field regardless. Only a
    genuinely new source must supply an exact ``dd_version`` to be snapshotted;
    inferring ``latest`` for a new source is still refused.

    A batch seeder that mixes re-seeds with the occasional genuinely-new source
    (a manifest mop-up over a corpus whose pins predate a later-added path)
    cannot know per-source which is which. It passes ``default_dd_version`` —
    the current version — which pins ONLY the genuinely-new sources; re-seeds
    ignore it and keep their stored pin. This is an explicit declaration, not
    ``latest`` inference, so the never-infer-latest contract still holds. A
    per-source ``dd_version`` that disagrees with a stored pin still raises.
    """
    prepared = [dict(source) for source in sources]
    dd_sources = [s for s in prepared if s.get("source_type") == "dd"]
    if not dd_sources:
        return prepared

    # Consult existing nodes FIRST — the pin state decides whether a source is
    # a re-seed (reuse stored snapshot) or a genuinely new one (must snapshot).
    existing = gc.query(
        """
        UNWIND $ids AS id
        OPTIONAL MATCH (source:StandardNameSource {id: id})
        RETURN id, source.id AS existing_id, source.dd_version AS dd_version,
               source.dd_documentation AS dd_documentation,
               source.dd_snapshot_pinned AS dd_snapshot_pinned
        """,
        ids=[source["id"] for source in dd_sources],
    )
    by_existing = {row["id"]: row for row in existing or []}

    requests: list[dict] = []  # only genuinely-new sources need snapshotting
    for source in dd_sources:
        row = by_existing.get(source["id"])
        already_pinned = bool(
            row and row.get("existing_id") and row.get("dd_snapshot_pinned")
        )
        if already_pinned:
            # Re-seed: reuse the stored pin. Validate an explicitly-supplied
            # version, but never require one, and never re-sample IMASNode —
            # the immutable snapshot already exists and ON MATCH preserves it.
            supplied = source.get("dd_version")
            stored = row.get("dd_version")
            if supplied and supplied != stored:
                raise ValueError(
                    f"existing DD source {source['id']!r} is pinned to "
                    f"{stored!r}, not {supplied!r}"
                )
            source["dd_version"] = stored
            continue
        if row and row.get("existing_id") and not row.get("dd_snapshot_pinned"):
            raise ValueError(
                f"existing DD source {source['id']!r} has no provable immutable "
                "snapshot; run the provenance backfill report"
            )
        version = source.get("dd_version") or default_dd_version
        if not version:
            raise ValueError(
                f"DD source {source.get('dd_path')!r} has no exact dd_version; "
                "source creation must never infer latest"
            )
        source["dd_version"] = version
        requests.append(
            {
                "id": source["id"],
                "path": source.get("dd_path"),
                "dd_version": version,
            }
        )

    if not requests:
        return prepared

    snapshots = gc.query(
        """
        UNWIND $requests AS request
        MATCH (version:DDVersion {id: request.dd_version, is_current: true})
        MATCH (node:IMASNode {id: request.path})
        OPTIONAL MATCH (node)-[:HAS_PARENT]->(parent:IMASNode)
        OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
        OPTIONAL MATCH (node)-[:HAS_COORDINATE]->(coordinate)
        RETURN request.id AS id, version.id AS dd_version,
               node.documentation AS dd_documentation,
               parent.id AS dd_parent_path,
               parent.documentation AS dd_parent_documentation,
               node.data_type AS dd_data_type,
               coalesce(node.unit, unit.id) AS dd_unit,
               collect(DISTINCT coordinate.id) AS dd_coordinates,
               node.lifecycle_status AS dd_lifecycle_status,
               node.lifecycle_version AS dd_lifecycle_version,
               node.description AS enhanced_description,
               node.enrichment_source AS enhancement_kind
        """,
        requests=requests,
    )
    by_id = {row["id"]: dict(row) for row in snapshots or []}
    missing = sorted(
        request["id"] for request in requests if request["id"] not in by_id
    )
    if missing:
        raise ValueError(
            "cannot capture exact DD snapshot for source(s): " + ", ".join(missing)
        )
    for source in prepared:
        if source.get("source_type") == "dd" and source["id"] in by_id:
            snapshot = by_id[source["id"]]
            from imas_codex.standard_names.dd_resolutions import resolve_dd_row

            context = resolve_dd_row(
                {
                    "path": source["dd_path"],
                    "unit": snapshot.get("dd_unit"),
                    "documentation": snapshot.get("dd_documentation"),
                    "data_type": snapshot.get("dd_data_type"),
                    "coordinates": snapshot.get("dd_coordinates") or (),
                    "lifecycle_status": snapshot.get("dd_lifecycle_status"),
                    "lifecycle_version": snapshot.get("dd_lifecycle_version"),
                },
                dd_version=snapshot["dd_version"],
            )
            effective = context.as_pipeline_item()
            snapshot.update(
                {
                    "dd_unit": effective["unit"],
                    "dd_documentation": effective["documentation"],
                    "dd_data_type": effective["data_type"],
                    "dd_coordinates": effective["coordinates"],
                    "dd_lifecycle_status": effective["lifecycle_status"],
                    "dd_lifecycle_version": effective["lifecycle_version"],
                    "raw_dd_context": effective["raw_dd_context"],
                    "published_dd_context": effective["published_dd_context"],
                    "dd_resolution_ids": effective["dd_resolution_ids"],
                    "dd_resolution_converged_ids": effective[
                        "dd_resolution_converged_ids"
                    ],
                    "dd_resolution_manifest_digest": effective[
                        "dd_resolution_manifest_digest"
                    ],
                    "_dd_resolution_marker": effective["_dd_resolution_marker"],
                }
            )
            source.update(snapshot)
    return prepared


def merge_standard_name_sources(
    sources: list[dict],
    *,
    force: bool = False,
    default_dd_version: str | None = None,
) -> int:
    """Batch MERGE StandardNameSource nodes.

    Each source dict must have: id, source_type, source_id, batch_key, status.
    DD sources additionally require ``dd_path`` and the exact ``dd_version``;
    their immutable semantic snapshot is captured centrally here. Optional:
    description and signal (for signal sources).

    ``default_dd_version`` pins genuinely-new DD sources that carry no per-source
    version (a mop-up seeder that mixes re-seeds with occasional new paths passes
    the current version here); re-seeds ignore it and keep their stored pin. See
    :func:`_pin_dd_source_snapshots`.

    On CREATE: sets all fields.
    On MATCH (existing node):
      - If force=True: resets to extracted, clears attempt_count/last_error/failed_at.
      - If status is 'stale': requeues to extracted.
      - Otherwise: preserves existing status (skip already-processed sources).

    Rejects source_type values not in {'dd', 'signals'} with ValueError.
    Returns count of nodes created or updated.
    """
    if not sources:
        return 0

    invalid = {s.get("source_type") for s in sources} - _VALID_PIPELINE_SOURCE_TYPES
    if invalid:
        raise ValueError(
            f"Invalid source_type(s) for pipeline: {invalid}. "
            f"Only {_VALID_PIPELINE_SOURCE_TYPES} are valid."
        )

    # Birth-invariant: a 'dd' source with no dd_path cannot have a FROM_DD_PATH
    # edge and would be an orphan from the moment it is written.  Drop these
    # before they reach the graph and log a WARNING so the caller knows.
    # Also stamp value-provenance (estimator) deterministically from the path:
    # measured/reconstructed/reference facets carry it as link metadata so the
    # estimator collapses to the base quantity's name (provenance-controlled-vocab).
    from imas_codex.standard_names.provenance import detect_value_provenance

    valid_sources: list[dict] = []
    for s in sources:
        if s.get("source_type") == "dd" and not s.get("dd_path"):
            logger.warning(
                "merge_standard_name_sources: skipping dd source %r — "
                "no dd_path supplied; writing would create an orphan "
                "(birth invariant violation).",
                s.get("id"),
            )
            continue
        if "provenance" not in s:
            _path = s.get("dd_path") or s.get("source_id") or ""
            s["provenance"] = detect_value_provenance(_path)[0]
        valid_sources.append(s)
    sources = valid_sources
    if not sources:
        return 0

    with GraphClient() as gc:
        sources = _pin_dd_source_snapshots(
            gc, sources, default_dd_version=default_dd_version
        )
        result = gc.query(
            """
            UNWIND $sources AS src
            MERGE (sns:StandardNameSource {id: src.id})
            ON CREATE SET
                sns.source_type = src.source_type,
                sns.source_id = src.source_id,
                sns.batch_key = src.batch_key,
                sns.status = src.status,
                sns.description = nullIf(src.description, ''),
                sns.physics_domain = src.physics_domain,
                sns.dd_version = src.dd_version,
                sns.provenance = src.provenance,
                sns.dd_documentation = src.dd_documentation,
                sns.dd_snapshot_pinned = true,
                sns.dd_parent_path = src.dd_parent_path,
                sns.dd_parent_documentation = src.dd_parent_documentation,
                sns.dd_data_type = src.dd_data_type,
                sns.dd_unit = src.dd_unit,
                sns.dd_coordinates = src.dd_coordinates,
                sns.dd_lifecycle_status = src.dd_lifecycle_status,
                sns.dd_lifecycle_version = src.dd_lifecycle_version,
                sns.enhanced_description = src.enhanced_description,
                sns.enhancement_kind = src.enhancement_kind,
                sns.attempt_count = 0
            ON MATCH SET
                sns.batch_key = src.batch_key,
                sns.description = coalesce(nullIf(src.description, ''), sns.description),
                sns.physics_domain = coalesce(src.physics_domain, sns.physics_domain),
                sns.provenance = coalesce(src.provenance, sns.provenance),
                sns.status = CASE
                    WHEN $force THEN 'extracted'
                    WHEN sns.status = 'stale' THEN 'extracted'
                    ELSE sns.status
                END,
                sns.attempt_count = CASE
                    WHEN $force THEN 0
                    ELSE sns.attempt_count
                END,
                sns.last_error = CASE
                    WHEN $force THEN null
                    ELSE sns.last_error
                END,
                sns.failed_at = CASE
                    WHEN $force THEN null
                    ELSE sns.failed_at
                END,
                sns.claimed_at = CASE
                    WHEN $force THEN null
                    ELSE sns.claimed_at
                END,
                sns.claim_token = CASE
                    WHEN $force THEN null
                    ELSE sns.claim_token
                END
            WITH sns, src
            // Create typed relationships to source entities
            FOREACH (_ IN CASE WHEN src.source_type = 'dd' AND src.dd_path IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (imas:IMASNode {id: src.dd_path})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
            )
            FOREACH (_ IN CASE WHEN src.source_type = 'signals' AND src.signal IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (sig:FacilitySignal {id: src.signal})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
            )
            RETURN count(sns) AS affected
            """,
            sources=sources,
            force=force,
        )
        return result[0]["affected"] if result else 0


@retry_on_deadlock()
def claim_standard_name_source_batch(
    batch_key: str,
    *,
    limit: int = 50,
    timeout_minutes: int = 30,
) -> tuple[str, list[dict]]:
    """Atomic full-batch claim of StandardNameSource nodes by batch_key.

    Claims up to ``limit`` extracted sources with matching batch_key.
    Uses two-step token verification to prevent double-claiming.
    Reclaims sources with stale claims (older than timeout_minutes).

    Returns (claim_token, claimed_sources) where each source dict has
    id, source_id, source_type, batch_key, description.
    """
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        # Step 1: Claim
        gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.batch_key = $batch_key
              AND (
                (sns.status = 'extracted' AND sns.claimed_at IS NULL)
                OR (sns.claimed_at IS NOT NULL
                    AND sns.claimed_at < datetime() - duration({minutes: $timeout}))
              )
            WITH sns ORDER BY rand() LIMIT $limit
            SET sns.claimed_at = datetime(),
                sns.claim_token = $token,
                sns.claim_seq = coalesce(sns.claim_seq, 0) + 1
            """,
            batch_key=batch_key,
            limit=limit,
            timeout=timeout_minutes,
            token=token,
        )
        # Step 2: Verify
        claimed = list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description,
                       sns.claim_token AS claim_token,
                       sns.claim_seq AS claim_seq
                """,
                token=token,
            )
        )
    return token, claimed


class SourceClaimResult(list[dict[str, Any]]):
    """Claim winners plus conflicts withheld before an attempt is charged."""

    def __init__(
        self,
        claimed: list[dict[str, Any]],
        *,
        withheld: list[dict[str, Any]],
    ) -> None:
        super().__init__(claimed)
        self.withheld = withheld


_EXPLICIT_SOURCE_CLAIM_QUERY = """
UNWIND $ids AS sns_id
MATCH (sns:StandardNameSource {id: sns_id})
OPTIONAL MATCH (sns)-[:PRODUCED_NAME]->(holder:StandardName)
WHERE NOT coalesce(holder.name_stage, '') IN $settled_stages
WITH sns,
     collect(DISTINCT holder {.id, stage: holder.name_stage}) AS holding
WHERE sns.status = 'extracted'
  AND (
    sns.claimed_at IS NULL
    OR sns.claimed_at < datetime() - duration({minutes: $timeout})
  )
FOREACH (_ IN CASE WHEN size(holding) = 0 THEN [1] ELSE [] END |
    SET sns.claimed_at = datetime(),
        sns.claim_token = $token,
        sns.claim_seq = coalesce(sns.claim_seq, 0) + 1,
        sns.attempt_count = coalesce(sns.attempt_count, 0) + 1
)
RETURN sns.id AS id,
       sns.source_id AS source_id,
       sns.source_type AS source_type,
       sns.batch_key AS batch_key,
       sns.description AS description,
       sns.status AS status,
       sns.claim_token AS claim_token,
       sns.claim_seq AS claim_seq,
       sns.attempt_count AS attempt_count,
       size(holding) = 0 AS claimed,
       holding
"""


@retry_on_deadlock()
def claim_explicit_standard_name_sources(
    source_ids: list[str],
    *,
    timeout_minutes: int = 30,
) -> SourceClaimResult:
    """Atomically claim the exact DD sources requested by a focused run.

    Every returned row carries the committed token and incremented sequence;
    missing, ineligible, or already-claimed sources are omitted. Sources bound
    to a non-settled identity are returned through ``result.withheld`` with the
    holding identity and stage, without consuming an attempt. Bindings only to
    superseded or exhausted identities remain claimable.
    """
    if not source_ids:
        return SourceClaimResult([], withheld=[])
    token = str(uuid.uuid4())
    sns_ids = [f"dd:{source_id}" for source_id in dict.fromkeys(source_ids)]
    with GraphClient() as gc:
        rows = gc.query(
            _EXPLICIT_SOURCE_CLAIM_QUERY,
            ids=sns_ids,
            token=token,
            timeout=timeout_minutes,
            settled_stages=sorted(_SETTLED_SOURCE_BINDING_STAGES),
        )
    claimed: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        is_claimed = bool(row.pop("claimed"))
        holding = row.pop("holding") or []
        if is_claimed:
            claimed.append(row)
            continue
        for holder in holding:
            withheld.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "holding_identity": holder["id"],
                    "holding_stage": holder["stage"],
                }
            )
    if withheld:
        logger.warning(
            "Withheld %d explicit source claim conflict(s): %s",
            len(withheld),
            ", ".join(
                f"{entry['id']} held by "
                f"{entry['holding_identity']}@{entry['holding_stage']}"
                for entry in withheld
            ),
        )
    return SourceClaimResult(claimed, withheld=withheld)


def _complete_source_fences(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only outcome items carrying a complete source ownership fence."""
    return [
        item
        for item in items
        if item.get("sns_id")
        and item.get("claim_token")
        and item.get("claim_seq") is not None
    ]


def _require_failure_reason(source_id: Any, reason: Any) -> str:
    """Return the trimmed failure reason, refusing a null or empty one.

    A source parked at ``failed`` with no ``last_error`` cannot be triaged:
    the reader cannot tell a transient fault from a permanent one, so the row
    is neither returned to the queue nor closed, and it accumulates silently.
    Every write path able to set a source status to ``failed`` therefore
    states its reason or refuses the write, in the same shape as the
    ``vocab_gap`` refusal.
    """
    text = str(reason or "").strip()
    if not text:
        raise ValueError(
            f"failed source outcome for {source_id!r} requires a non-empty reason"
        )
    return text


def _guard_failed_source_reasons(batch: list[dict[str, Any]]) -> None:
    """Refuse any outcome that parks a source at ``failed`` without a reason.

    Runs before the graph is opened, so a reasonless failure never reaches a
    transaction. The write itself coalesces ``last_error`` onto whatever the
    source already carried, which is exactly what makes an absent reason
    invisible at the write site.
    """
    for item in batch:
        if item.get("status") != "failed":
            continue
        item["last_error"] = _require_failure_reason(
            item.get("sns_id"), item.get("last_error")
        )


def _guard_existing_target_pairings(
    gc: Any, batch: list[dict[str, Any]], *, targets_may_be_missing: bool
) -> dict[str, str]:
    """Reject fresh pairings onto an EXISTING name that contradict its sources.

    Returns ``{sns_id: reason}`` for every pairing the attachment guard refuses.

    Scope is deliberate on both sides. It runs **before** the binding query,
    because that query stamps a provisional ``PRODUCED_NAME`` reservation and
    the guard reads exactly that edge to decide a source is already bound — a
    guard run afterwards would wave through every pairing it exists to judge.
    And it runs only where the target name already exists: a name minted by
    this very batch has no other sources to contradict and takes its unit from
    the candidate, so there is nothing to compare, while a source binding onto
    an established name is the case that admits a contradiction — two distinct
    vector fields of one device, or a unit of a different dimension. That
    binding previously reached the graph with no graph-aware check at all,
    because the compose-time gate sees only same-batch siblings and never the
    units.
    """
    from imas_codex.standard_names.attachment_audit import guard_source_pairings

    sources_by_name: dict[str, list[str]] = {}
    for item in batch:
        sn_id = item.get("sn_id")
        sns_id = item.get("sns_id")
        if sn_id and sns_id:
            sources_by_name.setdefault(sn_id, []).append(sns_id)
    if not sources_by_name:
        return {}

    if targets_may_be_missing:
        existing = {
            row["id"]
            for row in gc.query(
                """
                UNWIND $names AS name
                MATCH (sn:StandardName {id: name})
                RETURN sn.id AS id
                """,
                names=sorted(sources_by_name),
            )
        }
    else:
        # The caller binds only to targets that already exist, so asking the
        # graph again would cost a round trip to learn what it guarantees.
        existing = set(sources_by_name)
    refusals: dict[str, str] = {}
    for sn_id in sorted(existing):
        guarded = guard_source_pairings(gc, sn_id, sources_by_name[sn_id])
        for verdict in guarded.rejected:
            refusals[verdict.source_node_id] = verdict.reason
    return refusals


def _lock_claimed_name_bindings(
    gc: Any,
    batch: list[dict[str, Any]],
    *,
    allow_missing: bool,
    allow_own_pending_reservation: bool,
    allow_dead_end_revival: bool = False,
) -> list[dict[str, Any]]:
    """Lock source claims and classify their proposed name bindings.

    Stable lifecycle stages can be reused monotonically. A pending name is a
    private reservation: only the source whose exact claim created its
    token-and-sequence stamped ``PRODUCED_NAME`` edge may finish it. In compose
    mode a missing target is created and reserved here, while the source claim
    and target identity are locked by the same caller-owned transaction.
    Attachments require an existing stable target and never create a name.

    A binding is refused for three reasons, all reported the same way: the
    target's lifecycle stage forbids it, its unit disagrees, or the attachment
    guard finds the pairing inconsistent with the sources the name already
    carries (see :func:`_guard_existing_target_pairings`). A compose proposal
    has one governed lifecycle exception: an exact DD source may revive a
    superseded or exhausted spelling when no successor at any depth is live.
    The identity re-enters at ``drafted`` with pending validation and a linked
    ``StandardNameChange`` recording its prior stage and triggering source.
    Existing lineage edges remain intact. Attachments never receive that
    authority. Refusal is per pairing — one bad source never fails its batch —
    so this is the single point at which every claimed-source persister
    inherits the guard.

    A non-bindable target releases the exact source claim back to
    ``extracted`` without resetting ``attempt_count``. The retained attempt
    count remains subject to the normal claim cap, while ``last_error`` records
    the candidate identity, lifecycle state, and attempt number. Because this
    query runs through the caller's transaction, a later rich-write failure
    rolls the release back with every other graph effect.
    """
    attachment_refusals = _guard_existing_target_pairings(
        gc, batch, targets_may_be_missing=allow_missing
    )
    if attachment_refusals:
        batch = [
            {**item, "attachment_error": attachment_refusals.get(item["sns_id"])}
            for item in batch
        ]
    target_clause = (
        """
        MERGE (target:StandardName {id: b.sn_id})
        ON CREATE SET target.created_at = datetime(),
                      target.name_stage = $pending_stage,
                      target.status = 'draft',
                      target.reservation_source_id = sns.id,
                      target.reservation_claim_token = b.claim_token,
                      target.reservation_claim_seq = b.claim_seq,
                      target.updated_at = datetime()
        """
        if allow_missing
        else "OPTIONAL MATCH (target:StandardName {id: b.sn_id})"
    )
    binding_query = """
        UNWIND $batch AS b
        MATCH (sns:StandardNameSource {id: b.sns_id})
        WHERE sns.claim_token = b.claim_token
          AND sns.claim_seq = b.claim_seq
          AND sns.status = 'extracted'
        __TARGET_CLAUSE__
        WITH b, sns, target
        FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END |
          SET target.binding_lock_token = b.claim_token,
              target.binding_lock_seq = b.claim_seq
        )
        WITH b, sns, target
        CALL (b, sns, target) {
          WITH b, sns, target
          WHERE $allow_own_pending
            AND target.name_stage = $pending_stage
            AND target.reservation_source_id = sns.id
            AND target.reservation_claim_token = b.claim_token
            AND target.reservation_claim_seq = b.claim_seq
          MERGE (sns)-[reservation:PRODUCED_NAME]->(target)
          SET reservation.provisional = true,
              reservation.claim_token = b.claim_token,
              reservation.claim_seq = b.claim_seq,
              reservation.created_target = true
          REMOVE target.reservation_source_id,
                 target.reservation_claim_token,
                 target.reservation_claim_seq
          RETURN count(*) AS reservations
        }
        OPTIONAL MATCH (sns)-[owned:PRODUCED_NAME]->(target)
        WITH b, sns, target, owned,
             CASE
               WHEN $allow_dead_end_revival
                    AND b.source_type = 'dd'
                    AND target.name_stage IN $revivable_stages
                    AND b.attachment_error IS NULL
                    AND (b.unit IS NULL OR target.unit IS NULL
                         OR target.unit = b.unit)
                    AND EXISTS {
                      MATCH (sns)-[:FROM_DD_PATH]->(:IMASNode)
                    }
                    AND NOT EXISTS {
                      MATCH (target)-[:HAS_SUCCESSOR*1..]->(
                        successor:StandardName)
                      WHERE successor.name_stage IS NOT NULL
                        AND NOT (successor.name_stage IN $terminal_stages)
                    }
                    AND NOT EXISTS {
                      MATCH (successor:StandardName)-[:REFINED_FROM*1..]->(
                        target)
                      WHERE successor.name_stage IS NOT NULL
                        AND NOT (successor.name_stage IN $terminal_stages)
                    }
               THEN target.name_stage
               ELSE null
             END AS prior_stage
        FOREACH (_ IN CASE WHEN prior_stage IS NULL THEN [] ELSE [1] END |
          SET target.name_stage = $reviewable_stage,
              target.status = 'draft',
              target.validation_status = 'pending',
              target.validation_issues = null,
              target.validation_layer_summary = null,
              target.validated_at = null,
              target.reviewer_score_name = null,
              target.reviewer_scores_name = null,
              target.reviewer_comments_name = null,
              target.reviewer_comments_per_dim_name = null,
              target.reviewer_model_name = null,
              target.reviewed_name_at = null,
              target.review_quorum_shortfall = null,
              target.review_quorum_shortfall_at = null,
              target.review_resolution_method = null,
              target.refine_stop_reason = null,
              target.refine_stopped_at = null,
              target.refine_collision_name = null,
              target.claim_token = null,
              target.claimed_at = null,
              target.run_id = coalesce(sns.run_id, target.run_id)
        )
        FOREACH (_ IN CASE WHEN prior_stage IS NULL THEN [] ELSE [1] END |
          CREATE (change:StandardNameChange)
          SET change.id = 'sn-change:' + randomUUID(),
              change.from_name = prior_stage,
              change.to_name = $reviewable_stage,
              change.operation = 'revive_dead_end_identity',
              change.reason = 'source ' + sns.id +
                ' independently composed this retired identity while no live ' +
                'successor existed',
              change.origin = 'compose',
              change.run_id = sns.run_id,
              change.changed_at = datetime(),
              change.internal = true
          CREATE (target)-[:HAS_INTERNAL_CHANGE]->(change)
        )
        WITH b, sns, target, owned
        WITH b, sns, target, owned,
             CASE
               WHEN target.name_stage IN $stable_stages THEN true
               WHEN $allow_own_pending
                    AND target.name_stage = $pending_stage
                    AND owned.provisional = true
                    AND owned.claim_token = b.claim_token
                    AND owned.claim_seq = b.claim_seq
               THEN true
               ELSE false
             END AS lifecycle_bindable
        WITH b, sns, target, owned, lifecycle_bindable,
             CASE
               WHEN target IS NULL OR b.unit IS NULL
                    OR target.unit IS NULL OR target.unit = b.unit
               THEN true ELSE false
             END AS unit_bindable,
             b.attachment_error IS NULL AS attachment_bindable,
             CASE
               WHEN target IS NULL THEN '<missing>'
               ELSE coalesce(target.name_stage, '<unset>')
             END AS target_stage
        WITH b, sns, target, owned, lifecycle_bindable, unit_bindable,
             attachment_bindable, target_stage,
             lifecycle_bindable AND unit_bindable
               AND attachment_bindable AS bindable
        SET sns.claimed_at = datetime()
        WITH b, sns, target, owned, target_stage, bindable,
             lifecycle_bindable, unit_bindable, attachment_bindable
        FOREACH (_ IN CASE
          WHEN target.binding_lock_token = b.claim_token
           AND target.binding_lock_seq = b.claim_seq
          THEN [1] ELSE [] END |
          REMOVE target.binding_lock_token, target.binding_lock_seq
        )
        RETURN sns.id AS id,
               CASE
                 WHEN bindable THEN 'winner'
                 WHEN NOT lifecycle_bindable THEN 'lifecycle_collision'
                 WHEN NOT unit_bindable THEN 'unit_collision'
                 ELSE 'attachment_collision'
               END AS outcome,
               CASE
                 WHEN target.name_stage IN $stable_stages THEN 'stable_reuse'
                 WHEN owned.provisional = true
                      AND owned.claim_token = b.claim_token
                      AND owned.claim_seq = b.claim_seq
                 THEN 'owned_reservation'
                 ELSE 'collision'
               END AS binding_kind,
               b.sn_id AS candidate_id,
               target_stage,
               coalesce(sns.attempt_count, 0) AS attempt_count
        """.replace("__TARGET_CLAUSE__", target_clause)
    rows = gc.query(
        binding_query,
        batch=batch,
        allow_own_pending=allow_own_pending_reservation,
        allow_dead_end_revival=allow_dead_end_revival,
        stable_stages=sorted(_STABLE_BINDING_NAME_STAGES),
        revivable_stages=sorted(
            {
                NameStage.superseded.value,
                NameStage.exhausted.value,
            }
        ),
        terminal_stages=sorted(_TERMINAL_BINDING_NAME_STAGES),
        pending_stage=NameStage.pending.value,
        reviewable_stage=NameStage.drafted.value,
    )

    outcomes_by_source = {row["id"]: row.get("outcome", "winner") for row in rows}
    fenced_batch = [
        {
            **item,
            "preserve_current": outcomes_by_source[item["sns_id"]] == "winner",
        }
        for item in batch
        if item["sns_id"] in outcomes_by_source
    ]
    if fenced_batch:
        # A reclaimed source can carry abandoned provisional reservations even
        # when its new candidate wins. Only directly stamped relationships are
        # deletable. A winner preserves its current exact reservation; every
        # other provisional edge is stale. Projections and source-path mirrors
        # are never inferred to belong to a reservation because older writers
        # created those assertions without ownership metadata.
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sns:StandardNameSource {id: b.sns_id})
            WHERE sns.claim_token = b.claim_token
              AND sns.claim_seq = b.claim_seq
              AND sns.status = 'extracted'
            MATCH (sns)-[stale:PRODUCED_NAME]->(old:StandardName)
            WHERE stale.provisional = true
              AND stale.claim_token IS NOT NULL
              AND stale.claim_seq IS NOT NULL
              AND NOT (
                b.preserve_current = true
                AND old.id = b.sn_id
                AND stale.claim_token = b.claim_token
                AND stale.claim_seq = b.claim_seq
              )
            WITH old, collect(DISTINCT stale) AS stale_edges
            WITH old, stale_edges,
                 any(edge IN stale_edges
                     WHERE edge.created_target = true) AS owns_target
            FOREACH (stale_edge IN stale_edges |
              DELETE stale_edge
            )
            WITH DISTINCT old, owns_target
            WITH old,
                 owns_target
                   AND old.name_stage = $pending_stage
                   AND coalesce(old.source_paths, []) = []
                   AND NOT EXISTS { MATCH (old)--() }
                   AS delete_owned_orphan
            FOREACH (owned_target IN CASE
              WHEN delete_owned_orphan THEN [old] ELSE [] END |
              DELETE owned_target
            )
            """,
            batch=fenced_batch,
            pending_stage=NameStage.pending.value,
        )

    collision_ids = {
        row["id"] for row in rows if row.get("outcome", "winner") != "winner"
    }
    collision_batch = [item for item in batch if item["sns_id"] in collision_ids]
    if collision_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sns:StandardNameSource {id: b.sns_id})
            WHERE sns.claim_token = b.claim_token
              AND sns.claim_seq = b.claim_seq
              AND sns.status = 'extracted'
            OPTIONAL MATCH (target:StandardName {id: b.sn_id})
            SET sns.claimed_at = null,
                sns.claim_token = null,
                sns.status = 'extracted',
                sns.composed_at = null,
                sns.last_error = CASE
                  WHEN b.attachment_error IS NOT NULL
                  THEN 'standard-name attachment rejected: candidate "' +
                       b.sn_id + '" ' + b.attachment_error +
                       ' at composition attempt ' +
                       toString(coalesce(sns.attempt_count, 0))
                  WHEN target IS NULL OR NOT (target.name_stage IN $stable_stages)
                  THEN 'standard-name binding collision: candidate "' + b.sn_id +
                       '" has non-bindable name_stage "' +
                       coalesce(target.name_stage, '<missing>') +
                       '" at composition attempt ' +
                       toString(coalesce(sns.attempt_count, 0))
                  ELSE 'standard-name binding collision: candidate "' + b.sn_id +
                       '" unit "' + coalesce(b.unit, '<unset>') +
                       '" conflicts with existing unit "' +
                       coalesce(target.unit, '<unset>') +
                       '" at composition attempt ' +
                       toString(coalesce(sns.attempt_count, 0))
                END
            """,
            batch=collision_batch,
            stable_stages=sorted(_STABLE_BINDING_NAME_STAGES),
        )
    for row in rows:
        if row.get("outcome", "winner") == "winner":
            continue
        reason = attachment_refusals.get(row.get("id") or "")
        logger.warning(
            "Released source %s after %s for candidate %s at name stage %s "
            "(attempt %s retained)%s",
            row.get("id"),
            row.get("outcome"),
            row.get("candidate_id"),
            row.get("target_stage"),
            row.get("attempt_count"),
            f": {reason}" if reason else "",
        )
    return rows


@retry_on_deadlock()
def persist_claimed_source_outcomes(
    outcomes: list[dict[str, Any]],
) -> list[str]:
    """Consume exact source claims and persist terminal/retry diagnostics.

    The claim CAS and source mutation share one transaction. Items without a
    complete fence, and terminal failures without a reason, are rejected
    before opening the graph.
    """
    batch = _complete_source_fences(outcomes)
    if not batch:
        return []
    _guard_failed_source_reasons(batch)
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sns:StandardNameSource {id: b.sns_id})
            WHERE sns.claim_token = b.claim_token
              AND sns.claim_seq = b.claim_seq
              AND sns.status = 'extracted'
            SET sns.status = b.status,
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.last_error = coalesce(b.last_error, sns.last_error),
                sns.failed_at = CASE WHEN b.status = 'failed'
                                     THEN datetime() ELSE sns.failed_at END,
                sns.skip_reason = coalesce(b.skip_reason, sns.skip_reason),
                sns.skip_reason_detail = coalesce(
                    b.skip_reason_detail, sns.skip_reason_detail),
                sns.skipped_at = CASE WHEN b.status = 'skipped'
                                      THEN datetime() ELSE sns.skipped_at END
            RETURN sns.id AS id
            """,
            batch=batch,
        )
    return [row["id"] for row in rows]


@retry_on_deadlock()
def persist_claimed_vocab_gaps(
    gaps: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    source_type: str,
) -> list[str]:
    """Write gaps and consume their exact source claims in one transaction."""
    batch = _complete_source_fences(outcomes)
    if not batch:
        return []
    _guard_failed_source_reasons(batch)
    gaps_by_source: dict[str, list[dict[str, Any]]] = {}
    for gap in gaps:
        source_id = gap.get("source_id")
        if source_id:
            gaps_by_source.setdefault(source_id, []).append(gap)

    winner_ids: list[str] = []
    with GraphClient() as graph:
        with graph.session() as session:
            tx = session.begin_transaction()
            tx_gc = _TransactionQuery(tx)
            try:
                locked_rows = tx_gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    SET sns.claimed_at = datetime()
                    RETURN sns.id AS id, sns.source_id AS source_id
                    """,
                    batch=batch,
                )
                locked_ids = {row["id"] for row in locked_rows}
                locked_source_ids = {row["source_id"] for row in locked_rows}
                winner_outcomes = [
                    dict(item) for item in batch if item["sns_id"] in locked_ids
                ]
                actionable_source_ids = {
                    item["source_id"]
                    for item in winner_outcomes
                    if item.get("status") == "vocab_gap"
                }
                winner_gaps = [
                    gap
                    for source_id in actionable_source_ids
                    for gap in gaps_by_source.get(source_id, [])
                    if source_id in locked_source_ids
                ]
                for item in winner_outcomes:
                    if item.get("status") != "vocab_gap":
                        continue
                    source_id = item["source_id"]
                    source_gaps = gaps_by_source.get(source_id, [])
                    statements: list[str] = []
                    for gap in source_gaps:
                        reason = str(gap.get("reason") or "").strip()
                        if not reason:
                            raise ValueError(
                                "vocab_gap source outcome for "
                                f"{source_id!r} requires a non-empty reason"
                            )
                        segment = str(gap.get("segment") or "").strip()
                        token = str(gap.get("token") or "").strip()
                        if not segment or not token:
                            raise ValueError(
                                "vocab_gap source outcome for "
                                f"{source_id!r} requires a segment and token"
                            )
                        statement = (
                            f"missing {segment} vocabulary token {token!r}: {reason}"
                        )
                        if statement not in statements:
                            statements.append(statement)
                    if not statements:
                        raise ValueError(
                            "vocab_gap source outcome for "
                            f"{source_id!r} requires a non-empty reason"
                        )
                    item["last_error"] = "; ".join(statements)
                if winner_gaps:
                    written = write_vocab_gaps(
                        winner_gaps,
                        source_type,
                        gc=tx_gc,
                    )
                    if written == 0:
                        raise RuntimeError(
                            "actionable vocabulary gaps produced no graph records"
                        )

                rows = tx_gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    SET sns.status = b.status,
                        sns.claimed_at = null,
                        sns.claim_token = null,
                        sns.last_error = coalesce(b.last_error, sns.last_error),
                        sns.failed_at = CASE WHEN b.status = 'failed'
                                             THEN datetime() ELSE sns.failed_at END
                    RETURN sns.id AS id
                    """,
                    batch=winner_outcomes,
                )
                winner_ids = [row["id"] for row in rows]
                if set(winner_ids) != locked_ids:
                    raise RuntimeError(
                        "source claim changed during vocabulary-gap persistence"
                    )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    return winner_ids


@retry_on_deadlock()
def persist_claimed_attachments(
    attachments: list[dict[str, Any]],
) -> list[str]:
    """Attach exact source-claim winners to stable lifecycle targets.

    A terminal, transient, missing, or unowned-pending target releases the
    source through the same collision path as generated candidates. No source,
    DD projection, name scalar, or relationship is mutated unless both the
    exact source fence and stable target predicate survive to the final write.
    """
    batch = [
        {**item, "sn_id": item.get("standard_name")}
        for item in _complete_source_fences(attachments)
        if item.get("standard_name")
    ]
    if not batch:
        return []
    winner_ids: list[str] = []
    with GraphClient() as graph:
        with graph.session() as session:
            tx = session.begin_transaction()
            tx_gc = _TransactionQuery(tx)
            try:
                locked_rows = _lock_claimed_name_bindings(
                    tx_gc,
                    batch,
                    allow_missing=False,
                    allow_own_pending_reservation=False,
                    allow_dead_end_revival=False,
                )
                locked_ids = {
                    row["id"]
                    for row in locked_rows
                    if row.get("outcome", "winner") == "winner"
                }
                winner_batch = [item for item in batch if item["sns_id"] in locked_ids]
                if winner_batch:
                    rows = tx_gc.query(
                        """
                        UNWIND $batch AS b
                        MATCH (sns:StandardNameSource {id: b.sns_id})
                        WHERE sns.claim_token = b.claim_token
                          AND sns.claim_seq = b.claim_seq
                          AND sns.status = 'extracted'
                        MATCH (sn:StandardName {id: b.sn_id})
                        WHERE sn.name_stage IN $stable_stages
                        MATCH (src:IMASNode {id: b.source_id})
                        FOREACH (_ IN CASE
                          WHEN sns.compose_hint_status = 'open' THEN [1] ELSE [] END |
                          SET sns.compose_hint_status = 'consumed',
                              sns.compose_hint_consumed_at = datetime()
                        )
                        SET sns.status = 'attached',
                            sns.composed_at = datetime(),
                            sns.claimed_at = null,
                            sns.claim_token = null,
                            sns.produced_sn_id = sn.id,
                            sns.last_error = null
                        MERGE (sns)-[produced:PRODUCED_NAME]->(sn)
                        REMOVE produced.provisional,
                               produced.claim_token,
                               produced.claim_seq,
                               produced.created_target
                        MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                        WITH sns, sn, b, 'dd:' + b.source_id AS uri
                        SET sn.source_paths = CASE
                            WHEN uri IN coalesce(sn.source_paths, [])
                            THEN sn.source_paths
                            ELSE coalesce(sn.source_paths, []) + uri END,
                            sn.updated_at = datetime()
                        RETURN sns.id AS id
                        """,
                        batch=winner_batch,
                        stable_stages=sorted(_STABLE_BINDING_NAME_STAGES),
                    )
                    winner_ids = [row["id"] for row in rows]
                    if set(winner_ids) != locked_ids:
                        raise RuntimeError(
                            "source claim or target lifecycle changed during "
                            "transactional attachment persistence"
                        )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    return winner_ids


@retry_on_deadlock()
def stage_claimed_generated_candidates(
    candidates: list[dict[str, Any]],
) -> list[str]:
    """Reserve candidate source claims while staging names and source edges.

    A missing/stale fence produces no name, edge, stage, source, or counter
    mutation. The source stays claimed and ``extracted`` until the rich writer
    succeeds and :func:`_finalize_generated_name_stage` consumes the same fence.
    A rich-write crash therefore leaves a pending, automatically retryable
    reservation rather than a composed source stranded on an incomplete name.
    """
    batch = _complete_source_fences(candidates)
    if not batch:
        return []
    winner_ids: list[str] = []
    with GraphClient() as graph:
        with graph.session() as session:
            tx = session.begin_transaction()
            tx_gc = _TransactionQuery(tx)
            try:
                locked_rows = _lock_claimed_name_bindings(
                    tx_gc,
                    batch,
                    allow_missing=True,
                    allow_own_pending_reservation=True,
                    allow_dead_end_revival=True,
                )
                locked_ids = {
                    row["id"]
                    for row in locked_rows
                    if row.get("outcome", "winner") == "winner"
                }
                winner_ids = list(locked_ids)
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    return winner_ids


def fetch_claimed_source_metadata(token: str) -> list[dict]:
    """Fetch full metadata for claimed sources, joining source entities.

    For DD sources: joins IMASNode for documentation, unit, cluster info.
    For signal sources: joins FacilitySignal for description, unit, diagnostic.

    Returns list of dicts with source + joined metadata.
    """
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(imas:IMASNode)
                OPTIONAL MATCH (imas)-[:HAS_UNIT]->(u:Unit)
                OPTIONAL MATCH (imas)<-[:CONTAINS_PATH]-(c:SemanticCluster)
                OPTIONAL MATCH (sns)-[:FROM_SIGNAL]->(sig:FacilitySignal)
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description,
                       imas.id AS dd_path,
                       imas.documentation AS dd_documentation,
                       u.id AS unit,
                       c.label AS cluster_label,
                       c.scope AS cluster_scope,
                       sig.id AS signal_id,
                       sig.description AS signal_description
                """,
                token=token,
            )
        )


def mark_sources_composed(
    token: str,
    claim_seq: int,
    source_ids: list[str],
    standard_name_id: str,
) -> int:
    """Mark sources as composed and link to the produced StandardName.

    Fence-verified: only updates extracted sources matching the claim token and
    sequence.
    Creates PRODUCED_NAME relationship to the StandardName.
    Returns count of updated sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid})
            WHERE sns.claim_token = $token
              AND sns.claim_seq = $claim_seq
              AND sns.status = 'extracted'
            MATCH (sn:StandardName {id: $sn_id})
            SET sns.status = 'composed',
                sns.composed_at = datetime(),
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.produced_sn_id = sn.id
            MERGE (sns)-[:PRODUCED_NAME]->(sn)
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
            token=token,
            claim_seq=claim_seq,
            sn_id=standard_name_id,
        )
        return result[0]["affected"] if result else 0


def mark_sources_attached(
    token: str,
    claim_seq: int,
    source_ids: list[str],
    standard_name_id: str,
) -> int:
    """Mark sources as auto-attached to an existing StandardName.

    Used when a source matches an existing name without needing LLM composition.
    Exact-fence verified. Creates PRODUCED_NAME relationship.
    Returns count of updated sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid})
            WHERE sns.claim_token = $token
              AND sns.claim_seq = $claim_seq
              AND sns.status = 'extracted'
            MATCH (sn:StandardName {id: $sn_id})
            SET sns.status = 'attached',
                sns.composed_at = datetime(),
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.produced_sn_id = sn.id
            MERGE (sns)-[:PRODUCED_NAME]->(sn)
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
            token=token,
            claim_seq=claim_seq,
            sn_id=standard_name_id,
        )
        return result[0]["affected"] if result else 0


#: Reason recorded when a source spends its claim budget without a name. It
#: states the budget and nothing about cause, so a parked cohort carrying only
#: this string cannot be triaged from it.
_COMPOSE_CAP_REASON = "compose claim-attempt cap reached"

#: Closed set of dispositions a source parked at the compose claim-attempt cap
#: can carry. Every source at or past ``_MAX_COMPOSE_CLAIM_ATTEMPTS`` lands in
#: exactly one, so the cohort reads as a triage list rather than as a uniform
#: failure and each class states the next operation rather than the symptom.
CAP_PARKED_DISPOSITIONS = frozenset(
    {
        # A name already exists, so the cap bounds further claims rather than
        # the outcome and no composition is owed.
        "name_produced",
        # The data-dictionary quantity behind the source is gone, so no name can
        # ever be composed for it and the parked state is terminal.
        "upstream_quantity_removed",
        # The path names a coordinate or geometry component rather than a
        # physical quantity, so composition is not the applicable operation.
        "compose_not_applicable",
        # The recorded reason names a vocabulary gap: the concept is nameable
        # only once the grammar covers it, which no retry can produce.
        "vocabulary_gap",
        # A nameable quantity with a spent budget, a recorded cause and no
        # structural blocker: a candidate for revival once compose changes.
        "attempt_budget_exhausted",
        # The row records nothing about its own failure, so it cannot be triaged
        # until a reason exists. That absence is the finding, not a classification
        # in its own right: the class counts rows whose cause was lost, and the
        # only repair is to make the write path record one.
        "cause_not_recorded",
    }
)


def classify_parked_source(evidence: dict[str, Any]) -> str:
    """Return the triage disposition for one source parked at the cap.

    ``evidence`` carries what the parked row records about itself: ``produced``
    (the number of live produced names), the backing data-dictionary node's
    ``lifecycle_status`` and ``node_category``, and ``last_error``.

    Total by construction: every input returns a member of
    :data:`CAP_PARKED_DISPOSITIONS`, including the row that records nothing, so
    a census of the cohort has no residue to interpret and the answer cannot be
    silently wrong through an unhandled shape.

    A recorded cause outranks the node category, and the category outranks
    silence: it exists to classify the rows whose cause was lost.
    """
    if int(evidence.get("produced") or 0) > 0:
        return "name_produced"
    if str(evidence.get("lifecycle_status") or "").strip() == "removed":
        return "upstream_quantity_removed"
    reason = str(evidence.get("last_error") or "").strip()
    if reason and reason != _COMPOSE_CAP_REASON:
        if "vocabulary gap" in reason.lower():
            return "vocabulary_gap"
        return "attempt_budget_exhausted"
    if str(evidence.get("node_category") or "").strip() in {"geometry", "coordinate"}:
        return "compose_not_applicable"
    return "cause_not_recorded"


def mark_sources_failed(
    token: str,
    source_ids: list[str],
    error: str,
    *,
    max_attempts: int = 3,
) -> int:
    """Mark sources as failed with durable retry.

    Increments attempt_count. If below max_attempts, returns to 'extracted'
    for retry. At max_attempts, transitions to terminal 'failed' status.
    Token-verified. Returns count of updated sources.

    ``error`` is the reason the terminal state will carry, so a null or empty
    one is refused before the graph is opened rather than parking an
    untriageable source.
    """
    error = _require_failure_reason(source_ids, error)
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid, claim_token: $token})
            SET sns.attempt_count = coalesce(sns.attempt_count, 0) + 1,
                sns.last_error = $error,
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.status = CASE
                    WHEN coalesce(sns.attempt_count, 0) + 1 >= $max_attempts
                    THEN 'failed'
                    ELSE 'extracted'
                END,
                sns.failed_at = CASE
                    WHEN coalesce(sns.attempt_count, 0) + 1 >= $max_attempts
                    THEN datetime()
                    ELSE sns.failed_at
                END
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
            token=token,
            error=error,
            max_attempts=max_attempts,
        )
        return result[0]["affected"] if result else 0


_CAP_PARKED_QUERY = """
MATCH (sns:StandardNameSource)
WHERE coalesce(sns.attempt_count, 0) >= $cap
OPTIONAL MATCH (node:IMASNode {id: sns.source_id})
RETURN sns.id AS id, sns.last_error AS last_error,
       node.lifecycle_status AS lifecycle_status,
       size([(sns)-[:PRODUCED_NAME]->(:StandardName) | 1]) AS produced,
       node.node_category AS node_category
ORDER BY sns.id
"""


def record_parked_dispositions(
    rows: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Classify parked rows and write the disposition onto each source.

    The single classify-and-write path, shared by the reconcile pass that reads
    the whole cohort and by the writer that parks a source, so the precedence in
    :func:`classify_parked_source` exists once. ``rows`` carry the evidence
    aliases returned by the parking statement. A classification outside
    :data:`CAP_PARKED_DISPOSITIONS` refuses before any write, so no caller can
    end with a row carrying an unclassified disposition.
    """
    own = gc is None
    client = GraphClient() if own else gc
    try:
        written: list[dict[str, str]] = []
        for row in rows:
            disposition = classify_parked_source(row)
            if disposition not in CAP_PARKED_DISPOSITIONS:
                raise ValueError(f"row {row['id']!r} has no triage disposition")
            written.append({"id": row["id"], "disposition": disposition})
        if not dry_run and written:
            client.query(_CAP_DISPOSITION_WRITE, rows=written)
        counts = Counter(row["disposition"] for row in written)
        return {
            "parked": len(written),
            "written": 0 if dry_run else len(written),
            "counts": dict(sorted(counts.items())),
        }
    finally:
        if own:
            client.close()


def disposition_parked_sources(
    *,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Give every source parked at the compose cap a triage disposition.

    Reads the parked cohort with the evidence each row carries and delegates to
    :func:`record_parked_dispositions`. This is the reconcile net: the parking
    writer records a disposition as it parks, and this pass heals any row that
    predates that write path or was parked around it.
    """
    own = gc is None
    client = GraphClient() if own else gc
    try:
        rows = [
            dict(r)
            for r in client.query(_CAP_PARKED_QUERY, cap=_MAX_COMPOSE_CLAIM_ATTEMPTS)
        ]
        return record_parked_dispositions(rows, dry_run=dry_run, gc=client)
    finally:
        if own:
            client.close()


#: The rows the parking writer has already parked under the cap reason and left
#: without a disposition. Scoped by the reason the writer itself writes, so the
#: read can only see rows a parking event produced, and by the absence of the
#: disposition, so it is empty once the writer has recorded one.
_CAP_UNDISPOSED_QUERY = """
MATCH (sns:StandardNameSource)
WHERE sns.status = 'failed'
  AND sns.last_error = $cap_reason
  AND sns.parked_disposition IS NULL
OPTIONAL MATCH (node:IMASNode {id: sns.source_id})
RETURN sns.id AS id, sns.last_error AS last_error,
       node.lifecycle_status AS lifecycle_status,
       size([(sns)-[:PRODUCED_NAME]->(:StandardName) | 1]) AS produced,
       node.node_category AS node_category
ORDER BY sns.id
"""


def record_dispositions_for_undisposed_parks(
    *,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Stamp a disposition on the rows a parking event left undisposed.

    The parking writer calls this in the same tick as the write that parks, so
    a source arrives at ``failed`` already carrying a member of
    :data:`CAP_PARKED_DISPOSITIONS` and no later classification pass is needed
    for it. Rows already carrying a disposition are not in scope, so a tick that
    parks nothing and a tick that parks an already-disposed row both do nothing.
    """
    own = gc is None
    client = GraphClient() if own else gc
    try:
        rows = [
            dict(r)
            for r in client.query(_CAP_UNDISPOSED_QUERY, cap_reason=_COMPOSE_CAP_REASON)
        ]
        return record_parked_dispositions(rows, dry_run=dry_run, gc=client)
    finally:
        if own:
            client.close()


_CAP_DISPOSITION_WRITE = """
UNWIND $rows AS row
MATCH (sns:StandardNameSource {id: row.id})
SET sns.parked_disposition = row.disposition
RETURN count(sns) AS written
"""


_CAP_DISPOSITION_READBACK = """
MATCH (sns:StandardNameSource)
WHERE coalesce(sns.attempt_count, 0) >= $cap
RETURN sns.id AS id, sns.parked_disposition AS disposition
ORDER BY sns.id
"""


def census_parked_dispositions(*, gc: Any | None = None) -> dict[str, Any]:
    """Re-read the parked cohort and report the disposition it now carries.

    Reads the stored property back rather than the classification this process
    just computed, so the census tests the write instead of re-running the
    classifier that produced the values. It returns the count per class, the
    rows carrying no disposition and the rows carrying one outside
    :data:`CAP_PARKED_DISPOSITIONS`; a census reporting zero unclassified is
    then a statement about the graph and not about the classifier.
    """
    own = gc is None
    client = GraphClient() if own else gc
    try:
        rows = [
            dict(r)
            for r in client.query(
                _CAP_DISPOSITION_READBACK, cap=_MAX_COMPOSE_CLAIM_ATTEMPTS
            )
        ]
    finally:
        if own:
            client.close()
    counts = Counter(v for v in (r["disposition"] for r in rows) if v)
    unclassified = sorted(r["id"] for r in rows if not r["disposition"])
    unexpected = sorted(
        r["id"]
        for r in rows
        if r["disposition"] and r["disposition"] not in CAP_PARKED_DISPOSITIONS
    )
    return {
        "parked": len(rows),
        "counts": dict(sorted(counts.items())),
        "unclassified": unclassified,
        "unexpected": unexpected,
    }


def retry_failed_sources(
    source_paths: list[str],
    *,
    reason: str,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Return terminal or attempt-capped sources to the composition queue.

    This is the sanctioned recovery operation for two otherwise-unclaimable
    source states:

    * ``status='failed'`` after the composition retry budget was spent;
    * ``status='extracted'`` with ``attempt_count`` already at the claim cap.

    Before resetting the source, the operation creates one
    ``StandardNameSourceRetry`` audit event containing the prior status,
    attempt count, error, and operator reason. The source stores both the
    event-id scalar mirror and the ``HAS_RETRY_EVENT`` relationship. The write
    uses the read status and attempt count as a compare-and-set guard, so a
    concurrent worker cannot be rewound after it has already advanced.

    ``source_paths`` accepts either full ``StandardNameSource.id`` values
    (``dd:<path>`` / ``signals:<id>``) or their bare ``source_id`` values.
    Returns requested/eligible/retried/refused counts plus the eligible source
    ids and created event ids.
    """
    requested = sorted(
        {str(path).strip() for path in source_paths if str(path).strip()}
    )
    if not requested:
        raise ValueError("at least one source path is required")
    reason = reason.strip()
    if not reason:
        raise ValueError("a non-empty retry reason is required")

    own = gc is None
    client = GraphClient() if own else gc
    try:
        candidates = list(
            client.query(
                """
                MATCH (sns:StandardNameSource)
                WHERE sns.id IN $requested OR sns.source_id IN $requested
                WITH sns, coalesce(sns.attempt_count, 0) AS attempts
                WHERE sns.status = 'failed'
                   OR (sns.status = 'extracted' AND attempts >= $attempt_cap)
                RETURN sns.id AS id, sns.source_id AS source_path,
                       sns.status AS status,
                       attempts AS attempt_count, sns.last_error AS last_error
                ORDER BY sns.id
                """,
                requested=requested,
                attempt_cap=_MAX_COMPOSE_CLAIM_ATTEMPTS,
            )
        )
        eligible_ids = [row["id"] for row in candidates]
        matched_requests = {
            requested_path
            for requested_path in requested
            for row in candidates
            if requested_path
            in {
                row["id"],
                row.get("source_path") or row["id"].partition(":")[2],
            }
        }
        result: dict[str, Any] = {
            "requested": len(requested),
            "eligible": len(candidates),
            "retried": 0,
            "refused": len(requested) - len(matched_requests),
            "source_ids": eligible_ids,
            "event_ids": [],
            "dry_run": dry_run,
        }
        if dry_run or not candidates:
            return result

        items = [
            {
                "source_id": row["id"],
                "previous_status": row["status"],
                "previous_attempt_count": int(row["attempt_count"]),
                "previous_error": row.get("last_error"),
                "event_id": f"source-retry:{uuid.uuid4()}",
            }
            for row in candidates
        ]
        rows = list(
            client.query(
                """
                UNWIND $items AS item
                MATCH (sns:StandardNameSource {id: item.source_id})
                WHERE sns.status = item.previous_status
                  AND coalesce(sns.attempt_count, 0)
                      = item.previous_attempt_count
                  AND (
                    sns.status = 'failed'
                    OR (sns.status = 'extracted'
                        AND coalesce(sns.attempt_count, 0) >= $attempt_cap)
                  )
                CREATE (event:StandardNameSourceRetry {id: item.event_id})
                SET event.source_id = sns.id,
                    event.previous_status = item.previous_status,
                    event.previous_attempt_count = item.previous_attempt_count,
                    event.previous_error = item.previous_error,
                    event.reason = $reason,
                    event.retried_at = datetime()
                MERGE (sns)-[:HAS_RETRY_EVENT]->(event)
                SET sns.retry_events =
                        coalesce(sns.retry_events, []) + item.event_id,
                    sns.status = 'extracted',
                    sns.attempt_count = 0,
                    sns.claimed_at = null,
                    sns.claim_token = null,
                    sns.failed_at = null,
                    sns.last_error = null
                RETURN sns.id AS source_id, event.id AS event_id
                """,
                items=items,
                reason=reason,
                attempt_cap=_MAX_COMPOSE_CLAIM_ATTEMPTS,
            )
        )
        result["retried"] = len(rows)
        result["refused"] += len(candidates) - len(rows)
        result["source_ids"] = [row["source_id"] for row in rows]
        result["event_ids"] = [row["event_id"] for row in rows]
        return result
    finally:
        if own:
            client.close()


def retry_skipped_sources(
    source_paths: list[str],
    *,
    reason: str,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Return explicitly selected skipped sources to the composition queue.

    Eligibility is deliberately exact and fail-closed: the source must still
    be ``status='skipped'``, unclaimed, and have no scalar or relationship
    binding to a StandardName. Before resetting the source, the operation
    creates a ``StandardNameSourceRetry`` event. Its required
    ``previous_status='skipped'`` field is the schema-defined recovery-kind
    discriminator; the event also retains the prior attempt count, error, and
    operator reason.

    The compare-and-set write rechecks every eligibility predicate in the same
    transaction that records the event and changes the lifecycle state. It
    clears only exclusion/failure state needed for normal composition. Source
    identity, DD/signal relationships, and exact-source compose hints are not
    written and therefore remain intact.

    ``source_paths`` accepts either full ``StandardNameSource.id`` values or
    their bare ``source_id`` values. No patterns, siblings, or families are
    expanded.
    """
    requested = sorted(
        {str(path).strip() for path in source_paths if str(path).strip()}
    )
    if not requested:
        raise ValueError("at least one source path is required")
    reason = reason.strip()
    if not reason:
        raise ValueError("a non-empty retry reason is required")

    own = gc is None
    client = GraphClient() if own else gc
    try:
        candidates = list(
            client.query(
                """
                MATCH (sns:StandardNameSource)
                WHERE sns.id IN $requested OR sns.source_id IN $requested
                WITH sns, coalesce(sns.attempt_count, 0) AS attempts
                WHERE sns.status = 'skipped'
                  AND sns.claimed_at IS NULL
                  AND sns.claim_token IS NULL
                  AND sns.produced_sn_id IS NULL
                  AND NOT (sns)-[:PRODUCED_NAME]->(:StandardName)
                RETURN sns.id AS id, sns.source_id AS source_path,
                       sns.status AS status,
                       attempts AS attempt_count, sns.last_error AS last_error,
                       sns.skip_reason AS skip_reason,
                       sns.skip_reason_detail AS skip_reason_detail
                ORDER BY sns.id
                """,
                requested=requested,
            )
        )
        eligible_ids = [row["id"] for row in candidates]
        matched_requests = {
            requested_path
            for requested_path in requested
            for row in candidates
            if requested_path
            in {
                row["id"],
                row.get("source_path") or row["id"].partition(":")[2],
            }
        }
        result: dict[str, Any] = {
            "requested": len(requested),
            "eligible": len(candidates),
            "retried": 0,
            "refused": len(requested) - len(matched_requests),
            "source_ids": eligible_ids,
            "event_ids": [],
            "dry_run": dry_run,
        }
        if dry_run or not candidates:
            return result

        items = [
            {
                "source_id": row["id"],
                "previous_status": row["status"],
                "previous_attempt_count": int(row["attempt_count"]),
                "previous_error": _skipped_recovery_error(row),
                "previous_skip_reason": row.get("skip_reason"),
                "previous_skip_reason_detail": row.get("skip_reason_detail"),
                "event_id": f"source-retry:{uuid.uuid4()}",
            }
            for row in candidates
        ]
        rows = list(
            client.query(
                """
                UNWIND $items AS item
                MATCH (sns:StandardNameSource {id: item.source_id})
                WHERE sns.status = 'skipped'
                  AND item.previous_status = 'skipped'
                  AND coalesce(sns.attempt_count, 0)
                      = item.previous_attempt_count
                  AND coalesce(sns.skip_reason, '')
                      = coalesce(item.previous_skip_reason, '')
                  AND coalesce(sns.skip_reason_detail, '')
                      = coalesce(item.previous_skip_reason_detail, '')
                  AND sns.claimed_at IS NULL
                  AND sns.claim_token IS NULL
                  AND sns.produced_sn_id IS NULL
                  AND NOT (sns)-[:PRODUCED_NAME]->(:StandardName)
                CREATE (event:StandardNameSourceRetry {id: item.event_id})
                SET event.source_id = sns.id,
                    event.previous_status = item.previous_status,
                    event.previous_attempt_count = item.previous_attempt_count,
                    event.previous_error = item.previous_error,
                    event.reason = $reason,
                    event.retried_at = datetime()
                MERGE (sns)-[:HAS_RETRY_EVENT]->(event)
                SET sns.retry_events =
                        coalesce(sns.retry_events, []) + item.event_id,
                    sns.status = 'extracted',
                    sns.attempt_count = 0,
                    sns.claimed_at = null,
                    sns.claim_token = null,
                    sns.skip_reason = null,
                    sns.skip_reason_detail = null,
                    sns.skipped_at = null,
                    sns.failed_at = null,
                    sns.last_error = null
                RETURN sns.id AS source_id, event.id AS event_id
                """,
                items=items,
                reason=reason,
            )
        )
        result["retried"] = len(rows)
        result["refused"] += len(candidates) - len(rows)
        result["source_ids"] = [row["source_id"] for row in rows]
        result["event_ids"] = [row["event_id"] for row in rows]
        return result
    finally:
        if own:
            client.close()


def _skipped_recovery_error(source: dict[str, Any]) -> str | None:
    """Compose all durable explanations for a skipped recovery."""
    last_error = str(source.get("last_error") or "").strip()
    skip_reason = str(source.get("skip_reason") or "").strip()
    detail = str(source.get("skip_reason_detail") or "").strip()
    explanations = [last_error.rstrip(" ;")] if last_error.rstrip(" ;") else []
    if skip_reason:
        classification = f"skip_reason={skip_reason}"
        if detail:
            classification += f"; detail={detail}"
        explanations.append(classification)
    return "; ".join(explanations) or None


def mark_source_skipped(
    gc: GraphClient,
    source_id: str,
    *,
    reason: str,
    detail: str | None = None,
    source_type: str = "dd",
) -> int:
    """Mark a single StandardNameSource as skipped with audit trail.

    Used by the compose worker when a candidate cannot be promoted to a
    StandardName because the source DD path lacks a usable unit string
    (user invariant — the LLM never decides units, the DD source must
    provide one). The graph node id is derived as ``{source_type}:{source_id}``
    matching the convention in :func:`merge_standard_name_sources` and
    :func:`write_skipped_sources`.

    Args:
        gc: Open GraphClient. The caller is responsible for the connection
            lifecycle so this can be called from inside an existing
            ``with GraphClient() as gc:`` block.
        source_id: Bare DD path (or signal id). The full StandardNameSource
            id is derived as ``{source_type}:{source_id}``.
        reason: Machine-readable skip classification (e.g.
            ``'dd_unit_unresolvable'``).
        detail: Free-text detail (e.g. the raw DD unit string that
            triggered the skip).
        source_type: ``'dd'`` (default) or ``'signals'``.

    Returns:
        Number of StandardNameSource nodes updated (0 if no matching node
        exists; that is expected for sources that pre-date the
        StandardNameSource pipeline).
    """
    if not source_id:
        return 0
    sns_id = f"{source_type}:{source_id}"
    result = gc.query(
        """
        MATCH (sns:StandardNameSource {id: $sns_id})
        SET sns.status = 'skipped',
            sns.skip_reason = $reason,
            sns.skip_reason_detail = $detail,
            sns.claimed_at = null,
            sns.claim_token = null
        RETURN count(sns) AS affected
        """,
        sns_id=sns_id,
        reason=reason,
        detail=detail or "",
    )
    return result[0]["affected"] if result else 0


def mark_sources_stale(source_ids: list[str]) -> int:
    """Mark sources as stale (source entity no longer exists).

    Not token-verified — can be called from reconciliation outside claim context.
    Returns count of updated sources.
    """
    if not source_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid})
            SET sns.status = 'stale',
                sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
        )
        return result[0]["affected"] if result else 0


def write_skipped_sources(records: list[dict]) -> int:
    """Record DD paths that cannot be resolved to a standard name source.

    Each record must have: source_type, source_id, skip_reason,
    skip_reason_detail. Optional: description, dd_path (auto-derived from
    source_id for DD sources), signal (for signal sources), status
    (defaults to ``'skipped'`` for backward compatibility; may be
    ``'not_physical_quantity'`` for configuration metadata).

    The ``id`` is derived as ``{source_type}:{source_id}`` (matches the
    existing ``merge_standard_name_sources`` key convention).

    Idempotent — subsequent writes for the same id refresh skip_reason/
    skip_reason_detail but do not re-enqueue for composition.

    Returns count of nodes written.
    """
    if not records:
        return 0

    sources = []
    for r in records:
        source_type = r["source_type"]
        source_id = r["source_id"]
        sources.append(
            {
                "id": f"{source_type}:{source_id}",
                "source_type": source_type,
                "source_id": source_id,
                "skip_reason": r["skip_reason"],
                "skip_reason_detail": r.get("skip_reason_detail", ""),
                "description": r.get("description", ""),
                "status": r.get("status", "skipped"),
                "dd_path": r.get("dd_path")
                or (source_id if source_type == "dd" else None),
                "signal": r.get("signal")
                or (source_id if source_type == "signals" else None),
            }
        )

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sources AS src
            MERGE (sns:StandardNameSource {id: src.id})
            ON CREATE SET
                sns.source_type = src.source_type,
                sns.source_id = src.source_id,
                sns.status = src.status,
                sns.skip_reason = src.skip_reason,
                sns.skip_reason_detail = src.skip_reason_detail,
                sns.description = src.description,
                sns.attempt_count = 0
            ON MATCH SET
                sns.status = src.status,
                sns.skip_reason = src.skip_reason,
                sns.skip_reason_detail = src.skip_reason_detail,
                sns.description = coalesce(src.description, sns.description),
                sns.claimed_at = null,
                sns.claim_token = null
            WITH sns, src
            FOREACH (_ IN CASE WHEN src.source_type = 'dd' AND src.dd_path IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (imas:IMASNode {id: src.dd_path})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
            )
            FOREACH (_ IN CASE WHEN src.source_type = 'signals' AND src.signal IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (sig:FacilitySignal {id: src.signal})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
            )
            RETURN count(sns) AS affected
            """,
            sources=sources,
        )
        return result[0]["affected"] if result else 0


def list_skipped_sources(
    limit: int = 100,
    reason: str | None = None,
) -> list[dict]:
    """Query skipped/not_physical_quantity StandardNameSource records.

    Returns a list of dicts with keys: id, source_type, source_id,
    skip_reason, skip_reason_detail, description.

    Args:
        limit: Maximum rows to return.
        reason: Optional skip_reason filter.
    """
    where = "sns.status IN ['skipped', 'not_physical_quantity']"
    params: dict = {"limit": limit}
    if reason is not None:
        where += " AND sns.skip_reason = $reason"
        params["reason"] = reason

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource)
            WHERE {where}
            RETURN sns.id AS id,
                   sns.source_type AS source_type,
                   sns.source_id AS source_id,
                   sns.skip_reason AS skip_reason,
                   sns.skip_reason_detail AS skip_reason_detail,
                   sns.description AS description
            ORDER BY sns.skip_reason, sns.id
            LIMIT $limit
            """,
            **params,
        )
        return [dict(r) for r in result]


def get_skipped_source_counts(
    source_type: str | None = None,
) -> dict[str, int]:
    """Return counts of skipped StandardNameSource records grouped by skip_reason."""
    where = "sns.status = 'skipped'"
    params: dict = {}
    if source_type is not None:
        where += " AND sns.source_type = $source_type"
        params["source_type"] = source_type

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource)
            WHERE {where}
            RETURN coalesce(sns.skip_reason, 'unknown') AS skip_reason,
                   count(sns) AS cnt
            ORDER BY cnt DESC
            """,
            **params,
        )
        return {r["skip_reason"]: r["cnt"] for r in result}


def release_standard_name_source_claims(token: str) -> int:
    """Release all claims held by this token without changing status.

    Used for error recovery — release claims so other workers can pick them up.
    Returns count of released sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sns:StandardNameSource {claim_token: $token})
            SET sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS affected
            """,
            token=token,
        )
        return result[0]["affected"] if result else 0


# =============================================================================
# Polling-based work claiming — compose and review
# =============================================================================

# Settle window inserted before the post-claim winner re-read (see
# _verify_docs_claim_winners / _verify_name_claim_winners). Concurrent replicas
# that bind the same eligible node commit their claim SETs in a lock-serialised
# burst; the last committer's claim_seq wins the node. Re-reading immediately
# after our own commit can observe our claim_seq as current before a later
# racer commits, so every racer would momentarily look like the winner and each
# would fire a paid LLM call. Waiting out this short window lets the burst's
# final committer land, so exactly one replica survives the re-read. Kept small
# (batch LLM calls dwarf it) and overridable to 0 in tests.
_CLAIM_VERIFY_SETTLE_SECONDS = 0.5


# In-process tally of paid-call outcomes at the persist step, keyed by
# (run_id, pool). ``attempts`` counts every persist call that follows a paid
# LLM call; ``wasted`` counts those that no-oped because a concurrent replica
# already advanced the node past our claim (the claim-race residue). The run
# summary reads a snapshot to surface a wasted-paid-call ratio and warn when it
# exceeds the tripwire threshold. Populated by persist functions that can
# silently no-op (currently the docs axis — the names axis raises instead).
_persist_outcomes: dict[tuple[str, str], list[int]] = {}


def _record_persist_outcome(run_id: str | None, pool: str, *, persisted: bool) -> None:
    """Record one paid-call persist outcome for the (run_id, pool) tallies."""
    key = (run_id or "", pool)
    entry = _persist_outcomes.setdefault(key, [0, 0])
    entry[0] += 1
    if not persisted:
        entry[1] += 1


def persist_outcome_snapshot(run_id: str | None) -> dict[str, dict[str, int]]:
    """Return ``{pool: {"attempts": a, "wasted": w}}`` for *run_id*.

    ``wasted`` is the count of paid LLM calls whose persist no-oped because a
    concurrent replica had already claimed and advanced the node — the direct
    measure of claim-race waste. Empty when nothing was recorded for the run.
    """
    rid = run_id or ""
    return {
        pool: {"attempts": a, "wasted": w}
        for (r, pool), (a, w) in _persist_outcomes.items()
        if r == rid
    }


def reset_persist_outcomes(run_id: str | None = None) -> None:
    """Clear persist-outcome tallies (all runs, or just *run_id*)."""
    if run_id is None:
        _persist_outcomes.clear()
        return
    for key in [k for k in _persist_outcomes if k[0] == (run_id or "")]:
        del _persist_outcomes[key]


def claim_compose_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.claim_compose_sources."""
    from .graph_store import graph_audits

    return graph_audits.claim_compose_sources(*args, **kwargs)


def count_extracted_for_domain(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.count_extracted_for_domain."""
    from .graph_store import graph_audits

    return graph_audits.count_extracted_for_domain(*args, **kwargs)


def count_eligible_compose_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.count_eligible_compose_sources."""
    from .graph_store import graph_audits

    return graph_audits.count_eligible_compose_sources(*args, **kwargs)


def claim_review_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.claim_review_names."""
    from .graph_store import graph_audits

    return graph_audits.claim_review_names(*args, **kwargs)


def release_review_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.release_review_claims."""
    from .graph_store import graph_audits

    return graph_audits.release_review_claims(*args, **kwargs)


# Heartbeat interval: refresh a held claim well before the lease TTL so a
# long quorum-review / enrich batch cannot have its lease expire mid-flight
# and be re-claimed by a peer worker (which would duplicate the paid LLM
# spend). One third of the TTL leaves room for two refreshes before expiry
# even if one heartbeat is delayed.
_CLAIM_HEARTBEAT_SECONDS = import_module(
    ".graph_store.graph_audits", __package__
)._CLAIM_HEARTBEAT_SECONDS


def refresh_name_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.refresh_name_claims."""
    from .graph_store import graph_audits

    return graph_audits.refresh_name_claims(*args, **kwargs)


PROMOTE_STRANDED_NAME_WHERE = import_module(
    ".graph_store.graph_audits", __package__
).PROMOTE_STRANDED_NAME_WHERE

"""Canonical graph predicate for name-axis stranded promotion."""

_graph_audits = import_module(".graph_store.graph_audits", __package__)
_graph_audits.PROMOTE_STRANDED_DOCS_WHERE = (
    _graph_audits.PROMOTE_STRANDED_DOCS_WHERE.replace(
        "{docs_review_eligibility_where}", docs_review_eligibility_where()
    )
)
PROMOTE_STRANDED_DOCS_WHERE = _graph_audits.PROMOTE_STRANDED_DOCS_WHERE

"""Canonical graph predicate for docs-axis stranded promotion."""


def promote_stranded_reviewed(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.promote_stranded_reviewed."""
    from .graph_store import graph_audits

    return graph_audits.promote_stranded_reviewed(*args, **kwargs)


DOCS_AXIS_REDERIVE_NAME_WHERE = import_module(
    ".graph_store.graph_audits", __package__
).DOCS_AXIS_REDERIVE_NAME_WHERE

"""Scope for a docs-axis re-derivation: the name axis is already accepted."""

_DOCS_AXIS_MIRROR_AGREES = import_module(
    ".graph_store.graph_audits", __package__
)._DOCS_AXIS_MIRROR_AGREES

_DOCS_AXIS_MIRROR_DISAGREES = import_module(
    ".graph_store.graph_audits", __package__
)._DOCS_AXIS_MIRROR_DISAGREES


def _docs_axis_rederive_query(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._docs_axis_rederive_query."""
    from .graph_store import graph_audits

    return graph_audits._docs_axis_rederive_query(*args, **kwargs)


def reconcile_docs_axis_from_reviews(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_docs_axis_from_reviews."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_docs_axis_from_reviews(*args, **kwargs)


def _standard_name_source_upstream_present_cypher(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._standard_name_source_upstream_present_cypher."""
    from .graph_store import graph_audits

    return graph_audits._standard_name_source_upstream_present_cypher(*args, **kwargs)


def reconcile_standard_name_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_sources."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_sources(*args, **kwargs)


def fetch_manifest_source_release_rows(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.fetch_manifest_source_release_rows."""
    from .graph_store import graph_audits

    return graph_audits.fetch_manifest_source_release_rows(*args, **kwargs)


def reconcile_source_status_liveness(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_source_status_liveness."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_source_status_liveness(*args, **kwargs)


def retire_unreachable_hint_edits(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.retire_unreachable_hint_edits."""
    from .graph_store import graph_audits

    return graph_audits.retire_unreachable_hint_edits(*args, **kwargs)


def reconcile_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_vocab_gaps."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_vocab_gaps(*args, **kwargs)


def triage_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.triage_vocab_gaps."""
    from .graph_store import graph_audits

    return graph_audits.triage_vocab_gaps(*args, **kwargs)


# Skip classifications produced by the DD unit resolution leg. Everything else
# on ``skip_reason`` is a permanent eligibility exclusion (a temporal or local
# coordinate, configurable meaning, …) that no resolver change can lift, so the
# unit revival must never touch it.
_UNIT_SKIP_REASON_PREFIX = import_module(
    ".graph_store.graph_audits", __package__
)._UNIT_SKIP_REASON_PREFIX


def revive_unit_skipped_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.revive_unit_skipped_sources."""
    from .graph_store import graph_audits

    return graph_audits.revive_unit_skipped_sources(*args, **kwargs)


def _isn_grammar_context(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._isn_grammar_context."""
    from .graph_store import graph_audits

    return graph_audits._isn_grammar_context(*args, **kwargs)


def isn_vocabulary_signature(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.isn_vocabulary_signature."""
    from .graph_store import graph_audits

    return graph_audits.isn_vocabulary_signature(*args, **kwargs)


def retry_vocab_gap_sources_on_grammar_change(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.retry_vocab_gap_sources_on_grammar_change."""
    from .graph_store import graph_audits

    return graph_audits.retry_vocab_gap_sources_on_grammar_change(*args, **kwargs)


def reconcile_provenance(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_provenance."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_provenance(*args, **kwargs)


def reconcile_catalog_edit_origins(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_catalog_edit_origins."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_catalog_edit_origins(*args, **kwargs)


def reconcile_standard_name_cocos_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_cocos_links."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_cocos_links(*args, **kwargs)


def reconcile_standard_name_unit_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_unit_edges."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_unit_edges(*args, **kwargs)


def reconcile_constraint_fit_artifacts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_constraint_fit_artifacts."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_constraint_fit_artifacts(*args, **kwargs)


# Terminal / dead name stages — refined-away or given-up names whose edges have
# already been migrated off them; their denormalised scalar is left as-is.
_TERMINAL_NAME_STAGES = import_module(
    ".graph_store.graph_audits", __package__
)._TERMINAL_NAME_STAGES


def reconcile_standard_name_source_paths(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_source_paths."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_source_paths(*args, **kwargs)


def reconcile_catalog_status(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_catalog_status."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_catalog_status(*args, **kwargs)


def reconcile_reviewable_name_stage(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_reviewable_name_stage."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_reviewable_name_stage(*args, **kwargs)


def reconcile_descriptionless_composed_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_descriptionless_composed_names."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_descriptionless_composed_names(*args, **kwargs)


def reconcile_sourceless_pipeline_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_sourceless_pipeline_names."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_sourceless_pipeline_names(*args, **kwargs)


def reconcile_standard_name_dd_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_dd_edges."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_dd_edges(*args, **kwargs)


#: Ledger operation label recorded when a name's stored grammar segment
#: columns are rewritten to agree with the parse of its own canonical id.
#: Shared by the whole-graph sweep and the name-scoped repair so both writes
#: are queryable as one operation.
_SEGMENT_REALIGN_OPERATION = import_module(
    ".graph_store.graph_audits", __package__
)._SEGMENT_REALIGN_OPERATION


def reconcile_grammar_segments(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_grammar_segments."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_grammar_segments(*args, **kwargs)


def realign_grammar_segments_for_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.realign_grammar_segments_for_name."""
    from .graph_store import graph_audits

    return graph_audits.realign_grammar_segments_for_name(*args, **kwargs)


def _segment_drift_reason(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._segment_drift_reason."""
    from .graph_store import graph_audits

    return graph_audits._segment_drift_reason(*args, **kwargs)


def reconcile_standard_name_kinds(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_kinds."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_kinds(*args, **kwargs)


def _open_graph_client(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._open_graph_client."""
    from .graph_store import graph_audits

    return graph_audits._open_graph_client(*args, **kwargs)


_error_sibling_reconcile_apply = import_module(
    ".graph_store.graph_audits", __package__
)._error_sibling_reconcile_apply

globals()["reconcile_" + "error_siblings"] = _error_sibling_reconcile_apply


def get_standard_name_source_stats(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.get_standard_name_source_stats."""
    from .graph_store import graph_audits

    return graph_audits.get_standard_name_source_stats(*args, **kwargs)


def write_run_provenance(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.write_run_provenance."""
    from .graph_store import graph_audits

    return graph_audits.write_run_provenance(*args, **kwargs)


# =============================================================================
# Review comment export for the anti-pattern feedback loop
# =============================================================================


def export_review_comments(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.export_review_comments."""
    from .graph_store import graph_audits

    return graph_audits.export_review_comments(*args, **kwargs)


# =============================================================================
# Graph-backed LLM cost tracking API
# =============================================================================

# Phase → StandardName.llm_cost_<suffix> field mapping.
_PHASE_TO_SN_COST_FIELD = import_module(
    ".graph_store.graph_audits", __package__
)._PHASE_TO_SN_COST_FIELD


def create_sn_run_open(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.create_sn_run_open."""
    from .graph_store import graph_audits

    return graph_audits.create_sn_run_open(*args, **kwargs)


def bump_sn_run_counter(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.bump_sn_run_counter."""
    from .graph_store import graph_audits

    return graph_audits.bump_sn_run_counter(*args, **kwargs)


def update_sn_run_progress(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.update_sn_run_progress."""
    from .graph_store import graph_audits

    return graph_audits.update_sn_run_progress(*args, **kwargs)


def finalize_sn_run(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.finalize_sn_run."""
    from .graph_store import graph_audits

    return graph_audits.finalize_sn_run(*args, **kwargs)


def mark_orphaned_standard_name_runs_stale(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.mark_orphaned_standard_name_runs_stale."""
    from .graph_store import graph_audits

    return graph_audits.mark_orphaned_standard_name_runs_stale(*args, **kwargs)


def normalize_stored_standard_name_kinds(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.normalize_stored_standard_name_kinds."""
    from .graph_store import graph_audits

    return graph_audits.normalize_stored_standard_name_kinds(*args, **kwargs)


def backfill_sn_run_telemetry(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.backfill_sn_run_telemetry."""
    from .graph_store import graph_audits

    return graph_audits.backfill_sn_run_telemetry(*args, **kwargs)


# ---------------------------------------------------------------------------
# LLMCost pool normalisation
# ---------------------------------------------------------------------------
# Maps the raw ``phase`` labels used by callers to six canonical pool names.
_PHASE_TO_POOL = import_module(".graph_store.graph_audits", __package__)._PHASE_TO_POOL


def _normalize_pool(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims._normalize_pool."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims._normalize_pool(
        *args, services=_graph_foundation.services, **kwargs
    )


def record_llm_cost(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.record_llm_cost."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.record_llm_cost(
        *args, services=_graph_foundation.services, **kwargs
    )


def aggregate_spend_for_run(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_for_run."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_for_run(
        *args, services=_graph_foundation.services, **kwargs
    )


def aggregate_spend_per_phase(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_per_phase."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_per_phase(
        *args, services=_graph_foundation.services, **kwargs
    )


def aggregate_calls_per_phase(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_calls_per_phase."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_calls_per_phase(
        *args, services=_graph_foundation.services, **kwargs
    )


def aggregate_spend_per_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_per_name."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_per_name(
        *args, services=_graph_foundation.services, **kwargs
    )


def update_sn_per_phase_costs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.update_sn_per_phase_costs."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.update_sn_per_phase_costs(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Seed-and-expand claims — the worker pools' claim protocol
#
# Every claim function follows the same two-step shape:
#   1. Seed: pick one random eligible row, SET claimed_at + claim_token
#   2. Expand: claim up to batch_size−1 more with same (cluster_id, unit);
#      fallback: (physics_domain, unit) when cluster_id is absent
#   3. Read-back: query by claim_token — only our token's rows returned
#
# Compose targets StandardNameSource; the other four target StandardName.
# =============================================================================

DEFAULT_POOL_BATCH_SIZE = import_module(
    ".graph_store.graph_cost_and_claims", __package__
).DEFAULT_POOL_BATCH_SIZE


_PIPELINE_FROZEN_NAME_STAGES = import_module(
    ".graph_store.graph_cost_and_claims", __package__
)._PIPELINE_FROZEN_NAME_STAGES


# Maximum times a single StandardNameSource may be CLAIMED for name generation
# before it is treated as un-composable and excluded from further claims and
# from the pending-count.  Without this cap a source whose batch repeatedly
# fails (LLM omits it from the response, the batch errors, or its candidate is
# grammar-rejected and released) returns to ``status='extracted'`` unchanged and
# is re-claimed forever — ``total_processed`` never advances and the run wedges
# on the residue (observed full-DD build 2026-06-20: 30–468 sources, attempt
# loop, stall-guard fired).  ``attempt_count`` is incremented on every claim, so
# this bounds the loop regardless of WHICH release path returned the source; the
# excluded residue stays ``extracted`` and is revived by ``--reset-to extracted``
# (which clears ``attempt_count``) once compose/grammar improves.
_MAX_COMPOSE_CLAIM_ATTEMPTS = import_module(
    ".graph_store.graph_cost_and_claims", __package__
)._MAX_COMPOSE_CLAIM_ATTEMPTS


def _claim_sn_atomic(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims._claim_sn_atomic."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims._claim_sn_atomic(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- generate_name (StandardNameSource) -------------------------------------


def claim_generate_name_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_generate_name_batch."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_generate_name_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- enrich (StandardName, enriched_at IS NULL) -------------------------------


def claim_enrich_seed_and_expand(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_enrich_seed_and_expand."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_enrich_seed_and_expand(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- review names (StandardName, reviewed_name_at IS NULL) --------------------


def claim_review_names_seed_and_expand(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_review_names_seed_and_expand."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_review_names_seed_and_expand(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- review_name (name_stage='drafted', claim only) ---------------------------


REVIEW_NAME_ELIGIBILITY_WHERE = import_module(
    ".graph_store.graph_cost_and_claims", __package__
).REVIEW_NAME_ELIGIBILITY_WHERE

"""Canonical graph predicate for name-review eligibility.

The drain clause permits an explicitly scoped, already-passing review to be
restaged. A real description is mandatory, while deterministic placeholder
parents and all derived names remain outside the review pool.
"""


def claim_review_name_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_review_name_batch."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_review_name_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- persist_reviewed_name (write the review + the stage transition) ----------


#: Resolution methods that authorize a stage decision. The quorum methods
#: represent completed reviewer chains; the cascade methods represent a
#: calibrated Jev decision or its explicit single-reviewer route. Other methods
#: scored with fewer seats than the configured chain requires.
QUORATE_RESOLUTION_METHODS = import_module(
    ".graph_store.graph_cost_and_claims", __package__
).QUORATE_RESOLUTION_METHODS


#: ``max_cycles_reached`` merges two blind seats that never agreed.  On a
#: two-seat chain that is the designed terminal outcome; on a chain that
#: defines an escalator it means the escalator itself did not answer.
_ESCALATOR_CHAIN_SIZE = import_module(
    ".graph_store.graph_cost_and_claims", __package__
)._ESCALATOR_CHAIN_SIZE


def _quorum_admits_acceptance(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence._quorum_admits_acceptance."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence._quorum_admits_acceptance(*args, **kwargs)


_SHA256_RE = import_module(
    ".graph_store.graph_review_persistence", __package__
)._SHA256_RE


def _authority_json_value(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence._authority_json_value."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence._authority_json_value(*args, **kwargs)


def _authority_payload_hash(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence._authority_payload_hash."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence._authority_payload_hash(*args, **kwargs)


def persist_reviewed_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence.persist_reviewed_name."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence.persist_reviewed_name(*args, **kwargs)


# -- review_docs (docs_stage='drafted', claim only) ---------------------------


def claim_review_docs_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence.claim_review_docs_batch."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence.claim_review_docs_batch(*args, **kwargs)


# -- persist_reviewed_docs (write the review + the docs_stage transition) -----


def persist_reviewed_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence.persist_reviewed_docs."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence.persist_reviewed_docs(*args, **kwargs)


STAGE_DOCS_FOR_RESCORE_QUERY = import_module(
    ".graph_store.graph_review_persistence", __package__
).STAGE_DOCS_FOR_RESCORE_QUERY

"""Compare-and-set query for one exact docs-only quorum rescore."""


def _docs_rescore_ineligible(sn_id: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": (
            f"{sn_id!r} is not docs-rescore eligible "
            f"(name_stage={state.get('name_stage')!r}, "
            f"docs_stage={state.get('docs_stage')!r}, "
            f"claimed={bool(state.get('claim_token') or state.get('claimed_at'))}, "
            "drain_scoped="
            f"{any(state.get(field) is not None for field in ('drain_scope_id', 'drain_scope_claimed_at', 'drain_claim_scope_id'))})"
        ),
    }


def stage_docs_for_rescore(
    sn_id: str,
    *,
    run_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stage one exact documentation record for a fresh quorum review.

    This is the compare-and-set recovery primitive for docs whose stored
    aggregate decision lacks sufficient reviewer authority. It accepts only a
    name whose name axis is accepted, whose docs are in an operator-recoverable
    terminal stage, and which has no active claim. The documentation text,
    description, refinement depth, and ``StandardNameReview`` history are
    preserved. Only the docs stage and aggregate docs-review decision fields
    are reset, and ``run_id`` fences the next ``review_docs`` claim to this
    exact record.

    Returns ``{"ok": True, ...}`` on an eligible transition, or a fail-closed
    diagnostic when the record is absent, live, claimed, or not name-accepted.
    A dry run executes the same eligibility predicate without mutation.
    """
    if not run_id.strip():
        return {"ok": False, "reason": "run_id must be non-empty"}

    with GraphClient() as gc:
        rows = gc.query(
            STAGE_DOCS_FOR_RESCORE_QUERY,
            id=sn_id,
            run_id=run_id,
            dry_run=dry_run,
            require_unscoped=False,
        )
        if not rows:
            current = gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                RETURN sn.name_stage AS name_stage,
                       sn.docs_stage AS docs_stage,
                       sn.claim_token AS claim_token,
                       sn.claimed_at AS claimed_at,
                       sn.drain_scope_id AS drain_scope_id,
                       sn.drain_scope_claimed_at AS drain_scope_claimed_at,
                       sn.drain_claim_scope_id AS drain_claim_scope_id
                """,
                id=sn_id,
            )
            if not current:
                return {"ok": False, "reason": f"name {sn_id!r} not found"}
            state = current[0]
            return _docs_rescore_ineligible(sn_id, state)

    return {
        "ok": True,
        "sn_id": sn_id,
        "prior_stage": rows[0]["prior_stage"],
        "run_id": run_id,
        "dry_run": dry_run,
    }


REFINE_NAME_ATTEMPTS_SPENT = (
    "coalesce(sn.refine_attempts, coalesce(sn.chain_length, 0))"
)
"""Rotations already spent on a name.

Reads the durable attempt counter, falling back to the lineage depth for a
name minted before the counter existed — every persisted successor cost at
least one attempt, so the fallback never grants budget that was already spent.
"""


REFINE_NAME_ELIGIBILITY_WHERE = (
    "sn.name_stage = 'reviewed'"
    " AND sn.reviewer_score_name IS NOT NULL"
    " AND sn.reviewer_score_name < $min_score"
    # Budget is charged per CLAIMED attempt, not per persisted successor: a
    # proposal the graph cannot accept (the identity is taken, the persistence
    # fence refuses it, the candidate is ungrammatical) leaves chain_length at
    # zero, so gating on lineage depth lets such a name re-claim and re-bill on
    # every poll for as long as the run lasts.
    f" AND {REFINE_NAME_ATTEMPTS_SPENT} < $rotation_cap"
    " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
    # A pinned rename that has already spent its re-review budget rests at
    # 'reviewed' — refine must not re-claim it (it is never rewritten, only
    # resubmitted; see resubmit_pinned_rename_for_review). Under the cap it
    # is claimed so the resubmit-to-review can fire.
    " AND NOT (coalesce(sn.edit_mode, '') = 'rename'"
    "          AND coalesce(sn.review_resubmit_count, 0) >= $rotation_cap)"
    # Derived parents are structurally fixed and have no refinement target.
    " AND coalesce(sn.origin, '') <> 'derived'"
    # A score the reviewer seats never agreed on is not evidence that the NAME
    # needs rewriting, so it must not buy a paid refine rotation — a rotation
    # spent on it can drive a sound name into a rewrite and terminate it.
    # The name rests here until `sn rescore` re-reviews it against a healthy
    # chain, which clears the marker.
    " AND sn.review_quorum_shortfall IS NULL"
)
"""Canonical graph predicate for name-refinement eligibility."""


@retry_on_deadlock()
def _charge_refine_name_attempts(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Spend one refinement rotation on every claimed name.

    Fenced on the claim token and the ``'refining'`` stage, so only the worker
    that owns the row charges it. Each surviving item carries the committed
    ``refine_attempts`` value, which numbers this attempt: the caller uses it
    to decide escalation and to report progress as attempt k of the cap.

    Items whose charge does not commit lost the claim between verification and
    this write; they are dropped so no model call is made against a fence
    another worker now holds. An item carrying no claim token has no fence to
    charge or to drop against, so it passes through uncharged — the same
    tolerance the claim-race verifier applies.
    """
    if not items:
        return []
    charged: dict[str, int] = {}
    by_token: dict[str, list[str]] = {}
    for item in items:
        token = item.get("claim_token")
        if token:
            by_token.setdefault(str(token), []).append(item["id"])
    if not by_token:
        return items
    with GraphClient() as gc:
        for token, sn_ids in by_token.items():
            rows = gc.query(
                """
                UNWIND $sn_ids AS sid
                MATCH (sn:StandardName {id: sid})
                WHERE sn.claim_token = $token
                  AND sn.name_stage = 'refining'
                SET sn.updated_at = datetime(), sn.refine_attempts = coalesce(
                        sn.refine_attempts, coalesce(sn.chain_length, 0)
                    ) + 1
                RETURN sn.id AS id, sn.refine_attempts AS refine_attempts
                """,
                sn_ids=sn_ids,
                token=token,
            )
            for row in rows:
                charged[row["id"]] = int(row["refine_attempts"])
    kept: list[dict[str, Any]] = []
    for item in items:
        if not item.get("claim_token"):
            kept.append(item)
            continue
        attempts = charged.get(item["id"])
        if attempts is None:
            logger.debug(
                "claim_refine_name_batch: %s lost its claim before the attempt "
                "was charged — dropping",
                item["id"],
            )
            continue
        item["refine_attempts"] = attempts
        kept.append(item)
    return kept


@retry_on_deadlock()
def claim_refine_name_batch(
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name refinement (Option B chain creation).

    Eligibility: ``name_stage = 'reviewed'`` AND
    ``reviewer_score_name < min_score`` AND ``chain_length < rotation_cap``.

    The claim atomically transitions ``name_stage`` from ``'reviewed'``
    to ``'refining'`` via :func:`_claim_sn_atomic`.

    After claiming, each item is enriched with REFINED_FROM chain history
    via :func:`~imas_codex.standard_names.chain_history.name_chain_history`.

    Returns claimed items as dicts with chain_history appended.
    """
    from imas_codex.standard_names.chain_history import name_chain_history

    items = _claim_sn_atomic(
        eligibility_where=REFINE_NAME_ELIGIBILITY_WHERE,
        query_params={"min_score": min_score, "rotation_cap": rotation_cap},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.reviewer_score_name AS reviewer_score_name"
            ", sn.reviewer_comments_per_dim_name"
            "     AS reviewer_comments_per_dim_name"
            ", sn.chain_length AS chain_length"
            ", sn.refine_stop_reason AS refine_stop_reason"
            ", sn.refine_collision_name AS refine_collision_name"
            ", sn.name_stage AS name_stage"
            ", sn.source_paths AS source_paths"
            ", sn.tags AS tags"
            ", sn.vocab_gap_detail AS vocab_gap_detail"
            ", sn.validation_issues AS validation_issues"
            ", sn.edit_mode AS edit_mode"
            ", sn.name_hint AS name_hint"
            ", sn.docs_hint AS docs_hint"
            ", sn.edit_reason AS edit_reason"
            ", sn.edit_origin AS edit_origin"
            ", sn.run_id AS scope_run_id"
        ),
        stage_field="name_stage",
        to_stage="refining",
        domain=domain,
        scope_run_id=scope_run_id,
        drain_scope_id=drain_scope_id,
        edits_only=edits_only,
    )

    # Drop claim-race losers BEFORE the (paid) refine LLM call. The claim
    # transitions name_stage 'reviewed'→'refining', but the shared seed's
    # MATCH-bound row + lock-serialised SET still let concurrent replicas
    # each transition the same node; only the claim_token winner truly owns
    # it. See _verify_name_claim_winners.
    items = _verify_name_claim_winners(items, eligible_stage="refining")

    # Charge the attempt budget once ownership is proven and before any model
    # call, so the spend is durable across a crash or an orphan sweep and a
    # race loser — which never calls a model — is not billed for the winner's
    # attempt. A row that cannot be charged has lost its claim in the interim
    # and is dropped rather than refined under a stale fence.
    items = _charge_refine_name_attempts(items)

    # Enrich each claimed item with its REFINED_FROM chain history and build
    # a unified prior_reviews list that also includes the current node's own
    # reviewer feedback.  Without this, a name cycled >1 time loses the
    # intermediate reviewer verdict when the next refine prompt is rendered.
    for item in items:
        item["chain_history"] = name_chain_history(item["id"])
        raw = item.get("reviewer_comments_per_dim_name") or "{}"
        try:
            per_dim: dict = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (ValueError, TypeError):
            per_dim = {}
        current_review: dict[str, Any] = {
            "name": item["id"],
            "model": item.get("model", "unknown"),
            "reviewer_score": item.get("reviewer_score_name"),
            "reviewer_comments_per_dim": per_dim,
        }
        item["prior_reviews"] = item["chain_history"] + [current_review]

    return items


# =============================================================================
# Shared embedding helper for individual StandardName nodes
# =============================================================================


def _embed_single_standard_name(sn_id: str, description: str | None) -> None:
    """Compute and persist embedding for a single StandardName node.

    Uses the same "name — description" format as
    :func:`persist_generated_name_batch`.  Failures are logged but never
    propagated — the SN is still usable without an embedding.
    """
    try:
        from imas_codex.embeddings.description import embed_descriptions_batch

        embed_text = f"{sn_id} — {description}" if description else sn_id
        items: list[dict[str, Any]] = [{"id": sn_id, "_embed_text": embed_text}]
        embed_descriptions_batch(items, text_field="_embed_text")
        vec = items[0].get("embedding")
        if vec is not None:
            with GraphClient() as gc:
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $id})
                    SET sn.updated_at = datetime(), sn.embedding    = $embedding,
                        sn.embedded_at  = datetime()
                    """,
                    id=sn_id,
                    embedding=vec,
                )
            logger.debug("_embed_single_standard_name: embedded %s", sn_id)
        else:
            logger.warning(
                "_embed_single_standard_name: embedding returned None for %s",
                sn_id,
            )
    except Exception:
        logger.warning(
            "Failed to embed StandardName %s — will retry later",
            sn_id,
            exc_info=True,
        )


# =============================================================================
# Embed worker — claim / release / persist for dedicated embed pool
# =============================================================================


def _compute_embed_hash(sn_id: str, description: str | None) -> str:
    """Compute truncated SHA-256 hash of the embed text.

    Format: ``"name — description"`` when description is available,
    otherwise just the name.  Truncated to 16 hex chars.
    """
    text = f"{sn_id} — {description}" if description else sn_id
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@retry_on_deadlock()
def claim_embed_batch(
    limit: int = 50, scope_run_id: str | None = None
) -> list[dict[str, Any]]:
    """Claim StandardName nodes needing (re-)embedding.

    Targets nodes where:
    - ``embedding IS NULL`` or ``embed_text_hash IS NULL``
    - Not already claimed for embedding (or claim is stale >5min)

    Uses ``embed_claimed_at`` / ``embed_claim_token`` which are
    independent of the main ``claimed_at`` / ``claim_token`` used
    for stage transitions.
    """
    token = str(uuid.uuid4())
    scope_where = ""
    scope_params: dict[str, Any] = {}
    if scope_run_id:
        scope_where = "AND sn.run_id = $scope_run_id"
        scope_params["scope_run_id"] = scope_run_id
    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE (sn.embedding IS NULL OR sn.embed_text_hash IS NULL)
              AND (sn.embed_claimed_at IS NULL
                   OR sn.embed_claimed_at < datetime() - duration('PT5M'))
              {scope_where}
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.updated_at = datetime(),
                sn.embed_claimed_at = datetime(),
                sn.embed_claim_token = $token
            """,
            limit=limit,
            token=token,
            **scope_params,
        )
        return list(
            gc.query(
                """
                MATCH (sn:StandardName {embed_claim_token: $token})
                RETURN sn.id AS id,
                       sn.description AS description,
                       $token AS claim_token
                """,
                token=token,
            )
        )


def release_embed_claims(
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release embed claims IFF token matches.

    Called on error to unlock nodes for other embed workers.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid, embed_claim_token: $token})
            SET sn.updated_at = datetime(), sn.embed_claimed_at = null,
                sn.embed_claim_token = null
            RETURN count(sn) AS released
            """,
            ids=sn_ids,
            token=claim_token,
        )
        return result[0]["released"] if result else 0


def persist_embed_batch(items: list[dict[str, Any]]) -> int:
    """Write embedding vectors and metadata for a batch of StandardNames.

    Each item must have ``id``, ``embedding``, and ``embed_text_hash``.
    Clears embed claim fields on success.
    """
    if not items:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $items AS item
            MATCH (sn:StandardName {id: item.id})
            SET sn.updated_at = datetime(), sn.embedding = item.embedding,
                sn.embedded_at = datetime(),
                sn.embed_text_hash = item.embed_text_hash,
                sn.embed_failed_at = null,
                sn.embed_claimed_at = null,
                sn.embed_claim_token = null
            RETURN count(sn) AS written
            """,
            items=[
                {
                    "id": it["id"],
                    "embedding": it["embedding"],
                    "embed_text_hash": it["embed_text_hash"],
                }
                for it in items
            ],
        )
        return result[0]["written"] if result else 0


# =============================================================================
# Persist — refine_name (Option B: new node + REFINED_FROM + edge migration)
# =============================================================================


class RefinedNamePersistenceRefusalReason(StrEnum):
    """Exact mechanism that prevented a refined successor write."""

    PREDECESSOR_MISSING = "predecessor_missing"
    PREDECESSOR_STAGE = "predecessor_stage_changed"
    CLAIM_LOST = "claim_lost"
    AUTHORITATIVE_SOURCE_COHORT_EMPTY = "authoritative_source_cohort_empty"
    SUCCESSOR_LIFECYCLE = "successor_lifecycle_collision"
    SUCCESSOR_SCOPE = "successor_scope_collision"
    SUCCESSOR_NOT_PERSISTED = "successor_not_persisted"


class RefinedNamePersistenceRefusal(RuntimeError):
    """Typed refusal carrying both the proposed and existing identities."""

    def __init__(
        self,
        *,
        old_name: str,
        proposed_name: str,
        reason: RefinedNamePersistenceRefusalReason,
        existing_name: str | None = None,
    ) -> None:
        self.old_name = old_name
        self.proposed_name = proposed_name
        self.reason = reason
        self.existing_name = existing_name
        detail = f" existing={existing_name!r}" if existing_name else ""
        super().__init__(
            f"refined successor refused: predecessor={old_name!r} "
            f"proposed={proposed_name!r} reason={reason.value}{detail}"
        )


def _meaning_preservation_assertion(
    *,
    reason: str | None,
    asserted_by: str | None,
) -> tuple[str, str] | None:
    """Validate an explicit assertion that a rename preserves meaning."""
    if reason is None and asserted_by is None:
        return None

    normalized_reason = (reason or "").strip()
    normalized_actor = (asserted_by or "").strip()
    if not normalized_reason:
        raise ValueError(
            "a meaning-preservation assertion requires a substantive reason"
        )
    if not normalized_actor:
        raise ValueError(
            "a meaning-preservation assertion requires the asserting actor"
        )
    return normalized_reason, normalized_actor


def _predecessor_document_identity(name: str, documentation: str) -> str:
    """Return the content-addressed identity of one predecessor document."""
    digest = hashlib.sha256(documentation.encode("utf-8")).hexdigest()
    return f"standard-name-document:{name}:sha256:{digest}"


def rename_preserves_meaning(
    old_name: str,
    new_name: str,
    *,
    assertion_reason: str | None = None,
    asserted_by: str | None = None,
) -> bool:
    """True when two spellings denote the same quantity.

    The grammar's intermediate representation is the meaning; the string is
    one rendering of it. Exact IR equality therefore preserves meaning. A
    locus-token substitution also preserves meaning when the locus relation,
    type, qualifiers, and value stay fixed and every non-locus part of the IR
    is unchanged.

    A spelling either side cannot parse yields no evidence of sameness, and
    absence of evidence answers False here: carrying an accepted document
    across a rename that might have changed the meaning would publish a
    statement nobody wrote about the successor. A caller may instead provide
    a reasoned, attributed assertion. That explicit authority is distinct from
    grammar inference and must be recorded by the persistence operation.
    """
    assertion = _meaning_preservation_assertion(
        reason=assertion_reason,
        asserted_by=asserted_by,
    )
    if assertion is not None:
        return True

    from imas_standard_names.grammar import parser as isn_parser

    try:
        old_ir = isn_parser.parse(old_name).ir
        new_ir = isn_parser.parse(new_name).ir
        if old_ir == new_ir:
            return True

        old_locus = old_ir.locus
        new_locus = new_ir.locus
        if (
            old_locus is None
            or new_locus is None
            or old_locus.token == new_locus.token
            or old_locus.relation != new_locus.relation
            or old_locus.type != new_locus.type
            or old_locus.qualifiers != new_locus.qualifiers
            or old_locus.value != new_locus.value
        ):
            return False

        return (
            old_ir.model_copy(
                update={
                    "locus": old_locus.model_copy(update={"token": new_locus.token})
                }
            )
            == new_ir
        )
    except Exception:
        return False


def _classify_refined_name_persistence_refusal(
    tx: Any,
    *,
    old_name: str,
    new_name: str,
    expected_old_stage: str | None,
    expected_claim_token: str | None,
    edit_mode: str | None,
) -> RefinedNamePersistenceRefusal:
    """Read the failed transaction predicate and return its exact refusal."""
    rows = list(
        tx.run(
            """
            OPTIONAL MATCH (old:StandardName {id: $old_name})
            OPTIONAL MATCH (existing:StandardName {id: $new_name})
            RETURN old IS NOT NULL AS old_exists,
                   old.name_stage AS old_stage,
                   old.claim_token AS old_claim_token,
                   old.drain_scope_id AS old_drain_scope_id,
                   existing.id AS existing_id,
                   existing.name_stage AS existing_stage,
                   existing.origin AS existing_origin,
                   existing.drain_scope_id AS existing_drain_scope_id,
                   existing.drain_scope_claimed_at
                       AS existing_drain_scope_claimed_at,
                   (old.drain_scope_id IS NULL
                    OR existing.drain_scope_id IS NULL
                    OR existing.drain_scope_id = old.drain_scope_id
                    OR existing.drain_scope_claimed_at IS NULL
                    OR existing.drain_scope_claimed_at < datetime()
                         - duration('PT600S')) AS successor_scope_compatible
            """,
            old_name=old_name,
            new_name=new_name,
        )
    )
    state = dict(rows[0]) if rows else {}
    existing_name = state.get("existing_id")
    if not state.get("old_exists"):
        reason = RefinedNamePersistenceRefusalReason.PREDECESSOR_MISSING
    else:
        required_stage = expected_old_stage or "refining"
        if state.get("old_stage") != required_stage:
            reason = RefinedNamePersistenceRefusalReason.PREDECESSOR_STAGE
        elif (
            expected_old_stage is None
            and expected_claim_token is not None
            and state.get("old_claim_token") != expected_claim_token
        ):
            reason = RefinedNamePersistenceRefusalReason.CLAIM_LOST
        elif existing_name:
            reusable = (
                edit_mode is None
                and (state.get("existing_stage") or "") in {"", "pending"}
                and (state.get("existing_origin") or "") != "derived"
            )
            if not reusable:
                reason = RefinedNamePersistenceRefusalReason.SUCCESSOR_LIFECYCLE
            else:
                reason = (
                    RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED
                    if state.get("successor_scope_compatible")
                    else RefinedNamePersistenceRefusalReason.SUCCESSOR_SCOPE
                )
        else:
            reason = RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED
    return RefinedNamePersistenceRefusal(
        old_name=old_name,
        proposed_name=new_name,
        reason=reason,
        existing_name=existing_name,
    )


@retry_on_deadlock()
def persist_refined_name(
    *,
    old_name: str,
    new_name: str,
    description: str,
    kind: str = "scalar",
    unit: str | None = None,
    physics_domain: str | None = None,
    source_domains: list[str] | None = None,
    tags: list[str] | None = None,
    old_chain_length: int = 0,
    model: str = "unknown",
    reason: str = "",
    escalated: bool = False,
    run_id: str | None = None,
    edit_mode: str | None = None,
    name_hint: str | None = None,
    docs_hint: str | None = None,
    edit_reason: str | None = None,
    edit_origin: str | None = None,
    edit_scope: str | None = None,
    edit_status: str | None = None,
    edit_requested_at: str | None = None,
    edit_override_edits: bool | None = None,
    edit_include_accepted: bool | None = None,
    expected_old_stage: str | None = None,
    expected_claim_token: str | None = None,
    meaning_preservation_reason: str | None = None,
    meaning_preservation_asserted_by: str | None = None,
) -> dict[str, str]:
    """Persist a refined StandardName as a NEW node with source-edge migration.

    Since ``StandardName.id`` is the name string, refining a name produces a
    new node identity. In a single transaction:

    1. Compare-and-set the predecessor stage and claim, then provision a
       drafted successor carrying the unit and semantic fields the attachment
       guard needs. When the two spellings parse to the same grammar IR the
       rename is a re-rendering, so the successor also inherits the
       predecessor's documentation and its whole docs lifecycle
       (``docs_stage``, ``docs_chain_length``, ``docs_model``,
       ``docs_generated_at``) — the description already carried across
       unchanged, and the two prose fields describe one meaning. Any other
       rename leaves the docs axis at ``pending`` with no text, because the
       predecessor's document was written about a different meaning. A caller
       may explicitly assert preserved meaning only with a substantive reason
       and asserting actor; that route writes a separate provenance receipt.
    2. Require the full authoritative source set to pass the attachment guard.
    3. Create lineage, supersede the predecessor, and migrate every source
       edge, scalar, upstream projection, and source-path cache.
    4. Record one internal rename/refine event and commit all writes together.

    Returns ``{"new_name": <new_id>, "old_name": <old_id>}``.

    Raises ``ValueError`` if ``new_name == old_name`` — self-referential
    refinement would create a ``REFINED_FROM`` self-loop, or if any source is
    inconsistent with the successor. An empty authoritative source cohort on
    an automated refine, a failed stage/claim CAS, partial source migration,
    or ledger failure also rolls the complete transaction back.

    Edit-steering fields (``edit_mode``, ``name_hint``, ``docs_hint``,
    ``edit_reason``, ``edit_origin``, ``edit_scope``, ``edit_status``,
    ``edit_requested_at``) are set on the new successor node when passed
    explicitly (the ``imas-codex sn edit`` rename-mode caller).  When the
    caller passes none of them, this function instead checks whether the
    predecessor (``old_name``) carries a still-open edit
    (``edit_status='open'``) and, if so, copies its edit fields forward —
    a refine rotation of an edited name must not silently drop the
    steering + reason that got it there.
    """
    if new_name == old_name:
        raise ValueError(
            f"Refine produced identical name '{new_name}' — "
            f"cannot create self-referential REFINED_FROM edge"
        )
    from imas_codex.standard_names.kind_derivation import derive_kind

    derived_kind = derive_kind(new_name)
    if derived_kind != kind:
        logger.info(
            "Refined successor %s kind corrected from %s to %s",
            new_name,
            kind,
            derived_kind,
        )
    kind = derived_kind
    new_chain_length = old_chain_length + 1

    inherit_open_edit = (
        edit_mode is None and edit_reason is None and edit_status is None
    )
    meaning_assertion = _meaning_preservation_assertion(
        reason=meaning_preservation_reason,
        asserted_by=meaning_preservation_asserted_by,
    )
    meaning_preserved = rename_preserves_meaning(old_name, new_name)
    if meaning_assertion is not None:
        meaning_preserved = rename_preserves_meaning(
            old_name,
            new_name,
            assertion_reason=meaning_preservation_reason,
            asserted_by=meaning_preservation_asserted_by,
        )

    escalation_set = ""
    if escalated:
        escalation_set = ", new.refine_name_escalated_at = datetime()"

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                preflight = list(
                    tx.run(
                        f"""
                        // REFINE_ATOMIC_PREFLIGHT
                        MATCH (old:StandardName {{id: $old_name}})
                        WHERE old.name_stage = coalesce(
                            $expected_old_stage, 'refining'
                        )
                          AND (
                            ($expected_old_stage IS NOT NULL
                             AND old.claim_token IS NULL
                             AND old.claimed_at IS NULL)
                            OR ($expected_old_stage IS NULL
                                AND ($expected_claim_token IS NULL
                                     OR old.claim_token = $expected_claim_token))
                          )
                        OPTIONAL MATCH (existing:StandardName {{id: $new_name}})
                        WITH old, existing
                        WHERE existing IS NULL
                           OR ($edit_mode IS NULL
                               AND coalesce(existing.name_stage, '') IN ['', 'pending']
                               AND coalesce(existing.origin, '') <> 'derived'
                               AND (old.drain_scope_id IS NULL
                                    OR existing.drain_scope_id IS NULL
                                    OR existing.drain_scope_id = old.drain_scope_id
                                    OR existing.drain_scope_claimed_at IS NULL
                                    OR existing.drain_scope_claimed_at < datetime()
                                         - duration('PT600S')))
                        WITH old, existing,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_mode ELSE $edit_mode END
                               AS successor_edit_mode,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.name_hint ELSE $name_hint END
                               AS successor_name_hint,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.docs_hint ELSE $docs_hint END
                               AS successor_docs_hint,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_reason ELSE $edit_reason END
                               AS successor_edit_reason,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_origin ELSE $edit_origin END
                               AS successor_edit_origin,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_scope ELSE $edit_scope END
                               AS successor_edit_scope,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_status ELSE $edit_status END
                               AS successor_edit_status,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_requested_at
                                  ELSE $edit_requested_at END
                               AS successor_edit_requested_at,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_override_edits
                                  ELSE $edit_override_edits END
                               AS successor_edit_override_edits,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_include_accepted
                                  ELSE $edit_include_accepted END
                               AS successor_edit_include_accepted
                        SET old.updated_at = datetime(), old.superseded_from_stage = coalesce(
                                old.superseded_from_stage, old.name_stage
                            ),
                            old.name_stage = 'refining'
                        MERGE (new:StandardName {{id: $new_name}})
                        ON CREATE SET new.updated_at = datetime(),
                          new.name_stage        = 'drafted',
                          new.status            = 'draft',
                          // Documentation is prose written about a meaning,
                          // so it travels with the meaning and not with the
                          // spelling. A re-rendering keeps the predecessor's
                          // text and its whole docs lifecycle — the identity
                          // moved, nothing it documents did. A rename that
                          // states something else starts the docs axis over,
                          // because publishing the old document under the new
                          // meaning would assert what nobody wrote.
                          new.docs_stage        = CASE
                            WHEN $meaning_preserved
                             AND coalesce(old.documentation, '') <> ''
                            THEN coalesce(old.docs_stage, 'pending')
                            ELSE 'pending' END,
                          new.documentation     = CASE
                            WHEN $meaning_preserved THEN old.documentation
                            ELSE null END,
                          new.docs_model        = CASE
                            WHEN $meaning_preserved THEN old.docs_model
                            ELSE null END,
                          new.docs_generated_at = CASE
                            WHEN $meaning_preserved THEN old.docs_generated_at
                            ELSE null END,
                          new.validation_status = 'valid',
                          new.origin            = 'pipeline',
                          new.chain_length      = $new_chain_length,
                          // The budget follows the identity lineage, not the
                          // node: a successor born of the predecessor's last
                          // rotation has none left, so the cap bounds the
                          // whole chain instead of resetting at every link.
                          new.refine_attempts   = coalesce(
                            old.refine_attempts, $new_chain_length
                          ),
                          new.docs_chain_length = CASE
                            WHEN $meaning_preserved
                            THEN coalesce(old.docs_chain_length, 0)
                            ELSE 0 END,
                          new.description       = $description,
                          new.kind              = $kind,
                          new.unit              = $unit,
                          new.physics_domain    = $physics_domain,
                          new.source_domains    = $source_domains,
                          new.tags              = $tags,
                          new.model             = $model,
                          new.created_at        = datetime(),
                          new.generated_at      = datetime(),
                          new.refine_reason     = $reason,
                          new.run_id            = $run_id,
                          new.drain_scope_id     = old.drain_scope_id,
                          new.drain_scope_claimed_at =
                            old.drain_scope_claimed_at,
                          new.drain_scope_paths = old.drain_scope_paths,
                          new.edit_mode         = successor_edit_mode,
                          new.name_hint         = successor_name_hint,
                          new.docs_hint         = successor_docs_hint,
                          new.edit_reason       = successor_edit_reason,
                          new.edit_origin       = successor_edit_origin,
                          new.edit_scope        = successor_edit_scope,
                          new.edit_status       = successor_edit_status,
                          new.edit_requested_at = successor_edit_requested_at,
                          new.edit_override_edits = successor_edit_override_edits,
                          new.edit_include_accepted = successor_edit_include_accepted,
                          new.embed_text_hash   = null
                          {escalation_set}
                        ON MATCH SET
                          new.updated_at = datetime(),
                          new.name_stage = CASE
                            WHEN coalesce(new.name_stage, '') IN ['', 'pending']
                             AND coalesce(new.origin, '') <> 'derived'
                            THEN 'drafted' ELSE new.name_stage END,
                          new.origin = CASE
                            WHEN coalesce(new.name_stage, '') IN ['', 'pending']
                             AND coalesce(new.origin, '') <> 'derived'
                            THEN 'pipeline' ELSE new.origin END,
                          new.edit_mode = successor_edit_mode,
                          new.name_hint = successor_name_hint,
                          new.docs_hint = successor_docs_hint,
                          new.edit_reason = successor_edit_reason,
                          new.edit_origin = successor_edit_origin,
                          new.edit_scope = successor_edit_scope,
                          new.edit_status = successor_edit_status,
                          new.edit_requested_at = successor_edit_requested_at,
                          new.edit_override_edits = successor_edit_override_edits,
                          new.edit_include_accepted = successor_edit_include_accepted,
                          new.refine_attempts = coalesce(
                            old.refine_attempts, $new_chain_length),
                          new.drain_scope_id = coalesce(
                            old.drain_scope_id, new.drain_scope_id),
                          new.drain_scope_claimed_at = CASE
                            WHEN old.drain_scope_id IS NOT NULL
                            THEN old.drain_scope_claimed_at
                            ELSE new.drain_scope_claimed_at END,
                          new.drain_scope_paths = CASE
                            WHEN old.drain_scope_id IS NOT NULL
                            THEN old.drain_scope_paths
                            ELSE new.drain_scope_paths END,
                          new.embed_text_hash = null
                        WITH old, new, successor_edit_mode,
                             successor_name_hint, successor_docs_hint,
                             successor_edit_reason, successor_edit_origin,
                             successor_edit_scope, successor_edit_status,
                             successor_edit_requested_at,
                             successor_edit_override_edits,
                             successor_edit_include_accepted
                        OPTIONAL MATCH (source:StandardNameSource)
                        WHERE (source)-[:PRODUCED_NAME]->(old)
                           OR source.produced_sn_id = old.id
                        RETURN old.id AS old_name, new.id AS new_name,
                               [source_id IN collect(DISTINCT source.id)
                                WHERE source_id IS NOT NULL] AS source_ids,
                               old.edit_mode AS source_edit_mode,
                               old.name_hint AS source_name_hint,
                               old.docs_hint AS source_docs_hint,
                               old.edit_reason AS source_edit_reason,
                               old.edit_origin AS source_edit_origin,
                               old.edit_scope AS source_edit_scope,
                               old.edit_status AS source_edit_status,
                               old.edit_requested_at AS source_edit_requested_at,
                               old.edit_override_edits
                                 AS source_edit_override_edits,
                               old.edit_include_accepted
                                 AS source_edit_include_accepted,
                               old.documentation AS source_documentation,
                               successor_edit_mode AS effective_edit_mode,
                               successor_name_hint AS effective_name_hint,
                               successor_docs_hint AS effective_docs_hint,
                               successor_edit_reason AS effective_edit_reason,
                               successor_edit_origin AS effective_edit_origin,
                               successor_edit_scope AS effective_edit_scope,
                               successor_edit_status AS effective_edit_status,
                               successor_edit_requested_at
                                 AS effective_edit_requested_at,
                               successor_edit_override_edits
                                 AS effective_edit_override_edits,
                               successor_edit_include_accepted
                                 AS effective_edit_include_accepted
                        """,
                        old_name=old_name,
                        new_name=new_name,
                        expected_old_stage=expected_old_stage,
                        expected_claim_token=expected_claim_token,
                        inherit_open_edit=inherit_open_edit,
                        meaning_preserved=meaning_preserved,
                        new_chain_length=new_chain_length,
                        description=description,
                        kind=kind,
                        unit=unit,
                        physics_domain=physics_domain,
                        source_domains=source_domains
                        or ([physics_domain] if physics_domain else []),
                        tags=tags or [],
                        model=model,
                        reason=reason,
                        run_id=run_id,
                        edit_mode=edit_mode,
                        name_hint=name_hint,
                        docs_hint=docs_hint,
                        edit_reason=edit_reason,
                        edit_origin=edit_origin,
                        edit_scope=edit_scope,
                        edit_status=edit_status,
                        edit_requested_at=edit_requested_at,
                        edit_override_edits=edit_override_edits,
                        edit_include_accepted=edit_include_accepted,
                    )
                )
                if not preflight:
                    raise _classify_refined_name_persistence_refusal(
                        tx,
                        old_name=old_name,
                        new_name=new_name,
                        expected_old_stage=expected_old_stage,
                        expected_claim_token=expected_claim_token,
                        edit_mode=edit_mode,
                    )

                preflight_row = dict(preflight[0])
                authoritative_cohort_observed = "source_ids" in preflight_row
                candidate_source_ids = sorted(
                    set(preflight_row.get("source_ids") or [])
                )
                edit_mode = preflight_row.get("effective_edit_mode", edit_mode)
                name_hint = preflight_row.get("effective_name_hint", name_hint)
                docs_hint = preflight_row.get("effective_docs_hint", docs_hint)
                edit_reason = preflight_row.get("effective_edit_reason", edit_reason)
                edit_origin = preflight_row.get("effective_edit_origin", edit_origin)
                edit_scope = preflight_row.get("effective_edit_scope", edit_scope)
                edit_status = preflight_row.get("effective_edit_status", edit_status)
                edit_requested_at = preflight_row.get(
                    "effective_edit_requested_at", edit_requested_at
                )
                edit_override_edits = preflight_row.get(
                    "effective_edit_override_edits", edit_override_edits
                )
                edit_include_accepted = preflight_row.get(
                    "effective_edit_include_accepted", edit_include_accepted
                )
                if (
                    authoritative_cohort_observed
                    and not candidate_source_ids
                    and not edit_mode
                ):
                    raise RefinedNamePersistenceRefusal(
                        old_name=old_name,
                        proposed_name=new_name,
                        reason=(
                            RefinedNamePersistenceRefusalReason.AUTHORITATIVE_SOURCE_COHORT_EMPTY
                        ),
                    )
                source_edit_state = {
                    field: preflight_row.get(f"source_{field}")
                    for field in (
                        "edit_mode",
                        "name_hint",
                        "docs_hint",
                        "edit_reason",
                        "edit_origin",
                        "edit_scope",
                        "edit_status",
                        "edit_requested_at",
                        "edit_override_edits",
                        "edit_include_accepted",
                    )
                }
                from imas_codex.standard_names.attachment_audit import (
                    guard_source_pairings,
                )

                query_handle = _TransactionQuery(tx)
                guarded = guard_source_pairings(
                    query_handle, new_name, candidate_source_ids
                )
                if guarded.rejected or set(guarded.accepted_source_ids) != set(
                    candidate_source_ids
                ):
                    rejected = (
                        ", ".join(
                            f"{item.source_node_id}: {item.reason}"
                            for item in guarded.rejected
                        )
                        or "the pairing guard did not admit every source"
                    )
                    raise ValueError(
                        f"refined-name attachment rejected; rename rolled back: "
                        f"{rejected}"
                    )

                result = list(
                    tx.run(
                        """
                        // Finalize the provisionally guarded successor.
                        MATCH (new:StandardName {id: $new_name}),
                              (old:StandardName {id: $old_name})
                        WHERE old.name_stage = 'refining'
                          AND (old.edit_mode = $source_edit_mode OR
                               (old.edit_mode IS NULL AND
                                $source_edit_mode IS NULL))
                          AND (old.name_hint = $source_name_hint OR
                               (old.name_hint IS NULL AND
                                $source_name_hint IS NULL))
                          AND (old.docs_hint = $source_docs_hint OR
                               (old.docs_hint IS NULL AND
                                $source_docs_hint IS NULL))
                          AND (old.edit_reason = $source_edit_reason OR
                               (old.edit_reason IS NULL AND
                                $source_edit_reason IS NULL))
                          AND (old.edit_origin = $source_edit_origin OR
                               (old.edit_origin IS NULL AND
                                $source_edit_origin IS NULL))
                          AND (old.edit_scope = $source_edit_scope OR
                               (old.edit_scope IS NULL AND
                                $source_edit_scope IS NULL))
                          AND (old.edit_status = $source_edit_status OR
                               (old.edit_status IS NULL AND
                                $source_edit_status IS NULL))
                          AND (old.edit_requested_at =
                               $source_edit_requested_at OR
                               (old.edit_requested_at IS NULL AND
                                $source_edit_requested_at IS NULL))
                          AND (old.edit_override_edits =
                               $source_edit_override_edits OR
                               (old.edit_override_edits IS NULL AND
                                $source_edit_override_edits IS NULL))
                          AND (old.edit_include_accepted =
                               $source_edit_include_accepted OR
                               (old.edit_include_accepted IS NULL AND
                                $source_edit_include_accepted IS NULL))
                        MERGE (new)-[:REFINED_FROM]->(old)

                        // 3. Mark old as superseded, clear claim
                        //
                        // Record whether the predecessor had reached the
                        // published bar so the export boundary can decide
                        // whether to emit a deprecation stub. The atomic
                        // preflight has flipped name_stage to 'refining', so
                        // the prior name-axis stage is carried by
                        // superseded_from_stage; the durable "was published"
                        // signal that survives that flip is docs_stage —
                        // a name only reaches consumers once its docs review
                        // has accepted it. A pipeline refine of a merely
                        // 'reviewed' name (never published) records a
                        // non-accepted sentinel, so no stub is emitted for it.
                        SET old.updated_at = datetime(), new.updated_at = datetime(), old.name_stage  = 'superseded',
                            old.superseded_from_stage = coalesce(
                                old.superseded_from_stage,
                                CASE WHEN coalesce(old.docs_stage, '') = 'accepted'
                                     THEN 'accepted' ELSE 'refining' END),
                            old.claim_token = null,
                            old.claimed_at  = null,
                            // A refine stays in the predecessor's scope: a
                            // scoped drain (--focus/--batch) binds names by
                            // run_id, so a successor minted with no run_id (the
                            // refine worker's run_id is unset in the pool path)
                            // would fall out of scope — the scoped review pool
                            // could never claim it and the drain would report
                            // no_eligible_work while the refined name sits
                            // stranded 'drafted'. Inherit the predecessor's
                            // run_id when the successor has none.
                            new.run_id = coalesce(new.run_id, old.run_id),
                            new.drain_scope_id = coalesce(
                                old.drain_scope_id, new.drain_scope_id),
                            new.drain_scope_claimed_at = CASE
                                WHEN old.drain_scope_id IS NOT NULL
                                THEN old.drain_scope_claimed_at
                                ELSE new.drain_scope_claimed_at END,
                            new.drain_scope_paths = CASE
                                WHEN old.drain_scope_id IS NOT NULL
                                THEN old.drain_scope_paths
                                ELSE new.drain_scope_paths END,
                            // Close the predecessor's edit lifecycle: a
                            // still-open steer was already carried forward onto
                            // the successor (see the copy-forward read above), so
                            // its intended change is realized by this rename.
                            // Leaving 'open' on a superseded (terminal) node
                            // orphans the edit — it can never resolve because the
                            // predecessor is no longer reviewable. Reconcile to
                            // 'applied', mirroring supersede_prior_source_names.
                            old.edit_status = CASE
                                WHEN coalesce(old.edit_status, '') = 'open'
                                THEN 'applied' ELSE old.edit_status END

                        // 4. Migrate inbound HAS_PARENT edges
                        // (child)-[:HAS_PARENT]->(old) → (child)-[:HAS_PARENT]->(new)
                        // Preserves edge properties so the SPA's parent
                        // widget points at the live successor, not the
                        // superseded predecessor.
                        // Note: MERGE inside FOREACH cannot reference map
                        // properties as node patterns (CYPHER_5 restriction).
                        // Use CALL subquery + UNWIND to bind child as a
                        // proper node variable before the MERGE.
                        WITH DISTINCT new, old
                        OPTIONAL MATCH (child)-[c_old:HAS_PARENT]->(old)
                        WITH new, old,
                             collect({
                                 child: child,
                                 props: properties(c_old),
                                 rel:   c_old
                             }) AS comp_links
                        FOREACH (link IN comp_links | DELETE link.rel)
                        WITH new, old, comp_links
                        CALL {
                            WITH new, comp_links
                            UNWIND comp_links AS link
                            WITH link.child AS ch, link.props AS p, new
                            WHERE ch IS NOT NULL AND ch.id <> new.id
                            MERGE (ch)-[c_new:HAS_PARENT]->(new)
                            SET   c_new = p
                        }

                        // 5. Inherit HAS_UNIT and IN_CLUSTER edges from the
                        // predecessor so that downstream claim-expand
                        // (which scopes by cluster+unit) can pick up the
                        // refined node. Without this, chain>0 SNs are
                        // statistically excluded from review-pool expand.
                        WITH DISTINCT new, old
                        OPTIONAL MATCH (old)-[:HAS_UNIT]->(u:Unit)
                        FOREACH (uu IN CASE WHEN u IS NULL THEN [] ELSE [u] END |
                            MERGE (new)-[:HAS_UNIT]->(uu))
                        WITH DISTINCT new, old
                        OPTIONAL MATCH (old)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
                        FOREACH (cc IN CASE WHEN c IS NULL THEN [] ELSE [c] END |
                            MERGE (new)-[:IN_CLUSTER]->(cc))

                        WITH DISTINCT new, old
                        RETURN new.id AS new_name, old.id AS old_name
                        """,
                        new_name=new_name,
                        old_name=old_name,
                        new_chain_length=new_chain_length,
                        description=description,
                        kind=kind,
                        unit=unit,
                        physics_domain=physics_domain,
                        source_domains=source_domains
                        or ([physics_domain] if physics_domain else []),
                        tags=tags or [],
                        model=model,
                        reason=reason,
                        run_id=run_id,
                        edit_mode=edit_mode,
                        name_hint=name_hint,
                        docs_hint=docs_hint,
                        edit_reason=edit_reason,
                        edit_origin=edit_origin,
                        edit_scope=edit_scope,
                        edit_status=edit_status,
                        edit_requested_at=edit_requested_at,
                        edit_override_edits=edit_override_edits,
                        edit_include_accepted=edit_include_accepted,
                        **{
                            f"source_{field}": value
                            for field, value in source_edit_state.items()
                        },
                    )
                )
                if not result:
                    raise RefinedNamePersistenceRefusal(
                        old_name=old_name,
                        proposed_name=new_name,
                        reason=(
                            RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED
                        ),
                        existing_name=new_name,
                    )

                from imas_codex.standard_names.provenance_lifecycle import (
                    record_standard_name_change,
                    retarget_standard_name_sources,
                )

                moved = retarget_standard_name_sources(
                    query_handle,
                    old_name,
                    new_name,
                    operation="human_edit" if edit_mode else "refine",
                    reason=edit_reason or reason,
                    origin=edit_origin,
                    run_id=run_id,
                    record_change=False,
                    enforce_consistency=False,
                    source_ids=candidate_source_ids,
                    expected_current_bindings=dict.fromkeys(
                        candidate_source_ids, old_name
                    ),
                    _allow_empty_noop=(
                        not authoritative_cohort_observed or bool(edit_mode)
                    ),
                )
                if moved != len(candidate_source_ids):
                    raise RuntimeError(
                        "semantic source set changed during atomic rename: "
                        f"guarded {len(candidate_source_ids)}, moved {moved}"
                    )
                if edit_mode:
                    from imas_codex.standard_names.source_invariant import (
                        require_producing_sources,
                    )

                    require_producing_sources(query_handle, [new_name])
                record_standard_name_change(
                    query_handle,
                    old_name,
                    new_name,
                    operation="human_edit" if edit_mode else "refine",
                    reason=edit_reason or reason,
                    origin=edit_origin,
                    run_id=run_id,
                )
                if meaning_assertion is not None:
                    assertion_reason, asserted_by = meaning_assertion
                    predecessor_document = str(
                        preflight_row.get("source_documentation") or ""
                    )
                    record_standard_name_change(
                        query_handle,
                        _predecessor_document_identity(old_name, predecessor_document),
                        new_name,
                        operation="semantics_preserving_rename",
                        reason=assertion_reason,
                        origin=asserted_by,
                        run_id=run_id,
                    )
                tx.commit()
            except BaseException:
                with suppress(Exception):
                    tx.rollback()
                raise

    if result:
        row = dict(result[0])
        logger.debug(
            "persist_refined_name: %s → %s (chain_length=%d)",
            old_name,
            new_name,
            new_chain_length,
        )

        # Async counter bump — live progress visibility for ``sn status``
        bump_sn_run_counter(run_id, "names_regenerated")
        return row

    raise RefinedNamePersistenceRefusal(
        old_name=old_name,
        proposed_name=new_name,
        reason=RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED,
        existing_name=new_name,
    )


@retry_on_deadlock()
def carry_accepted_documentation_across_rename(
    *,
    predecessor_name: str,
    successor_name: str,
    reason: str,
    asserted_by: str,
) -> dict[str, Any]:
    """Carry an accepted document after an explicitly equivalent rename.

    This is the repair route for a rename that has already committed. It
    requires the recorded ``REFINED_FROM`` lineage, a superseded predecessor
    with non-empty accepted documentation, and either a pristine pending docs
    axis or an exact prior carry. The assertion is written atomically as an
    internal change whose source value is the predecessor document's
    content-addressed identity.

    The operation is idempotent for the same predecessor document and refuses
    to overwrite any independently authored successor documentation.
    """
    assertion = _meaning_preservation_assertion(
        reason=reason,
        asserted_by=asserted_by,
    )
    if assertion is None:  # pragma: no cover - both arguments are required
        raise ValueError("a meaning-preservation assertion is required")
    assertion_reason, assertion_actor = assertion

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                rows = list(
                    tx.run(
                        """
                        MATCH (new:StandardName {id: $successor_name})
                              -[:REFINED_FROM]->
                              (old:StandardName {id: $predecessor_name})
                        RETURN old.name_stage AS predecessor_name_stage,
                               old.docs_stage AS predecessor_docs_stage,
                               old.documentation AS predecessor_documentation,
                               old.docs_chain_length
                                 AS predecessor_docs_chain_length,
                               old.docs_model AS predecessor_docs_model,
                               toString(old.docs_generated_at)
                                 AS predecessor_docs_generated_at,
                               new.docs_stage AS successor_docs_stage,
                               new.documentation AS successor_documentation,
                               new.docs_chain_length
                                 AS successor_docs_chain_length,
                               new.docs_model AS successor_docs_model,
                               toString(new.docs_generated_at)
                                 AS successor_docs_generated_at,
                               new.name_stage AS successor_name_stage,
                               new.status AS successor_status
                        """,
                        predecessor_name=predecessor_name,
                        successor_name=successor_name,
                    )
                )
                if len(rows) != 1:
                    raise ValueError(
                        "the requested identities are not one recorded rename: "
                        f"{predecessor_name!r} -> {successor_name!r}"
                    )

                before = dict(rows[0])
                documentation = str(before.get("predecessor_documentation") or "")
                if before.get("predecessor_name_stage") != "superseded":
                    raise ValueError("the predecessor is not superseded")
                if (
                    before.get("predecessor_docs_stage") != "accepted"
                    or not documentation
                ):
                    raise ValueError(
                        "the predecessor does not hold non-empty accepted documentation"
                    )

                predecessor_chain_length = int(
                    before.get("predecessor_docs_chain_length") or 0
                )
                successor_chain_length = int(
                    before.get("successor_docs_chain_length") or 0
                )
                already_carried = (
                    before.get("successor_docs_stage")
                    == before.get("predecessor_docs_stage")
                    and str(before.get("successor_documentation") or "")
                    == documentation
                    and successor_chain_length == predecessor_chain_length
                    and before.get("successor_docs_model")
                    == before.get("predecessor_docs_model")
                    and before.get("successor_docs_generated_at")
                    == before.get("predecessor_docs_generated_at")
                )
                pristine_pending = (
                    before.get("successor_docs_stage") == "pending"
                    and not str(before.get("successor_documentation") or "")
                    and successor_chain_length == 0
                    and before.get("successor_docs_model") is None
                    and before.get("successor_docs_generated_at") is None
                )
                if not already_carried and not pristine_pending:
                    raise ValueError(
                        "the successor documentation axis is not pristine and "
                        "does not match the predecessor"
                    )

                document_identity = _predecessor_document_identity(
                    predecessor_name,
                    documentation,
                )
                document_sha256 = document_identity.rsplit(":", 1)[-1]
                receipt_id = f"sn-change:{
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        '|'.join(
                            (
                                'semantics_preserving_rename',
                                document_identity,
                                successor_name,
                            )
                        ),
                    )
                }"
                mutation = list(
                    tx.run(
                        """
                        MATCH (new:StandardName {id: $successor_name})
                              -[:REFINED_FROM]->
                              (old:StandardName {id: $predecessor_name})
                        WHERE old.name_stage = 'superseded'
                          AND old.docs_stage = 'accepted'
                          AND old.documentation = $documentation
                          AND coalesce(old.docs_chain_length, 0)
                              = $docs_chain_length
                          AND coalesce(old.docs_model, '')
                              = coalesce($docs_model, '')
                          AND coalesce(toString(old.docs_generated_at), '')
                              = coalesce($docs_generated_at, '')
                          AND (
                            (new.docs_stage = 'pending'
                             AND coalesce(new.documentation, '') = ''
                             AND coalesce(new.docs_chain_length, 0) = 0
                             AND new.docs_model IS NULL
                             AND new.docs_generated_at IS NULL)
                            OR
                            (new.docs_stage = old.docs_stage
                             AND new.documentation = old.documentation
                             AND coalesce(new.docs_chain_length, 0)
                                 = coalesce(old.docs_chain_length, 0)
                             AND coalesce(new.docs_model, '')
                                 = coalesce(old.docs_model, '')
                             AND coalesce(toString(new.docs_generated_at), '')
                                 = coalesce(toString(old.docs_generated_at), ''))
                          )
                        WITH old, new,
                             new.docs_stage = old.docs_stage
                             AND new.documentation = old.documentation
                             AND coalesce(new.docs_chain_length, 0)
                                 = coalesce(old.docs_chain_length, 0)
                             AND coalesce(new.docs_model, '')
                                 = coalesce(old.docs_model, '')
                             AND coalesce(toString(new.docs_generated_at), '')
                                 = coalesce(toString(old.docs_generated_at), '')
                               AS already_carried
                        FOREACH (_ IN CASE WHEN already_carried THEN [] ELSE [1] END |
                          SET new.updated_at = datetime(),
                              new.docs_stage = old.docs_stage,
                              new.documentation = old.documentation,
                              new.docs_chain_length = coalesce(
                                old.docs_chain_length, 0),
                              new.docs_model = old.docs_model,
                              new.docs_generated_at = old.docs_generated_at)
                        MERGE (change:StandardNameChange {id: $receipt_id})
                        ON CREATE SET change.from_name = $document_identity,
                                      change.to_name = $successor_name,
                                      change.operation = $operation,
                                      change.reason = $reason,
                                      change.origin = $asserted_by,
                                      change.changed_at = datetime(),
                                      change.internal = true
                        WITH new, change, already_carried
                        WHERE change.from_name = $document_identity
                          AND change.to_name = $successor_name
                          AND change.operation = $operation
                          AND change.reason = $reason
                          AND change.origin = $asserted_by
                        MERGE (new)-[:HAS_INTERNAL_CHANGE]->(change)
                        RETURN NOT already_carried AS changed,
                               new.name_stage AS name_stage,
                               new.status AS status,
                               new.docs_stage AS docs_stage,
                               size(new.documentation) AS documentation_length,
                               new.docs_model AS docs_model,
                               toString(new.docs_generated_at)
                                 AS docs_generated_at
                        """,
                        predecessor_name=predecessor_name,
                        successor_name=successor_name,
                        documentation=documentation,
                        docs_chain_length=predecessor_chain_length,
                        docs_model=before.get("predecessor_docs_model"),
                        docs_generated_at=before.get("predecessor_docs_generated_at"),
                        receipt_id=receipt_id,
                        document_identity=document_identity,
                        operation="semantics_preserving_rename",
                        reason=assertion_reason,
                        asserted_by=assertion_actor,
                    )
                )
                if len(mutation) != 1:
                    raise RuntimeError(
                        "rename documentation or its assertion receipt changed "
                        "during the transaction"
                    )
                tx.commit()
                result = dict(mutation[0])
                result.update(
                    {
                        "predecessor_name": predecessor_name,
                        "successor_name": successor_name,
                        "predecessor_document_identity": document_identity,
                        "predecessor_document_sha256": document_sha256,
                        "receipt_id": receipt_id,
                        "reason": assertion_reason,
                        "asserted_by": assertion_actor,
                    }
                )
                return result
            except BaseException:
                with suppress(Exception):
                    tx.rollback()
                raise


_GENERATED_SUPERSESSION_PREFLIGHT_QUERY = """
// GENERATED_SUPERSESSION_PREFLIGHT
UNWIND $pairs AS pr
OPTIONAL MATCH (src:IMASNode {id: pr.source_id})
OPTIONAL MATCH (new:StandardName {id: pr.new_name})
// The recomposed source itself, resolved through its DD edge rather than by
// rebuilding the composite id. One DD path can anchor several sources — a
// renamed path keeps a source under its superseded spelling alongside the
// current one — so the successor binding this pass just wrote is what singles
// the recomposed source out of them. Without it every alias reaches the
// predecessor, none of them can be told apart, and the batch aborts.
OPTIONAL MATCH (trigger:StandardNameSource)-[:FROM_DD_PATH]->(src)
WHERE EXISTS { (trigger)-[:PRODUCED_NAME]->(new) }
OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME]->(old:StandardName)
WHERE old.id <> pr.new_name
  AND NOT coalesce(old.name_stage, '') IN
      ['superseded', 'exhausted', 'contested', 'approved']
  AND old.catalog_pr_number IS NULL
  AND coalesce(old.origin, 'pipeline') <> 'derived'
// Skip self and any case where old already descends from new
// along the REFINED_FROM chain (would form a cycle).
  AND new.id <> old.id
  AND NOT (old)-[:REFINED_FROM*1..]->(new)
WITH pr, src, new, trigger, old
OPTIONAL MATCH (source:StandardNameSource)
WHERE old IS NOT NULL
  AND ((source)-[:PRODUCED_NAME]->(old)
       OR source.produced_sn_id = old.id
       OR (source)-[:PRODUCED_NAME]->(new))
// A live successor edge is what a composer emits, so it is the one fact that
// says this pairing was judged. Everything else reaches the predecessor only
// through the predecessor.
WITH pr, src, new, trigger, old, source,
     EXISTS { (source)-[:PRODUCED_NAME]->(new) } AS judged
RETURN pr.source_id AS requested_source_id,
       pr.new_name AS new_name,
       src IS NOT NULL AS requested_source_exists,
       new IS NOT NULL AS successor_exists,
       trigger.id AS trigger_source_id,
       old.id AS old_name,
       old.name_stage AS old_stage,
       [source_id IN collect(DISTINCT
          CASE WHEN judged THEN source.id END)
        WHERE source_id IS NOT NULL] AS judged_source_ids,
       [source_id IN collect(DISTINCT
          CASE WHEN judged THEN null ELSE source.id END)
        WHERE source_id IS NOT NULL] AS retained_source_ids
"""


def supersede_prior_source_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.supersede_prior_source_names."""
    from .graph_store import graph_supersession

    return graph_supersession.supersede_prior_source_names(*args, **kwargs)


def tombstone_supersede_into(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.tombstone_supersede_into."""
    from .graph_store import graph_supersession

    return graph_supersession.tombstone_supersede_into(*args, **kwargs)


SCALAR_SELECTED_DEDUP_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SCALAR_SELECTED_DEDUP_MANIFEST_SCHEMA
SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA


ScalarSelectedDedupConflict = import_module(
    ".graph_store.graph_supersession", __package__
).ScalarSelectedDedupConflict


def _scalar_selected_dedup_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._scalar_selected_dedup_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._scalar_selected_dedup_authority(*args, **kwargs)


def _lock_scalar_selected_dedup_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._lock_scalar_selected_dedup_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._lock_scalar_selected_dedup_authority(*args, **kwargs)


def _scalar_selected_dedup_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._scalar_selected_dedup_counts."""
    from .graph_store import graph_supersession

    return graph_supersession._scalar_selected_dedup_counts(*args, **kwargs)


def deduplicate_scalar_selected_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.deduplicate_scalar_selected_sources."""
    from .graph_store import graph_supersession

    return graph_supersession.deduplicate_scalar_selected_sources(*args, **kwargs)


SIGNED_SOURCE_DISPOSITION_ADJUDICATION_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_SOURCE_DISPOSITION_ADJUDICATION_SCHEMA
SIGNED_SOURCE_DISPOSITION_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_SOURCE_DISPOSITION_MANIFEST_SCHEMA
SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA
_SIGNED_SOURCE_DISPOSITIONS = import_module(
    ".graph_store.graph_supersession", __package__
)._SIGNED_SOURCE_DISPOSITIONS
_STRUCTURAL_LEGITIMACY_AUTHORITY_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
)._STRUCTURAL_LEGITIMACY_AUTHORITY_SCHEMA
_STRUCTURAL_LEGITIMACY_DISPOSITION = import_module(
    ".graph_store.graph_supersession", __package__
)._STRUCTURAL_LEGITIMACY_DISPOSITION
_ORPHAN_DISPOSITIONS = import_module(
    ".graph_store.graph_supersession", __package__
)._ORPHAN_DISPOSITIONS


SignedSourceDispositionConflict = import_module(
    ".graph_store.graph_supersession", __package__
).SignedSourceDispositionConflict


def _catalog_edit_adjudication_signature_hash(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._catalog_edit_adjudication_signature_hash."""
    from .graph_store import graph_supersession

    return graph_supersession._catalog_edit_adjudication_signature_hash(*args, **kwargs)


def _validate_signed_source_adjudication(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._validate_signed_source_adjudication."""
    from .graph_store import graph_supersession

    return graph_supersession._validate_signed_source_adjudication(*args, **kwargs)


def _validate_structural_legitimacy_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._validate_structural_legitimacy_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._validate_structural_legitimacy_authority(*args, **kwargs)


def _is_live_structural_child(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._is_live_structural_child."""
    from .graph_store import graph_supersession

    return graph_supersession._is_live_structural_child(*args, **kwargs)


def _signed_source_disposition_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_source_disposition_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_source_disposition_authority(*args, **kwargs)


def _lock_signed_source_disposition_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._lock_signed_source_disposition_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._lock_signed_source_disposition_authority(*args, **kwargs)


def _signed_source_disposition_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_source_disposition_counts."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_source_disposition_counts(*args, **kwargs)


def _signed_source_disposition_execution_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_source_disposition_execution_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_source_disposition_execution_authority(
        *args, **kwargs
    )


_catalog_source_disposition_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._catalog_source_disposition_apply
_catalog_source_disposition_signature = import_module(
    ".graph_store.graph_supersession", __package__
)._catalog_source_disposition_signature
_catalog_source_disposition_parameters = import_module(
    ".graph_store.graph_supersession", __package__
)._catalog_source_disposition_parameters
_catalog_source_disposition_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._catalog_source_disposition_apply

globals()["apply_" + "adjudicated_source_dispositions"] = (
    _catalog_source_disposition_apply
)
del _catalog_source_disposition_signature
del _catalog_source_disposition_parameters

SEMANTIC_MIRROR_REPAIR_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SEMANTIC_MIRROR_REPAIR_MANIFEST_SCHEMA
SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA
_PROJECTED_SEMANTIC_SOURCE_TYPES = import_module(
    ".graph_store.graph_supersession", __package__
)._PROJECTED_SEMANTIC_SOURCE_TYPES


SemanticMirrorRepairConflict = import_module(
    ".graph_store.graph_supersession", __package__
).SemanticMirrorRepairConflict


def _semantic_mirror_repair_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._semantic_mirror_repair_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._semantic_mirror_repair_authority(*args, **kwargs)


def _lock_semantic_mirror_repair_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._lock_semantic_mirror_repair_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._lock_semantic_mirror_repair_authority(*args, **kwargs)


def _semantic_mirror_repair_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._semantic_mirror_repair_counts."""
    from .graph_store import graph_supersession

    return graph_supersession._semantic_mirror_repair_counts(*args, **kwargs)


_semantic_mirror_repair_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._semantic_mirror_repair_apply
_semantic_mirror_repair_signature = import_module(
    ".graph_store.graph_supersession", __package__
)._semantic_mirror_repair_signature
_semantic_mirror_repair_parameters = import_module(
    ".graph_store.graph_supersession", __package__
)._semantic_mirror_repair_parameters
_semantic_mirror_repair_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._semantic_mirror_repair_apply

globals()["repair_" + "scalar_projection_mismatches"] = _semantic_mirror_repair_apply
del _semantic_mirror_repair_signature
del _semantic_mirror_repair_parameters

_ineligible_source_retirement_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._ineligible_source_retirement_apply
_ineligible_source_retirement_signature = import_module(
    ".graph_store.graph_supersession", __package__
)._ineligible_source_retirement_signature
_ineligible_source_retirement_parameters = import_module(
    ".graph_store.graph_supersession", __package__
)._ineligible_source_retirement_parameters
_ineligible_source_retirement_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._ineligible_source_retirement_apply

globals()["retire_" + "ineligible_standard_name_sources"] = (
    _ineligible_source_retirement_apply
)
del _ineligible_source_retirement_signature
del _ineligible_source_retirement_parameters


_SIGNED_ORPHAN_RETIREMENT_DISPOSITION = import_module(
    ".graph_store.graph_supersession", __package__
)._SIGNED_ORPHAN_RETIREMENT_DISPOSITION


def _validate_signed_orphan_retirement_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._validate_signed_orphan_retirement_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._validate_signed_orphan_retirement_authority(
        *args, **kwargs
    )


SIGNED_DUAL_AUTHORITY_RETIREMENT_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_DUAL_AUTHORITY_RETIREMENT_MANIFEST_SCHEMA
SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA


SignedDualAuthorityRetirementConflict = import_module(
    ".graph_store.graph_supersession", __package__
).SignedDualAuthorityRetirementConflict


def _validate_signed_dual_authority_retirement(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._validate_signed_dual_authority_retirement."""
    from .graph_store import graph_supersession

    return graph_supersession._validate_signed_dual_authority_retirement(
        *args, **kwargs
    )


def _signed_dual_authority_retirement_manifest(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_dual_authority_retirement_manifest."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_dual_authority_retirement_manifest(
        *args, **kwargs
    )


def _signed_dual_authority_change_id(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_dual_authority_change_id."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_dual_authority_change_id(*args, **kwargs)


_dual_authority_retirement_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._dual_authority_retirement_apply
_dual_authority_hidden_parameters = import_module(
    ".graph_store.graph_supersession", __package__
)._dual_authority_hidden_parameters
_dual_authority_retirement_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._dual_authority_retirement_apply

globals()["retire_signed_" + "dual_authority_targets"] = (
    _dual_authority_retirement_apply
)
del _dual_authority_hidden_parameters

_refused_target_orphan_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._refused_target_orphan_apply

globals()["retire_signed_" + "provenance_orphans"] = _refused_target_orphan_apply
EXHAUSTED_ORPHAN_SUPERSESSION_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).EXHAUSTED_ORPHAN_SUPERSESSION_MANIFEST_SCHEMA
EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA


ExhaustedOrphanSupersessionConflict = import_module(
    ".graph_store.graph_supersession", __package__
).ExhaustedOrphanSupersessionConflict


def classify_sourceless_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.classify_sourceless_standard_names."""
    from .graph_store import graph_supersession

    return graph_supersession.classify_sourceless_standard_names(*args, **kwargs)


def _exhausted_orphan_supersession_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._exhausted_orphan_supersession_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._exhausted_orphan_supersession_authority(*args, **kwargs)


def _exhausted_orphan_change_id(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._exhausted_orphan_change_id."""
    from .graph_store import graph_supersession

    return graph_supersession._exhausted_orphan_change_id(*args, **kwargs)


def supersede_exhausted_standard_name_orphans(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.supersede_exhausted_standard_name_orphans."""
    from .graph_store import graph_supersession

    return graph_supersession.supersede_exhausted_standard_name_orphans(*args, **kwargs)


ANCESTOR_SUPERSESSION_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).ANCESTOR_SUPERSESSION_MANIFEST_SCHEMA
ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA


SupersedeIntoAncestorConflict = import_module(
    ".graph_store.graph_supersession", __package__
).SupersedeIntoAncestorConflict


def _ancestor_supersession_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._ancestor_supersession_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._ancestor_supersession_authority(*args, **kwargs)


def _lock_ancestor_supersession_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._lock_ancestor_supersession_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._lock_ancestor_supersession_authority(*args, **kwargs)


def _detach_stale_ancestor_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._detach_stale_ancestor_sources."""
    from .graph_store import graph_supersession

    return graph_supersession._detach_stale_ancestor_sources(*args, **kwargs)


def supersede_into_ancestor(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.supersede_into_ancestor."""
    from .graph_store import graph_supersession

    return graph_supersession.supersede_into_ancestor(*args, **kwargs)


# =============================================================================
# Release helpers — seed-and-expand pools
# =============================================================================
#
# These are called by pool_loop when process() raises an exception so that
# claimed items are unlocked and become eligible for other workers.
#
# Every helper verifies ``claim_token`` (and ``expected_stage`` where
# applicable) in the WHERE clause before clearing claim state.  This
# prevents a late-arriving release from clobbering a fresh re-claim that
# was issued after the orphan sweep cleared the stale token.
#
# Return value: count of nodes actually released.  A return value less than
# len(ids) indicates concurrent intervention (orphan sweep or another
# worker) — callers may ignore this; it is logged at DEBUG.
#


def release_generate_name_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_generate_name_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_generate_name_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_generate_name_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_generate_name_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_generate_name_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_enrich_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_enrich_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_enrich_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_enrich_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_enrich_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_enrich_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_review_names_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_names_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_names_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_review_names_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_names_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_names_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_review_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_docs_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_docs_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_review_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_docs_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_docs_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


REFINE_CLAIM_RELEASE_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_claim_release", __package__
).REFINE_CLAIM_RELEASE_MANIFEST_SCHEMA

REFINE_CLAIM_RELEASE_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_claim_release", __package__
).REFINE_CLAIM_RELEASE_RECEIPT_SCHEMA

REFINE_CLAIM_RELEASE_PROJECTION_VERSION = import_module(
    ".graph_store.graph_claim_release", __package__
).REFINE_CLAIM_RELEASE_PROJECTION_VERSION


RefineClaimReleaseConflict = import_module(
    ".graph_store.graph_claim_release", __package__
).RefineClaimReleaseConflict


_LOCK_REFINE_CLAIM_RELEASE_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
)._LOCK_REFINE_CLAIM_RELEASE_QUERY


_READ_REFINE_CLAIM_RELEASE_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
)._READ_REFINE_CLAIM_RELEASE_QUERY


_WRITE_REFINE_CLAIM_RELEASE_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
)._WRITE_REFINE_CLAIM_RELEASE_QUERY


def _sorted_refine_claim_projection(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._sorted_refine_claim_projection."""
    from .graph_store import graph_claim_release

    return graph_claim_release._sorted_refine_claim_projection(
        *args, services=_graph_foundation.services, **kwargs
    )


def _project_refine_claim_release_state(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._project_refine_claim_release_state."""
    from .graph_store import graph_claim_release

    return graph_claim_release._project_refine_claim_release_state(
        *args, services=_graph_foundation.services, **kwargs
    )


def _predicted_refine_claim_release_state(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._predicted_refine_claim_release_state."""
    from .graph_store import graph_claim_release

    return graph_claim_release._predicted_refine_claim_release_state(
        *args, services=_graph_foundation.services, **kwargs
    )


def _refine_claim_release_manifest(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._refine_claim_release_manifest."""
    from .graph_store import graph_claim_release

    return graph_claim_release._refine_claim_release_manifest(
        *args, services=_graph_foundation.services, **kwargs
    )


def _refine_claim_no_change_receipt(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._refine_claim_no_change_receipt."""
    from .graph_store import graph_claim_release

    return graph_claim_release._refine_claim_no_change_receipt(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_stranded_refine_claim(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_stranded_refine_claim."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_stranded_refine_claim(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_refine_name_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_refine_name_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_refine_name_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def resubmit_pinned_rename_for_review(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.resubmit_pinned_rename_for_review."""
    from .graph_store import graph_claim_release

    return graph_claim_release.resubmit_pinned_rename_for_review(
        *args, services=_graph_foundation.services, **kwargs
    )


RESCORE_ELIGIBILITY_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
).RESCORE_ELIGIBILITY_QUERY


def stage_name_for_rescore(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.stage_name_for_rescore."""
    from .graph_store import graph_claim_release

    return graph_claim_release.stage_name_for_rescore(
        *args, services=_graph_foundation.services, **kwargs
    )


def restore_name_after_failed_rescore(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.restore_name_after_failed_rescore."""
    from .graph_store import graph_claim_release

    return graph_claim_release.restore_name_after_failed_rescore(
        *args, services=_graph_foundation.services, **kwargs
    )


#: Stop reasons proving that no further attempt on this name can succeed.
#: A collision names an identity the graph already holds, and refinement may
#: not take an occupied identity — folding into it carries source-migration
#: semantics that belong to ``sn edit``. A grammar or vocabulary failure is
#: reproduced by the same prompt every cycle. Measured over one run, 830 of
#: 830 collisions re-proposed the same identity, across two model vendors.
_TERMINAL_REFINE_STOP_REASONS = import_module(
    ".graph_store.graph_claim_release", __package__
)._TERMINAL_REFINE_STOP_REASONS


def stop_refine_name_attempt(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.stop_refine_name_attempt."""
    from .graph_store import graph_claim_release

    return graph_claim_release.stop_refine_name_attempt(
        *args, services=_graph_foundation.services, **kwargs
    )


def _mark_refine_docs_exhausted(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._mark_refine_docs_exhausted."""
    from .graph_store import graph_claim_release

    return graph_claim_release._mark_refine_docs_exhausted(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Docs-claim race resolution
# =============================================================================
# The shared _claim_sn_atomic seed binds its candidate row at MATCH time, then
# acquires the node write-lock at the SET. When two pool replicas race for the
# same drafted/pending node, both MATCH it as eligible; the SET serialises on
# the lock, so the LAST committer's claim_token / claim_seq wins — but BOTH
# replicas returned the node as "claimed" and proceeded to call the LLM. That
# phantom is the dominant docs-cost driver: a single accepted name was
# re-reviewed 27× (≈48 wasted reviewer LLM calls) purely because losing-race
# replicas still fired their reviewer quorum.
#
# _verify_docs_claim_winners closes the window by re-reading COMMITTED state
# after a short settle: each claim stamps a strictly-increasing per-node
# claim_seq, so a node is a genuine win only if it STILL holds our claim_token
# AND its committed claim_seq equals the one we were assigned — i.e. no later
# racer superseded us. The settle lets the lock-serialised claim burst finish
# committing before we read, so exactly one replica (the final committer) sees
# its own claim_seq as current; all earlier racers observe a higher seq and
# self-exclude before any LLM call. This is the generate_docs / review_docs
# analogue of the claim_token verify mandated for all claim functions, hardened
# against the "every racer is momentarily the token holder" false-positive.


def _verify_docs_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_docs_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_docs_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


def _verify_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Name-claim race resolution
# =============================================================================
# Identical mechanism to the docs axis (see _verify_docs_claim_winners): two
# pool replicas both bind the same eligible StandardName at the shared seed
# MATCH, the lock-serialised SET lets the LAST claim_token / claim_seq win, but
# BOTH replicas already returned the node and proceed to the (paid) name LLM
# call. Live evidence: review_name ran 339 reviewer calls for 64 distinct names
# (~5× amplification) on a multi-replica run. _verify_name_claim_winners closes
# the window via the same settle + claim_seq check as the docs axis: after the
# claim burst settles, a node is a genuine win only if it STILL holds our
# claim_token at the eligible name_stage with our assigned claim_seq; every
# earlier racer sees a higher committed seq and is dropped before any LLM call.


def _verify_name_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_name_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_name_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


def _verify_source_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_source_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_source_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# generate_docs — claim / persist / release
# =============================================================================
# Stage gate: name_stage='accepted' AND docs_stage='pending'
# The claim does NOT transition docs_stage (persist_generated_docs does that).


def claim_generate_docs_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.claim_generate_docs_batch."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.claim_generate_docs_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_generated_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.persist_generated_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.persist_generated_docs(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_generate_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_generate_docs_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_generate_docs_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_generate_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_generate_docs_failed_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_generate_docs_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# refine_docs — claim / persist / release (DocsRevision snapshot architecture)
# =============================================================================
# Stage gate: docs_stage='reviewed' AND (score below threshold OR open docs hold)
#             AND docs_chain_length < rotation_cap
# Fundamentally different from refine_name: docs refine is IN-PLACE on the
# existing SN node.  The OLD docs are snapshotted into a DocsRevision node
# linked via DOCS_REVISION_OF before the SN is updated.


def claim_refine_docs_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.claim_refine_docs_batch."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.claim_refine_docs_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Persist — refine_docs (snapshot current docs → DocsRevision, update SN)
# =============================================================================


def persist_refined_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.persist_refined_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.persist_refined_docs(
        *args, services=_graph_foundation.services, **kwargs
    )


def reset_standard_name_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.reset_standard_name_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.reset_standard_name_docs(
        *args, services=_graph_foundation.services, **kwargs
    )


def stamp_harmonized_families(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.stamp_harmonized_families."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.stamp_harmonized_families(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_refine_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_refine_docs_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_refine_docs_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_refine_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_refine_docs_failed_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_refine_docs_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Routing helper: desc-name similarity gate → REFINE_DOCS
# =============================================================================


def mark_for_refine_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.mark_for_refine_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.mark_for_refine_docs(
        *args, services=_graph_foundation.services, **kwargs
    )


@retry_on_deadlock()
def release_all_orphan_claims() -> dict[str, int]:
    """Release all claimed-but-unreleased StandardName and StandardNameSource nodes.

    Called from the ``run_sn_pools`` finally block after clean shutdown so
    that any batch still marked as claimed at process-exit is unlocked.
    Per-batch release already happens inside each process() try/finally, but
    batches in flight at the 60s grace-period timeout leave ``claimed_at``
    set permanently.  This sweep clears them unconditionally — it is safe
    because the run is over and no other process can be competing for the
    same tokens.

    A StandardName caught mid-refine sits at a transient ``'refining'`` stage.
    Clearing only its claim would strand it there — reviewable/refinable only
    after the periodic orphan sweep reverts the stage. Apply the same
    ``refining -> reviewed`` revert the orphan sweep uses (see
    ``orphan_sweep._SWEEP_QUERIES``) as part of the release, so shutdown leaves
    no node in a transient stage with a cleared claim.

    Returns a dict with ``"sn"`` and ``"sns"`` keys showing the counts
    of released nodes.
    """
    with GraphClient() as gc:
        sn_result = gc.query(
            """
            MATCH (n:StandardName)
            WHERE n.claimed_at IS NOT NULL
            SET n.updated_at = datetime(), n.name_stage = CASE WHEN n.name_stage = 'refining'
                                    THEN 'reviewed' ELSE n.name_stage END,
                n.docs_stage = CASE WHEN n.docs_stage = 'refining'
                                    THEN 'reviewed' ELSE n.docs_stage END,
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """
        )
        sns_result = gc.query(
            """
            MATCH (n:StandardNameSource)
            WHERE n.claimed_at IS NOT NULL
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """
        )
    sn_count = sn_result[0]["released"] if sn_result else 0
    sns_count = sns_result[0]["released"] if sns_result else 0
    return {"sn": sn_count, "sns": sns_count}


# ═══════════════════════════════════════════════════════════════════════
# Pool pending counts — single round-trip query for all 6 pools
# ═══════════════════════════════════════════════════════════════════════


def pool_pending_counts(
    *,
    min_score: float = 0.75,
    rotation_cap: int = 3,
    domain: str | None = None,
    scope_run_id: str | None = None,
    edits_only: bool = False,
) -> dict[str, int]:
    """Return pending-work counts for all six worker pools in one query.

    The six predicates mirror the eligibility criteria and optional scope in
    the corresponding ``claim_*_batch`` functions but do NOT filter on
    ``claimed_at`` — the counts reflect total eligible work,
    including items currently being processed.  This is correct for
    both throttle decisions (total queue depth matters) and display
    (users want to see total pending, not just unclaimed).

    Returns a dict keyed by pool name::

        {
            "generate_name": int,
            "review_name": int,
            "refine_name": int,
            "generate_docs": int,
            "review_docs": int,
            "refine_docs": int,
        }
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    source_scope = ""
    name_scope = ""
    params: dict[str, Any] = {
        "min_score": min_score,
        "rotation_cap": rotation_cap,
        "max_compose_attempts": _MAX_COMPOSE_CLAIM_ATTEMPTS,
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    }
    if domain:
        source_scope += " AND s.physics_domain = $domain"
        name_scope += " AND sn.physics_domain = $domain"
        params["domain"] = domain
    if scope_run_id:
        source_scope += " AND s.run_id = $scope_run_id"
        name_scope += " AND sn.run_id = $scope_run_id"
        params["scope_run_id"] = scope_run_id
    if edits_only:
        # Name-generation claims intentionally have no edits-only source path:
        # rename/docs edits start from an already-composed StandardName.
        source_scope += " AND false"
        name_scope += " AND coalesce(sn.edit_status, '') = 'open'"

    score_gate = (
        ""
        if (scope_run_id or edits_only)
        else (
            " AND (sn.reviewer_score_name IS NOT NULL"
            " OR (coalesce(sn.origin, '') = 'derived'"
            " AND EXISTS { MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)"
            " WHERE NOT (coalesce(kid.name_stage, '') IN"
            " ['superseded', 'exhausted', 'contested']) }))"
        )
    )

    cypher = f"""
    CALL {{ MATCH (s:StandardNameSource {{status: 'extracted'}})
           WHERE coalesce(s.attempt_count, 0) < $max_compose_attempts
             {source_scope}
           RETURN count(s) AS generate_name }}
    CALL {{ MATCH (sn:StandardName {{name_stage: 'drafted'}})
           WHERE sn.description IS NOT NULL
             AND sn.description <> $parent_desc_placeholder
             AND coalesce(sn.origin, '') <> 'derived'
             {name_scope}
           RETURN count(sn) AS review_name }}
    CALL {{ MATCH (sn:StandardName {{name_stage: 'reviewed'}})
           WHERE sn.reviewer_score_name < $min_score
             AND coalesce(sn.chain_length, 0) < $rotation_cap
             AND coalesce(sn.origin, '') <> 'derived'
             AND NOT (coalesce(sn.edit_mode, '') = 'rename'
                      AND coalesce(sn.review_resubmit_count, 0)
                          >= $rotation_cap)
             {name_scope}
           RETURN count(sn) AS refine_name }}
    CALL {{ MATCH (sn:StandardName
                   {{name_stage: 'accepted', docs_stage: 'pending'}})
           WHERE true {score_gate}
             {name_scope}
           RETURN count(sn) AS generate_docs }}
    CALL {{ MATCH (sn:StandardName {{docs_stage: 'drafted'}})
           WHERE NOT (sn.name_stage IN
                      ['superseded', 'exhausted', 'contested'])
             {score_gate}
             {name_scope}
           RETURN count(sn) AS review_docs }}
    CALL {{ MATCH (sn:StandardName {{docs_stage: 'reviewed'}})
           WHERE sn.reviewer_score_docs < $min_score
             AND coalesce(sn.docs_chain_length, 0) < $rotation_cap
             AND NOT (sn.name_stage IN
                      ['superseded', 'exhausted', 'contested'])
             {score_gate}
             {name_scope}
           RETURN count(sn) AS refine_docs }}
    RETURN generate_name, review_name, refine_name,
           generate_docs, review_docs, refine_docs
    """
    with GraphClient() as gc:
        rows = list(gc.query(cypher, **params))

    if not rows:
        return {
            "generate_name": 0,
            "review_name": 0,
            "refine_name": 0,
            "generate_docs": 0,
            "review_docs": 0,
            "refine_docs": 0,
        }
    r = rows[0]
    return {
        "generate_name": int(r.get("generate_name", 0)),
        "review_name": int(r.get("review_name", 0)),
        "refine_name": int(r.get("refine_name", 0)),
        "generate_docs": int(r.get("generate_docs", 0)),
        "review_docs": int(r.get("review_docs", 0)),
        "refine_docs": int(r.get("refine_docs", 0)),
    }


def scoped_terminal_residue(
    *,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
) -> dict[str, Any]:
    """Return claim and transient-stage residue inside one exact run scope."""
    if bool(scope_run_id) == bool(drain_scope_id):
        raise ValueError("exactly one scoped terminal selector is required")
    scope_id = scope_run_id or drain_scope_id
    scope_property = "run_id" if scope_run_id else "drain_scope_id"
    cypher = f"""
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.{scope_property} = $scope_id
        AND (sn.claim_token IS NOT NULL
             OR sn.claimed_at IS NOT NULL
             OR sn.name_stage = 'refining'
             OR sn.docs_stage = 'refining')
      WITH sn ORDER BY sn.id
      RETURN count(sn) AS name_count,
             collect({{
               id: sn.id,
               name_stage: sn.name_stage,
               docs_stage: sn.docs_stage,
               claim_token: sn.claim_token,
               claimed_at: toString(sn.claimed_at)
             }})[0..50] AS names
    }}
    CALL {{
      MATCH (sns:StandardNameSource)
      WHERE sns.{scope_property} = $scope_id
        AND (sns.claim_token IS NOT NULL OR sns.claimed_at IS NOT NULL)
      WITH sns ORDER BY sns.id
      RETURN count(sns) AS source_count,
             collect({{
               id: sns.id,
               status: sns.status,
               claim_token: sns.claim_token,
               claimed_at: toString(sns.claimed_at)
             }})[0..50] AS sources
    }}
    RETURN name_count, source_count, names, sources
    """
    with GraphClient() as gc:
        rows = list(gc.query(cypher, scope_id=scope_id))
    row = dict(rows[0]) if rows else {}
    name_count = int(row.get("name_count") or 0)
    source_count = int(row.get("source_count") or 0)
    return {
        "total": name_count + source_count,
        "name_count": name_count,
        "source_count": source_count,
        "names": list(row.get("names") or []),
        "sources": list(row.get("sources") or []),
    }


# =============================================================================
# enrich_parents — claim / fetch-children / persist / release
# =============================================================================
# Coverage deadlock break: a derived parent materialised by
# ``_materialize_derived_parent_rows`` carries the
# ``DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER`` until something writes a real
# description.  The placeholder excludes it from REVIEW_NAME (the claim gate
# drops it), and with no reviewer_score_name it is excluded from generate_docs
# (``reviewer_score_name IS NOT NULL`` gate) — neither reviewable nor
# documentable.  ``enrich_parents`` synthesises a real description GENERALISED
# over the parent's accepted children, embeds it locally, and routes the parent
# to ``name_stage='drafted'`` so it flows review → accept → docs like any other
# name.  Childless derived parents are legitimately unscoped and never claimed.


def _verify_enrich_parents_claim_winners(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop enrich-parents claim items that lost a concurrent-claim race.

    Same two-step claim_token verify mandated for every claim function (see
    :func:`_verify_name_claim_winners`): two replicas bind the same eligible
    derived parent at the shared seed MATCH, the lock-serialised SET lets the
    LAST claim_token win, but BOTH replicas returned the node and would proceed
    to the (paid) enrichment LLM call. This re-reads committed state and keeps
    only nodes that STILL hold our token AND remain enrichment-eligible (a
    placeholder derived parent) — the race loser's token was overwritten, so it
    is dropped before any LLM call.

    Returns the filtered item list (order preserved).
    """
    if not items:
        return items
    token = items[0].get("claim_token") or ""
    if not token:
        return items
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    ids = [it["id"] for it in items]
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
              AND sn.origin = 'derived'
              AND sn.description = $placeholder
            RETURN sn.id AS id
            """,
            ids=ids,
            token=token,
            placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
    winners = {r["id"] for r in rows}
    if len(winners) == len(items):
        return items
    logger.debug(
        "_verify_enrich_parents_claim_winners: %d/%d survived claim-race (token=%s)",
        len(winners),
        len(items),
        token[:8],
    )
    return [it for it in items if it["id"] in winners]


@retry_on_deadlock()
def claim_enrich_parents_batch(
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
) -> list[dict[str, Any]]:
    """Claim placeholder derived parents that still need a real description.

    Eligibility: ``origin = 'derived'`` AND the description is still the
    deterministic-parent placeholder AND the parent has at least one live
    (non-superseded/exhausted) ``HAS_PARENT`` child to ground on.  Childless
    placeholders (legitimately unscoped) never match and are skipped.

    The claim does NOT transition any stage — only ``claim_token`` /
    ``claimed_at`` are written.  The stage transition to ``'drafted'`` happens
    in :func:`persist_enriched_parent` once a real description + embedding
    exist, so a failed LLM call leaves the parent claimable as a placeholder.

    Returns claimed items as dicts with ``id``, ``name``, ``kind``, ``unit``,
    ``physics_domain``, ``claim_token``.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    where = (
        "sn.origin = 'derived'"
        " AND sn.description = $parent_desc_placeholder"
        " AND EXISTS { MATCH (child:StandardName)-[:HAS_PARENT]->(sn)"
        "   WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted', 'contested'] }"
    )
    query_params: dict[str, Any] = {
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    }
    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", coalesce(sn.name, sn.id) AS name",
        domain=domain,
        scope_run_id=scope_run_id,
        drain_scope_id=drain_scope_id,
        edits_only=edits_only,
    )
    # Two-step claim_token verify (mandated for every claim function): drop nodes
    # a concurrent replica won at the lock-serialised SET, before any enrichment
    # LLM call — otherwise multi-replica runs pay the enrichment cost N× per
    # parent (the docs/name amplification analogue).
    return _verify_enrich_parents_claim_winners(items)


def fetch_derived_parent_children(
    parent_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Fetch the grounding children for a batch of derived parents.

    For each parent id, returns its live (non-superseded/exhausted)
    ``HAS_PARENT`` children with the fields the enrichment prompt grounds on:
    the child standard name, its description, unit, and physics domain.  A
    child still carrying the placeholder description contributes its NAME (the
    name itself is meaningful) but an empty description, so the prompt grounds
    on whatever real descriptions exist plus the child name set.

    Returns ``{parent_id: [child_dict, ...]}``; parents with no live children
    map to an empty list.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    if not parent_ids:
        return {}
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $parent_ids AS pid
            MATCH (child:StandardName)-[:HAS_PARENT]->(p:StandardName {id: pid})
            WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted', 'contested']
            WITH pid, child
            ORDER BY child.id
            RETURN pid AS parent_id,
                   collect({
                     name: child.id,
                     description: CASE
                         WHEN child.description = $placeholder THEN null
                         ELSE child.description END,
                     unit: child.unit,
                     physics_domain: child.physics_domain,
                     kind: child.kind
                   }) AS children
            """,
            parent_ids=parent_ids,
            placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
    return {r["parent_id"]: list(r["children"]) for r in rows}


_STRUCTURAL_AUTHORITY_MECHANISM = "deterministic-grammar-peel"
_STRUCTURAL_AUTHORITY_SCHEMA = "StructuralNameAuthority"
_STRUCTURAL_AUTHORITY_CANONICALIZATION = "json-sort-keys"


class StructuralAuthorityConflict(RuntimeError):
    """A structural accept no longer matches its exact live child closure."""


def _structural_authority_identities() -> tuple[str, str]:
    """Return immutable identities for the executing code and schema bytes."""
    code_identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    schema_path = Path(__file__).parents[1] / "schemas" / "standard_name.yaml"
    schema_identity = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    return code_identity, schema_identity


def _structural_authority_snapshot(gc: Any, sn_id: str) -> dict[str, Any] | None:
    """Read one parent and its exact ordered live child rows."""
    rows = gc.query(
        """
        MATCH (parent:StandardName {id: $id})
        OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(parent)
        WHERE child IS NULL OR NOT coalesce(child.name_stage, '') IN
          ['superseded', 'exhausted', 'contested']
        WITH parent, child ORDER BY child.id
        RETURN parent.id AS parent_id,
               elementId(parent) AS parent_element_id,
               parent.name_stage AS name_stage,
               parent.origin AS origin,
               parent.claim_token AS claim_token,
               parent.reviewed_name_at AS reviewed_name_at,
               parent.docs_stage AS docs_stage,
               parent.validation_status AS validation_status,
               parent.embedded_at AS embedded_at,
               parent.chain_length AS chain_length,
               [(parent)-[:HAS_STRUCTURAL_AUTHORITY]->(existing)
                 | existing.id] AS authority_ids,
               [item IN collect(CASE WHEN child IS NULL THEN null ELSE {
                 id: child.id,
                 element_id: elementId(child),
                 name_stage: child.name_stage,
                 reviewer_score_name: child.reviewer_score_name,
                 reviewer_model_name: child.reviewer_model_name
               } END) WHERE item IS NOT NULL] AS children
        """,
        id=sn_id,
    )
    return dict(rows[0]) if rows else None


def _structural_authority_record(
    snapshot: dict[str, Any], *, accepting: bool
) -> dict[str, Any]:
    """Build a content-addressed signed authority from a live child snapshot."""
    child_rows = list(snapshot.get("children") or [])
    if not child_rows:
        raise StructuralAuthorityConflict(
            f"derived parent {snapshot.get('parent_id')!r} has no live children"
        )
    child_ids = [str(child["id"]) for child in child_rows]
    if child_ids != sorted(child_ids) or len(child_ids) != len(set(child_ids)):
        raise StructuralAuthorityConflict(
            "structural authority requires unique children in canonical id order"
        )

    parent_id = str(snapshot["parent_id"])
    created_at_value = datetime.now(UTC)
    created_at = created_at_value.isoformat()
    code_identity, schema_identity = _structural_authority_identities()
    signed_payload = {
        "accepted_name_id": parent_id,
        "child_rows": child_rows,
        "code_identity": code_identity,
        "created_at": created_at_value,
        "entailment_mechanism": _STRUCTURAL_AUTHORITY_MECHANISM,
        "operation": "accept-and-authorize"
        if accepting
        else "authorize-existing-accept",
        "schema_identity": schema_identity,
    }
    payload_sha256 = _authority_payload_hash(signed_payload)
    authority_id = f"structural-authority:{payload_sha256}"
    prefix = authority_id + ":"
    participants = [
        {
            "id": prefix + "participant:parent",
            "kind": "node",
            "graph_label": "StandardName",
            "signature_sha256": _authority_payload_hash(
                {
                    "id": parent_id,
                    "element_id": snapshot.get("parent_element_id"),
                    "name_stage": snapshot.get("name_stage"),
                }
            ),
        },
        *[
            {
                "id": prefix + f"participant:child:{index}",
                "kind": "node",
                "graph_label": "StandardName",
                "signature_sha256": _authority_payload_hash(child),
            }
            for index, child in enumerate(child_rows)
        ],
    ]
    mutations = [
        {
            "id": prefix + "mutation:authority",
            "order": 0,
            "kind": "add_relationship",
            "participant_id": participants[0]["id"],
            "arguments": json.dumps(
                {
                    "relationship": "HAS_STRUCTURAL_AUTHORITY",
                    "child_relationship": "ENTAILED_FROM_CHILD",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    ]
    if accepting:
        mutations.append(
            {
                "id": prefix + "mutation:accept",
                "order": 1,
                "kind": "set_properties",
                "participant_id": participants[0]["id"],
                "arguments": json.dumps(
                    {"name_stage": "accepted"},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    guards = [
        {
            "id": prefix + "guard:participants",
            "kind": "participant_lock",
            "implementation": "lock-exact-parent-and-live-children",
            "participant_ids": [participant["id"] for participant in participants],
            "expected": len(participants),
        },
        {
            "id": prefix + "guard:children",
            "kind": "closure_digest",
            "implementation": "ordered-live-child-row-digest",
            "participant_ids": [participant["id"] for participant in participants[1:]],
            "expected": payload_sha256,
        },
        {
            "id": prefix + "guard:replay",
            "kind": "replay_postcondition",
            "implementation": "one-structural-authority-per-accept",
            "participant_ids": [participants[0]["id"]],
            "expected": 1,
        },
    ]
    return {
        "id": authority_id,
        "child_rows": child_rows,
        "expected_name_stage": snapshot.get("name_stage"),
        "expected_claim_token": snapshot.get("claim_token"),
        "identity": {
            "id": prefix + "identity",
            "kind": "standard_name",
            "target_id": parent_id,
        },
        "signature": {
            "id": prefix + "signature",
            "profile": "each_row",
            "sha256": payload_sha256,
            "canonicalization": _STRUCTURAL_AUTHORITY_CANONICALIZATION,
        },
        "participants": participants,
        "selection": {
            "id": prefix + "selection",
            "mode": "exact_caller_set",
            "predicate": "exact-live-derived-parent-child-set",
        },
        "mutations": mutations,
        "guards": guards,
        "orphan_policy": "refuse",
        "accepted_name_id": parent_id,
        "child_ids": child_ids,
        "children": child_ids,
        "entailment_mechanism": _STRUCTURAL_AUTHORITY_MECHANISM,
        "created_at": created_at,
        "code_identity": code_identity,
        "schema_identity": schema_identity,
        "payload_sha256": payload_sha256,
    }


def _persist_structural_authority(
    gc: Any,
    record: dict[str, Any],
    *,
    parent_updates: dict[str, Any] | None = None,
    require_derived: bool = True,
    require_source_free: bool = False,
) -> bool:
    """Persist one authority and optional accept transition in one statement.

    ``require_source_free`` re-checks inside the same guarded statement that no
    ``StandardNameSource`` produces the parent. A caller that admits a parent on
    the strength of it having no producer must hold that fact through the write,
    or a source attached in between would be accepted without its name reviewed.
    """
    rows = gc.query(
        """
        MATCH (parent:StandardName {id: $parent_id})
        WHERE ($require_derived = false OR parent.origin = 'derived')
          AND ($require_source_free = false OR NOT EXISTS {
                MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(parent) })
          AND parent.name_stage = $expected_name_stage
          AND (parent.claim_token = $expected_claim_token
               OR (parent.claim_token IS NULL AND $expected_claim_token IS NULL))
          AND NOT (parent)-[:HAS_STRUCTURAL_AUTHORITY]->(:StructuralNameAuthority)
        OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(parent)
        WHERE child IS NULL OR NOT coalesce(child.name_stage, '') IN
          ['superseded', 'exhausted', 'contested']
        WITH parent, child ORDER BY child.id
        WITH parent, [c IN collect(child) WHERE c IS NOT NULL] AS children
        WHERE [c IN children | c.id] = $child_ids
          AND [c IN children | elementId(c)] = $child_element_ids
          AND size(children) > 0
        SET parent += $parent_updates,
            parent.updated_at = datetime(),
            parent._structural_authority_lock = true
        REMOVE parent._structural_authority_lock
        CREATE (authority:StructuralNameAuthority)
        SET authority = $authority_properties
        CREATE (parent)-[:HAS_STRUCTURAL_AUTHORITY]->(authority)
        CREATE (identity:RepairRowIdentity)
        SET identity = $identity
        CREATE (authority)-[:IDENTITY]->(identity)
        CREATE (signature:RepairAuthorityDigest)
        SET signature = $signature
        CREATE (authority)-[:SIGNATURES]->(signature)
        CREATE (selection:RepairSelection)
        SET selection = $selection
        CREATE (authority)-[:SELECTION]->(selection)
        WITH parent, authority, children
        UNWIND $participants AS participant_data
        CREATE (participant:RepairParticipant)
        SET participant = participant_data
        CREATE (authority)-[:PARTICIPANTS]->(participant)
        WITH DISTINCT parent, authority, children
        UNWIND $mutations AS mutation_data
        CREATE (mutation:RepairMutation)
        SET mutation = mutation_data
        CREATE (authority)-[:MUTATIONS]->(mutation)
        WITH DISTINCT parent, authority, children
        UNWIND $guards AS guard_data
        CREATE (guard:RepairGuard)
        SET guard = guard_data
        CREATE (authority)-[:GUARDS]->(guard)
        WITH DISTINCT parent, authority, children
        UNWIND children AS entailing_child
        CREATE (authority)-[:ENTAILED_FROM_CHILD]->(entailing_child)
        WITH DISTINCT parent, authority
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(parent)
        FOREACH (_ IN CASE WHEN $description IS NULL OR source IS NULL
                           THEN [] ELSE [1] END |
          SET source.description = $description)
        RETURN DISTINCT parent.name_stage AS name_stage,
                        authority.id AS authority_id
        """,
        parent_id=record["accepted_name_id"],
        child_ids=record["child_ids"],
        child_element_ids=[child["element_id"] for child in record["child_rows"]],
        child_rows=record["child_rows"],
        expected_name_stage=record["expected_name_stage"],
        expected_claim_token=record["expected_claim_token"],
        require_derived=require_derived,
        require_source_free=require_source_free,
        parent_updates=parent_updates or {},
        authority_properties={
            key: record[key]
            for key in (
                "id",
                "orphan_policy",
                "accepted_name_id",
                "child_ids",
                "children",
                "entailment_mechanism",
                "created_at",
                "code_identity",
                "schema_identity",
            )
        }
        | {
            "identity": record["identity"]["id"],
            "signatures": [record["signature"]["id"]],
            "participants": [item["id"] for item in record["participants"]],
            "selection": record["selection"]["id"],
            "mutations": [item["id"] for item in record["mutations"]],
            "guards": [item["id"] for item in record["guards"]],
        },
        identity=record["identity"],
        signature=record["signature"],
        selection=record["selection"],
        participants=record["participants"],
        mutations=record["mutations"],
        guards=record["guards"],
        description=(parent_updates or {}).get("description"),
    )
    returned_authority_ids = {row.get("authority_id") for row in rows}
    return returned_authority_ids == {record["id"]}


@retry_on_deadlock()
def persist_enriched_parent(
    *,
    sn_id: str,
    claim_token: str,
    description: str,
    embedding: list[float] | None,
    model: str,
    run_id: str | None = None,
) -> str:
    """Persist a synthesised derived-parent description and accept structurally.

    A derived parent's NAME is a deterministic grammar peel that already passed
    the two-clause admission gate, and its DESCRIPTION is a faithful
    generalisation over its already-RD-quorum-accepted children — so it inherits
    name validity by construction.  It therefore SKIPS REVIEW_NAME: routing a
    structurally-fixed abstraction through a name quorum that systematically
    penalises it for being less specific than its children only sends it to a
    futile refine→exhaust (measured: ~66% of derived parents scored <0.85,
    mean 0.778, despite clean grammar).  The description's quality is still
    gated downstream on the DOCS axis (generate_docs + review_docs).

    Single write:

    1. Verify the claim (token match OR cleared by the orphan sweep) and that
       the node is still a derived parent.
    2. SET ``description`` + ``embedding`` (+ ``embedded_at``), record the
       enrichment model/time, and transition ``name_stage`` directly to
       ``'accepted'``. The name is NOT scored — a derived parent name is a
       deterministic grammar peel that is never reviewed/refined, so a
       ``reviewer_score_name`` would be meaningless. ``reviewer_model_name`` is
       stamped ``'structural-inheritance'`` (audit marker) and ``reviewed_name_at``
       is set as the "name finalised, docs may proceed" signal. ``docs_stage``
       defaults to ``'pending'`` so it becomes docs-eligible immediately (the
       docs gates admit derived parents via ``origin``, not a name score).
    3. Mirror the description onto the parent's ``StandardNameSource`` so
       consolidation / export see the real text.

    Returns the new ``name_stage`` (``'accepted'``), or ``''`` when the node no
    longer matched (concurrent winner / not a derived parent).
    """
    description = _sanitize_doc_text(description)
    with GraphClient() as gc:
        snapshot = _structural_authority_snapshot(gc, sn_id)
        if snapshot and "parent_id" not in snapshot:
            return str(snapshot.get("name_stage") or "")
        if (
            snapshot is None
            or snapshot.get("origin") != "derived"
            or snapshot.get("claim_token") not in (claim_token, None)
            or not snapshot.get("children")
            or snapshot.get("authority_ids")
        ):
            logger.debug(
                "persist_enriched_parent: %s no longer matched (concurrent "
                "winner, missing child authority, or not a derived parent) — no-op",
                sn_id,
            )
            return ""
        record = _structural_authority_record(snapshot, accepting=True)
        now = datetime.now(UTC)
        applied = _persist_structural_authority(
            gc,
            record,
            parent_updates={
                "description": description,
                "embedding": embedding,
                "embedded_at": (
                    now if embedding is not None else snapshot.get("embedded_at")
                ),
                "parent_enriched_at": now,
                "parent_enrich_model": model,
                "name_stage": "accepted",
                "reviewer_model_name": "structural-inheritance",
                "reviewed_name_at": snapshot.get("reviewed_name_at") or now,
                "docs_stage": snapshot.get("docs_stage") or "pending",
                "validation_status": snapshot.get("validation_status") or "valid",
                "needs_composition": None,
                "claim_token": None,
                "claimed_at": None,
            },
        )
        if not applied:
            raise StructuralAuthorityConflict(
                f"derived parent {sn_id!r} changed before atomic acceptance"
            )

    new_stage = "accepted"
    logger.info(
        "\033[32menrich_parents\033[0m: %s → name_stage=%s — %s",
        sn_id,
        new_stage,
        (description or "")[:100],
    )
    # NB: deliberately no bump_sn_run_counter here — the SNRun ``names_enriched``
    # scalar is reconciled against the generate_docs pool total (see
    # run_sn_pools' async-counter drift check); enrich_parents telemetry is
    # surfaced via the per-pool completed-count log instead. The ``run_id``
    # parameter is retained for signature symmetry with the other persist_*
    # functions and possible future per-pool counters.
    return new_stage


#: Path scalars that a ``PRODUCED_NAME`` edge is meant to back. On a name no
#: source produces they are stale copies of a source that has since moved away,
#: and they read as provenance the graph itself contradicts.
_UNBACKED_SOURCE_PATH_SCALARS = ("source_path", "source_paths")


def _structural_accept_route(
    *, origin: str | None, producer_count: int, live_child_count: int
) -> str | None:
    """Return how a stuck structural parent may be accepted, or ``None``.

    ``derived``
        The parent already carries the derived-scaffold origin. Unchanged
        behaviour: a derived parent's name is a grammar peel over its children
        and is never name-reviewed, whether or not a source also produces it.

    ``source_free``
        The origin scalar is absent, yet the edges say this name is a structural
        parent that NO source produces. There is nothing to review the name
        against — the review it would need can never be earned — so its children
        entail it exactly as they entail a stamped derived parent. Decided from
        the edges rather than from ``origin``/``source_path``, because a scalar
        that disagrees with the topology is the stale half of the pair.

    A name with a live producing source and no scaffold origin is a reviewable
    name and is refused, as is any name without a live child.
    """
    if live_child_count < 1:
        return None
    if origin == "derived":
        return "derived"
    if origin is None and producer_count == 0:
        return "source_free"
    return None


@retry_on_deadlock()
def structural_accept_derived_parents(gc: Any | None = None) -> int:
    """Accept any derived parent stuck on the name axis structurally.

    A derived parent's name is a deterministic grammar peel that already passed
    the admission gate and generalises its accepted children by construction —
    so it is NEVER name-reviewed or refined (both claim gates exclude
    ``origin='derived'``). The enrich pool accepts placeholder parents directly
    (:func:`persist_enriched_parent`), but a derived parent can still reach
    ``drafted`` / ``reviewed`` / ``exhausted`` via OTHER paths — a child's
    refine_name producing the parent-equal general name, or a legacy
    materialise routing. Without this they would strand there forever (review
    no longer claims them, refine excludes them).

    This promotes every such parent that has a REAL (non-placeholder)
    description to ``name_stage='accepted'`` (no ``reviewer_score_name`` — the
    name is never reviewed, so scoring it is meaningless), stamping
    ``reviewer_model_name='structural-inheritance'`` and ``reviewed_name_at``
    (the "name finalised" signal the docs axis keys on). Idempotent and
    self-healing — safe to call at every ``sn run`` startup. Placeholder-
    description parents are left for the enrich pool (or are childless zombies).
    Returns the count promoted.

    A parent can also be a structural scaffold WITHOUT carrying the origin —
    a name whose producing source moved to another identity, leaving children
    attached and no ``PRODUCED_NAME`` edge behind. Its name cannot be reviewed
    either (there is no source to review it against), so it strands in exactly
    the same way. :func:`_structural_accept_route` admits that case from the
    edges, and the promotion stamps the scaffold origin and clears the path
    scalars no edge backs. The widening is fenced by ``require_source_free``:
    a name with a live producing source is refused, in the selection and again
    inside the guarded write.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    _own_gc = gc is None
    if _own_gc:
        gc = GraphClient().__enter__()
    try:
        rows = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE (sn.origin = 'derived' OR sn.origin IS NULL)
              AND sn.name_stage IN ['drafted', 'reviewed', 'exhausted', 'refining']
              AND sn.description IS NOT NULL
              AND sn.description <> $ph
              AND NOT (sn)-[:HAS_STRUCTURAL_AUTHORITY]->(
                :StructuralNameAuthority)
              AND EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(sn) }
            RETURN sn.id AS id,
                   sn.origin AS origin,
                   count {
                     (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                   } AS producer_count,
                   count {
                     (child:StandardName)-[:HAS_PARENT]->(sn)
                     WHERE NOT coalesce(child.name_stage, '') IN
                       ['superseded', 'exhausted', 'contested']
                   } AS live_child_count
            ORDER BY sn.id
            """,
            ph=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
        promoted = 0
        for row in rows:
            route = _structural_accept_route(
                origin=row.get("origin"),
                producer_count=int(row.get("producer_count") or 0),
                live_child_count=int(row.get("live_child_count") or 0),
            )
            if route is None:
                continue
            snapshot = _structural_authority_snapshot(gc, str(row["id"]))
            if snapshot is None or not snapshot.get("children"):
                continue
            record = _structural_authority_record(snapshot, accepting=True)
            now = datetime.now(UTC)
            parent_updates = {
                "name_stage": "accepted",
                "reviewer_model_name": "structural-inheritance",
                "reviewed_name_at": snapshot.get("reviewed_name_at") or now,
                "docs_stage": snapshot.get("docs_stage") or "pending",
                "chain_length": snapshot.get("chain_length") or 0,
                "claim_token": None,
                "claimed_at": None,
            }
            if route == "source_free":
                # Promotion is the statement that this name is a source-free
                # scaffold, so it stamps the scaffold origin and drops the path
                # scalars no edge backs in the same guarded write. Leaving them
                # would keep a path this name never had readable as provenance.
                parent_updates["origin"] = "derived"
                parent_updates.update(dict.fromkeys(_UNBACKED_SOURCE_PATH_SCALARS))
            if not _persist_structural_authority(
                gc,
                record,
                parent_updates=parent_updates,
                require_derived=route == "derived",
                require_source_free=route == "source_free",
            ):
                raise StructuralAuthorityConflict(
                    f"derived parent {row['id']!r} changed before atomic acceptance"
                )
            promoted += 1
        if promoted:
            logger.info(
                "structural_accept_derived_parents: promoted %d derived parent(s) "
                "to accepted (structural inheritance)",
                promoted,
            )
        return promoted
    finally:
        if _own_gc:
            gc.__exit__(None, None, None)


@retry_on_deadlock()
def backfill_structural_name_authorities(
    *, receipt_path: Path | None = None, gc: Any | None = None
) -> dict[str, Any]:
    """Authorize every childful bare structural accept, exactly once.

    The marker selects the historical cohort but grants no authority. Each
    written record is instead signed from the exact ordered live child rows.
    Childless parents remain unscoped and are reported without mutation.
    """
    own = gc is None
    client = GraphClient() if own else gc
    try:
        candidate_rows = client.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.name_stage = 'accepted'
              AND sn.reviewer_score_name IS NULL
              AND sn.reviewer_model_name = 'structural-inheritance'
              AND NOT (sn)-[:HAS_STRUCTURAL_AUTHORITY]->(
                :StructuralNameAuthority)
            RETURN sn.id AS id
            ORDER BY sn.id
            """
        )
        lacking_before = len(candidate_rows)
        written: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        refused: list[dict[str, str]] = []
        for candidate in candidate_rows:
            sn_id = str(candidate["id"])
            snapshot = _structural_authority_snapshot(client, sn_id)
            if snapshot is None:
                refused.append({"id": sn_id, "reason": "parent disappeared"})
                continue
            if snapshot.get("authority_ids"):
                refused.append(
                    {"id": sn_id, "reason": "authority appeared during selection"}
                )
                continue
            if not snapshot.get("children"):
                skipped.append({"id": sn_id, "reason": "no live children"})
                continue
            record = _structural_authority_record(snapshot, accepting=False)
            if not _persist_structural_authority(
                client, record, parent_updates={}, require_derived=False
            ):
                refused.append(
                    {"id": sn_id, "reason": "authority guard refused changed row"}
                )
                continue
            written.append(
                {
                    "id": sn_id,
                    "authority_id": record["id"],
                    "child_ids": record["child_ids"],
                    "payload_sha256": record["payload_sha256"],
                }
            )

        eligible_before = lacking_before - len(skipped)
        if refused:
            raise StructuralAuthorityConflict(
                "structural authority backfill refused rows: "
                + "; ".join(f"{row['id']}: {row['reason']}" for row in refused)
            )
        if len(written) != eligible_before:
            raise StructuralAuthorityConflict(
                "structural authority receipt cardinality mismatch: "
                f"eligible={eligible_before}, written={len(written)}"
            )
        receipt = {
            "schema": "imas-codex.structural-authority-backfill-receipt",
            "created_at": datetime.now(UTC).isoformat(),
            "lacking_authority_before": lacking_before,
            "eligible_before": eligible_before,
            "written_count": len(written),
            "written": written,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "refused_count": len(refused),
            "refused": refused,
        }
        if receipt_path is not None:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        return receipt
    finally:
        if own:
            client.close()


def find_orphan_parent_source_candidates(
    gc: Any | None = None,
) -> list[dict[str, Any]]:
    """Return childful parents eligible for structural source reconstruction."""
    own = gc is None
    client = GraphClient() if own else gc
    try:
        return list(
            client.query(
                """
                MATCH (parent:StandardName)
                WHERE NOT ( (:StandardNameSource)-[:PRODUCED_NAME]->(parent) )
                  AND NOT coalesce(parent.name_stage, '') IN
                    ['superseded', 'exhausted', 'contested']
                OPTIONAL MATCH (structural_source:StandardNameSource {
                  id: 'derived:' + parent.id
                })
                WITH parent, structural_source
                WHERE structural_source IS NULL
                   OR structural_source.status <> 'stale'
                WITH parent, structural_source
                WHERE structural_source IS NULL OR NOT EXISTS {
                  MATCH (structural_source)-[:PRODUCED_NAME]->(bound)
                  WHERE bound <> parent
                    AND NOT coalesce(bound.name_stage, '') IN
                      ['superseded', 'exhausted', 'contested']
                }
                MATCH (child)-[:HAS_PARENT]->(parent)
                WITH parent,
                     count(child) AS total_children,
                     count(CASE WHEN child.name_stage IS NOT NULL THEN 1 END)
                       AS composed
                WHERE total_children >= 1 AND total_children = composed
                MATCH (c)-[:HAS_PARENT]->(parent)
                OPTIONAL MATCH (cs:StandardNameSource)-[:PRODUCED_NAME]->(c)
                OPTIONAL MATCH (cs)-[:FROM_DD_PATH]->(imas:IMASNode)
                WITH parent,
                     [p IN collect(DISTINCT imas.id) WHERE p IS NOT NULL]
                       AS dd_paths
                RETURN parent.id AS parent_id,
                       parent.origin AS origin,
                       dd_paths
                ORDER BY parent.id
                """
            )
        )
    finally:
        if own:
            client.close()


def classify_orphan_parent_source_candidates(
    gc: Any,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Partition parent-source candidates by structural admission.

    Only ``origin='derived'`` nodes are structural scaffolds owned by the
    parent admission gate. Pipeline and catalog nodes can also be HAS_PARENT
    targets after refinement, but their identity and lifecycle remain owned by
    those origin-specific paths; provenance recovery must not reinterpret or
    retire them as derived scaffolds.

    Returns the exact ``repairable`` rows safe to pass to source seeding and the
    ``rejected_derived`` rows that must remain visible to fallback/unresolved
    classification until lifecycle normalization retires them.
    """
    rows = (
        find_orphan_parent_source_candidates(gc=gc)
        if candidates is None
        else list(candidates)
    )
    from imas_codex.standard_names.parents import is_admissible_parent_name

    repairable: list[dict[str, Any]] = []
    rejected_derived: list[dict[str, Any]] = []
    for row in rows:
        if row.get("origin") != "derived":
            repairable.append(row)
            continue
        result = is_admissible_parent_name(row["parent_id"], gc)
        if result.admit:
            repairable.append(row)
        else:
            rejected_derived.append({**row, "admission_reason": result.reason})
    return {
        "repairable": repairable,
        "rejected_derived": rejected_derived,
    }


def reconcile_orphan_parent_sources(
    gc: Any | None = None,
    *,
    dry_run: bool = False,
    classification: dict[str, list[dict[str, Any]]] | None = None,
) -> int:
    """Seed the missing structural provenance source for childful parents.

    A structural parent (a ``StandardName`` that is the ``HAS_PARENT`` target of
    at least one child) records its provenance as its own ``derived:<name>``
    source linked via ``PRODUCED_NAME``. A common child DD path does not realize
    the parent, so structural provenance has no ``FROM_DD_PATH`` relationship.
    Both normal materialization and this recovery path use
    :func:`_derived_parent_source_metadata`, so they cannot diverge on source
    identity.

    So a parent that acquires a non-null ``name_stage`` *before* it is seeded is
    skipped forever and left a provenance orphan — the exact leak this repairs.
    Two paths do it: a child's ``REFINE_NAME`` producing the parent-general name
    (which routes that name through ``REVIEW_NAME`` to ``accepted`` as
    ``origin='pipeline'``), and :func:`structural_accept_derived_parents` (which
    flips the stage to ``accepted`` but writes no source). Both leave an accepted
    parent with no ``PRODUCED_NAME`` source — a violation of the ledger invariant
    (:func:`~imas_codex.standard_names.ledger.find_provenance_orphans`).

    This decouples source-seeding from the acceptance path. For every parent with
    at least one child, all children composed, and no ``PRODUCED_NAME`` source, it
    materializes the structural source + ``PRODUCED_NAME`` without a DD
    realization edge. It is origin- and operator-kind-agnostic, but only a live
    parent may receive provenance. The canonical ``derived:<name>`` source must
    also be unbound or bound only to this parent; a source migrated to another
    live target is current authority for that target and must never be reused to
    revive its terminal predecessor. This only records provenance and NEVER
    mutates the parent's ``name_stage`` / ``origin`` / ``description``.
    Idempotent: a parent that gains a source drops out of the selector. A stale
    structural source is a lifecycle tombstone and is never rebound implicitly;
    an explicit lifecycle transition must make it eligible first. Childless
    sourceless names are not parents and are left untouched (a distinct stranded-
    name concern). Every derived candidate is re-checked against the current
    admission gate before seeding; an invalid scaffold remains source-less for
    ledgered lifecycle retirement rather than being legitimized by synthetic
    provenance. Non-derived parents bypass structural admission and retain their
    origin-owned recovery path.

    With ``dry_run=True``, returns the number eligible without writing.
    Otherwise returns the number of parent sources seeded.
    """
    own = gc is None
    client = GraphClient() if own else gc
    try:
        if classification is None:
            classification = classify_orphan_parent_source_candidates(client)
        for row in classification["rejected_derived"]:
            logger.warning(
                "reconcile_orphan_parent_sources: refusing to seed "
                "inadmissible derived parent %s: %s",
                row["parent_id"],
                row["admission_reason"],
            )
        rows = classification["repairable"]
        if dry_run:
            return len(rows)
        seeded = 0
        for r in rows:
            parent_id = r["parent_id"]
            meta = _derived_parent_source_metadata(parent_id)
            result = client.query(
                """
                MATCH (parent:StandardName {id: $parent_id})
                OPTIONAL MATCH (existing_source:StandardNameSource {
                  id: $source_node_id
                })
                WITH parent, existing_source
                WHERE NOT coalesce(parent.name_stage, '') IN
                        ['superseded', 'exhausted', 'contested']
                  AND NOT EXISTS {
                    MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(parent)
                  }
                  AND (existing_source IS NULL
                       OR existing_source.status <> 'stale')
                  AND (existing_source IS NULL OR NOT EXISTS {
                    MATCH (existing_source)-[:PRODUCED_NAME]->(bound)
                    WHERE bound <> parent
                      AND NOT coalesce(bound.name_stage, '') IN
                        ['superseded', 'exhausted', 'contested']
                  })
                MERGE (sns:StandardNameSource {id: $source_node_id})
                  ON CREATE SET sns.created_at = datetime(), sns.attempt_count = 0
                SET sns.status = coalesce(sns.status, 'composed'),
                    sns.source_type = $source_type,
                    sns.source_id = $source_id,
                    sns.batch_key = coalesce(sns.batch_key, $batch_key),
                    sns.composed_at = coalesce(sns.composed_at, datetime()),
                    sns.produced_sn_id = parent.id,
                    sns.claimed_at = null,
                    sns.claim_token = null
                MERGE (sns)-[:PRODUCED_NAME]->(parent)
                RETURN count(parent) AS seeded
                """,
                parent_id=parent_id,
                **meta,
            )
            seeded += result[0]["seeded"] if result else 1
    finally:
        if own:
            client.close()

    if seeded:
        logger.info(
            "reconcile_orphan_parent_sources: seeded %d missing parent "
            "provenance source(s)",
            seeded,
        )
    return seeded


def reconcile_orphan_parent_sources_batched(
    gc: Any,
    rows: list[dict[str, Any]],
    *,
    event_by_parent: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Seed an exact accepted-parent cohort in one graph query."""
    items = []
    for row in rows:
        parent_id = str(row["parent_id"])
        items.append(
            {
                "parent_id": parent_id,
                **_derived_parent_source_metadata(parent_id),
                "event": (event_by_parent or {}).get(parent_id),
            }
        )
    result = gc.query(
        """
        // STRUCTURAL_CLOSURE_BATCH_SEED_SOURCES
        UNWIND $items AS requested_item
        OPTIONAL MATCH (stale_source:StandardNameSource {
          id: requested_item.source_node_id
        })
        WITH collect(requested_item) AS requested_items,
             count(CASE WHEN stale_source.status = 'stale' THEN 1 END)
               AS stale_source_count,
             count(CASE WHEN EXISTS {
               MATCH (stale_source)-[:PRODUCED_NAME]->(bound_target)
               WHERE bound_target.id <> requested_item.parent_id
                 AND NOT coalesce(bound_target.name_stage, '') IN
                   ['superseded', 'exhausted', 'contested']
             } THEN 1 END) AS conflicting_binding_count
        WHERE stale_source_count = 0 AND conflicting_binding_count = 0
        UNWIND requested_items AS item
        MATCH (parent:StandardName {id: item.parent_id})
        WHERE parent.name_stage = 'accepted'
          AND NOT EXISTS {
            MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(parent)
          }
        MERGE (source:StandardNameSource {id: item.source_node_id})
          ON CREATE SET source.created_at = datetime(), source.attempt_count = 0
        SET source.status = coalesce(source.status, 'composed'),
            source.source_type = item.source_type,
            source.source_id = item.source_id,
            source.batch_key = coalesce(source.batch_key, item.batch_key),
            source.composed_at = coalesce(source.composed_at, datetime()),
            source.produced_sn_id = parent.id,
            source.claimed_at = null,
            source.claim_token = null
        MERGE (source)-[:PRODUCED_NAME]->(parent)
        FOREACH (_ IN CASE WHEN item.event IS NULL THEN [] ELSE [1] END |
          CREATE (change:StandardNameChange)
          SET change = item.event
          CREATE (parent)-[:HAS_INTERNAL_CHANGE]->(change)
        )
        RETURN collect(parent.id) AS ids
        """,
        items=items,
    )
    expected = {str(item["parent_id"]) for item in items}
    seeded_ids = set(result[0].get("ids") or []) if len(result) == 1 else set()
    if seeded_ids != expected:
        raise RuntimeError("batched accepted-parent source cardinality changed")
    return len(expected)


@retry_on_deadlock()
def release_enrich_parents_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release enrich_parents claims (token-verified, no stage change).

    Used both as the pool release adapter (clears any unprocessed items in a
    batch) and on LLM/processing failure — the parent stays a placeholder and
    is re-claimed on a later pass.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
            SET sn.updated_at = datetime(), sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    return result[0]["released"] if result else 0


# Preserve the moved callables for signature and source inspection.
_moved_graph_foundation = import_module(".graph_store.graph_foundation", __package__)
_ensure_json.__wrapped__ = _moved_graph_foundation._ensure_json
_ensure_list.__wrapped__ = _moved_graph_foundation._ensure_list
_scalar_domain.__wrapped__ = _moved_graph_foundation._scalar_domain
reclassify_standard_name_domain.__wrapped__ = (
    _moved_graph_foundation.reclassify_standard_name_domain
)
normalize_name_id.__wrapped__ = _moved_graph_foundation.normalize_name_id
_public_segment_token_index.__wrapped__ = (
    _moved_graph_foundation._public_segment_token_index
)
_public_grammar_segment_order.__wrapped__ = (
    _moved_graph_foundation._public_grammar_segment_order
)
_segments_from_ir.__wrapped__ = _moved_graph_foundation._segments_from_ir
_parse_grammar.__wrapped__ = _moved_graph_foundation._parse_grammar
_semantic_ir_expression.__wrapped__ = _moved_graph_foundation._semantic_ir_expression
strict_review_grammar_context.__wrapped__ = (
    _moved_graph_foundation.strict_review_grammar_context
)
_compute_link_status.__wrapped__ = _moved_graph_foundation._compute_link_status
_extract_links_from_docs.__wrapped__ = _moved_graph_foundation._extract_links_from_docs
_moved_graph_audits = import_module(".graph_store.graph_audits", __package__)
claim_compose_sources.__wrapped__ = _moved_graph_audits.claim_compose_sources
count_extracted_for_domain.__wrapped__ = _moved_graph_audits.count_extracted_for_domain
count_eligible_compose_sources.__wrapped__ = (
    _moved_graph_audits.count_eligible_compose_sources
)
claim_review_names.__wrapped__ = _moved_graph_audits.claim_review_names
release_review_claims.__wrapped__ = _moved_graph_audits.release_review_claims
refresh_name_claims.__wrapped__ = _moved_graph_audits.refresh_name_claims
promote_stranded_reviewed.__wrapped__ = _moved_graph_audits.promote_stranded_reviewed
_docs_axis_rederive_query.__wrapped__ = _moved_graph_audits._docs_axis_rederive_query
reconcile_docs_axis_from_reviews.__wrapped__ = (
    _moved_graph_audits.reconcile_docs_axis_from_reviews
)
_standard_name_source_upstream_present_cypher.__wrapped__ = (
    _moved_graph_audits._standard_name_source_upstream_present_cypher
)
reconcile_standard_name_sources.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_sources
)
fetch_manifest_source_release_rows.__wrapped__ = (
    _moved_graph_audits.fetch_manifest_source_release_rows
)
reconcile_source_status_liveness.__wrapped__ = (
    _moved_graph_audits.reconcile_source_status_liveness
)
retire_unreachable_hint_edits.__wrapped__ = (
    _moved_graph_audits.retire_unreachable_hint_edits
)
reconcile_vocab_gaps.__wrapped__ = _moved_graph_audits.reconcile_vocab_gaps
triage_vocab_gaps.__wrapped__ = _moved_graph_audits.triage_vocab_gaps
revive_unit_skipped_sources.__wrapped__ = (
    _moved_graph_audits.revive_unit_skipped_sources
)
_isn_grammar_context.__wrapped__ = _moved_graph_audits._isn_grammar_context
isn_vocabulary_signature.__wrapped__ = _moved_graph_audits.isn_vocabulary_signature
retry_vocab_gap_sources_on_grammar_change.__wrapped__ = (
    _moved_graph_audits.retry_vocab_gap_sources_on_grammar_change
)
reconcile_provenance.__wrapped__ = _moved_graph_audits.reconcile_provenance
reconcile_catalog_edit_origins.__wrapped__ = (
    _moved_graph_audits.reconcile_catalog_edit_origins
)
reconcile_standard_name_cocos_links.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_cocos_links
)
reconcile_standard_name_unit_edges.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_unit_edges
)
reconcile_constraint_fit_artifacts.__wrapped__ = (
    _moved_graph_audits.reconcile_constraint_fit_artifacts
)
reconcile_standard_name_source_paths.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_source_paths
)
reconcile_catalog_status.__wrapped__ = _moved_graph_audits.reconcile_catalog_status
reconcile_reviewable_name_stage.__wrapped__ = (
    _moved_graph_audits.reconcile_reviewable_name_stage
)
reconcile_descriptionless_composed_names.__wrapped__ = (
    _moved_graph_audits.reconcile_descriptionless_composed_names
)
reconcile_sourceless_pipeline_names.__wrapped__ = (
    _moved_graph_audits.reconcile_sourceless_pipeline_names
)
reconcile_standard_name_dd_edges.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_dd_edges
)
reconcile_grammar_segments.__wrapped__ = _moved_graph_audits.reconcile_grammar_segments
realign_grammar_segments_for_name.__wrapped__ = (
    _moved_graph_audits.realign_grammar_segments_for_name
)
_segment_drift_reason.__wrapped__ = _moved_graph_audits._segment_drift_reason
reconcile_standard_name_kinds.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_kinds
)
_open_graph_client.__wrapped__ = _moved_graph_audits._open_graph_client
get_standard_name_source_stats.__wrapped__ = (
    _moved_graph_audits.get_standard_name_source_stats
)
write_run_provenance.__wrapped__ = _moved_graph_audits.write_run_provenance
export_review_comments.__wrapped__ = _moved_graph_audits.export_review_comments
create_sn_run_open.__wrapped__ = _moved_graph_audits.create_sn_run_open
bump_sn_run_counter.__wrapped__ = _moved_graph_audits.bump_sn_run_counter
update_sn_run_progress.__wrapped__ = _moved_graph_audits.update_sn_run_progress
finalize_sn_run.__wrapped__ = _moved_graph_audits.finalize_sn_run
mark_orphaned_standard_name_runs_stale.__wrapped__ = (
    _moved_graph_audits.mark_orphaned_standard_name_runs_stale
)
normalize_stored_standard_name_kinds.__wrapped__ = (
    _moved_graph_audits.normalize_stored_standard_name_kinds
)
backfill_sn_run_telemetry.__wrapped__ = _moved_graph_audits.backfill_sn_run_telemetry
_moved_graph_review_persistence = import_module(
    ".graph_store.graph_review_persistence", __package__
)
_quorum_admits_acceptance.__wrapped__ = (
    _moved_graph_review_persistence._quorum_admits_acceptance
)
_authority_json_value.__wrapped__ = (
    _moved_graph_review_persistence._authority_json_value
)
_authority_payload_hash.__wrapped__ = (
    _moved_graph_review_persistence._authority_payload_hash
)
persist_reviewed_name.__wrapped__ = (
    _moved_graph_review_persistence.persist_reviewed_name
)
claim_review_docs_batch.__wrapped__ = (
    _moved_graph_review_persistence.claim_review_docs_batch
)
persist_reviewed_docs.__wrapped__ = (
    _moved_graph_review_persistence.persist_reviewed_docs
)
_moved_graph_supersession = import_module(
    ".graph_store.graph_supersession", __package__
)
supersede_prior_source_names.__wrapped__ = (
    _moved_graph_supersession.supersede_prior_source_names
)
tombstone_supersede_into.__wrapped__ = (
    _moved_graph_supersession.tombstone_supersede_into
)
_scalar_selected_dedup_authority.__wrapped__ = (
    _moved_graph_supersession._scalar_selected_dedup_authority
)
_lock_scalar_selected_dedup_authority.__wrapped__ = (
    _moved_graph_supersession._lock_scalar_selected_dedup_authority
)
_scalar_selected_dedup_counts.__wrapped__ = (
    _moved_graph_supersession._scalar_selected_dedup_counts
)
deduplicate_scalar_selected_sources.__wrapped__ = (
    _moved_graph_supersession.deduplicate_scalar_selected_sources
)
_catalog_edit_adjudication_signature_hash.__wrapped__ = (
    _moved_graph_supersession._catalog_edit_adjudication_signature_hash
)
_validate_signed_source_adjudication.__wrapped__ = (
    _moved_graph_supersession._validate_signed_source_adjudication
)
_validate_structural_legitimacy_authority.__wrapped__ = (
    _moved_graph_supersession._validate_structural_legitimacy_authority
)
_is_live_structural_child.__wrapped__ = (
    _moved_graph_supersession._is_live_structural_child
)
_signed_source_disposition_authority.__wrapped__ = (
    _moved_graph_supersession._signed_source_disposition_authority
)
_lock_signed_source_disposition_authority.__wrapped__ = (
    _moved_graph_supersession._lock_signed_source_disposition_authority
)
_signed_source_disposition_counts.__wrapped__ = (
    _moved_graph_supersession._signed_source_disposition_counts
)
_signed_source_disposition_execution_authority.__wrapped__ = (
    _moved_graph_supersession._signed_source_disposition_execution_authority
)
_semantic_mirror_repair_authority.__wrapped__ = (
    _moved_graph_supersession._semantic_mirror_repair_authority
)
_lock_semantic_mirror_repair_authority.__wrapped__ = (
    _moved_graph_supersession._lock_semantic_mirror_repair_authority
)
_semantic_mirror_repair_counts.__wrapped__ = (
    _moved_graph_supersession._semantic_mirror_repair_counts
)
_validate_signed_orphan_retirement_authority.__wrapped__ = (
    _moved_graph_supersession._validate_signed_orphan_retirement_authority
)
_validate_signed_dual_authority_retirement.__wrapped__ = (
    _moved_graph_supersession._validate_signed_dual_authority_retirement
)
_signed_dual_authority_retirement_manifest.__wrapped__ = (
    _moved_graph_supersession._signed_dual_authority_retirement_manifest
)
_signed_dual_authority_change_id.__wrapped__ = (
    _moved_graph_supersession._signed_dual_authority_change_id
)
classify_sourceless_standard_names.__wrapped__ = (
    _moved_graph_supersession.classify_sourceless_standard_names
)
_exhausted_orphan_supersession_authority.__wrapped__ = (
    _moved_graph_supersession._exhausted_orphan_supersession_authority
)
_exhausted_orphan_change_id.__wrapped__ = (
    _moved_graph_supersession._exhausted_orphan_change_id
)
supersede_exhausted_standard_name_orphans.__wrapped__ = (
    _moved_graph_supersession.supersede_exhausted_standard_name_orphans
)
_ancestor_supersession_authority.__wrapped__ = (
    _moved_graph_supersession._ancestor_supersession_authority
)
_lock_ancestor_supersession_authority.__wrapped__ = (
    _moved_graph_supersession._lock_ancestor_supersession_authority
)
_detach_stale_ancestor_sources.__wrapped__ = (
    _moved_graph_supersession._detach_stale_ancestor_sources
)
supersede_into_ancestor.__wrapped__ = _moved_graph_supersession.supersede_into_ancestor

_moved_graph_doc_links = import_module(".graph_store.graph_doc_links", __package__)
claim_unresolved_links.__wrapped__ = _moved_graph_doc_links.claim_unresolved_links
resolve_links_batch.__wrapped__ = _moved_graph_doc_links.resolve_links_batch
_sanitize_doc_text.__wrapped__ = _moved_graph_doc_links._sanitize_doc_text
_doc_link_mismatches.__wrapped__ = _moved_graph_doc_links._doc_link_mismatches
_normalize_bare_doc_links.__wrapped__ = _moved_graph_doc_links._normalize_bare_doc_links
resolve_doc_links.__wrapped__ = _moved_graph_doc_links.resolve_doc_links
_moved_graph_docs_persistence = import_module(
    ".graph_store.graph_docs_persistence", __package__
)
claim_generate_docs_batch.__wrapped__ = (
    _moved_graph_docs_persistence.claim_generate_docs_batch
)
persist_generated_docs.__wrapped__ = (
    _moved_graph_docs_persistence.persist_generated_docs
)
release_generate_docs_claims.__wrapped__ = (
    _moved_graph_docs_persistence.release_generate_docs_claims
)
release_generate_docs_failed_claims.__wrapped__ = (
    _moved_graph_docs_persistence.release_generate_docs_failed_claims
)
claim_refine_docs_batch.__wrapped__ = (
    _moved_graph_docs_persistence.claim_refine_docs_batch
)
persist_refined_docs.__wrapped__ = _moved_graph_docs_persistence.persist_refined_docs
reset_standard_name_docs.__wrapped__ = (
    _moved_graph_docs_persistence.reset_standard_name_docs
)
stamp_harmonized_families.__wrapped__ = (
    _moved_graph_docs_persistence.stamp_harmonized_families
)
release_refine_docs_claims.__wrapped__ = (
    _moved_graph_docs_persistence.release_refine_docs_claims
)
release_refine_docs_failed_claims.__wrapped__ = (
    _moved_graph_docs_persistence.release_refine_docs_failed_claims
)
mark_for_refine_docs.__wrapped__ = _moved_graph_docs_persistence.mark_for_refine_docs
_moved_graph_cost_and_claims = import_module(
    ".graph_store.graph_cost_and_claims", __package__
)
_normalize_pool.__wrapped__ = _moved_graph_cost_and_claims._normalize_pool
record_llm_cost.__wrapped__ = _moved_graph_cost_and_claims.record_llm_cost
aggregate_spend_for_run.__wrapped__ = (
    _moved_graph_cost_and_claims.aggregate_spend_for_run
)
aggregate_spend_per_phase.__wrapped__ = (
    _moved_graph_cost_and_claims.aggregate_spend_per_phase
)
aggregate_calls_per_phase.__wrapped__ = (
    _moved_graph_cost_and_claims.aggregate_calls_per_phase
)
aggregate_spend_per_name.__wrapped__ = (
    _moved_graph_cost_and_claims.aggregate_spend_per_name
)
update_sn_per_phase_costs.__wrapped__ = (
    _moved_graph_cost_and_claims.update_sn_per_phase_costs
)
_claim_sn_atomic.__wrapped__ = _moved_graph_cost_and_claims._claim_sn_atomic
claim_generate_name_batch.__wrapped__ = (
    _moved_graph_cost_and_claims.claim_generate_name_batch
)
claim_enrich_seed_and_expand.__wrapped__ = (
    _moved_graph_cost_and_claims.claim_enrich_seed_and_expand
)
claim_review_names_seed_and_expand.__wrapped__ = (
    _moved_graph_cost_and_claims.claim_review_names_seed_and_expand
)
claim_review_name_batch.__wrapped__ = (
    _moved_graph_cost_and_claims.claim_review_name_batch
)
_moved_graph_lineage = import_module(".graph_store.graph_lineage", __package__)
get_source_name_mapping.__wrapped__ = _moved_graph_lineage.get_source_name_mapping
_get_rich_source_name_mapping.__wrapped__ = (
    _moved_graph_lineage._get_rich_source_name_mapping
)
fetch_low_score_sources.__wrapped__ = _moved_graph_lineage.fetch_low_score_sources
fetch_review_feedback_for_sources.__wrapped__ = (
    _moved_graph_lineage.fetch_review_feedback_for_sources
)
fetch_reviewer_history_for_sources.__wrapped__ = (
    _moved_graph_lineage.fetch_reviewer_history_for_sources
)
fetch_docs_review_feedback_for_standard_names.__wrapped__ = (
    _moved_graph_lineage.fetch_docs_review_feedback_for_standard_names
)
fetch_docs_review_feedback_for_sns.__wrapped__ = (
    _moved_graph_lineage.fetch_docs_review_feedback_for_sns
)
_filter_admissible_parents.__wrapped__ = _moved_graph_lineage._filter_admissible_parents
_emit_magnitude_of_edges.__wrapped__ = _moved_graph_lineage._emit_magnitude_of_edges
_bootstrap_missing_derived_parent_targets.__wrapped__ = (
    _moved_graph_lineage._bootstrap_missing_derived_parent_targets
)
_write_standard_name_edges.__wrapped__ = _moved_graph_lineage._write_standard_name_edges
_clear_superseded_child_derived_parent_edges.__wrapped__ = (
    _moved_graph_lineage._clear_superseded_child_derived_parent_edges
)
rederive_structural_edges.__wrapped__ = _moved_graph_lineage.rederive_structural_edges
_rewire_has_parent_off_superseded.__wrapped__ = (
    _moved_graph_lineage._rewire_has_parent_off_superseded
)
_parse_parent_grammar.__wrapped__ = _moved_graph_lineage._parse_parent_grammar
_query_seedable_derived_parents.__wrapped__ = (
    _moved_graph_lineage._query_seedable_derived_parents
)
_query_legacy_repairable_derived_parents.__wrapped__ = (
    _moved_graph_lineage._query_legacy_repairable_derived_parents
)
_query_derived_parents_for_admission_cleanup.__wrapped__ = (
    _moved_graph_lineage._query_derived_parents_for_admission_cleanup
)
_delete_derived_parent_nodes.__wrapped__ = (
    _moved_graph_lineage._delete_derived_parent_nodes
)
sweep_orphaned_docs_revisions.__wrapped__ = (
    _moved_graph_lineage.sweep_orphaned_docs_revisions
)
_derived_parent_source_metadata.__wrapped__ = (
    _moved_graph_lineage._derived_parent_source_metadata
)
_validate_derived_parent_identity.__wrapped__ = (
    _moved_graph_lineage._validate_derived_parent_identity
)
_eligible_derived_parent_unit_children.__wrapped__ = (
    _moved_graph_lineage._eligible_derived_parent_unit_children
)
_materialize_derived_parent_rows.__wrapped__ = (
    _moved_graph_lineage._materialize_derived_parent_rows
)
_materialize_derived_parent_rows_batched.__wrapped__ = (
    _moved_graph_lineage._materialize_derived_parent_rows_batched
)
stamp_standard_name_units.__wrapped__ = _moved_graph_lineage.stamp_standard_name_units
seed_parent_sources.__wrapped__ = _moved_graph_lineage.seed_parent_sources
normalize_derived_parent_lifecycle.__wrapped__ = (
    _moved_graph_lineage.normalize_derived_parent_lifecycle
)
_canonicalize_lifecycleless_manifest_value.__wrapped__ = (
    _moved_graph_lineage._canonicalize_lifecycleless_manifest_value
)
_derived_parent_unit_from_eligible_children.__wrapped__ = (
    _moved_graph_lineage._derived_parent_unit_from_eligible_children
)
_spurious_stub_binding_authority.__wrapped__ = (
    _moved_graph_lineage._spurious_stub_binding_authority
)
_partition_lifecycleless_stub_rows.__wrapped__ = (
    _moved_graph_lineage._partition_lifecycleless_stub_rows
)
_lifecycleless_stub_manifest.__wrapped__ = (
    _moved_graph_lineage._lifecycleless_stub_manifest
)
_reconcile_spurious_stub_source_scalars.__wrapped__ = (
    _moved_graph_lineage._reconcile_spurious_stub_source_scalars
)
_query_skeleton_placeholders_for_cleanup.__wrapped__ = (
    _moved_graph_lineage._query_skeleton_placeholders_for_cleanup
)

_moved_graph_catalog_reset = import_module(
    ".graph_store.graph_catalog_reset", __package__
)
write_vocab_gaps.__wrapped__ = _moved_graph_catalog_reset.write_vocab_gaps
fetch_vocab_gap_adjudications.__wrapped__ = (
    _moved_graph_catalog_reset.fetch_vocab_gap_adjudications
)
claim_names_for_validation.__wrapped__ = (
    _moved_graph_catalog_reset.claim_names_for_validation
)
mark_names_validated.__wrapped__ = _moved_graph_catalog_reset.mark_names_validated
release_validation_claims.__wrapped__ = (
    _moved_graph_catalog_reset.release_validation_claims
)
claim_names_for_embedding.__wrapped__ = (
    _moved_graph_catalog_reset.claim_names_for_embedding
)
mark_names_embedded.__wrapped__ = _moved_graph_catalog_reset.mark_names_embedded
release_embedding_claims.__wrapped__ = (
    _moved_graph_catalog_reset.release_embedding_claims
)
get_validated_names.__wrapped__ = _moved_graph_catalog_reset.get_validated_names
mark_names_consolidated.__wrapped__ = _moved_graph_catalog_reset.mark_names_consolidated
reset_standard_names.__wrapped__ = _moved_graph_catalog_reset.reset_standard_names
clear_standard_names.__wrapped__ = _moved_graph_catalog_reset.clear_standard_names
clear_sn_subsystem.__wrapped__ = _moved_graph_catalog_reset.clear_sn_subsystem

_moved_graph_claim_release = import_module(
    ".graph_store.graph_claim_release", __package__
)
release_generate_name_claims.__wrapped__ = (
    _moved_graph_claim_release.release_generate_name_claims
)
release_generate_name_failed_claims.__wrapped__ = (
    _moved_graph_claim_release.release_generate_name_failed_claims
)
release_enrich_claims.__wrapped__ = _moved_graph_claim_release.release_enrich_claims
release_enrich_failed_claims.__wrapped__ = (
    _moved_graph_claim_release.release_enrich_failed_claims
)
release_review_names_claims.__wrapped__ = (
    _moved_graph_claim_release.release_review_names_claims
)
release_review_names_failed_claims.__wrapped__ = (
    _moved_graph_claim_release.release_review_names_failed_claims
)
release_review_docs_claims.__wrapped__ = (
    _moved_graph_claim_release.release_review_docs_claims
)
release_review_docs_failed_claims.__wrapped__ = (
    _moved_graph_claim_release.release_review_docs_failed_claims
)
_sorted_refine_claim_projection.__wrapped__ = (
    _moved_graph_claim_release._sorted_refine_claim_projection
)
_project_refine_claim_release_state.__wrapped__ = (
    _moved_graph_claim_release._project_refine_claim_release_state
)
_predicted_refine_claim_release_state.__wrapped__ = (
    _moved_graph_claim_release._predicted_refine_claim_release_state
)
_refine_claim_release_manifest.__wrapped__ = (
    _moved_graph_claim_release._refine_claim_release_manifest
)
_refine_claim_no_change_receipt.__wrapped__ = (
    _moved_graph_claim_release._refine_claim_no_change_receipt
)
release_stranded_refine_claim.__wrapped__ = (
    _moved_graph_claim_release.release_stranded_refine_claim
)
release_refine_name_claims.__wrapped__ = (
    _moved_graph_claim_release.release_refine_name_claims
)
resubmit_pinned_rename_for_review.__wrapped__ = (
    _moved_graph_claim_release.resubmit_pinned_rename_for_review
)
stage_name_for_rescore.__wrapped__ = _moved_graph_claim_release.stage_name_for_rescore
restore_name_after_failed_rescore.__wrapped__ = (
    _moved_graph_claim_release.restore_name_after_failed_rescore
)
stop_refine_name_attempt.__wrapped__ = (
    _moved_graph_claim_release.stop_refine_name_attempt
)
_mark_refine_docs_exhausted.__wrapped__ = (
    _moved_graph_claim_release._mark_refine_docs_exhausted
)
_verify_docs_claim_winners.__wrapped__ = (
    _moved_graph_claim_release._verify_docs_claim_winners
)
_verify_claim_winners.__wrapped__ = _moved_graph_claim_release._verify_claim_winners
_verify_name_claim_winners.__wrapped__ = (
    _moved_graph_claim_release._verify_name_claim_winners
)
_verify_source_claim_winners.__wrapped__ = (
    _moved_graph_claim_release._verify_source_claim_winners
)

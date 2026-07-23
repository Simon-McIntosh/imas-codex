"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import UTC
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.ledger import reattach_produced_name_edges

logger = logging.getLogger(__name__)


def _ensure_json(value: Any) -> str | None:
    """Ensure a value is a JSON string, not a raw dict/list (Neo4j rejects Maps)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _ensure_list(value: Any) -> list[str]:
    """Coerce *value* to a list of strings (for multi-valued fields).

    - ``None`` / empty string → ``[]``
    - scalar string → ``[value]``
    - list → passed through
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _scalar_domain(value: Any) -> str | None:
    """Coerce a possibly-list ``physics_domain`` value to a scalar string.

    Defensive helper for reads from the graph during the schema migration
    window — legacy nodes may still have a list-valued ``physics_domain``
    before ``sn clear`` is run.  Returns the first non-empty entry from
    a list, or the string itself, or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    return None


#: Physics abbreviation expansions applied before name persistence.
#: Keys are case-sensitive abbreviations as they appear in LLM output;
#: values are their lowercase ISN-compliant expansions.
_PHYSICS_ABBREVIATIONS: dict[str, str] = {
    "ExB": "e_cross_b",
    "BxGradB": "b_cross_grad_b",
}


def normalize_name_id(name: str) -> str:
    """Normalize a standard name ID: expand physics abbreviations, force lowercase.

    Applies known physics abbreviation expansions (case-sensitive match on the
    abbreviation, e.g. ``ExB`` → ``e_cross_b``) and then forces the entire
    string to lowercase.  This must be called before grammar round-trip
    validation so that ISN's parser sees only valid lowercase tokens.

    Args:
        name: Raw standard name ID as produced by the LLM.

    Returns:
        Normalized lowercase name ID with physics abbreviations expanded.
    """
    for abbrev, expansion in _PHYSICS_ABBREVIATIONS.items():
        name = name.replace(abbrev, expansion)
    return name.lower()


def _segments_from_model(parsed: Any) -> dict[str, str | None]:
    """Extract all bare-name segment columns from an ISN pydantic StandardName.

    Single extraction authority shared by the persist path
    (:func:`_parse_grammar`), the decomposition writer
    (:func:`_write_grammar_decomposition`), and the derived-parent seeding
    path (:func:`_parse_parent_grammar`). Graph column names equal the ISN
    pydantic ``StandardName`` segment attribute names, so the mapping is a
    direct ``model_dump`` projection over ``_GRAMMAR_SEGMENT_COLUMNS``.

    Args:
        parsed: Result of ``imas_standard_names.grammar.parse_standard_name``.

    Returns:
        Dict with one entry per segment in ``_GRAMMAR_SEGMENT_COLUMNS``;
        absent segments are ``None``, enum values are coerced to ``str``.
    """
    dump = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    return {
        seg: _coerce_segment_value(dump.get(seg)) for seg in _GRAMMAR_SEGMENT_COLUMNS
    }


def _parse_grammar(name: str) -> dict[str, Any]:
    """Parse ``name`` with the ISN grammar API.

    Accept/reject and segment extraction are owned by the ISN pydantic
    ``parse_standard_name()`` — the same authority
    ``_write_grammar_decomposition`` uses, so persist-time columns and
    decomposition-time columns always agree. The lower-level IR ``parse()``
    is retained ONLY to produce ``validation_diagnostics_json`` (the one
    artefact the pydantic layer does not provide).

    Returns a dict with ``grammar_parse_version`` (ISN package version
    string), ``validation_diagnostics_json`` (JSON array of diagnostic
    objects, ``"[]"`` when the IR parser rejects the name), and all
    bare-name segment columns. When the ISN pydantic model rejects the name
    (stacked tokens, bare generic base, unknown token, non-canonical order)
    the segment columns are all ``None``; the name's non-compliance is
    recorded authoritatively by ``validation_status='quarantined'`` at
    validate time (:func:`validate_worker`), not by a separate flag.
    """
    try:
        import dataclasses

        import imas_standard_names
        from imas_standard_names.grammar import parse_standard_name
        from imas_standard_names.grammar.parser import parse

        version: str = imas_standard_names.__version__
    except ImportError:
        return {
            "grammar_parse_version": None,
            "validation_diagnostics_json": None,
            **dict.fromkeys(_GRAMMAR_SEGMENT_COLUMNS),
        }

    # IR parse — diagnostics only, never segments.
    try:
        result = parse(name)
        diags = json.dumps([dataclasses.asdict(d) for d in result.diagnostics])
    except Exception:
        logger.debug(
            "ISN grammar parse rejected '%s' — storing empty diagnostics", name
        )
        diags = "[]"

    # Pydantic model — accept/reject + segment authority.
    try:
        parsed = parse_standard_name(name)
    except Exception:
        logger.debug(
            "ISN model rejected '%s' — fallback, segment columns cleared", name
        )
        return {
            "grammar_parse_version": version,
            "validation_diagnostics_json": diags,
            **dict.fromkeys(_GRAMMAR_SEGMENT_COLUMNS),
        }

    return {
        "grammar_parse_version": version,
        "validation_diagnostics_json": diags,
        **_segments_from_model(parsed),
    }


def _compute_link_status(links: list[str] | None) -> str | None:
    """Determine link resolution status from link prefixes.

    Returns 'resolved' if all links are ``name:`` or URL prefixed,
    'unresolved' if any link has ``dd:`` prefix (pending resolution),
    or None if no links exist.
    """
    if not links:
        return None
    has_unresolved = any(link.startswith("dd:") for link in links)
    return "unresolved" if has_unresolved else "resolved"


_LINK_RE = re.compile(
    r"""
    \[([^\]]*?)\]\(name:([a-z0-9_]+)\)  |  # markdown link [text](name:xxx)
    \bname:([a-z0-9_]+)\b                   # bare name:xxx reference
    """,
    re.VERBOSE,
)

_DD_LINK_RE = re.compile(
    r"""
    \[([^\]]*?)\]\(dd:([^\)]+)\)  |  # markdown link [text](dd:path)
    \bdd:([a-z0-9_/]+)\b             # bare dd:path reference
    """,
    re.VERBOSE,
)


def _extract_links_from_docs(
    documentation: str | None,
    *,
    self_name: str | None = None,
) -> list[str] | None:
    """Extract structured links from documentation markdown text.

    Looks for ``name:xxx`` and ``dd:path`` references in the documentation
    and returns an order-preserving deduplicated list of link strings
    suitable for the ``links`` field on StandardName nodes.

    When ``self_name`` is set, a ``name:<self_name>`` reference is
    dropped: prose may legitimately mention the current entry by name
    in one or more paragraphs, but the structured ``links`` index is a
    cross-reference list and a self-loop is meaningless. The ISNC
    SQLite catalog enforces UNIQUE(name, link) and a self-reference
    that survived deduplication used to crash ``validate_catalog``.
    """
    if not documentation:
        return None

    links: list[str] = []
    seen: set[str] = set()
    self_link = f"name:{self_name}" if self_name else None

    for m in _LINK_RE.finditer(documentation):
        name_id = m.group(2) or m.group(3)
        if name_id:
            ref = f"name:{name_id}"
            if ref == self_link:
                continue
            if ref not in seen:
                links.append(ref)
                seen.add(ref)

    for m in _DD_LINK_RE.finditer(documentation):
        dd_path = m.group(2) or m.group(3)
        if dd_path:
            ref = f"dd:{dd_path}"
            if ref not in seen:
                links.append(ref)
                seen.add(ref)

    return links if links else None


# =============================================================================
# Pipeline cost model — bucket queries & historical CPI
# =============================================================================

# Module-level TTL cache for pipeline buckets.
_pipeline_buckets_cache: dict[str, Any] = {
    "value": None,
    "ts": 0.0,
}

# Module-level cache for historical CPI (once per process).
_historical_cpi_cache: dict[str, Any] = {
    "value": None,
    "ts": 0.0,
}


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
        return list(results)


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
                   s.units AS units,
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


def get_existing_standard_names() -> set[str]:
    """Return the set of existing StandardName node IDs for deduplication."""
    with GraphClient() as gc:
        results = gc.query("MATCH (sn:StandardName) RETURN sn.id AS id")
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


def get_source_name_mapping(*, rich: bool = False) -> dict[str, dict]:
    """Return mapping of source_id → previous standard name details.

    Used by extract_worker in --force mode to inject per-path
    previous_name context so the LLM can improve on prior names.

    Args:
        rich: If True, return full SN metadata including documentation,
            links, and all linked DD paths. Used by --paths mode.

    Returns:
        Dict mapping source entity ID to dict with keys:
        name, description, kind, name_stage (and more if rich=True).
        If a source has multiple names, prefers the accepted one.
    """
    if rich:
        return _get_rich_source_name_mapping()

    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN src.id AS source_id,
                   sn.id AS name,
                   sn.description AS description,
                   sn.kind AS kind,
                   sn.name_stage AS name_stage
        """)
        mapping: dict[str, dict] = {}
        for r in results:
            sid = r["source_id"]
            # If multiple names exist for same source, prefer accepted
            if sid not in mapping or r.get("name_stage") == "accepted":
                mapping[sid] = {
                    "name": r["name"],
                    "description": r.get("description"),
                    "kind": r.get("kind"),
                    "name_stage": r.get("name_stage"),
                }
        return mapping


def _get_rich_source_name_mapping() -> dict[str, dict]:
    """Full SN metadata with documentation + all linked DD paths."""
    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (other_src)-[:HAS_STANDARD_NAME]->(sn)
            WHERE other_src <> src
            RETURN src.id AS source_id,
                   sn.id AS name,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   sn.links AS links,
                   sn.name_stage AS name_stage,
                   sn.reviewer_score_name AS reviewer_score,
                   sn.review_tier AS review_tier,
                   sn.validation_issues AS validation_issues,
                   u.id AS unit,
                   collect(DISTINCT other_src.id) AS linked_dd_paths
        """)
        mapping: dict[str, dict] = {}
        for r in results:
            sid = r["source_id"]
            if sid not in mapping or r.get("name_stage") == "accepted":
                mapping[sid] = {
                    "name": r["name"],
                    "description": r.get("description"),
                    "documentation": r.get("documentation"),
                    "kind": r.get("kind"),
                    "links": r.get("links"),
                    "name_stage": r.get("name_stage"),
                    "reviewer_score": r.get("reviewer_score"),
                    "review_tier": r.get("review_tier"),
                    "validation_issues": r.get("validation_issues"),
                    "unit": r.get("unit"),
                    "linked_dd_paths": [
                        p for p in (r.get("linked_dd_paths") or []) if p != sid
                    ],
                }
        return mapping


# =============================================================================
# Write helpers
# =============================================================================


def fetch_low_score_sources(
    *,
    min_score: float | None = None,
    domain: str | None = None,
    ids: str | None = None,
    limit: int | None = None,
    source_type: str = "dd",
) -> list[dict[str, Any]]:
    """Enumerate sources whose linked StandardName has ``reviewer_score_name < min_score``.

    Walks ``(:IMASNode|:FacilitySignal)-[:HAS_STANDARD_NAME]->(:StandardName)``
    and returns the originating source IDs along with the reviewer feedback
    needed to prompt a targeted regeneration. Used by the extract worker's
    regen mode (``sn run --min-score F``) to re-queue the exact sources
    whose names scored below the threshold.

    Only reviewed names are considered (``reviewer_score_name IS NOT NULL``), so
    unreviewed names are never pulled into regen. Duplicates per source_id
    are collapsed, keeping the lowest-scoring entry ("worst critique wins").

    Args:
        min_score: Reviewer-score threshold (0-1). Names with
            ``reviewer_score_name < min_score`` are returned. Required for any
            results; None short-circuits to an empty list.
        domain: Optional physics-domain filter applied to the StandardName.
        ids: Optional IDS-name filter (DD source only).
        limit: Optional cap on rows returned (ordered by worst score first).
        source_type: ``"dd"`` or ``"signals"``.

    Returns:
        List of dicts with ``source_id``, ``source_type``, ``review_feedback``.
    """
    if min_score is None:
        return []

    if source_type == "dd":
        match_clause = "MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)"
    elif source_type == "signals":
        match_clause = (
            "MATCH (src:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)"
        )
    else:
        raise ValueError(f"source_type must be 'dd' or 'signals', got {source_type!r}")

    where_clauses = [
        "sn.reviewer_score_name IS NOT NULL",
        "sn.reviewer_score_name < $min_score",
        "coalesce(sn.regen_count, 0) < 1",
    ]
    params: dict[str, Any] = {"min_score": float(min_score)}

    if domain:
        where_clauses.append("sn.physics_domain = $domain")
        params["domain"] = domain

    if ids and source_type == "dd":
        match_clause += "\n            MATCH (src)-[:IN_IDS]->(ids_node:IDS)"
        where_clauses.append("ids_node.id = $ids")
        params["ids"] = ids

    query = f"""
        {match_clause}
        WHERE {" AND ".join(where_clauses)}
        RETURN src.id AS source_id,
               sn.id AS previous_name,
               sn.description AS previous_description,
               sn.documentation AS previous_documentation,
               sn.reviewer_score_name AS reviewer_score,
               sn.review_tier AS review_tier,
               sn.reviewer_comments_name AS reviewer_comments,
               sn.reviewer_scores_name AS reviewer_scores_json,
               sn.validation_status AS validation_status
        ORDER BY coalesce(sn.reviewer_score_name, 1.0) ASC, src.id ASC
    """
    if limit is not None and limit > 0:
        query += "\n        LIMIT $limit"
        params["limit"] = int(limit)

    with GraphClient() as gc:
        rows = list(gc.query(query, **params))

    # Collapse duplicates: keep the worst-scoring feedback per source_id.
    by_source: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        sid = row.get("source_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores: dict[str, Any] | None = None
        if scores_json:
            try:
                scores = json.loads(scores_json)
            except (TypeError, ValueError):
                scores = None
        feedback = {
            "previous_name": row.get("previous_name"),
            "previous_description": row.get("previous_description"),
            "previous_documentation": row.get("previous_documentation"),
            "reviewer_score": row.get("reviewer_score"),
            "review_tier": row.get("review_tier"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores,
            "validation_status": row.get("validation_status"),
        }
        existing = by_source.get(sid)
        if existing is None:
            by_source[sid] = feedback
            order.append(sid)
        else:
            prev = existing.get("reviewer_score")
            cur = feedback.get("reviewer_score")
            if prev is None or (cur is not None and cur < prev):
                by_source[sid] = feedback

    return [
        {
            "source_id": sid,
            "source_type": source_type,
            "review_feedback": by_source[sid],
        }
        for sid in order
    ]


def fetch_review_feedback_for_sources(
    source_ids: list[str] | set[str] | None,
) -> dict[str, dict[str, Any]]:
    """Fetch prior reviewer feedback for a batch of StandardNameSource ids.

    Used by the compose worker when a regeneration run is invoked with
    ``--min-score F``. Each returned dict carries enough context
    to let the LLM understand what the previous reviewer objected to and
    adjust the new candidate accordingly.

    Args:
        source_ids: Iterable of source node ids (e.g. ``dd:equilibrium/...``
            or ``signals:tcv:...``). ``None`` or empty input returns ``{}``
            without hitting the graph.

    Returns:
        ``{source_id: feedback_dict}`` where feedback_dict has keys:

        - ``previous_name`` (str | None): prior standard-name id
        - ``previous_description`` (str | None)
        - ``previous_documentation`` (str | None)
        - ``reviewer_score`` (float | None): composite 0–1 score
        - ``review_tier`` (str | None): ``outstanding|good|inadequate|poor``
        - ``reviewer_comments`` (str | None): free-form reviewer critique
        - ``reviewer_scores`` (dict | None): parsed name-axis dimensional scores
        - ``validation_status`` (str | None): graph lifecycle state at
          fetch time

        Only sources that currently link to a StandardName with a
        non-null ``reviewer_score_name`` are returned — entries without prior
        review data are silently omitted (the caller can treat this as a
        cold-start and skip feedback injection).
    """
    if not source_ids:
        return {}

    ids = sorted({sid for sid in source_ids if sid})
    if not ids:
        return {}

    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS source_id
            MATCH (src {id: source_id})-[:HAS_STANDARD_NAME]->(sn:StandardName)
            WHERE sn.reviewer_score_name IS NOT NULL
            RETURN source_id AS source_id,
                   sn.id AS previous_name,
                   sn.description AS previous_description,
                   sn.documentation AS previous_documentation,
                   sn.reviewer_score_name AS reviewer_score,
                   sn.review_tier AS review_tier,
                   sn.reviewer_comments_name AS reviewer_comments,
                   sn.reviewer_scores_name AS reviewer_scores_json,
                   sn.reviewer_suggested_name AS reviewer_suggested_name,
                   sn.reviewer_suggestion_justification_name
                       AS reviewer_suggestion_justification,
                   sn.validation_status AS validation_status,
                   sn.edit_mode AS edit_mode,
                   sn.name_hint AS name_hint,
                   sn.docs_hint AS docs_hint,
                   sn.edit_reason AS edit_reason,
                   sn.edit_origin AS edit_origin
            """,
            ids=ids,
        )

    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("source_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores_dict: dict[str, Any] | None = None
        if scores_json:
            try:
                scores_dict = json.loads(scores_json)
            except (TypeError, ValueError):
                scores_dict = None
        # If a source_id has multiple linked SNs, prefer the one with the
        # lowest reviewer_score (the one most in need of revision).
        new_entry = {
            "previous_name": row.get("previous_name"),
            "previous_description": row.get("previous_description"),
            "previous_documentation": row.get("previous_documentation"),
            "reviewer_score": row.get("reviewer_score"),
            "review_tier": row.get("review_tier"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores_dict,
            "reviewer_suggested_name": row.get("reviewer_suggested_name") or None,
            "reviewer_suggestion_justification": (
                row.get("reviewer_suggestion_justification") or None
            ),
            "validation_status": row.get("validation_status"),
            "edit_mode": row.get("edit_mode"),
            "name_hint": row.get("name_hint"),
            "docs_hint": row.get("docs_hint"),
            "edit_reason": row.get("edit_reason"),
            "edit_origin": row.get("edit_origin"),
        }
        existing = mapping.get(sid)
        if existing is None:
            mapping[sid] = new_entry
            continue
        prev = existing.get("reviewer_score")
        cur = new_entry.get("reviewer_score")
        if prev is None or (cur is not None and cur < prev):
            mapping[sid] = new_entry
    return mapping


def fetch_reviewer_history_for_sources(
    source_ids: list[str] | set[str] | None,
) -> dict[str, dict[str, Any]]:
    """Fetch full StandardNameReview-node history per source for compose prompt enrichment.

    For each source that currently links to a StandardName with ≥1
    StandardNameReview node, returns:

    - ``latest``: the most recent StandardNameReview (score, comment ≤800 chars, model).
    - ``prior_themes``: n-gram theme extraction from older StandardNameReview comments,
      rendered as ``[{theme, count, example}]``.

    This complements :func:`fetch_review_feedback_for_sources` which injects
    only the denormalised aggregates from the StandardName node itself.  The
    history variant drills into *all* StandardNameReview nodes for richer compose context.

    Args:
        source_ids: Iterable of source-node ids (e.g. ``dd:equilibrium/...``).
            ``None`` or empty input returns ``{}`` without hitting the graph.

    Returns:
        ``{source_id: history_dict}`` where history_dict has keys:

        - ``latest`` (dict): ``{score, comment, model}`` of newest review.
        - ``prior_themes`` (list[dict]): theme summaries from older reviews,
          each ``{theme, count, example}``.  Empty when only one review exists.
    """
    if not source_ids:
        return {}

    ids = sorted({sid for sid in source_ids if sid})
    if not ids:
        return {}

    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS source_id
            MATCH (src {id: source_id})-[:HAS_STANDARD_NAME]->(sn:StandardName)
                  -[:HAS_REVIEW]->(r:StandardNameReview)
            RETURN source_id,
                   sn.id AS sn_id,
                   r.score AS score,
                   r.comments AS full_comment,
                   r.model AS model,
                   r.reviewed_at AS ts
            ORDER BY source_id, ts DESC
            """,
            ids=ids,
        )

    # Group rows by source_id
    from collections import defaultdict

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = row.get("source_id")
        if sid:
            by_source[sid].append(row)

    mapping: dict[str, dict[str, Any]] = {}
    for sid, reviews in by_source.items():
        if not reviews:
            continue

        # Latest review
        latest = reviews[0]
        comment_text = latest.get("full_comment") or ""
        history: dict[str, Any] = {
            "latest": {
                "score": latest.get("score"),
                "comment": comment_text[:800],
                "model": latest.get("model"),
            },
            "prior_themes": [],
        }

        # Extract themes from older reviews (N-1)
        if len(reviews) > 1:
            prior_comments = [
                r.get("full_comment") or ""
                for r in reviews[1:]
                if r.get("full_comment")
            ]
            if prior_comments:
                from imas_codex.standard_names.review.themes import (
                    _extract_themes_from_texts,
                )

                themes = _extract_themes_from_texts(prior_comments, top_n=5)
                # Build theme entries with count and example
                for theme in themes:
                    # Find matching example comment (first containing theme words)
                    example = ""
                    theme_words = set(theme.lower().split())
                    for c in prior_comments:
                        c_lower = c.lower()
                        if any(w in c_lower for w in theme_words):
                            example = c[:160]
                            break
                    history["prior_themes"].append(
                        {"theme": theme, "count": 1, "example": example}
                    )

        mapping[sid] = history

    return mapping


def fetch_docs_review_feedback_for_standard_names(
    sn_ids: list[str] | set[str] | None,
) -> dict[str, dict[str, Any]]:
    """Fetch prior docs-axis reviewer feedback keyed by StandardName id.

    Used by the enrich contextualise worker when re-enriching SNs that
    already have a docs-axis review on file.  Lets the LLM target the
    specific weaknesses (description quality, documentation_quality,
    completeness, physics_accuracy) the reviewer flagged previously
    instead of running blind.

    This is the docs-axis analogue of
    :func:`fetch_review_feedback_for_sources` — it keys on SN id (since
    enrich operates on SN nodes directly, not on sources) and pulls the
    ``*_docs`` reviewer fields plus ``validation_issues``.

    Args:
        sn_ids: Iterable of StandardName ids. ``None`` or empty returns
            ``{}`` without hitting the graph.

    Returns:
        ``{sn_id: feedback_dict}`` where feedback_dict has keys:

        - ``reviewer_score`` (float | None): docs-axis 0–1 score.
        - ``reviewer_comments`` (str | None): free-form docs critique.
        - ``reviewer_scores`` (dict | None): parsed docs-axis dimensional
          scores (description_quality / documentation_quality /
          completeness / physics_accuracy).
        - ``validation_issues`` (list[str] | None): tagged ISN validation
          issue strings, when present.

        Only SNs with ``reviewer_score_docs IS NOT NULL`` are returned —
        cold-start enrichments are silently omitted.
    """
    if not sn_ids:
        return {}

    ids = sorted({sid for sid in sn_ids if sid})
    if not ids:
        return {}

    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sn_id
            MATCH (sn:StandardName {id: sn_id})
            WHERE sn.reviewer_score_docs IS NOT NULL
            RETURN sn_id AS sn_id,
                   sn.reviewer_score_docs AS reviewer_score,
                   sn.reviewer_comments_docs AS reviewer_comments,
                   sn.reviewer_scores_docs AS reviewer_scores_json,
                   sn.validation_issues AS validation_issues
            """,
            ids=ids,
        )

    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("sn_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores_dict: dict[str, Any] | None = None
        if scores_json:
            try:
                scores_dict = json.loads(scores_json)
            except (TypeError, ValueError):
                scores_dict = None
        issues = row.get("validation_issues")
        if isinstance(issues, str):
            issues = [issues]
        elif issues is None:
            issues = None
        mapping[sid] = {
            "reviewer_score": row.get("reviewer_score"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores_dict,
            "validation_issues": issues if issues else None,
        }
    return mapping


# Plan 40 §17 — public rename. Canonical name is
# ``fetch_docs_review_feedback_for_standard_names`` (above); the legacy
# ``fetch_docs_review_feedback_for_sns`` alias is retained for one
# release with a DeprecationWarning. Phase 4 callsite migration has
# moved package-internal callers to the canonical name.
def fetch_docs_review_feedback_for_sns(
    sn_ids: list[str] | set[str] | None,
) -> dict[str, dict[str, Any]]:
    """Deprecated alias of :func:`fetch_docs_review_feedback_for_standard_names`."""
    import warnings

    warnings.warn(
        "fetch_docs_review_feedback_for_sns is deprecated; "
        "use fetch_docs_review_feedback_for_standard_names.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fetch_docs_review_feedback_for_standard_names(sn_ids)


def _filter_admissible_parents(
    co_batch: list[dict[str, Any]],
    gc: Any,
    *,
    full_rebuild: bool = False,
) -> list[dict[str, Any]]:
    """Drop HAS_PARENT edges whose target fails the admission gate.

    Two-clause gate (see parents.py D1):
      A. IR specificity — keep if the target name has qualifiers, locus,
         operators, projection, or mechanism in its ISN parse.
      B. Vector-like topology — keep bare-base targets if they have or
         will have ≥2 distinct-axis projection children. Considers
         both already-in-graph projections and the current batch.

    Skips legacy entries: if a parent has ``origin='catalog_edit'`` or
    is already ``name_stage='accepted'``, the edge is preserved
    regardless of admission verdict (catalog is authoritative).

    ``full_rebuild`` (set by ``rederive_structural_edges``, which passes the
    COMPLETE set of live names) makes admission **batch-authoritative**: Clause
    B axes and the single-child shadow veto are evaluated from the batch alone,
    ignoring transient graph topology. This is essential for idempotency — the
    stale, pre-reconcile graph children would otherwise flip the shadow-veto
    verdict for qualifier single-child parents (admitted one startup, vetoed
    and reaped the next), an unbounded mint/reap oscillation. With the batch as
    the sole source of truth, the verdict is a pure function of the live-name
    set, so the re-derivation reaches a fixpoint.
    """
    from imas_codex.standard_names.parents import is_admissible_parent_name

    if not co_batch:
        return co_batch

    targets = {e["to_name"] for e in co_batch}

    # Count batch-level projection axes per target (for Clause B with batch info)
    batch_axes: dict[str, set[str]] = {}
    # Distinct children proposed for each target within THIS batch.  Combined
    # with graph children so the single-child shadow veto sees the full child
    # set even before any edge is persisted.
    batch_children: dict[str, set[str]] = {}
    for e in co_batch:
        if e.get("operator_kind") == "projection" and e.get("axis"):
            batch_axes.setdefault(e["to_name"], set()).add(e["axis"])
        if e.get("from_name"):
            batch_children.setdefault(e["to_name"], set()).add(e["from_name"])

    # Bulk topology probe: pull existing graph state for all targets at once.
    # Besides projection axes (Clause B) and legacy protection, collect each
    # target's distinct graph children and, when it has exactly one child,
    # that child's DD sources vs the parent's own DD sources — the inputs to
    # the single-child shadow veto (Class-B duplicate suppression).
    graph_axes: dict[str, set[str]] = {t: set() for t in targets}
    graph_children: dict[str, set[str]] = {t: set() for t in targets}
    shadow_info: dict[str, dict[str, Any]] = {}
    legacy_protected: set[str] = set()
    try:
        rows = list(
            gc.query(
                """
                UNWIND $names AS nm
                OPTIONAL MATCH (child:StandardName)-[r:HAS_PARENT]->(p:StandardName {id: nm})
                  WHERE r.operator_kind = 'projection' AND r.axis IS NOT NULL
                WITH nm, collect(DISTINCT r.axis) AS axes
                OPTIONAL MATCH (anychild:StandardName)-[:HAS_PARENT]->(p3:StandardName {id: nm})
                WITH nm, axes, collect(DISTINCT anychild) AS children
                OPTIONAL MATCH (psrc:IMASNode)-[:HAS_STANDARD_NAME]->(p4:StandardName {id: nm})
                WITH nm, axes, children, collect(DISTINCT psrc.id) AS parent_sources
                OPTIONAL MATCH (csrc:IMASNode)-[:HAS_STANDARD_NAME]->(lone)
                  WHERE size(children) = 1 AND lone = children[0]
                WITH nm, axes, children, parent_sources,
                     collect(DISTINCT csrc.id) AS lone_child_sources
                OPTIONAL MATCH (p2:StandardName {id: nm})
                RETURN nm AS name, axes,
                       [c IN children | c.id] AS child_ids,
                       CASE WHEN size(children) = 1
                            THEN children[0].id ELSE null END AS lone_child_id,
                       CASE WHEN size(children) = 1
                            THEN coalesce(children[0].name_stage, '')
                            ELSE null END AS lone_child_stage,
                       CASE WHEN size(children) = 1
                            THEN coalesce(children[0].origin, 'pipeline')
                            ELSE null END AS lone_child_origin,
                       parent_sources,
                       lone_child_sources,
                       p2.origin AS origin,
                       p2.name_stage AS name_stage
                """,
                names=list(targets),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Parent-admission topology probe failed: %s; admitting all", exc)
        return co_batch

    for r in rows:
        nm = r.get("name")
        if not nm:
            continue
        graph_axes[nm] = set(r.get("axes") or [])
        graph_children[nm] = {c for c in (r.get("child_ids") or []) if c}
        if r.get("origin") == "catalog_edit" or r.get("name_stage") == "accepted":
            legacy_protected.add(nm)
        if r.get("lone_child_id"):
            shadow_info[nm] = {
                "child_id": r.get("lone_child_id"),
                "child_stage": r.get("lone_child_stage") or "",
                "child_origin": r.get("lone_child_origin") or "pipeline",
                "child_sources": {s for s in (r.get("lone_child_sources") or []) if s},
                "parent_sources": {s for s in (r.get("parent_sources") or []) if s},
            }

    if full_rebuild:
        # Batch-authoritative admission. The batch is the complete live
        # derivation, so discard transient graph topology (stale, pre-reconcile
        # children/axes) — using it makes the shadow-veto/Clause-B verdict
        # depend on graph state that this very pass rewrites, which oscillates.
        # Clause B then reads batch_axes only; the shadow veto reads
        # batch_children only, with each target's lone-batch-child DD sources
        # re-fetched from the static HAS_STANDARD_NAME edges (which the
        # re-derivation never touches, so the verdict is stable across passes).
        graph_axes = {t: set() for t in targets}
        graph_children = {t: set() for t in targets}
        shadow_info = {}
        lone_pairs = [
            {"target": t, "child": next(iter(ch))}
            for t, ch in batch_children.items()
            if len(ch) == 1
        ]
        if lone_pairs:
            try:
                srows = list(
                    gc.query(
                        """
                        UNWIND $pairs AS pr
                        MATCH (child:StandardName {id: pr.child})
                        OPTIONAL MATCH (csrc:IMASNode)-[:HAS_STANDARD_NAME]->(child)
                        OPTIONAL MATCH (psrc:IMASNode)-[:HAS_STANDARD_NAME]->
                              (:StandardName {id: pr.target})
                        RETURN pr.target AS target, pr.child AS child,
                               coalesce(child.name_stage, '') AS child_stage,
                               coalesce(child.origin, 'pipeline') AS child_origin,
                               collect(DISTINCT csrc.id) AS child_sources,
                               collect(DISTINCT psrc.id) AS parent_sources
                        """,
                        pairs=lone_pairs,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Batch shadow-source probe failed: %s; skipping veto", exc
                )
                srows = []
            for r in srows:
                shadow_info[r["target"]] = {
                    "child_id": r["child"],
                    "child_stage": r.get("child_stage") or "",
                    "child_origin": r.get("child_origin") or "pipeline",
                    "child_sources": {s for s in (r.get("child_sources") or []) if s},
                    "parent_sources": {s for s in (r.get("parent_sources") or []) if s},
                }

    def _is_single_child_shadow(target: str) -> tuple[bool, str]:
        """Batch-aware single-child shadow veto (mirrors parents.is_single_child_shadow).

        Combines graph children with the children proposed in THIS batch so a
        parent that genuinely groups ≥2 names is never vetoed, even before the
        sibling edges are written.
        """
        all_children = graph_children.get(target, set()) | batch_children.get(
            target, set()
        )
        if len(all_children) != 1:
            return False, "not a single-child parent"
        info = shadow_info.get(target)
        if not info:
            # Lone child exists only in this batch (not yet in graph) — no DD
            # source attached yet, so source-equivalence cannot be asserted.
            return False, "single batch child not yet sourced"
        if info["child_stage"] in ("superseded", "exhausted"):
            return False, "lone child superseded/exhausted"
        if info["child_origin"] == "derived":
            return False, "lone child is itself derived"
        child_sources = info["child_sources"]
        parent_sources = info["parent_sources"]
        if not child_sources:
            return False, "lone child has no DD source"
        if parent_sources - child_sources:
            return False, "parent independently sourced — not a shadow"
        return True, f"single-child shadow of {info['child_id']}"

    def _admit(target: str) -> tuple[bool, str]:
        # Legacy protection: don't disturb catalog-authoritative entries.
        if target in legacy_protected:
            return True, "legacy (catalog_edit or accepted)"
        # Suppression veto (Class-B): a less-specific shadow of a single
        # accepted sibling sourced from the same DD path is dropped even if it
        # would otherwise admit on Clause A/B.
        suppress, suppress_reason = _is_single_child_shadow(target)
        if suppress:
            return False, f"suppressed: {suppress_reason}"
        result = is_admissible_parent_name(target, gc=None)
        if result.admit:
            return True, f"clause A: {result.reason}"
        # Clause B with batch + graph topology
        combined = graph_axes.get(target, set()) | batch_axes.get(target, set())
        if len(combined) >= 2:
            return True, f"clause B: vector-like ({len(combined)} axes)"
        return False, result.reason

    kept: list[dict[str, Any]] = []
    dropped_count = 0
    dropped_targets: set[str] = set()
    for e in co_batch:
        admit, reason = _admit(e["to_name"])
        if admit:
            kept.append(e)
        else:
            dropped_count += 1
            dropped_targets.add(e["to_name"])

    if dropped_count:
        logger.info(
            "Parent admission gate: dropped %d HAS_PARENT edges to %d "
            "inadmissible targets (sample: %s)",
            dropped_count,
            len(dropped_targets),
            sorted(dropped_targets)[:5],
        )

    return kept


def _emit_magnitude_of_edges(names: list[dict[str, Any]], gc: Any) -> None:
    """Emit MAGNITUDE_OF edges for ``magnitude_of_<X>`` names.

    Passive — only fires when:
      1. The name being written matches the pattern ``magnitude_of_<X>``.
      2. ``<X>`` already exists in the graph as a ``kind='vector'`` SN.
      3. ``<X>`` is an admitted parent (not catalog-only metadata).

    Never speculatively creates magnitude SNs; the magnitude must already
    be sourced from a DD path or facility signal. See plan D5/D13.
    """
    candidates: list[dict[str, str]] = []
    for n in names:
        name_id = n.get("id")
        if not name_id or not name_id.startswith("magnitude_of_"):
            continue
        vector = name_id[len("magnitude_of_") :]
        if not vector:
            continue
        candidates.append({"magnitude": name_id, "vector": vector})

    if not candidates:
        return

    gc.query(
        """
        UNWIND $rows AS r
        MATCH (m:StandardName {id: r.magnitude})
        MATCH (v:StandardName {id: r.vector})
        WHERE v.kind = 'vector'
        MERGE (m)-[:MAGNITUDE_OF]->(v)
        """,
        rows=candidates,
    )


def _write_standard_name_edges(
    gc: Any, names: list[dict[str, Any]], *, full_rebuild: bool = False
) -> None:
    """Emit all structural edges for a batch of StandardName nodes.

    Called as a tail pass **after** all nodes in the batch have been
    MERGEd.  Forward-reference targets are MERGEd as bare placeholder
    ``StandardName`` nodes so the edge can be created immediately; their
    full properties arrive in the same batch, a later batch, or via
    catalog import.

    Handles the following edge types:

    - ``HAS_PARENT``: derived from the ISN grammar parser (one layer
      per call: unary prefix/postfix, binary, or projection).
    - ``HAS_ERROR``: uncertainty siblings — direction inverted relative to
      the derivation (inner → uncertainty form).
    - ``HAS_PREDECESSOR``: from ``predecessor`` or ``deprecates`` field.
    - ``HAS_SUCCESSOR``: from ``successor`` or ``superseded_by`` field.
    - ``IN_CLUSTER``: from ``primary_cluster_id`` field.
    - ``HAS_PHYSICS_DOMAIN``: from ``physics_domain`` field.

    Parameters
    ----------
    gc:
        Active ``GraphClient`` context (already open — do not open a new
        one inside this function).
    names:
        List of name dicts, each containing at minimum ``id``.  All other
        fields are optional; missing fields produce no edges.
    """
    from imas_codex.standard_names.derivation import derive_edges

    co_batch: list[dict[str, Any]] = []  # HAS_PARENT
    he_batch: list[dict[str, Any]] = []  # HAS_ERROR
    geo_batch: list[dict[str, Any]] = []  # HAS_LOCUS

    for n in names:
        name_id = n.get("id")
        if not name_id:
            continue
        for edge in derive_edges(name_id):
            if edge.edge_type == "HAS_PARENT":
                co_batch.append(
                    {
                        "from_name": edge.from_name,
                        "to_name": edge.to_name,
                        "operator": edge.props.get("operator"),
                        "operator_kind": edge.props.get("operator_kind"),
                        "role": edge.props.get("role"),
                        "separator": edge.props.get("separator"),
                        "axis": edge.props.get("axis"),
                        "shape": edge.props.get("shape"),
                    }
                )
            elif edge.edge_type == "HAS_ERROR":
                he_batch.append(
                    {
                        "from_name": edge.from_name,
                        "to_name": edge.to_name,
                        "error_type": edge.props.get("error_type"),
                    }
                )
            elif edge.edge_type == "HAS_LOCUS":
                geo_batch.append(
                    {
                        "from_name": edge.from_name,
                        "locus_token": edge.props.get("locus_token"),
                        "locus_relation": edge.props.get("locus_relation"),
                    }
                )

    # Phase 1 admission gate: drop HAS_PARENT edges to inadmissible
    # parents (bare-base category labels). See parents.py D1/D2 of the
    # deterministic-parent redesign plan.
    co_batch = _filter_admissible_parents(co_batch, gc, full_rebuild=full_rebuild)

    # Reconcile (NOT accrete) each processed child's structural HAS_PARENT
    # edges to EXACTLY the current admitted derivation. ``derive_edges`` is a
    # deterministic function of the child name, but the rule that computes the
    # structural parent has changed over the pipeline's life; MERGE-only writes
    # left a child pointing at a parent the CURRENT derivation no longer emits
    # (the ``toroidal_mhd_mode_number`` / diffusivity tangles). Without this
    # step every startup keeps the stale edge alive and, when the rule moved a
    # child to a new parent, orphans the old parent into a childless zombie —
    # the source of the derived-parent churn. Deleting structural HAS_PARENT
    # edges (those carrying an ``operator_kind`` — all HAS_PARENT edges are
    # derivation-produced) whose target is not in the freshly admitted set
    # makes the edge topology a pure function of the current grammar. Runs for
    # EVERY child in this batch, including those whose derivation now yields no
    # admissible parent (so their last stale edge is cleared).
    desired_by_child: dict[str, set[str]] = {}
    for e in co_batch:
        if e.get("from_name"):
            desired_by_child.setdefault(e["from_name"], set()).add(e["to_name"])
    recon = [
        {"child": n["id"], "keep": sorted(desired_by_child.get(n["id"], set()))}
        for n in names
        if n.get("id")
    ]
    if recon:
        gc.query(
            """
            UNWIND $recon AS rc
            MATCH (c:StandardName {id: rc.child})-[r:HAS_PARENT]->(p:StandardName)
            WHERE r.operator_kind IS NOT NULL AND NOT (p.id IN rc.keep)
            DELETE r
            """,
            recon=recon,
        )

    if co_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (tgt:StandardName {id: b.to_name})
            ON CREATE SET tgt.origin = 'derived',
                          tgt.name_stage = 'pending'
            MERGE (src)-[r:HAS_PARENT]->(tgt)
            SET r.operator      = b.operator,
                r.operator_kind = b.operator_kind,
                r.role          = b.role,
                r.separator     = b.separator,
                r.axis          = b.axis,
                r.shape         = b.shape
            WITH tgt, b.operator_kind AS ok
            WHERE tgt.name_stage IS NULL OR tgt.name_stage = 'pending'
            WITH tgt, ok
            WHERE ok IN ['projection', 'coordinate', 'unary_postfix']
            SET tgt.needs_composition = true
            """,
            batch=co_batch,
        )

        # Phase 3: MAGNITUDE_OF passive materialisation. When a written
        # name matches "magnitude_of_<X>" and X is an admitted vector
        # parent in the graph, emit (this)-[:MAGNITUDE_OF]->(X). Never
        # create the magnitude node speculatively — only links existing.
        _emit_magnitude_of_edges(names, gc)

    if he_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (tgt:StandardName {id: b.to_name})
            MERGE (src)-[r:HAS_ERROR]->(tgt)
            SET r.error_type = b.error_type
            """,
            batch=he_batch,
        )

    if geo_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (loc:Locus {id: b.locus_token})
            MERGE (src)-[r:HAS_LOCUS]->(loc)
            SET r.locus_token = b.locus_token,
                r.locus_relation = b.locus_relation
            """,
            batch=geo_batch,
        )

    # --- HAS_PREDECESSOR / HAS_SUCCESSOR ---
    # Support both 'predecessor'/'successor' (pipeline) and
    # 'deprecates'/'superseded_by' (catalog import).
    pred_batch: list[dict[str, str]] = []
    succ_batch: list[dict[str, str]] = []
    for n in names:
        name_id = n.get("id")
        if not name_id:
            continue
        predecessor = n.get("predecessor") or n.get("deprecates")
        if predecessor:
            pred_batch.append({"from_name": name_id, "to_name": predecessor})
        successor = n.get("successor") or n.get("superseded_by")
        if successor:
            succ_batch.append({"from_name": name_id, "to_name": successor})

    if pred_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (tgt:StandardName {id: b.to_name})
            MERGE (src)-[:HAS_PREDECESSOR]->(tgt)
            """,
            batch=pred_batch,
        )

    if succ_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (tgt:StandardName {id: b.to_name})
            MERGE (src)-[:HAS_SUCCESSOR]->(tgt)
            """,
            batch=succ_batch,
        )

    # --- IN_CLUSTER ---
    cluster_batch = [
        {"sn_id": n["id"], "cluster_id": n["primary_cluster_id"]}
        for n in names
        if n.get("id") and n.get("primary_cluster_id")
    ]
    if cluster_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.sn_id})
            MERGE (c:IMASSemanticCluster {id: b.cluster_id})
            MERGE (sn)-[:IN_CLUSTER]->(c)
            """,
            batch=cluster_batch,
        )

    # --- HAS_PHYSICS_DOMAIN (one edge per source domain) ---
    # Post-refactor: ``physics_domain`` is the scalar primary; the full
    # set of contributing domains lives in ``source_domains``.  Edges are
    # MERGEd from ``source_domains`` (plus the scalar primary as a
    # fallback for legacy callers) so that cross-domain discoverability
    # is preserved at the graph level.
    domain_batch = []
    for n in names:
        if not n.get("id"):
            continue
        domains: set[str] = set(_ensure_list(n.get("source_domains")))
        primary = _scalar_domain(n.get("physics_domain"))
        if primary:
            domains.add(primary)
        for d in domains:
            domain_batch.append({"sn_id": n["id"], "domain_id": d})
    if domain_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.sn_id})
            MERGE (d:PhysicsDomain {id: b.domain_id})
            MERGE (sn)-[:HAS_PHYSICS_DOMAIN]->(d)
            """,
            batch=domain_batch,
        )


def rederive_structural_edges() -> dict[str, int]:
    """Reconcile HAS_PARENT/HAS_ERROR edges to the current derivation of LIVE names.

    Structure is defined by **live** names only. Superseded/exhausted names are
    dead (refined away) and must not contribute structural parents: deriving
    edges from a dead name re-mints a parent that the childless reaper then
    deletes — an unbounded mint/reap churn across startups. So this:

    1. derives + reconciles edges for non-superseded/-exhausted names only
       (``_write_standard_name_edges`` deletes each child's stale derived edges
       and writes the current admitted set — idempotent);
    2. deletes any derived (``operator_kind``-bearing) HAS_PARENT edge still
       originating FROM a superseded/exhausted name, so dead names stop
       propping up zombie parents (those parents become reapable, then drain);
    3. migrates inbound HAS_PARENT edges off superseded PARENTS to their live
       successor (``_rewire_has_parent_off_superseded``, which skips
       projection-of-self successors).

    The result is a fixpoint: re-running on a stable live name set mints no new
    parents and orphans none. Returns processed/dead-edge/migrated counts.
    """
    with GraphClient() as gc:
        live = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE NOT (coalesce(sn.name_stage, '') IN ['superseded', 'exhausted', 'contested']) "
            "RETURN sn.id AS id"
        )
        names = [{"id": r["id"]} for r in live if r.get("id")]
        if not names:
            return {"processed": 0, "dead_edges_cleared": 0, "migrated": 0}
        _write_standard_name_edges(gc, names, full_rebuild=True)
        dead = gc.query(
            """
            MATCH (c:StandardName)-[r:HAS_PARENT]->(:StandardName)
            WHERE coalesce(c.name_stage, '') IN ['superseded', 'exhausted']
              AND r.operator_kind IS NOT NULL
            DELETE r
            RETURN count(r) AS n
            """
        )
        dead_cleared = int(dead[0]["n"]) if dead else 0
        migrated = _rewire_has_parent_off_superseded(gc)
    logger.info(
        "rederive_structural_edges: processed %d live names, cleared %d "
        "dead-name edges, migrated %d HAS_PARENT edges",
        len(names),
        dead_cleared,
        migrated,
    )
    return {
        "processed": len(names),
        "dead_edges_cleared": dead_cleared,
        "migrated": migrated,
    }


def _rewire_has_parent_off_superseded(gc: Any) -> int:
    """Rewire HAS_PARENT edges from superseded parents to their successors.

    A name that supersedes another carries a ``REFINED_FROM`` edge back
    to its predecessor. We walk the chain to find the live tip (the SN
    at the head whose ``name_stage`` is not ``superseded``) and migrate
    each inbound ``HAS_PARENT`` edge to that tip, preserving edge
    properties. If no live tip exists (every node on the chain is
    superseded — should not happen in practice), the edge is left
    alone so manual review can investigate.

    Self-loop guard: if a refine step renamed the *child* into the same
    name as its (now-superseded) parent — e.g. ``minimum_magnetic_field``
    got refined into ``minimum_magnetic_field_magnitude`` while
    ``minimum_magnetic_field_magnitude`` was already its HAS_PARENT
    parent — naive migration produces ``child -[HAS_PARENT]-> child``,
    a structural impossibility that crashes ISNC's
    ``validate_catalog`` topological sort with
    ``CycleError: nodes are in a cycle, ['x', 'x']``. In that case the
    edge is deleted outright; the new structural parent (if any) will
    be re-emitted next time ``derive_edges`` runs.

    Returns the number of HAS_PARENT edges actually migrated (delete-
    only self-loops are reported under ``deleted_self_loops`` in the
    logger but excluded from the migrated count).
    """
    # The live successor is the chain-head: the node reachable from
    # ``old`` by following ``<-[:REFINED_FROM]-`` zero or more times,
    # whose own ``name_stage`` is not 'superseded'. Cypher's variable-
    # length pattern returns paths; we pick the longest hop count to
    # land at the tip of the chain.
    result = list(
        gc.query(
            """
            MATCH (old:StandardName)
            WHERE old.name_stage = 'superseded'
              AND EXISTS { (child)-[:HAS_PARENT]->(old) }
            MATCH path = (tip:StandardName)-[:REFINED_FROM*1..]->(old)
            WHERE tip.name_stage <> 'superseded'
              // Migrate ONLY to a true rename of `old` — a successor with the
              // SAME identity segments (physical/geometric base, subject,
              // component). A REFINED_FROM "successor" that ADDS a qualifier/
              // projection or CHANGES the subject (e.g. ion_convection_velocity
              // -> radial_electron_convection_velocity, or major_radius ->
              // major_radius_of_flux_surface) is a more-specific CHILD, not a
              // rename; migrating `old`'s children onto it recreates the
              // domain/locus-mismatched tangle this function exists to repair.
              // The correct remedy for those is to un-supersede `old` (it is the
              // legitimate generic parent), so leave the edge in place here.
              // Conservative on NULL columns: if identity cannot be confirmed
              // equal, do not migrate.
              AND coalesce(tip.physical_base, '')  = coalesce(old.physical_base, '')
              AND coalesce(tip.geometric_base, '') = coalesce(old.geometric_base, '')
              AND coalesce(tip.subject, '')        = coalesce(old.subject, '')
              AND coalesce(tip.component, '')       = coalesce(old.component, '')
            WITH old, tip, length(path) AS hops
            ORDER BY hops DESC
            WITH old, head(collect(tip)) AS tip
            WHERE tip IS NOT NULL
            MATCH (child)-[c:HAS_PARENT]->(old)
            WITH tip, child, properties(c) AS props, c
            DELETE c
            // Drop the edge if migration would produce a self-loop;
            // any legitimate structural parent for `child` will be
            // re-emitted by the next derive_edges pass.
            WITH tip, child, props
            WHERE tip.id <> child.id
            MERGE (child)-[c_new:HAS_PARENT]->(tip)
            SET c_new = props
            RETURN count(c_new) AS migrated
            """
        )
    )
    if not result:
        return 0
    return int(result[0].get("migrated") or 0)


def _parse_parent_grammar(name_id: str) -> dict[str, str | None]:
    """Attempt ISN parse on a parent name to extract grammar fields.

    Returns a dict with bare-name segment keys. On parse/model failure, all
    values are None. Uses the same ``parse_standard_name`` +
    ``_segments_from_model`` authority as ``_parse_grammar`` and
    ``_write_grammar_decomposition``.
    """
    try:
        from imas_standard_names.grammar import parse_standard_name

        return _segments_from_model(parse_standard_name(name_id))
    except ImportError:
        logger.debug("ISN unavailable — no grammar for parent %s", name_id)
    except Exception:  # noqa: BLE001
        logger.debug("ISN model rejected parent %s", name_id)
    return dict.fromkeys(_GRAMMAR_SEGMENT_COLUMNS)


def _query_seedable_derived_parents(
    gc: Any,
    *,
    state_where: str,
) -> list[dict[str, Any]]:
    """Return structurally seedable derived-parent candidates."""
    return list(
        gc.query(
            f"""
            MATCH (parent:StandardName)
            WHERE (parent.origin IS NULL OR parent.origin = 'derived')
              AND ({state_where})
            MATCH (child)-[comp:HAS_PARENT]->(parent)
            WHERE comp.operator_kind IN
                  ['projection', 'coordinate', 'unary_postfix']
            WITH parent,
                 collect(DISTINCT child) AS children
            MATCH (any_child)-[any_comp:HAS_PARENT]->(parent)
            WITH parent, children,
                 collect(any_comp.operator_kind) AS edge_kinds,
                 count(any_child) AS total_children,
                 count(CASE WHEN any_child.name_stage IS NOT NULL
                       THEN 1 END) AS composed_children
            WHERE total_children = composed_children
              AND total_children >= 1
            UNWIND children AS child
            OPTIONAL MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(child)
            OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(imas:IMASNode)
            OPTIONAL MATCH (child)-[cedge:HAS_PARENT]->(parent)
            WITH parent, edge_kinds,
                 collect(DISTINCT {{
                     id: child.id,
                     unit: child.unit,
                     cocos: child.cocos_transformation_type,
                     physics_domain: child.physics_domain,
                     kind: child.kind,
                     op_kind: cedge.operator_kind
                 }}) AS child_data,
                 collect(DISTINCT imas.id) AS dd_paths
            RETURN parent.id AS parent_id,
                   child_data,
                   dd_paths,
                   edge_kinds
            """
        )
    )


def _query_legacy_repairable_derived_parents(
    gc: Any,
    *,
    state_where: str,
) -> list[dict[str, Any]]:
    """Return already-admitted legacy derived parents eligible for lifecycle repair."""
    return list(
        gc.query(
            f"""
            MATCH (parent:StandardName)
            WHERE (parent.origin IS NULL OR parent.origin = 'derived')
              AND ({state_where})
            MATCH (child)-[comp:HAS_PARENT]->(parent)
            WITH parent,
                 collect(DISTINCT child) AS children,
                 collect(comp.operator_kind) AS edge_kinds,
                 count(child) AS total_children,
                 count(CASE WHEN child.name_stage IS NOT NULL
                       THEN 1 END) AS composed_children,
                 count(CASE WHEN comp.operator_kind IN
                       ['projection', 'coordinate', 'unary_postfix']
                       THEN 1 END) AS seedable_edges
            WHERE total_children = composed_children
              AND total_children >= 1
              AND seedable_edges = 0
            UNWIND children AS child
            OPTIONAL MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(child)
            OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(imas:IMASNode)
            OPTIONAL MATCH (child)-[cedge:HAS_PARENT]->(parent)
            WITH parent, edge_kinds,
                 collect(DISTINCT {{
                     id: child.id,
                     unit: child.unit,
                     cocos: child.cocos_transformation_type,
                     physics_domain: child.physics_domain,
                     kind: child.kind,
                     op_kind: cedge.operator_kind
                 }}) AS child_data,
                 collect(DISTINCT imas.id) AS dd_paths
            RETURN parent.id AS parent_id,
                   child_data,
                   dd_paths,
                   edge_kinds
            """
        )
    )


def _query_accepted_derived_parents_for_cleanup(gc: Any) -> list[str]:
    """Return accepted derived-parent ids that should be re-checked for admission.

    These nodes predate the current admission gate and may have drifted into an
    exportable state even though the gate would now reject them. Restrict to
    ``origin='derived'`` so catalog-authoritative and pipeline-authored names
    are never touched by this cleanup.

    Any ``name_stage='accepted'`` derived parent is re-checked regardless of
    ``docs_stage``. A single-child-shadow parent minted in ``--names-only`` mode
    is ``name_stage='accepted'`` with ``docs_stage`` still ``drafted``/``null``;
    gating on ``docs_stage='accepted'`` let those escape the admission re-check
    even though :func:`is_admissible_parent_name` rejects them.
    """
    rows = gc.query(
        """
        MATCH (parent:StandardName)
        WHERE parent.origin = 'derived'
          AND parent.name_stage = 'accepted'
        MATCH (:StandardName)-[:HAS_PARENT]->(parent)
        RETURN DISTINCT parent.id AS parent_id
        """
    )
    return [str(r["parent_id"]) for r in rows if r.get("parent_id")]


def _delete_derived_parent_nodes(gc: Any, parent_ids: list[str]) -> int:
    """Delete derived-parent nodes and their review/derived-source scaffolding."""
    deleted = 0
    for parent_id in parent_ids:
        gc.query(
            """
            MATCH (sns:StandardNameSource {source_type: 'derived', source_id: $parent_id})
            DETACH DELETE sns
            """,
            parent_id=parent_id,
        )
        gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.produced_sn_id = $parent_id
            SET sns.produced_sn_id = null
            """,
            parent_id=parent_id,
        )
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName {id: $parent_id})
                OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(rv:StandardNameReview)
                OPTIONAL MATCH (sn)-[:DOCS_REVISION_OF]->(dr:DocsRevision)
                WITH sn, collect(DISTINCT rv) AS reviews,
                     collect(DISTINCT dr) AS revisions
                FOREACH (r IN reviews | DETACH DELETE r)
                FOREACH (d IN revisions | DETACH DELETE d)
                DETACH DELETE sn
                RETURN 1 AS deleted
                """,
                parent_id=parent_id,
            )
        )
        if rows:
            deleted += int(rows[0].get("deleted", 0))
    return deleted


def sweep_orphaned_docs_revisions(gc: Any) -> int:
    """Delete DocsRevision snapshots no longer linked to any StandardName.

    A ``DocsRevision`` is owned by exactly one ``StandardName`` via
    ``(sn)-[:DOCS_REVISION_OF]->(dr)``. When the owning name is reaped its
    revisions become unreachable scaffolding. Although ``_delete_derived_parent_nodes``
    now deletes a parent's revisions inline, this sweep is the idempotent
    backstop: it clears any revision orphaned by an earlier reap (or a future
    deletion path that forgets to). Runs every startup so orphaned docs content
    can never accumulate. Returns the number deleted.
    """
    rows = list(
        gc.query(
            """
            MATCH (dr:DocsRevision)
            WHERE NOT EXISTS { (:StandardName)-[:DOCS_REVISION_OF]->(dr) }
            WITH dr LIMIT 50000
            DETACH DELETE dr
            RETURN count(dr) AS n
            """
        )
    )
    n = int(rows[0]["n"]) if rows else 0
    if n:
        logger.info("sweep_orphaned_docs_revisions: deleted %d orphaned revisions", n)
    return n


def _derived_parent_source_metadata(
    parent_id: str,
    *,
    parent_dd_path: str | None,
) -> dict[str, str]:
    """Build a valid StandardNameSource identity for a derived parent."""
    if parent_dd_path:
        return {
            "source_node_id": f"dd:{parent_dd_path}",
            "source_type": "dd",
            "source_id": parent_dd_path,
            "batch_key": "derived_parent",
        }
    return {
        "source_node_id": f"derived:{parent_id}",
        "source_type": "derived",
        "source_id": parent_id,
        "batch_key": "derived_parent",
    }


def _materialize_derived_parent_rows(
    gc: Any,
    parents: list[dict[str, Any]],
    *,
    infer_kind_from_existing_topology: bool = False,
) -> int:
    """Materialize structurally eligible derived parents onto the docs lifecycle."""
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )
    from imas_codex.standard_names.parents import recompute_parent_kind

    seeded = 0
    for row in parents:
        parent_id = row["parent_id"]
        child_data = row.get("child_data") or []
        dd_paths = [p for p in (row.get("dd_paths") or []) if p]

        edge_kinds = row.get("edge_kinds") or []
        is_geometric = any(k == "coordinate" for k in edge_kinds)

        if infer_kind_from_existing_topology:
            kind = recompute_parent_kind(parent_id, gc)
        else:
            seedable_edge_kinds = {"projection", "coordinate", "unary_postfix"}
            if not any(k in seedable_edge_kinds for k in edge_kinds):
                logger.debug(
                    "Parent %s has only non-seedable edges (%s) — skipping",
                    parent_id,
                    edge_kinds,
                )
                continue

            is_magnitude_parent = all(k == "unary_postfix" for k in edge_kinds)
            kind = "scalar" if is_magnitude_parent else "vector"

        # Binary-operand children (e.g. ``ratio_of_X_to_Y``) carry a HAS_PARENT
        # edge to each operand, so the parent is an OPERAND of the ratio, not a
        # unit-sharing generalization of it. A dimensionless ratio child must
        # NOT constrain the parent's unit/cocos — otherwise a parent whose
        # qualifier/projection children all share a unit (e.g. m^-3) is wrongly
        # flagged heterogeneous by the ratio's '1' and skipped. Exclude binary
        # children from unit/cocos inheritance.
        unit_children = [c for c in child_data if c.get("op_kind") != "binary"]
        # Normalization-peel children must not constrain the parent's unit
        # either: a child carrying a normalization marker the parent lacks
        # (``normalized_particle_mass`` → ``particle_mass``) is the
        # dimensionless variant of the parent's PHYSICAL concept, so its
        # unit '1' is correct for the child and wrong for the parent. With
        # no other unit signal the parent stays unit-less rather than
        # inheriting a dimensionless stamp it cannot honour.
        _norm_markers = ("normalized", "normalised")
        if not any(m in parent_id.split("_") for m in _norm_markers):
            unit_children = [
                c
                for c in unit_children
                if not any(m in (c.get("id") or "").split("_") for m in _norm_markers)
            ]
        child_units = {c["unit"] for c in unit_children if c.get("unit")}
        if is_geometric:
            unit = None
        elif len(child_units) > 1:
            logger.warning(
                "Parent %s has children with heterogeneous units: %s — "
                "skipping (needs manual review)",
                parent_id,
                child_units,
            )
            continue
        else:
            unit = next(iter(child_units)) if child_units else None

        child_cocos = {c["cocos"] for c in unit_children if c.get("cocos")}
        cocos = next(iter(child_cocos)) if len(child_cocos) == 1 else None

        # Inherit the children's physics domain(s). A parent whose children
        # span MULTIPLE domains (e.g. torque across transport + mhd) must still
        # be scoped — pick the primary (most-frequent child domain, ties broken
        # alphabetically for determinism) and record the full provenance set in
        # source_domains. Leaving it None would export the parent as 'unscoped'.
        from collections import Counter

        child_domain_list = [
            c["physics_domain"] for c in child_data if c.get("physics_domain")
        ]
        if child_domain_list:
            _domain_counts = Counter(child_domain_list)
            _top = max(_domain_counts.values())
            physics_domain = sorted(d for d, n in _domain_counts.items() if n == _top)[
                0
            ]
            source_domains = sorted(set(child_domain_list))
        else:
            physics_domain = None
            source_domains = []

        grammar_fields = _parse_parent_grammar(parent_id)

        parent_dd_path = None
        if dd_paths:
            parts = [p.split("/") for p in dd_paths]
            common = []
            for segments in zip(*parts, strict=False):
                if len(set(segments)) == 1:
                    common.append(segments[0])
                else:
                    break
            if common:
                parent_dd_path = "/".join(common)

        props: dict[str, Any] = {
            "parent_id": parent_id,
            "unit": unit,
            "cocos_transformation_type": cocos,
            "physics_domain": physics_domain,
            "source_domains": source_domains,
            "kind": kind,
            "description": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
            "is_geometric_coordinate": is_geometric,
            "dd_path": parent_dd_path,
        }
        props.update(grammar_fields)
        props.update(
            _derived_parent_source_metadata(
                parent_id,
                parent_dd_path=parent_dd_path,
            )
        )

        gc.query(
            """
            MATCH (parent:StandardName {id: $parent_id})
            SET parent.name_stage = CASE
                    // Already name-reviewed → stays accepted (idempotent).
                    WHEN parent.reviewer_score_name IS NOT NULL THEN 'accepted'
                    // Unreviewed but review-ready (real description + embedding)
                    // → route through REVIEW_NAME before it can earn docs.
                    WHEN trim(coalesce(parent.description, '')) <> ''
                         AND parent.description <> $description
                         AND parent.embedding IS NOT NULL THEN 'drafted'
                    // Unreviewed with only the placeholder description → hold at
                    // accepted; the docs gate (reviewer_score_name IS NOT NULL)
                    // keeps it out of docs until enrichment + review fill it in.
                    ELSE 'accepted'
                END,
                parent.docs_stage = coalesce(parent.docs_stage, 'pending'),
                parent.validation_status = coalesce(parent.validation_status, 'valid'),
                parent.origin = 'derived',
                parent.kind = $kind,
                parent.unit = coalesce($unit, parent.unit),
                parent.cocos_transformation_type =
                    coalesce($cocos_transformation_type,
                             parent.cocos_transformation_type),
                parent.physics_domain =
                    coalesce($physics_domain, parent.physics_domain),
                parent.source_domains = CASE
                    WHEN $source_domains IS NOT NULL AND size($source_domains) > 0
                    THEN $source_domains ELSE parent.source_domains
                END,
                parent.description = CASE
                    WHEN trim(coalesce(parent.description, '')) = ''
                    THEN $description
                    ELSE parent.description
                END,
                parent.needs_composition = null,
                parent.chain_length = coalesce(parent.chain_length, 0),
                parent.docs_chain_length = coalesce(parent.docs_chain_length, 0),
                parent.physical_base =
                    coalesce($physical_base, parent.physical_base),
                parent.geometric_base =
                    coalesce($geometric_base, parent.geometric_base),
                parent.subject =
                    coalesce($subject, parent.subject),
                parent.transformation =
                    coalesce($transformation, parent.transformation),
                parent.component =
                    coalesce($component, parent.component),
                parent.coordinate =
                    coalesce($coordinate, parent.coordinate),
                parent.position =
                    coalesce($position, parent.position),
                parent.process =
                    coalesce($process, parent.process),
                parent.device =
                    coalesce($device, parent.device),
                parent.region =
                    coalesce($region, parent.region),
                parent.aggregation =
                    coalesce($aggregation, parent.aggregation),
                parent.orbit =
                    coalesce($orbit, parent.orbit),
                parent.population =
                    coalesce($population, parent.population),
                parent.object =
                    coalesce($object, parent.object),
                parent.geometry =
                    coalesce($geometry, parent.geometry),
                parent.is_geometric_coordinate =
                    coalesce(parent.is_geometric_coordinate,
                             $is_geometric_coordinate)
            WITH parent
            MERGE (sns:StandardNameSource {id: $source_node_id})
            ON CREATE SET sns.created_at = datetime(),
                          sns.attempt_count = 0
            SET sns.status = 'composed',
                sns.source_type = $source_type,
                sns.source_id = $source_id,
                sns.batch_key = coalesce(sns.batch_key, $batch_key),
                sns.description = CASE
                    WHEN trim(coalesce(sns.description, '')) = ''
                    THEN parent.description
                    ELSE sns.description
                END,
                sns.physics_domain = coalesce(parent.physics_domain, sns.physics_domain),
                sns.composed_at = coalesce(sns.composed_at, datetime()),
                sns.produced_sn_id = parent.id,
                sns.claimed_at = null,
                sns.claim_token = null
            MERGE (sns)-[:PRODUCED_NAME]->(parent)
            """,
            **props,
        )

        if parent_dd_path:
            gc.query(
                """
                MATCH (sns:StandardNameSource {id: $source_node_id})
                MATCH (imas:IMASNode {id: $dd_path})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
                """,
                source_node_id=props["source_node_id"],
                dd_path=parent_dd_path,
            )

        if unit:
            # Self-heal: drop any pre-existing HAS_UNIT edge (possibly to a
            # different Unit) before re-creating, so a unit correction replaces
            # the edge rather than leaving a stale one alongside the new one.
            # An SN must carry at most one HAS_UNIT edge, matching sn.unit.
            gc.query(
                """
                MATCH (sn:StandardName {id: $parent_id})
                OPTIONAL MATCH (sn)-[r:HAS_UNIT]->(:Unit)
                DELETE r
                WITH sn
                MERGE (u:Unit {id: $unit})
                MERGE (sn)-[:HAS_UNIT]->(u)
                """,
                parent_id=parent_id,
                unit=unit,
            )

        seeded += 1

    return seeded


def repair_normalization_peel_parent_units(gc: Any) -> list[str]:
    """Unset the dimensionless unit mis-inherited across a normalization peel.

    Before the seeder excluded normalization-peel children from unit
    inheritance, a derived parent whose only unit signal was a
    ``normalized_*`` child was stamped ``'1'`` — wrong whenever the parent's
    name asserts a dimensional head noun (``particle_mass`` is not
    dimensionless; its normalized child is). Repair is scoped to parents
    where ALL THREE hold, so genuinely dimensionless parents (e.g.
    collisionality) are untouched:

    - ``origin='derived'`` with unit ``'1'`` and no normalization marker of
      its own;
    - every unit-bearing child is a dimensionless normalization variant
      (the only possible source of the inherited ``'1'``);
    - a ``name_unit_consistency_check`` finding is on record — the name
      itself contradicts the dimensionless stamp.

    Clears ``unit`` and the ``HAS_UNIT`` edge; validation re-stamping is the
    caller's job (route the returned ids through the validation drain).
    Idempotent: repaired parents no longer match. Returns repaired ids.
    """
    rows = gc.query(
        """
        MATCH (sn:StandardName {unit: '1', origin: 'derived'})
        WHERE NOT any(t IN split(sn.id, '_')
                      WHERE t IN ['normalized', 'normalised'])
          AND sn.validation_issues IS NOT NULL
          AND any(issue IN sn.validation_issues
                  WHERE issue CONTAINS 'name_unit_consistency_check')
        MATCH (c)-[:HAS_PARENT]->(sn)
        WITH sn, collect(c) AS kids
        WHERE all(k IN kids
                  WHERE k.unit = '1'
                    AND any(t IN split(k.id, '_')
                            WHERE t IN ['normalized', 'normalised']))
        OPTIONAL MATCH (sn)-[r:HAS_UNIT]->(:Unit {id: '1'})
        DELETE r
        SET sn.unit = null
        RETURN sn.id AS id
        ORDER BY id
        """
    )
    repaired = [r["id"] for r in rows]
    if repaired:
        logger.info(
            "repair_normalization_peel_parent_units: cleared mis-inherited "
            "unit '1' on %d parent(s): %s",
            len(repaired),
            ", ".join(repaired),
        )
    return repaired


def seed_parent_sources(gc: Any | None = None) -> int:
    """Fully populate parent StandardName nodes from their children.

    Selects placeholder parents structurally: any ``StandardName`` node
    with ``name_stage IS NULL`` that is the target of at least one
    HAS_PARENT edge whose ``operator_kind`` is seedable
    (``projection``, ``coordinate``, ``unary_postfix``).

    This deliberately does NOT require the ``needs_composition`` flag.
    That flag is set by ``_write_standard_name_edges`` at edge-write
    time, but the seedable-edge-kinds set has grown over time
    (``unary_postfix`` was added later); placeholders created before
    the set expanded were never flagged and would otherwise stay
    orphaned even though the pipeline now considers them seedable.
    Matching structurally lets the pipeline self-heal on every run.

    Parent names are DETERMINISTIC — the name itself is grammar-derived
    from the components' shared base (e.g. ``magnetic_field`` is the
    only sensible parent of ``poloidal_magnetic_field`` /
    ``toroidal_magnetic_field`` / …). The name has no alternative to
    refine to: REVIEW_NAME produces noise scores on the placeholder
    description, REFINE_NAME has nothing to write. We therefore skip
    the name-axis pipeline entirely for deterministic parents — the
    quality-bearing work happens on the description+docs axis.

    1. Waits until ALL children of a parent have a ``name_stage`` set.
    2. Copies unit and cocos from children (asserts uniformity).
    3. Sets ``name_stage='accepted'`` directly. The children are
       already REVIEW_NAME-validated standard names; their structural
       parent inherits that validity by construction.
    4. Stamps ``origin='deterministic'`` so the semantic-sim gate,
       REFINE_NAME claim, and pool-pending counters all know to leave
       this node alone on the name axis.
    5. Sets ``description = DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER``
       so the export-time guard can refuse to publish until
       ``GENERATE_DOCS`` rewrites it with LLM-quality content.
    6. Sets ``docs_stage='pending'`` so the parent enters
       ``GENERATE_DOCS`` → ``REVIEW_DOCS`` immediately. The
       description+docs go through full RD-quorum review on the docs
       axis — that is the quality gate for these names.
    7. Creates a ``StandardNameSource`` with ``status='composed'`` for audit.
    8. Runs ISN parse to populate grammar fields.
    9. Clears ``needs_composition`` on the parent.

    Returns the number of parents fully populated.
    """
    _own_gc = gc is None
    if _own_gc:
        gc = GraphClient().__enter__()
    try:
        parents = _query_seedable_derived_parents(
            gc,
            state_where="parent.name_stage IS NULL",
        )

        if not parents:
            return 0

        seeded = _materialize_derived_parent_rows(gc, parents)
        logger.info("seed_parent_sources: populated %d parent SNs", seeded)
        return seeded
    finally:
        if _own_gc:
            gc.__exit__(None, None, None)


def normalize_derived_parent_lifecycle(gc: Any | None = None) -> int:
    """Repair eligible derived parents onto the docs lifecycle, idempotently."""
    _own_gc = gc is None
    if _own_gc:
        gc = GraphClient().__enter__()
    try:
        from imas_codex.standard_names.parents import is_admissible_parent_name

        state_where = (
            "parent.name_stage IS NULL "
            "OR parent.name_stage = 'pending' "
            "OR (parent.name_stage = 'accepted' AND parent.docs_stage IS NULL)"
        )
        seedable_parents = _query_seedable_derived_parents(
            gc,
            state_where=state_where,
        )
        legacy_parents = _query_legacy_repairable_derived_parents(
            gc,
            state_where=state_where,
        )
        accepted_unit_gap_where = (
            "parent.origin = 'derived' "
            "AND parent.name_stage = 'accepted' "
            "AND parent.docs_stage = 'accepted' "
            "AND parent.unit IS NULL"
        )
        accepted_seedable_unit_gaps = _query_seedable_derived_parents(
            gc,
            state_where=accepted_unit_gap_where,
        )
        accepted_legacy_unit_gaps = _query_legacy_repairable_derived_parents(
            gc,
            state_where=accepted_unit_gap_where,
        )
        cleanup_candidates = _query_accepted_derived_parents_for_cleanup(gc)
        invalid_accepted = []
        for parent_id in cleanup_candidates:
            result = is_admissible_parent_name(parent_id, gc)
            if not result.admit:
                invalid_accepted.append(parent_id)

        deleted = _delete_derived_parent_nodes(gc, invalid_accepted)
        if deleted:
            logger.info(
                "normalize_derived_parent_lifecycle: deleted %d inadmissible "
                "accepted derived parents",
                deleted,
            )

        # Reap childless derived "zombie" parents. A derived parent is a
        # structural abstraction over its children; once it has no incoming
        # HAS_PARENT edge at all it abstracts over nothing. The cleanup query
        # above only re-checks parents that STILL have a child, so orphaned
        # derived parents accumulated unboundedly (often carrying orphaned
        # docs). Reap those with ZERO incoming HAS_PARENT (truly orphaned):
        # ``rederive_structural_edges`` has already cleared edges from dead
        # (superseded/exhausted) names, so a parent of only-dead children is
        # now zero-incoming and reaps cleanly WITHOUT being re-minted next
        # startup (no live name derives it). Reaping on a "no LIVE child"
        # predicate instead would ping-pong: dead children keep re-deriving the
        # parent every startup. Scoped to ``origin='derived'`` so pipeline- and
        # catalog-authored names are never touched, even when momentarily
        # orphaned. Delete docs/reviews/derived-source scaffolding too.
        childless = [
            r["id"]
            for r in gc.query(
                """
                MATCH (p:StandardName {origin: 'derived'})
                WHERE NOT EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(p) }
                RETURN p.id AS id
                """
            )
            or []
        ]
        if childless:
            reaped = _delete_derived_parent_nodes(gc, childless)
            if reaped:
                deleted += reaped
                logger.info(
                    "normalize_derived_parent_lifecycle: reaped %d childless "
                    "derived parents",
                    reaped,
                )

        # Automatic orphaned-content cleanup: clear any DocsRevision left
        # dangling by a past reap (idempotent backstop — no orphaned docs
        # content can accumulate across startups).
        sweep_orphaned_docs_revisions(gc)

        if (
            not seedable_parents
            and not legacy_parents
            and not accepted_seedable_unit_gaps
            and not accepted_legacy_unit_gaps
        ):
            return deleted
        repaired = _materialize_derived_parent_rows(gc, seedable_parents)
        repaired += _materialize_derived_parent_rows(
            gc,
            legacy_parents,
            infer_kind_from_existing_topology=True,
        )
        repaired += _materialize_derived_parent_rows(
            gc,
            accepted_seedable_unit_gaps,
        )
        repaired += _materialize_derived_parent_rows(
            gc,
            accepted_legacy_unit_gaps,
            infer_kind_from_existing_topology=True,
        )
        logger.info(
            "normalize_derived_parent_lifecycle: repaired %d derived parent SNs",
            repaired,
        )
        return repaired + deleted
    finally:
        if _own_gc:
            gc.__exit__(None, None, None)


def write_standard_names(
    names: list[dict[str, Any]],
    *,
    override: bool = False,
) -> int:
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

    Performs conflict detection on ``unit``: if a StandardName already exists
    with a different unit value, that entry is skipped (not written)
    and a warning is logged.

    Returns the number of nodes written.
    """
    if not names:
        return 0

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
        return 0

    # Grammar parse gate — reject names that ISN cannot parse.
    # This is a hard invariant: if a name fails grammar round-trip, it is
    # invalid by definition and must not enter the graph.
    try:
        from imas_standard_names.grammar import parse_standard_name
    except ImportError:
        parse_standard_name = None  # type: ignore[assignment]

    if parse_standard_name is not None:
        valid_names: list[dict[str, Any]] = []
        for n in names:
            try:
                parse_standard_name(n["id"])
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
            return 0

    # A COCOS-dependent name (cocos_transformation_type set) must carry the
    # integer cocos convention so the HAS_COCOS edge below is created. The
    # catalog follows a single convention — the current DD version's (DDv4 →
    # COCOS 17) — so default it from the current DD whenever compose/refine
    # didn't thread it through. COCOS-independent names stay unlinked by design.
    if any(
        n.get("cocos_transformation_type") and n.get("cocos") is None for n in names
    ):
        with GraphClient() as _conv_gc:
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

    with GraphClient() as gc:
        # Conflict-detect on unit — same name with different unit is a data
        # integrity error.  Filter out conflicting entries rather than raising
        # so that non-conflicting entries can still proceed.
        unit_check_batch = [
            {"id": n["id"], "unit": n.get("unit")} for n in names if n.get("unit")
        ]
        if unit_check_batch:
            unit_conflicts = list(
                gc.query(
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
                    return 0

        # MERGE StandardName nodes with provenance — coalesce to preserve existing data
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.id})
            SET sn.source_types = coalesce(b.source_types, sn.source_types),
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
                sn.link_status = coalesce(b.link_status, sn.link_status),
                sn.review_input_hash = b.review_input_hash,
                sn.embedding = coalesce(b.embedding, sn.embedding),
                sn.embedded_at = coalesce(b.embedded_at, sn.embedded_at),
                sn.grammar_parse_version = coalesce(b.grammar_parse_version, sn.grammar_parse_version),
                sn.validation_diagnostics_json = coalesce(b.validation_diagnostics_json, sn.validation_diagnostics_json),
                sn.physical_base = coalesce(b.physical_base, sn.physical_base),
                sn.geometric_base = coalesce(b.geometric_base, sn.geometric_base),
                sn.subject = coalesce(b.subject, sn.subject),
                sn.component = coalesce(b.component, sn.component),
                sn.coordinate = coalesce(b.coordinate, sn.coordinate),
                sn.transformation = coalesce(b.transformation, sn.transformation),
                sn.position = coalesce(b.position, sn.position),
                sn.process = coalesce(b.process, sn.process),
                sn.device = coalesce(b.device, sn.device),
                sn.region = coalesce(b.region, sn.region),
                sn.aggregation = coalesce(b.aggregation, sn.aggregation),
                sn.orbit = coalesce(b.orbit, sn.orbit),
                sn.population = coalesce(b.population, sn.population),
                sn.object = coalesce(b.object, sn.object),
                sn.geometry = coalesce(b.geometry, sn.geometry),
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
                gc.query(
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

            gc.query(
                """
                UNWIND $batch AS b
                MERGE (sn:StandardName {id: b.id})
                SET sn.physics_domain = coalesce(b.physics_domain, sn.physics_domain),
                    sn.source_domains = coalesce(b.source_domains, sn.source_domains)
                """,
                batch=pd_batch,
            )

        # Create HAS_STANDARD_NAME relationships: entity → concept
        dd_names = [n for n in names if "dd" in (n.get("source_types") or [])]
        signal_names = [n for n in names if "signals" in (n.get("source_types") or [])]

        if dd_names:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                """,
                batch=[
                    {"id": n["id"], "source_id": n["source_id"]}
                    for n in dd_names
                    if n.get("source_id")
                ],
            )
        if signal_names:
            gc.query(
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
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (sn:StandardName {id: id})-[r:HAS_UNIT]->()
                DELETE r
                """,
                ids=unit_sn_ids,
            )
            gc.query(
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
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (c:COCOS {id: b.cocos})
                MERGE (sn)-[:HAS_COCOS]->(c)
                """,
                batch=cocos_batch,
            )

        # Create grammar decomposition: typed edges + per-segment columns
        token_miss_gaps = _write_grammar_decomposition(gc, [n["id"] for n in names])

        # Emit structural edges: HAS_PARENT, HAS_ERROR, HAS_PREDECESSOR,
        # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN.
        # Tail pass — all nodes in this batch exist before edges are written.
        _write_standard_name_edges(gc, names)

    # Persist token-miss gaps as VocabGap nodes (outside gc context —
    # write_vocab_gaps opens its own GraphClient)
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
            write_vocab_gaps(dd_gap_dicts, source_type="dd")
        if signal_gap_dicts:
            write_vocab_gaps(signal_gap_dicts, source_type="signals")

    # Sweep skeleton placeholders created by relationship-side MERGE on
    # uncomposed targets (HAS_PARENT, HAS_ERROR, HAS_PREDECESSOR,
    # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN). A real StandardName
    # always has at least a created_at OR generated_at timestamp; pure
    # skeletons (id-only) are detached and deleted.
    # Opens its own GraphClient — the surrounding `with` block has already
    # closed; write_vocab_gaps above follows the same pattern.
    with GraphClient() as gc:
        swept = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.created_at IS NULL
              AND sn.generated_at IS NULL
              AND sn.validation_status IS NULL
              AND sn.unit IS NULL
              AND sn.kind IS NULL
              AND sn.needs_composition IS NULL
              AND NOT EXISTS { ()-[:HAS_PARENT]->(sn) }
              AND NOT EXISTS { ()-[:HAS_ERROR]->(sn) }
            DETACH DELETE sn
            RETURN count(sn) AS swept
            """
        )
    swept_count = (swept[0]["swept"] if swept else 0) if swept else 0
    if swept_count:
        logger.info("Swept %d skeleton StandardName placeholder(s)", swept_count)

    written = len(names)
    logger.info("Wrote %d StandardName nodes", written)
    return written


def write_reviews(records: list[dict[str, Any]], *, skip_cost: bool = False) -> int:
    """MERGE ``StandardNameReview`` nodes and ``HAS_REVIEW`` edges from StandardName.

    Each record must contain:

    - ``id`` (str) — composite key
      ``{standard_name_id}:{axis}:{review_group_id}:{cycle_index}``
    - ``standard_name_id`` (str) — parent StandardName
    - ``model`` (str), ``model_family`` (str), ``is_canonical`` (bool)
    - ``score`` (float 0-1), ``scores_json`` (str), ``tier`` (str)
    - ``reviewed_at`` (str ISO 8601)

    RD-quorum fields (required for new-style reviews):
    - ``review_axis`` (str) — "names" or "docs"
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

    Returns the number of StandardNameReview records written.
    """
    if not records:
        return 0
    # Guard: must attach to an existing StandardName.
    valid = [r for r in records if r.get("id") and r.get("standard_name_id")]
    if not valid:
        return 0

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
                r.scores_json = b.scores_json,
                r.tier = b.tier,
                r.comments = b.comments,
                r.comments_per_dim_json = b.comments_per_dim_json,
                r.suggested_name = b.suggested_name,
                r.suggestion_justification = b.suggestion_justification,
                r.reviewed_at = b.reviewed_at,
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
                r.llm_at = b.llm_at,
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
                    "score": float(r.get("score") or 0.0),
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
                    axis = r.get("review_axis", "names")
                    if axis == "docs":
                        docs_cost_map[sn_id] = docs_cost_map.get(sn_id, 0.0) + cost
                    else:
                        name_cost_map[sn_id] = name_cost_map.get(sn_id, 0.0) + cost
            if name_cost_map:
                gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.llm_cost_review_name = coalesce(sn.llm_cost_review_name, 0.0) + b.cost,
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
                    SET sn.llm_cost_review_docs = coalesce(sn.llm_cost_review_docs, 0.0) + b.cost,
                        sn.review_docs_count = coalesce(sn.review_docs_count, 0) + 1,
                        sn.llm_cost = coalesce(sn.llm_cost, 0.0) + b.cost
                    """,
                    batch=[
                        {"sn_id": sn_id, "cost": cost}
                        for sn_id, cost in docs_cost_map.items()
                    ],
                )

    logger.info("Wrote %d StandardNameReview nodes", len(valid))
    return len(valid)


def update_review_aggregates(
    standard_name_ids: list[str],
    *,
    threshold: float = 0.2,
) -> int:
    """Recompute per-StandardName aggregates from attached StandardNameReview nodes.

    **Winning-group selection**: identifies the most recent review group
    whose ``resolution_method`` is one of ``quorum_consensus``,
    ``authoritative_escalation``, or ``single_review`` (excluding
    ``retry_item`` and ``max_cycles_reached``). Mirrors that group's
    final scores onto the SN aggregates.

    Also sets ``review_count``, ``review_mean_score``, and
    ``review_disagreement`` across all attached StandardNameReview nodes.

    Returns the number of StandardName nodes updated.
    """
    if not standard_name_ids:
        return 0
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
            WITH sn, count(r) AS n, avg(r.score) AS mean,
                 CASE WHEN count(r) > 1 THEN max(r.score) - min(r.score) ELSE 0.0 END AS spread
            SET sn.review_count = n,
                sn.review_mean_score = CASE WHEN n > 0 THEN mean ELSE null END,
                sn.review_disagreement = (n > 1 AND spread >= $threshold)
            RETURN sn.id AS id
            """,
            ids=standard_name_ids,
            threshold=float(threshold),
        )
        return len(list(rows or []))


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
            SET sn.reviewer_score_name = b.reviewer_score_name,
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
            SET sn.reviewer_score_docs = b.reviewer_score_docs,
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

    # Fallback: latest available version
    rows = list(
        gc.query(
            "MATCH (t:GrammarToken) "
            "RETURN DISTINCT t.version AS v ORDER BY v DESC LIMIT 1"
        )
        or []
    )
    if rows:
        fallback = rows[0]["v"]
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

    Plan 40 Phase 1 — replaces ``_write_segment_edges``. Always populates
    bare-name per-segment columns (``sn.physical_base``, ``sn.subject``, …)
    from the ISN parser, regardless of whether a closed-vocabulary
    GrammarToken exists for the value. Conditionally writes typed edges
    only when a GrammarToken does exist; on parser error clears all
    per-segment columns and edges (leaving the node unparseable). Such a
    name is quarantined authoritatively at validate time
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
        from imas_standard_names.grammar import parse_standard_name
        from imas_standard_names.graph.spec import segment_edge_specs
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
    columns_to_null = ",\n                    ".join(
        f"sn.{col} = null" for col in _GRAMMAR_SEGMENT_COLUMNS
    )
    columns_to_set = ",\n                ".join(
        f"sn.{col} = ${col}" for col in _GRAMMAR_SEGMENT_COLUMNS
    )
    typed_edge_foreach = "\n                ".join(
        f"FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = '{seg}' "
        f"THEN [1] ELSE [] END |\n                    "
        f"MERGE (sn)-[:{rel}]->(t)\n                )"
        for seg, rel in _SEGMENT_EDGE_TYPES
    )

    for sn_id in name_ids:
        try:
            parsed = parse_standard_name(sn_id)
        except Exception:
            logger.warning("Grammar parse failed for '%s' — recording fallback", sn_id)
            # Clear columns + typed/segment edges, set fallback=true
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
        # (_parse_grammar), so columns can never diverge between the two.
        column_values = _segments_from_model(parsed)

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

        try:
            edge_specs = segment_edge_specs(parsed)
        except Exception:
            logger.debug(
                "segment_edge_specs failed for '%s' — columns set, skipping edges",
                sn_id,
                exc_info=True,
            )
            continue

        # Idempotent: drop existing typed/segment edges before re-writing.
        gc.query(
            f"""
            MATCH (sn:StandardName {{id: $sn_id}})-[r:{edge_delete_types}]->(:GrammarToken)
            DELETE r
            """,
            sn_id=sn_id,
        )

        if not edge_specs:
            continue

        edges_param = [
            {
                "position": s.position,
                "segment": s.segment,
                "token": s.token,
            }
            for s in edge_specs
        ]

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


# Deprecation alias — Phase 1 retains old name for one release. Callers
# inside the package have been migrated; external pipeline code may still
# import this symbol. Removed in the release after Phase 4.
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


def persist_generated_name_batch(
    candidates: list[dict[str, Any]],
    *,
    compose_model: str,
    dd_version: str | None = None,
    cocos_version: int | None = None,
    run_id: str | None = None,
) -> int:
    """Persist a single generate-name batch immediately to graph.

    Called from within ``compose_batch`` after LLM success.
    Enriches candidates with provenance metadata, embeds the standard-name
    string, and extracts grammar fields before writing.

    Embedding uses the standard-name string (``id``) — not the description,
    which is added later by the enrich pipeline.  If embedding fails for a
    candidate, ``embed_failed_at`` is set so the name can be retried later;
    the name is still written to the graph so it can advance through review.

    After writing the StandardName nodes via :func:`write_standard_names`,
    atomically transitions each new SN to ``name_stage='drafted'`` and
    ``docs_stage='pending'`` (chain lengths = 0) and clears the
    ``StandardNameSource`` claim in a **single** Neo4j transaction — so that
    either all stage/claim updates land or none do.

    Returns the number of nodes written.
    """
    from datetime import UTC, datetime

    if not candidates:
        return 0

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
        # D5/P0.3: derive kind from name structure (overrides LLM default).
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

    written = write_standard_names(candidates)

    # --- Backfill primary_cluster_id from DD source path ---
    # The compose worker doesn't carry cluster info, so we derive it from
    # the graph: SN ← PRODUCED_NAME ← SNS → FROM_DD_PATH → IMAS → IN_CLUSTER → cluster.
    # Uses the same IDS > domain > global scope priority as enrichment.
    _backfill_cluster_from_sources(candidates)

    # --- Atomically transition stage + clear source claim ---
    # Build the batch excluding error-sibling candidates (no source node).
    # StandardNameSource.id has a source-type prefix (e.g. "dd:path" or
    # "signals:path"), so we must prepend the prefix derived from the
    # candidate's source_types field.
    finalize_batch = []
    for entry in candidates:
        if not entry.get("id"):
            continue
        if entry.get("model") == "deterministic:dd_error_modifier":
            continue
        raw_source_id = entry.get("source_id")
        if raw_source_id:
            _st = (entry.get("source_types") or ["dd"])[0]
            _prefix = "dd" if _st == "dd" else "signals"
            sns_id = f"{_prefix}:{raw_source_id}"
        else:
            sns_id = None
        finalize_batch.append(
            {
                "sn_id": entry["id"],
                "sns_id": sns_id,
                "model": compose_model,
            }
        )
    if finalize_batch:
        _finalize_generated_name_stage(finalize_batch)

    # --- Enforce one-source-one-name invariant (Class-A duplicate guard) ---
    # On ``--force``/regen the source may already carry a prior accepted
    # pipeline name. Retire it now so a later acceptance of this regenerated
    # name cannot leave two live accepted names competing for one source.
    # No-op in normal compose (the source had no prior pipeline name) and for
    # byte-identical regen (same node id is reused — nothing to supersede).
    supersede_pairs = [
        {"new_name": entry["id"], "source_id": entry["source_id"]}
        for entry in candidates
        if entry.get("id")
        and entry.get("source_id")
        and entry.get("model") != "deterministic:dd_error_modifier"
        and "dd" in (entry.get("source_types") or [])
    ]
    if supersede_pairs:
        supersede_prior_source_names(supersede_pairs)

    # Async counter bump — live progress visibility for ``sn status``
    if written > 0:
        bump_sn_run_counter(run_id, "names_composed", delta=written)

    return written


def _backfill_cluster_from_sources(candidates: list[dict[str, Any]]) -> None:
    """Backfill ``primary_cluster_id`` and ``IN_CLUSTER`` edge from DD sources.

    For each candidate, query the graph for the best cluster via the
    HAS_STANDARD_NAME → IMASNode → IN_CLUSTER path, preferring
    IDS-scope > domain-scope > global-scope clusters.
    """
    sn_ids = [c["id"] for c in candidates if c.get("id")]
    if not sn_ids:
        return

    try:
        with GraphClient() as gc:
            results = gc.query(
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
                SET sn.primary_cluster_id = best_cluster.id
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
        logger.debug("_backfill_cluster_from_sources: failed", exc_info=True)


@retry_on_deadlock()
def _finalize_generated_name_stage(
    batch: list[dict[str, Any]],
) -> None:
    """Set stage fields on new SNs and clear source claims — single transaction.

    Each item in *batch* must have ``sn_id`` (StandardName id), optionally
    ``sns_id`` (StandardNameSource id), and ``model``.

    In one transaction:
    - ``name_stage = 'drafted'``, ``chain_length = 0``
    - ``docs_stage = 'pending'``, ``docs_chain_length = 0``
    - ``generated_at = datetime()``, ``model = <model>``
    - Source: ``claim_token = null``, ``claimed_at = null``,
      ``status = 'composed'``, ``composed_at = datetime()``,
      ``produced_sn_id = sn.id``
    - Edge: ``(sns)-[:PRODUCED_NAME]->(sn)``
    """
    if not batch:
        return

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                tx.run(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.name_stage       = 'drafted',
                        sn.chain_length     = 0,
                        sn.docs_stage       = 'pending',
                        sn.docs_chain_length = 0,
                        sn.generated_at     = datetime(),
                        sn.model            = b.model
                    WITH sn, b
                    WHERE b.sns_id IS NOT NULL
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    SET sns.claim_token  = null,
                        sns.claimed_at   = null,
                        sns.status       = 'composed',
                        sns.composed_at  = datetime(),
                        sns.produced_sn_id = sn.id,
                        sn.run_id        = coalesce(sns.run_id, sn.run_id)
                    MERGE (sns)-[:PRODUCED_NAME]->(sn)
                    """,
                    batch=batch,
                )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    logger.debug(
        "_finalize_generated_name_stage: finalized %d SNs",
        len(batch),
    )


@retry_on_deadlock()
def write_vocab_gaps(
    gaps: list[dict[str, str]],
    source_type: str = "dd",
    *,
    skip_segment_filter: bool = False,
) -> int:
    """Persist VocabGap nodes and HAS_STANDARD_NAME_VOCAB_GAP relationships.

    Each gap dict has: source_id, segment, token, reason.

    Deduplicates VocabGap nodes by id (vocab_gap:{segment}:{token}).
    Creates HAS_STANDARD_NAME_VOCAB_GAP relationships from source entities with
    per-source reason as a relationship property.

    Args:
        gaps: List of gap dicts.
        source_type: Source type ('dd' or 'signals').
        skip_segment_filter: When ``True``, bypass the segment filter.
            Used by auto-VocabGap detection — we still want to
            track novel tokens for ISN review.

    Returns the number of VocabGap nodes written.
    """
    if not gaps:
        return 0

    from datetime import UTC, datetime

    if not skip_segment_filter:
        from imas_codex.standard_names.segments import filter_closed_segment_gaps

        # Drop gaps on pseudo segments (grammar_ambiguity) — not missing tokens.
        gaps, dropped = filter_closed_segment_gaps(gaps)
        if dropped:
            from collections import Counter

            drop_hist = Counter(g.get("segment") for g in dropped)
            logger.info(
                "write_vocab_gaps: skipped %d gaps on pseudo segments (%s)",
                len(dropped),
                ", ".join(f"{seg}={n}" for seg, n in drop_hist.most_common()),
            )
        if not gaps:
            return 0

    now = datetime.now(UTC).isoformat()

    # Classify gaps using the ISN-backed classify_gap() function
    from imas_codex.standard_names.segments import (
        NON_ACTIONABLE_GAP_CATEGORIES,
        classify_gap,
        is_valid_segment,
    )

    # Build deduplicated gap nodes and relationship batch
    gap_nodes: dict[str, dict] = {}
    rel_batch: list[dict] = []

    for g in gaps:
        segment = g["segment"]
        token = g["token"]
        gap_id = f"vocab_gap:{segment}:{token}"

        if gap_id not in gap_nodes:
            # Validate segment exists in ISN grammar
            if not is_valid_segment(segment):
                logger.warning(
                    "write_vocab_gaps: skipping gap with unknown segment "
                    "'%s' for token '%s'",
                    segment,
                    token,
                )
                continue

            category, actual_segments = classify_gap(segment, token)

            # Filter non-actionable gaps — these are not genuine vocabulary
            # deficiencies but LLM mis-classifications or decomposable compounds.
            # Shared with the source-marking path so the two never drift.
            if category in NON_ACTIONABLE_GAP_CATEGORIES:
                logger.debug(
                    "write_vocab_gaps: skipping %s gap %s (token '%s', segment '%s')",
                    category,
                    gap_id,
                    token,
                    segment,
                )
                continue

            gap_nodes[gap_id] = {
                "id": gap_id,
                "segment": segment,
                "token": token,
                "example_count": 0,
                "category": category,
                "actual_segments": actual_segments,
                "nearest_token": None,
                "nearest_similarity": None,
                "dedup_decision": None,
            }
        gap_nodes[gap_id]["example_count"] += 1

        # Carry the compose-time token-reuse adjudication (Task 5).  A node may
        # be observed many times; prefer a distinct_confirmed stamp (with its
        # nearest token + score) over an unchecked one so the strongest signal
        # wins and the ISN rotation sees the adjudication.
        _decision = g.get("dedup_decision")
        if _decision and gap_nodes[gap_id]["dedup_decision"] != "distinct_confirmed":
            gap_nodes[gap_id]["dedup_decision"] = _decision
            if _decision == "distinct_confirmed":
                gap_nodes[gap_id]["nearest_token"] = g.get("nearest_token")
                gap_nodes[gap_id]["nearest_similarity"] = g.get("nearest_similarity")

        rel_batch.append(
            {
                "gap_id": gap_id,
                "source_id": g["source_id"],
                "reason": g.get("reason", ""),
                "observed_at": now,
            }
        )

    with GraphClient() as gc:
        # MERGE VocabGap nodes — increment count, update timestamps
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (vg:VocabGap {id: b.id})
            SET vg.segment = b.segment,
                vg.token = b.token,
                vg.example_count = coalesce(vg.example_count, 0) + b.example_count,
                vg.category = b.category,
                vg.actual_segments = b.actual_segments,
                vg.first_seen_at = coalesce(vg.first_seen_at, datetime()),
                vg.last_seen_at = datetime()
            // Token-reuse adjudication: distinct_confirmed is sticky — never
            // let a later unchecked observation overwrite it; carry the
            // nearest token + score only on a confirming stamp.
            FOREACH (_ IN CASE
                WHEN b.dedup_decision IS NOT NULL
                     AND coalesce(vg.dedup_decision, '') <> 'distinct_confirmed'
                THEN [1] ELSE [] END |
                SET vg.dedup_decision = b.dedup_decision)
            FOREACH (_ IN CASE
                WHEN b.dedup_decision = 'distinct_confirmed'
                THEN [1] ELSE [] END |
                SET vg.nearest_token = b.nearest_token,
                    vg.nearest_similarity = b.nearest_similarity)
            """,
            batch=list(gap_nodes.values()),
        )

        # Create HAS_STANDARD_NAME_VOCAB_GAP relationships from the underlying
        # DD path / facility signal entity (the DD-catalog view).
        entity_label = "IMASNode" if source_type == "dd" else "FacilitySignal"
        gc.query(
            f"""
            UNWIND $batch AS b
            MATCH (vg:VocabGap {{id: b.gap_id}})
            MATCH (src:{entity_label} {{id: b.source_id}})
            MERGE (src)-[r:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
            SET r.reason = b.reason,
                r.observed_at = datetime(b.observed_at)
            """,
            batch=rel_batch,
        )

        # And a source-first link from the StandardNameSource itself — the
        # canonical, one-hop "why is this source blocked?" edge that reconcile
        # traverses. Its id is uniform across source types ({type}:{source_id}).
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (vg:VocabGap {id: b.gap_id})
            MATCH (sns:StandardNameSource {id: $prefix + b.source_id})
            MERGE (sns)-[r:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
            SET r.reason = b.reason,
                r.observed_at = datetime(b.observed_at)
            """,
            batch=rel_batch,
            prefix=f"{source_type}:",
        )

    written = len(gap_nodes)
    logger.info("Wrote %d VocabGap nodes from %d gap reports", written, len(gaps))
    return written


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

_CLAIM_TIMEOUT = "PT300S"  # 5 minutes — matches DEFAULT_CLAIM_TIMEOUT_SECONDS


@retry_on_deadlock()
def claim_names_for_validation(limit: int = 50) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim unvalidated StandardNames for ISN validation.

    Returns ``(token, items)`` where *token* must be passed to
    ``mark_names_validated`` or ``release_validation_claims``.
    """
    import uuid

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        # Step 1: claim with random ordering and unique token
        # Gate: any StandardName with a description that hasn't been validated.
        # No stage filter — names with NULL name_stage must also be validated.
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.description IS NOT NULL
              AND sn.validated_at IS NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            limit=limit,
            token=token,
            timeout=_CLAIM_TIMEOUT,
        )
        # Step 2: verify — only our token
        results = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(src)
            OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
            WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted', 'contested']
            RETURN sn.id AS id, sn.description AS description,
                   sn.documentation AS documentation, sn.kind AS kind,
                   sn.unit AS unit, sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.object AS object,
                   sn.physics_domain AS physics_domain,
                   sn.origin AS origin,
                   collect(DISTINCT src.id) AS source_ids,
                   collect(DISTINCT child.id) AS children
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


def mark_names_validated(
    token: str,
    results: list[dict[str, Any]],
) -> int:
    """Write validation results and release claims atomically.

    Each result dict must have ``id``, ``validation_issues`` (list[str]),
    ``validation_layer_summary`` (JSON string), and ``validation_status``
    (``"valid"`` or ``"quarantined"``).
    Token-verified: only updates nodes still claimed by this token.
    """
    if not results:
        return 0
    batch = []
    for r in results:
        batch.append(
            {
                "id": r["id"],
                "issues": r.get("validation_issues") or [],
                "summary": _ensure_json(r.get("validation_layer_summary")),
                "validation_status": r.get("validation_status", "valid"),
            }
        )
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id, claim_token: $token})
            SET sn.validated_at = datetime(),
                sn.validation_issues = b.issues,
                sn.validation_layer_summary = b.summary,
                sn.validation_status = b.validation_status,
                sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS marked
            """,
            batch=batch,
            token=token,
        )
        return result[0]["marked"] if result else 0


def release_validation_claims(token: str) -> int:
    """Release validation claims on error. Token-verified."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.claimed_at = null, sn.claim_token = null
            RETURN count(sn) AS released
            """,
            token=token,
        )
        return result[0]["released"] if result else 0


@retry_on_deadlock()
def claim_names_for_embedding(limit: int = 100) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim validated StandardNames needing embedding.

    Returns ``(token, items)`` with ``id`` and ``description``.
    """
    import uuid

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted', 'contested']
              AND sn.validated_at IS NOT NULL
              AND sn.embedding IS NULL
              AND sn.description IS NOT NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            limit=limit,
            token=token,
            timeout=_CLAIM_TIMEOUT,
        )
        results = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            RETURN sn.id AS id, sn.description AS description
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


def mark_names_embedded(
    token: str,
    embed_batch: list[dict[str, Any]],
) -> int:
    """Write embeddings and release claims. Token-verified.

    Each item in *embed_batch* must have ``id`` and ``embedding``.
    """
    if not embed_batch:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id, claim_token: $token})
            SET sn.embedding = b.embedding,
                sn.embedded_at = datetime(),
                sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS marked
            """,
            batch=[
                {"id": e["id"], "embedding": e["embedding"]}
                for e in embed_batch
                if e.get("embedding")
            ],
            token=token,
        )
        return result[0]["marked"] if result else 0


def release_embedding_claims(token: str) -> int:
    """Release embedding claims on error. Token-verified."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.claimed_at = null, sn.claim_token = null
            RETURN count(sn) AS released
            """,
            token=token,
        )
        return result[0]["released"] if result else 0


def get_validated_names(
    ids_filter: str | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Query all validated StandardNames for consolidation analysis.

    Read-only — no claims needed since consolidation is a batch analysis.
    Returns drafted names that have ``validated_at`` set and
    ``validation_status`` = ``'valid'``.
    """
    where_parts = [
        "sn.name_stage = 'drafted'",
        "sn.validated_at IS NOT NULL",
        "sn.validation_status = 'valid'",
    ]
    params: dict[str, Any] = {"limit": limit}

    if ids_filter:
        where_parts.append("ANY(p IN sn.source_paths WHERE p STARTS WITH $ids_prefix)")
        from imas_codex.standard_names.source_paths import ids_prefix_for_source_paths

        params["ids_prefix"] = ids_prefix_for_source_paths(ids_filter)

    where_clause = " AND ".join(where_parts)

    with GraphClient() as gc:
        results = gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {where_clause}
            OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(src)
            WITH sn, collect(DISTINCT src.id) AS source_ids
            RETURN sn.id AS id, sn.description AS description,
                   sn.documentation AS documentation, sn.kind AS kind,
                   sn.unit AS unit, sn.links AS links,
                   sn.source_paths AS source_paths,
                   source_ids
            LIMIT $limit
            """,
            **params,
        )
        return [dict(r) for r in results]


def mark_names_consolidated(name_ids: list[str]) -> int:
    """Mark names as consolidated (approved by cross-batch analysis)."""
    if not name_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            SET sn.consolidated_at = datetime()
            RETURN count(sn) AS marked
            """,
            ids=name_ids,
        )
        return result[0]["marked"] if result else 0


def reset_standard_names(
    *,
    from_stage: str | None = "drafted",
    to_stage: str | None = None,
    include_accepted: bool = False,
    source_filter: str | None = None,
    ids_filter: str | None = None,
    path_allowlist: list[str] | None = None,
    dry_run: bool = False,
    since: str | None = None,
    before: str | None = None,
    below_score: float | None = None,
    tiers: list[str] | None = None,
    validation_status: str | None = None,
) -> int:
    """Reset StandardName nodes to allow re-processing.

    Clears transient fields (embedding, embedded_at, model, generated_at)
    and removes HAS_STANDARD_NAME, HAS_UNIT, and
    HAS_COCOS relationships for matching nodes.

    Parameters
    ----------
    from_stage:
        Only reset nodes with this ``name_stage`` (default ``"drafted"``).
        ``None`` selects any live stage (terminal ``superseded`` /
        ``exhausted`` stages are always excluded; ``accepted`` requires
        ``include_accepted=True``).
    to_stage:
        Target ``name_stage`` after reset.  ``None`` (default) clears fields
        only without changing the stage.
    include_accepted:
        Required to reset ``name_stage='accepted'`` names — these are
        committed catalog entries and the operator must opt in explicitly.
    source_filter:
        Restrict to nodes whose ``source_types`` contains ``"dd"`` or
        ``"signals"``.
    ids_filter:
        Restrict to nodes whose HAS_STANDARD_NAME source path starts with this
        IDS name (matched via ``IMASNode -[:HAS_STANDARD_NAME]-> sn``).
    path_allowlist:
        Restrict to nodes attached to an IMASNode whose ``src.id`` is *exactly*
        in this list (matched via ``IMASNode -[:HAS_STANDARD_NAME]-> sn`` with
        ``src.id IN $path_allowlist``).  Unlike ``ids_filter`` this is an
        exact-path allowlist, not a prefix — names on any other path are left
        untouched.  Combines with ``ids_filter`` (both predicates apply).
    dry_run:
        Return the count of matching nodes without modifying anything.
    since:
        Only reset names with ``generated_at >= this`` ISO timestamp.
    before:
        Only reset names with ``generated_at < this`` ISO timestamp.
    below_score:
        Only reset names with ``reviewer_score < this`` value (0.0–1.0).
    tiers:
        Only reset names with ``review_tier`` in this list.
    validation_status:
        Only reset names with this ``validation_status`` value.

    Returns
    -------
    Number of nodes reset (or that would be reset in dry-run mode).
    """
    with GraphClient() as gc:
        params: dict[str, Any] = {}
        where_clauses = [
            "NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted', 'contested']"
        ]
        if from_stage is not None:
            where_clauses.append("sn.name_stage = $from_stage")
            params["from_stage"] = from_stage
        if not include_accepted and from_stage != "accepted":
            where_clauses.append("coalesce(sn.name_stage, '') <> 'accepted'")

        if source_filter:
            where_clauses.append("$source_filter IN sn.source_types")
            params["source_filter"] = source_filter

        if since:
            where_clauses.append("sn.generated_at >= datetime($since)")
            params["since"] = since
        if before:
            where_clauses.append("sn.generated_at < datetime($before)")
            params["before"] = before
        if below_score is not None:
            where_clauses.append("sn.reviewer_score_name < $below_score")
            params["below_score"] = below_score
        if tiers:
            where_clauses.append("sn.review_tier IN $tiers")
            params["tiers"] = tiers
        if validation_status:
            where_clauses.append("sn.validation_status = $validation_status")
            params["validation_status"] = validation_status

        where = " AND ".join(where_clauses)

        # Build the optional IMASNode-source scope. ids_filter is a prefix
        # match; path_allowlist is an exact-path membership test. Either (or
        # both) forces selection to go through the HAS_STANDARD_NAME join.
        src_clauses: list[str] = []
        if ids_filter:
            params["ids_prefix"] = ids_filter + "/"
            src_clauses.append("src.id STARTS WITH $ids_prefix")
        # Fail-closed: an empty allowlist selects the scoped branch and
        # matches NOTHING (``src.id IN []``). Treating ``[]`` as falsy here
        # would fall through to the unscoped branch and reset the whole graph
        # — the 1863-name-wipe shape. ``None`` means "no allowlist scope".
        if path_allowlist is not None:
            params["path_allowlist"] = list(path_allowlist)
            src_clauses.append("src.id IN $path_allowlist")
        use_src_join = bool(src_clauses)
        src_where = " AND ".join(src_clauses)

        if use_src_join:
            count_cypher = f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {where}
                AND {src_where}
                RETURN count(DISTINCT sn) AS n
            """
        else:
            count_cypher = f"""
                MATCH (sn:StandardName)
                WHERE {where}
                RETURN count(sn) AS n
            """

        result = gc.query(count_cypher, **params)
        count = result[0]["n"] if result else 0
        logger.info(
            "reset_standard_names: %d nodes match (from_stage=%s, source=%s, ids=%s)",
            count,
            from_stage,
            source_filter,
            ids_filter,
        )

        if dry_run or count == 0:
            return count

        if use_src_join:
            # Collect matching SN ids first, then operate on them. This keeps
            # every subsequent mutation scoped to exactly the names attached to
            # the in-scope source paths — names on other paths are untouched.
            collect_cypher = f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {where}
                AND {src_where}
                RETURN DISTINCT sn.id AS sn_id
            """
            rows = gc.query(collect_cypher, **params)
            sn_ids = [r["sn_id"] for r in rows]
            reset_params: dict[str, Any] = {"sn_ids": sn_ids}
            node_match = "MATCH (sn:StandardName) WHERE sn.id IN $sn_ids"
        else:
            reset_params = dict(params)
            node_match = f"MATCH (sn:StandardName) WHERE {where}"

        # Clear transient fields, optionally set a new stage
        if to_stage is not None:
            set_clause = (
                "sn.embedding = null, sn.embedded_at = null, sn.model = null, "
                "sn.generated_at = null, "
                "sn.cocos_transformation_type = null, sn.cocos = null, sn.dd_version = null, "
                "sn.name_stage = $to_stage"
            )
            reset_params["to_stage"] = to_stage
        else:
            set_clause = (
                "sn.embedding = null, sn.embedded_at = null, sn.model = null, "
                "sn.generated_at = null, "
                "sn.cocos_transformation_type = null, sn.cocos = null, sn.dd_version = null"
            )

        # Remove HAS_STANDARD_NAME, HAS_UNIT, and HAS_COCOS relationships and
        # clear the transient fields in a SINGLE statement so the reset is
        # atomic: a mid-sequence failure can no longer leave a node with its
        # edges stripped but its fields (or target stage) untouched, or vice
        # versa. Each DELETE runs in its own unit CALL subquery so the outer
        # cardinality stays one row per node and the SET applies exactly once.
        gc.query(
            f"""
            {node_match}
            CALL {{
                WITH sn
                OPTIONAL MATCH (sn)<-[r:HAS_STANDARD_NAME]-()
                DELETE r
            }}
            CALL {{
                WITH sn
                OPTIONAL MATCH (sn)-[r:HAS_UNIT]->()
                DELETE r
            }}
            CALL {{
                WITH sn
                OPTIONAL MATCH (sn)-[r:HAS_COCOS]->()
                DELETE r
            }}
            SET {set_clause}
            """,
            **reset_params,
        )

    logger.info("Reset %d StandardName nodes", count)
    return count


def clear_standard_names(
    *,
    stage_filter: list[str] | None = None,
    source_filter: str | None = None,
    ids_filter: str | None = None,
    path_allowlist: list[str] | None = None,
    include_accepted: bool = False,
    dry_run: bool = False,
    since: str | None = None,
    before: str | None = None,
    below_score: float | None = None,
    tiers: list[str] | None = None,
    validation_status: str | None = None,
) -> int:
    """Delete StandardName nodes and their relationships.

    Safety model (relationship-first):

    1. If ``ids_filter`` or ``source_filter`` is set, delete matching
       ``HAS_STANDARD_NAME`` relationships first.
    2. Then delete ``StandardName`` nodes that have zero remaining
       ``HAS_STANDARD_NAME`` edges.

    With no ``stage_filter`` every live stage is eligible except
    ``accepted`` (requires ``include_accepted=True``); the terminal
    ``superseded`` / ``exhausted`` stages are never deleted unless
    explicitly listed (they carry REFINED_FROM chain history).

    Parameters
    ----------
    stage_filter:
        List of ``name_stage`` values to delete.  ``None`` (default)
        selects all live stages, gated by ``include_accepted``.
    source_filter:
        Restrict to nodes whose ``source_types`` contains ``"dd"`` or
        ``"signals"``.
    ids_filter:
        Delete only names linked to an IMASNode whose id starts with this IDS
        name.  Relationships are removed first; nodes become orphans and are
        then deleted.
    path_allowlist:
        Delete only names linked to an IMASNode whose ``src.id`` is *exactly*
        in this list (``src.id IN $path_allowlist``).  Unlike ``ids_filter``
        this is an exact-path allowlist, not a prefix.  Relationships are
        removed first (relationship-first delete), so names attached to any
        other path — including accepted catalog names — keep their edges and
        are never deleted.  Combines with ``ids_filter``.
    include_accepted:
        Required to delete ``name_stage='accepted'`` names — these are
        committed catalog entries and the operator must opt in explicitly.
    dry_run:
        Return the count of nodes that would be deleted without modifying
        anything.
    since:
        Only clear names with ``generated_at >= this`` ISO timestamp.
    before:
        Only clear names with ``generated_at < this`` ISO timestamp.
    below_score:
        Only clear names with ``reviewer_score < this`` value (0.0–1.0).
    tiers:
        Only clear names with ``review_tier`` in this list.
    validation_status:
        Only clear names with this ``validation_status`` value.

    Returns
    -------
    Number of nodes deleted (or that would be deleted in dry-run mode).
    """
    effective_stages = list(stage_filter) if stage_filter else []
    # include_accepted only AUTHORIZES an explicitly-listed 'accepted' stage;
    # it must NEVER inject 'accepted' into a stage list the operator did not
    # name. ``--stage drafted --include-accepted`` deletes drafted names only —
    # not accepted catalog entries. Without the opt-in, an explicitly-listed
    # 'accepted' is dropped as a protection.
    if stage_filter is not None and not include_accepted:
        effective_stages = [s for s in effective_stages if s != "accepted"]

    with GraphClient() as gc:
        params: dict[str, Any] = {}
        sn_where_clauses: list[str] = []
        if stage_filter is None:
            sn_where_clauses.append(
                "NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted', 'contested']"
            )
            if not include_accepted:
                sn_where_clauses.append("coalesce(sn.name_stage, '') <> 'accepted'")
        else:
            params["stages"] = effective_stages
            sn_where_clauses.append("sn.name_stage IN $stages")

        if source_filter:
            sn_where_clauses.append("$source_filter IN sn.source_types")
            params["source_filter"] = source_filter

        if since:
            sn_where_clauses.append("sn.generated_at >= datetime($since)")
            params["since"] = since
        if before:
            sn_where_clauses.append("sn.generated_at < datetime($before)")
            params["before"] = before
        if below_score is not None:
            sn_where_clauses.append("sn.reviewer_score_name < $below_score")
            params["below_score"] = below_score
        if tiers:
            sn_where_clauses.append("sn.review_tier IN $tiers")
            params["tiers"] = tiers
        if validation_status:
            sn_where_clauses.append("sn.validation_status = $validation_status")
            params["validation_status"] = validation_status

        sn_where = " AND ".join(sn_where_clauses) if sn_where_clauses else "true"

        # Build the optional IMASNode-source scope. ids_filter is a prefix
        # match; path_allowlist is an exact-path membership test. Either (or
        # both) forces the relationship-first delete path so names on other
        # paths keep their edges and survive.
        src_clauses: list[str] = []
        if ids_filter:
            params["ids_prefix"] = ids_filter + "/"
            src_clauses.append("src.id STARTS WITH $ids_prefix")
        # Fail-closed: an empty allowlist selects the scoped branch and
        # matches NOTHING (``src.id IN []``). Treating ``[]`` as falsy here
        # would fall through to the unscoped branch and delete across the
        # whole graph — the 1863-name-wipe shape. ``None`` = no allowlist.
        if path_allowlist is not None:
            params["path_allowlist"] = list(path_allowlist)
            src_clauses.append("src.id IN $path_allowlist")
        use_src_join = bool(src_clauses)
        src_where = " AND ".join(src_clauses)
        # Same scope predicate rebound to an ``o`` producer, for the
        # out-of-scope-producer guard below.
        out_scope_where = src_where.replace("src.", "o.")

        if use_src_join:
            # Count the names that will ACTUALLY be deleted, not merely the ones
            # matched in scope: a name matched via an in-scope edge SURVIVES if
            # it also has a producer outside the scope (relationship-first
            # delete keeps that edge). Counting bare in-scope matches overcounts
            # such survivors, so the reported/dry-run number would exceed the
            # real deletions. Restrict to names whose producers are ALL in
            # scope (no out-of-scope producer → truly orphaned after the edge
            # delete).
            count_cypher = f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {sn_where}
                AND {src_where}
                WITH DISTINCT sn
                WHERE NOT EXISTS {{
                    MATCH (o:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
                    WHERE NOT ({out_scope_where})
                }}
                RETURN count(sn) AS n
            """
        else:
            count_cypher = f"""
                MATCH (sn:StandardName)
                WHERE {sn_where}
                RETURN count(sn) AS n
            """

        result = gc.query(count_cypher, **params)
        count = result[0]["n"] if result else 0
        logger.info(
            "clear_standard_names: %d nodes match (stages=%s, source=%s, ids=%s)",
            count,
            effective_stages or "all-live",
            source_filter,
            ids_filter,
        )

        if dry_run or count == 0:
            return count

        # Delete the in-scope reviews and StandardName nodes in a SINGLE
        # statement so the delete is atomic: a mid-sequence failure can no
        # longer leave a name with its producer edge stripped but the node
        # (or its review) still present, dropping the name out of every future
        # regeneration while orphaning its review.
        #
        # DETACH DELETE on the StandardName alone would orphan its
        # StandardNameReview (HAS_REVIEW goes StandardName -> StandardNameReview),
        # so the review is deleted in the same statement inside a unit CALL
        # subquery.
        if use_src_join:
            # Relationship-first, scope-safe: remove the in-scope producer
            # edges, then (orphan-guard) delete only the names that have NO
            # remaining HAS_STANDARD_NAME edge. Neo4j inserts an Eager barrier
            # between the DELETE and the NOT EXISTS read so the guard sees
            # post-delete state — a name still attached to any out-of-scope
            # path survives with its review intact (survivor-review handling:
            # the review is deleted only for names that actually go away).
            # Verified against a live graph in
            # tests/standard_names/test_clear_atomicity_graph.py.
            gc.query(
                f"""
                MATCH (src:IMASNode)-[rel:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {sn_where}
                AND {src_where}
                DELETE rel
                WITH DISTINCT sn
                WHERE NOT EXISTS {{ MATCH ()-[:HAS_STANDARD_NAME]->(sn) }}
                CALL {{
                    WITH sn
                    OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
                    DETACH DELETE r
                }}
                DETACH DELETE sn
                """,
                **params,
            )
        else:
            gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE {sn_where}
                CALL {{
                    WITH sn
                    OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
                    DETACH DELETE r
                }}
                DETACH DELETE sn
                """,
                **params,
            )

        # Step C: sweep up any fully-orphaned StandardNameReview nodes left by prior clears
        # (pre-p39 runs detached StandardName without deleting StandardNameReview).
        orphan_result = gc.query(
            """
            MATCH (r:StandardNameReview)
            WHERE NOT EXISTS { MATCH (:StandardName)-[:HAS_REVIEW]->(r) }
            WITH r LIMIT 10000
            DETACH DELETE r
            RETURN count(r) AS n
            """
        )
        orphan_count = orphan_result[0]["n"] if orphan_result else 0
        if orphan_count:
            logger.info("Swept %d orphaned StandardNameReview nodes", orphan_count)

        # Step D: delete the LLMCost ledger — but only on an UNSCOPED clear.
        # A full clear owns the whole ledger (every run's StandardName output
        # is being removed, so its cost rows are stale). A SCOPED clear (any
        # stage/source/ids/path/time/score/tier/validation filter) removes
        # only a slice of names and must NOT wipe the global cost ledger —
        # that would erase cost history for the names left intact.
        scoped = (
            stage_filter is not None
            or bool(source_filter)
            or bool(ids_filter)
            or path_allowlist is not None
            or bool(since)
            or bool(before)
            or below_score is not None
            or bool(tiers)
            or bool(validation_status)
        )
        if not scoped:
            gc.query("MATCH (c:LLMCost) DETACH DELETE c")

        # Step E: reset orphaned sources. Deleting a StandardName strands its
        # StandardNameSource at 'composed'/'attached' — a status the generate
        # pool never claims — so without this reset the source silently drops
        # out of all future regeneration. Orphan check (no remaining
        # PRODUCED_NAME edge) is inherently scope-correct: sources whose names
        # survived a filtered clear keep their status. The produced_sn_id
        # scalar mirror is nulled in the same operation so it never outlives
        # the deleted name (PRODUCED_NAME edge is the single source of truth).
        reset_result = gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status IN ['composed', 'attached']
            AND NOT (sns)-[:PRODUCED_NAME]->(:StandardName)
            SET sns.status = 'extracted',
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.produced_sn_id = null
            RETURN count(sns) AS n
            """
        )
        reset_count = reset_result[0]["n"] if reset_result else 0
        if reset_count:
            logger.info(
                "Reset %d orphaned StandardNameSource nodes to 'extracted'",
                reset_count,
            )

    logger.info("Deleted %d StandardName nodes", count)
    return count


def clear_sn_subsystem(
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Wipe every Standard Name the pipeline has produced.

    Deletes the six labels owned by the SN pipeline output:

    * ``StandardName`` — the generated names
    * ``StandardNameReview`` — RD-quorum review records
    * ``StandardNameSource`` — per-path extraction tracking
    * ``DocsRevision`` — refine_docs snapshot history (orphans without parent SN)
    * ``VocabGap`` — grammar vocabulary gap reports
    * ``SNRun`` — run audit / rotation memory
    * ``LLMCost`` — LLM call cost ledger rows

    **Grammar nodes** (``GrammarToken``, ``GrammarSegment``,
    ``GrammarTemplate``, ``ISNGrammarVersion``) are ISN-authoritative
    reference data and are never touched here. They stay in the graph so
    the vocabulary is immediately available for the next ``sn run``; the
    ``sn clear`` CLI re-seeds them from the installed ISN release after a
    wipe, and ``sn run`` auto-syncs at startup when the version differs.

    Parameters
    ----------
    dry_run:
        Count matching nodes without modifying the graph.

    Returns
    -------
    Dict mapping node label to deleted count. In dry-run mode,
    values are the current counts that would be deleted.
    """
    counts: dict[str, int] = {}
    labels = (
        "StandardName",
        "StandardNameReview",
        "StandardNameSource",
        "DocsRevision",
        "VocabGap",
        "SNRun",
        "LLMCost",
    )

    with GraphClient() as gc:

        def _count(label: str) -> int:
            r = gc.query(f"MATCH (n:{label}) RETURN count(n) AS n")
            return r[0]["n"] if r else 0

        for label in labels:
            counts[label] = _count(label)

        if dry_run:
            return counts

        # Delete order is significant: StandardNameReview BEFORE StandardName so
        # orphan StandardNameReview nodes can't linger if HAS_STANDARD_NAME edges
        # are missing (pre-p39 bug). DETACH DELETE handles remaining
        # edges on each pass. At SN-pipeline scale (~thousands of
        # nodes total) a single DETACH DELETE per label is sub-second.
        gc.query("MATCH (r:StandardNameReview) DETACH DELETE r")
        gc.query("MATCH (sn:StandardName) DETACH DELETE sn")
        gc.query("MATCH (s:StandardNameSource) DETACH DELETE s")
        gc.query("MATCH (d:DocsRevision) DETACH DELETE d")
        gc.query("MATCH (v:VocabGap) DETACH DELETE v")
        gc.query("MATCH (rr:SNRun) DETACH DELETE rr")
        gc.query("MATCH (c:LLMCost) DETACH DELETE c")

    total = sum(counts.values())
    logger.info("clear_sn_subsystem: deleted %d nodes (%s)", total, counts)

    return counts


# =============================================================================
# Link resolution
# =============================================================================

_MAX_LINK_RETRIES = 5


def claim_unresolved_links(limit: int = 20) -> list[dict[str, Any]]:
    """Claim StandardName nodes with unresolved links for resolution.

    Uses age-weighted random selection: nodes not checked recently have
    higher priority, preventing spin on temporarily unresolvable links.

    Returns list of dicts with ``id``, ``links``, ``link_retry_count``.
    """
    import uuid

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.link_status = 'unresolved'
              AND sn.claimed_at IS NULL
              AND coalesce(sn.link_retry_count, 0) < $max_retries
            WITH sn,
                 duration.between(
                     coalesce(sn.link_checked_at, datetime('2020-01-01')),
                     datetime()
                 ).minutes + 1.0 AS age_minutes
            ORDER BY rand() * age_minutes DESC
            LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            limit=limit,
            max_retries=_MAX_LINK_RETRIES,
            token=token,
        )
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName {claim_token: $token})
                RETURN sn.id AS id, sn.links AS links,
                       coalesce(sn.link_retry_count, 0) AS retry_count
                """,
                token=token,
            )
        )
    return [dict(r) for r in rows]


def resolve_links_batch(
    items: list[dict[str, Any]],
    *,
    override: bool = False,
    override_names: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve dd: links to name: links for a batch of names.

    For each ``dd:path`` link, checks if a StandardName exists that was
    generated from that path. If found, replaces with ``name:sn_id``.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection on catalog-edited names.
    override_names:
        Selective override — set of name IDs that should bypass protection
        even if they have ``origin='catalog_edit'``.
    """
    # Pipeline protection — links is a protected field
    from imas_codex.standard_names.protection import filter_protected

    items, skipped = filter_protected(
        items, override=override, override_names=override_names
    )
    if skipped:
        logger.warning(
            "resolve_links_batch: stripped protected fields from %d name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )

    resolved_count = 0
    still_unresolved = 0
    failed_count = 0

    with GraphClient() as gc:
        # Build lookup: dd_path -> standard_name_id
        all_dd_paths = set()
        for item in items:
            for link in item.get("links") or []:
                if link.startswith("dd:"):
                    all_dd_paths.add(link[3:])

        path_to_name: dict[str, str] = {}
        if all_dd_paths:
            rows = gc.query(
                """
                UNWIND $paths AS path
                MATCH (n:IMASNode {id: path})-[:HAS_STANDARD_NAME]->(sn:StandardName)
                RETURN path AS dd_path, sn.id AS sn_id
                """,
                paths=list(all_dd_paths),
            )
            for r in rows:
                path_to_name[r["dd_path"]] = r["sn_id"]

        for item in items:
            links = list(item.get("links") or [])
            new_links = []
            any_unresolved = False

            for link in links:
                if link.startswith("dd:"):
                    dd_path = link[3:]
                    if dd_path in path_to_name:
                        new_links.append(f"name:{path_to_name[dd_path]}")
                    else:
                        new_links.append(link)
                        any_unresolved = True
                else:
                    new_links.append(link)

            retry_count = item.get("retry_count", 0) + 1

            if any_unresolved and retry_count >= _MAX_LINK_RETRIES:
                status = "failed"
                failed_count += 1
            elif any_unresolved:
                status = "unresolved"
                still_unresolved += 1
            else:
                status = "resolved"
                resolved_count += 1

            gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                SET sn.links = $links,
                    sn.link_status = $status,
                    sn.link_checked_at = datetime(),
                    sn.link_retry_count = $retry_count,
                    sn.claimed_at = null,
                    sn.claim_token = null
                """,
                id=item["id"],
                links=new_links,
                status=status,
                retry_count=retry_count,
            )

    return {
        "resolved": resolved_count,
        "unresolved": still_unresolved,
        "failed": failed_count,
    }


# =============================================================================
# Documentation link resolution — post-drain cleanup
# =============================================================================


_NAME_LINK_RE = re.compile(r"\(name:([^)]+)\)")
# Bare ``[snake_case_name]`` brackets with NO ``(name:...)`` target — the LLM
# frequently writes these for related-name mentions; Markdown renders them as
# broken literal text. ``(?<!\!)`` skips image syntax; ``(?!\()`` skips
# already-formed ``[label](...)`` links.
_BARE_DOC_LINK_RE = re.compile(r"(?<!\!)\[([a-z][a-z0-9_]{3,})\](?!\()")

# When the LLM emits LaTeX with single backslashes inside its JSON output
# (e.g. ``$\theta$``, ``$\rho$``, ``\frac``), a lenient JSON decoder turns the
# ``\t``/``\r``/``\f``/``\b``/``\v`` prefix into the corresponding control
# character (TAB/CR/FF/BS/VT), corrupting the LaTeX (``$\rho$`` -> ``$<CR>ho$``).
# These control chars never appear legitimately in physics documentation, so map
# them back to the backslash form to reconstruct the command; any OTHER control
# char (NUL, ETX, …) is unrecoverable corruption and is stripped. Newline is kept.
_JSON_ESCAPE_CTRL = {"\b": r"\b", "\t": r"\t", "\f": r"\f", "\r": r"\r", "\v": r"\v"}
_STRIP_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")  # all control chars except \n


def _sanitize_doc_text(text: str | None) -> str | None:
    """Repair JSON-escape-mangled LaTeX control characters in LLM doc text."""
    if not text:
        return text
    for ctrl, repl in _JSON_ESCAPE_CTRL.items():
        if ctrl in text:
            text = text.replace(ctrl, repl)
    if _STRIP_CTRL_RE.search(text):
        text = _STRIP_CTRL_RE.sub("", text)
    return text


_DOC_LINK_RE = re.compile(r"\[([^\]]+)\]\(name:([a-z0-9_]+)\)")


def _doc_link_mismatches(gc: Any, documentation: str | None) -> list[str]:
    """Return ``[label](name:target)`` mismatches in *documentation*.

    A link is a mismatch when its snake_cased label is itself an EXISTING
    StandardName id different from the link target — the text promises one
    quantity while the href resolves to another. Human-readable labels that
    are not themselves ids are fine. Returns human-readable findings like
    ``"[radial_current_density] -> name:radial_total_current_density"``.
    """
    if not documentation:
        return []
    pairs = [
        (label.strip().lower().replace(" ", "_").replace("-", "_"), target, label)
        for label, target in _DOC_LINK_RE.findall(documentation)
    ]
    candidates = {lid for lid, target, _ in pairs if lid != target}
    if not candidates:
        return []
    rows = gc.query(
        "MATCH (x:StandardName) WHERE x.id IN $ids RETURN x.id AS id",
        ids=sorted(candidates),
    )
    existing = {r["id"] for r in (rows or [])}
    return [
        f"[{label}] -> name:{target}"
        for lid, target, label in pairs
        if lid != target and lid in existing
    ]


def _normalize_bare_doc_links(gc: Any, sn_id: str | None = None) -> int:
    """Convert bare ``[name]`` brackets in documentation to proper links or text.

    For every StandardName documentation containing ``[``: each bare
    ``[snake_case]`` (no ``(name:...)`` target) becomes ``[snake_case](name:snake_case)``
    when ``snake_case`` is a real StandardName id, else the brackets are stripped
    (leaving the plain word). Idempotent. Returns the number of docs updated.

    When *sn_id* is given the pass is scoped to that single node (the cheap
    accept-path variant — see :func:`persist_reviewed_docs`): only the tokens
    present in that one doc are checked for liveness, so no full-catalogue scan
    is performed.  When *sn_id* is ``None`` (the post-drain reconcile) it scans
    every doc and resolves token liveness against the whole catalogue.
    """
    if sn_id is not None:
        rows = list(
            gc.query(
                "MATCH (sn:StandardName {id: $id}) "
                "WHERE sn.documentation IS NOT NULL AND sn.documentation CONTAINS '[' "
                "RETURN sn.id AS id, sn.documentation AS docs",
                id=sn_id,
            )
        )
    else:
        rows = list(
            gc.query(
                "MATCH (sn:StandardName) "
                "WHERE sn.documentation IS NOT NULL AND sn.documentation CONTAINS '[' "
                "RETURN sn.id AS id, sn.documentation AS docs"
            )
        )
    if not rows:
        return 0
    # Only link to a NON-dead standard name (drafted/reviewed/accepted); a bare
    # token that is superseded/exhausted or not a name at all → strip brackets.
    if sn_id is not None:
        # Single-node scope: check liveness only for the bare tokens that
        # actually appear in this one doc — avoids a full-catalogue scan on
        # every accept.
        candidate_tokens = {
            m.group(1) for r in rows for m in _BARE_DOC_LINK_RE.finditer(r["docs"])
        }
        names = (
            {
                r["id"]
                for r in gc.query(
                    "MATCH (s:StandardName) "
                    "WHERE s.id IN $toks "
                    "AND NOT coalesce(s.name_stage, '') IN ['superseded', 'exhausted', 'contested'] "
                    "RETURN s.id AS id",
                    toks=list(candidate_tokens),
                )
            }
            if candidate_tokens
            else set()
        )
    else:
        names = {
            r["id"]
            for r in gc.query(
                "MATCH (s:StandardName) "
                "WHERE NOT coalesce(s.name_stage, '') IN ['superseded', 'exhausted', 'contested'] "
                "RETURN s.id AS id"
            )
        }

    def _repl(m: re.Match[str]) -> str:
        tok = m.group(1)
        return f"[{tok}](name:{tok})" if tok in names else tok

    updates = []
    for r in rows:
        new = _BARE_DOC_LINK_RE.sub(_repl, r["docs"])
        if new != r["docs"]:
            updates.append({"id": r["id"], "doc": new})
    for i in range(0, len(updates), 200):
        gc.query(
            "UNWIND $items AS it MATCH (sn:StandardName {id: it.id}) "
            "SET sn.documentation = it.doc",
            items=updates[i : i + 200],
        )
    if updates:
        logger.info(
            "resolve_doc_links: normalized bare [name] brackets in %d doc(s)%s",
            len(updates),
            f" (scoped to {sn_id})" if sn_id else "",
        )
    return len(updates)


def resolve_doc_links(gc: Any | None = None) -> dict[str, int]:
    """Rewrite stale (name:xxx) references in documentation text.

    Scans accepted StandardName documentation for ``(name:xxx)`` markdown
    links.  When a referenced name is superseded or exhausted:

    1. **Superseded with accepted successor** (via ``REFINED_FROM`` chain):
       rewrite the link to point to the accepted successor.
    2. **Dead-end** (exhausted/superseded with no accepted successor):
       remove the hyperlink, keeping the descriptive text intact.

    Also updates the cached ``links`` list and ``link_status`` to match
    the cleaned documentation.

    Parameters
    ----------
    gc:
        Optional active GraphClient.  A new one is opened if None.

    Returns
    -------
    Dict with keys: ``resolved`` (rewritten to successor), ``removed``
    (dead link stripped), ``unchanged`` (already valid).
    """
    _own_gc = gc is None
    if _own_gc:
        gc = GraphClient().__enter__()
    try:
        # Pass 0: normalize bare ``[name]`` brackets across ALL docs (any stage).
        # The LLM writes related-name mentions as bare ``[name]`` despite the
        # prompt rule; Markdown renders these broken. Deterministically convert
        # ``[x]`` → ``[x](name:x)`` when x is a real standard name, else strip the
        # brackets to plain text. Runs every rotation (this fn is called from the
        # pool loop's post-rotation reconcile).
        _normalize_bare_doc_links(gc)

        # Fetch all accepted names with documentation
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.name_stage IN ['accepted', 'approved']
                  AND sn.documentation IS NOT NULL
                RETURN sn.id AS id, sn.documentation AS docs, sn.links AS links
                """
            )
        )
        if not rows:
            return {"resolved": 0, "removed": 0, "unchanged": 0}

        # Valid link targets = every StandardName that is NOT genuinely dead
        # (superseded / exhausted). A link to a drafted/reviewed/accepted name
        # is valid and must be KEPT — only superseded/exhausted targets are
        # rewritten to an accepted successor or removed. (Using accepted-only
        # here wrongly stripped links to not-yet-accepted names — a link to a
        # drafted SN is a legitimate cross-reference that will publish.)
        valid_targets = {
            r["id"]
            for r in gc.query(
                "MATCH (s:StandardName) "
                "WHERE NOT coalesce(s.name_stage, '') IN ['superseded', 'exhausted', 'contested'] "
                "RETURN s.id AS id"
            )
        }

        # Build dead-target→accepted-successor resolution map via REFINED_FROM
        superseded_ids: set[str] = set()
        for row in rows:
            docs = row.get("docs") or ""
            for match in _NAME_LINK_RE.finditer(docs):
                target = match.group(1)
                if target not in valid_targets:
                    superseded_ids.add(target)

        if not superseded_ids:
            return {"resolved": 0, "removed": 0, "unchanged": 0}

        # Resolve each stale target to its accepted successor
        resolution_map: dict[str, str | None] = {}
        for target in superseded_ids:
            successors = gc.query(
                """
                MATCH (new:StandardName)-[:REFINED_FROM*]->(old:StandardName {id: $target})
                WHERE new.name_stage IN ['accepted', 'approved']
                RETURN new.id AS successor
                LIMIT 1
                """,
                target=target,
            )
            resolution_map[target] = successors[0]["successor"] if successors else None

        # Rewrite documentation and update graph
        resolved = 0
        removed = 0
        unchanged = 0

        for row in rows:
            docs = row.get("docs") or ""
            original_docs = docs

            def _replace_link(m: re.Match) -> str:
                nonlocal resolved, removed
                target = m.group(1)
                if target in valid_targets:
                    return m.group(0)  # valid (non-dead) link, keep as-is
                successor = resolution_map.get(target)
                if successor:
                    resolved += 1
                    return f"(name:{successor})"
                else:
                    removed += 1
                    return ""  # remove dead link marker

            new_docs = _NAME_LINK_RE.sub(_replace_link, docs)

            if new_docs != original_docs:
                # Extract updated links list from documentation. Order-
                # preserving dedup, drop self-references — the ISNC
                # SQLite UNIQUE(name, link) constraint trips on repeats
                # (e.g. plasma_stored_energy referencing itself three
                # times across separate paragraphs) and a self-link in
                # the structured `links` index is meaningless.
                self_link = f"name:{row['id']}"
                seen: set[str] = set()
                new_links: list[str] = []
                for m in _NAME_LINK_RE.finditer(new_docs):
                    ref = f"name:{m.group(1)}"
                    if ref == self_link or ref in seen:
                        continue
                    seen.add(ref)
                    new_links.append(ref)
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $id})
                    SET sn.documentation = $docs,
                        sn.links = $links,
                        sn.link_status = 'resolved'
                    """,
                    id=row["id"],
                    docs=new_docs,
                    links=new_links,
                )
            else:
                unchanged += 1

        logger.info(
            "resolve_doc_links: resolved=%d rewritten, %d removed, %d unchanged",
            resolved,
            removed,
            unchanged,
        )
        return {"resolved": resolved, "removed": removed, "unchanged": unchanged}
    finally:
        if _own_gc:
            gc.__exit__(None, None, None)


# =============================================================================
# Enrichment helpers — documentation iteration (Phase 3D)
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
            SET sn.description = b.description,
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
            source.update(by_id[source["id"]])
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
                sns.claim_token = $token
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
                       sns.description AS description
                """,
                token=token,
            )
        )
    return token, claimed


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
    source_ids: list[str],
    standard_name_id: str,
) -> int:
    """Mark sources as composed and link to the produced StandardName.

    Token-verified: only updates sources matching the claim_token.
    Creates PRODUCED_NAME relationship to the StandardName.
    Returns count of updated sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid, claim_token: $token})
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
            sn_id=standard_name_id,
        )
        return result[0]["affected"] if result else 0


def mark_sources_attached(
    token: str,
    source_ids: list[str],
    standard_name_id: str,
) -> int:
    """Mark sources as auto-attached to an existing StandardName.

    Used when a source matches an existing name without needing LLM composition.
    Token-verified. Creates PRODUCED_NAME relationship.
    Returns count of updated sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid, claim_token: $token})
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
            sn_id=standard_name_id,
        )
        return result[0]["affected"] if result else 0


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
    """
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

_CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes — matches DEFAULT_CLAIM_TIMEOUT_SECONDS

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


@retry_on_deadlock()
def claim_compose_sources(
    *,
    limit: int = 15,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> tuple[str, list[dict]]:
    """Claim extracted StandardNameSource nodes for composition (polling).

    Unlike :func:`claim_standard_name_source_batch` this does NOT filter by
    ``batch_key`` — it claims any extracted source. Workers use this in a
    polling loop to pick up the next available batch of work regardless of
    which batch grouping the extract phase produced.

    Uses the standard ``ORDER BY rand()`` + ``claim_token`` two-step verify
    pattern from the discovery pipelines.

    Returns ``(token, claimed_sources)`` where each source dict has keys
    ``id``, ``source_id``, ``source_type``, ``batch_key``, ``description``.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"

    with GraphClient() as gc:
        # Step 1: Atomically claim with random ordering + unique token
        gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status = 'extracted'
              AND (sns.claimed_at IS NULL
                   OR sns.claimed_at < datetime() - duration($cutoff))
            WITH sns ORDER BY rand() LIMIT $limit
            SET sns.claimed_at = datetime(),
                sns.claim_token = $token
            """,
            limit=limit,
            token=token,
            cutoff=cutoff,
        )
        # Step 2: Verify — only our token
        claimed = list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description
                """,
                token=token,
            )
        )

    logger.debug(
        "claim_compose_sources: requested %d, won %d (token=%s)",
        limit,
        len(claimed),
        token[:8],
    )
    return token, [dict(r) for r in claimed]


def count_extracted_for_domain(domain: str, source: str = "dd") -> int:
    """Count StandardNameSource nodes already extracted for a physics domain.

    Used by :func:`run_sn_pools` to decide whether the domain-specific
    extract seed step is needed before starting the pools.

    Args:
        domain: Physics domain string (e.g. ``"turbulence"``).
        source: Source type — ``"dd"`` (DD paths) or ``"signals"``.

    Returns:
        Number of extracted sources whose backing entity belongs to *domain*.
    """
    if source == "dd":
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (sns:StandardNameSource)-[:FROM_DD_PATH]->(n:IMASNode)
                WHERE sns.status = 'extracted'
                  AND n.physics_domain = $domain
                RETURN count(sns) AS cnt
                """,
                domain=domain,
            )
            return result[0]["cnt"] if result else 0
    return 0


def count_eligible_compose_sources(
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> int:
    """Count StandardNameSource nodes eligible for composition.

    Returns the number of extracted, unclaimed (or stale-claimed) sources.
    Used by polling workers for drain detection — when this returns 0 and
    no active leases remain, the compose phase is complete.
    """
    cutoff = f"PT{timeout_seconds}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status = 'extracted'
              AND (sns.claimed_at IS NULL
                   OR sns.claimed_at < datetime() - duration($cutoff))
            RETURN count(sns) AS cnt
            """,
            cutoff=cutoff,
        )
        return result[0]["cnt"] if result else 0


@retry_on_deadlock()
def claim_review_names(
    name_ids: list[str],
    *,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> tuple[str, list[str]]:
    """Claim specific StandardName nodes for review scoring.

    Only claims names from *name_ids* that are still eligible — not already
    claimed by another worker (unless the claim is stale).

    Uses the same ``claim_token`` two-step verify pattern as the compose
    claim functions.

    Returns ``(token, actually_claimed_ids)``.
    """
    if not name_ids:
        return "", []

    token = str(uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"

    with GraphClient() as gc:
        # Step 1: Atomically claim unclaimed/stale-claimed names
        gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            WHERE sn.claimed_at IS NULL
               OR sn.claimed_at < datetime() - duration($cutoff)
            SET sn.claimed_at = datetime(),
                sn.claim_token = $token
            """,
            ids=name_ids,
            token=token,
            cutoff=cutoff,
        )
        # Step 2: Verify — only our token
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            RETURN sn.id AS id
            """,
            token=token,
        )
        claimed = [r["id"] for r in result] if result else []

    logger.debug(
        "claim_review_names: requested %d, won %d (token=%s)",
        len(name_ids),
        len(claimed),
        token[:8],
    )
    return token, claimed


def release_review_claims(token: str) -> int:
    """Release all StandardName claims held by this token.

    Clears ``claimed_at`` and ``claim_token`` without changing any other
    fields.  Used for error recovery — released names become eligible for
    other workers.

    Returns count of released names.
    """
    if not token:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS affected
            """,
            token=token,
        )
        return result[0]["affected"] if result else 0


# Heartbeat interval: refresh a held claim well before the lease TTL so a
# long quorum-review / enrich batch cannot have its lease expire mid-flight
# and be re-claimed by a peer worker (which would duplicate the paid LLM
# spend). One third of the TTL leaves room for two refreshes before expiry
# even if one heartbeat is delayed.
_CLAIM_HEARTBEAT_SECONDS = _CLAIM_TIMEOUT_SECONDS // 3


def refresh_name_claims(sn_ids: list[str], claim_token: str) -> int:
    """Refresh the lease on StandardName nodes still held by *claim_token*.

    Compare-and-set on ``claim_token``: only nodes that STILL bear this token
    have their ``claimed_at`` bumped to now. A node whose lease already
    expired and was re-claimed by a peer (its ``claim_token`` overwritten) is
    left untouched — the heartbeat never steals back a claim it has lost. This
    is the liveness half of the claim lease: the claim SEED is a compare-and-set
    (two-step token verify); this keeps a legitimately-held claim from expiring
    under a batch that outruns the TTL.

    Returns the number of nodes whose lease was refreshed.
    """
    if not claim_token or not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid, claim_token: $token})
            SET sn.claimed_at = datetime()
            RETURN count(sn) AS refreshed
            """,
            ids=list(sn_ids),
            token=claim_token,
        )
        return result[0]["refreshed"] if result else 0


def promote_stranded_reviewed(
    min_score: float = DEFAULT_MIN_SCORE,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Promote names stranded at ``'reviewed'`` whose stored score now clears
    the active threshold.

    A name is scored once and its stage is decided against the threshold in
    force AT THAT TIME. When the acceptance threshold is later lowered, names
    that scored between the old and new thresholds are stuck at
    ``name_stage='reviewed'`` (or ``docs_stage='reviewed'``): the refine pool
    only claims names BELOW the threshold, so a name whose stored score already
    clears the current threshold is never re-touched and never accepted. This
    idempotent pass flips those to ``'accepted'`` on both axes.

    Guards (never promote a name that must go through the normal path):

    * ``reviewer_score_* >= min_score`` — only names that genuinely clear the
      CURRENT threshold; below-threshold names are left for refine.
    * name axis excludes ``edit_status='open'`` — a name carrying an unapplied
      edit (a rename / family-or-subtree cascade) must be accepted through
      :func:`persist_reviewed_name`, which applies the edit and cascades
      descendants atomically; a bare stage flip here would strand that cascade.
    * both axes exclude ``validation_status='quarantined'``.
    * docs axis requires ``name_stage='accepted'`` (docs is a post-name axis).

    Idempotent: a second run matches nothing (the names are already accepted).

    Returns ``{"name": n_name, "docs": n_docs}`` — the number promoted on each
    axis (or that would be promoted in ``dry_run``).
    """
    name_where = (
        "sn.name_stage = 'reviewed' "
        "AND sn.reviewer_score_name >= $min_score "
        "AND coalesce(sn.edit_status, '') <> 'open' "
        "AND coalesce(sn.validation_status, '') <> 'quarantined'"
    )
    docs_where = (
        "sn.docs_stage = 'reviewed' "
        "AND sn.reviewer_score_docs >= $min_score "
        "AND sn.name_stage = 'accepted' "
        "AND coalesce(sn.validation_status, '') <> 'quarantined'"
    )
    with GraphClient() as gc:
        if dry_run:
            n_name = gc.query(
                f"MATCH (sn:StandardName) WHERE {name_where} RETURN count(sn) AS n",
                min_score=min_score,
            )
            n_docs = gc.query(
                f"MATCH (sn:StandardName) WHERE {docs_where} RETURN count(sn) AS n",
                min_score=min_score,
            )
            return {
                "name": n_name[0]["n"] if n_name else 0,
                "docs": n_docs[0]["n"] if n_docs else 0,
            }
        # Name axis first so a name promoted here is eligible for the docs-axis
        # promotion in the same pass.
        r_name = gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {name_where}
            SET sn.name_stage = 'accepted'
            RETURN count(sn) AS n
            """,
            min_score=min_score,
        )
        r_docs = gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {docs_where}
            SET sn.docs_stage = 'accepted'
            RETURN count(sn) AS n
            """,
            min_score=min_score,
        )
    promoted = {
        "name": r_name[0]["n"] if r_name else 0,
        "docs": r_docs[0]["n"] if r_docs else 0,
    }
    if promoted["name"] or promoted["docs"]:
        logger.info(
            "promote_stranded_reviewed: promoted %d name(s) + %d docs to "
            "accepted (stored score >= %.3f threshold)",
            promoted["name"],
            promoted["docs"],
            min_score,
        )
    return promoted


def reconcile_standard_name_sources(source_type: str = "dd") -> dict:
    """Post-rebuild reconciliation of StandardNameSource nodes.

    1. Re-links sources to DD paths/signals that still exist
    2. Marks sources as stale if their upstream entity is gone
    3. Revives stale sources if their entity reappears

    Returns dict with counts: {relinked, stale_marked, revived}.
    """
    with GraphClient() as gc:
        if source_type == "dd":
            # Find stale: sources whose DD path no longer exists, or whose
            # node the DD-version lifecycle stamped removed (absent from the
            # current DD) — such sources must never re-enter the queue.
            stale = list(
                gc.query(
                    """
                    MATCH (sns:StandardNameSource {source_type: 'dd'})
                    WHERE (
                        NOT (sns)-[:FROM_DD_PATH]->(:IMASNode)
                        OR EXISTS { MATCH (l:IMASNode {id: sns.source_id})
                                    WHERE l.lifecycle_status = 'removed' }
                    )
                    AND NOT (sns.status = 'stale')
                    RETURN sns.id AS id
                    """
                )
            )
            stale_ids = [r["id"] for r in stale]
            stale_count = mark_sources_stale(stale_ids)

            # Revive: stale sources whose DD path now exists again
            revived = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'dd', status: 'stale'})
                MATCH (imas:IMASNode {id: sns.source_id})
                WHERE coalesce(imas.lifecycle_status, '') <> 'removed'
                SET sns.status = 'extracted',
                    sns.claimed_at = null,
                    sns.claim_token = null
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
                RETURN count(sns) AS count
                """
            )
            revived_count = revived[0]["count"] if revived else 0

            # Re-link: ensure FROM_DD_PATH exists for non-stale sources
            relinked = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'dd'})
                WHERE NOT (sns.status = 'stale')
                  AND NOT (sns)-[:FROM_DD_PATH]->()
                MATCH (imas:IMASNode {id: sns.source_id})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
                RETURN count(sns) AS count
                """
            )
            relinked_count = relinked[0]["count"] if relinked else 0
        else:
            # Signal reconciliation (same pattern, different relationships)
            stale = list(
                gc.query(
                    """
                    MATCH (sns:StandardNameSource {source_type: 'signals'})
                    WHERE NOT (sns)-[:FROM_SIGNAL]->(:FacilitySignal)
                    AND NOT (sns.status = 'stale')
                    RETURN sns.id AS id
                    """
                )
            )
            stale_ids = [r["id"] for r in stale]
            stale_count = mark_sources_stale(stale_ids)

            revived = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'signals', status: 'stale'})
                MATCH (sig:FacilitySignal {id: sns.source_id})
                SET sns.status = 'extracted',
                    sns.claimed_at = null,
                    sns.claim_token = null
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
                RETURN count(sns) AS count
                """
            )
            revived_count = revived[0]["count"] if revived else 0

            relinked = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'signals'})
                WHERE NOT (sns.status = 'stale')
                  AND NOT (sns)-[:FROM_SIGNAL]->()
                MATCH (sig:FacilitySignal {id: sns.source_id})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
                RETURN count(sns) AS count
                """
            )
            relinked_count = relinked[0]["count"] if relinked else 0

    return {
        "stale_marked": stale_count,
        "revived": revived_count,
        "relinked": relinked_count,
    }


def reconcile_vocab_gaps() -> dict[str, int]:
    """Re-validate all VocabGap nodes against the current ISN vocabulary.

    Performs three passes on every VocabGap node:

    1. **Delete false positives** — token now exists in the reported segment
       (ISN was upgraded since the gap was written).
    2. **Delete invalid-segment gaps** — segment name is not in ISN grammar.
    3. **Reclassify surviving gaps** — recompute ``category`` and
       ``actual_segments`` against the current ISN installation.  A gap
       previously ``absent`` may become ``wrong_slot_placement`` if ISN added
       the token in a different segment.

    Returns dict with counts:
        checked, deleted_false_positive, deleted_invalid_segment,
        deleted_open_segment, reclassified, remaining.
    """
    from imas_codex.standard_names.segments import (
        classify_gap,
        known_segments,
    )

    segs = known_segments()
    if segs is None:
        logger.warning(
            "reconcile_vocab_gaps: ISN unavailable — skipping reconciliation"
        )
        return {"checked": 0, "skipped": True}

    with GraphClient() as gc:
        all_gaps = list(
            gc.query(
                """
                MATCH (vg:VocabGap)
                RETURN vg.id AS id, vg.segment AS segment,
                       vg.token AS token, vg.category AS category
                """
            )
        )

    if not all_gaps:
        return {"checked": 0, "remaining": 0}

    to_delete: list[str] = []
    to_update: list[dict] = []
    stats: dict[str, int] = {
        "checked": len(all_gaps),
        "deleted_false_positive": 0,
        "deleted_invalid_segment": 0,
        "deleted_open_segment": 0,
        "reclassified": 0,
    }

    for gap in all_gaps:
        segment, token = gap["segment"], gap["token"]
        new_category, new_actual = classify_gap(segment, token)

        if new_category == "false_positive":
            to_delete.append(gap["id"])
            stats["deleted_false_positive"] += 1
        elif new_category == "invalid_segment":
            to_delete.append(gap["id"])
            stats["deleted_invalid_segment"] += 1
        elif new_category == "open_segment":
            to_delete.append(gap["id"])
            stats["deleted_open_segment"] += 1
        elif new_category != gap.get("category"):
            to_update.append(
                {
                    "id": gap["id"],
                    "category": new_category,
                    "actual_segments": new_actual,
                }
            )
            stats["reclassified"] += 1

    with GraphClient() as gc:
        if to_delete:
            gc.query(
                """
                UNWIND $ids AS gap_id
                MATCH (vg:VocabGap {id: gap_id})
                DETACH DELETE vg
                """,
                ids=to_delete,
            )
            logger.info(
                "reconcile_vocab_gaps: deleted %d false-positive/invalid gaps",
                len(to_delete),
            )

        if to_update:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (vg:VocabGap {id: b.id})
                SET vg.category = b.category,
                    vg.actual_segments = b.actual_segments,
                    vg.reconciled_at = datetime()
                """,
                batch=to_update,
            )
            logger.info(
                "reconcile_vocab_gaps: reclassified %d gaps",
                len(to_update),
            )

        # Sources parked at 'vocab_gap' whose blocking gaps have all been
        # resolved (no remaining source-first HAS_STANDARD_NAME_VOCAB_GAP edge)
        # revert to 'extracted' so the next generate pool retries them under the
        # upgraded vocabulary. Without this, a vocabulary fix never un-blocks the
        # sources that reported the gap. Uses the direct source→VocabGap edge; a
        # deleted gap node removes it via DETACH, a reclassified one keeps it.
        retried = gc.query(
            """
            MATCH (sns:StandardNameSource {status: 'vocab_gap'})
            WHERE NOT (sns)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(:VocabGap)
            SET sns.status = 'extracted',
                sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS n
            """
        )
        rows = list(retried) if retried else []
        stats["sources_retried"] = rows[0].get("n", 0) if rows else 0
        if stats["sources_retried"]:
            logger.info(
                "reconcile_vocab_gaps: reset %d vocab_gap sources to 'extracted'",
                stats["sources_retried"],
            )

    stats["remaining"] = stats["checked"] - len(to_delete)
    return stats


def reconcile_provenance() -> dict[str, int]:
    """Realign StandardNameSource provenance metadata with live StandardNames.

    The ``produced_sn_id`` scalar mirrors the ``PRODUCED_NAME`` edge, but on
    name supersede/delete/reset only the edge is removed — the scalar mirror
    is orphaned and silently points at a now-deleted name. Likewise a
    ``derived_parent`` scaffolding source can outlive the parent StandardName
    it was minted for. Both are audit pollution, not pipeline blockers, so
    this idempotent sweep is safe to run every rotation.

    Three passes:

    0. **Reattach live desyncs** — any ``StandardNameSource`` whose
       ``produced_sn_id`` scalar names a LIVE ``StandardName`` but whose
       ``PRODUCED_NAME`` edge is missing has the edge MERGEd back (the scalar
       is the recovery mirror; a missing edge is silent provenance loss).
    1. **NULL stale scalars** — any ``StandardNameSource.produced_sn_id`` whose
       target ``StandardName`` no longer exists is set to null.
    2. **Delete orphaned derived-parent scaffolding** — any
       ``StandardNameSource {batch_key: 'derived_parent'}`` (source_type
       ``derived`` or ``dd``) whose parent ``StandardName(id = source_id)``
       no longer exists is DETACH-DELETEd.

    Returns dict with counts: {edges_reattached, scalars_cleared,
    orphan_sources_deleted}.
    """
    with GraphClient() as gc:
        # Pass 0: heal live scalar/missing-edge desyncs before cleanup. Disjoint
        # from pass 1 (that only touches scalars whose target no longer exists).
        edges_reattached = reattach_produced_name_edges(gc=gc)

        scalars = gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.produced_sn_id IS NOT NULL
              AND NOT EXISTS {
                  MATCH (sn:StandardName {id: sns.produced_sn_id})
              }
            SET sns.produced_sn_id = null
            RETURN count(sns) AS n
            """
        )
        scalars_cleared = scalars[0]["n"] if scalars else 0

        orphans = gc.query(
            """
            MATCH (sns:StandardNameSource {batch_key: 'derived_parent'})
            WHERE sns.source_type IN ['derived', 'dd']
              AND NOT EXISTS {
                  MATCH (sn:StandardName {id: sns.source_id})
              }
            DETACH DELETE sns
            RETURN count(sns) AS n
            """
        )
        orphan_sources_deleted = orphans[0]["n"] if orphans else 0

    if edges_reattached or scalars_cleared or orphan_sources_deleted:
        logger.info(
            "reconcile_provenance: reattached %d missing PRODUCED_NAME edge(s), "
            "cleared %d stale produced_sn_id scalar(s), "
            "deleted %d orphaned derived-parent source(s)",
            edges_reattached,
            scalars_cleared,
            orphan_sources_deleted,
        )

    return {
        "edges_reattached": edges_reattached,
        "scalars_cleared": scalars_cleared,
        "orphan_sources_deleted": orphan_sources_deleted,
    }


def reconcile_standard_name_cocos_links(gc: Any | None = None) -> dict[str, int]:
    """Link COCOS-dependent standard names to the catalog's COCOS convention.

    A standard name with a ``cocos_transformation_type`` (psi_like, ip_like, …)
    is COCOS-dependent: its sign depends on the convention. The catalog follows
    a single convention — the current DD version's (DDv4 → COCOS 17) — so every
    such name should carry the integer ``cocos`` and a ``HAS_COCOS`` edge to
    that COCOS singleton. COCOS-independent names (no transformation type) are
    left unlinked by design.

    Idempotent and re-runnable: sets the ``cocos`` scalar where null and MERGEs
    the ``HAS_COCOS`` edge where missing, across every origin (pipeline, derived
    parents, catalog edits). No-op once the invariant holds.

    Returns dict: {convention, scalars_set, edges_created}.
    """
    own = gc is None
    client = GraphClient() if own else gc
    try:
        row = client.query(
            "MATCH (v:DDVersion {is_current: true}) RETURN v.cocos AS cocos"
        )
        convention = (
            row[0]["cocos"] if row and row[0].get("cocos") is not None else None
        )
        if convention is None:
            logger.debug(
                "reconcile_standard_name_cocos_links: no current DD COCOS convention — skipping"
            )
            return {"convention": 0, "scalars_set": 0, "edges_created": 0}

        scalars = client.query(
            """
            MATCH (s:StandardName)
            WHERE s.cocos_transformation_type IS NOT NULL AND s.cocos IS NULL
            SET s.cocos = $conv
            RETURN count(s) AS n
            """,
            conv=convention,
        )
        scalars_set = scalars[0]["n"] if scalars else 0

        edges = client.query(
            """
            MATCH (s:StandardName)
            WHERE s.cocos_transformation_type IS NOT NULL
              AND s.cocos IS NOT NULL
              AND NOT (s)-[:HAS_COCOS]->(:COCOS)
            MATCH (c:COCOS {id: s.cocos})
            MERGE (s)-[:HAS_COCOS]->(c)
            RETURN count(s) AS n
            """
        )
        edges_created = edges[0]["n"] if edges else 0
    finally:
        if own:
            client.close()

    if scalars_set or edges_created:
        logger.info(
            "reconcile_standard_name_cocos_links: convention=COCOS %s, set %d cocos scalar(s), "
            "created %d HAS_COCOS edge(s)",
            convention,
            scalars_set,
            edges_created,
        )
    return {
        "convention": convention,
        "scalars_set": scalars_set,
        "edges_created": edges_created,
    }


def reconcile_grammar_segments() -> dict[str, int]:
    """Realign each live name's grammar segment columns with its canonical id.

    The bare-name segments (``position``, ``component``, ``subject``, …) are a
    deterministic function of the canonical name id via the ISN parser. A name
    written by an out-of-grammar path — e.g. the since-removed bulk catalog
    import — can carry stale segments that disagree with its own id: an
    ``…_at_pedestal_top`` name storing ``position='pedestal'`` is the observed
    case. Such a name no longer recomposes to itself, and a re-composition from
    its DD leaf would mint a divergent ``_at_pedestal`` near-duplicate.

    This idempotent sweep re-parses every live name through the authoritative
    ISN parser and realigns any drifted segment column to the parse. Names the
    ISN grammar cannot parse are skipped (their segments are owned by the
    quarantine path, never wiped here). Safe to run every rotation.

    Returns ``{"names_realigned": n}``.
    """
    from imas_codex.standard_names.ledger import LIVE_NAME

    cols = _GRAMMAR_SEGMENT_COLUMNS
    select = ", ".join(f"sn.{c} AS {c}" for c in cols)
    set_clause = ", ".join(f"sn.{c} = b.{c}" for c in cols)
    batch: list[dict[str, Any]] = []
    with GraphClient() as gc:
        rows = gc.query(
            f"MATCH (sn:StandardName) WHERE {LIVE_NAME} RETURN sn.id AS id, {select}"
        )
        for r in rows:
            parsed = _parse_grammar(r["id"])
            # A successful ISN parse always yields a physical_base; an all-None
            # parse means the model rejected the name — leave its stored
            # segments untouched rather than wiping them.
            if not parsed.get("physical_base"):
                continue
            if any(parsed.get(c) != r.get(c) for c in cols):
                batch.append({"id": r["id"], **{c: parsed.get(c) for c in cols}})
        if batch:
            gc.query(
                f"UNWIND $batch AS b MATCH (sn:StandardName {{id: b.id}}) SET {set_clause}",
                batch=batch,
            )
    if batch:
        logger.info(
            "reconcile_grammar_segments: realigned %d name(s) to their canonical id parse",
            len(batch),
        )
    return {"names_realigned": len(batch)}


def reconcile_error_siblings() -> dict[str, int]:
    """Detect and mark stale error-sibling StandardNames.

    An error-sibling StandardName is identified by
    ``model='deterministic:dd_error_modifier'``. It is orphaned when
    no parent StandardName exists that the sibling's uncertainty
    operator wraps. Detection: strip the ``upper_uncertainty_of_`` /
    ``lower_uncertainty_of_`` / ``uncertainty_index_of_`` prefix and
    check whether the resulting parent name still has a StandardName
    node in the graph.

    Orphans are quarantined (``validation_status='quarantined'`` with
    ``quarantine_reason``) to prevent them from appearing in downstream
    phases and exports.

    Returns dict with counts: {stale_marked}.
    """
    from imas_codex.standard_names.error_siblings import ERROR_SUFFIX_TO_OPERATOR

    # Build the list of operator prefixes to strip
    prefixes = [f"{op}_of_" for op in ERROR_SUFFIX_TO_OPERATOR.values()]

    with GraphClient() as gc:
        # Find all error-sibling names
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.model = 'deterministic:dd_error_modifier'
                  AND coalesce(sn.validation_status, '') <> 'quarantined'
                RETURN sn.id AS id
                """
            )
            or []
        )

        orphan_ids: list[str] = []
        for row in rows:
            sn_id = row["id"]
            parent_name = None
            for prefix in prefixes:
                if sn_id.startswith(prefix):
                    parent_name = sn_id[len(prefix) :]
                    break

            if parent_name is None:
                # Can't determine parent — skip
                continue

            # Check if parent StandardName exists
            parent_check = list(
                gc.query(
                    "MATCH (p:StandardName {id: $pid}) RETURN p.id LIMIT 1",
                    pid=parent_name,
                )
                or []
            )
            if not parent_check:
                orphan_ids.append(sn_id)

        stale_count = 0
        if orphan_ids:
            gc.query(
                """
                UNWIND $ids AS sid
                MATCH (sn:StandardName {id: sid})
                SET sn.validation_status = 'quarantined',
                    sn.quarantine_reason = 'orphaned error sibling (parent name deleted)'
                """,
                ids=orphan_ids,
            )
            stale_count = len(orphan_ids)
            logger.info(
                "Reconciled %d orphaned error-sibling StandardNames → quarantined",
                stale_count,
            )

    return {"stale_marked": stale_count}


def get_standard_name_source_stats(
    source_type: str | None = None,
) -> dict[str, int]:
    """Get StandardNameSource status counts.

    Returns dict mapping status → count. Optionally filtered by source_type.
    """
    with GraphClient() as gc:
        where = ""
        params: dict = {}
        if source_type:
            where = "WHERE sns.source_type = $source_type"
            params["source_type"] = source_type
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource) {where}
            RETURN sns.status AS status, count(sns) AS count
            """,
            **params,
        )
        return {r["status"]: r["count"] for r in result}


def write_run_provenance(
    name_ids: list[str],
    run_id: str,
) -> int:
    """Stamp run provenance fields on StandardName nodes.

    Sets ``last_run_id`` and ``last_run_at`` on every name touched by
    the current ``sn run`` invocation.

    Args:
        name_ids: StandardName ids to stamp.
        run_id: UUID string identifying this ``sn run`` invocation.

    Returns:
        Number of nodes updated.
    """
    if not name_ids:
        return 0

    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            SET sn.last_run_id = $rid,
                sn.last_run_at = datetime($ts)
            RETURN count(sn) AS n
            """,
            ids=name_ids,
            rid=run_id,
            ts=now,
        )
        return result[0]["n"] if result else 0


# =============================================================================
# Review comment export (Phase F — anti-pattern feedback loop)
# =============================================================================


def export_review_comments(
    output_path: str | Path,
    *,
    domain: str | None = None,
) -> int:
    """Dump StandardNameReview node comment data to a JSONL file before ``sn clear``.

    Queries all ``StandardNameReview`` nodes (optionally filtered by the parent
    ``StandardName.physics_domain``) and writes one JSON record per
    line to *output_path*.  Each record contains:

    * ``name`` — ``StandardName.id``
    * ``domain`` — ``StandardName.physics_domain``
    * ``reviewer_model`` — the model that produced the review
    * ``score`` — numeric score (0–1)
    * ``comments_per_dim`` — parsed dict of per-dimension comments
    * ``comments`` — full free-text comment string
    * ``review_axis`` — "names" or "docs"
    * ``generated_at`` — ``StandardName.generated_at`` ISO string
    * ``reviewed_at`` — ``StandardNameReview.reviewed_at`` ISO string

    Parameters
    ----------
    output_path:
        Destination file path.  Parent directories are created if absent.
    domain:
        When provided, restrict to reviews on names with this
        ``physics_domain``.

    Returns
    -------
    Number of StandardNameReview records written (0 if none found).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {}
    where_clauses: list[str] = []
    if domain:
        where_clauses.append("sn.physics_domain = $domain")
        params["domain"] = domain

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cypher = f"""
        MATCH (sn:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview)
        {where}
        RETURN sn.id AS name,
               sn.physics_domain AS domain,
               r.reviewer_model AS reviewer_model,
               r.score AS score,
               r.comments_per_dim_json AS comments_per_dim_json,
               r.comments AS comments,
               r.review_axis AS review_axis,
               sn.generated_at AS generated_at,
               r.reviewed_at AS reviewed_at
        ORDER BY sn.id, r.reviewed_at
    """

    with GraphClient() as gc:
        rows = gc.query(cypher, **params)

    if not rows:
        logger.info(
            "export_review_comments: no StandardNameReview nodes found (domain=%s)",
            domain,
        )
        return 0

    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            # Parse comments_per_dim_json if it's a string
            cpd_raw = row.get("comments_per_dim_json")
            if isinstance(cpd_raw, str):
                try:
                    cpd = json.loads(cpd_raw)
                except (json.JSONDecodeError, ValueError):
                    cpd = {}
            elif isinstance(cpd_raw, dict):
                cpd = cpd_raw
            else:
                cpd = {}

            record: dict[str, Any] = {
                "name": row.get("name"),
                "domain": row.get("domain"),
                "reviewer_model": row.get("reviewer_model"),
                "score": row.get("score"),
                "comments_per_dim": cpd,
                "comments": row.get("comments"),
                "review_axis": row.get("review_axis"),
                "generated_at": str(row["generated_at"])
                if row.get("generated_at")
                else None,
                "reviewed_at": str(row["reviewed_at"])
                if row.get("reviewed_at")
                else None,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(
        "export_review_comments: wrote %d records to %s (domain=%s)",
        count,
        output_path,
        domain,
    )
    return count


# =============================================================================
# Graph-backed LLM cost tracking API  (Phase 2)
# =============================================================================

# Phase → StandardName.llm_cost_<suffix> field mapping.
_PHASE_TO_SN_COST_FIELD: dict[str, str] = {
    "generate_name": "llm_cost_generate_name",
    "review_name": "llm_cost_review_name",
    "refine_name": "llm_cost_refine_name",
    "generate_docs": "llm_cost_generate_docs",
    "review_docs": "llm_cost_review_docs",
    "refine_docs": "llm_cost_refine_docs",
}


def create_sn_run_open(
    run_id: str,
    *,
    started_at: Any,
    cost_limit: float,
    min_score: float | None = None,
) -> None:
    """Pre-create an ``SNRun`` node with ``status='started'``.

    Called at the START of ``run_sn_pools`` so that ``(LLMCost)-[:FOR_RUN]->
    (SNRun)`` edges have a target from the first LLM call onward.

    Uses MERGE so repeated calls (e.g. after a retry) are safe.
    Sets ``created_at`` explicitly so the timestamp is never NULL.
    """
    from imas_codex.graph.models import SNRun

    run = SNRun(
        id=run_id,
        started_at=started_at,
        cost_limit=round(cost_limit, 6),
        cost_spent=0.0,
        cost_total=0.0,
        events_total=0,
        min_score=min_score,
        status="started",
        cost_is_exact=True,
    )
    props = run.model_dump(mode="json")
    try:
        with GraphClient() as gc:
            gc.create_nodes("SNRun", [props])
            # Set created_at via Cypher datetime() for Neo4j-native timestamp
            gc.query(
                "MATCH (rr:SNRun {id: $run_id}) SET rr.created_at = datetime()",
                run_id=run_id,
            )
    except Exception as exc:
        logger.error("Failed to pre-create SNRun %s: %s", run_id, exc)
        raise


def bump_sn_run_counter(
    run_id: str | None,
    counter: str,
    delta: int = 1,
) -> None:
    """Atomically increment an SNRun counter (best-effort, non-blocking).

    Uses ``coalesce(run.X, 0) + $delta`` so concurrent bumps accumulate
    correctly — Neo4j's single-node write lock serialises the read-modify-
    write within the same transaction, preventing lost updates.

    Parameters
    ----------
    run_id:
        SNRun node id.  If ``None``, the bump is silently skipped (allows
        callers outside a run context to work unchanged).
    counter:
        Property name on the SNRun node to increment (e.g.
        ``"names_composed"``, ``"names_reviewed"``).
    delta:
        Amount to add (default 1).
    """
    if not run_id or delta <= 0:
        return

    _ALLOWED = {
        "names_composed",
        "names_reviewed",
        "names_regenerated",
        "names_enriched",
    }
    if counter not in _ALLOWED:
        logger.warning("bump_sn_run_counter: unknown counter %r — skipping", counter)
        return

    try:
        with GraphClient() as gc:
            gc.query(
                f"MATCH (rr:SNRun {{id: $run_id}}) "
                f"SET rr.{counter} = coalesce(rr.{counter}, 0) + $delta",
                run_id=run_id,
                delta=delta,
            )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug(
            "bump_sn_run_counter(%s, %s, %d) failed: %s",
            run_id,
            counter,
            delta,
            exc,
        )


def update_sn_run_progress(
    run_id: str,
    *,
    cost_spent: float,
    cost_total: float | None = None,
    events_total: int | None = None,
) -> None:
    """Periodic in-progress sync of ``SNRun`` telemetry fields.

    Mirrors the running spend total onto the graph node so ``imas-codex sn
    status`` reflects real progress even when a run is interrupted or
    crashes before :func:`finalize_sn_run` runs.  Best-effort: any graph
    error is logged at DEBUG and swallowed so we never poison the loop.

    When *cost_total* and *events_total* are provided (from the BudgetManager
    batch counter), they are written as absolute values alongside
    ``last_heartbeat = datetime()``.
    """
    set_parts = ["rr.cost_spent = $cost_spent", "rr.last_heartbeat = datetime()"]
    params: dict[str, object] = {
        "run_id": run_id,
        "cost_spent": round(float(cost_spent), 6),
    }
    if cost_total is not None:
        set_parts.append("rr.cost_total = $cost_total")
        params["cost_total"] = round(float(cost_total), 6)
    if events_total is not None:
        set_parts.append("rr.events_total = $events_total")
        params["events_total"] = int(events_total)

    try:
        with GraphClient() as gc:
            gc.query(
                "MATCH (rr:SNRun {id: $run_id}) SET " + ", ".join(set_parts),
                **params,
            )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("update_sn_run_progress(%s) failed: %s", run_id, exc)


def finalize_sn_run(
    run_id: str,
    *,
    status: str,
    cost_spent: float,
    cost_is_exact: bool = True,
    stopped_at: Any,
    elapsed_s: float | None = None,
    **summary_fields: Any,
) -> None:
    """Update an existing ``SNRun`` node at run end.

    Uses ``MATCH + SET`` (not CREATE) — the node must already exist
    (created by :func:`create_sn_run_open`).

    Aggregates ``cost_total`` and ``events_total`` from LLMCost children
    so the SNRun is an authoritative mirror of the LLMCost ledger.
    ``ended_at`` is set to the current graph timestamp.
    ``elapsed_s`` records wall-clock seconds between *started_at* and
    *stopped_at*; callers should pass the pre-computed value so finalization
    never needs to re-read the start timestamp from the graph.

    ``summary_fields`` may contain any other ``SNRun`` property such as
    ``domains_touched``, ``stop_reason``, ``pipeline_hash``,
    ``names_composed``, ``names_enriched``, etc.
    """
    # First, aggregate cost_total and events_total from LLMCost children
    agg_cost: float = 0.0
    agg_events: int = 0
    try:
        with GraphClient() as gc:
            agg_rows = list(
                gc.query(
                    "MATCH (c:LLMCost {run_id: $rid}) "
                    "RETURN sum(c.llm_cost) AS cost, count(c) AS events",
                    rid=run_id,
                )
            )
            if agg_rows:
                agg_cost = float(agg_rows[0].get("cost") or 0.0)
                agg_events = int(agg_rows[0].get("events") or 0)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "finalize_sn_run: LLMCost aggregation failed for %s: %s", run_id, exc
        )

    set_clauses = [
        "rr.status = $status",
        "rr.cost_spent = $cost_spent",
        "rr.cost_is_exact = $cost_is_exact",
        "rr.stopped_at = datetime($stopped_at)",
        "rr.ended_at = datetime()",
        "rr.cost_total = $cost_total",
        "rr.events_total = $events_total",
    ]
    params: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "cost_spent": round(cost_spent, 6),
        "cost_is_exact": cost_is_exact,
        "stopped_at": stopped_at
        if isinstance(stopped_at, str)
        else stopped_at.isoformat(),
        "cost_total": round(agg_cost, 6),
        "events_total": agg_events,
    }

    if elapsed_s is not None:
        set_clauses.append("rr.elapsed_s = $elapsed_s")
        params["elapsed_s"] = round(elapsed_s, 3)

    for key, value in summary_fields.items():
        set_clauses.append(f"rr.{key} = ${key}")
        params[key] = value

    cypher = (
        "MATCH (rr:SNRun {id: $run_id}) "
        "SET " + ", ".join(set_clauses) + " "
        "RETURN rr.id AS id"
    )
    try:
        with GraphClient() as gc:
            result = gc.query(cypher, **params)
            if not result:
                logger.warning("finalize_sn_run: no SNRun found with id=%s", run_id)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Failed to finalize SNRun %s: %s", run_id, exc)


def mark_orphaned_standard_name_runs_stale(
    *,
    current_run_id: str | None = None,
    max_age_hours: float = 6.0,
) -> int:
    """Finalize SNRun rows that never reached a terminal status.

    ``finalize_sn_run`` runs inside ``run_sn_pools``' ``finally`` block, so a
    clean/exception/SIGINT exit always closes the run. A run left at the
    open ``status='started'``/``'running'`` status therefore means the process
    died before Python could finalize (hard kill, OOM, node loss) — the row is
    an orphan that no live process will ever close.

    This sweep marks such rows ``status='stale'`` with
    ``stop_reason='orphaned_no_finalize'`` so ``sn status`` and provenance
    stop reporting them as in-flight. A run is only considered orphaned when
    its most recent liveness signal (``last_heartbeat``, else ``created_at``)
    is older than *max_age_hours* — an active run heartbeats via
    :func:`update_sn_run_progress`, so a generous threshold never touches a
    genuinely-running peer. The current run (``current_run_id``) is always
    excluded.

    Idempotent: a second call matches nothing (the rows are now ``'stale'``).
    Returns the number of runs marked stale.
    """
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (rr:SNRun)
            WHERE rr.status IN ['started', 'running']
              AND ($current_run_id IS NULL OR rr.id <> $current_run_id)
              AND coalesce(rr.last_heartbeat, rr.created_at, rr.stopped_at)
                  < datetime() - duration({hours: $max_age})
            SET rr.status = 'stale',
                rr.stop_reason = 'orphaned_no_finalize',
                rr.cost_is_exact = false,
                rr.ended_at = datetime()
            RETURN rr.id AS id
            """,
            current_run_id=current_run_id,
            max_age=max_age_hours,
        )
    marked = len(rows or [])
    if marked:
        logger.info(
            "mark_orphaned_standard_name_runs_stale: marked %d orphaned SNRun(s) stale "
            "(no finalize, older than %.1fh)",
            marked,
            max_age_hours,
        )
    return marked


def normalize_stored_standard_name_kinds() -> int:
    """Collapse any stored ``StandardName.kind`` outside the ISN Kind enum.

    The ISN catalog ``Kind`` enum (imas_standard_names) is the authority for
    the closed kind vocabulary. Legacy graph data written before the local
    vocabulary was unified with ISN may still carry a retired kind
    (``eigenfunction`` / ``spectrum``); this sweep rewrites each offending node
    to the value the ISN discriminator accepts via :func:`to_isn_kind` — the
    single source of truth for the collapse (retired kinds map to ``scalar``).

    Idempotent: a second call matches nothing (every node now holds an
    in-enum kind). Returns the number of nodes rewritten.
    """
    from imas_standard_names.models import Kind

    from imas_codex.standard_names.kind_derivation import to_isn_kind

    allowed = [k.value for k in Kind]
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.kind IS NOT NULL AND NOT sn.kind IN $allowed
            RETURN sn.id AS id, sn.kind AS kind
            """,
            allowed=allowed,
        )
        updates = [
            {"id": r["id"], "kind": to_isn_kind(r["kind"])} for r in (rows or [])
        ]
        if updates:
            gc.query(
                """
                UNWIND $updates AS u
                MATCH (sn:StandardName {id: u.id})
                SET sn.kind = u.kind
                """,
                updates=updates,
            )
    rewritten = len(updates)
    if rewritten:
        logger.info(
            "normalize_stored_standard_name_kinds: collapsed %d node(s) with a "
            "retired/off-enum kind to the ISN-canonical value",
            rewritten,
        )
    return rewritten


def backfill_sn_run_telemetry() -> list[dict[str, Any]]:
    """One-shot backfill of cost_total/events_total on existing SNRun nodes.

    Idempotent: only touches nodes where cost_total IS NULL.
    Aggregates from LLMCost children and writes the result.
    Returns a list of dicts with id, cost, events for each patched node.
    """
    cypher = """
    MATCH (r:SNRun)
    WHERE r.cost_total IS NULL
    OPTIONAL MATCH (c:LLMCost {run_id: r.id})
    WITH r, sum(c.llm_cost) AS cost_sum, count(c) AS event_count
    SET r.cost_total = coalesce(r.cost_total, cost_sum, 0.0),
        r.events_total = coalesce(r.events_total, event_count, 0)
    RETURN r.id AS id, r.cost_total AS cost, r.events_total AS events
    """
    with GraphClient() as gc:
        return list(gc.query(cypher))


# ---------------------------------------------------------------------------
# LLMCost pool normalisation (W4c)
# ---------------------------------------------------------------------------
# Maps the raw ``phase`` labels used by callers to six canonical pool names.
_PHASE_TO_POOL: dict[str, str] = {
    "generate": "compose",
    "generate_name": "compose",
    "compose": "compose",
    "regen": "refine_name",
    "refine_name": "refine_name",
    "refine_docs": "refine_docs",
    "validate": "validate",
    "validate_name": "validate",
    "review_names": "review",
    "review_docs": "review",
    "review": "review",
    "enrich": "enrich",
    "enrich_links": "enrich",
}


def _normalize_pool(phase: str) -> str:
    """Map a raw phase string to a canonical pool name."""
    return _PHASE_TO_POOL.get(phase, phase)


@retry_on_deadlock()
def record_llm_cost(
    *,
    run_id: str,
    phase: str,
    cycle: str | None = None,
    sn_ids: list[str] | None = None,
    model: str,
    cost: float,
    tokens_in: int,
    tokens_out: int,
    tokens_cached_read: int = 0,
    tokens_cached_write: int = 0,
    service: str = "standard-names",
    batch_id: str | None = None,
    overspend: float = 0.0,
    llm_at: Any | None = None,
) -> str:
    """Write an atomic ``LLMCost`` node and ``FOR_RUN`` edge.

    **Idempotency contract**: ``id`` is a deterministic UUID-5 over
    ``(run_id, phase, batch_id, model, llm_at_iso, cost, tokens_in,
    tokens_out)``.  The node is written with ``CREATE`` (not MERGE).
    If a uniqueness constraint violation fires (duplicate id), the
    exception is swallowed — the previous write already succeeded.

    Returns:
        The deterministic ``id`` string.
    """
    from datetime import UTC, datetime

    if llm_at is None:
        llm_at = datetime.now(UTC)
    llm_at_iso = llm_at.isoformat() if not isinstance(llm_at, str) else llm_at

    # Deterministic id — uuid5 over immutable call identity
    id_seed = (
        f"{run_id}|{phase}|{batch_id}|{model}"
        f"|{llm_at_iso}|{cost}|{tokens_in}|{tokens_out}"
    )
    spend_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_seed))

    sn_ids_clean = list(sn_ids) if sn_ids else []

    # Guard: reject obvious test fixture leaks. Tests that monkeypatch
    # ``get_model`` to return ``"test-model"`` have leaked ~50 orphan
    # LLMCost nodes into the production graph. Refuse them at the write
    # boundary unless the fixture author EXPLICITLY opts in via
    # ``IMAS_CODEX_ALLOW_TEST_MODEL=1`` (and uses a test-scoped graph
    # profile, never `codex`).
    #
    # NOTE: ``PYTEST_CURRENT_TEST`` does NOT bypass this guard. Tests
    # should mock ``record_llm_cost`` / ``GraphClient.query`` rather
    # than write fixture rows into a live graph.
    if model == "test-model" and os.environ.get("IMAS_CODEX_ALLOW_TEST_MODEL") != "1":
        logger.warning(
            "record_llm_cost: refusing model='test-model' (run=%s phase=%s) — "
            "fixture leak guard. Mock the write or set "
            "IMAS_CODEX_ALLOW_TEST_MODEL=1 against a test-scoped graph "
            "profile (never `codex`).",
            run_id,
            phase,
        )
        return spend_id

    cypher = """
        CREATE (c:LLMCost {
            id: $id,
            run_id: $run_id,
            phase: $phase,
            pool: $pool,
            event_type: $event_type,
            cycle: $cycle,
            sn_ids: $sn_ids,
            batch_id: $batch_id,
            overspend: $overspend,
            llm_model: $llm_model,
            llm_cost: $llm_cost,
            llm_tokens_in: $llm_tokens_in,
            llm_tokens_out: $llm_tokens_out,
            llm_tokens_cached_read: $llm_tokens_cached_read,
            llm_tokens_cached_write: $llm_tokens_cached_write,
            llm_service: $llm_service,
            llm_at: datetime($llm_at),
            for_run: $run_id
        })
        WITH c
        MATCH (rr:SNRun {id: $run_id})
        MERGE (c)-[:FOR_RUN]->(rr)
    """
    params = {
        "id": spend_id,
        "run_id": run_id,
        "phase": phase,
        "pool": _normalize_pool(phase),
        "event_type": phase,
        "cycle": cycle,
        "sn_ids": sn_ids_clean,
        "batch_id": batch_id,
        "overspend": round(overspend, 6),
        "llm_model": model,
        "llm_cost": round(cost, 6),
        "llm_tokens_in": tokens_in,
        "llm_tokens_out": tokens_out,
        "llm_tokens_cached_read": tokens_cached_read,
        "llm_tokens_cached_write": tokens_cached_write,
        "llm_service": service,
        "llm_at": llm_at_iso,
    }

    try:
        with GraphClient() as gc:
            gc.query(cypher, **params)
    except Exception as exc:
        # Swallow constraint violation (idempotent duplicate) —
        # the original write already succeeded.
        from neo4j.exceptions import ConstraintError

        if isinstance(exc, ConstraintError):
            logger.debug(
                "record_llm_cost: duplicate id=%s (idempotent), skipping",
                spend_id,
            )
        else:
            raise

    return spend_id


def aggregate_spend_for_run(run_id: str) -> float:
    """Return total LLM cost for a run by summing ``LLMCost`` nodes."""
    with GraphClient() as gc:
        result = gc.query(
            "MATCH (c:LLMCost {run_id: $run_id}) "
            "RETURN coalesce(sum(c.llm_cost), 0.0) AS total",
            run_id=run_id,
        )
        return float(result[0]["total"]) if result else 0.0


def aggregate_spend_per_phase(run_id: str) -> dict[str, float]:
    """Return ``{phase: total_cost}`` for a run."""
    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (c:LLMCost {run_id: $run_id}) "
            "RETURN c.phase AS phase, sum(c.llm_cost) AS total "
            "ORDER BY phase",
            run_id=run_id,
        )
        return {r["phase"]: float(r["total"]) for r in rows}


def aggregate_spend_per_name(run_id: str) -> dict[str, float]:
    """Return ``{sn_id: apportioned_cost}`` for a run.

    Per-name cost share is ``llm_cost / size(sn_ids)`` — each name
    in the ``sn_ids`` list gets an equal share.  Rows with an empty
    ``sn_ids`` list are skipped (e.g. L7 audit calls with no names).
    """
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (c:LLMCost {run_id: $run_id})
            WHERE size(c.sn_ids) > 0
            UNWIND c.sn_ids AS sn_id
            RETURN sn_id, sum(c.llm_cost / size(c.sn_ids)) AS apportioned
            """,
            run_id=run_id,
        )
        return {r["sn_id"]: float(r["apportioned"]) for r in rows}


def update_sn_per_phase_costs(run_id: str) -> int:
    """Push aggregated per-name costs into ``StandardName.llm_cost_*`` fields.

    For each ``LLMCost`` row in the run, apportions ``llm_cost / size(sn_ids)``
    to each name, grouped by phase.  Then writes the per-phase totals and the
    overall ``llm_cost`` onto the ``StandardName`` node.

    Returns:
        Number of ``StandardName`` nodes updated.
    """
    # Build per-(name, phase) apportionment
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (c:LLMCost {run_id: $run_id})
            WHERE size(c.sn_ids) > 0
            UNWIND c.sn_ids AS sn_id
            RETURN sn_id, c.phase AS phase,
                   sum(c.llm_cost / size(c.sn_ids)) AS apportioned
            """,
            run_id=run_id,
        )

    if not rows:
        return 0

    # Aggregate: {sn_id: {field: cost, ...}}
    per_name: dict[str, dict[str, float]] = {}
    for r in rows:
        sn_id = r["sn_id"]
        phase = r["phase"]
        cost = float(r["apportioned"])
        if sn_id not in per_name:
            per_name[sn_id] = {}
        field = _PHASE_TO_SN_COST_FIELD.get(phase)
        if field:
            per_name[sn_id][field] = per_name[sn_id].get(field, 0.0) + cost

    # Write back — batch all names in a single Cypher per phase-field
    updated_ids: set[str] = set()

    with GraphClient() as gc:
        for sn_id, fields in per_name.items():
            total_cost = sum(fields.values())
            set_parts = ["sn.llm_cost = $total"]
            params: dict[str, Any] = {
                "sn_id": sn_id,
                "total": round(total_cost, 6),
            }
            for field_name, field_cost in fields.items():
                set_parts.append(f"sn.{field_name} = ${field_name}")
                params[field_name] = round(field_cost, 6)

            result = gc.query(
                "MATCH (sn:StandardName {id: $sn_id}) "
                "SET " + ", ".join(set_parts) + " "
                "RETURN sn.id AS id",
                **params,
            )
            if result:
                updated_ids.add(sn_id)

    return len(updated_ids)


# =============================================================================
# Seed-and-expand claims  (Phase 8 worker pools)
#
# Each function follows the H4 pattern from plan.md Phase 8:
#   1. Seed: pick one random eligible row, SET claimed_at + claim_token
#   2. Expand: claim up to batch_size−1 more with same (cluster_id, unit);
#      fallback: (physics_domain, unit) when cluster_id is absent
#   3. Read-back: query by claim_token — only our token's rows returned
#
# Compose targets StandardNameSource; the other four target StandardName.
# =============================================================================

DEFAULT_POOL_BATCH_SIZE = 25

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
_MAX_COMPOSE_CLAIM_ATTEMPTS = 5


def _claim_sn_atomic(
    *,
    eligibility_where: str,
    query_params: dict[str, Any],
    batch_size: int,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    extra_return_fields: str = "",
    stage_field: str | None = None,
    to_stage: str | None = None,
    domain: str | None = None,
    scope_run_id: str | None = None,
    edits_only: bool = False,
    seed_extra_where: str = "",
    seed_with_extras: str = "",
    seed_order_by: str = "rand()",
) -> list[dict[str, Any]]:
    """Single-transaction stage-aware claim for StandardName nodes.

    All steps — seed, expand, and read-back — execute inside **one** Neo4j
    transaction.  If any step fails, the entire transaction rolls back and
    no partial claim state leaks into the graph.

    Callers should wrap this with ``@retry_on_deadlock()`` so that a
    ``TransientError`` (deadlock) causes the whole function — including
    token generation — to retry cleanly.

    Parameters
    ----------
    eligibility_where:
        Cypher WHERE fragment applied to ``sn`` (must NOT include ``WHERE``
        keyword itself).  May reference parameters in *query_params* via
        ``$name`` syntax.
    query_params:
        Extra Cypher parameters consumed by *eligibility_where*.
    batch_size:
        Maximum batch size (seed + expand).
    timeout_seconds:
        Stale-claim recovery window.
    extra_return_fields:
        Additional ``RETURN`` columns appended after the common set.
        Must start with a comma if non-empty, e.g.
        ``", sn.enriched_at AS enriched_at"``.
    stage_field:
        Optional stage property to transition at claim time
        (``'name_stage'`` or ``'docs_stage'``).  When *None* (default),
        no stage transition occurs — only ``claim_token`` and
        ``claimed_at`` are written.
    to_stage:
        Target stage value for the transition (e.g. ``'refining'``).
        Ignored when *stage_field* is *None*.
    domain:
        Optional physics domain to scope the claim.  When set, only
        StandardName nodes with ``physics_domain = $domain`` are
        considered eligible.  Used by ``sn run --physics-domain`` to
        prevent concurrent domain-specific runs from competing for
        each other's items.
    edits_only:
        When ``True``, restrict eligibility to StandardName nodes carrying
        ``edit_status = 'open'`` — the pending successors of a ``sn edit``.
        Composed exactly like *scope_run_id*'s ``scope_where`` and ANDed
        alongside it; the common use is *edits_only* alone (an edit is
        already focused, so no ``--focus``/DD-path is required).
    seed_extra_where:
        Optional additional Cypher WHERE conditions injected into the seed
        step after *eligibility_where* and *domain_where*.  Must begin
        with ``AND`` if non-empty.  Used to implement ordering-specific
        eligibility filters (e.g. parent-first escape hatches).
    seed_with_extras:
        Optional extra WITH columns computed before the ORDER BY in the
        seed step.  Must begin with a comma if non-empty, e.g.
        ``", CASE WHEN ... END AS priority"``.  Used alongside
        *seed_order_by* to express multi-column ordering.
    seed_order_by:
        ORDER BY expression for the seed step.  Defaults to ``rand()``
        (uniform random selection).  Override to implement priority-based
        claim ordering, e.g. ``"priority ASC, rand()"``.

    Returns
    -------
    list[dict]
        Claimed items as plain dicts.  Empty list when no eligible seed
        exists.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"
    params: dict[str, Any] = {**query_params, "token": token, "cutoff": cutoff}

    # Build optional SET clause for atomic stage transition.
    stage_set = ""
    if stage_field and to_stage:
        stage_set = f", sn.{stage_field} = $to_stage"
        params["to_stage"] = to_stage

    # Optional physics-domain scope (scalar comparison post-refactor).
    domain_where = ""
    if domain:
        domain_where = " AND sn.physics_domain = $domain"
        params["domain"] = domain

    # Optional run-id scope for --focus mode, plus an edit-scope predicate for
    # --edits mode. Both are ANDed into the same fragment and injected at every
    # seed/expand step, so they combine cleanly (edits_only is the common case).
    scope_where = ""
    if scope_run_id:
        scope_where += " AND sn.run_id = $scope_run_id"
        params["scope_run_id"] = scope_run_id
    if edits_only:
        scope_where += " AND coalesce(sn.edit_status, '') = 'open'"

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                # ── Step 1: Seed ─────────────────────────────────
                seed_result = list(
                    tx.run(
                        f"""
                        MATCH (sn:StandardName)
                        WHERE {eligibility_where}
                          AND (sn.claimed_at IS NULL
                               OR sn.claimed_at < datetime()
                                    - duration($cutoff))
                          {domain_where}
                          {scope_where}
                          {seed_extra_where}
                        WITH sn{seed_with_extras}
                        ORDER BY {seed_order_by} LIMIT 1
                        SET sn.claimed_at = datetime(),
                            sn.claim_token = $token,
                            sn.claim_seq = coalesce(sn.claim_seq, 0) + 1
                            {stage_set}
                        WITH sn
                        OPTIONAL MATCH (sn)-[:IN_CLUSTER]
                            ->(c:IMASSemanticCluster)
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        RETURN c.id AS _cluster_id, u.id AS _unit,
                               sn.physics_domain AS _physics_domain
                        """,
                        **params,
                    )
                )

                if not seed_result:
                    tx.close()
                    return []

                seed = dict(seed_result[0])
                cluster_id = seed.get("_cluster_id")
                unit = seed.get("_unit")
                # Defensive scalar coerce: legacy nodes may still hold a
                # list-valued physics_domain until ``sn clear`` has run.
                physics_domain = _scalar_domain(seed.get("_physics_domain"))
                expand_limit = batch_size - 1

                # ── Step 2: Expand ───────────────────────────────
                if expand_limit > 0:
                    expand_params: dict[str, Any] = {
                        **params,
                        "expand_limit": expand_limit,
                    }
                    if cluster_id is not None and unit is not None:
                        expand_params.update(cluster_id=cluster_id, unit=unit)
                        tx.run(
                            f"""
                            MATCH (sn:StandardName)
                            WHERE {eligibility_where}
                              AND sn.claimed_at IS NULL
                              {scope_where}
                            MATCH (sn)-[:IN_CLUSTER]
                                ->(:IMASSemanticCluster
                                    {{id: $cluster_id}})
                            MATCH (sn)-[:HAS_UNIT]
                                ->(:Unit {{id: $unit}})
                            WITH sn LIMIT $expand_limit
                            SET sn.claimed_at = datetime(),
                                sn.claim_token = $token,
                                sn.claim_seq = coalesce(sn.claim_seq, 0) + 1
                                {stage_set}
                            """,
                            **expand_params,
                        )
                    elif cluster_id is not None:
                        expand_params["cluster_id"] = cluster_id
                        tx.run(
                            f"""
                            MATCH (sn:StandardName)
                            WHERE {eligibility_where}
                              AND sn.claimed_at IS NULL
                              {scope_where}
                            MATCH (sn)-[:IN_CLUSTER]
                                ->(:IMASSemanticCluster
                                    {{id: $cluster_id}})
                            WITH sn LIMIT $expand_limit
                            SET sn.claimed_at = datetime(),
                                sn.claim_token = $token,
                                sn.claim_seq = coalesce(sn.claim_seq, 0) + 1
                                {stage_set}
                            """,
                            **expand_params,
                        )
                    elif physics_domain is not None and unit is not None:
                        expand_params.update(fallback_domain=physics_domain, unit=unit)
                        tx.run(
                            f"""
                            MATCH (sn:StandardName)
                            WHERE {eligibility_where}
                              AND sn.claimed_at IS NULL
                              {scope_where}
                              AND sn.physics_domain = $fallback_domain
                            MATCH (sn)-[:HAS_UNIT]
                                ->(:Unit {{id: $unit}})
                            WITH sn LIMIT $expand_limit
                            SET sn.claimed_at = datetime(),
                                sn.claim_token = $token,
                                sn.claim_seq = coalesce(sn.claim_seq, 0) + 1
                                {stage_set}
                            """,
                            **expand_params,
                        )
                    elif physics_domain is not None:
                        expand_params["fallback_domain"] = physics_domain
                        tx.run(
                            f"""
                            MATCH (sn:StandardName)
                            WHERE {eligibility_where}
                              AND sn.claimed_at IS NULL
                              {scope_where}
                              AND sn.physics_domain = $fallback_domain
                            WITH sn LIMIT $expand_limit
                            SET sn.claimed_at = datetime(),
                                sn.claim_token = $token,
                                sn.claim_seq = coalesce(sn.claim_seq, 0) + 1
                                {stage_set}
                            """,
                            **expand_params,
                        )
                    # else: no grouping key — seed-only batch

                # ── Step 3: Read-back by token ───────────────────
                results = list(
                    tx.run(
                        f"""
                        MATCH (sn:StandardName {{claim_token: $token}})
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        OPTIONAL MATCH (sn)-[:IN_CLUSTER]
                            ->(c:IMASSemanticCluster)
                        RETURN sn.id AS id,
                               sn.description AS description,
                               sn.documentation AS documentation,
                               sn.kind AS kind,
                               coalesce(u.id, sn.unit) AS unit,
                               c.id AS cluster_id,
                               sn.physics_domain AS physics_domain,
                               sn.validation_status
                                   AS validation_status,
                               sn.claim_token AS claim_token,
                               sn.claim_seq AS claim_seq
                               {extra_return_fields}
                        """,
                        token=token,
                    )
                )

                items = [dict(r) for r in results]
                tx.commit()

            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    logger.debug(
        "_claim_sn_atomic: claimed %d (token=%s)",
        len(items),
        token[:8],
    )
    return items


# -- generate_name (StandardNameSource) -------------------------------------


@retry_on_deadlock()
def claim_generate_name_batch(
    facility: str | None = None,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    edits_only: bool = False,
) -> list[dict[str, Any]]:
    """Claim StandardNameSource nodes (status='extracted') for name generation.

    Seed-and-expand: one random seed is claimed, then up to
    ``batch_size - 1`` additional sources sharing the same
    ``(cluster_id, unit)`` (via ``FROM_DD_PATH``→``IN_CLUSTER`` /
    ``HAS_UNIT``).  When no cluster exists the fallback key is
    ``(physics_domain, unit)``.

    All three steps (seed, expand, read-back) execute inside a **single**
    Neo4j transaction so that no partial claim state leaks on deadlock
    retry.

    Parameters
    ----------
    facility:
        Optional facility id to restrict claims to signal-backed sources
        from that facility.  ``None`` means all sources.
    batch_size:
        Maximum batch size including the seed.
    timeout_seconds:
        Stale-claim recovery window.
    domain:
        Optional physics domain to restrict claims to DD-backed sources
        whose linked IMASNode has ``physics_domain = domain``.  When set,
        the seed step only claims items from that domain, so concurrent
        ``--physics-domain`` runs do not compete for each other's seeds.

    Returns
    -------
    list[dict]
        Claimed sources as dicts with keys ``id``, ``source_id``,
        ``source_type``, ``batch_key``, ``description``.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"

    # Optional facility filter (only applies to signal sources).
    facility_where = ""
    facility_where_sns2 = ""
    extra_params: dict[str, Any] = {}
    if facility:
        facility_where = (
            "AND (sns.source_type = 'dd' OR EXISTS {"
            "  MATCH (sns)-[:FROM_SIGNAL]->(:FacilitySignal)"
            "    -[:AT_FACILITY]->(:Facility {id: $facility})"
            "})"
        )
        facility_where_sns2 = (
            "AND (sns2.source_type = 'dd' OR EXISTS {"
            "  MATCH (sns2)-[:FROM_SIGNAL]->(:FacilitySignal)"
            "    -[:AT_FACILITY]->(:Facility {id: $facility})"
            "})"
        )
        extra_params["facility"] = facility

    # Optional domain filter for DD sources — restricts seed to the target domain
    # so concurrent domain-specific runs don't compete for each other's items.
    domain_where = ""
    if domain:
        domain_where = (
            "AND (NOT (sns.source_type = 'dd') OR EXISTS {"
            "  MATCH (sns)-[:FROM_DD_PATH]->(n:IMASNode)"
            "  WHERE n.physics_domain = $domain"
            "})"
        )
        extra_params["domain"] = domain

    # Optional run-id scope for --focus mode (StandardNameSource).
    # edits_only additionally gates on the source's edit_status. Sources are
    # never edit successors (edit successors are already-composed StandardName
    # nodes), so this predicate makes an edit-scoped run claim NO new sources —
    # i.e. it correctly blocks fresh composition while draining pending edits.
    scope_sns_where = ""
    scope_sns2_where = ""
    if scope_run_id:
        scope_sns_where += " AND sns.run_id = $scope_run_id"
        scope_sns2_where += " AND sns2.run_id = $scope_run_id"
        extra_params["scope_run_id"] = scope_run_id
    if edits_only:
        scope_sns_where += " AND coalesce(sns.edit_status, '') = 'open'"
        scope_sns2_where += " AND coalesce(sns2.edit_status, '') = 'open'"

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                # ── Step 1: Seed (domain-uniform sampling) ───────
                # Two-level random selection: pick a random domain
                # first, then a random source within it.  This
                # prevents large domains (e.g. transport ~28% of
                # all sources) from dominating early batches.
                # When --domain is set, this collapses to a single-
                # domain rand() which is the desired behaviour.
                seed_result = list(
                    tx.run(
                        f"""
                        MATCH (sns:StandardNameSource)
                        WHERE sns.status = 'extracted'
                          AND coalesce(sns.attempt_count, 0) < $max_attempts
                          AND (sns.claimed_at IS NULL
                               OR sns.claimed_at < datetime()
                                    - duration($cutoff))
                          {facility_where}
                          {domain_where}
                          {scope_sns_where}
                        MATCH (sns)-[:FROM_DD_PATH]->(imas0:IMASNode)
                        WITH DISTINCT coalesce(imas0.physics_domain, 'general') AS pd
                        WITH pd ORDER BY rand() LIMIT 1
                        MATCH (sns2:StandardNameSource)
                              -[:FROM_DD_PATH]->(imas2:IMASNode)
                        WHERE sns2.status = 'extracted'
                          AND coalesce(sns2.attempt_count, 0) < $max_attempts
                          AND (sns2.claimed_at IS NULL
                               OR sns2.claimed_at < datetime()
                                    - duration($cutoff))
                          AND coalesce(imas2.physics_domain, 'general') = pd
                          {facility_where_sns2}
                          {scope_sns2_where}
                        WITH sns2 ORDER BY rand() LIMIT 1
                        SET sns2.claimed_at = datetime(),
                            sns2.claim_token = $token,
                            sns2.attempt_count = coalesce(sns2.attempt_count, 0) + 1
                        WITH sns2 AS sns
                        OPTIONAL MATCH (sns)-[:FROM_DD_PATH]
                            ->(imas:IMASNode)
                        OPTIONAL MATCH (imas)-[:IN_CLUSTER]
                            ->(c:IMASSemanticCluster)
                        OPTIONAL MATCH (imas)-[:HAS_UNIT]->(u:Unit)
                        OPTIONAL MATCH (sns)-[:FROM_SIGNAL]
                            ->(sig:FacilitySignal)
                        RETURN c.id AS _cluster_id,
                               CASE WHEN u IS NOT NULL THEN u.id
                                    WHEN sig IS NOT NULL
                                    THEN sig.unit
                                    ELSE null END AS _unit,
                               CASE WHEN imas IS NOT NULL
                                    THEN imas.physics_domain
                                    WHEN sig IS NOT NULL
                                    THEN sig.physics_domain
                                    ELSE null END
                                        AS _physics_domain,
                               sns.batch_key AS _batch_key
                        """,
                        token=token,
                        cutoff=cutoff,
                        max_attempts=_MAX_COMPOSE_CLAIM_ATTEMPTS,
                        **extra_params,
                    )
                )

                if not seed_result:
                    tx.close()
                    return []

                seed = dict(seed_result[0])
                cluster_id = seed.get("_cluster_id")
                unit = seed.get("_unit")
                physics_domain = seed.get("_physics_domain")
                batch_key = seed.get("_batch_key")
                expand_limit = batch_size - 1

                # ── Step 2: Expand ───────────────────────────────
                if expand_limit > 0:
                    expanded = False
                    if cluster_id is not None and unit is not None:
                        tx.run(
                            f"""
                            MATCH (sns:StandardNameSource)
                            WHERE sns.status = 'extracted'
                              AND coalesce(sns.attempt_count, 0) < $max_attempts
                              AND sns.claimed_at IS NULL
                              {facility_where}
                              {scope_sns_where}
                            MATCH (sns)-[:FROM_DD_PATH]
                                ->(imas:IMASNode)
                            MATCH (imas)-[:IN_CLUSTER]
                                ->(:IMASSemanticCluster
                                    {{id: $cluster_id}})
                            MATCH (imas)-[:HAS_UNIT]
                                ->(:Unit {{id: $unit}})
                            WITH sns LIMIT $expand_limit
                            SET sns.claimed_at = datetime(),
                                sns.claim_token = $token,
                                sns.attempt_count = coalesce(sns.attempt_count, 0) + 1
                            """,
                            token=token,
                            cluster_id=cluster_id,
                            unit=unit,
                            expand_limit=expand_limit,
                            max_attempts=_MAX_COMPOSE_CLAIM_ATTEMPTS,
                            **extra_params,
                        )
                        expanded = True
                    elif physics_domain is not None and unit is not None:
                        # IMASNode.physics_domain is a scalar String;
                        # use = not IN.
                        tx.run(
                            f"""
                            MATCH (sns:StandardNameSource)
                            WHERE sns.status = 'extracted'
                              AND coalesce(sns.attempt_count, 0) < $max_attempts
                              AND sns.claimed_at IS NULL
                              {facility_where}
                              {scope_sns_where}
                            MATCH (sns)-[:FROM_DD_PATH]
                                ->(imas:IMASNode)
                            WHERE imas.physics_domain
                                = $fallback_domain
                            MATCH (imas)-[:HAS_UNIT]
                                ->(:Unit {{id: $unit}})
                            WITH sns LIMIT $expand_limit
                            SET sns.claimed_at = datetime(),
                                sns.claim_token = $token,
                                sns.attempt_count = coalesce(sns.attempt_count, 0) + 1
                            """,
                            token=token,
                            fallback_domain=physics_domain,
                            unit=unit,
                            expand_limit=expand_limit,
                            max_attempts=_MAX_COMPOSE_CLAIM_ATTEMPTS,
                            **extra_params,
                        )
                        expanded = True

                    if not expanded and batch_key:
                        # Last resort: group by batch_key.
                        tx.run(
                            f"""
                            MATCH (sns:StandardNameSource)
                            WHERE sns.status = 'extracted'
                              AND coalesce(sns.attempt_count, 0) < $max_attempts
                              AND sns.claimed_at IS NULL
                              AND sns.batch_key = $batch_key
                              {facility_where}
                              {scope_sns_where}
                            WITH sns LIMIT $expand_limit
                            SET sns.claimed_at = datetime(),
                                sns.claim_token = $token,
                                sns.attempt_count = coalesce(sns.attempt_count, 0) + 1
                            """,
                            token=token,
                            batch_key=batch_key,
                            expand_limit=expand_limit,
                            max_attempts=_MAX_COMPOSE_CLAIM_ATTEMPTS,
                            **extra_params,
                        )

                # ── Step 3: Read-back ────────────────────────────
                results = list(
                    tx.run(
                        """
                        MATCH (sns:StandardNameSource
                               {claim_token: $token})
                        OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(im:IMASNode)
                        OPTIONAL MATCH (sns)-[:FROM_SIGNAL]->(fs:FacilitySignal)
                        RETURN sns.id AS id,
                               sns.source_id AS source_id,
                               sns.source_type AS source_type,
                               sns.batch_key AS batch_key,
                               sns.description AS description,
                               coalesce(sns.physics_domain, im.physics_domain, fs.physics_domain) AS physics_domain,
                               sns.claim_token AS claim_token
                        """,
                        token=token,
                    )
                )

                items = [dict(r) for r in results]
                tx.commit()

            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    # Drop claim-race losers BEFORE the (paid) name-composition LLM call (see
    # _verify_source_claim_winners). generate_name is gated on
    # StandardNameSource.status='extracted'.
    items = _verify_source_claim_winners(items)

    logger.debug(
        "claim_generate_name_batch: claimed %d (token=%s)",
        len(items),
        token[:8],
    )
    return items


# -- enrich (StandardName, enriched_at IS NULL) -------------------------------


@retry_on_deadlock()
def claim_enrich_seed_and_expand(
    min_score_threshold: float = 0.0,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes that lack documentation enrichment.

    Eligibility: ``validation_status = 'valid'`` AND ``enriched_at IS NULL``.

    Returns claimed items as dicts.
    """
    where = "sn.validation_status = 'valid' AND sn.enriched_at IS NULL"
    params: dict[str, Any] = {}

    return _claim_sn_atomic(
        eligibility_where=where,
        query_params=params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", sn.enriched_at AS enriched_at",
    )


# -- review names (StandardName, reviewed_name_at IS NULL) --------------------


@retry_on_deadlock()
def claim_review_names_seed_and_expand(
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    min_score: float = DEFAULT_MIN_SCORE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name-axis review scoring.

    Eligibility: ``reviewed_name_at IS NULL`` AND
    ``validation_status = 'valid'``.

    B3 exclusivity: names with ``reviewer_score_name < min_score`` are
    reserved for the regen pool and excluded here via
    ``coalesce(sn.reviewer_score_name, 1.0) >= $min_score``.
    """
    where = (
        "sn.reviewed_name_at IS NULL"
        " AND sn.validation_status = 'valid'"
        " AND coalesce(sn.reviewer_score_name, 1.0) >= $min_score"
    )
    return _claim_sn_atomic(
        eligibility_where=where,
        query_params={"min_score": min_score},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.reviewer_score_name AS reviewer_score_name"
            ", sn.reviewed_name_at AS reviewed_name_at"
        ),
    )


# -- review_name (Phase 8.1: name_stage='drafted', claim only) ----------------


@retry_on_deadlock()
def claim_review_name_batch(
    facility: str | None = None,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    edits_only: bool = False,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name review (Phase 8.1 stage machine).

    Eligibility: ``name_stage = 'drafted'`` AND ``claimed_at IS NULL``.

    Does NOT transition stage at claim time — stage remains ``'drafted'``
    until :func:`persist_reviewed_name` writes the final outcome.  Only
    ``claim_token`` and ``claimed_at`` are set so the orphan sweep can
    recover stuck claims.

    Returns claimed items as dicts with keys:
    ``id``, ``name``, ``description``, ``documentation``, ``kind``,
    ``unit``, ``tags``, ``physics_domain``, ``chain_length``,
    ``claim_token``.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    where = (
        "sn.name_stage = 'drafted'"
        " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
        # Gate: require a real description before review so the
        # semantic_similarity_check in the review worker can run. Exclude the
        # deterministic-parent placeholder — a derived parent still carrying it
        # has no real description to review and must wait for enrichment.
        " AND sn.description IS NOT NULL"
        " AND sn.description <> $parent_desc_placeholder"
        # Derived parents are NEVER name-reviewed (mirrors claim_refine_name_batch,
        # which already excludes origin='derived'). A derived parent's name is a
        # deterministic grammar peel that already passed the admission gate and
        # generalises its accepted children by construction — the name must NOT
        # change, so reviewing it is both wasteful and a route to a stuck
        # 'reviewed' state (review cannot rename, refine excludes them). They are
        # accepted STRUCTURALLY (persist_enriched_parent / the structural-accept
        # repair) with a child-inherited score; quality is gated on the docs axis.
        " AND coalesce(sn.origin, '') <> 'derived'"
    )
    query_params: dict[str, Any] = {
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    }
    if facility is not None:
        where += " AND sn.facility = $facility"
        query_params["facility"] = facility
    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.name AS name"
            ", sn.tags AS tags"
            ", coalesce(sn.chain_length, 0) AS chain_length"
            ", sn.name_stage AS name_stage"
            ", sn.origin AS origin"
            ", sn.edit_mode AS edit_mode"
            ", sn.name_hint AS name_hint"
            ", sn.docs_hint AS docs_hint"
            ", sn.edit_reason AS edit_reason"
            ", sn.edit_origin AS edit_origin"
            ", sn.physical_base AS physical_base"
            ", sn.geometry AS geometry"
            ", sn.grammar_parse_version AS grammar_parse_version"
        ),
        domain=domain,
        scope_run_id=scope_run_id,
        edits_only=edits_only,
    )
    # Drop claim-race losers BEFORE the reviewer quorum spends LLM calls (see
    # _verify_name_claim_winners). review_name is gated on name_stage='drafted'.
    return _verify_name_claim_winners(items, eligible_stage="drafted")


# -- persist_reviewed_name (Phase 8.1: write review + stage transition) -------


@retry_on_deadlock()
def persist_reviewed_name(
    *,
    sn_id: str,
    claim_token: str,
    score: float,
    scores: dict[str, Any] | None = None,
    comments: str | None = None,
    comments_per_dim: dict[str, Any] | None = None,
    model: str,
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    llm_cost: float | None = None,
    llm_tokens_in: int | None = None,
    llm_tokens_out: int | None = None,
    llm_tokens_cached_read: int | None = None,
    llm_tokens_cached_write: int | None = None,
    llm_at: str | None = None,
    llm_service: str | None = None,
    run_id: str | None = None,
    skip_review_node: bool = False,
) -> str:
    """Persist name-review results and transition ``name_stage``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. Compute target stage:
       - ``'accepted'`` if ``score >= min_score`` (score-canonical)
       - ``'exhausted'`` if ``chain_length >= rotation_cap`` and score below min_score
         (cap reached, no further refine — the escalated final attempt
         at chain_length == rotation_cap-1 has already been spent)
       - ``'reviewed'`` otherwise (eligible for refine_name pickup;
         at chain_length == rotation_cap-1 this routes through the
         Opus escalator in process_refine_name_batch)
    3. SET reviewer fields, ``name_stage``, clear claim state.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    score:
        Normalised review score (0-1).
    scores:
        Per-dimension sub-scores dict (written as JSON to
        ``reviewer_scores_name``).
    comments:
        Free-text reviewer comments (written to ``reviewer_comments_name``).
    comments_per_dim:
        Per-dimension comments dict (written as JSON to
        ``reviewer_comments_per_dim_name``).
    model:
        LLM model slug used for this review.
    min_score:
        Acceptance threshold.
    rotation_cap:
        Maximum chain depth before exhaustion (same value used by
        :func:`claim_refine_name_batch`).

    Returns
    -------
    str
        The new ``name_stage`` value (``'accepted'``, ``'reviewed'``, or
        ``'exhausted'``).  Returns ``''`` when the token did not match
        (no-op).
    """
    import json as _json

    # Read current chain_length — needed for exhaustion check.
    # Relax claim_token guard: accept token match OR cleared token (orphan
    # sweep may have nullified claim_token while the RD-quorum review was
    # still in flight).  The name_stage='drafted' check is the real CAS
    # guard — a name can only transition OUT of drafted once.
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.name_stage = 'drafted'
              AND (sn.claim_token = $token OR sn.claim_token IS NULL)
            RETURN coalesce(sn.chain_length, 0) AS chain_length,
                   sn.edit_status AS edit_status,
                   sn.edit_scope AS edit_scope,
                   sn.edit_override_edits AS edit_override_edits,
                   sn.edit_include_accepted AS edit_include_accepted
            """,
            id=sn_id,
            token=claim_token,
        )

    if not rows:
        logger.debug(
            "persist_reviewed_name: stage/token mismatch for %s (token=%s) — no-op",
            sn_id,
            claim_token[:8],
        )
        return ""

    chain_length: int = int(rows[0]["chain_length"])
    edit_status_before: str | None = rows[0].get("edit_status")
    edit_scope_before: str | None = rows[0].get("edit_scope")
    # Cascade-authorization opt-in recorded at edit time (coalesce to False —
    # a rename edit that did not opt in must NOT silently clobber protected
    # (catalog_edit) or accepted descendants when the root rename is accepted).
    edit_override_before: bool = bool(rows[0].get("edit_override_edits"))
    edit_include_before: bool = bool(rows[0].get("edit_include_accepted"))

    # ── Stage decision ────────────────────────────────────────────────
    # Score is canonical (rubric-driven 0–1).
    # Exhaustion fires only at chain_length >= rotation_cap so that
    # chain_length == rotation_cap-1 stays 'reviewed' and routes through
    # the Opus escalator in process_refine_name_batch (the dead branch
    # gated by `escalate = chain_length >= rotation_cap - 1`).  Pre-fix
    # the SN was marked 'exhausted' before the escalator could fire.
    if score >= min_score:
        target_stage = "accepted"
    elif chain_length >= rotation_cap:
        target_stage = "exhausted"
    else:
        target_stage = "reviewed"

    # ── Cascade atomicity preflight (edit rename acceptance) ─────────────
    # A family/subtree rename edit accepts the ROOT and cascades every
    # descendant id atomically.  Validate the WHOLE descendant cascade
    # (ISN grammar round-trip + uniqueness of every new id) BEFORE the
    # acceptance is written — so a constraint collision or a grammar-invalid
    # descendant id refuses the acceptance itself rather than leaving the
    # root accepted with a half-applied cascade.  On conflict the root is
    # NOT accepted (it stays reviewable) and no descendant is renamed:
    # nothing persisted.  The auto-commit query primitive cannot span the
    # acceptance and the rename in one Neo4j transaction, so atomicity is
    # achieved by gating the accept on a clean preflight; the residual
    # accept→apply window is the same TOCTOU the pool path carries.
    cascade_preflight_conflicts: list[str] = []
    if target_stage == "accepted" and edit_scope_before in ("family", "subtree"):
        from imas_codex.standard_names.cascade import cascade_descendants_of

        with GraphClient() as gc:
            pred_rows = gc.query(
                """
                MATCH (sn:StandardName {id: $id})-[:REFINED_FROM]->(pred:StandardName)
                RETURN pred.id AS pred_id
                """,
                id=sn_id,
            )
            old_root_pf = pred_rows[0]["pred_id"] if pred_rows else sn_id
            preflight = cascade_descendants_of(
                gc,
                successor_id=sn_id,
                old_root=old_root_pf,
                new_root=sn_id,
                dry_run=True,
                override_edits=edit_override_before,
                include_accepted=edit_include_before,
            )
        if preflight.conflicts:
            cascade_preflight_conflicts = preflight.conflicts
            # Refuse the acceptance — the reviewed decision cannot land
            # while its atomic consequences (the descendant cascade) are
            # invalid. Route back to 'reviewed' so an operator can resolve
            # the collision; nothing is accepted and nothing is renamed.
            target_stage = "reviewed"
            logger.warning(
                "persist_reviewed_name: %s scored acceptance but its "
                "descendant cascade has %d conflict(s) — refusing acceptance "
                "(name_stage=reviewed, no rename applied): %s",
                sn_id,
                len(preflight.conflicts),
                preflight.conflicts,
            )

    # ── Edit lifecycle decision ──────────────────────────────────────
    # Mirrors the `open → applied | exhausted | rejected` lifecycle
    # documented on EditStatus. A 'reviewed' outcome leaves the edit
    # 'open' — it rides the refine_name loop (persist_refined_name
    # propagates the still-open edit fields onto the next rotation).
    new_edit_status = edit_status_before
    if edit_status_before == "open":
        if target_stage == "accepted":
            new_edit_status = "applied"
        elif target_stage == "exhausted":
            new_edit_status = "exhausted"
        else:
            new_edit_status = "open"

    scores_json = _json.dumps(scores) if scores is not None else None
    comments_per_dim_json = (
        _json.dumps(comments_per_dim) if comments_per_dim is not None else None
    )

    with GraphClient() as gc:
        write_rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.name_stage = 'drafted'
              AND (sn.claim_token = $token OR sn.claim_token IS NULL)
            SET sn.reviewer_score_name        = $score,
                sn.reviewer_scores_name       = $scores_json,
                sn.reviewer_comments_name     = $comments,
                sn.reviewer_comments_per_dim_name = $comments_per_dim_json,
                sn.reviewer_model_name        = $model,
                sn.reviewed_name_at           = datetime(),
                sn.name_stage                 = $target_stage,
                sn.edit_status                = $new_edit_status,
                sn.claim_token                = null,
                sn.claimed_at                 = null
            RETURN sn.id AS id
            """,
            id=sn_id,
            token=claim_token,
            score=score,
            scores_json=scores_json,
            comments=comments,
            comments_per_dim_json=comments_per_dim_json,
            model=model,
            target_stage=target_stage,
            new_edit_status=new_edit_status,
        )

    # A concurrent reviewer already transitioned this node out of 'drafted'
    # between our readback and this SET — our review is a duplicate; report a
    # no-op so the worker does not double-count or emit a spurious event.
    if not write_rows:
        logger.debug(
            "persist_reviewed_name: %s already transitioned out of 'drafted' "
            "(concurrent reviewer won) — no-op",
            sn_id,
        )
        return ""

    logger.info(
        "persist_reviewed_name: %s → name_stage=%s (score=%.3f, chain=%d/%d)",
        sn_id,
        target_stage,
        score,
        chain_length,
        rotation_cap,
    )

    # Acceptance refused by the cascade preflight — record why on the node so
    # an operator can resolve the collision. Nothing was accepted, nothing
    # renamed; the name stays reviewable.
    if cascade_preflight_conflicts:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                SET sn.validation_issues = coalesce(sn.validation_issues, [])
                    + [c IN $conflicts | '[edit_cascade] ' + c]
                """,
                id=sn_id,
                conflicts=cascade_preflight_conflicts,
            )

    # ── Apply the staged descendant cascade atomically on edit acceptance ──
    # scope='family'|'subtree' means sn_id is the (possibly parent-mapped)
    # rename root of an `imas-codex sn edit` proposal — accepting it here is
    # the reviewed decision; descendants never individually re-enter LLM
    # review, they follow the root atomically. The cascade was already
    # preflighted clean above (no conflicts), so this apply reproduces that
    # validated plan against the LIVE subtree.
    if (
        target_stage == "accepted"
        and new_edit_status == "applied"
        and edit_scope_before in ("family", "subtree")
    ):
        from imas_codex.standard_names.cascade import cascade_descendants_of

        with GraphClient() as gc:
            pred_rows = gc.query(
                """
                MATCH (sn:StandardName {id: $id})-[:REFINED_FROM]->(pred:StandardName)
                RETURN pred.id AS pred_id
                """,
                id=sn_id,
            )
            old_root = pred_rows[0]["pred_id"] if pred_rows else sn_id
            cascade_result = cascade_descendants_of(
                gc,
                successor_id=sn_id,
                old_root=old_root,
                new_root=sn_id,
                dry_run=False,
                override_edits=edit_override_before,
                include_accepted=edit_include_before,
            )
            if cascade_result.conflicts:
                # The preflight was clean, so a conflict here is a genuine
                # concurrent-write race between preflight and apply. Record it
                # for operator follow-up; the acceptance already landed.
                logger.warning(
                    "persist_reviewed_name: edit accepted for %s but the "
                    "descendant cascade raced to %d conflict(s) after a clean "
                    "preflight — descendants left unrenamed: %s",
                    sn_id,
                    len(cascade_result.conflicts),
                    cascade_result.conflicts,
                )
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $id})
                    SET sn.validation_issues = coalesce(sn.validation_issues, [])
                        + [c IN $conflicts | '[edit_cascade] ' + c]
                    """,
                    id=sn_id,
                    conflicts=cascade_result.conflicts,
                )
            else:
                logger.info(
                    "persist_reviewed_name: edit-cascade applied for %s — "
                    "%d descendant(s) renamed",
                    sn_id,
                    len(cascade_result.renamed),
                )

    # ── Write StandardNameReview node + HAS_REVIEW edge (Finding 1 fix) ──
    # The single-reviewer worker path was previously SETting reviewer_*
    # fields on the SN node only — no Review node was ever created.
    # Mirror the RD-quorum schema: cycle_index=0, role='primary',
    # canonical=True, fresh review_group_id per call.
    #
    # When ``skip_review_node`` is True the caller (pool RD-quorum
    # worker) is responsible for writing per-cycle Review nodes itself
    # and only needs the SN-side stage transition + axis-slot mirror.
    if not skip_review_node:
        try:
            import uuid as _uuid
            from datetime import datetime as _dt

            _now_iso = llm_at or _dt.now(UTC).isoformat()
            _group_id = str(_uuid.uuid4())
            _review_id = f"{sn_id}:names:{_group_id}:0"
            # Map normalised score (0-1) to tier name per the rubric.
            if score >= 0.85:
                _tier = "outstanding"
            elif score >= 0.60:
                _tier = "good"
            elif score >= 0.40:
                _tier = "inadequate"
            else:
                _tier = "poor"
            write_reviews(
                [
                    {
                        "id": _review_id,
                        "standard_name_id": sn_id,
                        "model": model,
                        "reviewer_model": model,
                        "model_family": "other",
                        "is_canonical": True,
                        "score": float(score),
                        "scores_json": scores_json or "{}",
                        "tier": _tier,
                        "comments": comments or "",
                        "comments_per_dim_json": comments_per_dim_json,
                        "suggested_name": "",
                        "suggestion_justification": "",
                        "reviewed_at": _now_iso,
                        "review_axis": "names",
                        "cycle_index": 0,
                        "review_group_id": _group_id,
                        "resolution_role": "primary",
                        "resolution_method": None,
                        "llm_model": model,
                        "llm_cost": llm_cost,
                        "llm_tokens_in": llm_tokens_in,
                        "llm_tokens_out": llm_tokens_out,
                        "llm_tokens_cached_read": llm_tokens_cached_read,
                        "llm_tokens_cached_write": llm_tokens_cached_write,
                        "llm_at": _now_iso,
                        "llm_service": llm_service or "standard-names",
                    }
                ],
            )
        except Exception:
            # Don't let review-node bookkeeping fail the stage transition.
            logger.exception(
                "persist_reviewed_name: failed to write StandardNameReview for %s",
                sn_id,
            )

    # Async counter bump — live progress visibility for ``sn status``
    bump_sn_run_counter(run_id, "names_reviewed")

    return target_stage


# -- review_docs (Phase 8.1: docs_stage='drafted', claim only) ----------------


@retry_on_deadlock()
def claim_review_docs_batch(
    facility: str | None = None,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    edits_only: bool = False,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for docs review (Phase 8.1 stage machine).

    Eligibility: ``docs_stage = 'drafted'`` AND ``claimed_at IS NULL``.

    Does NOT transition stage at claim time — stage remains ``'drafted'``
    until :func:`persist_reviewed_docs` writes the final outcome.  Only
    ``claim_token`` and ``claimed_at`` are set so the orphan sweep can
    recover stuck claims.

    Returns claimed items as dicts with keys:
    ``id``, ``name``, ``description``, ``documentation``, ``kind``,
    ``unit``, ``tags``, ``physics_domain``, ``docs_chain_length``,
    ``claim_token``.
    """
    # Name-form-vetted gate (same invariant as generate_docs): a real
    # name-review score OR a structurally-accepted derived parent. CURATIVE-SCOPE
    # EXCEPTION: under a scope_run_id (curative family-docs wave) the names are
    # already-accepted and operator-authorised for re-docs, so the name-score
    # gate is dropped inside the scope (mirrors claim_generate_docs_batch) —
    # otherwise derived-leaf families draft docs that can never be reviewed.
    # The same operator authorisation holds under edits_only: an open sn-edit
    # is an explicit steered proposal on a name whose form is already
    # authoritative (catalog-imported names carry no reviewer_score_name), so
    # without this exemption staged docs edits are permanently unclaimable.
    score_gate = (
        ""
        if (scope_run_id or edits_only)
        else (
            " AND (sn.reviewer_score_name IS NOT NULL"
            " OR (coalesce(sn.origin, '') = 'derived'"
            "     AND EXISTS { MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)"
            "       WHERE NOT coalesce(kid.name_stage, '') IN"
            "       ['superseded', 'exhausted', 'contested'] }))"
        )
    )
    where = (
        "sn.docs_stage = 'drafted'"
        " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
        + score_gate
    )
    if facility is not None:
        where += " AND sn.facility = $facility"
        query_params: dict[str, Any] = {"facility": facility}
    else:
        query_params = {}
    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.name AS name"
            ", sn.tags AS tags"
            ", coalesce(sn.docs_chain_length, 0) AS docs_chain_length"
            ", sn.docs_stage AS docs_stage"
            ", sn.edit_mode AS edit_mode"
            ", sn.name_hint AS name_hint"
            ", sn.docs_hint AS docs_hint"
            ", sn.edit_reason AS edit_reason"
            ", sn.edit_origin AS edit_origin"
        ),
        domain=domain,
        scope_run_id=scope_run_id,
        edits_only=edits_only,
    )
    # Drop claim-race losers BEFORE the reviewer quorum spends LLM calls (see
    # _verify_docs_claim_winners). review_docs is gated on docs_stage='drafted'.
    return _verify_docs_claim_winners(items, eligible_stage="drafted")


# -- persist_reviewed_docs (Phase 8.1: write review + docs_stage transition) --


@retry_on_deadlock()
def persist_reviewed_docs(
    *,
    sn_id: str,
    claim_token: str,
    score: float,
    scores: dict[str, Any] | None = None,
    comments: str | None = None,
    comments_per_dim: dict[str, Any] | None = None,
    model: str,
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    llm_cost: float | None = None,
    llm_tokens_in: int | None = None,
    llm_tokens_out: int | None = None,
    llm_tokens_cached_read: int | None = None,
    llm_tokens_cached_write: int | None = None,
    llm_at: str | None = None,
    llm_service: str | None = None,
    run_id: str | None = None,
    skip_review_node: bool = False,
) -> str:
    """Persist docs-review results and transition ``docs_stage``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. Compute target stage:
       - ``'accepted'`` if ``score >= min_score`` (score-canonical)
       - ``'exhausted'`` if ``docs_chain_length >= rotation_cap``
         (cap reached, no further refine — the escalated final
         attempt at chain == rotation_cap-1 has already been spent)
       - ``'reviewed'`` otherwise (eligible for refine_docs pickup;
         at docs_chain_length == rotation_cap-1 this routes through
         the Opus escalator in process_refine_docs_batch)
    3. SET reviewer_docs fields, ``docs_stage``, clear claim state.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    score:
        Normalised review score (0-1).
    scores:
        Per-dimension sub-scores dict (written as JSON to
        ``reviewer_scores_docs``).
    comments:
        Free-text reviewer comments (written to ``reviewer_comments_docs``).
    comments_per_dim:
        Per-dimension comments dict (written as JSON to
        ``reviewer_comments_per_dim_docs``).
    model:
        LLM model slug used for this review.
    min_score:
        Acceptance threshold.
    rotation_cap:
        Maximum chain depth before exhaustion (same value used by
        :func:`claim_refine_docs_batch`).

    Returns
    -------
    str
        The new ``docs_stage`` value (``'accepted'``, ``'reviewed'``, or
        ``'exhausted'``).  Returns ``''`` when the token did not match
        (no-op).
    """
    import json as _json

    # Read current docs_chain_length — needed for exhaustion check.
    # Relax claim_token guard (same rationale as persist_reviewed_name).
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE (sn.claim_token = $token OR sn.claim_token IS NULL)
              AND sn.docs_stage = 'drafted'
              AND sn.name_stage = 'accepted'
            RETURN coalesce(sn.docs_chain_length, 0) AS docs_chain_length,
                   sn.documentation AS documentation,
                   sn.edit_status AS edit_status
            """,
            id=sn_id,
            token=claim_token,
        )

    if not rows:
        logger.debug(
            "persist_reviewed_docs: stage/token mismatch for %s (token=%s) — no-op",
            sn_id,
            claim_token[:8],
        )
        return ""

    docs_chain_length: int = int(rows[0]["docs_chain_length"])
    edit_status_before: str | None = rows[0].get("edit_status")

    # ── Stage decision ────────────────────────────────────────────────
    # Score is canonical (see persist_reviewed_name for rationale,
    # including the rotation_cap-1 vs rotation_cap escalator gate).
    if score >= min_score:
        target_stage = "accepted"
    elif docs_chain_length >= rotation_cap:
        target_stage = "exhausted"
    else:
        target_stage = "reviewed"

    # ── Link label/target gate (accept path only) ─────────────────────
    # A markdown link whose text names one standard name while its target
    # resolves to a different one is a physics-referencing error the LLM
    # reviewers systematically miss. Block promotion and route through
    # refine_docs with the findings in the comments; never exhaust a doc
    # over a link nit (accept at the cap and leave it to the batch lint).
    if target_stage == "accepted":
        try:
            with GraphClient() as gc:
                mismatches = _doc_link_mismatches(gc, rows[0].get("documentation"))
        except Exception:
            logger.debug(
                "persist_reviewed_docs: link-mismatch scan failed for %s "
                "(non-fatal; accepting)",
                sn_id,
                exc_info=True,
            )
            mismatches = []
        if mismatches and docs_chain_length < rotation_cap:
            target_stage = "reviewed"
            # Clamp below the accept threshold so claim_refine_docs_batch
            # (gated on reviewer_score_docs < min_score) picks the doc up.
            score = min(score, min_score - 0.001)
            note = (
                "link text names a different standard name than its target "
                "resolves to — make each label match its target id (or "
                "reword to plain prose): " + "; ".join(mismatches)
            )
            comments = f"{comments}\n{note}" if comments else note
            # Per-dim comments are what the refine prompt renders — the
            # free-text field is not in its context.
            comments_per_dim = {**(comments_per_dim or {}), "link_integrity": note}
            logger.info(
                "persist_reviewed_docs: %s demoted to reviewed — %d mismatched "
                "doc link(s)",
                sn_id,
                len(mismatches),
            )

    # ── Edit lifecycle decision (docs axis) ────────────────────────────
    # Computed after the link-mismatch demotion above so a demoted
    # 'accepted' → 'reviewed' correctly keeps the edit 'open' rather than
    # prematurely marking it 'applied'. Same open → applied | exhausted |
    # (stays open) lifecycle as persist_reviewed_name. Docs edits never
    # cascade — the transition is purely a lifecycle stamp on this node.
    new_edit_status = edit_status_before
    if edit_status_before == "open":
        if target_stage == "accepted":
            new_edit_status = "applied"
        elif target_stage == "exhausted":
            new_edit_status = "exhausted"
        else:
            new_edit_status = "open"

    scores_json = _json.dumps(scores) if scores is not None else None
    comments_per_dim_json = (
        _json.dumps(comments_per_dim) if comments_per_dim is not None else None
    )

    with GraphClient() as gc:
        write_rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE (sn.claim_token = $token OR sn.claim_token IS NULL)
              AND sn.name_stage = 'accepted'
              AND sn.docs_stage = 'drafted'
            SET sn.reviewer_score_docs        = $score,
                sn.reviewer_scores_docs       = $scores_json,
                sn.reviewer_comments_docs     = $comments,
                sn.reviewer_comments_per_dim_docs = $comments_per_dim_json,
                sn.reviewer_model_docs        = $model,
                sn.reviewed_docs_at           = datetime(),
                sn.docs_stage                 = $target_stage,
                sn.edit_status                = $new_edit_status,
                sn.claim_token                = null,
                sn.claimed_at                 = null
            RETURN sn.id AS id
            """,
            id=sn_id,
            token=claim_token,
            score=score,
            scores_json=scores_json,
            comments=comments,
            comments_per_dim_json=comments_per_dim_json,
            model=model,
            target_stage=target_stage,
            new_edit_status=new_edit_status,
        )
    # A concurrent reviewer already transitioned this node out of 'drafted'
    # between our readback and this SET — our review is a duplicate; report a
    # no-op so the worker does not double-count or emit a spurious event.
    if not write_rows:
        logger.debug(
            "persist_reviewed_docs: %s already transitioned out of 'drafted' "
            "(concurrent reviewer won) — no-op",
            sn_id,
        )
        return ""

    logger.info(
        "persist_reviewed_docs: %s → docs_stage=%s (score=%.3f, chain=%d/%d)",
        sn_id,
        target_stage,
        score,
        docs_chain_length,
        rotation_cap,
    )

    # ── Normalize bare [name] brackets AT acceptance (source-of-truth fix) ──
    # A doc written by a late in-flight generate/refine task can finalize with
    # a bare ``[name]`` bracket after the per-rotation reconcile has already
    # run. Normalizing the node here — the moment it promotes to accepted —
    # guarantees no accepted doc ever carries a bare bracket, regardless of
    # when it was written, so the published catalogue stays clean without the
    # operator's per-cycle manual ``resolve_doc_links`` sweep. Scoped to this
    # one node (no full-catalogue scan). The post-drain reconcile remains as a
    # belt-and-suspenders net for cross-references to names accepted later.
    if target_stage == "accepted":
        try:
            with GraphClient() as gc:
                _normalize_bare_doc_links(gc, sn_id=sn_id)
        except Exception:
            logger.debug(
                "persist_reviewed_docs: bare-link normalize failed for %s "
                "(non-fatal; post-drain reconcile will catch it)",
                sn_id,
                exc_info=True,
            )

    # ── Write StandardNameReview node + HAS_REVIEW edge (Finding 1 fix) ──
    # When ``skip_review_node`` is True the caller (pool RD-quorum
    # worker) writes per-cycle Review nodes itself.
    if not skip_review_node:
        try:
            import uuid as _uuid
            from datetime import datetime as _dt

            _now_iso = llm_at or _dt.now(UTC).isoformat()
            _group_id = str(_uuid.uuid4())
            _review_id = f"{sn_id}:docs:{_group_id}:0"
            if score >= 0.85:
                _tier = "outstanding"
            elif score >= 0.60:
                _tier = "good"
            elif score >= 0.40:
                _tier = "inadequate"
            else:
                _tier = "poor"
            write_reviews(
                [
                    {
                        "id": _review_id,
                        "standard_name_id": sn_id,
                        "model": model,
                        "reviewer_model": model,
                        "model_family": "other",
                        "is_canonical": True,
                        "score": float(score),
                        "scores_json": scores_json or "{}",
                        "tier": _tier,
                        "comments": comments or "",
                        "comments_per_dim_json": comments_per_dim_json,
                        "suggested_name": "",
                        "suggestion_justification": "",
                        "reviewed_at": _now_iso,
                        "review_axis": "docs",
                        "cycle_index": 0,
                        "review_group_id": _group_id,
                        "resolution_role": "primary",
                        "resolution_method": None,
                        "llm_model": model,
                        "llm_cost": llm_cost,
                        "llm_tokens_in": llm_tokens_in,
                        "llm_tokens_out": llm_tokens_out,
                        "llm_tokens_cached_read": llm_tokens_cached_read,
                        "llm_tokens_cached_write": llm_tokens_cached_write,
                        "llm_at": _now_iso,
                        "llm_service": llm_service or "standard-names",
                    }
                ],
            )
        except Exception:
            logger.exception(
                "persist_reviewed_docs: failed to write StandardNameReview for %s",
                sn_id,
            )

    # Async counter bump — live progress visibility for ``sn status``
    bump_sn_run_counter(run_id, "names_reviewed")

    return target_stage


# -- refine_name (StandardName, reviewed + low score + chain < cap) -----------


@retry_on_deadlock()
def claim_refine_name_batch(
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
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

    where = (
        "sn.name_stage = 'reviewed'"
        " AND sn.reviewer_score_name IS NOT NULL"
        " AND sn.reviewer_score_name < $min_score"
        " AND coalesce(sn.chain_length, 0) < $rotation_cap"
        " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
        # A pinned rename that has already spent its re-review budget rests at
        # 'reviewed' — refine must not re-claim it (it is never rewritten, only
        # resubmitted; see resubmit_pinned_rename_for_review). Under the cap it
        # IS claimed so the resubmit-to-review can fire.
        " AND NOT (coalesce(sn.edit_mode, '') = 'rename'"
        "          AND coalesce(sn.review_resubmit_count, 0) >= $rotation_cap)"
        # Derived parents (seeded by ``seed_parent_sources``) have no
        # refinement target — the name is structurally fixed. Skip
        # them so they don't enter the refine loop. The admission gate
        # filters bad derived names at write time, and
        # normalize_derived_parent_lifecycle deletes any inadmissible
        # accepted derived parents at startup (the self-healing cascade).
        " AND coalesce(sn.origin, '') <> 'derived'"
    )
    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params={"min_score": min_score, "rotation_cap": rotation_cap},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.reviewer_score_name AS reviewer_score_name"
            ", sn.reviewer_comments_per_dim_name"
            "     AS reviewer_comments_per_dim_name"
            ", sn.chain_length AS chain_length"
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
        ),
        stage_field="name_stage",
        to_stage="refining",
        domain=domain,
        scope_run_id=scope_run_id,
        edits_only=edits_only,
    )

    # Drop claim-race losers BEFORE the (paid) refine LLM call. The claim
    # transitions name_stage 'reviewed'→'refining', but the shared seed's
    # MATCH-bound row + lock-serialised SET still let concurrent replicas
    # each transition the same node; only the claim_token winner truly owns
    # it. See _verify_name_claim_winners.
    items = _verify_name_claim_winners(items, eligible_stage="refining")

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
                    SET sn.embedding    = $embedding,
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
            SET sn.embed_claimed_at = datetime(),
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
            SET sn.embed_claimed_at = null,
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
            SET sn.embedding = item.embedding,
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
) -> dict[str, str]:
    """Persist a refined StandardName as a NEW node with source-edge migration.

    This is the **Option B** persist: since ``StandardName.id`` IS the name
    string, refining a name produces a new node identity.  In a single
    transaction:

    1. MERGE new StandardName with ``name_stage='drafted'``,
       ``chain_length = old_chain_length + 1``.
    2. Create ``(new)-[:REFINED_FROM]->(old)`` edge.
    3. Mark old SN as ``name_stage='superseded'``, clear its claim.
    4. Migrate ``PRODUCED_NAME`` edges from StandardNameSource to new SN.
    5. Migrate ``HAS_STANDARD_NAME`` edges from IMASNode/FacilitySignal to
       new SN.

    Returns ``{"new_name": <new_id>, "old_name": <old_id>}``.

    Raises ``ValueError`` if ``new_name == old_name`` — self-referential
    refinement would create a ``REFINED_FROM`` self-loop.

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
    new_chain_length = old_chain_length + 1

    # Edit-field propagation for the "regular pipeline refine" caller (no
    # edit kwargs passed explicitly): if the predecessor's edit is still
    # open, ride its steering fields forward onto the successor. The
    # `imas-codex sn edit` rename-mode caller always passes edit_mode,
    # edit_reason, and edit_status explicitly, so this lookup is skipped
    # for that path.
    if edit_mode is None and edit_reason is None and edit_status is None:
        with GraphClient() as gc:
            _old_rows = gc.query(
                """
                MATCH (old:StandardName {id: $old_name})
                RETURN old.edit_status AS edit_status,
                       old.edit_mode AS edit_mode,
                       old.name_hint AS name_hint,
                       old.docs_hint AS docs_hint,
                       old.edit_reason AS edit_reason,
                       old.edit_origin AS edit_origin,
                       old.edit_scope AS edit_scope,
                       old.edit_requested_at AS edit_requested_at,
                       old.edit_override_edits AS edit_override_edits,
                       old.edit_include_accepted AS edit_include_accepted
                """,
                old_name=old_name,
            )
        if _old_rows and _old_rows[0].get("edit_status") == "open":
            _old = _old_rows[0]
            edit_mode = _old.get("edit_mode")
            name_hint = _old.get("name_hint")
            docs_hint = _old.get("docs_hint")
            edit_reason = _old.get("edit_reason")
            edit_origin = _old.get("edit_origin")
            edit_scope = _old.get("edit_scope")
            edit_status = _old.get("edit_status")
            edit_requested_at = _old.get("edit_requested_at")
            # Carry the cascade-authorization opt-in forward so a rename edit
            # that rotates through refine still respects the operator's
            # override_edits / include_accepted choice when finally accepted.
            edit_override_edits = _old.get("edit_override_edits")
            edit_include_accepted = _old.get("edit_include_accepted")

    escalation_set = ""
    if escalated:
        escalation_set = ", new.refine_name_escalated_at = datetime()"

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                result = list(
                    tx.run(
                        f"""
                        // 1. Create (or match) new SN with new id
                        MERGE (new:StandardName {{id: $new_name}})
                        ON CREATE SET
                          new.name_stage        = 'drafted',
                          new.docs_stage        = 'pending',
                          new.validation_status = 'valid',
                          new.origin            = 'pipeline',
                          new.chain_length      = $new_chain_length,
                          new.docs_chain_length = 0,
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
                          new.edit_mode         = $edit_mode,
                          new.name_hint         = $name_hint,
                          new.docs_hint         = $docs_hint,
                          new.edit_reason       = $edit_reason,
                          new.edit_origin       = $edit_origin,
                          new.edit_scope        = $edit_scope,
                          new.edit_status       = $edit_status,
                          new.edit_requested_at = $edit_requested_at,
                          new.edit_override_edits = $edit_override_edits,
                          new.edit_include_accepted = $edit_include_accepted
                          {escalation_set}

                        // 2. Link to predecessor
                        WITH new
                        MATCH (old:StandardName {{id: $old_name}})
                        WHERE old.name_stage = 'refining'
                        MERGE (new)-[:REFINED_FROM]->(old)

                        // 3. Mark old as superseded, clear claim
                        //
                        // Record whether the predecessor had reached the
                        // published bar so the export boundary can decide
                        // whether to emit a deprecation stub. The caller has
                        // already flipped name_stage to 'refining' (the WHERE
                        // gate above), so the pre-refining name-axis stage is
                        // no longer legible here; the durable "was published"
                        // signal that survives that flip is docs_stage —
                        // a name only reaches consumers once its docs review
                        // has accepted it. A pipeline refine of a merely
                        // 'reviewed' name (never published) records a
                        // non-accepted sentinel, so no stub is emitted for it.
                        SET old.name_stage  = 'superseded',
                            old.superseded_from_stage = coalesce(
                                old.superseded_from_stage,
                                CASE WHEN coalesce(old.docs_stage, '') = 'accepted'
                                     THEN 'accepted' ELSE 'refining' END),
                            old.claim_token = null,
                            old.claimed_at  = null,
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

                        // 4. Migrate PRODUCED_NAME edges
                        WITH new, old
                        OPTIONAL MATCH (src:StandardNameSource)-[r:PRODUCED_NAME]->(old)
                        WITH new, old,
                             collect(src) AS pn_sources,
                             collect(r)   AS pn_rels
                        FOREACH (rel IN pn_rels | DELETE rel)
                        WITH new, old, pn_sources
                        FOREACH (s IN pn_sources |
                            MERGE (s)-[:PRODUCED_NAME]->(new)
                            SET s.produced_sn_id = new.id)

                        // 5. Migrate HAS_STANDARD_NAME edges
                        WITH DISTINCT new, old
                        OPTIONAL MATCH (n)-[r2:HAS_STANDARD_NAME]->(old)
                        WITH new, old,
                             collect(n)  AS hsn_nodes,
                             collect(r2) AS hsn_rels
                        FOREACH (rel IN hsn_rels | DELETE rel)
                        WITH new, old, hsn_nodes
                        FOREACH (n IN hsn_nodes | MERGE (n)-[:HAS_STANDARD_NAME]->(new))

                        // 5b. Migrate inbound HAS_PARENT edges
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
                             collect({{
                                 child: child,
                                 props: properties(c_old),
                                 rel:   c_old
                             }}) AS comp_links
                        FOREACH (link IN comp_links | DELETE link.rel)
                        WITH new, old, comp_links
                        CALL {{
                            WITH new, comp_links
                            UNWIND comp_links AS link
                            WITH link.child AS ch, link.props AS p, new
                            WHERE ch IS NOT NULL AND ch.id <> new.id
                            MERGE (ch)-[c_new:HAS_PARENT]->(new)
                            SET   c_new = p
                        }}

                        // 6. Inherit HAS_UNIT and IN_CLUSTER edges from the
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
                    )
                )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    if result:
        row = dict(result[0])
        from imas_codex.standard_names.provenance_lifecycle import (
            retarget_standard_name_sources,
        )

        with GraphClient() as provenance_gc:
            retarget_standard_name_sources(
                provenance_gc,
                old_name,
                new_name,
                operation="human_edit" if edit_mode else "refine",
                reason=edit_reason or reason,
                origin=edit_origin,
                run_id=run_id,
            )
        logger.debug(
            "persist_refined_name: %s → %s (chain_length=%d)",
            old_name,
            new_name,
            new_chain_length,
        )

        # --- Embed the new refined name ---
        # Clear embed_text_hash so the dedicated embed worker picks it up.
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                SET sn.embed_text_hash = null
                """,
                id=new_name,
            )

        # Async counter bump — live progress visibility for ``sn status``
        bump_sn_run_counter(run_id, "names_regenerated")
        return row

    # Empty result means the ``MATCH (old) WHERE name_stage='refining'`` gate
    # did not bind — typically because orphan_sweep reverted the claim back
    # to 'reviewed' while the LLM call was in flight. Silently returning a
    # bare row dict here was the root cause of the refine_name silent-bug
    # (5 REFINED_FROM edges from 500+ claims observed 2026-05-18). Raise
    # explicitly so the worker's exception handler either releases the
    # claim cleanly or marks the SN exhausted.
    raise RuntimeError(
        f"persist_refined_name no-op: {old_name} → {new_name} — the old SN "
        f"was not in name_stage='refining' at persist time (likely reverted "
        f"by orphan_sweep). LLM call complete but no graph mutation occurred."
    )


@retry_on_deadlock()
def supersede_prior_source_names(
    pairs: list[dict[str, str]],
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
    - its ``origin`` is pipeline (``NULL`` / ``'pipeline'``) — never
      ``catalog_edit`` (catalog-authoritative) or ``derived`` (structural
      parent owned by the admission gate).

    For each superseded predecessor the helper marks it
    ``name_stage='superseded'``, clears its claim, and creates a
    ``(new)-[:REFINED_FROM]->(old)`` edge so chain history and the SPA's
    lineage widget stay intact.  ``HAS_STANDARD_NAME`` edges are **left in
    place** on the predecessor (they form the historical record); live
    accept/export queries already filter superseded names, so the source's
    effective name is unambiguous.

    Returns the number of predecessor names superseded.
    """
    pairs = [p for p in pairs if p.get("new_name") and p.get("source_id")]
    if not pairs:
        return 0

    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                UNWIND $pairs AS pr
                MATCH (src:IMASNode {id: pr.source_id})-[:HAS_STANDARD_NAME]->(old:StandardName)
                WHERE old.id <> pr.new_name
                  AND NOT coalesce(old.name_stage, '') IN ['superseded', 'exhausted', 'contested']
                  AND coalesce(old.origin, 'pipeline') = 'pipeline'
                MATCH (new:StandardName {id: pr.new_name})
                // Skip self and any case where old already descends from new
                // along the REFINED_FROM chain (would form a cycle).
                WHERE new.id <> old.id
                  AND NOT (old)-[:REFINED_FROM*1..]->(new)
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
                SET old.name_stage = 'superseded',
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
                pairs=pairs,
            )
            or []
        )

    superseded = len(rows)
    if superseded:
        from imas_codex.standard_names.provenance_lifecycle import (
            retarget_standard_name_sources,
        )

        with GraphClient() as provenance_gc:
            for row in rows:
                retarget_standard_name_sources(
                    provenance_gc,
                    row["old_name"],
                    row["new_name"],
                    operation="regenerate",
                )
        for r in rows:
            logger.info(
                "supersede_prior_source_names: %s superseded by %s (same source)",
                r.get("old_name"),
                r.get("new_name"),
            )
    return superseded


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
    'accepted')`` (so the P1 export emits a ``status: deprecated`` stub for it),
    clears its claim, and MERGEs ``(into)-[:REFINED_FROM]->(old)`` so the stub
    resolves to the live successor. Historical ``HAS_STANDARD_NAME`` /
    ``PRODUCED_NAME`` edges on ``old`` are left intact as the provenance record.

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

    with GraphClient() as gc:
        rows = gc.query(
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
        if not row.get("old_id"):
            return {"ok": False, "reason": f"name {old_id!r} not found"}
        if not row.get("into_id"):
            return {"ok": False, "reason": f"target {into_id!r} not found"}
        if row.get("into_stage") not in ("accepted", "approved"):
            return {
                "ok": False,
                "reason": (
                    f"target {into_id!r} is name_stage={row.get('into_stage')!r}, "
                    "not 'accepted' or 'approved' — supersede target must be the live "
                    "canonical name"
                ),
            }
        if row.get("cycle"):
            return {
                "ok": False,
                "reason": (
                    f"{old_id!r} already descends from {into_id!r} "
                    "(REFINED_FROM cycle) — cannot fold"
                ),
            }

        already = row.get("old_stage") == "superseded"
        result = {
            "ok": True,
            "old_id": old_id,
            "into_id": into_id,
            "old_prior_stage": row.get("old_stage"),
            "already_superseded": already,
            "dry_run": dry_run,
        }
        if dry_run:
            return result

        gc.query(
            """
            MATCH (old:StandardName {id: $old_id}),
                  (into:StandardName {id: $into_id})
            SET old.name_stage = 'superseded',
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
            """,
            old_id=old_id,
            into_id=into_id,
        )
        from imas_codex.standard_names.provenance_lifecycle import (
            retarget_standard_name_sources,
        )

        retarget_standard_name_sources(
            gc,
            old_id,
            into_id,
            operation="fold",
        )
    logger.info(
        "tombstone_supersede_into: %s superseded into %s (sfs=accepted, "
        "REFINED_FROM lineage merged)",
        old_id,
        into_id,
    )
    return result


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


@retry_on_deadlock()
def release_generate_name_claims(
    *,
    source_ids: list[str],
    claim_token: str,
) -> int:
    """Release StandardNameSource claims IFF ``claim_token`` matches.

    Clears ``claimed_at`` and ``claim_token`` so items become eligible for
    re-claim.  Used by the generate_name pool's error-recovery path.

    Returns the count of sources actually released.  A count less than
    ``len(source_ids)`` means the orphan sweep (or another worker) already
    cleared the stale claim — the caller can safely ignore this.
    """
    if not source_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (n:StandardNameSource {id: sid})
            WHERE n.claim_token = $token
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=source_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(source_ids):
        logger.debug(
            "release_generate_name_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(source_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_generate_name_failed_claims(
    *,
    source_ids: list[str],
    claim_token: str,
) -> int:
    """Release StandardNameSource claims on worker failure IFF token matches.

    Identical to :func:`release_generate_name_claims`; provided as the
    symmetric "failed" variant so callers can be explicit about intent.

    Returns the count of sources actually released.
    """
    return release_generate_name_claims(
        source_ids=source_ids,
        claim_token=claim_token,
    )


@retry_on_deadlock()
def release_enrich_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
) -> int:
    """Release StandardName enrichment claims IFF token (and stage) match.

    Parameters
    ----------
    sn_ids:
        StandardName node ids to release.
    claim_token:
        Token that was set at claim time; nodes with a different token are
        left untouched.
    expected_stage:
        If provided, also verify ``name_stage = $expected_stage`` before
        clearing.  Pass ``None`` (default) to skip stage verification.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_enrich_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_enrich_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> int:
    """Release StandardName enrichment claims on worker failure.

    Clears claim state and optionally reverts ``name_stage`` to *to_stage*
    when the worker processing failed and the item should be retried.

    Parameters
    ----------
    sn_ids:
        StandardName node ids to release.
    claim_token:
        Token set at claim time.
    from_stage:
        If provided, verify ``name_stage = $from_stage`` before acting.
    to_stage:
        If provided (and *from_stage* matches), revert ``name_stage`` to
        this value so the item is eligible for a fresh claim.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.name_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.name_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            SET {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_enrich_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_names_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
) -> int:
    """Release StandardName review-names claims IFF token (and stage) match.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_names_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_names_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> int:
    """Release StandardName review-names claims on worker failure.

    Clears claim state and optionally reverts ``name_stage``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.name_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.name_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            SET {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_names_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_docs_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
) -> int:
    """Release StandardName review-docs claims IFF token (and stage) match.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_docs_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_docs_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> int:
    """Release StandardName review-docs claims on worker failure.

    Clears claim state and optionally reverts ``name_stage``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.name_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.name_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            SET {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_docs_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_refine_name_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
) -> int:
    """Release StandardName refine-name claims IFF token (and stage) match.

    Clears ``claimed_at`` and ``claim_token`` and reverts
    ``name_stage`` from ``'refining'`` back to ``'reviewed'`` when
    the node is still in the refining stage.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.claimed_at = null,
                n.claim_token = null,
                n.name_stage = CASE
                    WHEN n.name_stage = 'refining' THEN 'reviewed'
                    ELSE n.name_stage
                END
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_refine_name_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_refine_name_failed_claims(
    *,
    sn_ids: list[str],
    token: str,
) -> int:
    """Release refine-name claims after LLM or processing failure.

    Token-and-stage verified: only reverts nodes where
    ``claim_token = $token AND name_stage = 'refining'``.  This prevents
    late-release from clobbering an SN that was already swept by orphan
    recovery or successfully persisted.

    Returns the count of nodes released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
              AND sn.name_stage = 'refining'
            SET sn.name_stage = 'reviewed',
                sn.claim_token = null,
                sn.claimed_at = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=token,
        )
    return result[0]["released"] if result else 0


def resubmit_pinned_rename_for_review(
    *,
    sn_id: str,
    token: str,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
) -> str:
    """Route a below-threshold pinned rename to a fresh review quorum.

    A rename edit (``edit_mode='rename'``) carries an operator-chosen name
    string — the name is a fixed decision, not a draft to be rewritten. When
    such a name scores below threshold (quorum variance on a borderline name),
    the refine pool must NOT try to reword it: re-emitting the identical pinned
    name trips the self-referential-refine guard and decomposing a lexicalised
    base trips grammar validation — either way the pinned name is wrongly marked
    ``exhausted`` and its (already-superseded) predecessor's quantity silently
    drops from export.

    Instead, resubmit the SAME name to review for a fresh quorum draw
    (``name_stage`` → ``'drafted'``, reviewer name-score cleared), bounded by
    ``review_resubmit_count < rotation_cap``. A borderline name whose siblings
    accept at 0.96+ typically clears on a fresh draw. When the cap is reached
    the name is left at ``'reviewed'`` (never exhausted) for operator
    resolution; :func:`claim_refine_name_batch` excludes capped pinned renames
    so they do not re-loop.

    Token-and-stage verified (``claim_token = $token AND name_stage =
    'refining'``). Returns ``'resubmitted'``, ``'capped'``, or ``''`` (no-op:
    token/stage mismatch — a concurrent sweep or worker already moved it).
    """
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            WHERE sn.claim_token = $token AND sn.name_stage = 'refining'
            WITH sn, coalesce(sn.review_resubmit_count, 0) AS n
            SET sn.name_stage = CASE WHEN n < $cap THEN 'drafted' ELSE 'reviewed' END,
                sn.review_resubmit_count = CASE WHEN n < $cap THEN n + 1 ELSE n END,
                sn.reviewer_score_name = CASE WHEN n < $cap
                                              THEN null ELSE sn.reviewer_score_name END,
                sn.claim_token = null,
                sn.claimed_at = null
            RETURN CASE WHEN n < $cap THEN 'resubmitted' ELSE 'capped' END AS outcome
            """,
            sn_id=sn_id,
            token=token,
            cap=rotation_cap,
        )
    return rows[0]["outcome"] if rows else ""


def stage_name_for_rescore(
    sn_id: str,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Return a stranded non-accepted name to review for a fresh quorum.

    Operator recovery path for names stuck at a terminal-but-unpublished
    name-axis stage — an ``exhausted`` name (refine cap reached, or a
    borderline name wrongly exhausted) or a ``reviewed`` name whose score never
    cleared. Reverts ``name_stage`` to ``'drafted'`` so a review re-scores it
    with a fresh quorum, clears the stale reviewer name-score, resets the
    re-review budget, and clears any claim. Edit fields are left intact so an
    attached hint (e.g. a "keep this form" steer) rides the fresh review.

    When *run_id* is given it is stamped on the node so a scoped review
    (``run_sn_pools(scope_run_id=run_id)``) claims exactly this name — the
    mechanism the ``sn rescore`` inline review uses. A predecessor that was
    already superseded stays superseded — the recovered name keeps its
    REFINED_FROM lineage and, once re-accepted, resolves the export gap it
    left behind.

    Refuses (``{"ok": False, "reason": ...}``) for names that must not be
    force-staged: not found, already ``accepted``/``approved``, ``superseded``
    (edit the successor instead), or already live (``drafted``/``refining``).

    Returns ``{"ok": True, "sn_id", "prior_stage", "run_id", "dry_run"}`` on
    success.
    """
    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (sn:StandardName {id: $id}) RETURN sn.name_stage AS stage",
            id=sn_id,
        )
        if not rows:
            return {"ok": False, "reason": f"name {sn_id!r} not found"}
        stage = rows[0].get("stage")
        if stage not in ("exhausted", "reviewed"):
            return {
                "ok": False,
                "reason": (
                    f"{sn_id!r} is name_stage={stage!r} — rescore only recovers "
                    "'exhausted' or 'reviewed' names (accepted names are already "
                    "live; superseded names should be recovered via their successor)"
                ),
            }
        result = {
            "ok": True,
            "sn_id": sn_id,
            "prior_stage": stage,
            "run_id": run_id,
            "dry_run": dry_run,
        }
        if dry_run:
            return result
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.name_stage IN ['exhausted', 'reviewed']
            // A quarantine stamped under an older grammar is stale by
            // definition here — the operator asks for a fresh pass, so
            // validation must re-run under the CURRENT grammar too.
            WITH sn, sn.validation_status = 'quarantined' AS was_quarantined
            SET sn.name_stage = 'drafted',
                sn.reviewer_score_name = null,
                sn.review_resubmit_count = 0,
                sn.claim_token = null,
                sn.claimed_at = null,
                sn.run_id = coalesce($run_id, sn.run_id),
                sn.validation_status = CASE WHEN was_quarantined
                    THEN 'pending' ELSE sn.validation_status END,
                sn.validation_issues = CASE WHEN was_quarantined
                    THEN null ELSE sn.validation_issues END,
                sn.validated_at = CASE WHEN was_quarantined
                    THEN null ELSE sn.validated_at END
            """,
            id=sn_id,
            run_id=run_id,
        )
    logger.info(
        "stage_name_for_rescore: %s (%s) → drafted (run_id=%s)", sn_id, stage, run_id
    )
    return result


def _mark_refine_vocab_gap_exhausted(
    *,
    sn_id: str,
    token: str,
    error_msg: str,
) -> None:
    """Mark a name exhausted when refine fails on a vocab gap.

    Vocab-gap errors are deterministic — the LLM keeps producing the same
    unregistered token. Instead of reverting to 'reviewed' (infinite loop),
    move to 'exhausted' and record the vocab gap reason.
    """
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            WHERE sn.claim_token = $token
              AND sn.name_stage = 'refining'
            SET sn.name_stage = 'exhausted',
                sn.claim_token = null,
                sn.claimed_at = null,
                // A name-axis exhaust closes only a name-steering edit.
                sn.edit_status = CASE WHEN sn.edit_status = 'open'
                                       AND sn.name_hint IS NOT NULL
                                      THEN 'exhausted'
                                      ELSE sn.edit_status END,
                sn.reviewer_comments_name =
                    coalesce(sn.reviewer_comments_name, '')
                    + ' [vocab_gap_exhaust] ' + $error_msg
            """,
            sn_id=sn_id,
            token=token,
            error_msg=error_msg[:300],
        )


def _mark_refine_docs_exhausted(
    *,
    sn_id: str,
    token: str,
    error_msg: str,
) -> None:
    """Mark a name's docs exhausted when refine_docs fails deterministically.

    A docs-validation failure (e.g. the LLM consistently returns docs that
    violate the ``RefinedDocs`` length constraints) is deterministic for a
    given item — reverting ``docs_stage`` to 'reviewed' would re-claim and
    re-burn budget forever.  Move to 'exhausted' instead and record the reason.
    """
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            WHERE sn.claim_token = $token
              AND sn.docs_stage = 'refining'
            SET sn.docs_stage = 'exhausted',
                sn.claim_token = null,
                sn.claimed_at = null,
                // A docs-axis exhaust closes only a docs-steering edit.
                sn.edit_status = CASE WHEN sn.edit_status = 'open'
                                       AND sn.docs_hint IS NOT NULL
                                      THEN 'exhausted'
                                      ELSE sn.edit_status END,
                sn.reviewer_comments_docs =
                    coalesce(sn.reviewer_comments_docs, '')
                    + ' [docs_refine_exhaust] ' + $error_msg
            """,
            sn_id=sn_id,
            token=token,
            error_msg=error_msg[:300],
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


def _verify_docs_claim_winners(
    items: list[dict[str, Any]],
    *,
    eligible_stage: str,
    settle_seconds: float = _CLAIM_VERIFY_SETTLE_SECONDS,
) -> list[dict[str, Any]]:
    """Drop docs-claim items that lost a concurrent-claim race.

    *items* are the rows returned by :func:`_claim_sn_atomic` (all sharing one
    ``claim_token``).  After a short *settle_seconds* pause — long enough for a
    lock-serialised claim burst to finish committing — this re-reads committed
    graph state and keeps only the nodes that STILL hold that token, remain at
    *eligible_stage*, AND carry the exact ``claim_seq`` this worker was
    assigned.  A race loser's ``claim_seq`` was superseded by a later
    committer's higher value, so it is dropped here before any LLM call.

    ``settle_seconds`` is overridable (tests pass ``0``); this function runs in
    a worker thread (claims go through ``asyncio.to_thread``), so the pause does
    not block the event loop.

    Returns the filtered item list (order preserved).
    """
    return _verify_claim_winners(
        items,
        stage_field="docs_stage",
        eligible_stage=eligible_stage,
        settle_seconds=settle_seconds,
        axis="docs",
    )


def _verify_claim_winners(
    items: list[dict[str, Any]],
    *,
    stage_field: str,
    eligible_stage: str,
    settle_seconds: float,
    axis: str,
) -> list[dict[str, Any]]:
    """Shared claim-race winner verifier for the docs and names axes.

    Keeps only items that still hold ``claim_token`` at *eligible_stage* with
    the assigned ``claim_seq`` after settling; see the module comment above
    :func:`_verify_docs_claim_winners` for the mechanism.
    """
    if not items:
        return items
    token = items[0].get("claim_token") or ""
    if not token:
        return items
    # Per-item (id, claim_seq). Legacy claims that predate the seq stamp fall
    # back to token+stage only so this never over-drops on missing data.
    seqs = {it["id"]: it.get("claim_seq") for it in items}
    have_seq = all(v is not None for v in seqs.values())
    # The settle only helps the claim_seq dedup — it gives a lock-serialised
    # claim burst time to finish committing so the seq re-read is decisive.
    # Without claim_seq (legacy nodes / mocked claims) there is nothing for the
    # pause to resolve, so skip it and fall through to the token-only check.
    if have_seq and settle_seconds > 0:
        time.sleep(settle_seconds)
    ids = [it["id"] for it in items]
    with GraphClient() as gc:
        rows = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (sn:StandardName {{id: sid}})
            WHERE sn.claim_token = $token
              AND sn.{stage_field} = $eligible_stage
            RETURN sn.id AS id, sn.claim_seq AS claim_seq
            """,
            ids=ids,
            token=token,
            eligible_stage=eligible_stage,
        )
    if have_seq:
        winners = {r["id"] for r in rows if r.get("claim_seq") == seqs[r["id"]]}
    else:
        winners = {r["id"] for r in rows}
    if len(winners) == len(items):
        return items
    logger.debug(
        "_verify_claim_winners[%s]: %d/%d survived claim-race (token=%s, stage=%s)",
        axis,
        len(winners),
        len(items),
        token[:8],
        eligible_stage,
    )
    return [it for it in items if it["id"] in winners]


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


def _verify_name_claim_winners(
    items: list[dict[str, Any]],
    *,
    eligible_stage: str,
    settle_seconds: float = _CLAIM_VERIFY_SETTLE_SECONDS,
) -> list[dict[str, Any]]:
    """Drop name-claim items that lost a concurrent-claim race.

    Names-axis twin of :func:`_verify_docs_claim_winners` — same settle +
    ``claim_seq`` mechanism, gated on ``name_stage`` instead of ``docs_stage``.
    """
    return _verify_claim_winners(
        items,
        stage_field="name_stage",
        eligible_stage=eligible_stage,
        settle_seconds=settle_seconds,
        axis="name",
    )


def _verify_source_claim_winners(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop generate_name source-claim items that lost a concurrent-claim race.

    The generate_name pool claims :class:`StandardNameSource` nodes (gate
    ``status='extracted'``) rather than StandardName nodes, so it cannot share
    :func:`_verify_name_claim_winners`.  Same mechanism: re-read committed
    ``claim_token`` ownership and keep only the sources this worker still owns
    at the eligible ``status='extracted'`` gate, dropping race losers before
    the (paid) name-composition LLM call.

    Returns the filtered item list (order preserved).
    """
    if not items:
        return items
    token = items[0].get("claim_token") or ""
    if not token:
        return items
    ids = [it["id"] for it in items]
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sns:StandardNameSource {id: sid})
            WHERE sns.claim_token = $token
              AND sns.status = 'extracted'
            RETURN sns.id AS id
            """,
            ids=ids,
            token=token,
        )
    winners = {r["id"] for r in rows}
    if len(winners) == len(items):
        return items
    logger.debug(
        "_verify_source_claim_winners: %d/%d survived claim-race (token=%s)",
        len(winners),
        len(items),
        token[:8],
    )
    return [it for it in items if it["id"] in winners]


# =============================================================================
# generate_docs — claim / persist / release
# =============================================================================
# Stage gate: name_stage='accepted' AND docs_stage='pending'
# The claim does NOT transition docs_stage (persist_generated_docs does that).


@retry_on_deadlock()
def claim_generate_docs_batch(
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    edits_only: bool = False,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes ready for generate_docs.

    Eligibility: ``name_stage = 'accepted'`` AND ``docs_stage = 'pending'``
    AND ``claimed_at IS NULL`` (or stale).

    The claim does NOT transition ``docs_stage`` — that happens in
    :func:`persist_generated_docs` so that a failed worker leaves the node
    cleanly at ``'pending'``.

    Each claimed item is enriched with REFINED_FROM chain history via
    :func:`~imas_codex.standard_names.chain_history.name_chain_history`
    and the name-review feedback fields so the LLM understands why this
    name was accepted.

    Returns claimed items as dicts.
    """
    from imas_codex.standard_names.chain_history import name_chain_history

    # Docs-eligibility gate: the name's FORM must be vetted before we spend
    # docs effort. Two ways to be vetted:
    #   - a regular name earns a ``reviewer_score_name`` by passing
    #     REVIEW_NAME (RD-quorum), OR
    #   - a derived parent's name is a deterministic grammar peel vetted by
    #     the admission gate (``is_admissible_parent_name``) — it is never
    #     name-reviewed/refined and carries NO name score by design (scoring
    #     a name we never change is meaningless). It is docs-eligible by
    #     being a structurally-accepted derived parent.
    #
    # CURATIVE-SCOPE EXCEPTION: when a ``scope_run_id`` is supplied the run is a
    # curative family-docs wave (``sn run --families`` / ``--scope-run-id``) that
    # has EXPLICITLY reset the docs of a bounded set of ALREADY-ACCEPTED names.
    # Those names are accepted (their form is authoritative — many are derived
    # leaves auto-accepted without a name score), so the operator has authorised
    # re-docs'ing them. Drop the name-score gate inside the scope so derived
    # leaf families (e.g. <species>_density_at_limiter) can be curatively
    # regenerated; the ``scope_run_id`` filter keeps the global backlog untouched.
    # The same operator authorisation holds under edits_only (an open sn-edit
    # is an explicit steered proposal; catalog-imported names carry no
    # reviewer_score_name and would otherwise never draft).
    score_gate = (
        ""
        if (scope_run_id or edits_only)
        else (
            " AND (sn.reviewer_score_name IS NOT NULL"
            " OR (coalesce(sn.origin, '') = 'derived'"
            "     AND EXISTS { MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)"
            "       WHERE NOT coalesce(kid.name_stage, '') IN"
            "       ['superseded', 'exhausted', 'contested'] }))"
        )
    )
    where = (
        "sn.name_stage = 'accepted' AND sn.docs_stage = 'pending'"
        " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
        + score_gate
    )

    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params={},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            # description, kind, physics_domain already in base readback —
            # listing them again would raise Neo4j 42N38 (duplicate return
            # item name).  Only add fields not present in _claim_sn_atomic's
            # fixed RETURN list.
            ", sn.tags AS tags"
            ", sn.reviewer_score_name AS reviewer_score_name"
            ", sn.reviewer_comments_name AS reviewer_comments_name"
            ", sn.chain_length AS chain_length"
            ", sn.docs_stage AS docs_stage"
            ", sn.name_stage AS name_stage"
            ", sn.edit_mode AS edit_mode"
            ", sn.name_hint AS name_hint"
            ", sn.docs_hint AS docs_hint"
            ", sn.edit_reason AS edit_reason"
            ", sn.edit_origin AS edit_origin"
        ),
        # stage_field=None → claim only, no stage transition
        domain=domain,
        scope_run_id=scope_run_id,
        edits_only=edits_only,
        # Parent-first ordering: nodes that have incoming HAS_PARENT edges
        # (i.e. nodes which are parents) receive priority 0 so they are
        # documented before their children.
        #
        # Escape hatch: do NOT block a child whose parent is still pending
        # but has never been claimed (claimed_at IS NULL) — the parent may
        # simply not have been seeded, and we should not wait indefinitely.
        # We only block when the parent is *actively* in-progress
        # (docs_stage='pending' AND claimed_at IS NOT NULL).
        seed_extra_where=(
            "AND NOT EXISTS {"
            " MATCH (sn)-[:HAS_PARENT]->(parent:StandardName)"
            " WHERE parent.docs_stage = 'pending'"
            " AND parent.claimed_at IS NOT NULL"
            " }"
        ),
        seed_with_extras=(
            ", CASE WHEN EXISTS { ()-[:HAS_PARENT]->(sn) }"
            " THEN 0 ELSE 1 END AS _docs_priority"
        ),
        seed_order_by="_docs_priority ASC, rand()",
    )

    # Drop claim-race losers BEFORE enrichment / LLM spend (see
    # _verify_docs_claim_winners). generate_docs is gated on docs_stage='pending'.
    items = _verify_docs_claim_winners(items, eligible_stage="pending")

    # Enrich each claimed item with REFINED_FROM chain history.
    for item in items:
        item["chain_history"] = name_chain_history(item["id"])

    logger.debug(
        "claim_generate_docs_batch: claimed %d",
        len(items),
    )
    return items


@retry_on_deadlock()
def persist_generated_docs(
    *,
    sn_id: str,
    claim_token: str,
    description: str,
    documentation: str,
    model: str,
    run_id: str | None = None,
) -> str:
    """Persist generate_docs results and transition ``docs_stage`` to ``'drafted'``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. SET ``description``, ``documentation``, ``docs_stage = 'drafted'``,
       ``docs_chain_length = 0``, ``docs_model``, ``docs_generated_at``,
       clear ``claim_token`` and ``claimed_at``.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    description:
        Short description (1–3 sentences) from the LLM.
    documentation:
        Rich markdown documentation from the LLM.
    model:
        LLM model identifier used for generation.

    Returns the new ``docs_stage`` value (``'drafted'``).
    Raises :exc:`ValueError` if token verification fails (no matching node).
    """
    # Repair any JSON-escape-mangled LaTeX control chars before persisting.
    description = _sanitize_doc_text(description)
    documentation = _sanitize_doc_text(documentation)
    with GraphClient() as gc:
        # Extract cross-references from documentation text. Pass the
        # SN id so self-references in prose do not land in the
        # structured `links` index.
        links = _extract_links_from_docs(documentation, self_name=sn_id)
        link_status = _compute_link_status(links) if links else None
        result = gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            WHERE sn.claim_token = $token
              AND sn.name_stage = 'accepted'
            SET sn.description      = $description,
                sn.documentation    = $documentation,
                sn.docs_stage       = 'drafted',
                sn.docs_chain_length = 0,
                sn.docs_model       = $model,
                sn.docs_generated_at = datetime(),
                sn.generate_docs_count = coalesce(sn.generate_docs_count, 0) + 1,
                sn.claim_token      = null,
                sn.claimed_at       = null,
                sn.links            = $links,
                sn.link_status      = $link_status
            RETURN sn.docs_stage AS docs_stage
            """,
            sn_id=sn_id,
            token=claim_token,
            description=description,
            documentation=documentation,
            model=model,
            links=links,
            link_status=link_status,
        )
    if not result:
        raise ValueError(
            f"persist_generated_docs: token mismatch or node not found for {sn_id!r}"
        )
    new_stage: str = result[0]["docs_stage"]
    logger.debug("persist_generated_docs: %s → docs_stage=%s", sn_id, new_stage)

    # Re-embed: docs generation sets/refines description, so the embedding
    # should reflect the latest "name — description" text.
    # Clear embed_text_hash so the dedicated embed worker picks it up.
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            SET sn.embed_text_hash = null
            """,
            id=sn_id,
        )

    # Async counter bump — live progress visibility for ``sn status``
    bump_sn_run_counter(run_id, "names_enriched")

    return new_stage


@retry_on_deadlock()
def release_generate_docs_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release generate_docs claims IFF token matches.

    Called after successful persist to clear any nodes whose claim state
    was not already cleared inside :func:`persist_generated_docs`
    (e.g., skipped items in a multi-item batch).

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (n:StandardName {id: sid})
            WHERE n.claim_token = $token
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_generate_docs_claims: %d/%d released (token=%s) — "
            "remainder already cleared or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_generate_docs_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release generate_docs claims after LLM or processing failure.

    Token-verified: only clears ``claim_token`` and ``claimed_at`` on nodes
    where ``claim_token = $token``.  Does NOT change ``docs_stage`` (it
    stays at ``'pending'`` since the claim never transitioned it).

    Returns the count of nodes released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
            SET sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_generate_docs_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


# =============================================================================
# refine_docs — claim / persist / release (DocsRevision snapshot architecture)
# =============================================================================
# Stage gate: docs_stage='reviewed' AND reviewer_score_docs < min_score
#             AND docs_chain_length < rotation_cap
# Fundamentally different from refine_name: docs refine is IN-PLACE on the
# existing SN node.  The OLD docs are snapshotted into a DocsRevision node
# linked via DOCS_REVISION_OF before the SN is updated.


@retry_on_deadlock()
def claim_refine_docs_batch(
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    edits_only: bool = False,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for docs refinement.

    Eligibility: ``docs_stage = 'reviewed'`` AND
    ``reviewer_score_docs < min_score`` AND
    ``docs_chain_length < rotation_cap``.

    The claim atomically transitions ``docs_stage`` from ``'reviewed'``
    to ``'refining'`` via :func:`_claim_sn_atomic`.

    After claiming, each item is enriched with DOCS_REVISION_OF chain
    history via :func:`~imas_codex.standard_names.chain_history.docs_chain_history`.

    Returns claimed items as dicts with docs_chain_history appended.
    """
    from imas_codex.standard_names.chain_history import docs_chain_history

    # Name-form vetted gate (same invariant as generate_docs). CURATIVE-SCOPE
    # EXCEPTION: dropped under a scope_run_id so curatively-reset derived-leaf
    # families can refine (mirrors claim_generate_docs_batch/claim_review_docs_batch).
    # Also dropped under edits_only — staged docs edits on catalog-imported
    # names carry no reviewer_score_name and would otherwise never refine.
    score_gate = (
        ""
        if (scope_run_id or edits_only)
        else (
            " AND (sn.reviewer_score_name IS NOT NULL"
            " OR (coalesce(sn.origin, '') = 'derived'"
            "     AND EXISTS { MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)"
            "       WHERE NOT coalesce(kid.name_stage, '') IN"
            "       ['superseded', 'exhausted', 'contested'] }))"
        )
    )
    where = (
        "sn.docs_stage = 'reviewed'"
        " AND sn.reviewer_score_docs IS NOT NULL"
        " AND sn.reviewer_score_docs < $min_score"
        " AND coalesce(sn.docs_chain_length, 0) < $rotation_cap"
        " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
        + score_gate
    )
    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params={"min_score": min_score, "rotation_cap": rotation_cap},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            # description, documentation, kind, physics_domain already in base
            # readback — listing them again raises Neo4j 42N38 (duplicate
            # return item name).  Only add fields not present in
            # _claim_sn_atomic's fixed RETURN list.
            ", sn.tags AS tags"
            ", sn.docs_stage AS docs_stage"
            ", sn.docs_chain_length AS docs_chain_length"
            ", sn.docs_model AS docs_model"
            ", sn.docs_generated_at AS docs_generated_at"
            ", sn.reviewer_score_docs AS reviewer_score_docs"
            ", sn.reviewer_comments_per_dim_docs"
            "     AS reviewer_comments_per_dim_docs"
            ", sn.reviewer_comments_docs AS reviewer_comments_docs"
            ", sn.edit_mode AS edit_mode"
            ", sn.name_hint AS name_hint"
            ", sn.docs_hint AS docs_hint"
            ", sn.edit_reason AS edit_reason"
            ", sn.edit_origin AS edit_origin"
        ),
        stage_field="docs_stage",
        to_stage="refining",
        domain=domain,
        scope_run_id=scope_run_id,
        edits_only=edits_only,
    )

    # Drop claim-race losers BEFORE the (paid) refine LLM call. The claim
    # transitions docs_stage 'reviewed'→'refining', but the shared seed's
    # MATCH-bound row + lock-serialised SET still let concurrent replicas
    # each transition the same node; only the claim_token winner truly owns
    # it. See _verify_docs_claim_winners.
    items = _verify_docs_claim_winners(items, eligible_stage="refining")

    # Enrich each claimed item with its DOCS_REVISION_OF chain history and build
    # a unified prior_docs_reviews list that also includes the current node's own
    # reviewer feedback.  Without this, docs cycled >1 time lose intermediate
    # reviewer verdict when the next refine-docs prompt is rendered.
    for item in items:
        item["docs_chain_history"] = docs_chain_history(item["id"], limit=5)
        raw = item.get("reviewer_comments_per_dim_docs") or "{}"
        try:
            per_dim: dict = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (ValueError, TypeError):
            per_dim = {}
        current_review: dict[str, Any] = {
            "model": item.get("docs_model", "unknown"),
            "reviewer_score": item.get("reviewer_score_docs"),
            "reviewer_comments_per_dim": per_dim,
            "documentation": item.get("documentation", ""),
        }
        item["prior_docs_reviews"] = item["docs_chain_history"] + [current_review]

    logger.debug(
        "claim_refine_docs_batch: claimed %d",
        len(items),
    )
    return items


# =============================================================================
# Persist — refine_docs (snapshot current docs → DocsRevision, update SN)
# =============================================================================


@retry_on_deadlock()
def persist_refined_docs(
    *,
    sn_id: str,
    claim_token: str,
    description: str,
    documentation: str,
    model: str,
    current_description: str,
    current_documentation: str,
    current_model: str | None = None,
    current_generated_at: str | None = None,
    reviewer_score_to_snapshot: float | None = None,
    reviewer_comments_to_snapshot: str | None = None,
    reviewer_comments_per_dim_to_snapshot: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist a refined docs revision with DocsRevision snapshot.

    Single-transaction:

    1. Verify ``claim_token`` + ``docs_stage = 'refining'``.
    2. CREATE ``DocsRevision`` snapshot of CURRENT state (before this refine):
       ``id = "{sn_id}#rev-{docs_chain_length}"`` (deterministic key).
    3. CREATE ``(sn)-[:DOCS_REVISION_OF]->(rev)`` edge.
    4. SET new description/documentation on SN, advance chain, clear claim.
    5. Clear ``reviewer_*_docs`` fields on the SN (new docs need fresh review).

    Returns ``{"docs_chain_length": <new>, "revision_id": <id>}``.
    Returns ``{"docs_chain_length": -1, "revision_id": ""}`` on token/stage
    mismatch (no-op).
    """
    # Repair any JSON-escape-mangled LaTeX control chars before persisting.
    description = _sanitize_doc_text(description)
    documentation = _sanitize_doc_text(documentation)
    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                result = list(
                    tx.run(
                        """
                        // 1. Match + verify
                        MATCH (sn:StandardName {id: $sn_id})
                        WHERE sn.claim_token = $token
                          AND sn.docs_stage = 'refining'
                          AND sn.name_stage = 'accepted'
                        WITH sn, coalesce(sn.docs_chain_length, 0) AS cur_chain

                        // 2. Create DocsRevision snapshot (deterministic id)
                        WITH sn, cur_chain,
                             $sn_id + '#rev-' + toString(cur_chain) AS rev_id
                        MERGE (rev:DocsRevision {id: rev_id})
                        ON CREATE SET
                          rev.sn_id                          = $sn_id,
                          rev.revision_number                = cur_chain,
                          rev.description                    = $cur_desc,
                          rev.documentation                  = $cur_doc,
                          rev.model                          = $cur_model,
                          rev.generated_at                   = $cur_gen_at,
                          rev.reviewer_score_docs            = $snap_score,
                          rev.reviewer_comments_docs         = $snap_comments,
                          rev.reviewer_comments_per_dim_docs = $snap_comments_dim,
                          rev.created_at                     = datetime()

                        // 3. Link SN → revision
                        WITH sn, rev, cur_chain
                        MERGE (sn)-[:DOCS_REVISION_OF]->(rev)

                        // 4. Update SN with new docs + advance chain
                        WITH sn, rev, cur_chain
                        SET sn.description       = $new_desc,
                            sn.documentation     = $new_doc,
                            sn.docs_stage        = 'drafted',
                            sn.docs_chain_length = cur_chain + 1,
                            sn.docs_model        = $model,
                            sn.docs_generated_at = datetime(),
                            sn.refine_docs_count = coalesce(sn.refine_docs_count, 0) + 1,
                            sn.claim_token       = null,
                            sn.claimed_at        = null,
                            // 5. Clear reviewer_*_docs — new docs need fresh review
                            sn.reviewer_score_docs            = null,
                            sn.reviewer_scores_docs           = null,
                            sn.reviewer_comments_per_dim_docs = null,
                            sn.reviewer_comments_docs         = null,
                            sn.reviewer_model_docs            = null,
                            sn.reviewed_docs_at               = null

                        RETURN cur_chain + 1 AS docs_chain_length,
                               rev.id        AS revision_id
                        """,
                        sn_id=sn_id,
                        token=claim_token,
                        cur_desc=current_description or "",
                        cur_doc=current_documentation or "",
                        cur_model=current_model,
                        cur_gen_at=current_generated_at,
                        snap_score=reviewer_score_to_snapshot,
                        snap_comments=reviewer_comments_to_snapshot,
                        snap_comments_dim=reviewer_comments_per_dim_to_snapshot,
                        new_desc=description,
                        new_doc=documentation,
                        model=model,
                    )
                )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    if result:
        row = dict(result[0])
        logger.debug(
            "persist_refined_docs: %s (chain_length=%d, rev=%s)",
            sn_id,
            row["docs_chain_length"],
            row["revision_id"],
        )

        # Re-embed: refined docs change the description, so update the
        # embedding to reflect the latest "name — description" text.
        # Clear embed_text_hash so the dedicated embed worker picks it up.
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                SET sn.embed_text_hash = null
                """,
                id=sn_id,
            )

        # Async counter bump — live progress visibility for ``sn status``
        bump_sn_run_counter(run_id, "names_regenerated")
        _record_persist_outcome(run_id, "refine_docs", persisted=True)
        return row
    _record_persist_outcome(run_id, "refine_docs", persisted=False)
    logger.debug(
        "persist_refined_docs: no-op for %s (token/stage mismatch)",
        sn_id,
    )
    return {"docs_chain_length": -1, "revision_id": ""}


@retry_on_deadlock()
def reset_standard_name_docs(
    *,
    sn_ids: list[str],
    run_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Reset the docs pipeline for an explicit set of StandardNames.

    The family-harmonization mark step: each eligible member has its
    current docs snapshotted to a :class:`DocsRevision` (same deterministic
    ``{id}#rev-{chain}`` scheme as :func:`persist_refined_docs`, advancing
    ``docs_chain_length`` so revision ids never collide), then its
    ``docs_stage`` is reset to ``'pending'`` and all reviewer-docs state is
    cleared so the generate_docs → review_docs gate re-runs from scratch.
    ``description``/``documentation`` are left in place — the generate_docs
    prompt treats them as untrusted scaffolding and rewrites them.

    Eligible: ``name_stage='accepted'`` AND ``docs_stage`` in
    accepted/exhausted/reviewed/drafted. Names already ``pending`` (or not
    in the docs pipeline) are skipped — they regenerate anyway.

    Args:
        sn_ids: Explicit StandardName ids to reset.
        run_id: When set, stamped onto ``sn.run_id`` so a subsequent
            pool run with ``scope_run_id=run_id`` claims ONLY these names
            (leaving the global docs-pending backlog untouched).
        dry_run: Report the eligible count without writing.

    Returns:
        ``{"eligible": n, "reset": n_or_0}``.
    """
    if not sn_ids:
        return {"eligible": 0, "reset": 0}

    eligibility = """
        UNWIND $sn_ids AS sid
        MATCH (sn:StandardName {id: sid})
        WHERE sn.name_stage = 'accepted'
          AND coalesce(sn.docs_stage, '')
              IN ['accepted', 'exhausted', 'reviewed', 'drafted']
    """
    with GraphClient() as gc:
        rows = gc.query(eligibility + " RETURN count(sn) AS eligible", sn_ids=sn_ids)
        eligible: int = rows[0]["eligible"] if rows else 0
        if dry_run or eligible == 0:
            return {"eligible": eligible, "reset": 0}

        result = gc.query(
            eligibility
            + """
            WITH sn, coalesce(sn.docs_chain_length, 0) AS cur_chain,
                 (sn.documentation IS NOT NULL AND sn.documentation <> '')
                 AS has_docs

            // Snapshot current docs (when present) exactly like a refine
            // pass would, so the pre-harmonization text stays recoverable.
            FOREACH (_ IN CASE WHEN has_docs THEN [1] ELSE [] END |
              MERGE (rev:DocsRevision {id: sn.id + '#rev-' + toString(cur_chain)})
              ON CREATE SET
                rev.sn_id                          = sn.id,
                rev.revision_number                = cur_chain,
                rev.description                    = coalesce(sn.description, ''),
                rev.documentation                  = coalesce(sn.documentation, ''),
                rev.model                          = sn.docs_model,
                rev.generated_at                   = sn.docs_generated_at,
                rev.reviewer_score_docs            = sn.reviewer_score_docs,
                rev.reviewer_comments_docs         = sn.reviewer_comments_docs,
                rev.reviewer_comments_per_dim_docs = sn.reviewer_comments_per_dim_docs,
                rev.created_at                     = datetime()
              MERGE (sn)-[:DOCS_REVISION_OF]->(rev)
            )

            SET sn.docs_stage        = 'pending',
                sn.docs_chain_length =
                    CASE WHEN has_docs THEN cur_chain + 1 ELSE cur_chain END,
                sn.docs_model        = null,
                sn.docs_generated_at = null,
                sn.claim_token       = null,
                sn.claimed_at        = null,
                sn.reviewer_score_docs            = null,
                sn.reviewer_scores_docs           = null,
                sn.reviewer_comments_per_dim_docs = null,
                sn.reviewer_comments_docs         = null,
                sn.reviewer_model_docs            = null,
                sn.reviewed_docs_at               = null,
                sn.embed_text_hash                = null,
                sn.run_id = CASE WHEN $run_id IS NULL
                                 THEN sn.run_id ELSE $run_id END
            RETURN count(sn) AS reset
            """,
            sn_ids=sn_ids,
            run_id=run_id,
        )

        # Members already docs-pending (stubs) need no reset — but they DO
        # need the scope stamp so they join the same scoped pool run.
        if run_id:
            gc.query(
                """
                UNWIND $sn_ids AS sid
                MATCH (sn:StandardName {id: sid})
                WHERE sn.name_stage = 'accepted'
                  AND coalesce(sn.docs_stage, 'pending') = 'pending'
                SET sn.run_id = $run_id
                """,
                sn_ids=sn_ids,
                run_id=run_id,
            )
    reset: int = result[0]["reset"] if result else 0
    logger.info(
        "reset_standard_name_docs: %d/%d reset to docs pending (run_id=%s)",
        reset,
        len(sn_ids),
        run_id,
    )
    return {"eligible": eligible, "reset": reset}


@retry_on_deadlock()
def stamp_harmonized_families(
    families: list[dict[str, Any]],
) -> int:
    """Stamp the §5 idempotency scalars on fully docs-accepted families.

    For each family dict (``{"parent": id|None, "members": [ids],
    "signature": str}``) whose live members are ALL ``docs_stage='accepted'``,
    set ``harmonized_at`` + ``harmonized_group_signature`` on every member
    AND on the shared parent (the harmonize worklist reads the stored
    signature off the parent node). Families with any non-accepted member
    are skipped (stamp after the next rotation).

    Returns the number of families stamped.
    """
    stamped = 0
    with GraphClient() as gc:
        for fam in families:
            member_ids = fam.get("members") or []
            signature = fam.get("signature") or ""
            if not member_ids or not signature:
                continue
            rows = gc.query(
                """
                UNWIND $ids AS sid
                MATCH (sn:StandardName {id: sid})
                WITH collect(sn) AS members,
                     sum(CASE WHEN sn.docs_stage = 'accepted' THEN 0 ELSE 1 END)
                     AS not_accepted
                WHERE not_accepted = 0 AND size(members) = size($ids)
                FOREACH (sn IN members |
                  SET sn.harmonized_at = datetime(),
                      sn.harmonized_group_signature = $sig
                )
                RETURN size(members) AS stamped_members
                """,
                ids=member_ids,
                sig=signature,
            )
            if rows and rows[0].get("stamped_members"):
                stamped += 1
                parent_id = fam.get("parent")
                if parent_id:
                    gc.query(
                        """
                        MATCH (p:StandardName {id: $pid})
                        SET p.harmonized_at = datetime(),
                            p.harmonized_group_signature = $sig
                        """,
                        pid=parent_id,
                        sig=signature,
                    )
    return stamped


@retry_on_deadlock()
def release_refine_docs_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release refine-docs claims, reverting ``docs_stage`` to ``'reviewed'``.

    Token-and-stage verified: only reverts nodes where
    ``claim_token = $token AND docs_stage = 'refining'``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
              AND sn.docs_stage = 'refining'
            SET sn.docs_stage  = 'reviewed',
                sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_refine_docs_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_refine_docs_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release refine-docs claims after LLM or processing failure.

    Token-and-stage verified: only reverts nodes where
    ``claim_token = $token AND docs_stage = 'refining'``.  Prevents
    late-release from clobbering an SN already swept by orphan recovery.

    Returns the count of nodes released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
              AND sn.docs_stage = 'refining'
            SET sn.docs_stage  = 'reviewed',
                sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_refine_docs_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


# =============================================================================
# Routing helper: desc-name similarity gate → REFINE_DOCS
# =============================================================================


@retry_on_deadlock()
def mark_for_refine_docs(
    sn_id: str,
    *,
    desc_name_similarity: float,
    claim_token: str,
    synthetic_score: float = 0.30,
) -> bool:
    """Route a derived parent to REFINE_DOCS by writing a synthetic docs-review score.

    Called from the REVIEW_NAME pre-step when ``origin='derived'`` AND
    ``desc_name_similarity < threshold``.  The node's description is
    considered misaligned with the name; we skip name scoring and instead
    push the item into the REFINE_DOCS pool.

    Mechanics:

    1. Verify ``claim_token`` so no other worker has claimed this node.
    2. Transition ``name_stage`` back to the pre-review state that allows
       the node to wait until ``name_stage='accepted'`` is later confirmed.
       Because ``seed_parent_sources`` already set ``name_stage='accepted'``
       for admitted parents, we restore that here so the node can still
       reach ``GENERATE_DOCS``.
    3. Set ``docs_stage='reviewed'`` with a synthetic ``reviewer_score_docs``
       below ``min_score`` so ``claim_refine_docs_batch`` picks it up.
    4. Store ``desc_name_similarity`` on the node for audit / reviewer context.
    5. Clear the name-review claim token.

    Returns ``True`` if the node was updated, ``False`` if the claim_token
    no longer matches (race condition — caller should skip).

    Args:
        sn_id: StandardName node id.
        desc_name_similarity: Cosine similarity value to store on the node.
        claim_token: Token set by the REVIEW_NAME claim step; used to
            confirm ownership before writing.
        synthetic_score: ``reviewer_score_docs`` value written (default
            ``0.30``, well below any ``min_score``).
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.claim_token = $token
            SET sn.desc_name_similarity  = $sim,
                sn.docs_stage            = 'reviewed',
                sn.reviewer_score_docs   = $synthetic_score,
                sn.reviewer_model_docs   = '(desc_name_similarity_gate)',
                sn.reviewer_comments_docs =
                    '(desc-name similarity ' + toString($sim) +
                    ' below threshold; routed to REFINE_DOCS)',
                sn.name_stage            = 'accepted',
                sn.claimed_at            = null,
                sn.claim_token           = null
            RETURN count(sn) AS updated
            """,
            id=sn_id,
            token=claim_token,
            sim=desc_name_similarity,
            synthetic_score=synthetic_score,
        )
    updated = result[0]["updated"] if result else 0
    logger.debug(
        "mark_for_refine_docs: %s sim=%.3f updated=%d",
        sn_id,
        desc_name_similarity,
        updated,
    )
    return bool(updated)


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
            SET n.name_stage = CASE WHEN n.name_stage = 'refining'
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
) -> dict[str, int]:
    """Return pending-work counts for all six worker pools in one query.

    The six predicates mirror the eligibility criteria in the
    corresponding ``claim_*_seed_and_expand`` functions but do NOT
    filter on ``claimed_at`` — the counts reflect total eligible work,
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

    cypher = """
    CALL { MATCH (s:StandardNameSource {status: 'extracted'})
           WHERE coalesce(s.attempt_count, 0) < $max_compose_attempts
           RETURN count(s) AS generate_name }
    CALL { MATCH (sn:StandardName {name_stage: 'drafted'})
           RETURN count(sn) AS review_name }
    CALL { MATCH (sn:StandardName {name_stage: 'reviewed'})
           WHERE sn.reviewer_score_name < $min_score
             AND coalesce(sn.chain_length, 0) < $rotation_cap
             AND coalesce(sn.origin, '') <> 'derived'
           RETURN count(sn) AS refine_name }
    CALL { MATCH (sn:StandardName {name_stage: 'accepted', docs_stage: 'pending'})
           RETURN count(sn) AS generate_docs }
    CALL { MATCH (sn:StandardName {docs_stage: 'drafted'})
           RETURN count(sn) AS review_docs }
    CALL { MATCH (sn:StandardName {docs_stage: 'reviewed'})
           WHERE sn.reviewer_score_docs < $min_score
             AND coalesce(sn.docs_chain_length, 0) < $rotation_cap
           RETURN count(sn) AS refine_docs }
    RETURN generate_name, review_name, refine_name,
           generate_docs, review_docs, refine_docs
    """
    with GraphClient() as gc:
        rows = list(
            gc.query(
                cypher,
                min_score=min_score,
                rotation_cap=rotation_cap,
                max_compose_attempts=_MAX_COMPOSE_CLAIM_ATTEMPTS,
            )
        )

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
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE (sn.claim_token = $token OR sn.claim_token IS NULL)
              AND sn.origin = 'derived'
            SET sn.description        = $description,
                sn.embedding          = $embedding,
                sn.embedded_at        = CASE WHEN $embedding IS NULL
                                             THEN sn.embedded_at
                                             ELSE datetime() END,
                sn.parent_enriched_at = datetime(),
                sn.parent_enrich_model = $model,
                // Structural acceptance — skip REVIEW_NAME (see docstring). The
                // name is a deterministic grammar peel of the children, never
                // reviewed/refined, so it carries NO reviewer_score_name (scoring
                // a name we never change is meaningless). ``reviewed_name_at`` is
                // the "name finalised, docs may proceed" signal; the docs gates
                // admit derived parents via origin, not a score.
                sn.name_stage = 'accepted',
                sn.reviewer_model_name = 'structural-inheritance',
                sn.reviewed_name_at = coalesce(sn.reviewed_name_at, datetime()),
                sn.docs_stage = coalesce(sn.docs_stage, 'pending'),
                sn.validation_status = coalesce(sn.validation_status, 'valid'),
                sn.needs_composition = null,
                sn.claim_token        = null,
                sn.claimed_at         = null
            RETURN sn.name_stage AS name_stage
            """,
            id=sn_id,
            token=claim_token,
            description=description,
            embedding=embedding,
            model=model,
        )
        if not rows:
            logger.debug(
                "persist_enriched_parent: %s no longer matched (concurrent "
                "winner or not a derived parent) — no-op",
                sn_id,
            )
            return ""
        # Mirror onto the parent's source row so consolidation/export agree.
        gc.query(
            """
            MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(
                sn:StandardName {id: $id})
            SET sns.description = $description
            """,
            id=sn_id,
            description=description,
        )

    new_stage: str = rows[0]["name_stage"]
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
            WHERE sn.origin = 'derived'
              AND sn.name_stage IN ['drafted', 'reviewed', 'exhausted', 'refining']
              AND sn.description IS NOT NULL
              AND sn.description <> $ph
            SET sn.name_stage = 'accepted',
                sn.reviewer_model_name = 'structural-inheritance',
                sn.reviewed_name_at = coalesce(sn.reviewed_name_at, datetime()),
                sn.docs_stage = coalesce(sn.docs_stage, 'pending'),
                sn.chain_length = coalesce(sn.chain_length, 0),
                sn.claim_token = null,
                sn.claimed_at = null
            RETURN count(sn) AS promoted
            """,
            ph=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
        promoted: int = rows[0]["promoted"] if rows else 0
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
            SET sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    return result[0]["released"] if result else 0

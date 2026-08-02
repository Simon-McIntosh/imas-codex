"""Graph operations for the DD build pipeline state machine.

Claim/complete/has_pending functions following the standard discovery
pattern from ``imas_codex.discovery.base.claims``.

IMASNode lifecycle::

    built → enriched → refined → embedded → classified

DDVersion lifecycle::

    extracted → built

Workers claim batches via ``claimed_at`` + ``claim_token`` two-step
verify, process them, then advance the status and clear the claim.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Collection

from imas_codex.core.node_categories import EMBEDDABLE_CATEGORIES, ENRICHABLE_CATEGORIES
from imas_codex.discovery.base.claims import (
    DEFAULT_CLAIM_TIMEOUT_SECONDS,
    retry_on_deadlock,
)
from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

CLAIM_TIMEOUT_SECONDS = DEFAULT_CLAIM_TIMEOUT_SECONDS


# =============================================================================
# IMASNode — enrichment claims (built → enriched)
# =============================================================================


@retry_on_deadlock()
def claim_paths_for_enrichment(
    limit: int = 50,
    *,
    ids_filter: set[str] | None = None,
) -> list[dict]:
    """Claim built IMASNodes for LLM enrichment.

    Returns list of dicts with path metadata needed by the enrichment
    pipeline.  Uses the standard claim_token two-step pattern.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"

    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {
        "status": "built",
        "cutoff": cutoff,
        "limit": limit,
        "token": token,
    }
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    with GraphClient() as gc:
        # Claim with shallow paths first, randomizing peers at the same depth.
        gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
              AND (p.claimed_at IS NULL
                   OR p.claimed_at < datetime() - duration($cutoff))
              {ids_clause}
            WITH p
            ORDER BY size(split(p.id, '/')) ASC, rand()
            LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            categories=list(ENRICHABLE_CATEGORIES),
            **params,
        )

        # Read back only paths fenced by this claim token.
        result = gc.query(
            """
            MATCH (p:IMASNode {claim_token: $token})
            RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
                   p.data_type AS data_type, p.ids AS ids,
                   p.cocos_transformation_type AS cocos_transformation_type,
                   p.enrichment_hash AS enrichment_hash
            """,
            token=token,
        )
        claimed = list(result)
        logger.debug(
            "claim_paths_for_enrichment: requested %d, won %d",
            limit,
            len(claimed),
        )
        return claimed


def mark_paths_enriched(updates: list[dict]) -> int:
    """Mark enriched paths: set status=enriched, clear claimed_at.

    Each update dict must have at minimum ``id``.  Additional fields
    (description, keywords, enrichment_hash, etc.) are set on the node.
    """
    if not updates:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $updates AS item
            MATCH (p:IMASNode {id: item.id})
            SET p.status = 'enriched',
                p.description = coalesce(item.description, p.description),
                p.keywords = coalesce(item.keywords, p.keywords),
                p.enrichment_hash = coalesce(item.enrichment_hash, p.enrichment_hash),
                p.enrichment_model = coalesce(item.enrichment_model, p.enrichment_model),
                p.enrichment_source = coalesce(item.enrichment_source, p.enrichment_source),
                p.enrich_llm_cost = coalesce(item.enrich_llm_cost, p.enrich_llm_cost),
                p.physics_domain = coalesce(item.physics_domain, p.physics_domain),
                p.enriched_at = datetime(),
                p.claimed_at = null,
                p.claim_token = null
            RETURN count(p) AS updated
            """,
            updates=updates,
        )
        count = result[0]["updated"] if result else 0
        return count


def release_enrichment_claims(path_ids: list[str]) -> None:
    """Release enrichment claims on error (clear claimed_at)."""
    if not path_ids:
        return
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:IMASNode {id: pid})
            WHERE p.claimed_at IS NOT NULL
            SET p.claimed_at = null, p.claim_token = null
            """,
            ids=path_ids,
        )


def has_pending_enrichment(*, ids_filter: set[str] | None = None) -> bool:
    """Check if there are built IMASNodes awaiting enrichment.

    Counts ALL built nodes — both unclaimed and actively claimed.
    This prevents premature phase completion when workers still have
    claimed batches in flight.
    """
    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {"status": "built"}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
              {ids_clause}
            RETURN count(p) AS pending
            """,
            categories=list(ENRICHABLE_CATEGORIES),
            **params,
        )
        return result[0]["pending"] > 0 if result else False


# =============================================================================
# IMASNode — refinement claims (enriched → refined)
# =============================================================================


@retry_on_deadlock()
def claim_paths_for_refinement(
    limit: int = 50,
    *,
    ids_filter: set[str] | None = None,
) -> list[dict]:
    """Claim enriched IMASNodes for Pass 2 refinement.

    Only claims nodes whose enrichable siblings are ALL past ``built``
    status (sibling-readiness barrier).  Uses the standard claim_token
    two-step pattern.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"

    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {
        "status": "enriched",
        "cutoff": cutoff,
        "limit": limit,
        "token": token,
    }
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    with GraphClient() as gc:
        # Claim enriched nodes only after their sibling context is ready.
        gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
              AND (p.claimed_at IS NULL
                   OR p.claimed_at < datetime() - duration($cutoff))
              {ids_clause}
              AND NOT EXISTS {{
                MATCH (p)-[:HAS_PARENT]->(parent)<-[:HAS_PARENT]-(sib:IMASNode)
                WHERE sib.node_category IN $categories
                  AND sib.status = 'built'
              }}
            WITH p
            ORDER BY rand()
            LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            categories=list(ENRICHABLE_CATEGORIES),
            **params,
        )

        # Read back only paths fenced by this claim token.
        result = gc.query(
            """
            MATCH (p:IMASNode {claim_token: $token})
            RETURN p.id AS id, p.name AS name, p.description AS description,
                   p.keywords AS keywords, p.data_type AS data_type,
                   p.ids AS ids, p.unit AS unit,
                   p.enrichment_source AS enrichment_source,
                   p.refinement_hash AS refinement_hash
            """,
            token=token,
        )
        claimed = list(result)
        logger.debug(
            "claim_paths_for_refinement: requested %d, won %d",
            limit,
            len(claimed),
        )
        return claimed


def mark_paths_refined(updates: list[dict]) -> int:
    """Mark refined paths: set status=refined, write refinement_hash, refined_at.

    Each update dict must have ``id`` and optionally ``description``,
    ``keywords``, ``refinement_hash``.
    """
    if not updates:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $updates AS item
            MATCH (p:IMASNode {id: item.id})
            SET p.status = 'refined',
                p.description = coalesce(item.description, p.description),
                p.keywords = coalesce(item.keywords, p.keywords),
                p.refinement_hash = coalesce(item.refinement_hash, p.refinement_hash),
                p.refine_llm_cost = coalesce(item.refine_llm_cost, p.refine_llm_cost),
                p.refined_at = datetime(),
                p.claimed_at = null,
                p.claim_token = null
            RETURN count(p) AS updated
            """,
            updates=updates,
        )
        count = result[0]["updated"] if result else 0
        return count


def release_refinement_claims(path_ids: list[str]) -> None:
    """Release refinement claims on error."""
    if not path_ids:
        return
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:IMASNode {id: pid})
            WHERE p.claimed_at IS NOT NULL
            SET p.claimed_at = null, p.claim_token = null
            """,
            ids=path_ids,
        )


def has_pending_refinement(*, ids_filter: set[str] | None = None) -> bool:
    """Check if there are enriched IMASNodes awaiting refinement.

    Counts ALL enriched nodes — both unclaimed and actively claimed.
    """
    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {"status": "enriched"}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
              {ids_clause}
            RETURN count(p) AS pending
            """,
            categories=list(ENRICHABLE_CATEGORIES),
            **params,
        )
        return result[0]["pending"] > 0 if result else False


# =============================================================================
# IMASNode — embedding claims (refined → embedded)
# =============================================================================


@retry_on_deadlock()
def claim_paths_for_embedding(limit: int = 500) -> list[dict]:
    """Claim refined IMASNodes for embedding generation.

    Returns list of dicts with path metadata needed by the embedding
    pipeline.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"

    with GraphClient() as gc:
        # Claim refined data nodes for embedding.
        # Prefer LLM-enriched nodes over template-enriched ones, but include
        # template-enriched nodes that have no embedding (e.g. after
        # --reset-to refined) to avoid an infinite spin where the worker
        # finds zero claimable paths.
        gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
              AND (p.enrichment_source <> 'template' OR p.embedding IS NULL)
              AND (p.claimed_at IS NULL
                   OR p.claimed_at < datetime() - duration($cutoff))
            WITH p
            ORDER BY rand()
            LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            status="refined",
            categories=list(EMBEDDABLE_CATEGORIES),
            cutoff=cutoff,
            limit=limit,
            token=token,
        )

        # Read back only paths fenced by this claim token.
        result = gc.query(
            """
            MATCH (p:IMASNode {claim_token: $token})
            RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
                   p.data_type AS data_type, p.ids AS ids, p.unit AS unit,
                   p.description AS description, p.keywords AS keywords,
                   p.cocos_transformation_type AS cocos_transformation_type,
                   p.physics_domain AS physics_domain,
                   p.node_type AS node_type, p.ndim AS ndim,
                   p.embedding_hash AS embedding_hash
            """,
            token=token,
        )
        claimed = list(result)
        logger.debug(
            "claim_paths_for_embedding: requested %d, won %d",
            limit,
            len(claimed),
        )
        return claimed


def mark_paths_embedded(path_ids: list[str]) -> int:
    """Mark embedded paths: set status=embedded, clear claimed_at."""
    if not path_ids:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:IMASNode {id: pid})
            SET p.status = 'embedded',
                p.claimed_at = null,
                p.claim_token = null
            RETURN count(p) AS updated
            """,
            ids=path_ids,
        )
        return result[0]["updated"] if result else 0


def release_embedding_claims(path_ids: list[str]) -> None:
    """Release embedding claims on error."""
    if not path_ids:
        return
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:IMASNode {id: pid})
            WHERE p.claimed_at IS NOT NULL
            SET p.claimed_at = null, p.claim_token = null
            """,
            ids=path_ids,
        )


def count_imas_nodes_by_status(
    *, node_categories: Collection[str] | None = None
) -> dict[str, int]:
    """Count IMASNode nodes grouped by status.

    Args:
        node_categories: Optional collection of node_category values to include.

    Returns dict mapping status → count, plus a 'total' key.
    """
    filter_cats = list(node_categories) if node_categories is not None else None
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.status IS NOT NULL
              AND ($filter_categories IS NULL OR p.node_category IN $filter_categories)
            RETURN p.status AS status, count(p) AS cnt
            """,
            filter_categories=filter_cats,
        )
        counts: dict[str, int] = {}
        total = 0
        for row in result:
            counts[row["status"]] = row["cnt"]
            total += row["cnt"]
        counts["total"] = total
        return counts


def has_pending_embedding() -> bool:
    """Check if there are refined IMASNodes awaiting embedding.

    Counts ALL refined nodes — both unclaimed and actively claimed.
    This prevents premature phase completion when workers still have
    claimed batches in flight (e.g. 4 embed workers each claim 500,
    the 4th goes idle with 1500 still being processed by the other 3).
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.status = $status
              AND p.node_category IN $categories
            RETURN count(p) AS pending
            """,
            status="refined",
            categories=list(EMBEDDABLE_CATEGORIES),
        )
        return result[0]["pending"] > 0 if result else False


# =============================================================================
# IMASNode — classification (embedded → classified)
# =============================================================================


def has_pending_classification(*, ids_filter: set[str] | None = None) -> bool:
    """Check if there are embedded IMASNodes awaiting classification."""
    ids_clause = "AND split(p.id, '/')[0] IN $ids_filter" if ids_filter else ""
    params: dict = {"status": "embedded"}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status = $status
              {ids_clause}
            RETURN count(p) > 0 AS pending
            """,
            **params,
        )
        return result[0]["pending"] if result else False


def mark_paths_classified(gc: GraphClient, path_ids: list[str]) -> int:
    """Mark classified paths: set status=classified, clear claimed_at."""
    if not path_ids:
        return 0

    result = gc.query(
        """
        UNWIND $ids AS pid
        MATCH (p:IMASNode {id: pid})
        SET p.status = 'classified',
            p.claimed_at = null,
            p.claim_token = null
        RETURN count(p) AS updated
        """,
        ids=path_ids,
    )
    return result[0]["updated"] if result else 0


# =============================================================================
# Orphan recovery
# =============================================================================


def reset_stale_imas_claims(*, timeout_seconds: int = CLAIM_TIMEOUT_SECONDS) -> int:
    """Release stale claims on IMASNode nodes."""
    cutoff = f"PT{timeout_seconds}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.claimed_at IS NOT NULL
              AND (p.claimed_at < datetime() - duration($cutoff)
                   OR p.claimed_at > datetime())
            SET p.claimed_at = null, p.claim_token = null
            RETURN count(p) AS reset_count
            """,
            cutoff=cutoff,
        )
        count = result[0]["reset_count"] if result else 0
        if count:
            logger.info("Released %d orphaned IMASNode claims", count)
        return count


# =============================================================================
# Status reset
# =============================================================================

# Fields to clear when resetting to each target status
_RESET_CLEAR_FIELDS: dict[str, list[str]] = {
    "built": [
        "description",
        "keywords",
        "enrichment_hash",
        "enrichment_model",
        "enrichment_source",
        "enriched_at",
        "enrich_llm_cost",
        "refinement_hash",
        "refined_at",
        "refine_llm_cost",
        "embedding",
        "embedding_hash",
        "embedded_at",
        "physics_domain",
        "domain_source",
        "domain_model",
        "domain_classified_at",
        "domain_input_hash",
    ],
    "enriched": [
        "refinement_hash",
        "refined_at",
        "refine_llm_cost",
        "embedding",
        "embedding_hash",
        "embedded_at",
        "domain_source",
        "domain_model",
        "domain_classified_at",
        "domain_input_hash",
    ],
    "refined": [
        "embedding",
        "embedding_hash",
        "embedded_at",
        "domain_source",
        "domain_model",
        "domain_classified_at",
        "domain_input_hash",
    ],
    "embedded": [
        "physics_domain",
        "domain_source",
        "domain_model",
        "domain_classified_at",
        "domain_input_hash",
    ],
}

# Statuses eligible for reset to each target
_RESET_SOURCE_STATUSES: dict[str, list[str]] = {
    "built": ["enriched", "refined", "embedded", "classified"],
    "enriched": ["refined", "embedded", "classified"],
    "refined": ["embedded", "classified"],
    "embedded": ["classified"],
}


def reset_imas_nodes(
    target_status: str,
    *,
    ids_filter: set[str] | None = None,
) -> int:
    """Reset IMASNode nodes to an earlier stage for reprocessing.

    Clears the stage's output properties (enrichment, refinement, embedding,
    or classification) and moves the node's status back, so the next build
    re-runs that LLM/embed stage on the *existing* node. Never deletes a node
    — the version-agnostic IMASNode identity and every provenance link it
    carries are preserved.

    Args:
        target_status: Target status (``built``, ``enriched``, ``refined``,
            or ``embedded``).
        ids_filter: Optional set of IDS names to limit the reset.

    Returns:
        Number of nodes affected.
    """
    if target_status not in _RESET_CLEAR_FIELDS:
        raise ValueError(
            f"Invalid target_status '{target_status}'. "
            f"Must be one of: {', '.join(_RESET_CLEAR_FIELDS)}"
        )

    clear_fields = _RESET_CLEAR_FIELDS[target_status]
    source_statuses = _RESET_SOURCE_STATUSES[target_status]

    ids_clause = "AND p.ids IN $ids_filter" if ids_filter else ""
    params: dict = {"source_statuses": source_statuses, "target": target_status}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    set_parts = [
        "p.status = $target",
        "p.claimed_at = null",
        "p.claim_token = null",
    ]
    for fld in clear_fields:
        set_parts.append(f"p.{fld} = null")
    set_clause = ", ".join(set_parts)

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.status IN $source_statuses
            {ids_clause}
            SET {set_clause}
            RETURN count(p) AS reset_count
            """,
            **params,
        )
        count = result[0]["reset_count"] if result else 0

    logger.info(
        "Reset %d IMASNode nodes to '%s'",
        count,
        target_status,
    )
    return count


def _exact_unit_correction_scope(
    path_ids: Collection[str] | None,
) -> frozenset[str] | None:
    """Normalize an optional exact-path scope without accepting one string."""
    if isinstance(path_ids, str):
        raise TypeError("path_ids must be a collection of exact paths, not a string")
    return None if path_ids is None else frozenset(path_ids)


def find_dd_unit_correction_drift(
    gc: GraphClient | None = None,
    *,
    path_ids: Collection[str] | None = None,
) -> list[dict[str, str]]:
    """Stored DD units that a ``correct_in_graph`` registry entry contradicts.

    Read-only. Asks :func:`imas_codex.units.resolve_dd_unit` — the same
    predicate the DD build uses — of every stored unit, and reports the paths
    whose stored value differs from what the registry says today. Only
    ``correct_in_graph`` entries can produce a difference: a suppression-only
    entry deliberately leaves the DD unit as declared so the standard-name
    mismatch axis keeps reporting it, and ``resolve_dd_unit`` returns those
    unchanged. When ``path_ids`` is provided, both the graph selector and a
    defensive in-process fence restrict results to those exact paths. An empty
    collection is a no-op, which makes operator-supplied bounded scopes safe.
    """
    from imas_codex.units import resolve_dd_unit

    scope = _exact_unit_correction_scope(path_ids)
    if scope is not None and not scope:
        return []

    own = gc is None
    client = GraphClient() if own else gc
    try:
        path_clause = "" if scope is None else "AND n.id IN $path_ids"
        params = {} if scope is None else {"path_ids": sorted(scope)}
        rows = client.query(
            f"""
            MATCH (n:IMASNode)
            WHERE n.unit IS NOT NULL AND n.unit <> ''
              {path_clause}
            RETURN n.id AS path, n.unit AS unit
            """,
            **params,
        )
    finally:
        if own:
            client.close()

    drift: list[dict[str, str]] = []
    for r in rows:
        if scope is not None and r["path"] not in scope:
            continue
        stored = r["unit"]
        expected = resolve_dd_unit(r["path"], stored)
        if expected is not None and expected != stored:
            drift.append({"path": r["path"], "stored": stored, "expected": expected})
    return drift


def reconcile_dd_unit_corrections(
    gc: GraphClient | None = None,
    *,
    path_ids: Collection[str] | None = None,
) -> dict[str, int]:
    """Apply registered self-contradiction unit corrections to the stored graph.

    ``resolve_dd_unit`` rewrites a DD-declared unit when the exceptions registry
    flags that path ``correct_in_graph`` — reserved for the case where the DD
    declares one documented quantity with two different dimensionalities, so
    there is no single DD answer to mirror and a standard name composed from the
    wrong facet inherits the wrong dimensionality. That rewrite runs at DD BUILD
    time only, so adding such an entry had no effect on paths already stored,
    and a full DD rebuild is far too expensive to run for a unit correction. A
    registry whose entries only take effect on a rebuild is a registry that
    silently does not work; this is the net.

    Rewrites both the ``unit`` scalar and the ``HAS_UNIT`` edge, replacing any
    existing unit edges with exactly one edge to the expected unit so the
    cardinality-one invariant holds even when legacy duplicate edges exist.
    Idempotent: once every stored unit equals what the registry resolves, the
    selector matches nothing. ``path_ids`` constrains both discovery and mutation
    to an exact caller-supplied set; an empty set performs no query.

    Returns dict: {checked, corrected}.
    """
    scope = _exact_unit_correction_scope(path_ids)
    if scope is not None and not scope:
        return {"checked": 0, "corrected": 0}

    own = gc is None
    client = GraphClient() if own else gc
    try:
        drift = find_dd_unit_correction_drift(client, path_ids=scope)
        if not drift:
            return {"checked": 0, "corrected": 0}

        client.query(
            """
            UNWIND $items AS item
            MATCH (n:IMASNode {id: item.path})
            OPTIONAL MATCH (n)-[r:HAS_UNIT]->(:Unit)
            DELETE r
            WITH DISTINCT n, item
            SET n.unit = item.expected
            MERGE (u:Unit {id: item.expected})
            MERGE (n)-[:HAS_UNIT]->(u)
            """,
            items=drift,
        )
    finally:
        if own:
            client.close()

    logger.warning(
        "reconcile_dd_unit_corrections: corrected %d DD node(s) whose stored "
        "unit contradicts a correct_in_graph registry entry (first few: %s)",
        len(drift),
        ", ".join(f"{d['path']} {d['stored']}→{d['expected']}" for d in drift[:5]),
    )
    return {"checked": len(drift), "corrected": len(drift)}

"""Cluster reconstruction, batching, and neighborhood context for review.

Data flow::

    graph (StandardName → IMASNode → SemanticCluster)
        → reconstruct_clusters_batch()        (batch cluster lookup)
        → group_into_review_batches()          (cluster × unit grouping)
        → build_neighborhood_context()         (semantic search for context)
        → enriched review batches              (ready for LLM review worker)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from imas_codex.standard_names.domain_priority import domain_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cluster reconstruction (single + batch)
# ---------------------------------------------------------------------------


def reconstruct_dominant_cluster(name_id: str, gc: Any) -> dict | None:
    """Query graph for the dominant cluster of a StandardName.

    Follows the path ``StandardName ← HAS_STANDARD_NAME ← IMASNode
    → MEMBER_OF → SemanticCluster`` and selects the cluster with the
    most source IMASNodes, breaking ties by scope (ids > domain > global)
    then by label.

    Args:
        name_id: StandardName id to look up.
        gc: An open :class:`~imas_codex.graph.client.GraphClient`.

    Returns:
        Cluster dict (cluster_id, cluster_label, cluster_description,
        scope, source_count) or ``None`` if no cluster found.
    """
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name_id})<-[:HAS_STANDARD_NAME]-(node:IMASNode)
              -[:MEMBER_OF]->(cluster:SemanticCluster)
        WITH cluster, count(DISTINCT node) AS source_count
        OPTIONAL MATCH (cluster)<-[:MEMBER_OF]-(other:IMASNode)
        WITH cluster, source_count, avg(1.0) AS sim
        RETURN cluster.id AS cluster_id,
               cluster.label AS cluster_label,
               cluster.description AS cluster_description,
               cluster.scope AS scope,
               source_count
        ORDER BY source_count DESC,
                 CASE cluster.scope WHEN 'ids' THEN 0 WHEN 'domain' THEN 1 ELSE 2 END,
                 cluster.label
        LIMIT 1
        """,
        name_id=name_id,
    )
    if not rows:
        return None

    row = rows[0]
    return {
        "cluster_id": row["cluster_id"],
        "cluster_label": row["cluster_label"],
        "cluster_description": row["cluster_description"],
        "scope": row["scope"],
        "source_count": row["source_count"],
    }


def reconstruct_clusters_batch(names: list[dict], gc: Any) -> dict[str, dict | None]:
    """Batch cluster reconstruction for multiple StandardNames.

    Single graph query returning all cluster info, grouped by name_id.
    For each name, selects the dominant cluster using priority:
    most source nodes → scope (ids > domain > global) → label tiebreak.

    Args:
        names: List of dicts with at least an ``id`` key.
        gc: An open :class:`~imas_codex.graph.client.GraphClient`.

    Returns:
        ``{name_id: cluster_dict_or_None}`` for every name in *names*.
    """
    name_ids = [n["id"] for n in names if n.get("id")]
    if not name_ids:
        return {}

    rows = gc.query(
        """
        UNWIND $name_ids AS nid
        MATCH (sn:StandardName {id: nid})<-[:HAS_STANDARD_NAME]-(node:IMASNode)
              -[:MEMBER_OF]->(cluster:SemanticCluster)
        WITH nid, cluster, count(DISTINCT node) AS source_count
        RETURN nid AS name_id,
               cluster.id AS cluster_id,
               cluster.label AS cluster_label,
               cluster.description AS cluster_description,
               cluster.scope AS scope,
               source_count
        ORDER BY nid, source_count DESC
        """,
        name_ids=name_ids,
    )

    # Group rows by name_id
    per_name: dict[str, list[dict]] = defaultdict(list)
    for row in rows or []:
        per_name[row["name_id"]].append(
            {
                "cluster_id": row["cluster_id"],
                "cluster_label": row["cluster_label"],
                "cluster_description": row["cluster_description"],
                "scope": row["scope"],
                "source_count": row["source_count"],
            }
        )

    # Select dominant cluster per name using the enrichment selector
    from imas_codex.standard_names.enrichment import select_primary_cluster

    result: dict[str, dict | None] = {}
    for nid in name_ids:
        candidates = per_name.get(nid, [])
        if not candidates:
            result[nid] = None
        elif len(candidates) == 1:
            result[nid] = candidates[0]
        else:
            # select_primary_cluster expects similarity_score key
            for c in candidates:
                c["similarity_score"] = c.get("source_count", 0)
            result[nid] = select_primary_cluster(candidates)

    return result


# ---------------------------------------------------------------------------
# Token estimation + batching
# ---------------------------------------------------------------------------


def estimate_name_tokens(name: dict) -> int:
    """Rough token estimate for a single name in an LLM prompt.

    Uses character-length heuristic (1 token ≈ 4 chars) plus 80 tokens
    of per-item scaffolding (JSON keys, formatting, etc.).
    """
    desc_len = len(name.get("description", "") or "")
    doc_len = len(name.get("documentation", "") or "")
    return (desc_len + doc_len) // 4 + 80


def group_into_review_batches(
    names: list[dict],
    clusters: dict[str, dict | None],
    *,
    max_batch_size: int = 25,
    token_budget: int = 8000,
) -> list[dict]:
    """Group names by (dominant_cluster_id × unit) for review batches.

    Within each group, fills batches respecting *token_budget* and
    *max_batch_size*.

    Args:
        names: StandardName dicts to batch.
        clusters: ``{name_id: cluster_dict_or_None}`` from
            :func:`reconstruct_clusters_batch`.
        max_batch_size: Hard cap on names per batch.
        token_budget: Soft cap on estimated tokens per batch.

    Returns:
        List of batch dicts, each with keys ``group_key``, ``names``,
        ``cluster``, ``estimated_tokens``.
    """
    if not names:
        return []

    # --- Build groups: (cluster_id / unit) ---------------------------------
    groups: dict[str, list[dict]] = defaultdict(list)

    for name in names:
        nid = name.get("id", "")
        cluster = clusters.get(nid)
        unit = name.get("unit") or "dimensionless"

        if cluster:
            group_key = f"{cluster['cluster_id']}/{unit}"
        else:
            domain = domain_key(name.get("physics_domain"))
            group_key = f"unclustered/{domain}/{unit}"

        groups[group_key].append(name)

    # --- Fill batches by token budget / max_batch_size ---------------------
    batches: list[dict] = []

    for group_key in sorted(groups):
        group_items = groups[group_key]
        # Determine the cluster for this group (from any member)
        sample_id = group_items[0].get("id", "")
        group_cluster = clusters.get(sample_id)

        current_batch: list[dict] = []
        current_tokens = 0

        for name in group_items:
            name_tokens = estimate_name_tokens(name)

            # Start new batch when adding this name would exceed limits
            if current_batch and (
                current_tokens + name_tokens > token_budget
                or len(current_batch) >= max_batch_size
            ):
                batches.append(
                    {
                        "group_key": group_key,
                        "names": current_batch,
                        "cluster": group_cluster,
                        "estimated_tokens": current_tokens,
                    }
                )
                current_batch = []
                current_tokens = 0

            current_batch.append(name)
            current_tokens += name_tokens

        # Flush remainder
        if current_batch:
            batches.append(
                {
                    "group_key": group_key,
                    "names": current_batch,
                    "cluster": group_cluster,
                    "estimated_tokens": current_tokens,
                }
            )

    logger.info(
        "Grouped %d names into %d review batches (max_batch_size=%d, token_budget=%d)",
        len(names),
        len(batches),
        max_batch_size,
        token_budget,
    )
    return batches


# ---------------------------------------------------------------------------
# Neighborhood context (semantic search)
# ---------------------------------------------------------------------------


def build_neighborhood_context(
    batch: dict,
    all_names: list[dict],
    k: int = 10,
) -> list[dict]:
    """Build semantic neighborhood context for a review batch.

    Searches for existing StandardNames near every candidate description and
    adds accepted names that share a candidate's unit while using a different
    physical base.  The latter makes split-base spellings visible even when
    semantic search does not place them near one another.

    Args:
        batch: Batch dict from :func:`group_into_review_batches`.
        all_names: Full StandardName catalog used for unit-anchored comparison.
        k: Base cap for semantic and unit-anchored comparator results.  The
            semantic cap grows to the batch size so every item receives a
            first-pass share.

    Returns:
        List of neighbor dicts with keys ``id``, ``description``, ``kind``,
        ``unit``, ``physical_base``, ``review_tier``, and
        ``comparison_basis``.  Empty list when no comparator is available.
    """
    try:
        from imas_codex.standard_names.search import search_standard_names_vector
    except Exception:
        logger.debug("search_standard_names_vector unavailable", exc_info=True)
        return []

    batch_names = batch.get("names", [])
    if not batch_names:
        return []
    batch_ids = {n.get("id", "") for n in batch_names}

    from imas_codex.standard_names.workers import _collect_nearby_name_comparators

    search_items = [
        {**name, "description": name.get("description") or name.get("id", "")}
        for name in batch_names
    ]

    def _search(query: str, *, k: int) -> list[dict]:
        try:
            results = search_standard_names_vector(query, k=k + len(batch_ids))
        except Exception:
            logger.debug("Neighborhood search failed for %r", query, exc_info=True)
            return []
        return [result for result in results if result.get("id", "") not in batch_ids]

    semantic = _collect_nearby_name_comparators(
        search_items,
        per_item_k=k,
        cap=k,
        search=_search,
    )
    for result in semantic:
        result.setdefault("comparison_basis", "semantic")

    unit_anchored: list[dict] = []
    unit_cap = max(k, len(batch_names))
    catalog = sorted(all_names, key=lambda name: name.get("id", ""))
    for candidate in batch_names:
        candidate_unit = candidate.get("unit")
        candidate_base = candidate.get("physical_base")
        if not candidate_unit or not candidate_base:
            continue
        for existing in catalog:
            existing_id = existing.get("id", "")
            existing_base = existing.get("physical_base")
            if (
                not existing_id
                or existing_id in batch_ids
                or existing.get("name_stage") != "accepted"
                or existing.get("unit") != candidate_unit
                or not existing_base
                or existing_base == candidate_base
            ):
                continue
            unit_anchored.append(
                {
                    **existing,
                    "comparison_basis": "same_unit_different_physical_base",
                }
            )
            if len(unit_anchored) >= unit_cap:
                break
        if len(unit_anchored) >= unit_cap:
            break

    # Unit-anchored comparators lead so a semantic cap cannot hide the signal.
    # Deduplicate both channels by catalog identity.
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in [*unit_anchored, *semantic]:
        rid = r.get("id", "")
        if rid and rid not in seen:
            seen.add(rid)
            deduped.append(r)

    # Return summary-only dicts
    return [
        {
            "id": r.get("id", ""),
            "description": r.get("description", ""),
            "kind": r.get("kind", ""),
            "unit": r.get("unit", ""),
            "physical_base": r.get("physical_base", ""),
            "review_tier": r.get("review_tier", ""),
            "comparison_basis": r.get("comparison_basis", "semantic"),
        }
        for r in deduped
    ]

"""Physics domain classifier for IMAS Data Dictionary paths.

Three-tier classification:
- Tier 1: LLM classification for physics-relevant paths (19K paths)
- Tier 2: Inheritance for error paths (from HAS_ERROR parent) and
  non-infrastructure metadata (from HAS_PARENT ancestor)
- Tier 3: None for infrastructure metadata (ids_properties/*, code/*)

Integrated as the CLASSIFY phase in the DD build pipeline, running after EMBED.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

#: Categories eligible for Tier 1 LLM classification
CLASSIFIABLE_CATEGORIES = frozenset(
    {
        "quantity",
        "structural",
        "representation",
        "geometry",
        "coordinate",
        "fit_artifact",
        "identifier",
    }
)

#: Default batch size for LLM classification
DEFAULT_BATCH_SIZE = 30

#: Service tag for cost attribution
SERVICE_TAG = "data-dictionary"

# =============================================================================
# Pydantic Response Models
# =============================================================================


class DomainClassification(BaseModel):
    """Single path domain assignment from LLM."""

    path_index: int = Field(description="1-based index matching input order")
    physics_domain: str = Field(description="Physics domain from closed vocabulary")


class DomainBatchResult(BaseModel):
    """Batch classification response from the LLM."""

    classifications: list[DomainClassification] = Field(
        description="Domain assignment for each path in the batch"
    )


# =============================================================================
# Main Orchestrator
# =============================================================================


async def classify_domains(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    model: str | None = None,
    on_cost: Callable[[float], None] | None = None,
    on_items: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run full three-tier domain classification.

    Runs Tier 3 first (infrastructure → None), Tier 2 (inheritance),
    then Tier 1 (LLM) for remaining classifiable paths.

    Args:
        gc: GraphClient instance.
        ids_filter: Optional set of IDS names to scope classification.
        force: Reclassify even if inputs unchanged (bypass hash check).
        dry_run: Log what would be classified without writing.
        model: Override model (defaults to get_model("language")).
        on_cost: Callback for cost accumulation.
        on_items: Callback for item count progress.

    Returns:
        Stats dict with counts per tier.
    """
    from imas_codex.settings import get_model as _get_model

    if model is None:
        model = _get_model("language")

    stats: dict[str, Any] = {
        "tier3_none": 0,
        "tier2_inherited": 0,
        "tier1_llm": 0,
        "tier1_general": 0,
        "tier1_retried": 0,
        "total_cost": 0.0,
        "model": model,
    }

    if dry_run:
        logger.info("dry_run=True — counting candidates without writing")

    # --- Tier 3: infrastructure metadata → physics_domain = null ---
    tier3_count = classify_tier3_none(gc, ids_filter=ids_filter, dry_run=dry_run)
    stats["tier3_none"] = tier3_count
    logger.info("Tier 3 (none/metadata): %d paths", tier3_count)

    # --- Tier 2: inherit from parent ---
    tier2_count = classify_tier2_inherit(
        gc, ids_filter=ids_filter, force=force, dry_run=dry_run
    )
    stats["tier2_inherited"] = tier2_count
    logger.info("Tier 2 (inherited): %d paths", tier2_count)

    # --- Tier 1: LLM classification for remaining paths ---
    paths = _query_unclassified_paths(gc, ids_filter=ids_filter, force=force)
    logger.info("Tier 1 candidates: %d paths", len(paths))

    if paths and not dry_run:
        results = await classify_tier1_llm(
            gc,
            paths,
            model=model,
            service=SERVICE_TAG,
            on_cost=on_cost,
            on_items=on_items,
        )

        # Write results to graph
        classified = 0
        general_count = 0
        for r in results:
            if r["physics_domain"] == "general":
                general_count += 1
            classified += 1

        _write_tier1_results(gc, results)
        stats["tier1_llm"] = classified
        stats["tier1_general"] = general_count
        stats["total_cost"] = sum(r.get("cost", 0.0) for r in results)

        logger.info(
            "Tier 1 (LLM): %d classified (%d general)", classified, general_count
        )

    if on_items:
        on_items(tier3_count + tier2_count + stats["tier1_llm"])

    return stats


# =============================================================================
# Tier 3: Infrastructure Metadata → None
# =============================================================================


def classify_tier3_none(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Set physics_domain = null for infrastructure metadata paths.

    Targets ids_properties/* and code/* subtrees that have no
    physics content.

    Returns:
        Count of nodes updated.
    """
    ids_clause = _ids_filter_clause(ids_filter, "n")

    if dry_run:
        cypher = f"""
        MATCH (n:IMASNode)
        WHERE (n.id CONTAINS '/ids_properties/' OR n.id CONTAINS '/code/')
          AND n.domain_source IS NULL
          {ids_clause}
        RETURN count(n) AS cnt
        """
        result = gc.query(cypher)
        return result[0]["cnt"] if result else 0

    cypher = f"""
    MATCH (n:IMASNode)
    WHERE (n.id CONTAINS '/ids_properties/' OR n.id CONTAINS '/code/')
      AND n.domain_source IS NULL
      {ids_clause}
    SET n.physics_domain = null,
        n.domain_source = 'none_metadata',
        n.domain_classified_at = datetime(),
        n.status = 'classified'
    RETURN count(n) AS updated
    """
    result = gc.query(cypher)
    return result[0]["updated"] if result else 0


# =============================================================================
# Tier 2: Inheritance
# =============================================================================


def classify_tier2_inherit(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Inherit physics_domain from parent for error and metadata paths.

    Two passes:
      a. Error paths: (parent)-[:HAS_ERROR]->(err) → err gets parent's domain
      b. Non-infra metadata: walk up HAS_PARENT to nearest classified ancestor

    Returns:
        Total count of nodes updated.
    """
    total = 0
    total += _inherit_from_error_parent(
        gc, ids_filter=ids_filter, force=force, dry_run=dry_run
    )
    total += _inherit_from_metadata_parent(
        gc, ids_filter=ids_filter, force=force, dry_run=dry_run
    )
    return total


def _inherit_from_error_parent(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Error paths inherit domain from their parent via HAS_ERROR."""
    ids_clause = _ids_filter_clause(ids_filter, "parent")
    needs_clause = (
        "AND (err.physics_domain IS NULL OR err.physics_domain = 'general')"
        if not force
        else ""
    )

    if dry_run:
        cypher = f"""
        MATCH (parent:IMASNode)-[:HAS_ERROR]->(err:IMASNode)
        WHERE parent.physics_domain IS NOT NULL
          AND parent.physics_domain <> 'general'
          {needs_clause}
          {ids_clause}
        RETURN count(err) AS cnt
        """
        result = gc.query(cypher)
        return result[0]["cnt"] if result else 0

    cypher = f"""
    MATCH (parent:IMASNode)-[:HAS_ERROR]->(err:IMASNode)
    WHERE parent.physics_domain IS NOT NULL
      AND parent.physics_domain <> 'general'
      {needs_clause}
      {ids_clause}
    SET err.physics_domain = parent.physics_domain,
        err.domain_source = 'inherited_from_parent',
        err.domain_classified_at = datetime(),
        err.status = 'classified'
    RETURN count(err) AS updated
    """
    result = gc.query(cypher)
    return result[0]["updated"] if result else 0


def _inherit_from_metadata_parent(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Non-infrastructure metadata inherits domain from HAS_PARENT ancestor."""
    ids_clause = _ids_filter_clause(ids_filter, "n")
    needs_clause = (
        "AND (n.physics_domain IS NULL OR n.physics_domain = 'general')"
        if not force
        else ""
    )

    if dry_run:
        cypher = f"""
        MATCH (n:IMASNode {{node_category: 'metadata'}})-[:HAS_PARENT]->(parent:IMASNode)
        WHERE NOT (n.id CONTAINS '/ids_properties/' OR n.id CONTAINS '/code/')
          AND parent.physics_domain IS NOT NULL
          AND parent.physics_domain <> 'general'
          {needs_clause}
          {ids_clause}
        RETURN count(n) AS cnt
        """
        result = gc.query(cypher)
        return result[0]["cnt"] if result else 0

    cypher = f"""
    MATCH (n:IMASNode {{node_category: 'metadata'}})-[:HAS_PARENT]->(parent:IMASNode)
    WHERE NOT (n.id CONTAINS '/ids_properties/' OR n.id CONTAINS '/code/')
      AND parent.physics_domain IS NOT NULL
      AND parent.physics_domain <> 'general'
      {needs_clause}
      {ids_clause}
    SET n.physics_domain = parent.physics_domain,
        n.domain_source = 'inherited_from_parent',
        n.domain_classified_at = datetime(),
        n.status = 'classified'
    RETURN count(n) AS updated
    """
    result = gc.query(cypher)
    return result[0]["updated"] if result else 0


# =============================================================================
# Tier 1: LLM Classification
# =============================================================================


async def classify_tier1_llm(
    gc: GraphClient,
    paths: list[dict[str, Any]],
    *,
    model: str,
    service: str = SERVICE_TAG,
    on_cost: Callable[[float], None] | None = None,
    on_items: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """LLM batch classification for physics paths.

    Batches paths by subtree for coherence, calls the LLM with
    domain_classifier prompt, validates responses, and retries
    paths that got "general" with expanded context.

    Args:
        gc: GraphClient instance.
        paths: List of path dicts from _query_unclassified_paths.
        model: LLM model identifier.
        service: Cost attribution service tag.
        on_cost: Callback for accumulated cost.
        on_items: Callback for completed item count.

    Returns:
        List of result dicts with keys:
        {id, physics_domain, domain_source, domain_model, cost}.
    """
    from imas_codex.core.physics_domain import PhysicsDomain

    valid_domains = {d.value for d in PhysicsDomain}

    # Gather context for all paths
    path_ids = [p["id"] for p in paths]
    contexts = gather_classification_context(gc, path_ids)

    # Build context lookup by id
    context_by_id = {c["id"]: c for c in contexts}

    # Build enriched path list with context
    enriched_paths = []
    for p in paths:
        ctx = context_by_id.get(p["id"], {})
        enriched_paths.append({**p, **ctx})

    # Batch by subtree
    batches = batch_by_subtree(enriched_paths, batch_size=DEFAULT_BATCH_SIZE)
    logger.info("Tier 1: %d paths in %d batches", len(paths), len(batches))

    all_results: list[dict[str, Any]] = []
    general_retry_candidates: list[dict[str, Any]] = []
    total_cost = 0.0

    for batch_idx, batch in enumerate(batches):
        batch_results, cost = await _classify_batch(
            batch, model=model, service=service, valid_domains=valid_domains
        )
        total_cost += cost

        for r in batch_results:
            if r["physics_domain"] == "general":
                general_retry_candidates.append(r)
            else:
                all_results.append(r)

        if on_cost:
            on_cost(cost)
        if on_items:
            on_items(len(batch))

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                "  batch %d/%d done (cost=$%.4f)",
                batch_idx + 1,
                len(batches),
                total_cost,
            )

    # --- Retry "general" paths with expanded context ---
    if general_retry_candidates:
        logger.info(
            "Retrying %d 'general' paths with expanded context",
            len(general_retry_candidates),
        )
        retry_ids = [r["id"] for r in general_retry_candidates]
        expanded = _gather_expanded_context(gc, retry_ids)

        retry_batches = batch_by_subtree(expanded, batch_size=DEFAULT_BATCH_SIZE)
        for batch in retry_batches:
            batch_results, cost = await _classify_batch(
                batch, model=model, service=service, valid_domains=valid_domains
            )
            total_cost += cost
            all_results.extend(batch_results)

            if on_cost:
                on_cost(cost)

    if on_cost:
        on_cost(0.0)  # signal done

    return all_results


async def _classify_batch(
    batch: list[dict[str, Any]],
    *,
    model: str,
    service: str,
    valid_domains: set[str],
) -> tuple[list[dict[str, Any]], float]:
    """Classify a single batch via LLM.

    Returns:
        Tuple of (results list, cost).
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt

    # Render prompt with batch context
    system_prompt = render_prompt("imas/domain_classifier", {"paths": batch})
    user_prompt = _format_batch_user_prompt(batch)

    llm_result = await acall_llm_structured(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=DomainBatchResult,
        service=service,
    )
    result_obj, cost, _tokens = llm_result

    # Map results back to path IDs
    results: list[dict[str, Any]] = []
    for classification in result_obj.classifications:
        idx = classification.path_index - 1  # 1-based → 0-based
        if idx < 0 or idx >= len(batch):
            logger.warning(
                "LLM returned out-of-range path_index=%d for batch size=%d",
                classification.path_index,
                len(batch),
            )
            continue

        domain = classification.physics_domain
        # Validate against enum
        if domain not in valid_domains:
            logger.warning(
                "Invalid domain '%s' for path '%s' — falling back to 'general'",
                domain,
                batch[idx].get("id", "?"),
            )
            domain = "general"

        results.append(
            {
                "id": batch[idx]["id"],
                "physics_domain": domain,
                "domain_source": "llm_classified",
                "domain_model": model,
                "cost": cost / max(len(result_obj.classifications), 1),
            }
        )

    return results, cost


def _format_batch_user_prompt(batch: list[dict[str, Any]]) -> str:
    """Format batch paths as a numbered list for the LLM user prompt."""
    lines = []
    for i, p in enumerate(batch, 1):
        parts = [
            f"{i}. path: {p.get('id', p.get('path', '?'))}",
            f"   description: {p.get('description', 'N/A')}",
            f"   units: {p.get('units', '')}",
            f"   parent_path: {p.get('parent_path', 'N/A')}",
            f"   parent_description: {p.get('parent_description', 'N/A')}",
            f"   siblings: {', '.join(p.get('siblings', [])[:10])}",
            f"   ids_name: {p.get('ids_name', 'N/A')}",
            f"   node_category: {p.get('node_category', 'N/A')}",
        ]
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


# =============================================================================
# Context Gathering
# =============================================================================


def gather_classification_context(
    gc: GraphClient, path_ids: list[str]
) -> list[dict[str, Any]]:
    """Query graph for rich classification context per path.

    Returns list of context dicts with:
    - id, description, units, parent_path, parent_description
    - siblings (same parent), ids_name, node_category

    Args:
        gc: GraphClient instance.
        path_ids: List of IMASNode IDs to gather context for.

    Returns:
        List of context dicts ready for prompt injection.
    """
    if not path_ids:
        return []

    cypher = """
    UNWIND $path_ids AS pid
    MATCH (n:IMASNode {id: pid})
    OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
    OPTIONAL MATCH (sibling:IMASNode)-[:HAS_PARENT]->(parent)
    WHERE sibling.id <> n.id
    WITH n, parent,
         collect(DISTINCT split(sibling.id, '/')[-1])[..15] AS siblings
    RETURN n.id AS id,
           n.description AS description,
           n.units AS units,
           parent.id AS parent_path,
           parent.description AS parent_description,
           siblings,
           split(n.id, '/')[0] AS ids_name,
           n.node_category AS node_category
    """
    results = gc.query(cypher, path_ids=path_ids)

    return [
        {
            "id": r["id"],
            "description": r.get("description") or "",
            "units": r.get("units") or "",
            "parent_path": r.get("parent_path") or "",
            "parent_description": r.get("parent_description") or "",
            "siblings": r.get("siblings") or [],
            "ids_name": r.get("ids_name") or "",
            "node_category": r.get("node_category") or "",
        }
        for r in results
    ]


def _gather_expanded_context(
    gc: GraphClient, path_ids: list[str]
) -> list[dict[str, Any]]:
    """Gather expanded context for retry of 'general' paths.

    Expands context by:
    - Including more siblings (up to 25)
    - Adding cluster peer paths

    Returns enriched path dicts ready for batching.
    """
    if not path_ids:
        return []

    cypher = """
    UNWIND $path_ids AS pid
    MATCH (n:IMASNode {id: pid})
    OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
    OPTIONAL MATCH (sibling:IMASNode)-[:HAS_PARENT]->(parent)
    WHERE sibling.id <> n.id
    OPTIONAL MATCH (n)-[:IN_CLUSTER]->(cluster)<-[:IN_CLUSTER]-(peer:IMASNode)
    WHERE peer.id <> n.id AND peer.physics_domain IS NOT NULL
      AND peer.physics_domain <> 'general'
    WITH n, parent,
         collect(DISTINCT split(sibling.id, '/')[-1])[..25] AS siblings,
         collect(DISTINCT {path: peer.id, domain: peer.physics_domain})[..5] AS cluster_peers
    RETURN n.id AS id,
           n.description AS description,
           n.units AS units,
           parent.id AS parent_path,
           parent.description AS parent_description,
           siblings,
           split(n.id, '/')[0] AS ids_name,
           n.node_category AS node_category,
           cluster_peers
    """
    results = gc.query(cypher, path_ids=path_ids)

    expanded = []
    for r in results:
        ctx = {
            "id": r["id"],
            "description": r.get("description") or "",
            "units": r.get("units") or "",
            "parent_path": r.get("parent_path") or "",
            "parent_description": r.get("parent_description") or "",
            "siblings": r.get("siblings") or [],
            "ids_name": r.get("ids_name") or "",
            "node_category": r.get("node_category") or "",
        }
        # Append cluster peer info to description for additional context
        peers = r.get("cluster_peers") or []
        if peers:
            peer_hints = [f"{p['path']}→{p['domain']}" for p in peers]
            ctx["description"] += f" [Cluster peers: {'; '.join(peer_hints)}]"
        expanded.append(ctx)

    return expanded


# =============================================================================
# Batching
# =============================================================================


def batch_by_subtree(
    paths: list[dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE
) -> list[list[dict[str, Any]]]:
    """Group paths by parent subtree for batch coherence.

    Paths from the same parent are batched together since they
    share structural context. Oversized groups are split.

    Args:
        paths: List of path dicts (must have 'parent_path' or 'id').
        batch_size: Maximum paths per batch.

    Returns:
        List of batches, each a list of path dicts.
    """
    # Group by parent
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in paths:
        parent = p.get("parent_path") or "/".join(p.get("id", "").split("/")[:-1])
        groups[parent].append(p)

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []

    for _parent, group in sorted(groups.items()):
        if len(group) > batch_size:
            # Split oversized groups
            for i in range(0, len(group), batch_size):
                batches.append(group[i : i + batch_size])
        elif len(current_batch) + len(group) > batch_size:
            # Current batch would overflow — flush it
            if current_batch:
                batches.append(current_batch)
            current_batch = group.copy()
        else:
            current_batch.extend(group)

    if current_batch:
        batches.append(current_batch)

    return batches


# =============================================================================
# Hash / Idempotency
# =============================================================================


def compute_domain_input_hash(path_context: dict[str, Any]) -> str:
    """Compute SHA-256 hash of classification inputs for idempotency.

    Hash is based on (description, units, parent_path) — the key
    inputs that determine domain assignment.

    Args:
        path_context: Dict with 'description', 'units', 'parent_path'.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    parts = [
        path_context.get("description") or "",
        path_context.get("units") or "",
        path_context.get("parent_path") or "",
    ]
    content = "|".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# Internal Helpers
# =============================================================================


def _query_unclassified_paths(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Query paths needing Tier 1 LLM classification.

    Returns paths that:
    - Have a classifiable node_category
    - Are not in infrastructure subtrees
    - Have not been classified yet (or force=True)
    """
    ids_clause = _ids_filter_clause(ids_filter, "n")
    needs_clause = "AND n.domain_source IS NULL" if not force else ""

    categories = list(CLASSIFIABLE_CATEGORIES)

    cypher = f"""
    MATCH (n:IMASNode)
    WHERE n.node_category IN $categories
      AND NOT (n.id CONTAINS '/ids_properties/' OR n.id CONTAINS '/code/')
      {needs_clause}
      {ids_clause}
    RETURN n.id AS id, n.node_category AS node_category,
           n.description AS description, n.units AS units
    """
    return gc.query(cypher, categories=categories)


def _write_tier1_results(gc: GraphClient, results: list[dict[str, Any]]) -> None:
    """Write Tier 1 classification results to graph."""
    if not results:
        return

    cypher = """
    UNWIND $items AS item
    MATCH (n:IMASNode {id: item.id})
    SET n.physics_domain = item.physics_domain,
        n.domain_source = item.domain_source,
        n.domain_model = item.domain_model,
        n.domain_classified_at = datetime(),
        n.status = 'classified'
    """
    # Batch writes in chunks of 500
    for i in range(0, len(results), 500):
        chunk = results[i : i + 500]
        items = [
            {
                "id": r["id"],
                "physics_domain": r["physics_domain"],
                "domain_source": r["domain_source"],
                "domain_model": r.get("domain_model", ""),
            }
            for r in chunk
        ]
        gc.query(cypher, items=items)

    logger.info("Wrote %d Tier 1 domain classifications to graph", len(results))


def _ids_filter_clause(ids_filter: set[str] | None, node_var: str) -> str:
    """Build a WHERE clause fragment for IDS filtering.

    Uses the convention that the first path segment is the IDS name.
    """
    if not ids_filter:
        return ""
    # Filter by first segment of the path ID
    ids_list = list(ids_filter)
    # Use ANY() to check first segment membership
    return f"AND split({node_var}.id, '/')[0] IN {ids_list!r}"

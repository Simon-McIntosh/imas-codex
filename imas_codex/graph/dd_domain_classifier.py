"""Physics domain classifier for IMAS Data Dictionary paths.

Three-tier classification (executed in order):
- Tier 3: None for infrastructure metadata (ids_properties/*, code/*)
- Tier 1: LLM classification for physics paths with descriptions
- Tier 2: Inheritance from nearest classified ancestor for remaining paths
  (errors via HAS_ERROR, coordinates, identifiers, description-less structural)

Ordering is critical: Tier 1 must run before Tier 2 so that parents
are LLM-classified before children inherit from them.

Integrated as the CLASSIFY phase in the DD build pipeline, running after EMBED.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_path_domain_overrides() -> tuple[tuple[tuple[str, ...], str], ...]:
    """Load deterministic DD-path → domain overrides.

    See ``imas_codex/definitions/physics/path_domain_overrides.json``. Each
    override is ``(substrings, domain)``; a path matches when every substring
    is present in the path id.
    """
    p = (
        Path(__file__).resolve().parents[1]
        / "definitions"
        / "physics"
        / "path_domain_overrides.json"
    )
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return ()
    out: list[tuple[tuple[str, ...], str]] = []
    for entry in data.get("overrides", []):
        match = entry.get("match") or []
        domain = entry.get("domain")
        if match and domain:
            out.append((tuple(match), domain))
    return tuple(out)


def apply_path_domain_override(path_id: str) -> str | None:
    """Return the pinned domain if *path_id* matches a deterministic override.

    Guards DD path patterns whose semantic subject the tier-1 LLM classifier
    mis-assigns against the correct IDS default (e.g. distributions gyrocenter
    orbit frequencies → transport, lh_antennas hardware pressure →
    auxiliary_heating). Returns ``None`` when no override applies.
    """
    if not path_id:
        return None
    for substrings, domain in _load_path_domain_overrides():
        if all(s in path_id for s in substrings):
            return domain
    return None


# =============================================================================
# Constants
# =============================================================================

#: Categories eligible for Tier 1 LLM classification.
#: Coordinates and identifiers are excluded — they inherit from parent.
CLASSIFIABLE_CATEGORIES = frozenset(
    {
        "quantity",
        "structural",
        "representation",
        "geometry",
        "fit_artifact",
    }
)

#: Categories that always inherit from their parent (Tier 2).
INHERIT_CATEGORIES = frozenset(
    {
        "error",
        "coordinate",
        "identifier",
    }
)

#: Cypher fragment matching infrastructure metadata paths.
_IS_INFRASTRUCTURE = (
    "("
    "  n.id CONTAINS '/ids_properties/'"
    "  OR n.id CONTAINS '/code/'"
    "  OR n.id ENDS WITH '/ids_properties'"
    "  OR n.id ENDS WITH '/code'"
    ")"
)

#: Cypher fragment excluding infrastructure metadata from classification.
_NOT_INFRASTRUCTURE = f"NOT {_IS_INFRASTRUCTURE}"

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

    Execution order: Tier 3 → Tier 1 → Tier 2.
    Parents must be LLM-classified (Tier 1) before children can
    inherit (Tier 2).

    Args:
        gc: GraphClient instance.
        ids_filter: Optional set of IDS names to scope classification.
        force: Reclassify even if inputs unchanged (bypass hash check).
        dry_run: Log what would be classified without writing.
        model: Override model (defaults to get_model("sn-classifier")).
        on_cost: Callback for cost accumulation.
        on_items: Callback for item count progress.

    Returns:
        Stats dict with counts per tier.
    """
    from imas_codex.settings import get_model as _get_model

    if model is None:
        # Domain classification feeds SN names' physics_domain, so it is an
        # SN-attributable seat with its own [sn-classifier] config rather than
        # borrowing the generic [language] model.
        model = _get_model("sn-classifier")

    stats: dict[str, Any] = {
        "tier3_none": 0,
        "tier1_llm": 0,
        "tier1_general": 0,
        "tier1_retried": 0,
        "tier2_inherited": 0,
        "tier2_residual": 0,
        "total_cost": 0.0,
        "model": model,
    }

    if dry_run:
        logger.info("dry_run=True — counting candidates without writing")

    # --- Tier 3: infrastructure metadata → physics_domain = null ---
    tier3_count = classify_tier3_none(gc, ids_filter=ids_filter, dry_run=dry_run)
    stats["tier3_none"] = tier3_count
    logger.info("Tier 3 (none/metadata): %d paths", tier3_count)

    # --- Tier 1: LLM classification for paths WITH descriptions ---
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

    # --- Tier 2: inherit from classified parents (AFTER Tier 1) ---
    tier2_count = classify_tier2_inherit(
        gc, ids_filter=ids_filter, force=force, dry_run=dry_run
    )
    stats["tier2_inherited"] = tier2_count
    logger.info("Tier 2 (inherited): %d paths", tier2_count)

    # --- Residual check ---
    residual = _count_residual_unclassified(gc, ids_filter=ids_filter)
    stats["tier2_residual"] = residual
    if residual > 0:
        logger.warning(
            "Tier 2 residual: %d paths remain without domain_source", residual
        )

    if on_items:
        on_items(tier3_count + stats["tier1_llm"] + tier2_count)

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
    params = _ids_filter_params(ids_filter)

    if dry_run:
        cypher = f"""
        MATCH (n:IMASNode)
        WHERE {_IS_INFRASTRUCTURE}
          AND n.domain_source IS NULL
          {ids_clause}
        RETURN count(n) AS cnt
        """
        result = gc.query(cypher, **params)
        return result[0]["cnt"] if result else 0

    cypher = f"""
    MATCH (n:IMASNode)
    WHERE {_IS_INFRASTRUCTURE}
      AND n.domain_source IS NULL
      {ids_clause}
    SET n.physics_domain = null,
        n.domain_source = 'none_metadata',
        n.domain_classified_at = datetime(),
        n.status = 'classified'
    RETURN count(n) AS updated
    """
    result = gc.query(cypher, **params)
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
    """Inherit physics_domain from classified ancestors.

    Runs AFTER Tier 1 so parents have been LLM-classified. Five passes:
      a. Error paths via HAS_ERROR
      b. Coordinates via HAS_PARENT
      c. Identifiers via HAS_PARENT
      d. Remaining unclassified (description-less structural, etc.) via HAS_PARENT
      e. Non-infrastructure metadata via HAS_PARENT

    Uses domain_source IS NULL to find unprocessed paths (not physics_domain
    checks), so paths with stale IDS-level domains are correctly caught.
    Requires the source parent to have domain_source IS NOT NULL (i.e.,
    classified in this or a prior run) to avoid inheriting stale data.

    Multi-hop: passes d and e use variable-length HAS_PARENT traversal
    to reach the nearest classified ancestor.

    Returns:
        Total count of nodes updated.
    """
    total = 0
    total += _inherit_from_error_parent(
        gc, ids_filter=ids_filter, force=force, dry_run=dry_run
    )
    total += _inherit_from_parent_by_category(
        gc,
        categories=["coordinate"],
        ids_filter=ids_filter,
        force=force,
        dry_run=dry_run,
    )
    total += _inherit_from_parent_by_category(
        gc,
        categories=["identifier"],
        ids_filter=ids_filter,
        force=force,
        dry_run=dry_run,
    )
    total += _inherit_remaining_unclassified(
        gc, ids_filter=ids_filter, force=force, dry_run=dry_run
    )
    total += _inherit_from_metadata_parent(
        gc, ids_filter=ids_filter, force=force, dry_run=dry_run
    )
    return total


def _needs_classification_clause(force: bool, node_var: str = "n") -> str:
    """WHERE fragment selecting nodes that need classification.

    Uses domain_source IS NULL (not physics_domain checks) to catch
    paths with stale IDS-level domains.
    """
    if force:
        return ""
    return f"AND {node_var}.domain_source IS NULL"


def _parent_is_classified_clause(parent_var: str = "parent") -> str:
    """WHERE fragment requiring the source parent to be classified.

    Only inherits from parents that have been through the classifier
    (domain_source IS NOT NULL), preventing inheritance of stale
    IDS-level domains.
    """
    return (
        f"AND {parent_var}.domain_source IS NOT NULL "
        f"AND {parent_var}.physics_domain IS NOT NULL"
    )


def _inherit_from_error_parent(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Error paths inherit domain from their parent via HAS_ERROR."""
    ids_clause = _ids_filter_clause(ids_filter, "parent")
    params = _ids_filter_params(ids_filter)
    needs_clause = _needs_classification_clause(force, "err")
    parent_clause = _parent_is_classified_clause("parent")

    if dry_run:
        cypher = f"""
        MATCH (parent:IMASNode)-[:HAS_ERROR]->(err:IMASNode)
        WHERE true
          {parent_clause}
          {needs_clause}
          {ids_clause}
        RETURN count(err) AS cnt
        """
        result = gc.query(cypher, **params)
        return result[0]["cnt"] if result else 0

    cypher = f"""
    MATCH (parent:IMASNode)-[:HAS_ERROR]->(err:IMASNode)
    WHERE true
      {parent_clause}
      {needs_clause}
      {ids_clause}
    SET err.physics_domain = parent.physics_domain,
        err.domain_source = 'inherited_from_parent',
        err.domain_classified_at = datetime(),
        err.status = 'classified'
    RETURN count(err) AS updated
    """
    result = gc.query(cypher, **params)
    count = result[0]["updated"] if result else 0
    if count:
        logger.info("  error paths inherited: %d", count)
    return count


def _inherit_from_parent_by_category(
    gc: GraphClient,
    *,
    categories: list[str],
    ids_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Paths of given categories inherit from nearest classified ancestor.

    Uses variable-length HAS_PARENT traversal (up to 5 hops) to handle
    structural chains where intermediate nodes lack descriptions.
    """
    ids_clause = _ids_filter_clause(ids_filter, "n")
    params: dict[str, Any] = {"categories": categories}
    params.update(_ids_filter_params(ids_filter))
    needs_clause = _needs_classification_clause(force, "n")

    if dry_run:
        cypher = f"""
        MATCH path = (n:IMASNode)-[:HAS_PARENT*1..5]->(ancestor:IMASNode)
        WHERE n.node_category IN $categories
          AND ancestor.domain_source IS NOT NULL
          AND ancestor.physics_domain IS NOT NULL
          {needs_clause}
          {ids_clause}
        WITH n, ancestor, length(path) AS dist
        ORDER BY dist ASC
        WITH n, head(collect(ancestor)) AS nearest
        WHERE nearest IS NOT NULL
        RETURN count(n) AS cnt
        """
        result = gc.query(cypher, **params)
        return result[0]["cnt"] if result else 0

    # Use shortestPath-based approach for nearest ancestor
    cypher = f"""
    MATCH (n:IMASNode)
    WHERE n.node_category IN $categories
      {needs_clause}
      {ids_clause}
    CALL {{
        WITH n
        MATCH path = (n)-[:HAS_PARENT*1..5]->(ancestor:IMASNode)
        WHERE ancestor.domain_source IS NOT NULL
          AND ancestor.physics_domain IS NOT NULL
        WITH ancestor, length(path) AS dist
        ORDER BY dist ASC
        LIMIT 1
        RETURN ancestor.physics_domain AS inherited_domain
    }}
    SET n.physics_domain = inherited_domain,
        n.domain_source = 'inherited_from_parent',
        n.domain_classified_at = datetime(),
        n.status = 'classified'
    RETURN count(n) AS updated
    """
    result = gc.query(cypher, **params)
    count = result[0]["updated"] if result else 0
    if count:
        logger.info("  %s paths inherited: %d", categories, count)
    return count


def _inherit_remaining_unclassified(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Catch-all: remaining unclassified non-infrastructure paths inherit
    from nearest classified ancestor via multi-hop HAS_PARENT traversal.

    Catches description-less structural paths, any other categories
    that fell through Tier 1 (e.g. quantity/representation without desc).
    """
    ids_clause = _ids_filter_clause(ids_filter, "n")
    params = _ids_filter_params(ids_filter)
    needs_clause = _needs_classification_clause(force, "n")

    if dry_run:
        cypher = f"""
        MATCH (n:IMASNode)
        WHERE n.domain_source IS NULL
          AND {_NOT_INFRASTRUCTURE}
          AND n.node_category <> 'error'
          {needs_clause}
          {ids_clause}
        RETURN count(n) AS cnt
        """
        result = gc.query(cypher, **params)
        return result[0]["cnt"] if result else 0

    cypher = f"""
    MATCH (n:IMASNode)
    WHERE n.domain_source IS NULL
      AND {_NOT_INFRASTRUCTURE}
      AND n.node_category <> 'error'
      {needs_clause}
      {ids_clause}
    CALL {{
        WITH n
        MATCH path = (n)-[:HAS_PARENT*1..10]->(ancestor:IMASNode)
        WHERE ancestor.domain_source IS NOT NULL
          AND ancestor.physics_domain IS NOT NULL
        WITH ancestor, length(path) AS dist
        ORDER BY dist ASC
        LIMIT 1
        RETURN ancestor.physics_domain AS inherited_domain
    }}
    SET n.physics_domain = inherited_domain,
        n.domain_source = 'inherited_from_parent',
        n.domain_classified_at = datetime(),
        n.status = 'classified'
    RETURN count(n) AS updated
    """
    result = gc.query(cypher, **params)
    count = result[0]["updated"] if result else 0
    if count:
        logger.info("  remaining unclassified paths inherited: %d", count)
    return count


def _inherit_from_metadata_parent(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Non-infrastructure metadata inherits domain from classified ancestor."""
    ids_clause = _ids_filter_clause(ids_filter, "n")
    params = _ids_filter_params(ids_filter)
    needs_clause = _needs_classification_clause(force, "n")

    if dry_run:
        cypher = f"""
        MATCH (n:IMASNode {{node_category: 'metadata'}})
        WHERE {_NOT_INFRASTRUCTURE}
          {needs_clause}
          {ids_clause}
        RETURN count(n) AS cnt
        """
        result = gc.query(cypher, **params)
        return result[0]["cnt"] if result else 0

    cypher = f"""
    MATCH (n:IMASNode {{node_category: 'metadata'}})
    WHERE {_NOT_INFRASTRUCTURE}
      {needs_clause}
      {ids_clause}
    CALL {{
        WITH n
        MATCH path = (n)-[:HAS_PARENT*1..10]->(ancestor:IMASNode)
        WHERE ancestor.domain_source IS NOT NULL
          AND ancestor.physics_domain IS NOT NULL
        WITH ancestor, length(path) AS dist
        ORDER BY dist ASC
        LIMIT 1
        RETURN ancestor.physics_domain AS inherited_domain
    }}
    SET n.physics_domain = inherited_domain,
        n.domain_source = 'inherited_from_parent',
        n.domain_classified_at = datetime(),
        n.status = 'classified'
    RETURN count(n) AS updated
    """
    result = gc.query(cypher, **params)
    count = result[0]["updated"] if result else 0
    if count:
        logger.info("  metadata paths inherited: %d", count)
    return count


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

        # Deterministic override: pin known DD path patterns whose semantic
        # subject the LLM mis-assigns against the correct IDS default.
        domain_source = "llm_classified"
        override = apply_path_domain_override(batch[idx]["id"])
        if override is not None and override != domain:
            logger.info(
                "path-domain override: %s %s -> %s",
                batch[idx]["id"],
                domain,
                override,
            )
            domain = override
            domain_source = "deterministic_override"

        results.append(
            {
                "id": batch[idx]["id"],
                "physics_domain": domain,
                "domain_source": domain_source,
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
    - Have a classifiable node_category (not coordinate/identifier/error)
    - Are not in infrastructure subtrees
    - Have a description (structural paths without descriptions fall
      through to Tier 2 inheritance instead)
    - Have not been classified yet (or force=True)
    """
    ids_clause = _ids_filter_clause(ids_filter, "n")
    params: dict[str, Any] = {"categories": list(CLASSIFIABLE_CATEGORIES)}
    params.update(_ids_filter_params(ids_filter))
    needs_clause = "AND n.domain_source IS NULL" if not force else ""

    # Require description for structural paths — description-less structural
    # nodes inherit from parent in Tier 2. Other categories (quantity,
    # geometry, representation, fit_artifact) still benefit from LLM
    # classification even with minimal descriptions via path/units/parent context.
    cypher = f"""
    MATCH (n:IMASNode)
    WHERE n.node_category IN $categories
      AND {_NOT_INFRASTRUCTURE}
      AND (n.node_category <> 'structural'
           OR (n.description IS NOT NULL AND trim(n.description) <> ''))
      {needs_clause}
      {ids_clause}
    RETURN n.id AS id, n.node_category AS node_category,
           n.description AS description, n.units AS units
    """
    return gc.query(cypher, **params)


def _count_residual_unclassified(
    gc: GraphClient,
    *,
    ids_filter: set[str] | None = None,
) -> int:
    """Count paths that remain unclassified after all tiers.

    These are paths with domain_source IS NULL that are not
    infrastructure metadata. Used for monitoring/alerting.
    """
    ids_clause = _ids_filter_clause(ids_filter, "n")
    params = _ids_filter_params(ids_filter)

    cypher = f"""
    MATCH (n:IMASNode)
    WHERE n.domain_source IS NULL
      AND {_NOT_INFRASTRUCTURE}
      {ids_clause}
    RETURN count(n) AS cnt
    """
    result = gc.query(cypher, **params)
    return result[0]["cnt"] if result else 0


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
    Callers must pass ``ids_filter_list=list(ids_filter)`` as a query parameter.
    """
    if not ids_filter:
        return ""
    return f"AND split({node_var}.id, '/')[0] IN $ids_filter_list"


def _ids_filter_params(ids_filter: set[str] | None) -> dict[str, Any]:
    """Return query parameters for IDS filtering.

    Companion to ``_ids_filter_clause`` — provides the ``ids_filter_list``
    parameter expected by the Cypher clause.
    """
    if not ids_filter:
        return {}
    return {"ids_filter_list": list(ids_filter)}

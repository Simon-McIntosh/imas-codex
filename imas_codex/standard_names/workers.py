"""Async workers for the standard-name build pipeline.

Five-phase generate pipeline:

    EXTRACT → COMPOSE → VALIDATE → CONSOLIDATE → PERSIST

- **extract**: queries graph for DD paths or facility signals, builds batches
- **compose**: LLM-generates standard names from extraction batches
- **validate**: validates names against grammar via round-trip + fields check
- **consolidate**: cross-batch dedup, conflict detection, coverage accounting
- **persist**: writes consolidated names to graph with provenance

Workers follow the ``dd_workers.py`` pattern: each is an async function
with signature ``async def worker(state, **_kwargs)`` that updates stats,
marks phases done, and respects ``state.should_stop()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re as _re
from collections.abc import Callable
from functools import cache as _cache
from typing import TYPE_CHECKING, Any

from imas_codex.standard_names.defaults import DEFAULT_ESCALATION_MODEL
from imas_codex.standard_names.source_paths import (
    encode_source_path,
    strip_dd_prefix,
)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetLease, BudgetManager
    from imas_codex.standard_names.sources.base import ExtractionBatch
    from imas_codex.standard_names.state import StandardNameBuildState

logger = logging.getLogger(__name__)


def normalize_spelling(name: str) -> str:
    """Deterministic British→American spelling normalization.

    Splits the underscore-delimited name into tokens and converts each
    via ``breame.spelling.get_american_spelling``.  A small domain-specific
    supplement catches physics terms missing from breame's dictionary.
    """
    from breame.spelling import get_american_spelling

    tokens = name.split("_")
    return "_".join(_DOMAIN_UK_US.get(t, get_american_spelling(t)) for t in tokens)


def normalize_prose_spelling(text: str) -> str:
    """British→American spelling normalization for prose text.

    Unlike ``normalize_spelling`` which operates on underscore-delimited
    name tokens, this function handles natural English prose — splitting
    on word boundaries and preserving punctuation, whitespace, and
    markdown formatting.
    """
    if not text:
        return text
    from breame.spelling import get_american_spelling

    def _replace(match: _re.Match) -> str:
        word = match.group(0)
        # Preserve case: if the word is capitalized, capitalize the replacement
        us = _DOMAIN_UK_US.get(word.lower(), get_american_spelling(word.lower()))
        if us == word.lower():
            return word  # no change
        if word[0].isupper():
            return us[0].upper() + us[1:]
        return us

    # Match word-like sequences (letters only, preserving everything else)
    return _re.sub(r"[A-Za-z]+", _replace, text)


# Physics-domain UK→US supplements not in breame's dictionary.
_DOMAIN_UK_US: dict[str, str] = {
    "linearised": "linearized",
    "linearise": "linearize",
    "discretised": "discretized",
    "discretise": "discretize",
    "symmetrised": "symmetrized",
    "symmetrise": "symmetrize",
    "parameterised": "parameterized",
    "parameterise": "parameterize",
    "centreline": "centerline",
    "centre": "center",
    "grey": "gray",
    "colour": "color",
    "favour": "favor",
    "favourite": "favorite",
    "honour": "honor",
    "labelled": "labeled",
    "labelling": "labeling",
    "travelled": "traveled",
    "travelling": "traveling",
    "catalogue": "catalog",
    "programme": "program",
    "analogue": "analog",
    "dialogue": "dialog",
}

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


def _dedup_adjacent_tokens(name: str, log: logging.Logger | None = None) -> str:
    """Remove adjacent duplicate tokens introduced by grammar round-trip.

    The ISN grammar (≤ 0.7.0rc27) has a known bug where ``parse →
    compose`` doubles certain tokens — e.g.
    ``magnetic_field_probe`` → ``magnetic_magnetic_field_probe``.

    This function collapses any ``tok_tok`` pair into a single ``tok``
    while preserving legitimate compounds via
    :data:`~imas_codex.standard_names.audits._COMPOUND_SUBJECT_PAIRS`.
    """
    from imas_codex.standard_names.audits import _COMPOUND_SUBJECT_PAIRS

    tokens = name.split("_")
    result: list[str] = [tokens[0]]
    for tok in tokens[1:]:
        if tok == result[-1] and f"{tok}_{tok}" not in _COMPOUND_SUBJECT_PAIRS:
            if log:
                log.debug(
                    "Dedup adjacent token %r in %r",
                    f"{tok}_{tok}",
                    name,
                )
            continue
        result.append(tok)
    return "_".join(result)


# ---------------------------------------------------------------------------
# Pre-validation gate (W4b): reject malformed LLM output before MERGE
# ---------------------------------------------------------------------------

_SNAKE_CASE_RE = _re.compile(r"[a-z][a-z0-9_]*")


def is_well_formed_candidate(raw_name: str) -> tuple[bool, str | None]:
    """Check whether *raw_name* is a plausible standard-name candidate.

    Returns ``(True, None)`` if the name passes all checks, or
    ``(False, reason)`` if it should be rejected **before** MERGE.

    Rejection criteria:
    - Empty or whitespace-only
    - Length > 100 characters (likely LLM monologue)
    - Contains newline, tab, ``{``, ``}``, ``\\``, or triple-dot
    - Not snake_case-shaped after stripping
    """
    if not raw_name or not raw_name.strip():
        return False, "empty_or_whitespace"
    name = raw_name.strip()
    if len(name) > 100:
        return False, "too_long"
    if any(ch in name for ch in ("\n", "\t", "{", "}", "\\")):
        return False, "illegal_chars"
    if "..." in name:
        return False, "triple_dot"
    if not _SNAKE_CASE_RE.fullmatch(name):
        return False, "not_snake_case"
    return True, None


def _coerce_llm_metric(value: Any) -> int:
    """Best-effort int coercion for LLM telemetry fields."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_llm_telemetry(
    llm_out: Any,
    *,
    parsed_obj: Any = None,
    tokens_fallback: Any = None,
) -> tuple[int, int, int, int]:
    """Normalize split token/cache telemetry across LLM return shapes."""

    def _from_attrs(obj: Any) -> tuple[int, int, int, int]:
        return (
            _coerce_llm_metric(getattr(obj, "input_tokens", 0)),
            _coerce_llm_metric(getattr(obj, "output_tokens", 0)),
            _coerce_llm_metric(getattr(obj, "cache_read_tokens", 0)),
            _coerce_llm_metric(getattr(obj, "cache_creation_tokens", 0)),
        )

    metrics = _from_attrs(llm_out)
    if any(metrics):
        return metrics

    if parsed_obj is not None:
        metrics = _from_attrs(parsed_obj)
        if any(metrics):
            return metrics

    if isinstance(tokens_fallback, dict):
        return (
            _coerce_llm_metric(tokens_fallback.get("input_tokens", 0)),
            _coerce_llm_metric(tokens_fallback.get("output_tokens", 0)),
            _coerce_llm_metric(tokens_fallback.get("cache_read_tokens", 0)),
            _coerce_llm_metric(
                tokens_fallback.get(
                    "cache_creation_tokens",
                    tokens_fallback.get("cache_write_tokens", 0),
                )
            ),
        )

    if isinstance(tokens_fallback, tuple | list):
        values = list(tokens_fallback)
        return (
            _coerce_llm_metric(values[0] if len(values) > 0 else 0),
            _coerce_llm_metric(values[1] if len(values) > 1 else 0),
            _coerce_llm_metric(values[2] if len(values) > 2 else 0),
            _coerce_llm_metric(values[3] if len(values) > 3 else 0),
        )

    return (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Auto-VocabGap detection for physical_base (W29)
# ---------------------------------------------------------------------------


@_cache
def _load_known_physical_bases() -> frozenset[str]:
    """Return the set of registered physical_base tokens from ISN.

    Reads the ``grammar/vocabularies/physical_bases.yml`` vocabulary file
    from the installed ``imas_standard_names`` package.  Falls back to an
    empty set if the package or file is unavailable.
    """
    try:
        from importlib.resources import files

        import yaml

        text = (
            files("imas_standard_names")
            .joinpath("grammar/vocabularies/physical_bases.yml")
            .read_text()
        )
        data = yaml.safe_load(text)
        bases = data.get("bases", {})
        if isinstance(bases, dict):
            return frozenset(bases.keys())
        return frozenset()
    except Exception:
        return frozenset()


def _auto_detect_physical_base_gaps(
    candidates: list,  # StandardNameCandidate instances or dicts
    known_bases: frozenset[str] | None = None,
) -> list[dict]:
    """Parse each candidate name and surface novel ``physical_base`` tokens.

    Returns a list of VocabGap-compatible dicts for bases not in the ISN
    registry.  These are auto-tracked so the LLM does not need to emit
    explicit ``vocab_gap`` exits for ``physical_base``.

    Args:
        candidates: Parsed ``StandardNameCandidate`` objects or dicts.
        known_bases: Pre-loaded set of registered physical_base tokens.
            Defaults to ``_load_known_physical_bases()`` if not provided.
    """
    if known_bases is None:
        known_bases = _load_known_physical_bases()

    gaps: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (source_id, base) dedup

    for c in candidates:
        try:
            # Handle both model instances and dicts
            if hasattr(c, "base_token"):
                base = c.base_token
                source_id = c.source_id
                base_kind = getattr(c, "base_kind", "quantity")
            elif isinstance(c, dict):
                base = c.get("id", "")  # dict candidates use 'id' for name
                source_id = c.get("source_id", "")
                # For dicts, parse the name to extract base
                try:
                    from imas_standard_names.grammar import parse_standard_name

                    parsed = parse_standard_name(base)
                    base = parsed.physical_base
                    base_kind = "quantity"
                except Exception:
                    continue
            else:
                continue

            if base_kind == "quantity" and base and base not in known_bases:
                key = (source_id, base)
                if key not in seen:
                    seen.add(key)
                    gaps.append(
                        {
                            "source_id": source_id,
                            "segment": "physical_base",
                            "token": base,
                            "reason": (
                                f"Novel physical_base proposed by compose: {base!r}"
                            ),
                        }
                    )
        except Exception:
            continue  # parse failures already handled elsewhere

    return gaps


# =============================================================================
# EXTRACT phase
# =============================================================================


async def extract_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Extract candidate quantities from graph entities into batches.

    For DD source: queries IMASNode paths, groups by cluster/IDS/prefix.
    Skips sources already linked via HAS_STANDARD_NAME unless --force.
    Stores ExtractionBatch objects in ``state.extracted``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_extract_worker")
    wlog.info("Starting extraction (source=%s)", state.source)

    def _on_status(text: str) -> None:
        state.extract_stats.status_text = text
        if state.loop_extract_stats is not None:
            state.loop_extract_stats.status_text = text
            # Stream sub-stage events so the user sees a timeline rather than
            # a single status string that can appear stuck during long
            # downstream phases (compose, enrich, review).
            try:
                state.loop_extract_stats.stream_queue.add(
                    [
                        {
                            "primary_text": "extract",
                            "primary_text_style": "dim cyan",
                            "description": text,
                        }
                    ]
                )
            except Exception:
                pass

    def _run() -> list:
        from imas_codex.standard_names.graph_ops import (
            get_existing_standard_names,
            get_named_source_ids,
        )
        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        _on_status("loading existing names…")
        existing = get_existing_standard_names()

        # Regen mode: when --min-score F is passed, re-queue sources whose
        # linked StandardName has reviewer_score < min_score. The subsequent
        # feedback-injection block (always-on) attaches reviewer_comments /
        # tier / scores to each item so compose regenerates with critique
        # in-prompt.
        if state.is_regen_mode() and state.source == "dd":
            from imas_codex.standard_names.graph_ops import (
                fetch_low_score_sources,
            )
            from imas_codex.standard_names.sources.dd import extract_specific_paths

            _on_status(f"loading sources below reviewer_score={state.min_score}…")
            regen_sources = fetch_low_score_sources(
                min_score=state.min_score,
                domain=state.domain_filter,
                ids=state.ids_filter,
                limit=state.limit,
                source_type="dd",
            )
            if not regen_sources:
                wlog.info(
                    "Regen mode (--min-score %s): no low-score sources found "
                    "(domain=%s, ids=%s, limit=%s)",
                    state.min_score,
                    state.domain_filter,
                    state.ids_filter,
                    state.limit,
                )
                return []
            wlog.info(
                "Regen mode: %d sources below reviewer_score=%s "
                "(domain=%s, ids=%s, limit=%s)",
                len(regen_sources),
                state.min_score,
                state.domain_filter,
                state.ids_filter,
                state.limit,
            )
            paths = [r["source_id"] for r in regen_sources]
            batches = extract_specific_paths(
                paths=paths,
                existing_names=existing,
                on_status=_on_status,
            )
            return batches

        # Source-level skip for resumability
        named_ids: set[str] = set()
        if not state.force:
            named_ids = get_named_source_ids()
            if named_ids:
                wlog.info("Skipping %d already-named sources", len(named_ids))

        if state.source == "dd":
            from imas_codex.standard_names.batching import (
                get_generate_batch_config,
            )

            batch_cfg = get_generate_batch_config()
            batches = extract_dd_candidates(
                ids_filter=state.ids_filter,
                domain_filter=state.domain_filter,
                limit=state.limit,
                existing_names=existing,
                on_status=_on_status,
                from_model=state.from_model,
                force=state.force,
                name_only=state.name_only,
                name_only_batch_size=state.name_only_batch_size,
                max_batch_size=batch_cfg["batch_size"],
                max_tokens=batch_cfg["max_tokens"],
            )
        else:
            wlog.error("Unknown source: %s", state.source)
            return []

        # Filter out already-named sources from batches
        if named_ids and not state.force:
            for batch in batches:
                batch.items = [
                    item
                    for item in batch.items
                    if item.get("path", item.get("signal_id")) not in named_ids
                ]
            # Remove empty batches
            batches = [b for b in batches if b.items]

        return batches

    batches = await asyncio.to_thread(_run)

    # Inject previous name context for --force regeneration
    if state.force:
        # --paths mode gets rich metadata (full docs, links, linked DD paths)
        use_rich = False

        def _get_mapping():
            from imas_codex.standard_names.graph_ops import get_source_name_mapping

            return get_source_name_mapping(rich=use_rich)

        source_names = await asyncio.to_thread(_get_mapping)
        injected = 0
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path and path in source_names:
                    item["previous_name"] = source_names[path]
                    injected += 1
        if injected:
            wlog.info("Injected previous_name context for %d items", injected)

    # Inject prior reviewer feedback for targeted regeneration (always on).
    # The compose prompt's {% if item.review_feedback %} block surfaces the
    # previous reviewer critique so the LLM can directly address it in the
    # new name. No-op when no prior feedback exists.
    def _get_feedback():
        from imas_codex.standard_names.graph_ops import (
            fetch_review_feedback_for_sources,
        )

        ids: set[str] = set()
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path:
                    ids.add(path)
        return fetch_review_feedback_for_sources(ids)

    feedback_map = await asyncio.to_thread(_get_feedback)
    fb_injected = 0
    for batch in batches:
        for item in batch.items:
            path = item.get("path", item.get("signal_id"))
            if path and path in feedback_map:
                item["review_feedback"] = feedback_map[path]
                fb_injected += 1
    if fb_injected:
        wlog.info(
            "Injected review_feedback for %d items",
            fb_injected,
        )

    # Inject full reviewer history from Review nodes (Phase 3).
    # Complements the single-review review_feedback above with the
    # complete chain of prior reviews — latest full + older themes.
    def _get_reviewer_history():
        from imas_codex.standard_names.graph_ops import (
            fetch_reviewer_history_for_sources,
        )

        ids: set[str] = set()
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path:
                    ids.add(path)
        return fetch_reviewer_history_for_sources(ids)

    history_map = await asyncio.to_thread(_get_reviewer_history)
    hist_injected = 0
    for batch in batches:
        for item in batch.items:
            path = item.get("path", item.get("signal_id"))
            if path and path in history_map:
                item["reviewer_history"] = history_map[path]
                hist_injected += 1
    if hist_injected:
        wlog.info(
            "Injected reviewer_history for %d items",
            hist_injected,
        )

    # Write StandardNameSource nodes for crash-resilient tracking
    if not state.dry_run and batches:
        from imas_codex.settings import get_dd_version
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        _dd_ver = get_dd_version()
        sources = []
        source_type = "dd" if state.source == "dd" else "signals"
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if not path:
                    continue
                sources.append(
                    {
                        "id": f"{source_type}:{path}",
                        "source_type": source_type,
                        "source_id": path,
                        "dd_path": path if source_type == "dd" else None,
                        "batch_key": batch.group_key,
                        "status": "extracted",
                        "dd_version": _dd_ver,
                        "description": item.get("description")
                        or item.get("documentation")
                        or "",
                    }
                )

        if sources:
            written = await asyncio.to_thread(
                merge_standard_name_sources, sources, force=state.force
            )
            wlog.info("Wrote %d StandardNameSource nodes to graph", written)

    total_items = sum(len(b.items) for b in batches)
    state.extracted = batches
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)
    if state.loop_extract_stats is not None:
        state.loop_extract_stats.total = total_items
        state.loop_extract_stats.processed = total_items
        state.loop_extract_stats.record_batch(total_items)

    wlog.info(
        "Extraction complete: %d batches, %d items",
        len(batches),
        total_items,
    )
    state.stats["extract_batches"] = len(batches)
    state.stats["extract_count"] = total_items

    state.extract_stats.freeze_rate()
    state.extract_stats.status_text = ""
    state.extract_phase.mark_done()
    state.extract_stats.stream_queue.add(
        [
            {
                "primary_text": "extract",
                "description": f"{total_items} paths in {len(batches)} batches",
            }
        ]
    )
    if state.loop_extract_stats is not None:
        state.loop_extract_stats.freeze_rate()
        state.loop_extract_stats.status_text = ""
        state.loop_extract_stats.stream_queue.add(
            [
                {
                    "primary_text": "extract",
                    "description": f"{total_items} paths in {len(batches)} batches",
                }
            ]
        )


# =============================================================================
# COMPOSE phase
# =============================================================================


def _search_nearby_names(query: str, k: int = 5) -> list[dict]:
    """Search for existing standard names near *query* for collision avoidance.

    Wraps :func:`imas_codex.standard_names.search.search_standard_names_vector`
    with graceful fallback — never raises, returns ``[]`` if graph or
    embeddings are unavailable.
    """
    try:
        from imas_codex.standard_names.search import search_standard_names_vector

        return search_standard_names_vector(query, k=k)
    except Exception:
        return []


# =============================================================================
# DD context enrichment — fetch rich graph data before composing
# =============================================================================

_DD_CONTEXT_QUERY = """
MATCH (n:IMASNode {id: $path})
OPTIONAL MATCH (n)-[:HAS_IDENTIFIER_SCHEMA]->(ident:IdentifierSchema)
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (parent)-[:HAS_CHILD]->(sibling:IMASNode)
WHERE sibling.id <> $path
  AND NOT (sibling.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
WITH n, ident, u, parent,
     collect(DISTINCT {
         path: sibling.id,
         description: sibling.description,
         data_type: sibling.data_type
     })[0..8] AS sibling_fields
RETURN n.coordinate1_same_as AS coordinate1,
       n.coordinate2_same_as AS coordinate2,
       n.coordinate3_same_as AS coordinate3,
       n.timebasepath AS timebase,
       n.cocos_transformation_type AS cocos_label,
       n.cocos_transformation_expression AS cocos_expression,
       n.lifecycle_status AS lifecycle_status,
       n.data_type AS data_type,
       ident.name AS identifier_schema_name,
       ident.documentation AS identifier_schema_doc,
       ident.options AS identifier_options,
       u.id AS unit_from_rel,
       parent.id AS parent_path,
       parent.description AS parent_description,
       sibling_fields
"""

_CROSS_IDS_QUERY = """
MATCH (n:IMASNode {id: $path})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WHERE c.scope IN ['global', 'domain']
WITH c ORDER BY c.scope ASC LIMIT 3
MATCH (member:IMASNode)-[:IN_CLUSTER]->(c)
WHERE member.id <> $path
  AND NOT (member.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
RETURN c.label AS cluster_label,
       c.description AS cluster_description,
       c.scope AS cluster_scope,
       collect(DISTINCT member.id)[0..5] AS member_paths
"""

_VERSION_HISTORY_QUERY = """
MATCH (vc:IMASNodeChange)-[:FOR_IMAS_PATH]->(n:IMASNode {id: $path})
WHERE vc.change_type IN [
    'path_added', 'cocos_transformation_type', 'sign_convention',
    'units', 'path_renamed', 'definition_clarification'
]
RETURN vc.id AS change_id, vc.change_type AS change_type
"""

_ERROR_FIELDS_QUERY = """
MATCH (d:IMASNode {id: $path})-[:HAS_ERROR]->(e:IMASNode)
RETURN e.id AS error_path
"""

_IDS_CONTEXT_QUERY = """
MATCH (ids:IDS {id: $ids_name})
OPTIONAL MATCH (child:IMASNode)-[:IN_IDS]->(ids)
WHERE child.id STARTS WITH $ids_prefix
  AND child.data_type IN ['STRUCTURE', 'STRUCT_ARRAY', 'FLT_1D']
  AND size([x IN split(child.id, '/') WHERE true]) = 2
WITH ids,
     collect(DISTINCT {
         name: child.id, description: child.description, data_type: child.data_type
     })[0..10] AS top_sections
RETURN ids.description AS ids_description,
       ids.documentation AS ids_documentation,
       top_sections
"""


def _enrich_ids_context(ids_name: str) -> dict | None:
    """Fetch IDS-level context for batch header.

    Returns dict with ids_description, ids_documentation, top_sections,
    or None if IDS not found.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = list(
            gc.query(
                _IDS_CONTEXT_QUERY,
                ids_name=ids_name,
                ids_prefix=f"{ids_name}/",
            )
        )
        if not rows:
            return None
        row = rows[0]
        sections = row.get("top_sections") or []
        valid = [s for s in sections if s.get("name")]
        return {
            "ids_description": row.get("ids_description") or "",
            "ids_documentation": row.get("ids_documentation") or "",
            "top_sections": valid,
        }


def _hybrid_search_neighbours_batch(
    gc: Any,
    items: list[tuple[str, str | None, str | None]],
    *,
    max_results: int = 15,
    search_k: int = 10,
) -> list[list[dict]]:
    """Batch-embed then search neighbours for multiple (path, desc, domain) items.

    Replaces the per-item N+1 embed pattern: collects all unique
    description-query texts, embeds them in **one** remote round-trip
    via :func:`embed_query_texts`, then fans out ``hybrid_dd_search``
    calls with pre-computed embeddings.

    Path-text queries (``"equilibrium/time_slice/..."``-style) don't
    need embedding — ``hybrid_dd_search`` detects them and uses
    text-only mode automatically.

    Args:
        gc: Active graph client.
        items: List of ``(path, description, physics_domain)`` tuples.
        max_results: Cap per-item neighbour count.
        search_k: ``k`` passed to ``hybrid_dd_search``.

    Returns:
        List of neighbour-lists in the same order as *items*.
    """
    from imas_codex.embeddings.description import embed_query_texts
    from imas_codex.graph.dd_search import hybrid_dd_search

    if not items:
        return []

    # ── 1. Collect unique description queries needing embedding ─────
    text_set: dict[str, None] = {}  # ordered-set via dict
    item_desc_queries: list[str] = []  # per-item desc query text
    for _path, description, _domain in items:
        desc_query = (description or "")[:200].strip()
        item_desc_queries.append(desc_query)
        if desc_query:
            text_set.setdefault(desc_query, None)

    unique_texts = list(text_set)

    # ── 2. Single batch embed call ──────────────────────────────────
    embed_cache: dict[str, list[float]] = {}
    if unique_texts:
        try:
            embeddings = embed_query_texts(unique_texts)
            # Map by index — handle length mismatch gracefully (server may
            # deduplicate or batch differently)
            for i, emb in enumerate(embeddings):
                if i < len(unique_texts):
                    embed_cache[unique_texts[i]] = emb
        except Exception:
            logger.warning(
                "Batch query embedding failed; falling back to text-only search",
                exc_info=True,
            )

    # ── 3. Per-item hybrid search (with pre-computed embeddings) ────
    all_results: list[list[dict]] = []
    # Collect all hit paths across items for a single batch SN resolution
    per_item_hits: list[dict[str, Any]] = []  # path → SearchHit per item

    for idx, (path, _desc, physics_domain) in enumerate(items):
        item_hits: dict[str, Any] = {}  # path → SearchHit (dedup)

        # Query 1: description-based
        desc_query = item_desc_queries[idx]
        if desc_query:
            pre_emb = embed_cache.get(desc_query)
            try:
                hits = hybrid_dd_search(
                    gc,
                    desc_query,
                    node_category="quantity",
                    physics_domain=physics_domain,
                    k=search_k,
                    embedding=pre_emb,
                )
                for h in hits:
                    if h.path != path:
                        item_hits[h.path] = h
            except Exception:
                logger.debug(
                    "Hybrid search (description) failed for %s",
                    path,
                    exc_info=True,
                )

        # Query 2: path-text (no embedding needed — text-only inside)
        try:
            hits = hybrid_dd_search(
                gc,
                path,
                node_category="quantity",
                k=search_k,
            )
            for h in hits:
                if h.path != path and h.path not in item_hits:
                    item_hits[h.path] = h
        except Exception:
            logger.debug("Hybrid search (path) failed for %s", path, exc_info=True)

        per_item_hits.append(item_hits)

    # ── 4. Single batch HAS_STANDARD_NAME resolution ───────────────
    # Gather all unique hit paths across all items
    all_hit_paths: set[str] = set()
    for item_hits in per_item_hits:
        all_hit_paths.update(item_hits)

    sn_map: dict[str, str | None] = {}
    if all_hit_paths:
        try:
            rows = gc.query(
                """
                UNWIND $paths AS pid
                MATCH (n:IMASNode {id: pid})
                OPTIONAL MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                RETURN n.id AS path, sn.id AS sn_id
                """,
                paths=list(all_hit_paths),
            )
            for r in rows or []:
                sn_map[r["path"]] = r.get("sn_id")
        except Exception:
            logger.debug("HAS_STANDARD_NAME batch pre-resolution failed", exc_info=True)

    # ── 5. Build per-item result dicts ──────────────────────────────
    for idx, (_path, _desc, _domain) in enumerate(items):
        item_hits = per_item_hits[idx]

        if not item_hits:
            all_results.append([])
            continue

        sorted_hits = sorted(item_hits.values(), key=lambda h: h.score, reverse=True)[
            :max_results
        ]

        neighbours: list[dict] = []
        for h in sorted_hits:
            sn_id = sn_map.get(h.path)
            tag = f"name:{sn_id}" if sn_id else f"dd:{h.path}"
            doc = (h.documentation or h.description or "")[:300]
            neighbours.append(
                {
                    "tag": tag,
                    "path": h.path,
                    "ids": h.ids_name,
                    "unit": h.units or "",
                    "physics_domain": h.physics_domain or "",
                    "doc_short": doc,
                    "cocos_label": h.cocos_transformation_type or "",
                    "lifecycle": h.lifecycle_status or "",
                    "node_type": h.node_type or "",
                    "score": float(h.score) if h.score is not None else None,
                }
            )
        all_results.append(neighbours)

    return all_results


def _hybrid_search_neighbours(
    gc: Any,
    path: str,
    description: str | None = None,
    physics_domain: str | None = None,
    max_results: int = 15,
    search_k: int = 10,
) -> list[dict]:
    """Single-item shim around :func:`_hybrid_search_neighbours_batch`.

    .. deprecated::
        Prefer :func:`_hybrid_search_neighbours_batch` for multi-item
        workloads.  This shim exists for any un-migrated callers.
    """
    results = _hybrid_search_neighbours_batch(
        gc,
        [(path, description, physics_domain)],
        max_results=max_results,
        search_k=search_k,
    )
    return results[0] if results else []


# Cap for graph-relationship neighbour injection (per path).
_RELATED_MAX_RESULTS = 5


# Compose retry: on grammar/validation failure, retry with expanded context.
# Values resolved from settings accessors; module-level constants kept for
# backwards compatibility with any direct importers.
def _retry_attempts() -> int:
    from imas_codex.settings import get_sn_retry_attempts

    return get_sn_retry_attempts()


def _retry_k_expansion() -> int:
    from imas_codex.settings import get_sn_retry_k_expansion

    return get_sn_retry_k_expansion()


def _mark_vocab_gap_sources(
    batch: list[dict],
    error_detail: str,
    source_kind: str,
) -> None:
    """Mark sources after a vocab-gap validation failure.

    When batch_size == 1 the offending source is known, so it is marked
    ``vocab_gap_compose`` (terminal — won't be reclaimed).  When
    batch_size > 1 we cannot identify which source triggered the failure,
    so all sources are marked ``failed`` with a retry-friendly reason —
    they will be reclaimed on the next ``sn run`` and retried in a
    different (possibly smaller) batch.
    """
    try:
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.graph_ops import mark_source_skipped

        single_source = len(batch) == 1
        reason = "vocab_gap_compose" if single_source else "vocab_gap_batch"

        with GraphClient() as gc:
            for item in batch:
                sid = item.get("source_id") or item.get("id", "")
                if not sid:
                    continue
                if single_source:
                    mark_source_skipped(
                        gc,
                        sid,
                        reason=reason,
                        detail=error_detail[:300],
                        source_type=source_kind,
                    )
                else:
                    # Reset to extracted so the source is reclaimed next run
                    sns_id = f"{source_kind}:{sid}"
                    gc.query(
                        """
                        MATCH (sns:StandardNameSource {id: $sns_id})
                        SET sns.status = 'extracted',
                            sns.claimed_at = null,
                            sns.claim_token = null,
                            sns.skip_reason = $reason,
                            sns.skip_reason_detail = $detail
                        """,
                        sns_id=sns_id,
                        reason=reason,
                        detail=error_detail[:300],
                    )
    except Exception as exc:
        logger.debug("Failed to mark sources after vocab gap: %s", exc)


def _related_path_neighbours(
    gc: Any,
    path: str,
    *,
    max_results: int = _RELATED_MAX_RESULTS,
) -> list[dict]:
    """Fetch explicit graph-relationship neighbours for a DD path.

    Calls :func:`related_dd_search` to discover paths related via
    cluster membership, shared coordinates, matching units, identifier
    schemas, or COCOS transformation type.  Returns a compact list of
    dicts suitable for Jinja template injection.
    """
    from imas_codex.graph.dd_search import related_dd_search

    try:
        result = related_dd_search(
            gc,
            path,
            relationship_types="all",
            max_results=max_results,
        )
    except Exception:
        logger.debug("related_dd_search failed for %s", path, exc_info=True)
        return []

    if not result.hits:
        return []

    neighbours: list[dict] = []
    for hit in result.hits:
        entry: dict[str, Any] = {
            "path": hit.path,
            "ids": hit.ids,
            "relationship_type": hit.relationship_type,
            "via": hit.via,
        }
        if hit.doc:
            entry["doc"] = hit.doc
        if hit.physics_domain:
            entry["physics_domain"] = hit.physics_domain
        neighbours.append(entry)

    return neighbours


def _enrich_batch_items(items: list[dict]) -> None:
    """Enrich batch items with rich DD context from the graph.

    Fetches coordinate specs, COCOS info, identifier schemas, sibling
    fields, cross-IDS cluster siblings, hybrid-search neighbours, and
    version history for each item. Modifies items in-place.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        for item in items:
            path = item.get("path")
            if not path:
                continue

            rows = list(gc.query(_DD_CONTEXT_QUERY, path=path))
            if not rows:
                continue

            row = rows[0]

            # BUG 9 fix: propagate authoritative DD unit from HAS_UNIT into
            # the batch item so compose_batch's unit-safety skip and the
            # downstream persist see the real DD unit (Pa, m, m^-2.W, …)
            # rather than None.  StandardNameSource does not (yet) carry a
            # ``unit`` property in the schema, so we re-derive it from the
            # IMASNode at enrich time.  When HAS_UNIT is absent AND the
            # data type is numeric, set unit to "1" (ISN dimensionless
            # convention) — safety factor q, beta, mode numbers, etc.
            if not item.get("unit"):
                unit_from_rel = row.get("unit_from_rel")
                if unit_from_rel:
                    item["unit"] = unit_from_rel
                else:
                    # Dimensionless: numeric DD paths with no HAS_UNIT
                    # relationship are genuinely dimensionless (q, beta,
                    # mode numbers, efficiencies, fractions, etc.).
                    data_type = row.get("data_type", "")
                    _NUMERIC_PREFIXES = (
                        "FLT_",
                        "INT_",
                        "CPX_",
                    )
                    if data_type and any(
                        data_type.startswith(p) for p in _NUMERIC_PREFIXES
                    ):
                        item["unit"] = "1"

            # Apply unit overrides AFTER re-injecting the DD unit.
            # The override engine corrects upstream DD defects (e.g.,
            # unit vectors tagged 'm' instead of dimensionless '1').
            # Without this, the BUG 9 re-injection above would bypass
            # overrides that were applied during extraction.
            if item.get("unit") and path:
                from imas_codex.standard_names.unit_overrides import resolve_unit

                resolved, meta = resolve_unit(path, item["unit"])
                if meta and meta.get("rule") == "override":
                    item["unit"] = resolved
                elif meta and meta.get("rule") == "skip":
                    item["unit"] = None  # will be caught by compose safety

            # Coordinate context
            coords = []
            for key in ("coordinate1", "coordinate2", "coordinate3"):
                val = row.get(key)
                if val:
                    coords.append(val)
            if coords:
                item["coordinate_paths"] = coords

            timebase = row.get("timebase")
            if timebase:
                item["timebase"] = timebase

            # COCOS
            cocos_label = row.get("cocos_label")
            cocos_expr = row.get("cocos_expression")
            if cocos_label:
                item["cocos_label"] = cocos_label
            if cocos_expr:
                item["cocos_expression"] = cocos_expr

            # Lifecycle
            lifecycle = row.get("lifecycle_status")
            if lifecycle and lifecycle != "active":
                item["lifecycle_status"] = lifecycle

            # Parent description (complements parent_path from extraction)
            parent_desc = row.get("parent_description")
            if parent_desc and not item.get("parent_description"):
                item["parent_description"] = parent_desc

            # Identifier schema
            ident_name = row.get("identifier_schema_name")
            if ident_name:
                item["identifier_schema"] = ident_name
                ident_doc = row.get("identifier_schema_doc")
                if ident_doc:
                    item["identifier_schema_doc"] = ident_doc

                # Parse identifier enum values from JSON-encoded options
                raw_options = row.get("identifier_options")
                if raw_options:
                    try:
                        parsed = (
                            json.loads(raw_options)
                            if isinstance(raw_options, str)
                            else raw_options
                        )
                        if isinstance(parsed, list):
                            item["identifier_values"] = [
                                {
                                    "name": opt.get("name", ""),
                                    "index": opt.get("index", 0),
                                    "description": opt.get("description", ""),
                                }
                                for opt in parsed
                                if opt.get("name")
                            ]
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Sibling fields (same parent, different leaf paths)
            siblings = row.get("sibling_fields") or []
            if siblings and isinstance(siblings, list):
                valid = [s for s in siblings if s.get("path")]
                if valid:
                    item["sibling_fields"] = valid

            # Cross-IDS cluster siblings
            cross_rows = list(gc.query(_CROSS_IDS_QUERY, path=path))
            if cross_rows:
                clusters = []
                cross_ids_paths = []
                for cr in cross_rows:
                    label = cr.get("cluster_label")
                    members = cr.get("member_paths") or []
                    if label and members:
                        clusters.append(
                            {
                                "label": label,
                                "description": cr.get("cluster_description") or "",
                                "scope": cr.get("cluster_scope") or "",
                                "members": members,
                            }
                        )
                        cross_ids_paths.extend(members)
                if clusters:
                    item["clusters"] = clusters
                if cross_ids_paths:
                    # Deduplicate
                    seen = set()
                    unique = []
                    for p in cross_ids_paths:
                        if p not in seen:
                            seen.add(p)
                            unique.append(p)
                    item["cross_ids_paths"] = unique[:8]

            # Version history (COCOS/sign changes are most important)
            version_rows = list(gc.query(_VERSION_HISTORY_QUERY, path=path))
            if version_rows:
                valid_changes = []
                for vr in version_rows:
                    change_id = vr.get("change_id") or ""
                    change_type = vr.get("change_type") or ""
                    # Version is encoded in the ID: path:change_type:version
                    parts = change_id.rsplit(":", 1)
                    version = parts[-1] if len(parts) >= 2 else ""
                    if version and change_type:
                        valid_changes.append(
                            {"version": version, "change_type": change_type}
                        )
                if valid_changes:
                    item["version_history"] = valid_changes

            # Hybrid-search neighbours are batched outside the per-item loop
            # (see below). Skip per-item call here.

            # Graph-relationship neighbours (cluster, coordinate, unit,
            # identifier, COCOS — explicit graph edges, not vector search).
            related = _related_path_neighbours(gc, path)
            if related:
                item["related_neighbours"] = related

            # Error companion fields (uncertainty: _error_upper/lower/index)
            error_rows = list(gc.query(_ERROR_FIELDS_QUERY, path=path))
            if error_rows:
                error_fields = [
                    ef["error_path"] for ef in error_rows if ef.get("error_path")
                ]
                if error_fields:
                    item["error_fields"] = error_fields

        # ── Batch hybrid-search neighbours (single embed round-trip) ───
        batch_tuples: list[tuple[str, str | None, str | None]] = []
        batch_indices: list[int] = []  # index into items
        for idx, item in enumerate(items):
            path = item.get("path")
            if not path:
                continue
            batch_tuples.append(
                (path, item.get("description"), item.get("physics_domain"))
            )
            batch_indices.append(idx)

        if batch_tuples:
            hybrid_results = _hybrid_search_neighbours_batch(gc, batch_tuples)
            for bi, hr in zip(batch_indices, hybrid_results, strict=True):
                if hr:
                    items[bi]["hybrid_neighbours"] = hr


def _is_attachment_consistent(source_id: str, sn_name: str) -> tuple[bool, str]:
    """Reject attachments where the DD path tense disagrees with the SN tense.

    E.g. ``change_in_electron_density`` may not be attached to
    ``core_profiles/.../density`` (a base quantity, not a change). Symmetric:
    a base-quantity SN may not absorb an ``instant_changes`` path.
    """
    change_prefixes = (
        "change_in_",
        "tendency_of_",
        "rate_of_",
        "rate_of_change_of_",
        "time_derivative_of_",
    )
    change_path_tokens = ("instant_changes", "/change/", "_delta", "tendency_")
    sn_is_change = any(sn_name.startswith(p) for p in change_prefixes)
    path_is_change = any(t in source_id for t in change_path_tokens)
    if sn_is_change and not path_is_change:
        return False, (
            f"tense mismatch: SN '{sn_name}' is a change/rate but path "
            f"'{source_id}' is a base quantity"
        )
    if path_is_change and not sn_is_change:
        return False, (
            f"tense mismatch: path '{source_id}' is a change/rate but SN "
            f"'{sn_name}' is a base quantity"
        )
    return True, ""


def _process_attachments_core(
    attachments: list,
    wlog: logging.LoggerAdapter | logging.Logger,
) -> dict[str, int]:
    """Stateless attachment-processing helper used by both compose paths.

    Filters by tense consistency, writes ``HAS_STANDARD_NAME`` edges and
    appends ``source_paths``, and returns ``{"accepted", "rejected"}``
    counts.  Caller is responsible for any state-side stats updates.
    """
    from imas_codex.graph.client import GraphClient

    rejected: list[tuple[str, str, str]] = []
    accepted: list = []
    for a in attachments:
        ok, reason = _is_attachment_consistent(a.source_id, a.standard_name)
        if ok:
            accepted.append(a)
        else:
            rejected.append((a.source_id, a.standard_name, reason))

    for src, sn, why in rejected:
        wlog.warning("Rejected attachment %s → %s: %s", src, sn, why)

    counts = {"accepted": 0, "rejected": len(rejected)}
    if not accepted:
        return counts

    batch = [
        {"source_id": a.source_id, "standard_name": a.standard_name} for a in accepted
    ]

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.standard_name})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                WITH sn, 'dd:' + b.source_id AS uri
                WHERE NOT uri IN coalesce(sn.source_paths, [])
                SET sn.source_paths = coalesce(sn.source_paths, []) + uri
                """,
                batch=batch,
            )

        for a in accepted:
            wlog.info("Attached %s → %s (%s)", a.source_id, a.standard_name, a.reason)

        counts["accepted"] = len(accepted)
    except Exception:
        wlog.warning("Failed to process attachments", exc_info=True)

    return counts


def _process_attachments(
    attachments: list, state: StandardNameBuildState, wlog: logging.LoggerAdapter
) -> None:
    """Attach DD paths to existing standard names without regeneration.

    Linear-path wrapper around :func:`_process_attachments_core` that
    folds counts into the build state's stats dictionary.
    """
    counts = _process_attachments_core(attachments, wlog)
    if counts["accepted"]:
        prev = state.stats.get("attachments", 0)
        state.stats["attachments"] = prev + counts["accepted"]
    if counts["rejected"]:
        state.stats["attachments_rejected"] = (
            state.stats.get("attachments_rejected", 0) + counts["rejected"]
        )


# ---------------------------------------------------------------------------
# StandardNameSource status updaters (Phase 5: incremental tracking)
# ---------------------------------------------------------------------------


def _update_sources_after_compose(
    candidates: list[dict], source: str, wlog: logging.LoggerAdapter
) -> None:
    """Update StandardNameSource nodes to 'composed' after successful batch composition.

    Error-sibling candidates (minted deterministically, model=
    ``deterministic:dd_error_modifier``) are excluded — their source
    IMASNodes are not extracted as StandardNameSource, so linking them
    here would produce false-positive "linking gap" warnings.  Their
    provenance is carried on ``StandardName.source_paths``.
    """
    from imas_codex.graph.client import GraphClient

    source_type = "dd" if source == "dd" else "signals"
    batch = []
    for c in candidates:
        if c.get("model") == "deterministic:dd_error_modifier":
            continue  # error siblings have no StandardNameSource
        source_id = c.get("source_id")
        sn_id = c.get("id")
        if source_id and sn_id:
            batch.append(
                {
                    "sns_id": f"{source_type}:{source_id}",
                    "sn_id": sn_id,
                }
            )

    if not batch:
        return

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $batch AS b
                MATCH (sns:StandardNameSource {id: b.sns_id})
                MATCH (sn:StandardName {id: b.sn_id})
                SET sns.status = 'composed',
                    sns.composed_at = datetime(),
                    sns.produced_sn_id = sn.id
                MERGE (sns)-[:PRODUCED_NAME]->(sn)
                RETURN count(sns) AS linked
                """,
                batch=batch,
            )
            linked = result[0]["linked"] if result else 0
        if linked < len(batch):
            wlog.warning(
                "Compose-linking gap: %d/%d sources had no matching "
                "StandardName (edge not written, source still 'extracted')",
                len(batch) - linked,
                len(batch),
            )
        wlog.debug("Updated %d StandardNameSource nodes to composed", linked)
    except Exception:
        wlog.warning("Failed to update StandardNameSource status", exc_info=True)


def _update_sources_after_attach(
    attachments: list, source: str, wlog: logging.LoggerAdapter
) -> None:
    """Update StandardNameSource nodes to 'attached' status."""
    from imas_codex.graph.client import GraphClient

    source_type = "dd" if source == "dd" else "signals"
    batch = []
    for a in attachments:
        if a.source_id:
            batch.append(
                {
                    "sns_id": f"{source_type}:{a.source_id}",
                    "sn_id": a.standard_name,
                }
            )

    if not batch:
        return

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $batch AS b
                MATCH (sns:StandardNameSource {id: b.sns_id})
                MATCH (sn:StandardName {id: b.sn_id})
                SET sns.status = 'attached',
                    sns.composed_at = datetime(),
                    sns.produced_sn_id = sn.id
                MERGE (sns)-[:PRODUCED_NAME]->(sn)
                RETURN count(sns) AS linked
                """,
                batch=batch,
            )
            linked = result[0]["linked"] if result else 0
        if linked < len(batch):
            wlog.warning(
                "Attach-linking gap: %d/%d sources had no matching "
                "StandardName (edge not written, source still 'extracted')",
                len(batch) - linked,
                len(batch),
            )
        wlog.debug("Updated %d StandardNameSource nodes to attached", linked)
    except Exception:
        wlog.warning(
            "Failed to update StandardNameSource attachment status", exc_info=True
        )


def _update_sources_after_vocab_gap(
    vocab_gaps: list[dict], source: str, wlog: logging.LoggerAdapter
) -> None:
    """Update StandardNameSource nodes to 'vocab_gap' status.

    Gaps reported on pseudo segments (e.g. ``grammar_ambiguity``) are
    ignored — they are not real vocabulary gaps and must not retire
    the source from future composition attempts.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.segments import is_open_segment

    source_type = "dd" if source == "dd" else "signals"
    source_ids = []
    skipped_pseudo = 0
    for vg in vocab_gaps:
        if is_open_segment(vg.get("segment")):
            skipped_pseudo += 1
            continue
        sid = vg.get("source_id")
        if sid:
            source_ids.append(f"{source_type}:{sid}")

    if skipped_pseudo:
        wlog.debug(
            "Skipped vocab_gap status update for %d pseudo-segment gaps",
            skipped_pseudo,
        )

    if not source_ids:
        return

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS sns_id
                MATCH (sns:StandardNameSource {id: sns_id})
                SET sns.status = 'vocab_gap'
                """,
                ids=source_ids,
            )
        wlog.debug("Updated %d StandardNameSource nodes to vocab_gap", len(source_ids))
    except Exception:
        wlog.warning(
            "Failed to update StandardNameSource vocab_gap status", exc_info=True
        )


def _update_sources_after_skip(
    skipped_ids: list[str],
    source: str,
    wlog: logging.LoggerAdapter | logging.Logger,
) -> None:
    """Clear claims and mark StandardNameSource nodes ``status='skipped'``.

    The compose LLM may decide a source is not a physics quantity and emit
    its ``source_id`` in ``result.skipped``.  Without this transition the
    source remains ``claimed`` forever and gets re-processed every run.
    """
    from imas_codex.graph.client import GraphClient

    if not skipped_ids:
        return

    source_type = "dd" if source == "dd" else "signals"
    sns_ids = [f"{source_type}:{sid}" for sid in skipped_ids if sid]
    if not sns_ids:
        return

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS sns_id
                MATCH (sns:StandardNameSource {id: sns_id})
                SET sns.status        = 'skipped',
                    sns.claim_token   = null,
                    sns.claimed_at    = null,
                    sns.skipped_at    = datetime()
                """,
                ids=sns_ids,
            )
        wlog.debug("Marked %d StandardNameSource nodes as skipped", len(sns_ids))
    except Exception:
        wlog.warning("Failed to update StandardNameSource skip status", exc_info=True)


def _search_reference_exemplars(
    items: list[dict],
    domains: list[str],
    *,
    k: int = 5,
) -> list[dict]:
    """Synthesise a query from batch items and return reference SN exemplars.

    Builds a query string from up to three item descriptions, calls
    :func:`search_standard_names_with_documentation`, and excludes any SN whose
    ``id`` matches an item already present in the batch (when items carry an
    ``existing_name``).  *domains* filters via embedding-quality semantics
    only (the search itself filters by ``validation_status='valid'``).

    Returns an empty list if no descriptions are available or the search
    backend is unavailable.
    """
    from imas_codex.standard_names.search import (
        search_standard_names_with_documentation,
    )

    desc_snippets = [
        item.get("description", "") for item in items[:3] if item.get("description")
    ]
    if not desc_snippets:
        return []
    synth_query = "; ".join(desc_snippets)

    # Real SN ids of items already present in the batch (cluster-aware
    # nearby retrieval injects these); avoid feeding the LLM duplicates.
    exclude_ids = sorted(
        {item.get("existing_name") for item in items if item.get("existing_name")}
    )

    try:
        return search_standard_names_with_documentation(
            synth_query, k=k, exclude_ids=exclude_ids or None
        )
    except Exception:
        logger.debug("Reference exemplar search failed", exc_info=True)
        return []


# Opus model for L7 borderline revision pass
_L7_REVISION_MODEL = DEFAULT_ESCALATION_MODEL
_L7_MIN_REMAINING_BUDGET = 0.50  # Skip L7 if remaining budget < this


async def _grammar_retry(
    original_name: str,
    parse_error: str,
    model: str,
    acall_fn,
) -> tuple[str | None, float, int, int]:
    """L6: Single grammar-failure re-prompt.

    Asks the LLM to revise a name that failed grammar round-trip,
    providing the parse error and a grammar cheat-sheet fragment.

    Returns ``(revised_name | None, cost_usd, tokens_in, tokens_out)``.
    """
    from pydantic import BaseModel, Field

    class GrammarRetryResponse(BaseModel):
        revised_name: str = Field(
            description="The revised standard name that passes grammar parsing"
        )
        explanation: str = Field(description="Brief explanation of the fix")

    retry_prompt = (
        f"The standard name `{original_name}` failed grammar parsing with error:\n"
        f"  {parse_error}\n\n"
        "Revise ONLY the name to pass the grammar round-trip. Rules:\n"
        "- Pattern: [subject_][physical_base|geometric_base][_component][_position][_process][_object]\n"
        "- ALL segments are CLOSED including physical_base (80 registered tokens).\n"
        "  Only use tokens from the registries. If no token fits, emit a vocab_gap.\n"
        "- Closed-vocabulary tokens (e.g. toroidal, parallel, thermal,\n"
        "  e_cross_b_drift, normalized, fast_ion, volume_averaged) MUST be\n"
        "  placed in their closed segment slot. Never absorb them into\n"
        "  physical_base.\n"
        "- No abbreviations, no provenance verbs, no unit suffixes\n"
        "- Return the MINIMAL fix — keep the name as close to the original as possible.\n"
    )

    try:
        llm_out = await acall_fn(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            response_model=GrammarRetryResponse,
            service="standard-names",
        )
        result, _cost, _tokens = llm_out
        return (
            result.revised_name if result else None,
            float(_cost or 0.0),
            getattr(llm_out, "input_tokens", 0) or 0,
            getattr(llm_out, "output_tokens", 0) or 0,
        )
    except Exception:
        return None, 0.0, 0, 0


async def _opus_revise_candidate(
    candidate: dict,
    domain_vocabulary: str,
    reviewer_themes: list[str],
    acall_fn,
) -> tuple[str | None, float, int, int]:
    """L7: Revision pass for candidates using Opus model.

    Returns ``(revised_name_or_None, cost_usd, tokens_in, tokens_out)``.
    The cost is always returned so callers can account for it even when
    the revision is discarded.
    """
    from pydantic import BaseModel, Field

    class OpusRevisionResponse(BaseModel):
        revised_name: str = Field(description="Improved standard name")
        explanation: str = Field(description="Why this revision is better")

    name = candidate.get("id", "")
    reason = candidate.get("reason", "")
    description = candidate.get("description", "")

    prompt_parts = [
        "A standard name candidate requires revision:",
        f"  Name: `{name}`",
        f"  Description: {description}",
        f"  Reason: {reason}",
        "",
        "Revise the name to be more precise, following ISN grammar rules.",
        "Pattern: [subject_][physical_base|geometric_base][_component][_position][_process][_object]",
        "",
    ]

    if domain_vocabulary:
        prompt_parts.append("Domain vocabulary (prefer these terms):")
        # Include first 10 lines of vocabulary
        for line in domain_vocabulary.split("\n")[:10]:
            prompt_parts.append(f"  {line}")
        prompt_parts.append("")

    if reviewer_themes:
        prompt_parts.append("Reviewer feedback themes to address:")
        for theme in reviewer_themes[:5]:
            prompt_parts.append(f"  - {theme}")
        prompt_parts.append("")

    prompt_parts.append("Return a revised name only if you can do CLEARLY better.")

    try:
        llm_out = await acall_fn(
            model=_L7_REVISION_MODEL,
            messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
            response_model=OpusRevisionResponse,
            service="standard-names",
        )
        result, _cost, _tokens = llm_out
        _ti = getattr(llm_out, "input_tokens", 0) or 0
        _to = getattr(llm_out, "output_tokens", 0) or 0
        if result:
            return result.revised_name, float(_cost or 0.0), _ti, _to
        return None, float(_cost or 0.0), _ti, _to
    except Exception:
        return None, 0.0, 0, 0


async def compose_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """LLM-generate standard names from extracted batches.

    Uses acall_llm_structured() with system/user prompt split for
    prompt caching.  Runs batches concurrently with semaphore.
    Results stored in ``state.composed``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_compose_worker")

    total_items = sum(len(b.items) for b in state.extracted)

    if state.dry_run:
        wlog.info("Dry run — skipping composition for %d items", total_items)
        state.compose_stats.total = total_items
        state.compose_stats.processed = total_items
        state.stats["compose_skipped"] = True
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    if not state.extracted:
        wlog.info("No batches to compose — skipping")
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameComposeBatch

    model = state.compose_model or get_model("sn-compose")
    context = build_compose_context()

    # Enrich batch items with rich DD context (coordinate specs, COCOS, siblings,
    # cross-IDS paths, version history)
    def _enrich_all_batches():
        for batch in state.extracted:
            _enrich_batch_items(batch.items)

    await asyncio.to_thread(_enrich_all_batches)
    wlog.info("Enriched batch items with DD context")

    # Propagate COCOS metadata to state (for downstream phases)
    if state.extracted:
        first_batch = state.extracted[0]
        state.dd_version = first_batch.dd_version
        state.cocos_version = first_batch.cocos_version
        state.cocos_params = first_batch.cocos_params

    # Pre-fetch IDS-level context for each unique IDS across batches
    ids_context_cache: dict[str, dict | None] = {}

    def _prefetch_ids_context():
        for batch in state.extracted:
            # Derive IDS names from item paths (first segment)
            ids_names = {
                item["path"].split("/")[0]
                for item in batch.items
                if item.get("path") and "/" in item["path"]
            }
            for ids_name in ids_names:
                if ids_name not in ids_context_cache:
                    ids_context_cache[ids_name] = _enrich_ids_context(ids_name)

    await asyncio.to_thread(_prefetch_ids_context)
    wlog.info("Fetched IDS context for %d IDS(s)", len(ids_context_cache))

    # Render system prompt once (cached via prompt caching)
    # Inject COCOS context for system prompt (all batches share one DD version)
    if state.extracted:
        first_batch = state.extracted[0]
        if first_batch.cocos_version:
            context["cocos_version"] = first_batch.cocos_version
            context["dd_version"] = first_batch.dd_version
            if first_batch.cocos_params:
                context["cocos_sigma_bp"] = first_batch.cocos_params.get("sigma_bp")
                context["cocos_e_bp"] = first_batch.cocos_params.get("e_bp")
                context["cocos_sigma_r_phi_z"] = first_batch.cocos_params.get(
                    "sigma_r_phi_z"
                )
                context["cocos_sigma_rho_theta_phi"] = first_batch.cocos_params.get(
                    "sigma_rho_theta_phi"
                )

    # --- L1: Domain-vocabulary pre-seeding ---
    # Inject validated domain vocabulary into system prompt context
    from imas_codex.standard_names.context import build_domain_vocabulary_preseed

    domain_vocab = ""
    if state.domain_filter:
        domain_vocab = await asyncio.to_thread(
            build_domain_vocabulary_preseed, state.domain_filter
        )
        if domain_vocab:
            wlog.info(
                "L1: Injected domain vocabulary preseed for %s", state.domain_filter
            )
    context["domain_vocabulary"] = domain_vocab

    # --- L4: Reviewer-theme extraction ---
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    if state.domain_filter:
        reviewer_themes = await asyncio.to_thread(
            extract_reviewer_themes, state.domain_filter
        )
        if reviewer_themes:
            wlog.info(
                "L4: Extracted %d reviewer themes for %s",
                len(reviewer_themes),
                state.domain_filter,
            )
    context["reviewer_themes"] = reviewer_themes

    # --- K3: Scored-example injection ---
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_compose_examples

    # Derive physics_domains from domain_filter and batch items
    batch_domains: list[str] = []
    if state.domain_filter:
        batch_domains = [state.domain_filter]
    else:
        # Collect unique domains from all batch items
        _domains = {
            item.get("physics_domain")
            for batch in state.extracted
            for item in batch.items
            if item.get("physics_domain")
        }
        batch_domains = sorted(_domains)

    def _load_scored_examples() -> list[dict]:
        with GraphClient() as gc:
            return load_compose_examples(gc, physics_domains=batch_domains, axis="name")

    compose_scored_examples = await asyncio.to_thread(_load_scored_examples)
    if compose_scored_examples:
        wlog.info(
            "K3: Loaded %d scored examples for compose (domains=%s)",
            len(compose_scored_examples),
            batch_domains or "all",
        )
    context["compose_scored_examples"] = compose_scored_examples

    # --- NC rules injection ---
    # Load naming-consistency rules from YAML so the system prompt
    # {% include "_nc_rules.md" %} block renders the full rule set.
    from imas_codex.llm.prompt_loader import load_prompt_config

    try:
        _rules_cfg = load_prompt_config("sn_composition_rules")
        context["composition_rules"] = _rules_cfg.get("composition_rules", [])
    except Exception:
        wlog.debug("NC rules YAML not found — skipping injection")
        context["composition_rules"] = []

    system_prompt = render_prompt("sn/generate_name_system", context)

    wlog.info(
        "Composing standard names for %d items in %d batches (model=%s)",
        total_items,
        len(state.extracted),
        model,
    )
    state.compose_stats.total = total_items

    from imas_codex.settings import get_compose_concurrency

    async def _compose_batch_body(
        batch: ExtractionBatch, lease: BudgetLease | None
    ) -> list[dict]:
        # Search for nearby existing names (per-item for better relevance)
        _nearby_seen: set[str] = set()
        nearby: list[dict] = []
        _PER_ITEM_K = 5
        _NEARBY_CAP = 30
        for item in batch.items:
            hint = item.get("description") or item.get("path", "")
            item_results = _search_nearby_names(hint, k=_PER_ITEM_K)
            for nr in item_results:
                nid = nr.get("id", "")
                if nid and nid not in _nearby_seen:
                    _nearby_seen.add(nid)
                    nearby.append(nr)
                    if len(nearby) >= _NEARBY_CAP:
                        break
            if len(nearby) >= _NEARBY_CAP:
                break

        # IDS-level context — collect for each IDS present in batch
        ids_names = sorted(
            {
                item["path"].split("/")[0]
                for item in batch.items
                if item.get("path") and "/" in item["path"]
            }
        )
        ids_contexts = []
        for iname in ids_names:
            info = ids_context_cache.get(iname)
            if info:
                ids_contexts.append({"ids_name": iname, **info})

        # Pre-render COCOS guidance for items
        if batch.cocos_params:
            from imas_codex.standard_names.context import render_cocos_guidance

            for item in batch.items:
                cocos_label = item.get("cocos_label")
                if cocos_label:
                    item["cocos_guidance"] = render_cocos_guidance(
                        cocos_label, batch.cocos_params
                    )

        # --- Rate-quantity detection ---
        # When DD documentation indicates a rate/time-derivative, inject a
        # hard constraint so the LLM uses tendency_of_/change_in_ prefix
        # and writes a consistent description (prevents instant_change_*
        # names and name/description verb drift).
        import re as _re

        _RATE_DOC_PATTERNS = _re.compile(
            r"\b(instantaneous change|signed change|rate of change"
            r"|time derivative|per unit time|instant change|d/dt"
            r"|tendency of|time-rate)\b",
            _re.IGNORECASE,
        )
        for item in batch.items:
            haystack = " ".join(
                str(item.get(k, "") or "") for k in ("description", "documentation")
            )
            if _RATE_DOC_PATTERNS.search(haystack):
                item["rate_hint"] = True

        # --- L2: Reference SN few-shot retrieval ---
        # Synthesize query from first 3 path descriptions
        reference_exemplars: list[dict] = []
        try:
            from imas_codex.standard_names.search import (
                search_standard_names_with_documentation,
            )

            desc_snippets = [
                item.get("description", "")
                for item in batch.items[:3]
                if item.get("description")
            ]
            if desc_snippets:
                synth_query = "; ".join(desc_snippets)
                # Exclude names already in this batch's candidate IDs
                batch_ids = [
                    item.get("path", "").replace("/", "_") for item in batch.items
                ]
                reference_exemplars = await asyncio.to_thread(
                    search_standard_names_with_documentation,
                    synth_query,
                    k=5,
                    exclude_ids=batch_ids,
                )
        except Exception:
            wlog.debug("L2: Reference exemplar search failed", exc_info=True)

        user_context = {
            "items": batch.items,
            "ids_name": batch.group_key,
            "ids_contexts": ids_contexts,
            "existing_names": sorted(batch.existing_names)[:200],
            "cluster_context": batch.context,
            "nearby_existing_names": nearby,
            "reference_exemplars": reference_exemplars,
            "cocos_version": batch.cocos_version,
            "dd_version": batch.dd_version,
        }
        # Name-only batches (Workstream 2a) render a leaner user prompt
        # that trades per-item cluster siblings / COCOS blocks / sibling
        # fields for a "identify natural sub-groups, then name" directive.
        # System prompt and per-candidate L6/L7 logic are unchanged so
        # prompt caching and grammar safety stay intact.
        prompt_template = (
            "sn/generate_name_dd_names"
            if batch.mode == "names"
            else "sn/generate_name_dd"
        )
        user_prompt = render_prompt(prompt_template, {**context, **user_context})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # --- Delta H: bounded retry loop for failed compositions ---
        # Accumulate full LLMResult fields across retries so the batch's
        # per-candidate cost/token attribution is accurate.  Compose is
        # unified with review/enrich here: every LLM call site extracts
        # the same cost/token fields from LLMResult and writes them to
        # the graph for single-source cost observability.
        _total_compose_cost = 0.0
        _total_tokens_in = 0
        _total_tokens_out = 0
        _total_cache_read = 0
        _total_cache_creation = 0
        _max_retries = _retry_attempts()
        for _compose_attempt in range(_max_retries + 1):
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-compose"),
                )
            except (ValueError, Exception) as compose_exc:
                _exc_str = str(compose_exc)
                if "not a registered" in _exc_str:
                    logger.warning(
                        "Compose: vocab gap validation error — marking "
                        "%d sources as vocab_gap: %s",
                        len(batch),
                        _exc_str[:200],
                    )
                    _mark_vocab_gap_sources(batch.items, _exc_str, "dd")
                    return []
                raise
            result, cost, tokens = llm_out
            _total_compose_cost += cost
            _total_tokens_in += getattr(llm_out, "input_tokens", 0) or 0
            _total_tokens_out += getattr(llm_out, "output_tokens", 0) or 0
            _total_cache_read += getattr(llm_out, "cache_read_tokens", 0) or 0
            _total_cache_creation += getattr(llm_out, "cache_creation_tokens", 0) or 0

            # Charge actual LLM cost to budget lease via typed event.
            # charge_event uses soft-charge semantics: the LLM has
            # already been paid for, so spend is always recorded.
            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=getattr(llm_out, "cache_read_tokens", 0) or 0,
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=tuple(
                        c.compose_name() for c in (result.candidates if result else [])
                    ),
                    batch_id=batch.group_key,
                    phase=getattr(state, "budget_phase_tag", "") or "generate_name",
                    service="standard-names",
                )
                _charge = lease.charge_event(cost, _event)
                if state.loop_stats is not None:
                    state.loop_stats.processed += max(1, len(result.candidates))
                    state.loop_stats.cost += cost
                    # Stream one item per candidate so the user sees names
                    # rotate through with grammar composition.
                    _items: list[dict[str, Any]] = []
                    for _cand in result.candidates[:50]:
                        _segs = []
                        if _cand.base_token:
                            _segs.append(f"base={_cand.base_token}")
                        for _q in _cand.qualifiers:
                            _segs.append(f"qualifier={_q}")
                        if _cand.projection_axis:
                            _segs.append(f"projection={_cand.projection_axis}")
                        if _cand.process_token:
                            _segs.append(f"process={_cand.process_token}")
                        if _cand.operator_token:
                            _segs.append(f"operator={_cand.operator_token}")
                        _desc = (
                            "  ".join(_segs) if _segs else ((_cand.reason or "")[:80])
                        )
                        try:
                            _composed = _cand.compose_name()
                        except Exception:
                            _composed = f"[IR error] base={_cand.base_token}"
                        _items.append(
                            {
                                "primary_text": _composed,
                                "primary_text_style": "white",
                                "description": _desc,
                            }
                        )
                    if not _items:
                        _items.append(
                            {
                                "primary_text": (
                                    f"sn={_event.sn_ids[0]}"
                                    if _event.sn_ids
                                    else f"batch={_event.batch_id}"
                                ),
                                "primary_text_style": "white",
                                "description": batch.group_key,
                            }
                        )
                    state.loop_stats.stream_queue.add(_items)
                if _charge.overspend > 0:
                    wlog.warning(
                        "Compose batch %s overspent reservation by $%.4f "
                        "(batch cost $%.4f); budget tracking will report overrun",
                        batch.group_key,
                        _charge.overspend,
                        cost,
                    )

            # Quick grammar round-trip check on all candidates
            _grammar_failures: list[str] = []
            try:
                from imas_standard_names.grammar import parse_standard_name

                for c in result.candidates:
                    try:
                        _name = c.compose_name()
                        parse_standard_name(_name)
                    except Exception:
                        try:
                            _grammar_failures.append(c.compose_name())
                        except Exception:
                            _grammar_failures.append(c.source_id)
            except ImportError:
                pass  # ISN not installed — skip check

            if not _grammar_failures or _compose_attempt >= _max_retries:
                break

            # Re-enrich items with expanded hybrid search for retry
            wlog.info(
                "Composition retry %d/%d: %d grammar failures (%s) "
                "— re-composing with expanded DD context",
                _compose_attempt + 1,
                _max_retries,
                len(_grammar_failures),
                ", ".join(_grammar_failures[:3]),
            )

            def _re_enrich_expanded():
                from imas_codex.graph.client import GraphClient

                with GraphClient() as gc:
                    batch_tuples = [
                        (
                            item.get("path"),
                            item.get("description"),
                            item.get("physics_domain"),
                        )
                        for item in batch.items
                        if item.get("path")
                    ]
                    if batch_tuples:
                        hybrid_results = _hybrid_search_neighbours_batch(
                            gc,
                            batch_tuples,
                            search_k=_retry_k_expansion(),
                        )
                        item_idx = 0
                        for item in batch.items:
                            if not item.get("path"):
                                continue
                            if hybrid_results[item_idx]:
                                item["hybrid_neighbours"] = hybrid_results[item_idx]
                            item_idx += 1

            await asyncio.to_thread(_re_enrich_expanded)

            _retry_reason = (
                f"Previous attempt failed: grammar round-trip failed for "
                f"{', '.join(_grammar_failures[:3])}. Consider expanded "
                f"neighbour context and produce a different name."
            )
            retry_render_ctx = {
                **context,
                **user_context,
                "retry_reason": _retry_reason,
            }
            user_prompt = render_prompt(prompt_template, retry_render_ctx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        state.compose_stats.cost += _total_compose_cost
        state.compose_stats.processed += len(batch.items)
        state.compose_stats.record_batch(len(batch.items))

        candidates = []
        for c in result.candidates:
            # Find the matching source item to get authoritative fields
            source_item = next(
                (item for item in batch.items if item.get("path") == c.source_id),
                None,
            )
            # Inject unit from DD (authoritative, not LLM output).
            #
            # User invariant (AGENTS.md "Unit safety"): the LLM never
            # decides units; the DD source must provide one.
            # - "1": dimensionless (ISN convention) — valid
            # - "-": DD dimensionless marker — normalize to "1"
            # - "mixed": heterogeneous dimensions — skip
            # - None/empty: enrichment couldn't resolve — skip
            raw_unit = source_item.get("unit") if source_item else None
            if raw_unit == "-":
                raw_unit = "1"  # normalize DD marker to ISN convention
            if raw_unit in ("mixed", None, ""):
                _skip_reason = (
                    "dd_unit_mixed_non_standard"
                    if raw_unit == "mixed"
                    else "dd_unit_unresolvable"
                )
                wlog.warning(
                    "compose: %s, skipping source=%s raw_unit=%r",
                    _skip_reason,
                    c.source_id,
                    raw_unit,
                )
                if c.source_id and not state.dry_run:
                    try:
                        from imas_codex.graph.client import GraphClient
                        from imas_codex.standard_names.graph_ops import (
                            mark_source_skipped,
                        )

                        _prefix = "dd" if state.source == "dd" else "signals"
                        with GraphClient() as _gc:
                            mark_source_skipped(
                                _gc,
                                c.source_id,
                                reason=_skip_reason,
                                detail=str(raw_unit),
                                source_type=_prefix,
                            )
                    except Exception as exc:
                        wlog.debug(
                            "Failed to mark source %s skipped: %s",
                            c.source_id,
                            exc,
                        )
                state.compose_stats.errors += 1
                continue
            unit = raw_unit

            # Inject physics_domain from DD (authoritative, like unit).
            # Post-refactor: scalar primary + source_domains list.
            raw_domain = source_item.get("physics_domain") if source_item else None
            physics_domain = raw_domain or None
            source_domains = [raw_domain] if raw_domain else []

            # Inject COCOS metadata from DD (authoritative, like unit)
            cocos_type = source_item.get("cocos_label") if source_item else None

            # Normalize name via grammar round-trip BEFORE persist
            # to avoid duplicate nodes if validate would rename
            name_id = normalize_spelling(c.compose_name())

            # W4b: Pre-validation gate — reject malformed LLM output
            # before it reaches MERGE.
            _well_formed, _reject_reason = is_well_formed_candidate(name_id)
            if not _well_formed:
                state.stats["compose_pre_validation_rejects"] = (
                    state.stats.get("compose_pre_validation_rejects", 0) + 1
                )
                wlog.warning(
                    "Pre-validation reject: %r (%s)",
                    name_id[:80],
                    _reject_reason,
                )
                # Mark source as failed (if identifiable)
                if c.source_id and not state.dry_run:
                    try:
                        from imas_codex.graph.client import GraphClient

                        _prefix = "dd" if state.source == "dd" else "signals"
                        with GraphClient() as gc:
                            gc.query(
                                "MATCH (sns:StandardNameSource {id: $id}) "
                                "SET sns.status = 'failed', "
                                "    sns.claimed_at = null, "
                                "    sns.claim_token = null",
                                id=f"{_prefix}:{c.source_id}",
                            )
                    except Exception:
                        wlog.debug("Failed to mark source %s as failed", c.source_id)
                continue

            grammar_failed = False
            try:
                from imas_standard_names.grammar import (
                    compose_standard_name,
                    parse_standard_name,
                )

                try:
                    parsed = parse_standard_name(name_id)
                except Exception as _order_exc:
                    # ISN ≥rc35 strict-order rejections carry the unique
                    # canonical spelling — adopt it deterministically instead
                    # of spending an L6 LLM retry. Anything else re-raises to
                    # the L6 path below.
                    _canonical = getattr(_order_exc, "canonical_form", None)
                    if not _canonical:
                        raise
                    wlog.debug(
                        "Canonical-order normalization: %r → %r",
                        name_id,
                        _canonical,
                    )
                    name_id = _canonical
                    parsed = parse_standard_name(name_id)
                normalized = compose_standard_name(parsed)
                if normalized != name_id:
                    wlog.debug(
                        "Pre-persist normalization: %r → %r", name_id, normalized
                    )
                    name_id = normalized

                # Fix grammar library bug where parse→compose doubles
                # adjacent tokens (e.g. magnetic_field_probe →
                # magnetic_magnetic_field_probe).  Safe because no
                # legitimate standard name has adjacent duplicate tokens.
                name_id = _dedup_adjacent_tokens(name_id, wlog)
            except Exception as gram_exc:
                grammar_failed = True
                wlog.debug("Grammar parse failed for %r — attempting L6 retry", name_id)

                # --- L6: Grammar-failure re-prompt (single retry) ---
                state.grammar_retries += 1
                try:
                    retry_name, _l6_cost, _l6_ti, _l6_to = await _grammar_retry(
                        name_id, str(gram_exc), model, acall_llm_structured
                    )
                    if lease and _l6_cost > 0:
                        _l6_event = LLMCostEvent(
                            model=model,
                            tokens_in=_l6_ti,
                            tokens_out=_l6_to,
                            sn_ids=(name_id,),
                            batch_id=f"{batch.group_key}-grammar-retry",
                            phase=(
                                getattr(state, "budget_phase_tag", "")
                                or "generate_name"
                            ),
                            service="standard-names",
                        )
                        lease.charge_event(_l6_cost, _l6_event)
                        if state.loop_stats is not None:
                            state.loop_stats.cost += _l6_cost
                            state.loop_stats.stream_queue.add(
                                [
                                    {
                                        "primary_text": f"sn={name_id}",
                                        "primary_text_style": "white",
                                        "description": f"{batch.group_key}-L6",
                                    }
                                ]
                            )
                    if retry_name and retry_name != name_id:
                        # Verify the retry result actually parses
                        parsed = parse_standard_name(retry_name)
                        normalized = compose_standard_name(parsed)
                        name_id = _dedup_adjacent_tokens(normalized, wlog)
                        grammar_failed = False
                        state.grammar_retries_succeeded += 1
                        wlog.info(
                            "L6: Grammar retry succeeded: %r → %r",
                            c.compose_name(),
                            name_id,
                        )
                except Exception:
                    wlog.debug("L6: Grammar retry also failed for %r", name_id)

            candidates.append(
                {
                    "id": name_id,
                    "source_types": ["dd"] if state.source == "dd" else ["signals"],
                    "source_id": c.source_id,
                    "kind": c.kind,
                    "source_paths": [
                        encode_source_path(
                            "dd" if state.source == "dd" else "signals", p
                        )
                        for p in (c.dd_paths or [])
                    ],
                    "reason": c.reason,
                    "unit": unit,
                    "physics_domain": physics_domain,
                    "source_domains": source_domains,
                    "cocos_transformation_type": cocos_type,
                    "cocos": batch.cocos_version,
                    "dd_version": batch.dd_version,
                    # L6: track grammar retry exhaustion
                    **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
                }
            )

            # --- B9: Mint error siblings deterministically ---
            # If the parent has HAS_ERROR edges, mint uncertainty
            # modifier siblings without LLM calls.
            if (
                not grammar_failed
                and source_item
                and source_item.get("has_errors")
                and source_item.get("error_node_ids")
            ):
                from imas_codex.standard_names.error_siblings import (
                    mint_error_siblings,
                )

                siblings = mint_error_siblings(
                    name_id,
                    error_node_ids=source_item["error_node_ids"],
                    unit=unit,
                    physics_domain=physics_domain,
                    cocos_type=cocos_type,
                    cocos_version=batch.cocos_version,
                    dd_version=batch.dd_version,
                )
                if siblings:
                    for s in siblings:
                        s["_from_error_sibling"] = True
                    candidates.extend(siblings)
                    state.error_siblings_minted = getattr(
                        state, "error_siblings_minted", 0
                    ) + len(siblings)
                    wlog.debug(
                        "B9: Minted %d error siblings for parent %r",
                        len(siblings),
                        name_id,
                    )

        # Collect vocab gaps and persist immediately
        if result.vocab_gaps:
            gap_dicts = []
            for vg in result.vocab_gaps:
                gap_dict = {
                    "source_id": vg.source_id,
                    "segment": vg.segment,
                    "token": vg.token,
                    "reason": vg.reason,
                }
                state.stats.setdefault("vocab_gaps", []).append(gap_dict)
                gap_dicts.append(gap_dict)

            # Persist to graph immediately so gaps survive cost-limit interruption
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            source_type = "dd" if state.source == "dd" else "signals"
            await asyncio.to_thread(write_vocab_gaps, gap_dicts, source_type)
            wlog.debug("Persisted %d vocab gaps to graph", len(gap_dicts))

            if not state.dry_run:
                _update_sources_after_vocab_gap(gap_dicts, state.source, wlog)

        # Auto-detect novel physical_base tokens in composed candidates (W29).
        # These are surfaced as VocabGap nodes for ISN review without
        # requiring the LLM to emit explicit vocab_gap exits.
        if candidates:
            auto_gaps = _auto_detect_physical_base_gaps(candidates)
            if auto_gaps:
                # Dedupe against LLM-emitted gaps (by source_id+segment+token)
                existing_keys = {
                    (g["source_id"], g["segment"], g["token"])
                    for g in state.stats.get("vocab_gaps", [])
                }
                novel = [
                    g
                    for g in auto_gaps
                    if (g["source_id"], g["segment"], g["token"]) not in existing_keys
                ]
                if novel:
                    from imas_codex.standard_names.graph_ops import write_vocab_gaps

                    source_type = "dd" if state.source == "dd" else "signals"
                    await asyncio.to_thread(
                        write_vocab_gaps,
                        novel,
                        source_type,
                        skip_segment_filter=True,
                    )
                    state.stats.setdefault("vocab_gaps", []).extend(novel)
                    wlog.info("Auto-detected %d novel physical_base gaps", len(novel))
        if result.attachments:
            _process_attachments(result.attachments, state, wlog)
            if not state.dry_run:
                _update_sources_after_attach(result.attachments, state.source, wlog)

        # --- GRAPH-STATE-MACHINE: persist immediately per batch ---
        # This ensures completed batches survive cost-limit cancellation.
        if candidates:
            from datetime import UTC, datetime

            from imas_codex.standard_names.graph_ops import persist_generated_name_batch

            # Pro-rata per-candidate cost/token attribution.  One LLM
            # batch call produces N candidates; divide cost and tokens
            # so SUM(sn.llm_cost) aggregates back to the batch total and
            # callers can extract spend directly from the graph instead
            # of scraping logs.  Error-sibling candidates minted
            # deterministically below are excluded from the denominator
            # (they carry no LLM cost).
            _llm_candidates = [
                c for c in candidates if not c.get("_from_error_sibling")
            ]
            _n = max(len(_llm_candidates), 1)
            _pro_rata_cost = _total_compose_cost / _n
            _pro_rata_in = _total_tokens_in // _n
            _pro_rata_out = _total_tokens_out // _n
            _pro_rata_cache_r = _total_cache_read // _n
            _pro_rata_cache_w = _total_cache_creation // _n
            _llm_at = datetime.now(UTC).isoformat()
            for c in candidates:
                if c.get("_from_error_sibling"):
                    continue
                c["llm_cost"] = _pro_rata_cost
                c["llm_model"] = model
                c["llm_service"] = "standard-names"
                c["llm_at"] = _llm_at
                c["llm_tokens_in"] = _pro_rata_in
                c["llm_tokens_out"] = _pro_rata_out
                c["llm_tokens_cached_read"] = _pro_rata_cache_r
                c["llm_tokens_cached_write"] = _pro_rata_cache_w

            # Tag regen-path candidates so persist increments regen_count
            if state.regen:
                for c in candidates:
                    c["regen_increment"] = True

            written = await asyncio.to_thread(
                persist_generated_name_batch,
                candidates,
                compose_model=model,
                dd_version=batch.dd_version,
                cocos_version=batch.cocos_version,
                run_id=getattr(state.budget_manager, "run_id", None)
                if state.budget_manager
                else None,
            )
            wlog.debug("Persisted %d names from batch %s", written, batch.group_key)

        # Update StandardNameSource nodes to composed status
        if candidates and not state.dry_run:
            _update_sources_after_compose(candidates, state.source, wlog)

        wlog.info(
            "Batch %s: %d composed, %d attached, %d skipped (cost=$%.4f)",
            batch.group_key,
            len(result.candidates),
            len(result.attachments),
            len(result.skipped),
            cost,
        )
        # Stream batch completion to progress display
        attach_label = (
            f"+{len(result.attachments)} attached  " if result.attachments else ""
        )
        state.compose_stats.stream_queue.add(
            [
                {
                    "primary_text": batch.group_key,
                    "description": (
                        f"{len(result.candidates)} names  {attach_label}${cost:.3f}"
                    ),
                }
            ]
        )
        return candidates

    # --- Polling-based work distribution (42-polling-workers) ---
    # N independent workers poll an asyncio.Queue for batches. Each worker
    # reserves budget before processing; on failure the batch is re-enqueued
    # and the worker sleeps before retrying — no silent drops.  This replaces
    # the previous asyncio.gather fan-out + Semaphore pattern.
    batch_queue: asyncio.Queue = asyncio.Queue()
    for _q_batch in state.extracted:
        batch_queue.put_nowait(_q_batch)

    composed: list[dict] = []
    errors = 0
    _MAX_BUDGET_RETRIES = 5

    async def _compose_polling_worker(worker_id: int) -> None:
        nonlocal errors
        budget_retries = 0

        while not state.should_stop():
            try:
                batch = batch_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Budget gate — reserve before doing any LLM work.
            # Per-item cost calibrated to W36 graph data: compose mean
            # $0.13/item, p95 $0.47/item.  Reserve at $0.20/item (mean+50%
            # headroom) to cover the typical extension and avoid draining
            # the global pool via in-flight overshoot, which would starve
            # downstream review phases.
            lease = None
            if state.budget_manager:
                estimated = len(batch.items) * 0.20
                phase_tag = getattr(state, "budget_phase_tag", "") or "generate_name"
                lease = state.budget_manager.reserve(estimated, phase=phase_tag)
                if lease is None:
                    budget_retries += 1
                    if (
                        budget_retries > _MAX_BUDGET_RETRIES
                        or state.budget_manager.exhausted()
                    ):
                        wlog.info(
                            "Worker %d: budget exhausted — stopping",
                            worker_id,
                        )
                        break
                    # Re-enqueue batch and wait for other workers to release
                    batch_queue.put_nowait(batch)
                    wlog.debug(
                        "Worker %d: budget reserve failed for %s, "
                        "re-enqueuing (retry %d/%d)",
                        worker_id,
                        batch.group_key,
                        budget_retries,
                        _MAX_BUDGET_RETRIES,
                    )
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    continue

            try:
                candidates = await _compose_batch_body(batch, lease)
                if candidates:
                    composed.extend(candidates)
                budget_retries = 0  # Reset on success
            except Exception as exc:
                errors += 1
                wlog.warning(
                    "Worker %d: batch %s failed: %s",
                    worker_id,
                    batch.group_key,
                    exc,
                )
            finally:
                if lease:
                    lease.release_unused()

    n_compose_workers = get_compose_concurrency()
    await asyncio.gather(
        *[_compose_polling_worker(i) for i in range(n_compose_workers)]
    )

    state.composed = composed
    state.compose_stats.errors = errors

    attached = state.stats.get("attachments", 0)
    wlog.info(
        "Composition complete: %d composed, %d attached, %d errors (cost=$%.4f)",
        len(composed),
        attached,
        errors,
        state.compose_stats.cost,
    )

    # --- Batch-size telemetry (Workstream 2a) ---
    # Report the distribution of items per batch and the name-only mode
    # indicator so rotation summaries can compare name-only vs default
    # throughput without bespoke log scraping.
    if state.extracted:
        sizes = [len(b.items) for b in state.extracted]
        total_items_in_batches = sum(sizes)
        name_only_batches = sum(
            1 for b in state.extracted if getattr(b, "mode", "default") == "names"
        )
        singleton_count = sum(1 for s in sizes if s == 1)
        wlog.info(
            "Batch telemetry: %d batches (%d name_only), total_items=%d, "
            "mean=%.2f, min=%d, max=%d, singletons=%d (%.1f%%), cost_per_batch=$%.4f",
            len(sizes),
            name_only_batches,
            total_items_in_batches,
            total_items_in_batches / len(sizes) if sizes else 0.0,
            min(sizes) if sizes else 0,
            max(sizes) if sizes else 0,
            singleton_count,
            100.0 * singleton_count / len(sizes) if sizes else 0.0,
            state.compose_stats.cost / len(sizes) if sizes else 0.0,
        )
        state.stats["compose_batches"] = len(sizes)
        state.stats["compose_batches_name_only"] = name_only_batches
        state.stats["compose_batch_mean_size"] = (
            total_items_in_batches / len(sizes) if sizes else 0.0
        )
        state.stats["compose_batch_singleton_pct"] = (
            100.0 * singleton_count / len(sizes) if sizes else 0.0
        )

    state.stats["generate_name_count"] = len(composed)
    state.stats["compose_errors"] = errors
    state.stats["compose_cost"] = state.compose_stats.cost
    state.stats["compose_model"] = model
    state.stats["grammar_retries"] = state.grammar_retries
    state.stats["grammar_retries_succeeded"] = state.grammar_retries_succeeded
    state.stats["opus_revisions_attempted"] = state.opus_revisions_attempted
    state.stats["opus_revisions_accepted"] = state.opus_revisions_accepted

    state.compose_stats.freeze_rate()
    state.compose_phase.mark_done()


# =============================================================================
# VALIDATE phase
# =============================================================================


def _validate_via_isn(
    entry: dict,
) -> tuple[list[str], dict]:
    """Construct ISN Pydantic model and collect ALL validation issues.

    Returns:
        (issues: list[str], layer_summary: dict)

    Compose is always name-only (ADR-1): validation uses ISN's name-only
    model. This function is purely an annotator — it never rejects entries.
    Parseability is checked upstream by the grammar round-trip in
    validate_worker. This function attaches quality annotations.
    """
    from pydantic import ValidationError

    issues: list[str] = []
    summary = {
        "pydantic": {"passed": True, "error_count": 0},
        "semantic": {"issue_count": 0, "skipped": False},
        "description": {"issue_count": 0},
    }

    # Map codex dict keys to ISN model fields
    from imas_codex.standard_names.kind_derivation import to_isn_kind

    # Name-only ISN variants forbid dd_paths and physics_domain — those
    # fields only exist on the full (non-name-only) model. Compose is
    # always name-only (ADR-1), so we omit them here. DD-path provenance
    # is retained on the codex graph node via `source_paths`.
    isn_dict: dict[str, Any] = {
        "name": entry.get("id", ""),
        "kind": to_isn_kind(entry.get("kind", "scalar")),
    }
    # ISN metadata kind forbids unit field entirely
    if isn_dict["kind"] != "metadata":
        # User invariant: by the time we build the ISN dict, the unit
        # must come from DD (B1a above filters invalid units before this
        # path is reachable). Defensively assert so a logic regression
        # surfaces in tests rather than silently pollute the catalog
        # with `unit="1"`.
        unit = entry.get("unit")
        assert unit, (
            f"ISN dict missing unit for {entry.get('id')!r}; "
            "B1a (dd_unit_unresolvable skip) invariant violation"
        )
        isn_dict["unit"] = unit

    # Layer 1: Pydantic model construction (fires 18 validators)
    model = None
    try:
        from imas_standard_names.models import create_standard_name_entry

        model = create_standard_name_entry(isn_dict, name_only=True)
    except ValidationError as e:
        summary["pydantic"]["passed"] = False
        summary["pydantic"]["error_count"] = len(e.errors())
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            issues.append(f"[pydantic:{field}] {err['msg']}")
    except Exception as e:
        # Non-validation errors (import issues, etc.) — don't crash
        summary["pydantic"]["passed"] = False
        summary["pydantic"]["error_count"] = 1
        issues.append(f"[pydantic:unknown] {e}")

    # Layer 2: Semantic checks (only if model constructed)
    if model is not None:
        try:
            from imas_standard_names.validation.semantic import run_semantic_checks

            sem_issues = run_semantic_checks({isn_dict["name"]: model})
            summary["semantic"]["issue_count"] = len(sem_issues)
            issues.extend(f"[semantic] {i}" for i in sem_issues)
        except Exception as e:
            summary["semantic"]["skipped"] = True
            issues.append(f"[semantic] check failed: {e}")
    else:
        summary["semantic"]["skipped"] = True

    # Layer 2b: Structural checks (cross-entry consistency, normally run
    # at catalog-load time by ISN's YamlStore). Single-entry invocation
    # surfaces the per-name structural defects (e.g. mode-with-no-base,
    # forbidden-token use) at compose/validate time rather than waiting
    # for catalog assembly. Empty result is the expected pass case.
    summary["structural"] = {"issue_count": 0, "skipped": False}
    if model is not None:
        try:
            from imas_standard_names.validation.structural import (
                run_structural_checks,
            )

            struct_issues = run_structural_checks({isn_dict["name"]: model})
            summary["structural"]["issue_count"] = len(struct_issues)
            issues.extend(f"[structural] {i}" for i in struct_issues)
        except Exception as e:
            summary["structural"]["skipped"] = True
            issues.append(f"[structural] check failed: {e}")
    else:
        summary["structural"]["skipped"] = True

    # Layer 2c: Codex-side canonical-token + preposition checks. Runs even
    # when the ISN model fails to construct, because the violation is
    # determined from the candidate name itself rather than the model.
    canon_issues = _check_canonical_locus_and_preposition(isn_dict.get("name", ""))
    summary["canonical"] = {"issue_count": len(canon_issues)}
    issues.extend(f"[canonical] {i}" for i in canon_issues)

    # Layer 3: Description quality
    try:
        from imas_standard_names.validation.description import validate_description

        desc_issues = validate_description(isn_dict)
        summary["description"]["issue_count"] = len(desc_issues)
        issues.extend(f"[description] {i['message']}" for i in desc_issues)
    except Exception as e:
        issues.append(f"[description] check failed: {e}")

    return issues, summary


# Canonical-locus catalog enforced at validate time. Pairs are
# {forbidden_synonym: canonical_token}. Generation that uses a synonym is
# annotated as a canonical-token violation and routed through the refine
# pool to be rewritten to the canonical form.
_CANONICAL_LOCUS_SYNONYMS: dict[str, str] = {
    "separatrix": "plasma_boundary",
    "last_closed_flux_surface": "plasma_boundary",
    "lcfs": "plasma_boundary",
    "divertor_plate": "divertor_target",
    "wall_surface": "wall",
    "first_wall_surface": "wall",
    "vacuum_vessel_wall": "wall",
    "pedestal_region": "pedestal",
    "edge_pedestal": "pedestal",
    "core_axis": "magnetic_axis",
}

# Bases that name an evaluated field (defined everywhere in the plasma
# and READ at a locus). When paired with a position/region locus, the
# relation MUST be `_at_`. `_of_` is reserved for intrinsic geometric
# properties (area, radius, elongation, …).
_FIELD_BASES: frozenset[str] = frozenset(
    {
        "temperature",
        "density",
        "pressure",
        "magnetic_field",
        "electric_field",
        "magnetic_flux",
        "flux",
        "current",
        "current_density",
        "voltage",
        "loop_voltage",
        "velocity",
        "velocity_magnitude",
        "magnetic_shear",
        "safety_factor",
        "particle_flux",
        "energy_flux",
        "momentum_flux",
        "power",
        "power_density",
        "mass_density",
        "electric_potential",
        "electrostatic_potential",
        "kinetic_energy",
        "internal_energy",
        "enthalpy",
        "entropy",
        "radiation_density",
        "halo_current",
    }
)


def _check_canonical_locus_and_preposition(name: str) -> list[str]:
    """Surface canonical-locus and preposition issues for a candidate name.

    Returns a list of ``WARNING - ...`` strings (zero on a clean name).
    Decisions:

    1. Locus token is a synonym for a canonical token → flag a rewrite.
    2. Locus relation is ``of`` + position/region + field base → flag a
       rewrite to ``_at_`` (the field-at-region anti-pattern).

    The check is permissive: when the IR parser cannot interpret the
    name, the function returns no issues — upstream grammar gates will
    have already quarantined that case.
    """
    issues: list[str] = []
    try:
        from imas_standard_names.grammar.parser import parse as ir_parse

        result = ir_parse(name)
        ir = getattr(result, "ir", None)
        if ir is None or ir.locus is None or ir.base is None:
            return issues

        locus_token = ir.locus.token
        relation = (
            ir.locus.relation.value
            if hasattr(ir.locus.relation, "value")
            else str(ir.locus.relation)
        )
        locus_type = (
            ir.locus.type.value
            if hasattr(ir.locus.type, "value")
            else str(ir.locus.type)
        )
        base_token = ir.base.token

        canonical = _CANONICAL_LOCUS_SYNONYMS.get(locus_token)
        if canonical:
            issues.append(
                f"WARNING - locus token '{locus_token}' is a synonym for "
                f"canonical '{canonical}'; rewrite using '{canonical}'"
            )

        if (
            relation == "of"
            and locus_type in {"position", "region"}
            and base_token in _FIELD_BASES
        ):
            issues.append(
                f"WARNING - field base '{base_token}' with "
                f"_of_{locus_token} should use _at_{locus_token} "
                "(evaluated fields sample AT a position/region)"
            )
    except Exception:
        pass
    return issues


def _is_quarantined(issues: list[str], layer_summary: dict) -> bool:
    """Determine whether validation issues are critical (quarantine the name).

    Critical failures that make a name unusable for publication:
    - Grammar round-trip failure (``parse_error:`` prefix)
    - Pydantic validation failure (layer 1 did not pass)
    - Empty or missing description (no ``id`` or empty string)
    - Invalid kind value
    - L3 critical audit failures (latex_def_check, synonym_check, multi_subject_check)
    - L6 grammar retry exhausted

    Non-critical issues (semantic warnings, description quality hints,
    non-critical audits) do NOT trigger quarantine — they are advisory.
    """
    # Grammar round-trip failures are always critical
    if any(i.startswith("parse_error:") for i in issues):
        return True

    # Grammar ambiguity is also critical — the name can't be reliably parsed
    if any("grammar:ambiguity" in i for i in issues):
        return True

    # Pydantic validation failure (model construction failed)
    pydantic = layer_summary.get("pydantic", {})
    if not pydantic.get("passed", True):
        return True

    # L6: Grammar retry exhausted
    if any("audit:grammar_retry_exhausted" in i for i in issues):
        return True

    # L3: Critical audit failures
    from imas_codex.standard_names.audits import has_critical_audit_failure

    if has_critical_audit_failure(issues):
        return True

    return False


async def validate_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Validate composed names via ISN grammar checks (claim loop).

    Graph-primary: claims unvalidated StandardName nodes, runs ISN
    three-layer validation + grammar round-trip, marks results on graph.
    Follows the claim/mark/release pattern from discovery workers.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_validate_worker")

    # Initialize finalize progress (3 steps: validate, consolidate, persist)
    state.finalize_stats.total = 3
    state.finalize_stats.status_text = "validating…"

    if state.dry_run:
        wlog.info("Dry run — skipping validation")
        count = sum(len(b.items) for b in state.extracted) if state.extracted else 0
        state.validate_stats.total = count
        state.validate_stats.processed = count
        state.stats["validate_skipped"] = True
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    from imas_standard_names.grammar import (
        StandardName,
        compose_standard_name,
        parse_standard_name,
    )

    from imas_codex.standard_names.graph_ops import (
        claim_names_for_validation,
        mark_names_validated,
        release_validation_claims,
    )

    _BATCH_SIZE = 50

    total_valid = 0
    total_invalid = 0
    idle_count = 0
    _MAX_IDLE = 5
    _BATCH_SIZE = 50

    wlog.info("Starting validation claim loop")

    while not state.stop_requested:
        # Claim a batch from graph
        token, items = await asyncio.to_thread(claim_names_for_validation, _BATCH_SIZE)

        if not items:
            idle_count += 1
            if idle_count >= _MAX_IDLE:
                wlog.info("No more unvalidated names — exiting")
                break
            await asyncio.sleep(2.0)
            continue
        idle_count = 0

        wlog.debug("Claimed %d names for validation (token=%s)", len(items), token[:8])

        # Process the claimed batch
        try:
            results: list[dict[str, Any]] = []
            batch_invalid = 0

            for entry in items:
                name = entry.get("id", "")
                try:
                    # Grammar round-trip validates parsability
                    # (normalization already done at compose time)
                    parsed = parse_standard_name(name)
                    compose_standard_name(parsed)

                    # Fields consistency check
                    fields_dict = {}
                    for fk in _GRAMMAR_FIELDS:
                        val = entry.get(fk)
                        if val:
                            fields_dict[fk] = val
                    if fields_dict:
                        try:
                            sn_fields = _convert_fields_to_grammar(fields_dict)
                            if sn_fields:
                                sn = StandardName(**sn_fields)
                                compose_standard_name(sn)
                        except Exception:
                            pass

                    # ISN three-layer validation (annotate, never reject)
                    issues, layer_summary = _validate_via_isn(entry)

                    # --- L3: Post-gen audits ---
                    try:
                        from imas_codex.standard_names.audits import run_audits

                        source_path = None
                        source_paths = entry.get("source_paths") or []
                        if source_paths:
                            # Use first source path for provenance check
                            source_path = strip_dd_prefix(source_paths[0])

                        audit_issues = run_audits(
                            candidate=entry,
                            existing_sns_in_domain=None,  # Synonym check needs embeddings — skip in basic mode
                            source_path=source_path,
                            source_cocos_type=entry.get("cocos_transformation_type"),
                        )
                        if audit_issues:
                            issues.extend(audit_issues)
                        state.audits_run += 1
                        if audit_issues:
                            state.audits_failed += 1
                    except Exception:
                        wlog.debug("L3: Audit failed for %r", name, exc_info=True)

                    # L6: grammar retry exhausted flag from compose
                    if entry.get("_grammar_retry_exhausted"):
                        issues.append("audit:grammar_retry_exhausted")

                    results.append(
                        {
                            "id": name,
                            "validation_issues": issues,
                            "validation_layer_summary": json.dumps(layer_summary),
                            "validation_status": (
                                "quarantined"
                                if _is_quarantined(issues, layer_summary)
                                else "valid"
                            ),
                        }
                    )
                except Exception as exc:
                    exc_msg = str(exc).lower()
                    issues: list[str] = []

                    # Classify specific grammar ambiguities
                    if "component" in exc_msg and "coordinate" in exc_msg:
                        issues.append(
                            f"grammar:ambiguity:component_coordinate_overlap: {name}"
                        )
                    elif "ambig" in exc_msg:
                        issues.append(f"grammar:ambiguity:unclassified: {name}")
                    else:
                        issues.append(
                            f"parse_error: grammar round-trip failed for {name}"
                        )

                    wlog.debug(
                        "Validation error for %r: %s — tagging with %s",
                        name,
                        exc_msg[:80],
                        issues[0].split(":")[0],
                    )
                    results.append(
                        {
                            "id": name,
                            "validation_issues": issues,
                            "validation_layer_summary": json.dumps({}),
                            "validation_status": "quarantined",
                        }
                    )
                    batch_invalid += 1

            # Mark results on graph (token-verified)
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            total_valid += marked
            total_invalid += batch_invalid
            state.validate_stats.processed += marked

            wlog.info(
                "Validated batch: %d marked, %d errors",
                marked,
                batch_invalid,
            )
            state.validate_stats.record_batch(marked)

        except Exception:
            wlog.warning("Validation batch failed — releasing claims", exc_info=True)
            await asyncio.to_thread(release_validation_claims, token)

    state.stats["validate_valid"] = total_valid
    state.stats["validate_invalid"] = total_invalid
    state.stats["audits_run"] = state.audits_run
    state.stats["audits_failed"] = state.audits_failed
    state.validate_stats.errors = total_invalid
    state.validate_stats.freeze_rate()
    state.validate_phase.mark_done()
    state.finalize_stats.processed = 1
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "validate",
                "description": f"{total_valid} valid  {total_invalid} invalid",
            }
        ]
    )


def _convert_fields_to_grammar(fields: dict) -> dict:
    """Convert string field values to grammar enum instances."""
    from imas_standard_names.grammar import (
        BinaryOperator,
        Component,
        GeometricBase,
        Object,
        Position,
        Process,
        Subject,
        Transformation,
    )

    enum_map = {
        "subject": Subject,
        "component": Component,
        "coordinate": Component,
        "position": Position,
        "process": Process,
        "transformation": Transformation,
        "geometric_base": GeometricBase,
        "object": Object,
        "binary_operator": BinaryOperator,
    }

    sn_fields: dict = {}
    for k, v in fields.items():
        if k == "physical_base":
            sn_fields[k] = v
        elif k in enum_map:
            sn_fields[k] = enum_map[k](v)
    return sn_fields


# =============================================================================
# CONSOLIDATE phase
# =============================================================================


async def consolidate_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Cross-batch consolidation: dedup, conflict detection, coverage accounting.

    Graph-primary: queries all validated StandardNames from graph, runs
    consolidation analysis, marks approved names with ``consolidated_at``.
    Read-only query (no claims needed — single-pass batch analysis).
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_consolidate_worker")

    state.finalize_stats.status_text = "consolidating…"

    if state.dry_run:
        wlog.info("Dry run — skipping consolidation")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    from imas_codex.standard_names.consolidation import consolidate_candidates
    from imas_codex.standard_names.graph_ops import (
        get_validated_names,
        mark_names_consolidated,
    )

    # Always read from graph — this is the primary data source
    validated = await asyncio.to_thread(
        get_validated_names,
        ids_filter=getattr(state, "ids_filter", None),
    )

    if not validated:
        wlog.info("No validated names to consolidate — skipping")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    wlog.info("Consolidating %d validated candidates from graph", len(validated))
    state.consolidate_stats.total = len(validated)

    result = await asyncio.to_thread(consolidate_candidates, validated)

    # Mark approved names with consolidated_at on graph
    approved_ids = [e["id"] for e in result.approved if e.get("id")]
    if approved_ids:
        marked = await asyncio.to_thread(mark_names_consolidated, approved_ids)
        wlog.info("Marked %d names as consolidated", marked)

    # Log results
    wlog.info(
        "Consolidation: %d approved, %d conflicts, %d coverage gaps, %d reused",
        len(result.approved),
        len(result.conflicts),
        len(result.coverage_gaps),
        len(result.reused),
    )

    # Record stats
    state.stats["consolidation"] = result.stats
    if result.conflicts:
        for conflict in result.conflicts:
            wlog.warning(
                "Conflict: %s (%s) — %s",
                conflict.standard_name,
                conflict.conflict_type,
                conflict.details,
            )
    if result.coverage_gaps:
        wlog.info("Coverage gaps: %d unmapped source paths", len(result.coverage_gaps))

    state.consolidate_stats.processed = len(validated)
    state.consolidate_stats.freeze_rate()
    state.consolidate_phase.mark_done()
    state.finalize_stats.processed = 2
    conflicts_count = len(result.conflicts) if result.conflicts else 0
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "consolidate",
                "description": (
                    f"{len(result.approved)} names  {conflicts_count} conflicts"
                ),
            }
        ]
    )


# =============================================================================
# PERSIST phase
# =============================================================================


async def persist_worker(state: StandardNameBuildState, **_kwargs) -> None:
    """Compute embeddings for consolidated StandardNames (claim loop).

    Graph-primary: claims unembedded StandardNames, computes embeddings,
    writes results back to graph. Names are already persisted by compose —
    this worker handles the embedding enrichment pass.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_persist_worker")

    state.finalize_stats.status_text = "embedding…"

    if state.dry_run:
        wlog.info("Dry run — skipping embedding")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    from imas_codex.standard_names.graph_ops import (
        claim_names_for_embedding,
        mark_names_embedded,
        release_embedding_claims,
    )

    total_embedded = 0
    idle_count = 0
    _MAX_IDLE = 5
    _BATCH_SIZE = 100

    wlog.info("Starting embedding claim loop")

    while not state.stop_requested:
        # Claim a batch from graph
        token, items = await asyncio.to_thread(claim_names_for_embedding, _BATCH_SIZE)

        if not items:
            idle_count += 1
            if idle_count >= _MAX_IDLE:
                wlog.info("No more names needing embedding — exiting")
                break
            await asyncio.sleep(2.0)
            continue
        idle_count = 0

        wlog.debug("Claimed %d names for embedding (token=%s)", len(items), token[:8])

        try:
            from imas_codex.embeddings.description import embed_descriptions_batch

            enriched = await asyncio.to_thread(embed_descriptions_batch, items)
            embed_batch = [
                {"id": e["id"], "embedding": e["embedding"]}
                for e in enriched
                if e.get("embedding")
            ]

            marked = await asyncio.to_thread(mark_names_embedded, token, embed_batch)
            total_embedded += marked

            wlog.info("Embedded batch: %d names", marked)
            state.persist_stats.processed += marked
            state.persist_stats.record_batch(marked)

        except Exception:
            wlog.warning("Embedding batch failed — releasing claims", exc_info=True)
            await asyncio.to_thread(release_embedding_claims, token)

    # Post-success cleanup: detach stale HAS_STANDARD_NAME for targeted paths
    # Only runs when --force/--paths regenerated names for specific paths
    if state.force and total_embedded > 0 and state.extracted:
        new_name_ids: set[str] = set()
        source_paths: set[str] = set()

        # Collect from graph — names we just embedded
        from imas_codex.graph.client import GraphClient

        def _get_recent_names():
            with GraphClient() as gc:
                results = gc.query(
                    """
                    MATCH (sn:StandardName)
                    WHERE sn.embedded_at IS NOT NULL
                      AND sn.pipeline_status IN ['named', 'drafted']
                    RETURN sn.id AS id, sn.source_paths AS source_paths
                    """
                )
                for r in results:
                    new_name_ids.add(r["id"])
                    for p in r["source_paths"] or []:
                        source_paths.add(p)

        await asyncio.to_thread(_get_recent_names)

        if source_paths and new_name_ids:

            def _cleanup_stale():
                # Strip dd: prefix for IMASNode.id lookup
                bare_paths = [strip_dd_prefix(p) for p in source_paths]
                with GraphClient() as gc:
                    result = list(
                        gc.query(
                            """
                            UNWIND $paths AS path
                            MATCH (n:IMASNode {id: path})-[r:HAS_STANDARD_NAME]->(sn:StandardName)
                            WHERE NOT (sn.id IN $keep_names)
                              AND sn.pipeline_status IN ['named', 'drafted', null]
                            DELETE r
                            RETURN count(r) AS detached
                            """,
                            paths=bare_paths,
                            keep_names=list(new_name_ids),
                        )
                    )
                    return result[0]["detached"] if result else 0

            detached = await asyncio.to_thread(_cleanup_stale)
            if detached:
                wlog.info("Cleaned %d stale HAS_STANDARD_NAME relationships", detached)
                state.stats["stale_detached"] = detached

    state.stats["persist_embedded"] = total_embedded
    wlog.info("Persist complete: %d embedded", total_embedded)
    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()
    state.finalize_stats.processed = 3
    state.finalize_stats.status_text = "done"
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "persist",
                "description": f"{total_embedded} embedded",
            }
        ]
    )
    state.finalize_stats.freeze_rate()


# =============================================================================
# Pool-mode batch processors (Phase 8)
# =============================================================================


async def compose_batch(
    batch: list[dict],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    regen: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Shared implementation for compose and regen pool batch processors.

    Takes a list of pre-claimed items (dicts with ``path``, ``description``,
    ``physics_domain``, etc.) and runs the compose pipeline:
    prompt → LLM → grammar validate → persist.

    **H5 — batch-scope domain context:**
    Domain vocabulary is derived from the *batch items* rather than a
    run-scoped ``state.domain_filter``, so pooled-mode batches get domain
    context even without ``--physics-domain``.

    **H6 — soft drain on stop_event:**
    Checks ``stop_event.is_set()`` before the LLM call and returns early
    (after persisting any in-flight results) if the event fires.

    Args:
        batch: Pre-claimed source items from the graph claim query.
        mgr: Shared :class:`BudgetManager` for cost tracking.
        stop_event: Cooperative shutdown signal.
        regen: If True, set ``regen_increment`` on persisted candidates.

    Returns:
        Count of successfully composed/regenerated candidates.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameComposeBatch

    if not batch:
        return 0

    model = get_model("sn-compose")
    context = build_compose_context()

    # ── H5: batch-scope domain context ─────────────────────────────────
    def _scalar_domain(d: object) -> str | None:
        """Normalise physics_domain to a scalar string (handles list or str)."""
        if isinstance(d, list):
            return d[0] if d else None
        return d  # type: ignore[return-value]

    domains_in_batch = sorted(
        {
            _scalar_domain(item.get("physics_domain"))
            for item in batch
            if item.get("physics_domain")
        }
        - {None}
    )

    from imas_codex.standard_names.context import build_domain_vocabulary_preseed

    domain_vocab_parts: list[str] = []
    for dom in domains_in_batch:
        vocab = await asyncio.to_thread(build_domain_vocabulary_preseed, dom)
        if vocab:
            domain_vocab_parts.append(vocab)
    context["domain_vocabulary"] = "\n".join(domain_vocab_parts)

    # ── L4: Reviewer-theme extraction (batch-scoped) ───────────────────
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    for dom in domains_in_batch:
        themes = await asyncio.to_thread(extract_reviewer_themes, dom)
        reviewer_themes.extend(themes)
    context["reviewer_themes"] = reviewer_themes[:12]

    # ── K3: Scored-example injection ───────────────────────────────────
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_compose_examples

    def _load_scored_examples() -> list[dict]:
        with GraphClient() as gc:
            return load_compose_examples(
                gc, physics_domains=domains_in_batch, axis="name"
            )

    compose_scored_examples = await asyncio.to_thread(_load_scored_examples)
    context["compose_scored_examples"] = compose_scored_examples

    # ── System prompt (cached per pool lifetime) ───────────────────────
    system_prompt = render_prompt("sn/generate_name_system", context)

    # ── Enrich items with DD context ───────────────────────────────────
    def _enrich():
        _enrich_batch_items(batch)

    await asyncio.to_thread(_enrich)

    # ── Search nearby existing names (per-item) ──────────────────────────
    # For each batch item, search using the item's description (or path
    # leaf) to get semantically relevant nearby names.  Deduplicate across
    # items and cap at 30 total to keep prompt size bounded.
    group_key = batch[0].get("path", "").split("/")[0] if batch else ""
    _nearby_seen: set[str] = set()
    nearby: list[dict] = []
    _PER_ITEM_K = 5
    _NEARBY_CAP = 30
    for item in batch:
        # Prefer item description for embedding query; fall back to path
        hint = item.get("description") or item.get("path", "")
        item_results = _search_nearby_names(hint, k=_PER_ITEM_K)
        for nr in item_results:
            nid = nr.get("id", "")
            if nid and nid not in _nearby_seen:
                _nearby_seen.add(nid)
                nearby.append(nr)
                if len(nearby) >= _NEARBY_CAP:
                    break
        if len(nearby) >= _NEARBY_CAP:
            break

    # ── IDS context ────────────────────────────────────────────────────
    ids_names = sorted(
        {
            item["path"].split("/")[0]
            for item in batch
            if item.get("path") and "/" in item["path"]
        }
    )
    ids_contexts: list[dict] = []
    for iname in ids_names:
        info = _enrich_ids_context(iname)
        if info:
            ids_contexts.append({"ids_name": iname, **info})

    # ── H6: pre-LLM stop check ────────────────────────────────────────
    if stop_event.is_set():
        return 0

    # ── B9: Reference exemplars — superseded by graph-backed
    # ``compose_scored_examples`` loaded above. The active prompt
    # template ``sn/generate_name_dd`` no longer renders a
    # ``reference_exemplars`` block (removed in commit 7a86069e), so
    # the per-batch ANN search has been retired here. The linear path
    # still uses ``_search_reference_exemplars`` for the names-only
    # template ``sn/generate_name_dd_names``.

    # ── B10: Cluster-aware existing-names roster ──────────────────────
    # Build name-only roster from any existing_name field carried on batch
    # items plus a graph query for SNs in the same IMASSemanticCluster.
    existing_names_set: set[str] = {
        item.get("existing_name") for item in batch if item.get("existing_name")
    }
    existing_names_set.discard(None)
    try:
        from imas_codex.graph.client import GraphClient

        path_ids = [item.get("path") for item in batch if item.get("path")]
        if path_ids:

            def _cluster_roster() -> list[str]:
                with GraphClient() as gc:
                    rows = gc.query(
                        """
                        UNWIND $paths AS p
                        MATCH (n:IMASNode {id: p})-[:IN_SEMANTIC_CLUSTER]->(cl)
                        MATCH (other:IMASNode)-[:IN_SEMANTIC_CLUSTER]->(cl)
                        MATCH (other)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                        WHERE coalesce(sn.validation_status, '') <> 'quarantined'
                          AND coalesce(sn.pipeline_status, '') <> 'superseded'
                        RETURN DISTINCT sn.id AS id
                        LIMIT 200
                        """,
                        paths=path_ids,
                    )
                    return [r["id"] for r in (rows or []) if r.get("id")]

            cluster_names = await asyncio.to_thread(_cluster_roster)
            existing_names_set.update(cluster_names)
    except Exception:
        logger.debug("Cluster-aware existing_names lookup failed", exc_info=True)
    existing_names = sorted(existing_names_set)[:200]

    # ── Render user prompt ─────────────────────────────────────────────
    user_context = {
        "items": batch,
        "ids_name": group_key,
        "ids_contexts": ids_contexts,
        "existing_names": existing_names,
        "cluster_context": "",
        "nearby_existing_names": nearby,
        "cocos_version": batch[0].get("cocos_version") if batch else None,
        "dd_version": batch[0].get("dd_version") if batch else None,
    }
    user_prompt = render_prompt("sn/generate_name_dd", {**context, **user_context})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # ── Budget reservation ─────────────────────────────────────────────
    # B12: reserve for retry_attempts + 1 LLM calls (re-enrichment retry)
    _max_retries = _retry_attempts()
    estimated = len(batch) * 0.20 * (_max_retries + 1)
    phase_tag = "regen" if regen else "generate_name"
    lease = mgr.reserve(estimated, phase=phase_tag)
    if lease is None:
        # Fallback: tracking-only lease so charge_event still records
        # spend.  Without this, _spent stays at 0 and hard_exhausted()
        # never fires — causing the pool to run indefinitely.
        lease = mgr.reserve(0.0, phase=phase_tag)

    try:
        # ── B12: bounded retry loop for failed compositions ────────────
        # Mirror of the linear path's Delta H retry: on grammar parse
        # failure across any candidate, re-enrich items with expanded DD
        # context and re-prompt.  Bounded by _retry_attempts().
        _total_compose_cost = 0.0
        _total_tokens_in = 0
        _total_tokens_out = 0
        _total_cache_read = 0
        _total_cache_creation = 0

        for _compose_attempt in range(_max_retries + 1):
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-compose"),
                )
            except (ValueError, Exception) as compose_exc:
                _exc_str = str(compose_exc)
                if "not a registered" in _exc_str:
                    logger.warning(
                        "Pool %s: vocab gap validation error — marking "
                        "%d sources as vocab_gap: %s",
                        phase_tag,
                        len(batch),
                        _exc_str[:200],
                    )
                    _mark_vocab_gap_sources(batch, _exc_str, "dd")
                    return 0
                raise
            result, cost, tokens = llm_out

            # Accumulate token/cost totals across retries
            _total_compose_cost += cost
            _attempt_tokens_in = getattr(llm_out, "input_tokens", 0) or 0
            _attempt_tokens_out = getattr(llm_out, "output_tokens", 0) or 0
            _attempt_cache_r = getattr(llm_out, "cache_read_tokens", 0) or 0
            _attempt_cache_w = getattr(llm_out, "cache_creation_tokens", 0) or 0
            _total_tokens_in += _attempt_tokens_in
            _total_tokens_out += _attempt_tokens_out
            _total_cache_read += _attempt_cache_r
            _total_cache_creation += _attempt_cache_w

            # Charge this attempt's cost immediately
            if lease:

                def _safe_sn_ids(_r=result) -> tuple[str, ...]:
                    ids: list[str] = []
                    for c in _r.candidates if _r else []:
                        try:
                            ids.append(c.compose_name())
                        except Exception:
                            ids.append(f"<compose-error:{c.segments.base_token}>")
                    return tuple(ids)

                _event = LLMCostEvent(
                    model=model,
                    tokens_in=_attempt_tokens_in,
                    tokens_out=_attempt_tokens_out,
                    tokens_cached_read=_attempt_cache_r,
                    tokens_cached_write=_attempt_cache_w,
                    sn_ids=_safe_sn_ids(),
                    batch_id=group_key,
                    phase=phase_tag,
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # Quick grammar round-trip check on all candidates
            _grammar_failures: list[str] = []
            try:
                from imas_standard_names.grammar import parse_standard_name

                for c in result.candidates:
                    try:
                        _name = c.compose_name()
                        parse_standard_name(_name)
                    except Exception:
                        try:
                            _grammar_failures.append(c.compose_name())
                        except Exception:
                            _grammar_failures.append(c.source_id)
            except ImportError:
                pass  # ISN not installed — skip check

            if not _grammar_failures or _compose_attempt >= _max_retries:
                break

            # Re-enrich items with expanded hybrid search for retry
            logger.info(
                "Pool %s: composition retry %d/%d: %d grammar failures (%s) "
                "— re-composing with expanded DD context",
                phase_tag,
                _compose_attempt + 1,
                _max_retries,
                len(_grammar_failures),
                ", ".join(_grammar_failures[:3]),
            )

            def _re_enrich_expanded():
                from imas_codex.graph.client import GraphClient

                with GraphClient() as gc:
                    batch_tuples = [
                        (
                            item.get("path"),
                            item.get("description"),
                            item.get("physics_domain"),
                        )
                        for item in batch
                        if item.get("path")
                    ]
                    if batch_tuples:
                        hybrid_results = _hybrid_search_neighbours_batch(
                            gc,
                            batch_tuples,
                            search_k=_retry_k_expansion(),
                        )
                        item_idx = 0
                        for item in batch:
                            if not item.get("path"):
                                continue
                            if hybrid_results[item_idx]:
                                item["hybrid_neighbours"] = hybrid_results[item_idx]
                            item_idx += 1

            await asyncio.to_thread(_re_enrich_expanded)

            _retry_reason = (
                f"Previous attempt failed: grammar round-trip failed for "
                f"{', '.join(_grammar_failures[:3])}. Consider expanded "
                f"neighbour context and produce a different name."
            )
            retry_render_ctx = {
                **context,
                **user_context,
                "retry_reason": _retry_reason,
            }
            user_prompt = render_prompt("sn/generate_name_dd", retry_render_ctx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        # Use accumulated totals for per-candidate cost attribution
        cost = _total_compose_cost
        tokens_in = _total_tokens_in
        tokens_out = _total_tokens_out
        tokens_cache_r = _total_cache_read
        tokens_cache_w = _total_cache_creation

        # ── Build candidates ───────────────────────────────────────────
        candidates: list[dict] = []
        wlog = logger  # plain logger; pool path has no WorkerLogAdapter
        source_kind = "dd"  # pool-mode generate-name is DD-only today
        for c in result.candidates:
            source_item = next(
                (item for item in batch if item.get("path") == c.source_id),
                None,
            )
            # User invariant (AGENTS.md "Unit safety"): the LLM never
            # decides units; the DD source must provide one.
            # - "1": dimensionless (ISN convention) — valid
            # - "-": DD dimensionless marker — normalize to "1"
            # - "mixed": heterogeneous dimensions — skip
            # - None/empty: enrichment couldn't resolve — skip
            raw_unit = source_item.get("unit") if source_item else None
            if raw_unit == "-":
                raw_unit = "1"  # normalize DD marker to ISN convention
            if raw_unit in ("mixed", None, ""):
                _skip_reason = (
                    "dd_unit_mixed_non_standard"
                    if raw_unit == "mixed"
                    else "dd_unit_unresolvable"
                )
                wlog.warning(
                    "compose(pool): %s, skipping source=%s raw_unit=%r",
                    _skip_reason,
                    c.source_id,
                    raw_unit,
                )
                if c.source_id:
                    try:
                        from imas_codex.graph.client import GraphClient
                        from imas_codex.standard_names.graph_ops import (
                            mark_source_skipped,
                        )

                        with GraphClient() as _gc:
                            mark_source_skipped(
                                _gc,
                                c.source_id,
                                reason=_skip_reason,
                                detail=str(raw_unit),
                                source_type=source_kind,
                            )
                    except Exception as exc:
                        wlog.debug(
                            "Failed to mark pool source %s skipped: %s",
                            c.source_id,
                            exc,
                        )
                continue
            unit = raw_unit

            raw_domain = source_item.get("physics_domain") if source_item else None
            physics_domain = raw_domain or None
            source_domains = [raw_domain] if raw_domain else []

            cocos_type = source_item.get("cocos_label") if source_item else None

            # B1/W4b: Pre-validation gate — reject malformed LLM output
            # before MERGE.  Mark the source as 'failed' so it is not
            # re-claimed forever.
            name_id = normalize_spelling(c.compose_name())
            _well_formed, _reject_reason = is_well_formed_candidate(name_id)
            if not _well_formed:
                wlog.warning(
                    "Pool %s: pre-validation reject %r (%s)",
                    phase_tag,
                    name_id[:80],
                    _reject_reason,
                )
                if c.source_id:
                    try:
                        from imas_codex.graph.client import GraphClient

                        with GraphClient() as gc:
                            gc.query(
                                "MATCH (sns:StandardNameSource {id: $id}) "
                                "SET sns.status = 'failed', "
                                "    sns.claimed_at = null, "
                                "    sns.claim_token = null",
                                id=f"{source_kind}:{c.source_id}",
                            )
                    except Exception:
                        wlog.debug("Failed to mark source %s as failed", c.source_id)
                continue

            grammar_failed = False
            try:
                from imas_standard_names.grammar import (
                    compose_standard_name,
                    parse_standard_name,
                )

                parsed = parse_standard_name(name_id)
                normalized = compose_standard_name(parsed)
                if normalized != name_id:
                    name_id = normalized
                name_id = _dedup_adjacent_tokens(name_id)
            except Exception as gram_exc:
                grammar_failed = True
                wlog.debug(
                    "Pool %s: grammar parse failed for %r — attempting B2 retry",
                    phase_tag,
                    name_id,
                )

                # B2: Single-shot grammar-failure retry.
                try:
                    retry_name, _r_cost, _r_ti, _r_to = await _grammar_retry(
                        name_id, str(gram_exc), model, acall_llm_structured
                    )
                    if lease and _r_cost > 0:
                        _r_event = LLMCostEvent(
                            model=model,
                            tokens_in=_r_ti,
                            tokens_out=_r_to,
                            sn_ids=(name_id,),
                            batch_id=f"{group_key}-grammar-retry",
                            phase=phase_tag,
                            service="standard-names",
                        )
                        lease.charge_event(_r_cost, _r_event)
                    if retry_name and retry_name != name_id:
                        try:
                            parsed = parse_standard_name(retry_name)
                            normalized = compose_standard_name(parsed)
                            name_id = _dedup_adjacent_tokens(normalized)
                            grammar_failed = False
                            wlog.info(
                                "Pool %s: B2 grammar retry succeeded %r → %r",
                                phase_tag,
                                c.compose_name(),
                                name_id,
                            )
                        except Exception:
                            wlog.debug(
                                "Pool %s: B2 retry result still un-parseable",
                                phase_tag,
                            )
                except Exception:
                    wlog.debug(
                        "Pool %s: B2 grammar retry failed for %r",
                        phase_tag,
                        name_id,
                    )

            cand = {
                "id": name_id,
                "source_types": [source_kind],
                "source_id": c.source_id,
                "description": normalize_prose_spelling(c.description or ""),
                "kind": c.kind,
                "source_paths": [
                    encode_source_path(source_kind, p) for p in (c.dd_paths or [])
                ],
                "reason": c.reason,
                "unit": unit,
                "physics_domain": physics_domain,
                "source_domains": source_domains,
                "cocos_transformation_type": cocos_type,
                "cocos": batch[0].get("cocos_version") if batch else None,
                "dd_version": batch[0].get("dd_version") if batch else None,
                **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
            }

            if regen:
                cand["regen_increment"] = True

            candidates.append(cand)

            # ── B3: Deterministic error-sibling minting ───────────────
            if (
                not grammar_failed
                and source_item
                and source_item.get("has_errors")
                and source_item.get("error_node_ids")
            ):
                try:
                    from imas_codex.standard_names.error_siblings import (
                        mint_error_siblings,
                    )

                    siblings = mint_error_siblings(
                        name_id,
                        error_node_ids=source_item["error_node_ids"],
                        unit=unit,
                        physics_domain=physics_domain,
                        cocos_type=cocos_type,
                        cocos_version=batch[0].get("cocos_version") if batch else None,
                        dd_version=batch[0].get("dd_version") if batch else None,
                    )
                    if siblings:
                        for s in siblings:
                            s["_from_error_sibling"] = True
                        candidates.extend(siblings)
                        wlog.debug(
                            "Pool %s: minted %d error siblings for %r",
                            phase_tag,
                            len(siblings),
                            name_id,
                        )
                except Exception:
                    wlog.debug(
                        "Pool %s: error-sibling minting failed for %r",
                        phase_tag,
                        name_id,
                        exc_info=True,
                    )

        # ── B8: Per-candidate cost / token attribution ────────────────
        # Pro-rata across LLM-generated candidates only — error siblings
        # are deterministic and carry no LLM cost.
        from datetime import UTC, datetime

        _llm_cands = [c for c in candidates if not c.get("_from_error_sibling")]
        n_cands = max(len(_llm_cands), 1)
        _llm_at = datetime.now(UTC).isoformat()
        for cand in candidates:
            if cand.get("_from_error_sibling"):
                continue
            cand["llm_cost"] = cost / n_cands
            cand["llm_model"] = model
            cand["llm_service"] = "standard-names"
            cand["llm_at"] = _llm_at
            cand["llm_tokens_in"] = tokens_in // n_cands
            cand["llm_tokens_out"] = tokens_out // n_cands
            cand["llm_tokens_cached_read"] = tokens_cache_r // n_cands
            cand["llm_tokens_cached_write"] = tokens_cache_w // n_cands

        # ── C1: Inline audits — populate validation_status ────────────
        try:
            from imas_codex.standard_names.audits import run_audits

            for cand in candidates:
                if cand.get("_from_error_sibling"):
                    cand.setdefault("validation_status", "valid")
                    continue
                src_paths = cand.get("source_paths") or []
                src_path = strip_dd_prefix(src_paths[0]) if src_paths else None
                try:
                    audit_issues = run_audits(
                        candidate=cand,
                        existing_sns_in_domain=None,
                        source_path=src_path,
                        source_cocos_type=cand.get("cocos_transformation_type"),
                    )
                except Exception:
                    audit_issues = []
                if cand.get("_grammar_retry_exhausted"):
                    audit_issues = list(audit_issues) + [
                        "audit:grammar_retry_exhausted"
                    ]
                cand["validation_issues"] = audit_issues
                cand["validation_status"] = (
                    "quarantined"
                    if _is_quarantined(list(audit_issues), {})
                    else "valid"
                )
        except Exception:
            wlog.debug("Pool %s: audits failed", phase_tag, exc_info=True)
            for cand in candidates:
                cand.setdefault("validation_status", "valid")

        # Strip private flag before persist
        for cand in candidates:
            cand.pop("_grammar_retry_exhausted", None)

        # ── B4: Vocab gaps — persist + clear sources ──────────────────
        if result.vocab_gaps:
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            gap_dicts = [
                {
                    "source_id": vg.source_id,
                    "segment": vg.segment,
                    "token": vg.token,
                    "reason": vg.reason,
                }
                for vg in result.vocab_gaps
            ]
            try:
                await asyncio.to_thread(write_vocab_gaps, gap_dicts, source_kind)
                wlog.debug(
                    "Pool %s: persisted %d vocab gaps", phase_tag, len(gap_dicts)
                )
            except Exception:
                wlog.warning(
                    "Pool %s: write_vocab_gaps failed", phase_tag, exc_info=True
                )
            try:
                await asyncio.to_thread(
                    _update_sources_after_vocab_gap, gap_dicts, source_kind, wlog
                )
            except Exception:
                wlog.debug(
                    "Pool %s: vocab_gap source-status update failed",
                    phase_tag,
                    exc_info=True,
                )

        # W29: auto-detect novel physical_base tokens.
        if result.candidates:
            try:
                auto_gaps = _auto_detect_physical_base_gaps(result.candidates)
                if auto_gaps:
                    from imas_codex.standard_names.graph_ops import write_vocab_gaps

                    await asyncio.to_thread(
                        write_vocab_gaps,
                        auto_gaps,
                        source_kind,
                        skip_segment_filter=True,
                    )
                    wlog.debug(
                        "Pool %s: auto-detected %d physical_base gaps",
                        phase_tag,
                        len(auto_gaps),
                    )
            except Exception:
                wlog.debug(
                    "Pool %s: physical_base gap detection failed",
                    phase_tag,
                    exc_info=True,
                )

        # ── B5: Attachments — write edges + clear source claims ───────
        if result.attachments:
            try:
                attach_counts = await asyncio.to_thread(
                    _process_attachments_core, result.attachments, wlog
                )
                wlog.info(
                    "Pool %s: attached %d (rejected %d)",
                    phase_tag,
                    attach_counts.get("accepted", 0),
                    attach_counts.get("rejected", 0),
                )
            except Exception:
                wlog.debug(
                    "Pool %s: attachment processing failed",
                    phase_tag,
                    exc_info=True,
                )
            try:
                await asyncio.to_thread(
                    _update_sources_after_attach,
                    result.attachments,
                    source_kind,
                    wlog,
                )
            except Exception:
                wlog.debug(
                    "Pool %s: attach source-status update failed",
                    phase_tag,
                    exc_info=True,
                )

        # ── B6: Skipped sources — clear claims so they don't loop ─────
        if result.skipped:
            try:
                await asyncio.to_thread(
                    _update_sources_after_skip,
                    list(result.skipped),
                    source_kind,
                    wlog,
                )
                wlog.debug(
                    "Pool %s: marked %d sources as skipped",
                    phase_tag,
                    len(result.skipped),
                )
            except Exception:
                wlog.debug(
                    "Pool %s: skipped source-status update failed",
                    phase_tag,
                    exc_info=True,
                )

        # ── Persist ────────────────────────────────────────────────────
        if candidates:
            from imas_codex.standard_names.graph_ops import persist_generated_name_batch

            written = await asyncio.to_thread(
                persist_generated_name_batch,
                candidates,
                compose_model=model,
                dd_version=batch[0].get("dd_version"),
                cocos_version=batch[0].get("cocos_version"),
                run_id=mgr.run_id,
            )
            logger.debug("Pool %s: persisted %d candidates", phase_tag, written)

            # ── Emit per-item events ──────────────────────────────────
            if on_event is not None:
                for cand in candidates:
                    on_event(
                        {
                            "pool": "generate_name",
                            "name": cand["id"],
                            "source": (cand.get("source_paths") or [""])[0],
                            "dd_path": (cand.get("source_paths") or [""])[0],
                            "model": model,
                            "cost": cand.get("llm_cost", 0.0),
                        }
                    )

        return len(candidates)

    except Exception:
        logger.exception("Pool %s: compose batch failed", phase_tag)
        raise
    finally:
        if lease:
            lease.release_unused()


async def process_generate_name_batch(
    batch: list[dict],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Pool-mode generate-name batch processor.

    Takes pre-claimed source items (dicts from the graph claim query),
    generates standard names via LLM, and persists results.

    Returns count of items successfully processed.
    """
    return await compose_batch(batch, mgr, stop_event, regen=False, on_event=on_event)


async def process_refine_name_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of StandardNames for name refinement (Option B).

    For each item in the batch:
    1. Walk the REFINED_FROM chain via ``chain_history`` (already enriched).
    2. Decide whether to escalate (chain_length ≥ rotation_cap - 1).
    3. Optionally fan out targeted DD context (plan 39 Phase 1 — gated).
    4. Call LLM to produce a refined name (``RefinedName`` response model).
    5. Run the Phase 1.5 dup guard before persisting (plan 39 §5.2).
    6. Persist via ``persist_refined_name`` (new node + edge migration).
    7. On failure, release claims via ``release_refine_name_failed_claims``.

    Returns count of items successfully processed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.canonical import find_name_key_duplicate
    from imas_codex.standard_names.defaults import (
        DEFAULT_ESCALATION_MODEL,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.fanout import (
        CandidateContext,
        FanoutScope,
        assign_arm,
        load_settings as load_fanout_settings,
        run_fanout,
        should_trigger_fanout,
    )
    from imas_codex.standard_names.graph_ops import (
        _mark_refine_vocab_gap_exhausted,
        persist_refined_name,
        release_refine_name_failed_claims,
    )
    from imas_codex.standard_names.models import RefinedName

    rotation_cap = DEFAULT_REFINE_ROTATIONS
    processed = 0

    # ── GraphClient lifecycle (plan 39 §10.1 I5) ─────────────────────
    # One client per cycle, reused by hybrid-neighbour search,
    # run_fanout, dup guard, and Fanout-node telemetry writes.
    # ``persist_refined_name`` opens its own client (different
    # transaction lifecycle) — that is intentional and unchanged.
    fanout_settings = load_fanout_settings()

    with GraphClient() as gc:
        for item in batch:
            if stop_event.is_set():
                break

            sn_id = item["id"]
            chain_length = item.get("chain_length", 0) or 0
            chain_history = item.get("chain_history", [])

            # ── Escalation decision ───────────────────────────────────
            escalate = chain_length >= rotation_cap - 1
            if escalate:
                model = DEFAULT_ESCALATION_MODEL
            else:
                # Refine tier (Sonnet 4.6 by default) — peeled off
                # [language] on 2026-05-03 after E3 acceptance audit
                # showed flash-lite refines lifted critiqued names at
                # ~5% vs ~42% for Sonnet compose.
                model = get_model("sn-refine")

            # ── Build prompt context ──────────────────────────────────
            path = item.get("source_paths", [""])[0] if item.get("source_paths") else ""

            # ── Load scored compose examples for refinement context ────
            from imas_codex.standard_names.example_loader import load_compose_examples

            try:
                _item_domain = item.get("physics_domain") or ""

                def _load_compose_examples(domain: str = _item_domain) -> list:
                    with GraphClient() as _gc:
                        return load_compose_examples(
                            _gc, physics_domains=[domain], axis="name"
                        )

                compose_scored_examples = await _asyncio.to_thread(
                    _load_compose_examples
                )
            except Exception:
                logger.debug(
                    "refine_name: scored-example load failed for %s",
                    sn_id,
                    exc_info=True,
                )
                compose_scored_examples = []

            # ── Parse vocab_gap_detail from JSON string (best-effort) ─────
            raw_vgd = item.get("vocab_gap_detail")
            vocab_gap_detail: dict | None = None
            if raw_vgd:
                try:
                    import json as _json

                    vocab_gap_detail = (
                        _json.loads(raw_vgd) if isinstance(raw_vgd, str) else raw_vgd
                    )
                except (ValueError, TypeError):
                    logger.debug(
                        "refine_name: could not parse vocab_gap_detail for %s", sn_id
                    )

            # validation_issues is stored as a list on the node directly
            validation_issues: list[str] | None = item.get("validation_issues") or None

            prompt_context: dict[str, Any] = {
                "item": item,
                "chain_history": chain_history,
                "chain_length": chain_length,
                "hybrid_neighbours": [],
                "fanout_evidence": "",
                "compose_scored_examples": compose_scored_examples,
                "vocab_gap_detail": vocab_gap_detail,
                "validation_issues": validation_issues,
            }

            # Attempt hybrid neighbour search (best-effort).  Uses the
            # cycle-scoped ``gc`` (plan 39 §10.1) — no fresh client.
            try:
                neighbours = _hybrid_search_neighbours(gc, path)
                prompt_context["hybrid_neighbours"] = [
                    {"path": n.get("path", ""), "description": n.get("description")}
                    for n in neighbours
                ]
            except Exception:
                logger.debug("Hybrid neighbour search failed for %s", sn_id)

            # ── DD path context (keywords, clusters, version history) ─
            if path:
                try:
                    _enrich_dd_path_context(gc, item, path)
                except Exception:
                    logger.debug(
                        "refine_name: dd_path_context failed for %s",
                        sn_id,
                        exc_info=True,
                    )

            # ── Fan-out trigger gate (plan 39 §5.1) ──────────────────
            # Plumb reviewer_comments_per_dim_name from the claim batch
            # through the trigger predicate.  Gate on ALL of:
            # chain_length > 0, chain_history present (B12 enrichment),
            # at least one allow-listed dim contains a trigger keyword.
            reviewer_comments = item.get("reviewer_comments_per_dim_name")
            fanout_eligible, reviewer_excerpt = should_trigger_fanout(
                reviewer_comments_per_dim=reviewer_comments,
                chain_length=chain_length,
                chain_history=chain_history,
                keywords=fanout_settings.refine_trigger_keywords,
                dims=fanout_settings.refine_trigger_comment_dims,
                char_cap=fanout_settings.refine_trigger_comment_chars,
            )

            # ── Budget reservation (tiered, plan 39 §7.3 I1) ──────────
            # Snapshot ``original_reservation`` *before* any extension
            # so the cost-attribution invariant test can verify the
            # delta is fully accounted for via LLMCost batch_id rows.
            base_estimate = 0.20  # single-item refine
            fanout_pad = 0.0
            if (
                fanout_eligible
                and fanout_settings.enabled
                and fanout_settings.sites.get("refine_name", False)
            ):
                fanout_pad = fanout_settings.cost_estimate_for(escalate=escalate)
            estimated = base_estimate + fanout_pad
            lease = mgr.reserve(estimated, phase="refine_name")
            if lease is None:
                lease = mgr.reserve(0.0, phase="refine_name")
            original_reservation = lease.reserved if lease else 0.0

            # ── Optional fan-out (plan 39 Phase 1) ───────────────────
            fanout_evidence = ""
            fanout_run_id: str | None = None
            fanout_arm: str | None = None
            if fanout_eligible and lease is not None:
                fanout_arm = assign_arm(
                    sn_id,
                    chain_length,
                    arm_percent=fanout_settings.refine_fanout_arm_percent,
                )
                import uuid as _uuid

                fanout_run_id = str(_uuid.uuid4())
                physics_dom = item.get("physics_domain")
                if isinstance(physics_dom, list):
                    physics_dom = physics_dom[0] if physics_dom else None
                ids_filter = item.get("ids_name") or None
                candidate_ctx = CandidateContext(
                    sn_id=sn_id,
                    name=sn_id,
                    path=path,
                    description=item.get("description") or "",
                    physics_domain=physics_dom or "",
                    chain_length=chain_length,
                )
                scope = FanoutScope(
                    physics_domain=physics_dom,
                    ids_filter=ids_filter,
                )
                try:
                    fanout_evidence = await run_fanout(
                        site="refine_name",
                        candidate=candidate_ctx,
                        reviewer_excerpt=reviewer_excerpt,
                        scope=scope,
                        gc=gc,
                        parent_lease=lease,
                        settings=fanout_settings,
                        arm=fanout_arm,
                        escalate=escalate,
                        fanout_run_id=fanout_run_id,
                    )
                except Exception:
                    logger.exception(
                        "fan-out failed for %s — proceeding with empty evidence",
                        sn_id,
                    )
                    fanout_evidence = ""
            prompt_context["fanout_evidence"] = fanout_evidence

            # Load composition rules for system prompt
            from imas_codex.llm.prompt_loader import load_prompt_config as _lpc

            try:
                _rules_cfg = _lpc("sn_composition_rules")
                prompt_context["composition_rules"] = _rules_cfg.get(
                    "composition_rules", []
                )
            except Exception:
                prompt_context["composition_rules"] = []

            user_prompt = render_prompt("sn/refine_name_user", prompt_context)
            try:
                system_prompt = render_prompt("sn/refine_name_system", prompt_context)
            except Exception:
                logger.debug("refine_name: system prompt render failed for %s", sn_id)
                system_prompt = None

            # ── LLM call ──────────────────────────────────────────────
            try:
                _messages = (
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    if system_prompt
                    else [{"role": "user", "content": user_prompt}]
                )
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=_messages,
                    response_model=RefinedName,
                    service="standard-names",
                )

                # acall_llm_structured returns an LLMResult that still supports
                # legacy tuple unpacking. Normalize telemetry from the real
                # result object and fall back only for older tuple-shaped tests.
                result_obj, cost, _tokens = llm_out
                (
                    llm_tokens_in,
                    llm_tokens_out,
                    llm_tokens_cached_read,
                    llm_tokens_cached_write,
                ) = _extract_llm_telemetry(
                    llm_out,
                    parsed_obj=result_obj,
                    tokens_fallback=_tokens,
                )

                # Charge cost to lease.  Stamp ``batch_id`` with the
                # ``fanout_run_id`` whenever fan-out fired so the
                # ``Fanout`` ↔ ``LLMCost`` join (plan 39 §8.3) works
                # for both arms.
                if lease:
                    _event = LLMCostEvent(
                        model=model,
                        tokens_in=llm_tokens_in,
                        tokens_out=llm_tokens_out,
                        tokens_cached_read=llm_tokens_cached_read,
                        tokens_cached_write=llm_tokens_cached_write,
                        sn_ids=(result_obj.name,),
                        phase=(
                            "refine_name+fanout" if fanout_run_id else "refine_name"
                        ),
                        service="standard-names",
                        batch_id=fanout_run_id,
                    )
                    lease.charge_event(cost, _event)

                # ── Phase 1.5 dup guard (plan 39 §5.2) ────────────
                # Deterministic name-key lookup AFTER B12 final
                # candidate, BEFORE persisting.  On hit we drop the
                # candidate and emit a ``dup_prevented`` log line.
                # Excludes ``old_name`` so the chain's predecessor is
                # never treated as a self-collision.
                dup_id = None
                try:
                    dup_id = find_name_key_duplicate(
                        gc,
                        result_obj.name,
                        exclude=sn_id,
                    )
                except Exception:
                    logger.debug(
                        "dup guard failed for %s — proceeding with persist",
                        sn_id,
                    )
                if dup_id:
                    logger.info(
                        "refine_name dup_prevented: %s → %s collides with %s",
                        sn_id,
                        result_obj.name,
                        dup_id,
                    )
                    # Release claim back to 'reviewed' so the cycle
                    # can pick it up again (with fresh feedback) or
                    # be marked superseded by manual review.
                    token = item.get("claim_token") or ""
                    try:
                        await _asyncio.to_thread(
                            release_refine_name_failed_claims,
                            sn_ids=[sn_id],
                            token=token,
                        )
                    except Exception:
                        logger.debug(
                            "release after dup_prevented failed for %s",
                            sn_id,
                        )
                    if on_event is not None:
                        on_event(
                            {
                                "pool": "refine_name",
                                "name": sn_id,
                                "old_name": sn_id,
                                "duplicate_of": dup_id,
                                "outcome": "dup_prevented",
                                "model": model,
                                "cost": cost,
                            }
                        )
                    continue

                # ── Persist ───────────────────────────────────────────
                await _asyncio.to_thread(
                    persist_refined_name,
                    old_name=sn_id,
                    new_name=result_obj.name,
                    description=normalize_prose_spelling(result_obj.description),
                    kind=result_obj.kind,
                    unit=item.get("unit"),
                    physics_domain=(
                        (
                            (item.get("physics_domain") or [None])[0]
                            if isinstance(item.get("physics_domain"), list)
                            else item.get("physics_domain")
                        )
                        or None
                    ),
                    source_domains=(
                        item.get("source_domains")
                        if isinstance(item.get("source_domains"), list)
                        else None
                    ),
                    old_chain_length=chain_length,
                    model=model,
                    reason=result_obj.reason,
                    escalated=escalate,
                    run_id=mgr.run_id,
                )
                processed += 1
                logger.info(
                    "refine_name: %s → %s (chain_length=%d, model=%s)",
                    sn_id,
                    result_obj.name,
                    chain_length + 1,
                    model,
                )

                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": result_obj.name,
                            "old_name": sn_id,
                            "new_name": result_obj.name,
                            "chain_length": chain_length + 1,
                            "escalated": escalate,
                            "model": model,
                            "cost": cost,
                            "fanout_run_id": fanout_run_id,
                            "fanout_arm": fanout_arm,
                            "original_reservation": original_reservation,
                            "llm_tokens_in": llm_tokens_in,
                            "llm_tokens_out": llm_tokens_out,
                            "llm_tokens_cached_read": llm_tokens_cached_read,
                            "llm_tokens_cached_write": llm_tokens_cached_write,
                        }
                    )

            except Exception as exc:
                logger.exception("refine_name failed for %s", sn_id)
                token = item.get("claim_token") or ""
                # Vocab-gap errors and self-referential refines are
                # deterministic — the LLM will keep producing the same
                # output.  Mark exhausted instead of reverting to
                # 'reviewed' (which re-enters the refine loop).
                _exc_str = str(exc)
                is_terminal = (
                    "not a registered" in _exc_str
                    or "self-referential REFINED_FROM" in _exc_str
                    or "kind must be one of" in _exc_str
                    or "validation error for RefinedName" in _exc_str
                )
                try:
                    if is_terminal:
                        await _asyncio.to_thread(
                            _mark_refine_vocab_gap_exhausted,
                            sn_id=sn_id,
                            token=token,
                            error_msg=str(exc)[:500],
                        )
                    else:
                        await _asyncio.to_thread(
                            release_refine_name_failed_claims,
                            sn_ids=[sn_id],
                            token=token,
                        )
                except Exception:
                    logger.debug(
                        "release/exhaust also failed for %s",
                        sn_id,
                    )
            finally:
                # Always return the unused portion of the lease to the pool.
                # Without this the unspent remainder leaks every iteration
                # and the pool exhausts at ~25 % of cost_limit.
                if lease is not None:
                    try:
                        lease.release_unused()
                    except Exception:
                        pass

    return processed


# =============================================================================
# RD-quorum helper (shared by review_name + review_docs pool workers)
# =============================================================================


async def _run_rd_quorum_cycles(
    *,
    sn_id: str,
    review_axis: str,
    response_model: Any,
    user_prompt: str,
    system_prompt: str,
    models: list[str],
    disagreement_threshold: float,
    rubric_dims: tuple[str, ...],
    lease: Any,
    phase: str,
    acall_llm_structured: Callable[..., Any],
) -> dict[str, Any] | None:
    """Run the configured RD-quorum reviewer chain for a single StandardName.

    Calls each configured model in turn (cycle 0 = primary, cycle 1 =
    secondary, cycle 2 = escalator). Cycle 1 always runs when ≥2 models
    are configured. Cycle 2 runs only when (a) ≥3 models are configured
    AND (b) any rubric dimension differs between cycles 0 and 1 by more
    than *disagreement_threshold* (after normalising 0–20 scores to 0–1).

    Persists no graph state directly — the caller is responsible for
    writing the returned ``records`` list via
    :func:`~imas_codex.standard_names.graph_ops.write_reviews` and for
    the SN-side stage transition.

    Each cycle's cost is charged to *lease* via a fresh
    :class:`~imas_codex.standard_names.budget.LLMCostEvent` tagged with
    the cycle id (``c0``/``c1``/``c2``).

    Returns
    -------
    dict | None
        ``None`` when no cycle produced a parseable result (caller
        should release the claim). Otherwise a dict with keys:

        - ``records`` — list of Review record dicts (one per cycle) ready
          for :func:`write_reviews`. Share a single ``review_group_id``.
        - ``winning_score`` — float 0-1, the consensus score that should
          mirror onto the SN axis slot.
        - ``winning_scores`` — dict[str, float] of per-dim consensus.
        - ``winning_comments`` — str.
        - ``winning_comments_per_dim`` — dict | None.
        - ``canonical_model`` — str, the model attribution for the SN
          axis slot (always cycle 0's model — the chain anchor).
        - ``resolution_method`` — one of
          ``single_review`` / ``quorum_consensus`` /
          ``authoritative_escalation`` / ``max_cycles_reached``.
        - ``total_cost`` — float, sum of all cycles.
        - ``total_tokens_in`` / ``total_tokens_out`` — int.
    """
    import uuid as _uuid
    from datetime import datetime as _dt

    from imas_codex.standard_names.budget import LLMCostEvent

    review_group_id = str(_uuid.uuid4())
    cycles: list[dict[str, Any]] = []  # parsed cycle results (cycle_index 0..N)
    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    async def _run_cycle(cycle_idx: int, model: str) -> dict[str, Any] | None:
        """Single LLM call. Returns parsed cycle dict or None on failure."""
        nonlocal total_cost, total_tokens_in, total_tokens_out
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=response_model,
                service="standard-names",
            )
            result_obj, cost, _tokens = llm_out
        except Exception:
            logger.exception(
                "rd_quorum %s cycle %d failed for %s (model=%s)",
                review_axis,
                cycle_idx,
                sn_id,
                model,
            )
            return None

        # Unwrap batch response: prompts ask for {"reviews": [...]}; we pass
        # one item at a time so reviews[0] is the per-item ReviewItem object.
        reviews_list = getattr(result_obj, "reviews", None)
        if reviews_list:
            try:
                result_obj = reviews_list[0]
            except (IndexError, TypeError):
                logger.warning(
                    "rd_quorum %s cycle %d returned empty reviews list for %s (model=%s)",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                )
                return None

        tokens_in = int(getattr(llm_out, "input_tokens", 0) or 0)
        tokens_out = int(getattr(llm_out, "output_tokens", 0) or 0)
        cached_read = int(getattr(llm_out, "cache_read_tokens", 0) or 0)
        cached_write = int(getattr(llm_out, "cache_creation_tokens", 0) or 0)

        total_cost += cost
        total_tokens_in += tokens_in
        total_tokens_out += tokens_out

        if lease is not None:
            try:
                lease.charge_event(
                    cost,
                    LLMCostEvent(
                        model=model,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        tokens_cached_read=cached_read,
                        tokens_cached_write=cached_write,
                        sn_ids=(sn_id,),
                        cycle=f"c{cycle_idx}",
                        phase=phase,
                        service="standard-names",
                    ),
                )
            except Exception:
                logger.debug(
                    "rd_quorum charge_event failed for %s c%d",
                    sn_id,
                    cycle_idx,
                    exc_info=True,
                )

        # Extract score and per-dim scores.
        try:
            score = float(result_obj.scores.score)
        except Exception:
            score = 0.0
        try:
            scores_dict = result_obj.scores.model_dump()
        except Exception:
            scores_dict = {}

        comments = getattr(result_obj, "reasoning", None) or ""
        comments_per_dim = None
        try:
            if getattr(result_obj, "comments", None) is not None:
                comments_per_dim = result_obj.comments.model_dump()
        except Exception:
            comments_per_dim = None

        return {
            "cycle_index": cycle_idx,
            "model": model,
            "score": score,
            "scores": scores_dict,
            "comments": comments,
            "comments_per_dim": comments_per_dim,
            "cost": cost,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cached_read": cached_read,
            "cached_write": cached_write,
        }

    # ── Cycle 0 (primary, blind) ───────────────────────────────────────
    c0 = await _run_cycle(0, models[0])
    if c0 is not None:
        cycles.append(c0)

    # ── Cycle 1 (secondary, blind) — runs whenever ≥ 2 models ──────────
    c1 = None
    if len(models) >= 2:
        c1 = await _run_cycle(1, models[1])
        if c1 is not None:
            cycles.append(c1)

    if not cycles:
        return None

    # ── Per-dimension disagreement (cycle 2 gate) ──────────────────────
    disagreement = False
    if c0 is not None and c1 is not None:
        for dim in rubric_dims:
            s0 = float(c0["scores"].get(dim, 0)) / 20.0
            s1 = float(c1["scores"].get(dim, 0)) / 20.0
            if abs(s0 - s1) > disagreement_threshold:
                disagreement = True
                break

    # ── Cycle 2 (escalator) ────────────────────────────────────────────
    c2 = None
    if disagreement and len(models) >= 3:
        c2 = await _run_cycle(2, models[2])
        if c2 is not None:
            cycles.append(c2)

    # ── Determine winning score + resolution method ────────────────────
    canonical_model = models[0]  # SN axis attribution always cycle-0 model
    if len(cycles) == 1:
        # Only cycle 0 succeeded (or chain is single-model)
        winning = cycles[0]
        if len(models) == 1:
            resolution_method = "single_review"
        else:
            # Cycle 1 failed — degrade to single_review semantics
            resolution_method = "single_review"
        winning_score = float(winning["score"])
        winning_scores = dict(winning["scores"])
        winning_comments = winning["comments"]
        winning_comments_per_dim = winning["comments_per_dim"]
    elif c2 is not None:
        # Escalator authoritative
        resolution_method = "authoritative_escalation"
        winning_score = float(c2["score"])
        winning_scores = dict(c2["scores"])
        winning_comments = c2["comments"]
        winning_comments_per_dim = c2["comments_per_dim"]
    else:
        # 2 cycles only (or escalator skipped/failed) — mean of c0 + c1
        winning_score = (float(c0["score"]) + float(c1["score"])) / 2.0
        winning_scores = {}
        for dim in rubric_dims:
            v0 = float(c0["scores"].get(dim, 0))
            v1 = float(c1["scores"].get(dim, 0))
            winning_scores[dim] = (v0 + v1) / 2.0
        # Merge comments preserving both reviewers' reasoning
        c0_comments = c0["comments"] or ""
        c1_comments = c1["comments"] or ""
        if c0_comments and c1_comments and c0_comments != c1_comments:
            winning_comments = f"[Primary] {c0_comments}\n[Secondary] {c1_comments}"
        else:
            winning_comments = c1_comments or c0_comments
        winning_comments_per_dim = c1["comments_per_dim"] or c0["comments_per_dim"]
        if disagreement:
            # Disputed but no escalator available
            resolution_method = "max_cycles_reached"
        else:
            resolution_method = "quorum_consensus"

    # ── Build Review records ───────────────────────────────────────────
    role_by_idx = {0: "primary", 1: "secondary", 2: "escalator"}
    method_by_idx = {0: None, 1: None, 2: None}
    if resolution_method == "single_review":
        method_by_idx[0] = "single_review"
    elif resolution_method == "authoritative_escalation":
        method_by_idx[2] = "authoritative_escalation"
    elif resolution_method == "max_cycles_reached":
        method_by_idx[1] = "max_cycles_reached"
    elif resolution_method == "quorum_consensus":
        method_by_idx[1] = "quorum_consensus"

    now_iso = _dt.utcnow().isoformat()
    records: list[dict[str, Any]] = []
    for c in cycles:
        idx = c["cycle_index"]
        score = float(c["score"])
        if score >= 0.85:
            tier = "outstanding"
        elif score >= 0.60:
            tier = "good"
        elif score >= 0.40:
            tier = "inadequate"
        else:
            tier = "poor"
        scores_json = json.dumps(c["scores"]) if c["scores"] else "{}"
        comments_per_dim_json = (
            json.dumps(c["comments_per_dim"]) if c["comments_per_dim"] else None
        )
        records.append(
            {
                "id": f"{sn_id}:{review_axis}:{review_group_id}:{idx}",
                "standard_name_id": sn_id,
                "model": c["model"],
                "reviewer_model": c["model"],
                "model_family": "other",
                "is_canonical": idx == 0,
                "score": score,
                "scores_json": scores_json,
                "tier": tier,
                "comments": c["comments"] or "",
                "comments_per_dim_json": comments_per_dim_json,
                "suggested_name": "",
                "suggestion_justification": "",
                "reviewed_at": now_iso,
                "review_axis": review_axis,
                "cycle_index": idx,
                "review_group_id": review_group_id,
                "resolution_role": role_by_idx.get(idx, "primary"),
                "resolution_method": method_by_idx.get(idx),
                "llm_model": c["model"],
                "llm_cost": c["cost"],
                "llm_tokens_in": c["tokens_in"],
                "llm_tokens_out": c["tokens_out"],
                "llm_tokens_cached_read": c["cached_read"],
                "llm_tokens_cached_write": c["cached_write"],
                "llm_at": now_iso,
                "llm_service": "standard-names",
            }
        )

    return {
        "records": records,
        "winning_score": float(winning_score),
        "winning_scores": winning_scores,
        "winning_comments": winning_comments,
        "winning_comments_per_dim": winning_comments_per_dim,
        "canonical_model": canonical_model,
        "resolution_method": resolution_method,
        "total_cost": total_cost,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "review_group_id": review_group_id,
        "disagreement": disagreement,
    }


async def process_review_name_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of drafted StandardNames for name review (RD-quorum).

    For each item in the batch:

    1. Build a name-axis review prompt via ``render_prompt("sn/review_names_user", ...)``.
    2. Run the configured RD-quorum chain
       (``[sn.review.names].models``):

       - **Cycle 0** (primary, blind) — always runs.
       - **Cycle 1** (secondary, blind) — runs when ≥ 2 models configured.
       - **Cycle 2** (escalator) — runs only when per-dimension disagreement
         between cycles 0 and 1 exceeds the configured threshold AND ≥ 3
         models are configured.
    3. Persist each cycle as a separate ``StandardNameReview`` node with
       proper ``cycle_index`` (0/1/2), ``review_group_id`` (UUID shared
       across cycles for the same SN), ``resolution_role``
       (primary/secondary/escalator) and ``resolution_method``
       (single_review / quorum_consensus / authoritative_escalation /
       max_cycles_reached).
    4. Persist via :func:`~imas_codex.standard_names.graph_ops.persist_reviewed_name`
       with ``skip_review_node=True`` (review nodes already written) and the
       *winning* score so the SN axis slots mirror the consensus and the
       state machine transitions ``name_stage`` to ``'accepted'``,
       ``'reviewed'`` or ``'exhausted'``.
    5. Update review aggregates so ``review_count`` /
       ``review_disagreement`` reflect the multi-cycle group.
    6. On LLM error: release the claim via
       :func:`~imas_codex.standard_names.graph_ops.release_review_names_failed_claims`.

    Returns count of items successfully reviewed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_sn_review_disagreement_threshold,
        get_sn_review_names_models,
    )
    from imas_codex.standard_names.defaults import (
        DEFAULT_MIN_SCORE,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        persist_reviewed_name,
        release_review_names_failed_claims,
        update_review_aggregates,
    )
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnlyBatch,
    )

    # ── Resolve review model chain ─────────────────────────────────────
    try:
        review_models = get_sn_review_names_models()
    except (ValueError, IndexError):
        review_models = []
    if not review_models:
        review_models = [DEFAULT_ESCALATION_MODEL]

    try:
        disagreement_threshold = get_sn_review_disagreement_threshold()
    except Exception:
        disagreement_threshold = 0.20

    processed = 0

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""

        # ── Skip review for quarantined names ──────────────────────────
        # Names that failed ISN 3-layer validation already have a low-tier
        # verdict — running RD-quorum on them is pure waste. Persist a
        # zero-score review (cycle 0 only, single_review) directly.
        if item.get("validation_status") == "quarantined":
            try:
                persist_reviewed_name(
                    sn_id=sn_id,
                    claim_token=claim_token,
                    score=0.0,
                    scores={
                        "grammar": 0.0,
                        "semantic": 0.0,
                        "convention": 0.0,
                        "completeness": 0.0,
                    },
                    comments="quarantined: skipped reviewer LLM call",
                    comments_per_dim=None,
                    model="(skipped: quarantined)",
                    llm_cost=0.0,
                    llm_tokens_in=0,
                    llm_tokens_out=0,
                    llm_tokens_cached_read=0,
                    llm_tokens_cached_write=0,
                    llm_service="standard-names",
                    run_id=mgr.run_id,
                )
                processed += 1
                if on_event:
                    on_event(
                        {
                            "type": "review_name_skipped_quarantined",
                            "sn_id": sn_id,
                        }
                    )
            except Exception:
                logger.debug(
                    "review_name: persist failed for quarantined %s",
                    sn_id,
                    exc_info=True,
                )
            continue

        # ── Semantic similarity gate (pre-LLM) ────────────────────────
        # Compute cosine similarity between name-as-text and description.
        # Names below the critical threshold are semantically ambiguous
        # (e.g. "co_passing_density" — density of what?).  Skip the
        # expensive LLM review and persist a synthetic low score that
        # routes the name to the refine_name pool.
        #
        # EXCEPT for deterministic parents: their description is the
        # canonical placeholder until ``GENERATE_DOCS`` runs, so the
        # cosine similarity is meaningless. Cosine sim between a
        # registered base token (``magnetic_field``) and a generic
        # placeholder string is always low — the gate would fire on
        # every deterministic parent that ever reached this point.
        # Deterministic parents auto-accept on the name axis anyway
        # (``seed_parent_sources`` writes ``name_stage='accepted'``);
        # this skip is defensive in case any drift puts one at
        # ``drafted``.
        from imas_codex.standard_names.audits import semantic_similarity_check
        from imas_codex.standard_names.defaults import (
            SEMANTIC_SIM_CRITICAL,
            SEMANTIC_SIM_SYNTHETIC_SCORE,
            SEMANTIC_SIM_WARNING,
        )

        sem_sim: float | None = None
        sem_issues: list[str] = []
        is_derived = item.get("origin") == "derived"
        if is_derived:
            # ── Phase 5: desc-name similarity gate for derived parents ────
            # For derived parents, the description may have been seeded from
            # a placeholder or a DD-derived string that doesn't align with
            # the name.  Compute desc_name_similarity and, if below
            # threshold, route to REFINE_DOCS instead of name scoring.
            # This keeps description-quality failures off the name axis.
            #
            # Unlike the general semantic_similarity_check (which uses the
            # quarantine-oriented SEMANTIC_SIM_CRITICAL threshold and routes
            # to REFINE_NAME), this gate uses the dedicated
            # desc_name_similarity_threshold and routes to REFINE_DOCS.
            from imas_codex.standard_names.desc_name_sim import (
                compute_desc_name_similarity,
                should_route_to_refine_docs,
            )

            desc_sim: float | None = None
            try:
                desc_sim = await _asyncio.to_thread(
                    compute_desc_name_similarity,
                    sn_id,
                    item.get("description") or "",
                )
            except Exception:
                logger.debug(
                    "review_name: desc_name_similarity failed for derived %s",
                    sn_id,
                    exc_info=True,
                )

            if desc_sim is not None and should_route_to_refine_docs(desc_sim):
                logger.info(
                    "review_name: desc-name sim gate FIRED for derived %s "
                    "(sim=%.3f) — routing to REFINE_DOCS",
                    sn_id,
                    desc_sim,
                )
                from imas_codex.standard_names.graph_ops import mark_for_refine_docs

                try:
                    mark_for_refine_docs(
                        sn_id,
                        desc_name_similarity=desc_sim,
                        claim_token=claim_token,
                    )
                    processed += 1
                    if on_event:
                        on_event(
                            {
                                "type": "review_name_desc_sim_gate_routed_docs",
                                "sn_id": sn_id,
                                "desc_name_similarity": desc_sim,
                            }
                        )
                except Exception:
                    logger.debug(
                        "review_name: mark_for_refine_docs failed for %s",
                        sn_id,
                        exc_info=True,
                    )
                continue

            # Below threshold not triggered (sim is high enough, or embed
            # failed — treat as "gate not applicable"). Log and fall through
            # to normal name scoring.
            if desc_sim is not None:
                logger.debug(
                    "review_name: derived %s desc-name sim=%.3f (gate clear)",
                    sn_id,
                    desc_sim,
                )
        else:
            try:
                sem_sim, sem_issues = await _asyncio.to_thread(
                    semantic_similarity_check,
                    sn_id,
                    item.get("description") or "",
                )
            except Exception:
                logger.debug(
                    "review_name: semantic_similarity_check failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # Persist semantic_sim on the node regardless of outcome
        if sem_sim is not None:
            try:
                from imas_codex.graph.client import GraphClient as _GC

                def _store_sim(_id=sn_id, _sim=sem_sim) -> None:
                    with _GC() as gc:
                        gc.query(
                            "MATCH (sn:StandardName {id: $id}) "
                            "SET sn.semantic_sim = $sim",
                            id=_id,
                            sim=_sim,
                        )

                await _asyncio.to_thread(_store_sim)
            except Exception:
                pass  # best-effort

        # Critical: skip LLM review, force into refine pipeline
        if sem_sim is not None and sem_sim < SEMANTIC_SIM_CRITICAL:
            logger.info(
                "review_name: semantic gate FAILED for %s (sim=%.3f < %.2f) — "
                "routing to refine",
                sn_id,
                sem_sim,
                SEMANTIC_SIM_CRITICAL,
            )
            try:
                persist_reviewed_name(
                    sn_id=sn_id,
                    claim_token=claim_token,
                    score=SEMANTIC_SIM_SYNTHETIC_SCORE,
                    scores={
                        "grammar": 15.0 / 20.0,
                        "semantic": 2.0 / 20.0,
                        "convention": 15.0 / 20.0,
                        "completeness": 5.0 / 20.0,
                    },
                    comments=(
                        f"semantic_similarity_gate: sim={sem_sim:.3f} below "
                        f"critical {SEMANTIC_SIM_CRITICAL:.2f}. Name is "
                        f"semantically ambiguous — a reader cannot determine "
                        f"the measured quantity from the name alone."
                    ),
                    comments_per_dim={
                        "semantic": (
                            f"Name-to-description embedding similarity is "
                            f"{sem_sim:.3f}, well below the {SEMANTIC_SIM_CRITICAL:.2f} "
                            f"threshold. The name does not stand alone."
                        ),
                    },
                    model="(semantic_similarity_gate)",
                    llm_cost=0.0,
                    llm_tokens_in=0,
                    llm_tokens_out=0,
                    llm_tokens_cached_read=0,
                    llm_tokens_cached_write=0,
                    llm_service="standard-names",
                    run_id=mgr.run_id,
                )
                processed += 1
                if on_event:
                    on_event(
                        {
                            "type": "review_name_semantic_gate_failed",
                            "sn_id": sn_id,
                            "semantic_sim": sem_sim,
                        }
                    )
            except Exception:
                logger.debug(
                    "review_name: persist failed for semantic-gated %s",
                    sn_id,
                    exc_info=True,
                )
            continue

        # Inject warning into item context for the reviewer
        if sem_sim is not None and sem_sim < SEMANTIC_SIM_WARNING:
            sem_warning_note = (
                f"⚠️ SEMANTIC AMBIGUITY WARNING: Name-to-description "
                f"embedding similarity is {sem_sim:.3f} (threshold "
                f"{SEMANTIC_SIM_WARNING:.2f}). Verify that the name is "
                f"self-describing — can someone determine the measured "
                f"quantity from the name alone?"
            )
            item["semantic_warning"] = sem_warning_note

        # ── Build prompt context ───────────────────────────────────────
        from imas_codex.standard_names.context import (
            _build_enum_lists,
            fetch_review_neighbours,
        )

        try:
            neighbours = fetch_review_neighbours(item)
        except Exception:
            logger.debug(
                "review_name: neighbour fetch failed for %s", sn_id, exc_info=True
            )
            neighbours = {
                "vector_neighbours": [],
                "same_base_neighbours": [],
                "same_path_neighbours": [],
            }

        # ── Load scored examples for reviewer calibration ──────────────
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.example_loader import load_review_examples

        try:
            _item_domain = item.get("physics_domain") or ""

            def _load_review_examples(domain: str = _item_domain) -> list:
                with GraphClient() as _gc:
                    return load_review_examples(
                        _gc, physics_domains=[domain], axis="name"
                    )

            review_scored_examples = await _asyncio.to_thread(_load_review_examples)
        except Exception:
            logger.debug(
                "review_name: scored-example load failed for %s", sn_id, exc_info=True
            )
            review_scored_examples = []

        # ── DD path context for reviewer (keywords, clusters, version history) ─
        _source_paths = item.get("source_paths") or []
        if _source_paths:
            _first_src = (
                _source_paths[0] if isinstance(_source_paths, list) else _source_paths
            )
            if isinstance(_first_src, str):
                _first_src = strip_dd_prefix(_first_src)
            if _first_src:
                try:
                    from imas_codex.graph.client import GraphClient as _GCDD

                    def _do_dd_ctx(_p=_first_src, _it=item) -> None:
                        with _GCDD() as _gc:
                            _enrich_dd_path_context(_gc, _it, _p)

                    await _asyncio.to_thread(_do_dd_ctx)
                except Exception:
                    logger.debug(
                        "review_name: dd_path_context failed for %s",
                        sn_id,
                        exc_info=True,
                    )

        prompt_context: dict[str, Any] = {
            "items": [item],
            **neighbours,
            **_build_enum_lists(),
            "review_scored_examples": review_scored_examples,
        }
        try:
            user_prompt = render_prompt("sn/review_names_user", prompt_context)
            system_prompt = render_prompt("sn/review_names_system", prompt_context)
        except Exception:
            logger.debug("review_name: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Review the standard name: {item.get('id', sn_id)}\n"
                f"Description: {item.get('description', '')}"
            )
            system_prompt = (
                "You are a quality reviewer for IMAS standard names in "
                "fusion plasma physics."
            )

        # ── Budget reservation (cover all cycles) ──────────────────────
        # Per-item cost ~$0.05; reserve worst-case across the chain
        # (every model called) with a 1.3× safety margin.
        per_item_estimate = 0.05
        worst_case = per_item_estimate * len(review_models) * 1.3
        lease = mgr.reserve(worst_case, phase="review_name")
        if lease is None:
            lease = mgr.reserve(0.0, phase="review_name")

        # ── RD-quorum cycles ──────────────────────────────────────────
        try:
            quorum = await _run_rd_quorum_cycles(
                sn_id=sn_id,
                review_axis="names",
                response_model=StandardNameQualityReviewNameOnlyBatch,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                models=review_models,
                disagreement_threshold=disagreement_threshold,
                rubric_dims=("grammar", "semantic", "convention", "completeness"),
                lease=lease,
                phase="review_name",
                acall_llm_structured=acall_llm_structured,
            )

            if quorum is None:
                # No cycles produced a valid result — release claim.
                if claim_token:
                    try:
                        await _asyncio.to_thread(
                            release_review_names_failed_claims,
                            sn_ids=[sn_id],
                            claim_token=claim_token,
                            from_stage="drafted",
                            to_stage="drafted",
                        )
                    except Exception:
                        logger.debug(
                            "release_review_names_failed_claims also failed for %s",
                            sn_id,
                        )
                continue

            # ── Persist all cycle Review records ──────────────────────
            from imas_codex.standard_names.graph_ops import write_reviews

            await _asyncio.to_thread(write_reviews, quorum["records"])

            # ── Stage transition with winning score ────────────────────
            new_stage = await _asyncio.to_thread(
                persist_reviewed_name,
                sn_id=sn_id,
                claim_token=claim_token,
                score=quorum["winning_score"],
                scores=quorum["winning_scores"],
                comments=quorum["winning_comments"],
                comments_per_dim=quorum["winning_comments_per_dim"],
                model=quorum["canonical_model"],
                min_score=DEFAULT_MIN_SCORE,
                rotation_cap=DEFAULT_REFINE_ROTATIONS,
                llm_cost=quorum["total_cost"],
                llm_tokens_in=quorum["total_tokens_in"],
                llm_tokens_out=quorum["total_tokens_out"],
                llm_tokens_cached_read=0,
                llm_tokens_cached_write=0,
                llm_service="standard-names",
                run_id=mgr.run_id,
                skip_review_node=True,
            )

            # ── Update aggregates so review_count / disagreement reflect group ─
            try:
                await _asyncio.to_thread(update_review_aggregates, [sn_id])
            except Exception:
                logger.debug(
                    "update_review_aggregates failed for %s", sn_id, exc_info=True
                )

            if new_stage:
                processed += 1
                _comment_log = (quorum["winning_comments"] or "")[:80]
                logger.info(
                    "review_name: %s → %s (score=%.3f, cycles=%d, method=%s) %s",
                    sn_id,
                    new_stage,
                    quorum["winning_score"],
                    len(quorum["records"]),
                    quorum["resolution_method"],
                    _comment_log,
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "review_name",
                            "name": sn_id,
                            "score": quorum["winning_score"],
                            "comment": quorum["winning_comments"] or "",
                            "stage": new_stage,
                            "model": quorum["canonical_model"],
                            "cost": quorum["total_cost"],
                            "cycles": len(quorum["records"]),
                            "resolution_method": quorum["resolution_method"],
                        }
                    )
            else:
                logger.debug("review_name: %s persist no-op (token mismatch?)", sn_id)

        except Exception:
            logger.exception("review_name failed for %s", sn_id)
            token = item.get("claim_token") or ""
            if token:
                try:
                    await _asyncio.to_thread(
                        release_review_names_failed_claims,
                        sn_ids=[sn_id],
                        claim_token=token,
                        from_stage="drafted",
                        to_stage="drafted",
                    )
                except Exception:
                    logger.debug(
                        "release_review_names_failed_claims also failed for %s",
                        sn_id,
                    )
        finally:
            # Always return the unused portion of the lease to the pool.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


# =============================================================================
# DD context enrichment for generate_docs
# =============================================================================

_DD_PATH_CONTEXT_QUERY = """
MATCH (n:IMASNode {id: $path})
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WHERE c.scope IN ['global', 'domain']
WITH n, collect(DISTINCT {
    label: c.label, description: c.description, scope: c.scope
})[0..3] AS clusters
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (vc:IMASNodeChange)-[:FOR_IMAS_PATH]->(n)
WHERE vc.change_type IN [
    'path_added', 'cocos_transformation_type', 'sign_convention',
    'units', 'path_renamed', 'definition_clarification'
]
WITH n, clusters, parent,
     collect(DISTINCT {change_id: vc.id, change_type: vc.change_type}) AS changes
RETURN n.keywords AS keywords,
       parent.description AS parent_description,
       clusters,
       changes
"""

_DOCS_GEN_ENRICH_QUERY = """
MATCH (sn:StandardName {id: $sn_id})
OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(imas:IMASNode)
RETURN sn.source_paths AS source_paths,
       sn.cocos_transformation_type AS cocos_label,
       collect(DISTINCT {
           id: imas.id,
           documentation: coalesce(imas.documentation, ''),
           description: coalesce(imas.description, ''),
           alias: imas.alias,
           unit: coalesce(imas.unit, '')
       }) AS dd_nodes
"""

_DOCS_GEN_NEARBY_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage = 'accepted'
  AND sn.physics_domain = $domain
  AND sn.id <> $sn_id
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS id,
       sn.description AS description,
       sn.kind AS kind,
       coalesce(u.id, sn.unit) AS unit
LIMIT 20
"""


def _enrich_dd_path_context(gc: Any, item: dict, source_path: str) -> None:
    """Enrich an item dict with DD path context (keywords, clusters, version history).

    Fetches lightweight context from the IMASNode for *source_path* and populates:

    - ``dd_keywords``: list of keyword strings from the IMASNode.
    - ``dd_clusters``: list of ``{label, description, scope}`` dicts.
    - ``dd_version_history``: list of ``{version, change_type}`` dicts.
    - ``dd_parent_description``: description of the parent structure node.

    Best-effort — failures are debug-logged and the item is left without
    the missing key.  Modifies *item* in-place.
    """
    try:
        rows = list(gc.query(_DD_PATH_CONTEXT_QUERY, path=source_path))
    except Exception:
        logger.debug(
            "_enrich_dd_path_context: query failed for %s", source_path, exc_info=True
        )
        return
    if not rows:
        return
    row = rows[0]

    # Keywords
    kw = row.get("keywords")
    if kw:
        if isinstance(kw, str):
            kw = [k.strip() for k in kw.split(",") if k.strip()]
        if kw:
            item["dd_keywords"] = kw

    # Parent description
    parent_desc = row.get("parent_description")
    if parent_desc:
        item["dd_parent_description"] = parent_desc

    # Clusters
    clusters_raw = row.get("clusters") or []
    clusters = [
        {
            "label": c["label"],
            "description": c.get("description") or "",
            "scope": c.get("scope") or "",
        }
        for c in clusters_raw
        if c.get("label")
    ]
    if clusters:
        item["dd_clusters"] = clusters

    # Version history
    changes = row.get("changes") or []
    version_history = []
    for ch in changes:
        change_id = ch.get("change_id") or ""
        change_type = ch.get("change_type") or ""
        parts = change_id.rsplit(":", 1)
        version = parts[-1] if len(parts) >= 2 else ""
        if version and change_type:
            version_history.append({"version": version, "change_type": change_type})
    if version_history:
        item["dd_version_history"] = version_history


def _enrich_for_docs_gen(
    gc: Any, items: list[dict], cocos_params: dict | None = None
) -> None:
    """Enrich generate_docs items with DD source context.

    Populates per-item keys (all guarded by ``if`` in the prompt template):

    - ``source_paths``: bare DD paths (``dd:`` prefix stripped).
    - ``dd_source_docs``: list of ``{id, documentation, unit}`` from linked
      :class:`~imas_codex.graph.models.IMASNode` nodes.
    - ``dd_aliases``: list of alias strings from IMASNodes.
    - ``related_neighbours``: graph-relationship neighbours derived from the
      first DD source path (via :func:`_related_path_neighbours`).
    - ``nearest_peers``: vector-similar :class:`StandardName` neighbours from
      :func:`_search_nearby_names` formatted as ``{tag, unit, physics_domain,
      doc_short, cocos_label}``.
    - ``cocos_label``: COCOS transformation type from the SN node.
    - ``cocos_guidance``: rendered sign convention guidance for the LLM.

    Modifies *items* in-place.  Never raises — individual item failures are
    debug-logged and the item is left without the missing key.
    """
    for item in items:
        sn_id = item.get("id")
        if not sn_id:
            continue

        # ── 1. Source paths + linked IMASNode documentation ───────────
        try:
            rows = list(gc.query(_DOCS_GEN_ENRICH_QUERY, sn_id=sn_id))
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: graph query failed for %s",
                sn_id,
                exc_info=True,
            )
            continue

        if not rows:
            continue
        row = rows[0]

        raw_paths = row.get("source_paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        source_paths = [
            strip_dd_prefix(p) for p in raw_paths if p and isinstance(p, str)
        ]
        if source_paths:
            item["source_paths"] = source_paths

        dd_nodes = [n for n in (row.get("dd_nodes") or []) if n and n.get("id")]
        if dd_nodes:
            item["dd_source_docs"] = [
                {
                    "id": n["id"],
                    "documentation": (
                        n.get("documentation") or n.get("description") or ""
                    ),
                    "unit": n.get("unit") or "",
                }
                for n in dd_nodes[:5]
            ]
            aliases = [n["alias"] for n in dd_nodes if n.get("alias")]
            if aliases:
                item["dd_aliases"] = aliases

        # ── 1b. COCOS sign convention context ─────────────────────────
        cocos_label = row.get("cocos_label")
        if cocos_label:
            item["cocos_label"] = cocos_label
            if cocos_params:
                try:
                    from imas_codex.standard_names.context import (
                        render_cocos_guidance,
                    )

                    item["cocos_guidance"] = render_cocos_guidance(
                        cocos_label, cocos_params
                    )
                except Exception:
                    logger.debug(
                        "_enrich_for_docs_gen: cocos guidance render failed for %s",
                        sn_id,
                        exc_info=True,
                    )

        # ── 2. Graph-relationship neighbours from first DD source path ─
        if source_paths:
            try:
                related = _related_path_neighbours(gc, source_paths[0])
                if related:
                    item["related_neighbours"] = related
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: related_neighbours failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # ── 2b. DD path context (keywords, clusters, version history) ─
        if source_paths:
            try:
                _enrich_dd_path_context(gc, item, source_paths[0])
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: dd_path_context failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # ── 3. Vector-similar SN neighbours ───────────────────────────
        description = item.get("description") or sn_id.replace("_", " ")
        try:
            peers_raw = _search_nearby_names(description, k=6)
            peers = [
                {
                    "tag": f"name:{p['id']}",
                    "unit": p.get("unit") or "",
                    "physics_domain": p.get("physics_domain") or "",
                    "doc_short": (p.get("description") or "")[:120],
                    "cocos_label": p.get("cocos_label") or "",
                }
                for p in peers_raw
                if p.get("id") and p.get("id") != sn_id
            ][:5]
            if peers:
                item["nearest_peers"] = peers
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: nearest_peers failed for %s",
                sn_id,
                exc_info=True,
            )

        # ── 4. Parent/child component context ─────────────────────────
        try:
            # Check if this SN is a component of a parent
            parent_rows = list(
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $sn_id})-[r:HAS_PARENT]->(parent:StandardName)
                    RETURN parent.id AS name,
                           parent.description AS description,
                           parent.documentation AS documentation,
                           r.axis AS axis
                    LIMIT 1
                    """,
                    sn_id=sn_id,
                )
            )
            if parent_rows:
                p = parent_rows[0]
                item["parent_sn"] = {
                    "name": p["name"],
                    "description": p.get("description") or "",
                    "documentation": p.get("documentation") or "",
                }
                item["component_axis"] = p.get("axis") or ""

            # Check if this SN has children (is a parent)
            child_rows = list(
                gc.query(
                    """
                    MATCH (child:StandardName)-[r:HAS_PARENT]->(sn:StandardName {id: $sn_id})
                    RETURN child.id AS name,
                           child.description AS description,
                           r.axis AS axis
                    ORDER BY child.id
                    """,
                    sn_id=sn_id,
                )
            )
            if child_rows:
                from imas_codex.standard_names.families import sort_by_axis_convention

                child_dicts = [
                    {
                        "name": c["name"],
                        "description": c.get("description") or "",
                        "axis": c.get("axis") or "",
                    }
                    for c in child_rows
                ]
                item["child_components"] = sort_by_axis_convention(child_dicts)
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: component context failed for %s",
                sn_id,
                exc_info=True,
            )

        # ── 5. Derivative / base-quantity context ─────────────────────────
        try:
            from imas_codex.standard_names.families import DD_DERIVATIVE_MAP

            parent_sn_data = item.get("parent_sn")
            if parent_sn_data:
                parent_name = parent_sn_data["name"]
                # Query parent unit and sibling derivatives sharing same parent
                bq_rows = list(
                    gc.query(
                        """
                        MATCH (parent:StandardName {id: $parent_name})
                        OPTIONAL MATCH (sib:StandardName)-[:HAS_PARENT]->(parent)
                        WHERE sib.id <> $sn_id
                        WITH parent, collect(sib.id) AS siblings
                        RETURN parent.unit AS unit, siblings
                        """,
                        parent_name=parent_name,
                        sn_id=sn_id,
                    )
                )
                if bq_rows:
                    bq_row = bq_rows[0]
                    item["base_quantity"] = {
                        "name": parent_name,
                        "description": parent_sn_data.get("description") or "",
                        "documentation": parent_sn_data.get("documentation") or "",
                        "unit": bq_row.get("unit") or "",
                    }
                    if sn_id in DD_DERIVATIVE_MAP:
                        numerator, denominator = DD_DERIVATIVE_MAP[sn_id]
                        raw_siblings = bq_row.get("siblings") or []
                        siblings = [s for s in raw_siblings if s]
                        item["derivative_context"] = {
                            "numerator": numerator,
                            "denominator": denominator,
                            "siblings": siblings,
                        }
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: derivative context failed for %s",
                sn_id,
                exc_info=True,
            )


def _nearby_names_for_docs_gen(gc: Any, items: list[dict]) -> list[dict]:
    """Return accepted StandardNames in the same physics domain(s) as *items*.

    Queries up to 20 accepted SNs per unique domain found across the batch.
    Results are deduplicated and capped at 40 total.  Used to populate the
    batch-level ``nearby_existing_names`` context key in
    :func:`process_generate_docs_batch`.
    """
    domains: set[str] = set()
    for item in items:
        dom = item.get("physics_domain")
        if isinstance(dom, list):
            domains.update(dom)
        elif dom:
            domains.add(dom)

    nearby: list[dict] = []
    seen_ids: set[str] = set()
    batch_ids: set[str] = {item["id"] for item in items if item.get("id")}

    for domain in sorted(domains):
        # Use the first batch item id as the exclusion anchor; the query
        # already uses != so any id from the batch suffices.
        anchor = next(iter(batch_ids), "")
        try:
            rows = list(gc.query(_DOCS_GEN_NEARBY_QUERY, domain=domain, sn_id=anchor))
        except Exception:
            logger.debug(
                "_nearby_names_for_docs_gen: query failed for domain=%s",
                domain,
                exc_info=True,
            )
            continue
        for r in rows:
            nid = r.get("id")
            if nid and nid not in seen_ids and nid not in batch_ids:
                seen_ids.add(nid)
                nearby.append(dict(r))
                if len(nearby) >= 40:
                    break
        if len(nearby) >= 40:
            break

    return nearby


async def process_generate_docs_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of accepted StandardNames for generate_docs (P4.1).

    For each item in the batch:

    1. Enrich items with DD source context via :func:`_enrich_for_docs_gen`
       and :func:`_nearby_names_for_docs_gen` (source_paths, dd_source_docs,
       dd_aliases, nearest_peers, related_neighbours, nearby_existing_names).
    2. Render prompt via ``render_prompt("sn/generate_docs_user", {...})`` with
       reviewer feedback (reviewer_score_name, reviewer_comments_name),
       chain history, and the new DD context as context.
    3. Use ``get_model("sn-docs")`` — docs generation model.
    4. Call ``acall_llm_structured`` with ``service="standard-names"`` and
       response_model=``GeneratedDocs``.
    5. Reserve budget and charge ``LLMCostEvent``.
    6. Persist via ``persist_generated_docs`` (transitions docs_stage → 'drafted').
    7. On error: ``release_generate_docs_failed_claims``.
    8. Stream per-item progress: name + first 100 chars of generated description.

    Returns count of items successfully processed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.graph_ops import (
        persist_generated_docs,
        release_generate_docs_failed_claims,
    )
    from imas_codex.standard_names.models import GeneratedDocs

    model = get_model("sn-docs")
    processed = 0

    # ── Enrich batch with DD context (source paths, docs, peers) ──────────
    nearby_existing_names: list[dict] = []
    compose_scored_examples: list[dict] = []
    try:

        def _do_enrich() -> list[dict]:
            with GraphClient() as gc:
                # Load COCOS singleton params for sign guidance rendering
                cocos_params = None
                try:
                    from imas_codex.settings import get_dd_version

                    dd_version = get_dd_version()
                    cocos_rows = list(
                        gc.query(
                            """
                            MATCH (dv:DDVersion {id: $ver})-[:COCOS]->(c:COCOS)
                            RETURN properties(c) AS cocos_params
                            """,
                            ver=dd_version,
                        )
                    )
                    if cocos_rows:
                        cocos_params = cocos_rows[0]["cocos_params"]
                except Exception:
                    logger.debug(
                        "process_generate_docs_batch: COCOS params load failed",
                        exc_info=True,
                    )
                _enrich_for_docs_gen(gc, batch, cocos_params=cocos_params)
                return _nearby_names_for_docs_gen(gc, batch)

        nearby_existing_names = await _asyncio.to_thread(_do_enrich)
    except Exception:
        logger.debug("process_generate_docs_batch: enrichment failed", exc_info=True)

    # ── Load scored examples for docs generation ──────────────────────
    try:
        from imas_codex.standard_names.example_loader import load_compose_examples

        batch_domains = list(
            {
                item.get("physics_domain", "")
                for item in batch
                if item.get("physics_domain")
            }
        )

        def _load_docs_scored_examples() -> list[dict]:
            with GraphClient() as gc:
                return load_compose_examples(
                    gc, physics_domains=batch_domains, axis="docs"
                )

        compose_scored_examples = await _asyncio.to_thread(_load_docs_scored_examples)
        if compose_scored_examples:
            logger.info(
                "generate_docs: Loaded %d scored examples (domains=%s)",
                len(compose_scored_examples),
                batch_domains or "all",
            )
    except Exception:
        logger.debug(
            "process_generate_docs_batch: scored examples load failed",
            exc_info=True,
        )

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        chain_history = item.get("chain_history") or []

        # ── Build prompt context ───────────────────────────────────────
        prompt_context: dict[str, Any] = {
            "item": item,
            "chain_history": chain_history,
            "nearby_existing_names": nearby_existing_names,
            "compose_scored_examples": compose_scored_examples,
        }

        try:
            user_prompt = render_prompt("sn/generate_docs_user", prompt_context)
        except Exception:
            logger.debug("generate_docs: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Generate description and documentation for the IMAS standard name: "
                f"{sn_id}\nKind: {item.get('kind', 'scalar')}\n"
                f"Unit: {item.get('unit', '')}"
            )

        try:
            system_prompt = render_prompt("sn/generate_docs_system", prompt_context)
        except Exception:
            logger.debug("generate_docs: system prompt render failed for %s", sn_id)
            system_prompt = None

        # ── Budget reservation ─────────────────────────────────────────
        estimated = 0.20
        lease = mgr.reserve(estimated, phase="generate_docs")
        if lease is None:
            lease = mgr.reserve(0.0, phase="generate_docs")

        # ── LLM call ──────────────────────────────────────────────────
        try:
            _messages = (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if system_prompt
                else [{"role": "user", "content": user_prompt}]
            )
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=GeneratedDocs,
                service="standard-names",
            )

            result_obj, cost, _tokens = llm_out

            # Charge cost to lease
            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=(getattr(llm_out, "cache_read_tokens", 0) or 0),
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=(sn_id,),
                    phase="generate_docs",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # ── Persist ───────────────────────────────────────────────
            await _asyncio.to_thread(
                persist_generated_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                description=normalize_prose_spelling(result_obj.description),
                documentation=normalize_prose_spelling(result_obj.documentation),
                model=model,
                run_id=mgr.run_id,
            )
            processed += 1

            # ── Per-item progress ──────────────────────────────────────
            # Keep log preview short; pass full description to on_event so
            # terminal-aware clipping can use the available display width.
            _desc_log = result_obj.description[:100]
            logger.info(
                "\033[32mgenerate_docs\033[0m: %s — %s",
                sn_id,
                _desc_log,
            )

            if on_event is not None:
                on_event(
                    {
                        "pool": "generate_docs",
                        "name": sn_id,
                        "description": result_obj.description,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception:
            logger.exception("generate_docs failed for %s", sn_id)
            try:
                await _asyncio.to_thread(
                    release_generate_docs_failed_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                logger.debug(
                    "release_generate_docs_failed_claims also failed for %s",
                    sn_id,
                )
        finally:
            # Always return the unused portion of the lease to the pool.
            # Without this the unspent remainder leaks every iteration and
            # the pool exhausts at ~25 % of cost_limit.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


async def process_review_docs_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of drafted StandardNames for docs review (RD-quorum).

    Mirrors :func:`process_review_name_batch` for the docs axis: runs the
    full ``[sn.review.docs].models`` chain (cycle 0 + cycle 1 unconditionally,
    cycle 2 only when per-dim disagreement exceeds threshold). Persists each
    cycle as a ``StandardNameReview`` node and transitions ``docs_stage`` via
    :func:`~imas_codex.standard_names.graph_ops.persist_reviewed_docs`.

    Returns count of items successfully reviewed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_model,
        get_sn_review_disagreement_threshold,
        get_sn_review_docs_models,
        get_sn_review_names_models,
    )
    from imas_codex.standard_names.defaults import (
        DEFAULT_MIN_SCORE,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        persist_reviewed_docs,
        release_review_docs_failed_claims,
        update_review_aggregates,
    )
    from imas_codex.standard_names.models import StandardNameQualityReviewDocsBatch

    # ── Resolve docs review chain ──────────────────────────────────────
    try:
        review_models = get_sn_review_docs_models()
    except (ValueError, IndexError):
        review_models = []
    if not review_models:
        try:
            review_models = get_sn_review_names_models()
        except (ValueError, IndexError):
            review_models = []
    if not review_models:
        # Refine tier (Sonnet 4.6) — defensive fallback only.
        review_models = [get_model("sn-refine")]

    try:
        disagreement_threshold = get_sn_review_disagreement_threshold()
    except Exception:
        disagreement_threshold = 0.20

    processed = 0

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""

        # ── Build prompt context ───────────────────────────────────────
        from imas_codex.standard_names.context import fetch_review_neighbours

        try:
            neighbours = fetch_review_neighbours(item)
        except Exception:
            logger.debug(
                "review_docs: neighbour fetch failed for %s", sn_id, exc_info=True
            )
            neighbours = {
                "vector_neighbours": [],
                "same_base_neighbours": [],
                "same_path_neighbours": [],
            }

        # ── Load scored examples for reviewer calibration ──────────────
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.example_loader import load_review_examples

        try:
            _item_domain = item.get("physics_domain") or ""

            def _load_review_examples(domain: str = _item_domain) -> list:
                with GraphClient() as _gc:
                    return load_review_examples(
                        _gc, physics_domains=[domain], axis="docs"
                    )

            review_scored_examples = await _asyncio.to_thread(_load_review_examples)
        except Exception:
            logger.debug(
                "review_docs: scored-example load failed for %s", sn_id, exc_info=True
            )
            review_scored_examples = []

        # ── DD path context for docs reviewer ──────────────────────────
        _source_paths = item.get("source_paths") or []
        if _source_paths:
            _first_src = (
                _source_paths[0] if isinstance(_source_paths, list) else _source_paths
            )
            if isinstance(_first_src, str):
                _first_src = strip_dd_prefix(_first_src)
            if _first_src:
                try:
                    from imas_codex.graph.client import GraphClient as _GCDD

                    def _do_dd_ctx(_p=_first_src, _it=item) -> None:
                        with _GCDD() as _gc:
                            _enrich_dd_path_context(_gc, _it, _p)

                    await _asyncio.to_thread(_do_dd_ctx)
                except Exception:
                    logger.debug(
                        "review_docs: dd_path_context failed for %s",
                        sn_id,
                        exc_info=True,
                    )

        prompt_context: dict[str, Any] = {
            "item": item,
            **neighbours,
            "review_scored_examples": review_scored_examples,
        }
        try:
            user_prompt = render_prompt("sn/review_docs_user", prompt_context)
            system_prompt = render_prompt("sn/review_docs_system", prompt_context)
        except Exception:
            logger.debug("review_docs: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Review the documentation for standard name: {item.get('id', sn_id)}\n"
                f"Description: {item.get('description', '')}\n"
                f"Documentation: {item.get('documentation', '')}"
            )
            system_prompt = (
                "You are a quality reviewer for IMAS standard name "
                "documentation in fusion plasma physics."
            )

        # ── Budget reservation (cover all cycles) ──────────────────────
        per_item_estimate = 0.05
        worst_case = per_item_estimate * len(review_models) * 1.3
        lease = mgr.reserve(worst_case, phase="review_docs")
        if lease is None:
            lease = mgr.reserve(0.0, phase="review_docs")

        try:
            quorum = await _run_rd_quorum_cycles(
                sn_id=sn_id,
                review_axis="docs",
                response_model=StandardNameQualityReviewDocsBatch,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                models=review_models,
                disagreement_threshold=disagreement_threshold,
                rubric_dims=(
                    "description_quality",
                    "documentation_quality",
                    "completeness",
                    "physics_accuracy",
                ),
                lease=lease,
                phase="review_docs",
                acall_llm_structured=acall_llm_structured,
            )

            if quorum is None:
                if claim_token:
                    try:
                        await _asyncio.to_thread(
                            release_review_docs_failed_claims,
                            sn_ids=[sn_id],
                            claim_token=claim_token,
                            from_stage="drafted",
                            to_stage="drafted",
                        )
                    except Exception:
                        logger.debug(
                            "release_review_docs_failed_claims also failed for %s",
                            sn_id,
                        )
                continue

            from imas_codex.standard_names.graph_ops import write_reviews

            await _asyncio.to_thread(write_reviews, quorum["records"])

            new_stage = await _asyncio.to_thread(
                persist_reviewed_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                score=quorum["winning_score"],
                scores=quorum["winning_scores"],
                comments=quorum["winning_comments"],
                comments_per_dim=quorum["winning_comments_per_dim"],
                model=quorum["canonical_model"],
                min_score=DEFAULT_MIN_SCORE,
                rotation_cap=DEFAULT_REFINE_ROTATIONS,
                llm_cost=quorum["total_cost"],
                llm_tokens_in=quorum["total_tokens_in"],
                llm_tokens_out=quorum["total_tokens_out"],
                llm_tokens_cached_read=0,
                llm_tokens_cached_write=0,
                llm_service="standard-names",
                run_id=mgr.run_id,
                skip_review_node=True,
            )

            try:
                await _asyncio.to_thread(update_review_aggregates, [sn_id])
            except Exception:
                logger.debug(
                    "update_review_aggregates failed for %s", sn_id, exc_info=True
                )

            if new_stage:
                processed += 1
                _comment_log = (quorum["winning_comments"] or "")[:80]
                logger.info(
                    "review_docs: %s → %s (score=%.3f, cycles=%d, method=%s) %s",
                    sn_id,
                    new_stage,
                    quorum["winning_score"],
                    len(quorum["records"]),
                    quorum["resolution_method"],
                    _comment_log,
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "review_docs",
                            "name": sn_id,
                            "score": quorum["winning_score"],
                            "comment": quorum["winning_comments"] or "",
                            "stage": new_stage,
                            "model": quorum["canonical_model"],
                            "cost": quorum["total_cost"],
                            "cycles": len(quorum["records"]),
                            "resolution_method": quorum["resolution_method"],
                        }
                    )
            else:
                logger.debug("review_docs: %s persist no-op (token mismatch?)", sn_id)

        except Exception:
            logger.exception("review_docs failed for %s", sn_id)
            token = item.get("claim_token") or ""
            if token:
                try:
                    await _asyncio.to_thread(
                        release_review_docs_failed_claims,
                        sn_ids=[sn_id],
                        claim_token=token,
                        from_stage="drafted",
                        to_stage="drafted",
                    )
                except Exception:
                    logger.debug(
                        "release_review_docs_failed_claims also failed for %s",
                        sn_id,
                    )
        finally:
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


# =============================================================================
# process_refine_docs_batch — DocsRevision snapshot + in-place docs update
# =============================================================================


async def process_refine_docs_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of reviewed StandardNames for docs refinement (P4.3).

    For each item in the batch:

    1. Read ``docs_chain_length``, decide model (escalation on final attempt).
    2. Render prompt with current docs, reviewer feedback, and chain history.
    3. Call LLM to produce refined docs (``RefinedDocs`` response model).
    4. Persist via ``persist_refined_docs`` — snapshots old docs into
       ``DocsRevision``, updates SN in-place, advances chain.
    5. On failure: release claims via ``release_refine_docs_failed_claims``.

    Returns count of items successfully processed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.defaults import (
        DEFAULT_ESCALATION_MODEL,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        persist_refined_docs,
        release_refine_docs_failed_claims,
    )
    from imas_codex.standard_names.models import RefinedDocs

    rotation_cap = DEFAULT_REFINE_ROTATIONS
    processed = 0

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        docs_chain_length = item.get("docs_chain_length", 0) or 0
        docs_chain_history = item.get("docs_chain_history", [])

        # ── Escalation decision ───────────────────────────────────
        escalate = docs_chain_length >= rotation_cap - 1
        if escalate:
            model = DEFAULT_ESCALATION_MODEL
        else:
            # Refine tier (Sonnet 4.6 by default) — see refine_name
            # comment + 2026-05-03 E3 acceptance audit.
            model = get_model("sn-refine")

        # ── Build prompt context ──────────────────────────────────
        prompt_context: dict[str, Any] = {
            "sn_name": sn_id,
            "description": item.get("description", ""),
            "documentation": item.get("documentation", ""),
            "kind": item.get("kind", "scalar"),
            "unit": item.get("unit", ""),
            "physics_domain": item.get("physics_domain", ""),
            "docs_chain_length": docs_chain_length,
            "docs_chain_history": docs_chain_history,
            "reviewer_score_docs": item.get("reviewer_score_docs"),
            "reviewer_comments_per_dim_docs": item.get(
                "reviewer_comments_per_dim_docs"
            ),
            "dd_paths": [],
        }

        # Best-effort DD path enrichment
        try:
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                dd_rows = gc.query(
                    """
                    MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName {id: $sn_id})
                    WHERE n:IMASNode
                    RETURN n.id AS path, n.ids AS ids,
                           n.unit AS unit,
                           n.documentation AS documentation,
                           n.description AS description
                    LIMIT 5
                    """,
                    sn_id=sn_id,
                )
                prompt_context["dd_paths"] = [dict(r) for r in dd_rows]
        except Exception:
            logger.debug("refine_docs: DD path enrichment failed for %s", sn_id)

        try:
            user_prompt = render_prompt("sn/refine_docs_user", prompt_context)
        except Exception:
            logger.exception("refine_docs: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Refine the documentation for standard name: {sn_id}\n"
                f"Current description: {item.get('description', '')}\n"
                f"Current documentation: {item.get('documentation', '')}\n"
                f"Reviewer feedback: {item.get('reviewer_comments_docs', '')}"
            )

        try:
            system_prompt = render_prompt("sn/refine_docs_system", prompt_context)
        except Exception:
            logger.debug("refine_docs: system prompt render failed for %s", sn_id)
            system_prompt = None

        # ── Budget reservation ─────────────────────────────────────
        estimated = 0.20
        lease = mgr.reserve(estimated, phase="refine_docs")
        if lease is None:
            lease = mgr.reserve(0.0, phase="refine_docs")

        # ── LLM call ──────────────────────────────────────────────
        try:
            _messages = (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if system_prompt
                else [{"role": "user", "content": user_prompt}]
            )
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=RefinedDocs,
                service="standard-names",
            )

            result_obj, cost, _tokens = llm_out

            # Charge cost to lease
            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=(getattr(llm_out, "cache_read_tokens", 0) or 0),
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=(sn_id,),
                    phase="refine_docs",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # ── Persist ───────────────────────────────────────────
            await _asyncio.to_thread(
                persist_refined_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                description=normalize_prose_spelling(result_obj.description),
                documentation=normalize_prose_spelling(result_obj.documentation),
                model=model,
                current_description=item.get("description") or "",
                current_documentation=item.get("documentation") or "",
                current_model=item.get("docs_model"),
                current_generated_at=(
                    str(item["docs_generated_at"])
                    if item.get("docs_generated_at")
                    else None
                ),
                reviewer_score_to_snapshot=item.get("reviewer_score_docs"),
                reviewer_comments_to_snapshot=item.get("reviewer_comments_docs"),
                reviewer_comments_per_dim_to_snapshot=item.get(
                    "reviewer_comments_per_dim_docs"
                ),
                run_id=mgr.run_id,
            )
            processed += 1

            desc_preview = result_obj.description[:80]
            logger.info(
                "\033[35mrefine_docs\033[0m: %s — %s (chain=%d, model=%s)",
                sn_id,
                desc_preview,
                docs_chain_length + 1,
                model,
            )

            if on_event is not None:
                on_event(
                    {
                        "pool": "refine_docs",
                        "name": sn_id,
                        "description": result_obj.description,
                        "revision": docs_chain_length + 1,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception:
            logger.exception("refine_docs failed for %s", sn_id)
            try:
                await _asyncio.to_thread(
                    release_refine_docs_failed_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                logger.debug(
                    "release_refine_docs_failed_claims also failed for %s",
                    sn_id,
                )
        finally:
            # Always return the unused portion of the lease to the pool.
            # Without this the unspent remainder leaks every iteration and
            # the pool exhausts at ~25 % of cost_limit.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


# ═══════════════════════════════════════════════════════════════════════
# Embed worker — batch-embed StandardName nodes
# ═══════════════════════════════════════════════════════════════════════


async def process_embed_batch(
    items: list[dict[str, Any]],
    mgr: Any,
    stop_event: asyncio.Event,
    *,
    on_event: Callable | None = None,
) -> int:
    """Batch-embed StandardName nodes.

    No LLM cost — uses embedding server only.  Does not charge budget.
    """
    from imas_codex.embeddings.description import embed_descriptions_batch
    from imas_codex.standard_names.graph_ops import (
        _compute_embed_hash,
        persist_embed_batch,
    )

    if stop_event.is_set():
        return 0

    # Build embed texts
    for item in items:
        desc = item.get("description")
        sn_id = item["id"]
        item["_embed_text"] = f"{sn_id} — {desc}" if desc else sn_id
        item["embed_text_hash"] = _compute_embed_hash(sn_id, desc)

    # Batch embed (no LLM cost — uses embedding server)
    await asyncio.to_thread(embed_descriptions_batch, items, text_field="_embed_text")

    # Filter to those that got embeddings
    to_persist = [
        {
            "id": it["id"],
            "embedding": it["embedding"],
            "embed_text_hash": it["embed_text_hash"],
        }
        for it in items
        if it.get("embedding") is not None
    ]

    if to_persist:
        written = await asyncio.to_thread(persist_embed_batch, to_persist)
    else:
        written = 0

    # Emit events for display
    if on_event:
        for it in to_persist:
            on_event(
                {
                    "pool": "embed_name",
                    "name": it["id"],
                }
            )

    return written

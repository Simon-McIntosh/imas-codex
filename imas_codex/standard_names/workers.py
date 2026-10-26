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
import inspect as _inspect
import json
import re as _re  # noqa: F401 - resolved by worker_common through _worker_dependency
from collections.abc import Callable
from functools import cache as _cache, lru_cache  # noqa: F401
from typing import TYPE_CHECKING, Any

from imas_codex.standard_names.budget import (
    bind_attempt_exposure,
    charge_billable_exception,
    model_provider_exposure,
)
from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    SEMANTIC_SIM_GATE_MODEL,  # noqa: F401
    SEMANTIC_SIM_GATE_RESOLUTION_METHOD,  # noqa: F401
)

# Kept as resolver targets: the task modules reach these through
# ``_worker_dependency``, which reads this module's globals.
from imas_codex.standard_names.provenance import detect_value_provenance  # noqa: F401
from imas_codex.standard_names.source_paths import (
    DD_PREFIX,
    encode_source_path,  # noqa: F401
    strip_dd_prefix,
)

from .worker_tasks import (
    worker_common as _moved_worker_common,
    worker_compose_and_validate as _moved_worker_compose_and_validate,
    worker_compose_batches as _moved_worker_compose_batches,
    worker_enrich_parents as _moved_worker_enrich_parents,
    worker_extraction_and_enrichment as _moved_worker_extraction_and_enrichment,
    worker_generate_docs as _moved_worker_generate_docs,
    worker_refine_names as _moved_worker_refine_names,
    worker_review_names as _moved_worker_review_names,
)


def _worker_dependency(name: str):
    """Resolve a worker collaborator through the entrypoint."""
    return globals()[name]


if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager

logger = _moved_worker_common.logger


def backfill_semantic_similarity_gate_resolution_method(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.backfill_semantic_similarity_gate_resolution_method."""
    return _moved_worker_common.backfill_semantic_similarity_gate_resolution_method(
        _worker_dependency, *args, **kwargs
    )


def normalize_spelling(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.normalize_spelling."""
    return _moved_worker_common.normalize_spelling(_worker_dependency, *args, **kwargs)


def normalize_prose_spelling(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.normalize_prose_spelling."""
    return _moved_worker_common.normalize_prose_spelling(
        _worker_dependency, *args, **kwargs
    )


# Spelled-out Greek letters become their Unicode symbols in description prose
# (the ISN description-field convention: φ, θ, ρ; frames as (R, φ, Z)).
# Lowercase only — a capitalized "Phi" may be the flux Φ or a sentence-start
# angle, so it is left for review rather than guessed.
_GREEK_WORD_TO_SYMBOL = _moved_worker_common._GREEK_WORD_TO_SYMBOL
_GREEK_WORD_RE = _moved_worker_common._GREEK_WORD_RE

# Backslash-escaped Greek commands: \phi \theta \rho \pi (and capitals) →
# Unicode symbol. Includes \pi because it is the common LaTeX form that leaks
# into descriptions. The trailing (?![a-zA-Z]) stops \phinx-style over-matching.
_GREEK_CMD_TO_SYMBOL = _moved_worker_common._GREEK_CMD_TO_SYMBOL
_GREEK_CMD_RE = _moved_worker_common._GREEK_CMD_RE
# A backslash stranded in front of an already-Unicode Greek symbol — the
# half-converted corruption (\φ) produced by the older word-only normalizer.
_GREEK_STRANDED_BACKSLASH_RE = _moved_worker_common._GREEK_STRANDED_BACKSLASH_RE
# A LaTeX command carrying a single braced argument: \mathbf{k} → k.
_LATEX_CMD_ARG_RE = _moved_worker_common._LATEX_CMD_ARG_RE
# A bare LaTeX command with no argument: \left \right \cdot \, → removed.
_LATEX_CMD_BARE_RE = _moved_worker_common._LATEX_CMD_BARE_RE
# Braced sub/superscripts: ^{1+} → 1+ ; _{i} → _i (keep the underscore).
_SUPERSCRIPT_BRACE_RE = _moved_worker_common._SUPERSCRIPT_BRACE_RE
_SUBSCRIPT_BRACE_RE = _moved_worker_common._SUBSCRIPT_BRACE_RE
# Math delimiters: $$…$$ and $…$ — drop the delimiter, keep inner content.
_MATH_DELIM_RE = _moved_worker_common._MATH_DELIM_RE


def normalize_description_notation(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.normalize_description_notation."""
    return _moved_worker_common.normalize_description_notation(
        _worker_dependency, *args, **kwargs
    )


def normalize_description_text(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.normalize_description_text."""
    return _moved_worker_common.normalize_description_text(
        _worker_dependency, *args, **kwargs
    )


# Physics-domain UK→US supplements not in breame's dictionary.
_DOMAIN_UK_US = _moved_worker_common._DOMAIN_UK_US

_GRAMMAR_FIELDS = _moved_worker_common._GRAMMAR_FIELDS


def _dedup_adjacent_tokens(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._dedup_adjacent_tokens."""
    return _moved_worker_common._dedup_adjacent_tokens(
        _worker_dependency, *args, **kwargs
    )


# ---------------------------------------------------------------------------
# Shape-parameter surface injection
# ---------------------------------------------------------------------------
#
# Dimensionless plasma shape descriptors are only meaningful *of* a surface.
# A bare ``triangularity`` conflates the boundary scalar
# (``equilibrium/.../boundary/triangularity``) with the flux-surface radial
# profile (``.../profiles_1d/triangularity``) and is not itself a measurable
# quantity. When the composer omits the surface locus we inject it
# deterministically from the source DD path so the leaf is always
# surface-explicit (``triangularity_of_plasma_boundary`` /
# ``triangularity_of_flux_surface``) — never bare. This also de-conflates the
# boundary and profile siblings into distinct names instead of one ambiguous
# parent. Derived structural parents are reached later by stripping
# qualifiers/projections, never by stripping a base's defining surface — so no
# bare shape-parameter parent is ever admitted (see ``parents.py``).
#
# WHICH bases require the surface is codex POLICY; the tokens themselves are
# ISN's. The policy is validated against the live ISN ``physical_base``
# vocabulary at load, so a token ISN renames or retires drops out of the set and
# is reported as drift rather than silently persisting as a spelling that
# matches nothing. ISN's own naming guidance states the requirement ("Use
# elongation_of_plasma_boundary or elongation_of_flux_surface, not elongation")
# but exposes no machine-readable flag for it, so the selection lives here.
_SHAPE_PARAMETER_BASE_POLICY = _moved_worker_common._SHAPE_PARAMETER_BASE_POLICY


def _isn_segment_tokens(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._isn_segment_tokens."""
    return _moved_worker_common._isn_segment_tokens(_worker_dependency, *args, **kwargs)


def _isn_physical_bases(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._isn_physical_bases."""
    return _moved_worker_common._isn_physical_bases(_worker_dependency, *args, **kwargs)


def shape_parameter_base_drift(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.shape_parameter_base_drift."""
    return _moved_worker_common.shape_parameter_base_drift(
        _worker_dependency, *args, **kwargs
    )


def _shape_parameter_bases(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._shape_parameter_bases."""
    return _moved_worker_common._shape_parameter_bases(
        _worker_dependency, *args, **kwargs
    )


def _rate_natured_bases(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._rate_natured_bases."""
    return _moved_worker_common._rate_natured_bases(_worker_dependency, *args, **kwargs)


def _state_resolution_tokens(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._state_resolution_tokens."""
    return _moved_worker_common._state_resolution_tokens(
        _worker_dependency, *args, **kwargs
    )


def _name_denotes_state(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._name_denotes_state."""
    return _moved_worker_common._name_denotes_state(_worker_dependency, *args, **kwargs)


def _name_denotes_rate(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._name_denotes_rate."""
    return _moved_worker_common._name_denotes_rate(_worker_dependency, *args, **kwargs)


def _ir_denotes_time_change(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._ir_denotes_time_change."""
    return _moved_worker_common._ir_denotes_time_change(
        _worker_dependency, *args, **kwargs
    )


def _ir_claims_time_change_in_scope(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._ir_claims_time_change_in_scope."""
    return _moved_worker_common._ir_claims_time_change_in_scope(
        _worker_dependency, *args, **kwargs
    )


def _name_claims_time_change(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._name_claims_time_change."""
    return _moved_worker_common._name_claims_time_change(
        _worker_dependency, *args, **kwargs
    )


def _time_change_claim_is_binary_scoped(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._time_change_claim_is_binary_scoped."""
    return _moved_worker_common._time_change_claim_is_binary_scoped(
        _worker_dependency, *args, **kwargs
    )


def _shape_parameter_surface(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._shape_parameter_surface."""
    return _moved_worker_common._shape_parameter_surface(
        _worker_dependency, *args, **kwargs
    )


def _inject_shape_parameter_surface(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._inject_shape_parameter_surface."""
    return _moved_worker_common._inject_shape_parameter_surface(
        _worker_dependency, *args, **kwargs
    )


# ---------------------------------------------------------------------------
# Pre-validation gate: reject malformed LLM output before MERGE
# ---------------------------------------------------------------------------

_SNAKE_CASE_RE = _moved_worker_common._SNAKE_CASE_RE


def is_well_formed_candidate(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.is_well_formed_candidate."""
    return _moved_worker_common.is_well_formed_candidate(
        _worker_dependency, *args, **kwargs
    )


def _claimed_source_ids(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._claimed_source_ids."""
    return _moved_worker_common._claimed_source_ids(_worker_dependency, *args, **kwargs)


def _strip_response_source_id(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._strip_response_source_id."""
    return _moved_worker_common._strip_response_source_id(
        _worker_dependency, *args, **kwargs
    )


def _review_dd_source_bindings(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._review_dd_source_bindings."""
    return _moved_worker_common._review_dd_source_bindings(
        _worker_dependency, *args, **kwargs
    )


def _review_dd_source_ids(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._review_dd_source_ids."""
    return _moved_worker_common._review_dd_source_ids(
        _worker_dependency, *args, **kwargs
    )


def _sanitize_dd_gap_evidence(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._sanitize_dd_gap_evidence."""
    return _moved_worker_common._sanitize_dd_gap_evidence(
        _worker_dependency, *args, **kwargs
    )


def _persist_dd_gap_evidence(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._persist_dd_gap_evidence."""
    return _moved_worker_common._persist_dd_gap_evidence(
        _worker_dependency, *args, **kwargs
    )


def _claimed_source_outcome(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._claimed_source_outcome."""
    return _moved_worker_common._claimed_source_outcome(
        _worker_dependency, *args, **kwargs
    )


def _sanitize_compose_result_sources(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._sanitize_compose_result_sources."""
    return _moved_worker_common._sanitize_compose_result_sources(
        _worker_dependency, *args, **kwargs
    )


def _filter_persist_candidates_to_claimed_sources(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._filter_persist_candidates_to_claimed_sources."""
    return _moved_worker_common._filter_persist_candidates_to_claimed_sources(
        _worker_dependency, *args, **kwargs
    )


def _withhold_binding_conflicted_sources(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._withhold_binding_conflicted_sources."""
    return _moved_worker_common._withhold_binding_conflicted_sources(
        _worker_dependency, *args, **kwargs
    )


def _candidate_source_node_id(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._candidate_source_node_id."""
    return _moved_worker_common._candidate_source_node_id(
        _worker_dependency, *args, **kwargs
    )


def _persist_description_checked_candidates(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._persist_description_checked_candidates."""
    return _moved_worker_common._persist_description_checked_candidates(
        _worker_dependency, *args, **kwargs
    )


def _mark_source_prevalidation_failed(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._mark_source_prevalidation_failed."""
    return _moved_worker_common._mark_source_prevalidation_failed(
        _worker_dependency, *args, **kwargs
    )


def provenance_canonical_names(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.provenance_canonical_names."""
    return _moved_worker_common.provenance_canonical_names(
        _worker_dependency, *args, **kwargs
    )


# Bare coordinate/infrastructure tokens that are NOT physics observables.
# A standard name composed from one of these is doomed: it fails the
# semantic-similarity gate (a reader cannot tell what is measured), then burns
# review + every refine rotation before exhausting.  Catch them at compose time
# and route the source to ``skipped`` instead of composing a doomed bare name.
_NON_NAMEABLE_BARE_TOKENS = _moved_worker_common._NON_NAMEABLE_BARE_TOKENS


def is_non_nameable_coordinate(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common.is_non_nameable_coordinate."""
    return _moved_worker_common.is_non_nameable_coordinate(
        _worker_dependency, *args, **kwargs
    )


def _coerce_llm_metric(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._coerce_llm_metric."""
    return _moved_worker_common._coerce_llm_metric(_worker_dependency, *args, **kwargs)


def _extract_llm_telemetry(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._extract_llm_telemetry."""
    return _moved_worker_common._extract_llm_telemetry(
        _worker_dependency, *args, **kwargs
    )


# ---------------------------------------------------------------------------
# Auto-VocabGap detection for physical_base
# ---------------------------------------------------------------------------


def _load_known_physical_bases(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._load_known_physical_bases."""
    return _moved_worker_common._load_known_physical_bases(
        _worker_dependency, *args, **kwargs
    )


def _auto_detect_physical_base_gaps(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._auto_detect_physical_base_gaps."""
    return _moved_worker_common._auto_detect_physical_base_gaps(
        _worker_dependency, *args, **kwargs
    )


def _compute_token_reuse_hits(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._compute_token_reuse_hits."""
    return _moved_worker_common._compute_token_reuse_hits(
        _worker_dependency, *args, **kwargs
    )


def _grammar_round_trip_failures(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._grammar_round_trip_failures."""
    return _moved_worker_common._grammar_round_trip_failures(
        _worker_dependency, *args, **kwargs
    )


def _build_grammar_retry_reason(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._build_grammar_retry_reason."""
    return _moved_worker_common._build_grammar_retry_reason(
        _worker_dependency, *args, **kwargs
    )


def _build_token_reuse_retry_reason(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._build_token_reuse_retry_reason."""
    return _moved_worker_common._build_token_reuse_retry_reason(
        _worker_dependency, *args, **kwargs
    )


def _active_editorial_gap_guidance(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._active_editorial_gap_guidance."""
    return _moved_worker_common._active_editorial_gap_guidance(
        _worker_dependency, *args, **kwargs
    )


def _stamp_dedup_decision(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_common._stamp_dedup_decision."""
    return _moved_worker_common._stamp_dedup_decision(
        _worker_dependency, *args, **kwargs
    )


# =============================================================================
# EXTRACT phase
# =============================================================================


async def extract_worker(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment.extract_worker."""
    return await _moved_worker_extraction_and_enrichment.extract_worker(
        _worker_dependency, *args, **kwargs
    )


# =============================================================================
# COMPOSE phase
# =============================================================================

_NEARBY_COMPARATOR_HARD_CEILING = (
    _moved_worker_extraction_and_enrichment._NEARBY_COMPARATOR_HARD_CEILING
)


def _search_nearby_names(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._search_nearby_names."""
    return _moved_worker_extraction_and_enrichment._search_nearby_names(
        _worker_dependency, *args, **kwargs
    )


def _collect_nearby_name_comparators(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._collect_nearby_name_comparators."""
    return _moved_worker_extraction_and_enrichment._collect_nearby_name_comparators(
        _worker_dependency, *args, **kwargs
    )


#: Lease floor for a rerank. The rerank's price is only known once it
#: returns, so the lease is opened for this before the call and the actual
#: spend is charged through the lease already held.
_RERANK_LEASE_FLOOR_USD = (
    _moved_worker_extraction_and_enrichment._RERANK_LEASE_FLOOR_USD
)


async def _rerank_comparators_for_compose(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._rerank_comparators_for_compose."""
    return (
        await _moved_worker_extraction_and_enrichment._rerank_comparators_for_compose(
            _worker_dependency, *args, **kwargs
        )
    )


def _book_rerank_spend(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._book_rerank_spend."""
    return _moved_worker_extraction_and_enrichment._book_rerank_spend(
        _worker_dependency, *args, **kwargs
    )


async def _reranked_nearby_comparators(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._reranked_nearby_comparators."""
    return await _moved_worker_extraction_and_enrichment._reranked_nearby_comparators(
        _worker_dependency, *args, **kwargs
    )


def _compose_budget_phase(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._compose_budget_phase."""
    return _moved_worker_extraction_and_enrichment._compose_budget_phase(
        _worker_dependency, *args, **kwargs
    )


async def _rerank_comparators_for_worker(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._rerank_comparators_for_worker."""
    return await _moved_worker_extraction_and_enrichment._rerank_comparators_for_worker(
        _worker_dependency, *args, **kwargs
    )


# =============================================================================
# DD context enrichment — fetch rich graph data before composing
# =============================================================================

_DD_CONTEXT_QUERY = _moved_worker_extraction_and_enrichment._DD_CONTEXT_QUERY

# Ancestor lineage context — walk HAS_PARENT up to the IDS root and collect
# every ancestor that carries a non-empty description/documentation. A DD
# *value* leaf is often a terse template stub (e.g. a per-species
# ``.../velocity_phi/<species>/value`` reads only "Deuterium (D)."), while the
# physically-meaningful text — the quantity AND its evaluation locus (e.g.
# "Ion toroidal rotation velocity … at the pedestal top") — lives on a parent
# quantity node. Surfacing the ancestor lineage lets both name and docs
# composition ground on the real physics + locus instead of the stub. Rich
# LLM-enriched descriptions are preferred over terse DD documentation; nearest
# ancestor first.
_ANCESTOR_CONTEXT_QUERY = (
    _moved_worker_extraction_and_enrichment._ANCESTOR_CONTEXT_QUERY
)

_CROSS_IDS_QUERY = _moved_worker_extraction_and_enrichment._CROSS_IDS_QUERY

_VERSION_HISTORY_QUERY = _moved_worker_extraction_and_enrichment._VERSION_HISTORY_QUERY

_ERROR_FIELDS_QUERY = _moved_worker_extraction_and_enrichment._ERROR_FIELDS_QUERY

# Per-member DD documentation grounding: a compose candidate's name can span
# N sibling leaves (family_siblings, cross-IDS equivalents). Fetch the terse
# DD-XML documentation for those member paths so the name is grounded on every
# leaf it covers, not just the primary. Terse doc only — DISTINCT from the rich
# enriched description the primary already renders.
_MEMBER_DOC_QUERY = _moved_worker_extraction_and_enrichment._MEMBER_DOC_QUERY

# Bound the per-item member-doc map so a large family/cluster cannot bloat the
# compose prompt; the primary path's full grounding is always rendered.
_MAX_MEMBER_DOCS = _moved_worker_extraction_and_enrichment._MAX_MEMBER_DOCS

# Truncate each terse member clause so the grounding stays compact.
_MEMBER_DOC_MAX_CHARS = _moved_worker_extraction_and_enrichment._MEMBER_DOC_MAX_CHARS

_IDS_CONTEXT_QUERY = _moved_worker_extraction_and_enrichment._IDS_CONTEXT_QUERY


def _enrich_ids_context(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._enrich_ids_context."""
    return _moved_worker_extraction_and_enrichment._enrich_ids_context(
        _worker_dependency, *args, **kwargs
    )


def _hybrid_search_neighbours_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._hybrid_search_neighbours_batch."""
    return _moved_worker_extraction_and_enrichment._hybrid_search_neighbours_batch(
        _worker_dependency, *args, **kwargs
    )


def _hybrid_search_neighbours(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._hybrid_search_neighbours."""
    return _moved_worker_extraction_and_enrichment._hybrid_search_neighbours(
        _worker_dependency, *args, **kwargs
    )


# Cap for graph-relationship neighbour injection (per path).
_RELATED_MAX_RESULTS = _moved_worker_extraction_and_enrichment._RELATED_MAX_RESULTS


# Compose retry: on grammar/validation failure, retry with expanded context.
# Values resolved from settings accessors; module-level constants kept for
# backwards compatibility with any direct importers.
def _retry_attempts(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._retry_attempts."""
    return _moved_worker_extraction_and_enrichment._retry_attempts(
        _worker_dependency, *args, **kwargs
    )


def _retry_k_expansion(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._retry_k_expansion."""
    return _moved_worker_extraction_and_enrichment._retry_k_expansion(
        _worker_dependency, *args, **kwargs
    )


def _mark_vocab_gap_sources(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._mark_vocab_gap_sources."""
    return _moved_worker_extraction_and_enrichment._mark_vocab_gap_sources(
        _worker_dependency, *args, **kwargs
    )


def _related_path_neighbours(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._related_path_neighbours."""
    return _moved_worker_extraction_and_enrichment._related_path_neighbours(
        _worker_dependency, *args, **kwargs
    )


def _enrich_batch_items(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._enrich_batch_items."""
    return _moved_worker_extraction_and_enrichment._enrich_batch_items(
        _worker_dependency, *args, **kwargs
    )


#: A DD path segment naming a time derivative. The DD writes the differential
#: explicitly: the differentiated quantity carries a leading ``d`` and the time
#: denominator is ``dt`` — ``dphase_dt``, ``ddensity_dt_total``, ``dn_e_dt``,
#: ``d_dvolume_drho_tor_dt`` — and a whole structure of such derivatives is
#: ``d_dt``. That leading ``d`` is what separates a derivative from the DD's
#: OTHER use of the same suffix, the deuterium-tritium reaction
#: (``emissivity_dt``, ``fusion_power_dt``), which differentiates nothing.
_TIME_DERIVATIVE_SEGMENT_RE = (
    _moved_worker_extraction_and_enrichment._TIME_DERIVATIVE_SEGMENT_RE
)

_AXIS_LEAVES = _moved_worker_extraction_and_enrichment._AXIS_LEAVES

# Device WORDS a hardware locus is expected to echo in its source path. Only
# when the name's locus contains one of these do we apply the zero-overlap
# rejection — a locus naming a place need not appear in the source path.
#
# This is codex-side POLICY over ordinary English device nouns, not a copy of
# ISN vocabulary: 18 of the 25 are not ISN locus tokens at all, and the 7 that
# are (bolometer, camera, detector, gauge, sensor, spectrometer, valve) are used
# here as word FRAGMENTS of compound loci. What the words cannot tell us is
# whether a locus names an object or a place — ``detector_pixel`` and
# ``antenna_row`` read as hardware and are declared ``position`` — so that half
# of the question is answered by ISN through ``is_place_locus``. Widening the
# trigger to every ISN entity locus instead would be a different rule: measured
# over the live corpus it takes the flag count from 27 to 189, newly rejecting
# whole families (``area_of_diagnostic_aperture``, ``…_of_optical_element``)
# whose locus is simply not spelled in the DD path.
_HARDWARE_LOCUS_WORDS = _moved_worker_extraction_and_enrichment._HARDWARE_LOCUS_WORDS


#: Ordinal position words the DD uses to index successive samples along ONE
#: object. IMAS-DD PATH vocabulary — the DD gives each sampled position its own
#: sub-structure and names it by its place along the object being sampled.
_ORDINAL_POSITION_WORDS = (
    _moved_worker_extraction_and_enrichment._ORDINAL_POSITION_WORDS
)


#: IMAS-DD path segments for solid hardware cross-section representations.
#: These are DD structure tokens, not a duplicate of ISN grammar vocabulary.
_SOLID_GEOMETRY_PATH_SEGMENTS = (
    _moved_worker_extraction_and_enrichment._SOLID_GEOMETRY_PATH_SEGMENTS
)

#: Words used by IMAS-DD path segments for optical ray/path representations.
#: The name-side comparison below derives its words from the parsed surface
#: locus; this set remains a DD representation policy, not a grammar token list.
_OPTICAL_GEOMETRY_PATH_WORDS = (
    _moved_worker_extraction_and_enrichment._OPTICAL_GEOMETRY_PATH_WORDS
)


def _is_ordinal_point_sample(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._is_ordinal_point_sample."""
    return _moved_worker_extraction_and_enrichment._is_ordinal_point_sample(
        _worker_dependency, *args, **kwargs
    )


def _vector_fields_conflict(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._vector_fields_conflict."""
    return _moved_worker_extraction_and_enrichment._vector_fields_conflict(
        _worker_dependency, *args, **kwargs
    )


def _content_tokens(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._content_tokens."""
    return _moved_worker_extraction_and_enrichment._content_tokens(
        _worker_dependency, *args, **kwargs
    )


def _expanded_path_tokens(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._expanded_path_tokens."""
    return _moved_worker_extraction_and_enrichment._expanded_path_tokens(
        _worker_dependency, *args, **kwargs
    )


#: DD IDS / structure names that abbreviate a device the standard-name grammar
#: spells in full. Distinct from the naming audit's abbreviation list, which
#: covers physics-concept contractions inside a NAME; these are IDS identifiers
#: that only ever appear on the DD PATH side. Not ISN grammar vocabulary — a DD
#: path token mapped to the words a locus token is built from.
_DD_DEVICE_EXPANSIONS = _moved_worker_extraction_and_enrichment._DD_DEVICE_EXPANSIONS


def _name_locus_token(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._name_locus_token."""
    return _moved_worker_extraction_and_enrichment._name_locus_token(
        _worker_dependency, *args, **kwargs
    )


def _geometry_representation_conflicts(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._geometry_representation_conflicts."""
    return _moved_worker_extraction_and_enrichment._geometry_representation_conflicts(
        _worker_dependency, *args, **kwargs
    )


def _dd_standard_name_units_dimensionally_agree(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._dd_standard_name_units_dimensionally_agree."""
    return _moved_worker_extraction_and_enrichment._dd_standard_name_units_dimensionally_agree(
        _worker_dependency, *args, **kwargs
    )


def _is_attachment_consistent(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._is_attachment_consistent."""
    return _moved_worker_extraction_and_enrichment._is_attachment_consistent(
        _worker_dependency, *args, **kwargs
    )


def _process_attachments_core(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._process_attachments_core."""
    return _moved_worker_extraction_and_enrichment._process_attachments_core(
        _worker_dependency, *args, **kwargs
    )


def _process_attachments(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._process_attachments."""
    return _moved_worker_extraction_and_enrichment._process_attachments(
        _worker_dependency, *args, **kwargs
    )


# ---------------------------------------------------------------------------
# StandardNameSource status updaters (incremental tracking)
# ---------------------------------------------------------------------------


def _update_sources_after_compose(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._update_sources_after_compose."""
    return _moved_worker_extraction_and_enrichment._update_sources_after_compose(
        _worker_dependency, *args, **kwargs
    )


def _update_sources_after_attach(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._update_sources_after_attach."""
    return _moved_worker_extraction_and_enrichment._update_sources_after_attach(
        _worker_dependency, *args, **kwargs
    )


def _update_sources_after_vocab_gap(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._update_sources_after_vocab_gap."""
    return _moved_worker_extraction_and_enrichment._update_sources_after_vocab_gap(
        _worker_dependency, *args, **kwargs
    )


def _update_sources_after_skip(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._update_sources_after_skip."""
    return _moved_worker_extraction_and_enrichment._update_sources_after_skip(
        _worker_dependency, *args, **kwargs
    )


def _persist_claimed_skips(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._persist_claimed_skips."""
    return _moved_worker_extraction_and_enrichment._persist_claimed_skips(
        _worker_dependency, *args, **kwargs
    )


def _consume_claimed_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._consume_claimed_vocab_gaps."""
    return _moved_worker_extraction_and_enrichment._consume_claimed_vocab_gaps(
        _worker_dependency, *args, **kwargs
    )


def _search_reference_exemplars(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._search_reference_exemplars."""
    return _moved_worker_extraction_and_enrichment._search_reference_exemplars(
        _worker_dependency, *args, **kwargs
    )


async def _grammar_retry(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._grammar_retry."""
    return await _moved_worker_extraction_and_enrichment._grammar_retry(
        _worker_dependency, *args, **kwargs
    )


async def _self_refine_candidate(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._self_refine_candidate."""
    return await _moved_worker_extraction_and_enrichment._self_refine_candidate(
        _worker_dependency, *args, **kwargs
    )


_COMPOSE_DOMAIN_STATE_QUERY = (
    _moved_worker_extraction_and_enrichment._COMPOSE_DOMAIN_STATE_QUERY
)


async def _write_composed_name_domains(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_extraction_and_enrichment._write_composed_name_domains."""
    return await _moved_worker_extraction_and_enrichment._write_composed_name_domains(
        _worker_dependency, *args, **kwargs
    )


async def compose_worker(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate.compose_worker."""
    return await _moved_worker_compose_and_validate.compose_worker(
        _worker_dependency, *args, **kwargs
    )


# =============================================================================
# VALIDATE phase
# =============================================================================


def _validate_via_isn(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate._validate_via_isn."""
    return _moved_worker_compose_and_validate._validate_via_isn(
        _worker_dependency, *args, **kwargs
    )


def _is_quarantined(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate._is_quarantined."""
    return _moved_worker_compose_and_validate._is_quarantined(
        _worker_dependency, *args, **kwargs
    )


def validate_name_candidate(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate.validate_name_candidate."""
    return _moved_worker_compose_and_validate.validate_name_candidate(
        _worker_dependency, *args, **kwargs
    )


async def drain_validation_backlog(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate.drain_validation_backlog."""
    return await _moved_worker_compose_and_validate.drain_validation_backlog(
        _worker_dependency, *args, **kwargs
    )


# Matches the backlog claim timeout in graph_ops so a stale scoped claim
# expires on the same schedule.
_SCOPED_VALIDATION_CLAIM_TIMEOUT = (
    _moved_worker_compose_and_validate._SCOPED_VALIDATION_CLAIM_TIMEOUT
)


def claim_ids_for_validation(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate.claim_ids_for_validation."""
    return _moved_worker_compose_and_validate.claim_ids_for_validation(
        _worker_dependency, *args, **kwargs
    )


async def drain_validation_for_ids(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate.drain_validation_for_ids."""
    return await _moved_worker_compose_and_validate.drain_validation_for_ids(
        _worker_dependency, *args, **kwargs
    )


async def validate_worker(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate.validate_worker."""
    return await _moved_worker_compose_and_validate.validate_worker(
        _worker_dependency, *args, **kwargs
    )


def _convert_fields_to_grammar(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate._convert_fields_to_grammar."""
    return _moved_worker_compose_and_validate._convert_fields_to_grammar(
        _worker_dependency, *args, **kwargs
    )


# =============================================================================
# CONSOLIDATE phase
# =============================================================================


async def consolidate_worker(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate.consolidate_worker."""
    return await _moved_worker_compose_and_validate.consolidate_worker(
        _worker_dependency, *args, **kwargs
    )


# =============================================================================
# PERSIST phase
# =============================================================================


async def persist_worker(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_and_validate.persist_worker."""
    return await _moved_worker_compose_and_validate.persist_worker(
        _worker_dependency, *args, **kwargs
    )


# =============================================================================
# Pool-mode batch processors
# =============================================================================


async def compose_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_batches.compose_batch."""
    return await _moved_worker_compose_batches.compose_batch(
        _worker_dependency, *args, **kwargs
    )


async def process_generate_name_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_compose_batches.process_generate_name_batch."""
    return await _moved_worker_compose_batches.process_generate_name_batch(
        _worker_dependency, *args, **kwargs
    )


# Substrings that identify a deterministic grammar / vocab / schema failure in
# an LLM-proposed refined name: unregistered grammar token, pydantic enum
# violation, a refine that produced an identical name, or ISN grammar
# composition rejecting the proposed structure.  Each is terminal rather than
# a crash, so the item is routed to the exhaust path instead of being
# re-claimed and re-charged.  The value is owned by the task module.
_GRAMMAR_FAILURE_MARKERS = _moved_worker_compose_batches._GRAMMAR_FAILURE_MARKERS


# The exception types are thrown across the refine and review paths; their
# canonical definitions are owned by the task module.
RefineCandidateValidationError = (
    _moved_worker_refine_names.RefineCandidateValidationError
)
RefineScopeContinuityError = _moved_worker_refine_names.RefineScopeContinuityError


def _refine_batch_scope_id(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_names._refine_batch_scope_id."""
    return _moved_worker_refine_names._refine_batch_scope_id(*args, **kwargs)


def _compose_refined_candidate_name(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_names._compose_refined_candidate_name."""
    return _moved_worker_refine_names._compose_refined_candidate_name(*args, **kwargs)


def _is_refine_grammar_failure(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_names._is_refine_grammar_failure."""
    return _moved_worker_refine_names._is_refine_grammar_failure(
        _worker_dependency, *args, **kwargs
    )


def _refine_failure_stop_reason(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_names._refine_failure_stop_reason."""
    return _moved_worker_refine_names._refine_failure_stop_reason(*args, **kwargs)


def _is_refine_docs_failure(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_names._is_refine_docs_failure."""
    return _moved_worker_refine_names._is_refine_docs_failure(*args, **kwargs)


def _return_unspent_refine_attempt(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_names._return_unspent_refine_attempt."""
    return _moved_worker_refine_names._return_unspent_refine_attempt(
        _worker_dependency, *args, **kwargs
    )


async def process_refine_name_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_names.process_refine_name_batch."""
    return await _moved_worker_refine_names.process_refine_name_batch(
        _worker_dependency, *args, **kwargs
    )


# =============================================================================
# RD-quorum helper (shared by review_name + review_docs pool workers)
# =============================================================================


# The value is owned by the refine task module.
_NAME_REVIEW_SOURCE_CONTEXT_LIMIT = (
    _moved_worker_refine_names._NAME_REVIEW_SOURCE_CONTEXT_LIMIT
)


# RD-quorum name review lives in worker_tasks.worker_review_names. The
# forwarders keep a patch or import through this module reaching the moved
# code; each is bound to its wrapped function below, before the signature
# stripping loop runs.


def _enrich_name_review_items(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_names._enrich_name_review_items."""
    return _moved_worker_review_names._enrich_name_review_items(
        _worker_dependency, *args, **kwargs
    )


def quorum_incomplete_snapshot(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_names.quorum_incomplete_snapshot."""
    return _moved_worker_review_names.quorum_incomplete_snapshot(*args, **kwargs)


def reset_quorum_incomplete(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_names.reset_quorum_incomplete."""
    return _moved_worker_review_names.reset_quorum_incomplete(*args, **kwargs)


async def _run_rd_quorum_cycles(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_names._run_rd_quorum_cycles."""
    return await _moved_worker_review_names._run_rd_quorum_cycles(
        _worker_dependency, *args, **kwargs
    )


async def _persisting_quorum_cycles(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_names._persisting_quorum_cycles."""
    return await _moved_worker_review_names._persisting_quorum_cycles(
        _worker_dependency, *args, **kwargs
    )


def semantic_gate_name_text(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_names.semantic_gate_name_text."""
    return _moved_worker_review_names.semantic_gate_name_text(*args, **kwargs)


async def process_review_name_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_names.process_review_name_batch."""
    return await _moved_worker_review_names.process_review_name_batch(
        _worker_dependency, *args, **kwargs
    )


# The closed vocabulary table is a value, not a call, so it is carried as an
# alias of the owning module's object.
_closed_vocabulary_definitions = (
    _moved_worker_review_names._closed_vocabulary_definitions
)


# =============================================================================
# DD context enrichment for generate_docs
# =============================================================================

_DD_PATH_CONTEXT_QUERY = _moved_worker_review_names._DD_PATH_CONTEXT_QUERY

_DOCS_GEN_ENRICH_QUERY = _moved_worker_review_names._DOCS_GEN_ENRICH_QUERY

_DOCS_GEN_COCOS_PARAMS_QUERY = _moved_worker_review_names._DOCS_GEN_COCOS_PARAMS_QUERY

_DOCS_GEN_PARENT_CHILDREN_QUERY = (
    _moved_worker_review_names._DOCS_GEN_PARENT_CHILDREN_QUERY
)

_DOCS_GEN_NEARBY_QUERY = _moved_worker_review_names._DOCS_GEN_NEARBY_QUERY


def _enrich_dd_path_context(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_generate_docs._enrich_dd_path_context."""
    return _moved_worker_generate_docs._enrich_dd_path_context(
        _worker_dependency, *args, **kwargs
    )


_enrich_dd_path_context.__wrapped__ = (
    _moved_worker_generate_docs._enrich_dd_path_context
)


def _enrich_for_docs_gen(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_generate_docs._enrich_for_docs_gen."""
    return _moved_worker_generate_docs._enrich_for_docs_gen(
        _worker_dependency, *args, **kwargs
    )


_enrich_for_docs_gen.__wrapped__ = _moved_worker_generate_docs._enrich_for_docs_gen


def _nearby_names_for_docs_gen(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_generate_docs._nearby_names_for_docs_gen."""
    return _moved_worker_generate_docs._nearby_names_for_docs_gen(
        _worker_dependency, *args, **kwargs
    )


_nearby_names_for_docs_gen.__wrapped__ = (
    _moved_worker_generate_docs._nearby_names_for_docs_gen
)


def _fill_template_sentence(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_generate_docs._fill_template_sentence."""
    return _moved_worker_generate_docs._fill_template_sentence(
        _worker_dependency, *args, **kwargs
    )


_fill_template_sentence.__wrapped__ = (
    _moved_worker_generate_docs._fill_template_sentence
)


def _instantiated_template_sentences(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_generate_docs._instantiated_template_sentences."""
    return _moved_worker_generate_docs._instantiated_template_sentences(
        _worker_dependency, *args, **kwargs
    )


_instantiated_template_sentences.__wrapped__ = (
    _moved_worker_generate_docs._instantiated_template_sentences
)


def _documentation_with_registry_form(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_generate_docs._documentation_with_registry_form."""
    return _moved_worker_generate_docs._documentation_with_registry_form(
        _worker_dependency, *args, **kwargs
    )


_documentation_with_registry_form.__wrapped__ = (
    _moved_worker_generate_docs._documentation_with_registry_form
)


async def process_generate_docs_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_generate_docs.process_generate_docs_batch."""
    return await _moved_worker_generate_docs.process_generate_docs_batch(
        _worker_dependency, *args, **kwargs
    )


process_generate_docs_batch.__wrapped__ = (
    _moved_worker_generate_docs.process_generate_docs_batch
)


async def process_enrich_parents_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_enrich_parents.process_enrich_parents_batch."""
    return await _moved_worker_enrich_parents.process_enrich_parents_batch(
        _worker_dependency, *args, **kwargs
    )


process_enrich_parents_batch.__wrapped__ = (
    _moved_worker_enrich_parents.process_enrich_parents_batch
)


_DOCS_REVIEW_VOLATILE_ITEM_FIELDS = (
    _moved_worker_enrich_parents._DOCS_REVIEW_VOLATILE_ITEM_FIELDS
)


def _canonical_docs_review_item(item: dict[str, Any]) -> dict[str, Any]:
    """Copy the prompt-bearing fields while excluding claim/lifecycle state."""
    import copy

    return {
        key: copy.deepcopy(value)
        for key, value in item.items()
        if key not in _DOCS_REVIEW_VOLATILE_ITEM_FIELDS
    }


def _bound_docs_review_dd_paths(item: dict[str, Any]) -> list[str]:
    """List every DD path bound to the name, preserving source order.

    Only DD paths are returned: the paths of ``dd`` source bindings, the
    ``dd_source_paths`` a prepared review item carries, or ``source_paths``
    entries carrying the ``dd:`` prefix. A facility signal path names no DD
    text, so it never becomes a required DD path. ``dd_source_paths`` is this
    function's own output for the item, kept beside the ``dd:``-stripped
    ``source_paths`` a prepared review item is normalised to, so the DD
    provenance survives that normalisation.
    """
    bindings = item.get("source_bindings") or []
    paths = [
        binding.get("dd_path") or binding.get("source_id") or binding.get("id")
        for binding in bindings
        if binding and binding.get("source_type") == "dd"
    ]
    if not paths:
        paths = item.get("dd_source_paths")
        if paths is None:
            paths = item.get("source_paths") or []
            if isinstance(paths, str):
                try:
                    paths = json.loads(paths)
                except (json.JSONDecodeError, TypeError):
                    paths = [paths]
            if isinstance(paths, str):
                paths = [paths]
            paths = [
                path
                for path in paths
                if isinstance(path, str) and path.startswith(DD_PREFIX)
            ]
    if isinstance(paths, str):
        paths = [paths]
    return list(
        dict.fromkeys(
            strip_dd_prefix(path) for path in paths if isinstance(path, str) and path
        )
    )


def _set_docs_review_dd_source_paths(item: dict[str, Any]) -> list[str]:
    """Normalise an item's DD paths, keeping the DD provenance beside them.

    ``source_paths`` is set to the bare DD paths so the DD-source read and the
    review prompt see paths without the ``dd:`` prefix. ``dd_source_paths``
    records the same list, so a later :func:`_bound_docs_review_dd_paths` call
    on the normalised item still recognises the paths as DD rather than as
    unlabelled ``source_paths`` entries. A facility signal path is dropped:
    it names no DD text.
    """
    dd_paths = _bound_docs_review_dd_paths(item)
    item["dd_source_paths"] = dd_paths
    item["source_paths"] = dd_paths
    return dd_paths


def _has_dd_source_reference(item: dict[str, Any]) -> bool:
    """Report whether an item names DD source text to compare claims against.

    A DD path reaches the item either as a ``dd`` source binding or as a
    ``dd:``-prefixed entry, which a binding of another source type can also
    carry. Items whose sources are all derived or manual name no DD text, so
    there is nothing for a claim comparison to read.
    """
    if _bound_docs_review_dd_paths(item):
        return True
    paths = item.get("source_paths")
    entries: list[Any] = list(paths) if isinstance(paths, (list, tuple)) else []
    bindings = item.get("source_bindings")
    if isinstance(bindings, (list, tuple)):
        entries.extend(bindings)
    for entry in entries:
        if isinstance(entry, str):
            if entry.startswith(DD_PREFIX):
                return True
            continue
        if isinstance(entry, dict) and any(
            isinstance(entry.get(key), str) and entry[key].startswith(DD_PREFIX)
            for key in ("dd_path", "source_id", "path", "id")
        ):
            return True
    return False


def _load_docs_review_source_item(sn_id: str) -> dict[str, Any]:
    """Read the same prompt-bearing projection returned by a docs claim."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (sn)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            RETURN sn.id AS id,
                   sn.id AS name,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   coalesce(u.id, sn.unit) AS unit,
                   c.id AS cluster_id,
                   sn.physics_domain AS physics_domain,
                   sn.validation_status AS validation_status,
                   sn.tags AS tags,
                   coalesce(sn.docs_chain_length, 0) AS docs_chain_length,
                   sn.docs_stage AS docs_stage,
                   sn.edit_mode AS edit_mode,
                   sn.name_hint AS name_hint,
                   sn.docs_hint AS docs_hint,
                   sn.edit_reason AS edit_reason,
                   sn.edit_origin AS edit_origin,
                   sn.origin AS origin,
                   sn.run_id AS run_id,
                   sn.source_paths AS source_paths,
                   [(source:StandardNameSource)-[:PRODUCED_NAME]->(sn) | {
                       id: source.id,
                       source_type: source.source_type,
                       source_id: source.source_id,
                       dd_path: source.dd_path,
                       dd_version: source.dd_version
                   }] AS source_bindings
            """,
            id=sn_id,
        )
    if len(rows) != 1:
        raise ValueError(f"documentation review target {sn_id!r} is not unique")
    return dict(rows[0])


def _load_docs_review_parent_children(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the canonical live children for one derived parent candidate."""
    if item.get("origin") != "derived":
        return []
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (p:StandardName {id: $id, origin: 'derived'})
            MATCH (c:StandardName)-[:HAS_PARENT]->(p)
            WHERE NOT (coalesce(c.name_stage, '') IN
                       ['superseded', 'exhausted', 'contested'])
            WITH c ORDER BY c.id
            RETURN collect({
                name: c.id,
                description: CASE WHEN c.description = $placeholder
                    THEN null ELSE c.description END,
                unit: c.unit,
                physics_domain: c.physics_domain
            })[..12] AS children
            """,
            id=item["id"],
            placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
    return list(rows[0].get("children") or []) if rows else []


def _load_docs_review_examples(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the scored documentation examples used by the reviewer prompt."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_review_examples

    domain = item.get("physics_domain") or ""
    with GraphClient() as gc:
        return list(load_review_examples(gc, physics_domains=[domain], axis="docs"))


def _enrich_docs_review_dd_context(item: dict[str, Any]) -> None:
    """Attach the same first-source DD context used at worker dispatch.

    A name with no bound source has no DD path to read, which is a valid
    state rather than an error: review proceeds on the name, description, and
    documentation alone. Reaching for a first element that is not there would
    abort preparation and fail the whole quorum for that name.
    """
    source_paths = item.get("source_paths") or []
    if isinstance(source_paths, list):
        first_source = source_paths[0] if source_paths else None
    else:
        first_source = source_paths
    if isinstance(first_source, str):
        first_source = strip_dd_prefix(first_source)
    if not first_source:
        return
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        _enrich_dd_path_context(gc, item, first_source)


def _docs_review_request_identity(payload: dict[str, Any]) -> str:
    """Hash a canonical rendered request and its priced reviewer sequence."""
    import hashlib

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _bounded_docs_escalation_factory(
    *,
    base_user_prompt: str,
    system_prompt: str,
    response_model: type[Any],
    input_token_bound: int,
) -> Callable[[list[dict[str, Any]]], str]:
    """Render the canonical critique request within a proven input bound.

    The canonical budget infrastructure proves input tokens are no greater
    than the complete serialized UTF-8 byte length plus chat framing. This
    factory binary-searches the critique prefix against that same complete
    request serialization, so the actual escalator request can never exceed
    the cataloged model input limit used for admission pricing.
    """
    if input_token_bound <= 0:
        raise ValueError("escalator input token bound must be positive")
    from imas_codex.standard_names.budget import _CHAT_FRAMING_TOKEN_ALLOWANCE

    serialized_byte_bound = input_token_bound - _CHAT_FRAMING_TOKEN_ALLOWANCE
    if serialized_byte_bound <= 0:
        raise ValueError("escalator input allowance cannot cover chat framing")
    marker = "\n\n## Prior reviewer critiques for authoritative escalation\n"
    schema = response_model.model_json_schema()

    def _serialized_size(user_prompt: str) -> int:
        return len(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_schema": schema,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def _render(prior_reviews: list[dict[str, Any]]) -> str:
        critique = json.dumps(
            prior_reviews,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        prefix = base_user_prompt + marker
        if _serialized_size(prefix) > serialized_byte_bound:
            raise ValueError("base escalation request exceeds catalog input bound")
        low, high = 0, len(critique)
        while low < high:
            midpoint = (low + high + 1) // 2
            if _serialized_size(prefix + critique[:midpoint]) <= serialized_byte_bound:
                low = midpoint
            else:
                high = midpoint - 1
        return prefix + critique[:low]

    return _render


async def prepare_docs_review_request(item: dict[str, Any]) -> dict[str, Any]:
    """Render and price one canonical docs-review request without mutation.

    This is the sole docs-review prompt construction path. Admission and the
    claimed worker both call it, so context, templates, reviewer models,
    response schema, effort settings, possible escalator exposure, and request
    identity cannot drift through duplicated preparation code.
    """
    import asyncio as _asyncio

    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_model,
        get_sn_review_disagreement_threshold,
        get_sn_review_docs_models,
        get_sn_review_escalation_reasoning_effort,
        get_sn_review_names_models,
        get_sn_review_reasoning_effort,
    )
    from imas_codex.standard_names.context import (
        conventions_context_for,
        fetch_review_neighbours,
    )
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewDocsBatch,
        StandardNameQualityReviewDocsParentBatch,
    )

    source_item = item
    if item.get("docs_stage") in {"accepted", "reviewed", "exhausted"}:
        source_item = await _asyncio.to_thread(
            _load_docs_review_source_item, str(item.get("id") or "")
        )
    prepared_item = _canonical_docs_review_item(source_item)
    try:
        review_models = list(get_sn_review_docs_models())
    except (ValueError, IndexError):
        review_models = []
    if not review_models:
        try:
            review_models = list(get_sn_review_names_models())
        except (ValueError, IndexError):
            review_models = []
    if not review_models:
        review_models = [get_model("sn-refine")]

    try:
        disagreement_threshold = get_sn_review_disagreement_threshold()
    except Exception:
        disagreement_threshold = 0.20

    try:
        children = await _asyncio.to_thread(
            _load_docs_review_parent_children, prepared_item
        )
    except Exception:
        logger.debug("review_docs: parent children fetch failed", exc_info=True)
        children = []
    if children:
        prepared_item["derived_children"] = children

    try:
        neighbours = await _asyncio.to_thread(fetch_review_neighbours, prepared_item)
    except Exception:
        logger.debug(
            "review_docs: neighbour fetch failed for %s",
            prepared_item.get("id"),
            exc_info=True,
        )
        neighbours = {
            "vector_neighbours": [],
            "same_base_neighbours": [],
            "same_path_neighbours": [],
        }
    try:
        review_examples = await _asyncio.to_thread(
            _load_docs_review_examples, prepared_item
        )
    except Exception:
        logger.debug(
            "review_docs: scored-example load failed for %s",
            prepared_item.get("id"),
            exc_info=True,
        )
        review_examples = []
    try:
        await _asyncio.to_thread(_enrich_docs_review_dd_context, prepared_item)
    except Exception:
        logger.debug(
            "review_docs: dd_path_context failed for %s",
            prepared_item.get("id"),
            exc_info=True,
        )

    prompt_context: dict[str, Any] = {
        "item": prepared_item,
        **neighbours,
        "review_scored_examples": review_examples,
    }
    prompt_context["conventions"] = conventions_context_for(prepared_item["id"])
    is_parent = bool(prepared_item.get("derived_children"))
    user_template = "sn/review_docs_parent_user" if is_parent else "sn/review_docs_user"
    system_template = (
        "sn/review_docs_parent_system" if is_parent else "sn/review_docs_system"
    )
    user_prompt = render_prompt(user_template, prompt_context)
    system_prompt = render_prompt(system_template, prompt_context)
    if is_parent and prompt_context["conventions"]:
        user_prompt += "\n\n" + render_prompt("sn/_conventions", prompt_context)
    response_model = (
        StandardNameQualityReviewDocsParentBatch
        if is_parent
        else StandardNameQualityReviewDocsBatch
    )
    rubric_dims = (
        ("generalization", "positioning", "physics_accuracy", "clarity")
        if is_parent
        else (
            "description_quality",
            "documentation_quality",
            "completeness",
            "physics_accuracy",
        )
    )
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    mandatory_exposures = [
        model_provider_exposure(
            model,
            base_messages,
            response_model=response_model,
            provider_attempts=1,
        )
        for model in review_models[:2]
    ]
    conditional_exposure = 0.0
    escalation_input_token_bound = None
    escalation_prompt_factory = None
    if len(review_models) >= 3:
        from imas_codex.discovery.base.llm import get_catalog_model_info

        catalog_bound = get_catalog_model_info(review_models[2]).get("max_input_tokens")
        if not isinstance(catalog_bound, int) or catalog_bound <= 0:
            raise ValueError("escalator catalog input allowance is not bounded")
        escalation_input_token_bound = catalog_bound
        escalation_prompt_factory = _bounded_docs_escalation_factory(
            base_user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_model=response_model,
            input_token_bound=catalog_bound,
        )
        worst_case_prompt = escalation_prompt_factory(
            [{"critique": "x" * catalog_bound}]
        )
        escalation_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": worst_case_prompt},
        ]
        conditional_exposure = model_provider_exposure(
            review_models[2],
            escalation_messages,
            response_model=response_model,
            provider_attempts=1,
        )
    exposures = [*mandatory_exposures]
    if conditional_exposure > 0:
        exposures.append(conditional_exposure)

    reasoning_effort = get_sn_review_reasoning_effort()
    escalation_reasoning_effort = get_sn_review_escalation_reasoning_effort()
    identity_payload = {
        "id": prepared_item["id"],
        "models": review_models,
        "messages": base_messages,
        "response_schema": response_model.model_json_schema(),
        "rubric_dims": list(rubric_dims),
        "disagreement_threshold": disagreement_threshold,
        "reasoning_effort": reasoning_effort,
        "escalation_reasoning_effort": escalation_reasoning_effort,
        "expected_exposures": exposures,
        "mandatory_exposures": mandatory_exposures,
        "conditional_exposure": conditional_exposure,
        "escalation_input_token_bound": escalation_input_token_bound,
    }
    veto_item = _canonical_docs_review_item(prepared_item)
    _set_docs_review_dd_source_paths(veto_item)
    if veto_item["source_paths"]:
        from imas_codex.standard_names.review.pipeline import _fetch_review_dd_context

        try:
            await _asyncio.to_thread(_fetch_review_dd_context, [veto_item])
        except Exception:
            logger.warning(
                "review_docs: DD source batch read failed for %s",
                veto_item["id"],
                exc_info=True,
            )
    return {
        **identity_payload,
        "veto_item": veto_item,
        "request_identity": _docs_review_request_identity(identity_payload),
        "expected_exposure": sum(exposures),
        "provider_policy_ceiling_is_separate": True,
        "response_model": response_model,
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "escalation_prompt_factory": escalation_prompt_factory,
    }


async def _release_docs_review_claim(item: dict[str, Any]) -> None:
    """Return one unreviewed documentation claim to the drafted pool."""
    import asyncio as _asyncio

    from imas_codex.standard_names.graph_ops import release_review_docs_failed_claims

    claim_token = item.get("claim_token") or ""
    if not claim_token:
        return
    await _asyncio.to_thread(
        release_review_docs_failed_claims,
        sn_ids=[item["id"]],
        claim_token=claim_token,
        from_stage="drafted",
        to_stage="drafted",
    )


def _documentation_conformance_note(item: dict[str, Any]) -> str | None:
    """Name the registry findings that bar a reviewed document from acceptance.

    A review score judges the prose; it does not check the opening against the
    selected template, a forbidden equation variant, or an equation for a
    registered concept written other than in its canonical form. Those are
    decided here, on the same findings the audit report carries. A canonical
    equation is the converged statement and is accepted. A name the grammar cannot parse
    selects no registry entry and so carries no finding.
    """
    from imas_standard_names.grammar.parser import ParseError

    from imas_codex.standard_names.docs_gates import (
        HouseStyleFindingClass,
        conformance_findings,
    )

    barring = {
        HouseStyleFindingClass.TEMPLATE_OPENING,
        HouseStyleFindingClass.FORBIDDEN_EQUATION_VARIANT,
        HouseStyleFindingClass.FORBIDDEN_LOCUS_GLOSS,
        HouseStyleFindingClass.LOCUS_DEFINITION,
        HouseStyleFindingClass.NON_CANONICAL_EQUATION,
    }
    try:
        findings = conformance_findings(
            item["id"],
            description=item.get("description") or "",
            documentation=item.get("documentation") or "",
        )
    except ParseError:
        logger.debug("review_docs: no registry selection for %s", item["id"])
        return None
    details = [
        f"{finding.finding_class.value}: {finding.detail} [{finding.registry_entry}]"
        for finding in findings
        if finding.finding_class in barring
    ]
    if not details:
        return None
    return (
        "documentation does not conform to the conventions registry — "
        "correct each finding: " + "; ".join(details)
    )


async def _documentation_banned_prose_note(
    item: dict[str, Any], mgr: BudgetManager
) -> str | None:
    """Hold a policy candidate only when prose adjudication confirms it."""
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.prose_adjudicator import _adjudicate_one
    from imas_codex.standard_names.prose_policy import (
        BANNED_PROSE_PATTERNS,
        banned_prose_findings,
    )

    text = f"{item.get('description') or ''}\n{item.get('documentation') or ''}"
    findings = banned_prose_findings(text)
    if not any(findings.values()):
        return None
    try:
        genuine = await _adjudicate_one(
            get_model("sn-prose-adjudicator"),
            item["id"],
            text,
            findings,
            reasoning_effort=get_reasoning_effort("sn-prose-adjudicator", "low"),
            service="standard-names",
            budget_manager=mgr,
        )
    except Exception:
        logger.warning(
            "review_docs: prose adjudication failed for %s; holding candidate",
            item["id"],
            exc_info=True,
        )
        genuine = True
    if not genuine:
        return None
    for category, patterns in BANNED_PROSE_PATTERNS.items():
        for pattern in patterns:
            if match := pattern.search(text):
                return f"banned prose ({category}): {match.group(0)!r}"
    return None


async def _documentation_contradiction_note(
    item: dict[str, Any], mgr: BudgetManager
) -> str | None:
    """Compare authored claims with bound DD text before accepting them."""
    from imas_codex.settings import get_model
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.review import jev_review

    required_paths = _bound_docs_review_dd_paths(item)
    sources = item.get("dd_source_docs") or []
    if not sources and not required_paths and item.get("dd_documentation"):
        sources = [{"documentation": item["dd_documentation"]}]
    present_paths = {
        strip_dd_prefix(str(source.get("id") or source.get("path") or ""))
        for source in sources
    }
    missing_paths = [path for path in required_paths if path not in present_paths]
    if missing_paths:
        logger.warning(
            "review_docs: DD source text missing for %s at %s",
            item["id"],
            ", ".join(missing_paths),
        )
        return "DD source text unavailable at " + ", ".join(missing_paths)
    empty_paths = [
        str(source.get("id") or source.get("path") or "DD source")
        for source in sources
        if not (source.get("documentation") or source.get("description"))
    ]
    if empty_paths:
        logger.warning(
            "review_docs: DD source text empty for %s at %s",
            item["id"],
            ", ".join(empty_paths),
        )
        return "DD source text unavailable at " + ", ".join(empty_paths)
    if not sources:
        if _has_dd_source_reference(item):
            logger.warning("review_docs: DD source text unavailable for %s", item["id"])
            return "DD source text unavailable; claim comparison could not run"
        return None

    states = [
        jev_review.build_docs_state({**item, "dd_source_docs": [source]})
        for source in sources
    ]
    claims = states[0]["documentation_claims"]
    if not claims:
        return None
    questions = {
        key: question
        for key, question in jev_review.docs_question_table(
            jev_review._default_checks(), claims
        ).items()
        if key.startswith("check:claim_contradicts_source:")
    }
    lease = mgr.reserve(0.001 * len(sources), phase="review_docs")
    if lease is None:
        logger.warning("review_docs: claim comparison unfunded for %s", item["id"])
        return "claim comparison unavailable: run budget refused the call"
    try:
        model = get_model("sn-judge")
        results, _ = await jev_review.decide_batch(
            states, questions, model=model, service=jev_review.SERVICE
        )
        if len(results) != len(sources):
            logger.warning("review_docs: claim result count differs for %s", item["id"])
            return "claim comparison unavailable: Jev omitted a DD source"
        contradictions = []
        unavailable = []
        for source, state, result in zip(sources, states, results, strict=True):
            source_path = str(source.get("id") or source.get("path") or "DD source")
            if result is None:
                unavailable.append(source_path)
                continue
            answers, spent = result
            lease.charge_event(
                spent,
                LLMCostEvent(
                    model=model,
                    tokens_in=0,
                    tokens_out=0,
                    sn_ids=(item["id"],),
                    phase="review_docs",
                ),
            )
            verdict = jev_review.aggregate_docs_review(
                state, jev_review._default_checks(), answers
            )
            if verdict["contradiction_veto"]:
                dd_text = source.get("documentation") or source.get("description") or ""
                contradictions.append(
                    f"{source_path}: {verdict['contradiction_reason']}; DD: {dd_text}"
                )
            elif any(
                answer is None
                or answer.get("choice") not in {"supports", "says nothing"}
                for answer in verdict["claim_answers"]
            ):
                unavailable.append(source_path)
        if contradictions:
            return "contradicts source: " + " | ".join(contradictions)
        if unavailable:
            logger.warning(
                "review_docs: claim answer missing for %s at %s",
                item["id"],
                ", ".join(unavailable),
            )
            return "claim comparison unavailable at " + ", ".join(unavailable)
        return None
    except Exception:
        logger.warning(
            "review_docs: claim comparison errored for %s; holding candidate",
            item["id"],
            exc_info=True,
        )
        return "claim comparison unavailable: Jev error"
    finally:
        lease.release_unused()


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
    from imas_codex.standard_names.defaults import (
        DEFAULT_MIN_SCORE,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        persist_reviewed_docs,
        update_review_aggregates,
    )

    processed = 0

    for item in batch:
        if stop_event.is_set():
            await _release_docs_review_claim(item)
            continue

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        lease = None

        try:
            request = await prepare_docs_review_request(item)

            from functools import partial

            from imas_codex.standard_names.review.cascade import adjudicate_review

            quorum = await adjudicate_review(
                "docs",
                request.get("veto_item", item),
                request["models"],
                partial(
                    _persisting_quorum_cycles,
                    sn_id=sn_id,
                    review_axis="docs",
                    response_model=request["response_model"],
                    user_prompt=request["user_prompt"],
                    system_prompt=request["system_prompt"],
                    disagreement_threshold=request["disagreement_threshold"],
                    rubric_dims=tuple(request["rubric_dims"]),
                    lease=lease,
                    budget_manager=mgr,
                    phase="review_docs",
                    acall_llm_structured=acall_llm_structured,
                    reasoning_effort=request["reasoning_effort"],
                    escalation_reasoning_effort=request["escalation_reasoning_effort"],
                    run_id=mgr.run_id,
                    escalation_prompt_factory=request.get("escalation_prompt_factory"),
                    review_group_id=None,
                ),
                mgr=mgr,
            )

            if quorum is None:
                await _release_docs_review_claim(item)
                continue

            score = quorum["winning_score"]
            comments = quorum["winning_comments"]
            comments_per_dim = quorum["winning_comments_per_dim"]
            conformance_note = _documentation_conformance_note(item)
            prose_note = await _documentation_banned_prose_note(item, mgr)
            contradiction_note = await _documentation_contradiction_note(
                request.get("veto_item", item), mgr
            )
            checked_holds = (
                ("registry_conformance", conformance_note, "conventions-registry"),
                ("banned_prose", prose_note, "prose-adjudicator"),
                (
                    "documentation_contradiction",
                    contradiction_note,
                    "jev-contradiction",
                ),
            )
            review_holds: list[dict[str, str]] = []
            for kind, note, raised_by in checked_holds:
                if note is None:
                    continue
                evidence = json.dumps(
                    {
                        "claim": note,
                        "documentation": item.get("documentation") or "",
                        "dd_source_docs": item.get("dd_source_docs") or [],
                    },
                    sort_keys=True,
                    default=str,
                )
                review_holds.append(
                    {
                        "kind": kind,
                        "finding": note,
                        "evidence": evidence,
                        "raised_by": raised_by,
                    }
                )
            if any(note is not None for _, note, _ in checked_holds):
                notes = [
                    note
                    for note in (conformance_note, prose_note, contradiction_note)
                    if note
                ]
                comments = "\n".join(filter(None, (comments, *notes)))
                comments_per_dim = dict(comments_per_dim or {})
                if conformance_note is not None:
                    comments_per_dim["registry_conformance"] = conformance_note
                if prose_note is not None:
                    comments_per_dim["banned_prose"] = prose_note
                if contradiction_note is not None:
                    comments_per_dim["contradicts_source"] = contradiction_note
                logger.info(
                    "review_docs: %s found documentation veto — %s",
                    sn_id,
                    "; ".join(notes),
                )
            new_stage = await _asyncio.to_thread(
                persist_reviewed_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                claim_seq=item.get("claim_seq"),
                score=score,
                scores=quorum["winning_scores"],
                comments=comments,
                comments_per_dim=comments_per_dim,
                review_holds=review_holds,
                checked_hold_kinds={kind for kind, _, _ in checked_holds},
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
                resolution_method=quorum["resolution_method"],
                reviewer_chain_size=len(request["models"]),
            )
            if new_stage:
                source_versions = _review_dd_source_bindings(item)
                await _asyncio.to_thread(
                    _persist_dd_gap_evidence,
                    quorum["dd_gaps"],
                    set(source_versions),
                    phase="review_docs",
                    reporter="review-docs",
                    source_versions=source_versions,
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

        except asyncio.CancelledError:
            await _release_docs_review_claim(item)
            raise
        except Exception:
            logger.exception("review_docs failed for %s", sn_id)
            try:
                await _release_docs_review_claim(item)
            except Exception:
                logger.debug("documentation claim cleanup also failed for %s", sn_id)
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
    """Process a batch of reviewed StandardNames for docs refinement.

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
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import conventions_context_for
    from imas_codex.standard_names.defaults import (
        DEFAULT_ESCALATION_MODEL,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        _mark_refine_docs_exhausted,
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
            # Dedicated refine seat — see the refine_name comment for why
            # refine owns its own model choice rather than sharing compose's.
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
            # Expert edit steering (imas-codex sn edit) — the refine_docs_user
            # template reads these as top-level context vars, not item.*,
            # because this prompt_context is a hand-built whitelist rather
            # than an `item` merge.
            "docs_hint": item.get("docs_hint"),
            "edit_reason": item.get("edit_reason"),
            "edit_origin": item.get("edit_origin"),
        }
        from imas_codex.standard_names.review.cascade import failed_check_instructions

        failed_checks = failed_check_instructions(item, "docs")
        if failed_checks:
            prompt_context["failed_checks"] = failed_checks
        prompt_context["conventions"] = conventions_context_for(sn_id)

        # Best-effort DD path enrichment
        from imas_codex.standard_names.dd_resolutions import DDResolutionError

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
                from imas_codex.settings import get_dd_version
                from imas_codex.standard_names.dd_resolutions import resolve_dd_rows

                raw_rows = [dict(row) for row in dd_rows]
                batch = resolve_dd_rows(raw_rows, dd_version=get_dd_version())
                if batch.refusals:
                    logger.warning(
                        "Refine context omitted %d changed DD fields: %s",
                        len(batch.refusals),
                        ", ".join(
                            f"{refusal.path}:{refusal.field.value}"
                            for refusal in batch.refusals
                        ),
                    )
                prompt_context["dd_paths"] = [
                    {
                        **raw_rows[resolved.row_index],
                        **resolved.context.as_pipeline_item(),
                    }
                    for resolved in batch.resolved
                ]
        except DDResolutionError:
            raise
        except Exception:
            logger.debug("refine_docs: DD path enrichment failed for %s", sn_id)

        # Parent-aware refine: inject children (origin='derived' only) so the
        # rewrite GENERALISES over them rather than re-hugging one child — the
        # parent-aware reviewer dock that routes single-child parents here is
        # for over-specialisation, which only children-grounding can fix.
        try:
            from imas_codex.graph.client import GraphClient as _GCRefKids
            from imas_codex.standard_names.defaults import (
                DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER as _PH_REF,
            )

            with _GCRefKids() as gc:
                krows = gc.query(
                    """
                    MATCH (p:StandardName {id: $sn_id}) WHERE p.origin = 'derived'
                    MATCH (c:StandardName)-[:HAS_PARENT]->(p)
                    WHERE NOT coalesce(c.name_stage, '') IN ['superseded', 'exhausted', 'contested']
                    WITH c ORDER BY c.id
                    RETURN c.id AS name,
                           CASE WHEN c.description = $ph THEN null
                                ELSE c.description END AS description,
                           c.unit AS unit, c.physics_domain AS physics_domain
                    LIMIT 12
                    """,
                    sn_id=sn_id,
                    ph=_PH_REF,
                )
                kids = [dict(r) for r in krows]
                if kids:
                    prompt_context["derived_children"] = kids
        except Exception:
            logger.debug("refine_docs: derived children fetch failed for %s", sn_id)

        # Sibling-family context so the rewrite converges on the family's
        # parallel documentation template rather than drifting further.
        try:
            from imas_codex.standard_names.context import fetch_sibling_family

            family = fetch_sibling_family(sn_id)
            if family and family.get("siblings"):
                prompt_context["sibling_family"] = family
        except Exception:
            logger.debug("refine_docs: sibling family fetch failed for %s", sn_id)

        # Locus-defining cross-link context, matching generated documentation.
        try:
            from imas_codex.standard_names.context import locus_context_for

            prompt_context["locus_context"] = locus_context_for(sn_id)
        except Exception:
            logger.debug("refine_docs: locus context fetch failed for %s", sn_id)

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

        _messages = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if system_prompt
            else [{"role": "user", "content": user_prompt}]
        )
        maximum_exposure = model_provider_exposure(
            model,
            _messages,
            response_model=RefinedDocs,
            provider_attempts=1,
        )
        lease = mgr.reserve(maximum_exposure, phase="refine_docs")
        if lease is None:
            continue

        # ── LLM call ──────────────────────────────────────────────
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=RefinedDocs,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-refine"),
                before_attempt=bind_attempt_exposure(
                    lease, model, _messages, response_model=RefinedDocs
                ),
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
                description=normalize_description_text(result_obj.description),
                documentation=_documentation_with_registry_form(
                    sn_id, result_obj.documentation, prompt_context["conventions"]
                ),
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

        except Exception as exc:
            charge_billable_exception(
                lease,
                exc,
                model=model,
                sn_ids=(sn_id,),
                phase="refine_docs",
            )
            # A deterministic ``RefinedDocs`` validation failure (docs that
            # consistently violate the schema length bounds) is a NORMAL
            # failed-refine outcome, not a crash — the model keeps producing
            # the same invalid docs for this item.  Mark it exhausted instead
            # of reverting to 'reviewed', which would re-claim and re-charge it
            # on a paid model every cycle (an infinite paid loop).  Log such
            # failures at WARNING; only unexpected errors get ERROR + traceback.
            is_terminal = _is_refine_docs_failure(exc)
            if is_terminal:
                logger.warning(
                    "refine_docs failed (deterministic, marking exhausted) for %s: %s",
                    sn_id,
                    str(exc)[:200],
                )
            else:
                logger.exception("refine_docs failed for %s", sn_id)
            try:
                if is_terminal:
                    await _asyncio.to_thread(
                        _mark_refine_docs_exhausted,
                        sn_id=sn_id,
                        token=claim_token,
                        error_msg=str(exc)[:500],
                    )
                else:
                    await _asyncio.to_thread(
                        release_refine_docs_failed_claims,
                        sn_ids=[sn_id],
                        claim_token=claim_token,
                    )
            except Exception:
                logger.debug(
                    "release/exhaust refine_docs also failed for %s",
                    sn_id,
                )
            if on_event is not None:
                on_event(
                    {
                        "pool": "refine_docs",
                        "name": sn_id,
                        "outcome": "refine_failed",
                        "model": model,
                        "cost": 0.0,
                    }
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


backfill_semantic_similarity_gate_resolution_method.__wrapped__ = (
    _moved_worker_common.backfill_semantic_similarity_gate_resolution_method
)
normalize_spelling.__wrapped__ = _moved_worker_common.normalize_spelling
normalize_prose_spelling.__wrapped__ = _moved_worker_common.normalize_prose_spelling
normalize_description_notation.__wrapped__ = (
    _moved_worker_common.normalize_description_notation
)
normalize_description_text.__wrapped__ = _moved_worker_common.normalize_description_text
_dedup_adjacent_tokens.__wrapped__ = _moved_worker_common._dedup_adjacent_tokens
_isn_segment_tokens.__wrapped__ = _moved_worker_common._isn_segment_tokens
_isn_physical_bases.__wrapped__ = _moved_worker_common._isn_physical_bases
shape_parameter_base_drift.__wrapped__ = _moved_worker_common.shape_parameter_base_drift
_shape_parameter_bases.__wrapped__ = _moved_worker_common._shape_parameter_bases
_rate_natured_bases.__wrapped__ = _moved_worker_common._rate_natured_bases
_state_resolution_tokens.__wrapped__ = _moved_worker_common._state_resolution_tokens
_name_denotes_state.__wrapped__ = _moved_worker_common._name_denotes_state
_name_denotes_rate.__wrapped__ = _moved_worker_common._name_denotes_rate
_ir_denotes_time_change.__wrapped__ = _moved_worker_common._ir_denotes_time_change
_ir_claims_time_change_in_scope.__wrapped__ = (
    _moved_worker_common._ir_claims_time_change_in_scope
)
_name_claims_time_change.__wrapped__ = _moved_worker_common._name_claims_time_change
_time_change_claim_is_binary_scoped.__wrapped__ = (
    _moved_worker_common._time_change_claim_is_binary_scoped
)
_shape_parameter_surface.__wrapped__ = _moved_worker_common._shape_parameter_surface
_inject_shape_parameter_surface.__wrapped__ = (
    _moved_worker_common._inject_shape_parameter_surface
)
is_well_formed_candidate.__wrapped__ = _moved_worker_common.is_well_formed_candidate
_claimed_source_ids.__wrapped__ = _moved_worker_common._claimed_source_ids
_strip_response_source_id.__wrapped__ = _moved_worker_common._strip_response_source_id
_review_dd_source_bindings.__wrapped__ = _moved_worker_common._review_dd_source_bindings
_review_dd_source_ids.__wrapped__ = _moved_worker_common._review_dd_source_ids
_sanitize_dd_gap_evidence.__wrapped__ = _moved_worker_common._sanitize_dd_gap_evidence
_persist_dd_gap_evidence.__wrapped__ = _moved_worker_common._persist_dd_gap_evidence
_claimed_source_outcome.__wrapped__ = _moved_worker_common._claimed_source_outcome
_sanitize_compose_result_sources.__wrapped__ = (
    _moved_worker_common._sanitize_compose_result_sources
)
_filter_persist_candidates_to_claimed_sources.__wrapped__ = (
    _moved_worker_common._filter_persist_candidates_to_claimed_sources
)
_withhold_binding_conflicted_sources.__wrapped__ = (
    _moved_worker_common._withhold_binding_conflicted_sources
)
_candidate_source_node_id.__wrapped__ = _moved_worker_common._candidate_source_node_id
_persist_description_checked_candidates.__wrapped__ = (
    _moved_worker_common._persist_description_checked_candidates
)
_mark_source_prevalidation_failed.__wrapped__ = (
    _moved_worker_common._mark_source_prevalidation_failed
)
provenance_canonical_names.__wrapped__ = _moved_worker_common.provenance_canonical_names
is_non_nameable_coordinate.__wrapped__ = _moved_worker_common.is_non_nameable_coordinate
_coerce_llm_metric.__wrapped__ = _moved_worker_common._coerce_llm_metric
_extract_llm_telemetry.__wrapped__ = _moved_worker_common._extract_llm_telemetry
_load_known_physical_bases.__wrapped__ = _moved_worker_common._load_known_physical_bases
_auto_detect_physical_base_gaps.__wrapped__ = (
    _moved_worker_common._auto_detect_physical_base_gaps
)
_compute_token_reuse_hits.__wrapped__ = _moved_worker_common._compute_token_reuse_hits
_grammar_round_trip_failures.__wrapped__ = (
    _moved_worker_common._grammar_round_trip_failures
)
_build_grammar_retry_reason.__wrapped__ = (
    _moved_worker_common._build_grammar_retry_reason
)
_build_token_reuse_retry_reason.__wrapped__ = (
    _moved_worker_common._build_token_reuse_retry_reason
)
_active_editorial_gap_guidance.__wrapped__ = (
    _moved_worker_common._active_editorial_gap_guidance
)
_stamp_dedup_decision.__wrapped__ = _moved_worker_common._stamp_dedup_decision

extract_worker.__wrapped__ = _moved_worker_extraction_and_enrichment.extract_worker
_search_nearby_names.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._search_nearby_names
)
_collect_nearby_name_comparators.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._collect_nearby_name_comparators
)
_rerank_comparators_for_compose.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._rerank_comparators_for_compose
)
_book_rerank_spend.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._book_rerank_spend
)
_reranked_nearby_comparators.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._reranked_nearby_comparators
)
_compose_budget_phase.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._compose_budget_phase
)
_rerank_comparators_for_worker.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._rerank_comparators_for_worker
)
_enrich_ids_context.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._enrich_ids_context
)
_hybrid_search_neighbours_batch.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._hybrid_search_neighbours_batch
)
_hybrid_search_neighbours.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._hybrid_search_neighbours
)
_retry_attempts.__wrapped__ = _moved_worker_extraction_and_enrichment._retry_attempts
_retry_k_expansion.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._retry_k_expansion
)
_mark_vocab_gap_sources.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._mark_vocab_gap_sources
)
_related_path_neighbours.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._related_path_neighbours
)
_enrich_batch_items.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._enrich_batch_items
)
_is_ordinal_point_sample.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._is_ordinal_point_sample
)
_vector_fields_conflict.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._vector_fields_conflict
)
_content_tokens.__wrapped__ = _moved_worker_extraction_and_enrichment._content_tokens
_expanded_path_tokens.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._expanded_path_tokens
)
_name_locus_token.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._name_locus_token
)
_geometry_representation_conflicts.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._geometry_representation_conflicts
)
_dd_standard_name_units_dimensionally_agree.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._dd_standard_name_units_dimensionally_agree
)
_is_attachment_consistent.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._is_attachment_consistent
)
_process_attachments_core.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._process_attachments_core
)
_process_attachments.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._process_attachments
)
_update_sources_after_compose.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._update_sources_after_compose
)
_update_sources_after_attach.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._update_sources_after_attach
)
_update_sources_after_vocab_gap.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._update_sources_after_vocab_gap
)
_update_sources_after_skip.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._update_sources_after_skip
)
_persist_claimed_skips.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._persist_claimed_skips
)
_consume_claimed_vocab_gaps.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._consume_claimed_vocab_gaps
)
_search_reference_exemplars.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._search_reference_exemplars
)
_grammar_retry.__wrapped__ = _moved_worker_extraction_and_enrichment._grammar_retry
_self_refine_candidate.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._self_refine_candidate
)
_write_composed_name_domains.__wrapped__ = (
    _moved_worker_extraction_and_enrichment._write_composed_name_domains
)

compose_batch.__wrapped__ = _moved_worker_compose_batches.compose_batch
process_generate_name_batch.__wrapped__ = (
    _moved_worker_compose_batches.process_generate_name_batch
)

process_refine_name_batch.__wrapped__ = (
    _moved_worker_refine_names.process_refine_name_batch
)
_refine_batch_scope_id.__wrapped__ = _moved_worker_refine_names._refine_batch_scope_id
_compose_refined_candidate_name.__wrapped__ = (
    _moved_worker_refine_names._compose_refined_candidate_name
)
_is_refine_grammar_failure.__wrapped__ = (
    _moved_worker_refine_names._is_refine_grammar_failure
)
_is_refine_docs_failure.__wrapped__ = _moved_worker_refine_names._is_refine_docs_failure
_refine_failure_stop_reason.__wrapped__ = (
    _moved_worker_refine_names._refine_failure_stop_reason
)
_return_unspent_refine_attempt.__wrapped__ = (
    _moved_worker_refine_names._return_unspent_refine_attempt
)

compose_worker.__wrapped__ = _moved_worker_compose_and_validate.compose_worker
_validate_via_isn.__wrapped__ = _moved_worker_compose_and_validate._validate_via_isn
_is_quarantined.__wrapped__ = _moved_worker_compose_and_validate._is_quarantined
validate_name_candidate.__wrapped__ = (
    _moved_worker_compose_and_validate.validate_name_candidate
)
drain_validation_backlog.__wrapped__ = (
    _moved_worker_compose_and_validate.drain_validation_backlog
)
claim_ids_for_validation.__wrapped__ = (
    _moved_worker_compose_and_validate.claim_ids_for_validation
)
drain_validation_for_ids.__wrapped__ = (
    _moved_worker_compose_and_validate.drain_validation_for_ids
)
validate_worker.__wrapped__ = _moved_worker_compose_and_validate.validate_worker
_convert_fields_to_grammar.__wrapped__ = (
    _moved_worker_compose_and_validate._convert_fields_to_grammar
)
consolidate_worker.__wrapped__ = _moved_worker_compose_and_validate.consolidate_worker
persist_worker.__wrapped__ = _moved_worker_compose_and_validate.persist_worker


_enrich_name_review_items.__wrapped__ = (
    _moved_worker_review_names._enrich_name_review_items
)
quorum_incomplete_snapshot.__wrapped__ = (
    _moved_worker_review_names.quorum_incomplete_snapshot
)
reset_quorum_incomplete.__wrapped__ = _moved_worker_review_names.reset_quorum_incomplete
_run_rd_quorum_cycles.__wrapped__ = _moved_worker_review_names._run_rd_quorum_cycles
_persisting_quorum_cycles.__wrapped__ = (
    _moved_worker_review_names._persisting_quorum_cycles
)
semantic_gate_name_text.__wrapped__ = _moved_worker_review_names.semantic_gate_name_text
process_review_name_batch.__wrapped__ = (
    _moved_worker_review_names.process_review_name_batch
)


for _forwarder in tuple(globals().values()):
    if not callable(_forwarder) or getattr(_forwarder, "__module__", None) != __name__:
        continue
    _wrapped = getattr(_forwarder, "__wrapped__", None)
    if _wrapped is None:
        continue
    _signature = _inspect.signature(_wrapped)
    if "_worker_dependency" in _signature.parameters:
        _forwarder.__signature__ = _signature.replace(
            parameters=tuple(_signature.parameters.values())[1:]
        )

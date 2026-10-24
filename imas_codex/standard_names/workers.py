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
import re as _re
from collections.abc import Callable
from functools import cache as _cache, lru_cache  # noqa: F401
from typing import TYPE_CHECKING, Any

from imas_codex.standard_names import vocabulary_definitions as vocabulary_source
from imas_codex.standard_names.budget import (
    bind_attempt_exposure,
    charge_billable_exception,
    model_provider_exposure,
)
from imas_codex.standard_names.defaults import (
    DEFAULT_ESCALATION_MODEL,
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
    worker_extraction_and_enrichment as _moved_worker_extraction_and_enrichment,
    worker_refine_names as _moved_worker_refine_names,
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


def _review_source_binding_sort_key(binding: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable order for exact-source review context."""
    return tuple(
        str(binding.get(field) or "")
        for field in (
            "id",
            "source_id",
            "dd_path",
            "dd_version",
            "dd_documentation",
            "dd_parent_path",
            "dd_parent_documentation",
        )
    )


def _enrich_name_review_items(items: list[dict[str, Any]]) -> None:
    """Attach exact source, canonical DD, and lossless grammar review context.

    Immutable leaf and parent definitions come from each exact
    ``StandardNameSource`` binding projected by the claim.  Rich relational
    context is added through the same batch enrichment used by composition;
    mutable graph enrichment never replaces the pinned source text.
    """
    from imas_codex.standard_names.graph_ops import strict_review_grammar_context

    stubs: list[dict[str, Any]] = []
    owners: list[dict[str, Any]] = []

    for item in items:
        try:
            item.update(strict_review_grammar_context(item["id"]))
        except (KeyError, TypeError, ValueError):
            logger.debug(
                "review_name: strict grammar context unavailable for %s",
                item.get("id"),
                exc_info=True,
            )

        raw_bindings = item.get("source_bindings")
        # Claim results always carry this key.  Its absence identifies direct
        # unit-test/legacy callers and avoids an unnecessary graph round-trip.
        if raw_bindings is None:
            continue
        ordered_bindings = sorted(
            (dict(binding) for binding in (raw_bindings or []) if binding),
            key=_review_source_binding_sort_key,
        )
        bindings: list[dict[str, Any]] = []
        seen_binding_ids: set[str] = set()
        for binding in ordered_bindings:
            binding_id = str(
                binding.get("id")
                or binding.get("source_id")
                or binding.get("dd_path")
                or ""
            )
            if binding_id in seen_binding_ids:
                continue
            seen_binding_ids.add(binding_id)
            bindings.append(binding)
        item["source_bindings"] = bindings
        dd_bindings = [
            binding for binding in bindings if binding.get("source_type") == "dd"
        ]
        if not dd_bindings:
            continue

        item["unpinned_source_count"] = sum(
            binding.get("dd_snapshot_pinned") is not True for binding in dd_bindings
        )
        context_bindings = dd_bindings[:_NAME_REVIEW_SOURCE_CONTEXT_LIMIT]
        omitted_count = len(dd_bindings) - len(context_bindings)
        if omitted_count:
            item["source_context_omitted"] = omitted_count

        source_paths = list(
            dict.fromkeys(
                strip_dd_prefix(
                    binding.get("dd_path") or binding.get("source_id") or ""
                )
                for binding in context_bindings
                if binding.get("dd_path") or binding.get("source_id")
            )
        )
        if source_paths:
            item["source_paths"] = source_paths
            item["source_id"] = source_paths[0]

        source_doc_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for binding in context_bindings:
            source_id = strip_dd_prefix(
                binding.get("dd_path") or binding.get("source_id") or ""
            )
            pinned = binding.get("dd_snapshot_pinned") is True
            dd_version = binding.get("dd_version") or ""
            from imas_codex.standard_names.dd_resolutions import resolve_dd_row

            context = resolve_dd_row(
                {
                    "path": source_id,
                    "unit": binding.get("dd_unit") or item.get("unit") or None,
                    "documentation": binding.get("dd_documentation"),
                    "data_type": binding.get("dd_data_type"),
                    "coordinates": binding.get("dd_coordinates") or (),
                    "lifecycle_status": binding.get("dd_lifecycle_status"),
                    "lifecycle_version": binding.get("dd_lifecycle_version"),
                },
                dd_version=dd_version,
            )
            binding.update(context.as_pipeline_item())
            unit = context.unit or ""
            description = (
                binding.get("enhanced_description") or binding.get("description") or ""
            )
            documentation = context.documentation or ""
            group_key = (
                pinned,
                dd_version,
                unit,
                documentation,
                description,
            )
            group = source_doc_groups.setdefault(
                group_key,
                {
                    "id": source_id,
                    "ids": [],
                    "unit": unit,
                    "description": description,
                    "documentation": documentation,
                    "dd_version": dd_version,
                    "snapshot_pinned": pinned,
                },
            )
            if source_id and source_id not in group["ids"]:
                group["ids"].append(source_id)
        item["dd_source_docs"] = list(source_doc_groups.values())

        parent_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for binding in context_bindings:
            parent_path = binding.get("dd_parent_path") or ""
            documentation = binding.get("dd_parent_documentation") or ""
            if not parent_path and not documentation:
                continue
            pinned = binding.get("dd_snapshot_pinned") is True
            dd_version = binding.get("dd_version") or ""
            if parent_path:
                from imas_codex.standard_names.dd_resolutions import resolve_dd_row

                parent_context = resolve_dd_row(
                    {"path": parent_path, "documentation": documentation},
                    dd_version=dd_version,
                )
                documentation = parent_context.documentation or ""
            group_key = (pinned, dd_version, documentation)
            group = parent_groups.setdefault(
                group_key,
                {
                    "path": parent_path,
                    "paths": [],
                    "documentation": documentation,
                    "source_ids": [],
                    "dd_version": dd_version,
                    "snapshot_pinned": pinned,
                },
            )
            if parent_path and parent_path not in group["paths"]:
                group["paths"].append(parent_path)
            source_id = binding.get("source_id") or ""
            if source_id and source_id not in group["source_ids"]:
                group["source_ids"].append(source_id)
        parent_contexts = list(parent_groups.values())
        if parent_contexts:
            item["dd_parent_contexts"] = parent_contexts
        source_hints = [
            {
                "source_id": binding.get("source_id") or "",
                "hint": binding.get("compose_hint") or "",
                "reason": binding.get("compose_hint_reason") or "",
            }
            for binding in context_bindings
            if binding.get("compose_hint")
        ]
        if source_hints:
            item["source_hints"] = source_hints

        if source_paths:
            stubs.append(
                {
                    "path": source_paths[0],
                    "description": item.get("description"),
                    "documentation": item["dd_source_docs"][0]["documentation"],
                    "unit": item["dd_source_docs"][0]["unit"],
                    "physics_domain": item.get("physics_domain"),
                }
            )
            owners.append(item)

    if not stubs:
        return
    try:
        _enrich_batch_items(stubs)
    except Exception:
        logger.debug("review_name: compose-parity DD enrichment failed", exc_info=True)
        return

    protected = {"path", "description", "documentation", "unit", "physics_domain"}
    for item, stub in zip(owners, stubs, strict=True):
        for key, value in stub.items():
            if key == "cross_ids_paths":
                item["semantic_comparators"] = [
                    {"path": path, "basis": "semantic_cluster"} for path in value
                ]
                continue
            if key not in protected and key not in item:
                item[key] = value


# In-process tally of RD-quorum reviews deferred because a ≥2-model quorum did
# NOT complete — a secondary reviewer failed (throttled or empty response), so
# only cycle 0 (or nothing) survived. Keyed by (run_id, review_axis). Accepting
# such a name on the single surviving review would silently advance it on ONE
# review when the profile demanded ≥2 — a critical correctness failure under
# provider throttling. Instead the review is deferred (the claim is released
# back to ``drafted``) and counted here so the run summary can surface the
# shortfall rather than let acceptance quietly degrade to single-model.
_quorum_incomplete_deferrals: dict[tuple[str, str], int] = {}


def _record_quorum_incomplete(run_id: str | None, review_axis: str) -> None:
    """Record one incomplete-quorum deferral for the (run_id, axis) tally."""
    key = (run_id or "", review_axis)
    _quorum_incomplete_deferrals[key] = _quorum_incomplete_deferrals.get(key, 0) + 1


def _unfunded_cause(unfunded: list[tuple[int, str, float]]) -> str:
    """One cause line naming every seat the run budget refused to fund.

    Renders each refused reservation as ``c<idx> <model> $<exposure>`` so an
    operator can see at a glance which seat a run's cost limit or reviewer
    replica count starved, rather than debugging the reviewer chain for a
    provider outage that never happened.
    """
    seats = ", ".join(
        f"c{idx} {model} ${exposure:.4f}" for idx, model, exposure in unfunded
    )
    return f"budget refused {len(unfunded)} seat(s): {seats}"


def quorum_incomplete_snapshot(run_id: str | None) -> dict[str, int]:
    """Return ``{review_axis: deferrals}`` for *run_id* (empty when none)."""
    rid = run_id or ""
    return {
        axis: n for (r, axis), n in _quorum_incomplete_deferrals.items() if r == rid
    }


def reset_quorum_incomplete(run_id: str | None = None) -> None:
    """Clear incomplete-quorum tallies (all runs, or just *run_id*)."""
    if run_id is None:
        _quorum_incomplete_deferrals.clear()
        return
    for key in [k for k in _quorum_incomplete_deferrals if k[0] == (run_id or "")]:
        del _quorum_incomplete_deferrals[key]


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
    phase: str,
    acall_llm_structured: Callable[..., Any],
    lease: Any = None,
    budget_manager: Any = None,
    reasoning_effort: str | None = None,
    escalation_reasoning_effort: str | None = None,
    run_id: str | None = None,
    escalation_prompt_factory: Callable[[list[dict[str, Any]]], str] | None = None,
    review_group_id: str | None = None,
) -> dict[str, Any] | None:
    """Run the configured RD-quorum reviewer chain for a single StandardName.

    Calls each configured model in turn (cycle 0 = primary, cycle 1 =
    secondary, cycle 2 = escalator). Cycle 1 runs when ≥2 models are
    configured and the primary survived — a lost primary already puts the
    two-review quorum out of reach, so the secondary would be billed for a
    review the completeness guard must discard. Cycle 2 runs only when (a) ≥3
    models are configured AND (b) any rubric dimension differs between cycles 0
    and 1 by more than *disagreement_threshold* (after normalising 0–20 scores
    to 0–1).

    Persists no graph state directly — the returned ``records`` list is
    written by :func:`_persisting_quorum_cycles`, and the SN-side stage
    transition belongs to the caller.

    Each cycle's cost is charged to *lease* via a fresh
    :class:`~imas_codex.standard_names.budget.LLMCostEvent` tagged with
    the cycle id (``c0``/``c1``/``c2``).

    Returns
    -------
    dict | None
        ``None`` when no cycle produced a parseable result (caller
        should release the claim). Otherwise a dict with keys:

        - ``records`` — list of Review record dicts (one per cycle) ready
          for :func:`~imas_codex.standard_names.graph_ops.write_reviews`.
          Share a single ``review_group_id``.
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

    from imas_codex.standard_names.budget import BudgetExceeded, LLMCostEvent

    review_group_id = review_group_id or str(_uuid.uuid4())
    cycles: list[dict[str, Any]] = []  # parsed cycle results (cycle_index 0..N)
    # Seats the run budget refused to fund, as (cycle index, model, exposure).
    # A refused reservation calls no provider and raises nothing, so without
    # this record an underfunded run is indistinguishable from a provider
    # outage: both surface only as a short quorum.
    unfunded: list[tuple[int, str, float]] = []
    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    async def _run_cycle(
        cycle_idx: int,
        model: str,
        *,
        cycle_user_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """Single LLM call. Returns parsed cycle dict or None on failure."""
        nonlocal total_cost, total_tokens_in, total_tokens_out
        # The escalator cycle (idx >= 2) fires only on a flagged disagreement,
        # so it gets a higher reasoning budget on just the contested items; the
        # base cycles run at the cheaper base effort (review is the cost-
        # dominant phase). Falls back to the base effort when unset.
        cycle_effort = (
            escalation_reasoning_effort
            if (cycle_idx >= 2 and escalation_reasoning_effort is not None)
            else reasoning_effort
        )
        cycle_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cycle_user_prompt or user_prompt},
        ]
        cycle_lease = lease
        owns_lease = False
        if budget_manager is not None:
            maximum_exposure = model_provider_exposure(
                model,
                cycle_messages,
                response_model=response_model,
                provider_attempts=1,
            )
            cycle_lease = budget_manager.reserve(maximum_exposure, phase=phase)
            if cycle_lease is None:
                unfunded.append((cycle_idx, model, maximum_exposure))
                logger.warning(
                    "rd_quorum %s cycle %d unfunded for %s (model=%s): the run "
                    "budget could not reserve $%.4f of pre-launch exposure — "
                    "the reviewer was never called",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                    maximum_exposure,
                )
                return None
            owns_lease = True
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=cycle_messages,
                response_model=response_model,
                service="standard-names",
                reasoning_effort=cycle_effort,
                before_attempt=bind_attempt_exposure(
                    cycle_lease, model, cycle_messages, response_model=response_model
                ),
            )
            result_obj, cost, _tokens = llm_out
        except Exception as exc:
            charge_billable_exception(
                cycle_lease,
                exc,
                model=model,
                sn_ids=(sn_id,),
                phase=phase,
            )
            logger.exception(
                "rd_quorum %s cycle %d failed for %s (model=%s)",
                review_axis,
                cycle_idx,
                sn_id,
                model,
            )
            if owns_lease:
                cycle_lease.release_unused()
            return None

        # Unwrap batch response: prompts ask for {"reviews": [...]}; we pass
        # one item at a time so reviews[0] is the per-item ReviewItem object.
        # A wrapper whose `reviews` is empty/None is a malformed/empty response
        # — treat it as a FAILED cycle (return None) rather than falling through
        # to `.scores` on the bare wrapper. That access raises AttributeError,
        # gets swallowed to score=0.0/tier='poor', and would persist a spurious
        # canonical review that drives a good name into refine/exhausted.
        _missing = object()
        reviews_list = getattr(result_obj, "reviews", _missing)
        if reviews_list is not _missing:
            if not reviews_list:
                logger.warning(
                    "rd_quorum %s cycle %d returned empty/no reviews for %s "
                    "(model=%s) — failed cycle",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                )
                if owns_lease:
                    cycle_lease.release_unused()
                return None
            try:
                result_obj = reviews_list[0]
            except (IndexError, TypeError):
                logger.warning(
                    "rd_quorum %s cycle %d returned unusable reviews for %s "
                    "(model=%s) — failed cycle",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                )
                if owns_lease:
                    cycle_lease.release_unused()
                return None

        tokens_in = int(getattr(llm_out, "input_tokens", 0) or 0)
        tokens_out = int(getattr(llm_out, "output_tokens", 0) or 0)
        cached_read = int(getattr(llm_out, "cache_read_tokens", 0) or 0)
        cached_write = int(getattr(llm_out, "cache_creation_tokens", 0) or 0)

        total_cost += cost
        total_tokens_in += tokens_in
        total_tokens_out += tokens_out

        if cycle_lease is not None:
            try:
                cycle_lease.charge_event(
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
            except BudgetExceeded:
                raise
            except Exception:
                logger.debug(
                    "rd_quorum charge_event failed for %s c%d",
                    sn_id,
                    cycle_idx,
                    exc_info=True,
                )
        if owns_lease:
            cycle_lease.release_unused()

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
        dd_gaps = list(getattr(result_obj, "dd_gaps", None) or [])

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
            "dd_gaps": dd_gaps,
        }

    # ── Cycle 0 (primary, blind) ───────────────────────────────────────
    c0 = await _run_cycle(0, models[0])
    if c0 is not None:
        cycles.append(c0)

    # ── Cycle 1 (secondary, blind) — runs whenever ≥ 2 models ──────────
    # A multi-seat profile demands two surviving base reviews, and the
    # escalator fires only when both base cycles already succeeded. So once
    # the primary is lost the quorum can no longer be reached, and calling the
    # secondary buys a review that the completeness guard below is obliged to
    # discard — paid tokens with no reachable outcome. Stop here instead.
    c1 = None
    if len(models) >= 2 and c0 is not None:
        c1 = await _run_cycle(1, models[1])
        if c1 is not None:
            cycles.append(c1)

    # ── Quorum-completeness guard ──────────────────────────────────────
    # A profile with ≥2 models demands ≥2 successful reviews. If a secondary
    # reviewer failed (throttled / empty response) only cycle 0 — or nothing —
    # survived. A single surviving review must NOT be accepted for a name the
    # profile intended for multi-model adjudication, or throttling would
    # silently advance names on ONE review. Defer instead: return None so the
    # caller releases the claim back to ``drafted`` (same path it already takes
    # for a total failure), and count the deferral so the run summary can
    # surface the shortfall. Cycle 2 is not part of this guard: two base seats
    # that AGREE are a complete quorum, but two that DISAGREE are exactly the
    # case the escalator exists to break — a disagreeing pair whose escalator
    # seat the budget refuses to fund is an INCOMPLETE quorum, and the
    # disagreement branch below defers it rather than publishing a mean.
    intended_model_count = len(models)
    successful_cycles = len(cycles)
    if intended_model_count >= 2 and successful_cycles < 2:
        # Name the cause. A seat the budget refused to fund never reached its
        # provider, so reporting only the shortfall sends an operator hunting a
        # provider outage when the fix is the run's cost limit or its reviewer
        # replica count.
        if unfunded:
            cause = _unfunded_cause(unfunded)
        else:
            cause = "reviewer call failed"
        logger.warning(
            "rd_quorum %s incomplete: %d/%d reviews succeeded for %s — "
            "deferring, NOT accepting on a single review (%s)",
            review_axis,
            successful_cycles,
            intended_model_count,
            sn_id,
            cause,
        )
        _record_quorum_incomplete(run_id, review_axis)
        return None

    if not cycles:
        # Single-model profile whose only cycle failed — nothing to persist.
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
        prior_reviews = [
            {
                "role": role,
                "model": review["model"],
                "score": review["score"],
                "scores_json": json.dumps(review["scores"], sort_keys=True),
                "reasoning": review["comments"],
                "comments_per_dim_json": json.dumps(
                    review["comments_per_dim"] or {}, sort_keys=True
                ),
            }
            for role, review in (("primary", c0), ("secondary", c1))
            if review is not None
        ]
        if escalation_prompt_factory is not None:
            escalation_user_prompt = escalation_prompt_factory(prior_reviews)
        else:
            critique_text = "\n".join(
                f"- {review['role']} ({review['model']}): "
                f"scores={review['scores_json']}; {review['reasoning']}"
                for review in prior_reviews
            )
            escalation_user_prompt = (
                f"{user_prompt}\n\n## Prior reviewer critiques for authoritative "
                f"escalation\n{critique_text}"
            )
        c2 = await _run_cycle(
            2,
            models[2],
            cycle_user_prompt=escalation_user_prompt,
        )
        if c2 is not None:
            cycles.append(c2)

    # ── Determine winning score + resolution method ────────────────────
    canonical_model = models[0]  # SN axis attribution always cycle-0 model
    if len(cycles) == 1:
        # Reachable only for a single-model profile (len(models) == 1): a
        # ≥2-model profile with fewer than 2 successful cycles is deferred by
        # the quorum-completeness guard above and never reaches here, so a
        # single review is never a silent degradation of a multi-model quorum.
        winning = cycles[0]
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
        if disagreement and unfunded:
            # Both base seats succeeded but DISAGREE, and the run budget refused
            # to fund the escalator seat that exists to break the tie. A mean of
            # two disagreeing blind seats is not a verdict, and publishing it
            # under max_cycles_reached would make a funding shortfall
            # indistinguishable from a profile with no tie-breaker configured.
            # Defer exactly as an incomplete quorum does: release the claim,
            # name the refused seats, and count the deferral.
            cause = _unfunded_cause(unfunded)
            logger.warning(
                "rd_quorum %s deferring disputed name %s — tie-breaker seat "
                "could not be funded (%s)",
                review_axis,
                sn_id,
                cause,
            )
            _record_quorum_incomplete(run_id, review_axis)
            return None
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
        "dd_gaps": [gap for cycle in cycles for gap in cycle["dd_gaps"]],
    }


async def _persisting_quorum_cycles(**kwargs: Any) -> dict[str, Any] | None:
    """Run the quorum cycles and persist their records before returning.

    The calibrated cascade records its shadow route after the incumbent's rows,
    so the incumbent writes its own reviews here instead of handing them back to
    the caller. A cycle chain that produced no reviews returns ``None`` and
    writes nothing.
    """
    result = await _run_rd_quorum_cycles(**kwargs)
    if result is not None:
        from imas_codex.standard_names.graph_ops import write_reviews

        await asyncio.to_thread(write_reviews, result["records"])
    return result


_closed_vocabulary_definitions = vocabulary_source.vocabulary_definitions


def semantic_gate_name_text(name: str) -> str:
    """Return the name-side text the semantic gate scores a description against.

    The gate is a cosine between a name and its description at a floor
    calibrated on multi-word identifiers. A name that is a single vocabulary
    token carries too little text to reach that floor at any description, so
    the coupling rejects the vocabulary's own bare tokens by construction
    rather than on merit. Where the closed vocabulary defines the token, its
    definition is the prose the description should agree with, so score
    against that instead of against the identifier.

    The substitution stays scoped to a name that parses to exactly one token:
    a qualifier carries no definition, and a composed definition of a
    multi-word name scores good names down onto the floor. Every other name —
    multi-token, unparseable, or a token the vocabulary leaves undefined —
    keeps the identifier text, so an unknown token still fails closed.
    """
    try:
        from imas_standard_names.grammar.parser import parse

        ir = parse(name, strict=True).ir
    except Exception:
        return name
    if ir.operators or ir.qualifiers or ir.projection or ir.locus or ir.mechanism:
        return name
    return vocabulary_source.vocabulary_definitions().get(ir.base.token) or name


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
        get_sn_review_escalation_reasoning_effort,
        get_sn_review_names_models,
        get_sn_review_reasoning_effort,
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

    await _asyncio.to_thread(_enrich_name_review_items, batch)

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
        # expensive LLM review and persist a synthetic low score as a
        # non-winning diagnostic verdict: the gate's resolution method
        # records a review_quorum_shortfall, and a refine claim requires
        # that shortfall to be null — so the row is unclaimable by the
        # refine_name pool.
        #
        # The name-side text comes from ``semantic_gate_name_text``: a name
        # that is a single defined vocabulary token is scored against the
        # vocabulary's definition of that token, because one token cannot
        # reach a floor calibrated on multi-word identifiers. Every other
        # name keeps the identifier text.
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
            SEMANTIC_SIM_GATE_MODEL,  # noqa: F811
            SEMANTIC_SIM_GATE_RESOLUTION_METHOD,  # noqa: F811
            SEMANTIC_SIM_SYNTHETIC_SCORE,
            SEMANTIC_SIM_WARNING,
        )

        sem_sim: float | None = None
        sem_issues: list[str] = []
        is_derived = item.get("origin") == "derived"
        if is_derived:
            # ── Desc-name similarity gate for derived parents ─────────────
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
                    semantic_gate_name_text(sn_id),
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
                            "SET sn.semantic_sim = $sim, sn.updated_at = datetime()",
                            id=_id,
                            sim=_sim,
                        )

                await _asyncio.to_thread(_store_sim)
            except Exception:
                pass  # best-effort

        # Critical: skip LLM review and record a non-winning diagnostic verdict.
        # The resulting quorum shortfall prevents both acceptance and a refine
        # claim until the lifecycle owns a deliberate route for this evidence.
        if sem_sim is not None and sem_sim < SEMANTIC_SIM_CRITICAL:
            logger.info(
                "review_name: semantic gate FAILED for %s (sim=%.3f < %.2f) — "
                "recording non-winning verdict",
                sn_id,
                sem_sim,
                SEMANTIC_SIM_CRITICAL,
            )
            try:
                persist_reviewed_name(
                    sn_id=sn_id,
                    claim_token=claim_token,
                    score=SEMANTIC_SIM_SYNTHETIC_SCORE,
                    scores={"semantic": float(sem_sim)},
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
                    model=SEMANTIC_SIM_GATE_MODEL,
                    llm_cost=0.0,
                    llm_tokens_in=0,
                    llm_tokens_out=0,
                    llm_tokens_cached_read=0,
                    llm_tokens_cached_write=0,
                    llm_service="standard-names",
                    run_id=mgr.run_id,
                    resolution_method=SEMANTIC_SIM_GATE_RESOLUTION_METHOD,
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

        # The reviewer must see the SAME closed token registry the composer
        # sees (closed_vocab_full + operators_full): a valid name built on a
        # less-common but registered base (etendue, opacity, …) otherwise reads
        # as an unknown base and is scored down, so it never accepts. The
        # registry is static and lands in the cached system-prompt prefix.
        from imas_codex.standard_names.context import build_compose_context

        prompt_context: dict[str, Any] = {
            **build_compose_context(),
            "items": [item],
            **neighbours,
            **_build_enum_lists(),
            "review_scored_examples": review_scored_examples,
            "prior_reviews": [],
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

        def _render_escalation_prompt(
            prior_reviews: list[dict[str, Any]],
            base_context: dict[str, Any] = prompt_context,
        ) -> str:
            return render_prompt(
                "sn/review_names_user",
                {**base_context, "prior_reviews": prior_reviews},
            )

        # ── RD-quorum cycles ──────────────────────────────────────────
        lease = None
        try:
            from functools import partial

            from imas_codex.standard_names.review.cascade import adjudicate_review

            quorum = await adjudicate_review(
                "name",
                item,
                review_models,
                partial(
                    _persisting_quorum_cycles,
                    sn_id=sn_id,
                    review_axis="name",
                    response_model=StandardNameQualityReviewNameOnlyBatch,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    disagreement_threshold=disagreement_threshold,
                    rubric_dims=("grammar", "semantic", "convention", "completeness"),
                    lease=lease,
                    budget_manager=mgr,
                    phase="review_name",
                    acall_llm_structured=acall_llm_structured,
                    reasoning_effort=get_sn_review_reasoning_effort(),
                    escalation_reasoning_effort=get_sn_review_escalation_reasoning_effort(),
                    run_id=mgr.run_id,
                    escalation_prompt_factory=_render_escalation_prompt,
                ),
                mgr=mgr,
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

            # ── Stage transition with winning score ────────────────────
            new_stage = await _asyncio.to_thread(
                persist_reviewed_name,
                sn_id=sn_id,
                claim_token=claim_token,
                claim_seq=item.get("claim_seq"),
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
                resolution_method=quorum["resolution_method"],
                reviewer_chain_size=len(review_models),
            )
            if new_stage:
                source_versions = _review_dd_source_bindings(item)
                await _asyncio.to_thread(
                    _persist_dd_gap_evidence,
                    quorum["dd_gaps"],
                    set(source_versions),
                    phase="review_name",
                    reporter="review-name",
                    source_versions=source_versions,
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
       n.documentation AS dd_documentation,
       n.description AS dd_description,
       n.unit AS dd_units,
       parent.description AS parent_description,
       clusters,
       changes
"""

_DOCS_GEN_ENRICH_QUERY = """
MATCH (sn:StandardName {id: $sn_id})
OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(imas:IMASNode)
RETURN sn.source_paths AS source_paths,
       sn.cocos_transformation_type AS cocos_label,
       sn.origin AS origin,
       collect(DISTINCT {
           id: imas.id,
           documentation: coalesce(imas.documentation, ''),
           description: coalesce(imas.description, ''),
           alias: imas.name,
           unit: coalesce(imas.unit, '')
       }) AS dd_nodes
"""

_DOCS_GEN_COCOS_PARAMS_QUERY = """
MATCH (dv:DDVersion {id: $ver})-[:HAS_COCOS]->(c:COCOS)
RETURN properties(c) AS cocos_params
"""

# Children grounding for a derived-parent docs item. A derived parent is an
# abstraction over these concrete instances; generate_docs must GENERALISE over
# them (describe the shared quantity), not over-specialise to any one child.
_DOCS_GEN_PARENT_CHILDREN_QUERY = """
MATCH (c:StandardName)-[:HAS_PARENT]->(p:StandardName {id: $sn_id})
WHERE NOT coalesce(c.name_stage, '') IN ['superseded', 'exhausted', 'contested']
RETURN c.id AS name,
       c.description AS description,
       c.unit AS unit,
       c.physics_domain AS physics_domain
ORDER BY c.id LIMIT 12
"""

_DOCS_GEN_NEARBY_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN ['accepted', 'approved']
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

    # DD ground truth — the authoritative definition the reviewer verifies
    # documentation claims against (physics_accuracy grounding).
    dd_doc = row.get("dd_documentation")
    if dd_doc:
        item["dd_documentation"] = dd_doc
    dd_desc = row.get("dd_description")
    if dd_desc:
        item["dd_description"] = dd_desc
    dd_units = row.get("dd_units")
    if dd_units:
        item["dd_units"] = dd_units

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
    - ``sibling_family``: the HAS_PARENT sibling family (parent, anchor,
      siblings) from :func:`~imas_codex.standard_names.context.fetch_sibling_family`
      — drives the parallel-structure directive in the docs prompts.

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
            from imas_codex.settings import get_dd_version
            from imas_codex.standard_names.dd_resolutions import resolve_dd_rows

            batch = resolve_dd_rows(
                [
                    {
                        "path": node["id"],
                        "unit": node.get("unit"),
                        "documentation": node.get("documentation"),
                    }
                    for node in dd_nodes
                ],
                dd_version=get_dd_version(),
            )
            if batch.refusals:
                logger.warning(
                    "Documentation context omitted %d changed DD fields: %s",
                    len(batch.refusals),
                    ", ".join(
                        f"{refusal.path}:{refusal.field.value}"
                        for refusal in batch.refusals
                    ),
                )
            item["dd_source_docs"] = [
                {
                    "id": n["id"],
                    # Rich-first: the LLM-enriched description grounds the docs
                    # prompts; fall back to terse source documentation only when
                    # no enriched description exists.
                    "documentation": n.get("description")
                    or context.documentation
                    or "",
                    "unit": context.unit or "",
                    "raw_dd_context": context.as_pipeline_item()["raw_dd_context"],
                    "published_dd_context": context.as_pipeline_item()[
                        "published_dd_context"
                    ],
                    "dd_resolution_ids": list(context.applied_resolution_ids),
                    "dd_resolution_converged_ids": list(
                        context.converged_resolution_ids
                    ),
                    "dd_resolution_manifest_digest": context.manifest_digest,
                }
                for resolved in batch.resolved
                if resolved.row_index < 5
                for n, context in [(dd_nodes[resolved.row_index], resolved.context)]
            ]
            aliases = [n["alias"] for n in dd_nodes if n.get("alias")]
            if aliases:
                item["dd_aliases"] = aliases

            # Surface the DD ancestor lineage: the primary source leaf is often
            # a terse template stub, while the quantity's physics + evaluation
            # locus live on a parent quantity node up the HAS_PARENT chain
            # (e.g. "…rotation velocity … at the pedestal top"). Mirrors the
            # name-generation grounding so docs ground on the real physics/locus
            # instead of the stub. Nearest rich ancestor first; capped.
            primary = dd_nodes[0]["id"]
            try:
                anc_rows = list(gc.query(_ANCESTOR_CONTEXT_QUERY, path=primary))
            except Exception:
                anc_rows = []
            lineage: list[dict[str, str]] = []
            seen_anc: set[str] = set()
            for a in anc_rows:
                text = (a.get("description") or "").strip() or (
                    a.get("documentation") or ""
                ).strip()
                if not text or text in seen_anc:
                    continue
                seen_anc.add(text)
                lineage.append({"path": a.get("path") or "", "text": text})
                if len(lineage) >= 4:
                    break
            if lineage:
                item["ancestor_context"] = lineage

        # ── 1a. Derived-parent children grounding ─────────────────────
        # A derived parent has no DD source of its own; its concrete physics
        # lives in its accepted children. Inject them so generate_docs writes a
        # GENERALISED parent doc (the shared quantity) rather than over-
        # specialising to one child's component/qualifier (observed defect:
        # ``perturbed_velocity`` documented as its single ``normalized_parallel``
        # child). The placeholder description is dropped (null) so only real
        # child meaning grounds the prompt.
        if row.get("origin") == "derived":
            try:
                from imas_codex.standard_names.defaults import (
                    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
                )

                kid_rows = list(gc.query(_DOCS_GEN_PARENT_CHILDREN_QUERY, sn_id=sn_id))
                children = [
                    {
                        "name": k["name"],
                        "description": (
                            None
                            if k.get("description")
                            == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
                            else k.get("description")
                        ),
                        "unit": k.get("unit"),
                        "physics_domain": k.get("physics_domain"),
                    }
                    for k in kid_rows
                ]
                if children:
                    item["derived_children"] = children
            except Exception:
                logger.debug(
                    "_enrich_for_docs_gen: derived children fetch failed for %s",
                    sn_id,
                    exc_info=True,
                )

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

        # ── 6. Sibling-family (parallel-structure) context ─────────────
        try:
            from imas_codex.standard_names.context import fetch_sibling_family

            family = fetch_sibling_family(sn_id, gc=gc)
            if family and family.get("siblings"):
                item["sibling_family"] = family
        except Exception:
            logger.debug(
                "_enrich_for_docs_gen: sibling family failed for %s",
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


def _fill_template_sentence(sentence: str, fields: dict[str, str]) -> str:
    """Substitute a name's own segments into one template sentence.

    Every ``{placeholder}`` the sentence carries is filled from the name. A
    sentence whose name supplies no value for one of its placeholders renders
    nothing, so a half-filled opening is never written.
    """
    filled = sentence.strip()
    for placeholder in set(_re.findall(r"\{([^{}]+)\}", filled)):
        if placeholder not in fields:
            return ""
        filled = filled.replace(f"{{{placeholder}}}", fields[placeholder])
    return filled


def _instantiated_template_sentences(
    name: str, templates: tuple[Any, ...]
) -> tuple[str | None, tuple[str, ...]]:
    """Fill the template sentences the grammar selects for one name.

    The grammar owns which sentences a name selects and in what order: each
    selected template's opening, then its note once per distinct note. Each
    sentence renders only where the name itself supplies its placeholders, so
    a name that does not fill its template's fields keeps the written opening.

    Returns the first renderable opening and the notes that render.
    """
    from imas_standard_names.grammar.parser import ParseError, parse
    from imas_standard_names.grammar.terms import template_sentences
    from imas_standard_names.grammar.vocab_loaders import load_subjects

    try:
        ir = parse(name, strict=True).ir
    except ParseError:
        return None, ()
    subject_tokens = [q.token for q in ir.qualifiers if q.token in load_subjects()]
    fields: dict[str, str] = {}
    if len(subject_tokens) == 1:
        fields["subject"] = subject_tokens[0].replace("_", " ")
    if ir.locus is not None:
        fields["locus"] = (
            f"{ir.locus.relation.value} {ir.locus.token.replace('_', ' ')}"
        )
    openings = {template.documentation.strip() for template in templates}
    opening: str | None = None
    notes: list[str] = []
    for sentence in template_sentences(name):
        filled = _fill_template_sentence(sentence, fields)
        if not filled:
            continue
        if sentence.strip() in openings:
            if opening is None:
                opening = filled
        elif filled not in notes:
            notes.append(filled)
    return opening, tuple(notes)


def _documentation_with_registry_form(
    name: str, documentation: str, conventions: dict[str, Any] | None
) -> str:
    """Write the selected openings, notes and canonical forms before persistence."""
    from imas_standard_names.grammar.terms import AppliedConventions

    from imas_codex.standard_names.docs_gates import (
        _DISPLAY_MATH_RE,
        HouseStyleFindingClass,
        _is_canonical_equation,
        _locus_registry_tokens,
        _opening_line,
        house_style_findings,
        locus_definition_sentence,
    )

    written = normalize_prose_spelling(documentation)
    if not conventions:
        return written
    applied = AppliedConventions.model_validate(conventions)
    opening, notes = _instantiated_template_sentences(name, applied.templates)
    missing_notes = [note for note in notes if note not in written]
    if opening is not None or missing_notes:
        lines = written.splitlines()
        model_opening = _opening_line(written)
        first = next(
            (i for i, line in enumerate(lines) if line.strip() == model_opening), None
        )
        if first is None:
            written = "\n\n".join(
                ([opening] if opening is not None else []) + missing_notes
            )
        else:
            block = [opening if opening is not None else lines[first], *missing_notes]
            lines[first : first + 1] = ["\n\n".join(block)]
            written = "\n".join(lines)

    concepts = {f"concept:{concept.token}": concept for concept in applied.concepts}
    for match in reversed(list(_DISPLAY_MATH_RE.finditer(written))):
        findings = house_style_findings(applied, documentation=f"$${match.group(1)}$$")
        concept = next(
            (
                concepts[finding.registry_entry]
                for finding in findings
                if finding.registry_entry in concepts
                and finding.finding_class
                in {
                    HouseStyleFindingClass.NON_CANONICAL_EQUATION,
                    HouseStyleFindingClass.FORBIDDEN_EQUATION_VARIANT,
                }
            ),
            None,
        )
        if concept is not None:
            written = (
                written[: match.start()]
                + f"$${concept.equation}$$"
                + written[match.end() :]
            )

    for concept in applied.concepts:
        if concept.restates_equation and not any(
            _is_canonical_equation(concept, match.group(1))
            for match in _DISPLAY_MATH_RE.finditer(written)
        ):
            written = f"{written.rstrip()}\n\n$${concept.equation}$$"

    locus_tokens = _locus_registry_tokens()
    for concept in applied.concepts:
        if concept.token not in locus_tokens:
            continue
        sentence = locus_definition_sentence(concept)
        if sentence and sentence not in written:
            written = f"{written.rstrip()}\n\n{sentence}"
        break
    return written


async def process_generate_docs_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of accepted StandardNames for generate_docs.

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
    from imas_codex.settings import get_model, get_reasoning_effort
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
                            _DOCS_GEN_COCOS_PARAMS_QUERY,
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
        # The docs user template reads ``item.chain_history``. Keep it on the
        # item so refined documentation sees predecessor history.
        item["chain_history"] = chain_history

        # ── Build prompt context ───────────────────────────────────────
        # Merge the cached compose context so the docs SYSTEM prompt's
        # ``_grammar_reference.md`` include renders the closed-vocabulary map
        # (otherwise the grammar block is empty). Cached after first call.
        from imas_codex.standard_names.context import (
            build_compose_context,
            conventions_context_for,
            locus_context_for,
        )

        prompt_context: dict[str, Any] = {
            **build_compose_context(),
            "item": item,
            "chain_history": chain_history,
            "nearby_existing_names": nearby_existing_names,
            "compose_scored_examples": compose_scored_examples,
            # Locus-defining cross-link context: the ISN-registry gloss and
            # position-defining standard quantity for this name's locus, so the
            # docs prompt can link e.g. *_at_pedestal_top ->
            # normalized_poloidal_flux_coordinate_of_pedestal. None when the name
            # has no locus or the locus carries no gloss/defining quantity.
            "locus_context": locus_context_for(sn_id),
            "conventions": conventions_context_for(sn_id),
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
            response_model=GeneratedDocs,
            provider_attempts=1,
        )
        lease = mgr.reserve(maximum_exposure, phase="generate_docs")
        if lease is None:
            continue

        # ── LLM call ──────────────────────────────────────────────────
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=GeneratedDocs,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-docs"),
                before_attempt=bind_attempt_exposure(
                    lease, model, _messages, response_model=GeneratedDocs
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
                    phase="generate_docs",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # ── Persist ───────────────────────────────────────────────
            await _asyncio.to_thread(
                persist_generated_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                description=normalize_description_text(result_obj.description),
                documentation=_documentation_with_registry_form(
                    sn_id, result_obj.documentation, prompt_context["conventions"]
                ),
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

        except Exception as _doc_exc:
            charge_billable_exception(
                lease,
                _doc_exc,
                model=model,
                sn_ids=(sn_id,),
                phase="generate_docs",
            )
            # A "token mismatch or node not found" persist error is an EXPECTED
            # race, not a failure: the orphan sweep reclaimed the item (docs
            # generation can outrun the claim TTL) or another worker took it.
            # The name is not lost — it stays at its prior stage and is
            # re-claimed. Log it quietly; reserve the full traceback for
            # genuinely unexpected errors.
            if "token mismatch or node not found" in str(_doc_exc):
                logger.warning(
                    "generate_docs claim lost for %s (reclaimed mid-generation) "
                    "— will retry on a later pass",
                    sn_id,
                )
            else:
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


async def process_enrich_parents_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Synthesize real descriptions for placeholder derived parents.

    A derived parent materialised by ``_materialize_derived_parent_rows`` keeps
    the deterministic placeholder description until something writes a real one.
    That placeholder deadlocks it: REVIEW_NAME drops it (no real description),
    so it never earns a ``reviewer_score_name``, so generate_docs drops it too.
    This worker breaks the deadlock — for each claimed parent:

    1. Fetch the parent's live children (the concrete instances it abstracts
       over) for grounding.
    2. Render the children-grounded enrichment prompt and call
       ``get_model("sn-parent-enrich")`` (compose-tier; generate_docs rewrites
       the full documentation downstream).
    3. Charge the budget lease.
    4. Embed the synthesised description locally (free) so REVIEW_NAME's
       semantic-similarity check has an embedding immediately.
    5. Persist via :func:`persist_enriched_parent` (writes description +
       embedding, routes ``name_stage`` to ``'drafted'``).

    Returns the count of parents successfully enriched.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.embeddings.description import embed_description
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.graph_ops import (
        fetch_derived_parent_children,
        persist_enriched_parent,
        release_enrich_parents_claims,
    )
    from imas_codex.standard_names.models import EnrichedParentDescription

    model = get_model("sn-parent-enrich")
    processed = 0

    # ── Fetch grounding children for the whole batch (one round-trip) ─────
    try:
        children_by_parent = await _asyncio.to_thread(
            fetch_derived_parent_children, [item["id"] for item in batch]
        )
    except Exception:
        logger.debug(
            "process_enrich_parents_batch: children fetch failed", exc_info=True
        )
        children_by_parent = {}

    # Static grammar/context block (cached after first render) so the system
    # prompt's grammar include renders the closed-vocabulary map.
    _base_ctx = build_compose_context()

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        # The claim returns id as the StandardName id (== the name string).
        item.setdefault("name", sn_id)
        children = children_by_parent.get(sn_id, [])

        # A parent with no live children cannot be grounded — skip (the claim
        # gate already requires ≥1 child, so this is defensive). Release it.
        if not children:
            logger.debug("enrich_parents: %s has no live children — releasing", sn_id)
            try:
                await _asyncio.to_thread(
                    release_enrich_parents_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                pass
            continue

        prompt_context: dict[str, Any] = {
            **_base_ctx,
            "item": item,
            "children": children,
        }
        try:
            user_prompt = render_prompt("sn/enrich_parent_user", prompt_context)
        except Exception:
            logger.debug("enrich_parents: user prompt render failed for %s", sn_id)
            _child_lines = "\n".join(
                f"- {c.get('name')}: {c.get('description') or ''}" for c in children
            )
            user_prompt = (
                f"Describe the derived parent standard name '{sn_id}' "
                f"(unit: {item.get('unit') or '—'}, kind: {item.get('kind') or 'scalar'}) "
                f"by generalising over its children:\n{_child_lines}"
            )
        try:
            system_prompt = render_prompt("sn/enrich_parent_system", prompt_context)
        except Exception:
            logger.debug("enrich_parents: system prompt render failed for %s", sn_id)
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
            response_model=EnrichedParentDescription,
            provider_attempts=1,
        )
        lease = mgr.reserve(maximum_exposure, phase="enrich_parents")
        if lease is None:
            continue

        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=EnrichedParentDescription,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-parent-enrich"),
                before_attempt=bind_attempt_exposure(
                    lease,
                    model,
                    _messages,
                    response_model=EnrichedParentDescription,
                ),
            )
            result_obj, cost, _tokens = llm_out

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
                    phase="enrich_parents",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            description = normalize_description_text(
                (result_obj.description or "").strip()
            )

            # Embed locally (free) so REVIEW_NAME has an embedding immediately.
            embedding = await _asyncio.to_thread(embed_description, description)

            await _asyncio.to_thread(
                persist_enriched_parent,
                sn_id=sn_id,
                claim_token=claim_token,
                description=description,
                embedding=embedding,
                model=model,
                run_id=mgr.run_id,
            )
            processed += 1

            if on_event is not None:
                on_event(
                    {
                        "pool": "enrich_parents",
                        "name": sn_id,
                        "description": description,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception as _enrich_exc:
            charge_billable_exception(
                lease,
                _enrich_exc,
                model=model,
                sn_ids=(sn_id,),
                phase="enrich_parents",
            )
            if "token mismatch or node not found" in str(_enrich_exc):
                logger.warning(
                    "enrich_parents claim lost for %s (reclaimed mid-enrich) "
                    "— will retry on a later pass",
                    sn_id,
                )
            else:
                logger.exception("enrich_parents failed for %s", sn_id)
            try:
                await _asyncio.to_thread(
                    release_enrich_parents_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                logger.debug("release_enrich_parents_claims also failed for %s", sn_id)
        finally:
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


_DOCS_REVIEW_VOLATILE_ITEM_FIELDS = frozenset(
    {
        "claim_token",
        "claim_seq",
        "claimed_at",
        "docs_stage",
        "run_id",
        "reviewer_score_docs",
        "reviewer_scores_docs",
        "reviewer_comments_docs",
        "reviewer_comments_per_dim_docs",
        "reviewer_model_docs",
        "docs_review_quorum_shortfall",
        "docs_review_quorum_shortfall_at",
        "reviewed_docs_at",
    }
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

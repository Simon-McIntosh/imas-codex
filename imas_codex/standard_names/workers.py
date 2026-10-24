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
import random
import re as _re
from collections.abc import Callable, Sequence
from functools import cache as _cache, lru_cache  # noqa: F401
from typing import TYPE_CHECKING, Any

from imas_codex.graph.models import RefineStopReason
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
from imas_codex.standard_names.provenance import detect_value_provenance
from imas_codex.standard_names.source_paths import (
    DD_PREFIX,
    encode_source_path,
    strip_dd_prefix,
)

from .worker_tasks import (
    worker_common as _moved_worker_common,
    worker_extraction_and_enrichment as _moved_worker_extraction_and_enrichment,
)


def _worker_dependency(name: str):
    """Resolve a worker collaborator through the entrypoint."""
    return globals()[name]


if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetLease, BudgetManager
    from imas_codex.standard_names.sources.base import ExtractionBatch
    from imas_codex.standard_names.state import StandardNameBuildState

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

    # --- Domain-vocabulary pre-seeding ---
    # Inject validated domain vocabulary into system prompt context
    from imas_codex.standard_names.context import build_domain_vocabulary_preseed

    domain_vocab = ""
    if state.domain_filter:
        domain_vocab = await asyncio.to_thread(
            build_domain_vocabulary_preseed, state.domain_filter
        )
        if domain_vocab:
            wlog.info("Injected domain vocabulary preseed for %s", state.domain_filter)
    context["domain_vocabulary"] = domain_vocab

    # --- Reviewer-theme extraction ---
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    if state.domain_filter:
        reviewer_themes = await asyncio.to_thread(
            extract_reviewer_themes, state.domain_filter
        )
        if reviewer_themes:
            wlog.info(
                "Extracted %d reviewer themes for %s",
                len(reviewer_themes),
                state.domain_filter,
            )
    context["reviewer_themes"] = reviewer_themes

    # --- Scored-example injection ---
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
            "Loaded %d scored examples for compose (domains=%s)",
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
        batch: ExtractionBatch, lease_box: list[BudgetLease]
    ) -> list[dict] | None:
        # Search for nearby existing names with a bounded share per item.
        nearby = await asyncio.to_thread(_collect_nearby_name_comparators, batch.items)
        nearby = await _rerank_comparators_for_worker(state, batch.items, nearby)

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

        # --- Reference SN few-shot retrieval ---
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
            wlog.debug("Reference exemplar search failed", exc_info=True)

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
        # Name-only batches render a leaner user prompt
        # that trades per-item cluster siblings / COCOS blocks / sibling
        # fields for a "identify natural sub-groups, then name" directive.
        # System prompt and the per-candidate grammar-retry / revision
        # logic are unchanged so
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

        lease = None
        if state.budget_manager:
            maximum_exposure = model_provider_exposure(
                model,
                messages,
                response_model=StandardNameComposeBatch,
                provider_attempts=1,
            )
            phase_tag = _compose_budget_phase(state)
            lease = state.budget_manager.reserve(maximum_exposure, phase=phase_tag)
            if lease is None:
                return None
            lease_box.append(lease)

        # --- Bounded retry loop for failed compositions ---
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
        _collected_dd_gaps: list[Any] = []
        _max_retries = _retry_attempts()
        # Token-reuse hits from the FINAL attempt: candidate-new tokens the
        # agent re-emitted after being shown a near-synonym registered token.
        # Carried to the post-loop VocabGap stamp as distinct_confirmed.
        _token_reuse_hits: dict[tuple[str, str], Any] = {}
        _editorial_guidance = ""
        _editorial_should_retry = False
        for _compose_attempt in range(_max_retries + 1):
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-compose"),
                    before_attempt=bind_attempt_exposure(
                        lease,
                        model,
                        messages,
                        response_model=StandardNameComposeBatch,
                    ),
                )
            except (ValueError, Exception) as compose_exc:
                charge_billable_exception(
                    lease,
                    compose_exc,
                    model=model,
                    batch_id=batch.group_key,
                    phase=getattr(state, "budget_phase_tag", "") or "generate_name",
                )
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
            _sanitize_compose_result_sources(
                result,
                _claimed_source_ids(batch.items),
                phase="generate_name",
            )
            _collected_dd_gaps.extend(result.dd_gaps)
            _total_compose_cost += cost
            _total_tokens_in += getattr(llm_out, "input_tokens", 0) or 0
            _total_tokens_out += getattr(llm_out, "output_tokens", 0) or 0
            _total_cache_read += getattr(llm_out, "cache_read_tokens", 0) or 0
            _total_cache_creation += getattr(llm_out, "cache_creation_tokens", 0) or 0

            # Charge the exact completed-call cost against the exposure that
            # was reserved before the provider request started.
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
                        for _operator in _cand.operators:
                            _segs.append(f"operator={_operator.token}")
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

            # Quick grammar round-trip check on all candidates, with the
            # per-failure repair advice the retry needs to converge.
            _grammar_failures, _grammar_advice = _grammar_round_trip_failures(
                result.candidates
            )

            # Component-token reuse check (free local embeddings): flag any
            # candidate-new token semantically near a registered same-segment
            # token so the agent gets a chained chance to reuse-or-confirm.
            _token_reuse_hits = await asyncio.to_thread(
                _compute_token_reuse_hits, result.vocab_gaps
            )
            _editorial_guidance, _editorial_should_retry = await asyncio.to_thread(
                _active_editorial_gap_guidance, result.vocab_gaps
            )

            if (
                not _grammar_failures
                and not _token_reuse_hits
                and not _editorial_should_retry
            ) or (_compose_attempt >= _max_retries):
                break

            # Re-enrich items with expanded hybrid search for retry
            wlog.info(
                "Composition retry %d/%d: %d grammar failures (%s), "
                "%d token-reuse hits, editorial_retry=%s — "
                "re-composing with expanded DD context",
                _compose_attempt + 1,
                _max_retries,
                len(_grammar_failures),
                ", ".join(_grammar_failures[:3]),
                len(_token_reuse_hits),
                _editorial_should_retry,
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

            # Only re-enrich when grammar failed; a token-reuse-only retry needs
            # no expanded neighbour context (the agent just reuses-or-confirms).
            if _grammar_failures:
                await asyncio.to_thread(_re_enrich_expanded)

            _reason_parts = []
            if _grammar_failures:
                _reason_parts.append(
                    _build_grammar_retry_reason(_grammar_failures, _grammar_advice)
                )
            if _token_reuse_hits:
                _reason_parts.append(_build_token_reuse_retry_reason(_token_reuse_hits))
            if _editorial_guidance:
                _reason_parts.append(_editorial_guidance)
            _retry_reason = "\n\n".join(_reason_parts)
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
        # Deterministic provenance collapse (see pooled path): one canonical
        # name per base quantity so estimator facets never drift apart.
        _prov_canon = provenance_canonical_names(result.candidates, log=wlog)
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
                state.compose_stats.errors += 1
                continue
            unit = raw_unit

            # A name's domain is its stored Jev decision, set by the one writer
            # after the batch is persisted. The DD source's own domain is not
            # carried onto the name here.
            physics_domain = None
            source_domains: list[str] = []

            # Inject COCOS metadata from DD (authoritative, like unit)
            cocos_type = source_item.get("cocos_label") if source_item else None

            # Shape-parameter leaves (triangularity/elongation/squareness) are
            # only meaningful *of* a surface; force the surface locus from the
            # source DD path when the composer left it bare so the leaf is
            # surface-explicit and the boundary/profile siblings de-conflate.
            _inject_shape_parameter_surface(c, c.source_id, wlog)

            # Normalize name via grammar round-trip BEFORE persist
            # to avoid duplicate nodes if validate would rename
            name_id = normalize_spelling(c.compose_name())
            # Provenance collapse: estimator facets of one quantity share the
            # canonical name (MERGE identity) regardless of per-facet drift.
            _pterm, _pbase = detect_value_provenance(c.source_id or "")
            if _pterm and _pbase in _prov_canon:
                name_id = _prov_canon[_pbase]

            # Pre-validation gate — reject malformed LLM output
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
                        _prefix = "dd" if state.source == "dd" else "signals"
                        _mark_source_prevalidation_failed(
                            c.source_id,
                            source_type=_prefix,
                            candidate_name=name_id,
                            reason=_reject_reason or "malformed",
                        )
                    except Exception:
                        wlog.debug("Failed to mark source %s as failed", c.source_id)
                continue

            # Non-nameable coordinate/infrastructure gate: a bare timestamp,
            # latency, counter, or metadata token is not a physics observable.
            # Composing it produces a doomed bare name that fails the semantic
            # gate and burns review + every refine rotation before exhausting.
            # Route the source to ``skipped`` (terminal) so it never re-claims.
            _non_nameable, _skip_reason = is_non_nameable_coordinate(name_id)
            if _non_nameable:
                state.stats["compose_non_nameable_skips"] = (
                    state.stats.get("compose_non_nameable_skips", 0) + 1
                )
                wlog.warning(
                    "Non-nameable skip: %r (%s)",
                    name_id[:80],
                    _skip_reason,
                )
                continue

            # Deterministic source<->name consistency gate (tense + state
            # resolution). The LLM sometimes emits a species-level sibling
            # for a state-resolved source (e.g. thermal_neutral_density
            # sourced from .../neutral/state/density_thermal) -- drop the
            # pairing so the source retries rather than persisting a
            # mis-attributed name. Same predicate the auto-attach path uses.
            if c.source_id:
                _paths = [
                    c.source_id,
                    *(p for p in (c.dd_paths or []) if p != c.source_id),
                ]
                _ok, _why = True, ""
                for _p in _paths:
                    _ok, _why = _is_attachment_consistent(
                        _p, name_id, existing_sources=[q for q in _paths if q != _p]
                    )
                    if not _ok:
                        break
                if not _ok:
                    state.stats["compose_consistency_rejects"] = (
                        state.stats.get("compose_consistency_rejects", 0) + 1
                    )
                    wlog.warning(
                        "Source-name consistency reject: %r (%s)",
                        name_id[:80],
                        _why,
                    )
                    continue

            grammar_failed = False
            try:
                from imas_codex.standard_names.grammar_adapter import (
                    parse_canonical_name,
                )

                parse_canonical_name(name_id)
            except Exception as gram_exc:
                grammar_failed = True
                wlog.debug("Grammar parse failed for %r — attempting retry", name_id)

                # --- Grammar-failure re-prompt (single retry) ---
                state.grammar_retries += 1
                try:
                    retry_name, _l6_cost, _l6_ti, _l6_to = await _grammar_retry(
                        name_id,
                        str(gram_exc),
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
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
                                        "description": f"{batch.group_key}-grammar-retry",
                                    }
                                ]
                            )
                    if retry_name and retry_name != name_id:
                        # Verify the retry result actually parses
                        parse_canonical_name(retry_name)
                        name_id = retry_name
                        grammar_failed = False
                        state.grammar_retries_succeeded += 1
                        wlog.info(
                            "Grammar retry succeeded: %r → %r",
                            c.compose_name(),
                            name_id,
                        )
                except Exception as retry_exc:
                    charge_billable_exception(
                        lease,
                        retry_exc,
                        model=model,
                        sn_ids=(name_id,),
                        batch_id=f"{batch.group_key}-grammar-retry",
                        phase=getattr(state, "budget_phase_tag", "") or "generate_name",
                    )
                    wlog.debug("Grammar retry also failed for %r", name_id)

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
                    "source_claim_token": (source_item or {}).get("claim_token"),
                    "source_claim_seq": (source_item or {}).get("claim_seq"),
                    # Track grammar-retry exhaustion
                    **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
                }
            )

            # --- Mint error siblings deterministically ---
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
                        "Minted %d error siblings for parent %r",
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

            # Stamp the compose-time token-reuse adjudication: gaps re-emitted
            # after being shown a near-synonym registered token are
            # distinct_confirmed; the rest are unchecked.
            _stamp_dedup_decision(gap_dicts, _token_reuse_hits)

            wlog.warning(
                "Retained %d legacy compose vocabulary gaps in memory only; "
                "graph persistence requires fenced pool claims",
                len(gap_dicts),
            )

        # Auto-detect novel physical_base tokens in composed candidates.
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
                    state.stats.setdefault("vocab_gaps", []).extend(novel)
                    wlog.info(
                        "Retained %d legacy physical-base gaps in memory only",
                        len(novel),
                    )
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

            persisted = await asyncio.to_thread(
                persist_generated_name_batch,
                candidates,
                compose_model=model,
                dd_version=batch.dd_version,
                cocos_version=batch.cocos_version,
                run_id=getattr(state.budget_manager, "run_id", None)
                if state.budget_manager
                else None,
                return_winner_ids=True,
            )
            winner_ids = persisted if isinstance(persisted, list) else []
            winner_source_ids = {
                winner.split(":", 1)[1]
                for winner in winner_ids
                if isinstance(winner, str) and ":" in winner
            }
            if not state.dry_run:
                await asyncio.to_thread(
                    _persist_dd_gap_evidence,
                    _collected_dd_gaps,
                    winner_source_ids,
                    phase="generate_name",
                    reporter="compose",
                    observed_dd_version=batch.dd_version,
                )
                await _write_composed_name_domains(candidates, winner_source_ids)
            written = len(winner_ids)
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

            lease_box: list[BudgetLease] = []

            try:
                candidates = await _compose_batch_body(batch, lease_box)
                if candidates is None:
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
                if lease_box:
                    lease_box[0].release_unused()

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

    # --- Batch-size telemetry ---
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
        # Invariant: by the time we build the ISN dict, the unit must come
        # from DD — the dd_unit_unresolvable skip above filters invalid units
        # before this path is reachable. Defensively assert so a regression
        # surfaces in tests rather than silently pollute the catalog
        # with `unit="1"`.
        unit = entry.get("unit")
        assert unit, (
            f"ISN dict missing unit for {entry.get('id')!r}; "
            "dd_unit_unresolvable skip invariant violation"
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
    from imas_codex.standard_names.audits import canonical_locus_check

    canon_issues = canonical_locus_check({"id": isn_dict.get("name", "")})
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


def _is_quarantined(issues: list[str], layer_summary: dict) -> bool:
    """Determine whether validation issues are critical (quarantine the name).

    Critical failures that make a name unusable for publication:
    - Grammar round-trip failure (``parse_error:`` prefix)
    - Pydantic validation failure (layer 1 did not pass)
    - Empty or missing description (no ``id`` or empty string)
    - Invalid kind value
    - Critical audit failures (latex_def_check, synonym_check, multi_subject_check)
    - Grammar retry exhausted
    - ISN semantic/structural ERROR-level issues (ISN's own catalog publish gate
      hard-fails on these, so a name carrying one can never be published)

    Non-critical issues (semantic WARNING/INFO hints, description quality hints,
    non-critical audits) do NOT trigger quarantine — they are advisory.
    """
    # Grammar round-trip failures are always critical
    if any(i.startswith("parse_error:") for i in issues):
        return True

    # ISN semantic/structural ERROR-level issues are catalog-blocking: ISN's
    # publish-time validator (imas_standard_names, run by the ISNC gate) hard-
    # fails on them, so a name carrying one can never be published. Mirror that
    # authority here — an ISN-invalid name (e.g. a bare geometric quantity such
    # as 'coordinate' that lacks the required object/geometry/position
    # qualifier) must not reach validation_status='valid' and slip into the
    # export set. ISN emits an explicit 'ERROR -' severity marker for hard
    # failures; WARNING/INFO-level semantic hints stay advisory.
    for i in issues:
        if (i.startswith("[semantic]") or i.startswith("[structural]")) and (
            " ERROR - " in i
        ):
            return True

    # Grammar ambiguity is also critical — the name can't be reliably parsed
    if any("grammar:ambiguity" in i for i in issues):
        return True

    # Pydantic validation failure (model construction failed)
    pydantic = layer_summary.get("pydantic", {})
    if not pydantic.get("passed", True):
        return True

    # Grammar retry exhausted
    if any("audit:grammar_retry_exhausted" in i for i in issues):
        return True

    # Critical audit failures
    from imas_codex.standard_names.audits import has_critical_audit_failure

    if has_critical_audit_failure(issues):
        return True

    return False


def validate_name_candidate(entry: dict[str, Any]) -> tuple[list[str], dict, str]:
    """Run the full name-admission gate on a single candidate and classify it.

    This is the ONE gate every newly-minted StandardName passes before it can
    reach ``accepted`` — the same checks a pipeline-generated candidate clears:
    grammar round-trip, the ISN Pydantic/semantic/structural/canonical/
    description layers (:func:`_validate_via_isn`), and the post-generation
    audits (:func:`run_audits`).  :func:`_is_quarantined` then decides whether
    the accumulated issues are critical.

    Both the pool ``generate_name`` path (inline audit) and the legacy
    ``validate_worker`` classify names this way; the ``sn edit`` rename path
    calls this so an operator-supplied replacement name rides *exactly* the
    same gate rather than being stamped ``valid`` on a privileged path.

    Returns ``(issues, layer_summary, validation_status)`` where
    ``validation_status`` is ``"valid"`` or ``"quarantined"``.
    """
    name = entry.get("id", "")
    is_derived_parent = entry.get("origin") == "derived"
    derived_children = entry.get("children") or []
    from imas_codex.standard_names.audits import derived_parent_structural_check

    try:
        from imas_standard_names.grammar import (
            StandardName,
            compose_standard_name,
        )

        from imas_codex.standard_names.grammar_adapter import parse_canonical_name

        # Grammar round-trip validates parsability.
        parse_canonical_name(name)

        # Fields-consistency check (best-effort — never rejects here).
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

        # ISN three-layer validation (annotate).
        issues, layer_summary = _validate_via_isn(entry)

        # Post-generation audits.
        try:
            from imas_codex.standard_names.audits import run_audits

            source_path = None
            source_paths = entry.get("source_paths") or []
            if source_paths:
                source_path = strip_dd_prefix(source_paths[0])
            audit_issues = run_audits(
                candidate=entry,
                existing_sns_in_domain=None,
                source_path=source_path,
                source_cocos_type=entry.get("cocos_transformation_type"),
            )
            if audit_issues:
                issues.extend(audit_issues)
        except Exception:
            logger.debug("validate_name_candidate: audits failed for %r", name)

        if entry.get("_grammar_retry_exhausted"):
            issues.append("audit:grammar_retry_exhausted")

        # A derived parent that DOES round-trip still owes its structural
        # contract — an orphan parent that happens to parse generalises nothing.
        if is_derived_parent:
            issues.extend(derived_parent_structural_check(name, derived_children))

        status = "quarantined" if _is_quarantined(issues, layer_summary) else "valid"
        return issues, layer_summary, status
    except Exception as exc:
        exc_msg = str(exc).lower()
        issues = []
        if "component" in exc_msg and "coordinate" in exc_msg:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")
        if is_derived_parent:
            issues.extend(derived_parent_structural_check(name, derived_children))
        return issues, {}, "quarantined"


async def drain_validation_backlog(batch_size: int = 50) -> dict[str, int]:
    """Drain every unvalidated StandardName through the admission gate.

    Standalone claim→validate→mark loop over ``claim_names_for_validation``
    (any name with a description and ``validated_at`` null). Deterministic and
    LLM-free — safe to run as a maintenance pass after ``--revalidate`` clears
    stamps, without composing or reviewing anything. Returns
    ``{"validated": n, "quarantined": m}``.
    """
    from imas_codex.standard_names.graph_ops import (
        claim_names_for_validation,
        mark_names_validated,
        release_validation_claims,
    )

    totals = {"validated": 0, "quarantined": 0}
    while True:
        token, items = await asyncio.to_thread(claim_names_for_validation, batch_size)
        if not items:
            break
        try:
            results: list[dict[str, Any]] = []
            for entry in items:
                issues, layer_summary, status = validate_name_candidate(entry)
                if status == "quarantined":
                    totals["quarantined"] += 1
                results.append(
                    {
                        "id": entry.get("id", ""),
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            totals["validated"] += marked
        except Exception:
            logger.warning(
                "drain_validation_backlog: batch failed — releasing claims",
                exc_info=True,
            )
            await asyncio.to_thread(release_validation_claims, token)
            raise
    return totals


# Matches the backlog claim timeout in graph_ops so a stale scoped claim
# expires on the same schedule.
_SCOPED_VALIDATION_CLAIM_TIMEOUT = "PT300S"


def claim_ids_for_validation(
    ids: Sequence[str], limit: int
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim named StandardNames for a scoped ISN re-validation.

    The id-scoped counterpart of ``claim_names_for_validation``: only nodes in
    *ids* whose ``validated_at`` is null are claimed, so a campaign re-stamp
    never touches names outside its batch. Returns ``(token, items)`` where the
    token must be passed to ``mark_names_validated`` or
    ``release_validation_claims``.
    """
    import uuid

    from imas_codex.graph.client import GraphClient

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IN $ids
              AND sn.description IS NOT NULL
              AND sn.validated_at IS NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token, sn.updated_at = datetime()
            """,
            ids=list(ids),
            limit=limit,
            token=token,
            timeout=_SCOPED_VALIDATION_CLAIM_TIMEOUT,
        )
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


async def drain_validation_for_ids(
    ids: Sequence[str], *, batch_size: int = 50
) -> dict[str, Any]:
    """Re-run the deterministic ISN audit on an explicit id scope.

    Id-scoped counterpart of :func:`drain_validation_backlog`. Claims only the
    named nodes whose ``validated_at`` is null — a campaign clears the stamp on
    a batch before calling this — runs the shared admission gate
    (:func:`validate_name_candidate`), and re-stamps ``validation_issues`` /
    ``validation_status`` / ``validated_at``. LLM-free; a genuine defect (e.g. a
    unit inconsistency) re-quarantines instead of washing to 'valid'. Returns
    ``{"validated": n, "quarantined": m, "requarantined_ids": [...],
    "cleared_ids": [...]}``.
    """
    from imas_codex.standard_names.graph_ops import (
        mark_names_validated,
        release_validation_claims,
    )

    ids = list(ids)
    totals: dict[str, Any] = {
        "validated": 0,
        "quarantined": 0,
        "requarantined_ids": [],
        "cleared_ids": [],
    }
    if not ids:
        return totals
    while True:
        token, items = await asyncio.to_thread(
            claim_ids_for_validation, ids, batch_size
        )
        if not items:
            break
        try:
            results: list[dict[str, Any]] = []
            for entry in items:
                issues, layer_summary, status = validate_name_candidate(entry)
                sid = entry.get("id", "")
                if status == "quarantined":
                    totals["quarantined"] += 1
                    totals["requarantined_ids"].append(sid)
                else:
                    totals["cleared_ids"].append(sid)
                results.append(
                    {
                        "id": sid,
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            totals["validated"] += marked
        except Exception:
            logger.warning(
                "drain_validation_for_ids: batch failed — releasing claims",
                exc_info=True,
            )
            await asyncio.to_thread(release_validation_claims, token)
            raise
    return totals


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
                # Single shared admission gate (see validate_name_candidate) —
                # the same classification the pool generate_name path and the
                # sn edit rename path apply, so every name is judged identically.
                issues, layer_summary, status = validate_name_candidate(entry)
                state.audits_run += 1
                if any(i.startswith("audit:") or i.startswith("[") for i in issues):
                    state.audits_failed += 1
                if status == "quarantined":
                    batch_invalid += 1
                    wlog.debug("Validation quarantined %r: %s", name, issues[:2])
                results.append(
                    {
                        "id": name,
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )

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
                      AND sn.name_stage = 'drafted'
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
                              AND coalesce(sn.name_stage, 'drafted') = 'drafted'
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
# Pool-mode batch processors
# =============================================================================


async def compose_batch(
    batch: list[dict],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    regen: bool = False,
    compose_model: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Shared implementation for compose and regen pool batch processors.

    Takes a list of pre-claimed items (dicts with ``path``, ``description``,
    ``physics_domain``, etc.) and runs the compose pipeline:
    prompt → LLM → grammar validate → persist.

    **Batch-scope domain context:**
    Domain vocabulary is derived from the *batch items* rather than a
    run-scoped ``state.domain_filter``, so pooled-mode batches get domain
    context even without ``--physics-domain``.

    **Soft drain on stop_event:**
    Checks ``stop_event.is_set()`` before the LLM call and returns early
    (after persisting any in-flight results) if the event fires.

    Args:
        batch: Pre-claimed source items from the graph claim query.
        mgr: Shared :class:`BudgetManager` for cost tracking.
        stop_event: Cooperative shutdown signal.
        regen: If True, set ``regen_increment`` on persisted candidates.
        compose_model: Optional configured model override for this run. When
            absent, the production ``sn-compose`` seat is resolved normally.

    Returns:
        Count of successfully composed/regenerated candidates.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_compose_self_refine,
        get_model,
        get_reasoning_effort,
    )
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameComposeBatch

    if not batch:
        return 0

    # Claims are already verified by claim_generate_name_batch. Re-verify here
    # as well so direct processor callers cannot bypass fencing before a paid
    # compose call.
    from imas_codex.standard_names.graph_ops import _verify_source_claim_winners

    batch = await asyncio.to_thread(_verify_source_claim_winners, batch)
    if not batch:
        return 0

    # Claim read-back carries name-side edit steering as scalar fields. The
    # compose template renders those fields through review_feedback, so nest
    # them here for pooled generation. Exact-source compose_hint stays separate
    # because its lifecycle and authority boundary belong to the source node.
    for item in batch:
        if item.get("name_hint") and item.get("edit_reason"):
            item["review_feedback"] = {
                "previous_name": item.get("previous_name"),
                "reviewer_score": None,
                **(item.get("review_feedback") or {}),
                "name_hint": item["name_hint"],
                "edit_reason": item["edit_reason"],
                "edit_origin": item.get("edit_origin"),
            }

    model = compose_model or get_model("sn-compose")
    context = build_compose_context()

    # ── Batch-scope domain context ─────────────────────────────────────
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

    # ── Reviewer-theme extraction (batch-scoped) ──────────────────────
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    for dom in domains_in_batch:
        themes = await asyncio.to_thread(extract_reviewer_themes, dom)
        reviewer_themes.extend(themes)
    context["reviewer_themes"] = reviewer_themes[:12]

    # ── Scored-example injection ───────────────────────────────────────
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
    # leaf) to get semantically relevant nearby names. Deduplicate across
    # items, give each item a share, and cap at 30 total.
    group_key = batch[0].get("path", "").split("/")[0] if batch else ""
    nearby = await asyncio.to_thread(_collect_nearby_name_comparators, batch)
    nearby, _rerank_cost = await _reranked_nearby_comparators(
        mgr, batch, nearby, phase="regen" if regen else "generate_name"
    )

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

    # ── Pre-LLM stop check ────────────────────────────────────────────
    if stop_event.is_set():
        return 0

    # ── Reference exemplars — superseded by graph-backed
    # ``compose_scored_examples`` loaded above. The active prompt
    # template ``sn/generate_name_dd`` no longer renders a
    # ``reference_exemplars`` block, so
    # the per-batch ANN search has been retired here. The linear path
    # still uses ``_search_reference_exemplars`` for the names-only
    # template ``sn/generate_name_dd_names``.

    # ── Cluster-aware existing-names roster ───────────────────────────
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
                          AND coalesce(sn.name_stage, '') <> 'superseded'
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

    # ── Accepted family siblings into the prompt ───────────────────────
    # A family member that belongs to a detected vector/geometric family
    # already carries its accepted siblings (see
    # attach_family_accepted_siblings / _enrich_batch_items). Fold those
    # spellings into each item's EXISTING neighbour block — the template
    # renders hybrid_neighbours as reusable ``name:`` entries — and into the
    # existing-name roster, so the composer follows the family's established
    # spelling instead of inventing one. No parallel prompt section is added.
    for item in batch:
        sibs = item.get("family_accepted_siblings")
        if not sibs or not item.get("path"):
            continue
        for s in sibs:
            existing_names_set.add(s["name"])
        neighbours = item.setdefault("hybrid_neighbours", [])
        seen_tags = {n.get("tag") for n in neighbours}
        for s in sibs:
            tag = f"name:{s['name']}"
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            neighbours.append(
                {
                    "tag": tag,
                    "path": s.get("path"),
                    "ids": "",
                    "unit": s.get("unit", ""),
                    "physics_domain": s.get("physics_domain", ""),
                    "doc_short": (
                        f"{s.get('description') or ''} (family axis: "
                        f"{s.get('axis') or ''})"
                    ).strip(),
                    "cocos_label": "",
                    "lifecycle": "accepted",
                    "node_type": "",
                    "score": None,
                    "_family_sibling": True,
                }
            )

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
    # Seed the lease with the first compose attempt; each further attempt and
    # each per-candidate grammar retry binds its own exposure before it runs.
    _max_retries = _retry_attempts()
    # Token-reuse hits from the FINAL attempt (carried to the VocabGap stamp).
    _token_reuse_hits: dict[tuple[str, str], Any] = {}
    _editorial_guidance = ""
    _editorial_should_retry = False
    maximum_exposure = model_provider_exposure(
        model,
        messages,
        response_model=StandardNameComposeBatch,
        provider_attempts=1,
    )
    phase_tag = "regen" if regen else "generate_name"
    lease = mgr.reserve(maximum_exposure, phase=phase_tag)
    if lease is None:
        # Signal the refusal rather than returning a count, matching the
        # review path. Returning would leave the batch's sources claimed with
        # nothing to release them until the claim ages out, so every replica
        # would spin on empty claims while the pool showed pending work — a
        # stall, not a budget verdict. Raising routes through the pool's
        # release hook, which frees the sources for the next attempt, and each
        # refused reservation advances the saturation counter so the run exits
        # promptly on the reason that actually stopped it.
        from imas_codex.standard_names.budget import BudgetExceeded

        raise BudgetExceeded(
            f"{phase_tag} batch of {len(batch)} has no reservable provider "
            f"exposure (${maximum_exposure:.4f} required)"
        )

    try:
        # ── Bounded retry loop for failed compositions ─────────────────
        # Mirror of the linear path's retry: on grammar parse
        # failure across any candidate, re-enrich items with expanded DD
        # context and re-prompt.  Bounded by _retry_attempts().
        _total_compose_cost = 0.0
        _total_tokens_in = 0
        _total_tokens_out = 0
        _total_cache_read = 0
        _total_cache_creation = 0
        _collected_dd_gaps: list[Any] = []

        for _compose_attempt in range(_max_retries + 1):
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-compose"),
                    before_attempt=bind_attempt_exposure(
                        lease,
                        model,
                        messages,
                        response_model=StandardNameComposeBatch,
                    ),
                )
            except (ValueError, Exception) as compose_exc:
                charge_billable_exception(
                    lease,
                    compose_exc,
                    model=model,
                    batch_id=group_key,
                    phase=phase_tag,
                )
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
            _sanitize_compose_result_sources(
                result,
                _claimed_source_ids(batch),
                phase=phase_tag,
            )
            _collected_dd_gaps.extend(result.dd_gaps)

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

            # Quick grammar round-trip check on all candidates, with the
            # per-failure repair advice the retry needs to converge.
            _grammar_failures, _grammar_advice = _grammar_round_trip_failures(
                result.candidates
            )

            # Component-token reuse check (free local embeddings): flag any
            # candidate-new token semantically near a registered same-segment
            # token so the agent gets a chained chance to reuse-or-confirm.
            _token_reuse_hits = await asyncio.to_thread(
                _compute_token_reuse_hits, result.vocab_gaps
            )
            _editorial_guidance, _editorial_should_retry = await asyncio.to_thread(
                _active_editorial_gap_guidance, result.vocab_gaps
            )

            if (
                not _grammar_failures
                and not _token_reuse_hits
                and not _editorial_should_retry
            ) or (_compose_attempt >= _max_retries):
                break

            # Re-enrich items with expanded hybrid search for retry
            logger.info(
                "Pool %s: composition retry %d/%d: %d grammar failures (%s), "
                "%d token-reuse hits, editorial_retry=%s — "
                "re-composing with expanded DD context",
                phase_tag,
                _compose_attempt + 1,
                _max_retries,
                len(_grammar_failures),
                ", ".join(_grammar_failures[:3]),
                len(_token_reuse_hits),
                _editorial_should_retry,
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

            # Only re-enrich when grammar failed; a token-reuse-only retry needs
            # no expanded neighbour context (the agent just reuses-or-confirms).
            if _grammar_failures:
                await asyncio.to_thread(_re_enrich_expanded)

            _reason_parts = []
            if _grammar_failures:
                _reason_parts.append(
                    _build_grammar_retry_reason(_grammar_failures, _grammar_advice)
                )
            if _token_reuse_hits:
                _reason_parts.append(_build_token_reuse_retry_reason(_token_reuse_hits))
            if _editorial_guidance:
                _reason_parts.append(_editorial_guidance)
            _retry_reason = "\n\n".join(_reason_parts)
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

        # The paid response can return after the stale-claim window. Fence it
        # again before source status, attachment, or candidate graph writes.
        current_claims = await asyncio.to_thread(
            _verify_source_claim_winners,
            batch,
            settle_seconds=0,
        )
        batch = current_claims
        _sanitize_compose_result_sources(
            result,
            _claimed_source_ids(batch),
            phase=phase_tag,
        )

        # Use accumulated totals for per-candidate cost attribution
        cost = _total_compose_cost
        tokens_in = _total_tokens_in
        tokens_out = _total_tokens_out
        tokens_cache_r = _total_cache_read
        tokens_cache_w = _total_cache_creation

        # ── Build candidates ───────────────────────────────────────────
        candidates: list[dict] = []
        disposition_winner_ids: set[str] = set()
        wlog = logger  # plain logger; pool path has no WorkerLogAdapter
        source_kind = "dd"  # pool-mode generate-name is DD-only today
        # Deterministic provenance collapse: one canonical name per base quantity
        # so estimator facets (measured/reconstructed/reference) never drift apart.
        _prov_canon = provenance_canonical_names(result.candidates, log=wlog)
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
                        skip_winners = _persist_claimed_skips(
                            [c.source_id],
                            batch,
                            source_type=source_kind,
                            reason=_skip_reason,
                            detail=str(raw_unit),
                        )
                        disposition_winner_ids.update(skip_winners)
                    except Exception as exc:
                        wlog.debug(
                            "Failed to mark pool source %s skipped: %s",
                            c.source_id,
                            exc,
                        )
                continue
            unit = raw_unit

            # A name's domain is its stored Jev decision, set by the one writer
            # after the batch is persisted; the DD source's own domain is not
            # carried onto the name here.
            physics_domain = None
            source_domains: list[str] = []

            cocos_type = source_item.get("cocos_label") if source_item else None

            # Shape-parameter leaves (triangularity/elongation/squareness) are
            # only meaningful *of* a surface; force the surface locus from the
            # source DD path when the composer left it bare so the leaf is
            # surface-explicit and the boundary/profile siblings de-conflate.
            _inject_shape_parameter_surface(c, c.source_id, wlog)

            # Pre-validation gate — reject malformed LLM output
            # before MERGE.  Mark the source as 'failed' so it is not
            # re-claimed forever.
            name_id = normalize_spelling(c.compose_name())
            # Provenance collapse: estimator facets of one quantity share the
            # canonical name (MERGE identity) regardless of per-facet drift.
            _pterm, _pbase = detect_value_provenance(c.source_id or "")
            if _pterm and _pbase in _prov_canon:
                name_id = _prov_canon[_pbase]
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
                        failed = _mark_source_prevalidation_failed(
                            c.source_id,
                            source_type=source_kind,
                            candidate_name=name_id,
                            reason=_reject_reason or "malformed",
                            claim_token=(source_item or {}).get("claim_token"),
                            claim_seq=(source_item or {}).get("claim_seq"),
                        )
                        if failed:
                            disposition_winner_ids.add(f"{source_kind}:{c.source_id}")
                    except Exception:
                        wlog.debug("Failed to mark source %s as failed", c.source_id)
                continue

            # Non-nameable coordinate/infrastructure gate: a bare timestamp,
            # latency, counter, or metadata token is not a physics observable.
            # Composing it produces a doomed bare name that fails the semantic
            # gate and burns review + every refine rotation before exhausting.
            # Route the source to ``skipped`` (terminal) so it never re-claims.
            _non_nameable, _skip_reason = is_non_nameable_coordinate(name_id)
            if _non_nameable:
                wlog.warning(
                    "Pool %s: non-nameable skip %r (%s)",
                    phase_tag,
                    name_id[:80],
                    _skip_reason,
                )
                if c.source_id:
                    try:
                        skip_winners = _persist_claimed_skips(
                            [c.source_id],
                            batch,
                            source_type=source_kind,
                            reason=_skip_reason or "non_nameable_coordinate",
                            detail=f"bare non-nameable token: {name_id}",
                        )
                        disposition_winner_ids.update(skip_winners)
                    except Exception:
                        wlog.debug("Failed to mark source %s skipped", c.source_id)
                continue

            # Deterministic source<->name consistency gate (tense + state
            # resolution). The LLM sometimes emits a species-level sibling
            # for a state-resolved source (e.g. thermal_neutral_density
            # sourced from .../neutral/state/density_thermal) -- drop the
            # pairing so the source retries rather than persisting a
            # mis-attributed name. Same predicate the auto-attach path uses.
            if c.source_id:
                _paths = [
                    c.source_id,
                    *(p for p in (c.dd_paths or []) if p != c.source_id),
                ]
                _ok, _why = True, ""
                for _p in _paths:
                    _ok, _why = _is_attachment_consistent(
                        _p, name_id, existing_sources=[q for q in _paths if q != _p]
                    )
                    if not _ok:
                        break
                if not _ok:
                    wlog.warning(
                        "Source-name consistency reject: %r (%s)",
                        name_id[:80],
                        _why,
                    )
                    continue

            grammar_failed = False
            try:
                from imas_codex.standard_names.grammar_adapter import (
                    parse_canonical_name,
                )

                parse_canonical_name(name_id)
            except Exception as gram_exc:
                grammar_failed = True
                wlog.debug(
                    "Pool %s: grammar parse failed for %r — attempting retry",
                    phase_tag,
                    name_id,
                )

                # Single-shot grammar-failure retry.
                try:
                    retry_name, _r_cost, _r_ti, _r_to = await _grammar_retry(
                        name_id,
                        str(gram_exc),
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
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
                            parse_canonical_name(retry_name)
                            name_id = retry_name
                            grammar_failed = False
                            wlog.info(
                                "Pool %s: grammar retry succeeded %r → %r",
                                phase_tag,
                                c.compose_name(),
                                name_id,
                            )
                        except Exception:
                            wlog.debug(
                                "Pool %s: retry result still un-parseable",
                                phase_tag,
                            )
                except Exception as retry_exc:
                    charge_billable_exception(
                        lease,
                        retry_exc,
                        model=model,
                        sn_ids=(name_id,),
                        batch_id=f"{group_key}-grammar-retry",
                        phase=phase_tag,
                    )
                    wlog.debug(
                        "Pool %s: grammar retry failed for %r",
                        phase_tag,
                        name_id,
                    )

            cand_description = normalize_description_text(c.description or "")

            # ── Free local self-refine pass (default off) ─────────────────
            # After compose + grammar normalization and BEFORE persist (so
            # before the paid review quorum), let the LOCAL compose model
            # critique its own name + description and emit an improved
            # candidate. Improve-or-no-op; the rewrite is re-validated and
            # discarded if it fails grammar. OFF = byte-identical to the path
            # without this block.
            #
            # The pass discards its LLMResult, so it can only run where the
            # call is free: on a paid route its spend would leave no cost
            # record and no reservation. Refuse rather than spend silently.
            if not grammar_failed and get_compose_self_refine():
                from imas_codex.discovery.base.llm import _is_local_model

                if not _is_local_model(model):
                    raise RuntimeError(
                        "compose self-refine records no cost and reserves no "
                        f"budget, so it cannot run on the metered route {model}"
                    )
                try:
                    _ref_name, _ref_desc = await _self_refine_candidate(
                        name_id,
                        cand_description,
                        c.segments.model_dump() if hasattr(c, "segments") else None,
                        source_item,
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
                    )
                    if _ref_name != name_id:
                        wlog.info(
                            "Pool %s: self-refine %r → %r",
                            phase_tag,
                            name_id,
                            _ref_name,
                        )
                    name_id = _ref_name
                    cand_description = _ref_desc
                except Exception:
                    wlog.debug(
                        "Pool %s: self-refine failed for %r — keeping original",
                        phase_tag,
                        name_id,
                    )

            cand = {
                "id": name_id,
                "source_types": [source_kind],
                "source_id": c.source_id,
                "description": cand_description,
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
                "source_claim_token": (source_item or {}).get("claim_token"),
                "source_claim_seq": (source_item or {}).get("claim_seq"),
                **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
            }

            if regen:
                cand["regen_increment"] = True

            candidates.append(cand)

            # ── Deterministic error-sibling minting ──────────────────
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

        # ── Per-candidate cost / token attribution ────────────────────
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

        # ── Inline audits — populate validation_status ────────────────
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

        # Re-check exact token+sequence ownership immediately before any graph
        # mutation. A worker that lost its source claim while the LLM was
        # running must not persist any part of that delayed response.
        current_claims = await asyncio.to_thread(
            _verify_source_claim_winners,
            batch,
            settle_seconds=0,
        )
        _sanitize_compose_result_sources(
            result,
            _claimed_source_ids(current_claims),
            phase=phase_tag,
        )
        current_source_ids = _claimed_source_ids(current_claims)
        candidates = _filter_persist_candidates_to_claimed_sources(
            candidates,
            current_source_ids,
            phase=phase_tag,
        )

        # ── Vocab gaps — persist + clear sources ──────────────────────
        if result.vocab_gaps:
            gap_dicts = [
                {
                    "source_id": vg.source_id,
                    "segment": vg.segment,
                    "token": vg.token,
                    "reason": vg.reason,
                }
                for vg in result.vocab_gaps
            ]
            # Stamp the compose-time token-reuse adjudication: gaps re-emitted
            # after being shown a near-synonym registered token are
            # distinct_confirmed; the rest are unchecked.
            _stamp_dedup_decision(gap_dicts, _token_reuse_hits)
            gap_dicts, gap_winner_ids = await asyncio.to_thread(
                _consume_claimed_vocab_gaps,
                gap_dicts,
                batch,
                source_type=source_kind,
                return_winner_ids=True,
            )
            disposition_winner_ids.update(gap_winner_ids)
            wlog.debug("Pool %s: persisted %d vocab gaps", phase_tag, len(gap_dicts))
        # ── Attachments — write edges + clear source claims ───────────
        if result.attachments:
            try:
                attach_counts = await asyncio.to_thread(
                    _process_attachments_core,
                    result.attachments,
                    wlog,
                    source_claims=batch,
                    return_winner_ids=True,
                )
                disposition_winner_ids.update(attach_counts["winner_ids"])
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
        # ── Skipped sources — clear claims so they don't loop ─────────
        if result.skipped:
            try:
                skipped_winner_ids = await asyncio.to_thread(
                    _persist_claimed_skips,
                    list(result.skipped),
                    batch,
                    source_type=source_kind,
                    reason="compose_model_skipped",
                )
                disposition_winner_ids.update(skipped_winner_ids)
                wlog.debug(
                    "Pool %s: marked %d sources as skipped",
                    phase_tag,
                    len(skipped_winner_ids),
                )
            except Exception:
                wlog.debug(
                    "Pool %s: skipped source-status update failed",
                    phase_tag,
                    exc_info=True,
                )

        # ── Persist ────────────────────────────────────────────────────
        if candidates:
            persisted = await asyncio.to_thread(
                _persist_description_checked_candidates,
                candidates,
                source_type=source_kind,
                phase=phase_tag,
                compose_model=model,
                dd_version=batch[0].get("dd_version"),
                cocos_version=batch[0].get("cocos_version"),
                run_id=mgr.run_id,
            )
            # Only the exact ID-returning persistence contract proves which
            # source claims won the CAS. A legacy count cannot authorize a
            # source-specific evidence mutation.
            winner_ids = persisted if isinstance(persisted, list) else []
            winner_id_set = set(winner_ids)
            disposition_winner_ids.update(winner_ids)
            if isinstance(persisted, list):
                candidates = [
                    candidate
                    for candidate in candidates
                    if f"{source_kind}:{candidate.get('source_id')}" in winner_id_set
                ]
                written = len(candidates)
            else:
                written = int(persisted)
            logger.debug("Pool %s: persisted %d candidates", phase_tag, written)

            # A persisted name's domain is its stored Jev decision, set by the
            # one writer rather than carried from the source item.
            if isinstance(persisted, list) and candidates:
                await _write_composed_name_domains(
                    candidates,
                    {str(c.get("source_id")) for c in candidates},
                )

            # Derive auto-gaps only from finalized source winners. A losing
            # source cannot authorize a gap mutation.
            if candidates:
                try:
                    auto_gaps = _auto_detect_physical_base_gaps(candidates)
                    if auto_gaps:
                        from imas_codex.standard_names.graph_ops import (
                            write_vocab_gaps,
                        )

                        await asyncio.to_thread(
                            write_vocab_gaps,
                            auto_gaps,
                            source_kind,
                            skip_segment_filter=True,
                        )
                except Exception:
                    wlog.debug(
                        "Pool %s: physical_base gap detection failed",
                        phase_tag,
                        exc_info=True,
                    )

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

        committed_source_ids = {
            winner.split(":", 1)[1]
            for winner in disposition_winner_ids
            if isinstance(winner, str) and winner.startswith(f"{source_kind}:")
        }
        source_versions = {
            source_id: item.get("dd_version")
            for item in batch
            if (source_id := (item.get("path") or item.get("source_id")))
            in committed_source_ids
        }
        await asyncio.to_thread(
            _persist_dd_gap_evidence,
            _collected_dd_gaps,
            committed_source_ids,
            phase=phase_tag,
            reporter="compose",
            source_versions=source_versions,
        )

        return written if candidates else 0

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
    compose_model: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Pool-mode generate-name batch processor.

    Takes pre-claimed source items (dicts from the graph claim query),
    generates standard names via LLM, and persists results.

    Returns count of items successfully processed.
    """
    return await compose_batch(
        batch,
        mgr,
        stop_event,
        regen=False,
        compose_model=compose_model,
        on_event=on_event,
    )


# Substrings that identify a deterministic grammar / vocab / schema failure
# in an LLM-proposed refined name.  These are NORMAL failed-refine outcomes
# (the model proposed an ungrammatical name), not crashes — the item is
# routed to the exhaust path rather than re-claimed and re-charged forever.
_GRAMMAR_FAILURE_MARKERS: tuple[str, ...] = (
    "not a registered",  # unregistered grammar token (base/qualifier/axis)
    "kind must be one of",  # legacy schema enum violation message
    "input should be",  # pydantic Literal enum violation (deterministic)
    "self-referential refined_from",  # refine produced an identical name
    "validation error for refinedname",  # pydantic RefinedName validation
    # ISN StandardName grammar composition raised inside the refine compose
    # step (compose_standard_name → StandardName.model_validate): the model
    # proposed a name whose tokens/structure are not grammatical (invalid
    # operator token, unregistered base, malformed transformation). Same
    # output every cycle, so terminal — not a crash.
    "validation error for standardname",
    "invalid operator token",  # deterministic fused-operator/coordinate misuse
)


class RefineCandidateValidationError(ValueError):
    """A refined candidate failed deterministic strict grammar validation."""


class RefineScopeContinuityError(RuntimeError):
    """A refine batch cannot prove one coherent successor scope."""


def _refine_batch_scope_id(
    batch: list[dict[str, Any]], scope_run_id: str | None
) -> str | None:
    """Validate and return the invocation's explicit focused scope, if any.

    ``BudgetManager.run_id`` identifies the audit run that pays for and counts
    the work. It is not the graph scope selected by ``--focus``. Only an
    explicit invocation scope activates focused continuity; durable run IDs on
    rows in an ordinary unscoped backlog are historical provenance and do not
    classify the invocation.

    In focused mode, the atomic claim read-back must prove that every
    predecessor carries the exact requested scope. Missing, blank, or
    mismatched claim metadata refuses before any model call or persistence.
    """
    if scope_run_id is None:
        return None
    if (
        not isinstance(scope_run_id, str)
        or not scope_run_id.strip()
        or scope_run_id != scope_run_id.strip()
    ):
        raise RefineScopeContinuityError(
            "refine invocation carries a blank or non-canonical scope identity"
        )

    for item in batch:
        claimed_scope = item.get("scope_run_id")
        if claimed_scope != scope_run_id:
            raise RefineScopeContinuityError(
                f"refine claim for {item.get('id')!r} carries scope "
                f"{claimed_scope!r}, expected {scope_run_id!r}"
            )
    return scope_run_id


def _compose_refined_candidate_name(result_obj: Any) -> str:
    """Compose one refined candidate name and type deterministic failures."""
    from imas_standard_names import ParseError
    from pydantic import ValidationError

    try:
        return result_obj.name
    except (ParseError, ValidationError, ValueError) as exc:
        raise RefineCandidateValidationError(
            f"refined candidate failed strict grammar validation: {exc}"
        ) from exc


def _is_refine_grammar_failure(exc: BaseException) -> bool:
    """Return True if *exc* is a deterministic grammar/validation refine failure.

    Such failures (e.g. the LLM proposing an unregistered qualifier token) are
    a normal failed-refine outcome, not a crash: they must be routed to the
    exhaust path so the item stops re-claiming and re-burning paid budget.
    """
    if isinstance(exc, RefineCandidateValidationError):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _GRAMMAR_FAILURE_MARKERS)


#: Markers within the deterministic set that identify a MISSING TOKEN rather
#: than a malformed composition. The distinction is the operator's next move:
#: a missing token is admitted on the ISN side, while a malformed composition
#: is a prompt or model problem in this repository.
_VOCABULARY_GAP_MARKERS: tuple[str, ...] = ("not a registered",)


def _refine_failure_stop_reason(
    exc: BaseException, *, terminal: bool
) -> RefineStopReason:
    """Classify a refine failure into the reason recorded on the name."""
    if not terminal:
        return RefineStopReason.transient_failure
    msg = str(exc).lower()
    if any(marker in msg for marker in _VOCABULARY_GAP_MARKERS):
        return RefineStopReason.vocabulary_gap
    return RefineStopReason.grammar_invalid


def _is_refine_docs_failure(exc: BaseException) -> bool:
    """Return True if *exc* is a deterministic ``RefinedDocs`` validation failure.

    A docs-refine that produces docs violating the ``RefinedDocs`` schema
    (e.g. consistently over/under the length bounds) is deterministic for a
    given item.  Reverting it to 'reviewed' would re-claim and re-burn paid
    budget forever, so route it to the docs-exhaust path instead.
    """
    return "validation error for refineddocs" in str(exc).lower()


#: Return the rotation an unchanged resubmission did not spend.
#:
#: The refine pool charges a rotation when it CLAIMS a name, before any model
#: call, so the charge buys an improvement attempt. A pinned rename
#: (``edit_mode = 'rename'``) is never rewritten — the pool routes the same
#: name back to a fresh review quorum — so that charge buys nothing. Left
#: alone it walks ``refine_attempts`` to the rotation cap, and the cap is also
#: the eligibility gate, so a name whose only consumer is the re-review loop
#: stops being claimed at 3 of 3. Fenced on the same claim token and
#: ``'refining'`` stage the resubmission checks, so the rotation can only be
#: returned while the claim that charged it is still live.
_RETURN_UNSPENT_REFINE_ATTEMPT = """
MATCH (sn:StandardName {id: $sn_id})
WHERE sn.claim_token = $token AND sn.name_stage = 'refining'
SET sn.refine_attempts = CASE WHEN coalesce(sn.refine_attempts, 0) > 0
                              THEN sn.refine_attempts - 1
                              ELSE sn.refine_attempts END
RETURN coalesce(sn.refine_attempts, 0) AS refine_attempts
"""


def _return_unspent_refine_attempt(gc: Any, *, sn_id: str, token: str) -> int | None:
    """Give back the rotation an unchanged resubmission did not spend.

    Returns the counter the name is left holding, or ``None`` when the fence
    did not match — a concurrent sweep or another worker already moved the
    row, in which case this pool no longer owns the charge and must not
    rewrite the counter under its own token.
    """
    try:
        rows = gc.query(
            _RETURN_UNSPENT_REFINE_ATTEMPT,
            sn_id=sn_id,
            token=token,
        )
    except Exception:
        logger.debug(
            "refine_name: could not return the unspent rotation for %s",
            sn_id,
            exc_info=True,
        )
    else:
        try:
            if rows:
                return int(rows[0].get("refine_attempts") or 0)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            # A best-effort return must never cost the item its resubmission.
            logger.debug("refine_name: unreadable rotation counter for %s", sn_id)
    return None


async def process_refine_name_batch(
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    scope_run_id: str | None = None,
) -> int:
    """Process a batch of StandardNames through successor-chain refinement.

    For each item in the batch:
    1. Walk the REFINED_FROM chain via ``chain_history`` (already enriched).
    2. Escalate when the claim charged the last rotation the cap allows.
    3. Optionally fan out targeted DD context (gated).
    4. Call LLM to produce a refined name (``RefinedName`` response model).
    5. Run the deterministic name-key dup guard before persisting.
    6. Persist via ``persist_refined_name`` (new node + edge migration).
    7. Close a failed attempt with :func:`stop_refine_name_attempt`, which
       records why it failed and parks the name when the reason is decided or
       the budget is spent.

    Returns count of items successfully processed.
    """
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import (
        LLMStructuredCallError,
        ProviderBudgetExhausted,
        acall_llm_structured,
    )
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
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
    from imas_codex.standard_names.fanout.dispatcher import proposer_exposure
    from imas_codex.standard_names.graph_ops import (
        RefinedNamePersistenceRefusal,
        RefinedNamePersistenceRefusalReason,
        bump_sn_run_counter,
        persist_refined_name,
        resubmit_pinned_rename_for_review,
        stop_refine_name_attempt,
    )
    from imas_codex.standard_names.models import RefinedName

    rotation_cap = DEFAULT_REFINE_ROTATIONS
    focus_scope_id = _refine_batch_scope_id(batch, scope_run_id)
    processed = 0

    async def _stop_attempt(
        item: dict[str, Any],
        reason: RefineStopReason,
        *,
        detail: str = "",
        collision_name: str | None = None,
    ) -> str:
        """Close this attempt on the graph, returning the stage written."""
        try:
            return await _asyncio.to_thread(
                stop_refine_name_attempt,
                sn_id=item["id"],
                token=item.get("claim_token") or "",
                reason=reason.value,
                detail=detail,
                collision_name=collision_name,
                rotation_cap=rotation_cap,
            )
        except Exception:
            logger.debug(
                "refine_name: could not record stop for %s (reason=%s)",
                item["id"],
                reason.value,
                exc_info=True,
            )
            return ""

    # ── GraphClient lifecycle ────────────────────────────────────────
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

            # ── Pinned-rename guard (never rewrite an operator's name) ──
            # A rename edit carries a fixed, operator-chosen name string.
            # Rewriting it is meaningless and destructive: re-emitting the
            # identical name trips the self-referential-refine guard and
            # decomposing a lexicalised base trips grammar validation — both
            # wrongly exhaust a correct name. Resubmit it to a fresh review
            # quorum instead (bounded), never spend an LLM rewrite on it.
            if (item.get("edit_mode") or "") == "rename":
                token = item.get("claim_token") or ""
                # The claim charged a rotation on the way in. Detect the
                # unchanged case — this step resubmits the name the item
                # already carries: no candidate is produced, so the rotation
                # has bought nothing and must not be spent. Without this a
                # pinned name walks to the cap on re-review alone, and the cap
                # is the eligibility gate, so it stops being claimed.
                attempts_left = _return_unspent_refine_attempt(
                    gc, sn_id=sn_id, token=token
                )
                try:
                    outcome = await _asyncio.to_thread(
                        resubmit_pinned_rename_for_review,
                        sn_id=sn_id,
                        token=token,
                        rotation_cap=rotation_cap,
                    )
                except Exception:
                    logger.debug(
                        "refine_name: pinned-rename resubmit failed for %s", sn_id
                    )
                    outcome = ""
                logger.info(
                    "refine_name: pinned rename %s not rewritten — %s "
                    "(rotation returned, %s remaining)",
                    sn_id,
                    outcome or "no-op",
                    attempts_left if attempts_left is not None else "unknown",
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": sn_id,
                            "old_name": sn_id,
                            "outcome": f"pinned_rename_{outcome or 'noop'}",
                            "model": "none",
                            "cost": 0.0,
                            # The rotation this resubmission did not spend.
                            # A caller that sums ``refine_attempts`` from
                            # events must not count it.
                            "refine_attempts": attempts_left,
                        }
                    )
                continue

            # ── Escalation decision ───────────────────────────────────
            # Keyed on the rotations SPENT, which the claim has just charged,
            # so the last one runs on the escalation seat even when nothing
            # this name proposed has ever persisted — the exact case the
            # lineage-depth test never reached.
            attempt = int(item.get("refine_attempts") or chain_length + 1)
            escalate = attempt >= rotation_cap
            logger.info(
                "refine_name: %s attempt %d/%d%s",
                sn_id,
                attempt,
                rotation_cap,
                (
                    f" (previous stop: {item['refine_stop_reason']})"
                    if item.get("refine_stop_reason")
                    else ""
                ),
            )
            if escalate:
                model = DEFAULT_ESCALATION_MODEL
            else:
                # Dedicated refine seat: a flash-lite tier lifted critiqued
                # names at ~5% against ~42% for the compose tier, so refine
                # owns its own model choice rather than sharing the compose
                # seat's.
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

            # Merge the cached compose context so the system prompt's
            # ``_grammar_reference.md`` include renders the closed-vocabulary
            # token map (``closed_vocab_full`` etc.). Without it the refiner
            # rewrites names with no vocabulary reference — the empty-grammar-
            # block silent gap. ``build_compose_context`` is ``_CONTEXT_CACHE``-
            # backed, so this is a dict lookup after the first call.
            from imas_codex.standard_names.context import build_compose_context

            prompt_context: dict[str, Any] = {
                **build_compose_context(),
                "item": item,
                "chain_history": chain_history,
                "chain_length": chain_length,
                "hybrid_neighbours": [],
                "fanout_evidence": "",
                "compose_scored_examples": compose_scored_examples,
                "vocab_gap_detail": vocab_gap_detail,
                "validation_issues": validation_issues,
            }
            from imas_codex.standard_names.review.cascade import (
                failed_check_instructions,
            )

            failed_checks = failed_check_instructions(item, "name")
            if failed_checks:
                prompt_context["failed_checks"] = failed_checks

            # Attempt hybrid neighbour search (best-effort).  Uses the
            # cycle-scoped ``gc`` — no fresh client.
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

            # ── Fan-out trigger gate ─────────────────────────────────
            # A routed refine fires only when one of its named failures is
            # a retrieval check at or above the calibrated threshold.
            check_trigger: dict[str, Any] = {}
            if failed_checks:
                from imas_codex.standard_names.review.checks import load_review_checks

                check_trigger = {
                    "checks": [
                        check
                        for check in load_review_checks()[0]
                        if check.axis == "name"
                    ],
                    "check_results": {
                        check["id"]: {"answered": True, "failure_probability": 1.0}
                        for check in failed_checks
                    },
                    "check_threshold": fanout_settings.refine_trigger_check_threshold,
                }
            fanout_eligible, reviewer_excerpt = should_trigger_fanout(**check_trigger)

            # ── Optional fan-out ─────────────────────────────────────
            lease = None
            fanout_evidence = ""
            fanout_run_id: str | None = None
            fanout_arm: str | None = None
            if (
                fanout_eligible
                and fanout_settings.enabled
                and fanout_settings.sites.get("refine_name", False)
            ):
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
                    fanout_maximum = proposer_exposure(
                        candidate=candidate_ctx,
                        reviewer_excerpt=reviewer_excerpt,
                        scope=scope,
                        settings=fanout_settings,
                    )
                    lease = mgr.reserve(fanout_maximum, phase="refine_name")
                    if lease is None:
                        continue
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
                finally:
                    if lease is not None:
                        lease.release_unused()
                        lease = None
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
                response_model=RefinedName,
                provider_attempts=1,
            )
            lease = mgr.reserve(maximum_exposure, phase="refine_name")
            if lease is None:
                continue
            original_reservation = lease.reserved
            cost = 0.0
            llm_tokens_in = 0
            llm_tokens_out = 0
            llm_tokens_cached_read = 0
            llm_tokens_cached_write = 0
            llm_charge_recorded = False
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=_messages,
                    response_model=RefinedName,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-refine"),
                    before_attempt=bind_attempt_exposure(
                        lease, model, _messages, response_model=RefinedName
                    ),
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
                # ``Fanout`` ↔ ``LLMCost`` join works for both arms.
                if lease:
                    _event = LLMCostEvent(
                        model=model,
                        tokens_in=llm_tokens_in,
                        tokens_out=llm_tokens_out,
                        tokens_cached_read=llm_tokens_cached_read,
                        tokens_cached_write=llm_tokens_cached_write,
                        sn_ids=(sn_id,),
                        phase=(
                            "refine_name+fanout" if fanout_run_id else "refine_name"
                        ),
                        service="standard-names",
                        batch_id=fanout_run_id,
                    )
                    lease.charge_event(cost, _event)
                    llm_charge_recorded = True

                # Strict composition can fail even after the response model
                # validates.  Cost is already recorded against the stable
                # predecessor before this derived accessor is evaluated.
                candidate_name = _compose_refined_candidate_name(result_obj)

                # ── Deterministic name-key dup guard ──────────────
                # Name-key lookup AFTER the final retry's
                # candidate, BEFORE persisting.  On hit we drop the
                # candidate and emit a ``dup_prevented`` log line.
                # Excludes ``old_name`` so the chain's predecessor is
                # never treated as a self-collision.
                dup_id = None
                try:
                    dup_id = find_name_key_duplicate(
                        gc,
                        candidate_name,
                        exclude=sn_id,
                    )
                except Exception:
                    logger.debug(
                        "dup guard failed for %s — proceeding with persist",
                        sn_id,
                    )
                if dup_id:
                    # The identity the refiner is reaching for is already
                    # held. Refinement may not take an occupied identity —
                    # folding into it migrates sources and belongs to
                    # `sn edit` — and the same prompt re-proposes the same
                    # identity on every cycle, across vendors. Park the name
                    # with the collision recorded rather than spending the
                    # rotations that remain on a decided outcome.
                    stage = await _stop_attempt(
                        item,
                        RefineStopReason.successor_collision,
                        detail=f"proposed {candidate_name} which already exists",
                        collision_name=dup_id,
                    )
                    collision_event = {
                        "pool": "refine_name",
                        "name": sn_id,
                        "old_name": sn_id,
                        "proposed_name": candidate_name,
                        "existing_name": dup_id,
                        "duplicate_of": dup_id,
                        "refusal_reason": "successor_collision",
                        "attempt": attempt,
                        "rotation_cap": rotation_cap,
                        "resulting_stage": stage,
                        "outcome": "dup_prevented",
                        "model": model,
                        "cost": cost,
                    }
                    logger.warning(
                        "refine_name_persistence_refused %s",
                        json.dumps(collision_event, sort_keys=True),
                    )
                    if on_event is not None:
                        on_event(collision_event)
                    continue

                # ── Persist ───────────────────────────────────────────
                await _asyncio.to_thread(
                    persist_refined_name,
                    old_name=sn_id,
                    new_name=candidate_name,
                    description=normalize_description_text(result_obj.description),
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
                    # A focused successor inherits the predecessor's durable
                    # scope inside persist_refined_name's transaction when
                    # run_id is null. The manager id remains reserved for the
                    # SNRun that paid for this invocation. Unscoped runs keep
                    # their established audit-id stamping behavior.
                    run_id=mgr.run_id if focus_scope_id is None else None,
                    expected_claim_token=item.get("claim_token"),
                )
                if focus_scope_id is not None:
                    bump_sn_run_counter(mgr.run_id, "names_regenerated")
                processed += 1
                logger.info(
                    "refine_name: %s → %s (chain_length=%d, model=%s)",
                    sn_id,
                    candidate_name,
                    chain_length + 1,
                    model,
                )

                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": candidate_name,
                            "old_name": sn_id,
                            "new_name": candidate_name,
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
                if isinstance(exc, LLMStructuredCallError | ProviderBudgetExhausted):
                    cost = exc.cost
                    llm_tokens_in = exc.input_tokens
                    llm_tokens_out = exc.output_tokens
                    llm_tokens_cached_read = exc.cache_read_tokens
                    llm_tokens_cached_write = exc.cache_creation_tokens
                    if not llm_charge_recorded:
                        llm_charge_recorded = charge_billable_exception(
                            lease,
                            exc,
                            model=model,
                            sn_ids=(sn_id,),
                            phase=(
                                "refine_name+fanout" if fanout_run_id else "refine_name"
                            ),
                            batch_id=fanout_run_id,
                        )
                if isinstance(exc, ProviderBudgetExhausted):
                    raise
                _exc_str = str(exc)
                if isinstance(exc, RefinedNamePersistenceRefusal):
                    claim_lost = exc.reason in {
                        RefinedNamePersistenceRefusalReason.CLAIM_LOST,
                        RefinedNamePersistenceRefusalReason.PREDECESSOR_STAGE,
                        RefinedNamePersistenceRefusalReason.PREDECESSOR_MISSING,
                    }
                    # A successor the persistence fence will never provision
                    # under that name is as decided as a name-key collision;
                    # a lost claim or a contended drain scope is not, so those
                    # keep the remaining rotations.
                    lifecycle_collision = (
                        exc.reason
                        is RefinedNamePersistenceRefusalReason.SUCCESSOR_LIFECYCLE
                    )
                    stage = await _stop_attempt(
                        item,
                        (
                            RefineStopReason.successor_lifecycle_collision
                            if lifecycle_collision
                            else RefineStopReason.transient_failure
                        ),
                        detail=f"{exc.reason.value}: proposed {exc.proposed_name}",
                        collision_name=exc.existing_name,
                    )
                    refusal_event = {
                        "pool": "refine_name",
                        "name": sn_id,
                        "old_name": sn_id,
                        "proposed_name": exc.proposed_name,
                        "existing_name": exc.existing_name,
                        "refusal_reason": exc.reason.value,
                        "attempt": attempt,
                        "rotation_cap": rotation_cap,
                        "resulting_stage": stage,
                        "outcome": "claim_lost" if claim_lost else "persist_refused",
                        "model": model,
                        "cost": cost,
                    }
                    logger.warning(
                        "refine_name_persistence_refused %s",
                        json.dumps(refusal_event, sort_keys=True),
                    )
                    if on_event is not None:
                        on_event(refusal_event)
                    continue
                # A grammar/validation failure (the LLM proposed an
                # ungrammatical name — e.g. an unregistered qualifier token) or
                # a self-referential refine is a NORMAL failed-refine outcome,
                # not a crash: the model keeps producing the same output for
                # this item.  Mark it exhausted (terminal) instead of reverting
                # to 'reviewed', which would re-claim and re-charge it on a paid
                # model every cycle — an infinite paid loop.  Log terminal
                # failures at WARNING; only genuinely unexpected errors get the
                # ERROR + traceback.
                is_terminal = _is_refine_grammar_failure(exc)
                if is_terminal:
                    logger.warning(
                        "refine_name failed (deterministic, marking exhausted) "
                        "for %s: %s",
                        sn_id,
                        _exc_str[:200],
                    )
                else:
                    logger.exception("refine_name failed for %s", sn_id)
                stage = await _stop_attempt(
                    item,
                    _refine_failure_stop_reason(exc, terminal=is_terminal),
                    detail=_exc_str[:500],
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": sn_id,
                            "old_name": sn_id,
                            "outcome": "refine_failed",
                            "attempt": attempt,
                            "rotation_cap": rotation_cap,
                            "resulting_stage": stage,
                            "model": model,
                            "cost": cost,
                            "llm_tokens_in": llm_tokens_in,
                            "llm_tokens_out": llm_tokens_out,
                            "llm_tokens_cached_read": llm_tokens_cached_read,
                            "llm_tokens_cached_write": llm_tokens_cached_write,
                        }
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


_NAME_REVIEW_SOURCE_CONTEXT_LIMIT = 8


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

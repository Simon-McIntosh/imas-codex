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

import asyncio  # noqa: F401 - resolved by task modules through _worker_dependency
import inspect as _inspect
import json  # noqa: F401 - resolved by task modules through _worker_dependency
import logging
import re as _re  # noqa: F401 - resolved by worker_common through _worker_dependency
from functools import cache as _cache, lru_cache  # noqa: F401
from typing import Any

from imas_codex.standard_names.budget import (
    bind_attempt_exposure,  # noqa: F401
    charge_billable_exception,  # noqa: F401
    model_provider_exposure,  # noqa: F401
)
from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,  # noqa: F401
    SEMANTIC_SIM_GATE_MODEL,  # noqa: F401
    SEMANTIC_SIM_GATE_RESOLUTION_METHOD,  # noqa: F401
)

# Kept as resolver targets: the task modules reach these through
# ``_worker_dependency``, which reads this module's globals.
from imas_codex.standard_names.provenance import detect_value_provenance  # noqa: F401
from imas_codex.standard_names.source_paths import (
    DD_PREFIX,  # noqa: F401
    encode_source_path,  # noqa: F401
    strip_dd_prefix,  # noqa: F401
)

from .worker_tasks import (
    worker_common as _moved_worker_common,
    worker_compose_and_validate as _moved_worker_compose_and_validate,
    worker_compose_batches as _moved_worker_compose_batches,
    worker_enrich_parents as _moved_worker_enrich_parents,
    worker_extraction_and_enrichment as _moved_worker_extraction_and_enrichment,
    worker_generate_docs as _moved_worker_generate_docs,
    worker_refine_docs as _moved_worker_refine_docs,
    worker_refine_names as _moved_worker_refine_names,
    worker_review_docs as _moved_worker_review_docs,
    worker_review_names as _moved_worker_review_names,
)


def _worker_dependency(name: str):
    """Resolve a worker collaborator through the entrypoint."""
    return globals()[name]


logger = logging.getLogger(__name__)


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


def _canonical_docs_review_item(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._canonical_docs_review_item."""
    return _moved_worker_review_docs._canonical_docs_review_item(
        _worker_dependency, *args, **kwargs
    )


_canonical_docs_review_item.__wrapped__ = (
    _moved_worker_review_docs._canonical_docs_review_item
)


def _bound_docs_review_dd_paths(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._bound_docs_review_dd_paths."""
    return _moved_worker_review_docs._bound_docs_review_dd_paths(
        _worker_dependency, *args, **kwargs
    )


_bound_docs_review_dd_paths.__wrapped__ = (
    _moved_worker_review_docs._bound_docs_review_dd_paths
)


def _set_docs_review_dd_source_paths(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._set_docs_review_dd_source_paths."""
    return _moved_worker_review_docs._set_docs_review_dd_source_paths(
        _worker_dependency, *args, **kwargs
    )


_set_docs_review_dd_source_paths.__wrapped__ = (
    _moved_worker_review_docs._set_docs_review_dd_source_paths
)


def _has_dd_source_reference(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._has_dd_source_reference."""
    return _moved_worker_review_docs._has_dd_source_reference(
        _worker_dependency, *args, **kwargs
    )


_has_dd_source_reference.__wrapped__ = (
    _moved_worker_review_docs._has_dd_source_reference
)


def _load_docs_review_source_item(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._load_docs_review_source_item."""
    return _moved_worker_review_docs._load_docs_review_source_item(
        _worker_dependency, *args, **kwargs
    )


_load_docs_review_source_item.__wrapped__ = (
    _moved_worker_review_docs._load_docs_review_source_item
)


def _load_docs_review_parent_children(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._load_docs_review_parent_children."""
    return _moved_worker_review_docs._load_docs_review_parent_children(
        _worker_dependency, *args, **kwargs
    )


_load_docs_review_parent_children.__wrapped__ = (
    _moved_worker_review_docs._load_docs_review_parent_children
)


def _load_docs_review_examples(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._load_docs_review_examples."""
    return _moved_worker_review_docs._load_docs_review_examples(
        _worker_dependency, *args, **kwargs
    )


_load_docs_review_examples.__wrapped__ = (
    _moved_worker_review_docs._load_docs_review_examples
)


def _enrich_docs_review_dd_context(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._enrich_docs_review_dd_context."""
    return _moved_worker_review_docs._enrich_docs_review_dd_context(
        _worker_dependency, *args, **kwargs
    )


_enrich_docs_review_dd_context.__wrapped__ = (
    _moved_worker_review_docs._enrich_docs_review_dd_context
)


def _docs_review_request_identity(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._docs_review_request_identity."""
    return _moved_worker_review_docs._docs_review_request_identity(
        _worker_dependency, *args, **kwargs
    )


_docs_review_request_identity.__wrapped__ = (
    _moved_worker_review_docs._docs_review_request_identity
)


def _bounded_docs_escalation_factory(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._bounded_docs_escalation_factory."""
    return _moved_worker_review_docs._bounded_docs_escalation_factory(
        _worker_dependency, *args, **kwargs
    )


_bounded_docs_escalation_factory.__wrapped__ = (
    _moved_worker_review_docs._bounded_docs_escalation_factory
)


async def prepare_docs_review_request(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs.prepare_docs_review_request."""
    return await _moved_worker_review_docs.prepare_docs_review_request(
        _worker_dependency, *args, **kwargs
    )


prepare_docs_review_request.__wrapped__ = (
    _moved_worker_review_docs.prepare_docs_review_request
)


async def _release_docs_review_claim(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._release_docs_review_claim."""
    return await _moved_worker_review_docs._release_docs_review_claim(
        _worker_dependency, *args, **kwargs
    )


_release_docs_review_claim.__wrapped__ = (
    _moved_worker_review_docs._release_docs_review_claim
)


def _documentation_conformance_note(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._documentation_conformance_note."""
    return _moved_worker_review_docs._documentation_conformance_note(
        _worker_dependency, *args, **kwargs
    )


_documentation_conformance_note.__wrapped__ = (
    _moved_worker_review_docs._documentation_conformance_note
)


async def _documentation_banned_prose_note(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._documentation_banned_prose_note."""
    return await _moved_worker_review_docs._documentation_banned_prose_note(
        _worker_dependency, *args, **kwargs
    )


_documentation_banned_prose_note.__wrapped__ = (
    _moved_worker_review_docs._documentation_banned_prose_note
)


async def _documentation_contradiction_note(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs._documentation_contradiction_note."""
    return await _moved_worker_review_docs._documentation_contradiction_note(
        _worker_dependency, *args, **kwargs
    )


_documentation_contradiction_note.__wrapped__ = (
    _moved_worker_review_docs._documentation_contradiction_note
)


async def process_review_docs_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_review_docs.process_review_docs_batch."""
    return await _moved_worker_review_docs.process_review_docs_batch(
        _worker_dependency, *args, **kwargs
    )


process_review_docs_batch.__wrapped__ = (
    _moved_worker_review_docs.process_review_docs_batch
)


# =============================================================================
# process_refine_docs_batch — DocsRevision snapshot + in-place docs update
# =============================================================================


async def process_refine_docs_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_docs.process_refine_docs_batch."""
    return await _moved_worker_refine_docs.process_refine_docs_batch(
        _worker_dependency, *args, **kwargs
    )


process_refine_docs_batch.__wrapped__ = (
    _moved_worker_refine_docs.process_refine_docs_batch
)


# ═══════════════════════════════════════════════════════════════════════
# Embed worker — batch-embed StandardName nodes
# ═══════════════════════════════════════════════════════════════════════


async def process_embed_batch(*args: Any, **kwargs: Any):
    """Forward to worker_tasks.worker_refine_docs.process_embed_batch."""
    return await _moved_worker_refine_docs.process_embed_batch(
        _worker_dependency, *args, **kwargs
    )


process_embed_batch.__wrapped__ = _moved_worker_refine_docs.process_embed_batch


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

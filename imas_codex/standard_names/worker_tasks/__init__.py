"""Worker task implementations for the standard-name pipeline.

Task functions that accept the composition root resolver also accept ordinary
caller arguments directly. Internal callers may still pass a resolver first.
"""

from __future__ import annotations

import functools
import inspect
from importlib import import_module
from typing import Any


def _resolve_dependency(name: str) -> Any:
    try:
        return resolve_owned_name(name)
    except AttributeError:
        pass
    from imas_codex.standard_names.workers import _worker_dependency

    return _worker_dependency(name)


def _bind_task(function: Any) -> Any:
    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_bound(*args: Any, **kwargs: Any) -> Any:
            if args and callable(args[0]):
                return await function(*args, **kwargs)
            return await function(_resolve_dependency, *args, **kwargs)

        wrapper = async_bound
    else:

        @functools.wraps(function)
        def bound(*args: Any, **kwargs: Any) -> Any:
            if args and callable(args[0]):
                return function(*args, **kwargs)
            return function(_resolve_dependency, *args, **kwargs)

        wrapper = bound

    for attribute in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(function, attribute):
            setattr(wrapper, attribute, getattr(function, attribute))
    return wrapper


_OWNERS_BY_MODULE = {
    "worker_common": (
        "Any",
        "_DOMAIN_UK_US",
        "_GRAMMAR_FIELDS",
        "_GREEK_CMD_RE",
        "_GREEK_CMD_TO_SYMBOL",
        "_GREEK_STRANDED_BACKSLASH_RE",
        "_GREEK_WORD_RE",
        "_GREEK_WORD_TO_SYMBOL",
        "_LATEX_CMD_ARG_RE",
        "_LATEX_CMD_BARE_RE",
        "_MATH_DELIM_RE",
        "_NON_NAMEABLE_BARE_TOKENS",
        "_SHAPE_PARAMETER_BASE_POLICY",
        "_SNAKE_CASE_RE",
        "_SUBSCRIPT_BRACE_RE",
        "_SUPERSCRIPT_BRACE_RE",
        "_active_editorial_gap_guidance",
        "_auto_detect_physical_base_gaps",
        "_build_grammar_retry_reason",
        "_build_token_reuse_retry_reason",
        "_cache",
        "_candidate_source_node_id",
        "_claimed_source_ids",
        "_claimed_source_outcome",
        "_coerce_llm_metric",
        "_compute_token_reuse_hits",
        "_dedup_adjacent_tokens",
        "_extract_llm_telemetry",
        "_filter_persist_candidates_to_claimed_sources",
        "_grammar_round_trip_failures",
        "_inject_shape_parameter_surface",
        "_ir_claims_time_change_in_scope",
        "_ir_denotes_time_change",
        "_isn_physical_bases",
        "_isn_segment_tokens",
        "_load_known_physical_bases",
        "_mark_source_prevalidation_failed",
        "_name_claims_time_change",
        "_name_denotes_rate",
        "_name_denotes_state",
        "_persist_dd_gap_evidence",
        "_persist_description_checked_candidates",
        "_rate_natured_bases",
        "_re",
        "_review_dd_source_bindings",
        "_review_dd_source_ids",
        "_sanitize_compose_result_sources",
        "_sanitize_dd_gap_evidence",
        "_shape_parameter_bases",
        "_shape_parameter_surface",
        "_stamp_dedup_decision",
        "_state_resolution_tokens",
        "_strip_response_source_id",
        "_time_change_claim_is_binary_scoped",
        "_withhold_binding_conflicted_sources",
        "annotations",
        "backfill_semantic_similarity_gate_resolution_method",
        "is_non_nameable_coordinate",
        "is_well_formed_candidate",
        "logger",
        "logging",
        "lru_cache",
        "normalize_description_notation",
        "normalize_description_text",
        "normalize_prose_spelling",
        "normalize_spelling",
        "provenance_canonical_names",
        "shape_parameter_base_drift",
    ),
    "worker_compose_and_validate": (
        "_SCOPED_VALIDATION_CLAIM_TIMEOUT",
        "_convert_fields_to_grammar",
        "_is_quarantined",
        "_validate_via_isn",
        "claim_ids_for_validation",
        "compose_worker",
        "consolidate_worker",
        "drain_validation_backlog",
        "drain_validation_for_ids",
        "persist_worker",
        "validate_name_candidate",
        "validate_worker",
    ),
    "worker_compose_batches": (
        "_GRAMMAR_FAILURE_MARKERS",
        "compose_batch",
        "process_generate_name_batch",
    ),
    "worker_enrich_parents": (
        "_DOCS_REVIEW_VOLATILE_ITEM_FIELDS",
        "process_enrich_parents_batch",
    ),
    "worker_extraction_and_enrichment": (
        "_ANCESTOR_CONTEXT_QUERY",
        "_AXIS_LEAVES",
        "_COMPOSE_DOMAIN_STATE_QUERY",
        "_CROSS_IDS_QUERY",
        "_DD_CONTEXT_QUERY",
        "_DD_DEVICE_EXPANSIONS",
        "_ERROR_FIELDS_QUERY",
        "_HARDWARE_LOCUS_WORDS",
        "_IDS_CONTEXT_QUERY",
        "_MAX_MEMBER_DOCS",
        "_MEMBER_DOC_MAX_CHARS",
        "_MEMBER_DOC_QUERY",
        "_NEARBY_COMPARATOR_HARD_CEILING",
        "_OPTICAL_GEOMETRY_PATH_WORDS",
        "_ORDINAL_POSITION_WORDS",
        "_RELATED_MAX_RESULTS",
        "_RERANK_LEASE_FLOOR_USD",
        "_SOLID_GEOMETRY_PATH_SEGMENTS",
        "_TIME_DERIVATIVE_SEGMENT_RE",
        "_VERSION_HISTORY_QUERY",
        "_book_rerank_spend",
        "_collect_nearby_name_comparators",
        "_compose_budget_phase",
        "_consume_claimed_vocab_gaps",
        "_content_tokens",
        "_dd_standard_name_units_dimensionally_agree",
        "_enrich_batch_items",
        "_enrich_ids_context",
        "_expanded_path_tokens",
        "_geometry_representation_conflicts",
        "_grammar_retry",
        "_hybrid_search_neighbours",
        "_hybrid_search_neighbours_batch",
        "_is_attachment_consistent",
        "_is_ordinal_point_sample",
        "_mark_vocab_gap_sources",
        "_name_locus_token",
        "_persist_claimed_skips",
        "_process_attachments",
        "_process_attachments_core",
        "_related_path_neighbours",
        "_rerank_comparators_for_compose",
        "_rerank_comparators_for_worker",
        "_reranked_nearby_comparators",
        "_retry_attempts",
        "_retry_k_expansion",
        "_search_nearby_names",
        "_search_reference_exemplars",
        "_self_refine_candidate",
        "_update_sources_after_attach",
        "_update_sources_after_compose",
        "_update_sources_after_skip",
        "_update_sources_after_vocab_gap",
        "_vector_fields_conflict",
        "_write_composed_name_domains",
        "extract_worker",
    ),
    "worker_generate_docs": (
        "_documentation_with_registry_form",
        "_enrich_dd_path_context",
        "_enrich_for_docs_gen",
        "_fill_template_sentence",
        "_instantiated_template_sentences",
        "_nearby_names_for_docs_gen",
        "process_generate_docs_batch",
    ),
    "worker_refine_docs": (
        "process_embed_batch",
        "process_refine_docs_batch",
    ),
    "worker_refine_names": (
        "RefineCandidateValidationError",
        "RefineScopeContinuityError",
        "_NAME_REVIEW_SOURCE_CONTEXT_LIMIT",
        "_RETURN_UNSPENT_REFINE_ATTEMPT",
        "_VOCABULARY_GAP_MARKERS",
        "_compose_refined_candidate_name",
        "_is_refine_docs_failure",
        "_is_refine_grammar_failure",
        "_refine_batch_scope_id",
        "_refine_failure_stop_reason",
        "_return_unspent_refine_attempt",
        "process_refine_name_batch",
    ),
    "worker_review_docs": (
        "_bound_docs_review_dd_paths",
        "_bounded_docs_escalation_factory",
        "_canonical_docs_review_item",
        "_docs_review_request_identity",
        "_documentation_banned_prose_note",
        "_documentation_conformance_note",
        "_documentation_contradiction_note",
        "_enrich_docs_review_dd_context",
        "_has_dd_source_reference",
        "_load_docs_review_examples",
        "_load_docs_review_parent_children",
        "_load_docs_review_source_item",
        "_release_docs_review_claim",
        "_set_docs_review_dd_source_paths",
        "prepare_docs_review_request",
        "process_review_docs_batch",
    ),
    "worker_review_names": (
        "_DD_PATH_CONTEXT_QUERY",
        "_DOCS_GEN_COCOS_PARAMS_QUERY",
        "_DOCS_GEN_ENRICH_QUERY",
        "_DOCS_GEN_NEARBY_QUERY",
        "_DOCS_GEN_PARENT_CHILDREN_QUERY",
        "_closed_vocabulary_definitions",
        "_enrich_name_review_items",
        "_persisting_quorum_cycles",
        "_quorum_incomplete_deferrals",
        "_record_quorum_incomplete",
        "_review_source_binding_sort_key",
        "_run_rd_quorum_cycles",
        "_unfunded_cause",
        "process_review_name_batch",
        "quorum_incomplete_snapshot",
        "reset_quorum_incomplete",
        "semantic_gate_name_text",
    ),
}
_OWNER_BY_NAME = {
    name: module for module, names in _OWNERS_BY_MODULE.items() for name in names
}


def resolve_owned_name(name: str) -> Any:
    """Resolve a moved symbol at its owning module."""
    module = _OWNER_BY_NAME.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(f".{module}", __name__), name)


for _name in (
    "worker_common",
    "worker_compose_and_validate",
    "worker_compose_batches",
    "worker_enrich_parents",
    "worker_extraction_and_enrichment",
    "worker_generate_docs",
    "worker_refine_docs",
    "worker_refine_names",
    "worker_review_docs",
    "worker_review_names",
):
    _module = import_module(f".{_name}", __name__)
    for _symbol, _function in vars(_module).copy().items():
        if (
            not callable(_function)
            or inspect.isclass(_function)
            or getattr(_function, "__module__", None) != _module.__name__
        ):
            continue
        try:
            _parameters = iter(inspect.signature(_function).parameters.values())
            _first = next(_parameters)
        except (TypeError, ValueError, StopIteration):
            continue
        if _first.name == "_worker_dependency":
            setattr(_module, _symbol, _bind_task(_function))

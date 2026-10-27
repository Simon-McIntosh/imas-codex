"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

from .graph_store import (
    graph_foundation as _graph_foundation,
    owner_module_path,
    resolve_owned_name as _resolve_owned_graph_name,
)

#: Collaborators the graph store modules read off the wired service object,
#: named by the attribute each module requests. The set is declared rather
#: than discovered: a name the store does not ask for must not become
#: reachable through the entrypoint again. The store's own reads are
#: compared against it by the facade-closure test.
_WIRED_SERVICE_NAMES: frozenset[str] = frozenset(
    {
        "Any",
        "DerivedParentCleanupRefusal",
        "ExactNameScopeConflict",
        "ExhaustedOrphanSupersessionConflict",
        "GraphClient",
        "LifecyclelessStubConflict",
        "ManifestDrainConflict",
        "NameStage",
        "Path",
        "ProtectedDeletionError",
        "QUORATE_RESOLUTION_METHODS",
        "RefineClaimReleaseConflict",
        "RefineStopReason",
        "ReviewDimensionVoidRefused",
        "ReviewWriteResult",
        "STAGE_DOCS_FOR_RESCORE_QUERY",
        "ScalarSelectedDedupConflict",
        "SemanticMirrorRepairConflict",
        "SignedSourceDispositionConflict",
        "SourceClaimResult",
        "StructuralAuthorityConflict",
        "SupersedeIntoAncestorConflict",
        "_ESCALATOR_CHAIN_SIZE",
        "_GENERATED_SUPERSESSION_PREFLIGHT_QUERY",
        "_GRAMMAR_SEGMENT_COLUMNS",
        "_MAX_COMPOSE_CLAIM_ATTEMPTS",
        "_MAX_LINK_RETRIES",
        "_PIPELINE_FROZEN_NAME_STAGES",
        "_ParentMaterializationCapture",
        "_SEGMENT_EDGE_TYPES",
        "_SHA256_RE",
        "_SKELETON_PLACEHOLDER_PREDICATE",
        "_TransactionQuery",
        "_ancestor_supersession_authority",
        "_authority_json_value",
        "_authority_payload_hash",
        "_backfill_cluster_from_sources",
        "_bootstrap_missing_derived_parent_targets",
        "_canonicalize_lifecycleless_manifest_value",
        "_catalog_edit_adjudication_signature_hash",
        "_claim_disposition",
        "_claim_sn_atomic",
        "_claimed_run_ids",
        "_clear_superseded_child_derived_parent_edges",
        "_coerce_segment_value",
        "_collect_run_statuses",
        "_complete_source_fences",
        "_compute_link_status",
        "_delete_derived_parent_nodes",
        "_derived_parent_source_metadata",
        "_derived_parent_unit_from_eligible_children",
        "_detach_stale_ancestor_sources",
        "_doc_link_mismatches",
        "_docs_axis_rederive_query",
        "_docs_review_winner_query_body",
        "_eligible_derived_parent_unit_children",
        "_emit_magnitude_of_edges",
        "_ensure_json",
        "_ensure_list",
        "_exact_dd_source_id",
        "_exact_name_scope_refusals",
        "_exhausted_orphan_change_id",
        "_exhausted_orphan_supersession_authority",
        "_extract_links_from_docs",
        "_filter_admissible_parents",
        "_get_rich_source_name_mapping",
        "_guard_existing_target_pairings",
        "_guard_failed_source_reasons",
        "_historical_cpi_cache",
        "_is_live_structural_child",
        "_isn_grammar_context",
        "_lock_ancestor_supersession_authority",
        "_lock_claimed_name_bindings",
        "_lock_manifest_drain_authority",
        "_lock_scalar_selected_dedup_authority",
        "_manifest_plan_fingerprint",
        "_materialize_derived_parent_rows",
        "_non_winning_review_resolution_methods",
        "_normalize_bare_doc_links",
        "_normalize_pool",
        "_parse_grammar",
        "_parse_parent_grammar",
        "_persist_outcomes",
        "_persist_structural_authority",
        "_pin_dd_source_snapshots",
        "_pipeline_buckets_cache",
        "_predicted_refine_claim_release_state",
        "_project_refine_claim_release_state",
        "_public_grammar_segment_order",
        "_query_derived_parents_for_admission_cleanup",
        "_query_legacy_repairable_derived_parents",
        "_query_seedable_derived_parents",
        "_query_skeleton_placeholders_for_cleanup",
        "_quorum_admits_acceptance",
        "_record_persist_outcome",
        "_refine_claim_no_change_receipt",
        "_refine_claim_release_manifest",
        "_require_failure_reason",
        "_resolve_grammar_token_version",
        "_resolve_synced_segments",
        "_resolved_manifest_drain_plan",
        "_review_tier",
        "_rewire_has_parent_off_superseded",
        "_sanitize_doc_text",
        "_scalar_domain",
        "_scalar_selected_dedup_authority",
        "_scalar_selected_dedup_counts",
        "_segment_drift_reason",
        "_segments_from_ir",
        "_signed_source_disposition_authority",
        "_signed_source_disposition_counts",
        "_skipped_recovery_error",
        "_sorted_refine_claim_projection",
        "_source_compose_hint_decision",
        "_spurious_stub_binding_authority",
        "_standard_name_source_upstream_present_cypher",
        "_structural_accept_route",
        "_structural_authority_identities",
        "_structural_authority_record",
        "_structural_authority_snapshot",
        "_validate_derived_parent_identity",
        "_validate_signed_orphan_retirement_authority",
        "_validate_signed_source_adjudication",
        "_verify_claim_winners",
        "_verify_docs_claim_winners",
        "_verify_enrich_parents_claim_winners",
        "_verify_name_claim_winners",
        "_verify_source_claim_winners",
        "_winning_review_resolution_methods",
        "_write_grammar_decomposition",
        "_write_standard_name_edges",
        "automatic_deletion_protections",
        "build_manifest_drain_plan",
        "bump_sn_run_counter",
        "canonicalize_manifest_drain_paths",
        "claim_manifest_drain_scope",
        "classify_manifest_drain_item",
        "classify_orphan_parent_source_candidates",
        "classify_parked_source",
        "classify_sourceless_standard_names",
        "clear_manifest_drain_scope",
        "datetime",
        "defaultdict",
        "deletion_change_cypher",
        "deletion_change_params",
        "deque",
        "docs_acceptance_set",
        "docs_accepted",
        "docs_review_eligibility_params",
        "docs_review_eligibility_where",
        "fetch_docs_review_feedback_for_standard_names",
        "filter_automatic_deletion_candidates",
        "find_name_links",
        "find_name_references",
        "find_orphan_parent_source_candidates",
        "find_prose_bracket_spans",
        "has_open_docs_hold_where",
        "hashlib",
        "isn_vocabulary_signature",
        "json",
        "logger",
        "mark_sources_stale",
        "normalize_name_id",
        "nullcontext",
        "open_holds",
        "os",
        "persist_generated_name_winners",
        "plan_review_dimension_void",
        "reattach_produced_name_edges",
        "reconcile_constraint_fit_artifacts",
        "reconcile_descriptionless_composed_names",
        "reconcile_sourceless_pipeline_names",
        "reconcile_standard_name_kinds",
        "record_parked_dispositions",
        "refuse_protected_automatic_deletion",
        "release_generate_name_claims",
        "release_manifest_drain_claims",
        "supersede_prior_source_names",
        "suppress",
        "sweep_orphaned_docs_revisions",
        "time",
        "update_review_aggregates",
        "uuid",
        "write_reviews",
        "write_standard_names",
        "write_vocab_gaps",
    }
)


def _closure_message(name: str) -> str:
    """Explain where a name is owned when this entrypoint refuses it."""
    try:
        owner = owner_module_path(name)
    except KeyError:
        return (
            f"{name!r} is not wired into the standard-name graph composition "
            "root; import it from the module that defines it"
        )
    return (
        f"{name!r} is not wired into the standard-name graph composition root; "
        f"import it from {owner}"
    )


class _GraphServices:
    """Collaborators the composition root supplies to the graph store modules."""

    def __getattr__(self, name: str) -> Any:
        if name not in _WIRED_SERVICE_NAMES:
            raise AttributeError(_closure_message(name))
        try:
            return _resolve_owned_graph_name(name)
        except AttributeError:
            return globals()[name]


def __getattr__(name: str) -> Any:
    """Refuse a moved name at the entrypoint, naming the module that owns it."""
    raise AttributeError(_closure_message(name))


_graph_foundation.wire_services(_GraphServices())


_structural_edge_reconcile_apply = import_module(
    ".graph_store.graph_lineage", __package__
)._structural_edge_reconcile_apply

globals()["reconcile_structural_edges_for_" + "standard_names"] = (
    _structural_edge_reconcile_apply
)

_normalization_peel_unit_repair_apply = import_module(
    ".graph_store.graph_lineage", __package__
)._normalization_peel_unit_repair_apply

globals()["repair_normalization_peel_parent_" + "units"] = (
    _normalization_peel_unit_repair_apply
)

_lifecycleless_stub_reconcile_apply = import_module(
    ".graph_store.graph_lineage", __package__
)._lifecycleless_stub_reconcile_apply

_lifecycleless_stub_reconcile_signature = import_module(
    ".graph_store.graph_lineage", __package__
)._lifecycleless_stub_reconcile_signature

_lifecycleless_stub_reconcile_parameters = import_module(
    ".graph_store.graph_lineage", __package__
)._lifecycleless_stub_reconcile_parameters

_lifecycleless_stub_reconcile_apply.__signature__ = inspect.Signature(
    [
        _lifecycleless_stub_reconcile_parameters["apply"],
        _lifecycleless_stub_reconcile_parameters["manifest_sha256"],
        _lifecycleless_stub_reconcile_parameters["gc"],
    ],
    return_annotation=_lifecycleless_stub_reconcile_signature.return_annotation,
)
globals()["reconcile_lifecycleless_" + "standard_name_stubs"] = (
    _lifecycleless_stub_reconcile_apply
)
del _lifecycleless_stub_reconcile_signature
del _lifecycleless_stub_reconcile_parameters

_error_sibling_reconcile_apply = import_module(
    ".graph_store.graph_audits", __package__
)._error_sibling_reconcile_apply

globals()["reconcile_" + "error_siblings"] = _error_sibling_reconcile_apply

_catalog_source_disposition_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._catalog_source_disposition_apply
_catalog_source_disposition_signature = import_module(
    ".graph_store.graph_supersession", __package__
)._catalog_source_disposition_signature
_catalog_source_disposition_parameters = import_module(
    ".graph_store.graph_supersession", __package__
)._catalog_source_disposition_parameters
_catalog_source_disposition_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._catalog_source_disposition_apply

globals()["apply_" + "adjudicated_source_dispositions"] = (
    _catalog_source_disposition_apply
)
del _catalog_source_disposition_signature
del _catalog_source_disposition_parameters

_semantic_mirror_repair_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._semantic_mirror_repair_apply
_semantic_mirror_repair_signature = import_module(
    ".graph_store.graph_supersession", __package__
)._semantic_mirror_repair_signature
_semantic_mirror_repair_parameters = import_module(
    ".graph_store.graph_supersession", __package__
)._semantic_mirror_repair_parameters
_semantic_mirror_repair_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._semantic_mirror_repair_apply

globals()["repair_" + "scalar_projection_mismatches"] = _semantic_mirror_repair_apply
del _semantic_mirror_repair_signature
del _semantic_mirror_repair_parameters

_ineligible_source_retirement_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._ineligible_source_retirement_apply
_ineligible_source_retirement_signature = import_module(
    ".graph_store.graph_supersession", __package__
)._ineligible_source_retirement_signature
_ineligible_source_retirement_parameters = import_module(
    ".graph_store.graph_supersession", __package__
)._ineligible_source_retirement_parameters
_ineligible_source_retirement_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._ineligible_source_retirement_apply

globals()["retire_" + "ineligible_standard_name_sources"] = (
    _ineligible_source_retirement_apply
)
del _ineligible_source_retirement_signature
del _ineligible_source_retirement_parameters

_dual_authority_retirement_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._dual_authority_retirement_apply
_dual_authority_hidden_parameters = import_module(
    ".graph_store.graph_supersession", __package__
)._dual_authority_hidden_parameters
_dual_authority_retirement_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._dual_authority_retirement_apply

globals()["retire_signed_" + "dual_authority_targets"] = (
    _dual_authority_retirement_apply
)
del _dual_authority_hidden_parameters

_refused_target_orphan_apply = import_module(
    ".graph_store.graph_supersession", __package__
)._refused_target_orphan_apply

globals()["retire_signed_" + "provenance_orphans"] = _refused_target_orphan_apply

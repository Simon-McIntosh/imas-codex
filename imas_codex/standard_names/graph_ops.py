"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

# ruff: noqa: F401
import functools
import hashlib
import inspect
import json
import logging  # noqa: F401
import os
import re
import time
import uuid
from collections import Counter, defaultdict, deque  # noqa: F401
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import NameStage, RefineStopReason, SNRunStatus
from imas_codex.graph.schema import GraphSchema
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.doc_links import (
    find_name_links,
    find_name_references,
    find_prose_bracket_spans,
)
from imas_codex.standard_names.ledger import reattach_produced_name_edges  # noqa: F401
from imas_codex.standard_names.protection import (
    ProtectedDeletionError,
    automatic_deletion_protections,
    filter_automatic_deletion_candidates,
    refuse_protected_automatic_deletion,
)
from imas_codex.standard_names.provenance_lifecycle import (
    deletion_change_cypher,
    deletion_change_params,
)
from imas_codex.standard_names.review.holds import (
    docs_acceptance_set,  # noqa: F401
    docs_acceptance_where,  # noqa: F401
    docs_accepted,  # noqa: F401
    has_open_docs_hold_where,
    open_holds,
    sync_holds,  # noqa: F401
)
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

# Wire graph-domain calls through the entrypoint at runtime.
from .graph_store import graph_foundation as _graph_foundation


class _GraphServices:
    def __getattr__(self, name: str) -> Any:
        return globals()[name]


_graph_foundation.services = _GraphServices()


logger = import_module(".graph_store.graph_foundation", __package__).logger


_STABLE_BINDING_NAME_STAGES = import_module(
    ".graph_store.graph_foundation", __package__
)._STABLE_BINDING_NAME_STAGES
_TERMINAL_BINDING_NAME_STAGES = import_module(
    ".graph_store.graph_foundation", __package__
)._TERMINAL_BINDING_NAME_STAGES
_SETTLED_SOURCE_BINDING_STAGES = import_module(
    ".graph_store.graph_foundation", __package__
)._SETTLED_SOURCE_BINDING_STAGES
_CLAIM_TIMEOUT_SECONDS = import_module(
    ".graph_store.graph_foundation", __package__
)._CLAIM_TIMEOUT_SECONDS

# The lowest observed daily structural-reaper total is 74.  Eighty allows a
# normal-sized cleanup while refusing the 93-identity deletion that exposed the
# missing guard.
DERIVED_PARENT_CLEANUP_DELETION_LIMIT = import_module(
    ".graph_store.graph_foundation", __package__
).DERIVED_PARENT_CLEANUP_DELETION_LIMIT


DerivedParentCleanupRefusal = import_module(
    ".graph_store.graph_foundation", __package__
).DerivedParentCleanupRefusal


_TransactionQuery = import_module(
    ".graph_store.graph_foundation", __package__
)._TransactionQuery


def _ensure_json(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._ensure_json."""
    from .graph_store import graph_foundation

    return graph_foundation._ensure_json(*args, **kwargs)


def _ensure_list(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._ensure_list."""
    from .graph_store import graph_foundation

    return graph_foundation._ensure_list(*args, **kwargs)


def _scalar_domain(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._scalar_domain."""
    from .graph_store import graph_foundation

    return graph_foundation._scalar_domain(*args, **kwargs)


def reclassify_standard_name_domain(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation.reclassify_standard_name_domain."""
    from .graph_store import graph_foundation

    return graph_foundation.reclassify_standard_name_domain(*args, **kwargs)


#: Physics abbreviation expansions applied before name persistence.
#: Keys are case-sensitive abbreviations as they appear in LLM output;
#: values are their lowercase ISN-compliant expansions.
_PHYSICS_ABBREVIATIONS = import_module(
    ".graph_store.graph_foundation", __package__
)._PHYSICS_ABBREVIATIONS


def normalize_name_id(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation.normalize_name_id."""
    from .graph_store import graph_foundation

    return graph_foundation.normalize_name_id(*args, **kwargs)


def _public_segment_token_index(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._public_segment_token_index."""
    from .graph_store import graph_foundation

    return graph_foundation._public_segment_token_index(*args, **kwargs)


def _public_grammar_segment_order(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._public_grammar_segment_order."""
    from .graph_store import graph_foundation

    return graph_foundation._public_grammar_segment_order(*args, **kwargs)


def _segments_from_ir(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._segments_from_ir."""
    from .graph_store import graph_foundation

    return graph_foundation._segments_from_ir(*args, **kwargs)


def _parse_grammar(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._parse_grammar."""
    from .graph_store import graph_foundation

    return graph_foundation._parse_grammar(*args, **kwargs)


def _semantic_ir_expression(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._semantic_ir_expression."""
    from .graph_store import graph_foundation

    return graph_foundation._semantic_ir_expression(*args, **kwargs)


def strict_review_grammar_context(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation.strict_review_grammar_context."""
    from .graph_store import graph_foundation

    return graph_foundation.strict_review_grammar_context(*args, **kwargs)


def _compute_link_status(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._compute_link_status."""
    from .graph_store import graph_foundation

    return graph_foundation._compute_link_status(*args, **kwargs)


_LINK_RE = import_module(".graph_store.graph_foundation", __package__)._LINK_RE

_DD_LINK_RE = import_module(".graph_store.graph_foundation", __package__)._DD_LINK_RE


def _extract_links_from_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_foundation._extract_links_from_docs."""
    from .graph_store import graph_foundation

    return graph_foundation._extract_links_from_docs(*args, **kwargs)


# =============================================================================
# Pipeline cost model — bucket queries & historical CPI
# =============================================================================

# Module-level TTL cache for pipeline buckets.
_pipeline_buckets_cache = import_module(
    ".graph_store.graph_foundation", __package__
)._pipeline_buckets_cache

# Module-level cache for historical CPI (once per process).
_historical_cpi_cache = import_module(
    ".graph_store.graph_foundation", __package__
)._historical_cpi_cache


def query_pipeline_buckets(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.query_pipeline_buckets."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.query_pipeline_buckets(
        *args, services=_graph_foundation.services, **kwargs
    )


def query_historical_cpi(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.query_historical_cpi."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.query_historical_cpi(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Read helpers — extraction candidates
# =============================================================================


def get_extraction_candidates_dd(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.get_extraction_candidates_dd."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.get_extraction_candidates_dd(
        *args, services=_graph_foundation.services, **kwargs
    )


def get_extraction_candidates_signals(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.get_extraction_candidates_signals."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.get_extraction_candidates_signals(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Deduplication
# =============================================================================


def get_existing_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.get_existing_standard_names."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.get_existing_standard_names(
        *args, services=_graph_foundation.services, **kwargs
    )


def get_named_source_ids(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.get_named_source_ids."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.get_named_source_ids(
        *args, services=_graph_foundation.services, **kwargs
    )


def partition_focus_by_accepted(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.partition_focus_by_accepted."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.partition_focus_by_accepted(
        *args, services=_graph_foundation.services, **kwargs
    )


MANIFEST_DRAIN_DISPOSITIONS = import_module(
    ".graph_store.graph_claim_scope", __package__
).MANIFEST_DRAIN_DISPOSITIONS


def canonicalize_manifest_drain_paths(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.canonicalize_manifest_drain_paths."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.canonicalize_manifest_drain_paths(
        *args, services=_graph_foundation.services, **kwargs
    )


def classify_manifest_drain_item(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.classify_manifest_drain_item."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.classify_manifest_drain_item(
        *args, services=_graph_foundation.services, **kwargs
    )


_MANIFEST_DRAIN_PLAN_QUERY = import_module(
    ".graph_store.graph_claim_scope", __package__
)._MANIFEST_DRAIN_PLAN_QUERY


def _resolved_manifest_drain_plan(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._resolved_manifest_drain_plan."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._resolved_manifest_drain_plan(
        *args, services=_graph_foundation.services, **kwargs
    )


def build_manifest_drain_plan(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.build_manifest_drain_plan."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.build_manifest_drain_plan(
        *args, services=_graph_foundation.services, **kwargs
    )


ManifestDrainConflict = import_module(
    ".graph_store.graph_claim_scope", __package__
).ManifestDrainConflict


def _manifest_plan_fingerprint(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._manifest_plan_fingerprint."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._manifest_plan_fingerprint(
        *args, services=_graph_foundation.services, **kwargs
    )


def _lock_manifest_drain_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._lock_manifest_drain_authority."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._lock_manifest_drain_authority(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_manifest_drain_scope(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.claim_manifest_drain_scope."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.claim_manifest_drain_scope(
        *args, services=_graph_foundation.services, **kwargs
    )


def prepare_manifest_drain_scope(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.prepare_manifest_drain_scope."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.prepare_manifest_drain_scope(
        *args, services=_graph_foundation.services, **kwargs
    )


def clear_manifest_drain_scope(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.clear_manifest_drain_scope."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.clear_manifest_drain_scope(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_manifest_drain_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.release_manifest_drain_claims."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.release_manifest_drain_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def finalize_manifest_drain_scope(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.finalize_manifest_drain_scope."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.finalize_manifest_drain_scope(
        *args, services=_graph_foundation.services, **kwargs
    )


def scope_focus_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.scope_focus_names."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.scope_focus_names(
        *args, services=_graph_foundation.services, **kwargs
    )


ExactNameScopeConflict = import_module(
    ".graph_store.graph_claim_scope", __package__
).ExactNameScopeConflict


_EXACT_NAME_SCOPE_PREFLIGHT_QUERY = import_module(
    ".graph_store.graph_claim_scope", __package__
)._EXACT_NAME_SCOPE_PREFLIGHT_QUERY


_EXACT_NAME_SCOPE_STAMP_QUERY = import_module(
    ".graph_store.graph_claim_scope", __package__
)._EXACT_NAME_SCOPE_STAMP_QUERY


_EXACT_NAME_SCOPE_RUN_STATUS_QUERY = import_module(
    ".graph_store.graph_claim_scope", __package__
)._EXACT_NAME_SCOPE_RUN_STATUS_QUERY


_EXACT_SCOPE_TERMINAL_NAME_STAGES = import_module(
    ".graph_store.graph_claim_scope", __package__
)._EXACT_SCOPE_TERMINAL_NAME_STAGES

_EXACT_SCOPE_TERMINAL_STATUSES = import_module(
    ".graph_store.graph_claim_scope", __package__
)._EXACT_SCOPE_TERMINAL_STATUSES


#: ``SNRunStatus`` values that end a run's ownership of its claims. Every other
#: status — ``started``, a status written by a future release, or a run row that
#: cannot be read at all — leaves the claim held, so the reading here fails
#: closed: only a run known to have stopped releases its claim.
_TERMINAL_RUN_STATUSES = import_module(
    ".graph_store.graph_claim_scope", __package__
)._TERMINAL_RUN_STATUSES


_CLAIM_ABSENT = import_module(
    ".graph_store.graph_claim_scope", __package__
)._CLAIM_ABSENT

_CLAIM_HELD = import_module(".graph_store.graph_claim_scope", __package__)._CLAIM_HELD

_CLAIM_RELEASED = import_module(
    ".graph_store.graph_claim_scope", __package__
)._CLAIM_RELEASED


def _claim_disposition(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._claim_disposition."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._claim_disposition(
        *args, services=_graph_foundation.services, **kwargs
    )


def _claimed_run_ids(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._claimed_run_ids."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._claimed_run_ids(
        *args, services=_graph_foundation.services, **kwargs
    )


def _collect_run_statuses(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._collect_run_statuses."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._collect_run_statuses(
        *args, services=_graph_foundation.services, **kwargs
    )


def _exact_name_scope_refusals(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._exact_name_scope_refusals."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._exact_name_scope_refusals(
        *args, services=_graph_foundation.services, **kwargs
    )


def scope_exact_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.scope_exact_standard_names."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.scope_exact_standard_names(
        *args, services=_graph_foundation.services, **kwargs
    )


def get_source_name_mapping(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.get_source_name_mapping."""
    from .graph_store import graph_lineage

    return graph_lineage.get_source_name_mapping(
        *args, services=_graph_foundation.services, **kwargs
    )


def _get_rich_source_name_mapping(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._get_rich_source_name_mapping."""
    from .graph_store import graph_lineage

    return graph_lineage._get_rich_source_name_mapping(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Write helpers
# =============================================================================


def fetch_low_score_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_low_score_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_low_score_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_review_feedback_for_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_review_feedback_for_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_review_feedback_for_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_reviewer_history_for_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_reviewer_history_for_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_reviewer_history_for_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_docs_review_feedback_for_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_docs_review_feedback_for_standard_names."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_docs_review_feedback_for_standard_names(
        *args, services=_graph_foundation.services, **kwargs
    )


# Public rename. The canonical name is
# ``fetch_docs_review_feedback_for_standard_names`` (above); the legacy
# ``fetch_docs_review_feedback_for_sns`` alias is retained for one
# release with a DeprecationWarning. Package-internal callers have already
# moved to the canonical name.
def fetch_docs_review_feedback_for_sns(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_docs_review_feedback_for_sns."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_docs_review_feedback_for_sns(
        *args, services=_graph_foundation.services, **kwargs
    )


def _filter_admissible_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._filter_admissible_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._filter_admissible_parents(
        *args, services=_graph_foundation.services, **kwargs
    )


def _emit_magnitude_of_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._emit_magnitude_of_edges."""
    from .graph_store import graph_lineage

    return graph_lineage._emit_magnitude_of_edges(
        *args, services=_graph_foundation.services, **kwargs
    )


def _bootstrap_missing_derived_parent_targets(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._bootstrap_missing_derived_parent_targets."""
    from .graph_store import graph_lineage

    return graph_lineage._bootstrap_missing_derived_parent_targets(
        *args, services=_graph_foundation.services, **kwargs
    )


def _write_standard_name_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._write_standard_name_edges."""
    from .graph_store import graph_lineage

    return graph_lineage._write_standard_name_edges(
        *args, services=_graph_foundation.services, **kwargs
    )


_structural_edge_reconcile_apply = import_module(
    ".graph_store.graph_lineage", __package__
)._structural_edge_reconcile_apply

globals()["reconcile_structural_edges_for_" + "standard_names"] = (
    _structural_edge_reconcile_apply
)


def _clear_superseded_child_derived_parent_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._clear_superseded_child_derived_parent_edges."""
    from .graph_store import graph_lineage

    return graph_lineage._clear_superseded_child_derived_parent_edges(
        *args, services=_graph_foundation.services, **kwargs
    )


def rederive_structural_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.rederive_structural_edges."""
    from .graph_store import graph_lineage

    return graph_lineage.rederive_structural_edges(
        *args, services=_graph_foundation.services, **kwargs
    )


def _rewire_has_parent_off_superseded(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._rewire_has_parent_off_superseded."""
    from .graph_store import graph_lineage

    return graph_lineage._rewire_has_parent_off_superseded(
        *args, services=_graph_foundation.services, **kwargs
    )


def _parse_parent_grammar(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._parse_parent_grammar."""
    from .graph_store import graph_lineage

    return graph_lineage._parse_parent_grammar(
        *args, services=_graph_foundation.services, **kwargs
    )


def _query_seedable_derived_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_seedable_derived_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._query_seedable_derived_parents(
        *args, services=_graph_foundation.services, **kwargs
    )


def _query_legacy_repairable_derived_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_legacy_repairable_derived_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._query_legacy_repairable_derived_parents(
        *args, services=_graph_foundation.services, **kwargs
    )


def _query_derived_parents_for_admission_cleanup(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_derived_parents_for_admission_cleanup."""
    from .graph_store import graph_lineage

    return graph_lineage._query_derived_parents_for_admission_cleanup(
        *args, services=_graph_foundation.services, **kwargs
    )


def _delete_derived_parent_nodes(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._delete_derived_parent_nodes."""
    from .graph_store import graph_lineage

    return graph_lineage._delete_derived_parent_nodes(
        *args, services=_graph_foundation.services, **kwargs
    )


def sweep_orphaned_docs_revisions(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.sweep_orphaned_docs_revisions."""
    from .graph_store import graph_lineage

    return graph_lineage.sweep_orphaned_docs_revisions(
        *args, services=_graph_foundation.services, **kwargs
    )


def _derived_parent_source_metadata(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._derived_parent_source_metadata."""
    from .graph_store import graph_lineage

    return graph_lineage._derived_parent_source_metadata(
        *args, services=_graph_foundation.services, **kwargs
    )


def _validate_derived_parent_identity(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._validate_derived_parent_identity."""
    from .graph_store import graph_lineage

    return graph_lineage._validate_derived_parent_identity(
        *args, services=_graph_foundation.services, **kwargs
    )


def _eligible_derived_parent_unit_children(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._eligible_derived_parent_unit_children."""
    from .graph_store import graph_lineage

    return graph_lineage._eligible_derived_parent_unit_children(
        *args, services=_graph_foundation.services, **kwargs
    )


def _materialize_derived_parent_rows(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._materialize_derived_parent_rows."""
    from .graph_store import graph_lineage

    return graph_lineage._materialize_derived_parent_rows(
        *args, services=_graph_foundation.services, **kwargs
    )


_ParentMaterializationCapture = import_module(
    ".graph_store.graph_lineage", __package__
)._ParentMaterializationCapture


def _materialize_derived_parent_rows_batched(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._materialize_derived_parent_rows_batched."""
    from .graph_store import graph_lineage

    return graph_lineage._materialize_derived_parent_rows_batched(
        *args, services=_graph_foundation.services, **kwargs
    )


_normalization_peel_unit_repair_apply = import_module(
    ".graph_store.graph_lineage", __package__
)._normalization_peel_unit_repair_apply

globals()["repair_normalization_peel_parent_" + "units"] = (
    _normalization_peel_unit_repair_apply
)


def stamp_standard_name_units(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.stamp_standard_name_units."""
    from .graph_store import graph_lineage

    return graph_lineage.stamp_standard_name_units(
        *args, services=_graph_foundation.services, **kwargs
    )


def seed_parent_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.seed_parent_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.seed_parent_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def normalize_derived_parent_lifecycle(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.normalize_derived_parent_lifecycle."""
    from .graph_store import graph_lineage

    return graph_lineage.normalize_derived_parent_lifecycle(
        *args, services=_graph_foundation.services, **kwargs
    )


LIFECYCLELESS_STUB_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_lineage", __package__
).LIFECYCLELESS_STUB_MANIFEST_SCHEMA

LIFECYCLELESS_STUB_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_lineage", __package__
).LIFECYCLELESS_STUB_RECEIPT_SCHEMA

_LIFECYCLELESS_STUB_LOCK_QUERY = import_module(
    ".graph_store.graph_lineage", __package__
)._LIFECYCLELESS_STUB_LOCK_QUERY


LifecyclelessStubConflict = import_module(
    ".graph_store.graph_lineage", __package__
).LifecyclelessStubConflict


_READ_LIFECYCLELESS_STUBS_QUERY = import_module(
    ".graph_store.graph_lineage", __package__
)._READ_LIFECYCLELESS_STUBS_QUERY


def _canonicalize_lifecycleless_manifest_value(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._canonicalize_lifecycleless_manifest_value."""
    from .graph_store import graph_lineage

    return graph_lineage._canonicalize_lifecycleless_manifest_value(
        *args, services=_graph_foundation.services, **kwargs
    )


def _derived_parent_unit_from_eligible_children(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._derived_parent_unit_from_eligible_children."""
    from .graph_store import graph_lineage

    return graph_lineage._derived_parent_unit_from_eligible_children(
        *args, services=_graph_foundation.services, **kwargs
    )


def _spurious_stub_binding_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._spurious_stub_binding_authority."""
    from .graph_store import graph_lineage

    return graph_lineage._spurious_stub_binding_authority(
        *args, services=_graph_foundation.services, **kwargs
    )


def _partition_lifecycleless_stub_rows(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._partition_lifecycleless_stub_rows."""
    from .graph_store import graph_lineage

    return graph_lineage._partition_lifecycleless_stub_rows(
        *args, services=_graph_foundation.services, **kwargs
    )


def _lifecycleless_stub_manifest(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._lifecycleless_stub_manifest."""
    from .graph_store import graph_lineage

    return graph_lineage._lifecycleless_stub_manifest(
        *args, services=_graph_foundation.services, **kwargs
    )


def _reconcile_spurious_stub_source_scalars(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._reconcile_spurious_stub_source_scalars."""
    from .graph_store import graph_lineage

    return graph_lineage._reconcile_spurious_stub_source_scalars(
        *args, services=_graph_foundation.services, **kwargs
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


_SKELETON_PLACEHOLDER_PREDICATE = import_module(
    ".graph_store.graph_lineage", __package__
)._SKELETON_PLACEHOLDER_PREDICATE


def _query_skeleton_placeholders_for_cleanup(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_skeleton_placeholders_for_cleanup."""
    from .graph_store import graph_lineage

    return graph_lineage._query_skeleton_placeholders_for_cleanup(
        *args, services=_graph_foundation.services, **kwargs
    )


def write_standard_names(
    names: list[dict[str, Any]],
    *,
    override: bool = False,
    gc: Any | None = None,
    return_written_ids: bool = False,
) -> int | list[str]:
    """MERGE StandardName nodes with HAS_STANDARD_NAME relationships.

    Relationship direction: entity → concept
      (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
      (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)

    Each dict in *names* must have at least:
      - ``id``: the composed standard name string
      - ``source_types``: ["dd"] or ["signals"] etc.
      - ``source_id``: the originating path / signal ID

    Optional fields: ``unit``, ``description``,
    ``documentation``, ``kind``, ``links``, ``source_paths``,
    ``validity_domain``, ``constraints``, ``model``,
    ``generated_at``,
    ``review_tier``,
    ``vocab_gap_detail``, ``validation_issues``,
    ``validation_layer_summary``, ``cocos_transformation_type``, ``dd_version``,
    ``isn_version``, ``codex_version``,
    ``review_input_hash``.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection — write protected fields
        even on catalog-edited names.
    gc:
        Optional caller-owned graph query handle. All writes use this handle,
        allowing an explicit transaction to own the complete rich write.
    return_written_ids:
        Return the identities that survived validation and conflict checks.

    Performs conflict detection on ``unit``: if a StandardName already exists
    with a different unit value, that entry is skipped (not written)
    and a warning is logged.

    Returns the number of nodes written.
    """
    if not names:
        return [] if return_written_ids else 0

    # Pipeline protection — strip catalog-owned fields from catalog_edit items
    from imas_codex.standard_names.protection import filter_protected

    names, skipped = filter_protected(names, override=override)
    if skipped:
        logger.warning(
            "write_standard_names: stripped protected fields from %d catalog-edited name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    if not names:
        return [] if return_written_ids else 0

    # Strict grammar identity gate — reject names that ISN cannot parse and
    # losslessly recompose.  The legacy flat projection is not an authority:
    # it intentionally cannot represent every valid ordered operator tree.
    # This is a hard invariant: if a name fails grammar round-trip, it is
    # invalid by definition and must not enter the graph.
    try:
        from imas_standard_names import compose, parse
    except ImportError:
        parse = None  # type: ignore[assignment]

    if parse is not None:
        valid_names: list[dict[str, Any]] = []
        for n in names:
            try:
                parsed = parse(n["id"], strict=True)
                if compose(parsed.ir) != n["id"]:
                    raise ValueError("strict grammar round-trip changed the name")
                valid_names.append(n)
            except Exception as exc:
                logger.warning(
                    "write_standard_names: rejecting '%s' — grammar parse failed: %s",
                    n["id"],
                    str(exc)[:120],
                )
        if len(valid_names) < len(names):
            logger.info(
                "Grammar gate: %d/%d names passed parse validation",
                len(valid_names),
                len(names),
            )
        names = valid_names
        if not names:
            return [] if return_written_ids else 0

    # A COCOS-dependent name (cocos_transformation_type set) must carry the
    # integer cocos convention so the HAS_COCOS edge below is created. The
    # catalog follows a single convention — the current DD version's (DDv4 →
    # COCOS 17) — so default it from the current DD whenever compose/refine
    # didn't thread it through. COCOS-independent names stay unlinked by design.
    if any(
        n.get("cocos_transformation_type") and n.get("cocos") is None for n in names
    ):
        with nullcontext(gc) if gc is not None else GraphClient() as _conv_gc:
            _conv_row = _conv_gc.query(
                "MATCH (v:DDVersion {is_current: true}) RETURN v.cocos AS cocos"
            )
        _convention = (
            _conv_row[0]["cocos"]
            if _conv_row and _conv_row[0].get("cocos") is not None
            else None
        )
        if _convention is not None:
            for n in names:
                if n.get("cocos_transformation_type") and n.get("cocos") is None:
                    n["cocos"] = _convention

    # Guard: warn when cocos_transformation_type is set but cocos integer is
    # still missing (e.g. no current DD convention resolvable).
    for n in names:
        if n.get("cocos_transformation_type") and n.get("cocos") is None:
            logger.warning(
                "StandardName '%s' has cocos_transformation_type='%s' but no cocos "
                "integer — HAS_COCOS edge will not be created",
                n["id"],
                n["cocos_transformation_type"],
            )

    if gc is None and isinstance(GraphClient, type):
        with GraphClient() as client:
            return write_standard_names(
                names,
                override=override,
                gc=client,
                return_written_ids=return_written_ids,
            )
    if gc is not None and type(gc) is GraphClient:
        from imas_codex.standard_names.signed_manifest import _TransactionQuery

        with gc.session() as session:
            transaction = session.begin_transaction()
            try:
                result = write_standard_names(
                    names,
                    override=override,
                    gc=_TransactionQuery(transaction),
                    return_written_ids=return_written_ids,
                )
                transaction.commit()
                return result
            except BaseException:
                transaction.rollback()
                raise

    with nullcontext(gc) if gc is not None else GraphClient() as write_gc:
        # Conflict-detect on unit — same name with different unit is a data
        # integrity error.  Filter out conflicting entries rather than raising
        # so that non-conflicting entries can still proceed.
        unit_check_batch = [
            {"id": n["id"], "unit": n.get("unit")} for n in names if n.get("unit")
        ]
        if unit_check_batch:
            unit_conflicts = list(
                write_gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.id})
                    WHERE sn.unit IS NOT NULL AND b.unit IS NOT NULL
                      AND sn.unit <> b.unit
                    RETURN sn.id AS name,
                           sn.unit AS existing_unit,
                           b.unit AS incoming_unit
                    """,
                    batch=unit_check_batch,
                )
                or []
            )
            if unit_conflicts:
                conflict_details = "; ".join(
                    f"{c['name']}: {c['existing_unit']} vs {c['incoming_unit']}"
                    for c in unit_conflicts
                )
                logger.warning("Unit conflicts detected: %s", conflict_details)
                conflicting_ids = {c["name"] for c in unit_conflicts}
                names = [n for n in names if n["id"] not in conflicting_ids]
                if not names:
                    logger.warning("All entries had unit conflicts — nothing to write")
                    return [] if return_written_ids else 0

        # MERGE StandardName nodes with provenance — coalesce to preserve existing data
        write_gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.id})
            ON CREATE SET sn.status = 'draft', sn.updated_at = datetime()
            SET sn.updated_at = datetime(),
                // A null catalog status is outside the enumerated vocabulary
                // for a pipeline-minted identity; only fill the gap, never
                // overwrite a non-null status this write did not set.
                sn.status = coalesce(sn.status, 'draft'),
                sn.source_types = coalesce(b.source_types, sn.source_types),
                sn.description = coalesce(nullIf(b.description, ''), sn.description),
                sn.documentation = coalesce(b.documentation, sn.documentation),
                sn.kind = coalesce(b.kind, sn.kind),
                sn.links = coalesce(b.links, sn.links),
                sn.source_paths = coalesce(b.source_paths, sn.source_paths),
                sn.validity_domain = coalesce(b.validity_domain, sn.validity_domain),
                sn.constraints = coalesce(b.constraints, sn.constraints),
                sn.unit = coalesce(b.unit, sn.unit),
                sn.cocos_transformation_type = coalesce(b.cocos_transformation_type, sn.cocos_transformation_type),
                sn.cocos = coalesce(b.cocos, sn.cocos),
                sn.dd_version = coalesce(b.dd_version, sn.dd_version),
                sn.isn_version = coalesce(b.isn_version, sn.isn_version),
                sn.codex_version = coalesce(b.codex_version, sn.codex_version),
                sn.model = coalesce(b.model, sn.model),
                sn.generated_at = coalesce(b.generated_at, sn.generated_at),
                sn.review_tier = coalesce(b.review_tier, sn.review_tier),
                sn.vocab_gap_detail = coalesce(b.vocab_gap_detail, sn.vocab_gap_detail),
                sn.validation_issues = coalesce(b.validation_issues, sn.validation_issues),
                sn.validation_layer_summary = coalesce(b.validation_layer_summary, sn.validation_layer_summary),
                sn.validation_status = coalesce(b.validation_status, sn.validation_status),
                sn.validated_at = CASE WHEN b.validation_status IS NOT NULL
                                  THEN datetime()
                                  ELSE sn.validated_at END,
                sn.link_status = coalesce(b.link_status, sn.link_status),
                sn.review_input_hash = b.review_input_hash,
                sn.embedding = coalesce(b.embedding, sn.embedding),
                sn.embedded_at = coalesce(b.embedded_at, sn.embedded_at),
                sn.grammar_parse_version = b.grammar_parse_version,
                sn.validation_diagnostics_json = b.validation_diagnostics_json,
                sn.physical_base = b.physical_base,
                sn.geometric_base = b.geometric_base,
                sn.subject = b.subject,
                sn.component = b.component,
                sn.coordinate = b.coordinate,
                sn.transformation = b.transformation,
                sn.position = b.position,
                sn.process = b.process,
                sn.device = b.device,
                sn.region = b.region,
                sn.aggregation = b.aggregation,
                sn.orbit = b.orbit,
                sn.population = b.population,
                sn.object = b.object,
                sn.geometry = b.geometry,
                sn.llm_cost_refine_name = CASE WHEN sn.generate_name_count IS NOT NULL
                                             AND sn.generate_name_count > 0
                                             AND b.llm_cost IS NOT NULL
                                        THEN coalesce(sn.llm_cost_refine_name, 0.0) + b.llm_cost
                                        ELSE sn.llm_cost_refine_name END,
                sn.refine_name_count = CASE WHEN sn.generate_name_count IS NOT NULL
                                            AND sn.generate_name_count > 0
                                            AND b.llm_cost IS NOT NULL
                                       THEN coalesce(sn.refine_name_count, 0) + 1
                                       ELSE sn.refine_name_count END,
                sn.llm_cost_generate_name = CASE WHEN b.llm_cost IS NOT NULL
                                      THEN coalesce(sn.llm_cost_generate_name, 0.0) + b.llm_cost
                                      ELSE sn.llm_cost_generate_name END,
                sn.generate_name_count = CASE WHEN b.llm_cost IS NOT NULL
                                   THEN coalesce(sn.generate_name_count, 0) + 1
                                   ELSE sn.generate_name_count END,
                sn.regen_count = CASE WHEN b.regen_increment = true
                                 THEN coalesce(sn.regen_count, 0) + 1
                                 ELSE sn.regen_count END,
                sn.llm_cost = CASE WHEN b.llm_cost IS NOT NULL
                              THEN coalesce(sn.llm_cost, 0.0) + b.llm_cost
                              ELSE sn.llm_cost END,
                sn.llm_model = coalesce(b.llm_model, sn.llm_model),
                sn.llm_service = coalesce(b.llm_service, sn.llm_service),
                sn.llm_at = coalesce(b.llm_at, sn.llm_at),
                sn.llm_tokens_in = coalesce(b.llm_tokens_in, sn.llm_tokens_in),
                sn.llm_tokens_out = coalesce(b.llm_tokens_out, sn.llm_tokens_out),
                sn.llm_tokens_cached_read = coalesce(b.llm_tokens_cached_read, sn.llm_tokens_cached_read),
                sn.llm_tokens_cached_write = coalesce(b.llm_tokens_cached_write, sn.llm_tokens_cached_write),
                sn.created_at = coalesce(sn.created_at, datetime())
            """,
            batch=[
                {
                    "id": n["id"],
                    "source_types": n.get("source_types") or None,
                    "description": n.get("description"),
                    "documentation": n.get("documentation"),
                    "kind": n.get("kind"),
                    "links": n.get("links") or None,
                    "source_paths": n.get("source_paths") or None,
                    "validity_domain": n.get("validity_domain"),
                    "constraints": n.get("constraints") or None,
                    "unit": n.get("unit"),
                    "cocos_transformation_type": n.get("cocos_transformation_type"),
                    "cocos": n.get("cocos"),
                    "dd_version": n.get("dd_version"),
                    "isn_version": n.get("isn_version"),
                    "codex_version": n.get("codex_version"),
                    "model": n.get("model"),
                    "generated_at": n.get("generated_at"),
                    "review_tier": n.get("review_tier"),
                    "vocab_gap_detail": _ensure_json(n.get("vocab_gap_detail")),
                    "validation_issues": n.get("validation_issues") or None,
                    "validation_layer_summary": _ensure_json(
                        n.get("validation_layer_summary")
                    ),
                    "validation_status": n.get("validation_status"),
                    "link_status": _compute_link_status(n.get("links")),
                    "review_input_hash": n.get("review_input_hash"),
                    "embedding": n.get("embedding"),
                    "embedded_at": n.get("embedded_at"),
                    "llm_cost": n.get("llm_cost"),
                    "llm_model": n.get("llm_model"),
                    "llm_service": n.get("llm_service"),
                    "llm_at": n.get("llm_at"),
                    "llm_tokens_in": n.get("llm_tokens_in"),
                    "llm_tokens_out": n.get("llm_tokens_out"),
                    "llm_tokens_cached_read": n.get("llm_tokens_cached_read"),
                    "llm_tokens_cached_write": n.get("llm_tokens_cached_write"),
                    "regen_increment": n.get("regen_increment"),
                    **_parse_grammar(n["id"]),
                }
                for n in names
            ],
        )

        # Promote-on-higher-rank: read existing scalar primary +
        # source_domains, compute promotion in Python, write back.
        # Replaces the legacy list-append-with-dedupe semantics.
        from imas_codex.standard_names.domain_ranking import (
            maybe_promote_domain,
            merge_source_domains,
        )

        pd_inputs = []
        for n in names:
            sn_id = n.get("id")
            if not sn_id:
                continue
            candidate = _scalar_domain(n.get("physics_domain"))
            incoming_sources = _ensure_list(n.get("source_domains"))
            if candidate and candidate not in incoming_sources:
                incoming_sources = [*incoming_sources, candidate]
            if not candidate and not incoming_sources:
                continue
            pd_inputs.append(
                {
                    "id": sn_id,
                    "candidate": candidate,
                    "incoming_sources": incoming_sources,
                }
            )

        if pd_inputs:
            # Read existing values for this batch so we can compute
            # promotion in Python (rank tables are not encodable in Cypher).
            existing_rows = (
                write_gc.query(
                    """
                    UNWIND $ids AS sid
                    MATCH (sn:StandardName {id: sid})
                    RETURN sn.id AS id,
                           sn.physics_domain AS physics_domain,
                           sn.source_domains AS source_domains
                    """,
                    ids=[p["id"] for p in pd_inputs],
                )
                or []
            )
            existing_map = {
                row["id"]: row
                for row in existing_rows  # type: ignore[index]
            }

            pd_batch = []
            for p in pd_inputs:
                row = existing_map.get(p["id"], {})
                existing_primary = _scalar_domain(row.get("physics_domain"))
                existing_sources = _ensure_list(row.get("source_domains"))
                # Legacy nodes may have list-valued physics_domain — fold
                # those into source_domains too.
                legacy_list = (
                    row.get("physics_domain")
                    if isinstance(row.get("physics_domain"), list)
                    else None
                )
                if legacy_list:
                    existing_sources = list(
                        dict.fromkeys([*existing_sources, *legacy_list])
                    )

                merged_sources = merge_source_domains(
                    existing_sources, *p["incoming_sources"]
                )
                promoted = maybe_promote_domain(existing_primary, p["candidate"])
                pd_batch.append(
                    {
                        "id": p["id"],
                        "physics_domain": promoted,
                        "source_domains": merged_sources or None,
                    }
                )

            write_gc.query(
                """
                UNWIND $batch AS b
                MERGE (sn:StandardName {id: b.id})
                SET sn.updated_at = datetime(), sn.physics_domain = coalesce(b.physics_domain, sn.physics_domain),
                    sn.source_domains = coalesce(b.source_domains, sn.source_domains)
                """,
                batch=pd_batch,
            )

        # Create HAS_STANDARD_NAME relationships: entity → concept
        dd_names = [n for n in names if "dd" in (n.get("source_types") or [])]
        signal_names = [n for n in names if "signals" in (n.get("source_types") or [])]

        if dd_names:
            from imas_codex.standard_names.workers import _is_attachment_consistent

            pairing_rows = write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                OPTIONAL MATCH (src:IMASNode {id: b.source_id})
                OPTIONAL MATCH (src)-[:HAS_UNIT]->(du:Unit)
                OPTIONAL MATCH (bound:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
                RETURN b.id AS id, b.source_id AS source_id,
                       coalesce(du.id, src.unit) AS dd_unit,
                       sn.unit AS sn_unit,
                       collect(DISTINCT bound.id) AS existing_dd_paths
                """,
                batch=[
                    {"id": n["id"], "source_id": n["source_id"]}
                    for n in dd_names
                    if n.get("source_id")
                ],
            )
            accepted_by_name: dict[str, list[str]] = {}
            accepted_pairs: list[dict[str, str]] = []
            for row in sorted(
                pairing_rows, key=lambda item: (item["id"], item["source_id"])
            ):
                accepted_paths = accepted_by_name.setdefault(
                    row["id"], sorted(row.get("existing_dd_paths") or [])
                )
                accepted, reason = _is_attachment_consistent(
                    row["source_id"],
                    row["id"],
                    existing_sources=tuple(accepted_paths),
                    dd_unit=row.get("dd_unit"),
                    sn_unit=row.get("sn_unit"),
                )
                if accepted:
                    accepted_pairs.append(
                        {"id": row["id"], "source_id": row["source_id"]}
                    )
                    accepted_paths.append(row["source_id"])
                else:
                    logger.warning(
                        "write_standard_names: rejected DD pairing %s → %s: %s",
                        row["source_id"],
                        row["id"],
                        reason,
                    )
            write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                """,
                batch=accepted_pairs,
            )
        if signal_names:
            write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (src:FacilitySignal {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                """,
                batch=[
                    {"id": n["id"], "source_id": n["source_id"]}
                    for n in signal_names
                    if n.get("source_id")
                ],
            )

        # Create HAS_UNIT relationships: StandardName → Unit
        #
        # Self-healing: for each SN carrying a unit in this batch, drop ALL
        # existing HAS_UNIT edges before re-creating, so a unit correction
        # (e.g. m.sr → m^2.sr) replaces the edge rather than leaving a stale
        # one alongside the new one. An SN must have at most ONE HAS_UNIT edge,
        # matching its sn.unit property. Mirrors the IMASNode self-heal in
        # build_dd.py. Scoped to the batch's SN ids that carry a unit — an SN
        # not supplying a unit in this batch (e.g. a docs-only refine) keeps
        # its existing edge untouched.
        units_batch = [
            {"id": n["id"], "unit": n["unit"]} for n in names if n.get("unit")
        ]
        if units_batch:
            unit_sn_ids = [b["id"] for b in units_batch]
            write_gc.query(
                """
                UNWIND $ids AS id
                MATCH (sn:StandardName {id: id})-[r:HAS_UNIT]->()
                DELETE r
                """,
                ids=unit_sn_ids,
            )
            write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MERGE (u:Unit {id: b.unit})
                SET u.symbol = coalesce(u.symbol, b.unit)
                MERGE (sn)-[:HAS_UNIT]->(u)
                """,
                batch=units_batch,
            )

        # Create HAS_COCOS relationships: StandardName → COCOS
        # Use MATCH (not MERGE) — COCOS singleton nodes already exist.
        cocos_batch = [
            {"id": n["id"], "cocos": n["cocos"]}
            for n in names
            if n.get("cocos") is not None
        ]
        if cocos_batch:
            write_gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (c:COCOS {id: b.cocos})
                MERGE (sn)-[:HAS_COCOS]->(c)
                """,
                batch=cocos_batch,
            )

        # Create grammar decomposition: typed edges + per-segment columns
        token_miss_gaps = _write_grammar_decomposition(
            write_gc, [n["id"] for n in names]
        )

        # Emit structural edges: HAS_PARENT, HAS_ERROR, HAS_PREDECESSOR,
        # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN.
        # Tail pass — all nodes in this batch exist before edges are written.
        skeleton_candidate_ids = _write_standard_name_edges(write_gc, names)

        from imas_codex.standard_names.source_invariant import require_producing_sources

        require_producing_sources(write_gc, [n["id"] for n in names])

    # Persist token-miss gaps through the same caller-owned graph handle when
    # one is supplied, so transactional callers keep every rich-write effect.
    if token_miss_gaps:
        # Build sn_id → source mapping from the names list
        sn_source_map: dict[str, tuple[str, str]] = {}
        for n in names:
            sn_id = n["id"]
            source_id = n.get("source_id")
            source_type = "dd" if "dd" in (n.get("source_types") or []) else "signals"
            if source_id and sn_id not in sn_source_map:
                sn_source_map[sn_id] = (source_id, source_type)

        # Group gaps by source_type for write_vocab_gaps
        dd_gap_dicts: list[dict[str, str]] = []
        signal_gap_dicts: list[dict[str, str]] = []
        for gap in token_miss_gaps:
            mapping = sn_source_map.get(gap["sn_id"])
            if not mapping:
                continue
            source_id, source_type = mapping
            gap_dict = {
                "source_id": source_id,
                "segment": gap["segment"],
                "token": gap["token"],
                "reason": f"Token-miss during grammar edge writing for '{gap['sn_id']}'",
            }
            if source_type == "dd":
                dd_gap_dicts.append(gap_dict)
            else:
                signal_gap_dicts.append(gap_dict)

        if dd_gap_dicts:
            write_vocab_gaps(dd_gap_dicts, source_type="dd", gc=gc)
        if signal_gap_dicts:
            write_vocab_gaps(signal_gap_dicts, source_type="signals", gc=gc)

    # Sweep skeleton placeholders created by this write's relationship-side
    # MERGE on uncomposed targets (HAS_PARENT, HAS_ERROR, HAS_PREDECESSOR,
    # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN). A real StandardName
    # always has at least a created_at OR generated_at timestamp; pure
    # skeletons (id-only) are detached and deleted.
    # Reuse a caller-owned transaction when present; ordinary callers retain
    # the independent graph context used by the public writer. An empty
    # candidate set performs no query: it must never degrade into a global
    # placeholder search.
    swept_count = 0
    if skeleton_candidate_ids:
        with nullcontext(gc) if gc is not None else GraphClient() as sweep_gc:
            skeleton_placeholder_ids = _query_skeleton_placeholders_for_cleanup(
                sweep_gc, skeleton_candidate_ids
            )
            if skeleton_placeholder_ids:
                refuse_protected_automatic_deletion(
                    sweep_gc,
                    skeleton_placeholder_ids,
                    operation="skeleton placeholder cleanup",
                )
                deletion_clause = deletion_change_cypher("sn")
                swept = sweep_gc.query(
                    f"""
                    MATCH (sn:StandardName)
                    WHERE sn.id IN $candidate_ids
                      AND {_SKELETON_PLACEHOLDER_PREDICATE}
                    {deletion_clause}
                    DETACH DELETE sn
                    RETURN count(sn) AS swept
                    """,
                    candidate_ids=skeleton_placeholder_ids,
                    **deletion_change_params(
                        "remove_skeleton_placeholder",
                        reason="relationship placeholder never became a composed name",
                    ),
                )
                swept_count = int(swept[0]["swept"]) if swept else 0
    if swept_count:
        logger.info("Swept %d skeleton StandardName placeholder(s)", swept_count)

    written = len(names)
    logger.info("Wrote %d StandardName nodes", written)
    if return_written_ids:
        return [name["id"] for name in names]
    return written


class ReviewWriteResult(int):
    """Written review count with suggestion-clearing measurements."""

    suggestions_cleared_non_distinct: int
    suggestions_cleared_unparseable: int

    def __new__(
        cls,
        written: int,
        *,
        suggestions_cleared_non_distinct: int = 0,
        suggestions_cleared_unparseable: int = 0,
    ) -> ReviewWriteResult:
        result = super().__new__(cls, written)
        result.suggestions_cleared_non_distinct = suggestions_cleared_non_distinct
        result.suggestions_cleared_unparseable = suggestions_cleared_unparseable
        return result


def write_reviews(
    records: list[dict[str, Any]], *, skip_cost: bool = False
) -> ReviewWriteResult:
    """MERGE ``StandardNameReview`` nodes and ``HAS_REVIEW`` edges from StandardName.

    Each record must contain:

    - ``id`` (str) — composite key
      ``{standard_name_id}:{axis}:{review_group_id}:{cycle_index}``
    - ``standard_name_id`` (str) — parent StandardName
    - ``model`` (str), ``model_family`` (str), ``is_canonical`` (bool)
    - ``score`` (float 0-1), ``scores_json`` (str), ``tier`` (str)
    - ``reviewed_at`` (str ISO 8601)

    RD-quorum fields (required for new-style reviews):
    - ``review_axis`` (str) — "name" or "docs"
    - ``cycle_index`` (int) — 0, 1, or 2
    - ``review_group_id`` (str) — UUID
    - ``resolution_role`` (str) — "primary", "secondary", or "escalator"
    - ``resolution_method`` (str | None)

    Optional: ``comments`` (str), ``llm_cost`` (float),
    ``llm_tokens_in`` (int), ``llm_tokens_out`` (int),
    ``llm_model`` (str), ``llm_at`` (str), ``llm_service`` (str).

    Version provenance (auto-injected if not provided):
    ``codex_version`` (str), ``isn_version`` (str).

    MERGE-by-``id`` semantics make re-runs idempotent when the same
    model reviews the same name at the same timestamp.

    Parameters
    ----------
    records:
        Review record dicts.
    skip_cost:
        When ``True``, skip accumulating ``llm_cost_review_name`` /
        ``llm_cost_review_docs`` on StandardName nodes.  Use when StandardNameReview records were
        already persisted inline (crash-safety path) to avoid
        double-counting.

    Returns the number of StandardNameReview records written, with counts of
    suggestions cleared as non-distinct or unparseable on the integer result.
    """
    if not records:
        return ReviewWriteResult(0)
    # Guard: must attach to an existing StandardName.
    valid = [r for r in records if r.get("id") and r.get("standard_name_id")]
    if not valid:
        return ReviewWriteResult(0)

    from imas_standard_names import compose, parse

    from imas_codex.standard_names.models import suggestion_is_semantically_distinct

    suggestions_cleared_non_distinct = 0
    suggestions_cleared_unparseable = 0
    prepared: list[dict[str, Any]] = []
    for record in valid:
        proposed_spelling = record.get("suggested_name")
        if proposed_spelling and not suggestion_is_semantically_distinct(
            str(record["standard_name_id"]), str(proposed_spelling)
        ):
            try:
                compose(parse(str(proposed_spelling)).ir)
            except Exception:
                suggestions_cleared_unparseable += 1
            else:
                suggestions_cleared_non_distinct += 1
            record = {
                **record,
                "suggested_name": None,
                "suggestion_justification": None,
            }
        prepared.append(record)
    valid = prepared

    # Auto-inject codex and ISN versions (same pattern as persist_generated_name_batch)
    try:
        import imas_standard_names

        _isn_ver = imas_standard_names.__version__
    except (ImportError, AttributeError):
        _isn_ver = None
    try:
        import importlib.metadata

        _codex_ver = importlib.metadata.version("imas-codex")
    except Exception:
        _codex_ver = None
    for r in valid:
        r.setdefault("isn_version", _isn_ver)
        r.setdefault("codex_version", _codex_ver)

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (r:StandardNameReview {id: b.id})
            SET r.standard_name_id = b.standard_name_id,
                r.model = b.model,
                r.reviewer_model = b.reviewer_model,
                r.model_family = b.model_family,
                r.is_canonical = b.is_canonical,
                r.score = b.score,
                r.judge_kind = b.judge_kind,
                r.judgment_model = b.judgment_model,
                r.judge_answers_json = b.judge_answers_json,
                r.judge_probs_json = b.judge_probs_json,
                r.judge_confidence_json = b.judge_confidence_json,
                r.scores_json = b.scores_json,
                r.tier = b.tier,
                r.comments = b.comments,
                r.comments_per_dim_json = b.comments_per_dim_json,
                r.suggested_name = b.suggested_name,
                r.suggestion_justification = b.suggestion_justification,
                r.reviewed_at = CASE
                    WHEN b.reviewed_at IS NULL THEN NULL
                    ELSE datetime(b.reviewed_at)
                END,
                r.review_axis = b.review_axis,
                r.cycle_index = b.cycle_index,
                r.review_group_id = b.review_group_id,
                r.resolution_role = b.resolution_role,
                r.resolution_method = b.resolution_method,
                r.llm_model = b.llm_model,
                r.llm_cost = b.llm_cost,
                r.llm_tokens_in = b.llm_tokens_in,
                r.llm_tokens_out = b.llm_tokens_out,
                r.llm_tokens_cached_read = b.llm_tokens_cached_read,
                r.llm_tokens_cached_write = b.llm_tokens_cached_write,
                r.llm_at = CASE
                    WHEN b.llm_at IS NULL THEN NULL
                    ELSE datetime(b.llm_at)
                END,
                r.llm_service = b.llm_service,
                r.codex_version = b.codex_version,
                r.isn_version = b.isn_version
            WITH r, b
            MATCH (sn:StandardName {id: b.standard_name_id})
            MERGE (sn)-[:HAS_REVIEW]->(r)
            """,
            batch=[
                {
                    "id": r["id"],
                    "standard_name_id": r["standard_name_id"],
                    "model": r.get("model") or "",
                    # reviewer_model is the consumer-facing alias; fall back to model
                    "reviewer_model": r.get("reviewer_model") or r.get("model") or "",
                    "model_family": r.get("model_family") or "other",
                    "is_canonical": bool(r.get("is_canonical", False)),
                    "score": None
                    if r.get("judge_kind") == "jev"
                    else float(r.get("score") or 0.0),
                    "judge_kind": r.get("judge_kind"),
                    "judgment_model": r.get("judgment_model"),
                    "judge_answers_json": _ensure_json(r.get("judge_answers_json")),
                    "judge_probs_json": _ensure_json(r.get("judge_probs_json")),
                    "judge_confidence_json": _ensure_json(
                        r.get("judge_confidence_json")
                    ),
                    "scores_json": _ensure_json(r.get("scores_json") or "{}"),
                    "tier": r.get("tier") or "unknown",
                    "comments": r.get("comments") or "",
                    "comments_per_dim_json": _ensure_json(
                        r.get("comments_per_dim_json")
                    ),
                    "suggested_name": r.get("suggested_name") or "",
                    "suggestion_justification": r.get("suggestion_justification") or "",
                    "reviewed_at": r.get("reviewed_at"),
                    "review_axis": r.get("review_axis"),
                    "cycle_index": r.get("cycle_index"),
                    "review_group_id": r.get("review_group_id"),
                    "resolution_role": r.get("resolution_role"),
                    "resolution_method": r.get("resolution_method"),
                    "llm_model": r.get("llm_model"),
                    # Defensive: coalesce None → 0 so every Review has
                    # non-NULL cost/token fields (quarantine & cache-hit).
                    "llm_cost": r.get("llm_cost")
                    if r.get("llm_cost") is not None
                    else 0.0,
                    "llm_tokens_in": r.get("llm_tokens_in")
                    if r.get("llm_tokens_in") is not None
                    else 0,
                    "llm_tokens_out": r.get("llm_tokens_out")
                    if r.get("llm_tokens_out") is not None
                    else 0,
                    "llm_tokens_cached_read": r.get("llm_tokens_cached_read")
                    if r.get("llm_tokens_cached_read") is not None
                    else 0,
                    "llm_tokens_cached_write": r.get("llm_tokens_cached_write")
                    if r.get("llm_tokens_cached_write") is not None
                    else 0,
                    "llm_at": r.get("llm_at"),
                    "llm_service": r.get("llm_service"),
                    "codex_version": r.get("codex_version"),
                    "isn_version": r.get("isn_version"),
                }
                for r in valid
            ],
        )

        # --- Accumulate review cost on StandardName ---
        # Build per-SN cost totals from the batch, split by review axis.
        # Skipped when skip_cost=True (records already persisted inline).
        if not skip_cost:
            name_cost_map: dict[str, float] = {}
            docs_cost_map: dict[str, float] = {}
            for r in valid:
                sn_id = r.get("standard_name_id")
                cost = r.get("llm_cost")
                if sn_id and cost:
                    axis = r.get("review_axis", "name")
                    if axis == "docs":
                        docs_cost_map[sn_id] = docs_cost_map.get(sn_id, 0.0) + cost
                    else:
                        name_cost_map[sn_id] = name_cost_map.get(sn_id, 0.0) + cost
            if name_cost_map:
                gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.updated_at = datetime(), sn.llm_cost_review_name = coalesce(sn.llm_cost_review_name, 0.0) + b.cost,
                        sn.review_name_count = coalesce(sn.review_name_count, 0) + 1,
                        sn.llm_cost = coalesce(sn.llm_cost, 0.0) + b.cost
                    """,
                    batch=[
                        {"sn_id": sn_id, "cost": cost}
                        for sn_id, cost in name_cost_map.items()
                    ],
                )
            if docs_cost_map:
                gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.updated_at = datetime(), sn.llm_cost_review_docs = coalesce(sn.llm_cost_review_docs, 0.0) + b.cost,
                        sn.review_docs_count = coalesce(sn.review_docs_count, 0) + 1,
                        sn.llm_cost = coalesce(sn.llm_cost, 0.0) + b.cost
                    """,
                    batch=[
                        {"sn_id": sn_id, "cost": cost}
                        for sn_id, cost in docs_cost_map.items()
                    ],
                )

    logger.info(
        "Wrote %d StandardNameReview nodes; cleared suggestions: "
        "non-distinct=%d, unparseable=%d",
        len(valid),
        suggestions_cleared_non_distinct,
        suggestions_cleared_unparseable,
    )
    return ReviewWriteResult(
        len(valid),
        suggestions_cleared_non_distinct=suggestions_cleared_non_distinct,
        suggestions_cleared_unparseable=suggestions_cleared_unparseable,
    )


def update_review_aggregates(
    standard_name_ids: list[str],
    *,
    threshold: float = 0.2,
) -> int:
    """Recompute per-StandardName aggregates from attached StandardNameReview nodes.

    Sets ``review_count``, ``review_mean_score``, and ``review_disagreement``
    across attached reviews except non-canonical Jev shadow judgments.
    Axis-specific score and resolution-method mirrors are owned by their
    dedicated write paths.

    Returns the number of StandardName nodes updated.
    """
    if not standard_name_ids:
        return 0
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE NOT (coalesce(sn.name_stage, '') IN $frozen_name_stages)
            OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
            WHERE NOT (coalesce(r.judge_kind, '') = 'jev'
                       AND coalesce(r.is_canonical, false) = false)
            WITH sn, count(r) AS n, avg(r.score) AS mean,
                 CASE WHEN count(r) > 1 THEN max(r.score) - min(r.score) ELSE 0.0 END AS spread
            SET sn.updated_at = datetime(), sn.review_count = n,
                sn.review_mean_score = CASE WHEN n > 0 THEN mean ELSE null END,
                sn.review_disagreement = (n > 1 AND spread >= $threshold)
            RETURN sn.id AS id
            """,
            ids=standard_name_ids,
            threshold=float(threshold),
            frozen_name_stages=list(_PIPELINE_FROZEN_NAME_STAGES),
        )
        return len(list(rows or []))


# ─────────────────────────────────────────────────────────────────────────────
# Voiding one review dimension
# ─────────────────────────────────────────────────────────────────────────────

_DIMENSION_SCALE = 20.0


class ReviewDimensionVoidRefused(ValueError):
    """A dimension void was refused; nothing was written."""


def _review_tier(score: float) -> str:
    """Tier label for a normalised score, matching the review persist path."""
    if score >= 0.85:
        return "outstanding"
    if score >= 0.60:
        return "good"
    if score >= 0.40:
        return "inadequate"
    return "poor"


def plan_review_dimension_void(
    *,
    scores: dict[str, Any],
    void_records: list[dict[str, Any]] | None,
    dimension: str,
    actor: str,
    reason: str,
    grammar_signature: str | None,
    at: str | None = None,
) -> dict[str, Any]:
    """Decide one dimension void without touching the graph.

    The review score is a stored mirror of a derivation — ``sum(dimensions) /
    (len(dimensions) * 20)`` at review time — so a void is expressible as a
    recomputation over the surviving dimensions. That is the whole reason this
    route can be honoured by consumers that know nothing about it: every reader
    of ``r.score`` reads the recomputed number.

    Refusals are decided here so that a caller writing only on success cannot
    half-apply one. What justifies a void is the caller's business; what this
    enforces is that a void is attributable, reasoned, non-duplicating, and
    never the whole review.

    The original per-dimension scores are NOT modified. ``scores`` is evidence
    and is returned unchanged; the void record sits beside it.

    Raises :class:`ReviewDimensionVoidRefused` for the four refusals — an
    absent dimension, the last surviving dimension, an empty reason, and a
    dimension already voided.
    """
    if not str(reason or "").strip():
        raise ReviewDimensionVoidRefused(
            f"voiding {dimension!r} needs a reason: the reason is the record, "
            "and a void without one is indistinguishable from a lost score"
        )
    if not str(actor or "").strip():
        raise ReviewDimensionVoidRefused(
            f"voiding {dimension!r} needs an actor: an unattributed void "
            "cannot be adjudicated later"
        )

    numeric = {
        key: float(value)
        for key, value in (scores or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if dimension not in numeric:
        raise ReviewDimensionVoidRefused(
            f"{dimension!r} is not a scored dimension of this review "
            f"(scored: {', '.join(sorted(numeric)) or 'none'})"
        )

    records = [dict(record) for record in (void_records or [])]
    already = [
        record
        for record in records
        if record.get("dimension") == dimension and record.get("void_active", True)
    ]
    if already:
        raise ReviewDimensionVoidRefused(
            f"{dimension!r} is already voided on this review "
            f"({already[0].get('void_reason')!r}) — one dimension carries one "
            "record, so a second void would only duplicate it"
        )

    voided_names = {
        record["dimension"]
        for record in records
        if record.get("void_active", True) and record.get("dimension") in numeric
    }
    surviving = {
        key: value for key, value in numeric.items() if key not in voided_names
    }
    surviving.pop(dimension, None)
    if not surviving:
        raise ReviewDimensionVoidRefused(
            f"{dimension!r} is the last surviving dimension of this review — a "
            "review with every dimension void is not evidence of anything, and "
            "discarding it is a whole-review decision, not a dimension void"
        )

    record = {
        "dimension": dimension,
        "void_active": True,
        "voided_at": at or datetime.now(UTC).isoformat(),
        "void_actor": str(actor),
        "void_reason": str(reason),
        "void_grammar_signature": grammar_signature,
    }
    records.append(record)
    score = (sum(surviving.values()) / len(surviving)) / _DIMENSION_SCALE
    return {
        "void_records": records,
        "void_record": record,
        "surviving_dimensions": dict(sorted(surviving.items())),
        "score": score,
        "tier": _review_tier(score),
    }


_VOID_REVIEW_DIMENSION_READ = """
MATCH (r:StandardNameReview {id: $review_id})
RETURN r.id AS id,
       r.standard_name_id AS standard_name_id,
       r.review_axis AS review_axis,
       r.score AS score,
       r.scores_json AS scores_json,
       r.voided_dimensions_json AS voided_dimensions_json
"""

# scores_json is deliberately absent from this SET: the original per-dimension
# score is evidence and survives the void untouched.
_VOID_REVIEW_DIMENSION_WRITE = """
MATCH (r:StandardNameReview {id: $review_id})
SET r.score = $score,
    r.tier = $tier,
    r.voided_dimensions_json = $voided_dimensions_json
RETURN r.id AS id
"""


def void_review_dimension(
    review_id: str,
    dimension: str,
    *,
    actor: str,
    reason: str,
    grammar_signature: str | None = None,
    gc: Any | None = None,
    dry_run: bool = False,
    refresh_aggregates: bool = True,
) -> dict[str, Any]:
    """Void ONE scored dimension of a review and record why in the graph.

    A dimension can be discredited without the rest of the review being wrong.
    Deleting it would destroy the evidence and leave the surviving score
    unexplained, so the dimension's number stays exactly where it was and gains
    a void record beside it carrying the actor, the reason, the timestamp and
    the vocabulary signature observed at that moment.

    ``reason`` is the caller's free text and carries the entire justification;
    this route asserts nothing about what discredits a dimension, and must not
    be read as encoding one cause. The signature and timestamp are recorded as
    facts about the environment at void time rather than as evidence for the
    decision — ``isn_version`` is build-time metadata from the installed
    distribution and reports what was packaged, not which vocabulary or prompt
    the review ran under, so neither question can be answered after the fact
    without capturing the answer here.

    The review's ``score`` is recomputed over the surviving dimensions. Because
    ``score`` was always a derivation of the per-dimension scores, that single
    write is what makes ``update_review_aggregates``, the canonical-review
    projection, the reviewer benchmarks and every other reader of ``r.score``
    honour the void — a marker nothing consults would be worse than no marker.

    ``review_mean_score`` is the one reader that cannot follow on its own: it is
    a STORED average, refreshed only when ``update_review_aggregates`` runs. So
    the void refreshes it for the reviewed name, leaving no window in which the
    review says one thing and the name-level aggregate another. Pass
    ``refresh_aggregates=False`` when voiding in bulk and aggregating once at
    the end.

    Also writes a ``StandardNameChange``, the same ledger a detach judgement
    writes, so the decision survives independently of the review node.

    Refuses — raising :class:`ReviewDimensionVoidRefused` with nothing written —
    when the dimension is not scored on this review, when it is the last
    surviving one, when no reason is given, and when it is already voided.
    """
    own = gc is None
    client: Any = GraphClient() if own else gc
    try:
        rows = list(client.query(_VOID_REVIEW_DIMENSION_READ, review_id=review_id))
        if not rows:
            raise ReviewDimensionVoidRefused(
                f"no StandardNameReview {review_id!r} — nothing to void"
            )
        row = dict(rows[0])
        try:
            scores = json.loads(row.get("scores_json") or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ReviewDimensionVoidRefused(
                f"review {review_id!r} has unreadable per-dimension scores"
            ) from exc
        try:
            existing = json.loads(row.get("voided_dimensions_json") or "[]")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ReviewDimensionVoidRefused(
                f"review {review_id!r} has an unreadable void record"
            ) from exc

        if grammar_signature is None:
            with suppress(Exception):
                grammar_signature = isn_vocabulary_signature()

        plan = plan_review_dimension_void(
            scores=scores,
            void_records=existing if isinstance(existing, list) else [],
            dimension=dimension,
            actor=actor,
            reason=reason,
            grammar_signature=grammar_signature,
        )
        result = {
            "ok": True,
            "review_id": review_id,
            "standard_name_id": row.get("standard_name_id"),
            "review_axis": row.get("review_axis"),
            "dimension": dimension,
            "score_before": row.get("score"),
            "score": plan["score"],
            "tier": plan["tier"],
            "surviving_dimensions": plan["surviving_dimensions"],
            "void_record": plan["void_record"],
            "dry_run": dry_run,
        }
        if dry_run:
            return result

        client.query(
            _VOID_REVIEW_DIMENSION_WRITE,
            review_id=review_id,
            score=plan["score"],
            tier=plan["tier"],
            voided_dimensions_json=json.dumps(plan["void_records"]),
        )

        sn_id = row.get("standard_name_id")
        if sn_id:
            from imas_codex.standard_names.provenance_lifecycle import (
                record_standard_name_change,
            )

            result["change_event_id"] = record_standard_name_change(
                client,
                sn_id,
                sn_id,
                operation="void_review_dimension",
                reason=(
                    f"{dimension} voided on review {review_id} by {actor}: {reason}"
                ),
                origin="review_dimension_void",
            )
            if refresh_aggregates:
                result["aggregates_refreshed"] = bool(update_review_aggregates([sn_id]))
    finally:
        if own:
            client.close()

    logger.info(
        "void_review_dimension: %s voided on %s by %s — score %s to %.6f (%s)",
        dimension,
        review_id,
        actor,
        result.get("score_before"),
        result["score"],
        reason,
    )
    return result


@functools.lru_cache(maxsize=1)
def _winning_review_resolution_methods(
    schema_path: Path | None = None,
) -> frozenset[str]:
    """Return the review methods the LinkML schema marks as winning.

    The enum description is deliberately the admission authority: newly added
    enum values remain excluded until the schema explicitly names them as
    eligible for winning-group slot mirroring.
    """
    import yaml

    if schema_path is None:
        schema_path = Path(__file__).parents[1] / "schemas" / "standard_name.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    enum = schema["enums"]["ReviewResolutionMethod"]
    permissible = frozenset(enum["permissible_values"])
    eligibility_text, marker, _ = str(enum.get("description") or "").partition(
        "are eligible as winning groups"
    )
    if not marker:
        raise RuntimeError(
            "ReviewResolutionMethod schema does not declare winning-group eligibility"
        )
    referenced = frozenset(re.findall(r"``([a-z][a-z0-9_]*)``", eligibility_text))
    winning = permissible & referenced
    if not winning:
        raise RuntimeError(
            "ReviewResolutionMethod schema declares no winning resolution methods"
        )
    return winning


@functools.lru_cache(maxsize=1)
def _non_winning_review_resolution_methods() -> frozenset[str]:
    """Return schema enum values not admitted as winning methods."""
    import yaml

    schema_path = Path(__file__).parents[1] / "schemas" / "standard_name.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    permissible = frozenset(
        schema["enums"]["ReviewResolutionMethod"]["permissible_values"]
    )
    return permissible - _winning_review_resolution_methods()


def docs_review_eligibility_params() -> dict[str, list[str]]:
    """Return schema-derived parameters for docs-review eligibility."""
    winning = _winning_review_resolution_methods()
    non_winning = _non_winning_review_resolution_methods()
    return {
        "docs_review_winning_methods": sorted(winning),
        "docs_review_non_winning_methods": sorted(non_winning),
    }


def _docs_review_winner_query_body(name_variable: str = "sn") -> str:
    """Return the single graph traversal that selects a winning docs group."""
    if name_variable != "sn":
        raise ValueError("docs-review eligibility requires the StandardName alias 'sn'")
    return """
        MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
        WHERE review.review_axis = 'docs'
          AND review.review_group_id IS NOT NULL
          AND NOT EXISTS {
              MATCH (sn)-[:HAS_REVIEW]->(non_winner:StandardNameReview)
              WHERE non_winner.review_axis = 'docs'
                AND non_winner.resolution_method
                    IN $docs_review_non_winning_methods
          }
        WITH review.review_group_id AS review_group_id,
             max(CASE WHEN coalesce(review.is_canonical, false)
                 THEN 1 ELSE 0 END) AS canonical_group,
             max(review.reviewed_at) AS group_reviewed_at,
             [item IN collect({
                 review_id: review.id,
                 reviewed_at: review.reviewed_at,
                 resolution_method: review.resolution_method,
                 score: review.score
             }) WHERE item.resolution_method IN $docs_review_winning_methods]
                 AS winning_records
        WHERE size(winning_records) > 0
        UNWIND winning_records AS winning_record
        WITH review_group_id, canonical_group, group_reviewed_at, winning_record
        ORDER BY canonical_group DESC, group_reviewed_at DESC,
                 review_group_id DESC, winning_record.reviewed_at DESC,
                 winning_record.review_id DESC
        WITH collect({
            review_group_id: review_group_id,
            review_id: winning_record.review_id,
            resolution_method: winning_record.resolution_method,
            score: winning_record.score
        })[0] AS winner
        RETURN winner
    """


def docs_review_eligibility_where() -> str:
    """Return the shared predicate for a reachable winning docs review."""
    return f"EXISTS {{{_docs_review_winner_query_body()}}}"


def property_coverage(
    *,
    label: str,
    properties: tuple[str, ...],
    schema: GraphSchema,
    gc: Any | None = None,
) -> dict[str, int]:
    """Return candidate and property counts, failing closed on zero coverage."""
    declared = schema.get_all_slots(label)
    for property_name in properties:
        if property_name not in declared:
            raise ValueError(f"Unknown property '{property_name}' on {label}")
    if not properties:
        raise ValueError("At least one property is required for coverage")

    projections = ["count(node) AS candidates"]
    projections.extend(
        f"count(node.{property_name}) AS {property_name}_covered"
        for property_name in properties
    )
    query = f"MATCH (node:{label})\nRETURN " + ",\n       ".join(projections)
    if gc is None:
        with GraphClient() as graph:
            rows = graph.query(query)
    else:
        rows = gc.query(query)

    keys = ["candidates", *(f"{name}_covered" for name in properties)]
    coverage = {key: int((dict(rows[0]).get(key) if rows else 0) or 0) for key in keys}
    candidates = coverage["candidates"]
    for property_name in properties:
        coverage_key = f"{property_name}_covered"
        if candidates > 0 and coverage[coverage_key] == 0:
            raise RuntimeError(
                f"Property coverage failed for {label}.{property_name}: "
                f"candidates={candidates}, {coverage_key}=0"
            )
    return coverage


def docs_review_property_coverage(gc: Any | None = None) -> dict[str, int]:
    """Assert that every review property used by eligibility has live coverage.

    Cypher silently returns zero for a missing property. The coverage row makes
    a misspelled or removed slot a hard failure before the eligibility result is
    trusted, and also verifies the ``name``/``docs`` axis vocabulary.
    """
    query = """
        MATCH (review:StandardNameReview)
        RETURN count(review) AS reviews,
               count(review.review_axis) AS axis_covered,
               count(review.review_group_id) AS group_covered,
               count(review.resolution_method) AS method_covered,
               count(CASE WHEN review.review_axis IN ['name', 'docs']
                     THEN 1 END) AS known_axis,
               count(CASE WHEN review.review_axis = 'docs'
                     THEN 1 END) AS docs_axis
        """
    if gc is None:
        with GraphClient() as graph:
            rows = graph.query(query)
    else:
        rows = gc.query(query)
    if not rows:
        return {
            "reviews": 0,
            "axis_covered": 0,
            "group_covered": 0,
            "method_covered": 0,
            "known_axis": 0,
            "docs_axis": 0,
        }
    coverage = {key: int(value or 0) for key, value in dict(rows[0]).items()}
    reviews = coverage["reviews"]
    missing = [
        key
        for key in ("axis_covered", "group_covered", "method_covered")
        if reviews and coverage[key] == 0
    ]
    if coverage["axis_covered"] != coverage["known_axis"]:
        missing.append("known_axis")
    if reviews and coverage["docs_axis"] == 0:
        missing.append("docs_axis")
    if missing:
        raise RuntimeError(
            "docs-review eligibility property coverage failed: "
            + ", ".join(sorted(set(missing)))
        )
    return coverage


def write_name_review_results(
    entries: list[dict[str, Any]],
    *,
    stats: dict[str, Any] | None = None,
) -> int:
    """Write name-axis review scores to StandardName nodes.

    Writes ``reviewer_score_name``, ``reviewed_name_at``, and all
    name-axis rubric fields. Does **not** touch any shared aggregate slots
    (those have been removed from the schema).

    The in-memory ``reviewer_score`` dict key is mapped to the graph
    property ``reviewer_score_name`` here — there is no generic
    ``reviewer_score`` graph property.

    Parameters
    ----------
    entries:
        Dicts with at least ``id`` and ``reviewer_score`` keys.
        The ``reviewer_score`` key is the in-memory generic name used
        by the pipeline; it maps to ``sn.reviewer_score_name`` on the
        graph.
    stats:
        Optional mutable dict to accumulate write counters.

    Returns
    -------
    int
        Number of StandardName nodes written.
    """
    if not entries:
        return 0
    if stats is None:
        stats = {}

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.updated_at = datetime(), sn.reviewer_score_name = b.reviewer_score_name,
                sn.reviewed_name_at = b.reviewed_name_at,
                sn.reviewer_scores_name = coalesce(b.reviewer_scores_name, sn.reviewer_scores_name),
                sn.reviewer_comments_name = coalesce(b.reviewer_comments_name, sn.reviewer_comments_name),
                sn.reviewer_comments_per_dim_name = coalesce(b.reviewer_comments_per_dim_name, sn.reviewer_comments_per_dim_name),
                sn.reviewer_model_name = coalesce(b.reviewer_model_name, sn.reviewer_model_name),
                sn.review_tier = coalesce(b.review_tier, sn.review_tier),
                sn.review_input_hash = b.review_input_hash,
                sn.reviewer_suggested_name = coalesce(nullIf(b.reviewer_suggested_name, ''), sn.reviewer_suggested_name),
                sn.reviewer_suggestion_justification_name = coalesce(nullIf(b.reviewer_suggestion_justification_name, ''), sn.reviewer_suggestion_justification_name),
                sn.llm_cost_review_name = coalesce(sn.llm_cost_review_name, 0.0) + coalesce(b.llm_cost_review_name, 0.0),
                sn.review_name_count = coalesce(sn.review_name_count, 0) + 1,
                sn.llm_cost = coalesce(sn.llm_cost, 0.0) + coalesce(b.llm_cost_review_name, 0.0)
            """,
            batch=[
                {
                    "id": e.get("_original_id") or e["id"],
                    "reviewer_score_name": e.get("reviewer_score"),
                    "reviewed_name_at": e.get("reviewed_at"),
                    "reviewer_scores_name": _ensure_json(e.get("reviewer_scores")),
                    "reviewer_comments_name": e.get("reviewer_comments"),
                    "reviewer_comments_per_dim_name": _ensure_json(
                        e.get("reviewer_comments_per_dim")
                    ),
                    "reviewer_model_name": e.get("reviewer_model"),
                    "review_tier": e.get("review_tier"),
                    "review_input_hash": e.get("review_input_hash"),
                    "reviewer_suggested_name": e.get("_suggested_name") or "",
                    "reviewer_suggestion_justification_name": e.get(
                        "_suggestion_justification"
                    )
                    or "",
                    "llm_cost_review_name": e.get("llm_cost") or 0.0,
                }
                for e in entries
            ],
        )

    written = len(entries)
    logger.info("write_name_review_results: wrote %d names", written)
    return written


def write_docs_review_results(
    entries: list[dict[str, Any]],
    *,
    stats: dict[str, Any] | None = None,
) -> int:
    """Write docs-axis review scores to StandardName nodes.

    Gate: entries whose ``reviewed_name_at IS NULL`` in the graph are
    skipped (logged as ERROR, counted in
    ``stats['docs_skipped_missing_name']``).

    Does **not** touch any shared aggregate slots
    (those have been removed from the schema).

    The in-memory ``reviewer_score`` dict key is mapped to the graph
    property ``reviewer_score_docs`` here — there is no generic
    ``reviewer_score`` graph property.

    Parameters
    ----------
    entries:
        Dicts with at least ``id`` and ``reviewer_score`` keys.
        The ``reviewer_score`` key is the in-memory generic name used
        by the pipeline; it maps to ``sn.reviewer_score_docs`` on the
        graph.
    stats:
        Optional mutable dict to accumulate skip/write counters.

    Returns
    -------
    int
        Number of StandardName nodes written.
    """
    if not entries:
        return 0
    if stats is None:
        stats = {}

    # Gate check: query graph for reviewed_name_at on target names
    entry_ids = [e["id"] for e in entries if e.get("id")]
    if not entry_ids:
        return 0
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            RETURN sn.id AS id, sn.reviewed_name_at AS reviewed_name_at
            """,
            ids=entry_ids,
        )
        gated: dict[str, bool] = {}
        for r in rows or []:
            gated[r["id"]] = r.get("reviewed_name_at") is not None

    passed: list[dict[str, Any]] = []
    skipped = 0
    for e in entries:
        eid = e.get("id", "")
        if not gated.get(eid, False):
            logger.error(
                "write_docs_review_results: skipping %r — reviewed_name_at IS NULL",
                eid,
            )
            skipped += 1
            continue
        passed.append(e)
    stats["docs_skipped_missing_name"] = (
        stats.get("docs_skipped_missing_name", 0) + skipped
    )
    if not passed:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.updated_at = datetime(), sn.reviewer_score_docs = b.reviewer_score_docs,
                sn.reviewed_docs_at = b.reviewed_docs_at,
                sn.reviewer_scores_docs = coalesce(b.reviewer_scores_docs, sn.reviewer_scores_docs),
                sn.reviewer_comments_docs = coalesce(b.reviewer_comments_docs, sn.reviewer_comments_docs),
                sn.reviewer_comments_per_dim_docs = coalesce(b.reviewer_comments_per_dim_docs, sn.reviewer_comments_per_dim_docs),
                sn.reviewer_model_docs = coalesce(b.reviewer_model_docs, sn.reviewer_model_docs),
                sn.review_input_hash = b.review_input_hash,
                sn.llm_cost_review_docs = coalesce(sn.llm_cost_review_docs, 0.0) + coalesce(b.llm_cost_review_docs, 0.0),
                sn.review_docs_count = coalesce(sn.review_docs_count, 0) + 1,
                sn.llm_cost = coalesce(sn.llm_cost, 0.0) + coalesce(b.llm_cost_review_docs, 0.0)
            """,
            batch=[
                {
                    "id": e["id"],
                    "reviewer_score_docs": e.get("reviewer_score"),
                    "reviewed_docs_at": e.get("reviewed_at"),
                    "reviewer_scores_docs": _ensure_json(e.get("reviewer_scores")),
                    "reviewer_comments_docs": e.get("reviewer_comments"),
                    "reviewer_comments_per_dim_docs": _ensure_json(
                        e.get("reviewer_comments_per_dim")
                    ),
                    "reviewer_model_docs": e.get("reviewer_model"),
                    "review_input_hash": e.get("review_input_hash"),
                    "llm_cost_review_docs": e.get("llm_cost") or 0.0,
                }
                for e in passed
            ],
        )

    written = len(passed)
    logger.info("write_docs_review_results: wrote %d names", written)
    return written


def _resolve_grammar_token_version(gc: GraphClient, isn_version: str) -> str | None:
    """Find the best GrammarToken version for HAS_SEGMENT resolution.

    Prefers exact match with the ISN runtime version.  When no tokens
    exist for that version (e.g. ISN was upgraded but the grammar has
    not yet been re-synced — ``sn run`` auto-syncs at startup), falls
    back to the latest available version so that token-miss detection
    does not produce false-positive VocabGap nodes.

    Returns ``None`` when no GrammarToken nodes exist at all.
    """
    # Fast path: check exact version
    rows = list(
        gc.query(
            "MATCH (t:GrammarToken {version: $v}) RETURN t.version LIMIT 1",
            v=isn_version,
        )
        or []
    )
    if rows:
        return isn_version

    # Fallback: graph authority first, semantic package-version order second.
    from imas_codex.standard_names.grammar_query import select_grammar_version

    rows = list(
        gc.query(
            "MATCH (t:GrammarToken) "
            "WITH DISTINCT t.version AS version "
            "OPTIONAL MATCH (v:ISNGrammarVersion {version: version}) "
            "RETURN version, coalesce(v.active, false) AS active"
        )
        or []
    )
    fallback = select_grammar_version(rows)
    if fallback is not None:
        logger.warning(
            "No GrammarToken nodes for ISN %s — falling back to %s. "
            "`sn run` auto-syncs the grammar at startup; re-run to update.",
            isn_version,
            fallback,
        )
        return fallback

    return None


#: ISN grammar segments stored as bare-name columns on StandardName nodes.
#: Canonical render order places the single-token modifier segments
#: (aggregation, orbit, population) ahead of subject, mirroring the ISN
#: render order ``<aggregation>_<orbit>_<population>_<subject>_<base>``.
_GRAMMAR_SEGMENT_COLUMNS = (
    "physical_base",
    "aggregation",
    "orbit",
    "population",
    "subject",
    "state",
    "transformation",
    "component",
    "coordinate",
    "process",
    "position",
    "region",
    "device",
    "geometric_base",
    "object",
    "geometry",
)

#: Segment → typed-edge relationship type. Mirrors the GrammarToken edge
#: convention declared in ``standard_name.yaml`` (grammar_*_token slots).
#: Only segments that have a corresponding HAS_* typed edge appear here;
#: ``coordinate``/``object``/``geometry`` carry no typed edge today.
_SEGMENT_EDGE_TYPES: tuple[tuple[str, str], ...] = (
    ("physical_base", "HAS_PHYSICAL_BASE"),
    ("subject", "HAS_SUBJECT"),
    ("transformation", "HAS_TRANSFORMATION"),
    ("component", "HAS_COMPONENT"),
    ("coordinate", "HAS_COORDINATE"),
    ("process", "HAS_PROCESS"),
    ("position", "HAS_POSITION"),
    ("region", "HAS_REGION"),
    ("device", "HAS_DEVICE"),
    ("geometric_base", "HAS_GEOMETRIC_BASE"),
    ("aggregation", "HAS_AGGREGATION"),
    ("orbit", "HAS_ORBIT"),
    ("population", "HAS_POPULATION"),
)


def _coerce_segment_value(value: Any) -> str | None:
    """Coerce a parser segment value (str / Enum / tuple / None) to a graph-safe scalar.

    Multi-token segments (e.g. ``zone`` on ``lower_outer_squareness``) arrive
    from the ISN model as a tuple/list of tokens; an absent multi-token
    segment is an EMPTY tuple. Join the tokens with ``_`` (matching the
    surface form, ``lower_outer``) and map the empty case to ``None`` so an
    absent segment never persists as the literal string ``"()"``.
    """
    if value is None:
        return None
    if isinstance(value, tuple | list):
        parts = [p for p in (_coerce_segment_value(v) for v in value) if p]
        return "_".join(parts) if parts else None
    # Enum → its .value
    val = getattr(value, "value", value)
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _resolve_synced_segments(
    gc: GraphClient, token_version: str | None
) -> frozenset[str]:
    """Return grammar segments that have GrammarToken nodes in the graph.

    Segments with zero synced tokens (e.g. ``physical_base`` when the ISN
    graph spec excludes it) generate ``segment_edge_specs`` entries from
    the ISN parser but can never match a GrammarToken node. Filtering
    these out prevents false-positive token-miss warnings and VocabGap
    nodes.
    """
    if token_version is None:
        return frozenset()

    rows = list(
        gc.query(
            "MATCH (t:GrammarToken {version: $v}) RETURN DISTINCT t.segment AS seg",
            v=token_version,
        )
        or []
    )
    return frozenset(r["seg"] for r in rows)


def _write_grammar_decomposition(
    gc: GraphClient, name_ids: list[str]
) -> list[dict[str, str]]:
    """Write per-segment columns and typed grammar edges on StandardName nodes.

    Replaces ``_write_segment_edges``. Always populates
    compatibility per-segment columns (``sn.physical_base``, ``sn.subject``,
    …) from the strict ISN IR, regardless of whether a closed-vocabulary
    GrammarToken exists for the value. Conditionally writes typed edges
    only when a GrammarToken does exist.  The flat Pydantic projection is not a
    validity gate: ordered unary chains and binary trees remain valid and keep
    their IR-derived columns. On strict parser error all columns and edges are
    cleared (leaving the node unparseable). Such a name is quarantined at
    validate time
    (``validation_status='quarantined'``) — the single-pipeline gate — so
    no separate fallback flag is recorded.

    Idempotent: re-running on existing nodes overwrites columns and
    refreshes typed edges. Missing segments are written as ``null`` so
    re-writing a name whose grammar narrowed clears stale columns.

    Args:
        gc: Open GraphClient session.
        name_ids: List of StandardName.id values to process.

    Returns:
        List of token-miss gaps detected against the closed vocabulary,
        each a dict with keys ``sn_id``, ``segment``, ``token``.
    """
    all_gaps: list[dict[str, str]] = []

    try:
        from imas_standard_names import __version__ as isn_version
        from imas_standard_names.grammar import compose, parse
    except ImportError:
        logger.debug("ISN grammar not available — skipping grammar decomposition")
        return all_gaps

    # Token version is required for typed-edge writing only; column-write
    # path runs even when no GrammarToken nodes exist.
    token_version = _resolve_grammar_token_version(gc, isn_version)

    # Discover which segments actually have GrammarToken nodes in the graph.
    # Segments with 0 synced tokens (e.g. physical_base) generate edge specs
    # from ISN but can never match — exclude them from token-miss detection.
    synced_segments = _resolve_synced_segments(gc, token_version)

    # Build Cypher fragments once from the trusted module constants. Property
    # and relationship names come from a closed module-level tuple (never user
    # input), so direct interpolation is safe.
    edge_delete_types = "|".join(
        ["HAS_SEGMENT", *(rel for _seg, rel in _SEGMENT_EDGE_TYPES)]
    )
    columns_to_null = (
        ",\n                    ".join(
            f"sn.{col} = null" for col in _GRAMMAR_SEGMENT_COLUMNS
        )
        + ",\n                    sn.updated_at = datetime()"
    )
    columns_to_set = (
        ",\n                ".join(
            f"sn.{col} = ${col}" for col in _GRAMMAR_SEGMENT_COLUMNS
        )
        + ",\n                sn.updated_at = datetime()"
    )
    typed_edge_foreach = "\n                ".join(
        f"FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = '{seg}' "
        f"THEN [1] ELSE [] END |\n                    "
        f"MERGE (sn)-[:{rel}]->(t)\n                )"
        for seg, rel in _SEGMENT_EDGE_TYPES
    )

    for sn_id in name_ids:
        try:
            parsed = parse(sn_id, strict=True)
            if compose(parsed.ir) != sn_id:
                raise ValueError("strict grammar round-trip changed the name")
        except Exception as exc:
            logger.warning("Grammar parse failed for '%s': %s", sn_id, exc)
            # Clear columns and typed/segment edges for genuinely invalid names.
            gc.query(
                f"""
                MATCH (sn:StandardName {{id: $sn_id}})
                OPTIONAL MATCH (sn)-[r:{edge_delete_types}]->(:GrammarToken)
                DELETE r
                WITH sn
                SET {columns_to_null}
                """,
                sn_id=sn_id,
            )
            continue

        # ---- Always-on column write ------------------
        # Shared extraction authority — identical to the persist path
        # (_parse_grammar), so columns cannot diverge between the two.
        column_values = _segments_from_ir(parsed.ir)

        gc.query(
            f"""
            MATCH (sn:StandardName {{id: $sn_id}})
            SET {columns_to_set}
            """,
            sn_id=sn_id,
            **column_values,
        )

        # ---- Conditional typed-edge write (closed-vocab) ------------------
        if token_version is None:
            # No GrammarToken corpus — column write is enough; skip edges.
            continue

        # Idempotent: drop existing typed/segment edges before re-writing.
        gc.query(
            f"""
            MATCH (sn:StandardName {{id: $sn_id}})-[r:{edge_delete_types}]->(:GrammarToken)
            DELETE r
            """,
            sn_id=sn_id,
        )

        edges_param = [
            {
                "position": position,
                "segment": segment,
                "token": token,
            }
            for position, segment in enumerate(_public_grammar_segment_order())
            if segment in _GRAMMAR_SEGMENT_COLUMNS
            if (token := column_values.get(segment)) is not None
            and segment in synced_segments
        ]
        if not edges_param:
            continue

        results = list(
            gc.query(
                f"""
                MATCH (sn:StandardName {{id: $sn_id}})
                UNWIND $edges AS edge
                OPTIONAL MATCH (t:GrammarToken {{
                    value: edge.token,
                    segment: edge.segment,
                    version: $token_version
                }})
                WITH sn, edge, t
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (sn)-[r:HAS_SEGMENT]->(t)
                    SET r.position = edge.position,
                        r.segment = edge.segment
                )
                {typed_edge_foreach}
                RETURN edge.token AS token,
                       edge.segment AS segment,
                       t IS NOT NULL AS matched
                """,
                sn_id=sn_id,
                edges=edges_param,
                token_version=token_version,
            )
            or []
        )

        # Only report token misses for segments that have GrammarToken
        # nodes in the graph.  Segments like physical_base are intentionally
        # excluded from the graph sync and would always produce false
        # positives.
        missing = [
            f"{r['segment']}:{r['token']}"
            for r in results
            if not r.get("matched", True) and r.get("segment") in synced_segments
        ]
        if missing:
            logger.warning(
                "Token-miss for '%s': %s (ISN %s, tokens %s) — vocab gap",
                sn_id,
                ", ".join(missing),
                isn_version,
                token_version,
            )
            for r in results:
                if not r.get("matched", True) and r.get("segment") in synced_segments:
                    all_gaps.append(
                        {
                            "sn_id": sn_id,
                            "segment": r["segment"],
                            "token": r["token"],
                        }
                    )

    return all_gaps


# Deprecation alias — the old name is retained for one release. Callers
# inside the package have been migrated; external pipeline code may still
# import this symbol, so it is removed only in the release after that.
def _write_segment_edges(gc: GraphClient, name_ids: list[str]) -> list[dict[str, str]]:
    """Deprecated alias for :func:`_write_grammar_decomposition`."""
    import warnings

    warnings.warn(
        "_write_segment_edges is deprecated; use _write_grammar_decomposition.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _write_grammar_decomposition(gc, name_ids)


# =============================================================================
# Immediate-persist helpers — graph-state-machine generate_name
# =============================================================================

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

#: Units for which scalar quantities can safely default to ``one_like``
#: COCOS transformation type — these are sign-invariant under all COCOS
#: conventions.  Do not include ``Wb``, ``T``, ``A``, ``V.s``, ``T.m``
#: or any unit that may carry a COCOS-dependent sign.
SAFE_SCALAR_COCOS_UNITS: frozenset[str] = frozenset(
    {
        "1",
        "m",
        "m^2",
        "m^3",
        "eV",
        "Pa",
        "kg.m^-3",
        "s",
        "s^-1",
        "Hz",
        "m^-3",
        "m.s^-1",
        "A.m^-2",
    }
)


_RETIRE_FINALIZED_SOURCE_GAPS = """
                    WITH sns
                    OPTIONAL MATCH
                        (sns)-[source_gap:HAS_STANDARD_NAME_VOCAB_GAP]->(:VocabGap)
                    WITH sns, collect(DISTINCT source_gap) AS source_gap_edges
                    OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(dd:IMASNode)
                    OPTIONAL MATCH (sns)-[:FROM_SIGNAL]->(signal:FacilitySignal)
                    WITH sns, source_gap_edges,
                         collect(DISTINCT dd) + collect(DISTINCT signal)
                             AS backing_entities
                    FOREACH (edge IN source_gap_edges | DELETE edge)
                    WITH sns,
                         CASE
                           WHEN size(backing_entities) = 1
                            AND (
                              (sns.source_type = 'dd'
                               AND sns.id = 'dd:' + sns.source_id
                               AND backing_entities[0].id = sns.source_id
                               AND 'IMASNode' IN labels(backing_entities[0]))
                              OR
                              (sns.source_type = 'signals'
                               AND sns.id = 'signals:' + sns.source_id
                               AND backing_entities[0].id = sns.source_id
                               AND 'FacilitySignal' IN labels(backing_entities[0]))
                            )
                           THEN backing_entities ELSE []
                         END AS exact_backing
                    UNWIND CASE
                      WHEN exact_backing = [] THEN [null]
                      ELSE exact_backing
                    END AS source_entity
                    OPTIONAL MATCH
                        (source_entity)-[entity_gap:HAS_STANDARD_NAME_VOCAB_GAP]
                        ->(:VocabGap)
                    WITH sns, collect(DISTINCT entity_gap) AS entity_gap_edges
                    FOREACH (edge IN entity_gap_edges | DELETE edge)
                    RETURN sns.id AS id
"""


def persist_generated_name_winners(
    candidates: list[dict[str, Any]],
    *,
    compose_model: str,
    dd_version: str | None = None,
    cocos_version: int | None = None,
    run_id: str | None = None,
) -> list[str]:
    """Persist a single generate-name batch immediately to graph.

    Called from within ``compose_batch`` after LLM success.
    Enriches candidates with provenance metadata, embeds the standard-name
    string, and extracts grammar fields before writing.

    Embedding uses the standard-name string (``id``) — not the description,
    which is added later by the enrich pipeline.  If embedding fails for a
    candidate, ``embed_failed_at`` is set so the name can be retried later;
    the name is still written to the graph so it can advance through review.

    An exact token-and-sequence source CAS acquires the source write lock. The
    rich name writer, lifecycle transition, cluster backfill, and predecessor
    retirement then run through that transaction before the source claim is
    consumed. Any failure rolls every graph effect back together.

    Returns the finalized ``StandardNameSource`` identities.
    """
    from datetime import UTC

    if not candidates:
        return []

    now = datetime.now(UTC).isoformat()
    from imas_codex.standard_names.kind_derivation import derive_kind

    # Inject tool version provenance
    try:
        import imas_standard_names

        _isn_ver = imas_standard_names.__version__
    except (ImportError, AttributeError):
        _isn_ver = None
    try:
        import importlib.metadata

        _codex_ver = importlib.metadata.version("imas-codex")
    except Exception:
        _codex_ver = None

    for entry in candidates:
        entry.setdefault("model", compose_model)
        entry.setdefault("dd_version", dd_version)
        entry.setdefault("isn_version", _isn_ver)
        entry.setdefault("codex_version", _codex_ver)
        # validation_status is set upstream by audit logic (run_audits +
        # _is_quarantined) in the pool path or validate_worker in the
        # legacy linear path.  When neither has run (e.g. dry-run paths
        # that bypass audits, or callers that build candidates manually),
        # default to 'valid' here so the column is never NULL and
        # downstream filters keep working.  Embedding failures set
        # embed_failed_at for retry but do not change validation_status.
        entry.setdefault("validation_status", "valid")
        entry.setdefault("generated_at", now)
        # Strip private markers used only during in-batch attribution.
        entry.pop("_from_error_sibling", None)
        # Normalize name ID: expand physics abbreviations (e.g. ExB → e_cross_b)
        # and enforce lowercase before grammar validation.
        raw_name = entry.get("id") or ""
        if raw_name:
            entry["id"] = normalize_name_id(raw_name)
        # Derive kind from the name's own structure, overriding the LLM default.
        name = entry.get("id") or ""
        if name:
            entry["kind"] = derive_kind(name)

        # Default cocos_transformation_type to "one_like" for safe scalars
        # when the extractor / DD node did not already annotate one.
        if (
            not entry.get("cocos_transformation_type")
            and entry.get("kind") == "scalar"
            and (entry.get("unit") or "") in SAFE_SCALAR_COCOS_UNITS
        ):
            entry["cocos_transformation_type"] = "one_like"

    # --- Embedding deferred to shared discovery embed_description_worker ---
    # Clear embedding so the background embed worker picks these up.
    for entry in candidates:
        entry["embedding"] = None
        entry["embed_text_hash"] = None

    # Establish exact source identities before opening the transaction. Missing
    # fences are rejected by _complete_source_fences; stale fences do not bind
    # in the lock query below.
    write_batch: list[dict[str, Any]] = []
    candidate_by_sns: dict[str, dict[str, Any]] = {}
    for entry in candidates:
        if entry.get("model") == "deterministic:dd_error_modifier":
            continue
        raw_source_id = entry.get("source_id")
        source_types = entry.get("source_types") or []
        if not raw_source_id or not source_types:
            continue
        source_type = source_types[0]
        sns_id = f"{'dd' if source_type == 'dd' else 'signals'}:{raw_source_id}"
        write_batch.append(
            {
                "sns_id": sns_id,
                "source_id": raw_source_id,
                "source_type": source_type,
                "sn_id": entry["id"],
                "unit": entry.get("unit"),
                "model": compose_model,
                "claim_token": entry.get("source_claim_token"),
                "claim_seq": entry.get("source_claim_seq"),
            }
        )
        candidate_by_sns[sns_id] = entry

    write_batch = _complete_source_fences(write_batch)
    if not write_batch:
        return []

    finalized_ids: list[str] = []
    with GraphClient() as graph:
        with graph.session() as session:
            tx = session.begin_transaction()
            tx_gc = _TransactionQuery(tx)
            try:
                locked_rows = _lock_claimed_name_bindings(
                    tx_gc,
                    write_batch,
                    allow_missing=True,
                    allow_own_pending_reservation=True,
                    allow_dead_end_revival=True,
                )
                locked_ids = {
                    row["id"]
                    for row in locked_rows
                    if row.get("outcome", "winner") == "winner"
                }
                binding_kinds = {
                    row["id"]: (
                        row.get("binding_kind")
                        or (
                            "stable_reuse"
                            if row.get("target_stage") in _STABLE_BINDING_NAME_STAGES
                            else "owned_reservation"
                        )
                    )
                    for row in locked_rows
                    if row.get("outcome", "winner") == "winner"
                }
                locked_candidates = [
                    candidate_by_sns[sns_id]
                    for sns_id in candidate_by_sns
                    if sns_id in locked_ids
                ]
                if not locked_candidates:
                    tx.commit()
                    return []

                reserved_candidates = [
                    candidate_by_sns[sns_id]
                    for sns_id, kind in binding_kinds.items()
                    if kind == "owned_reservation"
                ]
                stable_candidates = [
                    candidate_by_sns[sns_id]
                    for sns_id, kind in binding_kinds.items()
                    if kind == "stable_reuse"
                ]
                written_name_ids = {entry["id"] for entry in stable_candidates}
                if reserved_candidates:
                    written_names = write_standard_names(
                        reserved_candidates,
                        gc=tx_gc,
                        return_written_ids=True,
                    )
                    if isinstance(written_names, int):
                        if written_names != len(reserved_candidates):
                            raise RuntimeError(
                                "rich standard-name write rejected an exact-claim "
                                "winner"
                            )
                        written_names = [entry["id"] for entry in reserved_candidates]
                    written_name_ids.update(written_names)
                finalize_batch = [
                    {**item, "binding_kind": binding_kinds[item["sns_id"]]}
                    for item in write_batch
                    if item["sns_id"] in locked_ids
                    and item["sn_id"] in written_name_ids
                ]
                if len(finalize_batch) != len(locked_candidates):
                    raise RuntimeError(
                        "rich standard-name write rejected an exact-claim winner"
                    )

                reserved_finalize_batch = [
                    item
                    for item in finalize_batch
                    if item["binding_kind"] == "owned_reservation"
                ]
                stable_finalize_batch = [
                    item
                    for item in finalize_batch
                    if item["binding_kind"] == "stable_reuse"
                ]
                finalized_rows: list[dict[str, Any]] = []
                if reserved_finalize_batch:
                    finalized_rows.extend(
                        tx_gc.query(
                            """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    MATCH (sns)-[reservation:PRODUCED_NAME]->(
                        sn:StandardName {id: b.sn_id})
                    WHERE sn.name_stage = $pending_stage
                      AND reservation.provisional = true
                      AND reservation.claim_token = b.claim_token
                      AND reservation.claim_seq = b.claim_seq
                    CALL (b) {
                      WITH b
                      WHERE b.source_type = 'dd'
                      MATCH (source:IMASNode {id: b.source_id})
                      RETURN source
                      UNION
                      WITH b
                      WHERE b.source_type = 'signals'
                      MATCH (source:FacilitySignal {id: b.source_id})
                      RETURN source
                    }
                    FOREACH (_ IN CASE
                      WHEN sns.compose_hint_status = 'open' THEN [1] ELSE [] END |
                      SET sns.compose_hint_status = 'consumed',
                          sns.compose_hint_consumed_at = datetime()
                    )
                    SET sn.name_stage = 'drafted',
                        sn.chain_length = coalesce(sn.chain_length, 0),
                        sn.docs_stage = coalesce(sn.docs_stage, 'pending'),
                        sn.docs_chain_length = coalesce(sn.docs_chain_length, 0),
                        sn.generated_at = coalesce(sn.generated_at, datetime()),
                        sn.model = coalesce(sn.model, b.model),
                        sn.run_id = coalesce(sns.run_id, sn.run_id),
                        sn.updated_at = datetime(),
                        sns.claim_token = null,
                        sns.claimed_at = null,
                        sns.status = 'composed',
                        sns.composed_at = datetime(),
                        sns.produced_sn_id = sn.id,
                        sns.last_error = null
                    REMOVE reservation.provisional,
                           reservation.claim_token,
                           reservation.claim_seq,
                           reservation.created_target
                    MERGE (source)-[:HAS_STANDARD_NAME]->(sn)
                    WITH sns, sn, b, b.source_type + ':' + b.source_id AS uri
                    SET sn.source_paths = CASE
                      WHEN uri IN coalesce(sn.source_paths, [])
                      THEN sn.source_paths
                      ELSE coalesce(sn.source_paths, []) + uri END,
                      sn.updated_at = datetime()
                    __RETIRE_FINALIZED_SOURCE_GAPS__
                    """.replace(
                                "__RETIRE_FINALIZED_SOURCE_GAPS__",
                                _RETIRE_FINALIZED_SOURCE_GAPS,
                            ),
                            batch=reserved_finalize_batch,
                            pending_stage=NameStage.pending.value,
                        )
                    )
                if stable_finalize_batch:
                    finalized_rows.extend(
                        tx_gc.query(
                            """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    MATCH (sn:StandardName {id: b.sn_id})
                    WHERE sn.name_stage IN $stable_stages
                      AND (b.unit IS NULL OR sn.unit IS NULL OR sn.unit = b.unit)
                    CALL (b) {
                      WITH b
                      WHERE b.source_type = 'dd'
                      MATCH (source:IMASNode {id: b.source_id})
                      RETURN source
                      UNION
                      WITH b
                      WHERE b.source_type = 'signals'
                      MATCH (source:FacilitySignal {id: b.source_id})
                      RETURN source
                    }
                    FOREACH (_ IN CASE
                      WHEN sns.compose_hint_status = 'open' THEN [1] ELSE [] END |
                      SET sns.compose_hint_status = 'consumed',
                          sns.compose_hint_consumed_at = datetime()
                    )
                    SET sns.claim_token = null,
                        sns.claimed_at = null,
                        sns.status = 'composed',
                        sns.composed_at = datetime(),
                        sns.produced_sn_id = sn.id,
                        sns.last_error = null
                    MERGE (sns)-[produced:PRODUCED_NAME]->(sn)
                    REMOVE produced.provisional,
                           produced.claim_token,
                           produced.claim_seq,
                           produced.created_target
                    MERGE (source)-[:HAS_STANDARD_NAME]->(sn)
                    WITH sns, sn, b, b.source_type + ':' + b.source_id AS uri
                    SET sn.source_paths = CASE
                      WHEN uri IN coalesce(sn.source_paths, [])
                      THEN sn.source_paths
                      ELSE coalesce(sn.source_paths, []) + uri END,
                      sn.updated_at = datetime()
                    __RETIRE_FINALIZED_SOURCE_GAPS__
                    """.replace(
                                "__RETIRE_FINALIZED_SOURCE_GAPS__",
                                _RETIRE_FINALIZED_SOURCE_GAPS,
                            ),
                            batch=stable_finalize_batch,
                            stable_stages=sorted(_STABLE_BINDING_NAME_STAGES),
                        )
                    )
                finalized_ids = [row["id"] for row in finalized_rows]
                if set(finalized_ids) != {item["sns_id"] for item in finalize_batch}:
                    raise RuntimeError(
                        "source claim changed during transactional name persistence"
                    )

                scoped_rows = tx_gc.query(
                    """
                    UNWIND $source_ids AS source_id
                    MATCH (source:StandardNameSource {id: source_id})
                    WHERE source.drain_scope_id IS NOT NULL
                    MATCH (source)-[:PRODUCED_NAME]->(name:StandardName)
                    OPTIONAL MATCH (name)-[:HAS_PARENT*1..]->(parent:StandardName)
                    WHERE parent.origin = 'derived'
                    WITH source, name,
                         [p IN collect(DISTINCT parent) WHERE p IS NOT NULL]
                           AS parents
                    WHERE (name.drain_scope_id IS NULL
                           OR name.drain_scope_id = source.drain_scope_id
                           OR name.drain_scope_claimed_at IS NULL
                           OR name.drain_scope_claimed_at < datetime()
                                - duration('PT600S'))
                      AND all(parent IN parents WHERE
                           parent.drain_scope_id IS NULL
                           OR parent.drain_scope_id = source.drain_scope_id
                           OR parent.drain_scope_claimed_at IS NULL
                           OR parent.drain_scope_claimed_at < datetime()
                                - duration('PT600S'))
                    SET name.updated_at = datetime(), name.drain_scope_id = source.drain_scope_id,
                        name.drain_scope_claimed_at =
                          source.drain_scope_claimed_at,
                        name.drain_scope_paths = source.drain_scope_paths
                    FOREACH (parent IN parents |
                      SET parent.updated_at = datetime(), parent.drain_scope_id = source.drain_scope_id,
                          parent.drain_scope_claimed_at =
                            source.drain_scope_claimed_at,
                          parent.drain_scope_paths = reduce(
                            paths = coalesce(parent.drain_scope_paths, []),
                            path IN source.drain_scope_paths |
                            CASE WHEN path IN paths THEN paths
                                 ELSE paths + path END))
                    RETURN source.id AS id
                    """,
                    source_ids=finalized_ids,
                )
                expected_scoped = {
                    row["id"]
                    for row in tx_gc.query(
                        """
                        UNWIND $source_ids AS source_id
                        MATCH (source:StandardNameSource {id: source_id})
                        WHERE source.drain_scope_id IS NOT NULL
                        RETURN source.id AS id
                        """,
                        source_ids=finalized_ids,
                    )
                }
                if {row["id"] for row in scoped_rows} != expected_scoped:
                    raise ManifestDrainConflict(
                        "generated name or derived parent has a competing drain lease"
                    )

                finalized_reserved_candidates = [
                    candidate_by_sns[sns_id]
                    for sns_id in finalized_ids
                    if binding_kinds[sns_id] == "owned_reservation"
                ]
                if finalized_reserved_candidates:
                    _backfill_cluster_from_sources(
                        finalized_reserved_candidates, gc=tx_gc, strict=True
                    )
                supersede_pairs = [
                    {"new_name": entry["id"], "source_id": entry["source_id"]}
                    for entry in finalized_reserved_candidates
                    if entry.get("id")
                    and entry.get("source_id")
                    and "dd" in (entry.get("source_types") or [])
                ]
                if supersede_pairs:
                    supersede_prior_source_names(supersede_pairs, gc=tx_gc)
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    if finalized_ids:
        bump_sn_run_counter(run_id, "names_composed", delta=len(finalized_ids))
    return finalized_ids


def persist_generated_name_batch(
    candidates: list[dict[str, Any]],
    *,
    compose_model: str,
    dd_version: str | None = None,
    cocos_version: int | None = None,
    run_id: str | None = None,
    return_winner_ids: bool = False,
) -> int | list[str]:
    """Persist a generate-name batch and report its finalized sources."""
    winner_ids = persist_generated_name_winners(
        candidates,
        compose_model=compose_model,
        dd_version=dd_version,
        cocos_version=cocos_version,
        run_id=run_id,
    )
    return winner_ids if return_winner_ids else len(winner_ids)


def _backfill_cluster_from_sources(
    candidates: list[dict[str, Any]],
    *,
    gc: Any | None = None,
    strict: bool = False,
) -> None:
    """Backfill ``primary_cluster_id`` and ``IN_CLUSTER`` edge from DD sources.

    For each candidate, query the graph for the best cluster via the
    HAS_STANDARD_NAME → IMASNode → IN_CLUSTER path, preferring
    IDS-scope > domain-scope > global-scope clusters.
    """
    sn_ids = [c["id"] for c in candidates if c.get("id")]
    if not sn_ids:
        return

    try:
        with nullcontext(gc) if gc is not None else GraphClient() as write_gc:
            results = write_gc.query(
                """
                UNWIND $sn_ids AS sid
                MATCH (sn:StandardName {id: sid})
                WHERE sn.primary_cluster_id IS NULL
                OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(imas:IMASNode)
                    -[:IN_CLUSTER]->(c:IMASSemanticCluster)
                WITH sn, c
                ORDER BY sn.id,
                         CASE c.scope
                             WHEN 'ids' THEN 0
                             WHEN 'domain' THEN 1
                             WHEN 'global' THEN 2
                             ELSE 3
                         END
                WITH sn, collect(c)[0] AS best_cluster
                WHERE best_cluster IS NOT NULL
                SET sn.updated_at = datetime(), sn.primary_cluster_id = best_cluster.id
                MERGE (sn)-[:IN_CLUSTER]->(best_cluster)
                RETURN sn.id AS sn_id, best_cluster.id AS cluster_id
                """,
                sn_ids=sn_ids,
            )
            if results:
                logger.debug(
                    "_backfill_cluster_from_sources: linked %d SNs to clusters",
                    len(results),
                )
    except Exception:
        if strict:
            raise
        logger.debug("_backfill_cluster_from_sources: failed", exc_info=True)


@retry_on_deadlock()
def _finalize_generated_name_stage(
    batch: list[dict[str, Any]],
) -> list[str]:
    """Set stage fields on new SNs and clear source claims — single transaction.

    Each item in *batch* must have ``sn_id`` (StandardName id), optionally
    ``sns_id`` (StandardNameSource id), and ``model``.

    In one transaction:
    - initialise missing/pending name lifecycle to ``drafted`` without
      demoting an existing reviewed, accepted, exhausted, or superseded name.
      A node carrying no description is parked at ``pending`` instead: name
      review requires prose, so drafting a prose-less name mints work nothing
      can claim — and the source is about to read ``composed``, which the
      compose pool excludes, so the pair would strand permanently.
      :func:`reconcile_reviewable_name_stage` re-opens such a pair.
    - fill missing docs/generation defaults without overwriting later state
    - Source: ``claim_token = null``, ``claimed_at = null``,
      ``status = 'composed'``, ``composed_at = datetime()``,
      ``produced_sn_id = sn.id``
    - Edge: ``(sns)-[:PRODUCED_NAME]->(sn)``
    """
    if not batch:
        return []

    winner_ids: list[str] = []
    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                rows = list(
                    tx.run(
                        """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE b.claim_token IS NOT NULL
                      AND b.claim_seq IS NOT NULL
                      AND sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.updated_at = datetime(), sn.name_stage = CASE
                            WHEN (sn.name_stage IS NULL
                                  OR sn.name_stage = 'pending')
                                 AND sn.description IS NOT NULL
                                 AND trim(sn.description) <> ''
                            THEN 'drafted'
                            WHEN sn.name_stage IS NULL THEN 'pending'
                            ELSE sn.name_stage END,
                        sn.chain_length = coalesce(sn.chain_length, 0),
                        sn.docs_stage = coalesce(sn.docs_stage, 'pending'),
                        sn.docs_chain_length = coalesce(sn.docs_chain_length, 0),
                        sn.generated_at = coalesce(sn.generated_at, datetime()),
                        sn.model = coalesce(sn.model, b.model)
                    WITH sn, b, sns
                    WHERE sns IS NOT NULL
                    SET sn.updated_at = datetime(), sns.claim_token  = null,
                        sns.claimed_at   = null,
                        sns.status       = 'composed',
                        sns.composed_at  = datetime(),
                        sns.produced_sn_id = sn.id,
                        sns.last_error   = null,
                        sn.run_id        = coalesce(sns.run_id, sn.run_id)
                    MERGE (sns)-[produced:PRODUCED_NAME]->(sn)
                    REMOVE produced.provisional,
                           produced.claim_token,
                           produced.claim_seq,
                           produced.created_target
                    __RETIRE_FINALIZED_SOURCE_GAPS__
                    """.replace(
                            "__RETIRE_FINALIZED_SOURCE_GAPS__",
                            _RETIRE_FINALIZED_SOURCE_GAPS,
                        ),
                        batch=batch,
                    )
                )
                winner_ids = [row["id"] for row in rows]
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    logger.debug(
        "_finalize_generated_name_stage: finalized %d SNs",
        len(winner_ids),
    )
    return winner_ids


def write_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.write_vocab_gaps."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.write_vocab_gaps(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_vocab_gap_adjudications(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.fetch_vocab_gap_adjudications."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.fetch_vocab_gap_adjudications(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Claim/mark/release — graph-state-machine workers
#
# Follows the battle-tested pattern from discovery/code/graph_ops.py:
#   1. claim: ORDER BY rand(), SET claimed_at + claim_token
#   2. verify: re-query by claim_token (prevents double-claim)
#   3. process (caller)
#   4. mark: SET result fields + clear claimed_at/claim_token (token-verified)
#   5. release (on error): clear claimed_at/claim_token (token-verified)
# =============================================================================

_CLAIM_TIMEOUT = import_module(
    ".graph_store.graph_catalog_reset", __package__
)._CLAIM_TIMEOUT


def claim_names_for_validation(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.claim_names_for_validation."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.claim_names_for_validation(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_names_validated(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_validated."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_validated(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_validation_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.release_validation_claims."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.release_validation_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_names_for_embedding(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.claim_names_for_embedding."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.claim_names_for_embedding(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_names_embedded(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_embedded."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_embedded(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_embedding_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.release_embedding_claims."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.release_embedding_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def get_validated_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.get_validated_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.get_validated_names(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_names_consolidated(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_consolidated."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_consolidated(
        *args, services=_graph_foundation.services, **kwargs
    )


def reset_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.reset_standard_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.reset_standard_names(
        *args, services=_graph_foundation.services, **kwargs
    )


def clear_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.clear_standard_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.clear_standard_names(
        *args, services=_graph_foundation.services, **kwargs
    )


def clear_sn_subsystem(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.clear_sn_subsystem."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.clear_sn_subsystem(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Link resolution
# =============================================================================

_MAX_LINK_RETRIES = import_module(
    ".graph_store.graph_catalog_reset", __package__
)._MAX_LINK_RETRIES


def claim_unresolved_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links.claim_unresolved_links."""
    from .graph_store import graph_doc_links

    return graph_doc_links.claim_unresolved_links(
        *args, services=_graph_foundation.services, **kwargs
    )


def resolve_links_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links.resolve_links_batch."""
    from .graph_store import graph_doc_links

    return graph_doc_links.resolve_links_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Documentation link resolution — post-drain cleanup
# =============================================================================


_NAME_LINK_RE = import_module(".graph_store.graph_doc_links", __package__)._NAME_LINK_RE


# When the LLM emits LaTeX with single backslashes inside its JSON output
# (e.g. ``$\theta$``, ``$\rho$``, ``\frac``), a lenient JSON decoder turns the
# ``\t``/``\r``/``\f``/``\b``/``\v`` prefix into the corresponding control
# character (TAB/CR/FF/BS/VT), corrupting the LaTeX (``$\rho$`` -> ``$<CR>ho$``).
# These control chars never appear legitimately in physics documentation, so map
# them back to the backslash form to reconstruct the command; any OTHER control
# char (NUL, ETX, …) is unrecoverable corruption and is stripped. Newline is kept.
_JSON_ESCAPE_CTRL = import_module(
    ".graph_store.graph_doc_links", __package__
)._JSON_ESCAPE_CTRL

_STRIP_CTRL_RE = import_module(
    ".graph_store.graph_doc_links", __package__
)._STRIP_CTRL_RE


def _sanitize_doc_text(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links._sanitize_doc_text."""
    from .graph_store import graph_doc_links

    return graph_doc_links._sanitize_doc_text(
        *args, services=_graph_foundation.services, **kwargs
    )


# This matcher parses an already-resolved Markdown link into both its visible
# label and its target so the accept gate can compare their meanings.
# ``find_name_references`` owns reference discovery instead; it intentionally
# returns only identity/span/syntax and does not expose a Markdown link's label.
_DOC_LINK_RE = import_module(".graph_store.graph_doc_links", __package__)._DOC_LINK_RE

_DOC_TEXT_UNSET = import_module(
    ".graph_store.graph_doc_links", __package__
)._DOC_TEXT_UNSET


def _doc_link_mismatches(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links._doc_link_mismatches."""
    from .graph_store import graph_doc_links

    return graph_doc_links._doc_link_mismatches(
        *args, services=_graph_foundation.services, **kwargs
    )


def _normalize_bare_doc_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links._normalize_bare_doc_links."""
    from .graph_store import graph_doc_links

    return graph_doc_links._normalize_bare_doc_links(
        *args, services=_graph_foundation.services, **kwargs
    )


def resolve_doc_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links.resolve_doc_links."""
    from .graph_store import graph_doc_links

    return graph_doc_links.resolve_doc_links(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# StandardNameSource claims, fences and outcomes — defined in graph_store
# =============================================================================

CAP_PARKED_DISPOSITIONS = import_module(
    ".graph_store.graph_source_bindings", __package__
).CAP_PARKED_DISPOSITIONS
SourceClaimResult = import_module(
    ".graph_store.graph_source_bindings", __package__
).SourceClaimResult
_CAP_DISPOSITION_READBACK = import_module(
    ".graph_store.graph_source_bindings", __package__
)._CAP_DISPOSITION_READBACK
_CAP_DISPOSITION_WRITE = import_module(
    ".graph_store.graph_source_bindings", __package__
)._CAP_DISPOSITION_WRITE
_CAP_PARKED_QUERY = import_module(
    ".graph_store.graph_source_bindings", __package__
)._CAP_PARKED_QUERY
_CAP_UNDISPOSED_QUERY = import_module(
    ".graph_store.graph_source_bindings", __package__
)._CAP_UNDISPOSED_QUERY
_CLAIM_VERIFY_SETTLE_SECONDS = import_module(
    ".graph_store.graph_source_bindings", __package__
)._CLAIM_VERIFY_SETTLE_SECONDS
_COMPOSE_CAP_REASON = import_module(
    ".graph_store.graph_source_bindings", __package__
)._COMPOSE_CAP_REASON
_EXPLICIT_SOURCE_CLAIM_QUERY = import_module(
    ".graph_store.graph_source_bindings", __package__
)._EXPLICIT_SOURCE_CLAIM_QUERY
_MAX_SOURCE_COMPOSE_HINT_CHARS = import_module(
    ".graph_store.graph_source_bindings", __package__
)._MAX_SOURCE_COMPOSE_HINT_CHARS
_MAX_SOURCE_COMPOSE_HINT_REASON_CHARS = import_module(
    ".graph_store.graph_source_bindings", __package__
)._MAX_SOURCE_COMPOSE_HINT_REASON_CHARS
_VALID_PIPELINE_SOURCE_TYPES = import_module(
    ".graph_store.graph_source_bindings", __package__
)._VALID_PIPELINE_SOURCE_TYPES
_persist_outcomes = import_module(
    ".graph_store.graph_source_bindings", __package__
)._persist_outcomes


def write_enrichment_results(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.write_enrichment_results."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.write_enrichment_results(
        *args, services=_graph_foundation.services, **kwargs
    )


def _exact_dd_source_id(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._exact_dd_source_id."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._exact_dd_source_id(
        *args, services=_graph_foundation.services, **kwargs
    )


def _source_compose_hint_decision(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._source_compose_hint_decision."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._source_compose_hint_decision(
        *args, services=_graph_foundation.services, **kwargs
    )


def set_source_compose_hint(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.set_source_compose_hint."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.set_source_compose_hint(
        *args, services=_graph_foundation.services, **kwargs
    )


def _pin_dd_source_snapshots(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._pin_dd_source_snapshots."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._pin_dd_source_snapshots(
        *args, services=_graph_foundation.services, **kwargs
    )


def merge_standard_name_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.merge_standard_name_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.merge_standard_name_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_standard_name_source_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.claim_standard_name_source_batch."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.claim_standard_name_source_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_explicit_standard_name_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.claim_explicit_standard_name_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.claim_explicit_standard_name_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def _complete_source_fences(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._complete_source_fences."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._complete_source_fences(
        *args, services=_graph_foundation.services, **kwargs
    )


def _require_failure_reason(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._require_failure_reason."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._require_failure_reason(
        *args, services=_graph_foundation.services, **kwargs
    )


def _guard_failed_source_reasons(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._guard_failed_source_reasons."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._guard_failed_source_reasons(
        *args, services=_graph_foundation.services, **kwargs
    )


def _guard_existing_target_pairings(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._guard_existing_target_pairings."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._guard_existing_target_pairings(
        *args, services=_graph_foundation.services, **kwargs
    )


def _lock_claimed_name_bindings(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._lock_claimed_name_bindings."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._lock_claimed_name_bindings(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_claimed_source_outcomes(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.persist_claimed_source_outcomes."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.persist_claimed_source_outcomes(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_claimed_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.persist_claimed_vocab_gaps."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.persist_claimed_vocab_gaps(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_claimed_attachments(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.persist_claimed_attachments."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.persist_claimed_attachments(
        *args, services=_graph_foundation.services, **kwargs
    )


def stage_claimed_generated_candidates(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.stage_claimed_generated_candidates."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.stage_claimed_generated_candidates(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_claimed_source_metadata(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.fetch_claimed_source_metadata."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.fetch_claimed_source_metadata(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_sources_composed(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_sources_composed."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_sources_composed(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_sources_attached(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_sources_attached."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_sources_attached(
        *args, services=_graph_foundation.services, **kwargs
    )


def classify_parked_source(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.classify_parked_source."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.classify_parked_source(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_sources_failed(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_sources_failed."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_sources_failed(
        *args, services=_graph_foundation.services, **kwargs
    )


def record_parked_dispositions(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.record_parked_dispositions."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.record_parked_dispositions(
        *args, services=_graph_foundation.services, **kwargs
    )


def disposition_parked_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.disposition_parked_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.disposition_parked_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def record_dispositions_for_undisposed_parks(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.record_dispositions_for_undisposed_parks."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.record_dispositions_for_undisposed_parks(
        *args, services=_graph_foundation.services, **kwargs
    )


def census_parked_dispositions(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.census_parked_dispositions."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.census_parked_dispositions(
        *args, services=_graph_foundation.services, **kwargs
    )


def retry_failed_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.retry_failed_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.retry_failed_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def retry_skipped_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.retry_skipped_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.retry_skipped_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def _skipped_recovery_error(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._skipped_recovery_error."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._skipped_recovery_error(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_source_skipped(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_source_skipped."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_source_skipped(
        *args, services=_graph_foundation.services, **kwargs
    )


def mark_sources_stale(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_sources_stale."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_sources_stale(
        *args, services=_graph_foundation.services, **kwargs
    )


def write_skipped_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.write_skipped_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.write_skipped_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def list_skipped_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.list_skipped_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.list_skipped_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def get_skipped_source_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.get_skipped_source_counts."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.get_skipped_source_counts(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_standard_name_source_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.release_standard_name_source_claims."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.release_standard_name_source_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def _record_persist_outcome(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._record_persist_outcome."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._record_persist_outcome(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_outcome_snapshot(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.persist_outcome_snapshot."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.persist_outcome_snapshot(
        *args, services=_graph_foundation.services, **kwargs
    )


def reset_persist_outcomes(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.reset_persist_outcomes."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.reset_persist_outcomes(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_compose_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.claim_compose_sources."""
    from .graph_store import graph_audits

    return graph_audits.claim_compose_sources(*args, **kwargs)


def count_extracted_for_domain(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.count_extracted_for_domain."""
    from .graph_store import graph_audits

    return graph_audits.count_extracted_for_domain(*args, **kwargs)


def count_eligible_compose_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.count_eligible_compose_sources."""
    from .graph_store import graph_audits

    return graph_audits.count_eligible_compose_sources(*args, **kwargs)


def claim_review_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.claim_review_names."""
    from .graph_store import graph_audits

    return graph_audits.claim_review_names(*args, **kwargs)


def release_review_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.release_review_claims."""
    from .graph_store import graph_audits

    return graph_audits.release_review_claims(*args, **kwargs)


# Heartbeat interval: refresh a held claim well before the lease TTL so a
# long quorum-review / enrich batch cannot have its lease expire mid-flight
# and be re-claimed by a peer worker (which would duplicate the paid LLM
# spend). One third of the TTL leaves room for two refreshes before expiry
# even if one heartbeat is delayed.
_CLAIM_HEARTBEAT_SECONDS = import_module(
    ".graph_store.graph_audits", __package__
)._CLAIM_HEARTBEAT_SECONDS


def refresh_name_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.refresh_name_claims."""
    from .graph_store import graph_audits

    return graph_audits.refresh_name_claims(*args, **kwargs)


PROMOTE_STRANDED_NAME_WHERE = import_module(
    ".graph_store.graph_audits", __package__
).PROMOTE_STRANDED_NAME_WHERE

"""Canonical graph predicate for name-axis stranded promotion."""

_graph_audits = import_module(".graph_store.graph_audits", __package__)
_graph_audits.PROMOTE_STRANDED_DOCS_WHERE = (
    _graph_audits.PROMOTE_STRANDED_DOCS_WHERE.replace(
        "{docs_review_eligibility_where}", docs_review_eligibility_where()
    )
)
PROMOTE_STRANDED_DOCS_WHERE = _graph_audits.PROMOTE_STRANDED_DOCS_WHERE

"""Canonical graph predicate for docs-axis stranded promotion."""


def promote_stranded_reviewed(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.promote_stranded_reviewed."""
    from .graph_store import graph_audits

    return graph_audits.promote_stranded_reviewed(*args, **kwargs)


DOCS_AXIS_REDERIVE_NAME_WHERE = import_module(
    ".graph_store.graph_audits", __package__
).DOCS_AXIS_REDERIVE_NAME_WHERE

"""Scope for a docs-axis re-derivation: the name axis is already accepted."""

_DOCS_AXIS_MIRROR_AGREES = import_module(
    ".graph_store.graph_audits", __package__
)._DOCS_AXIS_MIRROR_AGREES

_DOCS_AXIS_MIRROR_DISAGREES = import_module(
    ".graph_store.graph_audits", __package__
)._DOCS_AXIS_MIRROR_DISAGREES


def _docs_axis_rederive_query(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._docs_axis_rederive_query."""
    from .graph_store import graph_audits

    return graph_audits._docs_axis_rederive_query(*args, **kwargs)


def reconcile_docs_axis_from_reviews(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_docs_axis_from_reviews."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_docs_axis_from_reviews(*args, **kwargs)


def _standard_name_source_upstream_present_cypher(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._standard_name_source_upstream_present_cypher."""
    from .graph_store import graph_audits

    return graph_audits._standard_name_source_upstream_present_cypher(*args, **kwargs)


def reconcile_standard_name_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_sources."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_sources(*args, **kwargs)


def fetch_manifest_source_release_rows(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.fetch_manifest_source_release_rows."""
    from .graph_store import graph_audits

    return graph_audits.fetch_manifest_source_release_rows(*args, **kwargs)


def reconcile_source_status_liveness(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_source_status_liveness."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_source_status_liveness(*args, **kwargs)


def retire_unreachable_hint_edits(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.retire_unreachable_hint_edits."""
    from .graph_store import graph_audits

    return graph_audits.retire_unreachable_hint_edits(*args, **kwargs)


def reconcile_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_vocab_gaps."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_vocab_gaps(*args, **kwargs)


def triage_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.triage_vocab_gaps."""
    from .graph_store import graph_audits

    return graph_audits.triage_vocab_gaps(*args, **kwargs)


# Skip classifications produced by the DD unit resolution leg. Everything else
# on ``skip_reason`` is a permanent eligibility exclusion (a temporal or local
# coordinate, configurable meaning, …) that no resolver change can lift, so the
# unit revival must never touch it.
_UNIT_SKIP_REASON_PREFIX = import_module(
    ".graph_store.graph_audits", __package__
)._UNIT_SKIP_REASON_PREFIX


def revive_unit_skipped_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.revive_unit_skipped_sources."""
    from .graph_store import graph_audits

    return graph_audits.revive_unit_skipped_sources(*args, **kwargs)


def _isn_grammar_context(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._isn_grammar_context."""
    from .graph_store import graph_audits

    return graph_audits._isn_grammar_context(*args, **kwargs)


def isn_vocabulary_signature(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.isn_vocabulary_signature."""
    from .graph_store import graph_audits

    return graph_audits.isn_vocabulary_signature(*args, **kwargs)


def retry_vocab_gap_sources_on_grammar_change(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.retry_vocab_gap_sources_on_grammar_change."""
    from .graph_store import graph_audits

    return graph_audits.retry_vocab_gap_sources_on_grammar_change(*args, **kwargs)


def reconcile_provenance(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_provenance."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_provenance(*args, **kwargs)


def reconcile_catalog_edit_origins(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_catalog_edit_origins."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_catalog_edit_origins(*args, **kwargs)


def reconcile_standard_name_cocos_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_cocos_links."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_cocos_links(*args, **kwargs)


def reconcile_standard_name_unit_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_unit_edges."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_unit_edges(*args, **kwargs)


def reconcile_constraint_fit_artifacts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_constraint_fit_artifacts."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_constraint_fit_artifacts(*args, **kwargs)


# Terminal / dead name stages — refined-away or given-up names whose edges have
# already been migrated off them; their denormalised scalar is left as-is.
_TERMINAL_NAME_STAGES = import_module(
    ".graph_store.graph_audits", __package__
)._TERMINAL_NAME_STAGES


def reconcile_standard_name_source_paths(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_source_paths."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_source_paths(*args, **kwargs)


def reconcile_catalog_status(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_catalog_status."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_catalog_status(*args, **kwargs)


def reconcile_reviewable_name_stage(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_reviewable_name_stage."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_reviewable_name_stage(*args, **kwargs)


def reconcile_descriptionless_composed_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_descriptionless_composed_names."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_descriptionless_composed_names(*args, **kwargs)


def reconcile_sourceless_pipeline_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_sourceless_pipeline_names."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_sourceless_pipeline_names(*args, **kwargs)


def reconcile_standard_name_dd_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_dd_edges."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_dd_edges(*args, **kwargs)


#: Ledger operation label recorded when a name's stored grammar segment
#: columns are rewritten to agree with the parse of its own canonical id.
#: Shared by the whole-graph sweep and the name-scoped repair so both writes
#: are queryable as one operation.
_SEGMENT_REALIGN_OPERATION = import_module(
    ".graph_store.graph_audits", __package__
)._SEGMENT_REALIGN_OPERATION


def reconcile_grammar_segments(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_grammar_segments."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_grammar_segments(*args, **kwargs)


def realign_grammar_segments_for_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.realign_grammar_segments_for_name."""
    from .graph_store import graph_audits

    return graph_audits.realign_grammar_segments_for_name(*args, **kwargs)


def _segment_drift_reason(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._segment_drift_reason."""
    from .graph_store import graph_audits

    return graph_audits._segment_drift_reason(*args, **kwargs)


def reconcile_standard_name_kinds(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.reconcile_standard_name_kinds."""
    from .graph_store import graph_audits

    return graph_audits.reconcile_standard_name_kinds(*args, **kwargs)


def _open_graph_client(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits._open_graph_client."""
    from .graph_store import graph_audits

    return graph_audits._open_graph_client(*args, **kwargs)


_error_sibling_reconcile_apply = import_module(
    ".graph_store.graph_audits", __package__
)._error_sibling_reconcile_apply

globals()["reconcile_" + "error_siblings"] = _error_sibling_reconcile_apply


def get_standard_name_source_stats(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.get_standard_name_source_stats."""
    from .graph_store import graph_audits

    return graph_audits.get_standard_name_source_stats(*args, **kwargs)


def write_run_provenance(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.write_run_provenance."""
    from .graph_store import graph_audits

    return graph_audits.write_run_provenance(*args, **kwargs)


# =============================================================================
# Review comment export for the anti-pattern feedback loop
# =============================================================================


def export_review_comments(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.export_review_comments."""
    from .graph_store import graph_audits

    return graph_audits.export_review_comments(*args, **kwargs)


# =============================================================================
# Graph-backed LLM cost tracking API
# =============================================================================

# Phase → StandardName.llm_cost_<suffix> field mapping.
_PHASE_TO_SN_COST_FIELD = import_module(
    ".graph_store.graph_audits", __package__
)._PHASE_TO_SN_COST_FIELD


def create_sn_run_open(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.create_sn_run_open."""
    from .graph_store import graph_audits

    return graph_audits.create_sn_run_open(*args, **kwargs)


def bump_sn_run_counter(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.bump_sn_run_counter."""
    from .graph_store import graph_audits

    return graph_audits.bump_sn_run_counter(*args, **kwargs)


def update_sn_run_progress(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.update_sn_run_progress."""
    from .graph_store import graph_audits

    return graph_audits.update_sn_run_progress(*args, **kwargs)


def finalize_sn_run(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.finalize_sn_run."""
    from .graph_store import graph_audits

    return graph_audits.finalize_sn_run(*args, **kwargs)


def mark_orphaned_standard_name_runs_stale(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.mark_orphaned_standard_name_runs_stale."""
    from .graph_store import graph_audits

    return graph_audits.mark_orphaned_standard_name_runs_stale(*args, **kwargs)


def normalize_stored_standard_name_kinds(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.normalize_stored_standard_name_kinds."""
    from .graph_store import graph_audits

    return graph_audits.normalize_stored_standard_name_kinds(*args, **kwargs)


def backfill_sn_run_telemetry(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_audits.backfill_sn_run_telemetry."""
    from .graph_store import graph_audits

    return graph_audits.backfill_sn_run_telemetry(*args, **kwargs)


# ---------------------------------------------------------------------------
# LLMCost pool normalisation
# ---------------------------------------------------------------------------
# Maps the raw ``phase`` labels used by callers to six canonical pool names.
_PHASE_TO_POOL = import_module(".graph_store.graph_audits", __package__)._PHASE_TO_POOL


def _normalize_pool(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims._normalize_pool."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims._normalize_pool(
        *args, services=_graph_foundation.services, **kwargs
    )


def record_llm_cost(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.record_llm_cost."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.record_llm_cost(
        *args, services=_graph_foundation.services, **kwargs
    )


def aggregate_spend_for_run(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_for_run."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_for_run(
        *args, services=_graph_foundation.services, **kwargs
    )


def aggregate_spend_per_phase(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_per_phase."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_per_phase(
        *args, services=_graph_foundation.services, **kwargs
    )


def aggregate_calls_per_phase(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_calls_per_phase."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_calls_per_phase(
        *args, services=_graph_foundation.services, **kwargs
    )


def aggregate_spend_per_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_per_name."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_per_name(
        *args, services=_graph_foundation.services, **kwargs
    )


def update_sn_per_phase_costs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.update_sn_per_phase_costs."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.update_sn_per_phase_costs(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Seed-and-expand claims — the worker pools' claim protocol
#
# Every claim function follows the same two-step shape:
#   1. Seed: pick one random eligible row, SET claimed_at + claim_token
#   2. Expand: claim up to batch_size−1 more with same (cluster_id, unit);
#      fallback: (physics_domain, unit) when cluster_id is absent
#   3. Read-back: query by claim_token — only our token's rows returned
#
# Compose targets StandardNameSource; the other four target StandardName.
# =============================================================================

DEFAULT_POOL_BATCH_SIZE = import_module(
    ".graph_store.graph_cost_and_claims", __package__
).DEFAULT_POOL_BATCH_SIZE


_PIPELINE_FROZEN_NAME_STAGES = import_module(
    ".graph_store.graph_cost_and_claims", __package__
)._PIPELINE_FROZEN_NAME_STAGES


# Maximum times a single StandardNameSource may be CLAIMED for name generation
# before it is treated as un-composable and excluded from further claims and
# from the pending-count.  Without this cap a source whose batch repeatedly
# fails (LLM omits it from the response, the batch errors, or its candidate is
# grammar-rejected and released) returns to ``status='extracted'`` unchanged and
# is re-claimed forever — ``total_processed`` never advances and the run wedges
# on the residue (observed full-DD build 2026-06-20: 30–468 sources, attempt
# loop, stall-guard fired).  ``attempt_count`` is incremented on every claim, so
# this bounds the loop regardless of WHICH release path returned the source; the
# excluded residue stays ``extracted`` and is revived by ``--reset-to extracted``
# (which clears ``attempt_count``) once compose/grammar improves.
_MAX_COMPOSE_CLAIM_ATTEMPTS = import_module(
    ".graph_store.graph_cost_and_claims", __package__
)._MAX_COMPOSE_CLAIM_ATTEMPTS


def _claim_sn_atomic(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims._claim_sn_atomic."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims._claim_sn_atomic(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- generate_name (StandardNameSource) -------------------------------------


def claim_generate_name_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_generate_name_batch."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_generate_name_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- enrich (StandardName, enriched_at IS NULL) -------------------------------


def claim_enrich_seed_and_expand(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_enrich_seed_and_expand."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_enrich_seed_and_expand(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- review names (StandardName, reviewed_name_at IS NULL) --------------------


def claim_review_names_seed_and_expand(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_review_names_seed_and_expand."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_review_names_seed_and_expand(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- review_name (name_stage='drafted', claim only) ---------------------------


REVIEW_NAME_ELIGIBILITY_WHERE = import_module(
    ".graph_store.graph_cost_and_claims", __package__
).REVIEW_NAME_ELIGIBILITY_WHERE

"""Canonical graph predicate for name-review eligibility.

The drain clause permits an explicitly scoped, already-passing review to be
restaged. A real description is mandatory, while deterministic placeholder
parents and all derived names remain outside the review pool.
"""


def claim_review_name_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_review_name_batch."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_review_name_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


# -- persist_reviewed_name (write the review + the stage transition) ----------


#: Resolution methods that authorize a stage decision. The quorum methods
#: represent completed reviewer chains; the cascade methods represent a
#: calibrated Jev decision or its explicit single-reviewer route. Other methods
#: scored with fewer seats than the configured chain requires.
QUORATE_RESOLUTION_METHODS = import_module(
    ".graph_store.graph_cost_and_claims", __package__
).QUORATE_RESOLUTION_METHODS


#: ``max_cycles_reached`` merges two blind seats that never agreed.  On a
#: two-seat chain that is the designed terminal outcome; on a chain that
#: defines an escalator it means the escalator itself did not answer.
_ESCALATOR_CHAIN_SIZE = import_module(
    ".graph_store.graph_cost_and_claims", __package__
)._ESCALATOR_CHAIN_SIZE


def _quorum_admits_acceptance(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence._quorum_admits_acceptance."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence._quorum_admits_acceptance(*args, **kwargs)


_SHA256_RE = import_module(
    ".graph_store.graph_review_persistence", __package__
)._SHA256_RE


def _authority_json_value(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence._authority_json_value."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence._authority_json_value(*args, **kwargs)


def _authority_payload_hash(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence._authority_payload_hash."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence._authority_payload_hash(*args, **kwargs)


def persist_reviewed_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence.persist_reviewed_name."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence.persist_reviewed_name(*args, **kwargs)


# -- review_docs (docs_stage='drafted', claim only) ---------------------------


def claim_review_docs_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence.claim_review_docs_batch."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence.claim_review_docs_batch(*args, **kwargs)


# -- persist_reviewed_docs (write the review + the docs_stage transition) -----


def persist_reviewed_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_review_persistence.persist_reviewed_docs."""
    from .graph_store import graph_review_persistence

    return graph_review_persistence.persist_reviewed_docs(*args, **kwargs)


STAGE_DOCS_FOR_RESCORE_QUERY = import_module(
    ".graph_store.graph_review_persistence", __package__
).STAGE_DOCS_FOR_RESCORE_QUERY

"""Compare-and-set query for one exact docs-only quorum rescore."""


# =============================================================================
# Refined name identity, rename and embedding persistence
# =============================================================================

REFINE_NAME_ATTEMPTS_SPENT = import_module(
    ".graph_store.graph_refinement", __package__
).REFINE_NAME_ATTEMPTS_SPENT

REFINE_NAME_ELIGIBILITY_WHERE = import_module(
    ".graph_store.graph_refinement", __package__
).REFINE_NAME_ELIGIBILITY_WHERE

RefinedNamePersistenceRefusalReason = import_module(
    ".graph_store.graph_refinement", __package__
).RefinedNamePersistenceRefusalReason

RefinedNamePersistenceRefusal = import_module(
    ".graph_store.graph_refinement", __package__
).RefinedNamePersistenceRefusal

_GENERATED_SUPERSESSION_PREFLIGHT_QUERY = import_module(
    ".graph_store.graph_refinement", __package__
)._GENERATED_SUPERSESSION_PREFLIGHT_QUERY


def _docs_rescore_ineligible(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._docs_rescore_ineligible."""
    from .graph_store import graph_refinement

    return graph_refinement._docs_rescore_ineligible(*args, **kwargs)


def _compute_embed_hash(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._compute_embed_hash."""
    from .graph_store import graph_refinement

    return graph_refinement._compute_embed_hash(*args, **kwargs)


def _meaning_preservation_assertion(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._meaning_preservation_assertion."""
    from .graph_store import graph_refinement

    return graph_refinement._meaning_preservation_assertion(*args, **kwargs)


def _predecessor_document_identity(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._predecessor_document_identity."""
    from .graph_store import graph_refinement

    return graph_refinement._predecessor_document_identity(*args, **kwargs)


def rename_preserves_meaning(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.rename_preserves_meaning."""
    from .graph_store import graph_refinement

    return graph_refinement.rename_preserves_meaning(*args, **kwargs)


def _classify_refined_name_persistence_refusal(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._classify_refined_name_persistence_refusal."""
    from .graph_store import graph_refinement

    return graph_refinement._classify_refined_name_persistence_refusal(*args, **kwargs)


def stage_docs_for_rescore(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.stage_docs_for_rescore."""
    from .graph_store import graph_refinement

    return graph_refinement.stage_docs_for_rescore(
        *args, services=_graph_foundation.services, **kwargs
    )


def _charge_refine_name_attempts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._charge_refine_name_attempts."""
    from .graph_store import graph_refinement

    return graph_refinement._charge_refine_name_attempts(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_refine_name_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.claim_refine_name_batch."""
    from .graph_store import graph_refinement

    return graph_refinement.claim_refine_name_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


def _embed_single_standard_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._embed_single_standard_name."""
    from .graph_store import graph_refinement

    return graph_refinement._embed_single_standard_name(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_embed_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.claim_embed_batch."""
    from .graph_store import graph_refinement

    return graph_refinement.claim_embed_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_embed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.release_embed_claims."""
    from .graph_store import graph_refinement

    return graph_refinement.release_embed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_embed_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.persist_embed_batch."""
    from .graph_store import graph_refinement

    return graph_refinement.persist_embed_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_refined_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.persist_refined_name."""
    from .graph_store import graph_refinement

    return graph_refinement.persist_refined_name(
        *args, services=_graph_foundation.services, **kwargs
    )


def carry_accepted_documentation_across_rename(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.carry_accepted_documentation_across_rename."""
    from .graph_store import graph_refinement

    return graph_refinement.carry_accepted_documentation_across_rename(
        *args, services=_graph_foundation.services, **kwargs
    )


def supersede_prior_source_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.supersede_prior_source_names."""
    from .graph_store import graph_supersession

    return graph_supersession.supersede_prior_source_names(*args, **kwargs)


def tombstone_supersede_into(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.tombstone_supersede_into."""
    from .graph_store import graph_supersession

    return graph_supersession.tombstone_supersede_into(*args, **kwargs)


SCALAR_SELECTED_DEDUP_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SCALAR_SELECTED_DEDUP_MANIFEST_SCHEMA
SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SCALAR_SELECTED_DEDUP_RECEIPT_SCHEMA


ScalarSelectedDedupConflict = import_module(
    ".graph_store.graph_supersession", __package__
).ScalarSelectedDedupConflict


def _scalar_selected_dedup_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._scalar_selected_dedup_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._scalar_selected_dedup_authority(*args, **kwargs)


def _lock_scalar_selected_dedup_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._lock_scalar_selected_dedup_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._lock_scalar_selected_dedup_authority(*args, **kwargs)


def _scalar_selected_dedup_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._scalar_selected_dedup_counts."""
    from .graph_store import graph_supersession

    return graph_supersession._scalar_selected_dedup_counts(*args, **kwargs)


def deduplicate_scalar_selected_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.deduplicate_scalar_selected_sources."""
    from .graph_store import graph_supersession

    return graph_supersession.deduplicate_scalar_selected_sources(*args, **kwargs)


SIGNED_SOURCE_DISPOSITION_ADJUDICATION_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_SOURCE_DISPOSITION_ADJUDICATION_SCHEMA
SIGNED_SOURCE_DISPOSITION_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_SOURCE_DISPOSITION_MANIFEST_SCHEMA
SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA
_SIGNED_SOURCE_DISPOSITIONS = import_module(
    ".graph_store.graph_supersession", __package__
)._SIGNED_SOURCE_DISPOSITIONS
_STRUCTURAL_LEGITIMACY_AUTHORITY_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
)._STRUCTURAL_LEGITIMACY_AUTHORITY_SCHEMA
_STRUCTURAL_LEGITIMACY_DISPOSITION = import_module(
    ".graph_store.graph_supersession", __package__
)._STRUCTURAL_LEGITIMACY_DISPOSITION
_ORPHAN_DISPOSITIONS = import_module(
    ".graph_store.graph_supersession", __package__
)._ORPHAN_DISPOSITIONS


SignedSourceDispositionConflict = import_module(
    ".graph_store.graph_supersession", __package__
).SignedSourceDispositionConflict


def _catalog_edit_adjudication_signature_hash(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._catalog_edit_adjudication_signature_hash."""
    from .graph_store import graph_supersession

    return graph_supersession._catalog_edit_adjudication_signature_hash(*args, **kwargs)


def _validate_signed_source_adjudication(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._validate_signed_source_adjudication."""
    from .graph_store import graph_supersession

    return graph_supersession._validate_signed_source_adjudication(*args, **kwargs)


def _validate_structural_legitimacy_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._validate_structural_legitimacy_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._validate_structural_legitimacy_authority(*args, **kwargs)


def _is_live_structural_child(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._is_live_structural_child."""
    from .graph_store import graph_supersession

    return graph_supersession._is_live_structural_child(*args, **kwargs)


def _signed_source_disposition_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_source_disposition_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_source_disposition_authority(*args, **kwargs)


def _lock_signed_source_disposition_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._lock_signed_source_disposition_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._lock_signed_source_disposition_authority(*args, **kwargs)


def _signed_source_disposition_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_source_disposition_counts."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_source_disposition_counts(*args, **kwargs)


def _signed_source_disposition_execution_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_source_disposition_execution_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_source_disposition_execution_authority(
        *args, **kwargs
    )


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

SEMANTIC_MIRROR_REPAIR_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SEMANTIC_MIRROR_REPAIR_MANIFEST_SCHEMA
SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA
_PROJECTED_SEMANTIC_SOURCE_TYPES = import_module(
    ".graph_store.graph_supersession", __package__
)._PROJECTED_SEMANTIC_SOURCE_TYPES


SemanticMirrorRepairConflict = import_module(
    ".graph_store.graph_supersession", __package__
).SemanticMirrorRepairConflict


def _semantic_mirror_repair_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._semantic_mirror_repair_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._semantic_mirror_repair_authority(*args, **kwargs)


def _lock_semantic_mirror_repair_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._lock_semantic_mirror_repair_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._lock_semantic_mirror_repair_authority(*args, **kwargs)


def _semantic_mirror_repair_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._semantic_mirror_repair_counts."""
    from .graph_store import graph_supersession

    return graph_supersession._semantic_mirror_repair_counts(*args, **kwargs)


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


_SIGNED_ORPHAN_RETIREMENT_DISPOSITION = import_module(
    ".graph_store.graph_supersession", __package__
)._SIGNED_ORPHAN_RETIREMENT_DISPOSITION


def _validate_signed_orphan_retirement_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._validate_signed_orphan_retirement_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._validate_signed_orphan_retirement_authority(
        *args, **kwargs
    )


SIGNED_DUAL_AUTHORITY_RETIREMENT_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_DUAL_AUTHORITY_RETIREMENT_MANIFEST_SCHEMA
SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA


SignedDualAuthorityRetirementConflict = import_module(
    ".graph_store.graph_supersession", __package__
).SignedDualAuthorityRetirementConflict


def _validate_signed_dual_authority_retirement(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._validate_signed_dual_authority_retirement."""
    from .graph_store import graph_supersession

    return graph_supersession._validate_signed_dual_authority_retirement(
        *args, **kwargs
    )


def _signed_dual_authority_retirement_manifest(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_dual_authority_retirement_manifest."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_dual_authority_retirement_manifest(
        *args, **kwargs
    )


def _signed_dual_authority_change_id(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._signed_dual_authority_change_id."""
    from .graph_store import graph_supersession

    return graph_supersession._signed_dual_authority_change_id(*args, **kwargs)


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
EXHAUSTED_ORPHAN_SUPERSESSION_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).EXHAUSTED_ORPHAN_SUPERSESSION_MANIFEST_SCHEMA
EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).EXHAUSTED_ORPHAN_SUPERSESSION_RECEIPT_SCHEMA


ExhaustedOrphanSupersessionConflict = import_module(
    ".graph_store.graph_supersession", __package__
).ExhaustedOrphanSupersessionConflict


def classify_sourceless_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.classify_sourceless_standard_names."""
    from .graph_store import graph_supersession

    return graph_supersession.classify_sourceless_standard_names(*args, **kwargs)


def _exhausted_orphan_supersession_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._exhausted_orphan_supersession_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._exhausted_orphan_supersession_authority(*args, **kwargs)


def _exhausted_orphan_change_id(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._exhausted_orphan_change_id."""
    from .graph_store import graph_supersession

    return graph_supersession._exhausted_orphan_change_id(*args, **kwargs)


def supersede_exhausted_standard_name_orphans(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.supersede_exhausted_standard_name_orphans."""
    from .graph_store import graph_supersession

    return graph_supersession.supersede_exhausted_standard_name_orphans(*args, **kwargs)


ANCESTOR_SUPERSESSION_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).ANCESTOR_SUPERSESSION_MANIFEST_SCHEMA
ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_supersession", __package__
).ANCESTOR_SUPERSESSION_RECEIPT_SCHEMA


SupersedeIntoAncestorConflict = import_module(
    ".graph_store.graph_supersession", __package__
).SupersedeIntoAncestorConflict


def _ancestor_supersession_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._ancestor_supersession_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._ancestor_supersession_authority(*args, **kwargs)


def _lock_ancestor_supersession_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._lock_ancestor_supersession_authority."""
    from .graph_store import graph_supersession

    return graph_supersession._lock_ancestor_supersession_authority(*args, **kwargs)


def _detach_stale_ancestor_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession._detach_stale_ancestor_sources."""
    from .graph_store import graph_supersession

    return graph_supersession._detach_stale_ancestor_sources(*args, **kwargs)


def supersede_into_ancestor(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_supersession.supersede_into_ancestor."""
    from .graph_store import graph_supersession

    return graph_supersession.supersede_into_ancestor(*args, **kwargs)


# =============================================================================
# Release helpers — seed-and-expand pools
# =============================================================================
#
# These are called by pool_loop when process() raises an exception so that
# claimed items are unlocked and become eligible for other workers.
#
# Every helper verifies ``claim_token`` (and ``expected_stage`` where
# applicable) in the WHERE clause before clearing claim state.  This
# prevents a late-arriving release from clobbering a fresh re-claim that
# was issued after the orphan sweep cleared the stale token.
#
# Return value: count of nodes actually released.  A return value less than
# len(ids) indicates concurrent intervention (orphan sweep or another
# worker) — callers may ignore this; it is logged at DEBUG.
#


def release_generate_name_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_generate_name_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_generate_name_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_generate_name_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_generate_name_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_generate_name_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_enrich_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_enrich_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_enrich_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_enrich_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_enrich_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_enrich_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_review_names_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_names_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_names_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_review_names_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_names_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_names_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_review_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_docs_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_docs_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_review_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_docs_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_docs_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


REFINE_CLAIM_RELEASE_MANIFEST_SCHEMA = import_module(
    ".graph_store.graph_claim_release", __package__
).REFINE_CLAIM_RELEASE_MANIFEST_SCHEMA

REFINE_CLAIM_RELEASE_RECEIPT_SCHEMA = import_module(
    ".graph_store.graph_claim_release", __package__
).REFINE_CLAIM_RELEASE_RECEIPT_SCHEMA

REFINE_CLAIM_RELEASE_PROJECTION_VERSION = import_module(
    ".graph_store.graph_claim_release", __package__
).REFINE_CLAIM_RELEASE_PROJECTION_VERSION


RefineClaimReleaseConflict = import_module(
    ".graph_store.graph_claim_release", __package__
).RefineClaimReleaseConflict


_LOCK_REFINE_CLAIM_RELEASE_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
)._LOCK_REFINE_CLAIM_RELEASE_QUERY


_READ_REFINE_CLAIM_RELEASE_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
)._READ_REFINE_CLAIM_RELEASE_QUERY


_WRITE_REFINE_CLAIM_RELEASE_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
)._WRITE_REFINE_CLAIM_RELEASE_QUERY


def _sorted_refine_claim_projection(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._sorted_refine_claim_projection."""
    from .graph_store import graph_claim_release

    return graph_claim_release._sorted_refine_claim_projection(
        *args, services=_graph_foundation.services, **kwargs
    )


def _project_refine_claim_release_state(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._project_refine_claim_release_state."""
    from .graph_store import graph_claim_release

    return graph_claim_release._project_refine_claim_release_state(
        *args, services=_graph_foundation.services, **kwargs
    )


def _predicted_refine_claim_release_state(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._predicted_refine_claim_release_state."""
    from .graph_store import graph_claim_release

    return graph_claim_release._predicted_refine_claim_release_state(
        *args, services=_graph_foundation.services, **kwargs
    )


def _refine_claim_release_manifest(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._refine_claim_release_manifest."""
    from .graph_store import graph_claim_release

    return graph_claim_release._refine_claim_release_manifest(
        *args, services=_graph_foundation.services, **kwargs
    )


def _refine_claim_no_change_receipt(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._refine_claim_no_change_receipt."""
    from .graph_store import graph_claim_release

    return graph_claim_release._refine_claim_no_change_receipt(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_stranded_refine_claim(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_stranded_refine_claim."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_stranded_refine_claim(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_refine_name_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_refine_name_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_refine_name_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def resubmit_pinned_rename_for_review(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.resubmit_pinned_rename_for_review."""
    from .graph_store import graph_claim_release

    return graph_claim_release.resubmit_pinned_rename_for_review(
        *args, services=_graph_foundation.services, **kwargs
    )


RESCORE_ELIGIBILITY_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
).RESCORE_ELIGIBILITY_QUERY


def stage_name_for_rescore(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.stage_name_for_rescore."""
    from .graph_store import graph_claim_release

    return graph_claim_release.stage_name_for_rescore(
        *args, services=_graph_foundation.services, **kwargs
    )


def restore_name_after_failed_rescore(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.restore_name_after_failed_rescore."""
    from .graph_store import graph_claim_release

    return graph_claim_release.restore_name_after_failed_rescore(
        *args, services=_graph_foundation.services, **kwargs
    )


#: Stop reasons proving that no further attempt on this name can succeed.
#: A collision names an identity the graph already holds, and refinement may
#: not take an occupied identity — folding into it carries source-migration
#: semantics that belong to ``sn edit``. A grammar or vocabulary failure is
#: reproduced by the same prompt every cycle. Measured over one run, 830 of
#: 830 collisions re-proposed the same identity, across two model vendors.
_TERMINAL_REFINE_STOP_REASONS = import_module(
    ".graph_store.graph_claim_release", __package__
)._TERMINAL_REFINE_STOP_REASONS


def stop_refine_name_attempt(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.stop_refine_name_attempt."""
    from .graph_store import graph_claim_release

    return graph_claim_release.stop_refine_name_attempt(
        *args, services=_graph_foundation.services, **kwargs
    )


def _mark_refine_docs_exhausted(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._mark_refine_docs_exhausted."""
    from .graph_store import graph_claim_release

    return graph_claim_release._mark_refine_docs_exhausted(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Docs-claim race resolution
# =============================================================================
# The shared _claim_sn_atomic seed binds its candidate row at MATCH time, then
# acquires the node write-lock at the SET. When two pool replicas race for the
# same drafted/pending node, both MATCH it as eligible; the SET serialises on
# the lock, so the LAST committer's claim_token / claim_seq wins — but BOTH
# replicas returned the node as "claimed" and proceeded to call the LLM. That
# phantom is the dominant docs-cost driver: a single accepted name was
# re-reviewed 27× (≈48 wasted reviewer LLM calls) purely because losing-race
# replicas still fired their reviewer quorum.
#
# _verify_docs_claim_winners closes the window by re-reading COMMITTED state
# after a short settle: each claim stamps a strictly-increasing per-node
# claim_seq, so a node is a genuine win only if it STILL holds our claim_token
# AND its committed claim_seq equals the one we were assigned — i.e. no later
# racer superseded us. The settle lets the lock-serialised claim burst finish
# committing before we read, so exactly one replica (the final committer) sees
# its own claim_seq as current; all earlier racers observe a higher seq and
# self-exclude before any LLM call. This is the generate_docs / review_docs
# analogue of the claim_token verify mandated for all claim functions, hardened
# against the "every racer is momentarily the token holder" false-positive.


def _verify_docs_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_docs_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_docs_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


def _verify_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Name-claim race resolution
# =============================================================================
# Identical mechanism to the docs axis (see _verify_docs_claim_winners): two
# pool replicas both bind the same eligible StandardName at the shared seed
# MATCH, the lock-serialised SET lets the LAST claim_token / claim_seq win, but
# BOTH replicas already returned the node and proceed to the (paid) name LLM
# call. Live evidence: review_name ran 339 reviewer calls for 64 distinct names
# (~5× amplification) on a multi-replica run. _verify_name_claim_winners closes
# the window via the same settle + claim_seq check as the docs axis: after the
# claim burst settles, a node is a genuine win only if it STILL holds our
# claim_token at the eligible name_stage with our assigned claim_seq; every
# earlier racer sees a higher committed seq and is dropped before any LLM call.


def _verify_name_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_name_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_name_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


def _verify_source_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_source_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_source_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# generate_docs — claim / persist / release
# =============================================================================
# Stage gate: name_stage='accepted' AND docs_stage='pending'
# The claim does NOT transition docs_stage (persist_generated_docs does that).


def claim_generate_docs_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.claim_generate_docs_batch."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.claim_generate_docs_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_generated_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.persist_generated_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.persist_generated_docs(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_generate_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_generate_docs_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_generate_docs_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_generate_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_generate_docs_failed_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_generate_docs_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# refine_docs — claim / persist / release (DocsRevision snapshot architecture)
# =============================================================================
# Stage gate: docs_stage='reviewed' AND (score below threshold OR open docs hold)
#             AND docs_chain_length < rotation_cap
# Fundamentally different from refine_name: docs refine is IN-PLACE on the
# existing SN node.  The OLD docs are snapshotted into a DocsRevision node
# linked via DOCS_REVISION_OF before the SN is updated.


def claim_refine_docs_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.claim_refine_docs_batch."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.claim_refine_docs_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Persist — refine_docs (snapshot current docs → DocsRevision, update SN)
# =============================================================================


def persist_refined_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.persist_refined_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.persist_refined_docs(
        *args, services=_graph_foundation.services, **kwargs
    )


def reset_standard_name_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.reset_standard_name_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.reset_standard_name_docs(
        *args, services=_graph_foundation.services, **kwargs
    )


def stamp_harmonized_families(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.stamp_harmonized_families."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.stamp_harmonized_families(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_refine_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_refine_docs_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_refine_docs_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_refine_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_refine_docs_failed_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_refine_docs_failed_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# Routing helper: desc-name similarity gate → REFINE_DOCS
# =============================================================================


def mark_for_refine_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.mark_for_refine_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.mark_for_refine_docs(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_all_orphan_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.release_all_orphan_claims."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.release_all_orphan_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


# ═══════════════════════════════════════════════════════════════════════
# Pool pending counts — single round-trip query for all 6 pools
# ═══════════════════════════════════════════════════════════════════════


def pool_pending_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.pool_pending_counts."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.pool_pending_counts(
        *args, services=_graph_foundation.services, **kwargs
    )


def scoped_terminal_residue(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.scoped_terminal_residue."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.scoped_terminal_residue(
        *args, services=_graph_foundation.services, **kwargs
    )


# =============================================================================
# enrich_parents — claim / fetch-children / persist / release
# =============================================================================
# Coverage deadlock break: a derived parent materialised by
# ``_materialize_derived_parent_rows`` carries the
# ``DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER`` until something writes a real
# description.  The placeholder excludes it from REVIEW_NAME (the claim gate
# drops it), and with no reviewer_score_name it is excluded from generate_docs
# (``reviewer_score_name IS NOT NULL`` gate) — neither reviewable nor
# documentable.  ``enrich_parents`` synthesises a real description GENERALISED
# over the parent's accepted children, embeds it locally, and routes the parent
# to ``name_stage='drafted'`` so it flows review → accept → docs like any other
# name.  Childless derived parents are legitimately unscoped and never claimed.


def _verify_enrich_parents_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._verify_enrich_parents_claim_winners."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._verify_enrich_parents_claim_winners(
        *args, services=_graph_foundation.services, **kwargs
    )


def claim_enrich_parents_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.claim_enrich_parents_batch."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.claim_enrich_parents_batch(
        *args, services=_graph_foundation.services, **kwargs
    )


def fetch_derived_parent_children(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.fetch_derived_parent_children."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.fetch_derived_parent_children(
        *args, services=_graph_foundation.services, **kwargs
    )


_STRUCTURAL_AUTHORITY_MECHANISM = import_module(
    ".graph_store.graph_parent_lifecycle", __package__
)._STRUCTURAL_AUTHORITY_MECHANISM

_STRUCTURAL_AUTHORITY_SCHEMA = import_module(
    ".graph_store.graph_parent_lifecycle", __package__
)._STRUCTURAL_AUTHORITY_SCHEMA

_STRUCTURAL_AUTHORITY_CANONICALIZATION = import_module(
    ".graph_store.graph_parent_lifecycle", __package__
)._STRUCTURAL_AUTHORITY_CANONICALIZATION


StructuralAuthorityConflict = import_module(
    ".graph_store.graph_parent_lifecycle", __package__
).StructuralAuthorityConflict


def _structural_authority_identities(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._structural_authority_identities."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._structural_authority_identities(
        *args, services=_graph_foundation.services, **kwargs
    )


def _structural_authority_snapshot(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._structural_authority_snapshot."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._structural_authority_snapshot(
        *args, services=_graph_foundation.services, **kwargs
    )


def _structural_authority_record(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._structural_authority_record."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._structural_authority_record(
        *args, services=_graph_foundation.services, **kwargs
    )


def _persist_structural_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._persist_structural_authority."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._persist_structural_authority(
        *args, services=_graph_foundation.services, **kwargs
    )


def persist_enriched_parent(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.persist_enriched_parent."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.persist_enriched_parent(
        *args, services=_graph_foundation.services, **kwargs
    )


#: Path scalars that a ``PRODUCED_NAME`` edge is meant to back. On a name no
#: source produces they are stale copies of a source that has since moved away,
#: and they read as provenance the graph itself contradicts.
_UNBACKED_SOURCE_PATH_SCALARS = import_module(
    ".graph_store.graph_parent_lifecycle", __package__
)._UNBACKED_SOURCE_PATH_SCALARS


def _structural_accept_route(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._structural_accept_route."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._structural_accept_route(
        *args, services=_graph_foundation.services, **kwargs
    )


def structural_accept_derived_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.structural_accept_derived_parents."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.structural_accept_derived_parents(
        *args, services=_graph_foundation.services, **kwargs
    )


def backfill_structural_name_authorities(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.backfill_structural_name_authorities."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.backfill_structural_name_authorities(
        *args, services=_graph_foundation.services, **kwargs
    )


def find_orphan_parent_source_candidates(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.find_orphan_parent_source_candidates."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.find_orphan_parent_source_candidates(
        *args, services=_graph_foundation.services, **kwargs
    )


def classify_orphan_parent_source_candidates(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.classify_orphan_parent_source_candidates."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.classify_orphan_parent_source_candidates(
        *args, services=_graph_foundation.services, **kwargs
    )


def reconcile_orphan_parent_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.reconcile_orphan_parent_sources."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.reconcile_orphan_parent_sources(
        *args, services=_graph_foundation.services, **kwargs
    )


def reconcile_orphan_parent_sources_batched(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.reconcile_orphan_parent_sources_batched."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.reconcile_orphan_parent_sources_batched(
        *args, services=_graph_foundation.services, **kwargs
    )


def release_enrich_parents_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.release_enrich_parents_claims."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.release_enrich_parents_claims(
        *args, services=_graph_foundation.services, **kwargs
    )


# Preserve the moved callables for signature and source inspection.
_moved_graph_foundation = import_module(".graph_store.graph_foundation", __package__)
_ensure_json.__wrapped__ = _moved_graph_foundation._ensure_json
_ensure_list.__wrapped__ = _moved_graph_foundation._ensure_list
_scalar_domain.__wrapped__ = _moved_graph_foundation._scalar_domain
reclassify_standard_name_domain.__wrapped__ = (
    _moved_graph_foundation.reclassify_standard_name_domain
)
normalize_name_id.__wrapped__ = _moved_graph_foundation.normalize_name_id
_public_segment_token_index.__wrapped__ = (
    _moved_graph_foundation._public_segment_token_index
)
_public_grammar_segment_order.__wrapped__ = (
    _moved_graph_foundation._public_grammar_segment_order
)
_segments_from_ir.__wrapped__ = _moved_graph_foundation._segments_from_ir
_parse_grammar.__wrapped__ = _moved_graph_foundation._parse_grammar
_semantic_ir_expression.__wrapped__ = _moved_graph_foundation._semantic_ir_expression
strict_review_grammar_context.__wrapped__ = (
    _moved_graph_foundation.strict_review_grammar_context
)
_compute_link_status.__wrapped__ = _moved_graph_foundation._compute_link_status
_extract_links_from_docs.__wrapped__ = _moved_graph_foundation._extract_links_from_docs
_moved_graph_audits = import_module(".graph_store.graph_audits", __package__)
claim_compose_sources.__wrapped__ = _moved_graph_audits.claim_compose_sources
count_extracted_for_domain.__wrapped__ = _moved_graph_audits.count_extracted_for_domain
count_eligible_compose_sources.__wrapped__ = (
    _moved_graph_audits.count_eligible_compose_sources
)
claim_review_names.__wrapped__ = _moved_graph_audits.claim_review_names
release_review_claims.__wrapped__ = _moved_graph_audits.release_review_claims
refresh_name_claims.__wrapped__ = _moved_graph_audits.refresh_name_claims
promote_stranded_reviewed.__wrapped__ = _moved_graph_audits.promote_stranded_reviewed
_docs_axis_rederive_query.__wrapped__ = _moved_graph_audits._docs_axis_rederive_query
reconcile_docs_axis_from_reviews.__wrapped__ = (
    _moved_graph_audits.reconcile_docs_axis_from_reviews
)
_standard_name_source_upstream_present_cypher.__wrapped__ = (
    _moved_graph_audits._standard_name_source_upstream_present_cypher
)
reconcile_standard_name_sources.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_sources
)
fetch_manifest_source_release_rows.__wrapped__ = (
    _moved_graph_audits.fetch_manifest_source_release_rows
)
reconcile_source_status_liveness.__wrapped__ = (
    _moved_graph_audits.reconcile_source_status_liveness
)
retire_unreachable_hint_edits.__wrapped__ = (
    _moved_graph_audits.retire_unreachable_hint_edits
)
reconcile_vocab_gaps.__wrapped__ = _moved_graph_audits.reconcile_vocab_gaps
triage_vocab_gaps.__wrapped__ = _moved_graph_audits.triage_vocab_gaps
revive_unit_skipped_sources.__wrapped__ = (
    _moved_graph_audits.revive_unit_skipped_sources
)
_isn_grammar_context.__wrapped__ = _moved_graph_audits._isn_grammar_context
isn_vocabulary_signature.__wrapped__ = _moved_graph_audits.isn_vocabulary_signature
retry_vocab_gap_sources_on_grammar_change.__wrapped__ = (
    _moved_graph_audits.retry_vocab_gap_sources_on_grammar_change
)
reconcile_provenance.__wrapped__ = _moved_graph_audits.reconcile_provenance
reconcile_catalog_edit_origins.__wrapped__ = (
    _moved_graph_audits.reconcile_catalog_edit_origins
)
reconcile_standard_name_cocos_links.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_cocos_links
)
reconcile_standard_name_unit_edges.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_unit_edges
)
reconcile_constraint_fit_artifacts.__wrapped__ = (
    _moved_graph_audits.reconcile_constraint_fit_artifacts
)
reconcile_standard_name_source_paths.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_source_paths
)
reconcile_catalog_status.__wrapped__ = _moved_graph_audits.reconcile_catalog_status
reconcile_reviewable_name_stage.__wrapped__ = (
    _moved_graph_audits.reconcile_reviewable_name_stage
)
reconcile_descriptionless_composed_names.__wrapped__ = (
    _moved_graph_audits.reconcile_descriptionless_composed_names
)
reconcile_sourceless_pipeline_names.__wrapped__ = (
    _moved_graph_audits.reconcile_sourceless_pipeline_names
)
reconcile_standard_name_dd_edges.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_dd_edges
)
reconcile_grammar_segments.__wrapped__ = _moved_graph_audits.reconcile_grammar_segments
realign_grammar_segments_for_name.__wrapped__ = (
    _moved_graph_audits.realign_grammar_segments_for_name
)
_segment_drift_reason.__wrapped__ = _moved_graph_audits._segment_drift_reason
reconcile_standard_name_kinds.__wrapped__ = (
    _moved_graph_audits.reconcile_standard_name_kinds
)
_open_graph_client.__wrapped__ = _moved_graph_audits._open_graph_client
get_standard_name_source_stats.__wrapped__ = (
    _moved_graph_audits.get_standard_name_source_stats
)
write_run_provenance.__wrapped__ = _moved_graph_audits.write_run_provenance
export_review_comments.__wrapped__ = _moved_graph_audits.export_review_comments
create_sn_run_open.__wrapped__ = _moved_graph_audits.create_sn_run_open
bump_sn_run_counter.__wrapped__ = _moved_graph_audits.bump_sn_run_counter
update_sn_run_progress.__wrapped__ = _moved_graph_audits.update_sn_run_progress
finalize_sn_run.__wrapped__ = _moved_graph_audits.finalize_sn_run
mark_orphaned_standard_name_runs_stale.__wrapped__ = (
    _moved_graph_audits.mark_orphaned_standard_name_runs_stale
)
normalize_stored_standard_name_kinds.__wrapped__ = (
    _moved_graph_audits.normalize_stored_standard_name_kinds
)
backfill_sn_run_telemetry.__wrapped__ = _moved_graph_audits.backfill_sn_run_telemetry
_moved_graph_review_persistence = import_module(
    ".graph_store.graph_review_persistence", __package__
)
_quorum_admits_acceptance.__wrapped__ = (
    _moved_graph_review_persistence._quorum_admits_acceptance
)
_authority_json_value.__wrapped__ = (
    _moved_graph_review_persistence._authority_json_value
)
_authority_payload_hash.__wrapped__ = (
    _moved_graph_review_persistence._authority_payload_hash
)
persist_reviewed_name.__wrapped__ = (
    _moved_graph_review_persistence.persist_reviewed_name
)
claim_review_docs_batch.__wrapped__ = (
    _moved_graph_review_persistence.claim_review_docs_batch
)
persist_reviewed_docs.__wrapped__ = (
    _moved_graph_review_persistence.persist_reviewed_docs
)
_moved_graph_supersession = import_module(
    ".graph_store.graph_supersession", __package__
)
supersede_prior_source_names.__wrapped__ = (
    _moved_graph_supersession.supersede_prior_source_names
)
tombstone_supersede_into.__wrapped__ = (
    _moved_graph_supersession.tombstone_supersede_into
)
_scalar_selected_dedup_authority.__wrapped__ = (
    _moved_graph_supersession._scalar_selected_dedup_authority
)
_lock_scalar_selected_dedup_authority.__wrapped__ = (
    _moved_graph_supersession._lock_scalar_selected_dedup_authority
)
_scalar_selected_dedup_counts.__wrapped__ = (
    _moved_graph_supersession._scalar_selected_dedup_counts
)
deduplicate_scalar_selected_sources.__wrapped__ = (
    _moved_graph_supersession.deduplicate_scalar_selected_sources
)
_catalog_edit_adjudication_signature_hash.__wrapped__ = (
    _moved_graph_supersession._catalog_edit_adjudication_signature_hash
)
_validate_signed_source_adjudication.__wrapped__ = (
    _moved_graph_supersession._validate_signed_source_adjudication
)
_validate_structural_legitimacy_authority.__wrapped__ = (
    _moved_graph_supersession._validate_structural_legitimacy_authority
)
_is_live_structural_child.__wrapped__ = (
    _moved_graph_supersession._is_live_structural_child
)
_signed_source_disposition_authority.__wrapped__ = (
    _moved_graph_supersession._signed_source_disposition_authority
)
_lock_signed_source_disposition_authority.__wrapped__ = (
    _moved_graph_supersession._lock_signed_source_disposition_authority
)
_signed_source_disposition_counts.__wrapped__ = (
    _moved_graph_supersession._signed_source_disposition_counts
)
_signed_source_disposition_execution_authority.__wrapped__ = (
    _moved_graph_supersession._signed_source_disposition_execution_authority
)
_semantic_mirror_repair_authority.__wrapped__ = (
    _moved_graph_supersession._semantic_mirror_repair_authority
)
_lock_semantic_mirror_repair_authority.__wrapped__ = (
    _moved_graph_supersession._lock_semantic_mirror_repair_authority
)
_semantic_mirror_repair_counts.__wrapped__ = (
    _moved_graph_supersession._semantic_mirror_repair_counts
)
_validate_signed_orphan_retirement_authority.__wrapped__ = (
    _moved_graph_supersession._validate_signed_orphan_retirement_authority
)
_validate_signed_dual_authority_retirement.__wrapped__ = (
    _moved_graph_supersession._validate_signed_dual_authority_retirement
)
_signed_dual_authority_retirement_manifest.__wrapped__ = (
    _moved_graph_supersession._signed_dual_authority_retirement_manifest
)
_signed_dual_authority_change_id.__wrapped__ = (
    _moved_graph_supersession._signed_dual_authority_change_id
)
classify_sourceless_standard_names.__wrapped__ = (
    _moved_graph_supersession.classify_sourceless_standard_names
)
_exhausted_orphan_supersession_authority.__wrapped__ = (
    _moved_graph_supersession._exhausted_orphan_supersession_authority
)
_exhausted_orphan_change_id.__wrapped__ = (
    _moved_graph_supersession._exhausted_orphan_change_id
)
supersede_exhausted_standard_name_orphans.__wrapped__ = (
    _moved_graph_supersession.supersede_exhausted_standard_name_orphans
)
_ancestor_supersession_authority.__wrapped__ = (
    _moved_graph_supersession._ancestor_supersession_authority
)
_lock_ancestor_supersession_authority.__wrapped__ = (
    _moved_graph_supersession._lock_ancestor_supersession_authority
)
_detach_stale_ancestor_sources.__wrapped__ = (
    _moved_graph_supersession._detach_stale_ancestor_sources
)
supersede_into_ancestor.__wrapped__ = _moved_graph_supersession.supersede_into_ancestor

_moved_graph_doc_links = import_module(".graph_store.graph_doc_links", __package__)
claim_unresolved_links.__wrapped__ = _moved_graph_doc_links.claim_unresolved_links
resolve_links_batch.__wrapped__ = _moved_graph_doc_links.resolve_links_batch
_sanitize_doc_text.__wrapped__ = _moved_graph_doc_links._sanitize_doc_text
_doc_link_mismatches.__wrapped__ = _moved_graph_doc_links._doc_link_mismatches
_normalize_bare_doc_links.__wrapped__ = _moved_graph_doc_links._normalize_bare_doc_links
resolve_doc_links.__wrapped__ = _moved_graph_doc_links.resolve_doc_links
_moved_graph_docs_persistence = import_module(
    ".graph_store.graph_docs_persistence", __package__
)
claim_generate_docs_batch.__wrapped__ = (
    _moved_graph_docs_persistence.claim_generate_docs_batch
)
persist_generated_docs.__wrapped__ = (
    _moved_graph_docs_persistence.persist_generated_docs
)
release_generate_docs_claims.__wrapped__ = (
    _moved_graph_docs_persistence.release_generate_docs_claims
)
release_generate_docs_failed_claims.__wrapped__ = (
    _moved_graph_docs_persistence.release_generate_docs_failed_claims
)
claim_refine_docs_batch.__wrapped__ = (
    _moved_graph_docs_persistence.claim_refine_docs_batch
)
persist_refined_docs.__wrapped__ = _moved_graph_docs_persistence.persist_refined_docs
reset_standard_name_docs.__wrapped__ = (
    _moved_graph_docs_persistence.reset_standard_name_docs
)
stamp_harmonized_families.__wrapped__ = (
    _moved_graph_docs_persistence.stamp_harmonized_families
)
release_refine_docs_claims.__wrapped__ = (
    _moved_graph_docs_persistence.release_refine_docs_claims
)
release_refine_docs_failed_claims.__wrapped__ = (
    _moved_graph_docs_persistence.release_refine_docs_failed_claims
)
mark_for_refine_docs.__wrapped__ = _moved_graph_docs_persistence.mark_for_refine_docs
_moved_graph_cost_and_claims = import_module(
    ".graph_store.graph_cost_and_claims", __package__
)
_normalize_pool.__wrapped__ = _moved_graph_cost_and_claims._normalize_pool
record_llm_cost.__wrapped__ = _moved_graph_cost_and_claims.record_llm_cost
aggregate_spend_for_run.__wrapped__ = (
    _moved_graph_cost_and_claims.aggregate_spend_for_run
)
aggregate_spend_per_phase.__wrapped__ = (
    _moved_graph_cost_and_claims.aggregate_spend_per_phase
)
aggregate_calls_per_phase.__wrapped__ = (
    _moved_graph_cost_and_claims.aggregate_calls_per_phase
)
aggregate_spend_per_name.__wrapped__ = (
    _moved_graph_cost_and_claims.aggregate_spend_per_name
)
update_sn_per_phase_costs.__wrapped__ = (
    _moved_graph_cost_and_claims.update_sn_per_phase_costs
)
_claim_sn_atomic.__wrapped__ = _moved_graph_cost_and_claims._claim_sn_atomic
claim_generate_name_batch.__wrapped__ = (
    _moved_graph_cost_and_claims.claim_generate_name_batch
)
claim_enrich_seed_and_expand.__wrapped__ = (
    _moved_graph_cost_and_claims.claim_enrich_seed_and_expand
)
claim_review_names_seed_and_expand.__wrapped__ = (
    _moved_graph_cost_and_claims.claim_review_names_seed_and_expand
)
claim_review_name_batch.__wrapped__ = (
    _moved_graph_cost_and_claims.claim_review_name_batch
)
_moved_graph_lineage = import_module(".graph_store.graph_lineage", __package__)
get_source_name_mapping.__wrapped__ = _moved_graph_lineage.get_source_name_mapping
_get_rich_source_name_mapping.__wrapped__ = (
    _moved_graph_lineage._get_rich_source_name_mapping
)
fetch_low_score_sources.__wrapped__ = _moved_graph_lineage.fetch_low_score_sources
fetch_review_feedback_for_sources.__wrapped__ = (
    _moved_graph_lineage.fetch_review_feedback_for_sources
)
fetch_reviewer_history_for_sources.__wrapped__ = (
    _moved_graph_lineage.fetch_reviewer_history_for_sources
)
fetch_docs_review_feedback_for_standard_names.__wrapped__ = (
    _moved_graph_lineage.fetch_docs_review_feedback_for_standard_names
)
fetch_docs_review_feedback_for_sns.__wrapped__ = (
    _moved_graph_lineage.fetch_docs_review_feedback_for_sns
)
_filter_admissible_parents.__wrapped__ = _moved_graph_lineage._filter_admissible_parents
_emit_magnitude_of_edges.__wrapped__ = _moved_graph_lineage._emit_magnitude_of_edges
_bootstrap_missing_derived_parent_targets.__wrapped__ = (
    _moved_graph_lineage._bootstrap_missing_derived_parent_targets
)
_write_standard_name_edges.__wrapped__ = _moved_graph_lineage._write_standard_name_edges
_clear_superseded_child_derived_parent_edges.__wrapped__ = (
    _moved_graph_lineage._clear_superseded_child_derived_parent_edges
)
rederive_structural_edges.__wrapped__ = _moved_graph_lineage.rederive_structural_edges
_rewire_has_parent_off_superseded.__wrapped__ = (
    _moved_graph_lineage._rewire_has_parent_off_superseded
)
_parse_parent_grammar.__wrapped__ = _moved_graph_lineage._parse_parent_grammar
_query_seedable_derived_parents.__wrapped__ = (
    _moved_graph_lineage._query_seedable_derived_parents
)
_query_legacy_repairable_derived_parents.__wrapped__ = (
    _moved_graph_lineage._query_legacy_repairable_derived_parents
)
_query_derived_parents_for_admission_cleanup.__wrapped__ = (
    _moved_graph_lineage._query_derived_parents_for_admission_cleanup
)
_delete_derived_parent_nodes.__wrapped__ = (
    _moved_graph_lineage._delete_derived_parent_nodes
)
sweep_orphaned_docs_revisions.__wrapped__ = (
    _moved_graph_lineage.sweep_orphaned_docs_revisions
)
_derived_parent_source_metadata.__wrapped__ = (
    _moved_graph_lineage._derived_parent_source_metadata
)
_validate_derived_parent_identity.__wrapped__ = (
    _moved_graph_lineage._validate_derived_parent_identity
)
_eligible_derived_parent_unit_children.__wrapped__ = (
    _moved_graph_lineage._eligible_derived_parent_unit_children
)
_materialize_derived_parent_rows.__wrapped__ = (
    _moved_graph_lineage._materialize_derived_parent_rows
)
_materialize_derived_parent_rows_batched.__wrapped__ = (
    _moved_graph_lineage._materialize_derived_parent_rows_batched
)
stamp_standard_name_units.__wrapped__ = _moved_graph_lineage.stamp_standard_name_units
seed_parent_sources.__wrapped__ = _moved_graph_lineage.seed_parent_sources
normalize_derived_parent_lifecycle.__wrapped__ = (
    _moved_graph_lineage.normalize_derived_parent_lifecycle
)
_canonicalize_lifecycleless_manifest_value.__wrapped__ = (
    _moved_graph_lineage._canonicalize_lifecycleless_manifest_value
)
_derived_parent_unit_from_eligible_children.__wrapped__ = (
    _moved_graph_lineage._derived_parent_unit_from_eligible_children
)
_spurious_stub_binding_authority.__wrapped__ = (
    _moved_graph_lineage._spurious_stub_binding_authority
)
_partition_lifecycleless_stub_rows.__wrapped__ = (
    _moved_graph_lineage._partition_lifecycleless_stub_rows
)
_lifecycleless_stub_manifest.__wrapped__ = (
    _moved_graph_lineage._lifecycleless_stub_manifest
)
_reconcile_spurious_stub_source_scalars.__wrapped__ = (
    _moved_graph_lineage._reconcile_spurious_stub_source_scalars
)
_query_skeleton_placeholders_for_cleanup.__wrapped__ = (
    _moved_graph_lineage._query_skeleton_placeholders_for_cleanup
)

_moved_graph_catalog_reset = import_module(
    ".graph_store.graph_catalog_reset", __package__
)
write_vocab_gaps.__wrapped__ = _moved_graph_catalog_reset.write_vocab_gaps
fetch_vocab_gap_adjudications.__wrapped__ = (
    _moved_graph_catalog_reset.fetch_vocab_gap_adjudications
)
claim_names_for_validation.__wrapped__ = (
    _moved_graph_catalog_reset.claim_names_for_validation
)
mark_names_validated.__wrapped__ = _moved_graph_catalog_reset.mark_names_validated
release_validation_claims.__wrapped__ = (
    _moved_graph_catalog_reset.release_validation_claims
)
claim_names_for_embedding.__wrapped__ = (
    _moved_graph_catalog_reset.claim_names_for_embedding
)
mark_names_embedded.__wrapped__ = _moved_graph_catalog_reset.mark_names_embedded
release_embedding_claims.__wrapped__ = (
    _moved_graph_catalog_reset.release_embedding_claims
)
get_validated_names.__wrapped__ = _moved_graph_catalog_reset.get_validated_names
mark_names_consolidated.__wrapped__ = _moved_graph_catalog_reset.mark_names_consolidated
reset_standard_names.__wrapped__ = _moved_graph_catalog_reset.reset_standard_names
clear_standard_names.__wrapped__ = _moved_graph_catalog_reset.clear_standard_names
clear_sn_subsystem.__wrapped__ = _moved_graph_catalog_reset.clear_sn_subsystem

_moved_graph_claim_release = import_module(
    ".graph_store.graph_claim_release", __package__
)
release_generate_name_claims.__wrapped__ = (
    _moved_graph_claim_release.release_generate_name_claims
)
release_generate_name_failed_claims.__wrapped__ = (
    _moved_graph_claim_release.release_generate_name_failed_claims
)
release_enrich_claims.__wrapped__ = _moved_graph_claim_release.release_enrich_claims
release_enrich_failed_claims.__wrapped__ = (
    _moved_graph_claim_release.release_enrich_failed_claims
)
release_review_names_claims.__wrapped__ = (
    _moved_graph_claim_release.release_review_names_claims
)
release_review_names_failed_claims.__wrapped__ = (
    _moved_graph_claim_release.release_review_names_failed_claims
)
release_review_docs_claims.__wrapped__ = (
    _moved_graph_claim_release.release_review_docs_claims
)
release_review_docs_failed_claims.__wrapped__ = (
    _moved_graph_claim_release.release_review_docs_failed_claims
)
_sorted_refine_claim_projection.__wrapped__ = (
    _moved_graph_claim_release._sorted_refine_claim_projection
)
_project_refine_claim_release_state.__wrapped__ = (
    _moved_graph_claim_release._project_refine_claim_release_state
)
_predicted_refine_claim_release_state.__wrapped__ = (
    _moved_graph_claim_release._predicted_refine_claim_release_state
)
_refine_claim_release_manifest.__wrapped__ = (
    _moved_graph_claim_release._refine_claim_release_manifest
)
_refine_claim_no_change_receipt.__wrapped__ = (
    _moved_graph_claim_release._refine_claim_no_change_receipt
)
release_stranded_refine_claim.__wrapped__ = (
    _moved_graph_claim_release.release_stranded_refine_claim
)
release_refine_name_claims.__wrapped__ = (
    _moved_graph_claim_release.release_refine_name_claims
)
resubmit_pinned_rename_for_review.__wrapped__ = (
    _moved_graph_claim_release.resubmit_pinned_rename_for_review
)
stage_name_for_rescore.__wrapped__ = _moved_graph_claim_release.stage_name_for_rescore
restore_name_after_failed_rescore.__wrapped__ = (
    _moved_graph_claim_release.restore_name_after_failed_rescore
)
stop_refine_name_attempt.__wrapped__ = (
    _moved_graph_claim_release.stop_refine_name_attempt
)
_mark_refine_docs_exhausted.__wrapped__ = (
    _moved_graph_claim_release._mark_refine_docs_exhausted
)
_verify_docs_claim_winners.__wrapped__ = (
    _moved_graph_claim_release._verify_docs_claim_winners
)
_verify_claim_winners.__wrapped__ = _moved_graph_claim_release._verify_claim_winners
_verify_name_claim_winners.__wrapped__ = (
    _moved_graph_claim_release._verify_name_claim_winners
)
_verify_source_claim_winners.__wrapped__ = (
    _moved_graph_claim_release._verify_source_claim_winners
)

_moved_graph_parent_lifecycle = import_module(
    ".graph_store.graph_parent_lifecycle", __package__
)
release_all_orphan_claims.__wrapped__ = (
    _moved_graph_parent_lifecycle.release_all_orphan_claims
)
pool_pending_counts.__wrapped__ = _moved_graph_parent_lifecycle.pool_pending_counts
scoped_terminal_residue.__wrapped__ = (
    _moved_graph_parent_lifecycle.scoped_terminal_residue
)
_verify_enrich_parents_claim_winners.__wrapped__ = (
    _moved_graph_parent_lifecycle._verify_enrich_parents_claim_winners
)
claim_enrich_parents_batch.__wrapped__ = (
    _moved_graph_parent_lifecycle.claim_enrich_parents_batch
)
fetch_derived_parent_children.__wrapped__ = (
    _moved_graph_parent_lifecycle.fetch_derived_parent_children
)
_structural_authority_identities.__wrapped__ = (
    _moved_graph_parent_lifecycle._structural_authority_identities
)
_structural_authority_snapshot.__wrapped__ = (
    _moved_graph_parent_lifecycle._structural_authority_snapshot
)
_structural_authority_record.__wrapped__ = (
    _moved_graph_parent_lifecycle._structural_authority_record
)
_persist_structural_authority.__wrapped__ = (
    _moved_graph_parent_lifecycle._persist_structural_authority
)
persist_enriched_parent.__wrapped__ = (
    _moved_graph_parent_lifecycle.persist_enriched_parent
)
_structural_accept_route.__wrapped__ = (
    _moved_graph_parent_lifecycle._structural_accept_route
)
structural_accept_derived_parents.__wrapped__ = (
    _moved_graph_parent_lifecycle.structural_accept_derived_parents
)
backfill_structural_name_authorities.__wrapped__ = (
    _moved_graph_parent_lifecycle.backfill_structural_name_authorities
)
find_orphan_parent_source_candidates.__wrapped__ = (
    _moved_graph_parent_lifecycle.find_orphan_parent_source_candidates
)
classify_orphan_parent_source_candidates.__wrapped__ = (
    _moved_graph_parent_lifecycle.classify_orphan_parent_source_candidates
)
reconcile_orphan_parent_sources.__wrapped__ = (
    _moved_graph_parent_lifecycle.reconcile_orphan_parent_sources
)
reconcile_orphan_parent_sources_batched.__wrapped__ = (
    _moved_graph_parent_lifecycle.reconcile_orphan_parent_sources_batched
)
release_enrich_parents_claims.__wrapped__ = (
    _moved_graph_parent_lifecycle.release_enrich_parents_claims
)

_moved_graph_claim_scope = import_module(".graph_store.graph_claim_scope", __package__)
query_pipeline_buckets.__wrapped__ = _moved_graph_claim_scope.query_pipeline_buckets
query_historical_cpi.__wrapped__ = _moved_graph_claim_scope.query_historical_cpi
get_extraction_candidates_dd.__wrapped__ = (
    _moved_graph_claim_scope.get_extraction_candidates_dd
)
get_extraction_candidates_signals.__wrapped__ = (
    _moved_graph_claim_scope.get_extraction_candidates_signals
)
get_existing_standard_names.__wrapped__ = (
    _moved_graph_claim_scope.get_existing_standard_names
)
get_named_source_ids.__wrapped__ = _moved_graph_claim_scope.get_named_source_ids
partition_focus_by_accepted.__wrapped__ = (
    _moved_graph_claim_scope.partition_focus_by_accepted
)
canonicalize_manifest_drain_paths.__wrapped__ = (
    _moved_graph_claim_scope.canonicalize_manifest_drain_paths
)
classify_manifest_drain_item.__wrapped__ = (
    _moved_graph_claim_scope.classify_manifest_drain_item
)
_resolved_manifest_drain_plan.__wrapped__ = (
    _moved_graph_claim_scope._resolved_manifest_drain_plan
)
build_manifest_drain_plan.__wrapped__ = (
    _moved_graph_claim_scope.build_manifest_drain_plan
)
_manifest_plan_fingerprint.__wrapped__ = (
    _moved_graph_claim_scope._manifest_plan_fingerprint
)
_lock_manifest_drain_authority.__wrapped__ = (
    _moved_graph_claim_scope._lock_manifest_drain_authority
)
claim_manifest_drain_scope.__wrapped__ = (
    _moved_graph_claim_scope.claim_manifest_drain_scope
)
prepare_manifest_drain_scope.__wrapped__ = (
    _moved_graph_claim_scope.prepare_manifest_drain_scope
)
clear_manifest_drain_scope.__wrapped__ = (
    _moved_graph_claim_scope.clear_manifest_drain_scope
)
release_manifest_drain_claims.__wrapped__ = (
    _moved_graph_claim_scope.release_manifest_drain_claims
)
finalize_manifest_drain_scope.__wrapped__ = (
    _moved_graph_claim_scope.finalize_manifest_drain_scope
)
scope_focus_names.__wrapped__ = _moved_graph_claim_scope.scope_focus_names
_claim_disposition.__wrapped__ = _moved_graph_claim_scope._claim_disposition
_claimed_run_ids.__wrapped__ = _moved_graph_claim_scope._claimed_run_ids
_collect_run_statuses.__wrapped__ = _moved_graph_claim_scope._collect_run_statuses
_exact_name_scope_refusals.__wrapped__ = (
    _moved_graph_claim_scope._exact_name_scope_refusals
)
scope_exact_standard_names.__wrapped__ = (
    _moved_graph_claim_scope.scope_exact_standard_names
)


_moved_graph_refinement = import_module(".graph_store.graph_refinement", __package__)
_docs_rescore_ineligible.__wrapped__ = _moved_graph_refinement._docs_rescore_ineligible
_compute_embed_hash.__wrapped__ = _moved_graph_refinement._compute_embed_hash
_meaning_preservation_assertion.__wrapped__ = (
    _moved_graph_refinement._meaning_preservation_assertion
)
_predecessor_document_identity.__wrapped__ = (
    _moved_graph_refinement._predecessor_document_identity
)
rename_preserves_meaning.__wrapped__ = _moved_graph_refinement.rename_preserves_meaning
_classify_refined_name_persistence_refusal.__wrapped__ = (
    _moved_graph_refinement._classify_refined_name_persistence_refusal
)
stage_docs_for_rescore.__wrapped__ = _moved_graph_refinement.stage_docs_for_rescore
_charge_refine_name_attempts.__wrapped__ = (
    _moved_graph_refinement._charge_refine_name_attempts
)
claim_refine_name_batch.__wrapped__ = _moved_graph_refinement.claim_refine_name_batch
_embed_single_standard_name.__wrapped__ = (
    _moved_graph_refinement._embed_single_standard_name
)
claim_embed_batch.__wrapped__ = _moved_graph_refinement.claim_embed_batch
release_embed_claims.__wrapped__ = _moved_graph_refinement.release_embed_claims
persist_embed_batch.__wrapped__ = _moved_graph_refinement.persist_embed_batch
persist_refined_name.__wrapped__ = _moved_graph_refinement.persist_refined_name
carry_accepted_documentation_across_rename.__wrapped__ = (
    _moved_graph_refinement.carry_accepted_documentation_across_rename
)


_moved_graph_source_bindings = import_module(
    ".graph_store.graph_source_bindings", __package__
)
write_enrichment_results.__wrapped__ = (
    _moved_graph_source_bindings.write_enrichment_results
)
_exact_dd_source_id.__wrapped__ = _moved_graph_source_bindings._exact_dd_source_id
_source_compose_hint_decision.__wrapped__ = (
    _moved_graph_source_bindings._source_compose_hint_decision
)
set_source_compose_hint.__wrapped__ = (
    _moved_graph_source_bindings.set_source_compose_hint
)
_pin_dd_source_snapshots.__wrapped__ = (
    _moved_graph_source_bindings._pin_dd_source_snapshots
)
merge_standard_name_sources.__wrapped__ = (
    _moved_graph_source_bindings.merge_standard_name_sources
)
claim_standard_name_source_batch.__wrapped__ = (
    _moved_graph_source_bindings.claim_standard_name_source_batch
)
claim_explicit_standard_name_sources.__wrapped__ = (
    _moved_graph_source_bindings.claim_explicit_standard_name_sources
)
_complete_source_fences.__wrapped__ = (
    _moved_graph_source_bindings._complete_source_fences
)
_require_failure_reason.__wrapped__ = (
    _moved_graph_source_bindings._require_failure_reason
)
_guard_failed_source_reasons.__wrapped__ = (
    _moved_graph_source_bindings._guard_failed_source_reasons
)
_guard_existing_target_pairings.__wrapped__ = (
    _moved_graph_source_bindings._guard_existing_target_pairings
)
_lock_claimed_name_bindings.__wrapped__ = (
    _moved_graph_source_bindings._lock_claimed_name_bindings
)
persist_claimed_source_outcomes.__wrapped__ = (
    _moved_graph_source_bindings.persist_claimed_source_outcomes
)
persist_claimed_vocab_gaps.__wrapped__ = (
    _moved_graph_source_bindings.persist_claimed_vocab_gaps
)
persist_claimed_attachments.__wrapped__ = (
    _moved_graph_source_bindings.persist_claimed_attachments
)
stage_claimed_generated_candidates.__wrapped__ = (
    _moved_graph_source_bindings.stage_claimed_generated_candidates
)
fetch_claimed_source_metadata.__wrapped__ = (
    _moved_graph_source_bindings.fetch_claimed_source_metadata
)
mark_sources_composed.__wrapped__ = _moved_graph_source_bindings.mark_sources_composed
mark_sources_attached.__wrapped__ = _moved_graph_source_bindings.mark_sources_attached
classify_parked_source.__wrapped__ = _moved_graph_source_bindings.classify_parked_source
mark_sources_failed.__wrapped__ = _moved_graph_source_bindings.mark_sources_failed
record_parked_dispositions.__wrapped__ = (
    _moved_graph_source_bindings.record_parked_dispositions
)
disposition_parked_sources.__wrapped__ = (
    _moved_graph_source_bindings.disposition_parked_sources
)
record_dispositions_for_undisposed_parks.__wrapped__ = (
    _moved_graph_source_bindings.record_dispositions_for_undisposed_parks
)
census_parked_dispositions.__wrapped__ = (
    _moved_graph_source_bindings.census_parked_dispositions
)
retry_failed_sources.__wrapped__ = _moved_graph_source_bindings.retry_failed_sources
retry_skipped_sources.__wrapped__ = _moved_graph_source_bindings.retry_skipped_sources
_skipped_recovery_error.__wrapped__ = (
    _moved_graph_source_bindings._skipped_recovery_error
)
mark_source_skipped.__wrapped__ = _moved_graph_source_bindings.mark_source_skipped
mark_sources_stale.__wrapped__ = _moved_graph_source_bindings.mark_sources_stale
write_skipped_sources.__wrapped__ = _moved_graph_source_bindings.write_skipped_sources
list_skipped_sources.__wrapped__ = _moved_graph_source_bindings.list_skipped_sources
get_skipped_source_counts.__wrapped__ = (
    _moved_graph_source_bindings.get_skipped_source_counts
)
release_standard_name_source_claims.__wrapped__ = (
    _moved_graph_source_bindings.release_standard_name_source_claims
)
_record_persist_outcome.__wrapped__ = (
    _moved_graph_source_bindings._record_persist_outcome
)
persist_outcome_snapshot.__wrapped__ = (
    _moved_graph_source_bindings.persist_outcome_snapshot
)
reset_persist_outcomes.__wrapped__ = _moved_graph_source_bindings.reset_persist_outcomes

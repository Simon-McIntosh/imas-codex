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


_graph_foundation.wire_services(_GraphServices())


logger = logging.getLogger(__name__)


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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def query_historical_cpi(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.query_historical_cpi."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.query_historical_cpi(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# =============================================================================
# Read helpers — extraction candidates
# =============================================================================


def get_extraction_candidates_dd(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.get_extraction_candidates_dd."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.get_extraction_candidates_dd(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def get_extraction_candidates_signals(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.get_extraction_candidates_signals."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.get_extraction_candidates_signals(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# =============================================================================
# Deduplication
# =============================================================================


def get_existing_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.get_existing_standard_names."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.get_existing_standard_names(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def get_named_source_ids(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.get_named_source_ids."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.get_named_source_ids(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def partition_focus_by_accepted(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.partition_focus_by_accepted."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.partition_focus_by_accepted(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


MANIFEST_DRAIN_DISPOSITIONS = import_module(
    ".graph_store.graph_claim_scope", __package__
).MANIFEST_DRAIN_DISPOSITIONS


def canonicalize_manifest_drain_paths(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.canonicalize_manifest_drain_paths."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.canonicalize_manifest_drain_paths(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def classify_manifest_drain_item(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.classify_manifest_drain_item."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.classify_manifest_drain_item(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


_MANIFEST_DRAIN_PLAN_QUERY = import_module(
    ".graph_store.graph_claim_scope", __package__
)._MANIFEST_DRAIN_PLAN_QUERY


def _resolved_manifest_drain_plan(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._resolved_manifest_drain_plan."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._resolved_manifest_drain_plan(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def build_manifest_drain_plan(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.build_manifest_drain_plan."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.build_manifest_drain_plan(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


ManifestDrainConflict = import_module(
    ".graph_store.graph_claim_scope", __package__
).ManifestDrainConflict


def _manifest_plan_fingerprint(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._manifest_plan_fingerprint."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._manifest_plan_fingerprint(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _lock_manifest_drain_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._lock_manifest_drain_authority."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._lock_manifest_drain_authority(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def claim_manifest_drain_scope(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.claim_manifest_drain_scope."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.claim_manifest_drain_scope(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def prepare_manifest_drain_scope(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.prepare_manifest_drain_scope."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.prepare_manifest_drain_scope(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def clear_manifest_drain_scope(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.clear_manifest_drain_scope."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.clear_manifest_drain_scope(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_manifest_drain_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.release_manifest_drain_claims."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.release_manifest_drain_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def finalize_manifest_drain_scope(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.finalize_manifest_drain_scope."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.finalize_manifest_drain_scope(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def scope_focus_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.scope_focus_names."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.scope_focus_names(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _claimed_run_ids(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._claimed_run_ids."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._claimed_run_ids(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _collect_run_statuses(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._collect_run_statuses."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._collect_run_statuses(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _exact_name_scope_refusals(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope._exact_name_scope_refusals."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope._exact_name_scope_refusals(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def scope_exact_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_scope.scope_exact_standard_names."""
    from .graph_store import graph_claim_scope

    return graph_claim_scope.scope_exact_standard_names(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def get_source_name_mapping(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.get_source_name_mapping."""
    from .graph_store import graph_lineage

    return graph_lineage.get_source_name_mapping(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _get_rich_source_name_mapping(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._get_rich_source_name_mapping."""
    from .graph_store import graph_lineage

    return graph_lineage._get_rich_source_name_mapping(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# =============================================================================
# Write helpers
# =============================================================================


def fetch_low_score_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_low_score_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_low_score_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def fetch_review_feedback_for_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_review_feedback_for_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_review_feedback_for_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def fetch_reviewer_history_for_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_reviewer_history_for_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_reviewer_history_for_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def fetch_docs_review_feedback_for_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.fetch_docs_review_feedback_for_standard_names."""
    from .graph_store import graph_lineage

    return graph_lineage.fetch_docs_review_feedback_for_standard_names(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _filter_admissible_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._filter_admissible_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._filter_admissible_parents(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _emit_magnitude_of_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._emit_magnitude_of_edges."""
    from .graph_store import graph_lineage

    return graph_lineage._emit_magnitude_of_edges(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _bootstrap_missing_derived_parent_targets(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._bootstrap_missing_derived_parent_targets."""
    from .graph_store import graph_lineage

    return graph_lineage._bootstrap_missing_derived_parent_targets(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _write_standard_name_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._write_standard_name_edges."""
    from .graph_store import graph_lineage

    return graph_lineage._write_standard_name_edges(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def rederive_structural_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.rederive_structural_edges."""
    from .graph_store import graph_lineage

    return graph_lineage.rederive_structural_edges(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _rewire_has_parent_off_superseded(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._rewire_has_parent_off_superseded."""
    from .graph_store import graph_lineage

    return graph_lineage._rewire_has_parent_off_superseded(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _parse_parent_grammar(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._parse_parent_grammar."""
    from .graph_store import graph_lineage

    return graph_lineage._parse_parent_grammar(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _query_seedable_derived_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_seedable_derived_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._query_seedable_derived_parents(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _query_legacy_repairable_derived_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_legacy_repairable_derived_parents."""
    from .graph_store import graph_lineage

    return graph_lineage._query_legacy_repairable_derived_parents(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _query_derived_parents_for_admission_cleanup(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._query_derived_parents_for_admission_cleanup."""
    from .graph_store import graph_lineage

    return graph_lineage._query_derived_parents_for_admission_cleanup(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _delete_derived_parent_nodes(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._delete_derived_parent_nodes."""
    from .graph_store import graph_lineage

    return graph_lineage._delete_derived_parent_nodes(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def sweep_orphaned_docs_revisions(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.sweep_orphaned_docs_revisions."""
    from .graph_store import graph_lineage

    return graph_lineage.sweep_orphaned_docs_revisions(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _derived_parent_source_metadata(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._derived_parent_source_metadata."""
    from .graph_store import graph_lineage

    return graph_lineage._derived_parent_source_metadata(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _validate_derived_parent_identity(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._validate_derived_parent_identity."""
    from .graph_store import graph_lineage

    return graph_lineage._validate_derived_parent_identity(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _eligible_derived_parent_unit_children(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._eligible_derived_parent_unit_children."""
    from .graph_store import graph_lineage

    return graph_lineage._eligible_derived_parent_unit_children(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _materialize_derived_parent_rows(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._materialize_derived_parent_rows."""
    from .graph_store import graph_lineage

    return graph_lineage._materialize_derived_parent_rows(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


_ParentMaterializationCapture = import_module(
    ".graph_store.graph_lineage", __package__
)._ParentMaterializationCapture


def _materialize_derived_parent_rows_batched(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._materialize_derived_parent_rows_batched."""
    from .graph_store import graph_lineage

    return graph_lineage._materialize_derived_parent_rows_batched(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def seed_parent_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.seed_parent_sources."""
    from .graph_store import graph_lineage

    return graph_lineage.seed_parent_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def normalize_derived_parent_lifecycle(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage.normalize_derived_parent_lifecycle."""
    from .graph_store import graph_lineage

    return graph_lineage.normalize_derived_parent_lifecycle(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _derived_parent_unit_from_eligible_children(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._derived_parent_unit_from_eligible_children."""
    from .graph_store import graph_lineage

    return graph_lineage._derived_parent_unit_from_eligible_children(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _spurious_stub_binding_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._spurious_stub_binding_authority."""
    from .graph_store import graph_lineage

    return graph_lineage._spurious_stub_binding_authority(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _partition_lifecycleless_stub_rows(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._partition_lifecycleless_stub_rows."""
    from .graph_store import graph_lineage

    return graph_lineage._partition_lifecycleless_stub_rows(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _lifecycleless_stub_manifest(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._lifecycleless_stub_manifest."""
    from .graph_store import graph_lineage

    return graph_lineage._lifecycleless_stub_manifest(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _reconcile_spurious_stub_source_scalars(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_lineage._reconcile_spurious_stub_source_scalars."""
    from .graph_store import graph_lineage

    return graph_lineage._reconcile_spurious_stub_source_scalars(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# Forwarders for the catalog write domain; the implementation lives in
# graph_store/graph_catalog.py and the facade supplies the service object.


def write_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.write_standard_names."""
    from .graph_store import graph_catalog

    return graph_catalog.write_standard_names(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def write_reviews(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.write_reviews."""
    from .graph_store import graph_catalog

    return graph_catalog.write_reviews(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def update_review_aggregates(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.update_review_aggregates."""
    from .graph_store import graph_catalog

    return graph_catalog.update_review_aggregates(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _review_tier(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._review_tier."""
    from .graph_store import graph_catalog

    return graph_catalog._review_tier(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def plan_review_dimension_void(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.plan_review_dimension_void."""
    from .graph_store import graph_catalog

    return graph_catalog.plan_review_dimension_void(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def void_review_dimension(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.void_review_dimension."""
    from .graph_store import graph_catalog

    return graph_catalog.void_review_dimension(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _winning_review_resolution_methods(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._winning_review_resolution_methods."""
    from .graph_store import graph_catalog

    return graph_catalog._winning_review_resolution_methods(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# The owning function is cached, so the forwarder carries its cache API and a
# caller that clears the cache reaches the entries the forwarder reads.
_winning_review_resolution_methods.cache_clear = import_module(
    ".graph_store.graph_catalog", __package__
)._winning_review_resolution_methods.cache_clear
_winning_review_resolution_methods.cache_info = import_module(
    ".graph_store.graph_catalog", __package__
)._winning_review_resolution_methods.cache_info


def _non_winning_review_resolution_methods(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._non_winning_review_resolution_methods."""
    from .graph_store import graph_catalog

    return graph_catalog._non_winning_review_resolution_methods(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def docs_review_eligibility_params(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.docs_review_eligibility_params."""
    from .graph_store import graph_catalog

    return graph_catalog.docs_review_eligibility_params(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _docs_review_winner_query_body(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._docs_review_winner_query_body."""
    from .graph_store import graph_catalog

    return graph_catalog._docs_review_winner_query_body(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def docs_review_eligibility_where(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.docs_review_eligibility_where."""
    from .graph_store import graph_catalog

    return graph_catalog.docs_review_eligibility_where(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def property_coverage(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.property_coverage."""
    from .graph_store import graph_catalog

    return graph_catalog.property_coverage(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def docs_review_property_coverage(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.docs_review_property_coverage."""
    from .graph_store import graph_catalog

    return graph_catalog.docs_review_property_coverage(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def write_name_review_results(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.write_name_review_results."""
    from .graph_store import graph_catalog

    return graph_catalog.write_name_review_results(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def write_docs_review_results(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.write_docs_review_results."""
    from .graph_store import graph_catalog

    return graph_catalog.write_docs_review_results(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _resolve_grammar_token_version(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._resolve_grammar_token_version."""
    from .graph_store import graph_catalog

    return graph_catalog._resolve_grammar_token_version(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _coerce_segment_value(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._coerce_segment_value."""
    from .graph_store import graph_catalog

    return graph_catalog._coerce_segment_value(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _resolve_synced_segments(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._resolve_synced_segments."""
    from .graph_store import graph_catalog

    return graph_catalog._resolve_synced_segments(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _write_grammar_decomposition(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._write_grammar_decomposition."""
    from .graph_store import graph_catalog

    return graph_catalog._write_grammar_decomposition(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _write_segment_edges(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._write_segment_edges."""
    from .graph_store import graph_catalog

    return graph_catalog._write_segment_edges(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_generated_name_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.persist_generated_name_winners."""
    from .graph_store import graph_catalog

    return graph_catalog.persist_generated_name_winners(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_generated_name_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog.persist_generated_name_batch."""
    from .graph_store import graph_catalog

    return graph_catalog.persist_generated_name_batch(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _backfill_cluster_from_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._backfill_cluster_from_sources."""
    from .graph_store import graph_catalog

    return graph_catalog._backfill_cluster_from_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _finalize_generated_name_stage(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog._finalize_generated_name_stage."""
    from .graph_store import graph_catalog

    return graph_catalog._finalize_generated_name_stage(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


ReviewWriteResult = import_module(
    ".graph_store.graph_catalog", __package__
).ReviewWriteResult


_DIMENSION_SCALE = import_module(
    ".graph_store.graph_catalog", __package__
)._DIMENSION_SCALE


ReviewDimensionVoidRefused = import_module(
    ".graph_store.graph_catalog", __package__
).ReviewDimensionVoidRefused


_VOID_REVIEW_DIMENSION_READ = import_module(
    ".graph_store.graph_catalog", __package__
)._VOID_REVIEW_DIMENSION_READ


_VOID_REVIEW_DIMENSION_WRITE = import_module(
    ".graph_store.graph_catalog", __package__
)._VOID_REVIEW_DIMENSION_WRITE


_GRAMMAR_SEGMENT_COLUMNS = import_module(
    ".graph_store.graph_catalog", __package__
)._GRAMMAR_SEGMENT_COLUMNS


_SEGMENT_EDGE_TYPES = import_module(
    ".graph_store.graph_catalog", __package__
)._SEGMENT_EDGE_TYPES


_GRAMMAR_FIELDS = import_module(
    ".graph_store.graph_catalog", __package__
)._GRAMMAR_FIELDS


SAFE_SCALAR_COCOS_UNITS = import_module(
    ".graph_store.graph_catalog", __package__
).SAFE_SCALAR_COCOS_UNITS


_RETIRE_FINALIZED_SOURCE_GAPS = import_module(
    ".graph_store.graph_catalog", __package__
)._RETIRE_FINALIZED_SOURCE_GAPS


def write_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.write_vocab_gaps."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.write_vocab_gaps(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def fetch_vocab_gap_adjudications(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.fetch_vocab_gap_adjudications."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.fetch_vocab_gap_adjudications(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def mark_names_validated(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_validated."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_validated(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_validation_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.release_validation_claims."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.release_validation_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def claim_names_for_embedding(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.claim_names_for_embedding."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.claim_names_for_embedding(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def mark_names_embedded(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_embedded."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_embedded(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_embedding_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.release_embedding_claims."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.release_embedding_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def get_validated_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.get_validated_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.get_validated_names(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def mark_names_consolidated(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.mark_names_consolidated."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.mark_names_consolidated(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def reset_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.reset_standard_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.reset_standard_names(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def clear_standard_names(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.clear_standard_names."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.clear_standard_names(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def clear_sn_subsystem(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_catalog_reset.clear_sn_subsystem."""
    from .graph_store import graph_catalog_reset

    return graph_catalog_reset.clear_sn_subsystem(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def resolve_links_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links.resolve_links_batch."""
    from .graph_store import graph_doc_links

    return graph_doc_links.resolve_links_batch(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _normalize_bare_doc_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links._normalize_bare_doc_links."""
    from .graph_store import graph_doc_links

    return graph_doc_links._normalize_bare_doc_links(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def resolve_doc_links(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_doc_links.resolve_doc_links."""
    from .graph_store import graph_doc_links

    return graph_doc_links.resolve_doc_links(
        *args, services=_graph_foundation.get_services(), **kwargs
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
_CLAIM_VERIFY_SETTLE_SECONDS = _graph_foundation._CLAIM_VERIFY_SETTLE_SECONDS
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _exact_dd_source_id(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._exact_dd_source_id."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._exact_dd_source_id(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _source_compose_hint_decision(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._source_compose_hint_decision."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._source_compose_hint_decision(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def set_source_compose_hint(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.set_source_compose_hint."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.set_source_compose_hint(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _pin_dd_source_snapshots(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._pin_dd_source_snapshots."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._pin_dd_source_snapshots(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def merge_standard_name_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.merge_standard_name_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.merge_standard_name_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def claim_standard_name_source_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.claim_standard_name_source_batch."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.claim_standard_name_source_batch(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def claim_explicit_standard_name_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.claim_explicit_standard_name_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.claim_explicit_standard_name_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _complete_source_fences(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._complete_source_fences."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._complete_source_fences(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _require_failure_reason(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._require_failure_reason."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._require_failure_reason(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _guard_failed_source_reasons(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._guard_failed_source_reasons."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._guard_failed_source_reasons(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _guard_existing_target_pairings(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._guard_existing_target_pairings."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._guard_existing_target_pairings(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _lock_claimed_name_bindings(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._lock_claimed_name_bindings."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._lock_claimed_name_bindings(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_claimed_source_outcomes(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.persist_claimed_source_outcomes."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.persist_claimed_source_outcomes(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_claimed_vocab_gaps(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.persist_claimed_vocab_gaps."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.persist_claimed_vocab_gaps(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_claimed_attachments(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.persist_claimed_attachments."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.persist_claimed_attachments(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def stage_claimed_generated_candidates(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.stage_claimed_generated_candidates."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.stage_claimed_generated_candidates(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def fetch_claimed_source_metadata(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.fetch_claimed_source_metadata."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.fetch_claimed_source_metadata(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def mark_sources_composed(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_sources_composed."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_sources_composed(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def mark_sources_attached(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_sources_attached."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_sources_attached(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def classify_parked_source(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.classify_parked_source."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.classify_parked_source(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def mark_sources_failed(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_sources_failed."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_sources_failed(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def record_parked_dispositions(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.record_parked_dispositions."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.record_parked_dispositions(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def disposition_parked_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.disposition_parked_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.disposition_parked_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def record_dispositions_for_undisposed_parks(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.record_dispositions_for_undisposed_parks."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.record_dispositions_for_undisposed_parks(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def census_parked_dispositions(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.census_parked_dispositions."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.census_parked_dispositions(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def retry_failed_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.retry_failed_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.retry_failed_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def retry_skipped_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.retry_skipped_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.retry_skipped_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _skipped_recovery_error(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._skipped_recovery_error."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._skipped_recovery_error(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def mark_source_skipped(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_source_skipped."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_source_skipped(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def mark_sources_stale(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.mark_sources_stale."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.mark_sources_stale(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def write_skipped_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.write_skipped_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.write_skipped_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def list_skipped_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.list_skipped_sources."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.list_skipped_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def get_skipped_source_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.get_skipped_source_counts."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.get_skipped_source_counts(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_standard_name_source_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.release_standard_name_source_claims."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.release_standard_name_source_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _record_persist_outcome(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings._record_persist_outcome."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings._record_persist_outcome(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_outcome_snapshot(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.persist_outcome_snapshot."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.persist_outcome_snapshot(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def reset_persist_outcomes(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_source_bindings.reset_persist_outcomes."""
    from .graph_store import graph_source_bindings

    return graph_source_bindings.reset_persist_outcomes(
        *args, services=_graph_foundation.get_services(), **kwargs
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


# The owning function is cached, so the forwarder carries its cache API and a
# caller that clears the cache reaches the entries the forwarder reads.
isn_vocabulary_signature.cache_clear = import_module(
    ".graph_store.graph_audits", __package__
).isn_vocabulary_signature.cache_clear
isn_vocabulary_signature.cache_info = import_module(
    ".graph_store.graph_audits", __package__
).isn_vocabulary_signature.cache_info


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
_PHASE_TO_POOL = _graph_foundation._PHASE_TO_POOL


def _normalize_pool(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims._normalize_pool."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims._normalize_pool(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def record_llm_cost(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.record_llm_cost."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.record_llm_cost(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def aggregate_spend_for_run(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_for_run."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_for_run(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def aggregate_spend_per_phase(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_per_phase."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_per_phase(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def aggregate_calls_per_phase(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_calls_per_phase."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_calls_per_phase(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def aggregate_spend_per_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.aggregate_spend_per_name."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.aggregate_spend_per_name(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def update_sn_per_phase_costs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.update_sn_per_phase_costs."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.update_sn_per_phase_costs(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# -- generate_name (StandardNameSource) -------------------------------------


def claim_generate_name_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_generate_name_batch."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_generate_name_batch(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# -- enrich (StandardName, enriched_at IS NULL) -------------------------------


def claim_enrich_seed_and_expand(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_enrich_seed_and_expand."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_enrich_seed_and_expand(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# -- review names (StandardName, reviewed_name_at IS NULL) --------------------


def claim_review_names_seed_and_expand(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_cost_and_claims.claim_review_names_seed_and_expand."""
    from .graph_store import graph_cost_and_claims

    return graph_cost_and_claims.claim_review_names_seed_and_expand(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _charge_refine_name_attempts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._charge_refine_name_attempts."""
    from .graph_store import graph_refinement

    return graph_refinement._charge_refine_name_attempts(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def claim_refine_name_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.claim_refine_name_batch."""
    from .graph_store import graph_refinement

    return graph_refinement.claim_refine_name_batch(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _embed_single_standard_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement._embed_single_standard_name."""
    from .graph_store import graph_refinement

    return graph_refinement._embed_single_standard_name(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def claim_embed_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.claim_embed_batch."""
    from .graph_store import graph_refinement

    return graph_refinement.claim_embed_batch(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_embed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.release_embed_claims."""
    from .graph_store import graph_refinement

    return graph_refinement.release_embed_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_embed_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.persist_embed_batch."""
    from .graph_store import graph_refinement

    return graph_refinement.persist_embed_batch(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_refined_name(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.persist_refined_name."""
    from .graph_store import graph_refinement

    return graph_refinement.persist_refined_name(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def carry_accepted_documentation_across_rename(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_refinement.carry_accepted_documentation_across_rename."""
    from .graph_store import graph_refinement

    return graph_refinement.carry_accepted_documentation_across_rename(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_generate_name_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_generate_name_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_generate_name_failed_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_enrich_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_enrich_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_enrich_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_enrich_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_enrich_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_enrich_failed_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_review_names_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_names_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_names_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_review_names_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_names_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_names_failed_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_review_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_docs_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_docs_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_review_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_review_docs_failed_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_review_docs_failed_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _project_refine_claim_release_state(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._project_refine_claim_release_state."""
    from .graph_store import graph_claim_release

    return graph_claim_release._project_refine_claim_release_state(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _predicted_refine_claim_release_state(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._predicted_refine_claim_release_state."""
    from .graph_store import graph_claim_release

    return graph_claim_release._predicted_refine_claim_release_state(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _refine_claim_release_manifest(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._refine_claim_release_manifest."""
    from .graph_store import graph_claim_release

    return graph_claim_release._refine_claim_release_manifest(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _refine_claim_no_change_receipt(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._refine_claim_no_change_receipt."""
    from .graph_store import graph_claim_release

    return graph_claim_release._refine_claim_no_change_receipt(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_stranded_refine_claim(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_stranded_refine_claim."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_stranded_refine_claim(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_refine_name_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.release_refine_name_claims."""
    from .graph_store import graph_claim_release

    return graph_claim_release.release_refine_name_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def resubmit_pinned_rename_for_review(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.resubmit_pinned_rename_for_review."""
    from .graph_store import graph_claim_release

    return graph_claim_release.resubmit_pinned_rename_for_review(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


RESCORE_ELIGIBILITY_QUERY = import_module(
    ".graph_store.graph_claim_release", __package__
).RESCORE_ELIGIBILITY_QUERY


def stage_name_for_rescore(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.stage_name_for_rescore."""
    from .graph_store import graph_claim_release

    return graph_claim_release.stage_name_for_rescore(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def restore_name_after_failed_rescore(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release.restore_name_after_failed_rescore."""
    from .graph_store import graph_claim_release

    return graph_claim_release.restore_name_after_failed_rescore(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _mark_refine_docs_exhausted(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._mark_refine_docs_exhausted."""
    from .graph_store import graph_claim_release

    return graph_claim_release._mark_refine_docs_exhausted(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _verify_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_claim_winners(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _verify_source_claim_winners(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_claim_release._verify_source_claim_winners."""
    from .graph_store import graph_claim_release

    return graph_claim_release._verify_source_claim_winners(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_generated_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.persist_generated_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.persist_generated_docs(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_generate_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_generate_docs_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_generate_docs_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_generate_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_generate_docs_failed_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_generate_docs_failed_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# =============================================================================
# Persist — refine_docs (snapshot current docs → DocsRevision, update SN)
# =============================================================================


def persist_refined_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.persist_refined_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.persist_refined_docs(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def reset_standard_name_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.reset_standard_name_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.reset_standard_name_docs(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def stamp_harmonized_families(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.stamp_harmonized_families."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.stamp_harmonized_families(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_refine_docs_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_refine_docs_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_refine_docs_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_refine_docs_failed_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.release_refine_docs_failed_claims."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.release_refine_docs_failed_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# =============================================================================
# Routing helper: desc-name similarity gate → REFINE_DOCS
# =============================================================================


def mark_for_refine_docs(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_docs_persistence.mark_for_refine_docs."""
    from .graph_store import graph_docs_persistence

    return graph_docs_persistence.mark_for_refine_docs(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_all_orphan_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.release_all_orphan_claims."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.release_all_orphan_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


# ═══════════════════════════════════════════════════════════════════════
# Pool pending counts — single round-trip query for all 6 pools
# ═══════════════════════════════════════════════════════════════════════


def pool_pending_counts(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.pool_pending_counts."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.pool_pending_counts(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def scoped_terminal_residue(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.scoped_terminal_residue."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.scoped_terminal_residue(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def claim_enrich_parents_batch(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.claim_enrich_parents_batch."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.claim_enrich_parents_batch(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def fetch_derived_parent_children(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.fetch_derived_parent_children."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.fetch_derived_parent_children(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _structural_authority_snapshot(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._structural_authority_snapshot."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._structural_authority_snapshot(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _structural_authority_record(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._structural_authority_record."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._structural_authority_record(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def _persist_structural_authority(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle._persist_structural_authority."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle._persist_structural_authority(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def persist_enriched_parent(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.persist_enriched_parent."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.persist_enriched_parent(
        *args, services=_graph_foundation.get_services(), **kwargs
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
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def structural_accept_derived_parents(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.structural_accept_derived_parents."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.structural_accept_derived_parents(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def backfill_structural_name_authorities(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.backfill_structural_name_authorities."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.backfill_structural_name_authorities(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def find_orphan_parent_source_candidates(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.find_orphan_parent_source_candidates."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.find_orphan_parent_source_candidates(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def classify_orphan_parent_source_candidates(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.classify_orphan_parent_source_candidates."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.classify_orphan_parent_source_candidates(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def reconcile_orphan_parent_sources(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.reconcile_orphan_parent_sources."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.reconcile_orphan_parent_sources(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def reconcile_orphan_parent_sources_batched(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.reconcile_orphan_parent_sources_batched."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.reconcile_orphan_parent_sources_batched(
        *args, services=_graph_foundation.get_services(), **kwargs
    )


def release_enrich_parents_claims(*args: Any, **kwargs: Any):
    """Forward to graph_store.graph_parent_lifecycle.release_enrich_parents_claims."""
    from .graph_store import graph_parent_lifecycle

    return graph_parent_lifecycle.release_enrich_parents_claims(
        *args, services=_graph_foundation.get_services(), **kwargs
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


_moved_graph_catalog = import_module(".graph_store.graph_catalog", __package__)
write_standard_names.__wrapped__ = _moved_graph_catalog.write_standard_names
write_reviews.__wrapped__ = _moved_graph_catalog.write_reviews
update_review_aggregates.__wrapped__ = _moved_graph_catalog.update_review_aggregates
_review_tier.__wrapped__ = _moved_graph_catalog._review_tier
plan_review_dimension_void.__wrapped__ = _moved_graph_catalog.plan_review_dimension_void
void_review_dimension.__wrapped__ = _moved_graph_catalog.void_review_dimension
_winning_review_resolution_methods.__wrapped__ = (
    _moved_graph_catalog._winning_review_resolution_methods
)
_non_winning_review_resolution_methods.__wrapped__ = (
    _moved_graph_catalog._non_winning_review_resolution_methods
)
docs_review_eligibility_params.__wrapped__ = (
    _moved_graph_catalog.docs_review_eligibility_params
)
_docs_review_winner_query_body.__wrapped__ = (
    _moved_graph_catalog._docs_review_winner_query_body
)
docs_review_eligibility_where.__wrapped__ = (
    _moved_graph_catalog.docs_review_eligibility_where
)
property_coverage.__wrapped__ = _moved_graph_catalog.property_coverage
docs_review_property_coverage.__wrapped__ = (
    _moved_graph_catalog.docs_review_property_coverage
)
write_name_review_results.__wrapped__ = _moved_graph_catalog.write_name_review_results
write_docs_review_results.__wrapped__ = _moved_graph_catalog.write_docs_review_results
_resolve_grammar_token_version.__wrapped__ = (
    _moved_graph_catalog._resolve_grammar_token_version
)
_coerce_segment_value.__wrapped__ = _moved_graph_catalog._coerce_segment_value
_resolve_synced_segments.__wrapped__ = _moved_graph_catalog._resolve_synced_segments
_write_grammar_decomposition.__wrapped__ = (
    _moved_graph_catalog._write_grammar_decomposition
)
_write_segment_edges.__wrapped__ = _moved_graph_catalog._write_segment_edges
persist_generated_name_winners.__wrapped__ = (
    _moved_graph_catalog.persist_generated_name_winners
)
persist_generated_name_batch.__wrapped__ = (
    _moved_graph_catalog.persist_generated_name_batch
)
_backfill_cluster_from_sources.__wrapped__ = (
    _moved_graph_catalog._backfill_cluster_from_sources
)
_finalize_generated_name_stage.__wrapped__ = (
    _moved_graph_catalog._finalize_generated_name_stage
)

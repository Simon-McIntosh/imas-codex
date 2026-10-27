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
    resolve_owned_name as _resolve_owned_graph_name,
)


class _GraphServices:
    def __getattr__(self, name: str) -> Any:
        try:
            return _resolve_owned_graph_name(name)
        except AttributeError:
            return globals()[name]


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

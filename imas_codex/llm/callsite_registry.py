"""Executable inventory of structured LLM dispatch routes.

The registry freezes semantic dispatch expressions rather than source line
numbers.  Each expression is identified by its carrier path, lexical scope,
dispatch mechanism, and occurrence within that scope.  The scanner uses the
running interpreter's grammar and treats a syntax error as an inventory
failure; a source file can never disappear from the inventory because it could
not be parsed.
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DispatchStyle = Literal["direct", "to-thread", "injected", "typed-sync", "typed-async"]
TransitionKind = Literal["legacy", "typed"]
Reachability = Literal["active", "active-public"]


@dataclass(frozen=True, slots=True)
class RouteBinding:
    """A service/seat/template route reachable through one source expression."""

    route_id: str
    service: str
    seat: str
    model_source: str
    templates: tuple[str, ...]
    asset_mode: Literal["legacy-template", "legacy-inline"]
    response_model_identity: str | None = None


@dataclass(frozen=True, slots=True, order=True)
class SourceCallIdentity:
    """Line-independent identity for a structured dispatch expression."""

    source_path: str
    scope: str
    dispatch_symbol: str
    occurrence: int = 1


@dataclass(frozen=True, slots=True)
class CallsiteRegistration:
    """Declared semantic identity and routes for one dispatch expression."""

    callsite_id: str
    source: SourceCallIdentity
    dispatch_style: DispatchStyle
    service_argument: str
    response_model_symbol: str
    reachability: Reachability
    routes: tuple[RouteBinding, ...]


@dataclass(frozen=True, slots=True)
class StructuredCall:
    """One structured dispatch expression found by the source scanner."""

    source: SourceCallIdentity
    dispatch_style: DispatchStyle
    service_argument: str | None
    model_argument: str | None
    response_model_argument: str | None
    transition_kind: TransitionKind
    callsite_id: str | None
    route_id: str | None
    line: int


@dataclass(frozen=True, slots=True)
class ProviderCall:
    """A raw provider call found outside the structured dispatcher."""

    source_path: str
    scope: str
    symbol: str
    line: int


class CallsiteSourceSyntaxError(RuntimeError):
    """Raised when a source file cannot be parsed with the project grammar."""


class CallsiteInventoryError(AssertionError):
    """Raised when source expressions drift from the executable registry."""


class CallsitePolicyError(ValueError):
    """A context-dispatch policy does not match an executable registry route."""


_MODEL_SOURCES = {
    "sn-benchmark.candidate": "sn-benchmark:candidates",
    "sn-benchmark.docs-candidate": "sn-benchmark:candidates",
    "sn-benchmark.compose-candidate": "sn-benchmark:compose",
    "sn-benchmark.reviewer-candidate": "sn-benchmark:reviewers",
    "sn-benchmark.refine-candidate": "sn-benchmark:refine",
    "sn-benchmark.judge-candidate": "sn-benchmark:judges",
    "sn-benchmark.reviewer": "sn-benchmark:judges",
    "sn-fanout.proposer": "sn-fanout:proposer",
    "sn-review.names": "sn-review:names",
    "sn-review.docs": "sn-review:docs",
}


def _route(service: str, seat: str, *templates: str) -> RouteBinding:
    asset_mode = (
        "legacy-inline"
        if any(template.startswith("inline:") for template in templates)
        else "legacy-template"
    )
    model_source = _MODEL_SOURCES.get(seat, f"section:{seat}")
    return RouteBinding(seat, service, seat, model_source, templates, asset_mode)


def _source(
    path: str,
    scope: str,
    dispatch_symbol: str,
    occurrence: int = 1,
) -> SourceCallIdentity:
    return SourceCallIdentity(path, scope, dispatch_symbol, occurrence)


CALLSITE_REGISTRY: tuple[CallsiteRegistration, ...] = (
    CallsiteRegistration(
        "dd.cluster-labeling",
        _source(
            "imas_codex/clusters/labeler.py",
            "ClusterLabeler.label_clusters",
            "call_llm_structured",
        ),
        "direct",
        "'data-dictionary'",
        "ClusterLabelBatch",
        "active",
        (_route("data-dictionary", "dd-enrichment", "clusters/labeler"),),
    ),
    CallsiteRegistration(
        "discovery.image-scoring",
        _source(
            "imas_codex/discovery/base/image.py",
            "score_images_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'facility-discovery'",
        "ImageScoreBatch",
        "active-public",
        (_route("facility-discovery", "vision", "wiki/image-captioner"),),
    ),
    CallsiteRegistration(
        "discovery.code-triage",
        _source(
            "imas_codex/discovery/code/workers.py",
            "triage_worker",
            "call_llm_structured",
        ),
        "to-thread",
        "'facility-discovery'",
        "FileTriageBatch",
        "active",
        (_route("facility-discovery", "language", "code/triage"),),
    ),
    CallsiteRegistration(
        "discovery.code-scoring",
        _source(
            "imas_codex/discovery/code/workers.py",
            "score_worker",
            "call_llm_structured",
        ),
        "to-thread",
        "'facility-discovery'",
        "FileScoreBatch",
        "active",
        (_route("facility-discovery", "language", "code/scorer"),),
    ),
    CallsiteRegistration(
        "discovery.path-scoring",
        _source(
            "imas_codex/discovery/paths/parallel.py",
            "_async_score_with_llm",
            "acall_llm_structured",
        ),
        "direct",
        "'facility-discovery'",
        "ScoreBatch",
        "active",
        (_route("facility-discovery", "language", "paths/scorer"),),
    ),
    CallsiteRegistration(
        "discovery.path-triage-sync",
        _source(
            "imas_codex/discovery/paths/scorer.py",
            "DirectoryTriager.triage_batch",
            "call_llm_structured",
        ),
        "direct",
        "'facility-discovery'",
        "TriageBatch",
        "active-public",
        (_route("facility-discovery", "language", "paths/triage"),),
    ),
    CallsiteRegistration(
        "discovery.path-triage-async",
        _source(
            "imas_codex/discovery/paths/scorer.py",
            "DirectoryTriager.async_triage_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'facility-discovery'",
        "TriageBatch",
        "active",
        (_route("facility-discovery", "language", "paths/triage"),),
    ),
    CallsiteRegistration(
        "discovery.signal-enrichment",
        _source(
            "imas_codex/discovery/signals/parallel.py",
            "enrich_worker",
            "acall_llm_structured",
        ),
        "direct",
        "'facility-discovery'",
        "SignalEnrichmentBatch",
        "active",
        (_route("facility-discovery", "language", "signals/enrichment"),),
    ),
    CallsiteRegistration(
        "discovery.signal-source-unwind",
        _source(
            "imas_codex/discovery/signals/parallel.py",
            "individualize_source_descriptions",
            "acall_llm_structured",
        ),
        "direct",
        "'facility-discovery'",
        "SignalSourceCodeUnwindBatch",
        "active",
        (_route("facility-discovery", "language", "signals/source_unwind"),),
    ),
    CallsiteRegistration(
        "discovery.static-pattern-enrichment",
        _source(
            "imas_codex/discovery/static/workers.py",
            "enrich_worker",
            "call_llm_structured",
            1,
        ),
        "to-thread",
        "'facility-discovery'",
        "StaticNodeBatch",
        "active",
        (_route("facility-discovery", "language", "discovery/static-enricher"),),
    ),
    CallsiteRegistration(
        "discovery.static-parent-enrichment",
        _source(
            "imas_codex/discovery/static/workers.py",
            "enrich_worker",
            "call_llm_structured",
            2,
        ),
        "to-thread",
        "'facility-discovery'",
        "StaticNodeBatch",
        "active",
        (_route("facility-discovery", "language", "discovery/static-enricher"),),
    ),
    CallsiteRegistration(
        "discovery.static-orphan-enrichment",
        _source(
            "imas_codex/discovery/static/workers.py",
            "enrich_worker",
            "call_llm_structured",
            3,
        ),
        "to-thread",
        "'facility-discovery'",
        "StaticNodeBatch",
        "active",
        (_route("facility-discovery", "language", "discovery/static-enricher"),),
    ),
    CallsiteRegistration(
        "discovery.document-scoring",
        _source(
            "imas_codex/discovery/wiki/scoring.py",
            "_score_documents_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'facility-discovery'",
        "DocumentScoreBatch",
        "active",
        (_route("facility-discovery", "language", "wiki/document-scorer"),),
    ),
    CallsiteRegistration(
        "discovery.wiki-page-scoring",
        _source(
            "imas_codex/discovery/wiki/scoring.py",
            "_score_pages_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'facility-discovery'",
        "WikiScoreBatch",
        "active",
        (_route("facility-discovery", "language", "wiki/scorer"),),
    ),
    CallsiteRegistration(
        "dd.domain-classification",
        _source(
            "imas_codex/graph/dd_domain_classifier.py",
            "_classify_batch",
            "acall_llm_structured",
        ),
        "direct",
        "service",
        "DomainBatchResult",
        "active",
        (
            _route("data-dictionary", "sn-classifier", "imas/domain_classifier"),
            _route(
                "standard-names", "sn-benchmark.candidate", "imas/domain_classifier"
            ),
        ),
    ),
    CallsiteRegistration(
        "dd.path-enrichment",
        _source(
            "imas_codex/graph/dd_enrichment.py",
            "enrich_imas_paths",
            "call_llm_structured",
        ),
        "direct",
        "'data-dictionary'",
        "IMASPathEnrichmentBatch",
        "active",
        (_route("data-dictionary", "dd-enrichment", "imas/enrichment"),),
    ),
    CallsiteRegistration(
        "dd.identifier-schema-enrichment",
        _source(
            "imas_codex/graph/dd_identifier_enrichment.py",
            "enrich_identifier_schemas",
            "call_llm_structured",
        ),
        "direct",
        "'data-dictionary'",
        "IdentifierEnrichmentBatch",
        "active",
        (_route("data-dictionary", "dd-enrichment", "imas/identifier_enrichment"),),
    ),
    CallsiteRegistration(
        "dd.identifier-node-enrichment",
        _source(
            "imas_codex/graph/dd_identifier_enrichment.py",
            "enrich_identifier_nodes",
            "call_llm_structured",
        ),
        "direct",
        "'data-dictionary'",
        "IdentifierNodeEnrichmentBatch",
        "active",
        (
            _route(
                "data-dictionary", "dd-enrichment", "imas/identifier_node_enrichment"
            ),
        ),
    ),
    CallsiteRegistration(
        "dd.ids-enrichment",
        _source(
            "imas_codex/graph/dd_ids_enrichment.py",
            "enrich_ids_nodes",
            "call_llm_structured",
        ),
        "direct",
        "'data-dictionary'",
        "IDSEnrichmentBatch",
        "active",
        (_route("data-dictionary", "dd-enrichment", "imas/ids_enrichment"),),
    ),
    CallsiteRegistration(
        "dd.worker-enrichment",
        _source(
            "imas_codex/graph/dd_workers.py", "_enrich_batch", "acall_llm_structured"
        ),
        "direct",
        "'data-dictionary'",
        "IMASPathEnrichmentBatch",
        "active",
        (_route("data-dictionary", "dd-enrichment", "imas/enrichment"),),
    ),
    CallsiteRegistration(
        "dd.worker-refinement",
        _source(
            "imas_codex/graph/dd_workers.py", "_refine_batch", "acall_llm_structured"
        ),
        "direct",
        "'data-dictionary'",
        "IMASPathEnrichmentBatch",
        "active",
        (_route("data-dictionary", "dd-enrichment", "imas/refinement"),),
    ),
    CallsiteRegistration(
        "mapping.signal-mapping-sync",
        _source("imas_codex/ids/mapping.py", "_call_llm", "call_llm_structured"),
        "direct",
        "'imas-mapping'",
        "caller-supplied",
        "active",
        (
            _route(
                "imas-mapping",
                "language",
                "mapping/target_assignment_system",
                "mapping/target_assignment",
                "mapping/signal_mapping_system",
                "mapping/signal_mapping",
                "mapping/assembly_system",
                "mapping/assembly",
            ),
        ),
    ),
    CallsiteRegistration(
        "mapping.signal-mapping-async",
        _source("imas_codex/ids/mapping.py", "_acall_llm", "acall_llm_structured"),
        "direct",
        "'imas-mapping'",
        "caller-supplied",
        "active",
        (
            _route(
                "imas-mapping",
                "language",
                "mapping/target_assignment_system",
                "mapping/target_assignment",
                "mapping/signal_mapping_system",
                "mapping/signal_mapping",
                "mapping/assembly_system",
                "mapping/assembly",
            ),
        ),
    ),
    CallsiteRegistration(
        "mapping.metadata-population",
        _source(
            "imas_codex/ids/metadata.py", "populate_metadata", "call_llm_structured"
        ),
        "direct",
        "'imas-mapping'",
        "MetadataPopulationResponse",
        "active",
        (
            _route(
                "imas-mapping",
                "language",
                "mapping/metadata_population_system",
                "mapping/metadata_population",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.benchmark-description-review",
        _source(
            "imas_codex/standard_names/benchmark.py",
            "score_descriptions",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "StandardNameQualityReviewDescriptionBatch",
        "active",
        (
            _route(
                "standard-names",
                "sn-benchmark.reviewer-candidate",
                "sn/review_description_system",
                "sn/review_description_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.benchmark-name-review",
        _source(
            "imas_codex/standard_names/benchmark.py",
            "score_with_reviewer",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "response_model",
        "active",
        (
            _route(
                "standard-names",
                "sn-benchmark.reviewer-candidate",
                "sn/review_names_system",
                "sn/review_names_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.benchmark-docs-generation",
        _source(
            "imas_codex/standard_names/benchmark.py",
            "generate_docs_for_candidates._gen_one",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "GeneratedDocs",
        "active",
        (
            _route(
                "standard-names",
                "sn-benchmark.docs-candidate",
                "sn/generate_docs_system",
                "sn/generate_docs_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.benchmark-name-composition",
        _source(
            "imas_codex/standard_names/benchmark.py",
            "_run_model",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "StandardNameComposeBatch",
        "active",
        (
            _route(
                "standard-names",
                "sn-benchmark.compose-candidate",
                "sn/generate_name_system",
                "sn/generate_name_dd",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.benchmark-docs-review",
        _source(
            "imas_codex/standard_names/benchmark.py",
            "score_with_reviewer._score_one_doc",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "response_model",
        "active",
        (
            _route(
                "standard-names",
                "sn-benchmark.reviewer-candidate",
                "sn/review_docs_system",
                "sn/review_docs_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.benchmark-name-refinement",
        _source(
            "imas_codex/standard_names/benchmark_roles.py",
            "run_refine_bench._refine_one",
            "acall_llm_structured",
            1,
        ),
        "direct",
        "'standard-names'",
        "RefinedName",
        "active",
        (
            _route(
                "standard-names",
                "sn-benchmark.refine-candidate",
                "sn/refine_name_system",
                "sn/refine_name_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.benchmark-refinement-judge",
        _source(
            "imas_codex/standard_names/benchmark_roles.py",
            "run_refine_bench._refine_one",
            "acall_llm_structured",
            2,
        ),
        "direct",
        "'standard-names'",
        "_RefineJudgement",
        "active",
        (
            _route(
                "standard-names",
                "sn-benchmark.judge-candidate",
                "inline:refinement-benchmark-judge-system",
                "inline:refinement-benchmark-judge-user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.fanout-proposal",
        _source(
            "imas_codex/standard_names/fanout/dispatcher.py",
            "propose",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "FanoutPlan",
        "active",
        (
            _route(
                "standard-names",
                "sn-fanout.proposer",
                "sn/fanout_propose",
                "inline:fanout-proposal-user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.physics-judgement",
        _source(
            "imas_codex/standard_names/physics_judge.py",
            "judge_name_physics",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "PhysicsVerdict",
        "active-public",
        (
            _route(
                "standard-names",
                "sn-benchmark.reviewer",
                "sn/judge_physics_correctness_system",
                "sn/judge_physics_correctness_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.prose-adjudication",
        _source(
            "imas_codex/standard_names/prose_adjudicator.py",
            "_adjudicate_one",
            "acall_llm_structured",
        ),
        "direct",
        "service",
        "ProseVerdict",
        "active",
        (
            _route(
                "standard-names",
                "sn-prose-adjudicator",
                "inline:prose-adjudicator-system",
                "inline:prose-adjudicator-user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.merge-notes",
        _source(
            "imas_codex/standard_names/release_notes.py",
            "build_merge_notes",
            "call_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "MergeNotes",
        "active",
        (
            _route(
                "standard-names",
                "sn-release-notes",
                "sn/merge_notes_system",
                "sn/merge_notes_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.release-notes",
        _source(
            "imas_codex/standard_names/release_notes.py",
            "build_pr_notes",
            "call_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "PrNotes",
        "active",
        (
            _route(
                "standard-names",
                "sn-release-notes",
                "sn/release_notes_system",
                "sn/release_notes_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.review-batch",
        _source(
            "imas_codex/standard_names/review/pipeline.py",
            "_review_single_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "response_model",
        "active",
        (
            _route("standard-names", "sn-review.names", "sn/review_names"),
            _route("standard-names", "sn-review.docs", "sn/review_docs"),
        ),
    ),
    CallsiteRegistration(
        "standard-names.grammar-retry",
        _source("imas_codex/standard_names/workers.py", "_grammar_retry", "acall_fn"),
        "injected",
        "'standard-names'",
        "GrammarRetryResponse",
        "active",
        (
            _route(
                "standard-names",
                "sn-compose",
                "inline:grammar-retry-user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.candidate-self-refinement",
        _source(
            "imas_codex/standard_names/workers.py", "_self_refine_candidate", "acall_fn"
        ),
        "injected",
        "'standard-names'",
        "SelfRefineResponse",
        "active",
        (
            _route(
                "standard-names",
                "sn-compose",
                "sn/self_refine_system",
                "sn/self_refine_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.quorum-cycle",
        _source(
            "imas_codex/standard_names/workers.py",
            "_run_rd_quorum_cycles._run_cycle",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "response_model",
        "active",
        (
            _route(
                "standard-names",
                "sn-review.names",
                "sn/review_names_system",
                "sn/review_names_user",
            ),
            _route(
                "standard-names",
                "sn-review.docs",
                "sn/review_docs_system",
                "sn/review_docs_user",
                "sn/review_docs_parent_system",
                "sn/review_docs_parent_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.documentation-generation",
        _source(
            "imas_codex/standard_names/workers.py",
            "process_generate_docs_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "GeneratedDocs",
        "active",
        (
            _route(
                "standard-names",
                "sn-docs",
                "sn/generate_docs_system",
                "sn/generate_docs_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.parent-enrichment",
        _source(
            "imas_codex/standard_names/workers.py",
            "process_enrich_parents_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "EnrichedParentDescription",
        "active",
        (
            _route(
                "standard-names",
                "sn-parent-enrich",
                "sn/enrich_parent_system",
                "sn/enrich_parent_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.documentation-refinement",
        _source(
            "imas_codex/standard_names/workers.py",
            "process_refine_docs_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "RefinedDocs",
        "active",
        (
            _route(
                "standard-names",
                "sn-refine",
                "sn/refine_docs_system",
                "sn/refine_docs_user",
            ),
            _route(
                "standard-names",
                "sn-escalation",
                "sn/refine_docs_system",
                "sn/refine_docs_user",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.name-composition-worker",
        _source(
            "imas_codex/standard_names/workers.py",
            "compose_worker._compose_batch_body",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "StandardNameComposeBatch",
        "active",
        (
            _route(
                "standard-names",
                "sn-compose",
                "sn/generate_name_system",
                "sn/generate_name_dd",
                "sn/generate_name_dd_names",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.name-composition-batch",
        _source(
            "imas_codex/standard_names/workers.py",
            "compose_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "StandardNameComposeBatch",
        "active",
        (
            _route(
                "standard-names",
                "sn-compose",
                "sn/generate_name_system",
                "sn/generate_name_dd",
            ),
        ),
    ),
    CallsiteRegistration(
        "standard-names.name-refinement",
        _source(
            "imas_codex/standard_names/workers.py",
            "process_refine_name_batch",
            "acall_llm_structured",
        ),
        "direct",
        "'standard-names'",
        "RefinedName",
        "active",
        (
            _route(
                "standard-names",
                "sn-refine",
                "sn/refine_name_system",
                "sn/refine_name_user",
            ),
            _route(
                "standard-names",
                "sn-escalation",
                "sn/refine_name_system",
                "sn/refine_name_user",
            ),
        ),
    ),
)


def get_callsite_registration(callsite_id: str) -> CallsiteRegistration:
    """Return one exact callsite registration or fail closed."""
    matches = [entry for entry in CALLSITE_REGISTRY if entry.callsite_id == callsite_id]
    if len(matches) != 1:
        raise CallsitePolicyError(
            f"Expected one registered callsite {callsite_id!r}; found {len(matches)}"
        )
    return matches[0]


def get_route_binding(
    callsite_id: str,
    *,
    route_id: str,
) -> RouteBinding:
    """Resolve an exact registered route for a future typed dispatch policy."""
    registration = get_callsite_registration(callsite_id)
    matches = [route for route in registration.routes if route.route_id == route_id]
    if len(matches) != 1:
        raise CallsitePolicyError(
            "Context policy does not identify one registered route: "
            f"callsite={callsite_id!r}, route={route_id!r}, matches={len(matches)}"
        )
    return matches[0]


_LEGACY_DISPATCH_SYMBOLS = frozenset(
    {"call_llm", "acall_llm", "call_llm_structured", "acall_llm_structured"}
)
_TYPED_DISPATCH_SYMBOLS = frozenset({"dispatch_context", "adispatch_context"})
_DISPATCH_MODULES = frozenset(
    {
        "imas_codex.discovery.base",
        "imas_codex.discovery.base.llm",
        "imas_codex.llm.context_dispatch",
    }
)


def _base_import_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {"asyncio": "asyncio"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in _DISPATCH_MODULES or imported.name == "asyncio":
                    bindings[imported.asname or imported.name.split(".")[0]] = (
                        imported.name
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module in _DISPATCH_MODULES:
                for imported in node.names:
                    if (
                        imported.name
                        in _LEGACY_DISPATCH_SYMBOLS | _TYPED_DISPATCH_SYMBOLS
                    ):
                        bindings[imported.asname or imported.name] = imported.name
            elif node.module == "asyncio":
                for imported in node.names:
                    if imported.name == "to_thread":
                        bindings[imported.asname or imported.name] = "asyncio.to_thread"
    return bindings


def _keyword_expression(node: ast.Call, name: str) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return ast.unparse(keyword.value)
    return None


class _StructuredCallVisitor(ast.NodeVisitor):
    def __init__(
        self,
        source_path: str,
        bindings: Mapping[str, str],
        registry: tuple[CallsiteRegistration, ...],
    ) -> None:
        self.source_path = source_path
        self.scopes: list[str] = []
        self.calls: list[StructuredCall] = []
        self._occurrences: dict[tuple[str, str], int] = {}
        self._bindings = dict(bindings)
        self._injected_by_scope = {
            (entry.source.source_path, entry.source.scope): entry.source.dispatch_symbol
            for entry in registry
            if entry.dispatch_style == "injected"
        }

    def _resolve(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self._bindings.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            owner = self._resolve(node.value)
            dotted = f"{owner}.{node.attr}" if owner else node.attr
            if dotted in self._bindings:
                return self._bindings[dotted]
            if node.attr in _LEGACY_DISPATCH_SYMBOLS | _TYPED_DISPATCH_SYMBOLS:
                return node.attr
            return dotted
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            if imported.name in _DISPATCH_MODULES or imported.name == "asyncio":
                self._bindings[imported.asname or imported.name.split(".")[0]] = (
                    imported.name
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in _DISPATCH_MODULES:
            for imported in node.names:
                if imported.name in _LEGACY_DISPATCH_SYMBOLS | _TYPED_DISPATCH_SYMBOLS:
                    self._bindings[imported.asname or imported.name] = imported.name
        elif node.module == "asyncio":
            for imported in node.names:
                if imported.name == "to_thread":
                    self._bindings[imported.asname or imported.name] = (
                        "asyncio.to_thread"
                    )

    def visit_Assign(self, node: ast.Assign) -> None:
        resolved = self._resolve(node.value)
        if resolved in _LEGACY_DISPATCH_SYMBOLS | _TYPED_DISPATCH_SYMBOLS:
            for target in node.targets:
                target_name = self._target_name(target)
                if target_name is not None:
                    self._bindings[target_name] = resolved
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        resolved = self._resolve(node.value) if node.value is not None else None
        if resolved in _LEGACY_DISPATCH_SYMBOLS | _TYPED_DISPATCH_SYMBOLS:
            target_name = self._target_name(node.target)
            if target_name is not None:
                self._bindings[target_name] = resolved
        self.generic_visit(node)

    def _target_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            owner = self._resolve(node.value)
            return f"{owner}.{node.attr}" if owner else None
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def _visit_scope(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        previous_bindings = self._bindings.copy()
        self.scopes.append(node.name)
        scope = ".".join(self.scopes)
        injected = self._injected_by_scope.get((self.source_path, scope))
        if injected is not None:
            parameter_names = {
                argument.arg
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            }
            if injected in parameter_names:
                self._bindings[injected] = injected
        self.generic_visit(node)
        self.scopes.pop()
        assigned_attributes = {
            name: value
            for name, value in self._bindings.items()
            if "." in name and previous_bindings.get(name) != value
        }
        self._bindings = previous_bindings
        self._bindings.update(assigned_attributes)

    def visit_Call(self, node: ast.Call) -> None:
        dispatch_symbol: str | None = None
        dispatch_style: DispatchStyle | None = None
        transition_kind: TransitionKind = "legacy"
        callsite_id: str | None = None
        route_id: str | None = None
        symbol = self._resolve(node.func)

        if symbol in _LEGACY_DISPATCH_SYMBOLS:
            dispatch_symbol = symbol
            dispatch_style = "direct"
        elif symbol == "asyncio.to_thread" and node.args:
            injected_symbol = self._resolve(node.args[0])
            if injected_symbol in _LEGACY_DISPATCH_SYMBOLS | _TYPED_DISPATCH_SYMBOLS:
                dispatch_symbol = injected_symbol
                dispatch_style = "to-thread"
                if injected_symbol in _TYPED_DISPATCH_SYMBOLS:
                    transition_kind = "typed"
        elif symbol in set(self._injected_by_scope.values()):
            dispatch_symbol = symbol
            dispatch_style = "injected"
            if (
                _keyword_expression(node, "callsite_id") is not None
                or _keyword_expression(node, "route_id") is not None
            ):
                transition_kind = "typed"
        elif symbol in _TYPED_DISPATCH_SYMBOLS:
            dispatch_symbol = symbol
            dispatch_style = (
                "typed-async" if symbol == "adispatch_context" else "typed-sync"
            )
            transition_kind = "typed"
        if transition_kind == "typed":
            argument_offset = 1 if symbol == "asyncio.to_thread" else 0
            expression = _keyword_expression(node, "callsite_id")
            if expression is None and len(node.args) >= argument_offset + 2:
                expression = ast.unparse(node.args[argument_offset + 1])
            if expression is not None:
                try:
                    parsed = ast.literal_eval(expression)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, str):
                    callsite_id = parsed
            route_expression = _keyword_expression(node, "route_id")
            if route_expression is not None:
                try:
                    parsed_route = ast.literal_eval(route_expression)
                except (ValueError, SyntaxError):
                    parsed_route = None
                if isinstance(parsed_route, str):
                    route_id = parsed_route

        if dispatch_symbol is not None and dispatch_style is not None:
            scope = ".".join(self.scopes) or "<module>"
            key = (scope, dispatch_symbol)
            occurrence = self._occurrences.get(key, 0) + 1
            self._occurrences[key] = occurrence
            self.calls.append(
                StructuredCall(
                    source=SourceCallIdentity(
                        self.source_path,
                        scope,
                        dispatch_symbol,
                        occurrence,
                    ),
                    dispatch_style=dispatch_style,
                    service_argument=_keyword_expression(node, "service"),
                    model_argument=_keyword_expression(
                        node,
                        "candidate_model" if transition_kind == "typed" else "model",
                    ),
                    response_model_argument=_keyword_expression(node, "response_model"),
                    transition_kind=transition_kind,
                    callsite_id=callsite_id,
                    route_id=route_id,
                    line=node.lineno,
                )
            )

        self.generic_visit(node)


def _parse_source(path: Path) -> ast.Module:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        location = f"{path}:{exc.lineno or '?'}:{exc.offset or '?'}"
        raise CallsiteSourceSyntaxError(
            f"Structured-call inventory cannot parse {location}: {exc.msg}"
        ) from exc


def scan_structured_calls(
    project_root: Path | str = Path("."),
    source_directory: str = "imas_codex",
    registry: tuple[CallsiteRegistration, ...] = CALLSITE_REGISTRY,
) -> tuple[StructuredCall, ...]:
    """Scan every project source file for semantic structured dispatches."""

    root = Path(project_root)
    source_root = root / source_directory
    calls: list[StructuredCall] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse_source(path)
        relative_path = path.relative_to(root).as_posix()
        visitor = _StructuredCallVisitor(
            relative_path, _base_import_bindings(tree), registry
        )
        visitor.visit(tree)
        calls.extend(visitor.calls)
    return tuple(calls)


def assert_registry_current(
    project_root: Path | str = Path("."),
    registry: tuple[CallsiteRegistration, ...] = CALLSITE_REGISTRY,
    *,
    typed_policy_registry: Mapping[str, object] | None = None,
) -> tuple[StructuredCall, ...]:
    """Return observed calls or raise with a complete registry drift report."""

    observed = scan_structured_calls(project_root, registry=registry)
    registered_by_source = {entry.source: entry for entry in registry}
    legacy_calls = [call for call in observed if call.transition_kind == "legacy"]
    typed_calls = [call for call in observed if call.transition_kind == "typed"]
    observed_by_source = {call.source: call for call in legacy_calls}
    typed_by_route = {
        (call.callsite_id, call.route_id): call
        for call in typed_calls
        if call.callsite_id is not None and call.route_id is not None
    }
    problems: list[str] = []

    if len(registered_by_source) != len(registry):
        problems.append("registry contains duplicate source identities")
    callsite_ids = [entry.callsite_id for entry in registry]
    if len(set(callsite_ids)) != len(callsite_ids):
        problems.append("registry contains duplicate callsite ids")
    if len(observed_by_source) != len(legacy_calls):
        problems.append("scanner produced duplicate legacy source identities")
    if len(typed_by_route) != len(typed_calls):
        problems.append(
            "typed dispatches require unique literal callsite and route identities"
        )

    for source in sorted(observed_by_source.keys() - registered_by_source.keys()):
        call = observed_by_source[source]
        problems.append(
            f"unregistered dispatch {source.source_path}:{call.line} "
            f"{source.scope} {source.dispatch_symbol} occurrence {source.occurrence}"
        )
    registered_ids = {entry.callsite_id for entry in registry}
    for callsite_id, route_id in sorted(
        identity for identity in typed_by_route if identity[0] not in registered_ids
    ):
        call = typed_by_route[(callsite_id, route_id)]
        problems.append(
            f"unregistered typed dispatch {call.source.source_path}:{call.line} "
            f"{callsite_id!r}"
        )
    registered_routes = {
        (entry.callsite_id, route.route_id)
        for entry in registry
        for route in entry.routes
    }
    for identity in sorted(set(typed_by_route) - registered_routes):
        call = typed_by_route[identity]
        problems.append(
            f"unregistered typed route {call.source.source_path}:{call.line} "
            f"{identity!r}"
        )
    for entry in registry:
        legacy = observed_by_source.get(entry.source)
        typed = [
            call
            for (callsite_id, _), call in typed_by_route.items()
            if callsite_id == entry.callsite_id
        ]
        if (legacy is None) == (not typed):
            problems.append(
                f"{entry.callsite_id} must have exactly one legacy or typed expression"
            )
            continue
        if typed and (
            typed[0].source.source_path != entry.source.source_path
            or typed[0].source.scope != entry.source.scope
        ):
            problems.append(
                f"{entry.callsite_id} typed carrier changed: "
                f"{entry.source.source_path}:{entry.source.scope} -> "
                f"{typed[0].source.source_path}:{typed[0].source.scope}"
            )

    for source in sorted(registered_by_source.keys() & observed_by_source.keys()):
        entry = registered_by_source[source]
        call = observed_by_source[source]
        if call.dispatch_style != entry.dispatch_style:
            problems.append(
                f"{entry.callsite_id} dispatch style changed: "
                f"{entry.dispatch_style!r} -> {call.dispatch_style!r}"
            )
        if call.service_argument != entry.service_argument:
            problems.append(
                f"{entry.callsite_id} service argument changed: "
                f"{entry.service_argument!r} -> {call.service_argument!r}"
            )

    for entry in registry:
        if not entry.routes:
            problems.append(f"{entry.callsite_id} has no registered route")
        for route in entry.routes:
            if (
                not route.route_id
                or not route.service
                or not route.seat
                or not route.model_source
                or not route.templates
            ):
                problems.append(f"{entry.callsite_id} has incomplete route metadata")
        route_ids = [route.route_id for route in entry.routes]
        if len(route_ids) != len(set(route_ids)):
            problems.append(f"{entry.callsite_id} has duplicate route identities")

    if problems:
        raise CallsiteInventoryError(
            "Structured LLM callsite inventory drift:\n- " + "\n- ".join(problems)
        )
    if typed_calls:
        from imas_codex.llm.dispatch_policy_registry import policy_registry_closure

        policy_registry_closure(observed, registry=typed_policy_registry)
    return observed


def assert_zero_legacy_dispatches(
    project_root: Path | str = Path("."),
    registry: tuple[CallsiteRegistration, ...] = CALLSITE_REGISTRY,
    *,
    typed_policy_registry: Mapping[str, object] | None = None,
) -> tuple[StructuredCall, ...]:
    """Enforce the final transition closure, including legacy wrapper removal."""
    observed = assert_registry_current(
        project_root, registry, typed_policy_registry=typed_policy_registry
    )
    problems: list[str] = []
    legacy = [call for call in observed if call.transition_kind == "legacy"]
    if legacy:
        problems.append(f"zero legacy expressions; found {len(legacy)}")
    from imas_codex.discovery import base
    from imas_codex.discovery.base import llm

    wrapper_names = (
        "call_llm",
        "acall_llm",
        "call_llm_structured",
        "acall_llm_structured",
    )
    wrapper_objects = {
        getattr(llm, name)
        for name in wrapper_names
        if callable(getattr(llm, name, None))
    }
    public_wrappers = sorted(
        {
            f"{module.__name__}.{name}"
            for module in (base, llm)
            for name, value in vars(module).items()
            if not name.startswith("_") and callable(value) and value in wrapper_objects
        }
    )
    if public_wrappers:
        problems.append(
            f"remove every public raw-message wrapper surface: {public_wrappers}"
        )
    if problems:
        raise CallsiteInventoryError(
            "Typed dispatch closure requires " + "; ".join(problems)
        )
    return observed


_RAW_PROVIDER_TRANSPORT = "imas_codex/discovery/base/llm.py"
_RAW_PROVIDER_HEALTH_PROBE = (
    "imas_codex/discovery/base/services.py",
    "_probe_litellm_local",
)
_RAW_PROVIDER_TRANSPORT_SCOPES = frozenset(
    {
        "_acompletion_local",
        "call_llm_structured",
        "acall_llm_structured",
        "_call_frozen_structured_transport",
        "_acall_frozen_structured_transport",
        "call_llm",
        "acall_llm",
    }
)


def _dotted_symbol(node: ast.expr) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


class _ProviderCallVisitor(ast.NodeVisitor):
    def __init__(self, source_path: str, aliases: dict[str, str]) -> None:
        self.source_path = source_path
        self.aliases = aliases
        self.scopes: list[str] = []
        self.calls: list[ProviderCall] = []

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            if imported.name == "litellm":
                self.aliases[imported.asname or imported.name] = "litellm"

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "litellm":
            for imported in node.names:
                if imported.name in {"completion", "acompletion"}:
                    self.aliases[imported.asname or imported.name] = (
                        f"litellm.{imported.name}"
                    )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def _visit_scope(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        previous_aliases = self.aliases.copy()
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()
        assigned_attributes = {
            name: value
            for name, value in self.aliases.items()
            if "." in name and previous_aliases.get(name) != value
        }
        self.aliases = previous_aliases
        self.aliases.update(assigned_attributes)

    def _resolve_symbol(self, node: ast.expr) -> str | None:
        symbol = _dotted_symbol(node)
        if symbol is None:
            return None
        first, separator, remainder = symbol.partition(".")
        if first in self.aliases:
            return self.aliases[first] + (separator + remainder if separator else "")
        return self.aliases.get(symbol, symbol)

    def _bind_target(self, target: ast.expr, symbol: str | None) -> None:
        if symbol is None:
            return
        target_name = _dotted_symbol(target)
        if target_name is not None:
            self.aliases[target_name] = symbol

    def visit_Assign(self, node: ast.Assign) -> None:
        symbol = self._resolve_symbol(node.value)
        for target in node.targets:
            self._bind_target(target, symbol)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        symbol = self._resolve_symbol(node.value) if node.value is not None else None
        self._bind_target(node.target, symbol)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        symbol = self._resolve_symbol(node.func)

        if symbol in {"litellm.completion", "litellm.acompletion"} or (
            symbol is not None
            and (
                symbol.endswith(".chat.completions.create")
                or symbol.endswith(".responses.create")
            )
        ):
            self.calls.append(
                ProviderCall(
                    source_path=self.source_path,
                    scope=".".join(self.scopes) or "<module>",
                    symbol=symbol,
                    line=node.lineno,
                )
            )
        self.generic_visit(node)


def _provider_import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "litellm":
                    aliases[imported.asname or imported.name] = "litellm"
        elif isinstance(node, ast.ImportFrom) and node.module == "litellm":
            for imported in node.names:
                if imported.name in {"completion", "acompletion"}:
                    aliases[imported.asname or imported.name] = (
                        f"litellm.{imported.name}"
                    )
    return aliases


def scan_provider_bypasses(
    project_root: Path | str = Path("."),
    source_directory: str = "imas_codex",
) -> tuple[ProviderCall, ...]:
    """Find raw provider dispatches outside the transport and health probe."""

    root = Path(project_root)
    bypasses: list[ProviderCall] = []
    for path in sorted((root / source_directory).rglob("*.py")):
        tree = _parse_source(path)
        relative_path = path.relative_to(root).as_posix()
        visitor = _ProviderCallVisitor(relative_path, _provider_import_aliases(tree))
        visitor.visit(tree)
        for call in visitor.calls:
            is_transport = (
                call.source_path == _RAW_PROVIDER_TRANSPORT
                and call.scope in _RAW_PROVIDER_TRANSPORT_SCOPES
            )
            is_health_probe = (
                call.source_path,
                call.scope,
            ) == _RAW_PROVIDER_HEALTH_PROBE
            if not is_transport and not is_health_probe:
                bypasses.append(call)
    return tuple(bypasses)


def assert_no_provider_bypasses(project_root: Path | str = Path(".")) -> None:
    """Fail if a business module dispatches directly to a provider SDK."""

    bypasses = scan_provider_bypasses(project_root)
    if bypasses:
        details = "\n".join(
            f"- {call.source_path}:{call.line} {call.scope} {call.symbol}"
            for call in bypasses
        )
        raise CallsiteInventoryError(f"Raw provider dispatch bypasses:\n{details}")

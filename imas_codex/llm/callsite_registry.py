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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DispatchStyle = Literal["direct", "to-thread", "injected"]
Reachability = Literal["active", "active-public"]


@dataclass(frozen=True, slots=True)
class RouteBinding:
    """A service/seat/template route reachable through one source expression."""

    service: str
    seat: str
    templates: tuple[str, ...]


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


def _route(service: str, seat: str, *templates: str) -> RouteBinding:
    return RouteBinding(service=service, seat=seat, templates=templates)


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
    service: str,
    seat: str,
    templates: tuple[str, ...],
) -> RouteBinding:
    """Resolve an exact registered route for a future typed dispatch policy."""
    registration = get_callsite_registration(callsite_id)
    selected = set(templates)
    matches = [
        route
        for route in registration.routes
        if route.service == service
        and route.seat == seat
        and selected
        and selected.issubset(route.templates)
    ]
    if len(matches) != 1:
        raise CallsitePolicyError(
            "Context policy does not identify one registered route: "
            f"callsite={callsite_id!r}, service={service!r}, seat={seat!r}, "
            f"templates={templates!r}, matches={len(matches)}"
        )
    return matches[0]


def _call_symbol(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _keyword_expression(node: ast.Call, name: str) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return ast.unparse(keyword.value)
    return None


class _StructuredCallVisitor(ast.NodeVisitor):
    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.scopes: list[str] = []
        self.calls: list[StructuredCall] = []
        self._occurrences: dict[tuple[str, str], int] = {}

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
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        dispatch_symbol: str | None = None
        dispatch_style: DispatchStyle | None = None
        symbol = _call_symbol(node.func)

        if symbol in {"call_llm_structured", "acall_llm_structured"}:
            dispatch_symbol = symbol
            dispatch_style = "direct"
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_thread"
            and node.args
        ):
            injected_symbol = _call_symbol(node.args[0])
            if injected_symbol in {"call_llm_structured", "acall_llm_structured"}:
                dispatch_symbol = injected_symbol
                dispatch_style = "to-thread"
        elif symbol == "acall_fn":
            dispatch_symbol = symbol
            dispatch_style = "injected"

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
                    model_argument=_keyword_expression(node, "model"),
                    response_model_argument=_keyword_expression(node, "response_model"),
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
) -> tuple[StructuredCall, ...]:
    """Scan every project source file for semantic structured dispatches."""

    root = Path(project_root)
    source_root = root / source_directory
    calls: list[StructuredCall] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse_source(path)
        relative_path = path.relative_to(root).as_posix()
        visitor = _StructuredCallVisitor(relative_path)
        visitor.visit(tree)
        calls.extend(visitor.calls)
    return tuple(calls)


def assert_registry_current(
    project_root: Path | str = Path("."),
    registry: tuple[CallsiteRegistration, ...] = CALLSITE_REGISTRY,
) -> tuple[StructuredCall, ...]:
    """Return observed calls or raise with a complete registry drift report."""

    observed = scan_structured_calls(project_root)
    registered_by_source = {entry.source: entry for entry in registry}
    observed_by_source = {call.source: call for call in observed}
    problems: list[str] = []

    if len(registered_by_source) != len(registry):
        problems.append("registry contains duplicate source identities")
    callsite_ids = [entry.callsite_id for entry in registry]
    if len(set(callsite_ids)) != len(callsite_ids):
        problems.append("registry contains duplicate callsite ids")
    if len(observed_by_source) != len(observed):
        problems.append("scanner produced duplicate source identities")

    for source in sorted(observed_by_source.keys() - registered_by_source.keys()):
        call = observed_by_source[source]
        problems.append(
            f"unregistered dispatch {source.source_path}:{call.line} "
            f"{source.scope} {source.dispatch_symbol} occurrence {source.occurrence}"
        )
    for source in sorted(registered_by_source.keys() - observed_by_source.keys()):
        entry = registered_by_source[source]
        problems.append(f"missing registered dispatch {entry.callsite_id}: {source!r}")

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
            if not route.service or not route.seat or not route.templates:
                problems.append(
                    f"{entry.callsite_id} has incomplete service/seat/template metadata"
                )

    if problems:
        raise CallsiteInventoryError(
            "Structured LLM callsite inventory drift:\n- " + "\n- ".join(problems)
        )
    return observed


_RAW_PROVIDER_TRANSPORT = "imas_codex/discovery/base/llm.py"
_RAW_PROVIDER_HEALTH_PROBE = (
    "imas_codex/discovery/base/services.py",
    "_probe_litellm_local",
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
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        symbol = _dotted_symbol(node.func)
        if symbol is not None:
            first, separator, remainder = symbol.partition(".")
            if first in self.aliases:
                symbol = self.aliases[first] + (
                    separator + remainder if separator else ""
                )

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
            is_transport = call.source_path == _RAW_PROVIDER_TRANSPORT
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

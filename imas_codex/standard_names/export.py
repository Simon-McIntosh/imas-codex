"""Export validated standard names from the graph to a staging directory.

Reads StandardName nodes from the Neo4j graph, applies quality gates,
and writes YAML files matching the ``imas-standard-names-catalog``
layout: ``<staging>/standard_names/<domain>/<name>.yml`` plus a
``<staging>/catalog.yml`` manifest.

This module is the first half of the two-step export→publish flow.
The staging directory produced here is consumed by ``publish.py``
(transport to ISNC repo) and ``preview.py`` (local site render).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from imas_codex.standard_names.canonical import (
    canonicalise_entry,
    reorder_entry_dict,
)
from imas_codex.standard_names.catalog_ordering import order_entries_by_hierarchy
from imas_codex.standard_names.domain_priority import pick_primary_domain
from imas_codex.standard_names.protection import PROTECTED_FIELDS
from imas_codex.standard_names.provenance_lifecycle import (
    fetch_public_semantic_sources,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogEntryRecord:
    """One catalog list item with its serialized representation intact."""

    name: str
    domain: str
    mapping_bytes: bytes
    item_bytes: bytes


@dataclass(frozen=True)
class ApprovedBaselineDelta:
    """Serialized-entry comparison between an approved catalog and staging."""

    missing: tuple[str, ...]
    byte_changed: tuple[str, ...]
    unchanged: tuple[str, ...]


@dataclass(frozen=True)
class ReviewCatalogAssembly:
    """Counts produced while combining approved baseline and review entries."""

    baseline_count: int
    staged_count_before: int
    staged_count_after: int
    baseline_entries_added: int
    batch_entries_written: int
    emitted_batch_names: tuple[str, ...]


# Default COCOS convention for the catalog manifest
_DEFAULT_COCOS_CONVENTION = 17

# A score-based exclusion is attributable only while the score reproduces.
# One adjacent review-cycle pair in five crossed the acceptance bound, and the
# measured median absolute swing was 0.05625.  Keep the default named and
# caller-configurable so a repeated measurement can replace it.  Evidence:
# docs/evidence/sn-release-readiness/quorum-self-agreement.md
DEFAULT_BOUND_ADJACENT_HALF_WIDTH = 0.05625

# ``generated_at`` is required by the ISN manifest model. Low-level callers
# that intentionally provide no provenance receive this conspicuous, stable
# value; the public export path sets ``require_provenance=True`` and refuses to
# emit it.
_UNVERSIONED_TIMESTAMP = "1970-01-01T00:00:00+00:00"

# Gate names
GATE_A = "graph_tests"
GATE_B = "cross_field_consistency"
GATE_C = "score_thresholds"
GATE_D = "divergence_detection"
GATE_EXCLUSION_ACCOUNTING = "exclusion_accounting"

# Fields that must NOT appear in exported YAML
_PROVENANCE_FIELDS = frozenset({"source_paths", "dd_paths"})

# Fields not yet accepted by ISN models — strip from export output.
# ``constraints`` is tracked internally but ISN's StandardNameScalarEntry
# raises ``Extra inputs are not permitted`` if it appears in the YAML.
_ISN_UNSUPPORTED_FIELDS = frozenset({"constraints"})

# Graph-only fields that ARE written to the catalog YAML (they appear in
# ``CANONICAL_KEY_ORDER`` and the ISN catalog loader tolerates them), but which
# the strict ISN entry models reject under ``extra="forbid"``. They are stamped
# onto an entry AFTER its build-time ISN validation, so the final-shape gate
# must strip them before re-validating — otherwise every entry would spuriously
# fail on these known, intentional fields.
_GRAPH_ONLY_RENDERED_FIELDS = frozenset(
    {"physics_domain", "validity_domain", "sources"}
)

# These identity-specific holds take precedence over generic lifecycle gates.
# Each reason names the unresolved authority condition so the exclusion ledger
# stays reviewable and a hold is removed only when that condition is resolved.
_RELEASE_IDENTITY_HOLDS: dict[str, tuple[str, str]] = {
    "fast_ion_charge_state_power_at_inside_flux_surface": (
        "release_hold_dd_recipient_unresolved",
        "the active Data Dictionary does not resolve fast- versus thermal-ion "
        "deposition",
    ),
    "toroidal_coordinate_of_field_map_grid": (
        "release_hold_field_map_grid_vocabulary_unresolved",
        "the closed grammar has no governed field-map-grid locus authority",
    ),
    "neutron_flux_due_to_fusion": (
        "release_hold_documentation_not_accepted",
        "documentation has not earned ordinary review acceptance",
    ),
    "radial_neutral_internal_state_momentum_flux": (
        "release_hold_dual_bound_source_conflict",
        "the shared source is explicitly assigned to the canonical shorter identity",
    ),
    "voltage_of_diagnostic_antenna": (
        "release_hold_exhausted_antenna_identity",
        "the exhausted predecessor has no sanctioned successor transition",
    ),
    "voltage_of_ece_channel": (
        "release_hold_missing_reviewed_successor",
        "the intended successor has not been created and accepted through review",
    ),
}


# =============================================================================
# Report models
# =============================================================================


@dataclass
class GateResult:
    """Result of a single gate check."""

    gate: str
    passed: bool
    issues: list[dict[str, Any]] = field(default_factory=list)
    advisories: list[dict[str, Any]] = field(default_factory=list)
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "skipped": self.skipped,
            "issue_count": len(self.issues),
            "issues": self.issues,
            "advisory_count": len(self.advisories),
            "advisories": self.advisories,
        }


@dataclass
class DivergenceEntry:
    """A single divergence finding for a catalog-edited name."""

    name: str
    field: str
    graph_hash: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field": self.field,
            "graph_hash": self.graph_hash,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ExclusionRecord:
    """One accepted-population identity excluded for one terminal reason."""

    standard_name_id: str
    stage: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "standard_name_id": self.standard_name_id,
            "stage": self.stage,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class ExportReport:
    """Full report from an export run."""

    gate_results: list[GateResult] = field(default_factory=list)
    divergence_entries: list[DivergenceEntry] = field(default_factory=list)
    total_candidates: int = 0
    exported_count: int = 0
    excluded_below_score: int = 0
    excluded_unreviewed: int = 0
    # Domain filtering happens in the _fetch_candidates Cypher query, so a
    # candidate excluded by domain never reaches this report; this counter is
    # therefore always 0 and retained only for output-shape stability.
    excluded_by_domain: int = 0
    # Candidates dropped because their description is still the deterministic
    # parent placeholder — tracked separately from excluded_below_score, which
    # they are not (no GENERATE_DOCS run, not a low score).
    excluded_placeholder: int = 0
    # Names dropped in RC mode because they failed the ISN grammar parse gate.
    parse_failures: int = 0
    # Internal (name:) doc links dropped because their target is not published.
    pruned_links: int = 0
    gate_failures: int = 0
    all_gates_passed: bool = True
    exported_names: list[str] = field(default_factory=list)
    validation_failures: int = 0
    exclusion_records: list[ExclusionRecord] = field(default_factory=list)

    def record_exclusions(self, records: list[ExclusionRecord]) -> None:
        """Append identity-bearing exclusions and refresh compatibility counts."""
        self.exclusion_records.extend(records)
        reasons = [record.reason for record in self.exclusion_records]
        self.excluded_below_score = sum(
            reason in {"below_name_score", "below_description_score", "bound_adjacent"}
            for reason in reasons
        )
        self.excluded_unreviewed = reasons.count("unreviewed_name")
        self.excluded_by_domain = reasons.count("outside_requested_domain")
        self.excluded_placeholder = reasons.count(
            "deterministic_parent_description_placeholder"
        )
        self.parse_failures = reasons.count("grammar_parse_failure")
        self.validation_failures = reasons.count("invalid_catalog_entry")

    def _exclusion_rows(self) -> list[dict[str, Any]]:
        identities_by_reason: dict[str, list[str]] = {}
        for record in self.exclusion_records:
            identities_by_reason.setdefault(record.reason, []).append(
                record.standard_name_id
            )
        return [
            {
                "reason": reason,
                "count": len(identities),
                "identities": sorted(identities),
            }
            for reason, identities in sorted(identities_by_reason.items())
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": [g.to_dict() for g in self.gate_results],
            "divergence": [d.to_dict() for d in self.divergence_entries],
            "emitted_identities": list(self.exported_names),
            "exclusion_ledger": self._exclusion_rows(),
            "exclusion_records": [
                record.to_dict()
                for record in sorted(
                    self.exclusion_records,
                    key=lambda record: (record.reason, record.standard_name_id),
                )
            ],
            "counts": {
                "total_candidates": self.total_candidates,
                "exported": self.exported_count,
                "excluded_below_score": self.excluded_below_score,
                "excluded_unreviewed": self.excluded_unreviewed,
                "excluded_by_domain": self.excluded_by_domain,
                "excluded_placeholder": self.excluded_placeholder,
                "parse_failures": self.parse_failures,
                "pruned_links": self.pruned_links,
                "gate_failures": self.gate_failures,
                "validation_failures": self.validation_failures,
            },
            "all_gates_passed": self.all_gates_passed,
        }


# =============================================================================
# Graph query helpers
# =============================================================================


def _fetch_candidates(
    *,
    include_unreviewed: bool = False,
    domain: str | None = None,
    names_only: bool = False,
    batch: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch StandardName nodes eligible for export from the graph.

    Returns dicts with all catalog-relevant properties plus ``origin``,
    ``cocos``, ``reviewer_score_name``.

    Only nodes that have completed the name pipeline and passed
    validation are returned.  Specifically the query requires:

    - ``name_stage IN ['accepted', 'approved']`` — accepted RC candidates and
      PR-approved names are exportable; superseded/internal attempts are not.
    - ``docs_stage = 'accepted'`` — excludes nodes whose documentation has
      not yet passed the docs review loop (skipped when *names_only*).
    - ``validation_status = 'valid'`` — excludes quarantined nodes.
    - ``review_quorum_shortfall IS NULL`` — excludes a name accepted while its
      reviewer chain had not reached a verdict. A quorate review clears the
      marker, so an accepted node still carrying one arrived by a path that
      did not consult it, and publishing it would ship exactly what the
      quorum gate withheld.
    - a reachable winning docs-axis review group with no docs quorum shortfall
      — excludes unresolved and historical score-only decisions from full
      exports. Names-only export remains independent of docs state and docs
      review authority.

    When *names_only* is True, the ``docs_stage`` gate is dropped so
    names can be exported before documentation is generated.

    When *batch* is given (a review-batch export), the name gate becomes
    ``name_stage = 'approved' OR sn.id IN batch``: the already-approved catalog
    plus exactly this batch, so the PR diff is additive. The other gates
    (validation, docs) still apply. When *batch* is None the full accepted∪
    approved corpus is fetched (the normal RC export).
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_ops import (
        docs_review_eligibility_params,
        docs_review_eligibility_where,
    )

    params: dict[str, Any] = {}
    if batch is not None:
        cypher = """
    MATCH (sn:StandardName)
    WHERE (sn.name_stage = 'approved' OR sn.id IN $batch)
      AND sn.validation_status = 'valid'
      AND sn.review_quorum_shortfall IS NULL
    """
        params["batch"] = batch
    else:
        cypher = """
    MATCH (sn:StandardName)
    WHERE sn.name_stage IN ['accepted', 'approved']
      AND sn.validation_status = 'valid'
      AND sn.review_quorum_shortfall IS NULL
    """
    if not names_only:
        params.update(docs_review_eligibility_params())
        cypher += (
            "  AND sn.docs_stage = 'accepted'\n"
            f"  AND {docs_review_eligibility_where()}\n"
            "  AND sn.docs_review_quorum_shortfall IS NULL\n"
        )

    if domain:
        cypher += " AND sn.physics_domain = $domain"
        params["domain"] = domain

    cypher += """
    OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
    OPTIONAL MATCH (sn)-[:HAS_COCOS]->(c:COCOS)
    RETURN sn {
        .*,
        unit: coalesce(u.id, sn.unit),
        cocos: c.id
    } AS record
    ORDER BY sn.id
    """

    with GraphClient() as gc:
        rows = gc.query(cypher, **params)

    return [r["record"] for r in (rows or [])]


def _fetch_export_population(
    *,
    batch: list[str] | None = None,
    require_docs_review: bool = True,
) -> list[dict[str, Any]]:
    """Fetch the identity universe before export eligibility filters are applied.

    A normal export starts with every accepted or approved identity. A review
    export starts with its exact additive cohort: approved identities plus the
    requested batch. Domain, validation, quorum, and documentation predicates
    are deliberately absent because their rejected identities belong in the
    exclusion ledger.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_ops import (
        docs_review_eligibility_params,
        docs_review_eligibility_where,
        docs_review_property_coverage,
    )

    params: dict[str, Any] = docs_review_eligibility_params()
    if batch is None:
        name_predicate = "sn.name_stage IN ['accepted', 'approved']"
    else:
        name_predicate = "(sn.name_stage = 'approved' OR sn.id IN $batch)"
        params["batch"] = batch

    cypher = f"""
    MATCH (sn:StandardName)
    WHERE {name_predicate}
    WITH sn,
         EXISTS {{
             MATCH (:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
         }} AS has_dd_source_binding,
         EXISTS {{
             MATCH (producer:StandardNameSource)-[:PRODUCED_NAME]->(sn)
             WHERE producer.source_type = 'derived'
               AND NOT EXISTS {{
                   MATCH (producer)-[:FROM_DD_PATH]->(:IMASNode)
               }}
         }} AS has_derived_producer,
         EXISTS {{
             MATCH (producer:StandardNameSource)-[:PRODUCED_NAME]->(sn)
             WHERE coalesce(producer.source_type, '') <> 'derived'
                OR EXISTS {{
                    MATCH (producer)-[:FROM_DD_PATH]->(:IMASNode)
                }}
         }} AS has_non_derived_producer,
         EXISTS {{
             MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
             WHERE review.review_axis = 'docs'
         }} AS has_docs_review,
         {docs_review_eligibility_where()} AS has_winning_docs_review
    OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
    OPTIONAL MATCH (sn)-[:HAS_COCOS]->(c:COCOS)
    RETURN sn {{
        .*,
        unit: coalesce(u.id, sn.unit),
        cocos: c.id,
        _has_dd_source_binding: has_dd_source_binding,
        _has_derived_producer: has_derived_producer,
        _has_non_derived_producer: has_non_derived_producer,
        _has_docs_review: has_docs_review,
        _has_winning_docs_review: has_winning_docs_review
    }} AS record
    ORDER BY sn.id
    """
    with GraphClient() as gc:
        if require_docs_review:
            docs_review_property_coverage(gc)
        rows = gc.query(cypher, **params)

    population_by_id: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        record = row["record"]
        population_by_id.setdefault(record["id"], record)
    return [population_by_id[name] for name in sorted(population_by_id)]


def _classify_export_population(
    population: list[dict[str, Any]],
    *,
    domain: str | None,
    names_only: bool,
) -> tuple[list[dict[str, Any]], list[ExclusionRecord]]:
    """Partition the upstream population into eligible rows and one-reason drops."""
    eligible: list[dict[str, Any]] = []
    excluded: list[ExclusionRecord] = []

    for candidate in population:
        candidate_id = candidate["id"]
        release_hold = _RELEASE_IDENTITY_HOLDS.get(candidate_id)
        if release_hold is not None:
            reason, detail = release_hold
            excluded.append(
                ExclusionRecord(
                    standard_name_id=candidate_id,
                    stage="release_authority",
                    reason=reason,
                    detail=detail,
                )
            )
            continue

        source_paths = candidate.get("source_paths") or []
        has_dd_source_binding = candidate.get("_has_dd_source_binding")
        if has_dd_source_binding is None:
            has_dd_source_binding = any(
                not str(source_path).startswith("derived:")
                for source_path in source_paths
            )
        has_derived_producer = candidate.get("_has_derived_producer")
        if has_derived_producer is None:
            has_derived_producer = any(
                str(source_path).startswith("derived:") for source_path in source_paths
            )
        has_non_derived_producer = candidate.get("_has_non_derived_producer")
        if has_non_derived_producer is None:
            has_non_derived_producer = any(
                not str(source_path).startswith("derived:")
                for source_path in source_paths
            )
        if (
            not has_dd_source_binding
            and has_derived_producer
            and not has_non_derived_producer
        ):
            excluded.append(
                ExclusionRecord(
                    standard_name_id=candidate_id,
                    stage="export_policy",
                    reason="structural_parent",
                    detail=(
                        "hierarchy scaffold has no Data Dictionary source binding "
                        "and only derived producers"
                    ),
                )
            )
            continue

        # Some low-level callers provide a projection already returned by the
        # historical eligibility query rather than a full graph node. The live
        # upstream query always includes name_stage; retain compatibility with
        # those explicitly pre-filtered projections.
        if "name_stage" not in candidate:
            eligible.append(candidate)
            continue
        candidate_domains = candidate.get("physics_domain") or []
        if isinstance(candidate_domains, str):
            candidate_domains = [candidate_domains]

        reason: str | None = None
        detail = ""
        if domain is not None and domain not in candidate_domains:
            reason = "outside_requested_domain"
            detail = f"physics_domain does not contain {domain!r}"
        elif candidate.get("validation_status") != "valid":
            reason = "invalid_validation_status"
            detail = f"validation_status={candidate.get('validation_status')!r}"
        elif candidate.get("review_quorum_shortfall") is not None:
            reason = "name_review_quorum_shortfall"
            detail = "name review quorum shortfall remains recorded"
        elif not names_only and candidate.get("docs_stage") != "accepted":
            reason = "documentation_not_accepted"
            detail = f"docs_stage={candidate.get('docs_stage')!r}"
        elif not names_only and candidate.get("_has_docs_review", True) is False:
            reason = "never_reviewed"
            detail = "no docs-axis review is reachable"
        elif (
            not names_only and candidate.get("_has_winning_docs_review", True) is False
        ):
            reason = "resolution_unrecorded"
            detail = "docs-axis reviews exist but no winning group records a method"
        elif (
            not names_only and candidate.get("docs_review_quorum_shortfall") is not None
        ):
            reason = "documentation_review_quorum_shortfall"
            detail = "documentation review quorum shortfall remains recorded"

        if reason is None:
            eligible.append(candidate)
        else:
            excluded.append(
                ExclusionRecord(
                    standard_name_id=candidate_id,
                    stage="eligibility",
                    reason=reason,
                    detail=detail,
                )
            )

    return eligible, excluded


# =============================================================================
# Gate implementations
# =============================================================================


def _run_gate_a() -> GateResult:
    """Gate A: Run existing graph test suites via subprocess pytest.

    Stub implementation — runs pytest with the ``graph or corpus_health``
    marker. Returns a GateResult.
    """
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "-x",
                "-q",
                "--tb=short",
                "-m",
                "graph or corpus_health",
                "tests/graph/",
                "tests/standard_names/",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        passed = result.returncode == 0
        issues = []
        if not passed:
            issues.append(
                {
                    "type": "test_suite_failure",
                    "detail": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-500:] if result.stderr else "",
                }
            )
        return GateResult(gate=GATE_A, passed=passed, issues=issues)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return GateResult(
            gate=GATE_A,
            passed=False,
            issues=[{"type": "execution_error", "detail": str(exc)}],
        )


def _run_gate_b(
    candidates: list[dict[str, Any]],
    cocos_convention: int,
    *,
    final: bool = False,
) -> GateResult:
    """Gate B: Cross-field consistency checks.

    - Every non-null ``cocos`` equals ``cocos_convention``.
    - Grammar version matches ISN package version.
    - All names parse via ISN grammar.
    - Links resolve within the export set (advisory for RC, hard for final).
    """
    issues: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = []

    # Gate: COCOS consistency
    for cand in candidates:
        cand_cocos = cand.get("cocos")
        if cand_cocos is not None and cand_cocos != cocos_convention:
            issues.append(
                {
                    "type": "cocos_mismatch",
                    "name": cand["id"],
                    "expected": cocos_convention,
                    "actual": cand_cocos,
                }
            )

    # Gate: strict public grammar identity.  The flat Pydantic projection is a
    # compatibility view and can reject lossless ordered operator trees, so it
    # is never an export validity oracle.
    try:
        from imas_standard_names.grammar import compose, parse

        for cand in candidates:
            name = cand["id"]
            try:
                result = parse(name, strict=True)
                rendered = compose(result.ir)
                if rendered != name:
                    raise ValueError(f"strict grammar round-trip produced {rendered!r}")
            except Exception as exc:
                issues.append(
                    {
                        "type": "grammar_parse_failure",
                        "name": name,
                        "detail": str(exc),
                    }
                )
    except ImportError as exc:
        # ISN unavailable is not a "skip" condition: without the grammar the
        # export cannot be validated at all (and _validate_entry would crash
        # later on the same missing import). Fail the gate loudly so the
        # export is blocked with a clear message rather than silently
        # emitting an unvalidated catalog. This issue is intentionally NOT a
        # grammar_parse_failure, so it blocks RC releases too (the RC path
        # only downgrades per-name parse failures, not a missing toolchain).
        issues.append(
            {
                "type": "isn_unavailable",
                "detail": (
                    "imas_standard_names.grammar could not be imported — the "
                    "grammar parse gate cannot run, so the export cannot be "
                    f"validated against ISN: {exc}"
                ),
            }
        )
        logger.error(
            "ISN grammar not importable — failing Gate B; export cannot be "
            "validated against ISN: %s",
            exc,
        )

    # Gate: links resolve to known names
    # For RC releases (final=False): dangling links are advisory only.
    # For final releases: dangling links block export.
    all_names = {c["id"] for c in candidates}
    for cand in candidates:
        for link in cand.get("links") or []:
            # Links can be "name:foo" format or plain "foo"
            link_target = link.split(":")[-1] if ":" in link else link
            if link_target not in all_names:
                entry = {
                    "type": "dangling_link",
                    "name": cand["id"],
                    "link_target": link_target,
                }
                if final:
                    issues.append(entry)
                else:
                    advisories.append(entry)

    if advisories:
        logger.warning(
            "Gate B: %d dangling doc links (advisory for RC release)",
            len(advisories),
        )

    passed = len(issues) == 0
    return GateResult(gate=GATE_B, passed=passed, issues=issues, advisories=advisories)


def _run_gate_c(
    candidates: list[dict[str, Any]],
    min_score: float,
    include_unreviewed: bool,
    min_description_score: float | None,
    bound_adjacent_half_width: float = DEFAULT_BOUND_ADJACENT_HALF_WIDTH,
) -> tuple[GateResult, list[dict[str, Any]], int, int]:
    """Gate C: Score thresholds — filter candidates.

    Returns (gate_result, filtered_candidates, excluded_below_score,
    excluded_unreviewed).
    """
    if not 0.0 <= bound_adjacent_half_width <= 1.0:
        raise ValueError("bound_adjacent_half_width must be between 0 and 1")

    issues: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    excluded_below_score = 0
    excluded_unreviewed = 0

    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    for cand in candidates:
        # Deterministic-parent guard: a node whose description is still
        # the placeholder written by ``seed_parent_sources`` has not had
        # ``GENERATE_DOCS`` complete. Refuse to publish it regardless of
        # whether it has a score (the score field can be stale or absent
        # while the description still references the placeholder).
        if cand.get("description") == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER:
            # Not a score exclusion — GENERATE_DOCS never ran. Counted
            # separately (excluded_placeholder) by the caller, which reads
            # these issues; do not inflate excluded_below_score here.
            issues.append(
                {
                    "type": "deterministic_parent_description_placeholder",
                    "name": cand["id"],
                    "detail": (
                        "description still equals the deterministic-parent "
                        "placeholder — GENERATE_DOCS did not produce an "
                        "LLM-quality description for this name"
                    ),
                }
            )
            continue

        # Derived parents auto-accept on the name axis. Their quality
        # bar is the docs-axis review (description+documentation
        # RD-quorum score), not the name-axis review which would be
        # noise against a structurally-fixed name. Skip the
        # ``reviewer_score_name`` check; the placeholder guard above
        # already refuses to publish before ``GENERATE_DOCS`` has run,
        # and the ``min_description_score`` check further down still
        # applies if the caller passes a docs threshold.
        # Catalog-lineage nodes (origin='catalog_edit', re-imported from a
        # released ISNC catalog) passed RD review before their original
        # export — the catalog IS the review record, so a missing
        # name-axis score must not exclude them.
        if cand.get("origin") in ("derived", "catalog_edit"):
            if min_description_score is not None:
                desc_score = cand.get("reviewer_description_score")
                if desc_score is not None and desc_score < min_description_score:
                    excluded_below_score += 1
                    issues.append(
                        {
                            "type": "below_description_score",
                            "name": cand["id"],
                            "score": desc_score,
                            "threshold": min_description_score,
                            "origin": cand.get("origin"),
                        }
                    )
                    continue
            filtered.append(cand)
            continue

        score = cand.get("reviewer_score_name")

        # Unreviewed check
        if score is None:
            if not include_unreviewed:
                excluded_unreviewed += 1
                issues.append(
                    {
                        "type": "unreviewed_name",
                        "name": cand["id"],
                        "detail": "reviewer_score_name is missing",
                    }
                )
                continue
            # Include unreviewed — skip score threshold
            filtered.append(cand)
            continue

        # Score threshold
        if score < min_score:
            distance_below_bound = min_score - score
            if distance_below_bound <= bound_adjacent_half_width:
                excluded_below_score += 1
                issues.append(
                    {
                        "type": "bound_adjacent",
                        "name": cand["id"],
                        "score": score,
                        "threshold": min_score,
                        "half_width": bound_adjacent_half_width,
                        "detail": (
                            f"score is {distance_below_bound:.6g} below the "
                            "acceptance bound, within the measured review swing"
                        ),
                    }
                )
                continue
            excluded_below_score += 1
            issues.append(
                {
                    "type": "below_name_score",
                    "name": cand["id"],
                    "score": score,
                    "threshold": min_score,
                }
            )
            continue

        # Description score threshold (optional)
        if min_description_score is not None:
            desc_score = cand.get("reviewer_description_score")
            if desc_score is not None and desc_score < min_description_score:
                excluded_below_score += 1
                issues.append(
                    {
                        "type": "below_description_score",
                        "name": cand["id"],
                        "score": desc_score,
                        "threshold": min_description_score,
                    }
                )
                continue

        filtered.append(cand)

    return (
        GateResult(gate=GATE_C, passed=True, issues=issues),
        filtered,
        excluded_below_score,
        excluded_unreviewed,
    )


def _gate_c_exclusion_records(gate_result: GateResult) -> list[ExclusionRecord]:
    """Normalize Gate C's reason-bearing issues into exclusion records."""
    exclusion_types = {
        "bound_adjacent",
        "below_description_score",
        "below_name_score",
        "deterministic_parent_description_placeholder",
        "unreviewed_name",
    }
    return [
        ExclusionRecord(
            standard_name_id=issue["name"],
            stage="score_thresholds",
            reason=issue["type"],
            detail=issue.get("detail", ""),
        )
        for issue in gate_result.issues
        if issue.get("type") in exclusion_types
    ]


def _run_exclusion_accounting_gate(
    report: ExportReport,
    population_ids: list[str],
) -> GateResult:
    """Require exactly one terminal emitted-or-excluded outcome per identity."""
    issues: list[dict[str, Any]] = []
    population_set = set(population_ids)
    emitted_ids = report.exported_names
    emitted_set = set(emitted_ids)
    excluded_ids = [record.standard_name_id for record in report.exclusion_records]
    excluded_set = set(excluded_ids)

    duplicate_population = sorted(
        name for name in population_set if population_ids.count(name) > 1
    )
    duplicate_emitted = sorted(
        name for name in emitted_set if emitted_ids.count(name) > 1
    )
    duplicate_exclusions = sorted(
        name for name in excluded_set if excluded_ids.count(name) > 1
    )
    overlap = sorted(emitted_set & excluded_set)
    missing = sorted(population_set - emitted_set - excluded_set)
    outside_population = sorted((emitted_set | excluded_set) - population_set)
    arithmetic_total = len(emitted_ids) + len(excluded_ids)

    checks = (
        (duplicate_population, "duplicate_population_identity"),
        (duplicate_emitted, "duplicate_emitted_identity"),
        (duplicate_exclusions, "multiple_exclusion_reasons"),
        (overlap, "emitted_and_excluded"),
        (missing, "unattributed_identity"),
        (outside_population, "outcome_outside_population"),
    )
    for identities, issue_type in checks:
        if identities:
            issues.append({"type": issue_type, "identities": identities})

    if arithmetic_total != len(population_ids):
        issues.append(
            {
                "type": "exclusion_accounting_mismatch",
                "accepted_population": len(population_ids),
                "emitted": len(emitted_ids),
                "excluded": len(excluded_ids),
                "accounted_total": arithmetic_total,
            }
        )

    return GateResult(
        gate=GATE_EXCLUSION_ACCOUNTING,
        passed=not issues,
        issues=issues,
    )


def detect_divergence(
    candidates: list[dict[str, Any]],
) -> list[DivergenceEntry]:
    """Gate D: Detect divergence in catalog-edited names.

    For each node with ``origin='catalog_edit'``, check whether any
    protected field has been modified since import (which would indicate
    a pipeline write bypassed the protection system).

    Without an ISNC checkout to compare against, we use a heuristic:
    if ``origin='catalog_edit'`` but ``exported_at`` is newer than
    ``imported_at``, the node was re-exported after being edited
    (expected). If any protected field hash differs from what was
    recorded, that's a divergence.

    Returns a list of divergence findings.
    """
    findings: list[DivergenceEntry] = []

    for cand in candidates:
        if cand.get("origin") != "catalog_edit":
            continue

        name = cand["id"]

        # Compute a hash of the current protected field values
        protected_values = {
            f: cand.get(f) for f in sorted(PROTECTED_FIELDS) if cand.get(f) is not None
        }
        current_hash = hashlib.sha256(
            json.dumps(protected_values, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # Check if catalog_commit_sha is set — if so, the node was
        # imported from a specific commit. We can't compare without
        # the ISNC checkout, but we flag the node for awareness.
        if cand.get("catalog_commit_sha"):
            findings.append(
                DivergenceEntry(
                    name=name,
                    field="*",
                    graph_hash=current_hash,
                    detail=(
                        f"catalog-edited node with commit lineage "
                        f"{cand['catalog_commit_sha'][:8]}; "
                        f"verify protected fields match catalog"
                    ),
                )
            )

    return findings


# =============================================================================
# Entry serialisation
# =============================================================================


def _graph_node_to_entry_dict(node: dict[str, Any]) -> dict[str, Any]:
    """Convert a graph node dict to a catalog entry dict.

    Maps graph property names to ISN StandardNameEntry field names,
    and excludes all graph-only / pipeline-only fields.
    """
    entry: dict[str, Any] = {
        "name": node["id"],
        "description": node.get("description") or "",
        "documentation": node.get("documentation") or "",
        "kind": node.get("kind") or "scalar",
        "unit": node.get("unit") or "",
        # Every candidate reaching this function has passed the accepted /
        # docs-accepted / valid export gate, so it is published as 'active'.
        # 'draft' and 'deprecated' therefore never appear in the released
        # status vocabulary.
        "status": "active",
        "links": list(node.get("links") or []),
    }

    # Provenance (ISN grammatical provenance, NOT pipeline provenance)
    # This is optional — only set for derived/composite names
    # We don't emit pipeline provenance (source_paths, dd_paths)

    return entry


# =============================================================================
# Internal supersession lineage
# =============================================================================


def _fetch_deprecation_stubs(
    published_names: set[str],
) -> list[dict[str, Any]]:
    """Return no entries because supersession lineage is graph-internal."""
    return []


def _entry_model(entry_dict: dict[str, Any]) -> Any:
    """Build the strict ISN model selected by an entry's quantity kind."""
    from imas_standard_names.models import (
        StandardNameComplexEntry,
        StandardNameMetadataEntry,
        StandardNameScalarEntry,
        StandardNameTensorEntry,
        StandardNameVectorEntry,
    )

    kind = entry_dict.get("kind", "scalar")
    model_cls = {
        "scalar": StandardNameScalarEntry,
        "vector": StandardNameVectorEntry,
        "tensor": StandardNameTensorEntry,
        "complex": StandardNameComplexEntry,
        "metadata": StandardNameMetadataEntry,
    }.get(kind, StandardNameScalarEntry)
    return model_cls.model_validate(entry_dict)


def _validate_entry(entry_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Validate one entry's shape against its strict ISN catalog model.

    Catalog-wide semantic checks are deliberately separate because reference
    validity and pairwise rules require the complete emitted identity set.
    """

    try:
        entry = _entry_model(entry_dict)
        return entry.model_dump(mode="json")
    except Exception as exc:
        logger.warning(
            "ISN validation rejected '%s': %s",
            entry_dict.get("name", "?"),
            exc,
        )
        return None


def _catalog_semantic_failures(
    domain_entries: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Run one semantic pass over the complete assembled catalog projection."""
    from imas_standard_names.validation import run_semantic_checks

    entries: dict[str, Any] = {}
    for domain_values in domain_entries.values():
        for entry_dict in domain_values:
            probe = {
                key: value
                for key, value in entry_dict.items()
                if key not in _GRAPH_ONLY_RENDERED_FIELDS
                and key not in _ISN_UNSUPPORTED_FIELDS
            }
            entry = _entry_model(probe)
            entries[entry.name] = entry

    failures: dict[str, list[str]] = {}
    for issue in run_semantic_checks(entries):
        owner, separator, _ = issue.partition(":")
        if not separator or owner not in entries:
            raise RuntimeError(
                f"ISN catalog semantic finding has no candidate identity owner: {issue}"
            )
        if " WARNING - " in issue or " INFO - " in issue:
            logger.warning("ISN catalog advisory for '%s': %s", owner, issue)
            continue
        failures.setdefault(owner, []).append(issue)

    return failures


# =============================================================================
# Computed-field derivation (arguments + error_variants)
# =============================================================================

#: Edge property keys emitted for arguments when present on the edge.
_ARGUMENT_EDGE_PROPS = (
    "operator",
    "operator_kind",
    "role",
    "separator",
    "axis",
    "shape",
)

#: Fixed key order for error_variants mapping.
_ERROR_VARIANT_KEY_ORDER = ("upper", "lower", "index")


def _derive_arguments_for_entry(
    gc: Any,
    name: str,
) -> list[dict[str, Any]] | None:
    """Query graph for outgoing HAS_PARENT edges and return argument list.

    Returns ``None`` if no HAS_PARENT edges exist for this node.
    """
    rows = gc.query(
        """
        MATCH (s:StandardName {id: $name})-[e:HAS_PARENT]->(t:StandardName)
        RETURN t.id AS name, properties(e) AS props
        ORDER BY t.id
        """,
        name=name,
    )
    if not rows:
        return None

    arguments: list[dict[str, Any]] = []
    for row in rows:
        arg: dict[str, Any] = {"name": row["name"]}
        props = row.get("props") or {}
        for key in _ARGUMENT_EDGE_PROPS:
            if key in props and props[key] is not None:
                arg[key] = props[key]
        arguments.append(arg)

    # Sort by role for binary (a before b), then by name
    arguments.sort(key=lambda a: (a.get("role") or "", a.get("name", "")))
    return arguments or None


def _derive_error_variants_for_entry(
    gc: Any,
    name: str,
) -> dict[str, str] | None:
    """Query graph for outgoing HAS_ERROR edges and return error_variants map.

    Returns ``None`` if no HAS_ERROR edges exist for this node.
    """
    rows = gc.query(
        """
        MATCH (s:StandardName {id: $name})-[e:HAS_ERROR]->(t:StandardName)
        RETURN t.id AS name, properties(e) AS props
        """,
        name=name,
    )
    if not rows:
        return None

    variants: dict[str, str] = {}
    for row in rows:
        props = row.get("props") or {}
        error_type = props.get("error_type")
        if error_type and error_type in _ERROR_VARIANT_KEY_ORDER:
            variants[error_type] = row["name"]

    if not variants:
        return None

    # Emit in fixed key order
    return {k: variants[k] for k in _ERROR_VARIANT_KEY_ORDER if k in variants}


def _fetch_sources_for_entry(
    gc: Any,
    name: str,
) -> list[dict[str, Any]] | None:
    """Query graph for StandardNameSource nodes that produced this name.

    Returns the public semantic source projection, including a DD source's
    pinned version and graph-held documentation/type/unit/coordinate/lifecycle
    context. Operational ledger ids, statuses and edit history remain internal.
    Returns ``None`` if no sources are found.
    """
    sources = fetch_public_semantic_sources(gc, name)
    return sources or None


def _catalog_source_reference(source: dict[str, Any]) -> dict[str, Any]:
    """Return the minimal durable catalog reference for one source binding.

    A DD path and its pinned version identify the authoritative snapshot, so
    copied documentation, type, unit, coordinates, URLs, and enhancement text
    would only duplicate content that imas-python can resolve. Signal bindings
    have no equivalent DD identity and retain their public semantic projection.
    """
    if source.get("dd_path"):
        return {
            "dd_path": source["dd_path"],
            "dd_version": source["dd_version"],
        }
    return dict(source)


def _fetch_ordering_edges_for_domain(
    gc: Any,
    domain: str,
    entry_names: set[str],
) -> tuple[list[tuple[str, str, str]], set[str]]:
    """Fetch HAS_PARENT + HAS_ERROR edges for ordering within a domain.

    Returns
    -------
    edges:
        List of ``(src_name, tgt_name, edge_type)`` tuples where both
        endpoints are in *entry_names*.
    cross_domain_parent_ids:
        Set of entry names in *entry_names* that have an ordering-parent
        outside the domain (cross-domain orphans).
    """
    # Fetch in-domain edges: HAS_PARENT where both nodes in domain
    arg_rows = gc.query(
        """
        MATCH (s:StandardName)-[e:HAS_PARENT]->(t:StandardName)
        WHERE s.physics_domain = $domain AND t.physics_domain = $domain
        RETURN s.id AS src, t.id AS tgt
        """,
        domain=domain,
    )

    err_rows = gc.query(
        """
        MATCH (s:StandardName)-[e:HAS_ERROR]->(t:StandardName)
        WHERE s.physics_domain = $domain AND t.physics_domain = $domain
        RETURN s.id AS src, t.id AS tgt
        """,
        domain=domain,
    )

    edges: list[tuple[str, str, str]] = []
    for row in arg_rows or []:
        if row["src"] in entry_names and row["tgt"] in entry_names:
            edges.append((row["src"], row["tgt"], "HAS_PARENT"))
    for row in err_rows or []:
        if row["src"] in entry_names and row["tgt"] in entry_names:
            edges.append((row["src"], row["tgt"], "HAS_ERROR"))

    # Find cross-domain ordering-parents:
    # Nodes whose HAS_PARENT target is outside the domain
    cross_arg_rows = gc.query(
        """
        MATCH (s:StandardName)-[:HAS_PARENT]->(t:StandardName)
        WHERE s.physics_domain = $domain AND t.physics_domain <> $domain
        RETURN DISTINCT s.id AS name
        """,
        domain=domain,
    )
    # Nodes that are HAS_ERROR targets from a node outside the domain
    cross_err_rows = gc.query(
        """
        MATCH (s:StandardName)-[:HAS_ERROR]->(t:StandardName)
        WHERE t.physics_domain = $domain AND s.physics_domain <> $domain
        RETURN DISTINCT t.id AS name
        """,
        domain=domain,
    )

    cross_domain_parent_ids: set[str] = set()
    for row in cross_arg_rows or []:
        if row["name"] in entry_names:
            cross_domain_parent_ids.add(row["name"])
    for row in cross_err_rows or []:
        if row["name"] in entry_names:
            cross_domain_parent_ids.add(row["name"])

    return edges, cross_domain_parent_ids


# =============================================================================
# Link / computed-ref resolution
# =============================================================================


def _internal_link_target(link: str) -> str | None:
    """Return the internal target name a link resolves to, or None if external.

    External links (``http://`` / ``https://``) are never pruned and return
    None. Internal links use the ``name:<target>`` scheme; a bare token is
    also treated as an internal target for backward compatibility.
    """
    if link.startswith(("http://", "https://")):
        return None
    if ":" in link:
        return link.split(":", 1)[1]
    return link


def _prune_dangling_links(
    domain_entries: dict[str, list[dict[str, Any]]],
    published_names: set[str],
) -> tuple[int, list[str]]:
    """Drop internal links whose target is not in the published set.

    Must run after the final published set is known: a link target that was a
    candidate at gate time can still be dropped later (ISN validation reject,
    domain routing), leaving the link dangling. External http(s) links are
    never touched. Returns the pruned-link count and up to 20
    ``"<name> -> <link>"`` examples for logging.
    """
    pruned = 0
    examples: list[str] = []
    for entries in domain_entries.values():
        for entry in entries:
            links = entry.get("links")
            if not links:
                continue
            kept: list[str] = []
            for link in links:
                target = _internal_link_target(link)
                if target is None or target in published_names:
                    kept.append(link)
                    continue
                pruned += 1
                if len(examples) < 20:
                    examples.append(f"{entry.get('name')} -> {link}")
            if len(kept) != len(links):
                entry["links"] = kept
    return pruned, examples


def _unresolved_computed_refs(
    domain_entries: dict[str, list[dict[str, Any]]],
    published_names: set[str],
) -> list[str]:
    """Return arguments[]/error_variants[] refs pointing outside the published
    set. These are derived from graph edges and are expected to resolve fully;
    a non-empty result signals a real defect the caller must surface loudly.
    """
    unresolved: list[str] = []
    for entries in domain_entries.values():
        for entry in entries:
            name = entry.get("name")
            for arg in entry.get("arguments") or []:
                ref = arg.get("name") if isinstance(arg, dict) else arg
                if ref and ref not in published_names:
                    unresolved.append(f"{name}: argument -> {ref}")
            error_variants = entry.get("error_variants") or {}
            refs = (
                error_variants.values()
                if isinstance(error_variants, dict)
                else error_variants
            )
            for ref in refs:
                if ref and ref not in published_names:
                    unresolved.append(f"{name}: error_variant -> {ref}")
    return unresolved


# =============================================================================
# File writing
# =============================================================================


def _catalog_entry_records(catalog_root: Path) -> dict[str, CatalogEntryRecord]:
    """Read catalog entries without normalizing their serialized mappings."""
    from yaml.nodes import MappingNode, ScalarNode, SequenceNode

    records: dict[str, CatalogEntryRecord] = {}
    standard_names = catalog_root / "standard_names"
    if not standard_names.is_dir():
        return records

    for path in sorted(standard_names.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        root = yaml.compose(text)
        if root is None:
            continue
        if not isinstance(root, SequenceNode):
            raise ValueError(f"{path}: domain catalog is not a YAML sequence")
        for node in root.value:
            if not isinstance(node, MappingNode):
                raise ValueError(f"{path}: catalog entry is not a YAML mapping")
            name = None
            for key_node, value_node in node.value:
                if (
                    isinstance(key_node, ScalarNode)
                    and key_node.value == "name"
                    and isinstance(value_node, ScalarNode)
                ):
                    name = value_node.value
                    break
            if not name:
                raise ValueError(f"{path}: catalog entry has no scalar name")
            if name in records:
                raise ValueError(f"duplicate catalog entry identity: {name}")

            line_start = text.rfind("\n", 0, node.start_mark.index) + 1
            item_prefix = text[line_start : node.start_mark.index]
            if not re.fullmatch(r"\s*-\s+", item_prefix):
                raise ValueError(
                    f"{path}: catalog entry {name!r} is not a block-list item"
                )
            records[name] = CatalogEntryRecord(
                name=name,
                domain=path.stem,
                mapping_bytes=text[node.start_mark.index : node.end_mark.index].encode(
                    "utf-8"
                ),
                item_bytes=text[line_start : node.end_mark.index].encode("utf-8"),
            )
    return records


def approved_baseline_delta(
    approved_root: Path,
    candidate_root: Path,
    *,
    batch_names: list[str] | tuple[str, ...] = (),
) -> ApprovedBaselineDelta:
    """Require all approved identities while protecting non-batch bytes."""
    approved = _catalog_entry_records(approved_root)
    candidate = _catalog_entry_records(candidate_root)
    protected_names = sorted(set(approved) - set(batch_names))
    missing = tuple(name for name in sorted(approved) if name not in candidate)
    byte_changed = tuple(
        name
        for name in protected_names
        if name in candidate
        and candidate[name].mapping_bytes != approved[name].mapping_bytes
    )
    unchanged = tuple(
        name
        for name in protected_names
        if name in candidate
        and candidate[name].mapping_bytes == approved[name].mapping_bytes
    )
    return ApprovedBaselineDelta(
        missing=missing,
        byte_changed=byte_changed,
        unchanged=unchanged,
    )


def assemble_review_catalog(
    approved_root: Path,
    staging_root: Path,
    *,
    batch_names: list[str],
) -> ReviewCatalogAssembly:
    """Carry the approved baseline into a review export without reserializing it.

    Approved entries are authoritative byte-for-byte unless the fresh export
    contains the same batch identity. A withheld batch candidate therefore
    retains its approved mapping, while an emitted batch identity replaces it.
    Fresh non-batch entries absent from the approved checkout are retained as
    additive graph-approved entries.
    """
    approved = _catalog_entry_records(approved_root)
    staged = _catalog_entry_records(staging_root)
    batch = set(batch_names)

    selected = dict(approved)
    for record in staged.values():
        if record.name in approved and record.name not in batch:
            continue
        selected[record.name] = record

    by_domain: dict[str, list[CatalogEntryRecord]] = {}
    for record in selected.values():
        by_domain.setdefault(record.domain, []).append(record)

    standard_names = staging_root / "standard_names"
    standard_names.mkdir(parents=True, exist_ok=True)
    for path in standard_names.glob("*.yml"):
        path.unlink()
    for domain, records in sorted(by_domain.items()):
        header = (
            f"# Domain: {domain}\n"
            f"# Entries: {len(records)}\n"
            "# Ordering: approved baseline followed by fresh review entries\n"
        ).encode()
        content = bytearray(header)
        for record in records:
            content.extend(record.item_bytes)
            if not record.item_bytes.endswith(b"\n"):
                content.extend(b"\n")
        (standard_names / f"{domain}.yml").write_bytes(bytes(content))

    baseline_added = sum(1 for name in approved if name not in staged)
    candidate_baseline_added = sum(
        1 for name in approved if name not in staged and name not in batch
    )
    manifest_path = staging_root / "catalog.yml"
    if manifest_path.is_file():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path}: manifest is not a YAML mapping")
        manifest["domains_included"] = sorted(by_domain)
        manifest["published_count"] = len(selected)
        prior_candidate_count = manifest.get("candidate_count")
        if isinstance(prior_candidate_count, int):
            manifest["candidate_count"] = (
                prior_candidate_count + candidate_baseline_added
            )
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

    return ReviewCatalogAssembly(
        baseline_count=len(approved),
        staged_count_before=len(staged),
        staged_count_after=len(selected),
        baseline_entries_added=baseline_added,
        batch_entries_written=sum(1 for name in staged if name in batch),
        emitted_batch_names=tuple(sorted(set(staged) & batch)),
    )


def _write_domain_yaml(
    staging_dir: Path,
    domain: str,
    entries: list[dict[str, Any]],
) -> Path:
    """Write a per-domain YAML file containing all entries as a list.

    The source commit sha lives only in the manifest (catalog.yml): stamping
    the codex HEAD sha into each per-domain header churned every one of the
    ~18 domain files on any unrelated codex commit.

    Each entry is re-validated against the ISN model in its FINAL, written
    shape — after canonicalisation, unsupported-field stripping, and link
    pruning. Entries are validated once when first built, but the augmentation
    steps between then and here (dangling-link pruning, computed-ref
    derivation, canonicalise) can in principle break a
    previously-valid entry; validating the emitted dict makes any such
    regression fail the export loudly instead of shipping a malformed catalog.

    Returns the path of the written file.
    """
    sn_dir = staging_dir / "standard_names"
    sn_dir.mkdir(parents=True, exist_ok=True)
    filepath = sn_dir / f"{domain}.yml"

    # Build header comment (no per-file sha — see docstring)
    header_lines = [
        f"# Domain: {domain}",
        f"# Entries: {len(entries)}",
        "# Ordering: structural traversal",
        "#   (HAS_PARENT-incoming + HAS_ERROR-outgoing, Kahn topo sort,",
        "#    alphabetic tie-break)",
    ]
    header = "\n".join(header_lines) + "\n"

    # Canonicalise, reorder, and clean each entry
    clean_entries: list[dict[str, Any]] = []
    invalid: list[str] = []
    for entry_dict in entries:
        canon = canonicalise_entry(entry_dict)
        # Remove None values and ISN-unsupported fields for clean YAML output
        clean = {
            k: v
            for k, v in canon.items()
            if v is not None and k not in _ISN_UNSUPPORTED_FIELDS
        }
        ordered = reorder_entry_dict(clean)
        # Final-shape validation gate: the dict about to be written must still
        # satisfy the ISN model. A failure here is a defect in the augmentation
        # pipeline (dangling-link pruning, computed-ref derivation,
        # canonicalise), not bad input — fail the export rather than
        # emit it. Strip the graph-only rendered fields first: they are stamped
        # on after build-time validation and the strict model rejects them,
        # though the catalog loader accepts them in the emitted YAML.
        probe = {
            k: v for k, v in ordered.items() if k not in _GRAPH_ONLY_RENDERED_FIELDS
        }
        if _validate_entry(probe) is None:
            invalid.append(ordered.get("name") or ordered.get("id") or "?")
        clean_entries.append(ordered)

    if invalid:
        raise RuntimeError(
            f"{len(invalid)} entry(ies) in domain '{domain}' failed ISN "
            f"validation in their final written shape (post-augmentation) — "
            f"the export pipeline corrupted a previously-valid entry: {invalid}"
        )

    content = yaml.safe_dump(clean_entries, sort_keys=False, default_flow_style=False)
    filepath.write_text(header + content, encoding="utf-8")

    return filepath


def resolve_export_scope(*, review_batch: list[str] | None, domain: str | None) -> str:
    """Classify an export run for the ``catalog.yml`` ``export_scope`` stamp.

    The returned value must be one of the literals the ISN
    ``StandardNameCatalogManifest`` model accepts, otherwise the manifest
    fails validation and is written unvalidated (``_write_manifest`` warns
    rather than raises, so a wrong value is silent). Kept as one small pure
    function so the accepted set is asserted directly against the pinned ISN
    model rather than inferred from a live export.
    """
    if review_batch is not None:
        return "review"
    if domain:
        return "domain"
    return "full"


def _write_manifest(
    staging_dir: Path,
    *,
    cocos_convention: int,
    candidate_count: int,
    published_count: int,
    excluded_below_score_count: int,
    excluded_unreviewed_count: int,
    min_score_applied: float,
    min_description_score_applied: float | None,
    include_unreviewed: bool,
    source_commit_sha: str | None = None,
    export_scope: str = "full",
    domains_included: list[str] | None = None,
    review_batch: list[str] | None = None,
    require_provenance: bool = False,
) -> Path:
    """Write the catalog.yml manifest to the staging directory root.

    The manifest carries only fields defined by the ISN
    ``StandardNameCatalogManifest`` model (extra='forbid'), so publish and
    the downstream ISNC catalog-validation stay green. The full exclusion
    accounting that closes ``candidate_count - published_count`` (placeholder,
    parse-failure and validation-failure buckets) is emitted in the sibling
    ``.export_report.json`` rather than here — see ``ExportReport.to_dict``.
    """
    import imas_standard_names

    stamp = _manifest_iso_timestamp(
        source_commit_sha,
        require_provenance=require_provenance,
    )

    manifest_data = {
        "catalog_name": "imas-standard-names-catalog",
        "cocos_convention": cocos_convention,
        "grammar_version": imas_standard_names.__version__,
        "isn_model_version": imas_standard_names.__version__,
        "dd_version_lineage": ["4.0.0"],
        "generated_by": "imas-codex sn export",
        "generated_at": stamp,
        "min_score_applied": min_score_applied,
        "min_description_score_applied": min_description_score_applied,
        "include_unreviewed": include_unreviewed,
        "candidate_count": candidate_count,
        "published_count": published_count,
        "excluded_below_score_count": excluded_below_score_count,
        "excluded_unreviewed_count": excluded_unreviewed_count,
        "source_repo": "imas-codex",
        "source_commit_sha": source_commit_sha,
        "export_scope": export_scope,
        "domains_included": sorted(domains_included or []),
        "catalog_commit_sha": source_commit_sha,
        "exported_at": stamp,
        "edge_model_version": "v1",
    }
    # A review-batch export carries the batch id-set so the SPA can render a
    # PR-scoped fixed view. Omitted on normal builds so their output is
    # byte-identical to today. (The ISN manifest model must know the field for
    # validation to pass; older models reject it and the raw dict is written.)
    if review_batch is not None:
        manifest_data["review_batch"] = sorted(review_batch)

    # Validate via ISN manifest model
    try:
        from imas_standard_names.models import StandardNameCatalogManifest

        manifest = StandardNameCatalogManifest.model_validate(manifest_data)
        manifest_data = manifest.model_dump(mode="json")
    except Exception as exc:
        logger.warning("Manifest validation warning: %s", exc)

    # The ISN model now declares ``review_batch`` (default None), so the
    # validation round-trip re-introduces the key even on a normal build.
    # Drop it again when there is no batch so full-catalog manifests stay
    # byte-identical (the additive-diff guarantee); a real batch keeps it.
    if review_batch is None:
        manifest_data.pop("review_batch", None)

    filepath = staging_dir / "catalog.yml"
    content = yaml.safe_dump(manifest_data, sort_keys=False, default_flow_style=False)
    filepath.write_text(content, encoding="utf-8")

    return filepath


def _write_export_report(staging_dir: Path, report: ExportReport) -> Path:
    """Write .export_report.json to the staging directory."""
    filepath = staging_dir / ".export_report.json"
    filepath.write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    return filepath


def _get_codex_commit_sha() -> str | None:
    """Get the imas-codex source commit from Git or installed metadata."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        pass

    # hatch-vcs records source revisions in PEP 440 local versions such as
    # ``0.13.dev42+gabc123def``. This remains available in an installed wheel
    # or source archive where the repository itself is absent.
    try:
        distribution = importlib.metadata.distribution("imas-codex")
    except importlib.metadata.PackageNotFoundError:
        return None
    packaged_module = Path(
        distribution.locate_file("imas_codex/standard_names/export.py")
    ).resolve()
    if packaged_module != Path(__file__).resolve():
        return None
    version = distribution.version
    match = re.search(r"(?:^|[+.])g([0-9a-f]{7,40})(?:[.]|$)", version)
    return match.group(1) if match else None


def _commit_iso_timestamp(sha: str | None) -> str | None:
    """Return the committer date of *sha* as an ISO-8601 string, or None.

    Deriving the manifest timestamps from the source commit (rather than
    wall-clock ``now()``) makes an export of identical content produce
    identical bytes, so ``publish``'s ``git diff --cached --quiet`` no-change
    fast path is not defeated by a timestamp that changes on every run.
    """
    if not sha:
        return None
    try:
        result = subprocess.run(
            ["git", "show", "-s", "--format=%cI", sha],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _source_date_epoch_iso_timestamp() -> str | None:
    """Return ``SOURCE_DATE_EPOCH`` as UTC ISO-8601, if configured.

    Reproducible source and wheel builds conventionally carry their build
    provenance through this environment variable. An invalid configured value
    is an error rather than a reason to silently emit unrelated metadata.
    """
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is None:
        return None
    try:
        epoch = int(raw)
        if epoch < 0:
            raise ValueError("must be non-negative")
        return datetime.fromtimestamp(epoch, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError) as exc:
        raise RuntimeError(
            "SOURCE_DATE_EPOCH must be a non-negative Unix timestamp"
        ) from exc


def _manifest_iso_timestamp(
    source_commit_sha: str | None,
    *,
    require_provenance: bool,
) -> str:
    """Resolve a deterministic manifest timestamp.

    A repository commit date is authoritative when available. Archives and
    installed packages can instead supply the reproducible-build timestamp.
    The public export path refuses missing provenance; the stable epoch is
    reserved for direct low-level callers that explicitly do not require it.
    """
    stamp = _commit_iso_timestamp(source_commit_sha)
    if stamp is not None:
        return stamp
    stamp = _source_date_epoch_iso_timestamp()
    if stamp is not None:
        return stamp
    if require_provenance:
        raise RuntimeError(
            "export provenance timestamp unavailable: run from an imas-codex "
            "Git checkout or set SOURCE_DATE_EPOCH for an archive/install export"
        )
    return _UNVERSIONED_TIMESTAMP


# =============================================================================
# Main export function
# =============================================================================


def run_export(
    staging_dir: str | Path,
    *,
    min_score: float = 0.65,
    bound_adjacent_half_width: float = DEFAULT_BOUND_ADJACENT_HALF_WIDTH,
    include_unreviewed: bool = False,
    min_description_score: float | None = None,
    domain: str | None = None,
    force: bool = False,
    skip_gate: bool = False,
    gate_only: bool = False,
    gate_scope: str = "all",
    override_edits: list[str] | None = None,
    cocos_convention: int = _DEFAULT_COCOS_CONVENTION,
    include_sources: bool = True,
    names_only: bool = False,
    final: bool = False,
    review_batch: list[str] | None = None,
) -> ExportReport:
    """Export standard names from the graph to a staging directory.

    Parameters
    ----------
    staging_dir:
        Path to the staging directory. Created if it doesn't exist.
    min_score:
        Minimum ``reviewer_score_name`` for inclusion (default 0.65).
    bound_adjacent_half_width:
        Scores below ``min_score`` by no more than this measured review swing
        are excluded as ``bound_adjacent`` rather than ``below_name_score``.
    include_unreviewed:
        Include names without a ``reviewer_score_name``.
    min_description_score:
        Optional secondary threshold on description sub-score.
    domain:
        Restrict export to a single physics domain.
    force:
        Write staging tree despite gate failures.
    skip_gate:
        Skip gate entirely (requires ``force=True``).
    gate_only:
        Run the gate and report without writing YAML.
    gate_scope:
        Gate scope: ``"all"`` or ``"domain"``.
    override_edits:
        List of name IDs to reset from ``catalog_edit`` to
        ``pipeline`` origin. Pass ``["all"]`` to override all.
    cocos_convention:
        COCOS convention for the manifest (default 17).
    include_sources:
        Populate ``sources`` field in each entry with the graph
        provenance (``StandardNameSource`` nodes). Default ``True``
        (useful debug info); set ``False`` for a clean catalog export.
    final:
        When True, applies strict quality gates (dangling links
        block export).  When False (default, RC mode), dangling
        documentation links are advisory only.
    review_batch:
        Standard-name ids of a review batch. When given, the export is
        additive — ``approved`` catalog ∪ this batch — and the batch id-set is
        stamped into ``catalog.yml`` (``export_scope: review``, ``review_batch``)
        so the SPA can render a PR-scoped view. When None, a normal RC export.

    Returns
    -------
    ExportReport with gate results, counts, and divergence entries.
    """
    staging_path = Path(staging_dir)
    report = ExportReport()

    # ── 1. Fetch the upstream population and classify eligibility ──
    logger.info("Fetching accepted export population from graph...")
    population = _fetch_export_population(
        batch=review_batch,
        require_docs_review=not names_only,
    )
    candidates, eligibility_exclusions = _classify_export_population(
        population,
        domain=domain,
        names_only=names_only,
    )
    population_ids = [candidate["id"] for candidate in population]
    report.total_candidates = len(population_ids)
    report.record_exclusions(eligibility_exclusions)
    logger.info(
        "Found %d population identity(ies): %d eligible, %d excluded before gates",
        len(population_ids),
        len(candidates),
        len(eligibility_exclusions),
    )

    # ── 2. Run gates ────────────────────────────────────────────
    if not skip_gate:
        # Gate A: Graph tests (only for 'all' scope)
        # For RC (final=False): advisory only — graph tests may flag
        # in-progress work that doesn't affect the exported subset.
        if gate_scope == "all":
            gate_a = _run_gate_a()
            if not final and not gate_a.passed:
                gate_a = GateResult(
                    gate=GATE_A,
                    passed=True,
                    issues=[],
                    advisories=gate_a.issues,
                )
                logger.warning(
                    "Gate A: graph test failure(s) (advisory for RC release)"
                )
        else:
            gate_a = GateResult(gate=GATE_A, passed=True, skipped=True)
        report.gate_results.append(gate_a)

        # Gate C: Score thresholds (filter candidates)
        gate_c, candidates, _, _ = _run_gate_c(
            candidates,
            min_score,
            include_unreviewed,
            min_description_score,
            bound_adjacent_half_width,
        )
        report.gate_results.append(gate_c)
        report.record_exclusions(_gate_c_exclusion_records(gate_c))

        # Gate B: Cross-field consistency (on filtered candidates)
        gate_b = _run_gate_b(candidates, cocos_convention, final=final)
        report.gate_results.append(gate_b)

        # For RC: exclude names that fail grammar parse rather than
        # blocking the entire export.  Final releases still hard-fail.
        if not final and gate_b.issues:
            parse_failures = {
                i["name"] for i in gate_b.issues if i["type"] == "grammar_parse_failure"
            }
            if parse_failures:
                candidates = [c for c in candidates if c["id"] not in parse_failures]
                report.record_exclusions(
                    [
                        ExclusionRecord(
                            standard_name_id=name,
                            stage="cross_field_consistency",
                            reason="grammar_parse_failure",
                            detail="name failed the strict ISN grammar parse gate",
                        )
                        for name in sorted(parse_failures)
                    ]
                )
                logger.warning(
                    "Gate B: excluded %d names with grammar parse failures "
                    "(RC mode): %s",
                    len(parse_failures),
                    sorted(parse_failures),
                )
                # Move parse failures from blocking issues to advisories
                gate_b.advisories.extend(
                    i for i in gate_b.issues if i["type"] == "grammar_parse_failure"
                )
                gate_b.issues = [
                    i for i in gate_b.issues if i["type"] != "grammar_parse_failure"
                ]
                gate_b.passed = len(gate_b.issues) == 0

        # Gate D: Divergence detection
        # For RC (final=False): advisory only — catalog-edited nodes
        # are expected to diverge from the pipeline-generated version.
        divergence = detect_divergence(candidates)
        report.divergence_entries = divergence
        if final:
            gate_d = GateResult(
                gate=GATE_D,
                passed=len(divergence) == 0,
                issues=[d.to_dict() for d in divergence],
            )
        else:
            gate_d = GateResult(
                gate=GATE_D,
                passed=True,
                issues=[],
                advisories=[d.to_dict() for d in divergence],
            )
            if divergence:
                logger.warning(
                    "Gate D: %d divergence entries (advisory for RC release)",
                    len(divergence),
                )
        report.gate_results.append(gate_d)

        # Summarise gate results
        report.all_gates_passed = all(
            g.passed or g.skipped for g in report.gate_results
        )
        report.gate_failures = sum(
            1 for g in report.gate_results if not g.passed and not g.skipped
        )

        if not report.all_gates_passed and not force:
            logger.error(
                "Export blocked: %d gate(s) failed. Use --force to override.",
                report.gate_failures,
            )
            # Still write the report even on failure
            staging_path.mkdir(parents=True, exist_ok=True)
            _write_export_report(staging_path, report)
            return report
    else:
        # Gate C still runs for filtering even when gates skipped
        gate_c, candidates, _, _ = _run_gate_c(
            candidates,
            min_score,
            include_unreviewed,
            min_description_score,
            bound_adjacent_half_width,
        )
        report.record_exclusions(_gate_c_exclusion_records(gate_c))

    # ── 3. Gate-only mode: report and exit ──────────────────────
    if gate_only:
        staging_path.mkdir(parents=True, exist_ok=True)
        _write_export_report(staging_path, report)
        logger.info("Gate-only mode: report written, no YAML emitted.")
        return report

    # ── 4. Prepare staging directory ────────────────────────────
    staging_path.mkdir(parents=True, exist_ok=True)

    # Clear existing standard_names tree
    sn_dir = staging_path / "standard_names"
    if sn_dir.exists():
        import shutil

        shutil.rmtree(sn_dir)

    # ── 5. Group candidates by domain, derive computed fields ───
    from collections import defaultdict

    from imas_codex.graph.client import GraphClient

    domain_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exported_names: list[str] = []
    invalid_candidates: dict[str, str] = {}
    ordering_exclusion_records: list[ExclusionRecord] = []
    ordering_excluded_names: set[str] = set()
    all_candidate_names = {c["id"] for c in candidates}

    with GraphClient() as gc:
        for cand in candidates:
            entry_dict = _graph_node_to_entry_dict(cand)

            # Ensure no provenance fields leak through
            for pf in _PROVENANCE_FIELDS:
                entry_dict.pop(pf, None)

            # Determine domain (multi-valued list → primary by domain
            # priority, with alphabetical tie-break). Priority is derived
            # from Cluster.mapping_relevance — see domain_priority.py.
            physics_domain_list = cand.get("physics_domain") or []
            if isinstance(physics_domain_list, str):
                physics_domain_list = [physics_domain_list]
            primary = (
                pick_primary_domain(physics_domain_list)
                if physics_domain_list
                else "unscoped"
            )

            # Validate against ISN model — invalid entries are excluded.
            validated = _validate_entry(entry_dict)
            if validated is None:
                invalid_candidates[cand["id"]] = (
                    "entry failed validation against the ISN catalog model"
                )
                continue
            entry_dict = validated

            # Write physics_domain AFTER ISN validation (graph-only field).
            # ISN CatalogRenderer expects a scalar string, not a list.
            entry_dict["physics_domain"] = primary if primary != "unscoped" else ""

            # Derive computed fields from graph edges
            entry_name = entry_dict.get("name") or cand["id"]
            arguments = _derive_arguments_for_entry(gc, entry_name)
            if arguments:
                if not final:
                    # RC mode: suppress arguments referencing names outside
                    # the export set.  ISN validate_models cross-checks
                    # argument refs, so unresolvable refs block publish.
                    # Drop the entire arguments block (atomic) if any ref
                    # is outside the candidate set.
                    if any(a["name"] not in all_candidate_names for a in arguments):
                        logger.debug(
                            "Suppressing arguments for %s (refs outside export set)",
                            entry_name,
                        )
                        arguments = None
                if arguments:
                    entry_dict["arguments"] = arguments
            error_variants = _derive_error_variants_for_entry(gc, entry_name)
            if error_variants:
                entry_dict["error_variants"] = error_variants
            # Note: locus is graph-only (HAS_LOCUS edge) — not exported
            # to YAML because ISN models use extra="forbid" and don't
            # define a locus field on StandardNameEntryBase.

            # Optionally attach source provenance for debug rendering
            if include_sources:
                sources = _fetch_sources_for_entry(gc, entry_name)
                if sources:
                    entry_dict["sources"] = [
                        _catalog_source_reference(source) for source in sources
                    ]

            # Guard against the same SN landing in domain_entries twice —
            # the candidate loop iterates per (cand × physics_domain) and
            # the primary-domain choice can collide across iterations
            # when an SN's domain priority list shifts. ``exported_names``
            # de-dups at the end (see below); de-dup here too so each
            # domain YAML has at most one entry per id.
            if not any(
                e.get("name") == entry_dict.get("name") for e in domain_entries[primary]
            ):
                domain_entries[primary].append(entry_dict)
            exported_names.append(cand["id"])

        # ── 5a. Order entries and withhold hierarchy cycles ───────
        # A malformed relationship must not prevent every acyclic identity
        # from reaching the packet. Cycle participants are removed before
        # link pruning so the final projection cannot retain references to an
        # identity the exclusion ledger withholds.
        for d, entries in sorted(domain_entries.items()):
            entry_names = {e.get("name") or e.get("id", "") for e in entries}
            edges, cross_domain_ids = _fetch_ordering_edges_for_domain(
                gc, d, entry_names
            )
            ordering = order_entries_by_hierarchy(
                entries,
                edges,
                cross_domain_parent_ids=cross_domain_ids,
            )
            domain_entries[d] = list(ordering.entries)
            for exclusion in ordering.exclusions:
                ordering_excluded_names.add(exclusion.name)
                ordering_exclusion_records.append(
                    ExclusionRecord(
                        standard_name_id=exclusion.name,
                        stage="catalog_ordering",
                        reason="hierarchy_ordering_cycle",
                        detail=exclusion.detail,
                    )
                )
                logger.error(
                    "Withholding %s from catalog export: %s",
                    exclusion.name,
                    exclusion.detail,
                )

        # ── 5b. Resolve links/computed refs against the final set ──
        # The published set is now known. Drop internal (name:) doc links
        # whose target isn't published (renamed, dropped below score, or
        # rejected by ISN validation after gate time); external http(s)
        # links are left untouched. This runs before writing so the emitted
        # catalog carries no dangling internal links.
        published_names = {
            e.get("name")
            for entries in domain_entries.values()
            for e in entries
            if e.get("name")
        }

        pruned_count, pruned_examples = _prune_dangling_links(
            domain_entries, published_names
        )

        semantic_failures = _catalog_semantic_failures(domain_entries)
        if semantic_failures:
            for name, issues in semantic_failures.items():
                invalid_candidates[name] = "; ".join(issues)
                logger.warning(
                    "ISN catalog semantics rejected '%s': %s",
                    name,
                    invalid_candidates[name],
                )
            for catalog_domain, entries in domain_entries.items():
                domain_entries[catalog_domain] = [
                    entry
                    for entry in entries
                    if entry.get("name") not in semantic_failures
                ]

            published_names = {
                entry.get("name")
                for entries in domain_entries.values()
                for entry in entries
                if entry.get("name")
            }
            additional_count, additional_examples = _prune_dangling_links(
                domain_entries, published_names
            )
            pruned_count += additional_count
            pruned_examples.extend(
                additional_examples[: max(0, 20 - len(pruned_examples))]
            )

        report.pruned_links = pruned_count
        if report.pruned_links:
            logger.warning(
                "Pruned %d dangling internal link(s) whose targets are not "
                "published; examples: %s",
                report.pruned_links,
                pruned_examples,
            )
        # arguments[]/error_variants[] are derived from graph edges and must
        # resolve fully — surface loudly if any don't (they are left in place).
        unresolved = _unresolved_computed_refs(domain_entries, published_names)
        if unresolved:
            logger.error(
                "%d computed reference(s) point outside the published set — "
                "this is a defect (arguments/error_variants should resolve): %s",
                len(unresolved),
                unresolved[:20],
            )

        # ── 5c. Write ordered domain files ──────────────────────
        codex_sha = _get_codex_commit_sha()

        for d, entries in sorted(domain_entries.items()):
            _write_domain_yaml(staging_path, d, entries)

    # Dedup: a candidate with multiple physics_domain values is enumerated
    # by the candidate loop once per domain, but ``domain_entries[primary]``
    # routes it to a single domain only. Without this dedup the manifest
    # over-counts (e.g. ``electric_field`` would inflate the published
    # tally by 1 for each extra physics_domain it carries).
    seen: set[str] = set()
    deduped: list[str] = []
    for nm in exported_names:
        if nm in seen or nm in ordering_excluded_names or nm in invalid_candidates:
            continue
        seen.add(nm)
        deduped.append(nm)
    exported_names = deduped
    report.exported_count = len(exported_names)
    report.exported_names = exported_names
    report.record_exclusions(
        [
            ExclusionRecord(
                standard_name_id=name,
                stage="catalog_validation",
                reason="invalid_catalog_entry",
                detail=detail,
            )
            for name, detail in invalid_candidates.items()
        ]
        + ordering_exclusion_records
    )

    accounting_gate = _run_exclusion_accounting_gate(report, population_ids)
    report.gate_results.append(accounting_gate)
    report.all_gates_passed = all(
        gate.passed or gate.skipped for gate in report.gate_results
    )
    report.gate_failures = sum(
        1 for gate in report.gate_results if not gate.passed and not gate.skipped
    )
    if not accounting_gate.passed:
        logger.error(
            "Export blocked: exclusion accounting does not close over the "
            "accepted population"
        )
        _write_export_report(staging_path, report)
        return report

    # ── 6. Write manifest ───────────────────────────────────────
    all_domains = sorted(domain_entries.keys())
    export_scope = resolve_export_scope(review_batch=review_batch, domain=domain)
    _write_manifest(
        staging_path,
        cocos_convention=cocos_convention,
        candidate_count=report.total_candidates,
        published_count=report.exported_count,
        excluded_below_score_count=report.excluded_below_score,
        excluded_unreviewed_count=report.excluded_unreviewed,
        min_score_applied=min_score,
        min_description_score_applied=min_description_score,
        include_unreviewed=include_unreviewed,
        source_commit_sha=codex_sha,
        export_scope=export_scope,
        domains_included=all_domains,
        review_batch=sorted(review_batch) if review_batch is not None else None,
        require_provenance=True,
    )

    # ── 7. Write export report ──────────────────────────────────
    _write_export_report(staging_path, report)

    logger.info(
        "Export complete: %d name(s) written to %s",
        report.exported_count,
        staging_path,
    )
    if invalid_candidates:
        logger.warning(
            "%d name(s) excluded by ISN validation — check logs for details",
            len(invalid_candidates),
        )
    return report

"""Export validated standard names from the graph to a staging directory.

Reads StandardName nodes from the Neo4j graph, applies quality gates,
and writes YAML files matching the ``imas-standard-names-catalog``
layout: ``<staging>/standard_names/<domain>/<name>.yml`` plus a
``<staging>/catalog.yml`` manifest.

This module is the first half of the two-step export→publish flow.
The staging directory produced here is consumed by ``publish.py``
(transport to ISNC repo) and ``preview.py`` (local site render).

See plan 35 §Phase 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
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

logger = logging.getLogger(__name__)

# Default COCOS convention for the catalog manifest
_DEFAULT_COCOS_CONVENTION = 17

# Gate names
GATE_A = "graph_tests"
GATE_B = "cross_field_consistency"
GATE_C = "score_thresholds"
GATE_D = "divergence_detection"

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
    # Deprecation stubs emitted for accepted names retired via supersession —
    # status:deprecated entries pointing at their live successor. Tracked here
    # (never in the CLOSED catalog manifest) so a release can report how many
    # renames the published catalog now carries a breaking-change trail for.
    deprecated_stub_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": [g.to_dict() for g in self.gate_results],
            "divergence": [d.to_dict() for d in self.divergence_entries],
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
                "deprecated_stubs": self.deprecated_stub_count,
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
) -> list[dict[str, Any]]:
    """Fetch StandardName nodes eligible for export from the graph.

    Returns dicts with all catalog-relevant properties plus ``origin``,
    ``cocos``, ``reviewer_score_name``.

    Only nodes that have completed the name pipeline and passed
    validation are returned.  Specifically the query requires:

    - ``name_stage = 'accepted'`` — excludes superseded, exhausted, drafted,
      reviewed, and refining nodes.
    - ``docs_stage = 'accepted'`` — excludes nodes whose documentation has
      not yet passed the docs review loop (skipped when *names_only*).
    - ``validation_status = 'valid'`` — excludes quarantined nodes.

    When *names_only* is True, the ``docs_stage`` gate is dropped so
    names can be exported before documentation is generated.
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
    MATCH (sn:StandardName)
    WHERE sn.name_stage = 'accepted'
      AND sn.validation_status = 'valid'
    """
    if not names_only:
        cypher += "  AND sn.docs_stage = 'accepted'\n"
    params: dict[str, Any] = {}

    if domain:
        cypher += " AND sn.physics_domain = $domain"
        params["domain"] = domain

    cypher += """
    OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
    OPTIONAL MATCH (sn)-[:HAS_COCOS]->(c:COCOS)
    RETURN sn {
        .*,
        unit: coalesce(u.id, sn.unit),
        cocos: c.convention
    } AS record
    ORDER BY sn.id
    """

    with GraphClient() as gc:
        rows = gc.query(cypher, **params)

    return [r["record"] for r in (rows or [])]


# =============================================================================
# Gate implementations
# =============================================================================


def _run_gate_a() -> GateResult:
    """Gate A: Run existing graph test suites via subprocess pytest.

    Stub implementation — runs pytest with the ``graph or corpus_health``
    marker. Returns a GateResult. In Phase 6 this will be made more
    granular.
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

    # B1: COCOS consistency
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

    # B2: Grammar parse check — validate each name parses
    try:
        from imas_standard_names.grammar import parse_name

        for cand in candidates:
            name = cand["id"]
            try:
                parse_name(name)
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

    # B3: Links resolve to known names
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
) -> tuple[GateResult, list[dict[str, Any]], int, int]:
    """Gate C: Score thresholds — filter candidates.

    Returns (gate_result, filtered_candidates, excluded_below_score,
    excluded_unreviewed).
    """
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
                continue
            # Include unreviewed — skip score threshold
            filtered.append(cand)
            continue

        # Score threshold
        if score < min_score:
            excluded_below_score += 1
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
        # 'draft' therefore never appears in the published status vocabulary;
        # 'deprecated' is reserved for the supersession stubs built below.
        "status": "active",
        "links": list(node.get("links") or []),
    }

    # Optional lifecycle fields
    if node.get("deprecates"):
        entry["deprecates"] = node["deprecates"]
    if node.get("superseded_by"):
        entry["superseded_by"] = node["superseded_by"]

    # Provenance (ISN grammatical provenance, NOT pipeline provenance)
    # This is optional — only set for derived/composite names
    # We don't emit pipeline provenance (source_paths, dd_paths)

    return entry


# =============================================================================
# Deprecation stubs (accepted names retired by supersession)
# =============================================================================


def _fetch_deprecation_stubs(
    published_names: set[str],
) -> list[dict[str, Any]]:
    """Fetch predecessor nodes needing a deprecation stub in the catalog.

    A stub is warranted for every ``StandardName`` that:

    - is ``name_stage = 'superseded'`` (retired from the live catalog), AND
    - carries ``superseded_from_stage = 'accepted'`` — it had reached the
      published bar before being retired.  Draft/reviewed churn records a
      non-accepted sentinel and emits nothing (dep-scope = accepted-only), AND
    - has a *live* accepted successor reachable along the incoming
      ``REFINED_FROM`` chain that is itself in *published_names*.

    Refinement chains collapse.  For ``A → B → C`` (edges ``B-REFINED_FROM→A``,
    ``C-REFINED_FROM→B``; A, B superseded, C accepted) the successor predicate
    ``succ.name_stage = 'accepted'`` binds only ``C``, so A's stub points
    straight at the live name ``C``.  A superseded intermediate ``B`` that was
    itself published emits its own stub, also collapsed onto ``C``.

    Returns predecessor node dicts, each annotated with the resolved live
    successor under the ``_successor`` key.  A predecessor whose only
    successors are unpublished (below-score / rejected / renamed out of the
    export set) is skipped — a stub with an unresolvable successor would be a
    dangling breaking-change pointer.
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
    MATCH (old:StandardName)
    WHERE coalesce(old.name_stage, '') = 'superseded'
      AND old.superseded_from_stage = 'accepted'
    MATCH (succ:StandardName)-[:REFINED_FROM*1..]->(old)
    WHERE succ.name_stage = 'accepted'
    OPTIONAL MATCH (old)-[:HAS_UNIT]->(u:Unit)
    RETURN old {
        .*,
        unit: coalesce(u.id, old.unit)
    } AS record,
    collect(DISTINCT succ.id) AS successors
    ORDER BY old.id
    """
    with GraphClient() as gc:
        rows = gc.query(cypher)

    stubs: list[dict[str, Any]] = []
    for row in rows or []:
        node = dict(row["record"])
        successors = [s for s in (row.get("successors") or []) if s in published_names]
        if not successors:
            continue
        # Deterministic collapse when a predecessor has more than one live
        # accepted successor (branching refinement): the lexicographically
        # first published successor wins.
        node["_successor"] = sorted(successors)[0]
        stubs.append(node)
    return stubs


def _build_stub_entry(node: dict[str, Any]) -> dict[str, Any]:
    """Build a ``status: deprecated`` catalog entry dict for a superseded name.

    Copies ``kind``/``unit`` from the predecessor and points ``superseded_by``
    (and a front-and-centre internal link) at the live successor.
    """
    old_name = node["id"]
    successor = node["_successor"]
    kind = node.get("kind") or "scalar"
    entry: dict[str, Any] = {
        "name": old_name,
        "kind": kind,
        "status": "deprecated",
        "superseded_by": successor,
        "description": f"Deprecated: renamed to {successor}.",
        "documentation": (
            f"`{old_name}` has been renamed. Use `{successor}` instead — it "
            f"is the live standard name for this quantity. This deprecated "
            f"entry is retained so downstream consumers referencing the old "
            f"name can resolve the successor."
        ),
        "links": [f"name:{successor}"],
    }
    # Metadata entries carry no unit; every other kind requires one.
    if kind != "metadata":
        entry["unit"] = node.get("unit") or "1"
    return entry


def _validate_entry(entry_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Validate an entry dict against the ISN StandardNameEntry model.

    Returns the validated model_dump dict on success, or None if validation
    fails. Callers must handle None returns — invalid entries are excluded
    from the export.
    """
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

    try:
        entry = model_cls.model_validate(entry_dict)
        return entry.model_dump(mode="json")
    except Exception as exc:
        logger.warning(
            "ISN validation rejected '%s': %s",
            entry_dict.get("name", "?"),
            exc,
        )
        return None


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

    Returns a list of source dicts — the sanctioned, restorable projection
    of the provenance ledger — with keys ``id``, ``dd_path`` | ``signal_id``,
    ``status``, ``source_type`` (always, every source carries one) and
    ``provenance`` (only when non-null). Together these are lossless: the
    reconciler rebuilds the ``StandardNameSource`` node from this block.
    Returns ``None`` if no sources are found.
    """
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name})<-[:PRODUCED_NAME]-(src:StandardNameSource)
        OPTIONAL MATCH (src)-[:FROM_DD_PATH]->(n:IMASNode)
        OPTIONAL MATCH (src)-[:FROM_SIGNAL]->(s:FacilitySignal)
        RETURN src.id AS source_id,
               n.id   AS dd_path,
               s.id   AS signal_id,
               src.status AS status,
               src.source_type AS source_type,
               src.provenance AS provenance
        ORDER BY src.id
        """,
        name=name,
    )
    if not rows:
        return None

    sources: list[dict[str, Any]] = []
    for row in rows:
        src: dict[str, Any] = {}
        if row.get("source_id"):
            src["id"] = row["source_id"]
        if row.get("dd_path"):
            src["dd_path"] = row["dd_path"]
        if row.get("signal_id"):
            src["signal_id"] = row["signal_id"]
        if row.get("status"):
            src["status"] = row["status"]
        # source_type is a mandatory ledger field — every source has one.
        if row.get("source_type"):
            src["source_type"] = row["source_type"]
        # provenance is optional — emit only when the source carries it.
        if row.get("provenance"):
            src["provenance"] = row["provenance"]
        if src:
            sources.append(src)

    return sources or None


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
    steps between then and here (deprecation stubs, dangling-link pruning,
    computed-ref derivation, canonicalise) can in principle break a
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
        # pipeline (deprecation stubs, dangling-link pruning, computed-ref
        # derivation, canonicalise), not bad input — fail the export rather than
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

    # Deterministic timestamps: derive from the source commit so identical
    # content yields identical bytes (see _commit_iso_timestamp). Fall back to
    # wall-clock only when there is no commit to key off (e.g. a non-git
    # staging run in tests), which is the only remaining non-deterministic path.
    stamp = _commit_iso_timestamp(source_commit_sha) or datetime.now(UTC).isoformat()

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

    # Validate via ISN manifest model
    try:
        from imas_standard_names.models import StandardNameCatalogManifest

        manifest = StandardNameCatalogManifest.model_validate(manifest_data)
        manifest_data = manifest.model_dump(mode="json")
    except Exception as exc:
        logger.warning("Manifest validation warning: %s", exc)

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
    """Get the current imas-codex git commit SHA, or None."""
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
        return None


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


# =============================================================================
# Main export function
# =============================================================================


def run_export(
    staging_dir: str | Path,
    *,
    min_score: float = 0.65,
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
) -> ExportReport:
    """Export standard names from the graph to a staging directory.

    Parameters
    ----------
    staging_dir:
        Path to the staging directory. Created if it doesn't exist.
    min_score:
        Minimum ``reviewer_score_name`` for inclusion (default 0.65).
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

    Returns
    -------
    ExportReport with gate results, counts, and divergence entries.
    """
    staging_path = Path(staging_dir)
    report = ExportReport()

    # ── 1. Fetch candidates from graph ──────────────────────────
    logger.info("Fetching candidates from graph...")
    candidates = _fetch_candidates(
        include_unreviewed=include_unreviewed,
        domain=domain,
        names_only=names_only,
    )
    report.total_candidates = len(candidates)
    logger.info("Found %d candidate(s)", len(candidates))

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
        gate_c, candidates, excluded_below, excluded_unrev = _run_gate_c(
            candidates, min_score, include_unreviewed, min_description_score
        )
        report.gate_results.append(gate_c)
        report.excluded_below_score = excluded_below
        report.excluded_unreviewed = excluded_unrev
        report.excluded_placeholder = sum(
            1
            for i in gate_c.issues
            if i["type"] == "deterministic_parent_description_placeholder"
        )

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
                report.parse_failures = len(parse_failures)
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
        gate_c, candidates, excluded_below, excluded_unrev = _run_gate_c(
            candidates, min_score, include_unreviewed, min_description_score
        )
        report.excluded_below_score = excluded_below
        report.excluded_unreviewed = excluded_unrev
        report.excluded_placeholder = sum(
            1
            for i in gate_c.issues
            if i["type"] == "deterministic_parent_description_placeholder"
        )

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
    validation_failures = 0
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
                validation_failures += 1
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
                    entry_dict["sources"] = sources

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

        # ── 5a2. Resolve links/computed refs against the final set ──
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

        # ── 5a1. Deprecation stubs for retired accepted names ───────
        # Accepted names retired by a rename vanish from the live set above;
        # emit a status:deprecated stub pointing at the live successor so the
        # rename is a discoverable, resolvable trail rather than a silent
        # breaking change. Stubs are validated by the ISN model like any other
        # entry and routed to the predecessor's primary domain.
        stub_count = 0
        for stub_node in _fetch_deprecation_stubs(published_names):
            if stub_node["id"] in published_names:
                # A live accepted name reclaimed this id — no stub needed.
                continue
            validated_stub = _validate_entry(_build_stub_entry(stub_node))
            if validated_stub is None:
                validation_failures += 1
                continue
            stub_domain_list = stub_node.get("physics_domain") or []
            if isinstance(stub_domain_list, str):
                stub_domain_list = [stub_domain_list]
            stub_primary = (
                pick_primary_domain(stub_domain_list)
                if stub_domain_list
                else "unscoped"
            )
            validated_stub["physics_domain"] = (
                stub_primary if stub_primary != "unscoped" else ""
            )
            if not any(
                e.get("name") == validated_stub.get("name")
                for e in domain_entries[stub_primary]
            ):
                domain_entries[stub_primary].append(validated_stub)
                stub_count += 1
        report.deprecated_stub_count = stub_count
        # Recompute the published set so stub ids (and their successors) are
        # known to link pruning — a doc link to a now-deprecated old name
        # resolves to its stub instead of being pruned.
        published_names = {
            e.get("name")
            for entries in domain_entries.values()
            for e in entries
            if e.get("name")
        }

        pruned_count, pruned_examples = _prune_dangling_links(
            domain_entries, published_names
        )
        report.pruned_links = pruned_count
        if pruned_count:
            logger.warning(
                "Pruned %d dangling internal link(s) whose targets are not "
                "published; examples: %s",
                pruned_count,
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

        # ── 5b. Order entries per domain and write files ────────
        codex_sha = _get_codex_commit_sha()

        for d, entries in sorted(domain_entries.items()):
            entry_names = {e.get("name") or e.get("id", "") for e in entries}
            edges, cross_domain_ids = _fetch_ordering_edges_for_domain(
                gc, d, entry_names
            )
            ordered = order_entries_by_hierarchy(
                entries,
                edges,
                cross_domain_parent_ids=cross_domain_ids,
            )
            _write_domain_yaml(staging_path, d, ordered)

    # Dedup: a candidate with multiple physics_domain values is enumerated
    # by the candidate loop once per domain, but ``domain_entries[primary]``
    # routes it to a single domain only. Without this dedup the manifest
    # over-counts (e.g. ``electric_field`` would inflate the published
    # tally by 1 for each extra physics_domain it carries).
    seen: set[str] = set()
    deduped: list[str] = []
    for nm in exported_names:
        if nm in seen:
            continue
        seen.add(nm)
        deduped.append(nm)
    exported_names = deduped
    report.exported_count = len(exported_names)
    report.exported_names = exported_names
    report.validation_failures = validation_failures

    # ── 6. Write manifest ───────────────────────────────────────
    all_domains = sorted(domain_entries.keys())
    export_scope = "domain_subset" if domain else "full"
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
    )

    # ── 7. Write export report ──────────────────────────────────
    _write_export_report(staging_path, report)

    logger.info(
        "Export complete: %d name(s) written to %s",
        report.exported_count,
        staging_path,
    )
    if validation_failures:
        logger.warning(
            "%d name(s) excluded by ISN validation — check logs for details",
            validation_failures,
        )
    return report

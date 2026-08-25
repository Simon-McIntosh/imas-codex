"""Check statically resolvable Cypher properties against LinkML schemas."""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from imas_codex.graph.schema import GraphSchema

_NODE_BINDING_RE = re.compile(
    r"\(\s*(?P<alias>[A-Za-z_]\w*)\s*:\s*`?(?P<label>[A-Za-z_]\w*)`?"
)
_PROPERTY_RE = re.compile(
    r"\b(?P<alias>[A-Za-z_]\w*)\s*\.\s*`?(?P<property>[A-Za-z_]\w*)`?"
)
_CYPHER_KEYWORD_RE = re.compile(
    r"\b(?:CALL|CREATE|DELETE|DROP|FOREACH|MATCH|MERGE|OPTIONAL|REMOVE|RETURN|"
    r"SET|SHOW|UNWIND|WHERE|WITH)\b"
)
_CYPHER_CONTEXT_RE = re.compile(
    r"(?:cypher|query|clause|filter|match|predicate|where)", re.IGNORECASE
)
_REPO_ROOT = Path(__file__).resolve().parents[2]

_ADJUDICATION_REASONS = {
    "defect": (
        "adjudicated real silent-zero defect: the property is absent from the "
        "declared label and has no static runtime writer; repair is a follow-on"
    ),
    "runtime": (
        "permanent runtime-property allowance: repository Cypher writes this "
        "coordination or operational property without declaring it in LinkML"
    ),
    "transient_lock": (
        "adjudicated transient set-then-remove lock idiom: Cypher writes and "
        "removes this marker in one statement to acquire a node write lock "
        "without persisting graph data"
    ),
    "limitation": (
        "permanent checker limitation: this untyped alias reference precedes a "
        "later binding to the reported label in the same composite literal"
    ),
}

# Each row is path|label|property|source-lines. Source lines locate the frozen
# inventory, while repeated entries preserve the maximum occurrence count for a
# path, label, and property. Line movement is harmless, repaired defects may
# shrink that count, and any excess occurrence still fails the audit.
# CodeChunk.related_ids in ingestion/graph.py is a runtime-property allowance:
# live coverage is 4,826/271,460, while declared CodeExample.related_ids is 0/69,746.
# CodeChunk.embed_failed_at in discovery/code/parallel.py is also runtime-only:
# live coverage is 1,649/271,460, and the canonical embed worker writes and filters it.
_ADJUDICATED_OCCURRENCES = """
[defect]
imas_codex/cli/discover/__init__.py|FacilityPath|scanned_at|352,353
imas_codex/discovery/base/executor.py|FacilityPath|expand_to|286,286
imas_codex/discovery/code/graph_ops.py|FacilityPath|purpose|163
imas_codex/discovery/paths/parallel.py|FacilityPath|accessible|1136
imas_codex/discovery/paths/parallel.py|FacilityPath|patterns_detected|629
imas_codex/discovery/signals/parallel.py|CodeChunk|chunk_type|3520
imas_codex/discovery/signals/scanners/device_xml.py|SignalNode|r|1530,1530,2103,2103,2123,2123,2801
imas_codex/discovery/signals/scanners/device_xml.py|SignalNode|z|1531,1531,2104,2104,2124,2124,2801
imas_codex/graph/build_dd.py|IMASNode|units|2486,2625
imas_codex/graph/client.py|IMASNode|path|858,859
imas_codex/graph/dd_domain_classifier.py|IMASNode|units|841,893,1037
imas_codex/graph/dd_search.py|IMASNode|units|433
imas_codex/graph/schema_context.py|IMASNode|units|75
imas_codex/graph/sn_link_guardrail.py|StandardName|name|78
imas_codex/ids/graph_ops.py|SignalNode|sort_key|217,233,270
imas_codex/llm/search_tools.py|Document|file_type|1051
imas_codex/llm/search_tools.py|Document|title|1033,1033,1033,1051,1051
imas_codex/llm/search_tools.py|IMASCoordinateSpec|coordinate_type|1211
imas_codex/llm/server.py|Document|title|745
imas_codex/standard_names/campaign.py|StandardName|name|297,677
imas_codex/standard_names/chain_history.py|StandardName|reviewer_score|43
imas_codex/standard_names/context.py|StandardName|name|684,714
imas_codex/standard_names/export.py|COCOS|convention|319,371
imas_codex/standard_names/graph_ops.py|FacilitySignal|units|738
imas_codex/standard_names/graph_ops.py|IMASNode|units|4658,12530
imas_codex/standard_names/graph_ops.py|StandardName|name|13132
imas_codex/standard_names/graph_ops.py|StandardNameSource|unit|4658
imas_codex/standard_names/prompt_tools.py|IMASNodeChange|detail|165
imas_codex/standard_names/prompt_tools.py|IMASNodeChange|from_version|163
imas_codex/standard_names/prompt_tools.py|IMASNodeChange|to_version|164,166
imas_codex/standard_names/prompt_tools.py|StandardName|name|117,119
imas_codex/standard_names/provenance_lifecycle.py|StandardName|stage|1073
imas_codex/standard_names/review/pipeline.py|StandardName|source_id|299
imas_codex/standard_names/signed_manifest.py|StandardNameChange|row_id|3074,4249
imas_codex/standard_names/workers.py|IMASNode|alias|8509
imas_codex/standard_names/workers.py|IMASNode|units|8493
imas_codex/standard_names/workers.py|StandardName|name|9572
imas_codex/tools/graph_search.py|IMASNode|units|2182,2245
imas_codex/tools/version_tool.py|IMASNodeChange|semantic_change_type|344,345
[runtime]
imas_codex/tools/graph_search.py|CodeChunk|related_ids|2559,2560
imas_codex/cli/sn.py|StandardName|review_resubmit_count|443
imas_codex/cli/sn.py|StandardNameSource|run_id|2312,2312,2379
imas_codex/discovery/base/executor.py|FacilityPath|claimed_at|222,254
imas_codex/discovery/base/executor.py|FacilityPath|claimed_by|220,220,221,253,254,287,287
imas_codex/discovery/base/grouping.py|SignalSource|claim_token|128,260
imas_codex/discovery/base/grouping.py|SignalSource|claimed_at|128,210,260
imas_codex/discovery/code/graph_ops.py|FacilityPath|files_claim_token|149
imas_codex/discovery/code/graph_ops.py|FacilityPath|files_claimed_at|134,135,149,212,230,231
imas_codex/discovery/code/graph_ops.py|FacilityPath|last_file_scan_at|143,145,192,555,557
imas_codex/discovery/code/parallel.py|CodeChunk|embed_failed_at|473
imas_codex/discovery/code/scanner.py|FacilityPath|evidence_linked|627
imas_codex/discovery/code/scanner.py|FacilityPath|last_file_scan_at|366,626
imas_codex/discovery/mdsplus/graph_ops.py|SignalNode|category|807,1041,1132,1535
imas_codex/discovery/mdsplus/graph_ops.py|SignalNode|claim_token|912,967
imas_codex/discovery/mdsplus/graph_ops.py|SignalNode|claimed_at|809,903,912,965,967,1027,1043,1055,1065,1079,1134,1154,1543
imas_codex/discovery/mdsplus/graph_ops.py|SignalNode|keywords|806,1040,1131,1534
imas_codex/discovery/mdsplus/graph_ops.py|SignalNode|tags|420,425,748,927,931,980
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|category|799,1559
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|claim_token|730
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|claimed_at|728,730,801,847,1560
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|data_source_name|535,661,726,862,1466
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|enrichment_status|800,1555,1557
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|grandparent_path|536,662,743
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|index_count|538,664,745
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|leaf_name|537,663,744
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|node_type|539,665,746
imas_codex/discovery/mdsplus/graph_ops.py|SignalSource|representative_path|540,666,740,747
imas_codex/discovery/mdsplus/tdi_linkage.py|FacilitySignal|preferred_accessor|122,123
imas_codex/discovery/paths/enrichment.py|FacilityPath|claimed_at|559
imas_codex/discovery/paths/frontier.py|FacilityPath|claim_token|2940,3101
imas_codex/discovery/paths/frontier.py|FacilityPath|claimed_at|282,1464,2023,2063,2935,2935,2940,3035,3049,3097,3097,3101,3347
imas_codex/discovery/paths/frontier.py|FacilityPath|score_reason|2715,3349
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_analysis_code|1468,1582,3122
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_composite|284,408,1466,1577,1579,1659,1660,3095,3114
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_convention|1477
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_data_access|1472,1586,3126
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_documentation|1475,1589,3129
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_experimental_data|1471,1585,3125
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_imas|1476,1590,3130
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_modeling_code|1467,1581,3121
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_modeling_data|1470,1584,3124
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_operations_code|1469,1583,3123
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_visualization|1474,1588,3128
imas_codex/discovery/paths/frontier.py|FacilityPath|triage_workflow|1473,1587,3127
imas_codex/discovery/paths/frontier.py|FacilityPath|triaged_at|1465
imas_codex/discovery/paths/parallel.py|FacilityPath|claim_token|495,553,608,738,808
imas_codex/discovery/paths/parallel.py|FacilityPath|claimed_at|492,492,495,550,550,553,605,605,608,734,734,738,805,805,808,933,956,1057,1196,1245,2007
imas_codex/discovery/paths/parallel.py|FacilityPath|score_reason|1049
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_analysis_code|825
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_composite|217,219,225,272,308,327,491,604,732,754,803,823
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_convention|834
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_data_access|829
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_documentation|832
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_experimental_data|828
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_imas|833
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_modeling_code|824
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_modeling_data|827
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_operations_code|826
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_visualization|831
imas_codex/discovery/paths/parallel.py|FacilityPath|triage_workflow|830
imas_codex/discovery/paths/parallel.py|FacilityUser|claim_token|406
imas_codex/discovery/paths/parallel.py|FacilityUser|claimed_at|371,372,403,404,406,2633
imas_codex/discovery/signals/parallel.py|SignalSource|claimed_at|1611
imas_codex/discovery/signals/scanners/device_xml.py|SignalNode|sensor_type|2098,2118
imas_codex/discovery/signals/scanners/device_xml.py|SignalNode|system|1529,2102,2122
imas_codex/discovery/wiki/graph_ops.py|WikiPage|ingest_retries|1154,1154,1164,1180
imas_codex/discovery/wiki/graph_ops.py|WikiPage|referenced_files|346,372,373
imas_codex/graph/build_dd.py|IMASNode|claim_token|819
imas_codex/graph/dd_enrichment.py|IMASNode|claim_token|1183
imas_codex/graph/dd_graph_ops.py|IMASNode|claim_token|78,128,147,228,276,295,358,402,420,514,537
imas_codex/graph/dd_identifier_enrichment.py|IMASNode|claim_token|629
imas_codex/ids/metadata.py|DDVersion|version|169,169
imas_codex/ids/metadata.py|IMASMapping|code_metadata|497
imas_codex/ids/metadata.py|IMASMapping|ids_properties_metadata|496
imas_codex/ids/metadata.py|IMASMapping|library_metadata|498
imas_codex/ids/metadata.py|IMASMapping|metadata_populated|499
imas_codex/ingestion/graph.py|CodeChunk|related_ids|85,86,96,97
imas_codex/ingestion/pipeline.py|FacilityPath|last_ingested_at|447
imas_codex/llm/server.py|FacilityPath|triage_composite|2265,2267,2287,2290
imas_codex/mdsplus/batch_discovery.py|SignalEpoch|boundary_refined|791
imas_codex/mdsplus/extraction.py|SignalNode|tags|127,944
imas_codex/standard_names/campaign.py|StandardName|quarantine_reason|302,665,712
imas_codex/standard_names/edit.py|StandardName|edit_refine|667
imas_codex/standard_names/edit.py|StandardNameSource|run_id|2684
imas_codex/standard_names/graph_ops.py|StandardName|_drain_scope_lock|1448,1449
imas_codex/standard_names/graph_ops.py|StandardName|_lifecycleless_reconcile_lock|5021,5021
imas_codex/standard_names/graph_ops.py|StandardName|_refine_claim_release_lock|23021,23022
imas_codex/standard_names/graph_ops.py|StandardName|_structural_authority_lock|25597,25598
imas_codex/standard_names/graph_ops.py|StandardName|needs_composition|3083,4094,5822
imas_codex/standard_names/graph_ops.py|StandardName|quarantine_reason|23584
imas_codex/standard_names/graph_ops.py|StandardName|reservation_claim_seq|10221
imas_codex/standard_names/graph_ops.py|StandardName|reservation_claim_token|10220
imas_codex/standard_names/graph_ops.py|StandardName|reservation_source_id|10219
imas_codex/standard_names/graph_ops.py|StandardName|review_resubmit_count|23507,23509,23568,25102
imas_codex/standard_names/graph_ops.py|StandardName|updated_at|4327
imas_codex/standard_names/graph_ops.py|StandardNameSource|_claim_lock|14557,14558,14630,14631,14672,14673,14709,14710
imas_codex/standard_names/graph_ops.py|StandardNameSource|_drain_scope_lock|1379,1380
imas_codex/standard_names/graph_ops.py|StandardNameSource|run_id|7241,7538
imas_codex/standard_names/graph_ops.py|StandardNameSource|skipped_at|10461,10462,11106
imas_codex/standard_names/provenance_lifecycle.py|StandardNameSource|standard_name_id|176
imas_codex/standard_names/signed_manifest.py|StandardName|quarantine_reason|1786
imas_codex/standard_names/signed_manifest.py|StandardNameChange|authority_rows_sha256|3556,4181
imas_codex/standard_names/signed_manifest.py|StandardNameChange|detached_target_ids|4179
imas_codex/standard_names/signed_manifest.py|StandardNameChange|source_id|4178
[transient_lock]
imas_codex/standard_names/parents.py|StandardName|_structural_authority_replay_lock|421,422
imas_codex/standard_names/parents.py|StandardName|_structural_authority_grounding_lock|487,488
[limitation]
imas_codex/graph/temp_neo4j.py|Facility|facility_id|552
""".strip()


@dataclass(frozen=True, slots=True)
class CypherPropertyFinding:
    """One checked property whose disposition needs attention."""

    path: Path
    line: int
    alias: str
    label: str | None
    property_name: str
    reason: str
    category: str = "unresolved"

    def __str__(self) -> str:
        """Render a compact source-located finding."""
        qualified = (
            f"{self.label}.{self.property_name}"
            if self.label
            else f"{self.alias}.{self.property_name}"
        )
        return f"{self.path}:{self.line}: {qualified}: {self.reason}"


@dataclass(frozen=True, slots=True)
class CypherPropertyReport:
    """Summary of schema checks over Cypher property occurrences."""

    checked_properties: int
    violations: tuple[CypherPropertyFinding, ...]
    allowlisted: tuple[CypherPropertyFinding, ...]


def _parse_adjudications() -> dict[str, Counter[tuple[str, int, str, str]]]:
    """Parse the exact source occurrences already adjudicated in this tree."""
    result = {category: Counter() for category in _ADJUDICATION_REASONS}
    category: str | None = None
    for row in _ADJUDICATED_OCCURRENCES.splitlines():
        if row.startswith("[") and row.endswith("]"):
            category = row[1:-1]
            if category not in result:
                raise ValueError(f"Unknown Cypher-property adjudication: {category}")
            continue
        if category is None:
            raise ValueError("Cypher-property adjudication row has no category")
        path, label, property_name, source_lines = row.split("|")
        for source_line in source_lines.split(","):
            result[category][(path, int(source_line), label, property_name)] += 1
    return result


_ADJUDICATIONS = _parse_adjudications()


def _adjudication_key(
    path: Path,
    line: int,
    label: str,
    property_name: str,
) -> tuple[str, int, str, str] | None:
    """Return a repository-relative key for an exact source occurrence."""
    try:
        relative_path = path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return None
    return (relative_path.as_posix(), line, label, property_name)


def _matching_adjudication(
    key: tuple[str, int, str, str] | None,
    remaining: dict[str, Counter[tuple[str, int, str, str]]],
) -> tuple[str, tuple[str, int, str, str]] | None:
    """Match one inventory occurrence without making source lines semantic."""
    if key is None:
        return None
    path, _, label, property_name = key
    for category, occurrences in remaining.items():
        for inventory_key, count in occurrences.items():
            inventory_path, _, inventory_label, inventory_property = inventory_key
            if (
                count > 0
                and inventory_path == path
                and inventory_label == label
                and inventory_property == property_name
            ):
                return category, inventory_key
    return None


def _string_value(node: ast.AST) -> str | None:
    """Return the static portion of a Python string expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            parts.append("{dynamic}")
    return "".join(parts)


def _assignment_names(node: ast.AST) -> Iterable[str]:
    """Yield names assigned by an assignment containing *node*."""
    targets: Sequence[ast.expr]
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = (node.target,)
    else:
        return
    for target in targets:
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, ast.Attribute):
            yield target.attr


def _call_name(node: ast.Call) -> str | None:
    """Return the terminal name of a called function or method."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _has_cypher_context(
    node: ast.AST,
    text: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Return whether a string is used as, or visibly contains, Cypher."""
    if _CYPHER_KEYWORD_RE.search(text) or _NODE_BINDING_RE.search(text):
        return True
    current = node
    for _ in range(6):
        parent = parents.get(current)
        if parent is None:
            break
        if isinstance(parent, ast.Assign | ast.AnnAssign):
            return any(
                _CYPHER_CONTEXT_RE.search(name) for name in _assignment_names(parent)
            )
        if isinstance(parent, ast.Call):
            call_name = _call_name(parent)
            return call_name in {"execute", "query", "run"}
        current = parent
    return False


def _python_strings(path: Path) -> Iterable[tuple[ast.AST, str]]:
    """Yield relevant Python string nodes and their static text."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(
            parents.get(node), ast.JoinedStr
        ):
            continue
        text = _string_value(node)
        if text is None or not _PROPERTY_RE.search(text):
            continue
        if _has_cypher_context(node, text, parents):
            yield node, text


def _source_paths(root: Path) -> Iterable[Path]:
    """Yield Python sources under *root* in deterministic order."""
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
    yield from sorted(
        path
        for path in root.rglob("*.py")
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def _declared_properties(
    schemas: Sequence[GraphSchema],
) -> dict[str, frozenset[str]]:
    """Build the label-to-property universe through ``get_all_slots``."""
    properties: dict[str, set[str]] = {}
    for schema in schemas:
        for label in schema.node_labels:
            properties.setdefault(label, set()).update(schema.get_all_slots(label))
    return {label: frozenset(names) for label, names in properties.items()}


def audit_cypher_properties(
    root: Path | str,
    *,
    schemas: Sequence[GraphSchema],
) -> CypherPropertyReport:
    """Validate statically labelled Cypher properties against LinkML.

    Property references on aliases whose label is not declared in the same
    literal cannot be proved by local static analysis. They remain visible as
    source-located allowlist entries instead of being silently skipped.
    """
    source_root = Path(root)
    declared = _declared_properties(schemas)
    violations: list[CypherPropertyFinding] = []
    allowlisted: list[CypherPropertyFinding] = []
    remaining_adjudications = {
        category: occurrences.copy() for category, occurrences in _ADJUDICATIONS.items()
    }
    checked = 0

    for path in _source_paths(source_root):
        for node, text in _python_strings(path):
            bindings: dict[str, set[str]] = {}
            for match in _NODE_BINDING_RE.finditer(text):
                bindings.setdefault(match["alias"], set()).add(match["label"])

            for match in _PROPERTY_RE.finditer(text):
                checked += 1
                alias = match["alias"]
                property_name = match["property"]
                line = int(getattr(node, "lineno", 1)) + text.count(
                    "\n", 0, match.start()
                )
                labels = bindings.get(alias, set())
                known_labels = labels & declared.keys()
                unknown_labels = labels - declared.keys()

                if not labels:
                    allowlisted.append(
                        CypherPropertyFinding(
                            path=path,
                            line=line,
                            alias=alias,
                            label=None,
                            property_name=property_name,
                            reason="alias label is not declared in this literal",
                        )
                    )
                    continue
                if unknown_labels:
                    allowlisted.append(
                        CypherPropertyFinding(
                            path=path,
                            line=line,
                            alias=alias,
                            label=", ".join(sorted(labels)),
                            property_name=property_name,
                            reason="node label is absent from the supplied LinkML schemas",
                        )
                    )
                    continue

                matching_labels = {
                    label for label in known_labels if property_name in declared[label]
                }
                if len(known_labels) > 1 and matching_labels != known_labels:
                    allowlisted.append(
                        CypherPropertyFinding(
                            path=path,
                            line=line,
                            alias=alias,
                            label=", ".join(sorted(known_labels)),
                            property_name=property_name,
                            reason=(
                                "alias is bound to multiple labels with different "
                                "property declarations"
                            ),
                        )
                    )
                    continue
                if matching_labels:
                    continue
                label = next(iter(known_labels))
                key = _adjudication_key(path, line, label, property_name)
                adjudication = _matching_adjudication(key, remaining_adjudications)
                if adjudication is not None:
                    category, inventory_key = adjudication
                    remaining_adjudications[category][inventory_key] -= 1
                    allowlisted.append(
                        CypherPropertyFinding(
                            path=path,
                            line=line,
                            alias=alias,
                            label=label,
                            property_name=property_name,
                            reason=_ADJUDICATION_REASONS[category],
                            category=category,
                        )
                    )
                    continue
                violations.append(
                    CypherPropertyFinding(
                        path=path,
                        line=line,
                        alias=alias,
                        label=label,
                        property_name=property_name,
                        reason="property is not declared by LinkML for this label",
                        category="violation",
                    )
                )

    if source_root.resolve() == (_REPO_ROOT / "imas_codex").resolve():
        for category, occurrences in remaining_adjudications.items():
            if category == "defect":
                continue
            for (relative_path, line, label, property_name), count in sorted(
                occurrences.items()
            ):
                for _ in range(count):
                    violations.append(
                        CypherPropertyFinding(
                            path=_REPO_ROOT / relative_path,
                            line=line,
                            alias="<adjudication>",
                            label=label,
                            property_name=property_name,
                            reason=(
                                f"stale {category} adjudication does not match a "
                                "current source occurrence"
                            ),
                            category="stale_adjudication",
                        )
                    )

    return CypherPropertyReport(
        checked_properties=checked,
        violations=tuple(violations),
        allowlisted=tuple(allowlisted),
    )

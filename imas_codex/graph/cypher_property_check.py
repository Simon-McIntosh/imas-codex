"""Check statically resolvable Cypher properties against LinkML schemas."""

from __future__ import annotations

import ast
import re
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


@dataclass(frozen=True, slots=True)
class CypherPropertyFinding:
    """One checked property whose disposition needs attention."""

    path: Path
    line: int
    alias: str
    label: str | None
    property_name: str
    reason: str

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
                violations.append(
                    CypherPropertyFinding(
                        path=path,
                        line=line,
                        alias=alias,
                        label=label,
                        property_name=property_name,
                        reason="property is not declared by LinkML for this label",
                    )
                )

    return CypherPropertyReport(
        checked_properties=checked,
        violations=tuple(violations),
        allowlisted=tuple(allowlisted),
    )

"""No module writes a standard name's domain outside the one writer.

``imas_codex.standard_names.domain_decision.write_name_domain`` sets
``physics_domain``, ``source_domains`` and the ``HAS_PHYSICS_DOMAIN`` edges in
one transaction so the three cannot disagree. This module reads every Python
module under ``imas_codex/standard_names/`` and its Cypher strings, and refuses
any ``SET`` of a name's ``physics_domain`` or ``source_domains``, and any
``MERGE``, ``CREATE`` or ``DELETE`` of a ``HAS_PHYSICS_DOMAIN`` relationship,
made outside the writer and the functions it calls.

A write whose target label is proven not to be ``StandardName`` is allowed —
a DD node's own domain is the source of truth the name's decision reads. A
variable whose label the statement leaves unlabelled counts as a name: the
reader cannot prove otherwise. A derived parent's domain is the one its
children carry, stated by
``domain_decision.derived_parent_domain_decision``; the materialiser writes no
domain itself, so this table has no exemptions.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

import imas_codex.standard_names as standard_names
from imas_codex.cli.sn import CYPHER_LITERAL_OR_COMMENT, cypher_mutation_clause
from imas_codex.graph.cypher_property_check import (
    NODE_BINDING_RE,
    cypher_set_blocks,
    python_cypher_strings,
)

PACKAGE_ROOT = Path(standard_names.__file__).resolve().parent
WRITER_MODULE = "domain_decision.py"
WRITER_FUNCTION = "write_name_domain"

DOMAIN_PROPS = frozenset({"physics_domain", "source_domains"})
DOMAIN_EDGE = "HAS_PHYSICS_DOMAIN"

_DOMAIN_HINT = re.compile(r"physics_domain|source_domains|" + DOMAIN_EDGE)

_PROPERTY_ASSIGN_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=")
_REL_WRITE_RE = re.compile(
    r"\b(MERGE|CREATE)\b[^\n;]*?\(\s*[A-Za-z_]\w*\s*\)[^\n;]*?"
    r"\[\s*(?:[A-Za-z_]\w*\s*)?:\s*([A-Za-z_]\w*)\s*[^\]]*\]"
)
_REL_BINDING_RE = re.compile(r"\[\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)")
_DELETE_RE = re.compile(r"\bDELETE\s+([A-Za-z_]\w*)")

#: Direct ``StandardName`` domain writes the one-writer rule has not retired
#: yet, keyed by the file, the enclosing function, the write and its target.
#: The guard refuses every other site, and refuses an entry that no longer
#: occurs, so this table can only shrink. It is empty: every derived-parent
#: domain write goes through ``write_name_domain``.
EXEMPT_SITES: dict[tuple[str, str, str, str], str] = {}


def _package_modules() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _domain_cypher(path: Path) -> list[tuple[int, str]]:
    """Every Python string in *path* that reads as Cypher touching a domain.

    ``python_cypher_strings`` decides which strings are Cypher; this adds the
    domain filter that decides which of those can write a name's domain.
    """
    return [
        (node.lineno, text)
        for node, text in python_cypher_strings(path, require_property=False)
        if _DOMAIN_HINT.search(text)
    ]


def _node_labels(text: str) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for match in NODE_BINDING_RE.finditer(text):
        labels.setdefault(match["alias"], set()).add(match["label"])
    return labels


def domain_writes(text: str) -> list[tuple[str, str]]:
    """The (write, target) pairs in one Cypher statement that set a name's domain.

    A property assignment counts when the variable is labelled ``StandardName``
    or carries no label at all, and does not when its label is something else —
    a DD node's own domain, or a ``StandardNameSource``'s copied one, is not a
    name's domain. Relationship writes count for a ``HAS_PHYSICS_DOMAIN`` merge,
    create or delete.

    The statement is read through the project's own Cypher readers: a statement
    with no mutation outside its literals and comments reads nothing, and the
    literal and comment spans are blanked before any clause is parsed, so a
    commented-out or quoted ``SET`` is not a write while a real one beside it
    still is.
    """
    if cypher_mutation_clause(text) is None:
        return []
    executable = CYPHER_LITERAL_OR_COMMENT.sub(" ", text)
    labels = _node_labels(executable)
    writes: list[tuple[str, str]] = []
    for _set, body_start, body_end in cypher_set_blocks(executable):
        body = executable[body_start:body_end]
        for match in _PROPERTY_ASSIGN_RE.finditer(body):
            variable, prop = match.group(1), match.group(2)
            if prop not in DOMAIN_PROPS:
                continue
            known = labels.get(variable, set())
            if known and "StandardName" not in known:
                continue
            writes.append((f"set {prop}", f"{variable}.{prop}"))
    for match in _REL_WRITE_RE.finditer(executable):
        verb, relationship = match.groups()
        if relationship == DOMAIN_EDGE:
            writes.append((f"{verb.lower()} {DOMAIN_EDGE}", match.group(0).strip()))
    edge_variables = {
        match.group(1)
        for match in _REL_BINDING_RE.finditer(executable)
        if match.group(2) == DOMAIN_EDGE
    }
    for match in _DELETE_RE.finditer(executable):
        if match.group(1) in edge_variables:
            writes.append((f"delete {DOMAIN_EDGE}", match.group(1)))
    return writes


def _module_level_constant_names(tree: ast.Module) -> dict[int, str]:
    """Map a module-level constant's line to the name it is bound to."""
    names: dict[int, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            continue
        value = statement.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        ):
            if isinstance(target, ast.Name):
                names[value.lineno] = target.id
    return names


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    """Map each string constant's line to the function that contains it."""
    owners: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                owners.setdefault(child.lineno, node.name)
    return owners


def _called_names(function: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _referenced_names(function: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}


def _writer_scope() -> tuple[set[tuple[str, str]], set[str]]:
    """The functions the writer reaches, and the Cypher constants it hands on.

    Returns ``((module, function) ...)`` for ``write_name_domain`` and every
    function it calls within the package, and the module-level constant names
    those functions reference, so the writer's own Cypher is attributed to it.
    """
    definitions: dict[str, list[tuple[str, ast.AST]]] = {}
    for path in _package_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                definitions.setdefault(node.name, []).append((relative, node))
    scope: set[tuple[str, str]] = set()
    pending = [(WRITER_MODULE, WRITER_FUNCTION)]
    constants: set[str] = set()
    while pending:
        module, name = pending.pop()
        if (module, name) in scope:
            continue
        scope.add((module, name))
        for owner, function in definitions.get(name, []):
            if owner != module:
                continue
            for callee in _called_names(function):
                for callee_owner, _ in definitions.get(callee, []):
                    if (callee_owner, callee) not in scope:
                        pending.append((callee_owner, callee))
            if module == WRITER_MODULE:
                constants |= _referenced_names(function)
    return scope, constants


@functools.cache
def _findings() -> list[tuple[str, int, str, str, str, str]]:
    """Every name-domain write in the package, with where it is and who owns it."""
    scope, writer_constants = _writer_scope()
    findings = []
    for path in _package_modules():
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = _enclosing_functions(tree)
        constants = _module_level_constant_names(tree)
        for lineno, text in _domain_cypher(path):
            owner = functions.get(lineno)
            allowed = owner is not None and (relative, owner) in scope
            if owner is None:
                allowed = (
                    relative == WRITER_MODULE
                    and constants.get(lineno, "") in writer_constants
                )
            for kind, target in domain_writes(text):
                findings.append(
                    (
                        relative,
                        lineno,
                        owner or constants.get(lineno, "?"),
                        kind,
                        target,
                        "writer" if allowed else "outside",
                    )
                )
    return findings


# --------------------------------------------------------------------------
# The scanner itself: it must see a planted write and must not refuse a
# non-StandardName label.


def test_scanner_sees_a_direct_scalar_write() -> None:
    cypher = (
        "MATCH (sn:StandardName {id: $sn_id}) "
        "SET sn.physics_domain = $domain, sn.source_domains = $domains"
    )
    assert domain_writes(cypher) == [
        ("set physics_domain", "sn.physics_domain"),
        ("set source_domains", "sn.source_domains"),
    ]


def test_scanner_sees_a_direct_edge_write() -> None:
    cypher = (
        "MATCH (sn:StandardName {id: $sn_id}) "
        "MERGE (d:PhysicsDomain {id: $domain}) "
        "MERGE (sn)-[:HAS_PHYSICS_DOMAIN]->(d)"
    )
    assert [kind for kind, _ in domain_writes(cypher)] == ["merge HAS_PHYSICS_DOMAIN"]


def test_scanner_sees_a_direct_edge_delete() -> None:
    cypher = (
        "MATCH (sn:StandardName {id: $sn_id}) "
        "OPTIONAL MATCH (sn)-[old:HAS_PHYSICS_DOMAIN]->(:PhysicsDomain) DELETE old"
    )
    assert ("delete HAS_PHYSICS_DOMAIN", "old") in domain_writes(cypher)


def test_scanner_treats_an_unlabelled_target_as_a_name() -> None:
    unlabelled = "MATCH (parent) SET parent.physics_domain = $domain"
    assert domain_writes(unlabelled) == [
        ("set physics_domain", "parent.physics_domain")
    ]


def test_scanner_allows_a_non_standard_name_label() -> None:
    """A DD path's own domain is the source of truth, not a name's domain."""
    cypher = (
        "MATCH (node:IMASNode {id: $id}) "
        "SET node.physics_domain = $domain "
        "MERGE (sns:StandardNameSource {id: $sns}) "
        "SET sns.physics_domain = $domain"
    )
    assert domain_writes(cypher) == []


def test_scanner_ignores_a_commented_out_set() -> None:
    """A line comment and a block comment are not writes."""
    cypher = (
        "MATCH (sn:StandardName {id: $sn_id})\n"
        "// SET sn.physics_domain = 'equilibrium'\n"
        "/* SET sn.source_domains = ['equilibrium'] */\n"
        "RETURN sn.id AS sn_id"
    )
    assert domain_writes(cypher) == []


def test_scanner_ignores_a_quoted_set() -> None:
    """A ``SET`` inside a string literal is text, not a clause."""
    cypher = (
        "MATCH (sn:StandardName {id: $sn_id}) "
        "SET sn.description = 'SET sn.physics_domain = 1' "
        "RETURN sn.id AS sn_id"
    )
    assert domain_writes(cypher) == []


def test_scanner_still_sees_a_real_set_beside_a_commented_one() -> None:
    """Blanking the comments and literals must not blank the real write."""
    cypher = (
        "MATCH (sn:StandardName {id: $sn_id})\n"
        "// SET sn.physics_domain = 'equilibrium'\n"
        "SET sn.description = 'SET sn.physics_domain = 1', "
        "sn.physics_domain = $domain\n"
        "RETURN sn.id AS sn_id"
    )
    assert domain_writes(cypher) == [("set physics_domain", "sn.physics_domain")]


def test_scanner_reads_a_where_clause_as_a_read() -> None:
    cypher = (
        "MATCH (sn:StandardName) WHERE sn.physics_domain = $domain "
        "RETURN sn.id AS sn_id"
    )
    assert domain_writes(cypher) == []


# --------------------------------------------------------------------------
# The package: the writer is the one site, and the exemptions are only the
# recorded, unfixed ones.


def test_writer_carries_its_own_writes() -> None:
    """Positive control: the scanner sees the writer's three writes and allows them."""
    writer_writes = {
        (kind, target)
        for relative, _lineno, _owner, kind, target, where in _findings()
        if relative == WRITER_MODULE and where == "writer"
    }
    assert ("set physics_domain", "sn.physics_domain") in writer_writes
    assert ("set source_domains", "sn.source_domains") in writer_writes
    assert ("merge HAS_PHYSICS_DOMAIN", "MERGE (sn)-[:HAS_PHYSICS_DOMAIN]") in (
        writer_writes
    )


def test_every_module_is_read() -> None:
    """An empty or short scan would pass every assertion below vacuously."""
    modules = _package_modules()
    assert len(modules) > 20
    assert (PACKAGE_ROOT / "graph_store" / "graph_lineage.py") in modules
    scanned = {relative for relative, *_ in _findings()}
    assert WRITER_MODULE in scanned


def test_no_module_writes_a_name_domain_outside_the_writer() -> None:
    outside = [
        finding
        for finding in _findings()
        if finding[5] == "outside"
        and (finding[0], finding[2], finding[3], finding[4]) not in EXEMPT_SITES
    ]
    assert outside == [], "\n".join(
        f"{relative}:{lineno} in {owner}: {kind} on {target} — route it through "
        f"write_name_domain, or record it in EXEMPT_SITES with its reason"
        for relative, lineno, owner, kind, target, _ in outside
    )


def test_exempt_sites_are_exactly_the_recorded_ones() -> None:
    """A retired exemption fails here, so the table cannot outlive its sites."""
    observed = {
        (relative, owner, kind, target)
        for relative, _lineno, owner, kind, target, where in _findings()
        if where == "outside"
    }
    assert observed == set(EXEMPT_SITES), (
        "unrecorded direct writes: "
        f"{sorted(observed - set(EXEMPT_SITES))}; stale exemptions: "
        f"{sorted(set(EXEMPT_SITES) - observed)}"
    )

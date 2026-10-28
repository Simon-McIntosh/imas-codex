"""Refuse module-namespace bindings and lookups built from string expressions.

A binding assembled at import time from two string literals —
``globals()["reconcile_" + "siblings"] = impl`` — creates a name that is
importable and greppable by nobody: the AST carries a ``Subscript`` target
rather than a ``Name``, so an audit that collects assignment targets skips it,
while ``getattr(module, name)`` still resolves it. The closure audits that hold
the standard-name composition roots therefore cannot see the symbol a facade
re-exposes, and the naming audit cannot see its name.

These checks refuse the pattern rather than its current instances: a dynamic key
in ``globals()`` is what makes a name unreadable to every static instrument.
There is deliberately no exemption list — an allowlisted dynamic key would move
the blind spot rather than remove it.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "imas_codex"


def _globals_subscripts(tree: ast.Module) -> list[ast.Subscript]:
    """Every subscript applied to a ``globals()`` call in *tree*."""
    subscripts: list[ast.Subscript] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        target = node.value
        if (
            isinstance(target, ast.Call)
            and isinstance(target.func, ast.Name)
            and target.func.id == "globals"
        ):
            subscripts.append(node)
    return subscripts


def _assignment_targets(tree: ast.Module) -> set[int]:
    """Object ids of the nodes that an assignment in *tree* binds."""
    targets: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pending = list(node.targets)
        elif isinstance(node, ast.AugAssign):
            pending = [node.target]
        else:
            continue
        while pending:
            target = pending.pop()
            if isinstance(target, ast.Subscript):
                targets.add(id(target))
            elif isinstance(target, ast.Tuple | ast.List):
                pending.extend(target.elts)
    return targets


def _has_literal_key(node: ast.Subscript) -> bool:
    """Whether the subscript's key is a string literal, so grep can find it."""
    key = node.slice
    return isinstance(key, ast.Constant) and isinstance(key.value, str)


def _violations(root: Path) -> list[str]:
    """Report every dynamic ``globals()`` key under *root*, as ``path:line``."""
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assigned = _assignment_targets(tree)
        for node in _globals_subscripts(tree):
            if _has_literal_key(node):
                continue
            kind = "binding" if id(node) in assigned else "lookup"
            violations.append(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
                f"{kind} through a non-literal globals() key"
            )
    return violations


def _scan(rule: str) -> list[str]:
    assert PACKAGE_ROOT.is_dir(), f"{PACKAGE_ROOT} is not a directory"
    return [
        entry for entry in _violations(PACKAGE_ROOT) if f": {rule} through" in entry
    ]


def test_no_dynamic_name_binding() -> None:
    """No module binds a name through ``globals()`` with a computed key."""
    assert _scan("binding") == []


def test_no_dynamic_name_lookup() -> None:
    """No module reads its namespace through ``globals()`` with a computed key."""
    assert _scan("lookup") == []

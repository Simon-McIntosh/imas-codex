"""Every collaborator a worker task asks for by name is declared.

A worker task resolves a collaborator through ``_WORKER_DEPENDENCIES``, which
reads the owning module for a task symbol and the composition root's declared
table otherwise. A name in neither raises ``KeyError`` from inside the task, and
the pool it belongs to retries the item without end. The audit walks the
constant names the task modules pass to the resolver and requires each to
resolve.
"""

from __future__ import annotations

import ast
from pathlib import Path

from imas_codex.standard_names import worker_tasks
from imas_codex.standard_names.workers import _WORKER_DEPENDENCIES

_TASK_PACKAGE = Path(worker_tasks.__file__).parent


def _requested_names(path: Path) -> set[str]:
    """The constant names a task module passes to the dependency resolver."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "_worker_dependency":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        value = node.args[0].value
        if isinstance(value, str):
            names.add(value)
    return names


def _all_requested_names() -> dict[str, set[str]]:
    requested: dict[str, set[str]] = {}
    for path in sorted(_TASK_PACKAGE.glob("worker_*.py")):
        names = _requested_names(path)
        if names:
            requested[path.name] = names
    return requested


def test_every_requested_collaborator_is_declared() -> None:
    """A name a task asks for is in the table or in its owning module."""
    requested = _all_requested_names()
    assert requested, "the walk found no dependency requests; the check is blind"

    undeclared = sorted(
        f"{module}: {name}"
        for module, names in requested.items()
        for name in names
        if name not in worker_tasks._OWNER_BY_NAME and name not in _WORKER_DEPENDENCIES
    )
    assert undeclared == [], undeclared


def test_every_requested_collaborator_resolves() -> None:
    """A declared name is not enough: the resolver must return something."""
    names = {name for names in _all_requested_names().values() for name in names}
    assert names, "the walk found no dependency requests; the check is blind"

    unresolved: list[str] = []
    for name in sorted(names):
        try:
            worker_tasks._resolve_dependency(name)
        except Exception as exc:
            unresolved.append(f"{name}: {type(exc).__name__}: {exc}")
    assert unresolved == [], unresolved


def test_the_review_batch_resolves_its_name_definition_note() -> None:
    """The note builder the review batch asks for is reachable by name."""
    from imas_codex.standard_names.worker_tasks import worker_review_docs

    resolved = worker_tasks._resolve_dependency("_name_definition_note")
    assert resolved is getattr(worker_review_docs, "_name_definition_note")
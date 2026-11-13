"""Every collaborator a worker task asks for by name is declared where it lives.

A task resolves a collaborator through ``_resolve_dependency``, which reads a
symbol from the task module that owns it and falls back to the composition
root's declared table. A name in neither raises ``KeyError`` from inside the
task, and the pool it belongs to retries the item without end.

The two tables answer different questions, so each requested name is held to the
one that owns it:

* a symbol a task module defines is declared in ``_OWNERS_BY_MODULE``, and the
  resolver reads it from there;
* a collaborator supplied from outside the task package is declared in
  ``_WORKER_DEPENDENCIES``.

A symbol a task module defines and files in the root table resolves by a second
route, which is the defect the placement check below exists to catch.
"""

from __future__ import annotations

import ast
from pathlib import Path

from imas_codex.standard_names import worker_tasks
from imas_codex.standard_names.workers import _WORKER_DEPENDENCIES

_TASK_PACKAGE = Path(worker_tasks.__file__).parent


def _task_module_paths() -> list[Path]:
    return sorted(_TASK_PACKAGE.glob("worker_*.py"))


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


def _defined_names(path: Path) -> set[str]:
    """The names a task module binds at module level by defining them itself.

    Imports are excluded: a name a task module imports is supplied from
    somewhere else, so it is not this module's symbol to declare.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _all_requested_names() -> dict[str, set[str]]:
    requested: dict[str, set[str]] = {}
    for path in _task_module_paths():
        names = _requested_names(path)
        if names:
            requested[path.stem] = names
    return requested


def _names_defined_by_task_modules() -> set[str]:
    defined: set[str] = set()
    for path in _task_module_paths():
        defined |= _defined_names(path)
    return defined


def test_every_requested_collaborator_is_declared() -> None:
    """A requested name is declared in at least one of the two tables."""
    requested = _all_requested_names()
    assert requested, "the walk found no dependency requests; the check is blind"

    undeclared = sorted(
        f"{module}: {name}"
        for module, names in requested.items()
        for name in names
        if name not in worker_tasks._OWNER_BY_NAME and name not in _WORKER_DEPENDENCIES
    )
    assert undeclared == [], undeclared


def test_a_task_module_symbol_is_owned_not_supplied_from_the_root() -> None:
    """A symbol a task module defines is resolved from its owning module."""
    requested = {name for names in _all_requested_names().values() for name in names}
    assert requested, "the walk found no dependency requests; the check is blind"

    owned_and_defined = _names_defined_by_task_modules() & requested

    not_owned = sorted(owned_and_defined - set(worker_tasks._OWNER_BY_NAME))
    assert not_owned == [], not_owned

    misfiled = sorted(owned_and_defined & set(_WORKER_DEPENDENCIES))
    assert misfiled == [], misfiled


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

    assert (
        "_name_definition_note" in worker_tasks._OWNERS_BY_MODULE["worker_review_docs"]
    )
    assert "_name_definition_note" not in _WORKER_DEPENDENCIES
    resolved = worker_tasks._resolve_dependency("_name_definition_note")
    assert resolved is worker_review_docs._name_definition_note

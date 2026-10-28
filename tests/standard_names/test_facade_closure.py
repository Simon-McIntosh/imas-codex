"""Closure guards for the standard-name composition roots.

Symbols that moved into ``graph_store`` and ``worker_tasks`` are reachable only
from the module that defines them. The entrypoints keep two structures for
internal wiring — the owner map in ``graph_store`` and the wired-collaborator
set in ``graph_ops`` — and each is held to the modules it describes rather than
to a hand-maintained list, so a drifted entry or a façade that resolves moved
names again fails here.
"""

from __future__ import annotations

import ast
import functools
import importlib
from pathlib import Path

import pytest

_STORE_PACKAGE = "imas_codex.standard_names.graph_store"
_TASK_PACKAGE = "imas_codex.standard_names.worker_tasks"

#: Logger objects every module defines for itself; they carry no behaviour and
#: resolving one through the owner map would name the wrong module's logger.
_NOT_MOVED = frozenset({"logger"})

#: ``graph_foundation`` is the module every sibling imports directly and holds
#: the service wiring itself; nothing resolves its names through the entrypoint
#: map, so its own names are not moved symbols.
_NOT_MOVED_MODULES = frozenset({"graph_foundation"})


def _store_modules() -> dict[str, object]:
    """Import every module the store package holds, mapped or not.

    The set comes from the package directory rather than from the owner map:
    a module the map omits is exactly the one whose public symbols would go
    unmapped unnoticed, so taking the set from the map would hide the defect
    this check exists to find.
    """
    from imas_codex.standard_names import graph_store

    modules: dict[str, object] = {}
    for path in sorted(Path(graph_store.__file__).parent.glob("*.py")):
        if path.stem == "__init__":
            continue
        modules[path.stem] = importlib.import_module(f".{path.stem}", _STORE_PACKAGE)
    return modules


def _moved_callables() -> dict[str, str]:
    """Moved symbols a caller could reach as a name, and their owner path.

    A ``functools.partial`` carries no ``__module__`` of its own, so it does not
    report the module that built it. Comparing ``__module__`` therefore dropped
    every partial-owned entrypoint — the ten manifest appliers among them — from
    this set, leaving them outside both the entrypoint scan and the forwarder
    scan while a facade could still re-expose them. The binding a store module
    holds is what makes a symbol its own, so a partial counts when the owner
    module's namespace carries that name.
    """
    from imas_codex.standard_names.graph_store import _OWNERS_BY_MODULE

    moved: dict[str, str] = {}
    for module_name, names in _OWNERS_BY_MODULE.items():
        module = importlib.import_module(f".{module_name}", _STORE_PACKAGE)
        for name in names:
            attribute = getattr(module, name, None)
            if attribute is None or name in _NOT_MOVED:
                continue
            if not _is_owned(module, name, attribute):
                continue
            if callable(attribute):
                moved[name] = f"{_STORE_PACKAGE}.{module_name}"
    return moved


def _is_owned(module: object, name: str, attribute: object) -> bool:
    """Whether *attribute*, bound as *name*, belongs to *module* itself."""
    if isinstance(attribute, functools.partial):
        return name in vars(module)
    return getattr(attribute, "__module__", None) == module.__name__


def _local_definitions(path: str) -> set[str]:
    """Names a module defines itself: functions, classes and assignments."""
    names: set[str] = set()
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _service_attributes_read(path: str) -> set[str]:
    """Attributes the store reads off the wired collaborator object.

    A module reaches the collaborators either as a ``get_services()`` call or
    through the ``services`` parameter the binding wrapper injects, and may
    hold a local alias of either, so each function scope is resolved first.
    """
    names: set[str] = set()
    for function in ast.walk(ast.parse(Path(path).read_text())):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = list(function.args.args) + list(function.args.kwonlyargs)
        collaborators = {a.arg for a in arguments if a.arg == "services"}
        statements = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Attribute))
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for node in statements:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if _supplies_collaborators(node.value, collaborators):
                    collaborators.update(
                        target.id for target in targets if isinstance(target, ast.Name)
                    )
            elif isinstance(node, ast.Attribute):
                if _is_services_call(node.value):
                    names.add(node.attr)
                elif (
                    isinstance(node.value, ast.Name) and node.value.id in collaborators
                ):
                    names.add(node.attr)
    return names


def _is_services_call(value: ast.expr) -> bool:
    if not isinstance(value, ast.Call):
        return False
    callee = value.func
    if isinstance(callee, ast.Name):
        return callee.id == "get_services"
    return isinstance(callee, ast.Attribute) and callee.attr == "get_services"


def _supplies_collaborators(value: ast.expr | None, collaborators: set[str]) -> bool:
    if isinstance(value, ast.Name):
        return value.id in collaborators
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "get_services"
    ) or (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "get_services"
    )


def _worker_task_names() -> set[str]:
    package = importlib.import_module(_TASK_PACKAGE)
    names: set[str] = set()
    for path in Path(package.__file__).parent.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    return names


def _owner_package_module(module: str | None) -> str | None:
    """Return the owner-package-relative module path an import names."""
    if not module:
        return None
    parts = module.split(".")
    for package in ("graph_store", "worker_tasks"):
        if package in parts:
            index = parts.index(package)
            return ".".join(parts[index:])
    return None


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map a local import alias to the symbol it brings in.

    An import may be relative (``from .graph_store.graph_catalog import x``) or
    absolute (``from imas_codex.standard_names.graph_store.graph_catalog``
    ``import x``). The absolute spelling is the common one in this repository,
    so a scan that understood only the relative form would accept a forwarder.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if _owner_package_module(node.module) is None:
            continue
        for imported in node.names:
            aliases[imported.asname or imported.name] = imported.name
    return aliases


def _delegating_functions(module: ast.Module, targets: set[str]) -> list[str]:
    """Names of functions whose whole body delegates to a target symbol."""
    aliases = _import_aliases(module)
    forwarders: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        delegated = _delegated_name(node)
        if delegated in targets or aliases.get(delegated) in targets:
            forwarders.append(node.name)
    return forwarders


def _delegated_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Return the symbol a function body only returns a call to, if any."""
    body = [
        statement
        for statement in node.body
        if not isinstance(statement, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    value = body[0].value
    if isinstance(value, ast.Call):
        callee = value.func
        if isinstance(callee, ast.Name):
            return callee.id
        if isinstance(callee, ast.Attribute):
            return callee.attr
    if isinstance(value, ast.Name):
        return value.id
    return None


def test_owner_map_entries_resolve_at_their_owner() -> None:
    """A map entry may name only a module that actually carries the symbol."""
    from imas_codex.standard_names.graph_store import (
        _OWNERS_BY_MODULE,
        resolve_owned_name,
    )

    stale: list[str] = []
    for module_name, names in _OWNERS_BY_MODULE.items():
        module = importlib.import_module(f".{module_name}", _STORE_PACKAGE)
        for name in names:
            if not hasattr(module, name):
                stale.append(f"{module_name}: {name}")
            elif resolve_owned_name(name) is not getattr(module, name):
                stale.append(f"{module_name}: {name} resolves elsewhere")
    assert stale == []


def test_owner_map_covers_every_public_definition() -> None:
    """A store module, and every public symbol in it, must enter the map."""
    from imas_codex.standard_names.graph_store import _OWNERS_BY_MODULE

    modules = _store_modules()
    assert modules, "the store package enumerated no modules"

    unmapped = sorted(set(modules) - set(_OWNERS_BY_MODULE) - _NOT_MOVED_MODULES)
    assert unmapped == [], f"store modules absent from the owner map: {unmapped}"

    missing: list[str] = []
    for module_name, module in modules.items():
        if module_name in _NOT_MOVED_MODULES:
            continue
        roster = set(_OWNERS_BY_MODULE[module_name])
        for name in sorted(_local_definitions(module.__file__)):
            if name.startswith("_") or name in _NOT_MOVED:
                continue
            if name not in roster:
                missing.append(f"{module_name}: {name}")
    assert missing == []


def test_entrypoint_refuses_moved_symbols() -> None:
    """Neither entrypoint resolves a moved symbol, and it names the owner."""
    from imas_codex.standard_names import graph_ops, workers

    moved = _moved_callables()
    assert moved, "the owner map named no moved callable; the check would pass blind"

    resolved: list[str] = []
    for name, _owner in moved.items():
        for facade in (graph_ops, workers):
            try:
                getattr(facade, name)
            except AttributeError:
                continue
            resolved.append(f"{facade.__name__}.{name}")
    assert resolved == []

    moved_name = "write_standard_names"
    with pytest.raises(AttributeError) as refusal:
        getattr(graph_ops, moved_name)
    assert "graph_store.graph_catalog" in str(refusal.value)


def test_service_bridge_resolves_only_wired_names() -> None:
    """The bridge serves the wired collaborators and nothing else."""
    from imas_codex.standard_names import graph_ops
    from imas_codex.standard_names.graph_store import graph_foundation

    store = Path(importlib.import_module(_STORE_PACKAGE).__file__).parent
    requested: set[str] = set()
    for path in store.glob("*.py"):
        requested |= _service_attributes_read(str(path))
    assert requested == set(graph_ops._WIRED_SERVICE_NAMES)

    services = graph_foundation.get_services()
    witness = next(
        name
        for name in sorted(_moved_callables())
        if name not in graph_ops._WIRED_SERVICE_NAMES and name not in vars(graph_ops)
    )
    with pytest.raises(AttributeError) as refusal:
        getattr(services, witness)
    assert _moved_callables()[witness] in str(refusal.value)


def test_facades_define_no_delegating_forwarder() -> None:
    """A one-line delegation is a forwarder however it is described."""
    from imas_codex.standard_names import graph_ops, workers

    targets = set(_moved_callables()) | _worker_task_names()
    assert targets, "the owner map named no delegable symbol; the check is blind"

    forwarders: list[str] = []
    for facade in (graph_ops, workers):
        module = ast.parse(Path(facade.__file__).read_text())
        forwarders.extend(
            f"{facade.__name__}.{name}"
            for name in _delegating_functions(module, targets)
        )
    assert forwarders == []


def test_forwarder_scan_catches_absolute_import_delegation() -> None:
    """A delegation spelled with an absolute import is still a forwarder.

    The scan is exercised on planted source so the detector itself is checked:
    the absolute spelling is the common one in this repository, and a scan
    that only understood relative imports would accept it silently.
    """
    targets = {"write_standard_names"}

    absolute = ast.parse(
        "from imas_codex.standard_names.graph_store.graph_catalog import (\n"
        "    write_standard_names as _impl,\n"
        ")\n"
        "\n"
        "def persist_names(names):\n"
        "    return _impl(names)\n"
    )
    assert _delegating_functions(absolute, targets) == ["persist_names"]

    relative = ast.parse(
        "from .graph_store.graph_catalog import write_standard_names\n"
        "\n"
        "def persist_names(names):\n"
        "    return write_standard_names(names)\n"
    )
    assert _delegating_functions(relative, targets) == ["persist_names"]

"""Prevent default values from hiding reads of undeclared model attributes."""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Extend this tuple when another standard-name surface needs the same guard.
MODULE_ROOTS = (
    Path("imas_codex/standard_names/review"),
    Path("imas_codex/standard_names/export.py"),
    Path("imas_codex/standard_names/loop.py"),
)

# Module and distribution metadata are intentionally dynamic, not model fields.
LEGITIMATELY_ABSENT_DEFAULTED_ATTRIBUTES = {
    "__file__": "Module metadata is assigned by Python's import machinery.",
    "__version__": "Distribution metadata is supplied by the installed package.",
}


def _python_modules(source_root: Path, module_roots: tuple[Path, ...]) -> list[Path]:
    """Return the explicit files and package trees covered by the guard."""
    modules: list[Path] = []
    for module_root in module_roots:
        candidate = source_root / module_root
        if candidate.is_dir():
            modules.extend(candidate.rglob("*.py"))
        else:
            modules.append(candidate)
    return sorted(modules)


def _decorator_names(decorators: list[ast.expr]) -> set[str]:
    """Normalize simple and qualified decorator spellings."""
    names: set[str] = set()
    for decorator in decorators:
        if isinstance(decorator, ast.Name):
            names.add(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            names.add(decorator.attr)
        elif isinstance(decorator, ast.Call):
            names.update(_decorator_names([decorator.func]))
    return names


def _declared_class_attributes(source_root: Path) -> set[str]:
    """Collect fields and properties declared by classes owned by imas_codex."""
    attributes: set[str] = set()
    for module in sorted((source_root / "imas_codex").rglob("*.py")):
        tree = ast.parse(module.read_text(), filename=str(module))
        for class_node in (
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ):
            for member in class_node.body:
                if isinstance(member, ast.AnnAssign) and isinstance(
                    member.target, ast.Name
                ):
                    attributes.add(member.target.id)
                elif isinstance(member, ast.Assign):
                    attributes.update(
                        target.id
                        for target in member.targets
                        if isinstance(target, ast.Name)
                    )
                elif isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    if "property" in _decorator_names(member.decorator_list):
                        attributes.add(member.name)

            for member in ast.walk(class_node):
                if not isinstance(member, ast.Assign | ast.AnnAssign):
                    continue
                targets = (
                    member.targets
                    if isinstance(member, ast.Assign)
                    else [member.target]
                )
                attributes.update(
                    target.attr
                    for target in targets
                    if isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                )
    return attributes


def _undeclared_defaulted_attributes(source_root: Path) -> list[tuple[Path, int, str]]:
    """Find literal three-argument getattr calls whose default hides no class field."""
    declared_attributes = _declared_class_attributes(source_root)
    violations: list[tuple[Path, int, str]] = []
    for module in _python_modules(source_root, MODULE_ROOTS):
        tree = ast.parse(module.read_text(), filename=str(module))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 3
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                continue
            attribute_name = node.args[1].value
            if (
                attribute_name not in declared_attributes
                and attribute_name not in LEGITIMATELY_ABSENT_DEFAULTED_ATTRIBUTES
            ):
                violations.append((module, node.lineno, attribute_name))
    return violations


def test_defaulted_literal_attributes_are_declared() -> None:
    """A default may not turn an undeclared attribute read into empty evidence."""
    violations = _undeclared_defaulted_attributes(REPOSITORY_ROOT)
    rendered = "\n".join(
        f"{path.relative_to(REPOSITORY_ROOT)}:{line}: {attribute_name}"
        for path, line, attribute_name in violations
    )
    assert not violations, f"defaulted undeclared attributes:\n{rendered}"

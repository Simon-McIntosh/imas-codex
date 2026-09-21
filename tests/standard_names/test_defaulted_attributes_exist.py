"""Prevent default values from hiding reads of undeclared model attributes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
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

# Floors make a scan that sees nothing fail instead of passing silently.
MINIMUM_SCANNED_MODULES = 8
MINIMUM_LITERAL_DEFAULTED_GETATTRS = 24


@dataclass(frozen=True)
class DefaultedAttributeScan:
    """The aperture and violations reported by one static scan."""

    module_count: int
    candidate_count: int
    roots: list[ModuleRootScan]
    violations: list[tuple[Path, int, str]]


@dataclass(frozen=True)
class ModuleRootScan:
    """The source and candidate coverage contributed by one configured root."""

    root: Path
    exists: bool
    module_count: int
    candidate_count: int


def _python_modules(source_root: Path, module_root: Path) -> list[Path]:
    """Return Python modules contributed by one explicit root."""
    candidate = source_root / module_root
    if candidate.is_dir():
        return sorted(candidate.rglob("*.py"))
    if candidate.is_file():
        return [candidate]
    return []


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


def _defaulted_attribute_scan(
    source_root: Path, module_roots: tuple[Path, ...] = MODULE_ROOTS
) -> DefaultedAttributeScan:
    """Scan the configured modules for defaulted reads and their declared names."""
    declared_attributes = _declared_class_attributes(source_root)
    candidates: list[tuple[Path, int, str]] = []
    violations: list[tuple[Path, int, str]] = []
    roots: list[ModuleRootScan] = []
    for module_root in module_roots:
        root_path = source_root / module_root
        modules = _python_modules(source_root, module_root)
        root_candidate_count = 0
        for module in modules:
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
                root_candidate_count += 1
                candidates.append((module, node.lineno, attribute_name))
                if (
                    attribute_name not in declared_attributes
                    and attribute_name not in LEGITIMATELY_ABSENT_DEFAULTED_ATTRIBUTES
                ):
                    violations.append((module, node.lineno, attribute_name))
        roots.append(
            ModuleRootScan(
                root=module_root,
                exists=root_path.exists(),
                module_count=len(modules),
                candidate_count=root_candidate_count,
            )
        )
    return DefaultedAttributeScan(
        module_count=sum(root.module_count for root in roots),
        candidate_count=len(candidates),
        roots=roots,
        violations=violations,
    )


def _undeclared_defaulted_attributes(source_root: Path) -> list[tuple[Path, int, str]]:
    """Return violations for callers that only need the guard verdict."""
    return _defaulted_attribute_scan(source_root).violations


def test_configured_module_roots_have_coverage() -> None:
    """Every configured root must contribute both modules and defaulted reads."""
    scan = _defaulted_attribute_scan(REPOSITORY_ROOT)
    assert scan.module_count >= MINIMUM_SCANNED_MODULES
    assert scan.candidate_count >= MINIMUM_LITERAL_DEFAULTED_GETATTRS
    for root in scan.roots:
        assert root.exists, f"configured module root is missing: {root.root}"
        assert root.module_count > 0, (
            f"configured module root has no modules: {root.root}"
        )
        assert root.candidate_count > 0, (
            f"configured module root has no literal defaulted getattr calls: {root.root}"
        )


def test_defaulted_literal_attributes_are_declared() -> None:
    """A default may not turn an undeclared attribute read into empty evidence."""
    scan = _defaulted_attribute_scan(REPOSITORY_ROOT)

    violations = scan.violations
    rendered = "\n".join(
        f"{path.relative_to(REPOSITORY_ROOT)}:{line}: {attribute_name}"
        for path, line, attribute_name in violations
    )
    assert not violations, f"defaulted undeclared attributes:\n{rendered}"


def test_defaulted_literal_attribute_fixture_is_reported(tmp_path: Path) -> None:
    """The guard has a permanent counterexample independent of production code."""
    source_file = tmp_path / "imas_codex/standard_names/review/fixture.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text('getattr(subject, "fixture_missing", [])\n')

    scan = _defaulted_attribute_scan(tmp_path, (source_file.relative_to(tmp_path),))

    assert scan.candidate_count == 1
    assert scan.violations == [(source_file, 1, "fixture_missing")]

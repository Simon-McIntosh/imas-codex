"""Prevent default values from hiding reads of undeclared model attributes."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Package the empty-container aperture is derived from: every module it holds
# that reads an attribute through a literal-name getattr with an empty-container
# default is scanned, whether or not one of the roots below names it.
SOURCE_PACKAGE = Path("imas_codex")

# Extend this tuple when another standard-name surface needs the same guard.
MODULE_ROOTS = (
    Path("imas_codex/standard_names/review"),
    Path("imas_codex/standard_names/export.py"),
    Path("imas_codex/standard_names/loop.py"),
)

# Bare constructor spellings of an empty container default.
_EMPTY_CONTAINER_CALLS = frozenset({"dict", "frozenset", "list", "set", "tuple"})

# Module and distribution metadata are intentionally dynamic, not model fields.
LEGITIMATELY_ABSENT_DEFAULTED_ATTRIBUTES = {
    "__file__": "Module metadata is assigned by Python's import machinery.",
    "__version__": "Distribution metadata is supplied by the installed package.",
    "annotations": (
        "LinkML owns this attribute: a slot definition carries its lifecycle "
        "annotations on the slot object itself, so the name resolves against "
        "the linkml-runtime schema objects rather than a class in this package."
    ),
}

# Floors make a scan that sees nothing fail instead of passing silently. Each
# floor is a ratchet set to the count this tree measures: raise it when the tree
# grows, never lower it to make a failure go away.
MINIMUM_SCANNED_MODULES = 11
MINIMUM_LITERAL_DEFAULTED_GETATTRS = 30
MINIMUM_EMPTY_CONTAINER_MODULES = 9
MINIMUM_EMPTY_CONTAINER_GETATTRS = 30


@dataclass(frozen=True)
class DefaultedAttributeScan:
    """The aperture and violations reported by one static scan."""

    module_count: int
    candidate_count: int
    roots: list[ModuleRootScan]
    empty_container_module_count: int
    empty_container_candidate_count: int
    violations: list[tuple[Path, int, str]]


@dataclass(frozen=True)
class ModuleRootScan:
    """The source and candidate coverage contributed by one configured root."""

    root: Path
    module_count: int
    candidate_count: int


def _python_modules(source_root: Path, module_root: Path) -> list[Path]:
    """Return Python modules contributed by one explicit root.

    A root that is neither a directory nor a file is a configuration fault, so
    it raises naming the path rather than contributing nothing: an unreachable
    root must fail the scan instead of quietly shrinking its aperture.
    """
    candidate = source_root / module_root
    if candidate.is_dir():
        return sorted(candidate.rglob("*.py"))
    if candidate.is_file():
        return [candidate]
    raise FileNotFoundError(
        f"configured module root is neither a directory nor a file: {candidate}"
    )


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


def _empty_container_default(node: ast.expr) -> bool:
    """Whether a default value is an empty container literal or constructor call.

    This is the defect shape: an empty default makes an absent attribute
    indistinguishable from a genuinely empty one at the read site.
    """
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name):
        return False
    return (
        node.func.id in _EMPTY_CONTAINER_CALLS and not node.args and not node.keywords
    )


def _parse_module(module: Path) -> ast.Module:
    """Parse one module, naming the file in any syntax error it raises."""
    return ast.parse(module.read_text(), filename=str(module))


def _literal_defaulted_getattrs(tree: ast.Module) -> list[tuple[int, str, ast.expr]]:
    """Return (line, attribute name, default) for one tree's defaulted reads.

    Only a literal attribute name makes a static claim about an attribute a
    class could declare; a computed name cannot be resolved by this scan.
    """
    found: list[tuple[int, str, ast.expr]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "getattr" or len(node.args) != 3:
            continue
        attribute_node = node.args[1]
        if not isinstance(attribute_node, ast.Constant):
            continue
        if not isinstance(attribute_node.value, str):
            continue
        found.append((node.lineno, attribute_node.value, node.args[2]))
    return found


def _declared_class_attributes(tree: ast.Module) -> set[str]:
    """Collect fields and properties declared by the classes in one tree."""
    attributes: set[str] = set()
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
                member.targets if isinstance(member, ast.Assign) else [member.target]
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
    """Scan the configured roots and the tree-derived defect surface.

    Two apertures share one verdict. The declared roots hold every literal-name
    defaulted read of the named standard-name surfaces. Its sibling is derived
    from the source tree: every module carrying a read whose default is an empty
    container, which is the shape that hides an absent attribute. Both facts come
    from one parse of each module, because parsing the trees dominates the scan.
    """
    declared_attributes: set[str] = set()
    package_reads: dict[Path, list[tuple[int, str, ast.expr]]] = {}
    for module in sorted((source_root / SOURCE_PACKAGE).rglob("*.py")):
        tree = _parse_module(module)
        declared_attributes.update(_declared_class_attributes(tree))
        package_reads[module] = _literal_defaulted_getattrs(tree)

    candidates: dict[tuple[Path, int], str] = {}
    violations: dict[tuple[Path, int], str] = {}
    roots: list[ModuleRootScan] = []
    for module_root in module_roots:
        modules = _python_modules(source_root, module_root)
        root_candidate_count = 0
        for module in modules:
            reads = package_reads.get(module)
            if reads is None:
                reads = _literal_defaulted_getattrs(_parse_module(module))
            for line, attribute_name, _default in reads:
                root_candidate_count += 1
                candidates[(module, line)] = attribute_name
                if (
                    attribute_name not in declared_attributes
                    and attribute_name not in LEGITIMATELY_ABSENT_DEFAULTED_ATTRIBUTES
                ):
                    violations[(module, line)] = attribute_name
        roots.append(
            ModuleRootScan(
                root=module_root,
                module_count=len(modules),
                candidate_count=root_candidate_count,
            )
        )
    empty_container_modules = 0
    empty_container_candidates = 0
    for module in sorted(package_reads):
        empty_reads = [
            (line, attribute_name)
            for line, attribute_name, default in package_reads[module]
            if _empty_container_default(default)
        ]
        if not empty_reads:
            continue
        empty_container_modules += 1
        for line, attribute_name in empty_reads:
            empty_container_candidates += 1
            candidates[(module, line)] = attribute_name
            if (
                attribute_name not in declared_attributes
                and attribute_name not in LEGITIMATELY_ABSENT_DEFAULTED_ATTRIBUTES
            ):
                violations[(module, line)] = attribute_name

    return DefaultedAttributeScan(
        module_count=sum(root.module_count for root in roots),
        candidate_count=len(candidates),
        roots=roots,
        empty_container_module_count=empty_container_modules,
        empty_container_candidate_count=empty_container_candidates,
        violations=[
            (module, line, attribute_name)
            for (module, line), attribute_name in sorted(
                violations.items(), key=lambda entry: (str(entry[0][0]), entry[0][1])
            )
        ],
    )


def _undeclared_defaulted_attributes(source_root: Path) -> list[tuple[Path, int, str]]:
    """Return violations for callers that only need the guard verdict."""
    return _defaulted_attribute_scan(source_root).violations


def test_configured_module_roots_have_coverage() -> None:
    """Every configured root must contribute both modules and defaulted reads."""
    scan = _defaulted_attribute_scan(REPOSITORY_ROOT)
    assert scan.module_count >= MINIMUM_SCANNED_MODULES, (
        f"scanned {scan.module_count} modules, below the ratchet floor of "
        f"{MINIMUM_SCANNED_MODULES}: the floor is a ratchet recording the "
        "aperture this tree measures, so raise it when the tree grows and never "
        "lower it to make a failure go away"
    )
    assert scan.candidate_count >= MINIMUM_LITERAL_DEFAULTED_GETATTRS, (
        f"scanned {scan.candidate_count} literal defaulted getattr calls, below "
        f"the ratchet floor of {MINIMUM_LITERAL_DEFAULTED_GETATTRS}: the floor is "
        "a ratchet recording the aperture this tree measures, so raise it when "
        "the tree grows and never lower it to make a failure go away"
    )
    for root in scan.roots:
        assert root.module_count > 0, (
            f"configured module root has no modules: {root.root}"
        )
        assert root.candidate_count > 0, (
            f"configured module root has no literal defaulted getattr calls: {root.root}"
        )


def test_empty_container_defaulted_surface_is_covered() -> None:
    """The defect shape's aperture is derived from the tree, not from a root list."""
    scan = _defaulted_attribute_scan(REPOSITORY_ROOT)

    assert scan.empty_container_module_count >= MINIMUM_EMPTY_CONTAINER_MODULES, (
        f"scanned {scan.empty_container_module_count} modules carrying an "
        f"empty-container default, below the ratchet floor of "
        f"{MINIMUM_EMPTY_CONTAINER_MODULES}: the floor is a ratchet recording the "
        "defect surface this tree measures, so raise it when the tree grows and "
        "never lower it to make a failure go away"
    )
    assert scan.empty_container_candidate_count >= MINIMUM_EMPTY_CONTAINER_GETATTRS, (
        f"scanned {scan.empty_container_candidate_count} empty-container defaulted "
        f"getattr calls, below the ratchet floor of "
        f"{MINIMUM_EMPTY_CONTAINER_GETATTRS}: the floor is a ratchet recording the "
        "defect surface this tree measures, so raise it when the tree grows and "
        "never lower it to make a failure go away"
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


@pytest.mark.parametrize("case", ["absent", "special-file"])
def test_misspelled_module_root_raises(tmp_path: Path, case: str) -> None:
    """A configured root that resolves to nothing must fail, not shrink silently.

    The refusal names a path that is neither a directory nor a regular file, so
    both halves of that condition are driven, not just the absent half.
    """
    if case == "absent":
        configured_root = Path("imas_codex/standard_names/reviwe")
    else:
        configured_root = Path("imas_codex/standard_names/review/guard-probe")
        (tmp_path / configured_root).parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(tmp_path / configured_root)

    with pytest.raises(FileNotFoundError) as refusal:
        _defaulted_attribute_scan(tmp_path, (configured_root,))

    assert str(tmp_path / configured_root) in str(refusal.value)

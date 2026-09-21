# Static check: a three-argument getattr must name a declared attribute.
#
# getattr(obj, "name", default) is a fail-open read: the two-argument form
# raises AttributeError when the attribute is missing, but the three-argument
# form swallows the miss into the default and the caller proceeds as though it
# had read the value. A reviewer that spells an attribute its target class does
# not declare therefore loses data silently.
#
# This module walks the review package source with ast, resolves the object of
# every three-argument getattr whose attribute is a string literal to the class
# named by its annotation, and fails when that class and its bases declare no
# such field. Nothing is imported by the check: the class index is built from
# package source, so the check runs on a revision that does not resolve against
# the installed environment.
#
# Resolution is by annotation, the only sound static route to the class of a
# local name. An object with no annotation, or one annotated Any or a bare
# container, is reported unresolved rather than judged; the scan asserts a floor
# on how many sites it resolved, so a resolver that stops seeing classes cannot
# pass as a clean tree. A site whose class resolves but whose bases leave the
# package is reported unverifiable rather than a violation, because an unseen
# base is not evidence of an undeclared field.

from __future__ import annotations

import ast
import difflib
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "imas_codex"
REVIEW_SUBTREE = "standard_names/review"

# Measured on the base revision: five of eighteen sites resolve to an annotated
# class. The floor reddens when the review package loses a resolvable site, so
# an instrument that silently narrows cannot report a clean tree. A commit that
# legitimately removes a resolvable site lowers this floor alongside it.
MIN_RESOLVED_SITES = 5


@dataclass(frozen=True)
class Site:
    # One three-argument getattr whose attribute is a string literal.
    path: str
    module: str
    lineno: int
    object_expr: str
    attribute: str

    def key(self) -> str:
        return (
            f"{self.path}:{self.lineno} "
            f"getattr({self.object_expr}, {self.attribute!r}, ...)"
        )


@dataclass(frozen=True)
class Finding:
    site: Site
    outcome: str  # declared | undeclared | unresolved | unverifiable
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def _of(self, *outcomes: str) -> list[Finding]:
        return [f for f in self.findings if f.outcome in outcomes]

    @property
    def resolved(self) -> list[Finding]:
        return self._of("declared", "undeclared")

    @property
    def violations(self) -> list[Finding]:
        return self._of("undeclared")

    @property
    def unresolved(self) -> list[Finding]:
        return self._of("unresolved")

    @property
    def unverifiable(self) -> list[Finding]:
        return self._of("unverifiable")


class _ClassInfo:
    __slots__ = ("name", "node", "module", "bases")

    def __init__(self, name: str, node: ast.ClassDef, module: str) -> None:
        self.name = name
        self.node = node
        self.module = module
        self.bases = node.bases

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"


class _Index:
    def __init__(self) -> None:
        self.classes: dict[str, _ClassInfo] = {}
        self.imports: dict[str, dict[str, str]] = {}

    def find(self, dotted: str, module: str) -> _ClassInfo | None:
        for key in (dotted, f"{module}.{dotted}"):
            if key in self.classes:
                return self.classes[key]
        imported = self.imports.get(module, {}).get(dotted)
        if imported is not None:
            return self.classes.get(imported)
        return None


def _module_name(path: Path, package_root: Path, package_name: str) -> str:
    rel = path.relative_to(package_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([package_name, *parts])


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _import_map(tree: ast.Module, module: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            origin = node.module or ""
            if node.level:
                parts = module.split(".")
                tail = [origin] if origin else []
                origin = ".".join([*parts[: len(parts) - node.level], *tail])
            for alias in node.names:
                names[alias.asname or alias.name] = f"{origin}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.setdefault(alias.asname or alias.name.split(".")[0], alias.name)
    return names


@lru_cache(maxsize=4)
def _build_index(package_root: Path, package_name: str) -> _Index:
    index = _Index()
    for path in sorted(package_root.rglob("*.py")):
        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        module = _module_name(path, package_root, package_name)
        index.imports[module] = _import_map(tree, module)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                index.classes[f"{module}.{node.name}"] = _ClassInfo(
                    node.name, node, module
                )
    return index


@dataclass(frozen=True)
class _Fields:
    names: frozenset[str]
    bases_resolved: bool


def _assigned_names(target: ast.expr) -> set[str]:
    # A bare name is a class-body declaration; self.x in a method is an
    # attribute the class declares at runtime.
    if isinstance(target, ast.Name):
        return {target.id}
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return {target.attr}
    return set()


def _declared_fields(
    info: _ClassInfo, index: _Index, seen: frozenset[str] = frozenset()
) -> _Fields:
    if info.qualname in seen:
        return _Fields(frozenset(), True)
    seen = seen | {info.qualname}
    names: set[str] = set()
    bases_ok = True
    for stmt in info.node.body:
        if isinstance(stmt, ast.AnnAssign):
            names |= _assigned_names(stmt.target)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                names |= _assigned_names(target)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        names |= _assigned_names(target)
                elif isinstance(sub, ast.AnnAssign):
                    names |= _assigned_names(sub.target)
    for base in info.bases:
        base_info = _annotation_class(base, info.module, index)
        if base_info is None:
            if _framework_base(base, info.module, index):
                # A framework base declares no data fields of its own, so the
                # subclass body is the complete declaration. Its field set is
                # known rather than unknown, and a class whose fields are known
                # is judged instead of reported unverifiable.
                continue
            bases_ok = False
            continue
        child = _declared_fields(base_info, index, seen)
        names |= child.names
        bases_ok = bases_ok and child.bases_resolved
    return _Fields(frozenset(names), bases_ok)


def _annotation_class(
    node: ast.expr | None, module: str, index: _Index
) -> _ClassInfo | None:
    # An imported string annotation is only literal once parsed.
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, str):
            return None
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    if isinstance(node, ast.Subscript):
        # A container annotation names the element class, not the object's.
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        for side in (node.left, node.right):
            found = _annotation_class(side, module, index)
            if found is not None:
                return found
        return None
    dotted = _dotted(node)
    if dotted is None:
        return None
    return index.find(dotted, module)


# Framework bases whose declared-field contribution is known to be empty: the
# subclass body is the whole declaration. Any other base outside the package
# keeps its fields unknown, because an unseen base may declare them.
_FRAMEWORK_BASES = frozenset({"pydantic.BaseModel"})


def _framework_base(node: ast.expr, module: str, index: _Index) -> bool:
    """True when a base is a framework base contributing no declared fields."""
    dotted = _dotted(node)
    if dotted is None:
        return False
    return index.imports.get(module, {}).get(dotted, dotted) in _FRAMEWORK_BASES


def _scope_annotations(fn: ast.AST) -> dict[str, ast.expr]:
    annotations: dict[str, ast.expr] = {}
    args = fn.args
    named = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    for arg in named:
        if arg.annotation is not None:
            annotations[arg.arg] = arg.annotation
    for arg in (args.vararg, args.kwarg):
        if arg is not None and arg.annotation is not None:
            annotations[arg.arg] = arg.annotation
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            annotations.setdefault(stmt.target.id, stmt.annotation)
    return annotations


def _enclosing_functions(tree: ast.Module, lineno: int) -> list[ast.AST]:
    chain: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = child.end_lineno or child.lineno
                if child.lineno <= lineno <= end:
                    chain.append(child)
                    visit(child)
                    return
            visit(child)

    visit(tree)
    return chain


def _resolve_object(
    name: str, module: str, tree: ast.Module, lineno: int, index: _Index
) -> _ClassInfo | None:
    for fn in _enclosing_functions(tree, lineno):
        annotation = _scope_annotations(fn).get(name)
        if annotation is not None:
            found = _annotation_class(annotation, module, index)
            if found is not None:
                return found
    for stmt in tree.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
            found = _annotation_class(stmt.annotation, module, index)
            if found is not None:
                return found
    return None


def _judge(site: Site, obj_node: ast.expr, tree: ast.Module, index: _Index) -> Finding:
    if not isinstance(obj_node, ast.Name):
        return Finding(site, "unresolved", "object is not a local name")
    info = _resolve_object(obj_node.id, site.module, tree, site.lineno, index)
    if info is None:
        return Finding(site, "unresolved", "no annotation names an indexed class")
    fields = _declared_fields(info, index)
    if site.attribute in fields.names:
        return Finding(site, "declared", info.qualname)
    if not fields.bases_resolved:
        return Finding(site, "unverifiable", f"{info.qualname} has an unindexed base")
    near = difflib.get_close_matches(site.attribute, sorted(fields.names), n=3)
    hint = f" nearest declared: {', '.join(near)};" if near else ""
    return Finding(site, "undeclared", f"{info.qualname} declares no such field;{hint}")


def scan(
    scan_root: str | Path | None = None,
    index_root: str | Path | None = None,
    package_name: str = PACKAGE_NAME,
    subtree: str = REVIEW_SUBTREE,
) -> Report:
    # The class index may come from a different tree than the sites, so a
    # scan fixture can carry a mutated site while classes stay unmutated.
    scan_root = Path(scan_root) if scan_root is not None else REPO_ROOT
    index_root = Path(index_root) if index_root is not None else scan_root
    package_root = scan_root / package_name
    index = _build_index(index_root / package_name, package_name)
    report = Report()
    for path in sorted((package_root / subtree).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        module = _module_name(path, package_root, package_name)
        relative = str(path.relative_to(scan_root))
        for node in ast.walk(tree):
            if not _is_literal_getattr(node):
                continue
            site = Site(
                relative,
                module,
                node.lineno,
                ast.unparse(node.args[0]),
                node.args[1].value,
            )
            report.findings.append(_judge(site, node.args[0], tree, index))
    return report


def _is_literal_getattr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    )


def assert_declared_attribute_access(
    scan_root: str | Path | None = None,
    index_root: str | Path | None = None,
    min_resolved: int = MIN_RESOLVED_SITES,
) -> Report:
    report = scan(scan_root, index_root)
    if report.violations:
        lines = [
            "three-argument getattr names an attribute its class does not declare:"
        ]
        lines += [f"  {f.site.key()} -> {f.detail}" for f in report.violations]
        raise AssertionError("\n".join(lines))
    resolved = report.resolved
    if len(resolved) < min_resolved:
        lines = [
            f"the scan resolved {len(resolved)} sites, below the floor "
            f"{min_resolved}: the instrument has narrowed. Unresolved sites:"
        ]
        lines += [f"  {f.site.key()} -> {f.detail}" for f in report.unresolved]
        raise AssertionError("\n".join(lines))
    return report


_STATE_FIXTURE = """
class Base:
    inherited: int = 0


class Target(Base):
    target: str = "names"

    def __post_init__(self) -> None:
        self.runtime = 1
"""


def _fixture(tmp_path: Path, body: str, extra: str = "") -> Path:
    pkg = tmp_path / "imas_codex"
    review = pkg / "standard_names" / "review"
    review.mkdir(parents=True)
    for init in (pkg / "__init__.py", pkg / "standard_names" / "__init__.py"):
        init.write_text("")
    (review / "__init__.py").write_text("")
    (review / "state.py").write_text(_STATE_FIXTURE)
    (review / "probe.py").write_text(
        "from __future__ import annotations\n"
        "from imas_codex.standard_names.review.state import Target\n\n"
        f"{extra}\n"
        "def worker(state: Target) -> None:\n"
        f"{body}"
    )
    return tmp_path


@pytest.mark.timeout(300)
def test_three_argument_getattr_names_a_declared_attribute() -> None:
    report = assert_declared_attribute_access()
    assert report.violations == []


def test_undeclared_attribute_is_refused(tmp_path: Path) -> None:
    root = _fixture(tmp_path, '    getattr(state, "no_such_field", None)\n')
    with pytest.raises(AssertionError) as excinfo:
        assert_declared_attribute_access(root, min_resolved=0)
    message = str(excinfo.value)
    assert "no_such_field" in message
    assert "Target declares no such field" in message


def test_declared_attribute_is_accepted(tmp_path: Path) -> None:
    root = _fixture(tmp_path, '    getattr(state, "target", None)\n')
    report = assert_declared_attribute_access(root, min_resolved=0)
    assert len(report.violations) == 0
    assert len(report.resolved) == 1


def test_inherited_attribute_is_accepted(tmp_path: Path) -> None:
    root = _fixture(tmp_path, '    getattr(state, "inherited", None)\n')
    report = assert_declared_attribute_access(root, min_resolved=0)
    assert len(report.resolved) == 1


def test_attribute_declared_in_a_method_is_accepted(tmp_path: Path) -> None:
    root = _fixture(tmp_path, '    getattr(state, "runtime", None)\n')
    report = assert_declared_attribute_access(root, min_resolved=0)
    assert len(report.resolved) == 1


def test_unannotated_object_is_reported_not_refused(tmp_path: Path) -> None:
    root = _fixture(tmp_path, '    getattr(obj, "anything", None)\n')
    report = assert_declared_attribute_access(root, min_resolved=0)
    assert report.unresolved
    assert report.violations == []


def test_base_outside_the_index_is_unverifiable(tmp_path: Path) -> None:
    root = _fixture(
        tmp_path,
        '    getattr(state, "target", None)\n',
        extra=(
            "class Local(UnknownBase):\n"
            "    pass\n\n"
            "def other(state2: Local) -> None:\n"
            '    getattr(state2, "missing", None)\n'
        ),
    )
    report = assert_declared_attribute_access(root, min_resolved=0)
    assert len(report.unverifiable) == 1
    assert len(report.resolved) == 1
    assert report.violations == []


def test_aperture_floor_refuses_a_scan_that_resolves_nothing(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path, '    getattr(obj, "anything", None)\n')
    with pytest.raises(AssertionError) as excinfo:
        assert_declared_attribute_access(root, min_resolved=1)
    assert "below the floor 1" in str(excinfo.value)


def _scratch_review_tree(tmp_path: Path, *, old: str, new: str) -> Path:
    """Copy the review package and replace one exact source fragment in it."""
    package = tmp_path / PACKAGE_NAME
    shutil.copytree(REPO_ROOT / PACKAGE_NAME / REVIEW_SUBTREE, package / REVIEW_SUBTREE)
    target = package / REVIEW_SUBTREE / "pipeline.py"
    text = target.read_text()
    assert old in text, "the fragment this control mutates is no longer in the source"
    target.write_text(text.replace(old, new, 1))
    return tmp_path


def test_a_reintroduced_findings_getattr_is_refused(tmp_path: Path) -> None:
    # The check exists because a three-argument getattr on the audit report's
    # ``findings`` returned the empty default on every call and lost the whole
    # Layer 1 layer. It must refuse that form at that parameter, which it cannot
    # do while the parameter is annotated Any.
    root = _scratch_review_tree(
        tmp_path,
        old="    findings: list[str] = []\n",
        new='    findings: list[str] = getattr(audit_report, "findings", [])\n',
    )
    with pytest.raises(AssertionError) as excinfo:
        assert_declared_attribute_access(root, index_root=REPO_ROOT, min_resolved=0)
    message = str(excinfo.value)
    assert "getattr(audit_report, 'findings', ...)" in message
    assert "AuditReport declares no such field" in message

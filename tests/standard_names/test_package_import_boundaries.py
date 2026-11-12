"""Keep the graph and worker packages free of sibling dependencies."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import pytest

# The override lets the rule's own negative control point it at a mutated copy
# of the package without touching the working tree.
PACKAGE_ROOT = Path(
    os.environ.get("SN_STANDARD_NAMES_ROOT")
    or Path(__file__).resolve().parents[2] / "imas_codex/standard_names"
)
PACKAGES = ("graph_store", "worker_tasks")

# Modules that sit directly in the package, as distinct from its subpackages.
TOP_LEVEL_MODULES = frozenset(
    path.stem for path in PACKAGE_ROOT.glob("*.py") if path.stem != "__init__"
)

# Sites that still import a private name from another top-level module. Each
# entry names the importing module, the line at the census that recorded it, and
# the private name. Matching is on (module, name) so unrelated edits that shift
# line numbers do not fail the rule; the list only shrinks, and an entry whose
# site has been closed fails as stale.
PRIVATE_NAME_EXEMPTIONS: frozenset[tuple[str, int, str]] = frozenset(
    {
        ("attachment_audit.py", 181, "_derived_parent_cypher"),
        ("attachment_audit.py", 1186, "_TransactionQuery"),
        ("benchmark_roles.py", 631, "_resolve_name"),
        ("campaign_pricing.py", 361, "_build_enum_lists"),
        ("catalog_reconcile.py", 31, "_normalize_field"),
        ("docs_gates.py", 15, "_dimension_container"),
        ("docs_gates.py", 15, "_dimensions_overlap"),
        ("docs_gates.py", 15, "_unit_dimensions"),
        ("edit.py", 98, "_isn_round_trip_ok"),
        ("export.py", 682, "_live_child_cypher"),
        ("harmonize.py", 40, "_pattern_family_rows_query"),
        ("protection.py", 185, "_pull_request_call"),
        ("protection.py", 185, "_pull_request_state"),
        ("provenance_lifecycle.py", 27, "_load_signed_stale_source_rows"),
        ("provenance_lifecycle.py", 638, "_TransactionQueryAdapter"),
        ("provenance_lifecycle.py", 1153, "_TransactionQueryAdapter"),
        ("provenance_lifecycle.py", 1394, "_TransactionQueryAdapter"),
        ("provenance_rebuild.py", 265, "_TransactionQueryAdapter"),
        ("release_notes.py", 151, "_git"),
        ("release_notes.py", 151, "_parse_entries"),
        ("signed_manifest.py", 7311, "_ATTACH_QUERY"),
        ("signed_manifest.py", 7311, "_DETACH_QUERY"),
        ("signed_manifest.py", 7311, "_HISTORICAL_NAME_STAGES"),
    }
)


def _sibling_module(node: ast.ImportFrom, stem: str) -> str | None:
    """Return the top-level sibling module *node* imports from, if any."""
    if node.level == 0:
        parts = (node.module or "").split(".")
        if len(parts) == 3 and parts[:2] == ["imas_codex", "standard_names"]:
            target = parts[2]
        else:
            return None
    elif node.level == 1:
        target = node.module or ""
        if not target:
            return None
    else:
        return None
    if target in TOP_LEVEL_MODULES and target != stem:
        return target
    return None


def _private_name_imports(module_path: Path) -> list[tuple[int, str]]:
    """Private names *module_path* imports from another top-level module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), str(module_path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if _sibling_module(node, module_path.stem) is None:
            continue
        found.extend(
            (node.lineno, alias.name)
            for alias in node.names
            if alias.name.startswith("_") and not alias.name.startswith("__")
        )
    return found


@pytest.mark.timeout(120)
def test_no_private_name_imports_between_top_level_modules() -> None:
    """A top-level module exposes a public name instead of a sibling's private one."""
    allowed = {(name, private) for name, _line, private in PRIVATE_NAME_EXEMPTIONS}
    violations: list[str] = []
    live: set[tuple[str, str]] = set()
    for module_path in sorted(PACKAGE_ROOT.glob("*.py")):
        for lineno, private in _private_name_imports(module_path):
            key = (module_path.name, private)
            live.add(key)
            if key not in allowed:
                violations.append(
                    f"{module_path.name}:{lineno}: imports private name {private!r} "
                    "from a sibling top-level module"
                )
    assert not violations, "private-name imports:\n" + "\n".join(violations)
    stale = allowed - live
    assert not stale, f"stale private-name exemptions: {sorted(stale)}"


def _package_modules(package: str) -> tuple[Path, ...]:
    return tuple(sorted((PACKAGE_ROOT / package).glob("*.py")))


@pytest.mark.parametrize(
    ("package", "module_path"),
    [(package, path) for package in PACKAGES for path in _package_modules(package)],
)
def test_package_imports_only_shared_module(package: str, module_path: Path) -> None:
    """Domain modules may import their package's shared foundation only."""
    allowed = "graph_foundation" if package == "graph_store" else "worker_common"
    siblings = {path.stem for path in _package_modules(package)} - {
        "__init__",
        module_path.stem,
        allowed,
    }
    tree = ast.parse(module_path.read_text(encoding="utf-8"), str(module_path))
    forbidden: list[tuple[int, str]] = []
    prefix = f"imas_codex.standard_names.{package}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if alias.name.startswith(prefix + ".") and parts[3] in siblings:
                    forbidden.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level == 1 and base in siblings:
                forbidden.append((node.lineno, base))
            elif node.level == 1 and not base:
                forbidden.extend(
                    (node.lineno, alias.name)
                    for alias in node.names
                    if alias.name in siblings
                )
            elif node.level == 2 and (
                base == package or base.startswith(package + ".")
            ):
                target = base.removeprefix(package).lstrip(".").split(".")[0]
                if target in siblings:
                    forbidden.append((node.lineno, base))
                elif not target:
                    forbidden.extend(
                        (node.lineno, alias.name)
                        for alias in node.names
                        if alias.name in siblings
                    )
            elif node.level == 0 and base.startswith(prefix + "."):
                if base.removeprefix(prefix + ".").split(".")[0] in siblings:
                    forbidden.append((node.lineno, base))
            elif node.level == 0 and base == prefix:
                forbidden.extend(
                    (node.lineno, alias.name)
                    for alias in node.names
                    if alias.name in siblings
                )
    assert not forbidden, f"{module_path}: forbidden sibling imports {forbidden}"


def test_unwired_graph_services_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone domain import reports missing wiring at first use."""
    from imas_codex.standard_names.graph_store import graph_audits, graph_foundation

    monkeypatch.setattr(graph_foundation, "_services", None)
    with pytest.raises(RuntimeError, match="Graph services are not wired"):
        graph_foundation.get_services()
    with pytest.raises(RuntimeError, match="Graph services are not wired"):
        graph_audits.promote_stranded_reviewed(dry_run=True)


def test_stranded_docs_predicate_is_complete() -> None:
    """The Cypher constant needs no entrypoint rewrite to become valid."""
    from imas_codex.standard_names.graph_store import graph_audits

    predicate = graph_audits.PROMOTE_STRANDED_DOCS_WHERE
    assert "{docs_review_eligibility_where}" not in predicate
    assert hashlib.sha256(predicate.encode()).hexdigest() == (
        "34a804254c71f1dab1148e22a7d282f9aa92daee0814d32da22d13e67ae7f9f7"
    )

"""Keep the graph and worker packages free of sibling dependencies."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "imas_codex/standard_names"
PACKAGES = ("graph_store", "worker_tasks")


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
    from imas_codex.standard_names.graph_store import graph_foundation

    monkeypatch.setattr(graph_foundation, "_services", None)
    with pytest.raises(RuntimeError, match="Graph services are not wired"):
        graph_foundation.get_services()


def test_stranded_docs_predicate_is_complete() -> None:
    """The Cypher constant needs no entrypoint rewrite to become valid."""
    from imas_codex.standard_names.graph_store import graph_audits

    predicate = graph_audits.PROMOTE_STRANDED_DOCS_WHERE
    assert "{docs_review_eligibility_where}" not in predicate
    assert hashlib.sha256(predicate.encode()).hexdigest() == (
        "34a804254c71f1dab1148e22a7d282f9aa92daee0814d32da22d13e67ae7f9f7"
    )

"""Whole-package source sets for structural scans of the standard-names layers.

The worker and graph layers each live in a package root module plus a
module-per-task subpackage (``workers.py`` + ``worker_tasks/``,
``graph_ops.py`` + ``graph_store/``), and code continues to move between the
two as the split proceeds. A scan that reads only the root module stops seeing
the code it was written to check, so every structural scan reads the whole
owning package through this module instead of naming one file.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "imas_codex" / "standard_names"

_WORKER_ROOT = _PACKAGE_DIR / "workers.py"
_WORKER_SUBPACKAGE = _PACKAGE_DIR / "worker_tasks"
_GRAPH_ROOT = _PACKAGE_DIR / "graph_ops.py"
_GRAPH_SUBPACKAGE = _PACKAGE_DIR / "graph_store"


def _read_split_modules() -> bool:
    """Whether scans read the split subpackages beside the root modules."""
    return True


def _module_paths(root: Path, subpackage: Path) -> tuple[Path, ...]:
    if not _read_split_modules():
        return (root,)
    return (root, *sorted(subpackage.glob("*.py")))


def worker_module_paths() -> tuple[Path, ...]:
    """Every module holding worker code: ``workers.py`` and ``worker_tasks/*``."""
    return _module_paths(_WORKER_ROOT, _WORKER_SUBPACKAGE)


def graph_module_paths() -> tuple[Path, ...]:
    """Every module holding graph code: ``graph_ops.py`` and ``graph_store/*``."""
    return _module_paths(_GRAPH_ROOT, _GRAPH_SUBPACKAGE)


def _sources(paths: tuple[Path, ...]) -> tuple[tuple[Path, str], ...]:
    return tuple((path, path.read_text(encoding="utf-8")) for path in paths)


def worker_sources() -> tuple[tuple[Path, str], ...]:
    """``(path, source)`` for every worker module, for per-module line scans."""
    return _sources(worker_module_paths())


def graph_sources() -> tuple[tuple[Path, str], ...]:
    """``(path, source)`` for every graph module, for per-module line scans."""
    return _sources(graph_module_paths())


def worker_trees() -> tuple[tuple[Path, ast.Module], ...]:
    """``(path, ast)`` for every worker module, for per-module AST scans."""
    return tuple(
        (path, ast.parse(source, filename=str(path)))
        for path, source in worker_sources()
    )


def graph_trees() -> tuple[tuple[Path, ast.Module], ...]:
    """``(path, ast)`` for every graph module, for per-module AST scans."""
    return tuple(
        (path, ast.parse(source, filename=str(path)))
        for path, source in graph_sources()
    )


def worker_source_text() -> str:
    """All worker module sources joined, for substring scans."""
    return "\n".join(source for _path, source in worker_sources())


def graph_source_text() -> str:
    """All graph module sources joined, for substring scans."""
    return "\n".join(source for _path, source in graph_sources())

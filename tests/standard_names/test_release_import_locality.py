"""Verify that circular imports between catalog_release and release_notes are avoided."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


def _module_level_release_notes_imports(path: Path) -> list[ast.ImportFrom]:
    """Return imports of release_notes declared directly in the module body."""
    tree = ast.parse(path.read_text())
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.rsplit(".", maxsplit=1)[-1] == "release_notes"
    ]


def _assert_no_module_level_release_notes_imports(path: Path) -> None:
    """Fail if the module imports release_notes while it initializes."""
    module_level_imports = _module_level_release_notes_imports(path)
    assert len(module_level_imports) == 0, (
        f"Found {len(module_level_imports)} module-level imports from release_notes in "
        f"{path.name}. All imports from release_notes must be deferred to function-local "
        "scope to avoid circular dependency."
    )


def test_catalog_release_no_module_level_release_notes_imports() -> None:
    """Assert that catalog_release.py has no module-level imports from release_notes.

    release_notes imports _github_slug and _run_git from catalog_release at module
    level. If catalog_release also imported from release_notes at module level, it
    would create a circular dependency. This test ensures that all imports from
    release_notes in catalog_release remain function-local.
    """
    catalog_release_path = (
        Path(__file__).parent.parent.parent
        / "iter_codex"
        / "standard_names"
        / "catalog_release.py"
    )
    _assert_no_module_level_release_notes_imports(catalog_release_path)


@pytest.mark.parametrize(
    "module_order",
    (
        (
            "iter_codex.standard_names.catalog_release",
            "iter_codex.standard_names.release_notes",
        ),
        (
            "iter_codex.standard_names.release_notes",
            "iter_codex.standard_names.catalog_release",
        ),
    ),
)
def test_both_import_orders_load(module_order: tuple[str, str]) -> None:
    """Verify that both import orders work: catalog_release before release_notes and vice versa.

    This ensures that the function-local imports in catalog_release successfully
    break the circular dependency and both modules can be loaded in either order.
    """
    first_module, second_module = module_order
    import_script = f"""\
import importlib

first = importlib.import_module({first_module!r})
second = importlib.import_module({second_module!r})
catalog_release = importlib.import_module("iter_codex.standard_names.catalog_release")
release_notes = importlib.import_module("iter_codex.standard_names.release_notes")

assert first is not None
assert second is not None
assert hasattr(catalog_release, "_run_git")
assert hasattr(catalog_release, "_github_slug")
assert hasattr(release_notes, "build_pr_notes")
"""

    result = subprocess.run(
        [sys.executable, "-c", import_script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Failed to import {first_module} then {second_module} in a clean "
        f"interpreter:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

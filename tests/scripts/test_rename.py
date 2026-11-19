"""Tests for the repeatable rename and its closing search."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rename.__main__ import main
from scripts.rename.apply import apply_rename, tree_digest
from scripts.rename.check import check_tree
from scripts.rename.cutover import check_cutover

KEPT_LIST = {
    "version": 1,
    "old_names": ["imas_codex", "imas-codex", "IMAS_CODEX", "IMAS Codex"],
    "kept_paths": [
        {
            "id": "evidence",
            "path": "docs/evidence",
            "surface": "evidence",
            "disposition": "kept",
            "reason": "historical evidence keeps its bytes",
            "min_hits": 1,
        }
    ],
    "kept_names": [
        {
            "id": "standard-names-import",
            "marker": "imas_standard_names",
            "reason": "the grammar library names the IMAS standard names",
        }
    ],
}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature tree with one covered package, a kept path and a kept name."""
    root = tmp_path / "tree"
    (root / "imas_codex").mkdir(parents=True)
    (root / "imas_codex" / "__init__.py").write_text(
        '"""imas_codex: a package.\n\nimport imas_standard_names\n'
        "from imas_codex.remote.executor import run\n"
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "imas-codex"\ndependencies = ["imas-standard-names==0.11.1"]\n'
        'entry = "imas_codex.cli:main"\n'
    )
    (root / ".mcp.json").write_text(
        '{"mcpServers": {"imas-cx": {"args": ["imas-codex"]}}}\n'
    )
    (root / "docs" / "evidence").mkdir(parents=True)
    (root / "docs" / "evidence" / "run.md").write_text(
        "The imas_codex run measured 3 surfaces.\n"
    )
    return root


@pytest.fixture
def kept_list(tmp_path: Path) -> Path:
    path = tmp_path / "kept_names.json"
    path.write_text(json.dumps(KEPT_LIST))
    return path


def test_apply_renames_text_paths_and_leaves_kept_bytes(
    tree: Path, kept_list: Path
) -> None:
    evidence_before = (tree / "docs" / "evidence" / "run.md").read_bytes()

    report = apply_rename(tree, kept_list=kept_list)

    assert (tree / "iter_codex" / "__init__.py").is_file()
    assert not (tree / "imas_codex").exists()
    module = (tree / "iter_codex" / "__init__.py").read_text()
    assert "iter_codex.remote.executor" in module
    assert "imas_standard_names" in module, "kept name must survive"
    assert '"iter-codex"' in (tree / "pyproject.toml").read_text()
    assert "iter-cx" in (tree / ".mcp.json").read_text()
    assert (tree / "docs" / "evidence" / "run.md").read_bytes() == evidence_before
    assert report.changed_file_count == 3
    assert report.renamed_count == 1
    assert report.old_name_hits_after == 1, "only the kept evidence hit remains"
    assert report.kept_hits == 1


def test_apply_is_idempotent(tree: Path, kept_list: Path) -> None:
    apply_rename(tree, kept_list=kept_list)
    first = tree_digest(tree)

    second = apply_rename(tree, kept_list=kept_list)

    assert second.changed_file_count == 0
    assert second.renamed_count == 0
    assert tree_digest(tree) == first


def test_dry_run_writes_nothing(tree: Path, kept_list: Path) -> None:
    before = tree_digest(tree)

    report = apply_rename(tree, kept_list=kept_list, dry_run=True)

    assert report.changed_file_count == 3
    assert tree_digest(tree) == before


def test_check_accepts_a_renamed_tree(tree: Path, kept_list: Path) -> None:
    apply_rename(tree, kept_list=kept_list)

    report = check_tree(tree, kept_list=kept_list)

    assert report.ok, report.violations
    assert report.leftover == {}
    assert report.kept_hits == 1
    assert report.kept_names == {"standard-names-import": 1}


def test_check_finds_a_planted_old_name(tree: Path, kept_list: Path) -> None:
    apply_rename(tree, kept_list=kept_list)
    planted = tree / "iter_codex" / "planted.py"
    planted.write_text("import imas_codex.cli\n")

    report = check_tree(tree, kept_list=kept_list)

    assert not report.ok
    leftovers = [v for v in report.violations if v.kind == "leftover"]
    assert [v.detail.split(":")[0] for v in leftovers] == ["iter_codex/planted.py"]


def test_check_reports_a_kept_name_altered_by_hand(tree: Path, kept_list: Path) -> None:
    apply_rename(tree, kept_list=kept_list)
    module = tree / "iter_codex" / "__init__.py"
    module.write_text(
        module.read_text().replace("imas_standard_names", "iter_standard_names")
    )

    report = check_tree(tree, kept_list=kept_list)

    assert not report.ok
    kinds = {v.kind for v in report.violations}
    assert "kept-name-missing" in kinds


def test_check_reports_a_kept_path_altered_by_hand(tree: Path, kept_list: Path) -> None:
    apply_rename(tree, kept_list=kept_list)
    evidence = tree / "docs" / "evidence" / "run.md"
    evidence.write_text(evidence.read_text().replace("imas_codex", "iter_codex"))

    report = check_tree(tree, kept_list=kept_list)

    assert not report.ok
    assert any(v.kind == "kept-path-altered" for v in report.violations)


def test_check_reports_a_kept_file_altered_without_losing_a_hit(
    tree: Path, kept_list: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "apply-report.json"
    assert (
        main(
            [
                "apply",
                "--tree",
                str(tree),
                "--kept-list",
                str(kept_list),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    receipt = json.loads(report_path.read_text())
    assert receipt["kept_altered"] == []

    evidence = tree / "docs" / "evidence" / "run.md"
    evidence.write_text(evidence.read_text() + "\nreworded by hand, no old name\n")

    # The hit floor is unchanged, so only the recorded bytes can see this.
    assert check_tree(tree, kept_list=kept_list).ok
    report = check_tree(tree, kept_list=kept_list, receipt=report_path)

    assert not report.ok
    altered = [v for v in report.violations if v.kind == "kept-path-altered"]
    assert [v.detail.split(" ")[0] for v in altered] == ["docs/evidence"]
    assert (
        main(
            [
                "check",
                "--tree",
                str(tree),
                "--kept-list",
                str(kept_list),
                "--receipt",
                str(report_path),
            ]
        )
        == 1
    )


def test_cli_exit_status_reflects_the_violations(tree: Path, kept_list: Path) -> None:
    apply_rename(tree, kept_list=kept_list)
    assert main(["check", "--tree", str(tree), "--kept-list", str(kept_list)]) == 0

    (tree / "iter_codex" / "planted.py").write_text("imas_codex\n")

    assert main(["check", "--tree", str(tree), "--kept-list", str(kept_list)]) == 1


def test_cutover_accounts_for_every_inventory_surface() -> None:
    report = check_cutover()

    assert report.ok, report.violations
    data = report.as_dict()
    assert data["inventory_count"] >= 16
    assert data["unaccounted"] == []


def test_cutover_reports_an_unaccounted_surface(tmp_path: Path) -> None:
    data = json.loads(Path("scripts/rename/cutover.json").read_text())
    data["inventory"].append({"id": "an-uncovered-surface", "source": "planted"})
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("an-uncovered-surface" in v for v in report.violations)

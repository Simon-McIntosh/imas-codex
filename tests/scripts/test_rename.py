"""Tests for the repeatable rename and its closing search."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.rename.__main__ import main
from scripts.rename.apply import apply_rename, tree_digest
from scripts.rename.check import check_tree
from scripts.rename.cutover import check_cutover

INVENTORY = Path(
    "docs/evidence/fragments/codac-uda-signals/current-name-consumer-inventory.html"
)
SURFACE_TABLES = (("-results", "-commands"), ("-preserve", "-recommendations"))


def inventory_labels(*, surfaces_only: bool = True) -> set[str]:
    """The surface labels the consumer inventory document carries.

    Two tables name the surfaces: the measured snapshot and the scientific
    names to retain. Their rows open with a plain ``<td>`` label, while the
    command and item tables open with ``<code>``, so the label pattern is what
    separates a surface from a receipt row. With ``surfaces_only`` false, the
    decision table is included as well, since cutover rows that are not a
    collector surface cite a decision instead.
    """
    text = INVENTORY.read_text()
    blocks = [
        text[text.index(start) : text.index(end)] for start, end in SURFACE_TABLES
    ]
    if not surfaces_only:
        blocks.append(text[text.index("-recommendations") :])
    return set(re.findall(r"<tr><td>([^<]+)</td>", "".join(blocks)))


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


def test_a_persisted_identifier_in_a_kept_path_survives_the_rename(
    tree: Path, kept_list: Path
) -> None:
    evidence = tree / "docs" / "evidence" / "run.md"
    evidence.write_text(
        evidence.read_text() + '{"schema": "imas-codex.repair-authority.v1"}\n'
    )
    module = tree / "imas_codex" / "__init__.py"
    module.write_text(
        module.read_text() + 'SCHEMA = "imas-codex.repair-authority.v1"\n'
    )

    report = apply_rename(tree, kept_list=kept_list)

    kept_module = tree / "iter_codex" / "__init__.py"
    assert 'SCHEMA = "imas-codex.repair-authority.v1"' in kept_module.read_text()
    assert report.kept_identifiers == {"imas-codex.repair-authority.v1": 1}
    renamed = check_tree(tree, kept_list=kept_list)
    assert renamed.ok
    assert renamed.leftover == {}
    assert renamed.kept_identifiers["imas-codex.repair-authority.v1"]["covered"] == 1


def test_a_module_path_in_evidence_does_not_protect_an_import(
    tree: Path, kept_list: Path
) -> None:
    evidence = tree / "docs" / "evidence" / "run.md"
    evidence.write_text(
        evidence.read_text() + "the inventory ran imas_codex.remote.executor\n"
    )
    hook = tree / "hatch_build_hooks.py"
    hook.write_text(
        "from imas_codex.resource_path_accessor import ResourcePathAccessor\n"
    )

    report = apply_rename(tree, kept_list=kept_list)

    assert report.kept_identifiers == {}
    assert "from iter_codex.resource_path_accessor import" in hook.read_text()


def test_cli_exit_status_reflects_the_violations(tree: Path, kept_list: Path) -> None:
    apply_rename(tree, kept_list=kept_list)
    assert main(["check", "--tree", str(tree), "--kept-list", str(kept_list)]) == 0

    (tree / "iter_codex" / "planted.py").write_text("imas_codex\n")

    assert main(["check", "--tree", str(tree), "--kept-list", str(kept_list)]) == 1


def test_check_refuses_an_empty_scan(tmp_path: Path, kept_list: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    report = check_tree(empty, kept_list=kept_list)

    assert report.files_scanned == 0
    assert "empty-scan" in {v.kind for v in report.violations}
    assert main(["check", "--tree", str(empty), "--kept-list", str(kept_list)]) == 1


def test_check_reports_a_blind_kept_search(
    tree: Path, kept_list: Path, tmp_path: Path
) -> None:
    apply_rename(tree, kept_list=kept_list)
    (tree / "docs" / "evidence" / "run.md").write_text(
        "renamed evidence, no old name\n"
    )
    data = json.loads(kept_list.read_text())
    data["kept_paths"][0]["min_hits"] = 0
    blind = tmp_path / "blind.json"
    blind.write_text(json.dumps(data))

    report = check_tree(tree, kept_list=blind)

    assert report.files_scanned > 0
    assert report.kept_hits == 0
    assert "kept-search-blind" in {v.kind for v in report.violations}


def test_check_refuses_a_receipt_that_unaccounts_a_kept_path(
    tree: Path, kept_list: Path, tmp_path: Path
) -> None:
    applied = apply_rename(tree, kept_list=kept_list)
    data = {"kept_digests": dict(applied.kept_digests)}
    data["kept_digests"].pop("evidence")
    stripped = tmp_path / "stripped-receipt.json"
    stripped.write_text(json.dumps(data))

    report = check_tree(tree, kept_list=kept_list, receipt=stripped)

    assert "kept-path-unaccounted" in {v.kind for v in report.violations}


def test_check_reconciles_hits_kept_protected_and_leftover(
    tree: Path, kept_list: Path
) -> None:
    (tree / "docs" / "evidence" / "run.md").write_text(
        "The imas_codex run measured 3 surfaces.\n"
        '{"schema": "imas-codex.authority.v1"}\n'
    )
    (tree / "schemas.yaml").write_text('{"schema": "imas-codex.authority.v1"}\n')

    report = check_tree(tree, kept_list=kept_list)
    data = report.as_dict()

    assert data["protected"] == 1
    assert data["reconciled"], data
    assert data["old_name_hits"] == (
        data["kept_hits"] + data["protected"] + data["leftover_count"]
    )


def test_check_refuses_totals_that_do_not_reconcile(
    tree: Path, kept_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tree / "docs" / "evidence" / "run.md").write_text(
        '{"schema": "imas-codex.authority.v1"}\n'
    )
    (tree / "schemas.yaml").write_text('{"schema": "imas-codex.authority.v1"}\n')
    # A pattern narrower than the identifiers it counts cannot see the
    # protected occurrences, so the totals stop adding up; the report must say so
    # rather than print three numbers that look plausible apart.
    monkeypatch.setattr(
        "scripts.rename.check.OLD_NAME_PATTERN", re.compile(r"imas_codex\.never")
    )

    report = check_tree(tree, kept_list=kept_list)

    assert report.protected == 1
    assert report.old_name_hits == 0
    assert "counts-unreconciled" in {v.kind for v in report.violations}
    assert not report.as_dict()["reconciled"]


def label_disagreements(data: dict) -> tuple[set[str], set[str]]:
    """The document labels no row claims, and the labels no document carries."""
    claimed = {
        row.get("document_label")
        for row in data["inventory"]
        if row.get("origin") == "inventory"
    }
    surfaces = inventory_labels()
    return surfaces - claimed, claimed - inventory_labels(surfaces_only=False)


def test_cutover_accounts_for_every_inventory_surface() -> None:
    data = json.loads(Path("scripts/rename/cutover.json").read_text())
    unclaimed, invented = label_disagreements(data)
    claimed_brief = {row["id"] for row in data["inventory"] if row["origin"] == "brief"}

    assert unclaimed == set()
    assert invented == set()
    assert claimed_brief == {"worktrees", "docs-state", "gate-scripts", "node-briefs"}

    report = check_cutover()

    assert report.ok, report.violations
    assert report.as_dict()["unaccounted"] == []


def test_cutover_rejects_a_label_the_document_does_not_carry(tmp_path: Path) -> None:
    data = json.loads(Path("scripts/rename/cutover.json").read_text())
    row = next(r for r in data["inventory"] if r["origin"] == "inventory")
    row["document_label"] = "Invented surface"
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    unclaimed, invented = label_disagreements(data)

    assert "Invented surface" in invented
    assert "Installed units" in unclaimed


def test_cutover_refuses_a_row_without_an_origin(tmp_path: Path) -> None:
    data = json.loads(Path("scripts/rename/cutover.json").read_text())
    del data["inventory"][0]["origin"]
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("has origin None" in v for v in report.violations)


def test_cutover_refuses_an_inventory_row_without_a_label(tmp_path: Path) -> None:
    data = json.loads(Path("scripts/rename/cutover.json").read_text())
    row = next(r for r in data["inventory"] if r["origin"] == "inventory")
    del row["document_label"]
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("names no label" in v for v in report.violations)


def test_cutover_reports_an_unaccounted_surface(tmp_path: Path) -> None:
    data = json.loads(Path("scripts/rename/cutover.json").read_text())
    data["inventory"].append({"id": "an-uncovered-surface", "source": "planted"})
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("an-uncovered-surface" in v for v in report.violations)

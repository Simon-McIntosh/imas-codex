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
from scripts.rename.tree import load_kept_list

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


KEPT_SPLIT = {
    **KEPT_LIST,
    "kept_names": [],
    "kept_paths": [
        {
            "id": "evidence",
            "path": "docs/evidence",
            "exclude": ["*.py", "*.sh"],
            "surface": "evidence data and prose",
            "disposition": "kept",
            "reason": "only the data and prose bytes are evidence; the Python is live code",
            "min_hits": 1,
        }
    ],
}


@pytest.fixture
def split_tree(tmp_path: Path) -> Path:
    """A tree whose evidence directory holds kept prose and live Python beside it."""
    root = tmp_path / "split"
    (root / "imas_codex").mkdir(parents=True)
    (root / "imas_codex" / "__init__.py").write_text("")
    fragment = root / "docs" / "evidence" / "fragments" / "probe"
    fragment.mkdir(parents=True)
    (fragment / "run.md").write_text("The imas_codex run measured 3 surfaces.\n")
    (fragment / "measure.py").write_text(
        "from imas_codex.graph import GraphClient\n\nprint(GraphClient)\n"
    )
    return root


@pytest.fixture
def split_list(tmp_path: Path) -> Path:
    path = tmp_path / "split_kept_names.json"
    path.write_text(json.dumps(KEPT_SPLIT))
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


def test_every_evidence_type_is_kept_and_only_its_code_is_renamed() -> None:
    kept = load_kept_list(Path("scripts/rename/kept_names.json"))
    evidence = next(
        entry for entry in kept.kept_paths if entry.id == "historical-evidence"
    )

    assert evidence.exclude and set(evidence.exclude) == {"*.py", "*.sh"}
    assert kept.covering("docs/evidence/fragments/probe/measure.py") is None
    assert kept.covering("docs/evidence/fragments/probe/measure.sh") is None
    for suffix in ("html", "json", "md", "csv", "svg", "txt"):
        assert kept.covering(f"docs/evidence/fragments/probe/run.{suffix}") is not None


def test_apply_renames_evidence_python_and_leaves_evidence_bytes(
    split_tree: Path, split_list: Path
) -> None:
    evidence = split_tree / "docs/evidence/fragments/probe/run.md"
    before = evidence.read_bytes()

    report = apply_rename(split_tree, kept_list=split_list)

    script = split_tree / "docs/evidence/fragments/probe/measure.py"
    assert "from iter_codex.graph import GraphClient" in script.read_text()
    assert evidence.read_bytes() == before
    assert "docs/evidence/fragments/probe/measure.py" in report.changed_files
    assert check_tree(split_tree, kept_list=split_list).ok


def test_check_reports_a_python_file_under_evidence_that_keeps_the_old_name(
    split_tree: Path, split_list: Path
) -> None:
    apply_rename(split_tree, kept_list=split_list)
    planted = split_tree / "docs/evidence/fragments/probe/late.py"
    planted.write_text("import imas_codex.cli\n")

    report = check_tree(split_tree, kept_list=split_list)

    assert not report.ok
    leftovers = [v for v in report.violations if v.kind == "leftover"]
    assert [v.detail.split(":")[0] for v in leftovers] == [
        "docs/evidence/fragments/probe/late.py"
    ]
    assert (
        main(["check", "--tree", str(split_tree), "--kept-list", str(split_list)]) == 1
    )


MOUNT_CONTEXT = {
    "id": "project-mount",
    "pattern": "(?<![\\w.~/-])/imas-codex/",
    "reason": "a page URL mounted under the project key is kept",
    "min_hits": 1,
}


def test_apply_keeps_a_mounted_url_and_renames_the_code_beside_it(
    split_tree: Path, split_list: Path
) -> None:
    data = json.loads(split_list.read_text())
    data["kept_contexts"] = [MOUNT_CONTEXT]
    split_list.write_text(json.dumps(data))
    script = split_tree / "docs/evidence/fragments/probe/measure.py"
    script.write_text(
        "from imas_codex.graph import GraphClient\n"
        'LINK = "/imas-codex/evidence/fragments/probe/run.html"\n'
    )

    report = apply_rename(split_tree, kept_list=split_list)

    text = script.read_text()
    assert "from iter_codex.graph import GraphClient" in text
    assert '"/imas-codex/evidence/fragments/probe/run.html"' in text
    assert report.reckon_key_hits == {"project-mount": 1}
    assert check_tree(split_tree, kept_list=split_list).ok


def test_check_reports_a_mounted_url_renamed_by_hand(
    split_tree: Path, split_list: Path
) -> None:
    data = json.loads(split_list.read_text())
    data["kept_contexts"] = [MOUNT_CONTEXT]
    split_list.write_text(json.dumps(data))
    script = split_tree / "docs/evidence/fragments/probe/measure.py"
    script.write_text('LINK = "/imas-codex/evidence/fragments/probe/run.html"\n')
    apply_rename(split_tree, kept_list=split_list)
    script.write_text(script.read_text().replace("/imas-codex/", "/iter-codex/"))

    report = check_tree(split_tree, kept_list=split_list)

    assert not report.ok
    assert "kept-name-missing" in {v.kind for v in report.violations}
    assert any("project-mount" in v.detail for v in report.violations)


def test_apply_leaves_reckon_state_bytes_and_check_reports_an_alteration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state_tree"
    (root / "imas_codex").mkdir(parents=True)
    (root / "imas_codex" / "__init__.py").write_text("")
    run_dir = root / "docs" / "state" / "imas-codex" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    record = run_dir / "record.md"
    record.write_text("run for project imas-codex\n")
    kept = {
        "version": 1,
        "old_names": ["imas_codex", "imas-codex"],
        "kept_paths": [
            {
                "id": "project-state",
                "path": "docs/state",
                "surface": "reckon state files",
                "disposition": "kept",
                "reason": "reckon keeps its project key and state directory",
                "min_hits": 1,
            }
        ],
        "kept_names": [],
    }
    kept_path = tmp_path / "state_kept.json"
    kept_path.write_text(json.dumps(kept))
    before = record.read_bytes()

    report = apply_rename(root, kept_list=kept_path)

    assert record.read_bytes() == before
    assert report.kept_altered == []
    assert (root / "docs" / "state" / "imas-codex").is_dir()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(report.as_dict()))
    record.write_text(record.read_text() + "altered by hand\n")

    found = check_tree(root, kept_list=kept_path, receipt=receipt)

    assert "kept-path-altered" in {v.kind for v in found.violations}


def test_apply_does_not_move_a_directory_that_holds_kept_bytes(
    split_tree: Path, split_list: Path
) -> None:
    kept_dir = split_tree / "docs" / "evidence" / "imas-codex-s10"
    kept_dir.mkdir()
    (kept_dir / "drain.md").write_text("the imas_codex drain ledger\n")
    (kept_dir / "probe.py").write_text("from imas_codex.graph import GraphClient\n")
    before = (kept_dir / "drain.md").read_bytes()

    report = apply_rename(split_tree, kept_list=split_list)

    assert kept_dir.is_dir(), "a directory holding kept bytes keeps its name"
    assert not (split_tree / "docs" / "evidence" / "iter-codex-s10").exists()
    assert (kept_dir / "drain.md").read_bytes() == before
    assert "from iter_codex.graph" in (kept_dir / "probe.py").read_text()
    assert report.kept_altered == []


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


def cutover_data() -> dict:
    return json.loads(Path("scripts/rename/cutover.json").read_text())


def cutover_step(data: dict, step_id: str) -> dict:
    return next(step for step in data["steps"] if step["id"] == step_id)


def test_cutover_states_that_the_checkout_directory_does_not_move() -> None:
    data = cutover_data()
    checkout = data["checkout_directory"]

    assert checkout["moves_at_cutover"] is False
    assert "does not move at cutover" in checkout["statement"]
    assert all("renamed checkout" not in step["step"] for step in data["steps"])


def test_cutover_refuses_a_checkout_that_moves_at_cutover(tmp_path: Path) -> None:
    data = cutover_data()
    data["checkout_directory"]["moves_at_cutover"] = True
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("moves_at_cutover" in v for v in report.violations)


def test_cutover_keeps_the_reckon_key_and_its_state_directory() -> None:
    data = cutover_data()
    project_key = cutover_step(data, "C7-reckon-project-key")

    assert project_key["disposition"] == "kept"
    assert set(project_key["unchanged"]) == {
        "project_key",
        "mounts_entry",
        "state_directory",
        "stored_reviews",
        "ledger_rows",
        "worktree_root",
        "rule",
    }
    assert "docs/state/imas-codex" in project_key["step"]
    assert "not repaired" in project_key["unchanged"]["rule"].lower()
    assert "later" in project_key["out_of_scope"]
    assert "nothing to undo" in project_key["undo"].lower()


def test_cutover_freezes_every_queued_request_before_the_rename() -> None:
    queued = cutover_step(cutover_data(), "C0-freeze")["queued_requests"]

    assert set(queued) == {
        "listing",
        "operator_instruction",
        "rule",
        "parked_worktree",
    }
    assert "drops" in queued["rule"]
    assert "before C1" in queued["operator_instruction"]
    gate_scripts = cutover_step(cutover_data(), "C9-gate-scripts")["kept_directory"]
    assert queued["rule"], "the freeze must say what each owner drops"
    assert gate_scripts.get("disposition") == "kept"


def test_cutover_refuses_a_sequence_without_the_queued_request_rule(
    tmp_path: Path,
) -> None:
    data = cutover_data()
    cutover_step(data, "C0-freeze").pop("queued_requests", None)
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any(
        "C0-freeze" in violation and "queued_requests" in violation
        for violation in report.violations
    )


def test_cutover_refuses_a_reckon_step_without_the_unchanged_block(
    tmp_path: Path,
) -> None:
    data = cutover_data()
    cutover_step(data, "C7-reckon-project-key").pop("unchanged", None)
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("unchanged" in violation for violation in report.violations)


def test_cutover_refuses_a_reckon_step_that_moves_the_state_directory(
    tmp_path: Path,
) -> None:
    data = cutover_data()
    step = cutover_step(data, "C7-reckon-project-key")
    step["step"] = "git mv docs/state/imas-codex docs/state/iter-codex"
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("git mv" in violation for violation in report.violations)


def test_cutover_refuses_a_gate_script_title_that_contradicts_its_body(
    tmp_path: Path,
) -> None:
    data = cutover_data()
    cutover_step(data, "C9-gate-scripts")["title"] = "Rewrite the saved gate scripts"
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("regenerate" in violation for violation in report.violations)


def test_cutover_refuses_evidence_narrowed_to_an_include_list(
    tmp_path: Path,
) -> None:
    data = cutover_data()
    kept = json.loads(Path("scripts/rename/kept_names.json").read_text())
    evidence = next(
        entry for entry in kept["kept_paths"] if entry["id"] == "historical-evidence"
    )
    evidence["include"] = ["*.md"]
    split = tmp_path / "kept_names.json"
    split.write_text(json.dumps(kept))
    data["tracked_rename"]["kept_list"] = str(split)
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("historical-evidence" in violation for violation in report.violations)


def test_cutover_refuses_evidence_that_stops_renaming_its_python(
    tmp_path: Path,
) -> None:
    data = cutover_data()
    kept = json.loads(Path("scripts/rename/kept_names.json").read_text())
    evidence = next(
        entry for entry in kept["kept_paths"] if entry["id"] == "historical-evidence"
    )
    evidence["exclude"] = ["*.sh"]
    split = tmp_path / "kept_names.json"
    split.write_text(json.dumps(kept))
    data["tracked_rename"]["kept_list"] = str(split)
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("probe.py" in violation for violation in report.violations)


def test_cutover_refuses_a_gate_script_directory_that_moves(tmp_path: Path) -> None:
    data = cutover_data()
    cutover_step(data, "C9-gate-scripts")["kept_directory"]["disposition"] = "renamed"
    path = tmp_path / "cutover.json"
    path.write_text(json.dumps(data))

    report = check_cutover(path)

    assert not report.ok
    assert any("kept_directory" in violation for violation in report.violations)


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

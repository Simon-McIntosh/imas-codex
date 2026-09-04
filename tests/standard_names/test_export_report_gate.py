"""The review release stops when its export report is unsuccessful."""

from __future__ import annotations

import subprocess
from pathlib import Path

from imas_codex.standard_names.catalog_release import run_review_release
from imas_codex.standard_names.export import ExportReport, GateResult


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def test_unsuccessful_export_stops_assembly_and_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bare = tmp_path / "origin.git"
    _git("init", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    _git("init", "-b", "main", cwd=catalog)
    _git("config", "user.email", "test@example.invalid", cwd=catalog)
    _git("config", "user.name", "Test User", cwd=catalog)
    _git("remote", "add", "origin", str(bare), cwd=catalog)
    (catalog / "README.md").write_text("catalog\n", encoding="utf-8")
    _git("add", "README.md", cwd=catalog)
    _git("commit", "-m", "initial catalog", cwd=catalog)
    _git("push", "origin", "main", cwd=catalog)

    focus = tmp_path / "batch.yaml"
    focus.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        "name: failed-export\n"
        "names:\n"
        "  - plasma_current\n",
        encoding="utf-8",
    )
    calls = {"assembly": 0, "publication": 0}

    def unsuccessful_export(**_kwargs) -> ExportReport:
        return ExportReport(
            gate_results=[
                GateResult(
                    gate="manifest_source_accounting",
                    passed=False,
                    issues=[{"type": "accounting_mismatch"}],
                )
            ],
            gate_failures=1,
            all_gates_passed=False,
        )

    def unexpected_assembly(*_args, **_kwargs):
        calls["assembly"] += 1
        raise AssertionError("catalog assembly must not consume a failed export")

    def unexpected_publication(**_kwargs):
        calls["publication"] += 1
        raise AssertionError("publication must not consume a failed export")

    monkeypatch.setattr(
        "imas_codex.standard_names.export.assemble_review_catalog",
        unexpected_assembly,
    )

    report = run_review_release(
        catalog,
        focus,
        "Review batch",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=unsuccessful_export,
        publisher=unexpected_publication,
        pr_creator=lambda **_kwargs: (1, "https://example.invalid/pull/1"),
        dd_gap_reader=lambda **_kwargs: [],
        upstream_repo="example/catalog",
        fork_owner="example",
    )

    assert report.errors == [
        "Export quality gates failed: manifest_source_accounting. "
        "Resolve the failed export before publishing."
    ]
    assert calls == {"assembly": 0, "publication": 0}

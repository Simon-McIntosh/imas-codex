"""A review release requires an explicit, attributable export verdict."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from imas_codex.standard_names.catalog_release import run_review_release


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture(scope="module")
def release_surface(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("release-verdict")
    remote = root / "origin.git"
    _git("init", "--bare", "-b", "main", str(remote), cwd=root)

    catalog = root / "catalog"
    catalog.mkdir()
    _git("init", "-b", "main", cwd=catalog)
    _git("config", "user.email", "test@example.invalid", cwd=catalog)
    _git("config", "user.name", "Test User", cwd=catalog)
    _git("remote", "add", "origin", str(remote), cwd=catalog)
    (catalog / "README.md").write_text("catalog\n", encoding="utf-8")
    _git("add", "README.md", cwd=catalog)
    _git("commit", "-m", "initial catalog", cwd=catalog)
    _git("push", "origin", "main", cwd=catalog)

    focus = root / "batch.yaml"
    focus.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        "name: verdict-read\n"
        "names:\n"
        "  - plasma_current\n",
        encoding="utf-8",
    )
    return catalog, focus, root


def _release_with_report(
    release_surface: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    export_report: SimpleNamespace,
) -> tuple[list[str], int]:
    catalog, focus, root = release_surface
    assembly_calls = 0

    def unexpected_assembly(*_args: object, **_kwargs: object) -> None:
        nonlocal assembly_calls
        assembly_calls += 1
        raise AssertionError("catalog assembly must not consume an unreadable verdict")

    monkeypatch.setattr(
        "imas_codex.standard_names.export.assemble_review_catalog",
        unexpected_assembly,
    )
    report = run_review_release(
        catalog,
        focus,
        "Review batch",
        staging_dir=root / "staging",
        bump="minor",
        reviews_dir=root / "reviews",
        exporter=lambda **_kwargs: export_report,
        publisher=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("publication must not consume an unreadable verdict")
        ),
        open_pr=False,
        dd_gap_reader=lambda **_kwargs: [],
    )
    return report.errors, assembly_calls


def test_missing_gate_verdict_is_refused(
    release_surface: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    errors, assembly_calls = _release_with_report(
        release_surface,
        monkeypatch,
        SimpleNamespace(exported_count=1, gate_results=[]),
    )

    assert errors == [
        "Export report is missing required verdict attribute "
        "'all_gates_passed'; release refused because export success was not observed."
    ]
    assert assembly_calls == 0


def test_failed_verdict_without_a_failed_gate_is_refused(
    release_surface: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    errors, assembly_calls = _release_with_report(
        release_surface,
        monkeypatch,
        SimpleNamespace(
            exported_count=1,
            all_gates_passed=False,
            gate_results=[],
        ),
    )

    assert errors == [
        "Export report has all_gates_passed=False but names no failed export gates; "
        "release refused because the failure cause was not observed."
    ]
    assert assembly_calls == 0


def test_failed_verdict_keeps_the_named_gate_refusal(
    release_surface: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    errors, assembly_calls = _release_with_report(
        release_surface,
        monkeypatch,
        SimpleNamespace(
            exported_count=1,
            all_gates_passed=False,
            gate_results=[
                SimpleNamespace(
                    gate="manifest_generability",
                    passed=False,
                    skipped=False,
                )
            ],
        ),
    )

    assert errors == [
        "Export quality gates failed: manifest_generability. "
        "Resolve the failed export before publishing."
    ]
    assert assembly_calls == 0

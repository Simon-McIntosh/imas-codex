"""A post-copy divergence must refuse the publish, never exit 0.

The release command historically printed ``Post-copy check found N diverged
entries`` (a ``logger.warning`` in ``run_publish``) and then exited 0 — a
command that succeeded by its own report while a name it was responsible for
never reached the graph in agreement. Measured on the 2026-09-07 WEST cut as
``196 diverged entries at exit 0``.

These tests pin both directions of the repair:

* the failure direction — a real divergence is recorded in ``report.errors``
  (so ``sn release`` exits non-zero) and the diverged tree is NOT committed;
* the healthy direction — an in-sync tree and a graph-unavailable tree both
  publish exactly as before, committing with an empty error list.

The ``CheckResult.describe_divergence`` helper is also pinned in both
directions: no divergence renders ``None``; a divergence renders an
actionable refusal naming the count and the first identities.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from iter_codex.standard_names.catalog_import import CheckResult
from iter_codex.standard_names.export import CATALOG_EDGE_MODEL_VERSION
from iter_codex.standard_names.publish import run_publish


@pytest.fixture()
def staging_dir(tmp_path: Path) -> Path:
    """A valid full-scope staging directory matching current export format."""
    staging = tmp_path / "staging"
    staging.mkdir()
    sn_dir = staging / "standard_names"
    sn_dir.mkdir(parents=True)
    entries = [
        {
            "name": "electron_temperature",
            "description": "Te",
            "documentation": "Docs",
            "kind": "scalar",
            "unit": "eV",
            "links": [],
            "constraints": [],
            "status": "draft",
        }
    ]
    (sn_dir / "equilibrium.yml").write_text(yaml.safe_dump(entries), encoding="utf-8")
    manifest = {
        "catalog_name": "imas-standard-names-catalog",
        "cocos_convention": 17,
        "grammar_version": "0.7.0",
        "isn_model_version": "0.7.0",
        "dd_version_lineage": ["4.0.0"],
        "generated_by": "test",
        "generated_at": "2024-01-01T00:00:00Z",
        "candidate_count": 1,
        "published_count": 1,
        "excluded_below_score_count": 0,
        "excluded_unreviewed_count": 0,
        "edge_model_version": CATALOG_EDGE_MODEL_VERSION,
        "domains_included": ["equilibrium"],
    }
    (staging / "catalog.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return staging


@pytest.fixture()
def isnc_repo(tmp_path: Path) -> Path:
    """A minimal ISNC git checkout with one initial commit."""
    isnc = tmp_path / "isnc"
    isnc.mkdir()
    subprocess.run(["git", "init"], cwd=isnc, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    (isnc / "README.md").write_text("# ISNC\n")
    subprocess.run(["git", "add", "."], cwd=isnc, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    return isnc


def _head_commit_isnc(isnc: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=isnc,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _patch_check(check_result: CheckResult):
    """Patch the in-scope check_catalog the publish path imports at call time."""
    return patch(
        "iter_codex.standard_names.catalog_import.check_catalog",
        return_value=check_result,
    )


class TestDescribeDivergence:
    """The divergence signal renders an actionable string only when real."""

    def test_agreement_renders_none(self) -> None:
        """An in-sync tree is not a divergence — the healthy path is silent."""
        result = CheckResult(in_sync=3, catalog_commit_sha="abc")
        assert result.describe_divergence() is None

    def test_single_diverged_identity_is_named(self) -> None:
        result = CheckResult(
            diverged=[{"name": "electron_temperature", "fields": {"unit": {}}}]
        )
        message = result.describe_divergence()
        assert message is not None
        assert "1 diverged entry" in message
        assert "electron_temperature" in message

    def test_many_diverged_identities_named_with_cap(self) -> None:
        result = CheckResult(
            diverged=[
                {"name": f"quantity_{i}", "fields": {"kind": {}}} for i in range(7)
            ]
        )
        message = result.describe_divergence()
        assert message is not None
        assert "7 diverged entries" in message
        for i in range(5):
            assert f"quantity_{i}" in message
        assert "and 2 more" in message


class TestPublishRefusesDivergedTree:
    """A real post-copy divergence must not be reported as a successful publish."""

    def test_diverged_tree_is_refused_with_an_error(self, staging_dir, isnc_repo):
        """Failure direction: divergence lands in report.errors and no commit."""
        diverged = CheckResult(
            diverged=[{"name": "electron_temperature", "fields": {"unit": {}}}],
            in_sync=0,
        )
        with _patch_check(diverged):
            report = run_publish(staging_dir, isnc_repo)

        assert report.errors, "a diverged publish must report an error"
        assert any("diverged" in err for err in report.errors), report.errors
        assert any("electron_temperature" in err for err in report.errors)
        # The refused tree must not be committed — the ISNC checkout keeps its
        # initial commit and the publish records no success SHA.
        assert report.commit_sha is None
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=isnc_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "init" in log.stdout

    def test_in_sync_tree_still_publishes(self, staging_dir, isnc_repo):
        """Healthy direction: agreement publishes and commits unchanged."""
        in_sync = CheckResult(in_sync=1, catalog_commit_sha="abc")
        with _patch_check(in_sync):
            report = run_publish(staging_dir, isnc_repo)

        assert report.errors == []
        assert report.commit_sha is not None
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=isnc_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "sn: update" in log.stdout

    def test_uncomparable_tree_still_publishes(self, staging_dir, isnc_repo):
        """Healthy direction: a graph-unavailable check is skipped, not a block."""
        with patch(
            "iter_codex.standard_names.catalog_import.check_catalog",
            side_effect=RuntimeError("graph unreachable"),
        ):
            report = run_publish(staging_dir, isnc_repo)

        assert report.errors == []
        assert report.commit_sha is not None
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=isnc_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "sn: update" in log.stdout

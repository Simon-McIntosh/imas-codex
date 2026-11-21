"""The per-reason exclusion accounting reaches the published catalog commit.

The exporter writes the full exclusion ledger (six reason counters plus the
identity-per-reason grouping) to staging/.export_report.json; publish must
transport and commit it beside catalog.yml or the accounting is discarded with
the staging directory. These tests fail if that delivery is reverted: a publish
that omits the report leaves a reviewer unable to verify that published_count
plus the accounted exclusions equals candidate_count.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from iter_codex.standard_names.export import CATALOG_EDGE_MODEL_VERSION
from iter_codex.standard_names.publish import run_publish


@pytest.fixture()
def staging_dir(tmp_path: Path) -> Path:
    """Create a valid staging directory carrying the export exclusion report."""
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
        },
        {
            "name": "ion_temperature",
            "description": "Ti",
            "documentation": "Docs",
            "kind": "scalar",
            "unit": "eV",
            "links": [],
            "constraints": [],
            "status": "draft",
        },
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
        "candidate_count": 4,
        "published_count": 2,
        "excluded_below_score_count": 0,
        "excluded_unreviewed_count": 0,
        "edge_model_version": CATALOG_EDGE_MODEL_VERSION,
        "domains_included": ["equilibrium"],
    }
    (staging / "catalog.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

    # The full accounting the manifest cannot carry: two exclusions with
    # named reasons close candidate_count - published_count = 2.
    report = {
        "emitted_identities": ["electron_temperature", "ion_temperature"],
        "exclusion_ledger": [
            {
                "reason": "outside_requested_domain",
                "count": 1,
                "identities": ["field_aligned_temperature"],
            },
            {
                "reason": "grammar_parse_failure",
                "count": 1,
                "identities": ["te_flux"],
            },
        ],
        "counts": {
            "total_candidates": 4,
            "exported": 2,
            "excluded_below_score": 0,
            "excluded_unreviewed": 0,
            "excluded_by_domain": 1,
            "excluded_placeholder": 0,
            "parse_failures": 1,
            "pruned_links": 0,
            "gate_failures": 0,
            "validation_failures": 0,
        },
    }
    (staging / ".export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return staging


@pytest.fixture()
def isnc_repo(tmp_path: Path) -> Path:
    """Create a mock ISNC git repository with an initial commit."""
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


def _git_show(repo: Path, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


class TestExportReportIsPublished:
    """The exclusion accounting is transported into and committed on ISNC."""

    @pytest.mark.parametrize(
        "export_report_name",
        [
            ".export_report.json",
        ],
    )
    def test_report_copied_and_committed(
        self, staging_dir: Path, isnc_repo: Path, export_report_name: str
    ) -> None:
        """A publish must place the report in the ISNC tree and commit it."""
        from unittest.mock import patch

        with patch(
            "iter_codex.standard_names.publish._fetch_expected_domains",
            return_value=None,
        ):
            report = run_publish(staging_dir, isnc_repo)

        assert not report.errors, f"Errors: {report.errors}"
        committed = _git_show(isnc_repo, export_report_name)
        staged_report = (staging_dir / export_report_name).read_bytes()
        assert committed == staged_report

    def test_report_in_dry_run_file_count(
        self, staging_dir: Path, isnc_repo: Path
    ) -> None:
        """Dry-run must count the report it would copy."""
        from unittest.mock import patch

        with patch(
            "iter_codex.standard_names.publish._fetch_expected_domains",
            return_value=None,
        ):
            report = run_publish(staging_dir, isnc_repo, dry_run=True)

        assert report.dry_run is True
        assert not report.errors
        # one domain yml + catalog.yml + the export report
        assert report.files_copied == 3

    def test_accounting_arithmetic_verifiable_from_committed_files(
        self, staging_dir: Path, isnc_repo: Path
    ) -> None:
        """A reviewer can close candidate_count - published_count from disk.

        The equation must hold from the two committed files alone: the
        manifest's candidate_count minus published_count equals the sum of
        the report's per-reason exclusion ledger, and the published count
        agrees with the emitted identity list.
        """
        from unittest.mock import patch

        with patch(
            "iter_codex.standard_names.publish._fetch_expected_domains",
            return_value=None,
        ):
            run_publish(staging_dir, isnc_repo)

        manifest = yaml.safe_load(_git_show(isnc_repo, "catalog.yml"))
        report = json.loads(_git_show(isnc_repo, ".export_report.json"))

        candidate_count = manifest["candidate_count"]
        published_count = manifest["published_count"]
        ledger_total = sum(row["count"] for row in report["exclusion_ledger"])

        assert candidate_count - published_count == ledger_total
        assert report["counts"]["total_candidates"] == candidate_count
        assert report["counts"]["exported"] == published_count
        assert len(report["emitted_identities"]) == published_count

    def test_report_absent_staging_still_publishes(
        self, staging_dir: Path, isnc_repo: Path
    ) -> None:
        """A legacy staging dir without the report is not blocked (additive)."""
        from unittest.mock import patch

        (staging_dir / ".export_report.json").unlink()
        with patch(
            "iter_codex.standard_names.publish._fetch_expected_domains",
            return_value=None,
        ):
            report = run_publish(staging_dir, isnc_repo)

        assert not report.errors, f"Errors: {report.errors}"
        assert not (isnc_repo / ".export_report.json").exists()

"""A rehearsal prints the accounting it exists to rehearse, and still cuts nothing.

The dry-run branch returns before the export leg, which is what makes a
rehearsal inert — and also what left an operator unable to ask it what a cut
would publish and what it would drop. The accounting lives in that leg, so a
rehearsal now drives it into a scratch tree removed before the return: the
figures appear, and the release staging directory, the frozen roster, the
review branch and the candidate counter are all still untouched.

Both halves are pinned here. The accounting half is driven through the
production exporter seam, with the leg's own report supplying the counts. The
inert half is asserted against a checkout that already carries a roster, a
review branch and a tag, so each emptiness check has a populated reading it
could be wrong about.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from imas_codex.standard_names import catalog_release
from imas_codex.standard_names.catalog_release import (
    compute_next_version,
    run_review_release,
)

_PR_TARGET = {
    "upstream_repo": "example-org/example-catalog",
    "fork_owner": "example-fork",
}


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def isnc_repo(tmp_path):
    """A local catalog checkout on 'main' with a bare 'origin' remote."""
    bare = tmp_path / "origin.git"
    _git("init", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    work = tmp_path / "isnc"
    work.mkdir()
    _git("init", "-b", "main", cwd=work)
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    _git("remote", "add", "origin", str(bare), cwd=work)
    (work / "README.md").write_text("isnc\n")
    _git("add", "README.md", cwd=work)
    _git("commit", "-m", "init", cwd=work)
    _git("push", "origin", "main", cwd=work)
    return work


def _write_names_focus(tmp_path, *, name="west-task-2e", filename="batch.yaml"):
    path = tmp_path / filename
    path.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        f"name: {name}\n"
        "names:\n"
        "  - poloidal_flux\n"
        "  - plasma_current\n",
        encoding="utf-8",
    )
    return path


def _no_dd_gaps(*, name_ids, gc=None):
    """Keep the rehearsal off the live graph; the batch is already resolved."""
    return []


def _export_report(
    *,
    total_candidates=220,
    exported_count=218,
    exclusion_counts=None,
    exclusion_records=None,
):
    """An export-leg report shaped exactly as ``_record_export_accounting`` reads it."""
    if exclusion_counts is None:
        exclusion_counts = {"missing_physics_domain": 1, "invalid_catalog_entry": 1}
    if exclusion_records is None:
        exclusion_records = [object()] * sum(exclusion_counts.values())
    return SimpleNamespace(
        total_candidates=total_candidates,
        exported_count=exported_count,
        exclusion_counts=exclusion_counts,
        exclusion_records=exclusion_records,
    )


class _RehearsalExporter:
    """Stands in for the production export leg, recording how it was driven."""

    def __init__(self, report=None, raises=None):
        self.report = report if report is not None else _export_report()
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(
        self, *, staging_dir, force, review_batch, manifest_sources=None, **kw
    ):
        self.calls.append(
            {
                "staging_dir": Path(staging_dir),
                "force": force,
                "review_batch": list(review_batch),
                "manifest_sources": manifest_sources,
            }
        )
        if self.raises is not None:
            raise self.raises
        return self.report

    @property
    def scratch_dir(self) -> Path:
        assert self.calls, "the export leg was never driven"
        return self.calls[-1]["staging_dir"]


@pytest.fixture
def rehearsal(tmp_path, isnc_repo, monkeypatch):
    """A rehearsable checkout: a roster, a review branch and a tag already exist."""
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    (reviews / "v0.1.0rc1+earlier.sn_names.yaml").write_text(
        "names: [poloidal_flux]\n", encoding="utf-8"
    )
    _git("tag", "v0.1.0rc1", cwd=isnc_repo)
    _git("branch", "review/v0.1.0rc1", cwd=isnc_repo)
    return SimpleNamespace(
        isnc=isnc_repo,
        reviews=reviews,
        staging=tmp_path / "staging",
        focus=_write_names_focus(tmp_path),
        tmp_path=tmp_path,
    )


def _rehearse(rehearsal, exporter, **overrides):
    return run_review_release(
        rehearsal.isnc,
        rehearsal.focus,
        "Review batch demo",
        staging_dir=rehearsal.staging,
        bump="minor",
        dry_run=True,
        reviews_dir=rehearsal.reviews,
        exporter=exporter,
        dd_gap_reader=_no_dd_gaps,
        **_PR_TARGET,
        **overrides,
    )


def _tree_state(rehearsal) -> dict:
    """Everything a rehearsal must leave untouched, read from the checkout."""
    return {
        "roster": sorted(p.name for p in rehearsal.reviews.iterdir()),
        "branches": _git("branch", "--list", "review/*", cwd=rehearsal.isnc).stdout,
        "tags": sorted(_git("tag", cwd=rehearsal.isnc).stdout.split()),
        "staging_exists": rehearsal.staging.exists(),
        "config_raw": ((rehearsal.isnc / "README.md").read_text(encoding="utf-8")),
    }


# ── The accounting half: three figures appear, as numbers ─────────────────


def test_rehearsal_prints_candidate_published_and_every_exclusion(
    rehearsal, monkeypatch, caplog
):
    """One invocation reports the cut's candidate count, published count and
    each exclusion with its mechanism."""
    leg = _RehearsalExporter(
        _export_report(
            total_candidates=220,
            exported_count=218,
            exclusion_counts={
                "missing_physics_domain": 1,
                "invalid_catalog_entry": 1,
            },
        )
    )
    monkeypatch.setattr(catalog_release, "_default_exporter", leg)
    caplog.set_level(logging.INFO)

    report = _rehearse(rehearsal, leg)

    assert report.errors == []
    # Half one, on the report the caller reads.
    assert report.candidate_count == 220
    assert report.published_count == 218
    assert report.exclusion_counts == {
        "missing_physics_domain": 1,
        "invalid_catalog_entry": 1,
    }
    assert report.accounting_residue == 0
    assert report.accounting_error == ""

    # Half one, on the recorded output: the three figures are printed as
    # numbers, and every exclusion names its mechanism.
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "220 candidate(s)" in text
    assert "218 published" in text
    assert "accounting residue 0" in text
    assert "excluded under missing_physics_domain: 1" in text
    assert "excluded under invalid_catalog_entry: 1" in text

    # The counts come out of the leg, read from the batch it was handed.
    assert leg.calls[-1]["review_batch"] == report.names
    assert leg.calls[-1]["manifest_sources"] is None
    assert report.batch_label == "west-task-2e"
    assert report.rc_version.startswith("v")
    assert report.exclusion_counts and report.names == [
        "plasma_current",
        "poloidal_flux",
    ]


def test_rehearsal_reports_a_residue_rather_than_hiding_it(
    rehearsal, monkeypatch, caplog
):
    """A candidate that is neither published nor excluded is reported, not
    silently absorbed into the arithmetic."""
    leg = _RehearsalExporter(
        _export_report(
            total_candidates=220,
            exported_count=218,
            exclusion_counts={"missing_physics_domain": 1},
        )
    )
    monkeypatch.setattr(catalog_release, "_default_exporter", leg)
    caplog.set_level(logging.INFO)

    report = _rehearse(rehearsal, leg)

    assert report.accounting_residue == 1
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "residue is 1" in text


def test_rehearsal_survives_an_unreachable_leg_and_says_so(
    rehearsal, monkeypatch, caplog
):
    """A rehearsal whose accounting cannot be measured still rehearses: the
    roster and branch it would cut are reported and the failure is named."""
    leg = _RehearsalExporter(raises=RuntimeError("export leg is unavailable"))
    monkeypatch.setattr(catalog_release, "_default_exporter", leg)
    caplog.set_level(logging.INFO)

    report = _rehearse(rehearsal, leg)

    assert report.accounting_error == "RuntimeError: export leg is unavailable"
    assert report.candidate_count == 0 and report.published_count == 0
    # It still reports what it would take, and it still writes nothing.
    assert report.rc_version
    assert report.branch.startswith("review/")
    assert not rehearsal.staging.exists()


# ── The inert half: the rehearsal reports, and cuts nothing ───────────────


def test_rehearsal_leaves_the_checkout_untouched_it_measured(
    rehearsal, monkeypatch, caplog
):
    """The accounting appears and nothing is written: no staging directory, no
    roster, no candidate artifact, no review branch, no tag, and the scratch
    tree the leg ran in is gone."""
    leg = _RehearsalExporter()
    monkeypatch.setattr(catalog_release, "_default_exporter", leg)
    caplog.set_level(logging.INFO)

    before = _tree_state(rehearsal)
    # Positive controls: the checks below can see a populated tree, so an
    # unchanged reading means unchanged rather than unreadable.
    assert before["roster"] == ["v0.1.0rc1+earlier.sn_names.yaml"]
    assert "review/v0.1.0rc1" in before["branches"]
    assert "v0.1.0rc1" in before["tags"]
    candidate_before = compute_next_version(
        rehearsal.isnc, "minor", final=False, build="west-task-2e"
    )

    report = _rehearse(rehearsal, leg)

    after = _tree_state(rehearsal)
    assert after == before, "a rehearsal changed the checkout it measured"
    assert not (rehearsal.reviews / f"{report.rc_version}.sn_names.yaml").exists()
    assert report.artifact_path.endswith(f"{report.rc_version}.sn_names.yaml")
    # The counter for a release that never happens does not move.
    assert (
        compute_next_version(rehearsal.isnc, "minor", final=False, build="west-task-2e")
        == candidate_before
    )
    assert report.pushed is False
    assert report.pr_number is None

    # The leg ran in a scratch tree, not in the release staging directory, and
    # that scratch tree is removed before the rehearsal returns.
    assert leg.scratch_dir != rehearsal.staging
    assert not leg.scratch_dir.exists()
    assert "sn-release-rehearsal-" in leg.scratch_dir.name


def test_a_published_run_still_records_the_same_accounting(
    rehearsal, monkeypatch, caplog
):
    """The recording is shared: a real cut reports the counts out of the leg it
    exports with, so the rehearsal's figures are the cut's figures."""
    leg = _RehearsalExporter(
        _export_report(
            total_candidates=5,
            exported_count=4,
            exclusion_counts={"missing_physics_domain": 1},
        )
    )
    report = catalog_release.ReviewReleaseReport(dry_run=False)

    catalog_release._record_export_accounting(report, leg.report)

    assert (report.candidate_count, report.published_count) == (5, 4)
    assert report.exclusion_counts == {"missing_physics_domain": 1}
    assert report.accounting_residue == 0
    assert report.to_dict()["exclusion_counts"] == {"missing_physics_domain": 1}

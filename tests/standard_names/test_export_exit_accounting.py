"""Export exits are counted: candidate == published + accounted, or refuse.

A real cut lost eighteen names while the declared exclusion counters read
zero and the accounting gate passed. The arithmetic is the only check that
cannot be fooled by an exit nobody instrumented, so it must hold over the
caller's whole candidate set and refuse (not warn) when it does not. A
zero from either existing counter is never a passing signal on its own.

The three known exits get counters:
  - the domain filter (missing_physics_domain / outside_requested_domain),
  - the quarantine verdict (invalid_validation_status), and
  - retired-stage identities reaching candidate selection at all
    (retired_identity) — previously dropped silently at the population
    boundary by the tombstone predicate, so they vanished unrecorded while
    the gate still passed.

Two score-threshold halves are also pinned here: the name-score filter is
applied to pipeline-origin names, while the derived/catalog_edit auto-accept
exemption publishes below the declared min_score_applied — so the header's
blanket ``min_score_applied`` is not the threshold in force for exempt
origins, which is why a published 0.568 can sit under a declared 0.65.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import yaml

from imas_codex.standard_names.export import run_export

# A null link_status was the strongest surviving signal on the six
# accepted-and-valid drops from the recorded cut (9 of 18 versus 2 of the
# published 196), but link_status is not consulted anywhere in export.py at
# this revision, so it cannot itself be the drop mechanism. Their mechanism is
# not isolated; the arithmetic refusal is what catches a future recurrence.


class _ReadOnlyGraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        return []


def _candidate(name: str, **overrides) -> dict:
    candidate = {
        "id": name,
        "name_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "_validation_observed_at": "2026-09-09T00:00:00Z",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
        "reviewer_score_name": 0.95,
        "description": f"Description for {name}.",
        "documentation": f"Documentation for {name}.",
        "kind": "scalar",
        "unit": "1",
        "physics_domain": "general",
        "links": [],
    }
    candidate.update(overrides)
    return candidate


def _run_fixture_export(
    staging_dir: Path,
    population: list[dict],
    *,
    retired_batch_ids: list[str] | None = None,
    review_batch: list[str] | None = None,
    classify_result: tuple[list[dict], list] | None = None,
) -> object:
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_export_population",
                return_value=population,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.graph.client.GraphClient",
                return_value=_ReadOnlyGraphClient(),
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_retired_batch_ids",
                return_value=retired_batch_ids or [],
            )
        )
        if classify_result is not None:
            stack.enter_context(
                patch(
                    "imas_codex.standard_names.export._classify_export_population",
                    return_value=classify_result,
                )
            )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._validate_entry",
                side_effect=lambda entry: entry,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_deprecation_stubs",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
                return_value=([], set()),
            )
        )
        stack.enter_context(
            patch("imas_codex.standard_names.export._write_domain_yaml")
        )
        return run_export(
            staging_dir,
            skip_gate=True,
            force=True,
            review_batch=review_batch,
        )


def _accounting_gate(report):
    return next(
        gate for gate in report.gate_results if gate.gate == "exclusion_accounting"
    )


# ---------------------------------------------------------------------------
# The arithmetic: reconcile passes, uncounted exit refuses
# ---------------------------------------------------------------------------


def test_reconciling_cut_passes(tmp_path: Path) -> None:
    """A cut whose published count plus accounted exclusions meets its
    candidate count is emitted; the arithmetic is not a wall."""
    population = [
        _candidate("emitted_name"),
        _candidate("quarantined_name", validation_status="quarantined"),
        _candidate("docs_pending_name", docs_stage="pending"),
        _candidate("unreviewed_name", reviewer_score_name=None),
    ]

    report = _run_fixture_export(tmp_path, population)

    accounting = _accounting_gate(report)
    assert accounting.passed
    assert report.all_gates_passed
    assert report.total_candidates == 4
    assert report.exported_count == 1
    assert report.exported_count + len(report.exclusion_records) == 4
    assert (tmp_path / "catalog.yml").exists()


def test_uncounted_exit_refuses_naming_shortfall(tmp_path: Path) -> None:
    """A name that leaves with no recorded exclusion is a refusal, and the
    refusal names the shortfall rather than proceeding with a warning."""
    population = [_candidate("emitted_name"), _candidate("vanished_name")]
    # Simulate an exit no mechanism counters: the classifier drops
    # ``vanished_name`` with no exclusion record at all.
    with patch(
        "imas_codex.standard_names.export._classify_export_population",
        return_value=([population[0]], []),
    ):
        report = _run_fixture_export(tmp_path, population)

    accounting = _accounting_gate(report)
    assert not accounting.passed
    assert not report.all_gates_passed
    assert any(
        issue["type"] == "unattributed_identity"
        and issue["identities"] == ["vanished_name"]
        for issue in accounting.issues
    )
    mismatch = next(
        issue
        for issue in accounting.issues
        if issue["type"] == "exclusion_accounting_mismatch"
    )
    assert mismatch["accepted_population"] == 2
    assert mismatch["emitted"] == 1
    assert mismatch["excluded"] == 0
    assert mismatch["unaccounted"] == 1
    assert not (tmp_path / "catalog.yml").exists()


# ---------------------------------------------------------------------------
# The three counters: domain, quarantine, retired
# ---------------------------------------------------------------------------


def test_domain_and_quarantine_exits_are_counted(tmp_path: Path) -> None:
    """The domain filter and the quarantine verdict each produce a recorded,
    named exclusion — a zero in the header counters is not the signal."""
    population = [
        _candidate("no_domain_name", physics_domain=[]),
        _candidate("quarantined_name", validation_status="quarantined"),
        _candidate("emitted_name"),
    ]

    report = _run_fixture_export(tmp_path, population)

    by_id = {record.standard_name_id: record for record in report.exclusion_records}
    assert by_id["no_domain_name"].reason == "missing_physics_domain"
    assert by_id["quarantined_name"].reason == "invalid_validation_status"
    assert report.total_candidates == 3
    assert report.exported_count + len(report.exclusion_records) == 3


def test_retired_stage_batch_member_is_counted_not_vanished(
    tmp_path: Path,
) -> None:
    """A retired identity listed in the caller's review batch is counted as an
    exclusion, not dropped silently at the population boundary.

    This is the recorded defect shape: the tombstone predicate removes the
    identity before the population is built, so baseline code reports the
    shrunk universe as the candidate count and the accounting gate passes
    while the handed-in name vanished with no record and no counter.
    """
    retired = "superseded_batch_member"
    live = "emitted_batch_member"
    # The population fetch simulates the tombstone drop: the retired member is
    # absent from the returned universe.
    population = [_candidate(live)]

    report = _run_fixture_export(
        tmp_path,
        population,
        retired_batch_ids=[retired],
        review_batch=[retired, live],
    )

    by_id = {record.standard_name_id: record for record in report.exclusion_records}
    assert by_id[retired].reason == "retired_identity"
    # The candidate count is the caller's handed-in set, not the shrunk one.
    assert report.total_candidates == 2
    assert report.exported_names == [live]
    accounting = _accounting_gate(report)
    assert accounting.passed
    assert report.all_gates_passed
    assert (tmp_path / "catalog.yml").exists()


def test_retired_batch_member_without_counter_refuses(tmp_path: Path) -> None:
    """Before the retired mechanism is counted the arithmetic refuses.

    A retired member the population fetch has already dropped, with no
    ``retired_identity`` counter in place, leaves the candidate count short of
    the handed-in batch: the refusal names the shortfall. The counter test above
    is what turns this refusal into a clean close.
    """
    retired = "superseded_orphan"
    live = "emitted_live_name"
    population = [_candidate(live)]

    report = _run_fixture_export(
        tmp_path,
        population,
        retired_batch_ids=[],
        review_batch=[retired, live],
    )

    accounting = _accounting_gate(report)
    assert not accounting.passed
    assert any(
        issue["type"] == "unattributed_identity" and issue["identities"] == [retired]
        for issue in accounting.issues
    )
    assert any(
        issue["type"] == "exclusion_accounting_mismatch"
        and issue["accepted_population"] == 2
        and issue["emitted"] == 1
        and issue["unaccounted"] == 1
        for issue in accounting.issues
    )


# ---------------------------------------------------------------------------
# The score-threshold anomaly: both halves
# ---------------------------------------------------------------------------


def test_score_filter_is_applied_to_pipeline_names(tmp_path: Path) -> None:
    """Half one: the name-score threshold genuinely excludes a pipeline-origin
    name below min_score_applied."""
    population = [
        _candidate(
            "below_threshold_pipeline_name",
            origin=None,
            reviewer_score_name=0.5,
        )
    ]

    report = _run_fixture_export(tmp_path, population)

    assert report.exported_names == []
    reasons = {record.reason for record in report.exclusion_records}
    assert "below_name_score" in reasons
    accounting = _accounting_gate(report)
    assert accounting.passed


def test_exempt_origin_can_publish_below_declared_min_score(
    tmp_path: Path,
) -> None:
    """Half two: the derived/catalog_edit auto-accept exemption bypasses the
    name-score threshold, so the header's blanket ``min_score_applied`` is not
    the threshold in force for exempt origins — explaining a published 0.568
    under a declared 0.65."""
    population = [
        _candidate(
            "curated_name_below_threshold",
            origin="catalog_edit",
            reviewer_score_name=0.568,
        )
    ]

    report = _run_fixture_export(tmp_path, population)

    assert report.all_gates_passed
    assert report.exported_names == ["curated_name_below_threshold"]
    manifest = yaml.safe_load((tmp_path / "catalog.yml").read_text(encoding="utf-8"))
    assert manifest["min_score_applied"] == 0.65
    assert manifest["published_count"] == 1

"""A manifest cut owes every source a name, and says so per source.

A cut driven by a manifest is a promise that every source in it can be generated
into a name. Where that is not true the source is a pipeline defect rather than a
reporting line, so the cut has to be able to refuse instead of publishing a count
of exclusions no downstream reader can act on.

The instrument is ``describe_manifest_generability`` over the export's own
disposition records, and the gate ``manifest_generability`` carries its verdict
into the same refusal the release paths already read. These tests pin the
verdict's exhaustiveness (every source appears once, as carried or as blocked),
its resolution (a source is blocked by one named mechanism, never a bucket), and
the two mechanisms a real manifest produces most often, a spent search and a
composition nothing scheduled. It also pins the explicit waiver that can settle
a permanent exclusion without removing it from the uncarried accounting. One
bucket named "excluded" is what lets an uncarried source read as accounted for,
so the mechanisms are kept distinct.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from imas_codex.standard_names.export import (
    ExportReport,
    SourceDispositionRecord,
    describe_manifest_generability,
    run_export,
)


class _ReadOnlyGraphClient:
    """A read-only client that answers every query with nothing."""

    def __enter__(self) -> _ReadOnlyGraphClient:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def query(self, cypher: str, **_params):
        return []


def _candidate(name: str, **overrides: object) -> dict:
    candidate: dict = {
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
        # A Data Dictionary extraction reached this name, which is what makes it
        # carriable at all: without producer evidence the eligibility classifier
        # withholds it and there would be no carried source to control against.
        "_has_dd_source_binding": True,
        "_has_derived_producer": False,
        "_has_non_derived_producer": True,
        "_has_live_child": False,
        "_is_parent": False,
    }
    candidate.update(overrides)
    return candidate


def _run_fixture_export(
    staging_dir: Path,
    population: list[dict],
    *,
    manifest_sources: list[dict],
):
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
        return run_export(
            staging_dir,
            skip_gate=True,
            force=True,
            manifest_sources=manifest_sources,
        )


def _manifest_sources() -> list[dict]:
    """A manifest with one source carried and three blocked, each differently.

    The blocked three are the classes a generability check has to tell apart
    before a release can decide whether to wait: a refusal the pipeline recorded,
    a search that spent its attempt budget, and a composition nothing has
    scheduled yet. All three are shapes a real manifest produces.
    """
    return [
        {
            "source_path": "equilibrium/accepted",
            "source_status": "composed",
            "standard_name_id": "accepted_name",
            "terminal_stage": "accepted",
        },
        {
            "source_path": "equilibrium/time_axis",
            "source_status": "extracted",
            "standard_name_id": None,
            "terminal_stage": None,
            "non_nameable_reason": "cause not recorded",
        },
        {
            "source_path": "equilibrium/camera_dimensions",
            "source_status": "failed",
            "standard_name_id": "extent_of_camera",
            "terminal_stage": "exhausted",
        },
        {
            "source_path": "equilibrium/emissivity_peak",
            "source_status": "skipped",
            "standard_name_id": None,
            "terminal_stage": None,
            "non_nameable_reason": "no_locus_for_quantity: emissivity peak",
        },
    ]


def _carried_only_manifest() -> list[dict]:
    return [_manifest_sources()[0]]


def test_every_manifest_source_is_carried_or_names_its_blocking_mechanism(
    tmp_path: Path,
) -> None:
    """The verdict accounts for each source once, with the mechanism named."""
    report = _run_fixture_export(
        tmp_path,
        [_candidate("accepted_name")],
        manifest_sources=_manifest_sources(),
    )

    verdict = report.manifest_generability
    assert verdict is not None
    assert verdict["manifest_size"] == 4
    assert verdict["generable"] is False
    assert verdict["carried"] == 1
    assert verdict["carried_sources"] == ["equilibrium/accepted"]
    # Exhaustive: every source is accounted for exactly once, so nothing falls
    # silently out of the report.
    assert (
        len(verdict["carried_sources"]) + len(verdict["uncarried_sources"])
        == verdict["manifest_size"]
    )

    blocked = {entry["source_path"]: entry for entry in verdict["uncarried_sources"]}
    assert sorted(blocked) == [
        "equilibrium/camera_dimensions",
        "equilibrium/emissivity_peak",
        "equilibrium/time_axis",
    ]
    assert {
        source_path: entry["mechanism"] for source_path, entry in blocked.items()
    } == {
        "equilibrium/camera_dimensions": "attempt_budget_exhausted",
        "equilibrium/emissivity_peak": "recorded_refusal",
        "equilibrium/time_axis": "composition_not_scheduled",
    }
    # Resolved, not bucketed: the three classes a release has to weigh
    # differently come out as three different mechanisms.
    assert len({entry["mechanism"] for entry in blocked.values()}) == 3
    assert verdict["mechanism_counts"] == {
        "recorded_refusal": 1,
        "attempt_budget_exhausted": 1,
        "composition_not_scheduled": 1,
    }
    # Each mechanism is named against the evidence that establishes it, so the
    # verdict can be argued with rather than merely believed.
    assert blocked["equilibrium/emissivity_peak"]["detail"] == (
        "no_locus_for_quantity: emissivity peak"
    )
    assert blocked["equilibrium/camera_dimensions"]["terminal_stage"] == "exhausted"
    assert blocked["equilibrium/time_axis"]["source_status"] == "extracted"


def test_a_cut_refuses_a_manifest_it_cannot_generate_rather_than_counting(
    tmp_path: Path,
) -> None:
    """The refusal reaches the release path, carrying the mechanisms with it."""
    report = _run_fixture_export(
        tmp_path,
        [_candidate("accepted_name")],
        manifest_sources=_manifest_sources(),
    )

    # The release paths refuse on ``all_gates_passed``; expressing this as a gate
    # rather than as a count in the report body is what makes them usable: the
    # ledger still counts, and the cut stops.
    assert report.all_gates_passed is False
    failed = [
        gate for gate in report.gate_results if not gate.passed and not gate.skipped
    ]
    assert [gate.gate for gate in failed] == ["manifest_generability"]

    gate = failed[0]
    assert gate.skipped is False
    summary = [
        issue for issue in gate.issues if issue["type"] == "manifest_not_generable"
    ]
    assert len(summary) == 1
    assert summary[0]["manifest_size"] == 4
    assert summary[0]["carried"] == 1
    assert summary[0]["uncarried"] == 3
    assert summary[0]["mechanism_counts"]["attempt_budget_exhausted"] == 1

    # Not a bare count: every uncarried source is named with its mechanism and
    # the status that establishes it, which is what a count cannot express.
    per_source = [
        issue for issue in gate.issues if issue["type"] == "source_not_carried"
    ]
    assert {issue["source_path"] for issue in per_source} == {
        "equilibrium/camera_dimensions",
        "equilibrium/emissivity_peak",
        "equilibrium/time_axis",
    }
    assert {issue["mechanism"] for issue in per_source} == {
        "attempt_budget_exhausted",
        "recorded_refusal",
        "composition_not_scheduled",
    }
    # The verdict travels with the report as well, so the artifact a reviewer
    # reads carries the same accounting the refusal was decided on.
    assert report.to_dict()["manifest_generability"] == report.manifest_generability


def test_a_manifest_whose_sources_all_reach_a_name_is_not_refused(
    tmp_path: Path,
) -> None:
    """The control: a fully carried manifest passes both the gate and the cut."""
    report = _run_fixture_export(
        tmp_path,
        [_candidate("accepted_name")],
        manifest_sources=_carried_only_manifest(),
    )

    verdict = report.manifest_generability
    assert verdict is not None
    assert verdict["generable"] is True
    assert verdict["manifest_size"] == 1
    assert verdict["carried_sources"] == ["equilibrium/accepted"]
    assert verdict["uncarried_sources"] == []
    assert verdict["mechanism_counts"] == {}
    assert report.all_gates_passed is True
    # The same check that refuses the manifest above is satisfied here: the
    # control is the check reporting generable, not the check being absent.
    generability = [
        gate for gate in report.gate_results if gate.gate == "manifest_generability"
    ]
    assert len(generability) == 1
    assert generability[0].passed is True
    assert generability[0].issues == []


def test_explicit_waiver_is_generable_and_remains_visible_as_uncarried() -> None:
    """Only the row's explicit waiver settles a permanent exclusion."""
    source_path = "equilibrium/time_slice/constraints/flux_loop/weight"
    reason = "dd_node_category_ineligible: Backing DD node category fit_artifact"
    waived = SourceDispositionRecord(
        source_path=source_path,
        disposition="waived",
        reason=reason,
        source_status="skipped",
    )

    verdict = describe_manifest_generability([waived])

    assert verdict["generable"] is True
    assert verdict["manifest_size"] == 1
    assert verdict["carried"] == 0
    assert verdict["uncarried"] == 1
    assert verdict["carried_sources"] == []
    assert verdict["mechanism_counts"] == {"waived": 1}
    assert verdict["uncarried_sources"] == [
        {
            "source_path": source_path,
            "mechanism": "waived",
            "source_status": "skipped",
            "standard_name_id": None,
            "terminal_stage": None,
            "detail": reason,
        }
    ]
    reconciliation = ExportReport(
        source_disposition_records=[waived],
        manifest_generability=verdict,
    ).to_dict()["source_reconciliation"]
    assert reconciliation["manifest_size"] == 1
    assert reconciliation["accounted"] == 1
    assert reconciliation["waived"] == 1
    assert reconciliation["rows"] == [waived.to_dict()]

    # The same permanent refusal without the explicit row disposition remains
    # a blocker: neither its category nor its reason grants a waiver by itself.
    unwaived = SourceDispositionRecord(
        source_path=source_path,
        disposition="documented_non_nameable",
        reason=reason,
        source_status="skipped",
    )
    unwaived_verdict = describe_manifest_generability([unwaived])
    assert unwaived_verdict["generable"] is False
    assert unwaived_verdict["mechanism_counts"] == {"recorded_refusal": 1}


def test_lost_refusal_cause_is_not_reported_as_unscheduled_composition() -> None:
    """A skipped source with the lost-cause sentinel records the refusal."""
    source_path = "equilibrium/time_axis"
    record = SourceDispositionRecord(
        source_path=source_path,
        disposition="documented_non_nameable",
        reason="cause not recorded",
        source_status="skipped",
    )

    verdict = describe_manifest_generability([record])

    assert verdict["generable"] is False
    assert verdict["mechanism_counts"] == {"refusal_cause_not_recorded": 1}
    assert verdict["uncarried_sources"] == [
        {
            "source_path": source_path,
            "mechanism": "refusal_cause_not_recorded",
            "source_status": "skipped",
            "standard_name_id": None,
            "terminal_stage": None,
            "detail": "cause not recorded",
        }
    ]

"""Tests for evidence-only Data Dictionary defect reporting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.dd_gaps import (
    _registry_inventory,
    get_dd_gap,
    get_dd_gap_stats,
    list_dd_gaps,
    reconcile_dd_gaps,
    sync_dd_unit_exception_gaps,
    write_dd_gaps,
)


def _graph_context(mock_gc: MagicMock):
    graph = MagicMock()
    graph.return_value.__enter__.return_value = mock_gc
    graph.return_value.__exit__.return_value = False
    return patch("imas_codex.standard_names.dd_gaps.GraphClient", graph)


def _evidence(path: str = "equilibrium/path") -> dict[str, str]:
    return {
        "path": path,
        "kind": "unit_defect",
        "reason": "measured twin declares Pa",
        "reporter": "unit-audit",
        "observed_dd_version": "4.1.0",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
    }


def test_empty_report_is_a_noop() -> None:
    assert write_dd_gaps([]) == {
        "reported": 0,
        "relationships": 0,
        "observations": 0,
        "ids": [],
        "dry_run": False,
    }


def test_report_requires_evidence() -> None:
    report = _evidence()
    report["reason"] = ""
    with pytest.raises(ValueError, match="requires a reason"):
        write_dd_gaps([report], dry_run=True)


def test_report_kind_comes_from_generated_enum() -> None:
    report = _evidence()
    report["kind"] = "invented"
    with pytest.raises(ValueError, match="invalid DDGapKind"):
        write_dd_gaps([report], dry_run=True)


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "registry_backend",
        "upstream_url",
        "resolved_dd_version",
        "triage_actor",
    ],
)
def test_automated_report_rejects_disposition_fields(field: str) -> None:
    report = _evidence()
    report[field] = "flagged"
    with pytest.raises(ValueError, match="evidence-only"):
        write_dd_gaps([report], dry_run=True)


def test_reference_path_and_value_must_be_paired() -> None:
    report = _evidence()
    report["reference_path"] = "equilibrium/reference"
    with pytest.raises(ValueError, match="must be supplied together"):
        write_dd_gaps([report], dry_run=True)


def test_duplicate_fact_preserves_distinct_observations() -> None:
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "equilibrium/path"}],
        [
            {
                "reported": 1,
                "relationships": 1,
                "observations": 2,
                "ids": ["dd_gap:equilibrium/path:unit_defect"],
            }
        ],
    ]
    first = _evidence()
    second = _evidence()
    second["reason"] = "documentation also says pressure"

    with _graph_context(mock_gc):
        result = write_dd_gaps([first, second])

    assert result["reported"] == 1
    assert result["relationships"] == 1
    assert result["observations"] == 2
    write_call = mock_gc.query.call_args_list[1]
    assert "DDGapObservation" in write_call.args[0]
    assert "HAS_OBSERVATION" in write_call.args[0]
    assert "gap.example_count = evidence_count" in write_call.args[0]
    assert "datetime(b.observed_at) < gap.first_seen_at" in write_call.args[0]
    assert "datetime(b.observed_at) > gap.last_seen_at" in write_call.args[0]
    assert len({row["observation_id"] for row in write_call.kwargs["batch"]}) == 2


def test_identical_evidence_has_stable_observation_identity() -> None:
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "equilibrium/path"}],
        [
            {
                "reported": 1,
                "relationships": 1,
                "observations": 1,
                "ids": ["dd_gap:equilibrium/path:unit_defect"],
            }
        ],
    ]
    with _graph_context(mock_gc):
        write_dd_gaps([_evidence(), _evidence()])

    batch = mock_gc.query.call_args_list[1].kwargs["batch"]
    assert batch[0]["observation_id"] == batch[1]["observation_id"]


def test_missing_exact_paths_abort_before_any_write() -> None:
    mock_gc = MagicMock()
    mock_gc.query.return_value = [{"id": "equilibrium/existing"}]
    reports = [_evidence("equilibrium/existing"), _evidence("equilibrium/missing")]

    with (
        _graph_context(mock_gc),
        pytest.raises(ValueError, match="equilibrium/missing"),
    ):
        write_dd_gaps(reports)

    assert mock_gc.query.call_count == 1
    assert "RETURN node.id AS id" in mock_gc.query.call_args.args[0]


def test_live_result_uses_actual_persisted_counts() -> None:
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "equilibrium/path"}],
        [
            {
                "reported": 1,
                "relationships": 0,
                "observations": 1,
                "ids": ["dd_gap:equilibrium/path:unit_defect"],
            }
        ],
    ]
    with _graph_context(mock_gc):
        result = write_dd_gaps([_evidence()])

    assert result["relationships"] == 0


def test_dry_run_validates_paths_but_does_not_write() -> None:
    mock_gc = MagicMock()
    mock_gc.query.return_value = [{"id": "equilibrium/path"}]
    with _graph_context(mock_gc):
        result = write_dd_gaps([_evidence()], dry_run=True)

    assert result == {
        "reported": 1,
        "relationships": 1,
        "observations": 1,
        "ids": ["dd_gap:equilibrium/path:unit_defect"],
        "dry_run": True,
    }
    assert mock_gc.query.call_count == 1


def test_registry_inventory_backfills_structured_unit_facts() -> None:
    entries = {
        "dd_unit_bugs": [
            {
                "path": "*/pressure/reconstructed",
                "dd_unit": "1",
                "correct_unit": "Pa",
                "correct_in_graph": True,
                "reason": "measured twin declares Pa",
            },
            {
                "path": "*/direction/[xyz]",
                "dd_unit": "m",
                "correct_unit": "1",
                "upstream_kind": "type_wiring",
                "upstream_url": "https://example.invalid/dd-filing",
                "reason": "unit-vector component is dimensionless",
            },
        ]
    }
    paths = ["equilibrium/pressure/reconstructed", "launcher/direction/x"]
    with patch(
        "imas_codex.standard_names.dd_gaps.load_exceptions",
        return_value=entries,
    ):
        nodes, relationships = _registry_inventory(paths, "4.1.0")

    assert len(nodes) == 3
    by_kind = {node["kind"]: node for node in nodes}
    assert by_kind["self_contradiction"]["status"] == "registered_exception"
    assert by_kind["unit_defect"]["registry_backend"] == "dd_unit_exceptions"
    assert by_kind["type_wiring"]["status"] == "upstream_issue"
    assert by_kind["type_wiring"]["upstream_url"] == (
        "https://example.invalid/dd-filing"
    )
    assert all(row["observed_dd_version"] == "4.1.0" for row in relationships)
    assert {row["observed_value"] for row in relationships} == {"1", "m"}
    assert {row["expected_value"] for row in relationships} == {"Pa", "1"}
    assert all(row["evidence_rule"] == "unit_equals_expected" for row in relationships)


def test_registry_sync_returns_persisted_counts() -> None:
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": "launcher/direction/x"}],
        [
            {
                "reported": 1,
                "relationships": 1,
                "observations": 1,
                "ids": ["dd_gap:*/direction/[xyz]:unit_defect"],
            }
        ],
    ]
    entries = {
        "dd_unit_bugs": [
            {
                "path": "*/direction/[xyz]",
                "dd_unit": "m",
                "correct_unit": "1",
                "reason": "unit-vector component is dimensionless",
            }
        ]
    }
    with (
        _graph_context(mock_gc),
        patch(
            "imas_codex.standard_names.dd_gaps.load_exceptions",
            return_value=entries,
        ),
    ):
        result = sync_dd_unit_exception_gaps()

    assert result == {
        "registry_entries": 1,
        "reported": 1,
        "relationships": 1,
        "observations": 1,
        "matched_paths": 1,
        "dry_run": False,
    }
    sync_query = mock_gc.query.call_args_list[2].args[0]
    assert "DDGapObservation" in sync_query
    assert "ON CREATE SET observation.dd_gap_id" in sync_query
    assert "SET observation.last_observed_at" not in sync_query


def test_registry_dry_run_reads_version_and_paths_only() -> None:
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": "launcher/direction/x"}],
    ]
    entries = {
        "dd_unit_bugs": [
            {
                "path": "*/direction/[xyz]",
                "dd_unit": "m",
                "correct_unit": "1",
                "reason": "unit-vector component is dimensionless",
            }
        ]
    }
    with (
        _graph_context(mock_gc),
        patch(
            "imas_codex.standard_names.dd_gaps.load_exceptions",
            return_value=entries,
        ),
    ):
        result = sync_dd_unit_exception_gaps(dry_run=True)

    assert result["dry_run"] is True
    assert result["relationships"] == 1
    assert result["observations"] == 1
    assert mock_gc.query.call_count == 2


def test_status_counts_are_grouped_on_both_axes() -> None:
    mock_gc = MagicMock()
    mock_gc.query.return_value = [
        {"status": "registered_exception", "kind": "unit_defect", "count": 3},
        {"status": "upstream_issue", "kind": "type_wiring", "count": 1},
    ]
    assert get_dd_gap_stats(mock_gc) == {
        "total": 4,
        "by_status": {"registered_exception": 3, "upstream_issue": 1},
        "by_kind": {"unit_defect": 3, "type_wiring": 1},
    }


def test_list_filters_exact_paths_names_and_generated_lifecycle_values() -> None:
    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "dd_gap:equilibrium/path:unit_defect",
            "path": "equilibrium/path",
            "kind": "unit_defect",
            "status": "upstream_issue",
            "source_paths": ["equilibrium/path"],
            "affected_name_ids": ["plasma_pressure"],
            "upstream_url": "https://example.invalid/issue/1",
            "registry_backend": None,
            "resolved_dd_version": None,
            "affected_path_count": 1,
        }
    ]

    rows = list_dd_gaps(
        statuses=["upstream_issue"],
        kinds=["unit_defect"],
        path_ids=["equilibrium/path"],
        name_ids=["plasma_pressure"],
        gc=gc,
    )

    assert rows[0]["source_paths"] == ["equilibrium/path"]
    call = gc.query.call_args
    assert call.kwargs == {
        "statuses": ["upstream_issue"],
        "kinds": ["unit_defect"],
        "path_ids": ["equilibrium/path"],
        "name_ids": ["plasma_pressure"],
    }
    query = call.args[0]
    assert "ORDER BY gap.id" in query
    assert "source.source_id IN source_paths" in query
    assert not any(token in query for token in ("CREATE ", "MERGE ", "SET ", "DELETE "))


@pytest.mark.parametrize("parameter", ["statuses", "kinds", "path_ids", "name_ids"])
def test_list_rejects_bare_string_filters(parameter: str) -> None:
    gc = MagicMock()
    with pytest.raises(ValueError, match="bare string"):
        list_dd_gaps(gc=gc, **{parameter: "equilibrium/path"})
    gc.query.assert_not_called()


def test_list_rejects_pattern_where_exact_path_is_required() -> None:
    gc = MagicMock()
    with pytest.raises(ValueError, match="exact paths"):
        list_dd_gaps(path_ids=["*/pressure"], gc=gc)
    gc.query.assert_not_called()


def test_get_exact_gap_includes_observations_and_state_history() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [
            {
                "id": "dd_gap:equilibrium/path:unit_defect",
                "path": "equilibrium/path",
                "kind": "unit_defect",
                "status": "triaged",
                "source_paths": ["equilibrium/z", "equilibrium/a"],
                "affected_name_ids": ["z_pressure", "a_pressure"],
                "affected_path_count": 2,
            }
        ],
        [
            {
                "id": "observation:later",
                "reason": "second observation",
                "first_observed_at": "2026-08-02T10:00:00Z",
            },
            {
                "id": "observation:first",
                "reason": "measured twin declares Pa",
                "first_observed_at": "2026-08-01T10:00:00Z",
            },
        ],
        [
            {
                "id": "change:later",
                "from_status": "triaged",
                "to_status": "upstream_issue",
                "changed_at": "2026-08-02T12:00:00Z",
            },
            {
                "id": "change:first",
                "from_status": "flagged",
                "to_status": "triaged",
                "changed_at": "2026-08-02T11:00:00Z",
            },
        ],
    ]

    fact = get_dd_gap("dd_gap:equilibrium/path:unit_defect", gc=gc)

    assert fact is not None
    assert fact["source_paths"] == ["equilibrium/a", "equilibrium/z"]
    assert fact["affected_name_ids"] == ["a_pressure", "z_pressure"]
    assert [row["id"] for row in fact["observations"]] == [
        "observation:first",
        "observation:later",
    ]
    assert [row["id"] for row in fact["state_changes"]] == [
        "change:first",
        "change:later",
    ]
    assert gc.query.call_count == 3
    assert all(
        not any(
            token in call.args[0] for token in ("CREATE ", "MERGE ", "SET ", "DELETE ")
        )
        for call in gc.query.call_args_list
    )


def test_get_gap_rejects_invalid_identity_before_graph_access() -> None:
    gc = MagicMock()
    with pytest.raises(ValueError, match="exact 'dd_gap"):
        get_dd_gap("equilibrium/path", gc=gc)
    gc.query.assert_not_called()


def test_list_normalizes_multi_value_and_row_order() -> None:
    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "dd_gap:z/path:unit_defect",
            "source_paths": ["z/path", "a/path"],
            "affected_name_ids": ["z_name", "a_name"],
        },
        {
            "id": "dd_gap:a/path:unit_defect",
            "source_paths": ["b/path", "a/path"],
            "affected_name_ids": ["b_name", "a_name"],
        },
    ]

    rows = list_dd_gaps(gc=gc)

    assert [row["id"] for row in rows] == [
        "dd_gap:a/path:unit_defect",
        "dd_gap:z/path:unit_defect",
    ]
    assert rows[0]["source_paths"] == ["a/path", "b/path"]
    assert rows[1]["affected_name_ids"] == ["a_name", "z_name"]


def test_reconcile_fails_closed_when_evidence_changes_after_preflight() -> None:
    gap_id = "dd_gap:equilibrium/path:unit_defect"
    gc = MagicMock()
    gc.query.side_effect = [
        [{"id": "4.1.1", "is_current": True}],
        [
            {
                "id": gap_id,
                "path": "equilibrium/path",
                "kind": "unit_defect",
                "status": "upstream_issue",
                "observed_dd_version": "4.1.0",
                "observed_value": "1",
                "expected_value": "Pa",
                "evidence_rule": "unit_equals_expected",
                "reference_path": "equilibrium/reference",
                "reference_value": "Pa",
                "source_paths": ["equilibrium/path"],
                "registry_backend": None,
            }
        ],
        [],
    ]

    result = reconcile_dd_gaps(
        "4.1.1",
        {"equilibrium/path": {"unit": "Pa"}},
        gc=gc,
    )

    assert result["resolved"] == 0
    assert result["conflicts"] == [gap_id]
    mutation = gc.query.call_args_list[2]
    query = mutation.args[0]
    assert "coalesce(gap.observed_value, '') = item.observed_value" in query
    assert "coalesce(gap.expected_value, '') = item.expected_value" in query
    assert "coalesce(gap.evidence_rule, '') = item.evidence_rule" in query
    assert "size(current_source_paths) = size(item.source_paths)" in query
    assert mutation.kwargs["batch"] == [
        {
            "id": gap_id,
            "expected_status": "upstream_issue",
            "path": "equilibrium/path",
            "kind": "unit_defect",
            "observed_dd_version": "4.1.0",
            "observed_value": "1",
            "expected_value": "Pa",
            "evidence_rule": "unit_equals_expected",
            "reference_path": "equilibrium/reference",
            "reference_value": "Pa",
            "registry_backend": "",
            "source_paths": ["equilibrium/path"],
            "validation_evidence": ("4.1.1 raw unit equals 'Pa' on equilibrium/path"),
        }
    ]

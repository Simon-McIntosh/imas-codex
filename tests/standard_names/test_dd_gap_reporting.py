"""Tests for evidence-only Data Dictionary defect reporting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.dd_gaps import (
    _registry_inventory,
    get_dd_gap_stats,
    sync_dd_unit_exception_gaps,
    write_dd_gaps,
)


def _graph_context(mock_gc: MagicMock):
    graph = MagicMock()
    graph.return_value.__enter__.return_value = mock_gc
    graph.return_value.__exit__.return_value = False
    return patch("imas_codex.standard_names.dd_gaps.GraphClient", graph)


def test_empty_report_is_a_noop() -> None:
    assert write_dd_gaps([]) == {
        "reported": 0,
        "relationships": 0,
        "ids": [],
        "dry_run": False,
    }


def test_report_requires_evidence() -> None:
    with pytest.raises(ValueError, match="requires a reason"):
        write_dd_gaps(
            [{"path": "equilibrium/path", "kind": "unit_defect", "reason": ""}],
            dry_run=True,
        )


def test_report_kind_comes_from_generated_enum() -> None:
    with pytest.raises(ValueError, match="invalid DDGapKind"):
        write_dd_gaps(
            [{"path": "equilibrium/path", "kind": "invented", "reason": "evidence"}],
            dry_run=True,
        )


def test_duplicate_fact_aggregates_count_and_keeps_edge_argument() -> None:
    mock_gc = MagicMock()
    mock_gc.query.return_value = []
    reports = [
        {
            "path": "equilibrium/path",
            "kind": "unit_defect",
            "reason": "measured twin declares Pa",
        },
        {
            "path": "equilibrium/path",
            "kind": "unit_defect",
            "reason": "documentation also says pressure",
        },
    ]
    with _graph_context(mock_gc):
        result = write_dd_gaps(reports)

    assert result["reported"] == 1
    node_call, edge_call = mock_gc.query.call_args_list
    assert node_call.kwargs["batch"][0]["example_count"] == 2
    assert "coalesce(gap.example_count, 0) + b.example_count" in node_call.args[0]
    assert "HAS_DD_GAP" in edge_call.args[0]
    assert edge_call.kwargs["batch"][0]["reason"] == "documentation also says pressure"


def test_dry_run_never_opens_graph() -> None:
    with patch("imas_codex.standard_names.dd_gaps.GraphClient") as graph:
        result = write_dd_gaps(
            [
                {
                    "path": "equilibrium/path",
                    "kind": "unit_defect",
                    "reason": "measured twin declares Pa",
                }
            ],
            dry_run=True,
        )
    assert result["reported"] == 1
    graph.assert_not_called()


def test_registry_inventory_is_one_fact_per_entry_plus_upstream_filing() -> None:
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
    paths = [
        "equilibrium/pressure/reconstructed",
        "launcher/direction/x",
        "wall/energy_fluxes/kinetic/neutral/state/incident/values",
    ]
    with patch(
        "imas_codex.standard_names.dd_gaps.load_exceptions",
        return_value=entries,
    ):
        nodes, relationships = _registry_inventory(paths)

    assert len(nodes) == 3
    by_kind = {node["kind"]: node for node in nodes}
    assert by_kind["self_contradiction"]["status"] == "registered_exception"
    assert by_kind["unit_defect"]["registry_backend"] == "dd_unit_exceptions"
    assert by_kind["type_wiring"]["status"] == "upstream_issue"
    assert by_kind["type_wiring"]["upstream_url"] == "https://example.invalid/dd-filing"
    assert len(relationships) == 3


def test_registry_sync_is_idempotent_and_dry_run_is_read_only() -> None:
    mock_gc = MagicMock()
    mock_gc.query.return_value = [{"id": "launcher/direction/x"}]
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

    assert result == {
        "registry_entries": 1,
        "reported": 1,
        "relationships": 1,
        "matched_paths": 1,
        "dry_run": True,
    }
    assert mock_gc.query.call_count == 1


def test_registry_write_uses_create_only_example_count() -> None:
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [[{"id": "launcher/direction/x"}], [], []]
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
        sync_dd_unit_exception_gaps()

    node_query = mock_gc.query.call_args_list[1].args[0]
    assert "ON CREATE SET" in node_query
    assert "gap.example_count = 1" in node_query
    assert "gap.example_count =" not in node_query.split("SET gap.path", 1)[1]


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

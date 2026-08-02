"""Tests for evidence-only Data Dictionary defect reporting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.dd_gaps import (
    DDGapRegistrySyncConflict,
    _evidence_token,
    _registry_inventory,
    _registry_sync_plan,
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


def _registry_fact(
    *,
    gap_id: str = "dd_gap:*/direction/[xyz]:unit_defect",
    path: str = "*/direction/[xyz]",
    kind: str = "unit_defect",
    source_paths: list[str] | None = None,
    status: str = "registered_exception",
    backend: str | None = "dd_unit_exceptions",
    upstream_url: str | None = None,
) -> dict[str, object]:
    return {
        "id": gap_id,
        "path": path,
        "kind": kind,
        "status": status,
        "registry_backend": backend,
        "upstream_url": upstream_url,
        "resolved_dd_version": None,
        "triaged_at": "2026-08-01T00:00:00Z",
        "triage_actor": "operator@example.org",
        "triage_reason": "curated registry entry",
        "status_changed_at": "2026-08-01T00:00:00Z",
        "status_changed_by": "operator@example.org",
        "status_change_reason": "curated registry entry",
        "validation_evidence": None,
        "first_seen_at": "2026-07-01T00:00:00Z",
        "last_seen_at": "2026-08-01T00:00:00Z",
        "example_count": 1,
        "observed_dd_version": "4.1.0",
        "observed_value": "m",
        "expected_value": "1",
        "evidence_rule": "unit_equals_expected",
        "reference_path": None,
        "reference_value": None,
        "source_paths": source_paths or ["launcher/direction/x"],
        "observation_ids": ["observation:1"],
        "state_change_ids": ["change:1"],
    }


def _transaction(mock_gc: MagicMock, side_effect: list[object]):
    session = MagicMock()
    transaction = MagicMock()
    mock_gc.session.return_value.__enter__.return_value = session
    mock_gc.session.return_value.__exit__.return_value = False
    session.begin_transaction.return_value = transaction
    transaction.run.side_effect = side_effect
    return transaction


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
        [_registry_fact()],
    ]
    transaction = _transaction(
        mock_gc,
        [
            [_registry_fact()],
            [
                {
                    "reported": 1,
                    "relationships": 1,
                    "observations": 1,
                    "ids": ["dd_gap:*/direction/[xyz]:unit_defect"],
                }
            ],
            [
                {
                    "id": "dd_gap:*/direction/[xyz]:unit_defect",
                    "exists": True,
                    "kind": "unit_defect",
                    "source_paths": ["launcher/direction/x"],
                }
            ],
            [],
        ],
    )
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
        "create": [],
        "update": ["dd_gap:*/direction/[xyz]:unit_defect"],
        "reclassify": [],
        "manual_required": [],
        "dry_run": False,
    }
    sync_query = transaction.run.call_args_list[1].args[0]
    assert "DDGapObservation" in sync_query
    assert "ON CREATE SET observation.dd_gap_id" in sync_query
    assert "SET observation.last_observed_at" not in sync_query
    transaction.commit.assert_called_once_with()
    transaction.rollback.assert_not_called()


def test_registry_dry_run_reads_version_and_paths_only() -> None:
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": "launcher/direction/x"}],
        [],
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
    assert result["create"] == ["dd_gap:*/direction/[xyz]:unit_defect"]
    assert result["update"] == []
    assert result["reclassify"] == []
    assert result["manual_required"] == []
    assert mock_gc.query.call_count == 3
    mock_gc.session.assert_not_called()


def test_registry_sync_reclassifies_exact_identity_in_one_transaction() -> None:
    path_pattern = "spi/injector/*_gas/flow_rate"
    old_id = f"dd_gap:{path_pattern}:unit_defect"
    new_id = f"dd_gap:{path_pattern}:self_contradiction"
    source_paths = [
        "spi/injector/fragmentation_gas/flow_rate",
        "spi/injector/propellant_gas/flow_rate",
    ]
    old = _registry_fact(
        gap_id=old_id,
        path=path_pattern,
        source_paths=source_paths,
    )
    old["observed_value"] = "s^-1"
    old["expected_value"] = "Pa.m^3.s^-1"
    entries = {
        "dd_unit_bugs": [
            {
                "path": path_pattern,
                "dd_unit": "s^-1",
                "correct_unit": "Pa.m^3.s^-1",
                "correct_in_graph": True,
                "reason": "flow rate carries pressure-volume throughput",
            }
        ]
    }
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": p} for p in source_paths],
        [old],
    ]
    transaction = _transaction(
        mock_gc,
        [
            [old],
            [
                {
                    "id": new_id,
                    "source_path_count": 2,
                    "observation_count": 1,
                    "state_change_count": 2,
                    "state_change_id": "change:registry-reclassification",
                }
            ],
            [
                {
                    "reported": 1,
                    "relationships": 2,
                    "observations": 2,
                    "ids": [new_id],
                }
            ],
            [
                {
                    "id": new_id,
                    "exists": True,
                    "kind": "self_contradiction",
                    "source_paths": source_paths,
                }
            ],
            [],
        ],
    )

    with (
        _graph_context(mock_gc),
        patch(
            "imas_codex.standard_names.dd_gaps.load_exceptions",
            return_value=entries,
        ),
    ):
        result = sync_dd_unit_exception_gaps()

    assert result["create"] == []
    assert result["update"] == []
    assert result["manual_required"] == []
    assert result["reclassify"] == [
        {
            "old_id": old_id,
            "new_id": new_id,
            "old_kind": "unit_defect",
            "new_kind": "self_contradiction",
            "expected_sync_token": result["reclassify"][0]["expected_sync_token"],
        }
    ]
    assert result["reclassify"][0]["expected_sync_token"].startswith(
        "dd-gap-registry-sync:"
    )
    migration = transaction.run.call_args_list[1]
    assert migration.kwargs["old_id"] == old_id
    assert migration.kwargs["new_id"] == new_id
    assert migration.kwargs["expected_status"] == "registered_exception"
    assert migration.kwargs["target_kind"] == "self_contradiction"
    assert migration.kwargs["expected_source_paths"] == source_paths
    assert migration.kwargs["expected_observation_ids"] == ["observation:1"]
    assert migration.kwargs["expected_state_change_ids"] == ["change:1"]
    query = migration.args[0]
    assert "SET gap.id = $new_id" in query
    assert "SET item.dd_gap_id" not in query
    assert "authoritative registry identity changed classification from" in query
    assert "DELETE" not in query
    set_clause = query.split("SET gap.id = $new_id", 1)[1].split("FOREACH", 1)[0]
    assert "first_seen_at" not in set_clause
    assert "triaged_at" not in set_clause
    assert "resolved_dd_version" not in set_clause
    assert _evidence_token(old) != _evidence_token(
        {**old, "id": new_id, "kind": "self_contradiction"}
    )
    transaction.commit.assert_called_once_with()
    transaction.rollback.assert_not_called()


def test_registry_sync_dry_run_reports_reclassification_without_mutation() -> None:
    pattern = "spi/injector/*_gas/flow_rate"
    old = _registry_fact(
        gap_id=f"dd_gap:{pattern}:unit_defect",
        path=pattern,
        source_paths=["spi/injector/fragmentation_gas/flow_rate"],
    )
    entries = {
        "dd_unit_bugs": [
            {
                "path": pattern,
                "dd_unit": "s^-1",
                "correct_unit": "Pa.m^3.s^-1",
                "correct_in_graph": True,
                "reason": "flow rate carries pressure-volume throughput",
            }
        ]
    }
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": "spi/injector/fragmentation_gas/flow_rate"}],
        [old],
    ]
    with (
        _graph_context(mock_gc),
        patch(
            "imas_codex.standard_names.dd_gaps.load_exceptions",
            return_value=entries,
        ),
    ):
        result = sync_dd_unit_exception_gaps(dry_run=True)

    assert result["create"] == []
    assert result["update"] == []
    assert result["reclassify"][0]["old_id"].endswith(":unit_defect")
    assert result["reclassify"][0]["new_id"].endswith(":self_contradiction")
    mock_gc.session.assert_not_called()


def test_registry_plan_fails_closed_on_ambiguous_identity() -> None:
    target = {
        "id": "dd_gap:pattern:self_contradiction",
        "path": "pattern",
        "kind": "self_contradiction",
        "registry_backend": "dd_unit_exceptions",
        "upstream_url": None,
    }
    observations = [{"gap_id": target["id"], "source_path": "exact/path"}]
    first = _registry_fact(
        gap_id="dd_gap:pattern:unit_defect",
        path="pattern",
        source_paths=["exact/path"],
    )
    second = _registry_fact(
        gap_id="dd_gap:pattern:doc_mismatch",
        path="pattern",
        kind="doc_mismatch",
        source_paths=["exact/path"],
    )

    plan = _registry_sync_plan([target], observations, [first, second])

    assert plan["create"] == []
    assert plan["update"] == []
    assert plan["reclassify"] == []
    assert plan["manual_required"] == [
        {
            "id": target["id"],
            "reason": "multiple facts match the authoritative registry identity",
            "candidate_ids": sorted([first["id"], second["id"]]),
        }
    ]


def test_registry_plan_rejects_target_collision_without_merging() -> None:
    target = {
        "id": "dd_gap:pattern:self_contradiction",
        "path": "pattern",
        "kind": "self_contradiction",
        "registry_backend": "dd_unit_exceptions",
        "upstream_url": None,
    }
    observations = [{"gap_id": target["id"], "source_path": "exact/path"}]
    existing_target = _registry_fact(
        gap_id=target["id"],
        path="pattern",
        kind="self_contradiction",
        source_paths=["exact/path"],
    )
    old = _registry_fact(
        gap_id="dd_gap:pattern:unit_defect",
        path="pattern",
        source_paths=["exact/path"],
    )

    plan = _registry_sync_plan([target], observations, [existing_target, old])

    assert plan["reclassify"] == []
    assert plan["manual_required"][0]["reason"] == (
        "target id collides with another matching registry fact"
    )
    assert plan["manual_required"][0]["candidate_ids"] == [old["id"]]


def test_registry_plan_refuses_related_fact_with_different_path_links() -> None:
    target = {
        "id": "dd_gap:pattern:self_contradiction",
        "path": "pattern",
        "kind": "self_contradiction",
        "registry_backend": "dd_unit_exceptions",
        "upstream_url": None,
    }
    observations = [{"gap_id": target["id"], "source_path": "current/path"}]
    old = _registry_fact(
        gap_id="dd_gap:pattern:unit_defect",
        path="pattern",
        source_paths=["stale/path"],
    )

    plan = _registry_sync_plan([target], observations, [old])

    assert plan["create"] == []
    assert plan["reclassify"] == []
    assert plan["manual_required"] == [
        {
            "id": target["id"],
            "reason": "registry evidence path set differs from existing fact",
            "candidate_ids": [old["id"]],
        }
    ]


def test_registry_plan_is_idempotent_and_ignores_unrelated_facts() -> None:
    target = {
        "id": "dd_gap:pattern:self_contradiction",
        "path": "pattern",
        "kind": "self_contradiction",
        "registry_backend": "dd_unit_exceptions",
        "upstream_url": None,
    }
    observations = [{"gap_id": target["id"], "source_path": "exact/path"}]
    existing_target = _registry_fact(
        gap_id=target["id"],
        path="pattern",
        kind="self_contradiction",
        source_paths=["exact/path"],
    )
    unrelated = _registry_fact(
        gap_id="dd_gap:other/path:unit_defect",
        path="other/path",
        source_paths=["other/path"],
    )

    plan = _registry_sync_plan([target], observations, [existing_target, unrelated])

    assert plan == {
        "create": [],
        "update": [target["id"]],
        "reclassify": [],
        "manual_required": [],
    }


def test_registry_sync_rejects_concurrent_lifecycle_drift_and_rolls_back() -> None:
    pattern = "spi/injector/*_gas/flow_rate"
    old = _registry_fact(
        gap_id=f"dd_gap:{pattern}:unit_defect",
        path=pattern,
        source_paths=["spi/injector/fragmentation_gas/flow_rate"],
    )
    changed = {**old, "status": "resolved_upstream"}
    entries = {
        "dd_unit_bugs": [
            {
                "path": pattern,
                "dd_unit": "s^-1",
                "correct_unit": "Pa.m^3.s^-1",
                "correct_in_graph": True,
                "reason": "flow rate carries pressure-volume throughput",
            }
        ]
    }
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": "spi/injector/fragmentation_gas/flow_rate"}],
        [old],
    ]
    transaction = _transaction(mock_gc, [[changed]])

    with (
        _graph_context(mock_gc),
        patch(
            "imas_codex.standard_names.dd_gaps.load_exceptions",
            return_value=entries,
        ),
        pytest.raises(DDGapRegistrySyncConflict, match="changed after preflight"),
    ):
        sync_dd_unit_exception_gaps()

    transaction.commit.assert_not_called()
    transaction.rollback.assert_called_once_with()
    assert transaction.run.call_count == 1


def test_registry_sync_rolls_back_when_atomic_rewrite_matches_nothing() -> None:
    pattern = "spi/injector/*_gas/flow_rate"
    old = _registry_fact(
        gap_id=f"dd_gap:{pattern}:unit_defect",
        path=pattern,
        source_paths=["spi/injector/fragmentation_gas/flow_rate"],
    )
    entries = {
        "dd_unit_bugs": [
            {
                "path": pattern,
                "dd_unit": "s^-1",
                "correct_unit": "Pa.m^3.s^-1",
                "correct_in_graph": True,
                "reason": "flow rate carries pressure-volume throughput",
            }
        ]
    }
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": "spi/injector/fragmentation_gas/flow_rate"}],
        [old],
    ]
    transaction = _transaction(mock_gc, [[old], []])

    with (
        _graph_context(mock_gc),
        patch(
            "imas_codex.standard_names.dd_gaps.load_exceptions",
            return_value=entries,
        ),
        pytest.raises(DDGapRegistrySyncConflict, match="no longer matches"),
    ):
        sync_dd_unit_exception_gaps()

    transaction.commit.assert_not_called()
    transaction.rollback.assert_called_once_with()


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
                "triaged_at": "2026-08-02T11:00:00Z",
                "status_changed_at": "2026-08-02T11:00:00Z",
                "status_changed_by": "operator@example.org",
                "status_change_reason": "evidence checked",
                "source_paths": ["equilibrium/z", "equilibrium/a"],
                "affected_name_ids": ["z_pressure", "a_pressure"],
                "affected_path_count": 2,
            }
        ],
        [
            {
                "id": "observation:later",
                "reason": "second observation",
                "reference_path": "equilibrium/reference/z",
                "reference_value": "Pa",
                "first_observed_at": "2026-08-02T10:00:00Z",
            },
            {
                "id": "observation:first",
                "reason": "measured twin declares Pa",
                "reference_path": "equilibrium/reference/a",
                "reference_value": "Pa",
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
    assert fact["triaged_at"] == "2026-08-02T11:00:00Z"
    assert fact["status_changed_by"] == "operator@example.org"
    assert [row["id"] for row in fact["observations"]] == [
        "observation:first",
        "observation:later",
    ]
    assert [row["id"] for row in fact["state_changes"]] == [
        "change:first",
        "change:later",
    ]
    assert fact["observations"][0]["reference_path"] == "equilibrium/reference/a"
    assert fact["observations"][1]["reference_value"] == "Pa"
    assert fact["evidence_token"].startswith("dd-gap-evidence:")
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


def test_evidence_token_is_order_independent_and_observation_sensitive() -> None:
    fact = {
        "path": "equilibrium/path",
        "kind": "unit_defect",
        "source_paths": ["equilibrium/z", "equilibrium/a"],
        "observations": [{"id": "observation:z"}, {"id": "observation:a"}],
        "example_count": 2,
        "first_seen_at": "2026-08-01T10:00:00Z",
        "last_seen_at": "2026-08-02T10:00:00Z",
    }
    reordered = {
        **fact,
        "source_paths": list(reversed(fact["source_paths"])),
        "observations": list(reversed(fact["observations"])),
    }
    concurrent = {
        **fact,
        "observations": [*fact["observations"], {"id": "observation:new"}],
        "example_count": 3,
        "last_seen_at": "2026-08-03T10:00:00Z",
    }

    assert _evidence_token(fact) == _evidence_token(reordered)
    assert _evidence_token(fact) != _evidence_token(concurrent)


def test_reconcile_fails_closed_when_evidence_changes_after_preflight() -> None:
    gap_id = "dd_gap:equilibrium/path:unit_defect"
    initial_seen = "2026-08-01T10:00:00Z"

    class RacingGraph:
        def __init__(self) -> None:
            self.observation_ids = ["observation:original"]
            self.example_count = 1
            self.last_seen_at = initial_seen
            self.mutation_batch: list[dict[str, object]] = []

        def query(self, cypher: str, **params):
            if "RETURN version.id AS id, version.is_current AS is_current" in cypher:
                return [{"id": "4.1.1", "is_current": True}]
            if "WHERE gap.status IN $statuses" in cypher:
                return [
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
                        "observation_ids": list(self.observation_ids),
                        "example_count": self.example_count,
                        "first_seen_at": initial_seen,
                        "last_seen_at": self.last_seen_at,
                        "registry_backend": None,
                    }
                ]
            if "UNWIND $batch AS item" in cypher:
                self.mutation_batch = params["batch"]
                self.observation_ids.append("observation:concurrent")
                self.example_count += 1
                self.last_seen_at = "2026-08-02T10:00:00Z"
                candidate = self.mutation_batch[0]
                evidence_still_matches = (
                    candidate["observation_ids"] == self.observation_ids
                    and candidate["example_count"] == self.example_count
                    and candidate["last_seen_at"] == self.last_seen_at
                )
                return [{"id": gap_id}] if evidence_still_matches else []
            raise AssertionError(f"unexpected query: {cypher}")

    gc = RacingGraph()

    result = reconcile_dd_gaps(
        "4.1.1",
        {"equilibrium/path": {"unit": "Pa"}},
        gc=gc,
    )

    assert gc.observation_ids == ["observation:original", "observation:concurrent"]
    assert result["resolved"] == 0
    assert result["conflicts"] == [gap_id]
    assert gc.mutation_batch == [
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
            "observation_ids": ["observation:original"],
            "example_count": 1,
            "first_seen_at": initial_seen,
            "last_seen_at": initial_seen,
            "validation_evidence": ("4.1.1 raw unit equals 'Pa' on equilibrium/path"),
        }
    ]

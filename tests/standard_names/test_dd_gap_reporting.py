"""Tests for evidence-only Data Dictionary defect reporting."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import ConstraintError

from imas_codex.standard_names.dd_gaps import (
    DDGapRegistrySyncConflict,
    _canonical_graph_value,
    _evidence_token,
    _registry_inventory,
    _registry_migration_parameters,
    _registry_sync_plan,
    _registry_sync_token,
    _require_online_ddgap_identity_constraint,
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


def _property_fingerprint(properties: dict[str, object]) -> list[dict[str, str]]:
    return [
        {"key": key, "type": type(value).__name__, "value": str(value)}
        for key, value in sorted(properties.items())
    ]


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
    paths = source_paths if source_paths is not None else ["launcher/direction/x"]
    gap_properties = {
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
        "affected_path_count": len(paths),
        "observed_dd_version": "4.1.0",
        "observed_value": "m",
        "expected_value": "1",
        "evidence_rule": "unit_equals_expected",
        "reference_path": None,
        "reference_value": None,
    }
    path_links = [
        {
            "source_id": source_path,
            "relationship_properties": {
                "reason": "curated registry entry",
                "reporter": "registry_backfill",
            },
            "relationship_fingerprint": _property_fingerprint(
                {
                    "reason": "curated registry entry",
                    "reporter": "registry_backfill",
                }
            ),
        }
        for source_path in paths
    ]
    observation_records = [
        {
            "id": "observation:1",
            "node_properties": {
                "id": "observation:1",
                "dd_gap_id": gap_id,
                "source_path": paths[0] if paths else path,
                "reason": "curated registry entry",
                "reporter": "registry_backfill",
            },
            "relationship_properties": {},
        }
    ]
    for record in observation_records:
        record["node_fingerprint"] = _property_fingerprint(record["node_properties"])
        record["relationship_fingerprint"] = _property_fingerprint(
            record["relationship_properties"]
        )
    state_change_records = [
        {
            "id": "change:1",
            "node_properties": {
                "id": "change:1",
                "dd_gap_id": gap_id,
                "from_status": "flagged",
                "to_status": status,
                "changed_by": "operator@example.org",
            },
            "relationship_properties": {},
        }
    ]
    for record in state_change_records:
        record["node_fingerprint"] = _property_fingerprint(record["node_properties"])
        record["relationship_fingerprint"] = _property_fingerprint(
            record["relationship_properties"]
        )
    incident_links = [
        {
            "relationship_id": f"path-link:{index}",
            "relationship_type": "HAS_DD_GAP",
            "relationship_properties": item["relationship_properties"],
            "relationship_fingerprint": item["relationship_fingerprint"],
            "outgoing": False,
            "other_id": item["source_id"],
            "other_labels": ["IMASNode"],
        }
        for index, item in enumerate(path_links)
    ]
    incident_links.extend(
        [
            {
                "relationship_id": "observation-link:1",
                "relationship_type": "HAS_OBSERVATION",
                "relationship_properties": {},
                "relationship_fingerprint": [],
                "outgoing": True,
                "other_id": "observation:1",
                "other_labels": ["DDGapObservation"],
            },
            {
                "relationship_id": "state-link:1",
                "relationship_type": "HAS_STATE_CHANGE",
                "relationship_properties": {},
                "relationship_fingerprint": [],
                "outgoing": True,
                "other_id": "change:1",
                "other_labels": ["DDGapStateChange"],
            },
        ]
    )
    return {
        "id": gap_id,
        "path": path,
        "kind": kind,
        "registry_backend": backend,
        "upstream_url": upstream_url,
        "gap_properties": gap_properties,
        "gap_property_fingerprint": _property_fingerprint(gap_properties),
        "source_paths": paths,
        "path_links": path_links,
        "observation_records": observation_records,
        "state_change_records": state_change_records,
        "identity_change_records": [],
        "incident_links": incident_links,
        "direct_name_links": [],
        "source_name_links": [],
    }


def _transaction(mock_gc: MagicMock, side_effect: list[object]):
    session = MagicMock()
    transaction = MagicMock()
    mock_gc.session.return_value.__enter__.return_value = session
    mock_gc.session.return_value.__exit__.return_value = False
    session.begin_transaction.return_value = transaction
    transaction.run.side_effect = side_effect
    return transaction


def _online_constraint(mock_gc: MagicMock) -> list[list[dict[str, object]]]:
    mock_gc.schema.constraint_statements.return_value = [
        "CREATE CONSTRAINT ddgap_id IF NOT EXISTS FOR (n:DDGap) REQUIRE n.id IS UNIQUE"
    ]
    return [
        [
            {
                "name": "ddgap_id",
                "type": "UNIQUENESS",
                "labelsOrTypes": ["DDGap"],
                "properties": ["id"],
                "ownedIndex": "ddgap_id",
            }
        ],
        [{"name": "ddgap_id", "state": "ONLINE"}],
    ]


def _reclassification_case() -> tuple[str, str, dict[str, object], dict[str, object]]:
    pattern = "spi/injector/*_gas/flow_rate"
    source_path = "spi/injector/fragmentation_gas/flow_rate"
    old = _registry_fact(
        gap_id=f"dd_gap:{pattern}:unit_defect",
        path=pattern,
        source_paths=[source_path],
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
    return pattern, source_path, old, entries


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
                    "exact_count": 1,
                    "kinds": ["unit_defect"],
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


def test_registry_apply_requires_schema_declared_online_identity_constraint() -> None:
    gc = MagicMock()
    gc.schema.constraint_statements.return_value = [
        "CREATE CONSTRAINT ddgap_id IF NOT EXISTS FOR (n:DDGap) REQUIRE n.id IS UNIQUE"
    ]
    gc.query.return_value = []

    with pytest.raises(DDGapRegistrySyncConflict, match="ddgap_id.*missing"):
        _require_online_ddgap_identity_constraint(gc)

    call = gc.query.call_args
    assert call.kwargs["constraint_name"] == "ddgap_id"
    assert "SHOW CONSTRAINTS" in call.args[0]
    assert "CREATE CONSTRAINT" not in call.args[0]


def test_registry_apply_rejects_nononline_constraint_index() -> None:
    gc = MagicMock()
    gc.schema.constraint_statements.return_value = [
        "CREATE CONSTRAINT ddgap_id IF NOT EXISTS FOR (n:DDGap) REQUIRE n.id IS UNIQUE"
    ]
    gc.query.side_effect = [
        [
            {
                "name": "ddgap_id",
                "type": "UNIQUENESS",
                "labelsOrTypes": ["DDGap"],
                "properties": ["id"],
                "ownedIndex": "ddgap_id",
            }
        ],
        [{"name": "ddgap_id", "state": "POPULATING"}],
    ]

    with pytest.raises(DDGapRegistrySyncConflict, match="not ONLINE"):
        _require_online_ddgap_identity_constraint(gc)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda fact: fact["observation_records"][0]["node_properties"].update(
            {"reason": "concurrently revised observation"}
        ),
        lambda fact: fact["state_change_records"][0]["node_properties"].update(
            {"reason": "concurrently revised state evidence"}
        ),
        lambda fact: fact["path_links"][0]["relationship_properties"].update(
            {"reason": "concurrently revised path evidence"}
        ),
    ],
    ids=["observation-property", "state-property", "relationship-property"],
)
def test_registry_sync_token_covers_complete_evidence_graph(mutation) -> None:
    fact = _registry_fact()
    changed = copy.deepcopy(fact)
    mutation(changed)

    assert _registry_sync_token(changed) != _registry_sync_token(fact)


def test_registry_sync_token_serializes_temporal_values_canonically() -> None:
    observed_at = datetime(2026, 8, 2, 10, 15, tzinfo=UTC)

    assert _canonical_graph_value(observed_at) == "2026-08-02T10:15:00+00:00"


def test_registry_migration_cas_compares_complete_graph_snapshots() -> None:
    target = {
        "id": "dd_gap:pattern:self_contradiction",
        "path": "pattern",
        "kind": "self_contradiction",
        "registry_backend": "dd_unit_exceptions",
        "affected_path_count": 1,
        "observed_dd_version": "4.1.0",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
    }
    old = _registry_fact(
        gap_id="dd_gap:pattern:unit_defect",
        path="pattern",
        source_paths=["exact/path"],
    )
    plan = _registry_sync_plan(
        [target],
        [{"gap_id": target["id"], "source_path": "exact/path"}],
        [old],
    )

    params = _registry_migration_parameters(plan["reclassify"][0])

    assert params["expected_gap_properties"] == old["gap_properties"]
    assert params["expected_observation_records"] == old["observation_records"]
    assert params["expected_state_change_records"] == old["state_change_records"]
    assert params["expected_path_links"] == old["path_links"]
    assert params["expected_incident_links"] == old["incident_links"]
    assert params["expected_direct_name_links"] == old["direct_name_links"]
    assert params["expected_source_name_links"] == old["source_name_links"]
    assert (
        params["expected_gap_property_fingerprint"] == old["gap_property_fingerprint"]
    )
    assert params["expected_cas_observation_records"] == [
        {
            "id": "observation:1",
            "node_fingerprint": old["observation_records"][0]["node_fingerprint"],
            "relationship_fingerprint": [],
        }
    ]
    assert params["observation_rekeys"] == [
        {
            "old_id": "observation:1",
            "new_id": params["observation_rekeys"][0]["new_id"],
        }
    ]
    assert params["observation_rekeys"][0]["new_id"].startswith("dd_gap_observation:")


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
    old["gap_properties"]["observed_value"] = "s^-1"  # type: ignore[index]
    old["gap_properties"]["expected_value"] = "Pa.m^3.s^-1"  # type: ignore[index]
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
        *_online_constraint(mock_gc),
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
                    "state_change_count": 1,
                    "identity_change_count": 1,
                    "identity_change_id": "identity-change:registry-reclassification",
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
                    "exact_count": 1,
                    "kinds": ["self_contradiction"],
                    "source_paths": source_paths,
                }
            ],
            [{"id": old_id, "exact_count": 0}],
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
    assert migration.kwargs["target_kind"] == "self_contradiction"
    assert migration.kwargs["expected_gap_properties"]["status"] == (
        "registered_exception"
    )
    assert [
        item["source_id"] for item in migration.kwargs["expected_path_links"]
    ] == source_paths
    assert migration.kwargs["expected_observation_records"][0]["id"] == (
        "observation:1"
    )
    assert migration.kwargs["expected_state_change_records"][0]["id"] == "change:1"
    query = migration.args[0]
    assert "SET gap.id = $new_id" in query
    assert "SET item.dd_gap_id" not in query
    assert "DDGapIdentityChange" in query
    assert "HAS_IDENTITY_CHANGE" in query
    assert "epochSeconds" in query
    assert "nanosecond" in query
    assert "properties(gap) = $expected_gap_properties" not in query
    assert "CREATE (gap)-[:HAS_STATE_CHANGE]" not in query
    assert "DELETE" not in query
    set_clause = query.split("SET gap.id = $new_id", 1)[1].split("CREATE", 1)[0]
    assert "first_seen_at" not in set_clause
    assert "triaged_at" not in set_clause
    assert "resolved_dd_version" not in set_clause
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
    changed = {
        **old,
        "gap_properties": {
            **old["gap_properties"],  # type: ignore[dict-item]
            "status": "resolved_upstream",
        },
    }
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
        *_online_constraint(mock_gc),
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


@pytest.mark.parametrize(
    "mutation",
    [
        lambda fact: fact["observation_records"][0]["node_properties"].update(
            {"reason": "changed after preflight"}
        ),
        lambda fact: fact["state_change_records"][0]["node_properties"].update(
            {"reason": "changed after preflight"}
        ),
        lambda fact: fact["path_links"][0]["relationship_properties"].update(
            {"reporter": "changed-after-preflight"}
        ),
    ],
    ids=["observation-property", "state-property", "relationship-property"],
)
def test_registry_sync_rejects_concurrent_evidence_graph_drift(mutation) -> None:
    _, source_path, old, entries = _reclassification_case()
    changed = copy.deepcopy(old)
    mutation(changed)
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": source_path}],
        [old],
        *_online_constraint(mock_gc),
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
        *_online_constraint(mock_gc),
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


def test_registry_sync_rolls_back_on_concurrent_target_constraint_violation() -> None:
    _, source_path, old, entries = _reclassification_case()
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": source_path}],
        [old],
        *_online_constraint(mock_gc),
    ]
    transaction = _transaction(
        mock_gc,
        [[old], ConstraintError("concurrent target id violates DDGap uniqueness")],
    )

    with (
        _graph_context(mock_gc),
        patch(
            "imas_codex.standard_names.dd_gaps.load_exceptions",
            return_value=entries,
        ),
        pytest.raises(ConstraintError, match="concurrent target id"),
    ):
        sync_dd_unit_exception_gaps()

    transaction.commit.assert_not_called()
    transaction.rollback.assert_called_once_with()


def test_registry_sync_rejects_duplicate_target_rows_during_postverify() -> None:
    pattern, source_path, old, entries = _reclassification_case()
    old_id = str(old["id"])
    new_id = f"dd_gap:{pattern}:self_contradiction"
    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        [{"id": "4.1.0"}],
        [{"id": source_path}],
        [old],
        *_online_constraint(mock_gc),
    ]
    transaction = _transaction(
        mock_gc,
        [
            [old],
            [{"id": new_id}],
            [
                {
                    "reported": 1,
                    "relationships": 1,
                    "observations": 1,
                    "ids": [new_id],
                }
            ],
            [
                {
                    "id": new_id,
                    "exact_count": 2,
                    "kinds": ["self_contradiction", "self_contradiction"],
                    "source_paths": [source_path],
                }
            ],
            [{"id": old_id, "exact_count": 0}],
        ],
    )

    with (
        _graph_context(mock_gc),
        patch(
            "imas_codex.standard_names.dd_gaps.load_exceptions",
            return_value=entries,
        ),
        pytest.raises(DDGapRegistrySyncConflict, match="invalid=.*self_contradiction"),
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

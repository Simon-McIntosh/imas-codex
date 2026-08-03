"""Contracts for exact, governed grammar projection reconciliation."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from imas_codex.standard_names import grammar_segment_reconciliation as reconciliation
from imas_codex.standard_names.source_authority import payload_hash

_NAMES = (
    "change_in_rotation_frequency_due_to_e_cross_b_drift",
    "neutral_internal_state_atomic_power_density_due_to_collisions",
    "ratio_of_particle_count_to_particle_simulated_count",
    "ratio_of_particle_temperature_to_particle_reference_temperature",
    "volume_averaged_time_derivative_of_electron_density",
)


def _candidate(name: str, *, current: bool = False) -> dict[str, object]:
    parsed = reconciliation._segment_projection(name)
    properties: dict[str, object] = {
        "id": name,
        "name_stage": "drafted",
        "docs_stage": "pending",
        "status": None,
        "validation_status": "valid",
        "origin": "pipeline",
        "unit": "1",
        "cocos": 17,
        "physics_domain": "transport",
        "downstream_labels": ["psi_like", "ip_like"],
    }
    properties.update(parsed if current else dict.fromkeys(parsed))
    return {
        "element_id": f"node:{name}",
        "labels": ["StandardName"],
        "properties": properties,
        "relationships": [
            {
                "element_id": f"source-link:{name}",
                "type": "PRODUCED_NAME",
                "direction": "in",
                "properties": {},
                "other_element_id": f"source:{name}",
                "other_labels": ["StandardNameSource"],
                "other_id": f"dd:{name}/value",
                "other_properties": {
                    "id": f"dd:{name}/value",
                    "source_type": "dd",
                },
            }
        ],
    }


def _manifest(
    candidates: list[dict[str, object]],
) -> reconciliation.GrammarSegmentManifest:
    rows = []
    for candidate in candidates:
        name = str(candidate["properties"]["id"])
        parsed = reconciliation._segment_projection(name)
        snapshots = reconciliation._snapshots(candidate, parsed)
        rows.append(
            {
                "name": name,
                "evidence_row_hash": "e" * 64,
                "expected_before_hash": payload_hash(snapshots["before"]),
                "expected_after_hash": payload_hash(snapshots["after"]),
                "expected_identity_hash": payload_hash(snapshots["identity"]),
                "expected_protection_hash": payload_hash(snapshots["protection"]),
                "expected_participant_ids_hash": payload_hash(
                    tuple(snapshots["participant_ids"])
                ),
                "expected_relationship_ids_hash": payload_hash(
                    tuple(snapshots["relationship_ids"])
                ),
                "west_intersection": 0,
                "test_intersection": 0,
            }
        )
    names = tuple(sorted(row["name"] for row in rows))
    return reconciliation.GrammarSegmentManifest(
        path=Path("/tmp/grammar-segments.json"),
        manifest_hash="a" * 64,
        source_manifest_hash="b" * 64,
        catalog_contract_hash="c" * 64,
        rows=tuple(sorted(rows, key=lambda row: row["name"])),
        names=names,
        allowlist_hash=payload_hash(names),
    )


def _closure_rows(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"name": candidate["properties"]["id"], "matches": [candidate]}
        for candidate in sorted(candidates, key=lambda item: item["properties"]["id"])
    ]


def test_exact_production_shape_plans_five_without_mutating_catalog_guards() -> None:
    candidates = [_candidate(name) for name in _NAMES]
    manifest = _manifest(candidates)

    plans, refusals = reconciliation._plan_rows(
        _closure_rows(candidates),
        manifest,
        reason="align exact parser-derived compatibility projections",
        changed_at=None,
    )

    assert not refusals
    assert [plan["name"] for plan in plans] == list(manifest.names)
    assert {plan["status"] for plan in plans} == {"planned"}
    for candidate in candidates:
        properties = candidate["properties"]
        assert properties["cocos"] == 17
        assert properties["downstream_labels"] == ["psi_like", "ip_like"]


@pytest.mark.parametrize("size", [1, 5])
def test_planning_query_count_is_constant_in_cohort_size(size: int) -> None:
    candidates = [_candidate(name) for name in _NAMES[:size]]
    manifest = _manifest(candidates)
    transaction = Mock()
    transaction.run.return_value = _closure_rows(candidates)

    plans, refusals = reconciliation._read_plan(
        transaction,
        manifest,
        reason="audit exact projections",
        changed_at=None,
    )

    assert not refusals
    assert len(plans) == size
    assert transaction.run.call_count == 1
    assert "UNWIND $names" in transaction.run.call_args.args[0]


def test_mixed_current_and_pending_cohort_is_not_an_atomic_apply_candidate() -> None:
    stale = _candidate(_NAMES[0])
    current = _candidate(_NAMES[1], current=True)
    manifest = _manifest([stale, current])
    current_plan, _ = reconciliation._plan_rows(
        _closure_rows([current]),
        _manifest([current]),
        reason="align projections",
        changed_at="2026-08-03T00:00:00+00:00",
    )
    event = (
        current_plan[0]["event"]
        if current_plan
        else reconciliation._event_payload(
            name=_NAMES[1],
            reason="align projections",
            manifest_hash=manifest.manifest_hash,
            before_hash=manifest.rows[1]["expected_before_hash"],
            after_hash=manifest.rows[1]["expected_after_hash"],
            identity_hash=manifest.rows[1]["expected_identity_hash"],
            protection_hash=manifest.rows[1]["expected_protection_hash"],
            participant_ids_hash=manifest.rows[1]["expected_participant_ids_hash"],
            relationship_ids_hash=manifest.rows[1]["expected_relationship_ids_hash"],
            changed_at="2026-08-03T00:00:00+00:00",
        )
    )
    current["relationships"].append(
        {
            "element_id": "event-link",
            "type": "HAS_INTERNAL_CHANGE",
            "direction": "out",
            "properties": {},
            "other_element_id": "event-node",
            "other_labels": ["StandardNameChange"],
            "other_id": event["id"],
            "other_properties": event,
        }
    )
    plans, refusals = reconciliation._plan_rows(
        _closure_rows([stale, current]),
        manifest,
        reason="align projections",
        changed_at="2026-08-03T00:00:00+00:00",
    )

    assert not refusals
    assert {plan["status"] for plan in plans} == {"planned", "already_current"}
    transaction = Mock()
    session = MagicMock()
    session.begin_transaction.return_value = transaction
    client = MagicMock()
    client.session.return_value.__enter__.return_value = session
    with (
        patch.object(
            reconciliation, "load_grammar_segment_manifest", return_value=manifest
        ),
        patch.object(reconciliation, "_read_plan", return_value=(plans, [])),
    ):
        receipt = reconciliation.reconcile_grammar_segments(
            manifest.path,
            reason="align projections",
            apply=True,
            expected_manifest_hash=manifest.manifest_hash,
            gc=client,
        )

    assert receipt["mode"] == "refused"
    assert "mixed pending and current" in receipt["refusals"][0]["reasons"][0]
    transaction.rollback.assert_called_once()


def test_refuses_unparseable_nonlive_west_and_fixture_rows() -> None:
    candidate = _candidate(_NAMES[0])
    candidate["properties"]["name_stage"] = "superseded"
    candidate["relationships"][0]["other_id"] = "signals:west:fixture:one"
    manifest = _manifest([candidate])
    with patch.object(
        reconciliation,
        "_segment_projection",
        return_value={"physical_base": None},
    ):
        _, refusals = reconciliation._plan_rows(
            _closure_rows([candidate]),
            manifest,
            reason="audit projections",
            changed_at=None,
        )

    reasons = refusals[0]["reasons"]
    assert "standard name is not live" in reasons
    assert "strict public ISN parser rejected the canonical name" in reasons
    assert "current graph closure intersects WEST" in reasons

    fixture = copy.deepcopy(candidate)
    fixture["properties"]["name_stage"] = "drafted"
    fixture["relationships"][0]["other_id"] = "fixture:standard-name"
    fixture_manifest = _manifest([fixture])
    _, fixture_refusals = reconciliation._plan_rows(
        _closure_rows([fixture]),
        fixture_manifest,
        reason="audit projections",
        changed_at=None,
    )
    assert (
        "current graph closure intersects test fixtures"
        in fixture_refusals[0]["reasons"]
    )


def test_duplicate_or_tampered_event_refuses_and_exact_event_is_idempotent() -> None:
    stale = _candidate(_NAMES[0])
    manifest = _manifest([stale])
    plans, refusals = reconciliation._plan_rows(
        _closure_rows([stale]),
        manifest,
        reason="align projections",
        changed_at="2026-08-03T00:00:00+00:00",
    )
    assert not refusals
    event = plans[0]["event"]
    current = _candidate(_NAMES[0], current=True)
    event_relationship = {
        "element_id": "event-link",
        "type": "HAS_INTERNAL_CHANGE",
        "direction": "out",
        "properties": {},
        "other_element_id": "event-node",
        "other_labels": ["StandardNameChange"],
        "other_id": event["id"],
        "other_properties": event,
    }
    current["relationships"].append(event_relationship)

    repeated, repeated_refusals = reconciliation._plan_rows(
        _closure_rows([current]),
        manifest,
        reason="align projections",
        changed_at="2026-08-03T01:00:00+00:00",
    )
    assert not repeated_refusals
    assert repeated[0]["status"] == "already_current"

    duplicate = copy.deepcopy(current)
    duplicate["relationships"].append(copy.deepcopy(event_relationship))
    _, duplicate_refusals = reconciliation._plan_rows(
        _closure_rows([duplicate]),
        manifest,
        reason="align projections",
        changed_at="2026-08-03T01:00:00+00:00",
    )
    assert "duplicate or tampered" in duplicate_refusals[0]["reasons"][0]

    tampered = copy.deepcopy(current)
    tampered["relationships"][-1]["other_properties"]["reason"] = "altered"
    _, tampered_refusals = reconciliation._plan_rows(
        _closure_rows([tampered]),
        manifest,
        reason="align projections",
        changed_at="2026-08-03T01:00:00+00:00",
    )
    assert "duplicate or tampered" in tampered_refusals[0]["reasons"][0]


def test_apply_query_changes_only_runtime_parser_projection_and_writes_event() -> None:
    columns = tuple(reconciliation._segment_projection(_NAMES[0]))
    cypher = reconciliation._apply_query(columns)

    for column in columns:
        assert f"name.{column} = item.after.{column}" in cypher
    assert "StandardNameChange" in cypher
    assert "HAS_INTERNAL_CHANGE" in cypher
    for protected in (
        "cocos",
        "physics_domain",
        "unit",
        "name_stage",
        "status",
        "psi_like",
        "ip_like",
    ):
        assert f"name.{protected} =" not in cypher


def test_apply_fails_closed_when_locked_reread_drifts() -> None:
    candidate = _candidate(_NAMES[0])
    manifest = _manifest([candidate])
    plans, refusals = reconciliation._plan_rows(
        _closure_rows([candidate]),
        manifest,
        reason="align projections",
        changed_at="2026-08-03T00:00:00+00:00",
    )
    assert not refusals
    drifted = copy.deepcopy(plans)
    drifted[0]["precondition_hash"] = "f" * 64
    transaction = Mock()
    transaction.run.return_value = [{"locked": len(plans[0]["relationship_ids"])}]
    transaction.rollback = Mock()
    session = MagicMock()
    session.begin_transaction.return_value = transaction
    client = MagicMock()
    client.session.return_value.__enter__.return_value = session

    with (
        patch.object(
            reconciliation, "load_grammar_segment_manifest", return_value=manifest
        ),
        patch.object(
            reconciliation, "_read_plan", side_effect=[(plans, []), (drifted, [])]
        ),
        patch.object(reconciliation, "lock_participants"),
        patch.object(reconciliation, "_lock_relationships"),
        pytest.raises(
            reconciliation.GrammarSegmentReconciliationConflict,
            match="changed after locks",
        ),
    ):
        reconciliation.reconcile_grammar_segments(
            manifest.path,
            reason="align projections",
            apply=True,
            expected_manifest_hash=manifest.manifest_hash,
            gc=client,
        )

    transaction.rollback.assert_called()

"""Composition-time revival of reusable retired identities.

A data-dictionary source is allowed to reoccupy an independently composed
spelling only when every recorded successor is retired too.  The lifecycle
transition remains explicit and reviewable through a linked internal change.
"""

from unittest.mock import MagicMock, patch

import pytest

_SOURCE_ID = "dd:summary/global_quantities/tau_energy/value"


def _candidate(name: str) -> dict[str, object]:
    return {
        "sns_id": _SOURCE_ID,
        "source_id": _SOURCE_ID.removeprefix("dd:"),
        "source_type": "dd",
        "sn_id": name,
        "unit": "s",
        "claim_token": "compose-owner",
        "claim_seq": 8,
    }


def _binding_query(graph: MagicMock) -> tuple[str, dict[str, object]]:
    for call in graph.query.call_args_list:
        if "AS outcome" in call.args[0]:
            return call.args[0], call.kwargs
    raise AssertionError("binding query was not executed")


@pytest.mark.parametrize(
    ("name", "prior_stage"),
    [
        ("energy_confinement_time", "superseded"),
        ("current_of_poloidal_field_coil", "exhausted"),
        ("vertical_outline_of_plasma_boundary", "superseded"),
        ("neutral_pressure", "exhausted"),
    ],
)
def test_composed_dead_end_identity_is_revived_for_review(
    name: str, prior_stage: str
) -> None:
    """All four observed dead-end spellings enter the bindable path."""
    from imas_codex.standard_names.graph_ops import _lock_claimed_name_bindings

    graph = MagicMock()
    graph.query.return_value = [
        {
            "id": _SOURCE_ID,
            "outcome": "winner",
            "binding_kind": "stable_reuse",
            "candidate_id": name,
            "target_stage": "drafted",
            "attempt_count": 5,
        }
    ]

    with patch(
        "imas_codex.standard_names.graph_ops._guard_existing_target_pairings",
        return_value={},
    ):
        result = _lock_claimed_name_bindings(
            graph,
            [_candidate(name)],
            allow_missing=True,
            allow_own_pending_reservation=True,
            allow_dead_end_revival=True,
        )

    assert result[0]["outcome"] == "winner"
    assert result[0]["target_stage"] == "drafted"
    cypher, params = _binding_query(graph)
    assert params["revivable_stages"] == ["exhausted", "superseded"]
    assert params["reviewable_stage"] == "drafted"
    assert "target.name_stage IN $revivable_stages" in cypher
    assert "b.source_type = 'dd'" in cypher
    assert "MATCH (sns)-[:FROM_DD_PATH]->(:IMASNode)" in cypher
    assert "[:HAS_SUCCESSOR*1..]" in cypher
    assert "[:REFINED_FROM*1..]" in cypher
    assert "SET target.name_stage = $reviewable_stage" in cypher
    assert "target.status = 'draft'" in cypher
    assert "target.validation_status = 'pending'" in cypher
    assert "CREATE (change:StandardNameChange" in cypher
    assert "change.from_name = prior_stage" in cypher
    assert "change.to_name = $reviewable_stage" in cypher
    assert "sns.id" in cypher
    assert prior_stage in params["revivable_stages"]


def test_live_successor_keeps_retired_identity_non_bindable() -> None:
    """An occupied spelling remains retired when its chain has a live answer."""
    from imas_codex.standard_names.graph_ops import _lock_claimed_name_bindings

    graph = MagicMock()

    def _query(cypher: str, **_params: object) -> list[dict[str, object]]:
        if "AS outcome" in cypher:
            return [
                {
                    "id": _SOURCE_ID,
                    "outcome": "lifecycle_collision",
                    "binding_kind": "collision",
                    "candidate_id": "energy_confinement_time",
                    "target_stage": "superseded",
                    "attempt_count": 5,
                }
            ]
        return []

    graph.query.side_effect = _query
    with patch(
        "imas_codex.standard_names.graph_ops._guard_existing_target_pairings",
        return_value={},
    ):
        result = _lock_claimed_name_bindings(
            graph,
            [_candidate("energy_confinement_time")],
            allow_missing=True,
            allow_own_pending_reservation=True,
            allow_dead_end_revival=True,
        )

    assert result[0]["outcome"] == "lifecycle_collision"
    cypher, _params = _binding_query(graph)
    assert cypher.count("NOT EXISTS {") >= 2
    assert "successor.name_stage IS NOT NULL" in cypher
    assert "successor.name_stage IN $terminal_stages" in cypher


def test_non_composition_binding_cannot_revive_retired_identity() -> None:
    """Attachment reuse never gains the composition-only revival authority."""
    from imas_codex.standard_names.graph_ops import _lock_claimed_name_bindings

    graph = MagicMock()
    graph.query.return_value = [
        {
            "id": _SOURCE_ID,
            "outcome": "lifecycle_collision",
            "binding_kind": "collision",
            "candidate_id": "energy_confinement_time",
            "target_stage": "superseded",
            "attempt_count": 5,
        }
    ]

    with patch(
        "imas_codex.standard_names.graph_ops._guard_existing_target_pairings",
        return_value={},
    ):
        result = _lock_claimed_name_bindings(
            graph,
            [_candidate("energy_confinement_time")],
            allow_missing=False,
            allow_own_pending_reservation=False,
        )

    assert result[0]["outcome"] == "lifecycle_collision"
    _cypher, params = _binding_query(graph)
    assert params["allow_dead_end_revival"] is False

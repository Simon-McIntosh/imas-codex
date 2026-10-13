"""The signal-role choice is accepted only from the closed vocabulary."""

from __future__ import annotations

import pytest

from imas_codex.discovery.base import judgment
from imas_codex.discovery.signals import role


def test_state_uses_only_the_recorded_text() -> None:
    state = role.signal_role_state(
        {
            "id": "sample",
            "name": "Current sensor",
            "description": "Observed current",
            "unit": "A",
            "glossary": [{"term": "CS", "meaning": "current sensor"}],
            "signal_role": "commanded",
        }
    )
    assert state == {
        "signal": {
            "name": "Current sensor",
            "description": "Observed current",
            "unit": "A",
            "glossary": [{"term": "CS", "meaning": "current sensor"}],
        }
    }
    assert set(role.signal_role_questions()["signal_role"]["criteria"]) == {
        "measured",
        "commanded",
        "reference_or_setpoint",
        "status_or_flag",
        "configuration_parameter",
        "derived_or_computed",
    }


@pytest.mark.asyncio
async def test_judge_accepts_choice_and_reports_response_cost(monkeypatch) -> None:
    async def fake_decision(model, state, questions, *, service):
        assert state["signal"]["name"] in {"Current sensor", "Current target"}
        assert state["signal"]["unit"] == "A"
        assert questions["signal_role"]["type"] == "choice"
        assert service == "facility-discovery"
        choice = (
            "measured"
            if state["signal"]["name"] == "Current sensor"
            else "reference_or_setpoint"
        )
        return {"signal_role": {"choice": choice}}, 0.002

    monkeypatch.setattr(judgment, "acall_decisions", fake_decision)
    accepted, cost, failed = await role.judge_signal_roles(
        [
            {"id": "observed", "name": "Current sensor", "unit": "A"},
            {"id": "target", "name": "Current target", "unit": "A"},
        ]
    )
    assert accepted == [
        {"id": "observed", "signal_role": "measured", "cost_usd": 0.002},
        {
            "id": "target",
            "signal_role": "reference_or_setpoint",
            "cost_usd": 0.002,
        },
    ]
    assert cost == 0.004
    assert failed == []


@pytest.mark.asyncio
async def test_judge_refuses_choice_outside_six_roles(monkeypatch) -> None:
    async def fake_decision(model, state, questions, *, service):
        return {"signal_role": {"choice": "unknown"}}, 0.002

    monkeypatch.setattr(judgment, "acall_decisions", fake_decision)
    with pytest.raises(ValueError, match="outside the six roles"):
        await role.judge_signal_roles([{"id": "sample", "name": "sensor"}])

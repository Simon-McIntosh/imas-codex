"""The signal-role choice is accepted only from the closed vocabulary."""

from __future__ import annotations

import pytest

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
    assert set(role.signal_role_questions()["signal_role"]["criteria"]) == set(
        role.ROLE_CRITERIA
    )


@pytest.mark.asyncio
async def test_judge_accepts_choice_and_reports_response_cost(monkeypatch) -> None:
    async def fake_judge(rows, state_for, questions_for, apply, **kwargs):
        assert state_for(rows[0])["signal"]["unit"] == "A"
        assert questions_for()["signal_role"]["type"] == "choice"
        return (
            apply([(rows[0], {"signal_role": {"choice": "measured"}}, 0.002)], 0.002),
            0.002,
            [],
        )

    monkeypatch.setattr(role, "judge_rows", fake_judge)
    accepted, cost, failed = await role.judge_signal_roles(
        [{"id": "sample", "name": "sensor", "unit": "A"}]
    )
    assert accepted == [{"id": "sample", "signal_role": "measured", "cost_usd": 0.002}]
    assert cost == 0.002
    assert failed == []


@pytest.mark.asyncio
async def test_judge_refuses_choice_outside_six_roles(monkeypatch) -> None:
    async def fake_judge(rows, state_for, questions_for, apply, **kwargs):
        return (
            apply([(rows[0], {"signal_role": {"choice": "unknown"}}, 0.002)], 0.002),
            0.002,
            [],
        )

    monkeypatch.setattr(role, "judge_rows", fake_judge)
    with pytest.raises(ValueError, match="outside the six roles"):
        await role.judge_signal_roles([{"id": "sample", "name": "sensor"}])

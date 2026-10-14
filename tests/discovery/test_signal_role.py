"""The signal-role choice is accepted only from the closed vocabulary.

Three checks pin the role vocabulary and the routing of a role judgement: the
criteria the question offers and their descriptions, the role and cost each
accepted row carries, and the response cost summed over the batch. Each names
the row, role or cost it reads, so a path that hardcodes a criteria set, a role,
or a cost fails it rather than agreeing with the value it produced.
"""

from __future__ import annotations

import pytest

from imas_codex.discovery.base import judgment
from imas_codex.discovery.signals import role

# The closed role vocabulary, written once here so a production role set that
# drops one is compared against a value the test supplies.
SIX_ROLES = {
    "measured",
    "commanded",
    "reference_or_setpoint",
    "status_or_flag",
    "configuration_parameter",
    "derived_or_computed",
}


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
    criteria = role.signal_role_questions()["signal_role"]["criteria"]
    assert set(criteria) == SIX_ROLES, criteria
    assert all(
        isinstance(description, str) and description.strip()
        for description in criteria.values()
    ), criteria


@pytest.mark.asyncio
async def test_judge_accepts_choice_and_reports_response_cost(monkeypatch) -> None:
    async def fake_decision(model, state, questions, *, service):
        # The judge is handed each row's name, which is what decides the
        # name-keyed choice below; a path that sent the judge a state without
        # it would otherwise agree with whatever choice the branch produced.
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

    accepted_by_id = {row["id"]: row for row in accepted}
    assert set(accepted_by_id) == {"observed", "target"}, accepted
    # Each row is checked on its own: a path that routed one candidate's role
    # onto the other, or hardcoded one role onto every accepted row, fails
    # here rather than agreeing with whatever the fake judge returned.
    assert accepted_by_id["observed"]["signal_role"] == "measured", accepted
    assert accepted_by_id["target"]["signal_role"] == "reference_or_setpoint", accepted
    # The per-row cost is pinned too, so a regression that puts the batch total
    # on one row and zero on the other fails here rather than only in the sum.
    assert accepted_by_id["observed"]["cost_usd"] == 0.002, accepted
    assert accepted_by_id["target"]["cost_usd"] == 0.002, accepted
    assert cost == 0.004, cost
    assert failed == []


@pytest.mark.asyncio
async def test_judge_refuses_choice_outside_six_roles(monkeypatch) -> None:
    async def fake_decision(model, state, questions, *, service):
        return {"signal_role": {"choice": "unknown"}}, 0.002

    monkeypatch.setattr(judgment, "acall_decisions", fake_decision)
    with pytest.raises(ValueError, match="outside the six roles"):
        await role.judge_signal_roles([{"id": "sample", "name": "sensor"}])

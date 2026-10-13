"""Judge a facility signal's physical role from its stored catalogue text."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from imas_codex.discovery.base.judgment import judge_rows
from imas_codex.settings import get_model

ROLE_CRITERIA = {
    "measured": "A direct observation from an instrument, sensor, or diagnostic, including a recorded plant measurement.",
    "commanded": "An actuator command or demand sent to equipment, including an applied control waveform.",
    "reference_or_setpoint": "A desired target, programmed reference, or setpoint against which control is compared, rather than the observed value.",
    "status_or_flag": "A discrete state, alarm, validity flag, interlock, or operating status.",
    "configuration_parameter": "A setting, identifier, calibration constant, metadata item, or analysis configuration, rather than a changing physical observation.",
    "derived_or_computed": "A calculated, reconstructed, fitted, simulated, or aggregated result rather than a direct observation.",
}


def signal_role_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the judge's input independent of graph state and source paths."""
    return {
        "signal": {
            "name": str(row.get("name") or ""),
            "description": str(row.get("description") or ""),
            "unit": str(row.get("unit") or ""),
            "glossary": row.get("glossary") or [],
        }
    }


def signal_role_questions() -> dict[str, Any]:
    return {
        "signal_role": {
            "type": "choice",
            "instructions": (
                "Choose the role of this signal itself from its name, description, unit and glossary. "
                "A sensor value remains measured even when used by control; a reference remains a "
                "reference even when it is a time series. Treat reconstructed and model outputs "
                "as computed. Treat alarms as flags even if their names contain a measured quantity."
            ),
            "criteria": ROLE_CRITERIA,
        }
    }


async def judge_signal_roles(
    rows: Sequence[Mapping[str, Any]],
    *,
    concurrency: int = 8,
    budget_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], float, list[Mapping[str, Any]]]:
    """Return accepted role choices and response-usage cost without graph writes."""

    def apply(answered, _cost):
        accepted = []
        for row, answers, paid in answered:
            role = answers.get("signal_role", {}).get("choice")
            if role not in ROLE_CRITERIA:
                raise ValueError(
                    f"signal role choice {role!r} is outside the six roles"
                )
            accepted.append({"id": row["id"], "signal_role": role, "cost_usd": paid})
        return accepted

    accepted, cost, failed = await judge_rows(
        rows,
        signal_role_state,
        signal_role_questions,
        apply,
        model=get_model("discovery-relevance"),
        service="facility-discovery",
        concurrency=concurrency,
        budget_seconds=budget_seconds,
    )
    return accepted or [], cost, failed

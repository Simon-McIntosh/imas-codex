"""Physics-correctness judge for SN composer benchmark.

Scores a composed standard name against the *physical* meaning of the DD
path it was derived from, rather than against the (gameable) grammar and
convention rubric. The judge asks whether the name is faithful to what the
quantity physically is and how it is measured — e.g. a Rogowski coil
measures enclosed current via induced voltage, so naming its signal
``current_of_rogowski_coil`` mis-states the measurement principle even
though it is grammatically valid.

This module provides the verdict schema (:class:`PhysicsVerdict`) and the
benchmark test-set loader (:func:`load_bench_paths`). Later tasks add the
LLM judge call, the calibration gate, and wire the judge into
``imas_codex/standard_names/benchmark.py``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import acall_llm_structured
from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.settings import get_reasoning_effort


class PhysicsVerdict(BaseModel):
    """Per-name physics-correctness verdict from the judge.

    Each boolean axis isolates one way a composed name can be physically
    wrong; ``faithful`` is the overall conjunctive verdict, and ``reason``
    carries the judge's free-text justification.
    """

    name: str = Field(description="The composed standard name being judged.")
    faithful: bool = Field(
        description="Overall verdict: the name is physically faithful to the "
        "quantity and its measurement on every axis below."
    )
    base_correct: bool = Field(
        description="The physical base (the underlying quantity) is correct."
    )
    measurement_principle_correct: bool = Field(
        description="The name does not mis-state how the quantity is measured "
        "(e.g. a Rogowski coil measures enclosed current via induced "
        "voltage, not the coil's own current)."
    )
    qualifiers_preserved: bool = Field(
        description="Physically meaningful qualifiers from the path (locus, "
        "source, extremum, ordering) are preserved in the name."
    )
    no_over_qualification: bool = Field(
        description="The name is not over-qualified with redundant or "
        "physically unwarranted modifiers."
    )
    valid: bool = Field(description="The name is a grammatically valid standard name.")
    reason: str = Field(
        description="Free-text justification for the verdict, citing the "
        "physics where relevant."
    )


def load_bench_paths(path: Path | None = None) -> list[dict]:
    """Load the physics benchmark test set.

    Returns a list of ``{"path", "category"}`` objects. When ``path`` is
    ``None`` the default ``research/physics_bench_paths.json`` shipped with
    the repo is used, resolved relative to this module.
    """
    if path is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "research"
            / "physics_bench_paths.json"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def judge_name_physics(
    name: str, path: str, unit: str | None, documentation: str, *, model: str
) -> PhysicsVerdict:
    """Score one composed name for physical faithfulness via the rubric judge."""
    system = render_prompt("sn/judge_physics_correctness_system", {})
    user = render_prompt(
        "sn/judge_physics_correctness_user",
        {
            "name": name,
            "path": path,
            "unit": unit or "dimensionless",
            "documentation": documentation or "",
        },
    )
    result, _cost, _tokens = await acall_llm_structured(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_model=PhysicsVerdict,
        service="standard-names",
        reasoning_effort=get_reasoning_effort("sn-review"),
    )
    return result


JudgeFn = Callable[
    [str, str, str | None, str], Awaitable[PhysicsVerdict] | PhysicsVerdict
]


async def score_physics_batch(
    candidates: list[dict], judge_fn: JudgeFn
) -> list[PhysicsVerdict]:
    """Score a list of ``{name, path, unit, documentation}`` dicts.

    ``judge_fn`` may be sync (tests) or async (production
    :func:`judge_name_physics` closed over a model). Runs sequentially —
    benchmark sets are small (~180 names).
    """
    out: list[PhysicsVerdict] = []
    for c in candidates:
        res = judge_fn(c["name"], c["path"], c.get("unit"), c.get("documentation", ""))
        if hasattr(res, "__await__"):
            res = await res
        out.append(res)
    return out


async def run_calibration_gate(
    gold: list[dict], judge_fn: JudgeFn, *, min_overall_agreement: float = 0.90
) -> dict:
    """Validate the judge against human gold labels before trusting it.

    Each gold entry: {name, path, unit, documentation, faithful: bool,
    hard_case: bool}. The judge is TRUSTED iff it catches every hard-case error
    (gold faithful=False, hard_case=True) AND overall agreement on
    ``faithful`` >= min_overall_agreement. Returns a report dict.
    """
    verdicts = await score_physics_batch(gold, judge_fn)
    by_name = {v.name: v for v in verdicts}
    agree = 0
    hc_total = hc_caught = 0
    misses: list[str] = []
    for g in gold:
        v = by_name.get(g["name"])
        if v is None:
            continue
        if v.faithful == g["faithful"]:
            agree += 1
        if g.get("hard_case") and g["faithful"] is False:
            hc_total += 1
            if v.faithful is False:
                hc_caught += 1
            else:
                misses.append(g["name"])
    n = len(gold)
    overall = agree / n if n else 0.0
    trusted = (hc_total == hc_caught) and (overall >= min_overall_agreement)
    return {
        "trusted": trusted,
        "overall_agreement": round(overall, 3),
        "hardcase_errors_total": hc_total,
        "hardcase_errors_caught": hc_caught,
        "hardcase_misses": misses,
        "n_gold": n,
    }

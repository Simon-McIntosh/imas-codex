from __future__ import annotations

import anyio

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.physics_judge import (
    PhysicsVerdict,
    load_bench_paths,
    run_calibration_gate,
    score_physics_batch,
)


def test_physics_verdict_fields():
    v = PhysicsVerdict(
        name="current_of_rogowski_coil",
        faithful=False,
        base_correct=True,
        measurement_principle_correct=False,
        qualifiers_preserved=True,
        no_over_qualification=True,
        valid=True,
        reason="Rogowski coil measures enclosed current via induced voltage, not coil current.",
    )
    assert v.faithful is False
    assert v.measurement_principle_correct is False


def test_load_bench_paths():
    paths = load_bench_paths()
    assert any(p["path"] == "magnetics/rogowski_coil/current/data" for p in paths)
    assert all("category" in p for p in paths)


def test_judge_prompts_render():
    sysp = render_prompt("sn/judge_physics_correctness_system", {})
    assert "Rogowski" in sysp and "measurement_principle_correct" in sysp
    userp = render_prompt(
        "sn/judge_physics_correctness_user",
        {
            "name": "current_of_rogowski_coil",
            "path": "magnetics/rogowski_coil/current",
            "unit": "A",
            "documentation": "Measured plasma current",
        },
    )
    assert (
        "current_of_rogowski_coil" in userp
        and "magnetics/rogowski_coil/current" in userp
    )


def _stub_judge(name, path, unit, documentation):
    # deterministic: rogowski is unfaithful, everything else faithful
    bad = "rogowski" in name
    return PhysicsVerdict(
        name=name,
        faithful=not bad,
        base_correct=True,
        measurement_principle_correct=not bad,
        qualifiers_preserved=True,
        no_over_qualification=True,
        valid=True,
        reason="enclosed current" if bad else "ok",
    )


def test_score_physics_batch():
    cands = [
        {
            "name": "current_of_rogowski_coil",
            "path": "magnetics/rogowski_coil/current",
            "unit": "A",
            "documentation": "d",
        },
        {
            "name": "electron_density",
            "path": "core_profiles/profiles_1d/electrons/density",
            "unit": "m^-3",
            "documentation": "d",
        },
    ]
    verdicts = anyio.run(score_physics_batch, cands, _stub_judge)
    assert verdicts[0].faithful is False
    assert verdicts[1].faithful is True


def test_calibration_gate_trusts_when_hardcases_caught():
    gold = [
        {
            "name": "current_of_rogowski_coil",
            "path": "magnetics/rogowski_coil/current",
            "unit": "A",
            "documentation": "d",
            "faithful": False,
            "hard_case": True,
        },
        {
            "name": "electron_density",
            "path": "core_profiles/profiles_1d/electrons/density",
            "unit": "m^-3",
            "documentation": "d",
            "faithful": True,
            "hard_case": False,
        },
    ]
    rep = anyio.run(run_calibration_gate, gold, _stub_judge)
    assert rep["trusted"] is True
    assert rep["hardcase_errors_caught"] == 1
    assert rep["hardcase_errors_total"] == 1


def test_calibration_gate_distrusts_when_hardcase_missed():
    # judge that says everything faithful -> misses the rogowski hard case
    def blind(name, path, unit, documentation):
        return PhysicsVerdict(
            name=name,
            faithful=True,
            base_correct=True,
            measurement_principle_correct=True,
            qualifiers_preserved=True,
            no_over_qualification=True,
            valid=True,
            reason="ok",
        )

    gold = [
        {
            "name": "current_of_rogowski_coil",
            "path": "p",
            "unit": "A",
            "documentation": "d",
            "faithful": False,
            "hard_case": True,
        }
    ]
    rep = anyio.run(run_calibration_gate, gold, blind)
    assert rep["trusted"] is False

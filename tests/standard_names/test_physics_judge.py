from __future__ import annotations

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.physics_judge import PhysicsVerdict, load_bench_paths


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
    assert any(p["path"] == "magnetics/rogowski_coil/current" for p in paths)
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

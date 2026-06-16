from __future__ import annotations

from imas_codex.standard_names.benchmark import ModelResult, apply_physics_metrics
from imas_codex.standard_names.physics_judge import PhysicsVerdict


def test_apply_physics_metrics():
    r = ModelResult(model="m")
    verdicts = [
        PhysicsVerdict(
            name="a",
            faithful=True,
            base_correct=True,
            measurement_principle_correct=True,
            qualifiers_preserved=True,
            no_over_qualification=True,
            valid=True,
            reason="ok",
        ),
        PhysicsVerdict(
            name="b",
            faithful=False,
            base_correct=True,
            measurement_principle_correct=False,
            qualifiers_preserved=True,
            no_over_qualification=True,
            valid=True,
            reason="bad",
        ),
    ]
    hard = {"b"}
    apply_physics_metrics(r, verdicts, hard)
    assert r.physics_total == 2 and r.physics_faithful_count == 1
    assert r.physics_rate == 0.5
    assert r.physics_hardcase_total == 1 and r.physics_hardcase_faithful == 0

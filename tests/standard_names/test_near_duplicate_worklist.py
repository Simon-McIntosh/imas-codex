"""The audit-backed near-duplicate worklist and its judgment shape."""

from __future__ import annotations

import asyncio
import math

from imas_codex.standard_names.audits import synonym_check
from imas_codex.standard_names.near_duplicates import build_worklist, judge_worklist


def _name(name: str, unit: str, vector: list[float]) -> dict:
    return {
        "id": name,
        "unit": unit,
        "description": name.replace("_", " "),
        "embedding": vector,
    }


def test_worklist_filters_units_threshold_and_groups_overlaps(monkeypatch):
    names = [
        _name("neutral_density", "m^-3", [1, 0, 0, 0]),
        _name("neutral_state_density", "m^-3", [1, 0, 0, 0]),
        _name("neutral_temperature", "eV", [0, 1, 0, 0]),
        _name("neutral_state_temperature", "eV", [0, 1, 0, 0]),
        _name("neutral_pressure", "Pa", [1, 0, 0, 0]),
        _name("electron_density", "m^-3", [0.93, math.sqrt(1 - 0.93**2), 0, 0]),
    ]
    worklist = build_worklist(names, west_names={"neutral_density"})
    assert [(pair["left"], pair["right"]) for pair in worklist["west"]] == [
        ("neutral_density", "neutral_state_density")
    ]
    assert [(pair["left"], pair["right"]) for pair in worklist["other"]] == [
        ("neutral_state_temperature", "neutral_temperature")
    ]
    first, second = worklist["west"][0], worklist["other"][0]
    assert first["cosine"] >= 0.95
    assert first["overlap"] == second["overlap"]
    assert set(first["divergent_segments"]) == {"subject"}
    assert not any(
        "pressure" in name
        for pairs in worklist.values()
        for pair in pairs
        for name in (pair["left"], pair["right"])
    )

    async def stub(states, questions, *, model, service):
        assert model == "seat-model"
        assert service == "standard-names"
        assert len(states) == 2
        assert len(questions) == 7
        return [
            (
                {
                    "quantity_relation": {
                        "type": "score",
                        "score": 2,
                        "probabilities": {0: 0.05, 1: 0.1, 2: 0.85},
                    },
                    **{
                        key: {"type": "noul", "noul": 0.2}
                        for key in questions
                        if key.startswith("differs_")
                    },
                },
                0.003,
            )
            for _ in states
        ], 0.006

    monkeypatch.setattr("imas_codex.standard_names.near_duplicates.decide_batch", stub)
    judged, cost = asyncio.run(
        judge_worklist(worklist["west"] + worklist["other"], model="seat-model")
    )
    assert cost == 0.006
    assert all(pair["verdict"] == "same quantity" for pair in judged)
    assert all(
        set(pair["segment_judgments"])
        == {"physical_base", "subject", "component", "operator", "locus", "process"}
        for pair in judged
    )


def test_synonym_check_preserves_default_and_can_select_worklist_threshold():
    candidate = {
        "id": "neutral_density",
        "unit": "m^-3",
        "description_embedding": [1, 0],
    }
    similar = {
        "name": "neutral_state_density",
        "unit": "m^-3",
        "description_embedding": [0.94, math.sqrt(1 - 0.94**2)],
    }
    different_unit = {
        "name": "neutral_pressure",
        "unit": "Pa",
        "description_embedding": [1, 0],
    }
    assert len(synonym_check(candidate, [similar, different_unit])) == 1
    assert synonym_check(candidate, [similar], threshold=0.95, inclusive=True) == []
    assert synonym_check(candidate, [different_unit]) == []
    assert len(synonym_check(candidate, [different_unit], same_unit_only=False)) == 1
    identical = {**similar, "description_embedding": [1, 0]}
    assert (
        len(synonym_check(candidate, [identical], threshold=1.0, inclusive=True)) == 1
    )


def test_pairs_sharing_a_name_form_one_overlap():
    names = [
        _name("fast_electron_density", "m^-3", [1, 0]),
        _name("thermal_electron_density", "m^-3", [1, 0]),
        _name("total_electron_density", "m^-3", [1, 0]),
    ]
    worklist = build_worklist(names, west_names={"fast_electron_density"})
    pairs = worklist["west"] + worklist["other"]
    assert len(worklist["west"]) == 2
    assert len(worklist["other"]) == 1
    assert len({pair["overlap"] for pair in pairs}) == 1
    assert all(pair["overlap_size"] == 3 for pair in pairs)


def test_shared_name_does_not_merge_different_grammar_axes():
    names = [
        _name("neutral_density", "m^-3", [1, 0]),
        _name("neutral_state_density", "m^-3", [1, 0]),
        _name("poloidal_neutral_density", "m^-3", [1, 0]),
    ]
    pairs = build_worklist(names)["other"]
    assert len(pairs) == 3
    assert len({pair["overlap"] for pair in pairs}) == 3

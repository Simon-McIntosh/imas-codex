"""Segment selection: chosen tokens render, refusals are recorded, the beam prunes.

The decisions layer is stubbed, so every test here exercises the module's own
logic — the tree walk, the beam and the render — against answers the test owns.
Each test names the mechanism it guards, and
``SS_SEGMENT_MUTATION`` removes that mechanism so the test can be shown to fail
without it:

``no-beam-prune``      the beam keeps every path, so a partition the beam would
                       have pruned re-enters the ranking
``no-grammar-refusal`` :func:`render_name` skips ISN's strict parse, so a name
                       the grammar refuses is reported as rendered
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from imas_codex.standard_names import segment_selection as ss

QUESTION = ss.QUESTION_NAME
NONE = ss.NONE


def _source(path: str = "/dd/electron/density", **extra: Any) -> dict[str, Any]:
    source: dict[str, Any] = {
        "path": path,
        "description": "the electron number density at the plasma boundary",
        "documentation": "Number density of electrons.",
    }
    source.update(extra)
    return source


def _distribution(
    criteria: Mapping[str, str], weights: Mapping[str, float]
) -> tuple[str, dict[str, float]]:
    """Turn raw weights into a normalised distribution over ``criteria``."""
    raw = {key: float(weights.get(key, 0.0)) for key in criteria}
    total = sum(raw.values())
    if total <= 0.0:
        raw = {key: 1.0 / len(criteria) for key in criteria}
        total = 1.0
    probs = {key: value / total for key, value in raw.items()}
    choice = max(probs, key=lambda key: (probs[key], key))
    return choice, probs


def _stub_decisions(weights_for):
    """A ``decide_batch`` stand-in returning ``weights_for``'s distribution.

    ``weights_for(question_criteria, state)`` returns raw per-option weights.
    The stub shapes them exactly as the endpoint would: a chosen option that is
    the argmax of the normalised distribution, plus the full distribution.
    """

    async def fake_decide_batch(states, questions, **kwargs):
        criteria = questions[QUESTION]["criteria"]
        results = []
        for state in states:
            weights = weights_for(criteria, state)
            choice, probs = _distribution(criteria, weights)
            results.append(
                ({QUESTION: {"choice": choice, "probabilities": probs}}, 0.0005)
            )
        return results, 0.0005 * len(results)

    return fake_decide_batch


def _choose_by_slot(choice_by_slot: Mapping[str, str]) -> Any:
    """A weights function picking ``choice_by_slot[slot]`` where offered."""

    def weights_for(
        criteria: Mapping[str, str], state: Mapping[str, Any]
    ) -> dict[str, float]:
        slot = state["slot"]
        want = choice_by_slot.get(slot, NONE)
        if want not in criteria:
            want = NONE if NONE in criteria else next(iter(criteria))
        return {key: (1.0 if key == want else 0.001) for key in criteria}

    return weights_for


@pytest.fixture(autouse=True)
def _mutation(monkeypatch: pytest.MonkeyPatch):
    """Apply the declared mutation, when one is named, for the negative control."""
    mutation = os.environ.get("SS_SEGMENT_MUTATION", "")
    if mutation == "no-beam-prune":
        monkeypatch.setattr(ss, "BEAM", 10_000)
    elif mutation == "no-grammar-refusal":
        original = ss.render_name

        def unrefusing(chosen):
            from imas_standard_names.grammar.render import compose

            return compose(ss._build_ir(chosen))

        monkeypatch.setattr(ss, "render_name", unrefusing)
        assert ss.render_name is not original
    yield


def test_chosen_tokens_render_through_isn(monkeypatch: pytest.MonkeyPatch):
    """A chosen token set renders to the name ISN accepts for it."""
    monkeypatch.setattr(
        ss,
        "decide_batch",
        _stub_decisions(
            _choose_by_slot({"physical_base": "density", "subject": "electron"})
        ),
    )
    run = _run([_source()])
    item = run.items[0]
    assert item.failure is None
    assert item.name == "electron_density"
    assert item.chosen()["physical_base"] == "density"
    assert item.chosen()["subject"] == "electron"


def test_grammar_refusal_is_recorded_and_not_repaired(monkeypatch: pytest.MonkeyPatch):
    """A name the grammar refuses is a failure on the item, never a fixed string."""
    # 'pressure' is a registered physical base that the grammar itself refuses
    # unless qualified ('generic base requires qualification'), so a source the
    # model slots it into bare is refused by ISN, not repaired here.
    monkeypatch.setattr(
        ss,
        "decide_batch",
        _stub_decisions(_choose_by_slot({"physical_base": "pressure"})),
    )
    run = _run([_source()])
    item = run.items[0]
    assert item.name is None
    assert item.failure is not None
    assert "refused" in item.failure
    # The source is still returned with its selections; only the name is refused.
    assert item.chosen()["physical_base"] == "pressure"


def _run(sources: Sequence[Mapping[str, Any]]) -> ss.SelectionRun:
    import asyncio

    return asyncio.run(ss.select_segments(list(sources), model="test-model"))


# ---------------------------------------------------------------------------
# The beam
# ---------------------------------------------------------------------------

_ENTITY_BEST = "antenna_strap"
_POSITION_BEST = "strike_point"
_REGION_BEST = "core_region"


def test_beam_keeps_the_three_best_paths_by_geometric_mean(
    monkeypatch: pytest.MonkeyPatch,
):
    """The beam keeps the top three paths, ranked by geometric-mean score.

    The locus slot's vocabulary ISN partitions by registry ``type``, so its tree
    has the four partitions entity / position / region / geometry. The weights
    are chosen so the beam's level-1 pruning is what decides the ranking: the
    ``geometry`` partition carries the single most-confident token (weight 1.0)
    but only 0.04 of the mass, so the beam drops it before its cheap token can
    win, and the third best path is region's. With pruning removed the geometry
    leaf (0.04) outranks region's (0.018) and the ranking changes.
    """
    asked_partitions: list[str] = []

    def weights_for(
        criteria: Mapping[str, str], state: Mapping[str, Any]
    ) -> dict[str, float]:
        route = state["route"]
        if not route:
            return {"entity": 0.6, "position": 0.3, "region": 0.06, "geometry": 0.04}
        partition = route[0]
        asked_partitions.append(partition)
        best = {
            "entity": _ENTITY_BEST,
            "position": _POSITION_BEST,
            "region": _REGION_BEST,
        }.get(partition)
        weights = dict.fromkeys(criteria, 0.0)
        if partition == "geometry":
            weights[next(iter(criteria))] = 1.0
        elif best is not None:
            share = {"entity": 1.0, "position": 1.0, "region": 0.66}[partition]
            weights[best] = share
            weights[NONE] = 1.0 - share
        return weights

    monkeypatch.setattr(ss, "decide_batch", _stub_decisions(weights_for))

    import asyncio

    tree = ss.build_slot_tree(ss.SLOT_LOCUS)
    ranked = asyncio.run(
        ss._search_slot(
            tree,
            [_source()],
            model="test-model",
            cost=None,
            spent=[],
            concurrency=1,
            budget_seconds=None,
        )
    )
    paths = ranked[0]

    assert [path.token for path in paths] == [
        _ENTITY_BEST,
        _POSITION_BEST,
        _REGION_BEST,
    ]
    assert paths[0].score == pytest.approx(math.sqrt(0.6 * 1.0))
    assert paths[1].score == pytest.approx(math.sqrt(0.3 * 1.0))
    assert paths[2].score == pytest.approx(math.sqrt(0.06 * 0.66))
    # Geometry's single confident token (0.04 total) sits just above region's
    # leaf (0.0396), so pruning geometry at level 1 is what makes region, not
    # geometry, the third path; without pruning the ranking changes.
    # The beam pruned geometry at level 1, so its tokens were never asked about.
    assert "geometry" not in asked_partitions

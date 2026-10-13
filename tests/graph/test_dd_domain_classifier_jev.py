"""The two-level category-then-domain Choice over the physics-domain guidance.

The tree asks which ISN category a DD path belongs to, then which domain within
that category, and multiplies the two probabilities into one figure saying how
confident it is. A path below the confidence threshold, or one whose Jev call
failed, is re-judged by the chat classifier reading the same guidance.

Every test here stubs both endpoints — the decisions batch and the chat call —
so the design is measured without a paid call. Each test names the mechanism it
holds, so removing that mechanism turns the test red rather than leaving it
green by accident.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.definitions.physics import (
    load_domain_guidance,
    load_isn_physics_domains,
)
from imas_codex.graph.dd_domain_classifier import (
    CATEGORY_BEAM_MARGIN,
    CATEGORY_QUESTION,
    DOMAIN_QUESTION,
    _category_options,
    _classify_batch_tree,
    _domain_options,
    build_category_question,
    build_domain_question,
    decide_domain_tree,
    gather_classification_context,
    tree_state,
)

PATH = "core_profiles/profiles_1d/electrons/temperature"


def context_row(**overrides) -> dict:
    """One ``gather_classification_context`` row, carrying the full context."""
    row = {
        "id": PATH,
        "description": "Electron temperature",
        "documentation": "Temperature of the thermal electron population.",
        "units": "eV",
        "parent_path": "core_profiles/profiles_1d/electrons",
        "parent_description": "Electron species quantities",
        "siblings": ["density", "pressure"],
        "ids_name": "core_profiles",
        "node_category": "quantity",
    }
    return {**row, **overrides}


def make_gc(row: dict) -> MagicMock:
    """A GraphClient whose context query returns *row*."""
    gc = MagicMock()
    gc.query.return_value = [row]
    return gc


def gathered(path: str = PATH) -> dict:
    """The context dict for one path, as the gatherer produces it."""
    return gather_classification_context(make_gc(context_row(id=path)), [path])[0]


def category_for(criteria: dict) -> str | None:
    """Which category a second-level criteria set belongs to."""
    offered = set(criteria)
    for category in _category_options():
        if set(_domain_options(category)) == offered:
            return category
    return None


class ScriptedDecisions:
    """The decisions endpoint replaced by a scripted map, per question level.

    Answers are keyed by the DD path a state carries, so a test states what the
    model would decide rather than where a path sits in a batch. A path mapped
    to ``None`` models a failed call. A second-level answer is keyed by
    ``(category, path)``, since one path is asked in each beamed category.
    """

    def __init__(
        self,
        categories: dict[str, tuple[str, dict[str, float]] | None],
        domains: dict[tuple[str, str], tuple[str, dict[str, float]]] | None = None,
        cost_per_state: float = 0.01,
    ) -> None:
        self.categories = categories
        self.domains = domains or {}
        self.cost_per_state = cost_per_state
        self.calls: list[dict] = []
        self.categories_asked: list[str] = []

    async def __call__(self, states, questions, **kwargs):
        self.calls.append(
            {"states": list(states), "questions": questions, "kwargs": kwargs}
        )
        results = []
        if CATEGORY_QUESTION in questions:
            criteria = questions[CATEGORY_QUESTION]["criteria"]
            for state in states:
                answer = self.categories.get(state["path"])
                if answer is None:
                    results.append(None)
                    continue
                choice, probabilities = answer
                assert choice in criteria, (
                    f"category {choice!r} is not offered: {sorted(criteria)}"
                )
                results.append(
                    (
                        {"choice": choice, "probabilities": probabilities},
                        self.cost_per_state,
                    )
                )
        else:
            criteria = questions[DOMAIN_QUESTION]["criteria"]
            category = category_for(criteria)
            assert category is not None, (
                "the second level was asked a set of domains no category contains"
            )
            self.categories_asked.append(category)
            for state in states:
                answer = self.domains.get((category, state["path"]))
                if answer is None:
                    results.append(None)
                    continue
                choice, probabilities = answer
                assert choice in criteria, (
                    f"domain {choice!r} is not offered by {category}: {sorted(criteria)}"
                )
                results.append(
                    (
                        {"choice": choice, "probabilities": probabilities},
                        self.cost_per_state,
                    )
                )
        return results, self.cost_per_state * len(results)


# ---------------------------------------------------------------------------
# The state every decision reads
# ---------------------------------------------------------------------------


def test_context_carries_the_path_the_description_and_the_documentation() -> None:
    """The gatherer collects the DD documentation clause the classifiers need.

    The chat classifier was sent a documentation field the gatherer never
    collected, so it always arrived empty. It is read from the graph now, and a
    dropped column turns this red.
    """
    context = gather_classification_context(make_gc(context_row()), [PATH])
    assert len(context) == 1
    assert context[0]["documentation"] == (
        "Temperature of the thermal electron population."
    )
    assert context[0]["description"] == "Electron temperature"


def test_tree_state_carries_every_context_field() -> None:
    """Both tree levels read one state holding the whole gathered context."""
    state = tree_state(gathered())
    assert state["path"] == PATH
    assert state["description"] == "Electron temperature"
    assert state["documentation"] == ("Temperature of the thermal electron population.")
    assert state["unit"] == "eV"
    assert state["parent"] == "core_profiles/profiles_1d/electrons"
    assert state["parent_description"] == "Electron species quantities"
    assert state["siblings"] == ["density", "pressure"]
    assert state["ids_name"] == "core_profiles"
    assert state["node_category"] == "quantity"


@pytest.mark.asyncio
async def test_the_endpoint_is_shown_the_description_and_the_documentation() -> None:
    """A decision is made from the full state, not from the path alone.

    A field the transport dropped would part here rather than silently changing
    a decision, so this holds the arrival of the context as well as its
    collection.
    """
    category = sorted(_category_options())[0]
    domain = sorted(_domain_options(category))[0]
    scripted = ScriptedDecisions(
        categories={PATH: (category, {category: 1.0})},
        domains={(category, PATH): (domain, {domain: 1.0})},
    )

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=scripted):
        await decide_domain_tree([gathered()], model="tree-model")

    assert scripted.calls, "the tree asked no question at all"
    for call in scripted.calls:
        assert call["states"][0]["description"] == "Electron temperature"
        assert call["states"][0]["documentation"] == (
            "Temperature of the thermal electron population."
        )


# ---------------------------------------------------------------------------
# The questions are drawn from the vocabulary, not from a frozen list
# ---------------------------------------------------------------------------


def test_category_options_are_exactly_the_isn_categories_with_their_scopes() -> None:
    """The first level offers one option per ISN category, carrying its scope."""
    isn_categories = {
        meta["category"]
        for meta in load_isn_physics_domains().values()
        if meta["category"]
    }
    assert isn_categories, (
        "ISN vocabulary read empty — the check cannot see its subject"
    )
    scopes = load_domain_guidance()["categories"]

    options = _category_options()
    assert set(options) == isn_categories
    for category, scope in options.items():
        assert scope == scopes[category]["scope"]
        assert scope.strip(), f"{category}: category offered with an empty scope"


def test_category_options_follow_the_vocabulary_not_a_frozen_list(monkeypatch) -> None:
    """A category added to the ISN vocabulary appears without a code change.

    A frozen list of categories would pass every test written against today's
    vocabulary and stop covering the vocabulary the day it grows.
    """
    vocabulary = load_isn_physics_domains()
    assert vocabulary, "ISN vocabulary read empty — the check cannot see its subject"

    extended = dict(vocabulary)
    extended["sentinel_domain_guidance"] = {
        "description": "A domain added upstream",
        "category": "sentinel_category_xyz",
        "ids": [],
    }
    monkeypatch.setattr(
        "imas_codex.definitions.physics.load_isn_physics_domains", lambda: extended
    )
    monkeypatch.setattr(
        "imas_codex.definitions.physics.load_domain_guidance",
        lambda: {
            "categories": {"sentinel_category_xyz": {"scope": "added upstream"}},
            "domains": {},
        },
    )
    options = _category_options()
    assert "sentinel_category_xyz" in options
    assert options["sentinel_category_xyz"] == "added upstream"


def test_domain_options_are_exactly_that_categorys_domains_with_their_guidance() -> (
    None
):
    """The second level offers one option per domain in the chosen category.

    Each option carries the domain's ISN one-line description beside its
    guidance scope, its included quantity kinds and its exclusions, so the
    question states what each domain covers rather than naming it.
    """
    vocabulary = load_isn_physics_domains()
    guidance = load_domain_guidance()["domains"]

    for category in _category_options():
        expected = {
            value for value, meta in vocabulary.items() if meta["category"] == category
        }
        options = _domain_options(category)
        assert set(options) == expected, f"{category}: domain set diverges from ISN"
        for value in expected:
            text = options[value]
            entry = guidance[value]
            assert vocabulary[value]["description"] in text
            assert entry["scope"].strip() in text
            for included in entry["includes"]:
                assert included in text, f"{value}: includes dropped {included!r}"
            for exclusion in entry["excludes"]:
                assert exclusion["quantity"] in text
                assert exclusion["neighbour"] in text


def test_every_offered_domain_belongs_to_the_category_that_offers_it() -> None:
    """No domain leaks across the categories the first level can return."""
    for category in _category_options():
        for value in _domain_options(category):
            assert load_isn_physics_domains()[value]["category"] == category


def test_both_questions_are_built_from_the_vocabulary() -> None:
    """Each question builder returns the options its reader produces."""
    first = build_category_question()
    assert first[CATEGORY_QUESTION]["type"] == "choice"
    assert first[CATEGORY_QUESTION]["criteria"] == _category_options()

    category = sorted(_category_options())[0]
    second = build_domain_question(category)
    assert second[DOMAIN_QUESTION]["type"] == "choice"
    assert second[DOMAIN_QUESTION]["criteria"] == _domain_options(category)


# ---------------------------------------------------------------------------
# The beam at the first level
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_near_tie_asks_the_domain_question_in_both_categories() -> None:
    """A first-level near tie is resolved by asking both, not by guessing.

    The second level is cheap beside committing to a category the first level
    so narrowly preferred, so the runner-up is asked as well.
    """
    winner, runner_up = sorted(_category_options())[:2]
    others = [name for name in _category_options() if name not in (winner, runner_up)]
    distribution = {winner: 0.4, runner_up: 0.39}
    for name in others:
        distribution[name] = 0.01

    scripted = ScriptedDecisions(
        categories={PATH: (winner, distribution)},
        domains={
            (winner, PATH): (
                sorted(_domain_options(winner))[0],
                {sorted(_domain_options(winner))[0]: 0.9},
            ),
            (runner_up, PATH): (
                sorted(_domain_options(runner_up))[0],
                {sorted(_domain_options(runner_up))[0]: 0.9},
            ),
        },
    )

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=scripted):
        decisions, _cost = await decide_domain_tree([gathered()], model="m")

    assert set(scripted.categories_asked) == {winner, runner_up}
    assert CATEGORY_BEAM_MARGIN > abs(0.4 - 0.39)
    decision = decisions[0]
    assert decision is not None
    assert decision["physics_category"] == winner


@pytest.mark.asyncio
async def test_a_clear_lead_asks_only_the_leading_category() -> None:
    """A confident first level asks one category, not two.

    The beam exists for ties; asking every category regardless would double the
    second-level cost to buy nothing where the first level was decisive.
    """
    winner, runner_up = sorted(_category_options())[:2]
    distribution = {winner: 0.95, runner_up: 0.02}
    for name in _category_options():
        distribution.setdefault(name, 0.01)

    scripted = ScriptedDecisions(
        categories={PATH: (winner, distribution)},
        domains={
            (winner, PATH): (
                sorted(_domain_options(winner))[0],
                {sorted(_domain_options(winner))[0]: 0.9},
            )
        },
    )

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=scripted):
        await decide_domain_tree([gathered()], model="m")

    assert scripted.categories_asked == [winner]


@pytest.mark.asyncio
async def test_the_probability_is_the_product_of_the_two_levels() -> None:
    """A path's confidence is the two levels multiplied, and it is recorded.

    One figure has to carry both levels' uncertainty, or a confident domain
    under a doubtful category would read as a confident decision.
    """
    category = sorted(_category_options())[0]
    domain = sorted(_domain_options(category))[0]
    scripted = ScriptedDecisions(
        categories={PATH: (category, {category: 0.5})},
        domains={(category, PATH): (domain, {domain: 0.4})},
    )

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=scripted):
        decisions, _cost = await decide_domain_tree([gathered()], model="m")

    decision = decisions[0]
    assert decision is not None
    assert decision["physics_domain"] == domain
    assert decision["physics_category"] == category
    assert decision["category_probability"] == pytest.approx(0.5)
    assert decision["domain_probability"] == pytest.approx(0.4)
    assert decision["probability"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_a_near_tie_is_decided_by_the_best_product_not_the_best_category() -> (
    None
):
    """The second level can overturn a first-level lead, and does.

    The winner is the (category, domain) pair with the highest product, so a
    narrower category lead loses to a decisive domain answer beneath it.
    """
    winner, runner_up = sorted(_category_options())[:2]
    winner_domain = sorted(_domain_options(winner))[0]
    runner_up_domain = sorted(_domain_options(runner_up))[0]
    distribution = {winner: 0.45, runner_up: 0.44}
    for name in _category_options():
        distribution.setdefault(name, 0.01)

    scripted = ScriptedDecisions(
        categories={PATH: (winner, distribution)},
        domains={
            (winner, PATH): (winner_domain, {winner_domain: 0.5}),
            (runner_up, PATH): (runner_up_domain, {runner_up_domain: 0.99}),
        },
    )

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=scripted):
        decisions, _cost = await decide_domain_tree([gathered()], model="m")

    decision = decisions[0]
    assert decision is not None
    assert decision["physics_category"] == runner_up
    assert decision["probability"] == pytest.approx(0.44 * 0.99)


# ---------------------------------------------------------------------------
# Chat below the threshold, and when a Jev call fails
# ---------------------------------------------------------------------------


def chat_stub(reply: str = "general", cost: float = 0.02) -> AsyncMock:
    """A chat classifier that answers every path it is handed, recording them."""
    stub = AsyncMock()

    async def _call(batch, *, model, service, valid_domains):
        stub.batches.append([b["id"] for b in batch])
        results = [
            {
                "id": b["id"],
                "physics_domain": reply,
                "domain_source": "llm_classified",
                "domain_model": model,
                "cost": cost,
            }
            for b in batch
        ]
        return results, cost

    stub.side_effect = _call
    stub.batches = []
    return stub


def tree_batch(
    category: str,
    domain: str,
    category_probability: float,
    domain_probability: float,
    cost_per_state: float = 0.005,
):
    """One path, scripted for a single-category tree run at a chosen confidence.

    The per-state cost is half a cent so the two levels of one path sum to a
    cent, distinguishable from the chat call's two.
    """
    scripted = ScriptedDecisions(
        categories={PATH: (category, {category: category_probability})},
        domains={(category, PATH): (domain, {domain: domain_probability})},
        cost_per_state=cost_per_state,
    )
    return scripted


@pytest.mark.asyncio
async def test_a_path_below_the_threshold_goes_to_the_chat_classifier() -> None:
    """A low-confidence path is re-judged by chat; a confident one is kept.

    The tree's probability is what routes the path: 0.45 is below the default
    threshold and reaches the chat call, 0.95 does not.
    """
    category = sorted(_category_options())[0]
    domain = sorted(_domain_options(category))[0]
    low = tree_batch(category, domain, 0.5, 0.9)
    chat = chat_stub(reply="transport", cost=0.02)

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=low):
        with patch("imas_codex.graph.dd_domain_classifier._classify_batch", new=chat):
            results, _cost = await _classify_batch_tree(
                [gathered()],
                model="tree-model",
                chat_model="chat-model",
                service="data-dictionary",
                valid_domains={domain, "general", "transport"},
            )

    assert chat.batches == [[PATH]], "the low-confidence path did not reach chat"
    assert results[0]["physics_domain"] == "transport"
    assert results[0]["domain_source"] != "jev_tree"
    assert results[0]["tree_probability"] == pytest.approx(0.45)

    high = tree_batch(category, domain, 1.0, 0.95)
    chat = chat_stub(reply="transport")
    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=high):
        with patch("imas_codex.graph.dd_domain_classifier._classify_batch", new=chat):
            results, _cost = await _classify_batch_tree(
                [gathered()],
                model="tree-model",
                chat_model="chat-model",
                service="data-dictionary",
                valid_domains={domain, "general", "transport"},
            )

    assert chat.batches == [], "a confident path was sent to chat anyway"
    assert results[0]["physics_domain"] == domain
    assert results[0]["domain_source"] == "jev_tree"
    assert results[0]["probability"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_a_failed_jev_call_falls_back_to_chat() -> None:
    """A Jev call that returns nothing routes its path to the chat classifier."""
    category = sorted(_category_options())[0]
    failed = ScriptedDecisions(categories={PATH: None})
    chat = chat_stub(reply="transport")

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=failed):
        with patch("imas_codex.graph.dd_domain_classifier._classify_batch", new=chat):
            results, _cost = await _classify_batch_tree(
                [gathered()],
                model="tree-model",
                chat_model="chat-model",
                service="data-dictionary",
                valid_domains={category, "general", "transport"},
            )

    assert failed.categories_asked == [], "a failed first level asked its second"
    assert chat.batches == [[PATH]]
    assert results[0]["physics_domain"] == "transport"


@pytest.mark.asyncio
async def test_a_failed_chat_call_leaves_the_path_for_tier_two() -> None:
    """A chat call that answers nothing contributes no result for its path.

    The path is left unwritten rather than mislabelled, so tier 2 inheritance
    picks it up instead of a guess being written under the tree's name.
    """
    category = sorted(_category_options())[0]
    domain = sorted(_domain_options(category))[0]
    low = tree_batch(category, domain, 0.2, 0.9)

    async def _empty(batch, *, model, service, valid_domains):
        return [], 0.0

    chat = AsyncMock(side_effect=_empty)

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=low):
        with patch("imas_codex.graph.dd_domain_classifier._classify_batch", new=chat):
            results, _cost = await _classify_batch_tree(
                [gathered()],
                model="tree-model",
                chat_model="chat-model",
                service="data-dictionary",
                valid_domains={domain},
            )

    assert results == [], "an unanswered path was written anyway"


@pytest.mark.asyncio
async def test_costs_from_both_classifiers_reach_the_tracker() -> None:
    """The tree's cost reaches the tracker and the chat cost joins the total.

    The tracker is handed to both tree levels, and the chat half's cost is
    added to the total the caller books, so a run accounts for what it spent
    whichever classifier decided a path.
    """
    category = sorted(_category_options())[0]
    domain = sorted(_domain_options(category))[0]
    low = tree_batch(category, domain, 0.5, 0.9)
    chat = chat_stub(cost=0.02)
    tracker = MagicMock()

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=low):
        with patch("imas_codex.graph.dd_domain_classifier._classify_batch", new=chat):
            _results, total = await _classify_batch_tree(
                [gathered()],
                model="tree-model",
                chat_model="chat-model",
                service="data-dictionary",
                valid_domains={domain, "general"},
                cost=tracker,
            )

    assert total == pytest.approx(0.01 + 0.02)
    for call in low.calls:
        assert call["kwargs"]["cost"] is tracker, "a tree level lost the cost tracker"


@pytest.mark.asyncio
async def test_classify_tier1_llm_reports_both_classifiers_costs() -> None:
    """Both classifiers' costs reach the run's on_cost callback.

    A run that spent on the tree and again on chat reports the summed spend, so
    the cost tracker records what the classification actually cost.
    """
    from imas_codex.graph.dd_domain_classifier import classify_tier1_llm

    category = sorted(_category_options())[0]
    domain = sorted(_domain_options(category))[0]
    low = tree_batch(category, domain, 0.5, 0.9)
    chat = chat_stub(cost=0.02)
    spent: list[float] = []
    gc = make_gc(context_row())

    with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=low):
        with patch("imas_codex.graph.dd_domain_classifier._classify_batch", new=chat):
            await classify_tier1_llm(
                gc,
                [{"id": PATH, "node_category": "quantity"}],
                model="chat-model",
                tree_model="tree-model",
                on_cost=spent.append,
            )

    assert spent[0] == pytest.approx(0.01 + 0.02), (
        f"the run did not book the tree's cost and the chat call's: {spent}"
    )

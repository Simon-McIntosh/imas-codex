"""Segment selection: meaning-bearing options, a shortlisted base, valid slots.

The decisions layer and the embedding client are stubbed, so every test here
exercises the module's own logic — the shortlist, the sequence, the tree walk,
the beam and the render — against answers the test owns.

``SS_SEGMENT_MUTATION`` removes one mechanism so the test that guards it can be
shown to fail without it:

``base-offers-none``    the base Choice offers ``none``, so a name could have no
                        base
``bare-option-text``    options are the bare token, so a near-synonym must be
                        judged from its spelling alone
``no-shortlist``        the base Choice is offered the whole base vocabulary
                        instead of the ranked shortlist
``no-grammar-filter``   a later slot offers every token, grammar-valid or not
``no-optional-none``    optional slots lose their ``none`` option
``no-beam-prune``       the beam keeps every path
``no-grammar-refusal``  ``render_name`` skips ISN's strict parse, so a name the
                        grammar refuses is reported as rendered
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from imas_codex.standard_names import (
    segment_selection as ss,
    vocabulary_definitions as vocabulary_source,
)

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


def _stub_decisions(weights_for, record=None):
    """A ``decide_batch`` stand-in returning ``weights_for``'s distribution.

    ``weights_for(criteria, state)`` returns raw per-option weights. The stub
    shapes them exactly as the endpoint would: a chosen option that is the
    argmax of the normalised distribution, plus the full distribution. When
    ``record`` is given, each call appends ``(slot, criteria)`` so a test can
    assert exactly which options a slot was offered.
    """

    async def fake_decide_batch(states, questions, **kwargs):
        criteria = questions[QUESTION]["criteria"]
        results = []
        for state in states:
            if record is not None:
                record.append((state["slot"], dict(criteria)))
            weights = weights_for(criteria, state)
            choice, probs = _distribution(criteria, weights)
            results.append(
                ({QUESTION: {"choice": choice, "probabilities": probs}}, 0.0005)
            )
        return results, 0.0005 * len(results)

    return fake_decide_batch


def _choose_by_slot(choice_by_slot: Mapping[str, str]):
    """A weights function picking ``choice_by_slot[slot]`` where offered."""

    def weights_for(criteria, state):
        slot = state["slot"]
        want = choice_by_slot.get(slot, NONE)
        if want not in criteria:
            want = NONE if NONE in criteria else next(iter(criteria))
        return {key: (1.0 if key == want else 0.001) for key in criteria}

    return weights_for


def _vector_for_text(text: str) -> list[float]:
    """A deterministic stand-in embedding, keyed on the token's own words.

    A retrieval text is ``"<token words>: <meaning>"``, so the head before the
    colon identifies the token: a query carrying the word ``density`` scores
    highest against the ``density`` token and next against tokens whose words
    contain it.
    """
    head = text.split(":")[0].strip().lower()
    if head == "density":
        return [1.0, 0.0]
    if "density" in head:
        return [0.9, 0.1]
    return [0.0, 1.0]


def _stub_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ss, "_embed", lambda texts: [_vector_for_text(text) for text in list(texts)]
    )


@pytest.fixture(autouse=True)
def _mutation(monkeypatch: pytest.MonkeyPatch):
    """Apply the declared mutation, when one is named, for the negative control.

    The tree and admissibility caches are cleared on both sides of every test: a
    tree built before a mutation would otherwise be reused inside it, and the
    caches are process-wide.
    """
    ss._build_tree.cache_clear()
    ss._admissible_cached.cache_clear()
    mutation = os.environ.get("SS_SEGMENT_MUTATION", "")

    if mutation == "base-offers-none":
        original = ss._build_tree
        monkeypatch.setattr(
            ss,
            "_build_tree",
            lambda slot, allowed, include_none: original(slot, allowed, True),
        )
    elif mutation == "bare-option-text":
        monkeypatch.setattr(ss, "option_text", lambda token: token)
    elif mutation == "no-shortlist":

        def whole_vocabulary(sources, *, size=ss.SHORTLIST_SIZE):
            tokens = list(ss._slot_tokens(ss.SLOT_PHYSICAL_BASE))
            return [tokens for _ in sources]

        monkeypatch.setattr(ss, "base_shortlists", whole_vocabulary)
    elif mutation == "no-grammar-filter":
        monkeypatch.setattr(
            ss, "admissible", lambda slot, chosen: tuple(ss._slot_tokens(slot))
        )
    elif mutation == "no-optional-none":
        original = ss._build_tree
        monkeypatch.setattr(
            ss,
            "_build_tree",
            lambda slot, allowed, include_none: original(slot, allowed, False),
        )
    elif mutation == "no-beam-prune":
        monkeypatch.setattr(ss, "BEAM", 10_000)
    elif mutation == "no-grammar-refusal":
        from imas_standard_names.grammar.render import compose

        monkeypatch.setattr(
            ss, "render_name", lambda chosen: compose(ss._build_ir(chosen))
        )

    yield
    monkeypatch.undo()
    ss._build_tree.cache_clear()
    ss._admissible_cached.cache_clear()


def _run(sources, **kwargs: Any) -> ss.SelectionRun:
    return asyncio.run(ss.select_segments(list(sources), model="test-model", **kwargs))


# ---------------------------------------------------------------------------
# The base is mandatory
# ---------------------------------------------------------------------------


def test_base_choice_offers_no_none(monkeypatch: pytest.MonkeyPatch):
    """The base slot's Choice offers no ``none``: every name carries a base.

    The base tree is built here the way the sequence builds it — over a
    shortlist and without ``none`` — and every offered token must be a
    vocabulary token. With the mechanism removed the base tree grows a ``none``
    option and the assertion fails.
    """
    shortlist = sorted(ss._slot_tokens(ss.SLOT_PHYSICAL_BASE))[:3]
    tree = ss.build_slot_tree(
        ss.SLOT_PHYSICAL_BASE, allowed=shortlist, include_none=False
    )
    offered = [child.key for child in tree.children]
    assert NONE not in offered
    assert set(offered) <= set(ss._slot_tokens(ss.SLOT_PHYSICAL_BASE))

    _stub_embedding(monkeypatch)
    record: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        ss,
        "decide_batch",
        _stub_decisions(_choose_by_slot({"physical_base": "density"}), record),
    )
    run = _run([_source()], shortlist_size=5)
    base_offers = [criteria for slot, criteria in record if slot == ss.SLOT_PHYSICAL_BASE]
    assert base_offers
    assert all(NONE not in criteria for criteria in base_offers)
    assert run.items[0].chosen()[ss.SLOT_PHYSICAL_BASE] == "density"


# ---------------------------------------------------------------------------
# Every option shows its meaning
# ---------------------------------------------------------------------------


def test_every_offered_option_carries_its_meaning():
    """An option's text is more than its token wherever ISN carries meaning.

    The locus registry defines every locus it lists, so every locus leaf's text
    must carry that definition rather than repeat the token. With the mechanism
    removed each option collapses to its bare token and the assertion fails.
    """
    text = ss.option_text("strike_point")
    assert text != "strike_point"
    assert "strike_point" in text
    definition = ss._vocabulary_definitions()["strike_point"]
    assert definition.split()[0] in text

    tree = ss.build_slot_tree(ss.SLOT_LOCUS)
    leaves = []
    for child in tree.children:
        if child.children:
            leaves.extend(child.children)
        else:
            leaves.append(child)
    for leaf in leaves:
        if leaf.key == NONE:
            continue
        assert leaf.description != leaf.key, f"{leaf.key} was offered unexplained"
        definition = vocabulary_source.vocabulary_definitions(include_loci=True).get(
            leaf.key
        )
        if definition:
            assert leaf.description == f"{leaf.key} — {definition}"


def test_definition_coverage_reports_the_slots_that_lack_definitions():
    """The coverage report counts offered tokens and those without a definition."""
    coverage = ss.definition_coverage()
    assert set(coverage) == set(ss.SLOT_ORDER)
    for slot, counts in coverage.items():
        assert counts["offered"] == len(ss._slot_tokens(slot))
        assert (
            counts["with_definition"] + counts["without_definition"]
            == counts["offered"]
        )
    # The locus registry defines every locus it lists; the base vocabulary does
    # not, and the report says so rather than assuming coverage.
    assert coverage[ss.SLOT_LOCUS]["without_definition"] == 0
    assert coverage[ss.SLOT_PHYSICAL_BASE]["without_definition"] > 0


# ---------------------------------------------------------------------------
# The base comes from a shortlist
# ---------------------------------------------------------------------------


def test_base_shortlist_ranks_by_similarity(monkeypatch: pytest.MonkeyPatch):
    """The shortlist is ordered by similarity of the source to a token's meaning."""
    _stub_embedding(monkeypatch)
    shortlists = ss.base_shortlists([_source()], size=5)
    assert shortlists[0][0] == "density"
    assert len(shortlists[0]) == 5
    assert len(set(shortlists[0])) == 5
    # A token whose words do not carry the query's word ranks below every token
    # that does, so the shortlist is the query's vocabulary and not the enum.
    unrelated = _source("/dd/void", description="a void measurement", documentation="")
    assert "density" not in ss.base_shortlists([unrelated], size=5)[0]


def test_base_is_asked_over_the_shortlist_and_membership_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
):
    """The base Choice offers the shortlist only; the item records membership.

    The offered criteria must be no wider than :data:`SHORTLIST_SIZE`, and the
    item must say whether its expected base survived retrieval — which is how a
    retrieval miss is told apart from a judgement miss. With the shortlist
    mechanism removed the base Choice is offered the whole vocabulary and the
    width assertion fails.
    """
    _stub_embedding(monkeypatch)
    record: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        ss,
        "decide_batch",
        _stub_decisions(_choose_by_slot({"physical_base": "density"}), record),
    )

    hit = _source(expected_base="density")
    miss = _source("/dd/ion/temperature", expected_base="magnetic_field")
    run = _run([hit, miss], shortlist_size=5)

    items = run.items
    assert items[0].base_in_shortlist is True
    assert items[1].base_in_shortlist is False
    assert items[0].shortlist == items[1].shortlist
    assert len(items[0].shortlist) == 5

    base_offers = [c for slot, c in record if slot == ss.SLOT_PHYSICAL_BASE]
    assert base_offers
    for criteria in base_offers:
        assert len(criteria) <= ss.SHORTLIST_SIZE
        assert set(criteria) == set(items[0].shortlist)


def test_an_unknown_expected_base_is_not_taken_as_a_miss(
    monkeypatch: pytest.MonkeyPatch,
):
    """An item that declares no expected base records no membership verdict."""
    _stub_embedding(monkeypatch)
    monkeypatch.setattr(
        ss,
        "decide_batch",
        _stub_decisions(_choose_by_slot({"physical_base": "density"})),
    )
    run = _run([_source()], shortlist_size=5)
    assert run.items[0].base_in_shortlist is None


# ---------------------------------------------------------------------------
# Slots in sequence, grammar-valid only
# ---------------------------------------------------------------------------


def test_later_slot_offers_only_tokens_that_compose(
    monkeypatch: pytest.MonkeyPatch,
):
    """Every token a later slot offers composes with the tokens already chosen.

    The base is chosen first, then the subject Choice is offered only its
    composing tokens, and the item renders. With the grammar filter removed the
    subject slot is offered tokens that do not compose with the chosen base and
    the assertion fails.
    """
    _stub_embedding(monkeypatch)
    record: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        ss,
        "decide_batch",
        _stub_decisions(
            _choose_by_slot({"physical_base": "density", "subject": "electron"}),
            record,
        ),
    )
    run = _run([_source()], shortlist_size=20)

    item = run.items[0]
    assert item.failure is None
    assert item.name == "electron_density"

    chosen = {ss.SLOT_PHYSICAL_BASE: "density"}
    allowed = set(ss.admissible(ss.SLOT_SUBJECT, chosen))
    subject_offers = [c for slot, c in record if slot == ss.SLOT_SUBJECT]
    assert subject_offers
    for criteria in subject_offers:
        offered = set(criteria) - {NONE}
        assert offered
        assert offered <= allowed
        assert NONE in criteria
    # The whole vocabulary is wider than the composing set, so the filter is
    # doing work rather than admitting everything.
    assert allowed < set(ss._slot_tokens(ss.SLOT_SUBJECT))


def test_optional_slot_can_be_left_empty(monkeypatch: pytest.MonkeyPatch):
    """An optional slot offers ``none`` and a source may leave it empty.

    With the ``none`` option removed the subject slot can only be answered with a
    token, and the assertion that its Choice offers ``none`` fails.
    """
    _stub_embedding(monkeypatch)
    record: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        ss,
        "decide_batch",
        _stub_decisions(_choose_by_slot({"physical_base": "density"}), record),
    )
    run = _run([_source()], shortlist_size=20)

    optional = (
        ss.SLOT_SUBJECT,
        ss.SLOT_COMPONENT,
        ss.SLOT_LOCUS,
        ss.SLOT_PROCESS,
        ss.SLOT_OPERATOR,
    )
    for slot in optional:
        offers = [c for answered, c in record if answered == slot]
        assert offers, f"{slot} was never asked"
        for criteria in offers:
            assert NONE in criteria, f"{slot} offered no way to stay empty"

    item = run.items[0]
    assert item.failure is None
    # 'density' alone is a name, so every optional slot was left empty.
    assert item.name == "density"
    assert item.chosen()[ss.SLOT_SUBJECT] is None


# ---------------------------------------------------------------------------
# Rendering, and refusal
# ---------------------------------------------------------------------------


def test_grammar_refusal_is_recorded_and_not_repaired(
    monkeypatch: pytest.MonkeyPatch,
):
    """A name the grammar refuses is a failure on the item, never a fixed string.

    'pressure' is a registered physical base that the grammar itself refuses
    unless qualified ('generic base requires qualification'), so a source the
    model slots it into bare is refused by ISN, not repaired here. The base
    shortlist is stubbed to offer it, since a genuinely refused token is exactly
    what the shortlist must not be trusted to filter.
    """
    monkeypatch.setattr(
        ss,
        "base_shortlists",
        lambda sources, *, size=ss.SHORTLIST_SIZE: [["pressure"] for _ in sources],
    )
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
    assert item.chosen()[ss.SLOT_PHYSICAL_BASE] == "pressure"


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
    are chosen so the beam's level-1 pruning decides the ranking: the
    ``geometry`` partition carries the single most-confident token (weight 1.0)
    but only 0.04 of the mass, so the beam drops it before its cheap token can
    win, and the third best path is region's. With pruning removed the geometry
    leaf (0.04) outranks region's (0.018) and the ranking changes.
    """
    asked_partitions: list[str] = []

    def weights_for(criteria, state):
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
    # Geometry was pruned at level 1, so its partition was never asked about.
    assert "geometry" not in asked_partitions


def test_a_failed_expansion_narrows_the_beam_and_terminates(
    monkeypatch: pytest.MonkeyPatch,
):
    """Every call reported as a failure costs one pass, never a re-ask.

    ``judgment.py`` reports a call that did not produce an answer as ``None``.
    The walk must treat that as a node contributing no child: a source that has
    no leaf yet then holds an empty beam, so it yields no path for the slot and
    the caller records the missing slot as a failure. A walk that re-queued the
    interior node would ask the same Choice again and again; the counter below
    trips on that rather than letting the call hang.
    """
    calls = {"n": 0}

    async def always_fails(states, questions, **kwargs):
        calls["n"] += 1
        if calls["n"] > 20:  # pragma: no cover - the defect's signature
            raise AssertionError("the walk re-asked a Choice without narrowing")
        return [None for _ in states], 0.0

    monkeypatch.setattr(ss, "decide_batch", always_fails)

    for slot in (ss.SLOT_PHYSICAL_BASE, ss.SLOT_LOCUS):
        tree = ss.build_slot_tree(slot)
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
        assert ranked[0] == [], f"{slot} kept a path after every expansion failed"
    # One pass per interior level: the root of each slot, and nothing re-asked.
    assert calls["n"] <= 4


def test_a_slot_with_no_surviving_path_is_that_items_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """An empty beam is that source's failure, and only that source's.

    One slot's calls all fail for one source, so :func:`_search_slot` yields no
    path for it and the slot is handed to the render missing. A name with no
    base cannot render, so :func:`select_segments` records the refusal on that
    item and renders no name for it. A sibling source in the same batch, whose
    every call answered, is selected and rendered as usual — the failure is the
    item's, not the run's.
    """
    empty = _source("/dd/electron/density")
    filled = _source("/dd/ion/temperature")
    chosen = {"physical_base": "density", "subject": "electron"}

    async def fail_one_source_per_slot(states, questions, **kwargs):
        criteria = questions[QUESTION]["criteria"]
        results = []
        for state in states:
            if (
                state["slot"] == ss.SLOT_PHYSICAL_BASE
                and state["source"]["path"] == empty["path"]
            ):
                results.append(None)
                continue
            want = chosen.get(state["slot"], NONE)
            if want not in criteria:
                want = NONE if NONE in criteria else next(iter(criteria))
            _, probs = _distribution(
                criteria, {key: (1.0 if key == want else 0.001) for key in criteria}
            )
            results.append(({QUESTION: {"choice": want, "probabilities": probs}}, 0.0))
        return results, 0.0

    monkeypatch.setattr(ss, "decide_batch", fail_one_source_per_slot)
    _stub_embedding(monkeypatch)
    run = _run([empty, filled], shortlist_size=20)

    failed_item, rendered_item = run.items
    assert ss.SLOT_PHYSICAL_BASE not in failed_item.slots
    assert failed_item.name is None
    assert failed_item.failure is not None
    assert ss.SLOT_PHYSICAL_BASE in failed_item.failure
    assert rendered_item.failure is None
    assert rendered_item.name == "electron_density"

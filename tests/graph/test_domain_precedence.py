"""One domain precedence, read by both the selector and the classifier tree.

An overlap between two physics domains is settled by a single declared order in
``imas_codex/definitions/physics/domains.yaml``. These tests hold the two
readers to that one order — the standard-name selector and the rendered
classifier questions — and hold the guidance loader to refusing an order that
misses or repeats a domain of ISN's vocabulary. Each test names the mechanism
it guards, so removing that mechanism turns the test red.
"""

from __future__ import annotations

import pytest

from imas_codex.definitions.physics import (
    load_domain_precedence,
    load_isn_physics_domains,
)
from imas_codex.graph.dd_domain_classifier import (
    CATEGORY_QUESTION,
    DOMAIN_QUESTION,
    PRECEDENCE_TIEBREAK_RULE,
    _category_precedence_rank,
    _precedence_rank,
    build_category_question,
    build_domain_question,
)
from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.domain_ranking import select_primary_domain

#: The phrase both the rendered questions and the chat prompt carry, so a reader
#: sent to the prompt can find the tie-break by the same words.
TIEBREAK_PHRASE = "earliest in the domain precedence"


def test_precedence_is_a_permutation_of_the_isn_vocabulary() -> None:
    """The declared order names every ISN domain exactly once.

    A domain left out is one a classifier could not place by precedence; a
    domain named twice would make the rank ambiguous.
    """
    vocabulary = load_isn_enum_domains()
    order = load_domain_precedence()
    assert vocabulary, "ISN vocabulary read empty — the check cannot see its subject"
    assert sorted(order) == sorted(vocabulary)
    assert len(order) == len(set(order))


def test_loader_refuses_an_order_missing_a_domain(monkeypatch) -> None:
    """A precedence missing a domain of the vocabulary is refused, not accepted."""
    import imas_codex.definitions.physics as physics

    complete = physics._declared_precedence()
    monkeypatch.setattr(physics, "_declared_precedence", lambda: complete[:-1])
    physics.load_domain_precedence.cache_clear()
    try:
        with pytest.raises(ValueError, match="missing"):
            physics.load_domain_precedence()
    finally:
        physics.load_domain_precedence.cache_clear()


def test_loader_refuses_an_order_repeating_a_domain(monkeypatch) -> None:
    """A precedence that names one domain twice is refused, not accepted."""
    import imas_codex.definitions.physics as physics

    complete = physics._declared_precedence()
    monkeypatch.setattr(
        physics, "_declared_precedence", lambda: [*complete, complete[0]]
    )
    physics.load_domain_precedence.cache_clear()
    try:
        with pytest.raises(ValueError, match="repeated"):
            physics.load_domain_precedence()
    finally:
        physics.load_domain_precedence.cache_clear()


def _category_names() -> list[str]:
    """Every category the first-level question offers, from the guidance order."""
    question = build_category_question()
    return list(question[CATEGORY_QUESTION]["criteria"])


def test_selector_and_domain_question_read_the_same_order() -> None:
    """The named selector and the rendered domain question agree on every overlap.

    Each category's question lists its domains earliest-first, and the selector
    must return exactly that first domain for the same set. If either reader
    held its own copy of the order, the two would disagree here.
    """
    rank = {domain: i for i, domain in enumerate(load_domain_precedence())}
    checked = 0
    for category in _category_names():
        options = list(build_domain_question(category)[DOMAIN_QUESTION]["criteria"])
        assert options, f"category {category!r} offered no domains"
        assert options == sorted(options, key=rank.__getitem__), (
            f"{category}: domain question options are not in precedence order"
        )
        assert select_primary_domain(options) == options[0], (
            f"{category}: the selector disagrees with the question's first option"
        )
        checked += 1
    assert checked, "no categories reached the question"


def test_category_question_states_the_rule_in_precedence_order() -> None:
    """The first-level question carries the tie-break and orders its options by it."""
    question = build_category_question()[CATEGORY_QUESTION]
    assert PRECEDENCE_TIEBREAK_RULE in question["instructions"]
    assert TIEBREAK_PHRASE in question["instructions"]

    rank = _category_precedence_rank(_precedence_rank())
    options = list(question["criteria"])
    assert options == sorted(options, key=lambda c: (rank.get(c, len(rank)), c))


def test_domain_question_states_the_tiebreak_rule() -> None:
    """The second-level question states the rule a reader applies to an overlap."""
    instructions = build_domain_question("core_plasma")[DOMAIN_QUESTION]["instructions"]
    assert PRECEDENCE_TIEBREAK_RULE in instructions
    assert TIEBREAK_PHRASE in instructions


def test_chat_prompt_states_the_rule_over_the_precedence_order() -> None:
    """The chat classifier's prompt carries the same rule and the same order.

    The prompt is rendered with the declared precedence, so its domain list and
    the tree's questions present one order to the two classifiers.
    """
    order = load_domain_precedence()
    prompt = render_prompt(
        "imas/domain_classifier", {"paths": [], "domain_precedence": order}
    )
    assert TIEBREAK_PHRASE in prompt, (
        "the chat prompt does not state the tie-break rule"
    )

    domains_section = prompt.split("## Physics Domains", 1)[1]
    positions = [domains_section.index(f"- `{domain}` — ") for domain in order]
    assert positions == sorted(positions), (
        "the prompt's domain list is not in precedence order"
    )


def load_isn_enum_domains() -> list[str]:
    """ISN's physics-domain values, the vocabulary the order must match."""
    return list(load_isn_physics_domains())

"""The name-review prompt must not hand-write operator placement rules.

Operator placement is a rendering rule owned by the ISN composer: an operator
whose operand carries an ``_of_`` / ``_at_`` / ``_due_to_`` locus tail renders
between the base group and that tail, and a name with no tail keeps its natural
prefix reading. Prose in a review prompt that states a fixed prefix-``_of_`` /
postfix-append convention is a second source of truth for that rule, and it
diverges the moment the renderer changes -- reviewers then propose the spelling
the composer will not produce.

The round-trip criterion (``parse(name) -> compose() == name``) already checks
placement against the live renderer, so it must survive.
"""

from pathlib import Path

import pytest

PROMPT_DIR = (
    Path(__file__).resolve().parents[2] / "iter_codex" / "llm" / "prompts" / "sn"
)
REVIEW_NAMES_SYSTEM = PROMPT_DIR / "review_names_system.md"


def _grammar_section(text: str) -> str:
    start = text.index("### 1. Grammar Correctness")
    return text[start : text.index("### 2.", start)]


@pytest.fixture(scope="module")
def grammar_section() -> str:
    return _grammar_section(REVIEW_NAMES_SYSTEM.read_text())


def test_round_trip_criterion_survives(grammar_section: str) -> None:
    """The criterion that delegates placement to the composer must remain."""
    assert "parse(name)" in grammar_section
    assert "compose() == name" in grammar_section


def test_no_hand_written_operator_placement_rule(grammar_section: str) -> None:
    """No prose restating where a prefix or postfix operator renders."""
    banned = [
        "prefix operators written with explicit",
        "correctly appended (not prefix",
        "_of_ scope marker",
        "`_of_` scope marker",
    ]
    found = [phrase for phrase in banned if phrase in grammar_section]
    assert not found, (
        "review_names_system.md restates operator placement, which the ISN "
        f"renderer owns: {found}. Delete the prose, or inject the rule from "
        "imas_standard_names at render time."
    )


def test_operator_placement_is_not_stated_elsewhere_in_the_prompt() -> None:
    """The rule must not reappear outside the grammar-correctness section."""
    text = REVIEW_NAMES_SYSTEM.read_text()
    section = _grammar_section(text)
    rest = text.replace(section, "")
    for phrase in ("scope marker", "not prefix `_of_` form"):
        assert phrase not in rest, (
            f"operator placement prose reappeared outside the grammar section: {phrase!r}"
        )

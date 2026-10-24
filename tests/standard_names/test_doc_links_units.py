"""The prose-bracket check takes its unit verdict from the unit owner."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.doc_links import (
    _is_prose_bracket,
    find_prose_bracket_phrases,
)


@pytest.mark.parametrize("unit", ["S", "ohm", "kohm", "ms"])
def test_unit_only_the_unit_owner_knows_is_notation(unit: str) -> None:
    assert not _is_prose_bracket(unit)


@pytest.mark.parametrize("content", ["m^-3", "K", "10^20 m^-3", "m/s"])
def test_unit_notation_verdicts_are_unchanged(content: str) -> None:
    assert not _is_prose_bracket(content)


@pytest.mark.parametrize("content", ["0, 1", "1", "-2.5"])
def test_brackets_without_a_letter_are_notation(content: str) -> None:
    assert not _is_prose_bracket(content)


@pytest.mark.parametrize(
    "content", ["banana", "total plasma current", "the core boundary"]
)
def test_word_the_unit_owner_rejects_is_prose(content: str) -> None:
    assert _is_prose_bracket(content)


def test_bracket_phrases_retain_only_what_the_unit_owner_rejects() -> None:
    text = "density [10^20 m^-3], conductivity [S], and [banana] at [0, 1]"

    assert find_prose_bracket_phrases(text) == ("banana",)

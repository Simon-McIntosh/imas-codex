"""The name-versus-definition question: one list of classes, one judge.

The check that a name and its definition state the same quantity is asked
once, by the review seat, with the blind readers' class sheet. These tests hold
that the classes exist in one place, that the prompt and the review taxonomy
read that one place, and that no second implementation decides the answer
beside the seat. They also hold what a verdict does: a flagged name is held for
review, a passing name is not, and a seat that did not answer holds the
candidate rather than accepting it.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parents[2] / "imas_codex"
SYSTEM_PROMPT = PACKAGE / "llm" / "prompts" / "sn" / "review_name_definition_system.md"
CHECK_ID = "definition_qualifier_absent_from_name"


def _classes() -> tuple[str, ...]:
    from imas_codex.standard_names.review.name_definition_seat import QUALIFIER_KINDS

    return QUALIFIER_KINDS


def _package_files(suffix: str) -> Iterator[Path]:
    for path in sorted(PACKAGE.rglob("*" + suffix)):
        if "__pycache__" in path.parts:
            continue
        yield path


def _class_sequences_in_python() -> list[tuple[str, int, list[str]]]:
    """Every literal sequence of three or more class names in the package."""

    classes = set(_classes())
    found: list[tuple[str, int, list[str]]] = []
    for path in _package_files(".py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                continue
            if not all(
                isinstance(element, ast.Constant) and isinstance(element.value, str)
                for element in node.elts
            ):
                continue
            strings = [str(element.value) for element in node.elts]
            if len(classes & set(strings)) >= 3:
                found.append(
                    (str(path.relative_to(PACKAGE.parent)), node.lineno, strings)
                )
    return found


def _yaml_lists(node: object) -> Iterator[list[str]]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _yaml_lists(value)
    elif isinstance(node, list):
        if all(isinstance(element, str) for element in node):
            yield list(node)
        else:
            for element in node:
                yield from _yaml_lists(element)


def _class_sequences_in_yaml() -> list[tuple[str, list[str]]]:
    """Every list of three or more class names in the package's data files."""

    classes = set(_classes())
    found: list[tuple[str, list[str]]] = []
    for path in _package_files(".yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for values in _yaml_lists(data):
            if len(classes & set(values)) >= 3:
                found.append((str(path.relative_to(PACKAGE.parent)), values))
    return found


def test_the_answer_classes_are_written_once() -> None:
    """A second list is a second answer to the question, and fails here."""

    sequences = _class_sequences_in_python()
    assert len(sequences) == 1, (
        "the classes the question answers with appear in more than one place, "
        f"so the copies can disagree: {sequences}"
    )
    path, _line, values = sequences[0]
    assert path == "imas_codex/standard_names/review/name_definition_seat.py", path
    assert values == list(_classes())
    assert _class_sequences_in_yaml() == []


def test_the_prompt_names_the_classes_in_order() -> None:
    text = SYSTEM_PROMPT.read_text(encoding="utf-8")
    headings = tuple(re.findall(r"^- \*\*([a-z][a-z-]*)\*\*", text, re.MULTILINE))
    assert headings == _classes()


def test_the_review_question_offers_the_same_classes() -> None:
    """The question's options are checked against a source it does not derive from.

    ``load_review_checks`` injects the seat's own tuple as the option list, so
    comparing the options with ``_classes()`` would pass on any content. The
    assertion is made against the prompt's headings below instead: a class
    renamed in the seat and not in the prompt reddens here as well as in
    :func:`test_the_prompt_names_the_classes_in_order`.
    """
    from imas_codex.standard_names.review.checks import load_review_checks

    checks, _ = load_review_checks()
    [check] = [entry for entry in checks if entry.id == CHECK_ID]
    question = check.question
    assert question is not None, "the review taxonomy no longer asks the question"
    headings = tuple(
        re.findall(
            r"^- \*\*([a-z][a-z-]*)\*\*",
            SYSTEM_PROMPT.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    assert tuple(question.options) == headings
    assert question.conforming == ["agrees"]


def test_a_taxonomy_written_class_list_is_refused() -> None:
    """A second list of classes in the taxonomy is refused, not preferred."""
    import pytest

    from imas_codex.standard_names.review.checks import _inject_name_definition_options

    with pytest.raises(ValueError, match="carries its own options"):
        _inject_name_definition_options(
            {
                "id": CHECK_ID,
                "question": {"options": ["agrees"], "conforming": ["agrees"]},
            }
        )


def test_the_question_has_no_second_judge() -> None:
    """A second place that answers the question is caught here.

    The word-matching implementation answered the question in code beside the
    seat and disagreed with it on 412 of the 2,044 labelled names. It is gone,
    and this refuses its identifiers coming back: either the lexical function,
    its review wiring, or its review check.
    """

    package = "\n".join(
        path.read_text(encoding="utf-8") for path in _package_files(".py")
    )
    for gone in (
        "name_definition_agreement_check",
        "run_name_definition_checks",
        "qualifier_name_lacks",
    ):
        assert gone not in package, (
            f"{gone!r} is a second judge of the name-versus-definition question; "
            "the seat is the only one"
        )

    seat_modules = [
        path.name
        for path in _package_files(".py")
        if "NameDefinitionBatch" in path.read_text(encoding="utf-8")
    ]
    assert seat_modules == ["name_definition_seat.py"], (
        f"more than one module carries a name-definition verdict schema: {seat_modules}"
    )


def test_a_flagged_verdict_raises_a_hold_naming_the_class() -> None:
    from imas_codex.standard_names.review.name_definition_seat import (
        name_definition_note,
    )

    note = name_definition_note(
        {"id": "momentum"},
        {
            "name": "momentum",
            "verdict": "flag",
            "reader_class": "abstract-parent-as-quantity",
            "reasoning": "the definition lists three distinct momenta",
        },
    )
    assert note is not None
    assert "abstract-parent-as-quantity" in note
    assert "three distinct momenta" in note


def test_a_passing_verdict_raises_no_hold() -> None:
    from imas_codex.standard_names.review.name_definition_seat import (
        name_definition_note,
    )

    assert (
        name_definition_note(
            {"id": "electron_temperature"},
            {
                "name": "electron_temperature",
                "verdict": "pass",
                "reader_class": "agrees",
            },
        )
        is None
    )


def test_an_unanswered_verdict_holds_rather_than_accepting() -> None:
    """A seat call that returned nothing must not read as an acceptance."""

    from imas_codex.standard_names.review.name_definition_seat import (
        name_definition_note,
    )

    note = name_definition_note(
        {"id": "momentum"},
        {"name": "momentum", "verdict": "unanswered", "reader_class": None},
    )
    assert note is not None
    assert "did not answer" in note


def test_a_flagged_verdict_is_persisted_as_a_review_hold() -> None:
    """A flagged name is not accepted by review: the hold is what stops it."""

    from imas_codex.standard_names.review.name_definition_seat import (
        name_definition_note,
    )
    from imas_codex.standard_names.worker_tasks.worker_review_docs import (
        review_holds_from_checks,
    )

    candidate = {"id": "momentum", "documentation": "Momentum is a family."}
    note = name_definition_note(
        candidate,
        {
            "name": "momentum",
            "verdict": "flag",
            "reader_class": "abstract-parent-as-quantity",
            "reasoning": "the definition lists three distinct momenta",
        },
    )
    holds = review_holds_from_checks(
        (("name_definition", note, "name-definition-seat"),), candidate
    )
    assert [hold["raised_by"] for hold in holds] == ["name-definition-seat"]
    assert holds[0]["kind"] == "name_definition"
    assert "three distinct momenta" in holds[0]["finding"]


def test_a_document_the_seat_passes_carries_no_hold() -> None:
    from imas_codex.standard_names.review.name_definition_seat import (
        name_definition_note,
    )
    from imas_codex.standard_names.worker_tasks.worker_review_docs import (
        review_holds_from_checks,
    )

    note = name_definition_note(
        {"id": "electron_temperature"},
        {
            "name": "electron_temperature",
            "verdict": "pass",
            "reader_class": "agrees",
        },
    )
    assert (
        review_holds_from_checks(
            (("name_definition", note, "name-definition-seat"),),
            {"id": "electron_temperature"},
        )
        == []
    )

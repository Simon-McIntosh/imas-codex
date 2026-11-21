"""The name review reads a qualifier against the source's enclosing structures.

A DD leaf often states only its field label ("Value", "Carbon") while the
counting basis, weighting or unit convention the name carries is stated one or
more structures above it. These tests fix what the review is shown, what it
records, and where it refuses to read: a statement that negates the qualifier or
that names another quantity's physical base is left to the judge, and the judge's
own answer is recorded beside every settled one.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from iter_codex.standard_names.review import jev_review

CARBON = "accumulated_carbon_equivalent_electron_count_due_to_gas_injection"
LEAF = "summary/gas_injection_accumulated/carbon/value"
PARENT = "summary/gas_injection_accumulated/carbon"
STRUCTURE = "summary/gas_injection_accumulated"
STATEMENT = (
    "Accumulated injected gas since the plasma breakdown in equivalent electrons"
)
NEGATED = "Accumulated injected gas, not counted in equivalent electrons"
OTHER_QUANTITY = "The equivalent electron density of the trapped population"


def _item(name: str = CARBON, statement: str = STATEMENT) -> dict:
    return {
        "id": name,
        "kind": "scalar",
        "unit": "1",
        "description": (
            "Cumulative post-breakdown carbon inventory from gas injection, "
            "expressed as equivalent-electron counts."
        ),
        "source_paths": [LEAF],
        "dd_source_docs": [
            {
                "id": LEAF,
                "ids": [LEAF],
                "unit": "1",
                "description": "",
                "documentation": "Carbon",
                "dd_version": "4.1.0",
                "snapshot_pinned": True,
                "enclosing_structures": [
                    {"path": PARENT, "text": "Carbon"},
                    {"path": STRUCTURE, "text": statement},
                ],
            }
        ],
    }


async def _decide_at(probability: float):
    async def decide(states, questions, **_kwargs):
        answers = [
            {
                name: ({"noul": probability} if name.startswith("check:") else {})
                for name in questions
            }
            for _state in states
        ]
        return [(answer, 0.0) for answer in answers], 1.0

    return decide


def _review(item: dict, probability: float = 0.9) -> dict:
    decide = asyncio.run(_decide_at(probability))
    with patch.object(jev_review, "decide_batch", decide):
        result = asyncio.run(jev_review.review_names([item]))
    return result.items[0]


def test_qualifier_stated_only_by_an_enclosing_structure_settles_the_check() -> None:
    """The statement reaches the state and settles both source-support checks."""
    record = _review(_item())

    observed = record["state"]["neighbourhood"]["dd_enclosing_structures"]
    assert observed["observed"] is True
    assert {"path": STRUCTURE, "text": STATEMENT} in observed["value"]

    for check_id in jev_review.SOURCE_SUPPORT_CHECKS:
        check = record["checks"][check_id]
        assert check["failed"] is False
        assert check["decision"] == "accept"
        assert check["failure_probability"] == 0.0
        support = check["source_support"]
        assert {
            "token": "equivalent_electron",
            "path": STRUCTURE,
            "text": STATEMENT,
        } in support["qualifiers"]
        assert check["answer"]["noul"] == 0.0
        assert check["answer"]["source_support"] == support
        # The judge answered these checks too; its answer is kept beside the
        # settled one rather than being lost.
        assert check["judge_failure_probability"] == 0.9


def test_negated_statement_is_left_to_the_judge() -> None:
    """A structure that names the words and denies them does not support them."""
    item = _item(statement=NEGATED)
    assert jev_review.enclosing_structure_support(item) is None

    record = _review(item, probability=0.86)

    for check_id in jev_review.SOURCE_SUPPORT_CHECKS:
        check = record["checks"][check_id]
        assert check["failure_probability"] == 0.86
        assert check["failed"] is True
        assert "source_support" not in check


def test_statement_about_another_quantity_is_left_to_the_judge() -> None:
    """A statement whose qualifier modifies another quantity's base does not support it."""
    item = _item(statement=OTHER_QUANTITY)
    assert jev_review.enclosing_structure_support(item) is None

    record = _review(item, probability=0.86)

    for check_id in jev_review.SOURCE_SUPPORT_CHECKS:
        check = record["checks"][check_id]
        assert check["failure_probability"] == 0.86
        assert "source_support" not in check


def test_qualifier_no_structure_states_stays_with_the_judge() -> None:
    """A qualifier stated by neither the leaf nor a structure is not settled."""
    planted = "accumulated_total_carbon_equivalent_electron_count_due_to_gas_injection"
    item = _item(planted)
    assert jev_review.enclosing_structure_support(item) is None

    record = _review(item, probability=0.86)

    for check_id in jev_review.SOURCE_SUPPORT_CHECKS:
        check = record["checks"][check_id]
        assert check["failed"] is True
        assert check["failure_probability"] == 0.86
        assert "source_support" not in check


def test_leaf_stated_qualifiers_need_no_enclosing_statement() -> None:
    """A leaf that states the qualifier itself leaves the judge alone."""
    item = _item()
    item["dd_source_docs"][0]["documentation"] = f"Carbon. {STATEMENT}."
    assert jev_review.enclosing_structure_support(item) is None

    record = _review(item)

    for check_id in jev_review.SOURCE_SUPPORT_CHECKS:
        assert record["checks"][check_id]["failure_probability"] == 0.9

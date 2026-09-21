"""The parked-source classifier stays inside its declared disposition set.

``classify_parked_source`` is total by construction: every input returns a
member of ``CAP_PARKED_DISPOSITIONS``. That totality is what lets a census of
the parked cohort be read as a closed triage list with no residue. The property
is a convention held by the two definitions agreeing, though, not by anything
the type system or a caller enforces -- so a seventh branch, or a renamed
disposition string, would leave the classifier returning a value the frozenset
does not name and nothing would say so.

These tests exercise one representative input per branch and require the
returned value to be a declared member, which fails the moment the function and
the frozenset drift apart instead of never.

Reaching every branch is not enough. The classifier resolves a row by the FIRST
branch that matches, so a case carrying one signal says nothing about what
happens when two branches are simultaneously true: a row at the cap that also
carries a real cause is classified by whichever test runs first, and reordering
two conditions silently reclassifies live rows while every single-signal case
still passes. ``ADJACENT_PAIR_CASES`` therefore supplies both signals of an
adjacent pair at once and names the precedence it pins, so swapping the two
conditions in a pair changes the value the case requires.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names import graph_ops

# One representative input per branch of classify_parked_source, keyed by the
# disposition that branch is written to return. The ordering follows the
# classifier's own precedence: produced, then removed, then a recorded cause,
# then the node category, then silence.
BRANCH_CASES: dict[str, dict[str, Any]] = {
    "name_produced": {"produced": 1},
    "upstream_quantity_removed": {"lifecycle_status": "removed"},
    "vocabulary_gap": {
        "last_error": "no grammar term covers the concept: a vocabulary gap"
    },
    "attempt_budget_exhausted": {"last_error": "compose request timed out"},
    "compose_not_applicable": {"node_category": "geometry"},
    "cause_not_recorded": {},
}


def test_every_branch_returns_a_declared_disposition() -> None:
    """Each branch's value is a member of the frozen disposition set."""
    classify = graph_ops.classify_parked_source
    members = graph_ops.CAP_PARKED_DISPOSITIONS
    undeclared = {
        branch: disposition
        for branch, evidence in BRANCH_CASES.items()
        if (disposition := classify(dict(evidence))) not in members
    }
    assert not undeclared, (
        f"classifier returned values CAP_PARKED_DISPOSITIONS does not declare: "
        f"{undeclared}; declared set is {sorted(members)}"
    )


def test_the_representative_inputs_cover_the_whole_declared_set() -> None:
    """Every declared disposition is actually reachable, so the set has no dead member."""
    reached = {
        graph_ops.classify_parked_source(dict(evidence))
        for evidence in BRANCH_CASES.values()
    }
    assert reached == set(graph_ops.CAP_PARKED_DISPOSITIONS)


# One case per adjacent pair of branches, keyed by the precedence it pins. Each
# evidence carries the signals of BOTH branches in the pair at once, so the case
# is decided by the order of the two conditions and not by either signal alone.
# Keyed by precedence rather than by branch position: a case named for a branch
# number would have to be renamed every time a branch is inserted, which is the
# edit these cases exist to catch.
ADJACENT_PAIR_CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "a produced name outranks a removed upstream quantity": (
        "name_produced",
        {"produced": 1, "lifecycle_status": "removed"},
    ),
    "a removed upstream quantity outranks a recorded cause": (
        "upstream_quantity_removed",
        {"lifecycle_status": "removed", "last_error": "compose request timed out"},
    ),
    "a recorded cause outranks the node category": (
        "attempt_budget_exhausted",
        {"last_error": "compose request timed out", "node_category": "geometry"},
    ),
    "a recorded vocabulary gap outranks the node category": (
        "vocabulary_gap",
        {
            "last_error": "no grammar term covers the concept: a vocabulary gap",
            "node_category": "coordinate",
        },
    ),
}


@pytest.mark.parametrize(
    ("expected", "evidence"),
    ADJACENT_PAIR_CASES.values(),
    ids=list(ADJACENT_PAIR_CASES),
)
def test_the_earlier_signal_of_an_adjacent_pair_wins(
    expected: str, evidence: dict[str, Any]
) -> None:
    """Two branches are true at once; the classifier resolves the earlier one."""
    returned = graph_ops.classify_parked_source(dict(evidence))
    assert returned == expected, (
        f"combined evidence {evidence} classified as {returned!r}, "
        f"but the pinned precedence is {expected!r}"
    )

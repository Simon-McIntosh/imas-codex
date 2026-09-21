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
"""

from __future__ import annotations

from typing import Any

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

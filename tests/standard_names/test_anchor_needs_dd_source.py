"""A family's anchor must be entailed by a Data Dictionary source.

A family takes its documentation template from its anchor, so the opening the
anchor sets for its members is only checkable against source text when the
anchor has a DD source of its own. A name with none — a derived parent folded
from other evidence — would template members it holds no textual authority
over, and its rewrite could never be verified.
"""

from __future__ import annotations

from imas_codex.standard_names.families import (
    _anchor_score_candidates,
    _eligible_anchor_candidate,
    member_has_dd_source,
    select_anchor,
)


def _member(
    name: str,
    *,
    docs_stage: str = "accepted",
    reviewer_score_docs: float | None = None,
    has_dd_source: bool = True,
    description: str = "a real description",
    documentation: str = "a real documentation body",
) -> dict:
    return {
        "id": name,
        "description": description,
        "documentation": documentation,
        "docs_stage": docs_stage,
        "reviewer_score_docs": reviewer_score_docs,
        "has_dd_source": has_dd_source,
    }


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------


def test_row_carrying_no_flag_is_dd_sourced():
    # A row built by a caller's own query states nothing about its sources, so
    # it is read as DD-sourced: a selector handed such rows ranks exactly as it
    # did before the flag existed.
    assert member_has_dd_source({"id": "x"}) is True
    assert member_has_dd_source({"id": "x", "has_dd_source": True}) is True
    assert member_has_dd_source({"id": "x", "has_dd_source": False}) is False


def test_eligible_candidate_requires_a_dd_source():
    assert _eligible_anchor_candidate(_member("kept")) is True
    assert _eligible_anchor_candidate(_member("dropped", has_dd_source=False)) is False


# ---------------------------------------------------------------------------
# select_anchor
# ---------------------------------------------------------------------------


def test_heuristic_skips_a_derived_parent_ranked_first():
    # The derived parent carries the best documentation signal the heuristic
    # reads — accepted, highest reviewer_score_docs — so it ranks first. It has
    # no DD source, so it is not an anchor candidate and the family takes the
    # next member that has one.
    members = [
        _member("derived_parent", reviewer_score_docs=0.99, has_dd_source=False),
        _member("sourced_sibling", reviewer_score_docs=0.80, has_dd_source=True),
    ]

    assert select_anchor(None, None, None, members) == "sourced_sibling"


def test_heuristic_skips_a_derived_parent_in_members_holding_the_parent_rule():
    # The same derived parent is also the family's parent node. Its own row is
    # the one that decides the accepted-parent rule, so the rule does not fire
    # for it either.
    members = [
        _member("derived_parent", reviewer_score_docs=0.99, has_dd_source=False),
        _member("sourced_sibling", reviewer_score_docs=0.80, has_dd_source=True),
    ]

    anchor = select_anchor(
        "derived_parent", "accepted", "a real parent description", members
    )

    assert anchor == "sourced_sibling"


def test_family_with_no_dd_sourced_member_has_no_anchor():
    # Every member — parent and children alike — lacks a DD source, so no
    # member can set a verifiable template and the family is left without an
    # anchor for the caller to record as deferred.
    members = [
        _member("derived_parent", reviewer_score_docs=0.99, has_dd_source=False),
        _member("drafted_child", docs_stage="drafted", has_dd_source=False),
    ]

    assert select_anchor(None, None, None, members) is None
    assert (
        select_anchor("derived_parent", "accepted", "a real parent description", members)
        is None
    )


def test_unsourced_member_is_not_the_last_resort_anchor():
    # The fallback branch (no accepted member) reads the same eligibility rule:
    # a non-placeholder member with no DD source is not chosen either.
    members = [
        _member("aaa_unsourced", docs_stage="drafted", has_dd_source=False),
        _member("zzz_sourced", docs_stage="drafted", has_dd_source=True),
    ]

    assert select_anchor(None, None, None, members) == "zzz_sourced"


# ---------------------------------------------------------------------------
# The judged anchor's candidate filter
# ---------------------------------------------------------------------------


def test_jev_candidates_exclude_a_member_without_a_dd_source():
    members = [
        _member("derived_parent", has_dd_source=False),
        _member("sourced_sibling", has_dd_source=True),
    ]

    candidates = _anchor_score_candidates(None, None, None, None, members)

    assert [row["id"] for row in candidates] == ["sourced_sibling"]
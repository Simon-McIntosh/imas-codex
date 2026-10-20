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
from imas_codex.standard_names.harmonize import assemble_family


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


def test_row_carrying_no_flag_is_not_dd_sourced():
    # A row that states nothing about its sources is read as having none. The
    # reader fails closed, so a caller that drops the flag disqualifies its
    # members rather than silently restoring the unguarded behaviour.
    assert member_has_dd_source({"id": "x"}) is False
    assert member_has_dd_source({"id": "x", "has_dd_source": None}) is False
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
        select_anchor(
            "derived_parent", "accepted", "a real parent description", members
        )
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


# ---------------------------------------------------------------------------
# assemble_family, end to end on rows from a family query
# ---------------------------------------------------------------------------


class _FamilyGC:
    """Answer the four family queries ``assemble_family`` issues.

    Dispatch is on the query parameters, which identify each statement
    unambiguously: the seed lookup, the seed's parent lookup, the parent's
    members, and the parent's own row. ``pattern_rows`` is returned for the
    pattern-family fallback, keyed by ``$sn_id``.
    """

    def __init__(
        self,
        *,
        seed: str,
        parent_rows: list[dict] | None = None,
        member_rows: list[dict] | None = None,
        parent_info_rows: list[dict] | None = None,
        pattern_rows: list[dict] | None = None,
    ) -> None:
        self._seed = seed
        self._parent_rows = parent_rows or []
        self._member_rows = member_rows or []
        self._parent_info_rows = parent_info_rows or []
        self._pattern_rows = pattern_rows
        self.calls: list[str] = []

    def query(self, cypher, **params):
        self.calls.append(cypher)
        if sorted(params) == ["seed"]:
            return self._parent_rows
        if "seed" in params:
            return [{"id": self._seed}]
        if "kinds" in params:
            return self._member_rows
        if "sn_id" in params:
            assert self._pattern_rows is not None, (
                "the seed has an eligible parent, so the pattern-family "
                "fallback must not be reached"
            )
            return self._pattern_rows
        if sorted(params) == ["parent_id"]:
            return self._parent_info_rows
        raise AssertionError(f"unexpected query params: {sorted(params)}")


def _family_row(
    name: str,
    *,
    docs_stage: str = "accepted",
    reviewer_score_docs: float | None = None,
    has_dd_source: bool = True,
) -> dict:
    """A member row as the family query returns it, before ``_row_to_member``."""
    return {
        "id": name,
        "description": f"description of {name}",
        "documentation": f"documentation of {name}",
        "docs_stage": docs_stage,
        "reviewer_score_docs": reviewer_score_docs,
        "operator_kind": "projection",
        "operator": None,
        "has_dd_source": has_dd_source,
        "physical_base": "base_quantity",
    }


def test_assemble_family_pattern_route_skips_an_unsourced_derived_parent():
    # The derived parent is itself a member of the pattern family and carries
    # the strongest documentation signal the heuristic reads, so it would rank
    # first. It has no DD source. The flag is projected by the family query and
    # must survive the row rebuild that hands the rows to the selector: if it
    # is dropped, no member is eligible and the family loses its anchor.
    gc = _FamilyGC(
        seed="seed_name",
        parent_rows=[],
        pattern_rows=[
            _family_row(
                "derived_parent", reviewer_score_docs=0.99, has_dd_source=False
            ),
            _family_row("sourced_sibling", reviewer_score_docs=0.80),
        ],
    )

    family = assemble_family("seed_name", gc=gc)

    assert family is not None
    assert family["grouping"] == "pattern"
    assert family["anchor"] == "sourced_sibling"


def test_assemble_family_parent_route_skips_an_unsourced_parent():
    # The parent is not one of the members, so its source state reaches the
    # selector only through the parent's own row from the graph. An accepted
    # parent with no DD source no longer wins by rule 1.
    gc = _FamilyGC(
        seed="seed_name",
        parent_rows=[
            {
                "parent_id": "derived_parent",
                "operator_kind": "projection",
                "operator": None,
            }
        ],
        member_rows=[_family_row("sourced_child")],
        parent_info_rows=[
            {
                "docs_stage": "accepted",
                "description": "a real parent description",
                "has_dd_source": False,
            }
        ],
    )

    family = assemble_family("seed_name", gc=gc)

    assert family is not None
    assert family["grouping"] == "parent"
    assert family["parent"] == "derived_parent"
    assert family["anchor"] == "sourced_child"


def test_assemble_family_parent_route_keeps_a_sourced_parent():
    # The positive control for the rule above: a parent that does have a DD
    # source still wins its accepted-parent rule. Fails if the parent's source
    # state stops being read from the graph, since the rule would then not
    # fire for any parent at all.
    gc = _FamilyGC(
        seed="seed_name",
        parent_rows=[
            {
                "parent_id": "sourced_parent",
                "operator_kind": "projection",
                "operator": None,
            }
        ],
        member_rows=[_family_row("child", reviewer_score_docs=0.99)],
        parent_info_rows=[
            {
                "docs_stage": "accepted",
                "description": "a real parent description",
                "has_dd_source": True,
            }
        ],
    )

    family = assemble_family("seed_name", gc=gc)

    assert family is not None
    assert family["anchor"] == "sourced_parent"

"""A report link is dead only when the graph holds no such name.

A harmonize report is scoped to one family, but the names a family's
documentation links to are not confined to that family. The dead-link test
must there resolve against every name the graph holds, not against the scoped
members alone, or a valid reference to a name outside the scope counts as
19 dead links on a 17-member family.
"""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS

#: Two members of one family, linking out to a third name outside it.
_MEMBERS = [
    {
        "id": "alpha_base_at_axis",
        "physical_base": "alpha_base",
        "description": "alpha base at the magnetic axis",
        "documentation": "Alpha base measured at the magnetic axis.",
        "docs_stage": "accepted",
        "harmonized_group_signature": None,
        "links": ["gamma_quantity_outside_scope"],
    },
    {
        "id": "alpha_base_at_boundary",
        "physical_base": "alpha_base",
        "description": "alpha base at the plasma boundary",
        "documentation": "Alpha base measured at the plasma boundary.",
        "docs_stage": "accepted",
        "harmonized_group_signature": None,
        "links": [],
    },
]

_TARGET = "gamma_quantity_outside_scope"


class _ScopeGraph:
    """GraphClient stand-in: one family row plus a settable graph name set."""

    def __init__(self, known_ids: set[str]) -> None:
        self.known_ids = known_ids

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "RETURN sn.id AS id" in cypher:
            return [{"id": name_id} for name_id in self.known_ids]
        row: dict[str, Any] = dict.fromkeys(PATTERN_INVARIANT_SEGMENTS, "")
        row["physical_base"] = "alpha_base"
        row["members"] = [dict(m) for m in _MEMBERS]
        return [row]


def _dead_links(known_ids: set[str]) -> int:
    from imas_codex.standard_names.harmonize import report_family_rows

    rows = report_family_rows(_ScopeGraph(known_ids))
    assert len(rows) == 1
    return rows[0]["lint"].get("link_dead_link", 0)


def test_target_in_graph_outside_scope_is_not_dead() -> None:
    """The graph holds the target, so the scoped report reports no dead link."""
    assert _dead_links({m["id"] for m in _MEMBERS} | {_TARGET}) == 0


def test_target_absent_from_graph_is_dead() -> None:
    """With the target nowhere in the graph the link is a dead link."""
    assert _dead_links({m["id"] for m in _MEMBERS}) == 1


def test_run_link_integrity_resolves_against_the_known_set() -> None:
    """The library call takes the graph-wide set for its dead-link test."""
    from imas_codex.standard_names.review.audits import run_link_integrity

    scoped = [dict(m) for m in _MEMBERS]

    in_graph = run_link_integrity(
        scoped, known_ids={m["id"] for m in scoped} | {_TARGET}
    )
    assert not [f for f in in_graph if f.finding_type == "dead_link"]

    absent = run_link_integrity(scoped, known_ids={m["id"] for m in scoped})
    dead = [f for f in absent if f.finding_type == "dead_link"]
    assert len(dead) == 1
    assert dead[0].target == _TARGET


def test_scoped_call_without_known_ids_keeps_the_scoped_set() -> None:
    """A caller that passes no graph set falls back to the scoped ids."""
    from imas_codex.standard_names.review.audits import run_link_integrity

    findings = run_link_integrity([dict(m) for m in _MEMBERS])
    dead = [f for f in findings if f.finding_type == "dead_link"]
    assert len(dead) == 1
    assert dead[0].target == _TARGET

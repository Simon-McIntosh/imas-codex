"""The rendered focus predicate selects a named source's members on the graph.

``build_focus_predicate`` is the claim's only selection rule, and the unit
tests in ``test_signals_focus.py`` exercise it against a mock that reimplements
the selection. This graph-marked test runs the rendered predicate against the
live graph: naming a ``SignalSource`` selects exactly that source's
``MEMBER_OF`` members. The MDAC ``magPbTCNNN`` source anchors the measure — its
14 members are the expected selection.

Run with ``-m graph``; the default markers exclude it.
"""

from __future__ import annotations

import pytest

from imas_codex.discovery.signals.parallel import build_focus_predicate

pytestmark = pytest.mark.graph

FACILITY = "jt-60sa"
SOURCE_MARKER = "magPbTCNNN"
CATEGORY_MARKER = "MDAC"
EXPECTED_MEMBERS = 14


def _mdac_source_id(graph) -> str:
    rows = graph.query(
        "MATCH (sg:SignalSource) "
        "WHERE sg.facility_id = $facility "
        "AND sg.id CONTAINS $category AND sg.id CONTAINS $marker "
        "RETURN sg.id AS id",
        facility=FACILITY,
        category=CATEGORY_MARKER,
        marker=SOURCE_MARKER,
    )
    assert rows, f"the {CATEGORY_MARKER} {SOURCE_MARKER} source is absent"
    return rows[0]["id"]


def _members(graph, source_id: str) -> set[str]:
    rows = graph.query(
        "MATCH (m:FacilitySignal)-[:MEMBER_OF]->(:SignalSource {id: $source}) "
        "RETURN m.id AS id",
        source=source_id,
    )
    return {row["id"] for row in rows}


def test_rendered_predicate_selects_the_named_source_members() -> None:
    from imas_codex.graph import GraphClient

    with GraphClient() as graph:
        source_id = _mdac_source_id(graph)
        expected = _members(graph, source_id)
        predicate = build_focus_predicate("s", [source_id])
        rows = graph.query(
            "MATCH (s:FacilitySignal {facility_id: $facility}) WHERE true "
            + predicate
            + " RETURN s.id AS id",
            facility=FACILITY,
            focus_items=[source_id],
        )

    assert len(expected) == EXPECTED_MEMBERS
    assert {row["id"] for row in rows} == expected

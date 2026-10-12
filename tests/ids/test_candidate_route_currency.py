"""A candidate route stands only while the text it was judged from is current.

The candidate stage stores the source and representative descriptions each
judgment read. A source whose descriptions have since been rewritten, by
re-enrichment or member individualization, is unjudged again: the candidate
claim and its pending check select it, and the escalated choice skips its
shortlist. The unit tests pin the write and the clear; the ``-m graph`` test
drives the claim predicate against a live database with a uniquely named
fixture source, then removes every trace of it.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from imas_codex.ids.graph_ops import clear_candidates, write_candidates


def _normalised(gc) -> str:
    return " ".join(gc.query.call_args.args[0].split())


class TestJudgedTextIsWrittenWithTheRoute:
    def test_write_binds_the_judged_descriptions(self):
        gc = MagicMock()
        gc.query.return_value = [{"written": 0}]

        write_candidates(
            "jet:PF:r",
            [],
            "escalated",
            gc,
            judged_description="coil current",
            judged_rep_description="PF coil 1 current",
        )

        kwargs = gc.query.call_args.kwargs
        assert kwargs["judged_description"] == "coil current"
        assert kwargs["judged_rep_description"] == "PF coil 1 current"

    def test_judged_descriptions_follow_the_route(self):
        """Both descriptions are set exactly when the route is, else nulled."""
        gc = MagicMock()
        gc.query.return_value = [{"written": 0}]

        write_candidates(
            "jet:PF:r",
            [],
            None,
            gc,
            judged_description="a",
            judged_rep_description="b",
        )

        statement = _normalised(gc)
        assert "written = $expected AND $route IS NOT NULL AS routed" in statement
        assert (
            "sg.candidate_judged_description = CASE WHEN routed "
            "THEN $judged_description ELSE null END"
        ) in statement
        assert (
            "sg.candidate_judged_rep_description = CASE WHEN routed "
            "THEN $judged_rep_description ELSE null END"
        ) in statement

    def test_clear_nulls_the_judged_descriptions(self):
        gc = MagicMock()
        gc.query.return_value = [{"edges_removed": 0, "routes_reset": 0}]

        clear_candidates("jet", gc)

        statement = _normalised(gc)
        assert "sg.candidate_judged_description = null" in statement
        assert "sg.candidate_judged_rep_description = null" in statement


@pytest.mark.graph
def test_rewritten_description_makes_a_route_unjudged():
    """Drive the claim predicate through every change that unsettles a route."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.ids.workers import (
        claim_sources_for_candidates,
        claim_sources_for_escalated,
        has_pending_candidate_work,
        release_mapping_claim,
    )

    marker = uuid.uuid4().hex[:12]
    facility = f"pytest-currency-{marker}"
    source_id = f"{facility}:fixture:source"
    rep_id = f"{facility}:fixture:rep"

    def pending() -> bool:
        return has_pending_candidate_work(facility)

    def describe(label: str, node_id: str, text: str) -> None:
        gc.query(
            f"MATCH (n:{label} {{id: $node_id}}) SET n.description = $text",
            node_id=node_id,
            text=text,
        )

    def judge(description: str, rep_description: str, route="escalated") -> None:
        write_candidates(
            source_id,
            [],
            route,
            gc,
            judged_description=description,
            judged_rep_description=rep_description,
        )

    with GraphClient() as gc:
        gc.query(
            """
            MERGE (rep:FacilitySignal {id: $rep_id})
            SET rep.facility_id = $facility, rep.description = 'coil 1 current'
            MERGE (sg:SignalSource {id: $source_id})
            SET sg.facility_id = $facility,
                sg.group_key = 'fixture:source',
                sg.status = 'enriched',
                sg.description = 'coil current',
                sg.representative_id = $rep_id
            """,
            source_id=source_id,
            rep_id=rep_id,
            facility=facility,
        )
        try:
            assert pending(), "a source with no route is unjudged"

            judge("coil current", "coil 1 current")
            assert not pending(), "a route judged from the current text stands"
            assert claim_sources_for_candidates(facility) == []
            escalated = claim_sources_for_escalated(facility)
            assert [row["id"] for row in escalated] == [source_id]
            release_mapping_claim(source_id)

            describe("FacilitySignal", rep_id, "coil 1 current, individualized")
            assert pending(), "a rewritten representative unsettles the route"
            assert claim_sources_for_escalated(facility) == []
            claimed = claim_sources_for_candidates(facility)
            assert [row["id"] for row in claimed] == [source_id]
            release_mapping_claim(source_id)

            judge("coil current", "coil 1 current, individualized")
            assert not pending(), "re-judging from the new text settles it again"

            describe("SignalSource", source_id, "poloidal field coil current")
            assert pending(), "a rewritten source description unsettles the route"

            judge("poloidal field coil current", "coil 1 current, individualized")
            assert not pending()

            judge("poloidal field coil current", "", route=None)
            assert pending(), "clearing the route leaves the source unjudged"

            gc.query(
                "MATCH (sg:SignalSource {id: $source_id}) "
                "SET sg.representative_id = null",
                source_id=source_id,
            )
            judge("poloidal field coil current", "")
            assert not pending(), "a source without a representative can settle"

            gc.query(
                "MATCH (sg:SignalSource {id: $source_id}) "
                "SET sg.candidate_judged_description = null",
                source_id=source_id,
            )
            assert pending(), "a route with no recorded judged text is unjudged"
        finally:
            gc.query(
                """
                MATCH (n) WHERE n.id IN [$source_id, $rep_id]
                DETACH DELETE n
                """,
                source_id=source_id,
                rep_id=rep_id,
            )
            leftover = gc.query(
                "MATCH (n) WHERE n.id IN [$source_id, $rep_id] RETURN count(n) AS n",
                source_id=source_id,
                rep_id=rep_id,
            )
            assert leftover[0]["n"] == 0

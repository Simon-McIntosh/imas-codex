"""Persistence tests for MAPPING_CANDIDATE edges.

The unit tests drive ``write_candidates`` and ``clear_candidates`` through a
mock graph client: they pin the count check, the claim release, the edge
replacement and the returned counts. One ``-m graph`` test exercises the same
functions against a live database with a uniquely named fixture source, then
removes every trace of it.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from imas_codex.ids.graph_ops import (
    CandidateWriteError,
    clear_candidates,
    write_candidates,
)


def _gc_returning(*responses):
    gc = MagicMock()
    gc.query.side_effect = list(responses)
    return gc


def _statement(gc) -> str:
    return gc.query.call_args.args[0]


def _normalised(gc) -> str:
    return " ".join(_statement(gc).split())


class TestWriteCandidates:
    def test_returns_written_count_and_sets_route(self):
        gc = _gc_returning([{"written": 2}])
        judgments = [
            {"path": "equilibrium/time_slice/profiles_1d/psi", "rank": 1},
            {"path": "core_profiles/profiles_1d/electrons/temperature", "rank": 2},
        ]

        written = write_candidates("jet:PF:r", judgments, "escalated", gc)

        assert written == 2
        assert gc.query.call_args.kwargs["route"] == "escalated"
        assert gc.query.call_args.kwargs["source_id"] == "jet:PF:r"

    def test_rejudge_deletes_prior_edges_before_writing(self):
        """A re-judge replaces the earlier edges: every call clears first."""
        first = _gc_returning([{"written": 1}])
        second = _gc_returning([{"written": 3}])

        write_candidates("jet:PF:r", [{"path": "a/b"}], "escalated", first)
        write_candidates(
            "jet:PF:r",
            [{"path": "a/b"}, {"path": "a/c"}, {"path": "a/d"}],
            "selected",
            second,
        )

        assert "OPTIONAL MATCH (sg)-[old:MAPPING_CANDIDATE]->(:IMASNode)" in (
            _normalised(first)
        )
        assert "DELETE old" in _normalised(first)
        assert "DELETE old" in _normalised(second)
        # The replacement runs in one statement, so the delete and the write
        # share a transaction.
        assert first.query.call_count == 1

    def test_missing_imnode_raises(self):
        """A candidate whose node vanished fails loudly, not silently."""
        gc = _gc_returning([{"written": 1}])
        judgments = [{"path": "a/b"}, {"path": "a/gone"}]

        with pytest.raises(CandidateWriteError):
            write_candidates("jet:PF:r", judgments, "escalated", gc)

    def test_claim_released_on_success(self):
        gc = _gc_returning([{"written": 1}])

        write_candidates("jet:PF:r", [{"path": "a/b"}], "escalated", gc)

        statement = _normalised(gc)
        assert "sg.mapping_claimed_at = null" in statement
        assert "sg.mapping_claim_token = null" in statement

    def test_claim_released_on_failure(self):
        """The release rides in the same statement that raises, so a failed
        write still frees the source's claim for the next pass."""
        gc = _gc_returning([{"written": 0}])

        with pytest.raises(CandidateWriteError):
            write_candidates("jet:PF:r", [{"path": "a/gone"}], "escalated", gc)

        statement = _normalised(gc)
        assert "sg.mapping_claimed_at = null" in statement
        assert "sg.mapping_claim_token = null" in statement

    def test_shortfall_keeps_route_null_and_releases_claim(self):
        """A shortfall must not mark the source judged: the route is guarded so
        the next pass re-claims and re-judges it, while the claim is still
        freed. Failing to gate the route on the written count would strand the
        source as 'judged' with partial edges and no way back."""
        gc = _gc_returning([{"written": 1}])
        judgments = [{"path": "a/b"}, {"path": "a/gone"}]

        with pytest.raises(CandidateWriteError):
            write_candidates("jet:PF:r", judgments, "selected", gc)

        statement = _normalised(gc)
        assert "CASE WHEN written = $expected THEN $route ELSE null END" in statement
        assert gc.query.call_args.kwargs["expected"] == 2
        assert "sg.mapping_claimed_at = null" in statement
        assert "sg.mapping_claim_token = null" in statement

    def test_empty_judgments_sets_the_route(self):
        """A no_candidate judgment is a real decision: no edges, but the route
        still lands, so the source is not re-judged forever."""
        gc = _gc_returning([{"written": 0}])

        written = write_candidates("jet:PF:r", [], "no_candidate", gc)

        assert written == 0
        assert gc.query.call_args.kwargs["route"] == "no_candidate"
        assert gc.query.call_args.kwargs["expected"] == 0
        assert "CASE WHEN written = $expected THEN $route ELSE null END" in (
            _normalised(gc)
        )

    def test_boolean_route_marks_the_edge(self):
        """The producer emits route as a boolean; a bool passes through."""
        gc = _gc_returning([{"written": 1}])

        write_candidates("jet:PF:r", [{"path": "a/b", "route": True}], "selected", gc)

        records = gc.query.call_args.kwargs["records"]
        assert records[0]["route"] is True

    def test_route_string_is_refused(self):
        """A route *string* such as 'escalated' must not mark the candidate
        selected: only the producer's boolean is accepted."""
        gc = _gc_returning([{"written": 1}])

        with pytest.raises(CandidateWriteError):
            write_candidates(
                "jet:PF:r", [{"path": "a/b", "route": "escalated"}], "escalated", gc
            )

        assert gc.query.call_count == 0

    def test_persists_arms_on_the_edge(self):
        """Each record's arms — the retrieval routes that returned it — are
        written so a cluster sibling can be told from a retrieval hit."""
        gc = _gc_returning([{"written": 1}])

        write_candidates(
            "jet:PF:r",
            [{"path": "a/b", "arms": ["cluster"], "route": True}],
            "selected",
            gc,
        )

        records = gc.query.call_args.kwargs["records"]
        assert records[0]["arms"] == ["cluster"]
        assert "r.arms = rec.arms" in _normalised(gc)

    def test_arms_defaults_to_empty_when_absent(self):
        gc = _gc_returning([{"written": 1}])

        write_candidates("jet:PF:r", [{"path": "a/b"}], "escalated", gc)

        assert gc.query.call_args.kwargs["records"][0]["arms"] == []

    def test_arms_not_a_list_of_strings_is_refused(self):
        """A bare string or a list of non-strings must not be written as arms."""
        gc = _gc_returning([{"written": 1}])
        with pytest.raises(CandidateWriteError):
            write_candidates(
                "jet:PF:r", [{"path": "a/b", "arms": "cluster"}], "escalated", gc
            )
        assert gc.query.call_count == 0

        gc2 = _gc_returning([{"written": 1}])
        with pytest.raises(CandidateWriteError):
            write_candidates(
                "jet:PF:r", [{"path": "a/b", "arms": [1, 2]}], "escalated", gc2
            )
        assert gc2.query.call_count == 0


class TestClearCandidates:
    def test_reports_edge_and_route_counts(self):
        gc = _gc_returning([{"edges_removed": 4, "routes_reset": 3}])

        result = clear_candidates("jet", gc)

        assert result == {"edges_removed": 4, "routes_reset": 3}
        statement = _normalised(gc)
        assert "DELETE rel" in statement
        assert "SET sg.candidate_route = null" in statement
        assert gc.query.call_args.kwargs["facility"] == "jet"

    def test_zero_when_no_rows(self):
        gc = _gc_returning([])

        assert clear_candidates("jet", gc) == {"edges_removed": 0, "routes_reset": 0}

    def test_null_counts_coalesce_to_zero(self):
        gc = _gc_returning([{"edges_removed": None, "routes_reset": 0}])

        assert clear_candidates("jet", gc) == {"edges_removed": 0, "routes_reset": 0}


# =============================================================================
# Live-database test
# =============================================================================


@pytest.mark.graph
def test_write_and_clear_candidates_round_trip():
    """Write and remove a fixture source's candidates against the live graph."""
    from imas_codex.graph.client import GraphClient

    marker = uuid.uuid4().hex[:12]
    facility = f"pytest-candidates-{marker}"
    source_id = f"{facility}:fixture:source"

    with GraphClient() as gc:
        node_rows = gc.query(
            "MATCH (n:IMASNode) RETURN n.id AS id ORDER BY n.id LIMIT 2"
        )
        node_ids = [row["id"] for row in node_rows]
        assert len(node_ids) == 2, "live graph needs at least two IMASNode ids"

        gc.query(
            """
            MERGE (sg:SignalSource {id: $source_id})
            SET sg.facility_id = $facility,
                sg.group_key = 'fixture:source',
                sg.status = 'enriched'
            """,
            source_id=source_id,
            facility=facility,
        )

        try:
            # Seed a stale claim so the test can prove the write releases it.
            gc.query(
                """
                MATCH (sg:SignalSource {id: $source_id})
                SET sg.mapping_claimed_at = datetime(),
                    sg.mapping_claim_token = 'fixture-token'
                """,
                source_id=source_id,
            )

            judgments = [
                {
                    "path": node_ids[0],
                    "rank": 1,
                    "retrieval_score": 0.9,
                    "ids": "pf_active",
                    "choice_probability": 0.8,
                    "p_same_quantity": 0.95,
                    "model": "jev-1.13",
                    "judged_at": "2026-10-06T00:00:00Z",
                    "route": True,
                },
                {
                    "path": node_ids[1],
                    "rank": 2,
                    "retrieval_score": 0.7,
                    "ids": "pf_passive",
                    "choice_probability": 0.2,
                    "p_same_quantity": 0.3,
                    "model": "jev-1.13",
                    "judged_at": "2026-10-06T00:00:00Z",
                    "route": False,
                },
            ]
            assert write_candidates(source_id, judgments, "selected", gc) == 2

            written = list(
                gc.query(
                    """
                    MATCH (sg:SignalSource {id: $source_id})
                          -[r:MAPPING_CANDIDATE]->(ip:IMASNode)
                    RETURN ip.id AS path, r.rank AS rank, r.route AS route
                    ORDER BY r.rank
                    """,
                    source_id=source_id,
                )
            )
            assert {row["path"] for row in written} == set(node_ids)
            selected = [row for row in written if row["route"]]
            assert [row["path"] for row in selected] == [node_ids[0]]

            claimed = gc.query(
                """
                MATCH (sg:SignalSource {id: $source_id})
                RETURN sg.mapping_claimed_at AS claimed_at,
                       sg.mapping_claim_token AS token,
                       sg.candidate_route AS route
                """,
                source_id=source_id,
            )
            assert claimed[0]["claimed_at"] is None
            assert claimed[0]["token"] is None
            assert claimed[0]["route"] == "selected"

            # A shortfall must fail closed: with one judgment pointing at a
            # node that does not exist, the write persists only one edge, keeps
            # the route null so the next pass re-judges the source, and still
            # releases the claim.
            gc.query(
                """
                MATCH (sg:SignalSource {id: $source_id})
                SET sg.mapping_claimed_at = datetime(),
                    sg.mapping_claim_token = 'fixture-token'
                """,
                source_id=source_id,
            )
            with pytest.raises(CandidateWriteError):
                write_candidates(
                    source_id,
                    [
                        {"path": node_ids[0], "rank": 1},
                        {"path": f"{facility}:missing:node", "rank": 2},
                    ],
                    "selected",
                    gc,
                )
            shortfall = gc.query(
                """
                MATCH (sg:SignalSource {id: $source_id})
                OPTIONAL MATCH (sg)-[r:MAPPING_CANDIDATE]->(ip:IMASNode)
                RETURN sg.candidate_route AS route,
                       sg.mapping_claimed_at AS claimed_at,
                       sg.mapping_claim_token AS token,
                       count(r) AS edges
                """,
                source_id=source_id,
            )
            assert shortfall[0]["route"] is None
            assert shortfall[0]["claimed_at"] is None
            assert shortfall[0]["token"] is None
            assert shortfall[0]["edges"] == 1

            # An empty judgment list is a real decision: a no_candidate route
            # with no edges still lands, so the source is not re-judged forever.
            assert write_candidates(source_id, [], "no_candidate", gc) == 0
            empty = gc.query(
                """
                MATCH (sg:SignalSource {id: $source_id})
                OPTIONAL MATCH (sg)-[r:MAPPING_CANDIDATE]->(ip:IMASNode)
                RETURN sg.candidate_route AS route, count(r) AS edges
                """,
                source_id=source_id,
            )
            assert empty[0]["route"] == "no_candidate"
            assert empty[0]["edges"] == 0

            # Re-judge with a single candidate: the earlier edges are replaced.
            assert (
                write_candidates(
                    source_id, [{"path": node_ids[1], "rank": 1}], "escalated", gc
                )
                == 1
            )
            after = gc.query(
                """
                MATCH (sg:SignalSource {id: $source_id})
                      -[r:MAPPING_CANDIDATE]->(ip:IMASNode)
                RETURN collect(ip.id) AS paths
                """,
                source_id=source_id,
            )
            assert after[0]["paths"] == [node_ids[1]]

            counts = clear_candidates(facility, gc)
            assert counts == {"edges_removed": 1, "routes_reset": 1}
        finally:
            gc.query(
                "MATCH (sg:SignalSource {id: $source_id}) DETACH DELETE sg",
                source_id=source_id,
            )

        # Zero residue: no fixture source and no candidate edges remain.
        residue = gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility})
            OPTIONAL MATCH (sg)-[r:MAPPING_CANDIDATE]->(:IMASNode)
            RETURN count(DISTINCT sg) AS sources, count(r) AS edges
            """,
            facility=facility,
        )
        assert residue[0]["sources"] == 0
        assert residue[0]["edges"] == 0

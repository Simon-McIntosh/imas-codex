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
    read_candidates,
    select_candidates,
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
            {"path": "equilibrium/time_slice/profiles_1d/psi", "rank": 1, "arms": []},
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "rank": 2,
                "arms": [],
            },
        ]

        written = write_candidates("jet:PF:r", judgments, "escalated", gc)

        assert written == 2
        assert gc.query.call_args.kwargs["route"] == "escalated"
        assert gc.query.call_args.kwargs["source_id"] == "jet:PF:r"

    def test_rejudge_deletes_prior_edges_before_writing(self):
        """A re-judge replaces the earlier edges: every call clears first."""
        first = _gc_returning([{"written": 1}])
        second = _gc_returning([{"written": 3}])

        write_candidates("jet:PF:r", [{"path": "a/b", "arms": []}], "escalated", first)
        write_candidates(
            "jet:PF:r",
            [
                {"path": "a/b", "arms": []},
                {"path": "a/c", "arms": []},
                {"path": "a/d", "arms": []},
            ],
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
        judgments = [{"path": "a/b", "arms": []}, {"path": "a/gone", "arms": []}]

        with pytest.raises(CandidateWriteError):
            write_candidates("jet:PF:r", judgments, "escalated", gc)

    def test_claim_released_on_success(self):
        gc = _gc_returning([{"written": 1}])

        write_candidates("jet:PF:r", [{"path": "a/b", "arms": []}], "escalated", gc)

        statement = _normalised(gc)
        assert "sg.mapping_claimed_at = null" in statement
        assert "sg.mapping_claim_token = null" in statement

    def test_claim_released_on_failure(self):
        """The release rides in the same statement that raises, so a failed
        write still frees the source's claim for the next pass."""
        gc = _gc_returning([{"written": 0}])

        with pytest.raises(CandidateWriteError):
            write_candidates(
                "jet:PF:r", [{"path": "a/gone", "arms": []}], "escalated", gc
            )

        statement = _normalised(gc)
        assert "sg.mapping_claimed_at = null" in statement
        assert "sg.mapping_claim_token = null" in statement

    def test_shortfall_keeps_route_null_and_releases_claim(self):
        """A shortfall must not mark the source judged: the route is guarded so
        the next pass re-claims and re-judges it, while the claim is still
        freed. Failing to gate the route on the written count would strand the
        source as 'judged' with partial edges and no way back."""
        gc = _gc_returning([{"written": 1}])
        judgments = [{"path": "a/b", "arms": []}, {"path": "a/gone", "arms": []}]

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

        write_candidates(
            "jet:PF:r", [{"path": "a/b", "route": True, "arms": []}], "selected", gc
        )

        records = gc.query.call_args.kwargs["records"]
        assert records[0]["route"] is True

    def test_route_string_is_refused(self):
        """A route *string* such as 'escalated' must not mark the candidate
        selected: only the producer's boolean is accepted."""
        gc = _gc_returning([{"written": 1}])

        with pytest.raises(CandidateWriteError):
            write_candidates(
                "jet:PF:r",
                [{"path": "a/b", "route": "escalated", "arms": []}],
                "escalated",
                gc,
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

    def test_arms_absent_is_refused(self):
        """The producer always emits arms, so an absent value means a caller
        forgot provenance: it is refused rather than defaulted to empty."""
        gc = _gc_returning([{"written": 1}])

        with pytest.raises(CandidateWriteError):
            write_candidates("jet:PF:r", [{"path": "a/b"}], "escalated", gc)

        assert gc.query.call_count == 0

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


class TestReadCandidates:
    def test_groups_rows_by_source_and_keeps_section_fields(self):
        gc = _gc_returning(
            [
                {
                    "source_id": "jet:PF:r",
                    "path": "equilibrium/time_slice/profiles_1d/psi",
                    "rank": 1,
                    "section": "equilibrium/time_slice",
                    "data_type": "STRUCT_ARRAY",
                    "timebasepath": "time",
                    "ndim": 2,
                },
                {
                    "source_id": "jet:PF:z",
                    "path": "magnetics/flux_loop/flux",
                    "rank": 1,
                    "section": "magnetics/flux_loop",
                    "data_type": "STRUCTURE",
                    "timebasepath": None,
                    "ndim": 0,
                },
            ]
        )

        edges = read_candidates(["jet:PF:r", "jet:PF:z"], gc)

        assert set(edges) == {"jet:PF:r", "jet:PF:z"}
        psi = edges["jet:PF:r"][0]
        assert psi["path"] == "equilibrium/time_slice/profiles_1d/psi"
        assert psi["section"] == "equilibrium/time_slice"
        assert psi["data_type"] == "STRUCT_ARRAY"
        assert psi["timebasepath"] == "time"
        assert psi["ndim"] == 2
        assert edges["jet:PF:z"][0]["data_type"] == "STRUCTURE"

    def test_reads_in_jev_order_and_derives_the_section(self):
        gc = _gc_returning([])

        read_candidates(["jet:PF:r"], gc)

        statement = _normalised(gc)
        assert "ORDER BY sg.id, r.rank" in statement
        assert "parts[0] + '/' + parts[1] AS section_id" in statement
        assert gc.query.call_args.kwargs["source_ids"] == ["jet:PF:r"]

    def test_no_source_ids_skips_the_query(self):
        gc = _gc_returning([])

        assert read_candidates([], gc) == {}
        assert gc.query.call_count == 0


class TestSelectCandidates:
    def test_marks_listed_edges_and_sets_the_route(self):
        gc = _gc_returning([{"matched": 2}])

        marked = select_candidates(
            "jet:PF:r",
            ["equilibrium/time_slice/profiles_1d/psi", "summary/ip"],
            gc,
        )

        assert marked == 2
        statement = _normalised(gc)
        assert "FOREACH (rel IN found | SET rel.route = true)" in statement
        assert "WHEN matched = $expected THEN 'selected'" in statement
        kwargs = gc.query.call_args.kwargs
        assert kwargs["source_id"] == "jet:PF:r"
        assert kwargs["paths"] == [
            "equilibrium/time_slice/profiles_1d/psi",
            "summary/ip",
        ]
        assert kwargs["expected"] == 2

    def test_a_path_with_no_edge_is_refused(self):
        gc = _gc_returning([{"matched": 1}])

        with pytest.raises(CandidateWriteError):
            select_candidates(
                "jet:PF:r",
                ["equilibrium/time_slice/profiles_1d/psi", "summary/missing"],
                gc,
            )


class TestCandidateRecordsStrict:
    def test_a_judgment_without_a_candidate_is_refused(self):
        from imas_codex.ids.workers import _candidate_records

        candidate = MagicMock()
        candidate.hit.path = "equilibrium/time_slice/profiles_1d/psi"
        candidate.hit.score = 0.9
        candidate.hit.ids_name = "equilibrium"
        candidate.arms = frozenset({"equilibrium"})

        judgment = MagicMock()
        judgment.path = "summary/missing"
        judgment.p_same_quantity = 0.5
        judgment.model = "jev-1.13"
        judgment.judged_at = "2026-10-06T00:00:00Z"

        with pytest.raises(CandidateWriteError):
            _candidate_records([judgment], [candidate], set())

    def test_a_matched_judgment_records_the_candidate_arms(self):
        from imas_codex.ids.workers import _candidate_records

        candidate = MagicMock()
        candidate.hit.path = "equilibrium/time_slice/profiles_1d/psi"
        candidate.hit.score = 0.9
        candidate.hit.ids_name = "equilibrium"
        candidate.arms = frozenset({"equilibrium"})

        judgment = MagicMock()
        judgment.path = "equilibrium/time_slice/profiles_1d/psi"
        judgment.p_same_quantity = 0.5
        judgment.model = "jev-1.13"
        judgment.judged_at = "2026-10-06T00:00:00Z"

        records = _candidate_records(
            [judgment], [candidate], {"equilibrium/time_slice/profiles_1d/psi"}
        )

        assert records[0]["arms"] == ["equilibrium"]
        assert records[0]["route"] is True


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
                    "arms": ["pf_active"],
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
                    "arms": [],
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
                        {"path": node_ids[0], "rank": 1, "arms": []},
                        {"path": f"{facility}:missing:node", "rank": 2, "arms": []},
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
                    source_id,
                    [{"path": node_ids[1], "rank": 1, "arms": []}],
                    "escalated",
                    gc,
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


@pytest.mark.graph
def test_read_and_select_candidates_round_trip():
    """Read a source's edges in Jev order and mark a two-IDS pick."""
    from imas_codex.graph.client import GraphClient

    marker = uuid.uuid4().hex[:12]
    facility = f"pytest-read-candidates-{marker}"
    source_id = f"{facility}:fixture:source"

    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (n:IMASNode) RETURN n.id AS id, n.ids AS ids ORDER BY n.id"
        )
        by_ids: dict[str, str] = {}
        for row in rows:
            by_ids.setdefault(row["ids"], row["id"])
        ids_names = sorted(by_ids)
        assert len(ids_names) >= 2, "live graph needs two IDSs"
        first, second = by_ids[ids_names[0]], by_ids[ids_names[1]]
        third = next(
            row["id"]
            for row in rows
            if row["ids"] == ids_names[0] and row["id"] != first
        )

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
            judgments = [
                {"path": first, "rank": 1, "ids": ids_names[0], "arms": []},
                {"path": second, "rank": 2, "ids": ids_names[1], "arms": []},
                {"path": third, "rank": 3, "ids": ids_names[0], "arms": []},
            ]
            assert write_candidates(source_id, judgments, "escalated", gc) == 3

            edges = read_candidates([source_id], gc)[source_id]
            assert [e["path"] for e in edges] == [first, second, third]
            assert edges[0]["section"] == "/".join(first.split("/")[:2])
            assert edges[0]["data_type"] is not None

            # A pick spanning two IDSs marks exactly those edges.
            assert select_candidates(source_id, [first, second], gc) == 2
            after = {
                e["path"]: e["route"]
                for e in read_candidates([source_id], gc)[source_id]
            }
            assert after == {first: True, second: True, third: False}
            route = gc.query(
                "MATCH (sg:SignalSource {id: $source_id}) "
                "RETURN sg.candidate_route AS route",
                source_id=source_id,
            )
            assert route[0]["route"] == "selected"

            with pytest.raises(CandidateWriteError):
                select_candidates(source_id, [first, "no/such/candidate"], gc)
        finally:
            gc.query(
                "MATCH (sg:SignalSource {id: $source_id}) DETACH DELETE sg",
                source_id=source_id,
            )

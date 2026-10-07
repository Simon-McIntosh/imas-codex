"""Signal sources for signals that grouping leaves ungrouped.

``detect_signal_sources`` groups signals by accessor pattern and forms a shared
source only for a pattern with at least ``min_instances`` members. A signal in a
smaller pattern has no source, and every mapping stage keys on a source, so such
a signal can never be mapped. The final pass in ``detect_signal_sources`` gives
each ungrouped signal a one-member source keyed by the signal itself.

The mocked tests pin the pass's shape: which group ids, member lists and
representatives it creates, and that the fetch excludes signals already in a
source. One ``-m graph`` test drives the same pass against a live database on a
uniquely named fixture facility, then enriches a one-member source and shows the
candidate stage claims it; it removes every trace of the fixture in a ``finally``
block.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.signals.parallel import detect_signal_sources


def _mock_gc(results):
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=results)
    return gc


def _merge_calls(gc):
    return [c for c in gc.query.call_args_list if "MERGE (sg:SignalSource" in c.args[0]]


def _kwargs(call):
    return call.kwargs


class TestOneMemberSources:
    def test_ungrouped_signal_forms_one_member_source(self):
        """A lone signal becomes a one-member source with itself as representative."""
        results = [
            {"id": "tcv:a1", "accessor": "GAS_001:X"},
            {"id": "tcv:a2", "accessor": "GAS_002:X"},
            {"id": "tcv:a3", "accessor": "GAS_003:X"},
            {"id": "tcv:lone", "accessor": "LONE:Y"},
        ]
        gc = _mock_gc(results)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient", return_value=gc
        ):
            groups, members = detect_signal_sources("tcv", min_instances=3)

        assert (groups, members) == (2, 4)

        merges = _merge_calls(gc)
        by_count = {_kwargs(c)["member_count"]: _kwargs(c) for c in merges}

        shared = by_count[3]
        assert set(shared["member_ids"]) == {"tcv:a1", "tcv:a2", "tcv:a3"}
        assert shared["rep_id"] == "tcv:a1"  # first accessor alphabetically
        assert shared["group_key"] == "GAS_NNN:X"

        lone = by_count[1]
        assert lone["member_ids"] == ["tcv:lone"]
        assert lone["rep_id"] == "tcv:lone"
        assert lone["group_key"] == "LONE:Y"
        assert lone["group_id"] == "tcv:LONE:Y"

    def test_two_signal_pattern_forms_two_one_member_sources(self):
        """A sub-threshold pattern forms no shared group; each signal stands alone."""
        results = [
            {"id": "tcv:p1", "accessor": "PAIR_01:V"},
            {"id": "tcv:p2", "accessor": "PAIR_02:V"},
        ]
        gc = _mock_gc(results)

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient", return_value=gc
        ):
            groups, members = detect_signal_sources("tcv", min_instances=3)

        assert (groups, members) == (2, 2)
        group_ids = {_kwargs(c)["group_id"] for c in _merge_calls(gc)}
        assert group_ids == {"tcv:PAIR_01:V", "tcv:PAIR_02:V"}

    def test_fetch_excludes_signals_already_in_a_source(self):
        """Re-running finds nothing: the fetch skips signals already MEMBER_OF."""
        gc = _mock_gc([])

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient", return_value=gc
        ):
            assert detect_signal_sources("tcv", min_instances=3) == (0, 0)

        fetch_statement = gc.query.call_args_list[0].args[0]
        assert "NOT EXISTS { (s)-[:MEMBER_OF]->(:SignalSource) }" in fetch_statement

    def test_creates_nothing_when_no_signals(self):
        gc = _mock_gc([])

        with patch(
            "imas_codex.discovery.signals.parallel.GraphClient", return_value=gc
        ):
            assert detect_signal_sources("tcv", min_instances=3) == (0, 0)

        assert _merge_calls(gc) == []


@pytest.mark.graph
def test_singleton_sources_round_trip():
    """Live: two ungrouped signals and one three-member pattern become three
    sources; a re-run adds nothing; enriching a one-member source makes it
    claimable by the candidate stage."""
    from imas_codex.discovery.signals.parallel import propagate_source_enrichment
    from imas_codex.graph.client import GraphClient
    from imas_codex.ids.workers import claim_sources_for_candidates

    marker = uuid.uuid4().hex[:12]
    facility = f"pytest-singleton-{marker}"
    signals = {
        f"{facility}:ALPHA_ONLY": "ALPHA_ONLY",
        f"{facility}:BETA_ONLY": "BETA_ONLY",
        f"{facility}:TRIP_001:V": "TRIP_001:V",
        f"{facility}:TRIP_002:V": "TRIP_002:V",
        f"{facility}:TRIP_003:V": "TRIP_003:V",
    }

    with GraphClient() as gc:
        gc.ensure_facility(facility)
        try:
            for signal_id, accessor in signals.items():
                gc.query(
                    """
                    MERGE (s:FacilitySignal {id: $id})
                    SET s.facility_id = $facility,
                        s.accessor = $accessor,
                        s.status = 'discovered'
                    """,
                    id=signal_id,
                    facility=facility,
                    accessor=accessor,
                )

            groups, members = detect_signal_sources(facility, min_instances=3)
            assert (groups, members) == (3, 5)

            rows = gc.query(
                """
                MATCH (sg:SignalSource {facility_id: $facility})
                OPTIONAL MATCH (m:FacilitySignal)-[:MEMBER_OF]->(sg)
                RETURN sg.id AS id, sg.member_count AS member_count,
                       sg.representative_id AS rep, collect(m.id) AS members
                """,
                facility=facility,
            )
            assert len(rows) == 3
            assert sorted(r["member_count"] for r in rows) == [1, 1, 3]
            for row in rows:
                assert row["rep"] in row["members"]
                assert row["member_count"] == len(row["members"])

            # Second run finds every signal already sourced: no new source and
            # no new edge.
            assert detect_signal_sources(facility, min_instances=3) == (0, 0)
            counts = gc.query(
                """
                MATCH (sg:SignalSource {facility_id: $facility})
                OPTIONAL MATCH (:FacilitySignal)-[r:MEMBER_OF]->(sg)
                RETURN count(DISTINCT sg) AS sources, count(r) AS edges
                """,
                facility=facility,
            )
            assert counts[0]["sources"] == 3
            assert counts[0]["edges"] == 5

            # Enrich one one-member source: the source reads enriched and the
            # candidate stage claims it.
            singleton = next(r for r in rows if r["member_count"] == 1)
            propagate_source_enrichment(
                singleton["rep"],
                {
                    "physics_domain": "machine_control",
                    "description": "A lone machine-control signal",
                    "name": "Lone Signal",
                    "keywords": ["machine"],
                },
            )
            status = gc.query(
                "MATCH (sg:SignalSource {id: $id}) RETURN sg.status AS status",
                id=singleton["id"],
            )
            assert status[0]["status"] == "enriched"

            claimed = claim_sources_for_candidates(facility)
            assert singleton["id"] in {row["id"] for row in claimed}
        finally:
            gc.query(
                "MATCH (s:FacilitySignal {facility_id: $facility}) DETACH DELETE s",
                facility=facility,
            )
            gc.query(
                "MATCH (sg:SignalSource {facility_id: $facility}) DETACH DELETE sg",
                facility=facility,
            )
            gc.query(
                "MATCH (f:Facility {id: $facility}) DETACH DELETE f",
                facility=facility,
            )

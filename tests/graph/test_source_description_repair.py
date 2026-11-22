"""A hand-edited feed description is reported by the snapshot comparison.

The comparison is the instrument a repair is checked with, so it has to notice
the thing it exists to notice: a stored feed label that no longer matches what
the feed supplied. This runs against a disposable graph, edits one stored
description by hand, and requires the comparison to name exactly that signal
while the untouched one still reads equal.
"""

import pytest

from iter_codex.discovery.signals.description_snapshot import (
    compare_source_descriptions,
)
from iter_codex.graph.client import GraphClient

pytestmark = pytest.mark.graph

FEED_SOURCE = "controlsmetadata:/variable_hmi"
FACILITY = "snapshot-fixture"


def _rows(graph: GraphClient) -> list[dict]:
    return list(
        graph.query(
            "MATCH (s:FacilitySignal {facility_id: $facility}) "
            "RETURN s.id AS id, s.description AS description, "
            "s.description_source AS description_source ",
            facility=FACILITY,
        )
    )


def test_hand_edited_feed_description_is_reported(disposable_graph):
    snapshot = [
        {
            "id": f"{FACILITY}:general/codac_uda/DEMO:IP",
            "feed_description": "Fixture feed label A",
            "feed_source": FEED_SOURCE,
        },
        {
            "id": f"{FACILITY}:general/codac_uda/DEMO:BT",
            "feed_description": "Fixture feed label B",
            "feed_source": FEED_SOURCE,
        },
    ]
    with GraphClient() as graph:
        for entry in snapshot:
            graph.query(
                "CREATE (:FacilitySignal {id: $id, facility_id: $facility, "
                "description: $description, description_source: $source})",
                id=entry["id"],
                facility=FACILITY,
                description=entry["feed_description"],
                source=entry["feed_source"],
            )

        assert compare_source_descriptions(_rows(graph), snapshot)["all_equal"]

        edited = snapshot[0]["id"]
        graph.query(
            "MATCH (s:FacilitySignal {id: $id}) SET s.description = $description",
            id=edited,
            description="A hand-edited label",
        )

        report = compare_source_descriptions(_rows(graph), snapshot)

    assert report["description_mismatch"] == [edited]
    assert report["source_mismatch"] == []
    assert report["missing"] == []
    assert not report["all_equal"]

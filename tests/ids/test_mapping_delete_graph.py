"""Live-graph test for the whole-mapping delete the engine clear path runs.

The unit tests drive a mock, so a delete statement whose scoping matched
nothing — or one that dropped only the bindings and left the mapping's
escalation evidence behind — would stay green there. This test builds a
throwaway mapping with one binding and one evidence node, clears it through
the engine path, and asserts all three artefacts are gone.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.ids.workers import clear_mappings_for_ids

FACILITY = "zz-delete-owner-test"
IDS_NAME = "test_ids"
SOURCE_ID = "zz-delete-owner-test:sig1"
IMAS_NODE_ID = "test_ids/thing/data"
MAPPING_ID = f"{FACILITY}:{IDS_NAME}"
EVIDENCE_ID = "zz-delete-owner-test:sig1:test_ids/thing/data:escalation"


def _counts(gc: GraphClient) -> dict[str, int]:
    """Count the three artefacts the delete owner claims to remove."""
    rows = gc.query(
        """
        OPTIONAL MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids})
        OPTIONAL MATCH (sg:SignalSource {id: $source})
        OPTIONAL MATCH (sg)-[r:MAPS_TO_IMAS]->(:IMASNode)
        OPTIONAL MATCH (sg)-[:HAS_EVIDENCE]->(ev:MappingEvidence {id: $ev})
        RETURN count(DISTINCT m) AS mappings,
               count(DISTINCT r) AS bindings,
               count(DISTINCT ev) AS evidence
        """,
        facility=FACILITY,
        ids=IDS_NAME,
        source=SOURCE_ID,
        ev=EVIDENCE_ID,
    )[0]
    return {
        "mappings": rows["mappings"],
        "bindings": rows["bindings"],
        "evidence": rows["evidence"],
    }


def _seed(gc: GraphClient) -> None:
    gc.query(
        """
        MERGE (m:IMASMapping {id: $mid})
        SET m.facility_id = $facility, m.ids_name = $ids
        MERGE (sg:SignalSource {id: $source})
        MERGE (ip:IMASNode {id: $imas})
        MERGE (ev:MappingEvidence {id: $ev})
        SET ev.evidence_type = 'escalation'
        MERGE (sg)-[:MAPS_TO_IMAS]->(ip)
        MERGE (sg)-[:HAS_EVIDENCE]->(ev)
        MERGE (m)-[:USES_SIGNAL_SOURCE]->(sg)
        """,
        mid=MAPPING_ID,
        facility=FACILITY,
        ids=IDS_NAME,
        source=SOURCE_ID,
        imas=IMAS_NODE_ID,
        ev=EVIDENCE_ID,
    )


def _purge(gc: GraphClient) -> None:
    gc.query(
        """
        MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids})
        DETACH DELETE m
        """,
        facility=FACILITY,
        ids=IDS_NAME,
    )
    gc.query(
        "MATCH (sg:SignalSource {id: $source}) DETACH DELETE sg",
        source=SOURCE_ID,
    )
    gc.query(
        "MATCH (ev:MappingEvidence {id: $ev}) DETACH DELETE ev",
        ev=EVIDENCE_ID,
    )


@pytest.mark.graph
def test_engine_clear_removes_binding_evidence_and_mapping():
    with GraphClient() as gc:
        _purge(gc)
        _seed(gc)

        # Positive control: the seeded mapping must be visible before the
        # delete, so a clear that silently matched nothing cannot read as one
        # that removed everything.
        before = _counts(gc)
        assert before == {"mappings": 1, "bindings": 1, "evidence": 1}

        deleted = clear_mappings_for_ids(FACILITY, [IDS_NAME])

        assert deleted == {"mappings": 1, "bindings": 1, "evidence": 1}
        assert _counts(gc) == {"mappings": 0, "bindings": 0, "evidence": 0}

        _purge(gc)

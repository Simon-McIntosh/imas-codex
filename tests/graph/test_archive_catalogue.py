"""Actual catalogue statements and provenance links in a disposable graph."""

import pytest

from imas_codex.discovery.signals.scanners import codac_uda as catalogue
from imas_codex.graph.client import GraphClient

pytestmark = pytest.mark.graph


def test_archive_catalogue_statements_are_idempotent_and_keep_provenance(
    disposable_graph,
):
    facility = "demo-archive-catalogue"
    archive = f"{facility}:codac_uda:slow"
    access = f"{facility}:codac_uda"
    with GraphClient() as graph:
        graph.query(
            "CREATE (:Facility {id: $facility, name: 'Demonstration'})",
            facility=facility,
        )
        graph.query(
            "CREATE (:DataAccess {id: $access, facility_id: $facility})",
            access=access,
            facility=facility,
        )
        archive_rows = [
            {
                "id": archive,
                "facility_id": facility,
                "data_system": "codac_uda",
                "role": "slow",
                "excluded": False,
            }
        ]
        assert (
            graph.query(
                catalogue.ARCHIVE_QUERY, facility=facility, archives=archive_rows
            )[0]["written"]
            == 1
        )
        entry = {
            "name": "DEMO:TEMP",
            "unit": {"value": "K"},
            "description": {"value": "Archive wording"},
        }
        signal = catalogue.CodacUDAScanner._signal(
            facility, archive, access, entry, "cohort"
        )
        parameters = {
            "facility": facility,
            "cohort": "cohort",
            "signals": [signal.model_dump(mode="json")],
        }
        for _ in range(2):
            assert graph.query(catalogue.SIGNAL_QUERY, **parameters)[0]["written"] == 1
        from imas_codex.discovery.signals.parallel import build_signal_scope_case

        scoped = graph.query(
            "MATCH (s:FacilitySignal {id: $id}) RETURN "
            + build_signal_scope_case()
            + " AS scanner",
            id=signal.id,
            static_sources=[],
        )
        assert scoped == [{"scanner": "codac_uda"}]
        rows = graph.query(
            "MATCH (s:FacilitySignal {id: $id}) RETURN count(s) AS total, count(s.metadata_archive) AS metadata, count(s.data_archive) AS confirmed",
            id=signal.id,
        )
        assert rows == [{"total": 1, "metadata": 1, "confirmed": 0}]
        assert (
            graph.query(catalogue.CONFIRM_QUERY, signal=signal.id, archive=archive)[0][
                "written"
            ]
            == 1
        )
        graph.query(
            "MATCH (s:FacilitySignal {id: $id}) SET s.status = 'enriched', s.claim_token = 'held'",
            id=signal.id,
        )
        graph.query(catalogue.SIGNAL_QUERY, **parameters)
        rows = graph.query(
            "MATCH (s:FacilitySignal {id: $id})-[:METADATA_ARCHIVE]->(m:ArchiveEndpoint), (s)-[:DATA_ARCHIVE]->(d:ArchiveEndpoint), (s)-[:DATA_ACCESS]->(a:DataAccess) RETURN s.status AS status, s.claim_token AS claim, s.data_archive AS serving, m.id AS metadata, d.id AS data, a.id AS access, s.archive_cohorts AS cohorts",
            id=signal.id,
        )
        assert rows == [
            {
                "status": "enriched",
                "claim": "held",
                "serving": archive,
                "metadata": archive,
                "data": archive,
                "access": access,
                "cohorts": ["cohort"],
            }
        ]
        graph.query(
            "MATCH (a:ArchiveEndpoint {id: $archive}) SET a.excluded = true",
            archive=archive,
        )
        assert (
            graph.query(catalogue.CONFIRM_QUERY, signal=signal.id, archive=archive)[0][
                "written"
            ]
            == 0
        )

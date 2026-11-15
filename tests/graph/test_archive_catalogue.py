"""Actual catalogue statements and provenance links in a disposable graph."""

import pytest

from imas_codex.discovery.signals.parallel import ingest_discovered_signals
from imas_codex.discovery.signals.scanners import codac_uda as catalogue
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import ArchiveEndpoint
from imas_codex.remote import codac_uda as service

pytestmark = pytest.mark.graph


def test_archive_catalogue_statements_are_idempotent_and_keep_provenance(
    disposable_graph,
):
    facility = "demo-archive-catalogue"
    archive = f"{facility}:codac_uda:slow"
    access = catalogue._access(facility, "slow").id
    with GraphClient() as graph:
        graph.query(
            "CREATE (:Facility {id: $facility, name: 'Demonstration'})",
            facility=facility,
        )
        catalogue._persist_archives(
            facility,
            [
                ArchiveEndpoint(
                    id=f"{facility}:codac_uda:{role}",
                    facility_id=facility,
                    data_system="codac_uda",
                    role=role,
                    excluded=False,
                )
                for role in ["slow", "fast"]
            ],
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
            assert ingest_discovered_signals(parameters["signals"]) == 1
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
            graph.query(
                catalogue.CONFIRM_QUERY,
                signal=signal.id,
                archive=f"{facility}:codac_uda:fast",
            )[0]["written"]
            == 1
        )
        graph.query(
            "MATCH (s:FacilitySignal {id: $id}) SET s.status = 'enriched', s.claim_token = 'held'",
            id=signal.id,
        )
        ingest_discovered_signals(parameters["signals"])
        rows = graph.query(
            "MATCH (s:FacilitySignal {id: $id})-[:METADATA_ARCHIVE]->(m:ArchiveEndpoint), (s)-[:DATA_ARCHIVE]->(d:ArchiveEndpoint), (s)-[:DATA_ACCESS]->(a:DataAccess) RETURN s.status AS status, s.claim_token AS claim, s.data_archive AS serving, m.id AS metadata, d.id AS data, a.id AS access, s.archive_cohorts AS cohorts",
            id=signal.id,
        )
        assert rows == [
            {
                "status": "enriched",
                "claim": "held",
                "serving": f"{facility}:codac_uda:fast",
                "metadata": archive,
                "data": f"{facility}:codac_uda:fast",
                "access": catalogue._access(facility, "fast").id,
                "cohorts": ["cohort"],
            }
        ]
        binding = graph.query(service.SIGNAL_QUERY, signal_id=signal.id)
        assert len(binding) == 1
        assert binding[0]["archive_id"] == f"{facility}:codac_uda:fast"
        assert binding[0]["archive_role"] == "fast"
        assert binding[0]["method_type"] == "codac_uda"
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

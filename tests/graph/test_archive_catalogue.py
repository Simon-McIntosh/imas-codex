"""Actual catalogue statements and provenance links in a disposable graph."""

import pytest

from iter_codex.discovery.signals.parallel import ingest_discovered_signals
from iter_codex.discovery.signals.scanners import codac_uda as catalogue
from iter_codex.graph.client import GraphClient
from iter_codex.graph.models import ArchiveEndpoint
from iter_codex.remote import codac_uda as service

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
            "archive_metadata": {"fixture": "complete archive document"},
            "data_type": "DOUBLE",
            "structure_fields": [{"name": "val", "type": "DOUBLE"}],
            "metadata_state": "available",
            "archive_labels": ["Enabled", "Disabled"],
            "archive_format": "%.2f",
            "archive_lower_display_limit": 0,
            "archive_upper_alarm_limit": 990,
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
        from iter_codex.discovery.signals.parallel import build_signal_scope_case

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
        fields = graph.query(
            "MATCH (s:FacilitySignal {id: $id}) RETURN s.unit AS unit, "
            "s.data_type AS data_type, count(s.archive_metadata) AS documents, "
            "count(s.structure_fields) AS structures, count(s.first_sample_ns) AS first_times, "
            "count(s.last_sample_ns) AS last_times, count(s.listed_archives) AS listings",
            id=signal.id,
        )
        assert fields == [
            {
                "unit": "K",
                "data_type": "DOUBLE",
                "documents": 1,
                "structures": 1,
                "first_times": 0,
                "last_times": 0,
                "listings": 1,
            }
        ]
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
        attributes = graph.query(
            "MATCH (s:FacilitySignal {id:$id}) RETURN s.metadata_state AS state, "
            "s.archive_labels AS labels,s.archive_format AS format, "
            "s.archive_lower_display_limit AS lower,s.archive_upper_alarm_limit AS alarm",
            id=signal.id,
        )
        assert attributes == [
            {
                "state": "available",
                "labels": ["Enabled", "Disabled"],
                "format": "%.2f",
                "lower": 0,
                "alarm": 990,
            }
        ]
        refused = catalogue.CodacUDAScanner._signal(
            facility,
            archive,
            access,
            {
                "name": entry["name"],
                "metadata_state": "unavailable",
                "metadata_error_code": 9105,
                "metadata_error": "fixture metadata refusal",
            },
            "cohort",
        )
        assert ingest_discovered_signals([refused.model_dump(mode="json")]) == 1
        unavailable = graph.query(
            "MATCH (s:FacilitySignal {id:$id}) RETURN s.metadata_state AS state, "
            "s.metadata_error_code AS code,count(s.unit) AS units,count(s.archive_metadata) AS documents, "
            "count(s.archive_labels) AS labels,count(s.archive_format) AS formats, "
            "count(s.archive_lower_display_limit) AS limits,s.status AS status,s.claim_token AS claim,s.data_archive AS serving",
            id=signal.id,
        )
        assert unavailable == [
            {
                "state": "unavailable",
                "code": 9105,
                "units": 0,
                "documents": 0,
                "labels": 0,
                "formats": 0,
                "limits": 0,
                "status": "enriched",
                "claim": "held",
                "serving": f"{facility}:codac_uda:fast",
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

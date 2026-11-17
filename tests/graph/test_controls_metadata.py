"""Feed imports are catalogue-limited, provenance-labelled and idempotent."""

import json

import numpy as np
import pytest

from imas_codex.discovery.signals.controls_metadata import (
    FEED_SOURCE,
    import_controls_metadata,
)
from imas_codex.discovery.signals.scanners.codac_uda import _persist_archives
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import ArchiveEndpoint, FacilitySignal
from imas_codex.remote.archive_catalogue import ArchiveCatalogue

pytestmark = pytest.mark.graph


class FixtureEncoder:
    def __init__(self):
        self.texts = []

    def embed_texts(self, texts):
        self.texts.extend(texts)
        return np.ones((len(texts), 256))


def test_controls_import_preserves_archive_state_and_excludes_unknowns(
    disposable_graph, tmp_path
):
    facility = "controls-fixture"
    known = "DEMO:TEMP_ALPHA"
    repeated = "DEMO:TEMP_BETA"
    missing = "DEMO:NOT_IN_CENSUS"
    catalogue_path = tmp_path / "catalogue.sqlite"
    catalogue = ArchiveCatalogue(catalogue_path)
    catalogue.seed({"slow": [known, repeated]}, timestamp="2026-01-01")
    catalogue.close()
    source = tmp_path / "feed.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": known,
                    "description": "Cryostat helium outlet temperature",
                    "unit": "K",
                    "data_type": "FLOAT",
                    "update_time": "2026-01-02 03:04:05",
                },
                {
                    "name": repeated,
                    "description": repeated,
                    "unit": "",
                    "data_type": "BOOL",
                    "update_time": "2026-01-02 03:04:05",
                },
                {
                    "name": missing,
                    "description": "Unknown pressure measurement",
                    "unit": "Pa",
                    "data_type": "FLOAT",
                    "update_time": "2026-01-02 03:04:05",
                },
            ]
        )
    )
    existing_id = f"{facility}:general/codac_uda/{known}"
    with GraphClient() as graph:
        graph.query("CREATE (:Facility {id: $id})", id=facility)
        _persist_archives(
            facility,
            [
                ArchiveEndpoint(
                    id=f"{facility}:codac_uda:slow",
                    facility_id=facility,
                    data_system="codac_uda",
                    role="slow",
                    excluded=False,
                )
            ],
        )
        graph.query(
            "CREATE (:FacilitySignal {id:$id,facility_id:$facility,accessor:$name,"
            "name:$name,status:'enriched',claim_token:'held',source_description:'Archive wording',"
            "data_archive:'confirmed',checked:true,unit:'V',data_type:'DOUBLE'})",
            id=existing_id,
            facility=facility,
            name=known,
        )
        encoder = FixtureEncoder()
        args = {
            "response": source,
            "catalogue": catalogue_path,
            "positive_control": known,
            "facility": facility,
            "graph": graph,
            "encoder": encoder,
        }
        first = import_controls_metadata(**args)
        assert first["entries_read"] == 3
        assert first["attached"] == 2
        assert first["signals_added"] == 1
        assert first["nodes_changed"] == 2
        assert first["left_out"] == {"absent_from_catalogue": 1}
        assert first["excluded_names"] == [missing]
        assert first["with_unit"] == 1
        assert first["name_only_descriptions"] == 1
        assert encoder.texts == ["Cryostat helium outlet temperature"]
        rows = graph.query(
            "MATCH (s:FacilitySignal {id:$id}) RETURN properties(s) AS props",
            id=existing_id,
        )
        props = rows[0]["props"]
        assert props["description"] == "Cryostat helium outlet temperature"
        assert props["description_source"] == FEED_SOURCE
        assert props["description_source_updated_at"] == "2026-01-02 03:04:05"
        assert props["unit"] == "K"
        assert props["data_type"] == "FLOAT"
        assert props["source_description"] == "Archive wording"
        assert props["status"] == "enriched"
        assert props["claim_token"] == "held"
        assert props["data_archive"] == "confirmed"
        assert props["checked"] is True
        FacilitySignal.model_validate(
            {**props, "embedded_at": props["embedded_at"].to_native()}
        )
        assert graph.query(
            "MATCH (s:FacilitySignal {facility_id:$facility}) "
            "RETURN count(s) AS total, sum(CASE WHEN EXISTS { "
            "(s)-[:AT_FACILITY]->(:Facility {id:$facility}) } THEN 1 ELSE 0 END) AS linked",
            facility=facility,
        ) == [{"total": 2, "linked": 2}]
        before = graph.query(
            "MATCH (s:FacilitySignal {facility_id:$facility}) RETURN s.id AS id, properties(s) AS props ORDER BY id",
            facility=facility,
        )
        second = import_controls_metadata(**args)
        assert second["signals_added"] == 0
        assert second["nodes_changed"] == 0
        assert before == graph.query(
            "MATCH (s:FacilitySignal {facility_id:$facility}) RETURN s.id AS id, properties(s) AS props ORDER BY id",
            facility=facility,
        )
        assert graph.query(
            "MATCH (s:FacilitySignal {id:$id}) RETURN count(s) AS total",
            id=f"{facility}:general/codac_uda/{missing}",
        ) == [{"total": 0}]


def test_description_provenance_survives_reembedding_but_not_a_rewrite(
    disposable_graph,
):
    from imas_codex.discovery.signals.parallel import write_signal_descriptions

    with GraphClient() as graph:
        graph.query(
            "CREATE (:FacilitySignal {id:'demo:temperature', description:'Helium inlet temperature', "
            "description_source:$source, description_source_updated_at:'2026-01-02 03:04:05'})",
            source=FEED_SOURCE,
        )
        write_signal_descriptions(
            [{"id": "demo:temperature", "description": "Helium inlet temperature"}],
            gc=graph,
            precomputed_embeddings={"demo:temperature": [1.0] * 256},
        )
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'demo:temperature'}) RETURN s.description_source AS source"
        ) == [{"source": FEED_SOURCE}]
        write_signal_descriptions(
            [{"id": "demo:temperature", "description": "Helium outlet temperature"}],
            gc=graph,
            precomputed_embeddings={"demo:temperature": [1.0] * 256},
        )
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'demo:temperature'}) "
            "RETURN s.description_source AS source, s.description_source_updated_at AS updated"
        ) == [{"source": None, "updated": None}]
        write_signal_descriptions(
            [{"id": "demo:temperature", "description": "TEMPERATURE"}],
            gc=graph,
            embed_text=lambda item: "",
            precomputed_embeddings={"demo:temperature": None},
        )
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'demo:temperature'}) RETURN s.embedding AS embedding"
        ) == [{"embedding": None}]


def test_archive_rescan_preserves_feed_unit_type_and_description(disposable_graph):
    from imas_codex.discovery.signals.scanners.codac_uda import (
        CodacUDAScanner,
        _access,
        _persist_signals,
    )

    facility = "controls-rescan"
    archive = f"{facility}:codac_uda:slow"
    with GraphClient() as graph:
        graph.query("CREATE (:Facility {id:$id})", id=facility)
        _persist_archives(
            facility,
            [
                ArchiveEndpoint(
                    id=archive,
                    facility_id=facility,
                    data_system="codac_uda",
                    role="slow",
                    excluded=False,
                )
            ],
        )
        signal = CodacUDAScanner._signal(
            facility,
            archive,
            _access(facility, "slow").id,
            {
                "name": "DEMO:TEMPERATURE",
                "archive_metadata": {"fixture": True},
                "unit": {"value": "V"},
                "description": {"value": "Archive wording"},
                "data_type": "DOUBLE",
            },
            "cohort",
        )
        assert _persist_signals(facility, "cohort", [signal]) == 1
        graph.query(
            "MATCH (s:FacilitySignal {id:$id}) SET s.description='Helium inlet temperature', "
            "s.description_source=$source, s.unit='K', s.data_type='FLOAT'",
            id=signal.id,
            source=FEED_SOURCE,
        )
        assert _persist_signals(facility, "cohort", [signal]) == 1
        assert graph.query(
            "MATCH (s:FacilitySignal {id:$id}) RETURN s.description AS description, "
            "s.unit AS unit, s.data_type AS data_type, s.source_description AS archive_wording",
            id=signal.id,
        ) == [
            {
                "description": "Helium inlet temperature",
                "unit": "K",
                "data_type": "FLOAT",
                "archive_wording": "Archive wording",
            }
        ]

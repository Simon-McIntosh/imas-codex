"""EDAS catalogue wording survives signal enrichment and rescanning."""

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from imas_codex.discovery.base.reset import SIGNAL_RESET_SPECS, reset_to_status
from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.models import (
    SignalEnrichmentBatch,
    SignalEnrichmentResult,
)
from imas_codex.discovery.signals.scanners.base import ScanResult
from imas_codex.discovery.signals.scanners.edas import EDASScanner
from imas_codex.graph import GraphClient

CATALOGUE_TEXT = "Lower FPPCC Current (LKAT2)"
SIGNAL_ID = "jt-60sa:general/mmsys_curlfpplkat"
SCAN_CONFIG = {"reference_shot": 101173, "api_path": "/api", "lib_path": "/lib"}


async def _scan_signal():
    catalogue = {
        "signals": [
            {
                "category": "MMSYS",
                "data_name": "curLFPPLKAT",
                "description": CATALOGUE_TEXT,
                "units": "A",
            }
        ],
        "ncats": 1,
    }
    remote = AsyncMock(return_value=json.dumps(catalogue))
    with patch("imas_codex.remote.executor.async_run_python_script", remote):
        result = await EDASScanner().scan("jt-60sa", "facility-host", SCAN_CONFIG)
    remote.assert_awaited_once()
    return result.signals[0]


@pytest.mark.asyncio
async def test_scanned_catalogue_text_survives_enrichment_and_discovered_reset():
    signal = await _scan_signal()
    assert signal.source_description == CATALOGUE_TEXT
    enriched = signal.model_copy(
        update={"description": "Model wording", "status": "enriched"}
    )
    assert enriched.source_description == CATALOGUE_TEXT
    reset = SIGNAL_RESET_SPECS["discovered"]
    assert "description" in reset.clear_fields
    assert "source_description" not in reset.clear_fields
    reset_row = enriched.model_dump()
    for field in reset.clear_fields:
        reset_row[field] = None
    reset_row["status"] = reset.target_status
    assert reset_row["source_description"] == CATALOGUE_TEXT


@pytest.mark.asyncio
async def test_reset_signal_prompt_uses_catalogue_text():
    signal = {
        "id": SIGNAL_ID,
        "facility_id": "jt-60sa",
        "accessor": "eddbreadTime('E101173', 'MMSYS', 'curLFPPLKAT', t1, t2)",
        "data_source_name": "edas",
        "data_source_path": "MMSYS/curLFPPLKAT",
        "discovery_source": "edas",
        "name": None,
        "description": None,
        "source_description": CATALOGUE_TEXT,
    }
    state = parallel.DataDiscoveryState(facility="jt-60sa", scanner_types=["edas"])
    claims = iter([[signal], []])
    prompts = []

    def claim(*_args, **_kwargs):
        batch = next(claims)
        if not batch:
            state.stop_requested = True
        return batch

    async def llm(*, messages, **_kwargs):
        prompts.append(messages[1]["content"])
        return (
            SignalEnrichmentBatch(
                results=[
                    SignalEnrichmentResult(
                        signal_index=1,
                        physics_domain="general",
                        name="Lower plasma control coil current",
                        description="Model wording",
                    )
                ]
            ),
            0.0,
            0,
        )

    with (
        patch.object(parallel, "claim_signals_for_enrichment", side_effect=claim),
        patch.object(parallel, "detect_signal_sources", return_value=(0, 0)),
        patch.object(parallel, "propagate_units_from_signal_nodes", return_value=0),
        patch.object(parallel, "fetch_tree_context", return_value={}),
        patch.object(parallel, "fetch_epoch_context", return_value={}),
        patch.object(parallel, "fetch_signal_code_refs", return_value={}),
        patch.object(parallel, "_fetch_code_chunks", return_value=[]),
        patch.object(parallel, "mark_signals_enriched"),
        patch.object(parallel, "mark_signals_underspecified"),
        patch.object(parallel, "propagate_source_enrichment", return_value=0),
        patch(
            "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
            return_value=[],
        ),
        patch("imas_codex.discovery.base.llm.acall_llm_structured", side_effect=llm),
    ):
        await parallel.enrich_worker(state)

    assert len(prompts) == 1
    assert f"source_description: {CATALOGUE_TEXT}" in prompts[0]


@pytest.mark.asyncio
async def test_edas_seed_uses_catalogue_only_update_for_existing_rows():
    signal = await _scan_signal()
    state = parallel.DataDiscoveryState(
        facility="jt-60sa",
        scanner_types=["edas"],
        facility_config={"data_systems": {"edas": SCAN_CONFIG}},
    )

    class Scanner:
        async def scan(self, **_kwargs):
            return ScanResult(signals=[signal])

    with (
        patch(
            "imas_codex.discovery.signals.scanners.base.get_scanner",
            return_value=Scanner(),
        ),
        patch.object(parallel, "ingest_discovered_signals", return_value=1) as ingest,
    ):
        await parallel.seed_worker(state)

    ingest.assert_called_once()
    assert ingest.call_args.kwargs == {}
    assert ingest.call_args.args[0][0]["source_description"] == CATALOGUE_TEXT
    assert ingest.call_args.args[0][0]["discovery_source"] == "edas"


@pytest.mark.graph
@pytest.mark.asyncio
async def test_reenumeration_only_refills_catalogue_text_on_enriched_signal():
    signal = await _scan_signal()
    facility = f"catalogue-text:{uuid4()}"
    signal_id = f"{facility}:signal"
    scanned = signal.model_dump(exclude_none=True)
    scanned.update(id=signal_id, facility_id=facility)

    class TransactionClient:
        def __init__(self, transaction):
            self.transaction = transaction

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, statement, **params):
            return [dict(row) for row in self.transaction.run(statement, **params)]

    with GraphClient() as graph, graph.session() as session:
        transaction = session.begin_transaction()
        try:
            transaction.run("MERGE (:Facility {id: $id})", id=facility).consume()
            transaction.run(
                """CREATE (:FacilitySignal {
                    id: $id, facility_id: $facility, status: 'enriched',
                    description: 'Existing model wording', name: 'Existing name'
                })""",
                id=signal_id,
                facility=facility,
            ).consume()
            with patch.object(
                parallel, "GraphClient", return_value=TransactionClient(transaction)
            ):
                assert parallel.ingest_discovered_signals([scanned]) == 1
            row = transaction.run(
                """MATCH (s:FacilitySignal {id: $id})
                RETURN s.status AS status, s.description AS description,
                       s.name AS name, s.source_description AS source_description""",
                id=signal_id,
            ).single()
            assert dict(row) == {
                "status": "enriched",
                "description": "Existing model wording",
                "name": "Existing name",
                "source_description": CATALOGUE_TEXT,
            }
            with patch.object(
                parallel, "GraphClient", return_value=TransactionClient(transaction)
            ):
                assert (
                    parallel.mark_signals_enriched(
                        [
                            {
                                "id": signal_id,
                                "physics_domain": "general",
                                "description": "Fresh model wording",
                                "name": "Fresh name",
                            }
                        ]
                    )
                    == 1
                )
            enriched_row = transaction.run(
                """MATCH (s:FacilitySignal {id: $id})
                RETURN s.description AS description,
                       s.source_description AS source_description""",
                id=signal_id,
            ).single()
            assert dict(enriched_row) == {
                "description": "Fresh model wording",
                "source_description": CATALOGUE_TEXT,
            }
            with patch(
                "imas_codex.graph.GraphClient",
                return_value=TransactionClient(transaction),
            ):
                assert reset_to_status(SIGNAL_RESET_SPECS["discovered"], facility) == 1
            reset_row = transaction.run(
                """MATCH (s:FacilitySignal {id: $id})
                RETURN s.status AS status, s.description AS description,
                       s.source_description AS source_description""",
                id=signal_id,
            ).single()
            assert dict(reset_row) == {
                "status": "discovered",
                "description": None,
                "source_description": CATALOGUE_TEXT,
            }
        finally:
            transaction.rollback()

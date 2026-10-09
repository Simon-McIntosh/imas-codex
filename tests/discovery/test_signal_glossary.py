"""Facility glossary context in EDAS signal enrichment prompts."""

from unittest.mock import patch

import pytest
import yaml

from imas_codex.discovery.base.facility import get_facility
from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.models import (
    SignalEnrichmentBatch,
    SignalEnrichmentResult,
)


@pytest.mark.asyncio
async def test_edas_prompt_scopes_category_and_term_context_to_signals():
    config = get_facility("jt-60sa")
    signals = [
        {
            "id": "coil-path",
            "accessor": "read coil",
            "discovery_source": "edas",
            "data_source_path": "MMSYS/curUFPLKAT",
            "name": None,
        },
        {
            "id": "coil-name",
            "accessor": "read name",
            "discovery_source": "edas",
            "data_source_path": "MMSYS/other",
            "name": "LFP coil current",
        },
        {
            "id": "other",
            "accessor": "read other",
            "discovery_source": "edas",
            "data_source_path": "PSRC/magFluxLp1",
            "name": None,
        },
    ]
    state = parallel.DataDiscoveryState(
        facility="jt-60sa", scanner_types=["edas"], facility_config=config
    )
    claims = iter([signals, []])
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
                        signal_index=index,
                        physics_domain="general",
                        name="Signal",
                        description="A signal",
                    )
                    for index in range(1, len(signals) + 1)
                ]
            ),
            0.0,
            0,
        )

    with (
        patch.object(parallel, "claim_signals_for_enrichment", side_effect=claim),
        patch.object(parallel, "prepare_signal_sources", return_value=(0, 0, 0)),
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
    category, coil_path, coil_name, unrelated = prompts[0].split("### Signal")
    assert "MMSYS" in category
    assert "coil currents" in category
    assert "glossary: UFP — upper fast plasma position control coil" in coil_path
    assert "glossary: LFP — lower fast plasma position control coil" in coil_name
    assert "HiTe" not in unrelated
    assert "FPPCC" not in unrelated
    assert "coil currents" not in unrelated


def test_facility_glossary_is_declared_and_loaded():
    with open("imas_codex/schemas/facility_config.yaml") as stream:
        schema = yaml.safe_load(stream)
    assert "category_descriptions" in schema["classes"]["EDASConfig"]["attributes"]
    assert "signal_glossary" in schema["classes"]["FacilityConfig"]["attributes"]

    config = get_facility("jt-60sa")
    edas = config["data_systems"]["edas"]
    descriptions = {
        entry["code"]: entry["description"] for entry in edas["category_descriptions"]
    }
    glossary = {entry["term"]: entry["meaning"] for entry in config["signal_glossary"]}
    assert descriptions["MMSYS"]
    assert "upper fast plasma position control coil (FPPCC)" in glossary["UFP"]
    assert "lower fast plasma position control coil (FPPCC)" in glossary["LFP"]
    assert "same coil current" in glossary["HiTe"]
    assert "HiTec" in glossary["HiTe"]
    assert "LKAT2" in glossary["LKAT"]
    assert config["signal_member_patterns"]["MMSYS"] == (
        r"^cur(?P<member>.+?)(?:TFLKAT|LKAT|HiTe)$"
    )

"""Enrichment results reach the signal they describe.

The enrich worker numbers signals in its prompt group by group (all of one
EDAS category, then the next), while the claimed batch arrives in claim order.
When categories interleave in the batch, prompt number N is not batch row N,
so results must be matched through the prompt's own numbering.
"""

import re
from unittest.mock import patch

import pytest

from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.models import (
    SignalEnrichmentBatch,
    SignalEnrichmentResult,
)

FACILITY = "jt-60sa"

# Claim order interleaves three EDAS categories, so grouping by category
# reorders the prompt: MMSYS rows 0 and 2 come first, then OFMC, then TOPICS.
BATCH = [
    ("MMSYS", "curEF1LKAT"),
    ("OFMC", "WfastIon"),
    ("MMSYS", "curEF4LKAT"),
    ("TOPICS", "NMESH"),
]


def _signal(category: str, data_name: str) -> dict:
    return {
        "id": f"{FACILITY}:general/{category.lower()}_{data_name.lower()}",
        "accessor": f"eddbreadTime('E101173', '{category}', '{data_name}', t1, t2)",
        "name": f"{category}/{data_name}",
        "data_source_name": "edas",
        "data_source_path": f"{category}/{data_name}",
        "discovery_source": "edas",
        "unit": None,
        "description": "",
        "facility_id": FACILITY,
    }


def _answer_by_prompt_number(user_prompt: str) -> SignalEnrichmentBatch:
    """Describe each prompt signal by the accessor printed under its number."""
    numbered = re.findall(r"### Signal (\d+)\naccessor: (.+)", user_prompt)
    return SignalEnrichmentBatch(
        results=[
            SignalEnrichmentResult(
                signal_index=int(number),
                physics_domain="general",
                name=f"name for {accessor}",
                description=f"describes {accessor}",
            )
            for number, accessor in numbered
        ]
    )


@pytest.mark.asyncio
async def test_interleaved_batch_keeps_each_description_on_its_signal():
    signals = [_signal(category, name) for category, name in BATCH]
    state = parallel.DataDiscoveryState(facility=FACILITY, scanner_types=["edas"])
    claims = iter([signals])
    enriched_rows: list[dict] = []

    def claim(*args, **kwargs):
        batch = next(claims, None)
        if batch is None:
            state.stop_requested = True
            return []
        return batch

    async def llm(*, messages, **kwargs):
        user_prompt = messages[1]["content"]
        return _answer_by_prompt_number(user_prompt), 0.0, 0

    def mark_enriched(entries, *args, **kwargs):
        enriched_rows.extend(entries)

    with (
        patch.object(parallel, "claim_signals_for_enrichment", side_effect=claim),
        patch.object(parallel, "detect_signal_sources", return_value=(0, 0)),
        patch.object(parallel, "propagate_units_from_signal_nodes", return_value=0),
        patch.object(parallel, "fetch_tree_context", return_value={}),
        patch.object(parallel, "fetch_epoch_context", return_value={}),
        patch.object(parallel, "fetch_signal_code_refs", return_value={}),
        patch.object(parallel, "_fetch_code_chunks", return_value=[]),
        patch.object(parallel, "mark_signals_enriched", side_effect=mark_enriched),
        patch.object(parallel, "mark_signals_underspecified"),
        patch.object(parallel, "propagate_source_enrichment", return_value=0),
        patch.object(parallel, "release_signal_claim") as release,
        patch(
            "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
            return_value=[],
        ),
        patch("imas_codex.discovery.base.llm.acall_llm_structured", side_effect=llm),
    ):
        await parallel.enrich_worker(state)

    by_id = {row["id"]: row["description"] for row in enriched_rows}
    assert by_id == {s["id"]: f"describes {s['accessor']}" for s in signals}
    release.assert_not_called()


@pytest.mark.asyncio
async def test_repeated_prompt_number_writes_one_row():
    """A model that repeats an index cannot write one result onto two rows."""
    signals = [_signal(category, name) for category, name in BATCH[:2]]
    state = parallel.DataDiscoveryState(facility=FACILITY, scanner_types=["edas"])
    claims = iter([signals])
    enriched_rows: list[dict] = []

    def claim(*args, **kwargs):
        batch = next(claims, None)
        if batch is None:
            state.stop_requested = True
            return []
        return batch

    async def llm(*, messages, **kwargs):
        first = SignalEnrichmentResult(
            signal_index=1,
            physics_domain="general",
            name="first",
            description="first answer",
        )
        repeat = first.model_copy(update={"description": "repeated answer"})
        return SignalEnrichmentBatch(results=[first, repeat]), 0.0, 0

    def mark_enriched(entries, *args, **kwargs):
        enriched_rows.extend(entries)

    with (
        patch.object(parallel, "claim_signals_for_enrichment", side_effect=claim),
        patch.object(parallel, "detect_signal_sources", return_value=(0, 0)),
        patch.object(parallel, "propagate_units_from_signal_nodes", return_value=0),
        patch.object(parallel, "fetch_tree_context", return_value={}),
        patch.object(parallel, "fetch_epoch_context", return_value={}),
        patch.object(parallel, "fetch_signal_code_refs", return_value={}),
        patch.object(parallel, "_fetch_code_chunks", return_value=[]),
        patch.object(parallel, "mark_signals_enriched", side_effect=mark_enriched),
        patch.object(parallel, "mark_signals_underspecified"),
        patch.object(parallel, "propagate_source_enrichment", return_value=0),
        patch.object(parallel, "release_signal_claim") as release,
        patch(
            "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
            return_value=[],
        ),
        patch("imas_codex.discovery.base.llm.acall_llm_structured", side_effect=llm),
    ):
        await parallel.enrich_worker(state)

    assert [row["description"] for row in enriched_rows] == ["first answer"]
    release.assert_called_once()


@pytest.mark.asyncio
async def test_reset_signal_without_a_name_is_enriched():
    """A reset clears the enriched name; the worker groups by the source path."""
    signals = [_signal(category, name) for category, name in BATCH]
    for signal in signals:
        signal["name"] = None
    state = parallel.DataDiscoveryState(facility=FACILITY, scanner_types=["edas"])
    claims = iter([signals])
    enriched_rows: list[dict] = []

    def claim(*args, **kwargs):
        batch = next(claims, None)
        if batch is None:
            state.stop_requested = True
            return []
        return batch

    async def llm(*, messages, **kwargs):
        return _answer_by_prompt_number(messages[1]["content"]), 0.0, 0

    def mark_enriched(entries, *args, **kwargs):
        enriched_rows.extend(entries)

    with (
        patch.object(parallel, "claim_signals_for_enrichment", side_effect=claim),
        patch.object(parallel, "detect_signal_sources", return_value=(0, 0)),
        patch.object(parallel, "propagate_units_from_signal_nodes", return_value=0),
        patch.object(parallel, "fetch_tree_context", return_value={}),
        patch.object(parallel, "fetch_epoch_context", return_value={}),
        patch.object(parallel, "fetch_signal_code_refs", return_value={}),
        patch.object(parallel, "_fetch_code_chunks", return_value=[]),
        patch.object(parallel, "mark_signals_enriched", side_effect=mark_enriched),
        patch.object(parallel, "mark_signals_underspecified"),
        patch.object(parallel, "propagate_source_enrichment", return_value=0),
        patch.object(parallel, "release_signal_claim"),
        patch(
            "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
            return_value=[],
        ),
        patch("imas_codex.discovery.base.llm.acall_llm_structured", side_effect=llm),
    ):
        await parallel.enrich_worker(state)

    by_id = {row["id"]: row["description"] for row in enriched_rows}
    assert by_id == {s["id"]: f"describes {s['accessor']}" for s in signals}

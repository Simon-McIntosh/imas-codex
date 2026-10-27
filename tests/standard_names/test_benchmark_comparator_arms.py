"""The benchmark harness carries the production comparator block and draws
its batch from a named source list.

The compose prompt's nearby-name comparator block is reranked by the compose
rerank switch on the production path. The benchmark harness renders the same
template, so the same switch can only move its prompt once the block is
carried. These tests pin that the block is carried from the production helpers
(off by default, byte-identical when off), that the rerank switch reorders it,
and that an explicit source list drives extraction. No paid call is made; the
model call, the retrieval and the rerank are all mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.benchmark import BenchmarkConfig, _run_model
from imas_codex.standard_names.models import StandardNameComposeBatch

# A minimal compose context: the render is lenient, so undefined schema
# variables collapse to empty. The test is about what the harness supplies,
# not about the schema context.
_CONTEXT: dict = {"subjects": ["electron"], "vocabulary_sections": []}

_BATCHES: list[dict] = [
    {
        "group_key": "test",
        "items": [
            {
                "path": "test/path",
                "description": "Test",
                "units": None,
                "data_type": "FLT_0D",
                "cluster_label": None,
            }
        ],
        "existing_names": [],
        "context": "IDS: test",
    }
]

# Two neighbours in embedding order: the reversing rerank flips them.
_NEIGHBOURS: list[dict] = [
    {"id": "aaa_first_neighbour", "description": "first", "kind": "scalar"},
    {"id": "zzz_second_neighbour", "description": "second", "kind": "scalar"},
]

_NEARBY_HEADER = "Nearby Existing Standard Names"


async def _capture_prompt(config: BenchmarkConfig) -> str:
    """Run one batch through ``_run_model`` and return the rendered user prompt.

    The model call is mocked; the prompt is rendered by the real template so
    the test sees the text the harness would send.
    """
    captured: dict[str, str] = {}

    async def _mock_llm(model, messages, response_model, **kwargs):
        captured["user"] = messages[1]["content"]
        return StandardNameComposeBatch(candidates=[], skipped=[]), 0.0, 0

    with patch(
        "imas_codex.discovery.base.llm.acall_llm_structured",
        side_effect=_mock_llm,
    ):
        await _run_model(
            model="test",
            extraction_batches=_BATCHES,
            config=config,
            reference={},
            system_prompt="System instructions",
            context=_CONTEXT,
        )
    return captured["user"]


@pytest.mark.asyncio
async def test_block_on_prompt_carries_the_neighbour() -> None:
    """With the field on, the rendered compose prompt carries the neighbour."""
    config = BenchmarkConfig(models=["test"], comparator_block=True)

    with patch(
        "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._collect_nearby_name_comparators",
        return_value=list(_NEIGHBOURS),
    ):
        prompt = await _capture_prompt(config)

    assert _NEARBY_HEADER in prompt
    assert "aaa_first_neighbour" in prompt
    assert "zzz_second_neighbour" in prompt


@pytest.mark.asyncio
async def test_rerank_switch_reorders_the_carried_block() -> None:
    """The rerank switch moves the block, so the two arms can differ.

    The switch off keeps the embedding order; the switch on with a reversing
    rerank flips it. This is the A/B the rerank measure needs the harness to
    expose.
    """
    config = BenchmarkConfig(models=["test"], comparator_block=True)
    reversed_neighbours = list(reversed(_NEIGHBOURS))

    with (
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._collect_nearby_name_comparators",
            return_value=list(_NEIGHBOURS),
        ),
        patch(
            "imas_codex.standard_names.context.comparator_rerank_enabled",
            return_value=False,
        ),
    ):
        prompt_off = await _capture_prompt(config)

    with (
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._collect_nearby_name_comparators",
            return_value=list(_NEIGHBOURS),
        ),
        patch(
            "imas_codex.standard_names.context.comparator_rerank_enabled",
            return_value=True,
        ),
        patch(
            "imas_codex.standard_names.context.rerank_name_comparators",
            new=AsyncMock(return_value=(reversed_neighbours, 0.0, "")),
        ),
    ):
        prompt_on = await _capture_prompt(config)

    assert prompt_off != prompt_on
    # Off keeps embedding order; on reverses it.
    assert prompt_off.index("aaa_first_neighbour") < prompt_off.index(
        "zzz_second_neighbour"
    )
    assert prompt_on.index("zzz_second_neighbour") < prompt_on.index(
        "aaa_first_neighbour"
    )


@pytest.mark.asyncio
async def test_block_off_render_matches_the_base_revision() -> None:
    """With the field off the prompt is byte-identical to the base render.

    The base revision supplied no ``nearby_existing_names``; the off path must
    reproduce that render exactly so switching the field off changes nothing.
    """
    from imas_codex.llm.prompt_loader import render_prompt

    config = BenchmarkConfig(models=["test"])
    prompt = await _capture_prompt(config)

    base_prompt = render_prompt(
        "sn/generate_name_dd",
        {
            **_CONTEXT,
            "items": _BATCHES[0]["items"],
            "ids_name": "test",
            "existing_names": [],
            "cluster_context": "IDS: test",
        },
    )

    assert prompt == base_prompt
    assert _NEARBY_HEADER not in prompt


def test_named_source_paths_drive_extraction() -> None:
    """A named source list is what extraction composes, not the reference set."""
    from imas_codex.standard_names.benchmark import _extract_candidates
    from imas_codex.standard_names.sources.base import ExtractionBatch

    named_paths = ["equilibrium/time_slice/profiles_1d/psi", "magnetics/b_field"]
    fake_batch = ExtractionBatch(
        source="dd",
        group_key="equilibrium",
        items=[{"path": p, "description": "x"} for p in named_paths],
        context="IDS: equilibrium",
        existing_names=set(),
    )
    captured: dict = {}

    def _fake_extract(*, explicit_paths, **kwargs):
        captured["explicit_paths"] = list(explicit_paths)
        return [fake_batch]

    config = BenchmarkConfig(models=["test"], source_paths=named_paths)

    with patch(
        "imas_codex.standard_names.sources.dd.extract_dd_candidates",
        side_effect=_fake_extract,
    ):
        batches = _extract_candidates(config)

    assert captured["explicit_paths"] == named_paths
    composed_paths = [item["path"] for item in batches[0]["items"]]
    assert composed_paths == named_paths

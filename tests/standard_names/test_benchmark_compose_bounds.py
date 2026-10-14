"""The harness compose call sends the production compose call's bounds.

A compose call the harness issues must reach the provider with the same
effective request the production compose worker sends for the same seat: the
same reasoning setting and the same model-family token and time bounds. A
parameter the harness sets and production omits is a difference the provider
acts on, so this test renders both calls' requests through the shared kwargs
builder and requires them to agree. No paid call is made; the model call is
stubbed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imas_codex.discovery.base.llm import _build_kwargs
from imas_codex.settings import get_reasoning_effort
from imas_codex.standard_names.benchmark import BenchmarkConfig, _run_model
from imas_codex.standard_names.models import StandardNameComposeBatch

# A self-hosted seat, so the kwargs builder applies the local model-family
# token and time bounds rather than a provider default.
_SEAT = "local/deepseek-v4.1-flash"

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


async def _capture_compose_call(config: BenchmarkConfig) -> dict:
    """Run one batch through ``_run_model`` and return the model-call kwargs."""
    captured: dict = {}

    async def _mock_llm(model, messages, response_model, **kwargs):
        captured.update(kwargs)
        captured["_model"] = model
        captured["_messages"] = messages
        captured["_response_model"] = response_model
        return StandardNameComposeBatch(candidates=[], skipped=[]), 0.0, 0

    with patch(
        "imas_codex.discovery.base.llm.acall_llm_structured",
        side_effect=_mock_llm,
    ):
        await _run_model(
            model=_SEAT,
            extraction_batches=_BATCHES,
            config=config,
            reference={},
            system_prompt="System instructions",
            context=_CONTEXT,
        )
    return captured


def _effective_request(call: dict) -> dict:
    """Render one compose call's parameters into the provider request.

    Both the harness and the production call are rendered through the same
    builder, so a parameter one passes and the other omits shows up as an
    unequal request rather than as a call-site detail.
    """
    return _build_kwargs(
        _SEAT,
        "EMPTY",
        call["_messages"],
        StandardNameComposeBatch,
        call.get("max_tokens"),
        call.get("temperature"),
        call.get("timeout"),
        service=call["service"],
        reasoning_effort=call["reasoning_effort"],
    )


@pytest.mark.asyncio
async def test_harness_compose_call_omits_temperature() -> None:
    """The harness must not set a temperature the production call leaves unset."""
    config = BenchmarkConfig(models=[_SEAT], temperature=0.0)

    call = await _capture_compose_call(config)

    assert "temperature" not in call
    assert "max_tokens" not in call
    assert "timeout" not in call
    assert "seed" not in call


@pytest.mark.asyncio
async def test_harness_compose_request_matches_production() -> None:
    """The rendered harness request equals the production compose request."""
    config = BenchmarkConfig(models=[_SEAT], temperature=0.0)

    call = await _capture_compose_call(config)
    production = {
        "_messages": call["_messages"],
        "_response_model": call["_response_model"],
        "service": "standard-names",
        "reasoning_effort": get_reasoning_effort("sn-compose"),
    }

    assert _effective_request(call) == _effective_request(production)


@pytest.mark.asyncio
async def test_harness_compose_call_carries_production_reasoning_effort() -> None:
    """The harness composes at the campaign reasoning effort, not a default."""
    config = BenchmarkConfig(models=[_SEAT])

    call = await _capture_compose_call(config)

    assert call["service"] == "standard-names"
    assert call["reasoning_effort"] == get_reasoning_effort("sn-compose")

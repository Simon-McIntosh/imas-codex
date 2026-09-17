"""Distinguishability of an endpoint drop from a model refusal on a compose batch.

The two failures reach the per-batch handler of ``_run_model`` as different
exceptions - a transport error that produced no provider response, versus a
structured-call error carrying one or more provider responses - and the test
asserts the recorded outcome names which of the two occurred.
"""

import asyncio
from unittest.mock import patch

import pytest

from imas_codex.discovery.base import llm as llm_mod
from imas_codex.standard_names import benchmark as bench


def _endpoint_drop_error() -> BaseException:
    """A transport failure observed before any provider response arrives."""
    return ConnectionError("Connection error.")


def _model_refusal_error() -> BaseException:
    """A provider response arrived and could not be turned into a valid batch."""
    return llm_mod._structured_call_error(
        "1 validation error for StandardNameComposeBatch",
        cost=0.0,
        input_tokens=11,
        output_tokens=7,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        response_count=2,
    )


async def _recorded(exc: BaseException) -> bench.ModelResult:
    async def fake_structured(**kwargs):
        raise exc

    config = bench.BenchmarkConfig(models=["probe-model"], names_only=True)
    batches = [{"group_key": "probe", "items": [{"id": "x", "description": "d"}]}]
    with patch.object(llm_mod, "acall_llm_structured", fake_structured):
        return await bench._run_model(
            model="probe-model",
            extraction_batches=batches,
            config=config,
            reference={},
            system_prompt="probe",
            context={},
        )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_endpoint_drop_error(), "endpoint_drop"),
        (_model_refusal_error(), "model_refusal"),
    ],
)
def test_batch_failure_cause_is_recorded(exc, expected):
    result = asyncio.run(_recorded(exc))
    assert getattr(result, "batch_error_causes", {}).get(expected) == 1, (
        f"compose failure {type(exc).__name__} was recorded as "
        f"{getattr(result, 'batch_error_causes', None)!r}, "
        f"which does not name {expected} (batch_errors={result.batch_errors})"
    )


def test_the_two_failures_are_not_recorded_identically():
    drop = asyncio.run(_recorded(_endpoint_drop_error()))
    refusal = asyncio.run(_recorded(_model_refusal_error()))
    assert drop.batch_errors == refusal.batch_errors == 1
    assert getattr(drop, "batch_error_causes", None) != getattr(
        refusal, "batch_error_causes", None
    ), "the recorded outcome is the same for an endpoint drop and a model refusal"
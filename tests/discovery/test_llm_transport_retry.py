"""A transport fault is retried by type inside the structured call's loop.

``_is_retryable`` classifies the error *message*. A dropped connection carries
an empty one (``httpx.ReadError``) and a bare connect refusal carries no phrase
its patterns match (``httpx.ConnectError``), so message classification alone
gives each of them a single attempt. The type test must decide these faults
before the text is read, and the attempt count must stay bounded by
``max_retries`` rather than growing a second loop on top of this one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel

from imas_codex.discovery.base.llm import acall_llm_structured

_MODEL = "openrouter/qwen/qwen-max"
_MAX_RETRIES = 3


class _Answer(BaseModel):
    name: str


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-test")


async def _call(max_retries: int = _MAX_RETRIES):
    return await acall_llm_structured(
        model=_MODEL,
        messages=[{"role": "user", "content": "propose a name"}],
        response_model=_Answer,
        max_retries=max_retries,
        retry_base_delay=0.0,
        service="standard-names",
    )


@pytest.mark.parametrize(
    "transport_error",
    [
        httpx.ConnectError("Cannot connect to host openrouter.ai:443"),
        httpx.ReadError(""),
    ],
)
async def test_transport_fault_is_retried_and_bounded(transport_error) -> None:
    fake = AsyncMock(side_effect=transport_error)
    sleep = AsyncMock(return_value=None)

    with (
        patch("litellm.acompletion", fake),
        patch("asyncio.sleep", sleep),
        pytest.raises(ValueError, match=f"failed after {_MAX_RETRIES} attempts"),
    ):
        await _call()

    assert fake.call_count == _MAX_RETRIES
    assert sleep.await_count == _MAX_RETRIES - 1


async def test_a_deterministic_error_is_still_given_one_attempt() -> None:
    """The type test must not widen the loop to every exception."""
    fake = AsyncMock(side_effect=ValueError("token is not a registered vocab token"))
    sleep = AsyncMock(return_value=None)

    with (
        patch("litellm.acompletion", fake),
        patch("asyncio.sleep", sleep),
        pytest.raises(ValueError, match="not a registered"),
    ):
        await _call()

    assert fake.call_count == 1
    assert sleep.await_count == 0


class _SeamTransportError(Exception):
    """A transport class carrying no phrase the message patterns match."""


async def test_the_retry_set_is_read_from_the_transient_helper() -> None:
    """The loop must ask ``_transient_transport_errors`` rather than restate it.

    The type set is the seam's own, so a fault class it adds is retried without
    a second copy of the list here. A hardcoded tuple in the retry loop gives
    this class one attempt and fails the test.
    """
    fake = AsyncMock(side_effect=_SeamTransportError("seam dropped"))
    sleep = AsyncMock(return_value=None)

    with (
        patch("litellm.acompletion", fake),
        patch("asyncio.sleep", sleep),
        patch(
            "imas_codex.discovery.base.llm._transient_transport_errors",
            return_value=(_SeamTransportError,),
        ),
        pytest.raises(ValueError, match=f"failed after {_MAX_RETRIES} attempts"),
    ):
        await _call()

    assert fake.call_count == _MAX_RETRIES
    assert sleep.await_count == _MAX_RETRIES - 1

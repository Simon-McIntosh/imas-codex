"""Request starts remain spaced across synchronous and asynchronous callers."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from iter_codex.discovery.base import request_pacer as module
from iter_codex.discovery.base.request_pacer import RequestPacer


@pytest.mark.parametrize("interval", (-1, float("inf"), float("nan")))
def test_invalid_interval_refused(interval):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        RequestPacer(interval)


def test_sync_wait_spaces_starts_and_skips_initial_sleep(monkeypatch):
    now = [10.0]
    sleep = Mock(side_effect=lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    pacer = RequestPacer(0.5)
    pacer.wait()
    sleep.assert_not_called()
    now[0] += 0.1
    pacer.wait()
    assert sleep.call_args.args[0] == pytest.approx(0.4)
    now[0] += 1
    pacer.wait()
    assert sleep.call_count == 1


@pytest.mark.asyncio
async def test_async_callers_cannot_share_a_request_slot(monkeypatch):
    now = [10.0]
    yield_control = asyncio.sleep

    async def advance(delay):
        deadline = now[0] + delay
        await yield_control(0)
        now[0] = max(now[0], deadline)

    sleep = AsyncMock(side_effect=advance)
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    pacer = RequestPacer(0.5)

    async def start():
        await pacer.wait_async()
        return now[0]

    starts = await asyncio.gather(*(start() for _ in range(4)))
    assert starts == [10.0, 10.5, 11.0, 11.5]
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 0.5, 0.5]


@pytest.mark.asyncio
async def test_cancelled_async_wait_does_not_consume_slot(monkeypatch):
    pacer = RequestPacer(0.5)
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    await pacer.wait_async()
    monkeypatch.setattr(
        module.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)
    )
    with pytest.raises(asyncio.CancelledError):
        await pacer.wait_async()
    assert pacer._last_start == 10.0
    assert not pacer._async_lock.locked()


@pytest.mark.parametrize("client_kind", ("confluence", "mediawiki", "async_mediawiki"))
@pytest.mark.asyncio
async def test_wiki_clients_delegate_spacing(client_kind):
    from iter_codex.discovery.wiki.confluence import ConfluenceClient
    from iter_codex.discovery.wiki.mediawiki import (
        AsyncMediaWikiClient,
        MediaWikiClient,
    )

    kind = {
        "confluence": ConfluenceClient,
        "mediawiki": MediaWikiClient,
        "async_mediawiki": AsyncMediaWikiClient,
    }[client_kind]
    instance = kind.__new__(kind)
    pacer = Mock(spec=RequestPacer)
    pacer.wait_async = AsyncMock()
    instance._request_pacer = pacer
    if client_kind == "async_mediawiki":
        await instance._rate_limit()
        pacer.wait_async.assert_awaited_once()
    else:
        instance._rate_limit()
        pacer.wait.assert_called_once()

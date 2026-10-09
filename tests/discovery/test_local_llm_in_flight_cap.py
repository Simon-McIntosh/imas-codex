"""A pipeline process never holds more local-lane requests than its cap."""

import asyncio
import threading
import time

from imas_codex.discovery.base import llm

KWARGS = {"api_base": "http://local.example/v1", "model": "local/test", "messages": []}


class Gauge:
    def __init__(self):
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def enter(self):
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self):
        with self.lock:
            self.current -= 1


def test_async_local_calls_never_exceed_the_cap(monkeypatch):
    gauge = Gauge()

    class Completions:
        async def create(self, **_kwargs):
            gauge.enter()
            await asyncio.sleep(0.05)
            gauge.leave()
            return "ok"

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(llm, "_LOCAL_IN_FLIGHT", threading.BoundedSemaphore(2))
    monkeypatch.setattr(llm, "_LOCAL_SLOT_POLL_SECONDS", 0.005)
    monkeypatch.setattr(llm, "_get_local_client", lambda *_args: Client())
    monkeypatch.setattr(llm, "_local_call_kwargs", lambda kwargs: kwargs)

    async def burst():
        return await asyncio.gather(*(llm._acompletion_local(KWARGS) for _ in range(8)))

    assert asyncio.run(burst()) == ["ok"] * 8
    assert gauge.peak == 2


def test_threaded_local_calls_never_exceed_the_cap(monkeypatch):
    gauge = Gauge()

    class Completions:
        def create(self, **_kwargs):
            gauge.enter()
            time.sleep(0.05)
            gauge.leave()
            return "ok"

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(llm, "_LOCAL_IN_FLIGHT", threading.BoundedSemaphore(3))
    monkeypatch.setattr(llm, "_get_local_sync_client", lambda *_args: Client())
    monkeypatch.setattr(llm, "_local_call_kwargs", lambda kwargs: kwargs)

    threads = [
        threading.Thread(target=llm._completion_local, args=(KWARGS,)) for _ in range(9)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert gauge.peak == 3

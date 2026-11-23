"""Plant tool calls complete while the REPL reinitializes its state.

A tool body imports support modules at call time. The REPL's reload used to
delete every ``iter_codex`` module from ``sys.modules``, so a reload that ran
while one of those imports was in flight made importlib pop a name that no
longer existed and raise ``KeyError`` for it. The tool call then failed with a
missing-module error that a fresh process never reproduced.

The reproduction below runs the reload from inside the tool's own call-time
import, which is the worst interleaving real threads can produce. The graph is
faked at the driver factory in the ``neo4j`` package, outside ``iter_codex``,
so the fake survives whatever the reload does to the module cache.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.machinery import PathFinder
from typing import Any

import neo4j
import pytest
from fastmcp import Client

from iter_codex.llm.server import AgentsServer

CALL_TIME_IMPORT = "iter_codex.tools.plant_search"

FACILITY = "demo"
SIGNAL_ARGS = {"facility": FACILITY, "plant_system": "DEMO-UNIT", "k": 1}
PULSE_ARGS = {"facility": FACILITY, "designation": "demo-coil"}
RECIPE_ARGS: dict[str, Any] = {}
CONTEXT_ARGS = {"resource": f"{FACILITY}:signal-alpha"}
ANALYTICS_ARGS = {"facility": FACILITY}

TOOL_ARGS = {
    "search_signals": SIGNAL_ARGS,
    "search_pulses": PULSE_ARGS,
    "get_signal_load_recipe": RECIPE_ARGS,
    "fetch_content": CONTEXT_ARGS,
    "signal_analytics": ANALYTICS_ARGS,
}

ORDERS = [
    ("search_signals", "search_pulses", "get_signal_load_recipe"),
    ("search_pulses", "signal_analytics", "search_signals"),
    ("fetch_content", "search_signals", "search_pulses"),
    ("get_signal_load_recipe", "fetch_content", "signal_analytics"),
]

PAGE_ROW = {"total": 1, "items": [{"id": f"{FACILITY}:item-alpha"}]}
CONTEXT_ROW = {
    "signal": {"id": CONTEXT_ARGS["resource"]},
    "facility": FACILITY,
    "component": {"id": f"{FACILITY}:component-alpha"},
    "control_hierarchy": [{"id": f"{FACILITY}:control-alpha", "level": 1}],
}
ANALYTICS_ROW = {"status": "present", "count": 1}


class _FakeSession:
    def run(self, cypher: str, **parameters: Any) -> list[dict[str, Any]]:
        if "control_hierarchy" in cypher:
            return [dict(CONTEXT_ROW)]
        if "count(s) AS count" in cypher:
            return [dict(ANALYTICS_ROW)]
        return [dict(PAGE_ROW)]

    def close(self) -> None:
        return None


class _FakeDriver:
    def session(self, *args: Any, **kwargs: Any) -> _FakeSession:
        return _FakeSession()

    def close(self) -> None:
        return None


@pytest.fixture(scope="module")
def plant_server() -> Iterator[AgentsServer]:
    original_driver = neo4j.GraphDatabase.driver
    neo4j.GraphDatabase.driver = lambda *args, **kwargs: _FakeDriver()
    try:
        yield AgentsServer(dd_only=False, no_embed=True)
    finally:
        neo4j.GraphDatabase.driver = original_driver


class _ReloadingLoader:
    """Call the server's reload inside one module's import."""

    def __init__(self, inner: Any, reload_fn: Any, state: dict[str, bool]) -> None:
        self._inner = inner
        self._reload_fn = reload_fn
        self._state = state

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        if not self._state["fired"]:
            self._state["fired"] = True
            self._reload_fn()
        self._inner.exec_module(module)


class _ReloadDuringImport:
    def __init__(self, target: str, reload_fn: Any) -> None:
        self.target = target
        self.reload_fn = reload_fn
        self.state = {"fired": False}

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self.target:
            return None
        sys.meta_path.remove(self)
        try:
            spec = PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _ReloadingLoader(spec.loader, self.reload_fn, self.state)
        return spec


@contextmanager
def reload_inside_call_time_import() -> Iterator[None]:
    """Make the next tool call import its support module during a reload.

    Every existing test process has the module already imported, so it is
    removed first: that is the state any completed reload leaves behind.
    """
    from iter_codex.llm.server import _get_repl

    reload_fn = _get_repl()["reload"]
    sys.modules.pop(CALL_TIME_IMPORT, None)
    finder = _ReloadDuringImport(CALL_TIME_IMPORT, reload_fn)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [("search_signals", "search_pulses"), ("search_pulses", "search_signals")],
)
async def test_signal_and_pulse_searches_survive_fifty_reloads(
    plant_server: AgentsServer, first: str, second: str
) -> None:
    async with Client(plant_server.mcp) as client:
        for _ in range(50):
            for name in (first, second):
                with reload_inside_call_time_import():
                    response = await client.call_tool(name, TOOL_ARGS[name])
                assert json.loads(response.content[0].text)["items"]


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ORDERS, ids=lambda order: "-".join(order))
async def test_every_plant_tool_completes_across_several_orders(
    plant_server: AgentsServer, order: tuple[str, ...]
) -> None:
    async with Client(plant_server.mcp) as client:
        for name in order:
            with reload_inside_call_time_import():
                response = await client.call_tool(name, TOOL_ARGS[name])
            assert response.content[0].text


def test_reload_keeps_loaded_modules(plant_server: AgentsServer) -> None:
    import iter_codex.tools.plant_search  # noqa: F401
    from iter_codex.llm.server import _get_repl

    _get_repl()["reload"]()
    assert CALL_TIME_IMPORT in sys.modules

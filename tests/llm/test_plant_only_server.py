"""The plant container publishes the plant catalogue and no write tool.

The server's published listing is the whole capability an agent sees, so it
must be exactly the plant tools named here — a write tool left on it would be
offered against a read-only package.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.server import AgentsServer


def _tool_names(server: AgentsServer) -> set[str]:
    return {
        k.removeprefix("tool:").rstrip("@")
        for k in server.mcp._local_provider._components
        if k.startswith("tool:")
    }


@pytest.fixture(scope="module")
def plant_server() -> AgentsServer:
    return AgentsServer(plant_only=True)


PLANT_TOOL_NAMES = {
    "search_signals",
    "search_pulses",
    "get_signal_load_recipe",
    "signal_analytics",
    "fetch_content",
    "get_facility_coverage",
}


def test_plant_only_publishes_exactly_the_plant_tools(plant_server):
    assert _tool_names(plant_server) == PLANT_TOOL_NAMES


def test_plant_only_implies_read_only(plant_server):
    assert plant_server.read_only is True


@pytest.mark.parametrize(
    "write_tool",
    ["repl", "add_to_graph", "update_facility_config", "edit_standard_name"],
)
def test_no_write_tool_is_listed(plant_server, write_tool):
    assert write_tool not in _tool_names(plant_server)


def test_control_a_read_write_server_does_list_a_write_tool():
    """The listing check can fail: with read-only mode off, repl is published."""
    server = AgentsServer(read_only=False, dd_only=False, include_standard_names=False)
    assert "repl" in _tool_names(server)

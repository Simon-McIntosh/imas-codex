"""The collection guard refuses a graph writer that is not routed away from production.

Each case builds a synthetic collection item over a planted function in this
module and hands it to the guard, so the hook is exercised through the same
attributes it reads at collection time. The planted functions are never
collected or called: only their source is inspected, which is what the guard
does to a real test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.graph import conftest as graph_conftest

_THIS_FILE = Path(__file__).resolve()


class _Item:
    """The attributes ``pytest_collection_modifyitems`` reads from an item."""

    def __init__(self, obj, *, fixturenames=(), marked=True):
        self.obj = obj
        self.fixturenames = list(fixturenames)
        self.fspath = _THIS_FILE
        self.nodeid = f"{__name__}::{obj.__name__}"
        self._marked = marked

    def get_closest_marker(self, name):
        return object() if name == "graph" and self._marked else None


def _refuse(items) -> None:
    """Run the guard over *items*, raising ``UsageError`` on a refusal."""
    graph_conftest.pytest_collection_modifyitems(None, items)


# ---------------------------------------------------------------------------
# Planted functions — read by the guard, never executed
# ---------------------------------------------------------------------------


def _planted_bare_client_writer():
    from imas_codex.graph import GraphClient

    with GraphClient() as client:
        client.query("CREATE (:PlantedProbe {id: 'direct'})")


def _planted_delegating_writer():
    from imas_codex.graph import GraphClient

    with GraphClient() as client:
        _planted_seed_nodes(client)


def _planted_seed_nodes(client):
    client.query("MERGE (u:Unit {id: 'W'})")


def _planted_disposable_writer(disposable_graph):
    from imas_codex.graph import GraphClient

    with GraphClient() as client:
        client.query("CREATE (:PlantedProbe {id: 'routed'})")


def _planted_reader():
    from imas_codex.graph import GraphClient

    with GraphClient() as client:
        client.query("MATCH (p:PlantedProbe) RETURN p.id AS id")


def _planted_reader_with_prose():
    """A reader whose documentation happens to SET out what it measures."""
    from imas_codex.graph import GraphClient

    with GraphClient() as client:
        client.query("MATCH (p:PlantedProbe) RETURN count(p) AS n")


# ---------------------------------------------------------------------------
# The guard's decisions
# ---------------------------------------------------------------------------


def test_refuses_graph_writer_holding_a_bare_client():
    with pytest.raises(pytest.UsageError, match="_planted_bare_client_writer"):
        _refuse([_Item(_planted_bare_client_writer)])


def test_refuses_writer_that_delegates_to_a_same_file_helper():
    """The write is one call away, in a function defined beside the test."""
    with pytest.raises(pytest.UsageError, match="_planted_delegating_writer"):
        _refuse([_Item(_planted_delegating_writer)])


def test_passes_writer_routed_to_the_disposable_graph():
    _refuse([_Item(_planted_disposable_writer, fixturenames=["disposable_graph"])])


def test_passes_reader_holding_a_bare_client():
    _refuse([_Item(_planted_reader)])


def test_passes_reader_whose_text_mentions_a_write_keyword():
    """Only a Cypher argument is inspected, not every string in the test."""
    _refuse([_Item(_planted_reader_with_prose)])


def test_ignores_tests_outside_the_graph_package():
    """A writer in another package is that package's business, not this guard."""
    item = _Item(_planted_bare_client_writer)
    item.fspath = Path("/nonexistent/elsewhere/test_other.py")
    _refuse([item])


def test_ignores_a_test_that_is_not_graph_marked():
    _refuse([_Item(_planted_bare_client_writer, marked=False)])


def test_names_every_offender_in_one_refusal():
    with pytest.raises(pytest.UsageError) as excinfo:
        _refuse(
            [
                _Item(_planted_reader),
                _Item(_planted_bare_client_writer),
                _Item(_planted_delegating_writer),
            ]
        )
    message = str(excinfo.value)
    assert "_planted_bare_client_writer" in message
    assert "_planted_delegating_writer" in message
    assert "_planted_reader" not in message

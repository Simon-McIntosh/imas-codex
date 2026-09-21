"""Guards over the aperture of the report-only review write suppression.

``sn review --report-only`` must emit its report while persisting nothing, so
the suppression has to sit on the one boundary every write-capable
``GraphClient`` method routes through rather than on a single convenience
method. These tests enumerate the boundary-reaching methods by introspection
and prove, per method, that the statements it would deliver are dropped.
"""

from __future__ import annotations

import contextlib
import inspect
import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from imas_codex.cli.sn import _suppress_review_graph_writes
from imas_codex.graph.client import GraphClient

#: The one route from the client to a Neo4j session. Every method that can
#: deliver a statement to the graph passes through it, so it is the boundary
#: the suppression must occupy.
_WRITE_BOUNDARY = "session"

#: Deliberately independent of the guard's own classifier: a statement is a
#: mutation here if it names a mutating clause as a bare uppercase word. The
#: two detectors share no code path, so a classifier that stops recognising a
#: statement cannot make this oracle agree with it.
_MUTATION_IN_STATEMENT = re.compile(r"\b(?:CREATE|MERGE|SET|DELETE|REMOVE|DROP)\b")


def _method_source(method: Any) -> str:
    try:
        return inspect.getsource(method)
    except (OSError, TypeError):  # pragma: no cover - builtins carry no source
        return ""


def _self_calls(source: str) -> set[str]:
    return set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", source))


def _write_capable_methods() -> list[str]:
    """Enumerate, by introspection, the methods that can reach the boundary.

    A method is write-capable when it maintains a call path to
    ``GraphClient.session`` — directly, or through other ``GraphClient``
    methods. Property accessors and dataclass helpers that cannot deliver a
    statement are absent; every writer is present, which is the set the
    suppression has to cover.

    The set is derived from the class as it exists rather than from a list
    maintained beside it, so a method added later is enumerated the moment it
    reaches the boundary.
    """
    calls = {
        name: _self_calls(_method_source(method))
        for name, method in inspect.getmembers(GraphClient, inspect.isfunction)
        if not name.startswith("__")
    }
    reaches = {_WRITE_BOUNDARY}
    changed = True
    while changed:
        changed = False
        for name, callees in calls.items():
            if name not in reaches and callees & reaches:
                reaches.add(name)
                changed = True
    return sorted(reaches - {_WRITE_BOUNDARY})


class _RecordingSession:
    """Stand-in for a Neo4j session that records every statement it is sent."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def run(self, cypher: str, *args: Any, **params: Any) -> list[dict[str, Any]]:
        self.statements.append(cypher)
        return []

    def close(self) -> None:
        pass

    def begin_transaction(self, *args: Any, **kwargs: Any) -> _RecordingSession:
        return self

    def commit(self) -> None:
        pass


class _RecordingDriver:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    def session(self, *args: Any, **kwargs: Any) -> _RecordingSession:
        return self._session

    def close(self) -> None:
        pass


def _probe_client() -> tuple[GraphClient, _RecordingSession]:
    """Build a client whose driver records statements instead of sending them.

    Nothing here reaches Neo4j: the driver is a recorder, so a statement that
    arrives is a statement the guard let through.
    """
    session = _RecordingSession()
    client = GraphClient.__new__(GraphClient)
    client._driver = _RecordingDriver(session)
    client._schema = MagicMock()
    return client, session


_SAMPLE_ARGUMENTS: dict[str, Any] = {
    "label": "Probe",
    "node_id": "probe:1",
    "props": {"id": "probe:1"},
    "items": [{"id": "probe:1"}],
    "batch_size": 1,
    "create_relationships": False,
    "from_label": "Probe",
    "from_id": "probe:1",
    "to_label": "Probe",
    "to_id": "probe:2",
    "rel_type": "LINKS",
    "facility_id": "probe",
    "facility_ids": ["probe"],
    "name": "Probe",
    "path": "dd/probe",
    "ids": "dd",
    "hostname": "probe",
    "cypher": "MATCH (n) RETURN n",
}


def _invoke(client: GraphClient, method_name: str) -> None:
    """Call one client method with synthesized arguments, ignoring its outcome.

    A method that needs the schema or a live node may raise; that is not the
    subject of these tests. What matters is whether a mutating statement
    reached the driver before it did.
    """
    if method_name == "session":
        with client.session() as session:
            session.run("MERGE (n:Probe {id: 'probe:1'})")
        return
    method = getattr(client, method_name)
    kwargs: dict[str, Any] = {}
    for name, parameter in inspect.signature(method).parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        if name in _SAMPLE_ARGUMENTS:
            kwargs[name] = _SAMPLE_ARGUMENTS[name]
        elif parameter.default is inspect.Parameter.empty:
            kwargs[name] = "probe"
    method(**kwargs)


def _mutations(session: _RecordingSession) -> list[str]:
    return [s for s in session.statements if _MUTATION_IN_STATEMENT.search(s)]


def test_enumerated_methods_include_every_writer() -> None:
    """The enumeration sees the writers, so an empty result cannot read as coverage."""
    enumerated = _write_capable_methods()
    assert {"create_node", "create_nodes", "create_relationship", "query"} <= set(
        enumerated
    )
    assert len(enumerated) >= 20


def test_guard_occupies_the_session_boundary() -> None:
    """The guard replaces the boundary method itself, not a convenience on it."""
    original = GraphClient.session
    with _suppress_review_graph_writes() as suppressed:
        assert GraphClient.session is not original
        assert suppressed == []
    assert GraphClient.session is original


@pytest.mark.parametrize("method_name", _write_capable_methods())
def test_no_write_capable_method_reaches_the_graph(method_name: str) -> None:
    """Every enumerated method's mutations are dropped, and counted as dropped.

    The unguarded half is the control: it shows the recorder receives the
    statements this method delivers, so the guarded half's empty result is a
    refusal rather than an instrument that never looked. It doubles as the
    per-method count, so a method whose control delivers nothing is visible in
    the assertion rather than passing silently. A method that would deliver
    nothing legitimately (a read helper) satisfies it with zero on both sides.
    """
    control_client, control_session = _probe_client()
    with contextlib.suppress(Exception):
        _invoke(control_client, method_name)
    would_write = _mutations(control_session)

    guarded_client, guarded_session = _probe_client()
    with _suppress_review_graph_writes() as suppressed:
        with contextlib.suppress(Exception):
            _invoke(guarded_client, method_name)

    assert _mutations(guarded_session) == []
    assert len(suppressed) == len(would_write)


def test_review_path_writes_of_each_shape_are_refused() -> None:
    """A write of each shape the review path uses is dropped and receipted.

    The three routes are the ones a narrower aperture misses: a Cypher mutation
    through ``query``, a node write, and an edge write. Each is delivered
    through ``GraphClient`` itself, and each must be counted in the receipt the
    guard hands back, so suppression is observed rather than inferred from an
    empty session.
    """
    client, session = _probe_client()
    with _suppress_review_graph_writes() as suppressed:
        client.query("MERGE (sn:StandardName {id: $id}) SET sn.score = 1", id="x")
        client.create_node("StandardName", "x", {"score": 1})
        client.create_relationship("StandardName", "x", "Unit", "u", "HAS_UNIT")

    assert session.statements == []
    assert suppressed == ["MERGE", "MERGE", "MERGE"]


def test_suppressed_write_returns_a_consumable_result() -> None:
    """A dropped statement still yields a result its caller can read.

    Suppression removes the effect of a mutation, not the value of the call:
    the in-tree consumers read a dropped statement's result rather than
    discarding it. The client's count helpers read ``single()``
    (``GraphClient.drop_all`` and ``GraphClient.get_stats``), and the DD
    resolution port writes consume the result of a statement run inside a
    transaction. Both reads have to work with no rows, so a dropped write
    reports that nothing was written instead of raising on the shape of its
    result.

    ``single()`` returning ``None`` and the iteration being empty are asserted
    on the same object, so an implementation returning a non-empty result
    cannot satisfy both.
    """
    client, session = _probe_client()

    with _suppress_review_graph_writes():
        with client.session() as opened:
            statement = opened.run("MERGE (n:Probe {id: 'probe:1'})")
            transaction = opened.begin_transaction()
            tx_statement = transaction.run("MERGE (n:Probe {id: 'probe:2'})")

    assert session.statements == []
    assert statement.single() is None
    assert list(statement) == []
    statement.consume()
    assert tx_statement.single() is None
    assert list(tx_statement) == []
    tx_statement.consume()

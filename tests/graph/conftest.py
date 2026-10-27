"""Fixtures for graph quality tests.

All tests in this package require a live Neo4j connection. If Neo4j
is not reachable, tests are skipped automatically via the top-level
conftest.py pytest_collection_modifyitems hook (visible as "skipped").
"""

from __future__ import annotations

import ast
import inspect
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from imas_codex.graph.schema import GraphSchema, get_schema

# Apply graph marker to all tests in this directory
pytestmark = pytest.mark.graph

_GRAPH_PACKAGE = Path(__file__).resolve().parent


# =============================================================================
# Schema helpers - derive configuration from LinkML schemas
# =============================================================================


def _get_schemas_dir() -> Path:
    """Get the schemas directory."""
    return Path(__file__).parent.parent.parent / "imas_codex" / "schemas"


def _get_combined_embeddable_labels() -> list[str]:
    """Get all embeddable labels from all schemas (facility + imas_dd)."""
    schemas_dir = _get_schemas_dir()
    labels = set()
    for schema_name in ["facility.yaml", "imas_dd.yaml"]:
        gs = GraphSchema(schemas_dir / schema_name)
        labels |= set(gs.embeddable_labels)
    return sorted(labels)


def _get_combined_description_embeddable_labels() -> list[str]:
    """Get labels with description+embedding from all schemas."""
    schemas_dir = _get_schemas_dir()
    labels = set()
    for schema_name in ["facility.yaml", "imas_dd.yaml"]:
        gs = GraphSchema(schemas_dir / schema_name)
        labels |= set(gs.description_embeddable_labels)
    return sorted(labels)


# Module-level caches (computed once per test session)
_ALL_EMBEDDABLE_LABELS: list[str] | None = None
_DESCRIPTION_EMBEDDABLE_LABELS: list[str] | None = None


def get_all_embeddable_labels() -> list[str]:
    """Get all node labels with embedding slots (cached)."""
    global _ALL_EMBEDDABLE_LABELS
    if _ALL_EMBEDDABLE_LABELS is None:
        _ALL_EMBEDDABLE_LABELS = _get_combined_embeddable_labels()
    return _ALL_EMBEDDABLE_LABELS


def get_description_embeddable_labels() -> list[str]:
    """Get node labels with description+embedding (cached)."""
    global _DESCRIPTION_EMBEDDABLE_LABELS
    if _DESCRIPTION_EMBEDDABLE_LABELS is None:
        _DESCRIPTION_EMBEDDABLE_LABELS = _get_combined_description_embeddable_labels()
    return _DESCRIPTION_EMBEDDABLE_LABELS


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def graph_client():
    """Session-scoped GraphClient that skips if Neo4j is unreachable.

    Does NOT run ``initialize_schema()`` — that is heavy DDL work that
    overwhelms SSH tunnels.  Tests needing constraints should use the
    ``graph_with_schema`` fixture instead.
    """
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    yield client
    client.close()


@pytest.fixture(scope="session")
def graph_with_schema(graph_client):
    """GraphClient with schema constraints initialized.

    Calls ``initialize_schema()`` once per session — idempotent but
    does many DDL round-trips.  Only used by constraint tests.
    Verifies connectivity after the DDL burst in case the tunnel dropped.
    """
    from imas_codex.graph.client import GraphClient

    try:
        graph_client.initialize_schema()
    except Exception:
        pass  # Non-fatal — constraint tests will report specifics

    # Verify the connection survived the DDL burst
    try:
        graph_client.query("RETURN 1 AS ping")
    except Exception:
        # Connection pool poisoned — recreate
        try:
            graph_client.close()
        except Exception:
            pass
        try:
            new_client = GraphClient()
            new_client.get_stats()
            yield new_client
            new_client.close()
            return
        except Exception as e:
            pytest.skip(f"Neo4j connection lost during schema init: {e}")

    yield graph_client


@pytest.fixture(scope="session")
def schema() -> GraphSchema:
    """The global GraphSchema instance."""
    return get_schema()


@pytest.fixture(scope="session")
def label_counts(graph_client) -> dict[str, int]:
    """Node counts per label, fetched once per session."""
    return graph_client.get_label_counts()


@pytest.fixture(scope="session")
def graph_stats(graph_client) -> dict[str, int]:
    """Total node and relationship counts."""
    return graph_client.get_stats()


@pytest.fixture(scope="session")
def graph_labels(graph_client) -> set[str]:
    """Labels actually present in the graph (excludes stale token-store labels)."""
    results = graph_client.query(
        "MATCH (n) UNWIND labels(n) AS l RETURN DISTINCT l AS label"
    )
    return {r["label"] for r in results}


@pytest.fixture(scope="session")
def graph_relationship_types(graph_client) -> set[str]:
    """Relationship types actually present in the graph (excludes stale types)."""
    results = graph_client.query(
        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationshipType"
    )
    return {r["relationshipType"] for r in results}


@pytest.fixture(scope="session")
def graph_constraints(graph_client) -> list[dict]:
    """All constraints in the graph."""
    return graph_client.query(
        "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
        "RETURN name, type, labelsOrTypes, properties"
    )


@pytest.fixture(scope="session")
def graph_indexes(graph_client) -> list[dict]:
    """All indexes in the graph."""
    return graph_client.query(
        "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state "
        "RETURN name, type, labelsOrTypes, properties, state"
    )


@pytest.fixture(scope="session")
def embedding_dimension() -> int:
    """Configured embedding dimension from pyproject.toml."""
    from imas_codex.settings import get_embedding_dimension

    return get_embedding_dimension()


# =============================================================================
# Disposable Neo4j for the graph-marked tests that write
# =============================================================================
#
# Most tests in this package read the live graph on purpose. A test that writes
# cannot: it would land its fixture rows in the project graph. Those tests get
# an empty instance of their own, started with the same machinery the
# standard_names package owns rather than a second kind of starter.
#
# The address is started once for the session, but the environment is bound per
# test and undone when that test ends. Binding it for the whole session would
# redirect every later live-graph reader to the empty instance — the writers
# would pass and the readers would silently measure nothing.

#: Fixtures that give a test the shared empty graph.
_DISPOSABLE_FIXTURES = frozenset(
    {"disposable_graph", "disposable_neo4j_uri", "disposable_neo4j_empty"}
)

#: Environment the disposable binding owns for one test.
_DISPOSABLE_ENV_KEYS = (
    "NEO4J_URI",
    "NEO4J_PASSWORD",
    "IMAS_CODEX_TEST_NEO4J_URI",
    "IMAS_CODEX_TEST_NEO4J_EPHEMERAL",
    "IMAS_CODEX_TEST_NEO4J_PASSWORD",
)


@pytest.fixture(scope="session")
def _disposable_neo4j_address() -> Iterator[str]:
    """One empty graph for this session, or the one the run advertises.

    Supplies only the address. No environment is changed here: the binding
    belongs to ``disposable_graph`` and lasts exactly one test.
    """
    from tests.standard_names.conftest import (
        _detach_store_removal,
        _start_empty_neo4j,
    )

    configured = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if configured:
        if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
            pytest.fail(
                "a disposable graph is required, but IMAS_CODEX_TEST_NEO4J_URI "
                "is set without IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1"
            )
        yield configured
        return

    try:
        directory, process, _log, bolt_port = _start_empty_neo4j()
    except Exception as exc:
        pytest.fail(f"no disposable Neo4j could start: {exc}")
        return
    try:
        yield f"bolt://127.0.0.1:{bolt_port}"
    finally:
        from imas_codex.graph import temp_neo4j

        try:
            temp_neo4j.stop_temp_neo4j(process)
        finally:
            _detach_store_removal(directory)


@pytest.fixture()
def disposable_graph(_disposable_neo4j_address, monkeypatch) -> Iterator[str]:
    """Bind one test to the shared empty graph and refuse the project graph.

    Refuses before wiping: an endpoint that turns out to be the project graph
    must be identified before anything is deleted from it. ``refuse_project_graph``
    resolves the project address with ``NEO4J_URI`` withheld, so it compares the
    disposable endpoint against production rather than against itself.
    """
    from imas_codex.graph.profiles import resolve_neo4j
    from tests.standard_names.conftest import _wipe, refuse_project_graph

    uri = _disposable_neo4j_address
    for key in _DISPOSABLE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEO4J_URI", uri)
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    monkeypatch.setenv("IMAS_CODEX_TEST_NEO4J_URI", uri)
    monkeypatch.setenv("IMAS_CODEX_TEST_NEO4J_EPHEMERAL", "1")

    refuse_project_graph(uri)

    resolved = resolve_neo4j(auto_tunnel=False).uri
    if resolved.rstrip("/") != uri.rstrip("/"):
        pytest.fail(
            f"the graph profile ignores NEO4J_URI and resolves to {resolved}: "
            f"a bare GraphClient() would reach past the disposable endpoint {uri}"
        )
    _wipe(uri)
    yield uri


# =============================================================================
# Collection guard — a graph writer must not reach the project graph
# =============================================================================
#
# A graph-marked test that builds a bare ``GraphClient()`` and writes has no
# disposable routing, so running the suite writes fixture rows into the project
# graph. That is invisible in a green run: the writes land, the assertions pass,
# and the only trace is data the project graph did not ask for. This guard
# refuses such a test at collection time, naming it, and the refusal is what
# keeps the hazard from returning.

#: Cypher that changes the graph. ``SET``, ``CREATE`` and ``MERGE`` cover the
#: fixtures these tests build; ``DELETE``/``DETACH`` and the constraint DDL cover
#: the clean-up and schema paths.
_WRITE_CYPHER = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|FOREACH|LOAD CSV)\b",
    re.IGNORECASE,
)

#: Functions whose names say they write. A test that calls one and also holds a
#: bare client is a writer.
_WRITE_CALL_PREFIXES = (
    "write_",
    "link_",
    "create_",
    "delete_",
    "upsert_",
    "merge_",
    "reconcile_",
    "persist_",
    "seed_",
    "backfill_",
    "attach_",
    "detach_",
    "accept_",
    "promote_",
    "retire_",
    "reset_",
    "repair_",
    "initialize_",
    "wipe",
)

#: Fixtures that yield a client resolved from the process environment, i.e. a
#: pool a bare ``GraphClient()`` would join.
_BARE_CLIENT_FIXTURES = frozenset({"_gc", "graph_client", "graph_with_schema"})


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


#: Methods that send Cypher. Only their string argument is inspected: scanning
#: every string in the test would read a docstring's prose (a sentence with the
#: word "set") as a write.
_CYPHER_METHODS = frozenset({"query", "run", "execute_query", "execute_write"})


def _writer_facts(source: str) -> tuple[bool, bool]:
    """Return ``(writes, bare_client)`` for one test function's source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, False

    writes = False
    bare_client = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None:
            continue
        if name.startswith(_WRITE_CALL_PREFIXES):
            writes = True
        if name in _CYPHER_METHODS and node.args:
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and _WRITE_CYPHER.search(first.value)
            ):
                writes = True
        if name == "GraphClient":
            has_uri = any(kw.arg == "uri" for kw in node.keywords) or node.args
            if not has_uri:
                bare_client = True
    return writes, bare_client


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Refuse a graph writer that is not routed to the disposable graph."""
    offenders: list[str] = []
    for item in items:
        if not item.get_closest_marker("graph"):
            continue
        if Path(str(item.fspath)).resolve().parent != _GRAPH_PACKAGE:
            continue
        fixturenames = set(item.fixturenames or ())
        if fixturenames & _DISPOSABLE_FIXTURES:
            continue
        try:
            source = inspect.getsource(item.obj)
        except (OSError, TypeError):
            continue
        writes, bare_client = _writer_facts(source)
        if not writes:
            continue
        if bare_client or (fixturenames & _BARE_CLIENT_FIXTURES):
            offenders.append(item.nodeid)

    if offenders:
        raise pytest.UsageError(
            "graph-marked tests that write must run on the disposable graph "
            "(request the 'disposable_graph' fixture): "
            + ", ".join(sorted(offenders))
        )

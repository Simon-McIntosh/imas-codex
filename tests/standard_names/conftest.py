"""Shared fixtures for standard name tests.

The ``mock_llm`` fixture replaces the real LLM call (``acall_llm_structured``)
with a deterministic, scripted response provider. Tests register expected
responses per (stage, model) and the mock returns them in registration order.

Stages map to worker pool names: ``'generate_name'``, ``'review_name'``,
``'refine_name'``, ``'generate_docs'``, ``'review_docs'``, ``'refine_docs'``.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


def pytest_collection_modifyitems(config, items):
    """Select graph items and keep their startup outside test call timeouts."""
    if config.getoption("markexpr") == "graph":
        deselected = [
            item for item in items if item.get_closest_marker("graph") is None
        ]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = [
                item for item in items if item.get_closest_marker("graph") is not None
            ]
    for item in items:
        if (
            "disposable_neo4j_uri" not in item.fixturenames
            and item.get_closest_marker("graph") is None
        ):
            continue
        existing = item.get_closest_marker("timeout")
        args = (
            existing.args
            if existing
            else (
                float(
                    item.config.getoption("timeout") or item.config.getini("timeout")
                ),
            )
        )
        kwargs = dict(existing.kwargs) if existing else {}
        kwargs["func_only"] = True
        item.add_marker(pytest.mark.timeout(*args, **kwargs), append=False)


# ---------------------------------------------------------------------------
# Structural guard — block live GraphClient connections in default-tier tests
# ---------------------------------------------------------------------------

#: Markers that indicate a test intentionally needs a live Neo4j connection.
_LIVE_GRAPH_MARKERS = frozenset(
    {"graph", "graph_mcp", "requires_graph", "fixture_only"}
)


#: Route prefixes that only ever name a test double.  A real OpenRouter route
#: is provider-qualified (``openrouter/<vendor>/<model>``), so a bare id or a
#: ``test``/``some`` vendor segment cannot collide with a production seat.
_SYNTHETIC_ROUTE_PREFIXES = ("test/", "openrouter/test/", "some/")


def clean_source_audit_graph(cls):
    """Give an export graph double an empty, internally consistent audit."""
    original_query = cls.query

    def query(self, cypher: str, **params):
        if (
            "WHERE NOT sn.status IN $terminal" in cypher
            and "RETURN count(sn) AS total" in cypher
        ):
            return [{"total": 0}]
        if cypher.strip() == (
            "MATCH (source:StandardNameSource) RETURN count(source) AS total"
        ):
            return [{"total": 0}]
        if "RETURN names, relations, changes, projections" in cypher:
            return [{"names": 0, "relations": 0, "changes": 0, "projections": 0}]
        return original_query(self, cypher, **params)

    cls.query = query
    return cls


def _is_synthetic_route(model: str) -> bool:
    """True when *model* names a test double rather than a real route."""
    return "/" not in model or model.startswith(_SYNTHETIC_ROUTE_PREFIXES)


@pytest.fixture(autouse=True)
def _bound_synthetic_model_exposure(monkeypatch):
    """Give mocked provider routes a finite lease in worker unit tests.

    Only ids that cannot name a real route are stubbed, and the membership
    test is structural rather than a catalog lookup.  Keying this off "the
    catalog does not know it" would silently capture every production seat
    newer than the bundled catalog — which is all of them — and no test would
    then exercise the pricing that bounds real spend.

    A stubbed double still gets a one-dollar ceiling rather than zero, because
    scripted responses carry nonzero synthetic cost and must charge against
    something.
    """
    from imas_codex.discovery.base.llm import _is_local_model
    from imas_codex.standard_names import budget, workers
    from imas_codex.standard_names.fanout import dispatcher
    from imas_codex.standard_names.review import pipeline

    original = budget.model_provider_exposure

    def _bounded(model, *args, **kwargs):
        if not _is_local_model(model) and _is_synthetic_route(model):
            return 1.0
        return original(model, *args, **kwargs)

    monkeypatch.setattr(workers, "model_provider_exposure", _bounded)
    monkeypatch.setattr(pipeline, "model_provider_exposure", _bounded)
    monkeypatch.setattr(dispatcher, "model_provider_exposure", _bounded)


@pytest.fixture(scope="session")
def _synthetic_dd_resolution_rows():
    """Load the deliberately minimal mock-only authority once per session."""
    from tests.standard_names.dd_resolution_test_data import (
        SYNTHETIC_GRAPH_RESOLUTION_ROWS,
    )

    return SYNTHETIC_GRAPH_RESOLUTION_ROWS


class SyntheticDDResolutionReader:
    """Serve a fixed DD-resolution authority to tests that must not reach a graph.

    The authority is a property of the DD version rather than of a graph, so a
    test that exercises Cypher against an empty instance reads the same rows the
    mock tier does instead of seeding a second copy of the real records.
    """

    def __init__(self, rows) -> None:
        self._rows = rows

    def read_resolutions(self):
        return self._rows


@pytest.fixture(autouse=True)
def _block_live_graph(request, monkeypatch, _synthetic_dd_resolution_rows):
    """Raise RuntimeError if a default-tier test attempts to open a real
    Neo4j connection.

    This guard patches ``GraphClient.__post_init__`` — the dataclass
    initialiser where the driver connection is established — directly on the
    class object.  Because the patch targets the class itself rather than a
    module-level name binding, it is effective regardless of how
    ``GraphClient`` was imported (``from ... import GraphClient``, local
    import, etc.).

    Tests marked with any live-graph marker (graph, graph_mcp,
    requires_graph, fixture_only) are exempt and receive an unpatched
    ``GraphClient``.
    """
    if any(request.node.get_closest_marker(m) for m in _LIVE_GRAPH_MARKERS):
        # Test is allowed to connect; yield without patching.
        yield
        return

    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import dd_resolutions

    monkeypatch.setattr(
        dd_resolutions,
        "dd_resolution_graph_reader",
        lambda: SyntheticDDResolutionReader(_synthetic_dd_resolution_rows),
    )

    def _blocked_post_init(self, *args, **kwargs):
        raise RuntimeError(
            f"GraphClient.__post_init__ called in a default-tier test "
            f"({request.node.nodeid!r}). "
            "Mark the test with @pytest.mark.graph (or requires_graph) if "
            "it intentionally needs a live Neo4j connection, or add a local "
            "mock/patch so it does not reach GraphClient at all."
        )

    with patch.object(GraphClient, "__post_init__", _blocked_post_init):
        yield


@pytest.fixture(scope="session", autouse=True)
def _cache_grammar_context():
    """Memoize ``get_grammar_context()`` for the whole test session.

    ISN's ``get_grammar_context()`` rebuilds and semantic-validates the full
    published catalog on every call (tens of seconds), and the
    ``GrammarSegments`` validator invokes it on every construction. Without a
    cache the suite is effectively unrunnable. The context is a pure function
    of the on-disk vocabularies, so a single build reused for the session is
    equivalent. Patches both the source symbol and the top-level re-export so
    the many local ``from imas_standard_names import get_grammar_context``
    call sites resolve to the cached value.
    """
    import warnings
    from contextlib import ExitStack

    import imas_standard_names as _isn
    import imas_standard_names.grammar.context as _ctx

    try:
        cached = _ctx.get_grammar_context()
    except Exception as exc:
        # A broken ISN pin (e.g. a grammar/catalog internal inconsistency that
        # raises ParseError while building the context) must not error every
        # test in this package at setup. Skip memoization and leave the real
        # function in place: pure-unit tests that never touch the grammar
        # context still run, and tests that genuinely need it fail (fast) on
        # their own rather than taking the whole package down at collection.
        warnings.warn(
            f"grammar context unavailable ({type(exc).__name__}: {exc}); "
            "skipping session cache — grammar-dependent tests may fail "
            "individually until the ISN pin is fixed",
            stacklevel=2,
        )
        yield
        return

    def _cached() -> Any:
        return cached

    with ExitStack() as stack:
        stack.enter_context(patch.object(_ctx, "get_grammar_context", _cached))
        if hasattr(_isn, "get_grammar_context"):
            stack.enter_context(patch.object(_isn, "get_grammar_context", _cached))
        yield


@pytest.fixture()
def sample_standard_names() -> list[dict]:
    """Sample standard name dicts for write_standard_names testing."""
    return [
        {
            "id": "electron_temperature",
            "source_types": ["dd"],
            "source_id": "core_profiles/profiles_1d/electrons/temperature",
            "physical_base": "temperature",
            "subject": "electron",
            "description": "Electron temperature profile",
            "documentation": "The electron temperature $T_e$ is measured by Thomson scattering.",
            "kind": "scalar",
            "links": ["ion_temperature", "electron_density"],
            "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
            "validity_domain": "core plasma",
            "constraints": ["T_e > 0"],
            "unit": "eV",
            "physics_domain": "core_profiles",
            "model": "test/model",
            "name_stage": "drafted",
            "generated_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "plasma_current",
            "source_types": ["signals"],
            "source_id": "tcv:ip/measured",
            "physical_base": "current",
            "description": "Plasma current",
            "unit": "A",
            "physics_domain": "magnetics",
            "kind": "scalar",
            "model": "test/model",
            "name_stage": "drafted",
            "generated_at": "2024-01-01T00:00:00Z",
        },
    ]


@pytest.fixture()
def mock_graph_client():
    """A mock GraphClient that records query calls."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.query = MagicMock(return_value=[])
    return client


@pytest.fixture()
def graph_client():
    """Real GraphClient; skips if Neo4j is unavailable.

    Used by graph-marked integration tests (e.g. test_physics_domain_multi).
    """
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    yield client
    client.close()


# ---------------------------------------------------------------------------
# Disposable Neo4j — the one convention for tests that run real Cypher
# ---------------------------------------------------------------------------
#
# A test that exercises real Cypher needs a graph of its own, and every such
# test in this package gets it the same way: one empty instance, named by both
# ``NEO4J_URI`` (the profile every ``GraphClient()`` resolves through) and
# ``IMAS_CODEX_TEST_NEO4J_URI`` (what the disposable-graph tests advertise), and
# a loud refusal when an endpoint names the project graph.
#
# The refusal resolves the project graph's address with ``NEO4J_URI`` withheld.
# Reading it back with the override in place compares the disposable endpoint
# against itself, which passes for every endpoint — including the project
# graph's own — and so protects nothing.

#: Environment this fixture owns for a test's duration.
_DISPOSABLE_ENV = (
    "NEO4J_URI",
    "NEO4J_PASSWORD",
    "IMAS_CODEX_TEST_NEO4J_URI",
    "IMAS_CODEX_TEST_NEO4J_EPHEMERAL",
    "IMAS_CODEX_TEST_NEO4J_LOG",
)


def project_graph_uris() -> set[str]:
    """Every address this checkout reaches the project graph at.

    Two spellings, because two hosts reach the same graph by different routes:
    the loopback port, which is what a login node's profile names and what an
    SSH tunnel would front, and the address service discovery reports for the
    scheduled instance, which is what a compute node actually connects to.
    Naming only the loopback refuses nothing on a cluster whose graph runs on a
    service node, and the port alone says nothing about which graph listens on
    it, so both are collected.

    Neither is resolved by opening a tunnel: discovery answers from the
    scheduler, so reading an address stays a read.
    """
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.remote.locations import resolve_service_url

    withheld = os.environ.pop("NEO4J_URI", None)
    try:
        profile = resolve_neo4j(auto_tunnel=False)
        uris = {f"bolt://localhost:{profile.bolt_port}"}
        discovered = resolve_service_url(
            profile.location, profile.bolt_port, protocol="bolt"
        )
        if discovered:
            uris.add(discovered)
        return {uri.rstrip("/") for uri in uris}
    finally:
        if withheld is not None:
            os.environ["NEO4J_URI"] = withheld


def refuse_project_graph(uri: str) -> None:
    """Fail the run when *uri* names the project graph.

    This is the protection the disposable fixtures exist for, so it fails
    rather than skips: an endpoint that cannot be told apart from production is
    a broken test setup, not a test that is not applicable here.
    """
    project_uris = project_graph_uris()
    if uri.rstrip("/") in project_uris:
        pytest.fail(
            f"refusing to run against the project graph URI ({uri}): "
            "these tests need a disposable graph; the project graph is at "
            f"{', '.join(sorted(project_uris))}"
        )


def _free_port_pair() -> tuple[int, int]:
    """Two distinct free loopback ports.

    Both sockets are held open until both are bound, so the two ports cannot
    be the same one -- binding them in turn would let the first port be handed
    out again for the second.
    """
    sockets = [socket.socket(), socket.socket()]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return int(sockets[0].getsockname()[1]), int(sockets[1].getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()


_DISPOSABLE_STARTUP_TIMEOUT_SECONDS = 240.0


def _start_empty_neo4j():
    """Start one empty Neo4j through the module that owns the lifecycle.

    Returns its scratch directory, process, log path and bolt port.  The
    directory is removed if the start fails: a failure here must leave
    neither a running instance nor a tree behind.
    """
    from imas_codex.graph import temp_neo4j

    scratch = os.environ.get("TMPDIR") or None
    directory = tempfile.TemporaryDirectory(prefix="disposable-neo4j-", dir=scratch)
    root = Path(directory.name)
    bolt_port, http_port = _free_port_pair()

    def _startup_expired(_signum, _frame):
        raise TimeoutError(
            "disposable Neo4j startup exceeded "
            f"{_DISPOSABLE_STARTUP_TIMEOUT_SECONDS:g} seconds"
        )

    previous_handler = signal.signal(signal.SIGALRM, _startup_expired)
    previous_timer = signal.setitimer(
        signal.ITIMER_REAL, _DISPOSABLE_STARTUP_TIMEOUT_SECONDS
    )
    try:
        process, log = temp_neo4j.start_temp_neo4j(
            root, bolt_port, http_port, load_dump=False
        )
    except BaseException:
        directory.cleanup()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
    return directory, process, log, bolt_port


def neo4j_test_password() -> str:
    """The password of the disposable graph, empty for an auth-free instance."""
    return os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")


def _auth() -> tuple[str, str] | None:
    password = neo4j_test_password()
    return ("neo4j", password) if password else None


def _wipe(uri: str) -> None:
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=_auth()) as driver:
        driver.verify_connectivity()
        driver.execute_query("MATCH (node) DETACH DELETE node")
        records, _, _ = driver.execute_query("MATCH (node) RETURN count(node) AS n")
        assert records[0]["n"] == 0


#: How long the session may wait for the detached remover to finish. It bounds
#: the guarantee in both directions: a removal that fails inside the wait fails
#: the session's teardown, and one still running when the wait ends is left
#: detached with its log beside the renamed tree under the job's own TMPDIR.
_STORE_REMOVAL_BUDGET_SECONDS = 2.0

#: The detached remover.  No ``ignore_errors``: a removal that fails keeps its
#: traceback in the log beside the trash and fails the session's teardown,
#: rather than passing off an error as a completed removal.
_STORE_REMOVER = "import shutil, sys; shutil.rmtree(sys.argv[1])"


def _detach_store_removal(directory: tempfile.TemporaryDirectory) -> str:
    """Remove the store directory without making the session wait for it.

    Neo4j's store is thousands of small files -- one per property, label and
    segment -- so deleting it costs one metadata operation per file.  On a
    shared filesystem every unlink is a network round trip, which is why a
    recursive delete of a temporary store takes tens of seconds.  That cost
    belongs to no test: the tree is renamed away in one step (the store path is
    gone at once) and a detached process removes it afterwards.

    Returns one line naming what happened, for the gate log.
    """
    root = Path(directory.name)
    trash = root.with_name(f"{root.name}-removing")
    started = time.monotonic()
    try:
        os.replace(root, trash)
    except OSError:
        # A tree that cannot be renamed still has to go; there is nothing to
        # detach, so the removal stays in the fixture's own teardown.
        directory.cleanup()
        return f"store {root} removed in place in {time.monotonic() - started:.2f}s"
    directory.cleanup()  # detach the finalizer: the tree no longer exists
    renamed = time.monotonic() - started
    remover_log = Path(f"{trash}.removal.log")
    with open(remover_log, "w") as handle:
        remover = subprocess.Popen(
            [sys.executable, "-I", "-c", _STORE_REMOVER, str(trash)],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    waited = 0.0
    while remover.poll() is None and waited < _STORE_REMOVAL_BUDGET_SECONDS:
        time.sleep(0.05)
        waited = time.monotonic() - started - renamed
    if remover.poll() is None:
        return (
            f"store {root} handed to pid {remover.pid}, detached "
            f"({renamed:.2f}s rename, {waited:.2f}s waited)"
        )
    if remover.returncode != 0:
        # A removal that failed is not a removal. Its log holds the traceback
        # and the tree is still on disk, so both stay and the session's
        # teardown reports the failure instead of claiming a clean exit.
        raise RuntimeError(
            f"removing the disposable Neo4j store {trash} failed with exit "
            f"code {remover.returncode}; the tree is left in place and its "
            f"log is {remover_log}"
        )
    remover_log.unlink(missing_ok=True)
    return (
        f"store {root} removed in {renamed + waited:.2f}s, "
        f"detached to pid {remover.pid}"
    )


@pytest.fixture(scope="session")
def disposable_neo4j_uri() -> Iterator[str]:
    """The one empty graph this package's real-Cypher tests share.

    Attaches to ``IMAS_CODEX_TEST_NEO4J_URI`` when a run supplies one (which
    must carry ``IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1``), and otherwise starts an
    empty instance of its own.  Either way the endpoint is compared against the
    project graph's resolved address and the run fails if they are the same, so
    no test here can write to the shared live graph.

    Autouse fixtures below point graph clients at this endpoint for graph-marked
    tests and explicit users. Other tests in the worker retain their own URI.
    """
    configured = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    directory = process = log = None
    if configured:
        if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
            pytest.fail(
                "a disposable graph is required, but IMAS_CODEX_TEST_NEO4J_URI "
                "is set without IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1"
            )
        uri = configured
    else:
        try:
            directory, process, log, bolt_port = _start_empty_neo4j()
        except Exception as exc:
            pytest.fail(f"no disposable Neo4j could start: {exc}")
        uri = f"bolt://127.0.0.1:{bolt_port}"

    previous = {key: os.environ.get(key) for key in _DISPOSABLE_ENV}
    os.environ.update(
        {
            "NEO4J_URI": uri,
            "NEO4J_PASSWORD": "",
            "IMAS_CODEX_TEST_NEO4J_URI": uri,
            "IMAS_CODEX_TEST_NEO4J_EPHEMERAL": "1",
        }
    )
    try:
        refuse_project_graph(uri)
        from imas_codex.graph.profiles import resolve_neo4j

        # The profile obeys NEO4J_URI, so this compares the override with
        # itself and is no evidence about the project graph -- the refusal
        # above is what protects production. What it does prove is that the
        # override is still honoured, which is the whole wiring: a test that
        # builds a bare GraphClient() must land on the instance it was given.
        resolved = resolve_neo4j(auto_tunnel=False).uri
        if resolved.rstrip("/") != uri.rstrip("/"):
            pytest.fail(
                f"the graph profile ignores NEO4J_URI and resolves to "
                f"{resolved}: a bare GraphClient() would reach past the "
                f"disposable endpoint {uri}"
            )
        _wipe(uri)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield uri
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if process is not None:
            from imas_codex.graph import temp_neo4j

            destination = previous.get("IMAS_CODEX_TEST_NEO4J_LOG")
            if destination and log is not None:
                import shutil

                shutil.copyfile(log, destination)
            stopped = time.monotonic()
            store_removal = "store removal not attempted"
            try:
                temp_neo4j.stop_temp_neo4j(process)
            finally:
                store_removal = _detach_store_removal(directory)
            blocking = time.monotonic() - stopped
            print(
                f"disposable Neo4j teardown: database stopped and "
                f"{store_removal}; {blocking:.2f}s blocking; store path "
                f"{directory.name} absent after session: "
                f"{not Path(directory.name).exists()}",
                flush=True,
            )


@pytest.fixture(scope="module", autouse=True)
def _module_disposable_graph_environment(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Bind graph modules before their module-scoped graph clients are built."""
    graph_items = any(
        item.get_closest_marker("graph") is not None and item.path == request.node.path
        for item in request.session.items
    )
    if not graph_items:
        yield
        return

    uri = request.getfixturevalue("disposable_neo4j_uri")
    previous = {key: os.environ.get(key) for key in _DISPOSABLE_ENV}
    os.environ.update(
        {
            "NEO4J_URI": uri,
            "NEO4J_PASSWORD": "",
            "IMAS_CODEX_TEST_NEO4J_URI": uri,
            "IMAS_CODEX_TEST_NEO4J_EPHEMERAL": "1",
        }
    )
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _disposable_graph_environment(request: pytest.FixtureRequest) -> Iterator[None]:
    """Bind graph-marked tests and explicit fixture users to the empty graph."""
    if (
        "disposable_neo4j_uri" not in request.fixturenames
        and request.node.get_closest_marker("graph") is None
    ):
        yield
        return

    uri = request.getfixturevalue("disposable_neo4j_uri")
    previous = {key: os.environ.get(key) for key in _DISPOSABLE_ENV}
    os.environ.update(
        {
            "NEO4J_URI": uri,
            "NEO4J_PASSWORD": "",
            "IMAS_CODEX_TEST_NEO4J_URI": uri,
            "IMAS_CODEX_TEST_NEO4J_EPHEMERAL": "1",
        }
    )
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture()
def disposable_neo4j_empty(disposable_neo4j_uri: str) -> str:
    """The shared endpoint, holding no nodes when the test starts."""
    _wipe(disposable_neo4j_uri)
    return disposable_neo4j_uri


@pytest.fixture()
def disposable_neo4j(disposable_neo4j_uri: str) -> tuple[str, str]:
    """The shared endpoint in the URI/password shape used by graph clients."""
    return disposable_neo4j_uri, neo4j_test_password()


# ---------------------------------------------------------------------------
# MockLLM — scripted LLM response fixture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ScriptedResponse:
    stage: str
    model: str | None  # None = match any model
    response: Any  # parsed pydantic instance OR dict that will be coerced
    cost: float = 0.0
    tokens: dict[str, int] | None = None


class MockLLM:
    """Scripted LLM mock for SN pipeline tests.

    Usage::

        def test_generate_then_review(mock_llm):
            mock_llm.add_response('generate_name', response=GeneratedName(name=...))
            mock_llm.add_response('review_name',   response=ReviewResult(score=0.9))
            await some_pipeline_step(...)
            assert mock_llm.calls_for('generate_name') == 1

    Fallthrough behaviour: if no scripted response matches a call, the mock
    raises ``RuntimeError`` so tests fail loudly rather than silently using
    the real LLM.
    """

    def __init__(self) -> None:
        self._queue: list[_ScriptedResponse] = []
        self._calls: list[dict[str, Any]] = []

    def add_response(
        self,
        stage: str,
        *,
        response: Any,
        model: str | None = None,
        cost: float = 0.0,
        tokens: dict[str, int] | None = None,
    ) -> None:
        """Register a scripted response. FIFO per stage."""
        self._queue.append(
            _ScriptedResponse(
                stage=stage,
                model=model,
                response=response,
                cost=cost,
                tokens=tokens or {"input": 0, "output": 0, "cached": 0},
            )
        )

    def calls_for(self, stage: str) -> int:
        """Return number of calls dispatched for *stage*."""
        return sum(1 for c in self._calls if c["stage"] == stage)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded call records."""
        return list(self._calls)

    async def _dispatch(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type,
        service: str | None = None,
        stage: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, float, dict[str, int]]:
        """Replacement coroutine for ``acall_llm_structured``."""
        # Stage priority: explicit kwarg > inferred from messages > 'unknown'
        resolved_stage = stage or _infer_stage(messages, service) or "unknown"
        self._calls.append(
            {
                "stage": resolved_stage,
                "model": model,
                "service": service,
                "messages": messages,
                "response_model": response_model.__name__,
            }
        )

        for i, scripted in enumerate(self._queue):
            if scripted.stage != resolved_stage:
                continue
            if scripted.model is not None and scripted.model != model:
                continue
            self._queue.pop(i)
            payload = scripted.response
            if isinstance(payload, dict):
                payload = response_model(**payload)
            return payload, scripted.cost, scripted.tokens

        raise RuntimeError(
            f"MockLLM: no scripted response for stage={resolved_stage!r} "
            f"model={model!r}. Registered: "
            f"{[(s.stage, s.model) for s in self._queue]}"
        )


def _infer_stage(
    messages: list[dict[str, str]],
    service: str | None,  # noqa: ARG001
) -> str | None:
    """Best-effort stage inference from system-prompt keywords.

    Workers that need precision should pass ``stage=`` explicitly; this
    function is a fallback for callers that don't.
    """
    if not messages:
        return None
    sys_content = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"), ""
    )
    s = sys_content.lower()
    # Check "refine" before "review" with specific phrase matching,
    # because review prompts mention "refinement" and refine prompts
    # mention "reviewer feedback" — single-keyword matching is ambiguous.
    if "you are refining" in s:
        if "documentation" in s or "description" in s or "docs" in s:
            return "refine_docs"
        return "refine_name"
    if "review" in s or "critic" in s:
        # Disambiguate review_name vs review_docs.  The name-axis review
        # prompt says "name-only mode" / "name-axis review"; the docs-axis
        # prompt says "evaluating…documentation" / "documentation text quality".
        # A naïve "documentation" check false-matches on the name prompt
        # because it mentions "documentation is filled in by a later pass".
        if "name-only mode" in s or "name-axis" in s:
            return "review_name"
        if (
            "documentation text quality" in s
            or "evaluating" in s
            and "documentation" in s
        ):
            return "review_docs"
        # Fallback: presence of "documentation quality" signals docs axis
        if "documentation quality" in s:
            return "review_docs"
        return "review_name"
    if "documentation" in s or "enrich" in s:
        return "generate_docs"
    if "compose" in s or "standard name" in s:
        return "generate_name"
    return None


@pytest.fixture()
def mock_llm():
    """Patch ``acall_llm_structured`` globally and yield a :class:`MockLLM`.

    All SN workers import ``acall_llm_structured`` lazily (function-local
    ``from imas_codex.discovery.base.llm import acall_llm_structured``), so
    patching the canonical module attribute is sufficient to intercept every
    call.
    """
    mock = MockLLM()

    # Patch at every known import site.  Function-local imports all resolve
    # to the same object in imas_codex.discovery.base.llm, so one target
    # covers the standard_names workers.  Additional sites (discovery, graph)
    # are included so the fixture is safe to use in cross-module tests.
    targets = [
        # Canonical definition — covers all function-local SN imports:
        #   workers.py, enrich_workers.py, benchmark.py, review/pipeline.py
        "imas_codex.discovery.base.llm.acall_llm_structured",
        # Re-export in discovery.base.__init__ (used by some discovery tests)
        "imas_codex.discovery.base.acall_llm_structured",
    ]

    patches = []
    for target in targets:
        try:
            p = patch(target, side_effect=mock._dispatch)
            patches.append(p)
            p.start()
        except (ImportError, AttributeError):
            # Not every call site exists in every build — patch what is there.
            pass

    try:
        yield mock
    finally:
        for p in patches:
            p.stop()

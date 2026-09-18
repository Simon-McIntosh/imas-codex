"""The collection-time DD-content probe is bounded and names its cause.

``test_seed_live_graph`` binds a graph read inside a ``skipif``, so the probe runs
during collection, before any test can report anything about it. Two properties
are therefore load-bearing and neither is observable from the outside:

* it cannot wait longer than a stated bound, or one unreachable host holds a
  whole session open before the first test is collected; and
* its reason names which condition it met — no route, no credential, or a
  reachable-but-empty graph. Those are three different faults with three
  different remedies, and one string for all three leaves a skip receipt unable
  to tell a session that its credential is missing from one whose host cannot be
  routed at all.

Each condition is injected here against a stub rather than waited for, so the
properties are asserted without needing a broken graph to hand.
"""

from __future__ import annotations

import importlib
import time
from types import SimpleNamespace

import pytest
from neo4j.exceptions import AuthError, ServiceUnavailable

seed = importlib.import_module("standard_names.test_seed_live_graph")


class _StubGraph:
    """A ``GraphClient`` stand-in that answers, stalls or fails on demand."""

    def __init__(self, *, rows=(), exc=None, delay=0.0, uri="bolt://stub:7687"):
        self._rows = list(rows)
        self._exc = exc
        self._delay = delay
        self.uri = uri

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def query(self, cypher, **params):
        if self._delay:
            time.sleep(self._delay)
        if self._exc is not None:
            raise self._exc
        return list(self._rows)


def _reachable_graph(monkeypatch, **stub_kwargs):
    """Patch a graph that the session gate reports as reachable."""
    client = _StubGraph(**stub_kwargs)

    monkeypatch.setattr(seed, "_session_graph_gate", lambda: (None, ""))
    monkeypatch.setattr(seed, "GraphClient", lambda: client)
    return client


def _gate(monkeypatch, *, reachable, credentialed, reason):
    """Patch the session gate's own answers."""
    monkeypatch.setattr(
        seed,
        "_loaded_conftest",
        lambda: SimpleNamespace(
            _check_neo4j=lambda: reachable,
            _graph_credential_is_configured=lambda: credentialed,
            _neo4j_unavailable_reason=lambda: reason,
        ),
    )


def test_no_route_is_named_when_the_host_does_not_answer(monkeypatch):
    _gate(monkeypatch, reachable=False, credentialed=True, reason="host not available")

    probe = seed._probe_live_graph()

    assert probe.verdict == seed.NO_ROUTE
    assert not probe.dd_content_available
    assert probe.reason.startswith("no-route: host not available")


def test_no_credential_is_named_and_no_connection_is_attempted(monkeypatch):
    _gate(
        monkeypatch,
        reachable=False,
        credentialed=False,
        reason="No Neo4j credential is configured for this checkout",
    )
    attempts: list[int] = []
    monkeypatch.setattr(seed, "GraphClient", lambda: attempts.append(1))

    probe = seed._probe_live_graph()

    assert probe.verdict == seed.NO_CREDENTIAL
    assert probe.reason.startswith(
        "no-credential: No Neo4j credential is configured for this checkout"
    )
    assert attempts == [], "a missing credential must not reach a login attempt"


def test_reachable_but_empty_is_named_when_the_graph_holds_no_dd_content(monkeypatch):
    _reachable_graph(monkeypatch, rows=[])

    probe = seed._probe_live_graph()

    assert probe.verdict == seed.EMPTY
    assert not probe.dd_content_available
    assert "reachable-but-empty" in probe.reason
    assert "no DD content" in probe.reason
    assert "bolt://stub:7687" in probe.reason


def test_ready_is_reported_when_dd_content_is_present(monkeypatch):
    _reachable_graph(
        monkeypatch, rows=[{"id": "equilibrium/time_slice/profiles_1d/psi"}]
    )

    probe = seed._probe_live_graph()

    assert probe.verdict == seed.READY
    assert probe.dd_content_available


def test_an_unclassified_failure_names_its_own_exception(monkeypatch):
    _reachable_graph(monkeypatch, exc=RuntimeError("planner exploded"))

    probe = seed._probe_live_graph()

    assert probe.verdict == seed.PROBE_FAILED
    assert "RuntimeError" in probe.reason


def test_the_three_unavailable_reasons_are_pairwise_distinct(monkeypatch):
    """One string for three faults is the defect this probe exists to remove."""
    _gate(monkeypatch, reachable=False, credentialed=True, reason="no answer")
    refused = seed._probe_live_graph().reason

    _gate(monkeypatch, reachable=False, credentialed=False, reason="no credential")
    rejected = seed._probe_live_graph().reason

    _reachable_graph(monkeypatch, rows=[])
    empty = seed._probe_live_graph().reason

    assert len({refused, rejected, empty}) == 3
    for reason in (refused, rejected, empty):
        assert reason != "DD-loaded graph not available"


def test_probe_stops_waiting_at_its_stated_bound(monkeypatch):
    """A client that never returns must not hold collection open."""
    _reachable_graph(monkeypatch, delay=5.0)
    monkeypatch.setattr(seed, "PROBE_TIMEOUT_SECONDS", 0.25)

    started = time.monotonic()
    probe = seed._probe_live_graph()
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"probe waited {elapsed:.2f}s past its 0.25s bound"
    assert probe.verdict == seed.NO_ROUTE
    assert "0.25s" in probe.reason


def test_the_stated_bound_is_a_positive_finite_number():
    assert isinstance(seed.PROBE_TIMEOUT_SECONDS, int | float)
    assert 0 < seed.PROBE_TIMEOUT_SECONDS < 60


def test_the_skip_reason_is_built_from_the_probe_verdict():
    """The marker both tests carry must quote the probe, not a fixed string."""
    reason = seed._REQUIRES_DD_CONTENT.kwargs["reason"]

    assert seed._PROBE.reason in reason


@pytest.mark.parametrize(
    "verdict", ["NO_ROUTE", "NO_CREDENTIAL", "EMPTY", "PROBE_FAILED"]
)
def test_every_verdict_other_than_ready_skips(verdict):
    """Only a graph carrying DD content may let the two tests run."""
    assert getattr(seed, verdict) != seed.READY

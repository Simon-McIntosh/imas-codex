"""A scoped pool run confines its source-drift refresh to its own names.

The refresh itself is guarded elsewhere. What no test reached before is the
loop half: :func:`_scope_name_ids` decides which names a scoped run owns, and
``run_sn_pools`` forwards that set to :func:`refresh_drifted_sources`. Drop
either half and a family apply sweeps the whole graph again, steering names no
one asked it to touch.

``_scope_name_ids`` also has to fail closed in two different ways. A scope it
cannot read must skip the refresh and say so, never read as "this run owns
nothing" and never fall open into an unscoped sweep. A ``RuntimeError`` it was
never meant to catch — the default-tier guard in ``conftest.py`` raises one
when a test opens a live ``GraphClient`` — must propagate, or the tests pass
without ever inspecting the guard.

The orchestrator driver and its patch set are owned by
``test_scoped_global_maintenance`` and imported here rather than copied: a
maintenance pass added to ``run_sn_pools`` is patched in one place, so a copy
would leave this file driving, and failing on, an unpatched writer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import patch

import pytest
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from imas_codex.standard_names import loop
from tests.standard_names import test_scoped_global_maintenance as scoped_maintenance


def _run(**kwargs: Any) -> tuple[Any, Any]:
    """Drive the shared orchestrator harness without bypassing maintenance."""
    summary, maintenance, *_ = asyncio.run(
        scoped_maintenance._run_loop(skip_global_maintenance=False, **kwargs)
    )
    return summary, maintenance["refresh_drifted_sources"]


def test_scoped_loop_forwards_its_own_name_set_to_the_refresh() -> None:
    """Dropping ``sn_ids=_refresh_ids`` lets the refresh go graph-wide."""
    summary, refresh = _run(scope_rows=[{"id": "alpha"}, {"id": "beta"}])

    refresh.assert_called_once()
    kwargs = refresh.call_args.kwargs
    assert kwargs["sn_ids"] == ["alpha", "beta"]
    assert kwargs["scope_run_id"] == summary.run_id


def test_unscoped_loop_leaves_the_refresh_graph_wide() -> None:
    _, refresh = _run(scope_run_id=None)

    refresh.assert_called_once()
    assert refresh.call_args.kwargs["sn_ids"] is None


def test_a_scope_that_owns_nothing_refreshes_nothing() -> None:
    """An empty readable scope is a target set of no names, not a wide sweep."""
    _, refresh = _run(scope_rows=[])

    refresh.assert_called_once()
    assert refresh.call_args.kwargs["sn_ids"] == []


def test_a_scope_answer_without_name_ids_skips_the_refresh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A response carrying no name ids cannot say which names the run owns."""
    with caplog.at_level(logging.WARNING):
        _, refresh = _run(scope_rows=[{"cnt": 2}])

    refresh.assert_not_called()
    assert "could not be read" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        ServiceUnavailable("no route to the graph"),
        SessionExpired("session expired"),
        TransientError("lock timeout"),
        OSError("network unreachable"),
        TimeoutError("read timed out"),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_a_scope_read_that_fails_skips_the_refresh(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """Every connection-class failure skips the refresh; none aborts the run.

    ``SessionExpired`` and ``TransientError`` — a lock timeout among them — are
    what a concurrent drain meets, and letting either escape turns a skipped
    refresh into a failed run.
    """
    with caplog.at_level(logging.WARNING):
        _, refresh = _run(graph_error=error)

    refresh.assert_not_called()
    assert "bounded-run" in caplog.text


def test_scope_selector_reads_both_ways_a_run_binds_its_names() -> None:
    """The selector must ask for direct-marked and source-produced names.

    Dropping either arm silently shrinks a scope: a family apply seeds its
    members through ``StandardNameSource`` and marks only some directly, so one
    arm alone returns the wrong set.
    """
    captured: dict[str, Any] = {}

    class _Graph:
        def query(self, cypher: str, **kwargs: Any) -> list[dict[str, str]]:
            captured["cypher"] = cypher
            captured["kwargs"] = kwargs
            return [{"id": "from-run"}, {"id": "from-source"}]

    class _Client:
        def __enter__(self) -> _Graph:
            return _Graph()

        def __exit__(self, *exc: object) -> bool:
            return False

    with patch("imas_codex.graph.client.GraphClient", _Client):
        ids = loop._scope_name_ids("scope-1")

    assert ids == ["from-run", "from-source"]
    assert captured["kwargs"] == {"rid": "scope-1"}
    assert "sn.run_id = $rid" in captured["cypher"]
    assert "StandardNameSource {run_id: $rid}" in captured["cypher"]
    assert "PRODUCED_NAME" in captured["cypher"]


def test_scope_selector_lets_the_default_tier_live_graph_guard_raise() -> None:
    """The selector's error handling must not swallow the conftest guard.

    ``conftest.py`` raises ``RuntimeError`` from ``GraphClient.__post_init__``
    when a default-tier test opens a live client. Catch-everything here turns
    that into an empty scope, so the guard's refusal reads as a passing test.
    """
    with pytest.raises(RuntimeError, match="default-tier test"):
        loop._scope_name_ids("scope-1")

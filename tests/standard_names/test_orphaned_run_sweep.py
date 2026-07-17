"""Orphaned-SNRun reconciliation.

``finalize_sn_run`` runs in ``run_sn_pools``' ``finally`` block, so a run left
at the open ``status='started'``/``'running'`` status means the process was
hard-killed before it could close the run. ``mark_orphaned_sn_runs_stale``
sweeps those rows to ``status='stale'`` at the next run start so ``sn status``
and provenance stop treating a dead run as in-flight.

Graph interaction is mocked (no live Neo4j): a capturing fake records the
single sweep query + params and returns pre-canned matched rows.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from imas_codex.standard_names import graph_ops


class _CapturingGraph:
    """Records every query + params and replays a canned result."""

    def __init__(self, result: list[dict[str, Any]]) -> None:
        self._result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _CapturingGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))
        return self._result


def _run(result: list[dict[str, Any]], **kwargs: Any):
    fake = _CapturingGraph(result)
    with patch.object(graph_ops, "GraphClient", return_value=fake):
        marked = graph_ops.mark_orphaned_sn_runs_stale(**kwargs)
    return marked, fake


class TestMarkOrphanedRunsStale:
    def test_marks_matched_rows_and_returns_count(self) -> None:
        marked, fake = _run(
            [{"id": "run-a"}, {"id": "run-b"}],
            current_run_id="run-current",
        )
        assert marked == 2
        # A single sweep query is issued.
        assert len(fake.calls) == 1
        cypher, params = fake.calls[0]
        # It only targets non-terminal runs and stamps the stale outcome.
        assert "status IN ['started', 'running']" in cypher
        assert "rr.status = 'stale'" in cypher
        assert "orphaned_no_finalize" in cypher
        # The current run is excluded and the age threshold is passed through.
        assert params["current_run_id"] == "run-current"
        assert params["max_age"] == 6.0

    def test_no_orphans_returns_zero(self) -> None:
        marked, fake = _run([], current_run_id="run-current")
        assert marked == 0
        assert len(fake.calls) == 1

    def test_custom_age_threshold_passed_through(self) -> None:
        _marked, fake = _run([], current_run_id="x", max_age_hours=1.5)
        _cypher, params = fake.calls[0]
        assert params["max_age"] == 1.5

    def test_current_run_id_optional(self) -> None:
        """A sweep with no current run (None) still issues the query; the
        Cypher's NULL guard keeps every non-terminal row eligible."""
        _marked, fake = _run([{"id": "r1"}])
        _cypher, params = fake.calls[0]
        assert params["current_run_id"] is None

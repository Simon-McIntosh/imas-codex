"""A drained source queue must exit cleanly; a stuck claim must not.

The generate_name claim seeds from a source reached through ``FROM_DD_PATH``
and skips sources already bound to a frozen produced name.  A source with no
DD path can never be selected.  If the pending count includes such a source,
the exit watchdog sees pending work that no pool can claim: the run never goes
idle, the stall timer fires, and a run whose claimable work is exhausted is
reported as a deadlock (``stalled``, exit 1) instead of a drain
(``no_eligible_work``, exit 0).

These tests drive the real pending-count query and the real exit watchdog.  The
graph is a fixture double: it answers the count from a source list and only
applies a restriction the query actually names, so a query whose predicate is
narrower or wider than the claim changes and the verdict changes with it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from imas_codex.cli.sn import _compute_pool_progress, _require_terminal_drain
from imas_codex.standard_names.pools import (
    PoolSpec,
    _idle_exhaustion_watchdog,
)

_POOLS = (
    "generate_name",
    "review_name",
    "refine_name",
    "generate_docs",
    "review_docs",
    "refine_docs",
    "enrich_parents",
)


@dataclass(frozen=True)
class _Source:
    """One StandardNameSource row reduced to what claimability turns on."""

    has_dd_path: bool
    frozen_produced: bool = False


class _FixtureGraph:
    """Graph double answering the aggregate pending-count query.

    ``generate_name`` counts only the sources the *query* can exclude the
    unclaimable ones from: a restriction the query does not name is one the
    count does not apply.  Every other pool is empty, so the verdict turns on
    the source queue alone.
    """

    def __init__(self, sources: list[_Source]) -> None:
        self._sources = sources
        self.queries: list[str] = []

    def __enter__(self) -> _FixtureGraph:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def query(self, cypher: str, **_params: object) -> list[dict[str, int]]:
        self.queries.append(cypher)
        generate_clause = _generate_name_clause(cypher)
        requires_dd_path = "FROM_DD_PATH" in generate_clause
        skips_frozen_produced = "PRODUCED_NAME" in generate_clause
        claimable = sum(
            1
            for source in self._sources
            if (not requires_dd_path or source.has_dd_path)
            and (not skips_frozen_produced or not source.frozen_produced)
        )
        row = dict.fromkeys(_POOLS, 0)
        row["generate_name"] = claimable
        row.update({f"{name}_done": 0 for name in _POOLS})
        return [row]


def _unclaimable_sources() -> list[_Source]:
    """The 101 derived and 3 manual sources job 1308077 left pending.

    Measured on the live graph: 104 sources at ``status='extracted'`` with
    ``attempt_count < 5`` and no ``FROM_DD_PATH`` edge, out of 10,141 sources
    (9,078 carrying that edge, which proves the query sees links it can see).
    """
    return [_Source(has_dd_path=False) for _ in range(104)]


def _pending_counts(sources: list[_Source]) -> dict[str, int]:
    graph = _FixtureGraph(sources)
    progress = _compute_pool_progress(
        graph,
        domains=None,
        rotation_cap=3,
        min_score=0.75,
    )
    return {name: counts["pending"] for name, counts in progress.items()}


async def _watchdog_verdict(counts: dict[str, int]) -> str:
    """Run the real exit watchdog over a pool set carrying *counts*."""
    pools = [
        PoolSpec(name=name, claim=_no_claim, process=_no_process) for name in _POOLS
    ]
    for pool in pools:
        pool.health.pending_count = counts[pool.name]
        pool.health._pending_observation_epoch = 0
    stop_event = asyncio.Event()
    idle_event = asyncio.Event()
    stalled_event = asyncio.Event()
    task = asyncio.create_task(
        _idle_exhaustion_watchdog(
            pools,
            stop_event,
            idle_event,
            stalled_event,
            poll=0.01,
            idle_polls=3,
            stall_seconds=0.05,
        )
    )
    try:
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        if not task.done():
            task.cancel()
    if idle_event.is_set():
        return "no_eligible_work"
    if stalled_event.is_set():
        return "stalled"
    return "timed_out"


async def _no_claim() -> dict[str, object] | None:
    return None


async def _no_process(_batch: dict[str, object]) -> int:
    return 0


def _verdict(sources: list[_Source]) -> str:
    counts = _pending_counts(sources)
    return asyncio.run(_watchdog_verdict(counts))


def test_unclaimable_pending_is_not_counted() -> None:
    """A source no claim can select is not pending work."""
    assert _pending_counts(_unclaimable_sources())["generate_name"] == 0


def test_a_claimable_source_is_still_counted() -> None:
    """Positive control: the count still sees work the claim can select."""
    sources = [_Source(has_dd_path=True), _Source(has_dd_path=False)]
    assert _pending_counts(sources)["generate_name"] == 1


def test_a_frozen_produced_name_is_not_counted() -> None:
    """A source already bound to a frozen produced name cannot be claimed."""
    sources = [_Source(has_dd_path=True, frozen_produced=True)]
    assert _pending_counts(sources)["generate_name"] == 0


def test_drained_run_reports_no_eligible_work_and_exits_zero() -> None:
    """Only unclaimable sources remain: the run drains instead of stalling."""
    verdict = _verdict(_unclaimable_sources())
    assert verdict == "no_eligible_work"
    _require_terminal_drain(verdict)  # returns; exits 0


def test_a_stuck_claim_still_reports_stalled_and_exits_one() -> None:
    """A genuinely claimable source with nothing in flight is still a stall."""
    verdict = _verdict([_Source(has_dd_path=True)])
    assert verdict == "stalled"
    with pytest.raises(SystemExit) as exit_info:
        _require_terminal_drain(verdict)
    assert exit_info.value.code == 1


def test_the_generate_count_mirrors_the_claim_seed_predicate() -> None:
    """Both pending-count queries must name the claim's own restrictions."""
    from imas_codex.standard_names.graph_store.graph_parent_lifecycle import (
        pool_pending_counts,
    )

    graph = _FixtureGraph([])
    _compute_pool_progress(graph, domains=None, rotation_cap=3, min_score=0.75)
    cli_query = graph.queries[-1]

    lifecycle_graph = _FixtureGraph([])
    with patch("imas_codex.graph.client.GraphClient", return_value=lifecycle_graph):
        pool_pending_counts(
            services=SimpleNamespace(_MAX_COMPOSE_CLAIM_ATTEMPTS=5),
        )
    lifecycle_query = lifecycle_graph.queries[-1]

    for label, cypher in (("cli", cli_query), ("lifecycle", lifecycle_query)):
        generate_clause = _generate_name_clause(cypher)
        assert "FROM_DD_PATH" in generate_clause, label
        assert "PRODUCED_NAME" in generate_clause, label


def _generate_name_clause(cypher: str) -> str:
    """Return the generate_name subquery of an aggregate pending-count query."""
    marker = "AS generate_name"
    end = cypher.index(marker)
    start = cypher.rindex("CALL {", 0, end)
    return cypher[start:end]

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
import uuid
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
    """The 101 derived and 3 manual sources a drained queue left pending.

    Measured on the live graph: 104 sources at ``status='extracted'`` with
    ``attempt_count < 5`` out of 10,141 (9,078 of which carry a
    ``FROM_DD_PATH`` edge), and all 104 pending sources without that edge.
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


def test_every_pending_count_names_the_claim_restrictions() -> None:
    """Both pending-count queries must name the claim's own restrictions."""
    from imas_codex.standard_names.graph_store.graph_parent_lifecycle import (
        pool_pending_counts,
    )

    cli_query = _generate_name_clause(_count_cypher([]))

    lifecycle_graph = _FixtureGraph([])
    with patch("imas_codex.graph.client.GraphClient", return_value=lifecycle_graph):
        pool_pending_counts(
            services=SimpleNamespace(
                _MAX_COMPOSE_CLAIM_ATTEMPTS=5,
                _PIPELINE_FROZEN_NAME_STAGES=("approved", "contested"),
            ),
        )
    lifecycle_query = _generate_name_clause(lifecycle_graph.queries[-1])

    for label, cypher in (("cli", cli_query), ("lifecycle", lifecycle_query)):
        assert "FROM_DD_PATH" in cypher, label
        assert "PRODUCED_NAME" in cypher, label


def test_the_count_and_the_claim_select_the_same_sources() -> None:
    """The pending count and the claim select the same sources on a fake."""
    sources = [
        _Source(has_dd_path=True),
        _Source(has_dd_path=True),
        _Source(has_dd_path=False),
        _Source(has_dd_path=True, frozen_produced=True),
    ]
    count_clause = _generate_name_clause(_count_cypher(sources))

    assert _selects(count_clause, sources) == 2
    assert _selects(_claim_seed_cypher(), sources) == _selects(count_clause, sources)


def test_both_count_sites_and_the_claim_read_one_predicate() -> None:
    """The claimability predicate is defined once and read by every site."""
    from imas_codex.standard_names.graph_store.graph_foundation import (
        source_claim_predicate,
    )

    count_clause = _generate_name_clause(_count_cypher([]))
    count_predicate = source_claim_predicate(alias="s")

    assert count_predicate in count_clause
    assert count_predicate.replace("(s)-", "(sns)-") in _claim_seed_cypher()


def _count_cypher(sources: list[_Source]) -> str:
    graph = _FixtureGraph(sources)
    _compute_pool_progress(graph, domains=None, rotation_cap=3, min_score=0.75)
    return graph.queries[-1]


def _selects(cypher: str, sources: list[_Source]) -> int:
    """Apply to a source list the restrictions the query names."""
    requires_dd_path = "FROM_DD_PATH" in cypher
    skips_frozen_produced = "PRODUCED_NAME" in cypher
    return sum(
        1
        for source in sources
        if (not requires_dd_path or source.has_dd_path)
        and (not skips_frozen_produced or not source.frozen_produced)
    )


def _claim_seed_cypher() -> str:
    """Return the Cypher the generate_name claim seeds with.

    The graph double answers with no rows, so the claim reports an empty batch
    after its first statement and nothing is written.
    """
    from imas_codex.standard_names.graph_store import graph_cost_and_claims

    graph = _RecordingGraph()
    graph_cost_and_claims.claim_generate_name_batch(
        services=SimpleNamespace(uuid=uuid, GraphClient=lambda: graph)
    )
    return graph.queries[0]


class _RecordingGraph:
    """Graph double that records the Cypher a claim runs."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def __enter__(self) -> _RecordingGraph:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def session(self) -> _RecordingGraph:
        return self

    def begin_transaction(self) -> _RecordingGraph:
        return self

    def run(self, cypher: str, **_params: object) -> list[dict[str, object]]:
        self.queries.append(cypher)
        return []

    def close(self) -> None:
        return None


def _generate_name_clause(cypher: str) -> str:
    """Return the generate_name subquery of an aggregate pending-count query."""
    marker = "AS generate_name"
    end = cypher.index(marker)
    start = cypher.rindex("CALL {", 0, end)
    return cypher[start:end]

"""A re-enriched signal keeps its recorded check instead of being read again."""

import asyncio
from unittest.mock import AsyncMock, patch

from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.parallel import DataDiscoveryState, check_worker
from imas_codex.graph.models import FacilitySignalStatus


class CapturingGraph:
    def __init__(self, carried):
        self.carried = carried
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, cypher, **params):
        self.queries.append((cypher, params))
        return [{"carried": self.carried}]


def test_only_unclaimed_enriched_signals_with_a_recorded_check_move(monkeypatch):
    graph = CapturingGraph(carried=4)
    monkeypatch.setattr(parallel, "GraphClient", lambda: graph)

    assert parallel.carry_forward_recorded_checks("jt-60sa") == 4

    cypher, params = graph.queries[0]
    where = cypher.split("WHERE", 1)[1].split("SET", 1)[0]
    assert "s.status = $enriched" in where
    assert "s.checked = true" in where
    assert "s.checked_at IS NOT NULL" in where
    assert "s.claimed_at IS NULL" in where
    assert "SET s.status = $checked" in cypher
    assert params == {
        "facility": "jt-60sa",
        "enriched": FacilitySignalStatus.enriched.value,
        "checked": FacilitySignalStatus.checked.value,
    }


def test_check_worker_keeps_recorded_checks_before_claiming():
    state = DataDiscoveryState(
        facility="jt-60sa",
        ssh_host="jt-60sa",
        scanner_types=["edas"],
        cost_limit=1.0,
    )
    calls = []

    def carry(facility):
        calls.append(("carry", facility))
        return 3

    def claim(facility, **_kwargs):
        calls.append(("claim", facility))
        state.stop_requested = True
        return []

    with (
        patch.object(parallel, "carry_forward_recorded_checks", side_effect=carry),
        patch.object(parallel, "claim_signals_for_check", side_effect=claim),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        asyncio.run(check_worker(state))

    assert calls == [("carry", "jt-60sa"), ("claim", "jt-60sa")]

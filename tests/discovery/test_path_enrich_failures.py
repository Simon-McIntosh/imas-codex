"""Remote outages and path errors have different enrichment retry costs."""

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from imas_codex.discovery.paths import enrichment, parallel


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(255, "enrich_directories.py"),
        subprocess.TimeoutExpired("enrich_directories.py", 300),
    ],
)
def test_unreachable_host_pauses_without_charging(monkeypatch, failure):
    state = parallel.DiscoveryState(facility="sample")
    claimed = [{"path": "/source", "path_purpose": "code"}]
    waits = []
    persisted = []

    async def remote(*_args, **_kwargs):
        raise failure

    async def wait(_state, attempt):
        waits.append(attempt)
        state.stop_requested = True

    environment = SimpleNamespace(python_command="python", setup_commands=[])
    monkeypatch.setattr(
        enrichment, "_build_enrich_input", lambda *_: ("host", {}, environment)
    )
    monkeypatch.setattr(enrichment, "async_run_python_script", remote)
    monkeypatch.setattr(
        parallel, "claim_paths_for_enriching", lambda *_a, **_k: claimed
    )

    def persist(*args):
        persisted.append(args)
        state.stop_requested = True
        return 0

    monkeypatch.setattr(parallel, "mark_enrichment_complete", persist)
    monkeypatch.setattr(parallel, "_revert_path_claims", lambda *_a: None)
    monkeypatch.setattr(parallel, "_wait_for_reachable_host", wait, raising=False)

    asyncio.run(parallel.enrich_worker(state))

    assert waits == [1]
    assert persisted == []
    assert state.enrich_phase.done is False


class PathGraph:
    """Track the fields used by the enrichment eligibility queries."""

    def __init__(self):
        self.failures = 0
        self.set_aside = False
        self.reason = None
        self.claimed = False
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, cypher, **params):
        self.queries.append((cypher, params))
        if "UNWIND $items" in cypher:
            if "p.enrich_failures = failures" in cypher:
                self.failures += 1
            if "p.enrich_error = item.error" in cypher:
                self.reason = params["items"][0]["error"]
            if (
                "p.enrich_set_aside = item.permanent OR failures >= $failure_limit"
                in cypher
            ):
                self.set_aside = self.failures >= params["failure_limit"]
            return []
        eligible = not self.set_aside or "p.enrich_set_aside" not in cypher
        if "AS pending_enrich" in cypher:
            pending = int(eligible)
            return [
                {
                    "pending": pending,
                    "pending_discovered": 0,
                    "pending_scanned": 0,
                    "pending_expand": 0,
                    "pending_enrich": pending,
                    "pending_score": 0,
                }
            ]
        if "SET p.claimed_at = datetime()" in cypher:
            self.claimed = eligible
            return []
        if "claim_token: $token" in cypher:
            return [{"path": "/source", "id": "sample:/source"}] if self.claimed else []
        return []


def test_reachable_failure_sets_path_aside_with_reason(monkeypatch):
    graph = PathGraph()
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: graph)
    for attempt in range(1, 6):
        parallel.mark_enrichment_complete(
            "sample", [{"path": "/source", "error": "script failed"}]
        )
        assert graph.failures == attempt
        assert graph.set_aside is (attempt == 5)
    assert graph.reason == "script failed"
    assert parallel.has_pending_work("sample") is False
    assert parallel.claim_paths_for_enriching("sample") == []


def test_set_aside_path_is_neither_pending_nor_claimable(monkeypatch):
    graph = PathGraph()
    graph.set_aside = True
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: graph)
    assert parallel.has_pending_work("sample") is False
    assert parallel.claim_paths_for_enriching("sample") == []

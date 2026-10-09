"""Remote outages and path errors have different enrichment retry costs."""

import asyncio
import subprocess
from types import SimpleNamespace

from imas_codex.discovery.base import reachability
from imas_codex.discovery.code import workers
from imas_codex.discovery.paths import enrichment, frontier, parallel


def test_code_scan_and_path_enrichment_share_host_reachability():
    assert workers.host_unreachable is reachability.host_unreachable
    assert workers.wait_for_reachable_host is reachability.wait_for_reachable_host
    assert parallel.host_unreachable is reachability.host_unreachable
    assert parallel.wait_for_reachable_host is reachability.wait_for_reachable_host
    assert enrichment.host_unreachable is reachability.host_unreachable


def test_unreachable_host_pauses_without_charging(monkeypatch):
    failure = subprocess.CalledProcessError(255, "enrich_directories.py")
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
    monkeypatch.setattr(parallel, "wait_for_reachable_host", wait)

    asyncio.run(parallel.enrich_worker(state))

    assert waits == [1]
    assert persisted == []
    assert state.enrich_phase.done is False


def test_timeout_recovers_partial_output_and_charges_unfinished_path(monkeypatch):
    state = parallel.DiscoveryState(facility="sample")
    claimed = [
        {"path": "/finished", "path_purpose": "code"},
        {"path": "/unfinished", "path_purpose": "code"},
    ]
    persisted = []
    waits = []
    timeout = subprocess.TimeoutExpired("enrich_directories.py", 300)
    timeout.output = '{"path": "/finished", "total_bytes": 42}\n{"path": '

    async def remote(*_args, **_kwargs):
        raise timeout

    async def wait(_state, attempt):
        waits.append(attempt)
        state.stop_requested = True

    def persist(_facility, results):
        persisted.extend(results)
        state.stop_requested = True
        return 1

    environment = SimpleNamespace(python_command="python", setup_commands=[])
    monkeypatch.setattr(
        enrichment, "_build_enrich_input", lambda *_: ("host", {}, environment)
    )
    monkeypatch.setattr(enrichment, "async_run_python_script", remote)
    monkeypatch.setattr(
        parallel, "claim_paths_for_enriching", lambda *_a, **_k: claimed
    )
    monkeypatch.setattr(parallel, "mark_enrichment_complete", persist)
    monkeypatch.setattr(parallel, "wait_for_reachable_host", wait)

    asyncio.run(parallel.enrich_worker(state))

    assert waits == []
    assert [(item["path"], item["error"]) for item in persisted] == [
        ("/finished", None),
        ("/unfinished", "timeout after 300s"),
    ]
    assert persisted[0]["total_bytes"] == 42
    assert state.enrich_stats.errors == 1


class PathGraph:
    """Track the fields used by the enrichment eligibility queries."""

    def __init__(self):
        self.failures = 0
        self.should_enrich = True
        self.reason = None
        self.skip_reason = None
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
                assert "p.should_enrich = CASE WHEN" in cypher
                assert "p.enrich_skip_reason = CASE WHEN" in cypher
                self.failures += 1
            if "p.enrich_error = item.error" in cypher:
                self.reason = params["items"][0]["error"]
            if self.failures >= params["failure_limit"]:
                self.should_enrich = False
                self.skip_reason = params["items"][0]["error"]
            return []
        eligible = self.should_enrich or "p.should_enrich IS NULL AND" not in cypher
        if "AS enrichment_ready" in cypher:
            fields = (
                "total discovered scanned scored skipped excluded claimed "
                "expansion_ready enrichment_ready enriched triaged explored "
                "score_ready max_depth"
            ).split()
            counts = dict.fromkeys(fields, 0)
            counts["total"] = 1
            counts["enrichment_ready"] = int(eligible)
            return [counts]
        if "RETURN p.path AS path" in cypher:
            return [{"path": "/source"}] if eligible else []
        if "AS has_work" in cypher:
            return [{"has_work": eligible}]
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
        assert graph.should_enrich is (attempt < 5)
    assert graph.reason == "script failed"
    assert graph.skip_reason == "script failed"
    assert parallel.has_pending_work("sample") is False
    assert parallel._has_pending_enrich_work("sample") is False
    assert parallel.claim_paths_for_enriching("sample") == []
    assert frontier.get_discovery_stats("sample")["enrichment_ready"] == 0
    assert enrichment.get_paths_pending_enrichment("sample") == []
    assert all("enrich_set_aside" not in cypher for cypher, _ in graph.queries)


def test_set_aside_path_is_neither_pending_nor_claimable(monkeypatch):
    graph = PathGraph()
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: graph)
    assert parallel.has_pending_work("sample") is True
    assert parallel._has_pending_enrich_work("sample") is True
    assert frontier.get_discovery_stats("sample")["enrichment_ready"] == 1
    assert enrichment.get_paths_pending_enrichment("sample") == ["/source"]
    graph.should_enrich = False
    assert parallel.has_pending_work("sample") is False
    assert parallel._has_pending_enrich_work("sample") is False
    assert parallel.claim_paths_for_enriching("sample") == []
    assert frontier.get_discovery_stats("sample")["enrichment_ready"] == 0
    assert enrichment.get_paths_pending_enrichment("sample") == []

"""Concurrent code ingestion claims must keep one owner per file."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from imas_codex.discovery.code.workers import _claim_code_files_for_ingestion


class ClaimGraph:
    def __init__(self, count=4, timeout=300):
        self.rows = [
            {"id": str(i), "path": f"/source/{i}.py", "status": "scored"}
            for i in range(count)
        ]
        self.timeout = timeout
        self.lock = threading.Lock()
        self.race = threading.Barrier(2)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, cypher, **params):
        if "RETURN count(sf) AS refreshed" in cypher:
            return self._refresh(params)
        if "SET sf.claimed_at" in cypher:
            if "SET f.name = f.name" in cypher:
                with self.lock:
                    return self._claim(params)
            # Both transactions read the same eligible batch before either writes.
            eligible = self._eligible(params)
            self.race.wait(timeout=2)
            if threading.current_thread().name.endswith("_1"):
                time.sleep(0.05)
            self._write_claims(eligible, params["token"])
            return []
        if "claim_token: $token" in cypher:
            return [
                row.copy()
                for row in self.rows
                if row.get("claim_token") == params["token"]
            ]
        raise AssertionError(cypher)

    def _eligible(self, params):
        now = time.monotonic()
        return [
            row
            for row in self.rows
            if row["status"] == "scored"
            and (
                row.get("claimed_at") is None or row["claimed_at"] < now - self.timeout
            )
        ][: params["limit"]]

    def _write_claims(self, rows, token):
        for row in rows:
            row["claimed_at"] = time.monotonic()
            row["claim_token"] = token

    def _claim(self, params):
        self._write_claims(self._eligible(params), params["token"])
        return []

    def _refresh(self, params):
        refreshed = 0
        for claim in params["claims"]:
            row = next(row for row in self.rows if row["id"] == claim["id"])
            if row.get("claim_token") == claim["token"] and row["status"] == "scored":
                row["claimed_at"] = time.monotonic()
                refreshed += 1
        return [{"refreshed": refreshed}]


def test_two_concurrent_claims_take_each_file_once():
    graph = ClaimGraph()
    with patch("imas_codex.graph.GraphClient", return_value=graph):
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="claim") as pool:
            futures = [
                pool.submit(_claim_code_files_for_ingestion, "test", limit=4)
                for _ in range(2)
            ]
            claims = [future.result(timeout=5) for future in futures]
    ids = [row["id"] for batch in claims for row in batch]
    assert sorted(ids) == ["0", "1", "2", "3"]


def test_live_claim_is_renewed_past_timeout():
    from imas_codex.discovery.base import claims as claim_settings
    from imas_codex.discovery.code import workers
    from imas_codex.discovery.code.state import FileDiscoveryState

    async def check():
        graph = ClaimGraph(count=1, timeout=0.2)
        state = FileDiscoveryState(facility="test")
        started = threading.Event()

        async def ingest_files(**_kwargs):
            started.set()
            # The real pipeline makes synchronous graph calls while ingesting.
            time.sleep(0.55)
            state.stop_requested = True
            return {
                "files": 1,
                "skipped": 0,
                "chunks": 1,
                "outcomes": {"/source/0.py": {"status": "ingested"}},
            }

        with (
            patch("imas_codex.graph.GraphClient", return_value=graph),
            patch.object(claim_settings, "DEFAULT_CLAIM_TIMEOUT_SECONDS", 0.2),
            patch.object(workers, "_settle_unclaimable_files", return_value=0),
            patch.object(
                workers, "_filter_duplicates", side_effect=lambda files: files
            ),
            patch.object(workers, "_mark_files_ingested", return_value=1),
            patch(
                "imas_codex.ingestion.pipeline.ingest_files", side_effect=ingest_files
            ),
        ):
            worker = asyncio.create_task(workers.code_worker(state, batch_size=1))
            await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2)
            await asyncio.sleep(0.4)
            contender = await asyncio.to_thread(
                _claim_code_files_for_ingestion, "test", limit=1
            )
            assert contender == []
            await asyncio.wait_for(worker, timeout=2)

    asyncio.run(check())

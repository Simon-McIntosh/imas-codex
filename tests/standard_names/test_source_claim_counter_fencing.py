"""Concurrency contracts for source claim ownership counters."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch


@dataclass
class _SourceState:
    """Thread-safe single-source graph state used by concurrent claimers."""

    attempt_count: int = 4
    claim_seq: int = 6
    claimed_at: str | None = None
    claim_token: str | None = None
    status: str = "extracted"
    lock: threading.Lock = field(default_factory=threading.Lock)
    queries: list[str] = field(default_factory=list)

    def eligible(self, max_attempts: int) -> bool:
        claim_available = self.claimed_at is None or self.claimed_at == "stale"
        return (
            self.status == "extracted"
            and self.attempt_count < max_attempts
            and claim_available
        )


class _Transaction:
    """Minimal transaction that models the query's lock-then-check ordering."""

    def __init__(self, state: _SourceState, barrier: threading.Barrier):
        self._state = state
        self._barrier = barrier
        self.closed = False

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self._state.queries.append(cypher)
        if "WITH DISTINCT coalesce(imas0.physics_domain" in cypher:
            assert cypher.index("SET sns2._claim_lock = true") < cypher.index(
                "REMOVE sns2._claim_lock"
            )
            assert cypher.index("REMOVE sns2._claim_lock") < cypher.rindex(
                "WHERE sns2.status = 'extracted'"
            )
            self._barrier.wait(timeout=5)
            with self._state.lock:
                if not self._state.eligible(params["max_attempts"]):
                    return []
                self._state.claimed_at = "fresh"
                self._state.claim_token = params["token"]
                self._state.claim_seq += 1
                self._state.attempt_count += 1
                return [
                    {
                        "_cluster_id": None,
                        "_unit": None,
                        "_physics_domain": "spectroscopy",
                        "_batch_key": "spectrometer_visible",
                    }
                ]

        if "{claim_token: $token}" in cypher:
            with self._state.lock:
                if self._state.claim_token != params["token"]:
                    return []
                return [
                    {
                        "id": "dd:spectrometer_visible/channel/isotope_ratios",
                        "source_id": "spectrometer_visible/channel/isotope_ratios",
                        "source_type": "dd",
                        "batch_key": "spectrometer_visible",
                        "description": "Per-isotope density ratio",
                        "physics_domain": "spectroscopy",
                        "claim_token": self._state.claim_token,
                        "claim_seq": self._state.claim_seq,
                        "attempt_count": self._state.attempt_count,
                    }
                ]

        raise AssertionError(f"unexpected query: {cypher}")

    def commit(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, state: _SourceState, barrier: threading.Barrier):
        self._state = state
        self._barrier = barrier

    def begin_transaction(self) -> _Transaction:
        return _Transaction(self._state, self._barrier)


class _GraphClient:
    def __init__(self, state: _SourceState, barrier: threading.Barrier):
        self._state = state
        self._barrier = barrier

    def __enter__(self) -> _GraphClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    @contextmanager
    def session(self):
        yield _Session(self._state, self._barrier)

    def query(self, _cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._state.lock:
            if (
                self._state.claim_token == params["token"]
                and self._state.status == "extracted"
            ):
                return [
                    {
                        "id": "dd:spectrometer_visible/channel/isotope_ratios",
                        "claim_seq": self._state.claim_seq,
                    }
                ]
            return []


def _claim_concurrently(state: _SourceState, count: int) -> list[list[dict[str, Any]]]:
    """Run *count* synchronized claimers against one shared source."""
    from imas_codex.standard_names.graph_ops import claim_generate_name_batch

    barrier = threading.Barrier(count)
    results: list[list[dict[str, Any]]] = []
    errors: list[BaseException] = []
    result_lock = threading.Lock()

    def _claim() -> None:
        try:
            value = claim_generate_name_batch(batch_size=1)
            with result_lock:
                results.append(value)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    with (
        patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            side_effect=lambda: _GraphClient(state, barrier),
        ),
        patch("imas_codex.standard_names.graph_ops.time.sleep"),
    ):
        threads = [threading.Thread(target=_claim) for _ in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    return results


def test_simultaneous_claimers_increment_only_the_owner() -> None:
    state = _SourceState()

    results = _claim_concurrently(state, 4)

    assert sum(bool(result) for result in results) == 1
    assert state.attempt_count == 5
    assert state.claim_seq == 7
    assert state.claim_token is not None


def test_released_source_retry_increments_once() -> None:
    state = _SourceState(attempt_count=2, claim_seq=8)
    first = _claim_concurrently(state, 1)
    assert first[0]
    assert (state.attempt_count, state.claim_seq) == (3, 9)

    state.claimed_at = None
    state.claim_token = None
    second = _claim_concurrently(state, 1)

    assert second[0]
    assert (state.attempt_count, state.claim_seq) == (4, 10)


def test_stale_claim_is_recovered_once() -> None:
    state = _SourceState(attempt_count=1, claim_seq=3, claimed_at="stale")

    results = _claim_concurrently(state, 3)

    assert sum(bool(result) for result in results) == 1
    assert (state.attempt_count, state.claim_seq) == (2, 4)


def test_attempt_cap_cannot_be_bypassed() -> None:
    state = _SourceState(attempt_count=5, claim_seq=11, claimed_at="stale")

    results = _claim_concurrently(state, 3)

    assert results == [[], [], []]
    assert (state.attempt_count, state.claim_seq) == (5, 11)
    assert state.claim_token is None

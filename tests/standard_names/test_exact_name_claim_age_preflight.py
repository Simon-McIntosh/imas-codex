"""The exact-name preflight treats an abandoned claim as reclaimable.

An exact ``--name`` scope refused any row carrying a claim token or timestamp
regardless of age, so a stale claim left by a worker that exited an hour
earlier refused as if it were live contention, forcing a hand-run orphan sweep
before every relaunch. The preflight now computes the same age cutoff the
orphan sweep uses and treats a claim older than it as reclaimable, keeping a
live refusal only for a claim that is genuinely current. These tests prove
both halves: the old claim no longer refuses, and the fresh claim still
refuses with a message naming it as live.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.defaults import DEFAULT_ORPHAN_SWEEP_TIMEOUT_S
from imas_codex.standard_names.graph_ops import (
    _EXACT_NAME_SCOPE_PREFLIGHT_QUERY,
    _EXACT_NAME_SCOPE_STAMP_QUERY,
    ExactNameScopeConflict,
    _exact_name_scope_refusals,
    scope_exact_standard_names,
)


def _candidate(name_id: str, **overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "id": name_id,
        "name_stage": "drafted",
        "status": "draft",
        "claimed_at": None,
        "claim_token": None,
        "claim_stale": None,
        "run_id": None,
        "drain_scope_id": None,
        "drain_scope_claimed_at": None,
        "drain_claim_scope_id": None,
    }
    candidate.update(overrides)
    return candidate


def _preflight_row(
    name_id: str,
    *,
    matches: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "requested_name": name_id,
        "matches": [_candidate(name_id)] if matches is None else matches,
        "fixture_producers": [],
    }


class TestRefusalAgeRule:
    """The age-aware claim decision in _exact_name_scope_refusals."""

    def test_abandoned_claim_no_longer_refuses(self) -> None:
        """A claim older than the orphan threshold is reclaimable, not refusal."""
        row = _preflight_row(
            "stale",
            matches=[
                _candidate(
                    "stale",
                    claimed_at="opaque-old-timestamp",
                    claim_token="dead-token",
                    claim_stale=True,
                )
            ],
        )
        assert _exact_name_scope_refusals([row]) == []

    def test_fresh_claim_still_refuses_as_live(self) -> None:
        """A claim inside the orphan window refuses with 'live' in the message."""
        row = _preflight_row(
            "fresh",
            matches=[
                _candidate(
                    "fresh",
                    claimed_at="opaque-recent-timestamp",
                    claim_token="live-token",
                    claim_stale=False,
                )
            ],
        )
        refusals = _exact_name_scope_refusals([row])
        assert refusals, "a fresh claim must refuse"
        assert "live" in refusals[0]

    def test_token_without_timestamp_refuses_as_live(self) -> None:
        """A token with no timestamp cannot be proven abandoned, so stays live.

        The orphan sweep also never clears a token whose claimed_at is null
        (its stale-token query requires a timestamp), so refusing here is
        consistent with the reaper rather than a wedge the sweep would clear.
        """
        row = _preflight_row(
            "token_only",
            matches=[_candidate("token_only", claimed_at=None, claim_token="opaque")],
        )
        refusals = _exact_name_scope_refusals([row])
        assert refusals
        assert "live" in refusals[0]

    def test_no_claim_passes(self) -> None:
        """An unclaimed row is untouched by the claim rule."""
        row = _preflight_row("clean")
        assert _exact_name_scope_refusals([row]) == []


class _FakeTransaction:
    """Two-query transaction stub mirroring the real scope's read/stamp split."""

    def __init__(
        self,
        preflight_rows: list[dict[str, object]],
        stamped_ids: list[str] | None = None,
    ) -> None:
        self.preflight_rows = preflight_rows
        self.stamped_ids = stamped_ids or []
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False
        self.committed = False

    def run(self, query: str, **params: object) -> list[dict[str, object]]:
        self.calls.append((query, params))
        if "EXACT_NAME_SCOPE_PREFLIGHT" in query:
            return self.preflight_rows
        if "EXACT_NAME_SCOPE_STAMP" in query:
            return [{"stamped_ids": self.stamped_ids}]
        raise AssertionError(f"unexpected query: {query}")

    def commit(self) -> None:
        self.committed = True
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, transaction: _FakeTransaction) -> None:
        self.transaction = transaction

    def begin_transaction(self) -> _FakeTransaction:
        return self.transaction

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.transaction.closed = True


class _FakeClient:
    def __init__(self, transaction: _FakeTransaction) -> None:
        self._session = _FakeSession(transaction)

    def session(self) -> _FakeSession:
        return self._session


class TestScopeEndToEnd:
    """The full scope accepts an abandoned claim and refuses a live one."""

    def test_abandoned_claim_is_reclaimed_in_one_stamp(self) -> None:
        """A stale claim passes preflight and is reclaimed by the stamp write."""
        transaction = _FakeTransaction(
            [
                _preflight_row(
                    "stale",
                    matches=[
                        _candidate(
                            "stale",
                            claimed_at="opaque-old-timestamp",
                            claim_token="dead-token",
                            claim_stale=True,
                        )
                    ],
                )
            ],
            stamped_ids=["stale"],
        )

        result = scope_exact_standard_names(
            ["stale"], "scope-run", gc=_FakeClient(transaction)
        )

        assert result["stamped"] == 1
        assert transaction.committed is True
        assert len(transaction.calls) == 2

    def test_fresh_claim_refuses_atomic(self) -> None:
        """A live claim refuses with 'live', and nothing is stamped."""
        row = _preflight_row(
            "fresh",
            matches=[
                _candidate(
                    "fresh",
                    claimed_at="opaque-recent-timestamp",
                    claim_token="live-token",
                    claim_stale=False,
                )
            ],
        )
        transaction = _FakeTransaction([row], stamped_ids=["fresh"])

        with pytest.raises(ExactNameScopeConflict, match="live"):
            scope_exact_standard_names(
                ["fresh"], "scope-run", gc=_FakeClient(transaction)
            )

        assert len(transaction.calls) == 1
        assert transaction.committed is False

    def test_queries_source_the_sweep_threshold(self) -> None:
        """Both queries carry the sweep's orphan timeout, not a restated number."""
        transaction = _FakeTransaction([_preflight_row("alpha")], stamped_ids=["alpha"])
        scope_exact_standard_names(
            ["alpha"], "scope-run", gc=_FakeClient(transaction)
        )

        assert len(transaction.calls) == 2
        for _query, params in transaction.calls:
            assert params["claim_timeout_s"] == DEFAULT_ORPHAN_SWEEP_TIMEOUT_S

    def test_stamp_clears_the_stale_claim(self) -> None:
        """The stamp write nulls the reclaimed claim, not just the run id."""
        transaction = _FakeTransaction([_preflight_row("alpha")], stamped_ids=["alpha"])
        scope_exact_standard_names(
            ["alpha"], "scope-run", gc=_FakeClient(transaction)
        )

        stamp_query = transaction.calls[1][0]
        assert "name.claimed_at = null" in stamp_query
        assert "name.claim_token = null" in stamp_query
        assert "SET name.run_id = $run_id" in stamp_query

    def test_preflight_projects_the_age_flag(self) -> None:
        """The preflight query computes claim age against the passed cutoff."""
        assert "claim_stale" in _EXACT_NAME_SCOPE_PREFLIGHT_QUERY
        assert "$claim_timeout_s" in _EXACT_NAME_SCOPE_PREFLIGHT_QUERY
        assert "$claim_timeout_s" in _EXACT_NAME_SCOPE_STAMP_QUERY

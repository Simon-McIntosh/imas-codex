"""An exact scope releases the claim of a run that has stopped.

Age is a proxy for the fact that matters: that no worker holds the claim. The
direct evidence is the owning run's status. A claim whose run reached a terminal
status is abandoned the instant that run stopped, whatever the claim's age, so
the preflight reclaims it instead of waiting out the orphan-sweep timeout. A
claim whose run is still executing refuses however old it is — that half is what
stops this from being a deletion of the check. A claim whose owning run cannot
be resolved keeps the age rule, which is then the only available evidence.

Both halves are asserted, and each fails against the age-only rule: the release
half because age alone refused a fresh claim, the hold half because age alone
released a claim whose run was still running.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.graph_ops import (
    _EXACT_NAME_SCOPE_STAMP_QUERY,
    ExactNameScopeConflict,
    _exact_name_scope_refusals,
    scope_exact_standard_names,
)

TERMINAL_STATUSES = ("completed", "interrupted", "failed", "degraded", "stale")


def _claim(
    name_id: str,
    *,
    claimed_at: str | None = "opaque-recent-timestamp",
    claim_token: str | None = "token",
    claim_stale: bool = False,
    run_id: str | None = None,
) -> dict[str, object]:
    return {
        "id": name_id,
        "name_stage": "drafted",
        "status": "draft",
        "claimed_at": claimed_at,
        "claim_token": claim_token,
        "claim_stale": claim_stale,
        "run_id": run_id,
        "drain_scope_id": None,
        "drain_scope_claimed_at": None,
        "drain_claim_scope_id": None,
    }


def _row(name_id: str, candidate: dict[str, object]) -> dict[str, object]:
    return {
        "requested_name": name_id,
        "matches": [candidate],
        "fixture_producers": [],
    }


class _RecordingTransaction:
    """Answers the preflight, the owning-run read, and the stamp write."""

    def __init__(
        self,
        preflight_rows: list[dict[str, object]],
        *,
        run_statuses: dict[str, str] | None = None,
        stamped_ids: list[str] | None = None,
    ) -> None:
        self.preflight_rows = preflight_rows
        self.run_statuses = run_statuses or {}
        self.stamped_ids = stamped_ids or []
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False
        self.committed = False

    def run(self, query: str, **params: object) -> list[dict[str, object]]:
        self.calls.append((query, params))
        if "EXACT_NAME_SCOPE_PREFLIGHT" in query:
            return self.preflight_rows
        if "EXACT_NAME_SCOPE_RUN_STATUS" in query:
            return [
                {"run_id": run_id, "status": self.run_statuses[run_id]}
                for run_id in params["run_ids"]
                if run_id in self.run_statuses
            ]
        if "EXACT_NAME_SCOPE_STAMP" in query:
            return [{"stamped_ids": self.stamped_ids}]
        raise AssertionError(f"unexpected query: {query}")

    def commit(self) -> None:
        self.committed = True
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, transaction: _RecordingTransaction) -> None:
        self.transaction = transaction

    def begin_transaction(self) -> _RecordingTransaction:
        return self.transaction

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.transaction.closed = True


class _FakeClient:
    def __init__(self, transaction: _RecordingTransaction) -> None:
        self._session = _FakeSession(transaction)

    def session(self) -> _FakeSession:
        return self._session


def _scope(transaction: _RecordingTransaction, names: list[str]) -> dict[str, object]:
    return scope_exact_standard_names(names, "scope-run", gc=_FakeClient(transaction))


class TestStoppedRunReleasesItsClaim:
    """A claim whose owning run is terminal is reclaimable at any age."""

    def test_fresh_claim_held_by_a_stopped_run_is_reclaimed(self) -> None:
        """The claim is recent, so only the run's status can release it."""
        transaction = _RecordingTransaction(
            [_row("held", _claim("held", claim_stale=False, run_id="run-done"))],
            run_statuses={"run-done": "completed"},
            stamped_ids=["held"],
        )

        result = _scope(transaction, ["held"])

        assert result["stamped"] == 1
        assert transaction.committed is True

    @pytest.mark.parametrize("status", TERMINAL_STATUSES)
    def test_every_terminal_status_releases(self, status: str) -> None:
        row = _row("held", _claim("held", claim_stale=False, run_id="run-done"))
        assert _exact_name_scope_refusals([row], {"run-done": status}) == []

    def test_released_claim_is_admitted_by_the_stamp(self) -> None:
        """The stamp cannot still demand a null claim for a released row."""
        transaction = _RecordingTransaction(
            [_row("held", _claim("held", claim_stale=False, run_id="run-done"))],
            run_statuses={"run-done": "completed"},
            stamped_ids=["held"],
        )

        _scope(transaction, ["held"])

        stamp_params = transaction.calls[-1][1]
        assert stamp_params["released_ids"] == ["held"]
        assert "$released_ids" in _EXACT_NAME_SCOPE_STAMP_QUERY


class TestLiveRunHoldsItsClaim:
    """A claim whose owning run is executing refuses however old it is."""

    def test_old_claim_held_by_a_live_run_refuses(self) -> None:
        """Age alone would have released this claim; the live run holds it."""
        row = _row("old", _claim("old", claim_stale=True, run_id="run-busy"))
        refusals = _exact_name_scope_refusals([row], {"run-busy": "started"})
        assert refusals, "a claim owned by a live run must refuse at any age"
        assert "live" in refusals[0]

    def test_old_claim_held_by_a_live_run_refuses_atomic(self) -> None:
        transaction = _RecordingTransaction(
            [_row("old", _claim("old", claim_stale=True, run_id="run-busy"))],
            run_statuses={"run-busy": "started"},
            stamped_ids=["old"],
        )

        with pytest.raises(ExactNameScopeConflict, match="live"):
            _scope(transaction, ["old"])

        assert transaction.committed is False
        assert len(transaction.calls) == 2


class TestUnresolvedRunKeepsTheAgeRule:
    """Without a run to resolve, age remains the only available evidence."""

    def test_absent_run_row_leaves_an_old_claim_reclaimable(self) -> None:
        row = _row("orphan", _claim("orphan", claim_stale=True, run_id="run-missing"))
        assert _exact_name_scope_refusals([row], {}) == []

    def test_absent_run_row_keeps_a_fresh_claim_live(self) -> None:
        row = _row("orphan", _claim("orphan", claim_stale=False, run_id="run-missing"))
        refusals = _exact_name_scope_refusals([row], {})
        assert refusals, "an unresolvable run leaves the age rule to decide"
        assert "live" in refusals[0]

    def test_legacy_claim_without_a_run_id_still_uses_age(self) -> None:
        row = _row("legacy", _claim("legacy", claim_stale=True, run_id=None))
        assert _exact_name_scope_refusals([row]) == []

    def test_token_without_timestamp_refuses_even_with_a_stopped_run(self) -> None:
        """The orphan sweep never clears a token with no timestamp either."""
        row = _row(
            "token_only",
            _claim(
                "token_only",
                claimed_at=None,
                claim_token="opaque",
                run_id="run-done",
            ),
        )
        refusals = _exact_name_scope_refusals([row], {"run-done": "completed"})
        assert refusals
        assert "live" in refusals[0]


class TestNoClaimIsUntouched:
    """Widening the release rule leaves an unclaimed row alone."""

    def test_unclaimed_row_passes_without_a_run_read(self) -> None:
        transaction = _RecordingTransaction(
            [_row("clean", _claim("clean", claimed_at=None, claim_token=None))],
            stamped_ids=["clean"],
        )

        result = _scope(transaction, ["clean"])

        assert result["stamped"] == 1
        assert len(transaction.calls) == 2
        assert all(
            "EXACT_NAME_SCOPE_RUN_STATUS" not in query
            for query, _params in transaction.calls
        )


def test_an_unknown_run_status_holds_the_claim() -> None:
    """Only a run known to have stopped releases its claim.

    A status this reader does not recognise — a value written by a newer
    release, or a row whose run was never finalised — leaves the claim held,
    so the reading fails closed rather than reclaiming live work.
    """
    row = _row("odd", _claim("odd", claim_stale=True, run_id="run-odd"))
    refusals = _exact_name_scope_refusals([row], {"run-odd": "reconciling"})
    assert refusals, "an unrecognised status must not be read as terminal"

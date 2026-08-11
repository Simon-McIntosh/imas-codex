"""A cost limit too small to hold one review quorum must say so at startup.

Review cycles reserve their expected provider exposure before contacting
anything, and every replica of a review pool reserves against the same pot. A
limit below one quorum's reservation therefore produces deferrals rather than
spend, which reads as a reviewer or provider failure. The run is still allowed
to proceed — the operator may be scoping a names-only or local-model pass — so
the check warns and names both remedies instead of refusing.
"""

from __future__ import annotations

import logging

import pytest

from imas_codex.standard_names.loop import _warn_if_cost_limit_cannot_fund_a_quorum


def _warn_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.fixture
def paid_quorum(monkeypatch: pytest.MonkeyPatch) -> float:
    """Seat both review axes on priced doubles and return one quorum's cost."""
    from imas_codex.standard_names import loop

    seats = ["priced/one", "priced/two", "priced/three"]
    monkeypatch.setattr(loop, "_SIZING_PROBE_REQUEST_BYTES", 1_000)

    def _exposure(model, messages, **kwargs):
        return 0.5

    monkeypatch.setattr(
        "imas_codex.standard_names.budget.model_provider_exposure", _exposure
    )
    monkeypatch.setattr("imas_codex.settings.get_sn_review_names_models", lambda: seats)
    monkeypatch.setattr("imas_codex.settings.get_sn_review_docs_models", lambda: seats)
    monkeypatch.setattr("imas_codex.settings.get_pool_replicas", lambda pool: 8)
    return 1.5


def test_limit_below_one_quorum_warns_with_both_remedies(
    paid_quorum: float, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning names the limit, the headroom needed, and how to fix it."""
    caplog.set_level(logging.WARNING, logger="imas_codex.standard_names.loop")

    _warn_if_cost_limit_cannot_fund_a_quorum(paid_quorum - 0.5)

    messages = _warn_records(caplog)
    assert messages, "an unfundable quorum must warn"
    joined = "\n".join(messages)
    assert "review_docs" in joined and "review_name" in joined
    assert "$1.00 cannot fund" in joined  # the resolved limit
    assert "$1.50" in joined  # one quorum's reservation
    assert "--cost-limit" in joined  # remedy one: more money
    assert "IMAS_CODEX_SN_POOLS_REVIEW_DOCS_REPLICAS" in joined  # remedy two
    assert "$12.00" in joined  # 8 replicas x one quorum


def test_limit_covering_a_quorum_is_silent(
    paid_quorum: float, caplog: pytest.LogCaptureFixture
) -> None:
    """A workable limit must not add noise to every run."""
    caplog.set_level(logging.WARNING, logger="imas_codex.standard_names.loop")

    _warn_if_cost_limit_cannot_fund_a_quorum(paid_quorum)

    assert _warn_records(caplog) == []


def test_unlimited_budget_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    """A zero limit means unlimited (local, zero-cost routes), not unfundable."""
    caplog.set_level(logging.WARNING, logger="imas_codex.standard_names.loop")

    _warn_if_cost_limit_cannot_fund_a_quorum(0.0)

    assert _warn_records(caplog) == []


def test_unpriceable_seats_do_not_manufacture_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A route without price authority is refused at admission, not sized here."""
    from imas_codex.standard_names.budget import BudgetExposureUnknown

    def _unpriced(model, messages, **kwargs):
        raise BudgetExposureUnknown(f"no proven price for {model}")

    monkeypatch.setattr(
        "imas_codex.standard_names.budget.model_provider_exposure", _unpriced
    )
    monkeypatch.setattr(
        "imas_codex.settings.get_sn_review_names_models", lambda: ["unpriced/route"]
    )
    monkeypatch.setattr(
        "imas_codex.settings.get_sn_review_docs_models", lambda: ["unpriced/route"]
    )
    caplog.set_level(logging.WARNING, logger="imas_codex.standard_names.loop")

    _warn_if_cost_limit_cannot_fund_a_quorum(0.01)

    assert _warn_records(caplog) == []


def test_probe_failure_never_blocks_a_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The check is advisory: a broken probe must not raise into the run."""
    monkeypatch.setattr(
        "imas_codex.settings.get_sn_review_names_models",
        lambda: (_ for _ in ()).throw(RuntimeError("config unreadable")),
    )
    caplog.set_level(logging.WARNING, logger="imas_codex.standard_names.loop")

    _warn_if_cost_limit_cannot_fund_a_quorum(0.01)

    assert _warn_records(caplog) == []

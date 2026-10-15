"""One cost cap covers a whole family apply: anchor and members together.

``sn harmonize --apply`` runs one pattern family as an anchor batch followed by
a member batch. ``-c/--cost-limit`` bounds the family, not each batch: what the
anchor spends is subtracted from the cap, and the members are drained against
the remainder, so the family stops at the cap rather than each batch receiving
the whole cap independently.

These tests drive the real ``apply_families`` and the real ``CampaignRunner``
against a drained stub that records the cost limit it was handed per batch and
spends up to that limit, so what they assert is the orchestration, not a mock
of it. The declared negative control restores the per-batch cap — each batch
given the whole limit — which lets the member batch walk past the family cap;
the family-cap test then fails.
"""

from __future__ import annotations

from typing import Any

import pytest

_ANCHOR = "base_at_axis"
_MEMBERS = ["base_at_boundary", "base_of_probe"]
_CAP = 5.0


class _RecordingDrain:
    """A drain that records the cap it was called with and spends up to it.

    ``spend_requests`` gives, per call, how much the batch would spend if it
    were unbounded; the returned spend is clamped to the cap it received,
    standing in for the drain's own ``BudgetManager`` honouring its
    ``cost_limit`` and stopping when it is reached.
    """

    def __init__(self, spend_requests: list[float]):
        self.spend_requests = spend_requests
        self.calls: list[tuple[str, float]] = []

    def __call__(self, run_id: str, cost_limit: float) -> float | None:
        request = self.spend_requests[len(self.calls)]
        self.calls.append((run_id, cost_limit))
        return min(request, cost_limit)


def _plan() -> list[dict[str, Any]]:
    """One family: the anchor alone, then the remaining members."""
    return [
        {
            "physical_base": "base",
            "invariant": [],
            "anchor": _ANCHOR,
            "members": [_ANCHOR, *_MEMBERS],
            "n": len([_ANCHOR, *_MEMBERS]),
            "group_signature": "sig",
            "harmonized": False,
            "decision": "apply",
            "batches": [[_ANCHOR], sorted(_MEMBERS)],
            "planned_calls": len([_ANCHOR, *_MEMBERS]),
            "member_scores": {},
        }
    ]


def _runner_kwargs() -> dict[str, Any]:
    """Injected runner operations: no graph, every batch accepts."""

    def mark_fn(batch: list[str], dry_run: bool = False) -> dict[str, Any]:
        return {"run_id": f"scope-{batch[0]}"}

    def fetch_refreshed_fn(gc: Any, batch: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "id": sid,
                "name_stage": "accepted",
                "docs_stage": "accepted",
                "description": f"base quantity for {sid}",
                "documentation": f"Refreshed docs for {sid}.",
            }
            for sid in batch
        ]

    return {
        "mark_fn": mark_fn,
        "clear_quarantine_fn": lambda gc, batch: len(batch),
        "fetch_refreshed_fn": fetch_refreshed_fn,
        "revalidate_fn": lambda gc, reintroduced, clean: {
            "requarantined": 0,
            "confirmed": len(clean),
        },
        "record_change_fn": lambda gc, ids, run_id, spec: len(ids),
    }


def _install_graph_edges(monkeypatch: Any) -> None:
    """Replace the per-family graph edges apply_families touches with no-ops."""
    from imas_codex.standard_names import harmonize

    monkeypatch.setattr(harmonize, "_record_family_anchor", lambda *a, **k: None)
    monkeypatch.setattr(harmonize, "refuse_recommended_members", lambda *a, **k: {})
    monkeypatch.setattr(harmonize, "_stamp_applied_family", lambda *a, **k: True)
    monkeypatch.setattr(harmonize, "_family_docs_scores", lambda *a, **k: {})
    monkeypatch.setattr(
        harmonize, "reconcile_family_reverse_links", lambda *a, **k: {}
    )


def _apply(drain: _RecordingDrain):
    from imas_codex.standard_names.campaign import CampaignBudget
    from imas_codex.standard_names.harmonize import apply_families

    budget = CampaignBudget(per_batch_cost_cap=_CAP, campaign_cost_ceiling=_CAP)
    return apply_families(
        object(),
        seeds=[_ANCHOR],
        min_size=1,
        drain_fn=drain,
        plan=_plan(),
        budget=budget,
        runner_kwargs=_runner_kwargs(),
    )


def test_family_cap_is_shared_between_anchor_and_members(monkeypatch: Any) -> None:
    """A 60%-of-cap anchor leaves its members exactly the remaining 40%."""
    _install_graph_edges(monkeypatch)
    # The anchor spends 60% of the cap; the members would overspend if unbounded.
    drain = _RecordingDrain([0.6 * _CAP, 100.0])

    outcome = _apply(drain)

    # Two drains: the anchor batch, then the member batch.
    assert [call[0] for call in drain.calls] == [
        f"scope-{_ANCHOR}",
        f"scope-{_MEMBERS[0]}",
    ]
    # The anchor batch is offered the whole cap.
    assert drain.calls[0][1] == pytest.approx(_CAP)
    # The member batch is offered only what the anchor left: 40% of the cap.
    assert drain.calls[1][1] == pytest.approx(_CAP - 0.6 * _CAP)
    assert drain.calls[1][1] == pytest.approx(0.4 * _CAP)

    # The members stop at the cap: their spend is bounded by the remaining 40%,
    # not the 100.0 they would have spent.
    result = outcome["applied"][0]["result"]
    assert result.total_cost == pytest.approx(_CAP)
    assert result.halted is False


def test_family_cap_bounds_the_family_not_each_batch(monkeypatch: Any) -> None:
    """A batch that spends the whole cap leaves the family nothing more to spend."""
    _install_graph_edges(monkeypatch)
    drain = _RecordingDrain([100.0, 100.0])

    outcome = _apply(drain)

    # The anchor takes the whole cap; the member batch is never drained, so the
    # family cannot add a second batch's spend on top of it.
    assert len(drain.calls) == 1
    assert drain.calls[0][0] == f"scope-{_ANCHOR}"
    assert drain.calls[0][1] == pytest.approx(_CAP)
    result = outcome["applied"][0]["result"]
    assert result.halted is True
    assert result.total_cost == pytest.approx(_CAP)

"""The dry run estimates cost from measured spend, and the family cap is airtight.

``sn harmonize --apply --dry-run`` used to multiply a fixed constant by the
planned calls. These tests drive the dry run against recorded ``LLMCost``
rows — the documentation-seat spend of the regenerations and the reviews that
follow them — and assert the printed estimate lands within 30% of the cost a
stubbed apply then reports, and that with nothing recorded the dry run says so
rather than guessing.

They also drive the real ``apply_families`` and ``CampaignRunner`` against a
drain that cannot measure its own spend and returns ``None``, asserting the
runner's fallback cost is charged to the family's remaining budget so a batch
that cannot report its cost still counts against the family cap.

The declared negative control restores the fixed constant and drops the
fallback charge: the measured-estimate test then reads the constant's figure,
and the cap-leak test lets the second batch receive the whole cap.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from click.testing import CliRunner

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS

#: The family's anchor member id.
_ANCHOR = "base_at_axis"
#: The remaining members; with the anchor the fixture family has 17 members.
_MEMBERS = [f"base_at_member_{i:02d}" for i in range(16)]


def _node(
    node_id: str,
    docs_stage: str,
    score: float | None = None,
    *,
    name_stage: str = "drafted",
    physical_base: str = "base",
) -> dict[str, Any]:
    return {
        "id": node_id,
        "physical_base": physical_base,
        "description": f"base quantity for {node_id}",
        "documentation": f"The base quantity documented for {node_id}.",
        "docs_stage": docs_stage,
        "reviewer_score_docs": score,
        "name_stage": name_stage,
        "harmonized_group_signature": None,
    }


def _family_nodes() -> list[dict[str, Any]]:
    """One 17-member family: the anchor accepted, the members pending."""
    return [
        _node(_ANCHOR, "accepted", 0.9),
        *(_node(mid, "drafted", None) for mid in _MEMBERS),
    ]


def _invariant(node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(node.get(seg) or "") for seg in PATTERN_INVARIANT_SEGMENTS)


class _RecordedCostGraph:
    """GraphClient stand-in answering the family read and the LLMCost read.

    The family query is grouped by the fixture nodes' invariant grammar
    signature, as the real query groups them. A query naming ``LLMCost``
    returns *cost_rows* — the recorded documentation-seat spend the dry run
    measures — shaped as the reader aggregates it (one row per run, carrying
    the run's spend and the name lists its calls covered).
    """

    def __init__(
        self,
        nodes: list[dict[str, Any]],
        cost_rows: list[dict[str, Any]] | None = None,
    ):
        self.nodes = [dict(n) for n in nodes]
        self.cost_rows = list(cost_rows or [])
        self.queries: list[str] = []

    def __enter__(self) -> _RecordedCostGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if "LLMCost" in cypher:
            return [dict(r) for r in self.cost_rows]
        if "sn.name_stage AS name_stage" in cypher:
            names = set(params.get("names") or [])
            return [
                {"id": n["id"], "name_stage": n.get("name_stage")}
                for n in self.nodes
                if n["id"] in names
            ]
        if "RETURN m.id AS id" in cypher and "members" not in cypher:
            return [{"id": n["id"]} for n in self.nodes]
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for node in self.nodes:
            groups.setdefault(_invariant(node), []).append(node)
        seeds = set(params.get("seeds") or [])
        rows = []
        for key, members in groups.items():
            if seeds and not (seeds & {m["id"] for m in members}):
                continue
            row = dict(zip(PATTERN_INVARIANT_SEGMENTS, key, strict=True))
            row["members"] = [dict(m) for m in members]
            rows.append(row)
        return rows

    def writes(self) -> list[str]:
        return [
            q
            for q in self.queries
            if any(k in q for k in ("MERGE", "CREATE", "SET ", "DELETE"))
        ]


def _install_graph(monkeypatch: Any, graph: Any) -> None:
    """Point the CLI's ``GraphClient`` at *graph*."""
    from imas_codex.graph import client as graph_client

    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)


def _invoke_dry_run(graph: Any) -> Any:
    from imas_codex.cli.sn import sn

    return CliRunner().invoke(
        sn, ["harmonize", "--family", _ANCHOR, "--apply", "--dry-run"]
    )


def _printed_estimate(output: str) -> float:
    match = re.search(r"estimated cost \$([0-9]+\.[0-9]+)", output)
    assert match is not None, output
    return float(match.group(1))


# ── The stubbed apply ────────────────────────────────────────────────────────


class _RecordingDrain:
    """A drain that reports a fixed spend per batch and records its caps."""

    def __init__(self, spends: list[float]):
        self.spends = spends
        self.calls: list[tuple[str, float]] = []

    def __call__(self, run_id: str, cost_limit: float) -> float | None:
        spend = self.spends[len(self.calls)]
        self.calls.append((run_id, cost_limit))
        return spend


def _plan() -> list[dict[str, Any]]:
    """The 17-member family: the anchor alone, then the remaining members."""
    members = [_ANCHOR, *_MEMBERS]
    return [
        {
            "physical_base": "base",
            "invariant": [],
            "anchor": _ANCHOR,
            "members": members,
            "n": len(members),
            "group_signature": "sig",
            "harmonized": False,
            "decision": "apply",
            "batches": [[_ANCHOR], sorted(_MEMBERS)],
            "planned_calls": len(members),
            "member_scores": {},
        }
    ]


def _runner_kwargs() -> dict[str, Any]:
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


def _install_family_edges(monkeypatch: Any) -> None:
    """Replace the per-family graph edges apply_families touches with no-ops."""
    from imas_codex.standard_names import harmonize

    monkeypatch.setattr(harmonize, "_record_family_anchor", lambda *a, **k: None)
    monkeypatch.setattr(harmonize, "refuse_recommended_members", lambda *a, **k: {})
    monkeypatch.setattr(harmonize, "_stamp_applied_family", lambda *a, **k: True)
    monkeypatch.setattr(harmonize, "_family_docs_scores", lambda *a, **k: {})


def _run_stubbed_apply(drain: _RecordingDrain):
    from imas_codex.standard_names.campaign import CampaignBudget
    from imas_codex.standard_names.harmonize import apply_families

    budget = CampaignBudget(per_batch_cost_cap=100.0, campaign_cost_ceiling=100.0)
    return apply_families(
        object(),
        seeds=[_ANCHOR],
        min_size=1,
        drain_fn=drain,
        plan=_plan(),
        budget=budget,
        runner_kwargs=_runner_kwargs(),
    )


# ── Measured estimate ────────────────────────────────────────────────────────


def _recorded_cost_rows() -> list[dict[str, Any]]:
    """Two recorded runs documenting the fixture's 17 names for $5.10 total.

    A 16-name run at $0.30 per name and a one-name run at $0.30 per name, so
    the per-call mean is $0.30 and the 17-member estimate is $5.10.
    """
    return [
        {"run_id": "run-members", "run_cost": 4.80, "name_lists": [list(_MEMBERS)]},
        {"run_id": "run-anchor", "run_cost": 0.30, "name_lists": [[_ANCHOR]]},
    ]


def test_estimate_matches_applied_cost_from_recorded_calls(monkeypatch: Any) -> None:
    """The dry-run estimate lands within 30% of the cost the stubbed apply reports."""
    _install_family_edges(monkeypatch)
    graph = _RecordedCostGraph(_family_nodes(), _recorded_cost_rows())
    _install_graph(monkeypatch, graph)

    result = _invoke_dry_run(graph)
    assert result.exit_code == 0, result.output

    estimate = _printed_estimate(result.output)
    # The basis is printed: the runs read and the mean cost per call.
    out = " ".join(result.output.split())
    assert "2 recent run(s) and 17 call(s) at $0.30 per call" in out

    # The stubbed apply reports what the family's batches spent.
    drain = _RecordingDrain([0.30, 5.40])  # anchor batch, then members
    outcome = _run_stubbed_apply(drain)
    applied = outcome["applied"][0]["result"].total_cost
    assert applied == pytest.approx(0.30 + 5.40)

    assert abs(estimate - applied) / applied <= 0.30, (estimate, applied)


def test_estimate_states_no_measurement_when_none_recorded(monkeypatch: Any) -> None:
    """With no recorded per-call cost the dry run says so rather than guessing."""
    _install_family_edges(monkeypatch)
    graph = _RecordedCostGraph(_family_nodes(), [])
    _install_graph(monkeypatch, graph)

    result = _invoke_dry_run(graph)

    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "estimated cost unavailable" in out
    assert "no measured per-call cost recorded" in out
    # No dollar figure is printed as if it were measured.
    assert not re.search(r"estimated cost \$[0-9]", out)


# ── The family cap is airtight against an unmeasured batch ───────────────────


def test_fallback_batch_cost_counts_against_family_cap(monkeypatch: Any) -> None:
    """A batch that returns None, charged through cost_fn, counts against the cap."""
    _install_family_edges(monkeypatch)

    def cost_fn(run_id: str) -> float:
        return 3.0

    family_plan = _plan()
    family_plan[0]["batches"] = [[_ANCHOR], sorted(_MEMBERS)]

    drain = _RecordingDrain([None, 0.0])  # the anchor cannot measure its own spend

    from imas_codex.standard_names.campaign import CampaignBudget
    from imas_codex.standard_names.harmonize import apply_families

    budget = CampaignBudget(per_batch_cost_cap=5.0, campaign_cost_ceiling=5.0)
    outcome = apply_families(
        object(),
        seeds=[_ANCHOR],
        min_size=1,
        drain_fn=drain,
        plan=family_plan,
        budget=budget,
        runner_kwargs={**_runner_kwargs(), "cost_fn": cost_fn},
    )

    # The anchor batch could not measure itself, so cost_fn charged it $3.00 to
    # the family; the member batch is offered only the remaining $2.00, not the
    # whole $5.00 cap again.
    assert drain.calls[0][1] == pytest.approx(5.0)
    assert drain.calls[1][1] == pytest.approx(2.0)
    result = outcome["applied"][0]["result"]
    # The fallback charge reached the family total too: the unmeasurable anchor
    # batch's cost_fn figure is what the runner and the family both count.
    assert result.total_cost == pytest.approx(3.0)

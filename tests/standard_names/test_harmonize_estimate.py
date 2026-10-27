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
from imas_codex.standard_names.graph_store import (
    graph_foundation as _owning_graph_foundation,
)

#: The family's anchor member id.
_ANCHOR = "base_at_axis"
#: The remaining members; with the anchor the fixture family has 17 members.
_MEMBERS = [f"base_at_member_{i:02d}" for i in range(16)]
#: The one ceiling the family's batches draw on, so a cap assertion reads the
#: budget rather than restating a literal.
_FAMILY_CEILING = 100.0


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
        "has_dd_source": True,
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
        run_spend: dict[str, float] | None = None,
    ):
        self.nodes = [dict(n) for n in nodes]
        self.cost_rows = list(cost_rows or [])
        #: Spend recorded per run id, for the runner's fallback cost hook.
        self.run_spend = dict(run_spend or {})
        self.queries: list[str] = []

    def __enter__(self) -> _RecordedCostGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if "AS total" in cypher and "run_id" in cypher:
            run_id = params.get("run_id")
            return [{"total": float(self.run_spend.get(run_id, 0.0))}]
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
    monkeypatch.setattr(harmonize, "reconcile_family_reverse_links", lambda *a, **k: {})


def _run_stubbed_apply(drain: _RecordingDrain):
    from imas_codex.standard_names.campaign import CampaignBudget
    from imas_codex.standard_names.harmonize import apply_families

    budget = CampaignBudget(
        per_batch_cost_cap=_FAMILY_CEILING, campaign_cost_ceiling=_FAMILY_CEILING
    )
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


#: The global row cap the read once applied before it grouped by run: a run
#: with more rows than this was cut mid-run, so its spend and its name list
#: were reported for only the rows the cap happened to keep. Retired from the
#: read; kept here so the whole-run test can be shown to fail against it.
_LEGACY_ROW_CAP = 4000
#: More rows than the retired cap, in one run.
_OVERSIZED_ROWS = _LEGACY_ROW_CAP + 1


class _LedgerGraph:
    """A graph stub answering the LLMCost read from raw ledger rows.

    It aggregates the rows the way the query asks. A query that caps the row
    stream *before* it groups by run admits only the newest ``row_cap`` rows,
    so a run whose own rows exceed the cap is cut. A query that selects the
    recent run ids first, bounded, then aggregates every row of those runs
    reports each run whole. The two branches are keyed on the query's own
    shape, so the read's structure — not the caller — decides the answer.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = sorted(rows, key=lambda r: r["llm_at"], reverse=True)
        self.queries: list[str] = []

    def __enter__(self) -> _LedgerGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if "LLMCost" not in cypher:
            return []
        rows = list(self.rows)
        if "$run_cap" in cypher:
            order: list[str] = []
            for row in rows:
                if row["run_id"] not in order:
                    order.append(row["run_id"])
            selected = set(order[: params.get("run_cap", len(order))])
            rows = [row for row in rows if row["run_id"] in selected]
        else:
            rows = rows[:_LEGACY_ROW_CAP]
        aggregated: dict[str, dict[str, Any]] = {}
        for row in rows:
            slot = aggregated.setdefault(
                row["run_id"], {"run_cost": 0.0, "name_lists": [], "rows": 0}
            )
            slot["run_cost"] += float(row["llm_cost"])
            slot["name_lists"].append(list(row["standard_name_ids"]))
            slot["rows"] += 1
        return [{"run_id": key, **slot} for key, slot in aggregated.items()]


def _oversized_run_ledger() -> list[dict[str, Any]]:
    """One run with more LLMCost rows than the retired cap, plus a small one.

    The big run bills $2.00 a row across ``_OVERSIZED_ROWS`` rows — a spend a
    global row cap cuts in half. The small run is older, so a row cap applied
    over the whole stream drops it entirely.
    """
    rows = [
        {
            "run_id": "run-big",
            "llm_cost": 2.0,
            "standard_name_ids": [_ANCHOR],
            "llm_at": f"2026-10-14T00:00:{i % 60:02d}Z",
        }
        for i in range(_OVERSIZED_ROWS)
    ]
    rows.append(
        {
            "run_id": "run-small",
            "llm_cost": 0.5,
            "standard_name_ids": [_MEMBERS[0]],
            "llm_at": "2026-10-13T00:00:00Z",
        }
    )
    return rows


def test_estimate_counts_a_run_whose_rows_exceed_the_row_cap() -> None:
    """A run with more rows than the row cap is counted whole, not cut."""
    from imas_codex.standard_names.harmonize import measure_doc_call_cost

    rows = _oversized_run_ledger()
    graph = _LedgerGraph(rows)

    estimate = measure_doc_call_cost(graph)

    assert estimate is not None
    # The big run's whole spend is counted: 4001 rows at $2.00, plus the small
    # run's $0.5 and the two names the two runs documented between them.
    assert estimate.total_usd == pytest.approx(2.0 * _OVERSIZED_ROWS + 0.5)
    assert estimate.runs == 2
    assert estimate.calls == 2
    # The read names the runs first and caps the run count, not the row stream.
    assert "$run_cap" in graph.queries[-1]


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

    # The stubbed apply runs the family's two planned batches against one
    # ceiling, in plan order. Assert the cap arithmetic the code computed —
    # each batch is offered the ceiling less what the earlier batch charged —
    # rather than restating the spends the drain was handed.
    drain = _RecordingDrain([0.30, 5.40])  # anchor batch, then members
    outcome = _run_stubbed_apply(drain)
    assert [run_id for run_id, _cap in drain.calls] == [
        f"scope-{_ANCHOR}",
        f"scope-{_MEMBERS[0]}",
    ]
    assert drain.calls[0][1] == pytest.approx(_FAMILY_CEILING)
    assert drain.calls[1][1] == pytest.approx(_FAMILY_CEILING - drain.spends[0])

    applied = outcome["applied"][0]["result"].total_cost
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
    # The fallback charge reached the family total. Assert the total against
    # the charge the hook returns for the anchor's run id — not against the
    # literal the test handed it — so the assertion reads the code's own
    # figure and the family-cap arithmetic beneath it.
    anchor_charge = cost_fn(f"scope-{_ANCHOR}")
    assert result.total_cost == pytest.approx(anchor_charge)
    # The member batch was offered the ceiling less that charge, so the second
    # cap is the code's arithmetic over the anchor's recorded spend.
    assert drain.calls[1][1] == pytest.approx(5.0 - anchor_charge)


# ── The CLI wires a measuring fallback, not the 0.00 stub ────────────────────


def test_cli_apply_wires_a_measuring_cost_fn(monkeypatch: Any) -> None:
    """The CLI's harmonize apply hands the runner a cost_fn that measures spend.

    The fallback is only as good as the hook the CLI passes: the runner's
    default is the ``default_cost`` stub, which returns 0.00, so an
    unmeasurable batch would be charged nothing to the family cap. The CLI
    must inject the recorded-spend hook the campaign path already uses.
    """
    from imas_codex.standard_names import graph_ops
    from imas_codex.standard_names.campaign import default_cost

    graph = _RecordedCostGraph(_family_nodes(), [], run_spend={f"scope-{_ANCHOR}": 2.5})
    _install_graph(monkeypatch, graph)
    # ``aggregate_spend_for_run`` resolves GraphClient from its own module.
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda *a, **k: graph)

    from imas_codex.standard_names import harmonize

    captured: dict[str, Any] = {}

    def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "plans": [],
            "applied": [],
            "recommended_guards": {},
            "force_reason": None,
            "force_run_id": None,
        }

    monkeypatch.setattr(harmonize, "apply_families", spy)

    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(sn, ["harmonize", "--family", _ANCHOR, "--apply"])
    assert result.exit_code == 0, result.output

    cost_fn = (captured.get("runner_kwargs") or {}).get("cost_fn")
    assert cost_fn is not None, (
        "the CLI passed no cost_fn; the family falls back to the 0.00 stub"
    )
    assert cost_fn is not default_cost
    # It measures: the run's recorded spend comes back, not the stub's 0.00.
    assert cost_fn(f"scope-{_ANCHOR}") == pytest.approx(2.5)

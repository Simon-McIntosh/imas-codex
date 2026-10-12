"""`sn harmonize --apply` drains a pattern family anchor-first, then members.

One pattern family, applied through a single campaign per family: the anchor is
drained as its own batch and must reach documentation acceptance before the
members are touched. The tests drive the real ``CampaignRunner`` and the real
``mark_members_for_regen`` against a fixture graph whose docs pipeline is
stubbed, so what they assert is the orchestration, not a mock of it:

* a failed anchor leaves the members untouched — no reset, no snapshot — and
  removing the anchor acceptance gate makes that test fail (the negative
  control logged beside the run);
* an accepted anchor is followed by exactly one member batch scoped to the
  other members;
* the snapshot is taken for every reset name before its reset, so the family
  restores from it;
* a repeated apply over an unchanged scope plans zero calls;
* ``--dry-run`` writes nothing and prints the planned calls and cost.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from click.testing import CliRunner

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS

_ANCHOR = "base_at_axis"
_MEMBERS = ["base_at_boundary", "base_of_probe"]


def _node(node_id: str, docs_stage: str, score: float | None) -> dict[str, Any]:
    return {
        "id": node_id,
        "physical_base": "base",
        "description": f"base quantity for {node_id}",
        "documentation": f"The base quantity documented for {node_id}.",
        "docs_stage": docs_stage,
        "reviewer_score_docs": score,
        "harmonized_group_signature": None,
    }


def _family_nodes() -> list[dict[str, Any]]:
    return [
        _node(_ANCHOR, "accepted", 0.9),
        _node(_MEMBERS[0], "drafted", None),
        _node(_MEMBERS[1], "drafted", None),
    ]


def _invariant(node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(node.get(seg) or "") for seg in PATTERN_INVARIANT_SEGMENTS)


class _FamilyGraph:
    """GraphClient stand-in answering the pattern-family read.

    Groups the fixture nodes by their invariant grammar signature — the
    query's own notion of a family — and records every query so a caller can
    assert none of them wrote.
    """

    def __init__(self, nodes: list[dict[str, Any]]):
        self.nodes = [dict(n) for n in nodes]
        self.queries: list[str] = []

    def __enter__(self) -> _FamilyGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
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


class _DocsPipeline:
    """The fixture docs graph: per-name documentation plus a revision stack.

    ``reset_standard_name_docs`` is replaced with :meth:`reset`, which pushes
    the name's current documentation onto ``revisions`` *before* clearing it —
    exactly the snapshot-before-reset order the real operation guarantees.
    """

    def __init__(self, nodes: list[dict[str, Any]]):
        self.docs = {n["id"]: n["documentation"] for n in nodes}
        self.revisions: dict[str, list[str]] = {}

    def reset(
        self, *, sn_ids: list[str], run_id: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        out = {"eligible": len(sn_ids), "reset": 0}
        for sid in sn_ids:
            self.revisions.setdefault(sid, []).append(self.docs.get(sid, ""))
            if not dry_run:
                self.docs[sid] = ""
                out["reset"] += 1
        return out


class _ApplyHarness:
    """Injected runner operations recording the batches and drain order."""

    def __init__(self, accepted_after: set[str], pipeline: _DocsPipeline):
        self.accepted_after = accepted_after
        self.pipeline = pipeline
        self.marked_batches: list[list[str]] = []
        self.drained: list[str] = []

    def mark_fn(self, batch: list[str], dry_run: bool = False) -> dict[str, Any]:
        """Record the batch, then run the real snapshot+reset+stamp."""
        from imas_codex.standard_names.harmonize import mark_members_for_regen

        self.marked_batches.append(list(batch))
        return mark_members_for_regen(batch, dry_run=dry_run)

    def clear_quarantine_fn(self, gc: Any, batch: list[str]) -> int:
        return len(batch)

    def drain_fn(self, run_id: str, cost_limit: float) -> float | None:
        self.drained.append(run_id)
        return 0.0

    def fetch_refreshed_fn(self, gc: Any, batch: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "id": sid,
                "name_stage": "accepted",
                "docs_stage": "accepted" if sid in self.accepted_after else "drafted",
                "description": f"refreshed {sid}",
                "documentation": f"Refreshed docs for {sid}.",
            }
            for sid in batch
        ]

    def revalidate_fn(self, gc: Any, reintroduced: Any, clean: Any) -> dict[str, int]:
        return {"requarantined": 0, "confirmed": len(clean)}

    def record_change_fn(self, gc: Any, ids: Any, run_id: str, spec: Any) -> int:
        return len(ids)

    def cost_fn(self, run_id: str) -> float:
        return 0.0


def _apply(gc: _FamilyGraph, harness: _ApplyHarness, *, thresholds: Any = None):
    from imas_codex.standard_names.harmonize import apply_families

    return apply_families(
        gc,
        seeds=[_ANCHOR],
        min_size=1,
        drain_fn=harness.drain_fn,
        thresholds=thresholds,
        runner_kwargs={
            "mark_fn": harness.mark_fn,
            "clear_quarantine_fn": harness.clear_quarantine_fn,
            "fetch_refreshed_fn": harness.fetch_refreshed_fn,
            "revalidate_fn": harness.revalidate_fn,
            "record_change_fn": harness.record_change_fn,
            "cost_fn": harness.cost_fn,
        },
    )


def _install_docs_pipeline(monkeypatch: Any, pipeline: _DocsPipeline) -> None:
    from imas_codex.standard_names import graph_ops

    monkeypatch.setattr(graph_ops, "reset_standard_name_docs", pipeline.reset)
    monkeypatch.setattr(
        graph_ops, "stamp_harmonized_families", lambda families: len(families)
    )


def _thresholds_or_gate_removed() -> Any:
    """The convergence gate, or the same gate with its anchor check removed.

    ``SHC_NO_ANCHOR_GATE`` sets the acceptance floor to zero, which is the
    declared negative control: it removes the anchor acceptance gate, so the
    member batch runs even though the anchor did not converge and the
    untouched-members test fails.
    """
    from imas_codex.standard_names.campaign import ConvergenceThresholds

    if os.environ.get("SHC_NO_ANCHOR_GATE"):
        return ConvergenceThresholds(min_docs_accept_rate=0.0)
    return ConvergenceThresholds()


def test_failed_anchor_leaves_members_untouched(monkeypatch: Any) -> None:
    """The anchor batch fails its gate, so no member is reset or snapshotted."""
    nodes = _family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    harness = _ApplyHarness(accepted_after=set(), pipeline=pipeline)
    graph = _FamilyGraph(nodes)

    outcome = _apply(graph, harness, thresholds=_thresholds_or_gate_removed())

    entry = outcome["applied"][0]
    assert entry["applied"] is True
    assert entry["result"].halted is True
    # Only the anchor was marked (and so reset); the members never were.
    assert harness.marked_batches == [[_ANCHOR]]
    assert set(pipeline.revisions) == {_ANCHOR}
    for member in _MEMBERS:
        assert member not in pipeline.revisions
        # ...and the member documentation is exactly as it was.
        assert pipeline.docs[member] == f"The base quantity documented for {member}."


def test_accepted_anchor_then_one_member_batch(monkeypatch: Any) -> None:
    """An accepted anchor converges, then one batch scoped to the members."""
    nodes = _family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    harness = _ApplyHarness(accepted_after={_ANCHOR, *_MEMBERS}, pipeline=pipeline)
    graph = _FamilyGraph(nodes)

    outcome = _apply(graph, harness, thresholds=_thresholds_or_gate_removed())

    entry = outcome["applied"][0]
    assert entry["result"].halted is False
    assert harness.marked_batches == [[_ANCHOR], sorted(_MEMBERS)]
    assert outcome["applied"][0]["stamped"] is True


def test_snapshot_precedes_every_reset_and_restores_the_family(
    monkeypatch: Any,
) -> None:
    """Every reset name has its pre-reset text snapshotted, restorable after."""
    nodes = _family_nodes()
    original = {n["id"]: n["documentation"] for n in nodes}
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    harness = _ApplyHarness(accepted_after={_ANCHOR, *_MEMBERS}, pipeline=pipeline)
    graph = _FamilyGraph(nodes)

    _apply(graph, harness, thresholds=_thresholds_or_gate_removed())

    for node_id in [_ANCHOR, *_MEMBERS]:
        # A snapshot was pushed before the reset, holding the original text.
        assert pipeline.revisions[node_id][0] == original[node_id]
        # ...and the reset then cleared it.
        assert pipeline.docs[node_id] == ""
    restored = {sid: revs[0] for sid, revs in pipeline.revisions.items()}
    assert restored == original


def test_second_apply_over_unchanged_scope_plans_zero_calls(
    monkeypatch: Any,
) -> None:
    """The stamped family is harmonized; a repeat apply plans nothing."""
    from imas_codex.standard_names.harmonize import (
        group_signature,
        plan_family_applies,
    )

    nodes = _family_nodes()
    signature = group_signature(nodes)
    for node in nodes:
        node["harmonized_group_signature"] = signature
    graph = _FamilyGraph(nodes)

    plans = plan_family_applies(graph, seeds=[_ANCHOR], min_size=1)

    assert len(plans) == 1
    assert plans[0]["harmonized"] is True
    assert plans[0]["decision"] == "harmonized"
    assert plans[0]["planned_calls"] == 0
    assert plans[0]["batches"] == []


def test_dry_run_prints_the_plan_and_writes_nothing(monkeypatch: Any) -> None:
    """`--apply --dry-run` reports the planned calls and cost, and writes nothing."""
    from imas_codex.cli.sn import sn
    from imas_codex.graph import client as graph_client

    nodes = _family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    graph = _FamilyGraph(nodes)
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)

    result = CliRunner().invoke(
        sn, ["harmonize", "--family", _ANCHOR, "--apply", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "planned call(s)" in result.output
    assert "estimated cost" in result.output
    assert graph.writes() == []
    assert pipeline.revisions == {}

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
* ``--dry-run`` writes nothing and prints the planned calls and cost;
* over a scope of several families, one whose member is refused makes the dry
  run report the whole apply as refused — zero calls, zero cost, the refusing
  family named and the others not reported as running — with the per-family
  refusal under ``--force``;
* the apply compares each member's documentation-axis score before the reset
  against its post-drain score and names, with both scores, every member whose
  score fell below acceptance and every member that fell — a family in which
  no score fell names none, and removing the comparison makes the
  below-threshold test fail (the second negative control logged beside the run).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from click.testing import CliRunner

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS
from imas_codex.standard_names.graph_store import (
    graph_docs_persistence as _owning_graph_docs_persistence,
)

_ANCHOR = "base_at_axis"
_MEMBERS = ["base_at_boundary", "base_of_probe"]


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

    def __init__(
        self,
        accepted_after: set[str],
        pipeline: _DocsPipeline,
        *,
        graph: _FamilyGraph | None = None,
        after_scores: dict[str, float | None] | None = None,
    ):
        self.accepted_after = accepted_after
        self.pipeline = pipeline
        self.graph = graph
        self.after_scores = after_scores or {}
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
        """Drain the scope, standing in for the review rewriting the docs.

        The stub review writes each name's post-refresh documentation score
        onto the fixture graph, so the apply's post-drain read sees what a
        real review would have recorded.
        """
        self.drained.append(run_id)
        if self.graph is not None:
            for node in self.graph.nodes:
                if node["id"] in self.after_scores:
                    node["reviewer_score_docs"] = self.after_scores[node["id"]]
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

    monkeypatch.setattr(
        _owning_graph_docs_persistence, "reset_standard_name_docs", pipeline.reset
    )
    monkeypatch.setattr(
        _owning_graph_docs_persistence,
        "stamp_harmonized_families",
        lambda families: len(families),
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


def _invoke_dry_run(
    monkeypatch: Any, graph: _FamilyGraph, pipeline: _DocsPipeline, *extra: str
):
    """Run ``harmonize --family <anchor> --apply --dry-run`` against the fixture."""
    from imas_codex.cli.sn import sn
    from imas_codex.graph import client as graph_client

    _install_docs_pipeline(monkeypatch, pipeline)
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)
    return CliRunner().invoke(
        sn, ["harmonize", "--family", _ANCHOR, "--apply", "--dry-run", *extra]
    )


def test_dry_run_refuses_a_family_holding_a_recommended_member(
    monkeypatch: Any,
) -> None:
    """A protected member refuses its family and drops its calls from the total.

    Names it and its evidence, writes nothing.
    """
    nodes = _family_nodes()
    nodes[1]["name_stage"] = "approved"  # the first non-anchor member
    pipeline = _DocsPipeline(nodes)
    graph = _FamilyGraph(nodes)

    result = _invoke_dry_run(monkeypatch, graph, pipeline)

    assert result.exit_code == 0, result.output
    assert "decision=refused" in result.output
    assert _MEMBERS[0] in result.output
    assert "name_stage=approved" in result.output
    # The refused family's calls are excluded: 3 members, 0 planned calls.
    assert "0 planned call(s)" in result.output
    assert graph.writes() == []
    assert pipeline.revisions == {}


def test_dry_run_counts_calls_for_a_forced_recommended_member(
    monkeypatch: Any,
) -> None:
    """With --force the protected family is kept and its calls are counted."""
    nodes = _family_nodes()
    nodes[1]["name_stage"] = "approved"
    pipeline = _DocsPipeline(nodes)
    graph = _FamilyGraph(nodes)

    result = _invoke_dry_run(
        monkeypatch, graph, pipeline, "--force", "lead authorised the density family"
    )

    assert result.exit_code == 0, result.output
    assert "decision=forced" in result.output
    assert _MEMBERS[0] in result.output
    assert "name_stage=approved" in result.output
    assert "3 planned call(s)" in result.output
    assert graph.writes() == []
    assert pipeline.revisions == {}


#: Two distinct pattern families (their invariant physical_base differs), so a
#: refusal in one can be measured against the other's planned calls.
_TWO_FAMILIES = [
    ("base_a", "a_at_axis", "a_at_boundary"),
    ("base_b", "b_at_axis", "b_at_boundary"),
]


def _two_family_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for base, anchor, member in _TWO_FAMILIES:
        nodes.append(_node(anchor, "accepted", 0.9, physical_base=base))
        nodes.append(_node(member, "drafted", physical_base=base))
    return nodes


def _invoke_dry_run_all(
    monkeypatch: Any, graph: _FamilyGraph, pipeline: _DocsPipeline, *extra: str
):
    """Run ``harmonize --all --apply --dry-run`` over the whole fixture."""
    from imas_codex.cli.sn import sn
    from imas_codex.graph import client as graph_client

    _install_docs_pipeline(monkeypatch, pipeline)
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)
    return CliRunner().invoke(
        sn, ["harmonize", "--all", "--apply", "--dry-run", *extra]
    )


def test_dry_run_reports_the_whole_apply_refused(monkeypatch: Any) -> None:
    """One refused family refuses the whole apply, not just its own calls.

    Over two families, one holding a protected member, the summary reports zero
    planned calls and zero cost and names the refusing family; the other family
    is not reported as running; nothing is written.
    """
    nodes = _two_family_nodes()
    nodes[1]["name_stage"] = "approved"  # the first base_a member
    pipeline = _DocsPipeline(nodes)
    graph = _FamilyGraph(nodes)

    result = _invoke_dry_run_all(monkeypatch, graph, pipeline)

    assert result.exit_code == 0, result.output
    # The refused family is named with its calls withheld.
    assert "decision=refused" in result.output
    assert "base_a" in result.output
    # The other family would not run either — it is not reported as applying.
    assert "decision=apply (blocked by refusal)" in result.output
    assert "base_b" in result.output
    assert "decision=apply  anchor" not in result.output
    # The apply as a whole is refused: zero calls, zero cost.
    assert "0 planned call(s)" in result.output
    assert "estimated cost $0.00" in result.output
    assert graph.writes() == []
    assert pipeline.revisions == {}


def test_dry_run_counts_both_families_under_force(monkeypatch: Any) -> None:
    """With --force both families count: the protected one is forced, not blocked."""
    nodes = _two_family_nodes()
    nodes[1]["name_stage"] = "approved"
    pipeline = _DocsPipeline(nodes)
    graph = _FamilyGraph(nodes)

    result = _invoke_dry_run_all(
        monkeypatch, graph, pipeline, "--force", "lead authorised the family"
    )

    assert result.exit_code == 0, result.output
    assert "decision=forced" in result.output
    assert "decision=blocked" not in result.output
    assert "4 planned call(s)" in result.output
    assert graph.writes() == []
    assert pipeline.revisions == {}


# ── Documentation-score regression report ───────────────────────────────────


def _regressions_or_removed(monkeypatch: Any) -> None:
    """Disable the score comparison when the declared mutation is set.

    The declared negative control removes the before/after comparison, so the
    below-threshold member is never named and the assertions in
    :func:`test_apply_names_a_member_below_threshold` fail.
    """
    if os.environ.get("SHC_NO_SCORE_COMPARISON"):
        from imas_codex.standard_names import harmonize

        monkeypatch.setattr(
            harmonize,
            "docs_score_regressions",
            lambda before, after, **kw: {"below_threshold": [], "fell": []},
        )


def _scored_family_nodes() -> list[dict[str, Any]]:
    """A family whose members all start accepted and scored."""
    return [
        _node(_ANCHOR, "accepted", 0.90, name_stage="accepted"),
        _node(_MEMBERS[0], "accepted", 0.90, name_stage="accepted"),
        _node(_MEMBERS[1], "accepted", 0.88, name_stage="accepted"),
    ]


def test_apply_names_a_member_below_threshold(monkeypatch: Any) -> None:
    """A member whose rewritten docs score below threshold is named with both scores."""
    nodes = _scored_family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    graph = _FamilyGraph(nodes)
    harness = _ApplyHarness(
        accepted_after={_ANCHOR, *_MEMBERS},
        pipeline=pipeline,
        graph=graph,
        after_scores={_MEMBERS[0]: 0.60},
    )
    _regressions_or_removed(monkeypatch)

    outcome = _apply(graph, harness)

    regressions = outcome["applied"][0]["score_regressions"]
    below = regressions["below_threshold"]
    assert [i["id"] for i in below] == [_MEMBERS[0]]
    assert below[0]["before"] == 0.90
    assert below[0]["after"] == 0.60
    fell = regressions["fell"]
    assert [i["id"] for i in fell] == [_MEMBERS[0]]
    assert fell[0]["before"] == 0.90
    assert fell[0]["after"] == 0.60


def test_apply_reports_no_fall_when_no_member_scores_worse(
    monkeypatch: Any,
) -> None:
    """A family in which no score fell reports neither below threshold nor fell."""
    nodes = _scored_family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    graph = _FamilyGraph(nodes)
    harness = _ApplyHarness(
        accepted_after={_ANCHOR, *_MEMBERS},
        pipeline=pipeline,
        graph=graph,
        after_scores={_MEMBERS[0]: 0.92, _MEMBERS[1]: 0.88},
    )
    _regressions_or_removed(monkeypatch)

    outcome = _apply(graph, harness)

    regressions = outcome["applied"][0]["score_regressions"]
    assert regressions["below_threshold"] == []
    assert regressions["fell"] == []


def test_score_comparison_reports_below_threshold_and_fall() -> None:
    """The comparison classifies a fell-below member and ignores unscored ones."""
    from imas_codex.standard_names.harmonize import docs_score_regressions

    regressions = docs_score_regressions(
        {"below_and_fell": 0.90, "rose": 0.60, "fell_only": 0.95, "same": 0.90},
        {"below_and_fell": 0.60, "rose": 0.70, "fell_only": 0.90, "same": 0.90},
    )

    assert [i["id"] for i in regressions["below_threshold"]] == [
        "below_and_fell",
        "rose",
    ]
    assert [i["id"] for i in regressions["fell"]] == ["below_and_fell", "fell_only"]


def test_apply_report_names_member_in_cli_output(monkeypatch: Any) -> None:
    """`--apply` prints the below-threshold member with both scores."""
    from imas_codex.cli.sn import sn
    from imas_codex.graph import client as graph_client
    from imas_codex.standard_names import harmonize

    class _Result:
        halted = False
        batches_run = 2
        total_accepted = 3
        total_touched = 3
        halt_reasons: list[str] = []

        def summary(self) -> dict[str, Any]:
            return {"halted": False}

    crafted = {
        "applied": [
            {
                "physical_base": "base",
                "decision": "apply",
                "anchor": _ANCHOR,
                "applied": True,
                "stamped": True,
                "result": _Result(),
                "score_regressions": {
                    "below_threshold": [
                        {"id": _MEMBERS[0], "before": 0.90, "after": 0.60}
                    ],
                    "fell": [{"id": _MEMBERS[0], "before": 0.90, "after": 0.60}],
                },
            }
        ]
    }
    graph = _FamilyGraph([])
    monkeypatch.setattr(harmonize, "plan_family_applies", lambda *a, **k: [])
    monkeypatch.setattr(harmonize, "apply_families", lambda *a, **k: crafted)
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)

    result = CliRunner().invoke(sn, ["harmonize", "--family", _ANCHOR, "--apply"])

    assert result.exit_code == 0, result.output
    assert _MEMBERS[0] in result.output
    assert "0.900" in result.output
    assert "0.600" in result.output

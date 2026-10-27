"""A harmonize run keeps its anchor until its family is stamped harmonized.

A family's apply resets its anchor first, so a runner interrupted between the
anchor's reset and the family's stamp leaves the pin mid-pipeline with a null
documentation score on that member. A plan re-derived in that window would rank
the members afresh and pick a different one, discarding the member the run
already started on. The recorded anchor keeps the run pointed at its own choice
until the family is stamped, at which point it is released and the next apply
selects normally.

The pin has one release, the family's harmonization stamp. A forced apply waives
the recommended-name refusal for the names it rewrites; it does not move the run
onto a different member.

Each negative control this file's tests are paired with is shown by mutating the
source temporarily and re-running the test, not by a switch the test reads from
its own environment: a test that can turn its own fixture off proves the fixture
is running, not that the code works without it. The mutated runs and their logs
are recorded with the run's manifest.
"""

from __future__ import annotations

import importlib
from typing import Any

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS
from imas_codex.standard_names.graph_store import (
    graph_claim_scope as _owning_graph_claim_scope,
    graph_docs_persistence as _owning_graph_docs_persistence,
)

_ANCHOR = "base_at_axis"
_OUTSCORER = "base_at_boundary"
_OWNED_NAME_QUERY = "MATCH (sn:StandardName) RETURN sn.id AS id"


def _harmonize() -> Any:
    return importlib.import_module("imas_codex.standard_names.harmonize")


def _node(
    node_id: str,
    docs_stage: str,
    score: float | None,
    *,
    anchor: str | None = None,
    signature: str | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "physical_base": "base",
        "physics_domain": "equilibrium",
        "description": f"base quantity for {node_id}",
        "documentation": f"The base quantity documented for {node_id}.",
        "docs_stage": docs_stage,
        "reviewer_score_docs": score,
        "name_stage": "drafted",
        "harmonized_group_signature": signature,
        "harmonized_anchor": anchor,
        "links": [],
        "has_dd_source": True,
    }


def _family_read(monkeypatch: Any, nodes: list[dict[str, Any]]) -> None:
    family = {
        "physical_base": "base",
        "invariant": ["" for _ in PATTERN_INVARIANT_SEGMENTS],
        "n": len(nodes),
        "members": nodes,
    }
    monkeypatch.setattr(
        _harmonize(), "fetch_pattern_families", lambda gc, **_kw: [family]
    )


class _Graph:
    """A client stub that mutates the fixture nodes for the pin queries."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.nodes = nodes

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        ids = set(params.get("ids") or [])
        rows = [n for n in self.nodes if n["id"] in ids]
        if "REMOVE sn.harmonized_anchor" in cypher:
            for row in rows:
                row["harmonized_anchor"] = None
        elif "SET sn.harmonized_anchor" in cypher:
            for row in rows:
                row["harmonized_anchor"] = params.get("anchor")
        return []


def test_recorded_anchor_reused_when_another_member_outscores_it(
    monkeypatch: Any,
) -> None:
    """A family mid-run keeps its recorded anchor over a higher-scoring member."""
    nodes = [
        _node(_ANCHOR, "drafted", None, anchor=_ANCHOR),
        _node(_OUTSCORER, "accepted", 0.9, anchor=_ANCHOR),
    ]
    _family_read(monkeypatch, nodes)

    plans = _harmonize().plan_family_applies(_Graph(nodes), min_size=1)

    assert plans[0]["decision"] == "apply"
    assert plans[0]["anchor"] == _ANCHOR, "the recorded anchor must be reused"


def test_stamped_family_releases_the_anchor_and_selects_normally(
    monkeypatch: Any,
) -> None:
    """A converged family releases its anchor, so the next plan reranks."""
    from imas_codex.standard_names import graph_ops

    monkeypatch.setattr(
        _owning_graph_docs_persistence,
        "stamp_harmonized_families",
        lambda families: len(families),
    )
    harmonize = _harmonize()
    nodes = [
        _node(_ANCHOR, "drafted", None, anchor=_ANCHOR),
        _node(_OUTSCORER, "accepted", 0.9, anchor=_ANCHOR),
    ]
    _family_read(monkeypatch, nodes)
    graph = _Graph(nodes)

    assert harmonize._stamp_applied_family(graph, _ANCHOR) is True

    plans = harmonize.plan_family_applies(graph, min_size=1)
    assert plans[0]["anchor"] == _OUTSCORER, "rerank once the anchor is released"


def test_forced_apply_reuses_the_recorded_anchor(monkeypatch: Any) -> None:
    """A force reason waives the recommended-name refusal, not the pin.

    The apply derives its own plan when none is handed in, so a force reason
    reaches the pin: a family held mid-run by a recorded anchor must still
    record that anchor under force, or a forced re-apply would walk the family
    onto another member.
    """
    harmonize = _harmonize()
    nodes = [
        _node(_ANCHOR, "drafted", None, anchor=_ANCHOR),
        _node(_OUTSCORER, "accepted", 0.9, anchor=_ANCHOR),
    ]
    _family_read(monkeypatch, nodes)

    class _Halted:
        halted = True

        def run(self, **_kw: Any) -> _Halted:
            return self

    monkeypatch.setattr(
        harmonize,
        "refuse_recommended_members",
        lambda gc, targets, force=None, reviews_dir=None: {},
    )
    monkeypatch.setattr(harmonize, "apply_guard_targets", lambda plans: [])
    recorded: list[Any] = []
    monkeypatch.setattr(
        harmonize,
        "_record_family_anchor",
        lambda gc, member_ids, anchor_id: recorded.append(anchor_id),
    )

    outcome = harmonize.apply_families(
        _Graph(nodes),
        min_size=2,
        drain_fn=lambda **_kw: None,
        force="re-run the boundary-density family",
        runner_factory=lambda _spec: _Halted(),
    )

    assert recorded == [_ANCHOR], "a force reason must not move the anchor"
    assert outcome["applied"][0]["anchor"] == _ANCHOR


def test_the_report_names_the_anchor_the_apply_would_use(
    monkeypatch: Any,
) -> None:
    """The report shows the pinned anchor the planning would apply.

    A family held mid-run carries a recorded anchor whose member is
    mid-pipeline (null score) while a sibling is accepted and outscores it.
    The report is read through the same pin the planning reads, so the anchor
    it prints is the one an apply would start on rather than the live ranking.
    """
    from imas_codex.standard_names import graph_ops

    harmonize = _harmonize()
    nodes = [
        _node(_ANCHOR, "drafted", None, anchor=_ANCHOR),
        _node(_OUTSCORER, "accepted", 0.9, anchor=_ANCHOR),
    ]
    _family_read(monkeypatch, nodes)
    monkeypatch.setattr(
        _owning_graph_claim_scope,
        "get_existing_standard_names",
        lambda gc=None: {n["id"] for n in nodes},
    )

    plan = harmonize.plan_family_applies(_Graph(nodes), min_size=1)
    report = harmonize.report_family_rows()

    assert plan[0]["anchor"] == _ANCHOR, "the dry run keeps the pinned anchor"
    assert report[0]["anchor"] == _ANCHOR, "the report names the pinned anchor"
    assert report[0]["anchor"] == plan[0]["anchor"]


def test_report_without_a_client_counts_an_out_of_scope_link_live(
    monkeypatch: Any,
) -> None:
    """With no client the report still resolves a link against the graph."""
    from imas_codex.standard_names import graph_ops

    harmonize = _harmonize()
    target = "gamma_quantity_outside_scope"
    linked = _node(_ANCHOR, "accepted", 0.9)
    linked["links"] = [target]
    nodes = [linked, _node(_OUTSCORER, "accepted", 0.8)]
    _family_read(monkeypatch, nodes)
    monkeypatch.setattr(
        _owning_graph_claim_scope,
        "get_existing_standard_names",
        lambda gc=None: {target} | {n["id"] for n in nodes},
    )
    rows = harmonize.report_family_rows()

    assert rows[0]["lint"].get("link_dead_link", 0) == 0


def test_the_graph_wide_name_read_has_one_definition() -> None:
    """No module beside graph_ops restates the graph-wide all-name read."""
    from pathlib import Path

    package = Path(_harmonize().__file__).parent
    offenders = sorted(
        path.name
        for path in package.glob("*.py")
        if path.name != "graph_ops.py" and _OWNED_NAME_QUERY in path.read_text()
    )

    assert offenders == [], f"all-name read restated in {offenders}"


def test_the_report_reads_the_all_name_set_through_the_owner(
    monkeypatch: Any,
) -> None:
    """A client in hand reaches the owner, rather than being forked from."""
    from imas_codex.standard_names import graph_ops

    harmonize = _harmonize()
    seen: list[Any] = []
    client = object()

    def _owner(gc: Any = None) -> set[str]:
        seen.append(gc)
        return {"only_name"}

    monkeypatch.setattr(
        _owning_graph_claim_scope, "get_existing_standard_names", _owner
    )

    assert harmonize.known_name_ids(client) == {"only_name"}
    assert seen == [client], "the caller's client must reach the owner"

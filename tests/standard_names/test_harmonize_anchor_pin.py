"""A harmonize run keeps its anchor until its family is stamped harmonized.

A family's apply resets its anchor first, so a runner interrupted between the
anchor's reset and the family's stamp leaves the anchor mid-pipeline with a
null documentation score. A plan re-derived in that window would rank the
members afresh and pick a different member, discarding the one the run already
started on. The recorded anchor is what keeps the run pointed at its own choice
until the family is stamped, at which point it is released and the next apply
selects normally.

Negative controls, each a mutation the test applies itself so a run under the
mutation is a failing run:

* ``SHC_NO_ANCHOR_PIN`` - the recorded anchor is ignored, so the reuse test
  fails (the plan falls back to the highest-scoring member).
* ``SHC_NO_ANCHOR_RELEASE`` - the anchor is never released on the stamp, so
  the release test fails (a stale anchor survives into the next plan).
* ``SHC_SCOPE_ONLY_DEAD_LINKS`` - the report resolves link targets against the
  scope alone, so the no-client report test fails.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS

_ANCHOR = "base_at_axis"
_OUTSCORER = "base_at_boundary"


def _harmonize() -> Any:
    return importlib.import_module("imas_codex.standard_names.harmonize")


def _node(
    node_id: str,
    docs_stage: str,
    score: float | None,
    *,
    description: str | None = None,
    anchor: str | None = None,
    signature: str | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "physical_base": "base",
        "physics_domain": "equilibrium",
        "description": description or f"base quantity for {node_id}",
        "documentation": f"The base quantity documented for {node_id}.",
        "docs_stage": docs_stage,
        "reviewer_score_docs": score,
        "name_stage": "drafted",
        "harmonized_group_signature": signature,
        "harmonized_anchor": anchor,
        "links": [],
    }


def _family(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "physical_base": "base",
        "invariant": ["" for _ in PATTERN_INVARIANT_SEGMENTS],
        "n": len(nodes),
        "members": nodes,
    }


def _family_read(monkeypatch: Any, nodes: list[dict[str, Any]]) -> None:
    def _read(gc: Any, **_kw: Any) -> list[dict[str, Any]]:
        return [_family(nodes)]

    monkeypatch.setattr(_harmonize(), "fetch_pattern_families", _read)


def _pin_removed(monkeypatch: Any) -> None:
    if os.environ.get("SHC_NO_ANCHOR_PIN"):
        monkeypatch.setattr(_harmonize(), "pinned_family_anchor", lambda *a, **k: None)


def _release_removed(monkeypatch: Any) -> None:
    if os.environ.get("SHC_NO_ANCHOR_RELEASE"):
        monkeypatch.setattr(_harmonize(), "_clear_family_anchor", lambda *a, **k: None)


class _Graph:
    """A client stub that mutates the fixture nodes for the pin queries."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.nodes = nodes
        self.queries: list[str] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        ids = set(params.get("ids") or [])
        rows = [n for n in self.nodes if n["id"] in ids]
        if "REMOVE sn.harmonized_anchor" in cypher:
            for row in rows:
                row["harmonized_anchor"] = None
        elif "SET sn.harmonized_anchor" in cypher:
            for row in rows:
                row["harmonized_anchor"] = params.get("anchor")
        return []


def _stamp_stub(monkeypatch: Any) -> None:
    from imas_codex.standard_names import graph_ops

    monkeypatch.setattr(
        graph_ops, "stamp_harmonized_families", lambda families: len(families)
    )


def test_recorded_anchor_reused_when_another_member_outscores_it(
    monkeypatch: Any,
) -> None:
    """A family mid-run keeps its recorded anchor over a higher-scoring member.

    The recorded anchor was reset by the run, so it carries no score; a second
    member is accepted and scores well. Normal selection would take the second
    member, but the family is not yet stamped harmonized, so the recorded
    anchor wins.
    """
    _pin_removed(monkeypatch)
    nodes = [
        _node(_ANCHOR, "drafted", None, anchor=_ANCHOR),
        _node(_OUTSCORER, "accepted", 0.9, anchor=_ANCHOR),
    ]
    _family_read(monkeypatch, nodes)

    plans = _harmonize().plan_family_applies(object(), min_size=1)

    assert plans[0]["decision"] == "apply"
    assert plans[0]["anchor"] == _ANCHOR, "the recorded anchor must be reused"


def test_stamped_family_releases_the_anchor_and_selects_normally(
    monkeypatch: Any,
) -> None:
    """A converged family releases its anchor, so the next plan reranks.

    The stamp is the moment the anchor's work is done. After it, a plan over
    the family reranks from the live scores rather than reusing the anchor the
    finished run had recorded.
    """
    _release_removed(monkeypatch)
    _stamp_stub(monkeypatch)
    harmonize = _harmonize()
    nodes = [
        _node(_ANCHOR, "drafted", None, anchor=_ANCHOR),
        _node(_OUTSCORER, "accepted", 0.9, anchor=_ANCHOR),
    ]
    _family_read(monkeypatch, nodes)
    graph = _Graph(nodes)

    assert harmonize._stamp_applied_family(graph, _ANCHOR) is True

    plans = harmonize.plan_family_applies(object(), min_size=1)
    assert plans[0]["anchor"] == _OUTSCORER, "rerank once the anchor is released"


def test_report_without_a_client_counts_an_out_of_scope_link_live(
    monkeypatch: Any,
) -> None:
    """With no client the report still resolves a link against the graph.

    The report is scoped to one family, but its members link out to names the
    graph holds outside it. Called with no client, it must read the whole
    catalog rather than fall back to the scoped members, or a valid
    out-of-scope reference counts as a dead link.
    """
    from imas_codex.standard_names import graph_ops

    harmonize = _harmonize()
    target = "gamma_quantity_outside_scope"
    linked = _node(_ANCHOR, "accepted", 0.9)
    linked["links"] = [target]
    nodes = [linked, _node(_OUTSCORER, "accepted", 0.8)]
    _family_read(monkeypatch, nodes)
    monkeypatch.setattr(
        graph_ops,
        "get_existing_standard_names",
        lambda gc=None: {n["id"] for n in nodes} | {target},
    )
    if os.environ.get("SHC_SCOPE_ONLY_DEAD_LINKS"):
        monkeypatch.setattr(
            harmonize,
            "known_name_ids",
            lambda gc=None: {n["id"] for n in nodes},
        )

    rows = harmonize.report_family_rows()

    assert rows[0]["lint"].get("link_dead_link", 0) == 0


_OWNED_NAME_QUERY = 'MATCH (sn:StandardName) RETURN sn.id AS id"'


def test_the_graph_wide_name_read_has_one_definition() -> None:
    """No module beside graph_ops restates the graph-wide all-name read.

    The read is owned by ``graph_ops.get_existing_standard_names``; a second
    statement is a second source of truth for which names the graph holds, and
    the dead-link test then answers from whichever copy its caller reached.
    """
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
    """A client in hand is passed to the caller's owner, not forked from it."""
    from imas_codex.standard_names import graph_ops

    harmonize = _harmonize()
    seen: list[Any] = []
    client = object()

    def _owner(gc: Any = None) -> set[str]:
        seen.append(gc)
        return {"only_name"}

    monkeypatch.setattr(graph_ops, "get_existing_standard_names", _owner)

    assert harmonize.known_name_ids(client) == {"only_name"}
    assert seen == [client], "the caller's client must reach the owner"

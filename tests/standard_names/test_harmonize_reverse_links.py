"""A harmonize apply leaves no sibling link without its reverse.

A member's ``links`` are derived from its own documentation, and each member's
documentation is regenerated from its own prompt, so two members told to link
"at least one adjacent sibling" need not choose each other: a converged family
can carry a one-way sibling link. The apply's last step before it stamps the
family therefore makes the sibling relation symmetric, and drops a link whose
target is not a member of the family.

The tests drive the real ``apply_families`` against a fixture graph whose
documentation-field writer is stubbed, so what they assert is the mechanism the
apply runs, not a mock of it:

* after the apply, the audit function itself reports zero ``missing_reverse``
  findings over the family's links;
* a planted link to a name outside the family is removed from the member's
  ``links`` and from its documentation prose, while a non-name link (an
  external URL) is carried through untouched;
* a member outside the family is not written at all;
* removing the step makes the first test fail — the same fixture apply then
  ends with ``missing_reverse`` findings, which is the negative control logged
  beside the run.
"""

from __future__ import annotations

import os
from typing import Any

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS

_ANCHOR = "base_at_axis"
_BOUNDARY = "base_at_boundary"
_PROBE = "base_of_probe"
#: A live name in the catalog that is NOT a member of the fixture family.
_OUTSIDE = "other_at_axis"
_HANDBOOK = "https://example.org/handbook"

_ANCHOR_DOC = (
    "The base quantity at the axis. Related: "
    "[base at boundary](name:base_at_boundary). "
    f"Handbook: {_HANDBOOK}"
)
_BOUNDARY_DOC = "The base quantity at the boundary."
_PROBE_DOC = (
    "The base quantity of the probe. See [base at axis](name:base_at_axis) "
    "and [other at axis](name:other_at_axis)."
)
_OUTSIDE_DOC = "A base quantity documented elsewhere."


def _node(
    node_id: str,
    *,
    physical_base: str = "base",
    documentation: str = "",
    links: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "physical_base": physical_base,
        "description": f"base quantity for {node_id}",
        "documentation": documentation,
        "links": list(links or []),
        "docs_stage": "accepted",
        "reviewer_score_docs": 0.9,
        "name_stage": "accepted",
        "harmonized_group_signature": None,
    }


def _family_nodes() -> list[dict[str, Any]]:
    """The fixture family: one one-way sibling link and one link out of family."""
    return [
        _node(
            _ANCHOR, documentation=_ANCHOR_DOC, links=[f"name:{_BOUNDARY}", _HANDBOOK]
        ),
        _node(_BOUNDARY, documentation=_BOUNDARY_DOC),
        _node(
            _PROBE,
            documentation=_PROBE_DOC,
            links=[f"name:{_ANCHOR}", f"name:{_OUTSIDE}"],
        ),
        # Carried in the graph so it is a live catalog name, but its grammar
        # puts it in another family — the target of the planted link.
    ] + [_node(_OUTSIDE, physical_base="other", documentation=_OUTSIDE_DOC)]


def _invariant(node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(node.get(seg) or "") for seg in PATTERN_INVARIANT_SEGMENTS)


class _LinkGraph:
    """GraphClient stand-in answering the pattern-family read and the link reads.

    Mirrors the fixture graph the family reads use, and answers the two link
    reads the reverse-link step issues — the member doc/link state and the set
    of live catalog ids — by dispatching on the query's own projection.
    """

    def __init__(self, nodes: list[dict[str, Any]]):
        self.nodes = [dict(n) for n in nodes]
        self.queries: list[str] = []
        self.written: list[list[dict[str, Any]]] = []

    def __enter__(self) -> _LinkGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def node(self, node_id: str) -> dict[str, Any] | None:
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if "sn.links AS links" in cypher:
            wanted = set(params.get("ids") or [])
            return [dict(n) for n in self.nodes if n["id"] in wanted]
        if "sn.name_stage AS name_stage" in cypher:
            names = set(params.get("names") or [])
            return [
                {"id": n["id"], "name_stage": n.get("name_stage")}
                for n in self.nodes
                if n["id"] in names
            ]
        if "RETURN sn.id AS id" in cypher:
            wanted = set(params.get("ids") or [])
            return [{"id": n["id"]} for n in self.nodes if n["id"] in wanted]
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

    # ── The stubbed documentation-field writer ─────────────────────────────

    def write_enrichment_results(self, results: list[dict[str, Any]]) -> int:
        """Stand in for the graph_ops writer that persists docs + links."""
        self.written.append([dict(r) for r in results])
        for item in results:
            node = self.node(item["id"])
            if node is None:
                continue
            node["documentation"] = item["documentation"]
            node["links"] = list(item["links"])
        return len(results)


class _ApplyHarness:
    """Injected runner operations: drain the scope, accept every refreshed name."""

    def __init__(self, accepted_after: set[str]):
        self.accepted_after = accepted_after
        self.marked_batches: list[list[str]] = []

    def mark_fn(self, batch: list[str], dry_run: bool = False) -> dict[str, Any]:
        self.marked_batches.append(list(batch))
        return {"run_id": "fixture", "eligible": len(batch), "reset": len(batch)}

    def clear_quarantine_fn(self, gc: Any, batch: list[str]) -> int:
        return len(batch)

    def drain_fn(self, run_id: str, cost_limit: float) -> float | None:
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


def _reverse_step_or_removed(monkeypatch: Any) -> None:
    """Disable the reverse-link step when the declared mutation is set.

    ``SHC_NO_REVERSE_STEP`` removes the step, so the fixture apply ends with
    ``missing_reverse`` findings and the assertions below fail.
    """
    if os.environ.get("SHC_NO_REVERSE_STEP"):
        from imas_codex.standard_names import harmonize

        monkeypatch.setattr(
            harmonize,
            "reconcile_family_reverse_links",
            lambda gc, member_ids, **kw: {
                "members": len(member_ids),
                "added": 0,
                "removed": 0,
                "written": 0,
            },
        )


def _apply(graph: _LinkGraph, monkeypatch: Any) -> dict[str, Any]:
    """Run the real apply over the fixture with the graph's writer stubbed."""
    from imas_codex.standard_names import graph_ops, harmonize

    monkeypatch.setattr(
        graph_ops, "write_enrichment_results", graph.write_enrichment_results
    )
    monkeypatch.setattr(
        graph_ops, "stamp_harmonized_families", lambda families: len(families)
    )
    _reverse_step_or_removed(monkeypatch)

    harness = _ApplyHarness(accepted_after={_ANCHOR, _BOUNDARY, _PROBE})
    return harmonize.apply_families(
        graph,
        seeds=[_ANCHOR],
        min_size=1,
        drain_fn=harness.drain_fn,
        runner_kwargs={
            "mark_fn": harness.mark_fn,
            "clear_quarantine_fn": harness.clear_quarantine_fn,
            "fetch_refreshed_fn": harness.fetch_refreshed_fn,
            "revalidate_fn": harness.revalidate_fn,
            "record_change_fn": harness.record_change_fn,
            "cost_fn": harness.cost_fn,
        },
    )


def _missing_reverse(graph: _LinkGraph) -> list[Any]:
    """The audit's own findings over the family's post-apply links."""
    from imas_codex.standard_names.review.audits import run_link_integrity

    member_ids = [_ANCHOR, _BOUNDARY, _PROBE]
    known_ids = {n["id"] for n in graph.nodes}
    rows = [
        {"id": nid, "links": list((graph.node(nid) or {}).get("links") or [])}
        for nid in member_ids
    ]
    return [
        f
        for f in run_link_integrity(rows, known_ids=known_ids)
        if f.finding_type == "missing_reverse"
    ]


def test_apply_leaves_no_sibling_link_without_its_reverse(monkeypatch: Any) -> None:
    """The audit reports zero missing_reverse findings after the fixture apply."""
    graph = _LinkGraph(_family_nodes())

    outcome = _apply(graph, monkeypatch)

    entry = outcome["applied"][0]
    assert entry["applied"] is True
    assert entry["result"].halted is False
    assert _missing_reverse(graph) == []
    # The reverse the step added is on the member's documentation too, since
    # the link index is derived from it.
    assert f"name:{_ANCHOR}" in graph.node(_BOUNDARY)["documentation"]


def test_apply_removes_a_link_out_of_the_family(monkeypatch: Any) -> None:
    """A link to a live name outside the family is dropped, an URL is kept."""
    graph = _LinkGraph(_family_nodes())

    _apply(graph, monkeypatch)

    probe = graph.node(_PROBE)
    assert probe["links"] == [f"name:{_ANCHOR}"]
    # The dropped reference is gone from the prose the index is derived from...
    assert f"name:{_OUTSIDE}" not in probe["documentation"]
    # ...while the non-name link is carried through untouched.
    assert graph.node(_ANCHOR)["links"] == [
        f"name:{_BOUNDARY}",
        _HANDBOOK,
        f"name:{_PROBE}",
    ]
    # A name outside the family is not written at all.
    assert graph.node(_OUTSIDE)["documentation"] == _OUTSIDE_DOC
    assert graph.node(_OUTSIDE)["links"] == []


def test_removing_the_step_leaves_missing_reverse_findings(monkeypatch: Any) -> None:
    """The declared negative control: without the step the findings remain."""
    monkeypatch.setenv("SHC_NO_REVERSE_STEP", "1")
    graph = _LinkGraph(_family_nodes())

    _apply(graph, monkeypatch)

    findings = _missing_reverse(graph)
    assert findings, "the fixture must show missing_reverse without the step"
    assert {(f.name_id, f.target) for f in findings} == {
        (_ANCHOR, _BOUNDARY),
        (_PROBE, _ANCHOR),
    }

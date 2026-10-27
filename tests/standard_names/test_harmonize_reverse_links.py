"""A harmonize apply leaves no sibling link without its reverse.

A member's ``links`` are derived from its own documentation, and each member's
documentation is regenerated from its own prompt, so two members told to link
"at least one adjacent sibling" need not choose each other: a converged family
can carry a one-way sibling link. The apply's last step before it stamps the
family makes the sibling relation symmetric.

Removal is a separate rule with a much narrower reach: a ``name:`` link goes
only when no name in the catalog carries its target, so no reverse could ever
exist. A reference to a live name outside the family is a legitimate
cross-family reference and survives — the step reconciles the sibling relation,
it does not police the rest of a member's references.

The tests drive the real ``apply_families`` against a fixture graph whose
documentation-field writer is stubbed, so what they assert is the mechanism the
apply runs, not a mock of it:

* after the fixture apply, the audit function itself reports zero
  ``missing_reverse`` findings over the family's links;
* a link to a live name outside the family survives, in the member's ``links``
  and in its documentation prose, and is not made reciprocal;
* a link to a name the catalog does not carry is removed, from the ``links``
  index and from the prose it is derived from;
* a member the catalog has edited is not written, and the step reports it
  rather than reporting a write that did not land.

Each assertion is coupled to its mechanism by a red run recorded beside the
green one, produced by removing that mechanism and running the same command.
"""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS
from imas_codex.standard_names.graph_store import (
    graph_docs_persistence as _owning_graph_docs_persistence,
    graph_source_bindings as _owning_graph_source_bindings,
)

_ANCHOR = "base_at_axis"
_BOUNDARY = "base_at_boundary"
_PROBE = "base_of_probe"
#: A live name in the catalog, in another pattern family — not a sibling.
_OUTSIDE = "other_at_axis"
#: A name no StandardName in the catalog carries: the target of a dead link.
_GHOST = "base_of_ghost"
_HANDBOOK = "https://example.org/handbook"

_ANCHOR_DOC = (
    "The base quantity at the axis. Related: "
    "[base at boundary](name:base_at_boundary). "
    f"Handbook: {_HANDBOOK}"
)
_BOUNDARY_DOC = "The base quantity at the boundary."
_PROBE_DOC = (
    "The base quantity of the probe. See [base at axis](name:base_at_axis), "
    "[other at axis](name:other_at_axis) and [the ghost](name:base_of_ghost)."
)
_OUTSIDE_DOC = "A base quantity documented elsewhere."


def _node(
    node_id: str,
    *,
    physical_base: str = "base",
    documentation: str = "",
    links: list[str] | None = None,
    name_stage: str = "accepted",
) -> dict[str, Any]:
    return {
        "id": node_id,
        "physical_base": physical_base,
        "description": f"base quantity for {node_id}",
        "documentation": documentation,
        "links": list(links or []),
        "docs_stage": "accepted",
        "reviewer_score_docs": 0.9,
        "name_stage": name_stage,
        "harmonized_group_signature": None,
        "has_dd_source": True,
    }


def _family_nodes(*, probe_stage: str = "accepted") -> list[dict[str, Any]]:
    """The fixture family plus two names outside it.

    ``probe_stage`` makes the probe catalog-edited, so its reconciliation is
    refused by the writer's protection filter.
    """
    return [
        _node(_OUTSIDE, physical_base="other", documentation=_OUTSIDE_DOC),
    ] + [
        _node(
            _ANCHOR, documentation=_ANCHOR_DOC, links=[f"name:{_BOUNDARY}", _HANDBOOK]
        ),
        _node(_BOUNDARY, documentation=_BOUNDARY_DOC),
        _node(
            _PROBE,
            documentation=_PROBE_DOC,
            links=[
                f"name:{_ANCHOR}",
                f"name:{_OUTSIDE}",
                f"name:{_GHOST}",
            ],
            name_stage=probe_stage,
        ),
    ]


def _invariant(node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(node.get(seg) or "") for seg in PATTERN_INVARIANT_SEGMENTS)


class _LinkGraph:
    """GraphClient stand-in answering the pattern-family reads and the link reads.

    Mirrors the fixture graph the family reads use, and answers the link reads
    the reverse-link step issues — the member doc/link state, the set of live
    catalog ids, and the approved-state read protection makes — by dispatching
    on each query's own projection.
    """

    def __init__(self, nodes: list[dict[str, Any]]):
        self.nodes = [dict(n) for n in nodes]
        self.queries: list[str] = []
        self.written: list[list[dict[str, Any]]] = []
        self.protected_by_writer: list[str] = []

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
        if "name_stage = 'approved'" in cypher:
            names = set(params.get("names") or [])
            return [
                {"id": n["id"]}
                for n in self.nodes
                if n["id"] in names and n.get("name_stage") == "approved"
            ]
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
        """Stand in for the graph_ops writer that persists docs + links.

        It reproduces the real writer's protection behaviour down to what it
        does with a stripped row: ``filter_protected`` hands the row back with
        its editorial fields removed, and the writer's Cypher writes each
        now-absent field as an empty one (``documentation = ""``,
        ``links = null``). A stub that treated a stripped row as *no change*
        would let "the protected member keeps its documentation" hold for the
        stub and fail never, which is the opposite of what the production
        writer does.
        """
        from imas_codex.standard_names.protection import filter_protected

        filtered, skipped = filter_protected(results)
        self.protected_by_writer.extend(skipped)
        self.written.append([dict(r) for r in filtered])
        for item in filtered:
            node = self.node(item["id"])
            if node is None:
                continue
            node["description"] = item.get("description") or ""
            node["documentation"] = item.get("documentation") or ""
            node["links"] = list(item.get("links") or []) or None
            node["validity_domain"] = item.get("validity_domain")
            node["constraints"] = item.get("constraints") or None
        return len(filtered)


class _ApplyHarness:
    """Injected runner operations: drain the scope, accept every refreshed name."""

    def __init__(self, accepted_after: set[str]):
        self.accepted_after = accepted_after

    def mark_fn(self, batch: list[str], dry_run: bool = False) -> dict[str, Any]:
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


def _apply(graph: _LinkGraph, monkeypatch: Any) -> dict[str, Any]:
    """Run the real apply over the fixture with the graph's writer stubbed."""
    from imas_codex.graph import client as graph_client
    from imas_codex.standard_names import graph_ops, harmonize

    monkeypatch.setattr(
        _owning_graph_source_bindings,
        "write_enrichment_results",
        graph.write_enrichment_results,
    )
    monkeypatch.setattr(
        _owning_graph_docs_persistence,
        "stamp_harmonized_families",
        lambda families: len(families),
    )
    # ``filter_protected`` reaches the graph for the approved-name set through
    # its own client, so the stub answers that read too.
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)

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


def _findings(graph: _LinkGraph, finding_type: str) -> list[Any]:
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
        if f.finding_type == finding_type
    ]


def test_apply_leaves_no_sibling_link_without_its_reverse(monkeypatch: Any) -> None:
    graph = _LinkGraph(_family_nodes())

    outcome = _apply(graph, monkeypatch)

    entry = outcome["applied"][0]
    assert entry["applied"] is True
    assert entry["result"].halted is False
    assert _findings(graph, "missing_reverse") == []
    # The reverse the step added is on the member's documentation too, since
    # the link index is derived from it.
    assert f"name:{_ANCHOR}" in graph.node(_BOUNDARY)["documentation"]


def test_a_live_name_outside_the_family_survives(monkeypatch: Any) -> None:
    """A cross-family reference is kept, in the index and in the prose."""
    graph = _LinkGraph(_family_nodes())

    _apply(graph, monkeypatch)

    probe = graph.node(_PROBE)
    # The reference to the out-of-family name is present in the index, once.
    assert [link for link in probe["links"] if link.startswith("name:")].count(
        f"name:{_OUTSIDE}"
    ) == 1
    # The prose the index is derived from still carries the reference...
    assert f"name:{_OUTSIDE}" in probe["documentation"]
    # ...and the out-of-family name was not made reciprocal: it is not a
    # sibling, so no such relation is being asserted.
    assert graph.node(_OUTSIDE)["documentation"] == _OUTSIDE_DOC
    assert graph.node(_OUTSIDE)["links"] == []
    # The non-name link is carried through untouched.
    assert graph.node(_ANCHOR)["links"] == [
        f"name:{_BOUNDARY}",
        _HANDBOOK,
        f"name:{_PROBE}",
    ]


def test_a_link_to_a_name_the_catalog_lacks_is_removed(monkeypatch: Any) -> None:
    """A dead name: link goes, from the index and from the prose."""
    graph = _LinkGraph(_family_nodes())
    assert _GHOST not in {n["id"] for n in graph.nodes}

    _apply(graph, monkeypatch)

    probe = graph.node(_PROBE)
    assert f"name:{_GHOST}" not in probe["links"]
    assert f"name:{_GHOST}" not in probe["documentation"]
    # The audit agrees: nothing dead is left to report.
    assert _findings(graph, "dead_link") == []


def test_a_catalog_edited_member_is_reported_not_silently_skipped(
    monkeypatch: Any,
) -> None:
    """Protection is not overridden, and the step names what it could not write.

    Driven against the step rather than through the apply: an apply over a
    committed member is refused earlier by ``refuse_recommended_members``, so
    the apply alone cannot reach this state. The step is reached directly here
    because it is the layer that must not overstate what it wrote.
    """
    from imas_codex.graph import client as graph_client
    from imas_codex.standard_names import graph_ops, harmonize

    graph = _LinkGraph(_family_nodes(probe_stage="approved"))
    monkeypatch.setattr(
        _owning_graph_source_bindings,
        "write_enrichment_results",
        graph.write_enrichment_results,
    )
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)

    step = harmonize.reconcile_family_reverse_links(graph, [_ANCHOR, _BOUNDARY, _PROBE])

    # The step reports the member protection withheld...
    assert step["protected"] == [_PROBE]
    # ...and the member keeps its own documentation and links, dead link and
    # all. This is the assertion the finding is about: handed a protected row
    # the writer strips it and writes each stripped field as empty — clearing
    # the member's documentation and links — so it stands before the counts, and
    # a regression fails here, naming the harm.
    probe = graph.node(_PROBE)
    assert f"name:{_GHOST}" in probe["links"]
    assert probe["documentation"] == _PROBE_DOC
    # The protected row never reached the writer at all, which is why the harm
    # above cannot happen: nothing for it to strip, and one fewer row written.
    assert graph.protected_by_writer == []
    assert step["written"] == 2
    # The counts cover the rows that were written and no others. The probe's
    # dead link is a removal the step computed for it, and it is not reported,
    # because the probe's row was withheld and nothing about it was written:
    # a count that included it would report a change that did not land.
    assert step["removed"] == 0
    assert step["added"] == 2
    assert [row["id"] for write_batch in graph.written for row in write_batch] == [
        _ANCHOR,
        _BOUNDARY,
    ]
    # The members that were not protected did get their reconciliation.
    assert f"name:{_PROBE}" in graph.node(_ANCHOR)["links"]
    assert f"name:{_ANCHOR}" in graph.node(_BOUNDARY)["links"]

"""The merge approval state machine and the ``contested`` holding state.

- Untouched batch names auto-promote accepted→approved on merge (with PR
  metadata); edited names re-trigger review (pass→approved, fail→contested).
- ``contested`` is frozen: pool-excluded (even as a parent's only child) and
  resolved only by sn edit / sn approve --override / sn revert.

The PR-diff reader and the review scorer are stubbed so the state machine is
tested without a live ISNC checkout or an LLM call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import merge as merge_mod
from imas_codex.standard_names.merge import (
    MergeChange,
    list_contested,
    override_approve_contested,
    revert_contested,
    run_merge,
)

PREFIX = "__contestedtest__"

PR = {
    "catalog_pr_number": 7,
    "catalog_pr_url": "https://github.com/x/y/pull/7",
    "catalog_merge_commit_sha": "abc123",
}


def _cleanup():
    with GraphClient() as gc:
        gc.query("MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n", p=PREFIX)


def _stage(gc, nid, **props):
    gc.query(
        "MERGE (n:StandardName {id: $id}) SET n += $props",
        id=nid,
        props=props,
    )


@pytest.fixture
def clean():
    _cleanup()
    yield
    _cleanup()


def _name_stage(nid):
    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (n:StandardName {id: $id}) RETURN n.name_stage AS s", id=nid
        )
    return rows[0]["s"] if rows else None


# ── auto-approve of untouched batch names ─────────────────────────────────


@pytest.mark.graph
def test_merge_auto_approves_untouched_batch(clean, monkeypatch):
    ready = f"{PREFIX}_ready"
    notready = f"{PREFIX}_notready"
    with GraphClient() as gc:
        _stage(gc, ready, name_stage="accepted", docs_stage="accepted")
        _stage(gc, notready, name_stage="accepted", docs_stage="drafted")

    # No reviewer edits — the whole batch was approved as-is.
    monkeypatch.setattr(merge_mod, "read_pr_changes", lambda *a, **k: [])

    with GraphClient() as gc:
        report = run_merge(
            isnc_dir="/unused",
            base_ref="main",
            batch=[ready, notready],
            gc=gc,
            **PR,
        )

    assert ready in report.auto_approved
    # docs not accepted → cannot approve yet; stays accepted.
    assert notready not in report.auto_approved
    assert _name_stage(ready) == "approved"
    assert _name_stage(notready) == "accepted"


@pytest.mark.graph
def test_merge_edited_name_excluded_from_auto_approve(clean, monkeypatch):
    edited = f"{PREFIX}_edited"
    with GraphClient() as gc:
        _stage(gc, edited, name_stage="accepted", docs_stage="accepted")

    change = MergeChange(sn_id=edited, axis="docs", new_value="new", old_value="old")
    monkeypatch.setattr(merge_mod, "read_pr_changes", lambda *a, **k: [change])
    monkeypatch.setattr(
        merge_mod,
        "apply_edit",
        lambda **k: SimpleNamespace(blocked=None, successor=None, run_id=None),
    )
    # Edited name passes re-review → normal accept+approve path (not auto).
    monkeypatch.setattr(merge_mod, "_score_proposal", lambda *a, **k: 0.95)
    monkeypatch.setattr(merge_mod, "_accept", lambda *a, **k: None)

    with GraphClient() as gc:
        report = run_merge(
            isnc_dir="/unused",
            base_ref="main",
            batch=[edited],
            gc=gc,
            **PR,
        )

    # The edited name went through the edit path, not the untouched auto path.
    assert edited not in report.auto_approved
    assert edited in report.accepted


# ── edited-fail → contested ───────────────────────────────────────────────


@pytest.mark.graph
def test_merge_edited_fail_contests(clean, monkeypatch):
    name = f"{PREFIX}_fail"
    with GraphClient() as gc:
        _stage(gc, name, name_stage="accepted", docs_stage="accepted")

    change = MergeChange(sn_id=name, axis="docs", new_value="bad", old_value="old")
    monkeypatch.setattr(merge_mod, "read_pr_changes", lambda *a, **k: [change])
    monkeypatch.setattr(
        merge_mod,
        "apply_edit",
        lambda **k: SimpleNamespace(blocked=None, successor=None, run_id=None),
    )
    monkeypatch.setattr(merge_mod, "_score_proposal", lambda *a, **k: 0.10)

    with GraphClient() as gc:
        report = run_merge(isnc_dir="/unused", base_ref="main", gc=gc, **PR)

    assert any(c["sn_id"] == name for c in report.contested)
    assert not report.quarantined
    assert _name_stage(name) == "contested"
    with GraphClient() as gc:
        row = gc.query(
            "MATCH (n:StandardName {id:$id}) RETURN n.contested_reason AS r", id=name
        )[0]
    assert "failed compliance re-review" in row["r"]


# ── resolution: override / revert / list ──────────────────────────────────


@pytest.mark.graph
def test_override_approve_and_revert(clean):
    over = f"{PREFIX}_over"
    rev = f"{PREFIX}_rev"
    with GraphClient() as gc:
        _stage(gc, over, name_stage="contested", contested_reason="x")
        _stage(gc, rev, name_stage="contested", contested_reason="y")

    listed = {r["id"] for r in list_contested()}
    assert {over, rev} <= listed

    assert override_approve_contested(over, reason="expert override") is True
    assert _name_stage(over) == "approved"

    assert revert_contested(rev, reason="drop the edit") is True
    assert _name_stage(rev) == "accepted"


@pytest.mark.graph
def test_resolution_noop_on_non_contested(clean):
    acc = f"{PREFIX}_acc"
    with GraphClient() as gc:
        _stage(gc, acc, name_stage="accepted")
    assert override_approve_contested(acc, reason="x") is False
    assert revert_contested(acc, reason="x") is False
    assert _name_stage(acc) == "accepted"


# ── contested is pool-excluded even as a parent's only child ───────────────


@pytest.mark.graph
def test_contested_child_not_live_for_enrich(clean):
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )
    from imas_codex.standard_names.graph_ops import claim_enrich_parents_batch

    parent = f"{PREFIX}_parent"
    child = f"{PREFIX}_child"
    with GraphClient() as gc:
        _stage(
            gc,
            parent,
            origin="derived",
            description=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
            name_stage="drafted",
        )
        _stage(gc, child, name_stage="contested")
        gc.query(
            "MATCH (c:StandardName {id:$c}),(p:StandardName {id:$p}) "
            "MERGE (c)-[:HAS_PARENT]->(p)",
            c=child,
            p=parent,
        )

    # A parent whose only child is contested has no LIVE child → not claimable.
    claimed = {i["id"] for i in claim_enrich_parents_batch(batch_size=200)}
    assert parent not in claimed

    # Promote the child to accepted → the parent becomes claimable.
    with GraphClient() as gc:
        gc.query(
            "MATCH (n:StandardName {id:$id}) SET n.name_stage='accepted'", id=child
        )
    claimed2 = {i["id"] for i in claim_enrich_parents_batch(batch_size=200)}
    assert parent in claimed2

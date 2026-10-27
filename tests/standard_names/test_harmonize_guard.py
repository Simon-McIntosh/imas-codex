"""`sn harmonize --apply` refuses names already recommended into a request.

A reset rewrites text a physicist has been asked to review, or has already
reviewed, so the apply refuses the whole family before the first member is
marked. Three classes are recommended, and so protected:

* a member of a frozen review roster whose candidate request is open;
* a name at ``name_stage=approved``;
* a name at ``name_stage=contested``.

The refusal names each protected identity with the evidence that protected it.
``--force REASON`` overrides and records REASON (with the evidence it
overrode) under the apply's own scope run id. Removing the guard call before
the reset lets the refusal test fail — the declared negative control, logged
beside the run.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS
from imas_codex.standard_names.graph_store import (
    graph_docs_persistence as _owning_graph_docs_persistence,
)

_ANCHOR = "base_at_axis"
_ROSTER_MEMBER = "base_at_boundary"
_APPROVED_MEMBER = "base_of_probe"
_MEMBERS = [_ROSTER_MEMBER, _APPROVED_MEMBER]


def _node(node_id: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "physical_base": "base",
        "description": f"base quantity for {node_id}",
        "documentation": f"The base quantity documented for {node_id}.",
        "docs_stage": "drafted",
        "reviewer_score_docs": None,
        "harmonized_group_signature": None,
        "has_dd_source": True,
    }


def _family_nodes() -> list[dict[str, Any]]:
    return [_node(_ANCHOR), _node(_ROSTER_MEMBER), _node(_APPROVED_MEMBER)]


def _invariant(node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(node.get(seg) or "") for seg in PATTERN_INVARIANT_SEGMENTS)


class _GuardGraph:
    """GraphClient stand-in answering the family read, the stage read and writes.

    Groups fixture nodes by invariant grammar signature (the query's own notion
    of a family), answers ``recommended_name_protections``'s name-stage read
    from ``stages``, and records the ``StandardNameChange`` the forced-override
    path creates so a caller can assert the recorded reason and run id.
    """

    def __init__(
        self, nodes: list[dict[str, Any]], stages: dict[str, str] | None = None
    ):
        self.nodes = [dict(n) for n in nodes]
        self.stages = dict(stages or {})
        self.queries: list[str] = []
        self.changes: list[dict[str, Any]] = []

    def __enter__(self) -> _GuardGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if "name_stage AS name_stage" in cypher:
            return [
                {"id": name, "name_stage": self.stages.get(name)}
                for name in params.get("names") or []
            ]
        if "CREATE (change:StandardNameChange" in cypher:
            self.changes.append(dict(params))
            return []
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
    """The fixture docs graph: per-name documentation plus a revision stack."""

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


class _Harness:
    """Injected runner operations recording the batches and drain order."""

    def __init__(self, accepted_after: set[str], pipeline: _DocsPipeline):
        self.accepted_after = accepted_after
        self.pipeline = pipeline
        self.marked_batches: list[list[str]] = []

    def mark_fn(self, batch: list[str], dry_run: bool = False) -> dict[str, Any]:
        from imas_codex.standard_names.harmonize import mark_members_for_regen

        self.marked_batches.append(list(batch))
        return mark_members_for_regen(batch, dry_run=dry_run)

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

    def runner_kwargs(self) -> dict[str, Any]:
        return {
            "mark_fn": self.mark_fn,
            "clear_quarantine_fn": self.clear_quarantine_fn,
            "fetch_refreshed_fn": self.fetch_refreshed_fn,
            "revalidate_fn": self.revalidate_fn,
            "record_change_fn": self.record_change_fn,
            "cost_fn": self.cost_fn,
        }


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


def _roster(tmp_path: Any, names: list[str]) -> str:
    """Write one frozen review roster recording an opened candidate request."""
    import yaml

    (tmp_path / "v0.4.0rc7+west.sn_names.yaml").write_text(
        yaml.safe_dump({"pr_number": 4242, "names": names}), encoding="utf-8"
    )
    return str(tmp_path)


def _apply(
    graph: _GuardGraph,
    harness: _Harness,
    *,
    force: str | None = None,
    reviews_dir: Any = None,
):
    from imas_codex.standard_names.harmonize import apply_families

    return apply_families(
        graph,
        seeds=[_ANCHOR],
        min_size=1,
        drain_fn=harness.drain_fn,
        force=force,
        reviews_dir=reviews_dir,
        runner_kwargs=harness.runner_kwargs(),
    )


def test_roster_and_approved_members_refused_without_force(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A family holding an open-roster member and an approved name is refused.

    The refusal names both, each with the evidence that protected it, and no
    member is marked, snapshotted or reset.
    """
    from imas_codex.standard_names.protection import RecommendedNameError

    nodes = _family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    harness = _Harness(accepted_after={_ANCHOR, *_MEMBERS}, pipeline=pipeline)
    graph = _GuardGraph(nodes, stages={_APPROVED_MEMBER: "approved"})

    with pytest.raises(RecommendedNameError) as excinfo:
        _apply(graph, harness, reviews_dir=_roster(tmp_path, [_ROSTER_MEMBER]))

    message = str(excinfo.value)
    assert _ROSTER_MEMBER in message
    assert "candidate_request=" in message
    assert _APPROVED_MEMBER in message
    assert "name_stage=approved" in message
    # Nothing was reset and no revision was snapshotted.
    assert harness.marked_batches == []
    assert pipeline.revisions == {}
    assert graph.changes == []


def test_contested_member_refused_without_force(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A name at ``contested`` is refused the same way, with its evidence."""
    from imas_codex.standard_names.protection import RecommendedNameError

    nodes = _family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    harness = _Harness(accepted_after={_ANCHOR, *_MEMBERS}, pipeline=pipeline)
    graph = _GuardGraph(nodes, stages={_APPROVED_MEMBER: "contested"})

    with pytest.raises(RecommendedNameError) as excinfo:
        _apply(graph, harness, reviews_dir=_roster(tmp_path, []))

    message = str(excinfo.value)
    assert _APPROVED_MEMBER in message
    assert "name_stage=contested" in message
    assert pipeline.revisions == {}


def test_forced_apply_records_reason_on_the_run(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """``--force REASON`` applies and records REASON against the run id."""
    nodes = _family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    harness = _Harness(accepted_after={_ANCHOR, *_MEMBERS}, pipeline=pipeline)
    graph = _GuardGraph(nodes, stages={_APPROVED_MEMBER: "approved"})

    outcome = _apply(
        graph,
        harness,
        force="typo in the anchor",
        reviews_dir=_roster(tmp_path, [_ROSTER_MEMBER]),
    )

    # The guard reported the classes it overrode, and the apply ran.
    assert set(outcome["recommended_guards"]) == {_ROSTER_MEMBER, _APPROVED_MEMBER}
    assert outcome["force_reason"] == "typo in the anchor"
    assert outcome["force_run_id"]
    assert harness.marked_batches  # the reset actually happened

    # The reason and the evidence it overrode are on the run record.
    recorded = {change["reason"]: change for change in graph.changes}
    assert len(graph.changes) == 2
    run_ids = {change["run_id"] for change in graph.changes}
    assert run_ids == {outcome["force_run_id"]}
    assert all("typo in the anchor" in reason for reason in recorded)
    joined = " ".join(recorded)
    assert "candidate_request=" in joined
    assert "name_stage=approved" in joined


def test_harmonize_help_describes_report_and_apply() -> None:
    """`sn harmonize --help` describes both the report and the apply modes."""
    from click.testing import CliRunner

    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(sn, ["harmonize", "--help"])

    assert result.exit_code == 0, result.output
    text = result.output
    assert "--report" in text
    assert "--apply" in text
    assert "--force" in text
    # The stale "only mode" phrasing is gone.
    assert "the only mode" not in text


def test_cli_refuses_recommended_member_without_force(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """`sn harmonize --apply` refuses through the CLI, naming the protected name."""
    from click.testing import CliRunner

    from imas_codex.cli.sn import sn
    from imas_codex.graph import client as graph_client
    from imas_codex.standard_names import catalog_release

    nodes = _family_nodes()
    pipeline = _DocsPipeline(nodes)
    _install_docs_pipeline(monkeypatch, pipeline)
    graph = _GuardGraph(nodes, stages={_APPROVED_MEMBER: "approved"})
    monkeypatch.setattr(graph_client, "GraphClient", lambda *a, **k: graph)
    monkeypatch.setattr(catalog_release, "default_reviews_dir", lambda: tmp_path)
    _roster(tmp_path, [_ROSTER_MEMBER])

    result = CliRunner().invoke(sn, ["harmonize", "--family", _ANCHOR, "--apply"])

    assert result.exit_code != 0
    assert _ROSTER_MEMBER in result.output
    assert "name_stage=approved" in result.output
    assert pipeline.revisions == {}

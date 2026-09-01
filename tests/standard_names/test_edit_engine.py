"""Tests for the ``sn edit`` engine (``imas_codex.standard_names.edit``).

Exercised against an in-memory ``FakeGraph`` that emulates both the plain
``query()`` surface used by ``edit.py``/``cascade.py`` and the transactional
``session()/begin_transaction()`` surface used by ``persist_refined_name``
and ``persist_refined_docs`` in ``graph_ops.py``. No live Neo4j — the
autouse ``_block_live_graph`` fixture (conftest.py) would refuse one anyway.

``imas_codex.standard_names.graph_ops.GraphClient`` is patched to return
the SAME ``FakeGraph`` instance passed to ``apply_edit(gc=...)`` so that
internal ``with GraphClient() as gc:`` blocks inside the persist_* helpers
observe the same state as edit.py's own direct queries.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from imas_codex.standard_names.edit import EditPlan, apply_edit

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"
_WRITE_REVIEWS_PATH = "imas_codex.standard_names.graph_ops.write_reviews"

_DEFAULT_NODE_FIELDS: dict[str, Any] = {
    "name_stage": None,
    "docs_stage": None,
    "description": "a description",
    "documentation": "some documentation",
    "docs_model": None,
    "docs_generated_at": None,
    "kind": "scalar",
    "unit": None,
    "physics_domain": None,
    "tags": [],
    "chain_length": 0,
    "docs_chain_length": 0,
    "origin": None,
    "reviewer_score_name": None,
    "reviewer_score_docs": None,
    "edit_mode": None,
    "name_hint": None,
    "docs_hint": None,
    "edit_reason": None,
    "edit_origin": None,
    "edit_scope": None,
    "edit_status": None,
    "edit_requested_at": None,
    "edit_override_edits": None,
    "edit_include_accepted": None,
    "claim_token": None,
    "claimed_at": None,
    "validation_issues": None,
    "validation_status": None,
    "validated_at": None,
    "run_id": None,
}


@dataclass
class FakeGraph:
    """In-memory stand-in for ``GraphClient`` covering the edit engine's
    query surface (edit.py's own ``// EDIT_*`` marked queries, cascade.py's
    walk/rename queries, and graph_ops.py's persist_refined_name /
    persist_reviewed_name / persist_reviewed_docs / persist_refined_docs /
    reset_standard_name_docs queries).
    """

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # child_id -> [{"parent_id", "operator", "operator_kind", "role",
    #               "separator", "axis", "shape"}]
    edges_by_child: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # successor_id -> predecessor_id  ((successor)-[:REFINED_FROM]->(predecessor))
    refined_from: dict[str, str] = field(default_factory=dict)
    # source_id -> sn_id  ((source)-[:PRODUCED_NAME]->(sn))
    produced_name: dict[str, str] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    docs_revisions: list[str] = field(default_factory=list)
    source_migration_events: set[str] = field(default_factory=set)

    # -- construction helpers ------------------------------------------------

    def add_node(self, sn_id: str, **fields: Any) -> None:
        row = dict(_DEFAULT_NODE_FIELDS)
        row.update(fields)
        self.nodes[sn_id] = row

    def add_edge(
        self,
        child: str,
        parent: str,
        operator: str | None,
        operator_kind: str,
        *,
        role: str | None = None,
        separator: str | None = None,
        axis: str | None = None,
        shape: str | None = None,
    ) -> None:
        self.edges_by_child.setdefault(child, []).append(
            {
                "parent_id": parent,
                "operator": operator,
                "operator_kind": operator_kind,
                "role": role,
                "separator": separator,
                "axis": axis,
                "shape": shape,
            }
        )

    def add_source(
        self,
        source_id: str,
        *,
        sn_id: str,
        status: str = "extracted",
        source_type: str = "dd",
    ) -> None:
        self.sources[source_id] = {
            "status": status,
            "source_type": source_type,
            "produced_sn_id": sn_id,
            "claimed_at": None,
            "claim_token": None,
            "attempt_count": 0,
        }
        self.produced_name[source_id] = sn_id

    def has_live_source(self, sn_id: str) -> bool:
        return any(
            target == sn_id
            and self.sources[sid].get("source_type") != "derived"
            and self.sources[sid].get("status") in ("composed", "attached")
            for sid, target in self.produced_name.items()
        )

    def refined_from_id(self, successor: str) -> str | None:
        return self.refined_from.get(successor)

    def has_successor(self, sn_id: str) -> bool:
        return sn_id in self.refined_from.values()

    def _descendants(self, root: str) -> set[str]:
        """All descendants reachable via inbound HAS_PARENT to *root*."""
        desc: set[str] = set()
        children_of: dict[str, list[str]] = {}
        for child, edges in self.edges_by_child.items():
            for e in edges:
                children_of.setdefault(e["parent_id"], []).append(child)
        stack = [root]
        while stack:
            cur = stack.pop()
            for c in children_of.get(cur, []):
                if c not in desc:
                    desc.add(c)
                    stack.append(c)
        return desc

    def _rename_node_id(self, old_id: str, new_id: str) -> None:
        if old_id in self.nodes:
            self.nodes[new_id] = self.nodes.pop(old_id)
        if old_id in self.edges_by_child:
            self.edges_by_child[new_id] = self.edges_by_child.pop(old_id)
        for edges in self.edges_by_child.values():
            for e in edges:
                if e["parent_id"] == old_id:
                    e["parent_id"] = new_id
        for sid, tgt in list(self.produced_name.items()):
            if tgt == old_id:
                self.produced_name[sid] = new_id
        for succ, pred in list(self.refined_from.items()):
            new_succ = new_id if succ == old_id else succ
            new_pred = new_id if pred == old_id else pred
            if new_succ != succ or new_pred != pred:
                del self.refined_from[succ]
                self.refined_from[new_succ] = new_pred

    # -- GraphClient-compatible surface --------------------------------------

    def __enter__(self) -> FakeGraph:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def close(self) -> None:
        pass

    @contextmanager
    def session(self):
        yield _FakeSession(self)

    # -- query() dispatch -----------------------------------------------------

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:  # noqa: PLR0911, PLR0912
        # ---- edit.py's own marked queries ----
        if "// EDIT_FETCH_TARGET" in cypher:
            sn_id = params["id"]
            node = self.nodes.get(sn_id)
            if node is None:
                return []
            has_children = any(
                e["parent_id"] == sn_id
                for edges in self.edges_by_child.values()
                for e in edges
            )
            return [
                {
                    "name_stage": node.get("name_stage"),
                    "docs_stage": node.get("docs_stage"),
                    "description": node.get("description"),
                    "documentation": node.get("documentation"),
                    "docs_model": node.get("docs_model"),
                    "docs_generated_at": node.get("docs_generated_at"),
                    "kind": node.get("kind"),
                    "unit": node.get("unit"),
                    "physics_domain": node.get("physics_domain"),
                    "origin": node.get("origin"),
                    "tags": node.get("tags"),
                    "chain_length": node.get("chain_length", 0) or 0,
                    "has_successor": self.has_successor(sn_id),
                    "has_children": has_children,
                    "has_live_source": self.has_live_source(sn_id),
                }
            ]

        if "// EDIT_CHECK_COLLISION" in cypher:
            return [{"n": 1 if params["id"] in self.nodes else 0}]

        if "// EDIT_FETCH_PARENT_EDGES" in cypher:
            edges = self.edges_by_child.get(params["id"], [])
            return [
                {
                    "parent_id": e["parent_id"],
                    "operator": e["operator"],
                    "operator_kind": e["operator_kind"],
                    "role": e["role"],
                    "separator": e["separator"],
                }
                for e in edges
            ]

        if "// EDIT_FETCH_SIBLINGS" in cypher:
            parent_id = params["parent_id"]
            target_id = params["target_id"]
            substring = params["substring"]
            n = 0
            for child, edges in self.edges_by_child.items():
                if child == target_id or substring not in child:
                    continue
                if any(e["parent_id"] == parent_id for e in edges):
                    n += 1
            return [{"n": n}]

        if "// EDIT_STAMP_VALIDATION" in cypher:
            node = self.nodes.get(params["id"])
            if node is not None:
                node["validation_status"] = params["status"]
                node["validation_issues"] = params["issues"]
                node["validated_at"] = "now"
            return []

        if "// EDIT_SET_REFINING" in cypher:
            node = self.nodes.get(params["id"])
            if node is not None:
                node["superseded_from_stage"] = (
                    node.get("superseded_from_stage") or node["name_stage"]
                )
                node["name_stage"] = "refining"
            return []

        if "// EDIT_STAMP_FIELDS" in cypher:
            node = self.nodes.get(params["id"])
            if node is not None:
                for key in (
                    "edit_mode",
                    "name_hint",
                    "docs_hint",
                    "edit_reason",
                    "edit_origin",
                    "edit_scope",
                    "edit_status",
                    "edit_requested_at",
                    "run_id",
                ):
                    node[key] = params[key]
            return []

        if "// EDIT_CLAIM_FOR_DOCS_REFINE" in cypher:
            node = self.nodes.get(params["id"])
            if node is not None and node.get("name_stage") == "accepted":
                node["docs_stage"] = "refining"
                node["claim_token"] = params["token"]
                node["claimed_at"] = "now"
            return []

        if "// EDIT_COUNT_PRODUCING_SOURCES" in cypher:
            sn_id = params["id"]
            n = sum(1 for tgt in self.produced_name.values() if tgt == sn_id)
            return [{"n": n}]

        if "// EDIT_RESET_SOURCES" in cypher:
            sn_id = params["id"]
            hit_ids = []
            for source_id, tgt in self.produced_name.items():
                if tgt != sn_id:
                    continue
                src = self.sources.get(source_id)
                if src is None:
                    continue
                src["status"] = "extracted"
                src["claimed_at"] = None
                src["claim_token"] = None
                src["attempt_count"] = 0
                hit_ids.append(source_id)
            return [{"id": sid} for sid in hit_ids]

        # ---- cascade.py: rename_cascade / cascade_descendants_of / shared walk ----
        if "root_exists" in cypher:
            old = params.get("old")
            new = params.get("new")
            return [
                {
                    "root_exists": old in self.nodes,
                    "origin": self.nodes.get(old, {}).get("origin") if old else None,
                    "name_stage": (
                        self.nodes.get(old, {}).get("name_stage") if old else None
                    ),
                    "target_exists": new in self.nodes,
                }
            ]

        if "count(s) AS n" in cypher:
            sn_id = params.get("id")
            return [{"n": 1 if sn_id in self.nodes else 0}]

        if "CASCADE_OLD_ROOT_FALLBACK_GUARD" in cypher:
            successor = params.get("successor")
            old_root = params.get("old_root")
            node = self.nodes.get(successor, {})
            predecessor_id = (
                old_root if self.refined_from.get(successor) == old_root else None
            )
            return [
                {
                    "successor_stage": node.get("name_stage"),
                    "edit_mode": node.get("edit_mode"),
                    "edit_status": node.get("edit_status"),
                    "edit_scope": node.get("edit_scope"),
                    "predecessor_id": predecessor_id,
                    "predecessor_stage": (
                        self.nodes.get(old_root, {}).get("name_stage")
                        if predecessor_id
                        else None
                    ),
                }
            ]

        if "OPTIONAL MATCH path = (parent)<-[:HAS_PARENT" in cypher:
            old = params.get("old")
            desc = self._descendants(old) if old else set()
            return [
                {
                    "id": d,
                    "origin": self.nodes.get(d, {}).get("origin"),
                    "name_stage": self.nodes.get(d, {}).get("name_stage"),
                }
                for d in sorted(desc)
            ]

        if "MATCH (child)-[r:HAS_PARENT]->(target)" in cypher:
            old = params.get("old")
            if not old:
                return []
            subtree = {old} | self._descendants(old)
            rows: list[dict[str, Any]] = []
            for child, edges in self.edges_by_child.items():
                for e in edges:
                    if e["parent_id"] in subtree and child in subtree:
                        rows.append(
                            {
                                "child_id": child,
                                "target_id": e["parent_id"],
                                "operator": e["operator"],
                                "operator_kind": e["operator_kind"],
                                "role": e["role"],
                                "separator": e["separator"],
                                "axis": e.get("axis"),
                                "shape": e.get("shape"),
                            }
                        )
            return rows

        if "WHERE sn IS NOT NULL" in cypher and "UNWIND $ids" in cypher:
            ids = params.get("ids") or []
            return [{"id": nid} for nid in ids if nid in self.nodes]

        if "SET sn.id = r.to" in cypher:
            for r in params.get("renames") or []:
                self._rename_node_id(r["from"], r["to"])
            return []

        # ---- graph_ops.py: persist_reviewed_name ----
        if (
            "sn.edit_status AS edit_status" in cypher
            and "sn.edit_scope AS edit_scope" in cypher
        ):
            node = self.nodes.get(params["id"])
            token = params.get("token")
            if node is None or node.get("name_stage") != "drafted":
                return []
            if not (
                node.get("claim_token") == token or node.get("claim_token") is None
            ):
                return []
            return [
                {
                    "chain_length": node.get("chain_length", 0) or 0,
                    "edit_status": node.get("edit_status"),
                    "edit_scope": node.get("edit_scope"),
                    "edit_override_edits": node.get("edit_override_edits"),
                    "edit_include_accepted": node.get("edit_include_accepted"),
                }
            ]

        if "sn.reviewer_score_name        = $score" in cypher:
            node = self.nodes.get(params["id"])
            token = params.get("token")
            if node is None or node.get("name_stage") != "drafted":
                return []
            if not (
                node.get("claim_token") == token or node.get("claim_token") is None
            ):
                return []
            node["reviewer_score_name"] = params["score"]
            node["reviewer_scores_name"] = params.get("scores_json")
            node["reviewer_comments_name"] = params.get("comments")
            node["name_stage"] = params["target_stage"]
            node["edit_status"] = params.get("new_edit_status")
            node["claim_token"] = None
            node["claimed_at"] = None
            return [{"id": params["id"]}]

        if "-[:REFINED_FROM]->(pred:StandardName)" in cypher:
            pred = self.refined_from.get(params["id"])
            return [{"pred_id": pred}] if pred else []

        if "sn.validation_issues = coalesce(sn.validation_issues, [])" in cypher:
            node = self.nodes.get(params["id"])
            if node is not None:
                existing = node.get("validation_issues") or []
                node["validation_issues"] = existing + [
                    f"[edit_cascade] {c}" for c in params.get("conflicts", [])
                ]
            return []

        # ---- graph_ops.py: persist_reviewed_docs ----
        if (
            "sn.documentation AS documentation" in cypher
            and "sn.edit_status AS edit_status" in cypher
        ):
            node = self.nodes.get(params["id"])
            token = params.get("token")
            if node is None:
                return []
            if not (
                node.get("claim_token") == token or node.get("claim_token") is None
            ):
                return []
            if (
                node.get("docs_stage") != "drafted"
                or node.get("name_stage") != "accepted"
            ):
                return []
            return [
                {
                    "docs_chain_length": node.get("docs_chain_length", 0) or 0,
                    "documentation": node.get("documentation"),
                    "edit_status": node.get("edit_status"),
                }
            ]

        if "sn.reviewer_score_docs        = $score" in cypher:
            node = self.nodes.get(params["id"])
            token = params.get("token")
            if node is None:
                return []
            if not (
                node.get("claim_token") == token or node.get("claim_token") is None
            ):
                return []
            if (
                node.get("name_stage") != "accepted"
                or node.get("docs_stage") != "drafted"
            ):
                return []
            node["reviewer_score_docs"] = params["score"]
            node["docs_stage"] = params["target_stage"]
            node["edit_status"] = params.get("new_edit_status")
            node["claim_token"] = None
            node["claimed_at"] = None
            return [{"id": params["id"]}]

        # ---- graph_ops.py: reset_standard_name_docs ----
        if "AS eligible" in cypher:
            sn_ids = params.get("sn_ids") or []
            n = sum(
                1
                for sid in sn_ids
                if self.nodes.get(sid, {}).get("name_stage") == "accepted"
                and (self.nodes.get(sid, {}).get("docs_stage") or "")
                in ("accepted", "exhausted", "reviewed", "drafted")
            )
            return [{"eligible": n}]

        if "AS reset" in cypher and "docs_chain_length" in cypher:
            sn_ids = params.get("sn_ids") or []
            n = 0
            for sid in sn_ids:
                node = self.nodes.get(sid)
                if node is None:
                    continue
                if node.get("name_stage") != "accepted":
                    continue
                if (node.get("docs_stage") or "") not in (
                    "accepted",
                    "exhausted",
                    "reviewed",
                    "drafted",
                ):
                    continue
                node["docs_stage"] = "pending"
                node["docs_model"] = None
                node["docs_generated_at"] = None
                node["claim_token"] = None
                node["claimed_at"] = None
                node["reviewer_score_docs"] = None
                run_id = params.get("run_id")
                if run_id is not None:
                    node["run_id"] = run_id
                n += 1
            return [{"reset": n}]

        if "AS current_bindings" in cypher and "manifest_recorded" in cypher:
            rows = []
            for source_id in params.get("source_ids") or []:
                source = self.sources.get(source_id)
                target = self.produced_name.get(source_id)
                rows.append(
                    {
                        "source_id": source_id,
                        "source_exists": source is not None,
                        "source_status": source.get("status") if source else None,
                        "scalar_binding": (
                            source.get("produced_sn_id") if source else None
                        ),
                        "actively_claimed": bool(
                            source
                            and (
                                source.get("claimed_at") is not None
                                or source.get("claim_token") is not None
                            )
                        ),
                        "current_bindings": [target] if target else [],
                        "manifest_recorded": (
                            params["manifest_event_id"] in self.source_migration_events
                        ),
                    }
                )
            return rows

        # Default: best-effort helpers (_doc_link_mismatches etc.) — safe no-op.
        return []

    # -- transaction surface (persist_refined_name / persist_refined_docs) --

    def _tx_run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "// REFINE_ATOMIC_PREFLIGHT" in cypher:
            old_name = params["old_name"]
            new_name = params["new_name"]
            old = self.nodes.get(old_name)
            expected_stage = params.get("expected_old_stage") or "refining"
            if old is None or old.get("name_stage") != expected_stage:
                return []
            existing = self.nodes.get(new_name)
            if existing is not None and not (
                params.get("edit_mode") is None
                and (existing.get("name_stage") or "") in ("", "pending")
                and (existing.get("origin") or "") != "derived"
            ):
                return []
            old["superseded_from_stage"] = (
                old.get("superseded_from_stage") or old["name_stage"]
            )
            old["name_stage"] = "refining"
            edit_fields = (
                "edit_mode",
                "name_hint",
                "docs_hint",
                "edit_reason",
                "edit_origin",
                "edit_scope",
                "edit_status",
                "edit_requested_at",
                "edit_override_edits",
                "edit_include_accepted",
            )
            source_edit_state = {field: old.get(field) for field in edit_fields}
            carry_open_edit = bool(params.get("inherit_open_edit")) and (
                old.get("edit_status") == "open"
            )
            successor_edit_state = {
                field: old.get(field) if carry_open_edit else params.get(field)
                for field in edit_fields
            }
            new_row = dict(_DEFAULT_NODE_FIELDS)
            new_row.update(
                name_stage="drafted",
                docs_stage="pending",
                origin="pipeline",
                chain_length=params["new_chain_length"],
                docs_chain_length=0,
                description=params["description"],
                kind=params["kind"],
                unit=params["unit"],
                physics_domain=params["physics_domain"],
                tags=params["tags"],
                model=params["model"],
                run_id=params.get("run_id"),
                **successor_edit_state,
            )
            if existing is None:
                self.nodes[new_name] = new_row
            else:
                existing["name_stage"] = "drafted"
                existing["origin"] = "pipeline"
                existing.update(successor_edit_state)
            return [
                {
                    "new_name": new_name,
                    "old_name": old_name,
                    "source_ids": sorted(
                        sid
                        for sid, target in self.produced_name.items()
                        if target == old_name
                    ),
                    **{
                        f"source_{field}": value
                        for field, value in source_edit_state.items()
                    },
                    **{
                        f"effective_{field}": value
                        for field, value in successor_edit_state.items()
                    },
                }
            ]

        if "MERGE (new)-[:REFINED_FROM]->(old)" in cypher:
            old_name = params["old_name"]
            new_name = params["new_name"]
            old = self.nodes.get(old_name)
            if old is None or old.get("name_stage") != "refining":
                return []
            fenced_fields = (
                "edit_mode",
                "name_hint",
                "docs_hint",
                "edit_reason",
                "edit_origin",
                "edit_scope",
                "edit_status",
                "edit_requested_at",
                "edit_override_edits",
                "edit_include_accepted",
            )
            if any(
                old.get(field) != params.get(f"source_{field}")
                for field in fenced_fields
            ):
                return []
            self.refined_from[new_name] = old_name
            old["name_stage"] = "superseded"
            old["claim_token"] = None
            old["claimed_at"] = None
            for _child, edges in self.edges_by_child.items():
                for e in edges:
                    if e["parent_id"] == old_name:
                        e["parent_id"] = new_name
            return [{"new_name": new_name, "old_name": old_name}]

        if "RETURN size(moved) AS moved" in cypher:
            old_name = params["old_name"]
            new_name = params["new_name"]
            selected = list(params.get("source_ids") or [])
            if not selected:
                return []
            if old_name not in self.nodes or new_name not in self.nodes:
                return []
            if any(
                source_id not in self.sources
                or self.sources[source_id].get("status") == "stale"
                or self.sources[source_id].get("claimed_at") is not None
                or self.sources[source_id].get("claim_token") is not None
                or self.sources[source_id].get("produced_sn_id") != old_name
                or self.produced_name.get(source_id) != old_name
                for source_id in selected
            ):
                return []

            for source_id in selected:
                self.produced_name[source_id] = new_name
                self.sources[source_id]["produced_sn_id"] = new_name

            if any(
                self.produced_name.get(source_id) != new_name
                or self.sources[source_id].get("produced_sn_id") != new_name
                for source_id in selected
            ):
                return []

            self.source_migration_events.add(params["manifest_event_id"])
            self.nodes[old_name]["source_paths"] = sorted(
                source_id
                for source_id, target in self.produced_name.items()
                if target == old_name
            )
            self.nodes[new_name]["source_paths"] = sorted(
                source_id
                for source_id, target in self.produced_name.items()
                if target == new_name
            )
            return [{"moved": len(selected)}]

        if "CREATE (change:StandardNameChange" in cypher:
            return []

        if "cur_desc" in params:
            sn_id = params["sn_id"]
            token = params["token"]
            node = self.nodes.get(sn_id)
            if node is None:
                return []
            if node.get("claim_token") != token:
                return []
            if (
                node.get("docs_stage") != "refining"
                or node.get("name_stage") != "accepted"
            ):
                return []
            cur_chain = node.get("docs_chain_length", 0) or 0
            revision_id = f"{sn_id}#rev-{cur_chain}"
            self.docs_revisions.append(revision_id)
            node["description"] = params["new_desc"]
            node["documentation"] = params["new_doc"]
            node["docs_stage"] = "drafted"
            node["docs_chain_length"] = cur_chain + 1
            node["docs_model"] = params["model"]
            node["claim_token"] = None
            node["claimed_at"] = None
            node["reviewer_score_docs"] = None
            return [{"docs_chain_length": cur_chain + 1, "revision_id": revision_id}]

        return []


class _FakeTx:
    def __init__(self, graph: FakeGraph) -> None:
        self.owner = graph
        self.graph = deepcopy(graph)
        self._closed = False

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "new_name" in params and "old_name" in params:
            return self.graph._tx_run(cypher, **params)
        if "cur_desc" in params:
            return self.graph._tx_run(cypher, **params)
        return self.graph.query(cypher, **params)

    def commit(self) -> None:
        self.owner.nodes = self.graph.nodes
        self.owner.edges_by_child = self.graph.edges_by_child
        self.owner.refined_from = self.graph.refined_from
        self.owner.produced_name = self.graph.produced_name
        self.owner.sources = self.graph.sources
        self.owner.docs_revisions = self.graph.docs_revisions
        self.owner.source_migration_events = self.graph.source_migration_events
        self._closed = True

    def closed(self) -> bool:
        return self._closed

    def rollback(self) -> None:
        self._closed = True

    def close(self) -> None:
        self._closed = True


class _FakeSession:
    def __init__(self, graph: FakeGraph) -> None:
        self.graph = graph

    def begin_transaction(self) -> _FakeTx:
        return _FakeTx(self.graph)


@contextmanager
def _patched_graph(fake: FakeGraph):
    """Patch graph_ops.GraphClient() to return *fake* + suppress write_reviews."""
    with patch(_GC_PATH, return_value=fake), patch(_WRITE_REVIEWS_PATH):
        yield


# =============================================================================
# Mode / axis / scope resolution + validation errors
# =============================================================================


class TestValidation:
    def test_two_modes_at_once_raises(self) -> None:
        fake = FakeGraph()
        with pytest.raises(ValueError, match="exactly one of"):
            apply_edit(target="x", hint="a", rename="b", reason="because", gc=fake)

    def test_no_mode_raises(self) -> None:
        fake = FakeGraph()
        with pytest.raises(ValueError, match="exactly one of"):
            apply_edit(target="x", reason="because", gc=fake)

    def test_missing_reason_raises(self) -> None:
        fake = FakeGraph()
        with pytest.raises(ValueError, match="non-empty reason"):
            apply_edit(target="x", hint="steer this way", reason="", gc=fake)

    def test_blank_reason_raises(self) -> None:
        fake = FakeGraph()
        with pytest.raises(ValueError, match="non-empty reason"):
            apply_edit(target="x", hint="steer this way", reason="   ", gc=fake)

    def test_invalid_origin_raises(self) -> None:
        fake = FakeGraph()
        with pytest.raises(ValueError, match="origin"):
            apply_edit(target="x", hint="steer", reason="why", origin="robot", gc=fake)

    def test_invalid_scope_raises(self) -> None:
        fake = FakeGraph()
        with pytest.raises(ValueError, match="scope"):
            apply_edit(
                target="x", hint="steer", reason="why", scope="everything", gc=fake
            )

    def test_unknown_target_is_blocked_not_raised(self) -> None:
        fake = FakeGraph()
        with _patched_graph(fake):
            plan = apply_edit(
                target="does_not_exist", hint="steer", reason="why", gc=fake
            )
        assert plan.blocked is not None
        assert "not found" in plan.blocked
        assert plan.applied is False

    def test_hint_axis_default_is_name(self) -> None:
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="accepted")
        fake.add_source("dd:x", sn_id="electron_temperature")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                hint="steer",
                reason="why",
                dry_run=True,
                gc=fake,
            )
        assert plan.axis == "name"
        assert plan.entry == "generate"

    def test_hint_invalid_axis_raises(self) -> None:
        fake = FakeGraph()
        with pytest.raises(ValueError, match="axis"):
            apply_edit(target="x", hint="steer", reason="why", axis="bogus", gc=fake)

    def test_rename_axis_forced_to_name(self) -> None:
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                rename="electron_density",
                reason="why",
                axis="docs",
                dry_run=True,
                gc=fake,
            )
        assert plan.axis == "name"

    def test_scope_default_leaf_only_self(self) -> None:
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                rename="electron_density",
                reason="why",
                dry_run=True,
                gc=fake,
            )
        assert plan.scope == "only_self"

    def test_scope_default_parent_subtree(self) -> None:
        fake = FakeGraph()
        fake.add_node("temperature", name_stage="accepted")
        fake.add_node("electron_temperature", name_stage="accepted")
        fake.add_edge("electron_temperature", "temperature", "electron", "qualifier")
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="density",
                reason="why",
                dry_run=True,
                gc=fake,
            )
        assert plan.scope == "subtree"


# =============================================================================
# Rename mode
# =============================================================================


class TestRenameEligibility:
    @pytest.mark.parametrize("stage", ["accepted", "reviewed", "exhausted", "drafted"])
    def test_eligible_stage_rename_applies(self, stage: str) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage=stage)
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="clarify component",
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is True
        assert plan.successor == "toroidal_plasma_current"
        assert "toroidal_plasma_current" in fake.nodes
        assert fake.nodes["toroidal_plasma_current"]["name_stage"] == "drafted"
        assert fake.nodes["plasma_current"]["name_stage"] == "superseded"
        assert fake.nodes["plasma_current"]["superseded_from_stage"] == stage

    def test_superseded_without_successor_is_eligible(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="superseded")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="resurrect with a clearer name",
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is True

    def test_superseded_with_successor_refused(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="superseded")
        fake.add_node("toroidal_plasma_current", name_stage="accepted")
        fake.refined_from["toroidal_plasma_current"] = "plasma_current"
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="ion_pressure",
                reason="try again",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "successor" in plan.blocked
        assert plan.applied is False

    def test_pending_stage_without_live_source_refused(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="pending")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "not eligible" in plan.blocked
        assert "no live source" in plan.blocked
        assert plan.applied is False
        assert "toroidal_plasma_current" not in fake.nodes

    def test_isn_invalid_rename_refused(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="totally_bogus_not_a_real_isn_name_zzz",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "ISN grammar" in plan.blocked
        assert "plasma_current" in fake.nodes
        assert fake.nodes["plasma_current"]["name_stage"] == "accepted"

    def test_collision_refused(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="accepted")
        fake.add_node("toroidal_plasma_current", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "already exists" in plan.blocked

    def test_dry_run_makes_zero_writes(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                dry_run=True,
                gc=fake,
            )
        assert plan.applied is False
        assert plan.blocked is None
        assert "toroidal_plasma_current" not in fake.nodes
        assert fake.nodes["plasma_current"]["name_stage"] == "accepted"

    def test_edit_fields_stamped_on_successor(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="clarify component",
                origin="agent",
                gc=fake,
            )
        succ = fake.nodes[plan.successor]
        assert succ["edit_mode"] == "rename"
        assert succ["name_hint"] == "toroidal_plasma_current"
        assert succ["edit_reason"] == "clarify component"
        assert succ["edit_origin"] == "agent"
        assert succ["edit_status"] == "open"
        assert succ["edit_requested_at"] is not None

    def test_run_id_stamped_on_successor_and_plan(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="clarify component",
                gc=fake,
            )
        assert plan.run_id is not None
        assert plan.run_id.startswith("sn-edit-")
        assert fake.nodes[plan.successor]["run_id"] == plan.run_id

    def test_run_id_is_none_for_blocked_rename(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="pending")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert plan.run_id is None

    def test_run_id_is_none_for_dry_run_rename(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                dry_run=True,
                gc=fake,
            )
        assert plan.run_id is None


# =============================================================================
# Rename of a name stranded below the review entry stage
# =============================================================================


def _stranded(
    *,
    name_stage: str | None = "pending",
    source_status: str = "composed",
    source_type: str = "dd",
    **node_fields: Any,
) -> FakeGraph:
    """A name resting below review entry that still holds a produced source."""
    fake = FakeGraph()
    fake.add_node("plasma_current", name_stage=name_stage, **node_fields)
    fake.add_source(
        "dd:magnetics/ip",
        sn_id="plasma_current",
        status=source_status,
        source_type=source_type,
    )
    return fake


@contextmanager
def _admitting_pairing_guard():
    """Admit every source pairing the rename hands to the write-time guard.

    That guard resolves DD paths, units and existing bindings against the real
    graph, which the FakeGraph does not model; its own behaviour is covered by
    the attachment-audit tests. Stubbing it keeps these tests on the one thing
    they pin — which names the rename admits.
    """
    from imas_codex.standard_names.attachment_audit import (
        AttachmentPairingGuardResult,
    )

    def _admit(gc: Any, sn_id: str, source_ids: list[str]):
        return AttachmentPairingGuardResult(tuple(sorted(set(source_ids))), ())

    with patch(
        "imas_codex.standard_names.attachment_audit.guard_source_pairings", _admit
    ):
        yield


class TestStrandedRename:
    """A quarantined, source-backed name below review entry can be renamed.

    Nothing in the pool loop can move such a name: the generate pool claims
    sources at ``status='extracted'`` (these are already composed/attached),
    the review pool claims ``name_stage='drafted'``, and a quarantined name is
    excluded from the review claim regardless of stage — so no stage advance
    would make it claimable. The rename edit is its only repair vehicle, and
    the replacement it mints is reviewed like any other candidate.
    """

    def test_quarantined_pending_with_live_source_is_renameable(self) -> None:
        from imas_codex.standard_names.provenance_lifecycle import (
            retarget_standard_name_sources,
        )

        fake = _stranded(validation_status="quarantined")
        with (
            _patched_graph(fake),
            _admitting_pairing_guard(),
            patch(
                "imas_codex.standard_names.provenance_lifecycle."
                "retarget_standard_name_sources",
                wraps=retarget_standard_name_sources,
            ) as migrate_sources,
        ):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="parser-invalid spelling",
                gc=fake,
            )
        migrate_sources.assert_called_once()
        assert migrate_sources.call_args.kwargs["source_ids"] == ["dd:magnetics/ip"]
        assert migrate_sources.call_args.kwargs["expected_current_bindings"] == {
            "dd:magnetics/ip": "plasma_current"
        }
        assert plan.blocked is None
        assert plan.applied is True
        assert plan.successor == "toroidal_plasma_current"
        assert fake.nodes["plasma_current"]["name_stage"] == "superseded"
        assert fake.nodes["plasma_current"]["superseded_from_stage"] == "pending"

    def test_renamed_product_enters_review_not_acceptance(self) -> None:
        fake = _stranded(validation_status="quarantined")
        with _patched_graph(fake), _admitting_pairing_guard():
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="parser-invalid spelling",
                gc=fake,
            )
        assert plan.entry == "review_name"
        successor = fake.nodes["toroidal_plasma_current"]
        assert successor["name_stage"] == "drafted"
        assert successor["reviewer_score_name"] is None

    def test_rename_leaves_predecessor_quarantine_intact(self) -> None:
        fake = _stranded(validation_status="quarantined")
        with _patched_graph(fake), _admitting_pairing_guard():
            apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="parser-invalid spelling",
                gc=fake,
            )
        assert fake.nodes["plasma_current"]["validation_status"] == "quarantined"
        # The replacement is judged on its own merits by the same admission
        # gate a generated candidate passes, not by inheriting 'quarantined'.
        successor = fake.nodes["toroidal_plasma_current"]
        assert successor["validated_at"] == "now"
        assert successor["validation_status"] in ("valid", "quarantined")

    def test_dry_run_admits_without_writing(self) -> None:
        fake = _stranded(validation_status="quarantined")
        before = deepcopy(fake.nodes)
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="parser-invalid spelling",
                dry_run=True,
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is False
        assert fake.nodes == before

    def test_successor_refuses_and_names_the_node_to_edit(self) -> None:
        fake = _stranded(validation_status="quarantined")
        fake.add_node("area_of_flux_surface", name_stage="accepted")
        fake.refined_from["area_of_flux_surface"] = "plasma_current"
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="parser-invalid spelling",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "edit the successor instead" in plan.blocked
        assert plan.applied is False
        assert "toroidal_plasma_current" not in fake.nodes

    @pytest.mark.parametrize("source_status", ["stale", "extracted", "failed"])
    def test_no_live_source_refused(self, source_status: str) -> None:
        fake = _stranded(source_status=source_status)
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "no live source" in plan.blocked
        assert plan.applied is False

    def test_derived_structural_source_is_not_a_live_source(self) -> None:
        fake = _stranded(source_type="derived")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "no live source" in plan.blocked

    def test_empty_description_refused(self) -> None:
        fake = _stranded(description="   ")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "no description" in plan.blocked
        assert plan.applied is False

    def test_derived_parent_refused(self) -> None:
        fake = _stranded(origin="derived")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "derived structural parent" in plan.blocked
        assert plan.applied is False

    def test_bare_node_without_stage_refused(self) -> None:
        fake = _stranded(name_stage=None)
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "carries no name_stage" in plan.blocked
        assert plan.applied is False

    @pytest.mark.parametrize("stage", ["refining", "contested"])
    def test_other_ineligible_stages_unchanged(self, stage: str) -> None:
        fake = _stranded(name_stage=stage)
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "not eligible for rename" in plan.blocked
        assert plan.applied is False


# =============================================================================
# persist_reviewed_name edit lifecycle
# =============================================================================


class TestEditLifecycle:
    def test_open_to_applied_on_accept(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        fake.add_node(
            "toroidal_plasma_current",
            name_stage="drafted",
            edit_status="open",
            edit_scope="only_self",
            claim_token="tok",
        )
        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="toroidal_plasma_current",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )
        assert stage == "accepted"
        assert fake.nodes["toroidal_plasma_current"]["edit_status"] == "applied"

    def test_open_to_exhausted_on_cap(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        fake.add_node(
            "toroidal_plasma_current",
            name_stage="drafted",
            edit_status="open",
            edit_scope="only_self",
            claim_token="tok",
            chain_length=3,
        )
        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="toroidal_plasma_current",
                claim_token="tok",
                score=0.10,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )
        assert stage == "exhausted"
        assert fake.nodes["toroidal_plasma_current"]["edit_status"] == "exhausted"

    def test_open_stays_open_on_reviewed(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        fake.add_node(
            "toroidal_plasma_current",
            name_stage="drafted",
            edit_status="open",
            edit_scope="only_self",
            claim_token="tok",
            chain_length=0,
        )
        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="toroidal_plasma_current",
                claim_token="tok",
                score=0.10,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )
        assert stage == "reviewed"
        assert fake.nodes["toroidal_plasma_current"]["edit_status"] == "open"

    def test_non_edit_node_untouched(self) -> None:
        """A regular (non-edit) drafted name's edit_status stays None."""
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        fake.add_node(
            "toroidal_plasma_current",
            name_stage="drafted",
            claim_token="tok",
        )
        with _patched_graph(fake):
            persist_reviewed_name(
                sn_id="toroidal_plasma_current",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )
        assert fake.nodes["toroidal_plasma_current"]["edit_status"] is None


# =============================================================================
# refine-rotation propagation of open edit fields
# =============================================================================


class TestRefineRotationPropagation:
    def test_open_edit_fields_propagate_across_refine(self) -> None:
        """A normal pipeline refine call (no edit kwargs) copies the open
        edit fields from predecessor to successor."""
        from imas_codex.standard_names.graph_ops import persist_refined_name

        fake = FakeGraph()
        fake.add_node(
            "toroidal_plasma_current",
            name_stage="refining",
            edit_mode="rename",
            name_hint="toroidal_plasma_current",
            edit_reason="clarify component",
            edit_origin="human",
            edit_scope="only_self",
            edit_status="open",
            edit_requested_at="2026-01-01T00:00:00+00:00",
        )
        with _patched_graph(fake):
            result = persist_refined_name(
                old_name="toroidal_plasma_current",
                new_name="poloidal_plasma_current",
                description="a description",
                old_chain_length=1,
                model="refine/x",
                reason="reviewer wanted a different qualifier",
            )
        new_node = fake.nodes[result["new_name"]]
        assert new_node["edit_mode"] == "rename"
        assert new_node["edit_reason"] == "clarify component"
        assert new_node["edit_status"] == "open"
        assert new_node["edit_requested_at"] == "2026-01-01T00:00:00+00:00"

    def test_closed_edit_does_not_propagate(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_refined_name

        fake = FakeGraph()
        fake.add_node(
            "toroidal_plasma_current",
            name_stage="refining",
            edit_mode="rename",
            edit_status="applied",  # already resolved — should NOT propagate
        )
        fake.add_source("dd:magnetics/ip", sn_id="toroidal_plasma_current")
        with _patched_graph(fake), _admitting_pairing_guard():
            result = persist_refined_name(
                old_name="toroidal_plasma_current",
                new_name="poloidal_plasma_current",
                description="a description",
                old_chain_length=1,
                model="refine/x",
            )
        new_node = fake.nodes[result["new_name"]]
        assert new_node["edit_mode"] is None
        assert new_node["edit_status"] is None

    def test_explicit_edit_kwargs_are_not_overridden_by_propagation(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_refined_name

        fake = FakeGraph()
        fake.add_node(
            "toroidal_plasma_current",
            name_stage="refining",
            edit_status="open",
            edit_reason="old reason",
        )
        with _patched_graph(fake):
            result = persist_refined_name(
                old_name="toroidal_plasma_current",
                new_name="poloidal_plasma_current",
                description="a description",
                old_chain_length=0,
                model="sn-edit",
                edit_mode="rename",
                edit_reason="explicit new reason",
                edit_status="open",
            )
        new_node = fake.nodes[result["new_name"]]
        assert new_node["edit_reason"] == "explicit new reason"


# =============================================================================
# Docs mode + hint mode
# =============================================================================


class TestDocsMode:
    def test_docs_replacement_enters_review(self) -> None:
        fake = FakeGraph()
        fake.add_node(
            "electron_temperature",
            name_stage="accepted",
            docs_stage="accepted",
            description="old desc",
            documentation="old docs",
        )
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new replacement documentation",
                reason="fix a physics error",
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is True
        assert plan.entry == "review_docs"
        node = fake.nodes["electron_temperature"]
        assert node["documentation"] == "new replacement documentation"
        assert node["docs_stage"] == "drafted"
        assert node["edit_mode"] == "docs"
        assert node["edit_status"] == "open"
        assert node["docs_hint"] == "new replacement documentation"
        assert plan.run_id is not None
        assert plan.run_id.startswith("sn-edit-")
        assert node["run_id"] == plan.run_id

    def test_docs_requires_accepted_name(self) -> None:
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="drafted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new docs",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "accepted" in plan.blocked

    def test_docs_dry_run_zero_writes(self) -> None:
        fake = FakeGraph()
        fake.add_node(
            "electron_temperature",
            name_stage="accepted",
            docs_stage="accepted",
            documentation="old docs",
        )
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                docs="new docs",
                reason="why",
                dry_run=True,
                gc=fake,
            )
        assert plan.applied is False
        assert plan.run_id is None
        assert fake.nodes["electron_temperature"]["documentation"] == "old docs"


class TestHintMode:
    def test_hint_name_axis_resets_sources(self) -> None:
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="reviewed")
        fake.add_source(
            "dd:core_profiles/electrons/temperature",
            sn_id="electron_temperature",
            status="composed",
        )
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                hint="prefer core-region wording",
                reason="reviewer flagged ambiguity",
                gc=fake,
            )
        assert plan.applied is True
        assert plan.entry == "generate"
        assert (
            fake.sources["dd:core_profiles/electrons/temperature"]["status"]
            == "extracted"
        )
        node = fake.nodes["electron_temperature"]
        assert node["edit_mode"] == "hint"
        assert node["name_hint"] == "prefer core-region wording"
        assert node["edit_status"] == "open"
        assert plan.run_id is not None
        assert plan.run_id.startswith("sn-edit-")
        assert node["run_id"] == plan.run_id

    def test_hint_docs_axis_resets_docs_stage(self) -> None:
        fake = FakeGraph()
        fake.add_node(
            "electron_temperature",
            name_stage="accepted",
            docs_stage="accepted",
            documentation="old docs",
        )
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                hint="mention Thomson scattering",
                axis="docs",
                reason="missing measurement context",
                gc=fake,
            )
        assert plan.applied is True
        node = fake.nodes["electron_temperature"]
        assert node["docs_hint"] == "mention Thomson scattering"
        assert node["docs_stage"] == "pending"
        assert node["run_id"] == plan.run_id

    def test_hint_dry_run_zero_writes(self) -> None:
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="accepted")
        fake.add_source("dd:x", sn_id="electron_temperature")
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                hint="steer",
                reason="why",
                dry_run=True,
                gc=fake,
            )
        assert plan.applied is False
        assert plan.run_id is None
        assert fake.nodes["electron_temperature"]["edit_mode"] is None

    def test_hint_name_axis_on_sourceless_target_blocks(self) -> None:
        """A derived/structural name has no producing source — a name-axis
        hint cannot regenerate it, so it must be a hard block (not a silent
        applied=True that strands edit_status='open')."""
        fake = FakeGraph()
        fake.add_node("plasma_boundary", name_stage="reviewed", origin="derived")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_boundary",
                hint="steer the base",
                reason="reviewer flagged it",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "no producing" in plan.blocked
        assert plan.applied is False
        assert plan.run_id is None
        # Nothing stamped — no stuck-open edit.
        assert fake.nodes["plasma_boundary"]["edit_status"] is None

    def test_hint_docs_axis_on_sourceless_target_allowed(self) -> None:
        """A docs-axis hint on a derived name is fine — it steers docs, not
        the name, so the zero-sources block does not apply."""
        fake = FakeGraph()
        fake.add_node(
            "plasma_boundary",
            name_stage="accepted",
            docs_stage="accepted",
            origin="derived",
            documentation="old docs",
        )
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_boundary",
                hint="mention the LCFS",
                axis="docs",
                reason="clarify",
                gc=fake,
            )
        assert plan.blocked is None
        assert plan.applied is True
        assert fake.nodes["plasma_boundary"]["docs_hint"] == "mention the LCFS"

    def test_hint_superseded_with_successor_refused(self) -> None:
        fake = FakeGraph()
        fake.add_node("electron_temperature", name_stage="superseded")
        fake.add_node("ion_temperature", name_stage="accepted")
        fake.refined_from["ion_temperature"] = "electron_temperature"
        with _patched_graph(fake):
            plan = apply_edit(
                target="electron_temperature",
                hint="steer",
                reason="why",
                gc=fake,
            )
        assert plan.blocked is not None
        assert "successor" in plan.blocked


# =============================================================================
# Rename cascade protections (override_edits / include_accepted opt-in)
# =============================================================================


class TestRenameCascadeProtections:
    """A subtree ``--rename`` must not silently clobber catalog-edited or
    accepted descendants. Without the opt-in flags they surface as cascade
    conflicts that BLOCK the edit; the flags allow the cascade; and the
    recorded flag values drive the acceptance-time cascade."""

    def _parent_with_child(self, **child_fields: Any) -> FakeGraph:
        # temperature (accepted parent) → ion_temperature (qualifier 'ion').
        # Renaming the parent to a locus form cascades the child. All names
        # round-trip through ISN grammar.
        fake = FakeGraph()
        fake.add_node("temperature", name_stage="accepted")
        fake.add_node("ion_temperature", **child_fields)
        fake.add_edge("ion_temperature", "temperature", "ion", "qualifier")
        return fake

    def _stale_fallback_graph(self) -> FakeGraph:
        fake = FakeGraph()
        fake.add_node(
            "temperature_of_plasma_boundary",
            name_stage="drafted",
            edit_mode="rename",
            edit_status="open",
            edit_scope="subtree",
            edit_include_accepted=True,
            edit_override_edits=False,
            claim_token="tok",
        )
        fake.add_node("temperature", name_stage="superseded")
        fake.refined_from["temperature_of_plasma_boundary"] = "temperature"
        fake.add_node("ion_temperature", name_stage="accepted")
        fake.add_edge("ion_temperature", "temperature", "ion", "qualifier")
        return fake

    def test_catalog_edit_descendant_blocks_by_default(self) -> None:
        fake = self._parent_with_child(name_stage="drafted", origin="catalog_edit")
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="temperature_of_plasma_boundary",
                reason="clarify the locus",
                dry_run=True,
                gc=fake,
            )
        assert plan.blocked is not None
        assert "catalog_edit" in plan.blocked
        assert plan.applied is False

    def test_override_edits_allows_catalog_edit_descendant(self) -> None:
        fake = self._parent_with_child(name_stage="drafted", origin="catalog_edit")
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="temperature_of_plasma_boundary",
                reason="clarify the locus",
                override_edits=True,
                dry_run=True,
                gc=fake,
            )
        assert plan.blocked is None
        assert any(
            r["to"] == "ion_temperature_of_plasma_boundary"
            for r in plan.cascade_planned
        )

    def test_accepted_descendant_blocks_by_default(self) -> None:
        fake = self._parent_with_child(name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="temperature_of_plasma_boundary",
                reason="clarify the locus",
                dry_run=True,
                gc=fake,
            )
        assert plan.blocked is not None
        assert "name_stage='accepted'" in plan.blocked
        assert plan.applied is False

    def test_include_accepted_allows_accepted_descendant(self) -> None:
        fake = self._parent_with_child(name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="temperature_of_plasma_boundary",
                reason="clarify the locus",
                include_accepted=True,
                dry_run=True,
                gc=fake,
            )
        assert plan.blocked is None
        assert any(
            r["to"] == "ion_temperature_of_plasma_boundary"
            for r in plan.cascade_planned
        )

    def test_default_flags_recorded_false_on_successor(self) -> None:
        fake = FakeGraph()
        fake.add_node("plasma_current", name_stage="accepted")
        with _patched_graph(fake):
            plan = apply_edit(
                target="plasma_current",
                rename="toroidal_plasma_current",
                reason="clarify component",
                gc=fake,
            )
        succ = fake.nodes[plan.successor]
        assert succ["edit_override_edits"] is False
        assert succ["edit_include_accepted"] is False

    def test_opt_in_flags_recorded_on_successor(self) -> None:
        fake = self._parent_with_child(name_stage="accepted", origin="catalog_edit")
        with _patched_graph(fake):
            plan = apply_edit(
                target="temperature",
                rename="temperature_of_plasma_boundary",
                reason="clarify the locus",
                override_edits=True,
                include_accepted=True,
                gc=fake,
            )
        assert plan.applied is True
        succ = fake.nodes[plan.successor]
        assert succ["edit_override_edits"] is True
        assert succ["edit_include_accepted"] is True

    def test_acceptance_cascade_honours_recorded_include_accepted(self) -> None:
        """Accepting the root rename cascades the accepted descendant only
        because include_accepted was recorded True at edit time."""
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        fake.add_node(
            "temperature_of_plasma_boundary",
            name_stage="drafted",
            edit_status="open",
            edit_scope="subtree",
            edit_include_accepted=True,
            edit_override_edits=False,
            claim_token="tok",
        )
        fake.add_node("temperature", name_stage="superseded")
        fake.refined_from["temperature_of_plasma_boundary"] = "temperature"
        fake.add_node("ion_temperature", name_stage="accepted")
        fake.add_edge(
            "ion_temperature", "temperature_of_plasma_boundary", "ion", "qualifier"
        )
        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="temperature_of_plasma_boundary",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )
        assert stage == "accepted"
        assert "ion_temperature_of_plasma_boundary" in fake.nodes
        assert "ion_temperature" not in fake.nodes

    def test_acceptance_cascade_refuses_when_recorded_false(self) -> None:
        """When include_accepted was NOT recorded, an accepted descendant
        makes the cascade preflight conflict — so the acceptance is refused
        atomically (nothing accepted, nothing renamed) rather than landing a
        grammar-inconsistent half-cascade with the accepted child stranded."""
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = FakeGraph()
        fake.add_node(
            "temperature_of_plasma_boundary",
            name_stage="drafted",
            edit_status="open",
            edit_scope="subtree",
            edit_include_accepted=False,
            edit_override_edits=False,
            claim_token="tok",
        )
        fake.add_node("ion_temperature", name_stage="accepted")
        fake.add_edge(
            "ion_temperature", "temperature_of_plasma_boundary", "ion", "qualifier"
        )
        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="temperature_of_plasma_boundary",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )
        # Acceptance refused — the protected (accepted, non-opted-in)
        # descendant cannot ride the cascade.
        assert stage == "reviewed"
        assert fake.nodes["temperature_of_plasma_boundary"]["edit_status"] == "open"
        # Nothing renamed: the accepted descendant is untouched.
        assert "ion_temperature" in fake.nodes
        assert "ion_temperature_of_plasma_boundary" not in fake.nodes
        # … and the refusal reason is surfaced for operator follow-up.
        issues = fake.nodes["temperature_of_plasma_boundary"].get("validation_issues")
        assert issues and any("edit_cascade" in v for v in issues)

    def test_old_root_fallback_collision_refuses_acceptance(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = self._stale_fallback_graph()
        fake.add_node(
            "ion_temperature_of_plasma_boundary",
            name_stage="accepted",
        )

        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="temperature_of_plasma_boundary",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert stage == "reviewed"
        successor = fake.nodes["temperature_of_plasma_boundary"]
        assert successor["name_stage"] == "reviewed"
        assert successor["edit_status"] == "open"
        assert "ion_temperature" in fake.nodes
        assert fake.edges_by_child["ion_temperature"][0]["parent_id"] == "temperature"

    def test_old_root_fallback_grammar_failure_refuses_acceptance(self) -> None:
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        fake = self._stale_fallback_graph()
        fake.edges_by_child["ion_temperature"][0]["operator"] = "not_a_grammar_token"

        with _patched_graph(fake):
            stage = persist_reviewed_name(
                sn_id="temperature_of_plasma_boundary",
                claim_token="tok",
                score=0.95,
                model="reviewer/x",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert stage == "reviewed"
        successor = fake.nodes["temperature_of_plasma_boundary"]
        assert successor["name_stage"] == "reviewed"
        assert successor["edit_status"] == "open"
        assert "ion_temperature" in fake.nodes
        assert fake.edges_by_child["ion_temperature"][0]["parent_id"] == "temperature"

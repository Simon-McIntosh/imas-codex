"""Tests for the parent-rename cascade machinery.

The cascade is exercised in three layers:

1. **Pure rule dispatch** — ``_cascade_target_name`` produces the
   correct child name for each ``operator_kind``.
2. **Mock-graph integration** — ``rename_cascade`` walks a stub
   ``query`` interface and produces correct plans, conflicts, and
   safety rejections without touching a live Neo4j.
3. **Audit log** — the rename writes a structured line per change.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names.cascade import (
    CascadeResult,
    _cascade_target_name,
    _isn_round_trip_ok,
    cascade_descendants_of,
    rename_cascade,
)
from imas_codex.standard_names.graph_ops import (
    reconcile_structural_edges_for_standard_names,
)

# ---------------------------------------------------------------------------
# Mock graph client
# ---------------------------------------------------------------------------


@dataclass
class _MockGraph:
    """In-memory stub matching ``GraphClient.query`` semantics.

    The cascade module issues four kinds of read queries plus one
    write.  We dispatch on Cypher-text substrings to map each call to
    pre-canned rows or to mutate the in-memory state for writes.
    """

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    # edges keyed by child_id → list of dicts with target_id + props
    edges_by_child: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    refined_from: dict[str, str] = field(default_factory=dict)
    write_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fail_transaction_on: str | None = None
    rollback_count: int = 0

    def add_node(
        self,
        nid: str,
        *,
        origin: str | None = None,
        name_stage: str | None = None,
        **fields: Any,
    ) -> None:
        self.nodes[nid] = {
            "origin": origin,
            "name_stage": name_stage,
            **fields,
        }

    def add_edge(
        self,
        child: str,
        target: str,
        operator: str,
        operator_kind: str,
        role: str | None = None,
        separator: str | None = None,
        axis: str | None = None,
        shape: str | None = None,
    ) -> None:
        self.edges_by_child.setdefault(child, []).append(
            {
                "target_id": target,
                "operator": operator,
                "operator_kind": operator_kind,
                "role": role,
                "separator": separator,
                "axis": axis,
                "shape": shape,
            }
        )

    def _ancestors(self, nid: str) -> set[str]:
        """All ancestors reachable via outbound HAS_PARENT from ``nid``."""
        seen: set[str] = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            for e in self.edges_by_child.get(cur, []):
                t = e["target_id"]
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return seen

    def _descendants(self, root: str) -> set[str]:
        """All descendants reachable via inbound HAS_PARENT to ``root``."""
        desc: set[str] = set()
        # Walk by reverse edges
        children_of: dict[str, list[str]] = {}
        for child, edges in self.edges_by_child.items():
            for e in edges:
                children_of.setdefault(e["target_id"], []).append(child)
        stack = [root]
        while stack:
            cur = stack.pop()
            for c in children_of.get(cur, []):
                if c not in desc:
                    desc.add(c)
                    stack.append(c)
        return desc

    def session(self) -> _MockSession:
        return _MockSession(self)

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:  # noqa: PLR0911 - dispatcher
        # ── Root existence + collision probe ──
        if "root_exists" in cypher:
            old = params.get("old")
            new = params.get("new")
            row = {
                "root_exists": old in self.nodes,
                "origin": self.nodes.get(old, {}).get("origin") if old else None,
                "name_stage": (
                    self.nodes.get(old, {}).get("name_stage") if old else None
                ),
                "target_exists": new in self.nodes,
            }
            return [row]

        # ── Descendant enumeration ──
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

        # ── Edge enumeration for subtree ──
        if "MATCH (child)-[r:HAS_PARENT]->(target)" in cypher:
            old = params.get("old")
            if not old:
                return []
            subtree = {old} | self._descendants(old)
            rows: list[dict[str, Any]] = []
            for child, edges in self.edges_by_child.items():
                for e in edges:
                    if e["target_id"] in subtree and child in subtree:
                        rows.append(
                            {
                                "child_id": child,
                                "target_id": e["target_id"],
                                "operator": e["operator"],
                                "operator_kind": e["operator_kind"],
                                "role": e["role"],
                                "separator": e["separator"],
                                "axis": e["axis"],
                                "shape": e["shape"],
                            }
                        )
            return rows

        # ── Collision recheck ──
        if "WHERE sn IS NOT NULL" in cypher and "UNWIND $ids" in cypher:
            ids = params.get("ids") or []
            return [{"id": nid} for nid in ids if nid in self.nodes]

        # ── Successor-exists probe (cascade_descendants_of) ──
        if "count(s) AS n" in cypher:
            nid = params.get("id")
            return [{"n": 1 if nid in self.nodes else 0}]

        # ── Guarded old-root fallback probe ──
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

        # ── Exact structural reconciliation existence guard ──
        if "RETURN requested AS id" in cypher and "AS exists" in cypher:
            return [
                {"id": name_id, "exists": name_id in self.nodes}
                for name_id in params.get("ids") or []
            ]

        # ── Canonical structural writer admission probe ──
        if "UNWIND $names AS nm" in cypher:
            return [
                {
                    "name": name_id,
                    "axes": [],
                    "child_ids": [],
                    "lone_child_id": None,
                    "lone_child_stage": None,
                    "lone_child_origin": None,
                    "parent_sources": [],
                    "lone_child_sources": [],
                    "origin": self.nodes.get(name_id, {}).get("origin"),
                    "name_stage": self.nodes.get(name_id, {}).get("name_stage"),
                }
                for name_id in params.get("names") or []
            ]

        # ── Canonical structural writer stale-edge reconciliation ──
        if "UNWIND $recon AS rc" in cypher and "DELETE r" in cypher:
            for row in params.get("recon") or []:
                child = row["child"]
                keep = set(row["keep"])
                self.edges_by_child[child] = [
                    edge
                    for edge in self.edges_by_child.get(child, [])
                    if not (
                        edge.get("operator_kind") is not None
                        and edge["target_id"] not in keep
                    )
                ]
            return []

        # ── Canonical structural writer HAS_PARENT materialization ──
        if "UNWIND $batch AS b" in cypher and "SET r.operator" in cypher:
            for row in params.get("batch") or []:
                child = row["from_name"]
                target = row["to_name"]
                self.nodes.setdefault(
                    target,
                    {"origin": "derived", "name_stage": "pending"},
                )
                edges = self.edges_by_child.setdefault(child, [])
                edge = next(
                    (
                        candidate
                        for candidate in edges
                        if candidate["target_id"] == target
                    ),
                    None,
                )
                props = {
                    "target_id": target,
                    "operator": row.get("operator"),
                    "operator_kind": row.get("operator_kind"),
                    "role": row.get("role"),
                    "separator": row.get("separator"),
                    "axis": row.get("axis"),
                    "shape": row.get("shape"),
                }
                if edge is None:
                    edges.append(props)
                else:
                    edge.update(props)
            return []

        # ── Rename write ──
        if "SET sn.id = r.to" in cypher:
            renames = params.get("renames") or []
            self.write_calls.append((cypher, params))
            # Apply renames in-place so collision detection of a
            # subsequent call sees the updated state.
            for r in renames:
                old_id = r["from"]
                new_id = r["to"]
                if old_id in self.nodes:
                    self.nodes[new_id] = self.nodes.pop(old_id)
                # Migrate edges keyed by child
                if old_id in self.edges_by_child:
                    self.edges_by_child[new_id] = self.edges_by_child.pop(old_id)
                # Migrate target references in remaining edges
                for edges in self.edges_by_child.values():
                    for e in edges:
                        if e["target_id"] == old_id:
                            e["target_id"] = new_id
            return []

        return []


class _MockTransaction:
    def __init__(self, owner: _MockGraph) -> None:
        self._owner = owner
        self._working = _MockGraph(
            nodes=deepcopy(owner.nodes),
            edges_by_child=deepcopy(owner.edges_by_child),
            refined_from=deepcopy(owner.refined_from),
            write_calls=deepcopy(owner.write_calls),
            fail_transaction_on=owner.fail_transaction_on,
            rollback_count=owner.rollback_count,
        )
        self._closed = False

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        marker = self._working.fail_transaction_on
        if marker and marker in cypher:
            raise RuntimeError(f"injected transaction failure at {marker}")
        return self._working.query(cypher, **params)

    def commit(self) -> None:
        self._owner.nodes = self._working.nodes
        self._owner.edges_by_child = self._working.edges_by_child
        self._owner.refined_from = self._working.refined_from
        self._owner.write_calls = self._working.write_calls
        self._closed = True

    def closed(self) -> bool:
        return self._closed

    def rollback(self) -> None:
        self._owner.rollback_count += 1
        self._closed = True

    def close(self) -> None:
        self._closed = True


class _MockSession:
    def __init__(self, graph: _MockGraph) -> None:
        self._transaction = _MockTransaction(graph)

    def __enter__(self) -> _MockSession:
        return self

    def __exit__(self, *exc: Any) -> bool:
        if not self._transaction.closed():
            self._transaction.rollback()
        return False

    def begin_transaction(self) -> _MockTransaction:
        return self._transaction


# ---------------------------------------------------------------------------
# Pure-logic tests (no graph, no ISN dependency on a specific name set)
# ---------------------------------------------------------------------------


class TestCascadeTargetNameDispatch:
    """Cascade rule table — exact per-kind formula."""

    def test_qualifier(self) -> None:
        edge = {"operator": "electron", "operator_kind": "qualifier"}
        assert _cascade_target_name(edge, "temperature") == "electron_temperature"

    def test_qualifier_with_compound_parent(self) -> None:
        edge = {"operator": "upper", "operator_kind": "qualifier"}
        assert (
            _cascade_target_name(edge, "elongation_of_closed_flux_surface")
            == "upper_elongation_of_closed_flux_surface"
        )

    def test_unary_prefix(self) -> None:
        edge = {"operator": "maximum", "operator_kind": "unary_prefix"}
        assert _cascade_target_name(edge, "temperature") == "maximum_of_temperature"

    def test_unary_postfix(self) -> None:
        edge = {"operator": "magnitude", "operator_kind": "unary_postfix"}
        assert (
            _cascade_target_name(edge, "magnetic_field") == "magnetic_field_magnitude"
        )

    def test_locus(self) -> None:
        edge = {"operator": "magnetic_axis", "operator_kind": "locus"}
        out = _cascade_target_name(
            edge,
            "major_radius",
            locus_relation="of",
            locus_token="magnetic_axis",
        )
        assert out == "major_radius_of_magnetic_axis"

    def test_locus_at_relation(self) -> None:
        edge = {"operator": "normalized_poloidal_flux", "operator_kind": "locus"}
        out = _cascade_target_name(
            edge,
            "safety_factor",
            locus_relation="at",
            locus_token="normalized_poloidal_flux",
        )
        assert out == "safety_factor_at_normalized_poloidal_flux"

    def test_locus_missing_relation_returns_none(self) -> None:
        edge = {"operator": "magnetic_axis", "operator_kind": "locus"}
        assert _cascade_target_name(edge, "x") is None

    def test_binary_role_a(self) -> None:
        edge = {
            "operator": "ratio",
            "operator_kind": "binary",
            "role": "a",
            "separator": "to",
        }
        out = _cascade_target_name(edge, "alpha", other_arg_name="beta")
        assert out == "ratio_of_alpha_to_beta"

    def test_binary_role_b(self) -> None:
        edge = {
            "operator": "ratio",
            "operator_kind": "binary",
            "role": "b",
            "separator": "to",
        }
        out = _cascade_target_name(edge, "beta", other_arg_name="alpha")
        assert out == "ratio_of_alpha_to_beta"

    def test_binary_without_other_arg_returns_none(self) -> None:
        edge = {
            "operator": "ratio",
            "operator_kind": "binary",
            "role": "a",
            "separator": "to",
        }
        assert _cascade_target_name(edge, "x") is None

    def test_projection_returns_none(self) -> None:
        edge = {
            "operator": "component",
            "operator_kind": "projection",
            "axis": "radial",
            "shape": "component",
        }
        assert _cascade_target_name(edge, "magnetic_field_v2") is None

    def test_coordinate_returns_none(self) -> None:
        edge = {
            "operator": "coordinate",
            "operator_kind": "coordinate",
            "axis": "radial",
        }
        assert _cascade_target_name(edge, "position_v2") is None

    def test_unknown_kind_returns_none(self) -> None:
        edge = {"operator": "x", "operator_kind": "no_such_kind"}
        assert _cascade_target_name(edge, "p") is None


# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------


class TestISNRoundTrip:
    def test_valid_name_passes(self) -> None:
        ok, _ = _isn_round_trip_ok("electron_temperature")
        assert ok is True

    def test_malformed_name_fails(self) -> None:
        ok, _ = _isn_round_trip_ok("123_invalid")
        assert ok is False

    def test_empty_string_fails(self) -> None:
        ok, _ = _isn_round_trip_ok("")
        assert ok is False


# ---------------------------------------------------------------------------
# rename_cascade — full integration via mock graph
# ---------------------------------------------------------------------------


class TestRenameCascadeBasic:
    def test_no_op_rename(self) -> None:
        gc = _MockGraph()
        gc.add_node("foo")
        result = rename_cascade(gc, "foo", "foo")
        assert result.conflicts
        assert "no-op" in result.conflicts[0] or "==" in result.conflicts[0]

    def test_unknown_root_aborts(self) -> None:
        gc = _MockGraph()
        # Use a name that DOES round-trip valid grammar so the cascade
        # reaches the existence check rather than failing earlier on
        # ISN validation.
        result = rename_cascade(gc, "temperature", "temperature_of_plasma_boundary")
        assert result.conflicts
        assert any("not found" in c for c in result.conflicts)

    def test_invalid_new_name_aborts(self) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        result = rename_cascade(gc, "temperature", "123_invalid")
        assert result.conflicts
        assert any("round-trip" in c for c in result.conflicts)

    def test_target_already_exists_aborts(self) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("electron_temperature")  # destination already in graph
        result = rename_cascade(gc, "temperature", "electron_temperature")
        assert result.conflicts
        assert any("already exists" in c for c in result.conflicts)

    def test_leaf_rename_no_descendants(self) -> None:
        """Rename a root with no descendants — applies cleanly."""
        gc = _MockGraph()
        gc.add_node("major_radius")
        result = rename_cascade(
            gc,
            "major_radius",
            "major_radius_of_magnetic_axis",
            dry_run=False,
        )
        assert result.conflicts == []
        assert result.total_descendants == 0
        assert len(result.renamed) == 1
        assert result.renamed[0] == {
            "from": "major_radius",
            "to": "major_radius_of_magnetic_axis",
        }
        # Write was applied
        assert any("SET sn.id = r.to" in c for c, _ in gc.write_calls)


# ---------------------------------------------------------------------------
# Projection non-cascade
# ---------------------------------------------------------------------------


class TestSemanticCohortProof:
    def test_exact_projection_transform_cascades(self) -> None:
        """A grammar-proven projection keeps its axis while its base follows."""
        gc = _MockGraph()
        gc.add_node("magnetic_field")
        gc.add_node("radial_magnetic_field")
        gc.add_edge(
            "radial_magnetic_field",
            "magnetic_field",
            operator="component",
            operator_kind="projection",
            axis="radial",
            shape="component",
        )

        result = rename_cascade(gc, "magnetic_field", "electric_field")

        assert result.conflicts == []
        assert {row["from"]: row["to"] for row in result.renamed} == {
            "magnetic_field": "electric_field",
            "radial_magnetic_field": "radial_electric_field",
        }

    def test_locus_edge_is_a_hard_boundary(self) -> None:
        gc = _MockGraph()
        gc.add_node("radial_coordinate")
        gc.add_node("radial_coordinate_of_control_surface")
        gc.add_edge(
            "radial_coordinate_of_control_surface",
            "radial_coordinate",
            operator="control_surface",
            operator_kind="locus",
        )

        result = rename_cascade(gc, "radial_coordinate", "radial_outline")

        assert result.conflicts == []
        assert result.renamed == [{"from": "radial_coordinate", "to": "radial_outline"}]
        assert result.skipped == [
            {
                "name": "radial_coordinate_of_control_surface",
                "reason": "operator_kind=locus is a semantic boundary",
            }
        ]

    def test_edge_metadata_must_match_grammar_structure(self) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("ion_temperature")
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="electron",
            operator_kind="qualifier",
        )

        result = rename_cascade(gc, "temperature", "density", dry_run=False)

        assert result.conflicts
        assert any("semantic proof" in conflict for conflict in result.conflicts)
        assert gc.write_calls == []
        assert set(gc.nodes) == {"temperature", "ion_temperature"}

    def test_projection_representation_mismatch_fails_closed(self) -> None:
        gc = _MockGraph()
        gc.add_node("magnetic_field")
        gc.add_node("radial_magnetic_field")
        gc.add_edge(
            "radial_magnetic_field",
            "magnetic_field",
            operator="component",
            operator_kind="projection",
            axis="vertical",
            shape="component",
        )

        result = rename_cascade(
            gc,
            "magnetic_field",
            "electric_field",
            dry_run=False,
        )

        assert result.conflicts
        assert any("semantic proof" in conflict for conflict in result.conflicts)
        assert gc.write_calls == []
        assert set(gc.nodes) == {"magnetic_field", "radial_magnetic_field"}

    def test_heterogeneous_operator_cohort_fails_before_write(self) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("ion_temperature")
        gc.add_node("maximum_of_temperature")
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )
        gc.add_edge(
            "maximum_of_temperature",
            "temperature",
            operator="maximum",
            operator_kind="unary_prefix",
        )

        result = rename_cascade(gc, "temperature", "density", dry_run=False)

        assert result.conflicts
        assert any("heterogeneous" in conflict for conflict in result.conflicts)
        assert gc.write_calls == []
        assert set(gc.nodes) == {
            "temperature",
            "ion_temperature",
            "maximum_of_temperature",
        }


# ---------------------------------------------------------------------------
# Qualifier cascade (the canonical use case)
# ---------------------------------------------------------------------------


class TestQualifierCascade:
    def test_qualifier_child_cascades(self) -> None:
        """``temperature`` → ``electron_temperature_of_core`` cascades
        ``ion_temperature`` (qualifier 'ion') → ``ion_temperature_of_core``."""
        # We rename a bare ``temperature`` to a valid locus form, and a
        # qualifier child ``ion_temperature`` must follow.  All names
        # parsed below are valid ISN grammar (ion + locus form).
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("ion_temperature")
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )

        new_root = "temperature_of_plasma_boundary"
        result = rename_cascade(gc, "temperature", new_root, dry_run=True)
        assert result.conflicts == []

        plan = {r["from"]: r["to"] for r in result.renamed}
        assert plan.get("temperature") == new_root
        assert plan.get("ion_temperature") == f"ion_{new_root}"


# ---------------------------------------------------------------------------
# Safety rejection
# ---------------------------------------------------------------------------


class TestSafetyChecks:
    def test_accepted_descendant_blocks_without_flag(self) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("ion_temperature", name_stage="accepted")
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )

        result = rename_cascade(
            gc,
            "temperature",
            "temperature_of_plasma_boundary",
        )
        assert result.conflicts
        assert any("name_stage='accepted'" in c for c in result.conflicts)

    def test_accepted_descendant_allows_with_flag(self) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("ion_temperature", name_stage="accepted")
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )

        result = rename_cascade(
            gc,
            "temperature",
            "temperature_of_plasma_boundary",
            include_accepted=True,
        )
        assert result.conflicts == []

    def test_catalog_edit_descendant_blocks_without_flag(self) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("ion_temperature", origin="catalog_edit")
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )

        result = rename_cascade(
            gc,
            "temperature",
            "temperature_of_plasma_boundary",
        )
        assert result.conflicts
        assert any("origin='catalog_edit'" in c for c in result.conflicts)

    def test_catalog_edit_descendant_allows_with_flag(self) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("ion_temperature", origin="catalog_edit")
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )

        result = rename_cascade(
            gc,
            "temperature",
            "temperature_of_plasma_boundary",
            override_edits=True,
        )
        assert result.conflicts == []


# ---------------------------------------------------------------------------
# ISN round-trip rejection inside cascade
# ---------------------------------------------------------------------------


class TestCascadeRoundTripRejection:
    def test_malformed_cascade_target_aborts(self) -> None:
        """If a cascade produces an invalid grammar token, abort.

        Force a child whose recovered name would not round-trip by
        using an invalid operator string.  We synthesise a child with
        an ``operator='???'`` qualifier that would compose to
        ``???_<new>`` — that won't parse.
        """
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("notaword_temperature")
        gc.add_edge(
            "notaword_temperature",
            "temperature",
            operator="notaword",  # not in ISN vocab → ISN parse rejects
            operator_kind="qualifier",
        )

        result = rename_cascade(
            gc,
            "temperature",
            "temperature_of_plasma_boundary",
        )
        # The dispatcher produces a candidate name; the round-trip
        # check fails because 'notaword' is not a valid qualifier token.
        # We assert the cascade aborts with at least one round-trip
        # conflict — the precise child name varies but the failure
        # surface does not.
        assert result.conflicts


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_dry_run_writes_audit_lines(self, tmp_path: Path) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")

        log_file = tmp_path / "parents_rename.log"
        result = rename_cascade(
            gc,
            "temperature",
            "temperature_of_plasma_boundary",
            dry_run=True,
            audit_log_path=log_file,
        )
        assert result.conflicts == []
        assert log_file.exists()
        content = log_file.read_text()
        # One line per rename, with the expected fields
        assert "mode=dry-run" in content
        assert "root=temperature->temperature_of_plasma_boundary" in content
        assert "from=temperature" in content
        assert "to=temperature_of_plasma_boundary" in content

    def test_commit_writes_audit_lines(self, tmp_path: Path) -> None:
        gc = _MockGraph()
        gc.add_node("temperature")
        gc.add_node("ion_temperature")
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )

        log_file = tmp_path / "parents_rename.log"
        result = rename_cascade(
            gc,
            "temperature",
            "temperature_of_plasma_boundary",
            dry_run=False,
            audit_log_path=log_file,
        )
        assert result.conflicts == []
        content = log_file.read_text()
        assert "mode=commit" in content
        # Two lines — root + qualifier child
        lines = [
            ln
            for ln in content.splitlines()
            if "root=temperature->temperature_of_plasma_boundary" in ln
        ]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Result dataclass shape
# ---------------------------------------------------------------------------


class TestCascadeResultShape:
    def test_result_is_dataclass(self) -> None:
        result = CascadeResult(old_name="a", new_name="b")
        # Verify the required attributes are present with sensible defaults.
        assert result.old_name == "a"
        assert result.new_name == "b"
        assert result.renamed == []
        assert result.skipped == []
        assert result.conflicts == []
        assert result.total_descendants == 0
        assert result.dry_run is True


# ---------------------------------------------------------------------------
# Misc — pytest invariants
# ---------------------------------------------------------------------------


class TestDescendantCascadeRecordsProvenance:
    """cascade_descendants_of (accept-time descendant cascade) must record a
    StandardNameChange and refresh source mirrors for every descendant it
    renames — parity with rename_cascade. Without it, accept-time cascades
    left renamed descendants with no change-history trail and stale source
    projections."""

    def test_records_change_and_refreshes_mirrors(self) -> None:
        from unittest.mock import patch

        gc = _MockGraph()
        # The root rename already landed: the successor id is live and a
        # qualifier child still carries the old-derived id.
        new_root = "temperature_of_plasma_boundary"
        gc.add_node(new_root)
        gc.add_node("ion_temperature")
        gc.add_edge(
            "ion_temperature",
            new_root,
            operator="ion",
            operator_kind="qualifier",
        )

        _PL = "imas_codex.standard_names.provenance_lifecycle"
        with (
            patch(f"{_PL}.record_standard_name_change") as rec,
            patch(f"{_PL}.refresh_renamed_source_mirrors") as refresh,
        ):
            result = cascade_descendants_of(
                gc,
                successor_id=new_root,
                old_root="temperature",
                new_root=new_root,
                dry_run=False,
            )

        assert result.conflicts == []
        assert {r["from"]: r["to"] for r in result.renamed} == {
            "ion_temperature": f"ion_{new_root}"
        }
        # One change event per renamed descendant, operation='cascade'.
        assert rec.call_count == 1
        _args, _kwargs = rec.call_args
        assert _args[1] == "ion_temperature"
        assert _args[2] == f"ion_{new_root}"
        assert _kwargs.get("operation") == "cascade"
        # Source mirrors refreshed once for the whole rename batch.
        refresh.assert_called_once()

    def test_dry_run_records_nothing(self) -> None:
        from unittest.mock import patch

        gc = _MockGraph()
        new_root = "temperature_of_plasma_boundary"
        gc.add_node(new_root)
        gc.add_node("ion_temperature")
        gc.add_edge(
            "ion_temperature", new_root, operator="ion", operator_kind="qualifier"
        )

        _PL = "imas_codex.standard_names.provenance_lifecycle"
        with (
            patch(f"{_PL}.record_standard_name_change") as rec,
            patch(f"{_PL}.refresh_renamed_source_mirrors") as refresh,
        ):
            cascade_descendants_of(
                gc,
                successor_id=new_root,
                old_root="temperature",
                new_root=new_root,
                dry_run=True,
            )
        rec.assert_not_called()
        refresh.assert_not_called()


class TestOldRootCascadeRecovery:
    """Accepted subtree edits recover descendants rebuilt below the predecessor."""

    old_root = "radiance_at_spectral_line"
    successor = "photon_radiance_at_spectral_line"
    old_child = "motional_stark_radiance_at_spectral_line"
    new_child = "motional_stark_photon_radiance_at_spectral_line"

    def _fallback_graph(self) -> _MockGraph:
        gc = _MockGraph()
        gc.add_node(
            self.successor,
            name_stage="accepted",
            edit_mode="rename",
            edit_status="applied",
            edit_scope="subtree",
        )
        gc.add_node(self.old_root, name_stage="superseded")
        gc.add_node(self.old_child, name_stage="accepted")
        gc.add_edge(
            self.old_child,
            self.old_root,
            operator="motional_stark",
            operator_kind="qualifier",
        )
        gc.refined_from[self.successor] = self.old_root
        return gc

    def test_falls_back_to_superseded_predecessor_subtree(self) -> None:
        gc = self._fallback_graph()

        result = cascade_descendants_of(
            gc,
            successor_id=self.successor,
            old_root=self.old_root,
            new_root=self.successor,
            dry_run=True,
            override_edits=True,
            include_accepted=True,
        )

        assert result.conflicts == []
        assert result.renamed == [{"from": self.old_child, "to": self.new_child}]
        assert result.total_descendants == 1
        assert all(rename["from"] != self.old_root for rename in result.renamed)

    @pytest.mark.parametrize(
        ("successor_updates", "old_stage", "has_lineage", "expected"),
        [
            ({"name_stage": "reviewed"}, "superseded", True, "not accepted"),
            ({"edit_mode": "hint"}, "superseded", True, "not a rename edit"),
            ({"edit_status": "open"}, "superseded", True, "not applied"),
            ({"edit_scope": "only_self"}, "superseded", True, "no cascade-bearing"),
            ({}, "superseded", False, "does not directly refine"),
            ({}, "accepted", True, "not superseded"),
        ],
    )
    def test_fallback_guard_failures(
        self,
        successor_updates: dict[str, Any],
        old_stage: str,
        has_lineage: bool,
        expected: str,
    ) -> None:
        gc = self._fallback_graph()
        gc.nodes[self.successor].update(successor_updates)
        gc.nodes[self.old_root]["name_stage"] = old_stage
        if not has_lineage:
            gc.refined_from.clear()

        result = cascade_descendants_of(
            gc,
            successor_id=self.successor,
            old_root=self.old_root,
            new_root=self.successor,
            dry_run=True,
            override_edits=True,
            include_accepted=True,
        )

        assert result.renamed == []
        assert any(expected in conflict for conflict in result.conflicts)

    def test_live_successor_subtree_takes_precedence(self) -> None:
        gc = self._fallback_graph()
        live_child = "ion_radiance_at_spectral_line"
        gc.add_node(live_child, name_stage="accepted")
        gc.add_edge(
            live_child,
            self.successor,
            operator="ion",
            operator_kind="qualifier",
        )

        result = cascade_descendants_of(
            gc,
            successor_id=self.successor,
            old_root=self.old_root,
            new_root=self.successor,
            dry_run=True,
            override_edits=True,
            include_accepted=True,
        )

        assert result.conflicts == []
        assert result.renamed == [
            {
                "from": live_child,
                "to": "ion_photon_radiance_at_spectral_line",
            }
        ]
        assert all(rename["from"] != self.old_child for rename in result.renamed)

    def test_staged_preacceptance_probe_validates_guarded_fallback(self) -> None:
        gc = self._fallback_graph()
        gc.nodes[self.successor].update(
            name_stage="drafted",
            edit_status="open",
        )

        result = cascade_descendants_of(
            gc,
            successor_id=self.successor,
            old_root=self.old_root,
            new_root=self.successor,
            dry_run=True,
            override_edits=True,
            include_accepted=True,
        )

        assert result.conflicts == []
        assert result.renamed == [{"from": self.old_child, "to": self.new_child}]
        assert result.total_descendants == 1

    def test_fallback_collision_fails_before_write(self) -> None:
        gc = self._fallback_graph()
        gc.add_node(self.new_child, name_stage="accepted")

        result = cascade_descendants_of(
            gc,
            successor_id=self.successor,
            old_root=self.old_root,
            new_root=self.successor,
            dry_run=False,
            override_edits=True,
            include_accepted=True,
        )

        assert any("collides" in conflict for conflict in result.conflicts)
        assert gc.write_calls == []
        assert self.old_child in gc.nodes

    def test_apply_excludes_root_and_reconciles_renamed_descendant(self) -> None:
        from unittest.mock import patch

        gc = self._fallback_graph()
        provenance_module = "imas_codex.standard_names.provenance_lifecycle"
        with (
            patch(f"{provenance_module}.record_standard_name_change"),
            patch(f"{provenance_module}.refresh_renamed_source_mirrors"),
        ):
            result = cascade_descendants_of(
                gc,
                successor_id=self.successor,
                old_root=self.old_root,
                new_root=self.successor,
                dry_run=False,
                override_edits=True,
                include_accepted=True,
            )

        assert result.conflicts == []
        assert result.renamed == [{"from": self.old_child, "to": self.new_child}]
        assert self.old_root in gc.nodes
        assert self.successor in gc.nodes
        assert self.new_child in gc.nodes
        assert gc.edges_by_child[self.new_child] == [
            {
                "target_id": self.successor,
                "operator": "motional_stark",
                "operator_kind": "qualifier",
                "role": None,
                "separator": None,
                "axis": None,
                "shape": None,
            }
        ]

    def test_reconciliation_failure_rolls_back_identity_and_topology(self) -> None:
        gc = self._fallback_graph()
        gc.fail_transaction_on = "UNWIND $recon AS rc"

        with pytest.raises(RuntimeError, match="injected transaction failure"):
            cascade_descendants_of(
                gc,
                successor_id=self.successor,
                old_root=self.old_root,
                new_root=self.successor,
                dry_run=False,
                override_edits=True,
                include_accepted=True,
            )

        assert self.old_child in gc.nodes
        assert self.new_child not in gc.nodes
        assert gc.edges_by_child[self.old_child][0]["target_id"] == self.old_root
        assert gc.rollback_count == 1

    def test_change_ledger_failure_rolls_back_complete_cascade(self) -> None:
        from unittest.mock import patch

        gc = self._fallback_graph()
        provenance_module = "imas_codex.standard_names.provenance_lifecycle"
        with (
            patch(
                f"{provenance_module}.record_standard_name_change",
                side_effect=RuntimeError("injected ledger failure"),
            ),
            pytest.raises(RuntimeError, match="injected ledger failure"),
        ):
            cascade_descendants_of(
                gc,
                successor_id=self.successor,
                old_root=self.old_root,
                new_root=self.successor,
                dry_run=False,
                override_edits=True,
                include_accepted=True,
            )

        assert self.old_child in gc.nodes
        assert self.new_child not in gc.nodes
        assert gc.edges_by_child[self.old_child][0]["target_id"] == self.old_root
        assert gc.rollback_count == 1

    def test_fallback_grammar_failure_fails_before_write(self) -> None:
        gc = self._fallback_graph()
        gc.edges_by_child[self.old_child][0]["operator"] = "not_a_grammar_token"

        result = cascade_descendants_of(
            gc,
            successor_id=self.successor,
            old_root=self.old_root,
            new_root=self.successor,
            dry_run=False,
            override_edits=True,
            include_accepted=True,
        )

        assert any("semantic proof" in conflict for conflict in result.conflicts)
        assert gc.write_calls == []


class TestExactStructuralReconciliation:
    def test_delegates_exact_existing_ids_to_canonical_writer(self) -> None:
        from unittest.mock import Mock, patch

        gc = Mock()
        gc.query.return_value = [
            {"id": "alpha", "exists": True},
            {"id": "beta", "exists": True},
        ]
        graph_ops_module = "imas_codex.standard_names.graph_ops"
        with patch(f"{graph_ops_module}._write_standard_name_edges") as writer:
            count = reconcile_structural_edges_for_standard_names(
                gc, ["alpha", "beta", "alpha"]
            )

        assert count == 2
        writer.assert_called_once_with(
            gc,
            [{"id": "alpha"}, {"id": "beta"}],
            expand_closure=False,
        )

    def test_missing_id_fails_before_structural_write(self) -> None:
        from unittest.mock import Mock, patch

        gc = Mock()
        gc.query.return_value = [{"id": "present", "exists": True}]
        graph_ops_module = "imas_codex.standard_names.graph_ops"
        with patch(f"{graph_ops_module}._write_standard_name_edges") as writer:
            with pytest.raises(ValueError, match="missing StandardName ids"):
                reconcile_structural_edges_for_standard_names(
                    gc, ["present", "missing"]
                )

        writer.assert_not_called()


@pytest.mark.parametrize(
    "op_kind",
    ["projection", "coordinate"],
)
def test_non_cascading_kinds(op_kind: str) -> None:
    """Both 'projection' and 'coordinate' are non-cascading by rule."""
    edge = {"operator": "x", "operator_kind": op_kind, "axis": "radial"}
    assert _cascade_target_name(edge, "new_parent") is None


# ---------------------------------------------------------------------------
# The plan the surface shows must match what the cascade actually persists
# ---------------------------------------------------------------------------


class TestDeepChainPlanConstruction:
    """A chain resolves through its whole depth, and a stopped ancestor
    reports its subtree as stopped rather than as a topology fault."""

    def test_linear_three_level_chain_plans_without_unreachable(self) -> None:
        """``temperature`` ← ``ion_temperature`` ← ``core_ion_temperature``.

        Nothing branches and every edge carries a provable qualifier, so all
        three levels belong in the plan and none of them is unreachable.
        """
        gc = _MockGraph()
        for name in ("temperature", "ion_temperature", "core_ion_temperature"):
            gc.add_node(name)
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="ion",
            operator_kind="qualifier",
        )
        gc.add_edge(
            "core_ion_temperature",
            "ion_temperature",
            operator="core",
            operator_kind="qualifier",
        )

        new_root = "temperature_of_plasma_boundary"
        result = rename_cascade(gc, "temperature", new_root, dry_run=True)

        assert result.conflicts == []
        assert not any(
            "unreachable in cascade" in conflict for conflict in result.conflicts
        )
        plan = {r["from"]: r["to"] for r in result.renamed}
        assert plan == {
            "temperature": new_root,
            "ion_temperature": f"ion_{new_root}",
            "core_ion_temperature": f"core_ion_{new_root}",
        }

    def test_subtree_below_a_boundary_is_stopped_not_unreachable(self) -> None:
        """A locus edge halts propagation; its own child is not a fault.

        The chain is strictly linear, so the grandchild is perfectly
        reachable — the cascade simply stops above it. Reporting it as
        ``unreachable`` made it a conflict, which refuses the whole edit.
        """
        gc = _MockGraph()
        for name in (
            "temperature",
            "temperature_of_plasma",
            "ion_temperature_of_plasma",
        ):
            gc.add_node(name)
        gc.add_edge(
            "temperature_of_plasma",
            "temperature",
            operator="of",
            operator_kind="locus",
        )
        gc.add_edge(
            "ion_temperature_of_plasma",
            "temperature_of_plasma",
            operator="ion",
            operator_kind="qualifier",
        )

        result = rename_cascade(gc, "temperature", "electron_temperature", dry_run=True)

        assert result.conflicts == []
        stopped = {row["name"]: row["reason"] for row in result.skipped}
        assert set(stopped) == {
            "temperature_of_plasma",
            "ion_temperature_of_plasma",
        }
        assert "semantic boundary" in stopped["temperature_of_plasma"]
        assert (
            "propagation stopped at ancestor 'temperature_of_plasma'"
            in stopped["ion_temperature_of_plasma"]
        )
        assert [r["from"] for r in result.renamed] == ["temperature"]

    def test_stopped_subtree_still_reports_the_ancestor_conflict(self) -> None:
        """A middle node whose proof fails keeps blocking the edit.

        Its descendant moves out of ``conflicts`` — one cause, one conflict —
        but the cause itself is untouched.
        """
        gc = _MockGraph()
        for name in ("temperature", "ion_temperature", "core_ion_temperature"):
            gc.add_node(name)
        gc.add_edge(
            "ion_temperature",
            "temperature",
            operator="electron",  # disagrees with the ISN derivation
            operator_kind="qualifier",
        )
        gc.add_edge(
            "core_ion_temperature",
            "ion_temperature",
            operator="core",
            operator_kind="qualifier",
        )

        result = rename_cascade(
            gc, "temperature", "temperature_of_plasma_boundary", dry_run=True
        )

        assert len(result.conflicts) == 1
        assert "semantic proof absent for 'ion_temperature'" in result.conflicts[0]
        assert not any(
            "unreachable in cascade" in conflict for conflict in result.conflicts
        )
        stopped = {row["name"]: row["reason"] for row in result.skipped}
        assert (
            "propagation stopped at ancestor 'ion_temperature'"
            in stopped["core_ion_temperature"]
        )


class TestDeferredCascadeIsReportedAsDeferred:
    """``sn edit`` prints the cascade before anything is written.

    ``EditPlan.cascade_deferred`` is applied only by the acceptance hook, so a
    root that is withheld or exhausted leaves every row in it unperformed.
    The surface has to say that rather than present the rows as done work.
    """

    @staticmethod
    def _rename_plan() -> Any:
        from imas_codex.standard_names.edit import EditPlan

        return EditPlan(
            target="emissivity_due_to_fusion",
            mode="rename",
            axis="name",
            scope="subtree",
            entry="review_name",
            successor="source_rate_due_to_fusion",
            cascade_deferred=[
                {
                    "from": "deuterium_deuterium_emissivity_due_to_fusion",
                    "to": "deuterium_deuterium_source_rate_due_to_fusion",
                },
                {
                    "from": "deuterium_tritium_emissivity_due_to_fusion",
                    "to": "deuterium_tritium_source_rate_due_to_fusion",
                },
            ],
            applied=True,
            run_id="sn-edit-20260903T000000Z",
        )

    def _render(self) -> str:
        from rich.console import Console

        from imas_codex.cli import sn as sn_cli

        recorder = Console(record=True, width=200)
        original = sn_cli.console
        sn_cli.console = recorder
        try:
            sn_cli._render_edit_plan(self._rename_plan(), followup_hint=False)
        finally:
            sn_cli.console = original
        return recorder.export_text()

    def test_render_names_the_deferral_and_what_it_waits_on(self) -> None:
        output = self._render()

        assert "deferred" in output
        assert "not yet applied" in output
        assert "awaiting source_rate_due_to_fusion reaching" in output
        assert "accepted" in output
        # The rows are still shown — the operator needs to see the consequence.
        assert "deuterium_tritium_emissivity_due_to_fusion" in output
        assert "deuterium_tritium_source_rate_due_to_fusion" in output

    def test_render_never_calls_the_deferred_rows_planned_work_done(self) -> None:
        output = self._render()

        assert "Cascade (planned renames)" not in output
        assert "renamed 2 descendant" not in output

    def test_applied_plan_actions_state_the_descendants_are_unchanged(self) -> None:
        from imas_codex.standard_names import edit as edit_module

        doc = edit_module.EditPlan.__doc__ or ""
        assert "deferred" in doc
        assert "reaches ``accepted``" in doc

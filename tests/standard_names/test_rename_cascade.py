"""Tests for the parent-rename cascade machinery.

The cascade is exercised in three layers:

1. **Pure rule dispatch** — ``_cascade_target_name`` produces the
   correct child name for each ``operator_kind`` per the D10 table.
2. **Mock-graph integration** — ``rename_cascade`` walks a stub
   ``query`` interface and produces correct plans, conflicts, and
   safety rejections without touching a live Neo4j.
3. **Audit log** — the rename writes a structured line per change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names.cascade import (
    CascadeResult,
    _cascade_target_name,
    _isn_round_trip_ok,
    rename_cascade,
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
    write_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def add_node(
        self,
        nid: str,
        *,
        origin: str | None = None,
        name_stage: str | None = None,
    ) -> None:
        self.nodes[nid] = {
            "origin": origin,
            "name_stage": name_stage,
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


class TestProjectionNonCascade:
    def test_projection_children_do_not_rename(self) -> None:
        """Renaming ``magnetic_field`` does NOT rename ``radial_magnetic_field``.

        Projection children carry independent identity — their axis
        prefix is the catalog entry, not the parent's name.
        """
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

        # Use a valid ISN target name.  ``magnetic_field`` itself is
        # already grammatical; any locus-suffixed form keeps it valid.
        result = rename_cascade(
            gc,
            "magnetic_field",
            "magnetic_field_of_plasma_boundary",
        )

        # Only the root is renamed; the projection child is skipped.
        assert result.conflicts == []
        renamed_from = [r["from"] for r in result.renamed]
        assert renamed_from == ["magnetic_field"]
        skipped_names = [s["name"] for s in result.skipped]
        assert "radial_magnetic_field" in skipped_names


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


@pytest.mark.parametrize(
    "op_kind",
    ["projection", "coordinate"],
)
def test_non_cascading_kinds(op_kind: str) -> None:
    """Both 'projection' and 'coordinate' are non-cascading by rule."""
    edge = {"operator": "x", "operator_kind": op_kind, "axis": "radial"}
    assert _cascade_target_name(edge, "new_parent") is None

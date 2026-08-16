"""Graph edge writer integration tests.

All tests are mocked — no live Neo4j required.  They verify that
``write_standard_names`` emits the correct Cypher queries with the
expected batch parameters for every structural edge type.

Edge types covered:
  HAS_PARENT      — derived from the ISN parser
  HAS_ERROR       — uncertainty siblings, inverted direction
  HAS_PREDECESSOR — from the ``deprecates`` field
  HAS_SUCCESSOR   — from the ``superseded_by`` field
  IN_CLUSTER      — from the ``primary_cluster_id`` field
  HAS_PHYSICS_DOMAIN — from the ``physics_domain`` field
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, call, patch

import pytest

imas_sn = pytest.importorskip("imas_standard_names")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_gc() -> MagicMock:
    """Return a fresh MagicMock for GraphClient."""
    gc = MagicMock()

    def _query(cypher: str, **params):
        if "STANDARD_NAME_EDGE_IDENTITY_DISCOVERY" in cypher:
            return [{"id": name_id} for name_id in params["ids"]]
        return []

    gc.query = MagicMock(side_effect=_query)
    return gc


def _call_write(names: list[dict], mock_gc: MagicMock) -> int:
    """Invoke ``write_standard_names`` with a mocked GraphClient."""
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        # Patch protection and segment edges to keep tests focused
        with (
            patch(
                "imas_codex.standard_names.protection.filter_protected",
                side_effect=lambda n, **kw: (n, []),
            ),
            patch(
                "imas_codex.standard_names.graph_ops._write_grammar_decomposition",
                return_value=[],
            ),
        ):
            from imas_codex.standard_names.graph_ops import write_standard_names

            return write_standard_names(names)


def _cyphers(mock_gc: MagicMock) -> list[str]:
    """Return all Cypher strings passed to gc.query."""
    return [c[0][0] for c in mock_gc.query.call_args_list]


def _batch_for(mock_gc: MagicMock, keyword: str) -> list[dict] | None:
    """Return the ``batch`` kwarg from the first WRITE query matching *keyword*.

    Skips topology-probe queries (which use ``names=`` rather than
    ``batch=``) so admission-gate machinery doesn't shadow the writes.
    """
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if keyword not in cypher:
            continue
        kw = c[1] if len(c) > 1 else {}
        if "batch" in kw:
            return kw["batch"]
        if len(c[0]) > 1 and isinstance(c[0][1], list):
            return c[0][1]
    return None


def _write_cyphers(mock_gc: MagicMock, keyword: str) -> list[str]:
    """Return all cyphers containing *keyword* that include a ``batch`` kwarg.

    Excludes admission-gate probe queries.
    """
    out: list[str] = []
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if keyword not in cypher:
            continue
        kw = c[1] if len(c) > 1 else {}
        if "batch" in kw or (len(c[0]) > 1 and isinstance(c[0][1], list)):
            out.append(cypher)
    return out


# ---------------------------------------------------------------------------
# HAS_PARENT: two names in one batch
# ---------------------------------------------------------------------------


class TestG1:
    """Write electron_temperature and maximum_of_electron_temperature in one batch.

    ``(maximum_of_electron_temperature)-[:HAS_PARENT {operator:'maximum'}]->(electron_temperature)``

    Uses the qualified ``electron_temperature`` (admissible) — bare
    ``temperature`` is rejected by both the ISN grammar validator and
    the admission gate.
    """

    PARENT = "electron_temperature"
    CHILD = "maximum_of_electron_temperature"

    def test_has_argument_edge_emitted(self) -> None:
        names = [
            {"id": self.PARENT, "unit": "eV"},
            {"id": self.CHILD, "unit": "eV"},
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        write_cyphers = _write_cyphers(mock_gc, "HAS_PARENT")
        assert write_cyphers, "No HAS_PARENT write cypher emitted"

    def test_has_argument_batch_contains_correct_edge(self) -> None:
        names = [
            {"id": self.PARENT, "unit": "eV"},
            {"id": self.CHILD, "unit": "eV"},
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_PARENT")
        assert batch is not None, "No HAS_PARENT batch found"
        edge = next(
            (
                b
                for b in batch
                if b["from_name"] == self.CHILD and b["to_name"] == self.PARENT
            ),
            None,
        )
        assert edge is not None, f"Expected edge not in batch: {batch}"
        assert edge["operator"] == "maximum"
        assert edge["operator_kind"] == "unary_prefix"


# ---------------------------------------------------------------------------
# HAS_PARENT: forward reference (target written later)
# ---------------------------------------------------------------------------


class TestG2:
    """Write child first, admissible parent in a later batch.

    After the second batch the edge must still be present (MERGE idempotent
    on re-run of the same name pair). Uses ``electron_temperature`` (an
    admissible qualifier-bearing parent) so the gate allows the edge.
    """

    PARENT = "electron_temperature"
    CHILD = "maximum_of_electron_temperature"

    def test_forward_ref_edge_present_after_second_batch(self) -> None:
        first_batch = [{"id": self.CHILD, "unit": "eV"}]
        second_batch = [{"id": self.PARENT, "unit": "eV"}]

        gc1 = _make_mock_gc()
        _call_write(first_batch, gc1)

        gc2 = _make_mock_gc()
        _call_write(second_batch, gc2)

        # First batch: HAS_PARENT should reference parent as placeholder
        batch1 = _batch_for(gc1, "HAS_PARENT")
        assert batch1 is not None
        assert any(b["to_name"] == self.PARENT for b in batch1)

        # Second batch: parent's own peel (qualifier → 'temperature') is
        # dropped by the admission gate, so no edge with parent as from_name.
        batch2 = _batch_for(gc2, "HAS_PARENT")
        if batch2:
            assert not any(b["from_name"] == self.PARENT for b in batch2), (
                "Bare-base parent should be dropped by the admission gate"
            )

    def test_target_is_matched_without_relationship_side_creation(self) -> None:
        """The HAS_PARENT writer may link but never create either endpoint.

        Uses an admissible parent (``electron_temperature``) so the
        admission gate keeps the edge. Bare-base parents like
        ``temperature`` are dropped by the gate and would emit no write.
        """
        names = [{"id": "maximum_of_electron_temperature", "unit": "eV"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        write_cyphers = _write_cyphers(mock_gc, "HAS_PARENT")
        assert write_cyphers, "No HAS_PARENT write cypher emitted"
        endpoint_merge = re.compile(r"MERGE\s*\([^)]*:StandardName")
        assert all(not endpoint_merge.search(c) for c in write_cyphers)
        assert any("MATCH (tgt:StandardName" in c for c in write_cyphers)


# ---------------------------------------------------------------------------
# HAS_ERROR: uncertainty sibling, inverted direction
# ---------------------------------------------------------------------------


class TestG3:
    """Write upper_uncertainty_of_temperature alone.

    ``(temperature)-[:HAS_ERROR {error_type:'upper'}]->(upper_uncertainty_of_temperature)``
    """

    def test_has_error_edge_emitted(self) -> None:
        names = [{"id": "upper_uncertainty_of_temperature", "unit": "eV"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("HAS_ERROR" in c for c in _cyphers(mock_gc))

    def test_has_error_direction_inverted(self) -> None:
        """from_name is temperature (inner), to_name is the uncertainty form."""
        names = [{"id": "upper_uncertainty_of_temperature", "unit": "eV"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_ERROR")
        assert batch is not None
        edge = next(
            (
                b
                for b in batch
                if b.get("to_name") == "upper_uncertainty_of_temperature"
            ),
            None,
        )
        assert edge is not None, f"HAS_ERROR edge not found in batch: {batch}"
        assert edge["from_name"] == "temperature"
        assert edge["error_type"] == "upper"


# ---------------------------------------------------------------------------
# HAS_PARENT: binary operator, two edges with role a/b
# ---------------------------------------------------------------------------


class TestG4:
    """Write ratio_of_electron_temperature_to_ion_temperature.

    Two HAS_PARENT edges with role a/b — uses qualifier-bearing parents
    so both pass the admission gate (bare ``temperature``/``pressure``
    would be dropped).
    """

    NAME = "ratio_of_electron_temperature_to_ion_temperature"

    def test_two_has_argument_edges(self) -> None:
        names = [{"id": self.NAME}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_PARENT")
        assert batch is not None

        edges = [b for b in batch if b["from_name"] == self.NAME]
        assert len(edges) == 2, f"Expected 2 edges, got {len(edges)}: {edges}"

        roles = {e["role"] for e in edges}
        assert roles == {"a", "b"}

    def test_binary_targets_correct(self) -> None:
        names = [{"id": self.NAME}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_PARENT")
        assert batch is not None

        edges = {e["role"]: e for e in batch if e["from_name"] == self.NAME}
        assert edges["a"]["to_name"] == "electron_temperature"
        assert edges["b"]["to_name"] == "ion_temperature"
        assert edges["a"]["operator"] == "ratio"
        assert edges["a"]["operator_kind"] == "binary"


# ---------------------------------------------------------------------------
# idempotency: writing the same batch twice
# ---------------------------------------------------------------------------


class TestG5:
    """Write same batch twice → edge Cypher is MERGE-based (idempotent)."""

    def test_has_argument_cypher_uses_merge(self) -> None:
        names = [{"id": "maximum_of_electron_temperature", "unit": "eV"}]

        for _ in range(2):
            mock_gc = _make_mock_gc()
            _call_write(names, mock_gc)

            write_cyphers = _write_cyphers(mock_gc, "HAS_PARENT")
            assert write_cyphers, "No HAS_PARENT write cypher emitted"
            assert any("MERGE" in c for c in write_cyphers), (
                "HAS_PARENT Cypher must use MERGE for idempotency"
            )

    def test_has_error_cypher_uses_merge(self) -> None:
        names = [{"id": "upper_uncertainty_of_temperature"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        for c in _cyphers(mock_gc):
            if "HAS_ERROR" in c:
                assert "MERGE" in c
                break


class TestEndpointAuthority:
    """Secondary edges consume authorized identities without minting names."""

    def test_secondary_edge_cypher_never_merges_standard_name_endpoints(self) -> None:
        mock_gc = _make_mock_gc()
        _call_write(
            [
                {
                    "id": "maximum_of_electron_temperature",
                    "unit": "eV",
                    "deprecates": "electron_temperature",
                    "superseded_by": "ion_temperature",
                },
                {"id": "upper_uncertainty_of_temperature", "unit": "eV"},
            ],
            mock_gc,
        )

        endpoint_merge = re.compile(r"MERGE\s*\([^)]*:StandardName")
        secondary = [
            cypher
            for cypher in _cyphers(mock_gc)
            if any(
                relation in cypher
                for relation in (
                    "HAS_PARENT",
                    "HAS_ERROR",
                    "HAS_LOCUS",
                    "HAS_PREDECESSOR",
                    "HAS_SUCCESSOR",
                )
            )
        ]
        assert secondary
        assert all(not endpoint_merge.search(cypher) for cypher in secondary)

    def test_missing_admitted_parent_uses_full_materializer_before_edge(self) -> None:
        from imas_codex.standard_names.graph_ops import _write_standard_name_edges

        graph = MagicMock()

        def _query(cypher: str, **params):
            if "STANDARD_NAME_EDGE_IDENTITY_DISCOVERY" in cypher:
                return [
                    {
                        "id": "maximum_of_electron_temperature",
                        "name_stage": "drafted",
                    }
                ]
            if "STANDARD_NAME_PARENT_BOOTSTRAP_CHILDREN" in cypher:
                return [
                    {
                        "parent_id": "electron_temperature",
                        "id": "maximum_of_electron_temperature",
                        "unit": "eV",
                        "cocos": None,
                        "physics_domain": "transport",
                        "kind": "scalar",
                        "op_kind": "unary_prefix",
                    }
                ]
            return []

        graph.query = MagicMock(side_effect=_query)
        _write_standard_name_edges(
            graph,
            [{"id": "maximum_of_electron_temperature"}],
        )

        bootstrap = next(
            call.args[0]
            for call in graph.query.call_args_list
            if "MERGE (parent:StandardName" in call.args[0]
        )
        assert "parent.docs_stage" in bootstrap
        assert "parent.validation_status" in bootstrap
        assert "MERGE (sns:StandardNameSource" in bootstrap
        assert "PRODUCED_NAME" in bootstrap
        assert "size(authorized_children) = size($bootstrap_edges)" in bootstrap
        assert "MERGE (parent)-[:HAS_UNIT]" in bootstrap
        assert "MERGE (child)-[relation:HAS_PARENT]" in bootstrap
        parent_write = next(
            call.args[0]
            for call in graph.query.call_args_list
            if "MERGE (src)-[r:HAS_PARENT]" in call.args[0]
        )
        assert "MATCH (src:StandardName" in parent_write
        assert "MATCH (tgt:StandardName" in parent_write
        assert "MERGE (src:StandardName" not in parent_write

    def test_rejected_intermediate_cannot_emit_locus_or_mint_identity(self) -> None:
        from imas_codex.standard_names.derivation import DerivedEdge
        from imas_codex.standard_names.graph_ops import _write_standard_name_edges

        graph = _make_mock_gc()
        edges = {
            "maximum_of_electron_temperature": [
                DerivedEdge(
                    edge_type="HAS_PARENT",
                    from_name="maximum_of_electron_temperature",
                    to_name="rejected_intermediate",
                    props={"operator_kind": "unary_prefix"},
                )
            ],
            "rejected_intermediate": [
                DerivedEdge(
                    edge_type="HAS_LOCUS",
                    from_name="rejected_intermediate",
                    to_name="ignored",
                    props={"locus_token": "magnetic_axis"},
                )
            ],
        }
        with (
            patch(
                "imas_codex.standard_names.derivation.derive_edges",
                side_effect=lambda name: edges.get(name, []),
            ),
            patch(
                "imas_codex.standard_names.graph_ops._filter_admissible_parents",
                return_value=[],
            ),
        ):
            _write_standard_name_edges(
                graph,
                [{"id": "maximum_of_electron_temperature"}],
            )

        locus_batch = _batch_for(graph, "HAS_LOCUS") or []
        assert all(row["from_name"] != "rejected_intermediate" for row in locus_batch)

    def test_failure_before_edge_persistence_leaves_no_partial_endpoint_write(
        self,
    ) -> None:
        from imas_codex.standard_names.graph_ops import _write_standard_name_edges

        graph = MagicMock()
        discovered = False

        def _query(cypher: str, **params):
            nonlocal discovered
            if "STANDARD_NAME_EDGE_IDENTITY_DISCOVERY" in cypher:
                discovered = True
                return [{"id": name_id} for name_id in params["ids"]]
            if discovered and "MERGE (src)-[r:HAS_PARENT]" in cypher:
                raise RuntimeError("injected edge persistence failure")
            return []

        graph.query = MagicMock(side_effect=_query)
        with pytest.raises(RuntimeError, match="injected edge persistence failure"):
            _write_standard_name_edges(
                graph,
                [{"id": "maximum_of_electron_temperature"}],
            )

        endpoint_merge = re.compile(r"MERGE\s*\([^)]*:StandardName")
        assert all(
            not endpoint_merge.search(call.args[0])
            for call in graph.query.call_args_list
        )

    def test_structural_rederive_excludes_null_lifecycle_roots(self) -> None:
        from imas_codex.standard_names import graph_ops

        graph = MagicMock()
        graph.__enter__.return_value = graph
        graph.__exit__.return_value = None
        graph.query.return_value = []

        with patch.object(graph_ops, "GraphClient", return_value=graph):
            assert graph_ops.rederive_structural_edges()["processed"] == 0

        root_query = graph.query.call_args_list[0].args[0]
        assert "sn.name_stage IS NOT NULL" in root_query


# ---------------------------------------------------------------------------
# HAS_PREDECESSOR from deprecates field
# ---------------------------------------------------------------------------


class TestG7:
    """Write StandardName with ``deprecates`` field → HAS_PREDECESSOR edge."""

    def test_has_predecessor_edge_pipeline(self) -> None:
        names = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "deprecates": "temperature_of_electrons",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("HAS_PREDECESSOR" in c for c in _cyphers(mock_gc))

    def test_has_predecessor_batch_pipeline(self) -> None:
        names = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "deprecates": "temperature_of_electrons",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_PREDECESSOR")
        assert batch is not None
        assert any(
            b["from_name"] == "electron_temperature"
            and b["to_name"] == "temperature_of_electrons"
            for b in batch
        )


# ---------------------------------------------------------------------------
# HAS_SUCCESSOR from superseded_by field
# ---------------------------------------------------------------------------


class TestG8:
    """Write StandardName with ``superseded_by`` field → HAS_SUCCESSOR edge."""

    def test_has_successor_edge_pipeline(self) -> None:
        names = [
            {
                "id": "ion_temperature",
                "unit": "eV",
                "superseded_by": "electron_temperature",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("HAS_SUCCESSOR" in c for c in _cyphers(mock_gc))

    def test_has_successor_batch_pipeline(self) -> None:
        names = [
            {
                "id": "ion_temperature",
                "unit": "eV",
                "superseded_by": "electron_temperature",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_SUCCESSOR")
        assert batch is not None
        assert any(
            b["from_name"] == "ion_temperature"
            and b["to_name"] == "electron_temperature"
            for b in batch
        )


# ---------------------------------------------------------------------------
# IN_CLUSTER from primary_cluster_id
# ---------------------------------------------------------------------------


class TestG9:
    """Write StandardName with ``primary_cluster_id`` → IN_CLUSTER edge."""

    def test_in_cluster_edge_emitted(self) -> None:
        names = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "primary_cluster_id": "cluster:electron_temperature_global",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("IN_CLUSTER" in c for c in _cyphers(mock_gc))

    def test_in_cluster_batch(self) -> None:
        names = [
            {
                "id": "electron_temperature",
                "unit": "eV",
                "primary_cluster_id": "cluster:electron_temperature_global",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "IN_CLUSTER")
        assert batch is not None
        assert any(
            b["sn_id"] == "electron_temperature"
            and b["cluster_id"] == "cluster:electron_temperature_global"
            for b in batch
        )

    def test_no_in_cluster_when_no_cluster_id(self) -> None:
        """No IN_CLUSTER Cypher emitted when primary_cluster_id is absent."""
        names = [{"id": "electron_temperature", "unit": "eV"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert not any("IN_CLUSTER" in c for c in _cyphers(mock_gc))


# ---------------------------------------------------------------------------
# HAS_PHYSICS_DOMAIN from physics_domain scalar
# ---------------------------------------------------------------------------


class TestG10:
    """Write StandardName with ``physics_domain='equilibrium'``
    → HAS_PHYSICS_DOMAIN edge to singleton PhysicsDomain {id:'equilibrium'}.
    """

    def test_has_physics_domain_edge_emitted(self) -> None:
        names = [
            {
                "id": "plasma_current",
                "unit": "A",
                "physics_domain": "equilibrium",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert any("HAS_PHYSICS_DOMAIN" in c for c in _cyphers(mock_gc))

    def test_has_physics_domain_batch(self) -> None:
        names = [
            {
                "id": "plasma_current",
                "unit": "A",
                "physics_domain": "equilibrium",
            }
        ]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        batch = _batch_for(mock_gc, "HAS_PHYSICS_DOMAIN")
        assert batch is not None
        assert any(
            b["sn_id"] == "plasma_current" and b["domain_id"] == "equilibrium"
            for b in batch
        )

    def test_no_has_physics_domain_when_absent(self) -> None:
        """No HAS_PHYSICS_DOMAIN Cypher when physics_domain is absent."""
        names = [{"id": "plasma_current", "unit": "A"}]
        mock_gc = _make_mock_gc()
        _call_write(names, mock_gc)

        assert not any("HAS_PHYSICS_DOMAIN" in c for c in _cyphers(mock_gc))

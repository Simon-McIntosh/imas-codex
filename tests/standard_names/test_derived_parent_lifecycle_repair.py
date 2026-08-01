"""Regression tests for derived-parent lifecycle normalization."""

from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
)


class _StatefulDerivedParentGraph:
    """Minimal in-memory graph stub for derived-parent repair tests."""

    def __init__(
        self,
        *,
        parent_id: str = "magnetic_field",
        origin: str | None = "derived",
        name_stage: str | None = "pending",
        docs_stage: str | None = None,
        description: str = "",
        chain_length: int | None = None,
        docs_chain_length: int | None = None,
        child_units: tuple[str | None, ...] = ("T",),
        child_domains: tuple[str | None, ...] = (None,),
        dd_paths: tuple[str, ...] = (),
        edge_kinds: tuple[str, ...] = ("projection",),
        children_complete: bool = True,
    ) -> None:
        self.parent: dict[str, object | None] = {
            "id": parent_id,
            "origin": origin,
            "name_stage": name_stage,
            "docs_stage": docs_stage,
            "description": description,
            "chain_length": chain_length,
            "docs_chain_length": docs_chain_length,
            "validation_status": None,
            "kind": None,
            "unit": None,
            "physics_domain": None,
            "claim_token": None,
            "documentation": None,
        }
        self.child_units = child_units
        self.child_domains = child_domains
        self.dd_paths = list(dd_paths)
        self.edge_kinds = list(edge_kinds)
        self.children_complete = children_complete
        self.sources: dict[str, dict[str, object | None]] = {}
        self.query_calls: list[tuple[str, dict]] = []
        self.tx_runs: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _eligible_for_seed_repair(self) -> bool:
        target_state = (
            self.parent["name_stage"] is None
            or self.parent["name_stage"] == "pending"
            or (
                self.parent["name_stage"] == "accepted"
                and self.parent["docs_stage"] is None
            )
        )
        return (
            self.parent["origin"] in (None, "derived")
            and target_state
            and self.children_complete
            and any(
                kind in {"projection", "coordinate", "unary_postfix"}
                for kind in self.edge_kinds
            )
        )

    def _eligible_for_legacy_repair(self) -> bool:
        target_state = (
            self.parent["name_stage"] is None
            or self.parent["name_stage"] == "pending"
            or (
                self.parent["name_stage"] == "accepted"
                and self.parent["docs_stage"] is None
            )
        )
        return (
            self.parent["origin"] in (None, "derived")
            and target_state
            and self.children_complete
            and bool(self.edge_kinds)
            and not any(
                kind in {"projection", "coordinate", "unary_postfix"}
                for kind in self.edge_kinds
            )
        )

    def _eligible_for_accepted_unit_repair(self, *, seedable: bool) -> bool:
        has_seedable = any(
            kind in {"projection", "coordinate", "unary_postfix"}
            for kind in self.edge_kinds
        )
        return (
            self.parent["origin"] == "derived"
            and self.parent["name_stage"] == "accepted"
            and self.parent["docs_stage"] == "accepted"
            and self.parent["unit"] is None
            and self.children_complete
            and (has_seedable if seedable else not has_seedable)
        )

    def _candidate_row(self) -> dict:
        child_data = []
        for idx, unit in enumerate(self.child_units):
            child_data.append(
                {
                    "id": f"child_{idx}",
                    "unit": unit,
                    "cocos": None,
                    "physics_domain": self.child_domains[idx],
                    "kind": "scalar",
                }
            )
        return {
            "parent_id": self.parent["id"],
            "child_data": child_data,
            "dd_paths": list(self.dd_paths),
            "edge_kinds": list(self.edge_kinds),
        }

    def query(self, cypher: str, **kwargs):
        self.query_calls.append((cypher, kwargs))
        if (
            "AND parent.docs_stage = 'accepted'" in cypher
            and "parent.unit IS NULL" in cypher
            and "RETURN parent.id AS parent_id" in cypher
        ):
            if "seedable_edges = 0" in cypher:
                return (
                    [self._candidate_row()]
                    if self._eligible_for_accepted_unit_repair(seedable=False)
                    else []
                )
            return (
                [self._candidate_row()]
                if self._eligible_for_accepted_unit_repair(seedable=True)
                else []
            )

        if (
            "parent.name_stage IN ['pending', 'accepted']" in cypher
            and "RETURN DISTINCT parent.id AS parent_id" in cypher
        ):
            # Admission cleanup is covered independently with the real query.
            return []

        if "seedable_edges = 0" in cypher and "RETURN parent.id AS parent_id" in cypher:
            return [self._candidate_row()] if self._eligible_for_legacy_repair() else []

        if "RETURN parent.id AS parent_id" in cypher:
            return [self._candidate_row()] if self._eligible_for_seed_repair() else []

        if "MERGE (sns:StandardNameSource {id: $source_node_id})" in cypher:
            assert (
                "WHERE EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(parent) }"
                in cypher
            )
            desc = str(self.parent.get("description") or "")
            if not desc.strip():
                self.parent["description"] = kwargs["description"]
            self.parent["name_stage"] = "accepted"
            self.parent["docs_stage"] = self.parent.get("docs_stage") or "pending"
            self.parent["origin"] = "derived"
            self.parent["validation_status"] = (
                self.parent.get("validation_status") or "valid"
            )
            self.parent["chain_length"] = self.parent.get("chain_length") or 0
            self.parent["docs_chain_length"] = self.parent.get("docs_chain_length") or 0
            self.parent["kind"] = kwargs["kind"]
            self.parent["unit"] = kwargs["unit"] or self.parent.get("unit")
            self.parent["physics_domain"] = kwargs["physics_domain"] or self.parent.get(
                "physics_domain"
            )
            self.sources[kwargs["source_node_id"]] = {
                "id": kwargs["source_node_id"],
                "source_type": kwargs["source_type"],
                "source_id": kwargs["source_id"],
                "batch_key": kwargs["batch_key"],
                "description": self.parent["description"],
            }
            return []

        if "MATCH (sns:StandardNameSource {id: $source_node_id})" in cypher:
            self.sources[kwargs["source_node_id"]]["dd_path"] = kwargs["dd_path"]
            return []

        if "MERGE (sn)-[:HAS_UNIT]->(u)" in cypher:
            assert (
                "WHERE EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(sn) }" in cypher
            )
            self.parent["unit"] = kwargs["unit"]
            return []

        if "RETURN count(DISTINCT r.axis) AS n" in cypher:
            return [{"n": 0}]

        # Post-claim race resolution (_verify_docs_claim_winners): echo back
        # ids that still hold our token at the eligible docs_stage.
        if (
            "WHERE sn.claim_token = $token" in cypher
            and "AND sn.docs_stage = $eligible_stage" in cypher
        ):
            ids = kwargs.get("ids", [])
            if (
                self.parent.get("claim_token") == kwargs.get("token")
                and self.parent.get("docs_stage") == kwargs.get("eligible_stage")
                and self.parent["id"] in ids
            ):
                return [{"id": self.parent["id"]}]
            return []

        # Childless-derived-parent reaper (normalize_derived_parent_lifecycle).
        # The single parent under test always carries live children in these
        # scenarios, so it is never a childless zombie — return nothing.
        if (
            "MATCH (p:StandardName {origin: 'derived'})" in cypher
            and "NOT EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(p) }" in cypher
            and "RETURN p.id AS id" in cypher
        ):
            return []

        # Orphaned-DocsRevision sweep (normalize_derived_parent_lifecycle).
        if "MATCH (dr:DocsRevision)" in cypher and "DETACH DELETE dr" in cypher:
            return [{"n": 0}]

        raise AssertionError(f"Unexpected query: {cypher}")

    @contextmanager
    def session(self):
        tx = _DerivedParentTx(self)
        session = SimpleNamespace(begin_transaction=MagicMock(return_value=tx))
        yield session


class _DerivedParentTx:
    def __init__(self, graph: _StatefulDerivedParentGraph) -> None:
        self.graph = graph
        self.closed = False

    def run(self, cypher: str, **kwargs):
        self.graph.tx_runs.append((cypher, kwargs))
        if "RETURN c.id AS _cluster_id" in cypher:
            if (
                self.graph.parent["name_stage"] == "accepted"
                and self.graph.parent["docs_stage"] == "pending"
                and self.graph.parent["claim_token"] is None
            ):
                self.graph.parent["claim_token"] = kwargs["token"]
                return iter(
                    [{"_cluster_id": None, "_unit": None, "_physics_domain": None}]
                )
            return iter([])

        if "MATCH (sn:StandardName {claim_token: $token})" in cypher:
            if self.graph.parent["claim_token"] != kwargs["token"]:
                return iter([])
            return iter(
                [
                    {
                        "id": self.graph.parent["id"],
                        "description": self.graph.parent["description"],
                        "documentation": self.graph.parent["documentation"],
                        "kind": self.graph.parent["kind"],
                        "unit": self.graph.parent["unit"],
                        "cluster_id": None,
                        "physics_domain": self.graph.parent["physics_domain"],
                        "validation_status": self.graph.parent["validation_status"],
                        "claim_token": self.graph.parent["claim_token"],
                        "reviewer_score_name": None,
                        "reviewer_comments_name": None,
                        "chain_length": self.graph.parent["chain_length"],
                        "docs_stage": self.graph.parent["docs_stage"],
                        "name_stage": self.graph.parent["name_stage"],
                    }
                ]
            )

        raise AssertionError(f"Unexpected tx.run call: {cypher}")

    def commit(self):
        return None

    def close(self):
        self.closed = True


def test_pending_placeholder_repair_does_not_project_common_dd_prefix() -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    graph = _StatefulDerivedParentGraph(
        name_stage="pending",
        docs_stage=None,
        description="",
        child_units=("T", "T"),
        child_domains=("magnetics", "magnetics"),
        dd_paths=(
            "equilibrium/time_slice/profiles_1d/b_field_tor",
            "equilibrium/time_slice/profiles_1d/b_field_pol",
        ),
    )

    repaired = normalize_derived_parent_lifecycle(graph)

    assert repaired == 1
    assert graph.parent["name_stage"] == "accepted"
    assert graph.parent["docs_stage"] == "pending"
    assert graph.parent["docs_chain_length"] == 0
    assert graph.parent["description"] == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
    assert graph.parent["kind"] == "vector"
    source = graph.sources["derived:magnetic_field"]
    assert source["source_type"] == "derived"
    assert source["source_id"] == "magnetic_field"
    assert "dd_path" not in source
    assert all(
        "MERGE (sns)-[:FROM_DD_PATH]" not in cypher
        for cypher, _params in graph.query_calls
    )


def test_single_child_parent_materialization_preserves_real_dd_source() -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    leaf = "spectrometer_visible/channel/processed_line/radiance"
    child = "photon_radiance_at_spectral_line"
    parent = "photon_radiance"
    graph = _StatefulDerivedParentGraph(
        parent_id=parent,
        child_units=("m^-2.s^-1.sr^-1",),
        child_domains=("spectroscopy",),
        dd_paths=(leaf,),
    )
    graph.sources[f"dd:{leaf}"] = {
        "id": f"dd:{leaf}",
        "source_type": "dd",
        "source_id": leaf,
        "produced_sn_id": child,
    }

    assert normalize_derived_parent_lifecycle(graph) == 1
    assert normalize_derived_parent_lifecycle(graph) == 0

    assert graph.sources[f"dd:{leaf}"] == {
        "id": f"dd:{leaf}",
        "source_type": "dd",
        "source_id": leaf,
        "produced_sn_id": child,
    }
    assert graph.sources[f"derived:{parent}"] == {
        "id": f"derived:{parent}",
        "source_type": "derived",
        "source_id": parent,
        "batch_key": "derived_parent",
        "description": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    }
    assert all(
        "MERGE (sns)-[:FROM_DD_PATH]" not in cypher
        for cypher, _params in graph.query_calls
    )


def test_legacy_accepted_null_docs_repair_uses_derived_source_when_needed() -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    graph = _StatefulDerivedParentGraph(
        parent_id="total_plasma_current",
        name_stage="accepted",
        docs_stage=None,
        description="   ",
        child_units=("A",),
        child_domains=(None,),
    )

    repaired = normalize_derived_parent_lifecycle(graph)

    assert repaired == 1
    assert graph.parent["docs_stage"] == "pending"
    assert graph.parent["description"] == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
    assert graph.sources["derived:total_plasma_current"]["source_type"] == "derived"
    assert graph.sources["derived:total_plasma_current"]["source_id"] == (
        "total_plasma_current"
    )


@pytest.mark.parametrize(
    ("parent_id", "edge_kinds"),
    [
        ("electron_density", ("qualifier",)),
        ("outer_electron_density", ("locus",)),
        ("outer_electron_density", ("qualifier", "locus")),
    ],
)
def test_legacy_qualifier_locus_parents_repair_scalar_safely(
    parent_id: str,
    edge_kinds: tuple[str, ...],
) -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    graph = _StatefulDerivedParentGraph(
        parent_id=parent_id,
        name_stage="pending",
        docs_stage=None,
        description="",
        child_units=("m^-3", "m^-3"),
        child_domains=("core_profiles", "core_profiles"),
        edge_kinds=edge_kinds,
    )

    repaired = normalize_derived_parent_lifecycle(graph)

    assert repaired == 1
    assert graph.parent["name_stage"] == "accepted"
    assert graph.parent["docs_stage"] == "pending"
    assert graph.parent["kind"] == "scalar"
    assert graph.parent["description"] == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER


def test_legacy_heterogeneous_unit_parent_is_skipped() -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    graph = _StatefulDerivedParentGraph(
        parent_id="inductance",
        name_stage="pending",
        docs_stage=None,
        description="",
        child_units=("1", "H"),
        child_domains=("pf_active", "pf_active"),
        edge_kinds=("qualifier",),
    )

    repaired = normalize_derived_parent_lifecycle(graph)

    assert repaired == 0
    assert graph.parent["name_stage"] == "pending"
    assert graph.parent["docs_stage"] is None
    assert graph.sources == {}


def test_identity_rejection_reaps_only_the_invalid_pending_parent() -> None:
    """The identity oracle reaps invalid scaffolds but preserves unresolved ones."""
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    def candidate(parent_id: str, *units: str | None) -> dict:
        return {
            "parent_id": parent_id,
            "origin": "derived",
            "name_stage": "pending",
            "child_data": [
                {
                    "id": f"{parent_id}_child_{index}",
                    "unit": unit,
                    "cocos": None,
                    "physics_domain": "core_transport",
                    "kind": "scalar",
                    "op_kind": "qualifier",
                }
                for index, unit in enumerate(units)
            ],
            "dd_paths": [],
            "edge_kinds": ["qualifier"],
        }

    invalid = candidate("internal_state_momentum_source", "kg.m^-1.s^-2")
    valid = candidate("electron_density", "m^-3")
    no_unit = candidate("flux_at_wall", None)
    heterogeneous = candidate("inductance", "1", "H")
    gc = MagicMock()
    gc.query.return_value = []

    def validate(parent_id: str, **_kwargs) -> list[str]:
        if parent_id == invalid["parent_id"]:
            return ["state requires a species subject"]
        return []

    def delete(_gc, parent_ids: list[str]) -> int:
        return len(parent_ids)

    with (
        patch(
            "imas_codex.standard_names.graph_ops._query_seedable_derived_parents",
            side_effect=[[], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_legacy_repairable_derived_parents",
            side_effect=[[invalid, valid, no_unit, heterogeneous], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_derived_parents_for_admission_cleanup",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.parents.recompute_parent_kind",
            return_value="scalar",
        ),
        patch(
            "imas_codex.standard_names.graph_ops._validate_derived_parent_identity",
            side_effect=validate,
        ) as identity_oracle,
        patch(
            "imas_codex.standard_names.graph_ops._delete_derived_parent_nodes",
            side_effect=delete,
        ) as delete_nodes,
    ):
        changed = normalize_derived_parent_lifecycle(gc)

    assert changed == 2
    assert delete_nodes.call_args_list == [
        call(gc, []),
        call(gc, [invalid["parent_id"]]),
    ]
    assert [item.args[0] for item in identity_oracle.call_args_list] == [
        invalid["parent_id"],
        valid["parent_id"],
    ]
    materialized_ids = [
        item.kwargs["parent_id"]
        for item in gc.query.call_args_list
        if "SET parent.name_stage" in item.args[0]
    ]
    assert materialized_ids == [valid["parent_id"]]


def test_pending_single_child_shadow_is_retired_with_deletion_ledger() -> None:
    """A source-less shadow scaffold is retired instead of being materialized."""
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle
    from imas_codex.standard_names.parents import AdmissionResult

    parent_id = "line_integrated_impurity_ion_velocity"
    candidate = {
        "parent_id": parent_id,
        "origin": "derived",
        "name_stage": "pending",
        "child_data": [
            {
                "id": "toroidal_line_integrated_impurity_ion_velocity",
                "unit": "m.s^-1",
                "cocos": "one_like",
                "physics_domain": "radiation_measurement_diagnostics",
                "kind": "scalar",
                "op_kind": "projection",
            }
        ],
        "dd_paths": [
            "spectrometer_x_ray_crystal/channel/profiles_line_integrated/velocity_tor"
        ],
        "edge_kinds": ["projection"],
    }
    gc = MagicMock()

    def query(cypher: str, **_kwargs):
        if "DETACH DELETE sn" in cypher:
            return [{"deleted": 1}]
        if "MATCH (p:StandardName {origin: 'derived'})" in cypher:
            return []
        if "MATCH (dr:DocsRevision)" in cypher:
            return [{"n": 0}]
        raise AssertionError(f"unexpected query: {cypher}")

    gc.query.side_effect = query
    with (
        patch(
            "imas_codex.standard_names.graph_ops._query_seedable_derived_parents",
            side_effect=[[candidate], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_legacy_repairable_derived_parents",
            side_effect=[[], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_derived_parents_for_admission_cleanup",
            return_value=[parent_id],
        ),
        patch(
            "imas_codex.standard_names.parents.is_admissible_parent_name",
            return_value=AdmissionResult(
                admit=False,
                reason="suppressed: single-child shadow",
                clause=None,
            ),
        ),
    ):
        changed = normalize_derived_parent_lifecycle(gc)

    assert changed == 1
    delete_call = next(
        item for item in gc.query.call_args_list if "DETACH DELETE sn" in item.args[0]
    )
    assert "CREATE (change:StandardNameChange" in delete_call.args[0]
    assert delete_call.kwargs["deletion_operation"] == "remove_derived_parent"
    assert not any(
        "SET parent.name_stage" in item.args[0] for item in gc.query.call_args_list
    )


def test_derived_parent_lifecycle_repair_is_idempotent() -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    graph = _StatefulDerivedParentGraph(
        name_stage="pending",
        docs_stage=None,
        description="",
    )

    assert normalize_derived_parent_lifecycle(graph) == 1
    assert normalize_derived_parent_lifecycle(graph) == 0
    assert len(graph.sources) == 1


def test_repaired_parent_is_claimable_for_generate_docs() -> None:
    from imas_codex.standard_names.graph_ops import (
        claim_generate_docs_batch,
        normalize_derived_parent_lifecycle,
    )

    graph = _StatefulDerivedParentGraph(
        name_stage="pending",
        docs_stage=None,
        description="",
    )
    assert normalize_derived_parent_lifecycle(graph) == 1

    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph),
        patch(
            "imas_codex.standard_names.chain_history.name_chain_history",
            return_value=[],
        ),
    ):
        items = claim_generate_docs_batch(batch_size=1)

    assert len(items) == 1
    assert items[0]["id"] == "magnetic_field"
    assert items[0]["name_stage"] == "accepted"
    assert items[0]["docs_stage"] == "pending"


@pytest.mark.parametrize(
    ("origin", "name_stage", "docs_stage"),
    [
        ("pipeline", "pending", None),
        ("derived", "accepted", "pending"),
    ],
)
def test_nonderived_or_already_normalized_nodes_are_untouched(
    origin: str,
    name_stage: str,
    docs_stage: str | None,
) -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    graph = _StatefulDerivedParentGraph(
        origin=origin,
        name_stage=name_stage,
        docs_stage=docs_stage,
        description="Existing description",
    )

    assert normalize_derived_parent_lifecycle(graph) == 0
    assert graph.parent["description"] == "Existing description"
    assert graph.sources == {}


def test_inadmissible_accepted_derived_parent_is_deleted() -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle
    from imas_codex.standard_names.parents import AdmissionResult

    gc = MagicMock()
    # The childless-zombie reaper SELECT must find nothing here (the deletion
    # under test is the inadmissible-accepted cleanup, not the reaper).
    gc.query.return_value = []

    with (
        patch(
            "imas_codex.standard_names.graph_ops._query_seedable_derived_parents",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_legacy_repairable_derived_parents",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_derived_parents_for_admission_cleanup",
            return_value=["pressure"],
        ),
        patch(
            "imas_codex.standard_names.parents.is_admissible_parent_name",
            return_value=AdmissionResult(
                admit=False,
                reason="bare base — no qualifier, locus, projection, operator, or mechanism",
                clause=None,
            ),
        ),
        patch(
            "imas_codex.standard_names.graph_ops._delete_derived_parent_nodes",
            return_value=1,
        ) as delete_nodes,
    ):
        repaired = normalize_derived_parent_lifecycle(gc)

    assert repaired == 1
    delete_nodes.assert_called_once_with(gc, ["pressure"])


def test_childless_derived_placeholder_is_retired() -> None:
    """A derived placeholder without children is not a durable abstraction."""
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    gc = MagicMock()

    def query(cypher: str, **_kwargs):
        if "NOT EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(p) }" in cypher:
            return [{"id": "impurity_ion_velocity", "unit": None}]
        if "MATCH (dr:DocsRevision)" in cypher:
            return [{"n": 0}]
        raise AssertionError(f"unexpected query: {cypher}")

    gc.query.side_effect = query
    with (
        patch(
            "imas_codex.standard_names.graph_ops._query_seedable_derived_parents",
            side_effect=[[], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_legacy_repairable_derived_parents",
            side_effect=[[], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_derived_parents_for_admission_cleanup",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._delete_derived_parent_nodes",
            side_effect=[0, 1],
        ) as delete_nodes,
    ):
        changed = normalize_derived_parent_lifecycle(gc)

    assert changed == 1
    assert delete_nodes.call_args_list == [
        call(gc, []),
        call(gc, ["impurity_ion_velocity"]),
    ]


@pytest.mark.parametrize(
    "snapshot",
    [
        "seedable",
        "legacy",
        "accepted_seedable_unit_gap",
        "accepted_legacy_unit_gap",
    ],
)
def test_reaped_snapshot_candidate_is_not_rematerialized(snapshot: str) -> None:
    """A childless parent cannot return through an earlier candidate snapshot."""
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    parent_id = "momentum_flux_limiter_coefficient_over_edge_region"
    candidate = {
        "parent_id": parent_id,
        "origin": "derived",
        "name_stage": "pending",
        "child_data": [
            {
                "id": "detached_child",
                "unit": None,
                "cocos": None,
                "physics_domain": "core_transport",
                "kind": "scalar",
                "op_kind": "qualifier",
            }
        ],
        "dd_paths": [],
        "edge_kinds": ["qualifier"],
    }
    seedable_results = [
        [candidate] if snapshot == "seedable" else [],
        [candidate] if snapshot == "accepted_seedable_unit_gap" else [],
    ]
    legacy_results = [
        [candidate] if snapshot == "legacy" else [],
        [candidate] if snapshot == "accepted_legacy_unit_gap" else [],
    ]
    gc = MagicMock()

    def query(cypher: str, **_kwargs):
        if "NOT EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(p) }" in cypher:
            return [{"id": parent_id}]
        if "MATCH (dr:DocsRevision)" in cypher:
            return [{"n": 0}]
        raise AssertionError(f"unexpected query: {cypher}")

    gc.query.side_effect = query
    with (
        patch(
            "imas_codex.standard_names.graph_ops._query_seedable_derived_parents",
            side_effect=seedable_results,
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_legacy_repairable_derived_parents",
            side_effect=legacy_results,
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_derived_parents_for_admission_cleanup",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._delete_derived_parent_nodes",
            side_effect=[0, 1],
        ) as delete_nodes,
        patch(
            "imas_codex.standard_names.graph_ops._materialize_derived_parent_rows",
            side_effect=AssertionError("reaped parent was rematerialized"),
        ) as materialize,
    ):
        changed = normalize_derived_parent_lifecycle(gc)

    assert changed == 1
    assert delete_nodes.call_args_list == [
        call(gc, []),
        call(gc, [parent_id]),
    ]
    materialize.assert_not_called()


def test_childless_parent_chain_is_reaped_to_fixpoint() -> None:
    """Deleting one shell must expose and reap every childless ancestor."""
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    parent_ids = [
        "momentum_flux_limiter_coefficient_over_edge_region",
        "flux_limiter_coefficient_over_edge_region",
        "limiter_coefficient_over_edge_region",
    ]
    candidates = [
        {
            "parent_id": parent_id,
            "origin": "derived",
            "name_stage": "pending",
            "child_data": [
                {
                    "id": f"detached_child_{index}",
                    "unit": None,
                    "cocos": None,
                    "physics_domain": "core_transport",
                    "kind": "scalar",
                    "op_kind": "qualifier",
                }
            ],
            "dd_paths": [],
            "edge_kinds": ["qualifier"],
        }
        for index, parent_id in enumerate(parent_ids)
    ]
    childless_batches = [[parent_id] for parent_id in parent_ids] + [[]]
    gc = MagicMock()

    def query(cypher: str, **_kwargs):
        if "NOT EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(p) }" in cypher:
            return [{"id": parent_id} for parent_id in childless_batches.pop(0)]
        if "MATCH (dr:DocsRevision)" in cypher:
            return [{"n": 0}]
        raise AssertionError(f"unexpected query: {cypher}")

    gc.query.side_effect = query
    with (
        patch(
            "imas_codex.standard_names.graph_ops._query_seedable_derived_parents",
            side_effect=[candidates, []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_legacy_repairable_derived_parents",
            side_effect=[[], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_derived_parents_for_admission_cleanup",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._delete_derived_parent_nodes",
            side_effect=lambda _gc, ids: len(ids),
        ) as delete_nodes,
        patch(
            "imas_codex.standard_names.graph_ops._materialize_derived_parent_rows",
            side_effect=AssertionError("childless ancestor was rematerialized"),
        ) as materialize,
    ):
        changed = normalize_derived_parent_lifecycle(gc)

    assert changed == 3
    assert delete_nodes.call_args_list == [
        call(gc, []),
        call(gc, [parent_ids[0]]),
        call(gc, [parent_ids[1]]),
        call(gc, [parent_ids[2]]),
    ]
    assert childless_batches == []
    materialize.assert_not_called()


def test_semantically_valid_derived_parent_survives_cleanup() -> None:
    """A species-qualified state parent remains structurally admissible."""
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    parent_id = "neutral_internal_state_momentum_source"
    gc = MagicMock()

    def query(cypher: str, **_kwargs):
        if "RETURN child.id AS child_id" in cypher:
            return []
        if "NOT EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(p) }" in cypher:
            return []
        if "MATCH (dr:DocsRevision)" in cypher:
            return [{"n": 0}]
        raise AssertionError(f"unexpected query: {cypher}")

    gc.query.side_effect = query
    with (
        patch(
            "imas_codex.standard_names.graph_ops._query_seedable_derived_parents",
            side_effect=[[], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_legacy_repairable_derived_parents",
            side_effect=[[], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_derived_parents_for_admission_cleanup",
            return_value=[parent_id],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._delete_derived_parent_nodes",
            return_value=0,
        ) as delete_nodes,
    ):
        changed = normalize_derived_parent_lifecycle(gc)

    assert changed == 0
    delete_nodes.assert_called_once_with(gc, [])


def test_cleanup_query_rechecks_pending_and_accepted_parents() -> None:
    """Admission cleanup covers both creation leaks and accepted residue."""
    from imas_codex.standard_names.graph_ops import (
        _query_derived_parents_for_admission_cleanup,
    )

    captured: dict[str, str] = {}

    class _Probe:
        def query(self, cypher: str, **kwargs):
            captured["cypher"] = cypher
            return [{"parent_id": "radius_of_magnetic_axis"}]

    result = _query_derived_parents_for_admission_cleanup(_Probe())
    assert result == ["radius_of_magnetic_axis"]
    assert "parent.name_stage IN ['pending', 'accepted']" in captured["cypher"]
    assert "parent.docs_stage" not in captured["cypher"]


def test_accepted_docs_complete_parent_with_missing_unit_is_repaired() -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    gc = MagicMock()
    candidate = {
        "parent_id": "fraction_of_flux_surface",
        "child_data": [
            {
                "id": "child_0",
                "unit": "1",
                "cocos": None,
                "physics_domain": "equilibrium",
                "kind": "scalar",
            }
        ],
        "dd_paths": [],
        "edge_kinds": ["qualifier"],
    }

    with (
        patch(
            "imas_codex.standard_names.graph_ops._query_seedable_derived_parents",
            side_effect=[[], []],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_legacy_repairable_derived_parents",
            side_effect=[[], [candidate]],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._query_derived_parents_for_admission_cleanup",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.graph_ops._materialize_derived_parent_rows",
            side_effect=[0, 0, 0, 1],
        ) as materialize,
    ):
        repaired = normalize_derived_parent_lifecycle(gc)

    assert repaired == 1
    assert materialize.call_args_list[-1].args[1] == [candidate]
    assert (
        materialize.call_args_list[-1].kwargs["infer_kind_from_existing_topology"]
        is True
    )


def test_run_sn_pools_normalizes_after_parent_seeding() -> None:
    from imas_codex.standard_names import loop

    tree = ast.parse(inspect.getsource(loop.run_sn_pools))
    maintenance_calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_global_maintenance_call"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    calls_by_name = {node.args[0].id: node for node in maintenance_calls}
    call_names = [node.args[0].id for node in maintenance_calls]

    assert "seed_parent_sources" in call_names
    assert "normalize_derived_parent_lifecycle" in call_names
    assert call_names.index("seed_parent_sources") < call_names.index(
        "normalize_derived_parent_lifecycle"
    )
    assert len(calls_by_name["normalize_derived_parent_lifecycle"].args) == 1

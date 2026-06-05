"""Regression tests for derived-parent lifecycle normalization."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        child_units: tuple[str | None, ...] = (None,),
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

    def _eligible_for_repair(self) -> bool:
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
        if "RETURN parent.id AS parent_id" in cypher:
            return [self._candidate_row()] if self._eligible_for_repair() else []

        if "MERGE (sns:StandardNameSource {id: $source_node_id})" in cypher:
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
            self.parent["unit"] = kwargs["unit"]
            return []

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


def test_pending_placeholder_repair_sets_docs_lifecycle_and_dd_source() -> None:
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
    assert graph.sources["dd:equilibrium/time_slice/profiles_1d"]["source_type"] == "dd"
    assert graph.sources["dd:equilibrium/time_slice/profiles_1d"]["source_id"] == (
        "equilibrium/time_slice/profiles_1d"
    )


def test_legacy_accepted_null_docs_repair_uses_manual_source_when_needed() -> None:
    from imas_codex.standard_names.graph_ops import normalize_derived_parent_lifecycle

    graph = _StatefulDerivedParentGraph(
        parent_id="total_plasma_current",
        name_stage="accepted",
        docs_stage=None,
        description="   ",
        child_units=(None,),
        child_domains=(None,),
    )

    repaired = normalize_derived_parent_lifecycle(graph)

    assert repaired == 1
    assert graph.parent["docs_stage"] == "pending"
    assert graph.parent["description"] == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
    assert graph.sources["manual:total_plasma_current"]["source_type"] == "manual"
    assert graph.sources["manual:total_plasma_current"]["source_id"] == (
        "total_plasma_current"
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


def test_run_sn_pools_normalizes_after_parent_seeding() -> None:
    from imas_codex.standard_names import loop

    src = inspect.getsource(loop.run_sn_pools)
    assert "repaired_parent_count = await asyncio.to_thread(" in src
    assert src.index("parent_count = await asyncio.to_thread(seed_parent_sources)") < src.index(
        "repaired_parent_count = await asyncio.to_thread("
    )

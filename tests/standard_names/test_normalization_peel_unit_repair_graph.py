"""Disposable-graph contract for normalization-peel parent unit repair."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.graph.profiles import resolve_neo4j
from imas_codex.standard_names.graph_ops import (
    repair_normalization_peel_parent_units,
)

_UNIT_FINDING = "name_unit_consistency_check: particle_mass expects kg"


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("normalization-peel unit repair requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("normalization-peel unit repair refuses the project graph URI")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query("MATCH (node) DETACH DELETE node")
    yield uri, password


@pytest.fixture
def graph(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    client = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="normalization-peel-unit-repair",
    )
    client.query("MATCH (node) DETACH DELETE node")
    yield client
    client.query("MATCH (node) DETACH DELETE node")
    client.close()


def _family(
    parent_id: str,
    *,
    children: list[dict[str, object]],
    origin: str = "derived",
    unit: str = "1",
    validation_issues: list[str] | None = None,
    unit_edge: bool = True,
) -> dict[str, object]:
    return {
        "parent": {
            "id": parent_id,
            "origin": origin,
            "unit": unit,
            "validation_issues": validation_issues
            if validation_issues is not None
            else [_UNIT_FINDING],
        },
        "children": children,
        "unit_edge": unit_edge,
    }


def _child(child_id: str, *, unit: str | None = "1") -> dict[str, object]:
    return {"id": child_id, "unit": unit}


def _seed_families(
    graph: GraphClient,
    families: list[dict[str, object]],
) -> None:
    graph.query(
        """
        UNWIND $families AS family
        CREATE (parent:StandardName)
        SET parent = family.parent
        FOREACH (_ IN CASE WHEN family.unit_edge THEN [1] ELSE [] END |
            MERGE (unit:Unit {id: family.parent.unit})
            MERGE (parent)-[:HAS_UNIT]->(unit)
        )
        WITH family, parent
        UNWIND family.children AS child_data
        CREATE (child:StandardName)
        SET child = child_data
        CREATE (child)-[:HAS_PARENT]->(parent)
        """,
        families=families,
    )


def _parent_states(graph: GraphClient) -> dict[str, dict[str, object]]:
    rows = graph.query(
        """
        MATCH (parent:StandardName)
        WHERE EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(parent) }
        OPTIONAL MATCH (parent)-[:HAS_UNIT]->(unit:Unit)
        WITH parent, collect(unit.id) AS unit_edges
        RETURN parent.id AS id,
               parent.origin AS origin,
               parent.unit AS unit,
               parent.validation_issues AS validation_issues,
               unit_edges
        ORDER BY id
        """
    )
    return {
        str(row["id"]): {
            "origin": row["origin"],
            "unit": row["unit"],
            "validation_issues": row["validation_issues"],
            "unit_edges": row["unit_edges"],
        }
        for row in rows
    }


def _snapshot(graph: GraphClient) -> bytes:
    nodes = graph.query(
        """
        MATCH (node)
        RETURN elementId(node) AS element_id,
               labels(node) AS labels,
               properties(node) AS properties
        ORDER BY element_id
        """
    )
    relationships = graph.query(
        """
        MATCH (start)-[relationship]->(end)
        RETURN elementId(relationship) AS element_id,
               type(relationship) AS type,
               properties(relationship) AS properties,
               elementId(start) AS start_id,
               elementId(end) AS end_id
        ORDER BY element_id
        """
    )
    return json.dumps(
        {"nodes": nodes, "relationships": relationships},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


@pytest.mark.graph
def test_mixed_cohort_returns_the_exact_admitted_partition(
    graph: GraphClient,
) -> None:
    _seed_families(
        graph,
        [
            _family(
                "particle_mass",
                children=[_child("normalized_particle_mass")],
            ),
            _family(
                "electric_current",
                children=[_child("normalised_electric_current")],
            ),
            _family(
                "normalized_collisionality",
                children=[_child("normalized_collisionality_at_midplane")],
            ),
            _family(
                "magnetic_field",
                children=[_child("normalized_magnetic_field")],
                validation_issues=["documentation_check: incomplete"],
            ),
            _family(
                "ion_density",
                children=[_child("normalized_ion_density")],
                origin="pipeline",
            ),
            _family(
                "electron_temperature",
                children=[_child("electron_temperature_at_midplane")],
            ),
        ],
    )

    assert repair_normalization_peel_parent_units(graph) == [
        "electric_current",
        "particle_mass",
    ]
    states = _parent_states(graph)
    assert {
        parent_id for parent_id, state in states.items() if state["unit"] is None
    } == {"electric_current", "particle_mass"}
    assert states["electric_current"]["unit_edges"] == []
    assert states["particle_mass"]["unit_edges"] == []
    assert states["normalized_collisionality"]["unit"] == "1"
    assert states["magnetic_field"]["unit"] == "1"
    assert states["ion_density"]["unit"] == "1"
    assert states["electron_temperature"]["unit"] == "1"


@pytest.mark.graph
def test_parent_with_own_normalization_marker_returns_exact_empty_result(
    graph: GraphClient,
) -> None:
    _seed_families(
        graph,
        [
            _family(
                "normalized_particle_mass",
                children=[_child("normalized_particle_mass_at_midplane")],
            )
        ],
    )
    before = _parent_states(graph)

    assert repair_normalization_peel_parent_units(graph) == []
    assert _parent_states(graph) == before


@pytest.mark.graph
def test_parent_without_recorded_unit_finding_returns_exact_empty_result(
    graph: GraphClient,
) -> None:
    _seed_families(
        graph,
        [
            _family(
                "particle_mass",
                children=[_child("normalized_particle_mass")],
                validation_issues=["documentation_check: incomplete"],
            )
        ],
    )
    before = _parent_states(graph)

    assert repair_normalization_peel_parent_units(graph) == []
    assert _parent_states(graph) == before


@pytest.mark.graph
def test_non_normalization_child_returns_exact_empty_result(
    graph: GraphClient,
) -> None:
    _seed_families(
        graph,
        [
            _family(
                "particle_mass",
                children=[
                    _child("normalized_particle_mass"),
                    _child("particle_mass_at_midplane"),
                ],
            )
        ],
    )
    before = _parent_states(graph)

    assert repair_normalization_peel_parent_units(graph) == []
    assert _parent_states(graph) == before


@pytest.mark.graph
def test_null_unit_child_is_outside_unit_bearing_predicate(
    graph: GraphClient,
) -> None:
    """A null-unit child cannot veto the unit-bearing-child predicate."""
    _seed_families(
        graph,
        [
            _family(
                "particle_mass",
                children=[_child("normalized_particle_mass", unit=None)],
            )
        ],
    )

    assert repair_normalization_peel_parent_units(graph) == ["particle_mass"]
    assert _parent_states(graph)["particle_mass"] == {
        "origin": "derived",
        "unit": None,
        "validation_issues": [_UNIT_FINDING],
        "unit_edges": [],
    }


@pytest.mark.graph
def test_scalar_only_candidate_is_admitted_without_a_unit_edge(
    graph: GraphClient,
) -> None:
    _seed_families(
        graph,
        [
            _family(
                "particle_mass",
                children=[_child("normalized_particle_mass")],
                unit_edge=False,
            )
        ],
    )

    assert repair_normalization_peel_parent_units(graph) == ["particle_mass"]
    assert _parent_states(graph)["particle_mass"] == {
        "origin": "derived",
        "unit": None,
        "validation_issues": [_UNIT_FINDING],
        "unit_edges": [],
    }


@pytest.mark.graph
def test_replay_after_repair_is_write_free(
    graph: GraphClient,
) -> None:
    _seed_families(
        graph,
        [
            _family(
                "particle_mass",
                children=[_child("normalized_particle_mass")],
            ),
            _family(
                "electric_current",
                children=[_child("normalised_electric_current")],
            ),
        ],
    )

    assert repair_normalization_peel_parent_units(graph) == [
        "electric_current",
        "particle_mass",
    ]
    before_replay = _snapshot(graph)

    assert repair_normalization_peel_parent_units(graph) == []
    assert _snapshot(graph) == before_replay

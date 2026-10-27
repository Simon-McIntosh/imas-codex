"""The lineage-side graph-store domain writers set a name's domain one way.

Every site in the graph-store lineage modules that sets a standard name's
``physics_domain``, its ``source_domains`` or a ``HAS_PHYSICS_DOMAIN`` edge goes
through :func:`imas_codex.standard_names.domain_decision.write_name_domain`, so
the three cannot disagree. Each test here plants a name on a disposable graph,
runs the real site, and reads the three back.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import (  # noqa: F401 — wires the graph services
    graph_ops,
)
from imas_codex.standard_names.graph_store import (
    graph_foundation as foundation,
    graph_lineage as lineage,
)


@pytest.fixture
def graph(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    client = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="name-domain-writer-lineage-sites",
    )
    client.query("MATCH (node) DETACH DELETE node")
    yield client
    client.query("MATCH (node) DETACH DELETE node")
    client.close()


class _BoundGraph:
    """Hand the routed site the disposable graph it must write through."""

    def __init__(self, client: GraphClient) -> None:
        self._client = client

    def __enter__(self) -> GraphClient:
        return self._client

    def __exit__(self, *args: object) -> None:
        return None


def _services() -> Any:
    """The wired composition root the module functions compose through."""
    return foundation.services_for_call()


def _read(graph: GraphClient, name_id: str) -> dict[str, Any]:
    rows = list(
        graph.query(
            """
            MATCH (sn:StandardName {id: $id})
            OPTIONAL MATCH (sn)-[:HAS_PHYSICS_DOMAIN]->(d:PhysicsDomain)
            RETURN sn.physics_domain AS physics_domain,
                   sn.source_domains AS source_domains,
                   collect(DISTINCT d.id) AS edge_domains,
                   sn.domain_correction AS correction
            """,
            id=name_id,
        )
    )
    return dict(rows[0])


def _seed_corrected_name(graph: GraphClient) -> None:
    graph.query(
        """
        CREATE (sn:StandardName {id: 'sn:plasma_current',
                                 physics_domain: 'general',
                                 source_domains: ['general']})
        CREATE (src:StandardNameSource {id: 'src:plasma_current'})
        CREATE (dd:IMASNode {id: 'dd:equilibrium/time_slice/profiles_1d/psi',
                             physics_domain: 'equilibrium'})
        CREATE (stale:PhysicsDomain {id: 'general'})
        CREATE (src)-[:PRODUCED_NAME]->(sn)
        CREATE (src)-[:FROM_DD_PATH]->(dd)
        CREATE (sn)-[:HAS_PHYSICS_DOMAIN]->(stale)
        """
    )


@pytest.mark.graph
def test_catalog_edit_correction_writes_through_the_one_writer(
    graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sn reclassify`` sets the scalar as a correction and the rest as the writer does.

    The corrected domain is the scalar, ``source_domains`` stays the name's DD
    sources' domains, and the edges are exactly those domains. A site that
    writes ``source_domains = [corrected]`` and one edge by hand fails the
    second and third assertions.
    """
    _seed_corrected_name(graph)
    monkeypatch.setattr(foundation, "GraphClient", lambda: _BoundGraph(graph))

    result = foundation.reclassify_standard_name_domain(
        "sn:plasma_current",
        "magnetic_field_diagnostics",
        reason="The signal measures plasma current, not flux.",
    )

    assert result["ok"] is True
    row = _read(graph, "sn:plasma_current")
    assert row["physics_domain"] == "magnetic_field_diagnostics"
    assert row["correction"] == "magnetic_field_diagnostics"
    assert set(row["source_domains"]) == {"equilibrium"}
    assert set(row["edge_domains"]) == {"equilibrium"}


def _seed_edge_pass_name(graph: GraphClient) -> None:
    """A name whose DD source says equilibrium while a stale edge says general."""
    graph.query(
        """
        CREATE (sn:StandardName {id: 'sn:plasma_current',
                                 physics_domain: 'general',
                                 source_domains: ['general'],
                                 name_stage: 'accepted'})
        CREATE (src:StandardNameSource {id: 'src:plasma_current'})
        CREATE (dd:IMASNode {id: 'dd:equilibrium/time_slice/profiles_1d/psi',
                             physics_domain: 'equilibrium'})
        CREATE (stale:PhysicsDomain {id: 'general'})
        CREATE (src)-[:PRODUCED_NAME]->(sn)
        CREATE (src)-[:FROM_DD_PATH]->(dd)
        CREATE (sn)-[:HAS_PHYSICS_DOMAIN]->(stale)
        """
    )


def _seed_derived_parent(graph: GraphClient) -> dict[str, Any]:
    """A derived parent whose only child's DD source says transport."""
    graph.query(
        """
        CREATE (parent:StandardName {id: 'sn:electron_temperature',
                                     physics_domain: 'transport',
                                     name_stage: 'accepted'})
        CREATE (child:StandardName {id: 'sn:electron_temperature_at_axis',
                                    physics_domain: 'transport'})
        CREATE (child)-[:HAS_PARENT]->(parent)
        CREATE (child_src:StandardNameSource {id: 'src:electron_temperature_at_axis'})
        CREATE (child_dd:IMASNode {id: 'dd:equilibrium/time_slice/profiles_1d/t_i',
                                   physics_domain: 'transport'})
        CREATE (child_src)-[:PRODUCED_NAME]->(child)
        CREATE (child_src)-[:FROM_DD_PATH]->(child_dd)
        """
    )
    return {
        "parent_id": "sn:electron_temperature",
        "authorized_unit": "eV",
        "edge_kinds": ["coordinate"],
        "child_data": [
            {
                "id": "sn:electron_temperature_at_axis",
                "unit": "eV",
                "physics_domain": "transport",
                "kind": "scalar",
                "cocos": None,
                "op_kind": "coordinate",
            }
        ],
    }


@pytest.mark.graph
def test_edge_pass_writes_the_domain_through_the_one_writer(
    graph: GraphClient,
) -> None:
    """``_write_standard_name_edges`` rewrites the edges to the source set.

    The name's DD source says ``equilibrium`` and a stale edge points at
    ``general``. Reading the three back must show the source set and the edges
    equal to ``equilibrium``: a site that merges the payload's domains without
    deleting the stale edge leaves ``general`` in place and fails.
    """
    _seed_edge_pass_name(graph)
    lineage._write_standard_name_edges(
        graph, [{"id": "sn:plasma_current"}], services=_services()
    )

    row = _read(graph, "sn:plasma_current")
    assert set(row["source_domains"]) == {"equilibrium"}
    assert set(row["edge_domains"]) == {"equilibrium"}


@pytest.mark.graph
def test_single_materializer_writes_the_domain_through_the_one_writer(
    graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_materialize_derived_parent_rows`` gives the parent its edges.

    The parent's scalar comes from its children, as before, and the source set
    and edges come from the writer. A site that sets only the two scalars writes
    no edge at all and fails on the edge set.
    """
    parent = _seed_derived_parent(graph)
    services = _services()
    monkeypatch.setattr(
        services, "_validate_derived_parent_identity", lambda *a, **k: []
    )
    assert (
        lineage._materialize_derived_parent_rows(graph, [parent], services=services)
        == 1
    )

    row = _read(graph, "sn:electron_temperature")
    assert row["physics_domain"] == "transport"
    assert set(row["source_domains"]) == {"transport"}
    assert set(row["edge_domains"]) == {"transport"}


@pytest.mark.graph
def test_batched_materializer_writes_the_domain_through_the_one_writer(
    graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_materialize_derived_parent_rows_batched`` writes the same three.

    The cohort-wide statement sets the scalars for the batch; the domain, its
    source set and the edges still come from the writer. A cohort statement that
    sets ``source_domains`` itself leaves the edge set empty and fails.
    """
    parent = _seed_derived_parent(graph)
    services = _services()
    monkeypatch.setattr(
        services, "_validate_derived_parent_identity", lambda *a, **k: []
    )
    assert (
        lineage._materialize_derived_parent_rows_batched(
            graph, [parent], services=services
        )
        == 1
    )

    row = _read(graph, "sn:electron_temperature")
    assert row["physics_domain"] == "transport"
    assert set(row["source_domains"]) == {"transport"}
    assert set(row["edge_domains"]) == {"transport"}

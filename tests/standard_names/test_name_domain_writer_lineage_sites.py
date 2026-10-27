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
from imas_codex.standard_names.graph_store import graph_foundation as foundation


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

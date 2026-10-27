"""A catalog-path name gets its domain from exactly one writer application.

Both halves of the graph-store split route every ``StandardName`` domain write
through :func:`imas_codex.standard_names.domain_decision.write_name_domain`.
The catalog write path delegates to that writer from its structural tail pass,
which is the only place on the path that sets ``physics_domain``,
``source_domains`` or a ``HAS_PHYSICS_DOMAIN`` edge.

This test drives the whole catalog path against a disposable graph and counts
the writer's applications, so a path that applies it twice and a path that
applies it never both fail. The payload carries a domain of its own, which the
merged path must ignore: the name already holds a decision the writer keeps.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.graph, pytest.mark.timeout(180)]

_PAYLOAD_DOMAIN = "transport"
_SOURCE_DOMAIN = "equilibrium"


def _plant_decided_name_with_dd_source(gc: Any, sn_id: str, dd_id: str) -> None:
    """A decided name produced by one source bound to one DD path.

    The stored decision names ``equilibrium``; the DD source carries the same
    domain, so the writer's three outputs agree and any second writer shows up
    as a disagreement rather than a coincidence.
    """
    gc.query(
        """
        MERGE (sn:StandardName {id: $sn_id})
        SET sn.name_stage = 'drafted', sn.origin = 'pipeline',
            sn.validation_status = 'valid', sn.chain_length = 0,
            sn.docs_stage = 'pending', sn.docs_chain_length = 0,
            sn.description = 'Test quantity', sn.kind = 'scalar', sn.unit = 'eV',
            sn.physics_domain = $stale_domain,
            sn.source_domains = [$stale_domain]
        """,
        stale_domain=_PAYLOAD_DOMAIN,
        sn_id=sn_id,
    )
    gc.query(
        """
        MATCH (sn:StandardName {id: $sn_id})
        MERGE (dd:IMASNode {id: $dd_id})
        SET dd.physics_domain = $dd_domain
        MERGE (src:StandardNameSource {id: $src_id})
        SET src.status = 'composed', src.source_type = 'dd',
            src.source_id = $dd_id, src.description = 'A test quantity',
            src.produced_sn_id = $sn_id,
            src.claimed_at = null, src.claim_token = null
        MERGE (src)-[:PRODUCED_NAME]->(sn)
        MERGE (src)-[:FROM_DD_PATH]->(dd)
        """,
        sn_id=sn_id,
        dd_id=dd_id,
        dd_domain=_SOURCE_DOMAIN,
        src_id=f"src:{dd_id}",
    )
    gc.query(
        """
        MATCH (sn:StandardName {id: $sn_id})
        SET sn.domain_decision_candidates = [$domain],
            sn.domain_decision_choice = $domain,
            sn.domain_decision_probability = 0.9,
            sn.domain_decision_model = 'test/model',
            sn.domain_decision_at = '2026-01-01T00:00:00+00:00'
        """,
        sn_id=sn_id,
        domain=_SOURCE_DOMAIN,
    )


class _RecordingClient:
    """Every query the catalog path runs, forwarded to the real graph."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.queries: list[str] = []

    def query(self, cypher: str, **params: Any) -> Any:
        self.queries.append(cypher)
        return self._inner.query(cypher, **params)


def _domain_state(gc: Any, sn_id: str) -> dict[str, Any]:
    rows = list(
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            OPTIONAL MATCH (sn)-[:HAS_PHYSICS_DOMAIN]->(d:PhysicsDomain)
            RETURN sn.physics_domain AS physics_domain,
                   sn.source_domains AS source_domains,
                   [x IN collect(DISTINCT d.id) WHERE x IS NOT NULL]
                     AS edge_domains
            """,
            id=sn_id,
        )
    )
    row = dict(rows[0])
    row["source_domains"] = row["source_domains"] or []
    row["edge_domains"] = row["edge_domains"] or []
    return row


def test_catalog_write_applies_the_domain_writer_exactly_once(
    disposable_neo4j_empty: str,
) -> None:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.graph_store import graph_lineage
    from imas_codex.standard_names.graph_store.graph_catalog import (
        write_standard_names,
    )

    # A grammar-valid identity: the catalog path rejects one it cannot parse,
    # which would make this measurement pass while exercising nothing.
    sn_id = "ion_temperature"
    applications: list[str] = []
    real_writer = getattr(graph_lineage, "write_name_domain", None)

    def _counting(gc: Any, name: str, decision: Any, **kwargs: Any) -> Any:
        applications.append(name)
        return real_writer(gc, name, decision, **kwargs)

    with GraphClient() as real:
        _plant_decided_name_with_dd_source(real, sn_id, "dd:routing/path")
        recorder = _RecordingClient(real)
        with patch.object(graph_lineage, "write_name_domain", _counting, create=True):
            write_standard_names(
                [
                    {
                        "id": sn_id,
                        "source_types": ["dd"],
                        "physics_domain": [_PAYLOAD_DOMAIN],
                        "source_domains": [_PAYLOAD_DOMAIN],
                        "name_stage": "drafted",
                    }
                ],
                gc=recorder,
            )
        settled = _domain_state(real, sn_id)
        direct_domain_writes = [
            cypher for cypher in recorder.queries if "sn.physics_domain =" in cypher
        ]

    assert applications == [sn_id], (
        "the catalog path applied the domain writer "
        f"{len(applications)} times, not exactly once"
    )
    assert settled["physics_domain"] == _SOURCE_DOMAIN, (
        "the payload domain replaced the name's stored decision"
    )
    assert set(settled["source_domains"]) == {_SOURCE_DOMAIN}
    assert set(settled["edge_domains"]) == {_SOURCE_DOMAIN}
    assert len(direct_domain_writes) == 1, (
        "a statement outside the one writer set the scalar: "
        f"{len(direct_domain_writes)} domain writes ran"
    )
    # The single application is the writer's atomic write of all three.
    assert "sn.source_domains = $source_domains" in direct_domain_writes[0]

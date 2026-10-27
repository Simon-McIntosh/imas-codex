"""The catalog-side writers set no domain of their own.

Two modules in ``imas_codex.standard_names.graph_store`` used to set a
``StandardName``'s ``physics_domain`` directly: the promote-on-higher-rank
promotion inside ``graph_catalog.write_standard_names``, and the successor
provision inside ``graph_refinement.persist_refined_name``. Each now leaves the
domain to :func:`imas_codex.standard_names.domain_decision.write_name_domain`,
which sets ``physics_domain``, ``source_domains`` and the
``HAS_PHYSICS_DOMAIN`` edges together.

Each test plants a name with a DD source, exercises one routed path against a
disposable graph, and asserts the three agree afterwards. Reverting either site
to its direct write turns its test red.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from imas_codex.standard_names.domain_decision import (
    DomainDecision,
    write_name_domain,
)

pytestmark = [pytest.mark.graph, pytest.mark.timeout(180)]

_PREFIX = "test_domain_route__"


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _plant_name_with_dd_source(
    gc: Any, sn_id: str, dd_id: str, *, dd_domain: str
) -> None:
    """A name produced by one source bound to one DD path carrying a domain."""
    gc.query(
        """
        MERGE (sn:StandardName {id: $sn_id})
        SET sn.name_stage = 'drafted', sn.origin = 'pipeline',
            sn.validation_status = 'valid', sn.chain_length = 0,
            sn.docs_stage = 'pending', sn.docs_chain_length = 0,
            sn.description = 'Test quantity', sn.kind = 'scalar', sn.unit = 'eV'
        """,
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
        dd_domain=dd_domain,
        src_id=f"src:{dd_id}",
    )


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
    return row


def _decided(name: str, domain: str) -> DomainDecision:
    return DomainDecision(
        name=name,
        candidates=(domain,),
        choice=domain,
        probability=0.9,
        model="test/model",
        decided_at="2026-01-01T00:00:00+00:00",
    )


def test_write_standard_names_leaves_the_domain_to_the_one_writer(
    disposable_neo4j_empty: str,
) -> None:
    """A payload domain is not written; the one writer sets all three."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.graph_store.graph_catalog import (
        write_standard_names,
    )

    sn_id = _uid("catalog")
    with GraphClient() as gc:
        _plant_name_with_dd_source(
            gc, sn_id, "dd:catalog/path", dd_domain="equilibrium"
        )
        write_standard_names(
            [
                {
                    "id": sn_id,
                    "source_types": ["dd"],
                    "physics_domain": ["transport"],
                    "name_stage": "drafted",
                }
            ],
            gc=gc,
        )
        untouched = _domain_state(gc, sn_id)
        assert untouched["physics_domain"] is None, (
            "write_standard_names wrote the payload domain; the one writer owns it"
        )
        assert untouched["source_domains"] == []

        write_name_domain(gc, sn_id, _decided(sn_id, "equilibrium"))
        settled = _domain_state(gc, sn_id)

    assert settled["physics_domain"] == "equilibrium"
    assert set(settled["source_domains"]) == {"equilibrium"}
    assert set(settled["edge_domains"]) == set(settled["source_domains"])


class _ProxyClient:
    """A graph client that is not a :class:`GraphClient`.

    A proxy, subclass or wrapper a caller hands in must reach the domain
    writer exactly as the concrete class does: a client accepted by the call
    path that silently skips the write drops a name's domain.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _ProxyClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def query(self, cypher: str, **params: Any) -> Any:
        return self._inner.query(cypher, **params)

    def session(self) -> Any:
        return self._inner.session()


class _ServicesBehindAProxy:
    """The bound services with a client that is not a GraphClient."""

    def __init__(self, services: Any, client: Any) -> None:
        self._services = services
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._services, name)

    def GraphClient(self) -> Any:
        return self._client


def test_persist_refined_name_writes_the_domain_for_a_client_that_is_not_a_graph_client(
    disposable_neo4j_empty: str,
) -> None:
    """A proxy client gets the domain written, not a skipped write."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.graph_store.graph_foundation import (
        services_for_call,
    )
    from imas_codex.standard_names.graph_store.graph_refinement import (
        persist_refined_name,
    )

    old = _uid("old_proxy")
    new = _uid("new_proxy")
    with GraphClient() as real:
        _plant_name_with_dd_source(real, old, "dd:proxy/path", dd_domain="equilibrium")
        real.query(
            """
            MATCH (sn:StandardName {id: $id})
            SET sn.name_stage = 'refining',
                sn.physics_domain = 'transport',
                sn.domain_decision_candidates = ['equilibrium', 'transport'],
                sn.domain_decision_choice = 'equilibrium',
                sn.domain_decision_probability = 0.9,
                sn.domain_decision_model = 'test/model',
                sn.domain_decision_at = '2026-01-01T00:00:00+00:00'
            """,
            id=old,
        )
        services = _ServicesBehindAProxy(services_for_call(), _ProxyClient(real))
        assert not isinstance(services.GraphClient(), GraphClient)

        persist_refined_name(
            old_name=old,
            new_name=new,
            description="Test quantity",
            services=services,
        )
        settled = _domain_state(real, new)

    assert settled["physics_domain"] == "equilibrium", (
        "the domain write was skipped for a client that is not a GraphClient"
    )
    assert set(settled["source_domains"]) == {"equilibrium"}
    assert set(settled["edge_domains"]) == set(settled["source_domains"])


def test_persist_refined_name_carries_the_stored_decision_through_the_writer(
    disposable_neo4j_empty: str,
) -> None:
    """A renamed successor's domain comes from the stored decision, with edges."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.graph_store.graph_refinement import (
        persist_refined_name,
    )

    old = _uid("old")
    new = _uid("new")
    with GraphClient() as gc:
        _plant_name_with_dd_source(gc, old, "dd:refine/path", dd_domain="equilibrium")
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            SET sn.name_stage = 'refining',
                sn.physics_domain = 'transport',
                sn.domain_decision_candidates = ['equilibrium', 'transport'],
                sn.domain_decision_choice = 'equilibrium',
                sn.domain_decision_probability = 0.9,
                sn.domain_decision_model = 'test/model',
                sn.domain_decision_at = '2026-01-01T00:00:00+00:00'
            """,
            id=old,
        )
        persist_refined_name(
            old_name=old,
            new_name=new,
            description="Test quantity",
        )
        settled = _domain_state(gc, new)

    assert settled["physics_domain"] == "equilibrium"
    assert set(settled["source_domains"]) == {"equilibrium"}
    assert set(settled["edge_domains"]) == set(settled["source_domains"])

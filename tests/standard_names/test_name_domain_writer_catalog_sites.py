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

from tests.standard_names.conftest import (
    domain_state,
    plant_name_with_dd_source,
)

pytestmark = [pytest.mark.graph, pytest.mark.timeout(180)]

_PREFIX = "test_domain_route__"


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def test_write_standard_names_leaves_the_domain_to_the_one_writer(
    disposable_neo4j_empty: str,
) -> None:
    """The payload domain is ignored; the one writer sets all three."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.graph_store.graph_catalog import (
        write_standard_names,
    )

    # The catalog path gates on the grammar, so this identity is a name the
    # grammar parses and recomposes — a synthetic id is rejected before the
    # write and the test would then measure nothing.
    sn_id = "electron_temperature"
    with GraphClient() as gc:
        plant_name_with_dd_source(
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
        settled = domain_state(gc, sn_id)

    assert settled["physics_domain"] == "equilibrium", (
        "write_standard_names wrote the payload domain instead of the domain "
        "its sources carry; the one writer owns the scalar"
    )
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
        plant_name_with_dd_source(real, old, "dd:proxy/path", dd_domain="equilibrium")
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
        settled = domain_state(real, new)

    assert settled["physics_domain"] == "equilibrium", (
        "the domain write was skipped for a client that is not a GraphClient"
    )
    assert set(settled["source_domains"]) == {"equilibrium"}
    assert set(settled["edge_domains"]) == set(settled["source_domains"])


class _ClientThatCannotQuery:
    """A client with a session but no query interface."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _ClientThatCannotQuery:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def session(self) -> Any:
        return self._inner.session()


def test_persist_refined_name_refuses_an_unwritable_client_before_it_commits(
    disposable_neo4j_empty: str,
) -> None:
    """A client that cannot write the domain leaves no committed successor."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.graph_store.graph_foundation import (
        services_for_call,
    )
    from imas_codex.standard_names.graph_store.graph_refinement import (
        persist_refined_name,
    )

    old = _uid("old_unwritable")
    new = _uid("new_unwritable")
    with GraphClient() as real:
        plant_name_with_dd_source(
            real, old, "dd:unwritable/path", dd_domain="equilibrium"
        )
        real.query(
            "MATCH (sn:StandardName {id: $id}) SET sn.name_stage = 'refining'",
            id=old,
        )
        services = _ServicesBehindAProxy(
            services_for_call(), _ClientThatCannotQuery(real)
        )

        with pytest.raises(TypeError):
            persist_refined_name(
                old_name=old,
                new_name=new,
                description="Test quantity",
                services=services,
            )
        rows = list(
            real.query("MATCH (sn:StandardName {id: $id}) RETURN sn.id AS id", id=new)
        )

    assert rows == [], (
        "the rename committed a successor whose domain the caller could not write"
    )


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
        plant_name_with_dd_source(gc, old, "dd:refine/path", dd_domain="equilibrium")
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
        settled = domain_state(gc, new)

    assert settled["physics_domain"] == "equilibrium"
    assert set(settled["source_domains"]) == {"equilibrium"}
    assert set(settled["edge_domains"]) == set(settled["source_domains"])

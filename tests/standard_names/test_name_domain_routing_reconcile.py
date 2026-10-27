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

A second statement-level check counts *any* clause that assigns the scalar,
whatever its spelling, so a direct write that avoids the literal
``sn.physics_domain =`` still fails: ``+=``, a map merge, or a map parameter.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import pytest

from tests.recording_graph_client import RecordingGraphClient
from tests.standard_names.test_name_domain_writer_catalog_sites import (
    _domain_state,
)

pytestmark = [pytest.mark.timeout(180)]

_PAYLOAD_DOMAIN = "transport"
_SOURCE_DOMAIN = "equilibrium"

# One SET clause: from the keyword to the next clause keyword that can follow it.
_SET_CLAUSE = re.compile(
    r"(?is)\bSET\b(.*?)(?=\b(?:SET|WITH|RETURN|MATCH|OPTIONAL|MERGE|CREATE"
    r"|DELETE|DETACH|REMOVE|UNWIND|CALL|FOREACH|WHERE|ON|ORDER|LIMIT|SKIP)\b|$)"
)
# `x.physics_domain = …`, `x.physics_domain += …`, `x.physics_domain |= {…}`
_PROPERTY_WRITE = re.compile(r"(?i)\.physics_domain\s*(?:\+?=|\|=)")
# `SET x = {…, physics_domain: …}` and `SET x += {physics_domain: …}`
_MAP_WRITE = re.compile(r"(?i)\b[A-Za-z_]\w*\s*\+?=\s*\{[^{}]*\bphysics_domain\s*:")
# `SET x += $props`, where `$props` carries the key
_MAP_PARAM_WRITE = re.compile(r"(?i)\b[A-Za-z_]\w*\s*\+?=\s*\$([A-Za-z_]\w*)")


def _sets_domain(cypher: str, params: dict[str, Any]) -> bool:
    """True when this statement assigns a ``physics_domain`` property.

    Scoped to SET clauses, so a read (a ``WHERE`` predicate, a ``RETURN``
    projection) never counts, and spelled so a write cannot hide behind a
    different assignment form.
    """
    for clause in _SET_CLAUSE.findall(cypher):
        if _PROPERTY_WRITE.search(clause) or _MAP_WRITE.search(clause):
            return True
        for param_name in _MAP_PARAM_WRITE.findall(clause):
            value = params.get(param_name)
            if isinstance(value, dict) and "physics_domain" in value:
                return True
    return False


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
        SET src.status = 'composed', src.description = 'A test quantity', src.source_type = 'dd',
            src.source_id = $dd_id,
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


class _ForwardingRecorder(RecordingGraphClient):
    """Capture every statement the catalog path runs, and execute it."""

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self._inner = inner

    def query(self, cypher: str, **params: Any) -> Any:
        self.calls.append((cypher, params))
        return self._inner.query(cypher, **params)


@pytest.mark.parametrize(
    "cypher, params",
    [
        ("MATCH (sn:StandardName) SET sn.physics_domain = $d", {"d": "x"}),
        ("MATCH (sn:StandardName) SET sn.physics_domain += [$d]", {"d": "x"}),
        (
            "MATCH (sn:StandardName) SET sn = {id: sn.id, physics_domain: $d}",
            {"d": "x"},
        ),
        ("MATCH (sn:StandardName) SET sn += {physics_domain: $d}", {"d": "x"}),
        (
            "MATCH (sn:StandardName) SET sn += $props",
            {"props": {"physics_domain": "x"}},
        ),
    ],
)
def test_the_detector_sees_each_assignment_spelling(
    cypher: str, params: dict[str, Any]
) -> None:
    """The scalar's detector must see a write it was not shown a template for."""
    assert _sets_domain(cypher, params)


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (sn:StandardName) WHERE sn.physics_domain = $d RETURN sn.id",
        "MATCH (sn:StandardName) RETURN sn.physics_domain AS physics_domain",
        "MATCH (sn:StandardName) SET sn.unit = $u",
    ],
)
def test_the_detector_ignores_a_read(cypher: str) -> None:
    """A projection or a predicate is not a write, so a count of writes is one."""
    assert not _sets_domain(cypher, {"d": "x", "u": "eV"})


@pytest.mark.graph
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
        recorder = _ForwardingRecorder(real)
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
        domain_writes = [
            cypher for cypher, params in recorder.calls if _sets_domain(cypher, params)
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
    assert len(domain_writes) == 1, (
        "the scalar was assigned by a statement outside the one writer: "
        f"{len(domain_writes)} domain writes ran"
    )
    # The single application is the writer's atomic write of all three.
    assert "sn.source_domains = $source_domains" in domain_writes[0]

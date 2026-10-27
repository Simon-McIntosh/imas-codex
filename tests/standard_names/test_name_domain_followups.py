"""The catalog write leaves a name's domain whole, and the writer's records keep.

Four defects the domain routing's reviews left open, each measured here against
the behaviour it names:

* a name written through ``write_standard_names`` with no stored decision ended
  with no ``physics_domain`` while its ``source_domains`` and edges were set;
* an undecided stored decision lost the candidates it was made among;
* ``domain_decision``'s undecided constructor was private to its owner while
  other modules imported it;
* ``persist_refined_name`` carried a ``source_domains`` parameter nothing read.

The disposable-graph tests start from an empty graph and plant their own
fixture nodes, so they measure the write path rather than a shared state.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

pytestmark = [pytest.mark.timeout(180)]

_DD_DOMAIN = "equilibrium"


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


def _plant_name_with_dd_source(gc: Any, sn_id: str, dd_id: str) -> None:
    """A drafted name produced by one source bound to one classified DD path."""
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
        dd_domain=_DD_DOMAIN,
        src_id=f"src:{dd_id}",
    )


@pytest.mark.graph
def test_write_standard_names_alone_leaves_an_undecided_name_whole(
    disposable_neo4j_empty: str,
) -> None:
    """A name's scalar, its source set and its edges agree after the write alone.

    No follow-up runs: the catalog write is the only call. The name carries a
    payload domain and no stored decision, so the scalar can only come from the
    one domain its DD sources carry — the writer's derived value, not a guess.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.graph_store.graph_catalog import (
        write_standard_names,
    )

    sn_id = "electron_temperature"
    with GraphClient() as gc:
        _plant_name_with_dd_source(gc, sn_id, "dd:followups/path")
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
        settled = _domain_state(gc, sn_id)

    assert settled["physics_domain"] == _DD_DOMAIN, (
        "a name written through write_standard_names alone carried no domain"
    )
    assert set(settled["source_domains"]) == {_DD_DOMAIN}
    assert set(settled["edge_domains"]) == set(settled["source_domains"])
    assert settled["physics_domain"] in settled["source_domains"], (
        "the scalar disagrees with the source set it was written beside"
    )


@pytest.mark.graph
def test_an_undecided_stored_decision_keeps_its_candidates(
    disposable_neo4j_empty: str,
) -> None:
    """A stored decision with candidates and no choice keeps the candidates."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.domain_decision import write_name_domain
    from imas_codex.standard_names.graph_store.graph_lineage import (
        _stored_domain_decisions,
    )

    sn_id = "electron_temperature"
    candidates = ["equilibrium", "core_plasma_physics", "general"]
    with GraphClient() as gc:
        _plant_name_with_dd_source(gc, sn_id, "dd:followups/candidates")
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            SET sn.physics_domain = $domain,
                sn.domain_decision_candidates = $candidates,
                sn.domain_decision_choice = null
            """,
            id=sn_id,
            domain=_DD_DOMAIN,
            candidates=candidates,
        )
        stored = _stored_domain_decisions(gc, [sn_id])[sn_id]
        assert stored is not None, "the graph holds the name, so it is not absent"
        assert not stored.candidates == (), (
            "the stored decision lost the candidates it was made among"
        )
        write_name_domain(gc, sn_id, stored)
        row = dict(
            next(
                iter(
                    gc.query(
                        """
                        MATCH (sn:StandardName {id: $id})
                        RETURN sn.domain_decision_candidates AS candidates
                        """,
                        id=sn_id,
                    )
                )
            )
        )

    assert list(stored.candidates) == candidates
    assert row["candidates"] == candidates, (
        "the writer overwrote the stored candidates with an empty list"
    )


def test_undecided_is_public_in_its_owner() -> None:
    """The undecided constructor is exported by ``domain_decision``."""
    from imas_codex.standard_names import domain_decision

    assert hasattr(domain_decision, "undecided")
    assert "undecided" in domain_decision.__all__
    decision = domain_decision.undecided("n", ("general",), "no answer")
    assert decision.candidates == ("general",)
    assert not decision.decided


def test_persist_refined_name_carries_no_source_domains_parameter() -> None:
    """A parameter no caller's value reaches is not carried."""
    from imas_codex.standard_names.graph_store.graph_refinement import (
        persist_refined_name,
    )

    parameters = inspect.signature(persist_refined_name).parameters
    assert "source_domains" not in parameters

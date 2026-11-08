"""The fold and refinement writers preserve acyclic, single-tip lineage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import edit
from imas_codex.standard_names.graph_store import graph_refinement

pytestmark = pytest.mark.graph

OLD = "electron_number_density"
ANCESTOR = "electron_density"
BRANCH = "electron_density_at_plasma_boundary"
PROPOSED = "electron_density_at_pedestal_top"
SOURCE_PATH = "core_profiles/profiles_1d/electrons/density"


def _seed(*, ancestor_stage: str, branch: bool = False) -> None:
    with GraphClient() as graph:
        graph.query(
            """
            CREATE (old:StandardName {id: $old, name_stage: 'accepted',
              status: 'draft', origin: 'pipeline', unit: 'm^-3',
              description: 'Electron number density', validation_status: 'valid'})
            CREATE (ancestor:StandardName {id: $ancestor, name_stage: $stage,
              status: 'draft', origin: 'pipeline', unit: 'm^-3',
              description: 'Electron number density', validation_status: 'valid',
              superseded_from_stage: 'accepted'})
            CREATE (old)-[:REFINED_FROM]->(ancestor)
            CREATE (dd:IMASNode {id: $path, unit: 'm^-3',
              documentation: 'Electron number density'})
            CREATE (source:StandardNameSource {id: $source_id,
              source_type: 'dd', source_id: $path, status: 'composed',
              produced_sn_id: $old})
            CREATE (source)-[:FROM_DD_PATH]->(dd)
            CREATE (source)-[:PRODUCED_NAME]->(old)
            CREATE (dd)-[:HAS_STANDARD_NAME]->(old)
            CREATE (unit:Unit {id: 'm^-3', symbol: 'm^-3'})
            CREATE (old)-[:HAS_UNIT]->(unit)
            CREATE (ancestor)-[:HAS_UNIT]->(unit)
            CREATE (dd)-[:HAS_UNIT]->(unit)
            """,
            old=OLD,
            ancestor=ANCESTOR,
            stage=ancestor_stage,
            path=SOURCE_PATH,
            source_id=f"dd:{SOURCE_PATH}",
        )
        if branch:
            graph.query(
                """
                MATCH (old:StandardName {id: $old})
                CREATE (branch:StandardName {id: $branch,
                  name_stage: 'accepted', status: 'draft', origin: 'pipeline',
                  unit: 'm^-3'})
                CREATE (branch)-[:REFINED_FROM]->(old)
                """,
                old=OLD,
                branch=BRANCH,
            )


def _state() -> dict:
    with GraphClient() as graph:
        names = graph.query(
            "MATCH (n:StandardName) RETURN n.id AS id, n.name_stage AS stage "
            "ORDER BY id"
        )
        edges = graph.query(
            "MATCH (a:StandardName)-[:REFINED_FROM]->(b:StandardName) "
            "RETURN a.id AS successor, b.id AS predecessor "
            "ORDER BY successor, predecessor"
        )
        receipts = graph.query(
            "MATCH (c:StandardNameChange) RETURN c.id AS id ORDER BY id"
        )
        sources = graph.query(
            "MATCH (s:StandardNameSource) RETURN s.id AS id, "
            "s.produced_sn_id AS produced ORDER BY id"
        )
    return {"names": names, "edges": edges, "receipts": receipts, "sources": sources}


def _refine(new_name: str) -> dict[str, str]:
    return graph_refinement.persist_refined_name(
        old_name=OLD,
        new_name=new_name,
        description="Electron number density",
        unit="m^-3",
        expected_old_stage="accepted",
        services=SimpleNamespace(
            GraphClient=GraphClient, bump_sn_run_counter=lambda *_: None
        ),
    )


def test_fold_onto_accepted_ancestor_refuses(
    disposable_neo4j_empty: str, seed_dd_resolutions
) -> None:
    _seed(ancestor_stage="accepted")
    seed_dd_resolutions([{"path": SOURCE_PATH, "effective": "m^-3"}])
    before = _state()

    result = edit.supersede_into(OLD, ANCESTOR)

    assert result["ok"] is False
    assert "REFINED_FROM cycle" in result["reason"]
    assert _state() == before


def test_fold_onto_tombstoned_ancestor_refuses(
    disposable_neo4j_empty: str, seed_dd_resolutions
) -> None:
    _seed(ancestor_stage="superseded")
    seed_dd_resolutions([{"path": SOURCE_PATH, "effective": "m^-3"}])
    before = _state()

    result = edit.supersede_into(OLD, ANCESTOR)

    assert result["ok"] is False
    assert "REFINED_FROM cycle" in result["reason"]
    assert _state() == before


def test_refine_onto_superseded_ancestor_refuses(
    disposable_neo4j_empty: str, seed_dd_resolutions
) -> None:
    _seed(ancestor_stage="superseded")
    seed_dd_resolutions([{"path": SOURCE_PATH, "effective": "m^-3"}])
    before = _state()

    with pytest.raises(graph_refinement.RefinedNamePersistenceRefusal) as refusal:
        _refine(ANCESTOR)

    assert "successor_lifecycle_collision" in str(refusal.value)
    assert _state() == before


def test_refine_onto_pending_ancestor_refuses(
    disposable_neo4j_empty: str, seed_dd_resolutions
) -> None:
    _seed(ancestor_stage="pending")
    seed_dd_resolutions([{"path": SOURCE_PATH, "effective": "m^-3"}])
    before = _state()

    with pytest.raises(graph_refinement.RefinedNamePersistenceRefusal) as refusal:
        _refine(ANCESTOR)

    assert "REFINED_FROM cycle" in str(refusal.value)
    assert _state() == before


def test_refine_with_existing_live_descendant_refuses(
    disposable_neo4j_empty: str, seed_dd_resolutions
) -> None:
    _seed(ancestor_stage="superseded", branch=True)
    seed_dd_resolutions([{"path": SOURCE_PATH, "effective": "m^-3"}])
    before = _state()

    with pytest.raises(graph_refinement.RefinedNamePersistenceRefusal) as refusal:
        _refine(PROPOSED)

    assert "live descendant" in str(refusal.value)
    assert BRANCH in str(refusal.value)
    assert _state() == before

"""Every lineage insertion refuses cycles within its writing transaction."""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient

pytestmark = pytest.mark.graph


def test_native_cycle_refusal_primitive(disposable_neo4j_empty: str) -> None:
    """A conditional temporal parse raises and rolls back the whole statement."""
    from neo4j.exceptions import ClientError

    with GraphClient() as gc:
        gc.query("CREATE (:StandardName {id: 'electron_density'})")
        with pytest.raises(ClientError, match="REFINED_FROM cycle"):
            gc.query("""
                MATCH (sn:StandardName)
                SET sn.description = 'must roll back'
                FOREACH (_ IN [1] |
                  FOREACH (member IN COLLECT {
                    WITH sn AS endpoint
                    MATCH (endpoint)-[:REFINED_FROM*0..]-(member:StandardName)
                    RETURN DISTINCT member ORDER BY member.id
                  } | SET member.id = member.id)
                  FOREACH (_ IN [1] |
                    SET sn.id = datetime('REFINED_FROM cycle: ' + sn.id))
                )
                RETURN sn.id
                """)
        assert gc.query("MATCH (sn:StandardName) RETURN sn.description AS text") == [
            {"text": None}
        ]


def _seed_removed_path(*, closes_cycle: bool) -> None:
    with GraphClient() as gc:
        gc.query("""
            CREATE (old:StandardName {id: 'electron_density',
              name_stage: 'accepted', status: 'draft'})
            CREATE (holder:StandardName {id: 'electron_number_density',
              name_stage: 'accepted', status: 'draft'})
            CREATE (path:IMASNode {id: 'density/removed'})
            CREATE (successor:IMASNode {id: 'density/current'})
            CREATE (path)-[:RENAMED_TO]->(successor)
            CREATE (source:StandardNameSource {id: 'dd:density/removed',
              source_id: 'density/removed', source_type: 'dd', status: 'composed'})
            CREATE (source)-[:PRODUCED_NAME]->(old)
            CREATE (current:StandardNameSource {id: 'dd:density/current',
              source_id: 'density/current', source_type: 'dd', status: 'composed'})
            CREATE (current)-[:PRODUCED_NAME]->(holder)
            """)
        if closes_cycle:
            gc.query("""
                MATCH (old:StandardName {id: 'electron_density'}),
                      (holder:StandardName {id: 'electron_number_density'})
                CREATE (old)-[:REFINED_FROM]->(holder)
                """)


def _state() -> list[dict]:
    with GraphClient() as gc:
        return gc.query("""
            MATCH (n) OPTIONAL MATCH (n)-[edge]->(other)
            RETURN labels(n) AS labels, properties(n) AS properties,
                   type(edge) AS edge, other.id AS other
            ORDER BY n.id, edge, other
            """)


def _assert_lineage_provenance(writer: str) -> None:
    """Read the newly created edge and its database-clock creation time."""
    with GraphClient() as gc:
        rows = gc.query("""
            MATCH ()-[edge:REFINED_FROM]->()
            RETURN edge.writer AS writer,
                   edge.created_at >= datetime() - duration('PT5M')
                   AND edge.created_at <= datetime() AS recent
            """)
    assert rows == [{"writer": writer, "recent": True}]


def _stale_writer() -> int:
    from types import SimpleNamespace

    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        mark_sources_stale,
    )

    return mark_sources_stale(
        ["dd:density/removed"], services=SimpleNamespace(GraphClient=GraphClient)
    )


def test_removed_path_writer_refuses_cycle(disposable_neo4j_empty: str) -> None:
    from neo4j.exceptions import ClientError

    _seed_removed_path(closes_cycle=True)
    before = _state()
    with pytest.raises(ClientError, match="REFINED_FROM cycle"):
        _stale_writer()
    assert _state() == before


def test_removed_path_writer_writes_acyclic_edge(disposable_neo4j_empty: str) -> None:
    _seed_removed_path(closes_cycle=False)
    assert _stale_writer() == 1
    _assert_lineage_provenance("graph_source_bindings.mark_sources_stale")
    with GraphClient() as gc:
        assert gc.query("""
            MATCH (holder:StandardName {id: 'electron_number_density'})
                  -[:REFINED_FROM]->(old:StandardName {id: 'electron_density'})
            RETURN old.name_stage AS stage
            """) == [{"stage": "superseded"}]


@pytest.mark.parametrize("writer", ["source_invariant", "provenance", "compose"])
@pytest.mark.parametrize("closes_cycle", [False, True], ids=["acyclic", "cycle"])
def test_last_source_writers(
    disposable_neo4j_empty: str, writer: str, closes_cycle: bool
) -> None:
    from types import SimpleNamespace

    from neo4j.exceptions import ClientError

    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        mark_sources_composed,
    )
    from imas_codex.standard_names.provenance_lifecycle import (
        retarget_standard_name_sources,
    )
    from imas_codex.standard_names.source_invariant import (
        supersede_names_losing_last_source,
    )
    from tests.standard_names.test_last_source_supersede import NEW, OLD, _seed

    _seed(
        source_status="extracted" if writer == "compose" else "attached",
        claim_token="tok" if writer == "compose" else None,
        claim_seq=3,
    )
    with GraphClient() as gc:
        if writer == "source_invariant":
            gc.query("MATCH (s:StandardNameSource)-[r:PRODUCED_NAME]->() DELETE r")
        if closes_cycle:
            gc.query(
                """
                MATCH (old:StandardName {id: $old}), (new:StandardName {id: $new})
                CREATE (bridge:StandardName {id: 'lineage_bridge'})
                CREATE (old)-[:REFINED_FROM]->(bridge)-[:REFINED_FROM]->(new)
                """,
                old=OLD,
                new=NEW,
            )
    before = _state()

    def apply() -> None:
        with GraphClient() as gc:
            if writer == "source_invariant":
                supersede_names_losing_last_source(gc, [(OLD, NEW)])
            elif writer == "provenance":
                retarget_standard_name_sources(
                    gc,
                    OLD,
                    NEW,
                    source_ids=["dd:test/last_source/one"],
                    expected_current_bindings={"dd:test/last_source/one": OLD},
                    enforce_consistency=False,
                )
            else:
                mark_sources_composed(
                    token="tok",
                    claim_seq=3,
                    source_ids=["dd:test/last_source/one"],
                    standard_name_id=NEW,
                    services=SimpleNamespace(GraphClient=GraphClient),
                )

    if closes_cycle:
        with pytest.raises(ClientError, match="REFINED_FROM cycle"):
            apply()
        assert _state() == before
    else:
        apply()
        _assert_lineage_provenance("source_invariant.supersede_drained_name_cypher")
        with GraphClient() as gc:
            assert gc.query(
                """
                MATCH (new:StandardName {id: $new})-[:REFINED_FROM]->
                      (old:StandardName {id: $old})
                RETURN old.name_stage AS stage
                """,
                old=OLD,
                new=NEW,
            ) == [{"stage": "superseded"}]


@pytest.mark.parametrize("writer", ["refine", "edit_fold", "tombstone_fold"])
@pytest.mark.parametrize("closes_cycle", [False, True], ids=["acyclic", "cycle"])
def test_identity_writers(
    disposable_neo4j_empty: str,
    seed_dd_resolutions,
    writer: str,
    closes_cycle: bool,
) -> None:
    from imas_codex.standard_names import edit
    from imas_codex.standard_names.graph_store.graph_refinement import (
        RefinedNamePersistenceRefusal,
    )
    from imas_codex.standard_names.graph_store.graph_supersession import (
        tombstone_supersede_into,
    )
    from tests.standard_names.test_lineage_cycle_writers import (
        ANCESTOR,
        OLD,
        SOURCE_PATH,
        _refine,
        _seed,
    )

    _seed(ancestor_stage="pending" if writer == "refine" else "accepted")
    seed_dd_resolutions([{"path": SOURCE_PATH, "effective": "m^-3"}])
    if not closes_cycle:
        with GraphClient() as gc:
            gc.query("MATCH ()-[r:REFINED_FROM]->() DELETE r")
    before = _state()
    if writer == "refine":
        if closes_cycle:
            with pytest.raises(
                RefinedNamePersistenceRefusal, match="REFINED_FROM cycle"
            ):
                _refine(ANCESTOR)
        else:
            _refine(ANCESTOR)
    else:
        result = (
            edit.supersede_into(OLD, ANCESTOR)
            if writer == "edit_fold"
            else tombstone_supersede_into(OLD, ANCESTOR)
        )
        assert result["ok"] is (not closes_cycle), result
        if closes_cycle:
            assert "REFINED_FROM cycle" in result["reason"]
    if closes_cycle:
        assert _state() == before
    else:
        _assert_lineage_provenance(
            {
                "refine": "graph_refinement.persist_refined_name",
                "edit_fold": "edit.supersede_into",
                "tombstone_fold": "graph_supersession.tombstone_supersede_into",
            }[writer]
        )
        with GraphClient() as gc:
            assert gc.query(
                """
                MATCH (:StandardName {id: $new})-[:REFINED_FROM]->
                      (:StandardName {id: $old}) RETURN count(*) AS edges
                """,
                old=OLD,
                new=ANCESTOR,
            ) == [{"edges": 1}]


@pytest.mark.parametrize("closes_cycle", [False, True], ids=["acyclic", "cycle"])
def test_regeneration_writer(disposable_neo4j_empty: str, closes_cycle: bool) -> None:
    from imas_codex.standard_names.graph_store.graph_supersession import (
        supersede_prior_source_names,
    )
    from tests.standard_names.test_compose_adopts_pending_stub import (
        PROBE_PATH,
        PROBE_STUB,
        _seed_finalized_successor,
    )

    with GraphClient() as gc:
        _seed_finalized_successor(gc)
        if closes_cycle:
            gc.query(
                """
                MATCH (old:StandardName {id: $old}),
                      (new:StandardName {id: 'ion_temperature'})
                CREATE (old)-[:REFINED_FROM]->(new)
                """,
                old=PROBE_STUB,
            )
    before = _state()

    def apply() -> int:
        with GraphClient() as gc:
            return supersede_prior_source_names(
                [{"new_name": "ion_temperature", "source_id": PROBE_PATH}], gc=gc
            )

    if closes_cycle:
        assert apply() == 0
        assert _state() == before
    else:
        assert apply() == 1
        _assert_lineage_provenance("graph_supersession.supersede_prior_source_names")
        with GraphClient() as gc:
            assert gc.query(
                """
                MATCH (:StandardName {id: 'ion_temperature'})-[:REFINED_FROM]->
                      (:StandardName {id: $old}) RETURN count(*) AS edges
                """,
                old=PROBE_STUB,
            ) == [{"edges": 1}]


@pytest.mark.parametrize("closes_cycle", [False, True], ids=["acyclic", "cycle"])
def test_reclaim_mutation_boundary(
    disposable_neo4j_empty: str, closes_cycle: bool
) -> None:
    """The mutation rechecks lineage even after a caller has preflighted it."""
    from neo4j.exceptions import ClientError

    from imas_codex.standard_names.edit import _RECLAIM_MUTATION_QUERY

    with GraphClient() as gc:
        gc.query("""
            CREATE (old:StandardName {id: 'electron_density', name_stage: 'accepted'})
            CREATE (tomb:StandardName {id: 'electron_number_density',
              name_stage: 'superseded'})
            """)
        if closes_cycle:
            gc.query("""
                MATCH (old:StandardName {id: 'electron_density'}),
                      (tomb:StandardName {id: 'electron_number_density'})
                CREATE (old)-[:REFINED_FROM]->(tomb)
                """)
        before = _state()
        params = {
            "old_id": "electron_density",
            "new_id": "electron_number_density",
            "old_stage": "accepted",
            "revived_stage": "reviewed",
            "description": "Electron number density",
            "unit": "m^-3",
            "reason": "restore the spelling",
            "origin": "user",
            "scope": "edits",
            "run_id": "reclaim-control",
            "change_id": "reclaim-receipt",
            "receipt": "{}",
            "changed_at": "2026-11-10T17:00:00+00:00",
            "link_lineage": True,
        }
        if closes_cycle:
            with pytest.raises(ClientError, match="REFINED_FROM cycle"):
                gc.query(_RECLAIM_MUTATION_QUERY, **params)
            assert _state() == before
        else:
            assert gc.query(_RECLAIM_MUTATION_QUERY, **params) == [
                {"revived_stage": "reviewed", "retired_stage": "superseded"}
            ]
            _assert_lineage_provenance("edit._apply_reclaim_rename")
            assert gc.query(
                """
                MATCH (:StandardName {id: $new_id})-[:REFINED_FROM]->
                      (:StandardName {id: $old_id}) RETURN count(*) AS edges
                """,
                old_id=params["old_id"],
                new_id=params["new_id"],
            ) == [{"edges": 1}]


def test_shared_guard_refuses_self_loop(disposable_neo4j_empty: str) -> None:
    from neo4j.exceptions import ClientError

    from imas_codex.standard_names.graph_store.graph_foundation import (
        refined_from_cycle_guard_cypher,
    )

    with GraphClient() as gc:
        gc.query("CREATE (:StandardName {id: 'electron_density'})")
        with pytest.raises(ClientError, match="REFINED_FROM cycle"):
            gc.query(
                "MATCH (sn:StandardName) "
                + refined_from_cycle_guard_cypher("sn", "sn")
                + " MERGE (sn)-[:REFINED_FROM]->(sn)"
            )
        assert gc.query("MATCH ()-[r:REFINED_FROM]->() RETURN count(r) AS edges") == [
            {"edges": 0}
        ]


def test_concurrent_disjoint_edges_cannot_close_a_cycle(
    disposable_neo4j_empty: str,
) -> None:
    """Writers with different endpoints share locks on their connected names."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from neo4j.exceptions import ClientError

    from imas_codex.standard_names.graph_store.graph_foundation import (
        refined_from_cycle_guard_cypher,
    )

    with GraphClient() as gc:
        gc.query("""
            CREATE (a:StandardName {id: 'lineage_a'})
            CREATE (b:StandardName {id: 'lineage_b'})
            CREATE (c:StandardName {id: 'lineage_c'})
            CREATE (d:StandardName {id: 'lineage_d'})
            CREATE (a)-[:REFINED_FROM]->(b)
            CREATE (c)-[:REFINED_FROM]->(d)
            """)
    barrier = Barrier(2, timeout=10)

    def insert(pair: tuple[str, str]) -> str:
        with GraphClient() as gc:
            barrier.wait()
            try:
                statement = (
                    "MATCH (successor:StandardName {id: $successor}), "
                    "(predecessor:StandardName {id: $predecessor}) "
                    + refined_from_cycle_guard_cypher("successor", "predecessor")
                    + " MERGE (successor)-[:REFINED_FROM]->(predecessor)"
                )
                with gc.session() as session:
                    session.execute_write(
                        lambda tx: list(
                            tx.run(statement, successor=pair[0], predecessor=pair[1])
                        )
                    )
            except ClientError as exc:
                assert "REFINED_FROM cycle" in str(exc)
                return "refused"
            return "written"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(
            pool.map(insert, [("lineage_b", "lineage_c"), ("lineage_d", "lineage_a")])
        ) == ["refused", "written"]
    with GraphClient() as gc:
        assert gc.query("""
            MATCH (sn:StandardName) WHERE EXISTS { (sn)-[:REFINED_FROM*1..]->(sn) }
            RETURN count(sn) AS cyclic
            """) == [{"cyclic": 0}]


def test_batch_source_retirements_cannot_cycle_each_other(
    disposable_neo4j_empty: str,
) -> None:
    """Edges proposed together must see earlier edges in the same statement."""
    from neo4j.exceptions import ClientError

    from imas_codex.standard_names.source_invariant import (
        supersede_names_losing_last_source,
    )

    with GraphClient() as gc:
        gc.query("""
            CREATE (:StandardName {id: 'electron_density', name_stage: 'drafted'})
            CREATE (:StandardName {id: 'electron_number_density', name_stage: 'drafted'})
            """)
        before = _state()
        with pytest.raises(ClientError, match="REFINED_FROM cycle"):
            supersede_names_losing_last_source(
                gc,
                [
                    ("electron_density", "electron_number_density"),
                    ("electron_number_density", "electron_density"),
                ],
            )
        assert _state() == before

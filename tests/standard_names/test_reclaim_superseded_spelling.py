"""A rename reclaims a tombstoned spelling in exactly the two ruled cases.

``sn edit --rename`` refused every superseded id, so a name could not move
back to a spelling an earlier rename had taken from it. These tests drive the
sanctioned rename writer on the disposable graph and read the resulting
lifecycle back, so a route that revives a stranger's history — or refuses the
two cases the ruling admits — fails here instead of in the catalog.
"""

from __future__ import annotations

import json

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.edit import apply_edit

pytestmark = pytest.mark.graph

TARGET = "electron_temperature_at_separatrix"
TOMBSTONE = "electron_temperature_at_plasma_boundary"
INTERMEDIATE = "electron_temperature_number_at_plasma_boundary"
UNRELATED = "ion_temperature_at_plasma_boundary"

VALUE_PATH = "summary/local/separatrix/electron_temperature"
AVERAGE_PATH = "summary/local/separatrix_average/electron_temperature"


def _reset() -> None:
    with GraphClient() as gc:
        gc.query("MATCH (n) DETACH DELETE n")


def _name(
    gc: GraphClient,
    name_id: str,
    *,
    stage: str,
    source_paths: list[str] | None = None,
    documentation: str | None = None,
    score: float | None = None,
) -> None:
    gc.query(
        """
        CREATE (sn:StandardName {
          id: $id, name_stage: $stage,
          status: CASE WHEN $stage = 'superseded' THEN 'superseded' ELSE 'draft' END,
          origin: 'pipeline', description: 'a description',
          documentation: $documentation, docs_stage: 'accepted',
          reviewer_score_name: $score,
          superseded_from_stage: CASE WHEN $stage = 'superseded' THEN 'accepted' ELSE null END,
          source_paths: $paths})
        """,
        id=name_id,
        stage=stage,
        documentation=documentation,
        score=score,
        paths=source_paths or [],
    )


def _lineage(gc: GraphClient, successor: str, predecessor: str) -> None:
    gc.query(
        """
        MATCH (succ:StandardName {id: $succ}), (pred:StandardName {id: $pred})
        CREATE (succ)-[:REFINED_FROM]->(pred)
        """,
        succ=successor,
        pred=predecessor,
    )


def _source(gc: GraphClient, sn_id: str, path: str, status: str = "attached") -> None:
    gc.query(
        """
        MATCH (sn:StandardName {id: $sn_id})
        CREATE (source:StandardNameSource {
          id: $source_id, source_type: 'dd', source_id: $path,
          status: $status, produced_sn_id: $sn_id})
        CREATE (source)-[:PRODUCED_NAME]->(sn)
        CREATE (dd:IMASNode {id: $path})
        CREATE (source)-[:FROM_DD_PATH]->(dd)
        """,
        sn_id=sn_id,
        source_id=f"dd:{path}",
        path=path,
        status=status,
    )


def _read(name_id: str) -> dict:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            RETURN sn.name_stage AS name_stage, sn.status AS status,
                   sn.documentation AS documentation,
                   sn.docs_stage AS docs_stage,
                   sn.superseded_from_stage AS superseded_from_stage,
                   sn.run_id AS run_id,
                   [(succ:StandardName)-[:REFINED_FROM]->(sn) | succ.id]
                     AS successors,
                   [source IN collect(source) WHERE source IS NOT NULL
                     | source.id] AS sources
            """,
            id=name_id,
        )
    assert rows, f"{name_id} is missing from the graph"
    return dict(rows[0])


def _change(old_id: str, new_id: str) -> dict:
    """The one reclaim receipt the ledger holds for this retired/revived pair."""
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (change:StandardNameChange {
              from_name: $old, to_name: $new, operation: 'reclaim_spelling'
            })
            RETURN change.operation AS operation, change.reason AS reason,
                   change.run_id AS run_id
            """,
            old=old_id,
            new=new_id,
        )
    assert len(rows) == 1, f"expected one change receipt, found {len(rows)}"
    receipt = dict(rows[0])
    receipt["reason"] = json.loads(receipt["reason"])
    return receipt


def test_closing_lineage_reclaims_the_tombstone_with_its_history():
    """The tombstone's lineage ends on the renaming name: the id comes back."""
    _reset()
    with GraphClient() as gc:
        _name(gc, TARGET, stage="accepted", source_paths=[VALUE_PATH])
        _name(gc, TOMBSTONE, stage="superseded", source_paths=[VALUE_PATH])
        _name(gc, INTERMEDIATE, stage="superseded", source_paths=[VALUE_PATH])
        _lineage(gc, INTERMEDIATE, TOMBSTONE)
        _lineage(gc, TARGET, INTERMEDIATE)
        # The tombstone keeps its paid review and its prose.
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            SET sn.documentation = 'the tombstone prose',
                sn.reviewer_score_name = 0.93
            """,
            id=TOMBSTONE,
        )
        _source(gc, TARGET, VALUE_PATH)

    plan = apply_edit(
        target=TARGET,
        rename=TOMBSTONE,
        reason="restore the ruled spelling after a reversed rename",
        gc=GraphClient(),
    )
    assert plan.blocked is None, plan.blocked
    assert plan.applied is True
    assert plan.successor == TOMBSTONE

    revived = _read(TOMBSTONE)
    assert revived["name_stage"] != "superseded"
    assert revived["documentation"] == "the tombstone prose"
    assert revived["docs_stage"] == "accepted"
    assert revived["sources"] == [f"dd:{VALUE_PATH}"]

    retired = _read(TARGET)
    assert retired["name_stage"] == "superseded"
    assert TOMBSTONE in retired["successors"]
    assert retired["sources"] == []

    receipt = _change(TARGET, TOMBSTONE)
    assert receipt["operation"] == "reclaim_spelling"
    assert receipt["run_id"] == plan.run_id
    assert receipt["reason"]["case"] == "closing_lineage"
    assert receipt["reason"]["from_name"] == TARGET
    assert receipt["reason"]["to_name"] == TOMBSTONE
    assert receipt["reason"]["prior_stage"] == "accepted"
    # The retired name records the stage it held, not the 'superseded' the
    # retire wrote
    assert retired["superseded_from_stage"] == "accepted"


def test_different_locus_holder_reclaims_the_tombstone():
    """The holder names another DD locus, so the spelling still names this one."""
    _reset()
    with GraphClient() as gc:
        _name(gc, TARGET, stage="accepted", source_paths=[VALUE_PATH])
        _name(gc, TOMBSTONE, stage="superseded", source_paths=[VALUE_PATH])
        _name(gc, UNRELATED, stage="accepted", source_paths=[AVERAGE_PATH])
        _lineage(gc, UNRELATED, TOMBSTONE)
        _source(gc, TARGET, VALUE_PATH)
        _source(gc, UNRELATED, AVERAGE_PATH)

    plan = apply_edit(
        target=TARGET,
        rename=TOMBSTONE,
        reason="take back the value-locus spelling from the average holder",
        gc=GraphClient(),
    )
    assert plan.blocked is None, plan.blocked
    assert plan.applied is True

    revived = _read(TOMBSTONE)
    assert revived["name_stage"] != "superseded"
    assert revived["sources"] == [f"dd:{VALUE_PATH}"]
    # The average holder keeps its own id and its own source.
    holder = _read(UNRELATED)
    assert holder["name_stage"] == "accepted"
    assert holder["sources"] == [f"dd:{AVERAGE_PATH}"]

    receipt = _change(TARGET, TOMBSTONE)
    assert receipt["operation"] == "reclaim_spelling"
    assert receipt["run_id"] == plan.run_id
    assert receipt["reason"]["case"] == "different_locus"
    assert receipt["reason"]["prior_stage"] == "accepted"
    assert _read(TARGET)["superseded_from_stage"] == "accepted"


def test_unrelated_tombstone_lineage_is_still_refused():
    """A tombstone whose lineage ends on another live name stays protected."""
    _reset()
    with GraphClient() as gc:
        _name(gc, TARGET, stage="accepted", source_paths=[VALUE_PATH])
        _name(gc, TOMBSTONE, stage="superseded", source_paths=[VALUE_PATH])
        _name(gc, UNRELATED, stage="accepted", source_paths=[VALUE_PATH])
        _lineage(gc, UNRELATED, TOMBSTONE)
        _source(gc, TARGET, VALUE_PATH)
        _source(gc, UNRELATED, VALUE_PATH)

    plan = apply_edit(
        target=TARGET,
        rename=TOMBSTONE,
        reason="this spelling belongs to an unrelated live name",
        gc=GraphClient(),
    )
    assert plan.applied is False
    assert plan.blocked is not None
    assert TOMBSTONE in (plan.blocked or "")

    tomb = _read(TOMBSTONE)
    assert tomb["name_stage"] == "superseded"
    assert tomb["run_id"] is None
    with GraphClient() as gc:
        receipts = gc.query(
            "MATCH (change:StandardNameChange) RETURN count(change) AS n"
        )
    assert receipts[0]["n"] == 0, "a refused reclaim must write no receipt"

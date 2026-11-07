"""A reclaim moves a tombstone's history to an archive instead of reviving it.

A tombstone can carry history recorded for a quantity a live successor already
holds — an average, say, whose spelling a value name now needs. Reviving that
tombstone would hang the average's reviews and prose on the value name, so the
reclaim archives the history beside its holder and renames the live name onto
the freed id. These tests drive the sanctioned rename writer on the disposable
graph and read both identities back, so a route that inherits the archived
quantity's rows — or admits a tombstone with no live holder at its locus —
fails here instead of in the catalog.
"""

from __future__ import annotations

import json

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.edit import apply_edit

pytestmark = pytest.mark.graph

TARGET = "toroidal_tritium_velocity_at_separatrix"
TOMBSTONE = "toroidal_tritium_velocity_at_plasma_boundary"
HOLDER = "flux_surface_averaged_toroidal_tritium_velocity_at_plasma_boundary"

VALUE_PATH = "summary/local/separatrix/velocity_phi/tritium/value"
AVERAGE_PATH = "summary/local/separatrix_average/velocity_phi/tritium/value"

TOMBSTONE_PROSE = "the flux-surface-averaged toroidal tritium velocity"
TARGET_PROSE = "the toroidal tritium velocity at the separatrix"


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
) -> None:
    gc.query(
        """
        CREATE (sn:StandardName {
          id: $id, name_stage: $stage,
          status: CASE WHEN $stage = 'superseded' THEN 'superseded' ELSE 'draft' END,
          origin: 'pipeline', description: 'a description',
          documentation: $documentation, docs_stage: 'accepted',
          superseded_from_stage: CASE WHEN $stage = 'superseded' THEN 'accepted' ELSE null END,
          source_paths: $paths})
        """,
        id=name_id,
        stage=stage,
        documentation=documentation,
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


def _review(gc: GraphClient, sn_id: str, review_id: str) -> None:
    gc.query(
        """
        MATCH (sn:StandardName {id: $sn_id})
        CREATE (review:StandardNameReview {id: $review_id, reviewer_score: 0.91})
        CREATE (sn)-[:HAS_REVIEW]->(review)
        """,
        sn_id=sn_id,
        review_id=review_id,
    )


def _source(gc: GraphClient, sn_id: str, path: str) -> None:
    gc.query(
        """
        MATCH (sn:StandardName {id: $sn_id})
        CREATE (source:StandardNameSource {
          id: $source_id, source_type: 'dd', source_id: $path,
          status: 'attached', produced_sn_id: $sn_id})
        CREATE (source)-[:PRODUCED_NAME]->(sn)
        CREATE (dd:IMASNode {id: $path, unit: 'm/s'})
        CREATE (source)-[:FROM_DD_PATH]->(dd)
        """,
        sn_id=sn_id,
        source_id=f"dd:{path}",
        path=path,
    )


def _read(name_id: str) -> dict:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            RETURN sn.name_stage AS name_stage, sn.status AS status,
                   sn.documentation AS documentation,
                   [review IN collect(DISTINCT review) WHERE review IS NOT NULL
                     | review.id] AS reviews,
                   [source IN collect(DISTINCT source) WHERE source IS NOT NULL
                     | source.id] AS sources
            """,
            id=name_id,
        )
    assert rows, f"{name_id} is missing from the graph"
    return dict(rows[0])


def _archive_for(tombstone_id: str) -> dict:
    """The archived identity holding the tombstone's history, if one exists."""
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (archived:ArchivedStandardName {archived_from: $tombstone_id})
            OPTIONAL MATCH (holder:StandardName)-[:HAS_ARCHIVED_HISTORY]->(archived)
            OPTIONAL MATCH (archived)-[:HAS_REVIEW]->(review:StandardNameReview)
            OPTIONAL MATCH (succ:StandardName)-[:REFINED_FROM]->(archived)
            RETURN archived.id AS id,
                   archived.name_stage AS name_stage,
                   archived.status AS status,
                   archived.documentation AS documentation,
                   holder.id AS holder,
                   [review IN collect(DISTINCT review) WHERE review IS NOT NULL
                     | review.id] AS reviews,
                   [succ IN collect(DISTINCT succ) WHERE succ IS NOT NULL
                     | succ.id] AS lineage
            """,
            tombstone_id=tombstone_id,
        )
    assert len(rows) == 1, f"expected one archive for {tombstone_id}, found {len(rows)}"
    return dict(rows[0])


def _receipt(tombstone_id: str) -> dict:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (change:StandardNameChange {
              from_name: $name, operation: 'archive_reclaim_history'
            })
            RETURN change.to_name AS archive_id, change.reason AS reason,
                   change.run_id AS run_id
            """,
            name=tombstone_id,
        )
    assert len(rows) == 1, f"expected one archive receipt, found {len(rows)}"
    receipt = dict(rows[0])
    receipt["reason"] = json.loads(receipt["reason"])
    return receipt


def _seed_average_case(gc: GraphClient) -> None:
    """A value name wants the spelling a folded average's tombstone holds."""
    _name(
        gc,
        TARGET,
        stage="accepted",
        source_paths=[VALUE_PATH],
        documentation=TARGET_PROSE,
    )
    _name(
        gc,
        TOMBSTONE,
        stage="superseded",
        source_paths=[AVERAGE_PATH],
        documentation=TOMBSTONE_PROSE,
    )
    _name(gc, HOLDER, stage="accepted", source_paths=[AVERAGE_PATH])
    _lineage(gc, HOLDER, TOMBSTONE)
    _review(gc, TOMBSTONE, "review:average:1")
    _source(gc, TARGET, VALUE_PATH)
    _source(gc, HOLDER, AVERAGE_PATH)


def test_history_apart_reclaims_the_spelling_and_archives_the_history(
    seed_dd_resolutions,
):
    """The holder keeps the average; the value name takes the freed spelling."""
    _reset()
    seed_dd_resolutions([{"path": VALUE_PATH, "effective": "m/s"}])
    with GraphClient() as gc:
        _seed_average_case(gc)

    plan = apply_edit(
        target=TARGET,
        rename=TOMBSTONE,
        reason="the value name needs the ruled spelling; the average keeps it",
        gc=GraphClient(),
    )
    assert plan.blocked is None, plan.blocked
    assert plan.applied is True
    assert plan.successor == TOMBSTONE

    # The live name on the reclaimed spelling carries none of the average's
    # history — no review row, no prose — and carries this name's source.
    live = _read(TOMBSTONE)
    assert live["name_stage"] != "superseded"
    assert live["reviews"] == []
    assert live["documentation"] != TOMBSTONE_PROSE
    assert live["sources"] == [f"dd:{VALUE_PATH}"]

    # The average's history moved to an archived identity beside its holder.
    archive = _archive_for(TOMBSTONE)
    assert archive["holder"] == HOLDER
    assert archive["reviews"] == ["review:average:1"]
    assert archive["documentation"] == TOMBSTONE_PROSE
    assert HOLDER in archive["lineage"]

    receipt = _receipt(TOMBSTONE)
    assert receipt["run_id"] == plan.run_id
    assert receipt["archive_id"] == archive["id"]
    assert receipt["reason"]["case"] == "history_apart"
    assert receipt["reason"]["holder"] == HOLDER
    assert receipt["reason"]["reclaiming_name"] == TARGET

    # The holder is untouched: its id, its stage and its own source.
    holder = _read(HOLDER)
    assert holder["name_stage"] == "accepted"
    assert holder["sources"] == [f"dd:{AVERAGE_PATH}"]

    # The reclaiming spelling is retired as an ordinary rename retires it.
    retired = _read(TARGET)
    assert retired["name_stage"] == "superseded"
    assert TOMBSTONE in _successors(TARGET)


def _successors(name_id: str) -> list[str]:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (succ:StandardName)-[:REFINED_FROM]->(sn:StandardName {id: $id})
            RETURN [succ IN collect(succ) | succ.id] AS successors
            """,
            id=name_id,
        )
    return list(rows[0]["successors"]) if rows else []


def test_a_tombstone_with_no_live_holder_at_its_locus_is_refused():
    """Without a live successor holding the quantity, the id stays protected."""
    _reset()
    with GraphClient() as gc:
        _name(gc, TARGET, stage="accepted", source_paths=[VALUE_PATH])
        _name(gc, TOMBSTONE, stage="superseded", source_paths=[AVERAGE_PATH])
        _source(gc, TARGET, VALUE_PATH)

    plan = apply_edit(
        target=TARGET,
        rename=TOMBSTONE,
        reason="no live name holds the quantity this tombstone recorded",
        gc=GraphClient(),
    )
    assert plan.applied is False
    assert plan.blocked is not None
    assert TOMBSTONE in (plan.blocked or "")

    tomb = _read(TOMBSTONE)
    assert tomb["name_stage"] == "superseded"
    with GraphClient() as gc:
        archives = gc.query(
            "MATCH (archived:ArchivedStandardName) RETURN count(archived) AS n"
        )
        receipts = gc.query(
            "MATCH (change:StandardNameChange) RETURN count(change) AS n"
        )
    assert archives[0]["n"] == 0, "a refused reclaim must archive nothing"
    assert receipts[0]["n"] == 0, "a refused reclaim must write no receipt"

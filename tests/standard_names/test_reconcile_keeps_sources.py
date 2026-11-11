"""A startup reconcile must not leave a live name without a producing source.

``reconcile_descriptionless_composed_names`` releases a composed source's
``PRODUCED_NAME`` edge so compose can rewrite a missing description. When the
name's spelling is one the strict writer refuses, compose can never rewrite it:
the release is final, and the name stays live at a draft stage with no producer.
That is a sourceless live name, which is exactly what the writer invariant
exists to prevent.

These tests drive the reconcile on the disposable graph and read the lifecycle
back. The unrecomposable case must leave the name either still produced or
retired; the recomposable case must keep the documented re-open behaviour.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_store.graph_audits import (
    reconcile_descriptionless_composed_names,
)
from imas_codex.standard_names.source_invariant import find_sourceless_live_names

pytestmark = pytest.mark.graph

#: A spelling the strict writer refuses: its residue matches no physical base,
#: so no compose pass can ever mint it again.
UNRECOMPOSABLE = "test_reconcile_zorble_width_of_fiber_bundle"

#: A spelling the strict writer accepts, so a re-open here is recoverable.
RECOMPOSABLE = "electron_temperature"


def _seed(name_id: str) -> None:
    """Create a descriptionless live draft produced by one composed DD source."""
    dd_path = "test/reconcile/" + name_id.replace("_", "/")
    with GraphClient() as gc:
        gc.query("MATCH (n) DETACH DELETE n")
        gc.query(
            """
            CREATE (sn:StandardName {
              id: $name, name_stage: 'drafted', status: 'draft',
              origin: 'pipeline', description: null})
            CREATE (dd:IMASNode {id: $path})
            CREATE (src:StandardNameSource {
              id: 'dd:' + $path, source_type: 'dd', source_id: $path,
              status: 'composed', produced_sn_id: $name})
            CREATE (src)-[:PRODUCED_NAME]->(sn)
            CREATE (src)-[:FROM_DD_PATH]->(dd)
            CREATE (dd)-[:HAS_STANDARD_NAME]->(sn)
            """,
            name=name_id,
            path=dd_path,
        )


def _state(name_id: str) -> dict:
    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName {id: $name})
                OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                RETURN sn.name_stage AS stage, sn.status AS status,
                       [s IN collect(DISTINCT src) WHERE s IS NOT NULL | s.id]
                         AS producers,
                       [(sn)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange) |
                         c.operation] AS receipts
                """,
                name=name_id,
            )
        )
    assert len(rows) == 1, f"{name_id} must survive the reconcile"
    return rows[0]


def test_unrecomposable_name_is_not_left_sourceless(disposable_neo4j_uri: str) -> None:
    """The reconcile must leave a name compose cannot re-mint with its source.

    Releasing the producer is the only way the repair can help a name compose
    may write again; on a spelling the strict writer rejects the release is
    final, so the repair must decline it and leave the name with the source it
    has rather than trading a stuck name for a sourceless one.
    """
    _seed(UNRECOMPOSABLE)
    with GraphClient() as gc:
        reconcile_descriptionless_composed_names(gc)
        sourceless = find_sourceless_live_names(gc, [UNRECOMPOSABLE])
        state = _state(UNRECOMPOSABLE)

    assert sourceless == [], (
        "the reconcile released the last producer of a name compose cannot "
        f"re-mint and left it live: {state}"
    )
    assert state["stage"] == "drafted", state
    assert state["producers"] == [
        "dd:test/reconcile/test/reconcile/zorble/width/of/fiber/bundle"
    ], state


def test_recomposable_name_is_reopened(disposable_neo4j_uri: str) -> None:
    """The documented re-open survives: compose can still rewrite the prose."""
    _seed(RECOMPOSABLE)
    with GraphClient() as gc:
        reconcile_descriptionless_composed_names(gc)
        state = _state(RECOMPOSABLE)
        rows = list(
            gc.query(
                "MATCH (s:StandardNameSource {id: 'dd:test/reconcile/electron/"
                "temperature'}) RETURN s.status AS status"
            )
        )

    assert state["stage"] == "pending", state
    assert state["producers"] == [], state
    assert [row["status"] for row in rows] == ["extracted"], rows

"""The supersession preflight, run against a live graph.

The rest of the supersession suite drives ``supersede_prior_source_names``
through a mocked client, so the preflight statement itself is only ever
pattern-matched as text — it is never parsed or executed, and its
classification is never applied to real topology. That leaves two whole
classes of defect invisible: a statement that does not run at all, and one
that runs but resolves the wrong number of rows because live data has a shape
no hand-written fixture anticipated.

Both matter here, because the preflight's Python caller treats an unresolvable
recomposed source as a hard error that rolls back the entire generate batch.
Anything that multiplies or empties the trigger row does not merely misclassify
one pairing — it destroys a whole pass of composed names.

Read-only: the statement is a MATCH/RETURN and nothing here writes.

Marked ``graph`` — only runs when a live Neo4j is available (auto-skipped
otherwise via the top-level conftest hook).
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import (
    _GENERATED_SUPERSESSION_PREFLIGHT_QUERY,
)

pytestmark = pytest.mark.graph

# Every DD path that more than one source anchors, paired with a name one of
# those sources actually composed. This is the topology that defeats resolving
# the recomposed source by its DD edge alone.
_SHARED_ANCHOR_PAIRS = """
MATCH (source:StandardNameSource)-[:FROM_DD_PATH]->(dd:IMASNode)
WITH dd, collect(DISTINCT source) AS anchored
WHERE size(anchored) > 1
UNWIND anchored AS source
MATCH (source)-[:PRODUCED_NAME]->(name:StandardName)
RETURN DISTINCT dd.id AS source_id, name.id AS new_name,
       [s IN anchored | s.id] AS anchored_source_ids
"""

_LIVE_PAIRS = """
MATCH (source:StandardNameSource)-[:FROM_DD_PATH]->(dd:IMASNode)
MATCH (source)-[:PRODUCED_NAME]->(name:StandardName)
RETURN DISTINCT dd.id AS source_id, name.id AS new_name
LIMIT $limit
"""

_EXPECTED_KEYS = {
    "requested_source_id",
    "new_name",
    "requested_source_exists",
    "successor_exists",
    "trigger_source_id",
    "old_name",
    "old_stage",
    "judged_source_ids",
    "retained_source_ids",
}


def _preflight(gc: GraphClient, pairs: list[dict[str, str]]) -> list[dict]:
    return list(gc.query(_GENERATED_SUPERSESSION_PREFLIGHT_QUERY, pairs=pairs) or [])


def _aborts(row: dict) -> bool:
    """The caller's fail-closed condition, mirrored.

    ``supersede_prior_source_names`` raises when the recomposed source cannot
    be found in the judged set, which rolls back every name the batch composed.
    A row with no predecessor is classified against nothing and never reaches
    the check, so it is excluded here exactly as the caller excludes it.
    """
    if not row.get("old_name"):
        return False
    trigger = row.get("trigger_source_id")
    return not trigger or trigger not in set(row.get("judged_source_ids") or [])


def test_preflight_statement_executes_and_returns_its_contract() -> None:
    """It parses, runs, and answers with the keys the caller unpacks."""
    with GraphClient() as gc:
        pairs = [
            {"new_name": row["new_name"], "source_id": row["source_id"]}
            for row in gc.query(_LIVE_PAIRS, limit=25)
        ]
        if not pairs:
            pytest.skip("no composed DD sources in this graph")
        rows = _preflight(gc, pairs)

    assert rows, "preflight returned nothing for live composed sources"
    for row in rows:
        assert set(row) == _EXPECTED_KEYS
    assert all(row["requested_source_exists"] for row in rows)
    assert all(row["successor_exists"] for row in rows)


def test_a_dd_path_anchoring_several_sources_yields_one_recomposed_source() -> None:
    """A DD rename leaves the superseded spelling anchored on the same node as
    the current one. Both reach the predecessor; only the one this pass bound
    to the successor is the recomposed source, and offering the other alongside
    it aborts a batch that should have run."""
    with GraphClient() as gc:
        cases = list(gc.query(_SHARED_ANCHOR_PAIRS) or [])
        if not cases:
            pytest.skip("no DD path in this graph anchors more than one source")
        results = [
            (
                case,
                _preflight(
                    gc,
                    [{"new_name": case["new_name"], "source_id": case["source_id"]}],
                ),
            )
            for case in cases
        ]

    for case, rows in results:
        triggers = {row["trigger_source_id"] for row in rows}
        assert None not in triggers, (
            f"{case['source_id']} anchors {case['anchored_source_ids']} and none "
            f"resolved as the source recomposed onto {case['new_name']}"
        )
        # Every resolved trigger is one of the sources anchored on that path,
        # and is bound to the successor rather than merely sharing the anchor.
        assert triggers <= set(case["anchored_source_ids"])
        assert not [row for row in rows if _aborts(row)], (
            f"{case['source_id']} -> {case['new_name']} would roll back the "
            f"generate batch that composed it"
        )


def test_no_live_binding_would_roll_back_its_own_generate_batch() -> None:
    """Sweep the live corpus: replaying each existing binding as the pass that
    produced it must never hit the fail-closed path. A hit means real topology
    the preflight cannot resolve, and every name in that batch is lost."""
    with GraphClient() as gc:
        pairs = [
            {"new_name": row["new_name"], "source_id": row["source_id"]}
            for row in gc.query(_LIVE_PAIRS, limit=2000)
        ]
        if not pairs:
            pytest.skip("no composed DD sources in this graph")
        aborting = [
            row
            for start in range(0, len(pairs), 200)
            for row in _preflight(gc, pairs[start : start + 200])
            if _aborts(row)
        ]

    assert not aborting, [
        (row["requested_source_id"], row["new_name"], row["trigger_source_id"])
        for row in aborting[:10]
    ]

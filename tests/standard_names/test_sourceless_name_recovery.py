"""A live pipeline name left with no producing source is healed or surfaced.

A consistency detach strips a source whose pairing the compose guard once
rejected, returns the freed source to ``'extracted'`` and records the event as
a ``'detach_inconsistent_attachment'`` ``StandardNameChange`` — but leaves the
name itself live with no producing source. When the detachment reason was
transient (the guard judged the pairing under an older grammar), nothing ever
re-pairs the two, because the freed source sits at ``'extracted'`` where no
pool revisits an already live name.

:func:`reconcile_sourceless_pipeline_names` closes that gap. It reads each
live pipeline-origin name's recorded detachment, re-asks the compose guard
under the CURRENT grammar, and reattaches the source when the reason no longer
holds, while a name whose reason still holds is surfaced as a blocker and left
alone. It runs from :func:`reconcile_reviewable_name_stage`, which is already
wired into the ``run_sn_pools`` startup reconcile.

The fixtures are synthetic and strictly prefix-isolated so they never collide
with — or mutate — real graph nodes: the semantic source path and the name id
both embed ``_PREFIX``, so ``_clean`` wipes every node this module creates and
``_gc`` never touches a real ``StandardName`` / ``StandardNameSource``. The
shared live graph may carry its own unrelated sourceless names (the census that
drove this reconcile counts them), so assertions are scoped to the created
nodes and to tolerant before/after census deltas rather than to absolute
graph-wide totals.
"""

from __future__ import annotations

import uuid

import pytest

_PREFIX = "test_reconcile_sourceless__"

#: A DD-path-like base-quantity string the guard reads as consistent with a
#: base name and inconsistent with a change/rate name. Fully synthetic.
_LAPSED_PATH = f"{_PREFIX}FAKE/electron/temperature"
#: A name the guard ACCEPTS against ``_LAPSED_PATH`` — a base quantity.
_LAPSED_KIND = "electron_temperature"
#: A name the guard REJECTS against ``_LAPSED_PATH`` — a change/rate.
_HELD_KIND = "change_in_electron_temperature"


@pytest.fixture()
def _gc():
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"Neo4j not available: {exc}")
    yield client
    client.close()


@pytest.fixture()
def _clean(_gc):
    def _wipe() -> None:
        for label in ("StandardName", "StandardNameSource", "StandardNameChange"):
            _gc.query(
                f"MATCH (n:{label}) WHERE n.id CONTAINS $p DETACH DELETE n",
                p=_PREFIX,
            )

    _wipe()
    yield
    _wipe()


def _brokered(_gc, *, name_stage: str, name_kind: str) -> tuple[str, str]:
    """Create a sourceless live name + a detached 'extracted' source + record.

    Mirrors the state a consistency detach leaves behind for a name whose
    source path maps one-to-one onto a ``dd:``-namespaced source node: the name
    carries no ``PRODUCED_NAME`` source and no ``source_paths`` entry, the
    source sits at ``'extracted'`` with no produced identity, and the
    detachment is recorded as a ``StandardNameChange`` linked from the name.

    The name id embeds *name_kind* so the compose guard parses it with the
    intended semantics. Returns ``(name_id, source_id)``.
    """
    name_id = f"{_PREFIX}{name_kind}_n_{uuid.uuid4().hex[:6]}"
    source_id = f"dd:{_LAPSED_PATH}"
    _gc.query(
        """
        MERGE (sn:StandardName {id: $name_id})
        SET sn.name_stage = $name_stage,
            sn.origin = 'pipeline',
            sn.source_paths = []
        MERGE (sns:StandardNameSource {id: $source_id})
        SET sns.source_id = $dd_path,
            sns.source_type = 'dd',
            sns.status = 'extracted',
            sns.attempt_count = 0,
            sns.produced_sn_id = null,
            sns.claimed_at = null,
            sns.claim_token = null
        MERGE (ch:StandardNameChange {id: $change_id})
        SET ch.from_name = $dd_path,
            ch.to_name = $name_id,
            ch.operation = 'detach_inconsistent_attachment',
            ch.reason = 'recorded rejection under an older grammar',
            ch.internal = true
        MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(ch)
        """,
        name_id=name_id,
        source_id=source_id,
        dd_path=_LAPSED_PATH,
        change_id=f"sn-change:{_PREFIX}rec_{uuid.uuid4().hex[:6]}",
        name_stage=name_stage,
    )
    return name_id, source_id


def _name(_gc, name_id: str) -> dict:
    rows = _gc.query(
        """
        MATCH (sn:StandardName {id: $name_id})
        RETURN coalesce(sn.source_paths, []) AS source_paths,
               size([(:StandardNameSource)-[:PRODUCED_NAME]->(sn) | 1])
                   AS produced_edges
        """,
        name_id=name_id,
    )
    return dict(rows[0])


def _source(_gc, source_id: str) -> dict:
    rows = _gc.query(
        """
        MATCH (sns:StandardNameSource {id: $source_id})
        RETURN sns.status AS status,
               sns.produced_sn_id AS produced_sn_id,
               size([(sns)-[:PRODUCED_NAME]->(:StandardName) | 1])
                   AS produced_edges
        """,
        source_id=source_id,
    )
    return dict(rows[0])


def _census(_gc) -> int:
    """Live pipeline-origin names with no producing source, whole graph."""
    rows = _gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.origin = 'pipeline'
          AND NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted']
          AND NOT (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        RETURN count(sn) AS n
        """
    )
    return int(rows[0]["n"]) if rows else 0


@pytest.mark.graph
def test_lapsed_detachment_is_reattached(_gc, _clean):
    """A source whose reason no longer holds is reattached to its name.

    The guard accepts ``_LAPSED_PATH`` against the base kind, so the recorded
    rejection is transient under the current grammar: the reconcile restores
    the provenance edge, the source's produced state, the ``source_paths``
    entry and (via provenance) the DD-side projection, and the census falls.
    """
    from imas_codex.standard_names.graph_ops import reconcile_sourceless_pipeline_names

    name_id, source_id = _brokered(
        _gc, name_stage="accepted", name_kind=_LAPSED_KIND
    )
    before = _census(_gc)

    result = reconcile_sourceless_pipeline_names(gc=_gc)

    assert result["reattached"] >= 1
    assert result["names_reattached"] >= 1
    nm = _name(_gc, name_id)
    assert nm["produced_edges"] == 1
    assert source_id in nm["source_paths"]
    src = _source(_gc, source_id)
    assert src["status"] == "composed"
    assert src["produced_sn_id"] == name_id
    assert src["produced_edges"] == 1
    assert _census(_gc) < before

    # A re-pair leaves a change record so history survives.
    rows = _gc.query(
        """
        MATCH (sn:StandardName {id: $name_id})-[:HAS_INTERNAL_CHANGE]->
              (ch:StandardNameChange)
        WHERE ch.operation = 'reattach_lapsed_detachment'
        RETURN count(ch) AS n
        """,
        name_id=name_id,
    )
    assert (rows[0]["n"] if rows else 0) >= 1


@pytest.mark.graph
def test_reason_that_still_holds_is_surfaced_and_left(_gc, _clean):
    """A name whose detachment reason still holds is blocked, never healed.

    The guard rejects ``_LAPSED_PATH`` against the change/rate kind, so the
    reason holds under the current grammar: the reconcile reports the name as
    a blocker and leaves the detached source untouched — the census is
    unchanged.
    """
    from imas_codex.standard_names.graph_ops import reconcile_sourceless_pipeline_names

    name_id, source_id = _brokered(_gc, name_stage="accepted", name_kind=_HELD_KIND)
    before = _census(_gc)

    result = reconcile_sourceless_pipeline_names(gc=_gc)

    assert result["reattached"] == 0
    assert result["blockers"] >= 1
    nm = _name(_gc, name_id)
    assert nm["produced_edges"] == 0
    # Left alone: still the detach-rewound state.
    src = _source(_gc, source_id)
    assert src["status"] == "extracted"
    assert src["produced_sn_id"] is None
    assert src["produced_edges"] == 0
    assert _census(_gc) == before


@pytest.mark.graph
def test_reattach_is_idempotent(_gc, _clean):
    """A second pass after a re-pair matches nothing and adds no edge."""
    from imas_codex.standard_names.graph_ops import reconcile_sourceless_pipeline_names

    name_id, _source_id = _brokered(
        _gc, name_stage="accepted", name_kind=_LAPSED_KIND
    )
    reconcile_sourceless_pipeline_names(gc=_gc)
    result = reconcile_sourceless_pipeline_names(gc=_gc)

    assert result["reattached"] == 0
    nm = _name(_gc, name_id)
    assert nm["produced_edges"] == 1


@pytest.mark.graph
def test_runs_from_startup_reviewable_reconcile(_gc, _clean):
    """The healing fires from the startup wiring, not only the focused call."""
    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    name_id, _source_id = _brokered(
        _gc, name_stage="reviewed", name_kind=_LAPSED_KIND
    )

    reconcile_reviewable_name_stage(gc=_gc)

    nm = _name(_gc, name_id)
    assert nm["produced_edges"] == 1

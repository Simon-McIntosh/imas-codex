"""A standard name must carry exactly one unit edge, matching its own scalar.

``StandardName`` declares ``HAS_UNIT`` with cardinality *one* in the LinkML
schema, and both writers that set a name's unit already self-heal — they drop
every pre-existing ``HAS_UNIT`` edge before merging the canonical one, so a unit
CORRECTION cannot leave the superseded edge behind alongside the new one. Names
whose unit was last written *before* that self-heal existed still carry the
residue, and no writer will ever touch them again: a name that has reached a
terminal stage is never re-composed, so the stale edge is permanent unless
something reconciles it.

:func:`reconcile_standard_name_unit_edges` is that net. It realigns any name
whose ``HAS_UNIT`` edge set disagrees with its own ``unit`` scalar — the scalar
is authoritative, being what every reader and every export consults — and is
idempotent, so it no-ops once the invariant holds. Wired into the ``run_sn_pools``
startup reconcile alongside the other structural self-heals.

Two units on one name is not a cosmetic desync: it makes the name's
dimensionality ambiguous to the attachment guard, which compares a source's DD
unit against the name's, so a name carrying both ``1`` and ``m^-2`` can absorb
sources of either dimensionality.
"""

from __future__ import annotations

import uuid

import pytest

_PREFIX = "test_unit_edge_reconcile__"


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
        for label in ("StandardName", "Unit"):
            _gc.query(
                f"MATCH (n:{label}) WHERE n.id CONTAINS $p DETACH DELETE n",
                p=_PREFIX,
            )

    _wipe()
    yield
    _wipe()


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _make_name(gc, *, unit: str, edge_units: list[str], stage: str = "accepted") -> str:
    """Create a name whose scalar is *unit* and whose edges are *edge_units*."""
    sn_id = _uid(stage)
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name = $id, sn.unit = $unit, sn.name_stage = $stage,
            sn.validation_status = 'valid'
        WITH sn
        UNWIND $edges AS eu
        MERGE (u:Unit {id: eu})
        MERGE (sn)-[:HAS_UNIT]->(u)
        """,
        id=sn_id,
        unit=unit,
        stage=stage,
        edges=edge_units,
    )
    return sn_id


def _edges(gc, sn_id: str) -> list[str]:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
        RETURN collect(DISTINCT u.id) AS units
        """,
        id=sn_id,
    )
    return sorted(u for u in (rows[0]["units"] if rows else []) if u)


# ---------------------------------------------------------------------------
# Unit-level: the reconcile is a no-op without a graph
# ---------------------------------------------------------------------------


def test_nothing_to_realign_writes_nothing():
    """With the invariant already held the reconcile reports zeros and no write.

    The selector returns no rows, so the write query must never be issued — a
    reconcile that fires a write on every run is not idempotent in any useful
    sense.
    """
    from unittest.mock import MagicMock

    from imas_codex.standard_names.graph_ops import (
        reconcile_standard_name_unit_edges,
    )

    gc = MagicMock()
    gc.query.return_value = []
    result = reconcile_standard_name_unit_edges(gc)
    assert result == {"names_realigned": 0, "edges_dropped": 0, "edges_created": 0}
    assert gc.query.call_count == 1  # the selector only


# ---------------------------------------------------------------------------
# Live graph: the invariant and its idempotency
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_stale_second_unit_edge_is_dropped(_gc, _clean):
    """A name carrying two units keeps only the one its scalar declares."""
    from imas_codex.standard_names.graph_ops import (
        reconcile_standard_name_unit_edges,
    )

    sn_id = _make_name(_gc, unit="1", edge_units=["1", "m^-2"])
    assert _edges(_gc, sn_id) == ["1", "m^-2"]

    result = reconcile_standard_name_unit_edges(_gc)

    assert _edges(_gc, sn_id) == ["1"]
    assert result["edges_dropped"] >= 1
    assert result["names_realigned"] >= 1


@pytest.mark.graph
def test_terminal_stage_is_realigned_too(_gc, _clean):
    """A superseded name is realigned — no writer will ever revisit it.

    Terminal names are excluded from the *live-unit* invariants (their unit may
    legitimately predate a correction), but the cardinality invariant is
    structural: a name has one unit whatever its stage, and a terminal name is
    precisely the case nothing else can heal.
    """
    from imas_codex.standard_names.graph_ops import (
        reconcile_standard_name_unit_edges,
    )

    sn_id = _make_name(_gc, unit="1", edge_units=["1", "m^-2"], stage="superseded")
    reconcile_standard_name_unit_edges(_gc)
    assert _edges(_gc, sn_id) == ["1"]


@pytest.mark.graph
def test_edge_disagreeing_with_the_scalar_is_replaced(_gc, _clean):
    """A single but WRONG edge is replaced by the scalar's unit.

    The scalar is authoritative: it is what the exports and the attachment
    guard's dimensionality rule read.
    """
    from imas_codex.standard_names.graph_ops import (
        reconcile_standard_name_unit_edges,
    )

    sn_id = _make_name(_gc, unit="Wb", edge_units=["W"])
    result = reconcile_standard_name_unit_edges(_gc)
    assert _edges(_gc, sn_id) == ["Wb"]
    assert result["edges_created"] >= 1


@pytest.mark.graph
def test_correct_name_is_untouched_and_pass_is_idempotent(_gc, _clean):
    """A name already holding the invariant is not rewritten, twice over."""
    from imas_codex.standard_names.graph_ops import (
        reconcile_standard_name_unit_edges,
    )

    good = _make_name(_gc, unit="m^-3", edge_units=["m^-3"])
    bad = _make_name(_gc, unit="1", edge_units=["1", "m^-2"])

    first = reconcile_standard_name_unit_edges(_gc)
    assert first["names_realigned"] >= 1

    second = reconcile_standard_name_unit_edges(_gc)
    assert second == {"names_realigned": 0, "edges_dropped": 0, "edges_created": 0}
    assert _edges(_gc, good) == ["m^-3"]
    assert _edges(_gc, bad) == ["1"]


@pytest.mark.graph
def test_name_without_a_unit_scalar_is_left_alone(_gc, _clean):
    """A name with no declared unit is a different concern — never touched.

    Stripping its edge would destroy the only unit information it has; a name
    awaiting a unit is not a cardinality violation.
    """
    from imas_codex.standard_names.graph_ops import (
        reconcile_standard_name_unit_edges,
    )

    sn_id = _uid("nounit")
    _gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name = $id, sn.name_stage = 'drafted', sn.validation_status = 'valid'
        MERGE (u:Unit {id: 'T'})
        MERGE (sn)-[:HAS_UNIT]->(u)
        """,
        id=sn_id,
    )
    reconcile_standard_name_unit_edges(_gc)
    assert _edges(_gc, sn_id) == ["T"]

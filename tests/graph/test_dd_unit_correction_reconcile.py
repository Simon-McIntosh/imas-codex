"""A registered self-contradiction correction must reach the stored DD graph.

``units.resolve_dd_unit`` rewrites a DD-declared unit when the exceptions
registry flags that path ``correct_in_graph`` — reserved for the case where the
DD contradicts *itself* on one quantity, so there is no single DD answer to
mirror and propagating the wrong dimensionality would corrupt a standard name
composed from the wrong facet.

That rewrite happens at DD BUILD time only. Adding a ``correct_in_graph`` entry
therefore had no effect on paths already in the graph, and a full DD rebuild is
expensive enough (and re-bills enrichment) that nobody runs one to land a unit
correction. A registry whose entries only take effect on a rebuild is a registry
that silently does not work.

:func:`reconcile_dd_unit_corrections` closes that gap: it re-asks
``resolve_dd_unit`` of every stored DD node and realigns the ones whose stored
unit disagrees with what the registry says today — the same predicate the build
uses, so the two can never drift. Idempotent.

Scope discipline: ONLY paths matching a ``correct_in_graph`` entry are touched.
A suppression-only entry deliberately leaves the DD unit as declared so the
mismatch axis keeps reporting it, and rewriting those would destroy the very
signal the axis exists to surface.
"""

from __future__ import annotations

import uuid

import pytest

_PREFIX = "test_dd_unit_correction__"


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
        for label in ("IMASNode", "Unit"):
            _gc.query(
                f"MATCH (n:{label}) WHERE n.id CONTAINS $p DETACH DELETE n",
                p=_PREFIX,
            )

    _wipe()
    yield
    _wipe()


def _node(gc, path: str, unit: str) -> str:
    """Create a DD node carrying *unit* on both the scalar and the edge."""
    gc.query(
        """
        MERGE (n:IMASNode {id: $path})
        SET n.unit = $unit, n.node_category = 'quantity',
            n.lifecycle_status = 'active'
        MERGE (u:Unit {id: $unit})
        MERGE (n)-[:HAS_UNIT]->(u)
        """,
        path=path,
        unit=unit,
    )
    return path


def _stored(gc, path: str) -> tuple[str | None, list[str]]:
    rows = gc.query(
        """
        MATCH (n:IMASNode {id: $path})
        OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
        RETURN n.unit AS scalar, collect(DISTINCT u.id) AS edges
        """,
        path=path,
    )
    if not rows:
        return None, []
    return rows[0]["scalar"], sorted(e for e in rows[0]["edges"] if e)


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------


def test_measurement_position_flux_is_corrected_in_graph():
    """A poloidal flux declared in watts is rewritten, not merely suppressed.

    The DD contradicts itself here: ``ece/channel/beam_tracing/beam/position/psi``
    and ``ece/channel/position/psi`` carry the same documentation ("Poloidal
    flux") while declaring ``Wb`` and ``W`` respectively. With only a
    suppression the composer inherits whichever facet it read, which is how an
    accepted poloidal-flux name came to declare total power.
    """
    from imas_codex.units import resolve_dd_unit

    assert resolve_dd_unit("ece/channel/position/psi", "W") == "Wb"
    # the facet that is already right must pass through untouched
    assert resolve_dd_unit("ece/channel/beam_tracing/beam/position/psi", "Wb") == "Wb"


def test_a_suppression_only_entry_is_not_rewritten():
    """A suppression-only DD bug keeps its declared unit in the graph.

    ``*/z_ion`` is a charge NUMBER the DD tags ``e``; the standard name
    correctly carries dimensionless and the axis suppresses the pair. The DD
    unit must stay as declared so the axis can keep reporting it.
    """
    from imas_codex.units import resolve_dd_unit

    assert resolve_dd_unit("core_profiles/profiles_1d/ion/z_ion", "e") == "e"


# ---------------------------------------------------------------------------
# The reconcile
# ---------------------------------------------------------------------------


def test_reconcile_reports_zero_without_candidates():
    """No stored node disagrees with the registry → zeros, and no write."""
    from unittest.mock import MagicMock

    from imas_codex.graph.dd_graph_ops import reconcile_dd_unit_corrections

    gc = MagicMock()
    gc.query.return_value = []
    result = reconcile_dd_unit_corrections(gc)
    assert result == {"checked": 0, "corrected": 0}
    assert gc.query.call_count == 1  # the selector only


@pytest.mark.graph
def test_stored_self_contradiction_is_realigned(_gc, _clean):
    """A stored node carrying the wrong facet's unit is corrected in place."""
    from imas_codex.graph.dd_graph_ops import reconcile_dd_unit_corrections

    # A synthetic path under the same registry glob as the real defect.
    path = _node(_gc, f"{_PREFIX}ids/channel/position/psi", "W")

    result = reconcile_dd_unit_corrections(_gc)

    scalar, edges = _stored(_gc, path)
    assert scalar == "Wb"
    assert edges == ["Wb"]
    assert result["corrected"] >= 1


@pytest.mark.graph
def test_suppression_only_path_is_untouched(_gc, _clean):
    """A node under a suppression-only entry keeps its declared unit."""
    from imas_codex.graph.dd_graph_ops import reconcile_dd_unit_corrections

    path = _node(_gc, f"{_PREFIX}core_profiles/profiles_1d/ion/z_ion", "e")
    reconcile_dd_unit_corrections(_gc)
    scalar, edges = _stored(_gc, path)
    assert scalar == "e"
    assert edges == ["e"]


@pytest.mark.graph
def test_reconcile_is_idempotent(_gc, _clean):
    """A second pass corrects nothing, so it is safe on every run."""
    from imas_codex.graph.dd_graph_ops import reconcile_dd_unit_corrections

    _node(_gc, f"{_PREFIX}ids/channel/position/psi", "W")
    first = reconcile_dd_unit_corrections(_gc)
    assert first["corrected"] >= 1
    second = reconcile_dd_unit_corrections(_gc)
    assert second["corrected"] == 0


@pytest.mark.graph
def test_already_correct_node_is_not_rewritten(_gc, _clean):
    """The facet that already declares the right unit is left alone."""
    from imas_codex.graph.dd_graph_ops import reconcile_dd_unit_corrections

    path = _node(_gc, f"{_PREFIX}ids/beam/position/psi", "Wb")
    reconcile_dd_unit_corrections(_gc)
    scalar, edges = _stored(_gc, path)
    assert scalar == "Wb"
    assert edges == ["Wb"]


@pytest.mark.graph
def test_no_live_dd_node_contradicts_the_registry(_gc):
    """Invariant: no stored DD unit disagrees with a correct_in_graph entry.

    The reconcile is wired into the ``sn run`` startup sweep, so this holds
    once it has run. A failure means a registry entry was added without the
    correction reaching the graph — exactly the silent-no-op this guards.
    """
    from imas_codex.graph.dd_graph_ops import find_dd_unit_correction_drift

    drift = find_dd_unit_correction_drift(_gc)
    assert not drift, (
        f"{len(drift)} DD node(s) carry a unit the exceptions registry marks "
        "correct_in_graph as wrong:\n  "
        + "\n  ".join(
            f"{d['path']}: stored {d['stored']!r} → {d['expected']!r}"
            for d in drift[:20]
        )
    )


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"

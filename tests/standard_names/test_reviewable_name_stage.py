"""Review-entry stage integrity for source-backed pipeline names.

A refined name is minted as a NEW node whose ``name_stage`` was, historically,
initialised only ``ON CREATE``. When a refine converges onto a name string that
already exists as an un-reviewed placeholder — a locus/parent scaffold created
at ``name_stage='pending'`` (or a bare ``MERGE``-created node with no stage) —
``ON CREATE`` does not fire, so the successor kept the placeholder stage even
though it now carries the migrated ``PRODUCED_NAME`` / ``HAS_STANDARD_NAME``
sources. ``REVIEW_NAME`` claims ``name_stage='drafted'``, so such a name was a
valid, source-backed quantity that could never enter review.

Two guarantees are covered:

* :func:`persist_refined_name` advances a placeholder-merged successor to
  ``drafted`` (unless it is a structural ``derived`` parent, which is reviewed
  structurally, not by the name quorum).
* :func:`reconcile_reviewable_name_stage` self-heals any name already stranded
  below ``drafted`` while carrying a real (non-``derived``) produced source.

The end-to-end guarantees run against a live graph (``@pytest.mark.graph``);
a fast import/return-shape check runs in the default tier.
"""

from __future__ import annotations

import uuid

import pytest

_PREFIX = "test_review_entry__"


# ---------------------------------------------------------------------------
# Fixtures (prefix-scoped; safe against the shared graph)
# ---------------------------------------------------------------------------


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
        _gc.query(
            "MATCH (n:StandardName) WHERE n.id STARTS WITH $p DETACH DELETE n",
            p=_PREFIX,
        )
        _gc.query(
            "MATCH (n:StandardNameSource) WHERE n.id STARTS WITH $p DETACH DELETE n",
            p=_PREFIX,
        )

    _wipe()
    yield
    _wipe()


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _create_sn(
    gc,
    sn_id: str,
    *,
    name_stage: str,
    origin: str | None = None,
    validation_status: str = "valid",
    chain_length: int = 0,
) -> None:
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage        = $name_stage,
            sn.origin            = $origin,
            sn.validation_status = $validation_status,
            sn.chain_length      = $chain_length,
            sn.docs_stage        = 'pending',
            sn.docs_chain_length = 0,
            sn.description        = 'Test quantity',
            sn.kind               = 'scalar',
            sn.unit               = 'eV'
        """,
        id=sn_id,
        name_stage=name_stage,
        origin=origin,
        validation_status=validation_status,
        chain_length=chain_length,
    )


def _create_source(gc, source_id: str, sn_id: str, *, source_type: str = "dd") -> None:
    gc.query(
        """
        MERGE (sns:StandardNameSource {id: $src_id})
        SET sns.status      = 'composed',
            sns.source_type = $source_type,
            sns.source_id   = 'test/path',
            sns.description  = 'A test quantity'
        WITH sns
        MATCH (sn:StandardName {id: $sn_id})
        MERGE (sns)-[:PRODUCED_NAME]->(sn)
        """,
        src_id=source_id,
        sn_id=sn_id,
        source_type=source_type,
    )


def _stage(gc, sn_id: str) -> str | None:
    rows = gc.query(
        "MATCH (sn:StandardName {id: $id}) RETURN sn.name_stage AS s", id=sn_id
    )
    return rows[0]["s"] if rows else None


def _origin(gc, sn_id: str) -> str | None:
    rows = gc.query("MATCH (sn:StandardName {id: $id}) RETURN sn.origin AS o", id=sn_id)
    return rows[0]["o"] if rows else None


# ---------------------------------------------------------------------------
# Part A — persist_refined_name advances a placeholder-merged successor
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_refine_into_pending_placeholder_advances_to_drafted(_gc, _clean):
    """A refine converging onto a bare pending placeholder yields a drafted name."""
    from imas_codex.standard_names.graph_ops import persist_refined_name

    old = _uid("old")
    new = _uid("placeholder")
    src = f"dd:{_uid('src')}"

    # The successor name already exists as an un-reviewed placeholder.
    _create_sn(_gc, new, name_stage="pending", origin=None)
    # The predecessor is mid-refine and carries the source.
    _create_sn(_gc, old, name_stage="refining", origin="pipeline")
    _create_source(_gc, src, old)

    persist_refined_name(
        old_name=old, new_name=new, description="refined", old_chain_length=0
    )

    assert _stage(_gc, new) == "drafted"
    assert _origin(_gc, new) == "pipeline"
    # Source migrated onto the successor.
    migrated = _gc.query(
        "MATCH (s:StandardNameSource {id:$s})-[:PRODUCED_NAME]->(sn:StandardName {id:$n}) "
        "RETURN count(*) AS c",
        s=src,
        n=new,
    )
    assert migrated[0]["c"] == 1


@pytest.mark.graph
def test_refine_into_derived_parent_placeholder_stays_pending(_gc, _clean):
    """A structural derived parent is not hijacked into the name review queue."""
    from imas_codex.standard_names.graph_ops import persist_refined_name

    old = _uid("old")
    new = _uid("derivedparent")
    src = f"dd:{_uid('src')}"

    _create_sn(_gc, new, name_stage="pending", origin="derived")
    _create_sn(_gc, old, name_stage="refining", origin="pipeline")
    _create_source(_gc, src, old)

    persist_refined_name(
        old_name=old, new_name=new, description="refined", old_chain_length=0
    )

    assert _stage(_gc, new) == "pending"
    assert _origin(_gc, new) == "derived"


@pytest.mark.graph
def test_refine_into_new_name_is_drafted(_gc, _clean):
    """The fresh-create path is unchanged — a brand-new successor is drafted."""
    from imas_codex.standard_names.graph_ops import persist_refined_name

    old = _uid("old")
    new = _uid("fresh")  # does not exist yet
    _create_sn(_gc, old, name_stage="refining", origin="pipeline")

    persist_refined_name(
        old_name=old, new_name=new, description="refined", old_chain_length=0
    )

    assert _stage(_gc, new) == "drafted"
    assert _origin(_gc, new) == "pipeline"


@pytest.mark.graph
def test_refine_into_accepted_name_is_not_demoted(_gc, _clean):
    """Converging onto a live accepted name must not reset its stage."""
    from imas_codex.standard_names.graph_ops import persist_refined_name

    old = _uid("old")
    new = _uid("accepted")
    _create_sn(_gc, new, name_stage="accepted", origin="pipeline")
    _create_sn(_gc, old, name_stage="refining", origin="pipeline")

    persist_refined_name(
        old_name=old, new_name=new, description="refined", old_chain_length=0
    )

    assert _stage(_gc, new) == "accepted"


# ---------------------------------------------------------------------------
# Part B — reconcile_reviewable_name_stage self-heals stranded names
# ---------------------------------------------------------------------------


def test_reconcile_reviewable_name_stage_exists_and_returns_count():
    """Fast import/return-shape guard (default tier, no live graph)."""
    from unittest.mock import MagicMock, patch

    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[{"advanced": 4}])
    result = reconcile_reviewable_name_stage(gc=mock_gc)
    assert result == {"names_advanced": 4}


@pytest.mark.graph
def test_reconcile_advances_stranded_pipeline_name(_gc, _clean):
    """A valid, source-backed name stuck at pending is advanced to drafted."""
    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    sn = _uid("stranded")
    src = f"dd:{_uid('src')}"
    _create_sn(_gc, sn, name_stage="pending", origin="pipeline")
    _create_source(_gc, src, sn)

    result = reconcile_reviewable_name_stage(gc=_gc)

    assert _stage(_gc, sn) == "drafted"
    assert result["names_advanced"] >= 1


@pytest.mark.graph
def test_reconcile_skips_derived_parent(_gc, _clean):
    """A derived structural parent placeholder is left pending."""
    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    sn = _uid("derived")
    src = f"dd:{_uid('src')}"
    _create_sn(_gc, sn, name_stage="pending", origin="derived")
    _create_source(_gc, src, sn, source_type="derived")

    reconcile_reviewable_name_stage(gc=_gc)

    assert _stage(_gc, sn) == "pending"


@pytest.mark.graph
def test_reconcile_skips_quarantined(_gc, _clean):
    """A quarantined name is not sent to review — it is regenerated instead."""
    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    sn = _uid("quarantined")
    src = f"dd:{_uid('src')}"
    _create_sn(
        _gc,
        sn,
        name_stage="pending",
        origin="pipeline",
        validation_status="quarantined",
    )
    _create_source(_gc, src, sn)

    reconcile_reviewable_name_stage(gc=_gc)

    assert _stage(_gc, sn) == "pending"


@pytest.mark.graph
def test_reconcile_is_idempotent(_gc, _clean):
    """A second pass advances nothing (already drafted); no double-write."""
    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    sn = _uid("stranded")
    src = f"dd:{_uid('src')}"
    _create_sn(_gc, sn, name_stage="pending", origin="pipeline")
    _create_source(_gc, src, sn)

    first = reconcile_reviewable_name_stage(gc=_gc)
    assert first["names_advanced"] >= 1
    second = reconcile_reviewable_name_stage(gc=_gc)
    # Only our node is guaranteed clean; assert it is not re-counted by
    # checking the node is stable and drafted.
    assert _stage(_gc, sn) == "drafted"
    assert isinstance(second["names_advanced"], int)

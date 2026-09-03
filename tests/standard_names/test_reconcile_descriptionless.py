"""A composed name with no prose is invisible to every pool, and must not persist.

:data:`REVIEW_NAME_ELIGIBILITY_WHERE` demands a non-null description, so a
``drafted`` name whose description is null can never be claimed for name
review.  Compose cannot refill the prose either: its producing source already
reads ``composed``, which the compose pool excludes.  Such a name is therefore
terminally stranded while still presenting as live work — an unscoped drain
exits with no eligible work graph-wide while the names sit there.

Two mechanisms close that state, and this module covers both.
:func:`reconcile_descriptionless_composed_names` returns an already-stranded
name to a state the pipeline can act on: the name drops back to ``pending`` and
its bound composed sources are re-opened to ``extracted`` with their produced
edge released, so the compose pool rewrites the missing prose. It runs from
:func:`reconcile_reviewable_name_stage`, which is already wired into the
``run_sn_pools`` startup reconcile, so the repair lands before the pools read
their work.
:func:`_finalize_generated_name_stage` is the guard at the source: it declines
to advance a name to ``drafted`` while the node carries no description, parking
it at ``pending`` where the reconcile can reach it rather than minting a draft
nothing can claim.
"""

from __future__ import annotations

import uuid

import pytest

_PREFIX = "test_reconcile_descriptionless__"


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
        for label in ("StandardName", "StandardNameSource"):
            _gc.query(
                f"MATCH (n:{label}) WHERE n.id CONTAINS $p DETACH DELETE n",
                p=_PREFIX,
            )

    _wipe()
    yield
    _wipe()


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _bind(
    gc,
    *,
    name_id: str,
    description: str | None,
    name_stage: str = "drafted",
    source_status: str = "composed",
) -> str:
    """Create one source bound to one name, mirroring a compose product."""
    source_id = f"dd:{name_id}"
    gc.query(
        """
        MERGE (sn:StandardName {id: $name_id})
        SET sn.name_stage = $name_stage,
            sn.description = $description,
            sn.validation_status = 'valid',
            sn.origin = 'pipeline',
            sn.source_paths = [$source_id]
        MERGE (sns:StandardNameSource {id: $source_id})
        SET sns.source_id = $name_id,
            sns.source_type = 'dd',
            sns.status = $source_status,
            sns.attempt_count = 3,
            sns.composed_at = datetime(),
            sns.produced_sn_id = $name_id,
            sns.claim_token = null,
            sns.claimed_at = null
        MERGE (sns)-[:PRODUCED_NAME]->(sn)
        """,
        name_id=name_id,
        description=description,
        name_stage=name_stage,
        source_id=source_id,
        source_status=source_status,
    )
    return source_id


def _name(gc, name_id: str) -> dict:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name_id})
        RETURN sn.name_stage AS name_stage,
               sn.description AS description,
               coalesce(sn.source_paths, []) AS source_paths,
               size([(:StandardNameSource)-[:PRODUCED_NAME]->(sn) | 1])
                   AS produced_edges
        """,
        name_id=name_id,
    )
    return dict(rows[0])


def _source(gc, source_id: str) -> dict:
    rows = gc.query(
        """
        MATCH (sns:StandardNameSource {id: $source_id})
        RETURN sns.status AS status,
               sns.attempt_count AS attempt_count,
               sns.composed_at AS composed_at,
               sns.produced_sn_id AS produced_sn_id,
               sns.claim_token AS claim_token
        """,
        source_id=source_id,
    )
    return dict(rows[0])


# ---------------------------------------------------------------------------
# The reconcile returns a stranded name and its sources to compose-eligibility
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_reconcile_reopens_descriptionless_drafted_name(_gc, _clean):
    """A drafted name with no prose drops to pending and frees its source."""
    from imas_codex.standard_names.graph_ops import (
        reconcile_descriptionless_composed_names,
    )

    stranded = _uid("stranded")
    stranded_source = _bind(_gc, name_id=stranded, description=None)

    result = reconcile_descriptionless_composed_names(gc=_gc)

    assert result["names_reopened"] >= 1
    assert result["sources_reopened"] >= 1

    name = _name(_gc, stranded)
    assert name["name_stage"] == "pending"
    assert name["produced_edges"] == 0
    assert stranded_source not in name["source_paths"]

    source = _source(_gc, stranded_source)
    assert source["status"] == "extracted"
    assert source["attempt_count"] == 0
    assert source["composed_at"] is None
    assert source["produced_sn_id"] is None
    assert source["claim_token"] is None


@pytest.mark.graph
def test_reconcile_leaves_a_described_drafted_name_untouched(_gc, _clean):
    """Prose present is the whole invariant — such a name is ordinary work."""
    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    described = _uid("described")
    described_source = _bind(
        _gc, name_id=described, description="Electron temperature at the pedestal top."
    )

    reconcile_reviewable_name_stage(gc=_gc)

    name = _name(_gc, described)
    assert name["name_stage"] == "drafted"
    assert name["produced_edges"] == 1
    assert described_source in name["source_paths"]
    assert _source(_gc, described_source)["status"] == "composed"


@pytest.mark.graph
def test_reconcile_is_idempotent_over_a_reopened_name(_gc, _clean):
    """A second pass finds nothing: the freed source no longer points anywhere."""
    from imas_codex.standard_names.graph_ops import (
        reconcile_descriptionless_composed_names,
    )

    stranded = _uid("idempotent")
    _bind(_gc, name_id=stranded, description=None)

    reconcile_descriptionless_composed_names(gc=_gc)
    second = reconcile_descriptionless_composed_names(gc=_gc)

    assert second["names_reopened"] == 0
    assert _name(_gc, stranded)["name_stage"] == "pending"


@pytest.mark.graph
def test_startup_reconcile_performs_the_repair(_gc, _clean):
    """The repair reaches the graph through the entrypoint the pools call.

    ``run_sn_pools`` already invokes :func:`reconcile_reviewable_name_stage` at
    startup, so routing the descriptionless repair through it is what makes the
    fix fire before the pools read their work.
    """
    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    stranded = _uid("via_entrypoint")
    stranded_source = _bind(_gc, name_id=stranded, description=None)

    reconcile_reviewable_name_stage(gc=_gc)

    assert _name(_gc, stranded)["name_stage"] == "pending"
    assert _source(_gc, stranded_source)["status"] == "extracted"


@pytest.mark.graph
def test_reconcile_does_not_advance_a_descriptionless_pending_name(_gc, _clean):
    """The stage advance only fires when the result is actually claimable.

    Advancing a name the review pool cannot claim moves it from
    stranded-at-pending to stranded-at-drafted and buys nothing, so a null
    description withholds the advance exactly as a quarantine does.
    """
    from imas_codex.standard_names.graph_ops import reconcile_reviewable_name_stage

    pending = _uid("pending_no_prose")
    _bind(_gc, name_id=pending, description=None, name_stage="pending")

    reconcile_reviewable_name_stage(gc=_gc)

    assert _name(_gc, pending)["name_stage"] == "pending"


# ---------------------------------------------------------------------------
# The guard at the source — compose cannot mint an unclaimable draft
# ---------------------------------------------------------------------------


def _claimed_compose_product(gc, *, name_id: str, description: str | None) -> dict:
    """Stage a claimed, extracted source and its freshly written name."""
    source_id = f"dd:{name_id}"
    token = str(uuid.uuid4())
    gc.query(
        """
        MERGE (sn:StandardName {id: $name_id})
        SET sn.description = $description,
            sn.validation_status = 'valid',
            sn.origin = 'pipeline'
        REMOVE sn.name_stage
        MERGE (sns:StandardNameSource {id: $source_id})
        SET sns.source_id = $name_id,
            sns.source_type = 'dd',
            sns.status = 'extracted',
            sns.claim_token = $token,
            sns.claim_seq = 1,
            sns.claimed_at = datetime()
        """,
        name_id=name_id,
        description=description,
        source_id=source_id,
        token=token,
    )
    return {
        "sn_id": name_id,
        "sns_id": source_id,
        "claim_token": token,
        "claim_seq": 1,
        "model": "test-compose",
    }


@pytest.mark.graph
def test_finalize_refuses_to_draft_a_name_with_no_description(_gc, _clean):
    """The compose finalize parks a prose-less product at pending, not drafted."""
    from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

    prose_less = _uid("guard_no_prose")
    item = _claimed_compose_product(_gc, name_id=prose_less, description=None)

    _finalize_generated_name_stage([item])

    assert _name(_gc, prose_less)["name_stage"] == "pending"


@pytest.mark.graph
def test_finalize_still_drafts_a_name_that_carries_prose(_gc, _clean):
    """The guard is narrow: an ordinary compose product still reaches review."""
    from imas_codex.standard_names.graph_ops import _finalize_generated_name_stage

    with_prose = _uid("guard_prose")
    item = _claimed_compose_product(
        _gc, name_id=with_prose, description="Line averaged neon density."
    )

    _finalize_generated_name_stage([item])

    name = _name(_gc, with_prose)
    assert name["name_stage"] == "drafted"
    assert _source(_gc, item["sns_id"])["status"] == "composed"

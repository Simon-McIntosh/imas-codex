"""Documentation vetoes are durable graph records, independent of scores."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_store import (
    graph_audits as _owning_graph_audits,
    graph_docs_persistence as _owning_graph_docs_persistence,
    graph_review_persistence as _owning_graph_review_persistence,
)
from imas_codex.standard_names.review import holds

pytestmark = [pytest.mark.graph, pytest.mark.timeout(180)]


@pytest.fixture
def planted_name(disposable_neo4j_empty: str) -> Iterator[str]:
    name_id = "electrostatic_potential"
    with GraphClient() as graph:
        graph.query("MATCH (node) DETACH DELETE node")
        graph.query(
            """
            CREATE (sn:StandardName {id: $id, name_stage: 'accepted',
                docs_stage: 'drafted', claim_token: 'token',
                docs_chain_length: 0, documentation: 'A potential.',
                description: 'Potential.', physics_domain: 'plasma',
                reviewer_score_name: 0.9})
            RETURN sn.id AS id
            """,
            id=name_id,
        )
        assert graph.query("MATCH (sn:StandardName) RETURN count(sn) AS n")[0]["n"] == 1
    yield name_id


def _review(name_id: str, score: float = 0.875) -> str:
    stage = _owning_graph_review_persistence.persist_reviewed_docs(
        sn_id=name_id,
        claim_token="token",
        score=score,
        model="test-reviewer",
        min_score=0.85,
        rotation_cap=3,
        skip_review_node=True,
        resolution_method="quorum_consensus",
        reviewer_chain_size=2,
    )
    with GraphClient() as graph:
        graph.query(
            """
            MATCH (sn:StandardName {id: $id})
            CREATE (sn)-[:HAS_REVIEW]->(review:StandardNameReview {
                id: $review_id, review_axis: 'docs',
                review_group_id: 'test-group', is_canonical: true,
                resolution_method: 'quorum_consensus', score: $score,
                reviewed_at: datetime()
            })
            RETURN review.id AS id
            """,
            id=name_id,
            review_id=f"{name_id}:docs:test-group:0",
            score=score,
        )
    return stage


def _raise(name_id: str, finding: str = "contradicts source") -> dict:
    return holds.raise_hold(
        name_id,
        "docs",
        "documentation_contradiction",
        finding,
        "Claim: generic quantity. Source: flux-surface-averaged potential.",
        "jev-contradiction",
    )


def _state(name_id: str) -> dict:
    with GraphClient() as graph:
        return graph.query(
            "MATCH (sn:StandardName {id: $id}) "
            "RETURN sn.docs_stage AS stage, sn.reviewer_score_docs AS score",
            id=name_id,
        )[0]


def _edit_status(name_id: str) -> str | None:
    with GraphClient() as graph:
        return graph.query(
            "MATCH (sn:StandardName {id: $id}) RETURN sn.edit_status AS status",
            id=name_id,
        )[0]["status"]


def test_held_document_keeps_judged_score_and_is_not_accepted(
    planted_name: str,
) -> None:
    _raise(planted_name)
    assert _review(planted_name) == "reviewed"
    assert _state(planted_name) == {"stage": "reviewed", "score": 0.875}
    assert not holds.docs_accepted(0.875, 0.85, planted_name)
    assert len(holds.open_holds(planted_name, "docs")) == 1


def test_unheld_review_acceptance_applies_open_edit(planted_name: str) -> None:
    with GraphClient() as graph:
        graph.query(
            "MATCH (sn:StandardName {id: $id}) SET sn.edit_status = 'open'",
            id=planted_name,
        )
    assert _review(planted_name) == "accepted"
    assert _state(planted_name) == {"stage": "accepted", "score": 0.875}
    assert _edit_status(planted_name) == "applied"


def test_clearing_hold_accepts_passing_document(planted_name: str) -> None:
    with GraphClient() as graph:
        graph.query(
            "MATCH (sn:StandardName {id: $id}) SET sn.edit_status = 'open'",
            id=planted_name,
        )
    hold = _raise(planted_name)
    _review(planted_name)
    assert _edit_status(planted_name) == "open"
    assert (
        holds.clear_holds_not_raised(planted_name, "docs", {hold["kind"]}, set()) == 1
    )
    assert _state(planted_name) == {"stage": "accepted", "score": 0.875}
    assert _edit_status(planted_name) == "applied"
    assert holds.docs_accepted(0.875, 0.85, planted_name)


def test_cap_hold_can_be_dismissed_into_acceptance(planted_name: str) -> None:
    """A terminal rotation must leave a held passing document triageable."""
    with GraphClient() as graph:
        graph.query(
            "MATCH (sn:StandardName {id: $id}) "
            "SET sn.docs_chain_length = 3, sn.edit_status = 'open'",
            id=planted_name,
        )
    hold = _raise(planted_name)
    assert _review(planted_name) == "reviewed"
    assert _state(planted_name) == {"stage": "reviewed", "score": 0.875}
    assert holds.dismiss_hold(hold["id"], "triager", "Source confirms the claim")
    assert _state(planted_name) == {"stage": "accepted", "score": 0.875}
    assert _edit_status(planted_name) == "applied"


def test_stranded_promotion_respects_docs_holds(planted_name: str) -> None:
    """A later pool pass cannot turn a held passing review into acceptance."""
    _raise(planted_name)
    assert _review(planted_name) == "reviewed"
    unheld = "unheld_passing_document"
    with GraphClient() as graph:
        graph.query(
            """
            CREATE (sn:StandardName {id: $id, name_stage: 'accepted',
                docs_stage: 'reviewed', reviewer_score_docs: 0.875,
                edit_status: 'open', documentation: 'An unheld document.'})
            CREATE (sn)-[:HAS_REVIEW]->(:StandardNameReview {
                id: $review_id, review_axis: 'docs',
                review_group_id: 'test-group', is_canonical: true,
                resolution_method: 'quorum_consensus', score: 0.875,
                reviewed_at: datetime()})
            """,
            id=unheld,
            review_id=f"{unheld}:docs:test-group:0",
        )
        assert graph.query("MATCH (sn:StandardName) RETURN count(sn) AS n")[0]["n"] == 2

    assert _owning_graph_audits.promote_stranded_reviewed(
        min_score=0.85, dry_run=True
    ) == {
        "name": 0,
        "docs": 1,
    }
    assert _state(planted_name) == {"stage": "reviewed", "score": 0.875}
    assert _owning_graph_audits.promote_stranded_reviewed(min_score=0.85) == {
        "name": 0,
        "docs": 1,
    }
    assert _state(planted_name) == {"stage": "reviewed", "score": 0.875}
    assert _state(unheld) == {"stage": "accepted", "score": 0.875}
    assert _edit_status(unheld) == "applied"


def test_stale_claim_cannot_raise_or_clear_review_holds(planted_name: str) -> None:
    """The hold set and judged score belong to the winning claim alone."""
    winner_hold = _raise(planted_name)
    with GraphClient() as graph:
        graph.query(
            """
            MATCH (sn:StandardName {id: $id})
            SET sn.claim_token = 'winner', sn.claim_seq = 2
            """,
            id=planted_name,
        )
    stale = _owning_graph_review_persistence.persist_reviewed_docs(
        sn_id=planted_name,
        claim_token="stale",
        claim_seq=1,
        score=0.875,
        model="test-reviewer",
        min_score=0.85,
        rotation_cap=3,
        skip_review_node=True,
        resolution_method="quorum_consensus",
        reviewer_chain_size=2,
        review_holds=[
            {
                "kind": "documentation_contradiction",
                "finding": "A different unsupported claim",
                "evidence": "Claim and source disagree",
                "raised_by": "jev-contradiction",
            }
        ],
        checked_hold_kinds={"documentation_contradiction"},
    )
    assert stale == ""
    assert _state(planted_name) == {"stage": "drafted", "score": None}
    assert [hold["id"] for hold in holds.open_holds(planted_name, "docs")] == [
        winner_hold["id"]
    ]

    valid = _owning_graph_review_persistence.persist_reviewed_docs(
        sn_id=planted_name,
        claim_token="winner",
        claim_seq=2,
        score=0.875,
        model="test-reviewer",
        min_score=0.85,
        rotation_cap=3,
        skip_review_node=True,
        resolution_method="quorum_consensus",
        reviewer_chain_size=2,
        review_holds=[
            {
                "kind": "documentation_contradiction",
                "finding": winner_hold["finding"],
                "evidence": winner_hold["evidence"],
                "raised_by": winner_hold["raised_by"],
            }
        ],
        checked_hold_kinds={"documentation_contradiction"},
    )
    assert valid == "reviewed"
    assert _state(planted_name) == {"stage": "reviewed", "score": 0.875}
    assert [hold["id"] for hold in holds.open_holds(planted_name, "docs")] == [
        winner_hold["id"]
    ]


def test_dismissed_finding_stays_dismissed_and_does_not_block(
    planted_name: str,
) -> None:
    hold = _raise(planted_name)
    _review(planted_name)
    assert holds.dismiss_hold(hold["id"], "triager", "Source wording is compatible")
    assert _state(planted_name)["stage"] == "accepted"
    again = _raise(planted_name, "  CONTRADICTS   SOURCE ")
    assert again["id"] == hold["id"]
    assert again["status"] == "dismissed"
    assert holds.open_holds(planted_name, "docs") == []


def test_different_finding_raises_new_hold(planted_name: str) -> None:
    first = _raise(planted_name)
    holds.dismiss_hold(first["id"], "triager", "False positive")
    second = _raise(planted_name, "A second source gives a different locus")
    assert second["id"] != first["id"]
    assert [hold["id"] for hold in holds.open_holds(planted_name, "docs")] == [
        second["id"]
    ]


def test_refine_claims_passing_score_with_open_hold(planted_name: str) -> None:
    _raise(planted_name)
    _review(planted_name)
    items = _owning_graph_docs_persistence.claim_refine_docs_batch(
        batch_size=1, min_score=0.85
    )
    assert [item["id"] for item in items] == [planted_name]
    assert _state(planted_name) == {"stage": "refining", "score": 0.875}


def test_refine_does_not_claim_passing_score_without_hold(planted_name: str) -> None:
    _review(planted_name)
    with GraphClient() as graph:
        graph.query(
            "MATCH (sn:StandardName {id: $id}) SET sn.docs_stage = 'reviewed'",
            id=planted_name,
        )
    assert (
        _owning_graph_docs_persistence.claim_refine_docs_batch(
            batch_size=1, min_score=0.85
        )
        == []
    )
    assert _state(planted_name) == {"stage": "reviewed", "score": 0.875}


LINT_FINDING = (
    "[error:lint:round_trip_failure] plasma_current: "
    "name does not survive parser round-trip"
)


def test_lint_error_raises_a_hold_and_keeps_the_judged_score(
    planted_name: str,
) -> None:
    """An error-severity lint finding is a hold, not a score clamp.

    The reviewer's own score survives so the refusal is visible and triageable
    as its own dimension; a document with an unresolved lint error cannot be
    accepted while the hold is open.
    """
    from imas_codex.standard_names.review import pipeline

    entry = {"id": planted_name, "reviewer_score": 0.9}
    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    pipeline._cap_docs_scores_for_lint_errors(
        [entry], {planted_name: LINT_FINDING}, wlog
    )

    assert entry["reviewer_score"] == 0.9, "the judged score must not be clamped"
    assert entry["audit_cap_finding"] == LINT_FINDING
    raised = holds.open_holds(planted_name, "docs")
    assert [record["kind"] for record in raised] == ["docs_lint_error"]
    assert raised[0]["finding"] == LINT_FINDING
    assert not holds.docs_accepted(0.9, 0.85, planted_name)


def test_cascade_veto_raises_a_hold_and_keeps_the_decision_score(
    planted_name: str, monkeypatch
) -> None:
    """The documentation contradiction veto is a hold, not a lowered score.

    The route still reads refine, but the judged decision score is the mapped
    one rather than a value driven below ``min_score``.
    """
    from imas_codex.standard_names.review import cascade
    from imas_codex.standard_names.review.checks import DOCS_DIMENSIONS

    thresholds = {
        "accept_threshold": 0.4,
        "score_kind": "rubric",
        "rubric": {},
    }
    monkeypatch.setattr(
        cascade, "_calibration", lambda: {"name": thresholds, "docs": thresholds}
    )
    monkeypatch.setattr(
        cascade,
        "rubric_scores",
        lambda axis, judgment, fit: dict.fromkeys(DOCS_DIMENSIONS, 0.96 * 20),
    )

    judgment = {"contradiction_veto": True, "checks": {}, "quality": 0.96}
    route = cascade.route_docs(judgment, probability=0.96)

    assert route.route == "refine"
    assert route.decision_score is not None
    assert route.decision_score >= 0.85, (
        "a veto must not drive the judged decision score below min_score"
    )
    assert route.failed_checks == ("claim_contradicts_source",)

    raised = cascade.sync_route_holds(planted_name, route, "docs")
    assert len(raised) == 1
    records = holds.open_holds(planted_name, "docs")
    assert [record["kind"] for record in records] == ["cascade_contradiction"]


def _route_pair(monkeypatch) -> tuple[object, object]:
    """A vetoing docs route and a clean one, from the same stub calibration."""
    from imas_codex.standard_names.review import cascade
    from imas_codex.standard_names.review.checks import DOCS_DIMENSIONS

    thresholds = {
        "accept_threshold": 0.4,
        "score_kind": "rubric",
        "rubric": {},
    }
    monkeypatch.setattr(
        cascade, "_calibration", lambda: {"name": thresholds, "docs": thresholds}
    )
    monkeypatch.setattr(
        cascade,
        "rubric_scores",
        lambda axis, judgment, fit: dict.fromkeys(DOCS_DIMENSIONS, 0.96 * 20),
    )
    vetoed = cascade.route_docs(
        {"contradiction_veto": True, "checks": {}, "quality": 0.96}, probability=0.96
    )
    clean = cascade.route_docs(
        {"contradiction_veto": False, "checks": {}, "quality": 0.96}, probability=0.96
    )
    return vetoed, clean


def test_resolved_cascade_veto_clears_the_hold(planted_name: str, monkeypatch) -> None:
    """A later pass that reports no contradiction releases the document.

    The veto is reconciled on every deciding pass, so the refusal ends with the
    finding: a hold left open would re-claim the document on each pass and walk
    it to the rotation cap.
    """
    from imas_codex.standard_names.review import cascade

    vetoed, clean = _route_pair(monkeypatch)
    assert vetoed.holds
    assert clean.holds == ()

    cascade.sync_route_holds(planted_name, vetoed, "docs")
    assert [r["kind"] for r in holds.open_holds(planted_name, "docs")] == [
        "cascade_contradiction"
    ]

    cascade.sync_route_holds(planted_name, clean, "docs")
    assert holds.open_holds(planted_name, "docs") == []


def test_a_name_axis_pass_leaves_a_documentation_contradiction_hold_open(
    planted_name: str, monkeypatch
) -> None:
    """Only the axis these checks run on has a verdict about their holds.

    A name-axis pass decides the name, not the documentation, so its route
    carries no contradiction verdict at all — and a route that carries no holds
    is indistinguishable from one that found none. Reconciling the
    documentation holds against it would therefore clear an open
    ``cascade_contradiction`` and release a document whose claim still
    contradicts its source. The reconciliation is confined to the axis it ran
    on, so the hold survives a pass on the other axis untouched.
    """
    from imas_codex.standard_names.review import cascade

    _, clean_docs = _route_pair(monkeypatch)
    hold = holds.raise_hold(
        planted_name,
        "docs",
        "cascade_contradiction",
        "claim contradicts its DD source",
        "Jev contradiction veto on claim_contradicts_source",
        "jev-cascade",
    )

    # A real name-axis route, and the pass that produced it.
    cascade.sync_route_holds(planted_name, clean_docs, "name")

    assert [r["id"] for r in holds.open_holds(planted_name, "docs")] == [hold["id"]]


def test_resolved_lint_error_clears_the_hold_and_releases_the_document(
    planted_name: str,
) -> None:
    """The lint check clears the hold its next pass no longer justifies.

    A pass runs the lint check over every scored entry for the batch, so an
    entry with no error finding reconciles: the hold is cleared and the
    document is released through the canonical gate, at the score the reviewer
    gave it. The refusal therefore lasts exactly as long as the finding does,
    which is what the score clamp this replaces also did.
    """
    from imas_codex.standard_names.review import pipeline

    with GraphClient() as graph:
        graph.query(
            "MATCH (sn:StandardName {id: $id}) SET sn.edit_status = 'open'",
            id=planted_name,
        )
    holds.raise_hold(
        planted_name,
        "docs",
        "docs_lint_error",
        LINT_FINDING,
        f"error-severity lint finding on {planted_name}",
        "review-lint",
    )
    assert _review(planted_name) == "reviewed"
    assert [r["kind"] for r in holds.open_holds(planted_name, "docs")] == [
        "docs_lint_error"
    ]

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    pipeline._cap_docs_scores_for_lint_errors(
        [{"id": planted_name, "reviewer_score": 0.875}], {}, wlog
    )

    assert holds.open_holds(planted_name, "docs") == []
    assert _state(planted_name) == {"stage": "accepted", "score": 0.875}
    assert _edit_status(planted_name) == "applied"


def _enrich_batch(monkeypatch, names: list[dict]) -> dict:
    """Drive the real enrich pass over one batch and return that batch.

    The audit did not run for this batch, which is the case under test: the
    pass maps that to an absent lint verdict rather than an empty one. The two
    context lookups the pass also makes are neutralised, so the batch core runs
    for real and nothing reaches a remote endpoint.
    """
    import asyncio
    from types import SimpleNamespace

    from imas_codex.discovery.base.progress import WorkerStats
    from imas_codex.discovery.base.supervision import PipelinePhase
    from imas_codex.standard_names.review import enrichment, pipeline

    monkeypatch.setattr(
        enrichment, "build_neighborhood_context", lambda *a, **k: [], raising=True
    )
    monkeypatch.setattr(pipeline, "_fetch_review_dd_context", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "_fetch_review_derived_children", lambda *a, **k: None
    )

    batch = {"names": names, "audit_findings": [], "error_lint_findings": {}}
    state = SimpleNamespace(
        review_batches=[batch],
        all_names=[],
        neighborhood_k=10,
        audit_report=None,
        skip_audit=False,
        enrich_stats=WorkerStats(),
        enrich_phase=PipelinePhase("enrich"),
        should_stop=lambda: False,
    )
    asyncio.run(pipeline.enrich_review_worker(state))
    return batch


def test_a_skipped_audit_reports_no_lint_verdict_and_leaves_the_hold(
    planted_name: str, monkeypatch
) -> None:
    """A batch whose audit did not run carries no lint verdict, not an empty one.

    Driven through the real enrich pass rather than by handing the cap a
    ``None``: the verdict is read off the batch the pass produces. A skipped
    audit recorded as an empty mapping reads as a clean audit, and the cap then
    clears every lint hold on the names it scored — so a document whose error
    lint finding still stands is released. The companion test
    ``test_resolved_lint_error_clears_the_hold_and_releases_the_document``
    covers the other direction, where the audit ran and found nothing and the
    hold is cleared.
    """
    from imas_codex.standard_names.review import pipeline

    hold = holds.raise_hold(
        planted_name,
        "docs",
        "docs_lint_error",
        LINT_FINDING,
        "error-severity lint finding",
        "review-lint",
    )
    batch = _enrich_batch(
        monkeypatch,
        [
            {
                "id": planted_name,
                "description": "plasma potential",
                "documentation": "Electrostatic potential of the plasma.",
            }
        ],
    )

    assert batch["error_lint_findings"] is None, (
        "a batch with no audit report must carry no lint verdict"
    )

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    pipeline._cap_docs_scores_for_lint_errors(
        [{"id": planted_name, "reviewer_score": 0.875}],
        batch["error_lint_findings"],
        wlog,
    )

    assert [r["id"] for r in holds.open_holds(planted_name, "docs")] == [hold["id"]]


def test_dismissal_does_not_accept_a_quarantined_document(
    planted_name: str,
) -> None:
    """A dismissed hold releases a document only through the canonical gate.

    The row carries everything the gate otherwise requires: an accepted name
    axis, a passing documentation score and a reachable winning documentation
    review. The quarantine is therefore the only clause that can refuse it, so
    the assertion is about the gate and not about review eligibility — a row
    refused for want of a review would pass this test with the quarantine
    clause deleted.
    """
    name_id = planted_name
    # The hold is raised first so the documentation stays refused while it is
    # reviewed, which is the state a dismissal starts from.
    hold = _raise(name_id, "unsupported claim")
    assert _review(name_id) == "reviewed"
    with GraphClient() as graph:
        graph.query(
            "MATCH (sn:StandardName {id: $id}) "
            "SET sn.validation_status = 'quarantined'",
            id=name_id,
        )

    assert holds.dismiss_hold(hold["id"], "triager", "checked by hand")
    assert holds.open_holds(name_id, "docs") == []
    assert _state(name_id) == {"stage": "reviewed", "score": 0.875}


def test_dismissal_does_not_accept_a_document_without_a_winning_review(
    disposable_neo4j_empty: str,
) -> None:
    """A passing score and an empty hold set are not sufficient for acceptance.

    A row with no reachable winning docs review never passes the canonical
    gate, so clearing its last hold cannot accept it.
    """
    name_id = "no_winning_review_document"
    with GraphClient() as graph:
        graph.query("MATCH (node) DETACH DELETE node")
        graph.query(
            """
            CREATE (sn:StandardName {id: $id, name_stage: 'accepted',
                docs_stage: 'reviewed', reviewer_score_docs: 0.9})
            """,
            id=name_id,
        )
    hold = holds.raise_hold(
        name_id,
        "docs",
        "documentation_contradiction",
        "unsupported claim",
        "claim and source disagree",
        "tester",
    )
    assert holds.dismiss_hold(hold["id"], "triager", "checked by hand")
    assert _state(name_id) == {"stage": "reviewed", "score": 0.9}


def test_refiner_input_contains_open_hold_finding(planted_name: str) -> None:
    _raise(planted_name, "The source specifies flux-surface averaging")
    _review(planted_name)
    item = _owning_graph_docs_persistence.claim_refine_docs_batch(batch_size=1)[0]
    comments = item["prior_docs_reviews"][-1]["reviewer_comments_per_dim"]
    assert "The source specifies flux-surface averaging" in comments.values()

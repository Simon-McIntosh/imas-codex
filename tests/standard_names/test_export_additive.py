"""Additive review-batch export: approved ∪ batch, and the catalog.yml marker.

Section 1 — the candidate predicate over a live graph: a batch export returns
the already-approved catalog plus exactly the batch (additive diff), still
subject to the validation and docs gates.
Section 2 — the manifest stamp: a review export writes ``export_scope: review``
and the ``review_batch`` id-set into catalog.yml; a normal export omits both.
"""

from __future__ import annotations

import pytest
import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.export import _fetch_candidates, _write_manifest

PREFIX = "__exptest__"
APPROVED1 = f"{PREFIX}_approved1"
BATCH_ACC1 = f"{PREFIX}_batch_acc1"
BATCH_ACC2 = f"{PREFIX}_batch_acc2"
ACCEPTED_OUT = f"{PREFIX}_accepted_out"
BATCH_QUAR = f"{PREFIX}_batch_quarantined"
BATCH_NODOCS = f"{PREFIX}_batch_nodocs"
BATCH_IDS = [BATCH_ACC1, BATCH_ACC2, BATCH_QUAR, BATCH_NODOCS]


def _cleanup():
    with GraphClient() as gc:
        gc.query("MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n", p=PREFIX)


@pytest.fixture
def export_graph():
    _cleanup()
    with GraphClient() as gc:
        gc.query(
            """
            MERGE (a:StandardName {id: $approved1})
              SET a.name_stage='approved', a.validation_status='valid',
                  a.docs_stage='accepted', a.physics_domain='equilibrium'
            MERGE (b1:StandardName {id: $batch_acc1})
              SET b1.name_stage='accepted', b1.validation_status='valid',
                  b1.docs_stage='accepted', b1.physics_domain='equilibrium'
            MERGE (b2:StandardName {id: $batch_acc2})
              SET b2.name_stage='accepted', b2.validation_status='valid',
                  b2.docs_stage='accepted', b2.physics_domain='equilibrium'
            MERGE (o:StandardName {id: $accepted_out})
              SET o.name_stage='accepted', o.validation_status='valid',
                  o.docs_stage='accepted', o.physics_domain='equilibrium'
            MERGE (q:StandardName {id: $batch_quar})
              SET q.name_stage='accepted', q.validation_status='quarantined',
                  q.docs_stage='accepted', q.physics_domain='equilibrium'
            MERGE (nd:StandardName {id: $batch_nodocs})
              SET nd.name_stage='accepted', nd.validation_status='valid',
                  nd.docs_stage='drafted', nd.physics_domain='equilibrium'
            """,
            approved1=APPROVED1,
            batch_acc1=BATCH_ACC1,
            batch_acc2=BATCH_ACC2,
            accepted_out=ACCEPTED_OUT,
            batch_quar=BATCH_QUAR,
            batch_nodocs=BATCH_NODOCS,
        )
    yield
    _cleanup()


def _ours(ids):
    return {i for i in ids if i.startswith(PREFIX)}


@pytest.mark.graph
def test_batch_export_is_additive(export_graph):
    approved_only = [r["id"] for r in _fetch_candidates(batch=[])]
    with_batch = [r["id"] for r in _fetch_candidates(batch=BATCH_IDS)]

    # Approved-only sees just the approved node; the batch adds exactly the two
    # valid, docs-accepted accepted names — the additive diff.
    assert _ours(approved_only) == {APPROVED1}
    assert _ours(with_batch) == {APPROVED1, BATCH_ACC1, BATCH_ACC2}
    assert _ours(set(with_batch)) - _ours(set(approved_only)) == {
        BATCH_ACC1,
        BATCH_ACC2,
    }

    # An accepted name NOT in the batch is excluded (no full-corpus dump).
    assert ACCEPTED_OUT not in with_batch
    # Batch members failing the validation / docs gates are excluded.
    assert BATCH_QUAR not in with_batch
    assert BATCH_NODOCS not in with_batch


@pytest.mark.graph
def test_batch_export_is_deterministic(export_graph):
    ours = [
        i
        for i in (r["id"] for r in _fetch_candidates(batch=BATCH_IDS))
        if i.startswith(PREFIX)
    ]
    assert ours == sorted(ours)


# ── Section 2 — manifest stamping (pure) ──────────────────────────────────


def _manifest_kwargs():
    return {
        "cocos_convention": 17,
        "candidate_count": 2,
        "published_count": 2,
        "excluded_below_score_count": 0,
        "excluded_unreviewed_count": 0,
        "min_score_applied": 0.65,
        "min_description_score_applied": None,
        "include_unreviewed": False,
        "source_commit_sha": None,
        "domains_included": ["equilibrium"],
    }


def test_manifest_stamps_review_batch(tmp_path):
    _write_manifest(
        tmp_path,
        export_scope="review",
        review_batch=["b_name", "a_name"],
        **_manifest_kwargs(),
    )
    data = yaml.safe_load((tmp_path / "catalog.yml").read_text())
    assert data["export_scope"] == "review"
    assert data["review_batch"] == ["a_name", "b_name"]  # sorted


def test_manifest_omits_review_batch_when_absent(tmp_path):
    _write_manifest(
        tmp_path,
        export_scope="full",
        review_batch=None,
        **_manifest_kwargs(),
    )
    data = yaml.safe_load((tmp_path / "catalog.yml").read_text())
    assert "review_batch" not in data

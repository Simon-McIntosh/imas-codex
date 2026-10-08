"""Focused document discovery claims only images below named Document paths."""

from __future__ import annotations

import asyncio

import click
import pytest

from imas_codex.cli.discover.common import resolve_focus_items
from imas_codex.cli.discover.documents import DocumentsOptions, run_documents_stage
from imas_codex.discovery.base.image import claim_images_for_scoring
from imas_codex.discovery.documents.pipeline import (
    DocumentDiscoveryState,
    _has_pending_image_documents,
    _has_pending_image_scores,
    has_pending_work,
    run_document_discovery,
)
from imas_codex.discovery.documents.workers import (
    _claim_image_documents,
    image_fetch_worker,
    image_score_worker,
)

FACILITY = "tcv"
DOCUMENTS = (
    {"id": "doc-a", "path": "/archive/selected/a.png", "document_type": "image"},
    {"id": "doc-b", "path": "/archive/other/b.png", "document_type": "image"},
)
IMAGES = (
    {"id": "img-a", "document_path": DOCUMENTS[0]["path"]},
    {"id": "img-b", "document_path": DOCUMENTS[1]["path"]},
)
PREFIX = "/archive/selected"


class Graph:
    def __init__(self):
        self.claimed_documents = []
        self.claimed_images = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def query(self, cypher, **params):
        assert params.get("facility", FACILITY) == FACILITY
        if "UNWIND $prefixes AS prefix" in cypher:
            return [
                {
                    "prefix": prefix,
                    "matches": sum(doc["path"].startswith(prefix) for doc in DOCUMENTS),
                }
                for prefix in params["prefixes"]
            ]

        if "MATCH (d:Document" in cypher and "SET d.claimed_at" in cypher:
            self.claimed_documents = self._selected_documents(cypher, params)
            return []
        if "MATCH (d:Document {claim_token:" in cypher:
            return self.claimed_documents
        if "RETURN count(d) > 0 AS has_work" in cypher:
            return [{"has_work": bool(self._selected_documents(cypher, params))}]

        if "MATCH (img:Image" in cypher and "SET img.claimed_at" in cypher:
            self.claimed_images = self._selected_images(cypher, params)
            return []
        if "MATCH (img:Image" in cypher and "claim_token: $token" in cypher:
            return self.claimed_images
        if "RETURN count(img) > 0 AS has_work" in cypher:
            return [{"has_work": bool(self._selected_images(cypher, params))}]
        raise AssertionError(cypher)

    @staticmethod
    def _selected_documents(cypher, params):
        if "d.path STARTS WITH prefix" not in cypher:
            return list(DOCUMENTS)
        prefixes = params["path_prefixes"]
        return [
            doc for doc in DOCUMENTS if any(doc["path"].startswith(p) for p in prefixes)
        ]

    @staticmethod
    def _selected_images(cypher, params):
        if "d.path STARTS WITH prefix" not in cypher:
            return list(IMAGES)
        assert "(d:Document {facility_id: $facility})-[:HAS_IMAGE]->(img)" in cypher
        prefixes = params["path_prefixes"]
        return [
            image
            for image in IMAGES
            if any(image["document_path"].startswith(p) for p in prefixes)
        ]


@pytest.fixture
def graph(monkeypatch):
    graph = Graph()
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: graph)
    return graph


def test_focused_fetch_claim_and_pending_check_ignore_other_documents(graph):
    assert [
        doc["id"] for doc in _claim_image_documents(FACILITY, path_prefixes=(PREFIX,))
    ] == ["doc-a"]
    assert _has_pending_image_documents(FACILITY, (PREFIX,))
    assert not _has_pending_image_documents(FACILITY, ("/absent",))
    assert [doc["id"] for doc in _claim_image_documents(FACILITY)] == [
        "doc-a",
        "doc-b",
    ]


def test_focused_scoring_claim_and_pending_check_ignore_other_images(graph):
    assert [
        image["id"]
        for image in claim_images_for_scoring(FACILITY, path_prefixes=(PREFIX,))
    ] == ["img-a"]
    assert _has_pending_image_scores(FACILITY, (PREFIX,))
    assert not _has_pending_image_scores(FACILITY, ("/absent",))
    assert not has_pending_work(FACILITY, ("/absent",))
    assert [image["id"] for image in claim_images_for_scoring(FACILITY)] == [
        "img-a",
        "img-b",
    ]


def test_pipeline_pending_checks_receive_path_prefixes(monkeypatch, graph):
    observed = []

    async def fake_engine(state, workers, **_):
        observed.extend([spec.name for spec in workers])
        state.image_phase.refresh_has_work()
        state.image_score_phase.refresh_has_work()

    monkeypatch.setattr(
        "imas_codex.discovery.documents.pipeline.run_discovery_engine", fake_engine
    )
    state = DocumentDiscoveryState(facility=FACILITY, path_prefixes=(PREFIX,))
    asyncio.run(run_document_discovery(state))
    assert observed == ["image", "vlm"]
    assert state.image_phase._cached_has_work
    assert state.image_score_phase._cached_has_work


def test_workers_forward_path_prefixes_to_both_claims(monkeypatch):
    observed = []

    def fake_fetch_claim(facility, limit, path_prefixes):
        observed.append(("fetch", facility, path_prefixes))
        return []

    def fake_score_claim(facility, limit, path_prefixes):
        observed.append(("score", facility, path_prefixes))
        return []

    monkeypatch.setattr(
        "imas_codex.discovery.documents.workers._claim_image_documents",
        fake_fetch_claim,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.base.image.claim_images_for_scoring", fake_score_claim
    )
    monkeypatch.setattr("imas_codex.discovery.base.facility.get_facility", lambda _: {})
    state = DocumentDiscoveryState(facility=FACILITY, path_prefixes=(PREFIX,))
    state.image_phase.mark_done()
    state.image_score_phase.mark_done()
    asyncio.run(image_fetch_worker(state))
    asyncio.run(image_score_worker(state))
    assert observed == [
        ("fetch", FACILITY, (PREFIX,)),
        ("score", FACILITY, (PREFIX,)),
    ]


@pytest.fixture
def stage(monkeypatch, graph):
    states = []
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda facility: {"id": facility, "ssh_host": "remote"},
    )

    async def fake_pipeline(state, **_):
        states.append(state)
        return {"images_fetched": 0, "images_captioned": 0, "cost": 0.0}

    def fake_run_discovery(_config, async_main, **_):
        return asyncio.run(async_main(None, None))

    monkeypatch.setattr(
        "imas_codex.discovery.documents.pipeline.run_document_discovery", fake_pipeline
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.run_discovery", fake_run_discovery
    )
    return states


@pytest.mark.parametrize(
    "manifest_body", ["- /archive/selected\n", "items:\n  - /archive/selected\n"]
)
def test_manifest_forms_produce_the_same_focused_stage(stage, tmp_path, manifest_body):
    manifest = tmp_path / "documents.yaml"
    manifest.write_text(manifest_body, encoding="utf-8")
    run_documents_stage(FACILITY, DocumentsOptions(flush=True, focus=(str(manifest),)))
    assert stage[-1].path_prefixes == (PREFIX,)


def test_named_path_reaches_the_stage_and_unknown_path_is_refused(stage):
    run_documents_stage(FACILITY, DocumentsOptions(flush=True, focus=(PREFIX,)))
    assert stage[-1].path_prefixes == (PREFIX,)
    with pytest.raises(click.UsageError, match="/missing"):
        run_documents_stage(FACILITY, DocumentsOptions(flush=True, focus=("/missing",)))
    assert len(stage) == 1


def test_shared_resolver_preserves_order_and_refuses_invalid_manifest(tmp_path):
    assert resolve_focus_items(("one two", "two", "three")) == [
        "one",
        "two",
        "three",
    ]
    manifest = tmp_path / "items.yaml"
    manifest.write_text("sources:\n  - one\n", encoding="utf-8")
    with pytest.raises(click.UsageError, match="items.yaml"):
        resolve_focus_items((str(manifest),))

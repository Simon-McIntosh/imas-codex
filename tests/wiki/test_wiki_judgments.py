"""The wiki ingest decision uses typed judgments and local content evidence."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from imas_codex.discovery.base.facility import get_facility
from imas_codex.discovery.wiki.entity_extraction import extract_facility_tool_mentions
from imas_codex.discovery.wiki.scoring import (
    build_content_judgment_questions,
    build_content_judgment_state,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "imas_codex/config/facilities"
_FACILITIES = [
    path.stem
    for path in sorted(_CONFIG_DIR.glob("*.yaml"))
    if (yaml.safe_load(path.read_text()) or {}).get("data_access_patterns")
]


def test_judgment_question_set_and_storage_slots():
    from imas_codex.graph.schema import get_schema

    questions = build_content_judgment_questions()
    assert set(questions) == {
        "ingest_relevance",
        "purpose",
        "score_data_documentation",
        "score_physics_content",
        "score_code_documentation",
        "score_data_access",
        "score_calibration",
        "score_imas_relevance",
    }
    assert questions["ingest_relevance"]["type"] == "noul"
    assert questions["purpose"]["type"] == "choice"
    assert all(
        questions[key]["type"] == "score"
        for key in questions
        if key.startswith("score_")
    )
    for class_name in ("WikiPage", "Document", "Image"):
        slots = get_schema().get_all_slots(class_name)
        assert {
            "judgment_model",
            "ingest_relevance",
            "purpose_probs",
            "purpose_confidence",
        } <= set(slots)
        for key in questions:
            if key.startswith("score_"):
                assert {f"{key}_probs", f"{key}_confidence"} <= set(slots)


@pytest.mark.parametrize("facility", _FACILITIES)
@pytest.mark.parametrize("kind", ["page", "document", "image"])
def test_state_carries_only_matches_in_own_text(facility: str, kind: str):
    config = get_facility(facility)
    patterns = config["data_access_patterns"]
    tools = patterns.get("key_tools") or []
    imports = patterns.get("code_import_patterns") or []
    candidate = next(
        value
        for value in tools + imports
        if extract_facility_tool_mentions(value, tools, imports)
    )
    text_key = "ocr_text" if kind == "image" else "preview_text"
    matched = build_content_judgment_state(
        {text_key: f"This page documents {candidate}."}, kind, facility, config
    )
    neutral = build_content_judgment_state(
        {text_key: "A neutral page about scheduling."}, kind, facility, config
    )
    assert matched["resource"]["data_access_tool_matches"]
    assert neutral["resource"]["data_access_tool_matches"] == []
    assert matched["facility"]["id"] == facility


@pytest.mark.parametrize(
    ("function_name", "node_alias"),
    [("claim_pages_for_ingesting", "wp"), ("claim_documents_for_ingesting", "wa")],
)
def test_ingest_claim_reads_jev_judgment(function_name: str, node_alias: str):
    from imas_codex.discovery.wiki import graph_ops

    client = MagicMock()
    client.query.return_value = []
    client.__enter__.return_value = client
    with patch.object(graph_ops, "GraphClient", return_value=client):
        getattr(graph_ops, function_name)("tcv")
    claim_query = client.query.call_args_list[0].args[0]
    assert f"{node_alias}.ingest_relevance >= $min_score" in claim_query
    assert f"{node_alias}.score_composite >= $min_score" not in claim_query


@pytest.mark.parametrize(
    "function_name", ["claim_pages_for_scoring", "claim_documents_for_scoring"]
)
def test_judgment_claim_includes_stale_model(function_name: str):
    from imas_codex.discovery.wiki import graph_ops

    client = MagicMock()
    client.query.side_effect = [[], [{"id": "tcv:stored"}]]
    client.__enter__.return_value = client
    with patch.object(graph_ops, "GraphClient", return_value=client):
        claimed = getattr(graph_ops, function_name)("tcv", 1)
    assert claimed == [{"id": "tcv:stored"}]
    query = client.query.call_args_list[0].args[0]
    assert "coalesce(n.judgment_model, '') <> $judgment_model" in query
    assert "n.preview_text IS NOT NULL" in query
    assert "ORDER BY CASE WHEN n.status" in query


def test_low_jev_judgment_refuses_high_legacy_score():
    from imas_codex.discovery.wiki import graph_ops

    client = MagicMock()
    client.__enter__.return_value = client
    with patch.object(graph_ops, "GraphClient", return_value=client):
        graph_ops.mark_pages_scored(
            "tcv",
            [
                {
                    "id": "tcv:old-high-score",
                    "preview_text": "Only an administrative notice",
                    "score_composite": 0.99,
                    "ingest_relevance": 0.01,
                }
            ],
        )
    assert client.query.call_args.kwargs["status"] == "skipped"


def test_calibrated_gate_admits_relevant_page_below_legacy_cutoff():
    from imas_codex.discovery.wiki import graph_ops

    client = MagicMock()
    client.__enter__.return_value = client
    with patch.object(graph_ops, "GraphClient", return_value=client):
        graph_ops.mark_pages_scored(
            "tcv",
            [
                {
                    "id": "tcv:measured-data-guide",
                    "preview_text": "Guide to measured diagnostic signals",
                    "ingest_relevance": 0.2,
                }
            ],
        )
    assert client.query.call_args.kwargs["status"] == "scored"


def test_language_model_response_models_contain_text_only():
    from imas_codex.discovery.wiki.models import (
        DocumentScoreResult,
        ImageCaptionResult,
        WikiScoreResult,
    )

    assert set(WikiScoreResult.model_fields) == {"id", "description"}
    assert set(DocumentScoreResult.model_fields) == {"id", "description"}
    assert set(ImageCaptionResult.model_fields) == {
        "id",
        "description",
        "ocr_text",
        "mermaid_diagram",
    }


def test_document_preview_keeps_judged_span_for_rejudgment():
    from imas_codex.discovery.wiki import graph_ops

    client = MagicMock()
    client.__enter__.return_value = client
    preview = "x" * 1500
    with patch.object(graph_ops, "GraphClient", return_value=client):
        graph_ops.mark_documents_scored(
            "tcv", [{"id": "tcv:manual", "preview_text": preview}]
        )
    stored = client.query.call_args.kwargs["batch"][0]["preview_text"]
    assert stored == preview


def test_page_description_is_followed_by_jev_judgment():
    from imas_codex.discovery.wiki import scoring
    from imas_codex.discovery.wiki.models import WikiScoreBatch, WikiScoreResult

    async def describe(**kwargs):
        assert kwargs["response_model"] is WikiScoreBatch
        return (
            WikiScoreBatch(
                results=[WikiScoreResult(id="tcv:manual", description="Signal manual")]
            ),
            0.02,
            0,
        )

    async def judge(items, kind, facility, config):
        assert kind == "page" and facility == "tcv"
        assert items[0]["description"] == "Signal manual"
        return [{**items[0], "ingest_relevance": 0.2, "score_cost": 0.001}], 0.001

    with (
        patch("imas_codex.discovery.base.llm.acall_llm_structured", describe),
        patch.object(scoring, "judge_content_items", judge),
    ):
        results, cost = asyncio.run(
            scoring._score_pages_batch(
                [{"id": "tcv:manual", "preview_text": "A diagnostic signal manual"}],
                "description-model",
                facility="tcv",
            )
        )
    assert results[0]["should_ingest"] is True
    assert results[0]["score_composite"] == 0.2
    assert cost == pytest.approx(0.021)


def test_image_vision_returns_text_before_jev_judgment():
    from imas_codex.discovery.wiki import scoring
    from imas_codex.discovery.wiki.models import ImageCaptionBatch, ImageCaptionResult

    tool = get_facility("tcv")["data_access_patterns"]["key_tools"][0]

    async def caption(**kwargs):
        assert kwargs["response_model"] is ImageCaptionBatch
        return (
            ImageCaptionBatch(
                results=[
                    ImageCaptionResult(
                        id="tcv:figure",
                        description="Signal diagram",
                        ocr_text=tool,
                    )
                ]
            ),
            0.03,
            0,
        )

    async def judge(items, kind, facility, config):
        assert kind == "image" and facility == "tcv"
        assert (
            tool
            in scoring.build_content_judgment_state(items[0], kind, facility, config)[
                "resource"
            ]["data_access_tool_matches"]
        )
        return [{**items[0], "ingest_relevance": 0.3, "score_cost": 0.001}], 0.001

    with (
        patch("imas_codex.discovery.base.llm.acall_llm_structured", caption),
        patch.object(scoring, "judge_content_items", judge),
    ):
        results, cost = asyncio.run(
            scoring._score_images_batch(
                [{"id": "tcv:figure", "image_data": "abc"}],
                "vision-model",
                facility_id="tcv",
            )
        )
    assert results[0]["ocr_text"] == tool
    assert results[0]["should_ingest"] is True
    assert cost == pytest.approx(0.031)

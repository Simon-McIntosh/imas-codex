"""The wiki ingest decision uses typed judgments and local content evidence."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from imas_codex.discovery.base.facility import get_facility
from imas_codex.discovery.wiki.entity_extraction import extract_facility_tool_mentions
from imas_codex.discovery.wiki.scoring import build_content_judgment_state

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "imas_codex/config/facilities"
_FACILITIES = [
    path.stem
    for path in sorted(_CONFIG_DIR.glob("*.yaml"))
    if (yaml.safe_load(path.read_text()) or {}).get("data_access_patterns")
]


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


@pytest.mark.parametrize("function_name", ["claim_pages_for_scoring", "claim_documents_for_scoring"])
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
            [{
                "id": "tcv:old-high-score",
                "preview_text": "Only an administrative notice",
                "score_composite": 0.99,
                "ingest_relevance": 0.01,
            }],
        )
    assert client.query.call_args.kwargs["status"] == "skipped"


def test_language_model_response_models_contain_text_only():
    from imas_codex.discovery.wiki.models import (
        DocumentScoreResult,
        ImageCaptionResult,
        WikiScoreResult,
    )

    assert set(WikiScoreResult.model_fields) == {"id", "description"}
    assert set(DocumentScoreResult.model_fields) == {"id", "description"}
    assert set(ImageCaptionResult.model_fields) == {
        "id", "description", "ocr_text", "mermaid_diagram"
    }

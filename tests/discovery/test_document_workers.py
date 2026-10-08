"""Document images receive text from vision before a typed judgment."""

import asyncio
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.base.image import mark_images_scored, score_images_batch


def test_shared_vision_response_contains_only_caption_ocr_and_diagram():
    class ReachedVision(Exception):
        pass

    async def capture_vision(**kwargs):
        response_model = kwargs["response_model"]
        item_model = get_args(response_model.model_fields["results"].annotation)[0]
        assert set(item_model.model_fields) == {
            "id",
            "description",
            "ocr_text",
            "mermaid_diagram",
        }
        raise ReachedVision

    with patch("imas_codex.discovery.base.llm.acall_llm_structured", capture_vision):
        with pytest.raises(ReachedVision):
            asyncio.run(
                score_images_batch(
                    [{"id": "tcv:figure", "image_data": "abc"}],
                    "vision-model",
                    facility_id="tcv",
                )
            )


def test_document_image_persistence_keeps_judgment_and_drops_image_bytes():
    client = MagicMock()
    client.__enter__.return_value = client
    with patch("imas_codex.graph.GraphClient", return_value=client):
        count = mark_images_scored(
            "tcv",
            [
                {
                    "id": "tcv:figure",
                    "image_data": "encoded",
                    "description": "A diagnostic diagram",
                    "ocr_text": "signal",
                    "ingest_relevance": 0.2,
                    "judgment_model": "judge",
                    "score_composite": 0.2,
                    "score_cost": 0.01,
                }
            ],
        )
    assert count == 1
    query = client.query.call_args.args[0]
    row = client.query.call_args.kwargs["rows"][0]
    assert row["fields"]["ingest_relevance"] == 0.2
    assert row["fields"]["description"] == "A diagnostic diagram"
    assert "image_data" not in row["fields"]
    assert "img.image_data = null" in query

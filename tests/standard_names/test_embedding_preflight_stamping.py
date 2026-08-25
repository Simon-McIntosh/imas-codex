"""Regression coverage for embedding-preflight persistence accounting."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class _RecordingGraphClient:
    """Capture persisted rows and model the timestamp side effect."""

    def __init__(self, rows: dict[str, dict[str, object]]) -> None:
        self.rows = rows
        self.persisted_batch: list[dict[str, object]] = []

    def __enter__(self) -> _RecordingGraphClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def query(self, _cypher: str, **params: object) -> list[dict[str, object]]:
        batch = params["batch"]
        assert isinstance(batch, list)
        self.persisted_batch = batch
        for item in batch:
            row = self.rows[str(item["id"])]
            row["embedding"] = item["embedding"]
            row["embedded_at"] = "timestamped"
        return []


@pytest.mark.timeout(120)
def test_preflight_timestamps_and_counts_only_written_vectors() -> None:
    """A candidate without description must not be represented as embedded."""
    from imas_codex.standard_names.review.audits import run_embedding_preflight

    rows: dict[str, dict[str, object]] = {
        "missing_description": {
            "id": "missing_description",
            "description": None,
            "embedding": None,
            "review_input_hash": None,
            "embedded_at": None,
        },
        "described_name": {
            "id": "described_name",
            "description": "A quantity with enough content to embed.",
            "embedding": None,
            "review_input_hash": None,
            "embedded_at": None,
        },
    }
    graph = _RecordingGraphClient(rows)

    def embed_available_descriptions(
        items: list[dict[str, object]],
        *,
        text_field: str,
        embedding_field: str,
    ) -> list[dict[str, object]]:
        for item in items:
            if item[text_field] is not None:
                item[embedding_field] = [0.25, 0.75]
        return items

    with (
        patch(
            "imas_codex.embeddings.description.embed_descriptions_batch",
            side_effect=embed_available_descriptions,
        ),
        patch("imas_codex.graph.client.GraphClient", return_value=graph),
    ):
        report = run_embedding_preflight(list(rows.values()))

    assert rows["missing_description"]["embedded_at"] is None
    assert report.refreshed_count == 1
    assert [item["id"] for item in graph.persisted_batch] == ["described_name"]

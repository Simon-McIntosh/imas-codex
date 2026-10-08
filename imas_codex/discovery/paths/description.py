"""Describe facility paths before their separate relevance judgment."""

from __future__ import annotations

import json
from typing import Any

from pydantic import model_validator

from imas_codex.discovery.paths.models import PathDescriptionBatch


class PathDescriptionResponse(PathDescriptionBatch):
    """Accept the path-keyed object also emitted by non-strict model endpoints."""

    @model_validator(mode="before")
    @classmethod
    def accept_path_keyed_answer(cls, value: Any) -> Any:
        if (
            isinstance(value, dict)
            and "results" not in value
            and all(
                isinstance(path, str) and isinstance(description, str)
                for path, description in value.items()
            )
        ):
            return {
                "results": [
                    {"path": path, "description": description}
                    for path, description in value.items()
                ]
            }
        return value


async def describe_paths(
    paths: list[dict],
    *,
    model: str,
    focus: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[PathDescriptionBatch, float, int]:
    """Return one factual description for each input path."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    if not paths:
        return PathDescriptionBatch(results=[]), 0.0, 0

    batch, cost, tokens = await acall_llm_structured(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Describe each directory from its evidence in one factual sentence. "
                    'Return only a JSON object shaped {"results": '
                    '[{"path": "input path", "description": "factual sentence"}]}, '
                    "with one entry per input path. Do not score, classify, or "
                    "decide whether to explore it."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"focus": focus, "directories": paths}, default=str
                ),
            },
        ],
        response_model=PathDescriptionResponse,
        service="facility-discovery",
        reasoning_effort=reasoning_effort,
    )
    answered = [item.path for item in batch.results]
    expected = [row["path"] for row in paths]
    if len(answered) != len(expected) or set(answered) != set(expected):
        raise ValueError("Path descriptions do not match the input paths")
    return batch, cost, tokens

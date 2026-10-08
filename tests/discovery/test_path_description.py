"""The path description call keeps each answer attached to its input path."""

import json

import pytest

from imas_codex.discovery.paths.models import PathDescriptionBatch


@pytest.mark.asyncio
async def test_parallel_description_prompt_names_batch_shape(monkeypatch):
    from imas_codex.discovery.base import judgment, llm
    from imas_codex.discovery.paths.parallel import _async_score_with_llm

    paths = [{"path": "/analysis/src", "total_files": 3}]

    async def describe(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        assert '"results"' in prompt
        assert '"path"' in prompt
        assert '"description"' in prompt
        assert "one entry per input path" in prompt
        return (
            PathDescriptionBatch.model_validate(
                {"results": [{"path": paths[0]["path"], "description": "Source files"}]}
            ),
            0.01,
            10,
        )

    async def judge(rows, _state_for, _questions_for, _apply, **_kwargs):
        assert rows[0]["description"] == "Source files"
        return [], 0.0, []

    monkeypatch.setattr(llm, "acall_llm_structured", describe)
    monkeypatch.setattr(judgment, "judge_rows", judge)
    results, cost = await _async_score_with_llm(paths)
    assert results == []
    assert cost == 0.01


@pytest.mark.asyncio
async def test_path_keyed_answer_replays_as_one_description_per_path(monkeypatch):
    from imas_codex.discovery.base import llm
    from imas_codex.discovery.paths.description import describe_paths

    paths = [{"path": "/analysis/src"}, {"path": "/analysis/data"}]
    answer = {
        "/analysis/src": "Source files",
        "/analysis/data": "Measured data",
    }

    async def replay(**kwargs):
        batch = llm._parse_structured_content(
            json.dumps(answer), kwargs["response_model"], kwargs["model"]
        )
        return batch, 0.01, 12

    monkeypatch.setattr(llm, "acall_llm_structured", replay)
    batch, cost, tokens = await describe_paths(paths, model="test-model")
    assert isinstance(batch, PathDescriptionBatch)
    assert [(item.path, item.description) for item in batch.results] == list(
        answer.items()
    )
    assert (cost, tokens) == (0.01, 12)


@pytest.mark.asyncio
async def test_description_refuses_missing_or_repeated_paths(monkeypatch):
    from imas_codex.discovery.base import llm
    from imas_codex.discovery.paths.description import describe_paths

    async def wrong(**kwargs):
        batch = kwargs["response_model"].model_validate(
            {
                "results": [
                    {"path": "/analysis/src", "description": "First"},
                    {"path": "/analysis/src", "description": "Again"},
                ]
            }
        )
        return batch, 0.0, 1

    monkeypatch.setattr(llm, "acall_llm_structured", wrong)
    with pytest.raises(ValueError, match="do not match"):
        await describe_paths(
            [{"path": "/analysis/src"}, {"path": "/analysis/data"}], model="test-model"
        )

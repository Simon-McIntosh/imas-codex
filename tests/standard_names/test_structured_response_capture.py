"""Regression coverage for recoverable structured provider responses."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from imas_codex.discovery.base.llm import call_llm_structured


class _Candidate(BaseModel):
    source_id: str
    spelling: str

    def render(self) -> str:
        raise RuntimeError("candidate cannot be rendered")


class _CandidateBatch(BaseModel):
    candidates: list[_Candidate]


class _Usage:
    prompt_tokens = 12
    completion_tokens = 8
    prompt_tokens_details = None


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()
        self.model = "test-model"
        self._hidden_params = {"response_cost": 0.0}


def test_completed_response_remains_recoverable_when_candidate_rendering_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-test")
    content = (
        '{"candidates": ['
        '{"source_id": "dd:first", "spelling": "first_candidate"}, '
        '{"source_id": "dd:second", "spelling": "second_candidate"}'
        "]}"
    )

    with patch("litellm.completion", return_value=_Response(content)):
        result = call_llm_structured(
            model="openrouter/test/model",
            messages=[{"role": "user", "content": "compose candidates"}],
            response_model=_CandidateBatch,
            service="standard-names",
            max_retries=1,
        )

    with pytest.raises(RuntimeError, match="candidate cannot be rendered"):
        result.parsed.candidates[0].render()

    recovered = _CandidateBatch.model_validate_json(result.raw_response_json)
    assert [candidate.source_id for candidate in recovered.candidates] == [
        "dd:first",
        "dd:second",
    ]
    assert recovered == result.parsed

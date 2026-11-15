"""Each model seat answers one live request through the lane it routes to.

These tests spend provider money and need network access plus the project's
credentials, so they carry the ``slow`` marker and stay out of the default run:
``pytest tests/config/test_model_seat_reachability.py -m slow``.

The seat list is the configuration's own, read through the accessors a
production caller uses, so a seat pointed at a model its route does not serve
fails here rather than at the start of a run. That is the live arm of the
catalog checks in ``test_model_seats.py``: the snapshot proves each seat names
a listed release, and these tests prove the lane still serves it.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from imas_codex.discovery.base.llm import (
    call_decisions,
    call_llm,
    call_llm_structured,
)
from imas_codex.settings import (
    _get_section,
    get_model,
    get_sn_benchmark_candidate_models,
    get_sn_benchmark_compose_models,
    get_sn_benchmark_reviewer_model,
    get_sn_benchmark_reviewer_models,
    get_sn_review_docs_models,
    get_sn_review_names_models,
    get_sn_review_profile_models,
)

pytestmark = pytest.mark.slow

PROMPT = [{"role": "user", "content": "Reply with the single word: ok"}]
QUESTIONS = {
    "role": {
        "type": "choice",
        "instructions": "Which role best describes the file?",
        "criteria": {
            "diagnostic_data_access": "reads shot data",
            "infrastructure_or_utility": "build files and utilities",
        },
    },
}


@pytest.fixture(autouse=True)
def no_live_decisions_requests():
    """Let this module reach the live decisions endpoint.

    The suite-wide guard refuses every live request to the paid decisions
    endpoint; this module is the one place allowed to spend on it, because the
    served-release probe here is the only evidence the decisions seats answer.
    """
    yield


def _chat_models() -> list[str]:
    models = [
        get_model(section)
        for section in (
            "sn-docs",
            "sn-refine",
            "sn-escalation",
            "sn-parent-enrich",
            "sn-classifier",
            "sn-prose-adjudicator",
            "sn-release-notes",
            "dd-enrichment",
            "reasoning",
            "cluster-labels",
            "ids-mapping",
            "discovery-triage",
            "discovery-score",
            "sn-compose",
        )
    ]
    models += list(get_sn_review_names_models())
    models += list(get_sn_review_docs_models())
    for profile in ("default", "local-only", "quality-cost-balanced", "opus-only"):
        models += list(get_sn_review_profile_models(profile))
    models += list(get_sn_benchmark_compose_models())
    models += list(get_sn_benchmark_candidate_models())
    models += list(get_sn_benchmark_reviewer_models())
    models.append(get_sn_benchmark_reviewer_model())
    proposer = _get_section("sn-fanout").get("proposer-model")
    if isinstance(proposer, str):
        models.append(proposer)
    return sorted(
        {model for model in models if not model.startswith(("typesafe/", "Qwen/"))}
    )


def _decisions_models() -> list[str]:
    return sorted(
        {
            get_model(section)
            for section in (
                "discovery-relevance",
                "mapping-candidates",
                "sn-judge",
                "sn-domain-tree",
            )
        }
    )


class _Reply(BaseModel):
    word: str


@pytest.mark.parametrize("model", _chat_models())
def test_chat_seat_answers_one_live_request(model: str) -> None:
    """The route a chat seat names accepts its model and returns a reply."""
    if model.startswith("local/"):
        # The local lane is served over the OpenAI-compatible client that the
        # structured call dispatches to; litellm refuses the ``local/`` prefix
        # outright, so this is the path a production local seat takes.
        result = call_llm_structured(
            model,
            [
                {
                    "role": "user",
                    "content": "Reply with one JSON object holding the word ok.",
                }
            ],
            _Reply,
            max_tokens=256,
            temperature=0.0,
            timeout=300,
            max_retries=2,
            service="seat-reachability",
        )
        parsed = result.parsed
        assert parsed.word, f"the lane serving {model!r} returned an empty reply"
        return
    response, _cost = call_llm(
        model,
        PROMPT,
        max_tokens=16,
        temperature=0.0,
        timeout=180,
        max_retries=2,
        service="seat-reachability",
    )
    content = response.choices[0].message.content
    served = getattr(response, "model", None) or model
    assert content is not None or response.choices[0].finish_reason, (
        f"the route serving {model!r} returned neither content nor a finish "
        f"reason; it answered as {served!r}"
    )


@pytest.mark.parametrize("model", _decisions_models())
def test_decisions_seat_answers_one_live_request(model: str) -> None:
    """The decisions endpoint serves the model a judgement seat names."""
    answers, cost = call_decisions(
        model,
        {"facility": "iter", "file": "seat-reachability.pro"},
        QUESTIONS,
        service="seat-reachability",
    )
    assert answers and answers["role"].get("choice"), (
        f"the decisions endpoint returned no answer for {model!r}"
    )
    assert cost >= 0.0, f"the decisions endpoint reported a negative cost for {model!r}"

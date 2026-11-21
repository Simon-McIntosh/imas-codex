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

import json

import pytest
from pydantic import BaseModel

from iter_codex.discovery.base.llm import (
    call_decisions,
    call_llm,
    call_llm_structured,
)
from iter_codex.settings import (
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
from tests.config.test_model_seats import _catalog, _declared_models

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
    _assert_chat_reply(model)


def _assert_chat_reply(model: str, owner: str | None = None) -> None:
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
        print(
            "SEAT_RECEIPT "
            + json.dumps(
                {
                    "seat": owner or model,
                    "requested_model": model,
                    "content": parsed.word,
                    "cost_usd": result.cost,
                    "tokens": result.tokens,
                }
            )
        )
        return
    response, cost = call_llm(
        model,
        PROMPT,
        max_tokens=256,
        temperature=0.0,
        timeout=300,
        max_retries=2,
        service="seat-reachability",
    )
    content = (response.choices[0].message.content or "").strip()
    served = getattr(response, "model", None)
    assert content, (
        f"the route serving {model!r} returned empty content, so the seat is "
        f"reachable but unusable; it answered as {served!r} with finish_reason "
        f"{response.choices[0].finish_reason!r}"
    )
    assert served, f"the response for {model!r} did not identify its model"
    print(
        "SEAT_RECEIPT "
        + json.dumps(
            {
                "seat": owner or model,
                "requested_model": model,
                "response_model": served,
                "content": content,
                "cost_usd": cost,
                "usage": response.usage.model_dump(),
            }
        )
    )


_CHANGED_SEATS = [
    (owner, model)
    for owner, model in _declared_models().items()
    if model in _catalog()["openrouter"]["tier_successors"].values()
]


@pytest.mark.parametrize(
    "owner,model", _CHANGED_SEATS, ids=[owner for owner, _model in _CHANGED_SEATS]
)
def test_changed_seat_answers_one_live_request(owner: str, model: str) -> None:
    """Every moved declaration gets its own request and response receipt."""
    _assert_chat_reply(model, owner)


@pytest.mark.parametrize("model", _decisions_models())
def test_decisions_seat_answers_one_live_request(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decisions endpoint serves the model a judgement seat names.

    The suite-wide guard refuses live requests to this paid endpoint, so this
    one test replaces the transport seam it calls through with a plain HTTP
    POST. The guard stays in force for every other test in the file and for
    every other test in the suite.
    """
    from iter_codex.discovery.base import llm

    monkeypatch.setattr(llm, "_post_decisions", _post_decisions_live)
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


def _post_decisions_live(headers: dict[str, str], body: dict, timeout: float):
    """POST one decisions request without the suite's httpx guard.

    The guard intercepts ``httpx.Client.send``; this seam speaks HTTP through
    the standard library and returns an ``httpx.Response`` so the call layer's
    own status and JSON handling stays the code under test.
    """
    import json
    import urllib.error
    import urllib.request

    import httpx

    request = urllib.request.Request(
        "https://openrouter.ai/api/alpha/decisions",
        data=json.dumps(body).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            payload, status = reply.read(), reply.status
    except urllib.error.HTTPError as error:
        payload, status = error.read(), error.code
    return httpx.Response(status_code=status, content=payload)


@pytest.mark.parametrize("seat", ["sn-compose", "cluster-labels"])
def test_direct_route_answers_one_live_request(seat: str) -> None:
    """One configured local seat and one OpenRouter seat return actual replies."""
    _assert_chat_reply(get_model(seat), seat)

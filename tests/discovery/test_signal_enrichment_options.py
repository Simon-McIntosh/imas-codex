"""Local enrichment options, transport constraints and bounded validation repair."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from imas_codex.discovery.base import llm
from imas_codex.discovery.signals.enrichment import (
    EnrichmentOptions,
    fit_enrichment_prompt,
    get_enrichment_options,
)
from imas_codex.discovery.signals.models import SignalEnrichmentBatch
from tests.discovery.test_signal_enrichment_benchmark import answer


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"schema-constraint": True}, {"local_schema_constraint": True}),
        ({"json-mode": True}, {"local_json_mode": True}),
        ({"validation-reask": True}, {"validation_reask": True}),
        ({"thinking": True}, {"reasoning_effort": "high"}),
        ({"thinking": False}, {"reasoning_effort": None}),
        ({"max-tokens": 32768}, {"max_tokens": 32768}),
        ({"batch-size": 3}, {}),
        ({"fitted-prompt": True}, {}),
    ],
)
def test_each_seat_option_is_explicit_and_other_models_keep_defaults(
    settings, expected
):
    with patch(
        "imas_codex.settings._get_section", return_value={"signal-enrichment": settings}
    ):
        options = get_enrichment_options("local/test")
        assert options.call_kwargs() == expected
        assert options.model_dump(by_alias=True) | settings == options.model_dump(
            by_alias=True
        )
        assert get_enrichment_options("openrouter/test") == EnrichmentOptions()
    assert EnrichmentOptions().call_kwargs() == {}


@pytest.mark.parametrize(
    "settings", [{"batch-size": 0}, {"max-tokens": 0}, {"unknown": True}]
)
def test_invalid_seat_options_are_refused(settings):
    with pytest.raises(ValueError):
        EnrichmentOptions.model_validate(settings)


def test_fitted_prompt_retains_task_enums_valid_wrapper_and_schema_last():
    prompt = fit_enrichment_prompt("FROZEN FULL TASK AND RETRIEVAL RULES")
    assert "FROZEN FULL TASK AND RETRIEVAL RULES" in prompt
    assert '"plant_systems"' in prompt
    assert '"keywords": [' in prompt
    assert '"results": [' in prompt
    schema = json.loads(prompt.split("Use exactly this schema:")[-1])
    assert schema == SignalEnrichmentBatch.model_json_schema()
    assert '"plant_systems"' in prompt.split("`physics_domain`:")[1].split("\n")[0]


def response(content):
    return SimpleNamespace(
        model="local/test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
    )


async def request(contents, **options):
    calls = []

    async def transport(kwargs):
        calls.append(llm._local_call_kwargs(kwargs))
        return response(contents.pop(0))

    with (
        patch.object(llm, "get_api_key", return_value="test"),
        patch.object(
            llm,
            "_build_kwargs",
            return_value={
                "api_base": "http://localhost:8000/v1",
                "model": "local/test",
                "messages": [{"role": "user", "content": "Original full context"}],
                "response_format": llm._to_json_schema_format(SignalEnrichmentBatch),
            },
        ),
        patch.object(llm, "_is_local_api_base", return_value=True),
        patch.object(llm, "_acompletion_local", side_effect=transport),
    ):
        try:
            result = await llm.acall_llm_structured(
                model="local/test",
                messages=[],
                response_model=SignalEnrichmentBatch,
                max_retries=5,
                service="facility-discovery",
                **options,
            )
            return result, calls
        except ValueError as exc:
            return exc, calls


@pytest.mark.asyncio
async def test_schema_constraint_reaches_actual_local_wire_options_only_when_enabled():
    _, baseline = await request([json.dumps(answer())])
    _, constrained = await request([json.dumps(answer())], local_schema_constraint=True)
    assert "extra_body" not in baseline[0]
    assert constrained[0]["extra_body"]["response_format"]["type"] == "json_schema"
    schema = constrained[0]["extra_body"]["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["PhysicsDomain"]["enum"]


@pytest.mark.asyncio
async def test_json_mode_reaches_local_wire_and_mutually_exclusive_constraints_are_refused():
    result, calls = await request([json.dumps(answer())], local_json_mode=True)
    assert isinstance(result, llm.LLMResult)
    assert calls[0]["extra_body"]["response_format"] == {"type": "json_object"}
    with pytest.raises(ValueError, match="choose either"):
        EnrichmentOptions(schema_constraint=True, json_mode=True)
    result, calls = await request(
        [json.dumps(answer())], local_json_mode=True, local_schema_constraint=True
    )
    assert isinstance(result, ValueError)
    assert calls == []


@pytest.mark.asyncio
async def test_reask_quotes_validation_error_and_accounts_for_both_calls():
    invalid = answer()
    invalid["results"][0]["keywords"] = "temperature"
    result, calls = await request(
        [json.dumps(invalid), json.dumps(answer())], validation_reask=True
    )
    assert len(calls) == 2
    assert result.response_count == 2
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert calls[1]["messages"][0]["content"] == "Original full context"
    assert calls[1]["messages"][1] == {
        "role": "assistant",
        "content": json.dumps(invalid),
    }
    assert "results.0.keywords" in calls[1]["messages"][2]["content"]
    assert "list_type" in calls[1]["messages"][2]["content"]


@pytest.mark.asyncio
async def test_twice_invalid_response_is_failed_and_never_delivered():
    result, calls = await request(["not JSON", "not JSON"], validation_reask=True)
    assert isinstance(result, ValueError)
    assert len(calls) == 2
    assert result.response_count == 2
    assert result.output_tokens == 20
    assert not isinstance(result, llm.LLMResult)


@pytest.mark.asyncio
async def test_raw_fences_trigger_reask_even_when_ordinary_parser_can_recover():
    fenced = "```json\n" + json.dumps(answer()) + "\n```"
    result, baseline = await request([fenced])
    assert isinstance(result, llm.LLMResult)
    result, repair = await request(
        [fenced, json.dumps(answer())], validation_reask=True
    )
    assert isinstance(result, llm.LLMResult)
    assert len(baseline) == 1
    assert len(repair) == 2


@pytest.mark.asyncio
async def test_twice_invalid_worker_response_produces_no_private_description(tmp_path):
    from scripts import benchmark_signal_enrichment as bench

    calls = []

    async def transport(kwargs):
        calls.append(kwargs)
        return response("not JSON")

    signal = {
        "id": "jt-60sa:synthetic",
        "accessor": "COOLING:TEMP",
        "name": "COOLING:TEMP",
        "unit": "K",
    }
    db = bench.open_store(tmp_path / "private.sqlite")
    with (
        patch.object(llm, "get_api_key", return_value="test"),
        patch.object(
            llm,
            "_build_kwargs",
            return_value={
                "api_base": "http://localhost:8000/v1",
                "model": "local/test",
                "messages": [{"role": "user", "content": "Catalogue temperature"}],
                "response_format": llm._to_json_schema_format(SignalEnrichmentBatch),
            },
        ),
        patch.object(llm, "_is_local_api_base", return_value=True),
    ):
        record = await bench.replay_batch(
            [signal],
            "jt-60sa",
            db,
            transport=transport,
            enrichment_options=EnrichmentOptions(validation_reask=True),
            call_options={"validation_reask": True},
        )
    assert len(calls) == 2
    assert record["outputs"] == []
    assert all(not r["score"]["parse_success"] for r in record["responses"])
    db.close()


def test_smaller_batches_preserve_shared_retrieval_and_selected_context():
    from scripts import (
        benchmark_signal_enrichment as bench,
        refine_signal_enrichment as refine,
    )

    messages = [
        {"role": "system", "content": "Unchanged task"},
        {
            "role": "user",
            "content": """Classify these 3 signals.
## Facility Wiki Reference
Shared authoritative evidence.
## EDAS Category: coil
Code: read_current(channel)
### Signal 1
accessor: COIL:CURRENT
source_description: Coil current
### Signal 2
accessor: COIL:VOLTAGE
source_description: Coil voltage
## EDAS Category: cooling
Wiki: circuit evidence
### Signal 3
accessor: WATER:TEMP
source_description: Cooling temperature
""",
        },
    ]
    selected = refine.subset_messages(messages, [1, 3])
    assert selected[0] == messages[0]
    assert "Classify these 2 signals." in selected[1]["content"]
    assert "COIL:VOLTAGE" not in selected[1]["content"]
    assert "### Signal 2\naccessor: WATER:TEMP" in selected[1]["content"]
    assert "Classify these 3 signals." in messages[1]["content"]
    for accessor in ("COIL:CURRENT", "WATER:TEMP"):
        assert refine.normalized_context(
            bench.target_context(selected, accessor)
        ) == refine.normalized_context(bench.target_context(messages, accessor))


def test_repaired_success_never_improves_raw_parse_rate_or_hides_failed_signals():
    from scripts import (
        benchmark_signal_enrichment as bench,
        refine_signal_enrichment as refine,
    )

    good = {
        "raw": json.dumps(answer()),
        "score": bench.score_response(json.dumps(answer()), 1),
    }
    bad = {
        "raw": "not JSON",
        "score": bench.score_response("not JSON", 1),
        "error": "Invalid JSON",
    }
    base = {
        "arm": "repair",
        "seconds": 1,
        "input_tokens": 2,
        "output_tokens": 3,
        "cost_usd": 0,
    }
    records = [
        {**base, "ids": ["ok"], "responses": [bad, good], "outputs": [{"id": "ok"}]},
        {**base, "ids": ["failed"], "responses": [bad, bad], "outputs": []},
    ]
    summary = refine.summarize(records, {"order": ["ok", "failed"]})["repair"]
    assert summary["raw_parse_rate"] == 0
    assert summary["delivered_after_reask"] == 1
    assert summary["delivered"] == 1
    assert summary["delivery_rate"] == 0.5
    assert summary["failed_signals_by_class"] == {"invalid_json": 1}
    assert summary["total_calls"] == 4


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        ("results.0.physics_domain: invalid enum", "enumeration"),
        ("results.0.keywords: list_type", "array_type"),
    ],
)
def test_raw_fences_do_not_hide_terminal_field_validation_failures(error, cause):
    from scripts import refine_signal_enrichment as refine

    response = {
        "raw": "```json\n{}\n```",
        "score": {"error": "Invalid JSON json_invalid"},
        "error": error,
    }
    assert refine.failure_class(response) == "invalid_json"
    assert refine.failure_class(response, terminal=True) == cause


def test_partial_coverage_and_server_abort_have_distinct_failure_counts():
    from scripts import refine_signal_enrichment as refine

    base = {
        "arm": "trial",
        "seconds": 1,
        "input_tokens": 2,
        "output_tokens": 3,
        "cost_usd": 0,
    }
    records = [
        {
            **base,
            "ids": ["delivered", "missing"],
            "responses": [
                {
                    "raw": "[]",
                    "score": {"parse_success": False, "error": "schema wrapper absent"},
                }
            ],
            "outputs": [{"id": "delivered"}],
        },
        {
            **base,
            "ids": ["aborted"],
            "responses": [
                {
                    "raw": "{",
                    "score": {"parse_success": False, "error": "json_invalid"},
                    "finish_reason": "abort",
                    "error": "abort",
                }
            ],
            "outputs": [],
        },
    ]
    result = refine.summarize(records, {"order": ["delivered", "missing", "aborted"]})[
        "trial"
    ]
    assert result["failed_signals_by_class"] == {"coverage": 1, "provider_abort": 1}

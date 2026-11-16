"""Context references, local review refusal and opt-in description controls."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.discovery.signals.enrichment import (
    CitedEnrichmentBatch,
    CitedStatement,
    DescriptionReview,
    EnrichmentOptions,
    context_items,
    enrich_signal_batch,
    filter_statements,
)
from imas_codex.discovery.signals.models import SignalEnrichmentBatch
from tests.discovery.test_signal_enrichment_benchmark import answer


def messages():
    return [
        {"role": "system", "content": "Original full task"},
        {"role": "user", "content": "source_description: Cooling temperature\nunit: K"},
    ]


def draft(*, cited=False):
    value = answer()
    value["results"][0]["description"] = (
        "Cooling temperature. Unsupported circuit detail."
    )
    if cited:
        value["results"][0]["statements"] = [
            {"text": "Cooling temperature.", "context_items": ["context-1"]},
            {"text": "Unsupported circuit detail.", "context_items": ["context-999"]},
            {"text": "Uncited detail.", "context_items": []},
        ]
    return (CitedEnrichmentBatch if cited else SignalEnrichmentBatch).model_validate(
        value
    )


async def run(options, results=None, model="local/test"):
    call = AsyncMock(
        side_effect=results or [(draft(cited=options.claim_evidence), 0.1, 10)]
    )
    output = await enrich_signal_batch(
        model=model,
        messages=messages(),
        options=options,
        call=call,
        temperature=0.3,
        service="facility-discovery",
    )
    return output, call


def test_nonexistent_context_item_is_dropped_and_counted():
    items = context_items("Cooling temperature\n\nunit: K")
    kept, drops = filter_statements(
        [
            CitedStatement(text="Cooling temperature.", context_items=["context-1"]),
            CitedStatement(text="Invented circuit.", context_items=["context-999"]),
            CitedStatement(text="Uncited circuit."),
            CitedStatement(
                text="Mixed citation.", context_items=["context-1", "missing"]
            ),
        ],
        items,
    )
    assert kept == ["Cooling temperature."]
    assert drops == 3
    assert items["context-3"] == "unit: K"


@pytest.mark.asyncio
async def test_cited_description_cannot_bypass_filter_with_uncited_prose():
    (batch, _, _), call = await run(EnrichmentOptions(claim_evidence=True))
    assert batch.results[0].description == "Cooling temperature."
    assert batch.evidence_drops == 2
    assert (
        "[context-1] source_description:"
        in call.call_args.kwargs["messages"][1]["content"]
    )
    assert call.call_args.kwargs["response_model"] is CitedEnrichmentBatch


@pytest.mark.asyncio
async def test_all_claims_dropped_delivers_no_unsupported_text():
    value = draft(cited=True)
    value.results[0].statements[0].context_items = ["absent"]
    (batch, _, _), _ = await run(
        EnrichmentOptions(claim_evidence=True), [(value, 0, 1)]
    )
    assert batch.results[0].description == ""
    assert batch.results[0].context_quality.value == "low"
    assert batch.evidence_drops == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [True, False])
async def test_separate_review_removes_claim_and_accounts_for_both_calls(thinking):
    review = DescriptionReview.model_validate(
        {
            "results": [
                {
                    "signal_index": 1,
                    "statements": [
                        {
                            "draft_index": 1,
                            "text": "Cooling temperature in kelvin.",
                            "context_items": ["context-1", "context-2"],
                        }
                    ],
                }
            ]
        }
    )
    (batch, cost, tokens), call = await run(
        EnrichmentOptions(review=True, review_thinking=thinking),
        [(draft(), 0.1, 10), (review, 0.2, 20)],
    )
    assert batch.results[0].description == "Cooling temperature in kelvin."
    assert batch.review_drops == 1
    assert batch.review_revisions == 1
    assert cost == pytest.approx(0.3)
    assert tokens == 30
    assert call.await_count == 2
    request = call.call_args.kwargs
    assert request["reasoning_effort"] == ("high" if thinking else None)
    context = json.loads(request["messages"][1]["content"])
    assert (
        context["context_items"]["context-1"]
        == "source_description: Cooling temperature"
    )
    assert context["drafts"]["1"][1] == "Unsupported circuit detail."


@pytest.mark.asyncio
async def test_review_failure_refuses_draft_instead_of_delivering_it():
    with pytest.raises(ValueError, match="review unavailable"):
        await run(
            EnrichmentOptions(review=True),
            [(draft(), 0, 1), ValueError("review unavailable")],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[], [1, 1], [2]])
async def test_incomplete_duplicate_or_unknown_review_signal_is_refused(rows):
    review = DescriptionReview(
        results=[{"signal_index": index, "statements": []} for index in rows]
    )
    with pytest.raises(ValueError, match="review"):
        await run(EnrichmentOptions(review=True), [(draft(), 0, 1), (review, 0, 1)])


@pytest.mark.asyncio
@pytest.mark.parametrize("indices", [[0], [3], [1, 1]])
async def test_review_cannot_add_or_duplicate_a_draft_statement(indices):
    review = DescriptionReview(
        results=[
            {
                "signal_index": 1,
                "statements": [
                    {
                        "draft_index": index,
                        "text": "Guess.",
                        "context_items": ["context-1"],
                    }
                    for index in indices
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="draft statements"):
        await run(EnrichmentOptions(review=True), [(draft(), 0, 1), (review, 0, 1)])


@pytest.mark.asyncio
async def test_temperature_and_length_options_are_independent_and_defaults_preserve_prompt():
    (batch, _, _), call = await run(EnrichmentOptions())
    assert batch.results[0].description == draft().results[0].description
    assert call.call_args.kwargs["messages"] == messages()
    (batch, _, _), call = await run(
        EnrichmentOptions(temperature=0, description_sentences=1)
    )
    assert batch.results[0].description == "Cooling temperature."
    assert batch.length_drops == 1
    assert call.call_args.kwargs["temperature"] == 0
    assert "at most 1 short sentence" in call.call_args.kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_grounding_instruction_can_be_removed_without_changing_context():
    _, grounded = await run(EnrichmentOptions(grounding=True))
    _, ungrounded = await run(EnrichmentOptions())
    assert (
        "Say less when that context says less"
        in grounded.call_args.kwargs["messages"][0]["content"]
    )
    assert "Say less when" not in ungrounded.call_args.kwargs["messages"][0]["content"]
    assert (
        grounded.call_args.kwargs["messages"][1]
        == ungrounded.call_args.kwargs["messages"][1]
    )


@pytest.mark.asyncio
async def test_paid_model_is_refused_for_grounding_or_review():
    with pytest.raises(ValueError, match="local model"):
        await run(EnrichmentOptions(review=True), model="openrouter/test")


def test_bracketed_identifiers_refer_to_the_same_supplied_context_item():
    kept, drops = filter_statements(
        [
            CitedStatement(text="Temperature.", context_items=["[context-1]"]),
            CitedStatement(text="Guess.", context_items=["[context-999]"]),
        ],
        context_items("Temperature"),
    )
    assert kept == ["Temperature."]
    assert drops == 1


@pytest.mark.asyncio
async def test_unconfigured_seat_preserves_description_whitespace():
    value = draft()
    value.results[0].description = "  Temperature.\n\n  Source detail.  "
    (batch, _, _), _ = await run(EnrichmentOptions(), [(value, 0, 1)])
    assert batch.results[0].description == value.results[0].description


def test_empty_citation_results_are_not_counted_as_delivered():
    from scripts.refine_signal_enrichment import summarize

    records = [
        {
            "arm": "empty",
            "ids": ["signal"],
            "seconds": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_usd": 0,
            "responses": [{"raw": "{}", "score": {"parse_success": True}}],
            "outputs": [{"id": "signal", "description": ""}],
        }
    ]
    result = summarize(records, {"order": ["signal"]})["empty"]
    assert result["delivered"] == 0
    assert result["empty_descriptions"] == 1
    assert result["delivery_rate"] == 0
    assert result["delivered_first"] == 0
    assert result["failed_signals_by_class"] == {"empty_description": 1}


def test_review_raw_score_checks_the_review_schema_without_draft_errors():
    from scripts.benchmark_signal_enrichment import score_response

    raw = DescriptionReview(
        results=[{"signal_index": 1, "statements": []}]
    ).model_dump_json()
    assert score_response(raw, 1)["parse_success"] is False
    score = score_response(raw, 1, response_model=DescriptionReview)
    assert score["parse_success"] is True
    assert score["complete"] is True
    assert "error" not in score
    assert (
        score_response("not JSON", 1, response_model=DescriptionReview)["parse_success"]
        is False
    )


def test_manual_seal_refuses_missing_readings_and_later_edits(tmp_path):
    from scripts.assess_signal_enrichment_grounding import seal, summarize

    (tmp_path / "sheets.json").write_text(json.dumps([{"packet": "blind", "rows": 1}]))
    (tmp_path / "readings.jsonl").write_text("")
    with pytest.raises(ValueError, match="exactly once"):
        seal(tmp_path)
    row = {
        "packet": "blind",
        "index": 0,
        "preference": "first",
        "reason": "Source states temperature.",
        "quality": {"first": "right", "second": "wrong"},
    }
    (tmp_path / "readings.jsonl").write_text(json.dumps(row) + "\n")
    seal(tmp_path)
    (tmp_path / "readings.jsonl").write_text(json.dumps(row) + "\n\n")
    with pytest.raises(ValueError, match="changed after sealing"):
        summarize(tmp_path)


def test_review_call_is_not_a_provider_retry_or_draft_reask():
    from scripts.refine_signal_enrichment import replacement_kind, summarize

    record = {
        "arm": "review",
        "ids": ["signal"],
        "seconds": 2,
        "input_tokens": 3,
        "output_tokens": 4,
        "cost_usd": 0,
        "responses": [
            {
                "raw": "{}",
                "response_schema": "SignalEnrichmentBatch",
                "score": {"parse_success": True},
            },
            {
                "raw": "{}",
                "response_schema": "DescriptionReview",
                "score": {"parse_success": False},
            },
            {
                "raw": "{}",
                "response_schema": "DescriptionReview",
                "validation_reask": True,
                "score": {"parse_success": True},
            },
        ],
        "outputs": [{"id": "signal", "description": "Temperature."}],
    }
    assert replacement_kind(record) is None
    result = summarize([record], {"order": ["signal"]})["review"]
    assert result["total_calls"] == 3
    assert result["review_calls"] == 2
    assert result["review_raw_parses"] == 0
    assert result["review_validation_reasks"] == 1
    assert result["other_retries"] == 0
    assert result["validation_reasks"] == 0


@pytest.mark.asyncio
async def test_production_private_sink_never_receives_missing_citation_statement(
    tmp_path,
):
    from imas_codex.discovery.base import llm
    from scripts import benchmark_signal_enrichment as bench
    from tests.discovery.test_signal_enrichment_options import response

    raw = draft(cited=True).model_dump_json()
    db = bench.open_store(tmp_path / "private.sqlite")
    with (
        patch.object(llm, "get_api_key", return_value="test"),
        patch.object(llm, "_is_local_api_base", return_value=True),
        patch.object(
            llm,
            "_build_kwargs",
            return_value={
                "api_base": "http://localhost:8000/v1",
                "model": "local/test",
                "messages": messages(),
                "response_format": llm._to_json_schema_format(CitedEnrichmentBatch),
            },
        ),
    ):
        record = await bench.replay_batch(
            [
                {
                    "id": "jt-60sa:synthetic",
                    "name": "TEMP",
                    "accessor": "TEMP",
                    "unit": "K",
                }
            ],
            "jt-60sa",
            db,
            transport=AsyncMock(return_value=response(raw)),
            messages=messages(),
            enrichment_options=EnrichmentOptions(claim_evidence=True),
        )
    db.close()
    assert len(record["outputs"]) == 1
    assert record["outputs"][0]["description"] == "Cooling temperature."
    assert record["grounding_counts"]["evidence_drops"] == 2
    assert "Unsupported circuit detail" not in json.dumps(record["outputs"])


@pytest.mark.asyncio
async def test_review_only_keeps_the_drafting_prompt_byte_identical():
    review = DescriptionReview(results=[{"signal_index": 1, "statements": []}])
    _, call = await run(
        EnrichmentOptions(review=True), [(draft(), 0, 1), (review, 0, 1)]
    )
    assert call.call_args_list[0].kwargs["messages"] == messages()
    assert call.call_args_list[0].kwargs["temperature"] == 0.3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [EnrichmentOptions(grounding=True), EnrichmentOptions(description_sentences=1)],
)
async def test_instruction_only_options_do_not_add_a_second_output_schema(options):
    _, call = await run(options)
    prompt = call.call_args.kwargs["messages"][0]["content"]
    assert "Original full task" in prompt
    assert "supersedes the schema above" not in prompt


def test_reading_context_keeps_support_retrieved_under_another_group():
    from scripts.refine_signal_enrichment import supplied_target_context

    source = [
        {
            "role": "user",
            "content": """Classify these 2 signals.
## EDAS Category: analysis
Wiki: FAME computes equilibria on a ten-millisecond grid after the discharge.
### Signal 1
accessor: FIT:FLAG
source_description: An unrelated signal field
## EDAS Category: FAME
Wiki: Database access conventions
### Signal 2
accessor: FAME:ANGLE
source_description: Field-line angle
""",
        }
    ]
    context = supplied_target_context(source, "FAME:ANGLE")
    assert (
        "FAME computes equilibria on a ten-millisecond grid after the discharge"
        in context
    )
    assert "Database access conventions" in context
    assert "Field-line angle" in context
    assert "unrelated signal field" not in context
    assert "Classify these 1 signals" in context

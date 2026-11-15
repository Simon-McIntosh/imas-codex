"""Strict benchmark accounting and private production-worker replay."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import benchmark_signal_enrichment as bench


def answer():
    return {
        "results": [
            {
                "signal_index": 1,
                "physics_domain": "plant_systems",
                "name": "Temperature sensor",
                "description": "Measured temperature of the cooling circuit.",
                "context_quality": "medium",
            }
        ]
    }


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "not JSON",
        "```json\n{}\n```",
        json.dumps(answer()["results"]),
        json.dumps({"results": [{"signal_index": 1}]}),
    ],
)
def test_malformed_response_is_counted_as_failure(content):
    result = bench.score_response(content, 1)
    assert result["parse_success"] is False
    assert result["matched"] == 0


def test_valid_response_is_seen_by_the_same_instrument():
    result = bench.score_response(json.dumps(answer()), 1)
    assert result["parse_success"] is True
    assert result["matched"] == 1
    assert result["complete"] is True


def test_missing_and_repeated_indices_are_not_full_coverage():
    payload = answer()
    payload["results"] *= 2
    result = bench.score_response(json.dumps(payload), 2)
    assert result["parse_success"] is True
    assert result["matched"] == 1
    assert result["complete"] is False


def test_changed_stored_enrichment_is_refused():
    before = [{"description": "Stored description", "unit": "K"}]
    bench.assert_unchanged(before, list(before))
    after = [{"description": "Rewritten description", "unit": "K"}]
    with pytest.raises(ValueError, match="stored enrichment changed"):
        bench.assert_unchanged(before, after)


def test_inputs_exclude_stored_model_answers():
    p = {
        "id": "iter:synthetic",
        "data_source_path": "COOLING:TEMP",
        "name": "Answer name",
        "description": "Answer description",
        "physics_domain": "plant_systems",
        "keywords": ["answer"],
        "unit": "K",
        "source_description": "Catalogue temperature",
    }
    signal = bench.input_signal({"id": p["id"], "properties": p})
    assert signal["name"] == p["data_source_path"]
    assert signal["description"] == ""
    assert "physics_domain" not in signal
    assert "keywords" not in signal
    assert signal["source_description"] == p["source_description"]
    assert signal["unit"] == "K"


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_worker_replay_writes_only_private_results(tmp_path, malformed):
    content = "not JSON" if malformed else json.dumps(answer())
    calls = []

    async def transport(kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="local/test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
        )

    signal = {
        "id": "iter:synthetic",
        "facility_id": "iter",
        "name": "COOLING:TEMP",
        "accessor": "COOLING:TEMP",
        "description": "",
        "source_description": "Temperature",
        "unit": "K",
        "discovery_source": "codac_uda",
    }
    db = bench.open_store(tmp_path / "private.sqlite")
    with (
        patch.object(bench.llm, "get_api_key", return_value="test"),
        patch.object(bench.llm, "get_api_key_for_service", return_value="test"),
        patch.object(
            bench.llm,
            "_build_kwargs",
            return_value={
                "api_base": "http://localhost:8000/v1",
                "model": "local/test",
            },
        ),
        patch.object(bench.llm, "_is_local_api_base", return_value=True),
    ):
        record = await bench.replay_batch([signal], "iter", db, transport=transport)
    assert len(calls) == 1
    assert len(record["responses"]) == 1
    assert record["responses"][0]["score"]["parse_success"] is (not malformed)
    if malformed:
        assert record["outputs"] == []
        assert "error" in record["responses"][0]
    else:
        assert record["outputs"][0]["id"] == signal["id"]
        assert record["outputs"][0]["unit"] == "K"
    assert signal["description"] == ""
    assert db.execute("SELECT count(*) FROM batches").fetchone()[0] == 1
    assert "Answer description" not in json.dumps(record["messages"])
    db.close()

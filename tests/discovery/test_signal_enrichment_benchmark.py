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


@pytest.mark.parametrize("cost", [0.0, float("nan"), -0.1, 1.01])
def test_paid_projection_refuses_missing_or_excessive_cost(cost):
    with pytest.raises(ValueError):
        bench.budget_guard(cost)


def test_paid_budget_reserves_the_next_call():
    assert bench.budget_guard(0.1) == 2.0
    with pytest.raises(ValueError, match="ceiling"):
        bench.budget_guard(0.1, actual=19.9)


@pytest.mark.asyncio
async def test_normal_context_is_captured_without_enrichment_calls(tmp_path):
    signal = {
        "id": "jt-60sa:synthetic",
        "facility_id": "jt-60sa",
        "name": "COOLING:TEMP",
        "accessor": "COOLING:TEMP",
        "source_description": "Temperature",
        "unit": "K",
        "discovery_source": "synthetic",
    }
    db = bench.open_store(tmp_path / "capture.sqlite")
    tree = {
        signal["id"]: {"node_description": "Authoritative sensor calibration context"}
    }
    with (
        patch.object(
            bench.parallel, "fetch_tree_context", return_value=tree
        ) as tree_read,
        patch.object(bench.parallel, "fetch_epoch_context", return_value={}),
        patch.object(bench.parallel, "fetch_signal_code_refs", return_value={}),
        patch.object(bench.parallel, "_fetch_code_chunks", return_value=[]),
        patch(
            "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
            return_value=[],
        ),
    ):
        record = await bench.replay_batch(
            [signal], "jt-60sa", db, capture_context=True, arm="context"
        )
    tree_read.assert_called_once_with([signal["id"]])
    assert (
        "Authoritative sensor calibration context"
        in record["messages"][0][1]["content"]
    )
    assert record["responses"] == []
    assert record["outputs"] == []
    db.close()


@pytest.mark.asyncio
async def test_fenced_response_stays_a_strict_failure_when_pipeline_recovers(tmp_path):
    content = "```json\n" + json.dumps(answer()) + "\n```"

    async def transport(kwargs):
        return SimpleNamespace(
            model="local/test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
        )

    messages = [
        {"role": "system", "content": "Fixed production schema"},
        {"role": "user", "content": "Fixed retrieved context"},
    ]
    signal = {
        "id": "jt-60sa:synthetic",
        "name": "COOLING:TEMP",
        "accessor": "COOLING:TEMP",
        "unit": "K",
    }
    db = bench.open_store(tmp_path / "recovery.sqlite")
    with (
        patch.object(bench.llm, "get_api_key", return_value="test"),
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
        record = await bench.replay_batch(
            [signal], "jt-60sa", db, transport=transport, messages=messages
        )
    assert record["responses"][0]["score"]["parse_success"] is False
    assert len(record["outputs"]) == 1
    assert record["messages"] == [messages]
    assert record["prompt_digest"] == bench.digest([messages])
    db.close()


def test_iter_reference_is_refused_before_any_graph_read(tmp_path):
    with patch.object(bench, "GraphClient") as graph:
        with pytest.raises(ValueError, match="non-ITER"):
            bench.prepare(tmp_path, "iter")
    graph.assert_not_called()


def test_judge_targets_one_signal_and_keeps_its_retrieved_evidence():
    text = """Facility conventions: currents are measured in amperes.
## EDAS Category: coil
Wiki: feedback control point.
Code: read_current(channel)
## Source snippet heading
Preserve source snippet text.
### Signal 1
accessor: COIL:CURRENT
source_description: Coil current
### Signal 2
accessor: COIL:VOLTAGE
source_description: Distracting voltage
## EDAS Category: cooling
Wiki: cooling evidence; target coil current uses sign convention X.
### Signal 1
accessor: WATER:TEMP
source_description: Distracting temperature
"""
    row = {
        "id": "facility:coil",
        "first_arm": "local",
        "signal": {"accessor": "COIL:CURRENT", "unit": "A"},
        "context": [{"role": "user", "content": text}],
        "first": "Current description",
        "second": "",
    }
    state = bench.blind_state(row, False)
    assert "conventions" in state["context"]
    assert "read_current" in state["context"]
    assert "Preserve source snippet" in state["context"]
    assert "Coil current" in state["context"]
    assert "Distracting" not in state["context"]
    assert "target coil current uses sign convention X" in state["context"]
    assert "first_arm" not in state
    assert "local" not in json.dumps(state)
    swapped = bench.blind_state(row, True)
    assert swapped["context"] == state["context"]
    assert swapped["signal"] == state["signal"]
    assert (swapped["first"], swapped["second"]) == ("", "Current description")
    with pytest.raises(ValueError, match="accessor absent"):
        bench.target_context(row["context"], "ABSENT")

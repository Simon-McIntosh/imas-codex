"""Retired spellings resolve one direct holder or explain the next retry."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.graph_store import (
    graph_foundation as foundation,
    graph_source_bindings as bindings,
)
from tests.standard_names.graph_transaction_doubles import TransactionDouble

PATH = "summary/global_quantities/tau_energy/value"
SOURCE = f"dd:{PATH}"
RETIRED = "energy_confinement_time"
SUCCESSOR = "global_energy_confinement_time"


def _successor(name=SUCCESSOR, *, stage="accepted", unit="s"):
    return {"id": name, "name_stage": stage, "unit": unit}


def _candidate():
    return {
        "sns_id": SOURCE,
        "source_id": PATH,
        "source_type": "dd",
        "sn_id": RETIRED,
        "unit": "s",
        "claim_token": "source-owner",
        "claim_seq": 6,
    }


class BindingGraph(TransactionDouble):
    """Return stored graph facts; the production code chooses the successor."""

    def __init__(self, successors, *, stage="superseded"):
        super().__init__()
        self.successors = successors
        self.edges = {RETIRED: successors}
        self.names = {RETIRED: {"id": RETIRED, "name_stage": stage, "unit": "s"}}
        self.names.update({target["id"]: dict(target) for target in successors})
        self.claim = "source-owner"
        self.attempts = 5
        self.error = None
        self.bound = []
        self.query.side_effect = self.answer

    def answer(self, cypher, **params):
        if "AS outcome" in cypher:
            assert "[:HAS_SUCCESSOR]" in cypher
            rows = []
            for item in params["batch"]:
                if item["claim_token"] != self.claim:
                    continue
                name = self.names[item["sn_id"]]
                stable = name["name_stage"] in params["stable_stages"]
                rows.append(
                    {
                        "id": SOURCE,
                        "outcome": "winner" if stable else "lifecycle_collision",
                        "binding_kind": "stable_reuse" if stable else "collision",
                        "candidate_id": item["sn_id"],
                        "target_stage": name["name_stage"],
                        "attempt_count": self.attempts,
                        "successors": self.edges.get(item["sn_id"], []),
                    }
                )
            return rows
        if "SET sns.claimed_at = null," in cypher:
            item = params["batch"][0]
            if self.claim == item["claim_token"]:
                self.claim = None
                self.error = (
                    f'standard-name binding collision: candidate "{item["sn_id"]}" '
                    f'has non-bindable name_stage "{self.names[item["sn_id"]]["name_stage"]}" '
                    f"at composition attempt {self.attempts}"
                )
        if "MERGE (sns)-[produced:PRODUCED_NAME]->(sn)" in cypher:
            rows = []
            for item in params["batch"]:
                assert self.claim == item["claim_token"]
                assert (
                    self.names[item["sn_id"]]["name_stage"] in params["stable_stages"]
                )
                self.bound.append(item["sn_id"])
                self.claim = None
                rows.append({"id": SOURCE})
            return rows
        return []


def _services(graph):
    services = SimpleNamespace(
        NameStage=foundation.NameStage,
        _guard_existing_target_pairings=MagicMock(return_value={}),
        GraphClient=lambda: graph,
        _TransactionQuery=foundation._TransactionQuery,
        normalize_name_id=lambda name: name,
        write_standard_names=MagicMock(),
        bump_sn_run_counter=MagicMock(),
    )
    services._complete_source_fences = lambda items: bindings._complete_source_fences(
        items, services=services
    )
    services._lock_claimed_name_bindings = lambda *args, **kwargs: (
        bindings._lock_claimed_name_bindings(*args, services=services, **kwargs)
    )
    return services


def _lock(graph, services=None):
    return bindings._lock_claimed_name_bindings(
        graph,
        [_candidate()],
        services=services or _services(graph),
        allow_missing=True,
        allow_own_pending_reservation=True,
    )


@pytest.mark.parametrize("stage", ["superseded", "exhausted"])
def test_one_live_successor_is_followed(stage):
    graph = BindingGraph([_successor()], stage=stage)
    services = _services(graph)
    rows = _lock(graph, services)
    assert rows[0]["outcome"] == "winner"
    assert rows[0]["resolved_id"] == SUCCESSOR
    assert graph.names[RETIRED]["name_stage"] == stage
    assert graph.attempts == 5
    assert graph.claim == "source-owner"
    guarded_names = [
        item["sn_id"]
        for call in services._guard_existing_target_pairings.call_args_list
        for item in call.args[1]
    ]
    assert SUCCESSOR in guarded_names


@pytest.mark.parametrize(
    "successors",
    [
        [],
        [_successor(), _successor("local_energy_confinement_time")],
        [_successor(), _successor("local_energy_confinement_time", unit="m")],
        [_successor(stage="superseded")],
        [_successor(stage="exhausted")],
        [_successor(stage=None)],
        [_successor(unit="m")],
        [_successor(unit=None)],
    ],
    ids=[
        "absent",
        "two-live",
        "two-one-unit-match",
        "superseded",
        "exhausted",
        "unset-stage",
        "unit-mismatch",
        "unset-unit",
    ],
)
def test_ambiguous_retired_or_incompatible_successor_is_not_followed(successors):
    graph = BindingGraph(successors)
    rows = _lock(graph)
    assert rows[0]["outcome"] == "lifecycle_collision"
    assert "resolved_id" not in rows[0]
    assert graph.claim is None
    assert RETIRED in graph.error
    assert graph.attempts == 5
    assert graph.bound == []


def test_successor_attachment_refusal_preserves_retired_retry_reason():
    graph = BindingGraph([_successor()])
    original = graph.answer

    def reject(cypher, **params):
        rows = original(cypher, **params)
        if "AS outcome" in cypher and params["batch"][0]["sn_id"] == SUCCESSOR:
            rows[0]["outcome"] = "attachment_collision"
        return rows

    graph.query.side_effect = reject
    rows = _lock(graph)
    assert rows[0]["outcome"] == "lifecycle_collision"
    assert RETIRED in graph.error
    assert graph.bound == []


def test_stale_claim_cannot_follow_successor():
    graph = BindingGraph([_successor()])
    graph.claim = "different-owner"
    assert _lock(graph) == []
    assert graph.claim == "different-owner"


def _persist(graph):
    from imas_codex.standard_names.graph_store.graph_catalog import (
        persist_generated_name_winners,
    )

    services = _services(graph)
    candidate = {
        "id": RETIRED,
        "source_id": PATH,
        "source_types": ["dd"],
        "description": "Energy confinement time.",
        "unit": "s",
        "source_claim_token": "source-owner",
        "source_claim_seq": 6,
    }
    result = persist_generated_name_winners(
        [candidate], compose_model="test", services=services
    )
    return result, services


def test_finalizer_binds_successor_without_rewriting_its_metadata():
    graph = BindingGraph([_successor()])
    result, services = _persist(graph)
    assert result == [SOURCE]
    assert graph.bound == [SUCCESSOR]
    assert graph.names[RETIRED]["name_stage"] == "superseded"
    services.write_standard_names.assert_not_called()
    graph.transaction.commit.assert_called_once()


def test_two_live_successors_never_bind():
    graph = BindingGraph([_successor(), _successor("local_energy_confinement_time")])
    result, services = _persist(graph)
    assert result == []
    assert graph.bound == []
    assert RETIRED in graph.error
    assert graph.attempts == 5
    services.write_standard_names.assert_not_called()


def test_retired_successor_is_not_traversed_to_a_live_grandchild():
    graph = BindingGraph([_successor(stage="superseded")])
    graph.names["local_energy_confinement_time"] = _successor(
        "local_energy_confinement_time"
    )
    graph.edges[SUCCESSOR] = [graph.names["local_energy_confinement_time"]]
    result, services = _persist(graph)
    assert result == []
    assert graph.bound == []
    assert all(
        item["sn_id"] == RETIRED
        for call in services._guard_existing_target_pairings.call_args_list
        for item in call.args[1]
    )


def test_retry_helper_renders_exact_source_spelling_and_refusal():
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_retired_spelling_retry_reason

    graph = BindingGraph([])
    _lock(graph)
    reader = MagicMock()
    reader.query.return_value = [{"id": SOURCE, "last_error": graph.error}]
    items = [{"id": SOURCE, "claim_token": "next-owner", "claim_seq": 7}]
    reason = build_retired_spelling_retry_reason(items, gc=reader)
    rendered = render_prompt(
        "sn/generate_name_dd",
        {
            "items": [{"path": PATH, "description": "Energy confinement time."}],
            "ids_name": "summary",
            "ids_contexts": {},
            "existing_names": [],
            "retry_reason": reason,
        },
    )
    assert SOURCE in rendered
    assert RETIRED in rendered
    assert graph.error in rendered
    assert "Do not propose" in rendered
    call = reader.query.call_args
    assert call.kwargs["claims"] == items
    assert "sns.claim_token = claim.claim_token" in call.args[0]
    assert "sns.claim_seq = claim.claim_seq" in call.args[0]
    assert "sns.status = 'extracted'" in call.args[0]


@pytest.mark.parametrize(
    "error",
    [
        None,
        "transport failed",
        'candidate "name" has non-bindable name_stage "pending"',
    ],
)
def test_retry_helper_ignores_unrelated_errors(error):
    from imas_codex.standard_names.context import build_retired_spelling_retry_reason

    graph = MagicMock()
    graph.query.return_value = [{"id": SOURCE, "last_error": error}]
    assert (
        build_retired_spelling_retry_reason(
            [{"id": SOURCE, "claim_token": "owner", "claim_seq": 1}], gc=graph
        )
        == ""
    )


def test_retry_helper_skips_items_without_a_complete_claim():
    from imas_codex.standard_names.context import build_retired_spelling_retry_reason

    graph = MagicMock()
    assert build_retired_spelling_retry_reason([{"id": SOURCE}], gc=graph) == ""
    graph.query.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_compose_prompts_keep_retired_refusal_through_internal_retry(monkeypatch):
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
        compose_batch,
    )

    refused = BindingGraph([])
    _lock(refused)
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.query.side_effect = lambda cypher, **kwargs: (
        [{"id": SOURCE, "last_error": refused.error}]
        if "sns.last_error AS last_error" in cypher
        else []
    )
    batch = [
        {
            "id": SOURCE,
            "path": PATH,
            "claim_token": "next-owner",
            "claim_seq": 7,
            "description": "Energy confinement time.",
            "unit": "s",
        }
    ]
    prompts = []
    editorial_reason = "Use the reviewed quantity token."

    async def compose(**kwargs):
        prompts.append(kwargs["messages"][-1]["content"])
        if len(prompts) == 2:
            raise RuntimeError("captured both compose prompts")
        return (
            SimpleNamespace(
                candidates=[], vocab_gaps=[], attachments=[], skipped=[], dd_gaps=[]
            ),
            0.0,
            0,
        )

    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_claim_release._verify_source_claim_winners",
        lambda items, **kwargs: items,
    )
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", lambda: graph)
    monkeypatch.setattr("imas_codex.discovery.base.llm.acall_llm_structured", compose)
    monkeypatch.setattr(
        "imas_codex.llm.prompt_loader.render_prompt",
        lambda name, context: (
            render_prompt(name, context) if name == "sn/generate_name_dd" else "system"
        ),
    )
    for target, value in {
        "imas_codex.settings.get_model": "test-model",
        "imas_codex.standard_names.context.build_compose_context": {},
        "imas_codex.standard_names.example_loader.load_compose_examples": [],
        "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._enrich_batch_items": None,
        "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._search_nearby_names": [],
        "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._enrich_ids_context": None,
        "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._retry_attempts": 1,
        "imas_codex.standard_names.worker_tasks.worker_common._active_editorial_gap_guidance": (
            editorial_reason,
            True,
        ),
    }.items():
        monkeypatch.setattr(target, MagicMock(return_value=value))

    with pytest.raises(RuntimeError, match="captured both compose prompts"):
        await compose_batch(batch, MagicMock(run_id="run"), asyncio.Event())

    assert len(prompts) == 2
    for prompt in prompts:
        assert PATH in prompt
        assert "Energy confinement time." in prompt
        assert RETIRED in prompt
        assert refused.error in prompt
        assert "Do not propose" in prompt
    assert editorial_reason in prompts[1]
    refusal_reads = [
        call
        for call in graph.query.call_args_list
        if "sns.last_error AS last_error" in call.args[0]
    ]
    assert len(refusal_reads) == 1
    assert refusal_reads[0].kwargs["claims"] == [
        {"id": SOURCE, "claim_token": "next-owner", "claim_seq": 7}
    ]

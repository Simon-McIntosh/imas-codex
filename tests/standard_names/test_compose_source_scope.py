"""Claimed source batches bound every compose-result graph mutation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from imas_codex.standard_names.models import StandardNameComposeBatch
from imas_codex.standard_names.workers import (
    _filter_persist_candidates_to_claimed_sources,
    _mark_source_prevalidation_failed,
    _sanitize_compose_result_sources,
)


def _candidate(source_id: str, dd_paths: list[str]) -> dict:
    return {
        "source_id": source_id,
        "segments": {
            "base_token": "temperature",
            "base_kind": "quantity",
        },
        "description": "Temperature.",
        "kind": "scalar",
        "dd_paths": dd_paths,
        "reason": "Direct temperature quantity.",
    }


def test_unclaimed_sibling_is_removed_from_every_result_projection() -> None:
    claimed = "equilibrium/time_slice/profiles_1d/temperature"
    sibling = "core_profiles/profiles_1d/electrons/temperature"
    response_sibling = f" {sibling}\t"
    result = StandardNameComposeBatch.model_validate(
        {
            "candidates": [_candidate(claimed, [claimed, response_sibling])],
            "attachments": [
                {
                    "source_id": response_sibling,
                    "standard_name": "electron_temperature",
                    "reason": "Nearby compatible quantity.",
                }
            ],
            "skipped": [response_sibling],
            "vocab_gaps": [
                {
                    "source_id": response_sibling,
                    "segment": "physical_base",
                    "token": "temperature",
                    "reason": "Related path.",
                }
            ],
        }
    )

    _sanitize_compose_result_sources(result, {claimed}, phase="generate_name")

    assert [candidate.source_id for candidate in result.candidates] == [claimed]
    assert result.candidates[0].dd_paths == [claimed]
    assert result.attachments == []
    assert result.skipped == []
    assert result.vocab_gaps == []


def test_response_source_whitespace_resolves_to_exact_claims() -> None:
    candidate_id = "equilibrium/time_slice/profiles_1d/temperature"
    candidate_path = "equilibrium/time_slice/profiles_1d/temperature_fit"
    attachment_id = "core_profiles/profiles_1d/electrons/temperature"
    skipped_id = "core_profiles/profiles_1d/electrons/temperature_fit"
    gap_id = "core_profiles/profiles_1d/ions/temperature"
    evidence_id = "equilibrium/time_slice/profiles_1d/pressure"
    reference_id = "equilibrium/time_slice/profiles_2d/pressure"
    allowed = {
        candidate_id,
        candidate_path,
        attachment_id,
        skipped_id,
        gap_id,
        evidence_id,
        reference_id,
    }
    candidate = SimpleNamespace(
        source_id=f" {candidate_id} ",
        dd_paths=[f"\t{candidate_id}", f"{candidate_path}\n"],
    )
    attachment = SimpleNamespace(source_id=f"{attachment_id} ")
    gap = SimpleNamespace(source_id=f" {gap_id}")
    evidence = {
        "path": f" {evidence_id}\t",
        "reference_path": f"\n{reference_id} ",
    }
    result = SimpleNamespace(
        candidates=[candidate],
        attachments=[attachment],
        skipped=[f" {skipped_id} "],
        vocab_gaps=[gap],
        dd_gaps=[evidence],
    )

    _sanitize_compose_result_sources(result, allowed, phase="generate_name")

    assert candidate.source_id == candidate_id
    assert candidate.dd_paths == [candidate_id, candidate_path]
    assert attachment.source_id == attachment_id
    assert result.skipped == [skipped_id]
    assert gap.source_id == gap_id
    assert evidence == {"path": evidence_id, "reference_path": reference_id}


def test_deterministic_sibling_without_own_claim_is_deferred() -> None:
    claimed = "equilibrium/time_slice/profiles_1d/temperature"
    candidates = [
        {"id": "temperature", "source_id": claimed},
        {
            "id": "temperature_error_upper",
            "source_id": f"{claimed}_error_upper",
            "_from_error_sibling": True,
        },
    ]

    kept = _filter_persist_candidates_to_claimed_sources(
        candidates,
        {claimed},
        phase="generate_name",
    )

    assert kept == [candidates[0]]


def test_prevalidation_failure_records_durable_diagnostics() -> None:
    with patch(
        "imas_codex.standard_names.graph_ops.persist_claimed_source_outcomes",
        return_value=["dd:equilibrium/time_slice/profiles_1d/gm6"],
    ) as persist:
        result = _mark_source_prevalidation_failed(
            "equilibrium/time_slice/profiles_1d/gm6",
            source_type="dd",
            candidate_name="invalid name",
            reason="not_snake_case",
            claim_token="winner",
            claim_seq=6,
        )

    assert result is True
    outcome = persist.call_args.args[0][0]
    assert outcome["sns_id"] == "dd:equilibrium/time_slice/profiles_1d/gm6"
    assert outcome["claim_token"] == "winner"
    assert outcome["claim_seq"] == 6
    assert outcome["status"] == "failed"
    assert "not_snake_case" in outcome["last_error"]


def test_prevalidation_failure_without_complete_fence_is_noop() -> None:
    with patch(
        "imas_codex.standard_names.graph_ops.persist_claimed_source_outcomes"
    ) as persist:
        result = _mark_source_prevalidation_failed(
            "equilibrium/time_slice/profiles_1d/gm6",
            source_type="dd",
            candidate_name="invalid name",
            reason="not_snake_case",
            claim_token=None,
            claim_seq=6,
        )

    assert result is False
    persist.assert_not_called()


def test_source_outcome_without_complete_fence_does_not_open_graph() -> None:
    from imas_codex.standard_names.graph_ops import persist_claimed_source_outcomes

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as graph_client:
        winners = persist_claimed_source_outcomes(
            [
                {
                    "sns_id": "dd:equilibrium/time_slice/q",
                    "claim_token": None,
                    "claim_seq": 5,
                    "status": "skipped",
                }
            ]
        )

    assert winners == []
    graph_client.assert_not_called()

"""Persistence boundary for reviewer name suggestions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_store import (
    graph_catalog as _owning_graph_catalog,
    graph_foundation as _owning_graph_foundation,
)


def _review_record(
    reviewed_identity: str,
    proposed_spelling: str,
    *,
    suffix: str,
) -> dict[str, object]:
    scores = {
        "grammar": 17,
        "semantic": 13,
        "convention": 19,
        "completeness": 11,
    }
    comments_per_dimension = {
        "grammar": "The grammar is sound.",
        "semantic": "The meaning needs attention.",
        "convention": "The convention is consistent.",
        "completeness": "The identity is complete.",
    }
    return {
        "id": f"{reviewed_identity}:name:suggestion-persist:{suffix}",
        "standard_name_id": reviewed_identity,
        "model": "reviewer/example",
        "reviewer_model": "reviewer/example",
        "model_family": "other",
        "is_canonical": False,
        "score": 0.75,
        "scores_json": json.dumps(scores, sort_keys=True),
        "tier": "good",
        "comments": "Keep this assessment unchanged.",
        "comments_per_dim_json": json.dumps(comments_per_dimension, sort_keys=True),
        "suggested_name": proposed_spelling,
        "suggestion_justification": f"Justification for {proposed_spelling}.",
        "reviewed_at": "2026-09-06T08:00:00+00:00",
        "review_axis": "name",
        "cycle_index": 0,
        "review_group_id": "suggestion-persist",
        "resolution_role": "primary",
        "resolution_method": "single_review",
        "llm_model": "reviewer/example",
        "llm_cost": 0.125,
        "llm_tokens_in": 101,
        "llm_tokens_out": 53,
        "llm_tokens_cached_read": 7,
        "llm_tokens_cached_write": 3,
        "llm_at": "2026-09-06T08:00:00+00:00",
        "llm_service": "standard-names",
    }


def test_write_reviews_clears_only_non_distinct_or_unparseable_suggestions() -> None:
    reviewed_identity = (
        "energy_flux_normalized_due_to_perturbed_parallel_vector_potential"
    )
    equivalent_spelling = (
        "normalized_energy_flux_due_to_perturbed_parallel_vector_potential"
    )
    distinct_spelling = (
        "particle_flux_normalized_due_to_perturbed_parallel_vector_potential"
    )
    records = [
        _review_record(reviewed_identity, equivalent_spelling, suffix="equivalent"),
        _review_record(reviewed_identity, distinct_spelling, suffix="distinct"),
        _review_record(reviewed_identity, "not_a_standard_name", suffix="unparseable"),
    ]
    assessment_fields = (
        "score",
        "scores_json",
        "tier",
        "comments",
        "comments_per_dim_json",
    )
    graph = MagicMock()
    graph.query.return_value = []

    with patch.object(_owning_graph_foundation, "GraphClient") as graph_client:
        graph_client.return_value.__enter__.return_value = graph
        graph_client.return_value.__exit__.return_value = False
        result = _owning_graph_catalog.write_reviews(records, skip_cost=True)

    write_call = next(
        call
        for call in graph.query.call_args_list
        if call.args and "MERGE (r:StandardNameReview" in call.args[0]
    )
    persisted = write_call.kwargs["batch"]

    assert result == 3
    assert result.suggestions_cleared_non_distinct == 1
    assert result.suggestions_cleared_unparseable == 1

    assert persisted[0]["suggested_name"] == ""
    assert persisted[0]["suggestion_justification"] == ""
    assert persisted[1]["suggested_name"] == distinct_spelling
    assert (
        persisted[1]["suggestion_justification"]
        == records[1]["suggestion_justification"]
    )
    assert persisted[2]["suggested_name"] == ""
    assert persisted[2]["suggestion_justification"] == ""

    for original, written in zip(records, persisted, strict=True):
        for field in assessment_fields:
            assert written[field] == original[field]

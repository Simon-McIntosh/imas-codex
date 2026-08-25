"""Shared graph traversal for docs-review eligibility."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imas_codex.cli.sn import _compute_pool_progress
from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.export import (
    _classify_export_population,
    _fetch_candidates,
    _fetch_export_population,
)


class _Graph:
    def __init__(self, responses: list[list[dict]] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.responses = list(responses or [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        self.calls.append((cypher, params))
        return self.responses.pop(0) if self.responses else []


def _assert_shared_eligibility(cypher: str, params: dict) -> None:
    assert graph_ops.docs_review_eligibility_where() in cypher
    assert "docs_review_resolution_method IS NOT NULL" not in cypher
    assert params["docs_review_winning_methods"] == [
        "authoritative_escalation",
        "quorum_consensus",
        "single_review",
    ]
    assert params["docs_review_non_winning_methods"] == [
        "max_cycles_reached",
        "retry_item",
    ]


def test_winning_methods_are_derived_from_schema() -> None:
    params = graph_ops.docs_review_eligibility_params()

    assert set(params["docs_review_winning_methods"]) | set(
        params["docs_review_non_winning_methods"]
    ) == {
        "authoritative_escalation",
        "max_cycles_reached",
        "quorum_consensus",
        "retry_item",
        "single_review",
    }
    assert set(params["docs_review_winning_methods"]).isdisjoint(
        params["docs_review_non_winning_methods"]
    )


def test_export_gate_and_population_use_shared_traversal() -> None:
    candidate_graph = _Graph()
    with patch("imas_codex.graph.client.GraphClient", return_value=candidate_graph):
        assert _fetch_candidates() == []
    _assert_shared_eligibility(*candidate_graph.calls[0])

    population_graph = _Graph()
    with (
        patch("imas_codex.graph.client.GraphClient", return_value=population_graph),
        patch.object(graph_ops, "docs_review_property_coverage", return_value={}),
    ):
        assert _fetch_export_population() == []
    query, params = population_graph.calls[0]
    _assert_shared_eligibility(query, params)
    assert "review.review_axis = 'docs'" in query
    assert "_has_docs_review: has_docs_review" in query


def test_export_reason_distinguishes_absence_from_unrecorded_resolution() -> None:
    common = {
        "name_stage": "accepted",
        "validation_status": "valid",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
    }
    population = [
        {
            **common,
            "id": "winning_review",
            "_has_docs_review": True,
            "_has_winning_docs_review": True,
        },
        {
            **common,
            "id": "non_winning_review",
            "_has_docs_review": True,
            "_has_winning_docs_review": False,
        },
        {
            **common,
            "id": "no_docs_review",
            "_has_docs_review": False,
            "_has_winning_docs_review": False,
        },
    ]

    eligible, exclusions = _classify_export_population(
        population, domain=None, names_only=False
    )

    assert [row["id"] for row in eligible] == ["winning_review"]
    assert {row.standard_name_id: row.reason for row in exclusions} == {
        "no_docs_review": "never_reviewed",
        "non_winning_review": "resolution_unrecorded",
    }
    assert len(eligible) + len(exclusions) == len(population)


def test_pending_count_and_claim_use_the_same_atomic_predicate() -> None:
    pools = (
        "generate_name",
        "review_name",
        "refine_name",
        "generate_docs",
        "review_docs",
        "refine_docs",
        "enrich_parents",
    )
    row = {key: 0 for pool in pools for key in (pool, f"{pool}_done")}
    progress_graph = _Graph([[row]])
    _compute_pool_progress(progress_graph, None, 3, 0.85)
    _assert_shared_eligibility(*progress_graph.calls[0])

    captured: dict = {}

    def fake_claim(**kwargs):
        captured.update(kwargs)
        return []

    with patch.object(graph_ops, "_claim_sn_atomic", side_effect=fake_claim):
        assert graph_ops.claim_refine_docs_batch(min_score=0.85) == []

    _assert_shared_eligibility(captured["eligibility_where"], captured["query_params"])
    assert captured["stage_field"] == "docs_stage"
    assert captured["to_stage"] == "refining"


def test_stranded_promotion_uses_shared_traversal() -> None:
    graph = _Graph([[{"n": 0}], [{"n": 0}]])
    with patch.object(graph_ops, "GraphClient", return_value=graph):
        assert graph_ops.promote_stranded_reviewed(dry_run=True) == {
            "name": 0,
            "docs": 0,
        }

    docs_query, params = graph.calls[1]
    _assert_shared_eligibility(docs_query, params)


def test_property_coverage_requires_plural_axis_and_every_filtered_property() -> None:
    complete = {
        "reviews": 5,
        "axis_covered": 5,
        "group_covered": 5,
        "method_covered": 3,
        "known_axis": 5,
        "docs_axis": 2,
    }
    graph = _Graph([[complete]])
    with patch.object(graph_ops, "GraphClient", return_value=graph):
        assert graph_ops.docs_review_property_coverage() == complete
    query = graph.calls[0][0]
    assert "['name', 'docs']" in query
    assert "review.review_axis = 'docs'" in query

    missing_method = {**complete, "method_covered": 0}
    with (
        patch.object(graph_ops, "GraphClient", return_value=_Graph([[missing_method]])),
        pytest.raises(RuntimeError, match="method_covered"),
    ):
        graph_ops.docs_review_property_coverage()

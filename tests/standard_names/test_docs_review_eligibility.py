"""Shared graph traversal for docs-review eligibility."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from imas_codex.cli.sn import _compute_pool_progress
from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.export import (
    _classify_export_population,
    _fetch_candidates,
    _fetch_export_population,
)
from imas_codex.standard_names.graph_store import (
    graph_audits as _owning_graph_audits,
    graph_catalog as _owning_graph_catalog,
    graph_cost_and_claims as _owning_graph_cost_and_claims,
    graph_docs_persistence as _owning_graph_docs_persistence,
    graph_foundation as _owning_graph_foundation,
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
    assert _owning_graph_catalog.docs_review_eligibility_where() in cypher
    assert "docs_review_resolution_method IS NOT NULL" not in cypher
    expected = _owning_graph_catalog.docs_review_eligibility_params()
    assert {key: params[key] for key in expected} == expected
    assert "semantic_similarity_gate" in params["docs_review_non_winning_methods"]


def _schema_resolution_methods() -> set[str]:
    schema_path = Path(graph_ops.__file__).parents[1] / "schemas" / "standard_name.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    return set(schema["enums"]["ReviewResolutionMethod"]["permissible_values"])


def test_winning_methods_are_derived_from_schema() -> None:
    params = _owning_graph_catalog.docs_review_eligibility_params()

    assert (
        set(params["docs_review_winning_methods"])
        | set(params["docs_review_non_winning_methods"])
        == _schema_resolution_methods()
    )
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
        patch.object(
            _owning_graph_catalog, "docs_review_property_coverage", return_value={}
        ),
    ):
        assert _fetch_export_population() == []
    query, params = population_graph.calls[0]
    _assert_shared_eligibility(query, params)
    assert "review.review_axis = 'docs'" in query
    assert "_has_docs_review: has_docs_review" in query


def test_export_reason_distinguishes_absence_from_unrecorded_resolution() -> None:
    common = {
        "name_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "_validation_observed_at": "2026-09-09T00:00:00Z",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
        # The topology the population query projects onto every candidate: a
        # Data Dictionary extraction reached this name directly. Without
        # producer evidence the eligibility classifier withholds the candidate
        # before it reaches the docs-review branches under test.
        "_has_dd_source_binding": True,
        "_has_derived_producer": False,
        "_has_non_derived_producer": True,
        "_has_live_child": False,
        "_is_parent": False,
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

    with patch.object(
        _owning_graph_cost_and_claims, "_claim_sn_atomic", side_effect=fake_claim
    ):
        assert (
            _owning_graph_docs_persistence.claim_refine_docs_batch(min_score=0.85) == []
        )

    _assert_shared_eligibility(captured["eligibility_where"], captured["query_params"])
    assert captured["stage_field"] == "docs_stage"
    assert captured["to_stage"] == "refining"


def test_stranded_promotion_uses_shared_traversal() -> None:
    graph = _Graph([[{"n": 0}], [{"n": 0}]])
    with patch.object(_owning_graph_foundation, "GraphClient", return_value=graph):
        assert _owning_graph_audits.promote_stranded_reviewed(dry_run=True) == {
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
    with patch.object(_owning_graph_foundation, "GraphClient", return_value=graph):
        assert _owning_graph_catalog.docs_review_property_coverage() == complete
    query = graph.calls[0][0]
    assert "['name', 'docs']" in query
    assert "review.review_axis = 'docs'" in query

    missing_method = {**complete, "method_covered": 0}
    with (
        patch.object(
            _owning_graph_foundation,
            "GraphClient",
            return_value=_Graph([[missing_method]]),
        ),
        pytest.raises(RuntimeError, match="method_covered"),
    ):
        _owning_graph_catalog.docs_review_property_coverage()

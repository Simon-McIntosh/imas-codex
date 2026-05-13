"""Tests for the scored-example loader (K3).

Validates that ``load_compose_examples`` and ``load_review_examples``
correctly query reviewed StandardName nodes, handle domain fallback,
parse JSON fields, and produce deterministic ordering.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.example_loader import (
    load_compose_examples,
    load_review_examples,
)


def _make_gc(query_responses: list[list[dict]] | None = None) -> MagicMock:
    """Build a mock GraphClient with canned query responses.

    *query_responses* is a list of result-lists; each call to ``gc.query()``
    pops the next one.  When exhausted, returns ``[]``.
    """
    gc = MagicMock()
    remaining = list(query_responses or [])

    def _query_side_effect(*_args, **_kwargs):
        if remaining:
            return remaining.pop(0)
        return []

    gc.query = MagicMock(side_effect=_query_side_effect)
    return gc


def _sn_row(
    *,
    name_id: str = "electron_temperature",
    score: float = 0.82,
    domain: str = "transport",
    scores: dict | None = None,
    comments_per_dim: dict | None = None,
    physical_base: str | None = None,
    subject: str | None = None,
    component: str | None = None,
    coordinate: str | None = None,
    transformation: str | None = None,
    position: str | None = None,
    process: str | None = None,
    geometric_base: str | None = None,
    semantic_sim: float | None = None,
) -> dict:
    """Build a dict mimicking a StandardName graph row."""
    return {
        "id": name_id,
        "description": f"Description of {name_id}",
        "documentation": f"Docs for {name_id}",
        "kind": "scalar",
        "unit": "eV",
        "reviewer_score": score,
        "reviewer_scores_json": json.dumps(scores or {"dim_a": 18, "dim_b": 16}),
        "reviewer_comments_per_dim_json": json.dumps(
            comments_per_dim or {"dim_a": "Good", "dim_b": "OK"}
        ),
        "reviewer_comments": "Overall good quality",
        "physics_domain": domain,
        "physical_base": physical_base,
        "subject": subject,
        "component": component,
        "coordinate": coordinate,
        "transformation": transformation,
        "position": position,
        "process": process,
        "geometric_base": geometric_base,
        "semantic_sim": semantic_sim,
    }


class TestEmptyGraph:
    """Empty graph → returns [] for both loaders (critical: zero-opp cold start)."""

    def test_compose_empty(self) -> None:
        gc = _make_gc()
        result = load_compose_examples(gc, physics_domains=[], axis="name")
        assert result == []

    def test_review_empty(self) -> None:
        gc = _make_gc()
        result = load_review_examples(gc, physics_domains=["transport"], axis="name")
        assert result == []

    def test_compose_empty_with_domains(self) -> None:
        gc = _make_gc()
        result = load_compose_examples(
            gc, physics_domains=["magnetics", "transport"], axis="name"
        )
        assert result == []


class TestSingleReviewedName:
    """Graph with 1 reviewed name at score 0.82 → returned for 0.80 target bucket."""

    def test_returns_for_matching_bucket(self) -> None:
        row = _sn_row(score=0.82)
        # 4 targets × (domain-scoped + fallback) → need many responses.
        # Only the 0.80 target should match (|0.82 - 0.80| = 0.02 ≤ 0.05).
        # Provide it for the domain-scoped query at 0.80 target.
        # Empty for all other targets.
        # Targets sorted DESC: 1.00, 0.80, 0.65, 0.40
        responses: list[list[dict]] = [
            [],  # 1.00 domain-scoped → empty
            [],  # 1.00 fallback → empty
            [row],  # 0.80 domain-scoped → match
            # Remaining targets: empty
            [],  # 0.65 domain-scoped → empty
            [],  # 0.65 fallback → empty
            [],  # 0.40 domain-scoped → empty
            [],  # 0.40 fallback → empty
        ]
        gc = _make_gc(responses)
        result = load_compose_examples(gc, physics_domains=["transport"], axis="name")
        assert len(result) == 1
        assert result[0]["id"] == "electron_temperature"
        assert result[0]["reviewer_score"] == 0.82
        assert result[0]["target_score"] == 0.80

    def test_score_alias(self) -> None:
        """Template alias 'score' should equal 'reviewer_score'."""
        row = _sn_row(score=0.82)
        responses: list[list[dict]] = [[], [], [row]]
        gc = _make_gc(responses)
        result = load_compose_examples(gc, physics_domains=["transport"], axis="name")
        assert len(result) == 1
        assert result[0]["score"] == result[0]["reviewer_score"]


class TestDomainScoping:
    """Domain-scoped query returns only matching domain."""

    def test_domain_match(self) -> None:
        transport_row = _sn_row(domain="transport", name_id="te", score=1.0)
        # Domain-scoped for transport should return transport_row only
        responses: list[list[dict]] = [
            [transport_row],  # 1.00 domain-scoped → transport match
        ]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(1.0,),
        )
        assert len(result) == 1
        assert result[0]["id"] == "te"
        assert result[0]["domain"] == "transport"

    def test_fallback_to_all_domains(self) -> None:
        """No names in requested domain → fallback returns names from other domains."""
        other_row = _sn_row(domain="magnetics", name_id="ip", score=1.0)
        responses: list[list[dict]] = [
            [],  # 1.00 domain-scoped → empty (no transport names)
            [other_row],  # 1.00 fallback → magnetics name
        ]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(1.0,),
        )
        assert len(result) == 1
        assert result[0]["id"] == "ip"
        assert result[0]["domain"] == "magnetics"


class TestDeterministicOrdering:
    """Two calls with identical graph return identical result order."""

    def test_deterministic(self) -> None:
        row_a = _sn_row(name_id="aaa_quantity", score=1.0)
        row_b = _sn_row(name_id="bbb_quantity", score=1.0)
        # Both match the 1.0 target with per_bucket=2
        responses1: list[list[dict]] = [
            [row_a, row_b],  # 1.00 domain-scoped
        ]
        responses2: list[list[dict]] = [
            [row_a, row_b],  # 1.00 domain-scoped
        ]
        gc1 = _make_gc(responses1)
        gc2 = _make_gc(responses2)
        result1 = load_compose_examples(
            gc1,
            physics_domains=["transport"],
            axis="name",
            target_scores=(1.0,),
            per_bucket=2,
        )
        result2 = load_compose_examples(
            gc2,
            physics_domains=["transport"],
            axis="name",
            target_scores=(1.0,),
            per_bucket=2,
        )
        assert [r["id"] for r in result1] == [r["id"] for r in result2]


class TestJSONParsing:
    """scores_json and comments_json are deserialised into dicts."""

    def test_scores_parsed(self) -> None:
        scores = {"description_quality": 18, "documentation_quality": 15}
        comments = {
            "description_quality": "Clear and concise",
            "documentation_quality": "Needs more detail",
        }
        row = _sn_row(scores=scores, comments_per_dim=comments)
        responses: list[list[dict]] = [[], [], [row]]
        gc = _make_gc(responses)
        result = load_compose_examples(gc, physics_domains=["transport"], axis="name")
        assert len(result) == 1
        assert result[0]["scores"] == scores
        assert result[0]["dimension_comments"] == comments

    def test_null_json_fields(self) -> None:
        """None JSON fields → empty dicts."""
        row = _sn_row()
        row["reviewer_scores_json"] = None
        row["reviewer_comments_per_dim_json"] = None
        responses: list[list[dict]] = [[], [], [row]]
        gc = _make_gc(responses)
        result = load_compose_examples(gc, physics_domains=["transport"], axis="name")
        assert len(result) == 1
        assert result[0]["scores"] == {}
        assert result[0]["dimension_comments"] == {}

    def test_invalid_json_fields(self) -> None:
        """Malformed JSON → empty dicts (no crash)."""
        row = _sn_row()
        row["reviewer_scores_json"] = "not-valid-json"
        row["reviewer_comments_per_dim_json"] = "{broken"
        responses: list[list[dict]] = [[], [], [row]]
        gc = _make_gc(responses)
        result = load_compose_examples(gc, physics_domains=["transport"], axis="name")
        assert len(result) == 1
        assert result[0]["scores"] == {}
        assert result[0]["dimension_comments"] == {}


class TestDocsReviewerDimNames:
    """Docs-style reviewer with independent dim names passes through unmodified."""

    def test_independent_dims(self) -> None:
        scores = {
            "description_quality": 17,
            "documentation_quality": 14,
            "completeness": 19,
            "physics_accuracy": 18,
        }
        comments = {
            "description_quality": "Clear",
            "documentation_quality": "Could improve",
            "completeness": "All covered",
            "physics_accuracy": "Correct",
        }
        row = _sn_row(scores=scores, comments_per_dim=comments)
        responses: list[list[dict]] = [[row]]
        gc = _make_gc(responses)
        result = load_review_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(0.82,),
            tolerance=0.05,
        )
        assert len(result) == 1
        # Dim names passed through without modification
        assert set(result[0]["scores"].keys()) == set(scores.keys())
        assert set(result[0]["dimension_comments"].keys()) == set(comments.keys())


class TestNoDuplicates:
    """Same name appearing at multiple targets is deduplicated."""

    def test_dedup(self) -> None:
        row = _sn_row(name_id="electron_temperature", score=0.82)
        # Score 0.82 could match both 0.80 (|0.02| ≤ 0.05) and 0.85 targets
        responses: list[list[dict]] = [
            [row],  # first target
            [row],  # second target (same name)
        ]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(0.85, 0.80),
            tolerance=0.05,
        )
        # Only appears once
        assert len(result) == 1
        assert result[0]["id"] == "electron_temperature"


class TestEmptyDomainsList:
    """Empty physics_domains list skips domain-scoped query → uses fallback."""

    def test_empty_domains_uses_fallback(self) -> None:
        row = _sn_row(score=1.0)
        # With empty domains, no domain-scoped query is issued.
        # Only fallback queries.
        responses: list[list[dict]] = [
            [row],  # 1.00 fallback (no domain-scoped since domains=[])
        ]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=[],
            axis="name",
            target_scores=(1.0,),
        )
        assert len(result) == 1
        assert result[0]["id"] == "electron_temperature"


class TestReviewExamplesMatchesCompose:
    """load_review_examples uses same logic as load_compose_examples."""

    def test_review_returns_same_structure(self) -> None:
        row = _sn_row(score=1.0)
        responses: list[list[dict]] = [[row]]
        gc = _make_gc(responses)
        result = load_review_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(1.0,),
        )
        assert len(result) == 1
        # Must have all template-required keys
        required_keys = {
            "id",
            "description",
            "documentation",
            "kind",
            "unit",
            "reviewer_score",
            "scores",
            "dimension_comments",
            "physics_domain",
            "target_score",
            "score",
            "domain",
            "issues",
            "grammar_segments",
            "semantic_sim",
        }
        assert required_keys.issubset(set(result[0].keys()))


class TestGrammarFieldProjection:
    """Grammar decomposition fields are projected from graph into examples."""

    def test_grammar_segments_populated(self) -> None:
        row = _sn_row(
            name_id="electron_temperature",
            score=1.0,
            physical_base="temperature",
            subject="electron",
        )
        responses: list[list[dict]] = [[row]]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(1.0,),
        )
        assert len(result) == 1
        gf = result[0]["grammar_segments"]
        assert gf["physical_base"] == "temperature"
        assert gf["subject"] == "electron"
        # Fields not set should be absent
        assert "component" not in gf
        assert "coordinate" not in gf

    def test_all_grammar_segments(self) -> None:
        row = _sn_row(
            name_id="radial_magnetic_field",
            score=0.95,
            physical_base="magnetic_field",
            component="radial",
            position="separatrix",
        )
        responses: list[list[dict]] = [[row]]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=[],
            axis="name",
            target_scores=(0.95,),
            tolerance=0.10,
        )
        assert len(result) == 1
        gf = result[0]["grammar_segments"]
        assert gf["physical_base"] == "magnetic_field"
        assert gf["component"] == "radial"
        assert gf["position"] == "separatrix"
        assert "subject" not in gf

    def test_empty_grammar_segments_when_none(self) -> None:
        """If no grammar segments populated, grammar_segments is empty dict."""
        row = _sn_row(score=0.80)
        responses: list[list[dict]] = [[], [row]]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(0.80,),
        )
        assert len(result) == 1
        assert result[0]["grammar_segments"] == {}


class TestSemanticSimProjection:
    """Semantic similarity is projected from graph into examples."""

    def test_semantic_sim_present(self) -> None:
        row = _sn_row(score=1.0, semantic_sim=0.89)
        responses: list[list[dict]] = [[row]]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(1.0,),
        )
        assert len(result) == 1
        assert result[0]["semantic_sim"] == 0.89

    def test_semantic_sim_none_when_missing(self) -> None:
        row = _sn_row(score=0.65)
        responses: list[list[dict]] = [[], [row]]
        gc = _make_gc(responses)
        result = load_compose_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(0.65,),
        )
        assert len(result) == 1
        assert result[0]["semantic_sim"] is None


class TestNameStageFiltering:
    """Only accepted names should appear in examples (Cypher WHERE clause)."""

    def test_query_includes_name_stage_filter(self) -> None:
        """Verify the Cypher query sent to Neo4j includes name_stage filter."""
        gc = _make_gc([])
        load_compose_examples(
            gc,
            physics_domains=["transport"],
            axis="name",
            target_scores=(1.0,),
        )
        # Inspect the Cypher queries sent
        assert gc.query.call_count >= 1
        first_call_cypher = gc.query.call_args_list[0][0][0]
        assert "name_stage" in first_call_cypher
        assert "'accepted'" in first_call_cypher

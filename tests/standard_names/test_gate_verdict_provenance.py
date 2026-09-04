"""Typed provenance for deterministic embedding-similarity verdicts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from imas_codex.standard_names import graph_ops

_METHOD = "semantic_similarity_gate"
_MODEL_IDENTITY = "(semantic_similarity_gate)"
_WINNING_METHODS = frozenset(
    {"quorum_consensus", "authoritative_escalation", "single_review"}
)


class _SimilarityGraph:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def query(self, _cypher: str, **_params):
        return []


class _BackfillGraph:
    def __init__(self, eligible: int = 36) -> None:
        self.eligible = eligible
        self.calls: list[tuple[str, dict]] = []

    def query(self, cypher: str, **params):
        self.calls.append((cypher, params))
        updated = self.eligible
        self.eligible = 0
        return [{"updated": updated}]


def test_similarity_method_is_declared_without_expanding_winning_authority() -> None:
    schema_path = (
        Path(__file__).parents[2] / "imas_codex" / "schemas" / "standard_name.yaml"
    )
    graph_ops._winning_review_resolution_methods.cache_clear()

    assert graph_ops._winning_review_resolution_methods(schema_path) == _WINNING_METHODS
    assert _METHOD in graph_ops._non_winning_review_resolution_methods()


def test_similarity_method_keeps_acceptance_and_refine_claims_closed() -> None:
    shortfall = graph_ops._quorum_admits_acceptance(_METHOD, reviewer_chain_size=None)

    assert shortfall is not None
    assert _METHOD in shortfall
    assert _METHOD not in graph_ops.QUORATE_RESOLUTION_METHODS
    assert (
        "sn.review_quorum_shortfall IS NULL" in graph_ops.REFINE_NAME_ELIGIBILITY_WHERE
    )


def test_similarity_gate_persists_only_the_measured_dimension() -> None:
    from imas_codex.standard_names.workers import process_review_name_batch

    persist = MagicMock(return_value="reviewed")
    with (
        patch(
            "imas_codex.settings.get_sn_review_names_models",
            return_value=["unused-reviewer"],
        ),
        patch(
            "imas_codex.standard_names.workers._enrich_name_review_items",
        ),
        patch(
            "imas_codex.standard_names.audits.semantic_similarity_check",
            return_value=(0.20, []),
        ),
        patch(
            "imas_codex.standard_names.graph_ops.persist_reviewed_name",
            persist,
        ),
        patch(
            "imas_codex.graph.client.GraphClient",
            return_value=_SimilarityGraph(),
        ),
    ):
        processed = asyncio.run(
            process_review_name_batch(
                [
                    {
                        "id": "ambiguous_density",
                        "description": "A quantity whose carrier is not stated.",
                        "claim_token": "claim-token",
                        "validation_status": "valid",
                    }
                ],
                SimpleNamespace(run_id="run-id"),
                asyncio.Event(),
            )
        )

    assert processed == 1
    persisted = persist.call_args.kwargs
    assert persisted["resolution_method"] == _METHOD
    assert persisted["model"] == _MODEL_IDENTITY
    assert persisted["scores"] == {"semantic": 0.20}


def test_similarity_gate_backfill_uses_model_identity_and_is_idempotent() -> None:
    from imas_codex.standard_names.workers import (
        backfill_semantic_similarity_gate_resolution_method,
    )

    graph = _BackfillGraph()

    assert backfill_semantic_similarity_gate_resolution_method(graph) == 36
    assert backfill_semantic_similarity_gate_resolution_method(graph) == 0

    cypher, params = graph.calls[0]
    assert "review.model = $model_identity" in cypher
    assert "review.reviewer_model = $model_identity" in cypher
    assert "review.score" not in cypher
    assert params == {
        "model_identity": _MODEL_IDENTITY,
        "resolution_method": _METHOD,
    }

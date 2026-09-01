"""Ordinary pipeline work cannot mutate catalog-dispositioned names."""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import patch

import pytest

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.promote import (
    mark_catalog_name_approved,
    resolve_contested_override,
)


@pytest.mark.parametrize(
    "claim",
    [
        graph_ops.claim_generate_docs_batch,
        graph_ops.claim_review_docs_batch,
        graph_ops.claim_refine_docs_batch,
        graph_ops.claim_review_name_batch,
        graph_ops.claim_refine_name_batch,
        graph_ops.claim_enrich_parents_batch,
    ],
)
def test_standard_name_pool_claims_share_the_frozen_stage_guard(claim: Any) -> None:
    """Every StandardName pool reaches the one guarded atomic claim primitive."""
    seeded = {
        "approved_name": "approved",
        "contested_name": "contested",
        "accepted_name": "accepted",
    }
    assert [
        name for name, stage in seeded.items() if stage not in {"approved", "contested"}
    ] == ["accepted_name"]
    assert "_claim_sn_atomic" in inspect.getsource(claim)
    source = inspect.getsource(graph_ops._claim_sn_atomic)
    assert source.count("IN $frozen_name_stages") >= 6
    assert "_PIPELINE_FROZEN_NAME_STAGES" in source


def test_generate_name_excludes_sources_bound_to_frozen_names() -> None:
    """The source pool applies the same guard before and after its write lock."""
    source = inspect.getsource(graph_ops.claim_generate_name_batch)
    assert source.count("MATCH (sns)-[:PRODUCED_NAME]->(frozen:StandardName)") >= 4
    assert source.count("MATCH (sns2)-[:PRODUCED_NAME]->(frozen:StandardName)") >= 2
    assert source.count("frozen.name_stage IN $frozen_name_stages") >= 6


class _ReviewGraph:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.statements: list[str] = []

    def __enter__(self) -> _ReviewGraph:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def query(self, statement: str, **_parameters: Any) -> list[dict[str, Any]]:
        self.statements.append(statement)
        if "update_review_aggregates" in inspect.stack()[1].function:
            return (
                []
                if self.stage in {"approved", "contested"}
                else [{"id": "accepted_name"}]
            )
        return []


@pytest.mark.parametrize("stage", ["approved", "contested"])
def test_review_persistence_and_aggregates_skip_frozen_names(stage: str) -> None:
    graph = _ReviewGraph(stage)
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        assert (
            graph_ops.persist_reviewed_docs(
                sn_id=f"{stage}_name",
                claim_token="claim",
                score=0.9,
                model="reviewer",
            )
            == ""
        )
        assert graph_ops.update_review_aggregates([f"{stage}_name"]) == 0
    aggregate = graph.statements[-1]
    assert "NOT (coalesce(sn.name_stage, '') IN $frozen_name_stages)" in aggregate


class _PromotionGraph:
    def __init__(self, *, name_stage: str, docs_stage: str) -> None:
        self.name_stage = name_stage
        self.docs_stage = docs_stage

    def query(self, statement: str, **parameters: Any) -> list[dict[str, str]]:
        if "name_stage: 'contested'" in statement:
            if self.name_stage != "contested":
                return []
            self.name_stage = "approved"
            self.docs_stage = "accepted"
            return [{"id": parameters["name"]}]
        if (
            self.name_stage not in {"accepted", "approved"}
            or self.docs_stage != "accepted"
        ):
            return []
        self.name_stage = "approved"
        self.docs_stage = "accepted"
        return [{"id": parameters["name"]}]

    def close(self) -> None:
        return None


def test_approval_and_contested_override_settle_documentation() -> None:
    auto = _PromotionGraph(name_stage="accepted", docs_stage="accepted")
    assert mark_catalog_name_approved(
        "auto_name",
        catalog_pr_number=3,
        catalog_pr_url="https://example.invalid/pull/3",
        catalog_merge_commit_sha="abc123",
        gc=auto,
    )
    assert (auto.name_stage, auto.docs_stage) == ("approved", "accepted")

    resolved = _PromotionGraph(name_stage="contested", docs_stage="drafted")
    assert resolve_contested_override(
        "resolved_name", reason="Human disposition.", gc=resolved
    )
    assert (resolved.name_stage, resolved.docs_stage) == ("approved", "accepted")

"""Tests for the accepted-documentation resolution-method projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from imas_codex.standard_names import graph_ops


def test_schema_derivation_excludes_non_winning_terminal_states() -> None:
    methods = graph_ops._winning_review_resolution_methods()

    assert methods == {
        "quorum_consensus",
        "authoritative_escalation",
        "single_review",
    }
    assert "max_cycles_reached" not in methods
    assert "retry_item" not in methods


def test_new_enum_value_is_not_silently_admitted(tmp_path: Path) -> None:
    schema_path = tmp_path / "standard_name.yaml"
    schema_path.write_text(
        """
enums:
  ReviewResolutionMethod:
    description: >-
      Only groups with ``quorum_consensus`` are eligible as winning groups.
    permissible_values:
      quorum_consensus: {}
      future_terminal: {}
""".lstrip(),
        encoding="utf-8",
    )

    assert graph_ops._winning_review_resolution_methods(schema_path) == {
        "quorum_consensus"
    }


def test_schema_without_eligibility_declaration_fails_closed(tmp_path: Path) -> None:
    schema_path = tmp_path / "standard_name.yaml"
    schema_path.write_text(
        """
enums:
  ReviewResolutionMethod:
    description: Review result states.
    permissible_values:
      quorum_consensus: {}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="winning-group eligibility"):
        graph_ops._winning_review_resolution_methods(schema_path)


class _GraphClient:
    def __init__(self, results: list[list[dict[str, Any]]]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> _GraphClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))
        return self.results.pop(0)


def test_projection_is_docs_scoped_canonical_and_idempotent() -> None:
    candidate = {
        "standard_name_id": "electron_temperature",
        "review_group_id": "docs-group",
        "review_id": "electron_temperature:docs:docs-group:1",
        "resolution_method": "quorum_consensus",
    }
    client = _GraphClient([[candidate], [candidate], []])

    with patch.object(graph_ops, "GraphClient", return_value=client):
        first = graph_ops.project_docs_review_resolution_methods()
        second = graph_ops.project_docs_review_resolution_methods()

    assert first == [candidate]
    assert second == []
    assert len(client.calls) == 3

    selection_cypher, selection_params = client.calls[0]
    assert "review.review_axis = 'docs'" in selection_cypher
    assert "canonical_group DESC" in selection_cypher
    assert "review.resolution_method IN $winning_methods" not in selection_cypher
    assert "item.resolution_method IN $winning_methods" in selection_cypher
    assert selection_params["winning_methods"] == [
        "authoritative_escalation",
        "quorum_consensus",
        "single_review",
    ]
    assert selection_params["non_winning_methods"] == [
        "max_cycles_reached",
        "retry_item",
    ]
    assert "non_winner.review_axis = 'docs'" in selection_cypher
    assert "max_cycles_reached" not in selection_params["winning_methods"]
    assert "retry_item" not in selection_params["winning_methods"]

    mutation_cypher, mutation_params = client.calls[1]
    assert "sn.docs_review_resolution_method IS NULL" in mutation_cypher
    assert "sn.name_stage = 'accepted'" in mutation_cypher
    assert "sn.docs_stage = 'accepted'" in mutation_cypher
    assert mutation_params == {"candidates": [candidate]}


def test_non_winning_repair_is_exact_and_axis_scoped() -> None:
    cleared = {
        "standard_name_id": "tritium_density",
        "cleared_method": "single_review",
    }
    client = _GraphClient([[cleared]])

    with patch.object(graph_ops, "GraphClient", return_value=client):
        result = graph_ops.clear_non_winning_docs_review_resolution_methods(
            ["tritium_density"]
        )

    assert result == [cleared]
    cypher, params = client.calls[0]
    assert "review.review_axis = 'docs'" in cypher
    assert "sn.docs_review_resolution_method IS NOT NULL" in cypher
    assert "SET sn.docs_review_resolution_method = null" in cypher
    assert params == {
        "standard_name_ids": ["tritium_density"],
        "non_winning_methods": ["max_cycles_reached", "retry_item"],
    }

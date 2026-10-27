"""Rescore stages only identities whose catalog status can be published."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_store import (
    graph_claim_release as _owning_graph_claim_release,
)
from imas_codex.standard_names.graph_store.graph_claim_release import (
    stage_name_for_rescore,
)


class _GuardedGraph:
    def __init__(self, *, stage: str, status: str, successor: str | None = None):
        self.node = {
            "name_stage": stage,
            "status": status,
        }
        self.successor = successor

    def __enter__(self) -> _GuardedGraph:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "RETURN prior_stage AS prior_stage" in cypher:
            predicate = re.search(r"\bWHERE\s+(.*?)\s+WITH sn,", cypher, re.S)
            assert predicate is not None
            for clause in re.split(r"\s+AND\s+", predicate.group(1)):
                clause = clause.strip()
                stage_match = re.fullmatch(r"sn\.(\w+) IN (\[[^\]]+\])", clause)
                status_match = re.fullmatch(
                    r"NOT coalesce\(sn\.(\w+), '([^']*)'\) IN (\[[^\]]+\])",
                    clause,
                )
                if stage_match:
                    property_name, values = stage_match.groups()
                    if self.node.get(property_name) not in ast.literal_eval(values):
                        return []
                elif status_match:
                    property_name, default, values = status_match.groups()
                    value = self.node.get(property_name)
                    if (default if value is None else value) in ast.literal_eval(
                        values
                    ):
                        return []
                else:
                    raise AssertionError(f"unhandled rescore predicate: {clause}")
            previous = self.node["name_stage"]
            if not params["dry_run"]:
                assignment = re.search(
                    r"SET sn\.updated_at = datetime\(\), sn\.name_stage = '([^']+)'",
                    cypher,
                )
                assert assignment is not None
                self.node["name_stage"] = assignment.group(1)
            return [{"prior_stage": previous}]
        if "RETURN sn.name_stage AS stage" in cypher:
            return [
                {
                    "stage": self.node["name_stage"],
                    "status": self.node["status"],
                    "successor": self.successor,
                }
            ]
        raise AssertionError(f"unexpected query: {cypher}")


@pytest.mark.parametrize("status", ["superseded", "deprecated"])
def test_terminal_status_is_refused_without_mutation(status: str) -> None:
    graph = _GuardedGraph(stage="exhausted", status=status)
    before = graph.node.copy()
    with patch(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        return_value=graph,
    ):
        result = stage_name_for_rescore("plasma_current")

    assert not result["ok"]
    assert graph.node == before
    assert f"status={status!r}" in result["reason"]
    assert "signed transition" in result["reason"]
    assert "signed_manifest.py" in result["reason"]
    assert "successor" not in result["reason"]


def test_draft_status_still_rescores() -> None:
    graph = _GuardedGraph(stage="exhausted", status="draft")
    with patch(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        return_value=graph,
    ):
        result = stage_name_for_rescore("plasma_current", run_id="review-run")

    assert result["ok"]
    assert result["prior_stage"] == "exhausted"
    assert graph.node == {
        "name_stage": "drafted",
        "status": "draft",
    }


def test_superseded_stage_names_signed_transition_and_existing_successor() -> None:
    graph = _GuardedGraph(stage="superseded", status="draft", successor="replacement")
    with patch(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        return_value=graph,
    ):
        result = stage_name_for_rescore("plasma_current")

    assert not result["ok"]
    assert "name_stage='superseded'" in result["reason"]
    assert "signed transition" in result["reason"]
    assert "replacement" in result["reason"]


@pytest.mark.graph
def test_rescore_eligibility_statement_plans_on_server() -> None:
    with GraphClient() as graph, graph.session() as session:
        plan = (
            session.run(
                "EXPLAIN " + _owning_graph_claim_release.RESCORE_ELIGIBILITY_QUERY,
                id="volume_averaged_electron_number_density_over_scrape_off_layer",
                run_id=None,
                dry_run=True,
            )
            .consume()
            .plan
        )

    assert plan is not None


def test_graph_statements_avoid_infix_not_in() -> None:
    source = ast.parse(Path(graph_ops.__file__).read_text())
    invalid = [
        node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "MATCH" in node.value
        and " NOT IN " in node.value
    ]
    assert not invalid

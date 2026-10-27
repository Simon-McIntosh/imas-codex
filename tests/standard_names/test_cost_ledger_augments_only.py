"""The LLM cost ledger survives every Standard Name state reset."""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_store import (
    graph_catalog_reset as _owning_graph_catalog_reset,
    graph_foundation as _owning_graph_foundation,
)

_COST_BINDING = re.compile(r"\(\s*([A-Za-z_]\w*)\s*:\s*LLMCost\b")
_DELETED_BINDING = re.compile(
    r"\b(?:DETACH\s+)?DELETE\s+([A-Za-z_]\w*)\b", re.IGNORECASE
)


def _llm_cost_deletions(package: Path) -> list[str]:
    violations: list[str] = []
    for path in package.rglob("*.py"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            bound_costs = set(_COST_BINDING.findall(node.value))
            deleted = set(_DELETED_BINDING.findall(node.value))
            if bound_costs & deleted:
                violations.append(f"{path}:{node.lineno}")
    return violations


class _GraphState:
    def __init__(self) -> None:
        self.pipeline_rows = {
            "StandardName": 1,
            "StandardNameReview": 1,
            "StandardNameSource": 1,
            "DocsRevision": 1,
            "VocabGap": 1,
            "SNRun": 1,
        }
        self.cost_rows = [1.25, 2.75]
        self.source = {
            "status": "composed",
            "claimed_at": "2026-09-09T10:00:00Z",
            "claim_token": "claim",
            "produced_sn_id": "paid_identity",
        }
        self.queries: list[str] = []

    def __enter__(self) -> _GraphState:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @property
    def ledger_snapshot(self) -> tuple[int, float]:
        return len(self.cost_rows), sum(self.cost_rows)

    def query(self, cypher: str, **_kwargs: Any) -> list[dict[str, int]]:
        self.queries.append(cypher)

        if "MATCH (c:LLMCost) DETACH DELETE c" in cypher:
            self.cost_rows.clear()
            return []

        count_match = re.search(r"MATCH \(n:(\w+)\) RETURN count\(n\) AS n", cypher)
        if count_match:
            label = count_match.group(1)
            if label == "LLMCost":
                return [{"n": len(self.cost_rows)}]
            return [{"n": self.pipeline_rows.get(label, 0)}]

        if "MATCH (sn:StandardName)" in cypher and "RETURN count(sn) AS n" in cypher:
            return [{"n": self.pipeline_rows["StandardName"]}]

        if "DETACH DELETE sn" in cypher:
            self.pipeline_rows["StandardName"] = 0
            self.pipeline_rows["StandardNameReview"] = 0
            return [{"deleted": 1}]

        deletion_aliases = {
            "MATCH (r:StandardNameReview) DETACH DELETE r": "StandardNameReview",
            "MATCH (s:StandardNameSource) DETACH DELETE s": "StandardNameSource",
            "MATCH (d:DocsRevision) DETACH DELETE d": "DocsRevision",
            "MATCH (v:VocabGap) DETACH DELETE v": "VocabGap",
            "MATCH (rr:SNRun) DETACH DELETE rr": "SNRun",
        }
        for statement, label in deletion_aliases.items():
            if statement in cypher:
                self.pipeline_rows[label] = 0
                return []

        if "MATCH (r:StandardNameReview)" in cypher and "DETACH DELETE r" in cypher:
            deleted = self.pipeline_rows["StandardNameReview"]
            self.pipeline_rows["StandardNameReview"] = 0
            return [{"n": deleted}]

        if "SET sns.status = 'extracted'" in cypher:
            self.source.update(
                status="extracted",
                claimed_at=None,
                claim_token=None,
                produced_sn_id=None,
            )
            return [{"n": 1}]

        return []


def test_no_production_statement_deletes_an_llm_cost_node() -> None:
    package = Path(graph_ops.__file__).resolve().parents[1]
    assert _llm_cost_deletions(package) == []


def test_unscoped_clear_preserves_cost_count_and_spend_while_resetting_state() -> None:
    state = _GraphState()
    ledger_before = state.ledger_snapshot

    with patch.object(_owning_graph_foundation, "GraphClient", return_value=state):
        deleted = _owning_graph_catalog_reset.clear_standard_names()

    assert deleted == 1
    assert state.ledger_snapshot == ledger_before
    assert state.pipeline_rows["StandardName"] == 0
    assert state.pipeline_rows["StandardNameReview"] == 0
    assert state.source == {
        "status": "extracted",
        "claimed_at": None,
        "claim_token": None,
        "produced_sn_id": None,
    }


def test_subsystem_clear_preserves_cost_count_and_spend_while_wiping_state() -> None:
    state = _GraphState()
    ledger_before = state.ledger_snapshot

    with patch.object(_owning_graph_foundation, "GraphClient", return_value=state):
        deleted = _owning_graph_catalog_reset.clear_sn_subsystem()

    assert state.ledger_snapshot == ledger_before
    assert set(deleted) == set(state.pipeline_rows)
    assert all(count == 1 for count in deleted.values())
    assert all(count == 0 for count in state.pipeline_rows.values())

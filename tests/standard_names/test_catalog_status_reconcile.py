"""Catalog-status maintenance preserves approval as the only active writer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from imas_codex.standard_names.graph_ops import (
    persist_reviewed_name,
    reconcile_catalog_status,
    stop_refine_name_attempt,
)


class _CatalogGraph:
    def __init__(self, names: list[dict[str, Any]]) -> None:
        self.names = names
        self.queries: list[str] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, int]]:
        del params
        self.queries.append(cypher)
        if "SET sn.status = 'superseded'" in cypher:
            matches = [
                name
                for name in self.names
                if name["name_stage"] == "superseded"
                and name["status"] in (None, "draft")
            ]
            target = "superseded"
        elif (
            "SET sn.status = 'draft'," in cypher
            and "sn.validation_status = 'quarantined'" in cypher
        ):
            matches = [
                name
                for name in self.names
                if name["name_stage"] == "exhausted"
                and (
                    name["status"] != "draft"
                    or name["validation_status"] != "quarantined"
                )
            ]
            for name in matches:
                name["status"] = "draft"
                name["validation_status"] = "quarantined"
            return [{"changed": len(matches)}]
        elif "SET sn.status = 'draft'" in cypher:
            matches = [name for name in self.names if name["status"] is None]
            target = "draft"
        else:
            raise AssertionError(f"unexpected query: {cypher}")

        for name in matches:
            name["status"] = target
        return [{"changed": len(matches)}]


def test_reconcile_maps_unset_and_terminal_statuses_idempotently() -> None:
    names = [
        {
            "id": "live_unset",
            "name_stage": "accepted",
            "status": None,
            "validation_status": "valid",
        },
        {
            "id": "live_draft",
            "name_stage": "reviewed",
            "status": "draft",
            "validation_status": "valid",
        },
        {
            "id": "superseded_unset",
            "name_stage": "superseded",
            "status": None,
            "validation_status": "valid",
        },
        {
            "id": "superseded_draft",
            "name_stage": "superseded",
            "status": "draft",
            "validation_status": "valid",
        },
        {
            "id": "superseded_terminal",
            "name_stage": "superseded",
            "status": "superseded",
            "validation_status": "valid",
        },
        {
            "id": "exhausted_unset",
            "name_stage": "exhausted",
            "status": None,
            "validation_status": "valid",
        },
        {
            "id": "exhausted_draft",
            "name_stage": "exhausted",
            "status": "draft",
            "validation_status": "valid",
        },
        {
            "id": "exhausted_terminal",
            "name_stage": "exhausted",
            "status": "deprecated",
            "validation_status": "valid",
        },
        {
            "id": "exhausted_quarantined",
            "name_stage": "exhausted",
            "status": "deprecated",
            "validation_status": "quarantined",
        },
        {
            "id": "live_active",
            "name_stage": "approved",
            "status": "active",
            "validation_status": "valid",
        },
        {
            "id": "superseded_active",
            "name_stage": "superseded",
            "status": "active",
            "validation_status": "valid",
        },
        {
            "id": "exhausted_active",
            "name_stage": "exhausted",
            "status": "active",
            "validation_status": "valid",
        },
    ]
    graph = _CatalogGraph(names)

    assert reconcile_catalog_status(gc=graph) == {
        "drafted": 1,
        "superseded": 2,
        "quarantined": 5,
        "deprecated": 0,
        "total_changed": 8,
    }
    assert {
        name["id"]: (name["status"], name["validation_status"]) for name in names
    } == {
        "live_unset": ("draft", "valid"),
        "live_draft": ("draft", "valid"),
        "superseded_unset": ("superseded", "valid"),
        "superseded_draft": ("superseded", "valid"),
        "superseded_terminal": ("superseded", "valid"),
        "exhausted_unset": ("draft", "quarantined"),
        "exhausted_draft": ("draft", "quarantined"),
        "exhausted_terminal": ("draft", "quarantined"),
        "exhausted_quarantined": ("draft", "quarantined"),
        "live_active": ("active", "valid"),
        "superseded_active": ("active", "valid"),
        "exhausted_active": ("draft", "quarantined"),
    }
    assert reconcile_catalog_status(gc=graph) == {
        "drafted": 0,
        "superseded": 0,
        "quarantined": 0,
        "deprecated": 0,
        "total_changed": 0,
    }
    assert all("SET sn.status = 'active'" not in query for query in graph.queries)
    assert all("SET sn.status = 'deprecated'" not in query for query in graph.queries)


def _context_graph(*query_results: list[dict[str, Any]]) -> MagicMock:
    graph = MagicMock()
    graph.__enter__ = MagicMock(return_value=graph)
    graph.__exit__ = MagicMock(return_value=False)
    graph.query = MagicMock(side_effect=query_results)
    return graph


def test_review_exhaustion_quarantines_at_the_stage_write() -> None:
    graph = _context_graph(
        [
            {
                "id": "electron_temperature",
                "chain_length": 0,
                "refine_attempts": 3,
                "validation_status": "valid",
                "edit_status": None,
                "edit_scope": None,
                "edit_mode": None,
                "edit_override_edits": False,
                "edit_include_accepted": False,
            }
        ],
        [{"id": "electron_temperature"}],
    )

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        stage = persist_reviewed_name(
            sn_id="electron_temperature",
            claim_token="claim-token",
            score=0.5,
            model="reviewer",
            rotation_cap=3,
            skip_review_node=True,
            resolution_method="quorum_consensus",
            reviewer_chain_size=2,
        )

    write_query = graph.query.call_args_list[1].args[0]
    assert stage == "exhausted"
    assert "WHEN $target_stage = 'exhausted' THEN 'quarantined'" in write_query


def test_stopped_refinement_quarantines_when_it_exhausts() -> None:
    graph = _context_graph([{"stage": "exhausted"}])

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        stage = stop_refine_name_attempt(
            sn_id="electron_temperature",
            token="claim-token",
            reason="attempts_exhausted",
            rotation_cap=3,
        )

    write_query = graph.query.call_args.args[0]
    assert stage == "exhausted"
    assert "WHEN target_stage = 'exhausted' THEN 'quarantined'" in write_query

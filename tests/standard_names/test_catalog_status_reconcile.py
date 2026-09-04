"""Catalog-status maintenance preserves approval as the only active writer."""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.graph_ops import reconcile_catalog_status


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
        elif "SET sn.status = 'deprecated'" in cypher:
            matches = [
                name
                for name in self.names
                if name["name_stage"] == "exhausted"
                and name["status"] in (None, "draft")
            ]
            target = "deprecated"
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
        {"id": "live_unset", "name_stage": "accepted", "status": None},
        {"id": "live_draft", "name_stage": "reviewed", "status": "draft"},
        {"id": "superseded_unset", "name_stage": "superseded", "status": None},
        {"id": "superseded_draft", "name_stage": "superseded", "status": "draft"},
        {
            "id": "superseded_terminal",
            "name_stage": "superseded",
            "status": "superseded",
        },
        {"id": "exhausted_unset", "name_stage": "exhausted", "status": None},
        {"id": "exhausted_draft", "name_stage": "exhausted", "status": "draft"},
        {
            "id": "exhausted_terminal",
            "name_stage": "exhausted",
            "status": "deprecated",
        },
        {"id": "live_active", "name_stage": "approved", "status": "active"},
        {
            "id": "superseded_active",
            "name_stage": "superseded",
            "status": "active",
        },
        {"id": "exhausted_active", "name_stage": "exhausted", "status": "active"},
    ]
    graph = _CatalogGraph(names)

    assert reconcile_catalog_status(gc=graph) == {
        "drafted": 1,
        "superseded": 2,
        "deprecated": 2,
        "total_changed": 5,
    }
    assert {name["id"]: name["status"] for name in names} == {
        "live_unset": "draft",
        "live_draft": "draft",
        "superseded_unset": "superseded",
        "superseded_draft": "superseded",
        "superseded_terminal": "superseded",
        "exhausted_unset": "deprecated",
        "exhausted_draft": "deprecated",
        "exhausted_terminal": "deprecated",
        "live_active": "active",
        "superseded_active": "active",
        "exhausted_active": "active",
    }
    assert reconcile_catalog_status(gc=graph) == {
        "drafted": 0,
        "superseded": 0,
        "deprecated": 0,
        "total_changed": 0,
    }
    assert all("SET sn.status = 'active'" not in query for query in graph.queries)

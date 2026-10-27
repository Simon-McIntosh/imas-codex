"""Catalog-status maintenance preserves approval as the only active writer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from imas_codex.standard_names.export import _classify_export_population
from imas_codex.standard_names.graph_store.graph_audits import reconcile_catalog_status
from imas_codex.standard_names.graph_store.graph_catalog_reset import (
    mark_names_validated,
)
from imas_codex.standard_names.graph_store.graph_claim_release import (
    stop_refine_name_attempt,
)
from imas_codex.standard_names.graph_store.graph_review_persistence import (
    persist_reviewed_name,
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
        elif "sn.name_stage = 'exhausted'" in cypher:
            matches = [
                name
                for name in self.names
                if name["name_stage"] == "exhausted" and name["status"] != "draft"
            ]
            for name in matches:
                name["status"] = "draft"
            return [{"changed": len(matches)}]
        elif "HAS_INTERNAL_CHANGE" in cypher:
            matches = [
                name
                for name in self.names
                if name["name_stage"] not in (None, "superseded", "exhausted")
                and name["status"] == "superseded"
                and name.get("fold_receipt", False)
            ]
            target = "draft"
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
        "drafted": 5,
        "revived": 0,
        "superseded": 2,
        "quarantined": 0,
        "deprecated": 0,
        "total_changed": 7,
    }
    assert {
        name["id"]: (name["status"], name["validation_status"]) for name in names
    } == {
        "live_unset": ("draft", "valid"),
        "live_draft": ("draft", "valid"),
        "superseded_unset": ("superseded", "valid"),
        "superseded_draft": ("superseded", "valid"),
        "superseded_terminal": ("superseded", "valid"),
        "exhausted_unset": ("draft", "valid"),
        "exhausted_draft": ("draft", "valid"),
        "exhausted_terminal": ("draft", "valid"),
        "exhausted_quarantined": ("draft", "quarantined"),
        "live_active": ("active", "valid"),
        "superseded_active": ("active", "valid"),
        "exhausted_active": ("draft", "valid"),
    }
    assert reconcile_catalog_status(gc=graph) == {
        "drafted": 0,
        "revived": 0,
        "superseded": 0,
        "quarantined": 0,
        "deprecated": 0,
        "total_changed": 0,
    }
    assert all("SET sn.status = 'active'" not in query for query in graph.queries)
    assert all("SET sn.status = 'deprecated'" not in query for query in graph.queries)


def test_reconcile_does_not_turn_exhaustion_into_validation_failure() -> None:
    names = [
        {
            "id": "exhausted_valid",
            "name_stage": "exhausted",
            "status": None,
            "validation_status": "valid",
            "validation_issues": [],
        },
        {
            "id": "exhausted_failed",
            "name_stage": "exhausted",
            "status": None,
            "validation_status": "quarantined",
            "validation_issues": ["[semantic] invalid unit"],
        },
    ]
    graph = _CatalogGraph(names)

    reconcile_catalog_status(gc=graph)

    assert names == [
        {
            "id": "exhausted_valid",
            "name_stage": "exhausted",
            "status": "draft",
            "validation_status": "valid",
            "validation_issues": [],
        },
        {
            "id": "exhausted_failed",
            "name_stage": "exhausted",
            "status": "draft",
            "validation_status": "quarantined",
            "validation_issues": ["[semantic] invalid unit"],
        },
    ]
    assert all(
        "validation_status = 'quarantined'" not in query for query in graph.queries
    )


def test_validation_failure_writer_keeps_quarantine_and_issues() -> None:
    graph = _context_graph([{"marked": 1}])

    with patch(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        return_value=graph,
    ):
        assert (
            mark_names_validated(
                "validation-token",
                [
                    {
                        "id": "failed_name",
                        "validation_issues": ["[semantic] invalid unit"],
                        "validation_layer_summary": {"semantic": {"passed": False}},
                        "validation_status": "quarantined",
                    }
                ],
            )
            == 1
        )

    query = graph.query.call_args.args[0]
    params = graph.query.call_args.kwargs
    assert "sn.validation_issues = b.issues" in query
    assert "sn.validation_status = b.validation_status" in query
    assert params["batch"][0]["validation_status"] == "quarantined"
    assert params["batch"][0]["issues"] == ["[semantic] invalid unit"]


def test_export_reader_withholds_valid_but_exhausted_names() -> None:
    eligible, excluded = _classify_export_population(
        [
            {
                "id": "exhausted_valid",
                "name_stage": "exhausted",
                "status": "draft",
                "validation_status": "valid",
                "_validation_observed_at": "2026-09-09T00:00:00Z",
                "physics_domain": [],
                "_has_dd_source_binding": True,
                "_has_derived_producer": False,
                "_has_non_derived_producer": True,
                "_has_live_child": False,
                "_is_parent": False,
            },
            {
                "id": "accepted_valid",
                "name_stage": "accepted",
                "status": "draft",
                "validation_status": "valid",
                "_validation_observed_at": "2026-09-09T00:00:00Z",
                "physics_domain": [],
                "_has_dd_source_binding": True,
                "_has_derived_producer": False,
                "_has_non_derived_producer": True,
                "_has_live_child": False,
                "_is_parent": False,
            },
        ],
        domain=None,
        names_only=True,
    )

    assert [row["id"] for row in eligible] == ["accepted_valid"]
    assert [(row.standard_name_id, row.reason) for row in excluded] == [
        ("exhausted_valid", "name_not_accepted")
    ]


def _context_graph(*query_results: list[dict[str, Any]]) -> MagicMock:
    graph = MagicMock()
    graph.__enter__ = MagicMock(return_value=graph)
    graph.__exit__ = MagicMock(return_value=False)
    graph.query = MagicMock(side_effect=query_results)
    return graph


def test_review_exhaustion_does_not_quarantine_at_the_stage_write() -> None:
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

    with patch(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        return_value=graph,
    ):
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
    assert "WHEN $grammar_issue IS NOT NULL THEN 'quarantined'" in write_query
    assert "WHEN $target_stage = 'exhausted' THEN 'quarantined'" not in write_query


def test_stopped_refinement_does_not_quarantine_when_it_exhausts() -> None:
    graph = _context_graph([{"stage": "exhausted"}])

    with patch(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        return_value=graph,
    ):
        stage = stop_refine_name_attempt(
            sn_id="electron_temperature",
            token="claim-token",
            reason="attempts_exhausted",
            rotation_cap=3,
        )

    write_query = graph.query.call_args.args[0]
    assert stage == "exhausted"
    write_query = " ".join(write_query.split())
    assert (
        "WHEN target_stage = 'exhausted'"
        " AND $reason IN ['grammar_invalid', 'vocabulary_gap']" in write_query
    )
    assert "THEN 'quarantined'" in write_query

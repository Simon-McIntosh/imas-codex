"""Disposition for sources parked at the compose attempt cap."""

from __future__ import annotations

import pytest

from imas_codex.standard_names import graph_ops


class StubClient:
    """Canned graph: reads return parked rows, writes are recorded."""

    def __init__(self, rows):
        self.rows = rows
        self.writes = []

    def query(self, cypher, **params):
        if "SET sns.parked_disposition" in cypher:
            self.writes.append(list(params["rows"]))
            return [{"written": len(params["rows"])}]
        return list(self.rows)

    def close(self):
        return None


def test_each_class_is_a_member_and_is_reached() -> None:
    c = graph_ops.classify_parked_source
    cases = {
        "name_produced": {"produced": 1},
        "upstream_quantity_removed": {"lifecycle_status": "removed"},
        "compose_not_applicable": {"node_category": "geometry"},
        "vocabulary_gap": {"last_error": "a vocabulary gap in beta"},
        "attempt_budget_exhausted": {"last_error": "timed out"},
        "cause_not_recorded": {},
    }
    assert set(cases) == graph_ops.CAP_PARKED_DISPOSITIONS
    for expected, evidence in cases.items():
        assert c(evidence) == expected
    assert c({"last_error": graph_ops._COMPOSE_CAP_REASON}) == "cause_not_recorded"
    assert c({"node_category": "geometry", "last_error": "a vocabulary gap"}) == "vocabulary_gap"


def test_classifier_is_total_over_unhandled_shapes() -> None:
    for shape in (
        {},
        {"produced": None, "lifecycle_status": "", "last_error": "  "},
        {"produced": 0, "node_category": "quantity"},
        {"node_category": None, "last_error": None},
    ):
        assert graph_ops.classify_parked_source(shape) in graph_ops.CAP_PARKED_DISPOSITIONS


def test_pass_refuses_a_row_the_classifier_cannot_place(monkeypatch) -> None:
    stub = StubClient([{"id": "dd:a"}, {"id": "dd:b"}])
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: stub)
    monkeypatch.setattr(graph_ops, "classify_parked_source", lambda e: "")
    with pytest.raises(ValueError):
        graph_ops.disposition_parked_sources()
    assert stub.writes == []


def test_pass_writes_a_member_for_every_parked_row(monkeypatch) -> None:
    rows = [{"id": "dd:a", "produced": 1}, {"id": "dd:b", "lifecycle_status": "removed"}]
    stub = StubClient(rows)
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: stub)
    out = graph_ops.disposition_parked_sources()
    assert out["parked"] == 2
    assert out["written"] == 2
    assert {
        r["id"]: r["disposition"] for r in stub.writes[0]
    } == {"dd:a": "name_produced", "dd:b": "upstream_quantity_removed"}


def test_census_reports_a_row_carrying_no_disposition(monkeypatch) -> None:
    rows = [{"id": "dd:a", "disposition": "name_produced"}, {"id": "dd:b", "disposition": None}]
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: StubClient(rows))
    out = graph_ops.census_parked_dispositions()
    assert out["parked"] == 2
    assert out["unclassified"] == ["dd:b"]
    assert out["unexpected"] == []
    assert out["counts"] == {"name_produced": 1}
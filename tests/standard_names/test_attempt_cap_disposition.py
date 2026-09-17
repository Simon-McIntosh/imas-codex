"""Disposition for sources parked at the compose attempt cap."""

from __future__ import annotations

import pytest

from imas_codex.standard_names import graph_ops, orphan_sweep


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
    assert (
        c({"node_category": "geometry", "last_error": "a vocabulary gap"})
        == "vocabulary_gap"
    )


def test_classifier_is_total_over_unhandled_shapes() -> None:
    for shape in (
        {},
        {"produced": None, "lifecycle_status": "", "last_error": "  "},
        {"produced": 0, "node_category": "quantity"},
        {"node_category": None, "last_error": None},
    ):
        assert (
            graph_ops.classify_parked_source(shape) in graph_ops.CAP_PARKED_DISPOSITIONS
        )


def test_pass_refuses_a_row_the_classifier_cannot_place(monkeypatch) -> None:
    stub = StubClient([{"id": "dd:a"}, {"id": "dd:b"}])
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: stub)
    monkeypatch.setattr(graph_ops, "classify_parked_source", lambda e: "")
    with pytest.raises(ValueError):
        graph_ops.disposition_parked_sources()
    assert stub.writes == []


def test_pass_writes_a_member_for_every_parked_row(monkeypatch) -> None:
    rows = [
        {"id": "dd:a", "produced": 1},
        {"id": "dd:b", "lifecycle_status": "removed"},
    ]
    stub = StubClient(rows)
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: stub)
    out = graph_ops.disposition_parked_sources()
    assert out["parked"] == 2
    assert out["written"] == 2
    assert {r["id"]: r["disposition"] for r in stub.writes[0]} == {
        "dd:a": "name_produced",
        "dd:b": "upstream_quantity_removed",
    }


def test_census_reports_a_row_carrying_no_disposition(monkeypatch) -> None:
    rows = [
        {"id": "dd:a", "disposition": "name_produced"},
        {"id": "dd:b", "disposition": None},
    ]
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: StubClient(rows))
    out = graph_ops.census_parked_dispositions()
    assert out["parked"] == 2
    assert out["unclassified"] == ["dd:b"]
    assert out["unexpected"] == []
    assert out["counts"] == {"name_produced": 1}


class ParkingStub:
    """Graph that parks *park_count* sources and serves what it parked."""

    def __init__(self, *, parked_rows, park_count):
        self.parked_rows = parked_rows
        self.park_count = park_count
        self.queries: list[str] = []
        self.writes: list[list[dict[str, str]]] = []

    def query(self, cypher, **params):
        self.queries.append(cypher)
        if "parked_disposition IS NULL" in cypher:
            return [dict(r) for r in self.parked_rows]
        if "SET sns.parked_disposition" in cypher:
            self.writes.append(list(params["rows"]))
            return [{"written": len(params["rows"])}]
        if graph_ops._COMPOSE_CAP_REASON in cypher:
            return [{"n": self.park_count}]
        return [{"n": 0}]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        return None


def _park(monkeypatch, *, parked_rows, park_count):
    stub = ParkingStub(parked_rows=parked_rows, park_count=park_count)
    monkeypatch.setattr(orphan_sweep, "GraphClient", lambda: stub)
    return stub


def test_parking_writer_stamps_a_disposition_as_it_parks(monkeypatch) -> None:
    """The sweep parks a source and records its disposition in the same tick."""
    stub = _park(
        monkeypatch,
        parked_rows=[
            {"id": "dd:geom", "node_category": "geometry"},
            {"id": "dd:name", "produced": 1},
        ],
        park_count=2,
    )

    counts = orphan_sweep._orphan_sweep_tick(timeout_s=300)

    assert counts["compose_attempt_cap"] == 2
    assert len(stub.writes) == 1
    stamped = stub.writes[0]
    assert all(r["disposition"] in graph_ops.CAP_PARKED_DISPOSITIONS for r in stamped)
    assert {r["id"]: r["disposition"] for r in stamped} == {
        "dd:geom": "compose_not_applicable",
        "dd:name": "name_produced",
    }


def test_parking_writer_writes_nothing_when_it_parks_nothing(monkeypatch) -> None:
    stub = _park(monkeypatch, parked_rows=[], park_count=0)

    counts = orphan_sweep._orphan_sweep_tick(timeout_s=300)

    assert counts["compose_attempt_cap"] == 0
    assert stub.writes == []
    assert not [q for q in stub.queries if "parked_disposition IS NULL" in q]


def test_parking_writer_refuses_a_row_it_cannot_place(monkeypatch) -> None:
    stub = _park(
        monkeypatch,
        parked_rows=[{"id": "dd:x", "node_category": "geometry"}],
        park_count=1,
    )
    monkeypatch.setattr(graph_ops, "classify_parked_source", lambda e: "")

    with pytest.raises(ValueError):
        orphan_sweep._orphan_sweep_tick(timeout_s=300)

    assert stub.writes == []

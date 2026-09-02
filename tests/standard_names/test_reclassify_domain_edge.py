"""Atomic domain-property and relationship persistence tests."""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names import graph_ops


class _Graph:
    def __init__(self, *, fail_mutation: bool = False) -> None:
        self.domain = "general"
        self.domain_edges = {"general"}
        self.fail_mutation = fail_mutation
        self.queries: list[str] = []

    def __enter__(self) -> _Graph:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if "RETURN name.physics_domain AS domain" in cypher:
            return [{"domain": self.domain, "stage": "accepted"}]

        assert "SET name.physics_domain = $domain" in cypher
        assert "MERGE (name)-[:HAS_PHYSICS_DOMAIN]->(domain)" in cypher
        if self.fail_mutation:
            raise RuntimeError("edge write failed")

        self.domain = params["domain"]
        self.domain_edges = {params["domain"]}
        return [{"id": params["id"]}]


def test_reclassify_writes_property_and_edge_in_one_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph()
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)

    result = graph_ops.reclassify_standard_name_domain(
        "plasma_current",
        "magnetic_field_diagnostics",
        reason="The signal measures plasma current.",
    )

    assert result["ok"] is True
    assert graph.domain == "magnetic_field_diagnostics"
    assert graph.domain_edges == {"magnetic_field_diagnostics"}
    assert len(graph.queries) == 2


def test_failed_edge_write_rolls_back_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph(fail_mutation=True)
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: graph)

    with pytest.raises(RuntimeError, match="edge write failed"):
        graph_ops.reclassify_standard_name_domain(
            "plasma_current",
            "magnetic_field_diagnostics",
            reason="The signal measures plasma current.",
        )

    assert graph.domain == "general"
    assert graph.domain_edges == {"general"}

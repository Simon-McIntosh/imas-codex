"""Atomic domain-property and relationship persistence tests."""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names import graph_ops  # noqa: F401 — wires the services
from imas_codex.standard_names.graph_store import (
    graph_foundation as _owning_graph_foundation,
)


class _Graph:
    """A graph that applies the domain write only when it is one statement.

    A statement that sets ``physics_domain`` without touching the
    ``HAS_PHYSICS_DOMAIN`` edges — or the reverse — is recorded as split, so a
    site that writes the property and the edge separately fails the atomicity
    assertion rather than passing on the state it happens to leave behind.
    """

    def __init__(self, *, fail_mutation: bool = False) -> None:
        self.domain = "general"
        self.domain_edges = {"general"}
        self.source_domains = ["general"]
        self.correction = None
        self.fail_mutation = fail_mutation
        self.split_statements = 0
        self.queries: list[str] = []

    def __enter__(self) -> _Graph:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if "RETURN name.physics_domain AS domain, name.name_stage AS stage" in cypher:
            return [{"domain": self.domain, "stage": "accepted"}]
        if "RETURN name.id AS id" in cypher:
            return [{"id": params["id"]}]
        if "RETURN sn.physics_domain AS physics_domain" in cypher:
            return [
                {
                    "physics_domain": self.domain,
                    "own_domains": ["equilibrium"],
                    "child_ids": [],
                    "child_dd_domains": [],
                    "child_stored_domains": [],
                }
            ]

        sets_domain = "sn.physics_domain = " in cypher
        writes_edges = "HAS_PHYSICS_DOMAIN" in cypher
        if sets_domain != writes_edges:
            self.split_statements += 1
        if self.fail_mutation:
            raise RuntimeError("edge write failed")
        if not (sets_domain and writes_edges):
            raise AssertionError(f"unexpected domain query: {cypher}")

        self.domain = params["physics_domain"]
        self.source_domains = list(params["source_domains"])
        self.domain_edges = set(params["edge_domains"])
        self.correction = params["correction"]
        return [{"sn_id": params["sn_id"]}]


def test_reclassify_writes_property_and_edge_in_one_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph()
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda: graph)

    result = _owning_graph_foundation.reclassify_standard_name_domain(
        "plasma_current",
        "magnetic_field_diagnostics",
        reason="The signal measures plasma current.",
    )

    assert result["ok"] is True
    assert graph.split_statements == 0
    assert graph.domain == "magnetic_field_diagnostics"
    assert graph.correction == "magnetic_field_diagnostics"
    assert set(graph.domain_edges) == set(graph.source_domains)


def test_failed_edge_write_rolls_back_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph(fail_mutation=True)
    monkeypatch.setattr(_owning_graph_foundation, "GraphClient", lambda: graph)

    with pytest.raises(RuntimeError, match="edge write failed"):
        _owning_graph_foundation.reclassify_standard_name_domain(
            "plasma_current",
            "magnetic_field_diagnostics",
            reason="The signal measures plasma current.",
        )

    assert graph.split_statements == 0
    assert graph.domain == "general"
    assert graph.domain_edges == {"general"}
    assert graph.correction is None

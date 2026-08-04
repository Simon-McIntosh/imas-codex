"""Disposable-Neo4j compilation and access-plan checks for exact run audit."""

from __future__ import annotations

import os

import pytest

from imas_codex.standard_names.run_audit import (
    _DELTA_EVIDENCE_QUERY,
    _RUN_EVIDENCE_QUERY,
    _TARGET_EVIDENCE_QUERY,
)

pytestmark = pytest.mark.graph


def _operator_types(plan: object) -> list[str]:
    if plan is None:
        return []
    if isinstance(plan, dict):
        operator = plan.get("operatorType") or plan.get("operator_type")
        children = plan.get("children", [])
    else:
        operator = getattr(plan, "operator_type", None)
        children = getattr(plan, "children", [])
    operators = [str(operator)] if operator else []
    for child in children:
        operators.extend(_operator_types(child))
    return operators


def test_disposable_graph_compiles_all_bounded_queries_without_global_scans() -> None:
    """EXPLAIN catches lost WITH variables without reading production data."""
    from imas_codex.graph.client import GraphClient

    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("exact run audit graph test requires an ephemeral graph")

    params = {
        "name_id": "exact_run_audit_compile_fixture",
        "scope_uuid": "80a80eaa-f2d4-4162-8027-45ee1ae2d07e",
        "run_id_prefix": "run-prefix",
        "launched_at": "2026-08-04T00:00:00+00:00",
        "completed_at": "2026-08-04T00:05:00+00:00",
        "west_source_ids": [],
        "fixture_source_id_prefix": "dd:test_review_entry__",
    }
    plans: dict[str, list[str]] = {}
    with GraphClient(
        uri=uri,
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", ""),
        graph_name="ephemeral-exact-run-audit",
    ) as gc:
        with gc.session() as session:
            for label, query in (
                ("target", _TARGET_EVIDENCE_QUERY),
                ("run", _RUN_EVIDENCE_QUERY),
                ("deltas", _DELTA_EVIDENCE_QUERY),
            ):
                result = session.run("EXPLAIN " + query, **params)
                plans[label] = _operator_types(result.consume().plan)

    assert set(plans) == {"target", "run", "deltas"}
    for operators in plans.values():
        assert "AllNodesScan" not in operators
        assert not any("AllRelationshipsScan" in operator for operator in operators)
        assert not any("VarLengthExpand" in operator for operator in operators)

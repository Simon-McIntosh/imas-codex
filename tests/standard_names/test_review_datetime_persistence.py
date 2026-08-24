"""Timestamp type contracts for standard-name review persistence."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from neo4j import GraphDatabase
from neo4j.time import DateTime

from imas_codex.graph.client import GraphClient as RealGraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names import graph_ops


def _record(
    standard_name_id: str,
    *,
    axis: str,
    cycle_index: int,
    reviewed_at: str,
) -> dict[str, object]:
    group_id = "datetime-persistence"
    model = "openrouter/example/reviewer"
    return {
        "id": f"{standard_name_id}:{axis}:{group_id}:{cycle_index}",
        "standard_name_id": standard_name_id,
        "model": model,
        "reviewer_model": model,
        "model_family": "other",
        "is_canonical": True,
        "score": 0.9,
        "scores_json": "{}",
        "tier": "outstanding",
        "comments": "",
        "comments_per_dim_json": None,
        "suggested_name": "",
        "suggestion_justification": "",
        "reviewed_at": reviewed_at,
        "review_axis": axis,
        "cycle_index": cycle_index,
        "review_group_id": group_id,
        "resolution_role": "primary",
        "resolution_method": "single_review",
        "llm_model": model,
        "llm_cost": 0.0,
        "llm_tokens_in": 0,
        "llm_tokens_out": 0,
        "llm_tokens_cached_read": 0,
        "llm_tokens_cached_write": 0,
        "llm_at": reviewed_at,
        "llm_service": "standard-names",
    }


@pytest.mark.parametrize("axis", ["name", "docs"])
def test_write_reviews_converts_iso_parameters_to_neo4j_datetimes(axis: str) -> None:
    reviewed_at = "2026-08-03T21:10:35.157740+00:00"
    record = _record(
        f"datetime_{axis}",
        axis=axis,
        cycle_index=0,
        reviewed_at=reviewed_at,
    )
    graph = MagicMock()
    graph.query.return_value = []

    with patch.object(graph_ops, "GraphClient") as graph_client:
        graph_client.return_value.__enter__.return_value = graph
        graph_client.return_value.__exit__.return_value = False
        assert graph_ops.write_reviews([record]) == 1

    write_call = next(
        call
        for call in graph.query.call_args_list
        if call.args and "MERGE (r:StandardNameReview" in call.args[0]
    )
    cypher = write_call.args[0]
    batch = write_call.kwargs["batch"]

    assert "ELSE datetime(b.reviewed_at)" in cypher
    assert "ELSE datetime(b.llm_at)" in cypher
    assert "WHEN b.reviewed_at IS NULL THEN NULL" in cypher
    assert "WHEN b.llm_at IS NULL THEN NULL" in cypher
    assert batch[0]["id"] == record["id"]
    assert batch[0]["reviewed_at"] == reviewed_at
    assert batch[0]["llm_at"] == reviewed_at


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("review timestamp persistence requires a disposable graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("review timestamp persistence refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


@pytest.mark.graph
def test_name_and_docs_reviews_persist_as_ordered_datetimes(
    disposable_neo4j: tuple[str, str],
) -> None:
    uri, password = disposable_neo4j
    fixture_id = f"review_datetime_{uuid.uuid4().hex}"
    standard_name_ids = [f"{fixture_id}_name", f"{fixture_id}_docs"]
    records = [
        _record(
            standard_name_ids[0],
            axis="name",
            cycle_index=0,
            reviewed_at="2026-08-03T21:10:34.157740+00:00",
        ),
        _record(
            standard_name_ids[1],
            axis="docs",
            cycle_index=0,
            reviewed_at="2026-08-03T21:10:35.157740+00:00",
        ),
    ]
    review_ids = [str(record["id"]) for record in records]

    def client() -> RealGraphClient:
        return RealGraphClient(
            uri=uri,
            username="neo4j",
            password=password,
            graph_name="disposable-review-datetime",
        )

    seed_client = client()
    try:
        seed_client.query(
            "UNWIND $ids AS id CREATE (:StandardName {id: id})",
            ids=standard_name_ids,
        )
        with patch.object(graph_ops, "GraphClient", side_effect=client):
            assert graph_ops.write_reviews(records) == 2

        rows = seed_client.query(
            """
            UNWIND $ids AS id
            MATCH (:StandardName {id: id})-[:HAS_REVIEW]->(review:StandardNameReview)
            RETURN review.id AS id,
                   review.review_axis AS axis,
                   review.reviewed_at AS reviewed_at,
                   review.llm_at AS llm_at,
                   valueType(review.reviewed_at) AS reviewed_at_type,
                   valueType(review.llm_at) AS llm_at_type
            ORDER BY review.reviewed_at, review.id
            """,
            ids=standard_name_ids,
        )

        assert [row["axis"] for row in rows] == ["name", "docs"]
        assert [row["id"] for row in rows] == review_ids
        assert all(row["reviewed_at_type"] == "ZONED DATETIME NOT NULL" for row in rows)
        assert all(row["llm_at_type"] == "ZONED DATETIME NOT NULL" for row in rows)
        assert all(isinstance(row["reviewed_at"], DateTime) for row in rows)
        assert all(isinstance(row["llm_at"], DateTime) for row in rows)
        assert rows[0]["reviewed_at"] < rows[1]["reviewed_at"]
        assert rows[0]["llm_at"] < rows[1]["llm_at"]
    finally:
        seed_client.query(
            "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
            ids=standard_name_ids + review_ids,
        )
        assert seed_client.query(
            "MATCH (node) WHERE node.id IN $ids RETURN count(node) AS count",
            ids=standard_name_ids + review_ids,
        ) == [{"count": 0}]
        seed_client.close()

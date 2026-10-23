"""Documentation vetoes are durable graph records, independent of scores."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from neo4j import GraphDatabase

from imas_codex.graph import temp_neo4j
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.review import holds

pytestmark = [pytest.mark.graph, pytest.mark.timeout(180)]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[str]:
    """Start one empty graph, and refuse any connection to the project graph."""
    scratch = Path(os.environ["TMPDIR"])
    with tempfile.TemporaryDirectory(prefix="review-holds-", dir=scratch) as directory:
        root = Path(directory)
        for part in ("data", "logs", "conf", "run", "tmp"):
            (root / part).mkdir()
        bolt_port, http_port = _free_port(), _free_port()
        temp_neo4j.write_temp_neo4j_conf(root / "conf", bolt_port, http_port)
        image = temp_neo4j._neo4j_image()
        command = [
            "apptainer",
            "exec",
            "--bind",
            f"{root}/data:/data",
            "--bind",
            f"{root}/logs:/logs",
            "--bind",
            f"{root}/conf:/var/lib/neo4j/conf",
            "--bind",
            f"{root}/run:/var/lib/neo4j/run",
            "--bind",
            f"{root}/tmp:/tmp",
            "--writable-tmpfs",
            str(image),
            "neo4j",
            "console",
        ]
        server_log = root / "logs" / "neo4j-temp.log"
        with server_log.open("w") as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            temp_neo4j._wait_for_bolt_ready(process, bolt_port, server_log)
            uri = f"bolt://127.0.0.1:{bolt_port}"
            with GraphDatabase.driver(uri, auth=None) as driver:
                records, _, _ = driver.execute_query(
                    "MATCH (node) RETURN count(node) AS n"
                )
                assert records[0]["n"] == 0
            previous = {
                key: os.environ.get(key)
                for key in (
                    "NEO4J_URI",
                    "NEO4J_PASSWORD",
                    "IMAS_CODEX_TEST_NEO4J_URI",
                    "IMAS_CODEX_TEST_NEO4J_EPHEMERAL",
                )
            }
            os.environ.update(
                {
                    "NEO4J_URI": uri,
                    "NEO4J_PASSWORD": "",
                    "IMAS_CODEX_TEST_NEO4J_URI": uri,
                    "IMAS_CODEX_TEST_NEO4J_EPHEMERAL": "1",
                }
            )
            try:
                with GraphClient() as graph:
                    assert graph.uri == uri
                yield uri
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        finally:
            temp_neo4j.stop_temp_neo4j(process)
            if destination := os.environ.get("IMAS_CODEX_TEST_NEO4J_LOG"):
                shutil.copyfile(server_log, destination)


@pytest.fixture
def planted_name(disposable_neo4j: str) -> Iterator[str]:
    name_id = "electrostatic_potential"
    with GraphClient() as graph:
        graph.query("MATCH (node) DETACH DELETE node")
        graph.query(
            """
            CREATE (sn:StandardName {id: $id, name_stage: 'accepted',
                docs_stage: 'drafted', claim_token: 'token',
                docs_chain_length: 0, documentation: 'A potential.',
                description: 'Potential.', physics_domain: 'plasma',
                reviewer_score_name: 0.9})
            RETURN sn.id AS id
            """,
            id=name_id,
        )
        assert graph.query("MATCH (sn:StandardName) RETURN count(sn) AS n")[0]["n"] == 1
    yield name_id


def _review(name_id: str, score: float = 0.875) -> str:
    stage = graph_ops.persist_reviewed_docs(
        sn_id=name_id,
        claim_token="token",
        score=score,
        model="test-reviewer",
        min_score=0.85,
        rotation_cap=3,
        skip_review_node=True,
        resolution_method="quorum_consensus",
        reviewer_chain_size=2,
    )
    with GraphClient() as graph:
        graph.query(
            """
            MATCH (sn:StandardName {id: $id})
            CREATE (sn)-[:HAS_REVIEW]->(review:StandardNameReview {
                id: $review_id, review_axis: 'docs',
                review_group_id: 'test-group', is_canonical: true,
                resolution_method: 'quorum_consensus', score: $score,
                reviewed_at: datetime()
            })
            RETURN review.id AS id
            """,
            id=name_id,
            review_id=f"{name_id}:docs:test-group:0",
            score=score,
        )
    return stage


def _raise(name_id: str, finding: str = "contradicts source") -> dict:
    return holds.raise_hold(
        name_id,
        "docs",
        "documentation_contradiction",
        finding,
        "Claim: generic quantity. Source: flux-surface-averaged potential.",
        "jev-contradiction",
    )


def _state(name_id: str) -> dict:
    with GraphClient() as graph:
        return graph.query(
            "MATCH (sn:StandardName {id: $id}) "
            "RETURN sn.docs_stage AS stage, sn.reviewer_score_docs AS score",
            id=name_id,
        )[0]


def test_held_document_keeps_judged_score_and_is_not_accepted(
    planted_name: str,
) -> None:
    _raise(planted_name)
    assert _review(planted_name) == "reviewed"
    assert _state(planted_name) == {"stage": "reviewed", "score": 0.875}
    assert not holds.docs_accepted(0.875, 0.85, planted_name)
    assert len(holds.open_holds(planted_name, "docs")) == 1


def test_clearing_hold_accepts_passing_document(planted_name: str) -> None:
    hold = _raise(planted_name)
    _review(planted_name)
    assert (
        holds.clear_holds_not_raised(planted_name, "docs", {hold["kind"]}, set()) == 1
    )
    assert _state(planted_name) == {"stage": "accepted", "score": 0.875}
    assert holds.docs_accepted(0.875, 0.85, planted_name)


def test_dismissed_finding_stays_dismissed_and_does_not_block(
    planted_name: str,
) -> None:
    hold = _raise(planted_name)
    _review(planted_name)
    assert holds.dismiss_hold(hold["id"], "triager", "Source wording is compatible")
    assert _state(planted_name)["stage"] == "accepted"
    again = _raise(planted_name, "  CONTRADICTS   SOURCE ")
    assert again["id"] == hold["id"]
    assert again["status"] == "dismissed"
    assert holds.open_holds(planted_name, "docs") == []


def test_different_finding_raises_new_hold(planted_name: str) -> None:
    first = _raise(planted_name)
    holds.dismiss_hold(first["id"], "triager", "False positive")
    second = _raise(planted_name, "A second source gives a different locus")
    assert second["id"] != first["id"]
    assert [hold["id"] for hold in holds.open_holds(planted_name, "docs")] == [
        second["id"]
    ]


def test_refine_claims_passing_score_with_open_hold(planted_name: str) -> None:
    _raise(planted_name)
    _review(planted_name)
    items = graph_ops.claim_refine_docs_batch(batch_size=1, min_score=0.85)
    assert [item["id"] for item in items] == [planted_name]
    assert _state(planted_name) == {"stage": "refining", "score": 0.875}


def test_refiner_input_contains_open_hold_finding(planted_name: str) -> None:
    _raise(planted_name, "The source specifies flux-surface averaging")
    _review(planted_name)
    item = graph_ops.claim_refine_docs_batch(batch_size=1)[0]
    comments = item["prior_docs_reviews"][-1]["reviewer_comments_per_dim"]
    assert "The source specifies flux-surface averaging" in comments.values()

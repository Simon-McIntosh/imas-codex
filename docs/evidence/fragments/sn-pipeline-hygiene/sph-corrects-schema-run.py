#!/usr/bin/env python
"""Evidence for declaring the CORRECTS relationship in the graph schema.

Receipt-correction records written on 2026-11-10 link to the retry receipts
they correct through a ``CORRECTS`` edge the schema did not declare. Declaring
it records existing graph state; the declaration creates no edge.

Phases, each leaving an artifact under the run directory:

``live``
    Read every live ``CORRECTS`` edge with both endpoint label sets and
    compare them against the endpoint pairs the generated models declare.

``negative-control``
    Start an empty Neo4j and plant a ``CORRECTS`` edge between the labels the
    declaration does name: the delivered test must pass. Then plant one between
    labels the declaration does not name (``StandardNameSource`` ->
    ``StandardName``) and the same test must fail. The clean arm keeps the red
    arm honest: it shows the check sees the declared pair rather than failing
    for a setup reason, and it shows the instrument reads something known
    present before any absence is reported.

Both arms run the repository's own test, so the assertion that fails is the
delivered one rather than a copy of it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261111T221906013871-sph-corrects-schema"
)
REPO = Path(__file__).resolve().parents[4]
TEST_ID = (
    "tests/graph/test_schema_compliance.py::TestLabelsAndRelationships"
    "::test_corrects_edges_join_declared_endpoints"
)
RELATIONSHIP_TYPE = "CORRECTS"

_DECLARED_PAIR_EDGE = (
    "CREATE (a:StandardNameSourceRetry {id: 'nc-declared-from'})"
    "-[:CORRECTS]->(b:StandardNameSourceRetry {id: 'nc-declared-to'})"
)
_UNDECLARED_PAIR_EDGE = (
    "CREATE (a:StandardNameSource {id: 'nc-undeclared-from'})"
    "-[:CORRECTS]->(b:StandardName {id: 'nc-undeclared-to'})"
)


def save(name: str, payload) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )


def declared_endpoint_pairs() -> list[tuple[str, str]]:
    from imas_codex.graph.schema_context_data import RELATIONSHIPS

    return sorted(
        (from_label, to_label)
        for from_label, rel_type, to_label, _cardinality in RELATIONSHIPS
        if rel_type == RELATIONSHIP_TYPE
    )


def phase_live() -> dict:
    from imas_codex.graph.client import GraphClient

    declared = declared_endpoint_pairs()
    client = GraphClient()
    try:
        rows = client.query(
            "MATCH (a)-[:CORRECTS]->(b) "
            "RETURN labels(a) AS from_labels, labels(b) AS to_labels, "
            "count(*) AS edges"
        )
    finally:
        client.close()

    pairs = [
        {
            "from_labels": row["from_labels"],
            "to_labels": row["to_labels"],
            "edges": row["edges"],
            "declared": any(
                from_label in row["from_labels"] and to_label in row["to_labels"]
                for from_label, to_label in declared
            ),
        }
        for row in rows
    ]
    payload = {
        "declared_endpoint_pairs": declared,
        "endpoint_pairs_in_graph": pairs,
        "total_edges": sum(row["edges"] for row in rows),
        "mislabelled_edges": sum(row["edges"] for row in pairs if not row["declared"]),
    }
    save("live", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _plant(uri: str, statement: str, label: str) -> list[dict]:
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=None) as driver:
        driver.execute_query(statement)
        records, _, _ = driver.execute_query(
            "MATCH (a)-[:CORRECTS]->(b) "
            "RETURN labels(a) AS from_labels, labels(b) AS to_labels"
        )
    edges = [dict(record) for record in records]
    print(f"{label}: {edges}")
    return edges


def _run_test(uri: str, log_path: Path) -> int:
    env = dict(os.environ)
    env.update(
        {
            "NEO4J_URI": uri,
            "NEO4J_PASSWORD": "",
            "IMAS_CODEX_TEST_NEO4J_URI": uri,
            "IMAS_CODEX_TEST_NEO4J_EPHEMERAL": "1",
            "PYTHONPATH": str(REPO),
        }
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-m",
        "graph",
        "--timeout=300",
        TEST_ID,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    log_path.write_text(
        f"# revision: {_revision()} tree: {REPO} endpoint: {uri} "
        f"command: {' '.join(command)}\n"
        f"# module under test: tests/graph/test_schema_compliance.py\n"
        + completed.stdout
        + completed.stderr
        + f"EXIT={completed.returncode}\n"
    )
    return completed.returncode


def _revision() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def phase_negative_control() -> dict:
    from imas_codex.graph import temp_neo4j
    from tests.standard_names.conftest import (
        _detach_store_removal,
        _start_empty_neo4j,
    )

    directory, process, _log, bolt_port = _start_empty_neo4j()
    uri = f"bolt://127.0.0.1:{bolt_port}"
    clean_log = RUN / "nc-clean.log"
    red_log = RUN / "negative-control.log"
    try:
        clean_edges = _plant(uri, _DECLARED_PAIR_EDGE, "declared-pair only")
        clean_exit = _run_test(uri, clean_log)
        red_edges = _plant(uri, _UNDECLARED_PAIR_EDGE, "undeclared pair planted")
        red_exit = _run_test(uri, red_log)
    finally:
        try:
            temp_neo4j.stop_temp_neo4j(process)
        finally:
            _detach_store_removal(directory)

    payload = {
        "test_id": TEST_ID,
        "clean_arm": {"exit_status": clean_exit, "edges": clean_edges},
        "red_arm": {"exit_status": red_exit, "edges": red_edges},
        "clean_log": str(clean_log),
        "red_log": str(red_log),
    }
    save("negative-control", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["live", "negative-control"])
    args = parser.parse_args()
    if args.phase == "live":
        phase_live()
        return 0
    phase_negative_control()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

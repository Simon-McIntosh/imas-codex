#!/usr/bin/env python3
"""Capture and diagnose terminal DD composition sources without graph writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient

STATUS_QUERY = """
MATCH (s:StandardNameSource {source_type: 'dd'})
RETURN s.status AS status,
       EXISTS { MATCH (s)-[:PRODUCED_NAME]->(:StandardName) } AS produced,
       count(*) AS count ORDER BY status, produced
"""
SOURCE_QUERY = """
MATCH (s:StandardNameSource {source_type: 'dd'})
WHERE s.status IN ['failed', 'vocab_gap']
OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(n:StandardName)
WITH s, collect(n { .id, .name_stage, .docs_stage, .unit, .description,
     .reviewer_score_name, .reviewer_score_docs, .superseded_by,
     .created_at, .last_run_id }) AS names
OPTIONAL MATCH (s)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(g:VocabGap)
RETURN s.id AS id, s.source_id AS path, s.status AS status,
       s.attempt_count AS attempts, s.claim_seq AS claim_seq,
       s.last_error AS last_error, s.failed_at AS failed_at,
       s.composed_at AS composed_at, s.batch_key AS batch_key,
       s.description AS description, s.unit AS unit,
       s.produced_sn_id AS produced_sn_id, s.run_id AS run_id,
       s.parked_disposition AS parked_disposition, names,
       collect(g { .id, .segment, .token, .category, .editorial_disposition,
         .editorial_target, .editorial_reason }) AS gaps
ORDER BY id LIMIT 1000
"""
KEY_QUERY = """
MATCH (s:StandardNameSource {source_type: 'dd'})
RETURN count(*) AS candidates, count(s.id) AS with_id,
       count(s.source_id) AS with_source_id, count(s.status) AS with_status,
       count(s.last_error) AS with_last_error, count(s.attempt_count) AS with_attempts
"""
RUN_QUERY = """
MATCH (r:SNRun) WHERE r.id STARTS WITH '58a799f1'
RETURN r.id AS id, r.started_at AS started_at, r.ended_at AS ended_at,
       r.stop_reason AS stop_reason, r.status AS status,
       r.cost_spent AS cost_spent, r.cost_limit AS cost_limit,
       r.cost_total AS cost_total, r.cost_is_exact AS cost_is_exact,
       r.names_composed AS names_composed, r.names_enriched AS names_enriched,
       r.names_reviewed AS names_reviewed LIMIT 2
"""


def revision(tree: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()


def capture(destination: Path, log_path: Path) -> None:
    import imas_codex.graph.client as client

    tree = Path(__file__).resolve().parents[4]
    snapshot = {
        "observed_at": datetime.now(UTC).isoformat(),
        "tree": str(tree),
        "revision": revision(tree),
        "measurement_module": client.__file__,
        "measurement_cwd": str(Path.cwd().resolve()),
        "client_revision": revision(Path(client.__file__).resolve().parents[2]),
        "log_path": str(log_path),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "queries": {},
    }
    print(json.dumps({k: v for k, v in snapshot.items() if k != "queries"}))
    with GraphClient() as gc:
        for name, query in (
            ("keys", KEY_QUERY),
            ("status_before", STATUS_QUERY),
            ("sources", SOURCE_QUERY),
            ("run", RUN_QUERY),
            ("status_after", STATUS_QUERY),
        ):
            assert query.lstrip().startswith("MATCH ")
            snapshot["queries"][name] = {"cypher": query, "rows": gc.query(query)}
            print(f"captured {name}: {len(snapshot['queries'][name]['rows'])} rows")
    destination.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
    keys = snapshot["queries"]["keys"]["rows"][0]
    assert keys["candidates"] == keys["with_id"] == keys["with_source_id"]
    assert keys["with_last_error"] > 0 and keys["with_attempts"] > 0
    before = snapshot["queries"]["status_before"]["rows"]
    assert before == snapshot["queries"]["status_after"]["rows"]
    sources = snapshot["queries"]["sources"]["rows"]
    counts = Counter((s["status"], bool(s["names"])) for s in sources)
    expected = {
        (r["status"], r["produced"]): r["count"]
        for r in before
        if r["status"] in ("failed", "vocab_gap")
    }
    assert dict(counts) == expected
    assert len(sources) < 1000 and len(snapshot["queries"]["run"]["rows"]) == 1
    print("COHORT " + json.dumps({str(k): v for k, v in counts.items()}))
    print("ERRORS " + json.dumps(Counter(s["last_error"] for s in sources)))
    print("RUN " + json.dumps(snapshot["queries"]["run"]["rows"], default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    args = parser.parse_args()
    capture(args.snapshot, args.run_log)


if __name__ == "__main__":
    main()

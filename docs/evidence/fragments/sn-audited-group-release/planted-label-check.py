"""Freeze source-only judgments before comparing planted labels and readers.

Collection projects identifiers only from the audit; no labels, citations or
reader verdicts enter the source-only packet. Graph access is read-only and
restricted to these identifiers, their producers and enclosing DD structures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "planted-label-check.json"
PAGE = ROOT / "planted-label-check.html"


def audit_module():
    spec = importlib.util.spec_from_file_location(
        "blind_audit", ROOT / "blind-audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def revision():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def write_data(data):
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def collect():
    if DATA.exists():
        raise ValueError("Refusing to overwrite the source-only record")
    audit = audit_module()
    # Only this projection of the label-bearing file is exposed to the judge.
    ids = sorted(
        row["id"]
        for row in json.loads((ROOT / "blind-audit.json").read_text())[
            "planted_detail"
        ]["rows"]
    )
    assert len(ids) == len(set(ids)) == 40
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.source_paths import split_source_paths
    from imas_codex.standard_names.source_reads import source_paths_projection
    from imas_codex.standard_names.worker_tasks.worker_common import (
        _ANCESTOR_CONTEXT_QUERY,
        ancestor_statements,
    )

    gc = GraphClient()
    try:
        rows = list(
            gc.query(
                f"""
                UNWIND $ids AS wanted
                MATCH (sn:StandardName {{id: wanted}})
                RETURN sn.id AS id, coalesce(sn.description,'') AS description,
                       coalesce(sn.documentation,'') AS documentation,
                       coalesce(sn.unit,'') AS unit,
                       coalesce(sn.physics_domain,'') AS physics_domain,
                       {source_paths_projection()} AS source_paths
                ORDER BY id
                """,
                ids=ids,
            )
        )
        assert {r["id"] for r in rows} == set(ids), "Missing planted identifiers"
        paths = sorted(
            {
                path
                for row in rows
                for path in split_source_paths(row["source_paths"]).get("dd", [])
            }
        )
        docs = {
            r["path"]: r
            for r in gc.query(
                """
                UNWIND $paths AS pid MATCH (n:IMASNode {id: pid})
                RETURN n.id AS path, coalesce(n.description,'') AS description,
                       coalesce(n.documentation,'') AS documentation,
                       coalesce(n.unit,'') AS unit
                """,
                paths=paths,
            )
        }
        assert set(docs) == set(paths), "Missing DD source text"
        duplicates = {
            r["path"]: sorted(r["names"])
            for r in gc.query(
                f"""
                UNWIND $paths AS pid
                MATCH (src:StandardNameSource {{id: 'dd:' + pid}})
                      -[:PRODUCED_NAME]->(sn:StandardName)
                WHERE coalesce(src.status,'') <> 'stale' AND {audit.LIVE}
                RETURN pid AS path, collect(DISTINCT sn.id) AS names
                """,
                paths=paths,
            )
        }
        for row in rows:
            row["dd_paths"] = split_source_paths(row["source_paths"]).get("dd", [])
            row["dd_docs"] = [docs[p] for p in row["dd_paths"]]
            row["ancestors"] = (
                ancestor_statements(
                    list(gc.query(_ANCESTOR_CONTEXT_QUERY, path=row["dd_paths"][0]))
                )
                if row["dd_paths"]
                else []
            )
            audit.attach_duplicate_candidates([row], duplicates)
            row["reader_visible_text"] = audit.render_user_prompt(row)
            audit.assert_blind(row["reader_visible_text"], row)
        receipt = {
            "requested_names": len(ids),
            "returned_names": len(rows),
            "requested_dd_paths": len(paths),
            "returned_dd_paths": len(docs),
            "names_with_dd_text": sum(bool(r["dd_docs"]) for r in rows),
            "names_with_ancestor_text": sum(bool(r["ancestors"]) for r in rows),
            "names_with_duplicate_candidates": sum(
                bool(r["other_live_names"]) for r in rows
            ),
            "producer_source_groups": len(duplicates),
            "self_producer_memberships": sum(
                row["id"] in duplicates.get(path, [])
                for row in rows
                for path in row["dd_paths"]
            ),
            "graph_write_statements": 0,
            "model_calls": 0,
        }
    finally:
        gc.close()
    rows = [
        {key: row[key] for key in ("id", "source_paths", "reader_visible_text")}
        for row in rows
    ]
    write_data(
        {
            "phase": "source-only",
            "collected_at": datetime.now(UTC).isoformat(),
            "collection_revision": revision(),
            "collection_receipt": receipt,
            "class_list": list(audit.CLASS_KEYS),
            "blindness": {
                "exposed_before_judgment": "Sorted identifiers and source-only packets",
                "withheld": ["known", "citation", "A", "B", "C"],
                "historical_packet_caveat": (
                    "The audit stores verdicts but no full per-name prompt snapshots; "
                    "these are current graph fields rendered by the original prompt "
                    "function, not proof the text has remained unchanged since reading."
                ),
                "events": ["Read live plan", "Collected source-only packets"],
            },
            "packets": rows,
            "packet_digest": digest(rows),
            "judgments": [],
        }
    )
    print(json.dumps(receipt, sort_keys=True))


def verify_sources():
    """Require the duplicate-candidate join to recover each known producer."""
    from imas_codex.graph.client import GraphClient

    data = json.loads(DATA.read_text())
    packets = data["packets"]
    ids = sorted({p for r in packets for p in r["source_paths"] if p.startswith("dd:")})
    gc = GraphClient()
    try:
        rows = list(
            gc.query(
                """
                UNWIND $ids AS sid
                MATCH (src:StandardNameSource {id: sid})-[:PRODUCED_NAME]->(n:StandardName)
                WHERE coalesce(src.status,'') <> 'stale'
                  AND coalesce(n.name_stage,'') <> 'superseded'
                  AND coalesce(n.status,'') <> 'superseded'
                RETURN sid AS source, collect(DISTINCT n.id) AS names
                """,
                ids=ids,
            )
        )
    finally:
        gc.close()
    by_source = {r["source"]: r["names"] for r in rows}
    self_matches = sum(
        p["id"] in by_source.get(src, []) for p in packets for src in p["source_paths"]
    )
    expected = sum(len(p["source_paths"]) for p in packets)
    assert len(by_source) == len(ids), "Producer join missed known source nodes"
    assert self_matches == expected, "Producer join missed known names"
    candidates = {
        p["id"]: sorted(
            {
                n
                for src in p["source_paths"]
                for n in by_source.get(src, [])
                if n != p["id"]
            }
        )[:10]
        for p in packets
    }
    assert not any(candidates.values()), (
        "Duplicate candidates differ from frozen packet"
    )
    receipt = {
        "producer_source_groups": len(by_source),
        "self_producer_memberships": self_matches,
        "expected_self_memberships": expected,
        "other_live_name_candidates": sum(len(v) for v in candidates.values()),
        "positive_control": "Every bound DD producer recovered its own planted name",
    }
    data["collection_receipt"]["duplicate_join_control"] = receipt
    write_data(data)
    print(json.dumps(receipt, sort_keys=True))


def check_blind(data):
    packets = {p["id"]: p for p in data["packets"]}
    judgments = data["judgments"]
    assert len(packets) == len(judgments) == 40, "Require forty unique judgments"
    assert {j["id"] for j in judgments} == set(packets), "Judgment coverage differs"
    assert digest(data["packets"]) == data["packet_digest"], (
        "Source packet digest changed"
    )
    assert digest(judgments) == data["blindness"]["judgment_digest"], (
        "Judgments changed"
    )
    for judgment in judgments:
        text = packets[judgment["id"]]["reader_visible_text"]
        assert judgment["evidence_quote"] in text, (
            "Quote absent from packet: " + judgment["id"]
        )
        assert judgment["evidence_path"] in text, (
            "Source binding absent: " + judgment["id"]
        )
        assert judgment["verdict"] in ("defective", "sound")
        assert bool(judgment["defect_classes"]) == (judgment["verdict"] == "defective")
        assert set(judgment["defect_classes"]) <= set(data["class_list"])
        assert isinstance(judgment["decidable_from_text"], bool)
        assert not ({"known", "citation", "A", "B", "C"} & judgment.keys())
    control = data["collection_receipt"]["duplicate_join_control"]
    assert (
        control["self_producer_memberships"] == control["expected_self_memberships"] > 0
    )
    print(
        "PASS: 40 source-only judgments; quotes, bindings, digests and producer control"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--check-blind", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.verify_sources:
        verify_sources()
    elif args.check_blind:
        check_blind(json.loads(DATA.read_text()))
    else:
        parser.error("Choose --collect")


if __name__ == "__main__":
    main()

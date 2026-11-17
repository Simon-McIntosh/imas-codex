"""Freeze complete source packets before adjudication and reader comparison.

The existing audit owns packet rendering, blindness, classes and intervals.
Packets retain unabridged text and the ancestry of every source. Git commits
are the independent seals: changing a verdict and its digest cannot reseal it.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import subprocess
import zlib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "equilibrium-adjudication.json"
PAGE = ROOT / "equilibrium-adjudication.html"
RUN = Path(
    os.environ.get(
        "EQUILIBRIUM_ADJUDICATION_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261117T155147248477-sagr-equilibrium-adjudication",
    )
)
RELATIVE_DATA = (
    "docs/evidence/fragments/sn-audited-group-release/equilibrium-adjudication.json"
)
SEED = 2026111707


def sibling(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), ROOT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def load(path=DATA):
    return json.loads(Path(path).read_text())


def save(value, path=DATA):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert path != DATA or len(text.encode()) <= 300_000, (
        "Repository data file exceeds ceiling"
    )
    Path(path).write_text(text)


def digest(value):
    return sibling("planted-label-check").digest(value)


def packet_data(data):
    if "packet_payload_zlib_base85" in data:
        return data | json.loads(
            zlib.decompress(base64.b85decode(data["packet_payload_zlib_base85"]))
        )
    return data


def packets(data):
    return sibling("reader-calibration").source_packets(packet_data(data))


def compact(data):
    payload = {
        key: data.pop(key)
        for key in ("text_pool", "packets", "dd_versions", "packet_versions")
    }
    data["packet_payload_zlib_base85"] = base64.b85encode(
        zlib.compress(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
            level=9,
        )
    ).decode()


def attach_families(rendered):
    """Expose only structural family fields, never the census's audit counts."""
    families = [
        {key: f[key] for key in ("parent_path", "family_type", "member_groups")}
        for f in load(ROOT / "domain-groups.json")["families"]["spanning"]
        if "equilibrium" in f["member_groups"]
    ]
    assert len(families) == 72
    for packet in rendered:
        related = [
            f for f in families if packet["id"] in f["member_groups"]["equilibrium"]
        ]
        if related:
            packet["reader_visible_text"] += (
                "\nStructural family context (no earlier audit verdicts):\n"
                + json.dumps(related, ensure_ascii=False, sort_keys=True)
            )
    return {
        "touching_equilibrium": len(families),
        "structural_projection_digest": digest(families),
    }


def collect():
    assert not DATA.exists(), "Refuse to overwrite packet freeze"
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.source_reads import source_paths_projection

    audit = sibling("blind-audit")
    gc = GraphClient()
    try:
        control = list(
            gc.query(f"""
            MATCH (sn:StandardName {{physics_domain: 'equilibrium'}})
            WHERE {audit.LIVE}
            RETURN count(sn) AS members, count(sn.id) AS identities,
                   count(sn.description) AS descriptions
        """)
        )[0]
        assert control["members"] == control["identities"] > 0
        assert control["descriptions"] > 0
        rows = list(
            gc.query(f"""
            MATCH (sn:StandardName {{physics_domain: 'equilibrium'}})
            WHERE {audit.LIVE}
            RETURN sn.id AS id, coalesce(sn.description,'') AS description,
                   coalesce(sn.documentation,'') AS documentation,
                   coalesce(sn.unit,'') AS unit,
                   sn.physics_domain AS physics_domain,
                   {source_paths_projection()} AS source_paths
            ORDER BY id
        """)
        )
        assert len(rows) == control["members"]
        for row in rows:
            row["source_paths"] = sorted(set(row["source_paths"]))
        paths = sorted({p for r in rows for p in r["source_paths"]})
        source_records = list(
            gc.query(
                """
            UNWIND $ids AS sid MATCH (src:StandardNameSource {id: sid})
            RETURN src.id AS id, src.source_type AS source_type,
                   src.description AS description, src.dd_version AS dd_version,
                   src.dd_documentation AS dd_documentation,
                   src.dd_parent_path AS dd_parent_path,
                   src.dd_parent_documentation AS dd_parent_documentation,
                   src.dd_unit AS dd_unit
        """,
                ids=paths,
            )
        )
        assert {r["id"] for r in source_records} == set(paths)
        source_records = {r["id"]: r for r in source_records}
        dd_paths = [p for p in paths if p.startswith("dd:")]
        # One invocation of the established context builder per distinct source
        # preserves every enclosing chain instead of only one per name.
        contexts = [{"id": p, "source_paths": [p]} for p in dd_paths]
        audit.load_dd_context(gc, contexts)
        context = {r["id"]: r for r in contexts}
        assert all(len(r["dd_docs"]) == 1 for r in contexts)
        versions = [
            {"path": p, "dd_version": source_records[p]["dd_version"]} for p in dd_paths
        ]
        assert len(versions) == len(dd_paths)
        by_path = {
            r["path"]: r["names"]
            for r in gc.query(
                f"""
            UNWIND $paths AS pid
            MATCH (src:StandardNameSource {{id: 'dd:' + pid}})-[:PRODUCED_NAME]->(sn:StandardName)
            WHERE coalesce(src.status,'') <> 'stale' AND {audit.LIVE}
            RETURN pid AS path, collect(DISTINCT sn.id) AS names
        """,
                paths=[p[3:] for p in dd_paths],
            )
        }
        memberships = sum(
            r["id"] in by_path.get(p[3:], [])
            for r in rows
            for p in r["source_paths"]
            if p.startswith("dd:")
        )
        expected = sum(p.startswith("dd:") for r in rows for p in r["source_paths"])
        assert memberships == expected > 0, "Source join missed known producers"
        # Rendering remains the audit's function; full packets must not clip.
        audit.clip = lambda text, limit: (
            " ".join((text or "").split()) or "(none recorded)"
        )
        rendered = []
        for row in rows:
            row["dd_paths"] = [
                p[3:] for p in row["source_paths"] if p.startswith("dd:")
            ]
            row["dd_docs"] = [
                context[p]["dd_docs"][0] for p in row["source_paths"] if p in context
            ]
            row["ancestors"] = []
            seen = set()
            for p in row["source_paths"]:
                if p not in context:
                    continue
                for ancestor in context[p]["ancestors"]:
                    if ancestor["path"] not in seen:
                        row["ancestors"].append(ancestor)
                        seen.add(ancestor["path"])
            audit.attach_duplicate_candidates([row], by_path)
            text = audit.render_user_prompt(row).replace(
                "Enclosing structures of the first source, nearest first:",
                "Enclosing structures of every source (deduplicated), nearest first per chain:",
            )
            text += (
                "\nProducer source records (raw text and extraction snapshot):\n"
                + json.dumps(
                    [source_records[p] for p in row["source_paths"]],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            audit.assert_blind(text, row)
            rendered.append(
                {
                    "id": row["id"],
                    "source_paths": row["source_paths"],
                    "reader_visible_text": text,
                }
            )
    finally:
        gc.close()
    family_receipt = attach_families(rendered)
    data = {
        "phase": "packets-frozen",
        "collection_revision": git("rev-parse", "HEAD"),
        "collected_at": datetime.now(UTC).isoformat(),
        "members": [r["id"] for r in rows],
        "member_digest": digest([r["id"] for r in rows]),
        "packets": rendered,
        "packet_digest": digest(rendered),
        "packet_versions": {p["id"]: digest(p) for p in rendered},
        "dd_versions": versions,
        "class_list": list(audit.DEFECT_CLASSES),
        "instructions": audit.system_prompt(),
        "blindness": {
            "prior_readings_opened": False,
            "model_calls": 0,
            "graph_writes": 0,
            "exposure": "Live plan aggregate context, helper source code, frozen source packets; no earlier individual verdicts, review scores, calibration labels or audit rows",
        },
        "collection_receipt": control
        | {
            "families": family_receipt,
            "source_paths": len(paths),
            "non_dd_sources": [p for p in paths if not p.startswith("dd:")],
            "producer_memberships": memberships,
            "expected_memberships": expected,
            "names_without_source": sum(not r["source_paths"] for r in rows),
            "source_chains_with_text": sum(bool(r["ancestors"]) for r in contexts),
        },
        "judgments": [],
    }
    sibling("reader-calibration").compact_packets(data)
    save(data, RUN / "source-packets-uncompressed.json")
    compact(data)
    assert packets(data) == rendered
    save(data)
    print(json.dumps(data["collection_receipt"]))


def committed_snapshot(phase):
    for sha in git("rev-list", "--reverse", "HEAD", "--", RELATIVE_DATA).splitlines():
        record = json.loads(git("show", f"{sha}:{RELATIVE_DATA}"))
        if record["phase"] == phase:
            return sha, record
    raise AssertionError("Missing independent committed seal: " + phase)


def validate(data):
    audit = sibling("blind-audit")
    visible = packets(data)
    assert data["members"] == sorted(set(data["members"]))
    assert [p["id"] for p in visible] == data["members"]
    assert digest(data["members"]) == data["member_digest"]
    assert digest(visible) == data["packet_digest"], "Frozen packets changed"
    assert {p["id"]: digest(p) for p in visible} == packet_data(data)["packet_versions"]
    assert not data["blindness"]["prior_readings_opened"]
    assert data["blindness"]["model_calls"] == data["blindness"]["graph_writes"] == 0
    assert data["class_list"] == [list(r) for r in audit.DEFECT_CLASSES]
    for packet in visible:
        audit.assert_blind(packet["reader_visible_text"], packet)
    judgments = data["judgments"]
    if judgments:
        assert {j["id"] for j in judgments} == set(data["members"])
        assert len(judgments) == len(visible)
        assert digest(judgments) == data["judgment_digest"]
        by_id = {p["id"]: p for p in visible}
        for judgment in judgments:
            text = by_id[judgment["id"]]["reader_visible_text"]
            assert judgment["verdict"] in ("sound", "defective", "undecidable")
            assert bool(judgment["defect_classes"]) == (
                judgment["verdict"] == "defective"
            )
            assert set(judgment["defect_classes"]) <= set(audit.CLASS_KEYS)
            assert judgment["evidence_quote"] in text, judgment["id"]
            assert judgment["evidence_quote"] in judgment["reason"], judgment["id"]
            assert judgment["evidence_path"] in text, judgment["id"]
    return visible


def check(path=DATA):
    data = load(path)
    validate(data)
    packet_sha, frozen_packets = committed_snapshot("packets-frozen")
    for key, value in frozen_packets.items():
        if key not in ("phase", "judgments"):
            assert data[key] == value, "Packet seal changed: " + key
    if data["judgments"]:
        verdict_sha, frozen_verdicts = committed_snapshot("verdicts-frozen")
        assert data["judgments"] == frozen_verdicts["judgments"], (
            "Verdict changed after blind commit"
        )
        assert data["judgment_digest"] == frozen_verdicts["judgment_digest"]
        assert git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", verdict_sha
        ).splitlines() == [RELATIVE_DATA], "Verdict commit touched reader data"
        assert git("merge-base", packet_sha, verdict_sha) == packet_sha
    print(
        f"PASS: {len(data['members'])} immutable source packets; {len(data['judgments'])} blind verdicts; independent git seals verified"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.check:
        check(args.data)


if __name__ == "__main__":
    main()

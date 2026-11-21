"""Freeze complete source packets before adjudication and reader comparison.

The existing audit owns packet rendering, blindness, classes and intervals.
Packets retain unabridged text and the ancestry of every source. Git commits
are the independent seals: changing a verdict and its digest cannot reseal it.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import zlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "equilibrium-adjudication.json"
PAGE = ROOT / "equilibrium-adjudication.html"
PENDING_PACKETS = ROOT / "equilibrium-pending-packets.json"
PENDING_MEMBERS = (
    "radial_coordinate_of_conductor_cross_section",
    "vertical_outline",
)
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
    return subprocess.check_output(
        ["git", "-C", str(ROOT.parents[3]), *args], text=True
    ).strip()


def load(path=DATA):
    return decode_record(json.loads(Path(path).read_text()))


def decode_record(value):
    value = dict(value)
    for key in ("judgments", "report", "coverage"):
        payload = value.pop(key + "_payload_zlib_base85", None)
        if payload is not None:
            value[key] = json.loads(zlib.decompress(base64.b85decode(payload)))
    return value


def save(value, path=DATA):
    value = dict(value)
    for key in ("judgments", "report", "coverage"):
        if value.get(key):
            value[key + "_payload_zlib_base85"] = base64.b85encode(
                zlib.compress(
                    json.dumps(value.pop(key), ensure_ascii=False).encode(), level=9
                )
            ).decode()
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
    from iter_codex.graph.client import GraphClient
    from iter_codex.standard_names.source_reads import source_paths_projection

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


def committed_snapshot(phase, required=True):
    for sha in git("rev-list", "--reverse", "HEAD", "--", RELATIVE_DATA).splitlines():
        record = decode_record(json.loads(git("show", f"{sha}:{RELATIVE_DATA}")))
        if record["phase"] == phase:
            return sha, record
    if required:
        raise AssertionError("Missing independent committed seal: " + phase)
    return None


def committed_field(key):
    for sha in git("rev-list", "--reverse", "HEAD", "--", RELATIVE_DATA).splitlines():
        record = decode_record(json.loads(git("show", f"{sha}:{RELATIVE_DATA}")))
        if key in record:
            return sha, record
    raise AssertionError("Missing independently committed field: " + key)


def collect_pending_packets():
    """Supplement the frozen builder output with every enclosing statement."""
    from iter_codex.graph.client import GraphClient
    from iter_codex.standard_names.worker_tasks.worker_common import (
        _ANCESTOR_CONTEXT_QUERY,
        ancestor_statements,
    )

    data = load()
    assert "coverage" not in data and not PENDING_PACKETS.exists()
    visible = packets(data)
    sources = sorted(
        {s[3:] for p in visible for s in p["source_paths"] if s.startswith("dd:")}
    )
    depth = max(s.count("/") for s in sources)
    query = (
        _ANCESTOR_CONTEXT_QUERY.replace(
            "MATCH path =", "UNWIND $paths AS source_path\nMATCH path ="
        )
        .replace("id: $path", "id: source_path")
        .replace("*1..8", f"*1..{depth}")
        .replace(
            "RETURN length(path)", "RETURN source_path AS source_path, length(path)"
        )
        .replace("ORDER BY depth ASC", "ORDER BY source_path, depth ASC")
    )
    gc = GraphClient()
    try:
        rows = list(gc.query(query, paths=sources))
    finally:
        gc.close()
    save({"rows": rows}, RUN / "pending-ancestor-rows.json")
    assert any(r["path"] == "equilibrium/time_slice/boundary" for r in rows), (
        "Known-present context control missing"
    )
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_path"]].append(row)
    fields = []
    indices = {}
    source_fields = {}
    for source in sources:
        required = []
        for row in by_source[source]:
            for field in ("documentation", "description"):
                text = " ".join((row.get(field) or "").split())
                if not text:
                    continue
                item = {"ancestor": row["path"], "field": field, "text": text}
                identity = digest(item)
                if identity not in indices:
                    indices[identity] = len(fields)
                    fields.append(item)
                required.append(indices[identity])
        source_fields[source] = required
    coverage = {"ancestor_fields": fields, "source_fields": source_fields}
    missing = coverage_missing(data, coverage)
    original_missing = [
        {key: item[key] for key in ("member", "source", "ancestor", "field", "text")}
        for item in data["source_completeness"]["missing_fields"]
    ]
    assert sorted(missing, key=digest) == sorted(original_missing, key=digest), (
        "Ancestor text changed beyond recorded omissions"
    )
    assert {m["member"] for m in missing} == set(PENDING_MEMBERS)
    corrected = []
    for packet in visible:
        if packet["id"] not in PENDING_MEMBERS:
            continue
        ancestors = []
        seen = set()
        for source in packet["source_paths"]:
            if not source.startswith("dd:"):
                continue
            for row in ancestor_statements(by_source[source[3:]], limit=None):
                if row["path"] not in seen:
                    ancestors.append(row)
                    seen.add(row["path"])
        text = packet["reader_visible_text"]
        start = text.index("\nEnclosing structures of every source")
        endings = [
            text.find(marker, start)
            for marker in (
                "\nOther live standard names",
                "\nReturn every fixed defect class",
            )
        ]
        end = min(i for i in endings if i >= 0)
        text = (
            text[:start]
            + "\nEnclosing structures of every source, nearest first per chain:\n"
            + "\n".join(f"  {a['path']}: {a['text']}" for a in ancestors)
            + text[end:]
        )
        text = text.replace(
            "Return every fixed defect class that applies, or an empty list, with the wording that decides it.\n",
            "",
        )
        text = text.replace(
            "Structural family context (no earlier audit verdicts):",
            "Structural family context:",
        )
        item = {
            "id": packet["id"],
            "source_paths": packet["source_paths"],
            "packet_text": text,
            "dd_versions": [
                v
                for v in packet_data(data)["dd_versions"]
                if v["path"] in packet["source_paths"]
            ],
        }
        for source in packet["source_paths"]:
            for index in source_fields[source.removeprefix("dd:")]:
                assert fields[index]["text"] in text
        item["packet_digest"] = digest(item)
        corrected.append(item)
    payload = {"packets": corrected}
    assert len(corrected) == 2
    assert not any(
        word in json.dumps(payload).lower()
        for word in ("verdict", "reader", "defect_classes", "candidate_cause")
    )
    save(payload, PENDING_PACKETS)
    data["coverage"] = coverage
    data["coverage_receipt"] = {
        "collected_at": datetime.now(UTC).isoformat(),
        "source_count": len(sources),
        "ancestor_rows": len(rows),
        "positive_control": "equilibrium/time_slice/boundary",
        "coverage_digest": digest(coverage),
        "pending_packets_digest": digest(payload),
    }
    save(data)
    print(
        f"COVERAGE: {len(sources)} source chains; {len(rows)} ancestor rows; {len(missing)} known omitted fields; two complete neutral packets; zero graph writes"
    )


def coverage_missing(data, coverage):
    missing = []
    for packet in packets(data):
        for source in packet["source_paths"]:
            if not source.startswith("dd:"):
                continue
            assert source[3:] in coverage["source_fields"], (
                "Source lacks completeness coverage"
            )
            for index in coverage["source_fields"][source[3:]]:
                field = coverage["ancestor_fields"][index]
                if field["text"] not in packet["reader_visible_text"]:
                    missing.append({"member": packet["id"], "source": source, **field})
    return missing


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
        pending = data.get("pending", [])
        if pending:
            assert [p["id"] for p in pending] == list(PENDING_MEMBERS), (
                "Pending exceptions changed"
            )
            assert all(p["status"] == "pending_fresh_adjudication" for p in pending)
        expected = set(data["members"]) - {p["id"] for p in pending}
        assert {j["id"] for j in judgments} == expected, (
            "Eligible member lacks a verdict"
        )
        assert len(judgments) == len(expected)
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


def exclude_pending():
    data = load()
    assert "pending" not in data
    _, sealed = committed_snapshot("verdicts-frozen")
    assert data["judgments"] == sealed["judgments"]
    handoff = load(PENDING_PACKETS)
    missing = data["source_completeness"]["missing_fields"]
    assert {m["member"] for m in missing} == set(PENDING_MEMBERS)
    data["pending"] = [
        {
            "id": packet["id"],
            "status": "pending_fresh_adjudication",
            "omitted_fields": [m for m in missing if m["member"] == packet["id"]],
            "corrected_packet_digest": packet["packet_digest"],
        }
        for packet in handoff["packets"]
    ]
    data["judgments"] = [
        j for j in sealed["judgments"] if j["id"] not in PENDING_MEMBERS
    ]
    data["judgment_digest"] = digest(data["judgments"])
    data.pop("report", None)
    validate(data)
    check_coverage(data)
    save(data)
    print(
        "EXCLUSIONS: 208 original verdicts unchanged; exactly two pending fresh adjudication; six omitted fields retained; no new verdicts"
    )


def check_coverage(data):
    assert [p["id"] for p in data.get("pending", [])] == list(PENDING_MEMBERS), (
        "Exactly the two authorized pending members are required"
    )
    coverage_sha, sealed = committed_field("coverage")
    assert data["coverage"] == sealed["coverage"], (
        "Committed ancestor requirements changed"
    )
    assert data["coverage_receipt"] == sealed["coverage_receipt"]
    assert digest(data["coverage"]) == data["coverage_receipt"]["coverage_digest"]
    assert data["source_completeness"] == sealed["source_completeness"]
    source_ids = {
        s[3:] for p in packets(data) for s in p["source_paths"] if s.startswith("dd:")
    }
    assert set(data["coverage"]["source_fields"]) == source_ids
    missing = coverage_missing(data, data["coverage"])
    expected_missing = [
        {key: item[key] for key in ("member", "source", "ancestor", "field", "text")}
        for item in sealed["source_completeness"]["missing_fields"]
    ]
    assert sorted(missing, key=digest) == sorted(expected_missing, key=digest), (
        "A member outside the pending exceptions lacks a full packet"
    )
    assert len(missing) == 6 and {m["member"] for m in missing} == set(PENDING_MEMBERS)
    payload = load(PENDING_PACKETS)
    pending_path = str(PENDING_PACKETS.relative_to(ROOT.parents[3]))
    assert payload == json.loads(git("show", f"{coverage_sha}:{pending_path}")), (
        "Corrected neutral packets changed after commit"
    )
    assert set(payload) == {"packets"}
    assert [p["id"] for p in payload["packets"]] == list(PENDING_MEMBERS)
    assert digest(payload) == data["coverage_receipt"]["pending_packets_digest"]
    assert not any(
        word in json.dumps(payload).lower()
        for word in ("verdict", "reader", "defect_classes", "candidate_cause")
    )
    for pending, packet in zip(data["pending"], payload["packets"], strict=True):
        assert set(pending) == {
            "id",
            "status",
            "omitted_fields",
            "corrected_packet_digest",
        }
        assert pending["status"] == "pending_fresh_adjudication"
        assert pending["omitted_fields"] == [
            m
            for m in sealed["source_completeness"]["missing_fields"]
            if m["member"] == pending["id"]
        ]
        assert pending["corrected_packet_digest"] == packet["packet_digest"]
        assert packet["packet_digest"] == digest(
            {k: v for k, v in packet.items() if k != "packet_digest"}
        )
        sibling("blind-audit").assert_blind(packet["packet_text"], packet)
        for source in packet["source_paths"]:
            for index in data["coverage"]["source_fields"][source[3:]]:
                assert (
                    data["coverage"]["ancestor_fields"][index]["text"]
                    in packet["packet_text"]
                ), "Corrected packet lacks ancestor text"
    assert len(data["judgments"]) == 208


def seal_verdicts():
    data = load()
    assert data["phase"] == "packets-frozen" and not data["judgments"]
    check(include_report=False)
    judgments = [
        json.loads(line)
        for line in (RUN / "blind-judgments.jsonl").read_text().splitlines()
    ]
    data.update(
        phase="verdicts-frozen",
        judgments=judgments,
        judgment_digest=digest(judgments),
        verdict_completed_at=datetime.now(UTC).isoformat(),
        adjudication_method="Direct packet-by-packet adjudication; no reader or model calls; earlier individual readings unopened until verdict commit",
    )
    validate(data)
    save(data)
    print(f"SEALED: {len(judgments)} blind verdicts; commit only {RELATIVE_DATA}")


def check(path=DATA, include_report=True):
    data = load(path)
    validate(data)
    packet_sha, frozen_packets = committed_snapshot("packets-frozen")
    for key, value in frozen_packets.items():
        if key not in ("phase", "judgments"):
            assert data[key] == value, "Packet seal changed: " + key
    verdict_seal = committed_snapshot("verdicts-frozen", required=False)
    if verdict_seal:
        verdict_sha, frozen_verdicts = verdict_seal
        pending = {p["id"] for p in data.get("pending", [])}
        expected = [j for j in frozen_verdicts["judgments"] if j["id"] not in pending]
        assert data["judgments"] == expected, "Verdict changed after blind commit"
        assert data["phase"] == "verdicts-frozen", (
            "Adjudication phase changed after seal"
        )
        assert data["judgment_digest"] == digest(expected)
        assert git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", verdict_sha
        ).splitlines() == [RELATIVE_DATA], "Verdict commit touched reader data"
        assert git("merge-base", packet_sha, verdict_sha) == packet_sha
    else:
        assert not data["judgments"], "Verdicts require an independent committed seal"
    if include_report and data["judgments"]:
        check_coverage(data)
        assert "report" in data, "Completed adjudication requires its report"
    if include_report and "report" in data:
        report = data["report"]
        assert report == summarize(data, report["comparison_inputs"])
        assert PAGE.read_text() == render_page(data), "Rendered evidence changed"
        assert len(report["lead_review"]["members"]) == 20
        assert not set(report["lead_review"]["members"]) & set(PENDING_MEMBERS)
        assert PAGE.read_text().count('class="review-card"') == 20
        assert PAGE.read_text().count("<textarea ") == 20
        assert "<details open" not in PAGE.read_text()
        print(
            "PASS: 208 complete packets and unchanged sealed verdicts; exactly two pending with six omitted fields; all result counts exclude pending members; seeded 20-member page reproduced; lead verdict fields empty"
        )
    print(
        f"PASS: {len(data['members'])} immutable source packets; {len(data['judgments'])} blind verdicts; independent git seals verified"
    )


def collect_comparison():
    """Read historical judgments only after the independent verdict seal exists."""
    check(include_report=False)
    verdict_sha, frozen = committed_snapshot("verdicts-frozen")
    from iter_codex.graph.client import GraphClient

    audit = sibling("blind-audit")
    third_path = ROOT / "frontier-third-reader.json"
    third = load(third_path)
    members = set(frozen["members"])
    earlier = {}
    for kind in ("flagged", "control", "planted"):
        for row in third[kind]["rows"]:
            if row["name"] not in members:
                continue
            classes = row["C"]
            if isinstance(classes, dict):
                if not row.get("read"):
                    continue
                classes = classes["classes"]
            earlier.setdefault(row["name"], {})["C"] = sorted(classes)
    gc = GraphClient()
    try:
        rows = list(
            gc.query(
                """
                UNWIND $ids AS name
                MATCH (sn:StandardName {id: name})-[:HAS_REVIEW]->(r:StandardNameReview)
                WHERE r.review_group_id = $group AND r.resolution_role = 'audit'
                RETURN sn.id AS name, r.standard_name_id AS back_reference,
                       r.comments AS comments
                """,
                ids=sorted(members),
                group=audit.PRIOR_GROUP,
            )
        )
    finally:
        gc.close()
    save({"rows": rows}, RUN / "historical-graph-rows.json")
    assert rows, "Historical audit query missed known-present comparison names"
    ignored_seats = []
    for row in rows:
        assert row["name"] == row["back_reference"]
        payload = json.loads(row["comments"])
        seat = payload.get("reader_seat", "")
        if seat not in ("A", "B"):
            ignored_seats.append(seat)
            continue
        classes = sorted(payload["defect_classes"])
        assert set(classes) <= set(audit.CLASS_KEYS)
        target = earlier.setdefault(row["name"], {})
        assert seat not in target or target[seat] == classes
        target[seat] = classes
    controls = 0
    for kind in ("flagged", "control"):
        for row in third[kind]["rows"]:
            if row["name"] in members:
                for seat in ("A", "B"):
                    assert earlier[row["name"]][seat] == sorted(row[seat])
                    controls += 1
    assert controls > 0, "No known-present cross-source historical control"
    calibration = sibling("reader-calibration")
    corrected_path = calibration.RUN / "corrected-readings.jsonl"
    corrected = {}
    for row in calibration.recorded_calls() if corrected_path.exists() else []:
        if row.get("error") or row["name"] not in members:
            continue
        target = corrected.setdefault(row["name"], {})
        assert row["seat"] not in target
        target[row["seat"]] = sorted(row["defect_classes"])
    receipt = {
        "verdict_commit_before_exposure": verdict_sha,
        "opened_after_verdict_commit_at": datetime.now(UTC).isoformat(),
        "graph_audit_group": audit.PRIOR_GROUP,
        "bounded_member_ids": sorted(members),
        "graph_rows": len(rows),
        "ignored_non_pair_seats": ignored_seats,
        "known_present_cross_source_matches": controls,
        "third_reader_source": str(third_path),
        "third_reader_sha256": hashlib.sha256(third_path.read_bytes()).hexdigest(),
        "corrected_reader_source": str(corrected_path),
        "corrected_reader_status": "available"
        if corrected_path.exists()
        else "receipt path absent; optional corrected-instruction overlap unavailable",
        "corrected_reader_sha256": hashlib.sha256(
            corrected_path.read_bytes()
        ).hexdigest()
        if corrected_path.exists()
        else None,
        "earlier": earlier,
        "corrected": corrected,
        "graph_writes": 0,
        "model_calls": 0,
    }
    save(receipt, RUN / "comparison-inputs.json")
    print(
        f"COMPARISON: {len(rows)} bounded historical rows; {controls} known-present cross-source matches; "
        f"earlier members {len(earlier)}; corrected members {len(corrected)}; graph writes 0; model calls 0"
    )


def share(count, total):
    return {
        "count": count,
        "total": total,
        "share": count / total if total else None,
        "wilson_95": list(sibling("blind-audit").wilson(count, total)),
    }


def review_draw(judgments):
    """Largest-remainder allocation, seeded draws within all verdict strata."""
    strata = defaultdict(list)
    for row in judgments:
        strata[row["verdict"]].append(row["id"])
    size = 20
    n = len(judgments)
    allocation = {key: size * len(names) // n for key, names in strata.items()}
    remainders = sorted(strata, key=lambda k: (-(size * len(strata[k]) % n), k))
    for key in remainders[: size - sum(allocation.values())]:
        allocation[key] += 1
    assert allocation["sound"] > 0 and allocation["defective"] > 0
    rng = random.Random(SEED)
    selected = []
    for key in sorted(strata):
        selected.extend(rng.sample(sorted(strata[key]), allocation[key]))
    rng.shuffle(selected)
    return {
        "seed": SEED,
        "algorithm": "Largest-remainder proportional allocation across all three verdict strata; sorted strata and names; Python Random.sample within strata then Random.shuffle",
        "allocation": dict(sorted(allocation.items())),
        "members": selected,
        "lead_verdicts": dict.fromkeys(selected, ""),
    }


def compare_readers(judgments, readings):
    out = {}
    for seat in ("A", "B", "C"):
        observed = [j for j in judgments if seat in readings.get(j["id"], {})]
        paired = [j for j in observed if j["verdict"] != "undecidable"]
        rows = []
        for j in observed:
            classes = readings[j["id"]][seat]
            rows.append(
                {
                    "id": j["id"],
                    "adjudication": j["verdict"],
                    "reader_classes": classes,
                    "binary_agreement": None
                    if j["verdict"] == "undecidable"
                    else bool(classes) == (j["verdict"] == "defective"),
                    "exact_class_agreement": None
                    if j["verdict"] == "undecidable"
                    else set(classes) == set(j["defect_classes"]),
                }
            )
        agreement = sum(r["binary_agreement"] is True for r in rows)
        exact = sum(r["exact_class_agreement"] is True for r in rows)
        defective = [j for j in paired if j["verdict"] == "defective"]
        sound = [j for j in paired if j["verdict"] == "sound"]
        out[seat] = {
            "observed": len(observed),
            "undecidable_reference": len(observed) - len(paired),
            "binary_agreement": share(agreement, len(paired)),
            "exact_class_agreement": share(exact, len(paired)),
            "detected_defective": share(
                sum(bool(readings[j["id"]][seat]) for j in defective), len(defective)
            ),
            "flagged_sound": share(
                sum(bool(readings[j["id"]][seat]) for j in sound), len(sound)
            ),
            "rows": rows,
        }
    return out


CAUSES = {
    "physical_scope_qualifier_lost": "The naming stage drops a source-stated component, geometric reference or physical locus.",
    "source_bindings_conflate_distinct_definitions": "Producer bindings pool quantities whose loci, operations or physical meanings cannot share the stored definition.",
    "boundary_forced_to_lcfs": "Definition templates replace the source's selectable boundary with the last closed flux surface.",
    "normalization_or_weighting_qualifier_lost": "Naming drops an explicit normalization or weighting that distinguishes the source quantity.",
    "coordinate_reference_qualifier_lost": "Naming or definition construction drops or substitutes the source's coordinate convention or reference axis.",
    "vacuum_source_qualifier_lost": "Naming drops a vacuum-field contribution restriction and can admit plasma-response or total-field interpretations.",
    "pressure_pedestal_qualifier_lost": "The pressure-pedestal locus collapses to an unspecified pedestal in the name.",
    "transformation_equation_replaced_by_base_quantity": "Definition construction substitutes a base quantity or an inconsistent equation for the named transform or derivative.",
    "averaging_qualifier_lost": "A source's average or aggregation disappears from the name or is explicitly excluded by a pointwise definition.",
    "absolute_extremum_substituted_for_signed_extremum": "A signed safety-factor minimum is replaced with the minimum of its absolute value.",
    "population_basis_qualifier_lost": "The source's thermal or total population basis disappears from the name or is mixed across bindings.",
    "physical_scope_qualifier_replaced": "Definition construction substitutes a different geometric or coordinate scope for the source's scope.",
    "locus_template_overrides_source": "A generic geometric template overrides the source's stated branch or point-selection rule.",
    "consumer_context_used_as_quantity_domain": "Domain assignment follows an equilibrium consumer instead of the iron-core or structural quantity itself.",
}


def summarize(data, comparison):
    judgments = data["judgments"]
    eligible = {j["id"] for j in judgments}
    assert comparison["eligible_members"] == sorted(eligible)
    assert all(set(comparison[key]) <= eligible for key in ("earlier", "corrected")), (
        "Pending member present in reader comparison"
    )
    n = len(judgments)
    counts = Counter(j["verdict"] for j in judgments)
    classes = Counter(c for j in judgments for c in j["defect_classes"])
    candidates = defaultdict(list)
    for row in judgments:
        for cause in row["candidate_cause"].split("|"):
            if cause:
                candidates[cause].append(row["id"])
        if "wrong_physics_domain" in row["defect_classes"]:
            candidates["consumer_context_used_as_quantity_domain"].append(row["id"])
    special_classes = {"duplicate_of_live_name", "wrong_physics_domain"}
    return {
        "population": n,
        "pending": data["pending"],
        "coverage_commit": committed_field("coverage")[0],
        "verdicts": {
            v: share(counts[v], n) for v in ("sound", "defective", "undecidable")
        },
        "classes": {c: share(classes[c], n) for c, _ in data["class_list"]},
        "candidate_pipeline_defects": [
            {
                "cause": cause,
                "mechanism": CAUSES[cause],
                "count": len(names),
                "members": sorted(names),
            }
            for cause, names in sorted(
                candidates.items(), key=lambda r: (-len(r[1]), r[0])
            )
            if len(names) >= 3
        ],
        "duplicate_or_wrong_domain_only": [
            j["id"]
            for j in judgments
            if j["defect_classes"] and set(j["defect_classes"]) <= special_classes
        ],
        "undecidable": [j["id"] for j in judgments if j["verdict"] == "undecidable"],
        "defect_share_if_all_undecidable_sound": counts["defective"] / n,
        "defect_share_if_all_undecidable_defective": (
            counts["defective"] + counts["undecidable"]
        )
        / n,
        "lead_review": review_draw(judgments),
        "comparison_inputs": comparison,
        "historical_reader_agreement": compare_readers(
            judgments, comparison["earlier"]
        ),
        "corrected_reader_agreement": compare_readers(
            judgments, comparison["corrected"]
        ),
        "packet_commit": committed_snapshot("packets-frozen")[0],
        "verdict_commit": committed_snapshot("verdicts-frozen")[0],
    }


def percentage(value):
    return "not estimable" if value is None else f"{100 * value:.1f}%"


def interval(value):
    if not value["total"]:
        return "not estimable"
    low, high = value["wilson_95"]
    return f"{percentage(low)}–{percentage(high)}"


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
        )
        + "</tbody></table>"
    )


def render_page(data):
    report = data["report"]
    visible = {p["id"]: p for p in packets(data)}
    judgments = {j["id"]: j for j in data["judgments"]}
    e = escape
    result = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Equilibrium source adjudication and lead review</title>",
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1.5rem;color:#222}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{padding:.55rem;text-align:left;vertical-align:top;border-bottom:1px solid #ccc;overflow-wrap:anywhere}th{font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.45 ui-monospace,monospace}summary{cursor:pointer;font-weight:600}details{margin:1rem 0}textarea{min-height:8rem;width:95%;font:inherit}code{overflow-wrap:anywhere}.review-card{margin:3rem 0;padding-top:1rem;border-top:1px solid #777}a{color:#245c91}small{font-size:.85em}@media print{body{max-width:none}.review-card{break-before:page}}</style></head><body>",
        "<h1>Equilibrium: judge the source packet first</h1>",
        '<p><strong>208 adjudicated members; two pending fresh adjudication.</strong> The original verdicts for the 208 complete packets are preserved exactly. <code>radial_coordinate_of_conductor_cross_section</code> and <code>vertical_outline</code> are pending because their original packets omitted six enclosing-text fields. Both are excluded from every verdict and class count, rate, interval, shared-cause count, reader comparison and review draw on this page. They were not re-judged after reader exposure. Complete corrected packets are available in <a href="equilibrium-pending-packets.json">the neutral packet handoff</a>, which contains no verdict or reader information. The pending-member table below identifies every omitted field.</p>',
        "<p>Twenty members were drawn by a recorded seed from only the 208 complete, adjudicated packets. Read each packet, enter your own verdict in the blank column, then open the disclosure to compare. Every packet below is complete and contains no earlier reader verdict, score or adjudication. Fields are blank in this file; entered text is not automatically saved.</p>",
        f"<p>Packet seal <code>{report['packet_commit']}</code>; blind verdict seal <code>{report['verdict_commit']}</code>. The verdict seal predates all individual reader exposure. No model calls, paid reader calls or graph writes were made for this adjudication.</p>",
    ]
    for i, name in enumerate(report["lead_review"]["members"], 1):
        j = judgments[name]
        p = visible[name]
        result.extend(
            [
                f'<article class="review-card" id="review-{i}"><h2>{i}. {e(name)}</h2>',
                f"<p><small>Packet SHA-256: {digest(p)}</small></p><pre>{e(p['reader_visible_text'])}</pre>",
                table(
                    [
                        "Lead verdict and reason",
                        "Committed adjudication — reveal after judging",
                    ],
                    [
                        [
                            f'<textarea aria-label="Lead verdict for {e(name)}"></textarea>',
                            '<details class="adjudication"><summary>Reveal committed adjudication</summary>'
                            f"<p><strong>{j['verdict']}</strong>: {e(', '.join(j['defect_classes']) or 'no defect class')}</p>"
                            f"<p>{e(j['reason'])}</p><p>Evidence binding: <code>{e(j['evidence_path'])}</code></p></details>",
                        ]
                    ],
                ),
                "</article>",
            ]
        )
    result.extend(
        [
            '<details id="audit-results"><summary>Open results, pending members, methods and all 208 retained verdicts after reviewing</summary>',
            "<h2>Adjudicated cohort</h2>",
            "<p>The frozen membership frame has 210 names: 208 complete packets retain their original blind verdicts; exactly two are pending and excluded from all results. Full ancestor-field coverage is independently committed, and the gate verifies every source chain against it. The two corrected packets include all enclosing statements without the inherited four-statement cap. Raw producer text and structural family context are retained. Text is whitespace-normalized without clipping after context selection. The compact JSON stores lossless compressed packets, verdicts and coverage requirements; this script decodes them. The freeze is a point-in-time snapshot, not a claim about later graph changes.</p>",
            "<h3>Pending fresh adjudication — excluded from all results</h3>",
            table(
                [
                    "Pending member",
                    "Source binding",
                    "Omitted ancestor",
                    "Field",
                    "Omitted source words",
                ],
                [
                    [
                        e(p["id"]),
                        e(field["source"]),
                        e(field["ancestor"]),
                        e(field["field"]),
                        e(field["text"]),
                    ]
                    for p in data["pending"]
                    for field in p["omitted_fields"]
                ],
            ),
            table(
                [
                    "Verdict",
                    f"Count / {report['population']}",
                    "Share",
                    "95% Wilson interval",
                ],
                [
                    [
                        v,
                        f"{r['count']} / {r['total']}",
                        percentage(r["share"]),
                        interval(r),
                    ]
                    for v, r in report["verdicts"].items()
                ],
            ),
            "<p>Counts and shares refer only to the 208 retained adjudications. Pending members contribute to no numerator or denominator. Wilson intervals are supplied as binomial descriptive intervals for the requested proportions; they do not quantify adjudicator error or sampling uncertainty for this fixed evaluated cohort. Undecidable is not counted as sound. Classes overlap and each class uses the same 208 evaluated members as denominator.</p>",
            table(
                [
                    "Defect class",
                    "Meaning",
                    f"Count / {report['population']}",
                    "Share",
                    "95% Wilson interval",
                ],
                [
                    [
                        e(c),
                        e(label),
                        f"{report['classes'][c]['count']} / {report['population']}",
                        percentage(report["classes"][c]["share"]),
                        interval(report["classes"][c]),
                    ]
                    for c, label in data["class_list"]
                ],
            ),
            f"<p>If every undecidable case were sound, the defective share would be {percentage(report['defect_share_if_all_undecidable_sound'])}; if every undecidable case were defective, it would be {percentage(report['defect_share_if_all_undecidable_defective'])}. These are bounds on unresolved labels, not confidence limits.</p>",
            "<h2>Candidate pipeline defects shared by at least three members</h2>",
            "<p>These are mechanism candidates inferred from repeated source-to-name or source-to-definition mismatches. They are not causal traces through production prompts and do not authorize member-by-member patching. Groups can overlap; each member links to its committed wording, reason and source binding.</p>",
            table(
                ["Candidate mechanism", "Members", "Committed members"],
                [
                    [
                        e(row["mechanism"]),
                        str(row["count"]),
                        "<br>".join(
                            f'<a href="#member-{e(name)}">{e(name)}</a>'
                            for name in row["members"]
                        ),
                    ]
                    for row in report["candidate_pipeline_defects"]
                ],
            ),
            "<h2>Duplicate or wrong-domain classes only</h2>",
            "<p>These members have no other defect class. An accepted alternative below is a semantic identity to consider, not a graph edit or a new spelling approved for release.</p>",
            table(
                [
                    "Stored name",
                    "Only classes",
                    "Source binding",
                    "Reason and intended distinction",
                ],
                [
                    [
                        e(name),
                        e(", ".join(judgments[name]["defect_classes"])),
                        e(judgments[name]["evidence_path"]),
                        e(judgments[name]["reason"]),
                    ]
                    for name in report["duplicate_or_wrong_domain_only"]
                ],
            ),
            "<h2>Undecidable from the supplied text</h2>",
            "<p>Some derived producers have no independent text, some sources conflict on the coordinate convention, and some definitions assert a precise external shape formula that the packet only cites. No external paper, additional graph scope or earlier reader label was used to fill those gaps. The source owner must supply the missing definition, convention or cited formula before these cases can be decided.</p>",
            table(
                ["Name", "Evidence binding", "Why it could not be decided"],
                [
                    [
                        e(name),
                        e(judgments[name]["evidence_path"]),
                        e(judgments[name]["reason"]),
                    ]
                    for name in report["undecidable"]
                ],
            ),
            "<h2>Earlier readers: comparison only</h2>",
            "<p>Historical A/B audit rows were read from the graph after the verdict commit; C comes from the retained third-reader evidence. All comparison inputs shown here are restricted to the 208 retained members; pending members are excluded even when a historical reading exists. Corrected-instruction calibration readings are shown separately. No reader output contributed to an adjudication. Agreement is computed only on observed, decidable overlaps; an absent reading is never treated as sound and undecidable references are excluded from binary and exact-class agreement. The observed subsets are selected historical cohorts, not random validation samples. Source text and instruction versions may differ, so agreement does not isolate model ability or the effect of the instruction change.</p>",
        ]
    )
    for label, key in (
        ("Historical audit instructions", "historical_reader_agreement"),
        ("Corrected-instruction calibration subset", "corrected_reader_agreement"),
    ):
        if (
            key == "corrected_reader_agreement"
            and not report["comparison_inputs"]["corrected"]
        ):
            result.append(
                "<h3>Corrected-instruction calibration overlap unavailable</h3><p>The earlier calibration document names a per-call receipt file that is absent in this execution environment. Its aggregate scores do not recover individual class lists, so no corrected-instruction agreement is inferred. The required historical A/B/C comparisons above remain available.</p>"
            )
            continue
        result.extend(
            [
                f"<h3>{label}</h3>",
                table(
                    [
                        "Reader",
                        "Observed",
                        "Undecidable reference",
                        "Binary agreement",
                        "95% interval",
                        "Exact class-set agreement",
                        "Detected defective",
                        "Flagged sound",
                    ],
                    [
                        [
                            seat,
                            str(row["observed"]),
                            str(row["undecidable_reference"]),
                            f"{row['binary_agreement']['count']}/{row['binary_agreement']['total']} ({percentage(row['binary_agreement']['share'])})",
                            interval(row["binary_agreement"]),
                            f"{row['exact_class_agreement']['count']}/{row['exact_class_agreement']['total']}",
                            f"{row['detected_defective']['count']}/{row['detected_defective']['total']}",
                            f"{row['flagged_sound']['count']}/{row['flagged_sound']['total']}",
                        ]
                        for seat, row in report[key].items()
                    ],
                ),
                "<details><summary>Per-member reader comparisons</summary>",
                table(
                    [
                        "Reader",
                        "Name",
                        "Committed verdict",
                        "Reader classes",
                        "Binary agreement",
                        "Exact classes",
                    ],
                    [
                        [
                            seat,
                            e(r["id"]),
                            r["adjudication"],
                            e(", ".join(r["reader_classes"]) or "none"),
                            "excluded: undecidable"
                            if r["binary_agreement"] is None
                            else str(r["binary_agreement"]),
                            "excluded: undecidable"
                            if r["exact_class_agreement"] is None
                            else str(r["exact_class_agreement"]),
                        ]
                        for seat, row in report[key].items()
                        for r in row["rows"]
                    ],
                ),
                "</details>",
            ]
        )
    comparison = report["comparison_inputs"]
    result.extend(
        [
            "<h2>Reproducibility and review draw</h2>",
            f"<p>Seed <code>{SEED}</code>. {e(report['lead_review']['algorithm'])}. Allocation: {e(json.dumps(report['lead_review']['allocation'], sort_keys=True))}. All twenty lead verdicts remain blank. The lead must complete this independent check before the group proceeds; this document grants no release authority.</p>",
            f"<p>Comparison opened at {e(comparison['opened_after_verdict_commit_at'])} after verdict commit <code>{comparison['verdict_commit_before_exposure']}</code>. The original bounded-read receipt and its known-present controls remain in the run record. The comparison tables and their stored inputs are restricted to eligible members. Non-pair audit seats are excluded using the existing helper’s rule. Complete coverage and the neutral pending packets are independently sealed at <code>{report['coverage_commit']}</code>.</p>",
            "<p>The seven classes and reconciled omission instruction come from the existing blind-audit helper. A source-stated component, locus, basis, coordinate, weight or normalization omitted from a name is a defect; stylistic preferences and unstated qualifiers are not. Missing source text yields undecidable unless a source-independent class can be established. Derived producer descriptions are retained as evidence, not invented DD text. Generic-versus-specific derived identities and ambiguous source conventions remain matters for the lead where the committed reason says so.</p>",
            "<p>Run <code>uv run --no-sync python docs/evidence/fragments/sn-audited-group-release/equilibrium-adjudication.py --check</code> in the repository environment. The gate requires exactly the two named pending exceptions, complete source context and an unchanged committed verdict for every other member, exclusion of pending members from all results, complete neutral handoff packets, and a twenty-member draw only from the retained cohort. Altering a retained verdict and recomputing its digest, erasing a verdict, or omitting any other packet context is refused. No paid calls were authorized or made: actual reader spend USD 0; authorized paid-call ceiling USD 0.</p>",
            "<h2>Every retained committed verdict</h2>",
            "<table><thead><tr><th>Name and stored definition</th><th>Quoted source binding</th><th>Verdict and classes</th><th>Quoted reason</th></tr></thead><tbody>",
        ]
    )
    for name in judgments:
        p = visible[name]
        j = judgments[name]
        definition = (
            p["reader_visible_text"]
            .split("Definition (stored description): ", 1)[1]
            .split("\n", 1)[0]
        )
        sources = f"<code>{e(j['evidence_path'])}</code><p><small>{len(p['source_paths'])} frozen bindings</small></p>"
        result.append(
            f'<tr id="member-{e(name)}"><td><code>{e(name)}</code><p>{e(definition)}</p></td><td>{sources}</td><td><strong>{j["verdict"]}</strong><p>{e(", ".join(j["defect_classes"]) or "none")}</p></td><td>{e(j["reason"])}</td></tr>'
        )
    result.extend(["</tbody></table></details></body></html>"])
    return "\n".join(result) + "\n"


def report():
    check(include_report=False)
    data = load()
    comparison = load(RUN / "comparison-inputs.json")
    assert (
        comparison["verdict_commit_before_exposure"]
        == committed_snapshot("verdicts-frozen")[0]
    )
    assert comparison["bounded_member_ids"] == data["members"]
    eligible = {j["id"] for j in data["judgments"]}
    comparison = {
        key: value
        for key, value in comparison.items()
        if key
        not in (
            "bounded_member_ids",
            "graph_rows",
            "ignored_non_pair_seats",
            "known_present_cross_source_matches",
            "earlier",
            "corrected",
        )
    } | {
        "eligible_members": sorted(eligible),
        "source_snapshot": str(RUN / "comparison-inputs.json"),
        "earlier": {
            name: rows
            for name, rows in load(RUN / "comparison-inputs.json")["earlier"].items()
            if name in eligible
        },
        "corrected": {
            name: rows
            for name, rows in load(RUN / "comparison-inputs.json")["corrected"].items()
            if name in eligible
        },
    }
    data["report"] = summarize(data, comparison)
    save(data)
    PAGE.write_text(render_page(data))
    print(
        json.dumps(
            {
                "verdicts": {
                    v: r["count"] for v, r in data["report"]["verdicts"].items()
                },
                "candidate_pipeline_defects": len(
                    data["report"]["candidate_pipeline_defects"]
                ),
                "duplicate_or_wrong_domain_only": len(
                    data["report"]["duplicate_or_wrong_domain_only"]
                ),
                "lead_review_allocation": data["report"]["lead_review"]["allocation"],
                "data_bytes": DATA.stat().st_size,
                "html_bytes": PAGE.stat().st_size,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seal-verdicts", action="store_true")
    parser.add_argument("--collect-comparison", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--collect-pending-packets", action="store_true")
    parser.add_argument("--exclude-pending", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.check:
        check(args.data)
    elif args.seal_verdicts:
        seal_verdicts()
    elif args.collect_comparison:
        collect_comparison()
    elif args.report:
        report()
    elif args.collect_pending_packets:
        collect_pending_packets()
    elif args.exclude_pending:
        exclude_pending()


if __name__ == "__main__":
    main()

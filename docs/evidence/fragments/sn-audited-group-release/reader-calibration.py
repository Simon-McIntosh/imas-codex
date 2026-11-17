"""Freeze source judgments, then calibrate three blind audit readers.

Candidates are drawn in seeded round-robin order across the joint earlier
reader flag strata. The adjudicator sees only source packets. All standing
labels from the source-only check are retained. New judgments are added in draw order
until at least forty labels of each kind stand; every decidable judgment
from that prefix is retained in the reader experiment. No graph writes are made.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "reader-calibration.json"
PAGE = ROOT / "reader-calibration.html"
RELATIVE_DATA = str(DATA.relative_to(ROOT.parents[3]))
RUN = Path(
    os.environ.get(
        "READER_CALIBRATION_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261117T120150171460-sagr-reader-calibration",
    )
)
SEED = 2026111703
TARGET = 40
CEILING = 10.0
BEFORE = "Report a missing qualifier only when neither the source leaf nor any enclosing structure states it."
AFTER = "Report a missing qualifier when the source leaf or an enclosing structure states it and the name leaves it out."


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    )


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def revision():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def audit_module():
    spec = importlib.util.spec_from_file_location(
        "blind_audit", ROOT / "blind-audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_candidates():
    """Project only identities and strata; no per-name verdict reaches the judge."""
    audit = load(ROOT / "blind-audit.json")
    third = load(ROOT / "frontier-third-reader.json")
    first = load(ROOT / "planted-label-check.json")
    excluded = {p["id"] for p in first["packets"]}
    readings = audit_module().prior_audit_readings()
    save(RUN / "earlier-readings.json", readings)
    pair = {
        name: (bool(r["A"]["classes"]), bool(r["B"]["classes"]))
        for name, r in readings.items()
        if "A" in r and "B" in r
    }
    ids = set(pair)
    ids -= {r["id"] for r in audit["planted_detail"]["rows"]}
    ids -= {r["name"] for r in audit["missing_pairs"]}
    assert len(ids) == 339
    third_flags = {
        r["name"]: bool(r["C"])
        for kind in ("flagged", "control")
        for r in third[kind]["rows"]
    }
    strata = defaultdict(list)
    for name in sorted(ids - excluded):
        a, b = pair.get(name, (False, False))
        strata[
            f"{int(a)}{int(b)}"
            + (str(int(third_flags[name])) if name in third_flags else "u")
        ].append(name)
    rng = random.Random(SEED)
    order = []
    for key in sorted(strata):
        rng.shuffle(strata[key])
    while any(strata.values()):
        for key in sorted(strata):
            if strata[key]:
                order.append(strata[key].pop())
    return order, {
        "seed": SEED,
        "frame_size": len(ids),
        "eligible": len(order),
        "method": "Round-robin across joint A/B/C defective flags; u means C not read; shuffle each sorted stratum with random.Random(seed)",
        "frame_digest": digest(sorted(ids)),
        "order_digest": digest(order),
    }


def collect(count):
    assert not DATA.exists(), "Refuse to overwrite the calibration record"
    order, draw = draw_candidates()
    first = load(ROOT / "planted-label-check.json")
    standing = {r["id"] for r in first["comparison"] if r["label_verdict"] == "stands"}
    packets = [p for p in first["packets"] if p["id"] in standing]
    judgments = [j for j in first["judgments"] if j["id"] in standing]
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.source_reads import source_paths_projection

    audit = audit_module()
    ids = order[:count]
    gc = GraphClient()
    try:
        rows = list(
            gc.query(
                f"""
            UNWIND $ids AS wanted MATCH (sn:StandardName {{id: wanted}})
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
        assert {r["id"] for r in rows} == set(ids), "Missing requested names"
        audit.load_dd_context(gc, rows)
        paths = sorted({p for r in rows for p in r["dd_paths"]})
        sources = list(
            gc.query(
                f"""
            UNWIND $paths AS pid MATCH (src:StandardNameSource {{id: 'dd:' + pid}})-[:PRODUCED_NAME]->(sn:StandardName)
            WHERE coalesce(src.status,'') <> 'stale' AND {audit.LIVE}
            RETURN pid AS path, collect(DISTINCT sn.id) AS names
        """,
                paths=paths,
            )
        )
        by_path = {r["path"]: r["names"] for r in sources}
        memberships = sum(
            r["id"] in by_path.get(p, []) for r in rows for p in r["dd_paths"]
        )
        expected = sum(len(r["dd_paths"]) for r in rows)
        assert memberships == expected > 0, "Producer join missed a known membership"
        assert {p for r in rows for p in r["dd_paths"]} == {
            d["path"] for r in rows for d in r["dd_docs"]
        }, "Missing source text"
        audit.attach_duplicate_candidates(rows, by_path)
        for row in rows:
            text = audit.render_user_prompt(row)
            audit.assert_blind(text, row)
            packets.append(
                {
                    "id": row["id"],
                    "source_paths": row["source_paths"],
                    "reader_visible_text": text,
                }
            )
        receipt = {
            "requested": len(ids),
            "returned": len(rows),
            "dd_paths": len(paths),
            "producer_memberships": memberships,
            "expected_memberships": expected,
            "duplicate_candidates": sum(len(r["other_live_names"]) for r in rows),
            "graph_writes": 0,
            "model_calls": 0,
        }
    finally:
        gc.close()
    # A packet payload may exceed the repository's single-file size ceiling.
    packet_path = RUN / "source-packets.json"
    save(packet_path, packets)
    save(
        DATA,
        {
            "phase": "source-only",
            "collection_revision": revision(),
            "collected_at": datetime.now(UTC).isoformat(),
            "draw": draw | {"candidate_order": ids},
            "standing_ids": sorted(standing),
            "packet_file": str(packet_path),
            "packet_digest": digest(packets),
            "collection_receipt": receipt,
            "judgments": judgments,
            "cohort_rule": "Retain every standing label, then first new decidable labels in seeded draw order until forty defective and forty sound; retain surplus adjudications separately. Freeze every judgment before any new reader comparison.",
            "blindness": {
                "new_judge_exposure": "Source-only packets and aggregate plan context; no per-name reader flags, strata, labels or rationales",
                "inherited": "Standing source-only judgments committed by the first hand check; earlier reader results were already compared there",
            },
        },
    )
    print(json.dumps(receipt))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collect", type=int)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--read", action="store_true")
    p.add_argument("--check-blind", action="store_true")
    args = p.parse_args()
    if args.collect:
        collect(args.collect)
    elif args.check_blind:
        check_blind(load(DATA))
    elif args.pilot or args.read:
        asyncio.run(read_corrected(pilot_only=args.pilot))


def source_packets(data):
    """Reconstruct exact reader text from a shared lossless line dictionary."""
    if "text_pool" not in data:
        return data["packets"]
    return [
        {
            "id": p["id"],
            "source_paths": p["source_paths"],
            "reader_visible_text": "\n".join(
                data["text_pool"][i] for i in p["text_lines"]
            ),
        }
        for p in data["packets"]
    ]


def compact_packets(data):
    packets = source_packets(data)
    pool, indices = [], {}
    compact = []
    for p in packets:
        numbers = []
        for line in p["reader_visible_text"].split("\n"):
            if line not in indices:
                indices[line] = len(pool)
                pool.append(line)
            numbers.append(indices[line])
        compact.append(
            {"id": p["id"], "source_paths": p["source_paths"], "text_lines": numbers}
        )
    data["text_pool"], data["packets"] = pool, compact
    assert source_packets(data) == packets


def check_blind(data):
    packets = {p["id"]: p for p in source_packets(data)}
    all_packets = {p["id"]: p for p in load(data["packet_file"])}
    assert digest(list(all_packets.values())) == data["packet_digest"], (
        "Collected packets changed"
    )
    assert digest(list(packets.values())) == data["cohort_packet_digest"], (
        "Cohort packets changed"
    )
    assert all(p == all_packets[n] for n, p in packets.items())
    assert digest(data["judgments"]) == data["judgment_digest"], (
        "Blind judgments changed"
    )
    by_name = {j["id"]: j for j in data["judgments"]}
    assert len(by_name) == len(data["judgments"])
    assert len(data["cohort"]) == len(set(data["cohort"])) == len(packets)
    assert set(data["cohort"]) == set(packets)
    counts = Counter(by_name[n]["verdict"] for n in data["cohort"])
    assert counts["defective"] >= TARGET and counts["sound"] >= TARGET
    assert set(data["standing_ids"]) <= set(data["cohort"])
    assert not set(data["blindness"]["exposure_exclusions"]) & set(by_name)
    first = load(ROOT / "planted-label-check.json")
    original = {j["id"]: j for j in first["judgments"]}
    assert all(by_name[n] == original[n] for n in data["standing_ids"])
    audit = audit_module()
    for n, judgment in by_name.items():
        packet = all_packets[n]["reader_visible_text"]
        assert judgment["evidence_quote"] in packet, "Quote missing: " + n
        assert judgment["evidence_path"] in packet, "Source binding missing: " + n
        assert judgment["verdict"] in ("defective", "sound", "undecidable")
        assert bool(judgment["defect_classes"]) == (judgment["verdict"] == "defective")
        assert set(judgment["defect_classes"]) <= set(audit.CLASS_KEYS)
    assert AFTER in audit.system_prompt() and BEFORE not in audit.system_prompt()
    print(
        "PASS: source-only cohort has at least 40 defective and 40 sound; all 26 standing labels retained; quoted sources and lossless packets verified"
    )


BLIND_COMMIT = "4aed6185c57bf6eb1e9de7ca86b324aa3bdb6869"


def frozen_record():
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BLIND_COMMIT}:{RELATIVE_DATA}"]
    )
    frozen = json.loads(raw)
    assert frozen["phase"] == "judgments-frozen"
    assert not {"results", "scores", "comparison"} & frozen.keys()
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BLIND_COMMIT, "HEAD"],
        check=True,
    )
    current = load(DATA)
    for key, value in frozen.items():
        if key != "phase":
            assert current[key] == value, (
                "Frozen field changed after blind commit: " + key
            )
    check_blind(frozen)
    return frozen


def reading_order(data):
    order = sorted(data["cohort"])
    random.Random(SEED + 1).shuffle(order)
    return order


def recorded_calls():
    path = RUN / "corrected-readings.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


async def read_corrected(*, pilot_only):
    """Use the audit's call owner with the frozen exact source-only user text."""
    from imas_codex.discovery.base.llm import get_openrouter_max_price

    data = frozen_record()
    audit = audit_module()
    readers = (*audit.READERS, audit.third_reader())
    audit.assert_readers_not_pipeline(readers)
    packets = {p["id"]: p for p in source_packets(data)}
    # Preserve the original rendered packet byte for byte; only the system
    # omission instruction differs. The call still owns schema, retries,
    # token cap, prompt blindness, receipt fields and service attribution.
    audit.render_user_prompt = lambda row: row["reader_visible_text"]
    order = reading_order(data)
    system = audit.system_prompt()
    save(
        RUN / "corrected-prompt.json",
        {
            "system": system,
            "readers": readers,
            "blind_commit": BLIND_COMMIT,
            "order": order,
        },
    )
    records = recorded_calls()
    assert not any(r.get("error") or r.get("cost") is None for r in records), (
        "An unpriced or failed call requires adjudication before retry"
    )
    spent = sum(r["cost"] for r in records)
    done = {(r["seat"], r["name"]) for r in records}
    assert len(done) == len(records), "Duplicate billed call"
    if not pilot_only:
        pilot = load(RUN / "pilot.json")
        assert pilot["complete"] and pilot["projected_usd"] <= CEILING, (
            "Pilot projection exceeds ceiling"
        )
        assert pilot["names"] == order[:6]
    names = order[:6] if pilot_only else order
    for index, name in enumerate(names):
        todo = [r for r in readers if (r["seat"], name) not in done]
        reserves = []
        for reader in todo:
            rates = get_openrouter_max_price(reader["model"])
            # UTF-8 bytes bound text tokens conservatively, with schema and
            # message overhead. Three attempts are the audit call's retry cap.
            input_bound = (
                len((system + packets[name]["reader_visible_text"]).encode()) + 4096
            )
            reserves.append(
                3
                * (
                    input_bound * rates["prompt"] / 1e6
                    + audit.MAX_TOKENS * rates["completion"] / 1e6
                    + rates["request"]
                )
            )
        assert spent + sum(reserves) <= CEILING, (
            f"Cannot reserve bounded calls within USD {CEILING}; spent {spent}"
        )
        spend = {"total": spent}
        replies = await asyncio.gather(
            *(
                audit.judge_one(reader, packets[name], spend, ceiling=CEILING)
                for reader in todo
            )
        )
        for reply in replies:
            reply["instruction_digest"] = digest(system)
            reply["packet_digest"] = digest(packets[name])
            reply["recorded_at"] = datetime.now(UTC).isoformat()
            with (RUN / "corrected-readings.jsonl").open("a") as handle:
                handle.write(json.dumps(reply, ensure_ascii=False) + "\n")
        records.extend(replies)
        assert not any(r.get("error") or r.get("cost") is None for r in replies), (
            "Failed or unpriced call; stopped without retrying the experiment"
        )
        spent = sum(r["cost"] for r in records)
        assert spent <= CEILING
        print(
            f"{index + 1}/{len(names)} names complete; recorded spend USD {spent:.6f} / {CEILING:.2f}",
            flush=True,
        )
    if pilot_only:
        pilot_records = [r for r in records if r["name"] in names]
        assert len(pilot_records) == 18
        paid = sum(r["cost"] for r in pilot_records)
        projection = paid / 6 * len(order)
        receipt = {
            "names": names,
            "calls": len(pilot_records),
            "complete": True,
            "spend_usd": paid,
            "projected_usd": projection,
            "ceiling_usd": CEILING,
            "all_names": len(order),
            "method": "Recorded cost of six names through all three readers, multiplied by cohort size / 6",
            "per_reader_usd": {
                r["seat"]: sum(
                    x["cost"] for x in pilot_records if x["seat"] == r["seat"]
                )
                for r in readers
            },
        }
        save(RUN / "pilot.json", receipt)
        print("PILOT " + json.dumps(receipt), flush=True)
        assert projection <= CEILING, (
            "Pilot projection exceeds USD 10; no full run authorized by the protocol"
        )


if __name__ == "__main__":
    main()

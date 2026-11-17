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
import math
import os
import random
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html import escape
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
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--collect", type=int)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--read", action="store_true")
    p.add_argument("--check-blind", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--through", type=int, default=0)
    args = p.parse_args()
    if args.collect:
        collect(args.collect)
    elif args.check_blind:
        check_blind(load(DATA))
    elif args.pilot or args.read:
        asyncio.run(read_corrected(pilot_only=args.pilot, through=args.through))
    elif args.report:
        report()
    elif args.check:
        check(args.data)


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


def frozen_record(path=DATA):
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
    current = load(path)
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


def spending_rates(audit):
    from imas_codex.settings import get_openrouter_pricing

    rates = {}
    for reader in (*audit.READERS, audit.third_reader()):
        price = get_openrouter_pricing(reader["model"])
        if not price:
            candidate = next(
                p
                for p in audit.model_selection()["candidates"]
                if p["model"] == reader["model"]
            )
            price = {
                "prompt": candidate["prompt_usd_per_mtok"],
                "completion": candidate["completion_usd_per_mtok"],
            }
        rates[reader["model"]] = {
            "prompt": max(float(price["prompt"]), float(price.get("cache_write") or 0)),
            "completion": float(price["completion"]),
            "request": 0.0,
            "image": 0.0,
        }
    # The unfiltered reader is reserved at the shared text policy's upper
    # token rates, above its listed price. No per-request charge is invented.
    rates[audit.READERS[0]["model"]].update(prompt=20.0, completion=100.0)
    return rates


def call_bound(packet, system, rates, output_ceiling):
    input_bound = len((system + packet["reader_visible_text"]).encode()) + 4096
    return (input_bound * rates["prompt"] + output_ceiling * rates["completion"]) / 1e6


def budget_account(records, packets, system, rates):
    priced = sum(r["cost"] for r in records if r.get("cost") is not None)
    unknown = [r for r in records if r.get("cost") is None]
    charges = [
        r.get(
            "worst_case_usd",
            call_bound(
                packets[r["name"]],
                system,
                rates[r["model"]],
                r.get("output_token_ceiling", 4000),
            ),
        )
        for r in unknown
    ]
    return {
        "priced_usd": priced,
        "unpriced_calls": len(unknown),
        "unpriced_reserved_usd": sum(charges),
        "charged_usd": priced + sum(charges),
        "ceiling_usd": CEILING,
    }


async def read_corrected(*, pilot_only, through=0):
    """Use frozen packets and the audit call owner; reserve one call at a time."""
    from imas_codex.discovery.base import llm

    data = frozen_record()
    audit = audit_module()
    readers = (*audit.READERS, audit.third_reader())
    rates = spending_rates(audit)
    original_build = llm._build_kwargs

    def build_kwargs(*args, **kwargs):
        built = original_build(*args, **kwargs)
        model = args[0] if args else kwargs["model"]
        if model == audit.READERS[0]["model"]:
            built.get("extra_body", {}).pop("provider", None)
        return built

    llm._build_kwargs = build_kwargs
    # Keep the two readers whose price filters admitted calls on their known
    # bounds. The first reader's filter is removed by the narrow wrapper above.
    llm.get_openrouter_max_price = lambda model: rates[model]
    original_call = llm.acall_llm_structured
    ceilings = {"A": 4000, "B": 16000, "C": 4000}

    async def bounded_call(**kwargs):
        seat = next(r["seat"] for r in readers if r["model"] == kwargs["model"])
        kwargs["max_tokens"] = ceilings[seat]
        kwargs["output_token_ceiling"] = ceilings[seat]
        kwargs["max_retries"] = 1
        return await original_call(**kwargs)

    llm.acall_llm_structured = bounded_call
    audit.assert_readers_not_pipeline(readers)
    packets = {p["id"]: p for p in source_packets(data)}
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
            "spend_reservation_rates": rates,
            "output_token_ceilings": ceilings,
            "provider_filter_removed": ["A"],
            "max_attempts_per_call": 1,
            "max_inflight_calls": 1 if pilot_only else 6,
        },
    )
    records = recorded_calls()
    done = {(r["seat"], r["name"]) for r in records if not r.get("error")}
    assert len(done) == sum(not r.get("error") for r in records), (
        "Duplicate successful call"
    )
    if not pilot_only:
        pilot = load(RUN / "pilot.json")
        assert pilot["complete"] and pilot["projected_usd"] <= CEILING, (
            "Pilot projection exceeds ceiling"
        )
        assert pilot["names"] == order[:6]
    names = order[:6] if pilot_only else (order[:through] if through else order)
    gate = asyncio.Semaphore(1 if pilot_only else 6)
    condition = asyncio.Condition()
    held = 0.0
    active = 0
    stopped = None
    completed_names = {n for n in names if all((r["seat"], n) in done for r in readers)}

    async def one(reader, name):
        nonlocal held, active, stopped
        async with gate:
            worst = call_bound(
                packets[name], system, rates[reader["model"]], ceilings[reader["seat"]]
            )
            async with condition:
                while True:
                    if stopped:
                        return
                    account = budget_account(records, packets, system, rates)
                    if account["charged_usd"] + held + worst <= CEILING:
                        break
                    if not active:
                        stopped = f"Next call cannot fit: charged {account['charged_usd']:.6f} + bound {worst:.6f} > {CEILING}"
                        condition.notify_all()
                        return
                    await condition.wait()
                held += worst
                active += 1
                save(
                    RUN / "spend-status.json",
                    account
                    | {
                        "inflight_reserved_usd": held,
                        "active_calls": active,
                        "next_name": name,
                        "next_reader": reader["seat"],
                        "next_call_bound_usd": worst,
                    },
                )
            reply = await audit.judge_one(
                reader,
                packets[name],
                {"total": account["charged_usd"]},
                ceiling=CEILING,
            )
            reply.update(
                instruction_digest=digest(system),
                packet_digest=digest(packets[name]),
                recorded_at=datetime.now(UTC).isoformat(),
                output_token_ceiling=ceilings[reader["seat"]],
                worst_case_usd=worst,
            )
            async with condition:
                with (RUN / "corrected-readings.jsonl").open("a") as handle:
                    handle.write(json.dumps(reply, ensure_ascii=False) + "\n")
                records.append(reply)
                held -= worst
                active -= 1
                account = budget_account(records, packets, system, rates)
                save(
                    RUN / "spend-status.json",
                    account
                    | {"inflight_reserved_usd": max(0.0, held), "active_calls": active},
                )
                assert account["charged_usd"] + held <= CEILING + 1e-9
                if reply.get("error"):
                    stopped = (
                        "Reader call failed; null-cost receipt reserved at worst case: "
                        + reply["error"]
                    )
                else:
                    done.add((reader["seat"], name))
                    if name not in completed_names and all(
                        (r["seat"], name) in done for r in readers
                    ):
                        completed_names.add(name)
                        print(
                            f"{len(completed_names)}/{len(names)} names complete; priced USD {account['priced_usd']:.6f}; charged USD {account['charged_usd']:.6f} / {CEILING:.2f}",
                            flush=True,
                        )
                condition.notify_all()

    await asyncio.gather(
        *(
            one(reader, name)
            for name in names
            for reader in readers
            if (reader["seat"], name) not in done
        )
    )
    if stopped:
        raise RuntimeError(stopped)
    if pilot_only:
        good = [r for r in records if not r.get("error") and r["name"] in names]
        assert len(good) == 18
        paid = sum(r["cost"] for r in good)
        account = budget_account(records, packets, system, rates)
        projection = paid / 6 * len(order) + account["unpriced_reserved_usd"]
        receipt = {
            "names": names,
            "calls": len(good),
            "complete": True,
            "spend_usd": paid,
            "projected_usd": projection,
            "ceiling_usd": CEILING,
            "all_names": len(order),
            "method": "Successful six-name priced spend multiplied by cohort size / 6, plus worst-case charge for every null-cost failed call",
            "account": account,
            "per_reader_usd": {
                r["seat"]: sum(x["cost"] for x in good if x["seat"] == r["seat"])
                for r in readers
            },
        }
        save(RUN / "pilot.json", receipt)
        print("PILOT " + json.dumps(receipt), flush=True)
        assert projection <= CEILING, (
            "Pilot projection exceeds ceiling; stopped before full read"
        )


RULES = ("A", "B", "C", "two_of_three")


def rate(count, total):
    if not total:
        return {"count": count, "total": total, "rate": None, "ci_95": [0.0, 1.0]}
    interval = audit_module().wilson(count, total)
    return {
        "count": count,
        "total": total,
        "rate": count / total,
        "ci_95": list(interval),
    }


def flag(readings, name, rule, cls=None):
    if rule == "two_of_three":
        values = [flag(readings, name, seat, cls) for seat in ("A", "B", "C")]
        return None if None in values else sum(values) >= 2
    classes = readings.get(name, {}).get(rule)
    if classes is None:
        return None
    return bool(classes) if cls is None else cls in classes


def scores(names, judgments, readings, *, reference=None):
    out = {}
    for rule in RULES:
        ids = [
            n
            for n in names
            if flag(readings, n, rule) is not None
            and (reference is None or flag(reference, n, rule) is not None)
        ]
        defective = [n for n in ids if judgments[n]["verdict"] == "defective"]
        sound = [n for n in ids if judgments[n]["verdict"] == "sound"]
        recall = rate(sum(flag(readings, n, rule) for n in defective), len(defective))
        alarm = rate(sum(flag(readings, n, rule) for n in sound), len(sound))
        point = (
            recall["rate"] is not None
            and alarm["rate"] is not None
            and recall["rate"] >= 0.9
            and alarm["rate"] <= 0.1
        )
        precision = all(
            r["rate"] is not None and max(abs(e - r["rate"]) for e in r["ci_95"]) <= 0.1
            for r in (recall, alarm)
        )
        out[rule] = {
            "recall": recall,
            "false_alarm": alarm,
            "meets_point_thresholds": point,
            "meets_precision": precision,
            "fit_to_decide": point and precision,
        }
    return out


def earlier_readings(data):
    earlier = load(RUN / "earlier-readings.json")
    assert digest(earlier) == data["earlier_readings_digest"]
    out = {
        n: {seat: value["classes"] for seat, value in r.items()}
        for n, r in earlier.items()
    }
    third = load(ROOT / "frontier-third-reader.json")
    for kind in ("flagged", "control", "planted"):
        for row in third[kind]["rows"]:
            c = row["C"]
            if isinstance(c, dict):
                if row.get("read"):
                    out.setdefault(row["name"], {})["C"] = c["classes"]
            else:
                out.setdefault(row["name"], {})["C"] = c
    return out


def strata_for_sample(earlier, data):
    planted = {
        r["id"] for r in load(ROOT / "blind-audit.json")["planted_detail"]["rows"]
    }
    strata = {}
    for name, readings in earlier.items():
        if name in planted or not {"A", "B"} <= readings.keys():
            continue
        strata[name] = f"{int(bool(readings['A']))}{int(bool(readings['B']))}" + (
            str(int(bool(readings["C"]))) if "C" in readings else "u"
        )
    assert len(strata) == data["draw"]["frame_size"] == 339
    assert digest(sorted(strata)) == data["draw"]["frame_digest"]
    return strata


def verify_draw(data, earlier):
    strata = strata_for_sample(earlier, data)
    pools = defaultdict(list)
    for name in sorted(strata):
        pools[strata[name]].append(name)
    rng = random.Random(SEED)
    for key in sorted(pools):
        rng.shuffle(pools[key])
    order = []
    while any(pools.values()):
        for key in sorted(pools):
            if pools[key]:
                order.append(pools[key].pop())
    assert digest(order) == data["draw"]["order_digest"], "Seeded draw changed"
    assert (
        order[: len(data["draw"]["candidate_order"])] == data["draw"]["candidate_order"]
    )
    judgments = {j["id"]: j for j in data["judgments"]}
    counts = Counter(judgments[n]["verdict"] for n in data["standing_ids"])
    expected = list(data["standing_ids"])
    for name in order:
        if name not in judgments:
            assert name in data["blindness"]["exposure_exclusions"], (
                "Draw prefix skipped an unexposed case"
            )
            continue
        j = judgments[name]
        if j["verdict"] in ("defective", "sound"):
            expected.append(name)
            counts[j["verdict"]] += 1
        if counts["defective"] >= TARGET and counts["sound"] >= TARGET:
            break
    assert expected == data["cohort"], (
        "Cohort differs from the first complete label prefix"
    )


def class_scores(names, judgments, earlier, corrected):
    out = {}
    for key in audit_module().CLASS_KEYS:
        positive = [n for n in names if key in judgments[n]["defect_classes"]]
        out[key] = {}
        for rule in RULES:
            paired = [n for n in positive if flag(earlier, n, rule) is not None]
            out[key][rule] = {
                "corrected_all": rate(
                    sum(flag(corrected, n, rule, key) for n in positive), len(positive)
                ),
                "earlier_paired": rate(
                    sum(flag(earlier, n, rule, key) for n in paired), len(paired)
                ),
                "corrected_paired": rate(
                    sum(flag(corrected, n, rule, key) for n in paired), len(paired)
                ),
            }
    return out


def catalog_estimate(data, judgments, earlier, corrected, chosen, replicates=10000):
    """Project the corrected flag share over the original joint flag strata.

    A joint name bootstrap preserves dependence between flags and hand labels
    on the newly drawn cases. The inherited standing plants are resampled
    separately within their two hand-label classes. Intervals are conditional
    on transport of reader error rates and on the source-only adjudications.
    """
    strata = strata_for_sample(earlier, data)
    frames = Counter(strata.values())
    groups = defaultdict(list)
    for n in data["cohort"]:
        if n in strata:
            groups[strata[n]].append(n)
    assert set(groups) == set(frames), (
        "A historical flag stratum has no corrected reading"
    )
    inherited = {
        v: [n for n in data["standing_ids"] if judgments[n]["verdict"] == v]
        for v in ("defective", "sound")
    }

    def estimate(group_names, inherited_names):
        q = sum(
            frames[h]
            / 339
            * sum(flag(corrected, n, chosen) for n in group_names[h])
            / len(group_names[h])
            for h in frames
        )
        names = [n for ns in group_names.values() for n in ns] + inherited_names
        defective = [n for n in names if judgments[n]["verdict"] == "defective"]
        sound = [n for n in names if judgments[n]["verdict"] == "sound"]
        r = sum(flag(corrected, n, chosen) for n in defective) / len(defective)
        f = sum(flag(corrected, n, chosen) for n in sound) / len(sound)
        if r <= f:
            return q, r, f, None
        return q, r, f, (q - f) / (r - f)

    q, r, f, p = estimate(groups, data["standing_ids"])
    assert p is not None, "Selected reader has no positive discrimination"
    rng = random.Random(SEED + 2)
    bootstrap = []
    invalid = 0
    for _ in range(replicates):
        group_names = {h: rng.choices(ns, k=len(ns)) for h, ns in groups.items()}
        inherited_names = [
            n for ns in inherited.values() for n in rng.choices(ns, k=len(ns))
        ]
        value = estimate(group_names, inherited_names)[3]
        if value is None:
            invalid += 1
        else:
            bootstrap.append(max(0.0, min(1.0, value)))
    bootstrap.sort()
    assert invalid / replicates < 0.025, (
        "Correction unidentified in too many bootstrap draws"
    )
    ci = [
        bootstrap[int(0.025 * (len(bootstrap) - 1))],
        bootstrap[int(0.975 * (len(bootstrap) - 1))],
    ]
    return {
        "reader": chosen,
        "sample_names": 339,
        "corrected_sample_readings": sum(map(len, groups.values())),
        "joint_strata": {
            h: {
                "frame": frames[h],
                "hand_checked_and_reread": len(groups[h]),
                "corrected_flagged": sum(flag(corrected, n, chosen) for n in groups[h]),
            }
            for h in sorted(frames)
        },
        "weighted_corrected_flag_share": q,
        "recall": r,
        "false_alarm": f,
        "formula": "p = (q - f) / (r - f), from q = r*p + f*(1-p)",
        "raw_corrected_rate": p,
        "corrected_rate": max(0.0, min(1.0, p)),
        "ci_95": ci,
        "implied_defective_names": max(0.0, min(1.0, p)) * 339,
        "implied_count_ci_95": [v * 339 for v in ci],
        "bootstrap": {
            "replicates": replicates,
            "seed": SEED + 2,
            "invalid_discrimination": invalid,
            "method": "Percentile joint case bootstrap within historical reader strata; inherited standing plants resampled within hand-label class; rates clipped to [0,1]",
        },
        "limits": [
            "Projected corrected-instruction flag share, not a full corrected rereading of all 339 names.",
            "Assumes reader recall and false alarms transport from this selected calibration cohort to the catalog.",
            "The interval conditions on the selected reader and fixed hand labels; it does not include reader-selection uncertainty or mistakes in manual adjudication.",
            "Equal allocation across earlier-reader strata, sequential stopping at forty sound labels, undecidable exclusions, accidental-exposure exclusions and inherited plants can produce spectrum and selection bias; the interval does not remove these biases.",
            "The original 339-name sample was itself stratified by physics domain; this estimates that sample, not a population-weighted census rate for the entire live catalog.",
            "Old audit packets were not retained; changes of stored text between earlier calls and these versioned packets cannot be separated from the instruction change.",
        ],
    }


def derived_report(data):
    for filename, wanted in data["input_digests"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == wanted, (
            "Earlier evidence changed: " + filename
        )
    attempts = recorded_calls()
    records = [r for r in attempts if not r.get("error")]
    assert len(records) == len(data["cohort"]) * 3
    corrected = defaultdict(dict)
    for r in records:
        assert (
            not r.get("error")
            and r["cost"] is not None
            and math.isfinite(r["cost"])
            and r["cost"] >= 0
        )
        assert r["name"] in data["cohort"] and r["seat"] in ("A", "B", "C")
        assert r["seat"] not in corrected[r["name"]], "Duplicate reader receipt"
        corrected[r["name"]][r["seat"]] = r["defect_classes"]
    assert all(set(corrected[n]) == {"A", "B", "C"} for n in data["cohort"])
    earlier = earlier_readings(data)
    verify_draw(data, earlier)
    judgments = {j["id"]: j for j in data["judgments"]}
    names = data["cohort"]
    after = scores(names, judgments, corrected)
    before_paired = scores(names, judgments, earlier, reference=corrected)
    after_paired = scores(names, judgments, corrected, reference=earlier)
    individual = [s for s in ("A", "B", "C") if after[s]["fit_to_decide"]] or [
        "A",
        "B",
        "C",
    ]
    chosen = max(
        individual,
        key=lambda s: (
            (after[s]["recall"]["rate"] + 1 - after[s]["false_alarm"]["rate"]) / 2,
            after[s]["recall"]["rate"],
            s,
        ),
    )
    spent = sum(r["cost"] for r in records)
    account = budget_account(
        attempts,
        {p["id"]: p for p in source_packets(data)},
        audit_module().system_prompt(),
        spending_rates(audit_module()),
    )
    assert account["charged_usd"] <= CEILING
    prompt = load(RUN / "corrected-prompt.json")
    assert prompt["system"] == audit_module().system_prompt()
    assert all(r["instruction_digest"] == digest(prompt["system"]) for r in records)
    packets = {p["id"]: p for p in source_packets(data)}
    assert all(r["packet_digest"] == digest(packets[r["name"]]) for r in records)
    pilot = load(RUN / "pilot.json")
    assert (
        pilot["complete"] and pilot["calls"] == 18 and pilot["projected_usd"] <= CEILING
    )
    pilot_records = [r for r in records if r["name"] in pilot["names"]]
    assert abs(sum(r["cost"] for r in pilot_records) - pilot["spend_usd"]) < 1e-9
    assert max(r["recorded_at"] for r in pilot_records) < min(
        r["recorded_at"] for r in records if r["name"] not in pilot["names"]
    )
    return {
        "blind_commit": BLIND_COMMIT,
        "instruction_before": BEFORE,
        "instruction_after": AFTER,
        "corrected": after,
        "earlier_paired": before_paired,
        "corrected_paired": after_paired,
        "per_class": class_scores(names, judgments, earlier, corrected),
        "chosen_reader": chosen,
        "chosen_reader_fit": after[chosen]["fit_to_decide"],
        "selection_rule": "Among fit individual readers if any, otherwise among all three: highest balanced accuracy, then higher recall, then seat id; majority rule reported separately.",
        "catalog_estimate": catalog_estimate(
            data, judgments, earlier, corrected, chosen
        ),
        "spend": {
            "recorded_usd": spent,
            "ceiling_usd": CEILING,
            "calls": len(records),
            "unknown_cost_calls": account["unpriced_calls"],
            "account": account,
            "attempts": len(attempts),
            "by_reader": {
                s: sum(r["cost"] for r in records if r["seat"] == s)
                for s in ("A", "B", "C")
            },
            "pilot": pilot,
        },
        "receipts": {
            "path": str(RUN / "corrected-readings.jsonl"),
            "sha256": hashlib.sha256(
                (RUN / "corrected-readings.jsonl").read_bytes()
            ).hexdigest(),
            "prompt_path": str(RUN / "corrected-prompt.json"),
            "prompt_digest": digest(prompt),
        },
        "interval": "Two-sided 95% Wilson score intervals (z=1.96); each endpoint must be within ten percentage points of the point estimate for fitness. Percentile bootstrap for corrected catalog estimate.",
        "paired_note": "Earlier and corrected rates use exactly the same observed names separately for each reader/rule. C and majority have fewer paired names because C was not called on every earlier sound stratum. Earlier readings remain under contradictory instructions; field drift prevents causal attribution solely to the prompt.",
    }


def rate_text(r):
    if not r["total"]:
        return "0/0; not estimable"
    low, high = r["ci_95"]
    return f"{r['count']}/{r['total']} = {100 * r['rate']:.1f}% (95% {100 * low:.1f}–{100 * high:.1f}%)"


def score_table(values, *, fitness=False):
    head = "<table><thead><tr><th>Reader / rule</th><th>Recall</th><th>False alarm</th>"
    if fitness:
        head += "<th>Point thresholds</th><th>Precision</th><th>Fit</th>"
    lines = [head + "</tr></thead><tbody>"]
    for seat, value in values.items():
        line = (
            "<tr><th>"
            + ("Two of three" if seat == "two_of_three" else seat)
            + "</th><td>"
            + rate_text(value["recall"])
            + "</td><td>"
            + rate_text(value["false_alarm"])
            + "</td>"
        )
        if fitness:
            line += "".join(
                "<td>" + ("yes" if value[k] else "no") + "</td>"
                for k in ("meets_point_thresholds", "meets_precision", "fit_to_decide")
            )
        lines.append(line + "</tr>")
    return "\n".join(lines + ["</tbody></table>"])


def render_page(data):
    result = data["results"]
    estimate = result["catalog_estimate"]
    chosen = result["chosen_reader"]
    fits = [s for s, v in result["corrected"].items() if v["fit_to_decide"]]
    verdict = (
        "Fit to decide: " + ", ".join(fits) + "."
        if fits
        else "No reader or two-of-three rule meets the required fitness bar."
    )
    models = {
        r["seat"]: r["model"] for r in load(RUN / "corrected-prompt.json")["readers"]
    }
    account = result["spend"]["account"]
    pilot = result["spend"]["pilot"]
    judgments = {j["id"]: j for j in data["judgments"]}
    earlier = earlier_readings(data)
    records = [r for r in recorded_calls() if not r.get("error")]
    corrected = defaultdict(dict)
    for r in records:
        corrected[r["name"]][r["seat"]] = r["defect_classes"]
    lines = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<meta name="docs-project" content="imas-codex">',
        '<meta name="reckon-type" content="evidence-fragment">',
        '<meta name="plan-slug" content="sn-audited-group-release">',
        "<title>Audit readers calibrated against source-only labels</title>",
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #ccc;padding:.5rem}code{overflow-wrap:anywhere;font-size:.85em}.scroll{overflow-x:auto}blockquote{margin:.5rem 0;padding-left:.75rem;border-left:2px solid #929292}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style></head><body><main>",
        '<h1 id="reader-calibration-title">Audit readers calibrated against source-only labels</h1>',
        '<p id="reader-calibration-summary"><strong>'
        + verdict
        + "</strong> The corrected instructions were read by all three readers on 84 hand-checked names: 44 defective and 40 sound. All 26 labels that stood in the first check are included. Three additional adjudications are undecidable and excluded.</p>",
        "<p>Reader "
        + chosen
        + " is the "
        + ("fit" if result["chosen_reader_fit"] else "best measured individual")
        + " reader by balanced accuracy, with recall used to break ties. This measurement does not itself authorize a group release or choose an interim deciding rule where none is fit.</p>",
        "<h2>Spend and the six-name pilot</h2>",
        f"<p>Recorded priced spend: <strong>USD {account['priced_usd']:.6f}</strong>; the {account['unpriced_calls']} null-cost failed calls are charged <strong>USD {account['unpriced_reserved_usd']:.6f}</strong> at its worst-case bound. Budget-accounted total: <strong>USD {account['charged_usd']:.6f} / USD {CEILING:.2f}</strong>. There are {len(records)} successful readings and {result['spend']['attempts'] - len(records)} retained failed attempt(s). No null receipt is counted as free.</p>",
        f"<p>The seeded six-name pilot completed all 18 successful readings for USD {pilot['spend_usd']:.6f}. Scaling that cost to 84 names and adding the failed-call reserve projected <strong>USD {pilot['projected_usd']:.6f} / USD {CEILING:.2f}</strong>; the full read began afterward. Two readings paid before resumption were retained. The initial local reservation refusal made no calls; the next attempt returned reader A’s price-filter refusal while B and C returned paid readings. After adjudication, A’s provider max-price filter was removed and every next call was checked against the running charged total plus its worst-case token cost, including the reserved bounds of any other calls in flight.</p>",
        "<p>Calls use the original audit’s <code>judge_one</code> and shared structured-call owner. On resumption, output ceilings are A 4000 and C 4000 tokens; B used 8000 initially and 16000 after an empty length-limited response. Each resumed call has one provider attempt. The inherited B receipt includes its earlier length retry. Reader A is reserved at conservative text rates of USD 20 / 100 per million input/output tokens; the other readers use the recorded bounded text prices. These budget controls and any text drift limit causal attribution solely to the omission sentence.</p>",
        "<h2>The omission instruction: before and after</h2>",
        "<p>The unchanged class definition says that a qualifier stated by the source and absent from the name is a defect. The later rule contradicted it:</p>",
        "<table><thead><tr><th>Earlier instruction, retained with earlier readings</th><th>Corrected instruction</th></tr></thead><tbody><tr><td>"
        + escape(BEFORE)
        + "</td><td>"
        + escape(AFTER)
        + "</td></tr></tbody></table>",
        "<h2>Corrected instructions: entire checked set</h2>",
        "<p>Recall is the share of hand-defective names with any reader flag; a matching defect class is not required for this headline measure. False alarm is the share of hand-sound names flagged. Two-of-three is scored directly, without an independence assumption. The point thresholds are recall ≥90% and false alarm ≤10%. Precision requires each endpoint of each two-sided 95% Wilson interval to be within ten percentage points of its point estimate.</p>",
        score_table(result["corrected"], fitness=True),
        "<h2>Earlier instructions and corrected instructions on the same names</h2>",
        "<p>" + escape(result["paired_note"]) + "</p>",
        "<h3>Earlier instructions</h3>",
        score_table(result["earlier_paired"]),
        "<h3>Corrected instructions, paired subset only</h3>",
        score_table(result["corrected_paired"]),
        "<h2>Recall by defect class</h2>",
        "<p>Here a detection requires the reader to name the same class as the hand judgment. The majority rule requires at least two readers to name that class. A multi-defect name appears in every applicable row. Classes with no hand-positive case are not estimable.</p>",
        '<div class="scroll"><table><thead><tr><th>Class</th><th>Reader / rule</th><th>Corrected, entire set</th><th>Earlier, paired</th><th>Corrected, same paired names</th></tr></thead><tbody>',
    ]
    for cls, seats in result["per_class"].items():
        for seat, values in seats.items():
            lines.append(
                "<tr><td><code>"
                + cls
                + "</code></td><td>"
                + seat
                + "</td>"
                + "".join(
                    "<td>" + rate_text(values[k]) + "</td>"
                    for k in ("corrected_all", "earlier_paired", "corrected_paired")
                )
                + "</tr>"
            )
    lines += [
        "</tbody></table></div>",
        "<h2>What reader "
        + chosen
        + " implies for the original 339 sampled names</h2>",
        "<p>The 26 inherited plants are outside the original sample. The other 58 checked names were drawn across the joint historical flag strata; the corrected flag rate in each stratum is weighted by that stratum’s original share of the 339-name sample. This is a projection from those corrected rereadings, not a new reading of every one of the 339 names.</p>",
        "<table><thead><tr><th>Historical flags A/B/C (u = C unread)</th><th>Names in 339</th><th>Hand-checked and reread</th><th>Corrected "
        + chosen
        + " flags</th></tr></thead><tbody>",
    ]
    for h, v in estimate["joint_strata"].items():
        lines.append(
            f"<tr><td>{h}</td><td>{v['frame']}</td><td>{v['hand_checked_and_reread']}</td><td>{v['corrected_flagged']}</td></tr>"
        )
    low, high = estimate["ci_95"]
    lines += [
        "</tbody></table>",
        f"<p>The weighted corrected-instruction flag share is q = {estimate['weighted_corrected_flag_share']:.6f}. Measured recall is r = {estimate['recall']:.6f}; measured false-alarm rate is f = {estimate['false_alarm']:.6f}. From <code>q = r·p + f·(1−p)</code>, the correction is <code>p = (q−f)/(r−f)</code> = ({estimate['weighted_corrected_flag_share']:.6f} − {estimate['false_alarm']:.6f}) / ({estimate['recall']:.6f} − {estimate['false_alarm']:.6f}) = <strong>{100 * estimate['corrected_rate']:.1f}%</strong>, with a conditional 95% interval of <strong>{100 * low:.1f}–{100 * high:.1f}%</strong>. That implies {estimate['implied_defective_names']:.1f} of 339 sampled names (interval {estimate['implied_count_ci_95'][0]:.1f}–{estimate['implied_count_ci_95'][1]:.1f}). The unclipped corrected rate is {estimate['raw_corrected_rate']:.6f}; physical rates and bootstrap draws are clipped to [0,1].</p>",
        "<p>The interval uses 10,000 seeded joint case bootstrap replicates. Cases are resampled within historical flag strata, preserving dependence between hand labels and reader flags; inherited standing plants are resampled within their hand-label class. It propagates the measured flag share and calibration error together.</p>",
        "<ul>",
    ]
    lines += ["<li>" + escape(t) + "</li>" for t in estimate["limits"]]
    lines += [
        "</ul>",
        "<h2>Blindness, selection and source provenance</h2>",
        "<p>All judgments were committed before corrected reader calls at <code>"
        + BLIND_COMMIT
        + "</code>. New candidates were selected by seeded round-robin sampling across the joint A/B/C flag strata, not by one reader. Every decidable judgment in the prefix ending when both classes reached forty was retained; the cohort was not trimmed to exactly forty defects. The hand adjudicator saw source-only packets. Seven identities exposed accidentally to historical rationales by an earlier search were excluded before labeling. The inherited 26 standing labels had already been compared with earlier readers in the first check; they were not independently reblinded here.</p>",
        '<p>The graph collection returned 100/100 requested new names and recovered all 148/148 expected producer/name memberships, a positive control for the source join. It wrote no graph properties and made no model calls. Source packets retain the stored name, definition, documentation, unit, domain, source paths, source leaf text, enclosing text and supplied duplicate candidates. Repeated lines are stored through a lossless text dictionary in <a href="/imas-codex/evidence/fragments/sn-audited-group-release/reader-calibration.json">the committed data file</a>. The full 126 collected packets, including unused candidates, remain in the run directory with their digest in that data file.</p>',
        "<p>Hand labels mean what the displayed text establishes under the fixed classes, not an independent experimental validation or a new physicist sign-off. A sound case means no fixed defect is established from the shown packet; missing source text limits which classes can be judged. Packet rendering inherits the original audit’s clipping limits. The earlier audit did not retain exact per-name source packets, so the earlier/corrected comparison is not an exact historical replay.</p>",
        "<h2>Every source judgment and its reader flags</h2>",
        "<p>A reader flag is shown as the exact class list; “none” means a successful empty list and “not read” means no earlier observation. Rationale and token/cost receipts are retained in the run’s <code>corrected-readings.jsonl</code>, whose digest is checked by the script.</p>",
        '<div class="scroll"><table><thead><tr><th>Name and hand judgment</th><th>Source evidence and reason</th><th>Earlier A / B / C</th><th>Corrected A / B / C</th></tr></thead><tbody>',
    ]
    for n in data["cohort"]:
        j = judgments[n]
        cells = []
        for mapping in (earlier, corrected):
            cells.append(
                "<td>"
                + "<br>".join(
                    escape(
                        s + ": " + (", ".join(mapping.get(n, {}).get(s, [])) or "none")
                        if s in mapping.get(n, {})
                        else s + ": not read"
                    )
                    for s in ("A", "B", "C")
                )
                + "</td>"
            )
        lines.append(
            "<tr><td><code>"
            + escape(n)
            + "</code><p>"
            + j["verdict"]
            + "</p><code>"
            + escape(", ".join(j["defect_classes"]) or "none")
            + "</code></td><td><code>"
            + escape(j["evidence_path"])
            + "</code><blockquote>"
            + escape(j["evidence_quote"])
            + "</blockquote><p>"
            + escape(j["reason"])
            + "</p></td>"
            + "".join(cells)
            + "</tr>"
        )
    lines += [
        "</tbody></table></div>",
        "<h2>Undecidable judgments</h2><table><thead><tr><th>Name</th><th>Reason</th></tr></thead><tbody>",
    ]
    lines += [
        "<tr><td><code>"
        + escape(j["id"])
        + "</code></td><td>"
        + escape(j["reason"])
        + "</td></tr>"
        for j in data["judgments"]
        if j["verdict"] == "undecidable"
    ]
    lines += [
        "</tbody></table>",
        "<h2>Reader identities and reproducibility</h2><table><thead><tr><th>Reader</th><th>Recorded model</th></tr></thead><tbody>",
    ]
    lines += [
        "<tr><td>" + s + "</td><td><code>" + escape(m) + "</code></td></tr>"
        for s, m in models.items()
    ]
    lines += [
        "</tbody></table>",
        "<p>The companion script’s explicit <code>--check</code> verifies the blind commit, exact frozen fields, packet and input digests, successful reader coverage, call receipt digests, pilot ordering, spend including null-cost reservations, recomputed counts and intervals, correction arithmetic, and this rendered page. The negative control alters a frozen judgment after the blind commit; the check must refuse before comparing readers.</p>",
        "</main></body></html>",
    ]
    return "\n".join(lines) + "\n"


def report():
    frozen = frozen_record()
    result = derived_report(frozen)
    data = frozen | {"phase": "compared", "results": result}
    save(DATA, data)
    assert DATA.stat().st_size <= 300_000, "Data file exceeds repository size ceiling"
    PAGE.write_text(render_page(data))
    print(
        "REPORT "
        + json.dumps(
            {
                "corrected": result["corrected"],
                "chosen_reader": result["chosen_reader"],
                "catalog_estimate": result["catalog_estimate"],
                "spend": result["spend"],
            }
        ),
        flush=True,
    )


def check(path):
    frozen = frozen_record(path)
    data = load(path)
    assert data["phase"] == "compared", "Reader comparison not completed"
    assert data["results"] == derived_report(frozen), "Recomputed reader results differ"
    assert PAGE.read_text() == render_page(data), "Rendered page differs"
    assert path.stat().st_size <= 300_000
    # Positive controls prove the arithmetic sees a defect, a sound flag and
    # correlated votes; an empty class list must never stand in for a failure.
    j = {"bad": {"verdict": "defective"}, "good": {"verdict": "sound"}}
    r = {
        "bad": {"A": ["unit_wrong"], "B": ["unit_wrong"], "C": []},
        "good": {"A": [], "B": ["unit_wrong"], "C": []},
    }
    values = scores(["bad", "good"], j, r)
    assert (
        values["A"]["recall"]["count"] == 1 and values["A"]["false_alarm"]["count"] == 0
    )
    assert values["B"]["false_alarm"]["count"] == 1
    assert (
        values["two_of_three"]["recall"]["count"] == 1
        and values["two_of_three"]["false_alarm"]["count"] == 0
    )
    assert flag({"bad": {"A": []}}, "bad", "C") is None
    print(
        "PASS: frozen judgments, 252 corrected readings, paired comparisons, class recall, spend, corrected estimate and HTML verified",
        flush=True,
    )
    for seat, v in data["results"]["corrected"].items():
        print(
            seat,
            "recall",
            rate_text(v["recall"]),
            "false alarm",
            rate_text(v["false_alarm"]),
            "fit",
            v["fit_to_decide"],
        )


if __name__ == "__main__":
    main()

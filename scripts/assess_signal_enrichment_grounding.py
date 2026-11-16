"""Build blinded context-reading sheets and verify sealed manual assessments.

No automated verdict enters selection. The private key is opened only after
all readings have been written and their digest has been sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import benchmark_signal_enrichment as bench
from scripts.refine_signal_enrichment import supplied_target_context


def sheets(root: Path, output: Path, arms: list[str], sample: str) -> None:
    cohort = json.loads((root / "cohort.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    ids = (
        selection["screen"]
        if sample == "screen"
        else selection["first"] + selection["second"]
    )
    records = bench.load_records(root)
    contexts = {
        id: row["messages"][0]
        for row in records
        if row["arm"] == "context"
        for id in row["ids"]
    }
    outputs = {
        arm: {
            row["id"]: row
            for batch in records
            if batch["arm"] == arm
            for row in batch["outputs"]
        }
        for arm in [*arms, "paid"]
    }
    refs = {row["id"]: row for row in cohort["reference"]}
    output.mkdir(parents=True, exist_ok=True)
    key, manifest = {}, []
    for arm in arms:
        packet = hashlib.sha256(
            ("blind-context-packet:" + arm + sample).encode()
        ).hexdigest()[:12]
        rows = []
        for index, id in enumerate(ids):
            first = int(hashlib.sha256((packet + id).encode()).hexdigest(), 16) % 2 == 0
            local = outputs[arm].get(id, {}).get("description", "")
            paid = outputs["paid"][id]["description"]
            signal = bench.input_signal(refs[id])
            rows.append(
                {
                    "index": index,
                    "id": id,
                    "signal": signal,
                    "context": supplied_target_context(
                        contexts[id], signal["accessor"]
                    ),
                    "first": local if first else paid,
                    "second": paid if first else local,
                }
            )
            key[f"{packet}:{index}"] = {
                "arm": arm,
                "local": "first" if first else "second",
                "id": id,
            }
        bench.save(output / f"{packet}.json", rows)
        manifest.append(
            {"packet": packet, "rows": len(rows), "digest": bench.digest(rows)}
        )
    bench.save(output / "key.json", key)
    bench.save(output / "sheets.json", manifest)


def seal(output: Path) -> None:
    rows = [
        json.loads(line)
        for line in (output / "readings.jsonl").read_text().splitlines()
    ]
    sheets = json.loads((output / "sheets.json").read_text())
    expected = {
        (row["packet"], index) for row in sheets for index in range(row["rows"])
    }
    observed = {(row["packet"], row["index"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("manual reading must cover each blind pair exactly once")
    for row in rows:
        if (
            row["preference"] not in {"first", "second", "equivalent"}
            or not row["reason"].strip()
        ):
            raise ValueError("each reading requires a blind preference and a reason")
        if set(row["quality"]) != {"first", "second"} or not set(
            row["quality"].values()
        ) <= {"wrong", "partly", "right"}:
            raise ValueError("each description requires wrong, partly or right")
    bench.save(
        output / "seal.json",
        {
            "readings": len(rows),
            "sha256": hashlib.sha256(
                (output / "readings.jsonl").read_bytes()
            ).hexdigest(),
            "method": "Manual context reading, blind to arm; sealed before key and before new automated verdicts.",
        },
    )


def summarize(output: Path) -> dict:
    sealed = json.loads((output / "seal.json").read_text())
    if (
        hashlib.sha256((output / "readings.jsonl").read_bytes()).hexdigest()
        != sealed["sha256"]
    ):
        raise ValueError("manual readings changed after sealing")
    key = json.loads((output / "key.json").read_text())
    result = {}
    for row in [
        json.loads(line)
        for line in (output / "readings.jsonl").read_text().splitlines()
    ]:
        identity = key[f"{row['packet']}:{row['index']}"]
        group = result.setdefault(identity["arm"], [])
        local = identity["local"]
        paid = "second" if local == "first" else "first"
        group.append(
            {
                "id": identity["id"],
                "quality": row["quality"][local],
                "paid_quality": row["quality"][paid],
                "preference": "equivalent"
                if row["preference"] == "equivalent"
                else "local"
                if row["preference"] == local
                else "paid",
                "reason": row["reason"],
            }
        )
    bench.save(output / "unblinded.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sheets", "seal", "summarize"])
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--sample", choices=["screen", "all"], default="screen")
    args = parser.parse_args()
    if args.action == "sheets":
        sheets(args.store, args.output, args.arms, args.sample)
    elif args.action == "seal":
        seal(args.output)
    else:
        result = summarize(args.output)
        from collections import Counter

        print(
            json.dumps(
                {
                    arm: dict(Counter(row["quality"] for row in rows))
                    for arm, rows in result.items()
                }
            )
        )


if __name__ == "__main__":
    main()

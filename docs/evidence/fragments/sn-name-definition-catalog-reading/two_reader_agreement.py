"""Derive the two-reader agreement over every published name read blind.

Reads the three label sets (the first sample, the fresh sample and the
catalog sheets), applies one rule to each name, and writes the agreement
beside the seat's flag for the same name.

Rule: a name both readers give any class other than ``agrees`` is mismatched;
a name both give ``agrees`` is agreeing; any other name is disputed.
"""

import json
import sys
from pathlib import Path

FRAGMENTS = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEAT = (
    FRAGMENTS / "sn-definitions-by-construction" / "name-definition-seat-catalog.json"
)
FIRST = FRAGMENTS / "sn-name-definition-second-reading" / "two-reader-labels.json"
FRESH = FRAGMENTS / "sn-name-definition-fresh-labels"


def verdict(class_a: str, class_b: str) -> str:
    flag_a, flag_b = class_a != "agrees", class_b != "agrees"
    if flag_a and flag_b:
        return "mismatched"
    if not flag_a and not flag_b:
        return "agreeing"
    return "disputed"


def sheet_rows(directory: Path, parts: list[str], source: str) -> list[dict]:
    rows = []
    for part in parts:
        sheet = json.loads((directory / f"sheet-part-{part}.json").read_text())
        a = json.loads((directory / f"labels-a-part-{part}.json").read_text())
        b = json.loads((directory / f"labels-b-part-{part}.json").read_text())
        for item, row_a, row_b in zip(sheet["items"], a, b, strict=True):
            if not item["id"] == row_a["id"] == row_b["id"]:
                raise SystemExit(
                    f"{source} part {part}: ids out of step at {item['id']}"
                )
            rows.append(
                {
                    "name": item["name"],
                    "source": source,
                    "reader_a": row_a["class"],
                    "reader_b": row_b["class"],
                    "evidence_a": row_a.get("evidence", ""),
                    "evidence_b": row_b.get("evidence", ""),
                }
            )
    return rows


def first_sample_rows() -> list[dict]:
    rows = json.loads(FIRST.read_text())["rows"]
    return [
        {
            "name": row["name"],
            "source": "first-sample",
            "reader_a": row["first"],
            "reader_b": row["second"],
            "evidence_a": "",
            "evidence_b": "",
        }
        for row in rows
    ]


def main() -> int:
    seat = {f["name"]: f for f in json.loads(SEAT.read_text())["findings"]}
    rows = first_sample_rows()
    rows += sheet_rows(FRESH, ["1", "2", "3"], "fresh-sample")
    rows += sheet_rows(HERE, [f"{n:02d}" for n in range(1, 13)], "catalog-sheets")
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)):
        raise SystemExit("a name was read in more than one label set")
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        row["verdict"] = verdict(row["reader_a"], row["reader_b"])
        row["seat_flagged"] = row["name"] in seat
        row["seat_qualifier"] = seat.get(row["name"], {}).get("qualifier", "")
        bucket = totals.setdefault(row["source"], {})
        key = f"{row['verdict']}|seat-{'flagged' if row['seat_flagged'] else 'passed'}"
        bucket[key] = bucket.get(key, 0) + 1
    overall: dict[str, int] = {}
    for bucket in totals.values():
        for key, count in bucket.items():
            overall[key] = overall.get(key, 0) + count
    out = {
        "schema": "two-reader-agreement",
        "rule": "both readers give a class other than agrees: mismatched; both give agrees: agreeing; otherwise disputed",
        "names": len(rows),
        "by_source": totals,
        "overall": overall,
        "seat_flags_not_read_here": sorted(set(seat) - set(names)),
        "rows": sorted(rows, key=lambda r: r["name"]),
    }
    (HERE / "two-reader-agreement.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("names", "by_source", "overall")}, indent=1))
    print("seat flags on names not read here:", len(out["seat_flags_not_read_here"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

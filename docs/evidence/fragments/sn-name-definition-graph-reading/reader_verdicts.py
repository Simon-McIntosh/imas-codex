"""Settle each export candidate from the blind readings it has received.

Rule, per candidate name:

* label carried from the first two readers and both flag: ``flagged``;
* label carried and both pass, with no further reading: ``stands``;
* label carried with a further reading (the two disagreed, or the chat seat
  flagged a name both passed): ``flagged`` when two of the three readers flag,
  otherwise ``stands``;
* text changed since it was read, or never read: two fresh readings decide;
  both flag is ``flagged``, both pass is ``stands``, a split is ``undecided``
  until a tie-break reading exists, after which two of three decides.

A flag is any class other than ``agrees``.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIE_BREAK_PART = "07"


def further_readings() -> tuple[dict[str, list[dict]], dict[str, dict]]:
    readings: dict[str, list[dict]] = {}
    items: dict[str, dict] = {}
    for sheet_path in sorted(HERE.glob("sheet-part-*.json")):
        part = sheet_path.stem.rsplit("-", 1)[1]
        sheet = json.loads(sheet_path.read_text())
        for item in sheet["items"]:
            items.setdefault(item["name"], item)
        for label_path in sorted(HERE.glob(f"labels-*-part-{part}.json")):
            reader = label_path.stem.split("-")[1]
            labels = json.loads(label_path.read_text())
            for item, row in zip(sheet["items"], labels, strict=True):
                if item["id"] != row["id"]:
                    raise SystemExit(
                        f"{label_path.name}: ids out of step at {item['id']}"
                    )
                readings.setdefault(item["name"], []).append(
                    {
                        "reader": reader,
                        "class": row["class"],
                        "evidence": row.get("evidence", ""),
                    }
                )
    return readings, items


def settle() -> dict:
    reconciliation = json.loads((HERE / "reconciliation.json").read_text())
    candidates = reconciliation["candidates"]
    readings, items = further_readings()
    rows = []
    for name, carried in candidates["label-carried"].items():
        extra = readings.get(name, [])
        flags = [carried["a"] != "agrees", carried["b"] != "agrees"]
        flags += [r["class"] != "agrees" for r in extra]
        classes = [carried["a"], carried["b"]] + [r["class"] for r in extra]
        if carried["v"] == "mismatched":
            verdict = "flagged"
        elif carried["v"] == "agreeing" and not extra:
            verdict = "stands"
        elif not extra:
            verdict = "undecided"
        else:
            verdict = "flagged" if sum(flags) >= 2 else "stands"
        rows.append(
            {
                "name": name,
                "status": "label-carried",
                "carried_verdict": carried["v"],
                "classes": classes,
                "seat_flagged": bool(carried.get("f")),
                "verdict": verdict,
                "evidence": [r["evidence"] for r in extra if r["class"] != "agrees"],
            }
        )
    for status in ("text-changed", "never-read"):
        for name in candidates[status]:
            fresh = readings.get(name, [])
            flags = [r["class"] != "agrees" for r in fresh]
            if len(fresh) < 2:
                verdict = "undecided"
            elif len(fresh) == 2 and sum(flags) == 1:
                verdict = "undecided"
            else:
                verdict = "flagged" if sum(flags) >= 2 else "stands"
            rows.append(
                {
                    "name": name,
                    "status": status,
                    "carried_verdict": "",
                    "classes": [r["class"] for r in fresh],
                    "seat_flagged": False,
                    "verdict": verdict,
                    "evidence": [
                        r["evidence"] for r in fresh if r["class"] != "agrees"
                    ],
                }
            )
    totals: dict[str, int] = {}
    for row in rows:
        key = f"{row['status']}|{row['carried_verdict'] or 'fresh'}|{row['verdict']}"
        totals[key] = totals.get(key, 0) + 1
    by_verdict: dict[str, int] = {}
    for row in rows:
        by_verdict[row["verdict"]] = by_verdict.get(row["verdict"], 0) + 1
    out = {
        "schema": "reader-verdicts",
        "rule": __doc__.strip(),
        "candidates": reconciliation["candidate_count"],
        "by_verdict": by_verdict,
        "totals": dict(sorted(totals.items())),
        "rows": sorted(rows, key=lambda r: r["name"]),
    }
    if len(rows) != reconciliation["candidate_count"]:
        raise SystemExit("verdict rows do not cover the candidate set")
    return out, items


def write_tie_break_sheet(out: dict, items: dict[str, dict]) -> int:
    undecided = [r["name"] for r in out["rows"] if r["verdict"] == "undecided"]
    sheet_path = HERE / f"sheet-part-{TIE_BREAK_PART}.json"
    if sheet_path.exists() or not undecided:
        return len(undecided)
    template = json.loads((HERE / "sheet-part-01.json").read_text())
    sheet = {
        "classes": template["classes"],
        "items": [
            {**items[name], "id": f"tie-{index:03d}"}
            for index, name in enumerate(sorted(undecided), start=1)
        ],
    }
    sheet_path.write_text(json.dumps(sheet, indent=1) + "\n")
    brief = (HERE / "reader-c-part-01.md").read_text()
    brief = brief.replace("sheet-part-01.json", f"sheet-part-{TIE_BREAK_PART}.json")
    brief = brief.replace(
        "labels-c-part-01.json", f"labels-e-part-{TIE_BREAK_PART}.json"
    )
    (HERE / f"reader-e-part-{TIE_BREAK_PART}.md").write_text(brief)
    return len(undecided)


def main() -> int:
    out, items = settle()
    undecided = write_tie_break_sheet(out, items)
    (HERE / "reader-verdicts.json").write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {k: out[k] for k in ("candidates", "by_verdict", "totals")}, indent=1
        )
    )
    print("undecided names:", undecided)
    return 0


if __name__ == "__main__":
    sys.exit(main())

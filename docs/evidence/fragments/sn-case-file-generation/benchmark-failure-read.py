"""Read the record benchmark's held-out failures and verify the read.

The 50 regenerated records, the judge's verdicts on them and my own verdicts,
written blind, live beside this file. `--check` re-derives every number the
report states, re-runs the draw from its recorded seed, and refuses when one of
my committed verdicts has been altered. `--html` renders the report.

    uv run python docs/evidence/fragments/sn-case-file-generation/benchmark-failure-read.py --check
    uv run python docs/evidence/fragments/sn-case-file-generation/benchmark-failure-read.py --html
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import html
import json
import random
import sys
import zlib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPORT = HERE / "benchmark-failure-read.json"
RECORDS = HERE / "benchmark-failure-read-records.jsonl"
VERDICTS = HERE / "benchmark-failure-read-verdicts.jsonl"
BENCHMARK = HERE / "benchmark.json"
REFERENCE = HERE / "reference-set.json"
ARM = "one-pass/frontier"
ARCHIVED_CASE_FILES = Path(
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261118T064806928346-scfg-case-file/case-files"
)

# sha256 of "<member>|<verdict>" over the 50 rows in report order. Any edit to a
# committed verdict, or to the row order, breaks this and --check fails.
VERDICT_DIGEST = "aea31915cf10892d892978c1edda84bf0910189bb2669749468abb7abd5e1f14"

MY_VERDICTS = {"right", "wrong", "alternative"}
QUOTE_SOURCES = {"judge_reason", "reference_point", "blocker", "case_text"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reference_members() -> dict[str, dict[str, Any]]:
    payload = json.loads(REFERENCE.read_text())["member_payload_zlib_base85"]
    members = json.loads(zlib.decompress(base64.b85decode(payload)).decode())
    return {member["id"]: member for member in members}


def report() -> dict[str, Any]:
    return json.loads(REPORT.read_text())


def verdict_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(f"{row['member']}|{row['my_verdict']}" for row in rows).encode()
    ).hexdigest()


def draw(seed: int) -> dict[str, list[str]]:
    """Repeat the recorded draw: 20 sound_wrong, 20 defective_wrong, 10 right."""
    rows = [r for r in json.loads(BENCHMARK.read_text())["rows"] if r["arm"] == ARM]
    sound = sorted(
        {r["member"] for r in rows if r["incumbent_right"] and r["verdict"] == "wrong"}
    )
    defective = sorted(
        {
            r["member"]
            for r in rows
            if not r["incumbent_right"] and r["verdict"] == "wrong"
        }
    )
    right = sorted({r["member"] for r in rows if r["verdict"] == "right"})
    rng = random.Random(seed)
    return {
        "sound_wrong": rng.sample(sound, 20),
        "defective_wrong": rng.sample(defective, 20),
        "right": rng.sample(right, 10),
    }


def derived_counts(
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    judge: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exact = [
        row
        for row in rows
        if row["name_matches_reference"]
        and row["unit_generated"] == row["unit_reference"]
    ]
    group_mismatch = [
        row for row in rows if row["group_generated"] != row["group_reference"]
    ]
    causes = collections.Counter(row["cause"] for row in rows)
    order = [
        "none: record right and judged right",
        "judge wrong: group alone",
        "judge wrong: group plus a point",
        "judge wrong: group plus a name wording",
        "judge wrong: a point only the incumbent's documentation states",
        "reference too narrow: a sound name the list does not allow",
        "reference too narrow: the only right name is an opaque spelling",
        "generator blocked on a name it could derive",
        "generator lost a detail the case file states",
        "the case file lacks what the right answer needs",
    ]
    return {
        "rows": len(rows),
        "causes": {key: causes.get(key, 0) for key in order},
        "my_verdicts": dict(collections.Counter(row["my_verdict"] for row in rows)),
        "judge_verdicts": dict(
            collections.Counter(judge[row["member"]]["verdict"] for row in rows)
        ),
        "judge_failure_classes": dict(
            collections.Counter(
                judge[row["member"]]["failure_class"] or "none" for row in rows
            )
        ),
        "acceptability_agreement": sum(
            1
            for row in rows
            if (row["my_verdict"] == "right")
            == (judge[row["member"]]["verdict"] == "right")
        ),
        "exact_name_and_unit": len(exact),
        "exact_name_and_unit_passed": sum(
            1 for row in exact if judge[row["member"]]["verdict"] == "right"
        ),
        "group_mismatch": len(group_mismatch),
        "group_mismatch_passed": sum(
            1 for row in group_mismatch if judge[row["member"]]["verdict"] == "right"
        ),
        "group_match_passed": sum(
            1
            for row in rows
            if row["group_generated"] == row["group_reference"]
            and judge[row["member"]]["verdict"] == "right"
        ),
        "name_equals_incumbent": sum(
            1 for row in rows if row["name_matches_incumbent"]
        ),
        "incumbent_name_already_a_right_name": sum(
            1 for row in rows if row["incumbent_name"] == row["right_names"][0]
        ),
        "right_name_widths": {
            str(width): count
            for width, count in sorted(
                collections.Counter(len(row["right_names"]) for row in rows).items()
            )
        },
        "blockers_declared": sum(
            1 for row in rows if records[row["member"]]["draft"].get("blocker")
        ),
    }


def check() -> int:
    doc = report()
    rows = doc["rows"]
    records = {record["member"]: record for record in load_jsonl(RECORDS)}
    judge = {row["member"]: row for row in load_jsonl(VERDICTS)}
    reference = reference_members()
    problems: list[str] = []

    if len(rows) != 50 or len({row["member"] for row in rows}) != 50:
        problems.append(f"expected 50 distinct rows, found {len(rows)}")

    stored = doc["selection"]
    redrawn = draw(doc["read_seed"])
    for stratum, members in redrawn.items():
        if sorted(members) != sorted(stored.get(stratum, [])):
            problems.append(f"seed {doc['read_seed']} does not redraw {stratum}")
    order = [m for s in ("sound_wrong", "defective_wrong", "right") for m in stored[s]]
    if [row["member"] for row in rows] != order:
        problems.append("rows are not in draw order")
    for row in rows:
        expected = next(s for s, members in stored.items() if row["member"] in members)
        if row["stratum"] != expected:
            problems.append(
                f"{row['member']}: stratum {row['stratum']} != draw {expected}"
            )

    digest = verdict_digest(rows)
    if not VERDICT_DIGEST:
        problems.append("VERDICT_DIGEST is not pinned in this script")
    elif digest != VERDICT_DIGEST:
        problems.append(f"verdict digest {digest[:16]} != pinned {VERDICT_DIGEST[:16]}")
    if doc["verdict_digest"] != VERDICT_DIGEST:
        problems.append(
            "report carries a different verdict digest than the script pins"
        )

    for row in rows:
        member = row["member"]
        if row["my_verdict"] not in MY_VERDICTS:
            problems.append(f"{member}: verdict {row['my_verdict']!r}")
        if not str(row["my_reason"]).strip():
            problems.append(f"{member}: no reason")
        if member not in records:
            problems.append(f"{member}: no regenerated record")
            continue
        if member not in judge:
            problems.append(f"{member}: no judge verdict")
            continue
        verdict = judge[member]
        for key, source in (
            ("judge_verdict", "verdict"),
            ("judge_failure_class", "failure_class"),
            ("judge_reason", "reason"),
        ):
            if row[key] != verdict.get(source):
                problems.append(f"{member}: {key} differs from the verdict file")

        draft = records[member]["draft"]
        ref = reference[member]
        flags = {
            "name_matches_reference": (draft.get("name") or "") in ref["right_names"],
            "name_matches_incumbent": (draft.get("name") or "")
            == (ref.get("record_name") or ""),
        }
        for key, value in flags.items():
            if row[key] != value:
                problems.append(f"{member}: {key} is stale")
        if row["group_reference"] != ref["group"] or row["group_generated"] != (
            draft.get("group") or ""
        ):
            problems.append(f"{member}: group fields are stale")
        if row["unit_reference"] != ref["unit"] or row["unit_generated"] != (
            draft.get("unit") or ""
        ):
            problems.append(f"{member}: unit fields are stale")
        if row["right_names"] != ref["right_names"]:
            problems.append(f"{member}: right names are stale")
        if row["definition_points"] != ref["definition_points"]:
            problems.append(f"{member}: definition points are stale")
        if row["incumbent_name"] != (ref.get("record_name") or ""):
            problems.append(f"{member}: incumbent name is stale")

        source = row["quote_source"]
        if source not in QUOTE_SOURCES:
            problems.append(f"{member}: unknown quote source {source!r}")
        elif source == "judge_reason":
            if any(part not in verdict["reason"] for part in row["quote"].split(" | ")):
                problems.append(
                    f"{member}: quote {row['quote']!r} not in the judge reason"
                )
        elif source == "reference_point":
            if not any(row["quote"] in point for point in ref["definition_points"]):
                problems.append(
                    f"{member}: quote {row['quote']!r} not in a reference point"
                )
        elif source == "blocker":
            if row["quote"] != (draft.get("blocker") or ""):
                problems.append(
                    f"{member}: quote {row['quote']!r} is not the declared blocker"
                )
        elif source == "case_text":
            if not row["case_quote"] or row["quote"] not in row["case_quote"]:
                problems.append(
                    f"{member}: case quote does not sit inside the recorded line"
                )

        if ARCHIVED_CASE_FILES.is_dir():
            archived = json.loads((ARCHIVED_CASE_FILES / f"{member}.json").read_text())
            if records[member]["case_digest"] != archived["digest"]:
                problems.append(f"{member}: regenerated from a different case file")

    counts = derived_counts(rows, records, judge, reference)
    if counts != doc["counts"]:
        for key, value in counts.items():
            if doc["counts"].get(key) != value:
                problems.append(
                    f"count {key}: report {doc['counts'].get(key)!r} != {value!r}"
                )

    if set(doc["questions"]) != {"a", "b", "c", "d"}:
        problems.append("questions a-d are not all answered")
    if len(doc["changes"]) != 3:
        problems.append("the three changes are not stated")

    print(f"rows: {counts['rows']}")
    print(f"my verdicts: {counts['my_verdicts']}")
    print(f"judge verdicts: {counts['judge_verdicts']}")
    print(f"acceptability agreement: {counts['acceptability_agreement']}/50")
    print(
        f"exact name+unit: {counts['exact_name_and_unit']} rows, {counts['exact_name_and_unit_passed']} passed"
    )
    print(
        f"group mismatch: {counts['group_mismatch']} rows, {counts['group_mismatch_passed']} passed"
    )
    for cause, count in counts["causes"].items():
        print(f"  {count:2d}  {cause}")
    print(f"verdict digest: {digest}")
    if problems:
        print(f"\nFAIL ({len(problems)} problems)", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


def render_html() -> Path:
    doc = report()
    rows = doc["rows"]
    counts = doc["counts"]
    mine = " &middot; ".join(f"{k} {v}" for k, v in counts["my_verdicts"].items())
    theirs = " &middot; ".join(f"{k} {v}" for k, v in counts["judge_verdicts"].items())
    parts: list[str] = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Why the record benchmark measured nothing</title>",
        "<style>body{font:14px/1.45 system-ui,sans-serif;margin:2rem;max-width:80rem}"
        "table{border-collapse:collapse;margin:.6rem 0 1.6rem}"
        "th,td{border:1px solid #bbb}"
        "td{padding:.2rem .45rem;vertical-align:top}code{font-size:12px}"
        ".wrong{background:#fde8e8}.right{background:#e8f7e8}.alt{background:#fff6dd}"
        "</style>",
        "<h1>Why the record benchmark measured nothing</h1>",
        f"<p>One-pass/frontier regeneration of the 50 seeded held-out rows, read blind. "
        f"Seed {doc['read_seed']}. Generation USD {doc['regeneration']['generation_spend_usd']}, "
        f"judging USD 0.1525.</p>",
        f"<p>{counts['exact_name_and_unit']} records reproduced the reference name and unit and "
        f"{counts['exact_name_and_unit_passed']} passed; {counts['group_mismatch']} rows differ from "
        f"the reference group and {counts['group_mismatch_passed']} passed.</p>",
        f"<p><b>My verdicts</b>: {mine} &nbsp; <b>Judge</b>: {theirs}</p>",
        "<h2>Verdicts and causes</h2>",
        "<table><tr><th>rows</th><th>cause</th></tr>",
    ]
    for cause, count in counts["causes"].items():
        parts.append(f"<tr><td>{count}</td><td>{html.escape(cause)}</td></tr>")
    parts.append("</table>")

    parts.append(
        "<h2>The 50 rows</h2><table><tr><th>member</th><th>stratum</th>"
        "<th>my verdict</th><th>judge</th><th>group</th><th>name</th>"
        "<th>cause</th><th>quote</th></tr>"
    )
    for row in rows:
        cls = {"right": "right", "alternative": "alt"}.get(row["my_verdict"], "wrong")
        group = f"{html.escape(row['group_generated'] or '(none)')} / {html.escape(row['group_reference'])}"
        parts.append(
            f"<tr class='{cls}'><td><code>{html.escape(row['member'])}</code></td>"
            f"<td>{html.escape(row['stratum'])}</td>"
            f"<td>{row['my_verdict']}</td>"
            f"<td>{row['judge_verdict']}"
            f"{' (' + html.escape(row['judge_failure_class']) + ')' if row['judge_failure_class'] else ''}</td>"
            f"<td>{group}</td>"
            f"<td>{'as reference' if row['name_matches_reference'] else 'different'}</td>"
            f"<td>{html.escape(row['cause'])}</td>"
            f"<td>&ldquo;{html.escape(row['quote'])}&rdquo;</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h2>Questions</h2>")
    for key in "abcd":
        q = doc["questions"][key]
        parts.append(
            f"<h3>({key}) {html.escape(q['question'])}</h3><p>{html.escape(q['answer'])}</p>"
        )

    parts.append(
        "<h2>Three changes</h2><table><tr><th>#</th><th>change</th><th>counts behind it</th></tr>"
    )
    for change in doc["changes"]:
        parts.append(
            f"<tr><td>{change['order']}</td><td>{html.escape(change['change'])}</td>"
            f"<td>{html.escape(change['counts'])}</td></tr>"
        )
    parts.append("</table>")

    target = HERE / "benchmark-failure-read.html"
    target.write_text("\n".join(parts) + "\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify the committed read"
    )
    parser.add_argument(
        "--html", action="store_true", help="render benchmark-failure-read.html"
    )
    args = parser.parse_args()
    if not (args.check or args.html):
        parser.error("choose --check or --html")
    status = 0
    if args.check:
        status = check()
    if args.html:
        print(f"wrote {render_html()}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

"""Count, per stage and channel, the context the pipeline gives a model.

Every model-calling standard-name stage now writes a context receipt for each
name it handles. This census asks the same question offline over two cohorts:
the equilibrium group and the reference set's held-out members. For each name
it builds the case file the pipeline can draw on, renders each stage's prompt
from it without sending anything, and records which of the stage's expected
channels arrived. Nothing is sent to a model, so metered spend is zero and no
stored standard-name text changes.

The report leads with the omissions ranked by how many names each affects, and
cross-tabulates the held-out members adjudicated defective against the channels
their receipts were missing. Two controls are run in the same pass: a name
carrying every channel must report complete at every stage, and withholding
one channel from one stage must make that stage report the omission and the
census count it. ``--check`` re-verifies the committed JSON, including the
negative control, and exits non-zero if either control has stopped holding.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from iter_codex.standard_names import context_receipt as receipts

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
DATA = ROOT / "context-receipts.json"
PAGE = ROOT / "context-receipts.html"
RUN = Path(
    os.environ.get(
        "CONTEXT_RECEIPT_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261122T184327216910-context-receipts",
    )
)
ADJUDICATION = (
    ROOT.parent / "sn-audited-group-release" / "equilibrium-adjudication.json"
)
REFERENCE = ROOT / "reference-set.json"
DEFECTS = ROOT / "benchmark-second-round-incumbent-defects.json"
SCHEMA = "imas-codex.standard-name-context-receipts/1"
NODE = "context-receipts"
NEGATIVE_STAGE = "documentation"
NEGATIVE_CHANNEL = "enclosing_structure_text"
NEGATIVE_LOG = Path(
    os.environ.get(
        "CONTEXT_RECEIPT_NEGATIVE_LOG",
        "/home/ITER/mcintos/.cache/iter-codex-context-receipts/negative-control.log",
    )
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "--no-optional-locks", *args], text=True
    ).strip()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_members() -> list[str]:
    payload = json.loads(REFERENCE.read_text())
    encoded = payload.get("member_payload_zlib_base85")
    if not encoded:
        raise SystemExit("reference-set.json carries no member payload")
    members = json.loads(zlib.decompress(base64.b85decode(encoded)).decode())
    return [str(member["id"]) for member in members]


def reference_verdicts() -> dict[str, str]:
    payload = json.loads(REFERENCE.read_text())
    encoded = payload.get("member_payload_zlib_base85")
    members = json.loads(zlib.decompress(base64.b85decode(encoded)).decode())
    return {
        str(member["id"]): str(member.get("verdict") or "unknown") for member in members
    }


def held_out_members() -> list[str]:
    from iter_codex.standard_names.benchmark import assign_development_split

    members = reference_members()
    split = assign_development_split(
        members, seed=20261118, strata=reference_verdicts()
    )
    return [member for member in members if split[member] == "held_out"]


def defect_members() -> dict[str, str]:
    """Return ``member -> defect class`` for the adjudicated defective names."""
    payload = json.loads(DEFECTS.read_text())
    mapping: dict[str, str] = {}
    for block in payload.get("classes") or ():
        label = str(block.get("class") or "")
        for member in block.get("members") or ():
            mapping[str(member["member"])] = label
    return mapping


def equilibrium_members() -> list[str]:
    payload = json.loads(ADJUDICATION.read_text())
    return [str(member) for member in payload["members"]]


_STORED_QUERY = """
MATCH (sn:StandardName)
WHERE sn.id IN $ids
RETURN sn.id AS id, sn.description AS description, sn.documentation AS documentation,
       sn.unit AS unit, sn.kind AS kind, sn.name_stage AS name_stage
"""


def stored_state(gc: Any, ids: list[str]) -> dict[str, Any]:
    rows = sorted(
        (dict(row) for row in gc.query(_STORED_QUERY, ids=ids)),
        key=lambda row: row.get("id") or "",
    )
    plain = json.loads(json.dumps(rows, default=str, sort_keys=True))
    return {"rows": len(rows), "digest": digest(plain)}


def row_of(receipt_set: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in receipts.STAGES:
        receipt = receipt_set.get(stage)
        if receipt is None:
            continue
        stages[stage] = {
            "complete": receipt.complete,
            "expected": list(receipt.expected),
            "missing": list(receipt.missing),
            "digest": receipt.prompt_digest[:12],
        }
    return stages


def build(ids: list[str], *, limit: int = 0) -> dict[str, Any]:
    from iter_codex.graph.client import GraphClient
    from iter_codex.standard_names.case_file import build_name_case_file

    started = time.time()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    by_stage: dict[str, list[Any]] = defaultdict(list)
    records: dict[str, Any] = {}
    with GraphClient() as gc:
        before = stored_state(gc, ids)
        for index, name in enumerate(ids, 1):
            if limit and index > limit:
                break
            try:
                record = build_name_case_file(name, gc=gc)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append({"id": name, "error": f"{type(exc).__name__}: {name}"})
                continue
            records[name] = record
            receipt_set: dict[str, Any] = {}
            for stage in receipts.STAGES:
                receipt = receipts.receipt_for_stage(stage, record)
                receipt_set[stage] = receipt
                by_stage[stage].append(receipt)
            rows.append({"id": name, **row_of(receipt_set)})
            if index % 25 == 0:
                print(f"{index}/{len(ids)} in {time.time() - started:.0f}s", flush=True)
        after = stored_state(gc, ids)
    return {
        "rows": rows,
        "failures": failures,
        "by_stage": by_stage,
        "records": records,
        "elapsed_seconds": round(time.time() - started, 1),
        "stored_before": before,
        "stored_after": after,
    }


def controls(result: dict[str, Any]) -> dict[str, Any]:
    """Find a name complete at every stage, and withhold one channel from one."""
    positive = None
    for row in result["rows"]:
        entries = [value for value in row.values() if isinstance(value, dict)]
        if entries and all(entry["complete"] for entry in entries):
            positive = row["id"]
            break
    if positive is None:
        raise SystemExit(
            "no name in the cohort carries every channel; the positive control "
            f"cannot be shown. {len(result['rows'])} rows were read."
        )
    record = result["records"][positive]
    withheld_receipt = receipts.receipt_for_stage(
        NEGATIVE_STAGE, record, withhold=[NEGATIVE_CHANNEL]
    )
    base = receipts.summarise(complete_receipts(record))
    counted = receipts.summarise(complete_receipts(record) + [withheld_receipt])
    rank = next(
        index
        for index, row in enumerate(counted["omissions"], 1)
        if row["stage"] == NEGATIVE_STAGE and row["channel"] == NEGATIVE_CHANNEL
    )
    return {
        "positive": {
            "name": positive,
            "complete_at_every_stage": True,
            "stages": list(receipts.STAGES),
        },
        "negative": {
            "name": positive,
            "stage": NEGATIVE_STAGE,
            "channel": NEGATIVE_CHANNEL,
            "reported_complete": withheld_receipt.complete,
            "missing": list(withheld_receipt.missing),
            "absent_before": base["table"][NEGATIVE_STAGE][NEGATIVE_CHANNEL]["absent"],
            "absent_after": counted["table"][NEGATIVE_STAGE][NEGATIVE_CHANNEL][
                "absent"
            ],
            "rank": rank,
        },
    }


def complete_receipts(record: Any) -> list[Any]:
    return [receipts.receipt_for_stage(stage, record) for stage in receipts.STAGES]


def cohort_receipts(result: dict[str, Any], members: set[str]) -> dict[str, list[Any]]:
    return {
        stage: [r for r in receipts_by_stage if r.name in members]
        for stage, receipts_by_stage in result["by_stage"].items()
    }


def summarise_cohort(result: dict[str, Any], members: set[str]) -> dict[str, Any]:
    by_stage = cohort_receipts(result, members)
    flat = [receipt for group in by_stage.values() for receipt in group]
    return receipts.summarise(flat)


def cross_tabulation(
    result: dict[str, Any], defects: dict[str, str], held_out: set[str]
) -> dict[str, Any]:
    rows = {row["id"]: row for row in result["rows"]}
    per_channel: Counter[str] = Counter()
    per_stage_channel: Counter[tuple[str, str]] = Counter()
    per_class: dict[str, dict[str, Any]] = {}
    for member, label in sorted(defects.items()):
        row = rows.get(member)
        block = per_class.setdefault(label, {"members": 0, "channels": Counter()})
        block["members"] += 1
        if row is None:
            continue
        seen: set[str] = set()
        for stage in receipts.STAGES:
            entry = row.get(stage)
            if not entry:
                continue
            for channel in entry["missing"]:
                per_stage_channel[(stage, channel)] += 1
                if channel not in seen:
                    seen.add(channel)
                    per_channel[channel] += 1
                    block["channels"][channel] += 1
    return {
        "defective_held_out": len(defects),
        "held_out": len(held_out),
        "per_channel": dict(
            sorted(per_channel.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "per_stage_channel": [
            {"stage": stage, "channel": channel, "names": count}
            for (stage, channel), count in sorted(
                per_stage_channel.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ],
        "per_class": [
            {
                "class": label,
                "members": block["members"],
                "channels": dict(
                    sorted(block["channels"].items(), key=lambda kv: (-kv[1], kv[0]))
                ),
            }
            for label, block in sorted(per_class.items())
        ],
    }


def _cell(value: object) -> str:
    return f"<td>{escape(str(value))}</td>"


def _table(identifier: str, headers: list[str], rows: list[tuple]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(_cell(value) for value in row) + "</tr>" for row in rows
    )
    return (
        f'<table id="{identifier}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def document(report: dict[str, Any]) -> str:
    populations = report["populations"]
    controls = report["controls"]
    cross = report["cross_tab"]
    provenance = report["provenance"]
    parts: list[str] = []

    def add(tag: str, suffix: str, text: str) -> None:
        parts.append(f'<{tag} id="{NODE}-{suffix}">{text}</{tag}>')

    total = len(report["rows"])
    add(
        "h1",
        "title",
        "What each model-calling stage was given, and what it lacked",
    )
    add(
        "p",
        "summary",
        f"{total} names (the {report['inputs']['adjudication']['members']} "
        "equilibrium members and the "
        f"{report['inputs']['reference']['held_out']} held-out reference members) "
        "were read through the case file each stage can draw on. Every model-calling "
        f"stage's own prompt template was rendered for every name and sent nowhere: "
        f"<strong>{provenance['graph_writes']} graph writes and "
        f"{provenance['model_calls']} model calls</strong>. The table below ranks the "
        "channels a stage expected and did not receive.",
    )

    gate_revision = os.environ.get("CONTEXT_RECEIPT_REVISION") or report["revision"]
    gate_result = os.environ.get("CONTEXT_RECEIPT_GATE_RESULT", "").strip()
    gate_sentence = (
        f" Its whole-directory gate reported <code>{escape(gate_result)}</code>."
        if gate_result
        else ""
    )
    add(
        "p",
        "provenance",
        f"Census data at revision <code>{report['revision']}</code>, tree "
        f"<code>{report['tree']}</code>, generated {report['generated_at']}. The "
        f"gates and control runs beside it were read at revision "
        f"<code>{gate_revision}</code>.{gate_sentence}",
    )

    add("h2", "omissions-head", "Omissions, ranked by how many names each affects")
    omission_rows = [
        (
            row["stage"],
            row["channel"],
            row["names"],
            row["expected"],
            f"{row['share_absent']:.0%}",
        )
        for row in populations["all"]["omissions"]
    ]
    if omission_rows:
        parts.append(
            _table(
                f"{NODE}-omissions-table",
                ["Stage", "Channel", "Names absent", "Names expecting it", "Share"],
                omission_rows,
            )
        )
    else:
        add(
            "p", "omissions-none", "No stage was expected a channel it did not receive."
        )

    parts.append(
        '<figure id="context-receipts-channel-coverage">'
        '<img src="/imas-codex/figures/sn-case-file-generation/context-receipts/'
        'channel-coverage.svg" '
        'alt="A stage by channel matrix of the share of names that carried each '
        'expected channel, drawn from context-receipts.json">'
        "<figcaption>Share of the names that expected each channel and carried it, "
        "per stage. A blank cell is a channel the stage does not expect. Drawn by "
        "<code>docs/figures/sn-case-file-generation/context-receipts/"
        "make_channel_coverage.py</code> from "
        "<code>context-receipts.json</code>.</figcaption></figure>"
    )

    add("h2", "cohorts-head", "Each cohort on its own")
    for key, label in (
        ("equilibrium", "Equilibrium members"),
        ("held_out", "Held-out reference members"),
    ):
        block = populations[key]
        size = sum(1 for row in report["rows"] if key in row.get("populations", ()))
        add("h3", f"{key}-head", f"{label}: {size} names")
        rows = [
            (
                row["stage"],
                row["channel"],
                row["names"],
                row["expected"],
                f"{row['share_absent']:.0%}",
            )
            for row in block["omissions"]
        ]
        if rows:
            parts.append(
                _table(
                    f"{NODE}-{key}-omissions-table",
                    ["Stage", "Channel", "Names absent", "Names expecting it", "Share"],
                    rows,
                )
            )
        else:
            add("p", f"{key}-none", "Every expected channel was present.")

    add("h2", "controls-head", "The controls")
    parts.append(
        _table(
            f"{NODE}-controls-table",
            ["Control", "Setup", "Result"],
            [
                (
                    "positive",
                    f"{controls['positive']['name']} carries every channel",
                    "complete at every stage"
                    if controls["positive"]["complete_at_every_stage"]
                    else "NOT complete",
                ),
                (
                    "negative",
                    f"withhold {controls['negative']['channel']} from "
                    f"{controls['negative']['stage']}",
                    f"reported complete={controls['negative']['reported_complete']}, "
                    f"missing={', '.join(controls['negative']['missing'])}, "
                    f"census absent {controls['negative']['absent_before']} -> "
                    f"{controls['negative']['absent_after']}",
                ),
            ],
        )
    )

    add(
        "h2",
        "cross-head",
        "The adjudicated-defective held-out names against the channels they lacked",
    )
    add(
        "p",
        "cross-names",
        f"{cross['defective_held_out']} of the {cross['held_out']} held-out members "
        "were adjudicated defective on name or definition. Counting, over those "
        "names, which expected channels their receipts were missing:",
    )
    parts.append(
        _table(
            f"{NODE}-cross-table",
            ["Channel", "Defective names missing it"],
            [(channel, count) for channel, count in cross["per_channel"].items()],
        )
    )
    parts.append(
        _table(
            f"{NODE}-cross-class-table",
            ["Adjudicated class", "Names", "Channels they lacked"],
            [
                (
                    block["class"],
                    block["members"],
                    ", ".join(
                        f"{channel} ({count})"
                        for channel, count in block["channels"].items()
                    )
                    or "-",
                )
                for block in cross["per_class"]
            ],
        )
    )

    add("h2", "table-head", "Every stage and channel")
    stage_rows = []
    for stage in receipts.STAGES:
        for channel, block in populations["all"]["table"].get(stage, {}).items():
            stage_rows.append(
                (
                    stage,
                    channel,
                    block["expected"],
                    block["present"],
                    block["absent"],
                    f"{block['share_present']:.0%}"
                    if block["share_present"] is not None
                    else "-",
                )
            )
    parts.append(
        _table(
            f"{NODE}-stage-table",
            ["Stage", "Channel", "Expected", "Present", "Absent", "Present share"],
            stage_rows,
        )
    )

    add(
        "p",
        "digest-note",
        "The digest beside each row is over the prompt this census rendered from "
        "the name's case file, so it shows that a stage's template is deterministic "
        "for fixed inputs. A live stage's receipt carries the digest of the exact "
        "prompt it sent, which also reflects the context the running pipeline "
        "assembled.",
    )

    never_expected = sorted(
        set(receipts.CHANNELS)
        - {channel for group in receipts.STAGE_CHANNELS.values() for channel in group}
    )
    add(
        "p",
        "never-expected",
        "No current stage template reads "
        + ", ".join(f"<code>{channel}</code>" for channel in never_expected)
        + ", so no stage is expected to receive it. It is reported here rather "
        "than counted as an omission.",
    )

    add("h2", "writes-head", "Nothing was sent and nothing stored changed")
    add(
        "p",
        "writes",
        f"The census renders templates and reads the graph. It reports "
        f"<code>model_calls: {provenance['model_calls']}</code> and "
        f"<code>graph_writes: {provenance['graph_writes']}</code>. The stored text of "
        f"the {provenance['stored_before']['rows']} names read was digested before "
        f"(<code>{provenance['stored_before']['digest'][:16]}</code>) and after "
        f"(<code>{provenance['stored_after']['digest'][:16]}</code>): "
        f"{'identical' if provenance['stored_unchanged'] else 'DIFFERENT'}.",
    )

    add("h2", "rows-head", "Every name")
    parts.append(
        _table(
            f"{NODE}-rows-table",
            ["Name", "Stages complete", "Missing channels"],
            [
                (
                    row["id"],
                    f"{sum(1 for entry in row.values() if isinstance(entry, dict) and entry['complete'])}"
                    f"/{len(receipts.STAGES)}",
                    ", ".join(
                        f"{stage}: {', '.join(entry['missing'])}"
                        for stage, entry in row.items()
                        if isinstance(entry, dict) and entry["missing"]
                    )
                    or "-",
                )
                for row in report["rows"]
            ],
        )
    )
    body = "\n".join(parts)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="docs-project" content="iter-codex">\n'
        '<meta name="reckon-type" content="evidence-fragment">\n'
        '<meta name="plan-slug" content="sn-case-file-generation">\n'
        f'<meta name="node-id" content="{NODE}">\n'
        "<title>Context receipts for every model-calling stage</title>\n"
        "</head>\n<body>\n<main>\n"
        f"{body}\n</main>\n</body>\n</html>\n"
    )


def annotate_rows(result: dict[str, Any]) -> None:
    equilibrium = set(equilibrium_members())
    held = set(held_out_members())
    defects = defect_members()
    for row in result["rows"]:
        row["populations"] = [
            name
            for name, members in (("equilibrium", equilibrium), ("held_out", held))
            if row["id"] in members
        ]
        row["defect"] = defects.get(row["id"])


def table_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the per-stage/channel counts from the committed per-name rows."""
    counts: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        for stage in receipts.STAGES:
            entry = row.get(stage)
            if not isinstance(entry, dict):
                continue
            bucket = counts.setdefault(stage, {})
            for channel in entry.get("expected") or ():
                counts_for = bucket.setdefault(channel, [0, 0, 0])
                counts_for[0] += 1
                if channel in entry["missing"]:
                    counts_for[2] += 1
                else:
                    counts_for[1] = counts_for[1] + 1
    table: dict[str, Any] = {}
    for stage in receipts.STAGES:
        if stage not in counts:
            continue
        table[stage] = {}
        for channel, (expected, present, absent) in counts[stage].items():
            table[stage][channel] = {
                "expected": expected,
                "present": present,
                "absent": absent,
            }
    return table


def check(report: dict[str, Any]) -> list[str]:
    """Return the names of every invariant the committed report fails."""
    problems: list[str] = []
    if report.get("schema") != SCHEMA:
        problems.append(f"schema is not {SCHEMA}")
    positive = report["controls"]["positive"]
    if not positive.get("complete_at_every_stage"):
        problems.append("positive control did not report complete at every stage")
    negative = report["controls"]["negative"]
    if negative.get("reported_complete") is not False:
        problems.append("negative control reported the context complete")
    if negative.get("channel") not in negative.get("missing", []):
        problems.append("negative control did not name the withheld channel")
    if negative.get("absent_after") != negative.get("absent_before") + 1:
        problems.append(
            "negative control was not counted: absent "
            f"{negative.get('absent_before')} -> {negative.get('absent_after')}"
        )
    if report["provenance"]["model_calls"] != 0:
        problems.append("the census reported a model call")
    if not report["provenance"]["stored_unchanged"]:
        problems.append("stored standard-name text changed during the run")
    rows = report["rows"]
    per_stage = report["populations"]["all"]["names"]
    if not isinstance(per_stage, dict) or any(
        count != len(rows) for count in per_stage.values()
    ):
        problems.append("a stage's population count does not match the row count")
    recomputed = table_from_rows(rows)
    for stage, channels in recomputed.items():
        for channel, counts in channels.items():
            recorded = report["populations"]["all"]["table"].get(stage, {}).get(channel)
            if recorded is None:
                problems.append(f"{stage}/{channel} is in the rows but not the table")
                continue
            for key in ("expected", "present", "absent"):
                if recorded[key] != counts[key]:
                    problems.append(
                        f"{stage}/{channel} {key}: recorded {recorded[key]}, "
                        f"rows give {counts[key]}"
                    )
    return problems


def build_report(*, limit: int = 0) -> dict[str, Any]:
    equilibrium = equilibrium_members()
    held = held_out_members()
    defects = defect_members()
    ids = sorted(set(equilibrium) | set(held))
    result = build(ids, limit=limit)
    annotate_rows(result)
    populations = {
        "equilibrium": summarise_cohort(result, set(equilibrium)),
        "held_out": summarise_cohort(result, set(held)),
        "all": summarise_cohort(result, set(ids)),
    }
    provenance = {
        "model_calls": 0,
        "graph_writes": 0,
        "graph_statements": sum(
            record.provenance.get("graph_statements", 0)
            for record in result["records"].values()
        ),
        "stored_before": result["stored_before"],
        "stored_after": result["stored_after"],
        "stored_unchanged": (
            result["stored_before"]["digest"] == result["stored_after"]["digest"]
        ),
    }
    return {
        "schema": SCHEMA,
        "node": NODE,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": (
            "uv run --no-sync python "
            "docs/evidence/fragments/sn-case-file-generation/context-receipts.py"
        ),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "inputs": {
            "adjudication": {
                "path": str(ADJUDICATION.relative_to(ROOT.parent)),
                "digest": file_sha256(ADJUDICATION),
                "members": len(equilibrium),
            },
            "reference": {
                "path": REFERENCE.name,
                "digest": file_sha256(REFERENCE),
                "members": len(reference_members()),
                "held_out": len(held),
            },
            "defects": {
                "path": DEFECTS.name,
                "digest": file_sha256(DEFECTS),
                "defective_held_out": len(defects),
            },
        },
        "stages": list(receipts.STAGES),
        "channels": list(receipts.CHANNELS),
        "populations": populations,
        "controls": controls(result),
        "cross_tab": cross_tabulation(result, defects, set(held)),
        "provenance": provenance,
        "elapsed_seconds": result["elapsed_seconds"],
        "failures": result["failures"],
        "rows": result["rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--render",
        action="store_true",
        help="re-render the page from the committed JSON without reading the graph",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.render:
        PAGE.write_text(document(json.loads(DATA.read_text())))
        print(f"re-rendered {PAGE}")
        return 0
    if args.check:
        report = json.loads(DATA.read_text())
        problems = check(report)
        if problems:
            print("context-receipts.json FAILED its checks:")
            for problem in problems:
                print(" -", problem)
            return 1
        print(
            f"context-receipts.json verified: {len(report['rows'])} names, "
            f"{len(report['populations']['all']['omissions'])} omissions"
        )
        return 0
    report = build_report(limit=args.limit)
    text = json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True)
    DATA.write_text(text + "\n")
    PAGE.write_text(document(report))
    NEGATIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with NEGATIVE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "at": report["generated_at"],
                    "revision": report["revision"],
                    "positive": report["controls"]["positive"],
                    "negative": report["controls"]["negative"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(
        f"{len(report['rows'])} names, {len(text)} bytes, "
        f"{report['elapsed_seconds']}s, failures {len(report['failures'])}"
    )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())

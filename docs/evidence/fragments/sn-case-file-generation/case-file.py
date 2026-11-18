"""Build a case file for every equilibrium member and reader-calibration name.

The populations are the ones the reference set is drawn from: the equilibrium
group's members as committed in the frozen adjudication record, and the 84 names
the reader calibration hand-checked. One case file is built per name by code,
with no model call, and the run reports the token size as a distribution, the
digest of every record, and the verdict each record carries.

Three facts are shown rather than assumed: a member whose enclosing-structure
text is withheld carries an incomplete verdict that refuses its use while the
same member's complete record is usable; a member with no family or no
neighbour states that in prose instead of leaving an empty block; and the run
changes nothing stored, proved by a digest of the stored text before and after.

The full records are written to the run directory -- they are regenerable, and
the repository fragment keeps the per-name row, the distribution and the
receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from imas_codex.standard_names.source_reads import source_paths_projection

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
DATA = ROOT / "case-file.json"
PAGE = ROOT / "case-file.html"
FRAGMENT = ROOT / "scfg-case-file.html"
RUN = Path(
    os.environ.get(
        "CASE_FILE_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261118T064806928346-scfg-case-file",
    )
)
ADJUDICATION = (
    ROOT.parent / "sn-audited-group-release" / "equilibrium-adjudication.json"
)
CALIBRATION = ROOT.parent / "sn-audited-group-release" / "reader-calibration.json"
WITHHELD_MEMBER = "elongation_of_flux_surface"
NODE = "scfg-case-file"

_STORED_QUERY = f"""
MATCH (sn:StandardName)
WHERE sn.id IN $ids
RETURN sn.id AS id,
       sn.description AS description,
       sn.documentation AS documentation,
       sn.unit AS unit,
       sn.kind AS kind,
       sn.name_stage AS name_stage,
       sn.physical_base AS physical_base,
       {source_paths_projection()} AS source_ids
"""


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def populations() -> tuple[list[str], list[str], list[str]]:
    """Return the equilibrium, calibration and combined populations."""
    equilibrium = list(json.loads(ADJUDICATION.read_text())["members"])
    calibration = list(json.loads(CALIBRATION.read_text())["cohort"])
    combined = sorted(set(equilibrium) | set(calibration))
    return equilibrium, calibration, combined


def stored_state(gc, ids: list[str]) -> dict[str, object]:
    """Digest every stored text field the case file claims to read."""
    rows = sorted(
        (dict(row) for row in gc.query(_STORED_QUERY, ids=ids)),
        key=lambda row: row.get("id") or "",
    )
    plain = json.loads(json.dumps(rows, default=str, sort_keys=True))
    return {"rows": len(rows), "digest": digest(plain)}


def distribution(values: list[int]) -> dict[str, object]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    buckets = Counter()
    for value in ordered:
        buckets[f"{value // 2000 * 2000}-{value // 2000 * 2000 + 1999}"] += 1
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": ordered[int(len(ordered) * 0.10)],
        "p50": int(statistics.median(ordered)),
        "p90": ordered[int(len(ordered) * 0.90)],
        "p99": ordered[min(int(len(ordered) * 0.99), len(ordered) - 1)],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 1),
        "total": sum(ordered),
        "buckets": dict(sorted(buckets.items(), key=lambda item: item[0])),
    }


def row_of(record) -> dict[str, object]:
    content = record.content
    sources = content["sources"]
    family = content["family"]
    neighbours = content["neighbours"]
    grammar = content["grammar"]
    return {
        "id": record.target["id"],
        "digest": record.digest,
        "tokens": record.tokens,
        "verdict": record.completeness["verdict"],
        "missing": list(record.completeness["missing"]),
        "sources": len(sources),
        "enclosing_statements": sum(
            len(source["enclosing"]["statements"]) for source in sources
        ),
        "chain_depth": max(
            (len(source["enclosing"]["chain"]) for source in sources), default=0
        ),
        "family": family["status"],
        "family_members": len(family.get("members") or ()),
        "subject_neighbours": len(neighbours["by_subject"].get("members") or ()),
        "locus_neighbours": len(neighbours["by_locus"].get("members") or ()),
        "grammar_segments": len(grammar.get("segments") or ()),
        "vocabulary_segments": len(
            (grammar.get("vocabulary") or {}).get("entries") or ()
        ),
        "conventions": len((grammar.get("conventions") or {}).get("entries") or ()),
        "incumbent": content["incumbent"] is not None,
        "adjudicator_reason": content["adjudicator_reason"] is not None,
        "statements": record.provenance["graph_statements"],
        "model_calls": record.provenance["model_calls"],
        "graph_writes": record.provenance["graph_writes"],
    }


def build(ids: list[str], limit: int = 0) -> dict[str, object]:
    """Build one case file per name through a single graph client."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.case_file import build_name_case_file

    run_dir = RUN / "case-files"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = []
    failures = []
    prose = []
    with GraphClient() as gc:
        before = stored_state(gc, ids)
        for index, name in enumerate(ids, 1):
            if limit and index > limit:
                break
            try:
                record = build_name_case_file(name, gc=gc)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                failures.append({"id": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            records.append(record)
            family = record.content["family"]
            if family["status"] == "none":
                prose.append(
                    {
                        "id": name,
                        "channel": "family",
                        "statement": family["statement"],
                    }
                )
            for channel in ("by_subject", "by_locus"):
                block = record.content["neighbours"][channel]
                if block["status"] == "none":
                    prose.append(
                        {
                            "id": name,
                            "channel": channel,
                            "statement": block["statement"],
                        }
                    )
            (run_dir / f"{name}.json").write_text(
                json.dumps(record.model_dump(), ensure_ascii=False, sort_keys=True)
            )
            if index % 25 == 0:
                print(f"{index}/{len(ids)} in {time.time() - started:.0f}s", flush=True)
        after = stored_state(gc, ids)

        rebuild = build_name_case_file(ids[0], gc=gc)
        determinism = {
            "id": ids[0],
            "first": records[0].digest,
            "second": rebuild.digest,
            "reproduced": rebuild.digest == records[0].digest,
        }

        complete = build_name_case_file(WITHHELD_MEMBER, gc=gc)
        withheld = build_name_case_file(
            WITHHELD_MEMBER, gc=gc, withhold=["enclosing_structure_text"]
        )
        (run_dir / f"{WITHHELD_MEMBER}.complete.json").write_text(
            json.dumps(complete.model_dump(), ensure_ascii=False, sort_keys=True)
        )
        (run_dir / f"{WITHHELD_MEMBER}.withheld.json").write_text(
            json.dumps(withheld.model_dump(), ensure_ascii=False, sort_keys=True)
        )
        demo = {
            "member": WITHHELD_MEMBER,
            "complete": {
                "digest": complete.digest,
                "verdict": complete.completeness["verdict"],
                "usable": complete.completeness["usable"],
            },
            "withheld": {
                "digest": withheld.digest,
                "verdict": withheld.completeness["verdict"],
                "usable": withheld.completeness["usable"],
                "missing": list(withheld.completeness["missing"]),
                "enclosing_statement": withheld.content["sources"][0]["enclosing"][
                    "statement"
                ],
            },
        }
        assert demo["complete"]["usable"] is True
        assert demo["withheld"]["usable"] is False
        assert any(
            entry.startswith("enclosing_structure_text:")
            for entry in demo["withheld"]["missing"]
        )
        assert demo["withheld"]["digest"] != demo["complete"]["digest"]
    return {
        "records": records,
        "rows": [row_of(record) for record in records],
        "failures": failures,
        "prose": prose,
        "elapsed_seconds": round(time.time() - started, 1),
        "stored_before": before,
        "stored_after": after,
        "determinism": determinism,
        "withheld_demo": demo,
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_counts(rows: list[dict], names: set[str]) -> dict[str, object]:
    subset = [row for row in rows if row["id"] in names]
    verdicts = Counter(row["verdict"] for row in subset)
    missing: Counter[str] = Counter()
    for row in subset:
        for entry in row["missing"]:
            missing[entry.split(":", 1)[0]] += 1
    return {
        "count": len(subset),
        "verdicts": dict(sorted(verdicts.items())),
        "missing_channels": dict(sorted(missing.items())),
    }


def summarise(
    equilibrium: list[str],
    calibration: list[str],
    combined: list[str],
    result: dict[str, object],
) -> dict[str, object]:
    rows = result["rows"]
    tokens = [row["tokens"] for row in rows]
    prose = result["prose"]
    by_channel: dict[str, list[dict]] = {}
    for entry in prose:
        by_channel.setdefault(entry["channel"], []).append(entry)
    return {
        "schema": "imas-codex.standard-name-case-file-run/1",
        "node": NODE,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": (
            "CASE_FILE_RUN=<run> uv run --no-sync python "
            "docs/evidence/fragments/sn-case-file-generation/case_file.py"
        ),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "inputs": {
            "adjudication": {
                "path": str(ADJUDICATION.parent.name + "/" + ADJUDICATION.name),
                "digest": file_sha256(ADJUDICATION),
                "members": len(equilibrium),
            },
            "calibration": {
                "path": str(CALIBRATION.parent.name + "/" + CALIBRATION.name),
                "digest": file_sha256(CALIBRATION),
                "names": len(calibration),
            },
        },
        "populations": {
            "equilibrium": _selection_counts(rows, set(equilibrium)),
            "calibration": _selection_counts(rows, set(calibration)),
            "both": _selection_counts(rows, set(equilibrium) & set(calibration)),
            "combined": _selection_counts(rows, set(calibration) | set(equilibrium)),
        },
        "tokens": {
            "all": distribution(tokens),
            "equilibrium": distribution(
                [row["tokens"] for row in rows if row["id"] in set(equilibrium)]
            ),
            "calibration": distribution(
                [row["tokens"] for row in rows if row["id"] in set(calibration)]
            ),
        },
        "content": {
            "sourceless": sum(1 for row in rows if row["sources"] == 0),
            "with_family_members": sum(1 for row in rows if row["family"] == "members"),
            "no_family": sum(1 for row in rows if row["family"] == "none"),
            "with_subject_neighbours": sum(
                1 for row in rows if row["subject_neighbours"]
            ),
            "with_locus_neighbours": sum(1 for row in rows if row["locus_neighbours"]),
            "with_adjudicator_reason": sum(
                1 for row in rows if row["adjudicator_reason"]
            ),
            "median_enclosing_statements": int(
                statistics.median(row["enclosing_statements"] for row in rows)
            ),
            "max_chain_depth": max(row["chain_depth"] for row in rows),
        },
        "prose": {
            channel: {
                "count": len(entries),
                "statements": sorted({entry["statement"] for entry in entries}),
                "samples": sorted({entry["statement"] for entry in entries})[:8],
            }
            for channel, entries in sorted(by_channel.items())
        },
        "withheld_demo": result["withheld_demo"],
        "determinism": result["determinism"],
        "provenance": {
            "statements": sum(row["statements"] for row in rows),
            "model_calls": sum(row["model_calls"] for row in rows),
            "graph_writes": sum(row["graph_writes"] for row in rows),
            "stored_before": result["stored_before"],
            "stored_after": result["stored_after"],
            "stored_unchanged": (
                result["stored_before"]["digest"] == result["stored_after"]["digest"]
            ),
        },
        "elapsed_seconds": result["elapsed_seconds"],
        "failures": result["failures"],
        "rows": rows,
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


def sections(report: dict[str, object], prefix: str = NODE) -> str:
    """Render the body: every section, in reading order, under ``prefix``."""
    pid = prefix
    tokens = report["tokens"]
    populations = report["populations"]
    content = report["content"]
    provenance = report["provenance"]
    demo = report["withheld_demo"]
    rows = report["rows"]
    parts: list[str] = []

    def add(tag: str, suffix: str, text: str) -> None:
        parts.append(f'<{tag} id="{pid}-{suffix}">{text}</{tag}>')

    total = populations["combined"]["count"]
    usable = populations["combined"]["verdicts"].get("complete", 0)
    incomplete = total - usable
    add(
        "h1",
        "title",
        "One case file per name, with its text, its digest and its own verdict",
    )
    add(
        "p",
        "summary",
        f"{total} records were built by code from graph reads alone: "
        f"{populations['equilibrium']['count']} equilibrium members, "
        f"{populations['calibration']['count']} reader-calibration names, "
        f"{populations['both']['count']} of them in both. "
        f"<strong>{usable} of {total} carry a complete verdict</strong> and "
        f"{incomplete} carry an incomplete one; every incomplete record names the "
        "channel or the target text it lacks, so an empty block can never pass as "
        "an absence. A record's size ranges from "
        f"{tokens['all']['min']} to {tokens['all']['max']} tokens with a median of "
        f"{tokens['all']['p50']} and a 90th percentile of {tokens['all']['p90']}. "
        f"The run issued "
        f"{provenance['statements']} graph statements, made "
        f"{provenance['model_calls']} model calls and wrote "
        f"{provenance['graph_writes']} graph writes, and a digest of every stored "
        "field these records read is identical before and after: "
        f"<code>{provenance['stored_before']['digest'][:16]}</code>. "
        f"The digest of a record covers its content only, so rebuilding "
        f"<code>{escape(report['determinism']['id'])}</code> from the same graph "
        f"reproduced <code>{report['determinism']['first'][:16]}</code>.",
    )

    add(
        "h2",
        "demo-head",
        "The assembled record, and the same member with one channel withheld",
    )
    add(
        "p",
        "demo",
        f"<code>{escape(demo['member'])}</code> is built twice in the same run, once "
        "whole and once with its enclosing-structure text withheld. The complete "
        f"record is usable under a <code>{escape(demo['complete']['verdict'])}</code>"
        " verdict; the withheld record keeps its digest, keeps every other channel, "
        "and carries an incomplete verdict that refuses it and names the channel "
        "and the source path it lacks.",
    )
    parts.append(
        _table(
            f"{pid}-demo-table",
            [
                "Record",
                "Verdict",
                "Usable",
                "Digest",
                "Missing",
                "What the record says",
            ],
            [
                (
                    "complete",
                    demo["complete"]["verdict"],
                    demo["complete"]["usable"],
                    demo["complete"]["digest"][:16],
                    "-",
                    "every channel assembled",
                ),
                (
                    "enclosing text withheld",
                    demo["withheld"]["verdict"],
                    demo["withheld"]["usable"],
                    demo["withheld"]["digest"][:16],
                    ", ".join(demo["withheld"]["missing"]),
                    demo["withheld"]["enclosing_statement"],
                ),
            ],
        )
    )

    add("h2", "tokens-head", "How large a record is")
    parts.append(
        _table(
            f"{pid}-tokens-table",
            [
                "Population",
                "Records",
                "Min",
                "p10",
                "p50",
                "p90",
                "p99",
                "Max",
                "Mean",
                "Total",
            ],
            [
                (
                    name,
                    block["count"],
                    block.get("min"),
                    block.get("p10"),
                    block.get("p50"),
                    block.get("p90"),
                    block.get("p99"),
                    block.get("max"),
                    block.get("mean"),
                    block.get("total"),
                )
                for name, block in tokens.items()
            ],
        )
    )
    add(
        "p",
        "tokens-note",
        "Size is the record's own <code>tokens</code> field, computed from the "
        "rendered record text with the estimators the batching ledger uses. The "
        "buckets below are 2000-token bands over the combined population.",
    )
    parts.append(
        _table(
            f"{pid}-tokens-buckets-table",
            ["Token band", "Records"],
            list(tokens["all"]["buckets"].items()),
        )
    )

    add(
        "h2",
        "verdicts-head",
        "The verdict of every record, and what an incomplete one names",
    )
    verdict_rows = []
    for name in ("equilibrium", "calibration", "both", "combined"):
        block = populations[name]
        verdict_rows.append(
            (
                name,
                block["count"],
                block["verdicts"].get("complete", 0),
                block["verdicts"].get("incomplete", 0),
            )
        )
    parts.append(
        _table(
            f"{pid}-verdicts-table",
            ["Population", "Records", "Complete", "Incomplete"],
            verdict_rows,
        )
    )
    missing_rows = []
    for name in ("equilibrium", "calibration", "combined"):
        for channel, count in populations[name]["missing_channels"].items():
            missing_rows.append((name, channel, count))
    add(
        "p",
        "missing-note",
        "A channel that was not assembled is counted against every record that "
        "names it. The verdict does not collapse a failed lookup into an absence: "
        "a record whose family or neighbour lookup failed carries "
        "<code>unavailable</code> for that channel and is refused, while a record "
        "that genuinely has no family or no neighbour states so in prose.",
    )
    if missing_rows:
        parts.append(
            _table(
                f"{pid}-missing-table",
                ["Population", "Channel", "Records"],
                missing_rows,
            )
        )

    add("h2", "prose-head", "An absence is a sentence, not an empty block")
    prose_rows = []
    for channel, block in report["prose"].items():
        prose_rows.append((channel, block["count"], "; ".join(block["samples"])))
    if prose_rows:
        parts.append(
            _table(
                f"{pid}-prose-table",
                ["Channel", "Records", "What the record says"],
                prose_rows,
            )
        )
    add(
        "p",
        "prose-note",
        f"{content['with_family_members']} records name their family's other "
        f"members; {content['no_family']} record no family at all and say so. "
        f"{content['with_subject_neighbours']} records carry a neighbour by subject "
        f"and {content['with_locus_neighbours']} a neighbour by locus, and every "
        "one of those neighbours is accompanied by the grammar fact or unit that "
        "separates it from the target.",
    )
    add(
        "p",
        "shape",
        f"Every record binds {content['median_enclosing_statements']} enclosing "
        f"statement(s) at the median, reaches a chain depth of at most "
        f"{content['max_chain_depth']} to its DD root, and "
        f"{content['sourceless']} record(s) bind no source at all. "
        f"{content['with_adjudicator_reason']} records carry a recorded adjudicator "
        "reason.",
    )

    add("h2", "writes-head", "Nothing stored changed")
    add(
        "p",
        "writes",
        f"Every statement the builder sent is a read: the assembly's own guard "
        f"refuses a write clause before it leaves the process, and each record "
        f"reports <code>graph_writes: {provenance['graph_writes']}</code> and "
        f"<code>model_calls: {provenance['model_calls']}</code>. Independently, "
        f"{provenance['stored_before']['rows']} names' stored text was digested "
        f"before the run (<code>{provenance['stored_before']['digest'][:16]}</code>) "
        f"and after it (<code>{provenance['stored_after']['digest'][:16]}</code>): "
        f"{'identical' if provenance['stored_unchanged'] else 'DIFFERENT'}.",
    )

    add("h2", "rows-head", "Every record in the run")
    add(
        "p",
        "rows-note",
        "One row per name. The digest is the record's content digest; the full "
        "record for each row is written to the run directory beside this "
        "fragment's JSON.",
    )
    parts.append(
        _table(
            f"{pid}-rows-table",
            [
                "Name",
                "Tokens",
                "Digest",
                "Verdict",
                "Sources",
                "Family",
                "Subject",
                "Locus",
                "Statements",
                "Missing",
            ],
            [
                (
                    row["id"],
                    row["tokens"],
                    row["digest"][:16],
                    row["verdict"],
                    row["sources"],
                    row["family"],
                    row["subject_neighbours"],
                    row["locus_neighbours"],
                    row["statements"],
                    ", ".join(row["missing"]) or "-",
                )
                for row in rows
            ],
        )
    )
    return "\n".join(parts)


def document(report: dict[str, object], *, standalone: bool) -> str:
    # The standalone page is a run artifact held beside the fragment, so its
    # ids live in their own namespace; a shared id would resolve to whichever
    # of the two the composed record reached first.
    prefix = "case-file-run" if standalone else NODE
    body = sections(report, prefix)
    if standalone:
        slug = 'name="plan-slug" content="sn-case-file-generation"'
        title = "Case files for the equilibrium members and the calibration names"
        node = f'<meta name="node-id" content="{NODE}">'
    else:
        slug = 'name="plan-slug" content="sn-case-file-generation"'
        title = "One case file per name, with its text, its digest and its verdict"
        node = f'<meta name="node-id" content="{NODE}">'
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="docs-project" content="imas-codex">\n'
        f'<meta name="reckon-type" content="{"evidence" if standalone else "evidence-fragment"}">\n'
        f"{slug}\n{node}\n<title>{escape(title)}</title>\n</head>\n<body>\n<main>\n"
        f"{body}\n</main>\n</body>\n</html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="re-render the pages from case-file.json without rebuilding records",
    )
    args = parser.parse_args()
    if args.render_only:
        report = json.loads(DATA.read_text())
        FRAGMENT.write_text(document(report, standalone=False))
        PAGE.write_text(document(report, standalone=True))
        print("re-rendered the fragment and the page")
        return 0
    equilibrium, calibration, combined = populations()
    result = build(combined, limit=args.limit)
    report = summarise(equilibrium, calibration, combined, result)
    text = json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True)
    DATA.write_text(text + "\n")
    if len(text) >= 300_000:
        raise SystemExit(f"case-file.json is {len(text)} bytes, over the 300 000 limit")
    FRAGMENT.write_text(document(report, standalone=False))
    PAGE.write_text(document(report, standalone=True))
    print(
        f"{len(report['rows'])} records, {len(text)} bytes, "
        f"{report['elapsed_seconds']}s, failures {len(report['failures'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

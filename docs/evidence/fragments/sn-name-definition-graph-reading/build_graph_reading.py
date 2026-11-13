#!/usr/bin/env python3
"""Anchor the name-against-definition reading to the graph's export candidates.

The graph is the only source. The candidate set is the population the release
export would carry, read by calling the export's own selection rather than by
restating its rule. Labels the two readers recorded against the text of an
earlier export survive only where the graph's text is unchanged; every other
candidate is drawn again from graph text for a further blind reading.

Modes:
  --read    read the live graph and write the candidate set and record cache
  --build   reconcile the labels and draw the further sheets from the cache
  --check   verify the committed artifacts and run the relabel negative control
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FRAGMENTS = REPO / "docs/evidence/fragments"
CATALOG_READING = FRAGMENTS / "sn-name-definition-catalog-reading"
SECOND_READING = FRAGMENTS / "sn-name-definition-second-reading"
FRESH_LABELS = FRAGMENTS / "sn-name-definition-fresh-labels"
AGREEMENT = CATALOG_READING / "two-reader-agreement.json"
CANDIDATE_SET = HERE / "candidate-set.json"
RECONCILIATION = HERE / "reconciliation.json"
DRAW = HERE / "sheet-draw.json"
RECORDS = Path(
    os.environ.get(
        "GRAPH_READING_RECORDS",
        Path.home() / ".cache/imas-codex-graph-reading/records.json",
    )
)
FIELDS = ("name", "unit", "description", "documentation")
PART_SIZE = 110
SIZE_LIMIT = 300_000
STATUSES = ("label-carried", "text-changed", "never-read")
VERDICTS = ("agreeing", "mismatched", "disputed")
EARLIER_SHEETS = [
    *(SECOND_READING / f"blind-sheet-part-{number}.json" for number in range(1, 4)),
    SECOND_READING / "blind-sheet.json",
    *(FRESH_LABELS / f"sheet-part-{number}.json" for number in range(1, 4)),
    *(CATALOG_READING / f"sheet-part-{number:02d}.json" for number in range(1, 13)),
]
SOURCE_NAMES = {
    SECOND_READING: "second-reading",
    FRESH_LABELS: "fresh-labels",
    CATALOG_READING: "catalog-sheets",
}
FORBIDDEN = (
    "docs/evidence/fragments/sn-definitions-by-construction/",
    "docs/evidence/fragments/sn-name-definition-second-reading/",
    "docs/evidence/fragments/sn-name-definition-fresh-labels/",
    "docs/evidence/fragments/sn-name-definition-catalog-reading/",
    "`docs/plans/`",
    "`~/.cache/`",
    "any other worktree",
    "any tool that returns review results for\nstandard names",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise(value):
    return " ".join((value or "").split())


def field_digest(value):
    return hashlib.sha256(normalise(value).encode()).hexdigest()[:16]


def digest_of_fields(shas):
    joined = "|".join(shas[field] for field in FIELDS)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def text_digest(item):
    return digest_of_fields({field: field_digest(item[field]) for field in FIELDS})


def verdict_of(reader_a, reader_b):
    if reader_a != "agrees" and reader_b != "agrees":
        return "mismatched"
    if reader_a == "agrees" and reader_b == "agrees":
        return "agreeing"
    return "disputed"


def utcnow():
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def earlier_items():
    """Every sheet item both readers saw, keyed by name."""
    items = {}
    sources = {}
    for path in EARLIER_SHEETS:
        source = SOURCE_NAMES[path.parent]
        for item in read_json(path)["items"]:
            row = {field: item[field] for field in FIELDS}
            previous = items.get(item["name"])
            if previous is not None:
                assert previous == row, (
                    f"name read under two different texts: {item['name']}"
                )
            items[item["name"]] = row
            sources.setdefault(item["name"], source)
    return items, sources


def classes_block():
    blocks = [
        read_json(path)["classes"]
        for path in (
            SECOND_READING / "blind-sheet.json",
            FRESH_LABELS / "sheet-part-1.json",
            CATALOG_READING / "sheet-part-01.json",
        )
    ]
    assert blocks[0] == blocks[1] == blocks[2], "earlier sheets disagree on classes"
    return blocks[0]


def read_graph():
    """Call the release export's own candidate selection, then read graph text."""
    from imas_codex.standard_names.export import (
        _classify_export_population,
        _fetch_export_population,
        _graph_node_to_entry_dict,
    )

    started = utcnow()
    origin = dt.datetime.now(dt.UTC)
    population = _fetch_export_population(batch=None, require_docs_review=True)
    candidates, exclusions = _classify_export_population(
        population, domain=None, names_only=False
    )
    elapsed = (dt.datetime.now(dt.UTC) - origin).total_seconds()

    records = {}
    for candidate in candidates:
        entry = _graph_node_to_entry_dict(candidate)
        records[candidate["id"]] = {field: entry[field] for field in FIELDS}
    assert len(records) == len(candidates), "candidate ids are not unique"
    assert records, "the export selection returned no candidates"

    read_items, _sources = earlier_items()
    agreement = read_json(AGREEMENT)
    read_names = [row["name"] for row in agreement["rows"]]
    assert set(read_items) == set(read_names), "agreement and sheets disagree on names"
    retired_names = sorted(set(read_names) - set(records))

    # Positive control: a name known to be a candidate must come back from the
    # same lookup the retired names use, so an empty result is not silent.
    control = sorted(records)[0]
    retired = read_retired_states([control, *retired_names])
    assert control in retired, "positive control missing from the name lookup"

    CANDIDATE_SET.write_text(
        json.dumps(
            {
                "schema": "sn-candidate-set",
                "rule": {
                    "module": "imas_codex/standard_names/export.py",
                    "called": (
                        "_classify_export_population("
                        "_fetch_export_population(batch=None, "
                        "require_docs_review=True), domain=None, names_only=False)"
                    ),
                    "used_by": "run_export",
                    "selection": (
                        "name_stage in [accepted, approved]; not superseded by "
                        "status or name_stage; valid and dated validation; no "
                        "name review quorum shortfall; reachable winning docs "
                        "review group; docs_stage accepted; no docs quorum "
                        "shortfall; a producing source reaches the name"
                    ),
                },
                "read_at": started,
                "read_seconds": round(elapsed, 3),
                "graph_read": {
                    "population": len(population),
                    "eligible": len(candidates),
                    "excluded": len(exclusions),
                },
                "count": len(candidates),
                "names": sorted(records),
            },
            indent=1,
        )
        + "\n"
    )
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text(
        json.dumps(
            {"read_at": started, "records": records, "retired": retired},
            ensure_ascii=False,
        )
        + "\n"
    )
    print(
        f"{len(population)} population identities; {len(candidates)} eligible "
        f"candidates; {len(exclusions)} excluded before gates; read in "
        f"{elapsed:.1f}s; {len(retired_names)} read names retired; "
        f"{sum(1 for name in retired_names if name not in retired)} absent"
    )


def read_retired_states(names):
    """What the graph says of read names, keyed by name; absent names are omitted."""
    if not names:
        return {}
    from imas_codex.graph.client import GraphClient

    cypher = """
    UNWIND $names AS id
    MATCH (sn:StandardName {id: id})
    OPTIONAL MATCH (successor:StandardName)-[:REFINED_FROM]->(sn)
    RETURN id,
           sn.name_stage AS name_stage,
           sn.status AS status,
           collect(DISTINCT successor.id) AS successors
    """
    with GraphClient() as gc:
        rows = gc.query(cypher, names=list(names))
    states = {}
    for row in rows or []:
        states[row["id"]] = {
            "name_stage": row.get("name_stage"),
            "status": row.get("status"),
            "superseded_by": sorted(row.get("successors") or []),
        }
    return states


# ── reconcile and draw ──────────────────────────────────────────────────────


def reconcile():
    candidate_set = read_json(CANDIDATE_SET)
    cache = read_json(RECORDS)
    records = cache["records"]
    retired = cache["retired"]
    rows = {row["name"]: row for row in read_json(AGREEMENT)["rows"]}
    items, sources = earlier_items()

    assert candidate_set["names"] == sorted(records), "candidate set and cache disagree"
    assert set(items) == set(rows), "agreement and sheets disagree on names"

    tables = {status: {} for status in STATUSES}
    totals = dict.fromkeys(STATUSES, 0)
    verdict_counts = dict.fromkeys(VERDICTS, 0)
    for name in candidate_set["names"]:
        text = records[name]
        sha = text_digest(text)
        row = rows.get(name)
        if row is None:
            tables["never-read"][name] = sha
            totals["never-read"] += 1
            continue
        entry = {
            "a": row["reader_a"],
            "b": row["reader_b"],
            "v": row["verdict"],
            "s": sha,
        }
        if row["seat_flagged"]:
            entry["f"] = 1
        if sha == text_digest(items[name]):
            tables["label-carried"][name] = entry
            verdict_counts[row["verdict"]] += 1
            totals["label-carried"] += 1
        else:
            shas = {field: field_digest(text[field]) for field in FIELDS}
            entry["d"] = [
                field
                for field in FIELDS
                if shas[field] != field_digest(items[name][field])
            ]
            entry["fs"] = shas
            tables["text-changed"][name] = entry
            totals["text-changed"] += 1
    retired_rows = []
    for name in sorted(set(rows) - set(candidate_set["names"])):
        state = retired.get(name)
        if state is None:
            graph = {"absent": True}
        else:
            graph = {"absent": False, **state}
        retired_rows.append({"name": name, "read_from": sources[name], "from": graph})
    assert sum(totals.values()) == len(candidate_set["names"])

    return {
        "schema": "sn-reconciliation",
        "candidate_count": candidate_set["count"],
        "graph_read_at": candidate_set["read_at"],
        "comparison": "whitespace normalisation only",
        "fields": {
            "shape": (
                "candidates maps each status to a name-keyed table; a never-read "
                "value is the graph text digest alone"
            ),
            "a": "reader a's class",
            "b": "reader b's class",
            "v": "verdict",
            "f": "1 when the chat seat flagged the name",
            "s": "digest of the graph's name|unit|description|documentation",
            "d": "fields whose graph text differs from what the readers saw",
            "fs": "per-field graph text digests, text-changed rows only",
        },
        "totals": totals,
        "verdict_counts_label_carried": verdict_counts,
        "second_readings": {
            "text_changed": totals["text-changed"],
            "never_read": totals["never-read"],
        },
        "candidates": tables,
        "no_longer_candidates": {
            "count": len(retired_rows),
            "absent": sum(1 for row in retired_rows if row["from"]["absent"]),
            "superseded": sum(
                1 for row in retired_rows if row["from"].get("superseded_by")
            ),
            "names": retired_rows,
        },
    }


def third_reading_names(recon):
    """The names item 3 sends to a further blind reading, split by reader count."""
    tables = recon["candidates"]
    two = list(tables["text-changed"]) + list(tables["never-read"])
    one = [
        name
        for name, entry in tables["label-carried"].items()
        if entry["v"] == "disputed" or entry.get("f")
    ]
    return two, one


def shuffled_parts(two, one, seed_two, seed_one):
    groups = [
        ("two-reader", sorted(two), seed_two),
        ("one-reader", sorted(one), seed_one),
    ]
    parts = []
    number = 0
    for group, names, seed in groups:
        pool = list(names)
        random.Random(seed).shuffle(pool)
        for start in range(0, len(pool), PART_SIZE):
            number += 1
            parts.append((number, group, seed, pool[start : start + PART_SIZE]))
    return parts


def draw_sheets(recon, records, seed_two, seed_one):
    classes = classes_block()
    sheets = []
    for _number, _group, _seed, names in shuffled_parts(
        *third_reading_names(recon), seed_two, seed_one
    ):
        items = [
            {"id": f"item-{offset:04d}", **{f: records[name][f] for f in FIELDS}}
            for offset, name in enumerate(names, 1)
        ]
        sheets.append({"classes": classes, "items": items})
    return sheets


def receipt_for(recon, parts, seed_two, seed_one):
    return {
        "seed_two_reader": seed_two,
        "seed_one_reader": seed_one,
        "randomization": (
            "Python random.Random; each group's names sorted then shuffled once "
            "with its own recorded seed, then cut into consecutive parts of at "
            "most 110"
        ),
        "candidate_count": recon["candidate_count"],
        "two_reader_names": len(third_reading_names(recon)[0]),
        "one_reader_names": len(third_reading_names(recon)[1]),
        "part_count": len(parts),
        "input_sha256": {
            "candidate-set.json": digest(CANDIDATE_SET),
            "reconciliation.json": digest(RECONCILIATION),
            "two-reader-agreement.json": digest(AGREEMENT),
        },
        "parts": [
            {
                "part": number,
                "group": group,
                "seed": seed,
                "first_id": f"item-{1:04d}" if names else None,
                "last_id": f"item-{len(names):04d}" if names else None,
                "items": len(names),
                "names": names,
            }
            for number, group, seed, names in parts
        ],
    }


BRIEF_TEMPLATE = CATALOG_READING / "reader-a-part-01.md"


def brief(reader, part, count, first_id):
    text = BRIEF_TEMPLATE.read_text()
    text = text.replace(
        "a blind reading of 103 standard names",
        f"a blind reading of {count} standard names",
    )
    text = text.replace(
        "docs/evidence/fragments/sn-name-definition-catalog-reading/sheet-part-01.json",
        f"docs/evidence/fragments/sn-name-definition-graph-reading/sheet-part-{part:02d}.json",
    )
    text = text.replace(
        "every file in `docs/evidence/fragments/sn-name-definition-catalog-reading/` except your own `sheet-part-01.json`",
        "every file in `docs/evidence/fragments/sn-name-definition-catalog-reading/`, "
        "every file in `docs/evidence/fragments/sn-name-definition-graph-reading/` "
        f"except your own `sheet-part-{part:02d}.json`",
    )
    text = text.replace(
        "docs/evidence/fragments/sn-name-definition-catalog-reading/labels-a-part-01.json",
        f"docs/evidence/fragments/sn-name-definition-graph-reading/labels-{reader}-part-{part:02d}.json",
    )
    text = text.replace('"id": "item-0001"', f'"id": "{first_id}"')
    return text


def build():
    recon = reconcile()
    RECONCILIATION.write_text(encoded(recon))
    records = read_json(RECORDS)["records"]
    seed_two, seed_one = 20261113, 20261114
    parts = shuffled_parts(*third_reading_names(recon), seed_two, seed_one)
    sheets = draw_sheets(recon, records, seed_two, seed_one)
    receipt = receipt_for(recon, parts, seed_two, seed_one)
    verify(recon, sheets, receipt)
    for number, sheet in enumerate(sheets, 1):
        content = encoded(sheet)
        assert len(content.encode()) <= SIZE_LIMIT, f"sheet {number} is oversize"
        (HERE / f"sheet-part-{number:02d}.json").write_text(content)
    DRAW.write_text(json.dumps(receipt, indent=1) + "\n")
    for number, group, _seed, names in parts:
        first_id = f"item-{1:04d}" if names else "item-0001"
        for reader in ("c", "d") if group == "two-reader" else ("c",):
            (HERE / f"reader-{reader}-part-{number:02d}.md").write_text(
                brief(reader, number, len(names), first_id)
            )
    print(
        f"{recon['candidate_count']} candidates; "
        + ", ".join(f"{status} {recon['totals'][status]}" for status in STATUSES)
    )
    print(f"{len(parts)} parts; {receipt['two_reader_names']} names need two readings")


def entry_digest(entry):
    return entry if isinstance(entry, str) else entry["s"]


def verify_statuses(recon, items, rows):
    """Recompute every status from the committed text digests and refuse drift."""
    tables = recon["candidates"]
    assert set(tables) == set(STATUSES), "candidate tables are incomplete"
    counts = dict.fromkeys(STATUSES, 0)
    seen = set()
    for status, table in tables.items():
        for name, entry in table.items():
            assert name not in seen, f"a candidate appears more than once: {name}"
            seen.add(name)
            item = items.get(name)
            sha = entry_digest(entry)
            if item is None:
                expected = "never-read"
            elif text_digest(item) == sha:
                expected = "label-carried"
            else:
                expected = "text-changed"
            assert status == expected, (
                f"refused: {name} is recorded {status} but its graph text "
                f"differs from what the readers saw, so it is {expected}"
            )
            if expected != "never-read":
                row = rows[name]
                assert entry["a"] == row["reader_a"], f"reader a drifted: {name}"
                assert entry["b"] == row["reader_b"], f"reader b drifted: {name}"
                assert entry["v"] == verdict_of(row["reader_a"], row["reader_b"]), (
                    f"verdict drifted: {name}"
                )
                assert bool(entry.get("f")) == bool(row["seat_flagged"]), (
                    f"seat flag drifted: {name}"
                )
            if expected == "text-changed":
                shas = entry["fs"]
                assert set(shas) == set(FIELDS), f"field digests incomplete: {name}"
                assert digest_of_fields(shas) == sha, (
                    f"field digests do not compose the text digest: {name}"
                )
                differing = [f for f in FIELDS if shas[f] != field_digest(item[f])]
                assert entry["d"] == differing, f"differing fields misrecorded: {name}"
            counts[expected] += 1
    return counts


def verify(recon, sheets, receipt):
    """Refuse an artifact set that does not meet the node's contract."""
    items, _sources = earlier_items()
    rows = {row["name"]: row for row in read_json(AGREEMENT)["rows"]}
    counts = verify_statuses(recon, items, rows)
    assert recon["totals"] == counts, "status totals disagree with the entries"
    assert sum(counts.values()) == recon["candidate_count"], (
        "status totals do not sum to the candidate count"
    )
    verdicts = dict.fromkeys(VERDICTS, 0)
    for entry in recon["candidates"]["label-carried"].values():
        verdicts[entry["v"]] += 1
    assert recon["verdict_counts_label_carried"] == verdicts, (
        "verdict counts disagree with the carried labels"
    )

    names = {name for table in recon["candidates"].values() for name in table}
    leftover = set(rows) - names
    retired = recon["no_longer_candidates"]["names"]
    assert {row["name"] for row in retired} == leftover, (
        "retired names do not match the read names outside the candidate set"
    )
    assert recon["no_longer_candidates"]["count"] == len(leftover), (
        "retired count disagrees with its entries"
    )

    two, one = third_reading_names(recon)
    expected = set(two) | set(one)
    assert len(expected) == len(two) + len(one), (
        "a name needs both one and two readings"
    )
    drawn = [item["name"] for sheet in sheets for item in sheet["items"]]
    assert len(set(drawn)) == len(drawn), "sheet names are not distinct"
    assert set(drawn) == expected, "sheets do not hold exactly the names item 3 lists"
    assert len(sheets) == receipt["part_count"], (
        "sheet count disagrees with the receipt"
    )
    for sheet in sheets:
        assert len(sheet["items"]) <= PART_SIZE, "a sheet holds more than 110 names"
        assert sheet["classes"] == classes_block(), "sheet classes differ"
        for item in sheet["items"]:
            assert set(item) == {"id", *FIELDS}, "sheet item shape differs"
    for part in receipt["parts"]:
        assert part["items"] <= PART_SIZE, "a part holds more than 110 names"


def check():
    """Verify the committed artifacts without the graph or the record cache."""
    candidate_set = read_json(CANDIDATE_SET)
    recon = read_json(RECONCILIATION)
    receipt = read_json(DRAW)
    items, _sources = earlier_items()
    rows = {row["name"]: row for row in read_json(AGREEMENT)["rows"]}

    assert candidate_set["rule"]["module"] == "imas_codex/standard_names/export.py"
    assert candidate_set["read_at"].endswith("Z"), "graph read time not recorded"
    assert candidate_set["names"] == sorted(set(candidate_set["names"])), (
        "candidate names are not distinct and sorted"
    )
    assert candidate_set["count"] == len(candidate_set["names"])
    assert candidate_set["count"] > 0, "candidate set is empty"
    assert recon["candidate_count"] == candidate_set["count"], "candidate counts differ"
    assert {name for table in recon["candidates"].values() for name in table} == set(
        candidate_set["names"]
    ), "reconciliation does not cover the candidate set exactly"

    print(f"candidate count: {candidate_set['count']}")
    print(
        "candidate rule: "
        f"{candidate_set['rule']['module']} {candidate_set['rule']['called']} "
        f"(read {candidate_set['read_at']}, "
        f"{candidate_set['read_seconds']}s, "
        f"{candidate_set['graph_read']['eligible']} eligible of "
        f"{candidate_set['graph_read']['population']} population)"
    )

    counts = verify_statuses(recon, items, rows)
    print(
        "every candidate in exactly one status: "
        + ", ".join(f"{status} {counts[status]}" for status in STATUSES)
    )
    total = sum(counts.values())
    assert total == recon["candidate_count"], "status totals do not sum to the count"
    assert recon["totals"] == counts, "recorded totals disagree with the entries"
    print(f"status totals: {total} = {recon['totals']}")
    print(
        "verdict counts among label-carried: "
        + ", ".join(
            f"{verdict} {count}"
            for verdict, count in recon["verdict_counts_label_carried"].items()
        )
    )

    parts = shuffled_parts(
        *third_reading_names(recon),
        receipt["seed_two_reader"],
        receipt["seed_one_reader"],
    )
    assert receipt == receipt_for(
        recon, parts, receipt["seed_two_reader"], receipt["seed_one_reader"]
    ), "sheet receipt does not match the committed inputs and seeds"
    sheets = [read_json(HERE / f"sheet-part-{part[0]:02d}.json") for part in parts]
    verify(recon, sheets, receipt)
    digests = {
        name: entry_digest(entry)
        for table in recon["candidates"].values()
        for name, entry in table.items()
    }
    seen = 0
    for (number, _group, _seed, names), sheet in zip(parts, sheets, strict=True):
        assert [item["name"] for item in sheet["items"]] == names, (
            f"sheet {number} does not hold the drawn names in the drawn order"
        )
        for offset, item in enumerate(sheet["items"], 1):
            assert item["id"] == f"item-{offset:04d}", (
                f"item id differs: sheet {number}"
            )
            assert text_digest(item) == digests[item["name"]], (
                f"sheet {number} text differs from the graph text: {item['name']}"
            )
            seen += 1
        assert len(sheet["items"]) <= PART_SIZE, f"sheet {number} is over 110 names"
    print(
        f"sheets hold {seen} names, each once, in {len(parts)} parts, largest "
        f"{max(len(part[3]) for part in parts)} names"
    )

    verify_briefs(parts)
    negative_control(recon, items, rows)
    print("refused nothing beyond the negative control; every artifact is consistent")


def verify_briefs(parts):
    for number, group, _seed, names in parts:
        readers = ["c", "d"] if group == "two-reader" else ["c"]
        for reader in readers:
            path = HERE / f"reader-{reader}-part-{number:02d}.md"
            assert path.exists(), f"missing reader brief: {path.name}"
            text = path.read_text()
            assert text == brief(reader, number, len(names), "item-0001"), (
                f"brief differs from its template: {path.name}"
            )
            mine = f"labels-{reader}-part-{number:02d}.json"
            assert mine in text, f"brief does not name its own output: {path.name}"
            assert f"sheet-part-{number:02d}.json" in text, (
                f"brief does not name its own sheet: {path.name}"
            )
            other = "d" if reader == "c" else "c"
            assert f"labels-{other}-part-{number:02d}" not in text, (
                f"brief names another reader's output: {path.name}"
            )
            for forbidden in FORBIDDEN:
                assert forbidden in text, (
                    f"brief omits an exclusion ({forbidden}): {path.name}"
                )
            lowered = text.lower()
            for phrase in ("second reading", "third reading", "earlier reading"):
                assert phrase not in lowered, (
                    f"brief says other readings exist: {path.name}"
                )
        extra = HERE / f"reader-d-part-{number:02d}.md"
        if group == "one-reader":
            assert not extra.exists(), (
                f"one-reader part carries a second brief: {extra.name}"
            )
    print(
        f"{sum(2 if group == 'two-reader' else 1 for _n, group, _s, _names in parts)} "
        "reader briefs, one per required reading, each naming one distinct output"
    )


def negative_control(recon, items, rows):
    """Show the status guard refusing a text-changed name relabelled as carried."""
    changed = recon["candidates"]["text-changed"]
    assert changed, "no text-changed name exists, so the guard cannot be shown to fire"
    mutated = copy.deepcopy(recon)
    victim, entry = next(iter(changed.items()))
    mutated["candidates"]["label-carried"][victim] = entry
    del mutated["candidates"]["text-changed"][victim]
    try:
        verify_statuses(mutated, items, rows)
    except AssertionError as exc:
        print(f"negative control refused a text-changed name relabelled carried: {exc}")
    else:
        raise AssertionError("the guard accepted a relabelled text-changed name")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read", action="store_true", help="read the live graph")
    mode.add_argument("--build", action="store_true", help="reconcile and draw")
    mode.add_argument("--check", action="store_true", help="verify committed artifacts")
    args = parser.parse_args()
    if args.read:
        read_graph()
    elif args.build:
        build()
    else:
        check()


if __name__ == "__main__":
    main()

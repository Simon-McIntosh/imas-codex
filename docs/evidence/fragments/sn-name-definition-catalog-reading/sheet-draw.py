#!/usr/bin/env python3
"""Draw reproducible blind sheets covering every published catalog name not yet read."""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SECOND_READING = REPO / "docs/evidence/fragments/sn-name-definition-second-reading"
FRESH = REPO / "docs/evidence/fragments/sn-name-definition-fresh-labels"
CATALOG = Path("/home/ITER/mcintos/Code/imas-standard-names-catalog")
FIELDS = ("name", "unit", "description", "documentation")
PART_SIZE = 110
READERS = ("a", "b")


def read_json(path):
    return json.loads(path.read_text())


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_names():
    """Every name two readers have already read, from both earlier fragments."""
    rows = read_json(SECOND_READING / "two-reader-labels.json")["rows"]
    second = {row["name"] for row in rows}
    assert len(second) == len(rows) == 381, "expected 381 second-reading names"
    fresh = set()
    for part in (1, 2, 3):
        sheet = read_json(FRESH / f"sheet-part-{part}.json")
        fresh.update(item["name"] for item in sheet["items"])
    assert len(fresh) == 360, "expected 360 fresh-label names"
    return second | fresh


def inputs(catalog):
    revision = subprocess.check_output(
        ["git", "-C", str(catalog), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(catalog), "status", "--porcelain", "--", "standard_names"],
        text=True,
    )
    assert not dirty, "catalog standard_names must match its recorded revision"
    read = read_names()
    classes = read_json(FRESH / "sheet-part-1.json")["classes"]
    assert classes == read_json(SECOND_READING / "blind-sheet.json")["classes"]
    domains = {}
    hashes = {}
    seen = set()
    for path in sorted((catalog / "standard_names").glob("*.yml")):
        rows = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        assert isinstance(rows, list) and rows, f"invalid domain: {path.name}"
        for row in rows:
            assert all(isinstance(row[key], str) for key in FIELDS), row["name"]
            assert row["name"] not in seen, f"duplicate catalog name: {row['name']}"
            seen.add(row["name"])
        domains[path.name] = sorted(rows, key=lambda row: row["name"])
        hashes[path.name] = digest(path)
    assert read <= seen, "already-read positive control: names missing from catalog"
    return revision, read, classes, domains, hashes


def part_sizes(total, count):
    base, remainder = divmod(total, count)
    return [base + 1 if index < remainder else base for index in range(count)]


def draw(catalog, seed):
    revision, read, classes, domains, hashes = inputs(catalog)
    eligible = [
        (domain, row)
        for domain in sorted(domains)
        for row in domains[domain]
        if row["name"] not in read
    ]
    assert eligible, "no eligible names"
    rng = random.Random(seed)
    rng.shuffle(eligible)
    part_count = (len(eligible) + PART_SIZE - 1) // PART_SIZE
    sizes = part_sizes(len(eligible), part_count)
    sheets = []
    first = 0
    for size in sizes:
        chunk = eligible[first : first + size]
        items = [
            {"id": f"item-{first + offset:04d}", **{key: row[key] for key in FIELDS}}
            for offset, (_domain, row) in enumerate(chunk, 1)
        ]
        sheets.append({"classes": classes, "items": items})
        part_domains = {}
        for domain, _row in chunk:
            part_domains[domain] = part_domains.get(domain, 0) + 1
        first += size
    receipt = {
        "seed": seed,
        "catalog_revision": revision,
        "catalog_path": str(catalog),
        "randomization": (
            "Python random.Random; catalog read as sorted domain filenames and "
            "names sorted inside each domain; every eligible name pooled and "
            "shuffled once, then cut into consecutive parts of at most "
            f"{PART_SIZE}"
        ),
        "published": sum(len(rows) for rows in domains.values()),
        "excluded_as_already_read": len(read),
        "eligible": len(eligible),
        "part_count": part_count,
        "names_per_part": sizes,
        "parts": [
            {
                "part": number,
                "first_id": sheet["items"][0]["id"],
                "last_id": sheet["items"][-1]["id"],
                "items": len(sheet["items"]),
                "domains": part_domains,
            }
            for number, sheet in enumerate(sheets, 1)
        ],
        "input_sha256": {
            "two-reader-labels.json": digest(SECOND_READING / "two-reader-labels.json"),
            "sheet-part-1.json": digest(FRESH / "sheet-part-1.json"),
            "sheet-part-2.json": digest(FRESH / "sheet-part-2.json"),
            "sheet-part-3.json": digest(FRESH / "sheet-part-3.json"),
            "catalog_domains": hashes,
        },
    }
    return sheets, receipt, read, eligible


def validate(sheets, read, classes, eligible_names):
    items = [item for sheet in sheets for item in sheet["items"]]
    names = [item["name"] for item in items]
    ids = [item["id"] for item in items]
    assert all(len(sheet["items"]) <= PART_SIZE for sheet in sheets), "part over 110"
    assert len(items) == len(eligible_names), "item count differs from eligible count"
    assert len(set(names)) == len(names), "duplicate name across sheets"
    assert len(set(ids)) == len(ids), "duplicate item id across sheets"
    assert not set(names) & read, "previously read name present"
    assert set(names) <= set(eligible_names), "item outside the eligible names"
    assert all(sheet["classes"] == classes for sheet in sheets), "classes differ"
    assert all(set(item) == {"id", *FIELDS} for item in items), "item shape differs"
    assert all(sheet["items"] for sheet in sheets), "empty part"


def brief(reader, part, count, first_id):
    text = (FRESH / "reader-a-part-1.md").read_text()
    old_header = "a blind reading of 120 standard names"
    text = text.replace(old_header, f"a blind reading of {count} standard names")
    text = text.replace(
        "docs/evidence/fragments/sn-name-definition-fresh-labels/sheet-part-1.json",
        f"docs/evidence/fragments/sn-name-definition-catalog-reading/sheet-part-{part:02d}.json",
    )
    text = text.replace(
        "docs/evidence/fragments/sn-name-definition-fresh-labels/labels-a-part-1.json",
        f"docs/evidence/fragments/sn-name-definition-catalog-reading/labels-{reader}-part-{part:02d}.json",
    )
    text = text.replace(
        "every file in `docs/evidence/fragments/sn-name-definition-fresh-labels/` except your own `sheet-part-1.json`, `docs/plans/`,",
        "`docs/evidence/fragments/sn-name-definition-fresh-labels/`,\n"
        "every file in `docs/evidence/fragments/sn-name-definition-catalog-reading/` "
        f"except your own `sheet-part-{part:02d}.json`, `docs/plans/`,",
    )
    text = text.replace('"id": "item-001"', f'"id": "{first_id}"')
    return text


def check(catalog):
    receipt = read_json(HERE / "sheet-draw.json")
    expected, recomputed, read, eligible = draw(catalog, receipt["seed"])
    assert receipt == recomputed, "draw receipt does not match inputs and seed"
    paths = [
        HERE / f"sheet-part-{number:02d}.json"
        for number in range(1, receipt["part_count"] + 1)
    ]
    sheets = [read_json(path) for path in paths]
    classes = read_json(FRESH / "sheet-part-1.json")["classes"]
    validate(sheets, read, classes, {row["name"] for _domain, row in eligible})
    for path, sheet in zip(paths, expected, strict=True):
        assert path.read_text() == encoded(sheet), f"draw differs: {path.name}"
        assert path.stat().st_size <= 300_000, f"oversize repository data: {path.name}"
    outputs = set()
    for number, sheet in enumerate(sheets, 1):
        count = len(sheet["items"])
        first_id = sheet["items"][0]["id"]
        for reader in READERS:
            path = HERE / f"reader-{reader}-part-{number:02d}.md"
            text = path.read_text()
            assert text == brief(reader, number, count, first_id), (
                f"brief differs from adapted template: {path.name}"
            )
            mine = f"labels-{reader}-part-{number:02d}.json"
            found = [
                name
                for name in ("labels-a-part-", "labels-b-part-")
                if name in text
            ]
            assert found == [f"labels-{reader}-part-"], "wrong output binding"
            assert mine in text, "own output file not named"
            other = "b" if reader == "a" else "a"
            assert f"labels-{other}-" not in text and f"reader-{other}-" not in text, (
                "brief names another reader's file"
            )
            for forbidden in (
                "docs/evidence/fragments/sn-definitions-by-construction/",
                "docs/evidence/fragments/sn-name-definition-second-reading/",
                "docs/evidence/fragments/sn-name-definition-fresh-labels/",
                "docs/evidence/fragments/sn-name-definition-catalog-reading/` except your own",
                "`docs/plans/`",
                "`~/.cache/`",
                "any other worktree",
                "any tool that returns review results for\nstandard names",
            ):
                assert forbidden in text, f"brief omits exclusion: {forbidden}"
            assert f"sheet-part-{number:02d}.json" in text, "own sheet not named"
            outputs.add(mine)
    assert len(outputs) == 2 * receipt["part_count"], "briefs share an output file"
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / "sheets"
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("*.md", "*.py"))
        contaminated = json.loads((copy / paths[0].name).read_text())
        contaminated["items"][0]["name"] = sorted(read)[0]
        (copy / paths[0].name).write_text(encoded(contaminated))
        try:
            validate(
                [read_json(copy / path.name) for path in paths],
                read,
                classes,
                {row["name"] for _domain, row in eligible},
            )
        except AssertionError as exc:
            assert str(exc) == "previously read name present", str(exc)
            print(f"negative control refused a substituted already-read name: {exc}")
        else:
            raise AssertionError("exclusion guard failed to refuse a read name")
    print(
        f"{receipt['eligible']} eligible names; {sum(len(s['items']) for s in sheets)} "
        f"items; {len({i['name'] for s in sheets for i in s['items']})} distinct names; "
        f"0 names shared with the {receipt['excluded_as_already_read']} already read"
    )
    print(
        f"{receipt['part_count']} parts, largest {max(receipt['names_per_part'])} names "
        f"(limit {PART_SIZE}); class keys equal to the earlier sheets: "
        f"{', '.join(sorted(classes))}"
    )
    print(
        f"{2 * receipt['part_count']} reader briefs, each naming one distinct output file; "
        f"catalog {receipt['catalog_revision']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument(
        "--seed", type=int, help="draw seed; defaults to the existing receipt"
    )
    parser.add_argument(
        "--check", action="store_true", help="verify all artifacts without writing"
    )
    args = parser.parse_args()
    if args.check:
        check(args.catalog)
        return
    seed = args.seed
    if seed is None:
        seed = read_json(HERE / "sheet-draw.json")["seed"]
    sheets, receipt, read, eligible = draw(args.catalog, seed)
    validate(sheets, read, sheets[0]["classes"], {row["name"] for _domain, row in eligible})
    for number, sheet in enumerate(sheets, 1):
        content = encoded(sheet)
        assert len(content.encode()) <= 300_000, (
            f"part {number} exceeds repository data limit"
        )
        (HERE / f"sheet-part-{number:02d}.json").write_text(content)
    (HERE / "sheet-draw.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for number, sheet in enumerate(sheets, 1):
        count = len(sheet["items"])
        first_id = sheet["items"][0]["id"]
        for reader in READERS:
            path = HERE / f"reader-{reader}-part-{number:02d}.md"
            path.write_text(brief(reader, number, count, first_id))
    check(args.catalog)


if __name__ == "__main__":
    main()
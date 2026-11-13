#!/usr/bin/env python3
"""Draw reproducible, stratified blind sheets from published catalog entries."""

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EARLIER = REPO / "docs/evidence/fragments/sn-name-definition-second-reading"
CATALOG = Path("/home/ITER/mcintos/Code/imas-standard-names-catalog")
FIELDS = ("name", "unit", "description", "documentation")
TOTAL = 360
PART_SIZE = 120


def read_json(path):
    return json.loads(path.read_text())


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(catalog):
    revision = subprocess.check_output(
        ["git", "-C", str(catalog), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(catalog), "status", "--porcelain", "--", "standard_names"],
        text=True,
    )
    assert not dirty, "catalog standard_names must match its recorded revision"
    labelled_rows = read_json(EARLIER / "two-reader-labels.json")["rows"]
    labelled = {row["name"] for row in labelled_rows}
    assert len(labelled_rows) == len(labelled) == 381, "expected 381 labelled names"
    classes = read_json(EARLIER / "blind-sheet.json")["classes"]
    domains = {}
    hashes = {}
    seen = set()
    for path in sorted((catalog / "standard_names").glob("*.yml")):
        rows = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        assert isinstance(rows, list) and rows, f"invalid domain: {path.name}"
        for row in rows:
            assert all(isinstance(row[k], str) for k in FIELDS), row["name"]
            assert row["name"] not in seen, f"duplicate catalog name: {row['name']}"
            seen.add(row["name"])
        domains[path.name] = sorted(rows, key=lambda row: row["name"])
        hashes[path.name] = digest(path)
    assert domains and len(seen - labelled) >= TOTAL, "insufficient eligible names"
    assert labelled <= seen, (
        "labelled-name positive control: names missing from catalog"
    )
    return revision, labelled, classes, domains, hashes


def quotas(eligible):
    """Use eligible-size weights and largest remainders with a domain minimum."""
    population = sum(len(rows) for rows in eligible.values())
    allocation = {
        domain: max(int(len(rows) >= 5), TOTAL * len(rows) // population)
        for domain, rows in eligible.items()
    }
    assert sum(allocation.values()) <= TOTAL, "domain minimum exceeds draw budget"
    while sum(allocation.values()) < TOTAL:
        available = [d for d in eligible if allocation[d] < len(eligible[d])]
        domain = min(
            available,
            key=lambda d: (-(TOTAL * len(eligible[d]) - allocation[d] * population), d),
        )
        allocation[domain] += 1
    return allocation


def draw(catalog, seed):
    revision, labelled, classes, domains, hashes = inputs(catalog)
    eligible = {
        domain: [row for row in rows if row["name"] not in labelled]
        for domain, rows in domains.items()
    }
    allocation = quotas(eligible)
    rng = random.Random(seed)
    selected = []
    for domain, rows in eligible.items():
        selected.extend(rng.sample(rows, allocation[domain]))
    rng.shuffle(selected)
    items = [
        {"id": f"item-{index:03d}", **{key: row[key] for key in FIELDS}}
        for index, row in enumerate(selected, 1)
    ]
    sheets = [
        {"classes": classes, "items": items[start : start + PART_SIZE]}
        for start in range(0, TOTAL, PART_SIZE)
    ]
    receipt = {
        "seed": seed,
        "catalog_revision": revision,
        "catalog_path": str(catalog),
        "allocation": "largest remainders weighted by eligible domain size; minimum one for domains with at least five eligible names; alphabetical ties",
        "randomization": "Python random.Random; sorted domain filenames and names; sample without replacement per domain, then shuffle all selected entries",
        "labelled_names": len(labelled),
        "excluded_as_already_labelled": sum(
            row["name"] in labelled for rows in domains.values() for row in rows
        ),
        "published": sum(map(len, domains.values())),
        "eligible": sum(map(len, eligible.values())),
        "drawn": len(items),
        "input_sha256": {
            "two-reader-labels.json": digest(EARLIER / "two-reader-labels.json"),
            "blind-sheet.json": digest(EARLIER / "blind-sheet.json"),
            "reader-brief-part-1.md": digest(EARLIER / "reader-brief-part-1.md"),
            "catalog_domains": hashes,
        },
        "domains": {
            domain: {
                "published": len(rows),
                "excluded_as_already_labelled": len(rows) - len(eligible[domain]),
                "eligible": len(eligible[domain]),
                "drawn": allocation[domain],
            }
            for domain, rows in domains.items()
        },
    }
    return sheets, receipt, labelled


def validate(sheets, labelled, classes):
    items = [item for sheet in sheets for item in sheet["items"]]
    names = [item["name"] for item in items]
    ids = [item["id"] for item in items]
    assert len(sheets) == 3 and [len(s["items"]) for s in sheets] == [120] * 3
    assert len(items) == len(set(names)) == len(set(ids)) == TOTAL
    assert not set(names) & labelled, "previously labelled name present"
    assert all(s["classes"] == classes for s in sheets), "classes differ"
    assert all(set(item) == {"id", *FIELDS} for item in items), "item shape differs"


def brief(reader, part):
    template = (EARLIER / "reader-brief-part-1.md").read_text()
    start = template.index("## Input")
    template = (
        "# Brief: a blind reading of 120 standard names against their definitions\n\n"
        "Read only the sheet assigned below and make your own judgments.\n\n"
        + template[start:]
    )
    old = "docs/evidence/fragments/sn-name-definition-second-reading"
    fresh = "docs/evidence/fragments/sn-name-definition-fresh-labels"
    template = template.replace(
        f"{old}/blind-sheet-part-1.json", f"{fresh}/sheet-part-{part}.json"
    )
    template = template.replace(
        f"{old}/labels-part-1.json", f"{fresh}/labels-{reader}-part-{part}.json"
    )
    first_id = (part - 1) * PART_SIZE + 1
    template = template.replace('"id": "item-01"', f'"id": "item-{first_id:03d}"')
    template = template.replace(
        "`docs/evidence/fragments/sn-definitions-by-construction/`, every other file in this directory except your own sheet, `docs/plans/`,",
        "`docs/evidence/fragments/sn-definitions-by-construction/`,\n"
        "`docs/evidence/fragments/sn-name-definition-second-reading/`,\n"
        f"every file in `{fresh}/` except your own `sheet-part-{part}.json`, `docs/plans/`,",
    )
    template = template.replace(
        "documentation.\n\n## What to write",
        "documentation. Your output file is for writing your judgments, not an evidence source.\n\n## What to write",
    )
    return template


def check(catalog):
    recorded = read_json(HERE / "sheet-draw.json")
    expected, receipt, labelled = draw(catalog, recorded["seed"])
    sheets = [read_json(HERE / f"sheet-part-{p}.json") for p in range(1, 4)]
    classes = read_json(EARLIER / "blind-sheet.json")["classes"]
    validate(sheets, labelled, classes)
    assert recorded == receipt, "draw receipt does not match inputs and allocation"
    for part, sheet in enumerate(expected, 1):
        path = HERE / f"sheet-part-{part}.json"
        assert path.read_text() == encoded(sheet), f"draw differs: {path.name}"
        assert path.stat().st_size <= 300_000, f"oversize repository data: {path.name}"
    outputs = set()
    for reader in ("a", "b"):
        for part in range(1, 4):
            text = (HERE / f"reader-{reader}-part-{part}.md").read_text()
            assert text == brief(reader, part), "brief differs from adapted template"
            names = re.findall(r"labels-[ab]-part-[123]\.json", text)
            assert names == [f"labels-{reader}-part-{part}.json"], (
                "wrong output binding"
            )
            outputs.update(names)
    assert len(outputs) == 6
    contaminated = json.loads(json.dumps(sheets))
    contaminated[0]["items"][0]["name"] = sorted(labelled)[0]
    try:
        validate(contaminated, labelled, classes)
    except AssertionError as exc:
        assert str(exc) == "previously labelled name present", str(exc)
        print(f"negative control refused: {exc}")
    else:
        raise AssertionError("exclusion guard failed to refuse a labelled name")
    print("360 items, 360 distinct names, 0 names shared with the 381 labelled names")
    print(
        "three parts of 120; class keys equal to the earlier sheet; six reader briefs each naming a distinct output file"
    )
    print(
        "360 distinct ids; full classes equal; published text exact; seeded draw byte-identical; domain quotas exact"
    )
    print(
        f"catalog {receipt['catalog_revision']}; {receipt['eligible']} eligible; {receipt['excluded_as_already_labelled']} excluded"
    )
    print("output files: " + ", ".join(sorted(outputs)))


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
    sheets, receipt, labelled = draw(args.catalog, seed)
    validate(sheets, labelled, sheets[0]["classes"])
    for part, sheet in enumerate(sheets, 1):
        content = encoded(sheet)
        assert len(content.encode()) <= 300_000, (
            f"part {part} exceeds repository data limit"
        )
        (HERE / f"sheet-part-{part}.json").write_text(content)
    (HERE / "sheet-draw.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for reader in ("a", "b"):
        for part in range(1, 4):
            (HERE / f"reader-{reader}-part-{part}.md").write_text(brief(reader, part))
    check(args.catalog)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Draw the stratified name/definition labelling sample from the published catalog.

The sample is drawn per domain file in proportion to that file's published
name count, with a floor of ten names for any file that has ten or more, and
every name from a file that has fewer. The draw is seeded so the record can be
reproduced from the catalog revision alone.

The ten names cited in the plan's opening census are added outside the draw
and flagged, so the record can settle them individually.

Usage:
  name-definition-sample-run.py --catalog <dir> --out <sample.json> [--list <tsv>]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

SEED = 20261112
BASE_SAMPLE = 300
FLOOR = 10

CITED = [
    ("momentum_flux_due_to_e_cross_b_drift", "qualifier-missing-from-name"),
    ("argon_source_rate_due_to_gas_injection", "qualifier-missing-from-name"),
    ("deuterium_deuterium_emissivity_due_to_fusion", "qualifier-missing-from-name"),
    ("incident_neutral_particle_flux_at_wall", "qualifier-missing-from-name"),
    ("alfven_frequency", "qualifier-missing-from-name"),
    ("temperature_of_coil_conductor", "definition-narrower-than-name"),
    ("energy_of_detector_pixel", "definition-narrower-than-name"),
    ("field_line_length", "definition-narrower-than-name"),
    ("torque_density", "abstract-parent-as-quantity"),
    ("neon_density", "abstract-parent-as-quantity"),
]


def load(catalog: Path) -> dict[str, list[dict]]:
    files = {}
    for path in sorted(catalog.glob("*.yml")):
        with path.open() as fh:
            files[path.name] = yaml.safe_load(fh) or []
    return files


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Proportional share of BASE_SAMPLE, floored at FLOOR where the file allows."""
    alloc = {}
    for name, n in counts.items():
        if n <= FLOOR:
            alloc[name] = n
        else:
            alloc[name] = max(FLOOR, round(n * BASE_SAMPLE / total))
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--list", type=Path)
    ap.add_argument(
        "--labels",
        type=Path,
        help="reader's labels as [[index, name, class, pattern], ...]; enables --compose-json/--compose-html",
    )
    ap.add_argument("--compose-json", type=Path)
    ap.add_argument("--compose-html", type=Path)
    ap.add_argument("--second-reader-sheet", type=Path)
    ap.add_argument("--second-reader-labels", type=Path)
    args = ap.parse_args()

    files = load(args.catalog)
    counts = {f: len(recs) for f, recs in files.items()}
    total = sum(counts.values())
    alloc = allocate(counts, total)

    rng = random.Random(SEED)
    rows = []
    for fname in sorted(files):
        recs = sorted(files[fname], key=lambda r: r["name"])
        k = min(alloc[fname], len(recs))
        for rec in rng.sample(recs, k):
            rows.append(
                {
                    "name": rec["name"],
                    "file": fname,
                    "unit": " ".join((rec.get("unit") or "").split()),
                    "description": " ".join((rec.get("description") or "").split()),
                    "documentation": " ".join((rec.get("documentation") or "").split()),
                    "added": False,
                }
            )

    by_name = {r["name"]: r for r in rows}
    for name, class_in_plan in CITED:
        if name in by_name:
            by_name[name]["added"] = True
            by_name[name]["plan_class"] = class_in_plan
            continue
        src = next(
            (f for f, recs in files.items() if any(r["name"] == name for r in recs)),
            None,
        )
        if src is None:
            raise SystemExit(f"cited name not in catalog: {name}")
        rec = next(r for r in files[src] if r["name"] == name)
        row = {
            "name": name,
            "file": src,
            "unit": " ".join((rec.get("unit") or "").split()),
            "description": " ".join((rec.get("description") or "").split()),
            "documentation": " ".join((rec.get("documentation") or "").split()),
            "added": True,
            "plan_class": class_in_plan,
        }
        rows.append(row)
        by_name[name] = row

    rows.sort(key=lambda r: (r["file"], r["name"]))
    payload = {
        "catalog_revision": "4a48c6c57d08bc2d77e6e47e456ca10f73996eb0",
        "catalog_tag": "v0.4.0rc8",
        "seed": SEED,
        "base_sample": BASE_SAMPLE,
        "floor_per_file": FLOOR,
        "published_total": total,
        "per_file_published": counts,
        "per_file_allocated": alloc,
        "sample_size": len(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")

    if args.list:
        lines = []
        for r in rows:
            lines.append(
                "\t".join(
                    [
                        ("ADDED " if r["added"] else "") + r["name"],
                        r["file"],
                        r["unit"] or "-",
                        r["description"],
                        r["documentation"],
                    ]
                )
            )
        args.list.write_text("\n".join(lines) + "\n")

    print(f"sample {len(rows)} of {total} across {len(files)} files, seed {SEED}")
    print(
        "per file:",
        " ".join(f"{f.replace('.yml', '')}={alloc[f]}" for f in sorted(alloc)),
    )

    if args.labels:
        if not (args.compose_json and args.compose_html):
            raise SystemExit("--labels needs both --compose-json and --compose-html")
        reader = None
        if args.second_reader_sheet and args.second_reader_labels:
            reader = join_second_reader(
                args.second_reader_sheet, args.second_reader_labels
            )
        compose(
            Path(args.out),
            Path(args.labels),
            args.compose_json,
            args.compose_html,
            reader,
        )
    return 0


CLASS_ORDER = [
    "qualifier-missing-from-name",
    "definition-narrower-than-name",
    "abstract-parent-as-quantity",
    "unit-does-not-fit-name",
    "agrees",
]

CLASS_DEFINITIONS = {
    "qualifier-missing-from-name": "the definition adds an identifying attribute the name lacks "
    "(normalisation, a vector or component where the name states a scalar, an ion-only population "
    "where the name states an element, a reference-relative value, an equivalent-electron count, "
    "an averaged form, extra weighting), so the name alone denotes a broader or different quantity",
    "definition-narrower-than-name": "the definition restricts applicability to a subset of the "
    "objects the name covers (a coil family, an instrument, a plasma regime or surface, a layer)",
    "abstract-parent-as-quantity": "the definition declares a parent, umbrella or structural quantity "
    "whose population, component, averaging or specialised forms are fixed by more specific instances",
    "unit-does-not-fit-name": "the unit's physical dimension contradicts the quantity the name states",
    "agrees": "the definition states the same scope the name states",
}


RULING = {
    "date": "2026-11-12",
    "text": "a bare species name counts ions summed over all charge states, with neutral "
    "atoms excluded",
    "effect": "a name that states a species and a definition that states the ion population "
    "only name the same quantity, so the pop-ion-only wording is agreement rather than a "
    "missing qualifier",
    "relabelled": ("qualifier-missing-from-name", "pop-ion-only"),
}


def class_under_ruling(cls, pattern):
    """The class once the species-population ruling is applied to the recorded class."""
    return "agrees" if (cls, pattern) == RULING["relabelled"] else cls


def tally(rows, key, published):
    """Count one class column and project it per file onto the published set."""
    counts = dict.fromkeys(CLASS_ORDER, 0)
    for row in rows:
        counts[row[key]] += 1
    sampled = {}
    for row in rows:
        sampled[row["file"]] = sampled.get(row["file"], 0) + 1
    proj = dict.fromkeys(CLASS_ORDER, 0.0)
    for fname, n_total in published.items():
        n_drawn = sampled.get(fname, 0)
        if not n_drawn:
            continue
        for c in CLASS_ORDER:
            in_file = sum(1 for r in rows if r["file"] == fname and r[key] == c)
            proj[c] += in_file / n_drawn * n_total
    n = len(rows)
    total = sum(published.values())
    return {
        "counts": counts,
        "share": {c: round(counts[c] / n, 4) for c in CLASS_ORDER},
        "projected": {c: round(proj[c], 1) for c in CLASS_ORDER},
        "projected_rounded": {c: int(round(proj[c])) for c in CLASS_ORDER},
        "naive": {c: round(counts[c] / n * total, 1) for c in CLASS_ORDER},
    }


def kappa(pairs):
    """Cohen's kappa for two binary columns given as a list of (a, b) pairs."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected >= 1 else (observed - expected) / (1 - expected)


def join_second_reader(sheet_path, labels_path):
    """Map the second reader's item ids to names, then to the class it gave each name."""
    sheet = json.loads(sheet_path.read_text())
    labels = json.loads(labels_path.read_text())
    by_id = {item["id"]: item["name"] for item in sheet["items"]}
    return {by_id[x["id"]]: x["class"] for x in labels if x["id"] in by_id}


def agreement(rows, reader):
    """Compare the second reader's classes to this record's, joined by name."""
    mine = {row["name"]: row for row in rows}
    pairs = [
        (
            name,
            mine[name]["class"],
            mine[name]["class_under_ruling"],
            mine[name]["pattern"],
            cls,
        )
        for name, cls in reader.items()
        if name in mine
    ]
    binary = [
        (0 if p[2] == "agrees" else 1, 0 if p[4] == "agrees" else 1) for p in pairs
    ]
    return {
        "joined": len(pairs),
        "names_the_reader_labelled_not_in_sample": sorted(set(reader) - set(mine)),
        "exact_class_agreement_recorded": sum(1 for p in pairs if p[1] == p[4]),
        "exact_class_agreement_under_ruling": sum(1 for p in pairs if p[2] == p[4]),
        "exact_class_agreement_under_ruling_share": round(
            sum(1 for p in pairs if p[2] == p[4]) / len(pairs), 4
        ),
        "mismatch_versus_agrees_agreement": round(
            sum(1 for a, b in binary if a == b) / len(binary), 4
        ),
        "mismatch_versus_agrees_kappa": round(kappa(binary), 4),
        "disagreements": [
            {
                "name": p[0],
                "class_recorded": p[1],
                "class_under_ruling": p[2],
                "pattern": p[3],
                "class_second_reader": p[4],
            }
            for p in pairs
            if p[2] != p[4]
        ],
        "disagreements_unresolved": True,
    }


def compose(sample_path, labels_path, json_out, html_out, reader=None):
    """Join the drawn sample to the reader's labels and emit the record."""
    sample = json.loads(sample_path.read_text())
    rows = sample["rows"]
    labels = json.loads(labels_path.read_text())
    by_index = {int(i): (cls, pattern) for i, _name, cls, pattern in labels}

    for n, row in enumerate(rows, 1):
        row["class"], row["pattern"] = by_index[n]
        row["class_under_ruling"] = class_under_ruling(row["class"], row["pattern"])

    n_sample = len(rows)
    published = sample["per_file_published"]
    sampled_per_file = {}
    for row in rows:
        sampled_per_file[row["file"]] = sampled_per_file.get(row["file"], 0) + 1

    recorded = tally(rows, "class", published)
    ruled = tally(rows, "class_under_ruling", published)
    counts, proj, naive = recorded["counts"], recorded["projected"], recorded["naive"]

    cited = []
    for row in rows:
        if not row.get("added"):
            continue
        plan_class = row["plan_class"]
        cited.append(
            {
                "name": row["name"],
                "plan_class": plan_class,
                "recorded_class": row["class"],
                "agrees_with_plan": plan_class == row["class"],
                "pattern": row["pattern"],
            }
        )

    groups = {}
    for row in rows:
        if row["class"] == "agrees":
            continue
        key = (row["class"], row["pattern"])
        groups.setdefault(key, []).append(row["name"])
    pattern_groups = [
        {
            "class": cls,
            "pattern": pat,
            "count": len(names),
            "examples": names[:5],
        }
        for (cls, pat), names in sorted(
            groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )
    ]

    subset_file = labels_path.parent / "second-reader-subset.json"
    subset_indices = (
        json.loads(subset_file.read_text())["indices"] if subset_file.exists() else None
    )

    payload = {
        "plan_section": "sn-definitions-by-construction#s4",
        "catalog_revision": sample["catalog_revision"],
        "catalog_tag": sample["catalog_tag"],
        "catalog_path": "/home/ITER/mcintos/Code/imas-standard-names-catalog/standard_names",
        "published_total": sample["published_total"],
        "draw": {
            "seed": sample["seed"],
            "base_sample": sample["base_sample"],
            "floor_per_file": sample["floor_per_file"],
            "sample_size": n_sample,
            "files": len(published),
            "per_file_published": published,
            "per_file_allocated": sample["per_file_allocated"],
            "per_file_sampled": sampled_per_file,
            "added_outside_draw": [r["name"] for r in rows if r.get("added")],
        },
        "class_definitions": CLASS_DEFINITIONS,
        "reader": "one reader, reading name, description, documentation and unit together; "
        "no pipeline judge, review check or decide_batch produced any label",
        "ruling": RULING,
        "headline_reading": "class_under_ruling",
        "class_counts": counts,
        "class_share_of_sample": recorded["share"],
        "projection": {
            "method": "per-file stratified: each file's class share applied to that file's "
            "published count, summed over the 28 files",
            "per_class_projected": {c: round(proj[c], 1) for c in CLASS_ORDER},
            "per_class_projected_rounded": {
                c: int(round(proj[c])) for c in CLASS_ORDER
            },
            "naive_sample_share_projected": {
                c: round(naive[c], 1) for c in CLASS_ORDER
            },
        },
        "class_counts_under_ruling": ruled["counts"],
        "class_share_of_sample_under_ruling": ruled["share"],
        "projection_under_ruling": {
            "method": "the same per-file stratified projection, taken over the class the ruling "
            "gives each row",
            "per_class_projected": ruled["projected"],
            "per_class_projected_rounded": ruled["projected_rounded"],
            "naive_sample_share_projected": ruled["naive"],
        },
        "cited_names": cited,
        "pattern_groups": pattern_groups,
        "unit_check": {
            "method": "for each sampled name the unit's dimension was compared against the "
            "quantity the name states",
            "mismatches": counts["unit-does-not-fit-name"],
            "positive_control": {
                "name": "incident_neutral_particle_flux_at_wall",
                "unit": "s^-1",
                "why": "a rate (s^-1) on a name that states a flux (m^-2.s^-1); the same name is "
                "listed under 'particle_flux s^-1' by the unit census already committed in this "
                "directory, which shows the dimension check finds a conflict known to be present",
            },
        },
        "second_reader": (
            {
                "status": "met",
                "reader": "a separate reader, with no sight of this record's labels, and told "
                "the species-population ruling before it read",
                "joined_by": "name, through the sheet's item id",
                "subset_prepared": "60 names drawn with seed 7714 across all 28 files",
                "subset_indices": subset_indices,
                **agreement(rows, reader),
            }
            if reader
            else {
                "status": "unmet in this record",
                "subset_prepared": "60 names drawn with seed 7714 across all 28 files",
                "subset_indices": subset_indices,
                "agreement": None,
            }
        ),
        "rows": rows,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=1) + "\n")
    html_out.write_text(render_html(payload))
    print("wrote", json_out, "and", html_out)


def render_html(p):
    """Render the record as an evidence fragment."""
    n = p["draw"]["sample_size"]
    ru = p["class_counts_under_ruling"]
    su = p["class_share_of_sample_under_ruling"]
    pu = p["projection_under_ruling"]["per_class_projected"]
    moved = sum(1 for row in p["rows"] if row["class_under_ruling"] != row["class"])
    rows = ""
    for r in p["rows"]:
        rows += (
            "<tr>"
            f"<td><code>{r['name']}</code></td>"
            f"<td><code>{r['unit'] or '—'}</code></td>"
            f"<td>{r['file'].replace('.yml', '')}</td>"
            f"<td>{r['class']}</td>"
            f"<td>{r['class_under_ruling']}</td>"
            f"<td>{'' if r['pattern'] == '-' else r['pattern']}</td>"
            "</tr>\n"
        )
    class_rows = ""
    for c in CLASS_ORDER:
        class_rows += (
            "<tr>"
            f"<td>{c}</td>"
            f"<td>{p['class_counts'][c]}</td>"
            f"<td>{p['class_share_of_sample'][c] * 100:.1f}&#8201;%</td>"
            f"<td>{p['projection']['per_class_projected'][c]}</td>"
            f"<td>{ru[c]}</td>"
            f"<td>{su[c] * 100:.1f}&#8201;%</td>"
            f"<td>{pu[c]}</td>"
            "</tr>\n"
        )
    cited_rows = "".join(
        "<tr>"
        f"<td><code>{c['name']}</code></td>"
        f"<td>{c['plan_class']}</td>"
        f"<td>{c['recorded_class']}</td>"
        f"<td>{c['pattern']}</td>"
        "</tr>\n"
        for c in p["cited_names"]
    )
    group_rows = "".join(
        "<tr>"
        f"<td>{g['class']}</td>"
        f"<td>{g['pattern']}</td>"
        f"<td>{g['count']}</td>"
        f"<td>{', '.join('<code>' + x + '</code>' for x in g['examples'])}</td>"
        "</tr>\n"
        for g in p["pattern_groups"]
    )
    sr = p["second_reader"]
    if sr.get("status") == "met":
        dis_rows = "".join(
            "<tr>"
            f"<td><code>{d['name']}</code></td>"
            f"<td>{d['class_recorded']}</td>"
            f"<td>{d['class_under_ruling']}</td>"
            f"<td>{d['class_second_reader']}</td>"
            "</tr>\n"
            for d in sr["disagreements"]
        )
        sr_html = f"""<p>A separate reader, with no sight of this record's labels and told the
species-population ruling before it read, labelled the same 60 names. Joining the two sets by name,
<strong>{sr["exact_class_agreement_under_ruling"]} of {sr["joined"]} carry the same class</strong>
({sr["exact_class_agreement_under_ruling_share"] * 100:.1f}&#8201;%) under the ruling, and
{sr["exact_class_agreement_recorded"]} of {sr["joined"]} under the labels as first recorded. Read as
the cruder question the labels answer first — mismatch or agreement — the two agree on
{sr["mismatch_versus_agrees_agreement"] * 100:.1f}&#8201;% of the subset, Cohen's
&kappa;&#8201;=&#8201;{sr["mismatch_versus_agrees_kappa"]:.3f}.</p>
<p id="sdbc-name-definition-count-disagreement-note">The {len(sr["disagreements"])} disagreements are
listed as they stand. Neither reader's class is preferred and none is resolved here.</p>
<table id="sdbc-name-definition-count-disagree-table">
<thead><tr><th>Name</th><th>Class recorded</th><th>Under ruling</th><th>Second reader</th></tr></thead>
<tbody>
{dis_rows}</tbody>
</table>"""
    else:
        sr_html = (
            "<p>That reading is not in this record, and a second reading taken by this node "
            "would not be independent.</p>"
        )
    per_file = "".join(
        "<tr>"
        f"<td>{f.replace('.yml', '')}</td>"
        f"<td>{c}</td>"
        f"<td>{p['draw']['per_file_allocated'][f]}</td>"
        f"<td>{p['draw']['per_file_sampled'][f]}</td>"
        "</tr>\n"
        for f, c in sorted(p["draw"]["per_file_published"].items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence-fragment">
<meta name="plan-slug" content="sn-definitions-by-construction">
<meta name="node-id" content="sdbc-name-definition-count">
<title>Every published definition counted against its name</title>
</head>
<body>
<main>
<h1 id="sdbc-name-definition-count-title">A labelled sample of {n} published names, each read against its definition</h1>

<p id="sdbc-name-definition-count-summary">
{n} of the {p["published_total"]} names published in <code>{p["catalog_tag"]}</code>
(<code>{p["catalog_revision"][:8]}</code>) were drawn across all {p["draw"]["files"]} domain files,
in proportion to each file's published count and with a floor of {p["draw"]["floor_per_file"]} per
file that has that many. Each was labelled by reading its name, description, documentation and
unit together. <strong>Under the species-population ruling {ru["agrees"]} agree with their name and
{n - ru["agrees"]} do not</strong>: {ru["qualifier-missing-from-name"]}
carry a qualifier the name lacks, {ru["definition-narrower-than-name"]} are narrower
than the name, {ru["abstract-parent-as-quantity"]} publish an abstract parent as a
quantity, and {ru["unit-does-not-fit-name"]} has a unit whose dimension does not fit
its name. Projected per file onto the published set, the qualifier class accounts for about
{pu["qualifier-missing-from-name"]} names.
</p>

<h2 id="sdbc-name-definition-count-verdict">What the sample measures, both ways</h2>
<p id="sdbc-name-definition-count-ruling-text">The ruling of {p["ruling"]["date"]} changed one
wording: {p["ruling"]["text"]}. The right-hand columns are the headline reading, with the {moved}
names whose only defect was that wording moved to <em>agrees</em>; the left-hand columns keep the
labels as first recorded, so both readings stay in the record.</p>
<table id="sdbc-name-definition-count-classes">
<thead><tr><th>Class</th><th>Sample (recorded)</th><th>Share</th><th>Projected (recorded)</th><th>Sample (ruling)</th><th>Share (ruling)</th><th>Projected (ruling)</th></tr></thead>
<tbody>
{class_rows}</tbody>
</table>
<p id="sdbc-name-definition-count-projection">The stratified projection applies each file's
class share to that file's own published count and sums over the files, which is what makes the
per-file floor harmless: a file drawn at 10 of 34 contributes its own rate, not a
sample-wide rate. The same projection is applied to both readings, so the two columns differ only
by the {moved} names the ruling moved, never by method.</p>

<h2 id="sdbc-name-definition-count-cited">The ten names the opening census cited</h2>
<p>Each was added outside the random draw. {sum(1 for c in p["cited_names"] if c["agrees_with_plan"])} of {len(p["cited_names"])} carry the class the census gave them.</p>
<table id="sdbc-name-definition-count-cited-table">
<thead><tr><th>Name</th><th>Class in §1</th><th>Class recorded here</th><th>Wording</th></tr></thead>
<tbody>
{cited_rows}</tbody>
</table>
<p id="sdbc-name-definition-count-cited-note">
<code>incident_neutral_particle_flux_at_wall</code> is recorded here as
<em>unit-does-not-fit-name</em> rather than under the census's qualifier row. The census's reading
is not wrong — the definition is a wall-integrated rate — but the discriminating evidence is the
definition's unit, <code>s^-1</code>: a rate on a name that states a flux. Recording it by the unit
keeps the class readable as one instrument's verdict, and it is the positive control for this
record's unit check, because a census already committed beside this file lists the same name in its
<code>particle_flux s^-1</code> group.
</p>

<h2 id="sdbc-name-definition-count-patterns">The wordings that produced the mismatches</h2>
<p>Each pattern is a sentence or convention a writer re-uses, so a class read down this table is
read as a family of wordings rather than as a count of unrelated defects. The classes are the ones
recorded; under the ruling, the pop-ion-only row is agreement.</p>
<table id="sdbc-name-definition-count-pattern-table">
<thead><tr><th>Class</th><th>Wording</th><th>Count</th><th>Examples</th></tr></thead>
<tbody>
{group_rows}</tbody>
</table>

<h2 id="sdbc-name-definition-count-sample">The draw</h2>
<p>Seed <strong>{p["draw"]["seed"]}</strong>, drawn by
<code>name-definition-sample-run.py</code> beside this file, which re-draws the sample from the
catalog revision alone.</p>
<table id="sdbc-name-definition-count-alloc">
<thead><tr><th>Domain file</th><th>Published</th><th>Allocated</th><th>Sampled</th></tr></thead>
<tbody>
{per_file}</tbody>
</table>

<h2 id="sdbc-name-definition-count-second-reader">The second reading</h2>
{sr_html}

<h2 id="sdbc-name-definition-count-rows">The labelled rows</h2>
<details>
<summary>All {n} rows</summary>
<table id="sdbc-name-definition-count-row-table">
<thead><tr><th>Name</th><th>Unit</th><th>File</th><th>Class recorded</th><th>Under ruling</th><th>Wording</th></tr></thead>
<tbody>
{rows}</tbody>
</table>
</details>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

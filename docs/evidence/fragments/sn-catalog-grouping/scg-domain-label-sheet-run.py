#!/usr/bin/env python
"""Build the blind physics-domain label sheet for standard-name grouping.

The sheet holds twenty reference names and 150 sampled live names. Each row
carries only what a labeller may see: the name, description, unit, the parsed
grammar segments, and up to two DD source bindings. It carries an empty
``label`` and empty ``note`` for a hand label, and never the stored
``physics_domain`` or the source domains. Those two go to the incumbent file.

The sampling and the fixed seed come from
:mod:`imas_codex.standard_names.judgment_goldens`; the pattern family comes
from :mod:`imas_codex.standard_names.families`, the single definition of a
family. The physics-domain option list is read from the installed ISN
package's ``physics_domains.yml`` rather than copied.
"""

from __future__ import annotations

import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.families import pattern_signature
from imas_codex.standard_names.judgment_goldens import _sample, assign_group_splits

FRAGMENTS = Path(__file__).resolve().parent
ROOT = FRAGMENTS.parents[3]

SHEET_PATH = FRAGMENTS / "scg-domain-label-sheet.json"
INCUMBENT_PATH = FRAGMENTS / "scg-domain-incumbent.json"
FRAGMENT_PATH = FRAGMENTS / "scg-domain-label-sheet.html"
SCHEMA_PATH = ROOT / "imas_codex" / "schemas" / "standard_name.yaml"

REFERENCE_FIXED = (
    "electron_temperature",
    "electron_density",
    "safety_factor",
    "plasma_current",
    "poloidal_magnetic_flux",
)
REFERENCE_SIZE = 20
SAMPLE_SIZE = 150

#: A live name is one whose name pipeline reached a terminal good stage.
LIVE_STAGES = ("accepted", "approved")

#: The stored grammar columns that key a pattern family.
_INVARIANT_FIELDS = (
    "physical_base",
    "geometric_base",
    "transformation",
    "process",
    "object",
)

_LIVE_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN $stages
RETURN sn.id AS name,
       sn.description AS description,
       sn.unit AS unit,
       sn.physics_domain AS physics_domain,
       coalesce(sn.source_domains, []) AS source_domains,
       coalesce(sn.source_paths, []) AS source_paths,
       sn.physical_base AS physical_base,
       sn.geometric_base AS geometric_base,
       sn.subject AS subject,
       sn.component AS component,
       sn.coordinate AS coordinate,
       sn.transformation AS transformation,
       sn.process AS process,
       sn.position AS position,
       sn.region AS region,
       sn.object AS object,
       sn.geometry AS geometry,
       sn.device AS device
"""


def read_live(graph: GraphClient) -> list[dict[str, Any]]:
    """Return every live name with the fields the sheet and incumbent need."""
    return graph.query(_LIVE_QUERY, stages=list(LIVE_STAGES)) or []


def segments_of(row: dict[str, Any]) -> dict[str, str]:
    """Project a live row's stored grammar columns onto the row's axes."""
    locus = (
        row.get("position"),
        row.get("region"),
        row.get("object"),
        row.get("geometry"),
    )
    return {
        "physical_base": row.get("physical_base") or "",
        "geometric_base": row.get("geometric_base") or "",
        "subject": row.get("subject") or "",
        # ``component`` and ``coordinate`` are parser aliases for one ISN axis.
        "component": row.get("component") or row.get("coordinate") or "",
        "operator": row.get("transformation") or "",
        "locus": "|".join(value for value in locus if value),
        "process": row.get("process") or "",
    }


def family_of(row: dict[str, Any]) -> str:
    """The family signature of a live row, from the grammar definition."""
    return "|".join(pattern_signature({f: row.get(f) for f in _INVARIANT_FIELDS}))


def dd_bindings(row: dict[str, Any]) -> list[str]:
    """The DD source paths of a live row, in stored order."""
    return [p for p in (row.get("source_paths") or []) if p.startswith("dd:")]


def sheet_row(row: dict[str, Any]) -> dict[str, Any]:
    """One blind row: no stored domain and no source domains appear here."""
    bindings = dd_bindings(row)
    return {
        "name": row["name"],
        "description": row.get("description") or "",
        "unit": row.get("unit") or "",
        "segments": segments_of(row),
        "sources": [
            {"path": p, "ids": p[len("dd:") :].split("/", 1)[0]} for p in bindings[:2]
        ],
        "label": "",
        "note": "",
    }


def select_reference(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The fixed five plus the fifteen most DD-bound names, one per base.

    Returns the chosen rows in order and the fixed names absent from the live
    population; an absent name's slot stays empty and is never substituted.
    """
    by_name = {row["name"]: row for row in rows}
    present = [by_name[name] for name in REFERENCE_FIXED if name in by_name]
    absent = [name for name in REFERENCE_FIXED if name not in by_name]

    excluded = set(REFERENCE_FIXED)
    candidates = [
        row for row in rows if row["name"] not in excluded and dd_bindings(row)
    ]
    candidates.sort(key=lambda row: (-len(dd_bindings(row)), row["name"]))
    chosen: list[dict[str, Any]] = []
    used_bases: set[str] = set()
    for row in candidates:
        if len(chosen) >= REFERENCE_SIZE - len(present):
            break
        base = row.get("physical_base") or ""
        if base in used_bases:
            continue
        used_bases.add(base)
        chosen.append(row)
    return present + chosen, absent


def sample_units(rows: list[dict[str, Any]], exclude: set[str]) -> list[dict[str, Any]]:
    """150 names spread round-robin over family and incumbent domain.

    Reuses ``_sample`` from the judgment fixtures so the draw and its fixed
    seed are the ones the evaluation already trusts.
    """
    candidates = [
        {
            "unit": row["name"],
            "physics_domain": row.get("physics_domain") or "",
            "pattern_family": family_of(row),
        }
        for row in rows
        if row["name"] not in exclude
    ]
    return _sample(candidates, SAMPLE_SIZE)


def load_options() -> list[dict[str, str]]:
    """The ISN physics-domain values with descriptions, read from ISN."""
    import imas_standard_names

    path = (
        Path(imas_standard_names.__file__).parent
        / "grammar"
        / "vocabularies"
        / "physics_domains.yml"
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        {
            "value": value,
            "description": spec.get("description", ""),
            "category": spec.get("category", ""),
        }
        for value, spec in document["physics_domains"].items()
    ]


def build(
    graph: GraphClient,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """The sheet, incumbent domains and sampled families as plain data."""
    live = read_live(graph)
    by_name = {row["name"]: row for row in live}

    reference, absent = select_reference(live)
    sampled = sample_units(live, {row["name"] for row in reference})
    splits = assign_group_splits(sampled, "pattern_family")

    reference_rows = []
    for row in reference:
        entry = sheet_row(row)
        entry["reference"] = True
        reference_rows.append(entry)

    sample_out = []
    for choice in sampled:
        entry = sheet_row(by_name[choice["unit"]])
        entry["split"] = splits[choice["pattern_family"]]
        sample_out.append(entry)

    sheet = {
        "reference_rule": (
            "the live accepted or approved names with the most DD source "
            "bindings, at most one per physical base, excluding the fixed five"
        ),
        "reference_absent": absent,
        "options": load_options(),
        "reference": reference_rows,
        "sample": sample_out,
    }
    incumbent = {
        row["name"]: {
            "physics_domain": row.get("physics_domain"),
            "source_domains": list(row.get("source_domains") or []),
        }
        for row in reference + [by_name[choice["unit"]] for choice in sampled]
    }
    return sheet, incumbent, sampled


def check_blindness(sheet: dict[str, Any]) -> int:
    """Fail if any sheet row carries a domain field; return the row count."""
    offenders = []
    for block in ("reference", "sample"):
        for row in sheet[block]:
            for key in row:
                if "domain" in key.lower():
                    offenders.append((block, row["name"], key))
    if offenders:
        raise AssertionError(f"sheet rows carry a domain field: {offenders}")
    return len(sheet["reference"]) + len(sheet["sample"])


def check_query_properties() -> list[str]:
    """Audit this driver's Cypher literals against the LinkML schema."""
    from imas_codex.graph.cypher_property_check import audit_cypher_properties
    from imas_codex.graph.schema import GraphSchema

    report = audit_cypher_properties(
        Path(__file__).resolve(), schemas=(GraphSchema(SCHEMA_PATH),)
    )
    return [str(violation) for violation in report.violations]


def _counts(
    sheet: dict[str, Any],
    incumbent: dict[str, Any],
    sampled: list[dict[str, Any]],
) -> dict[str, Any]:
    """The sample's counts by family, incumbent domain and split."""
    sample = sheet["sample"]
    return {
        "families": Counter(row["pattern_family"] for row in sampled),
        "domains": Counter(
            (incumbent[row["name"]]["physics_domain"] or "(null)") for row in sample
        ),
        "splits": Counter(row["split"] for row in sample),
        "null_domain": sum(
            1 for row in sample if not incumbent[row["name"]]["physics_domain"]
        ),
    }


def _table(headers: tuple[str, ...], items: list[tuple[str, int]]) -> str:
    """A two-column table of label and count."""
    head = "".join(f"<th>{html.escape(name)}</th>" for name in headers)
    body = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{value}</td></tr>"
        for key, value in items
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_fragment(
    sheet: dict[str, Any],
    incumbent: dict[str, Any],
    sampled: list[dict[str, Any]],
) -> str:
    """The evidence fragment: rule, counts, blindness check and options."""
    counts = _counts(sheet, incumbent, sampled)
    picks = "".join(
        f"<li><code>{html.escape(row['name'])}</code> — "
        f"{html.escape(row['description'][:90])}</li>"
        for row in sheet["reference"]
    )
    options = "".join(
        f"<li><code>{html.escape(item['value'])}</code> — "
        f"{html.escape(item['description'])}</li>"
        for item in sheet["options"]
    )
    rows = len(sheet["reference"]) + len(sheet["sample"])
    body = [
        '<section id="scg-domain-label-sheet">',
        "  <h2>Blind physics-domain label sheet</h2>",
        "  <p>Twenty reference names and 150 sampled live names, each with the "
        "name, description, unit, parsed grammar segments and up to two DD source "
        "bindings, an empty <code>label</code> and <code>note</code>. No row "
        "carries the stored <code>physics_domain</code> or the source domains; "
        "those live in <code>scg-domain-incumbent.json</code>, keyed by name.</p>",
        '  <h3 id="scg-domain-label-sheet-reference">Reference rule and picks</h3>',
        f"  <p>{html.escape(sheet['reference_rule'])}. Fixed five absent from the "
        f"live population: "
        f"{html.escape(', '.join(sheet['reference_absent']) or 'none')}.</p>",
        f"  <ol>{picks}</ol>",
        '  <h3 id="scg-domain-label-sheet-counts">Sample counts</h3>',
        f"  <p>Rows on the sheet: {rows} (20 reference, "
        f"{len(sheet['sample'])} sample).</p>",
        _table(("split", "rows"), sorted(counts["splits"].items())),
        _table(("incumbent domain", "rows"), sorted(counts["domains"].items())),
        f"  <p>Sample rows with a null stored domain: {counts['null_domain']}.</p>",
        "  <p>Sample rows spread over "
        f"{len(counts['families'])} pattern families; family size (rows per "
        "family) below.</p>",
        _table(
            ("rows per family", "families"),
            sorted(Counter(counts["families"].values()).items()),
        ),
        '  <h3 id="scg-domain-label-sheet-blindness">Blindness check</h3>',
        "  <p>No sheet row carries a field whose name contains "
        f"<code>domain</code>; checked over all {rows} rows.</p>",
        '  <h3 id="scg-domain-label-sheet-options">Physics-domain options '
        "(from ISN)</h3>",
        f"  <ol>{options}</ol>",
        "</section>",
    ]
    return "\n".join(body)


def write_json(path: Path, data: Any) -> None:
    """Write *data* as canonical JSON so two runs compare byte for byte."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def preserve_annotations(sheet: dict[str, Any], path: Path) -> None:
    """Keep hand labels and approval when regenerating the same name roster."""
    if not path.exists():
        return
    prior = json.loads(path.read_text(encoding="utf-8"))
    for block in ("reference", "sample"):
        old_rows = prior[block]
        new_rows = sheet[block]
        if [row["name"] for row in old_rows] != [row["name"] for row in new_rows]:
            raise ValueError(
                f"{block} roster changed; inspect labels before rebuilding"
            )
        for old, new in zip(old_rows, new_rows, strict=True):
            for field in ("label", "note", "label_source"):
                if field in old:
                    new[field] = old[field]
    if "reference_approved" in prior:
        sheet["reference_approved"] = prior["reference_approved"]


def main() -> int:
    with GraphClient() as graph:
        sheet, incumbent, sampled = build(graph)
    rows = check_blindness(sheet)
    violations = check_query_properties()
    if violations:
        print("cypher property violations:", *violations, sep="\n")
        return 2
    preserve_annotations(sheet, SHEET_PATH)
    write_json(SHEET_PATH, sheet)
    write_json(INCUMBENT_PATH, incumbent)
    FRAGMENT_PATH.write_text(
        render_fragment(sheet, incumbent, sampled), encoding="utf-8"
    )
    print(
        f"reference={len(sheet['reference'])} sample={len(sheet['sample'])} "
        f"options={len(sheet['options'])} rows={rows}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

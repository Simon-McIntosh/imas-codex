"""Plan, run and report the family and operator wording pass over the live catalog.

The pass brings every member of a divergent species family to the family's one
wording, every departing bare-species density to the settled population
sentence, and every operator name that paraphrases its operator to the stored
operator sentence. Stored text changes only through ``sn edit``; this script
never writes the graph itself.

Phases:

``plan``    read the closing census and the live rows, classify every name, and
            write the work list with one batch per family.
``status``  read the progress file and say what has settled.
``report``  assemble the delivered JSON: before/after per name, per family and
            per operator forms, the names left with a reason, and the spend
            read from each edit's own run id.
``census``  re-read the census counts from the live graph by running the
            committed closing census live phase.
"""

from __future__ import annotations

import argparse
import collections
import html as html_lib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CENSUS = HERE / "rewrite-closing-census.json"
CACHE = Path.home() / ".cache" / "imas-codex-live-rewrite"
WORKLIST = CACHE / "family-worklist.json"
PROGRESS = CACHE / "family-pass-progress.jsonl"
OUT = HERE / "family-wording-pass.json"
HTML_OUT = HERE / "family-wording-pass.html"
FIG_DIR = REPO / "docs" / "figures" / "sn-family-wording-pass"
CENSUS_BEFORE = CACHE / "census-before-pass.json"
CEILING_USD = 30.0
PROBE_FAMILY = "ratio_of_<X>_density_to_electron_density"
PROBE_LINE_USD = 0.10

#: One stored wording per family; a family is a species token substituted into
#: this template, the same measure the closing census takes.
SPECIES = [
    "helium_3", "helium_4", "deuterium_tritium",
    "deuterium", "tritium",
    "helium", "hydrogen", "lithium",
    "beryllium", "boron", "carbon",
    "nitrogen", "oxygen", "neon",
    "argon", "krypton", "xenon",
    "iron", "tungsten", "total_hydrogenic",
]

#: Population qualifiers that take the name off the bare-species rule: the
#: quantity is one charge state, or a neutral population, and the settled
#: sentence would misstate it.
OFF_POPULATION_TOKENS = ("neutral", "charge_state", "internal_state")

#: The settled sentence is a statement about a particle number density, so it
#: applies only where the unit is one.
PARTICLE_DENSITY_UNIT = "m^-3"

POPULATION_HINT = (
    "State the species population with the stored sentence verbatim: "
    "\"The density counts ions summed over all charge states; neutral atoms are "
    "excluded.\" Do not restate the sum in other words and do not describe the "
    "quantity as elemental, neutral-inclusive, or as counting all particles."
)
POPULATION_REASON = (
    "The settled population for a bare species density counts ions summed over "
    "all charge states, with neutral atoms excluded (lead ruling 2026-11-12). "
    "This definition states another population or omits the settled sentence."
)
FAMILY_HINT = (
    "Write the definition in its family's one form: name the quantity this "
    "member reports and carry the settled shared sentences unchanged, so the "
    "text matches the family's other members apart from the species."
)
OPERATOR_REASON = (
    "The catalog carries one stored wording per operator and the stored "
    "sentence is rendered into every name that applies it. This definition "
    "paraphrases the operator instead of carrying the stored sentence."
)

#: A family whose members state a species-specific equivalent-electron factor
#: cannot share one template: the number is species data, not wording. The name
#: also needs the equivalent_electron qualifier, which is a rename.
COUNTING_FACTOR_REASON = (
    "the members state a species-specific equivalent-electron factor, so no one "
    "template carries the family; the name also needs the equivalent_electron "
    "qualifier, a rename this pass does not make"
)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_census():
    return json.loads(CENSUS.read_text())


def live_rows(names):
    from imas_codex.graph.client import GraphClient

    query = """
    MATCH (sn:StandardName) WHERE sn.id IN $names
    OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
    WITH sn, collect(DISTINCT u.id)[0] AS u
    OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
    RETURN sn.id AS id, coalesce(sn.description,'') AS description,
           coalesce(sn.documentation,'') AS documentation,
           coalesce(u, sn.unit, '') AS unit,
           coalesce(sn.docs_stage,'') AS docs_stage,
           coalesce(sn.name_stage,'') AS name_stage,
           sn.reviewer_score_docs AS score,
           coalesce(sn.status,'') AS status,
           collect(DISTINCT src.path) AS sources,
           toString(sn.updated_at) AS updated_at
    ORDER BY sn.id
    """
    with GraphClient() as client:
        return [dict(record) for record in client.query(query, names=sorted(names))]


def family_template(name):
    pattern = re.compile(r"(?<![a-z])(" + "|".join(SPECIES) + r")(?![a-z])")
    return pattern.sub("<X>", name)


def operator_of(name):
    return [
        operator
        for operator in (
            "flux_surface_averaged",
            "volume_averaged",
            "line_averaged",
            "line_integrated",
            "time_derivative",
        )
        if operator in name
    ]


def off_population(name, unit):
    """Why the settled population sentence does not apply to *name*."""

    if unit != PARTICLE_DENSITY_UNIT:
        return (
            "not a particle number density",
            "the unit is %s, so the sentence about a counted ion density does "
            "not describe this quantity" % (unit or "unrecorded"),
        )
    for token, description in (
        ("neutral", "neutral"),
        ("charge_state", "single-charge-state"),
        ("internal_state", "internal-state"),
    ):
        if token in name:
            return (
                "source includes neutrals or a single charge state",
                "the name selects a %s population, which the bare-species "
                "sentence would misstate; per the lead ruling the name needs a "
                "rename with its source evidence and its text is left alone here"
                % description,
            )
    return None


def plan():
    census = load_census()["phases"]["live"]
    rewrite = json.loads((HERE / "live-rewrite-bulk.json").read_text())
    settled_by_first_pass = {
        entry["name"]
        for entry in rewrite["before_after"]
        if entry.get("docs_stage") in ("accepted", "reviewed")
    }
    counting_class = {entry["name"] for entry in rewrite["waiting_equivalent_electron"]}

    families = census["families"]["divergent_families"]
    departing = [entry["name"] for entry in census["population"]["departing"]]
    operators = census["operators"]
    omitted = {}
    stored = {}
    for operator, entry in operators.items():
        omitted[operator] = set(entry["forms"].get("<omitted>", []))
        stored[operator] = entry["stored_sentence"]

    family_of = {}
    for family, entry in families.items():
        for name in entry["names"]:
            family_of.setdefault(name, []).append(family)

    union = set(family_of) | set(departing) | set().union(*omitted.values())
    rows = {row["id"]: row for row in live_rows(union)}

    countable_families = {
        family
        for family in families
        if any(name in counting_class for name in families[family]["names"])
    }

    left = []
    work = {}
    for name in sorted(union):
        row = rows[name]
        if name in settled_by_first_pass and name not in family_of:
            left.append(
                {"name": name, "reason": "settled by the first pass",
                 "detail": "the first rewrite already landed this name's wording"}
            )
            continue
        reason_off = off_population(name, row["unit"])
        if reason_off and name in departing:
            left.append(
                {"name": name, "reason": reason_off[0], "detail": reason_off[1]}
            )
            continue
        if any(family in countable_families for family in family_of.get(name, [])):
            left.append(
                {
                    "name": name,
                    "reason": "family cannot reach one wording",
                    "detail": COUNTING_FACTOR_REASON,
                }
            )
            continue
        operators_here = [op for op in operator_of(name) if name in omitted[op]]
        is_species = name in departing
        fams = family_of.get(name, [])
        if not (operators_here or is_species):
            left.append(
                {
                    "name": name,
                    "reason": "no stored wording to carry",
                    "detail": (
                        "the name is divergent on the family measure, but the "
                        "catalog records no wording for the family beyond the "
                        "population clause; authoring a template here would be "
                        "freehand text, so the family is left with its "
                        "divergence recorded rather than converted"
                    ),
                    "families": fams,
                }
            )
            continue
        hint_parts = []
        if is_species:
            hint_parts.append(POPULATION_HINT)
        for operator in operators_here:
            hint_parts.append(
                'Carry the stored operator sentence verbatim: "%s"'
                % stored[operator]
            )
        reason_parts = []
        if is_species:
            reason_parts.append(POPULATION_REASON)
        if operators_here:
            reason_parts.append(OPERATOR_REASON)
        work[name] = {
            "name": name,
            "families": fams,
            "operators": operators_here,
            "species": is_species,
            "unit": row["unit"],
            "hint": " ".join(hint_parts),
            "reason": " ".join(reason_parts),
            "docs_stage": row["docs_stage"],
            "score": row["score"],
            "description": row["description"],
            "documentation": row["documentation"],
        }

    # the species sentence first, then the operator sentences, in batches of
    # about forty names so each job is one submit and one settle per name
    batches = []
    species_names = sorted(n for n in work if work[n]["species"])
    operator_names = sorted(n for n in work if work[n]["operators"] and not work[n]["species"])
    for start in range(0, len(species_names), 40):
        chunk = species_names[start : start + 40]
        batches.append({
            "id": "species:%d" % (start // 40 + 1),
            "kind": "species",
            "names": chunk,
            "hint": work[chunk[0]]["hint"],
            "reason": work[chunk[0]]["reason"],
        })
    for start in range(0, len(operator_names), 40):
        chunk = operator_names[start : start + 40]
        batches.append({
            "id": "operator:%d" % (start // 40 + 1),
            "kind": "operator",
            "names": chunk,
            "hint": work[chunk[0]]["hint"],
            "reason": work[chunk[0]]["reason"],
        })

    payload = {
        "schema": "family-wording-worklist/1",
        "generated_at": now(),
        "population_source": str(CENSUS),
        "counts": {
            "union": len(union),
            "worked": len(work),
            "left": len(left),
            "families": len(families),
            "family_members": sum(len(e["names"]) for e in families.values()),
            "departing": len(departing),
            "omitted_operator_names": len(set().union(*omitted.values())),
        },
        "left": left,
        "work": work,
        "batches": batches,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    WORKLIST.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", WORKLIST)
    print(json.dumps(payload["counts"], indent=1))
    for batch in batches:
        print("  %-46s %3d names" % (batch["id"], len(batch["names"])))


def entry_key(kv):
    return (kv[1]["member_count"], kv[0])


def status():
    settled = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("name") and record.get("outcome"):
                settled[record["name"]] = record["outcome"]
    worklist = json.loads(WORKLIST.read_text())
    work = worklist["work"]
    print("worked %d, settled %d" % (len(work), len(set(work) & set(settled))))
    for batch in worklist["batches"]:
        done = sum(1 for n in batch["names"] if n in settled)
        print("  %-46s %3d/%3d settled" % (batch["id"], done, len(batch["names"])))


def census_module():
    """Load the committed closing census so its measures are reused, not copied."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "closing_census", HERE / "rewrite-closing-census.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settled_records():
    """The latest settled progress record per name."""

    records = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("name") and record.get("outcome") not in (None, "pending"):
                records[record["name"]] = record
    return records


def live_rows_all():
    from imas_codex.graph.client import GraphClient

    module = census_module()
    with GraphClient() as client:
        return module, [dict(row) for row in client.query(module.LIVE_QUERY)]


def templates_of(module, row_by_name, names):
    """Description templates a family's members carry, by the census's own measure."""

    counter = collections.Counter()
    for name in names:
        row = row_by_name.get(name)
        if row is None:
            continue
        text = re.sub(r"\s+", " ", module.SPECIES_WORD_RE.sub("<X>", row.get("description") or ""))
        counter[text.strip()] += 1
    return counter


def report():
    before = json.loads(CENSUS_BEFORE.read_text())["phases"]["live"]
    worklist = json.loads(WORKLIST.read_text())
    records = settled_records()
    module, rows = live_rows_all()
    row_by_name = {row["id"]: row for row in rows}
    after_families = module.family_census(rows)
    before_operator_forms = {
        operator: before["operators"][operator]["form_count"] for operator in module.OPERATORS
    }
    after_operators = module.operator_census(rows, before_operator_forms)

    worked = list(records.values())
    for record in worked:
        record.pop("log", None)

    families = []
    for family, entry in before["families"]["divergent_families"].items():
        members = entry["names"]
        member_set = set(members)
        after_templates = templates_of(module, row_by_name, members)
        settled = [records[name] for name in members if name in records]
        families.append(
            {
                "family": family,
                "member_count": len(members),
                "template_count_before": entry["template_count"],
                "template_count_after": len(after_templates),
                "templates_before": entry["templates"],
                "templates_after": [
                    {"count": count, "text": text[:240]}
                    for text, count in after_templates.most_common()
                ],
                "unit_count_before": entry["unit_count"],
                "unit_count_after": len({row_by_name[n]["unit"] for n in members}),
                "worked_count": len(settled),
                "outcomes": dict(collections.Counter(r["outcome"] for r in settled)),
                "names": sorted(members),
                "left": [item for item in worklist["left"] if item["name"] in member_set],
                "spend_usd": round(sum(r["spend_usd"] for r in settled), 6),
                "records": [r["name"] for r in settled],
            }
        )

    operators = []
    for operator in module.OPERATORS:
        front = before["operators"][operator]
        back = after_operators[operator]
        settled = [r for r in worked if operator in r.get("operators")]
        operators.append(
            {
                "operator": operator,
                "stored_sentence": front["stored_sentence"],
                "name_count_before": front["name_count"],
                "name_count_after": back["name_count"],
                "form_count_before": front["form_count"],
                "form_count_after": back["form_count"],
                "omitted_before": front["omitted_count"],
                "omitted_after": back["omitted_count"],
                "forms_before": front["forms"],
                "forms_after": back["forms"],
                "worked_count": len(settled),
                "outcomes": dict(collections.Counter(r["outcome"] for r in settled)),
                "spend_usd": round(sum(r["spend_usd"] for r in settled), 6),
            }
        )

    total = round(sum(r["spend_usd"] for r in worked), 6)
    rated = [r for r in worked if r["spend_usd"] > 0]
    score_drops = [
        {
            "name": r["name"],
            "score_before": r.get("score_before"),
            "score_after": r.get("score_after"),
        }
        for r in worked
        if r.get("score_after") is not None
        if (r.get("score_after") or 0) < (r.get("score_before") or 0)
    ]

    live = {
        "families": after_families,
        "operators": after_operators,
        "population": module.population_census(rows),
    }
    census_after = {
        "read_from": "live graph rows through the committed census measures",
        "families": {
            key: live["families"][key]
            for key in ("family_count", "member_count", "one_wording_count", "divergent_count")
        },
        "operators": {
            operator: {
                key: live["operators"][operator][key]
                for key in ("name_count", "form_count", "omitted_count")
            }
            for operator in module.OPERATORS
        },
        "population": {
            "species_names": live["population"]["species_names"],
            "departing_count": len(live["population"]["departing"]),
            "mentioning_neutral": live["population"]["mentioning_neutral"],
            "silent_on_the_sum": live["population"]["silent_on_the_sum"],
        },
    }

    probe_records = [r for r in worked if PROBE_FAMILY in (r.get("families") or [])]
    probe = {
        "family": PROBE_FAMILY,
        "names": len(probe_records),
        "spend_usd": round(sum(r["spend_usd"] for r in probe_records), 6),
        "rate_per_name_usd": round(
            sum(r["spend_usd"] for r in probe_records) / len(probe_records), 6
        )
        if probe_records
        else 0.0,
        "line_usd": PROBE_LINE_USD,
        "outcomes": dict(collections.Counter(r["outcome"] for r in probe_records)),
        "note": (
            "the probe is one whole small family run before the rest, so its "
            "rate is read against the line before any further spend"
        ),
    }
    payload = {
        "schema": "family-wording-pass/1",
        "generated_at": now(),
        "plan_section": "sn-definitions-by-construction#s6",
        "population_source": str(CENSUS_BEFORE),
        "probe": probe,
        "before_counts": before["families"],
        "worked_count": len(worked),
        "left_count": len(worklist["left"]),
        "families": families,
        "operators": operators,
        "worked": sorted(worked, key=lambda r: r["name"]),
        "left": worklist["left"],
        "spend": {
            "pass_total_usd": total,
            "ceiling_usd": CEILING_USD,
            "rate_per_name_usd": round(total / len(rated), 6) if rated else 0.0,
            "rated_names": len(rated),
        },
        "score_drops": score_drops,
        "census_after": census_after,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    HTML_OUT.write_text(render_html(payload))
    print("wrote", OUT)
    print("wrote", HTML_OUT)
    print(json.dumps({k: payload[k] for k in ("worked_count", "left_count")}, indent=1))
    print(json.dumps(payload["spend"], indent=1))


def render_html(payload):
    """The delivered page: forms before and after, names worked and left, spend."""

    esc = html_lib.escape
    parts = []
    add = parts.append

    def table(head, rows, cls=""):
        out = ['<table class="%s">' % cls, "<thead><tr>"]
        out.extend("<th>%s</th>" % esc(str(cell)) for cell in head)
        out.append("</tr></thead><tbody>")
        for row in rows:
            out.append("<tr>")
            out.extend("<td>%s</td>" % cell for cell in row)
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    add('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">')
    add("<title>Family and operator wording pass</title>")
    add(
        "<style>"
        "body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
        "max-width:1180px;margin:2.4em auto;padding:0 1em;color:#1c1c1c}"
        "h1{font-size:1.7em;margin-bottom:.2em}h2{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.2em}"
        "table{border-collapse:collapse;width:100%;margin:.8em 0;font-size:.92em}"
        "th,td{text-align:left;padding:.32em .5em;border-bottom:1px solid #eee;vertical-align:top}"
        "th{font-weight:600;background:#fafafa}"
        "td.num{text-align:right;font-variant-numeric:tabular-nums}"
        "code{background:#f4f4f4;padding:.1em .3em;border-radius:3px}"
        "details{margin:.5em 0 .9em}summary{cursor:pointer;color:#225}"
        ".muted{color:#666}.bad{color:#a11a1a}"
        "</style></head><body>"
    )
    add("<h1>Family and operator wording pass</h1>")
    spend = payload["spend"]
    add(
        '<p class="muted">generated %s &middot; %d names worked &middot; %d left with a '
        "reason &middot; %.4f USD against a %.2f USD ceiling</p>"
        % (
            esc(payload["generated_at"]),
            payload["worked_count"],
            payload["left_count"],
            spend["pass_total_usd"],
            spend["ceiling_usd"],
        )
    )

    probe = payload["probe"]
    add("<h2>Cost probe</h2>")
    add(
        "<p>One whole small family, <code>%s</code>, was run before any other spend. "
        "It settled %d names for %.4f USD, a rate of %.4f USD a name against the "
        "%.2f USD a name line.</p>"
        % (
            esc(probe["family"]),
            probe["names"],
            probe["spend_usd"],
            probe["rate_per_name_usd"],
            probe["line_usd"],
        )
    )

    add("<h2>Species families</h2>")
    add(
        "<p>A family is divergent when its members carry more than one description "
        "template or more than one unit, the measure the closing census takes. The "
        "table reads the same measure after the pass.</p>"
    )
    rows = []
    for family in payload["families"]:
        rows.append(
            [
                "<code>%s</code>" % esc(family["family"]),
                '<span class="num">%d</span>' % family["member_count"],
                '<span class="num">%d &rarr; %d</span>'
                % (family["template_count_before"], family["template_count_after"]),
                '<span class="num">%d &rarr; %d</span>'
                % (family["unit_count_before"], family["unit_count_after"]),
                '<span class="num">%d</span>' % family["worked_count"],
                '<span class="num">%.4f</span>' % family["spend_usd"],
            ]
        )
    add(
        table(
            ["family", "members", "templates", "units", "names worked", "spend [USD]"],
            rows,
        )
    )
    for family in payload["families"]:
        add("<details><summary>%s &mdash; wordings</summary>" % esc(family["family"]))
        add("<p class=\"muted\">before</p><ol>")
        for item in family["templates_before"]:
            add("<li>(%d) %s</li>" % (item["count"], esc(item["text"])))
        add("</ol><p class=\"muted\">after</p><ol>")
        for item in family["templates_after"]:
            add("<li>(%d) %s</li>" % (item["count"], esc(item["text"])))
        add("</ol></details>")

    add("<h2>Operators</h2>")
    add(
        "<p>A name that applies an operator carries the stored operator sentence "
        "verbatim; a form the pipeline cannot match is counted as omitted.</p>"
    )
    rows = []
    for operator in payload["operators"]:
        rows.append(
            [
                "<code>%s</code>" % esc(operator["operator"]),
                '<span class="num">%d</span>' % operator["name_count_after"],
                '<span class="num">%d &rarr; %d</span>'
                % (operator["form_count_before"] or 0, operator["form_count_after"] or 0),
                '<span class="num">%d &rarr; %d</span>'
                % (operator["omitted_before"], operator["omitted_after"]),
                '<span class="num">%d</span>' % operator["worked_count"],
                '<span class="num">%.4f</span>' % operator["spend_usd"],
            ]
        )
    add(
        table(
            ["operator", "names", "forms", "omitted", "names worked", "spend [USD]"],
            rows,
        )
    )
    for operator in payload["operators"]:
        add(
            "<details><summary>%s &mdash; stored sentence</summary><p>%s</p>"
            "<p class=\"muted\">forms after</p><ul>"
            % (esc(operator["operator"]), esc(operator["stored_sentence"] or ""))
        )
        for form, members in (operator["forms_after"] or {}).items():
            add("<li><code>%s</code> (%d)</li>" % (esc(form), len(members)))
        add("</ul></details>")

    add("<h2>Names worked</h2>")
    add(
        "<p>Every name this pass settled, with its review score before and after "
        "and the text it carries now.</p>"
    )
    rows = []
    for record in payload["worked"]:
        rows.append(
            [
                "<code>%s</code>" % esc(record["name"]),
                esc(record["outcome"]),
                '<span class="num">%s &rarr; %s</span>'
                % (
                    "%.3f" % record["score_before"] if record.get("score_before") is not None else "&mdash;",
                    "%.3f" % record["score_after"] if record.get("score_after") is not None else "&mdash;",
                ),
                '<span class="num">%.4f</span>' % record.get("spend_usd", 0.0),
            ]
        )
    add(table(["name", "outcome", "docs score", "spend [USD]"], rows))
    for record in payload["worked"]:
        add(
            "<details><summary><code>%s</code></summary>"
            "<p class=\"muted\">description before</p><p>%s</p>"
            "<p class=\"muted\">description after</p><p>%s</p>"
            "<p class=\"muted\">documentation before</p><p>%s</p>"
            "<p class=\"muted\">documentation after</p><p>%s</p></details>"
            % (
                esc(record["name"]),
                esc(record.get("description_before") or ""),
                esc(record.get("description_after") or ""),
                esc((record.get("documentation_before") or "")[:4000]),
                esc((record.get("documentation_after") or "")[:4000]),
            )
        )

    add("<h2>Names left</h2>")
    add("<p>Names the pass did not settle, each with the reason it was left.</p>")
    rows = []
    for item in payload["left"]:
        rows.append(
            [
                "<code>%s</code>" % esc(item["name"]),
                esc(item["reason"]),
                esc(item.get("detail", "")),
            ]
        )
        pass
    add(table(["name", "reason", "detail"], rows))

    add("<h2>Spend</h2>")
    rows = [
        ["pass total", '<span class="num">%.4f</span>' % spend["pass_total_usd"]],
        ["ceiling", '<span class="num">%.2f</span>' % spend["ceiling_usd"]],
        [
            "measured rate",
            '<span class="num">%.4f USD a name over %d names</span>'
            % (spend["rate_per_name_usd"], spend["rated_names"]),
        ],
    ]
    for family in payload["families"]:
        rows.append(
            ["family %s" % esc(family["family"]), '<span class="num">%.4f</span>' % family["spend_usd"]]
        )
    for operator in payload["operators"]:
        rows.append(
            ["operator %s" % esc(operator["operator"]), '<span class="num">%.4f</span>' % operator["spend_usd"]]
        )
    add(table(["item", "value"], rows))

    if payload.get("census_after"):
        add("<h2>Closing census, re-read from the live graph</h2>")
        rows = []
        for key, value in payload["census_after"]["families"].items():
            rows.append([esc(key), '<span class="num">%s</span>' % esc(str(value))])
        for operator, counts in payload["census_after"]["operators"].items():
            for key, value in counts.items():
                rows.append(
                    ["%s %s" % (esc(operator), esc(key)), '<span class="num">%s</span>' % esc(str(value))]
                )
        for key, value in payload["census_after"].get("population", {}).items():
            rows.append(
                ["population %s" % esc(key), '<span class="num">%s</span>' % esc(str(value))]
            )
        add(table(["measure", "value"], rows))

    add("</body></html>")
    return "\n".join(parts)


def census():
    """Re-read the census counts from the live graph through the committed measures.

    The committed census owns its own output file, so this phase prints rather
    than writing: the same measures, applied to the rows read now.
    """

    module, rows = live_rows_all()
    before = json.loads(CENSUS_BEFORE.read_text())["phases"]["live"]
    after_families = module.family_census(rows)
    forms_before = {
        operator: before["operators"][operator]["form_count"] for operator in module.OPERATORS
    }
    after_operators = module.operator_census(rows, forms_before)
    population = module.population_census(rows)
    print(
        json.dumps(
            {
                "families": {
                    key: after_families[key]
                    for key in (
                        "family_count",
                        "member_count",
                        "one_wording_count",
                        "divergent_count",
                    )
                },
                "operators": {
                    operator: {
                        key: after_operators[operator][key]
                        for key in ("name_count", "form_count", "omitted_count")
                    }
                    for operator in module.OPERATORS
                },
                "population": {
                    "species_names": population["species_names"],
                    "departing_count": len(population["departing"]),
                    "mentioning_neutral": population["mentioning_neutral"],
                    "silent_on_the_sum": population["silent_on_the_sum"],
                },
            },
            indent=1,
        )
    )


def figure():
    """Draw the before/after wordings and spend as one figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(OUT.read_text())
    families = payload["families"]
    operators = payload["operators"]

    before = "#929292"
    after = "#2f6f9f"

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    labels = [f["family"].replace("<X>", "X") for f in families]
    y = range(len(families))
    ry = range(len(operators))
    height = 0.38
    left.barh(
        [i - height / 2 for i in y],
        [f["template_count_before"] for f in families],
        height=height,
        color=before,
        label="before",
    )
    left.barh(
        [i + height / 2 for i in y],
        [f["template_count_after"] for f in families],
        height=height,
        color=after,
        label="after",
    )
    left.set_yticks(list(y))
    left.set_yticklabels(labels, fontsize=11)
    left.invert_yaxis()
    left.set_xlabel("description templates in the family", fontsize=16)
    left.tick_params(axis="x", labelsize=14)
    left.spines["top"].set_visible(False)
    left.spines["right"].set_visible(False)

    right.barh(
        [i - height / 2 for i in ry],
        [o["omitted_before"] for o in operators],
        height=height,
        color=before,
        label="before",
    )
    right.barh(
        [i + height / 2 for i in ry],
        [o["omitted_after"] for o in operators],
        height=height,
        color=after,
        label="after",
    )
    right.set_yticks(list(ry))
    right.set_yticklabels(
        [o["operator"].replace("_", " ") for o in operators], fontsize=14
    )
    right.invert_yaxis()
    right.set_xlabel("names omitting the\nstored sentence", fontsize=16)
    right.tick_params(axis="x", labelsize=14)
    right.spines["top"].set_visible(False)
    right.spines["right"].set_visible(False)

    handles, legend_labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=14,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "wordings-before-after.svg"
    fig.savefig(path, format="svg", metadata={"Date": None})
    print("wrote", path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["plan", "status", "report", "census", "figure"])
    args = parser.parse_args(argv)
    {"plan": plan, "status": status, "report": report, "census": census, "figure": figure}[
        args.phase
    ]()


if __name__ == "__main__":
    main()
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
import ast
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
SCORE_FLOOR = 0.85
SOURCELESS_RECORDED = 10
PROBE_FAMILY = "ratio_of_<X>_density_to_electron_density"
PROBE_LINE_USD = 0.10

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
    from iter_codex.graph.client import GraphClient

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


def operator_of(name):
    """The operators *name* carries, from the census module's own list."""

    return [operator for operator in census_module().OPERATORS if operator in name]


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
            if record.get("name") and record.get("outcome") is not None:
                records[record["name"]] = record
    return records


def live_rows_all():
    from iter_codex.graph.client import GraphClient

    module = census_module()
    with GraphClient() as client:
        return module, [dict(row) for row in client.query(module.LIVE_QUERY)]


def log_score(record):
    """The review score an edit's own log recorded, for a candidate that did not land."""

    path = record.get("log")
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hits = re.findall(r"[│|]\s*(0\.\d+|1\.0+)\s*[│|]", text)
    return float(hits[-1]) if hits else None


def form_is_stored(form, sentence):
    """Whether a censused form is the stored sentence, ignoring punctuation case."""

    def flat(text):
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    return flat(form) == flat(sentence)


def source_evidence(names):
    """The DD source paths and their descriptions, for names that need a rename."""

    from iter_codex.graph.client import GraphClient

    query = """
    MATCH (sn:StandardName) WHERE sn.id IN $names
    OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
    WITH sn,
         collect(DISTINCT coalesce(src.dd_path, src.id, '')) AS paths,
         collect(DISTINCT coalesce(src.description, '')) AS texts
    RETURN sn.id AS id,
           [p IN paths WHERE p <> ''] AS paths,
           [t IN texts WHERE t <> ''] AS texts
    ORDER BY sn.id
    """
    with GraphClient() as client:
        return {
            dict(record)["id"]: dict(record) for record in client.query(query, names=sorted(names))
        }


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
    left_by_name = {item["name"]: item for item in worklist["left"]}

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
                "non_stored_after": [
                    {
                        "text": text[:300],
                        "names": sorted(names),
                        "left": [
                            left_by_name[name]
                            for name in sorted(names)
                            if name in left_by_name
                        ],
                    }
                    for text, names in back["forms"].items()
                    if text != "<omitted>" and not form_is_stored(text, front["stored_sentence"])
                ],
                "omitted_names_after": sorted(back["forms"].get("<omitted>", [])),
                "omitted_left": [
                    left_by_name[name]
                    for name in sorted(back["forms"].get("<omitted>", []))
                    if name in left_by_name
                ],
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

    outcomes = collections.Counter(r["outcome"] for r in worked)
    held = []
    for record in worked:
        if record["outcome"] == "accepted":
            continue
        if record["outcome"] == "drafted":
            why = (
                "the edit reached its time bound with a draft stored and no review "
                "verdict, so the earlier text stands"
            )
        elif record["outcome"] == "exhausted":
            why = "review used its refine budget without a candidate above the threshold"
        else:
            why = "review scored the candidate below the threshold, so the earlier text stands"
        held.append(
            {
                "name": record["name"],
                "outcome": record["outcome"],
                "why": why,
                "log_score": log_score(record)
                if record.get("score_after") is None
                else record.get("score_after"),
                "score_before": record.get("score_before"),
                "score_after": record.get("score_after"),
                "spend_usd": record["spend_usd"],
            }
        )
    held.sort(key=lambda item: (-(item["log_score"] or 0), item["name"]))

    worked_names = {record["name"] for record in worked}
    unrouted = [
        {
            "name": name,
            "operator": entry["operator"],
            "form": form["text"],
            "why": (
                "the work list reached operator names through the omitted bucket, "
                "and this name paraphrases its operator in its own words instead, "
                "so it was neither worked nor listed as left"
            ),
        }
        for entry in operators
        for form in entry["non_stored_after"]
        for name in form["names"]
        if name not in worked_names and name not in left_by_name
    ]

    rename_left = [
        item
        for item in worklist["left"]
        if item["reason"].startswith("source includes neutrals")
    ]
    evidence = source_evidence([item["name"] for item in rename_left]) if rename_left else {}
    rename_candidates = [
        {
            "name": item["name"],
            "detail": item["detail"],
            "source_paths": evidence.get(item["name"], {}).get("paths", []),
            "source_texts": [
                text[:320]
                for text in (evidence.get(item["name"], {}).get("texts") or [])[:2]
            ],
        }
        for item in rename_left
    ]

    left_by_reason = collections.Counter(item["reason"] for item in worklist["left"])
    selection = {
        "species_family_count": before["families"]["family_count"],
        "species_family_members": before["families"]["member_count"],
        "families_at_one_wording_before": before["families"]["one_wording_count"],
        "species_names_departing_before": len(before["population"]["departing"]),
        "operator_omitted_names_before": sum(
            before["operators"][operator]["omitted_count"] for operator in module.OPERATORS
        ),
        "selected": sum(len(batch["names"]) for batch in worklist["batches"]),
        "selected_reason": (
            "the pass carries only wordings that are ruled and stored: the settled "
            "species population sentence and each operator's stored sentence. A "
            "family with no stored sentence would need a template written by hand, "
            "which this pass does not do, so its members are listed with that reason "
            "instead of converted"
        ),
        "left_by_reason": dict(left_by_reason.most_common()),
        "left_total": len(worklist["left"]),
        "worked_total": len(worked),
        "worked_note": (
            "the worked total includes the three probe-family names probed "
            "before the work list was settled, so it exceeds the selected count"
        ),
        "scores": {
            "lowest_after": min(
                (r["score_after"] for r in worked if r.get("score_after") is not None),
                default=None,
            ),
            "below_floor": sum(
                1
                for r in worked
                if r.get("score_after") is not None and r["score_after"] < SCORE_FLOOR
            ),
            "dropped": len(score_drops),
            "worked_total": len(worked),
            "with_score_before": sum(1 for r in worked if r.get("score_before") is not None),
            "with_score_after": sum(1 for r in worked if r.get("score_after") is not None),
        },
    }

    sourceless_before = before.get("sourceless", {}).get("count")
    with_source = None
    try:
        from iter_codex.graph.client import GraphClient

        with GraphClient() as client:
            sourceless_after = module.sourceless(client)
        with_source = sourceless_after
    except Exception as error:  # noqa: BLE001 - the count is evidence, not a gate
        with_source = {"error": str(error)[:200], "count": None, "names": []}

    population_after = live["population"]
    done_when = {
        "read_at": now(),
        "clauses": [
            {
                "clause": "every species family at one stored wording",
                "measure": "families whose members carry one description template and one unit",
                "before": "%d of %d"
                % (
                    before["families"]["one_wording_count"],
                    before["families"]["family_count"],
                ),
                "after": "%d of %d"
                % (
                    after_families["one_wording_count"],
                    after_families["family_count"],
                ),
                "state": "not met",
                "note": (
                    "a family reaches one wording only through a stored sentence; "
                    "the families without one are listed with that reason"
                ),
            },
            {
                "clause": "species names stating the ruled population sentence",
                "measure": "species names whose definition departs from the statement",
                "before": str(len(before["population"]["departing"])),
                "after": str(len(population_after["departing"])),
                "state": "departing names remain",
            },
            {
                "clause": "one stored form per operator",
                "measure": "distinct forms each operator is written in, and names omitting it",
                "before": ", ".join(
                    "%s %d forms / %d omitted"
                    % (
                        operator,
                        before["operators"][operator]["form_count"],
                        before["operators"][operator]["omitted_count"],
                    )
                    for operator in module.OPERATORS
                ),
                "after": ", ".join(
                    "%s %d forms / %d omitted"
                    % (
                        operator,
                        after_operators[operator]["form_count"],
                        after_operators[operator]["omitted_count"],
                    )
                    for operator in module.OPERATORS
                ),
                "state": "reduced, not one",
                "note": (
                    "the form count includes the omitted bucket, so a count of two "
                    "with a non-zero omitted count means the stored sentence plus "
                    "the names that still do not state it"
                ),
                "remaining_non_stored_forms": [
                    {
                        "operator": entry["operator"],
                        "text": form["text"],
                        "names": form["names"],
                    }
                    for entry in operators
                    for form in entry["non_stored_after"]
                ],
                "remaining_omitted_names": {
                    entry["operator"]: entry["omitted_names_after"]
                    for entry in operators
                    if entry["omitted_names_after"]
                },
            },
            {
                "clause": "live names with no producer edge at or below the recorded ten",
                "measure": "iter-codex sn sourceless",
                "before": str(sourceless_before),
                "after": str(with_source.get("count")),
                "state": "within the recorded bound"
                if (with_source.get("count") or 0) <= SOURCELESS_RECORDED
                else "above the recorded bound",
            },
            {
                "clause": "no name left below the publishing score threshold unlisted",
                "measure": (
                    f"documentation review score after each edit, against {SCORE_FLOOR:.2f}"
                ),
                "before": (
                    f"{selection['scores']['with_score_before']} of "
                    f"{selection['scores']['worked_total']} worked names carry a score "
                    "before"
                ),
                "after": (
                    f"{selection['scores']['with_score_after']} of "
                    f"{selection['scores']['worked_total']} worked names carry a score "
                    f"after; {selection['scores']['below_floor']} are below "
                    f"{SCORE_FLOOR:.2f}, all among the {len(held)} held"
                ),
                "state": "met"
                if {
                    r["name"]
                    for r in worked
                    if r.get("score_after") is not None and r["score_after"] < SCORE_FLOOR
                }
                <= {item["name"] for item in held}
                else "not met",
                "note": (
                    "a name whose score would fall below the floor keeps its earlier "
                    "text, so it is held rather than landed; names with no review "
                    "score are the ones whose edit never reached a scored candidate"
                ),
            },
            {
                "clause": "spend inside the ceiling at the measured rate",
                "measure": "LLMCost spend per edit joined to that edit's own run id",
                "before": "0 USD",
                "after": (
                    f"{total:.6f} USD of {CEILING_USD:.2f} at "
                    f"{(total / len(rated) if rated else 0.0):.6f} a name over "
                    f"{len(rated)} rated names"
                ),
                "state": "met" if total <= CEILING_USD else "above the ceiling",
            },
        ],
        "sourceless_names": with_source.get("names", []),
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
        "done_when": done_when,
        "selection": selection,
        "outcomes": dict(outcomes),
        "held": held,
        "rename_candidates": rename_candidates,
        "unrouted": unrouted,
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

    done = payload["done_when"]
    add("<h2>Done-when, read from the live graph</h2>")
    add(
        '<p class="muted">counts re-read at %s through the closing census measures; '
        "the before column is the census the pass started from</p>" % esc(done["read_at"])
    )
    add(
        table(
            ["clause", "measure", "before", "after", "state"],
            [
                [
                    esc(clause["clause"]),
                    esc(clause["measure"]),
                    esc(clause["before"]),
                    esc(clause["after"]),
                    (
                        '<span class="bad">%s</span>' % esc(clause["state"])
                        if "not" in clause["state"] or "above" in clause["state"]
                        else esc(clause["state"])
                    )
                    + (
                        '<br><span class="muted">%s</span>' % esc(clause["note"])
                        if clause.get("note")
                        else ""
                    ),
                ]
                for clause in done["clauses"]
            ],
        )
    )
    if done.get("sourceless_names"):
        add(
            "<p>Names with no producer edge: %s</p>"
            % ", ".join("<code>%s</code>" % esc(name) for name in done["sourceless_names"])
        )

    selection = payload["selection"]
    add("<h2>What the pass took, and what it left</h2>")
    add(
        table(
            ["measure", "count"],
            [
                ["species families", str(selection["species_family_count"])],
                ["species family members", str(selection["species_family_members"])],
                [
                    "families already at one wording",
                    str(selection["families_at_one_wording_before"]),
                ],
                [
                    "species names departing from the ruled statement",
                    str(selection["species_names_departing_before"]),
                ],
                [
                    "operator names omitting the stored sentence",
                    "all five together, %d" % selection["operator_omitted_names_before"],
                ],
                [
                    "names the pass worked",
                    "%d <span class=\"muted\">(%s)</span>"
                    % (selection["worked_total"], esc(selection["worked_note"])),
                ],
                ["names left with a reason", str(selection["left_total"])],
            ],
        )
    )
    add("<p>%s</p>" % esc(selection["selected_reason"]))
    add(
        table(
            ["reason a name was left", "count"],
            [
                [esc(reason), '<span class="num">%d</span>' % count]
                for reason, count in selection["left_by_reason"].items()
            ],
        )
    )

    outcomes = payload["outcomes"]
    add("<h2>Outcomes</h2>")
    scores = payload["selection"]["scores"]
    add(
        '<p class="muted">%d names carry a lower review score after the pass than '
        "before, and %d worked names sit below the %.2f floor; every one of them "
        "is held rather than accepted, as the verify phase asserts, and the lowest "
        "score after the pass is %s. A revised definition scores lower whenever the "
        "reviewer weighs the stored sentences differently, so a drop is recorded "
        "rather than treated as a defect; the floor is what holds a name back.</p>"
        % (
            scores["dropped"],
            scores["below_floor"],
            SCORE_FLOOR,
            "%.4f" % scores["lowest_after"] if scores["lowest_after"] else "unrecorded",
        )
    )
    add(
        '<p class="muted">Score coverage: '
        f"{scores['with_score_before']} of {scores['worked_total']} worked names carry "
        f"a score before the edit and {scores['with_score_after']} carry one after. "
        "The remainder are names whose earlier record carried no score, or whose "
        "edit never reached a scored candidate; both are counted here rather than "
        "left implicit in the worked table.</p>"
    )
    add(
        table(
            ["outcome", "count"],
            [
                [esc(name), '<span class="num">%d</span>' % count]
                for name, count in sorted(outcomes.items())
            ],
        )
    )
    if payload["held"]:
        add("<h3>Names held back, with why</h3>")
        add(
            table(
                ["name", "outcome", "score", "why"],
                [
                    [
                        "<code>%s</code>" % esc(item["name"]),
                        esc(item["outcome"]),
                        esc("%.2f" % item["log_score"]) if item["log_score"] else "",
                        esc(item["why"]),
                    ]
                    for item in payload["held"]
                ],
            )
        )
    if payload["rename_candidates"]:
        add("<h3>Names whose source needs a rename instead</h3>")
        add(
            "<p>The bare-species sentence would misstate these populations, so their "
            "text is left alone: the name is wrong, not the wording. Each needs a "
            "rename with the source evidence below.</p>"
        )
        add(
            table(
                ["name", "why", "source paths", "source description"],
                [
                    [
                        "<code>%s</code>" % esc(item["name"]),
                        esc(item["detail"]),
                        "<br>".join("<code>%s</code>" % esc(p) for p in item["source_paths"])
                        or '<span class="muted">no path recorded</span>',
                        "<br>".join(esc(t) for t in item["source_texts"])
                        or '<span class="muted">no description recorded</span>',
                    ]
                    for item in payload["rename_candidates"]
                ],
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
        add("</ul>")
        if operator["non_stored_after"]:
            add("<p class=\"muted\">forms still not the stored sentence</p><ul>")
            for form in operator["non_stored_after"]:
                add(
                    f"<li><code>{esc(form['text'])}</code><br>"
                    + ", ".join(
                        f"<code>{esc(name)}</code>" for name in form["names"]
                    )
                    + "</li>"
                )
            add("</ul>")
        if operator["omitted_left"]:
            add("<p class=\"muted\">names omitting the sentence, with why</p><ul>")
            for item in operator["omitted_left"]:
                add(
                    f"<li><code>{esc(item['name'])}</code> &mdash; "
                    f"{esc(item['reason'])}</li>"
                )
            add("</ul>")
        add("</details>")

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
    add(table(["name", "reason", "detail"], rows))
    if payload.get("unrouted"):
        add("<h3>Names outside the work list that still carry a non-stored form</h3>")
        add(
            table(
                ["name", "operator", "form it carries", "why it was not routed"],
                [
                    [
                        f"<code>{esc(item['name'])}</code>",
                        esc(item["operator"]),
                        f"<code>{esc(item['form'])}</code>",
                        esc(item["why"]),
                    ]
                    for item in payload["unrouted"]
                ],
            )
        )

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


def census_measure_copies(
    source: str, *, species: frozenset[str], family_pattern: str | None
) -> tuple[list[int], list[str]]:
    """Species lists and family templates a module writes for itself.

    The census module owns both measures, so a copy under any name is drift:
    a list literal holding three or more of its species names, whatever the
    assignment is called, and a function holding its species pattern, whatever
    the template is called.
    """
    tree = ast.parse(source)
    lists = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.List | ast.Tuple | ast.Set)
        and len(
            species
            & {
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, str)
            }
        )
        >= 3
    ]
    templates = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and family_pattern is not None
        and any(
            isinstance(inner, ast.Constant)
            and isinstance(inner.value, str)
            and inner.value == family_pattern
            for inner in ast.walk(node)
        )
    ]
    return lists, templates


def verify(delivered_path=None):
    """Re-derive every count the delivered report states, from the live graph.

    Run after any change to the report. It reads the catalog again through the
    committed census measures and fails on a stated count that no longer
    matches, on spend above the ceiling or disagreeing with the records it sums,
    on a name whose live documentation score sits below the floor without being
    listed as held, and on a name the after-census calls non-stored that is
    neither worked nor listed as left. Two live reads over named populations
    plus one catalog read; nothing is written.
    """

    delivered_path = Path(delivered_path or OUT)
    delivered = json.loads(delivered_path.read_text())
    stated = delivered["census_after"]
    module, rows = live_rows_all()
    before = json.loads(CENSUS_BEFORE.read_text())["phases"]["live"]
    forms_before = {operator: before["operators"][operator]["form_count"] for operator in module.OPERATORS}
    families = module.family_census(rows)
    operators = module.operator_census(rows, forms_before)
    population = module.population_census(rows)
    row_by_name = {row["id"]: row for row in rows}

    checks = []

    def check(label, reported, live):
        checks.append((label, reported, live))

    # The species list and the family measure are the closing census's, so this
    # script must not carry a second copy that can drift from them. Scanning
    # this module's own literals catches a copy under any name — a renamed
    # SPECIES_NAMES, an inlined tuple, or a re-implemented template holding the
    # census's species pattern — where a test for two remembered identifiers
    # caught only the two spellings that were deleted.
    copied_lists, copied_templates = census_measure_copies(
        Path(__file__).read_text(encoding="utf-8"),
        species=frozenset(module.SPECIES),
        family_pattern=getattr(
            getattr(module, "SPECIES_NAME_RE", None), "pattern", None
        ),
    )
    check("this script carries no second species list", [], copied_lists)
    check("this script carries no second family template", [], copied_templates)

    for key in ("family_count", "member_count", "one_wording_count", "divergent_count"):
        check(f"species families {key}", stated["families"][key], families[key])
    for key in ("species_names", "departing_count", "mentioning_neutral", "silent_on_the_sum"):
        check(f"species population {key}", stated["population"][key], population[key])
    for operator in module.OPERATORS:
        for key in ("name_count", "form_count", "omitted_count"):
            check(
                f"{operator} {key}",
                stated["operators"][operator][key],
                operators[operator][key],
            )

    from iter_codex.graph.client import GraphClient

    with GraphClient() as client:
        sourceless_after = module.sourceless(client)
    sourceless_clause = next(
        clause
        for clause in delivered["done_when"]["clauses"]
        if clause["clause"].startswith("live names with no producer edge")
    )
    check(
        "sourceless live names stated",
        int(sourceless_clause["after"]),
        sourceless_after["count"],
    )
    checks.append(
        (
            "sourceless live names within the recorded bound",
            True,
            sourceless_after["count"] <= SOURCELESS_RECORDED,
        )
    )

    records = settled_records()
    worked = list(records.values())
    total = round(sum(record.get("spend_usd") or 0 for record in worked), 6)
    rated = [record for record in worked if (record.get("spend_usd") or 0) > 0]
    check("worked names", delivered["worked_count"], len(worked))
    check("spend total", delivered["spend"]["pass_total_usd"], total)
    check("spend within the ceiling", True, total <= CEILING_USD)
    check("rated names", delivered["spend"]["rated_names"], len(rated))
    check(
        "spend rate",
        delivered["spend"]["rate_per_name_usd"],
        round(total / len(rated), 6) if rated else 0.0,
    )

    with_after = [record for record in worked if record.get("score_after") is not None]
    below_floor = [record for record in with_after if record["score_after"] < SCORE_FLOOR]
    held_names = {item["name"] for item in delivered["held"]}
    check("names carrying a score after", delivered["selection"]["scores"]["with_score_after"], len(with_after))
    check("names below the score floor", delivered["selection"]["scores"]["below_floor"], len(below_floor))
    check(
        "every name below the floor is held",
        set(),
        {record["name"] for record in below_floor} - held_names,
    )
    check(
        "every live score below the floor is held",
        set(),
        {
            record["name"]
            for record in worked
            if (row_by_name.get(record["name"], {}).get("score_docs") or 0) < SCORE_FLOOR
            and row_by_name.get(record["name"], {}).get("score_docs") is not None
            and record["name"] not in held_names
        },
    )
    drops_reported = {item["name"] for item in delivered["score_drops"]}
    drops_live = {
        record["name"]
        for record in with_after
        if record.get("score_before") is not None
        and record["score_after"] < record["score_before"]
    }
    check("score drops listed (count)", len(drops_reported), len(drops_live))
    check("score drops listed (same names)", True, drops_reported == drops_live)

    worklist = json.loads(WORKLIST.read_text())
    left = worklist["left"]
    check("left names", delivered["left_count"], len(left))
    check("every left name carries a non-empty reason", 0, sum(1 for item in left if not item["reason"]))
    accounted = set(records) | {item["name"] for item in left}
    unrouted = {item["name"] for item in delivered["unrouted"]}
    check(
        "every non-stored after-form name is worked, left or unrouted",
        set(),
        {
            name
            for entry in operators.values()
            for form, names in entry["forms"].items()
            for name in names
            if not form_is_stored(form, entry["stored_sentence"])
            and name not in accounted
            and name not in unrouted
        },
    )

    for label, reported, live in checks:
        print(("PASS" if reported == live else "FAIL"), label, "reported", reported, "live", live)
    mismatched = [item for item in checks if item[1] != item[2]]
    print("checked", len(checks), "mismatched", len(mismatched))
    return 1 if mismatched else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["plan", "status", "report", "census", "figure", "verify"]
    )
    parser.add_argument(
        "--delivered",
        default=None,
        help="report to verify; defaults to the file this script writes",
    )
    args = parser.parse_args(argv)
    if args.phase == "verify":
        return verify(args.delivered)
    {"plan": plan, "status": status, "report": report, "census": census, "figure": figure}[
        args.phase
    ]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

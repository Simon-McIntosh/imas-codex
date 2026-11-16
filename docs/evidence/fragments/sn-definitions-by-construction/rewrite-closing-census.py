# frozen-record: this script no longer runs as committed. Its committed reading
# is the record of the revision it ran at, not a reading this tree can
# regenerate. Ran at 19daa7a6db6e1a0212c2be6cd4c82b11957e0cd5, the last revision this file changed in.
# Names the tree no longer defines: imas_codex.graph.temp_neo4j.
# The format is deliberate: it is a record, not a description of a
# defect still to repair.
"""Closing census of the live standard-name catalog after the wording rewrite.

Measures every clause of the section's done-when over every live name. The live
graph is read only. The one place this script writes is a disposable Neo4j it
starts itself, which carries a planted family and shows the census reporting
the divergence planted in it.

Phases: ``live``, ``seat``, ``planted``, ``figure``. Each phase writes the
census JSON before the next begins, so a phase that fails keeps the phases
before it.
"""

from __future__ import annotations
import argparse
import asyncio
import collections
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = HERE / "rewrite-closing-census.json"
SEAT_PARTIAL = HERE / ".rewrite-closing-census-seat.jsonl"
REWRITE = HERE / "live-rewrite-bulk.json"
RENDERED = HERE / "rendered-wording-census.json"
FIGURE = (
    REPO
    / "docs"
    / "figures"
    / "sn-definitions-by-construction"
    / "sdbc-rewrite-closing-census"
    / "wordings-before-after.svg"
)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OPERATORS = (
    "flux_surface_averaged",
    "volume_averaged",
    "line_averaged",
    "line_integrated",
    "time_derivative",
)

#: The species list the published family census used, kept identical so the
#: reading before the first write and the reading after the rewrite are the
#: same measure.
SPECIES = [
    "helium_3", "helium_4", "deuterium_tritium",
    "deuterium", "tritium",
    "helium", "hydrogen", "lithium",
    "beryllium", "boron", "carbon",
    "nitrogen", "oxygen", "neon",
    "argon", "krypton", "xenon",
    "iron", "tungsten", "total_hydrogenic",
]


def _species_words(species):
    words = [name.replace("_", "-") for name in species]
    return words + ["deuterium–tritium", "hydrogenic"]


BASELINE_SPECIES = [name for name in SPECIES if name != "helium"]
SPECIES_WORDS = _species_words(SPECIES)
BASELINE_SPECIES_WORDS = _species_words(BASELINE_SPECIES)

SPECIES_NAME_RE = re.compile(r"(?<![a-z])(" + "|".join(SPECIES) + r")(?![a-z])")
SPECIES_WORD_RE = re.compile("|".join(SPECIES_WORDS), re.IGNORECASE)
BASELINE_NAME_RE = re.compile(r"(?<![a-z])(" + "|".join(BASELINE_SPECIES) + r")(?![a-z])")
BASELINE_WORD_RE = re.compile("|".join(BASELINE_SPECIES_WORDS), re.IGNORECASE)
NEUTRAL_WORD_RE = re.compile(r"\bneutrals?\b|z\s*=\s*0", re.IGNORECASE)

#: The readings taken before the first write, each beside the artifact holding
#: it, so every count this run produces is shown against its predecessor.
EARLIER = {
    "operators": {
        "flux_surface_averaged": {"forms": 8, "names": 32},
        "source": "rendered-wording-census.json published_baseline",
    },
    "families": {
        "divergent": 20,
        "total": 20,
        "members": 281,
        "source": "published-family-census.txt, recorded in rendered-wording-census.json",
    },
    "population": {
        "species_names": 82,
        "stating_charge_state_sum": 27,
        "mentioning_neutral": 22,
        "source": "live-rewrite-bulk.json censuses, read before the first rewrite",
    },
    "units": {
        "resolved_duplicates_note": "each of these was resolved to its recorded spelling",
        "pre_resolution_spellings": {
            "torque": {"N.m": 23, "kg.m^2.s^-2": 6},
            "torque_density": {"N.m^-2": 77, "kg.m^-1.s^-2": 9},
            "source_rate": {"s^-1": 48, "Hz": 11},
            "rate": {"s^-1": 21, "Hz": 6},
        },
        "pre_resolution_dimension_mismatches": 41,
        "source": "the unit-spelling section's committed reading",
    },
    "scores": {"publishing_threshold": 0.85, "source": "defaults.DEFAULT_MIN_SCORE"},
    "seat": {
        "judged": 1977,
        "flagged": 429,
        "cost_per_name_usd": 9e-05,
        "source": "name-definition-seat-catalog.json",
    },
    "sourceless": {"count": 10, "source": "live-rewrite-bulk.json sourceless"},
    "spend": {
        "rewrite_total_usd": 70.2411,
        "rewrite_ceiling_usd": 90.0,
        "section_ceiling_usd": 150.0,
        "source": "live-rewrite-bulk.json spend",
    },
}

LIVE_QUERY = """
MATCH (sn:StandardName)
WHERE coalesce(sn.name_stage, '') <> 'superseded'
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
WITH sn, collect(DISTINCT u.id) AS unit_ids
OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
WITH sn, unit_ids, collect(DISTINCT child.id) AS children
RETURN sn.id AS id,
       coalesce(sn.description, '') AS description,
       coalesce(sn.documentation, '') AS documentation,
       coalesce(unit_ids[0], sn.unit, '') AS unit,
       coalesce(sn.physics_domain, '') AS physics_domain,
       coalesce(sn.kind, '') AS kind,
       coalesce(sn.name_stage, '') AS name_stage,
       coalesce(sn.docs_stage, '') AS docs_stage,
       sn.reviewer_score_docs AS score_docs,
       toString(sn.updated_at) AS updated_at,
       children
ORDER BY sn.id
"""

WATERMARK_QUERY = """
MATCH (sn:StandardName)
RETURN count(sn) AS nodes, toString(max(sn.updated_at)) AS last_write
"""

SOURCELESS_QUERY = """
MATCH (sn:StandardName)
WHERE coalesce(sn.name_stage, '') <> 'superseded'
  AND coalesce(sn.unit, '') <> ''
WITH sn, size([(src:StandardNameSource)-[:PRODUCED_NAME]->(sn) | 1]) AS edges
WHERE edges = 0
RETURN sn.id AS id
ORDER BY sn.id
"""


def load_census() -> dict[str, Any]:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {
        "schema": "rewrite-closing-census/1",
        "node": "sdbc-rewrite-closing-census",
        "plan_section": "sn-definitions-by-construction#s6",
        "phases": {},
    }


def save_census(census: dict[str, Any]) -> None:
    tmp = OUT.with_name(OUT.name + ".partial")
    tmp.write_text(json.dumps(census, indent=1, sort_keys=True) + "\n")
    tmp.replace(OUT)


def live_rows(client: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in client.query(LIVE_QUERY):
        row = dict(record)
        row["children"] = [child for child in (row.get("children") or []) if child]
        rows.append(row)
    return rows


def conventions_for(name: str) -> Any:
    from imas_standard_names.grammar.parser import ParseError
    from imas_standard_names.grammar.terms import select_conventions

    try:
        return select_conventions(name)
    except ParseError:
        return None


def operator_sentence(operator: str, rows: list[dict[str, Any]]) -> str | None:
    """The stored sentence every name carrying *operator* has to state."""

    from imas_codex.standard_names.docs_gates import shared_part_sentences

    for row in rows:
        if operator not in row["id"]:
            continue
        conventions = conventions_for(row["id"])
        if conventions is None:
            continue
        for anchor, text in shared_part_sentences(row["id"], conventions):
            if anchor == "concept" + ":" + operator:
                return text
    return None


def operator_census(rows, before):
    """Definition wordings per operator, beside the reading before the rewrite."""

    from imas_codex.standard_names.harmonize import shared_part_form_counts

    out = {}
    for operator in OPERATORS:
        names = [row for row in rows if operator in row["id"]]
        sentence = operator_sentence(operator, rows)
        entry = {
            "name_count": len(names),
            "stored_sentence": sentence,
            "before_form_count": before.get(operator),
        }
        if sentence and names:
            forms = shared_part_form_counts(names, lambda entry_row: sentence)
            entry["form_count"] = len(forms)
            entry["forms"] = {
                form: sorted(members) for form, members in forms.items()
            }
            entry["omitted_count"] = len(forms.get("<omitted>", []))
        else:
            entry["form_count"] = None
            entry["forms"] = {}
            entry["omitted_count"] = 0
            entry["note"] = "no name in this population carries the operator"
        out[operator] = entry
    return out


def family_template(name, name_re=None):
    return (name_re or SPECIES_NAME_RE).sub("<X>", name)


def family_census(rows, min_members=3, name_re=None, word_re=None):
    """Description wordings per species family: the measure the baseline used."""

    name_re = name_re or SPECIES_NAME_RE
    word_re = word_re or SPECIES_WORD_RE
    families = collections.defaultdict(list)
    for row in rows:
        families[family_template(row["id"], name_re)].append(row)
    species_families = {}
    for key, members in families.items():
        if "<X>" in key and len(members) >= min_members:
            species_families[key] = members
    divergent = {}
    agreed = []
    description_divergent = 0
    unit_divergent = 0
    for key in sorted(species_families):
        members = species_families[key]
        templates = collections.Counter(
            re.sub(r"\s+", " ", word_re.sub("<X>", row["description"] or "")).strip()
            for row in members
        )
        units = collections.Counter((row.get("unit") or "") for row in members)
        description_divergent += int(len(templates) > 1)
        unit_divergent += int(len(units) > 1)
        if len(templates) > 0 and (len(templates) > 1 or len(units) > 1):
            divergent[key] = {
                "member_count": len(members),
                "template_count": len(templates),
                "unit_count": len(units),
                "templates": [
                    {"count": count, "text": text[:240]}
                    for text, count in templates.most_common()
                ],
                "units": dict(units.most_common()),
                "names": sorted(row["id"] for row in members),
            }
        else:
            agreed.append(key)
    return {
        "family_count": len(species_families),
        "member_count": sum(len(members) for members in species_families.values()),
        "one_wording_count": len(agreed),
        "divergent_count": len(divergent),
        "description_divergent_count": description_divergent,
        "unit_divergent_count": unit_divergent,
        "measure": (
            "a family is divergent when its members carry more than one description "
            "template or more than one unit, the measure the census before the first write used"
        ),
        "one_wording_families": sorted(agreed),
        "divergent_families": divergent,
    }


def population_census(rows):
    """Every species name against the population the lead ruled."""

    from imas_codex.standard_names.docs_gates import (
        SPECIES_POPULATION_SENTENCE,
        is_species_density_name,
        population_contradiction,
    )
    from imas_codex.standard_names.harmonize import written_part_form

    species = [row for row in rows if is_species_density_name(row["id"])]
    departing = []
    stating = 0
    mentioning_neutral = 0
    for row in species:
        text = (row["description"] or "") + "\n" + (row["documentation"] or "")
        if NEUTRAL_WORD_RE.search(text):
            mentioning_neutral += 1
        contradiction = population_contradiction(row["id"], text)
        form = written_part_form(row["documentation"], SPECIES_POPULATION_SENTENCE)
        states = form != "<omitted>"
        stating += int(states)
        if contradiction:
            departing.append(
                {
                    "name": row["id"],
                    "reason": "states the opposite population",
                    "phrase": contradiction,
                }
            )
        elif not states:
            departing.append(
                {
                    "name": row["id"],
                    "reason": "does not write the settled wording",
                    "phrase": "no sentence in the definition overlaps the settled wording",
                    "names_the_sum": bool(
                        re.search(r"charge.?state|summed over", text, re.IGNORECASE)
                    ),
                }
            )
    unparaphrased = [
        entry["name"]
        for entry in departing
        if entry["reason"] == "does not write the settled wording" and not entry["names_the_sum"]
    ]
    paraphrased = [
        entry["name"]
        for entry in departing
        if entry["reason"] == "does not write the settled wording" and entry["names_the_sum"]
    ]
    return {
        "ruled_statement": SPECIES_POPULATION_SENTENCE,
        "paraphrasing_the_sum": len(paraphrased),
        "paraphrasing_names": sorted(paraphrased),
        "silent_on_the_sum": len(unparaphrased),
        "silent_names": sorted(unparaphrased),
        "species_names": len(species),
        "states_settled_sentence": stating,
        "mentioning_neutral": mentioning_neutral,
        "departing_count": len(departing),
        "departing": sorted(departing, key=lambda entry: entry["name"]),
    }


def score_census(rows, rewrite):
    """Documentation scores of the rewritten names against the publish threshold."""

    from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

    by_id = {row["id"]: row for row in rows}
    held = []
    absent = []
    for entry in rewrite["before_after"]:
        name = entry["name"]
        row = by_id.get(name)
        if row is None:
            absent.append(name)
            continue
        score = row.get("score_docs")
        if score is None:
            held.append({"name": name, "score": None, "why": "no documentation score"})
        elif float(score) < DEFAULT_MIN_SCORE:
            held.append(
                {"name": name, "score": float(score), "why": "below threshold"}
            )
    scores = sorted(
        float(row["score_docs"]) for row in rows if row.get("score_docs") is not None
    )
    return {
        "threshold": DEFAULT_MIN_SCORE,
        "rewritten_names": len(rewrite["before_after"]),
        "live_names_scored": len(scores),
        "held_count": len(held),
        "held": held,
        "rewritten_absent_from_live": sorted(absent),
        "median_score_of_live": round(scores[len(scores) // 2], 2) if scores else None,
    }


def sourceless(client):
    names = [str(dict(record)["id"]) for record in client.query(SOURCELESS_QUERY)]
    return {
        "measure": (
            "live StandardName nodes carrying a unit and no incoming PRODUCED_NAME "
            "edge, counted by size() over the edges rather than by the copied "
            "source_paths property"
        ),
        "names": names,
        "count": len(names),
    }


def operator_phrase_re(operator):
    words = operator.split("_")
    words[-1] = words[-1][:6]
    return re.compile(r"[\s\-]+".join(words) + "[a-z]*", re.IGNORECASE)


def operator_phrase_census(rows):
    """Whether each operator's concept is named in the name's own text at all.

    A name can carry the operator and define it in its own words, which the
    stored sentence's wording test reads as an omission. This second reading
    asks only whether the operator is named, so the two together separate a
    paraphrase from a definition that never states the operator.
    """

    out = {}
    for operator in OPERATORS:
        pattern = operator_phrase_re(operator)
        names = [row for row in rows if operator in row["id"]]
        absent = [
            row["id"]
            for row in names
            if not pattern.search((row["description"] or "") + " " + (row["documentation"] or ""))
        ]
        out[operator] = {
            "name_count": len(names),
            "names_naming_the_operator": len(names) - len(absent),
            "names_not_naming_it": absent,
        }
    return out


def annotate_family_reasons(families, rewrite):
    """Say why each family still carries more than one wording.

    A family the rewrite never touched has a reason its population explains;
    one it touched only in part was never brought to one wording as a whole.
    """

    rewritten = {entry["name"] for entry in rewrite["before_after"]}
    for entry in families["divergent_families"].values():
        members = entry["names"]
        touched = sum(1 for name in members if name in rewritten)
        entry["rewritten_members"] = touched
        if touched == 0:
            entry["reason"] = "no member was in the rewrite population"
        elif touched < len(members):
            entry["reason"] = (
                "%d of %d members were rewritten; the family was never brought to "
                "one wording as a whole" % (touched, len(members))
            )
        else:
            entry["reason"] = "every member was rewritten and the descriptions still differ"


def rewritten_subset(rows, rewrite, before):
    """The same measures over the names the rewrite itself wrote."""
    names = {entry["name"] for entry in rewrite["before_after"]}
    subset = [row for row in rows if row["id"] in names]
    return {
        "name_count": len(subset),
        "operators": operator_census(subset, before),
        "families": family_census(subset),
    }


def phase_live():
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.audits import live_unit_spelling_census

    census = load_census()
    rewrite = json.loads(REWRITE.read_text())
    rendered = json.loads(RENDERED.read_text())
    before = {
        "flux_surface_averaged": rendered["published_baseline"][
            "flux_surface_averaged_forms"
        ]
    }
    with GraphClient() as client:
        watermark_before = dict(client.query(WATERMARK_QUERY)[0])
        rows = live_rows(client)
        print("live names read: %d" % len(rows), flush=True)
        result = {
            "watermark_before": watermark_before,
            "live_name_count": len(rows),
            "live_name_rule": "name_stage <> 'superseded', the unit census rule",
            "operators": operator_census(rows, before),
            "operator_phrases": operator_phrase_census(rows),
            "families": family_census(rows),
            "families_baseline_list": family_census(
                rows, name_re=BASELINE_NAME_RE, word_re=BASELINE_WORD_RE
            ),
            "rewritten_subset": rewritten_subset(rows, rewrite, before),
            "population": population_census(rows),
            "units": live_unit_spelling_census(client),
            "scores": score_census(rows, rewrite),
            "sourceless": sourceless(client),
            "earlier_readings": EARLIER,
        }
        watermark_after = dict(client.query(WATERMARK_QUERY)[0])
        result["watermark_after"] = watermark_after
        if watermark_before == watermark_after:
            result["live_graph_write"] = (
                "none: every statement issued to the live graph is a read"
            )
        else:
            result["live_graph_write"] = (
                "watermark moved during the run; reported, attributed to no one"
            )
    annotate_family_reasons(result["families"], rewrite)
    census["phases"]["live"] = result
    save_census(census)
    print("operators:", {op: e["form_count"] for op, e in result["operators"].items()}, flush=True)
    print("families divergent:", result["families"]["divergent_count"], "of", result["families"]["family_count"], flush=True)
    print("population departing:", result["population"]["departing_count"], flush=True)
    print("unit duplicates:", len(result["units"]["duplicate_quantities"]), flush=True)
    print("unit mismatches:", len(result["units"]["dimension_mismatches"]), flush=True)
    print("held:", result["scores"]["held_count"], flush=True)
    print("sourceless:", result["sourceless"]["count"], flush=True)


def phase_seat():
    import imas_codex.standard_names.review.name_definition_seat as seat
    from imas_codex.graph.client import GraphClient

    census = load_census()
    with GraphClient() as client:
        rows = live_rows(client)
    recorded = {}
    if SEAT_PARTIAL.exists():
        for line in SEAT_PARTIAL.read_text().splitlines():
            if line.strip():
                verdict = json.loads(line)
                recorded[verdict["name"]] = verdict
    live_ids = set(row["id"] for row in rows)
    recorded = dict((n, v) for n, v in recorded.items() if n in live_ids)
    remaining = [row for row in rows if row["id"] not in recorded]
    print("seat: recorded %d, remaining %d" % (len(recorded), len(remaining)), flush=True)
    cost = 0.0
    if remaining:
        items = []
        for row in remaining:
            items.append({
                "id": row["id"],
                "description": row["description"],
                "documentation": row["documentation"],
                "unit": row["unit"],
                "derived_children": row["children"],
            })
        verdicts, cost = asyncio.run(
            seat.judge_name_definitions(
                items, concurrency=12, progress_path=SEAT_PARTIAL,
            )
        )
        for verdict in verdicts:
            recorded[verdict["name"]] = verdict
    stage = dict((row["id"], row["name_stage"]) for row in rows)

    def stage_of(name):
        return stage.get(name)

    flagged = sorted(n for n, v in recorded.items() if v["verdict"] == "flag")
    unanswered = sorted(n for n, v in recorded.items() if v["verdict"] == "unanswered")
    by_qualifier = collections.Counter(recorded[n].get("qualifier") or "unanswered" for n in flagged)
    census["phases"]["seat"] = {
        "model": seat.review_model(),
        "prompt": "sn/review_name_definition_system",
        "judged": len(recorded) - len(unanswered),
        "flagged": len(flagged),
        "unanswered": len(unanswered),
        "unanswered_names": unanswered,
        "by_qualifier": dict(sorted(by_qualifier.items())),
        "cost_usd_this_run": round(cost, 4),
        "cost_usd": round(
            float(census["phases"].get("seat", {}).get("cost_usd", 0.0)) + cost, 4
        ),
        "flagged_names": flagged,
        "flagged_published": sum(
            1 for n in flagged if stage_of(n) == "published"
        ),
        "judged_published": sum(1 for n in recorded if stage_of(n) == "published"),
        "earlier": EARLIER["seat"],
    }
    save_census(census)
    print("seat judged %d flagged %d unanswered %d cost %.4f" % (len(recorded) - len(unanswered), len(flagged), len(unanswered), cost), flush=True)


def _free_port_pair():
    import socket

    sockets = [socket.socket(), socket.socket()]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return int(sockets[0].getsockname()[1]), int(sockets[1].getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()


def _start_neo4j():
    import os
    import tempfile

    from imas_codex.graph import temp_neo4j

    scratch = os.environ.get("TMPDIR") or None
    directory = tempfile.TemporaryDirectory(prefix="closing-census-", dir=scratch)
    bolt_port, http_port = _free_port_pair()
    process, log = temp_neo4j.start_temp_neo4j(
        Path(directory.name), bolt_port, http_port, load_dump=False
    )
    return directory, process, log, bolt_port


def _stop_neo4j(directory, process):
    try:
        process.terminate()
        process.wait(timeout=120)
    except Exception:
        traceback.print_exc()
    finally:
        directory.cleanup()


def _planted_members(sentence):
    members = []
    for species in ("argon", "iron", "boron", "neon", "helium"):
        description = "Number density of %s ions in the plasma." % species
        documentation = description + " " + sentence
        if species == "helium":
            description = "Number density of helium atoms in the plasma."
            documentation = (
                description
                + " The density counts ions and neutral atoms together, so the ions are "
                "not counted on their own."
            )
        members.append({
            "id": species + "_density",
            "description": description,
            "documentation": documentation,
        })
    return members


def phase_planted():
    """Plant a five-member family in a disposable graph and read it back.
"""

    from imas_codex.standard_names.docs_gates import (
        SPECIES_POPULATION_SENTENCE,
        population_contradiction,
    )
    from imas_codex.standard_names.harmonize import shared_part_form_counts

    census = load_census()
    directory, process, log, bolt_port = _start_neo4j()
    try:
        from neo4j import GraphDatabase
        members = _planted_members(SPECIES_POPULATION_SENTENCE)
        with GraphDatabase.driver("bolt://127.0.0.1:" + str(bolt_port)) as driver:
            with driver.session() as session:
                session.run(
                    "UNWIND $members AS m "
                    "CREATE (:StandardName {id: m.id, description: m.description, "
                    "documentation: m.documentation, unit: 'm^-3', "
                    "name_stage: 'accepted', physics_domain: 'edge_plasma_physics'})",
                    members=members,
                )
                rows = []
                result = session.run(
                    "MATCH (sn:StandardName) "
                    "RETURN sn.id AS id, sn.description AS description, "
                    "sn.documentation AS documentation ORDER BY sn.id"
                )
                for record in result:
                    rows.append(dict(record))
        forms = shared_part_form_counts(rows, lambda entry: SPECIES_POPULATION_SENTENCE)
        control_rows = [row for row in rows if row["id"] != "helium_density"]
        control = shared_part_form_counts(control_rows, lambda entry: SPECIES_POPULATION_SENTENCE)
        family = family_census(rows)
        contradictions = sorted(
            row["id"]
            for row in rows
            if population_contradiction(
                row["id"], (row["description"] or "") + "\n" + (row["documentation"] or "")
            )
        )
        census["phases"]["planted"] = {
            "graph": "a disposable Neo4j started, written and removed by this run",
            "family": sorted(row["id"] for row in rows),
            "divergent_member": "helium_density",
            "population_form_count": len(forms),
            "population_forms": dict((form, sorted(m)) for form, m in forms.items()),
            "control_form_count": len(control),
            "control_forms": dict((form, sorted(m)) for form, m in control.items()),
            "family_divergent_count": family["divergent_count"],
            "family_count": family["family_count"],
            "family_divergent_families": sorted(family["divergent_families"]),
            "population_contradiction_names": contradictions,
        }
        save_census(census)
        census_planted = census["phases"]["planted"]
        print("planted: forms %d, control forms %d" % (census_planted["population_form_count"], census_planted["control_form_count"]), flush=True)
    finally:
        _stop_neo4j(directory, process)


def _plot_style(plt):
    try:
        plt.style.use("data-ink")
    except OSError:
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
                "axes.labelsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 14,
                "font.size": 14,
            }
        )
    return {"after": "#2a6fb0", "before": "#929292", "limit": "#929292"}


def phase_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    census = load_census()
    live = census["phases"]["live"]
    rendered = json.loads(RENDERED.read_text())
    before_families = rendered["published_baseline"]["species_families_with_more_than_one_wording"]

    colours = _plot_style(plt)
    fig, (ax_op, ax_fam) = plt.subplots(1, 2, figsize=(14, 7.2))

    operators = live["operators"]
    labels = [op.replace("_", " ") for op in operators]
    after = [entry["form_count"] or 0 for entry in operators.values()]
    before_op = [entry.get("before_form_count") for entry in operators.values()]
    y = list(range(len(labels)))
    ax_op.barh([v + 0.19 for v in y], after, height=0.36, color=colours["after"], label="after the rewrite")
    ax_op.barh(
        [v - 0.19 for v in y],
        [b if b is not None else 0 for b in before_op],
        height=0.36,
        color=colours["before"],
        label="before the rewrite",
    )
    ax_op.set_yticks(y)
    ax_op.set_yticklabels(labels)
    ax_op.invert_yaxis()
    ax_op.set_xlabel("definition wordings [count]")
    ax_op.set_title("operator sentence")
    span = max(after + [b for b in before_op if b] or [1])
    ax_op.set_xlim(0, span * 1.25)
    for row, value in zip(y, after):
        ax_op.text(value + span * 0.03, row + 0.19, str(value), va="center", fontsize=14, color=colours["after"])
    for row, value in zip(y, before_op):
        if value is None:
            ax_op.text(span * 0.02, row - 0.19, "not measured", va="center", fontsize=11, color=colours["before"])
        else:
            ax_op.text(value + span * 0.03, row - 0.19, str(value), va="center", fontsize=14, color=colours["before"])

    families = live["families"]
    matches = re.findall(r"\d+", str(before_families))
    before_divergent = int(matches[0]) if matches else 0
    before_total = int(matches[1] if len(matches) > 1 else before_divergent)
    before_one = before_total - before_divergent
    groups = ["one wording\nshared by the family\nmembers", "more than one\nwording in the family"]
    before_vals = [before_one, before_divergent]
    after_vals = [families["one_wording_count"], families["divergent_count"]]
    x = [0.0, 1.0]
    ax_fam.bar([v - 0.18 for v in x], before_vals, width=0.34, color=colours["before"])
    ax_fam.bar([v + 0.18 for v in x], after_vals, width=0.34, color=colours["after"])
    ax_fam.set_xticks(x)
    ax_fam.set_xticklabels(groups)
    ax_fam.set_ylabel("species families [count]")
    ax_fam.set_title("species family description wordings")
    top = max(before_vals + after_vals + [1])
    ax_fam.set_ylim(0, top * 1.2)
    for pos, value in zip(x, before_vals):
        ax_fam.text(pos - 0.18, value + top * 0.03, str(value), ha="center", fontsize=14, color=colours["before"])
    for pos, value in zip(x, after_vals):
        ax_fam.text(pos + 0.18, value + top * 0.03, str(value), ha="center", fontsize=14, color=colours["after"])

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colours["before"]),
        plt.Rectangle((0, 0), 1, 1, color=colours["after"]),
    ]
    fig.legend(
        handles,
        [
            "before the first write, published names",
            "after the rewrite, every live name",
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg")
    plt.close(fig)
    census["phases"]["figure"] = {
        "path": str(FIGURE.relative_to(REPO)),
        "before_family_baseline": before_families,
        "before_operator_forms": dict((op, entry.get("before_form_count")) for op, entry in operators.items()),
        "after_operator_forms": dict((op, entry["form_count"]) for op, entry in operators.items()),
        "after_family_one_wording": families["one_wording_count"],
        "after_family_divergent": families["divergent_count"],
        "after_figure_count": families["family_count"],
    }
    save_census(census)
    print("figure written:", FIGURE, flush=True)


PHASES = {
    "live": phase_live,
    "planted": phase_planted,
    "figure": phase_figure,
    "seat": phase_seat,
}

#: Cheap evidence first, the paid seat last, so a run cut short keeps the
#: clauses that cost nothing to measure.
ORDER = ("live", "planted", "figure", "seat")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("live", "planted", "figure", "seat", "all"), default="all"
    )
    args = parser.parse_args(argv)
    names = ORDER if args.phase == "all" else (args.phase,)
    failed = []
    for name in names:
        print("=== phase %s" % name, flush=True)
        try:
            PHASES[name]()
        except Exception:
            traceback.print_exc()
            failed.append(name)
            print("=== phase %s FAILED" % name, flush=True)
            continue
        print("=== phase %s done" % name, flush=True)
    if failed:
        print("failed phases:", ", ".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

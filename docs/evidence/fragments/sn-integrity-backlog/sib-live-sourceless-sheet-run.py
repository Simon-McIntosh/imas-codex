"""Build a read-only adjudication sheet for live names without DD sources."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import classify_sourceless_standard_names
from imas_codex.tools.graph_search import GraphSearchTool
from imas_codex.units import units_are_equivalent

HERE = Path(__file__).resolve().parent
SHEET = HERE / "sib-live-sourceless-sheet.json"
PAGE = HERE / "sib-live-sourceless-sheet.html"
EXPECTED = {
    "toroidal_wave_vector_normalized_of_beam_tracing_beam",
    "electrostatic_potential_at_outboard_midplane",
    "electrostatic_potential_at_wall",
    "neutron_flux_due_to_fusion",
    "root_mean_square_spectral_width_of_spectrometer_channel",
    "count_normalized_at_detector_pixel",
    "bulk_electron_temperature_flux_surface_averaged_at_last_closed_flux_surface",
    "coolant_absorbed_energy_accumulated_of_plasma_facing_component",
    "flux_surface_averaged_parallel_electric_field_at_separatrix",
    "volume_averaged_electron_number_density_over_scrape_off_layer",
}

# A recommendation is a proposed lead ruling, not a graph mutation.
RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "bulk_electron_temperature_flux_surface_averaged_at_last_closed_flux_surface": {
        "action": "bind",
        "target": "summary/local/separatrix_average/t_e/value",
        "holder_wrong_reason": "The current holder, electron_temperature_at_separatrix, explicitly excludes a flux-surface average in its documentation.",
        "reason": "The separatrix is the last closed flux surface, and this DD scalar is its flux-surface-averaged electron temperature in eV. Its accepted holder describes a local temperature and explicitly excludes this average, so that binding needs to be reassigned.",
    },
    "coolant_absorbed_energy_accumulated_of_plasma_facing_component": {
        "action": "hold",
        "reason": "Calorimetry energy_cumulated is for a generic calorimetry component, while this name requires a plasma-facing component; energy_total also includes post-pulse cooling. Neither path identifies the same component and accumulation interval together.",
    },
    "count_normalized_at_detector_pixel": {
        "action": "hold",
        "reason": "The dimensionless detector paths are raw counts or response functions, without the same-frame reference count in the denominator. The nuclei_n path is a target-nuclei count, yet its holder calls it an atomic fraction; that suspect binding does not supply a normalized pixel count.",
    },
    "electrostatic_potential_at_outboard_midplane": {
        "action": "hold",
        "reason": "The potential profiles are flux-surface averages and the probe paths are at probe positions, not the outboard midplane. A radial-coordinate path locates that midplane but measures metres rather than potential.",
    },
    "electrostatic_potential_at_wall": {
        "action": "hold",
        "reason": "The V-valued profile paths are flux-surface averages, and Langmuir-probe potential is at the probe rather than immediately adjacent to a plasma-facing wall. The unchained holder was judged a different quantity.",
    },
    "flux_surface_averaged_parallel_electric_field_at_separatrix": {
        "action": "bind",
        "target": "summary/local/separatrix_average/e_field_parallel/value",
        "reason": "The DD scalar is the signed separatrix flux-surface average of the parallel projection, defined as ⟨E·B⟩/B0, with the same V.m^-1 unit. The nearby separatrix/e_field_parallel path is local rather than surface averaged.",
    },
    "neutron_flux_due_to_fusion": {
        "action": "retire",
        "target": "total_neutron_rate",
        "reason": "The accepted holder already represents the aggregate neutron production rate from DT, DD and TT fusion channels, including the reacting classes; both name a plasma-integrated rate in Hz despite the word flux here. The alternative neutron_diagnostic/neutron_flux_total is held by exhausted power_due_to_fusion_reactions, whose power description is wrong for an s^-1 path.",
    },
    "root_mean_square_spectral_width_of_spectrometer_channel": {
        "action": "hold",
        "reason": "The available Lorentzian instrument-function scale is a full width at half height, not an RMS width. Wavelength grids and the centre wavelength also measure different properties despite sharing metres.",
    },
    "toroidal_wave_vector_normalized_of_beam_tracing_beam": {
        "action": "retire",
        "target": "normalized_toroidal_wave_vector_of_beam_tracing_beam",
        "reason": "The accepted holder describes the same signed toroidal direction cosine k_phi/|k| in the same cylindrical frame and holds both normalized toroidal DD paths. The unnormalized k_phi paths have inverse-length units and are distinct.",
    },
    "volume_averaged_electron_number_density_over_scrape_off_layer": {
        "action": "hold",
        "reason": "The edge density paths are spatial profiles, while the core global average stops at the LCFS and excludes the scrape-off layer. The lineage pedestal offset and SOL density decay length are neither a volume average over the SOL.",
    },
}

SUPPLEMENTAL_PATHS = {
    "bulk_electron_temperature_flux_surface_averaged_at_last_closed_flux_surface": [
        "summary/local/separatrix_average/t_e/value"
    ],
    "neutron_flux_due_to_fusion": ["summary/fusion/neutron_rates/total/value"],
}


def read(gc: GraphClient, cypher: str, **params: Any) -> list[dict[str, Any]]:
    started = time.monotonic()
    rows = gc.query(cypher, **params)
    elapsed = time.monotonic() - started
    if elapsed >= 10:
        raise RuntimeError(f"graph read exceeded ten seconds ({elapsed:.2f}s)")
    return rows


def prior_record(name: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def row(doc: dict[str, Any], field: str) -> Any:
        collection = doc.get(field, [])
        if isinstance(collection, dict):
            return collection.get(name)
        if isinstance(collection, list):
            return next(
                (
                    item
                    for item in collection
                    if item.get("name", item.get("id")) == name
                ),
                None,
            )
        return None

    search = records["search"]
    held = records["held"]
    unchained = records["unchained"]
    lineage = records["lineage"]
    return {
        "searched_sources": {
            "worklist": row(search, "worklist"),
            "answer": row(search, "rejudged") or row(search, "judged"),
            "referral": row(search, "corrected_referred_to_lead")
            or row(search, "referred_to_lead"),
            "no_surviving_candidate": row(search, "corrected_no_surviving_candidate")
            or row(search, "no_surviving_candidate"),
        },
        "held_search_hits": row(held, "per_name"),
        "unchained_holders": row(unchained, "per_name"),
        "lineage_sources": {
            "worklist": row(lineage, "worklist"),
            "answer": row(lineage, "judged"),
            "referral": row(lineage, "referred_to_lead"),
        },
    }


def hit_record(hit: Any, name_unit: str | None) -> dict[str, Any]:
    unit = hit.units
    return {
        "path": hit.path,
        "score": hit.score,
        "documentation": hit.documentation,
        "description": hit.description,
        "unit": unit,
        "physics_domain": hit.physics_domain,
        "data_type": hit.data_type,
        "unit_agrees": units_are_equivalent(name_unit, unit),
        "unit_status": (
            "unavailable"
            if not unit
            else "agrees"
            if units_are_equivalent(name_unit, unit)
            else "mismatch"
        ),
        "holders": [],
    }


async def search_both(
    tool: GraphSearchTool,
    name: str,
    description: str,
    name_unit: str | None,
    major: int,
) -> dict[str, list[dict[str, Any]]]:
    searches = {}
    for label, query in (
        ("description", description or name.replace("_", " ")),
        ("name", name.replace("_", " ")),
    ):
        result = await tool.search_dd_paths(
            query, max_results=8, dd_version=major, include_summary_ids=True
        )
        if result.error:
            raise RuntimeError(f"DD search failed for {name} ({label}): {result.error}")
        searches[label] = [hit_record(hit, name_unit) for hit in result.hits[:8]]
    merged = {}
    for label, hits in searches.items():
        for hit in hits:
            candidate = merged.get(hit["path"])
            if candidate is None or hit["score"] > candidate["score"]:
                merged[hit["path"]] = {**hit, "found_by": [label]}
            elif label not in candidate["found_by"]:
                candidate["found_by"].append(label)
    searches["strongest"] = [
        hit["path"]
        for hit in sorted(
            merged.values(), key=lambda hit: (-hit["score"], hit["path"])
        )[:8]
    ]
    return searches


def attach_holders(gc: GraphClient, rows: list[dict[str, Any]]) -> None:
    paths = sorted(
        {
            hit["path"]
            for row in rows
            for hits in (
                row["fresh_search"]["description"],
                row["fresh_search"]["name"],
            )
            for hit in hits
        }
        | {hit["path"] for row in rows for hit in row["supplemental_candidates"]}
    )
    source_ids = ["dd:" + path for path in paths]
    held = read(
        gc,
        """MATCH (source:StandardNameSource)
        WHERE source.id IN $source_ids
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(holder:StandardName)
        WHERE holder IS NOT NULL AND NOT holder.status IN $terminal
        RETURN source.id AS source_id,
               collect(DISTINCT CASE WHEN holder IS NULL THEN null ELSE
                 {name: holder.id, description: holder.description,
                  unit: holder.unit, stage: holder.name_stage} END) AS holders
        ORDER BY source_id LIMIT 500""",
        source_ids=source_ids,
        terminal=["deprecated", "superseded"],
    )
    holders = {row["source_id"].removeprefix("dd:"): row["holders"] for row in held}
    for row in rows:
        for hits in [
            row["fresh_search"]["description"],
            row["fresh_search"]["name"],
            row["supplemental_candidates"],
        ]:
            for hit in hits:
                hit["holders"] = holders.get(hit["path"], [])
                hit["live_holders"] = [
                    holder
                    for holder in hit["holders"]
                    if holder["stage"] != "exhausted"
                ]
                hit["holder_unit_mismatch"] = any(
                    bool(holder.get("unit") and hit["unit"])
                    and not units_are_equivalent(holder["unit"], hit["unit"])
                    for holder in hit["holders"]
                )
                hit["holder_unit_unavailable"] = any(
                    not holder.get("unit") or not hit["unit"]
                    for holder in hit["holders"]
                )


def supplemental_candidates(
    gc: GraphClient, name: str, name_unit: str | None
) -> list[dict[str, Any]]:
    paths = SUPPLEMENTAL_PATHS.get(name, [])
    if not paths:
        return []
    found = read(
        gc,
        """MATCH (dd:IMASNode) WHERE dd.id IN $paths
        RETURN dd.id AS path, dd.documentation AS documentation,
               dd.description AS description, dd.unit AS unit,
               dd.physics_domain AS physics_domain, dd.data_type AS data_type
        ORDER BY path LIMIT 20""",
        paths=paths,
    )
    if len(found) != len(paths):
        raise RuntimeError(f"supplemental DD path missing for {name}")
    return [
        {
            **hit,
            "score": None,
            "unit_agrees": units_are_equivalent(name_unit, hit["unit"]),
            "unit_status": (
                "agrees" if units_are_equivalent(name_unit, hit["unit"]) else "mismatch"
            ),
            "holders": [],
            "inspection_reason": "Exact sibling path found in the DD branch after a related search hit",
        }
        for hit in found
    ]


def collect() -> dict[str, Any]:
    records = {
        key: json.loads((HERE / filename).read_text(encoding="utf-8"))
        for key, filename in {
            "search": "sib-restore-searched-sources.json",
            "held": "sib-settle-held-search-hits.json",
            "unchained": "sib-settle-unchained-holders.json",
            "lineage": "sib-restore-lineage-sources.json",
        }.items()
    }
    gc = GraphClient()
    try:
        classified = classify_sourceless_standard_names(gc)
        live = sorted(
            (
                row
                for row in classified
                if row["name_stage"] != "exhausted"
                and row["population"]
                not in {"derived_parent", "orphan_awaiting_retirement"}
            ),
            key=lambda row: row["id"],
        )
        current = read(
            gc,
            "MATCH (version:DDVersion {is_current: true}) "
            "RETURN version.id AS id, version.major AS major LIMIT 2",
        )
        if len(current) != 1:
            raise RuntimeError(
                f"expected exactly one current DD version, got {current}"
            )
        facts = read(
            gc,
            """MATCH (name:StandardName)
            WHERE name.id IN $ids
            RETURN name.id AS id, name.description AS description,
                   name.unit AS unit, name.physics_domain AS physics_domain,
                   name.name_stage AS stage, name.documentation AS documentation
            ORDER BY id LIMIT 100""",
            ids=[row["id"] for row in live],
        )
        if len(facts) != len(live):
            raise RuntimeError(f"name facts incomplete: {len(facts)}/{len(live)}")
        positive = read(
            gc,
            """MATCH (source:StandardNameSource {id: $source_id})
            MATCH (source)-[:PRODUCED_NAME]->(holder:StandardName)
            RETURN holder.id AS id LIMIT 10""",
            source_id="dd:core_profiles/profiles_1d/electrons/temperature",
        )
        if not positive:
            raise RuntimeError("holder read positive control found no known holder")
        by_id = {row["id"]: row for row in facts}
        tool = GraphSearchTool(gc)
        rows = []
        for classified_row in live:
            name = classified_row["id"]
            record = by_id[name]
            searches = asyncio.run(
                search_both(
                    tool,
                    name,
                    record["description"] or "",
                    record["unit"],
                    current[0]["major"],
                )
            )
            rows.append(
                {
                    "name": name,
                    "record": record,
                    "classification": classified_row,
                    "prior_steps": prior_record(name, records),
                    "fresh_search": searches,
                    "supplemental_candidates": supplemental_candidates(
                        gc, name, record["unit"]
                    ),
                }
            )
        attach_holders(gc, rows)
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "dd_version": current[0]["id"],
            "census_total": len(classified),
            "live_non_derived_sourceless_count": len(rows),
            "expected_only": sorted(EXPECTED - {row["name"] for row in rows}),
            "live_only": sorted({row["name"] for row in rows} - EXPECTED),
            "holder_positive_control": positive,
            "graph_writes": 0,
            "llm_calls": 0,
            "spend_usd": 0,
            "rows": rows,
        }
    finally:
        gc.close()


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def prior_html(prior: dict[str, Any]) -> str:
    parts = []
    searched = prior["searched_sources"]
    work = searched["worklist"] or {}
    if work:
        kept = (
            ", ".join(
                f"{esc(item['path'])} ({esc(item.get('unit'))})"
                for item in work.get("candidates", [])
            )
            or "none"
        )
        dropped = (
            "; ".join(
                f"{esc(item['path'])}: {esc(item.get('reason'))}"
                for item in work.get("dropped_hits", [])
            )
            or "none"
        )
        answer = (searched["answer"] or {}).get("answer", {})
        referral = searched["referral"] or {}
        parts.append(
            f"<p><strong>DD restore:</strong> kept {kept}; dropped {dropped}. "
            f"Answer: {esc(answer.get('choice'))}; referral: "
            f"{esc(referral.get('reason'))}.</p>"
        )
    missing = searched["no_surviving_candidate"]
    if missing:
        parts.append(
            f"<p><strong>No surviving search candidate:</strong> {esc(missing)}.</p>"
        )
    held = prior["held_search_hits"]
    if held:
        holders = (
            ", ".join(
                f"{esc(item['id'])} on {esc(item.get('held_path'))}"
                for item in held.get("holders", [])
            )
            or "none"
        )
        parts.append(
            f"<p><strong>Held hits:</strong> {holders}; "
            f"answer {esc(held.get('choice'))} "
            f"(P={esc(held.get('probability'))}, confidence={esc(held.get('confidence'))}); "
            f"{esc(held.get('reason'))}.</p>"
        )
    unchained = prior["unchained_holders"]
    if unchained:
        parts.append(
            f"<p><strong>Unchained holder:</strong> {esc(unchained.get('holder'))} "
            f"on {esc(', '.join(unchained.get('shared_paths', [])))}; "
            f"answer {esc(unchained.get('choice'))}; {esc(unchained.get('reason'))}.</p>"
        )
    lineage = prior["lineage_sources"]
    line_work = lineage["worklist"] or {}
    if line_work:
        paths = ", ".join(line_work.get("candidate_paths", [])) or "none"
        dropped = "; ".join(line_work.get("dropped_candidates", [])) or "none"
        answer = (lineage["answer"] or {}).get("answer", {})
        parts.append(
            f"<p><strong>Lineage:</strong> {esc(paths)}; "
            f"dropped {esc(dropped)}; "
            f"answer {esc(answer.get('choice'))}.</p>"
        )
    return "".join(parts) or "<p>No prior candidate recorded.</p>"


def hits_html(hits: list[dict[str, Any]]) -> str:
    items = []
    for hit in hits:
        holders = (
            ", ".join(
                f"{holder['name']} ({holder['stage']})" for holder in hit["holders"]
            )
            or "none"
        )
        agreement = "unit " + hit["unit_status"]
        suspect = "; holder unit mismatch" if hit["holder_unit_mismatch"] else ""
        if hit["holder_unit_unavailable"]:
            suspect += "; holder unit unavailable"
        items.append(
            "<li><code>"
            + esc(hit["path"])
            + "</code> — "
            + esc(hit["documentation"] or hit["description"])
            + "; unit "
            + esc(hit["unit"])
            + "; domain "
            + esc(hit["physics_domain"])
            + "; holder "
            + esc(holders)
            + "; live holders "
            + esc(len(hit["live_holders"]))
            + "; "
            + agreement
            + suspect
            + "</li>"
        )
    return "<ol>" + "".join(items) + "</ol>"


def render(data: dict[str, Any]) -> str:
    count = data["recommendation_counts"]
    table_rows = []
    for row in data["rows"]:
        rec = row["recommendation"]
        record = row["record"]
        searches = row["fresh_search"]
        by_path = {
            hit["path"]: hit for hit in searches["description"] + searches["name"]
        }
        table_rows.append(
            "<tr><th scope='row'><code>"
            + esc(row["name"])
            + "</code><br>"
            + esc(record["stage"])
            + "</th><td>"
            + esc(record["description"])
            + "; <strong>"
            + esc(record["unit"])
            + "</strong>; "
            + esc(record["physics_domain"])
            + "<p>Documentation: "
            + esc((record["documentation"] or "")[:500] or "none recorded")
            + "</p></td><td>"
            + prior_html(row["prior_steps"])
            + "</td><td>"
            + hits_html([by_path[path] for path in searches["strongest"]])
            + (
                "<p><strong>Inspected DD sibling:</strong></p>"
                + hits_html(row["supplemental_candidates"])
                if row["supplemental_candidates"]
                else ""
            )
            + "<details><summary>Both searches (top eight each)</summary>"
            + "<p>Description query:</p>"
            + hits_html(searches["description"])
            + "<p>Name query:</p>"
            + hits_html(searches["name"])
            + "</details></td><td><strong>"
            + esc(rec["action"])
            + "</strong> "
            + esc(rec.get("target") or "")
            + "<p>"
            + esc(rec["reason"])
            + "</p></td></tr>"
        )
    mismatch = ""
    if data["expected_only"] or data["live_only"]:
        mismatch = (
            "<p>Live set differs from the expected list. Expected only: "
            + esc(data["expected_only"])
            + "; live only: "
            + esc(data["live_only"])
            + ".</p>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Live standard names without a DD source — adjudication sheet</title>"
        "<body><main><h1>Live standard names without a DD source</h1>"
        f"<p>Current DD {esc(data['dd_version'])}; {len(table_rows)} live, non-derived "
        "sourceless names. Proposed rulings: "
        f"retire {count['retire']}, bind {count['bind']}, hold {count['hold']}. "
        "No graph write or LLM call; USD 0. COCOS 17 sign conventions.</p>"
        + mismatch
        + "<table><thead><tr><th>Name and stage</th><th>Record</th>"
        "<th>Prior answers</th><th>Fresh DD candidates</th>"
        "<th>Proposed ruling and physics reason</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></main></body></html>\n"
    )


def validate(data: dict[str, Any]) -> None:
    names = [row["name"] for row in data["rows"]]
    if (
        len(names) != len(set(names))
        or len(names) != data["live_non_derived_sourceless_count"]
    ):
        raise RuntimeError("sheet rows do not match the live sourceless census")
    for row in data["rows"]:
        name = row["name"]
        searches = row["fresh_search"]
        if not searches["description"] or not searches["name"]:
            raise RuntimeError(f"missing DD search results for {name}")
        if any(
            len(searches[label]) > 8 for label in ("description", "name", "strongest")
        ):
            raise RuntimeError(f"DD search exceeds top-eight bound for {name}")
        rec = row["recommendation"]
        action = rec.get("action")
        if action not in {"retire", "bind", "hold"} or not rec.get("reason"):
            raise RuntimeError(f"invalid recommendation for {name}")
        if action == "hold":
            if rec.get("target"):
                raise RuntimeError(f"held name has a target: {name}")
            continue
        target = rec.get("target")
        if not target:
            raise RuntimeError(f"recommendation lacks a target: {name}")
        candidates = {
            hit["path"]: hit
            for hits in [
                searches["description"],
                searches["name"],
                row["supplemental_candidates"],
            ]
            for hit in hits
        }
        if action == "bind":
            hit = candidates.get(target)
            if hit is None or not hit["unit_agrees"] or hit["data_type"] == "STRUCTURE":
                raise RuntimeError(f"binding lacks a unit-matched DD scalar: {name}")
            if hit["live_holders"] and not rec.get("holder_wrong_reason"):
                raise RuntimeError(f"binding would take a live holder's path: {name}")
        if action == "retire" and not any(
            holder["name"] == target and holder["stage"] != "exhausted"
            for hit in candidates.values()
            for holder in hit["holders"]
        ):
            raise RuntimeError(f"retirement target is not a live DD holder: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    data = collect()
    if args.collect:
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    missing = {row["name"] for row in data["rows"]} - RECOMMENDATIONS.keys()
    if missing:
        raise RuntimeError(f"missing recommendations: {sorted(missing)}")
    for row in data["rows"]:
        row["recommendation"] = RECOMMENDATIONS[row["name"]]
    validate(data)
    data["recommendation_counts"] = dict(
        Counter(row["recommendation"]["action"] for row in data["rows"])
    )
    for action in ("retire", "bind", "hold"):
        data["recommendation_counts"].setdefault(action, 0)
    SHEET.write_text(
        json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    PAGE.write_text(render(data), encoding="utf-8")
    print(
        f"rows={len(data['rows'])} dd={data['dd_version']} "
        f"recommendations={data['recommendation_counts']} "
        f"expected_only={data['expected_only']} live_only={data['live_only']} "
        "graph_writes=0 llm_calls=0 spend_usd=0"
    )


if __name__ == "__main__":
    main()

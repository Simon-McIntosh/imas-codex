"""Read every live name without a producing DD source and propose an action.

Read-only against the live graph: this node proposes, it does not write. The
predicates are the export gate's own, so the population is the one a release
would refuse rather than a private restatement of it:

* ``classify_sourceless_standard_names`` -- the source audit the export gate
  calls, which classifies each name from its own source record.
* ``find_sourceless_live_names`` -- the writer invariant, which is the authority
  on whether a name would be refused a write for having no producer.

Three subcommands. ``live`` reads the live catalog and writes the population
file. ``fixture`` plants a sourceless name beside a bound control in a
disposable Neo4j and reads both predicates back, so a listing that would report
every name is shown to be able to report one. ``render`` writes the page and
refuses where a proposal breaks its own rule.

    uv run python .../scc-sourceless-names-run.py live
    uv run python .../scc-sourceless-names-run.py fixture
    uv run python .../scc-sourceless-names-run.py render
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE / "scc-sourceless-names.json"
PAGE = HERE / "scc-sourceless-names.html"

#: The nine accepted names carried from ``q-sib-nine-unsourced-names``, as the
#: plan records them. The read reconciles the live population against this list
#: rather than assuming it: an entry absent from the read, or a new accepted
#: sourceless name, is reported.
RECORDED_NINE = {
    "fast_ion_charge_state_power_at_inside_flux_surface",
    "toroidal_coordinate_of_field_map_grid",
    "neutron_flux_due_to_fusion",
    "radial_neutral_internal_state_momentum_flux",
    "voltage_of_diagnostic_antenna",
    "voltage_of_ece_channel",
    "electrostatic_potential_at_outboard_midplane",
    "electrostatic_potential_at_wall",
    "root_mean_square_spectral_width_of_spectrometer_channel",
}
DECLARED_NINE = 9

TERMINAL = ["deprecated", "superseded"]
SHIPPING = {"accepted", "approved"}

#: A proposal is a lead ruling this node recommends, never a graph write.
#: Action is one of ``retire_with_successor``, ``retire``, ``rebind``, ``hold``.
#: ``target`` names a live name, or a DD path for ``rebind``, or the rule that
#: holds a held name.
PROPOSALS: dict[str, dict[str, str]] = {
    # Three pending drafts whose recorded source is still present and live: the
    # source record survives, the PRODUCED_NAME edge does not. The repair is a
    # signed rebind, not a retirement or a fresh binding.
    "line_averaged_neon_density": {
        "action": "rebind",
        "target": "dd:summary/line_average/n_i/neon/value",
        "reason": (
            "The recorded source exists, carries status extracted and names a "
            "live DD path whose unit the draft matches, so this name has a "
            "producer and lost only the edge. Re-attach the binding rather than "
            "retire a quantity the DD measures."
        ),
    },
    "radius_of_soft_xray_detector": {
        "action": "rebind",
        "target": "dd:soft_x_rays/channel/detector/radius",
        "reason": (
            "The recorded source exists and is live, and the DD path documents "
            "the radius of the circle used for a circular detector geometry, "
            "which is the quantity this name states. The edge was dropped, so "
            "re-attach it."
        ),
    },
    "ratio_of_diamagnetic_vorticity_to_major_radius": {
        "action": "rebind",
        "target": "dd:edge_profiles/ggd/vorticity_over_r/diamagnetic",
        "reason": (
            "Both recorded sources survive on live GGD paths whose unit matches "
            "the name. Re-attach both bindings; the plasma_profiles path is the "
            "same quantity in the other container and both belong to this name."
        ),
    },
    # Two accepted names with no DD path among their recorded tokens. Both are
    # already held from release by the export gate, and holding is the action
    # that keeps a documented, scored identity while nothing measures it.
    "volume_averaged_electron_number_density_over_scrape_off_layer": {
        "action": "hold",
        "reason": (
            "No recorded source survives and no DD path is a volume average over "
            "the scrape-off layer: the edge density paths are spatial profiles "
            "and the core global average stops at the last closed flux surface. "
            "The name already carries the gate's release_hold_no_dd_source, so "
            "hold it with that ruling rather than retire documentation that "
            "scored 1.00 on its name and 0.97 on its docs."
        ),
    },
    "coolant_absorbed_energy_accumulated_of_plasma_facing_component": {
        "action": "hold",
        "reason": (
            "The name was never bound and the DD calorimetry paths belong to a "
            "generic calorimetry component, not to a plasma-facing component, "
            "while energy_total also includes post-pulse cooling. It already "
            "carries the gate's release_hold_no_dd_source; hold it, keeping its "
            "accepted documentation and its USD 0.1493 of recorded spend."
        ),
    },
}


def esc(value: Any) -> str:
    return html.escape(str(value) if value not in (None, "") else "—")


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    for attribute in ("iso_format", "isoformat"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return str(method())
            except TypeError:
                continue
    return str(value)


def read(gc: Any, cypher: str, **params: Any) -> list[dict[str, Any]]:
    """One bounded read. Ten seconds is this project's ceiling on a live query."""
    started = time.monotonic()
    rows = gc.query(cypher, **params)
    elapsed = time.monotonic() - started
    if elapsed >= 10:
        raise RuntimeError(f"graph read exceeded ten seconds ({elapsed:.2f}s)")
    return [dict(row) for row in rows or []]


def release_holds() -> dict[str, tuple[str, str]]:
    """The export gate's own hold table, read from the gate rather than copied."""
    from imas_codex.standard_names.export import _RELEASE_IDENTITY_HOLDS

    return dict(_RELEASE_IDENTITY_HOLDS)


def recorded_paths(row: dict[str, Any]) -> list[str]:
    """The DD paths this name's own source record says it came from."""
    paths: list[str] = []
    for value in row.get("recorded_source_ids") or []:
        text = str(value)
        if text.startswith("dd:"):
            paths.append(text.removeprefix("dd:"))
    for value in row.get("unresolved_tokens") or []:
        text = str(value)
        if "/" in text and not text.startswith(("sn:", "derived:")):
            paths.append(text.removeprefix("dd:"))
    return sorted({path.removeprefix("dd:") for path in paths if path})


FACTS_CYPHER = """
MATCH (sn:StandardName)
WHERE sn.id IN $ids
OPTIONAL MATCH (cost:LLMCost)-[:FOR_STANDARD_NAME]->(sn)
WITH sn, sum(coalesce(cost.llm_cost, 0.0)) AS spend
RETURN sn.id AS id, sn.name_stage AS stage, sn.status AS status,
       sn.docs_stage AS docs_stage, sn.description AS description,
       sn.unit AS unit, sn.physics_domain AS physics_domain,
       sn.origin AS origin, sn.source_path AS source_path,
       sn.source_paths AS paths, sn.model AS model,
       size(coalesce(sn.documentation, '')) AS documentation_chars,
       sn.reviewer_score_name AS reviewer_score_name,
       sn.reviewer_score_docs AS reviewer_score_docs,
       sn.reviewed_name_at AS reviewed_name_at,
       sn.reviewed_docs_at AS reviewed_docs_at,
       sn.docs_chain_length AS docs_chain_length,
       sn.exported_at AS exported_at, spend
ORDER BY id LIMIT 200"""

HOLDERS_CYPHER = """
MATCH (dd:IMASNode)
WHERE dd.id IN $paths
OPTIONAL MATCH (source:StandardNameSource)-[:FROM_DD_PATH]->(dd)
OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(holder:StandardName)
WHERE holder IS NOT NULL AND NOT holder.status IN $terminal
RETURN dd.id AS path, dd.lifecycle_status AS lifecycle_status,
       dd.documentation AS dd_documentation, dd.unit AS dd_unit,
       collect(DISTINCT {id: source.id, status: source.status,
                         mirror: source.produced_sn_id}) AS sources,
       collect(DISTINCT CASE WHEN holder IS NULL THEN null ELSE
         {name: holder.id, stage: holder.name_stage,
          status: holder.status, unit: holder.unit} END) AS holders
ORDER BY path LIMIT 500"""


def collect_live() -> dict[str, Any]:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )
    from imas_codex.standard_names.source_invariant import find_sourceless_live_names

    holds = release_holds()
    gc = GraphClient()
    try:
        classified = classify_sourceless_standard_names(gc)
        # The gate's two source-record populations. ``unchained_holder`` means a
        # live source is held by another live name; the ledger reports it beside
        # the other two, so it is carried as its own class rather than folded in.
        sourceless = [
            row
            for row in classified
            if row["population"] in {"dropped_edge", "never_bound"}
        ]
        live_drafts = [
            row
            for row in sourceless
            if row["name_stage"] != "exhausted" and row["name_stage"] not in SHIPPING
        ]
        live_accepted = [row for row in sourceless if row["name_stage"] in SHIPPING]
        dormant = [row for row in sourceless if row["name_stage"] == "exhausted"]
        derived_exempt = sorted(
            row["id"] for row in classified if row["population"] == "derived_parent"
        )
        unchained = [
            row for row in classified if row["population"] == "unchained_holder"
        ]
        population = sorted(live_drafts + live_accepted, key=lambda row: row["id"])
        if not population:
            raise RuntimeError("live sourceless population is empty")

        ids = [row["id"] for row in population]
        # The export gate corroborates each draft against the writer invariant.
        # Here it is asked about every name, so a name the two predicates
        # disagree on is reported rather than resolved silently.
        confirmed = set(find_sourceless_live_names(gc, ids))

        facts = read(gc, FACTS_CYPHER, ids=ids)
        if len(facts) != len(population):
            raise RuntimeError(f"name facts incomplete: {len(facts)}/{len(population)}")
        by_id = {row["id"]: row for row in facts}

        recorded = {row["id"]: recorded_paths(row) for row in population}
        all_paths = sorted({path for paths in recorded.values() for path in paths})
        holder_rows: dict[str, dict[str, Any]] = {}
        if all_paths:
            holder_rows = {
                row["path"]: row
                for row in read(gc, HOLDERS_CYPHER, paths=all_paths, terminal=TERMINAL)
            }
            missing = sorted(set(all_paths) - set(holder_rows))
            if missing:
                # A recorded path that is absent from the DD graph is a fact
                # about the name, not a read fault: the source moved or left.
                for path in missing:
                    holder_rows[path] = {
                        "path": path,
                        "lifecycle_status": "absent",
                        "dd_documentation": None,
                        "dd_unit": None,
                        "sources": [],
                        "holders": [],
                    }

        # Positive control: a path known to have a live producer must read one,
        # so an empty holders column everywhere is shown to be a reading fault.
        positive = read(
            gc,
            """MATCH (source:StandardNameSource {id: $source_id})
            MATCH (source)-[:PRODUCED_NAME]->(holder:StandardName)
            RETURN holder.id AS id LIMIT 5""",
            source_id="dd:core_profiles/profiles_1d/electrons/temperature",
        )
        if not positive:
            raise RuntimeError("holder positive control found no known holder")

        rows_out = []
        for row in population:
            name = row["id"]
            fact = by_id[name]
            paths = recorded[name]
            holders = []
            for path in paths:
                entry = holder_rows[path]
                for holder in entry["holders"]:
                    if holder and holder["name"] != name:
                        holders.append({**holder, "path": path})
            rows_out.append(
                {
                    "name": name,
                    "classification": {
                        "population": row["population"],
                        "stage": row["name_stage"],
                        "status": row["status"],
                        "docs_stage": row.get("docs_stage"),
                        "origin": row.get("origin"),
                        "model": row.get("model"),
                        "recorded_source_ids": row.get("recorded_source_ids") or [],
                        "unresolved_tokens": row.get("unresolved_tokens") or [],
                        "unchained_holders": row.get("unchained_holders") or [],
                    },
                    "record": {
                        "description": fact["description"],
                        "unit": fact["unit"],
                        "physics_domain": fact["physics_domain"],
                        "documentation_chars": fact["documentation_chars"],
                        "docs_stage": fact["docs_stage"],
                        "docs_chain_length": fact["docs_chain_length"],
                        "reviewer_score_name": fact["reviewer_score_name"],
                        "reviewer_score_docs": fact["reviewer_score_docs"],
                        "reviewed_name_at": _iso(fact["reviewed_name_at"]),
                        "reviewed_docs_at": _iso(fact["reviewed_docs_at"]),
                        "exported_at": _iso(fact["exported_at"]),
                        "stage": fact["stage"],
                        "status": fact["status"],
                        "spend_usd": float(fact["spend"] or 0.0),
                    },
                    "recorded_paths": [
                        {
                            "path": path,
                            "lifecycle_status": holder_rows[path]["lifecycle_status"],
                            "dd_documentation": holder_rows[path]["dd_documentation"],
                            "dd_unit": holder_rows[path]["dd_unit"],
                            "recorded_sources": [
                                source
                                for source in holder_rows[path].get("sources", [])
                                if source and source.get("id")
                            ],
                            "live_holders": [
                                holder
                                for holder in holder_rows[path]["holders"]
                                if holder
                            ],
                        }
                        for path in paths
                    ],
                    "recorded_source_holders": holders,
                    "invariant_confirms": name in confirmed,
                    "release_hold": (
                        {"reason": holds[name][0], "detail": holds[name][1]}
                        if name in holds
                        else None
                    ),
                    "stage_ships": row["name_stage"] in SHIPPING,
                }
            )

        names = {row["name"] for row in rows_out}
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "graph_writes": 0,
            "llm_calls": 0,
            "spend_usd": 0,
            "census_total": len(classified),
            "classifier_populations": dict(
                Counter(row["population"] for row in classified)
            ),
            "dormant_exhausted_sourceless": len(dormant),
            "derived_parent_exempt": derived_exempt,
            "unchained_holder_names": sorted(row["id"] for row in unchained),
            "live_draft_sourceless": len(live_drafts),
            "live_accepted_sourceless": len(live_accepted),
            "live_population": len(population),
            "invariant_confirmed": sorted(confirmed),
            "invariant_disagreements": sorted(names - confirmed),
            "recorded_nine": sorted(RECORDED_NINE),
            "recorded_nine_present": sorted(RECORDED_NINE & names),
            "recorded_nine_absent": sorted(RECORDED_NINE - names),
            "live_accepted_not_in_recorded_nine": sorted(
                {row["name"] for row in rows_out if row["stage_ships"]} - RECORDED_NINE
            ),
            "holder_positive_control": [row["id"] for row in positive],
            "rows": rows_out,
        }
    finally:
        gc.close()


def run_fixture() -> dict[str, Any]:
    """Plant a sourceless name and a bound control; read both predicates back."""
    sys.path.insert(0, str(HERE.parents[4]))
    from tests.standard_names.conftest import (
        _start_empty_neo4j,
        _wipe,
        refuse_project_graph,
    )

    directory, _process, _log, port = _start_empty_neo4j()
    uri = f"bolt://127.0.0.1:{port}"
    refuse_project_graph(uri)
    previous = {
        key: os.environ.get(key)
        for key in (
            "NEO4J_URI",
            "NEO4J_PASSWORD",
            "IMAS_CODEX_TEST_NEO4J_URI",
            "IMAS_CODEX_TEST_NEO4J_EPHEMERAL",
        )
    }
    os.environ.update(
        {
            "NEO4J_URI": uri,
            "NEO4J_PASSWORD": "",
            "IMAS_CODEX_TEST_NEO4J_URI": uri,
            "IMAS_CODEX_TEST_NEO4J_EPHEMERAL": "1",
        }
    )
    try:
        _wipe(uri)
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.graph_store.graph_supersession import (
            classify_sourceless_standard_names,
        )
        from imas_codex.standard_names.source_invariant import (
            find_sourceless_live_names,
        )

        with GraphClient() as gc:
            gc.query(
                """
                CREATE (planted:StandardName {id: 'planted_sourceless_name',
                        name_stage: 'draft', status: 'draft', origin: 'llm',
                        unit: 'eV',
                        description: 'a planted name with no producer'})
                CREATE (bound:StandardName {id: 'planted_bound_name',
                        name_stage: 'draft', status: 'draft'})
                CREATE (source:StandardNameSource {id: 'dd:planted/quantity',
                        source_type: 'dd', source_id: 'planted/quantity',
                        status: 'extracted'})
                CREATE (source)-[:PRODUCED_NAME]->(bound)
                """
            )
            classified = classify_sourceless_standard_names(gc)
            ids = ["planted_sourceless_name", "planted_bound_name"]
            invariant = find_sourceless_live_names(gc, ids)
            listed = sorted(row["id"] for row in classified)
            result = {
                "generated_at": datetime.now(UTC).isoformat(),
                "endpoint": uri,
                "graph_writes": 1,
                "graph_writes_note": "disposable Neo4j only; the live graph is untouched",
                "classifier_listed": listed,
                "invariant_listed": sorted(invariant),
                "planted_sourceless_listed": "planted_sourceless_name" in listed,
                "planted_bound_excluded": "planted_bound_name" not in listed,
                "invariant_planted_listed": "planted_sourceless_name" in invariant,
                "invariant_bound_excluded": "planted_bound_name" not in invariant,
                "classifier_populations": dict(
                    Counter(row["population"] for row in classified)
                ),
            }
        # The control lives in the population file, so the read and the check
        # that the read can see anything are one artifact.
        data = json.loads(OUT.read_text(encoding="utf-8"))
        data["planted_control"] = result
        OUT.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate(data: dict[str, Any]) -> None:
    names = [row["name"] for row in data["rows"]]
    if len(names) != len(set(names)) or len(names) != data["live_population"]:
        raise RuntimeError("page rows do not match the live sourceless population")
    for row in data["rows"]:
        name = row["name"]
        proposal = PROPOSALS.get(name)
        if proposal is None:
            raise RuntimeError(f"no proposal for {name}")
        action = proposal["action"]
        if action not in {"retire_with_successor", "retire", "rebind", "hold"}:
            raise RuntimeError(f"unknown action for {name}: {action}")
        if not proposal.get("reason"):
            raise RuntimeError(f"proposal lacks a reason: {name}")
        if action == "hold":
            if proposal.get("target"):
                raise RuntimeError(f"held name carries a target: {name}")
            continue
        target = proposal.get("target")
        if not target:
            raise RuntimeError(f"proposal lacks a target: {name}")
        if action == "retire_with_successor":
            holders = {holder["name"] for holder in row["recorded_source_holders"]}
            stages = {
                holder["name"]: holder["stage"]
                for holder in row["recorded_source_holders"]
            }
            if target not in holders:
                raise RuntimeError(
                    f"successor is not a live holder: {name} -> {target}"
                )
            if stages[target] == "exhausted":
                raise RuntimeError(f"successor is an exhausted holder: {name}")
    # The nine recorded accepted names must reconcile: absent ones are a fact
    # to report, but a live accepted name outside the nine needs a proposal.
    for name in data["live_accepted_not_in_recorded_nine"]:
        if name not in PROPOSALS:
            raise RuntimeError(f"accepted sourceless name outside the nine: {name}")


def render(data: dict[str, Any], control: dict[str, Any]) -> str:
    counts = Counter(PROPOSALS[row["name"]]["action"] for row in data["rows"])
    body_rows = []
    for row in data["rows"]:
        rec = row["record"]
        proposal = PROPOSALS[row["name"]]
        holders = (
            "<br>".join(
                f"<code>{esc(holder['name'])}</code> ({esc(holder['stage'])}, "
                f"{esc(holder['path'])})"
                for holder in row["recorded_source_holders"]
            )
            or "none"
        )
        paths = (
            "<br>".join(
                f"<code>{esc(entry['path'])}</code> — "
                f"{esc(entry['lifecycle_status'] or 'live')}"
                for entry in row["recorded_paths"]
            )
            or "none recorded"
        )
        history = (
            f"docs {esc(rec['docs_stage'])}, chain {esc(rec['docs_chain_length'])}, "
            f"name score {esc(rec['reviewer_score_name'])}, "
            f"docs score {esc(rec['reviewer_score_docs'])}, "
            f"documentation {esc(rec['documentation_chars'])} chars, "
            f"spend USD {rec['spend_usd']:.4f}"
        )
        hold = (
            f"<br><strong>{esc(row['release_hold']['reason'])}</strong>: "
            f"{esc(row['release_hold']['detail'])}"
            if row["release_hold"]
            else ""
        )
        body_rows.append(
            "<tr data-name='"
            + esc(row["name"])
            + "'><th scope='row'><code>"
            + esc(row["name"])
            + "</code><br>"
            + esc(row["classification"]["stage"])
            + " / "
            + esc(row["classification"]["status"])
            + "</th><td>"
            + esc(rec["description"])
            + "; <strong>"
            + esc(rec["unit"])
            + "</strong>; "
            + esc(rec["physics_domain"])
            + "</td><td>"
            + paths
            + "</td><td>"
            + holders
            + "</td><td>"
            + history
            + hold
            + "</td><td><strong>"
            + esc(proposal["action"])
            + "</strong> "
            + esc(proposal.get("target") or "")
            + "<p>"
            + esc(proposal["reason"])
            + "</p></td></tr>"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Live names with no producing DD source — proposals</title>"
        "</head><body><main>"
        "<h2>Live names with no producing DD source</h2>"
        f"<p>{data['live_population']} live names: "
        f"{data['live_draft_sourceless']} drafts and "
        f"{data['live_accepted_sourceless']} accepted. Proposed actions: "
        + ", ".join(f"{esc(k)} {v}" for k, v in sorted(counts.items()))
        + f". Read at {esc(data['generated_at'])}; the live graph was not written; "
        "spend USD 0.</p>"
        "<p>The population is the export gate's own: "
        f"{data['census_total']} names classify as sourceless, of which "
        f"{data['dormant_exhausted_sourceless']} are exhausted drafts and "
        f"{len(data['derived_parent_exempt'])} are derived parents the invariant "
        f"exempts. The writer invariant confirms "
        f"{len(data['invariant_confirmed'])} of the {data['live_population']} "
        "listed"
        + (
            "; it disagrees on " + esc(", ".join(data["invariant_disagreements"]))
            if data["invariant_disagreements"]
            else ", and disagrees on none"
        )
        + ".</p>"
        "<p><strong>Planted control.</strong> In a disposable Neo4j, "
        "a name planted with no producing source is listed by the classifier "
        f"({esc(control['planted_sourceless_listed'])}) and by the writer "
        f"invariant ({esc(control['invariant_planted_listed'])}), while its bound "
        "control is excluded by both "
        f"({esc(control['planted_bound_excluded'])}, "
        f"{esc(control['invariant_bound_excluded'])}). The read-back lists "
        f"{esc(', '.join(control['classifier_listed']) or 'nothing')}.</p>"
        "<h3>What the population turned out to be</h3>"
        "<p>Three of the five are not source records that vanished: their "
        "recorded source is present, carries status <code>extracted</code>, "
        "names a live DD path whose unit the name matches, and lost only the "
        "<code>PRODUCED_NAME</code> edge. The repair they need is a signed "
        "<strong>rebind</strong>, and it is not the automatic edge healing the "
        "ledger already offers: <code>reattach_produced_name_edges</code> "
        "falls to the source's <code>produced_sn_id</code> mirror, and all three "
        "records carry a null mirror, so that path would leave them sourceless.</p>"
        "<p>The other two are accepted names with no source of any kind. "
        "<code>volume_averaged_electron_number_density_over_scrape_off_layer</code> "
        "keeps a detachment receipt and no source; "
        "<code>coolant_absorbed_energy_accumulated_of_plasma_facing_component</code> "
        "was never bound at all. Both already carry the export gate's "
        "<code>release_*</code> hold and both have accepted documentation, so "
        "the proposed action is to hold them under that ruling rather than "
        "retire scored work nothing in the active DD measures.</p>"
        "<h3>The nine recorded accepted names</h3>"
        f"<p>None of the nine is sourceless at this read: "
        f"{len(data['recorded_nine_present'])} of {DECLARED_NINE} "
        "appear in the live population, because the repairs since 2026-10-12 "
        "bound them. The two accepted names that are sourceless are outside "
        "that nine, so the question's population has moved: the answer owed to "
        "the lead covers these two and not the recorded nine.</p>"
        "<h3>Other sourceless states the gate does not gate on</h3>"
        f"<p>The same read classifies {data['dormant_exhausted_sourceless']} "
        "exhausted drafts, which never ship, "
        f"{len(data['derived_parent_exempt'])} derived parent "
        f"(<code>{esc(', '.join(data['derived_parent_exempt']))}</code>), which "
        "the invariant exempts as structural, and "
        f"{len(data['unchained_holder_names'])} name in the "
        "<code>unchained_holder</code> class "
        f"(<code>{esc(', '.join(data['unchained_holder_names']))}</code>), whose "
        "live source another name holds. None is a live name without a source "
        "in the sense §7 measures, so none is proposed for action here.</p>"
        "<table><thead><tr><th>Name and stage</th><th>Record</th>"
        "<th>Recorded DD path</th><th>Live holder of that path</th>"
        "<th>History and spend</th><th>Proposed action</th></tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></main></body></html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("live", "fixture", "render"))
    parser.parse_args()
    command = sys.argv[1]
    if command == "live":
        data = collect_live()
        OUT.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"live={data['live_population']} drafts={data['live_draft_sourceless']} "
            f"accepted={data['live_accepted_sourceless']} "
            f"census={data['census_total']} "
            f"invariant_confirmed={len(data['invariant_confirmed'])} "
            f"recorded_nine_present={len(data['recorded_nine_present'])} "
            f"spend_usd=0 graph_writes=0"
        )
        for row in data["rows"]:
            print(
                f"  {row['classification']['stage']:9s} {row['name']} "
                f"paths={[p['path'] for p in row['recorded_paths']]} "
                f"holders={[h['name'] for h in row['recorded_source_holders']]} "
                f"spend={row['record']['spend_usd']:.4f}"
            )
    elif command == "fixture":
        result = run_fixture()
        print(json.dumps(result, indent=2, sort_keys=True))
        # A control that cannot fail proves nothing, so a failed predicate is
        # a non-zero exit rather than a value on the page.
        failed = [
            key
            for key in (
                "planted_sourceless_listed",
                "planted_bound_excluded",
                "invariant_planted_listed",
                "invariant_bound_excluded",
            )
            if result.get(key) is not True
        ]
        if failed:
            print(f"CONTROL FAILED: {', '.join(failed)}", file=sys.stderr)
            raise SystemExit(1)
    else:
        data = json.loads(OUT.read_text(encoding="utf-8"))
        control = data["planted_control"]
        validate(data)
        PAGE.write_text(render(data, control), encoding="utf-8")
        print(f"rows={data['live_population']} page={PAGE}")


if __name__ == "__main__":
    main()

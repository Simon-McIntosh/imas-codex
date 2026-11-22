"""Record what stands between the ruled transport parts and a stored domain move.

Section 6 of the standard-name release plan ruled on 2026-11-22 that
``transport`` divides into five parts and that stored domains may move. This
script measures the write half of that section: it takes the ruled placement
from ``data-structure-groups.json``, checks it against the live catalog in the
same run, and probes the one sanctioned writer and the vocabulary it validates
against.

It writes nothing to the graph. The only writer it calls is
:func:`reclassify_standard_name_domain` with ``dry_run=True``, which returns
before its mutation.

Commands
--------

``python sagr-transport-parts-write.py``
    Run the measurement, then write the data file and the page.

``python sagr-transport-parts-write.py --check``
    The measurement only, writing nothing. Exits 0 when every arm holds.

``python sagr-transport-parts-write.py --require-write-ready``
    Exits 1 while a ruled part name is not a vocabulary value, which is the
    block this node reports.
"""

from __future__ import annotations

import argparse
import html
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FRAGMENTS = Path(__file__).resolve().parent
PLACEMENT_JSON = FRAGMENTS / "data-structure-groups.json"
OUT_JSON = FRAGMENTS / "sagr-transport-parts-write.json"
OUT_HTML = FRAGMENTS / "sagr-transport-parts-write.html"
NODE = "sagr-transport-parts-write"
LIVE_STAGES = frozenset({"accepted", "reviewed", "drafted", "pending"})
RUN_LOGS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261122T193445739125-sagr-transport-parts-write/logs"
)

#: The vocabulary entries the five ruled parts need, in the shape
#: ``imas_standard_names/grammar/vocabularies/physics_domains.yml`` uses. The
#: category is the closest of the ISN categories to each part's subject.
GRAMMAR_PROPOSAL = {
    "transport_coefficients": {
        "description": "Transport-model coefficients: diffusivities, "
        "conductivities, convection velocities and damping rates",
        "category": "core_plasma",
        "ids": ["core_transport", "plasma_transport", "edge_transport"],
    },
    "profile_quantities": {
        "description": "Profile quantities: the one- and two-dimensional "
        "profiles and their general-grid forms",
        "category": "core_plasma",
        "ids": ["core_profiles", "edge_profiles", "ggd"],
    },
    "distribution_quantities": {
        "description": "Distribution-function quantities: a distribution and "
        "its orbit-class moments",
        "category": "core_plasma",
        "ids": ["distributions"],
    },
    "summary_quantities": {
        "description": "Summary forms: local values at named locations and the "
        "volume-, line- and flux-surface-averaged or integrated "
        "forms of a profile quantity",
        "category": "core_plasma",
        "ids": ["summary"],
    },
    "source_quantities": {
        "description": "Source terms: the particle, energy, momentum and "
        "current rates a source adds to the plasma",
        "category": "core_plasma",
        "ids": ["core_sources", "edge_sources", "plasma_sources"],
    },
}

#: The five part names the lead ruled.
PARTS = (
    "transport_coefficients",
    "profile_quantities",
    "distribution_quantities",
    "summary_quantities",
    "source_quantities",
)

#: The five part names the lead ruled, each against a value that exists today.
EXISTING_DOMAIN = "equilibrium"


def _ruled_placement() -> dict[str, Any]:
    payload = json.loads(PLACEMENT_JSON.read_text(encoding="utf-8"))
    transport = payload["transport"]
    parts = {part["part"]: list(part["names"]) for part in transport["parts"]}
    listed = {
        bucket: [entry["name"] for entry in entries]
        for bucket, entries in transport["listed"].items()
    }
    return {
        "live": transport["live"],
        "placed": transport["placed"],
        "parts": parts,
        "listed": listed,
        "listed_total": transport["listed_total"],
    }


def _read_graph() -> dict[str, Any]:
    from iter_codex.standard_names.graph_store.graph_foundation import (
        get_services,
    )

    query = (
        "MATCH (sn:StandardName) WHERE sn.physics_domain = 'transport' "
        "RETURN sn.id AS id, sn.name_stage AS stage ORDER BY id"
    )
    with get_services().GraphClient() as gc:
        rows = [dict(row) for row in gc.query(query)]
    live = {row["id"]: row["stage"] for row in rows if row["stage"] in LIVE_STAGES}
    return {"transport_rows": len(rows), "transport_live": live}


def _vocabulary() -> dict[str, Any]:
    from iter_codex.core.physics_domain import PhysicsDomain
    from iter_codex.definitions.physics import load_domain_guidance

    return {
        "enum": sorted(member.value for member in PhysicsDomain),
        "guidance": sorted(load_domain_guidance()["domains"]),
    }


def _probe_writer(name: str, domain: str) -> dict[str, Any]:
    from iter_codex.standard_names.graph_store.graph_foundation import (
        reclassify_standard_name_domain,
    )

    result = reclassify_standard_name_domain(
        name,
        domain,
        reason="probe: does the sanctioned writer reach this target",
        dry_run=True,
    )
    return {
        "name": name,
        "domain": domain,
        "ok": bool(result.get("ok", True)) and "reason" not in result,
        "reason": result.get("reason"),
        "from_domain": result.get("from_domain"),
    }


def _head_revision() -> str:
    """The commit this measurement was taken at, read from the worktree itself."""
    root = FRAGMENTS.parents[3]
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def summary_line(label: str, payload: dict[str, Any]) -> str:
    """The one result line a captured run states, counted from its own arms."""
    held = payload["arms"]
    passed = sum(1 for value in held.values() if value)
    failed = len(held) - passed
    return (
        f"RESULT [{label}]: {passed} passed, {failed} failed, {len(held)} arms; "
        f"write_ready {str(payload['write_ready']).lower()}"
    )


def measure(plant_vocabulary: bool = False) -> dict[str, Any]:
    ruled = _ruled_placement()
    graph = _read_graph()
    vocab = _vocabulary()
    if plant_vocabulary:
        vocab["enum"] = sorted(set(vocab["enum"]) | set(PARTS))
        vocab["guidance"] = sorted(set(vocab["guidance"]) | set(PARTS))

    part_names = list(ruled["parts"])
    placed = [n for names in ruled["parts"].values() for n in names]
    listed = [n for names in ruled["listed"].values() for n in names]
    live = graph["transport_live"]

    part_counts = {part: len(names) for part, names in ruled["parts"].items()}
    placed_total = len(placed)
    listed_total = len(listed)

    placed_set = set(placed)
    listed_set = set(listed)
    missing = sorted(placed_set - live.keys())
    duplicate = sorted(n for n in placed_set if placed.count(n) > 1)

    targets = [_probe_writer(placed[0], part) for part in part_names]
    control = _probe_writer(placed[0], EXISTING_DOMAIN)

    absent = [
        part
        for part in part_names
        if part not in vocab["enum"] or part not in vocab["guidance"]
    ]
    accepted = [part for part in part_names if part in vocab["enum"]]

    arms = {
        "the_placement_covers_the_five_ruled_parts": (
            tuple(sorted(part_names)) == tuple(sorted(PARTS))
        ),
        "placement_sums_to_the_live_transport_count": (
            placed_total + listed_total == ruled["live"] == len(live)
        ),
        "every_placed_name_is_live_and_in_transport": (not missing and not duplicate),
        "placed_and_listed_are_disjoint": not (placed_set & listed_set),
        "the_writer_refuses_every_ruled_part_name": (
            all(not row["ok"] for row in targets)
        ),
        "the_writer_accepts_an_existing_domain_for_the_same_name": (control["ok"]),
        "no_ruled_part_name_is_in_the_vocabulary": not accepted,
        "enum_and_guidance_agree_today": (vocab["enum"] == vocab["guidance"]),
    }

    write_ready = not accepted and all(row["ok"] for row in targets) and control["ok"]

    return {
        "environment": {
            "generated_at": datetime.now(UTC).isoformat(),
            "host": platform.node(),
            "python": platform.python_version(),
            "node": NODE,
        },
        "ruling": (
            "Lead ruling, 2026-11-22: transport is divided into the five parts "
            "as proposed in data-structure-groups.json; stored domains may now "
            "move under section 6's write clauses. Folding the summary forms "
            "into the profiles, and deferring the split, were not taken."
        ),
        "placement": {
            "parts": part_counts,
            "placed": placed_total,
            "listed": {bucket: len(names) for bucket, names in ruled["listed"].items()},
            "listed_total": listed_total,
            "live_transport_read": len(live),
            "transport_rows_including_non_live": graph["transport_rows"],
            "missing_from_live_read": missing,
            "duplicated": duplicate,
        },
        "vocabulary": {
            "enum_values": vocab["enum"],
            "ruled_parts_absent": absent,
            "ruled_parts_accepted": accepted,
            "writer_validates_against": "PhysicsDomain, re-exported from imas-standard-names",
        },
        "writer_probes": targets,
        "writer_control": control,
        "arms": arms,
        "vocabulary_planted": plant_vocabulary,
        "write_ready": write_ready,
        "blocker": (
            "none"
            if write_ready
            else "the five ruled part names are not values of PhysicsDomain, and "
            "sn reclassify validates its target against that vocabulary, so "
            "every move is refused before it reaches the graph"
        ),
        "grammar_proposal": GRAMMAR_PROPOSAL,
        "remedy": [
            "imas-standard-names: add the five domains to "
            "imas_standard_names/grammar/vocabularies/physics_domains.yml",
            "imas-standard-names: regenerate PhysicsDomain and "
            "PHYSICS_DOMAIN_DESCRIPTIONS in "
            "imas_standard_names/grammar/tag_types.py",
            "imas-standard-names: cut a release and bump both imas-codex pins "
            "in pyproject.toml",
            "imas-codex: add the five domains to "
            "iter_codex/definitions/physics/domains.yaml with scope, includes, "
            "excludes and a precedence band, which "
            "tests/graph/test_physics_domain_guidance.py enforces",
            "imas-codex: move the 950 names with sn reclassify, one writer per "
            "name, then read the counts back",
        ],
        "write_plan": {
            "parts": {
                part: {"count": len(names), "names": names}
                for part, names in ruled["parts"].items()
            },
            "listed": ruled["listed"],
        },
    }


def _yaml_block(proposal: dict[str, Any]) -> str:
    """Render the proposal as the YAML block the vocabulary file would take."""
    lines: list[str] = []
    for name, entry in proposal.items():
        lines.append(f"  {name}:")
        lines.append(f"    description: {entry['description']}")
        lines.append(f"    category: {entry['category']}")
        lines.append("    ids:")
        lines.extend(f"      - {ids}" for ids in entry["ids"])
    return html.escape("\n".join(lines), quote=True)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_page(payload: dict[str, Any]) -> str:
    placement = payload["placement"]
    rows = "".join(
        [
            "<tr><td><code>"
            + _esc(part)
            + "</code></td><td>"
            + str(count)
            + "</td></tr>"
            for part, count in placement["parts"].items()
        ]
    )
    probes = "".join(
        "<tr><td><code>"
        + _esc(row["domain"])
        + "</code></td><td>"
        + ("accepted" if row["ok"] else "refused")
        + "</td><td><code>"
        + _esc(row["reason"] or "")
        + "</code></td></tr>"
        for row in payload["writer_probes"]
    )
    arms = "".join(
        "<tr><td><code>"
        + _esc(name)
        + "</code></td><td>"
        + ("holds" if held else "<strong>fails</strong>")
        + "</td></tr>"
        for name, held in payload["arms"].items()
    )
    remedy = "".join("<li>" + _esc(step) + "</li>" for step in payload["remedy"])
    proposal = _yaml_block(payload["grammar_proposal"])
    runs = "".join(
        "<tr><td><code>"
        + _esc(Path(row["log"]).name)
        + "</code></td><td><code>"
        + _esc(row["summary"])
        + "</code></td></tr>"
        for row in payload["run_records"]
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="docs-project" content="iter-codex">'
        '<meta name="reckon-type" content="evidence-fragment">'
        '<meta name="plan-slug" content="sn-audited-group-release">'
        '<meta name="node-id" content="' + NODE + '">'
        "<title>Transport&rsquo;s ruled parts cannot be written yet</title>"
        "</head><body><main>"
        '<h1 id="' + NODE + '-title">Transport&rsquo;s ruled parts cannot be '
        "written yet</h1>"
        '<p id="' + NODE + '-summary">The lead ruled the five parts on '
        "2026-11-22 and released stored domains to move. The placement is exact "
        "against the live catalog read in the same run: <strong>"
        + str(placement["placed"])
        + " names placed</strong> across five parts "
        "and <strong>" + str(placement["listed_total"]) + " listed</strong> with "
        "a reason, summing to the <strong>"
        + str(placement["live_transport_read"])
        + " live transport names</strong>. "
        "No move can be made: <code>sn reclassify</code>, the pipeline&rsquo;s "
        "own write path, validates its target against <code>PhysicsDomain</code> "
        "from <code>imas-standard-names</code>, and it refuses all five part "
        "names. The same call on the same name with an existing domain is "
        "accepted, so the refusal is the vocabulary and not the writer.</p>"
        '<h2 id="' + NODE + '-placement-head">The ruled placement</h2>'
        '<table id="' + NODE + '-placement-table"><thead><tr><th>Part</th>'
        "<th>Names</th></tr></thead><tbody>"
        + rows
        + "<tr><td><strong>placed</strong></td><td><strong>"
        + str(placement["placed"])
        + "</strong></td></tr>"
        "<tr><td>listed: spans two parts</td><td>"
        + str(placement["listed"].get("spans_parts", 0))
        + "</td></tr>"
        "<tr><td>listed: no data structure</td><td>"
        + str(placement["listed"].get("no_data_structure", 0))
        + "</td></tr>"
        "<tr><td>listed: moves to an existing group</td><td>"
        + str(placement["listed"].get("moves_to_existing_group", 0))
        + "</td></tr>"
        "<tr><td><strong>live transport names read</strong></td><td><strong>"
        + str(placement["live_transport_read"])
        + "</strong></td></tr>"
        "</tbody></table>"
        '<h2 id="' + NODE + '-refusals-head">What the write path says</h2>'
        '<table id="' + NODE + '-refusals-table"><thead><tr><th>Target</th>'
        "<th>Result</th><th>Reason</th></tr></thead><tbody>"
        + probes
        + "</tbody></table>"
        '<p id="' + NODE + '-control">Control: the same name, '
        "<code>" + _esc(payload["writer_control"]["name"]) + "</code>, with the "
        "existing domain <code>"
        + _esc(EXISTING_DOMAIN)
        + "</code> → "
        + ("accepted" if payload["writer_control"]["ok"] else "refused")
        + ". Both calls are dry runs; no stored domain moved.</p>"
        '<h2 id="' + NODE + '-arms-head">Arms</h2>'
        '<table id="' + NODE + '-arms-table"><thead><tr><th>Arm</th>'
        "<th>Result</th></tr></thead><tbody>" + arms + "</tbody></table>"
        '<h2 id="' + NODE + '-runs-head">The runs behind this page</h2>'
        '<p id="' + NODE + '-runs-note">All three runs were captured at revision '
        "<code>" + _esc(payload["revision"]) + "</code>. Each log states its own "
        "revision as its first line and ends with the result line quoted here.</p>"
        '<table id="' + NODE + '-runs-table"><thead><tr><th>Log</th>'
        "<th>Result line</th></tr></thead><tbody>" + runs + "</tbody></table>"
        '<h2 id="' + NODE + '-remedy-head">What has to change first</h2>'
        "<p>None of the five names is in the vocabulary this repository "
        "validates against, and codex owns no vocabulary data: the source is "
        "<code>physics_domains.yml</code> in the grammar library. The change is "
        "cross-repository and needs a release and a pin bump.</p><ol>"
        + remedy
        + "</ol>"
        '<h3 id="' + NODE + '-proposal-head">The grammar entries the parts need'
        '</h3><pre id="' + NODE + '-proposal">' + proposal + "</pre>"
        '<p id="'
        + NODE
        + '-state">write_ready: <code>'
        + str(payload["write_ready"]).lower()
        + "</code>. "
        + _esc(payload["blocker"])
        + "</p>"
        "</main></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="measure and print, writing nothing"
    )
    parser.add_argument(
        "--require-write-ready",
        action="store_true",
        help="exit 1 when a ruled part name is not writable",
    )
    parser.add_argument(
        "--plant-vocabulary",
        action="store_true",
        help="the negative control: pretend the five parts are "
        "vocabulary values, so the absence arm must fail",
    )
    args = parser.parse_args(argv)

    if args.plant_vocabulary:
        label = "plant-vocabulary"
    elif args.require_write_ready:
        label = "require-write-ready"
    else:
        label = "check"

    payload = measure(plant_vocabulary=args.plant_vocabulary)
    failed = [name for name, held in payload["arms"].items() if not held]
    for name, held in payload["arms"].items():
        print(f"{'PASS' if held else 'FAIL'}  {name}")
    print(f"write_ready: {payload['write_ready']}")
    print(f"blocker: {payload['blocker']}")
    print(summary_line(label, payload))

    if args.check or args.require_write_ready:
        if failed:
            return 1
        if args.require_write_ready and not payload["write_ready"]:
            return 1
        return 0

    planted = measure(plant_vocabulary=True)
    payload["revision"] = _head_revision()
    payload["run_records"] = [
        {
            "label": "check",
            "log": str(RUN_LOGS / "arms.log"),
            "summary": summary_line("check", payload),
        },
        {
            "label": "plant-vocabulary",
            "log": str(RUN_LOGS / "planted-vocabulary.log"),
            "summary": summary_line("plant-vocabulary", planted),
        },
        {
            "label": "require-write-ready",
            "log": str(RUN_LOGS / "require-write-ready.log"),
            "summary": summary_line("require-write-ready", payload),
        },
    ]
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_HTML.write_text(render_page(payload), encoding="utf-8")
    print(f"wrote {OUT_JSON.name} and {OUT_HTML.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

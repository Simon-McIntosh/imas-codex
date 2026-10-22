#!/usr/bin/env python
"""Record bringing two accepted outboard-midplane names in line with one definition.

The catalog now records a single outboard-midplane definition and uses it
everywhere: the point on the low-field side where the flux surface reaches its
maximum major radius (imas-standard-names ``9130410``, read here through the
editable sibling checkout). Two accepted names needed their documentation
brought in line, and this record drives each through the pipeline's own routes,
then reads both names' lifecycle, scores, documentation and reviewer comments
from the live graph.

* ``radial_separatrix_distance_at_outboard_midplane`` — its stored text predates
  the merge. The docs writer appends the locus registry's defining sentence to
  generated documentation, and the sentence it had appended named a midplane
  height the data's midplane identifier selects. A steered documentation edit
  regenerated the text against the merged registry, so the appended sentence is
  the single-midplane one.
* ``radial_coordinate_at_outboard_midplane`` — its source is
  ``equilibrium/time_slice/profiles_1d/r_outboard``, taken at the magnetic-axis
  height, which coincides with the flux surface's maximum-major-radius point
  only for an up-down-symmetric equilibrium. A steered documentation edit asked
  the text to state that physics.

Nothing here writes to the graph. Modes:
    locus      render locus_definition_sentence() for outboard_midplane
    snapshot   read both names' lifecycle, scores and documentation
    comments   read the docs-axis reviewer comments for both names
    report     assemble the fragment's JSON record beside this runner
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "shc-midplane-docs-refresh.json")

NAMES = [
    "radial_separatrix_distance_at_outboard_midplane",
    "radial_coordinate_at_outboard_midplane",
]

_RUNS = (
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261022T175735626669-shc-midplane-docs-refresh"
)
_LOGS = os.path.join(_RUNS, "logs")

#: One generation and one funded review per name. Each edit is staged without an
#: inline review so the whole review runs once inside a single cost-capped
#: scope, as the split gap names' regeneration did.
INVOCATIONS = {
    "radial_separatrix_distance": {
        "dry_run": (
            "uv run --no-sync imas-codex sn edit "
            "radial_separatrix_distance_at_outboard_midplane "
            "--hint '<keep the body; the appended locus sentence must be the "
            "merged single-midplane definition (max major radius)>' "
            "--axis docs --scope self "
            "--reason '<lead single-midplane ruling 2026-10-22>' --dry-run"
        ),
        "edit_stage": (
            "uv run --no-sync imas-codex sn edit "
            "radial_separatrix_distance_at_outboard_midplane "
            "--hint '<keep the existing body unchanged: the opening sentence, "
            "the defining equation for the mapped separatrix separation, its "
            "symbol definitions, the flux-surface-mapping statement, the sign "
            "convention, and the cross-reference to the radial plasma-boundary "
            "outline. The locus sentence appended for the outboard midplane "
            "must be the merged single-midplane definition: the point on the "
            "low-field side where the flux surface reaches its maximum major "
            "radius. Do not describe a midplane height selected by the data, "
            "and name no DD field, IDS path or option value.>' "
            "--axis docs --scope self "
            "--reason '<the lead's single-midplane ruling of 2026-10-22: the "
            "catalog records one outboard-midplane definition and uses it "
            "everywhere. This name's stored documentation still carries the "
            "superseded per-dataset midplane-identifier wording, because the "
            "docs writer appends the superseded locus sentence; regenerating it "
            "against the merged registry replaces that sentence while changing "
            "nothing else this name says.>' --stage-only"
        ),
        "edit_run_id": "sn-edit-20261022T180859Z",
        "docs_review": (
            "uv run --no-sync imas-codex sn run --docs-only "
            "--scope-run-id sn-edit-20261022T180859Z --cost-limit 0.60 "
            "--skip-global-maintenance"
        ),
        "stage_log": os.path.join(_LOGS, "radial-sep-edit-stage.log"),
        "log": os.path.join(_LOGS, "radial-sep-docs-review.log"),
    },
    "radial_coordinate": {
        "dry_run": (
            "uv run --no-sync imas-codex sn edit "
            "radial_coordinate_at_outboard_midplane "
            "--hint '<keep the body; add, in physics terms, that the "
            "magnetic-axis height coincides with the surface maximum-R point "
            "only for up-down-symmetric equilibria>' "
            "--axis docs --scope self "
            "--reason '<lead single-midplane ruling 2026-10-22>' --dry-run"
        ),
        "edit_stage": (
            "uv run --no-sync imas-codex sn edit "
            "radial_coordinate_at_outboard_midplane "
            "--hint '<keep the existing body: the opening definition, the "
            "cylindrical R = sqrt(x^2+y^2) relation and the right-handed frame "
            "statement, the flux-surface label and the exclusion of the inboard "
            "intersection, and both cross-references. Add, in physics terms, "
            "that the source value is taken at the magnetic-axis height, and "
            "that this height coincides with the point of maximum major radius "
            "on the flux surface - the outboard midplane - only when the "
            "equilibrium is up-down symmetric; for a non-up-down-symmetric "
            "equilibrium the maximum-major-radius point lies away from the "
            "magnetic-axis height, so the value at that height is not the "
            "surface maximum-R point where it does not coincide. Name no DD "
            "field, IDS path or option value.>' "
            "--axis docs --scope self "
            "--reason '<the lead's single-midplane ruling of 2026-10-22: the "
            "catalog records one outboard-midplane definition and uses it "
            "everywhere. This name's source is "
            "equilibrium/time_slice/profiles_1d/r_outboard, evaluated at the "
            "magnetic-axis height; that height equals the maximum-major-radius "
            "point of the flux surface only for an up-down-symmetric "
            "equilibrium, so the documentation must state the symmetry "
            "condition under which the value matches the outboard midplane.>' "
            "--stage-only"
        ),
        "edit_run_id": "sn-edit-20261022T180925Z",
        "docs_review": (
            "uv run --no-sync imas-codex sn run --docs-only "
            "--scope-run-id sn-edit-20261022T180925Z --cost-limit 0.46 "
            "--time 8 --skip-global-maintenance"
        ),
        "stage_log": os.path.join(_LOGS, "radial-coord-edit-stage.log"),
        "log": os.path.join(_LOGS, "radial-coord-docs-review2.log"),
        "first_run_log": os.path.join(_LOGS, "radial-coord-docs-review.log"),
        "first_run_note": (
            "the first docs-only run for this name spent 0.540045 of its 0.60 "
            "limit and then could not reserve the tie-breaker seat ($0.0634 of "
            "pre-launch exposure against a pool of about $0.0600), so the "
            "quorum was incomplete and no score was persisted; the run's own "
            "budget bound the verdict, and the intervention run above carries "
            "the headroom"
        ),
    },
}

#: The merged locus registry entry, checked by rendering it rather than assumed.
LOCUS_DEFINITION_CHECK = {
    "concept": "outboard_midplane",
    "library": "imas_standard_names (editable sibling checkout)",
    "registry_file": "imas_standard_names/grammar/vocabularies/locus_registry.yml",
    "rendered_sentence": (
        "The low-field-side equatorial midplane: the point of maximum major "
        "radius on the flux surface in question. For a value at the separatrix "
        "intersection use outboard_midplane_separatrix. Where "
        "$R = \\max_{S} R,\\quad R > R_{\\mathrm{axis}}$, $S$ is the magnetic "
        "flux surface the point lies on; $R_{\\mathrm{axis}}$ is major radius "
        "of the magnetic axis."
    ),
    "names_a_second_height": False,
    "names_the_midplane_identifier": False,
    "method": (
        "locus_definition_sentence() (imas_codex/standard_names/docs_gates.py) "
        "was called for the outboard_midplane concept the writer appends, "
        "obtained through select_conventions(name).concepts; the rendered string "
        "was then searched for the terms the superseded definition used."
    ),
}

#: The state both names were in before this node ran, read from the live graph.
BEFORE = {
    "radial_separatrix_distance_at_outboard_midplane": {
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "reviewer_score_name": 1.0,
        "reviewer_score_docs": 0.90625,
        "docs_model": "openrouter/openai/gpt-6.1-sol",
        "source_paths": [
            "dd:langmuir_probes/embedded/distance_separatrix_midplane",
            "dd:langmuir_probes/reciprocating/plunge/distance_separatrix_midplane",
            "dd:thomson_scattering/channel/distance_separatrix_midplane",
        ],
        "description": (
            "Signed major-radius offset from the magnetic separatrix after a "
            "measurement position is mapped along its flux surface to the "
            "outboard midplane, positive in the scrape-off layer."
        ),
        "names_midplane_identifier": True,
        "superseded_locus_sentence": (
            "The low-field-side equatorial midplane: points at the midplane "
            "height with major radius greater than the magnetic-axis radius. "
            "The data's midplane identifier selects that height, or it is the "
            "magnetic-axis height when none is given. For a value at the "
            "separatrix intersection use outboard_midplane_separatrix"
        ),
    },
    "radial_coordinate_at_outboard_midplane": {
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "reviewer_score_name": 0.9875,
        "reviewer_score_docs": 0.9625,
        "docs_model": "openrouter/openai/gpt-5.6-luna",
        "source_paths": ["dd:equilibrium/time_slice/profiles_1d/r_outboard"],
        "description": (
            "Major-radius coordinate of the low-field-side intersection between "
            "a magnetic flux surface and the equatorial midplane."
        ),
        "names_midplane_identifier": False,
        "states_symmetry_condition": False,
    },
}

#: The superseded locus wording the read-back check looks for.
MIDPLANE_IDENTIFIER_RE = re.compile(
    r"midplane[\s_-]*identifier|identifier\s+selects", re.IGNORECASE
)
#: A DD field name, an explicit IDS/Cypher path, or a DD option value.
DD_FIELD_RE = re.compile(
    r"\bmidplane_identifier\b|"
    r"\bequilibrium/[a-z0-9_/]+|"
    r"\bsummary/[a-z0-9_/]+|"
    r"\blangmuir_probes/[a-z0-9_/]+|"
    r"\bthomson_scattering/[a-z0-9_/]+|"
    r"\b[a-z_]+_dummy\b|\bdr_dz_zero[a-z_]*\b"
)
OPTION_RE = re.compile(r"'(dr_dz_zero[a-z_]*)'|\boption\s+'[^']+'")


def _load_json(path: str):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def locus_sentence(name: str) -> str:
    from imas_codex.standard_names.docs_gates import locus_definition_sentence
    from imas_standard_names.grammar.terms import select_conventions

    applied = select_conventions(name)
    for concept in applied.concepts:
        if concept.token == "outboard_midplane":
            return locus_definition_sentence(concept)
    return ""


def snapshot() -> dict:
    """Read both names' lifecycle, scores and documentation from the graph."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.id IN $ids
                RETURN sn.id AS id, properties(sn) AS props
                ORDER BY sn.id
                """,
                ids=NAMES,
            )
        )
        out: dict = {"names": {}, "uri": gc.uri}
        for row in rows:
            props = dict(row["props"])
            out["names"][row["id"]] = {
                "name_stage": props.get("name_stage"),
                "docs_stage": props.get("docs_stage"),
                "reviewer_score_name": props.get("reviewer_score_name"),
                "reviewer_score_docs": props.get("reviewer_score_docs"),
                "docs_chain_length": props.get("docs_chain_length"),
                "docs_model": props.get("docs_model"),
                "edit_status": props.get("edit_status"),
                "edit_mode": props.get("edit_mode"),
                "source_paths": props.get("source_paths"),
                "description": props.get("description"),
                "documentation": props.get("documentation"),
                "run_id": props.get("run_id"),
                "documentation_present": bool(props.get("documentation")),
            }
        out["_missing"] = sorted(set(NAMES) - set(out["names"]))
    return out


def before() -> dict:
    """The pre-node documentation, read from the graph's own revision history.

    DocsRevision keeps each persisted documentation revision with its score, so
    the before-state text is read rather than transcribed. The pre-node revision
    is the one carrying the score the live graph held when this node began.
    """
    revisions = _load_json(os.path.join(_LOGS, "docs-revisions.json")) or {}
    out = {name: dict(entry) for name, entry in BEFORE.items()}
    for name, entry in out.items():
        target = entry.get("reviewer_score_docs")
        candidates = [
            rev
            for rev in revisions.get(name, [])
            if rev.get("reviewer_score_docs") == target
        ]
        if candidates:
            rev = max(candidates, key=lambda item: item.get("revision_number") or 0)
            entry["documentation"] = rev.get("documentation")
            entry["description"] = rev.get("description")
            entry["revision_number"] = rev.get("revision_number")
            entry["documentation_source"] = "DocsRevision"
        else:
            entry["documentation_source"] = "unavailable"
    return out


def comments() -> dict:
    """Docs-axis reviewer comments for both names, oldest first."""
    data = _load_json(os.path.join(_LOGS, "reviewer-comments.json"))
    if data is None:
        return {name: [] for name in NAMES}
    return data


def _graph_comments() -> dict:
    """Docs-axis reviewer comments read live from the graph."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                MATCH (r:StandardNameReview)
                WHERE r.standard_name_id IN $ids
                RETURN r.standard_name_id AS id, properties(r) AS props
                """,
                ids=NAMES,
            )
        )
    out: dict = {name: [] for name in NAMES}
    for row in rows:
        props = dict(row["props"])
        if (props.get("review_axis") or "") != "docs":
            continue
        out.setdefault(row["id"], []).append(
            {
                key: props.get(key)
                for key in (
                    "cycle_index",
                    "reviewer_model",
                    "resolution_role",
                    "score",
                    "tier",
                    "comments",
                    "reviewed_at",
                )
            }
        )
    for name in out:
        out[name].sort(key=lambda item: str(item.get("reviewed_at") or ""))
    return out


def read_checks(state: dict) -> dict:
    """Per-name read-back: the score, and that the text names no midplane id."""
    threshold = 0.85
    checks: dict = {}
    for name in NAMES:
        entry = state["names"].get(name, {})
        text = " ".join(
            str(part)
            for part in (entry.get("description"), entry.get("documentation"))
        )
        ident_hits = sorted(set(MIDPLANE_IDENTIFIER_RE.findall(text)))
        field_hits = sorted(set(DD_FIELD_RE.findall(text)))
        option_hits = sorted(set(OPTION_RE.findall(text)))
        score = entry.get("reviewer_score_docs")
        checks[name] = {
            "has_documentation": bool(entry.get("documentation")),
            "docs_stage": entry.get("docs_stage"),
            "score": score,
            "score_at_or_above_threshold": (
                score is not None and float(score) >= threshold
            ),
            "names_the_midplane_identifier": bool(ident_hits),
            "midplane_identifier_hits": ident_hits,
            "names_dd_field_or_path": bool(field_hits),
            "dd_field_or_path_hits": field_hits,
            "names_option_value": bool(option_hits),
            "option_value_hits": option_hits,
        }
    return checks


def _summary_table(path: str) -> dict:
    """Read a run's own summary table and exit status out of its log."""
    try:
        with open(path) as handle:
            text = handle.read()
    except FileNotFoundError:
        return {"log_missing": path}
    out: dict = {"log": path}
    for key in ("stop_reason", "cost_spent", "cost_limit", "names_reviewed"):
        match = re.search(rf"│\s*{key}\s*│\s*([^│]+)│", text)
        if match:
            out[key] = match.group(1).strip()
    match = re.search(r"EXIT=(\d+)", text)
    if match:
        out["exit_code"] = int(match.group(1))
    return out


def spend() -> dict:
    """Per-run spend, read from each run's own summary table in its log."""
    per_run = {
        label: _summary_table(spec["log"]) for label, spec in INVOCATIONS.items()
    }
    first = _summary_table(INVOCATIONS["radial_coordinate"]["first_run_log"])
    total = 0.0
    for entry in list(per_run.values()) + [first]:
        try:
            total += float(entry.get("cost_spent", 0.0))
        except (TypeError, ValueError):
            pass
    return {
        "authorized_usd": 1.50,
        "basis": "each run's own summary table in its log",
        "runs": per_run,
        "radial_coordinate_first_run": first,
        "node_total_usd": round(total, 6),
        "node_remaining_usd": round(1.50 - total, 6),
    }


def review_verdicts() -> dict:
    """The last docs review round per name: seats, scores, and the accepted flag.

    A round is the set of seats sharing one ``reviewed_at`` stamp. The live
    ``reviewer_score_docs`` follows the latest persisted revision, which a
    following refinement can reset, so the verdict is read from the review
    rounds themselves.
    """
    data = comments() or {}
    out: dict = {}
    for name in NAMES:
        rounds: dict = {}
        for seat in data.get(name, []):
            if seat.get("reviewer_model") == "typesafe/jev-1.13":
                continue
            rounds.setdefault(str(seat.get("reviewed_at")), []).append(
                {
                    "reviewer_model": seat.get("reviewer_model"),
                    "score": seat.get("score"),
                }
            )
        if not rounds:
            out[name] = {"rounds": {}, "last_round": None}
            continue
        last = sorted(rounds)[-1]
        scores = [s["score"] for s in rounds[last] if s.get("score") is not None]
        score = round(sum(scores) / len(scores), 6) if scores else None
        out[name] = {
            "rounds": rounds,
            "last_round": last,
            "last_round_seats": rounds[last],
            "last_round_score": score,
            "accepted": bool(score is not None and score >= 0.85),
        }
    return out


def report() -> dict:
    state = snapshot()
    checks = read_checks(state)
    return {
        "node": "shc-midplane-docs-refresh",
        "route": INVOCATIONS,
        "locus_definition_check": LOCUS_DEFINITION_CHECK,
        "graph_uri": state.get("uri"),
        "before": before(),
        "after": state["names"],
        "read_checks": checks,
        "review_verdicts": review_verdicts(),
        "reviewer_comments": comments(),
        "spend": spend(),
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "locus":
        for name in NAMES:
            print(name, "->", repr(locus_sentence(name)))
    elif mode == "snapshot":
        print(json.dumps(snapshot(), indent=2, default=str, sort_keys=True))
    elif mode == "comments":
        print(json.dumps(comments(), indent=2, default=str, sort_keys=True))
    elif mode == "report":
        record = report()
        with open(RECORD, "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        print(json.dumps(record["read_checks"], indent=2, default=str))
        print("wrote", RECORD)
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""Record the outcome of regenerating two names' documentation after the split.

The split gave each name one quantity: ``gap_at_outboard_midplane`` keeps the
separatrix separation, and
``radial_separatrix_distance_at_outboard_midplane`` holds the three mapped
measurement distances. This record drives the documentation stage over the two
named standard names through the pipeline's own routes, then reads both names'
lifecycle, scores, documentation and reviewer comments from the live graph and
evaluates the read-back checks the work item asks for.

The documentation stage's pools claim ``generate_docs`` (name accepted, docs
pending), ``review_docs`` (docs drafted) and ``refine_docs`` is the refinement
arm. Neither name was in a claimable state at the start — one held exhausted
docs, the other a name that had not cleared review — so each was first moved
into the documentation stage through a sanctioned route and then driven
through it:

* ``gap_at_outboard_midplane`` — ``sn edit --hint --axis docs`` attached a
  steering direction and reason, resetting its docs out of ``exhausted`` and
  regenerating them; the documentation pools then reviewed them below the
  acceptance threshold, so a second steered edit re-attempted it.
* ``radial_separatrix_distance_at_outboard_midplane`` — no ``sn edit`` mode can
  re-review a name in place
  without renaming it, so the name was re-reviewed through the pipeline's own
  rotation (``sn run --names-only --name``), which scored it as it stands and
  accepted it; its documentation then ran and was accepted.

Nothing here writes to the graph. Modes:
    snapshot   read both names' lifecycle, scores and documentation
    report     assemble the fragment's JSON record beside this runner
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "shc-gap-docs-regen.json")

NAMES = [
    "gap_at_outboard_midplane",
    "radial_separatrix_distance_at_outboard_midplane",
]

_RUNS = "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261020T202608740663-shc-gap-docs-regen"
_LOGS = os.path.join(_RUNS, "logs")

#: The first steering edit: it reset gap's docs out of ``exhausted`` and
#: regenerated them under the documentation seat.
GAP_EDIT_COMMAND = (
    "uv run --no-sync imas-codex sn edit gap_at_outboard_midplane "
    "--hint '<locus text in physics terms: the separation between the inner and "
    "outer separatrices at the outboard midplane, naming no DD field, IDS path "
    "or option value>' --axis docs --scope self --reason '<the lead's "
    "one-quantity-per-name ruling of 2026-10-18>'"
)
GAP_EDIT_RUN_ID = "sn-edit-20261020T205652Z"
GAP_EDIT_NOTE = (
    "The edit attached and reset+regenerated the docs (docs_stage exhausted -> "
    "drafted) but its inline review was stopped by the harness's ten-minute "
    "background-command limit before producing a score, leaving "
    "edit_status='open' and a claim. The review was then completed by the docs "
    "pools."
)

#: gap's first documentation review: the docs-only scope, run against the
#: edit's own run id (the rolling iteration over the ``--name`` form was
#: refused while the interrupted edit still held a worker claim).
GAP_DOCS_COMMAND = (
    "uv run --no-sync imas-codex sn run --docs-only "
    "--scope-run-id sn-edit-20261020T205652Z --cost-limit 0.45 "
    "--skip-global-maintenance"
)
GAP_DOCS_LOG = os.path.join(_LOGS, "gap-scoped-review.log")

#: The second steering edit, staged without the inline review so the whole
#: review runs once inside a single cost-capped scope.
GAP_STEER2_EDIT_COMMAND = (
    "uv run --no-sync imas-codex sn edit gap_at_outboard_midplane "
    "--hint '<reconcile the locus sentence: Z* is how the outboard midplane is "
    "located for this quantity, so the trailing midplane sentence describes "
    "that same height and never a second one; keep the opening, equation and "
    "cross-references; name no DD field, IDS path or option value>' "
    "--axis docs --scope self --reason '<both reviewers found the body "
    "asserting a midplane height beside Z*>' --stage-only"
)
GAP_STEER2_EDIT_LOG = os.path.join(_LOGS, "gap-steer2-edit-stage.log")
GAP_STEER2_RUN_ID = "sn-edit-20261020T215855Z"

#: The sweep tick that released the threaded refining claim the interrupted
#: refinement left behind, so the re-edit could attach.
GAP_SWEEP_NOTE = (
    "The refinement the first review triggered claimed gap and died, leaving "
    "docs_stage='refining' with a claim. The pipeline's own orphan sweep "
    "(docs_refining arm, age threshold 1800 s) reverted it to 'reviewed' and "
    "cleared the claim; the stage was never written by hand."
)
GAP_SWEEP_LOG = os.path.join(_LOGS, "gap-orphan-sweep-tick.log")

#: gap's second documentation review: one scope over the re-edit's run id.
GAP_STEER2_DOCS_COMMAND = (
    "uv run --no-sync imas-codex sn run --docs-only "
    "--scope-run-id sn-edit-20261020T215855Z --cost-limit 0.28 "
    "--skip-global-maintenance"
)
GAP_STEER2_DOCS_LOG = os.path.join(_LOGS, "gap-steer2-docs-review.log")

#: What that scope did and why it could not settle the name. The generation ran
#: and spent; the three-seat review quorum could never reserve its second seat,
#: so every cycle deferred without accepting and no score was persisted.
GAP_STEER2_OUTCOME = {
    "generate_docs": {"processed": 1, "spent_usd": 0.0958},
    "review_docs": {
        "processed": 0,
        "spent_usd": 0.1572,
        "note": (
            "107 review cycles deferred for want of budget; one seat ran and "
            "was discarded because the quorum was incomplete, so no score was "
            "persisted. The pool's spend of 0.1572 bought no accepted verdict."
        ),
    },
    "run_total_usd": 0.2530,
    "run_stop": (
        "stopped deliberately after the budget wall proved permanent: the "
        "review pool's remaining lease was $0.0456 against a next-seat need of "
        "$0.0457, so no further cycle could ever succeed."
    ),
    "review_outcome": (
        "no new score. The steered candidate is persisted at docs_stage="
        "'drafted' with reviewer_score_docs null and docs_chain_length 0, and "
        "the next funded review rotation claims it."
    ),
    "steer_effect": (
        "the regenerated text keeps the trailing midplane sentence unchanged, "
        "still asserting the midplane height beside Z*, which is the defect "
        "both reviewers docked. The steer did not achieve the reconciliation "
        "it asked for, independently of the review not running."
    ),
}

#: radial's name: no ``sn edit`` mode re-reviews a name in place, so the name
#: was re-reviewed through the pipeline's own rotation.
RADIAL_NAME_COMMAND = (
    "uv run --no-sync imas-codex sn run --names-only "
    "--name radial_separatrix_distance_at_outboard_midplane --cost-limit 0.30 "
    "--skip-global-maintenance"
)
RADIAL_NAME_LOG = os.path.join(_LOGS, "radial-name-review.log")
RADIAL_COMMENTS_BEFORE = os.path.join(_LOGS, "radial-name-reviewer-comments-before.json")
RADIAL_COMMENTS_AFTER = os.path.join(_LOGS, "radial-name-reviewer-comments-after.json")

#: radial's documentation: the docs-only scope over the now-accepted name.
RADIAL_DOCS_COMMAND = (
    "uv run --no-sync imas-codex sn run --docs-only "
    "--name radial_separatrix_distance_at_outboard_midplane --cost-limit 0.45 "
    "--skip-global-maintenance"
)
RADIAL_DOCS_LOG = os.path.join(_LOGS, "radial-docs-review.log")

#: The state both names were in when the first documentation-only attempt was
#: refused, so the record carries a before; ``after`` is the live read below.
BEFORE = {
    "gap_at_outboard_midplane": {
        "name_stage": "accepted",
        "docs_stage": "exhausted",
        "reviewer_score_docs": 0.75,
        "documentation_present": True,
        "note": (
            "docs regenerated by an earlier steered edit and left exhausted "
            "below threshold; the text still named the DD field "
            "midplane_identifier and the option value dr_dz_zero_sep"
        ),
    },
    "radial_separatrix_distance_at_outboard_midplane": {
        "name_stage": "drafted",
        "docs_stage": "pending",
        "reviewer_score_docs": None,
        "reviewer_score_name": 0.80625,
        "documentation_present": False,
        "note": (
            "revived to drafted by the split; its name score 0.80625; it could "
            "receive no documentation until its name cleared acceptance"
        ),
    },
}

#: gap's state between the two steering edits.
BEFORE_STEER2 = {
    "gap_at_outboard_midplane": {
        "name_stage": "accepted",
        "docs_stage": "reviewed",
        "reviewer_score_docs": 0.50,
        "docs_chain_length": 2,
        "note": (
            "the first steering edit's regenerated documentation was reviewed "
            "at 0.5625 and then 0.500, below the 0.85 threshold; the refinement "
            "that followed claimed the name and died, leaving docs_stage "
            "'refining' until the orphan sweep reverted it"
        ),
    }
}

#: The path probes that establish why radial's name is re-reviewed by the
#: rotation rather than by an ``sn edit``.
RADIAL_EDIT_PROBES = [
    {
        "command": (
            "sn edit ... --rename radial_separatrix_distance_at_outboard_midplane "
            "--dry-run"
        ),
        "result": (
            "BLOCKED: a StandardName "
            "'radial_separatrix_distance_at_outboard_midplane' already exists "
            "— a same-spelling rename is refused, and any other rename would "
            "respell the lead's ruled name"
        ),
    },
    {
        "command": (
            "sn rescore radial_separatrix_distance_at_outboard_midplane --dry-run"
        ),
        "result": (
            "EXIT=2: 'is name_stage=drafted, status=draft — rescore only "
            "recovers exhausted or reviewed names'"
        ),
    },
    {
        "command": "sn edit ... --hint <ruling> --axis name --dry-run",
        "result": (
            "accepted (entry=generate) — but a name-axis hint enters "
            "GENERATE_NAME, which regenerates the name rather than judging it "
            "as it stands, so it is not an in-place re-review"
        ),
    },
]

#: A DD field name, an explicit IDS/Cypher path, or a DD option value.
DD_FIELD_RE = re.compile(
    r"\bmidplane_identifier\b|\bdd:[a-z_]+/|\bsummary/[a-z_/]+|"
    r"\b[a-z_]+_dummy\b|\bdr_dz_zero[a-z_]*\b"
)
OPTION_RE = re.compile(r"'(dr_dz_zero[a-z_]*)'|\boption\s+'[^']+'")


def _load(path: str) -> dict:
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}


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
        out = {"names": {}, "uri": gc.uri}
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
                "description": props.get("description"),
                "documentation": props.get("documentation"),
                "source_paths": props.get("source_paths"),
                "run_id": props.get("run_id"),
                "claimed_at": str(props.get("claimed_at") or ""),
                "documentation_present": bool(props.get("documentation")),
            }
        out["_missing"] = sorted(set(NAMES) - set(out["names"]))
    return out


def read_checks(state: dict) -> dict:
    """Per-name read-back checks the plan asks for."""
    threshold = 0.85
    checks = {}
    for name in NAMES:
        entry = state["names"].get(name, {})
        text = " ".join(
            str(part)
            for part in (entry.get("description"), entry.get("documentation"))
        )
        field_hits = sorted(set(DD_FIELD_RE.findall(text)))
        option_hits = sorted(set(OPTION_RE.findall(text)))
        score = entry.get("reviewer_score_docs")
        checks[name] = {
            "has_documentation": bool(entry.get("documentation")),
            "score": score,
            "score_at_or_above_threshold": (
                score is not None and float(score) >= threshold
            ),
            "names_dd_field_or_path": bool(field_hits),
            "dd_field_or_path_hits": field_hits,
            "names_option_value": bool(option_hits),
            "option_value_hits": option_hits,
        }
    return checks


def report() -> dict:
    """Assemble the fragment's JSON record."""
    state = snapshot()
    checks = read_checks(state)
    return {
        "node": "shc-gap-docs-regen",
        "route": {
            "gap_edit": GAP_EDIT_COMMAND,
            "gap_edit_run_id": GAP_EDIT_RUN_ID,
            "gap_edit_note": GAP_EDIT_NOTE,
            "gap_docs": GAP_DOCS_COMMAND,
            "gap_docs_log": GAP_DOCS_LOG,
            "gap_orphan_sweep_note": GAP_SWEEP_NOTE,
            "gap_orphan_sweep_log": GAP_SWEEP_LOG,
            "gap_steer2_edit": GAP_STEER2_EDIT_COMMAND,
            "gap_steer2_edit_log": GAP_STEER2_EDIT_LOG,
            "gap_steer2_run_id": GAP_STEER2_RUN_ID,
            "gap_steer2_docs": GAP_STEER2_DOCS_COMMAND,
            "gap_steer2_docs_log": GAP_STEER2_DOCS_LOG,
            "radial_name": RADIAL_NAME_COMMAND,
            "radial_name_log": RADIAL_NAME_LOG,
            "radial_docs": RADIAL_DOCS_COMMAND,
            "radial_docs_log": RADIAL_DOCS_LOG,
        },
        "radial_edit_limitation": {
            "finding": (
                "no sn edit mode re-reviews a name in place: hint mode on the "
                "name axis enters GENERATE_NAME (a regeneration, not a "
                "re-review as it stands), rename mode changes the spelling and "
                "its same-name form is refused, and --docs touches only the "
                "documentation axis; sn rescore refuses a drafted name. The "
                "name was therefore re-reviewed through the pipeline's own "
                "rotation, which judges it as it stands without renaming."
            ),
            "probes": RADIAL_EDIT_PROBES,
        },
        "gap_steer2_outcome": GAP_STEER2_OUTCOME,
        "radial_name_review": {
            "comments_before": _load(RADIAL_COMMENTS_BEFORE),
            "comments_after": _load(RADIAL_COMMENTS_AFTER),
        },
        "state": state["names"],
        "before": BEFORE,
        "before_steer2": BEFORE_STEER2,
        "after": state["names"],
        "read_checks": checks,
        "spend": {
            "authorized_usd": 1.0,
            "basis": "each run's own summary table in its log",
            "gap_docs_usd": 0.406322,
            "radial_name_usd": 0.060014,
            "radial_docs_usd": 0.226306,
            "gap_steer2_run_usd": 0.2530,
            "node_total_usd": 0.945642,
            "node_remaining_usd": 0.054358,
            "note": (
                "the node's measured spend across both attempts. The "
                "interrupted steering edit's generation cost is not in the "
                "first attempt's total. The second attempt's review quorum "
                "needed roughly $0.20 against $0.054 left, so the cap bound "
                "the verdict: the ceiling is the reason the name is "
                "unreviewed, not the pipeline."
            ),
        },
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "snapshot":
        print(json.dumps(snapshot(), indent=2, default=str, sort_keys=True))
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

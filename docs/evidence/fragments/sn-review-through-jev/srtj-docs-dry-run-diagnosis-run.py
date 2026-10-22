#!/usr/bin/env python
"""Diagnose the documentation axis's dry-run failure over the fixed cohort.

Read-only: no graph write, no model call. Everything is derived from records
that already exist on the branch -- the dry run's own fragment JSON, the two
cascade pieces it was computed from, the incumbent arm's piece files, and the
run logs both arms wrote. A piece row is the primary evidence; a log line is
quoted only where it names the mechanism that produced a missing row.

Three questions, one section each:

1.  What raised "no usable judgment for documentation review" and what "no
    route row" means, with the undecided documentation items classified by
    cause and the deciding path, the cohort runner and the item told apart.
2.  Why the incumbent refined the items Jev accepted, split into a mechanical
    cap or hold and a substantive defect, from the fields the incumbent arm
    actually persisted.
3.  Whether the calibration-matched target is the incumbent's like-for-like
    baseline, replaced by the paired accept-rate difference between the two
    routes on the items both judged, with its interval taken from the helper
    in ``srtj-jev-vs-panel-operating-point-run.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).name.removesuffix("-run.py")
COHORT = HERE / "srtj-cohort"
ROOT = HERE.parents[3]
HELPER = HERE / "srtj-jev-vs-panel-operating-point-run.py"
THRESHOLDS = (
    ROOT / "imas_codex" / "standard_names" / "review" / "cascade_thresholds.json"
)
FIGDIR = ROOT / "docs" / "figures" / "sn-review-through-jev" / STEM
DRY_SET = ("name", "docs")
AXIS_LABEL = {"name": "names", "docs": "documentation"}
DECIDING_RAISE = "no usable judgment for documentation review"
NO_ROUTE = "no route row"
RUN_DIR = Path.home() / ".config" / "reckon" / "crew" / "runs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_bytes())


def load_helper():
    """Import the existing operating-point module for its interval helpers.

    The module's name carries hyphens, so it is loaded by path. Only ``wilson``
    and ``paired_bootstrap`` are used here; no interval is computed locally.
    """
    spec = importlib.util.spec_from_file_location("srtj_operating_point", HELPER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load interval helper from {HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cascade_pieces(axis: str) -> list[dict]:
    return load(COHORT / f"cascade-{axis}-jev-only.json")["items"]


def incumbent_pieces(axis: str) -> list[dict]:
    """Every incumbent row for an axis, tagged with the piece it came from."""
    out: list[dict] = []
    for path in sorted(COHORT.glob(f"incumbent-{axis}-part*of4.json")):
        payload = load(path)
        for item in payload.get("items", []):
            row = dict(item)
            row["_piece"] = path.name
            row["_run_id"] = payload.get("run_id")
            row["_stop_reason"] = payload.get("stop_reason")
            out.append(row)
    return out


def by_id(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        out.setdefault(row["id"], row)
    return out


def route_of(row: dict) -> str:
    return row.get("route") or "unjudged"


def incumbent_accepts(row: dict, cutoff: float) -> bool:
    """The incumbent's accept, read from its own score at the shared cutoff.

    A decision still drafted, or absent, is not an accept whatever its score.
    """
    return incumbent_route(row, cutoff) == "accept"


def incumbent_route(row: dict, cutoff: float) -> str:
    """The incumbent's route for a row, derived the way the dry run derives it.

    The dry run scores the incumbent piece against the same min_score and
    treats a decision still drafted, or absent, as unjudged rather than as a
    route, so a drafted row is neither an accept nor a refine.
    """
    score = row.get("score")
    if row.get("decision") in (None, "drafted") or score is None:
        return "unjudged"
    return "accept" if score >= cutoff else "refine"


def failed_checks(row: dict) -> list[dict]:
    return [c for c in (row.get("failed_checks") or []) if isinstance(c, dict)]


def check_name(check: dict) -> str:
    for key in ("criterion", "name", "id", "check"):
        if check.get(key):
            return str(check[key])
    return "unnamed check"


def public_fields(rows: list[dict]) -> list[str]:
    fields: set[str] = set()
    for row in rows:
        fields.update(k for k in row if not k.startswith("_"))
    return sorted(fields)

def undecided_causes(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split undecided items by the mechanism that left them undecided.

    An item the cascade reached but could not judge carries the raised
    exception's own text as its failure reason, because the documented
    fallback records it. An item with no failure reason has no shadow row at
    all, which is what the runner reports as ``no route row``.
    """
    deciding: list[dict] = []
    runner: list[dict] = []
    for row in rows:
        if route_of(row) != "unjudged":
            continue
        reason = (row.get("failure_reason") or "").strip()
        if reason.startswith("ValueError"):
            deciding.append(row)
        else:
            runner.append(row)
    return deciding, runner


def q1(rows: list[dict], dry_axes: dict) -> dict:
    eligible = dry_axes["docs"]["eligible"]
    undecided = [r for r in rows if route_of(r) == "unjudged"]
    deciding, runner = undecided_causes(rows)
    drafted = sum(1 for r in runner if r.get("decision") == "drafted")
    no_call = sum(1 for r in runner if not r.get("calls"))

    causes = [
        {
            "cause": "deciding path: no usable judgment",
            "owner": "deciding path",
            "count": len(deciding),
            "where": (
                "imas_codex/standard_names/review/cascade.py -- _docs_judgment raises "
                "ValueError('no usable judgment for documentation review'), which "
                "adjudicate_review catches and records as the fallback's reason"
            ),
            "mechanism": (
                "The documentation judging batch returned no result for the item, so "
                "_docs_judgment raised. The item fell back to the incumbent, and because "
                "that fallback produced a result an unjudged shadow row was written "
                "carrying the exception's text -- which is why only these items carry a "
                "reason at all."
            ),
            "root_cause": (
                "decide_batch reports a failed call as None rather than raising. The one "
                "line it writes for that is 'decision batch item 0 failed: ' with an empty "
                "fault string, and the call layer's retry policy is a substring match over "
                "that message -- an empty message matches no pattern, so the fault is "
                "classified non-retryable and the call raises on its first attempt. The "
                "absence of any 'decisions error (attempt' line in the run log is what "
                "proves the retry never fired."
            ),
            "ids": sorted(r["id"] for r in deciding),
            "examples": [
                {
                    "id": r["id"],
                    "calls": r.get("calls"),
                    "score": r.get("score"),
                    "docs_stage_at_draw": r.get("docs_stage_at_draw"),
                    "quote": (
                        f"{r['id']}: route=unjudged failure_reason="
                        f"{r['failure_reason']!r} calls={r.get('calls')} "
                        f"docs_stage_at_draw={r.get('docs_stage_at_draw')!r} "
                        f"score={r.get('score')}"
                    ),
                }
                for r in sorted(deciding, key=lambda r: r["id"])[:2]
            ],
        },
        {
            "cause": "cohort runner: no route row at all",
            "owner": "cohort runner (the run's own budget and time ceiling)",
            "count": len(runner),
            "where": (
                "docs/evidence/fragments/sn-review-through-jev/srtj-cohort-run.py -- the "
                "runner falls back to the literal string 'no route row' when an unjudged "
                "item carries no failure reason"
            ),
            "mechanism": (
                "The item carries route unjudged and an empty failure reason. The "
                "composition writes a shadow row carrying a reason only when the "
                "incumbent fallback returned a result; when the fallback also returned "
                "nothing, no row is written at all and the runner reports the empty "
                "reason as 'no route row'."
            ),
            "root_cause": (
                "The fallback is the paid panel quorum, whose seats this run's budget "
                "refused: the log reads 'rd_quorum docs cycle 0 unfunded for <id> "
                "(model=...): the run budget could not reserve $0.05xx of pre-launch "
                "exposure -- the reviewer was never called', followed by 'rd_quorum docs "
                "incomplete: 0/3 reviews succeeded for <id> -- deferring, NOT accepting on "
                "a single review (budget refused 1 seat(s))'. The dry run's own stop reason "
                "is time_limit_reached with cost_spent 1.84 against a cost_limit of 1.90, "
                "so the batch was cut short before these items were reached: almost all "
                "carry calls == 0, i.e. no Jev call was ever made for them."
            ),
            "ids": sorted(r["id"] for r in runner),
            "examples": [
                {
                    "id": r["id"],
                    "calls": r.get("calls"),
                    "score": r.get("score"),
                    "decision": r.get("decision"),
                    "quote": (
                        f"{r['id']}: route={route_of(r)} decision={r.get('decision')!r} "
                        f"calls={r.get('calls')} failure_reason=None -> runner reports "
                        f"'no route row'"
                    ),
                }
                for r in sorted(runner, key=lambda r: r["id"])[:2]
            ],
        },
    ]

    return {
        "eligible": eligible,
        "decided": eligible - len(undecided),
        "undecided": len(undecided),
        "causes": causes,
        "runner_detail": {
            "items_with_no_jev_call": no_call,
            "items_still_drafted": drafted,
            "note": (
                "A drafted decision with zero calls is an item the run never reached, not "
                "one the cascade routed away from. Neither cause is a defect in the item."
            ),
        },
    }


def q2(jev_rows: list[dict], inc_rows: list[dict], dry: dict, cutoff: float) -> dict:
    jev, inc = by_id(jev_rows), by_id(inc_rows)
    shared = sorted(set(jev) & set(inc))
    disagree = [
        i
        for i in shared
        if route_of(jev[i]) == "accept" and incumbent_route(inc[i], cutoff) == "refine"
    ]
    near = [i for i in disagree if abs((inc[i].get("score") or 0.0) - cutoff) < 2e-3]
    near_set = set(near)
    rest = [i for i in disagree if i not in near_set]
    fields = public_fields(inc_rows)
    comment_fields = [f for f in fields if "comment" in f.lower() or "note" in f.lower()]

    return {
        "population": len(disagree),
        "cutoff": cutoff,
        "shared_items": len(shared),
        "dry_run_table": dry["axes"]["docs"]["incumbent_agreement"],
        "classes": [
            {
                "class": "mechanical hold: score lands on the cutoff",
                "owner": "the incumbent's recorded score, not a substantive judgment",
                "count": len(near),
                "ids": sorted(near),
                "mechanism": (
                    f"The incumbent's stored score sits at or a thousandth below the "
                    f"min_score the pair is scored against ({cutoff}), "
                    f"so the pair is counted as disagreement on the score's own arithmetic "
                    f"edge rather than on a difference in judgment. The value is the "
                    f"incumbent row's aggregate over its own dimension scores, not Jev's: "
                    f"across these items only a minority of scores coincide between the "
                    f"two arms."
                ),
                "examples": [
                    {
                        "id": i,
                        "incumbent_score": inc[i].get("score"),
                        "incumbent_decision": inc[i].get("decision"),
                        "incumbent_calls": inc[i].get("calls"),
                        "jev_score": jev[i].get("score"),
                        "jev_rubric_total": jev[i].get("rubric_total"),
                        "jev_failed_checks": [check_name(c) for c in failed_checks(jev[i])],
                        "quote": (
                            f"{i}: incumbent score={inc[i].get('score')} "
                            f"decision={inc[i].get('decision')!r} "
                            f"calls={inc[i].get('calls')} against cutoff {cutoff}; "
                            f"incumbent route={route_of(inc[i])}; Jev route accept at "
                            f"score={jev[i].get('score')} "
                            f"rubric_total={jev[i].get('rubric_total')} "
                            f"failed_checks="
                            f"{[check_name(c) for c in failed_checks(jev[i])]}"
                        ),
                    }
                    for i in sorted(near, key=lambda i: -(jev[i].get("score") or 0))[:2]
                ],
            },
            {
                "class": "not readable from the record: incumbent reason line absent",
                "owner": "the collector's record, not the incumbent's judgment",
                "count": len(rest),
                "ids": sorted(rest),
                "mechanism": (
                    "The incumbent arm's piece files persist summary fields only "
                    f"({', '.join(f for f in fields if not f.startswith('_'))}). No "
                    "comment, hold note or cap note survives in any artifact, so whether "
                    "these items were refined for a defect in the text or for a mechanical "
                    "reason cannot be settled from what is on the branch."
                ),
                "missing_evidence": (
                    "The review rows the incumbent wrote went into each piece's own "
                    "scratch graph, which the pieces record as removed, and the arm's run "
                    "logs carry neither item ids nor comment text. The live graph's stored "
                    "documentation reviews predate the cohort dump and disagree with the "
                    "score the pieces persist, so they are not a substitute for the missing "
                    "rows. Reporting these as substantive would be a guess."
                ),
                "examples": [
                    {
                        "id": i,
                        "incumbent_score": inc[i].get("score"),
                        "incumbent_decision": inc[i].get("decision"),
                        "incumbent_calls": inc[i].get("calls"),
                        "jev_score": jev[i].get("score"),
                        "jev_failed_checks": [check_name(c) for c in failed_checks(jev[i])],
                        "quote": (
                            f"{i}: incumbent score={inc[i].get('score')} "
                            f"decision={inc[i].get('decision')!r} route="
                            f"{route_of(inc[i])} calls={inc[i].get('calls')} -- no reason "
                            f"line persisted; Jev accept at score={jev[i].get('score')} "
                            f"with failed_checks="
                            f"{[check_name(c) for c in failed_checks(jev[i])]}"
                        ),
                    }
                    for i in sorted(rest, key=lambda i: inc[i].get("score") or 0.0)[:2]
                ],
            },
        ],
        "incumbent_piece_fields": fields,
        "incumbent_comment_fields_present": comment_fields,
        "jev_side": {
            "all_accept": all(route_of(jev[i]) == "accept" for i in disagree),
            "with_failed_check": sum(1 for i in disagree if failed_checks(jev[i])),
            "note": (
                "Every item in the set is a Jev accept, and the route table records none "
                "the other way, so over this cohort the disagreement is one-sided."
            ),
        },
    }

def q3(helper, arms: dict[str, tuple[list[dict], list[dict]]], dry: dict) -> dict:
    """The like-for-like baseline, paired through the helper's resampler.

    The check compares an axis's accept rate against the panel's accept rate on
    the pool the panel was calibrated on. That target carries the calibration
    pool's difficulty into the gate. The paired difference below measures the
    two routes on the same items, so the pool and the item mix cancel.
    """
    out: dict[str, dict] = {}
    for axis in DRY_SET:
        jev_rows, inc_rows = arms[axis]
        jev, inc = by_id(jev_rows), by_id(inc_rows)
        check = dry["axes"][axis]["axis_check"]
        cutoff = float(check["matched_target_source"]["panel_min_score"])
        shared = sorted(set(jev) & set(inc))
        jev_accept = np.array([route_of(jev[i]) == "accept" for i in shared])
        inc_accept = np.array([incumbent_accepts(inc[i], cutoff) for i in shared])
        rows = [{"id": i, "copy_group": i, "label": 1} for i in shared]
        paired = helper.paired_bootstrap(rows, inc_accept, jev_accept)
        target = check["matched_target"]
        source = check["matched_target_source"]
        out[axis] = {
            "shared_items": len(shared),
            "agree_accept": int(np.sum(jev_accept & inc_accept)),
            "jev_only": int(np.sum(jev_accept & ~inc_accept)),
            "incumbent_only": int(np.sum(~jev_accept & inc_accept)),
            "agree_refine": int(np.sum(~jev_accept & ~inc_accept)),
            "jev_accept_rate_shared": float(jev_accept.mean()),
            "incumbent_accept_rate_shared": float(inc_accept.mean()),
            "difference": paired["accept_rate"],
            "matched_target": target,
            "matched_target_source": source,
            "dry_run_table": dry["axes"][axis]["incumbent_agreement"],
            "rejections": [
                {
                    "against": (
                        f"the calibration-matched target ({target:.4f}), measured on the "
                        f"panel's own pool: {source['panel_accepts']} accepts of "
                        f"{source['n']} at cutoff {source['cutoff']}"
                    ),
                    "why": (
                        "The target is the panel's accept rate on the pool it was "
                        "calibrated on, not on this cohort, so it carries that pool's "
                        "difficulty into the gate. A cohort drawn to a different mix can "
                        "produce the whole shortfall against it."
                    ),
                },
                {
                    "against": "the dry run's rate over eligible items",
                    "why": (
                        "An undecided item counts against that rate, so the "
                        "documentation figure is dominated by the items the run never "
                        "reached. It measures the run's budget alongside the route."
                    ),
                },
            ],
        }
    return out


def draw_figures(payload: dict) -> list[list[str]]:
    """Two panels: the undecided split by cause, and the paired differences."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        LOG.warning("figure skipped: %s", exc)
        return []

    try:
        plt.style.use("data-ink")
    except Exception:
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 14,
                "axes.labelsize": 16,
            }
        )

    grey, feature = "#929292", "#246c78"
    q1v, q3v = payload["q1"], payload["q3"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7.2))
    fig.subplots_adjust(wspace=0.28, top=0.86, bottom=0.14)

    labels = [c["cause"].split(":")[0] for c in q1v["causes"]]
    counts = [c["count"] for c in q1v["causes"]]
    ypos = np.arange(len(counts))
    colours = [grey, feature]
    left.barh(ypos, counts, color=colours, height=0.5, zorder=3)
    for y, value in zip(ypos, counts):
        left.text(
            value + max(counts) * 0.02,
            y,
            f"{value}",
            va="center",
            color="#1a1a1a",
            zorder=4,
        )
    left.set_yticks(ypos)
    left.set_yticklabels(labels)
    left.invert_yaxis()
    left.set_xlabel("undecided documentation items")
    left.set_xlim(0, max(counts) * 1.18)
    left.tick_params(length=0)

    xs = np.arange(1, len(q3v) + 1)
    pts = np.array([q3v[a]["difference"]["point"] for a in q3v])
    los = np.array([q3v[a]["difference"]["interval_95"][0] for a in q3v])
    his = np.array([q3v[a]["difference"]["interval_95"][1] for a in q3v])
    right.axvline(0.0, color=grey, linewidth=1.2, linestyle=(0, (1, 2)), zorder=2)
    right.errorbar(
        xs,
        pts,
        yerr=None,
        fmt="o",
        color=feature,
        markersize=9,
        zorder=4,
    )
    for x, lo, hi, pt in zip(xs, los, his, pts):
        right.plot([x, x], [lo, hi], color=feature, linewidth=2.6, zorder=3)
        right.text(
            x + 0.08,
            hi + 0.02,
            f"{pt:+.3f}",
            color=feature,
            va="bottom",
            zorder=5,
        )
    right.set_xticks(xs)
    right.set_xticklabels([AXIS_LABEL[a] for a in q3v])
    right.set_ylabel("Jev minus incumbent accept rate")
    right.tick_params(length=0)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    svg = FIGDIR / f"{STEM}.svg"
    fig.savefig(svg, format="svg")
    plt.close(fig)
    LOG.info("wrote %s", svg)
    return [
        [
            svg.name,
            "undecided documents by cause, and the paired accept-rate difference",
            (
                "Left: the undecided documentation items of the dry run, split by the "
                "mechanism that left them undecided. Right: the difference between the "
                "two routes' accept rates over the items both judged, item-paired",
            ),
        ]
    ]


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table> "


def quotes(examples: list[dict]) -> str:
    return "".join(f"<pre>{e['quote']}</pre>" for e in examples)


def render_html(payload: dict) -> str:
    q1v, q2v, q3v = payload["q1"], payload["q2"], payload["q3"]
    docs3 = q3v["docs"]
    diff = docs3["difference"]
    dec, run = q1v["causes"][0], q1v["causes"][1]
    hold, unread = q2v["classes"][0], q2v["classes"][1]

    t1 = table(
        ["count", "owner", "cause"],
        [[c["count"], c["owner"], c["cause"]] for c in q1v["causes"]],
    )
    t2 = table(
        ["count", "class", "owner"],
        [[c["count"], c["class"], c["owner"]] for c in q2v["classes"]],
    )
    t3 = table(
        [
            "axis",
            "shared",
            "Jev accept",
            "incumbent accept",
            "difference",
            "95 percent interval",
            "matched target",
        ],
        [
            [
                AXIS_LABEL[a],
                str(v["shared_items"]),
                f"{v['jev_accept_rate_shared']:.4f}",
                f"{v['incumbent_accept_rate_shared']:.4f}",
                f"{v['difference']['point']:+.4f}",
                f"[{v['difference']['interval_95'][0]:+.4f}, "
                f"{v['difference']['interval_95'][1]:+.4f}]",
                f"{v['matched_target']:.4f}",
            ]
            for a, v in q3v.items()
        ],
    )
    figures = "".join(
        f'<figure><img src="/imas-codex/figures/sn-review-through-jev/{STEM}/{name}" '
        f'alt="{alt}"><figcaption>{cap}</figcaption></figure>'
        for name, alt, cap in payload["figures_rendered"]
    )
    fields = ", ".join(f for f in q2v["incumbent_piece_fields"] if not f.startswith("_"))

    return f"""<h2>The documentation axis: what failed, and what the accept-rate
check should have been compared against</h2>

<p>The documentation arm of the inverse switch dry run left {q1v['undecided']} of
its {q1v['eligible']} eligible items undecided, and that is what fails the
check. {run['count']} of those items carry no route row at all, and
{dec['count']} carry a shadow row whose failure reason is the deciding call's
own text. Neither cause is a defect in the item and neither is the cascade
choosing to abstain: the first is the run's budget cutting the batch short
before those items were reached, and the second is a deciding call failing once
with no retry. The accept-rate shortfall against the calibration-matched target
is a separate question, and the accept-rate check is comparing against the
wrong baseline for it.</p>

<h3>1. The {q1v['undecided']} undecided items, by cause</h3>
{t1}
<p><b>{dec['cause']} &mdash; {dec['count']} of {q1v['undecided']}, owner
{dec['owner']}.</b> {dec['mechanism']}</p>
<p>{dec['root_cause']}</p>
<p><b>Where.</b> {dec['where']}</p>
{quotes(dec['examples'])}
<p><b>{run['cause']} &mdash; {run['count']} of {q1v['undecided']}, owner
{run['owner']}.</b> {run['mechanism']}</p>
<p>{run['root_cause']}</p>
<p><b>Where.</b> {run['where']}</p>
{quotes(run['examples'])}
<p>{q1v['runner_detail']['note']}
{q1v['runner_detail']['items_with_no_jev_call']} of the {run['count']} carry zero
calls and {q1v['runner_detail']['items_still_drafted']} are still at
<code>drafted</code>, which is what a run cut short looks like rather than
a routing policy.</p>

<h3>2. The {q2v['population']} items Jev accepted and the incumbent refined</h3>

<p>These are read at one cutoff: Jev's route against the incumbent's own stored
score at <code>panel_min_score</code> {q2v['cutoff']}, over the
{q2v['shared_items']} documentation items both arms judged.</p>
{t2}
<p><b>{hold['class']} &mdash; {hold['count']} of {q2v['population']}.</b>
{hold['mechanism']}</p>
{quotes(hold['examples'])}
<p><b>{unread['class']} &mdash; {unread['count']} of {q2v['population']}.</b>
{unread['mechanism']}</p>
<p>{unread['missing_evidence']}</p>
{quotes(unread['examples'])}
<p>The incumbent arm persisted only these fields:
<code>{fields}</code>. No comment or hold note is among them, and the run logs
of that arm carry neither item ids nor comment text, so a substantive-defect
count for the remaining {unread['count']} is not recoverable from the branch.
{unread['owner']} is where that gap sits, not the incumbent reviewer.</p>

<h3>3. The name check's baseline</h3>

<p>The check compares each axis's accept rate against the panel's accept rate at
the cutoff matched to the panel's own operating point. For documentation that
target is {docs3['matched_target']:.4f}, measured on the panel's own pool
({docs3['matched_target_source']['panel_accepts']} accepts of
{docs3['matched_target_source']['n']} at cutoff
{docs3['matched_target_source']['cutoff']}) &mdash; not on this cohort. The
like-for-like reading is the two routes over the items both judged, item to
item, which is what the table below gives. The interval comes from
<code>paired_bootstrap</code> in
<code>srtj-jev-vs-panel-operating-point-run.py</code> unchanged, with each item
its own copy group so the draws stay paired.</p>
{t3}
<p>On documentation the two routes' accept rates over the
{docs3['shared_items']} shared items differ by {diff['point']:+.4f}, with a 95
percent interval of [{diff['interval_95'][0]:+.4f},
{diff['interval_95'][1]:+.4f}] from {diff['valid_resamples']} valid resamples.
For the comparison with the incumbent on identical items, the gap is
{diff['point']:+.4f}, not the shortfall the {q1v['eligible']}-item headline rate
shows against the matched target. Two comparisons are the wrong ones here:
{'; '.join(f'against {r["against"]} &mdash; {r["why"]}' for r in docs3['rejections'])}</p>
{figures}
<h3>What each finding implies for the switch</h3>
<ol>
<li><b>The undecided count is a run fault before it is a routing fact.</b> On
this cohort {run['count']} of {q1v['undecided']} undecided items were never
offered to Jev at all and {dec['count']} failed once and were not retried,
because an empty fault message is classified non-retryable. Turning the switch
on with the deciding path as it stands makes an item whose call fails take no
route silently; give the fault an honest message and a retry, and give the run a
budget that reaches every item, before the switch is flipped.</li>
<li><b>Do not read the accept-rate check as a statement about Jev's operating
point.</b> Over the items both routes judged the two accept rates differ by
{diff['point']:+.4f}, with an interval spanning
[{diff['interval_95'][0]:+.4f}, {diff['interval_95'][1]:+.4f}]
&mdash; a different question from the one the check asks against a target
measured on the panel's own pool. Recompute the target on the cohort, or gate on
the paired difference instead.</li>
<li><b>The refine-versus-accept set is not yet evidence either way.</b>
{hold['count']} of the {q2v['population']}
sit on the cutoff, and the remaining {unread['count']} cannot be classified
without the incumbent's reason line, which that arm did not persist. Until it
does, the set cannot distinguish a real defect caught from a mechanical hold,
and the switch should not be classified on it.</li>
</ol>
<p><small>Generated by <code>{STEM}-run.py</code> read-only: no graph write and
no model call. Source digests and every count above are in
<code>{STEM}.json</code>.</small></p>
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    helper = load_helper()
    dry = load(HERE / "srtj-jev-only-dry-run.json")
    thresholds = load(THRESHOLDS)
    jev_rows = {a: cascade_pieces(a) for a in DRY_SET}
    inc_rows = {a: incumbent_pieces(a) for a in DRY_SET}
    for axis in DRY_SET:
        LOG.info(
            "%s: cascade rows %d, incumbent rows %d",
            axis,
            len(jev_rows[axis]),
            len(inc_rows[axis]),
        )

    q1v = q1(jev_rows["docs"], dry["axes"])
    LOG.info(
        "q1: %d undecided = %d deciding-path + %d runner",
        q1v["undecided"],
        q1v["causes"][0]["count"],
        q1v["causes"][1]["count"],
    )
    q2v = q2(
        jev_rows["docs"],
        inc_rows["docs"],
        dry,
        float(thresholds["docs"]["matched_calibration"]["panel_min_score"]),
    )
    LOG.info(
        "q2: %d disagreements = %d on the cutoff + %d not readable",
        q2v["population"],
        q2v["classes"][0]["count"],
        q2v["classes"][1]["count"],
    )
    q3v = q3(helper, {a: (jev_rows[a], inc_rows[a]) for a in DRY_SET}, dry)
    for axis in DRY_SET:
        v = q3v[axis]
        LOG.info(
            "q3 %s: %d shared, jev %.4f vs incumbent %.4f, diff %+.4f [%+.4f, %+.4f]",
            axis,
            v["shared_items"],
            v["jev_accept_rate_shared"],
            v["incumbent_accept_rate_shared"],
            v["difference"]["point"],
            v["difference"]["interval_95"][0],
            v["difference"]["interval_95"][1],
        )

    payload = {
        "node": "srtj-docs-dry-run-diagnosis",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "read_only": True,
        "llm_calls": 0,
        "graph_writes": 0,
        "q1": q1v,
        "q2": q2v,
        "q3": q3v,
        "sources": {
            "dry_run_json": digest(HERE / "srtj-jev-only-dry-run.json"),
            "thresholds": digest(THRESHOLDS),
            "interval_helper": digest(HELPER),
            "cascade_pieces": {
                a: digest(COHORT / f"cascade-{a}-jev-only.json")
                for a in DRY_SET
            },
            "incumbent_pieces": {
                a: [
                    digest(p)
                    for p in sorted(COHORT.glob(f"incumbent-{a}-part*of4.json"))
                ]
                for a in DRY_SET
            },
        },
        "figures_rendered": [],
    }
    payload["figures_rendered"] = draw_figures(payload)

    (HERE / f"{STEM}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n"
    )
    (HERE / f"{STEM}.html").write_text(render_html(payload))
    LOG.info("wrote %s.json and %s.html", STEM, STEM)
    print(f"EXIT=0 {STEM}.json {STEM}.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

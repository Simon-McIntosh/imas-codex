"""Apply the registered anchor-adoption rule over the ten recorded anchor families.

Reads the per-family arm records written by the family runner (``<family>-heuristic.json``
and ``<family>-jev.json`` under ``ssog-anchor-ab/``) plus the fixed family selection, and
applies the rule registered before any apply:

    Adopt the Jev anchor only if its arm has fewer total lint findings after
    harmonization across the families and no member's documentation review score
    is lower than that same member's score in the heuristic arm.

No harmonize run and no model call happens here: every number below is read from a
record already on disk.

Completeness classes. ``outcome_status: complete`` in a record means only that the
scratch instance stayed contained; it says nothing about how far the harmonize apply
got. An arm is ``full`` when it ran every planned batch, ``partial`` when it halted
part way (carrying the halt reason), and ``incomplete`` when no completed record
exists at all. Two arms are compared on lint only when they stopped at the same
batch, so a family whose arms ran different batch counts is reported as not
like-for-like rather than scored.

Member scores are read from ``scores_after``; a receipt's ``member_scores`` carries a
member's pre-apply value. A null score is missing, never zero. Per-member score falls
within one arm are also recorded, and an arm's own anchor counts when its score fell.

Subcommands:
    build     write the JSON, the figure and the fragment
    selftest  drive the classifier and the rule over synthetic records whose
              verdicts are known, so a broken comparison shows red
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
RECORDS = HERE / "ssog-anchor-ab"

RESULTS = HERE / "ssog-anchor-ab-results.json"
FRAGMENT = HERE / "ssog-anchor-ab.html"
FIGURE_DIR = REPO / "docs/figures/sn-selection-over-generation/ssog-anchor-ab-aggregate"
FIGURE = FIGURE_DIR / "lint-before-after.svg"

ARMS = ("heuristic", "jev")
NODE = "ssog-anchor-ab-aggregate"
EARLIER_ID = "ssog-anchor-ab-earlier-blocked"
CEILING_USD = 50.0

# Colour names the arm; the two states of one arm are lightness steps of its hue.
COLOURS = {
    ("heuristic", "before"): "#a9c9cd",
    ("heuristic", "after"): "#246c78",
    ("jev", "before"): "#e2c39a",
    ("jev", "after"): "#b06a12",
}
BAR_H = 0.18


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _classify(record: dict) -> dict:
    """Score one arm record's completeness, its lint and its per-member score moves."""
    summary = record.get("harmonize_summary") or {}
    ran = summary.get("batches_run")
    total = summary.get("batches_total")
    after = record.get("scores_after") or {}
    before = record.get("scores_before") or {}
    members = record.get("members") or []

    if record.get("lint_after") is None or not summary:
        completeness = "incomplete"
    elif ran == total:
        completeness = "full"
    else:
        completeness = "partial"

    falls = [
        {"member": member, "before": before.get(member), "after": after.get(member)}
        for member in members
        if isinstance(before.get(member), (int, float))
        and isinstance(after.get(member), (int, float))
        and after.get(member) < before.get(member)
    ]

    receipts = record.get("harmonize_receipts") or []
    receipt_total = (
        receipts[0]["result"].get("total_cost")
        if receipts and "result" in receipts[0]
        else None
    )
    anchor = record.get("anchor")

    return {
        "family": record.get("family"),
        "arm": record.get("arm"),
        "anchor": anchor,
        "heuristic_anchor": record.get("heuristic_anchor"),
        "jev_anchor": record.get("jev_anchor"),
        "kind": record.get("kind"),
        "members": members,
        "completeness": completeness,
        "batches_run": ran,
        "batches_total": total,
        "halt_reasons": summary.get("halt_reasons") or [],
        "outcome_status": record.get("outcome_status"),
        "applied": record.get("applied"),
        "stamped": bool(record.get("stamped")),
        "containment_holds": (record.get("containment") or {}).get("holds"),
        "lint_before": record.get("lint_before"),
        "lint_after": record.get("lint_after"),
        "lint_before_classes": record.get("lint_before_classes") or {},
        "lint_after_classes": record.get("lint_after_classes") or {},
        "scores_before": {m: before.get(m) for m in members},
        "scores_after": {m: after.get(m) for m in members},
        "score_falls": falls,
        "anchor_score_fell": any(f["member"] == anchor for f in falls),
        "calls": record.get("calls"),
        "planned_calls": record.get("planned_calls"),
        "ledger_calls": record.get("ledger_calls"),
        "spend_usd_ledger": record.get("ledger_spend_usd"),
        "spend_usd_receipt": receipt_total,
        "wall_seconds": record.get("wall_seconds"),
        "run_ids": record.get("run_ids"),
        "attempt_outcome": record.get("attempt_outcome"),
    }


def _score_check(heuristic: dict, jev: dict) -> dict:
    """Per member, is any Jev-arm after-score below the heuristic arm's for that member?"""
    lower = []
    for member in heuristic["members"]:
        h = heuristic["scores_after"].get(member)
        j = jev["scores_after"].get(member)
        if isinstance(h, (int, float)) and isinstance(j, (int, float)) and j < h:
            lower.append({"member": member, "heuristic": h, "jev": j})
    return {
        "admissible": (
            heuristic["completeness"] == "full" and jev["completeness"] == "full"
        ),
        "members_lower_in_jev": lower,
        "passes": not lower,
    }


def _family_verdict(heuristic: dict, jev: dict) -> dict:
    both_measured = (
        heuristic["lint_after"] is not None and jev["lint_after"] is not None
    )
    like_for_like = both_measured and heuristic["batches_run"] == jev["batches_run"]
    scores = _score_check(heuristic, jev)

    if not both_measured:
        verdict = "incomparable"
        reason = "one arm has no completed record, so no lint comparison can be formed"
    elif not like_for_like:
        verdict = "not-like-for-like"
        reason = (
            f"arms stopped at different batches (heuristic {heuristic['batches_run']}/"
            f"{heuristic['batches_total']}, Jev {jev['batches_run']}/{jev['batches_total']}); "
            "the lint difference tracks the batch count, not the anchor"
        )
    elif jev["lint_after"] < heuristic["lint_after"]:
        verdict, reason = "jev-fewer", "the Jev arm ends with fewer lint findings"
    elif jev["lint_after"] > heuristic["lint_after"]:
        verdict, reason = "heuristic-fewer", "the Jev arm ends with more lint findings"
    else:
        verdict, reason = "equal", "both arms end with the same lint count"

    return {
        "verdict": verdict,
        "reason": reason,
        "like_for_like": like_for_like,
        "lint_advantage_jev": (
            heuristic["lint_after"] - jev["lint_after"] if both_measured else None
        ),
        "score_check": scores,
    }


def _totals(rows: list[dict]) -> dict:
    both = [
        (r["heuristic"], r["jev"])
        for r in rows
        if r["heuristic"]["lint_after"] is not None
        and r["jev"]["lint_after"] is not None
    ]
    lfl = [(h, j) for h, j in both if h["batches_run"] == j["batches_run"]]
    anomalous = "current_density_due_to_anomalous_transport"
    lfl_clean = [(h, j) for h, j in lfl if h["family"] != anomalous]

    def count(predicate) -> int:
        return sum(1 for r in rows if predicate(r["verdict"]["verdict"]))

    return {
        "families_recorded": len(rows),
        "families_with_both_arms_measured": len(both),
        "lint_after_heuristic_both_measured": sum(h["lint_after"] for h, _ in both),
        "lint_after_jev_both_measured": sum(j["lint_after"] for _, j in both),
        "families_like_for_like": len(lfl),
        "lint_after_heuristic_like_for_like": sum(h["lint_after"] for h, _ in lfl),
        "lint_after_jev_like_for_like": sum(j["lint_after"] for _, j in lfl),
        "families_like_for_like_excluding_worsened": len(lfl_clean),
        "lint_after_heuristic_like_for_like_excluding_worsened": sum(
            h["lint_after"] for h, _ in lfl_clean
        ),
        "lint_after_jev_like_for_like_excluding_worsened": sum(
            j["lint_after"] for _, j in lfl_clean
        ),
        "families_equal": count(lambda v: v == "equal"),
        "families_jev_fewer": count(lambda v: v == "jev-fewer"),
        "families_heuristic_fewer": count(lambda v: v == "heuristic-fewer"),
        "families_not_like_for_like": count(lambda v: v == "not-like-for-like"),
        "families_incomparable": count(lambda v: v == "incomparable"),
        "families_with_a_lower_jev_member_score": sum(
            1 for r in rows if not r["verdict"]["score_check"]["passes"]
        ),
        "families_both_full_for_scores": sum(
            1 for r in rows if r["verdict"]["score_check"]["admissible"]
        ),
    }


def _prior_cut_usd() -> float | None:
    """Confirmed spend the earlier cut of this A/B booked before this node ran.

    Read once, from the results file the earlier cut left at this path, and then
    carried forward inside this node's own payload so a re-run still finds it.
    """
    if not RESULTS.exists():
        return None
    try:
        existing = _load(RESULTS)
    except json.JSONDecodeError:
        return None
    if isinstance(existing.get("confirmed_spend_usd"), (int, float)):
        return float(existing["confirmed_spend_usd"])
    prior = (existing.get("spend") or {}).get("prior_cut_usd")
    return float(prior) if isinstance(prior, (int, float)) else None


def build() -> dict:
    selection = _load(RECORDS / "selection.json")
    selected = selection["selected"]
    prior_cut_usd = _prior_cut_usd()

    rows = []
    for entry in selected:
        family = entry["seed"]
        row = {
            "family_seed": family,
            "kind": entry["kind"],
            "physical_base": entry["physical_base"],
        }
        for arm in ARMS:
            row[arm] = _classify(_load(RECORDS / f"{family}-{arm}.json"))
        row["verdict"] = _family_verdict(row["heuristic"], row["jev"])
        rows.append(row)

    totals = _totals(rows)

    ledger = sum((r[arm]["spend_usd_ledger"] or 0.0) for r in rows for arm in ARMS)
    halt_cost = 0.0
    for r in rows:
        text = r["heuristic"].get("attempt_outcome") or ""
        match = re.search(r"cost_spent=([0-9.]+)", text)
        if match:
            halt_cost = float(match.group(1))
    scorer = sum(entry["jev_cost"] for entry in selected)
    arms_and_scorer = ledger + halt_cost + scorer
    spend = {
        "arms_ledger_usd": round(ledger, 6),
        "wall_recombination_heuristic_halt_usd": halt_cost,
        "scorer_usd": round(scorer, 6),
        "prior_cut_usd": round(prior_cut_usd, 6) if prior_cut_usd is not None else None,
        "total_usd": round(arms_and_scorer + (prior_cut_usd or 0.0), 6),
        "ceiling_usd": CEILING_USD,
        "note": (
            "Per-arm ledger figures are the LLMCost window totals and are what the sums use. "
            "The harmonize receipts carry a CLI total as well; the two agree everywhere except "
            "the visible-wavelength Jev arm, whose receipt reads USD 1.0162 against its ledger's "
            "USD 0.841339 (the receipt spans both batches while the ledger window closed after "
            "the second). The wall-recombination heuristic arm has no ledger total because its "
            "apply died on its own cap; its halt reason records cost_spent USD 1.50578, which is "
            "the figure used here. The prior-cut figure is the confirmed spend the first cut of "
            "this A/B booked before it stopped on its worker deadline."
        ),
    }

    lint_ok_total = (
        totals["lint_after_jev_both_measured"]
        < totals["lint_after_heuristic_both_measured"]
    )
    like_for_like_ok = (
        totals["lint_after_jev_like_for_like"]
        < totals["lint_after_heuristic_like_for_like"]
    )
    score_ok = totals["families_with_a_lower_jev_member_score"] == 0

    overall = {
        "adopt_jev": False,
        "condition_fewer_total_lint_across_families": {
            "met_on_both_measured_subset": lint_ok_total,
            "met_like_for_like": like_for_like_ok,
            "detail": (
                f"over the {totals['families_with_both_arms_measured']} families where both arms "
                f"completed, heuristic {totals['lint_after_heuristic_both_measured']} against Jev "
                f"{totals['lint_after_jev_both_measured']}, but the tenth family "
                "(energy_flux_at_wall_due_to_recombination) has no heuristic measurement, so the "
                "ten-family total the rule names cannot be formed; over the "
                f"{totals['families_like_for_like']} families whose arms reached the same batch "
                f"the totals are equal at {totals['lint_after_heuristic_like_for_like']}"
            ),
        },
        "condition_no_lower_member_scores": {
            "met": score_ok,
            "detail": (
                "a Jev-arm member scores below the same member's heuristic-arm value in "
                f"{totals['families_with_a_lower_jev_member_score']} of "
                f"{totals['families_recorded']} families"
            ),
        },
        "verdict": "do not adopt the Jev anchor",
        "reason": (
            "the second condition fails outright on complete, like-for-like evidence - every "
            "family carrying after-scores in both arms has at least one member scoring lower "
            "under the Jev anchor - and the first is not established, since on the families "
            "where both arms reached the same batch the lint counts are identical and the "
            "tenth family has no heuristic measurement"
        ),
    }

    recommendation = (
        "Keep the heuristic anchor rather than adopting the Jev scorer: the Jev arm fails the "
        "rule's score condition in nine of ten families and shows no like-for-like lint "
        "advantage, so the evidence that would change the call is a re-run holding both arms to "
        "the same batch count (or forcing both to a full two of two under a raised per-arm "
        "budget) in which the Jev arm ends with strictly fewer lint findings and no member's "
        "after-score falls below that member's heuristic-arm value."
    )

    return {
        "node": NODE,
        "adoption_rule": selection["adoption_rule"],
        "source_records": "docs/evidence/fragments/sn-selection-over-generation/ssog-anchor-ab/",
        "source_export": selection.get("source_export"),
        "arm_records_read": len(rows) * 2,
        "rows": [
            {"family_seed": r["family_seed"], "kind": r["kind"], "arm": arm, **r[arm]}
            for r in rows
            for arm in ARMS
        ],
        "families": [
            {
                "family_seed": r["family_seed"],
                "kind": r["kind"],
                "physical_base": r["physical_base"],
                "heuristic_anchor": r["heuristic"]["heuristic_anchor"],
                "jev_anchor": r["jev"]["jev_anchor"],
                "heuristic": r["heuristic"],
                "jev": r["jev"],
                "verdict": r["verdict"],
            }
            for r in rows
        ],
        "totals": totals,
        "spend": spend,
        "overall_verdict": overall,
        "recommendation": recommendation,
    }


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def _check_text(fig) -> None:
    """Refuse to write a figure whose text leaves the canvas or collides with other text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    boxes = []
    seen = set()
    seen_text = set()
    for artist in fig.findobj():
        if not hasattr(artist, "get_window_extent") or not hasattr(artist, "get_text"):
            continue
        if not artist.get_text().strip() or id(artist) in seen:
            continue
        seen.add(id(artist))
        box = artist.get_window_extent(renderer)
        # Matplotlib carries an unlaid-out twin of each tick label at the origin;
        # a sub-pixel box is that twin, not text a reader can see.
        if (box.x1 - box.x0) <= 1 or (box.y1 - box.y0) <= 1:
            continue
        key = (
            artist.get_text(),
            round(box.x0),
            round(box.y0),
            round(box.x1),
            round(box.y1),
        )
        # The same string at the same place is one visible label found twice.
        if key in seen_text:
            continue
        seen_text.add(key)
        boxes.append((artist.get_text(), box))
    problems = []
    for text, box in boxes:
        if box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1:
            problems.append(f"clipped at the canvas edge: {text!r}")
        if box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
            problems.append(f"clipped at the canvas edge: {text!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                problems.append(
                    f"overlapping text: {boxes[i][0]!r} and {boxes[j][0]!r}"
                )
    if problems:
        raise SystemExit("figure text check failed:\n  " + "\n  ".join(problems))


def figure(payload: dict, render_check: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.style.use("data-ink")

    families = payload["families"]
    labels = [f["family_seed"].replace("x_", "x ").replace("_", " ") for f in families]
    ys = list(range(len(families)))

    fig, ax = plt.subplots(figsize=(14, 8))
    for arm, offsets in (("heuristic", (-1.5, -0.5)), ("jev", (0.5, 1.5))):
        for state, offset in zip(("before", "after"), offsets, strict=True):
            centres = [y + offset * BAR_H for y in ys]
            values = [
                f[arm]["lint_before"] if state == "before" else f[arm]["lint_after"]
                for f in families
            ]
            ax.barh(
                centres,
                [v if v is not None else 0.0 for v in values],
                height=BAR_H * 0.9,
                color=COLOURS[(arm, state)],
                zorder=3,
            )
            for centre, value in zip(centres, values, strict=True):
                if value is None:
                    ax.plot(
                        [0.18],
                        [centre],
                        marker="x",
                        color="#929292",
                        markersize=9,
                        zorder=4,
                    )
                    ax.text(
                        0.55,
                        centre,
                        "no completed heuristic arm",
                        va="center",
                        fontsize=12,
                        color="#52514e",
                    )
                    continue
                if value == 0:
                    # A zero drawn as a five-hundredths sliver would misstate the
                    # count; a mark on the axis keeps the row readable as a
                    # measured zero rather than a missing state.
                    ax.plot(
                        [0.0],
                        [centre],
                        marker="o",
                        markersize=4,
                        color=COLOURS[(arm, state)],
                        zorder=5,
                    )
                    continue

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("lint findings [count]")
    ax.set_ylabel("family")
    ax.set_xlim(0, 14.5)
    ax.set_xticks(range(0, 15, 2))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        Patch(facecolor=COLOURS[(arm, state)])
        for arm in ARMS
        for state in ("before", "after")
    ]
    fig.legend(
        handles,
        ["heuristic before", "heuristic after", "Jev before", "Jev after"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _check_text(fig)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # A fixed salt keeps matplotlib's generated SVG element ids stable across
    # runs, so a re-run rewrites identical bytes instead of churning the diff.
    matplotlib.rcParams["svg.hashsalt"] = NODE
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    if render_check is not None:
        render_check.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(render_check, format="png", dpi=100, metadata={"Date": None})
    plt.close(fig)


# --------------------------------------------------------------------------
# Fragment
# --------------------------------------------------------------------------


def _cell(value) -> str:
    return "missing" if value is None else str(value)


def _table(payload: dict) -> str:
    head = (
        "<thead><tr>"
        "<th>Family seed</th><th>Arm</th><th>Anchor</th><th>Completeness</th>"
        "<th>Batches</th><th>Lint before &rarr; after</th>"
        "<th>Member scores after</th><th>Calls</th><th>Spend ledger [USD]</th>"
        "<th>Stamped</th></tr></thead>"
    )
    body = []
    for f in payload["families"]:
        for arm in ARMS:
            rec = f[arm]
            classes = ", ".join(
                f"{k} {v}" for k, v in sorted((rec["lint_after_classes"] or {}).items())
            )
            score_txt = "; ".join(
                f"{m} {_cell(rec['scores_after'].get(m))}" for m in rec["members"]
            )
            completeness = rec["completeness"]
            if completeness == "incomplete" and rec["attempt_outcome"]:
                completeness += " (stopped on budget, exit 1)"
            elif completeness == "partial" and rec["halt_reasons"]:
                completeness += f" ({rec['halt_reasons'][0]})"
            lint_txt = (
                f"{_cell(rec['lint_before'])} &rarr; {_cell(rec['lint_after'])}"
                + (f" ({classes})" if classes else "")
            )
            body.append(
                "<tr>"
                f"<td>{f['family_seed']}</td><td>{arm}</td>"
                f"<td><code>{rec['anchor']}</code></td>"
                f"<td>{completeness}</td>"
                f"<td>{_cell(rec['batches_run'])}/{_cell(rec['batches_total'])}</td>"
                f"<td>{lint_txt}</td>"
                f"<td>{score_txt}</td>"
                f"<td>{_cell(rec['calls'])}</td>"
                f"<td>{_cell(rec['spend_usd_ledger'])}</td>"
                f"<td>{'yes' if rec['stamped'] else 'no'}</td>"
                "</tr>"
            )
    return (
        f'<table id="{NODE}-table">{head}<tbody>' + "".join(body) + "</tbody></table>"
    )


def _earlier_body(existing: str) -> str:
    """Keep the earlier record the fragment already holds, under its own heading."""
    if f'id="{EARLIER_ID}"' in existing:
        match = re.search(rf'<section id="{EARLIER_ID}".*?</section>', existing, re.S)
        return match.group(0) if match else ""
    inner = re.sub(r"^\s*<section[^>]*>", "", existing, count=1)
    inner = re.sub(r"</section>\s*$", "", inner).strip()
    return (
        f'<section id="{EARLIER_ID}" data-reckon="evidence">\n'
        f'  <h3 id="{EARLIER_ID}-heading">Earlier record: the first cut of this A/B, '
        "blocked at the fixed worker deadline</h3>\n"
        f"  {inner}\n</section>".replace("\n  ", "\n")
    )


def fragment(payload: dict) -> None:
    totals = payload["totals"]
    spend = payload["spend"]

    incomplete = [
        f["family_seed"]
        for f in payload["families"]
        if f["heuristic"]["lint_after"] is None
    ]
    nll = [
        f for f in payload["families"] if f["verdict"]["verdict"] == "not-like-for-like"
    ]
    nll_txt = "; ".join(
        f"<code>{f['family_seed']}</code> (heuristic {f['heuristic']['batches_run']} of "
        f"{f['heuristic']['batches_total']} batches against Jev {f['jev']['batches_run']} of "
        f"{f['jev']['batches_total']})"
        for f in nll
    )
    equal_lfl = [
        f["family_seed"]
        for f in payload["families"]
        if f["verdict"]["verdict"] == "equal"
    ]

    parts = [
        f'<section id="{NODE}" data-reckon="evidence">',
        f'  <h2 id="{NODE}-heading">Anchor A/B over ten families: the Jev anchor is not adopted</h2>',
        f'  <p id="{NODE}-verdict"><strong>Verdict: do not adopt the Jev anchor.</strong> '
        f"The rule registered at <code>{payload['adoption_rule']['registered_at']}</code> "
        "requires the Jev arm to show fewer lint findings across the families with no member's "
        "documentation score lower than that same member's in the heuristic arm. "
        f"<strong>The score condition fails in "
        f"{totals['families_with_a_lower_jev_member_score']} of {totals['families_recorded']} "
        "families</strong> — every family that carries after-scores in both arms has at least one "
        "member scoring lower under the Jev anchor. <strong>The lint condition is not "
        "established:</strong> over the "
        f"{totals['families_like_for_like']} families whose arms reached the same batch the lint "
        f"totals are identical ({totals['lint_after_heuristic_like_for_like']} each), and the "
        f"tenth family ("
        f"<code>{incomplete[0] if incomplete else 'none'}</code>) has no heuristic measurement at "
        "all, so the ten-family total the rule names cannot be formed. Neither condition is "
        "satisfied, so the anchor reverts to the heuristic.</p>",
        f'  <h3 id="{NODE}-totals">What the rule reads</h3>',
        f'  <p id="{NODE}-totals-body">Across the '
        f"{totals['families_with_both_arms_measured']} families where both arms completed, lint "
        f"after harmonization is <strong>heuristic "
        f"{totals['lint_after_heuristic_both_measured']} against Jev "
        f"{totals['lint_after_jev_both_measured']}</strong>. Across the "
        f"{totals['families_like_for_like']} families whose arms stopped at the same batch it is "
        f"<strong>{totals['lint_after_heuristic_like_for_like']} against "
        f"{totals['lint_after_jev_like_for_like']}</strong>, and excluding the anomalous-transport "
        f"family, where harmonization made both arms worse, it is "
        f"{totals['lint_after_heuristic_like_for_like_excluding_worsened']} against "
        f"{totals['lint_after_jev_like_for_like_excluding_worsened']}. Family by family: "
        f"{totals['families_equal']} equal ({', '.join(equal_lfl)}), "
        f"{totals['families_jev_fewer']} with the Jev arm fewer, "
        f"{totals['families_heuristic_fewer']} with the heuristic arm fewer, "
        f"{totals['families_not_like_for_like']} not like-for-like and "
        f"{totals['families_incomparable']} incomparable.</p>",
        f'  <p id="{NODE}-anomalous"><strong>current_density_due_to_anomalous_transport is worse '
        "under both anchors.</strong> Both arms begin at 5 findings and end at 12, identically "
        "for both anchors and stamped in neither, and their halt reasons agree "
        "(<code>docs acceptance 80.00% &lt; 90.00% (4/5 touched)</code>). The rise is a property "
        "of harmonization on this family and not of the anchor chosen, so it is reported as "
        "harmonization making the family worse under either anchor.</p>",
        f'  <h3 id="{NODE}-table-heading">One row per family and arm</h3>',
        f"  {_table(payload)}",
        f'  <h3 id="{NODE}-partials">What the partial applies mean for the verdict</h3>',
        f'  <p id="{NODE}-partials-body"><code>outcome_status: complete</code> on nearly every '
        "record means only that the containment check held — the scratch instance stayed "
        "contained and the live graph saw nothing. It says nothing about how far the apply "
        "reached. Most arms halted after batch 1 of 2, several at 0% documentation acceptance, "
        f"which is why only {totals['families_equal'] + totals['families_jev_fewer'] + totals['families_heuristic_fewer']} "
        "families compare their arms at the same batch and just two arms are stamped at all. "
        f"<strong>Not like for like:</strong> {nll_txt}. In each of these the arm that ran the "
        "second batch ends with the lower count, which is the batch count rather than the "
        "anchor — the visible-wavelength Jev arm ran two batches where the heuristic arm ran one, "
        "and the passive-loop heuristic arm ran two where the Jev arm ran one. Where the arms did "
        "stop together the counts are identical. <strong>Incomplete:</strong> the "
        f"wall-recombination heuristic arm stopped on its internal budget at USD "
        f"{spend['wall_recombination_heuristic_halt_usd']} before writing any completed record, "
        "so it is recorded as heuristic-incomplete and never as a Jev win by default. "
        "<strong>Consequence for the verdict:</strong> the lint half of the rule is left unable "
        "to discriminate by the partial applies, so the verdict rests on the score condition, "
        "which fails on complete, like-for-like evidence in nine families. A re-run holding both "
        "arms to the same batch count is what would give the lint half a reading.</p>",
        f'  <p id="{NODE}-spend">Ledger spend across the twenty arm records read here is '
        f"<strong>USD {spend['arms_ledger_usd']}</strong>, plus USD "
        f"{spend['wall_recombination_heuristic_halt_usd']} recorded as the wall-recombination "
        f"heuristic arm's halt cost and USD {spend['scorer_usd']} for the Jev anchor scorer, "
        f"giving <strong>USD "
        f"{round(spend['arms_ledger_usd'] + spend['wall_recombination_heuristic_halt_usd'] + spend['scorer_usd'], 6)}"
        f" for this cut</strong> and <strong>USD {spend['total_usd']} in total</strong> once the "
        f"USD {spend['prior_cut_usd']} the first cut booked is included, against the USD "
        f"{spend['ceiling_usd']:.0f} ceiling. {spend['note']}</p>",
        f'  <figure id="{NODE}-figure"><img src="/imas-codex/figures/sn-selection-over-generation/'
        f'{NODE}/lint-before-after.svg" alt="Per family, lint findings before and after '
        'harmonization for the heuristic and Jev anchor arms, as four bars per family row.">'
        "<figcaption>Lint findings before and after harmonization, one row per family, four bars "
        "per row: the heuristic arm's before and after, then the Jev arm's. A grey cross marks a "
        "state no completed record holds, on the wall-recombination heuristic arm, which stopped "
        "on its budget before an after-count was taken, and a small round mark on the axis is a "
        "measured zero rather than a missing state. Anomalous transport rises from 5 to 12 "
        "in both arms; where the arms stopped at the same batch they end level, and the two rows "
        "whose bars differ are the two whose arms ran different batch counts.</figcaption>"
        "</figure>",
        f'  <p id="{NODE}-recommendation"><strong>Recommendation for the lead.</strong> '
        f"{payload['recommendation']}</p>",
        _earlier_body(FRAGMENT.read_text() if FRAGMENT.exists() else ""),
        "</section>",
    ]
    FRAGMENT.write_text("\n".join(parts) + "\n")


# --------------------------------------------------------------------------
# Self-test: drive the classifier and the rule over records whose verdicts are known
# --------------------------------------------------------------------------


def _synthetic(
    lint_after, batches=None, scores_after=None, members=("m1", "m2")
) -> dict:
    record = {
        "family": "synthetic",
        "arm": "x",
        "anchor": "a",
        "members": list(members),
        "harmonize_summary": {"batches_run": batches, "batches_total": 2}
        if batches
        else None,
        "lint_before": 1,
        "lint_after": lint_after,
        "scores_before": dict.fromkeys(members, 0.9),
        "scores_after": scores_after or dict.fromkeys(members, 0.95),
        "harmonize_receipts": [],
        "containment": {"holds": True},
    }
    return _classify(record)


def selftest() -> int:
    failures = []
    checks = 0

    def check(name, got, want):
        nonlocal checks
        checks += 1
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    same = {"m1": 0.95, "m2": 0.95}

    # Completeness classification, including the record whose apply died.
    check(
        "class-full",
        _synthetic(0, batches=2, scores_after=same)["completeness"],
        "full",
    )
    check(
        "class-partial",
        _synthetic(1, batches=1, scores_after=same)["completeness"],
        "partial",
    )
    check(
        "class-incomplete", _synthetic(None, batches=None)["completeness"], "incomplete"
    )

    # The rule's lint half.
    verdict = _family_verdict(
        _synthetic(0, batches=2, scores_after=same),
        _synthetic(3, batches=2, scores_after=same),
    )
    check("heuristic-fewerh", verdict["verdict"], "heuristic-fewer")

    verdict = _family_verdict(
        _synthetic(4, batches=2, scores_after=same),
        _synthetic(1, batches=2, scores_after=same),
    )
    check("jev-fewer", verdict["verdict"], "jev-fewer")
    check("jev-fewer-scores-pass", verdict["score_check"]["passes"], True)

    # The guarded thing made to happen: a Jev member lower must refuse.
    verdict = _family_verdict(
        _synthetic(4, batches=2, scores_after=same),
        _synthetic(1, batches=2, scores_after={"m1": 0.5, "m2": 0.95}),
    )
    check("score-refusal", verdict["score_check"]["passes"], False)
    check(
        "score-refusal-names-member",
        verdict["score_check"]["members_lower_in_jev"],
        [{"member": "m1", "heuristic": 0.95, "jev": 0.5}],
    )

    # A null score is missing, never lower.
    verdict = _family_verdict(
        _synthetic(4, batches=2, scores_after=same),
        _synthetic(1, batches=2, scores_after={"m1": None, "m2": 0.95}),
    )
    check("null-not-lower", verdict["score_check"]["passes"], True)

    # Different batches is not like for like, however the lint reads.
    verdict = _family_verdict(
        _synthetic(7, batches=1, scores_after=same),
        _synthetic(2, batches=2, scores_after=same),
    )
    check("not-like-for-like", verdict["verdict"], "not-like-for-like")

    # An arm with no completed record is incomparable.
    verdict = _family_verdict(
        _synthetic(None, batches=None), _synthetic(2, batches=1, scores_after=same)
    )
    check("incomparable", verdict["verdict"], "incomparable")

    # A fall in an arm's own anchor is recorded.
    record = _synthetic(1, batches=1, scores_after={"m1": 0.4, "m2": 0.95})
    check(
        "score-fall-recorded",
        record["score_falls"],
        [{"member": "m1", "before": 0.9, "after": 0.4}],
    )

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print(f"selftest: {checks} checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="build", choices=("build", "selftest", "figure")
    )
    parser.add_argument(
        "--render-check",
        metavar="PATH",
        help="also render the figure to a PNG at PATH, for inspection",
    )
    args = parser.parse_args()

    if args.command == "selftest":
        return selftest()

    payload = build()
    if args.command == "build":
        RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {RESULTS.relative_to(REPO)}")
    figure(payload, Path(args.render_check) if args.render_check else None)
    print(f"wrote {FIGURE.relative_to(REPO)}")
    if args.command == "build":
        fragment(payload)
        print(f"wrote {FRAGMENT.relative_to(REPO)}")
    print(f"overall verdict: {payload['overall_verdict']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

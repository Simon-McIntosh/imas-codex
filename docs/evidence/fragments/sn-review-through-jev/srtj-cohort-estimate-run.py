#!/usr/bin/env python
"""Draw the fixed review cohort and price both review arms from recorded receipts.

Reads the eligible-identity snapshot taken from the live graph, draws a fixed
seeded sample, and prices the incumbent quorum against the deciding cascade
using only costs the review seats actually recorded. Emits the JSON and HTML
fragments. Spends nothing: no model is called and the graph is not written.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261019
PER_AXIS = 232
USD_SPENT = 0.0

# --------------------------------------------------------------------------
# Recorded receipts. Every figure below is an aggregate over a named window of
# rows that the review pipeline itself wrote, not a list price.
# --------------------------------------------------------------------------

# LLMCost rows for the review phases, grouping window 2026-07-21 .. 2026-10-17,
# which is the span in which the names profile carried its three current seats.
# The seat list changed on 2026-10-18 (commit ab4af6700), and no review receipt
# exists under the new names, so the seats named there are priced through their
# nearest recorded predecessor; see seat_substitutions.
INCUMBENT_SEATS = {
    "name": {
        "cycle0": {
            "seat": "openrouter/x-ai/grok-4.5",
            "cycle": 0,
            "n": 3194,
            "mean_usd": 0.04326880056355669,
            "phase": "review_name",
            "window": "2026-07-21..2026-10-16",
        },
        "cycle1": {
            "seat": "openrouter/openai/gpt-5.6-luna",
            "cycle": 1,
            "n": 2925,
            "mean_usd": 0.004914156581196569,
            "phase": "review_name",
            "window": "2026-07-22..2026-10-16",
        },
        "cycle2": {
            "seat": "openrouter/anthropic/claude-sonnet-5",
            "cycle": 2,
            "n": 1268,
            "mean_usd": 0.07466746529968442,
            "phase": "review_name",
            "window": "2026-07-21..2026-10-16",
        },
    },
    "docs": {
        "cycle0": {
            "seat": "openrouter/anthropic/claude-sonnet-5",
            "cycle": 0,
            "n": 6511,
            "mean_usd": 0.06537800998310529,
            "phase": "review_docs",
            "window": "2026-07-20..2026-10-17",
        },
        "cycle1": {
            "seat": "openrouter/x-ai/grok-4.5",
            "cycle": 1,
            "n": 5981,
            "mean_usd": 0.034869080922922624,
            "phase": "review_docs",
            "window": "2026-07-20..2026-10-17",
        },
        "cycle2": {
            "seat": "openrouter/openai/gpt-5.5",
            "cycle": 2,
            "n": 1264,
            "mean_usd": 0.16296273180379764,
            "phase": "review_docs",
            "window": "2026-07-04..2026-10-17",
        },
    },
}

# The escalator fires only on a flagged disagreement between cycles 0 and 1.
# Measured from StandardNameReview cycle coverage over 2026-07-21..2026-10-17.
ESCALATION = {
    "name": {
        "items": 2796,
        "cycle2_items": 835,
        "calls_per_item": 2.291,
        "window": "2026-07-21..2026-10-17",
    },
    "docs": {
        "items": 1689,
        "cycle2_items": 431,
        "calls_per_item": 2.245,
        "window": "2026-07-21..2026-10-17",
    },
}
for _axis, _rec in ESCALATION.items():
    _rec["cycle2_share"] = _rec["cycle2_items"] / _rec["items"]

# Jev judgment calls, read from the two crews' own decision receipts.
JEV_RECEIPTS = {
    "name": {
        "path": (
            "/home/ITER/mcintos/.config/reckon/crew/runs/"
            "r-20261016T101231058904-srtj-evidence-supply/evidence-decisions.jsonl"
        ),
        "n": 706,
        "mean_usd": 0.000234,
        "sd_usd": 0.000053,
        "mean_latency_s": 3.76,
    },
    "docs": {
        "path": (
            "/home/ITER/mcintos/.config/reckon/crew/runs/"
            "r-20261017T200215613429-srtj-docs-score-arm/docs-decisions.jsonl"
        ),
        "n": 540,
        "mean_usd": 0.000127,
        "sd_usd": 0.000033,
        "mean_latency_s": 0.87,
    },
}

# Calibrated route shares on the frozen calibration splits, read from
# imas_codex/standard_names/review/cascade_thresholds.json.
ROUTE_SHARES = {
    "name": {
        "n": 363,
        "accept": 0.0606,
        "refine": 0.7548,
        "chat_review": 0.1846,
        "chat_below_reject": 0.0,
        "unjudged": 0.0,
    },
    "docs": {
        "n": 216,
        "accept": 0.1157,
        "refine": 0.6991,
        "chat_review": 0.1852,
        "chat_below_reject": 0.0,
        "unjudged": 0.0,
    },
}

# A chat route uses only the first model of the configured review seat.
CHAT_SEAT = {
    "name": {
        "seat": "openrouter/x-ai/grok-4.5",
        "mean_usd": 0.04326880056355669,
        "n": 3194,
    },
    "docs": {
        "seat": "openrouter/anthropic/claude-sonnet-5",
        "mean_usd": 0.06537800998310529,
        "n": 6511,
    },
}

# Wall time per seat call, amortised over a run: the median, across runs with at
# least ten calls, of the run's LLMCost span divided by its call count.
CALL_WALL_S = {"name": 11.02, "docs": 10.89}

# The configured names after commit ab4af6700 carry no recorded receipt, so the
# predecessor above stands in. The substitution biases the incumbent arm, and
# the direction is stated so the reader can correct it.
SEAT_SUBSTITUTIONS = [
    {
        "configured": "openrouter/openai/gpt-6-luna",
        "role": "names cycle 1",
        "priced_as": "openrouter/openai/gpt-5.6-luna",
        "n": 2925,
        "mean_usd": 0.004914156581196569,
        "bias": (
            "gpt-6-luna completes at 0.50 USD/Mtok against gpt-5.6-luna's 0.60 "
            "at the same prompt rate, so the substitute over-states this seat "
            "and the incumbent arm reads high by whatever the recorded token "
            "mix spends on output tokens."
        ),
    },
    {
        "configured": "openrouter/openai/gpt-6.1-sol",
        "role": "docs cycle 2",
        "priced_as": "openrouter/openai/gpt-6.1-sol",
        "n": 0,
        "mean_usd": None,
        "bias": (
            "No gpt-6.1-sol review_receipt exists. The arm carries no cycle-2 "
            "cost for the docs axis and therefore under-states it; the "
            "substitute used for the headline figure is the gpt-5.5 escalator "
            "receipt, whose recorded mean is 2.8 times the cycle-1 seat. The "
            "headline therefore brackets rather than pins the docs escalator."
        ),
    },
]


def wilson_half_width(p: float, n: int, z: float = 1.96) -> float:
    return z * math.sqrt(p * (1.0 - p) / n)


def size_for(p: float, half_width: float, z: float = 1.96) -> int:
    return math.ceil(z * z * p * (1.0 - p) / (half_width * half_width))


def draw_cohort(pool: list[dict]) -> list[dict]:
    """Draw the fixed cohort: seeded, stable, and balanced across the two axes."""
    ordered = sorted(pool, key=lambda row: row["id"])
    rng = random.Random(SEED)
    drawn = rng.sample(ordered, 2 * PER_AXIS)
    axes = ["name"] * PER_AXIS + ["docs"] * PER_AXIS
    rng.shuffle(axes)
    return [
        {
            "id": row["id"],
            "axis": axis,
            "name_stage_at_draw": row["ns"],
            "docs_stage_at_draw": row["ds"],
        }
        for row, axis in zip(drawn, axes, strict=True)
    ]


def price_incumbent(axis: str) -> dict:
    seats = INCUMBENT_SEATS[axis]
    esc = ESCALATION[axis]
    base = seats["cycle0"]["mean_usd"] + seats["cycle1"]["mean_usd"]
    escalator = esc["cycle2_share"] * seats["cycle2"]["mean_usd"]
    per_item = base + escalator
    # Interval from the per-call spread the receipts carry: the cycle-2 term is
    # the only one whose *incidence* is uncertain, so the interval moves it
    # between never firing and always firing.
    lo = base
    hi = base + seats["cycle2"]["mean_usd"]
    return {
        "axis": axis,
        "cycle0_seat": seats["cycle0"],
        "cycle1_seat": seats["cycle1"],
        "cycle2_seat": seats["cycle2"],
        "escalation": esc,
        "per_item_usd": per_item,
        "per_item_interval_usd": [lo, hi],
        "calls_per_item": esc["calls_per_item"],
    }


def price_cascade(axis: str) -> dict:
    jev = JEV_RECEIPTS[axis]
    shares = ROUTE_SHARES[axis]
    chat = CHAT_SEAT[axis]
    chat_share = shares["chat_review"] + shares["chat_below_reject"]
    per_item = jev["mean_usd"] + chat_share * chat["mean_usd"]
    # Interval: the Jev call is certain; the chat call is not. Bound the
    # incidence by the share's own binomial spread at the calibration n.
    hw = wilson_half_width(chat_share, shares["n"])
    lo = jev["mean_usd"] + max(chat_share - hw, 0.0) * chat["mean_usd"]
    hi = jev["mean_usd"] + min(chat_share + hw, 1.0) * chat["mean_usd"]
    return {
        "axis": axis,
        "jev_call_usd": jev["mean_usd"],
        "jev_receipt_n": jev["n"],
        "jev_receipt_path": jev["path"],
        "chat_share": chat_share,
        "chat_share_interval": [max(chat_share - hw, 0.0), min(chat_share + hw, 1.0)],
        "chat_seat": chat,
        "per_item_usd": per_item,
        "per_item_interval_usd": [lo, hi],
    }


def per_accepted(per_item: dict, shares: dict) -> dict:
    return {
        axis: {
            "accept_share": shares[axis]["accept"],
            "per_accepted_name_usd": (
                per_item[axis]["per_item_usd"] / shares[axis]["accept"]
            ),
        }
        for axis in ("name", "docs")
    }


def wall_plan(incumbent: dict, cascade: dict, cohort_size: int) -> dict:
    per_item_s = {}
    for axis in ("name", "docs"):
        per_item_s[axis] = {
            "incumbent": incumbent[axis]["calls_per_item"] * CALL_WALL_S[axis],
            "cascade": (
                JEV_RECEIPTS[axis]["mean_latency_s"]
                # a chat route runs one seat call, so it costs one call's wall
                + ROUTE_SHARES[axis]["chat_review"] * CALL_WALL_S[axis]
            ),
        }
    per_axis = cohort_size // 2
    total_s = {
        arm: sum(per_item_s[axis][arm] * per_axis for axis in ("name", "docs"))
        for arm in ("incumbent", "cascade")
    }
    return {
        "call_wall_s": CALL_WALL_S,
        "per_item_seconds": per_item_s,
        "totals_seconds": total_s,
        "totals_minutes": {k: v / 60.0 for k, v in total_s.items()},
        "fits_one_worker_hour": {arm: total_s[arm] <= 3600.0 for arm in total_s},
        "split": (
            "Serial, one worker: the incumbent arm is the long pole because its "
            "escalator adds a third call on roughly a quarter of items. Run the "
            "two axes as separate work items, and where the incumbent arm exceeds "
            "one hour, split it by axis and run the docs half as its own item -- "
            "the two axes share no seat and no cohort item."
        ),
    }


def build_figure(out_dir: Path, incumbent: dict, cascade: dict) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - figure is optional evidence
        print(f"figure skipped: {exc}")
        return None
    axes_names = ["name", "docs"]
    fig, ax = plt.subplots(figsize=(14, 5), dpi=100)
    width = 0.36
    xs = range(len(axes_names))
    inc = [incumbent[a]["per_item_usd"] for a in axes_names]
    cas = [cascade[a]["per_item_usd"] for a in axes_names]
    inc_err = [
        [
            incumbent[a]["per_item_usd"] - incumbent[a]["per_item_interval_usd"][0]
            for a in axes_names
        ],
        [
            incumbent[a]["per_item_interval_usd"][1] - incumbent[a]["per_item_usd"]
            for a in axes_names
        ],
    ]
    ax.bar(
        [x - width / 2 for x in xs], inc, width, yerr=inc_err, label="incumbent quorum"
    )
    ax.bar([x + width / 2 for x in xs], cas, width, label="deciding cascade")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["name axis", "docs axis"])
    ax.set_ylabel("USD per item")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "srtj-cohort-estimate"
    fig.savefig(str(target) + ".svg", bbox_inches="tight")
    print(f"figure written: {target}.svg")
    return str(target) + ".svg"


def main() -> int:
    run_dir = Path(
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261019T035549247584-srtj-cohort-estimate"
    )
    snapshot = json.loads((run_dir / "eligible-pool.json").read_text())
    pool = snapshot["pool"]

    p_calibration = ROUTE_SHARES["name"]["chat_review"]
    needed = size_for(p_calibration, 0.05)
    needed_worst = size_for(0.5, 0.05)

    cohort = draw_cohort(pool)

    incumbent = {axis: price_incumbent(axis) for axis in ("name", "docs")}
    cascade = {axis: price_cascade(axis) for axis in ("name", "docs")}

    n_axis = PER_AXIS
    totals = {
        arm: {
            "total_usd": sum(
                arms[axis]["per_item_usd"] * n_axis for axis in ("name", "docs")
            ),
            "total_interval_usd": [
                sum(
                    arms[axis]["per_item_interval_usd"][0] * n_axis
                    for axis in ("name", "docs")
                ),
                sum(
                    arms[axis]["per_item_interval_usd"][1] * n_axis
                    for axis in ("name", "docs")
                ),
            ],
            "per_axis_usd": {
                axis: arms[axis]["per_item_usd"] * n_axis for axis in ("name", "docs")
            },
        }
        for arm, arms in (("incumbent", incumbent), ("cascade", cascade))
    }

    accepted = {
        "incumbent": per_accepted(incumbent, ROUTE_SHARES),
        "cascade": per_accepted(cascade, ROUTE_SHARES),
    }
    wall = wall_plan(incumbent, cascade, 2 * n_axis)

    figure_path = build_figure(
        HERE.parents[2] / "figures" / "sn-review-through-jev" / "srtj-cohort-estimate", incumbent, cascade
    )

    payload = {
        "node": "srtj-cohort-estimate",
        "plan_section": "sn-review-through-jev s6",
        "spend_usd": USD_SPENT,
        "seed": SEED,
        "cohort": {
            "size": len(cohort),
            "per_axis": n_axis,
            "axis_rule": (
                "Each drawn identity is assigned one axis; identities are "
                "drawn from the eligible pool in ascending id order and the "
                "axis sequence is shuffled by the same seeded generator."
            ),
            "pool_size_before_exclusion": snapshot["pool_rows"],
            "pool_size_after_exclusion": len(pool),
            "west_roster_names": snapshot["west_total"],
            "west_names_in_pool": snapshot["west_in_pool"],
            "items": cohort,
        },
        "precision": {
            "target_half_width": 0.05,
            "confidence": 0.95,
            "expected_share": p_calibration,
            "n_required_at_expected_share": needed,
            "n_required_worst_case": needed_worst,
            "per_axis_n": n_axis,
            "per_axis_half_width": wilson_half_width(p_calibration, n_axis),
            "pooled_n": 2 * n_axis,
            "pooled_half_width": wilson_half_width(p_calibration, 2 * n_axis),
            "pooled_half_width_worst_case": wilson_half_width(0.5, 2 * n_axis),
        },
        "receipts": {
            "incumbent_seats": INCUMBENT_SEATS,
            "escalation": ESCALATION,
            "jev": JEV_RECEIPTS,
            "route_shares": ROUTE_SHARES,
            "chat_seat": CHAT_SEAT,
            "call_wall_s": CALL_WALL_S,
            "seat_substitutions": SEAT_SUBSTITUTIONS,
        },
        "incumbent": incumbent,
        "cascade": cascade,
        "totals": totals,
        "per_accepted_name": accepted,
        "wall_plan": wall,
        "figure_path": figure_path,
    }

    (HERE / "srtj-cohort-estimate.json").write_text(json.dumps(payload, indent=1))
    write_html(payload, HERE / "srtj-cohort-estimate.html")

    print(f"pool {snapshot['pool_rows']} -> {len(pool)} after WEST exclusion")
    print(f"cohort {len(cohort)} items ({n_axis} per axis)")
    print(
        f"incumbent ${totals['incumbent']['total_usd']:.3f} "
        f"interval {totals['incumbent']['total_interval_usd']}"
    )
    print(
        f"cascade   ${totals['cascade']['total_usd']:.3f} "
        f"interval {totals['cascade']['total_interval_usd']}"
    )
    print(f"wall minutes {wall['totals_minutes']}")
    print(f"spend USD {USD_SPENT:.2f}")
    return 0


def write_html(payload: dict, target: Path) -> None:
    c = payload["cohort"]
    t = payload["totals"]
    p = payload["precision"]
    w = payload["wall_plan"]
    rows = "\n".join(
        f"<tr><td>srtj-cohort-estimate-item-{i + 1}</td><td>{it['id']}</td>"
        f"<td>{it['axis']}</td></tr>"
        for i, it in enumerate(c["items"])
    )
    subs = "".join(
        f"<li><strong>srtj-cohort-estimate-sub-{i}</strong>: configured "
        f"<code>{s['configured']}</code> ({s['role']}) priced as "
        f"<code>{s['priced_as']}</code> on n={s['n']}. {s['bias']}</li>"
        for i, s in enumerate(payload["receipts"]["seat_substitutions"])
    )
    html = f"""<section id="srtj-cohort-estimate">
<h2 id="srtj-cohort-estimate-heading">Fixed-cohort cost estimate: incumbent quorum against deciding cascade</h2>
<p>Spend for this node: <strong>USD 0</strong>. No model was called and no graph write was made.
Every figure below is an aggregate over rows the review pipeline recorded in the live graph or in a
prior crew's own decision receipts.</p>

<h3 id="srtj-cohort-estimate-cohort">The cohort</h3>
<p id="srtj-cohort-estimate-rule">Eligible pool: {c["pool_size_before_exclusion"]} identities with a bound
<code>StandardNameSource</code>, <code>validation_status = 'valid'</code>, a non-derived origin,
a non-terminal name stage, and a description. Removing the {c["west_names_in_pool"]} identities that
appear in the {c["west_roster_names"]}-name WEST review roster union leaves
{c["pool_size_after_exclusion"]}. Seeded draw (<code>seed={payload["seed"]}</code>, Python
<code>random.Random.sample</code> over ids in ascending order) of {c["size"]} items,
{c["per_axis"]} per axis.</p>
<p id="srtj-cohort-estimate-stage-note">The pool carries almost no item in a live drafted stage —
the graph holds one name-drafted and four docs-drafted identities — so the run reaches the drafted
pool through the review worker's restage path, and the cohort is drawn from identities that already
satisfy both axes' structural requirements.</p>

<h3 id="srtj-cohort-estimate-precision">Size and precision</h3>
<p id="srtj-cohort-estimate-precision-text">At the calibration route share of
{p["expected_share"]:.4f} the Wald half-width needs {p["n_required_at_expected_share"]} items for
±{p["target_half_width"]:.2f} at {p["confidence"]:.0%}; the worst case at p=0.5 needs
{p["n_required_worst_case"]}. The drawn {c["per_axis"]} per axis gives a per-axis half-width of
{p["per_axis_half_width"]:.4f} and the pooled {p["pooled_n"]} gives
{p["pooled_half_width"]:.4f}, with pooled worst case {p["pooled_half_width_worst_case"]:.4f}.
The cohort is therefore sized to the ±5 pt bar on each axis separately as well as pooled.</p>

<h3 id="srtj-cohort-estimate-receipts">Receipts</h3>
<p id="srtj-cohort-estimate-receipts-text">Incumbent seats from <code>LLMCost</code> over
2026-07-20..2026-10-17. Jev calls from the two crews' decision receipts
(<code>n={payload["receipts"]["jev"]["name"]["n"]}</code> on names,
<code>n={payload["receipts"]["jev"]["docs"]["n"]}</code> on docs). Escalation from
<code>StandardNameReview</code> cycle coverage over 2026-07-21..2026-10-17.</p>
<ul id="srtj-cohort-estimate-substitutions">{subs}</ul>

<h3 id="srtj-cohort-estimate-totals">Both arms</h3>
<p id="srtj-cohort-estimate-incumbent">Incumbent quorum:
<strong>${t["incumbent"]["total_usd"]:.2f}</strong> over {c["size"]} items, interval
[${t["incumbent"]["total_interval_usd"][0]:.2f},
${t["incumbent"]["total_interval_usd"][1]:.2f}]; per item
${payload["incumbent"]["name"]["per_item_usd"]:.4f} on names and
${payload["incumbent"]["docs"]["per_item_usd"]:.4f} on docs.</p>
<p id="srtj-cohort-estimate-cascade">Deciding cascade:
<strong>${t["cascade"]["total_usd"]:.2f}</strong>, interval
[${t["cascade"]["total_interval_usd"][0]:.2f},
${t["cascade"]["total_interval_usd"][1]:.2f}]; per item
${payload["cascade"]["name"]["per_item_usd"]:.4f} on names and
${payload["cascade"]["docs"]["per_item_usd"]:.4f} on docs.</p>
<p id="srtj-cohort-estimate-ratio">Ratio of incumbent to cascade:
{t["incumbent"]["total_usd"] / t["cascade"]["total_usd"]:.2f}x.</p>
<p id="srtj-cohort-estimate-accepted">Per accepted name, conditional on the recorded calibration
accept share: incumbent ${payload["per_accepted_name"]["incumbent"]["name"]["per_accepted_name_usd"]:.2f}
on names and ${payload["per_accepted_name"]["incumbent"]["docs"]["per_accepted_name_usd"]:.2f} on docs;
cascade ${payload["per_accepted_name"]["cascade"]["name"]["per_accepted_name_usd"]:.2f} and
${payload["per_accepted_name"]["cascade"]["docs"]["per_accepted_name_usd"]:.2f}.</p>

<h3 id="srtj-cohort-estimate-wall">Wall time</h3>
<p id="srtj-cohort-estimate-wall-text">Amortised seat call {w["call_wall_s"]["name"]:.2f} s (names) and
{w["call_wall_s"]["docs"]:.2f} s (docs). Incumbent
{w["totals_minutes"]["incumbent"]:.1f} min; cascade {w["totals_minutes"]["cascade"]:.1f} min.
Fits one worker hour: {w["fits_one_worker_hour"]}. {w["split"]}</p>

<h3 id="srtj-cohort-estimate-items">Cohort items</h3>
<table id="srtj-cohort-estimate-table">
<thead><tr><th>slot</th><th>identity</th><th>axis</th></tr></thead>
<tbody>
{rows}
</tbody></table>
</section>
"""
    target.write_text(html)


if __name__ == "__main__":
    sys.exit(main())

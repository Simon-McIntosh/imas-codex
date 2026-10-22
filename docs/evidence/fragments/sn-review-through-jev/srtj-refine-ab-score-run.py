"""Score the refine A/B against the blind labels.

The two refine arms — one driven by the reviewer's free-text comment, one
driven by the failed checks — produced a name each for every case in the
held-out cohort. A fresh worker labelled those names blind. This driver joins
the arm key to the labels and reports each arm's post-refine acceptance over
all rows, over the direct-source rows and over the lineage-recovered rows, its
cost per accepted name, and the paired difference between the arms.

Reads four committed fragments and writes the score JSON, its HTML record and
a machine-readable receipt. No graph access, no model call, no spend.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from imas_codex.discovery.base import evaluation
from imas_codex.discovery.base.evaluation import bootstrap_interval

HERE = Path(__file__).resolve().parent
SHEET = HERE / "srtj-refine-ab-label-sheet.json"
LABELS = HERE / "srtj-refine-ab-labels.json"
ARM_KEY = HERE / "srtj-refine-ab-arm-key.json"
RUN_RESULTS = HERE / "srtj-refine-ab-run-results.json"
OUT_JSON = HERE / "srtj-refine-ab-score.json"
OUT_HTML = HERE / "srtj-refine-ab-score.html"
ANCHOR = "srtj-refine-ab-score"

ARMS = ("comments", "checks")
ROUTES = ("all", "direct", "lineage")
RESAMPLES = 10000
RESAMPLE_SEED = 20261022
LEVEL = 0.95


def load() -> tuple[list[dict], list[dict], dict, dict]:
    return (
        json.loads(SHEET.read_text()),
        json.loads(LABELS.read_text()),
        json.loads(ARM_KEY.read_text()),
        json.loads(RUN_RESULTS.read_text()),
    )


def arm_credits(key: dict) -> dict[str, dict[str, set[str]]]:
    """Row id -> arm -> the case ids whose output for that arm is that row.

    The key names an arm only for the cases whose output *is* the row's output,
    so an identical pair of outputs is one row crediting both arms. Nothing on
    the sheet is read here: the arm comes from the key, never from a row's
    position on the sheet.
    """
    return {
        sheet_id: {arm: set(entry["arms"][arm]) for arm in entry["arms"]}
        for sheet_id, entry in key["rows"].items()
    }


def case_rows(credits: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, str]]:
    """Case id -> arm -> the sheet row id carrying that arm's output."""
    by_case: dict[str, dict[str, str]] = {}
    for sheet_id, arms in credits.items():
        for arm, cases in arms.items():
            for case in cases:
                by_case.setdefault(case, {})[arm] = sheet_id
    return by_case


def _acceptance_rate(_labels, accepted) -> float:
    """The share of an arm's outputs the labels accept.

    The statistic is a mean, so ``bootstrap_interval``'s ``labels`` argument is
    unused: the carrier passed in is the vector under test (``accepted``), and
    the pairing lives in ``scores``/``scores_b``.
    """
    return float(np.mean(accepted))


def acceptance_interval(values: np.ndarray) -> dict:
    """The 95% percentile bootstrap interval for one arm's acceptance rate."""
    carrier = values.astype(float)
    return bootstrap_interval(
        carrier,
        carrier,
        statistic=_acceptance_rate,
        n_resamples=RESAMPLES,
        seed=RESAMPLE_SEED,
        level=LEVEL,
    )


def paired_difference(checks: np.ndarray, comments: np.ndarray) -> dict:
    """The 95% interval for acceptance(checks) - acceptance(comments), paired.

    One observation per case, resampled by the evaluation module's paired
    bootstrap with the arms aligned case for case. The ``labels`` carrier is
    unused by the mean statistic.
    """
    return bootstrap_interval(
        checks.astype(float),
        checks,
        scores_b=comments,
        statistic=_acceptance_rate,
        n_resamples=RESAMPLES,
        seed=RESAMPLE_SEED,
        level=LEVEL,
    )


def route_cases(routes: dict[str, dict], route: str) -> list[str]:
    if route == "all":
        return sorted(routes)
    return sorted(c for c in routes if routes[c]["source_route"] == route)


def arm_stream(
    cases: list[str], by_case: dict[str, dict[str, str]], arm: str
) -> list[str]:
    return [by_case[case][arm] for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(f"measurement_module: {evaluation.__file__}", flush=True)
    print(f"measurement_cwd: {Path.cwd().resolve()}", flush=True)

    inputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (SHEET, LABELS, ARM_KEY, RUN_RESULTS)
    }
    sheet, labels, key, results = load()

    sheet_by_id = {row["sheet_id"]: row for row in sheet}
    label_of_row = {row["sheet_id"]: row["label"] for row in labels}
    assert set(sheet_by_id) == set(label_of_row)
    assert len(label_of_row) == len(labels) == len(sheet) == 124

    credits = arm_credits(key)
    by_case = case_rows(credits)
    routes = key["case_source_routes"]
    contributing = sorted(by_case)
    assert set(contributing) == set(routes)
    assert all(set(by_case[c]) == set(ARMS) for c in contributing)

    spend = {arm: results["arms"][arm]["spend_usd"] for arm in ARMS}
    report: dict[str, dict] = {}
    paired: dict[str, dict] = {}
    for route in ROUTES:
        cases = route_cases(routes, route)
        block: dict = {"cases": len(cases)}
        vectors: dict[str, np.ndarray] = {}
        for arm in ARMS:
            stream = arm_stream(cases, by_case, arm)
            assert len(set(stream)) == len(stream), "one row per case per arm"
            values = np.array([label_of_row[r] for r in stream], dtype=int)
            vectors[arm] = values
            accepted = int(values.sum())
            interval = acceptance_interval(values)
            block[arm] = {
                "rows": len(stream),
                "cases": len(stream),
                "accepted_rows": accepted,
                "accepted_cases": accepted,
                "acceptance_rows": accepted / len(stream),
                "acceptance_cases": accepted / len(stream),
                "acceptance_interval": {
                    "point": interval["point"],
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                    "level": interval["level"],
                    "n_resamples": interval["n_resamples"],
                    "seed": interval["seed"],
                },
                "spend_usd": spend[arm],
                "cost_per_accepted_usd": spend[arm] / accepted if accepted else None,
            }
        report[route] = block
        interval = paired_difference(vectors["checks"], vectors["comments"])
        paired[route] = {
            "cases": len(cases),
            "difference": "checks - comments",
            "point": interval["point"],
            "lower": interval["lower"],
            "upper": interval["upper"],
            "level": interval["level"],
            "n_resamples": interval["n_resamples"],
            "seed": interval["seed"],
        }

    counts = {
        "acceptable": sum(1 for row in labels if row["label"] == 1),
        "not_acceptable": sum(1 for row in labels if row["label"] == 0),
        "total": len(labels),
    }
    folded = sum(1 for row in sheet if len(credits[row["sheet_id"]]) == 2)
    assert folded == key["identical_output_pairs"]

    overall = report["all"]
    comparison = {
        "checks_at_least_comments": (
            overall["checks"]["acceptance_rows"]
            >= overall["comments"]["acceptance_rows"]
        ),
        "checks_acceptance": overall["checks"]["acceptance_rows"],
        "comments_acceptance": overall["comments"]["acceptance_rows"],
        "acceptance_gap": (
            overall["checks"]["acceptance_rows"]
            - overall["comments"]["acceptance_rows"]
        ),
        "checks_accepted": overall["checks"]["accepted_rows"],
        "comments_accepted": overall["comments"]["accepted_rows"],
        "checks_cost_per_accepted_usd": overall["checks"]["cost_per_accepted_usd"],
        "comments_cost_per_accepted_usd": overall["comments"]["cost_per_accepted_usd"],
        "checks_spend_usd": spend["checks"],
        "comments_spend_usd": spend["comments"],
        "paired_interval_contains_zero": (
            paired["all"]["lower"] <= 0.0 <= paired["all"]["upper"]
        ),
    }

    payload = {
        "node": ANCHOR,
        "cohort": {
            "complete_pairs": len(key["cases_in_sheet"]),
            "contributing_cases": len(contributing),
            "sheet_rows": len(sheet),
            "direct_cases": len(route_cases(routes, "direct")),
            "lineage_recovered_cases": len(route_cases(routes, "lineage")),
            "lineage_ambiguous_cases": len(key["lineage_ambiguous_cases"]),
            "identical_output_pairs": key["identical_output_pairs"],
            "lineage_control_cases": len(key["lineage_control"]),
            "lineage_control_agreements": sum(
                1 for row in key["lineage_control"] if row["agrees"]
            ),
        },
        "labels": counts,
        "spend_usd": {
            "comments": spend["comments"],
            "checks": spend["checks"],
            "refine_total_usd": results["refine_spend_usd"],
        },
        "axes": report,
        "paired_interval": paired,
        "comparison": comparison,
        "guards": {
            "llm_calls": 0,
            "spend_usd": 0,
            "graph_updates": 0,
            "arm_read_from": "arm key only; no row position or sheet order read",
            "bootstrap": (
                "imas_codex.discovery.base.evaluation.bootstrap_interval, "
                f"{RESAMPLES} seeded paired draws"
            ),
        },
        "input_sha256": inputs,
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    assert json.loads(OUT_JSON.read_text()) == payload

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{100.0 * value:.1f}%"

    def usd(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    axis_rows = []
    for route in ROUTES:
        for arm in ARMS:
            entry = report[route][arm]
            axis_rows.append(
                "<tr>"
                f"<td>{route}</td><td>{arm}</td>"
                f"<td>{entry['rows']}</td><td>{entry['accepted_rows']}</td>"
                f"<td>{pct(entry['acceptance_rows'])}</td>"
                f"<td>[{pct(entry['acceptance_interval']['lower'])}, "
                f"{pct(entry['acceptance_interval']['upper'])}]</td>"
                f"<td>{usd(entry['spend_usd'])}</td>"
                f"<td>{usd(entry['cost_per_accepted_usd'])}</td>"
                "</tr>"
            )
    paired_rows = [
        "<tr>"
        f"<td>{route}</td><td>{paired[route]['cases']}</td>"
        f"<td>{100 * paired[route]['point']:+.1f} pp</td>"
        f"<td>[{100 * paired[route]['lower']:+.1f}, "
        f"{100 * paired[route]['upper']:+.1f}] pp</td>"
        "</tr>"
        for route in ROUTES
    ]

    verdict = "at least" if comparison["checks_at_least_comments"] else "below"
    zero_clause = (
        "including zero"
        if comparison["paired_interval_contains_zero"]
        else "excluding zero"
    )

    OUT_HTML.write_text(
        f'<h2 id="{ANCHOR}">Scoring the refine arms against the blind labels</h2>\n'
        "<p>The two refine arms were scored on the "
        f"{len(sheet)} rows of the blind sheet, which the key assigns to "
        f"{len(contributing)} of the {len(key['cases_in_sheet'])} complete pairs. "
        f"The comment-driven arm is accepted on {comparison['comments_accepted']} of its "
        f"outputs and the check-driven arm on {comparison['checks_accepted']}. "
        f"{len(route_cases(routes, 'direct'))} cases reach the sheet directly and "
        f"{len(route_cases(routes, 'lineage'))} by lineage recovery; "
        f"{key['identical_output_pairs']} identical output pairs collapse into one row "
        "credited to both arms. A collapsed row is credited to each arm, never split. "
        "The arm of a row is read from the arm key alone — never from the sheet's order "
        "or a row's position.</p>\n"
        f'<h3 id="{ANCHOR}-acceptance">Post-refine acceptance and cost per accepted name</h3>\n'
        "<p>Acceptance is the share of an arm's outputs the blind labels accept. Row and "
        "case figures coincide here: each contributing case reaches the sheet once per "
        "arm, so an accepted count is the same per row and per case. The interval is the "
        f"{int(LEVEL * 100)}% percentile bootstrap over cases ({RESAMPLES} seeded draws, "
        "<code>imas_codex.discovery.base.evaluation.bootstrap_interval</code>).</p>\n"
        "<table><thead><tr><th>Route</th><th>Arm</th><th>Rows</th><th>Accepted</th>"
        "<th>Acceptance</th><th>95% interval</th><th>Arm spend [USD]</th>"
        "<th>Cost per accepted</th></tr></thead><tbody>"
        + "\n".join(axis_rows)
        + "</tbody></table>\n"
        "<p>Cost per accepted name is the arm's whole refine spend divided by the accepted "
        "outputs on the route; the spend is metered once for the arm and repeated across "
        "the routes, so a route's figure is that arm spend over the route's accepted "
        "outputs, not a separately metered cost.</p>\n"
        f'<h3 id="{ANCHOR}-paired">Paired difference between the arms</h3>\n'
        "<p>Over the cases where both arms produced an output, the paired difference is "
        "acceptance(checks) minus acceptance(comments), one observation per case, resampled "
        "by the same bootstrap with the pairing held. The interval is in percentage points "
        "of acceptance.</p>\n"
        "<table><thead><tr><th>Route</th><th>Cases</th><th>Difference</th>"
        "<th>95% interval</th></tr></thead><tbody>"
        + "\n".join(paired_rows)
        + "</tbody></table>\n"
        f'<h3 id="{ANCHOR}-comparison">Comparison statement</h3>\n'
        f"<p>The check-driven arm's acceptance is <strong>{verdict}</strong> the "
        f"comment-driven arm's: {pct(comparison['checks_acceptance'])} "
        f"({comparison['checks_accepted']}/{overall['checks']['rows']}) against "
        f"{pct(comparison['comments_acceptance'])} "
        f"({comparison['comments_accepted']}/{overall['comments']['rows']}), a gap of "
        f"{100 * comparison['acceptance_gap']:+.1f} percentage points. The paired "
        f"{int(LEVEL * 100)}% interval for that difference has point "
        f"{100 * paired['all']['point']:+.1f} pp and bounds "
        f"[{100 * paired['all']['lower']:+.1f}, {100 * paired['all']['upper']:+.1f}] pp, "
        f"{zero_clause}. At that acceptance the check-driven arm costs "
        f"USD {usd(comparison['checks_cost_per_accepted_usd'])} per accepted name against "
        f"USD {usd(comparison['comments_cost_per_accepted_usd'])} for the comment-driven "
        f"arm, on arm spends of USD {usd(comparison['checks_spend_usd'])} and "
        f"USD {usd(comparison['comments_spend_usd'])}.</p>\n"
        "<p>The keyword trigger is not retired here; that is the next node. Nothing in the "
        "repository is written except this record, its JSON and the labels test.</p>\n"
        f"<p>Driver: <code>{html.escape(Path(__file__).name)}</code>; receipt: "
        f"<code>{html.escape(str(args.receipt))}</code>. No graph read or write, no model "
        "call, USD 0.</p>\n"
    )

    receipt = {
        "summary": {
            "rows": len(sheet),
            "contributing_cases": len(contributing),
            "comments_acceptance_all": comparison["comments_acceptance"],
            "checks_acceptance_all": comparison["checks_acceptance"],
            "comments_accepted_all": comparison["comments_accepted"],
            "checks_accepted_all": comparison["checks_accepted"],
            "paired_point_all": paired["all"]["point"],
            "paired_lower_all": paired["all"]["lower"],
            "paired_upper_all": paired["all"]["upper"],
            "comments_cost_per_accepted_usd": comparison[
                "comments_cost_per_accepted_usd"
            ],
            "checks_cost_per_accepted_usd": comparison["checks_cost_per_accepted_usd"],
            "checks_at_least_comments": comparison["checks_at_least_comments"],
            "llm_calls": 0,
            "spend_usd": 0,
        },
        "input_sha256": inputs,
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (OUT_JSON, OUT_HTML)
        },
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")

    print("SUMMARY: " + json.dumps(receipt["summary"], sort_keys=True), flush=True)
    print(f"WROTE: {OUT_JSON}\nWROTE: {OUT_HTML}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

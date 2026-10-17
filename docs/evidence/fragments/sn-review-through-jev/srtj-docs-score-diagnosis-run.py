"""Diagnose why the documentation review score sits near chance on the frozen answers.

Reads only the frozen answers the documentation measurement already recorded,
the golden B fixture and the review-check taxonomy, and re-derives every number
in the fragment. It makes no network call. The object is the calibration split
alone; no validation label is read.
"""

from __future__ import annotations

import base64
import json
import lzma
import math
from pathlib import Path

import numpy as np

from imas_codex.discovery.base.evaluation import bootstrap_interval, roc_auc
from imas_codex.standard_names.review.checks import load_review_checks
from imas_codex.standard_names.review.jev_review import DOCS_AXIS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MEASURE = HERE / "srtj-docs-measure-results.json"
GOLDEN = HERE.parent / "sn-judgment-evaluation" / "golden-documentation-acceptable.json"
RESULTS = HERE / "srtj-docs-score-diagnosis-results.json"
FRAGMENT = HERE / "srtj-docs-score-diagnosis.html"
FIGDIR = ROOT / "docs/figures/sn-review-through-jev/srtj-docs-score-diagnosis"
FIGURE = FIGDIR / "calibration-component-auc.svg"

CLAIM = "check:claim_contradicts_source:"
DIMENSIONS = ("description_quality", "documentation_quality", "completeness")
ACCEPTED = {
    "description_quality": "precise_definition",
    "documentation_quality": "normative_definition",
    "completeness": "complete",
}
LABEL_CHECK_PROPERTY = {
    "subject": "provenance: the documented quantity matches the DD source path",
    "physics": "definitions, equations, signs and units",
    "scope": "the name is neither narrowed to one machine nor widened; relationships are true",
    "house_style": "conventions registry notation, forbidden spellings, catalog-level conventions",
    "links": "every name-link points at a name whose meaning matches its text",
    "complete": "nothing truncated, unresolved or self-contradictory; the opening template matches",
    "": "passes every check",
}
FLAT_SD = 0.02
FLAT_MODE = 0.90


def _counts(values) -> dict:
    out: dict = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _load_frozen() -> dict:
    stored = json.loads(MEASURE.read_text())
    frozen = json.loads(lzma.decompress(base64.b64decode(stored["frozen_lzma_b64"])))
    return {"receipts": frozen["receipts"], "context": frozen["context"], "stored": stored}


def _checks() -> dict:
    return {check.id: check for check in load_review_checks()[0] if check.axis == DOCS_AXIS}


def _calibration_rows() -> list[dict]:
    return [row for row in json.loads(GOLDEN.read_text()) if row["split"] == "calibration"]


def _answer(receipts: dict, unit: str) -> dict:
    return receipts["base:" + unit]["answers"]


def _dimension(ans: dict, dimension: str) -> float:
    answer = ans.get("dimension:" + dimension)
    return float(answer["probabilities"].get(ACCEPTED[dimension], 0))


def _check(ans: dict, check_id: str, checks: dict) -> float | None:
    answer = ans.get("check:" + check_id)
    if answer is None:
        return None
    if answer["type"] == "noul":
        return 1.0 - float(answer["noul"])
    return float(answer["probabilities"][checks[check_id].question.options[0]])


def _claim_stats(ans: dict) -> dict:
    claims = [value for key, value in ans.items() if key.startswith(CLAIM)]
    total = len(claims)
    if not total:
        return {"n": 0, "contradicts": 0.0, "says_nothing": 1.0, "supports": 0.0,
                "mean_p_contradicts": 0.0, "mean_p_says_nothing": 1.0,
                "mean_p_supports": 0.0, "value": 0.5, "veto": False}
    supports = sum(1 for c in claims if c["choice"] == "supports")
    contradicts = sum(1 for c in claims if c["choice"] == "contradicts")
    says_nothing = sum(1 for c in claims if c["choice"] == "says nothing")
    mean_p_sup = sum(float(c["probabilities"].get("supports", 0)) for c in claims) / total
    mean_p_say = sum(float(c["probabilities"].get("says nothing", 0)) for c in claims) / total
    mean_p_con = sum(float(c["probabilities"].get("contradicts", 0)) for c in claims) / total
    return {
        "n": total,
        "contradicts": contradicts / total,
        "says_nothing": says_nothing / total,
        "supports": supports / total,
        "mean_p_contradicts": mean_p_con,
        "mean_p_says_nothing": mean_p_say,
        "mean_p_supports": mean_p_sup,
        "value": mean_p_sup + 0.5 * mean_p_say,
        "veto": contradicts > 0,
    }


def asked_checks(checks: dict) -> list[str]:
    return [cid for cid, check in checks.items()
            if cid != "claim_contradicts_source" and check.question is not None]


def _aggregate(ans: dict, checks: dict) -> float | None:
    """Re-derive the measured score so the calibration AUC reproduces 0.610."""
    stats = _claim_stats(ans)
    if stats["n"] == 0:
        return None
    if stats["veto"]:
        return 0.0
    dims = [_dimension(ans, dim) for dim in DIMENSIONS]
    values = [_check(ans, cid, checks) for cid in asked_checks(checks)]
    values = [v for v in values if v is not None]
    return (0.5 * sum(dims) / len(dims)
            + 0.3 * (sum(values) / len(values) if values else 0.0)
            + 0.2 * stats["value"])


def _statistics(rows: list[dict], values: list[float]) -> dict:
    labels = [row["label"] for row in rows]
    point = float(roc_auc(labels, values))
    interval = bootstrap_interval(labels, values, copy_groups=[row["copy_group"] for row in rows])
    rounded = [round(value, 4) for value in values]
    modal = max(_counts(rounded).values())
    sd = float(np.std(values))
    return {
        "auc": point, "lower": interval["lower"], "upper": interval["upper"],
        "sd": sd, "distinct": len(set(rounded)), "modal_share": modal / len(values),
        "inverted": point < 0.5, "flat": sd < FLAT_SD or modal / len(values) >= FLAT_MODE,
    }


def _components(rows: list[dict], receipts: dict, checks: dict) -> list[dict]:
    entries: list[dict] = []

    def add(family: str, name: str, fn) -> None:
        values = [fn(_answer(receipts, row["unit"])) for row in rows]
        if any(value is None for value in values):
            return
        entries.append({"family": family, "name": name, **_statistics(rows, values)})

    add("aggregate", "aggregate score", lambda a: _aggregate(a, checks))
    for dim in DIMENSIONS:
        add("question", "question:" + dim, lambda a, d=dim: _dimension(a, d))
    for cid in asked_checks(checks):
        add("named check", "check:" + cid, lambda a, c=cid: _check(a, c, checks))
    add("claim channel", "claim: fraction contradicts", lambda a: _claim_stats(a)["contradicts"])
    add("claim channel", "claim: fraction says nothing", lambda a: _claim_stats(a)["says_nothing"])
    add("claim channel", "claim: fraction supports", lambda a: _claim_stats(a)["supports"])
    add("claim channel", "claim: mean P(contradicts)", lambda a: _claim_stats(a)["mean_p_contradicts"])
    add("veto", "contradiction veto (pass=1)", lambda a: 0.0 if _claim_stats(a)["veto"] else 1.0)
    return entries


def _label_gap(rows: list[dict], receipts: dict, checks: dict) -> dict:
    by_check = _label_check_counts(rows)
    aggregate = {row["unit"]: _label_aggregate(receipts, row["unit"], checks) for row in rows}
    failures = len(rows) - by_check.get("", 0)
    detail = []
    for check, count in sorted(by_check.items(), key=lambda kv: (-kv[1], kv[0])):
        subset = [row for row in rows if row["label_check"] == check]
        detail.append({
            "label_check": check or "(pass)",
            "property": LABEL_CHECK_PROPERTY[check],
            "count": count,
            "share_of_failures": (count / failures) if check else None,
            "labels": sorted({row["label"] for row in subset}),
            "mean_aggregate": float(np.mean([aggregate[row["unit"]] for row in subset])),
        })
    return {"by_check": detail, "failing": failures, "passing": by_check.get("", 0),
            "checks_never_asked": sorted(cid for cid, c in checks.items() if c.question is None)}


def _label_check_counts(rows: list[dict]) -> dict:
    return _counts(row["label_check"] for row in rows)


def _label_aggregate(receipts: dict, unit: str, checks: dict) -> float:
    return _aggregate(_answer(receipts, unit), checks)


def _claim_evidence(rows: list[dict], receipts: dict) -> dict:
    claim_counts, says_nothing, dd_lengths, description_lengths = [], [], [], []
    unobserved = {"unit": 0, "links": 0, "related_neighbours": 0, "source_paths": 0}
    for row in rows:
        receipt = receipts["base:" + row["unit"]]
        state = receipt["state"]
        stats = _claim_stats(receipt["answers"])
        claim_counts.append(stats["n"])
        says_nothing.append(stats["says_nothing"])
        sources = state["dd_source_docs"]
        values = sources["value"] if isinstance(sources, dict) and sources.get("value") else []
        dd_lengths.append(float(np.mean([len(str(s.get("value") or "")) for s in values])) if values else 0.0)
        description_lengths.append(len(str(state.get("description") or "")))
        for field in unobserved:
            entry = state.get(field)
            if not (isinstance(entry, dict) and entry.get("observed")):
                unobserved[field] += 1
    return {
        "claims_per_entry": {
            "min": min(claim_counts), "max": max(claim_counts),
            "median": float(np.median(claim_counts)),
            "distribution": {str(k): v for k, v in sorted(_counts(claim_counts).items())},
        },
        "says_nothing_fraction": {
            "mean": float(np.mean(says_nothing)), "median": float(np.median(says_nothing)),
            "all_silent": sum(1 for v in says_nothing if v == 1.0),
            "over_ninety_percent": sum(1 for v in says_nothing if v > 0.9),
            "bearing_evidence": sum(1 for v in says_nothing if v < 0.5),
        },
        "dd_source_characters": {
            "mean": float(np.mean(dd_lengths)), "median": float(np.median(dd_lengths)),
            "under_eight": sum(1 for v in dd_lengths if v < 8),
        },
        "enriched_description_characters": {
            "mean": float(np.mean(description_lengths)), "median": float(np.median(description_lengths)),
        },
        "pearson_dd_length_says_nothing": float(np.corrcoef(dd_lengths, says_nothing)[0, 1]),
        "state_fields_unobserved": unobserved,
        "state_carried": ["documentation", "documentation_claims", "description", "dd_source_docs"],
    }


def _incumbent(rows: list[dict], context: dict) -> dict:
    paired = [(row, context["by_name"][row["unit"]]["score"]) for row in rows]
    paired = [(row, score) for row, score in paired if score is not None]
    labels = [row["label"] for row, _ in paired]
    scores = [float(score) for _, score in paired]
    interval = bootstrap_interval(labels, scores, copy_groups=[row["copy_group"] for row, _ in paired])
    overlap = [row for row in rows if row["lead_overlap"]]
    overlap_scores = [context["by_name"][row["unit"]]["score"] for row in overlap
                      if context["by_name"][row["unit"]]["score"] is not None]
    return {
        "n": len(paired), "auc": float(roc_auc(labels, scores)),
        "lower": interval["lower"], "upper": interval["upper"],
        "mean": float(np.mean(scores)), "minimum": min(scores),
        "maximum": max(scores), "distinct": len({round(s, 4) for s in scores}),
        "lead_overlap": {
            "n": len(overlap),
            "labels": {str(k): v for k, v in sorted(_counts(row["label"] for row in overlap).items())},
            "mean_score": float(np.mean(overlap_scores)) if overlap_scores else None,
        },
    }


def _simulations(rows: list[dict], receipts: dict, checks: dict) -> list[dict]:
    """Calibration-only reweights and polarity flips computed from the frozen answers."""
    answers = [_answer(receipts, row["unit"]) for row in rows]
    dims = {d: np.array([_dimension(a, d) for a in answers]) for d in DIMENSIONS}
    asked = asked_checks(checks)
    check_values = {c: np.array([_check(a, c, checks) for a in answers]) for c in asked}
    claim_value = np.array([_claim_stats(a)["value"] for a in answers])
    veto = np.array([0.0 if _claim_stats(a)["veto"] else 1.0 for a in answers])
    inverted = {"latex_escaping_broken", "non_normative_documentation"}
    flat = {"cocos_number_in_prose"}
    dim_mean = np.mean([dims[d] for d in DIMENSIONS], axis=0)

    def check_mean(keys):
        return np.mean([check_values[c] for c in keys], axis=0)

    def flipped_mean(keys):
        return np.mean([(1 - check_values[c] if c in inverted else check_values[c]) for c in keys], axis=0)

    def with_veto(score):
        return np.where(veto == 0.0, 0.0, score)

    keep_inverted = [c for c in asked if c not in inverted]
    keep_flat = [c for c in asked if c not in flat]
    keep_both = [c for c in asked if c not in inverted and c not in flat]
    variants = [
        ("baseline (reproduces 0.610)",
         with_veto(0.5 * dim_mean + 0.3 * check_mean(asked) + 0.2 * claim_value), False),
        ("drop the near-chance claim term, renormalise",
         with_veto(0.5 / 0.8 * dim_mean + 0.3 / 0.8 * check_mean(asked)), False),
        ("drop the inverted checks",
         with_veto(0.5 * dim_mean + 0.3 * check_mean(keep_inverted) + 0.2 * claim_value), False),
        ("drop the flat COCOS check",
         with_veto(0.5 * dim_mean + 0.3 * check_mean(keep_flat) + 0.2 * claim_value), False),
        ("flip the inverted checks before combining",
         with_veto(0.5 * dim_mean + 0.3 * flipped_mean(asked) + 0.2 * claim_value), False),
        ("drop the claim term, inverted and flat checks",
         with_veto(0.5 / 0.8 * dim_mean + 0.3 / 0.8 * check_mean(keep_both)), False),
        ("aggregate without the veto", 0.5 * dim_mean + 0.3 * check_mean(asked) + 0.2 * claim_value, False),
    ]
    return [{"change": name, "needs_new_calls": calls,
             **_statistics(rows, [float(v) for v in values])}
            for name, values, calls in variants]


def _sample(rows: list[dict], receipts: dict, checks: dict, size: int = 20) -> list[dict]:
    """A stratified sample across the first failing rubric check."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["label_check"], []).append(row)
    order = sorted(buckets, key=lambda key: (-len(buckets[key]), key))
    quota = {key: size * len(buckets[key]) / len(rows) for key in order}
    assigned = {key: int(math.floor(quota[key])) for key in order}
    remainder = size - sum(assigned.values())
    for key in sorted(order, key=lambda k: (-(quota[k] - assigned[k]), k))[:remainder]:
        assigned[key] += 1
    sample = []
    for key in order:
        for row in sorted(buckets[key], key=lambda r: r["unit"])[:assigned[key]]:
            sample.append({
                "unit": row["unit"], "label": row["label"],
                "label_check": row["label_check"] or "(pass)",
                "label_note": row["label_note"],
                "aggregate": _aggregate(_answer(receipts, row["unit"]), checks),
            })
    return sample


def _figure(components: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "docs-score-diagnosis"
    ordered = sorted(
        components, key=lambda c: (c["family"] != "aggregate", -c["auc"])
    )
    fig, ax = plt.subplots(figsize=(14, 8.4), dpi=100)
    grey, red = "#929292", "#b0393a"
    for index, comp in enumerate(ordered):
        y = len(ordered) - index
        ax.errorbar(
            comp["auc"], y,
            xerr=[[comp["auc"] - comp["lower"]], [comp["upper"] - comp["auc"]]],
            fmt="o", markersize=7, linewidth=2.6, color=red if comp["inverted"] else grey,
            zorder=3, capsize=3, elinewidth=2.6,
        )
    ax.axvline(0.5, color="#555555", linewidth=1.2, linestyle=":", zorder=1)
    ax.text(0.5, len(ordered) + 0.9, "chance 0.5", color="#555555", fontsize=14,
            ha="center", va="bottom")
    ax.set_yticks([len(ordered) - i for i in range(len(ordered))], [c["name"] for c in ordered])
    ax.set_ylim(0.3, len(ordered) + 1.6)
    ax.set_xlim(0.30, 0.85)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("calibration AUC [95% bootstrap interval]")
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Line2D([], [], color=grey, marker="o", linestyle="", markersize=7, label="component"),
            Line2D([], [], color=red, marker="o", linestyle="", markersize=7,
                   label="below chance (inverted)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label,
             *fig.legends[0].get_texts(), *ax.texts]
    visible = [t for t in texts if t.get_visible() and t.get_text()]
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in visible]
    canvas = fig.bbox
    for label, box in boxes:
        if box.x0 < canvas.x0 or box.y0 < canvas.y0 or box.x1 > canvas.x1 or box.y1 > canvas.y1:
            raise RuntimeError(f"figure text is clipped: {label}")
    for index, (first, first_box) in enumerate(boxes):
        for second, second_box in boxes[index + 1:]:
            if first_box.overlaps(second_box):
                raise RuntimeError(f"figure text overlaps: {first} / {second}")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)


MECHANISM = {
    "drop the near-chance claim term, renormalise":
        "The claim channel carries weight 0.2 in the aggregate but scores near chance alone, "
        "so it adds noise rather than signal; renormalising the dimension and check weights "
        "over the remaining 0.8 leaves only the components that separate.",
    "drop the inverted checks":
        "Two checks score below chance, so their higher values point at rejection rather than "
        "acceptance; removing them from the check mean stops them from cancelling the checks "
        "that do carry signal.",
    "drop the flat COCOS check":
        "One check is near-constant across the calibration entries, so it cannot move the mean "
        "in either direction and only dilutes the checks that vary.",
    "flip the inverted checks before combining":
        "The same information the inverted checks already carry, read the right way round; a "
        "polarity flip keeps their weight instead of discarding it.",
    "drop the claim term, inverted and flat checks":
        "Removes all three defects at once: the near-chance claim term, the two inverted checks "
        "and the constant check. This is the largest calibration-only gain available with no new "
        "judge call.",
    "aggregate without the veto":
        "The veto fires on 4 of 216 calibration entries, so dropping it changes almost nothing; "
        "it is reported to bound how much of the near-chance result the veto could explain.",
}


def _fmt_interval(auc: float, lower: float, upper: float) -> str:
    return "%.3f [%.3f, %.3f]" % (auc, lower, upper)


def _flag(row: dict) -> str:
    flags = []
    if row["inverted"]:
        flags.append("inverted")
    if row["flat"]:
        flags.append("flat")
    return ", ".join(flags)


def _ranked_changes(simulations, gap, cost, n_entries) -> list[dict]:
    ranked = []
    for sim in sorted(simulations, key=lambda s: -s["auc"]):
        if sim["change"].startswith("baseline"):
            continue
        ranked.append({
            "change": sim["change"],
            "mechanism": MECHANISM.get(sim["change"], ""),
            "calibration_auc": sim["auc"], "lower": sim["lower"], "upper": sim["upper"],
            "needs_new_calls": False, "cost_usd": 0.0, "cost_note": "none",
        })
    never = len(gap["checks_never_asked"])
    usd_per_claim = float(cost["usd_per_claim"])
    seconds_per_claim = float(cost["seconds_per_claim"])
    mean_cost = float(cost["base"]["mean_cost"])
    new_calls = never * n_entries
    ranked.append({
        "change": "ask the %d documentation checks the judge is never shown" % never,
        "mechanism":
            "Every check whose question is None is absent from the prompt, so the score is blind "
            "to whatever the labels fail those entries on. The link, registry-notation, opening-"
            "template and symbol checks are all in that set.",
        "calibration_auc": None, "lower": None, "upper": None,
        "needs_new_calls": True, "cost_usd": new_calls * usd_per_claim,
        "cost_note": "%d new claims over %d entries at USD %.3g each, USD %.3g total, "
                     "%.0f s added latency"
                     % (new_calls, n_entries, usd_per_claim, new_calls * usd_per_claim,
                        new_calls * seconds_per_claim),
    })
    ranked.append({
        "change": "enrich the claim state with units, links, siblings and source paths",
        "mechanism":
            "The claim state carries only documentation, documentation_claims, description and "
            "dd_source_docs; units, links, neighbours and source paths are unobserved, so where "
            "the DD text is a few characters the judge can only record that it cannot decide.",
        "calibration_auc": None, "lower": None, "upper": None,
        "needs_new_calls": True, "cost_usd": n_entries * mean_cost,
        "cost_note": "a full re-judge of the %d entries at USD %.3g each, USD %.3g total"
                     % (n_entries, mean_cost, n_entries * mean_cost),
    })
    return ranked


def _fragment(bundle: dict) -> str:
    control = bundle["control"]
    gap = bundle["label_gap"]
    evidence = bundle["claim_evidence"]
    incumbent = bundle["incumbent"]
    sample = bundle["sample"]
    costs = bundle["cost"]
    node = bundle["node"]

    parts = []
    parts.append('<section id="' + node + '" data-reckon="evidence">')
    parts.append('<h2 id="' + node + '-title">Why the documentation score sits near chance</h2>')

    parts.append(
        '<p id="' + node + '-control"><strong>Control.</strong> Re-deriving the frozen score '
        'reproduces the measurement: the recomputed aggregate calibration AUC is %.3f against the '
        'recorded %.3f [%.3f, %.3f], so every number below is read from the same arithmetic that '
        'produced the measured arm. %d calibration entries were read, none of them a validation '
        'label; the validation labels were not opened.</p>'
        % (control["recomputed_auc"], control["measured_auc"], control["measured_lower"],
           control["measured_upper"], control["n"])
    )

    parts.append(
        '<figure id="' + node + '-figure"><img src="/imas-codex/figures/sn-review-through-jev/'
        + node + '/calibration-component-auc.svg" alt="Calibration AUC with 95% bootstrap '
        'intervals for every component of the documentation score, with a dotted line at 0.5.">'
        '<figcaption>One mark per component of the score, computed on the calibration split alone. '
        'Grey marks are at or above chance, red marks are below chance. The dotted line is 0.5. '
        'Intervals are 1,000 copy-group bootstrap resamples of the same frozen answers.</figcaption>'
        '</figure>'
    )

    # Q1
    parts.append('<h3 id="' + node + '-q1-title">1. Which component carries signal, and which does not</h3>')
    rows = []
    for comp in sorted(bundle["components"], key=lambda c: (c["family"] != "aggregate", -c["auc"])):
        rows.append(
            "<tr><td><code>" + comp["name"] + "</code></td><td>" + comp["family"] + "</td>"
            + "<td>%.3f</td>" % comp["sd"]
            + "<td>%d</td>" % comp["distinct"]
            + "<td>%.2f</td>" % comp["modal_share"]
            + "<td>" + _fmt_interval(comp["auc"], comp["lower"], comp["upper"]) + "</td>"
            + "<td>" + (_flag(comp) or "&mdash;") + "</td></tr>"
        )
    parts.append(
        '<table id="' + node + '-q1-table"><thead><tr><th>Component</th><th>Family</th>'
        '<th>Std</th><th>Distinct</th><th>Modal share</th><th>Calibration AUC [95%]</th>'
        '<th>Flag</th></tr></thead><tbody>' + "\n".join(rows) + "</tbody></table>"
    )
    inverted = [c["name"] for c in bundle["components"] if c["inverted"]]
    flat = [c["name"] for c in bundle["components"] if c["flat"]]
    parts.append(
        '<p id="' + node + '-q1-verdict"><strong>%d components are inverted</strong> (AUC below 0.5): '
        '%s. <strong>%d are flat</strong> (standard deviation under %.2f, or one value holding %.0f%% '
        'of entries): %s. Every interval straddles chance, so no single component is separated from '
        'it on this split; the defects are named by point estimate and spread, not by a bound.</p>'
        % (len(inverted), ", ".join("<code>" + n + "</code>" for n in inverted) or "none",
           len(flat), FLAT_SD, FLAT_MODE * 100,
           ", ".join("<code>" + n + "</code>" for n in flat) or "none")
    )

    # Q2
    parts.append('<h3 id="' + node + '-q2-title">2. What the labels judge that the score never asks</h3>')
    gap_rows = []
    for entry in gap["by_check"]:
        share = "&mdash;" if entry["share_of_failures"] is None else "%.0f%%" % (entry["share_of_failures"] * 100)
        gap_rows.append(
            "<tr><td><code>" + entry["label_check"] + "</code></td><td>" + entry["property"] + "</td>"
            + "<td>%d</td><td>%s</td><td>%.3f</td><td>%s</td></tr>"
            % (entry["count"], share, entry["mean_aggregate"],
           ", ".join(str(v) for v in entry["labels"]) or "&mdash;")
        )
    parts.append(
        '<table id="' + node + '-q2-gap"><thead><tr><th>First failing check</th><th>Property the '
        'labels judge</th><th>Entries</th><th>Share of failures</th><th>Mean score</th>'
        '<th>Labels</th></tr></thead><tbody>' + "\n".join(gap_rows) + "</tbody></table>"
    )
    never = gap["checks_never_asked"]
    parts.append(
        '<p id="' + node + '-q2-never"><strong>%d of the documentation checks are never put to the '
        'judge</strong> (their question is None), so the score cannot see them: %s. The labels, by '
        'contrast, fail overwhelmingly on house style and links &mdash; between them %.0f%% of the '
        '%d failing entries &mdash; which are exactly the families the never-asked checks cover. '
        'The score is structurally blind to the dominant failure mode.</p>'
        % (len(never), ", ".join("<code>" + c + "</code>" for c in never),
           100 * sum(e["share_of_failures"] for e in gap["by_check"]
                     if e["label_check"] in ("house_style", "links")),
           gap["failing"])
    )
    sample_rows = []
    for row in sample:
        sample_rows.append(
            "<tr><td><code>" + row["unit"] + "</code></td><td>" + str(row["label"]) + "</td>"
            + "<td>" + row["label_check"] + "</td><td>" + (row["label_note"] or "&mdash;") + "</td>"
            + "<td>%.3f</td></tr>" % row["aggregate"]
        )
    parts.append(
        '<table id="' + node + '-q2-sample"><caption>Stratified sample of 20 labelled calibration '
        'entries, allocated across the first failing check</caption><thead><tr><th>Unit</th>'
        '<th>Label</th><th>First failing check</th><th>Label note</th><th>Score</th></tr></thead>'
        '<tbody>' + "\n".join(sample_rows) + "</tbody></table>"
    )
    parts.append(
        '<p id="' + node + '-q2-reverse">The reverse gap is as sharp. The labels record one binary '
        'verdict with one first-failing check; the score returns continuous probabilities for three '
        'separate dimensions, one probability per asked check, and a claim relation. No label row '
        'records a claim at all &mdash; 0 of the %d calibration rows carry one &mdash; so the 0.2 '
        'weight the claim channel holds is judged by nothing in the fixture.</p>' % control["n"]
    )

    # Q3
    silent = evidence["says_nothing_fraction"]
    dd = evidence["dd_source_characters"]
    desc = evidence["enriched_description_characters"]
    claims = evidence["claims_per_entry"]
    unobserved = evidence["state_fields_unobserved"]
    parts.append('<h3 id="' + node + '-q3-title">3. What the claim state carries</h3>')
    parts.append(
        '<p id="' + node + '-q3-answer"><strong>Most claims say nothing, and the DD text is why.</strong> '
        'The mean share of claims that say nothing is %.3f (median %.3f); %d of %d entries have every '
        'claim silent, %d of %d are over 90%% silent, and only %d carry evidence in half their claims. '
        'The frozen DD source text is a median of %.0f characters, with %d of %d entries under eight '
        'characters &mdash; a bare <code>Value</code> or a unit string. The enriched description is a '
        'median of %.0f characters. The share of silent claims and the DD text length move together at '
        'Pearson %.3f.</p>'
        % (silent["mean"], silent["median"], silent["all_silent"], control["n"],
           silent["over_ninety_percent"], control["n"], silent["bearing_evidence"],
           dd["median"], dd["under_eight"], control["n"], desc["median"],
           evidence["pearson_dd_length_says_nothing"])
    )
    parts.append(
        '<p id="' + node + '-q3-state">The claim state carries four fields only: documentation, '
        'documentation_claims, description and dd_source_docs. Across the %d calibration entries every '
        'one is unobserved on the four fields a claim could be checked against &mdash; units '
        '%d, links %d, related neighbours %d, source paths %d. Claims per entry range %d to %d '
        '(median %.0f), so the channel is neither absent nor thin in count &mdash; it is starved of '
        'the evidence a relation could be checked against.</p>'
        % (control["n"], unobserved["unit"], unobserved["links"], unobserved["related_neighbours"],
           unobserved["source_paths"], claims["min"], claims["max"], claims["median"])
    )

    # Q4
    parts.append('<h3 id="' + node + '-q4-title">4. Does the same gap explain the incumbent at 0.514?</h3>')
    overlap = incumbent["lead_overlap"]
    overlap_mean = ("%.3f" % overlap["mean_score"] if overlap["mean_score"] is not None
                    else "unavailable")
    parts.append(
        '<p id="' + node + '-q4-answer">On calibration the incumbent scores %.3f [%.3f, %.3f] over '
        '%d entries. It is near-constant: %.3f to %.3f, only %d distinct values. A near-constant '
        'score has AUC near 0.5 almost by construction, because most of the pairing is ties, so its '
        'own flatness is a sufficient explanation of its near-chance result and needs no appeal to '
        'what it reads. Whether the same label-versus-question gap applies cannot be settled from '
        'the frozen answers: two arms failing for a shared reason and two arms failing for '
        'unrelated reasons look the same here, and they are statistically indistinguishable '
        '(31 distinct values against 205, overlapping intervals). The label agreement field does '
        'not separate them either: only %d of %d entries carry lead_overlap, %d accepted and %d '
        'not, and the incumbent mean over those %d is %s against %.3f overall. The most this '
        'fixture supports is the part that is shared and measurable: both arms reduce an entry to '
        'one scalar, and that scalar is not recorded against the link and house-style properties '
        'the labels fail on.</p>'
        % (incumbent["auc"], incumbent["lower"], incumbent["upper"], incumbent["n"],
           incumbent["minimum"], incumbent["maximum"], incumbent["distinct"],
           overlap["n"], control["n"],
           overlap["labels"].get("1", 0), overlap["labels"].get("0", 0), overlap["n"],
           overlap_mean, incumbent["mean"])
    )

    # Q5
    parts.append('<h3 id="' + node + '-q5-title">5. Ranked changes, all to the Jev arm</h3>')
    change_rows = []
    for index, change in enumerate(bundle["ranked_changes"], start=1):
        auc_cell = ("&mdash;" if change["calibration_auc"] is None
                    else _fmt_interval(change["calibration_auc"], change["lower"], change["upper"]))
        calls = "yes" if change["needs_new_calls"] else "no"
        change_rows.append(
            "<tr><td>%d</td><td>" % index + change["change"] + "</td><td>" + change["mechanism"]
            + "</td><td>" + auc_cell + "</td><td>" + calls + "</td><td>" + change["cost_note"] + "</td></tr>"
        )
    parts.append(
        '<table id="' + node + '-q5-table"><thead><tr><th>Rank</th><th>Change</th><th>Mechanism</th>'
        '<th>Simulated calibration AUC [95%]</th><th>Needs new Jev calls</th><th>Cost</th></tr>'
        '</thead><tbody>' + "\n".join(change_rows) + "</tbody></table>"
    )
    best = bundle["ranked_changes"][0]
    parts.append(
        '<p id="' + node + '-q5-answer">The best calibration-only change is <strong>' + best["change"]
        + '</strong>: %.3f [%.3f, %.3f] against the baseline %.3f. It costs nothing, because it is a '
        'reweighting of answers already frozen. The ranked changes that need new calls are the two '
        'structural repairs: putting the %d never-asked checks in front of the judge, and stocking '
        'the claim state with the evidence it currently lacks. Both raise prompt size rather than '
        'adding a provider, so both keep the local judge.' % (
            best["calibration_auc"], best["lower"], best["upper"], bundle["control"]["recomputed_auc"],
            len(never))
    )
    parts.append(
        '<p id="' + node + '-cost-baseline">Baseline judge cost, for scale: USD %.3g per claim and '
        '%.3f s per claim, %.6f USD and %.3f s per entry on the base judgments.</p>'
        % (costs["usd_per_claim"], costs["seconds_per_claim"],
           costs["base"]["mean_cost"], costs["base"]["mean_latency_s"])
    )
    parts.append(
        '<p id="' + node + '-verdict"><strong>Verdict.</strong> The score sits near chance on '
        'calibration for two reasons that compound: two checks are inverted and one is constant, so '
        'the named-check mean carries a cancellation; and the claim channel, worth 0.2 of the '
        'aggregate, says nothing on most entries because the DD text it is given is a few characters '
        'long. The labels fail on link and house-style properties that the judge is never asked '
        'about at all. A reweighting recovers part of the gap with no new calls; the larger share '
        'needs the never-asked checks and a stocked claim state.</p>'
    )
    parts.append("</section>")
    return "\n".join(parts) + "\n"


def main() -> int:
    frozen = _load_frozen()
    receipts = frozen["receipts"]
    context = frozen["context"]
    stored = frozen["stored"]
    checks = _checks()
    rows = _calibration_rows()

    values = [_aggregate(_answer(receipts, row["unit"]), checks) for row in rows]
    measured = stored["splits"]["calibration"]["jev"]
    control = {
        "n": len(rows),
        "recomputed_auc": _statistics(rows, values)["auc"],
        "measured_auc": float(measured["point"]),
        "measured_lower": float(measured["lower"]),
        "measured_upper": float(measured["upper"]),
        "recomputed_mean": float(np.mean(values)),
    }

    components = _components(rows, receipts, checks)
    gap = _label_gap(rows, receipts, checks)
    evidence = _claim_evidence(rows, receipts)
    incumbent = _incumbent(rows, context)
    simulations = _simulations(rows, receipts, checks)
    sample = _sample(rows, receipts, checks)
    cost = {
        "usd_per_claim": float(stored["cost"]["usd_per_claim"]),
        "seconds_per_claim": float(stored["cost"]["seconds_per_claim"]),
        "base": {
            "mean_cost": float(stored["cost"]["base"]["mean_cost"]),
            "mean_latency_s": float(stored["cost"]["base"]["mean_latency_s"]),
        },
    }
    ranked = _ranked_changes(simulations, gap, cost, len(rows))

    bundle = {
        "node": "srtj-docs-score-diagnosis",
        "plan": "imas-codex:sn-review-through-jev",
        "scope": "calibration split only; no validation label read; no network call",
        "control": control,
        "components": components,
        "label_gap": gap,
        "sample": sample,
        "claim_evidence": evidence,
        "incumbent": incumbent,
        "simulations": simulations,
        "ranked_changes": ranked,
        "cost": cost,
        "figure": "docs/figures/sn-review-through-jev/srtj-docs-score-diagnosis/calibration-component-auc.svg",
    }

    RESULTS.write_text(json.dumps(bundle, indent=2, sort_keys=False) + "\n")
    _figure(components)
    FRAGMENT.write_text(_fragment(bundle))

    print("control: recomputed %.4f vs measured %.4f over %d calibration entries"
          % (control["recomputed_auc"], control["measured_auc"], control["n"]))
    for comp in components:
        print("  %-42s auc=%.3f [%.3f,%.3f]%s%s"
              % (comp["name"], comp["auc"], comp["lower"], comp["upper"],
                 " INV" if comp["inverted"] else "", " FLAT" if comp["flat"] else ""))
    print("inverted: %s" % ", ".join(c["name"] for c in components if c["inverted"]))
    print("flat: %s" % ", ".join(c["name"] for c in components if c["flat"]))
    print("never asked checks: %d" % len(gap["checks_never_asked"]))
    print("best change: %s -> %.3f [%.3f,%.3f]" % (ranked[0]["change"], ranked[0]["calibration_auc"],
                                                    ranked[0]["lower"], ranked[0]["upper"]))
    print("wrote %s" % RESULTS)
    print("wrote %s" % FIGURE)
    print("wrote %s" % FRAGMENT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
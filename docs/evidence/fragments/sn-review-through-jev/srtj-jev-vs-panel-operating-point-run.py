"""Compare frozen reviewers at matched acceptance operating points."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import lzma
import math
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from imas_codex.discovery.base import evaluation
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.review import cascade_calibration as calibration

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
STEM = Path(__file__).name.removesuffix("-run.py")
SPLITS = ("calibration", "validation")
METRICS = (
    "accept_rate",
    "accept_precision",
    "accept_recall",
    "refine_precision",
    "refine_recall",
)
JEV = "#246c78"
PANEL = "#747474"
RESAMPLES = 10000
SEED = 20261020


def read(path):
    raw = path.read_bytes()
    return json.loads(lzma.decompress(raw) if path.suffix == ".xz" else raw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(numerator, denominator):
    if not denominator:
        return None
    p = numerator / denominator
    z = 1.959963984540054
    divisor = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / divisor
    half = (
        z
        * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2))
        / divisor
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def measures(labels, accepted):
    y = np.asarray(labels, dtype=int)
    a = np.asarray(accepted, dtype=bool)
    tp = int(np.sum(a & (y == 1)))
    fp = int(np.sum(a & (y == 0)))
    fn = int(np.sum(~a & (y == 1)))
    tn = int(np.sum(~a & (y == 0)))
    counts = (
        (tp + fp, len(y)),
        (tp, tp + fp),
        (tp, tp + fn),
        (tn, tn + fn),
        (tn, tn + fp),
    )
    return {
        key: {
            "numerator": num,
            "denominator": den,
            "point": num / den if den else None,
            "wilson_95": wilson(num, den),
        }
        for key, (num, den) in zip(METRICS, counts, strict=True)
    }


def independent_tally(rows, accepted):
    """Use identity sets to check the array confusion matrix independently."""
    good = {row["id"] for row in rows if row["label"] == 1}
    bad = {row["id"] for row in rows if row["label"] == 0}
    chosen = {row["id"] for row, flag in zip(rows, accepted, strict=True) if flag}
    refused = (good | bad) - chosen
    assert len(good | bad) == len(rows), "identity collision in paired rows"
    return (
        (len(chosen), len(rows)),
        (len(chosen & good), len(chosen)),
        (len(chosen & good), len(good)),
        (len(refused & bad), len(refused)),
        (len(refused & bad), len(bad)),
    )


def check_tallies(rows, accepted, result):
    for key, (num, den) in zip(METRICS, independent_tally(rows, accepted), strict=True):
        assert (num, den) == (result[key]["numerator"], result[key]["denominator"]), (
            f"independent tally differs: {key}"
        )


def tie_key(identity):
    return hashlib.sha256(("acceptance-order:" + identity).encode()).hexdigest()


def same_rate(scores, ids, count):
    """Accept an exact quota, breaking boundary ties without labels."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], tie_key(ids[i])))
    accepted = np.zeros(len(scores), dtype=bool)
    accepted[order[:count]] = True
    cutoff = float(scores[order[count - 1]]) if count else None
    tied = [i for i, value in enumerate(scores) if value == cutoff]
    selected_ties = sum(accepted[i] for i in tied)
    return accepted, {
        "cutoff": cutoff,
        "accepted": count,
        "boundary_ties": len(tied),
        "boundary_ties_accepted": int(selected_ties),
        "pure_threshold": selected_ties in (0, len(tied)),
        "tie_rule": "SHA256('acceptance-order:' + canonical identity), ascending",
    }


def precision_target(labels, scores, target):
    """Report attainable thresholds and an explicitly randomised exact match."""
    y = np.asarray(labels)
    points = []
    for cutoff in sorted(set(scores), reverse=True):
        chosen = np.asarray(scores) >= cutoff
        n = int(chosen.sum())
        tp = int(y[chosen].sum())
        points.append(
            {
                "cutoff": float(cutoff),
                "accepted": n,
                "tp": tp,
                "precision": tp / n,
                "recall": tp / int(y.sum()),
            }
        )
    feasible = [p for p in points if p["precision"] >= target - 1e-12]
    if not feasible:
        return {"attainable": None, "exact_threshold": False, "randomised_exact": None}
    best = max(feasible, key=lambda p: (p["recall"], p["accepted"]))
    exact = [p for p in points if abs(p["precision"] - target) < 1e-12]
    exact_point = max(exact, key=lambda p: p["recall"]) if exact else None
    crossings = []
    for high, low in zip(points, points[1:], strict=False):
        if high["precision"] >= target > low["precision"]:
            dtp, dn = low["tp"] - high["tp"], low["accepted"] - high["accepted"]
            fraction = (target * high["accepted"] - high["tp"]) / (dtp - target * dn)
            expected_tp = high["tp"] + fraction * dtp
            expected_n = high["accepted"] + fraction * dn
            assert 0 <= fraction <= 1
            assert abs(expected_tp / expected_n - target) < 1e-12
            crossings.append(
                {
                    "cutoff": low["cutoff"],
                    "boundary_accept_probability": fraction,
                    "expected_accepted": expected_n,
                    "expected_tp": expected_tp,
                    "precision": target,
                    "recall": expected_tp / int(y.sum()),
                }
            )
    chosen = np.asarray(scores) >= best["cutoff"]
    return {
        "target_precision": target,
        "attainable": best,
        "metrics": measures(y, chosen),
        "exact_threshold": exact_point,
        "randomised_exact": max(crossings, key=lambda p: p["recall"])
        if crossings
        else None,
        "interpretation": "Within-split descriptive selection using labels; not a holdout deployment estimate.",
    }


def paired_bootstrap(rows, panel, jev):
    """Keep copy groups paired and policies fixed across bootstrap draws."""
    buckets = {}
    for i, row in enumerate(rows):
        buckets.setdefault(row["copy_group"], []).append(i)
    groups = list(buckets.values())
    y = np.array([r["label"] for r in rows])
    rng = np.random.default_rng(SEED)
    samples = {key: [] for key in METRICS}
    for _ in range(RESAMPLES):
        idx = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        pm, jm = measures(y[idx], panel[idx]), measures(y[idx], jev[idx])
        for key in METRICS:
            if pm[key]["point"] is not None and jm[key]["point"] is not None:
                samples[key].append(jm[key]["point"] - pm[key]["point"])
    pm, jm = measures(y, panel), measures(y, jev)
    return {
        key: {
            "point": jm[key]["point"] - pm[key]["point"]
            if pm[key]["point"] is not None and jm[key]["point"] is not None
            else None,
            "interval_95": np.quantile(values, [0.025, 0.975]).tolist()
            if values
            else None,
            "valid_resamples": len(values),
        }
        for key, values in samples.items()
    }


def input_rows():
    sources = {
        "name_receipts": calibration.DEFAULT_NAME_RECEIPTS,
        "name_context": calibration.DEFAULT_NAME_RECEIPTS.parent
        / "evidence-context.json",
        "name_golden": calibration.GOLDEN_NAME,
        "docs_receipts": calibration.DEFAULT_DOCS_RECEIPTS,
        "docs_golden": calibration.GOLDEN_DOCS,
        "docs_panel": HERE / "srtj-docs-measure-results.json",
        "name_fit": ROOT / "imas_codex/standard_names/review/cascade_thresholds.json",
        "docs_fit": HERE / "srtj-docs-score-arm-results.json",
    }
    name_context = read(sources["name_context"])["entries"]
    name_receipts = {
        r["name"]: r
        for r in map(json.loads, sources["name_receipts"].read_text().splitlines())
        if r.get("mode") == "base" and r.get("answered")
    }
    docs_package = read(sources["docs_receipts"])
    prior = read(sources["docs_panel"])
    docs_panel = json.loads(
        lzma.decompress(base64.b64decode(prior["frozen_lzma_b64"]))
    )["context"]
    result, coverage, examples = {}, {}, {}
    for axis, builder in (
        ("name", calibration._name_axis),
        ("docs", calibration._docs_axis),
    ):
        golden = read(sources[axis + "_golden"])
        canonical = {}
        for row in golden:
            identity = row["canonical_unit"]
            if identity in canonical:
                assert (row["label"], row["split"], row["copy_group"]) == (
                    canonical[identity]["label"],
                    canonical[identity]["split"],
                    canonical[identity]["copy_group"],
                )
            canonical.setdefault(identity, row)
        scored, _ = builder(sources[axis + "_receipts"], sources[axis + "_golden"])
        scored_by_id = {r["id"]: r for r in scored}
        assert len(scored_by_id) == len(scored)
        excluded, kept = [], []
        for identity, gold in canonical.items():
            row = scored_by_id.get(identity)
            reason = None
            if row is None:
                reason = "no_jev_score"
            elif axis == "name":
                entry = name_context[identity]
                state = name_receipts[identity]["state"]["name"]
                assert state["id"] == identity
                if any(
                    state.get(k) != entry.get(k)
                    for k in ("description", "kind", "unit", "documentation")
                ):
                    reason = "name_snapshot_mismatch"
                elif entry.get("quorum") is None:
                    reason = "no_panel_score"
                elif (
                    entry.get("quorum_method")
                    not in {"quorum_consensus", "authoritative_escalation"}
                    or entry.get("quorum_shortfall") is not None
                ):
                    reason = "unverified_panel_resolution"
                panel_score = entry.get("quorum")
            else:
                state = docs_package["receipts"]["base:" + identity]["state"]
                assert (
                    hashlib.sha256(gold["documentation"].encode()).hexdigest()
                    == gold["source_evidence"]
                )
                if (
                    not docs_panel["by_name"][identity]["document_match"]
                    or state["documentation"] != gold["documentation"]
                    or state["description"] != gold["description"]
                ):
                    reason = "documentation_revision_mismatch"
                elif docs_panel["by_name"][identity]["score"] is None:
                    reason = "no_panel_score"
                panel_score = docs_panel["by_name"][identity]["score"]
            if reason:
                excluded.append(
                    {"id": identity, "split": gold["split"], "reason": reason}
                )
                continue
            assert row["label"] == gold["label"] and row["split"] == gold["split"]
            kept.append(
                {
                    "id": identity,
                    "label": gold["label"],
                    "split": gold["split"],
                    "copy_group": gold["copy_group"],
                    "jev": row["score"],
                    "panel": panel_score,
                }
            )
        assert len(kept) + len(excluded) == len(canonical)
        assert any(r["panel"] >= 0.85 for r in kept)
        assert len({r["jev"] for r in kept}) > 10
        result[axis] = kept
        coverage[axis] = {
            "golden_rows": len(golden),
            "canonical_items": len(canonical),
            "jev_scored": len(scored),
            "paired": len(kept),
            "exclusions": excluded,
            "counts_by_reason": dict(Counter(e["reason"] for e in excluded)),
            "revision_mismatches": sum("mismatch" in e["reason"] for e in excluded),
            "historical_panel_input_hashes_available": False,
            "all_jev_auc": {
                split: evaluation.roc_auc(
                    [r["label"] for r in scored if r["split"] == split],
                    [r["score"] for r in scored if r["split"] == split],
                )
                for split in SPLITS
            },
        }
        for split in SPLITS:
            assert sum(r["split"] == split for r in kept) + sum(
                r["split"] == split for r in excluded
            ) == sum(r["split"] == split for r in canonical.values())
        examples[axis] = []
        for row in sorted(kept, key=lambda r: abs(r["jev"] - r["panel"]), reverse=True)[
            :3
        ]:
            identity = row["id"]
            description = (
                name_context[identity]["description"]
                if axis == "name"
                else canonical[identity]["description"]
            )
            bindings = (
                name_context[identity].get("source_paths") or []
                if axis == "name"
                else [r["path"] for r in canonical[identity]["dd_sources"]]
            )
            examples[axis].append(
                {**row, "description": description, "source_paths": bindings}
            )
        print(
            "INPUT",
            axis,
            len(canonical),
            len(scored),
            len(kept),
            coverage[axis]["counts_by_reason"],
            flush=True,
        )
    assert coverage["name"]["jev_scored"] == 701
    assert coverage["docs"]["jev_scored"] == 400
    assert coverage["name"]["paired"] == 542
    assert coverage["docs"]["paired"] == 386
    assert (
        abs(coverage["name"]["all_jev_auc"]["validation"] - 0.7452455825097334) < 1e-6
    )
    assert (
        abs(coverage["docs"]["all_jev_auc"]["validation"] - 0.7525798525798526) < 1e-9
    )
    assert "flux_surface_averaged_total_ion_density_at_plasma_boundary" in {
        e["id"]
        for e in coverage["docs"]["exclusions"]
        if e["reason"] == "documentation_revision_mismatch"
    }
    assert docs_panel["known_present"]["score"] == 1.0
    return (
        result,
        coverage,
        examples,
        {
            key: {"path": str(path), "sha256": digest(path)}
            for key, path in sources.items()
        },
    )


def cohort_costs():
    result = {}
    for axis in ("name", "docs"):
        rows, files, total_spend = [], [], 0.0
        for path in sorted((HERE / "srtj-cohort").glob(f"incumbent-{axis}-part*.json")):
            record = read(path)
            assert math.isclose(
                sum(row["spend_usd"] for row in record["items"]),
                record["spend_usd"],
                abs_tol=1e-9,
            ), f"cohort item costs do not reconcile: {path}"
            successful = [
                r
                for r in record["items"]
                if r["failure_reason"] is None
                and r.get("score") is not None
                and r["decision"] in ("accepted", "reviewed")
            ]
            files.append(
                {
                    "path": str(path),
                    "sha256": digest(path),
                    "status": record["status"],
                    "item_count": len(record["items"]),
                    "completed_items": len(successful),
                    "spend_usd": record["spend_usd"],
                }
            )
            rows.extend(successful)
            total_spend += record["spend_usd"]
        assert len({r["id"] for r in rows}) == len(rows)
        assert rows and all(r["spend_usd"] > 0 for r in rows)
        costs = [r["spend_usd"] for r in rows]
        result[axis] = {
            "completed_items": len(rows),
            "completed_item_cost_usd": sum(costs),
            "mean_usd": sum(costs) / len(rows),
            "median_usd": float(np.median(costs)),
            "all_recorded_spend_usd": total_spend,
            "all_recorded_spend_per_completion_usd": total_spend / len(rows),
            "files": files,
            "caveat": "Observed completed cohort items, not golden items. Docs pieces ended incomplete; cost conditional on completion is selection-biased.",
        }
    return result


def analyse(rows):
    y = [r["label"] for r in rows]
    panel = np.array([r["panel"] >= DEFAULT_MIN_SCORE for r in rows])
    jev, cutoff = same_rate(
        [r["jev"] for r in rows], [r["id"] for r in rows], int(panel.sum())
    )
    pm, jm = measures(y, panel), measures(y, jev)
    tied = [r for r in rows if r["jev"] == cutoff["cutoff"]]
    outside_tp = sum(r["label"] for r in rows if r["jev"] > cutoff["cutoff"])
    good_ties = sum(r["label"] for r in tied)
    selected = cutoff["boundary_ties_accepted"]
    min_tp = outside_tp + max(0, selected - (len(tied) - good_ties))
    max_tp = outside_tp + min(selected, good_ties)
    cutoff["boundary_acceptable"] = good_ties
    cutoff["acceptable_accepts_range_over_tie_orders"] = [min_tp, max_tp]
    cutoff["accept_precision_range_over_tie_orders"] = [
        min_tp / int(panel.sum()),
        max_tp / int(panel.sum()),
    ]
    cutoff["expected_acceptable_accepts_uniform_tie"] = (
        outside_tp + selected * good_ties / len(tied)
    )
    check_tallies(rows, panel, pm)
    check_tallies(rows, jev, jm)
    same_precision = precision_target(
        y, [r["jev"] for r in rows], pm["accept_precision"]["point"]
    )
    return {
        "n": len(rows),
        "acceptable": sum(y),
        "defective": len(y) - sum(y),
        "copy_groups": len({r["copy_group"] for r in rows}),
        "panel": pm,
        "jev_same_rate": jm,
        "same_rate_threshold": cutoff,
        "paired_difference": paired_bootstrap(rows, panel, jev),
        "jev_at_panel_precision": same_precision,
        "auc": {
            "panel": evaluation.roc_auc(y, [r["panel"] for r in rows]),
            "jev": evaluation.roc_auc(y, [r["jev"] for r in rows]),
        },
    }


def pct(x):
    return "undefined" if x is None else f"{100 * x:.2f}%"


def metric_text(value):
    interval = value["wilson_95"]
    ci = f"[{pct(interval[0])}, {pct(interval[1])}]" if interval else "undefined"
    return f"{value['numerator']}/{value['denominator']} = {pct(value['point'])} {ci}"


def difference_text(value):
    if value["point"] is None:
        return "undefined (no refined panel items)"
    interval = value["interval_95"]
    return f"{100 * value['point']:+.2f} [{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}] pp"


def save_checked(fig, path):
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        (artist.get_text(), artist.get_window_extent(renderer))
        for artist in fig.findobj(Text)
        if artist.get_visible() and artist.get_text()
    ]
    for label, box in boxes:
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(
            box.x1, box.y1
        ), f"clipped figure label: {label}"
    for i, (label, box) in enumerate(boxes):
        for other, bounds in boxes[i + 1 :]:
            assert not box.overlaps(bounds), (
                f"overlapping figure labels: {label}, {other}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None})
    print("FIGURE text bounds and pairwise overlap checks passed", flush=True)


def figure(report, rows, run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "reviewer-operating-points"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.08, top=0.87, hspace=0.43, wspace=0.25
    )
    for i, axis in enumerate(("name", "docs")):
        for j, split in enumerate(SPLITS):
            ax = axes[i, j]
            sample = [r for r in rows[axis] if r["split"] == split]
            labels = np.array([r["label"] for r in sample])
            for judge, color, style, zorder in (
                ("panel", PANEL, "--", 2),
                ("jev", JEV, "-", 3),
            ):
                scores = np.array([r[judge] for r in sample])
                points = []
                for cutoff in sorted(set(scores), reverse=True):
                    accepted = scores >= cutoff
                    points.append((accepted.mean(), labels[accepted].mean()))
                ax.plot(
                    *zip(*points, strict=True),
                    color=color,
                    linestyle=style,
                    linewidth=3 if judge == "jev" else 2.6,
                    zorder=zorder,
                )
            block = report["axes"][axis][split]
            for judge, marker, color in (
                ("panel", "o", PANEL),
                ("jev_same_rate", "D", JEV),
            ):
                value = block[judge]
                ax.scatter(
                    value["accept_rate"]["point"],
                    value["accept_precision"]["point"],
                    marker=marker,
                    facecolors="none" if judge == "panel" else color,
                    edgecolors=color,
                    linewidths=2 if judge == "panel" else 1,
                    s=230 if judge == "panel" else 60,
                    zorder=4 if judge == "panel" else 5,
                )
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.03)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_xlabel("accept rate")
            ax.set_ylabel("accept precision")
            ax.text(
                0,
                1.10,
                f"{'Names' if axis == 'name' else 'Documentation'} · {split}",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
    fig.legend(
        handles=[
            Line2D([], [], color=JEV, linewidth=3, label="Jev logistic"),
            Line2D(
                [], [], color=PANEL, linewidth=2.6, linestyle="--", label="paid panel"
            ),
            Line2D(
                [],
                [],
                color=PANEL,
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markersize=11,
                linestyle="",
                label="panel at 0.85",
            ),
            Line2D(
                [], [], color=JEV, marker="D", linestyle="", label="Jev at matched rate"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    path = (
        ROOT / "docs/figures/sn-review-through-jev" / STEM / "precision-acceptance.svg"
    )
    save_checked(fig, path)
    preview = run / "precision-acceptance.png"
    fig.savefig(preview, dpi=100)
    plt.close(fig)
    report["figure"] = {
        "svg": str(path),
        "preview": str(preview),
        "mechanical_check": "all visible text within canvas; no text overlaps; no direct labels on data",
    }


def verdict(report):
    pieces = []
    for axis in ("name", "docs"):
        block = report["axes"][axis]["validation"]
        delta = block["paired_difference"]["accept_precision"]
        state = (
            "worse than Jev"
            if delta["interval_95"][0] > 0
            else "better than Jev"
            if delta["interval_95"][1] < 0
            else "not demonstrably different from Jev"
        )
        extra = (
            block["jev_same_rate"]["accept_precision"]["numerator"]
            - block["panel"]["accept_precision"]["numerator"]
        )
        pieces.append(
            f"{'Names' if axis == 'name' else 'Documentation'}: at the panel's "
            f"{pct(block['panel']['accept_rate']['point'])} accept rate, the panel is {state}; "
            f"Jev accepts {extra:+d} more acceptable items among {block['panel']['accept_rate']['numerator']} "
            f"accepts, precision difference {difference_text(delta)}. "
            f"Accept recall is {pct(block['jev_same_rate']['accept_recall']['point'])} versus "
            f"{pct(block['panel']['accept_recall']['point'])}; refine recall is "
            f"{pct(block['jev_same_rate']['refine_recall']['point'])} versus "
            f"{pct(block['panel']['refine_recall']['point'])}. "
            f"Observed panel cost is USD {report['costs'][axis]['mean_usd']:.6f} per completed cohort item "
            f"(n={report['costs'][axis]['completed_items']})."
        )
    return " ".join(pieces)


def fragment(report):
    out = [
        f'<section id="{STEM}">',
        "<h2>Jev and the paid panel at matched operating points</h2>",
        f"<p><strong>{html.escape(report['verdict'])}</strong></p>",
        "<p>LLM spend for this measurement: USD 0; no calls or graph access. "
        "Accept means score ≥ 0.85 for the panel; refine means its binary complement. "
        "This measures score decisions, not the three-way cascade with chat routing.</p>",
        "<h3>Paired populations and score provenance</h3>",
        "<p>The Jev score is the landed logistic on each axis, including the documentation "
        "contradiction hard zero. No model is fitted on validation. Both judges in each row "
        "see the same retained identities; the calibration name fit is in-sample. "
        "Same-rate and same-precision thresholds are selected separately within each split "
        "for this descriptive comparison. They are not deployment cutoffs.</p>",
        "<p>Panel name input is checked against the frozen Jev name, description, kind, unit "
        "and documentation. The panel receipts have no historical input hashes: snapshot "
        "agreement cannot prove that an older review saw unchanged context. Documentation "
        "requires the frozen panel document-match flag and Jev text equal to golden text. "
        "Known revision mismatches are excluded, not silently treated as bad labels.</p>",
        "<table><thead><tr><th>Axis</th><th>Golden unique / rows</th><th>Jev scored</th><th>Paired</th><th>Excluded</th></tr></thead><tbody>",
    ]
    for axis, coverage in report["coverage"].items():
        out.append(
            f"<tr><td>{axis}</td><td>{coverage['canonical_items']} / {coverage['golden_rows']}</td>"
            f"<td>{coverage['jev_scored']}</td><td>{coverage['paired']}</td>"
            f"<td>{html.escape(str(coverage['counts_by_reason']))}</td></tr>"
        )
    out += [
        "</tbody></table>",
        "<p>All metric intervals are nominal 95% Wilson intervals with their actual denominator. "
        "Paired differences are Jev minus panel, using 10,000 seeded copy-group bootstrap draws "
        "with both policies held fixed. Counts were independently checked by identity-set "
        "intersections. A zero denominator is undefined, never zero precision. "
        "Replicates with undefined ratios are omitted for that metric; valid counts are in the JSON. "
        "A degenerate bootstrap interval on a uniform observed outcome is conditional on these "
        "data, not proof of population certainty. Grouped difference intervals address dependence; "
        "marginal Wilson intervals do not.</p>",
        f'<figure id="{STEM}-figure"><img src="/imas-codex/figures/sn-review-through-jev/{STEM}/precision-acceptance.svg" '
        'alt="Accept precision against accept rate for Jev and panel on calibration and validation, per axis.">'
        "<figcaption>Pure score-threshold curves join attainable points. A grey circle marks the "
        "panel's 0.85 cutoff; a teal diamond marks Jev accepting the same count. "
        "A diamond inside a score tie uses the stated label-blind identity ordering. "
        "The split and axis labels identify each panel. These curves use golden labels, not panel verdicts.</figcaption></figure>",
    ]
    for split in ("validation", "calibration"):
        out.append(f"<h3>{split.capitalize()}</h3>")
        out.append(
            "<table><thead><tr><th>Axis / metric</th><th>Panel at 0.85 [95%]</th>"
            "<th>Jev at same accept rate [95%]</th><th>Jev − panel [paired 95%]</th></tr></thead><tbody>"
        )
        for axis in ("name", "docs"):
            b = report["axes"][axis][split]
            for key in METRICS:
                out.append(
                    f"<tr><td>{axis} / {key.replace('_', ' ')}</td><td>{metric_text(b['panel'][key])}</td>"
                    f"<td>{metric_text(b['jev_same_rate'][key])}</td>"
                    f"<td>{difference_text(b['paired_difference'][key])}</td></tr>"
                )
        out.append("</tbody></table>")
        for axis in ("name", "docs"):
            b = report["axes"][axis][split]
            cutoff = b["same_rate_threshold"]
            out.append(
                f"<p>{axis}: n={b['n']}, acceptable={b['acceptable']}, defective={b['defective']}; "
                f"paired AUC Jev {b['auc']['jev']:.6f}, panel {b['auc']['panel']:.6f}. "
                f"Matched-rate Jev cutoff {cutoff['cutoff']:.12g}; boundary ties accepted "
                f"{cutoff['boundary_ties_accepted']}/{cutoff['boundary_ties']}. "
                f"Pure threshold: {cutoff['pure_threshold']}.</p>"
            )
            if not cutoff["pure_threshold"]:
                bounds = cutoff["accept_precision_range_over_tie_orders"]
                out.append(
                    f"<p>The boundary contains {cutoff['boundary_acceptable']} acceptable items. "
                    f"Every label-blind boundary ordering has accept precision within "
                    f"[{pct(bounds[0])}, {pct(bounds[1])}]; uniform random tie selection gives "
                    f"{cutoff['expected_acceptable_accepts_uniform_tie']:.3f} expected acceptable accepts. "
                    "The reported integer result uses SHA256('acceptance-order:' + identity). "
                    "Here the cutoff is zero: matching the panel's rate would override Jev's "
                    "contradiction veto. This is a score-ranking comparison only, not a proposed route.</p>"
                )
            p = b["jev_at_panel_precision"]
            if p["attainable"]:
                a = p["attainable"]
                out.append(
                    f"<p>At the panel precision target {pct(p['target_precision'])}, the "
                    f"maximum-recall attainable Jev threshold is {a['cutoff']:.12g}: "
                    f"accept precision {metric_text(p['metrics']['accept_precision'])}; "
                    f"accept recall {metric_text(p['metrics']['accept_recall'])}. "
                    f"An exactly equal deterministic precision is {'attainable' if p['exact_threshold'] else 'not attainable'}.</p>"
                )
                if p["randomised_exact"]:
                    x = p["randomised_exact"]
                    out.append(
                        f"<p>An exact precision match in expectation is possible at cutoff {x['cutoff']:.12g} "
                        f"by accepting each boundary-score item with probability {x['boundary_accept_probability']:.6f}: "
                        f"expected acceptable accepts {x['expected_tp']:.3f}/{x['expected_accepted']:.3f}, "
                        f"recall {pct(x['recall'])}. These are fractional expected counts, "
                        "not an observed deterministic result or a calibrated production policy.</p>"
                    )
                elif p["exact_threshold"]:
                    exact = p["exact_threshold"]
                    out.append(
                        f"<p>Exact deterministic match: threshold {exact['cutoff']:.12g}, "
                        f"accept precision {exact['tp']}/{exact['accepted']} = {pct(exact['precision'])}, "
                        f"accept recall {exact['tp']}/{b['acceptable']} = {pct(exact['recall'])}.</p>"
                    )
                else:
                    out.append(
                        "<p>No adjacent threshold mixture attains the target exactly: "
                        "the maximum-recall threshold already exceeds the panel precision. "
                        "The attainable result above is the comparison at no worse precision.</p>"
                    )
                if a["cutoff"] == 0 or (
                    p["randomised_exact"] and p["randomised_exact"]["cutoff"] == 0
                ):
                    out.append(
                        "<p>The precision-matching result at zero also accepts contradiction-veto "
                        "items. It is not attainable while preserving the deployed hard veto.</p>"
                    )
    out += [
        "<h3>Agreement between splits and limitations</h3>",
        f"<p>{html.escape(report['split_agreement'])}</p>",
        "<p>The name validation cohort has been exposed repeatedly; prior evidence-refresh work "
        "did not improve its discrimination. These intervals measure sampling variation conditional "
        "on the frozen scores, not model drift or tuning uncertainty. Equal rate can erase a "
        "ranking advantage if both judges accept almost everything. The paid panel is not a "
        "gold label, and this comparison does not authorise switching on the cascade.</p>",
        "<p>The published name AUC near 0.745 uses all 338 scored validation items. On the "
        "259 identities with verified panel resolution it is 0.764812 versus panel 0.726195; "
        "0.706 is not the matched frozen-panel AUC reproduced here. Documentation uses 176 "
        "paired validation items: Jev 0.753739 versus panel 0.514156.</p>",
        "<h3>Observed paid-panel costs</h3>",
    ]
    for axis, cost in report["costs"].items():
        out.append(
            f"<p>{axis}: USD {cost['completed_item_cost_usd']:.6f} / {cost['completed_items']} "
            f"completed items = USD {cost['mean_usd']:.6f} per item; median USD {cost['median_usd']:.6f}. "
            f"All available piece spend is USD {cost['all_recorded_spend_usd']:.6f}, or "
            f"USD {cost['all_recorded_spend_per_completion_usd']:.6f} per completion including incomplete-item spend. "
            f"{html.escape(cost['caveat'])} These are observed receipts, not the earlier cost forecast.</p>"
        )
    out.append("<h3>Representative same-identity disagreements</h3>")
    for axis, examples in report["examples"].items():
        for row in examples:
            out.append(
                f"<p>{axis}: <code>{html.escape(row['id'])}</code> — {html.escape(row['description'])} "
                f"Golden {'acceptable' if row['label'] else 'defective'}; panel {row['panel']:.4f}, "
                f"Jev {row['jev']:.4f}. Source bindings: "
                f"{html.escape(', '.join(row['source_paths']) or 'none recorded')}.</p>"
            )
    out += [
        "<h3>Reproducibility</h3>",
        f'<p><a href="/imas-codex/evidence/fragments/sn-review-through-jev/{STEM}.json">Full JSON with exclusions, '
        f"per-item scores, input digests and counts</a>; "
        f'<a href="/imas-codex/evidence/fragments/sn-review-through-jev/{STEM}-run.py">Offline driver</a>. '
        f"Gate log: <code>{html.escape(report['gate_log'])}</code>. "
        "Positive controls reproduce the frozen AUCs and known scored item. "
        "A deliberately corrupted tally is refused; the real documentation revision mismatch is detected. "
        "SVG text bounds and overlaps are checked on save; the PNG preview is inspected before delivery.</p>",
        "</section>",
    ]
    (HERE / (STEM + ".html")).write_text("\n".join(out) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"# revision: {revision}; tree: {ROOT}; command: {shlex.join(sys.argv)}",
        flush=True,
    )
    print(f"# measurement_module: {evaluation.__file__}", flush=True)
    print(f"# measurement_cwd: {Path.cwd().resolve()}", flush=True)
    assert DEFAULT_MIN_SCORE == 0.85
    assert os.environ.get("SLURM_JOB_ID"), "measurement must run on compute"
    probe = [{"id": "acceptable", "label": 1}, {"id": "defective", "label": 0}]
    correct = measures([1, 0], [True, False])
    check_tallies(probe, [True, False], correct)
    corrupt = json.loads(json.dumps(correct))
    corrupt["accept_precision"]["numerator"] = 0
    try:
        check_tallies(probe, [True, False], corrupt)
    except AssertionError as exc:
        print(f"CONTROL refused corrupt tally: {exc}", flush=True)
    else:
        raise AssertionError("corrupt tally was not refused")
    chosen, tie = same_rate([0.0, 0.0], ["acceptable", "defective"], 1)
    assert chosen.sum() == 1 and not tie["pure_threshold"]
    rows, coverage, examples, sources = input_rows()
    report = {
        "schema": "paired-review-operating-points",
        "score": "landed logistic, documentation contradiction hard zero retained",
        "panel_threshold": DEFAULT_MIN_SCORE,
        "bootstrap": {
            "resamples": RESAMPLES,
            "seed": SEED,
            "unit": "copy_group",
            "thresholds": "fixed after within-split descriptive selection",
        },
        "coverage": coverage,
        "inputs": sources,
        "examples": examples,
        "costs": cohort_costs(),
        "axes": {},
        "gate_log": str(run / "measurement.log"),
        "llm_spend_usd": 0,
    }
    for axis in ("name", "docs"):
        report["axes"][axis] = {}
        for split in SPLITS:
            report["axes"][axis][split] = analyse(
                [r for r in rows[axis] if r["split"] == split]
            )
            print("MEASURED", axis, split, flush=True)
    report["verdict"] = verdict(report)
    agreement = []
    for axis in ("name", "docs"):
        delta = [
            report["axes"][axis][s]["paired_difference"]["accept_precision"]
            for s in SPLITS
        ]
        significant = [d["interval_95"][0] > 0 for d in delta]
        agreement.append(
            f"{axis}: calibration precision difference {difference_text(delta[0])}; "
            f"validation {difference_text(delta[1])}. "
            + (
                "Both splits show no statistically clear separation at the panel's rate."
                if not any(significant)
                else "The direction favours Jev on both splits, but calibration's clear gain "
                "does not persist as a statistically clear validation gain."
            )
        )
    report["split_agreement"] = " ".join(agreement)
    report["paired_items"] = {
        axis: [
            [r[k] for k in ("id", "label", "split", "copy_group", "jev", "panel")]
            for r in value
        ]
        for axis, value in rows.items()
    }
    report["paired_item_columns"] = [
        "id",
        "golden_label",
        "split",
        "copy_group",
        "jev_score",
        "panel_score",
    ]
    target = HERE / (STEM + ".json")
    if args.verify_only:
        saved = read(target)
        assert saved["axes"] == report["axes"]
        assert saved["coverage"] == report["coverage"]
        assert saved["inputs"] == report["inputs"]
        assert saved["costs"] == report["costs"]
        assert saved["paired_items"] == report["paired_items"]
        print(
            "REPLAY committed metrics, counts, inputs and per-item scores identical",
            flush=True,
        )
    else:
        figure(report, rows, run)
        encoded = json.dumps(report, separators=(",", ":"), allow_nan=False)
        assert len(encoded.encode()) <= 300000
        target.write_text(encoded + "\n")
        fragment(report)
    print("SUMMARY " + report["verdict"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        print("EXIT=1", flush=True)
        sys.exit(1)
    print("EXIT=0", flush=True)

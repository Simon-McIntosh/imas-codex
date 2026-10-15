#!/usr/bin/env python
"""Which name-review questions carry signal, read from the saved Jev answers.

The prototype name review scored AUC 0.515 on calibration and 0.560 on
validation, and its calibration cut accepted no validation name.  This script
diagnoses that shortfall without spending anything: it reads the saved per-name
decisions, asks how well each individual question separates the golden labels
on the calibration split, then reads validation once per frozen aggregation.

Nothing here calls Jev, an LLM or the graph.  The saved answers, the golden
labels and the quorum scores are read from disk; the metric primitives, the
question table and the prototype's own score are imported rather than copied.

Every choice is fixed on the calibration split before validation is read, and
validation is read exactly once per aggregation.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "sn-judgment-evaluation" / "golden-name-acceptable.json"
RESULTS = HERE / "srtj-name-signal-diagnosis-results.json"
FRAGMENT = HERE / "srtj-name-signal-diagnosis.html"
FIGURES = (
    HERE.parents[2] / "figures" / "sn-review-through-jev" / "srtj-name-signal-diagnosis"
)

SAVED_RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261015T063535477621-srtj-name-review-measure"
)
DECISIONS = SAVED_RUN / "name-decisions.jsonl"
CONTEXT = SAVED_RUN / "name-context.json"

CONTROL_CALIBRATION = 0.515
CONTROL_VALIDATION = 0.560
CONTROL_TOLERANCE = 0.001
QUORATE_METHODS = frozenset({"quorum_consensus", "authoritative_escalation"})
N_RESAMPLES = 1000
RESAMPLE_SEED = 123
CUT = 0.870819
DIMENSIONS = ("grammar", "semantic", "convention", "completeness")
LABEL_COLOR = {0: "#a2643c", 1: "#2b6b78"}
LABEL_NAME = {0: "rejected (label 0)", 1: "accepted (label 1)"}
AGGREGATION_LABELS = (
    ("existing_score", "Existing prototype score (control)"),
    ("selected_mean", "Mean of selected questions, sign-aligned"),
    ("logistic_l2", "L2 logistic regression over all questions"),
)


def _load_saved_answers() -> dict[str, dict] | dict:
    answers: dict[str, dict] = {}
    for line in DECISIONS.read_text().splitlines():
        row = json.loads(line)
        if row["mode"] != "base" or not row["answered"]:
            continue
        answers[row["name"]] = row["answers"]
    return answers


def _prototype():
    """Import the measurement script so its score and loader are reused."""
    os.environ.setdefault("RECKON_MEASURE_RUN", str(SAVED_RUN))
    spec = importlib.util.spec_from_file_location(
        "_srtj_measure", HERE / "srtj-name-review-measure-run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _question_table(checks):
    from imas_codex.standard_names.review.jev_review import name_question_table

    return name_question_table(checks)


def _accepted_dimension(table):
    """The accepting category the taxonomy authors first for each dimension."""
    return {
        dimension: next(iter(table[f"dimension:{dimension}"]["criteria"]))
        for dimension in DIMENSIONS
    }


def _question_values(answers, checks, accepted_dimension) -> dict[str, float | None]:
    """Each question's acceptance value for one name, or None where unanswered."""
    values: dict[str, float | None] = {}
    for dimension, accepted in accepted_dimension.items():
        key = f"dimension:{dimension}"
        answer = answers.get(key)
        if not answer:
            values[key] = None
            continue
        values[key] = float((answer.get("probabilities") or {}).get(accepted, 0.0))
    for check in checks:
        if check.question is None:
            continue
        key = f"check:{check.id}"
        answer = answers.get(key)
        if not answer:
            values[key] = None
            continue
        if check.question.kind.value == "noul":
            values[key] = 1.0 - float(answer["noul"])
        else:
            acceptable = check.question.options[0]
            values[key] = float(
                (answer.get("probabilities") or {}).get(acceptable, 0.0)
            )
    return values


def _build_rows(fixture, context, answers, checks, table, accepted_dimension):
    canonical: dict[str, dict] = {}
    for row in fixture:
        canonical.setdefault(row["canonical_unit"], row)
    rows = []
    for name, row in canonical.items():
        entry = context["entries"][name]
        answer = answers.get(name)
        if not entry["description"] or answer is None:
            continue
        quorum = (
            float(entry["quorum"])
            if entry.get("quorum") is not None
            and entry.get("quorum_method") in QUORATE_METHODS
            and entry.get("quorum_shortfall") is None
            else None
        )
        rows.append(
            {
                "name": name,
                "label": int(row["label"]),
                "split": row["split"],
                "copy_group": row["copy_group"],
                "description": entry["description"],
                "quorum": quorum,
                "values": _question_values(answer, checks, accepted_dimension),
            }
        )
    return rows


def _bootstrap(labels, scores, groups, scores_b=None):
    from imas_codex.discovery.base.evaluation import bootstrap_interval

    return bootstrap_interval(
        labels,
        scores,
        scores_b=scores_b,  # type: ignore[arg-type]
        copy_groups=groups,
        n_resamples=N_RESAMPLES,
        seed=RESAMPLE_SEED,
    )


def _per_question(cal_rows, question_names):
    reports = []
    for key in question_names:
        present = [row for row in cal_rows if row["values"].get(key) is not None]
        labels = [row["label"] for row in present]
        values = [row["values"][key] for row in present]
        groups = [row["copy_group"] for row in present]
        report: dict = {
            "question": key,
            "n": len(present),
            "missing": len(cal_rows) - len(present),
            "values": {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            },
        }
        if len(set(labels)) < 2:
            report.update({"auc": None, "excludes_half": False, "inverted": False})
        else:
            auc = _bootstrap(labels, values, groups)
            report["auc"] = auc
            report["excludes_half"] = auc["lower"] > 0.5 or auc["upper"] < 0.5
            report["inverted"] = auc["point"] < 0.5
        reports.append(report)
    return reports


def _mean_over_selected(rows, selected, aligned_signs):
    """One mean per row over the selected questions, each sign-aligned."""
    scores: dict[str, float] = {}
    for row in rows:
        picked = [row["values"][key] for key in selected]
        if any(value is None for value in picked):
            continue
        aligned = [
            value if aligned_signs[key] else 1.0 - value
            for key, value in zip(selected, picked, strict=True)
        ]
        scores[row["name"]] = float(np.mean(aligned))
    return scores


def _fit_logistic(cal_rows, question_names):
    """L2 logistic regression, regularisation chosen by grouped CV inside calibration."""
    from sklearn import set_config
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold

    set_config(enable_metadata_routing=True)

    complete = [
        row
        for row in cal_rows
        if all(row["values"].get(key) is not None for key in question_names)
    ]
    features = np.asarray(
        [[row["values"][key] for key in question_names] for row in complete],
        dtype=float,
    )
    labels = np.asarray([row["label"] for row in complete], dtype=int)
    groups = np.asarray([row["copy_group"] for row in complete])
    model = LogisticRegressionCV(
        Cs=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        cv=GroupKFold(n_splits=5),
        l1_ratios=(0.0,),
        solver="lbfgs",
        scoring="roc_auc",
        max_iter=5000,
        random_state=RESAMPLE_SEED,
        use_legacy_attributes=True,
    )
    model.fit(features, labels, groups=groups)
    return model


def _logistic_scores(model, rows, question_names):
    usable = [
        row
        for row in rows
        if all(row["values"].get(key) is not None for key in question_names)
    ]
    if not usable:
        return {}
    features = np.asarray(
        [[row["values"][key] for key in question_names] for row in usable], dtype=float
    )
    probabilities = model.predict_proba(features)[:, 1]
    return {
        row["name"]: float(probability)
        for row, probability in zip(usable, probabilities, strict=True)
    }


def _aggregate_report(rows, score_by_name):
    labels = [row["label"] for row in rows if row["name"] in score_by_name]
    scores = [
        score_by_name[row["name"]] for row in rows if row["name"] in score_by_name
    ]
    groups = [row["copy_group"] for row in rows if row["name"] in score_by_name]
    if not scores:
        return {"n": 0, "auc": None}
    report = {"n": len(scores), "auc": _bootstrap(labels, scores, groups)}
    return report


def _quorum_comparison(rows, score_by_name):
    from imas_codex.discovery.base.evaluation import bootstrap_interval, paired_auc

    matched = [
        row
        for row in rows
        if row["quorum"] is not None and row["name"] in score_by_name
    ]
    if len({row["label"] for row in matched}) < 2:
        return {}
    y = [row["label"] for row in matched]
    a = [score_by_name[row["name"]] for row in matched]
    b = [row["quorum"] for row in matched]
    copy = [row["copy_group"] for row in matched]
    return {
        "quorum_matched_n": len(matched),
        "quorum_paired": paired_auc(y, a, b),
        "quorum_difference_interval": bootstrap_interval(
            y,
            a,
            scores_b=b,
            copy_groups=copy,
            n_resamples=N_RESAMPLES,
            seed=RESAMPLE_SEED,
        ),
    }


def _control_check(cal_report, val_report):
    cal = cal_report["auc"]["point"]
    val = val_report["auc"]["point"]
    return {
        "calibration": cal,
        "validation": val,
        "targets": {
            "calibration": CONTROL_CALIBRATION,
            "validation": CONTROL_VALIDATION,
            "tolerance": CONTROL_TOLERANCE,
        },
        "reproduces": abs(cal - CONTROL_CALIBRATION) <= CONTROL_TOLERANCE
        and abs(val - CONTROL_VALIDATION) <= CONTROL_TOLERANCE,
    }


def _ten_rejected(val_rows, score_by_name, question_names):
    rejected = [row for row in val_rows if row["label"] == 0]
    rejected.sort(key=lambda row: score_by_name[row["name"]], reverse=True)
    picked = []
    for row in rejected[:10]:
        ranked = sorted(
            (
                (key, row["values"][key])
                for key in question_names
                if row["values"].get(key) is not None
            ),
            key=lambda item: item[1],
        )[:3]
        picked.append(
            {
                "name": row["name"],
                "description": row["description"],
                "quorum": row["quorum"],
                "score": score_by_name[row["name"]],
                "lowest": [[key, value] for key, value in ranked],
            }
        )
    return picked


def _plot(cal_rows, val_rows, score_by_name):
    import matplotlib.pyplot as plt

    try:
        plt.style.use("data-ink")
    except OSError:
        pass
    plt.rcParams["svg.hashsalt"] = "srtj-name-signal-diagnosis"
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    for axis, (title, rows) in zip(
        axes, (("calibration", cal_rows), ("validation", val_rows)), strict=True
    ):
        for label in (0, 1):
            values = [
                score_by_name[row["name"]]
                for row in rows
                if row["label"] == label and row["name"] in score_by_name
            ]
            axis.hist(
                values,
                bins=bins,
                histtype="stepfilled",
                alpha=0.45,
                color=LABEL_COLOR[label],
                edgecolor=LABEL_COLOR[label],
                linewidth=2.6,
                label=LABEL_NAME[label],
            )
        axis.axvline(CUT, color="#555555", linestyle=":", linewidth=1.2)
        axis.text(
            CUT,
            axis.get_ylim()[1] * 0.98,
            f"cut {CUT:.3f}",
            ha="right",
            va="top",
            fontsize=20,
            color="#555555",
        )
        axis.set_xlabel("prototype acceptance score")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(0.0, 1.0)
        axis.grid(False)
        axis.annotate(
            title,
            xy=(0.02, 0.96),
            xycoords="axes fraction",
            fontsize=22,
            color="#222222",
            ha="left",
            va="top",
        )
    axes[0].set_ylabel("names")
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        loc="lower center",
        ncol=2,
        fontsize=20,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / "score-by-label.svg"
    figure.savefig(target, format="svg", metadata={"Date": None})
    target.write_text(
        "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
    )
    plt.close(figure)
    return target


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _interval(auc):
    return (
        "n/a"
        if auc is None
        else f"{auc['point']:.3f} [{auc['lower']:.3f}, {auc['upper']:.3f}]"
    )


def _render_fragment(result):
    rows = []
    for item in result["per_question"]:
        cell = _interval(item["auc"])
        if item["auc"] is not None:  # noqa: E711
            if item["excludes_half"]:
                cell += " ✱"
        flag = ""
        if item["inverted"]:
            flag = " — inverted"
        elif item["excludes_half"]:
            flag = " — signal"
        rows.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}{}</td></tr>".format(
                html.escape(item["question"]),
                item["n"],
                _fmt(item["values"]["min"]),
                _fmt(item["values"]["max"]),
                _fmt(item["values"]["std"]),
                cell,
                flag,
            )
        )
    aggregation_rows = []
    for name, label in AGGREGATION_LABELS:
        item = result["aggregations"][name]
        delta = item["validation"].get("quorum_difference_interval")
        delta_cell = (
            f"{delta['point']:+.3f} [{delta['lower']:+.3f}, {delta['upper']:+.3f}]"
            if delta
            else "n/a"
        )
        aggregation_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                label,
                _interval(item["calibration"]["auc"]),
                _interval(item["validation"]["auc"]),
                item["validation"].get("quorum_matched_n", 0),
                delta_cell,
            )
        )
    picked = [
        "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(item["name"]),
            html.escape(item["description"][:110]),
            _fmt(item["quorum"]),
            ", ".join(
                f"{html.escape(key)}={value:.2f}" for key, value in item["lowest"]
            ),
        )
        for item in result["ten_rejected"]
    ]
    control = result["control"]
    verdict = "reproduces" if control["reproduces"] else "DOES NOT reproduce"
    parts = [
        '<section id="srtj-name-signal-diagnosis">',
        '<h2 id="srtj-name-signal-diagnosis-title">Which name-review questions carry signal</h2>',
        (
            f'<p id="srtj-name-signal-diagnosis-control"><strong>Control.</strong> The '
            f"existing prototype score {verdict} the measurement: calibration "
            f"{control['calibration']:.3f} against the recorded "
            f"{CONTROL_CALIBRATION:.3f}, validation {control['validation']:.3f} against "
            f"the recorded {CONTROL_VALIDATION:.3f}. The loader read "
            f"{result['coverage']['names']} rich-state names from "
            f"{result['coverage']['answered']} saved base answers, "
            f"{result['coverage']['questions']} questions in all; every choice below is "
            "fixed on calibration before "
            "validation is read.</p>"
        ),
        '<h3 id="srtj-name-signal-diagnosis-questions-title">Each question alone</h3>',
        (
            '<p id="srtj-name-signal-diagnosis-questions">Calibration AUC with a 95% '
            "copy-group bootstrap interval (1,000 resamples). ✱ marks a question whose "
            "interval excludes 0.5. A question scoring below 0.5 is inverted: it is "
            "informative, but its higher values point at rejection rather than "
            "acceptance.</p>"
        ),
        '<table id="srtj-name-signal-diagnosis-questions-table"><thead><tr>'
        "<th>Question</th><th>Names</th><th>Min</th><th>Max</th><th>Std</th>"
        "<th>Calibration AUC [95%]</th></tr></thead><tbody>",
        *rows,
        "</tbody></table>",
        '<h3 id="srtj-name-signal-diagnosis-aggregations-title">Aggregations</h3>',
        (
            '<p id="srtj-name-signal-diagnosis-aggregations">Each aggregation was fixed on '
            "calibration and then validation was read once for it. The mean covers the "
            f"{len(result['selection']['selected'])} questions whose calibration interval "
            "excludes 0.5, sign-aligned; the logistic regression uses all questions. The "
            "L2 strength is chosen by grouped cross-validation inside calibration. "
            "Validation AUC is on all scored validation names; Δ is against the quorum’s "
            "own score on the names both cover, paired.</p>"
        ),
        '<table id="srtj-name-signal-diagnosis-aggregations-table"><thead><tr>'
        "<th>Aggregation</th><th>Calibration AUC [95%]</th><th>Validation AUC [95%]</th>"
        "<th>Matched names</th><th>Δ vs quorum [95%]</th></tr></thead><tbody>",
        *aggregation_rows,
        "</tbody></table>",
        '<h3 id="srtj-name-signal-diagnosis-cut-title">Why the cut accepted nothing</h3>',
        (
            f'<p id="srtj-name-signal-diagnosis-cut">The prototype score is nearly flat '
            f"across the two labels: the calibration cut {CUT:.3f} sits above almost every "
            "validation score, so it accepted none.</p>"
        ),
        (
            '<figure id="srtj-name-signal-diagnosis-figure"><img '
            'src="/imas-codex/figures/sn-review-through-jev/srtj-name-signal-diagnosis/'
            'score-by-label.svg" alt="Prototype acceptance score distribution by golden '
            'label, calibration and validation panels"><figcaption>Prototype acceptance '
            "score by golden label on the calibration (left) and validation (right) "
            f"splits. The dotted line is the calibration cut {CUT:.3f}; it lies above the "
            "score mass of both labels, which is why it accepts no validation name."
            "</figcaption></figure>"
        ),
        '<h3 id="srtj-name-signal-diagnosis-misses-title">What the questions miss</h3>',
        (
            '<p id="srtj-name-signal-diagnosis-misses">The ten validation names the golden '
            "labels reject whose prototype score is highest, with the quorum score and the "
            "three questions on which Jev was most permissive.</p>"
        ),
        '<table id="srtj-name-signal-diagnosis-misses-table"><thead><tr>'
        "<th>Name</th><th>Description</th><th>Quorum</th><th>Lowest three questions</th>"
        "</tr></thead><tbody>",
        *picked,
        "</tbody></table>",
        "</section>",
    ]
    FRAGMENT.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prove-guard",
        action="store_true",
        help="complement the control scores to show the reproduction guard refuses",
    )
    args = parser.parse_args()

    from imas_codex.standard_names.review.checks import load_review_checks
    from imas_codex.standard_names.review.jev_review import NAME_AXIS

    prototype = _prototype()
    fixture = prototype.load_labelled_fixture(FIXTURE)
    context = json.loads(CONTEXT.read_text())
    answers = _load_saved_answers()
    checks = [check for check in load_review_checks()[0] if check.axis == NAME_AXIS]
    table = _question_table(checks)
    accepted_dimension = _accepted_dimension(table)

    rows = _build_rows(fixture, context, answers, checks, table, accepted_dimension)
    cal_rows = [row for row in rows if row["split"] == "calibration"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    question_names = list(table.keys())

    existing = {
        row["name"]: score
        for row in rows
        if (score := prototype._score(answers[row["name"]], checks)) is not None
    }
    if args.prove_guard:
        existing = {name: 1.0 - value for name, value in existing.items()}

    per_question = _per_question(cal_rows, question_names)
    selected = [item["question"] for item in per_question if item["excludes_half"]]
    aligned_signs = {
        item["question"]: item["auc"]["point"] >= 0.5
        for item in per_question
        if item["excludes_half"] and item["auc"] is not None
    }

    mean_scores = _mean_over_selected(rows, selected, aligned_signs)
    model = _fit_logistic(cal_rows, question_names)
    logistic_scores = _logistic_scores(model, rows, question_names)

    result = {
        "method": {
            "score": (
                "The prototype's own score, reused by importing the measurement script's "
                "_score: half the mean of four dimension acceptance values plus half the "
                "mean of the answered checks."
            ),
            "question_value": (
                "A dimension question's acceptance value is the probability of its "
                "accepting category; a Noul check's is 1 minus its defect probability; a "
                "Choice check's is the authored acceptable option's probability."
            ),
            "selection": (
                "A question is selected when its calibration 95% copy-group bootstrap "
                "interval excludes 0.5. A selected question whose AUC is below 0.5 is "
                "sign-aligned by complementing its value, since its higher values point at "
                "rejection."
            ),
            "validation": (
                "Every aggregation was fixed on calibration and its validation score read "
                "once. Validation AUC is on all scored validation names; Δ is the paired "
                "difference against the quorum's own score on the names both cover, with "
                "the same copy-group bootstrap."
            ),
        },
        "coverage": {
            "names": len(rows),
            "answered": len(answers),
            "calibration": len(cal_rows),
            "validation": len(val_rows),
            "questions": len(question_names),
        },
        "control": _control_check(
            _aggregate_report(cal_rows, existing), _aggregate_report(val_rows, existing)
        ),
        "per_question": per_question,
        "selection": {
            "rule": "calibration 95% interval excludes 0.5",
            "selected": selected,
            "inverted": [item["question"] for item in per_question if item["inverted"]],
        },
        "aggregations": {},
        "validation_reads": {
            "existing_score": 1,
            "selected_mean": 1,
            "logistic_l2": 1,
        },
        "ten_rejected": _ten_rejected(val_rows, existing, question_names),
    }

    for key, scores in (
        ("existing_score", existing),
        ("selected_mean", mean_scores),
        ("logistic_l2", logistic_scores),
    ):
        calibration = _aggregate_report(cal_rows, scores)
        validation = _aggregate_report(val_rows, scores)
        validation.update(_quorum_comparison(val_rows, scores))
        result["aggregations"][key] = {
            "calibration": calibration,
            "validation": validation,
            "c": float(model.C_[0]) if key == "logistic_l2" else None,
        }

    if not result["control"]["reproduces"]:
        raise RuntimeError(
            "control did not reproduce the measurement: calibration "
            f"{result['control']['calibration']:.3f} against {CONTROL_CALIBRATION}, "
            f"validation {result['control']['validation']:.3f} against "
            f"{CONTROL_VALIDATION}"
        )

    figure = _plot(cal_rows, val_rows, existing)
    result["figure"] = str(figure.relative_to(HERE.parents[2]))
    RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _render_fragment(result)
    print(
        json.dumps(
            {
                "control": result["control"],
                "selected": selected,
                "validation": {
                    name: result["aggregations"][name]["validation"]["auc"]
                    for name in result["aggregations"]
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

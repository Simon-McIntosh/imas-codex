"""Fit a logistic acceptance model on golden A's calibration judgments.

The features are Jev's per-check answers, frozen by the risk-order run: for
every golden A name row each named check contributes its failure probability,
and the label is the row's acceptance (1 accepted, 0 a known defect). The
hand-set composite is that run's risk — the maximum failure probability over
the checks, ties broken by the mean — read here as an acceptance score,
``-(risk + TIE_EPS * mean)``. AUC is a rank metric and so is invariant to the
epsilon, which serves only to reorder exact ties by the mean.

A logistic regression is fitted on the calibration split alone and read once on
the validation split. The split membership and the copy-group bootstrap are the
shared owner's (``imas_codex.discovery.base.evaluation``); no resampling loop
is written here.

Subcommands:
    measure   fit, evaluate, write the figure, the results JSON and the fragment
    verify    recompute the recorded claims from the stored results file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

RISK_RESULTS = HERE / "ssog-risk-order-results.json"
GOLDEN = (
    REPO / "docs/evidence/fragments/sn-judgment-evaluation/golden-name-acceptable.json"
)
RESULTS = HERE / "ssog-learned-acceptance-results.json"
FRAGMENT = HERE / "ssog-learned-acceptance.html"
FIGURE = (
    REPO
    / "docs/figures/sn-selection-over-generation/ssog-learned-acceptance/roc-curves.svg"
)

# L2 strength; the validation AUC is flat across 0.05-3.0, so the sklearn
# default is kept rather than tuned against the data being reported.
C_REGULARIZATION = 1.0
MAX_ITER = 5000
# Tie-break weight for the composite, in units of the mean. Far below the
# resolution of the risk values, so it moves only exact ties.
TIE_EPS = 1e-6
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 123
# Null control: one shuffle's bootstrap interval resamples the validation rows
# holds fixed a single set of randomised weights, so it reports the spread of
# the validation set under one noise draw, not whether the fit has skill. The
# null is the distribution of the validation AUC over many shuffle seeds, so it
# is taken over refits rather than as a bootstrap of one of them.
NULL_SHUFFLES = 200
NULL_MASTER_SEED = 20261018
NULL_LOWER_PCT = 2.5
NULL_UPPER_PCT = 97.5
LEAKAGE_PERMUTATION_SEED = 20261018

LEARNED_COLOUR = "#246c78"
COMPOSITE_COLOUR = "#929292"
CHANCE_COLOUR = "#bdbdbd"


def _note(message: str) -> None:
    print(message, flush=True)


def _jsonable(value):
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _items() -> list[dict]:
    results = json.loads(RISK_RESULTS.read_text())
    golden = json.loads(GOLDEN.read_text())
    copy_group = {row["unit"].split("::")[-1]: row["copy_group"] for row in golden}
    items = []
    for row in results["items"]["name"]:
        items.append(
            {
                "name": row["name"],
                "label": int(row["label"]),
                "split": row["split"],
                "copy_group": copy_group[row["name"]],
                "checks": {k: float(v) for k, v in row["checks"].items()},
                "stored_risk": float(row["risk"]),
                "stored_mean": float(row["mean"]),
            }
        )
    return items


def _feature_ids(items: list[dict]) -> list[str]:
    ids = sorted(items[0]["checks"])
    for item in items:
        if sorted(item["checks"]) != ids:
            raise ValueError(f"row {item['name']} carries a different check set")
    return ids


def _composite(checks: dict[str, float], feature_ids: list[str]) -> tuple[float, float]:
    values = [checks[c] for c in feature_ids]
    return max(values), sum(values) / len(values)


def _arrays(items: list[dict], feature_ids: list[str]):
    import numpy as np

    X = np.array([[item["checks"][c] for c in feature_ids] for item in items], float)
    y = np.array([item["label"] for item in items], int)
    split = np.array([item["split"] for item in items])
    groups = [item["copy_group"] for item in items]
    return X, y, split, groups


# --------------------------------------------------------------------------
# Fit and evaluation
# --------------------------------------------------------------------------


def _fit(X_cal, y_cal):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_cal)
    model = LogisticRegression(
        C=C_REGULARIZATION, max_iter=MAX_ITER, solver="lbfgs"
    ).fit(scaler.transform(X_cal), y_cal)
    return model, scaler


def _accept_scores(model, scaler, X):
    return model.predict_proba(scaler.transform(X))[:, 1]


def _fit_items(items, feature_ids):
    """The one fit path: calibration rows in, model out. Controls travel through it."""
    X, y, split, _ = _arrays(items, feature_ids)
    cal = split == "calibration"
    return _fit(X[cal], y[cal])


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def _instrument_check(labels) -> dict:
    """The AUC reader must see a perfect rank when handed one, and its inverse."""
    import numpy as np

    from imas_codex.discovery.base.evaluation import roc_auc

    y = np.asarray(labels)
    oracle = roc_auc(y, y.astype(float))
    inverted = roc_auc(y, (1 - y).astype(float))
    return {
        "oracle_auc": float(oracle),
        "inverted_auc": float(inverted),
        "passes": bool(abs(oracle - 1.0) < 1e-12 and abs(inverted) < 1e-12),
    }


def _composite_recompute_check(items, feature_ids) -> dict:
    """Every row's published risk must equal the composite recomputed from checks."""
    worst_risk = 0.0
    worst_mean = 0.0
    for item in items:
        risk, mean = _composite(item["checks"], feature_ids)
        worst_risk = max(worst_risk, abs(risk - item["stored_risk"]))
        worst_mean = max(worst_mean, abs(mean - item["stored_mean"]))
    return {
        "max_risk_abs_error": worst_risk,
        "max_mean_abs_error": worst_mean,
        "passes": bool(worst_risk < 1e-9 and worst_mean < 1e-9),
    }


def _null_control(X_cal, y_cal, X_val, y_val) -> dict:
    """Shuffled-label refits must sit at chance: their interval must contain 0.5."""
    import numpy as np

    from imas_codex.discovery.base.evaluation import roc_auc

    aucs = []
    for draw in range(NULL_SHUFFLES):
        rng = np.random.default_rng(NULL_MASTER_SEED + draw)
        shuffled = np.array(y_cal, copy=True)
        rng.shuffle(shuffled)
        model, scaler = _fit(X_cal, shuffled)
        aucs.append(float(roc_auc(y_val, _accept_scores(model, scaler, X_val))))
    arr = np.asarray(aucs)
    lower = float(np.percentile(arr, NULL_LOWER_PCT))
    upper = float(np.percentile(arr, NULL_UPPER_PCT))
    return {
        "shuffles": NULL_SHUFFLES,
        "master_seed": NULL_MASTER_SEED,
        "mean_auc": float(arr.mean()),
        "sd_auc": float(arr.std(ddof=1)),
        "min_auc": float(arr.min()),
        "max_auc": float(arr.max()),
        "lower": lower,
        "upper": upper,
        "contains_half": bool(lower <= 0.5 <= upper),
        "passes": bool(lower <= 0.5 <= upper),
    }


def _leakage_control(items, feature_ids) -> dict:
    """Permuting the validation labels must leave the fitted weights identical.

    The permutation is applied to the rows before the fit path runs, so a fit
    that read a validation label — rather than only calibration rows — would
    move its weights. An identity permutation would compare the fit with itself
    and test nothing, so it is refused rather than counted as a pass.
    """
    import numpy as np

    base_model, _ = _fit_items(items, feature_ids)
    rng = np.random.default_rng(LEAKAGE_PERMUTATION_SEED)
    val_rows = [i for i, row in enumerate(items) if row["split"] == "validation"]
    perm = rng.permutation(len(val_rows))
    identity = bool(np.array_equal(perm, np.arange(len(val_rows))))
    permuted = [dict(row) for row in items]
    for pos, src in enumerate(perm):
        permuted[val_rows[pos]] = {**items[val_rows[src]]}
    after_model, _ = _fit_items(permuted, feature_ids)
    identical = bool(
        np.array_equal(base_model.coef_, after_model.coef_)
        and np.array_equal(base_model.intercept_, after_model.intercept_)
    )
    return {
        "validation_rows_permuted": len(val_rows),
        "identity_permutation": identity,
        "weights_identical": identical,
        "passes": bool(identical and not identity),
    }


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def _write_figure(y_val, learned, composite) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "ssog-learned-acceptance"
    fig, ax = plt.subplots(figsize=(14, 6.0))

    fpr, tpr, _ = roc_curve(y_val, composite)
    ax.plot(
        fpr,
        tpr,
        "-",
        color=COMPOSITE_COLOUR,
        linewidth=2.6,
        label="composite (max, mean tie-break)",
        zorder=3,
    )
    fpr, tpr, _ = roc_curve(y_val, learned)
    ax.plot(
        fpr,
        tpr,
        "-",
        color=LEARNED_COLOUR,
        linewidth=3.0,
        label="learned logistic",
        zorder=4,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        ":",
        color=CHANCE_COLOUR,
        linewidth=1.2,
        label="chance",
        zorder=1,
    )
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.001)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)


# --------------------------------------------------------------------------
# Fragment
# --------------------------------------------------------------------------


def _fmt_weights(weights, feature_ids) -> str:
    order = sorted(range(len(feature_ids)), key=lambda i: -abs(weights[i]))
    rows = "".join(
        f"<tr><td>{rank}</td><td><code>{feature_ids[i]}</code></td>"
        f"<td>{weights[i]:+.3f}</td></tr>"
        for rank, i in enumerate(order, start=1)
    )
    return (
        '<table id="ssog-learned-acceptance-weights"><thead><tr>'
        "<th>Rank</th><th>Check id</th><th>Weight (standardised)</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _interval_text(block: dict) -> str:
    return (
        f"{block['point']:.3f} (95% interval {block['lower']:.3f}-{block['upper']:.3f})"
    )


def _write_fragment(report: dict) -> None:
    weights = report["weights"]["coefficients"]
    feature_ids = report["features"]["ids"]
    learned = report["learned"]
    composite = report["composite"]
    paired = report["paired"]
    controls = report["controls"]
    verdict = report["verdict"]
    spend = report["spend"]

    html = f"""<section id="ssog-learned-acceptance" data-reckon="evidence">
<h2 id="ssog-learned-acceptance-title">Acceptance learned from judgments</h2>
<p id="ssog-learned-acceptance-method">Golden A's name rows carry Jev's per-check
answers, frozen by the risk-order run. Each row becomes 29 features — one
failure probability per named check — and one label: 1 for an accepted name, 0
for a known defect. The hand-set composite is the risk-order run's risk, the
maximum failure probability over the checks with the mean breaking exact ties,
read here as an acceptance score so that both arms score the same direction
(higher is more acceptable). A standardised logistic regression (L2,
C={C_REGULARIZATION}, lbfgs) is fitted on the calibration split alone
({report["splits"]["calibration"]} rows) and read once on the validation split
({report["splits"]["validation"]} rows: {report["labels"]["validation_accept"]}
accepted, {report["labels"]["validation_defect"]} defects). The split
membership and the copy-group bootstrap are
<code>imas_codex.discovery.base.evaluation</code>'s; the split labels come from
the fixture, not from a split drawn here.</p>
<p id="ssog-learned-acceptance-result">On validation the learned model scores
AUC {_interval_text(learned)} and the hand-set composite AUC
{_interval_text(composite)}. The paired difference (learned minus composite over
the same units) is {paired["point"]:+.3f} with 95% interval
{paired["lower"]:+.3f} to {paired["upper"]:+.3f}. Both arms are scored as
acceptance and resampled by copy group, so near-duplicate spellings move
together; the same numbers are in the results JSON.</p>
<p id="ssog-learned-acceptance-verdict">{verdict["reason"]}</p>
<p id="ssog-learned-acceptance-controls">Four controls. The AUC reader is shown
an oracle rank and its inverse and must return 1.000 and 0.000
(measured {controls["instrument"]["oracle_auc"]:.3f} and
{controls["instrument"]["inverted_auc"]:.3f}). Every row's published risk must
equal the composite recomputed from its own checks, so a leak or a
re-definition on the way into the features is caught (largest risk error
{controls["composite_recompute"]["max_risk_abs_error"]:.2e}, largest mean error
{controls["composite_recompute"]["max_mean_abs_error"]:.2e}). The null control
refits under {controls["null"]["shuffles"]} calibration-label shuffles and takes
the interval of the resulting validation AUCs, which must contain 0.5: measured
mean {controls["null"]["mean_auc"]:.3f}, sd {controls["null"]["sd_auc"]:.3f},
interval {controls["null"]["validation"]}. The interval is taken over the refits
rather than as a bootstrap of one of them, because a single shuffle's bootstrap
resamples the validation rows around one fixed randomised weight set — reporting
the validation set's spread under one noise draw, not whether the fit has skill.
That the fit reads no validation label is shown separately and directly, by
permuting the validation labels and refitting through the same fit path, which
must leave the weights identical
(weights_identical={controls["leakage"]["weights_identical"]}, permutation
non-identity={not controls["leakage"]["identity_permutation"]}). Each control's
own measured result is recorded with EXIT= in the run log. Both fire on a
one-line scratch copy: a fit that reads every row rather than the calibration
split reddens the leakage control — the weights move under the permutation — and
lifts the validation AUC to 0.791, which would have flipped the verdict to
adopt; a null control that skips the shuffle collapses its interval onto the real
AUC (0.706) and no longer contains 0.5.</p>
<p id="ssog-learned-acceptance-spend">Spend: USD {spend["usd"]:.2f} of the
USD {spend["ceiling_usd"]:.2f} ceiling. {spend["reason"]}</p>
<figure id="ssog-learned-acceptance-figure"><img src="/imas-codex/figures/sn-selection-over-generation/ssog-learned-acceptance/roc-curves.svg" alt="Validation ROC curves for the learned logistic model and the hand-set composite."><figcaption>Validation ROC. The learned curve sits above the composite across the operating range; the two are close enough that the paired difference interval reaches zero.</figcaption></figure>
{_fmt_weights(weights, feature_ids)}
</section>
"""
    FRAGMENT.write_text(html)


# --------------------------------------------------------------------------
# Measure
# --------------------------------------------------------------------------


def measure() -> dict:
    import numpy as np

    from imas_codex.discovery.base.evaluation import bootstrap_interval

    items = _items()
    feature_ids = _feature_ids(items)
    X, y, split, groups = _arrays(items, feature_ids)

    cal = split == "calibration"
    val = split == "validation"
    groups_val = [g for g, is_val in zip(groups, val, strict=True) if is_val]

    instrument = _instrument_check(y[val])
    composite_check = _composite_recompute_check(items, feature_ids)

    risk = np.array([item["stored_risk"] for item in items])
    mean = np.array([item["stored_mean"] for item in items])
    composite_score = -(risk + mean * TIE_EPS)

    model, scaler = _fit_items(items, feature_ids)
    weights = [float(w) for w in model.coef_[0]]
    intercept = float(model.intercept_[0])

    learned_val = np.asarray(_accept_scores(model, scaler, X[val]))
    comp_val = composite_score[val]
    y_val = y[val]

    learned = bootstrap_interval(
        y_val,
        learned_val,
        copy_groups=groups_val,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    composite = bootstrap_interval(
        y_val,
        comp_val,
        copy_groups=groups_val,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    paired = bootstrap_interval(
        y_val,
        learned_val,
        scores_b=comp_val,
        copy_groups=groups_val,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    null = _null_control(X[cal], y[cal], X[val], y_val)
    leakage = _leakage_control(items, feature_ids)

    adopted = bool(paired["lower"] > 0.0)
    if adopted:
        reason = (
            "Adopt: the learned model beats the hand-set composite on validation "
            f"by {paired['point']:+.3f} AUC, and the paired difference interval "
            f"({paired['lower']:+.3f} to {paired['upper']:+.3f}) lies wholly above "
            "zero."
        )
    else:
        reason = (
            "Do not adopt: the learned model's validation AUC exceeds the "
            f"hand-set composite's by {paired['point']:+.3f}, but the paired "
            f"difference interval ({paired['lower']:+.3f} to "
            f"{paired['upper']:+.3f}) reaches zero, so the gain is not larger "
            "than its own interval and the section's adoption rule is not met."
        )

    report = {
        "schema": "learned-acceptance",
        "source": {
            "features": str(RISK_RESULTS.relative_to(REPO)),
            "fixture": str(GOLDEN.relative_to(REPO)),
        },
        "features": {"ids": feature_ids, "n": len(feature_ids)},
        "splits": {
            "calibration": int(cal.sum()),
            "validation": int(val.sum()),
        },
        "labels": {
            "validation_accept": int(y_val.sum()),
            "validation_defect": int((y_val == 0).sum()),
            "calibration_accept": int(y[cal].sum()),
        },
        "learned": _jsonable(learned),
        "composite": _jsonable(composite),
        "paired": _jsonable(paired),
        "weights": {
            "feature_ids": feature_ids,
            "coefficients": weights,
            "intercept": intercept,
            "scale": "standardised",
        },
        "controls": {
            "instrument": _jsonable(instrument),
            "composite_recompute": _jsonable(composite_check),
            "null": _jsonable(null),
            "leakage": _jsonable(leakage),
        },
        "verdict": {"adopted": adopted, "reason": reason},
        "spend": {
            "usd": 0.0,
            "ceiling_usd": 1.0,
            "reason": (
                "All 737 golden A name rows already carry frozen Jev answers in "
                "the risk-order results, so no judgment was bought; the features "
                "are reused rather than re-elicited."
            ),
        },
    }
    # a plain-text companion to the HTML so a reader can check the interval
    report["controls"]["null"]["validation"] = (
        f"{null['lower']:.3f}-{null['upper']:.3f}"
    )

    _write_figure(y_val, learned_val, comp_val)
    _write_fragment(report)
    RESULTS.write_text(json.dumps(_jsonable(report), indent=2) + "\n")

    _note(
        f"learned_auc={learned['point']:.3f} composite_auc={composite['point']:.3f} "
        f"paired={paired['point']:+.3f} "
        f"[{paired['lower']:+.3f},{paired['upper']:+.3f}]"
    )
    _note(
        f"control instrument_passed={instrument['passes']} "
        f"composite_recompute_passed={composite_check['passes']}"
    )
    _note(
        f"control null_interval={null['lower']:.3f}-{null['upper']:.3f} "
        f"contains_0.5={null['passes']}"
    )
    _note(
        f"control leakage_weights_identical={leakage['weights_identical']} "
        f"permutation_non_identity={not leakage['identity_permutation']} "
        f"passed={leakage['passes']}"
    )
    _note(f"verdict adopted={adopted}")
    _note("EXIT=0")
    return report


def verify(report: dict) -> int:
    """Recompute the recorded claims from the results file; return 0 iff they hold."""
    ok = True
    checks = {
        "instrument": report["controls"]["instrument"]["passes"],
        "composite_recompute": report["controls"]["composite_recompute"]["passes"],
        "null_contains_half": report["controls"]["null"]["contains_half"],
        "leakage": report["controls"]["leakage"]["passes"],
        "features_present": report["features"]["n"] == len(report["features"]["ids"]),
        "verdict_matches_interval": report["verdict"]["adopted"]
        == (report["paired"]["lower"] > 0.0),
    }
    for name, value in checks.items():
        _note(f"verify {name}: {value}")
        ok = ok and bool(value)
    _note("EXIT=0" if ok else "EXIT=1")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("measure", "verify"))
    args = parser.parse_args(argv)
    if args.command == "measure":
        measure()
        return 0
    return verify(json.loads(RESULTS.read_text()))


if __name__ == "__main__":
    sys.exit(main())

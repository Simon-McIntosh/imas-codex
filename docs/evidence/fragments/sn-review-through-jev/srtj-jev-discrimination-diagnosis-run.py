"""Diagnose frozen review discrimination without making judge or graph calls."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import html
import json
import lzma
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
STEM = Path(__file__).name.removesuffix("-run.py")
OUTPUT = Path(__file__).parent
SPLITS = ("calibration", "validation")


def read(path):
    raw = path.read_bytes()
    return json.loads(lzma.decompress(raw) if path.suffix == ".xz" else raw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def summary(rows, scores):
    pairs = [(r, s) for r, s in zip(rows, scores, strict=True) if s is not None]
    labels = [r["label"] for r, _ in pairs]
    values = [float(s) for _, s in pairs]
    assert all(math.isfinite(s) for s in values)
    return {
        "n": len(pairs),
        "acceptable": sum(labels),
        "defective": len(labels) - sum(labels),
        "auc": auc(labels, values),
        "distinct": len(set(values)),
        "mean_acceptable": float(np.mean([s for (r, s) in pairs if r["label"]]))
        if sum(labels)
        else None,
        "mean_defective": float(np.mean([s for (r, s) in pairs if not r["label"]]))
        if len(labels) - sum(labels)
        else None,
    }


def field_value(state, field):
    if "name" in state:
        container = (
            state["name"]["segments"]
            if field in state["name"]["segments"]
            else state["name"]
        )
        if field not in container:
            container = state["neighbourhood"]
        return container.get(field), field in container
    key = "name_id" if field == "id" else field
    return state.get(key), key in state


def presence(state, field):
    value, present = field_value(state, field)
    if not present:
        return "missing"
    if isinstance(value, dict) and "observed" in value:
        if not value["observed"]:
            return "unobserved"
        value = value["value"]
    if value is None:
        return "null"
    return "empty" if value in ("", [], {}) else "nonempty"


def unwrapped(state, field):
    value, _ = field_value(state, field)
    return (
        value.get("value") if isinstance(value, dict) and "observed" in value else value
    )


def label_class(row, axis):
    return (
        (row.get("label_check") or "acceptable")
        if axis == "docs"
        else row["label_source"]
    )


def confusion(rows, probs, cutoff):
    if cutoff is None:
        return None
    pairs = [(r, p) for r, p in zip(rows, probs, strict=True) if p is not None]
    tp = sum(not r["label"] and p >= cutoff for r, p in pairs)
    fp = sum(r["label"] and p >= cutoff for r, p in pairs)
    defects = sum(not r["label"] for r, _ in pairs)
    acceptable = sum(r["label"] for r, _ in pairs)
    return {
        "cutoff": cutoff,
        "defects_caught": tp,
        "false_flags": fp,
        "defect_recall": tp / defects if defects else None,
        "false_flag_rate": fp / acceptable if acceptable else None,
    }


def check_report(check, rows, block, axis):
    out = {
        "id": check.id,
        "dimension": check.dimension,
        "kind": check.question.kind.value if check.question else "deterministic_audit",
        "question": check.question.text if check.question else None,
        "audit_finding_class": check.audit_finding_class,
        "evidence_fields": check.evidence_fields,
        "saved_diagnostic": block["check_diagnostics"].get(check.id),
        "splits": {},
    }
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        probs = [r["failure"].get(check.id) for r in selected]
        stats = summary(selected, [1 - p if p is not None else None for p in probs])
        stats["answered"] = stats["n"]
        stats["unanswered"] = len(selected) - stats["n"]
        stats["separation"] = (
            (stats["mean_acceptable"] - stats["mean_defective"])
            if stats["auc"] is not None
            else None
        )
        stats["midpoint"] = confusion(selected, probs, 0.5)
        cutoff = (out["saved_diagnostic"] or {}).get("threshold")
        stats["calibration_cutoff"] = confusion(selected, probs, cutoff)
        stats["selected_veto"] = confusion(
            selected, probs, block["check_thresholds"].get(check.id)
        )
        stats["evidence"] = {
            f: dict(Counter(presence(r["state"], f) for r in selected))
            for f in check.evidence_fields
        }
        # Golden records identify whole-item defects, never individual check truth.
        stats["exact_check_defect_labels"] = None
        stats["calibratable_check_truth"] = False
        stats["label_limitation"] = (
            "No check-specific golden truth; overall defect recall is not check recall."
        )
        stats["class_proxy"] = {}
        for cls in sorted(
            {label_class(r["golden"], axis) for r in selected if not r["label"]}
        ):
            group = [
                r
                for r in selected
                if r["label"] or label_class(r["golden"], axis) == cls
            ]
            gp = [r["failure"].get(check.id) for r in group]
            item = summary(group, [1 - p if p is not None else None for p in gp])
            item["fewer_than_five_defects"] = item["defective"] < 5
            stats["class_proxy"][cls] = item
        out["splits"][split] = stats
    return out


def counter_rows(rows, axis):
    return [
        {"split": s, "label": label, "label_class": c, "count": n}
        for (s, label, c), n in sorted(
            Counter(
                (r["split"], r["label"], label_class(r, axis)) for r in rows
            ).items()
        )
    ]


def attach(rows, records, golden, axis, cascade, block):
    for row in rows:
        row["golden"] = golden[row["id"]]
        record = records[row["id"]]
        row["state"] = record["state"]
        row["answers"] = record["answers"]
        row["failure"] = {
            k: v["failure_probability"] if v["answered"] else None
            for k, v in row["judgment"]["checks"].items()
        }
        row["dimensions"] = cascade.rubric_scores(
            axis, row["judgment"], block["rubric"]
        )
        routed = cascade._route(axis, row["judgment"], row["score"], thresholds=block)
        row["routed_dimensions"] = dict(routed.scores)
        row["rubric_total"] = sum(row["dimensions"].values()) / 80
        row["routed_total"] = routed.total
        row["route"] = routed.route
        row["features"] = cascade.rubric_features(axis, row["judgment"])


def disagreements(rows, axis, block):
    ranked = []
    for row in rows:
        others = [
            r for r in rows if r["split"] == row["split"] and r["label"] != row["label"]
        ]
        score = row["rubric_total"]
        wrong = sum(
            (r["rubric_total"] > score if row["label"] else r["rubric_total"] < score)
            + 0.5 * (r["rubric_total"] == score)
            for r in others
        )
        ranked.append((wrong / len(others), row))
    out = []
    for error, row in sorted(ranked, key=lambda x: (-x[0], x[1]["id"]))[:20]:
        contributions = []
        for dim, model in block["rubric"]["dimensions"].items():
            for feature, weight in model["weights"].items():
                value = row["features"][dim][feature]
                contributions.append(
                    {
                        "dimension": dim,
                        "feature": feature,
                        "weight": weight,
                        "compliance": value,
                        "penalty_from_perfect_logit": weight * (1 - value),
                    }
                )
        contributions.sort(key=lambda x: -x["penalty_from_perfect_logit"])
        answers = {k: v for k, v in row["answers"].items() if k.startswith("check:")}
        out.append(
            {
                "name": row["id"],
                "split": row["split"],
                "label": row["label"],
                "label_class": label_class(row["golden"], axis),
                "label_note": row["golden"].get("label_note"),
                "source_evidence": row["golden"]["source_evidence"],
                "rank_error": error,
                "rubric_total": row["rubric_total"],
                "logistic_score": row["score"],
                "route": row["route"],
                "description": unwrapped(row["state"], "description"),
                "source_paths": unwrapped(row["state"], "source_paths"),
                "dimensions": row["dimensions"],
                "driver_contributions": contributions[:8],
                "failure_probabilities": row["failure"],
                "check_answers": answers,
            }
        )
    return out


def ablations(rows, axis, checks, block, cascade):
    """Remove one fitted term, replacing it with its calibration mean."""
    variants = []
    baseline = {
        s: summary(
            [r for r in rows if r["split"] == s],
            [r["rubric_total"] for r in rows if r["split"] == s],
        )["auc"]
        for s in SPLITS
    }
    for dim, model in block["rubric"]["dimensions"].items():
        for key in model["weights"]:
            fit = copy.deepcopy(block["rubric"])
            mean = float(
                np.mean(
                    [
                        r["features"][dim][key]
                        for r in rows
                        if r["split"] == "calibration"
                    ]
                )
            )
            fit["dimensions"][dim]["intercept"] += (
                fit["dimensions"][dim]["weights"].pop(key) * mean
            )
            stats = {}
            for split in SPLITS:
                group = [r for r in rows if r["split"] == split]
                result = summary(
                    group,
                    [
                        sum(cascade.rubric_scores(axis, r["judgment"], fit).values())
                        / 80
                        for r in group
                    ],
                )
                stats[split] = {
                    "auc": result["auc"],
                    "delta": result["auc"] - baseline[split],
                }
            variants.append(
                {
                    "dimension": dim,
                    "feature": key,
                    "calibration_mean": mean,
                    "splits": stats,
                }
            )
    return sorted(variants, key=lambda v: -v["splits"]["calibration"]["delta"])


def controls():
    assert auc([0, 1], [0, 1]) == 1
    assert auc([0, 1], [1, 0]) == 0
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert auc([1], [1]) is None
    assert presence({"description": "known"}, "description") == "nonempty"
    assert presence({"description": "known"}, "absent") == "missing"
    assert presence({"links": {"observed": True, "value": []}}, "links") == "empty"
    assert (
        presence({"links": {"observed": False, "value": None}}, "links") == "unobserved"
    )
    print(
        "CONTROL perfect AUC=1; reversed AUC=0; tied AUC=0.5; absent class=null; nonempty/empty/unobserved/missing distinguished",
        flush=True,
    )


def render(report):
    def esc(value):
        return html.escape(str(value))

    def fmt(value):
        return "unknown" if value is None else f"{value:.3f}"

    parts = [
        f'<section id="{STEM}">',
        "<h2>Where frozen review signal is lost</h2>",
        "<p>Offline diagnosis; no LLM calls and USD 0 spent. The cascade remains in shadow. "
        "AUC is oriented toward acceptable items; 0.5 is chance. Each check uses 1 − failure probability. "
        "Calibration results are in-sample diagnostics; validation has been inspected repeatedly and is no longer an untouched deployment holdout.</p>",
        "<p>All dimensions use the landed 0–20 rubric mapping before the route ceiling. "
        "Documentation physics accuracy has checks but no category question. Dimensions were fitted to whole-item binary labels, not dimension-specific gold ratings. "
        "Observed-empty evidence is distinct from missing or unobserved evidence; a null optional grammar segment need not be a defect. "
        "Deterministic audits do not ask Jev and may read richer inputs than the prompt state.</p>",
    ]

    def table(headers, data):
        parts.append(
            "<table><thead><tr>"
            + "".join(f"<th>{esc(x)}</th>" for x in headers)
            + "</tr></thead><tbody>"
        )
        parts.extend(
            "<tr>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>"
            for row in data
        )
        parts.append("</tbody></table>")

    for axis, data in report["axes"].items():
        parts.append(f"<h3>{esc(axis)}: labels and score reproduction</h3>")
        table(
            [
                "Split",
                "Golden unique",
                "Scored",
                "Acceptable",
                "Defective",
                "Logistic AUC",
                "Raw rubric AUC",
                "Routed rubric AUC",
            ],
            [
                [
                    s,
                    data["coverage"][s]["golden_unique"],
                    data["summary"][s]["n"],
                    data["summary"][s]["acceptable"],
                    data["summary"][s]["defective"],
                    fmt(data["logistic"][s]["auc"]),
                    fmt(data["summary"][s]["auc"]),
                    fmt(data["routed"][s]["auc"]),
                ]
                for s in SPLITS
            ],
        )
        table(
            ["Split", "Label class", "Label", "Golden count"],
            [
                [r["split"], r["label_class"], r["label"], r["count"]]
                for r in data["labels"]
            ],
        )
        parts.append(
            "<p>No golden row labels a specific taxonomy check. Exact defect counts for all checks are therefore unknown, not zero. "
            "The JSON reports every check against each broad label class as a proxy; these are not check-specific sensitivity estimates. "
            "Classes with fewer than five defects are: "
            + esc(data["thin_classes"])
            + ".</p>"
        )
        table(
            [
                "Dimension",
                "Calibration AUC",
                "Validation AUC",
                "Validation category AUC",
            ],
            [
                [
                    d,
                    fmt(v["calibration"]["auc"]),
                    fmt(v["validation"]["auc"]),
                    fmt(data["categories"][d]["validation"]["auc"]),
                ]
                for d, v in data["dimensions"].items()
            ],
        )
        parts.append(
            "<h4>Every check</h4><p>C/V denotes calibration/validation. TP and FP use a common failure cutoff of 0.5; "
            "they count whole-item defects and acceptable items. The JSON also evaluates the saved calibration cutoff and selected veto cutoff. "
            "The claim check here measures non-support, including “says nothing”; its separate hard veto measures contradiction only.</p>"
        )
        table(
            [
                "Check / dimension",
                "Answered C/V",
                "AUC C/V",
                "Separation C/V",
                "TP C/V",
                "FP C/V",
            ],
            [
                [
                    c["id"] + " / " + c["dimension"],
                    "/".join(str(c["splits"][s]["answered"]) for s in SPLITS),
                    "/".join(fmt(c["splits"][s]["auc"]) for s in SPLITS),
                    "/".join(fmt(c["splits"][s]["separation"]) for s in SPLITS),
                    "/".join(
                        str(
                            (c["splits"][s]["midpoint"] or {}).get(
                                "defects_caught", "unknown"
                            )
                        )
                        for s in SPLITS
                    ),
                    "/".join(
                        str(
                            (c["splits"][s]["midpoint"] or {}).get(
                                "false_flags", "unknown"
                            )
                        )
                        for s in SPLITS
                    ),
                ]
                for c in data["checks"]
            ],
        )
        parts.append("<h4>Evidence present in the actual frozen states</h4>")
        table(
            [
                "Check",
                "Field",
                "Calibration nonempty / empty / unknown",
                "Validation nonempty / empty / unknown",
            ],
            [
                [
                    c["id"],
                    f,
                    *[
                        " / ".join(
                            str(v)
                            for v in (
                                c["splits"][s]["evidence"][f].get("nonempty", 0),
                                c["splits"][s]["evidence"][f].get("empty", 0),
                                sum(
                                    c["splits"][s]["evidence"][f].get(k, 0)
                                    for k in ("missing", "null", "unobserved")
                                ),
                            )
                        )
                        for s in SPLITS
                    ],
                ]
                for c in data["checks"]
                for f in c["evidence_fields"]
            ],
        )
        parts.append(
            "<h4>Twenty worst-ranked golden items</h4><p>Rank error is the fraction of opposite-label items in the same split "
            "ordered incorrectly by the raw rubric, with ties counting one half. It lies in [0,1]; the largest errors across both splits are shown. "
            "Check contributions are exact weighted compliance shortfalls from a perfect answer in the fitted dimension logit, not causal attributions. "
            "All check answers and probabilities are retained in the JSON.</p>"
        )
        for row in data["disagreements"]:
            parts.append(
                "<details><summary>"
                + esc(
                    f"{row['name']} — {row['split']}, label {row['label']}, rank error {row['rank_error']:.3f}, rubric {row['rubric_total']:.3f}"
                )
                + "</summary>"
            )
            parts.append(
                "<p>"
                + esc(row["description"])
                + "</p><p>Source bindings: "
                + esc(row["source_paths"])
                + "</p><p>Golden evidence: "
                + esc(row["source_evidence"])
                + "; "
                + esc(row["label_note"] or row["label_class"])
                + "</p>"
            )
            table(
                ["Dimension", "Score / 20"],
                [[d, fmt(v)] for d, v in row["dimensions"].items()],
            )
            table(
                ["Driver", "Answer", "Weight", "Penalty from perfect"],
                [
                    [
                        d["feature"],
                        json.dumps(
                            row["check_answers"].get(
                                "check:" + d["feature"], {"compliance": d["compliance"]}
                            ),
                            ensure_ascii=False,
                        ),
                        fmt(d["weight"]),
                        fmt(d["penalty_from_perfect_logit"]),
                    ]
                    for d in row["driver_contributions"]
                ],
            )
            parts.append("</details>")
    parts.append(
        "<h3>Instrument checks and limitations</h3><p>"
        + esc(report["limitations"])
        + "</p>"
    )
    parts.append(
        "<p>Control and reproduction receipt: "
        + esc(json.dumps(report["controls"], ensure_ascii=False))
        + "</p>"
    )
    parts.append(
        "<p>Source hashes, coverage exclusions, every broad-class proxy, term ablations and full disagreement answers are in the adjacent JSON. "
        "The driver records the imported modules and code revision in its durable gate log.</p>"
    )
    parts.append("<h3>Ranked improvements</h3><ol>")
    for item in report["improvements"]:
        parts.append(
            "<li><strong>"
            + esc(item["change"])
            + "</strong><p>"
            + esc(item["evidence"])
            + "</p><p>Expected AUC effect: "
            + esc(item["expected_auc_effect"])
            + "</p></li>"
        )
    parts.append("</ol></section>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=ROOT)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_root.resolve()))
    from imas_codex.standard_names.review import (
        cascade,
        cascade_calibration as cal,
        jev_review as jev,
    )
    from imas_codex.standard_names.review.checks import load_review_checks

    def refuse(*args, **kwargs):
        raise RuntimeError("Offline diagnosis refuses judge calls")

    jev.decide_batch = refuse
    print("measurement_module=" + cascade.__file__, flush=True)
    print("measurement_cwd=" + str(Path.cwd().resolve()), flush=True)
    controls()
    threshold_path = Path(cascade.__file__).with_name("cascade_thresholds.json")
    thresholds = read(threshold_path)
    assert all("rubric" in thresholds[a] for a in ("name", "docs")), (
        "The rubric mapping must be landed before this diagnosis runs"
    )
    inputs = [
        threshold_path,
        Path(cascade.__file__),
        Path(cal.__file__),
        Path(jev.__file__),
        args.model_root / "imas_codex/llm/prompts/sn/review_checks.yaml",
    ]
    for name in (
        "srtj-evidence-supply-results.json",
        "srtj-docs-score-arm-results.json",
        "srtj-docs-score-diagnosis-results.json",
    ):
        inputs.append(OUTPUT / name)
    state_report = read(inputs[-3])
    name_states = json.loads(
        lzma.decompress(base64.b85decode(state_report["frozen_states"]["data"]))
    )
    name_path = cal.DEFAULT_NAME_RECEIPTS
    docs_path = cal.DEFAULT_DOCS_RECEIPTS
    inputs.extend([name_path, docs_path])
    name_records = {}
    for line in name_path.read_text().splitlines():
        record = json.loads(line)
        if record.get("mode") == "base" and record.get("answered"):
            name_records[record["name"]] = record
    assert len(name_states) == state_report["frozen_states"]["count"] == 701
    assert set(name_states) <= set(name_records)
    # The compressed states are the exact states frozen after evidence enrichment.
    mismatches = [
        name for name in name_states if name_states[name] != name_records[name]["state"]
    ]
    assert not mismatches, f"Name state provenance mismatch: {mismatches[:3]}"
    frozen_docs = read(docs_path)
    assert (
        digest(docs_path)
        == read(OUTPUT / "srtj-docs-score-arm-results.json")["frozen_sha256"]
    )
    docs_records = {
        k.removeprefix("base:"): v
        for k, v in frozen_docs["receipts"].items()
        if k.startswith("base:")
    }
    report = {
        "schema": "frozen-review-discrimination",
        "llm_spend_usd": 0,
        "axes": {},
        "controls": {},
        "improvements": [],
    }
    checks = load_review_checks()[0]
    for axis, fixture, receipt, records in [
        ("name", cal.GOLDEN_NAME, name_path, name_records),
        ("docs", cal.GOLDEN_DOCS, docs_path, docs_records),
    ]:
        inputs.append(fixture)
        labels = read(fixture)
        unique = {}
        for row in labels:
            name = row["canonical_unit"]
            if name in unique:
                assert (row["split"], row["label"]) == (
                    unique[name]["split"],
                    unique[name]["label"],
                )
            unique.setdefault(name, row)
        rows, _ = (
            cal._name_axis(receipt, fixture)
            if axis == "name"
            else cal._docs_axis(receipt, fixture)
        )
        block = thresholds[axis]
        attach(rows, records, unique, axis, cascade, block)
        selected_checks = [c for c in checks if c.axis == axis]
        data = {
            "labels": counter_rows(list(unique.values()), axis),
            "raw_label_rows": counter_rows(labels, axis),
            "coverage": {},
            "summary": {},
            "logistic": {},
            "routed": {},
            "dimensions": {},
            "categories": {},
            "checks": [],
        }
        data["thin_classes"] = [
            r for r in data["labels"] if r["label"] == 0 and r["count"] < 5
        ]
        for split in SPLITS:
            group = [r for r in rows if r["split"] == split]
            eligible = [r for r in unique.values() if r["split"] == split]
            missing = sorted(
                {r["canonical_unit"] for r in eligible} - {r["id"] for r in group}
            )
            data["coverage"][split] = {
                "golden_unique": len(eligible),
                "scored": len(group),
                "excluded": missing,
            }
            data["summary"][split] = summary(group, [r["rubric_total"] for r in group])
            data["logistic"][split] = summary(group, [r["score"] for r in group])
            data["routed"][split] = summary(group, [r["routed_total"] for r in group])
            for dimension in rows[0]["dimensions"]:
                data["dimensions"].setdefault(dimension, {})[split] = summary(
                    group, [r["dimensions"][dimension] for r in group]
                )
                data["categories"].setdefault(dimension, {})[split] = summary(
                    group, [r["features"][dimension].get("category") for r in group]
                )
        data["checks"] = [check_report(c, rows, block, axis) for c in selected_checks]
        data["disagreements"] = disagreements(rows, axis, block)
        data["ablations"] = ablations(rows, axis, selected_checks, block, cascade)
        report["axes"][axis] = data
        # Reproduction is tested against the artifact that owns the mapping.
        for split in SPLITS:
            expected = block["report"][split]["raw_rubric_auc"]
            assert abs(expected - data["summary"][split]["auc"]) < 1e-10, (
                axis,
                split,
                expected,
                data["summary"][split],
            )
        print(axis + " " + json.dumps(data["summary"]), flush=True)
        print(
            axis
            + " best calibration term ablations "
            + json.dumps(data["ablations"][:3]),
            flush=True,
        )
        report["controls"][axis] = {
            "record_count": len(records),
            "scored": len(rows),
            "description_nonempty": sum(
                presence(r["state"], "description") == "nonempty" for r in rows
            ),
            "source_docs_nonempty": sum(
                presence(r["state"], "dd_source_docs") == "nonempty" for r in rows
            ),
            "rubric_auc_reproduced": True,
        }
    report["controls"]["name_state_mismatches"] = len(mismatches)
    report["controls"]["docs_receipt_digest_verified"] = True
    report["limitations"] = (
        "Golden truth is whole-item only. Name label_source is provenance, not a defect taxonomy; docs label_check is a broad primary reason, not a complete per-check annotation. "
        "Thus no check has a measured exact defect count, and no check-specific sensitivity can be calibrated from these receipts. "
        "AUC by check and dimension is against overall defensibility. Deterministic docs audit values reproduce the calibration builder with a one-item catalog and known-id set; "
        "this context is narrower than the catalog frozen alongside the receipts. Absent name audit answers are unknown. "
        "Ablations hold every other parameter fixed and replace a term by its calibration mean; they are diagnostics, not refitted or independently validated improvements. "
        "Repeated validation inspection makes a fresh grouped holdout necessary before deployment. Missing evidence and sparse labels do not identify a numerical future AUC gain."
    )
    report["inputs"] = [{"path": str(p.resolve()), "sha256": digest(p)} for p in inputs]
    report["model_revision"] = subprocess.check_output(
        ["git", "-C", str(args.model_root), "rev-parse", "HEAD"], text=True
    ).strip()
    report["improvements"] = recommendations(report)
    encoded = json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert len(encoded.encode()) <= 300000, f"Report too large: {len(encoded.encode())}"
    (OUTPUT / (STEM + ".json")).write_text(encoded)
    (OUTPUT / (STEM + ".html")).write_text(render(report))
    print(
        "SUMMARY "
        + json.dumps(
            {
                "name_validation_auc": report["axes"]["name"]["summary"]["validation"][
                    "auc"
                ],
                "docs_validation_auc": report["axes"]["docs"]["summary"]["validation"][
                    "auc"
                ],
                "checks": sum(len(a["checks"]) for a in report["axes"].values()),
                "dimensions": sum(
                    len(a["dimensions"]) for a in report["axes"].values()
                ),
                "disagreements": sum(
                    len(a["disagreements"]) for a in report["axes"].values()
                ),
                "llm_spend_usd": 0,
                "json_bytes": len(encoded.encode()),
            }
        ),
        flush=True,
    )


def recommendations(report):
    out = []
    for axis in ("name", "docs"):
        best = report["axes"][axis]["ablations"][0]
        out.append(
            {
                "change": f"Drop or downweight {axis} term {best['feature']} in {best['dimension']}",
                "candidate": "drop a noisy check"
                if best["feature"] != "category"
                else "change a dimension's mapping",
                "evidence": f"Highest calibration-selected single-term ablation; replacing the term by its calibration mean, holding every other parameter fixed: {best['splits']}",
                "expected_auc_effect": "The measured deltas above are sensitivity estimates, not promised gains; refit within grouped calibration and test on a fresh holdout.",
            }
        )
    out.extend(
        [
            {
                "change": "Supply source and relationship evidence where the frozen states are empty or unobserved",
                "candidate": "supply missing evidence",
                "evidence": "Per-check evidence tables identify the precise missing fields. The current docs arm already observes units, links and source paths; the earlier diagnosis of wholly absent fields is obsolete. Link target meanings, source ancestor descriptions and family provenance must be distinguished from merely carrying identifier strings.",
                "expected_auc_effect": "Direction expected positive for affected checks; numerical gain is not identifiable without new paired answers. Zero immediate change to frozen scores.",
            },
            {
                "change": "Add check-specific golden labels, especially documentation completeness and name provenance defects",
                "candidate": "add golden labels of a named kind",
                "evidence": "No exact check truth exists. Documentation complete has fewer than five defects; name sources identify how labels were obtained rather than which check should fire. Annotate independent copy groups with multiple applicable checks and explicit acceptable controls.",
                "expected_auc_effect": "Zero direct change to frozen AUC; enables valid check calibration and reduces uncertainty. No evidence supports a point forecast.",
            },
            {
                "change": "Rewrite ambiguous source-comparison questions only after supplying their evidence",
                "candidate": "rewrite a check's question",
                "evidence": "Claim non-support pools contradiction with says-nothing, while the hard veto uses contradiction only. Worst-ranked items expose the answers and evidence needed to distinguish unsupported, contradictory and wrong-object claims. Preserve separate outcomes and never count unobserved evidence as a false claim.",
                "expected_auc_effect": "Unknown until a paired question arm is judged; frozen receipts cannot predict responses to rewritten questions. Keep the hard contradiction rule and measure category-specific recall.",
            },
        ]
    )
    return out


if __name__ == "__main__":
    main()

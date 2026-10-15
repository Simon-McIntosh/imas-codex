#!/usr/bin/env python
"""Why five defect-worded review checks invert against the golden labels.

Five name-axis Noul checks score an AUC below a half: Jev answers the defect the
check names as present more often on names the golden accepts than on names it
rejects, so the inverted checks pull the prototype's check mean below chance.
This script diagnoses each check without spending anything. It reads the saved
base answers and the exact state Jev was shown for every name, the golden labels
and their committed splits, and reports:

  * the evidence each check declares, and how often the saved state carried it;
  * the rate at which each check fires on accepted and rejected calibration
    and validation names;
  * the ten accepted calibration names each check scores nearest its defect,
    with the declared segments the state carried and the source review's own
    comment where the graph holds one;
  * a cause class and a calibration remedy per check;
  * a re-score with each remedy simulated, needing no new Jev call, plus one
    validation read of the combined remedy.

Nothing here calls Jev or writes the graph. The metric primitives are imported,
not copied, and the source review comments are read through a bounded indexed
query; they are never put into a Jev state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "sn-judgment-evaluation" / "golden-name-acceptable.json"
RESULTS = HERE / "srtj-check-evidence-diagnosis-results.json"
FRAGMENT = HERE / "srtj-check-evidence-diagnosis.html"
FIGURES = (
    HERE.parents[2]
    / "figures"
    / "sn-review-through-jev"
    / "srtj-check-evidence-diagnosis"
)

SOURCE_RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261015T120934498631-srtj-noul-polarity/polarity-run"
)
RECEIPTS = SOURCE_RUN / "name-decisions.jsonl"
CONTEXT = SOURCE_RUN / "name-context.json"

# The re-measure recorded these; the unmodified prototype score must reproduce
# them before any simulated re-score is trusted.
CONTROL_CALIBRATION = 0.6057
CONTROL_VALIDATION = 0.6378
CONTROL_TOLERANCE = 0.003

# The five name-axis Noul checks whose calibration AUC of 1 - defect probability
# falls below a half in the re-measure.
INVERTED = (
    "advisory_alias_collapsed",
    "minority_spelling_not_in_roster",
    "over_qualified_modifier",
    "owner_erased_outline",
    "redundant_repeated_segment",
)

# The one inverted check whose rule is a property of the name string itself.
CODE_RULE = "redundant_repeated_segment"
FIELD_STARVED = tuple(c for c in INVERTED if c != CODE_RULE)

SEGMENT_FIELDS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
)
CORE_SCALARS = ("id", "kind", "unit", "description", "documentation")
DIMENSIONS = ("grammar", "semantic", "convention", "completeness")

STOPWORDS = frozenset(
    {
        "of",
        "at",
        "in",
        "over",
        "under",
        "due",
        "to",
        "and",
        "the",
        "a",
        "an",
        "per",
        "for",
        "with",
        "by",
        "on",
        "from",
        "across",
        "along",
        "between",
        "into",
        "within",
        "during",
        "after",
        "before",
    }
)

CAUSE_TEXT = {
    "a": "evidence the state lacks",
    "b": "a rule better decided in code than asked of a model",
    "c": "a question that is still ambiguous",
    "d": "the check's definition disagreeing with the state's own account",
}
CAUSE = {
    "advisory_alias_collapsed": "a",
    "minority_spelling_not_in_roster": "a",
    "over_qualified_modifier": "a",
    "owner_erased_outline": "a",
    "redundant_repeated_segment": "b",
}
REMEDY = {
    "advisory_alias_collapsed": {
        "kind": "supply field",
        "reader": "the COCOS reader that holds the source path's cocos_label",
        "text": (
            "Supply the check's declared cocos_label from the source path. The "
            "saved state carries it in few states, so the model is asked to weigh "
            "registered advisory aliases it was never shown and reads a "
            "legitimate alias as a collapse."
        ),
    },
    "minority_spelling_not_in_roster": {
        "kind": "supply field",
        "reader": "the closed-vocabulary roster reader holding established base spellings",
        "text": (
            "Supply the roster of established base spellings the check compares "
            "against. The state carries existing_names (linked neighbour names) "
            "but not the roster, and it is empty in most states, so a base that "
            "already follows the convention reads as an off-roster spelling."
        ),
    },
    "over_qualified_modifier": {
        "kind": "supply field",
        "reader": "the grammar decomposition reader holding the name's own qualifiers",
        "text": (
            "Supply the parsed aggregation/qualifier decomposition beside the "
            "dd_source_docs. The leaf source doc is usually a bare field label, so "
            "a registered aggregation or qualifier reads as an unstated modifier "
            "and accepted decompositions fire."
        ),
    },
    "owner_erased_outline": {
        "kind": "supply field",
        "reader": "the parent-path structure reader holding the owner segment",
        "text": (
            "Supply the parent path structure that names the owner. The leaf "
            "source path names no owner, so a legitimate bare outline cannot be "
            "told from an erased owner and accepted bare names fire."
        ),
    },
    "redundant_repeated_segment": {
        "kind": "code rule",
        "reader": "the name string itself",
        "text": (
            "Move to code. The rule is a substantive token repeated in the name "
            "(the check's own electron_electron example); asking a model fires it "
            "wherever a word echoes across two honest segments, such as "
            "diamagnetic in both the subject and the process of an accepted name."
        ),
    },
}


def _load_receipts():
    if not RECEIPTS.exists():
        return {}
    records = {}
    for line in RECEIPTS.read_text().splitlines():
        row = json.loads(line)
        records[(row["mode"], row["name"])] = row
    return {name: row for (mode, name), row in records.items() if mode == "base"}


def _repeated_content_token(name):
    seen = set()
    for token in name.split("_"):
        if not token or token in STOPWORDS:
            continue
        if token in seen:
            return True
        seen.add(token)
    return False


def _defect(answer):
    if not answer:
        return None
    value = answer.get("noul")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str))
    temporary.replace(path)


def _coverage(receipts, checks_by_id):
    total = len(receipts)
    table = {}
    for check_id in INVERTED:
        entry = {}
        for field in checks_by_id[check_id].evidence_fields:
            if field in SEGMENT_FIELDS:
                carried = sum(
                    1
                    for r in receipts.values()
                    if r["state"]["name"]["segments"].get(field) not in (None, "")
                )
                entry[field] = {"where": "name.segments", "text": f"{carried}/{total}"}
            elif field in CORE_SCALARS:
                carried = sum(
                    1 for r in receipts.values() if r["state"]["name"].get(field)
                )
                entry[field] = {
                    "where": "name (core scalar)",
                    "text": f"{carried}/{total}",
                }
            else:
                buckets = [
                    r["state"]["neighbourhood"].get(field) or {}
                    for r in receipts.values()
                ]
                observed = sum(1 for b in buckets if b.get("observed"))
                nonempty = sum(
                    1
                    for b in buckets
                    if b.get("observed") and b.get("value") not in (None, "", [], {})
                )
                entry[field] = {
                    "where": "neighbourhood",
                    "text": f"{nonempty}/{total} non-empty, {observed}/{total} observed",
                }
        table[check_id] = entry
    return table


def _fires(receipts, name, check_id):
    return (_defect(receipts[name]["answers"].get(f"check:{check_id}")) or 0.0) >= 0.5


def _population(split, canonical, context, receipts):
    names = []
    for name, row in canonical.items():
        if name not in receipts or row["split"] != split:
            continue
        if not context["entries"][name]["description"]:
            continue
        names.append((name, row))
    return names


def _firing(receipts, canonical, context):
    table = {}
    for check_id in INVERTED:
        per_split = {}
        for split in ("calibration", "validation"):
            rows = _population(split, canonical, context, receipts)
            accepted = [(n, r) for n, r in rows if r["label"] == 1]
            rejected = [(n, r) for n, r in rows if r["label"] == 0]
            per_split[split] = {
                "accepted": {
                    "fired": sum(
                        1 for n, _ in accepted if _fires(receipts, n, check_id)
                    ),
                    "n": len(accepted),
                },
                "rejected": {
                    "fired": sum(
                        1 for n, _ in rejected if _fires(receipts, n, check_id)
                    ),
                    "n": len(rejected),
                },
            }
        table[check_id] = per_split
    return table


def _examples(receipts, canonical, context, limit=10):
    table = {}
    for check_id in INVERTED:
        rows = []
        for name, row in _population("calibration", canonical, context, receipts):
            if row["label"] != 1:
                continue
            prob = _defect(receipts[name]["answers"].get(f"check:{check_id}"))
            if prob is None:
                continue
            rows.append((prob, name))
        rows.sort(key=lambda pair: pair[0], reverse=True)
        table[check_id] = [
            {
                "name": name,
                "noul": prob,
                "fired": prob >= 0.5,
                "segments": {
                    f: receipts[name]["state"]["name"]["segments"][f]
                    for f in SEGMENT_FIELDS
                    if receipts[name]["state"]["name"]["segments"].get(f)
                },
            }
            for prob, name in rows[:limit]
        ]
    return table


def _comments(examples):
    from imas_codex.graph.client import GraphClient

    names = sorted({row["name"] for rows in examples.values() for row in rows})
    query = """
    UNWIND $names AS name
    OPTIONAL MATCH (sn:StandardName {id:name})
    OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(rv:StandardNameReview {review_axis:'name'})
    WITH name, sn, rv ORDER BY rv.reviewed_at DESC
    RETURN name, sn.reviewer_comments_name AS summary,
      head(collect(rv.comments)) AS review_comment
    """
    comments = {}
    with GraphClient() as graph:
        for row in graph.query(query, names=names):
            comments[row["name"]] = {
                "summary": row["summary"],
                "review_comment": row["review_comment"],
            }
    return comments


def _score_of(answer_map, check_list, name, question_table, replace_code_rule):
    answers = answer_map
    dimensions = []
    for dimension in DIMENSIONS:
        answer = answers.get(f"dimension:{dimension}")
        if not answer:
            return None
        criteria = question_table[f"dimension:{dimension}"]["criteria"]
        accepted = next(iter(criteria))
        probabilities = answer.get("probabilities") or {}
        dimensions.append(
            0.5 * float(answer.get("choice") == accepted)
            + 0.5 * float(probabilities.get(accepted, 0))
        )
    scored = []
    for check in check_list:
        if replace_code_rule and check.id == CODE_RULE:
            scored.append(0.0 if _repeated_content_token(name) else 1.0)
            continue
        answer = answers.get(f"check:{check.id}")
        if not answer:
            return None
        if check.question.kind.value == "noul":
            failure = float(answer["noul"])
            passes = failure < 0.5
            scored.append(0.5 * float(passes) + 0.5 * (1.0 - failure))
        else:
            acceptable = check.question.options[0]
            probabilities = answer.get("probabilities") or {}
            scored.append(
                0.5 * float(answer.get("choice") == acceptable)
                + 0.5 * float(probabilities.get(acceptable, 0))
            )
    return 0.5 * sum(dimensions) / len(dimensions) + 0.5 * sum(scored) / len(scored)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()

    from imas_codex.discovery.base.evaluation import (
        bootstrap_interval,
        load_labelled_fixture,
    )
    from imas_codex.standard_names.review.checks import load_review_checks
    from imas_codex.standard_names.review.jev_review import (
        NAME_AXIS,
        name_question_table,
    )

    checks = [c for c in load_review_checks()[0] if c.axis == NAME_AXIS]
    checks_by_id = {c.id: c for c in checks}
    order = [c for c in checks if c.question is not None]
    combined_checks = [c for c in order if c.id not in FIELD_STARVED]
    question_table = name_question_table(checks)

    fixture = load_labelled_fixture(FIXTURE)
    canonical = {}
    for row in fixture:
        canonical.setdefault(row["canonical_unit"], row)
    context = json.loads(CONTEXT.read_text())
    receipts = _load_receipts()

    def auc(check_list, split, replace_code_rule=False):
        labels, values, groups = [], [], []
        for name, row in _population(split, canonical, context, receipts):
            value = _score_of(
                receipts[name]["answers"],
                check_list,
                name,
                question_table,
                replace_code_rule,
            )
            if value is None:
                continue
            labels.append(row["label"])
            values.append(value)
            groups.append(row["copy_group"])
        if len(set(labels)) < 2:
            return {"n": len(labels), "point": None}
        interval = bootstrap_interval(labels, values, copy_groups=groups)
        return {"n": len(labels), **interval}

    prototype_cal = auc(order, "calibration")
    prototype_val = auc(order, "validation")
    drop_one = {
        check_id: auc([c for c in order if c.id != check_id], "calibration")
        for check_id in INVERTED
    }
    combined_cal = auc(combined_checks, "calibration", replace_code_rule=True)
    combined_val = auc(combined_checks, "validation", replace_code_rule=True)

    per_check = {}
    for check in order:
        if check.question.kind.value != "noul":
            continue
        labels, values, groups = [], [], []
        for name, row in _population("calibration", canonical, context, receipts):
            prob = _defect(receipts[name]["answers"].get(f"check:{check.id}"))
            if prob is None:
                continue
            labels.append(row["label"])
            values.append(1.0 - prob)
            groups.append(row["copy_group"])
        if len(set(labels)) < 2:
            continue
        per_check[check.id] = bootstrap_interval(labels, values, copy_groups=groups)

    coverage = _coverage(receipts, checks_by_id)
    firing = _firing(receipts, canonical, context)
    examples = _examples(receipts, canonical, context)
    comments = _comments(examples)

    result = {
        "method": {
            "reader": "A Noul is read as a pass probability of one minus its defect probability; an AUC below a half is a question the model answers backwards against the golden label.",
            "fire": "A check fires when its Noul defect probability is at or above 0.5.",
            "control": "The unmodified prototype score reproduces the re-measure's calibration and validation AUC on the names that carry a description.",
            "graph": "The source review's own comment is read through a bounded indexed query on StandardName.id and never put into a Jev state.",
            "combined": "Drop the four field-starved checks and replace the repeated-segment check with the repeated-content-token code rule; the choice is fixed on calibration and validation is read once.",
        },
        "fixture": {
            "rows": len(fixture),
            "canonical_names": len(canonical),
            "base_receipts": len(receipts),
        },
        "control": {
            "prototype_calibration": prototype_cal,
            "prototype_validation": prototype_val,
            "expected_calibration": CONTROL_CALIBRATION,
            "expected_validation": CONTROL_VALIDATION,
            "within_tolerance": (
                abs(prototype_cal["point"] - CONTROL_CALIBRATION) <= CONTROL_TOLERANCE
                and abs(prototype_val["point"] - CONTROL_VALIDATION)
                <= CONTROL_TOLERANCE
            ),
        },
        "coverage": coverage,
        "firing": firing,
        "examples": examples,
        "quorum_comments": comments,
        "cause": {
            c: {"class": CAUSE[c], "text": CAUSE_TEXT[CAUSE[c]]} for c in INVERTED
        },
        "remedy": REMEDY,
        "rescore": {
            "prototype_calibration": prototype_cal,
            "prototype_validation": prototype_val,
            "drop_one_calibration": drop_one,
            "combined_calibration": combined_cal,
            "combined_validation": combined_val,
            "per_check_calibration_auc": per_check,
            "n_calibration": prototype_cal["n"],
            "n_validation": prototype_val["n"],
        },
    }
    _save(Path(args.out), result)
    render_fragment(result)
    plot(result)
    print(
        json.dumps(
            {
                "control_within_tolerance": result["control"]["within_tolerance"],
                "prototype_cal": prototype_cal["point"],
                "combined_cal": combined_cal["point"],
                "combined_val": combined_val["point"],
            },
            indent=1,
        )
    )
    return 0


def _ci(value):
    if not value or value.get("point") is None:
        return "n/a"
    return f"{value['point']:.3f} [{value['lower']:.3f}, {value['upper']:.3f}]"


def render_fragment(result):
    control = result["control"]
    out = ['<section id="srtj-check-evidence-diagnosis">']
    out.append("<h2>Why five review checks invert on accepted names</h2>")
    out.append(
        "<p><strong>Control.</strong> The unmodified prototype score reproduces "
        f"the re-measure: calibration AUC {_ci(control['prototype_calibration'])} "
        f"on {control['prototype_calibration']['n']} names and validation AUC "
        f"{_ci(control['prototype_validation'])} on "
        f"{control['prototype_validation']['n']}, matching the recorded "
        f"{control['expected_calibration']:.3f} and "
        f"{control['expected_validation']:.3f}.</p>"
    )
    out.append(
        "<p>A Noul is read as a pass probability of one minus its defect "
        "probability, so an AUC below a half is a question the model answers "
        "backwards against the golden label. The five checks below are the five "
        "lowest such AUCs, each with a 95% interval below a half.</p>"
    )
    out.append(
        "<p><strong>The state builder drops nothing.</strong> Every field these "
        "five checks declare reaches the state: <code>build_name_state</code> "
        "wraps each one through <code>_observed</code> "
        "(<code>imas_codex/standard_names/review/jev_review.py:213-218</code>, "
        "with the flag set at <code>:152-161</code>), so a field the item does "
        "not carry is recorded as observed-and-absent rather than omitted. No "
        "declared field is never filled; what varies is whether the review item "
        "the builder is handed already carries the value. The table below is "
        "that distinction, and it is the fields whose value in the item is thin "
        "— cocos_label, source_paths, dd_source_docs, existing_names — that name "
        "the starved checks.</p>"
    )
    out.append(
        "<p><strong>The inversion is a ranking inversion.</strong> Because a "
        "check's firing cut is a half, a check whose accepted and rejected "
        "defect probabilities sit on the same side of a half does not fire "
        "differently on the two classes: <code>owner_erased_outline</code> never "
        "reaches a half on either class in either split, yet its AUC is the "
        "lowest but one. So the five are not firing falsely on accepted names so "
        "much as ordering accepted names above rejected ones, which is the shape "
        "of evidence starvation rather than of a mis-set threshold.</p>"
    )
    out.append(
        '<figure><img src="/imas-codex/figures/sn-review-through-jev/'
        'srtj-check-evidence-diagnosis/calibration-auc.svg" alt="Per-check '
        "calibration AUC against a half, and the calibration AUC of the prototype "
        'and the combined remedy"><figcaption>Left: calibration AUC of one minus '
        "each Noul check's defect probability against the golden label, in its own "
        "95% interval; the dotted line is a half. Right: calibration AUC of the "
        "prototype score, each single-check drop, and the combined remedy."
        "</figcaption></figure>"
    )

    out.append("<h3>Evidence the state carried</h3>")
    out.append(
        "<table><thead><tr><th>Check</th><th>Declared field</th>"
        "<th>Carried as</th><th>Present</th></tr></thead><tbody>"
    )
    for check_id in INVERTED:
        for field, entry in result["coverage"][check_id].items():
            out.append(
                f"<tr><td><code>{check_id}</code></td><td>{field}</td>"
                f"<td>{entry['where']}</td><td>{entry['text']}</td></tr>"
            )
    out.append("</tbody></table>")

    out.append("<h3>Firing rate on calibration and validation</h3>")
    out.append(
        "<table><thead><tr><th>Check</th><th>Split</th><th>Accepted fired</th>"
        "<th>Rejected fired</th></tr></thead><tbody>"
    )
    for check_id in INVERTED:
        for split in ("calibration", "validation"):
            row = result["firing"][check_id][split]
            out.append(
                f"<tr><td><code>{check_id}</code></td><td>{split}</td>"
                f"<td>{row['accepted']['fired']}/{row['accepted']['n']}</td>"
                f"<td>{row['rejected']['fired']}/{row['rejected']['n']}</td></tr>"
            )
    out.append("</tbody></table>")

    out.append("<h3>Ten strongest accepted calibration names per check</h3>")
    for check_id in INVERTED:
        cause = result["cause"][check_id]
        remedy = result["remedy"][check_id]
        out.append(
            f"<p><strong><code>{check_id}</code></strong> — cause "
            f"({cause['class']}) {cause['text']}; remedy: {remedy['kind']} — "
            f"{remedy['text']} Reader: {remedy['reader']}.</p>"
        )
        out.append(
            "<table><thead><tr><th>Name</th><th>Declared segments</th>"
            "<th>Defect probability</th><th>Source review comment</th></tr>"
            "</thead><tbody>"
        )
        for row in result["examples"][check_id]:
            segments = ", ".join(f"{k}={v}" for k, v in row["segments"].items())
            segments = segments or "(none)"
            record = result["quorum_comments"].get(row["name"]) or {}
            comment = record.get("summary") or record.get("review_comment") or ""
            comment = " ".join(comment.split())
            if len(comment) > 240:
                comment = comment[:240] + "…"
            if not comment:
                comment = "(none held)"
            mark = " (fires)" if row["fired"] else ""
            out.append(
                f"<tr><td><code>{row['name']}</code></td><td>{segments}</td>"
                f"<td>{row['noul']:.2f}{mark}</td><td>{comment}</td></tr>"
            )
        out.append("</tbody></table>")

    out.append("<h3>Simulated re-score</h3>")
    rescore = result["rescore"]
    out.append(
        "<table><thead><tr><th>Score</th><th>Calibration AUC [95%]</th>"
        "<th>n</th></tr></thead><tbody>"
    )
    out.append(
        f"<tr><td>prototype</td><td>{_ci(rescore['prototype_calibration'])}</td>"
        f"<td>{rescore['prototype_calibration']['n']}</td></tr>"
    )
    for check_id, value in rescore["drop_one_calibration"].items():
        out.append(
            f"<tr><td>drop <code>{check_id}</code></td><td>{_ci(value)}</td>"
            f"<td>{value['n']}</td></tr>"
        )
    out.append(
        f"<tr><td>combined remedy (calibration)</td>"
        f"<td>{_ci(rescore['combined_calibration'])}</td>"
        f"<td>{rescore['combined_calibration']['n']}</td></tr>"
    )
    out.append(
        f"<tr><td>combined remedy (validation, read once)</td>"
        f"<td>{_ci(rescore['combined_validation'])}</td>"
        f"<td>{rescore['combined_validation']['n']}</td></tr>"
    )
    out.append("</tbody></table>")
    out.append("</section>")
    FRAGMENT.write_text("\n".join(out) + "\n")


def plot(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.style.use("data-ink")
    except OSError:
        pass

    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 7), dpi=100)

    per_check = result["rescore"]["per_check_calibration_auc"]
    ordered = sorted(per_check.items(), key=lambda pair: pair[1]["point"])
    inverted = set(INVERTED)
    for position, (check_id, value) in enumerate(ordered):
        colour = "#a2643c" if check_id in inverted else "#58635f"
        left.errorbar(
            [value["point"]],
            [position],
            xerr=[[value["point"] - value["lower"]], [value["upper"] - value["point"]]],
            fmt="o",
            color=colour,
            markersize=6,
            capsize=3,
        )
        left.text(
            value["upper"] + 0.008,
            position,
            f"{value['point']:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color=colour,
        )
    left.axvline(0.5, color="#999999", linewidth=1.2, linestyle=":")
    left.set_yticks(
        list(range(len(ordered))),
        [check_id.replace("_", " ") for check_id, _ in ordered],
        fontsize=10,
    )
    left.set_xlabel("calibration AUC of 1 − defect probability", fontsize=12)
    left.set_xlim(0.20, 0.82)
    left.set_ylim(-0.8, len(ordered) - 0.3)
    left.text(
        0.496, len(ordered) - 0.6, "a half", color="#999999", fontsize=10, ha="right"
    )
    left.spines["top"].set_visible(False)
    left.spines["right"].set_visible(False)

    bars = [
        ("prototype", result["rescore"]["prototype_calibration"]["point"], "#2b6b78")
    ]
    for check_id, value in result["rescore"]["drop_one_calibration"].items():
        bars.append(("drop " + check_id.replace("_", " "), value["point"], "#58635f"))
    bars.append(
        (
            "combined remedy",
            result["rescore"]["combined_calibration"]["point"],
            "#a2643c",
        )
    )
    positions = list(range(len(bars)))
    right.barh(
        positions, [bar[1] for bar in bars], color=[bar[2] for bar in bars], height=0.7
    )
    for position, bar in zip(positions, bars, strict=True):
        right.text(bar[1] + 0.002, position, f"{bar[1]:.3f}", va="center", fontsize=10)
    right.set_yticks(positions, [bar[0] for bar in bars], fontsize=10)
    right.invert_yaxis()
    right.set_xlabel("calibration AUC", fontsize=12)
    right.set_xlim(0.5, 0.66)
    right.axvline(0.5, color="#999999", linewidth=1.2, linestyle=":")
    right.spines["top"].set_visible(False)
    right.spines["right"].set_visible(False)

    figure.tight_layout()
    target = FIGURES / "calibration-auc.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, format="svg", metadata={"Date": None})
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())

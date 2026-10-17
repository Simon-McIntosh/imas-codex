#!/usr/bin/env python
"""Measure a per-name Jev domain Choice against two rule controls.

One TypeSafe Choice is asked per labelled standard name. Its options are only
the candidates the Data Dictionary supplies for that name: the stored domains
of its source paths, the graph domains of its source IDSs, the band umbrella of
each band those candidates fall in, and ``general``. Code builds the question
from the guidance file; the model supplies the judgment.

The two controls are re-measured at the current graph with the preceding rule
comparison's own selectors: the set-precedence rule and the IDS-first rule with
the corrected IDS map the graph now carries. Accuracy intervals are the harness's
paired-group bootstrap; each control difference is the paired bootstrap of the
same names. The adoption rule is written into the results JSON with a timestamp
before any validation call is made, and validation is read once only if the rule
is met.

Modes
  --probe          score one name, print its answer and the spend, then stop.
  (default)        score approved and calibration, evaluate, and write results.
  --continue       resume from saved approved/calibration answers and run the
                   remaining stage (validation, then figure and fragment).
  --rescore        rebuild the fragment and figure from saved results, no spend.
  --shuffle-check  re-score the saved answers with labels shuffled within each
                   cohort; prints EXIT= and refuses when the calibration drop
                   does not exceed the bootstrap half-width.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt

from imas_codex.discovery.base.evaluation import bootstrap_interval
from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.client import GraphClient
from imas_codex.graph.dd_domain_classifier import SERVICE_TAG

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STEM = Path(__file__).stem.removesuffix("-run")
RESULTS = HERE / f"{STEM}-results.json"
FRAGMENT = HERE / f"{STEM}.html"
FIGURE = ROOT / "docs/figures/sn-catalog-grouping" / STEM / "accuracy.svg"
GUIDANCE = ROOT / "imas_codex/definitions/physics/domains.yaml"

SEAT = "sn-domain-tree"
QUESTION = "physics_domain"
CONCURRENCY = 8
SPEND_CEILING_USD = 0.50
SHUFFLE_SEED = 20261017

RULE = (
    "Jev meets the bar only if it beats both controls on the approved set and "
    "on the calibration set: strictly more correct names on each. Validation "
    "labels are read once, and only if the bar is met."
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("source_domain_rules", HERE / "scg-name-domain-rule-measure-run.py")
design = _load("ids_domain_design", HERE / "scg-ids-domain-design-run.py")
umb = _load("band_umbrella", HERE / "scg-name-domain-umbrella-measure-run.py")

UMBRELLAS = umb.UMBRELLAS
SCOPES = design.SCOPES
BAND_OF_NAME = {
    domain: band["band"] for band in design.BANDS for domain in band["domains"]
}
assert set(BAND_OF_NAME) == set(design.RANK)
assert {band["band"] for band in design.BANDS} == set(UMBRELLAS)
CONTROL_SET = "set_precedence"
CONTROL_IDS = "ids_first_tier_then_band"
ARMS = (CONTROL_SET, CONTROL_IDS, "jev")
ARM_LABELS = {
    CONTROL_SET: "set precedence",
    CONTROL_IDS: "IDS-first (graph map)",
    "jev": "Jev choice",
}
EARLIER = {
    "set_precedence": {"approved": 9, "calibration": 53, "validation": 52},
    "ids_first_tier_then_band_corrected": {
        "approved": 9,
        "calibration": 55,
        "validation": 54,
    },
}
COHORTS = ("approved", "calibration", "validation")


def model_id() -> str:
    from imas_codex.settings import get_model

    return get_model(SEAT)


def graph_ids_map() -> dict[str, str]:
    """Each IDS's physics domain as the graph holds it."""
    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (n:IDS) WHERE n.physics_domain IS NOT NULL "
            "RETURN n.id AS id, n.physics_domain AS domain"
        )
    mapping = {row["id"]: row["domain"] for row in rows}
    assert mapping, "known-present IDS domain read returned nothing"
    assert mapping.get("core_profiles"), "IDS domain read missed core_profiles"
    return mapping


def ids_of(path: str) -> str:
    return path.removeprefix("dd:").split("/", 1)[0]


def option_text(domain: str) -> str:
    """One domain's scope, includes and excludes from the guidance file."""
    meta = SCOPES.get(domain) or {}
    parts = [str(meta.get("scope") or "").strip()]
    if meta.get("includes"):
        parts.append("Includes: " + "; ".join(str(i) for i in meta["includes"]))
    for exclusion in meta.get("excludes") or []:
        parts.append(
            "Excludes {quantity} (-> {neighbour})".format(
                quantity=exclusion.get("quantity", ""),
                neighbour=exclusion.get("neighbour", ""),
            )
            + (f": {exclusion['rule']}" if exclusion.get("rule") else "")
        )
    return "\n".join(part for part in parts if part)


def candidates(sources: dict[str, str | None], ids_map: dict[str, str]) -> list[str]:
    """The domains the DD supplies for a name: its paths', its IDSs' and umbrellas."""
    found: set[str] = {"general"}
    for path, domain in sources.items():
        if domain:
            found.add(domain)
        prior = ids_map.get(ids_of(path))
        if prior:
            found.add(prior)
    for domain in list(found):
        band = BAND_OF_NAME.get(domain)
        if band:
            found.add(UMBRELLAS[band])
    return sorted(found)


def path_counts(sources: dict[str, str | None]) -> dict[str, int]:
    counts = Counter(d for d in sources.values() if d)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_state(
    row: dict,
    sources: dict[str, str | None],
    ids_map: dict[str, str],
    options: list[str],
) -> dict:
    source_ids = sorted({ids_of(path) for path in sources})
    return {
        "name": row["name"],
        "description": row.get("description") or "",
        "source_paths": [
            {"path": path, "stored_domain": domain or "unclassified"}
            for path, domain in sorted(sources.items())
        ],
        "source_ids": [
            {"ids": ids, "graph_domain": ids_map.get(ids) or "unclassified"}
            for ids in source_ids
        ],
        "source_path_count_by_candidate_domain": {
            domain: path_counts(sources).get(domain, 0) for domain in options
        },
    }


def build_question(options: list[str]) -> dict:
    instructions = (
        "Choose the one physics domain that best describes the standard name "
        "given in the state, judging what the quantity measures rather than "
        "which IDS carries it. Every option is a domain the Data Dictionary "
        "supplies for this name: a stored domain of one of its source paths, "
        "the domain of one of its source IDSs, a broader band umbrella one of "
        "those falls under, or general. Prefer the most specific option that "
        "plainly fits the quantity; choose general only when no narrower "
        "option fits. The options are listed most specific first."
    )
    order = sorted(options, key=lambda d: (BAND_OF_NAME[d] != "general", d))
    return {
        QUESTION: {
            "type": "choice",
            "instructions": instructions,
            "criteria": {domain: option_text(domain) for domain in order},
        }
    }


def answer_record(answer) -> dict | None:
    if not answer:
        return None
    probabilities = answer.get("probabilities") or {}
    return {
        "choice": answer.get("choice"),
        "confidence": answer.get("confidence"),
        "probabilities": {str(k): float(v) for k, v in probabilities.items()},
        "probability_of_choice": float(
            {str(k): float(v) for k, v in probabilities.items()}.get(
                str(answer.get("choice")), 0.0
            )
        ),
    }


async def score_one(state: dict, question: dict, model: str) -> dict:
    results, cost = await decide_batch(
        [state],
        question,
        model=model,
        service=SERVICE_TAG,
        concurrency=CONCURRENCY,
    )
    answers = results[0][0] if results and results[0] else None
    answer = answers.get(QUESTION) if isinstance(answers, dict) else None
    return {"answer": answer_record(answer), "cost_usd": cost}


def interval(correct: list[bool]) -> dict:
    return base.interval(correct)


def paired_difference(a: list[bool], b: list[bool]) -> dict:
    """Paired bootstrap of the accuracy difference a - b over the same names."""
    labels = [label for _ in a for label in (0.0, 1.0)]
    va = [score for hit in a for score in (float(not hit), 1.0)]
    vb = [score for hit in b for score in (float(not hit), 1.0)]
    groups = [index for index in range(len(a)) for _ in (0, 1)]
    result = bootstrap_interval(labels, va, scores_b=vb, copy_groups=groups)
    return {
        "count": len(a),
        "correct_this": sum(a),
        "correct_this_name": "jev",
        "correct_other": sum(b),
        "difference": (sum(a) - sum(b)) / len(a),
        "lower": 2 * result["lower"],
        "upper": 2 * result["upper"],
    }


def jev_correct(rows: list[dict], answers: dict[str, dict]) -> list[bool]:
    return [
        (answers.get(row["name"]) or {}).get("choice") == row["label"] for row in rows
    ]


def control_correct(
    rows: list[dict],
    sources: dict[str, dict[str, str | None]],
    ids_map: dict[str, str],
    rule: str,
) -> list[bool]:
    if rule == CONTROL_SET:
        return [
            base.choose_domain(sources[row["name"]], CONTROL_SET) == row["label"]
            for row in rows
        ]
    return [
        design.choose(sources[row["name"]], CONTROL_IDS, ids_map) == row["label"]
        for row in rows
    ]


def arm_answers(
    rows: list[dict],
    sources: dict[str, dict[str, str | None]],
    ids_map: dict[str, str],
) -> dict[str, list[bool]]:
    return {
        CONTROL_SET: control_correct(rows, sources, ids_map, CONTROL_SET),
        CONTROL_IDS: control_correct(rows, sources, ids_map, CONTROL_IDS),
        "jev": [],
    }


def shuffled(rows: list[dict], seed: int) -> list[dict]:
    """The same rows with the labels permuted within the cohort."""
    rng = random.Random(seed)
    labels = [row["label"] for row in rows]
    rng.shuffle(labels)
    return [dict(row, label=label) for row, label in zip(rows, labels, strict=True)]


def accuracy_above_probability(
    rows: list[dict], answers: dict[str, dict], threshold: float
) -> dict:
    kept = [
        row
        for row in rows
        if ((answers.get(row["name"]) or {}).get("probability_of_choice") or 0.0)
        > threshold
    ]
    correct = sum(
        (answers.get(row["name"]) or {}).get("choice") == row["label"] for row in kept
    )
    return {
        "threshold": threshold,
        "count": len(kept),
        "correct": correct,
        "accuracy": (correct / len(kept)) if kept else None,
    }


def make_figure(metrics: dict, cohorts: list[str]) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    style = Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle"
    if style.exists():
        plt.style.use(style)
    plt.rcParams["svg.hashsalt"] = "name-domain-jev"
    colours = {CONTROL_SET: "#1a1a1a", CONTROL_IDS: "#929292", "jev": "#286c83"}
    fig, axes = plt.subplots(1, len(cohorts), figsize=(14, 5.2), sharey=True)
    if len(cohorts) == 1:
        axes = [axes]
    direct: list[tuple[object, object, float]] = []
    for axis, cohort in zip(axes, cohorts, strict=True):
        for index, arm in enumerate(ARMS):
            item = metrics[cohort][arm]
            axis.errorbar(
                item["accuracy"],
                index,
                xerr=[
                    [item["accuracy"] - item["lower"]],
                    [item["upper"] - item["accuracy"]],
                ],
                fmt="o",
                color=colours[arm],
                linewidth=3.0 if arm == "jev" else 2.4,
                capsize=3,
                markersize=8,
                zorder=3 if arm == "jev" else 2,
            )
            label = axis.annotate(
                f"{item['correct']}/{item['count']}",
                (item["accuracy"], index),
                xytext=(0, -22),
                textcoords="offset points",
                ha="center",
                fontsize=14,
                color=colours[arm],
            )
            direct.append((label, axis, index))
        axis.set_xlim(0, 1.02)
        axis.set_xticks((0.25, 0.5, 0.75))
        axis.set_xlabel("accuracy")
        axis.set_title(cohort, fontsize=16)
        axis.set_yticks(range(len(ARMS)))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    axes[0].set_yticklabels([ARM_LABELS[arm] for arm in ARMS], fontsize=14)
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(ARMS) - 0.3, -0.7)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=colours[arm],
            label=ARM_LABELS[arm],
        )
        for arm in ARMS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.19, top=0.9, wspace=0.1)
    fig.canvas.draw()
    canvas = fig.bbox
    boxes: list[tuple[str, object]] = []
    for artist in fig.findobj(match=plt.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(fig.canvas.get_renderer())
        assert canvas.contains(*box.get_points()[0]) and canvas.contains(
            *box.get_points()[1]
        ), artist.get_text()
        for prior, prior_box in boxes:
            assert not box.overlaps(prior_box), (prior, artist.get_text())
        boxes.append((artist.get_text(), box))
    for label, axis, index in direct:
        box = label.get_window_extent(fig.canvas.get_renderer())
        mark_y = axis.transData.transform((0, index))[1]
        assert box.y1 < mark_y - 2, label.get_text()
        if index + 1 < len(ARMS):
            next_y = axis.transData.transform((0, index + 1))[1]
            assert box.y0 > next_y + 2, label.get_text()
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)
    for path in FIGURE.parent.glob("*"):
        if path.suffix != ".svg":
            path.unlink()


def metric_text(item: dict | None) -> str:
    if item is None:
        return "not read"
    return (
        f"{item['correct']}/{item['count']} "
        f"({item['accuracy'] or 0:.3f}; {item['lower']:.3f}-{item['upper']:.3f})"
    )


def render(payload: dict) -> None:
    state = payload["state"]
    pre = payload["pre_registration"]
    jev = payload["jev"]
    lines = [
        f'<section id="{STEM}">',
        "<h2>A per-name Jev domain Choice measured against two rule controls</h2>",
        "<p>One TypeSafe Choice per labelled standard name, on a "
        f"<code>{escape(payload['model'])}</code> seat. The options are only the "
        "domains the Data Dictionary supplies for that name: the stored domains "
        "of its source paths, the graph domains of its source IDSs, the band "
        "umbrella of each band those candidates fall in, and <code>general</code>. "
        "Each option carries its scope, includes and excludes rendered from "
        "<code>imas_codex/definitions/physics/domains.yaml</code>. Code builds "
        "the question; the model supplies only the judgment. No graph write was "
        "made.</p>",
        "<p>The two controls are re-measured at the current graph with the "
        "preceding rule comparison's own selectors: set precedence, and the "
        "IDS-first rule carrying the corrected IDS map the graph now holds (read "
        "from <code>IDS.physics_domain</code>, not the JSON). The IDS map has "
        "changed since the controls were last measured, so this run's figures are "
        "the ones that stand.</p>",
        "<h3>Controls at the current graph</h3>",
        "<table><tr><th>Rule</th><th>Approved</th><th>Calibration</th>"
        "<th>Validation</th><th>Earlier approved/calibration/validation</th></tr>",
    ]
    earlier_key = {
        CONTROL_SET: "set_precedence",
        CONTROL_IDS: "ids_first_tier_then_band_corrected",
    }
    for arm in (CONTROL_SET, CONTROL_IDS):
        cells = "".join(
            f"<td>{metric_text(payload['metrics'][cohort][arm])}</td>"
            for cohort in COHORTS
        )
        old = EARLIER[earlier_key[arm]]
        lines.append(
            f"<tr><td>{escape(ARM_LABELS[arm])}</td>{cells}"
            f"<td>{old['approved']}/20, {old['calibration']}/73, "
            f"{old['validation']}/77</td></tr>"
        )
    lines.append(
        "</table><p>Set precedence reproduces the earlier figures exactly, so the "
        "harness is the earlier measurement's. The IDS-first control moves only "
        "because the graph's IDS map changed.</p>"
    )
    lines.append("<h3>The Jev Choice</h3>")
    lines.append(
        f"<p>Approved plus calibration: Jev {payload['metrics']['approved']['jev']['correct']}"
        f"/20 and {payload['metrics']['calibration']['jev']['correct']}/73. "
    )
    lines.append(
        "Paired differences against each control (Jev minus control), 95% paired "
        "bootstrap intervals:</p>"
    )
    lines.append(
        "<table><tr><th>Control</th><th>Cohort</th><th>Jev - control</th>"
        "<th>Interval</th></tr>"
    )
    for arm in (CONTROL_SET, CONTROL_IDS):
        for cohort in ("approved", "calibration"):
            paired = payload["paired_by_cohort"][cohort][arm]
            lines.append(
                f"<tr><td>{escape(ARM_LABELS[arm])}</td><td>{escape(cohort)}</td>"
                f"<td>{paired['difference']:+.3f} "
                f"({paired['correct_this']} vs {paired['correct_other']})</td>"
                f"<td>{paired['lower']:+.3f} to {paired['upper']:+.3f}</td></tr>"
            )
    lines.append("</table>")
    prob = payload["probability"]
    lines.append(
        "<h3>Probability distribution</h3><table><tr><th>Cohort</th>"
        "<th>Jev calls</th><th>Chosen-probability &gt; 0.8 (correct)</th>"
        "<th>Whole-cohort interval</th></tr>"
    )
    for cohort in ("approved", "calibration"):
        above = prob[cohort]
        lines.append(
            f"<tr><td>{escape(cohort)}</td><td>{jev['calls'][cohort]}</td>"
            f"<td>{above['correct']}/{above['count']}</td>"
            f"<td>{metric_text(payload['metrics'][cohort]['jev'])}</td></tr>"
        )
    lines.append("</table>")
    lines.append(
        "<h3>Pre-registered rule</h3>"
        f"<p>{escape(pre['rule'])} Written at "
        f"<code>{escape(pre['written_at_utc'])}</code>, before any validation "
        "call. </p>"
    )
    if payload["validation_read"]:
        lines.append(
            "<p>The bar was met, so validation labels were read once for Jev and "
            "the two controls.</p>"
        )
    else:
        lines.append(
            "<p>The bar was <strong>not</strong> met, so validation labels were "
            "not read. Jev lost against a control on at least one of approved or "
            "calibration.</p>"
        )
        lines.append(
            "<table><tr><th>Name</th><th>Cohort</th><th>Label</th>"
            "<th>Jev</th><th>Set precedence</th><th>IDS-first</th></tr>"
        )
        for item in payload["losses"]:
            lines.append(
                "<tr>"
                + "".join(
                    f"<td>{escape(str(value))}</td>"
                    for value in (
                        item["name"],
                        item["cohort"],
                        item["label"],
                        item["jev"],
                        item["set_precedence"],
                        item["ids_first"],
                    )
                )
                + "</tr>"
            )
        lines.append("</table>")
    caption = (
        "Correct names over total for each arm, with 95% paired-group bootstrap "
        "intervals. Blue is the Jev choice; the controls are drawn in neutral "
        "greys. Validation carries a Jev mark here because the pre-registered "
        "bar was met and its labels were read once."
        if payload["validation_read"]
        else "Correct names over total for each arm, with 95% paired-group "
        "bootstrap intervals. Blue is the Jev choice; the controls are drawn in "
        "neutral greys. Validation carries no Jev mark: the pre-registered bar "
        "was not met, so its labels were not read."
    )
    lines.append(
        f'<figure><img src="/imas-codex/figures/sn-catalog-grouping/{STEM}/'
        f'accuracy.svg" alt="Accuracy per arm per labelled cohort"><figcaption>'
        f"{caption}</figcaption></figure>"
    )
    shuffle = payload["shuffle_check"]
    if shuffle:
        lines.append(
            "<h3>Shuffled-label control</h3><p>Re-scoring the saved Jev answers "
            "with the labels permuted within the calibration cohort (fixed seed "
            f"{shuffle['seed']}) drops correct-of-cohort to "
            f"{shuffle['shuffled_correct']}/{shuffle['count']}, a fall of "
            f"{shuffle['drop']} against the unshuffled figure. The bootstrap "
            f"half-width is {shuffle['half_width']:.3f}, so the drop exceeds it "
            "and the scorer demonstrably reads the labels.</p>"
        )
    spent = payload["spend"]
    lines.append(
        "<h3>Spend</h3><p>"
        f"{spent['calls']} Jev calls, {spent['estimated_tokens']:,} estimated "
        f"input tokens, USD {spent['usd']:.6f} against a ceiling of USD "
        f"{spent['ceiling_usd']:.2f}.</p>"
    )
    lines.append(
        "<h3>Single-option names</h3><p>After deduplication, "
        f"{state['single_option']['count']} labelled names have one candidate "
        "and take it without a call"
        + (
            ": " + escape(", ".join(state["single_option"]["names"]))
            if state["single_option"]["names"]
            else ""
        )
        + ".</p>"
    )
    lines.append(
        "<h3>Per-name answers</h3><table><tr><th>Name</th><th>Cohort</th>"
        "<th>Label</th><th>Jev domain</th><th>P</th><th>Set precedence</th>"
        "<th>IDS-first</th><th>Options</th></tr>"
    )
    for row in payload["per_name"]:
        lines.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in (
                    row["name"],
                    row["cohort"],
                    row["label"],
                    row["jev"] or "-",
                    f"{row['probability']:.2f}"
                    if row["options_count"] > 1
                    else "single",
                    row["set_precedence"] or "-",
                    row["ids_first"] or "-",
                    ", ".join(row["options"]),
                )
            )
            + "</tr>"
        )
    lines.append(
        "</table><p>Every name's options, Jev answer with its full probability "
        "distribution, and both controls' answers are in the adjacent results "
        "JSON. Validation answers appear only when the bar was met.</p></section>"
    )
    FRAGMENT.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--shuffle-check", dest="shuffle", action="store_true")
    args = parser.parse_args()
    if args.shuffle:
        return shuffle_check()
    if args.rescore:
        payload = json.loads(RESULTS.read_text())
        make_figure(payload["metrics"], payload["cohorts_read"])
        render(payload)
        print(f"rescored from {RESULTS}")
        return 0
    sheet = json.loads(base.SHEET.read_text())
    approved = sheet["reference"]
    calibration = [row for row in sheet["sample"] if row["split"] == "calibration"]
    validation = [row for row in sheet["sample"] if row["split"] == "validation"]
    assert (len(approved), len(calibration), len(validation)) == (20, 73, 77)
    ids_map = graph_ids_map()
    rows, query_seconds = base.source_rows()
    memo: dict[str, dict[str, str | None]] = {}
    sources = {name: base.sources_for(name, rows, memo) for name in rows}
    assert "core_plasma_physics" in sources["electron_temperature"].values(), (
        "known-present source-domain control failed"
    )
    labelled = {row["name"] for row in (*approved, *calibration, *validation)}
    missing = [name for name in labelled if name not in rows]
    assert not missing, f"labelled names absent from live census: {missing}"

    options = {name: candidates(sources[name], ids_map) for name in labelled}
    single = sorted(name for name in labelled if len(options[name]) == 1)
    first_stage = approved + calibration
    model = model_id()
    print(
        f"model={model} live={len(rows)} labelled={len(labelled)} "
        f"single_option={len(single)} query_seconds={query_seconds:.3f}"
    )

    if args.probe:
        row = first_stage[0]
        states = {
            row["name"]: build_state(
                row, sources[row["name"]], ids_map, options[row["name"]]
            )
        }
        outcome = asyncio.run(
            score_one(states[row["name"]], build_question(options[row["name"]]), model)
        )
        print(json.dumps(outcome["answer"], indent=2)[:2000])
        print(f"cost_usd={outcome['cost_usd']:.6f}")
        return 0

    state = {
        "live_names": len(rows),
        "query_seconds": round(query_seconds, 3),
        "single_option": {"count": len(single), "names": single},
        "known_present_control": {
            "name": "electron_temperature",
            "has_core_plasma_physics_source": True,
        },
        "ids_map_source": "graph IDS.physics_domain",
        "candidate_definition": (
            "source paths' stored domains; source IDSs' graph domains; the band "
            "umbrella of each band those candidates fall in; general"
        ),
    }

    answers: dict[str, dict] = {}
    spent = 0.0
    calls = 0
    estimated_tokens = 0
    if args.resume and RESULTS.exists():
        prior = json.loads(RESULTS.read_text())
        answers = prior.get("answers") or {}
        spent = prior.get("spend", {}).get("usd", 0.0)
        calls = prior.get("spend", {}).get("calls", 0)
        estimated_tokens = prior.get("spend", {}).get("estimated_tokens", 0)
        print(f"resumed {len(answers)} saved answers, spent={spent:.6f}")

    async def run_stage(cohort_rows: list[dict]) -> tuple[dict, float, int, int]:
        gate = asyncio.Semaphore(CONCURRENCY)

        async def one(row: dict):
            name = row["name"]
            if name in answers:
                return name, None
            if len(options[name]) == 1:
                choice = options[name][0]
                return name, {
                    "answer": {
                        "choice": choice,
                        "confidence": 1.0,
                        "probabilities": {choice: 1.0},
                        "probability_of_choice": 1.0,
                        "single_option": True,
                    },
                    "cost_usd": 0.0,
                }
            async with gate:
                return name, await score_one(
                    build_state(row, sources[name], ids_map, options[name]),
                    build_question(options[name]),
                    model,
                )

        outcomes = await asyncio.gather(*(one(row) for row in cohort_rows))
        stage_spent = 0.0
        stage_calls = 0
        stage_tokens = 0
        for name, outcome in outcomes:
            if outcome is None:
                continue
            answers[name] = outcome["answer"]
            if not outcome["answer"].get("single_option"):
                stage_spent += outcome["cost_usd"]
                stage_calls += 1
                stage_tokens += len(
                    json.dumps(
                        build_state(
                            next(r for r in cohort_rows if r["name"] == name),
                            sources[name],
                            ids_map,
                            options[name],
                        )
                    )
                ) + len(json.dumps(build_question(options[name])))
        return answers, stage_spent, stage_calls, stage_tokens

    def write_results(stage_label, validation_read, cohorts_read):
        metrics = {}
        paired_by_cohort = {}
        per_name = []
        for cohort in cohorts_read:
            cohort_rows = {
                "approved": approved,
                "calibration": calibration,
                "validation": validation,
            }[cohort]
            metrics[cohort] = {
                arm: interval(
                    jev_correct(cohort_rows, answers)
                    if arm == "jev"
                    else control_correct(cohort_rows, sources, ids_map, arm)
                )
                for arm in ARMS
            }
            if cohort in ("approved", "calibration"):
                paired_by_cohort[cohort] = {
                    arm: paired_difference(
                        jev_correct(cohort_rows, answers),
                        control_correct(cohort_rows, sources, ids_map, arm),
                    )
                    for arm in (CONTROL_SET, CONTROL_IDS)
                }
        for row in (*approved, *calibration, *(validation if validation_read else [])):
            name = row["name"]
            answer = answers.get(name) or {}
            per_name.append(
                {
                    "name": name,
                    "cohort": (
                        "approved"
                        if row in approved
                        else ("calibration" if row in calibration else "validation")
                    ),
                    "label": row["label"],
                    "options": options[name],
                    "options_count": len(options[name]),
                    "jev": answer.get("choice"),
                    "probability": answer.get("probability_of_choice"),
                    "probabilities": answer.get("probabilities"),
                    "confidence": answer.get("confidence"),
                    "single_option": bool(answer.get("single_option")),
                    "set_precedence": base.choose_domain(sources[name], CONTROL_SET),
                    "ids_first": design.choose(sources[name], CONTROL_IDS, ids_map),
                    "source_paths": dict(sorted(sources[name].items())),
                    "source_id_domains": {
                        ids: ids_map.get(ids)
                        for ids in sorted({ids_of(p) for p in sources[name]})
                    },
                }
            )
        losses = []
        for cohort, cohort_rows in (
            ("approved", approved),
            ("calibration", calibration),
        ):
            for row in cohort_rows:
                name = row["name"]
                jev = (answers.get(name) or {}).get("choice")
                if jev == row["label"]:
                    continue
                sp = base.choose_domain(sources[name], CONTROL_SET)
                ids = design.choose(sources[name], CONTROL_IDS, ids_map)
                if row["label"] in (sp, ids):
                    losses.append(
                        {
                            "name": name,
                            "cohort": cohort,
                            "label": row["label"],
                            "jev": jev,
                            "set_precedence": sp,
                            "ids_first": ids,
                        }
                    )
        prob = {
            cohort: accuracy_above_probability(
                {"approved": approved, "calibration": calibration}[cohort],
                answers,
                0.8,
            )
            for cohort in ("approved", "calibration")
        }
        calls_by_cohort = {
            cohort: sum(
                1
                for row in {"approved": approved, "calibration": calibration}[cohort]
                if not (answers.get(row["name"]) or {}).get("single_option")
            )
            for cohort in ("approved", "calibration")
        }
        payload = {
            "model": PAYLOAD_MODEL,
            "seat": SEAT,
            "read_only_graph": True,
            "graph_writes": 0,
            "stage": stage_label,
            "measured_at_utc": datetime.now(UTC).isoformat(),
            "state": state,
            "cohorts_read": list(cohorts_read),
            "validation_read": validation_read,
            "pre_registration": pre_registration,
            "metrics": metrics,
            "paired_by_cohort": paired_by_cohort,
            "probability": prob,
            "jev": {"calls": calls_by_cohort},
            "losses": losses,
            "per_name": per_name,
            "answers": answers,
            "shuffle_check": SHUFFLE_HOLDER.get("result"),
            "spend": {
                "calls": calls,
                "usd": round(spent, 6),
                "estimated_tokens": estimated_tokens,
                "ceiling_usd": SPEND_CEILING_USD,
            },
        }
        RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    pre_registration = {
        "rule": RULE,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "selection_set": "approved and calibration",
    }
    PAYLOAD_MODEL = model
    SHUFFLE_HOLDER: dict = {}

    answers, s_spent, s_calls, s_tokens = asyncio.run(run_stage(first_stage))
    spent += s_spent
    calls += s_calls
    estimated_tokens += s_tokens
    print(f"approved+calibration: spent={spent:.6f} calls={calls}")
    if spent > SPEND_CEILING_USD:
        print(f"SPEND CEILING EXCEEDED: {spent:.6f}", file=sys.stderr)
        return 2

    payload = write_results("approved+calibration", False, ("approved", "calibration"))
    approved_ok = payload["metrics"]["approved"]["jev"]["correct"] > max(
        payload["metrics"]["approved"][arm]["correct"]
        for arm in (CONTROL_SET, CONTROL_IDS)
    )
    calibration_ok = payload["metrics"]["calibration"]["jev"]["correct"] > max(
        payload["metrics"]["calibration"][arm]["correct"]
        for arm in (CONTROL_SET, CONTROL_IDS)
    )
    bar_met = approved_ok and calibration_ok
    print(
        f"bar_met={bar_met} approved_ok={approved_ok} calibration_ok={calibration_ok}"
    )

    if bar_met:
        answers, s_spent, s_calls, s_tokens = asyncio.run(run_stage(validation))
        spent += s_spent
        calls += s_calls
        estimated_tokens += s_tokens
        if spent > SPEND_CEILING_USD:
            print(f"SPEND CEILING EXCEEDED: {spent:.6f}", file=sys.stderr)
            return 2
        payload = write_results(
            "approved+calibration+validation",
            True,
            ("approved", "calibration", "validation"),
        )

    make_figure(payload["metrics"], payload["cohorts_read"])
    render(payload)

    print(
        "metrics "
        + json.dumps(
            {
                cohort: {
                    arm: f"{payload['metrics'][cohort][arm]['correct']}"
                    f"/{payload['metrics'][cohort][arm]['count']}"
                    for arm in ARMS
                }
                for cohort in payload["cohorts_read"]
            }
        )
    )
    print(f"calls={calls} estimated_tokens={estimated_tokens} spend_usd={spent:.6f}")
    print(f"wrote {RESULTS}, {FRAGMENT}")
    return 0


def shuffle_check() -> int:
    payload = json.loads(RESULTS.read_text())
    answers = payload.get("answers") or {}
    # Rebuild the labelled cohorts from the per-name record.
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for row in payload["per_name"]:
        by_cohort[row["cohort"]].append(row)
    cohort = by_cohort["calibration"]
    unshuffled = sum(
        (answers.get(row["name"]) or {}).get("choice") == row["label"] for row in cohort
    )
    shuffled_rows = shuffled(cohort, SHUFFLE_SEED)
    shuffled_correct = sum(
        (answers.get(row["name"]) or {}).get("choice") == row["label"]
        for row in shuffled_rows
    )
    item = payload["metrics"]["calibration"]["jev"]
    half_width = (item["upper"] - item["lower"]) / 2
    drop = unshuffled - shuffled_correct
    print(
        f"calibration unshuffled={unshuffled}/{len(cohort)} "
        f"shuffled={shuffled_correct}/{len(cohort)} drop={drop} "
        f"half_width={half_width:.3f}"
    )
    ok = drop > half_width
    payload["shuffle_check"] = {
        "seed": SHUFFLE_SEED,
        "cohort": "calibration",
        "count": len(cohort),
        "unshuffled_correct": unshuffled,
        "shuffled_correct": shuffled_correct,
        "drop": drop,
        "half_width": half_width,
        "passed": ok,
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
    render(payload)
    print(f"shuffle_check={'pass' if ok else 'FAIL'}")
    print(f"EXIT={0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

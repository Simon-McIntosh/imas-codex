"""Measure documentation judgments from frozen evidence and Jev receipts."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import html
import json
import os
import time
import zlib
from collections import Counter
from pathlib import Path

from imas_codex.discovery.base.evaluation import (
    bootstrap_interval,
    choose_operating_point,
    cost_latency_summary,
    frozen_state_variation,
    option_order_stability,
    paired_auc,
)
from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_model
from imas_codex.standard_names.review.checks import load_review_checks
from imas_codex.standard_names.review.jev_review import (
    DOCS_AXIS,
    SERVICE,
    aggregate_docs_review,
    build_docs_state,
    docs_question_table,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GOLDEN = HERE.parent / "sn-judgment-evaluation" / "golden-documentation-acceptable.json"
PLANTED = HERE / "golden-documentation-planted.json"
RESULTS = HERE / "srtj-docs-measure-results.json"
FRAGMENT = HERE / "srtj-docs-measure.html"
FIGURE = ROOT / "docs/figures/sn-review-through-jev/srtj-docs-measure/paired-auc.svg"
RUN = Path(os.environ["RECKON_MEASURE_RUN"])
RECEIPTS = RUN / "docs-decisions.jsonl"
CONTEXT = RUN / "docs-incumbent.json"
CALIBRATION = RUN / "docs-calibration.json"
MODEL = get_model("sn-judge")
CEILING = 1.50
CLAIM_PREFIX = "check:claim_contradicts_source:"
DIMENSIONS = ("description_quality", "documentation_quality", "completeness")


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _fixture() -> list[dict]:
    rows = json.loads(GOLDEN.read_text())
    if len(rows) != 400 or Counter(row["split"] for row in rows) != {
        "calibration": 216,
        "validation": 184,
    }:
        raise RuntimeError("golden composition differs from the recorded draw")
    if any(row["label_source"] != "coordinator" for row in rows):
        raise RuntimeError("incumbent-authored golden label is in the comparison")
    return rows


def _receipts() -> dict[str, dict]:
    return (
        {
            row["key"]: row
            for line in RECEIPTS.read_text().splitlines()
            if (row := json.loads(line))
        }
        if RECEIPTS.exists()
        else {}
    )


def _append(row: dict) -> None:
    with RECEIPTS.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _graph_context() -> dict:
    if CONTEXT.exists():
        return json.loads(CONTEXT.read_text())
    rows = _fixture()
    names = [row["unit"] for row in rows]
    graph = GraphClient()
    found = {}
    query = """
    UNWIND $names AS name
    OPTIONAL MATCH (sn:StandardName {id:name})
    RETURN name, sn.documentation AS documentation,
      sn.reviewer_score_docs AS incumbent, sn.description AS description
    LIMIT 80
    """
    for start in range(0, len(names), 80):
        for result in graph.query(query, names=names[start : start + 80]):
            found[result["name"]] = result
    if len(found) != len(names):
        raise RuntimeError("bounded graph read missed golden names")
    matched = {
        row["unit"]: {
            "score": found[row["unit"]]["incumbent"],
            "document_match": hashlib.sha256(
                (found[row["unit"]]["documentation"] or "").encode()
            ).hexdigest()
            == row["source_evidence"],
        }
        for row in rows
    }
    if not any(value["score"] is not None for value in matched.values()):
        raise RuntimeError("no incumbent scores, including the known scored sample")
    if not all(value["document_match"] for value in matched.values()):
        raise RuntimeError(
            "a graph score belongs to documentation other than the golden text"
        )
    context = {
        "by_name": matched,
        "score_coverage": sum(value["score"] is not None for value in matched.values()),
        "document_matches": sum(value["document_match"] for value in matched.values()),
        "known_present": matched["electron_energy"],
    }
    _write(CONTEXT, context)
    print(
        json.dumps({key: value for key, value in context.items() if key != "by_name"})
    )
    return context


def _item(row: dict) -> dict:
    return {
        "id": row["unit"],
        "description": row["description"],
        "documentation": row["documentation"],
        "conventions": row["conventions"],
        "dd_source_docs": [
            {"path": source["path"], "value": source["documentation"]}
            for source in row["dd_sources"]
        ],
    }


def _planted_item(row: dict, source: dict) -> dict:
    return {
        **_item(source),
        "documentation": row["documentation"],
        "dd_source_docs": [
            {
                "path": row["dd_source"]["path"],
                "value": row["dd_source"]["documentation"],
            }
        ],
    }


def _checks():
    return [check for check in load_review_checks()[0] if check.axis == DOCS_AXIS]


def _make_entry(key: str, item: dict, checks, questions: dict | None = None) -> dict:
    state = build_docs_state(item, checks=checks)
    return {
        "key": key,
        "state": state,
        "questions": questions
        or docs_question_table(checks, state["documentation_claims"]),
    }


async def _judge(entries: list[dict], concurrency: int) -> None:
    existing = _receipts()
    pending = [entry for entry in entries if entry["key"] not in existing]
    if not pending:
        return
    booked = sum(row["cost_usd"] for row in existing.values())
    if booked >= CEILING:
        raise RuntimeError("Jev spend ceiling reached")
    if len(existing) >= 10:
        mean = booked / len(existing)
        projected = booked + mean * len(pending)
        print(f"booked USD {booked:.6f}; projected USD {projected:.6f}", flush=True)
        if projected > CEILING:
            raise RuntimeError("Jev projection exceeds USD 1.50")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(entry):
        async with semaphore:
            start = time.monotonic()
            result, cost = await decide_batch(
                [entry["state"]],
                entry["questions"],
                model=MODEL,
                service=SERVICE,
                concurrency=1,
            )
            return {
                "key": entry["key"],
                "state": entry["state"],
                "answers": result[0][0] if result[0] else None,
                "cost_usd": cost,
                "latency_s": time.monotonic() - start,
                "question_count": len(entry["questions"]),
            }

    tasks = [asyncio.create_task(one(entry)) for entry in pending]
    for index, task in enumerate(asyncio.as_completed(tasks), 1):
        row = await task
        _append(row)
        booked += row["cost_usd"]
        print(
            f"{index}/{len(pending)} {row['key']} answered={row['answers'] is not None} booked_usd={booked:.6f}",
            flush=True,
        )
        if booked >= CEILING:
            for other in tasks:
                other.cancel()
            raise RuntimeError("Jev spend ceiling reached during run")


def _base_entries(split: str, limit: int = 0) -> list[dict]:
    checks = _checks()
    rows = [row for row in _fixture() if row["split"] == split]
    if limit:
        rows = rows[:limit]
    return [_make_entry("base:" + row["unit"], _item(row), checks) for row in rows]


def _planted_entries() -> list[dict]:
    checks = _checks()
    sources = {row["unit"]: row for row in _fixture()}
    copies = json.loads(PLANTED.read_text())["copies"]
    if Counter(row["claim_label"] for row in copies) != {
        "contradicts": 60,
        "says_nothing": 20,
    }:
        raise RuntimeError("planted cohort differs from the recorded draw")
    return [
        _make_entry(
            "plant:" + row["id"],
            _planted_item(row, sources[row["source_unit"]]),
            checks,
        )
        for row in copies
    ]


def _stability_entries() -> list[dict]:
    checks = _checks()
    receipts = _receipts()
    rows = [row for row in _fixture() if row["split"] == "calibration"]
    selected = [
        row
        for row in rows
        if row["dd_sources"] and receipts.get("base:" + row["unit"], {}).get("answers")
    ][:20]
    if len(selected) != 20:
        raise RuntimeError("fewer than 20 answered calibration states with DD evidence")
    entries = []
    for row in selected:
        base = receipts["base:" + row["unit"]]
        state = {
            **base["state"],
            "documentation_claims": base["state"]["documentation_claims"][:1],
        }
        key = CLAIM_PREFIX + "0"
        question = docs_question_table(checks, state["documentation_claims"])[key]
        for mode in ("repeat", "reverse"):
            if mode == "reverse":
                question = {
                    **question,
                    "criteria": dict(reversed(list(question["criteria"].items()))),
                }
            entries.append(
                {
                    "key": f"{mode}:{row['unit']}",
                    "state": state,
                    "questions": {key: question},
                }
            )
    return entries


def _score(answers: dict | None) -> float | None:
    if not answers:
        return None
    dimension_values = []
    for dimension in DIMENSIONS:
        answer = answers.get("dimension:" + dimension)
        if not answer:
            return None
        accepted = next(
            iter(
                docs_question_table(_checks(), [])["dimension:" + dimension]["criteria"]
            )
        )
        dimension_values.append(float(answer.get("probabilities", {}).get(accepted, 0)))
    claims = [answer for key, answer in answers.items() if key.startswith(CLAIM_PREFIX)]
    if not claims:
        return None
    if any(answer["choice"] == "contradicts" for answer in claims):
        return 0.0
    claim_value = sum(
        answer.get("probabilities", {}).get("supports", 0)
        + 0.5 * answer.get("probabilities", {}).get("says nothing", 0)
        for answer in claims
    ) / len(claims)
    checks = []
    for key, answer in answers.items():
        if not key.startswith("check:") or key.startswith(CLAIM_PREFIX):
            continue
        if answer["type"] == "noul":
            checks.append(1 - float(answer["noul"]))
        else:
            first = next(iter(answer["probabilities"]))
            checks.append(float(answer["probabilities"][first]))
    return (
        0.5 * sum(dimension_values) / len(dimension_values)
        + 0.3 * (sum(checks) / len(checks) if checks else 0)
        + 0.2 * claim_value
    )


def _scored(rows: list[dict], receipts: dict) -> list[dict]:
    scored = []
    for row in rows:
        answer = receipts.get("base:" + row["unit"], {}).get("answers")
        score = _score(answer)
        if score is not None:
            scored.append(
                {
                    "unit": row["unit"],
                    "label": row["label"],
                    "split": row["split"],
                    "copy_group": row["copy_group"],
                    "score": score,
                }
            )
    return scored


def _split_metrics(rows: list[dict], context: dict) -> dict:
    y = [row["label"] for row in rows]
    a = [row["score"] for row in rows]
    groups = [row["copy_group"] for row in rows]
    result = {"n": len(rows), "jev": bootstrap_interval(y, a, copy_groups=groups)}
    paired = [
        row for row in rows if context["by_name"][row["unit"]]["score"] is not None
    ]
    if len({row["label"] for row in paired}) == 2:
        y = [row["label"] for row in paired]
        a = [row["score"] for row in paired]
        b = [context["by_name"][row["unit"]]["score"] for row in paired]
        groups = [row["copy_group"] for row in paired]
        result["incumbent"] = {
            "n": len(paired),
            "paired": paired_auc(y, a, b),
            "jev_interval": bootstrap_interval(y, a, copy_groups=groups),
            "incumbent_interval": bootstrap_interval(y, b, copy_groups=groups),
            "difference_interval": bootstrap_interval(
                y, a, scores_b=b, copy_groups=groups
            ),
        }
    return result


def _calibrate() -> dict:
    rows = _scored(
        [row for row in _fixture() if row["split"] == "calibration"], _receipts()
    )
    if len(rows) != 216:
        raise RuntimeError(f"calibration has {len(rows)} of 216 complete answers")
    threshold = choose_operating_point(
        rows, cost_false_positive=1, cost_false_negative=1
    )["threshold"]
    result = {
        "score_rule": "fixed 0.5 dimension + 0.3 named-check + 0.2 claim probability; any chosen contradiction scores zero",
        "threshold": threshold,
        "rows": len(rows),
        "metrics": _split_metrics(rows, _graph_context()),
    }
    _write(CALIBRATION, result)
    print(json.dumps(result), flush=True)
    return result


def _planted_metrics(receipts: dict, threshold: float) -> dict:
    checks = _checks()
    copies = json.loads(PLANTED.read_text())["copies"]
    table = {expected: Counter() for expected in ("contradicts", "says_nothing")}
    veto = Counter()
    scored = Counter()
    for row in copies:
        receipt = receipts.get("plant:" + row["id"])
        if not receipt or not receipt["answers"]:
            raise RuntimeError("planted answer missing")
        claims = receipt["state"]["documentation_claims"]
        matches = [
            index
            for index, claim in enumerate(claims)
            if row["planted_sentence"] in claim
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"planted claim located {len(matches)} times for {row['id']}"
            )
        answer = receipt["answers"][CLAIM_PREFIX + str(matches[0])]
        table[row["claim_label"]][answer["choice"]] += 1
        decision = aggregate_docs_review(receipt["state"], checks, receipt["answers"])
        veto[row["claim_label"] + ":" + str(decision["contradiction_veto"])] += 1
        score = _score(receipt["answers"])
        scored[
            row["claim_label"] + ":" + str(score is not None and score >= threshold)
        ] += 1
    return {
        "confusion": {key: dict(value) for key, value in table.items()},
        "veto": dict(veto),
        "threshold_accept": dict(scored),
    }


def _stability_metrics(receipts: dict) -> dict:
    original = []
    reversed_order = []
    repeated = []
    names = [
        key.removeprefix("repeat:") for key in receipts if key.startswith("repeat:")
    ]
    for name in names:
        base = receipts["base:" + name]
        repeat = receipts["repeat:" + name]
        reverse = receipts["reverse:" + name]
        if not repeat["answers"] or not reverse["answers"]:
            raise RuntimeError("stability answer missing")
        key = CLAIM_PREFIX + "0"
        first = base["answers"][key]["choice"]
        again = repeat["answers"][key]["choice"]
        flipped = reverse["answers"][key]["choice"]
        original.append(first)
        reversed_order.append(flipped)
        digest = hashlib.sha256(
            json.dumps(repeat["state"], sort_keys=True).encode()
        ).hexdigest()
        repeated.extend(
            [
                {"digest": digest, "fields": {key: first}},
                {"digest": digest, "fields": {key: again}},
            ]
        )
    if len(names) != 20:
        raise RuntimeError("stability sample is incomplete")
    return {
        "claims": len(names),
        "option_order": option_order_stability(original, reversed_order),
        "frozen_state": frozen_state_variation(repeated),
    }


def _summarize(receipts: dict, context: dict, calibration: dict) -> dict:
    rows = _fixture()
    scored = _scored(rows, receipts)
    if len(scored) != 400:
        raise RuntimeError(f"only {len(scored)} of 400 base entries answered")
    if not all(
        "plant:" + copy["id"] in receipts
        for copy in json.loads(PLANTED.read_text())["copies"]
    ):
        raise RuntimeError("planted cohort incomplete")
    validation = [row for row in scored if row["split"] == "validation"]
    base_calls = [
        {"cost": receipt["cost_usd"], "latency_s": receipt["latency_s"]}
        for key, receipt in receipts.items()
        if key.startswith("base:")
    ]
    base_claims = sum(
        len(receipt["state"]["documentation_claims"])
        for key, receipt in receipts.items()
        if key.startswith("base:")
    )
    spend = sum(receipt["cost_usd"] for receipt in receipts.values())
    if spend > CEILING:
        raise RuntimeError("measured spend exceeds ceiling")
    result = {
        "schema": "documentation-review-measure",
        "inputs": {
            "golden_sha256": hashlib.sha256(GOLDEN.read_bytes()).hexdigest(),
            "planted_sha256": hashlib.sha256(PLANTED.read_bytes()).hexdigest(),
            "model": MODEL,
            "incumbent_graph_document_matches": context["document_matches"],
            "incumbent_score_coverage": context["score_coverage"],
        },
        "method": {
            "score": calibration["score_rule"],
            "threshold": calibration["threshold"],
            "fit_split": "calibration",
            "validation_reads": 1,
            "bootstrap": "paired resampling by copy_group, 1000 draws",
        },
        "splits": {
            "calibration": calibration["metrics"],
            "validation": _split_metrics(validation, context),
        },
        "planted": _planted_metrics(receipts, calibration["threshold"]),
        "stability": _stability_metrics(receipts),
        "cost": {
            "spend_usd": spend,
            "ceiling_usd": CEILING,
            "base": cost_latency_summary(base_calls),
            "base_claims": base_claims,
            "usd_per_claim": sum(call["cost"] for call in base_calls) / base_claims,
            "seconds_per_claim": sum(call["latency_s"] for call in base_calls)
            / base_claims,
        },
    }
    comparator = result["splits"]["validation"].get("incumbent")
    difference = comparator["difference_interval"]["lower"] if comparator else None
    order = result["stability"]["option_order"]["flip_rate"]
    frozen = result["stability"]["frozen_state"]["fields"]
    changed = sum(field["states_changed"] for field in frozen.values())
    observed = sum(field["states"] for field in frozen.values())
    variation = changed / observed if observed else None
    result["stability"]["frozen_change_rate"] = variation
    result["verdict"] = {
        "rule": "paired validation lower bound > -0.02; threshold fitted on calibration; option-order and frozen-state change each <= 0.05; cost below incumbent",
        "paired_lower_bound": difference,
        "auc_pass": difference is not None and difference > -0.02,
        "stability_pass": order <= 0.05 and variation is not None and variation <= 0.05,
        "cost_pass": None,
        "status": "shadow",
        "reason": "Incumbent documentation-review cost not included in graph mirror; replacement requires a verified cost comparator as well as all other gates.",
    }
    return result


def _freeze(receipts: dict) -> str:
    raw = json.dumps(receipts, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.b64encode(zlib.compress(raw, level=9)).decode()


def _thaw(value: str) -> dict:
    return json.loads(zlib.decompress(base64.b64decode(value)))


def _figure(result: dict) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "documentation-review-auc"
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for index, split in enumerate(("calibration", "validation")):
        pair = result["splits"][split]["incumbent"]
        for shift, key, colour, label in (
            (0.13, "incumbent_interval", "#747474", "incumbent"),
            (-0.13, "jev_interval", "#246c78", "Jev"),
        ):
            interval = pair[key]
            ax.errorbar(
                interval["point"],
                index + shift,
                xerr=[
                    [interval["point"] - interval["lower"]],
                    [interval["upper"] - interval["point"]],
                ],
                fmt="o",
                markersize=8,
                linewidth=3 if label == "Jev" else 2.6,
                color=colour,
                label=label if index == 0 else None,
                zorder=3 if label == "Jev" else 2,
            )
    ax.set_yticks([0, 1], ["calibration", "validation"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC [95% bootstrap interval]")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)


def _fragment(result: dict) -> None:
    splits = result["splits"]

    def fmt(interval):
        return f"{interval['point']:.3f} [{interval['lower']:.3f}, {interval['upper']:.3f}]"

    rows = "".join(
        f"<tr><th>{split}</th><td>{splits[split]['incumbent']['n']}</td><td>{fmt(splits[split]['incumbent']['jev_interval'])}</td><td>{fmt(splits[split]['incumbent']['incumbent_interval'])}</td><td>{fmt(splits[split]['incumbent']['difference_interval'])}</td></tr>"
        for split in ("calibration", "validation")
    )
    confusion = result["planted"]["confusion"]
    claim_rows = "".join(
        f"<tr><th>{expected}</th>"
        + "".join(
            f"<td>{confusion[expected].get(answer, 0)}</td>"
            for answer in ("supports", "contradicts", "says nothing")
        )
        + "</tr>"
        for expected in ("contradicts", "says_nothing")
    )
    veto = result["planted"]["veto"]
    cost = result["cost"]
    stability = result["stability"]
    verdict = result["verdict"]
    text = f"""<section id="srtj-docs-measure" data-reckon="evidence">
<h2 id="srtj-docs-measure-heading">Documentation review measured on golden B</h2>
<p id="srtj-docs-measure-method">400 coordinator-labelled entries, with 216 calibration and 184 validation entries. The graph documentation hash matched all {result["inputs"]["incumbent_graph_document_matches"]} golden texts; {result["inputs"]["incumbent_score_coverage"]} carried an incumbent documentation score. The fixed score combines dimension, named-check and claim probabilities, with a hard contradiction veto. Its operating threshold {result["method"]["threshold"]:.4f} was chosen on calibration before the validation split was read once. Bootstrap resamples copy groups together.</p>
<figure id="srtj-docs-measure-figure"><img src="/imas-codex/figures/sn-review-through-jev/srtj-docs-measure/paired-auc.svg" alt="Paired Jev and incumbent AUC estimates with 95% intervals for calibration and validation."><figcaption>Each split compares the same entries with both scores available. Intervals use 1,000 paired bootstrap draws by copy group.</figcaption></figure>
<table id="srtj-docs-measure-auc"><thead><tr><th>split</th><th>paired entries</th><th>Jev AUC [95%]</th><th>incumbent AUC [95%]</th><th>Jev − incumbent [95%]</th></tr></thead><tbody>{rows}</tbody></table>
<table id="srtj-docs-measure-claims"><caption>Planted claim against Jev's selected relation</caption><thead><tr><th>planted label</th><th>supports</th><th>contradicts</th><th>says nothing</th></tr></thead><tbody>{claim_rows}</tbody></table>
<p id="srtj-docs-measure-veto">The contradiction veto fired on {veto.get("contradicts:True", 0)} of 60 contradiction copies and missed {veto.get("contradicts:False", 0)}. It fired on {veto.get("says_nothing:True", 0)} of 20 silent controls. A contradiction copy that passes the veto is counted as a miss, even if another dimension rejects it.</p>
<p id="srtj-docs-measure-stability">On {stability["claims"]} fixed claims, reversing the Choice criteria changed {stability["option_order"]["flips"]} of {stability["option_order"]["n"]} answers ({stability["option_order"]["flip_rate"]:.1%}). Repeating a frozen state changed {stability["frozen_change_rate"]:.1%} of claim answers.</p>
<p id="srtj-docs-measure-cost">Jev spent USD {cost["spend_usd"]:.4f} of the USD {cost["ceiling_usd"]:.2f} ceiling. Base judgments averaged USD {cost["base"]["mean_cost"]:.6f} and {cost["base"]["mean_latency_s"]:.3f} s per entry; across {cost["base_claims"]} claims, USD {cost["usd_per_claim"]:.6f} and {cost["seconds_per_claim"]:.3f} s per claim.</p>
<p id="srtj-docs-measure-verdict"><strong>Deployment verdict: {html.escape(verdict["status"])}.</strong> Paired validation lower bound {verdict["paired_lower_bound"]:.3f} versus the required −0.020; stability {"passes" if verdict["stability_pass"] else "fails"} its 5% limits. {html.escape(verdict["reason"])}</p>
</section>"""
    FRAGMENT.write_text(text)


def _finalize() -> None:
    if not CALIBRATION.exists():
        raise RuntimeError("calibration must be fixed before validation")
    receipts = _receipts()
    calibration = json.loads(CALIBRATION.read_text())
    context = _graph_context()
    result = _summarize(receipts, context, calibration)
    result["frozen_zlib_b64"] = _freeze(receipts)
    if len(json.dumps(result).encode()) > 300_000:
        raise RuntimeError("results JSON exceeds repository data-file size limit")
    _write(RESULTS, result)
    _figure(result)
    _fragment(result)
    print(
        json.dumps(
            {
                "splits": result["splits"],
                "planted": result["planted"],
                "stability": result["stability"],
                "cost": result["cost"],
                "verdict": result["verdict"],
            }
        ),
        flush=True,
    )


def _rescore() -> None:
    saved = json.loads(RESULTS.read_text())
    receipts = _thaw(saved["frozen_zlib_b64"])
    calibration = json.loads(CALIBRATION.read_text())
    context = json.loads(CONTEXT.read_text())
    rebuilt = _summarize(receipts, context, calibration)
    for key, value in rebuilt.items():
        if saved[key] != value:
            raise RuntimeError(f"rescore changed {key}")
    _figure(saved)
    _fragment(saved)
    print("rescore reproduced every stored figure", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "probe",
            "calibration",
            "fit",
            "validation",
            "planted",
            "stability",
            "finalize",
            "rescore",
        ),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "probe":
        asyncio.run(_judge(_base_entries("calibration", 10), args.concurrency))
        receipts = _receipts()
        sample = [receipts[entry["key"]] for entry in _base_entries("calibration", 10)]
        mean = sum(row["cost_usd"] for row in sample) / 10
        print(
            f"probe 10 entries USD {sum(row['cost_usd'] for row in sample):.6f}; projected 520 base/planted/stability calls USD {mean * 520:.6f}"
        )
    elif args.mode == "calibration":
        asyncio.run(_judge(_base_entries("calibration"), args.concurrency))
    elif args.mode == "fit":
        _calibrate()
    elif args.mode == "validation":
        if not CALIBRATION.exists():
            raise RuntimeError("validation is sealed until calibration is fixed")
        if any(
            key.startswith("base:") and row["split"] == "validation"
            for row in _fixture()
            for key in _receipts()
            if key == "base:" + row["unit"]
        ):
            raise RuntimeError("validation has already been read")
        asyncio.run(_judge(_base_entries("validation"), args.concurrency))
    elif args.mode == "planted":
        asyncio.run(_judge(_planted_entries(), args.concurrency))
    elif args.mode == "stability":
        asyncio.run(_judge(_stability_entries(), args.concurrency))
    elif args.mode == "finalize":
        _finalize()
    else:
        _rescore()


if __name__ == "__main__":
    main()

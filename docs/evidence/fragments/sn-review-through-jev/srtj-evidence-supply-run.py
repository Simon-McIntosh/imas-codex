"""Measure source-backed name review against the frozen golden labels."""

from __future__ import annotations

import argparse
import asyncio
import base64
import html
import importlib.util
import json
import os
import zlib
from pathlib import Path

from imas_codex.discovery.base.evaluation import bootstrap_interval
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.audits import repeated_token_check
from imas_codex.standard_names.review.checks import (
    JevQuestion,
    JevQuestionKind,
    ReviewCheck,
    load_review_checks,
)
from imas_codex.standard_names.review.jev_review import _owner_segment
from imas_codex.standard_names.workers import _DD_CONTEXT_QUERY

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "srtj-evidence-supply-results.json"
FRAGMENT = HERE / "srtj-evidence-supply.html"
RUN = Path(
    os.environ.get(
        "RECKON_MEASURE_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261016T101231058904-srtj-evidence-supply",
    )
)
CONTEXT = RUN / "evidence-context.json"
RECEIPTS = RUN / "evidence-decisions.jsonl"
PRIOR = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261015T120934498631-srtj-noul-polarity/polarity-run"
)
CEILING_USD = 0.50
CHECK_IDS = (
    "advisory_alias_collapsed",
    "minority_spelling_not_in_roster",
    "over_qualified_modifier",
    "owner_erased_outline",
    "redundant_repeated_segment",
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure():
    os.environ.setdefault("RECKON_MEASURE_RUN", str(RUN))
    return _module(HERE / "srtj-name-review-measure-run.py", "_name_measure")


def _polarity():
    os.environ.setdefault("RECKON_MEASURE_RUN", str(RUN))
    return _module(HERE / "srtj-noul-polarity-run.py", "_name_polarity")


def _diagnosis():
    return _module(HERE / "srtj-name-signal-diagnosis-run.py", "_name_diagnosis")


def _read_receipts(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("mode") == "base" and record.get("answered"):
            records[record["name"]] = record
    return records


def _prepare() -> None:
    """Read DD context through the compose reader's indexed IMASNode query."""
    context = json.loads((PRIOR / "name-context.json").read_text())
    entries = context["entries"]
    paths = sorted(
        {
            path.removeprefix("dd:")
            for entry in entries.values()
            for path in entry.get("source_paths") or []
            if path.startswith("dd:")
        }
    )
    if not paths:
        raise RuntimeError("the source-path reader received no DD paths")
    graph = GraphClient()
    source = {}
    for path in paths:
        rows = graph.query(_DD_CONTEXT_QUERY, path=path)
        if rows:
            source[path] = rows[0]
    if not source:
        raise RuntimeError("the IMASNode reader missed every known source path")
    known = entries["poloidal_magnetic_flux"]
    if not known.get("cocos_label"):
        raise RuntimeError("positive COCOS control is absent in the prior context")
    known_paths = [path.removeprefix("dd:") for path in known.get("source_paths") or []]
    if not any(source.get(path, {}).get("cocos_label") for path in known_paths):
        raise RuntimeError("the DD reader missed known poloidal flux COCOS labels")
    for entry in entries.values():
        own_paths = [
            path.removeprefix("dd:")
            for path in entry.get("source_paths") or []
            if path.startswith("dd:")
        ]
        rows = [source[path] for path in own_paths if path in source]
        entry["cocos_label"] = next(
            (row["cocos_label"] for row in rows if row.get("cocos_label")),
            entry.get("cocos_label"),
        )
        entry["parent_path"] = next(
            (row["parent_path"] for row in rows if row.get("parent_path")), None
        )
        entry["owner_segment"] = _owner_segment(entry)
    CONTEXT.write_text(json.dumps(context, separators=(",", ":"), default=str))
    print(
        json.dumps(
            {
                "source_paths": len(paths),
                "reader_hits": len(source),
                "cocos_names": sum(
                    bool(e.get("cocos_label")) for e in entries.values()
                ),
                "owner_names": sum(
                    bool(e.get("owner_segment")) for e in entries.values()
                ),
            }
        ),
        flush=True,
    )


async def _collect(concurrency: int) -> None:
    """Write every per-name receipt before proceeding to the next batch."""
    polarity = _polarity()
    context = json.loads(CONTEXT.read_text())
    fixture = _measure().load_labelled_fixture(_measure().FIXTURE)
    names = sorted({row["canonical_unit"] for row in fixture})
    checks = [check for check in load_review_checks()[0] if check.axis == "name"]
    questions = polarity.name_question_table(checks)
    existing = _read_receipts(RECEIPTS) if RECEIPTS.exists() else {}
    booked = sum(row["cost_usd"] for row in existing.values())
    pending = [
        name
        for name in names
        if name not in existing and context["entries"][name].get("description")
    ]
    if booked >= CEILING_USD:
        raise RuntimeError(f"Jev spend {booked:.6f} reached USD {CEILING_USD:.2f}")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(name):
        async with semaphore:
            return await polarity._one(name, "base", context, checks, questions, {})

    tasks = [asyncio.create_task(one(name)) for name in pending]
    try:
        with RECEIPTS.open("a") as stream:
            for completed, task in enumerate(asyncio.as_completed(tasks), 1):
                receipt = await task
                stream.write(
                    json.dumps(receipt, separators=(",", ":"), default=str) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
                booked += receipt["cost_usd"]
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"answered {completed}/{len(pending)} Jev USD {booked:.6f}",
                        flush=True,
                    )
                if booked >= CEILING_USD:
                    raise RuntimeError("Jev spend reached the USD 0.50 ceiling")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


def _old_checks(current: list[ReviewCheck]) -> list[ReviewCheck]:
    repeated = ReviewCheck(
        id="redundant_repeated_segment",
        axis="name",
        dimension="convention",
        evidence_fields=["physical_base", "subject", "component"],
        retrieval=False,
        question=JevQuestion(
            kind=JevQuestionKind.NOUL,
            true_case="defect",
            text="Does the name repeat a redundant segment (e.g. electron_electron)?",
        ),
    )
    return [repeated if check.id == repeated.id else check for check in current]


def _auc(rows: list[dict], values: dict[str, float]) -> dict:
    paired = [row for row in rows if row["name"] in values]
    y = [row["label"] for row in paired]
    x = [values[row["name"]] for row in paired]
    groups = [row["copy_group"] for row in paired]
    return {
        "n": len(paired),
        "auc": bootstrap_interval(y, x, copy_groups=groups),
    }


def _score_and_report() -> dict:
    """Lock calibration first; read validation only after fitting the logistic."""
    measure = _measure()
    diagnosis = _diagnosis()
    context = json.loads(CONTEXT.read_text())
    current = _read_receipts(RECEIPTS)
    old = _read_receipts(PRIOR / "name-decisions.jsonl")
    checks = [check for check in load_review_checks()[0] if check.axis == "name"]
    old_checks = _old_checks(checks)
    canonical = {}
    for row in measure.load_labelled_fixture(measure.FIXTURE):
        canonical.setdefault(row["canonical_unit"], row)
    rows = [
        {
            "name": name,
            "label": int(row["label"]),
            "split": row["split"],
            "copy_group": row["copy_group"],
        }
        for name, row in canonical.items()
        if name in old and name in current
    ]
    calibration = [row for row in rows if row["split"] == "calibration"]
    validation = [row for row in rows if row["split"] == "validation"]
    if len(calibration) < 300:
        raise RuntimeError("calibration is missing too many matched answers")
    per_check = {}
    for check_id in CHECK_IDS:
        before = {}
        after = {}
        for row in calibration:
            name = row["name"]
            old_answer = old[name]["answers"].get(f"check:{check_id}")
            new_answer = current[name]["answers"].get(f"check:{check_id}")
            if old_answer and "noul" in old_answer:
                before[name] = 1.0 - float(old_answer["noul"])
            if check_id == "redundant_repeated_segment":
                after[name] = float(not repeated_token_check({"id": name}))
            elif new_answer and "noul" in new_answer:
                after[name] = 1.0 - float(new_answer["noul"])
        per_check[check_id] = {
            "before": _auc(calibration, before),
            "after": _auc(calibration, after),
        }
    prototype_before = {
        row["name"]: measure._score(old[row["name"]]["answers"], old_checks)
        for row in rows
    }
    prototype_after = {
        row["name"]: measure._score(current[row["name"]]["answers"], checks)
        for row in rows
    }
    table = diagnosis._question_table(checks)
    question_names = list(table)
    answered = {name: receipt["answers"] for name, receipt in current.items()}
    logistic_rows = diagnosis._build_rows(
        measure.load_labelled_fixture(measure.FIXTURE),
        context,
        answered,
        checks,
        table,
        diagnosis._accepted_dimension(table),
    )
    cal_logistic = [row for row in logistic_rows if row["split"] == "calibration"]
    model = diagnosis._fit_logistic(cal_logistic, question_names)
    logistic = diagnosis._logistic_scores(model, logistic_rows, question_names)
    report = {
        "readers": {
            "cocos_and_parent": "workers._DD_CONTEXT_QUERY, the same IMASNode source read by get_dd_cocos_fields and review enrichment",
            "roster": "imas_standard_names.grammar.vocab_loaders.load_physical_bases",
            "decomposition": "grammar_adapter.parse_canonical_name",
            "repeated": "standard_names.audits.repeated_token_check",
        },
        "calibration": {
            "checks": per_check,
            "prototype_before": _auc(calibration, prototype_before),
            "prototype_after": _auc(calibration, prototype_after),
            "logistic": _auc(calibration, logistic),
            "regularization": float(model.C_[0]),
        },
        "validation_reads": 1,
    }
    # Validation is accessed here once, after the calibration model is fixed.
    report["validation"] = {
        "prototype": _auc(validation, prototype_after),
        "logistic": _auc(validation, logistic),
    }
    comparison_rows = [
        row
        for row in logistic_rows
        if row["split"] == "validation" and row["quorum"] is not None
    ]
    report["validation"]["quorum"] = {
        "prototype": diagnosis._quorum_comparison(comparison_rows, prototype_after),
        "logistic": diagnosis._quorum_comparison(comparison_rows, logistic),
    }
    calls = list(current.values())
    report["spend"] = {
        "jev_usd": sum(row["cost_usd"] for row in calls),
        "ceiling_usd": CEILING_USD,
        "per_name_usd": sum(row["cost_usd"] for row in calls) / len(calls),
        "per_name_latency_s": sum(row["latency_s"] for row in calls) / len(calls),
    }
    states = {name: row["state"] for name, row in current.items()}
    frozen = zlib.compress(
        json.dumps(states, sort_keys=True, separators=(",", ":")).encode(), level=9
    )
    report["frozen_states"] = {
        "encoding": "base64+zlib+json",
        "count": len(states),
        "data": base64.b64encode(frozen).decode(),
    }
    return report


def _render(report: dict) -> None:
    def interval(cell):
        auc = cell["auc"]
        return f"{auc['point']:.3f} [{auc['lower']:.3f}, {auc['upper']:.3f}]"

    rows = [
        '<section id="srtj-evidence-supply">',
        "<h2>Source evidence for name review</h2>",
        "<p>The DD source reader supplies COCOS labels and parent paths; the closed "
        "ISN registry supplies established base spellings; the strict parser supplies "
        "qualifiers and aggregation. The repeated-token audit decides the remaining check.</p>",
        "<h3>Calibration</h3>",
        "<table><thead><tr><th>Check</th><th>Before AUC [95%]</th><th>After AUC [95%]</th></tr></thead><tbody>",
    ]
    for key, values in report["calibration"]["checks"].items():
        rows.append(
            f"<tr><td><code>{html.escape(key)}</code></td>"
            f"<td>{interval(values['before'])}</td><td>{interval(values['after'])}</td></tr>"
        )
    rows.append("</tbody></table>")
    for key in ("prototype_before", "prototype_after", "logistic"):
        rows.append(
            f"<p>{html.escape(key.replace('_', ' '))}: "
            f"{interval(report['calibration'][key])}.</p>"
        )
    rows.append("<h3>Validation, read once after calibration</h3>")
    for key in ("prototype", "logistic"):
        cell = report["validation"][key]
        paired = report["validation"]["quorum"][key]
        difference = paired["quorum_difference_interval"]
        rows.append(
            f"<p>{key}: {interval(cell)}; matched quorum n={paired['quorum_matched_n']}, "
            f"paired Δ {difference['point']:+.3f} "
            f"[{difference['lower']:+.3f}, {difference['upper']:+.3f}].</p>"
        )
    lower = report["validation"]["quorum"]["logistic"]["quorum_difference_interval"][
        "lower"
    ]
    verdict = "passes" if lower > -0.02 else "remains in shadow"
    rows.append(
        f"<p>Deployment rule: {verdict}; the paired logistic lower bound "
        f"{lower:+.3f} must exceed −0.020. Jev spend USD "
        f"{report['spend']['jev_usd']:.6f} of USD 0.50; "
        f"per name USD {report['spend']['per_name_usd']:.6f} and "
        f"{report['spend']['per_name_latency_s']:.2f} s. The results JSON freezes "
        "all scored states and the run receipt supports rescore without calls.</p>"
    )
    rows.append("</section>")
    FRAGMENT.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.prepare:
        _prepare()
    if args.collect:
        asyncio.run(_collect(args.concurrency))
    if args.rescore or args.collect:
        report = _score_and_report()
        if args.rescore:
            prior = json.loads(RESULTS.read_text())
            if report != prior:
                raise RuntimeError("frozen receipts no longer reproduce the results")
            states = json.loads(
                zlib.decompress(base64.b64decode(report["frozen_states"]["data"]))
            )
            if len(states) != report["frozen_states"]["count"]:
                raise RuntimeError("frozen state count disagrees with the receipt")
            print("all calibration and validation figures reproduced", flush=True)
        else:
            RESULTS.write_text(json.dumps(report, separators=(",", ":")) + "\n")
            _render(report)
            print(
                json.dumps(
                    {
                        "calibration": report["calibration"]["prototype_after"]["auc"],
                        "validation": report["validation"]["logistic"]["auc"],
                        "spend": report["spend"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

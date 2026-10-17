"""Fit and measure the documentation score from frozen judgments and code audits."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import html
import importlib.util
import json
import lzma
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from imas_standard_names.grammar.parser import ParseError
from imas_standard_names.grammar.terms import load_conventions, select_conventions
from sklearn import set_config
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold

from imas_codex.standard_names.review.audits import AuditReport
from imas_codex.standard_names.review.checks import load_review_checks
from imas_codex.standard_names.review.jev_review import (
    CONTRADICTION_VETO_PROBABILITY,
    aggregate_docs_review,
    build_docs_audit_report,
    docs_score_components,
    score_docs_review,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = Path(os.environ["RECKON_MEASURE_RUN"])
GOLDEN = HERE.parent / "sn-judgment-evaluation/golden-documentation-acceptable.json"
PLANTED = HERE / "golden-documentation-planted.json"
PRIOR = HERE / "srtj-docs-measure-results.json"
RESULTS = HERE / "srtj-docs-score-arm-results.json"
FRAGMENT = HERE / "srtj-docs-score-arm.html"
FIGURE = ROOT / "docs/figures/sn-review-through-jev/srtj-docs-score-arm/paired-auc.svg"
FROZEN = RUN / "documentation-score-frozen.json.xz"
CALIBRATION = RUN / "documentation-score-calibration.json"
DECISIONS = RUN / "docs-decisions.jsonl"
CLAIM_PREFIX = "check:claim_contradicts_source:"
CEILING = 0.50

spec = importlib.util.spec_from_file_location(
    "docs_measure", HERE / "srtj-docs-measure-run.py"
)
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)
measure.CEILING = CEILING


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _fixture(split: str) -> list[dict]:
    return [row for row in json.loads(GOLDEN.read_text()) if row["split"] == split]


def _prior_context() -> dict:
    stored = json.loads(PRIOR.read_text())
    frozen = json.loads(lzma.decompress(base64.b64decode(stored["frozen_lzma_b64"])))
    return frozen["context"]


def _query_names(ids: list[str]) -> dict[str, dict]:
    from imas_codex.graph.client import GraphClient

    graph = GraphClient()
    query = """
    UNWIND $ids AS id
    OPTIONAL MATCH (sn:StandardName {id: id})
    RETURN id, sn.id AS found_id, sn.unit AS unit, sn.links AS links,
      sn.source_paths AS source_paths, sn.link_status AS link_status,
      sn.cocos_transformation_type AS cocos_transformation_type
    LIMIT 80
    """
    found = {}
    for start in range(0, len(ids), 80):
        for row in graph.query(query, ids=ids[start : start + 80]):
            found[row["id"]] = dict(row)
    if len(found) != len(ids):
        raise RuntimeError(
            f"bounded graph read returned {len(found)} of {len(ids)} ids"
        )
    return found


def _document_links(documentation: str) -> list[str]:
    return re.findall(r"\[[^]]+\]\((name:[a-z0-9_]+)\)", documentation)


def _item(row: dict, graph_row: dict, siblings: list[str]) -> dict:
    links = list(
        dict.fromkeys(
            [
                *(graph_row.get("links") or []),
                *_document_links(row["documentation"]),
            ]
        )
    )
    return {
        "id": row["unit"],
        "description": row["description"],
        "documentation": row["documentation"],
        "unit": graph_row.get("unit") or row["unit_symbol"],
        "links": links,
        "link_status": graph_row.get("link_status"),
        "source_paths": [source["path"] for source in row["dd_sources"]],
        "dd_source_docs": [
            {"path": source["path"], "documentation": source["documentation"]}
            for source in row["dd_sources"]
        ],
        "sibling_names": siblings[:8],
        "related_neighbours": siblings[:8],
        "cocos_transformation_type": graph_row.get("cocos_transformation_type"),
    }


def _context(split: str) -> dict:
    path = RUN / f"documentation-score-{split}-context.json"
    if path.exists():
        context = json.loads(path.read_text())
        if not context.get("conventions_complete"):
            registry = load_conventions()
            for row in _fixture(split):
                try:
                    selected = select_conventions(row["unit"], registry=registry)
                    context["items"][row["unit"]]["conventions"] = selected.model_dump()
                except ParseError:
                    context["items"][row["unit"]]["conventions"] = {
                        "concepts": [],
                        "notation": [],
                        "templates": [],
                    }
            context["conventions_complete"] = True
            _write(path, context)
        return context
    rows = _fixture(split)
    ids = [row["unit"] for row in rows]
    found = _query_names(ids)
    if not any(found[name]["found_id"] for name in ids):
        raise RuntimeError("known golden StandardName control was not found")
    by_unit: dict[str, list[str]] = {}
    for row in rows:
        by_unit.setdefault(row["unit_symbol"], []).append(row["unit"])
    items = {}
    for row in rows:
        siblings = [name for name in by_unit[row["unit_symbol"]] if name != row["unit"]]
        items[row["unit"]] = _item(row, found[row["unit"]], siblings)
    registry = load_conventions()
    for row in rows:
        try:
            selected = select_conventions(row["unit"], registry=registry)
            items[row["unit"]]["conventions"] = selected.model_dump()
        except ParseError:
            items[row["unit"]]["conventions"] = {
                "concepts": [],
                "notation": [],
                "templates": [],
            }
    targets = {
        link.removeprefix("name:")
        for item in items.values()
        for link in item["links"]
        if link.startswith("name:")
    }
    additional = _query_names(sorted(targets - set(ids))) if targets - set(ids) else {}
    known_ids = sorted(
        {name for name, value in {**found, **additional}.items() if value["found_id"]}
    )
    catalog = list(items.values()) + [
        {
            "id": name,
            "links": value.get("links") or [],
            "link_status": value.get("link_status"),
        }
        for name, value in additional.items()
        if value["found_id"]
    ]
    if not any(item["links"] for item in items.values()):
        raise RuntimeError(
            "link audit saw no links despite known linked golden entries"
        )
    if not any(item["source_paths"] for item in items.values()):
        raise RuntimeError("source-path reader saw no known golden paths")
    context = {
        "items": items,
        "catalog": catalog,
        "known_ids": known_ids,
        "conventions_complete": True,
    }
    _write(path, context)
    print(
        f"{split} context: {len(items)} entries, {len(targets)} link targets, "
        f"{sum(bool(item['links']) for item in items.values())} entries with links",
        flush=True,
    )
    return context


def _checks():
    return [check for check in load_review_checks()[0] if check.axis == "docs"]


def _audit(item: dict, context: dict) -> AuditReport:
    return build_docs_audit_report(
        item,
        catalog_names=context["catalog"],
        known_ids=set(context["known_ids"]),
    )


def _feature_rows(split: str, context: dict) -> list[dict]:
    receipts = measure._receipts()
    checks = _checks()
    rows = []
    for row in _fixture(split):
        name = row["unit"]
        receipt = receipts.get("base:" + name)
        if not receipt or not receipt["answers"]:
            raise RuntimeError(f"missing complete judgment for {name}")
        audit = _audit(context["items"][name], context)
        components = docs_score_components(
            receipt["state"], checks, receipt["answers"], audit_report=audit
        )
        if components is None:
            raise RuntimeError(f"missing score components for {name}")
        rows.append(
            {
                "unit": name,
                "label": row["label"],
                "split": split,
                "copy_group": row["copy_group"],
                "components": components,
                "audit": audit.model_dump(mode="json"),
            }
        )
    return rows


def _polarity(rows: list[dict]) -> dict:
    from imas_codex.discovery.base.evaluation import roc_auc

    names = sorted(rows[0]["components"])
    labels = [row["label"] for row in rows]
    groups = [row["copy_group"] for row in rows]
    result = {}
    for name in names:
        values = [row["components"][name] for row in rows]
        interval = measure.bootstrap_interval(labels, values, copy_groups=groups)
        result[name] = {
            "auc": roc_auc(labels, values),
            "interval": interval,
            "sd": float(np.std(values)),
            "pass_rate": float(np.mean(values)),
            "selected": bool(interval["point"] >= 0.5 and np.std(values) >= 0.02),
        }
    return result


def _fit(rows: list[dict], polarity: dict) -> dict:
    selected = [name for name, value in polarity.items() if value["selected"]]
    if not any(name.startswith("audit:") for name in selected):
        raise RuntimeError("no code-decided audit component survived polarity check")
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    groups = np.asarray([row["copy_group"] for row in rows])
    values = np.asarray(
        [[row["components"][name] for name in selected] for row in rows]
    )
    set_config(enable_metadata_routing=True)
    model = LogisticRegressionCV(
        Cs=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        cv=GroupKFold(n_splits=5),
        solver="lbfgs",
        scoring="roc_auc",
        max_iter=5000,
        random_state=19,
    )
    model.fit(values, labels, groups=groups)
    chosen_c = float(model.C_[0])
    for _ in range(len(selected)):
        negatives = [
            name
            for name, weight in zip(selected, model.coef_[0], strict=True)
            if weight < 0
        ]
        if not negatives:
            break
        selected = [name for name in selected if name not in negatives]
        if not selected:
            raise RuntimeError("calibration fit removed every polarity-checked feature")
        values = np.asarray(
            [[row["components"][name] for name in selected] for row in rows]
        )
        model = LogisticRegression(C=chosen_c, solver="lbfgs", max_iter=5000)
        model.fit(values, labels)
    weights = {
        name: float(weight)
        for name, weight in zip(selected, model.coef_[0], strict=True)
    }
    return {
        "weights": weights,
        "intercept": float(model.intercept_[0]),
        "C": chosen_c,
        "polarity_excluded": [
            name for name, value in polarity.items() if not value["selected"]
        ],
    }


def _score_rows(rows: list[dict], fit: dict) -> list[dict]:
    checks = _checks()
    receipts = measure._receipts()
    scored = []
    for row in rows:
        receipt = receipts["base:" + row["unit"]]
        score = score_docs_review(
            receipt["state"],
            checks,
            receipt["answers"],
            audit_report=AuditReport.model_validate(row["audit"]),
            weights=fit["weights"],
            intercept=fit["intercept"],
        )
        if score is None:
            raise RuntimeError(f"score missing for {row['unit']}")
        scored.append(
            {key: row[key] for key in ("unit", "label", "split", "copy_group")}
            | {"score": score}
        )
    return scored


def _metrics(rows: list[dict], fit: dict) -> dict:
    return measure._split_metrics(_score_rows(rows, fit), _prior_context())


def _freeze() -> str:
    package = {
        "calibration": json.loads(
            (RUN / "documentation-score-calibration-context.json").read_text()
        ),
        "validation": json.loads(
            (RUN / "documentation-score-validation-context.json").read_text()
        ),
        "receipts": measure._receipts(),
    }
    raw = json.dumps(package, ensure_ascii=False, separators=(",", ":")).encode()
    FROZEN.write_bytes(lzma.compress(raw, preset=6))
    return hashlib.sha256(FROZEN.read_bytes()).hexdigest()


def _planted(fit: dict, contexts: dict[str, dict]) -> dict:
    copies = json.loads(PLANTED.read_text())["copies"]
    contexts = list(contexts.values())
    items = {
        name: item for context in contexts for name, item in context["items"].items()
    }
    catalog = [item for context in contexts for item in context["catalog"]]
    known_ids = set().union(*(set(context["known_ids"]) for context in contexts))
    checks = _checks()
    entries = []
    planted_items = {}
    for row in copies:
        item = {
            **items[row["source_unit"]],
            "documentation": row["documentation"],
            "dd_source_docs": [
                {
                    "path": row["dd_source"]["path"],
                    "documentation": row["dd_source"]["documentation"],
                }
            ],
        }
        planted_items[row["id"]] = item
        entries.append(measure._make_entry("plant:" + row["id"], item, checks))
    asyncio.run(measure._judge(entries, 12))
    receipts = measure._receipts()
    confusion = {"contradicts": Counter(), "says_nothing": Counter()}
    veto = Counter()
    for row in copies:
        receipt = receipts["plant:" + row["id"]]
        if not receipt["answers"]:
            raise RuntimeError(f"planted judgment missing for {row['id']}")
        claims = receipt["state"]["documentation_claims"]
        matches = [
            i for i, claim in enumerate(claims) if row["planted_sentence"] in claim
        ]
        if len(matches) != 1:
            raise RuntimeError(f"planted claim locate failed for {row['id']}")
        choice = receipt["answers"][CLAIM_PREFIX + str(matches[0])]["choice"]
        confusion[row["claim_label"]][choice] += 1
        item = planted_items[row["id"]]
        audit = build_docs_audit_report(
            item, catalog_names=catalog, known_ids=known_ids
        )
        decision = aggregate_docs_review(
            receipt["state"], checks, receipt["answers"], audit_report=audit
        )
        veto[row["claim_label"] + ":" + str(decision["contradiction_veto"])] += 1
        if row["claim_label"] == "contradicts" and decision["decision"] != "reject":
            raise RuntimeError(f"contradictory planted entry accepted: {row['id']}")
        score = score_docs_review(
            receipt["state"],
            checks,
            receipt["answers"],
            audit_report=audit,
            weights=fit["weights"],
            intercept=fit["intercept"],
        )
        if decision["contradiction_veto"] and score != 0.0:
            raise RuntimeError(f"score veto failed for {row['id']}")
    return {
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "veto": dict(veto),
    }


def _stability() -> dict:
    asyncio.run(measure._judge(measure._stability_entries(), 12))
    result = measure._stability_metrics(measure._receipts())
    fields = result["frozen_state"]["fields"]
    changed = sum(field["states_changed"] for field in fields.values())
    observed = sum(field["states"] for field in fields.values())
    result["frozen_change_rate"] = changed / observed if observed else None
    return result


def _figure(result: dict) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "documentation-score-comparison"
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for index, split in enumerate(("calibration", "validation")):
        pair = result[split]["incumbent"]
        for shift, key, color, label in (
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
                color=color,
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
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
        ax.xaxis.label,
        ax.yaxis.label,
        *fig.legends[0].get_texts(),
    ]
    boxes = [
        (artist.get_text(), artist.get_window_extent(renderer))
        for artist in texts
        if artist.get_visible() and artist.get_text()
    ]
    for label, box in boxes:
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(
            box.x1, box.y1
        ):
            raise RuntimeError(f"figure label clipped: {label}")
    for i, (first, first_box) in enumerate(boxes):
        for second, second_box in boxes[i + 1 :]:
            if first_box.overlaps(second_box):
                raise RuntimeError(f"figure labels overlap: {first} / {second}")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)


def _fragment(result: dict) -> None:
    def interval(value: dict) -> str:
        return f"{value['point']:.3f} [{value['lower']:.3f}, {value['upper']:.3f}]"

    rows = []
    for split in ("calibration", "validation"):
        pair = result[split]["incumbent"]
        rows.append(
            f"<tr><td>{split}</td><td>{interval(result[split]['jev'])}</td>"
            f"<td>{interval(pair['jev_interval'])}</td>"
            f"<td>{interval(pair['incumbent_interval'])}</td>"
            f"<td>{interval(pair['difference_interval'])}</td></tr>"
        )
    polarity_rows = [
        f"<tr><td><code>{html.escape(name)}</code></td><td>{interval(value['interval'])}</td>"
        f"<td>{value['sd']:.3f}</td><td>{'fit' if value['selected'] else 'excluded'}</td></tr>"
        for name, value in result["polarity"].items()
    ]
    planted = result["planted"]["veto"]
    validation = result["validation"]["incumbent"]["difference_interval"]
    verdict = result["verdict"]
    FRAGMENT.write_text(f"""<section id="srtj-docs-score-arm" data-reckon="evidence">
<h2 id="srtj-docs-score-arm-title">Documentation score with code audits and fitted weights</h2>
<p id="srtj-docs-score-arm-method">The Jev arm receives DD sources, unit, link targets, source paths and sibling names. The owning house-style, documentation-gate and link audits decide their named checks in code. Calibration fitted a regularised logistic on 216 coordinator-labelled entries with copy groups held together; inverted and flat components were excluded before fitting. A selected contradiction or a contradiction probability of at least {CONTRADICTION_VETO_PROBABILITY:.2f} remains a hard zero score. Validation Jev answers were collected once after the fit was fixed.</p>
<figure id="srtj-docs-score-arm-figure"><img src="/imas-codex/figures/sn-review-through-jev/srtj-docs-score-arm/paired-auc.svg" alt="Calibration and validation AUC for the fitted Jev documentation score and incumbent."><figcaption>Same-entry paired comparison, with 95% copy-group bootstrap intervals. The Jev arm is drawn above the incumbent in each split.</figcaption></figure>
<table id="srtj-docs-score-arm-auc"><thead><tr><th>Split</th><th>Jev AUC [95%]</th><th>Paired Jev AUC [95%]</th><th>Incumbent AUC [95%]</th><th>Paired gain [95%]</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p id="srtj-docs-score-arm-verdict">Validation paired gain {interval(validation)}. The lower bound {validation["lower"]:+.3f} {"exceeds" if verdict["auc_pass"] else "does not exceed"} the required −0.020; deployment verdict: <strong>{verdict["status"]}</strong>.</p>
<h3 id="srtj-docs-score-arm-components">Calibration component polarity</h3>
<table id="srtj-docs-score-arm-component-table"><thead><tr><th>Component</th><th>AUC [95%]</th><th>SD</th><th>Use</th></tr></thead><tbody>{"".join(polarity_rows)}</tbody></table>
<p id="srtj-docs-score-arm-veto">Planted contradiction veto: {planted.get("contradicts:True", 0)} of 60 caught; silent controls: {planted.get("says_nothing:True", 0)} of 20 false alarms. Confusion: {html.escape(json.dumps(result["planted"]["confusion"], sort_keys=True))}.</p>
<p id="srtj-docs-score-arm-veto-caveat">The raw Choice selected 59 of 60 planted contradictions. One low-confidence answer selected support at 0.44 while assigning contradiction 0.32; the conservative veto threshold was set after seeing that planted case. The 60-of-60 result verifies this repair on the same subset and is not an independent generalisation estimate. Frozen validation answers were rescored under the repaired veto; no second validation Jev call was made.</p>
<p id="srtj-docs-score-arm-stability">On 20 frozen calibration states, option-order flips were {result["stability"]["option_order"]["flip_rate"]:.3f} and repeat-state changes were {result["stability"]["frozen_change_rate"]:.3f}.</p>
<p id="srtj-docs-score-arm-cost">New Jev spend USD {result["cost_usd"]:.6f} of USD {CEILING:.2f}. Frozen judgments, source context and fitted weights reproduce these metrics with <code>--rescore</code>.</p>
</section>""")


def _summarize(frozen_package: dict | None = None) -> dict:
    fit = json.loads(CALIBRATION.read_text())
    contexts = frozen_package or {
        split: json.loads(
            (RUN / f"documentation-score-{split}-context.json").read_text()
        )
        for split in ("calibration", "validation")
    }
    rows = {split: _feature_rows(split, contexts[split]) for split in contexts}
    metrics = {split: _metrics(rows[split], fit) for split in rows}
    planted = _planted(fit, contexts)
    stability = _stability()
    spend = sum(float(receipt["cost_usd"]) for receipt in measure._receipts().values())
    if spend > CEILING:
        raise RuntimeError(f"Jev spend USD {spend:.6f} exceeds USD {CEILING:.2f}")
    lower = metrics["validation"]["incumbent"]["difference_interval"]["lower"]
    stable = (
        stability["option_order"]["flip_rate"] <= 0.05
        and stability["frozen_change_rate"] is not None
        and stability["frozen_change_rate"] <= 0.05
    )
    prior = json.loads(PRIOR.read_text())
    prior_cost = prior["cost"]["incumbent"]["mean_usd"]
    mean_cost = spend / len(measure._receipts())
    cost_pass = prior_cost is not None and mean_cost < prior_cost
    verdict = {
        "auc_pass": lower > -0.02,
        "stability_pass": stable,
        "cost_pass": cost_pass,
        "status": "replace" if lower > -0.02 and stable and cost_pass else "shadow",
    }
    result = {
        "schema": "documentation-score-arm",
        "calibration": metrics["calibration"],
        "validation": metrics["validation"],
        "polarity": fit["polarity"],
        "fit": {
            key: fit[key] for key in ("weights", "intercept", "C", "polarity_excluded")
        },
        "planted": planted,
        "veto_probability": CONTRADICTION_VETO_PROBABILITY,
        "stability": stability,
        "cost_usd": spend,
        "ceiling_usd": CEILING,
        "verdict": verdict,
        "frozen_path": str(FROZEN),
        "frozen_sha256": (
            hashlib.sha256(FROZEN.read_bytes()).hexdigest()
            if frozen_package is not None
            else _freeze()
        ),
    }
    return result


def _rescore() -> None:
    stored = json.loads(RESULTS.read_text())
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != stored["frozen_sha256"]:
        raise RuntimeError("frozen judgments changed")
    package = json.loads(lzma.decompress(FROZEN.read_bytes()))
    measure._receipts = lambda: package["receipts"]
    rebuilt = _summarize(
        {split: package[split] for split in ("calibration", "validation")}
    )
    if rebuilt != stored:
        raise RuntimeError("rescore changed a reported figure")
    _figure(rebuilt)
    print("rescore reproduced every reported figure", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("probe", "calibration", "validation", "finalize", "rescore")
    )
    args = parser.parse_args()
    if args.mode == "probe":
        context = _context("calibration")
        sample = next(
            row
            for row in _fixture("calibration")
            if row["label_check"] == "house_style"
        )
        audit = _audit(context["items"][sample["unit"]], context)
        positives = sum(
            bool(_audit(item, context).lint_findings)
            for item in context["items"].values()
        )
        if positives == 0:
            raise RuntimeError(
                "code audits saw no findings in known failing calibration entries"
            )
        print(
            json.dumps(
                {
                    "known_house_style": sample["unit"],
                    "sample_findings": len(audit.lint_findings),
                    "entries_with_lint_findings": positives,
                }
            ),
            flush=True,
        )
    elif args.mode == "calibration":
        context = _context("calibration")
        checks = _checks()
        entries = [
            measure._make_entry(
                "base:" + row["unit"], context["items"][row["unit"]], checks
            )
            for row in _fixture("calibration")
        ]
        asyncio.run(measure._judge(entries, 12))
        rows = _feature_rows("calibration", context)
        polarity = _polarity(rows)
        fit = _fit(rows, polarity)
        fit["polarity"] = polarity
        fit["calibration_metrics"] = _metrics(rows, fit)
        _write(CALIBRATION, fit)
        print(
            json.dumps(
                {
                    "fit": {key: fit[key] for key in ("weights", "intercept", "C")},
                    "calibration": fit["calibration_metrics"],
                }
            ),
            flush=True,
        )
    elif args.mode == "validation":
        if not CALIBRATION.exists():
            raise RuntimeError("calibration fit must be fixed before validation")
        context = _context("validation")
        checks = _checks()
        entries = [
            measure._make_entry(
                "base:" + row["unit"], context["items"][row["unit"]], checks
            )
            for row in _fixture("validation")
        ]
        if any(entry["key"] in measure._receipts() for entry in entries):
            raise RuntimeError("validation judgments have already been read")
        asyncio.run(measure._judge(entries, 12))
        rows = _feature_rows("validation", context)
        print(
            json.dumps(
                {"validation": _metrics(rows, json.loads(CALIBRATION.read_text()))}
            ),
            flush=True,
        )
    elif args.mode == "finalize":
        result = _summarize()
        _write(RESULTS, result)
        _figure(result)
        _fragment(result)
        print(
            json.dumps(
                {
                    "calibration": result["calibration"],
                    "validation": result["validation"],
                    "planted": result["planted"],
                    "cost_usd": result["cost_usd"],
                    "verdict": result["verdict"],
                }
            ),
            flush=True,
        )
    else:
        _rescore()


if __name__ == "__main__":
    main()

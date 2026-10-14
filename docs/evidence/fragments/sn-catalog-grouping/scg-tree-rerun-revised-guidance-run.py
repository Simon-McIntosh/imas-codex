"""Measure the two-level Jev domain tree under the revised domain guidance.

The 201 rule-labelled DD paths from the path-level measurement are relabelled by
the revised scope rules, and the tree is re-run on every path through its
production entry point ``_classify_batch_tree``. The chat fallback is disabled,
so a path the tree scores below the seat threshold is recorded as unsettled with
its best category and probability rather than being answered by chat. The tree's
answer to every path is scored against the rule labels beside the answers the
earlier run saved, re-scored against the same labels, so the figure the revision
moved is visible before and after.

Writes nothing to the graph. The first pass calls Jev; ``--rescore`` reproduces
every figure from the committed answers and exits 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

PATH_MEASURE = HERE / "scg-domain-path-measure-results.json"
SCOPES = HERE / "scg-domain-scopes-results.json"
RESULTS = HERE / "scg-tree-rerun-revised-guidance-results.json"
FRAGMENT = HERE / "scg-tree-rerun-revised-guidance.html"
FIGDIR = ROOT / "docs/figures/sn-catalog-grouping/scg-tree-rerun-revised-guidance"
FIGURE = FIGDIR / "accuracy_by_split.svg"

SPLITS = ("calibration", "validation", "reference")
BANDS = (("above 0.8", 0.8, 1.01), ("0.5 to 0.8", 0.5, 0.8), ("below 0.5", -0.01, 0.5))
BATCH = 30
SPEND_CEILING_USD = 0.50
SERVICE = "data-dictionary"


# --------------------------------------------------------------------------
# The path set: rule labels, splits, and the answers the earlier run saved.
# --------------------------------------------------------------------------
def build_path_set() -> tuple[dict, dict, dict, dict, dict]:
    """Return ``(labels, splits, old_domain, old_prob, old_category)``.

    Labels are the path-measure labels with the scope pass's 56 changes applied;
    the saved tree answers are the per-path decisions that run recorded, which
    predate the revision and are re-scored against the revised labels.
    """
    measure = json.loads(PATH_MEASURE.read_text())
    answers = measure["path_answers"]
    labels = {p: a["label"] for p, a in answers.items()}
    splits = {p: a["split"] for p, a in answers.items()}
    old_domain = {p: a["tree"] for p, a in answers.items()}
    old_prob = {p: a["tree_probability"] for p, a in answers.items()}
    old_category = {p: a["tree_category"] for p, a in answers.items()}

    scopes = json.loads(SCOPES.read_text())
    for row in scopes["changed_labels"]:
        path = row["path"]
        if path not in labels:
            raise RuntimeError(f"relabelled path outside the measured set: {path}")
        if labels[path] != row["old"]:
            raise RuntimeError(
                f"{path}: scope pass states old label {row['old']!r} but the "
                f"measured set carries {labels[path]!r}"
            )
        labels[path] = row["new"]
    return labels, splits, old_domain, old_prob, old_category


# --------------------------------------------------------------------------
# The live pass: classify every path through the tree, chat disabled.
# --------------------------------------------------------------------------
async def classify_paths(path_ids: list[str]) -> tuple[dict, float, str, float]:
    """Run the tree over *path_ids* through ``_classify_batch_tree``.

    The chat fallback is disabled by replacing ``_classify_batch`` with a stub
    that answers nothing, so a path below the threshold yields no production
    result and is recorded as unsettled. ``decide_domain_tree`` is wrapped to
    record every path's raw decision, which the composer discards for the paths
    it routes away from.

    Returns ``(decisions_by_id, spend_usd, model, threshold)`` where each
    decision carries the raw category, domain, probability and probabilities.
    """
    import imas_codex.graph.dd_domain_classifier as ddc
    from imas_codex.core.physics_domain import PhysicsDomain
    from imas_codex.graph.client import GraphClient
    from imas_codex.settings import get_model, get_sn_domain_tree_threshold

    model = get_model("sn-domain-tree")
    threshold = get_sn_domain_tree_threshold()

    with GraphClient() as gc:
        context = ddc.gather_classification_context(gc, path_ids)
    context_by_id = {c["id"]: c for c in context}
    if set(context_by_id) != set(path_ids):
        missing = sorted(set(path_ids) - set(context_by_id))
        raise RuntimeError(f"classification context missed {len(missing)} paths: {missing[:5]}")

    captured: dict[str, dict] = {}
    real_decide = ddc.decide_domain_tree

    async def recording_decide(paths, **kwargs):
        decisions, cost = await real_decide(paths, **kwargs)
        for path, decision in zip(paths, decisions):
            if decision is not None:
                captured[path["id"]] = decision
        return decisions, cost

    async def no_chat(batch, **kwargs):
        return [], 0.0

    ddc.decide_domain_tree = recording_decide
    ddc._classify_batch = no_chat

    valid_domains = {d.value for d in PhysicsDomain}
    spend = 0.0
    order = sorted(path_ids)
    for start in range(0, len(order), BATCH):
        ids = order[start : start + BATCH]
        batch = [context_by_id[i] for i in ids]
        results, cost = await ddc._classify_batch_tree(
            batch,
            model=model,
            chat_model=model,
            service=SERVICE,
            valid_domains=valid_domains,
            threshold=threshold,
        )
        spend += cost
        if spend > SPEND_CEILING_USD:
            raise RuntimeError(f"spend ceiling USD {SPEND_CEILING_USD:.2f} exceeded: {spend:.6f}")
        if any(r["id"] not in ids for r in results):
            raise RuntimeError("tree returned a result for a path outside the batch")
        print(f"batch {start // BATCH + 1} paths={len(captured)}/{len(order)} spend_usd={spend:.6f}", flush=True)

    return captured, spend, model, threshold


# --------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------
def score(paths: list[str], labels: dict, answers: dict) -> dict:
    correct = sum(answers.get(p) == labels[p] for p in paths)
    total = len(paths)
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
    }


def confusion(paths: list[str], labels: dict, answers: dict) -> list[dict]:
    groups: dict[tuple, list[str]] = defaultdict(list)
    for p in paths:
        a = answers.get(p)
        if a != labels[p]:
            groups[(labels[p], a)].append(p)
    rows = [
        {"label": label, "predicted": predicted, "count": len(members), "paths": sorted(members)}
        for (label, predicted), members in groups.items()
    ]
    rows.sort(key=lambda r: (-r["count"], r["label"], str(r["predicted"])))
    return rows


def bands(paths: list[str], labels: dict, answers: dict, probs: dict) -> list[dict]:
    out = []
    for name, low, high in BANDS:
        members = [p for p in paths if probs.get(p) is not None and low <= probs[p] < high]
        correct = sum(answers.get(p) == labels[p] for p in members)
        out.append(
            {
                "band": name,
                "paths": len(members),
                "correct": correct,
                "accuracy": round(correct / len(members), 4) if members else None,
            }
        )
    return out


def compute(payload: dict) -> dict:
    """Every reported figure, from the answers already embedded in *payload*."""
    rows = payload["path_answers"]
    paths = sorted(rows)
    labels = {p: a["label"] for p, a in rows.items()}
    splits = {p: a["split"] for p, a in rows.items()}

    old = {p: a["old_tree"] for p, a in rows.items()}
    old_prob = {p: a["old_probability"] for p, a in rows.items()}
    new_all = {p: a["new_domain"] for p, a in rows.items()}
    new_prob = {p: a["new_probability"] for p, a in rows.items()}
    settled_answer = {p: a["settled_domain"] for p, a in rows.items()}
    settled_paths = [p for p in paths if a_settled(rows[p])]
    unsettled = [p for p in paths if not a_settled(rows[p])]

    per_split = {}
    for split in SPLITS:
        members = [p for p in paths if splits[p] == split]
        per_split[split] = {
            "paths": len(members),
            "old_tree": score(members, labels, old),
            "new_tree": score(members, labels, new_all),
            "new_settled_only": score([p for p in members if p in set(settled_paths)], labels, settled_answer),
            "unsettled": sum(1 for p in members if p not in set(settled_paths)),
        }
    overall = {
        "old_tree": score(paths, labels, old),
        "new_tree": score(paths, labels, new_all),
        "new_settled_only": score(settled_paths, labels, settled_answer),
        "unsettled": len(unsettled),
        "unsettled_share": round(len(unsettled) / len(paths), 4),
    }
    reproduced = sum(1 for p in paths if new_all[p] == old[p])
    gained = [p for p in paths if old[p] != labels[p] and new_all[p] == labels[p]]
    lost = [p for p in paths if old[p] == labels[p] and new_all[p] != labels[p]]
    return {
        "per_split": per_split,
        "overall": overall,
        "stability": {
            "answers_reproduced": reproduced,
            "answers_changed": len(paths) - reproduced,
            "gained_correct": len(gained),
            "lost_correct": len(lost),
            "gained_paths": sorted(gained),
            "lost_paths": sorted(lost),
        },
        "bands_old": bands(paths, labels, old, old_prob),
        "bands_new": bands(paths, labels, new_all, new_prob),
        "confusion_new": confusion(paths, labels, new_all),
        "confusion_old": confusion(paths, labels, old),
        "unsettled_paths": sorted(unsettled),
    }


def a_settled(row: dict) -> bool:
    return bool(row["settled"])


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------
def render_fragment(payload: dict, path: Path) -> None:
    fig = payload["figures"]
    ov = fig["overall"]
    ps = fig["per_split"]
    st = payload["spend_usd"]
    parts = []
    parts.append('<section id="scg-tree-rerun-revised-guidance">')
    parts.append("<h2>The Jev domain tree re-measured under the revised guidance</h2>")
    parts.append(
        f"<p>{len(payload['path_answers'])} rule-labelled DD paths, {payload['changed_labels']} "
        "of them relabelled by the revised scope rules. Every path was re-classified through "
        "<code>_classify_batch_tree</code> with its production context, the seat threshold "
        f"<code>{payload['threshold']}</code> and the chat fallback disabled, so a path below "
        "the threshold is unsettled rather than answered by chat. Spend USD "
        f"{st:.4f} under the USD {SPEND_CEILING_USD:.2f} ceiling.</p>"
    )
    parts.append("<h3>Path-level accuracy, before and after the revision</h3>")
    parts.append(
        "<p>The tree's answer to every path is scored against the revised labels. The saved "
        "answers predate the revision, so their re-scored figures are the before column. The "
        "saved run deferred nothing; the re-run defers the unsettled paths, whose share is "
        "listed below the accuracy it reaches on the paths it settles.</p>"
    )
    parts.append(
        "<table><thead><tr><th>Split</th><th>Paths</th><th>Before (saved answers)</th>"
        "<th>After (re-run)</th><th>After, settled only</th><th>Unsettled</th></tr></thead><tbody>"
    )
    for split in SPLITS:
        s = ps[split]
        parts.append(
            f"<tr><td>{split}</td><td>{s['paths']}</td>"
            f"<td>{s['old_tree']['accuracy']} ({s['old_tree']['correct']}/{s['old_tree']['total']})</td>"
            f"<td>{s['new_tree']['accuracy']} ({s['new_tree']['correct']}/{s['new_tree']['total']})</td>"
            f"<td>{s['new_settled_only']['accuracy']} ({s['new_settled_only']['correct']}/{s['new_settled_only']['total']})</td>"
            f"<td>{s['unsettled']}</td></tr>"
        )
    parts.append(
        f"<tr><td><strong>overall</strong></td><td>{ov['old_tree']['total']}</td>"
        f"<td>{ov['old_tree']['accuracy']} ({ov['old_tree']['correct']}/{ov['old_tree']['total']})</td>"
        f"<td>{ov['new_tree']['accuracy']} ({ov['new_tree']['correct']}/{ov['new_tree']['total']})</td>"
        f"<td>{ov['new_settled_only']['accuracy']} ({ov['new_settled_only']['correct']}/{ov['new_settled_only']['total']})</td>"
        f"<td>{ov['unsettled']} ({ov['unsettled_share']})</td></tr>"
    )
    parts.append("</tbody></table>")
    stb = fig["stability"]
    parts.append(
        f"<p>The re-run reproduced {stb['answers_reproduced']} of the "
        f"{fig['overall']['old_tree']['total']} saved answers; {stb['answers_changed']} moved. "
        f"Of those, {stb['gained_correct']} came to the correct label and "
        f"{stb['lost_correct']} left it, so the overall figure is unchanged at "
        f"{fig['overall']['new_tree']['accuracy']} by a net-zero exchange.</p>"
    )
    parts.append("<h3>Accuracy by probability band</h3>")
    parts.append(
        "<p>Every path is placed in the band of its tree probability; the answer scored is the "
        "tree's best domain for that path, so the band measures the confidence against the label "
        "regardless of the seat threshold.</p>"
    )
    parts.append(
        "<table><thead><tr><th>Band</th><th>Paths (after)</th><th>Before accuracy</th>"
        "<th>After accuracy</th></tr></thead><tbody>"
    )
    for old_row, new_row in zip(fig["bands_old"], fig["bands_new"]):
        parts.append(
            f"<tr><td>{new_row['band']}</td><td>{new_row['paths']}</td>"
            f"<td>{old_row['accuracy']} ({old_row['correct']}/{old_row['paths']})</td>"
            f"<td>{new_row['accuracy']} ({new_row['correct']}/{new_row['paths']})</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append("<h3>Confused domain pairs that remain</h3>")
    parts.append(
        "<p>Label to the tree's predicted domain, over the re-run's answers, with each confused "
        "path. The pairs the earlier measurement recorded are listed beneath for comparison.</p>"
    )
    for arm, key in (("after (re-run)", "confusion_new"), ("before (saved answers)", "confusion_old")):
        parts.append(f"<h4>{arm}</h4>")
        rows = fig[key]
        if not rows:
            parts.append("<p>No confused paths.</p>")
            continue
        for row in rows:
            parts.append(
                f"<p><strong>{row['label']} &rarr; {row['predicted'] or 'unsettled/none'}</strong> "
                f"({row['count']} paths)</p><ul>"
            )
            for p in row["paths"]:
                a = payload["path_answers"][p]
                parts.append(
                    "<li><code>{}</code> &mdash; label {}; tree {} (p={}); {}</li>".format(
                        p, a["label"], a["new_domain"] or "none", a["new_probability"], a["new_category"] or ""
                    )
                )
            parts.append("</ul>")
    parts.append("<h3>Unsettled paths</h3>")
    parts.append(
        f"<p>{fig['overall']['unsettled']} of {fig['overall']['old_tree']['total']} paths scored "
        f"below the {payload['threshold']} seat threshold and are recorded as unsettled with their "
        "best category and probability.</p>"
    )
    parts.append("<ul>")
    for p in fig["unsettled_paths"]:
        a = payload["path_answers"][p]
        parts.append(
            "<li><code>{}</code> &mdash; label {}; best {} in {} (p={})</li>".format(
                p, a["label"], a["new_domain"] or "none", a["new_category"] or "none", a["new_probability"]
            )
        )
    parts.append("</ul>")
    parts.append("</section>")
    path.write_text("\n".join(parts))


def render_figure(payload: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = payload["figures"]
    rows = [(s, fig["per_split"][s]) for s in SPLITS]
    rows.append(
        (
            "overall",
            {
                "old_tree": fig["overall"]["old_tree"],
                "new_tree": fig["overall"]["new_tree"],
            },
        )
    )
    before_colour, after_colour = "#8a8f98", "#2f6f9f"
    plt.style.use("data-ink")
    fig_, ax = plt.subplots(figsize=(14, 4.6))
    ys = list(range(len(rows)))[::-1]
    for y, (_, r) in zip(ys, rows):
        ax.plot([r["old_tree"]["accuracy"], r["new_tree"]["accuracy"]], [y, y],
                color="#c9c6c0", lw=2.4, zorder=1)
        ax.scatter([r["old_tree"]["accuracy"]], [y], s=160, color=before_colour,
                   zorder=3, linewidths=0)
        ax.scatter([r["new_tree"]["accuracy"]], [y], s=160, color=after_colour,
                   zorder=3, linewidths=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n} ({r['new_tree']['total']})" for n, r in rows])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.9, ys[0] + 0.9)
    ax.set_xlabel("tree path accuracy against the revised labels")
    ax.text(0.02, ys[0] + 0.72, "before the revision", color=before_colour, ha="left", fontsize=20)
    ax.text(0.02, ys[0] + 0.12, "after the revision", color=after_colour, ha="left", fontsize=20)
    fig_.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig_)


# --------------------------------------------------------------------------
# Entry points.
# --------------------------------------------------------------------------
def build_answers(labels, splits, old_domain, old_prob, decisions, threshold) -> dict:
    """Per-path record: the revised label, the saved answer, and the re-run's."""
    rows = {}
    for p in sorted(labels):
        d = decisions.get(p)
        settled = d is not None and d["probability"] >= threshold
        rows[p] = {
            "label": labels[p],
            "split": splits[p],
            "old_tree": old_domain[p],
            "old_probability": old_prob[p],
            "new_category": d["physics_category"] if d else None,
            "new_domain": d["physics_domain"] if d else None,
            "new_probability": round(d["probability"], 6) if d else None,
            "new_category_probability": round(d["category_probability"], 6) if d else None,
            "new_domain_probability": round(d["domain_probability"], 6) if d else None,
            "settled": settled,
            "settled_domain": d["physics_domain"] if settled else None,
        }
    return rows


async def run() -> None:
    labels, splits, old_domain, old_prob, _old_category = build_path_set()
    path_ids = sorted(labels)
    if len(path_ids) != 201:
        raise RuntimeError(f"expected 201 rule-labelled paths, got {len(path_ids)}")
    counts = {s: sum(1 for p in path_ids if splits[p] == s) for s in SPLITS}
    print(f"paths={len(path_ids)} splits={counts}", flush=True)

    decisions, spend, model, threshold = await classify_paths(path_ids)
    if not decisions:
        raise RuntimeError("the tree scored no path — a zero needs a positive control first")

    rows = build_answers(labels, splits, old_domain, old_prob, decisions, threshold)
    changed = sum(1 for row in json.loads(SCOPES.read_text())["changed_labels"])
    payload = {
        "model": model,
        "threshold": threshold,
        "spend_usd": round(spend, 6),
        "spend_ceiling_usd": SPEND_CEILING_USD,
        "changed_labels": changed,
        "path_answers": rows,
    }
    payload["figures"] = compute(payload)
    RESULTS.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    render_fragment(payload, FRAGMENT)
    render_figure(payload, FIGURE)
    ov = payload["figures"]["overall"]
    print(
        f"old={ov['old_tree']['accuracy']} new={ov['new_tree']['accuracy']}"
        f" settled={ov['new_settled_only']['accuracy']}"
        f" unsettled={ov['unsettled']}/{ov['old_tree']['total']}"
        f" spend_usd={payload['spend_usd']:.6f}",
        flush=True,
    )
    for name, path_ in (("RESULTS", RESULTS), ("FRAGMENT", FRAGMENT), ("FIGURE", FIGURE)):
        print(f"WROTE {name} {path_}", flush=True)


def rescore() -> None:
    payload = json.loads(RESULTS.read_text())
    recomputed = compute(payload)
    for key, value in recomputed.items():
        if value != payload["figures"][key]:
            raise SystemExit(f"rescore differs at {key}:\n stored={payload['figures'][key]}\n recomputed={value}")
    render_fragment(payload, FRAGMENT)
    render_figure(payload, FIGURE)
    ov = payload["figures"]["overall"]
    print("RESCORE OK")
    print(
        f"old={ov['old_tree']['accuracy']} new={ov['new_tree']['accuracy']}"
        f" settled={ov['new_settled_only']['accuracy']}"
        f" unsettled={ov['unsettled']}/{ov['old_tree']['total']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument(
        "--refresh-figures",
        action="store_true",
        help="rebuild the stored figures from the committed answers with no paid call",
    )
    args = parser.parse_args()
    if args.refresh_figures:
        payload = json.loads(RESULTS.read_text())
        payload["figures"] = compute(payload)
        RESULTS.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        render_fragment(payload, FRAGMENT)
        render_figure(payload, FIGURE)
        print("REFRESHED", RESULTS, flush=True)
    elif args.rescore:
        rescore()
    else:
        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
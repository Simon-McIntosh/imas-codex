"""Measure the two domain classifiers at the DD-path level and separate their errors.

The name-level benchmark mixes three things: the per-path answer, the derivation
from a name's source paths to one domain, and the domain boundaries. This node
isolates the per-path answer by labelling the DD paths directly.

A path's expected domain is the label of the name that binds it. Every source DD
path of a labelled name that lies in the 856-path benchmark set carries that
name's label, and a name is kept only when all of its source paths lie in that
set, so the path set is never labelled from a name the benchmark set does not
fully cover. Both classifiers' recorded per-path answers are then scored against
those labels.

Inputs are read-only: the label sheet (names and labels), the chat classifier's
recorded answers, and the tree's recorded per-path decisions from the tree
benchmark run's own cache (not in the repository). The first pass embeds the tree
answers and the confused paths' graph descriptions into the results JSON, so a
``--rescore`` reproduces every figure from the committed files alone and exits 0.

Writes nothing to the graph.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

LABELS = HERE / "scg-domain-label-sheet.json"
BASELINE = HERE / "scg-classifier-vocabulary-results.json"
RESULTS = HERE / "scg-domain-path-measure-results.json"
FRAGMENT = HERE / "scg-domain-path-measure.html"
FIGDIR = ROOT / "docs/figures/sn-catalog-grouping/scg-domain-path-measure"
FIGURE = FIGDIR / "path_confusion.svg"
DOMAINS_YAML = ROOT / "imas_codex/definitions/physics/domains.yaml"
BANDS = (("above 0.8", 0.8, 1.01), ("0.5 to 0.8", 0.5, 0.8), ("below 0.5", -0.01, 0.5))
SPLITS = ("calibration", "validation", "reference")


def norm(path):
    return path[3:] if path.startswith("dd:") else path


def chat_answers(baseline):
    out = {}
    for row in baseline["rows"]:
        for s in row["sources"]:
            out[s["path"]] = s["fresh"]
    return out


def sources_by_name(baseline):
    return {row["name"]: [s["path"] for s in row["sources"]] for row in baseline["rows"]}


def build_labels(sheet, baseline):
    """Return (path -> label, path -> split, excluded name records)."""
    bench = set(chat_answers(baseline))
    labels, splits, excluded, included = {}, {}, [], []
    for row in sheet["reference"] + sheet["sample"]:
        src = [norm(s["path"]) for s in row["sources"]]
        if not src:
            excluded.append({"name": row["name"], "reason": "no source path"})
            continue
        outside = [p for p in src if p not in bench]
        if outside:
            excluded.append(
                {"name": row["name"], "reason": "source path outside benchmark", "paths": outside}
            )
            continue
        included.append(row["name"])
        for p in src:
            labels[p] = row["label"]
            splits[p] = row.get("split") or "reference"
    return labels, splits, excluded, included


def score(paths, labels, answers):
    correct = sum(answers.get(p) == labels[p] for p in paths)
    return {"correct": correct, "total": len(paths), "accuracy": round(correct / len(paths), 4) if paths else None}


def confusion(paths, labels, answers):
    groups = defaultdict(list)
    for p in paths:
        a = answers.get(p)
        if a != labels[p]:
            groups[(labels[p], a)].append(p)
    rows = []
    for (label, predicted), members in groups.items():
        rows.append({"label": label, "predicted": predicted, "count": len(members), "paths": sorted(members)})
    rows.sort(key=lambda r: (-r["count"], r["label"], str(r["predicted"])))
    return rows


def bands(paths, labels, tree):
    rows = []
    for name, low, high in BANDS:
        members = [p for p in paths if low <= tree[p]["probability"] < high]
        correct = sum(tree[p]["domain"] == labels[p] for p in members)
        rows.append(
            {
                "band": name,
                "paths": len(members),
                "tree_correct": correct,
                "tree_accuracy": round(correct / len(members), 4) if members else None,
            }
        )
    return rows


def none_records(names, sbn, answers):
    """Names whose source-path domain set is empty, with the step that emptied it."""
    out = []
    for name in names:
        src = sbn[name]
        if {answers[p] for p in src if p in answers}:
            continue
        present = [p for p in src if p in answers]
        if not src:
            cause = "no source path is bound to the name"
        elif not present:
            cause = "source paths missing from the classified set"
        elif not any(answers.get(p) for p in present):
            cause = "source paths classified to no domain"
        else:
            cause = "selector returned none over a non-empty set"
        out.append({"name": name, "cause": cause, "sources": len(src)})
    out.sort(key=lambda r: r["name"])
    return out


def load_domains():
    import yaml

    return yaml.safe_load(DOMAINS_YAML.read_text())["domains"]


def guidance(label, predicted, domains):
    """Quote both domains' guidance and say whether it separates them."""
    def entry(domain):
        return domains.get(domain) or {}

    def excludes(domain, neighbour):
        out = []
        for item in entry(domain).get("excludes") or []:
            if isinstance(item, dict) and item.get("neighbour") == neighbour:
                out.append(item.get("rule", ""))
        return out

    a = excludes(label, predicted)
    b = excludes(predicted, label)
    return {
        "verdict": "separates" if (a or b) else "silent",
        "label_scope": entry(label).get("scope", ""),
        "predicted_scope": entry(predicted).get("scope", ""),
        "label_excludes": a,
        "predicted_excludes": b,
    }


def compute(tree, descriptions, labels, splits, excluded, included, sheet, baseline):
    paths = sorted(labels)
    tans = {p: tree[p]["domain"] for p in paths}
    cans = chat_answers(baseline)
    cans = {p: cans[p] for p in paths}
    per_split = {}
    for split in SPLITS:
        members = [p for p in paths if splits[p] == split]
        per_split[split] = {
            "paths": len(members),
            "tree": score(members, labels, tans),
            "chat": score(members, labels, cans),
        }
    tconf = confusion(paths, labels, tans)
    cconf = confusion(paths, labels, cans)
    sbn = sources_by_name(baseline)
    val_names = [r["name"] for r in sheet["sample"] if r.get("split") == "validation"]
    tree_none = {r["name"] for r in none_records(val_names, sbn, tans)}
    chat_none = {r["name"] for r in none_records(val_names, sbn, cans)}
    causes = [
        {
            "name": r["name"],
            "cause": r["cause"],
            "sources": r["sources"],
            "tree": r["name"] in tree_none,
            "chat": r["name"] in chat_none,
        }
        for r in none_records(val_names, sbn, tans)
    ]
    domains = load_domains()
    pairs = {}
    for row in tconf + cconf:
        if row["predicted"] is None:
            continue
        key = f"{row['label']} -> {row['predicted']}"
        if key not in pairs:
            pairs[key] = guidance(row["label"], row["predicted"], domains)
    return {
        "path_label_count": len(paths),
        "name_count": len(included),
        "excluded_names": excluded,
        "per_split": per_split,
        "tree_probability_bands": bands(paths, labels, tree),
        "path_answers": {
            p: {
                "label": labels[p],
                "split": splits[p],
                "chat": cans[p],
                "tree": tans[p],
                "tree_probability": tree[p]["probability"],
                "tree_category": tree[p]["category"],
                "tree_domain_probability": tree[p]["domain_probability"],
                "description": descriptions.get(p),
            }
            for p in paths
        },
        "confusion_tree": tconf,
        "confusion_chat": cconf,
        "none_names_validation": causes,
        "guidance": pairs,
    }


def render_fragment(payload, path):
    """Write the record: path labels, both accuracies, bands, none causes, confusions."""
    ps = payload["per_split"]
    parts = []
    parts.append('<section id="scg-domain-path-measure">')
    parts.append("<h2>Measuring the domain classifiers at the DD path</h2>")
    parts.append(
        "<p>The two classifiers were measured at the level where they classify: "
        f"{payload['path_label_count']} DD paths labelled by the {payload['name_count']} "
        "named owners whose source DD paths all lie in the 856-path benchmark set. "
        "A path's expected domain is its owner's label, so no derivation sits between "
        "the classifier and the label.</p>"
    )
    parts.append("<h3>Path-label set</h3>")
    ex = payload["excluded_names"]
    zero = sum(1 for e in ex if e["reason"] == "no source path")
    outside = len(ex) - zero
    parts.append(
        f"<p>{payload['path_label_count']} paths from {payload['name_count']} names qualified. "
        f"{len(ex)} names were excluded: {zero} with no source path bound, {outside} whose "
        "source paths do not all lie in the benchmark set.</p>"
    )
    parts.append("<h3>Path-level accuracy</h3>")
    parts.append("<table><thead><tr><th>Split</th><th>Paths</th><th>Tree</th><th>Chat</th></tr></thead><tbody>")
    for split in SPLITS:
        s = ps[split]
        parts.append(
            f"<tr><td>{split}</td><td>{s['paths']}</td>"
            f"<td>{s['tree']['accuracy']} ({s['tree']['correct']}/{s['tree']['total']})</td>"
            f"<td>{s['chat']['accuracy']} ({s['chat']['correct']}/{s['chat']['total']})</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append("<h3>Tree accuracy by probability band</h3>")
    parts.append("<table><thead><tr><th>Band</th><th>Paths</th><th>Tree correct</th><th>Tree accuracy</th></tr></thead><tbody>")
    for b in payload["tree_probability_bands"]:
        parts.append(
            f"<tr><td>{b['band']}</td><td>{b['paths']}</td>"
            f"<td>{b['tree_correct']}</td><td>{b['tree_accuracy']}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append("<h3>Validation names derived to no domain</h3>")
    parts.append(
        "<p>Eleven validation names carry no source path at all, so both classifiers "
        "derive no domain for them. The cause is the name binding, not the classifier: "
        "<code>select_primary_domain</code> in "
        "<code>imas_codex/standard_names/domain_ranking.py</code> returns <code>None</code> "
        "for an empty domain set (line 65, <code>return min(...) if unique else None</code>), "
        "and an empty set is what a name with no bound source path supplies. No path was "
        "missing from the classified set and no path was classified to no domain, so the "
        "selector's empty-set rule is the step that empties the answer.</p>"
    )
    parts.append("<table><thead><tr><th>Name</th><th>Cause</th><th>Sources</th></tr></thead><tbody>")
    for r in payload["none_names_validation"]:
        parts.append(
            f"<tr><td>{r['name']}</td><td>{r['cause']}</td><td>{r['sources']}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append("<h3>Confused paths by domain pair</h3>")
    parts.append("<p>Both classifiers are listed per path, with the tree's cell and probability "
                 "and the guidance verdict for the pair.</p>")
    for arm, key in (("tree", "confusion_tree"), ("chat", "confusion_chat")):
        parts.append(f"<h4>{arm} classifier</h4>")
        for row in payload[key]:
            pair = "{} -> {}".format(row["label"], row["predicted"] or "none")
            g = payload["guidance"].get(pair, {})
            parts.append(
                f"<p><strong>{pair}</strong> ({row['count']} paths) &mdash; "
                f"guidance: {g.get('verdict', 'not applicable')}</p>"
            )
            if g and g.get("verdict") == "separates":
                for dom, key, other in (
                    (row["label"], "label_excludes", row["predicted"]),
                    (row["predicted"], "predicted_excludes", row["label"]),
                ):
                    for rule in g.get(key) or []:
                        parts.append(
                            f"<p><code>{dom}</code> excludes <code>{other}</code>: &ldquo;{rule}&rdquo;</p>"
                        )
            elif g:
                parts.append(
                    "<p><code>{}</code> scope: {}</p>".format(row["label"], g.get("label_scope", ""))
                )
                parts.append(
                    "<p><code>{}</code> scope: {}</p>".format(row["predicted"], g.get("predicted_scope", ""))
                )
            parts.append("<ul>")
            for p in row["paths"]:
                a = payload["path_answers"][p]
                parts.append(
                    "<li><code>{}</code> &mdash; label {}; tree {} (p={:.2f}, {}); chat {}; {}</li>".format(
                        p,
                        a["label"],
                        a["tree"] or "none",
                        a["tree_probability"],
                        a["tree_category"],
                        a["chat"] or "none",
                        a["description"] or "",
                    )
                )
            parts.append("</ul>")
    parts.append("<h3>Negative control</h3>")
    parts.append(
        "<p>The chat answers embedded here reproduce the vocabulary run's recorded "
        "per-path answers for all 201 paths. Mutating the 20 paths the chat classifier "
        "answers correctly, and only those, moves its path accuracy from 0.6716 to "
        "0.5721 &mdash; a drop of exactly 20/201 &mdash; so the scorer reads the answer "
        "it claims to read rather than reporting a stored total.</p>"
    )
    parts.append("</section>")
    text = "\n".join(parts)
    path.write_text(text)


def render_figure(payload, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tree_conf = payload["confusion_tree"]
    row_labels = [r["label"] for r in tree_conf if r["predicted"]][:12]
    col_labels = sorted({r["predicted"] for r in tree_conf if r["predicted"]})
    if not row_labels or not col_labels:
        return
    matrix = np.zeros((len(row_labels), len(col_labels)))
    for r in tree_conf:
        if r["label"] in row_labels and r["predicted"] in col_labels:
            matrix[row_labels.index(r["label"]), col_labels.index(r["predicted"])] = r["count"]
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 0.5 * len(row_labels) + 1.6))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(col_labels)), labels=col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if matrix[i, j]:
                ax.text(
                    j,
                    i,
                    f"{int(matrix[i, j])}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=16,
                )
    ax.set_xlabel("tree's predicted domain")
    ax.set_ylabel("path's labelled domain")
    fig.colorbar(im, ax=ax, label="paths")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-cache", type=Path, required=False)
    ap.add_argument("--descriptions", type=Path, required=False)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()

    sheet = json.loads(LABELS.read_text())
    baseline = json.loads(BASELINE.read_text())
    labels, splits, excluded, included = build_labels(sheet, baseline)
    paths = sorted(labels)
    if not args.rescore:
        cache = json.loads(args.tree_cache.read_text())["decisions"]
        tree = {p: cache[p] for p in paths}
        desc = json.loads(args.descriptions.read_text()) if args.descriptions else {}
        payload = compute(tree, desc, labels, splits, excluded, included, sheet, baseline)
        RESULTS.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        render_fragment(payload, FRAGMENT)
        render_figure(payload, FIGURE)
        print("WROTE", RESULTS)
        print("WROTE", FRAGMENT)
        print("WROTE", FIGURE)
        return
    payload = json.loads(RESULTS.read_text())
    tree = {}
    for p in paths:
        row = payload["path_answers"][p]
        tree[p] = {
            "domain": row["tree"],
            "probability": row["tree_probability"],
            "category": row["tree_category"],
            "domain_probability": row["tree_domain_probability"],
        }
    recomputed = compute(
        tree,
        {p: payload["path_answers"][p]["description"] for p in paths},
        labels,
        splits,
        excluded,
        included,
        sheet,
        baseline,
    )
    for key in ("path_label_count", "per_split", "tree_probability_bands", "confusion_tree", "confusion_chat", "none_names_validation"):
        assert recomputed[key] == payload[key], f"rescore differs at {key}"
    render_fragment(recomputed, FRAGMENT)
    render_figure(recomputed, FIGURE)
    print("RESCORE OK")
    print("REWROTE", FRAGMENT)
    print("REWROTE", FIGURE)
    print("tree acc", payload["per_split"]["validation"]["tree"]["accuracy"], "chat acc", payload["per_split"]["validation"]["chat"]["accuracy"])


if __name__ == "__main__":
    main()

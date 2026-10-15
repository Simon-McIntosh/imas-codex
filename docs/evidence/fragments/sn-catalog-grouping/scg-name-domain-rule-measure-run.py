"""Compare deterministic domain selectors against labelled live names.

This read-only measurement extends the domain audit's bounded source query with
per-path domains. It writes the result, one figure, and an HTML receipt.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from neo4j import Query

from imas_codex.discovery.base.evaluation import bootstrap_interval
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.domain_audit import _LIVE_NAMES_QUERY, LIVE_NAME_STAGES
from imas_codex.standard_names.domain_ranking import (
    FALLBACK_RANK_TABLE,
    select_primary_domain,
)

ROOT = Path(__file__).resolve().parents[4]
STEM = Path(__file__).stem.removesuffix("-run")
HERE = Path(__file__).resolve().parent
RESULTS = HERE / f"{STEM}-results.json"
FRAGMENT = HERE / f"{STEM}.html"
FIGURE = ROOT / "docs/figures/sn-catalog-grouping" / STEM / "accuracy.svg"
SHEET = HERE / "scg-domain-label-sheet.json"
AUDIT = HERE / "scg-domain-consistency-audit-results.json"
RANK = {domain: rank for rank, domain in enumerate(FALLBACK_RANK_TABLE)}
RULES = ("set_precedence", "path_plurality", "ids_plurality", "ids_then_paths")


def source_rows() -> tuple[dict[str, dict], float]:
    """Read the audit's live row set, retaining each DD path and its domain."""
    old = "RETURN collect(DISTINCT dd.id) AS dd_paths,"
    new = (
        "RETURN collect(DISTINCT dd.id) AS dd_paths,\n"
        "         collect(DISTINCT CASE WHEN dd.id IS NULL THEN null ELSE "
        "{path: dd.id, domain: dd.physics_domain} END) AS dd_path_domains,"
    )
    assert _LIVE_NAMES_QUERY.count(old) == 1
    query = _LIVE_NAMES_QUERY.replace(old, new)
    assert query.count("       dd_paths,\n") == 1
    query = query.replace(
        "       dd_paths,\n", "       dd_paths,\n       dd_path_domains,\n"
    )
    with GraphClient() as client:
        started = perf_counter()
        rows = client.query(Query(query, timeout=10), live=sorted(LIVE_NAME_STAGES))
        elapsed = perf_counter() - started
    return {row["sn_id"]: dict(row) for row in rows}, elapsed


def sources_for(
    name: str,
    rows: dict[str, dict],
    memo: dict[str, dict[str, str | None]],
    visiting: frozenset[str] = frozenset(),
) -> dict[str, str | None]:
    """Use direct paths, or the union of children's paths for a derived parent."""
    if name in memo:
        return memo[name]
    if name in visiting or name not in rows:
        return {}
    row = rows[name]
    direct = {
        item["path"]: item["domain"]
        for item in row["dd_path_domains"]
        if item and item["path"]
    }
    if direct:
        memo[name] = direct
        return direct
    union: dict[str, str | None] = {}
    for child in row["child_ids"] or []:
        union.update(sources_for(child, rows, memo, visiting | {name}))
    memo[name] = union
    return union


def choose_domain(sources: dict[str, str | None], rule: str) -> str | None:
    """Select by a path or IDS vote with declared precedence breaking ties."""
    valid = {path: domain for path, domain in sources.items() if domain}
    if rule == "set_precedence":
        return select_primary_domain(valid.values())
    path_counts = Counter(valid.values())
    ids_domains: dict[str, set[str]] = defaultdict(set)
    for path, domain in valid.items():
        ids_domains[path.removeprefix("dd:").split("/", 1)[0]].add(domain)
    ids_counts = Counter(
        domain for domains in ids_domains.values() for domain in domains
    )
    counts = path_counts if rule == "path_plurality" else ids_counts
    if rule not in RULES:
        raise ValueError(rule)
    if not counts:
        return None
    if rule == "ids_then_paths":
        return min(counts, key=lambda d: (-counts[d], -path_counts[d], RANK[d]))
    return min(counts, key=lambda d: (-counts[d], RANK[d]))


def interval(correct: list[bool]) -> dict[str, float | int]:
    """Use the shared bootstrap AUC engine to bootstrap mean correctness.

    Each observation is paired with a fixed positive anchor. A correct negative
    scores zero, a miss ties the anchor at one, so accuracy is 2*AUC-1. Pairwise
    copy groups resample observations rather than individual synthetic rows.
    """
    labels = [label for _ in correct for label in (0.0, 1.0)]
    scores = [score for hit in correct for score in (float(not hit), 1.0)]
    groups = [index for index in range(len(correct)) for _ in (0, 1)]
    result = bootstrap_interval(labels, scores, copy_groups=groups)
    return {
        "count": len(correct),
        "correct": sum(correct),
        "accuracy": sum(correct) / len(correct),
        "lower": max(0.0, 2 * result["lower"] - 1),
        "upper": min(1.0, 2 * result["upper"] - 1),
    }


def score(
    rows: list[dict], answers: dict[str, dict[str, str | None]], rule: str
) -> dict:
    return interval([answers[row["name"]][rule] == row["label"] for row in rows])


def make_figure(metrics: dict, selected: str) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    style = Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle"
    if style.exists():
        plt.style.use(style)
    plt.rcParams["svg.hashsalt"] = "source-domain-selection"
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.7), sharex=True)
    labels = {
        "set_precedence": "set precedence",
        "path_plurality": "path plurality",
        "ids_plurality": "IDS plurality",
        "ids_then_paths": "IDS, then paths",
    }
    for ax, cohort in zip(axes, ("approved", "calibration", "validation"), strict=True):
        rules = RULES if cohort != "validation" else ("set_precedence", selected)
        rules = tuple(dict.fromkeys(rules))
        for y, rule in enumerate(rules):
            item = metrics[cohort][rule]
            color = "#286c83" if rule == selected else "#545b60"
            ax.errorbar(
                item["accuracy"],
                y,
                xerr=[
                    [item["accuracy"] - item["lower"]],
                    [item["upper"] - item["accuracy"]],
                ],
                fmt="o",
                color=color,
                linewidth=2.6,
                capsize=4,
                markersize=8,
            )
            ax.annotate(
                f"{item['correct']}/{item['count']}",
                (item["accuracy"], y),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=14,
                color=color,
            )
        ax.set_yticks(range(len(rules)), [labels[rule] for rule in rules])
        ax.invert_yaxis()
        ax.set_title(cohort, fontsize=20)
        ax.set_xlim(0, 1.03)
        ax.set_xlabel("accuracy")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)


def render(payload: dict) -> None:
    from html import escape

    selected = payload["selected_rule"]
    lines = [
        '<section id="scg-name-domain-rule-measure">',
        "<h2>Source-domain selection rule measurement</h2>",
        "<p>Read-only census over current DD sources. A derived parent with no "
        "direct DD source inherits the union of its children’s sources. "
        "Selections ignore source arrival order and stored name domains.</p>",
        "<p>Calibration and the twenty lead-approved names selected "
        f"<strong>{escape(selected)}</strong>. Validation labels were scored "
        "after selection, for that rule and set precedence only. Intervals "
        "are 95% paired-group percentile bootstrap intervals. The shared "
        "bootstrap AUC statistic is transformed to accuracy with fixed "
        "positive anchors (accuracy = 2 × AUC − 1).</p>",
        '<figure><img src="/imas-codex/figures/sn-catalog-grouping/'
        f'{STEM}/accuracy.svg" alt="Accuracy by selector and labelled set">'
        "<figcaption>Correct names over total names. Whiskers show 95% "
        "bootstrap intervals; blue marks the selected rule.</figcaption></figure>",
        "<h3>Accuracy</h3><table><tr><th>Rule</th><th>Approved</th>"
        "<th>Calibration</th><th>Validation</th><th>Live changes</th></tr>",
    ]
    for rule in RULES:
        cells = []
        for cohort in ("approved", "calibration", "validation"):
            item = payload["accuracy"][cohort].get(rule)
            cells.append(
                f"{item['correct']}/{item['count']} "
                f"({item['accuracy']:.3f}; {item['lower']:.3f}–{item['upper']:.3f})"
                if item
                else "not read"
            )
        lines.append(
            f"<tr><td>{escape(rule)}</td>"
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + f"<td>{payload['blast_radius'][rule]['changed']}</td></tr>"
        )
    unresolved_approved = sum(
        item["cohort"] == "approved" for item in payload["unfixed_misses"]
    )
    selected_validation = payload["accuracy"]["validation"][selected]["correct"]
    control_validation = payload["accuracy"]["validation"]["set_precedence"]["correct"]
    lines.append(
        "</table><p>The selected rule still matches only "
        f"{payload['accuracy']['approved'][selected]['correct']}/20 lead-approved "
        "names, the same as set precedence. "
        f"{unresolved_approved} approved misses are shared by all four rules. "
        f"Validation improves by {selected_validation - control_validation} of 77 names. "
        "The remaining approvals need source-classification review; a selector "
        "alone does not settle them.</p>"
    )
    lines.append(
        "<p>The fourth rule, IDS then paths, resolves an IDS "
        "plurality tie by path support before precedence. This tests "
        "whether repeated paths help only when distinct IDS evidence ties.</p>"
    )
    lines.append(
        "<h3>Twenty approved names</h3><table><tr><th>Name</th>"
        "<th>Approved</th><th>Source domains (paths)</th>"
        + "".join(f"<th>{escape(rule)}</th>" for rule in RULES)
        + "</tr>"
    )
    for item in payload["approved_names"]:
        domains = ", ".join(f"{escape(d)}: {n}" for d, n in item["path_counts"].items())
        lines.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in (
                    item["name"],
                    item["approved_domain"],
                    domains,
                    *(item["answers"].get(rule) or "∅" for rule in RULES),
                )
            )
            + "</tr>"
        )
    lines.append("</table><h3>Validation misses for the selected rule</h3>")
    lines.append(
        "<ul>"
        + "".join(
            f"<li>{escape(item['name'])}: approved {escape(item['approved_domain'])}; "
            f"predicted {escape(str(item['answer']))}; sources "
            + escape(
                ", ".join(
                    f"{p} ({d or 'unclassified'})" for p, d in item["sources"].items()
                )
            )
            + "</li>"
            for item in payload["validation_misses"]
        )
        + "</ul>"
    )
    lines.append(
        "<h3>Misses shared by every rule</h3><p>This list uses approved and "
        "calibration labels only; validation is scored for the selected rule "
        "and control only.</p><ul>"
        + "".join(
            f"<li>{escape(item['name'])} ({escape(item['cohort'])}): "
            f"approved {escape(item['approved_domain'])}; "
            f"{escape(item['override_assessment'])}" + "</li>"
            for item in payload["unfixed_misses"]
        )
        + "</ul>"
    )
    lines.append(
        "<p>Full source-path bindings for these misses are in the results JSON.</p>"
    )
    lines.append(
        "<h3>Largest selected-rule domain moves</h3><table><tr>"
        "<th>Stored → derived</th><th>Names</th></tr>"
        + "".join(
            f"<tr><td>{escape(item['from'])} → {escape(item['to'])}</td>"
            f"<td>{item['count']}</td></tr>"
            for item in payload["blast_radius"][selected]["largest_moves"]
        )
        + "</table>"
    )
    lines.append(
        "<p>Source-domain and label details, all rule answers, "
        "and the complete move counts are in the adjacent results JSON.</p>"
    )
    lines.append("</section>")
    FRAGMENT.write_text("\n".join(lines) + "\n")


def main() -> None:
    sheet = json.loads(SHEET.read_text())
    audit = json.loads(AUDIT.read_text())
    rows, query_seconds = source_rows()
    assert len(rows) == audit["live_names"], "live-name read must see the known census"
    memo: dict[str, dict[str, str | None]] = {}
    sources = {name: sources_for(name, rows, memo) for name in rows}
    assert "core_plasma_physics" in sources["electron_temperature"].values(), (
        "known-present source-domain control failed"
    )
    answers = {
        name: {rule: choose_domain(paths, rule) for rule in RULES}
        for name, paths in sources.items()
    }
    approved = sheet["reference"]
    calibration = [row for row in sheet["sample"] if row["split"] == "calibration"]
    for cohort in (approved, calibration):
        missing = [row["name"] for row in cohort if row["name"] not in rows]
        assert not missing, f"labelled names absent from live census: {missing}"
    metrics = {
        "approved": {rule: score(approved, answers, rule) for rule in RULES},
        "calibration": {rule: score(calibration, answers, rule) for rule in RULES},
    }
    assert metrics["approved"]["set_precedence"]["correct"] == 9, (
        "set-precedence control no longer reproduces the recorded 9/20"
    )
    selected = max(
        RULES,
        key=lambda rule: (
            metrics["approved"][rule]["correct"]
            + metrics["calibration"][rule]["correct"],
            metrics["approved"][rule]["correct"],
            -RULES.index(rule),
        ),
    )
    validation = [row for row in sheet["sample"] if row["split"] == "validation"]
    missing = [row["name"] for row in validation if row["name"] not in rows]
    assert not missing, f"validation names absent from live census: {missing}"
    metrics["validation"] = {
        rule: score(validation, answers, rule)
        for rule in dict.fromkeys(("set_precedence", selected))
    }
    blast = {}
    for rule in RULES:
        changes = Counter(
            (row.get("physics_domain") or "∅", answers[name][rule] or "∅")
            for name, row in rows.items()
            if row.get("physics_domain") != answers[name][rule]
        )
        blast[rule] = {
            "changed": sum(changes.values()),
            "moves": [
                {"from": old, "to": new, "count": n}
                for (old, new), n in sorted(
                    changes.items(), key=lambda x: (-x[1], x[0])
                )
            ],
        }
        blast[rule]["largest_moves"] = [
            move
            for move in blast[rule]["moves"]
            if move["from"] != "∅" and move["to"] != "∅"
        ][:10]

    def detailed(row: dict, rule: str) -> dict:
        name = row["name"]
        return {
            "name": name,
            "approved_domain": row["label"],
            "answer": answers[name][rule],
            "sources": dict(sorted(sources[name].items())),
        }

    approved_detail = []
    for row in approved:
        name = row["name"]
        approved_detail.append(
            {
                **detailed(row, selected),
                "path_counts": dict(
                    sorted(
                        Counter(
                            d or "unclassified" for d in sources[name].values()
                        ).items()
                    )
                ),
                "answers": answers[name],
            }
        )
    misses = [
        detailed(row, selected)
        for row in validation
        if answers[row["name"]][selected] != row["label"]
    ]
    unfixed = []
    for cohort_name, cohort in (("approved", approved), ("calibration", calibration)):
        for row in cohort:
            name = row["name"]
            if all(answers[name][rule] != row["label"] for rule in RULES):
                item = detailed(row, selected)
                item["cohort"] = cohort_name
                single_override_paths = []
                for path in item["sources"]:
                    revised = item["sources"] | {path: row["label"]}
                    if choose_domain(revised, selected) == row["label"]:
                        single_override_paths.append(path)
                item["one_path_override_suffices"] = bool(single_override_paths)
                item["one_path_override_example"] = (
                    single_override_paths[0] if single_override_paths else None
                )
                item["all_source_domains_conflict"] = bool(item["sources"]) and all(
                    domain != row["label"] for domain in item["sources"].values()
                )
                if not item["sources"]:
                    item["override_assessment"] = (
                        "no DD source binding; a path override has no target"
                    )
                else:
                    conflict = (
                        "approved label conflicts with every DD source classification"
                        if item["all_source_domains_conflict"]
                        else "some DD sources carry the approved domain but lose the vote"
                    )
                    override = (
                        f"one path override at {single_override_paths[0]} fixes "
                        "the selected rule"
                        if single_override_paths
                        else "one path override cannot fix the selected rule"
                    )
                    item["override_assessment"] = f"{conflict}; {override}"
                unfixed.append(item)
    payload = {
        "read_only": True,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "live_names": len(rows),
        "query_seconds": round(query_seconds, 3),
        "source_control": {
            "name": "electron_temperature",
            "contains_core_plasma_physics": True,
            "set_precedence_approved": "9/20",
        },
        "rule_descriptions": {
            "set_precedence": "Highest declared precedence among source domains.",
            "path_plurality": "Most DD source paths, then precedence.",
            "ids_plurality": "Most distinct source IDS, counting each domain once per IDS, then precedence.",
            "ids_then_paths": "Most distinct source IDS, then path count, then precedence.",
        },
        "selection": "Maximum correct on approved plus calibration; approved count and rule order break ties.",
        "selected_rule": selected,
        "decision_caveat": {
            "selected_approved_correct": metrics["approved"][selected]["correct"],
            "control_approved_correct": metrics["approved"]["set_precedence"][
                "correct"
            ],
            "approved_misses_shared_by_all_rules": sum(
                item["cohort"] == "approved" for item in unfixed
            ),
            "validation_gain_over_control": (
                metrics["validation"][selected]["correct"]
                - metrics["validation"]["set_precedence"]["correct"]
            ),
        },
        "accuracy": metrics,
        "approved_names": approved_detail,
        "validation_misses": misses,
        "unfixed_misses": unfixed,
        "blast_radius": blast,
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
    make_figure(metrics, selected)
    render(payload)
    print(
        f"live names: {len(rows)}; query: {query_seconds:.3f}s; approved control: 9/20"
    )
    for cohort, scored in metrics.items():
        print(
            cohort, {rule: f"{m['correct']}/{m['count']}" for rule, m in scored.items()}
        )
    print(f"selected: {selected}; live changes: {blast[selected]['changed']}")
    print(f"validation misses: {len(misses)}; unfixed misses: {len(unfixed)}")
    print(f"wrote {RESULTS}, {FIGURE}, {FRAGMENT}")


if __name__ == "__main__":
    main()

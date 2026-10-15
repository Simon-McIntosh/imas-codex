"""Measure band umbrellas against labelled names and a bounded live census."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_PATH = HERE / "scg-name-domain-rule-measure-run.py"
SPEC = importlib.util.spec_from_file_location("source_domain_rules", BASE_PATH)
assert SPEC and SPEC.loader
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

STEM = Path(__file__).stem.removesuffix("-run")
RESULTS = HERE / f"{STEM}-results.json"
FRAGMENT = HERE / f"{STEM}.html"
FIGURE = ROOT / "docs/figures/sn-catalog-grouping" / STEM / "accuracy.svg"
GUIDANCE = ROOT / "imas_codex/definitions/physics/domains.yaml"
THRESHOLDS = (0.5, 0.6, 2 / 3)
BASE_RULES = ("set_precedence", "ids_then_paths")
UMBRELLAS = {
    "core_physics": "core_plasma_physics",
    "heating_current_drive": "auxiliary_heating",
    "edge": "edge_plasma_physics",
    "diagnostics": "plasma_measurement_diagnostics",
    "operations_engineering": "machine_operations",
    "data": "data_management",
    "general": "general",
}


def rule_name(base_rule: str, threshold: float) -> str:
    return f"{base_rule}_umbrella_{threshold:.3f}"


def guidance_bands() -> tuple[dict[str, str], dict[str, dict]]:
    guidance = yaml.safe_load(GUIDANCE.read_text())
    bands = guidance["precedence"]
    domain_to_band = {
        domain: band["band"] for band in bands for domain in band["domains"]
    }
    assert set(UMBRELLAS) == {band["band"] for band in bands}
    assert set(domain_to_band) == set(base.RANK)
    choices = {}
    for band in bands:
        name = band["band"]
        umbrella = UMBRELLAS[name]
        assert umbrella in band["domains"]
        choices[name] = {
            "domain": umbrella,
            "band_rationale": band["rationale"],
            "umbrella_scope": guidance["domains"][umbrella]["scope"],
        }
    return domain_to_band, choices


def ids_support(sources: dict[str, str | None]) -> tuple[dict[str, set[str]], set[str]]:
    by_domain: dict[str, set[str]] = defaultdict(set)
    all_ids = set()
    for path, domain in sources.items():
        if domain is None:
            continue
        ids = path.removeprefix("dd:").split("/", 1)[0]
        by_domain[domain].add(ids)
        all_ids.add(ids)
    return by_domain, all_ids


def choose_umbrella(
    sources: dict[str, str | None],
    base_rule: str,
    threshold: float,
    domain_to_band: dict[str, str],
) -> str | None:
    """Apply a band umbrella when no domain owns more than the IDS threshold."""
    fallback = base.choose_domain(sources, base_rule)
    by_domain, all_ids = ids_support(sources)
    if not all_ids:
        return fallback
    bands: dict[str, set[str]] = defaultdict(set)
    for domain, ids in by_domain.items():
        bands[domain_to_band[domain]].update(ids)
    if len(bands) > 1:
        if max(len(ids) for ids in bands.values()) / len(all_ids) <= threshold:
            return "general"
        return fallback
    if len(by_domain) > 1:
        if max(len(ids) for ids in by_domain.values()) / len(all_ids) <= threshold:
            return UMBRELLAS[next(iter(bands))]
    return fallback


def rule_answers(
    sources: dict[str, dict[str, str | None]], domain_to_band: dict[str, str]
) -> dict[str, dict[str, str | None]]:
    answers = {}
    for name, paths in sources.items():
        answer = {rule: base.choose_domain(paths, rule) for rule in BASE_RULES}
        answer.update(
            {
                rule_name(rule, threshold): choose_umbrella(
                    paths, rule, threshold, domain_to_band
                )
                for rule in BASE_RULES
                for threshold in THRESHOLDS
            }
        )
        answers[name] = answer
    return answers


def verify_rule_cases(domain_to_band: dict[str, str]) -> None:
    same_band_tie = {
        "dd:core_profiles/temperature": "equilibrium",
        "dd:summary/temperature": "core_plasma_physics",
    }
    mixed_band_tie = {
        "dd:core_profiles/temperature": "equilibrium",
        "dd:bolometer/temperature": "spectroscopy",
    }
    domain_majority = {
        "dd:core_profiles/temperature": "equilibrium",
        "dd:equilibrium/temperature": "equilibrium",
        "dd:summary/temperature": "core_plasma_physics",
    }
    assert (
        choose_umbrella(same_band_tie, "set_precedence", 0.5, domain_to_band)
        == "core_plasma_physics"
    )
    assert (
        choose_umbrella(mixed_band_tie, "ids_then_paths", 0.5, domain_to_band)
        == "general"
    )
    assert (
        choose_umbrella(domain_majority, "set_precedence", 0.5, domain_to_band)
        == "equilibrium"
    )


def changes_from_base(
    cohort: list[dict],
    answers: dict[str, dict[str, str | None]],
    candidate: str,
    control: str,
) -> dict[str, list[dict]]:
    fixes, breaks = [], []
    for row in cohort:
        name, label = row["name"], row["label"]
        before, after = answers[name][control], answers[name][candidate]
        item = {
            "name": name,
            "label": label,
            "base_answer": before,
            "umbrella_answer": after,
        }
        if before != label and after == label:
            fixes.append(item)
        elif before == label and after != label:
            breaks.append(item)
    return {"fixes": fixes, "new_breaks": breaks}


def blast_radius(
    rows: dict[str, dict], answers: dict[str, dict[str, str | None]], rule: str
) -> dict:
    moves = Counter(
        (row.get("physics_domain") or "∅", answers[name][rule] or "∅")
        for name, row in rows.items()
        if row.get("physics_domain") != answers[name][rule]
    )
    return {
        "changed_from_stored": sum(moves.values()),
        "moves": [
            {"from": old, "to": new, "count": count}
            for (old, new), count in sorted(
                moves.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def cross_rule_blast(
    answers: dict[str, dict[str, str | None]], left: str, right: str
) -> dict:
    moves = Counter(
        (row[left] or "∅", row[right] or "∅")
        for row in answers.values()
        if row[left] != row[right]
    )
    return {
        "changed": sum(moves.values()),
        "moves": [
            {"from": old, "to": new, "count": count}
            for (old, new), count in sorted(
                moves.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def make_figure(metrics: dict, selected: str) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    style = Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle"
    if style.exists():
        plt.style.use(style)
    plt.rcParams["svg.hashsalt"] = "domain-band-umbrella"
    shown = (
        *BASE_RULES,
        *(
            rule_name(rule, threshold)
            for rule in BASE_RULES
            for threshold in THRESHOLDS
        ),
    )
    labels = [
        "set precedence",
        "IDS then paths",
        "set + umbrella 0.50",
        "set + umbrella 0.60",
        "set + umbrella 0.67",
        "IDS + umbrella 0.50",
        "IDS + umbrella 0.60",
        "IDS + umbrella 0.67",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 8.4), sharey=True)
    direct_labels = []
    for axis, cohort in zip(
        axes, ("approved", "calibration", "validation"), strict=True
    ):
        for index, rule in enumerate(shown):
            item = metrics[cohort].get(rule)
            if item is None:
                continue
            colour = "#286c83" if rule == selected else "#565d60"
            axis.errorbar(
                item["accuracy"],
                index,
                xerr=[
                    [item["accuracy"] - item["lower"]],
                    [item["upper"] - item["accuracy"]],
                ],
                fmt="o",
                color=colour,
                linewidth=3.0 if rule == selected else 2.6,
                capsize=3,
                markersize=8,
            )
            label = axis.annotate(
                f"{item['correct']}/{item['count']}",
                (item["accuracy"], index),
                xytext=(0, -24),
                textcoords="offset points",
                ha="center",
                fontsize=20,
                color=colour,
            )
            direct_labels.append((label, axis, index))
        axis.set_xlim(0, 1.02)
        axis.set_xticks((0.25, 0.5, 0.75))
        axis.set_xlabel("accuracy")
        axis.set_title(cohort, fontsize=20)
        axis.set_yticks(range(len(shown)))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(shown) - 0.2, -0.4)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.15, top=0.9, wspace=0.12)
    fig.canvas.draw()
    canvas = fig.bbox
    text_boxes = []
    for artist in fig.findobj(match=plt.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(fig.canvas.get_renderer())
        assert canvas.contains(*box.get_points()[0]) and canvas.contains(
            *box.get_points()[1]
        ), artist.get_text()
        for prior, prior_box in text_boxes:
            assert not box.overlaps(prior_box), (prior, artist.get_text())
        text_boxes.append((artist.get_text(), box))
    for label, axis, index in direct_labels:
        box = label.get_window_extent(fig.canvas.get_renderer())
        mark_y = axis.transData.transform((0, index))[1]
        assert box.y1 < mark_y - 2, label.get_text()
        if index + 1 < len(shown):
            next_y = axis.transData.transform((0, index + 1))[1]
            assert box.y0 > next_y + 2, label.get_text()
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)


def metric_text(item: dict | None) -> str:
    if item is None:
        return "not read"
    return f"{item['correct']}/{item['count']} ({item['accuracy']:.3f}; {item['lower']:.3f}–{item['upper']:.3f})"


def render(payload: dict) -> None:
    selected = payload["selected_rule"]
    rules = payload["candidate_rules"]
    lines = [
        f'<section id="{STEM}">',
        "<h2>Band umbrella measurement for name domains</h2>",
        "<p>A read-only, zero-spend census reused the source-path loader, selectors and paired bootstrap scoring of the preceding rule comparison. The base controls reproduced 9/20 approved names, with calibration at 53/73 for set precedence and 56/73 for IDS then paths. The known-present electron-temperature source still carries core_plasma_physics.</p>",
        "<p>A source IDS contributes once to each domain and band it touches. When every classified source is in one band, at least two domains occur, and no domain exceeds the threshold share of distinct IDS, the rule selects that band's umbrella. When sources span bands and no band exceeds the threshold, it selects general. Otherwise it uses the base selector. Thresholds are 0.50, 0.60 and 0.667. A tie at the threshold is not a majority.</p>",
        f"<p>Approved plus calibration accuracy selected <strong>{escape(selected)}</strong> among six umbrella variants before validation was read. Validation was then scored once for that rule and the two base controls. Accuracy intervals are 95% paired-group bootstrap intervals, transformed from the shared AUC statistic.</p>",
        f'<figure><img src="/imas-codex/figures/sn-catalog-grouping/{STEM}/accuracy.svg" alt="Selector accuracy with confidence intervals for the three labelled sets"><figcaption>Correct names over total in approved, calibration and validation sets. Blue marks the chosen rule; untested validation candidates have no mark.</figcaption></figure>',
        "<h3>Accuracy and live blast radius</h3><table><tr><th>Rule</th><th>Approved</th><th>Calibration</th><th>Validation</th><th>Changed from stored</th><th>Changed from base</th></tr>",
    ]
    for rule in rules:
        blast = payload["blast_radius"][rule]
        lines.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in (
                    rule,
                    *(
                        metric_text(payload["accuracy"][cohort].get(rule))
                        for cohort in ("approved", "calibration", "validation")
                    ),
                    blast["changed_from_stored"],
                    payload["against_base"].get(rule, {}).get("changed", "—"),
                )
            )
            + "</tr>"
        )
    paired_base = selected.split("_umbrella_")[0]
    deltas = {
        cohort: payload["accuracy"][cohort][selected]["correct"]
        - payload["accuracy"][cohort][paired_base]["correct"]
        for cohort in ("approved", "calibration", "validation")
    }
    diagnostic_misses = sum(
        item["approved_domain"] == "plasma_measurement_diagnostics"
        for item in payload["unreachable_approved"]
    )
    lines.append(
        "</table><p>Relative to its paired base, the selected rule changes "
        f"approved by {deltas['approved']:+d}/20, calibration by "
        f"{deltas['calibration']:+d}/73, and validation by "
        f"{deltas['validation']:+d}/77. It changes "
        f"{payload['against_base'][selected]['changed']} live answers beyond "
        f"that base. Its selection-set total is "
        f"{payload['selection_counts']['chosen_umbrella']} versus "
        f"{payload['selection_counts']['best_base']} for the best base control. "
        f"{diagnostic_misses} approved diagnostic-geometry "
        "labels remain unreachable because their sources span bands. "
        "The calibration and validation losses do not support adopting this "
        "umbrella selector as the single derivation.</p>"
    )
    lines.append(
        "<h3>Umbrella choices from guidance scopes</h3><table><tr><th>Band</th><th>Umbrella</th><th>Scope and limitation</th></tr>"
    )
    for band, item in payload["umbrellas"].items():
        lines.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(v))}</td>"
                for v in (band, item["domain"], item["umbrella_scope"])
            )
            + "</tr>"
        )
    lines.append(
        "</table><p>These are testable umbrella proxies, not name-level production overrides. The edge, heating, operations and data scopes each cover only part of their band; the measurement tests whether that loss of specificity helps the labelled names.</p>"
    )
    lines.append(
        "<h3>Twenty lead-approved names</h3><table><tr><th>Name</th><th>Approved</th><th>Source domains (paths)</th>"
        + "".join(f"<th>{escape(rule)}</th>" for rule in rules)
        + "</tr>"
    )
    for row in payload["approved_names"]:
        counts = ", ".join(
            f"{domain}: {count}" for domain, count in row["path_counts"].items()
        )
        cells = (
            row["name"],
            row["approved_domain"],
            counts,
            *(row["answers"][rule] or "∅" for rule in rules),
        )
        lines.append(
            "<tr>"
            + "".join(f"<td>{escape(str(value))}</td>" for value in cells)
            + "</tr>"
        )
    lines.append("</table><h3>Fixes and new breaks against each base</h3>")
    for cohort, comparisons in payload["label_changes"].items():
        for control, detail in comparisons.items():
            fixes = ", ".join(item["name"] for item in detail["fixes"]) or "none"
            breaks = ", ".join(item["name"] for item in detail["new_breaks"]) or "none"
            lines.append(
                f"<p>{escape(cohort)} versus {escape(control)}: fixes {escape(fixes)}; new breaks {escape(breaks)}.</p>"
            )
    lines.append("<h3>Approved labels no rule reaches</h3><ul>")
    for row in payload["unreachable_approved"]:
        lines.append(
            f"<li>{escape(row['name'])}: approved {escape(row['approved_domain'])}; source domains {escape(', '.join(row['source_domains']))}; conflicts with every source: {str(row['all_source_domains_conflict']).lower()}</li>"
        )
    lines.append(
        "</ul><p>Full source-path bindings, each rule answer, and all live move counts are in the adjacent results JSON. No graph writes or LLM calls were made.</p></section>"
    )
    FRAGMENT.write_text("\n".join(lines) + "\n")


def main() -> None:
    sheet = json.loads(base.SHEET.read_text())
    audit = json.loads(base.AUDIT.read_text())
    domain_to_band, choices = guidance_bands()
    verify_rule_cases(domain_to_band)
    rows, query_seconds = base.source_rows()
    assert len(rows) == audit["live_names"], (
        "live census differs from known-present audit"
    )
    memo = {}
    sources = {name: base.sources_for(name, rows, memo) for name in rows}
    assert "core_plasma_physics" in sources["electron_temperature"].values()
    answers = rule_answers(sources, domain_to_band)
    approved = sheet["reference"]
    calibration = [row for row in sheet["sample"] if row["split"] == "calibration"]
    validation = [row for row in sheet["sample"] if row["split"] == "validation"]
    assert (len(approved), len(calibration), len(validation)) == (20, 73, 77)
    assert all(row["name"] in rows for row in (*approved, *calibration, *validation))
    candidates = [
        rule_name(rule, threshold) for rule in BASE_RULES for threshold in THRESHOLDS
    ]
    all_rules = [*BASE_RULES, *candidates]
    metrics = {
        "approved": {rule: base.score(approved, answers, rule) for rule in all_rules},
        "calibration": {
            rule: base.score(calibration, answers, rule) for rule in all_rules
        },
    }
    assert [metrics["approved"][rule]["correct"] for rule in BASE_RULES] == [9, 9]
    assert [metrics["calibration"][rule]["correct"] for rule in BASE_RULES] == [53, 56]
    selected = max(
        candidates,
        key=lambda rule: (
            metrics["approved"][rule]["correct"]
            + metrics["calibration"][rule]["correct"],
            metrics["approved"][rule]["correct"],
            -candidates.index(rule),
        ),
    )
    metrics["validation"] = {
        rule: base.score(validation, answers, rule) for rule in (*BASE_RULES, selected)
    }
    blast = {rule: blast_radius(rows, answers, rule) for rule in all_rules}
    against_base = {
        rule: cross_rule_blast(answers, rule.split("_umbrella_")[0], rule)
        for rule in candidates
    }
    comparisons = {
        cohort: {
            control: changes_from_base(labelled, answers, selected, control)
            for control in BASE_RULES
        }
        for cohort, labelled in (
            ("approved", approved),
            ("calibration", calibration),
            ("validation", validation),
        )
    }
    approved_details = []
    unreachable = []
    for row in approved:
        name = row["name"]
        path_domains = sources[name]
        approved_details.append(
            {
                "name": name,
                "approved_domain": row["label"],
                "path_counts": dict(
                    sorted(
                        Counter(
                            d or "unclassified" for d in path_domains.values()
                        ).items()
                    )
                ),
                "sources": dict(sorted(path_domains.items())),
                "answers": answers[name],
            }
        )
        if all(answers[name][rule] != row["label"] for rule in all_rules):
            known = sorted({domain for domain in path_domains.values() if domain})
            unreachable.append(
                {
                    "name": name,
                    "approved_domain": row["label"],
                    "source_domains": known,
                    "all_source_domains_conflict": bool(known)
                    and row["label"] not in known,
                    "sources": dict(sorted(path_domains.items())),
                }
            )
    payload = {
        "read_only": True,
        "graph_writes": 0,
        "llm_calls": 0,
        "live_names": len(rows),
        "known_present_control": {
            "name": "electron_temperature",
            "source_domain": "core_plasma_physics",
            "approved_base_counts": {
                rule: metrics["approved"][rule]["correct"] for rule in BASE_RULES
            },
            "calibration_base_counts": {
                rule: metrics["calibration"][rule]["correct"] for rule in BASE_RULES
            },
        },
        "thresholds": list(THRESHOLDS),
        "umbrellas": choices,
        "selection": "Best of six umbrella variants by approved plus calibration correct; approved correct then declared rule order break ties. Base controls are compared but not candidate variants.",
        "selected_rule": selected,
        "selection_counts": {
            "chosen_umbrella": metrics["approved"][selected]["correct"]
            + metrics["calibration"][selected]["correct"],
            "best_base": max(
                metrics["approved"][rule]["correct"]
                + metrics["calibration"][rule]["correct"]
                for rule in BASE_RULES
            ),
        },
        "candidate_rules": all_rules,
        "accuracy": metrics,
        "approved_names": approved_details,
        "label_changes": comparisons,
        "unreachable_approved": unreachable,
        "blast_radius": blast,
        "against_base": against_base,
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
    make_figure(metrics, selected)
    render(payload)
    print(f"live={len(rows)} query_seconds={query_seconds:.3f} source_control=present")
    print("rule_controls=passed")
    for cohort, scored in metrics.items():
        print(
            cohort,
            {
                rule: f"{item['correct']}/{item['count']}"
                for rule, item in scored.items()
            },
        )
    print(f"selected={selected} unreachable_approved={len(unreachable)}")
    print(f"wrote {RESULTS}, {FIGURE}, {FRAGMENT}")


if __name__ == "__main__":
    main()

"""Measure an IDS-first domain rule, its tie-break hierarchy and its Jev reach.

Read-only: no graph write and no LLM or Jev call. Reuses the source-path
loader, the control selectors and the paired-bootstrap scorer of the preceding
rule comparison, and the guidance file for the precedence bands.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import yaml

from imas_codex.graph.client import GraphClient

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
FIGURE = ROOT / "docs/figures/sn-catalog-grouping" / STEM / "ids_domain_mix.svg"
GUIDANCE = ROOT / "imas_codex/definitions/physics/domains.yaml"
IDS_MAP = ROOT / "imas_codex/definitions/physics/ids_domains.json"
RANK = base.RANK
CONTROLS = ("set_precedence", "ids_then_paths")
VARIANTS = ("ids_first_tier_then_band", "ids_first_band_then_tier")
ALL_RULES = (*CONTROLS, *VARIANTS, *(f"{v}_corrected" for v in VARIANTS))
CORRECTED = {rule for rule in ALL_RULES if rule.endswith("_corrected")}
WEAK_SHARE = 0.5


def load_ids_map() -> dict[str, str]:
    return dict(json.loads(IDS_MAP.read_text())["ids_domain_mappings"])


def load_bands() -> tuple[dict[str, int], list[dict], dict[str, dict]]:
    guidance = yaml.safe_load(GUIDANCE.read_text())
    bands = guidance["precedence"]  # seven bands, every domain exactly once
    band_of = {
        domain: index for index, band in enumerate(bands) for domain in band["domains"]
    }
    assert set(band_of) == set(RANK), "the precedence bands must cover every domain"
    return band_of, bands, guidance["domains"]


BAND_OF, BANDS, SCOPES = load_bands()


def ids_of(path: str) -> str:
    return path.removeprefix("dd:").split("/", 1)[0]


def ids_tier(ids: str, ids_map: dict[str, str]) -> int:
    """Zero for an IDS the guidance puts in the core-physics band, else one.

    A plasma-state IDS therefore outranks a diagnostic IDS when two of their
    mapped domains tie, which is the shape a shared quantity such as a
    temperature or a current presents.
    """
    mapped = ids_map.get(ids)
    return 0 if mapped is not None and BAND_OF[mapped] == 0 else 1


def ids_support(
    sources: dict[str, str | None], ids_map: dict[str, str]
) -> tuple[Counter, dict[str, set[str]]]:
    """Count, per domain, the source IDSs that map to it or carry it."""
    carried: dict[str, set[str]] = defaultdict(set)
    for path, domain in sources.items():
        if domain:
            carried[ids_of(path)].add(domain)
    support: Counter = Counter()
    supporters: dict[str, set[str]] = defaultdict(set)
    for ids, domains in carried.items():
        vote = set(domains)
        mapped = ids_map.get(ids)
        if mapped:
            vote.add(mapped)
        for domain in vote:
            support[domain] += 1
            supporters[domain].add(ids)
    return support, supporters


def choose_ids_first(
    sources: dict[str, str | None], rule: str, ids_map: dict[str, str]
) -> str | None:
    """Argmax of IDS-domain support, ties broken by the declared hierarchy."""
    support, supporters = ids_support(sources, ids_map)
    if not support:
        return None
    best = max(support.values())
    top = [domain for domain in support if support[domain] == best]
    if len(top) == 1:
        return top[0]
    band_first = "band_then_tier" in rule

    def key(domain: str) -> tuple:
        tier = min(ids_tier(ids, ids_map) for ids in supporters[domain])
        lead = (BAND_OF[domain], tier) if band_first else (tier, BAND_OF[domain])
        return (*lead, RANK[domain], domain)

    return min(top, key=key)


def choose(
    sources: dict[str, str | None], rule: str, ids_map: dict[str, str]
) -> str | None:
    if rule in CONTROLS:
        return base.choose_domain(sources, rule)
    assert rule in VARIANTS or rule.removesuffix("_corrected") in VARIANTS, rule
    return choose_ids_first(sources, rule, ids_map)


def per_ids_mix() -> tuple[dict[str, dict], float]:
    """Read every DD path's stored domain, grouped by its IDS."""
    query = (
        "MATCH (n:IMASNode) "
        "RETURN split(n.id,'/')[0] AS ids, n.physics_domain AS domain, "
        "coalesce(n.domain_source,'none') AS source, count(*) AS paths"
    )
    started = perf_counter()
    with GraphClient() as client:
        rows = client.query(query)
    elapsed = perf_counter() - started
    assert rows, "known-present DD census returned nothing"
    mix: dict[str, dict] = {}
    for row in rows:
        entry = mix.setdefault(
            row["ids"], {"domains": defaultdict(int), "domain_source": defaultdict(int)}
        )
        entry["domains"][row["domain"] or "unclassified"] += row["paths"]
        entry["domain_source"][row["source"]] += row["paths"]
    return mix, elapsed


def summarise_mix(mix: dict[str, dict], ids_map: dict[str, str]) -> dict:
    entries = {}
    for ids, entry in sorted(mix.items()):
        domains = dict(sorted(entry["domains"].items(), key=lambda kv: (-kv[1], kv[0])))
        total = sum(domains.values())
        modal, modal_paths = next(iter(domains.items()))
        mapped = ids_map.get(ids)
        entries[ids] = {
            "mapped_domain": mapped,
            "total_paths": total,
            "domains": domains,
            "domain_source": dict(sorted(entry["domain_source"].items())),
            "modal_domain": modal,
            "modal_share": modal_paths / total,
            "mapped_share": (domains.get(mapped, 0) / total) if mapped else 0.0,
            "distinct_domains": len(domains),
            "agrees": mapped == modal,
        }
    mostly = [i for i, e in entries.items() if e["modal_share"] >= 0.5]
    contradicted = [
        i
        for i, e in entries.items()
        if e["mapped_domain"] is not None
        and e["mapped_share"] < 0.5
        and e["mapped_domain"] != e["modal_domain"]
    ]
    return {
        "ids_count": len(entries),
        "path_total": sum(e["total_paths"] for e in entries.values()),
        "single_domain_ids": sorted(
            i for i, e in entries.items() if e["distinct_domains"] == 1
        ),
        "mostly_one_domain_ids": sorted(mostly),
        "mixed_ids": sorted(set(entries) - set(mostly)),
        "map_disagrees_ids": sorted(i for i, e in entries.items() if not e["agrees"]),
        "map_contradicted_ids": sorted(contradicted),
        "entries": entries,
    }


def corrected_map(mix: dict[str, dict], ids_map: dict[str, str]) -> dict[str, str]:
    """Replace a mapped domain its own paths contradict with their modal domain."""
    proposed = dict(ids_map)
    for ids, entry in mix.items():
        domains = {d: n for d, n in entry["domains"].items() if d != "unclassified"}
        total = sum(domains.values())
        if not total:
            continue
        modal = max(domains.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if modal and modal != ids_map.get(ids) and domains[modal] / total >= 0.5:
            proposed[ids] = modal
    return proposed


def rule_answers(
    sources: dict[str, dict[str, str | None]],
    ids_map: dict[str, str],
    proposed: dict[str, str],
) -> dict[str, dict[str, str | None]]:
    return {
        name: {
            rule: choose(paths, rule, proposed if rule in CORRECTED else ids_map)
            for rule in ALL_RULES
        }
        for name, paths in sources.items()
    }


def jev_reach(
    sources: dict[str, dict[str, str | None]], ids_map: dict[str, str]
) -> dict:
    """Names the rule leaves tied, or with no strict IDS majority."""
    tied, weak = {}, {}
    for name, paths in sources.items():
        support, _ = ids_support(paths, ids_map)
        if not support:
            continue
        best = max(support.values())
        top = [domain for domain in support if support[domain] == best]
        distinct_ids = len({ids_of(path) for path, domain in paths.items() if domain})
        if len(top) > 1:
            tied[name] = sorted(top)
        elif distinct_ids and best / distinct_ids < WEAK_SHARE:
            weak[name] = [top[0]]
    return {"tied": dict(sorted(tied.items())), "weak": dict(sorted(weak.items()))}


def blast(
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
                moves.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ],
    }


def make_figure(summary: dict) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    style = Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle"
    if style.exists():
        plt.style.use(style)
    plt.rcParams["svg.hashsalt"] = "ids-domain-mix"
    entries = summary["entries"]
    ids = sorted(entries, key=lambda i: (entries[i]["mapped_share"], i))
    y = list(range(len(ids)))
    agree = "#286c83"
    disagree = "#a8632a"
    fig, axes = plt.subplots(2, 1, figsize=(14, 34), sharex=False)
    shares = [entries[i]["mapped_share"] for i in ids]
    axis = axes[0]
    axis.scatter(
        shares, y, c=[agree if s >= 0.5 else disagree for s in shares], s=24, zorder=3
    )
    axis.axvline(0.5, color="#6b7075", linewidth=1.2, linestyle=":", zorder=1)
    axis.set_xlim(-0.02, 1.02)
    axis.set_xlabel("share of the IDS's paths that carry its mapped domain")
    axis.set_yticks(y, ids, fontsize=11)
    axis.set_ylim(len(ids) - 0.4, -1.2)
    axis.set_title("paths agree with the IDS map", fontsize=14)
    axis.annotate(
        "half the paths",
        (0.5, -0.9),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=11,
        color="#6b7075",
    )
    axis = axes[1]
    counts = [entries[i]["distinct_domains"] for i in ids]
    axis.scatter(counts, y, c=agree, s=24, zorder=3)
    axis.set_xlabel("distinct domains among the IDS's paths")
    axis.set_xticks(range(1, max(counts) + 1))
    axis.set_ylim(len(ids) - 0.4, -1.2)
    axis.set_title("paths concentrate on one domain", fontsize=14)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    fig.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=agree,
                label="at or above half",
            ),
            plt.Line2D(
                [], [], marker="o", linestyle="none", color=disagree, label="below half"
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.9995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.23, right=0.97, bottom=0.055, top=0.955, hspace=0.12)
    fig.canvas.draw()
    canvas = fig.bbox
    renderer = fig.canvas.get_renderer()
    # Text outside the canvas is caught by the figure's own tight bbox, which counts
    # every drawn artist; tick labels that no axes draws carry coordinates outside the
    # canvas but are never rendered, so they are excluded from the overlap pass below.
    tight = fig.get_tightbbox(renderer)
    assert canvas.contains(*tight.get_points()[0]) and canvas.contains(
        *tight.get_points()[1]
    ), (
        "text escapes the canvas",
        tight.bounds,
        canvas.bounds,
    )
    boxes: list[tuple[str, object]] = []
    for artist in fig.findobj(match=plt.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if not all(canvas.contains(*corner) for corner in box.get_points()):
            continue
        for prior, prior_box in boxes:
            assert not box.overlaps(prior_box), (prior, artist.get_text())
        boxes.append((artist.get_text(), box))
    fig.savefig(FIGURE, format="svg", metadata={"Date": "2026-10-16"})
    plt.close(fig)


def metric_text(item: dict | None) -> str:
    if item is None:
        return "not read"
    return (
        f"{item['correct']}/{item['count']} "
        f"({item['accuracy']:.3f}; {item['lower']:.3f}–{item['upper']:.3f})"
    )


COHORTS = ("approved", "calibration", "validation")


def cohort_row(payload: dict, rule: str) -> str:
    parts = []
    for cohort in COHORTS:
        item = payload["accuracy"].get(cohort, {}).get(rule)
        parts.append(
            f"{cohort} {item['correct']} of {item['count']}"
            if item
            else f"{cohort} not read"
        )
    return ", ".join(parts)


def render(payload: dict) -> None:
    selected = payload["selected_rule"]
    mix = payload["ids_mix_summary"]
    reach = payload["jev_reach"]
    changes = payload["ids_map_summary"]["proposed_changes"]
    proposed_rows = (
        "".join(
            f"<tr><td><code>{escape(ids)}</code></td>"
            f"<td>{escape(payload['ids_map'][ids])}</td><td>{escape(new)}</td>"
            f"<td>the IDS's own paths take {escape(new)} for at least half its "
            f"{payload['ids_mix_summary']['entries'][ids]['total_paths']} paths</td></tr>"
            for ids, new in sorted(changes.items())
        )
        or "<tr><td colspan='4'>no entry changes</td></tr>"
    )
    lines = [
        f'<section id="{STEM}">',
        "<h2>Physics domains from IDSs first, with hierarchy and Jev tie-breaks</h2>",
        "<p>Read-only census: no graph write, no LLM call and no Jev call. The "
        "source-path loader, the two control selectors and the paired-bootstrap "
        "scorer are the preceding rule comparison's; this node adds the IDS map "
        "as a prior, a per-IDS path mix, and a designed tie-break.</p>",
        "<h3>1. The use case</h3>",
        "<p>A domain is read by five kinds of consumer.</p><ul>"
        "<li><strong>Graph build and classification.</strong> "
        "<code>imas_codex/graph/build_dd.py:3495,3528,3583</code> sets a path's "
        "domain from its IDS default on the rebuild, and preserves a classified "
        "value. <code>imas_codex/graph/dd_domain_classifier.py:1589,1601</code> "
        "classifies the paths. <code>imas_codex/graph/dd_graph_ops.py:125</code> "
        "coalesces on upsert. "
        "<code>imas_codex/standard_names/graph_ops.py:3387-3409</code> merges one "
        "<code>HAS_PHYSICS_DOMAIN</code> edge per <code>source_domains</code> entry.</li>"
        "<li><strong>Catalog export and grouping.</strong> "
        "<code>imas_codex/standard_names/export.py:2164-2189</code> groups entries "
        "into per-domain buckets and records <code>domains_included</code>; "
        "<code>:844-864,3299-3327</code> excludes an entry with no domain. "
        "<code>imas_codex/standard_names/batching.py:11</code> keys a name-only "
        "batch by domain and unit. "
        "<code>imas_codex/standard_names/consolidation.py:49-117</code> merges a "
        "group's domains into one scalar plus a union. "
        "<code>imas_codex/standard_names/families.py:665</code> prints the domain "
        "as a family column.</li>"
        "<li><strong>MCP search filters.</strong> "
        "<code>imas_codex/llm/server.py:2414,2508,2559,2611,2792,3184,3302</code> "
        "take a domain filter per name and per signal; "
        "<code>imas_codex/llm/sn_tools.py:345-414</code> matches a name's scalar "
        "<em>or</em> any value in <code>source_domains</code>. "
        "<code>imas_codex/llm/search_tools.py:327,751-785,1012,1750</code> "
        "post-filters signals, docs and code. "
        "<code>imas_codex/graph/domain_queries.py:85,98,459</code> holds the "
        "Cypher. <code>imas_codex/llm/search_formatters.py:649</code> renders "
        "the domain into result text.</li>"
        "<li><strong>Documentation and review prompts.</strong> "
        "<code>imas_codex/llm/prompt_loader.py:576-629,928</code> renders the "
        "domain vocabulary and the per-category scope into every prompt "
        "declaring <code>physics_domains</code>; "
        "<code>imas_codex/standard_names/review/consolidation.py:189-323</code> "
        "groups review items by domain; "
        "<code>imas_codex/standard_names/harmonize.py:519-653</code> attributes "
        "domain-scoped lint findings to a family.</li>"
        "<li><strong>Name derivation.</strong> "
        "<code>imas_codex/standard_names/domain_ranking.py:19-29</code>, "
        "<code>imas_codex/standard_names/domain_priority.py:10-29</code>, and "
        "<code>imas_codex/standard_names/enrichment.py:78-117</code>.</li></ul>",
        "<p>Three properties follow, and the design is judged against them. A "
        "name needs <em>exactly one</em> scalar domain, because a filter has one "
        "value to match and export opens one bucket; it may carry several "
        "<code>source_domains</code> so it stays discoverable from every domain "
        "it draws on. The domain must be <em>stable</em>: the precedence is a "
        "tie-break read independently by the classifier and by name selection, "
        "so a derivation that moved with cluster relevance could never be "
        "audited. And it must be <em>human-browsable</em>: the scope text is "
        "rendered into prompts a person reads, and the chat category list is "
        "ordered by the same precedence.</p>",
        "<h3>2. Where a domain comes from today</h3>",
        f"<p>The IDS map <code>imas_codex/definitions/physics/ids_domains.json</code> "
        f"holds {payload['ids_map_summary']['entries']} entries, one domain each, "
        f"over {payload['ids_map_summary']['domains']} distinct domains, created "
        f"{escape(payload['ids_map_summary']['generated'][:10])}. "
        "<code>core_profiles</code> maps to <code>transport</code> and "
        "<code>summary</code> to <code>general</code>; both are entries the paths "
        "below contradict.</p>",
        "<p>The build reads that map through "
        "<code>imas_codex/core/physics_categorization.py:23-42</code> and applies "
        "it at <code>build_dd.py:3583-3588</code>, preserving an existing value "
        "when <code>domain_source</code> is set. The classifier "
        "(<code>dd_domain_classifier.py</code>) decides in four tiers: "
        "infrastructure metadata to null (line 321), a Jev two-level "
        "<code>decide_batch</code> tree (line 1240, question rendering 1009-1025) "
        "with a chat fallback (line 794), inheritance from the nearest classified "
        "ancestor (line 369), and "
        "<code>path_domain_overrides.json</code> (line 37, applied at line 63). "
        "Jev is reached only through <code>_safe_decide_batch</code> "
        "(lines 1067-1083), called at lines 1125 and 1152; there is no "
        "<code>judge_rows</code> and no <code>rerank_pool</code> in the file. "
        "The tree's state (lines 1009-1025) carries the path, its IDS, "
        "description, documentation, unit, parent and siblings, and never the "
        "path's own stored domain. The name derivation is "
        "<code>domain_ranking.select_primary_domain</code> (line 19), whose rank "
        "is <code>domains.yaml</code>'s precedence read through "
        "<code>load_domain_precedence</code> — not a graph cluster index.</p>",
        "<p><strong>So Jev is used now at exactly one place</strong>: the DD path "
        "classifier's first tier. Measured over the paths, the current "
        "<code>domain_source</code> histogram is "
        + ", ".join(
            f"{escape(str(source))} {count:,}"
            for source, count in payload["domain_source_histogram"].items()
        )
        + ". Jev wrote 4,480 paths and the chat fallback 7,551; the rest is "
        "inheritance and metadata. Nothing writes a <em>name's</em> domain "
        "through a Jev judgment today.</p>",
        "<p><strong>The two stored findings both hold as fixed at HEAD.</strong> "
        "The tree read its answer at the wrong level: "
        "<code>result[0]</code> was read as the answer instead of "
        "<code>answers[DOMAIN_QUESTION]</code>, so the second-level question was "
        "never asked and every path fell through to chat. The answer is now "
        "unwrapped through its question name by <code>_answer_for</code> "
        "(lines 1028-1041), used at line 1164 for the domain level and line 1178 "
        "for the category level. The chat renderer dropped the documentation "
        "field: <code>_format_batch_user_prompt</code> (lines 879-895) now emits "
        "<code>documentation</code> at line 886. The candidate-domain list is a "
        "separate render (<code>llm/prompts/imas/domain_classifier.md</code>, "
        "lines 34-48, via <code>_domain_option_text</code> at 931-946) and "
        "renders each domain's description, scope, includes and excludes.</p>",
        "<h3>3. How the IDSs group</h3>",
        f"<p>Over {mix['path_total']:,} DD paths in {mix['ids_count']} IDSs, "
        f"{len(mix['single_domain_ids'])} IDSs carry a single domain, "
        f"{len(mix['mostly_one_domain_ids'])} are mostly one domain and "
        f"{len(mix['mixed_ids'])} are mixed. "
        f"{len(mix['map_disagrees_ids'])} IDSs have a modal domain other than "
        f"their map's, and {len(mix['map_contradicted_ids'])} are contradicted "
        "outright, under half their paths carrying the mapped domain.</p>",
        f'<figure><img src="/imas-codex/figures/sn-catalog-grouping/{STEM}/'
        'ids_domain_mix.svg" alt="Per-IDS share of paths on its mapped domain and '
        'distinct domain count"><figcaption>One row per IDS. Above: the share of '
        "the IDS's paths that carry the IDS's mapped domain. Below: how many "
        "distinct domains its paths carry. Blue agrees with the map, ochre does "
        "not.</figcaption></figure>",
        "<h3>4. The IDS-first rule, measured</h3>",
        "<p>A source IDS votes once for its mapped domain and once for each "
        "domain its own source paths carry; the rule takes the domain with the "
        "most IDS votes. A tie is broken by the declared hierarchy: the "
        "core-physics IDS tier, then the precedence band of the guidance file, "
        "then the rank, then the domain name. Tier 0 is an IDS whose mapped "
        "domain sits in the guidance's core-physics band; every other IDS is "
        "tier 1, so a plasma-state IDS outranks a diagnostic IDS on a shared "
        "quantity. The reversed variant puts the band first and measures the "
        "lead's ordering question directly.</p>",
        f"<p>Approved plus calibration selected <strong>{escape(selected)}</strong> "
        "before validation was read. Validation was then read once for it and "
        "the two controls.</p>",
        "<p>Correct answers, written as correct out of the cohort size, in the "
        "order approved, calibration, validation: "
        + "; ".join(f"{escape(rule)} {cohort_row(payload, rule)}" for rule in ALL_RULES)
        + ".</p>",
        "<p>Set precedence reproduces the figures the precedence rule was "
        "measured at before, so the harness is the earlier measurement's. The "
        "IDS-first rule does not beat the controls everywhere, and the "
        "selection above is not evidence that it does. On the approved set the "
        "corrected variant ties both controls and the uncorrected variant loses "
        "a name; on calibration the corrected variant sits one name behind the "
        "IDS-then-paths control; it leads on validation alone, by one name, and "
        "one name on 77 lies inside the paired-bootstrap interval. What the IDS "
        "map buys is therefore not accuracy over a plurality of source paths: "
        "it is a stated prior, a derivation whose tie-break can be written down "
        "and audited, and a tie set small enough for a Jev step to settle.</p>",
        "<h3>5. The Jev tie-break, designed not run</h3>",
        f"<p>{reach['tied_count']} live names reach a tie and "
        f"{reach['weak_count']} more reach the rule without a strict IDS "
        f"majority, so <strong>{reach['total']} of {payload['live_names']}</strong> "
        f"live names ({reach['reaching_share_of_live']:.1%}) would reach it, "
        "against "
        + ", ".join(f"{n} in {c}" for c, n in reach["by_cohort"].items())
        + ". The primitive is a TypeSafe <em>Choice</em>; the question is "
        "authored by code and the model supplies the judgment, exactly as the "
        "hierarchical-classification cookbook decomposes it — the code owns "
        "the traversal and the model picks among the siblings at the level "
        "reached. Its state carries the name, description and unit, the source "
        "IDSs and their mapped domains, and the source paths with their stored "
        "domains. Its criteria are the tied candidate domains only, each keyed "
        "to the guidance scope text so the model sees what each domain covers "
        "and excludes. Its probability combines with the IDS prior as a "
        "posterior over the tied set: the Choice distribution re-weighted by the "
        "IDS support share and renormalised, so the model breaks the tie rather "
        "than choosing from nothing. Read the answer only when its confidence "
        "clears 0.5, the docs' floor for genuine uncertainty; below it, the "
        "precedence rank decides and the case is logged for a ruling.</p>",
        "<p>A second Jev step the cookbook's shape suggests is re-deciding the "
        f"<em>IDS map itself</em>: one Choice per IDS over the "
        f"{payload['physics_domain_count']} domains, with the IDS's name and a "
        "sample of its paths as state and the domain scope text as criteria. "
        "That is "
        f"{payload['ids_map_summary']['entries']} calls, one per IDS, and it "
        "would supply the corrected map this node instead derives by rule.</p>",
        f"<p>Cost estimate. {reach['total']} tie-break calls for the live "
        "catalogue, plus 87 for the IDS map, is 0 spend at rest and one call "
        "per name at repair time. Each call's state is the name, its description "
        "and its source bindings, roughly 400 to 700 tokens, and its criteria "
        "are the tied domains' scope text, roughly 30 to 60 tokens each. The "
        "pages read carry no price table for Jev, so the estimate is stated in "
        "calls and tokens rather than in currency: "
        f"{reach['total']} calls at about 1,000 input tokens each, under a "
        "million tokens for the whole catalogue, and 87 calls for the map.</p>",
        "<p>Docs read for this design: the installed skill "
        "<code>typesafe-ai/SKILL.md</code>; live "
        "<code>docs.typesafe.ai/llms.txt</code>; the "
        "<code>hierarchical_classification</code> cookbook; "
        "<code>primitives/choice.md</code>; <code>confidence.md</code>.</p>",
        "<h3>6. Recommendation</h3>",
        f"<p>{escape(payload['recommendation']['design'])}</p>",
        "<p>The three designs on the same three sets. Set precedence answers "
        f"{cohort_row(payload, 'set_precedence')}; the IDS-first rule "
        f"{cohort_row(payload, selected)}. The umbrella measure run beside this "
        "node reached at best 12 of 20 approved, 51 of 73 calibration and 52 of "
        "77 validation across its thresholds, and 11, 53 and 52 for the variant "
        "it selected — its fragment is "
        "<code>sn-catalog-grouping/scg-name-domain-umbrella-measure.html</code>, "
        "and this node does not re-measure it. So the umbrella rule leads on "
        "the approved set, where an umbrella is what the lead approved, and no "
        "rule separates from the controls on calibration; only validation "
        "favours the IDS-first rule, by one name. Adopting it and the corrected "
        f"map would change "
        f"{payload['blast_radius'][selected]['changed_from_stored']:,} live "
        "names against the stored values, the same order as the set-precedence "
        f"rule's {payload['blast_radius']['set_precedence']['changed_from_stored']:,}, "
        "so the cost of the choice is a repair pass either way.</p>",
        "<p>Two things only a ruling settles. Whether the corrected map or the "
        "map as stored is adopted: the sets do not separate them, and the 23 "
        "entries the paths contradict include the IDSs whose every source path "
        "carries the mapped domain's own child, as <code>core_profiles</code> "
        "does for <code>core_plasma_physics</code>. Whether a diagnostic IDS "
        "outranking no plasma-state IDS is intended: the tier puts a "
        "plasma-state IDS first on a shared quantity, and the reversed variant "
        "measuring the band first is what the lead's ordering question asks.</p>",
        "<h3>Proposed IDS-map corrections</h3>",
        "<table><tr><th>IDS</th><th>Mapped domain</th><th>Proposed</th>"
        "<th>Evidence</th></tr>",
        proposed_rows,
        "</table>",
        "<p>All rule answers, per-IDS mixes, move counts and the tie-break "
        "design are in the adjacent results JSON.</p>",
        "</section>",
    ]
    FRAGMENT.write_text("\n".join(lines) + "\n")


def main() -> None:
    sheet = json.loads(base.SHEET.read_text())
    audit = json.loads(base.AUDIT.read_text())
    ids_map = load_ids_map()
    assert len(ids_map) == 87, len(ids_map)
    mix, mix_seconds = per_ids_mix()
    proposed = corrected_map(mix, ids_map)
    summary = summarise_mix(mix, ids_map)
    source_histogram = Counter()
    for entry in mix.values():
        source_histogram.update(entry["domain_source"])
    assert summary["path_total"] == 61366, summary["path_total"]

    rows, query_seconds = base.source_rows()
    assert len(rows) == audit["live_names"], "live-name read must see the known census"
    memo: dict[str, dict[str, str | None]] = {}
    sources = {name: base.sources_for(name, rows, memo) for name in rows}
    assert "core_plasma_physics" in sources["electron_temperature"].values(), (
        "known-present source-domain control failed"
    )
    assert "core_profiles" in {
        ids_of(path) for path in sources["electron_temperature"]
    }, "known-present IDS-mix control failed"

    answers = rule_answers(sources, ids_map, proposed)
    approved = sheet["reference"]
    calibration = [row for row in sheet["sample"] if row["split"] == "calibration"]
    validation = [row for row in sheet["sample"] if row["split"] == "validation"]
    assert (len(approved), len(calibration), len(validation)) == (20, 73, 77)
    for cohort in (approved, calibration, validation):
        missing = [row["name"] for row in cohort if row["name"] not in rows]
        assert not missing, f"labelled names absent from live census: {missing}"

    metrics = {
        "approved": {rule: base.score(approved, answers, rule) for rule in ALL_RULES},
        "calibration": {
            rule: base.score(calibration, answers, rule) for rule in ALL_RULES
        },
    }
    assert [metrics["approved"][rule]["correct"] for rule in CONTROLS] == [9, 9], (
        "control arms no longer reproduce the recorded 9/20"
    )
    assert [metrics["calibration"][rule]["correct"] for rule in CONTROLS] == [53, 56], (
        "control arms no longer reproduce the recorded calibration counts"
    )
    selected = max(
        VARIANTS + tuple(f"{v}_corrected" for v in VARIANTS),
        key=lambda rule: (
            metrics["approved"][rule]["correct"]
            + metrics["calibration"][rule]["correct"],
            metrics["approved"][rule]["correct"],
            -VARIANTS.index(rule.removesuffix("_corrected")),
            rule.endswith("_corrected"),
        ),
    )
    metrics["validation"] = {
        rule: base.score(validation, answers, rule)
        for rule in dict.fromkeys((*CONTROLS, selected))
    }

    blast_radius = {rule: blast(rows, answers, rule) for rule in ALL_RULES}
    reach = jev_reach(sources, ids_map)
    cohort_of = {row["name"]: "approved" for row in approved}
    cohort_of.update({row["name"]: "calibration" for row in calibration})
    cohort_of.update({row["name"]: "validation" for row in validation})
    reaching = set(reach["tied"]) | set(reach["weak"])
    reach_counts = Counter(cohort_of.get(name, "unlabelled") for name in reaching)
    selected_map = proposed if selected in CORRECTED else ids_map
    reach_selected = jev_reach(sources, selected_map)
    reaching_selected = set(reach_selected["tied"]) | set(reach_selected["weak"])

    payload = {
        "read_only": True,
        "graph_writes": 0,
        "llm_calls": 0,
        "jev_calls": 0,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "live_names": len(rows),
        "query_seconds": round(query_seconds, 3),
        "ids_mix_seconds": round(mix_seconds, 3),
        "physics_domain_count": len(RANK),
        "known_present_controls": {
            "live_names_expected": audit["live_names"],
            "live_names_read": len(rows),
            "electron_temperature_has_core_plasma_physics": True,
            "dd_path_total": summary["path_total"],
            "dd_path_total_expected": 61366,
        },
        "ids_map_summary": {
            "entries": len(ids_map),
            "domains": len(set(ids_map.values())),
            "generated": json.loads(IDS_MAP.read_text())["metadata"]["created"],
            "proposed_changes": {
                ids: d for ids, d in proposed.items() if d != ids_map[ids]
            },
        },
        "ids_map": dict(sorted(ids_map.items())),
        "proposed_ids_map": dict(sorted(proposed.items())),
        "domain_source_histogram": dict(
            sorted(source_histogram.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "ids_mix_summary": summary,
        "rule_definitions": {
            "set_precedence": "Highest declared precedence among source domains (control).",
            "ids_then_paths": "Most distinct source IDS, then path count, then precedence (control).",
            "ids_first_tier_then_band": "Most source IDSs voting for a domain; ties by core-physics IDS tier, then precedence band, then rank. Current IDS map.",
            "ids_first_band_then_tier": "Same support; ties by precedence band, then IDS tier, then rank. Current IDS map.",
            "ids_first_tier_then_band_corrected": "Tier-then-band with the proposed corrected IDS map.",
            "ids_first_band_then_tier_corrected": "Band-then-tier with the proposed corrected IDS map.",
        },
        "support_definition": "A source IDS votes once for its mapped domain and once for each domain its own source paths carry.",
        "ids_tier_definition": "Tier 0: the IDS's mapped domain sits in the guidance's core-physics band. Tier 1: every other IDS.",
        "corrected_map_rule": "An IDS whose modal path domain differs from its map and covers at least half its paths takes the modal domain.",
        "selection": "Maximum correct on approved plus calibration among the four IDS-first variants; approved count, declared variant order and the corrected-map flag break ties in that order. Controls are compared but are not candidates.",
        "selected_rule": selected,
        "accuracy": metrics,
        "blast_radius": blast_radius,
        "jev_reach": {
            "weak_share_threshold": WEAK_SHARE,
            "tied_count": len(reach["tied"]),
            "weak_count": len(reach["weak"]),
            "total": len(reaching),
            "reaching_share_of_live": len(reaching) / len(rows),
            "by_cohort": dict(sorted(reach_counts.items())),
            "under_selected_map_total": len(reaching_selected),
            "tied_examples": dict(sorted(reach["tied"].items())[:12]),
            "weak_examples": dict(sorted(reach["weak"].items())[:12]),
        },
        "recommendation": {
            "design": (
                "Adopt the IDS-first rule with the corrected IDS map as the one "
                "derivation of a name's scalar domain: candidate domains are the "
                "source IDSs' mapped domains and the source paths' stored "
                "domains, the support vote is the IDS count, and the tie-break "
                "is the declared hierarchy above. Keep the IDS map as the prior "
                "it is and repair only the entries its own paths contradict, "
                "rather than re-deriving it. Reach for the designed Jev "
                "tie-break only on the names the rule leaves tied."
            ),
        },
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
    make_figure(summary)
    render(payload)
    print(f"live={len(rows)} ids={len(ids_map)} paths={summary['path_total']}")
    print(
        "ids-mix: "
        f"single={len(summary['single_domain_ids'])} "
        f"mostly={len(summary['mostly_one_domain_ids'])} "
        f"mixed={len(summary['mixed_ids'])} "
        f"map_disagrees={len(summary['map_disagrees_ids'])} "
        f"map_contradicted={len(summary['map_contradicted_ids'])}"
    )
    for cohort, scored in metrics.items():
        print(
            cohort, {rule: f"{m['correct']}/{m['count']}" for rule, m in scored.items()}
        )
    print(f"selected={selected} jev_reach={len(reaching)}")
    print(f"wrote {RESULTS}, {FIGURE}, {FRAGMENT}")


if __name__ == "__main__":
    main()

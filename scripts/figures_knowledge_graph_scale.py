"""Draw the scale and structure of the knowledge graph as one slide-sized figure.

Each system in the graph is drawn from a real snowball sample of its nodes and
the relationships among them. TCV, JET and JT-60SA split into code, wiki and
signal clusters; the IMAS Data Dictionary, the Standard Names catalog and ITER
stay whole. Every cluster's area is in proportion to its real node count and
every cluster is drawn at one dot density, so a dot stands for about the same
number of nodes everywhere. Graphviz sfdp lays each cluster out from its own
topology, so its hubs, trees and dense cores are the graph's. Grey lines are
sampled relationships that cross systems. Each system is named with its node
count, and text is set in Arial, written into the SVG as text so a slide renders
it in the slide's own font. The top of the canvas is left clear for the slide
title.

    uv run python scripts/figures_knowledge_graph_scale.py --query    # counts + sample
    uv run python scripts/figures_knowledge_graph_scale.py            # draw from cache

The query pass writes ``graph-stats.json`` beside the figure and the sampled
subgraph to ``SAMPLE`` so a redraw never needs the graph.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures" / "knowledge-graph-scale"
STATS = FIGURES / "graph-stats.json"
SVG = FIGURES / "knowledge-graph-scale.svg"
CACHE = Path.home() / ".cache" / "imas-codex-knowledge-graph-scale"
SAMPLE = CACHE / "graph-sample.json"
LAYOUT = CACHE / "layout.json"

SLIDE_WIDTH, HEIGHT = 14.0, 7.2  # in: the slide canvas, declared at 31.1 x 16 cm
TITLE_SPACE = 1.1  # in kept clear at the top for the slide's own title

NODES_PER_SAMPLE = 250  # one sampled node stands for this many in the graph
NEIGHBOURS_PER_STEP = 5  # snowball fan-out per node, so no hub swallows a sample

# Labels that belong to the IMAS Data Dictionary rather than to a facility.
DD_LABELS = [
    "IMASNode",
    "IDS",
    "IMASNodeChange",
    "DDVersion",
    "IMASSemanticCluster",
    "IdentifierSchema",
    "Unit",
    "COCOS",
    "IMASCoordinateSpec",
    "PhysicsDomain",
    "DDGap",
    "DDGapObservation",
    "DDResolution",
    "DDGapIdentityChange",
]
SN_PREFIXES = [
    "StandardName",
    "Grammar",
    "ISNGrammar",
    "VocabGap",
    "DocsRevision",
    "StructuralNameAuthority",
    "Locus",
    "PromotionCandidate",
    "SNRun",
    "ReviewHold",
]
FACILITIES = ["tcv", "jet", "jt-60sa", "iter"]

# Facility nodes carry facility_id; a Facility node is its own facility.
TRIPLES = """
MATCH (a)-[r]->(b)
RETURN labels(a)[0] AS la,
       coalesce(a.facility_id, CASE WHEN a:Facility THEN a.id END) AS fa,
       type(r) AS t,
       labels(b)[0] AS lb,
       coalesce(b.facility_id, CASE WHEN b:Facility THEN b.id END) AS fb,
       count(*) AS c
"""
NODES = """
MATCH (n)
RETURN labels(n)[0] AS l,
       coalesce(n.facility_id, CASE WHEN n:Facility THEN n.id END) AS f,
       count(*) AS c
"""


def system_of(label: str, facility: str | None) -> str:
    if facility and facility != "test_facility":
        return facility
    if label in DD_LABELS:
        return "dd"
    if label.startswith(tuple(SN_PREFIXES)):
        return "sn"
    return "other"


def query_stats(gc) -> dict:
    nodes: Counter = Counter()
    by_label: dict[str, Counter] = {}
    for row in gc.query(NODES):
        s = system_of(row["l"], row["f"])
        nodes[s] += row["c"]
        by_label.setdefault(s, Counter())[row["l"]] += row["c"]
    internal: Counter = Counter()
    cross: Counter = Counter()
    cross_types: dict[str, Counter] = {}
    for row in gc.query(TRIPLES):
        a, b = system_of(row["la"], row["fa"]), system_of(row["lb"], row["fb"])
        if a == b:
            internal[a] += row["c"]
        else:
            key = "|".join(sorted((a, b)))
            cross[key] += row["c"]
            cross_types.setdefault(key, Counter())[row["t"]] += row["c"]
    return {
        "total_nodes": sum(nodes.values()),
        "total_links": sum(internal.values()) + sum(cross.values()),
        "nodes": dict(nodes),
        "internal_links": dict(internal),
        "cross_links": dict(cross),
        "cross_link_types": {k: dict(v.most_common()) for k, v in cross_types.items()},
        "nodes_by_label": {k: dict(v.most_common()) for k, v in by_label.items()},
    }


# ---------------------------------------------------------------------- sampling


def membership(system: str, var: str) -> str:
    """Cypher predicate: does node `var` belong to `system`."""
    if system in FACILITIES:
        return f"{var}.facility_id = '{system}'"
    if system == "dd":
        return f"any(l IN labels({var}) WHERE l IN $dd)"
    return f"any(l IN labels({var}) WHERE any(p IN $sn WHERE l STARTS WITH p))"


# Sampled relationships that cross systems: (from system, to system, query).
CROSSINGS = [
    (
        "jt-60sa",
        "dd",
        "MATCH (a:SignalSource {facility_id:'jt-60sa'})-[:MAPPING_CANDIDATE]->"
        "(b:IMASNode) WITH a, b ORDER BY rand() LIMIT 120",
    ),
    (
        "jet",
        "dd",
        "MATCH (a:IMASMapping {facility_id:'jet'}) MATCH (b:IDS {id: a.ids_name}) "
        "WITH a, b",
    ),
    (
        "tcv",
        "dd",
        "MATCH (a:SignalNode {facility_id:'tcv'})-[:HAS_UNIT]->(b:Unit) "
        "WITH a, b ORDER BY rand() LIMIT 30",
    ),
    (
        "sn",
        "dd",
        "MATCH (a:StandardNameSource)-[:FROM_DD_PATH]->(b:IMASNode) "
        "WITH a, b ORDER BY rand() LIMIT 60",
    ),
]


def snowball(gc, system: str, budget: int, seeds: list[str], rng) -> dict[str, str]:
    """Grow a sample from seeds along real relationships inside one system."""
    params = {"dd": DD_LABELS, "sn": SN_PREFIXES}
    nodes: dict[str, str] = {}
    for row in gc.query(
        "UNWIND $ids AS id MATCH (n) WHERE elementId(n) = id "
        "RETURN id, labels(n)[0] AS l",
        ids=seeds,
    ):
        nodes[row["id"]] = row["l"]
    frontier = list(nodes)
    expand = f"""
    UNWIND $ids AS id MATCH (n) WHERE elementId(n) = id
    CALL {{
      WITH n
      MATCH (n)-[r]-(m)
      WHERE type(r) <> 'AT_FACILITY' AND NOT m:Facility AND {membership(system, "m")}
      RETURN m LIMIT {NEIGHBOURS_PER_STEP}
    }}
    RETURN elementId(m) AS id, labels(m)[0] AS l
    """
    while frontier and len(nodes) < budget:
        rng.shuffle(frontier)
        batch, frontier = frontier[:150], frontier[150:]
        for row in gc.query(expand, ids=batch, **params):
            if row["id"] not in nodes and len(nodes) < budget:
                nodes[row["id"]] = row["l"]
                frontier.append(row["id"])
    return nodes


def seed_nodes(gc, system: str, by_label: dict[str, int], count: int) -> list[str]:
    """Random seeds spread over a system's labels in proportion to their size."""
    params = {"dd": DD_LABELS, "sn": SN_PREFIXES}
    total = sum(by_label.values())
    seeds: list[str] = []
    for label, n in by_label.items():
        k = round(count * n / total)
        if k == 0:
            continue
        seeds += [
            r["id"]
            for r in gc.query(
                f"MATCH (n:`{label}`) WHERE {membership(system, 'n')} "
                f"WITH n ORDER BY rand() LIMIT {k} RETURN elementId(n) AS id",
                **params,
            )
        ]
    return seeds


def query_sample(gc, stats: dict) -> dict:
    rng = random.Random(11)
    crossing_edges = []
    forced: dict[str, list[str]] = defaultdict(list)
    for a_sys, b_sys, match in CROSSINGS:
        for row in gc.query(match + " RETURN elementId(a) AS a, elementId(b) AS b"):
            crossing_edges.append((row["a"], row["b"], a_sys, b_sys))
            forced[a_sys].append(row["a"])
            forced[b_sys].append(row["b"])
    nodes: dict[str, dict] = {}
    for system in [*FACILITIES, "dd", "sn"]:
        budget = max(12, round(stats["nodes"][system] / NODES_PER_SAMPLE))
        seeds = list(dict.fromkeys(forced[system]))
        seeds += seed_nodes(
            gc, system, stats["nodes_by_label"][system], max(4, budget // 12)
        )
        sample = snowball(gc, system, budget, seeds, rng)
        for node_id, label in sample.items():
            nodes[node_id] = {"system": system, "label": label}
        print(f"{system}: {len(sample)} sampled nodes")
    ids = list(nodes)
    edges = set()
    for i in range(0, len(ids), 500):
        for row in gc.query(
            "UNWIND $ids AS id MATCH (a) WHERE elementId(a) = id "
            "MATCH (a)-[r]->(b) WHERE type(r) <> 'AT_FACILITY' "
            "RETURN id AS a, elementId(b) AS b",
            ids=ids[i : i + 500],
        ):
            if row["b"] in nodes and row["a"] != row["b"]:
                edges.add((row["a"], row["b"]))
    internal = [[a, b] for a, b in edges if nodes[a]["system"] == nodes[b]["system"]]
    crossing = [[a, b] for a, b, *_ in crossing_edges if a in nodes and b in nodes]
    print(f"{len(internal)} internal edges, {len(crossing)} crossing edges")
    return {"nodes": nodes, "internal_edges": internal, "crossing_edges": crossing}


# ---------------------------------------------------------------------- layout


def sfdp(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    """Multilevel force-directed positions from Graphviz sfdp."""
    index = {n: i for i, n in enumerate(graph.nodes)}
    dot = ["graph G {", "node [shape=point];"]
    dot += [f"{i};" for i in index.values()]
    dot += [f"{index[a]} -- {index[b]};" for a, b in graph.edges]
    dot.append("}")
    plain = subprocess.run(
        [
            "sfdp",
            "-Tplain",
            "-Goverlap=true",
            "-Gpack=true",
            "-Gstart=7",
            "-Grepulsiveforce=2.5",
        ],
        input="\n".join(dot),
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    # this sfdp build exits 1 on the overlap-removal warning yet still lays out
    if not plain.rstrip().endswith("stop"):
        raise RuntimeError("sfdp produced no layout")
    names = list(graph.nodes)
    pos = {}
    for line in plain.splitlines():
        part = line.split()
        if part and part[0] == "node":
            pos[names[int(part[1])]] = (float(part[2]), float(part[3]))
    return pos


# What a facility's sub-clusters hold; any other facility label joins "code".
CATEGORIES = {
    "wiki": {"WikiChunk", "WikiPage", "Image", "Document", "DocSource"},
    "signals": {
        "FacilitySignal",
        "SignalSource",
        "SignalNode",
        "DataReference",
        "Diagnostic",
        "DataSource",
        "DataAccess",
        "SignalEpoch",
        "CalibrationEpoch",
        "TDIFunction",
        "IMASMapping",
    },
}


def group_of(system: str, label: str) -> str:
    """A large facility splits into code, wiki and signals; others stay whole."""
    if system not in SATELLITES:
        return system
    for category, members in CATEGORIES.items():
        if label in members:
            return f"{system}/{category}"
    return f"{system}/code"


DOTS_PER_SQUARE_INCH = 300
SMALL_COMPONENT = 6
CROSSING_SHARE = 0.4  # most of a group's dots a crossing sample may take  # components below this size are scattered, not laid out
FISHEYE = 0.85  # radial exponent below 1 spreads a cluster's dense centre
REFINE_STEPS = 60
FOOTPRINT = 0.6  # least footprint of a component, relative to its node share
EQUALISE = 0.6  # 0 keeps the force layout's density, 1 makes it uniform


def spread_components(graph: nx.Graph, pos: dict) -> dict:
    """Open each component's dense core in place.

    sfdp stacks a hub's leaves on top of one another. A short Fruchterman-
    Reingold pass inside each component spreads them, and the result is put back
    on the component's own sfdp centre and footprint, so the packing of the
    components is kept.
    """
    out = dict(pos)
    every = np.array(list(pos.values()))
    span = np.percentile(np.hypot(*(every - np.median(every, axis=0)).T), 95)
    per_root_node = span / math.sqrt(len(every))
    for comp in nx.connected_components(graph):
        if len(comp) < 20:
            continue
        sub = graph.subgraph(comp)
        start = np.array([pos[n] for n in sub.nodes])
        centre = start.mean(axis=0)
        reach = np.percentile(np.hypot(*(start - centre).T), 95) or 1.0
        # a dense component gets at least the footprint its node count earns
        reach = max(reach, FOOTPRINT * per_root_node * math.sqrt(len(comp)))
        refined = nx.spring_layout(
            sub,
            pos={
                n: (p - centre) / reach for n, p in zip(sub.nodes, start, strict=True)
            },
            iterations=REFINE_STEPS,
            seed=5,
        )
        xy = np.array([refined[n] for n in sub.nodes])
        xy -= np.median(xy, axis=0)
        r = np.hypot(*xy.T)
        r_norm = r / (np.percentile(r, 95) or 1.0)
        # blend each node's radius toward its rank's radius in a uniform disk,
        # which thins a dense core while keeping radial order and direction
        rank = np.sqrt((np.argsort(np.argsort(r)) + 0.5) / len(r))
        r_new = (1 - EQUALISE) * r_norm + EQUALISE * rank
        xy *= (reach * r_new / np.maximum(r, 1e-12))[:, None]
        for n, p in zip(sub.nodes, xy + centre, strict=True):
            out[n] = tuple(p)
    return out


def thin(sample: dict, stats: dict) -> set[str]:
    """Nodes to draw: every group at one dot density, in snowball order.

    The snowball over-draws some groups (signals above all), so a group keeps
    only as many sampled nodes as its drawn area holds at DOTS_PER_SQUARE_INCH.
    The sample is stored in snowball order, so the kept prefix stays connected.
    """
    nodes = sample["nodes"]
    geo = geometry(stats)

    def cap(n: str) -> float:
        g = group_of(nodes[n]["system"], nodes[n]["label"])
        return DOTS_PER_SQUARE_INCH * math.pi * geo[g][2] ** 2

    keep: set[str] = set()
    taken: Counter = Counter()
    # crossing endpoints first, but never more than a share of a group's dots,
    # so the group keeps room for its own structure
    for a, b in sample["crossing_edges"]:
        ga, gb = (group_of(nodes[n]["system"], nodes[n]["label"]) for n in (a, b))
        fits = all(
            n in keep or taken[g] < CROSSING_SHARE * cap(n)
            for n, g in ((a, ga), (b, gb))
        )
        if fits:
            for n, g in ((a, ga), (b, gb)):
                if n not in keep:
                    keep.add(n)
                    taken[g] += 1
    for n, d in nodes.items():
        g = group_of(d["system"], d["label"])
        if n not in keep and taken[g] < cap(n):
            keep.add(n)
            taken[g] += 1
    return keep


def layout(sample: dict, stats: dict) -> dict[str, list[float]]:
    """sfdp per group, normalised so 99 % of its nodes sit inside a unit disk."""
    nodes = sample["nodes"]
    keep = thin(sample, stats)
    graphs: dict[str, nx.Graph] = defaultdict(nx.Graph)
    for node_id, node in nodes.items():
        if node_id in keep:
            graphs[group_of(node["system"], node["label"])].add_node(node_id)
    for a, b in sample["internal_edges"]:
        if a not in keep or b not in keep:
            continue
        ga = group_of(nodes[a]["system"], nodes[a]["label"])
        if ga == group_of(nodes[b]["system"], nodes[b]["label"]):
            graphs[ga].add_edge(a, b)
    positions: dict[str, list[float]] = {}
    for group, full in graphs.items():
        # tiny components (isolated nodes, pairs, triples) would be packed on a
        # grid by sfdp; they scatter over the disk instead, one point each
        comps = list(nx.connected_components(full))
        small = [sorted(c) for c in comps if len(c) < SMALL_COMPONENT]
        large = set().union(*[c for c in comps if len(c) >= SMALL_COMPONENT])
        graph = full.subgraph(large)
        if large:
            pos = spread_components(graph, sfdp(graph))
            xy = np.array([pos[n] for n in graph.nodes])
            xy -= (xy.max(axis=0) + xy.min(axis=0)) / 2
            reach = np.percentile(np.hypot(*xy.T), 99) or 1.0
            xy /= reach
            r = np.hypot(*xy.T)
            far = r > 1.0  # pull the last percent of outliers onto the rim
            xy[far] /= r[far, None]
            r = np.minimum(r, 1.0)
            # fisheye: open dense cores so a hub's neighbourhood reads as structure
            xy *= (np.power(r, FISHEYE) / np.maximum(r, 1e-9))[:, None]
            for n, p in zip(graph.nodes, xy, strict=True):
                positions[n] = [float(p[0]), float(p[1])]
        golden = math.pi * (3 - math.sqrt(5))
        for i, comp in enumerate(small):
            rho = math.sqrt((i + 0.5) / len(small))
            cx, cy = rho * math.cos(i * golden), rho * math.sin(i * golden)
            for j, n in enumerate(comp):
                a = 2 * math.pi * j / len(comp)
                spread = 0.025 if len(comp) > 1 else 0.0
                positions[n] = [cx + spread * math.cos(a), cy + spread * math.sin(a)]
        print(f"laid out {group}: {graph.number_of_nodes()} nodes")
    return positions


# ---------------------------------------------------------------------- drawing

SLATE = "#3d4654"  # the Data Dictionary: the reference every system maps onto
INK = "#222222"
CROSS = "#7a7a7a"
CATEGORY_INK = "#5a5a5a"
GAP = 0.08  # in, between a facility's sub-clusters

SYSTEMS = {
    # key: (display name, colour, centre x, centre y, label side). A facility's
    # centre is its code cluster; wiki and signals sit beside it.
    "jet": ("JET", "#d0751f", 1.75, 2.6, "above"),
    "tcv": ("TCV", "#2f6db5", 5.9, 4.0, "above"),
    "dd": ("IMAS Data Dictionary", SLATE, 8.6, 1.75, "below"),
    "jt-60sa": ("JT-60SA", "#2e8b57", 11.9, 4.4, "above"),
    "sn": ("Standard Names", "#c2407a", 12.4, 1.9, "below"),
    "iter": ("ITER", "#8a5cc4", 8.0, 5.0, "right"),
}
# Direction, in degrees from the code cluster, of each satellite cluster; None
# points it at the Data Dictionary so the mapping lines run short and straight.
SATELLITES = {
    "jet": {"wiki": 90, "signals": None},
    "tcv": {"wiki": 180, "signals": -20},
    "jt-60sa": {"signals": None, "wiki": -51},
}
# Where each sub-cluster's name sits, in degrees around that sub-cluster.
CATEGORY_LABEL_ANGLE = {
    "jet": {"code": -90, "wiki": 180, "signals": -90},
    "tcv": {"code": -130, "wiki": 90, "signals": -20},
    "jt-60sa": {"code": 0, "wiki": -51, "signals": -141},
}
RADIUS_PER_ROOT_NODE = 1.55 / math.sqrt(653_458)  # in: area reads as node count

# Arial, falling back to its metric twins Liberation Sans or Nimbus Sans where
# Arial is not installed, so the layout check measures the widths a slide will
# draw. SVG text stays text, so the slide sets it in its own Arial.
FONT = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans"],
    "svg.fonttype": "none",
}


def use_style() -> None:
    plt.style.use("data-ink")
    plt.rcParams.update(FONT)


def human(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.0f} k" if n >= 1e4 else f"{n / 1e3:.2f} k"
    return f"{n:.0f}"


def size_of(count: int) -> float:
    return max(0.16, RADIUS_PER_ROOT_NODE * math.sqrt(count))


def geometry(stats: dict) -> dict[str, tuple[float, float, float]]:
    """Centre and radius of every group, area in proportion to real node count."""
    out = {}
    for system, (_, _, cx, cy, _) in SYSTEMS.items():
        if system not in SATELLITES:
            out[system] = (cx, cy, size_of(stats["nodes"][system]))
            continue
        counts: Counter = Counter()
        for label, n in stats["nodes_by_label"][system].items():
            counts[group_of(system, label)] += n
        rc = size_of(counts[f"{system}/code"])
        out[f"{system}/code"] = (cx, cy, rc)
        for category, angle in SATELLITES[system].items():
            r = size_of(counts[f"{system}/{category}"])
            d = rc + r + GAP
            dx, dy = SYSTEMS["dd"][2] - cx, SYSTEMS["dd"][3] - cy
            a = math.atan2(dy, dx) if angle is None else math.radians(angle)
            out[f"{system}/{category}"] = (
                cx + d * math.cos(a),
                cy + d * math.sin(a),
                r,
            )
    return out


def directional_text(ax, x, y, angle, text, **kw):
    """Text anchored on the side of (x, y) that faces away along angle."""
    c, s = math.cos(angle), math.sin(angle)
    return ax.text(
        x,
        y,
        text,
        ha="left" if c > 0.35 else ("right" if c < -0.35 else "center"),
        va="bottom" if s > 0.35 else ("top" if s < -0.35 else "center"),
        **kw,
    )


def draw(stats: dict, sample: dict, pos: dict, out: Path, draft: Path | None) -> None:
    use_style()
    fig = plt.figure(figsize=(SLIDE_WIDTH, HEIGHT), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.set(xlim=(0, SLIDE_WIDTH), ylim=(0, HEIGHT), aspect="equal")
    nodes = sample["nodes"]
    geo = geometry(stats)
    group = {n: group_of(d["system"], d["label"]) for n, d in nodes.items()}

    xy = {}
    for n in nodes:
        if n in pos and group[n] in geo:
            gx, gy, r = geo[group[n]]
            xy[n] = (gx + r * pos[n][0], gy + r * pos[n][1])
    degree: Counter = Counter()
    for a, b in sample["internal_edges"]:
        degree[a] += 1
        degree[b] += 1

    # crossing relationships beneath everything, in neutral grey
    cross = [(xy[a], xy[b]) for a, b in sample["crossing_edges"] if a in xy and b in xy]
    ax.add_collection(
        LineCollection(cross, colors=CROSS, linewidths=0.5, alpha=0.25, zorder=1)
    )
    for system, (_, colour, *_rest) in SYSTEMS.items():
        inner, between = [], []
        for a, b in sample["internal_edges"]:
            if nodes[a]["system"] != system or a not in xy or b not in xy:
                continue
            (inner if group[a] == group[b] else between).append((xy[a], xy[b]))
        ax.add_collection(
            LineCollection(between, colors=colour, linewidths=0.3, alpha=0.15, zorder=1)
        )
        ax.add_collection(
            LineCollection(inner, colors=colour, linewidths=0.4, alpha=0.45, zorder=2)
        )
        members = [n for n in xy if nodes[n]["system"] == system]
        pts = np.array([xy[n] for n in members])
        size = np.array([1.0 + 1.5 * math.sqrt(degree[n]) for n in members])
        ax.scatter(*pts.T, s=np.minimum(size, 14), color=colour, lw=0, zorder=3)

    labels = []
    # sub-cluster names, just outside each cluster
    for system, angles in CATEGORY_LABEL_ANGLE.items():
        for cat, angle in angles.items():
            gx, gy, r = geo[f"{system}/{cat}"]
            a = math.radians(angle)
            reach = r + 0.08
            labels.append(
                directional_text(
                    ax,
                    gx + reach * math.cos(a),
                    gy + reach * math.sin(a),
                    a,
                    cat,
                    fontsize=14,
                    color=CATEGORY_INK,
                    zorder=6,
                )
            )

    for key, (name, colour, _, _, side) in SYSTEMS.items():
        circles = [g for k, g in geo.items() if k.split("/")[0] == key]
        x0 = min(x - r for x, _, r in circles)
        x1 = max(x + r for x, _, r in circles)
        y0 = min(y - r for _, y, r in circles)
        y1 = max(y + r for _, y, r in circles)
        name_y, lx, ha = {
            "below": (y0 - 0.12, (x0 + x1) / 2, "center"),
            "above": (y1 + 0.75, (x0 + x1) / 2, "center"),
            "right": ((y0 + y1) / 2 + 0.38, x1 + 0.15, "left"),
        }[side]
        detail = f"{human(stats['nodes'][key])} nodes"
        for text, y, size, weight, c in (
            (name, name_y, 22, "bold", colour),
            (detail, name_y - 0.45, 16, "normal", INK),
        ):
            labels.append(
                ax.text(
                    lx,
                    y,
                    text,
                    ha=ha,
                    va="top",
                    fontsize=size,
                    fontweight=weight,
                    color=c,
                    zorder=6,
                )
            )

    if draft is not None:
        check_layout(fig, labels, refuse=False)
        fig.savefig(draft, dpi=100, facecolor="white")
        plt.close(fig)
        return
    check_layout(fig, labels)
    save(fig, out)


def check_layout(fig, labels, *, refuse: bool = True) -> None:
    """Refuse a figure whose text leaves the canvas, enters the title space, or
    collides."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top = fig.bbox.y1 * (1 - TITLE_SPACE / HEIGHT)
    boxes = [(t, t.get_window_extent(renderer)) for t in labels]
    faults = [
        f"off canvas: {t.get_text()!r}"
        for t, b in boxes
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > top
    ]
    faults += [
        f"overlap: {t1.get_text()!r} / {t2.get_text()!r}"
        for (t1, b1), (t2, b2) in combinations(boxes, 2)
        if b1.overlaps(b2)
    ]
    if faults:
        message = "layout check refused the figure:\n  " + "\n  ".join(faults)
        if refuse:
            raise SystemExit(message)
        print(message)


def save(fig, svg: Path) -> None:
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, format="svg", metadata={"Date": None}, facecolor="white")
    plt.close(fig)
    text = svg.read_text()
    # declared at slide height (16 cm) so it drops onto a slide at that size
    text = text.replace(f'width="{SLIDE_WIDTH * 72:g}pt"', 'width="31.1cm"', 1)
    text = text.replace(f'height="{HEIGHT * 72:g}pt"', 'height="16cm"', 1)
    svg.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")
    print(svg.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--query", action="store_true", help="re-count and re-sample the live graph"
    )
    parser.add_argument(
        "--relayout", action="store_true", help="recompute the cached layout"
    )
    parser.add_argument(
        "--draft", type=Path, help="write a PNG here, reporting layout faults"
    )
    args = parser.parse_args()
    if args.query:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        stats = query_stats(gc)
        FIGURES.mkdir(parents=True, exist_ok=True)
        STATS.write_text(json.dumps(stats, indent=1) + "\n")
        CACHE.mkdir(parents=True, exist_ok=True)
        SAMPLE.write_text(json.dumps(query_sample(gc, stats)))
    stats = json.loads(STATS.read_text())
    sample = json.loads(SAMPLE.read_text())
    if args.query or args.relayout or not LAYOUT.exists():
        LAYOUT.write_text(json.dumps(layout(sample, stats)))
    draw(stats, sample, json.loads(LAYOUT.read_text()), SVG, args.draft)


if __name__ == "__main__":
    main()

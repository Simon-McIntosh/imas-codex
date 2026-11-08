"""Draw the scale of the knowledge graph as one slide-sized figure.

Each system in the graph (one disk per fusion facility, one for the IMAS Data
Dictionary, one for the standard-name catalog) is drawn as a sunflower of dots,
one dot per thousand nodes, so disk area reads as node count. Faint chords
inside a disk stand for its internal relationships, one chord per five thousand.
Grey bands between disks carry the relationships that cross systems, with width
growing as the square root of their count.

    uv run python scripts/figures_knowledge_graph_scale.py            # query graph, draw
    uv run python scripts/figures_knowledge_graph_scale.py --cached   # redraw from stats

The query pass writes ``graph-stats.json`` beside the figure so a redraw never
needs the graph.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures" / "knowledge-graph-scale"
STATS = FIGURES / "graph-stats.json"
SVG = FIGURES / "knowledge-graph-scale.svg"

SLIDE_WIDTH, HEIGHT = 14.0, 7.2  # in: the slide canvas, declared at 31.1 x 16 cm

NODES_PER_DOT = 1_000
LINKS_PER_CHORD = 5_000
DOT_SPACING = 0.072  # in, between neighbouring dots in a sunflower

# Labels that belong to the IMAS Data Dictionary rather than to a facility.
DD_LABELS = {
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
}
SN_PREFIXES = (
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
)

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
    if label.startswith(SN_PREFIXES):
        return "sn"
    return "other"


def query_stats() -> dict:
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
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


# ---------------------------------------------------------------------- drawing

SLATE = "#3d4654"  # the Data Dictionary: the reference every system maps onto
LINK = "#b8b8b8"
INK = "#222222"

SYSTEMS = {
    # key: (display name, colour, centre x, centre y, label side)
    "jet": ("JET", "#d0751f", 2.1, 3.1, "below"),
    "tcv": ("TCV", "#2f6db5", 5.85, 4.3, "above"),
    "dd": ("IMAS Data Dictionary", SLATE, 9.45, 2.05, "below"),
    "jt-60sa": ("JT-60SA", "#2e8b57", 12.2, 5.0, "above"),
    "sn": ("Standard Names", "#c2407a", 12.45, 2.85, "below"),
    "iter": ("ITER", "#8a5cc4", 8.4, 4.6, "right"),
}

# Crossing links worth naming, keyed by the sorted system pair.
LINK_NAMES = {
    "dd|jt-60sa": "mapping candidates",
    "dd|tcv": "units",
    "dd|sn": "sources",
}


def human(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.0f} k"
    return f"{n:.0f}"


def sunflower(n: int, cx: float, cy: float) -> list[tuple[float, float]]:
    golden = math.pi * (3 - math.sqrt(5))
    return [
        (
            cx + DOT_SPACING * math.sqrt(i + 0.5) * math.cos(i * golden),
            cy + DOT_SPACING * math.sqrt(i + 0.5) * math.sin(i * golden),
        )
        for i in range(n)
    ]


def radius(n: int) -> float:
    return DOT_SPACING * math.sqrt(max(n, 1))


def chords(points, n: int, rng: random.Random):
    """Join n random pairs of nearby dots: a web whose density reads as links."""
    out = []
    reach = 6 * DOT_SPACING
    for _ in range(n):
        a = rng.choice(points)
        near = [p for p in points if 0 < math.dist(a, p) < reach]
        if near:
            out.append((a, rng.choice(near)))
    return out


def draw(stats: dict, out: Path, draft: Path | None = None) -> None:
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(SLIDE_WIDTH, HEIGHT), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.set(xlim=(0, SLIDE_WIDTH), ylim=(0, HEIGHT), aspect="equal")
    rng = random.Random(7)
    labels = []

    # crossing links first, beneath everything, running edge to edge
    drawn = {
        pair: count
        for pair, count in stats["cross_links"].items()
        if all(s in SYSTEMS for s in pair.split("|")) and count >= 1_000
    }
    for pair, count in drawn.items():
        a, b = pair.split("|")
        (ax_, ay), (bx, by) = SYSTEMS[a][2:4], SYSTEMS[b][2:4]
        d = math.dist((ax_, ay), (bx, by))
        ux, uy = (bx - ax_) / d, (by - ay) / d
        ra = radius(round(stats["nodes"][a] / NODES_PER_DOT)) + 0.1
        rb = radius(round(stats["nodes"][b] / NODES_PER_DOT)) + 0.1
        width = 3 + 21 * math.sqrt(count / max(drawn.values()))
        ax.plot(
            [ax_ + ra * ux, bx - rb * ux],
            [ay + ra * uy, by - rb * uy],
            color=LINK,
            lw=width,
            solid_capstyle="butt",
            zorder=1,
        )
        f = (ra + (d - ra - rb) / 2) / d  # midpoint of the visible gap
        mx, my = ax_ + f * (bx - ax_), ay + f * (by - ay)
        name = LINK_NAMES.get(pair, "links")
        labels.append(
            ax.text(
                mx,
                my,
                f"{human(count)}\n{name}",
                ha="center",
                va="center",
                fontsize=14,
                color=INK,
                linespacing=1.1,
                zorder=6,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"},
            )
        )

    for key, (name, colour, cx, cy, side) in SYSTEMS.items():
        n_nodes = stats["nodes"].get(key, 0)
        n_links = stats["internal_links"].get(key, 0)
        dots = sunflower(max(1, round(n_nodes / NODES_PER_DOT)), cx, cy)
        for p, q in chords(dots, round(n_links / LINKS_PER_CHORD), rng):
            ax.plot(
                [p[0], q[0]], [p[1], q[1]], color=colour, lw=0.7, alpha=0.4, zorder=2
            )
        xs, ys = zip(*dots, strict=True)
        ax.scatter(xs, ys, s=10, color=colour, linewidths=0, zorder=3)
        # name in the system colour, counts beneath it, in clear space by the disk
        r = radius(len(dots))
        name_y, ha = {
            "below": (cy - r - 0.15, "center"),
            "above": (cy + r + 0.85, "center"),
            "right": (cy + 0.4, "left"),
        }[side]
        lx = cx + r + 0.2 if side == "right" else cx
        detail = f"{human(n_nodes)} nodes · {human(n_links)} links"
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

    labels.append(
        ax.text(
            0.25,
            HEIGHT - 0.25,
            f"{human(stats['total_nodes'])} nodes",
            ha="left",
            va="top",
            fontsize=34,
            fontweight="bold",
            color=INK,
        )
    )
    labels.append(
        ax.text(
            0.25,
            HEIGHT - 0.95,
            f"{human(stats['total_links'])} links",
            ha="left",
            va="top",
            fontsize=34,
            color=INK,
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
    """Refuse a figure whose text leaves the canvas or collides."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(t, t.get_window_extent(renderer)) for t in labels]
    faults = [
        f"off canvas: {t.get_text()!r}"
        for t, b in boxes
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1
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
        "--cached", action="store_true", help="redraw from graph-stats.json"
    )
    parser.add_argument(
        "--draft", type=Path, help="write a PNG here, reporting layout faults"
    )
    args = parser.parse_args()
    if args.cached:
        stats = json.loads(STATS.read_text())
    else:
        stats = query_stats()
        FIGURES.mkdir(parents=True, exist_ok=True)
        STATS.write_text(json.dumps(stats, indent=1) + "\n")
        print(STATS.relative_to(ROOT))
    draw(stats, SVG, args.draft)


if __name__ == "__main__":
    main()

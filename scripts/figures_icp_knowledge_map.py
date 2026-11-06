"""Illustrate ITER's engineering data as a road map of France.

An illustration, not a census. Each town is one body of data ITER owns
(documents, CAD models, drawings, non-conformance reports, ...). The motorways
are the links between them: infrastructure ITER builds, pays for and keeps,
which is the organisation's memory. Each named relationship runs along the real
motorways between its two towns and is labelled with a plain ontological verb
(``Documents references CAD Models``). Cars on the roads are agents shuttling
information between the towns on behalf of the engineer at ITER; any model can
drive them, and the roads stay.

Geography: the metropolitan outline is the simplified ``france-geojson``
(Licence Ouverte / Etalab); the roads are Natural Earth 10m major highways
(public domain), clipped to that outline.

Usage::

    uv run python scripts/figures_icp_knowledge_map.py
    uv run python scripts/figures_icp_knowledge_map.py --draft /tmp/x.png
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
OUTLINE = ROOT / "scripts/figure_data/france_metropole.geojson"
ROADS = ROOT / "scripts/figure_data/france_motorways.geojson"
FIGURES = ROOT / "docs/figures/icp-knowledge-map"

COS_LAT = math.cos(math.radians(46.5))  # equirectangular scale at mid-France
LON, LAT = (-4.95, 8.35), (42.25, 51.2)  # mainland France, Corsica outside
HEIGHT = 7.2  # in; the width follows France so the country fills the frame
WIDTH = round(HEIGHT * (LON[1] - LON[0]) * COS_LAT / (LAT[1] - LAT[0]), 2)

LAND = "#f2f1ee"
COAST = "#b4b1aa"
ROAD = "#b9b6b0"
INK = "#2b2b2b"
MUTED = "#6b6b6b"
AGENT = "#e07020"  # every car is an agent: one hue for one role
USER = "#2e6fb5"  # the engineer at ITER


@dataclass(frozen=True)
class Town:
    key: str
    name: str  # the data it holds
    system: str  # the system that holds it
    lon: float
    lat: float
    label: tuple[float, float, str]  # offset in points; dy >= 0 stacks above


TOWNS = [
    Town("doc", "Documents", "IDM", 2.35, 48.86, (-14, 8, "right")),
    Town(
        "cad",
        "CAD Models",
        "CATIA · Dassault Systèmes",
        1.44,
        43.60,
        (-10, -10, "right"),
    ),
    Town("drw", "2D Drawings", "SMDD", -0.58, 44.84, (-12, 2, "right")),
    Town("ncr", "Non-Conformance Reports", "NCR", 3.06, 50.63, (-12, 2, "right")),
    Town("cmp", "Components", "functional references", 4.84, 45.76, (12, 2, "left")),
    Town(
        "ana", "Analysis Results", "simulation archive", 3.88, 43.61, (-4, -12, "left")
    ),
    Town("con", "Contracts", "procurement · SAP", -1.68, 48.11, (-4, -12, "center")),
    Town("req", "Requirements", "RQM", 7.75, 48.57, (-8, -10, "right")),
    Town("pbs", "Plant Breakdown", "PBS", 5.04, 47.32, (0, 8, "center")),
]
ITER = (5.71, 43.69)  # Saint-Paul-lez-Durance

# (subject, verb, object, (fraction along the route, label nudge in points)):
# illustrative, not a schema.
RELATIONS = [
    ("doc", "references", "cad", (0.5, 0, 0)),
    ("cad", "has", "ana", (0.5, 0, 0)),
    ("cad", "is drawn as", "drw", (0.5, 0, 0)),
    ("ncr", "cites", "doc", (0.5, 0, 0)),
    ("drw", "depicts", "cmp", (0.5, 0, 0)),
    ("con", "delivers", "doc", (0.3, -34, 0)),
    ("doc", "specifies", "req", (0.5, 0, 0)),
    ("cmp", "is part of", "pbs", (0.5, 0, 0)),
    ("cmp", "is installed at", "iter", (0.5, 0, 0)),
]

# Waypoints that keep a route on the obvious motorway, by subject.
VIA: dict[str, list[tuple[float, float]]] = {}

# Cars: (relation index, fraction along its route).
CARS = [(0, 0.3), (1, 0.18), (4, 0.75), (8, 0.7), (3, 0.25)]

ARROWS = "→↗↑↖←↙↓↘"


def project(lon: float, lat: float) -> tuple[float, float]:
    return lon * COS_LAT, lat


def place(key: str) -> tuple[float, float]:
    if key == "iter":
        return project(*ITER)
    town = next(t for t in TOWNS if t.key == key)
    return project(town.lon, town.lat)


def load_roads() -> list[list[tuple[float, float]]]:
    features = json.loads(ROADS.read_text())["features"]
    return [
        [project(lon, lat) for lon, lat in f["geometry"]["coordinates"]]
        for f in features
        if f["properties"]["type"] == "Major Highway"
    ]


class RoadGraph:
    """The motorway polylines as a graph, so a relation can follow the roads."""

    def __init__(self, roads: list[list[tuple[float, float]]], snap: float = 0.1):
        self.nodes: list[tuple[float, float]] = []
        self.edges: dict[int, list[tuple[int, float]]] = {}
        index: dict[tuple[float, float], int] = {}

        def node(p):
            key = (round(p[0], 4), round(p[1], 4))
            if key not in index:
                index[key] = len(self.nodes)
                self.nodes.append(p)
            return index[key]

        ends = []
        for line in roads:
            ids = [node(p) for p in line]
            for a, b in zip(ids, ids[1:], strict=False):
                self.link(a, b)
            ends += [ids[0], ids[-1]]
        # polylines that meet at a junction rarely share a vertex: join them
        for e in ends:
            for n, p in enumerate(self.nodes):
                if n != e and math.dist(p, self.nodes[e]) < snap:
                    self.link(e, n)

    def link(self, a: int, b: int) -> None:
        d = math.dist(self.nodes[a], self.nodes[b])
        self.edges.setdefault(a, []).append((b, d))
        self.edges.setdefault(b, []).append((a, d))

    def nearest(self, p: tuple[float, float]) -> int:
        return min(range(len(self.nodes)), key=lambda n: math.dist(self.nodes[n], p))

    def path(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> list[tuple[float, float]]:
        start, goal = self.nearest(a), self.nearest(b)
        queue, best, prev = [(0.0, start)], {start: 0.0}, {}
        while queue:
            dist, n = heapq.heappop(queue)
            if n == goal:
                break
            for m, d in self.edges.get(n, []):
                if dist + d < best.get(m, math.inf):
                    best[m], prev[m] = dist + d, n
                    heapq.heappush(queue, (dist + d, m))
        if goal not in best:
            raise SystemExit(f"no motorway route between {a} and {b}")
        points, n = [], goal
        while n != start:
            points.append(self.nodes[n])
            n = prev[n]
        points.append(self.nodes[start])
        return [a, *reversed(points), b]


def along(points, f: float) -> tuple[float, float, float]:
    """Point and heading at fraction f of a polyline's length."""
    lengths = [math.dist(p, q) for p, q in zip(points, points[1:], strict=False)]
    target, run = f * sum(lengths), 0.0
    for (p, q), d in zip(zip(points, points[1:], strict=False), lengths, strict=False):
        if run + d >= target and d > 0:
            t = (target - run) / d
            return (
                p[0] + t * (q[0] - p[0]),
                p[1] + t * (q[1] - p[1]),
                math.atan2(q[1] - p[1], q[0] - p[0]),
            )
        run += d
    p, q = points[-2], points[-1]
    return q[0], q[1], math.atan2(q[1] - p[1], q[0] - p[0])


def draw_land(ax) -> None:
    for polygon in json.loads(OUTLINE.read_text())["geometry"]["coordinates"]:
        if min(lon for lon, _ in polygon[0]) > LON[1]:  # Corsica
            continue
        ring = [project(lon, lat) for lon, lat in polygon[0]]
        ax.add_patch(
            Polygon(
                ring,
                closed=True,
                facecolor=LAND,
                edgecolor=COAST,
                linewidth=1.0,
                zorder=1,
            )
        )


def draw_car(ax, x: float, y: float, heading: float, size: float = 0.21) -> None:
    """A small car seen from above, pointing along the road."""
    c, s = math.cos(heading), math.sin(heading)

    def rot(u, v):
        return x + size * (u * c - v * s), y + size * (u * s + v * c)

    body = [
        rot(u, v)
        for u, v in [
            (-1, -0.5),
            (0.8, -0.5),
            (1, -0.3),
            (1, 0.3),
            (0.8, 0.5),
            (-1, 0.5),
        ]
    ]
    glass = [
        rot(u, v) for u, v in [(0.15, -0.36), (0.55, -0.32), (0.55, 0.32), (0.15, 0.36)]
    ]
    ax.add_patch(
        Polygon(
            body,
            closed=True,
            facecolor=AGENT,
            edgecolor="white",
            linewidth=1.2,
            zorder=6,
        )
    )
    ax.add_patch(Polygon(glass, closed=True, facecolor="white", linewidth=0, zorder=7))


def draw_user(ax) -> list[Text]:
    x, y = project(*ITER)
    ax.add_patch(
        Circle((x, y), 0.20, facecolor=USER, edgecolor="white", linewidth=1.5, zorder=8)
    )
    ax.add_patch(
        Circle((x, y + 0.045), 0.055, facecolor="white", linewidth=0, zorder=9)
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.085, y - 0.12),
            0.17,
            0.07,
            boxstyle="round,pad=0,rounding_size=0.035",
            facecolor="white",
            linewidth=0,
            zorder=9,
        )
    )
    name = ax.annotate(
        "ITER",
        (x, y),
        xytext=(16, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=USER,
        zorder=10,
    )
    sub = ax.annotate(
        "you are here",
        (x, y),
        xytext=(16, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=11,
        color=USER,
        zorder=10,
    )
    return [name, sub]


def draw_towns(ax) -> list[Text]:
    labels = []
    for town in TOWNS:
        x, y = project(town.lon, town.lat)
        hub = town.key == "doc"
        ax.plot(
            x,
            y,
            "o",
            markersize=16 if hub else 11,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=2.6,
            zorder=7,
        )
        if hub:
            ax.plot(x, y, "o", markersize=5.5, color=INK, zorder=8)
        dx, dy, ha = town.label
        name_dy, sub_dy = (dy + 13, dy) if dy >= 0 else (dy, dy - 17)
        va = "bottom" if dy >= 0 else "top"
        halo = [pe.withStroke(linewidth=3, foreground=LAND)]
        labels.append(
            ax.annotate(
                town.name,
                (x, y),
                xytext=(dx, name_dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=14,
                fontweight="bold",
                color=INK,
                zorder=10,
                path_effects=halo,
            )
        )
        labels.append(
            ax.annotate(
                town.system,
                (x, y),
                xytext=(dx, sub_dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=11,
                color=MUTED,
                zorder=10,
                path_effects=halo,
            )
        )
    return labels


def draw_relations(ax, routes) -> list[Text]:
    labels = []
    for (subject, verb, obj, (f, nx, ny)), points in zip(
        RELATIONS, routes, strict=True
    ):
        x, y, _ = along(points, f)
        sx, sy = place(subject)
        ox, oy = place(obj)
        sector = round(math.atan2(oy - sy, ox - sx) / (math.pi / 4)) % 8
        labels.append(
            ax.annotate(
                f"{verb} {ARROWS[sector]}",
                (x, y),
                xytext=(nx, ny),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=12,
                fontstyle="italic",
                color=INK,
                zorder=9,
                path_effects=[pe.withStroke(linewidth=4, foreground=LAND)],
            )
        )
    return labels


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


def save(fig, stem: str, labels, draft: Path | None) -> Path:
    if draft is not None:
        check_layout(fig, labels, refuse=False)
        fig.savefig(draft, dpi=100, facecolor="white")
        plt.close(fig)
        return draft
    check_layout(fig, labels)
    FIGURES.mkdir(parents=True, exist_ok=True)
    svg = FIGURES / f"{stem}.svg"
    fig.savefig(svg, format="svg", metadata={"Date": None}, facecolor="white")
    text = svg.read_text()
    # declared at slide height (16 cm) so it drops onto a slide at that size
    text = text.replace(
        f'width="{WIDTH * 72:g}pt"', f'width="{16 * WIDTH / HEIGHT:.1f}cm"', 1
    )
    text = text.replace(f'height="{HEIGHT * 72:g}pt"', 'height="16cm"', 1)
    svg.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")
    plt.close(fig)
    print(svg.relative_to(ROOT))
    return svg


def map_figure(draft: Path | None = None) -> Path:
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.set(xlim=[v * COS_LAT for v in LON], ylim=LAT, aspect="equal")
    draw_land(ax)
    roads = load_roads()
    for line in roads:
        xs, ys = zip(*line, strict=True)
        ax.plot(xs, ys, color=ROAD, linewidth=1.2, solid_capstyle="round", zorder=2)
    graph = RoadGraph(roads)
    routes = []
    for s, _, o, _ in RELATIONS:
        stops = [place(s), *(project(*v) for v in VIA.get(s, [])), place(o)]
        legs = [graph.path(a, b) for a, b in zip(stops, stops[1:], strict=False)]
        routes.append(
            [p for i, leg in enumerate(legs) for p in (leg if i == 0 else leg[1:])]
        )
    for points in routes:  # the stretch of road a relation travels, tying town to road
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=INK, linewidth=2.6, solid_capstyle="round", zorder=3)
    for i, f in CARS:
        draw_car(ax, *along(routes[i], f))
    labels = draw_towns(ax) + draw_relations(ax, routes) + draw_user(ax)
    return save(fig, "icp-data-as-france", labels, draft)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--draft",
        type=Path,
        help="write a PNG here, reporting layout faults instead of refusing",
    )
    map_figure(parser.parse_args().draft)


if __name__ == "__main__":
    main()

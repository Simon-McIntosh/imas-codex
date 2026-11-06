"""Illustrate ITER's engineering data as a road map of France, built in stages.

An illustration, not a census. Each town is one body of data ITER owns
(documents, CAD models, drawings, non-conformance reports, ...). The motorways
are the links between them: infrastructure ITER builds, pays for and keeps,
which is the organisation's memory. Each named relationship runs along the real
motorways between its two towns and carries a plain ontological verb
(``Documents references CAD Models``). Vehicles are agents' requests: they only
ever travel the relationships, the way a traversal runs on a live graph
database, and shuttle what they read back the way they came. They drive beneath
every label. Whatever the vehicle, the roads stay ours. ITER sits east of
the map, joined to it by one on-ramp: each burst of vehicles is a question
leaving ITER, and each vehicle's return is an answer coming back.

The build sequence shares one map, at one size and position, so the images stack:

a. the data systems: town names with an icon each, on white;
b. France beneath them, a dot per town and the ITER emblem, no links;
c. the road network with each relationship and its verb;
d. live traffic: many concurrent requests, one model;
e. the same traffic driven by several models, one colour each;
f. the whole fleet: several models and kinds of vehicle, under heavier load;
g. a question typed into Lucy's chat sends traversals into the busy graph,
   and the answer streams back (full slide width, the map on the left).

A separate full-width grid shows the same repositories as cards (``--grid``); with
``--thumbnails`` it carries example IDM pages and is written to a gitignored
``private`` folder, because those pages are internal documents.

Geography: the metropolitan outline is the simplified ``france-geojson``
(Licence Ouverte / Etalab); the roads are Natural Earth 10m major highways
(public domain), clipped to that outline.

Usage::

    uv run python scripts/figures_icp_knowledge_map.py            # every image
    uv run python scripts/figures_icp_knowledge_map.py --only c   # one image
    uv run python scripts/figures_icp_knowledge_map.py --only c --draft /tmp/x.png
    uv run python scripts/figures_icp_knowledge_map.py --grid [--thumbnails DIR]
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import math
import random
import re
import textwrap
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts/figure_data"
OUTLINE = DATA / "france_metropole.geojson"
ROADS = DATA / "france_motorways.geojson"
EMBLEM = DATA / "iter_logo.svg"
FIGURES = ROOT / "docs/figures/icp-knowledge-map"

COS_LAT = math.cos(math.radians(46.5))  # equirectangular scale at mid-France
LON, LAT = (-4.95, 8.35), (42.25, 51.2)  # mainland France, Corsica outside
HEIGHT = 7.2  # in; the width follows France so the country fills the frame
WIDTH = round(HEIGHT * (LON[1] - LON[0]) * COS_LAT / (LAT[1] - LAT[0]), 2)

LAND = "#f2f1ee"
COAST = "#b4b1aa"
ROAD = "#b9b6b0"
INK = "#2b2b2b"
EMBLEM_YELLOW = "#FDBB30"  # ITER Noon Yellow, as the emblem file carries it
MUTED = "#6b6b6b"

# One colour per model family driving the agents.
CLAUDE = "#D97757"
CODEX = "#141414"  # OpenAI's brand is black and white; Codex carries no hue of its own
LOCAL = "#4d6bfe"  # a locally served open-weight model
LOCAL_SECOND = "#2e8b57"  # a second local model


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
        (-14, -10, "right"),
    ),
    Town("drw", "2D Drawings", "SMDD", -0.58, 44.84, (-14, 2, "right")),
    Town("ncr", "Non-Conformance Reports", "NCR", 3.06, 50.63, (-14, 2, "right")),
    Town("cmp", "Components", "functional references", 4.84, 45.76, (14, 2, "left")),
    Town(
        "ana", "Analysis Results", "simulation archive", 3.88, 43.61, (-4, -14, "left")
    ),
    Town("con", "Contracts", "procurement · SAP", -1.68, 48.11, (-4, -14, "center")),
    Town("req", "Requirements", "RQM", 7.75, 48.57, (-10, -12, "right")),
    Town("pbs", "Plant Breakdown", "PBS", 5.04, 47.32, (0, 10, "center")),
]

# Whoever asks, at ITER or at a Domestic Agency: east of France, off the data map,
# joined to it by one on-ramp to the Plant Breakdown, the hub every record cites.
ITER = (7.45, 46.75)
ON_RAMP = "pbs"

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
    ("ana", "assesses", "cmp", (0.5, 0, 0)),
    ("doc", "is filed under", "pbs", (0.32, 0, 0)),
]

ARROWS = "→↗↑↖←↙↓↘"


def project(lon: float, lat: float) -> tuple[float, float]:
    return lon * COS_LAT, lat


def place(key: str) -> tuple[float, float]:
    if key == "iter":
        return project(*ITER)
    town = next(t for t in TOWNS if t.key == key)
    return project(town.lon, town.lat)


# --------------------------------------------------------------------------- roads


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


def relation_routes(roads) -> list[list[tuple[float, float]]]:
    """The road each relationship runs along, then ITER's on-ramp last."""
    graph = RoadGraph(roads)
    routes = [graph.path(place(s), place(o)) for s, _, o, _ in RELATIONS]
    return [*routes, graph.path(place("iter"), place(ON_RAMP))]


def length(points) -> float:
    return sum(math.dist(p, q) for p, q in zip(points, points[1:], strict=False))


def along(points, f: float) -> tuple[float, float, float]:
    """Point and heading at fraction f of a polyline's length."""
    target, run = f * length(points), 0.0
    for p, q in zip(points, points[1:], strict=False):
        d = math.dist(p, q)
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


# --------------------------------------------------------------------------- glyphs


def land_patch() -> list[Polygon]:
    patches = []
    for polygon in json.loads(OUTLINE.read_text())["geometry"]["coordinates"]:
        if min(lon for lon, _ in polygon[0]) > LON[1]:  # Corsica
            continue
        ring = [project(lon, lat) for lon, lat in polygon[0]]
        patches.append(Polygon(ring, closed=True))
    return patches


def svg_path(d: str) -> MplPath:
    """Parse the absolute M/L/H/V/C/Z subset of SVG path data the emblem uses."""
    tokens = re.findall(r"[MLHVCZ]|-?\d*\.?\d+(?:e-?\d+)?", d)
    verts, codes, i, cmd = [], [], 0, None
    x = y = sx = sy = 0.0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd, i = tokens[i], i + 1
            if cmd == "Z":
                verts.append((sx, sy))
                codes.append(MplPath.CLOSEPOLY)
                continue
        nums = tokens[i:]
        if cmd == "M":
            x, y = float(nums[0]), float(nums[1])
            sx, sy = x, y
            verts.append((x, y))
            codes.append(MplPath.MOVETO)
            i, cmd = i + 2, "L"
        elif cmd == "L":
            x, y = float(nums[0]), float(nums[1])
            verts.append((x, y))
            codes.append(MplPath.LINETO)
            i += 2
        elif cmd in "HV":
            if cmd == "H":
                x = float(nums[0])
            else:
                y = float(nums[0])
            verts.append((x, y))
            codes.append(MplPath.LINETO)
            i += 1
        elif cmd == "C":
            pts = [(float(nums[k]), float(nums[k + 1])) for k in (0, 2, 4)]
            verts += pts
            codes += [MplPath.CURVE4] * 3
            x, y = pts[-1]
            i += 6
    return MplPath(verts, codes)


def draw_emblem(ax, width: float = 0.8) -> None:
    """ITER's emblem: the noon sun with the name cut out of it."""
    d = re.search(r'd="([^"]+)"', EMBLEM.read_text()).group(1)
    path = svg_path(d)
    x, y = project(*ITER)
    scale = width / 120.0
    verts = [(x + (u - 60) * scale, y - (v - 30) * scale) for u, v in path.vertices]
    ax.add_patch(
        PathPatch(
            MplPath(verts, path.codes),
            facecolor=EMBLEM_YELLOW,
            edgecolor="none",
            zorder=8,
        )
    )


def draw_icon(ax, kind: str, x: float, y: float, s: float = 0.2) -> None:
    """A simple line icon for each kind of data, centred on its town."""

    def pts(coords):
        return [(x + s * u, y + s * v) for u, v in coords]

    style = {"edgecolor": INK, "linewidth": 1.8, "zorder": 7, "joinstyle": "round"}

    def poly(coords, face="white", **kw):
        ax.add_patch(
            Polygon(pts(coords), closed=True, facecolor=face, **{**style, **kw})
        )

    def line(coords, **kw):
        xs, ys = zip(*pts(coords), strict=True)
        ax.plot(
            xs,
            ys,
            color=INK,
            linewidth=kw.get("lw", 1.6),
            solid_capstyle="round",
            zorder=8,
        )

    if kind == "doc":
        poly([(-0.7, -1), (0.7, -1), (0.7, 0.55), (0.25, 1), (-0.7, 1)])
        poly([(0.25, 1), (0.25, 0.55), (0.7, 0.55)])
        for v in (0.25, -0.15, -0.55):
            line([(-0.4, v), (0.4, v)])
    elif kind == "cad":
        r = 0.95
        top = [(0, r), (0.87 * r, 0.5 * r), (0, 0), (-0.87 * r, 0.5 * r)]
        poly(top, face="#ffffff")
        poly(
            [(-0.87 * r, 0.5 * r), (0, 0), (0, -r), (-0.87 * r, -0.5 * r)],
            face="#c9c7c2",
        )
        poly(
            [(0, 0), (0.87 * r, 0.5 * r), (0.87 * r, -0.5 * r), (0, -r)], face="#8f8c86"
        )
    elif kind == "drw":
        poly([(-1, -0.75), (1, -0.75), (1, 0.75), (-1, 0.75)])
        poly([(0.35, -0.75), (1, -0.75), (1, -0.4), (0.35, -0.4)])
        ax.add_patch(
            Circle((x - 0.25 * s, y + 0.05 * s), 0.38 * s, facecolor="none", **style)
        )
        line([(-0.8, 0.05), (0.3, 0.05)], lw=1.0)
        line([(-0.25, -0.5), (-0.25, 0.6)], lw=1.0)
    elif kind == "ncr":
        poly([(0, 1), (1.05, -0.8), (-1.05, -0.8)])
        line([(0, 0.45), (0, -0.2)], lw=2.6)
        ax.add_patch(
            Circle(
                (x, y - 0.5 * s), 0.09 * s, facecolor=INK, edgecolor="none", zorder=8
            )
        )
    elif kind == "cmp":
        teeth, coords = 8, []
        for k in range(teeth * 4):
            a = 2 * math.pi * k / (teeth * 4)
            rr = 1.0 if (k % 4) in (1, 2) else 0.74
            coords.append((rr * math.cos(a), rr * math.sin(a)))
        poly(coords)
        ax.add_patch(Circle((x, y), 0.3 * s, facecolor="white", **style))
    elif kind == "pbs":
        line([(0, 0.45), (0, 0.05)])
        line([(-0.7, 0.05), (0.7, 0.05)])
        for u in (-0.7, 0, 0.7):
            line([(u, 0.05), (u, -0.3)])
        poly([(-0.35, 0.45), (0.35, 0.45), (0.35, 0.95), (-0.35, 0.95)])
        for u in (-0.7, 0, 0.7):
            poly(
                [
                    (u - 0.27, -0.3),
                    (u + 0.27, -0.3),
                    (u + 0.27, -0.75),
                    (u - 0.27, -0.75),
                ]
            )
    elif kind == "req":
        poly([(-0.75, -1), (0.75, -1), (0.75, 0.9), (-0.75, 0.9)])
        for v in (0.45, -0.05, -0.55):
            line([(-0.5, v), (-0.35, v - 0.15), (-0.12, v + 0.12)], lw=1.6)
            line([(0.05, v), (0.5, v)], lw=1.4)
    elif kind == "con":
        poly([(-0.7, -1), (0.7, -1), (0.7, 1), (-0.7, 1)])
        for v in (0.55, 0.2):
            line([(-0.45, v), (0.45, v)])
        sig = [
            (-0.45 + 0.9 * t / 20, -0.35 + 0.12 * math.sin(t * 1.3)) for t in range(21)
        ]
        line(sig, lw=1.4)
        ax.add_patch(
            Circle(
                (x + 0.42 * s, y - 0.7 * s),
                0.2 * s,
                facecolor=INK,
                edgecolor="none",
                zorder=8,
            )
        )
    elif kind == "ana":
        ax.add_patch(
            Rectangle((x - s, y - 0.8 * s), 2 * s, 1.6 * s, facecolor="white", **style)
        )
        cmap = plt.get_cmap("viridis")
        for k, frac in enumerate((1.0, 0.75, 0.5, 0.27)):
            ax.add_patch(
                Ellipse(
                    (x - 0.15 * s, y),
                    1.7 * s * frac,
                    1.35 * s * frac,
                    angle=20,
                    facecolor=cmap(0.15 + 0.25 * k),
                    edgecolor="none",
                    zorder=8,
                )
            )


ICON_KIND = {t.key: t.key for t in TOWNS}


def car_shape(
    kind: str,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    """Outline, glazing and size of a vehicle seen from above, pointing along +u."""
    if kind == "truck":
        body = [
            (-1.6, -0.55),
            (0.55, -0.55),
            (0.62, -0.47),
            (1.15, -0.45),
            (1.15, 0.45),
            (0.62, 0.47),
            (0.55, 0.55),
            (-1.6, 0.55),
        ]
        glass = [(0.75, -0.38), (1.02, -0.36), (1.02, 0.36), (0.75, 0.38)]
        return body, glass, 0.2
    if kind == "bike":
        body = [(-1, 0), (-0.6, -0.36), (0.7, -0.36), (1, 0), (0.7, 0.36), (-0.6, 0.36)]
        return body, [(-0.15, -0.13), (0.15, -0.13), (0.15, 0.13), (-0.15, 0.13)], 0.19
    body = [(-1, -0.5), (0.8, -0.5), (1, -0.3), (1, 0.3), (0.8, 0.5), (-1, 0.5)]
    return body, [(0.15, -0.36), (0.55, -0.32), (0.55, 0.32), (0.15, 0.36)], 0.21


def place_shape(coords, x, y, heading, size):
    c, s = math.cos(heading), math.sin(heading)
    return [(x + size * (u * c - v * s), y + size * (u * s + v * c)) for u, v in coords]


# --------------------------------------------------------------------------- labels


def halo_over_land(ax, text: Text, land: list[Polygon] | None, width: float) -> None:
    """Draw a land-coloured halo behind text, clipped to the land so it never shows on white."""
    if land is None:
        return
    halo = ax.annotate(
        text.get_text(),
        text.xy,
        xytext=text.xyann,
        textcoords=text.anncoords,
        ha=text.get_ha(),
        va=text.get_va(),
        fontsize=text.get_fontsize(),
        fontweight=text.get_fontweight(),
        fontstyle=text.get_fontstyle(),
        color=LAND,
        zorder=text.get_zorder() - 0.5,
        path_effects=[pe.Stroke(linewidth=width, foreground=LAND)],
    )
    clip = MplPath.make_compound_path(
        *[p.get_path().transformed(p.get_patch_transform()) for p in land]
    )
    halo.set_clip_path(clip, ax.transData)


def draw_town_labels(ax, land) -> list[Text]:
    labels = []
    for town in TOWNS:
        x, y = place(town.key)
        dx, dy, ha = town.label
        name_dy, sub_dy = (dy + 13, dy) if dy >= 0 else (dy, dy - 17)
        va = "bottom" if dy >= 0 else "top"
        name = ax.annotate(
            town.name,
            (x, y),
            xytext=(dx, name_dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=14,
            fontweight="bold",
            color=INK,
            zorder=11,
        )
        sub = ax.annotate(
            town.system,
            (x, y),
            xytext=(dx, sub_dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=11,
            color=MUTED,
            zorder=11,
        )
        for t in (name, sub):
            halo_over_land(ax, t, land, 3)
        labels += [name, sub]
    return labels


def draw_relation_labels(ax, routes, land) -> list[Text]:
    labels = []
    for (subject, verb, obj, (f, nx, ny)), points in zip(
        RELATIONS, routes, strict=True
    ):
        x, y, _ = along(points, f)
        sx, sy = place(subject)
        ox, oy = place(obj)
        sector = round(math.atan2(oy - sy, ox - sx) / (math.pi / 4)) % 8
        label = ax.annotate(
            f"{verb} {ARROWS[sector]}",
            (x, y),
            xytext=(nx, ny),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=12,
            fontstyle="italic",
            color=INK,
            zorder=11,
        )
        halo_over_land(ax, label, land, 4)
        labels.append(label)
    return labels


# --------------------------------------------------------------------------- stages


SLIDE_WIDTH = 14.0  # in: the full-width slide canvas


def base(stage: str, *, wide: bool = False):
    """Draw one static stage; every stage shares the map's size and frame.

    ``wide`` puts the same map, at the same size, on the left of a full-width
    slide canvas, leaving the right for a panel.
    """
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(SLIDE_WIDTH if wide else WIDTH, HEIGHT), dpi=100)
    ax = fig.add_axes((0, 0, WIDTH / SLIDE_WIDTH if wide else 1, 1))
    ax.axis("off")
    ax.set(
        xlim=[v * COS_LAT for v in LON], ylim=LAT, aspect="equal", autoscale_on=False
    )
    land = None
    routes = []
    if stage == "a":
        for town in TOWNS:
            draw_icon(ax, ICON_KIND[town.key], *place(town.key))
    else:
        land = land_patch()
        for patch in land:
            patch.set(facecolor=LAND, edgecolor=COAST, linewidth=1.0, zorder=1)
            ax.add_patch(patch)
        for town in TOWNS:
            x, y = place(town.key)
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
        draw_emblem(ax)
    labels = draw_town_labels(ax, land)
    if stage == "c":
        roads = load_roads()
        for line in roads:
            xs, ys = zip(*line, strict=True)
            ax.plot(xs, ys, color=ROAD, linewidth=1.2, solid_capstyle="round", zorder=2)
        routes = relation_routes(roads)
        for points in routes:
            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, color=INK, linewidth=2.6, solid_capstyle="round", zorder=3)
        labels += draw_relation_labels(ax, routes[: len(RELATIONS)], land)
    return fig, ax, labels, routes


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


STEMS = {
    "a": "icp-map-a-data-systems",
    "b": "icp-map-b-france",
    "c": "icp-map-c-relationships",
    "d": "icp-map-d-live-graph",
    "e": "icp-map-e-many-models",
    "f": "icp-map-f-whole-fleet",
    "g": "icp-map-g-lucy-chat",
}


def save_static(stage: str, draft: Path | None) -> Path:
    fig, _, labels, _ = base(stage)
    if draft is not None:
        check_layout(fig, labels, refuse=False)
        fig.savefig(draft, dpi=100, facecolor="white")
        plt.close(fig)
        return draft
    check_layout(fig, labels)
    FIGURES.mkdir(parents=True, exist_ok=True)
    svg = FIGURES / f"{STEMS[stage]}.svg"
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


# --------------------------------------------------------------------------- traffic
#
# Vehicles only ever travel the relationships: a request enters the graph at a
# node, walks one to three relationships, and shuttles what it read back the way
# it came, like a traversal on a live graph database. The other roads are scenery.

FPS = 15
SPEED = 7.0  # map units per second: a hop across France takes about a second
LOOP_SECONDS = 10.0
PULSE_SECONDS = 0.45  # how long a node glows after it is read
SEED = 11
OTHERS = "#9a9893"  # requests from other users, receding behind the subject's


@dataclass
class Trip:
    start: float  # seconds into the loop
    points: list  # the traversal polyline, out and back
    cum: list  # cumulative length along it
    visits: list  # (distance along the polyline, town) for each node read
    colour: str
    kind: str

    @property
    def duration(self) -> float:
        return self.cum[-1] / SPEED


def relation_graph(routes) -> dict[str, list[tuple[str, list]]]:
    """Each town's neighbours over the relationships, with the road between them."""
    graph: dict[str, list[tuple[str, list]]] = {}
    edges = [*RELATIONS, ("iter", "asks", ON_RAMP, None)]
    for (a, _, b, _), points in zip(edges, routes, strict=True):
        graph.setdefault(a, []).append((b, points))
        graph.setdefault(b, []).append((a, list(reversed(points))))
    return graph


def make_trip(walk: list[str], graph, start: float, colour: str, kind: str) -> Trip:
    """Traverse the walk out and back again, reading each node on the way."""
    legs = []
    for a, b in zip(walk, walk[1:], strict=False):
        legs.append(next(pts for n, pts in graph[a] if n == b))
    legs += [list(reversed(leg)) for leg in reversed(legs)]
    points, cum, visits = [legs[0][0]], [0.0], [(0.0, walk[0])]
    stops = walk[1:] + list(reversed(walk[:-1]))
    for leg, town in zip(legs, stops, strict=True):
        for p in leg[1:]:
            cum.append(cum[-1] + math.dist(points[-1], p))
            points.append(p)
        visits.append((cum[-1], town))
    return Trip(start, points, cum, visits, colour, kind)


def random_walk(graph, rng) -> list[str]:
    """A question's path: ITER, the on-ramp, then one to three relationships on."""
    walk = ["iter", ON_RAMP]
    for _ in range(rng.choice((1, 1, 2, 2, 3))):
        options = [n for n, _ in graph[walk[-1]] if n not in walk]
        if not options:
            break
        walk.append(rng.choice(options))
    return walk


# Each burst is one question: (model colour, vehicle) for the vehicles it sends.
FLEETS = {
    "d": [[(CLAUDE, "car")]],
    "e": [
        [(CLAUDE, "car")],
        [(CODEX, "car")],
        [(LOCAL, "car")],
        [(LOCAL_SECOND, "car")],
    ],
    "f": [
        [(CLAUDE, "car"), (CLAUDE, "bike")],
        [(CODEX, "truck"), (CODEX, "car")],
        [(LOCAL, "car"), (LOCAL, "truck")],
        [(LOCAL_SECOND, "bike"), (LOCAL_SECOND, "car")],
    ],
    "g": [[(OTHERS, "car")]],
}
BURSTS = {"d": 4, "e": 4, "f": 6, "g": 5}  # questions per loop


def traffic(stage: str, graph) -> list[Trip]:
    """Bursts of requests: each question sends a few vehicles out from ITER at once,
    each on its own traversal, and every one brings its answer back to ITER."""
    rng = random.Random(SEED + ord(stage))
    trips = []
    for q in range(BURSTS[stage]):
        fleet = FLEETS[stage][q % len(FLEETS[stage])]
        depart = LOOP_SECONDS * (q + 0.1 + 0.3 * rng.random()) / BURSTS[stage]
        for k in range(rng.randint(3, 5)):
            colour, kind = fleet[k % len(fleet)]
            walk = random_walk(graph, rng)
            trip = make_trip(walk, graph, depart + 0.18 * k, colour, kind)
            if trip.duration < LOOP_SECONDS * 0.9:
                trips.append(trip)
    return trips


def locate(trip: Trip, s: float) -> tuple[float, float, float]:
    i = max(1, min(len(trip.cum) - 1, bisect.bisect_left(trip.cum, s)))
    p, q = trip.points[i - 1], trip.points[i]
    seg = trip.cum[i] - trip.cum[i - 1]
    f = 0.0 if seg == 0 else (s - trip.cum[i - 1]) / seg
    return (
        p[0] + f * (q[0] - p[0]),
        p[1] + f * (q[1] - p[1]),
        math.atan2(q[1] - p[1], q[0] - p[0]),
    )


class TrafficLayer:
    """Vehicles and node pulses for a set of trips, animated on one map axes."""

    def __init__(self, ax, trips: list[Trip], *, loop: float | None, zorder: float = 5):
        self.trips, self.loop = trips, loop
        self.vehicles = []
        for trip in trips:
            body, glass, _ = car_shape(trip.kind)
            pb = Polygon(
                body,
                closed=True,
                facecolor=trip.colour,
                edgecolor="white",
                linewidth=1.1,
                zorder=zorder,
                visible=False,
            )
            pg = Polygon(
                glass,
                closed=True,
                facecolor="white",
                linewidth=0,
                zorder=zorder + 0.1,
                visible=False,
            )
            ax.add_patch(pb)
            ax.add_patch(pg)
            self.vehicles.append((pb, pg))
        self.pulses = {}
        for key in [t.key for t in TOWNS] + ["iter"]:
            ring = Circle(
                place(key),
                0.1,
                facecolor="none",
                linewidth=2.6,
                zorder=6,
                visible=False,
            )
            ax.add_patch(ring)
            self.pulses[key] = ring

    def elapsed(self, trip: Trip, t: float) -> float:
        e = t - trip.start
        return e % self.loop if self.loop else e

    def update(self, t: float) -> None:
        glow: dict[str, tuple[float, str]] = {}
        for trip, (pb, pg) in zip(self.trips, self.vehicles, strict=True):
            e = self.elapsed(trip, t)
            moving = 0 <= e < trip.duration
            pb.set_visible(moving)
            pg.set_visible(moving)
            if moving:
                x, y, heading = locate(trip, e * SPEED)
                body, glass, size = car_shape(trip.kind)
                pb.set_xy(place_shape(body, x, y, heading, size))
                pg.set_xy(place_shape(glass, x, y, heading, size))
            for s, town in trip.visits:
                age = e - s / SPEED
                if 0 <= age < PULSE_SECONDS and (
                    town not in glow or age < glow[town][0]
                ):
                    glow[town] = (age, trip.colour)
        for town, ring in self.pulses.items():
            if town in glow:
                age, colour = glow[town]
                f = age / PULSE_SECONDS
                ring.set(
                    visible=True,
                    radius=(0.45 if town == "iter" else 0.13) + 0.3 * f,
                    alpha=1 - f,
                    edgecolor=colour,
                )
            else:
                ring.set_visible(False)


def save_gif(fig, update, frames: int, stem: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    gif = FIGURES / f"{stem}.gif"
    FuncAnimation(fig, update, frames=frames, blit=False).save(
        gif,
        writer=PillowWriter(fps=FPS),
        dpi=100,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)
    print(
        gif.relative_to(ROOT),
        f"{gif.stat().st_size / 1e6:.1f} MB, {frames / FPS:.0f} s",
    )
    return gif


def save_traffic(stage: str, draft: Path | None) -> Path:
    """Live-graph traffic on the stage (c) map, looping seamlessly."""
    fig, ax, labels, routes = base("c")
    graph = relation_graph(routes)
    layer = TrafficLayer(ax, traffic(stage, graph), loop=LOOP_SECONDS)
    frames = int(LOOP_SECONDS * FPS)
    check_layout(fig, labels)
    if draft is not None:
        layer.update(LOOP_SECONDS * 0.37)
        fig.savefig(draft, dpi=100, facecolor="white")
        plt.close(fig)
        return draft
    return save_gif(fig, lambda f: layer.update(f / FPS), frames, STEMS[stage])


# --------------------------------------------------------------------------- lucy

LUCY_TYPE = 1.6  # seconds to type the question
LUCY_DISPATCH = 2.0  # the request reaches the graph
LUCY_STREAM = 1.4  # seconds to stream the answer
LUCY_HOLD = 3.0
CHAT_GREY = "#ecebe8"

# The question, and the traversals Lucy sends from ITER in parallel to answer it.
LUCY_QUESTION = "How do we close NCR-0421, and what has been done so far?"
LUCY_WALKS = [
    (["iter", "pbs", "doc", "ncr"], "NCR-0421 is at stage 1b; rework agreed"),
    (
        ["iter", "pbs", "doc", "cad", "drw"],
        "rework report v2 approved; as-built drawing not updated",
    ),
    (["iter", "pbs", "doc", "req"], "acceptance requirements met"),
]
LUCY_ANSWER = (
    "Rework is done and evidenced. To close: update the as-built drawing, "
    "then file the stage 2 closure."
)


def save_lucy(draft: Path | None) -> Path:
    """A question typed into Lucy sends traversals into a graph busy with other users."""
    fig, ax, labels, routes = base("c", wide=True)
    graph = relation_graph(routes)
    ours = [
        make_trip(walk, graph, LUCY_DISPATCH + 0.2 * k, CLAUDE, "car")
        for k, (walk, _) in enumerate(LUCY_WALKS)
    ]
    back = max(t.start + t.duration for t in ours)
    total = back + 0.4 + LUCY_STREAM + LUCY_HOLD
    others = TrafficLayer(ax, traffic("g", graph), loop=LOOP_SECONDS, zorder=4.6)
    mine = TrafficLayer(ax, ours, loop=None, zorder=5.2)
    for ring in mine.pulses.values():
        ring.set_zorder(6.2)

    p = fig.add_axes(
        (WIDTH / SLIDE_WIDTH + 0.03, 0.04, 1 - WIDTH / SLIDE_WIDTH - 0.05, 0.92)
    )
    p.axis("off")
    p.set(xlim=(0, 1), ylim=(0, 1))
    texts = [
        p.text(
            0.0,
            0.965,
            "Lucy",
            va="center",
            ha="left",
            fontsize=18,
            fontweight="bold",
            color=INK,
        ),
        p.text(
            0.13,
            0.962,
            "ITER engineering assistant",
            va="center",
            ha="left",
            fontsize=13,
            color=MUTED,
        ),
    ]
    question = wrap(LUCY_QUESTION, 34)
    asked = p.text(
        0.99,
        0.87,
        question,
        va="top",
        ha="right",
        fontsize=14,
        color=INK,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": CHAT_GREY, "edgecolor": "none"},
    )
    status = p.text(
        0.01,
        0.70,
        f"Sending {len(ours)} requests into the graph",
        va="top",
        ha="left",
        fontsize=13,
        fontstyle="italic",
        color=MUTED,
    )
    returns = [
        p.text(
            0.01,
            0.635 - 0.075 * k,
            wrap(f"✓ {found}", 46),
            va="top",
            ha="left",
            fontsize=13,
            color=INK,
            linespacing=1.2,
        )
        for k, (_, found) in enumerate(LUCY_WALKS)
    ]
    reply = wrap(LUCY_ANSWER, 40)
    answer = p.text(
        0.01,
        0.36,
        reply,
        va="top",
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=CLAUDE,
        linespacing=1.3,
    )
    texts += [asked, status, *returns, answer]

    def update(frame: int):
        t = frame / FPS
        others.update(t)
        mine.update(t)
        shown = int(len(question) * min(1.0, max(0.0, t - 0.2) / LUCY_TYPE))
        asked.set_text(question[:shown] or " ")
        asked.set_visible(t > 0.2)
        status.set_visible(t >= LUCY_DISPATCH - 0.3)
        for trip, line in zip(ours, returns, strict=True):
            line.set_visible(t >= trip.start + trip.duration)
        streamed = int(len(reply) * min(1.0, max(0.0, t - back - 0.4) / LUCY_STREAM))
        answer.set_text(reply[:streamed])
        answer.set_visible(streamed > 0)
        return []

    frames = int(total * FPS)
    update(frames - 1)
    if draft is not None:
        check_layout(fig, labels + texts, refuse=False)
        update(int((LUCY_DISPATCH + 0.8) * FPS))
        fig.savefig(draft, dpi=100, facecolor="white")
        plt.close(fig)
        return draft
    check_layout(fig, labels + texts)
    return save_gif(fig, update, frames, STEMS["g"])


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


# --------------------------------------------------------------------------- grid

# Repositories in reading order for the full-width grid: five above, four below.
GRID_ORDER = ["doc", "cad", "ncr", "drw", "ana", "cmp", "pbs", "req", "con"]

# Example pages for the grid, rendered from IDM into a private directory outside
# git (they are internal documents): town -> (page image stem, source, crop box as
# fractions of the page: left, top, right, bottom).
THUMBNAILS = {
    "doc": ("doc-22K5JQ-p1", "22K5JQ v10.1 p1", (0.0, 0.0, 1.0, 0.36), "fit"),
    "cad": ("cad-GMXYZV-p3", "GMXYZV v1.0 p3", (0.21, 0.355, 0.70, 0.575), "fill"),
    "drw": ("drw-PJ35TN-p4", "PJ35TN v3.6 p4", (0.0, 0.0, 0.6, 0.55), "fill"),
    "ana": ("ana-FQXFRE-p56", "FQXFRE v1.2 p56", (0.03, 0.57, 0.50, 0.765), "fill"),
    "ncr": ("ncr-86NKDC-p4", "86NKDC v1.1 p4", (0.07, 0.555, 0.93, 0.762), "fill"),
    "pbs": ("pbs-PLB8KW-p3", "PLB8KW v1.1 p3", (0.03, 0.04, 0.56, 0.86), "fill"),
    "req": ("req-X3Y95E-p13", "X3Y95E v2.2 p13", (0.08, 0.03, 0.92, 0.36), "fill"),
    "cmp": ("cmp-7VLKY2-p12", "7VLKY2 v1.0 p12", (0.05, 0.22, 0.42, 0.955), "fit"),
    "con": ("con-4F6AQ8-p1", "4F6AQ8 v1.0 p1", (0.04, 0.02, 0.96, 0.42), "fit"),
}
PRIVATE = FIGURES / "private"  # gitignored: served by the docs SPA, never pushed


# Contract pages keep their header and lose their body: from this fraction of the
# crop's height down, every line of text becomes a solid bar.
REDACT_FROM: dict[str, float] = {"con": 0.695}


def redact_lines(image, start: float, ink: int = 150, bar: str = "#3a3a3a"):
    """Blank each line of text below ``start`` with a bar spanning its extent."""
    from PIL import ImageDraw

    grey = image.convert("L")
    w, h = grey.size
    px = grey.load()
    dark_rows = [
        y for y in range(int(start * h), h) if sum(px[x, y] < ink for x in range(w)) > 2
    ]
    lines, run = [], []
    for y in dark_rows:
        if run and y > run[-1] + 2:
            lines.append(run)
            run = []
        run.append(y)
    if run:
        lines.append(run)
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for rows in lines:
        cols = [x for x in range(w) if any(px[x, y] < ink for y in rows)]
        if len(rows) < 3 or not cols:
            continue  # a rule or a speck, not a line of text
        draw.rectangle((min(cols), rows[0] - 1, max(cols), rows[-1] + 1), fill=bar)
    return out


def fit_box(image, box_aspect: float):
    """Letterbox an image on white to the box's aspect ratio, showing all of it."""
    from PIL import Image

    w, h = image.size
    size = (w, int(w / box_aspect)) if w / h > box_aspect else (int(h * box_aspect), h)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - w) // 2, (size[1] - h) // 2))
    return canvas


def fill_box(image, box_aspect: float):
    """Centre-crop an image to the box's aspect ratio so it fills the box."""
    w, h = image.size
    if w / h > box_aspect:
        nw = int(h * box_aspect)
        return image.crop(((w - nw) // 2, 0, (w + nw) // 2, h))
    nh = int(w / box_aspect)
    return image.crop((0, 0, w, nh))  # keep the top: titles and logos live there


def grid_figure(thumbnails: Path | None) -> Path:
    """ITER's data repositories as a regular full-width grid of cards."""
    from PIL import Image

    plt.style.use("data-ink")
    fig = plt.figure(figsize=(SLIDE_WIDTH, HEIGHT), dpi=100)
    card_w, gap, image_h = 2.4, 0.3, 2.05
    row_top = (HEIGHT - 0.45, HEIGHT - 0.45 - image_h - 1.25)
    texts = []
    for i, key in enumerate(GRID_ORDER):
        row, col = (0, i) if i < 5 else (1, i - 5)
        per_row = 5 if row == 0 else 4
        left = (SLIDE_WIDTH - per_row * card_w - (per_row - 1) * gap) / 2
        x0, top = left + col * (card_w + gap), row_top[row]
        box = (
            x0 / SLIDE_WIDTH,
            (top - image_h) / HEIGHT,
            card_w / SLIDE_WIDTH,
            image_h / HEIGHT,
        )
        cell = fig.add_axes(box)
        cell.set_xticks([])
        cell.set_yticks([])
        cell.axis("off")
        town = next(t for t in TOWNS if t.key == key)
        source = None
        if thumbnails is not None and key in THUMBNAILS:
            stem, source, (a, b, c, d), mode = THUMBNAILS[key]
            page = Image.open(thumbnails / f"{stem}.png").convert("RGB")
            w, h = page.size
            crop = page.crop((int(a * w), int(b * h), int(c * w), int(d * h)))
            if key in REDACT_FROM:
                crop = redact_lines(crop, REDACT_FROM[key])
            shape = fill_box if mode == "fill" else fit_box
            cell.imshow(shape(crop, card_w / image_h), interpolation="lanczos")
            cell.set_aspect("auto")
        else:
            half = 1.25
            cell.set(
                xlim=(-half * card_w / image_h, half * card_w / image_h),
                ylim=(-half, half),
                aspect="equal",
            )
            draw_icon(cell, ICON_KIND[key], 0.0, 0.0, s=0.62)
        cx = (x0 + card_w / 2) / SLIDE_WIDTH
        name_y = (top - image_h - 0.12) / HEIGHT
        texts.append(
            fig.text(
                cx,
                name_y,
                town.name,
                ha="center",
                va="top",
                fontsize=15,
                fontweight="bold",
                color=INK,
            )
        )
        texts.append(
            fig.text(
                cx,
                name_y - 0.042,
                town.system,
                ha="center",
                va="top",
                fontsize=13,
                color=MUTED,
            )
        )
        if source:
            texts.append(
                fig.text(
                    cx,
                    name_y - 0.078,
                    f"ITER_D_{source}",
                    ha="center",
                    va="top",
                    fontsize=10.5,
                    color=MUTED,
                    family="DejaVu Sans Mono",
                )
            )
    check_layout(fig, texts)
    if thumbnails is None:
        out = FIGURES / "icp-map-repositories.svg"
        fig.savefig(out, format="svg", metadata={"Date": None}, facecolor="white")
        text = out.read_text()
        text = text.replace(f'width="{SLIDE_WIDTH * 72:g}pt"', 'width="31.1cm"', 1)
        text = text.replace(f'height="{HEIGHT * 72:g}pt"', 'height="16cm"', 1)
        out.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")
    else:
        PRIVATE.mkdir(parents=True, exist_ok=True)
        out = PRIVATE / "icp-map-repositories-thumbnails.png"
        fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(out.relative_to(ROOT))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", choices=sorted(STEMS), help="draw one image of the sequence"
    )
    parser.add_argument(
        "--draft",
        type=Path,
        help="write a PNG here, reporting layout faults instead of refusing",
    )
    parser.add_argument("--grid", action="store_true", help="draw the repository grid")
    parser.add_argument(
        "--thumbnails",
        type=Path,
        help="directory of rendered IDM pages; the grid then carries them (private output)",
    )
    args = parser.parse_args()
    if args.grid:
        grid_figure(args.thumbnails)
        return
    for stage in [args.only] if args.only else sorted(STEMS):
        if stage in "abc":
            save_static(stage, args.draft)
        elif stage == "g":
            save_lucy(args.draft)
        else:
            save_traffic(stage, args.draft)


if __name__ == "__main__":
    main()

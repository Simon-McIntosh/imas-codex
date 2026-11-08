"""Illustrate ITER's engineering data as a road map of France, built in stages.

An illustration, not a census. Each town is one body of data ITER owns
(documents, CAD models, drawings, interface sheets, non-conformance reports,
...). The motorways are the links between them: infrastructure ITER builds, pays
for and keeps, which is the organisation's memory. Each named relationship runs
along the real motorways between its two towns and carries a plain ontological
verb (``Documents references CAD Models``). Vehicles are agents' requests: they
only ever travel the relationships, the way a traversal runs on a live graph
database, and shuttle what they read back the way they came. They drive beneath
every label. Whatever the vehicle, the roads stay ours. ITER sits east of the
map, joined to it by one on-ramp: each burst of vehicles is a question leaving
ITER, and each vehicle's return is an answer coming back.

The sequence opens on a full-width grid of the repositories as cards (``--grid``;
with ``--thumbnails`` it carries example IDM pages and is written to a gitignored
``private`` folder, because those pages are internal documents). The stages that
follow share one slide canvas with France centred at one size and position, so
each image builds on the last:

a. the data systems: town names with an icon each, on white;
b. France beneath them, a dot per town and the ITER emblem, no links;
c. the relationships, each along its route with its verb;
d. live traffic: many concurrent requests, one model;
e. the same traffic driven by several models, one colour each;
f. the whole fleet: several models and kinds of vehicle, under heavier load.

Geography: the metropolitan outline is the simplified ``france-geojson``
(Licence Ouverte / Etalab); the roads are Natural Earth 10m major and secondary
highways (public domain), clipped to that outline.

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
from matplotlib.transforms import Bbox

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

LAND = "white"  # France as an outline: no fill, so nothing competes with the links
RELATION = "#8e3a8e"  # every relationship, route and verb: the network ITER owns
COAST = "#b4b1aa"
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
    Town("req", "Requirements", "RQM", 7.75, 48.57, (12, 2, "left")),
    Town("pbs", "Plant Breakdown", "PBS", 5.04, 47.32, (-12, -10, "right")),
    Town("ifc", "Interfaces", "interface sheets", 6.18, 48.69, (-14, -20, "left")),
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
    ("con", "delivers", "doc", (0.28, 0, 0)),
    ("doc", "specifies", "req", (0.3, 0, 0)),
    ("cmp", "is part of", "pbs", (0.4, 0, 0)),
    ("ana", "assesses", "cmp", (0.5, 0, 0)),
    ("doc", "is filed under", "pbs", (0.5, -26, -4)),
    ("ifc", "joins", "pbs", (0.58, -18, 0)),
    ("ifc", "states", "req", (0.6, 0, 0)),
    ("ifc", "constrains", "cad", (0.70, 0, 0)),
]


def project(lon: float, lat: float) -> tuple[float, float]:
    return lon * COS_LAT, lat


def place(key: str) -> tuple[float, float]:
    if key == "iter":
        return project(*ITER)
    town = next(t for t in TOWNS if t.key == key)
    return project(town.lon, town.lat)


# --------------------------------------------------------------------------- roads


SECONDARY_COST = 1.5  # a secondary highway is taken only when it saves real distance
TAKEN_COST = 4.0  # a road another relationship already runs along is avoided
SHARED_LIMIT = 2.0  # map units of taken road beyond which a relationship finds its own


def load_roads() -> list[tuple[list[tuple[float, float]], bool]]:
    """Every highway polyline, flagged True when it is a secondary highway."""
    features = json.loads(ROADS.read_text())["features"]
    return [
        (
            [project(lon, lat) for lon, lat in f["geometry"]["coordinates"]],
            f["properties"]["type"] != "Major Highway",
        )
        for f in features
    ]


Segment = frozenset[tuple[float, float]]


def segments(points) -> set[Segment]:
    return {
        frozenset(((round(p[0], 4), round(p[1], 4)), (round(q[0], 4), round(q[1], 4))))
        for p, q in zip(points, points[1:], strict=False)
    }


class RoadGraph:
    """The highway polylines as a graph, so a relation can follow the roads."""

    def __init__(
        self, roads: list[tuple[list[tuple[float, float]], bool]], snap: float = 0.1
    ):
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
        for line, secondary in roads:
            ids = [node(p) for p in line]
            for a, b in zip(ids, ids[1:], strict=False):
                self.link(a, b, SECONDARY_COST if secondary else 1.0)
            ends += [ids[0], ids[-1]]
        # polylines that meet at a junction rarely share a vertex: join them
        for e in ends:
            for n, p in enumerate(self.nodes):
                if n != e and math.dist(p, self.nodes[e]) < snap:
                    self.link(e, n)

    def link(self, a: int, b: int, cost: float = 1.0) -> None:
        d = cost * math.dist(self.nodes[a], self.nodes[b])
        self.edges.setdefault(a, []).append((b, d))
        self.edges.setdefault(b, []).append((a, d))

    def nearest(self, p: tuple[float, float]) -> int:
        return min(range(len(self.nodes)), key=lambda n: math.dist(self.nodes[n], p))

    def path(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        taken: set[Segment] = frozenset(),
    ) -> list[tuple[float, float]]:
        """Cheapest route from a to b, paying TAKEN_COST on every taken segment."""
        start, goal = self.nearest(a), self.nearest(b)
        queue, best, prev = [(0.0, start)], {start: 0.0}, {}
        while queue:
            dist, n = heapq.heappop(queue)
            if n == goal:
                break
            for m, d in self.edges.get(n, []):
                if taken and segments([self.nodes[n], self.nodes[m]]) <= taken:
                    d *= TAKEN_COST
                if dist + d < best.get(m, math.inf):
                    best[m], prev[m] = dist + d, n
                    heapq.heappush(queue, (dist + d, m))
        if goal not in best:
            raise SystemExit(f"no highway route between {a} and {b}")
        points, n = [], goal
        while n != start:
            points.append(self.nodes[n])
            n = prev[n]
        points.append(self.nodes[start])
        return [a, *reversed(points), b]


def shared_length(points, taken: set[Segment]) -> float:
    return sum(
        math.dist(p, q)
        for p, q in zip(points, points[1:], strict=False)
        if segments([p, q]) <= taken
    )


def relation_routes(roads) -> list[list[tuple[float, float]]]:
    """The road each relationship runs along, then ITER's on-ramp last.

    A relationship takes its shortest motorway route, unless a long stretch of it
    is road earlier relationships already run along: two relationships on one
    road read as one. It then takes its own way over every highway, secondary
    ones included, paying extra for any road already taken."""
    motorways = RoadGraph([(line, False) for line, secondary in roads if not secondary])
    highways = RoadGraph(roads)
    routes, taken = [], set()
    for s, _, o, _ in RELATIONS:
        try:
            route = motorways.path(place(s), place(o))
        except SystemExit:  # no motorway joins them
            route = None
        if route is None or shared_length(route, taken) > SHARED_LIMIT:
            route = highways.path(place(s), place(o), taken)
        routes.append(route)
        taken |= segments(route)
    return [*routes, motorways.path(place("iter"), place(ON_RAMP))]


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
    elif kind == "ifc":
        # a token split in two: each system keeps one half, and the interface
        # is the join between them
        zig = [(0.08, 0.9), (-0.12, 0.45), (0.12, 0.0), (-0.12, -0.45), (0.08, -0.9)]
        gap = 0.1
        poly([(-1, -0.9), *[(u - gap, v) for u, v in reversed(zig)], (-1, 0.9)])
        poly([(1, 0.9), *[(u + gap, v) for u, v in zig], (1, -0.9)], face="#c9c7c2")
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


def heading_at(points, s: float, window: float = 0.1) -> float:
    """The road's local direction at distance ``s``, from a short chord around it."""
    total = length(points)
    x0, y0, _ = along(points, max(0.0, s - window) / total)
    x1, y1, _ = along(points, min(total, s + window) / total)
    return math.atan2(y1 - y0, x1 - x0)


CHEVRON = [(1.0, 0.0), (-0.7, 0.8), (-0.25, 0.0), (-0.7, -0.8)]  # a ">" arrowhead


def draw_relation_labels(ax, routes, others: list[Text]) -> list[Text]:
    """Each verb on its own road, with a ">" beside it on the road pointing from
    subject to object along the road's local direction; both in the highlight.

    An arrowhead takes the nearest place on its road, either side of its verb,
    where it touches no text at all."""
    renderer = ax.figure.canvas.get_renderer()
    labels = []
    for (_, verb, _, (f, nx, ny)), points in zip(RELATIONS, routes, strict=True):
        x, y, _ = along(points, f)
        label = ax.annotate(
            verb,
            (x, y),
            xytext=(nx, ny),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=RELATION,
            zorder=11,
            # a white backing, so the road stops cleanly either side of the verb
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
            },
        )
        labels.append(label)
    boxes = [t.get_window_extent(renderer).padded(3) for t in [*others, *labels]]
    for town in TOWNS:  # and keep clear of the town dots the roads arrive at
        x, y = ax.transData.transform(place(town.key))
        boxes.append(Bbox.from_extents(x - 9, y - 9, x + 9, y + 9))
    arrows = []
    for (_, _, _, (f, _, _)), points in zip(RELATIONS, routes, strict=True):
        total, start = length(points), f * length(points)
        candidates = [  # 0.03 map units at a time, alternating sides
            start + 0.03 * (step // 2 + 1) * (1 if step % 2 == 0 else -1)
            for step in range(120)
        ]
        candidates = [s for s in candidates if 0.2 < s < total - 0.2] or [start]
        for s in candidates:
            xa, ya, _ = along(points, s / total)
            arrow = Polygon(
                place_shape(CHEVRON, xa, ya, heading_at(points, s), 0.17),
                closed=True,
                facecolor=RELATION,
                edgecolor="white",
                linewidth=1.0,
                zorder=10,
            )
            ax.add_patch(arrow)
            extent = arrow.get_window_extent(renderer)
            if not any(extent.overlaps(b) for b in boxes):
                break
            if s != candidates[-1]:
                arrow.remove()
        # with no clear place the last try stays, and the layout check names it
        arrows.append(arrow)
    ax.figure.relation_arrows = arrows
    return labels


# --------------------------------------------------------------------------- stages


SLIDE_WIDTH = 14.0  # in: the full-width slide canvas


def base(stage: str):
    """Draw one stage on the full-width slide canvas with France centred, at one
    size and position, so every image and animation stacks on the last."""
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(SLIDE_WIDTH, HEIGHT), dpi=100)
    ax = fig.add_axes(((1 - WIDTH / SLIDE_WIDTH) / 2, 0, WIDTH / SLIDE_WIDTH, 1))
    ax.axis("off")
    ax.set(
        xlim=[v * COS_LAT for v in LON], ylim=LAT, aspect="equal", autoscale_on=False
    )
    routes = []
    land = land_patch()
    for patch in land:
        patch.set(facecolor=LAND, edgecolor=COAST, linewidth=1.0, zorder=1)
        ax.add_patch(patch)
    if stage == "a":
        for town in TOWNS:
            draw_icon(ax, ICON_KIND[town.key], *place(town.key))
    else:
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
        routes = relation_routes(load_roads())
        for points in routes:
            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, color=INK, linewidth=3.0, solid_capstyle="round", zorder=3)
        labels += draw_relation_labels(ax, routes[: len(RELATIONS)], labels)
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
    for arrow in getattr(fig, "relation_arrows", []):
        extent = arrow.get_window_extent(renderer)
        faults += [
            f"arrowhead on text: {t.get_text()!r}"
            for t, b in boxes
            if extent.overlaps(b)
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
    text = text.replace(f'width="{SLIDE_WIDTH * 72:g}pt"', 'width="31.1cm"', 1)
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
}
BURSTS = {"d": 4, "e": 4, "f": 6}  # questions per loop


def shortest_walk(graph, start: str, target: str) -> list[str]:
    """The fewest relationships from ``start`` to ``target``, as a list of towns."""
    previous, frontier = {start: None}, [start]
    while frontier and target not in previous:
        reached = []
        for town in frontier:
            for n, _ in graph[town]:
                if n not in previous:
                    previous[n] = town
                    reached.append(n)
        frontier = reached
    walk, town = [], target
    while town is not None:
        walk.append(town)
        town = previous[town]
    return list(reversed(walk))


def planned_walk(graph, rng, target: str) -> list[str]:
    """A question aimed at one repository, sometimes reading one step beyond it."""
    walk = ["iter", *shortest_walk(graph, ON_RAMP, target)]
    onward = [n for n, _ in graph[target] if n not in walk]
    if onward and rng.random() < 0.35:
        walk.append(rng.choice(onward))
    return walk


def traffic(stage: str, graph) -> list[Trip]:
    """Bursts of requests: each question sends a few vehicles out from ITER at once,
    each on its own traversal, and every one brings its answer back to ITER.

    The traffic is staged: within each loop the questions are aimed so that every
    repository is read at least once, and the loop is refused if one is missed."""
    rng = random.Random(SEED + ord(stage))
    unread = sorted(t.key for t in TOWNS if t.key != ON_RAMP)
    rng.shuffle(unread)
    per_burst = math.ceil(len(unread) / BURSTS[stage])
    trips = []
    for q in range(BURSTS[stage]):
        fleet = FLEETS[stage][q % len(FLEETS[stage])]
        depart = LOOP_SECONDS * (q + 0.1 + 0.3 * rng.random()) / BURSTS[stage]
        aimed = 0
        for k in range(rng.randint(3, 5)):
            colour, kind = fleet[k % len(fleet)]
            if unread and aimed < per_burst:
                walk = planned_walk(graph, rng, unread[0])
                aimed += 1
            else:
                walk = random_walk(graph, rng)
            trip = make_trip(walk, graph, depart + 0.18 * k, colour, kind)
            if trip.duration < LOOP_SECONDS * 0.9:
                trips.append(trip)
                unread = [t for t in unread if t not in walk]
    read = {town for trip in trips for _, town in trip.visits}
    missed = sorted({t.key for t in TOWNS} - read)
    if missed:
        raise SystemExit(f"traffic for stage {stage} never reads: {', '.join(missed)}")
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


# --------------------------------------------------------------------------- grid

# Names too wide for a card at presentation size, broken over two lines.
GRID_NAMES = {"ncr": "Non-Conformance\nReports"}
GRID_SYSTEMS = {"cad": "CATIA\nDassault Systèmes"}

# Repositories in reading order for the full-width grid: five above, five below.
GRID_ORDER = ["doc", "cad", "ncr", "drw", "ana", "cmp", "pbs", "ifc", "req", "con"]

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
    "ifc": ("ifc-F7TX5Y-p9", "F7TX5Y v1.10 p9", (0.14, 0.145, 0.86, 0.44), "fit"),
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
    row_top = (HEIGHT - 0.3, HEIGHT - 0.3 - image_h - 1.5)
    texts = []
    for i, key in enumerate(GRID_ORDER):
        row, col = (0, i) if i < 5 else (1, i - 5)
        per_row = 5 if row == 0 else len(GRID_ORDER) - 5
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
        if thumbnails is not None and key in THUMBNAILS:
            stem, _, (a, b, c, d), mode = THUMBNAILS[key]
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
        name = GRID_NAMES.get(key, town.name)
        lines = name.count("\n") + 1
        texts.append(
            fig.text(
                cx,
                name_y,
                name,
                ha="center",
                va="top",
                fontsize=20,
                fontweight="bold",
                color=INK,
                linespacing=1.1,
            )
        )
        texts.append(
            fig.text(
                cx,
                name_y - 0.052 * lines - 0.006,
                GRID_SYSTEMS.get(key, town.system),
                ha="center",
                va="top",
                fontsize=16,
                linespacing=1.15,
                color=MUTED,
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
        else:
            save_traffic(stage, args.draft)


if __name__ == "__main__":
    main()

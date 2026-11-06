"""Illustrate ITER's engineering data as a road map of France, built in stages.

An illustration, not a census. Each town is one body of data ITER owns
(documents, CAD models, drawings, non-conformance reports, ...). The motorways
are the links between them: infrastructure ITER builds, pays for and keeps,
which is the organisation's memory. Each named relationship runs along the real
motorways between its two towns and carries a plain ontological verb
(``Documents references CAD Models``). Cars are agents carrying information
between the towns; they drive beneath every label, because they are a device
for movement and not content. Whatever the car, the roads stay ours.

The build sequence shares one canvas and one frame, so the images stack:

a. the data systems: town names with an icon each, on white;
b. France beneath them, a dot per town and the ITER emblem, no links;
c. the road network with each relationship and its verb;
d. an animation of (c) with bursts of single-model traffic;
e. the same traffic driven by several models, one colour each;
f. the same with several kinds of vehicle as well.

Geography: the metropolitan outline is the simplified ``france-geojson``
(Licence Ouverte / Etalab); the roads are Natural Earth 10m major highways
(public domain), clipped to that outline. The emblem is ITER's own, from
iter.org, for internal presentations.

Usage::

    uv run python scripts/figures_icp_knowledge_map.py            # every image
    uv run python scripts/figures_icp_knowledge_map.py --only c   # one image
    uv run python scripts/figures_icp_knowledge_map.py --only c --draft /tmp/x.png
"""

from __future__ import annotations

import argparse
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
MUTED = "#6b6b6b"
EMBLEM_YELLOW = "#FDBB30"  # ITER Noon Yellow, as the emblem file carries it

# One colour per model family driving the agents.
CLAUDE = "#D97757"
CODEX = "#141414"  # OpenAI's brand is black and white; Codex carries no hue of its own
LOCAL = "#4d6bfe"  # a locally served open-weight model


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

ARROWS = "→↗↑↖←↙↓↘"

# Animation: a seamless loop of bursts of traffic.
FPS, LOOP_SECONDS = 15, 8.0
SPEED = 2.4  # map units per second
SEED = 7


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
    graph = RoadGraph(roads)
    return [graph.path(place(s), place(o)) for s, _, o, _ in RELATIONS]


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
        ax.add_patch(Circle((x, y), 0.3 * s, facecolor=LAND, **style))
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
        body = [(-1.6, -0.55), (0.55, -0.55), (0.55, 0.55), (-1.6, 0.55)]
        glass = [(0.75, -0.42), (1.05, -0.38), (1.05, 0.38), (0.75, 0.42)]
        return (
            body + [(0.62, 0.5), (1.15, 0.45), (1.15, -0.45), (0.62, -0.5)],
            glass,
            0.2,
        )
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


def base(stage: str):
    """Draw one static stage; every stage shares the canvas and the frame."""
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
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
    if stage in "cdef":
        roads = load_roads()
        for line in roads:
            xs, ys = zip(*line, strict=True)
            ax.plot(xs, ys, color=ROAD, linewidth=1.2, solid_capstyle="round", zorder=2)
        routes = relation_routes(roads)
        for points in routes:
            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, color=INK, linewidth=2.6, solid_capstyle="round", zorder=3)
        labels += draw_relation_labels(ax, routes, land)
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
    "d": "icp-map-d-traffic",
    "e": "icp-map-e-many-models",
    "f": "icp-map-f-many-vehicles",
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


@dataclass(frozen=True)
class Trip:
    route: int
    depart: float  # seconds into the loop
    forward: bool
    colour: str
    kind: str


def schedule(stage: str) -> list[Trip]:
    """Bursts of trips: a few cars leave together, then the road goes quiet."""
    rng = random.Random(SEED)
    fleet = {
        "d": [(CLAUDE, "car")],
        "e": [(CLAUDE, "car"), (CODEX, "car"), (LOCAL, "car")],
        "f": [
            (CLAUDE, "car"),
            (CODEX, "car"),
            (LOCAL, "car"),
            (CLAUDE, "truck"),
            (CODEX, "bike"),
            (LOCAL, "truck"),
            (CLAUDE, "bike"),
        ],
    }[stage]
    trips = []
    for route in range(len(RELATIONS)):
        bursts = {"d": (1, 2, 2), "e": (2, 2, 3), "f": (2, 3, 3)}[stage]
        for _ in range(rng.choice(bursts)):
            start, forward = rng.uniform(0, LOOP_SECONDS), rng.random() < 0.6
            colour, kind = rng.choice(fleet)
            for k in range(rng.randint(2, 5)):
                if stage != "d" and rng.random() < 0.5:
                    colour, kind = rng.choice(fleet)
                trips.append(
                    Trip(
                        route, (start + 0.28 * k) % LOOP_SECONDS, forward, colour, kind
                    )
                )
    return trips


def save_animation(stage: str, draft: Path | None) -> Path:
    fig, ax, labels, routes = base(stage)
    check_layout(fig, labels)
    lengths = [length(r) for r in routes]
    trips = schedule(stage)
    patches = []
    for trip in trips:
        body, glass, _ = car_shape(trip.kind)
        edge = "white"
        p_body = Polygon(
            body,
            closed=True,
            facecolor=trip.colour,
            edgecolor=edge,
            linewidth=1.1,
            zorder=5,
            visible=False,
        )
        p_glass = Polygon(
            glass,
            closed=True,
            facecolor="white",
            linewidth=0,
            zorder=5.5,
            visible=False,
        )
        ax.add_patch(p_body)
        ax.add_patch(p_glass)
        patches.append((p_body, p_glass))
    frames = int(FPS * LOOP_SECONDS)

    def update(frame: int):
        t = frame / FPS
        for trip, (p_body, p_glass) in zip(trips, patches, strict=True):
            duration = lengths[trip.route] / SPEED
            elapsed = (t - trip.depart) % LOOP_SECONDS
            moving = elapsed < duration
            p_body.set_visible(moving)
            p_glass.set_visible(moving)
            if not moving:
                continue
            f = elapsed / duration
            x, y, heading = along(routes[trip.route], f if trip.forward else 1 - f)
            if not trip.forward:
                heading += math.pi
            body, glass, size = car_shape(trip.kind)
            p_body.set_xy(place_shape(body, x, y, heading, size))
            p_glass.set_xy(place_shape(glass, x, y, heading, size))
        return [p for pair in patches for p in pair]

    if draft is not None:
        update(int(frames * 0.4))
        fig.savefig(draft, dpi=100, facecolor="white")
        plt.close(fig)
        return draft
    FIGURES.mkdir(parents=True, exist_ok=True)
    gif = FIGURES / f"{STEMS[stage]}.gif"
    FuncAnimation(fig, update, frames=frames, blit=False).save(
        gif,
        writer=PillowWriter(fps=FPS),
        dpi=100,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)
    print(gif.relative_to(ROOT), f"{gif.stat().st_size / 1e6:.1f} MB")
    return gif


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
    args = parser.parse_args()
    for stage in [args.only] if args.only else sorted(STEMS):
        if stage in "abc":
            save_static(stage, args.draft)
        else:
            save_animation(stage, args.draft)


if __name__ == "__main__":
    main()

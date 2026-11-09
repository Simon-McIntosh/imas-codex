"""Draw ITER's AI stack as a pyramid, from the infrastructure ITER owns up to its users.

An illustration of an architecture, not a census. Reading from the bottom:

- **ICP4**, the existing engineering and plant data systems;
- the **knowledge graph** built on top of them by ETL, which itself becomes ITER
  infrastructure: durable, owned, and drawn in the relation colour of the ICP
  knowledge-map figures, whose ringed nodes it reuses;
- a **function layer** of read-only tools over the graph (Lucy tools, imas-codex
  tools), third-party connectors (Outlook, Jira, Confluence) to the left, and a
  REPL to the right as a code-generation route for whatever the tools do not cover;
- the **interfaces** people use: Claude Desktop, Lucy chat, domain dashboards, a
  sandbox, and Clive, a Claude orchestrator. Every interface but Claude Desktop can
  also run on the **local lane**, open-weight models served on ITER's own GPUs;
- the **users**, one bar per role, spanning the interfaces that role reaches;
- a cap of **tokens per month** through each interface, on a log scale, split into
  frontier and local models.

Colour separates what is deterministic (purple, the infrastructure ITER owns) from
what is agentic: coral where a frontier model drives it, blue for the local lane.
The usage numbers are sized from the headcounts below to show proportion, not
measured.

Usage::

    uv run python scripts/figures_ai_stack.py
    uv run python scripts/figures_ai_stack.py --draft /tmp/x.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/ai-stack"
STEM = "ai-stack"

WIDTH, HEIGHT = 14.0, 7.0  # in: wider than the slide canvas, leaving room for a title
U = 10  # drawing units per inch, so the axes run 0..140 by 0..70

# Deterministic: the infrastructure ITER owns, in the ICP knowledge map's relation colour.
DETERMINISTIC = "#8e3a8e"
DETERMINISTIC_FILL = "#f6eef6"
# Agentic, driven by a frontier model: the ephemeral layer that runs over the graph.
FRONTIER = "#d9622b"
FRONTIER_FILL = "#fcefe8"
# Agentic on the local lane: open-weight models on ITER's GPUs, the ICP map's local blue.
LOCAL = "#4d6bfe"
USER = "#6b6b6b"
INK = "#2b2b2b"
MUTED = "#5f5f5f"
NEUTRAL_EDGE = "#b4b1aa"
BAND_FILL = "#f4f3f0"

TITLE, BODY, SMALL = 16, 14, 13  # pt, inside the stack
OUTER = 18  # pt, the layer names
OUTER_SMALL = 14  # pt, the legend and the annotations outside the boxes
PT = 7.2  # points per drawing unit


@dataclass
class Placed:
    """A text artist and, optionally, the box it must stay inside."""

    text: plt.Text
    inside: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 in data


class Canvas:
    def __init__(self):
        plt.style.use("data-ink")
        # Arial where it is installed; Nimbus Sans shares its metrics, so the layout
        # check measures the same widths, and the SVG keeps its text as text in Arial.
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Nimbus Sans"]
        plt.rcParams["svg.fonttype"] = "none"
        self.fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=100)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.axis("off")
        self.ax.set(xlim=(0, WIDTH * U), ylim=(0, HEIGHT * U), autoscale_on=False)
        self.placed: list[Placed] = []

    def box(self, x0, y0, x1, y1, *, edge, fill="white", lw=1.6, dashed=False, z=2):
        self.ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle="round,pad=0,rounding_size=0.9",
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
                linestyle=(0, (4, 3)) if dashed else "solid",
                zorder=z,
            )
        )

    def text(self, x, y, s, *, size=BODY, inside=None, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        t = self.ax.text(x, y, s, fontsize=size, zorder=10, **kw)
        self.placed.append(Placed(t, inside))
        return t

    def card(
        self, x0, y0, x1, y1, title, subtitle=None, *, kind, size=TITLE, weight="bold"
    ):
        edge, fill = (
            (FRONTIER, FRONTIER_FILL)
            if kind == "agentic"
            else (DETERMINISTIC, DETERMINISTIC_FILL)
        )
        self.box(x0, y0, x1, y1, edge=edge, fill=fill)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inside = (x0, y0, x1, y1)
        if subtitle is None:
            self.text(cx, cy, title, size=size, inside=inside, weight=weight)
            return
        # stack the title above the subtitle, centred as one block
        title_h = (title.count("\n") + 1) * size * 1.2 / PT
        sub_h = (subtitle.count("\n") + 1) * SMALL * 1.1 / PT
        gap = 0.4
        top = cy + (title_h + gap + sub_h) / 2
        self.text(cx, top, title, size=size, inside=inside, weight=weight, va="top")
        self.text(
            cx,
            top - title_h - gap,
            subtitle,
            size=SMALL,
            inside=inside,
            color=MUTED,
            linespacing=1.1,
            va="top",
        )

    def chip(self, x0, y0, x1, y1, label, *, kind):
        edge = FRONTIER if kind == "agentic" else DETERMINISTIC
        self.box(x0, y0, x1, y1, edge=edge, fill="white", lw=1.2, z=4)
        self.text(
            (x0 + x1) / 2, (y0 + y1) / 2, label, size=SMALL, inside=(x0, y0, x1, y1)
        )

    def check_layout(self, *, refuse: bool = True) -> None:
        """Refuse text that leaves the canvas, collides, or escapes its box."""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        to_data = self.ax.transData.inverted()
        boxes = [(p, p.text.get_window_extent(renderer)) for p in self.placed]
        faults = [
            f"off canvas: {p.text.get_text()!r}"
            for p, b in boxes
            if b.x0 < 0
            or b.y0 < 0
            or b.x1 > self.fig.bbox.x1
            or b.y1 > self.fig.bbox.y1
        ]
        faults += [
            f"overlap: {p1.text.get_text()!r} / {p2.text.get_text()!r}"
            for (p1, b1), (p2, b2) in combinations(boxes, 2)
            if b1.overlaps(b2)
        ]
        for p, b in boxes:
            if p.inside is None:
                continue
            (x0, y0), (x1, y1) = to_data.transform([(b.x0, b.y0), (b.x1, b.y1)])
            bx0, by0, bx1, by1 = p.inside
            px, py = 0.4, 0.1  # clear margin to the box edge, in drawing units
            if x0 < bx0 + px or x1 > bx1 - px or y0 < by0 + py or y1 > by1 - py:
                faults.append(f"escapes its box: {p.text.get_text()!r}")
        if faults:
            message = "layout check refused the figure:\n  " + "\n  ".join(faults)
            if refuse:
                raise SystemExit(message)
            print(message)


# --------------------------------------------------------------------------- layout
#
# The stack is a pyramid: each layer is a band whose sides slope inward as it rises,
# so the infrastructure is the widest base and the users sit at the narrow top. The
# layer names sit in the right margin, beside their band.

CENTRE, BASE_HALF, SLOPE = 64.0, 58.5, 0.2  # half-width = BASE_HALF - SLOPE * y
LAYER_X = 124.0

LAYERS = {  # (y0, y1) of each band
    "icp4": (0.6, 5.8),
    "graph": (8.6, 14.8),
    "functions": (15.8, 28.0),
    "interfaces": (29.0, 41.4),
    "users": (42.2, 60.6),
}
USAGE = (61.6, 68.6)  # the tokens-per-month cap above the users

LAYER_NAMES = {
    "users": ["users"],
    "interfaces": ["interfaces"],
    "functions": ["functions"],
    "graph": ["knowledge", "graph"],
    "icp4": ["ICP4"],
}

# Top to bottom: each role reaches from Claude Desktop to the interface named, and
# no one stops at Claude Desktop, because every role needs ITER's data through Lucy.
USERS = [
    ("project support", 10, "clive"),
    ("prototyping with domain experts", 40, "sandbox"),
    ("code developer", 150, "sandbox"),
    ("commissioning engineer", 300, "dash"),
    ("planning", 80, "dash"),
    ("configuration control", 60, "lucy"),
    ("administrative support", 300, "lucy"),
]

# Tokens per month through each interface, (frontier, local), sized from the roles
# above: administrative use is occasional and light; code developers and project
# support are the heavy users, on both lanes; dashboards and Lucy run mostly local.
USAGE_TOKENS = {
    "desktop": (2.0e9, 0.0),
    "lucy": (6.0e8, 4.0e9),
    "dash": (3.0e8, 8.0e9),
    "sandbox": (2.5e10, 1.5e10),
    "clive": (1.5e10, 6.0e10),
}
DECADES = (8, 11)  # the log axis runs 10^8 .. 10^11 tokens per month

LUCY_TOOLS = [
    ("ask Lucy", "agentic"),
    ("search records", "deterministic"),
    ("get records", "deterministic"),
    ("relationships", "deterministic"),
    ("versions", "deterministic"),
    ("abbreviations", "deterministic"),
]

CODEX_TOOLS = ["signals", "DD paths", "standard names", "facility docs"]

SOURCES = ["IDM", "CATIA", "SMDD", "NCR", "SAP", "RQM", "PBS", "UDA", "DAN", "DS"]

INTERFACES: dict[str, tuple[float, float]] = {}


def span(y: float, pad: float = 1.4) -> tuple[float, float]:
    """The usable x range inside the pyramid at height y."""
    half = BASE_HALF - SLOPE * y - pad
    return CENTRE - half, CENTRE + half


def spread(x0: float, x1: float, widths: list[float], gap: float = 1.0):
    """Lay boxes of the given relative widths across x0..x1, gap apart."""
    scale = (x1 - x0 - gap * (len(widths) - 1)) / sum(widths)
    out, x = [], x0
    for w in widths:
        out.append((x, x + w * scale))
        x += w * scale + gap
    return out


def grid(c: Canvas, x0, x1, top, names, cols, *, height=2.6, pitch=3.1, gap=1.0):
    """Lay chips in rows of `cols` from x0 to x1, the first row's top at `top`."""
    w = (x1 - x0 - (cols - 1) * gap) / cols
    for i, (name, kind) in enumerate(names):
        r, col = divmod(i, cols)
        x = x0 + col * (w + gap)
        y = top - r * pitch
        c.chip(x, y - height, x + w, y, name, kind=kind)


def draw_bands(c: Canvas) -> None:
    for y0, y1 in LAYERS.values():
        (a0, a1), (b0, b1) = span(y0, 0), span(y1, 0)
        c.ax.add_patch(
            Polygon(
                [(a0, y0), (a1, y0), (b1, y1), (b0, y1)],
                closed=True,
                facecolor=BAND_FILL,
                edgecolor="none",
                zorder=0,
            )
        )


def draw_layer_names(c: Canvas) -> None:
    for key, lines in LAYER_NAMES.items():
        y0, y1 = LAYERS[key]
        mid = (y0 + y1) / 2
        for i, line in enumerate(lines):
            dy = (len(lines) - 1) / 2 * 3.0 - i * 3.0
            c.text(LAYER_X, mid + dy, line, size=OUTER, ha="left", weight="bold")


def draw_foundation(c: Canvas) -> None:
    y0, y1 = LAYERS["icp4"]
    y0, y1 = y0 + 0.7, y1 - 0.7
    g0 = LAYERS["graph"][0] + 0.6
    for (x, xe), system in zip(
        spread(*span(y1), [1] * len(SOURCES)), SOURCES, strict=True
    ):
        c.box(x, y0, xe, y1, edge=DETERMINISTIC, fill="white", lw=1.2, z=4)
        c.text(
            (x + xe) / 2,
            (y0 + y1) / 2,
            system,
            size=BODY,
            weight="bold",
            inside=(x, y0, xe, y1),
        )
        # ETL: each source fans into the graph
        xm = (x + xe) / 2
        c.ax.plot(
            [xm, CENTRE + (xm - CENTRE) * 0.6],
            [y1, g0],
            color=DETERMINISTIC,
            lw=1.2,
            alpha=0.6,
            zorder=1,
        )
    c.text(
        CENTRE,
        (y1 + g0) / 2,
        "ETL",
        size=OUTER_SMALL,
        weight="bold",
        color=DETERMINISTIC,
        backgroundcolor="white",
    )
    # the graph becomes ITER infrastructure, alongside ICP4
    bottom, top = LAYERS["icp4"][0], LAYERS["graph"][1]
    xb = 4.8
    c.ax.plot(
        [xb + 1, xb, xb, xb + 1],
        [bottom, bottom, top, top],
        color=DETERMINISTIC,
        lw=1.6,
        zorder=3,
    )
    c.text(
        xb - 2.5,
        (bottom + top) / 2,
        "ITER\ninfrastructure",
        size=OUTER_SMALL,
        rotation=90,
        color=DETERMINISTIC,
        weight="bold",
        linespacing=1.0,
    )


def draw_graph(c: Canvas) -> None:
    y0, y1 = LAYERS["graph"]
    y0, y1 = y0 + 0.6, y1 - 0.6
    x0, x1 = span(y1)
    c.box(x0, y0, x1, y1, edge=DETERMINISTIC, fill=DETERMINISTIC_FILL, lw=2.0)
    c.text(
        x0 + 1.5,
        (y0 + y1) / 2,
        "ITER knowledge graph",
        size=TITLE,
        weight="bold",
        ha="left",
        color=DETERMINISTIC,
        inside=(x0, y0, x1, y1),
    )
    c.text(
        x0 + 30.0,
        (y0 + y1) / 2,
        "schema · data · knowledge",
        size=SMALL,
        color=MUTED,
        ha="left",
        inside=(x0, y0, x1, y1),
    )
    # a strip of the network, drawn as the ICP knowledge map draws its nodes and links
    lo, hi = y0 + 1.3, y1 - 1.3
    n = 11
    xs = [x0 + 56 + i * (x1 - x0 - 59) / (n - 1) for i in range(n)]
    nodes = [(x, hi if i % 2 == 0 else lo) for i, x in enumerate(xs)]
    edges = [(i, i + 1) for i in range(n - 1)]
    edges += [(i, i + 2) for i in range(0, n - 2, 2)] + [(3, 5), (7, 9)]
    for a, b in edges:
        (xa, ya), (xb, yb) = nodes[a], nodes[b]
        c.ax.plot(
            [xa, xb],
            [ya, yb],
            color=DETERMINISTIC,
            lw=2.0,
            zorder=4,
            solid_capstyle="round",
        )
    for i, (x, y) in enumerate(nodes):
        hub = i in (4, 7)
        c.ax.plot(
            x,
            y,
            "o",
            markersize=10 if hub else 7,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=2.0,
            zorder=6,
        )
        if hub:
            c.ax.plot(x, y, "o", markersize=3.5, color=INK, zorder=7)


def draw_functions(c: Canvas) -> None:
    y0, y1 = LAYERS["functions"]
    y0, y1 = y0 + 0.8, y1 - 0.8
    cx, lx, ix, rx = spread(*span(y1), [12.0, 49.5, 32.5, 9.5])
    pad = 0.8
    header = 2.6

    # third-party connectors, to the left
    cx0, cx1 = cx
    c.box(cx0, y0, cx1, y1, edge=NEUTRAL_EDGE, fill="white", lw=1.2, dashed=True)
    connectors = [("Outlook", "d"), ("Jira", "d"), ("Confluence", "d")]
    grid(c, cx0 + pad, cx1 - pad, y1 - pad, connectors, 1, height=2.5, pitch=3.0)

    # Lucy tools: what Lucy chat calls, and where other assistants plug into the graph
    for (gx0, gx1), title, tools, cols in [
        (lx, "Lucy tools", LUCY_TOOLS, 3),
        (ix, "imas-codex tools", [(t, "d") for t in CODEX_TOOLS], 2),
    ]:
        c.box(gx0, y0, gx1, y1, edge=NEUTRAL_EDGE, fill="white", lw=1.2, dashed=True)
        c.text(
            gx0 + 1.0,
            y1 - header / 2 - 0.3,
            title,
            size=BODY,
            weight="bold",
            ha="left",
            inside=(gx0, y0, gx1, y1),
        )
        grid(
            c,
            gx0 + pad,
            gx1 - pad,
            y1 - header - 0.5,
            tools,
            cols,
            height=2.8,
            pitch=3.4,
        )

    # the REPL: generated code, run against the graph, for whatever no tool covers
    c.card(rx[0], y0, rx[1], y1, "REPL", "code\ngen", kind="agentic")

    # read-only access from the function layer down into the graph
    xa = span(y0)[0] + 32
    g1 = LAYERS["graph"][1] - 0.6
    c.ax.annotate(
        "",
        xy=(xa, g1 - 0.1),
        xytext=(xa, y0),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4, "mutation_scale": 14},
        zorder=8,
    )
    c.text(
        xa + 1.0, (g1 + y0) / 2, "read-only", size=OUTER_SMALL, color=MUTED, ha="left"
    )


def draw_interfaces(c: Canvas) -> None:
    y0, y1 = LAYERS["interfaces"]
    lane0, lane1 = y0 + 0.6, y0 + 2.9
    y0, y1 = lane1 + 0.9, y1 - 0.6
    cols = spread(*span(y1), [11.5, 15, 33, 14, 15])
    for key, xs in zip(
        ["desktop", "lucy", "dash", "sandbox", "clive"], cols, strict=True
    ):
        INTERFACES[key] = xs
    x0, x1 = INTERFACES["desktop"]
    c.card(x0, y0, x1, y1, "Claude\nDesktop", None, kind="agentic")
    x0, x1 = INTERFACES["lucy"]
    c.card(x0, y0, x1, y1, "Lucy chat", "engineering\nassistant", kind="agentic")
    x0, x1 = INTERFACES["dash"]
    c.box(x0, y0, x1, y1, edge=NEUTRAL_EDGE, fill="white", lw=1.2, dashed=True)
    c.text(
        (x0 + x1) / 2,
        y1 - 1.4,
        "domain dashboards",
        size=SMALL,
        color=MUTED,
        inside=(x0, y0, x1, y1),
    )
    mid = (x0 + x1) / 2
    for (a, b), label, kind in [
        ((x0 + 0.8, mid - 0.5), "commissioning", "deterministic"),
        ((mid + 0.5, x1 - 0.8), "overnight digest", "agentic"),
    ]:
        c.card(
            a,
            y0 + 0.7,
            b,
            y1 - 2.8,
            label,
            None,
            kind=kind,
            size=SMALL,
            weight="normal",
        )
    x0, x1 = INTERFACES["sandbox"]
    c.card(x0, y0, x1, y1, "Sandbox", "full\ndevelopment", kind="agentic")
    x0, x1 = INTERFACES["clive"]
    c.card(x0, y0, x1, y1, "Clive", "Claude\norchestrator", kind="agentic")

    # the local lane: every interface but Claude Desktop can run on ITER's own GPUs
    lx0, lx1 = INTERFACES["lucy"][0], INTERFACES["clive"][1]
    c.ax.add_patch(
        FancyBboxPatch(
            (lx0, lane0),
            lx1 - lx0,
            lane1 - lane0,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=LOCAL,
            edgecolor="none",
            zorder=3,
        )
    )
    c.text(
        (lx0 + lx1) / 2,
        (lane0 + lane1) / 2,
        "local lane · H200 + B300 GPUs · DeepSeek v4.1 Flash",
        size=SMALL,
        color="white",
        weight="bold",
        inside=(lx0, lane0, lx1, lane1),
    )
    for key in ["lucy", "dash", "sandbox", "clive"]:
        a, b = INTERFACES[key]
        taps = [(a + b) / 2] if key != "dash" else [a + (b - a) / 4, b - (b - a) / 4]
        for x in taps:
            c.ax.plot([x, x], [lane1, y0], color=LOCAL, lw=2.6, zorder=2)


def draw_users(c: Canvas) -> None:
    _, y1 = LAYERS["users"]
    pitch, height = 2.6, 2.4
    top = y1 - 0.4
    for i, (role, count, reach) in enumerate(USERS):
        y = top - i * pitch
        edge0, edge1 = span(y, 0.6)
        left = max(INTERFACES["desktop"][0], edge0)
        x1 = min(INTERFACES[reach][1], edge1)
        c.box(left, y - height, x1, y, edge=USER, fill=USER, lw=0, z=3)
        inside = (left, y - height, x1, y)
        c.text(
            left + 1.0,
            y - height / 2,
            role,
            size=SMALL,
            color="white",
            ha="left",
            inside=inside,
        )
        c.text(
            x1 - 1.0,
            y - height / 2,
            f"{count}",
            size=SMALL,
            color="white",
            ha="right",
            inside=inside,
        )


def usage_y(tokens: float) -> float:
    lo, hi = DECADES
    y0, y1 = USAGE
    return y0 + (math.log10(tokens) - lo) / (hi - lo) * (y1 - y0)


def draw_usage(c: Canvas) -> None:
    """Tokens per month through each interface, frontier beside local, log scaled."""
    y0, _ = USAGE
    edge0, edge1 = span(y0, 0.6)
    for key, (x0, x1) in INTERFACES.items():
        x0, x1 = max(x0, edge0), min(x1, edge1)
        frontier, local = USAGE_TOKENS[key]
        mid = (x0 + x1) / 2
        for (a, b), tokens, colour in [
            ((x0, mid - 0.3), frontier, FRONTIER),
            ((mid + 0.3, x1), local, LOCAL),
        ]:
            if tokens <= 0:
                continue
            c.ax.add_patch(
                Rectangle(
                    (a, y0),
                    b - a,
                    usage_y(tokens) - y0,
                    facecolor=colour,
                    edgecolor="none",
                    zorder=3,
                )
            )
    left = max(INTERFACES["desktop"][0], edge0)
    right = min(INTERFACES["clive"][1], edge1)
    c.ax.plot([left, right], [y0, y0], color=INK, lw=1.2, zorder=4)
    lo, hi = DECADES
    for d in range(lo, hi + 1):
        y = usage_y(10.0**d)
        c.ax.plot([left - 0.8, left], [y, y], color=INK, lw=1.2, zorder=4)
        c.text(left - 1.2, y, f"$10^{{{d}}}$", size=OUTER_SMALL - 1, ha="right")
    c.ax.plot([left, left], [y0, usage_y(10.0**hi)], color=INK, lw=1.2, zorder=4)
    mid = (USAGE[0] + USAGE[1]) / 2
    c.text(LAYER_X, mid + 1.5, "tokens", size=OUTER, ha="left", weight="bold")
    c.text(LAYER_X, mid - 1.5, "per month", size=OUTER_SMALL, ha="left")


def draw_legend(c: Canvas) -> None:
    x, y = LAYER_X, 46.4
    for i, (label, edge, fill) in enumerate(
        [
            ("deterministic", DETERMINISTIC, DETERMINISTIC_FILL),
            ("frontier", FRONTIER, FRONTIER_FILL),
            ("local", LOCAL, LOCAL),
        ]
    ):
        yy = y - i * 3.0
        c.box(x, yy - 1.1, x + 2.8, yy + 1.1, edge=edge, fill=fill, lw=1.6)
        c.text(x + 3.6, yy, label, size=OUTER_SMALL, ha="left")


def draw(draft: Path | None) -> Path:
    c = Canvas()
    draw_bands(c)
    draw_layer_names(c)
    draw_foundation(c)
    draw_graph(c)
    draw_functions(c)
    draw_interfaces(c)
    draw_users(c)
    draw_usage(c)
    draw_legend(c)
    if draft is not None:
        c.check_layout(refuse=False)
        c.fig.savefig(draft, dpi=100, facecolor="white")
        plt.close(c.fig)
        return draft
    c.check_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    svg = FIGURES / f"{STEM}.svg"
    c.fig.savefig(svg, format="svg", metadata={"Date": None}, facecolor="white")
    text = svg.read_text()
    # declared at slide width (31.1 cm), so it drops onto a slide under its title
    text = text.replace(f'width="{WIDTH * 72:g}pt"', 'width="31.1cm"', 1)
    text = text.replace(
        f'height="{HEIGHT * 72:g}pt"', f'height="{31.1 * HEIGHT / WIDTH:.2f}cm"', 1
    )
    svg.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")
    plt.close(c.fig)
    print(svg.relative_to(ROOT))
    return svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draft", type=Path, help="write a PNG here, unchecked")
    args = parser.parse_args()
    draw(args.draft)


if __name__ == "__main__":
    main()

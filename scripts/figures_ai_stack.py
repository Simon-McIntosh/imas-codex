"""Draw ITER's AI stack as layers, from the infrastructure ITER owns up to its users.

An illustration of an architecture, not a census. Reading from the bottom:

- **ICP4**, the existing engineering and plant data systems;
- the **knowledge graph** built on top of them by ETL, which itself becomes ITER
  infrastructure: durable, owned, and drawn in the relation colour of the ICP
  knowledge-map figures, whose ringed nodes it reuses;
- a **function layer** of read-only tools over the graph (Lucy tools, imas-codex
  tools), third-party connectors (Outlook, Jira, Confluence) to the left, and a
  REPL to the right as a code-generation route for whatever the tools do not cover;
- the **interfaces** people use: Claude Desktop, Lucy chat, domain dashboards, a
  sandbox, and Clive, a Claude orchestrator serving open-weight models on local GPUs;
- the **users**, one bar per role, spanning the interfaces that role reaches.

Colour separates what is deterministic (purple, the infrastructure ITER owns) from
what is agentic or LLM-driven (coral, the ephemeral layer that runs over it).

Usage::

    uv run python scripts/figures_ai_stack.py
    uv run python scripts/figures_ai_stack.py --draft /tmp/x.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/ai-stack"
STEM = "ai-stack"

WIDTH, HEIGHT = 14.0, 6.4  # in: wider than the slide canvas, leaving room for a title
U = 10  # drawing units per inch, so the axes run 0..140 by 0..72

# Deterministic: the infrastructure ITER owns, in the ICP knowledge map's relation colour.
DETERMINISTIC = "#8e3a8e"
DETERMINISTIC_FILL = "#f6eef6"
# Agentic or LLM-driven: the ephemeral layer that runs over the infrastructure.
AGENTIC = "#d9622b"
AGENTIC_FILL = "#fcefe8"
USER = "#6b6b6b"
INK = "#2b2b2b"
MUTED = "#6b6b6b"
NEUTRAL_EDGE = "#b4b1aa"

TITLE, BODY, SMALL = 14, 12, 11  # pt, inside the stack
OUTER = 18  # pt, the layer names
OUTER_SMALL = 14  # pt, the legend and the annotations outside the boxes


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
            (AGENTIC, AGENTIC_FILL)
            if kind == "agentic"
            else (DETERMINISTIC, DETERMINISTIC_FILL)
        )
        self.box(x0, y0, x1, y1, edge=edge, fill=fill)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inside = (x0, y0, x1, y1)
        if subtitle is None:
            self.text(cx, cy, title, size=size, inside=inside, weight=weight)
        else:
            # stack the title above the subtitle, centred as one block
            # one line of text is size * linespacing points; 7.2 pt per drawing unit
            title_h = (title.count("\n") + 1) * size * 1.2 / 7.2
            sub_h = (subtitle.count("\n") + 1) * SMALL * 1.1 / 7.2
            gap = 0.5
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
        edge = AGENTIC if kind == "agentic" else DETERMINISTIC
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
            px, py = 0.4, 0.15  # clear margin to the box edge, in drawing units
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

CENTRE, BASE_HALF, SLOPE = 63.0, 59.5, 0.27  # half-width = BASE_HALF - SLOPE * y
LAYER_X = 124.0
BAND_FILL = "#f4f3f0"

LAYERS = {  # (y0, y1) of each band
    "icp4": (1.0, 8.6),
    "graph": (11.8, 20.6),
    "functions": (22.0, 37.8),
    "interfaces": (39.2, 50.2),
    "users": (51.6, 63.4),
}


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


LAYER_NAMES = {
    "users": ["users"],
    "interfaces": ["interfaces"],
    "functions": ["functions"],
    "graph": ["knowledge", "graph"],
    "icp4": ["ICP4"],
}


def draw_layer_names(c: Canvas) -> None:
    for key, lines in LAYER_NAMES.items():
        y0, y1 = LAYERS[key]
        mid = (y0 + y1) / 2
        for i, line in enumerate(lines):
            dy = (len(lines) - 1) / 2 * 3.0 - i * 3.0
            c.text(LAYER_X, mid + dy, line, size=OUTER, ha="left", weight="bold")


# Each role reaches from Claude Desktop to the interface named. No one stops at
# Claude Desktop: every role needs ITER's data, which arrives through Lucy at least.
USERS = [
    ("code developer", "clive"),
    ("commissioning engineer", "dash"),
    ("planning", "dash"),
    ("configuration control", "lucy"),
    ("administrative support", "lucy"),
]

LUCY_TOOLS = [
    ("ask Lucy", "agentic"),
    ("search records", "deterministic"),
    ("get records", "deterministic"),
    ("relationships", "deterministic"),
    ("versions", "deterministic"),
    ("folders", "deterministic"),
    ("record schema", "deterministic"),
    ("abbreviations", "deterministic"),
    ("document file", "deterministic"),
]

CODEX_TOOLS = [
    "signals",
    "analytics",
    "DD paths",
    "standard names",
    "docs · wiki",
    "coverage",
]

SOURCES = [
    ("IDM", "documents"),
    ("CATIA", "CAD"),
    ("SMDD", "drawings"),
    ("NCR", "quality"),
    ("SAP", "contracts"),
    ("RQM", "requirements"),
    ("PBS", "breakdown"),
    ("UDA", "plant data"),
    ("DAN", "plant data"),
    ("DS", "plant data"),
]

INTERFACES: dict[str, tuple[float, float]] = {}


def draw_interfaces(c: Canvas) -> None:
    y0, y1 = LAYERS["interfaces"]
    y0, y1 = y0 + 1.0, y1 - 1.0
    cols = spread(*span(y1), [12.5, 13, 26.5, 11, 15.5])
    for key, xs in zip(
        ["desktop", "lucy", "dash", "sandbox", "clive"], cols, strict=True
    ):
        INTERFACES[key] = xs
    x0, x1 = INTERFACES["desktop"]
    c.card(x0, y0, x1, y1, "Claude\nDesktop", None, kind="agentic", size=BODY)
    x0, x1 = INTERFACES["lucy"]
    c.card(
        x0, y0, x1, y1, "Lucy chat", "engineering\nassistant", kind="agentic", size=BODY
    )
    x0, x1 = INTERFACES["dash"]
    c.box(x0, y0, x1, y1, edge=NEUTRAL_EDGE, fill="white", lw=1.2, dashed=True)
    c.text(
        (x0 + x1) / 2,
        y1 - 1.6,
        "domain dashboards",
        size=SMALL,
        color=MUTED,
        inside=(x0, y0, x1, y1),
    )
    mid = (x0 + x1) / 2
    c.card(
        x0 + 0.8,
        y0 + 0.8,
        mid - 0.5,
        y1 - 3.3,
        "commissioning\nstatus",
        None,
        kind="deterministic",
        size=SMALL,
        weight="normal",
    )
    c.card(
        mid + 0.5,
        y0 + 0.8,
        x1 - 0.8,
        y1 - 3.3,
        "overnight\ndigest · cron",
        None,
        kind="agentic",
        size=SMALL,
        weight="normal",
    )
    x0, x1 = INTERFACES["sandbox"]
    c.card(x0, y0, x1, y1, "Sandbox", "full\ndevelopment", kind="agentic", size=BODY)
    x0, x1 = INTERFACES["clive"]
    c.card(
        x0,
        y0,
        x1,
        y1,
        "Clive",
        "Claude orchestrator\nH200 + B300 GPUs\nDS v4.1 Flash",
        kind="agentic",
        size=BODY,
    )


def draw_users(c: Canvas) -> None:
    y0, y1 = LAYERS["users"]
    pitch, height = 2.3, 1.8
    top = y1 - 0.7
    for i, (role, reach) in enumerate(USERS):
        y = top - i * pitch
        left = span(y)[0]
        x1 = min(INTERFACES[reach][1], span(y)[1])
        c.box(left, y - height, x1, y, edge=USER, fill=USER, lw=0, z=3)
        c.text(
            left + 1.0,
            y - height / 2,
            role,
            size=SMALL,
            color="white",
            ha="left",
            inside=(left, y - height - 0.3, x1, y + 0.3),
        )


def draw_functions(c: Canvas) -> None:
    y0, y1 = LAYERS["functions"]
    y0, y1 = y0 + 1.0, y1 - 1.0
    (cx, lx, ix, rx) = spread(*span(y1), [11.2, 40.6, 29.4, 9.2])
    gap = 1.0

    # third-party connectors, to the left and a little higher
    cx0, cx1 = cx
    cy0, cy1 = y0 + 1.6, y1
    c.box(cx0, cy0, cx1, cy1, edge=NEUTRAL_EDGE, fill="white", lw=1.2, dashed=True)
    c.text(
        (cx0 + cx1) / 2,
        cy1 - 1.6,
        "connectors",
        size=SMALL,
        color=MUTED,
        inside=(cx0, cy0, cx1, cy1),
    )
    for i, name in enumerate(["Outlook", "Jira", "Confluence"]):
        top = cy1 - 3.4 - i * 3.2
        c.chip(cx0 + 0.6, top - 2.4, cx1 - 0.6, top, name, kind="deterministic")

    # Lucy tools: what Lucy chat calls, and where other assistants plug into the graph
    lx0, lx1 = lx
    c.box(lx0, y0, lx1, y1, edge=NEUTRAL_EDGE, fill="white", lw=1.2, dashed=True)
    c.text(
        lx0 + 1.0,
        y1 - 1.6,
        "Lucy tools",
        size=BODY,
        weight="bold",
        ha="left",
        inside=(lx0, y0, lx1, y1),
    )
    cols = 3
    w = (lx1 - lx0 - 2 * 0.8 - (cols - 1) * gap) / cols
    for i, (name, kind) in enumerate(LUCY_TOOLS):
        r, col = divmod(i, cols)
        x = lx0 + 0.8 + col * (w + gap)
        top = y1 - 3.4 - r * 3.2
        c.chip(x, top - 2.5, x + w, top, name, kind=kind)

    # imas-codex tools over the fusion data dictionary, signals and facility docs
    ix0, ix1 = ix
    c.box(ix0, y0, ix1, y1, edge=NEUTRAL_EDGE, fill="white", lw=1.2, dashed=True)
    c.text(
        ix0 + 1.0,
        y1 - 1.6,
        "imas-codex tools",
        size=BODY,
        weight="bold",
        ha="left",
        inside=(ix0, y0, ix1, y1),
    )
    cols = 2
    w = (ix1 - ix0 - 2 * 0.8 - (cols - 1) * gap) / cols
    for i, name in enumerate(CODEX_TOOLS):
        r, col = divmod(i, cols)
        x = ix0 + 0.8 + col * (w + gap)
        top = y1 - 3.4 - r * 3.2
        c.chip(x, top - 2.5, x + w, top, name, kind="deterministic")

    # the REPL: generated code, run against the graph, for whatever no tool covers
    c.card(rx[0], y0, rx[1], y1, "REPL", "code\ngeneration", kind="agentic")


def draw_graph(c: Canvas) -> None:
    y0, y1 = LAYERS["graph"]
    y0, y1 = y0 + 0.8, y1 - 0.8
    x0, x1 = span(y1)
    c.box(x0, y0, x1, y1, edge=DETERMINISTIC, fill=DETERMINISTIC_FILL, lw=2.0)
    c.text(
        x0 + 1.5,
        (y0 + y1) / 2 + 1.4,
        "ITER knowledge graph",
        size=TITLE,
        weight="bold",
        ha="left",
        color=DETERMINISTIC,
        inside=(x0, y0, x1, y1),
    )
    c.text(
        x0 + 1.5,
        (y0 + y1) / 2 - 1.8,
        "schema · data · knowledge",
        size=SMALL,
        color=MUTED,
        ha="left",
        inside=(x0, y0, x1, y1),
    )
    # a strip of the network, drawn as the ICP knowledge map draws its nodes and links
    lo, hi = y0 + 1.8, y1 - 1.8
    xs = [x0 + 36 + i * (x1 - x0 - 39) / 12 for i in range(13)]
    nodes = [(x, hi if i % 2 == 0 else lo) for i, x in enumerate(xs)]
    edges = [(i, i + 1) for i in range(len(nodes) - 1)]
    edges += [
        (0, 2),
        (2, 4),
        (3, 5),
        (4, 6),
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (10, 12),
    ]
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
        hub = i in (4, 9)
        c.ax.plot(
            x,
            y,
            "o",
            markersize=11 if hub else 8,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=2.0,
            zorder=6,
        )
        if hub:
            c.ax.plot(x, y, "o", markersize=4, color=INK, zorder=7)
    # read-only access from the function layer down into the graph
    xa = x0 + 30
    c.ax.annotate(
        "",
        xy=(xa, y1 - 0.2),
        xytext=(xa, LAYERS["functions"][0] + 1.0),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4, "mutation_scale": 14},
        zorder=8,
    )
    c.text(
        xa + 1.0,
        (y1 + LAYERS["functions"][0]) / 2 + 0.5,
        "read-only",
        size=OUTER_SMALL,
        color=MUTED,
        ha="left",
    )


def draw_foundation(c: Canvas) -> None:
    y0, y1 = LAYERS["icp4"]
    y0, y1 = y0 + 0.8, y1 - 0.8
    cols = spread(*span(y1), [1.25 if s == "RQM" else 1 for s, _ in SOURCES])
    g0 = LAYERS["graph"][0] + 0.8
    gx0, gx1 = span(g0 + 8)
    for (x, xe), (system, holds) in zip(cols, SOURCES, strict=True):
        c.box(x, y0, xe, y1, edge=DETERMINISTIC, fill="white", lw=1.2, z=4)
        c.text(
            (x + xe) / 2,
            (y0 + y1) / 2 + 1.0,
            system,
            size=BODY,
            weight="bold",
            inside=(x, y0, xe, y1),
        )
        c.text(
            (x + xe) / 2,
            (y0 + y1) / 2 - 1.3,
            holds,
            size=SMALL - 1,
            color=MUTED,
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
    top = LAYERS["graph"][1]
    xb = 2.4
    c.ax.plot(
        [xb + 1, xb, xb, xb + 1],
        [LAYERS["icp4"][0], LAYERS["icp4"][0], top, top],
        color=DETERMINISTIC,
        lw=1.6,
        zorder=3,
    )
    c.text(
        xb - 1.4,
        (LAYERS["icp4"][0] + top) / 2,
        "ITER infrastructure",
        size=OUTER_SMALL,
        rotation=90,
        color=DETERMINISTIC,
        weight="bold",
    )


def draw_legend(c: Canvas) -> None:
    x, y = 1.5, 62.2
    for i, (label, edge, fill) in enumerate(
        [
            ("deterministic", DETERMINISTIC, DETERMINISTIC_FILL),
            ("agentic · LLM", AGENTIC, AGENTIC_FILL),
        ]
    ):
        yy = y - i * 3.2
        c.box(x, yy - 1.1, x + 2.8, yy + 1.1, edge=edge, fill=fill, lw=1.6)
        c.text(x + 3.8, yy, label, size=OUTER_SMALL, ha="left")


def draw(draft: Path | None) -> Path:
    c = Canvas()
    draw_bands(c)
    draw_layer_names(c)
    draw_foundation(c)
    draw_graph(c)
    draw_functions(c)
    draw_interfaces(c)
    draw_users(c)
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

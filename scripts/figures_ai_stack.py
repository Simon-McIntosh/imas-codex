"""Draw ITER's AI stack as a pyramid, from the infrastructure ITER owns up to its users.

An illustration of an architecture, not a census. Reading from the bottom:

- the **ITER Collaborative Platform** (ICP4), the existing engineering and plant
  data systems;
- the **knowledge graph** built on top of them by extract, transform and load, which
  itself becomes ITER infrastructure: durable, owned, and drawn with the ringed
  nodes of the ICP knowledge-map figures;
- a **function layer** of read-only tools over the graph (Lucy tools, imas-codex
  tools), third-party connectors (Outlook, Jira, Confluence) to the left, and a
  REPL to the right as a code-generation route for whatever the tools do not cover;
- the **interfaces** people use: Claude Desktop, Lucy chat, domain dashboards, a
  sandbox, and Clive, a Claude orchestrator. Every interface but Claude Desktop can
  also run on **on-premises inference**, open-weight models on ITER's own GPUs;
- the **users**, one bar per role, spanning the interfaces that role reaches.

Colour separates what is deterministic, the infrastructure ITER owns, from what is
agentic: driven by a frontier model, or by an on-premises one.

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
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures/ai-stack"
STEM = "ai-stack"

WIDTH, HEIGHT = 14.0, 6.75  # in: wider than the slide canvas, leaving room for a title
U = 10  # drawing units per inch, so the axes run 0..140 by 0..67.5

# Deterministic in the ICP knowledge map's relation plum; agentic work in coral where a
# frontier model drives it, teal where it can also run on-premises.
COLOURS = ("#8e3a8e", "#cf5f36", "#1d8585")  # deterministic, frontier, on-premises

INK = "#2b2b2b"
MUTED = "#5f5f5f"
NEUTRAL_EDGE = "#b4b1aa"
BAND_FILL = "#f4f3f0"
USER_INK = "#141414"  # the role names
# the thin bar under each name, in a deep blue no layer of the stack uses
USER_FILL = "#4a7bbd"

TITLE, BODY, SMALL = 16, 14, 13  # pt, inside the stack
BAR = 15  # pt, the user roles
OUTER = 18  # pt, the layer names
OUTER_SMALL = 14  # pt, the legend and the annotations outside the boxes
PT = 7.2  # points per drawing unit


def tint(colour: str, amount: float = 0.1) -> tuple[float, float, float]:
    """The colour mixed into white, `amount` of the way from white."""
    return tuple(1 - amount * (1 - v) for v in to_rgb(colour))


@dataclass(frozen=True)
class Roles:
    deterministic: str
    frontier: str
    onprem: str

    def fill(self, kind: str):
        return tint(getattr(self, kind))


@dataclass
class Placed:
    """A text artist and, optionally, the box it must stay inside."""

    text: plt.Text
    inside: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 in data


class Canvas:
    def __init__(self, roles: Roles):
        plt.style.use("data-ink")
        # Arial where it is installed; Nimbus Sans shares its metrics, so the layout
        # check measures the same widths, and the SVG keeps its text as text in Arial.
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Nimbus Sans"]
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["svg.hashsalt"] = STEM  # stable ids, so a re-render diffs clean
        self.roles = roles
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
        """A rounded box in its role's colour, a title and an optional subtitle."""
        self.box(
            x0, y0, x1, y1, edge=getattr(self.roles, kind), fill=self.roles.fill(kind)
        )
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inside = (x0, y0, x1, y1)
        if subtitle is None:
            self.text(cx, cy, title, size=size, inside=inside, weight=weight)
            return
        lines = [subtitle] if isinstance(subtitle, str) else list(subtitle)
        title_h, sub_h, gap = size * 1.2 / PT, SMALL * 1.2 / PT, 0.4
        top = cy + (title_h + gap + sub_h * len(lines)) / 2
        self.text(cx, top, title, size=size, inside=inside, weight=weight, va="top")
        for i, line in enumerate(lines):
            self.text(
                cx,
                top - title_h - gap - i * sub_h,
                line,
                size=SMALL,
                inside=inside,
                color=MUTED,
                va="top",
            )

    def onprem_badge(self, x1, y1):
        """A dot on a card's top edge: this interface can also run on-premises."""
        self.ax.plot(
            x1 - 2.0,
            y1,
            "o",
            markersize=14,
            markerfacecolor=self.roles.onprem,
            markeredgecolor="white",
            markeredgewidth=2.0,
            zorder=9,
        )

    def chip(self, x0, y0, x1, y1, label, *, kind):
        self.box(x0, y0, x1, y1, edge=getattr(self.roles, kind), lw=1.2, z=4)
        self.text(
            (x0 + x1) / 2, (y0 + y1) / 2, label, size=SMALL, inside=(x0, y0, x1, y1)
        )

    def group(self, x0, y0, x1, y1, title=None):
        """A dashed neutral frame that gathers chips or cards under a heading."""
        self.box(x0, y0, x1, y1, edge=NEUTRAL_EDGE, lw=1.2, dashed=True)
        if title is not None:
            self.text(
                x0 + 1.0,
                y1 - 1.5,
                title,
                size=BODY,
                weight="bold",
                ha="left",
                inside=(x0, y0, x1, y1),
            )

    def check_layout(self, *, refuse: bool = True) -> None:
        """Refuse text that wraps, leaves the canvas, collides, or escapes its box."""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        to_data = self.ax.transData.inverted()
        boxes = [(p, p.text.get_window_extent(renderer)) for p in self.placed]
        faults = [
            f"wraps: {p.text.get_text()!r}"
            for p, _ in boxes
            if "\n" in p.text.get_text()
        ]
        faults += [
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
            px, py = 0.6, 0.1  # clear margin to the box edge, in drawing units
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
# layer names lean against its right-hand slope, beside their band.

CENTRE, BASE_HALF, SLOPE = 63.5, 58.5, 0.2  # half-width = BASE_HALF - SLOPE * y
LAYER_GAP = 1.5  # between a band's sloped right edge and its name

LAYERS = {  # (y0, y1) of each band
    "icp": (2.6, 7.8),
    "graph": (8.8, 14.8),
    "functions": (15.8, 30.6),
    "interfaces": (31.6, 43.8),
    "users": (44.8, 63.2),
}

LAYER_NAMES = {  # the users are named above their bars, in draw_users
    "interfaces": ["interfaces"],
    "functions": ["functions"],
    "graph": ["graph"],
    "icp": ["ICP4"],
}

# Top to bottom: each bar's roles reach from Claude Desktop to the interface named, and
# no one stops at Claude Desktop, because every role needs ITER's data through Lucy.
# The widest reach sits lowest, so the bars taper upward as the pyramid does.
USERS = [
    ("office productivity", "lucy"),
    ("commissioning engineer · planning · configuration", "dash"),
    ("prototyping with domain experts · science · code developer", "sandbox"),
    ("assembly and commissioning · ITER infrastructure", "clive"),
]

LUCY_TOOLS = [
    "ask Lucy",
    "search records",
    "get records",
    "relationships",
    "versions",
    "abbreviations",
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


def grid(c: Canvas, x0, x1, top, names, cols, *, kinds=None, height=2.8, pitch=3.4):
    """Lay chips in rows of `cols` from x0 to x1, the first row's top at `top`."""
    gap = 1.0
    w = (x1 - x0 - (cols - 1) * gap) / cols
    for i, name in enumerate(names):
        r, col = divmod(i, cols)
        x = x0 + col * (w + gap)
        y = top - r * pitch
        kind = (kinds or {}).get(name, "deterministic")
        c.chip(x, y - height, x + w, y, name, kind=kind)


def draw_bands(c: Canvas) -> None:
    # the users stand on the stack rather than forming a layer of it, so no band
    for key, (y0, y1) in LAYERS.items():
        if key == "users":
            continue
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
        pitch = 2.9
        mid = (y0 + y1) / 2
        for i, line in enumerate(lines):
            y = mid + (len(lines) - 1) / 2 * pitch - i * pitch
            # each line leans on the pyramid's slope at its own height
            x = span(y, 0)[1] + LAYER_GAP
            c.text(x, y, line, size=OUTER, ha="left", weight="bold")


def draw_foundation(c: Canvas) -> None:
    y0, y1 = LAYERS["icp"]
    y0, y1 = y0 + 0.7, y1 - 0.7
    g0 = LAYERS["graph"][0] + 0.6
    for (x, xe), system in zip(
        spread(*span(y1), [1] * len(SOURCES)), SOURCES, strict=True
    ):
        c.box(x, y0, xe, y1, edge=c.roles.deterministic, lw=1.2, z=4)
        c.text(
            (x + xe) / 2,
            (y0 + y1) / 2,
            system,
            size=BODY,
            weight="bold",
            inside=(x, y0, xe, y1),
        )
        # each source rises straight into the graph
        xm = (x + xe) / 2
        c.ax.plot(
            [xm, xm],
            [y1, g0],
            color=c.roles.deterministic,
            lw=2.4,
            alpha=0.7,
            solid_capstyle="butt",
            zorder=1,
        )
    # the graph becomes ITER infrastructure, alongside the platform it is built on
    bottom, top = LAYERS["icp"][0], LAYERS["graph"][1]
    xb = 4.4
    c.ax.plot(
        [xb + 1, xb, xb, xb + 1],
        [bottom, bottom, top, top],
        color=c.roles.deterministic,
        lw=1.6,
        zorder=3,
    )
    # two rotated lines, read bottom to top, so the label fits beside the bracket
    for x, line in [(xb - 3.1, "ITER"), (xb - 1.1, "infrastructure")]:
        c.text(
            x,
            (bottom + top) / 2,
            line,
            size=OUTER_SMALL - 2,
            rotation=90,
            color=c.roles.deterministic,
            weight="bold",
        )


def draw_graph(c: Canvas) -> None:
    y0, y1 = LAYERS["graph"]
    y0, y1 = y0 + 0.6, y1 - 0.6
    x0, x1 = span(y1)
    colour = c.roles.deterministic
    c.box(x0, y0, x1, y1, edge=colour, fill=c.roles.fill("deterministic"), lw=2.0)
    c.text(
        x0 + 1.5,
        (y0 + y1) / 2,
        "ITER knowledge graph",
        size=TITLE,
        weight="bold",
        ha="left",
        color=colour,
        inside=(x0, y0, x1, y1),
    )
    c.text(
        x0 + 30.0,
        (y0 + y1) / 2,
        "search & traverse",
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
            [xa, xb], [ya, yb], color=colour, lw=2.0, zorder=4, solid_capstyle="round"
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
    y0, y1 = y0 + 0.7, y1 - 0.7
    cx, lx, ix, rx = spread(*span(y1), [12.5, 33.5, 33.5, 17.0])
    pad, header = 0.8, 3.0

    c.group(cx[0], y0, cx[1], y1, None)
    grid(
        c,
        cx[0] + pad,
        cx[1] - pad,
        y1 - pad - 0.4,
        ["Outlook", "Jira", "Confluence"],
        1,
        height=3.0,
        pitch=4.1,
    )
    c.group(lx[0], y0, lx[1], y1, "Lucy tools")
    grid(
        c,
        lx[0] + pad,
        lx[1] - pad,
        y1 - header,
        LUCY_TOOLS,
        2,
        kinds={"ask Lucy": "frontier"},
    )
    c.group(ix[0], y0, ix[1], y1, "imas-codex tools")
    grid(c, ix[0] + pad, ix[1] - pad, y1 - header, CODEX_TOOLS, 2)
    # the REPL: generated code, run against the graph, for whatever no tool covers
    c.card(rx[0], y0, rx[1], y1, "REPL", "code generation", kind="frontier")


def draw_interfaces(c: Canvas) -> None:
    y0, y1 = LAYERS["interfaces"]
    y0, y1 = y0 + 0.6, y1 - 0.9
    cols = spread(*span(y1), [19, 12.5, 18.5, 22.5, 19.5])
    for key, xs in zip(
        ["desktop", "lucy", "dash", "sandbox", "clive"], cols, strict=True
    ):
        INTERFACES[key] = xs
    c.card(
        *INTERFACES["desktop"][:1],
        y0,
        INTERFACES["desktop"][1],
        y1,
        "Claude Desktop",
        kind="frontier",
    )
    x0, x1 = INTERFACES["lucy"]
    c.card(x0, y0, x1, y1, "Lucy chat", kind="frontier")
    # domain dashboards: a deterministic status view above a scheduled agentic digest
    x0, x1 = INTERFACES["dash"]
    c.box(x0, y0, x1, y1, edge=NEUTRAL_EDGE, lw=1.2, dashed=True)
    c.text(
        (x0 + x1) / 2,
        y1 - 1.5,
        "domain dashboards",
        size=SMALL,
        color=MUTED,
        inside=(x0, y0, x1, y1),
    )
    for i, (label, kind) in enumerate(
        [("commissioning", "deterministic"), ("overnight digest", "frontier")]
    ):
        top = y1 - 3.0 - i * 3.5
        c.card(
            x0 + 0.8,
            top - 2.9,
            x1 - 0.8,
            top,
            label,
            kind=kind,
            size=SMALL,
            weight="normal",
        )
    x0, x1 = INTERFACES["sandbox"]
    c.card(
        x0,
        y0,
        x1,
        y1,
        "Claude Code",
        ("sandbox · full development", "H200 GPUs", "DeepSeek v4.1 Flash"),
        kind="frontier",
    )
    x0, x1 = INTERFACES["clive"]
    c.card(
        x0,
        y0,
        x1,
        y1,
        "Clive",
        ("Claude orchestrator", "B300 GPUs", "DeepSeek v4.1 Flash"),
        kind="frontier",
    )

    # every interface but Claude Desktop can also run on ITER's own GPUs
    x0, x1 = INTERFACES["dash"]
    c.onprem_badge(x1 - 0.8, y1 - 3.0 - 3.5)
    for key in ["lucy", "sandbox", "clive"]:
        c.onprem_badge(INTERFACES[key][1], y1)


def draw_users(c: Canvas) -> None:
    _, y1 = LAYERS["users"]
    pitch, label_h, gap, height = 4.5, 2.3, 0.1, 0.8
    top = y1 - 0.4
    # the users stand above the stack, so their name sits over the bars, centred
    x0, x1 = INTERFACES["desktop"][0], INTERFACES["clive"][1]
    c.text((x0 + x1) / 2, y1 + 1.8, "users", size=OUTER, weight="bold")
    order = list(INTERFACES)
    for i, (role, reach) in enumerate(USERS):
        y = top - i * pitch
        # the role's name on its own line, and under it a thin bar of one cell per
        # interface the role uses, each cell edge to edge with its card below
        left = INTERFACES[order[0]][0]
        c.text(
            left,
            y - label_h / 2,
            role,
            size=BAR,
            color=USER_INK,
            ha="left",
        )
        bar_top = y - label_h - gap
        for key in order[: order.index(reach) + 1]:
            cx0, cx1 = INTERFACES[key]
            c.ax.add_patch(
                FancyBboxPatch(  # fully rounded ends: the radius is half the height
                    (cx0, bar_top - height),
                    cx1 - cx0,
                    height,
                    boxstyle=f"round,pad=0,rounding_size={height / 2}",
                    facecolor=USER_FILL,
                    edgecolor="none",
                    zorder=3,
                )
            )


def draw_legend(c: Canvas) -> None:
    # top right, in the clear space beside the user bars
    x, y = WIDTH * U - 19.0, HEIGHT * U - 2.2
    for i, (label, kind) in enumerate(
        [
            ("deterministic", "deterministic"),
            ("frontier", "frontier"),
            ("open-weight workers", "onprem"),
        ]
    ):
        yy = y - i * 3.0
        if kind == "onprem":
            # two lines, each its own artist, with the dot centred on the pair
            c.ax.plot(
                x + 1.4,
                yy - 1.2,
                "o",
                markersize=14,
                markerfacecolor=c.roles.onprem,
                markeredgecolor="white",
                markeredgewidth=2.0,
                zorder=9,
            )
            first, second = label.rsplit(" ", 1)
            c.text(x + 3.6, yy, first, size=OUTER_SMALL, ha="left")
            c.text(x + 3.6, yy - 2.4, second, size=OUTER_SMALL, ha="left")
            continue
        c.box(
            x,
            yy - 1.1,
            x + 2.8,
            yy + 1.1,
            edge=getattr(c.roles, kind),
            fill=c.roles.fill(kind),
            lw=1.6,
        )
        c.text(x + 3.6, yy, label, size=OUTER_SMALL, ha="left")


def draw(draft: Path | None) -> Path:
    c = Canvas(Roles(*COLOURS))
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

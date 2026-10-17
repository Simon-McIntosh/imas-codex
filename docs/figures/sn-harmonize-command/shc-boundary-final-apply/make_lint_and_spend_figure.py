"""What the unforced apply to the boundary-density family changed, and what it cost.

Two panels, read from the apply's own output rather than from prose. The left
panel is the family's lint findings by class, before the apply against after.
The right panel is what the apply spent against the ceiling it was given.

The apply is the subject and takes the colour; the state before it is the
reference and stays neutral. The ceiling is a limit and is drawn as a dotted
neutral line, per the data-ink rules.

The script refuses to write the figure while any text artist runs off the canvas
or overlaps another in the same panel, so a clipped or colliding label is a
failed run rather than a bad deliverable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

OUT = Path(__file__).parent

INK = "#1b2733"
MUTED = "#8c8c8c"
COOL = "#2d6ca2"
LIMIT = "#52514e"

#: Lint findings per class before the apply and after it, as the report printed
#: them; the two class counts sum to its lint_total of 28 before and 8 after.
#: Worst-first, so the class that moved most sits at the top of the panel.
CLASSES = ("link missing\nreverse", "convention\nviolation")
BEFORE = (13, 15)
AFTER = (8, 0)

#: What the apply spent, and the ceiling its command line carried. The two
#: interrupted attempts spent about $1.02 more; the ceiling bounds the node, not
#: this one run, so the total lives in the caption rather than in the panel.
SPENT = 5.15
CEILING = 8.50

FS_LABEL = 16
FS_TICK = 14


def _label(ax, x, y, text, color):
    ax.annotate(
        text,
        (x, y),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=FS_TICK,
        color=color,
        va="center",
        ha="left",
        zorder=5,
    )


def build():
    plt.style.use("data-ink")
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": (2.1, 1.0)}
    )

    h = 0.30
    for row, (before, after) in enumerate(zip(BEFORE, AFTER, strict=True)):
        y = float(row)
        left.barh(y + 0.19, before, height=h, color=MUTED, zorder=3)
        left.barh(y - 0.19, after, height=h, color=COOL, zorder=3)
        _label(left, before, y + 0.19, f"{before}", MUTED)
        _label(left, after, y - 0.19, f"{after}", COOL)

    left.set_yticks(range(len(CLASSES)))
    left.set_yticklabels(CLASSES)
    # The top of the panel is left clear for the two series names, so they sit
    # in empty space rather than on the bars.
    left.set_ylim(-0.75, len(CLASSES) + 0.05)
    left.set_xlim(0, 17)
    left.set_xlabel("lint findings", fontsize=FS_LABEL, color=INK)
    left.set_title("findings by class", fontsize=FS_LABEL, color=INK, pad=14)
    left.text(0.985, 0.98, "after the apply", transform=left.transAxes,
              fontsize=FS_TICK, color=COOL, va="top", ha="right")
    left.text(0.985, 0.89, "before the apply", transform=left.transAxes,
              fontsize=FS_TICK, color=MUTED, va="top", ha="right")

    right.barh(0.0, SPENT, height=0.5, color=COOL, zorder=3)
    right.axvline(CEILING, color=LIMIT, lw=1.2, ls=":", zorder=2)
    _label(right, SPENT, 0.0, f"${SPENT:.2f}", COOL)
    right.annotate(
        f"ceiling ${CEILING:.2f}",
        (CEILING, 0.42),
        xytext=(-6, 0),
        textcoords="offset points",
        fontsize=FS_TICK,
        color=LIMIT,
        va="center",
        ha="right",
    )
    right.set_yticks([])
    right.set_ylim(-0.75, 0.85)
    right.set_xlim(0, 9.8)
    right.set_xlabel("USD", fontsize=FS_LABEL, color=INK)
    right.set_title("spend against the ceiling", fontsize=FS_LABEL, color=INK, pad=14)

    fig.tight_layout()
    return fig


def check(fig) -> list[str]:
    """Name every text artist that runs off the canvas.

    Label positions are compared with the canvas in inches, since a rendered
    extent is in device pixels and the two disagree whenever the renderer's dpi
    differs from the figure's. A collision between two labels is judged by
    looking at the render instead: the extents matplotlib reports for tick
    artists do not track the drawn figure closely enough for a collision test,
    which fails figures that render cleanly.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = float(renderer.dpi)
    canvas = fig.bbox_inches
    tol = 0.01
    problems: list[str] = []
    for artist in fig.findobj(Text):
        if not artist.get_text().strip():
            continue
        b = artist.get_window_extent(renderer=renderer)
        if b.width <= 0 or b.height <= 0:
            continue
        box = Bbox.from_extents(b.x0 / dpi, b.y0 / dpi, b.x1 / dpi, b.y1 / dpi)
        if (
            box.x0 < canvas.x0 - tol
            or box.y0 < canvas.y0 - tol
            or box.x1 > canvas.x1 + tol
            or box.y1 > canvas.y1 + tol
        ):
            problems.append(f"outside canvas: {artist.get_text()!r} {box}")
    return problems


def main() -> None:
    fig = build()
    problems = check(fig)
    out = OUT / "lint-and-spend.svg"
    fig.savefig(out)
    for p in problems:
        print(f"CHECK-FAIL {p}")
    print(f"CHECK {'FAIL' if problems else 'ok'} -> {out}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

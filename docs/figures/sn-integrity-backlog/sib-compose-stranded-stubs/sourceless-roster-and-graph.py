#!/usr/bin/env python3
"""Draw the stranded-stub figure and refuse to write it if a label misbehaves.

Left panel: the seven roster stubs split the way ``export.py`` splits them,
before the scoped run and now. Right panel: every non-roster source by status,
which is the population the scope control is about.

Run from the repository root; the SVG lands beside this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

GATE = "#b03a2e"
DRAFT = "#2a6f97"
NEUTRAL = "#929292"

BEFORE = {"gate": 0, "draft": 7, "dormant": 0}
NOW = {"gate": 0, "draft": 5, "dormant": 2}
STATUS = [
    ("composed", 3069),
    ("attached", 2203),
    ("stale", 1873),
    ("skipped", 1410),
    ("extracted", 1402),
    ("failed", 100),
    ("not_physical_quantity", 10),
]


def check(fig) -> None:
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    x0, y0, w, h = fig.bbox.bounds
    canvas = (x0, y0, x0 + w, y0 + h)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        box = t.get_window_extent(renderer)
        if (
            box.x0 < canvas[0] - 1
            or box.y0 < canvas[1] - 1
            or box.x1 > canvas[2] + 1
            or box.y1 > canvas[3] + 1
        ):
            raise SystemExit(f"label outside the canvas: {t.get_text()!r}")
        boxes.append((t.get_text(), box))
    for i, (name_a, a) in enumerate(boxes):
        for name_b, b in boxes[i + 1 :]:
            if a.overlaps(b):
                raise SystemExit(f"labels overlap: {name_a!r} and {name_b!r}")


def main() -> None:
    plt.style.use("data-ink")
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(14, 7.2), dpi=100, width_ratios=[1, 1.35]
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.90, bottom=0.22, wspace=0.32)
    fig.patch.set_facecolor("white")
    for ax in (ax0, ax1):
        ax.set_facecolor("white")

    cats = ["export gate\n(blocks)", "draft\n(reported)", "dormant\n(exhausted)"]
    keys = ["gate", "draft", "dormant"]
    colours = [GATE, DRAFT, NEUTRAL]
    h = 0.34
    for i, (key, colour) in enumerate(zip(keys, colours)):
        y0 = i + h / 2
        ax0.barh(y0, BEFORE[key], height=h, color=colour, alpha=0.45, zorder=2)
        ax0.barh(y0 - h / 2 - 0.02, NOW[key], height=h, color=colour, zorder=3)
        ax0.text(
            BEFORE[key] + 0.12, y0, f"before {BEFORE[key]}",
            va="center", ha="left", fontsize=14, color=colour, zorder=4,
        )
        ax0.text(
            NOW[key] + 0.12, y0 - h / 2 - 0.02, f"now {NOW[key]}",
            va="center", ha="left", fontsize=14, color=colour, zorder=4,
        )
    ax0.set_yticks(range(len(cats)), cats)
    ax0.set_xlim(0, 8.6)
    ax0.set_xticks([0, 2, 4, 6, 8])
    ax0.set_xlabel("roster names [count]")
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.tick_params(axis="y", length=0)
    ax0.set_title("seven roster stubs, gate's own split", fontsize=16, loc="left")

    labels = [name.replace("_", " ") for name, _ in STATUS]
    values = [n for _, n in STATUS]
    ax1.barh(range(len(values)), values, height=0.62, color=NEUTRAL, zorder=2)
    for i, value in enumerate(values):
        ax1.text(
            value + 40, i, f"{value:,}", va="center", ha="left", fontsize=14, zorder=3
        )
    ax1.set_yticks(range(len(values)), labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 3600)
    ax1.set_xticks([0, 1_000, 2_000, 3_000])
    ax1.set_xlabel(
        "non-roster sources [count]\n"
        "in the window: 0 created, 0 composed, 0 name changes"
    )
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    ax1.set_title("whole graph, roster excluded: 10,067 sources", fontsize=16, loc="left")

    check(fig)
    out = Path(__file__).with_suffix(".svg")
    fig.savefig(out, format="svg", facecolor="white", transparent=False, metadata={"Date": None})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
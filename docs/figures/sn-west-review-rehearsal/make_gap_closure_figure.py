"""Figures for the WEST batch gap-closure record.

Two panels, because the section turned on two different numbers being wrong:
acceptance against the manifest over the session, and the double-name defect
re-classified by whether it can actually reach the exported catalog.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent

INK = "#1b2733"
MUTED = "#7a8899"
GOOD = "#0a7d5a"
WARN = "#b5651d"
BAD = "#a02c2c"
COOL = "#2d6ca2"

# Acceptance against the 355-source manifest at each measured point this session.
MEASURED_POINTS = [
    ("plan's\nlast measure", 296),
    ("re-measured\nat entry", 289),
    ("after the\n/z fold", 289),
    ("after the\nreview drain", 301),
]

# The double-name defect, re-classified. The first number is what a
# PRODUCED_NAME count reports; the rest partition the HAS_STANDARD_NAME
# realizations, which is what the export actually reads.
AMBIGUITY = [
    ("counted on\nprovenance edges", 37, MUTED),
    ("only one end\nexported", 13, COOL),
    ("parent/child by\nconstruction", 8, COOL),
    ("genuinely ambiguous\nin the export", 14, BAD),
]


def acceptance_panel(ax) -> None:
    labels = [s[0] for s in MEASURED_POINTS]
    vals = [s[1] for s in MEASURED_POINTS]
    x = range(len(vals))
    ax.plot(x, vals, "-o", color=COOL, lw=2, ms=7, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(
            f"{v}",
            (i, v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
    ax.axhline(355, color=MUTED, ls=":", lw=1)
    ax.annotate(
        "355 eligible WEST sources",
        (len(vals) - 1, 355),
        textcoords="offset points",
        xytext=(0, -14),
        ha="right",
        fontsize=8.5,
        color=MUTED,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.set_ylim(275, 362)
    ax.set_ylabel("sources with an accepted name", fontsize=9.5, color=INK)
    ax.set_title(
        "Acceptance moved 289 → 301; the entry number was 7 below the plan's",
        fontsize=10.5,
        color=INK,
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6eaef", lw=0.8)
    ax.set_axisbelow(True)


def ambiguity_panel(ax) -> None:
    labels = [a[0] for a in AMBIGUITY]
    vals = [a[1] for a in AMBIGUITY]
    colors = [a[2] for a in AMBIGUITY]
    y = range(len(vals))
    ax.barh(list(y), vals, color=colors, height=0.62, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(
            f"{v}",
            (v, i),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 44)
    ax.set_xlabel("WEST DD paths", fontsize=9.5, color=INK)
    ax.set_title(
        "Only 14 of the 37 can reach the catalog ambiguous",
        fontsize=10.5,
        color=INK,
        pad=10,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#e6eaef", lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    fig.patch.set_facecolor("white")
    acceptance_panel(axes[0])
    ambiguity_panel(axes[1])
    fig.tight_layout(pad=1.8)
    fig.savefig(OUT / "gap-closure.png", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'gap-closure.png'}")


main()

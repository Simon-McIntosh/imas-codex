"""Render the JT-60SA mapping-study agreement figure.

One horizontal stacked bar per system (§8 to §13), each segment counting rows
by how imas-ambix's score-handoff placed the generated binding against its
hand-built tokamap: agreeing, disagreeing, unplaced or refused.

Counts are transcribed from the imas-ambix score JSONs cited in the study
document (`docs/research/jt60sa-mapping-study.html`); the flux-loop bar marks
the three cross-structure rows pending imas-ambix's index and scale check.

Run from the repository root with the shared project environment:

    uv run --no-sync python scripts/figures_jt60sa_mapping_study.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures/jt60sa-mapping-study/agreement.svg"

# Colour names a disposition and keeps naming it across the study.
AGREE = "#1baf7a"
DISAGREE = "#eb6834"
UNPLACED = "#9a9892"
REFUSED = "#4a3aa7"

# Per section: (label, agreeing, disagreeing, unplaced, refused, pending).
# The flux-loop row (§10) is the flux-loop subset of the §9 magnetics run and
# is pending imas-ambix's index and scale check, so its three cross-structure
# rows are drawn as disagreeing and flagged pending.
ROWS = [
    ("§8  PF and TF coils", 7, 6, 12, 1, False),
    ("§9  Magnetics", 10, 0, 7, 2, False),
    ("§10  Flux loops", 0, 3, 0, 0, True),
    ("§11  Electron cyclotron heating", 0, 0, 18, 3, False),
    ("§12  Gas injection", 0, 0, 5, 0, False),
    ("§13  Cryogenic group", 0, 0, 326, 0, False),
]


def render() -> Path:
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 6.4))
    labels = [r[0] for r in ROWS]
    series = [
        (AGREE, "agreeing"),
        (DISAGREE, "disagreeing"),
        (UNPLACED, "unplaced"),
        (REFUSED, "refused"),
    ]
    # Bars are normalised to 100% of each system's dispositioned rows, so the
    # composition is legible across systems whose absolute row counts differ by
    # two orders of magnitude. Counts are direct-labelled inside each segment
    # and the absolute total sits at the bar's right end.
    ypos = list(range(len(ROWS)))[::-1]
    totals = [r[1] + r[2] + r[3] + r[4] for r in ROWS]
    left = [0.0] * len(ROWS)
    for idx, (colour, name) in enumerate(series):
        counts = [r[1 + idx] for r in ROWS]
        width = [100.0 * c / t for c, t in zip(counts, totals, strict=True)]
        ax.barh(
            ypos,
            width,
            left=left,
            height=0.62,
            color=colour,
            edgecolor="white",
            linewidth=2.0,
            label=name,
            zorder=3,
        )
        # Direct-label each segment that is wide enough to hold its count.
        for y, c, w, l0 in zip(ypos, counts, width, left, strict=True):
            if c == 0:
                continue
            if w >= 7:
                ax.text(
                    l0 + w / 2,
                    y,
                    f"{c:d}",
                    va="center",
                    ha="center",
                    fontsize=20,
                    color="white",
                    zorder=5,
                )
            else:
                ax.text(
                    l0 + w / 2,
                    y + 0.40,
                    f"{c:d}",
                    va="bottom",
                    ha="center",
                    fontsize=18,
                    color=colour,
                    zorder=5,
                )
        left = [l0 + w for l0, w in zip(left, width, strict=True)]

    # Absolute row count (or the pending flag) at each bar's right end.
    for y, total, r in zip(ypos, totals, ROWS, strict=True):
        ax.text(
            101,
            y,
            ("pending" if r[5] else f"{total:d} rows"),
            va="center",
            ha="left",
            fontsize=18,
            color="#52514e",
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("share of dispositioned rows [%]")
    ax.set_xlim(0, 112)
    ax.margins(y=0.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=4)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(render())

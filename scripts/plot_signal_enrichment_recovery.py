"""Compare grounded description delivery with elapsed cohort throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from imas_codex.plotting import save_figure
from imas_codex.plotting.style import BASE_STYLE, CANVAS, REFERENCE, STYLE, SUBJECT


def draw(report: dict, output: Path, preview: Path):
    plt.style.use(BASE_STYLE)
    plt.rcParams.update(STYLE)
    rows = report["figure_rows"]
    figure, axes = plt.subplots(1, 2, figsize=CANVAS, sharey=True)
    figure.subplots_adjust(left=0.29, right=0.96, bottom=0.15, top=0.86, wspace=0.32)
    for index, row in enumerate(rows):
        selected, current = row.get("selected", False), row.get("current", False)
        color = SUBJECT if selected else REFERENCE
        marker = "D" if selected else "s" if current else "o"
        values = (row["delivered"], row["seconds_per_thousand"] / 60)
        for axis, value in zip(axes, values, strict=True):
            axis.scatter(
                [value],
                [index],
                color=color,
                marker=marker,
                s=70,
                zorder=3 if selected else 2,
            )
            axis.annotate(
                f"{value:.0f}" if axis is axes[0] else f"{value:.1f}",
                (value, index),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=14,
                color=color,
            )
    axes[0].set_yticks(range(len(rows)), [row["label"] for row in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlim(min(row["delivered"] for row in rows) - 2, 203)
    axes[0].set_xticks(sorted({min(row["delivered"] for row in rows), 200}))
    axes[0].axvline(200, color=REFERENCE, linestyle=":", linewidth=1.2, zorder=1)
    axes[0].set_xlabel("Descriptions delivered [of 200]")
    maximum = max(row["seconds_per_thousand"] / 60 for row in rows)
    axes[1].set_xlim(0, maximum * 1.22)
    axes[1].set_xlabel("Wall time [min / 1000 signals]")
    axes[1].tick_params(axis="y", left=False)
    axes[1].spines["left"].set_visible(False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color=REFERENCE,
            label="Settings in use",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=REFERENCE,
            label="Other variants",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            color=SUBJECT,
            label="Selected proposal",
        ),
        Line2D(
            [],
            [],
            linestyle=":",
            linewidth=1.2,
            color=REFERENCE,
            label="Required delivery",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=2,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output)
    save_figure(figure, preview)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.report.read_text()), args.output, args.preview)


if __name__ == "__main__":
    main()

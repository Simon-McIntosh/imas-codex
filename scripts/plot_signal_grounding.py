"""Plot sealed manual error counts beside private replay delivery rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iter_codex.plotting import save_figure
from iter_codex.plotting.style import BASE_STYLE, CANVAS, REFERENCE, STYLE, SUBJECT


def draw(report: dict, output: Path, preview: Path) -> None:
    plt.style.use(BASE_STYLE)
    plt.rcParams.update(STYLE)
    rows = report["figure_rows"]
    figure, axes = plt.subplots(1, 2, figsize=CANVAS, sharey=True)
    figure.subplots_adjust(left=0.31, right=0.96, bottom=0.15, top=0.96, wspace=0.3)
    y = list(range(len(rows)))
    colors = [SUBJECT if row.get("selected") else REFERENCE for row in rows]
    axes[0].scatter([row["wrong"] for row in rows], y, color=colors, s=70, zorder=3)
    axes[1].scatter(
        [100 * row["delivery"] for row in rows], y, color=colors, s=70, zorder=3
    )
    axes[0].set_yticks(y, [f"{row['label']} (n={row['n']})" for row in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlim(-1, max(row["n"] for row in rows) + 3)
    axes[1].set_xlim(-3, 112)
    axes[1].set_xticks([0, 50, 100])
    axes[0].set_xlabel("Wrong descriptions by hand")
    axes[1].set_xlabel("Descriptions delivered [%]")
    axes[1].tick_params(axis="y", left=False)
    axes[1].spines["left"].set_visible(False)
    for index, row in enumerate(rows):
        axes[0].annotate(
            str(row["wrong"]),
            (row["wrong"], index),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            color=colors[index],
            fontsize=14,
        )
        axes[1].annotate(
            f"{100 * row['delivery']:.0f}",
            (100 * row["delivery"], index),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            color=colors[index],
            fontsize=14,
        )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
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

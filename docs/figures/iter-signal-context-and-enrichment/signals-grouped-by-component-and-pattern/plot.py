"""Render measured group-size distributions and component pattern counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from imas_codex.plotting.style import (
    BASE_STYLE,
    CANVAS,
    REFERENCE,
    SUBJECT,
    save_figure,
)


def plot_grouping(receipt: dict, output: Path, preview: Path) -> None:
    """Use the shared checked exporter and existing subject/reference colours."""
    data = receipt["first"]
    plt.style.use(BASE_STYLE)
    figure, axes = plt.subplots(1, 2, figsize=CANVAS)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.19, top=0.86, wspace=0.32)
    for key, label, colour, style, zorder in (
        ("control_group_size_distribution", "control nodes", REFERENCE, "--", 2),
        ("group_size_distribution", "components", SUBJECT, "-", 3),
    ):
        histogram = sorted((int(size), count) for size, count in data[key].items())
        sizes, counts = np.array(histogram).T
        cumulative = np.cumsum(counts) * 100 / np.sum(counts)
        axes[0].step(
            sizes,
            cumulative,
            where="post",
            label=label,
            color=colour,
            linestyle=style,
            linewidth=3 if label == "components" else 2.6,
            zorder=zorder,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("variables per group")
    axes[0].set_ylabel("groups at or below size [%]")
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    maximum = max(
        int(size)
        for key in ("group_size_distribution", "control_group_size_distribution")
        for size in data[key]
    )
    axes[0].set_xlim(1, maximum * 1.1)
    axes[0].set_xticks([10**power for power in range(int(np.log10(maximum)) + 1)])
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.08, 0.99),
        ncol=2,
    )

    values = [data["field_patterns"], data["category_patterns"]]
    bars = axes[1].bar([0, 1], values, color=SUBJECT, width=0.55, zorder=3)
    axes[1].set_xticks([0, 1], ["category + fields", "category only"])
    axes[1].set_ylabel("pattern descriptions")
    upper = max(values) * 1.22
    ticks = MaxNLocator(nbins=5, integer=True).tick_values(0, upper)
    axes[1].set_yticks([tick for tick in ticks if 0 <= tick <= upper])
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].set_ylim(0, upper)
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.035,
            f"{value:,}",
            ha="center",
            va="bottom",
        )
    save_figure(figure, output)
    save_figure(figure, preview)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    plot_grouping(json.loads(args.receipt.read_text()), args.output, args.preview)
    print(f"Checked figure: {args.output}")
    print(f"Preview for visual inspection: {args.preview}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Draw, for every stage, the share of names that carried each expected channel.

Rows are the model-calling stages; columns are the channels

Left: one cell per stage and channel, coloured by the share of the names that
expected that channel and carried it, with the share written in the cell. A
channel a stage does not expect is left blank.

Every number is read from the census JSON committed beside the evidence page,
so regenerating the figure re-derives its numbers and a drifting census shows up
as a drifting figure. Run from the repository root; the SVG lands beside this
script and the evidence page embeds it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from iter_codex.plotting.style import save_figure, use_style  # noqa: E402

EVIDENCE = Path("docs/evidence/fragments/sn-case-file-generation/context-receipts.json")
SVG = Path("docs/figures/sn-case-file-generation/context-receipts/channel-coverage.svg")


def load() -> tuple[list[str], list[str], dict[str, dict[str, dict]], int]:
    report = json.loads(EVIDENCE.read_text())
    table = report["populations"]["all"]["table"]
    stages = list(table)
    channels: list[str] = []
    for stage in stages:
        for channel in table[stage]:
            if channel not in channels:
                channels.append(channel)
    return stages, channels, table, len(report["rows"])


def draw(stages, channels, table, names, path) -> None:
    shares = np.full((len(channels), len(stages)), np.nan)
    for row, channel in enumerate(channels):
        for column, stage in enumerate(stages):
            block = table[stage].get(channel)
            if block is None or not block["expected"]:
                continue
            shares[row, column] = 100.0 * block["present"] / block["expected"]

    figure, axes = plt.subplots(figsize=(14, 7.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    image = axes.imshow(shares, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    axes.set_xticks(range(len(stages)), stages, rotation=90, ha="center")
    axes.set_yticks(range(len(channels)), channels)
    axes.set_xlabel("stage")
    axes.set_ylabel("channel the stage expects")

    for row in range(len(channels)):
        for column in range(len(stages)):
            value = shares[row, column]
            if np.isnan(value):
                continue
            axes.text(
                column,
                row,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color="white" if value < 55 else "#202020",
            )

    axes.tick_params(length=0)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    bar = figure.colorbar(image)
    bar.set_label("names carrying the channel [%]")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)

    save_figure(figure, path)


def main() -> None:
    use_style()
    stages, channels, table, names = load()
    draw(stages, channels, table, names, SVG)
    print(f"{len(stages)} stages x {len(channels)} channels over {names} names")
    print(f"wrote {SVG}")


if __name__ == "__main__":
    main()

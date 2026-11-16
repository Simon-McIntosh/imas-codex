"""Plot raw parsing and delivered coverage from the private replay summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imas_codex.plotting.style import (
    BASE_STYLE,
    CANVAS,
    REFERENCE,
    STYLE,
    SUBJECT,
    save_figure,
)


def plot(summary_path: Path, output: Path, preview: Path):
    summary = json.loads(summary_path.read_text())
    names = {
        "baseline-first": "Baseline, first run",
        "baseline-second": "Baseline, second run",
        "schema": "Schema constraint",
        "json-mode": "JSON mode",
        "prompt": "Fitted output prompt",
        "thinking": "Thinking, larger token bound",
        "repair": "One validation re-ask",
        "batch-five": "Batch size five",
        "batch-three": "Batch size three",
        "batch-one": "Batch size one",
        "combination": "Combined changes, JSON mode",
        "combination-without-schema": "Combined changes, no constraint",
    }
    arms = [arm for arm in names if arm in summary]
    plt.style.use(BASE_STYLE)
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=CANVAS)
    fig.subplots_adjust(left=0.31, right=0.94, bottom=0.13, top=0.85)
    y = list(range(len(arms)))
    ax.scatter(
        [100 * summary[a]["raw_parse_rate"] for a in arms],
        [v - 0.13 for v in y],
        color=REFERENCE,
        marker="o",
        s=64,
        label="Raw responses parsed",
        zorder=2,
    )
    ax.scatter(
        [100 * summary[a]["delivery_rate"] for a in arms],
        [v + 0.13 for v in y],
        color=SUBJECT,
        marker="D",
        s=64,
        label="Signals delivered",
        zorder=3,
    )
    ax.set_yticks(y, [names[a] for a in arms])
    ax.invert_yaxis()
    ax.set_ylim(len(arms) - 0.4, -0.6)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share [%]")
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(loc="upper center", ncol=2, frameon=False)
    save_figure(fig, output)
    save_figure(fig, preview)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("preview", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output, args.preview)


if __name__ == "__main__":
    main()

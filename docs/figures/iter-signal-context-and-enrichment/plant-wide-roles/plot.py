"""Render the decided split and the role membership of the plant-wide assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iter_codex.plotting.style import BASE_STYLE, SUBJECT, save_figure

SOURCE_COLOURS = {
    "signal_list_row": "#b8327e",
    "controls_metadata": "#2e8b57",
    "field_pattern": SUBJECT,
}
SOURCE_LABELS = {
    "signal_list_row": "signal-list row",
    "controls_metadata": "controls metadata",
    "field_pattern": "field-code table",
}
AXES = ("quantity", "unit", "role")


def plot_split(receipt: dict, output: Path, preview: Path) -> None:
    summary = receipt["summary"]
    plt.style.use(BASE_STYLE)
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 7.2))
    figure.subplots_adjust(left=0.09, right=0.97, bottom=0.17, top=0.89, wspace=0.34)

    width = 0.26
    for offset, source in enumerate(SOURCE_COLOURS):
        counts = [summary[axis]["decided_by_source"].get(source, 0) for axis in AXES]
        positions = [index + (offset - 1) * width for index in range(len(AXES))]
        bars = left.bar(
            positions,
            counts,
            width=width,
            color=SOURCE_COLOURS[source],
            label=SOURCE_LABELS[source],
        )
        for bar, count in zip(bars, counts, strict=True):
            if not count:
                continue
            left.annotate(
                f"{count:,}",
                (bar.get_x() + bar.get_width() / 2, count),
                ha="center",
                va="bottom",
                fontsize=10,
                color=SOURCE_COLOURS[source],
                xytext=(0, 2),
                textcoords="offset points",
            )
    left.set_xticks(range(len(AXES)), [axis.capitalize() for axis in AXES])
    left.set_ylabel("signals decided")
    left.set_ylim(0, 2000)
    left.set_yticks([0, 500, 1000, 1500, 2000])
    left.legend(frameon=False, loc="upper right", fontsize=11)

    members = summary["role"]["decided_members"]
    roles = sorted(members, key=members.get)
    colours = ["#e07b1a" if role == "setpoint" else SUBJECT for role in roles]
    bars = right.barh(roles, [members[role] for role in roles], color=colours)
    for bar, role in zip(bars, roles, strict=True):
        right.annotate(
            f"{members[role]:,}",
            (members[role], bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=10,
            xytext=(4, 0),
            textcoords="offset points",
        )
    right.set_xlabel("signals decided as the role")
    right.set_xlim(0, 1650)
    right.set_xticks([0, 500, 1000, 1500])

    save_figure(figure, output, decoration=list(left.patches) + list(right.patches))
    save_figure(figure, preview, decoration=list(left.patches) + list(right.patches))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("preview", type=Path)
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text())
    plot_split(receipt, arguments.output, arguments.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

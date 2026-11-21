"""Render the private metadata checkpoint and separate frozen pilot audit."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from iter_codex.plotting.style import (
    BASE_STYLE,
    CANVAS,
    REFERENCE,
    STYLE,
    SUBJECT,
    save_figure,
)

plt.style.use(BASE_STYLE)
plt.rcParams.update(STYLE)
figure, axis = plt.subplots(figsize=CANVAS)
axis.set_axis_off()
axis.set(xlim=(0, 1), ylim=(0, 1))
figure.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)

stages = [
    (
        0.20,
        0.84,
        "Independent census",
        "988,145 slow + 6,861 fast\n995,006 distinct selectors",
    ),
    (
        0.70,
        0.84,
        "Bounded native readers",
        "Disjoint selector ranges, at most eight\nType and full metadata; no data reads",
    ),
    (
        0.70,
        0.49,
        "Private SQLite catalogue",
        "Documents, states, error codes\nAll listing memberships retained",
    ),
    (
        0.20,
        0.49,
        "Resume or cached repeat",
        "Commit each bounded batch\nSkip every completed metadata outcome",
    ),
    (
        0.20,
        0.14,
        "Frozen pilot graph",
        "94,633 existing signals\nNo connection from the metadata scan",
    ),
    (
        0.70,
        0.14,
        "Separate population audit",
        "Read-only counts around the scan\nBefore and after must agree",
    ),
]
for x, y, label, detail in stages:
    axis.text(x, y, label, ha="center", va="center", fontsize=16, color=SUBJECT)
    axis.text(x, y - 0.085, detail, ha="center", va="center", fontsize=14)
for start, end in [
    ((0.40, 0.80), (0.48, 0.80)),
    ((0.70, 0.68), (0.70, 0.56)),
    ((0.48, 0.46), (0.40, 0.46)),
    ((0.48, 0.11), (0.40, 0.11)),
]:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.2,
            color=REFERENCE,
        )
    )
output = Path(__file__).parent / "catalogue-recovery.svg"
save_figure(figure, output)
print("Checked text bounds and overlaps:", output)
if preview := os.environ.get("CATALOGUE_PREVIEW"):
    save_figure(figure, Path(preview))
    print("Rendered inspection image:", preview)

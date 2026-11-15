"""Render the census, durable metadata and ingest acknowledgement boundary."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from imas_codex.plotting.style import (
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
        0.15,
        0.84,
        "Independent census",
        "988,145 slow + 6,861 fast\n995,006 distinct selectors",
    ),
    (
        0.65,
        0.84,
        "Bounded metadata requests",
        "One request per variable\n500 variables per checkpoint",
    ),
    (
        0.65,
        0.49,
        "Durable metadata checkpoint",
        "Full documents and archive membership\nPer-variable failures retain their errors",
    ),
    (
        0.15,
        0.49,
        "Pipeline ingest owner",
        "Refresh archive metadata\nPreserve enrichment and serving provenance",
    ),
    (
        0.15,
        0.14,
        "Acknowledged graph identities",
        "Mark batch written only after acknowledgement\nCompleted batches are never fetched again",
    ),
]
for x, y, label, detail in stages:
    axis.text(x, y, label, ha="center", va="center", fontsize=16, color=SUBJECT)
    axis.text(x, y - 0.085, detail, ha="center", va="center", fontsize=14)
for start, end in [
    ((0.35, 0.80), (0.43, 0.80)),
    ((0.65, 0.68), (0.65, 0.56)),
    ((0.43, 0.46), (0.35, 0.46)),
    ((0.15, 0.33), (0.15, 0.21)),
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
axis.text(
    0.65,
    0.14,
    "On interruption: replay cached metadata\nwhose graph write was not acknowledged",
    ha="center",
    va="center",
    fontsize=14,
)
output = Path(__file__).parent / "catalogue-recovery.svg"
save_figure(figure, output)
print("Checked text bounds and overlaps:", output)

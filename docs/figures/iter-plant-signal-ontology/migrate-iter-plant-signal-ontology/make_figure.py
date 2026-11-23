"""Draw the ITER plant ontology's before and after counts on one axis.

Every row is a class or relationship the approved ontology decides. A grey
marker is the count read before the migration and a blue marker the count read
after it (measured live, 2026-11-23). Retired names fall to zero and their
replacements rise from zero; the two magnitudes are the same, which is what the
figure is for.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iter_codex.plotting import save_figure
from iter_codex.plotting.style import BASE_STYLE, CANVAS, REFERENCE, STYLE, SUBJECT

# A count of zero cannot be placed on a log axis, so it sits on its own tick.
ZERO = 0.5
TICKS = [ZERO, 1, 10, 100, 1000, 10000]
TICK_LABELS = ["0", "1", "10", "100", "1 000", "10 000"]

ROWS = [
    ("PlantStructure", 18, 0),
    ("PlantBreakdownNode", 0, 18),
    ("PlantContextDocument", 7, 0),
    ("Document (ITER)", 808, 815),
    ("ArchiveFieldCode", 0, 259),
    ("PARENT_PLANT_STRUCTURE", 16, 0),
    ("PARENT_PLANT_NODE", 0, 16),
    ("CONTEXT_FOR_PLANT", 7, 0),
    ("DOCUMENT_PLANT_STRUCTURE", 0, 7),
    ("CANDIDATE_PLANT_COMPONENT", 30031, 0),
    ("HAS_FIELD_CODE", 0, 30031),
    ("IN_ARCHIVE_CONTROL_NODE", 60283, 0),
    ("FIELD_CODE_SOURCE_ROW", 0, 13),
    ("CONTROL_NODE_PLANT_STRUCTURE", 0, 4),
]


def _position(value: int) -> float:
    return ZERO if value == 0 else float(value)


def draw(output) -> None:
    plt.style.use(BASE_STYLE)
    plt.rcParams.update(STYLE)
    figure, axis = plt.subplots(figsize=CANVAS)
    figure.subplots_adjust(left=0.30, right=0.90, bottom=0.13, top=0.88)
    y = list(range(len(ROWS)))
    leaders = []
    for index, (_, before, after) in enumerate(ROWS):
        start, end = _position(before), _position(after)
        (leader,) = axis.plot(
            [start, end], [index, index], color=REFERENCE, linewidth=1.2, zorder=1
        )
        leaders.append(leader)
    axis.scatter(
        [_position(before) for _, before, _ in ROWS],
        y,
        color=REFERENCE,
        s=110,
        zorder=3,
        label="before",
    )
    axis.scatter(
        [_position(after) for _, _, after in ROWS],
        y,
        color=SUBJECT,
        s=110,
        zorder=4,
        label="after",
    )
    axis.set_xscale("log")
    axis.set_xlim(0.38, 95000)
    axis.set_xticks(TICKS, TICK_LABELS)
    axis.set_yticks(y, [name for name, _, _ in ROWS])
    axis.invert_yaxis()
    axis.set_xlabel("ITER nodes and relationships [count]")
    axis.tick_params(axis="y", left=False)
    for axis_edge in ("top", "right", "left"):
        axis.spines[axis_edge].set_visible(False)
    axis.grid(False)
    for index, (_, before, after) in enumerate(ROWS):
        axis.annotate(
            f"{before:,}".replace(",", " "),
            (_position(before), index),
            xytext=(-9, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color="#3d3d3d",
            fontsize=14,
        )
        axis.annotate(
            f"{after:,}".replace(",", " "),
            (_position(after), index),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="#1f1f1f",
            fontsize=14,
        )
    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=14,
        markerscale=None,
    )
    save_figure(figure, output, decoration=leaders)


if __name__ == "__main__":
    from pathlib import Path

    draw(Path(__file__).with_name("migration-counts.svg"))

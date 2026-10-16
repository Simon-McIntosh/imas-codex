"""Draw where the twenty-five corrected IDS nodes land, by target domain.

Reads the reversal record written by the refresh so the figure is derived from
the rows the apply actually moved rather than from the map file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
STEM = "scg-ids-domain-refresh"
ROWS = (
    ROOT
    / "docs/evidence/fragments/sn-catalog-grouping"
    / "scg-ids-domain-refresh-rows.json"
)
FIGURE = ROOT / "docs/figures/sn-catalog-grouping" / STEM / "ids_moved_by_target.svg"


def main() -> None:
    summary = json.loads(ROWS.read_text())
    targets = Counter(row["after"] for row in summary["changed_ids"])
    labels = sorted(targets, key=lambda name: (targets[name], name))
    counts = [targets[name] for name in labels]

    style = Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle"
    if style.exists():
        plt.style.use(style)
    plt.rcParams["svg.hashsalt"] = STEM

    colour = "#2f7f74"
    fig, axis = plt.subplots(figsize=(14, 6))
    y = range(len(labels))
    axis.barh(list(y), counts, height=0.62, color=colour, zorder=3)
    for position, count in zip(y, counts, strict=True):
        axis.text(
            count + 0.12,
            position,
            str(count),
            va="center",
            ha="left",
            fontsize=14,
            color=colour,
        )
    axis.set_yticks(list(y), labels, fontsize=14)
    axis.set_xlim(0, max(counts) + 1)
    axis.set_xlabel("IDS nodes moved into the target domain", fontsize=16)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    fig.subplots_adjust(left=0.28, right=0.96, bottom=0.13, top=0.97)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    tight = fig.get_tightbbox(renderer)
    assert canvas.contains(*tight.get_points()[0]) and canvas.contains(
        *tight.get_points()[1]
    ), ("text escapes the canvas", tight.bounds, canvas.bounds)
    boxes: list[tuple[str, object]] = []
    for artist in fig.findobj(match=plt.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if not all(canvas.contains(*corner) for corner in box.get_points()):
            continue
        for prior, prior_box in boxes:
            assert not box.overlaps(prior_box), (prior, artist.get_text())
        boxes.append((artist.get_text(), box))

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg", metadata={"Date": "2026-10-16"})
    plt.close(fig)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()

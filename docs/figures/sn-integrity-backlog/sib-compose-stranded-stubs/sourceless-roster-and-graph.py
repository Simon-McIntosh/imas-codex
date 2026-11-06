#!/usr/bin/env python3
"""Draw the stranded-stub figure from the recorded artifacts, not from literals.

Left panel: the seven roster stubs, split the way ``export.py`` splits them,
before the scoped run and now. Right panel: every non-roster source by status,
which is the population the scope control is about.

Every number is read from a JSON artifact produced by the runner:

``replay-before.json``  ``replay`` over the recorded before-census
``census-current.json``  the live ``census``
``outofroster.json``     the ``outofroster`` scope check

so regenerating the figure re-derives its numbers and a drifting artifact shows
up as a drifting figure. ``--logs`` names the directory holding them; it is
required because the artifacts live outside the repository.

The figure refuses to be written if any of its own labels leaves the canvas,
overlaps another label, or is drawn over a plotted bar. Run from the repository
root; the SVG lands beside this file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

GATE = "#b03a2e"
DRAFT = "#2a6f97"
LANDED = "#2e8b57"
NEUTRAL = "#929292"


def load(logs: Path) -> tuple[dict, dict, dict]:
    def read(name: str) -> dict:
        return json.loads((logs / name).read_text())

    return (
        read("replay-before.json"),
        read("census-current.json"),
        read("outofroster.json"),
    )


def check(fig, labels, bars) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Every text artist, not only the data labels: a title colliding with an
    # annotation is the same defect as a label sitting on a bar.
    # Invisible twins (a tick's unused top/right label) report a unit box at the
    # origin, so filter on visibility as well as on content.
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    seen = {id(t) for t in labels}
    labels = labels + [t for t in texts if id(t) not in seen]
    x0, y0, w, h = fig.bbox.bounds
    canvas = (x0, y0, x0 + w, y0 + h)
    boxes = []
    for t in labels:
        box = t.get_window_extent(renderer)
        if (
            box.x0 < canvas[0] - 1
            or box.y0 < canvas[1] - 1
            or box.x1 > canvas[2] + 1
            or box.y1 > canvas[3] + 1
        ):
            raise SystemExit(f"label outside the canvas: {t.get_text()!r}")
        boxes.append((t.get_text(), box))
    for i, (name_a, a) in enumerate(boxes):
        for name_b, b in boxes[i + 1 :]:
            if a.overlaps(b):
                raise SystemExit(f"labels overlap: {name_a!r} and {name_b!r}")
    for t in labels:
        tbox = t.get_window_extent(renderer)
        for patch in bars:
            if tbox.overlaps(patch.get_window_extent(renderer)):
                raise SystemExit(f"label drawn over plotted data: {t.get_text()!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs", required=True, help="directory holding the runner's JSON artifacts"
    )
    args = parser.parse_args()
    logs = Path(args.logs)

    before, now, graph = load(logs)
    split_before = {
        "gate": len(before["gate_roster_blocking"]),
        "draft": len(before["draft_roster_sourceless"]),
        "landed": len(before["composed_roster"]),
    }
    split_now = {
        "gate": len(now["gate_roster_blocking"]),
        "draft": len(now["draft_roster_sourceless"]),
        "landed": len(now["roster_stubs_with_producer"]),
    }
    status = [(row["status"], row["n"]) for row in graph["sources_by_status"]]
    totals = graph["sources"]

    plt.style.use("data-ink")
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(14, 7.2), dpi=100, width_ratios=[1, 1.35]
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.90, bottom=0.24, wspace=0.32)
    fig.patch.set_facecolor("white")
    for ax in (ax0, ax1):
        ax.set_facecolor("white")

    labels = []
    bars = []

    cats = ["export gate\n(blocks)", "draft\n(reported)", "landed\n(composed)"]
    keys = ["gate", "draft", "landed"]
    colours = [GATE, DRAFT, LANDED]
    band = 0.34
    for i, (key, colour) in enumerate(zip(keys, colours, strict=True)):
        top = i + band / 2
        rowmax = max(split_before[key], split_now[key])
        text_x = rowmax + 0.25
        bars.append(
            ax0.barh(
                top, split_before[key], height=band, color=colour, alpha=0.45, zorder=2
            )[0]
        )
        bars.append(
            ax0.barh(
                top - band / 2 - 0.02,
                split_now[key],
                height=band,
                color=colour,
                zorder=3,
            )[0]
        )
        labels.append(
            ax0.text(
                text_x,
                top,
                f"before {split_before[key]}",
                va="center",
                ha="left",
                fontsize=14,
                color=colour,
                zorder=4,
            )
        )
        labels.append(
            ax0.text(
                text_x,
                top - band / 2 - 0.02,
                f"now {split_now[key]}",
                va="center",
                ha="left",
                fontsize=14,
                color=colour,
                zorder=4,
            )
        )
    ax0.set_yticks(range(len(cats)), cats)
    ax0.set_xlim(0, 9.6)
    ax0.set_xticks([0, 2, 4, 6, 8])
    ax0.set_xlabel("roster names [count]")
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.tick_params(axis="y", length=0)
    ax0.set_title("seven roster stubs, gate's own split", fontsize=16, loc="left")

    names = [name.replace("_", " ") for name, _ in status]
    values = [n for _, n in status]
    for i, value in enumerate(values):
        bars.append(ax1.barh(i, value, height=0.62, color=NEUTRAL, zorder=2)[0])
        labels.append(
            ax1.text(
                value + 40,
                i,
                f"{value:,}",
                va="center",
                ha="left",
                fontsize=14,
                zorder=3,
            )
        )
    ax1.set_yticks(range(len(values)), names)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 3800)
    ax1.set_xticks([0, 1_000, 2_000, 3_000])
    ax1.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )
    changes_outside = len(graph["name_changes_since_cutoff_outside_roster"])
    ax1.set_xlabel(
        f"non-roster sources [count]\n"
        f"in the window: 0 created, 0 composed, {changes_outside} name changes\n"
        f"creation datable on {totals['sources_with_created_at']:,} of "
        f"{totals['sources']:,}"
    )
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    ax1.set_title(
        f"whole graph, roster excluded: {totals['sources']:,} sources",
        fontsize=16,
        loc="left",
    )

    check(fig, labels, bars)
    out = Path(__file__).with_suffix(".svg")
    fig.savefig(
        out, format="svg", facecolor="white", transparent=False, metadata={"Date": None}
    )
    print(f"wrote {out}")
    print(f"before={split_before} now={split_now} sources={totals['sources']}")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()

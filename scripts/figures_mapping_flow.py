"""Draw where each facility's signals map into the IMAS Data Dictionary.

A ribbon chart on one slide-sized canvas. JET's signals stand on the left and
JT-60SA's on the right, each bar split into the signals already mapped, those
with a proposed target, and those still to map. Ribbons carry the mapped and
proposed signals into the IDSs they target, drawn as bars down the middle, and
every height is in proportion to a signal count. JT-60SA's mapped targets are
its MAPS_TO_IMAS edges and its proposals are each source's top-ranked
MAPPING_CANDIDATE. JET's mapped targets are the ``mapping_target_path`` stored
on each mapped source, because its MAPS_TO_IMAS edges are absent from the graph.
Signals are counted through MEMBER_OF, so only signals grouped into a signal
source appear. The top of the canvas is left clear for the slide title.

    uv run python scripts/figures_mapping_flow.py --query    # read the live graph
    uv run python scripts/figures_mapping_flow.py            # draw from the cache
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from figures_knowledge_graph_scale import (  # noqa: E402
    FIGURES,
    HEIGHT,
    SLATE,
    SLIDE_WIDTH,
    SYSTEMS,
    TITLE_SPACE,
    check_layout,
    human,
    save,
)

DATA = FIGURES / "mapping-flow-graph.json"
SVG = FIGURES / "mapping-flow.svg"
FACILITIES = ["jet", "jt-60sa"]

# Every source of a facility: its member signals, its stored targets and its
# top-ranked candidate path.
SOURCES = """
    MATCH (s:SignalSource {facility_id: $f})
    OPTIONAL MATCH (m:FacilitySignal)-[:MEMBER_OF]->(s)
    WITH s, count(m) AS members
    OPTIONAL MATCH (s)-[:MAPS_TO_IMAS]->(b:IMASNode)
    WITH s, members, collect(DISTINCT b.id) AS bound
    OPTIONAL MATCH (s)-[c:MAPPING_CANDIDATE]->(i:IMASNode)
    WITH s, members, bound, i ORDER BY c.rank
    WITH s, members, bound, collect(i.id)[0] AS proposed
    RETURN s.id AS source, members,
           CASE WHEN size(bound) > 0 THEN bound
                WHEN s.mapping_target_path IS NOT NULL THEN [s.mapping_target_path]
                ELSE [] END AS mapped,
           proposed
"""


def query(gc) -> dict:
    return {f: gc.query(SOURCES, f=f) for f in FACILITIES}


STAGES = ["mapped", "proposed", "to map"]


def flows(data: dict) -> tuple[dict, dict]:
    """Signals per (facility, stage, IDS), and signals per (facility, stage).

    A source mapped to several IDSs shares its signals among them evenly.
    """
    by_ids = defaultdict(float)
    totals = defaultdict(float)
    for f in FACILITIES:
        for row in data[f]:
            if row["mapped"]:
                stage, targets = "mapped", row["mapped"]
            elif row["proposed"]:
                stage, targets = "proposed", [row["proposed"]]
            else:
                totals[f, "to map"] += row["members"]
                continue
            totals[f, stage] += row["members"]
            ids = [t.split("/")[0] for t in targets]
            for name in ids:
                by_ids[f, stage, name] += row["members"] / len(ids)
    return by_ids, totals


INK = "#4a4a4a"
BOTTOM, TOP = 0.3, HEIGHT - TITLE_SPACE - 1.2  # the ribbons' vertical span
BAR = 0.3  # in, bar width
X = {"jet": 2.3, "dd": SLIDE_WIDTH / 2 - BAR / 2, "jt-60sa": 11.4}
GAP = 0.05  # in, between stacked bars
OTHER = 0.025  # IDSs taking less than this share of the middle column are merged
ALPHA = {"mapped": 0.85, "proposed": 0.3, "to map": 0.22}  # ribbons
SHADE = {"mapped": 1.0, "proposed": 0.6, "to map": 0.25}  # facility bars


def ribbon(ax, x0, y0, x1, y1, height, colour, alpha):
    """A smooth band of constant height from (x0, y0) to (x1, y1), by its base."""
    t = np.linspace(0, 1, 60)
    s = t * t * (3 - 2 * t)
    x = x0 + (x1 - x0) * t
    base = y0 + (y1 - y0) * s
    ax.fill_between(x, base, base + height, color=colour, alpha=alpha, lw=0, zorder=1)


def draw(data: dict, out, draft) -> None:
    by_ids, totals = flows(data)
    middle = defaultdict(float)
    for (_, _, name), n in by_ids.items():
        middle[name] += n
    whole = sum(middle.values())
    named = sorted(
        (k for k, v in middle.items() if v >= OTHER * whole), key=lambda k: -middle[k]
    )
    order = ["other IDSs", *named]  # the long tail sits level with the facility tops
    ids_of = {k: (k if k in named else "other IDSs") for k in middle}
    flow = defaultdict(float)
    for (f, stage, name), n in by_ids.items():
        flow[f, stage, ids_of[name]] += n

    columns = {f: sum(totals[f, s] for s in STAGES) for f in FACILITIES}
    unit = (TOP - BOTTOM - GAP * len(order)) / max(*columns.values(), whole)

    plt.style.use("data-ink")
    fig = plt.figure(figsize=(SLIDE_WIDTH, HEIGHT))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, SLIDE_WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    labels = []

    def text(x, y, s, **kw):
        kw.setdefault("fontsize", 15)
        kw.setdefault("color", INK)
        kw.setdefault("va", "center")
        labels.append(ax.text(x, y, s, zorder=6, **kw))

    # the middle column: one bar per IDS, fed from both sides, spread over the
    # full height so every IDS can carry its name
    spread = (TOP - BOTTOM - whole * unit) / (len(order) - 1)
    y = TOP
    slots = {}  # IDS: [left inflow cursor, right inflow cursor]
    for name in order:
        h = sum(flow[f, s, name] for f in FACILITIES for s in STAGES) * unit
        ax.add_patch(
            plt.Rectangle((X["dd"], y - h), BAR, h, color=SLATE, lw=0, zorder=3)
        )
        slots[name] = [y, y]
        text(
            X["dd"] - 0.08,
            y - h / 2,
            name.replace("_", " "),
            fontsize=14,
            color=SLATE,
            ha="right",
            bbox={"fc": "white", "ec": "none", "pad": 0.5, "alpha": 0.85},
        )
        y -= h + spread
    text(
        X["dd"] + BAR / 2,
        TOP + 0.75,
        "IMAS Data Dictionary",
        fontsize=22,
        fontweight="bold",
        color=SLATE,
        ha="center",
    )
    text(X["dd"] + BAR / 2, TOP + 0.3, f"{len(middle)} IDSs", ha="center")

    # each facility: a bar of its signals by stage, ribbons into the middle
    for f in FACILITIES:
        colour = SYSTEMS[f][1]
        left = f == "jet"
        x = X[f]
        side = {"ha": "right"} if left else {"ha": "left"}
        tx = x - 0.12 if left else x + BAR + 0.12
        text(
            x + BAR / 2,
            TOP + 0.75,
            SYSTEMS[f][0],
            fontsize=22,
            fontweight="bold",
            color=colour,
            ha="center",
        )
        text(x + BAR / 2, TOP + 0.3, f"{human(columns[f])} signals", ha="center")
        y = (BOTTOM + TOP + columns[f] * unit) / 2
        for stage in STAGES:
            if not totals[f, stage]:
                continue
            h = totals[f, stage] * unit
            ax.add_patch(
                plt.Rectangle(
                    (x, y - h), BAR, h, color=colour, lw=0, alpha=SHADE[stage], zorder=3
                )
            )
            text(tx, y - h / 2, f"{human(totals[f, stage])} {stage}", **side)
            cursor = y
            for name in order:
                hh = flow[f, stage, name] * unit
                if not hh:
                    continue
                slot = slots[name]
                if left:
                    ribbon(
                        ax,
                        x + BAR,
                        cursor - hh,
                        X["dd"],
                        slot[0] - hh,
                        hh,
                        colour,
                        ALPHA[stage],
                    )
                    slot[0] -= hh
                else:
                    ribbon(
                        ax,
                        X["dd"] + BAR,
                        slot[1] - hh,
                        x,
                        cursor - hh,
                        hh,
                        colour,
                        ALPHA[stage],
                    )
                    slot[1] -= hh
                cursor -= hh
            y -= h

    check_layout(fig, labels, refuse=draft is None)
    if draft:
        fig.savefig(draft, dpi=100, facecolor="white")
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", action="store_true", help="read the live graph")
    parser.add_argument("--draft", help="write a PNG here, reporting layout faults")
    args = parser.parse_args()
    if args.query:
        from imas_codex.graph.client import GraphClient

        DATA.write_text(json.dumps(query(GraphClient())) + "\n")
    draw(json.loads(DATA.read_text()), SVG, args.draft)


if __name__ == "__main__":
    main()

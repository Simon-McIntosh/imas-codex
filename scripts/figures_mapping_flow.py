"""Draw how ``imas-codex imas map`` turns graph content into IMAS mappings.

JET is drawn above and JT-60SA below, both feeding the IMAS Data Dictionary on
the right. Every disc is a node label, with its area in proportion to the live
node count. Solid lines are relationships the graph stores, drawn wider for
more edges. Dashed grey lines are context that ``map`` reads at run time and
does not record: semantic wiki and code hits, DD clusters, and the other
facility's bindings. The relationship labels name what each stored edge
carries, so the figure answers which parts of a mapping are nodes and which
are edges with attributes. The top of the canvas is left clear for the slide
title.

    uv run python scripts/figures_mapping_flow.py --query    # count the live graph
    uv run python scripts/figures_mapping_flow.py            # draw from the counts
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from figures_knowledge_graph_scale import (  # noqa: E402
    FIGURES,
    HEIGHT,
    SLATE,
    SLIDE_WIDTH,
    SYSTEMS,
    check_layout,
    human,
    save,
)
from matplotlib.patches import FancyArrowPatch  # noqa: E402

STATS = FIGURES / "mapping-flow-stats.json"
SVG = FIGURES / "mapping-flow.svg"
FACILITIES = ["jet", "jt-60sa"]

# Each query returns one row per facility: (facility, count).
COUNTS = {
    "wiki": "MATCH (n:WikiChunk) WHERE n.facility_id IN $f "
    "RETURN n.facility_id AS f, count(*) AS n",
    "code": "MATCH (n:CodeChunk) WHERE n.facility_id IN $f "
    "RETURN n.facility_id AS f, count(*) AS n",
    "signals": "MATCH (n:FacilitySignal) WHERE n.facility_id IN $f "
    "RETURN n.facility_id AS f, count(*) AS n",
    "sources": "MATCH (n:SignalSource) WHERE n.facility_id IN $f "
    "RETURN n.facility_id AS f, count(*) AS n",
    "validated": "MATCH (n:SignalSource) WHERE n.facility_id IN $f "
    "AND n.mapping_status = 'validated' RETURN n.facility_id AS f, count(*) AS n",
    "dispositions": "MATCH (n:SignalSource) WHERE n.facility_id IN $f "
    "AND n.mapping_disposition IS NOT NULL AND n.mapping_disposition <> 'mapped' "
    "RETURN n.facility_id AS f, count(*) AS n",
    "member_of": "MATCH (s:FacilitySignal)-[:MEMBER_OF]->(g:SignalSource) "
    "WHERE g.facility_id IN $f RETURN g.facility_id AS f, count(*) AS n",
    "candidate": "MATCH (g:SignalSource)-[:MAPPING_CANDIDATE]->(:IMASNode) "
    "WHERE g.facility_id IN $f RETURN g.facility_id AS f, count(*) AS n",
    "binding": "MATCH (g:SignalSource)-[:MAPS_TO_IMAS]->(:IMASNode) "
    "WHERE g.facility_id IN $f RETURN g.facility_id AS f, count(*) AS n",
    "binding_ids": "MATCH (g:SignalSource)-[:MAPS_TO_IMAS]->(i:IMASNode) "
    "WHERE g.facility_id IN $f "
    "RETURN g.facility_id AS f, count(DISTINCT split(i.id, '/')[0]) AS n",
    "mappings": "MATCH (m:IMASMapping) WHERE m.facility_id IN $f "
    "RETURN m.facility_id AS f, count(*) AS n",
    "uses": "MATCH (m:IMASMapping)-[:USES_SIGNAL_SOURCE]->(:SignalSource) "
    "WHERE m.facility_id IN $f RETURN m.facility_id AS f, count(*) AS n",
    "populates": "MATCH (m:IMASMapping)-[:POPULATES]->(:IMASNode) "
    "WHERE m.facility_id IN $f RETURN m.facility_id AS f, count(*) AS n",
    "evidence": "MATCH (g:SignalSource)-[:HAS_EVIDENCE]->(:MappingEvidence) "
    "WHERE g.facility_id IN $f RETURN g.facility_id AS f, count(*) AS n",
}
DD_COUNTS = {
    "paths": "MATCH (n:IMASNode) RETURN count(*) AS n",
    "clusters": "MATCH (n:IMASSemanticCluster) RETURN count(*) AS n",
}


def query_counts(gc) -> dict:
    stats = {f: {} for f in FACILITIES}
    for key, cypher in COUNTS.items():
        for f in FACILITIES:
            stats[f][key] = 0
        for row in gc.query(cypher, f=FACILITIES):
            stats[row["f"]][key] = row["n"]
    stats["dd"] = {key: gc.query(q)[0]["n"] for key, q in DD_COUNTS.items()}
    return stats


GREY = "#8c8c8c"  # context read at run time and never stored
LABEL_INK = "#4a4a4a"
RADIUS_PER_NODE = 0.5 / math.sqrt(120_000)  # in: disc area reads as node count
DD_AT = (12.1, 3.05)
MAPPING_AT = {"jet": (9.4, 5.45), "jt-60sa": (10.9, 0.62)}
ROW_Y = {"jet": 4.45, "jt-60sa": 1.65}
OUTWARD = {"jet": 1, "jt-60sa": -1}  # away from the gap between the rows
X = {"wiki": 0.75, "code": 2.15, "signals": 3.6, "sources": 5.7, "run": 7.7}


def radius(count: int) -> float:
    return max(0.07, RADIUS_PER_NODE * math.sqrt(count))


def width(count: int) -> float:
    return 1.0 + 1.1 * math.log10(max(count, 1))


def toward(a, ra, b):
    """The point on a disc of radius ``ra`` at ``a`` that faces ``b``."""
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    return a[0] + ra * (b[0] - a[0]) / d, a[1] + ra * (b[1] - a[1]) / d


def bend_point(a, b, rad):
    """Midpoint of matplotlib's arc3 curve from ``a`` to ``b``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    cx, cy = (a[0] + b[0]) / 2 + rad * dy, (a[1] + b[1]) / 2 - rad * dx
    return 0.25 * a[0] + 0.5 * cx + 0.25 * b[0], 0.25 * a[1] + 0.5 * cy + 0.25 * b[1]


def draw(stats: dict, out, draft) -> None:
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(SLIDE_WIDTH, HEIGHT))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, SLIDE_WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")
    labels = []

    def disc(at, count, colour):
        ax.add_patch(plt.Circle(at, radius(count), color=colour, lw=0, zorder=3))
        return at, radius(count)

    def text(x, y, s, **kw):
        kw.setdefault("fontsize", 14)
        kw.setdefault("color", LABEL_INK)
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        labels.append(ax.text(x, y, s, zorder=6, **kw))

    def boxed(x, y, s, colour, **kw):
        text(
            x,
            y,
            s,
            color=colour,
            fontsize=13,
            bbox={"fc": "white", "ec": "none", "pad": 1.0, "alpha": 0.92},
            **kw,
        )

    def link(a, b, *, count=1, colour=GREY, rad=0.0, dashed=False, head=True):
        (pa, ra), (pb, rb) = a, b
        ax.add_patch(
            FancyArrowPatch(
                toward(pa, ra, pb),
                toward(pb, rb, pa),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>" if head else "-",
                mutation_scale=13,
                lw=1.5 if dashed else width(count),
                ls=(0, (4, 3)) if dashed else "-",
                color=colour,
                alpha=0.8 if dashed else 0.6,
                shrinkA=3,
                shrinkB=3,
                zorder=2,
            )
        )

    dd = stats["dd"]
    ddisc = disc(DD_AT, dd["paths"], SLATE)
    ddx, ddy = DD_AT
    r_dd = radius(dd["paths"])
    text(
        ddx + 0.2,
        ddy + r_dd + 0.55,
        "IMAS Data\nDictionary",
        color=SLATE,
        fontsize=20,
        fontweight="bold",
        va="bottom",
    )
    text(
        ddx + 0.2,
        ddy + r_dd + 0.12,
        f"{human(dd['paths'])} paths, {human(dd['clusters'])} clusters",
        va="bottom",
    )

    rings = {}
    for f in FACILITIES:
        name, colour = SYSTEMS[f][0], SYSTEMS[f][1]
        s, y, o = stats[f], ROW_Y[f], OUTWARD[f]
        va_in = "top" if o > 0 else "bottom"  # text set below JET, above JT-60SA
        va_out = "bottom" if o > 0 else "top"
        wiki = disc((X["wiki"], y + o * 0.55), s["wiki"], colour)
        code = disc((X["code"], y + o * 0.55), s["code"], colour)
        signals = disc((X["signals"], y), s["signals"], colour)
        sources = disc((X["sources"], y), s["sources"], colour)
        run = ((X["run"], y), 0.24)
        rings[f] = run
        ax.add_patch(
            plt.Circle(run[0], run[1], fc="white", ec=colour, lw=2.4, zorder=4)
        )
        text(X["run"], y, "map", fontsize=13, color=colour)

        text(
            X["wiki"] - 0.35,
            y + o * 1.45,
            name,
            color=colour,
            fontsize=22,
            fontweight="bold",
            ha="left",
        )
        for (at, r), what, count in (
            (wiki, "wiki chunks", s["wiki"]),
            (code, "code chunks", s["code"]),
        ):
            text(at[0], at[1] - o * (r + 0.12), f"{what}\n{human(count)}", va=va_in)
        text(
            X["signals"],
            y - o * (signals[1] + 0.12),
            f"signals\n{human(s['signals'])}",
            va=va_in,
        )
        text(
            X["sources"],
            y - o * (sources[1] + 0.12),
            f"signal sources\n{human(s['sources'])}",
            va=va_in,
        )

        # stored inputs: signals grouped into sources
        link(signals, sources, count=s["member_of"], colour=colour)
        boxed(
            (X["signals"] + X["sources"]) / 2 + 0.1,
            y + o * 0.22,
            f"MEMBER_OF {human(s['member_of'])}",
            colour,
            va=va_out,
        )

        # read at run time, never stored
        link(sources, run, colour=GREY, dashed=True)
        link(wiki, run, colour=GREY, dashed=True, rad=-0.32 * o)
        link(code, run, colour=GREY, dashed=True, rad=-0.3 * o)
        link(ddisc, run, colour=GREY, dashed=True, rad=-0.3 * o)

        # what the run writes back, each edge named with what it carries
        written = []
        if s["candidate"]:
            written.append(
                (
                    s["candidate"],
                    -0.35 * o,
                    f"MAPPING_CANDIDATE {human(s['candidate'])}\n"
                    "rank, p(same quantity), model",
                )
            )
        if s["binding"]:
            written.append(
                (
                    s["binding"],
                    0.1 * o,
                    f"MAPS_TO_IMAS {human(s['binding'])}, "
                    f"{s['binding_ids']} IDSs\n"
                    "transform, units, confidence",
                )
            )
        for count, rad, label in written:
            link(run, ddisc, count=count, colour=colour, rad=rad)
            boxed(*bend_point(run[0], DD_AT, rad), label, colour)
        if not written:
            boxed(
                10.0,
                y + o * 0.1,
                "no MAPPING_CANDIDATE or\n"
                f"MAPS_TO_IMAS edges; {s['validated']}\n"
                "sources marked validated",
                colour,
            )

        mapping = disc(MAPPING_AT[f], s["mappings"], colour)
        uses_rad = 0.2 * o if o > 0 else -0.12
        link(mapping, sources, count=s["uses"], colour=colour, rad=uses_rad)
        boxed(
            *bend_point(mapping[0], sources[0], uses_rad),
            f"USES_SIGNAL_SOURCE {human(s['uses'])}",
            colour,
        )
        text(
            mapping[0][0] + 0.2,
            mapping[0][1],
            f"{s['mappings']} IMASMapping nodes",
            color=colour,
            ha="left",
        )
        if s["populates"]:
            link(mapping, ddisc, count=s["populates"], colour=colour, rad=0.25)
            px, py = bend_point(mapping[0], DD_AT, 0.25)
            boxed(px + 0.12, py, f"POPULATES {s['populates']}", colour, ha="left")
        if s["evidence"]:
            evidence = disc((X["sources"] + 0.55, y + o * 0.75), s["evidence"], colour)
            link(sources, evidence, count=s["evidence"], colour=colour)
            boxed(
                evidence[0][0] - 0.15,
                evidence[0][1],
                f"{s['evidence']} MappingEvidence\n(escalations)",
                colour,
                ha="right",
            )

    # one facility's mappings are read as examples when mapping another
    link(rings["jt-60sa"], rings["jet"], colour=GREY, dashed=True)
    boxed(
        X["run"],
        (ROW_Y["jet"] + ROW_Y["jt-60sa"]) / 2,
        "other facility's\nmappings",
        GREY,
    )
    boxed(4.4, 5.72, "read by map, no edge stored", GREY)

    check_layout(fig, labels, refuse=draft is None)
    if draft:
        fig.savefig(draft, dpi=100, facecolor="white")
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", action="store_true", help="count the live graph")
    parser.add_argument("--draft", help="write a PNG here, reporting layout faults")
    args = parser.parse_args()
    if args.query:
        from imas_codex.graph.client import GraphClient

        STATS.write_text(json.dumps(query_counts(GraphClient()), indent=1) + "\n")
    draw(json.loads(STATS.read_text()), SVG, args.draft)


if __name__ == "__main__":
    main()

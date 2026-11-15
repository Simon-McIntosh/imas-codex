#!/usr/bin/env python3
"""Draw where the held names ended, by the hold kind that classified them.

Left panel: names per hold kind, split by the outcome the pass reached. Right
panel: the USD the pass spent on each kind, with the accepted share marked.

Every number is read from the evidence JSON this fragment committed beside it,
so regenerating the figure re-derives its numbers and a drifting report shows up
as a drifting figure. Run from the repository root; the SVG lands beside this
script and the evidence page embeds it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from imas_codex.plotting.style import BASE_STYLE, save_figure  # noqa: E402

EVIDENCE = Path(
    "docs/evidence/fragments/sn-definitions-by-construction/held-names-refine-pass.json"
)
SVG = Path("docs/figures/sn-held-names-refine-pass/outcomes-by-hold-kind.svg")

ACCEPTED = "#2e8b57"
GRAMMAR = "#c9a227"
GROUPING = "#8f8f8f"
DECISION = "#b03a2e"
UNSETTLED = "#2a6f97"
SPEND = "#4c6f8f"

GROUPS = (
    ("accepted", ACCEPTED),
    ("waiting on grammar", GRAMMAR),
    ("held: grouping parent", GROUPING),
    ("needs a rebind or fold decision", DECISION),
    ("unsettled", UNSETTLED),
)


def group_of(outcome: str) -> str:
    """Fold the outcome strings into the five groups the figure draws."""
    if outcome == "accepted":
        return "accepted"
    if outcome.startswith("waiting"):
        return "waiting on grammar"
    if outcome.startswith("held"):
        return "held: grouping parent"
    if "rebind" in outcome:
        return "needs a rebind or fold decision"
    return "unsettled"


def load() -> tuple[list[tuple[str, dict[str, int], float, int]], int]:
    """Read the counts, the per-kind split and the spend from the evidence."""
    doc = json.loads(EVIDENCE.read_text())
    by_kind = doc["outcomes"]["by_hold_kind"]
    split: dict[str, dict[str, int]] = {kind: {} for kind in by_kind}
    for row in doc["names"]:
        kind = row.get("hold_category")
        if kind not in split:
            continue
        name = group_of(row["outcome"])
        split[kind][name] = split[kind].get(name, 0) + 1
    rows = []
    for kind, slot in by_kind.items():
        counts = split[kind]
        if sum(counts.values()) != slot["names"]:
            raise SystemExit(
                f"{kind}: {sum(counts.values())} names by outcome, "
                f"{slot['names']} in the by-kind table"
            )
        rows.append((kind, counts, slot["spend_usd"], slot["accepted"]))
    rows.sort(key=lambda r: (-sum(r[1].values()), r[0]))
    return rows, len(doc["names"])


def main() -> None:
    rows, population = load()
    plt.style.use(str(BASE_STYLE))
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(14, 8.4), dpi=100, width_ratios=[1.45, 1]
    )
    fig.subplots_adjust(left=0.235, right=0.975, top=0.855, bottom=0.105, wspace=0.32)
    fig.patch.set_facecolor("white")

    positions = list(range(len(rows)))
    height = 0.62
    for index, (_kind, counts, spend, accepted) in enumerate(rows):
        left = 0
        for name, colour in GROUPS:
            count = counts.get(name, 0)
            if not count:
                continue
            ax0.barh(
                index,
                count,
                left=left,
                height=height,
                color=colour)
            if count >= 3:
                ax0.text(
                    left + count / 2,
                    index,
                    str(count),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=13,
                )
            left += count
        ax0.text(left + 0.6, index, str(left), va="center", fontsize=13)
        ax1.barh(index, spend, height=height, color=SPEND)
        ax1.text(
            spend + max(r[2] for r in rows) * 0.015,
            index,
            f"USD {spend:.2f} · {accepted}/{left}",
            va="center",
            fontsize=12.5,
        )

    ax0.set_yticks(positions)
    ax0.set_yticklabels([kind for kind, *_ in rows])
    ax0.invert_yaxis()
    widest = max(sum(r[1].values()) for r in rows)
    ax0.set_xlim(0, widest * 1.14)
    ax0.set_xticks(range(0, widest + 1, 10))
    ax0.set_xlabel("Names")
    ax0.set_title("Where each hold kind ended")

    ax1.set_yticks(positions)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    dearest = max(r[2] for r in rows)
    ax1.set_xlim(0, dearest * 1.95)
    ax1.set_xticks(range(0, int(dearest) + 1, 2))
    ax1.set_xlabel("Spend (USD)")
    ax1.set_title("Spend on the kind")

    for axis in (ax0, ax1):
        axis.tick_params(axis="y", length=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour) for _, colour in GROUPS
    ]
    fig.legend(
        handles,
        ["accepted", "grammar wait", "grouping parent", "rebind or fold", "unsettled"],
        loc="upper left",
        bbox_to_anchor=(0.235, 0.94),
        ncol=5,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.4,
    )
    fig.text(
        0.02,
        0.972,
        f"{population} rows: the 136 held names plus the 15 successor "
        "identities their renames created",
        fontsize=15,
    )

    save_figure(fig, SVG)
    print(f"wrote {SVG}")


if __name__ == "__main__":
    main()

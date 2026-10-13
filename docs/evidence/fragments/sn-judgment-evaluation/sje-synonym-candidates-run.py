"""Count same-unit near-duplicate pairs across cosine thresholds and emit the widest pool under budget.

Reads the live accepted or approved standard names that carry a unit and an
embedding, runs the audit-backed same-unit comparison at each candidate cosine
threshold, and writes the pairs at the widest threshold whose pool stays at or
under the pair budget. No judgment is applied: the emitted pairs carry grammar
divergence and an empty prescreen slot, never a verdict or probability.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.near_duplicates import build_worklist

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "golden-synonym-candidates.json"
FRAGMENT = HERE / "sje-synonym-candidates.html"

THRESHOLDS = (0.95, 0.92, 0.90, 0.88)
PAIR_BUDGET = 800
PAGE = 100


def read_names(graph: GraphClient) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        page = graph.query(
            "MATCH (n:StandardName) "
            "WHERE n.name_stage IN ['accepted', 'approved'] "
            "AND n.embedding IS NOT NULL AND n.unit IS NOT NULL "
            "RETURN n.id AS id, n.unit AS unit, n.description AS description, "
            "n.embedding AS embedding ORDER BY n.id SKIP $offset LIMIT $limit",
            offset=offset,
            limit=PAGE,
        )
        rows.extend(page)
        if len(page) < PAGE:
            break
        offset += len(page)
    if not rows:
        raise RuntimeError(
            "Live accepted-name query returned no rows; inspect the graph"
        )
    print(f"live_accepted_embedded_names={len(rows)}", flush=True)
    return rows


def pair_count(pairs: list[dict]) -> int:
    return len(pairs)


def choose_threshold(counts: dict[float, int]) -> float:
    """Lowest (widest) threshold whose pool is within budget."""
    for threshold in sorted(counts, reverse=False):
        if counts[threshold] <= PAIR_BUDGET:
            return threshold
    raise RuntimeError(f"No threshold within budget {PAIR_BUDGET}: counts={counts}")


def divergent_kind(segment: str) -> str:
    return {
        "physical_base": "physical base",
        "subject": "subject",
        "component": "component",
        "operator": "operator",
        "locus": "locus",
        "process": "process",
        "spelling": "spelling",
    }.get(segment, segment)


def write_fragment(
    counts: dict[float, int],
    chosen: float,
    pairs: list[dict],
    live_names: int,
) -> None:
    kinds = Counter(
        divergent_kind(segment)
        for pair in pairs
        for segment in pair["divergent_segments"]
    )
    rows = "\n".join(
        f"      <tr><td>{html.escape(kind)}</td><td>{count}</td></tr>"
        for kind, count in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    count_rows = "\n".join(
        f"      <tr><td>{threshold:.2f}</td><td>{counts[threshold]}</td></tr>"
        for threshold in THRESHOLDS
    )
    overlaps = len({pair["overlap"] for pair in pairs})
    FRAGMENT.write_text(
        f"""<section id="sje-synonym-candidates">
  <h2 id="sje-synonym-candidates-title">Same-unit synonym candidate pool for golden D</h2>
  <p id="sje-synonym-candidates-scope">
    The live accepted and approved graph held {live_names:,} names with units and
    embeddings. The audit-backed same-unit comparison was counted at four cosine
    thresholds and the widest pool staying within {PAIR_BUDGET} pairs was written.
    The written pool is the {chosen:.2f} threshold: {len(pairs)} pairs in
    {overlaps} overlaps. No judgment is attached to any pair; the prescreen slot
    is empty for the coordinator.
  </p>

  <table id="sje-synonym-candidates-threshold-counts">
    <thead><tr><th>Cosine threshold</th><th>Same-unit pairs</th></tr></thead>
    <tbody>
{count_rows}
    </tbody>
  </table>

  <p id="sje-synonym-candidates-chosen">
    Widest threshold within budget: <strong>{chosen:.2f}</strong>.
  </p>

  <table id="sje-synonym-candidates-divergent-kinds">
    <thead><tr><th>Divergent grammar segment</th><th>Pairs</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>
"""
    )


def run() -> None:
    graph = GraphClient()
    names = read_names(graph)

    counts: dict[float, int] = {}
    pools: dict[float, list[dict]] = {}
    for threshold in THRESHOLDS:
        worklist = build_worklist(names, threshold=threshold)
        pairs = worklist["west"] + worklist["other"]
        counts[threshold] = pair_count(pairs)
        pools[threshold] = pairs
        print(f"threshold={threshold:.2f} pairs={counts[threshold]}", flush=True)

    if not all(counts.values()):
        raise RuntimeError(f"No pairs at some threshold: {counts}")

    chosen = choose_threshold(counts)
    selected = sorted(
        pools[chosen], key=lambda pair: (pair["overlap"], -pair["cosine"])
    )
    print(f"chosen_threshold={chosen:.2f} pairs={len(selected)}", flush=True)

    payload = {
        "threshold": chosen,
        "threshold_counts": {f"{t:.2f}": int(counts[t]) for t in THRESHOLDS},
        "live_names": len(names),
        "pair_budget": PAIR_BUDGET,
        "pairs": [
            {
                "left": pair["left"],
                "right": pair["right"],
                "unit": pair["unit"],
                "cosine": pair["cosine"],
                "description_left": pair["description_left"],
                "description_right": pair["description_right"],
                "divergent_segments": pair["divergent_segments"],
                "overlap": pair["overlap"],
                "prescreen": {},
            }
            for pair in selected
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n")
    write_fragment(counts, chosen, payload["pairs"], len(names))
    print(
        f"wrote {OUTPUT.name} bytes={OUTPUT.stat().st_size} "
        f"pairs={len(payload['pairs'])}",
        flush=True,
    )


if __name__ == "__main__":
    run()

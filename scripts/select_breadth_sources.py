"""Select a breadth-stratified, popularity-weighted set of DD source paths.

For a large-scale SN rotation that must TOUCH EVERY nameable IDS while
bounding depth per IDS — surfacing vocabulary gaps and prompt-robustness
issues across the full domain space without committing the whole queue.

Selection strategy:
- One stratum per IDS (breadth: every IDS with nameable sources is hit).
- Within each IDS, rank by a DD-intrinsic popularity proxy and take the
  top ``per_ids`` paths:
    * node_category == 'quantity' AND is_leaf  — only nameable physics leaves
      (geometry/coordinate/metadata/structural nodes do not yield names).
    * shallower ``depth`` first — fundamental quantities sit near the top of
      the IDS tree; deeply-nested ones are niche.
    * longer ``doc_length`` first — well-documented paths are the heavily
      used, canonical quantities.
  (The graph carries no facility-usage cross-references, so these
  DD-intrinsic signals are the available popularity proxy.)

Prints the space-separated path list on stdout (consumable as trailing
``sn run`` arguments) and a per-IDS coverage summary on stderr.

Usage:
    uv run python scripts/select_breadth_sources.py [--per-ids N] [--status STATUS]
"""

from __future__ import annotations

import argparse
import sys


def select_breadth_sources(per_ids: int = 8, status: str = "extracted") -> list[str]:
    """Return up to ``per_ids`` popularity-ranked source ids per IDS."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (s:StandardNameSource {status: $status})-[:FROM_DD_PATH]->(n:IMASNode)
            WHERE n.node_category = 'quantity' AND n.is_leaf = true
            WITH n.ids AS ids, s.source_id AS sid,
                 coalesce(n.doc_length, 0) AS doc, coalesce(n.depth, 99) AS depth
            ORDER BY ids, depth ASC, doc DESC
            WITH ids, collect(sid)[0..$per_ids] AS top
            RETURN ids, top ORDER BY ids
            """,
            status=status,
            per_ids=per_ids,
        )
    paths: list[str] = []
    for r in rows or []:
        top = r["top"]
        paths.extend(top)
        print(f"{r['ids']:38s} {len(top)}", file=sys.stderr)
    print(
        f"--- {len(paths)} sources across {len(rows or [])} IDSs ---",
        file=sys.stderr,
    )
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-ids", type=int, default=8)
    ap.add_argument("--status", default="extracted")
    args = ap.parse_args()
    paths = select_breadth_sources(per_ids=args.per_ids, status=args.status)
    print(" ".join(paths))


if __name__ == "__main__":
    main()

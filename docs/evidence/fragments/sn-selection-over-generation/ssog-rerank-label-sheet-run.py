"""Build the comparator label sheet: both orders per item, for coordinator grading.

For each drawn DD source this records the comparator block twice — the
embedding-ordered top ``TOP_K`` from :func:`_collect_nearby_name_comparators`
and the reranked top ``TOP_K`` from
:func:`imas_codex.standard_names.context.rerank_name_comparators` — and emits
one JSON row per item carrying the union of the two top-list views with an
empty ``grade`` field the coordinator fills. The driver only reads: the
embedding search reads the graph, the rerank spends on the judge seat, and
nothing is persisted.

Run::

    uv run python docs/evidence/fragments/sn-selection-over-generation/ssog-rerank-label-sheet-run.py --out <sheet.json>
    uv run python ...ssog-rerank-label-sheet-run.py --pilot 2   # spend check only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

SEED = 20261013
N_ITEMS = 30
POOL_K = 20
TOP_K = 10
BOUNDARY_SUFFIX = "_density_at_plasma_boundary"
BOUNDARY_RESERVE = 10
BOUNDARY_FAMILY = "boundary_density"


# ---------------------------------------------------------------------------
# Item draw
# ---------------------------------------------------------------------------


def load_population(gc: Any) -> list[dict[str, Any]]:
    """Every DD source with a description and a domain, in path order.

    The row set is the StandardNameSource label carrying a non-empty
    description and a physics domain; the ordered path is the stable key the
    draw samples against.
    """
    rows = gc.query(
        """
        MATCH (s:StandardNameSource)
        WHERE s.source_id IS NOT NULL
          AND s.source_id CONTAINS '/'
          AND s.description IS NOT NULL AND s.description <> ''
          AND s.physics_domain IS NOT NULL
        RETURN DISTINCT s.source_id AS path,
               s.description AS description,
               s.physics_domain AS domain
        ORDER BY path
        """
    )
    return [dict(r) for r in rows]


def load_boundary_paths(gc: Any) -> list[str]:
    """DD sources that produced a boundary-density standard name."""
    rows = gc.query(
        """
        MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
        WHERE n.id ENDS WITH $suffix
        RETURN DISTINCT s.source_id AS path
        ORDER BY path
        """,
        suffix=BOUNDARY_SUFFIX,
    )
    return [r["path"] for r in rows]


def draw_items(
    rows: Sequence[dict[str, Any]],
    boundary_paths: Sequence[str],
    *,
    seed: int = SEED,
    count: int = N_ITEMS,
    boundary_reserve: int = BOUNDARY_RESERVE,
) -> list[dict[str, Any]]:
    """Draw *count* DD sources reproducibly, seeded, spread across domains.

    The boundary-density family's sources are reserved first so the family is
    always represented; the remaining slots fill round-robin across the domains
    in sorted order, each domain shuffled by the seeded RNG. The result is
    deterministic for a fixed population and seed.
    """
    by_path = {r["path"]: r for r in rows}

    chosen: list[dict[str, Any]] = []
    chosen_paths: set[str] = set()
    for path in boundary_paths[:boundary_reserve]:
        row = by_path.get(path)
        if row is not None and path not in chosen_paths:
            chosen.append({**row, "family": BOUNDARY_FAMILY})
            chosen_paths.add(path)

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["path"] in chosen_paths:
            continue
        by_domain.setdefault(row["domain"], []).append(row)

    rng = random.Random(seed)
    domains = sorted(by_domain)
    for domain in domains:
        rng.shuffle(by_domain[domain])

    pointers = {domain: 0 for domain in domains}
    while len(chosen) < count:
        progressed = False
        for domain in domains:
            if len(chosen) >= count:
                break
            pool = by_domain[domain]
            if pointers[domain] >= len(pool):
                continue
            row = pool[pointers[domain]]
            pointers[domain] += 1
            progressed = True
            if row["path"] not in chosen_paths:
                chosen.append({**row, "family": None})
                chosen_paths.add(row["path"])
        if not progressed:
            break
    return chosen


# ---------------------------------------------------------------------------
# Row / sheet
# ---------------------------------------------------------------------------


def _rank_map(order: Sequence[dict[str, Any]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for index, candidate in enumerate(order):
        name = candidate.get("id", "")
        if name and name not in ranks:
            ranks[name] = index + 1
    return ranks


async def build_row(
    item: dict[str, Any],
    *,
    search: Callable[..., list[dict]] | None = None,
    rerank: Callable[..., Any] | None = None,
    pool_k: int = POOL_K,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Record one item's comparator block in both orders.

    The embedding order comes from the shared nearby-name block; the rerank
    reorders that same pool. Both orders are recorded by rank, and the candidate
    list is the union of the two top-*top_k* views with an empty ``grade``.
    """
    from imas_codex.standard_names.workers import _collect_nearby_name_comparators

    if rerank is None:
        from imas_codex.standard_names.context import rerank_name_comparators

        rerank = rerank_name_comparators

    pool = _collect_nearby_name_comparators(
        [item], per_item_k=pool_k, cap=pool_k, search=search
    )
    if not pool:
        return {
            "path": item.get("path"),
            "domain": item.get("domain"),
            "description": item.get("description"),
            "family": item.get("family"),
            "rerank_cost_usd": 0.0,
            "seat": "",
            "candidates": [],
        }

    ordered, cost, seat = await rerank([item], pool)

    embedding_top = pool[:top_k]
    rerank_top = list(ordered)[:top_k]
    embedding_ranks = _rank_map(pool)
    rerank_ranks = _rank_map(ordered)

    union: dict[str, dict[str, Any]] = {}
    for candidate in [*embedding_top, *rerank_top]:
        name = candidate.get("id", "")
        if name and name not in union:
            union[name] = candidate

    candidates = [
        {
            "name": name,
            "description": candidate.get("description", ""),
            "embedding_rank": embedding_ranks.get(name),
            "rerank_rank": rerank_ranks.get(name),
            "grade": None,
        }
        for name, candidate in union.items()
    ]
    return {
        "path": item.get("path"),
        "domain": item.get("domain"),
        "description": item.get("description"),
        "family": item.get("family"),
        "rerank_cost_usd": float(cost),
        "seat": seat,
        "candidates": candidates,
    }


async def build_sheet(
    items: Sequence[dict[str, Any]],
    *,
    search: Callable[..., list[dict]] | None = None,
    rerank: Callable[..., Any] | None = None,
    seed: int = SEED,
    pool_k: int = POOL_K,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Build the whole sheet, one item at a time, accumulating the spend."""
    rows = []
    total_cost = 0.0
    for item in items:
        row = await build_row(
            item, search=search, rerank=rerank, pool_k=pool_k, top_k=top_k
        )
        total_cost += row["rerank_cost_usd"]
        rows.append(row)
    return {
        "seed": seed,
        "n_items": len(rows),
        "pool_k": pool_k,
        "top_k": top_k,
        "rerank_cost_usd_total": round(total_cost, 6),
        "items": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run(sheet: dict[str, Any], out: Path) -> None:
    out.write_text(json.dumps(sheet, indent=2, sort_keys=False) + "\n")
    print(
        f"wrote {out} :: items={sheet['n_items']} "
        f"candidates={sum(len(r['candidates']) for r in sheet['items'])} "
        f"spend_usd={sheet['rerank_cost_usd_total']}"
    )


async def _amain(args: argparse.Namespace) -> int:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        population = load_population(gc)
        boundary = load_boundary_paths(gc)
    print(f"population={len(population)} boundary_family={len(boundary)}")

    items = draw_items(population, boundary, seed=args.seed, count=args.items)
    if args.pilot:
        items = items[: args.pilot]

    sheet = await build_sheet(items, seed=args.seed)
    per_item = sheet["rerank_cost_usd_total"] / max(1, sheet["n_items"])
    projected = per_item * args.items
    print(
        f"pilot={sheet['n_items']} spend_usd={sheet['rerank_cost_usd_total']:.6f} "
        f"per_item={per_item:.6f} projected_{args.items}={projected:.4f}"
    )
    if args.pilot:
        if projected > args.ceiling:
            print(f"REFUSED: projected {projected:.4f} > ceiling {args.ceiling}")
            return 3
        return 0

    if sheet["rerank_cost_usd_total"] > args.ceiling:
        print(f"REFUSED: spend exceeded ceiling {args.ceiling}")
        return 3
    _run(sheet, Path(args.out))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ssog-rerank-label-sheet.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--items", type=int, default=N_ITEMS)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--ceiling", type=float, default=0.50)
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
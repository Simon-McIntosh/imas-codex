"""Build the link-label sheet the coordinator labels with Jev's verdicts.

Driver for the ``ssog-link-label-sheet`` node. It reads the accepted names'
documentation from the
live graph, finds every documentation link whose target names a standard name
(:func:`imas_codex.standard_names.doc_links.find_name_links`), and marks the
self-links — links that point back at the documenting name — with
:func:`find_self_links`. It draws a sample of at least 150 links that includes
every self-link and spreads the rest across the documenting names' physics
domains, then asks Jev one Noul per link through
:func:`imas_codex.discovery.base.judgment.decide_batch` under the ``sn-judge``
seat and the ``standard-names`` service: does the link text describe the
target's description? The state carries the link text, the target's name and
description, and the documenting name.

The sheet it writes is one row per sampled link: the documenting name, the
link text, the target's name and description, whether the link is a self-link,
Jev's probability, and an empty ``label`` field for the coordinator. Nothing
here writes graph state.

A pilot of ten calls runs first; the run stops if its measured per-call cost
extrapolates above the USD 1.00 the plan allows.

Usage::

    python ssog-link-label-sheet-run.py --out ssog-link-label-sheet.json
    python ssog-link-label-sheet-run.py --out sheet.json --pilot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from imas_codex.standard_names.doc_links import (
    NameLink,
    find_name_links,
    link_judgement_state,
    link_label_question,
)

SEAT = "sn-judge"
SERVICE = "standard-names"
MIN_ROWS = 150
PILOT_SIZE = 10
COST_CEILING_USD = 1.00


def _load_names() -> list[dict[str, Any]]:
    """Every standard name whose documentation carries a ``name:`` link, bounded."""

    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        return gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.documentation CONTAINS 'name:'
            RETURN sn.id AS id, sn.description AS description,
                   sn.documentation AS documentation, sn.physics_domain AS domain
            """
        )


def _descriptions() -> dict[str, str]:
    """The description of every standard name, so a link target's own may be shown."""

    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (sn:StandardName) RETURN sn.id AS id, sn.description AS description"
        )
    return {row["id"]: row["description"] or "" for row in rows}


def _candidate_links(
    names: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Every documentation link in the cohort, tagged with its documenting name."""

    candidates: list[dict[str, Any]] = []
    for row in names:
        documentation = row["documentation"] or ""
        for link in find_name_links(documentation):
            candidates.append(
                {
                    "documenting_name": row["id"],
                    "domain": row["domain"] or "unclassified",
                    "link": link,
                }
            )
    return candidates


def _sample(
    candidates: list[dict[str, Any]],
    *,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    """All self-links plus a domain-spread round-robin sample up to *budget*.

    Self-links are the cases the judgment catches nothing of — the deterministic
    half of link-label analysis — so the plan requires every one of them. The
    rest round-robin across the documenting names' domains, which also spreads
    the sample across Jev's verdict range because each domain carries its own
    mix of correct and wrong-neighbour links.
    """

    self_links = [
        candidate
        for candidate in candidates
        if candidate["link"].target == candidate["documenting_name"]
    ]
    chosen = list(self_links)
    if len(chosen) >= budget:
        return chosen[:budget]

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["link"].target != candidate["documenting_name"]:
            by_domain[candidate["domain"]].append(candidate)

    rng = random.Random(seed)
    for group in by_domain.values():
        rng.shuffle(group)
    order = sorted(by_domain)
    index = 0
    while len(chosen) < budget and any(by_domain[domain] for domain in order):
        domain = order[index % len(order)]
        index += 1
        if by_domain[domain]:
            chosen.append(by_domain[domain].pop())
    return chosen


def _probability(answer: dict[str, Any] | None) -> float | None:
    """Jev's probability that the link text describes the target, or ``None``."""

    if answer is None:
        return None
    decision = answer.get("label_describes_target") or {}
    return decision.get("noul")


async def _judge(
    candidates: list[dict[str, Any]],
    descriptions: dict[str, str],
    *,
    concurrency: int,
) -> tuple[list[float | None], float, float]:
    """Ask Jev one link-label Noul per candidate; return probabilities and cost."""

    from imas_codex.discovery.base.judgment import decide_batch
    from imas_codex.settings import get_model

    states = [
        link_judgement_state(
            candidate["link"],
            documenting_name=candidate["documenting_name"],
            target_description=descriptions.get(candidate["link"].target, ""),
        )
        for candidate in candidates
    ]
    start = time.monotonic()
    results, cost = await decide_batch(
        states,
        link_label_question(),
        model=get_model(SEAT),
        service=SERVICE,
        concurrency=concurrency,
    )
    elapsed = time.monotonic() - start
    return [_probability(answer) for answer, _ in results], cost, elapsed


def _rows(
    candidates: list[dict[str, Any]],
    probabilities: list[float | None],
    descriptions: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, probability in zip(candidates, probabilities, strict=True):
        link: NameLink = candidate["link"]
        rows.append(
            {
                "documenting_name": candidate["documenting_name"],
                "link_text": link.label,
                "target_name": link.target,
                "target_description": descriptions.get(link.target, ""),
                "is_self_link": link.target == candidate["documenting_name"],
                "jev_probability": probability,
                "label": None,
            }
        )
    return rows


def _report(
    names: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    sampled: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    self_total = sum(1 for c in candidates if c["link"].target == c["documenting_name"])
    self_sampled = sum(1 for r in rows if r["is_self_link"])
    domains = len({c["domain"] for c in sampled})
    print(f"names with name: links: {len(names)}")
    print(f"name links found: {len(candidates)}  self-links: {self_total}")
    print(
        f"sampled: {len(rows)}  self-links in sample: {self_sampled}  domains: {domains}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="sheet JSON path")
    parser.add_argument("--pilot", action="store_true", help="judge ten links and stop")
    parser.add_argument("--limit", type=int, default=None, help="cap the sample size")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    names = _load_names()
    candidates = _candidate_links(names)

    budget = args.limit or max(MIN_ROWS, 0)
    if args.pilot:
        budget = PILOT_SIZE
    else:
        budget = max(budget, MIN_ROWS) + 30  # headroom so every self-link rides
    sampled = _sample(candidates, budget=budget, seed=args.seed)

    descriptions = _descriptions()

    if args.pilot:
        probabilities, cost, elapsed = asyncio.run(
            _judge(sampled, descriptions, concurrency=args.concurrency)
        )
        per_call = cost / max(1, len(sampled))
        projected = per_call * (len(sampled) * 20)
        for candidate, probability in zip(sampled, probabilities, strict=True):
            print(
                f"  {candidate['link'].target} <- {candidate['documenting_name']}"
                f"  p={probability}"
            )
        print(
            f"pilot: {len(sampled)} calls  ${cost:.4f}  "
            f"per-call ${per_call:.5f}  {elapsed:.1f}s"
        )
        print(
            f"projected for 200 calls: ${projected:.4f}  ceiling ${COST_CEILING_USD:.2f}"
        )
        if projected > COST_CEILING_USD:
            raise SystemExit(f"pilot projects ${projected:.4f} > ceiling")
        return

    probabilities, cost, elapsed = asyncio.run(
        _judge(sampled, descriptions, concurrency=args.concurrency)
    )
    rows = _rows(sampled, probabilities, descriptions)
    _report(names, candidates, sampled, rows)
    scored = sum(1 for r in rows if r["jev_probability"] is not None)
    print(f"judged {scored}/{len(rows)} links  spend ${cost:.4f}  {elapsed:.1f}s")

    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

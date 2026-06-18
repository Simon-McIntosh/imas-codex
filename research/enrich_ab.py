"""Focused A/B harness for DD-enrichment prompt iteration.

Re-runs the ENRICH (Pass 1) prompt on a chosen set of DD paths against the live
model + context WITHOUT writing the graph, and prints raw doc -> current
description -> candidate description side by side. Lets us iterate the prompt on
known-hard / currently-incorrect paths before re-enriching the whole DD.

Usage:
    uv run python research/enrich_ab.py <dd_path> [<dd_path> ...]
    uv run python research/enrich_ab.py --suspect   # a curated hard set
"""

from __future__ import annotations

import sys

from imas_codex.discovery.base.llm import call_llm_structured
from imas_codex.graph.client import GraphClient
from imas_codex.graph.dd_enrichment import (
    IMASPathEnrichmentBatch,
    build_enrichment_messages,
    gather_path_context,
)
from imas_codex.settings import get_model

# Known-hard / previously-fabricated paths (terse raw doc -> over-stated desc).
SUSPECT = [
    "equilibrium/time_slice/constraints/diamagnetic_flux",
    "equilibrium/time_slice/global_quantities/psi_external_average",
    "equilibrium/time_slice/global_quantities/v_external",
    "equilibrium/time_slice/constraints/faraday_angle",
    "equilibrium/time_slice/constraints/mse_polarization_angle",
    "equilibrium/time_slice/constraints/b_field_pol_probe",
    "equilibrium/time_slice/constraints/flux_loop",
    "equilibrium/time_slice/constraints/ip",
]


def main(argv: list[str]) -> None:
    paths = SUSPECT if (not argv or argv[0] == "--suspect") else argv
    model = get_model("dd-enrichment")
    with GraphClient() as gc:
        current = {
            r["id"]: r["d"]
            for r in gc.query(
                "MATCH (n:IMASNode) WHERE n.id IN $p RETURN n.id AS id, n.description AS d",
                p=paths,
            )
        }
        raw = {
            r["id"]: r["doc"]
            for r in gc.query(
                "MATCH (n:IMASNode) WHERE n.id IN $p RETURN n.id AS id, n.documentation AS doc",
                p=paths,
            )
        }
        ctxs = gather_path_context(gc, [{"id": p} for p in paths], {})
        msgs = build_enrichment_messages(ctxs, {})
        result, cost, _ = call_llm_structured(
            model=model,
            messages=msgs,
            response_model=IMASPathEnrichmentBatch,
            service="data-dictionary",
        )
    by_idx = {r.path_index: r.description for r in result.results}
    print(f"model={model}  cost=${cost:.4f}  paths={len(paths)}\n")
    for i, ctx in enumerate(ctxs, 1):
        pid = ctx["id"]
        print(f"### {pid}")
        print(f"  RAW doc : {(raw.get(pid) or '(none)')[:160]}")
        print(f"  CURRENT : {(current.get(pid) or '(none)')[:220]}")
        print(f"  NEW     : {by_idx.get(i, '(missing)')[:220]}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])

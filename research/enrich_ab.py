"""Focused A/B harness for DD-enrichment prompt iteration.

Re-runs the ENRICH (Pass 1) prompt against the live model + context WITHOUT
writing the graph, and compares raw doc -> current description -> candidate
description. Lets us iterate the prompt on known-hard / terse-doc paths before
re-enriching the whole DD.

Modes:
    uv run python research/enrich_ab.py --suspect          # curated hard set
    uv run python research/enrich_ab.py <dd_path> ...       # explicit paths
    uv run python research/enrich_ab.py --sweep [--per-ids 6] [--max 400]
        # cross-IDS sweep of terse-doc quantity leaves (where fabrication
        # concentrates), batched, with an auto-flag for NEW descriptions that
        # assert a specific absent from the source material.

The auto-flag is a triage heuristic, not ground truth: it flags a candidate
when NEW contains a direction / mechanism / location / weighting phrase that is
absent from the path's raw documentation, path string, AND ancestor docs — i.e.
a specific with no visible source. Review flagged ones by hand.
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

# Specific claims whose presence in NEW but absence from source = likely fabrication.
SPECIFIC_TOKENS: dict[str, list[str]] = {
    "direction": [
        "poloidal",
        "toroidal",
        "radial",
        "parallel",
        "perpendicular",
        "vertical",
        "azimuthal",
        "diamagnetic",
    ],
    "mechanism": [
        "due to",
        "arising from",
        "driven by",
        "caused by",
        "resulting from",
        "produced by",
    ],
    "location": [
        "at the plasma boundary",
        "at the magnetic axis",
        "at the wall",
        "at the separatrix",
        "at the edge",
        "at the last closed",
    ],
    "weighting": [
        "weighted by",
        "current-weighted",
        "volume-averaged",
        "area-averaged",
        "flux-surface averaged",
        "flux surface averaged",
        "line-integrated",
        "line integrated",
    ],
}


# A specific is "implied" (correctly sourced) when the path/IDS carries a token
# that conventionally entails it — tf=toroidal field, /z=vertical, profiles_1d=
# radial profile, bessel_N=Bessel-weighted, etc. Suppress these so the flag
# isolates genuinely unsourced specifics.
PATH_IMPLIES: dict[str, list[str]] = {
    "poloidal": ["pf", "psi", "pol", "poloidal", "b_field_pol", "bpol"],
    "toroidal": ["tf", "phi", "tor", "toroidal", "b_field_tor", "n_tor", "n_phi"],
    "radial": ["profiles_1d", "radius", "radial", "rho_tor", "/r", "_r"],
    "vertical": ["/z", "_z", "height", "magnetic_moment_z"],
    "parallel": ["parallel", "_par", "j_par", "pressure_parallel"],
    "perpendicular": ["perp", "radial", "velocity/radial"],
    "diamagnetic": ["diamag"],
    "volume-averaged": ["volume_average", "vol_avg"],
    "line-integrated": ["interferometer", "polarimeter", "refractometer", "line"],
    "weighted by": ["bessel", "weight"],
    "flux-surface averaged": ["flux_surface", "fsa"],
}


def flag_unsourced(new_desc: str, source_text: str, path: str) -> list[str]:
    new_l, src_l, path_l = new_desc.lower(), source_text.lower(), path.lower()
    hits = []
    for cat, toks in SPECIFIC_TOKENS.items():
        for t in toks:
            if t not in new_l or t in src_l:
                continue
            # Suppress when the path/IDS conventionally implies the specific.
            if any(marker in path_l for marker in PATH_IMPLIES.get(t, [])):
                continue
            hits.append(f"{cat}:{t}")
    return hits


def _sample_paths(gc: GraphClient, per_ids: int, cap: int) -> list[str]:
    rows = gc.query(
        """
        MATCH (n:IMASNode)
        WHERE n.node_category = 'quantity'
          AND n.description IS NOT NULL
          AND n.documentation IS NOT NULL
          AND size(n.documentation) < 80
        WITH split(n.id, '/')[0] AS ids, n
        ORDER BY rand()
        WITH ids, collect(n.id)[0..$per_ids] AS sample
        UNWIND sample AS pid
        RETURN pid
        LIMIT $cap
        """,
        per_ids=per_ids,
        cap=cap,
    )
    return [r["pid"] for r in rows]


def main(argv: list[str]) -> None:
    model = get_model("dd-enrichment")
    per_ids, cap = 6, 400
    if argv and argv[0] == "--sweep":
        rest = argv[1:]
        if "--per-ids" in rest:
            per_ids = int(rest[rest.index("--per-ids") + 1])
        if "--max" in rest:
            cap = int(rest[rest.index("--max") + 1])
        paths = None  # resolved from graph below
    elif not argv or argv[0] == "--suspect":
        paths = SUSPECT
    else:
        paths = argv

    with GraphClient() as gc:
        if paths is None:
            paths = _sample_paths(gc, per_ids, cap)
        raw = {
            r["id"]: (r["doc"] or "")
            for r in gc.query(
                "MATCH (n:IMASNode) WHERE n.id IN $p RETURN n.id AS id, n.documentation AS doc",
                p=paths,
            )
        }
        current = {
            r["id"]: (r["d"] or "")
            for r in gc.query(
                "MATCH (n:IMASNode) WHERE n.id IN $p RETURN n.id AS id, n.description AS d",
                p=paths,
            )
        }
        # Batch through enrichment exactly as production does.
        results: dict[str, str] = {}
        src_text: dict[str, str] = {}
        total_cost = 0.0
        BATCH = 40
        for i in range(0, len(paths), BATCH):
            chunk = paths[i : i + BATCH]
            ctxs = gather_path_context(gc, [{"id": p} for p in chunk], {})
            for ctx in ctxs:
                anc = " ".join(
                    (a.get("documentation") or "") for a in ctx.get("ancestors", [])
                )
                src_text[ctx["id"]] = f"{raw.get(ctx['id'], '')} {ctx['id']} {anc}"
            msgs = build_enrichment_messages(ctxs, {})
            res, cost, _ = call_llm_structured(
                model=model,
                messages=msgs,
                response_model=IMASPathEnrichmentBatch,
                service="data-dictionary",
            )
            total_cost += cost
            for j, ctx in enumerate(ctxs, 1):
                desc = next(
                    (r.description for r in res.results if r.path_index == j), ""
                )
                results[ctx["id"]] = desc

    # Report
    flagged = []
    changed = 0
    for pid in paths:
        new = results.get(pid, "")
        if not new:
            continue
        if new.strip() != current.get(pid, "").strip():
            changed += 1
        hits = flag_unsourced(new, src_text.get(pid, ""), pid)
        if hits:
            flagged.append((pid, hits, new))

    print(f"\n{'=' * 70}")
    print(
        f"model={model}  paths={len(paths)}  changed={changed}  "
        f"flagged={len(flagged)}  cost=${total_cost:.4f}"
    )
    print(f"{'=' * 70}\n")
    print("### FLAGGED (NEW asserts a specific absent from raw doc + path + ancestors)")
    for pid, hits, new in flagged:
        print(f"\n* {pid}")
        print(f"    raw : {(raw.get(pid) or '(none)')[:120]}")
        print(f"    new : {new[:200]}")
        print(f"    flags: {', '.join(hits)}")
    if not flagged:
        print("  (none — no unsourced specifics detected in the sample)")


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
"""Runner for the stranded-stub composition node.

Two subcommands, both writing one JSON document to ``--out``:

``census``
    Read-only. Reports, for the nine roster DD paths: each source's claim
    state, whether it has a producing name, and the pending stub that holds the
    path. It also splits the sourceless population exactly as ``export.py``
    does, because those sets are not interchangeable: only a name in
    ``_SHIPPING_NAME_STAGES`` (``accepted``, ``approved``) blocks the export
    source gate, a draft-stage name is reported as ``sourceless_draft_names``
    and does not block, and an ``exhausted`` name is dormant. Both constants
    and the writer invariant are imported from ``export`` rather than restated
    here, so this census cannot drift from the gate it describes.

``compose``
    Invokes the sanctioned pipeline, ``sn run``, scoped by ``--focus`` to the
    nine roster paths, capped by ``--cost-limit``, and captures its whole
    stdout/stderr to ``--log``. ``--reset-to extracted`` is added when
    ``--reset`` is given, which clears the pending stub in the same exact-path
    allowlist so composition has an identity to bind.

The roster is fixed in this file rather than passed in so a later reader can
re-run the census and get the same row set.

The census is the evidence: a blocking set that shrinks tells nothing unless
the same read can see a name known to be sourceless, so ``census`` also reports
the gate population it scanned and refuses to report an empty roster as an
absence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROSTER_NAMES = [
    "line_averaged_neon_density",
    "radial_coordinate_of_reflector",
    "vertical_coordinate_of_reflector",
    "radius_of_soft_xray_detector",
    "ratio_of_diamagnetic_vorticity_to_major_radius",
    "toroidal_coordinate_of_spectrometer",
    "toroidal_tritium_velocity",
]

ROSTER_PATHS = [
    "summary/line_average/n_i/neon/value",
    "spectrometer_x_ray_crystal/channel/reflector/sphere_centre/r",
    "spectrometer_x_ray_crystal/channel/reflector/sphere_centre/z",
    "soft_x_rays/channel/detector/radius",
    "edge_profiles/ggd/vorticity_over_r/diamagnetic",
    "plasma_profiles/ggd/vorticity_over_r/diamagnetic",
    "spectrometer_visible/channel/geometry_matrix/interpolated/phi",
    "summary/local/pedestal/velocity_phi/tritium/value",
    "summary/local/separatrix/velocity_phi/tritium/value",
]


def census() -> dict:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.export import (
        _RELEASE_IDENTITY_HOLDS,
        _SHIPPING_NAME_STAGES,
    )
    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )
    from imas_codex.standard_names.source_invariant import find_sourceless_live_names

    with GraphClient() as gc:
        sources = [
            dict(row)
            for row in gc.query(
                """
                MATCH (s:StandardNameSource)
                WHERE s.dd_path IN $paths
                OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(p:StandardName)
                OPTIONAL MATCH (n:IMASNode {id: s.dd_path})-[:HAS_STANDARD_NAME]
                    ->(stub:StandardName)
                RETURN s.dd_path AS path, s.status AS status,
                       coalesce(s.attempt_count, 0) AS attempt_count,
                       s.claim_seq AS claim_seq,
                       s.parked_disposition AS parked,
                       s.last_error AS last_error,
                       collect(DISTINCT p.id) AS produced,
                       collect(DISTINCT stub.id) AS projected_stubs
                ORDER BY s.dd_path
                """,
                paths=ROSTER_PATHS,
            )
        ]
        stubs = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.id IN $ids
                OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                OPTIONAL MATCH (sn)-[:HAS_INTERNAL_CHANGE]->(
                    change:StandardNameChange {operation: 'adopt_pending_projection'})
                RETURN sn.id AS id, sn.name_stage AS name_stage,
                       sn.status AS status,
                       sn.description AS description, sn.unit AS unit,
                       size([x IN collect(DISTINCT src) WHERE x IS NOT NULL]) AS producers,
                       size([x IN collect(DISTINCT change) WHERE x IS NOT NULL])
                           AS adopt_receipts
                ORDER BY sn.id
                """,
                ids=ROSTER_NAMES,
            )
        ]
        audit = classify_sourceless_standard_names(gc)
        sourceless = [
            row
            for row in audit
            if row["population"] in {"dropped_edge", "never_bound"}
        ]
        # Reproduce the export source gate as it is written, not as it reads:
        # only a shipping-stage name blocks (export.py's source_losses), a
        # non-shipping name is reported and does not, and an exhausted name is
        # dormant. The writer invariant is the authority on which of these are
        # genuinely sourceless, and export intersects the two reads, so this
        # does too rather than trusting the classifier alone.
        confirmed = set(
            find_sourceless_live_names(gc, [row["id"] for row in sourceless])
        )

    dormant = [r for r in sourceless if r["name_stage"] == "exhausted"]
    draft = [
        r
        for r in sourceless
        if r["name_stage"] != "exhausted"
        and r["name_stage"] not in _SHIPPING_NAME_STAGES
    ]
    blocking = [
        r
        for r in sourceless
        if r["name_stage"] in _SHIPPING_NAME_STAGES
        and r["id"] not in _RELEASE_IDENTITY_HOLDS
        and r["id"] in confirmed
    ]
    blocking_ids = sorted(row["id"] for row in blocking)
    draft_ids = sorted(row["id"] for row in draft)
    return {
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shipping_name_stages": sorted(_SHIPPING_NAME_STAGES),
        "roster_paths": ROSTER_PATHS,
        "roster_names": ROSTER_NAMES,
        "sources": sources,
        "stubs": stubs,
        "roster_sources_present": len(sources),
        "roster_stubs_present": len(stubs),
        "gate_blocking_count": len(blocking_ids),
        "gate_blocking_ids": blocking_ids,
        "gate_roster_blocking": sorted(set(blocking_ids) & set(ROSTER_NAMES)),
        "draft_sourceless_count": len(draft_ids),
        "draft_sourceless_ids": draft_ids,
        "draft_roster_sourceless": sorted(set(draft_ids) & set(ROSTER_NAMES)),
        "dormant_sourceless_count": len(dormant),
        "sourceless_population": len(sourceless),
        "confirmed_sourceless": len(confirmed),
        "gate_audit_population": len(audit),
        "roster_stubs_with_producer": sorted(
            row["id"] for row in stubs if row["producers"] > 0
        ),
        "roster_stubs_adopted": sorted(
            row["id"] for row in stubs if row["adopt_receipts"] > 0
        ),
    }


def compose(reset: bool, cost_limit: float, minutes: float, log_path: Path) -> int:
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "imas-codex",
        "sn",
        "run",
        "--focus",
        " ".join(ROSTER_PATHS),
        "--names-only",
        "--cost-limit",
        f"{cost_limit:.2f}",
        "--time",
        f"{minutes:g}",
    ]
    if reset:
        cmd += ["--reset-to", "extracted"]
    env = dict(os.environ)
    header = (
        f"# command: {' '.join(cmd)}\n"
        f"# cwd: {Path.cwd()}\n"
        f"# started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )
    with log_path.open("w") as log:
        log.write(header)
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(
            f"\n# finished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        )
        log.write(f"EXIT={proc.returncode}\n")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_census = sub.add_parser("census")
    p_census.add_argument("--out", required=True)

    p_compose = sub.add_parser("compose")
    p_compose.add_argument("--log", required=True)
    p_compose.add_argument("--cost-limit", type=float, default=2.00)
    p_compose.add_argument("--minutes", type=float, default=12)
    p_compose.add_argument("--reset", action="store_true")

    args = parser.parse_args()
    if args.command == "census":
        payload = census()
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(
            f"roster_sources={payload['roster_sources_present']} "
            f"roster_stubs={payload['roster_stubs_present']} "
            f"gate_blocking={payload['gate_blocking_count']} "
            f"gate_roster_blocking={len(payload['gate_roster_blocking'])} "
            f"draft_sourceless={payload['draft_sourceless_count']} "
            f"draft_roster={len(payload['draft_roster_sourceless'])} "
            f"stubs_producing={len(payload['roster_stubs_with_producer'])} "
            f"stubs_adopted={len(payload['roster_stubs_adopted'])}"
        )
        return 0
    return compose(
        reset=args.reset,
        cost_limit=args.cost_limit,
        minutes=args.minutes,
        log_path=Path(args.log),
    )


if __name__ == "__main__":
    sys.exit(main())

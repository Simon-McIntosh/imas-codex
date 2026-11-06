#!/usr/bin/env python3
"""Runner for the stranded-stub composition node.

Three subcommands, each writing one JSON document to ``--out``:

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

``replay``
    Read-only, no database. Takes a ``census`` document recorded earlier and
    reclassifies its stubs with the same imported constants, so a "before"
    number is produced by the same code path as the "now" number instead of
    being asserted in prose. A stub with no producer is sourceless by
    construction, which is what lets a snapshot stand in for the writer
    invariant without a live database.

``outofroster``
    Read-only. The scope control: a whole-graph comparison of source
    composition, producing edges and name changes, with the roster excluded,
    between a stated cut-off and now. Source ``attempt_count`` has no audit
    trail, so that field is reported as an after-only total and the limitation
    is stated rather than papered over.

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
            row for row in audit if row["population"] in {"dropped_edge", "never_bound"}
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
        "dormant_roster": sorted(
            row["id"] for row in dormant if row["id"] in set(ROSTER_NAMES)
        ),
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


def replay(path: Path) -> dict:
    from imas_codex.standard_names.export import (
        _RELEASE_IDENTITY_HOLDS,
        _SHIPPING_NAME_STAGES,
    )

    recorded = json.loads(path.read_text())
    stubs = recorded["stubs"]
    blocking = [
        row
        for row in stubs
        if row["name_stage"] in _SHIPPING_NAME_STAGES
        and row["id"] not in _RELEASE_IDENTITY_HOLDS
        and row["producers"] == 0
    ]
    draft = [
        row
        for row in stubs
        if row["name_stage"] != "exhausted"
        and row["name_stage"] not in _SHIPPING_NAME_STAGES
        and row["producers"] == 0
    ]
    dormant = [row for row in stubs if row["name_stage"] == "exhausted"]
    return {
        "replayed_from": str(path),
        "recorded_at": recorded.get("observed_at"),
        "shipping_name_stages": sorted(_SHIPPING_NAME_STAGES),
        "replayed_stages": sorted({row["name_stage"] for row in stubs}),
        "roster_stubs_present": len(stubs),
        "gate_roster_blocking": sorted(row["id"] for row in blocking),
        "draft_roster_sourceless": sorted(row["id"] for row in draft),
        "dormant_roster": sorted(row["id"] for row in dormant),
        "composed_roster": sorted(row["id"] for row in stubs if row["producers"] > 0),
    }


def outofroster(since: str) -> dict:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        sources = list(
            gc.query(
                """
                MATCH (s:StandardNameSource)
                WHERE s.dd_path IS NULL OR NOT s.dd_path IN $paths
                OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(p:StandardName)
                WITH s, count(DISTINCT p) AS produced
                RETURN count(s) AS sources,
                       sum(CASE WHEN produced > 0 THEN 1 ELSE 0 END) AS with_producer,
                       sum(produced) AS producing_edges,
                       sum(coalesce(s.attempt_count, 0)) AS attempt_total,
                       sum(CASE WHEN coalesce(s.attempt_count, 0) > 0 THEN 1 ELSE 0 END)
                           AS with_attempts,
                       sum(CASE WHEN s.created_at <= datetime($since) THEN 1 ELSE 0 END)
                           AS existing_at_cut,
                       sum(CASE WHEN s.created_at > datetime($since) THEN 1 ELSE 0 END)
                           AS created_since,
                       sum(CASE WHEN s.composed_at > datetime($since)
                                THEN 1 ELSE 0 END) AS composed_since,
                       sum(CASE WHEN s.created_at IS NOT NULL THEN 1 ELSE 0 END)
                           AS sources_with_created_at
                """,
                paths=ROSTER_PATHS,
                since=since,
            )
        )[0]
        by_status = [
            dict(row)
            for row in gc.query(
                """
                MATCH (s:StandardNameSource)
                WHERE s.dd_path IS NULL OR NOT s.dd_path IN $paths
                RETURN s.status AS status, count(*) AS n ORDER BY n DESC
                """,
                paths=ROSTER_PATHS,
            )
        ]
        changes = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
                WHERE c.changed_at >= datetime($since)
                RETURN sn.id IN $names AS roster,
                       c.operation AS operation, count(*) AS n
                """,
                names=ROSTER_NAMES,
                since=since,
            )
        ]
        composed_changes = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
                WHERE c.changed_at >= datetime($since) AND NOT sn.id IN $names
                RETURN c.id AS id, sn.id AS name, c.operation AS operation,
                       c.changed_at AS changed_at, c.from_name AS from_name,
                       c.to_name AS operation_to
                LIMIT 25
                """,
                names=ROSTER_NAMES,
                since=since,
            )
        ]
    return {
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cutoff": since,
        "roster_excluded_paths": len(ROSTER_PATHS),
        "roster_excluded_names": len(ROSTER_NAMES),
        "sources": dict(sources),
        "sources_by_status": by_status,
        "name_changes_since_cutoff": changes,
        "name_changes_since_cutoff_outside_roster": composed_changes,
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

    p_replay = sub.add_parser("replay")
    p_replay.add_argument("--census", required=True)
    p_replay.add_argument("--out", required=True)

    p_scope = sub.add_parser("outofroster")
    p_scope.add_argument("--since", required=True)
    p_scope.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command in {"replay", "outofroster"}:
        payload = (
            replay(Path(args.census))
            if args.command == "replay"
            else outofroster(args.since)
        )
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        for key in sorted(payload):
            if isinstance(payload[key], (int, str, list)):
                print(f"{key}={payload[key]}")
        return 0
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

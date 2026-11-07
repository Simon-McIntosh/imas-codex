#!/usr/bin/env python3
"""Runner for the five-stub composition node.

Three subcommands, each writing one JSON document to ``--out``:

``state``
    Read-only. Records the roster's five names and seven DD paths as the graph
    holds them: each stub's stage, producers, description and unit, and each
    source's status, attempt count and stored last error. It also counts the
    sourceless live draft population through
    ``source_invariant.find_sourceless_live_names`` over every live name, so
    the before and after figures come from the same code path rather than from
    prose. ``latest_run_started_at`` is the cut-off used to select the compose
    run's own ``SNRun`` row afterwards.

``compose``
    Invokes the sanctioned pipeline, ``sn run``, scoped with ``--focus`` to the
    seven roster paths and capped with ``--cost-limit``, capturing its whole
    stdout/stderr to ``--log``. ``--dry-run`` previews the pools and writes
    nothing. ``--reset`` adds ``--reset-to extracted`` so the focused paths,
    which are parked at their attempt cap by earlier composition, are staged
    again for this run; the reset is exact-path scoped.

``readback``
    Read-only. Selects the compose run by its own id -- the single ``SNRun``
    row created after the recorded cut-off -- and reports that row's spend.
    It records each stub's outcome (adopted with producer, description and
    unit, or the stored refusal) and lists every ``StandardNameChange`` made
    after the cut-off, so a change outside the five is visible rather than
    assumed absent.

The roster is fixed in this file rather than passed in so a later reader can
re-run the read and get the same row set.
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
    "radius_of_soft_xray_detector",
    "ratio_of_diamagnetic_vorticity_to_major_radius",
    "toroidal_coordinate_of_spectrometer",
    "toroidal_tritium_velocity",
]

ROSTER_PATHS = [
    "summary/line_average/n_i/neon/value",
    "soft_x_rays/channel/detector/radius",
    "edge_profiles/ggd/vorticity_over_r/diamagnetic",
    "plasma_profiles/ggd/vorticity_over_r/diamagnetic",
    "spectrometer_visible/channel/geometry_matrix/interpolated/phi",
    "summary/local/pedestal/velocity_phi/tritium/value",
    "summary/local/separatrix/velocity_phi/tritium/value",
]


def _stubs(gc) -> list[dict]:
    return [
        dict(row)
        for row in gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IN $ids
            OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            WITH sn, collect(DISTINCT src) AS srcs
            RETURN sn.id AS id, sn.name_stage AS name_stage,
                   sn.status AS status, sn.description AS description,
                   sn.unit AS unit,
                   [s IN srcs WHERE s IS NOT NULL | s.dd_path] AS producer_paths,
                   [s IN srcs WHERE s IS NOT NULL | s.status] AS producer_status,
                   size([s IN srcs WHERE s IS NOT NULL]) AS producers
            ORDER BY sn.id
            """,
            ids=ROSTER_NAMES,
        )
    ]


def _sources(gc) -> list[dict]:
    return [
        dict(row)
        for row in gc.query(
            """
            MATCH (s:StandardNameSource)
            WHERE s.dd_path IN $paths
            OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(n:StandardName)
            RETURN s.dd_path AS path, s.status AS status,
                   coalesce(s.attempt_count, 0) AS attempt_count,
                   s.claim_seq AS claim_seq,
                   s.parked_disposition AS parked_disposition,
                   s.last_error AS last_error,
                   collect(DISTINCT n.id) AS produced
            ORDER BY s.dd_path
            """,
            paths=ROSTER_PATHS,
        )
    ]


def _sourceless_live_drafts(gc) -> dict:
    from imas_codex.standard_names.source_invariant import (
        find_sourceless_live_names,
    )

    live_ids = [
        str(row["id"])
        for row in gc.query(
            """
            MATCH (sn:StandardName)
            WHERE coalesce(sn.name_stage, '') <> 'superseded'
              AND coalesce(sn.status, '') <> 'superseded'
            RETURN sn.id AS id ORDER BY sn.id
            """
        )
    ]
    sourceless = find_sourceless_live_names(gc, live_ids)
    return {
        "live_population": len(live_ids),
        "sourceless_live_count": len(sourceless),
        "sourceless_live_ids": sourceless,
        "sourceless_in_roster": sorted(set(sourceless) & set(ROSTER_NAMES)),
    }


def _latest_run_started_at(gc) -> str | None:
    rows = list(
        gc.query(
            """
            MATCH (r:SNRun)
            RETURN r.id AS id, toString(r.started_at) AS started_at
            ORDER BY r.started_at DESC LIMIT 1
            """
        )
    )
    return rows[0]["started_at"] if rows else None


def state() -> dict:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        stubs = _stubs(gc)
        sources = _sources(gc)
        drafts = _sourceless_live_drafts(gc)
        latest = _latest_run_started_at(gc)
    return {
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster_names": ROSTER_NAMES,
        "roster_paths": ROSTER_PATHS,
        "stubs": stubs,
        "sources": sources,
        "stubs_present": len(stubs),
        "sources_present": len(sources),
        "stubs_with_producer": sorted(
            row["id"] for row in stubs if row["producers"] > 0
        ),
        "sourceless_live_drafts": drafts,
        "latest_run_started_at": latest,
    }


def readback(since: str) -> dict:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        stubs = _stubs(gc)
        sources = _sources(gc)
        drafts = _sourceless_live_drafts(gc)
        runs = [
            dict(row)
            for row in gc.query(
                """
                MATCH (r:SNRun)
                WHERE datetime(r.started_at) >= datetime($since)
                RETURN r.id AS id, r.status AS status,
                       toString(r.started_at) AS started_at,
                       toString(r.finished_at) AS finished_at,
                       r.cost_spent AS cost_spent, r.cost_total AS cost_total,
                       r.stop_reason AS stop_reason,
                       r.names_composed AS names_composed,
                       r.names_reviewed AS names_reviewed
                ORDER BY r.started_at
                """,
                since=since,
            )
        ]
        changes = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
                WHERE c.changed_at >= datetime($since)
                RETURN sn.id AS name, c.operation AS operation,
                       toString(c.changed_at) AS changed_at,
                       c.from_name AS from_name, c.to_name AS to_name
                ORDER BY c.changed_at
                """,
                since=since,
            )
        ]
        receipts = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
                WHERE c.changed_at >= datetime($since)
                  AND c.operation = 'adopt_pending_projection'
                RETURN sn.id AS name, c.id AS change_id,
                       toString(c.changed_at) AS changed_at
                ORDER BY sn.id
                """,
                since=since,
            )
        ]
    changed_names = sorted({row["name"] for row in changes})
    return {
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "since": since,
        "roster_names": ROSTER_NAMES,
        "stubs": stubs,
        "sources": sources,
        "sourceless_live_drafts": drafts,
        "runs": runs,
        "run_ids": [row["id"] for row in runs],
        "adoption_receipts": receipts,
        "name_changes_since_cutoff": changes,
        "changed_names": changed_names,
        "changed_names_outside_roster": sorted(set(changed_names) - set(ROSTER_NAMES)),
    }


def compose(
    *,
    dry_run: bool,
    reseed: bool,
    skip_global_maintenance: bool,
    cost_limit: float,
    minutes: float,
    log_path: Path,
) -> int:
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
    if reseed:
        cmd += ["--reseed"]
    if skip_global_maintenance:
        cmd += ["--skip-global-maintenance"]
    if dry_run:
        cmd += ["--dry-run"]
    header = (
        f"# command: {' '.join(cmd)}\n"
        f"# cwd: {Path.cwd()}\n"
        f"# tree: {Path(__file__).resolve()}\n"
        f"# started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )
    with log_path.open("w") as log:
        log.write(header)
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ)
        log.write(
            f"\n# finished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        )
        log.write(f"EXIT={proc.returncode}\n")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_state = sub.add_parser("state")
    p_state.add_argument("--out", required=True)

    p_compose = sub.add_parser("compose")
    p_compose.add_argument("--log", required=True)
    p_compose.add_argument("--cost-limit", type=float, default=2.00)
    p_compose.add_argument("--minutes", type=float, default=12)
    p_compose.add_argument("--reseed", action="store_true")
    p_compose.add_argument("--skip-global-maintenance", action="store_true")
    p_compose.add_argument("--dry-run", action="store_true")

    p_back = sub.add_parser("readback")
    p_back.add_argument("--since", required=True)
    p_back.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.command == "state":
        payload = state()
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        drafts = payload["sourceless_live_drafts"]
        print(
            f"stubs_present={payload['stubs_present']} "
            f"sources_present={payload['sources_present']} "
            f"stubs_with_producer={len(payload['stubs_with_producer'])} "
            f"sourceless_live={drafts['sourceless_live_count']} "
            f"of_live={drafts['live_population']} "
            f"sourceless_in_roster={len(drafts['sourceless_in_roster'])} "
            f"latest_run={payload['latest_run_started_at']}"
        )
        return 0

    if args.command == "readback":
        payload = readback(args.since)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        drafts = payload["sourceless_live_drafts"]
        print(
            f"run_ids={payload['run_ids']} "
            f"changed_names={payload['changed_names']} "
            f"outside_roster={payload['changed_names_outside_roster']} "
            f"sourceless_live={drafts['sourceless_live_count']}"
        )
        return 0

    return compose(
        dry_run=args.dry_run,
        reseed=args.reseed,
        skip_global_maintenance=args.skip_global_maintenance,
        cost_limit=args.cost_limit,
        minutes=args.minutes,
        log_path=Path(args.log),
    )


if __name__ == "__main__":
    sys.exit(main())

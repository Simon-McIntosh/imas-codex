#!/usr/bin/env python
"""Read the gap-family split's before-state and exercise the sanctioned owners.

The node moves three midplane measurement distances off
``gap_at_outboard_midplane`` onto a revived
``radial_separatrix_distance_at_outboard_midplane``.  The revival of a
superseded identity is a precondition for the move: the sanctioned
PRODUCED_NAME mover refuses a retired successor, so the target must be live
before any source can be retargeted onto it.

This script performs no write.  It reads the before-state with a
property-existence control on every property it reports, binds each of the
family's four sources to whichever live names currently hold it, parses both
names through the project's own ISN grammar, and runs the sanctioned owners in
their preview modes so their verdicts are recorded rather than inferred.

Usage:
    uv run --no-sync python docs/evidence/fragments/sn-harmonize-command/shc-gap-family-split-run.py
"""

from __future__ import annotations

import json
import sys

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.cascade import _isn_parser, _isn_round_trip_ok
from imas_codex.standard_names.edit import supersede_into

TARGET = "radial_separatrix_distance_at_outboard_midplane"
HOLDER = "gap_at_outboard_midplane"
FAMILY = [
    TARGET,
    HOLDER,
    "distance_at_outboard_midplane",
    "radial_gap",
]
#: The four sources the two names must hold between them, before and after.
FAMILY_SOURCES = [
    "dd:summary/boundary/distance_inner_outer_separatrices/value",
    "dd:langmuir_probes/reciprocating/plunge/distance_separatrix_midplane",
    "dd:langmuir_probes/embedded/distance_separatrix_midplane",
    "dd:thomson_scattering/channel/distance_separatrix_midplane",
]
#: Properties every StandardName carries, used as the read's positive control.
CONTROL_PRESENT = "id"
#: A property no StandardName is expected to carry, used as the negative
#: control: the same read must report this one as absent.
CONTROL_ABSENT = "this_property_is_never_set_by_any_writer"
LIVE_STAGES = ("pending", "drafted", "reviewed", "accepted", "approved", "refining")


def read_names(gc: GraphClient) -> dict[str, object]:
    """Read the family's lifecycle scalars with an existence control each.

    Every reported property is read together with its ``properties()``
    membership, so a null value and a missing property cannot be confused: a
    writer that never set the property and a writer that cleared it both read
    as null through a bare property access, and only the membership tells them
    apart.
    """
    rows = list(
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IN $ids
            RETURN sn.id AS id,
                   properties(sn) AS props
            ORDER BY sn.id
            """,
            ids=FAMILY,
        )
    )
    out: dict[str, object] = {}
    for row in rows:
        props = dict(row["props"])
        entry = {
            "name_stage": props.get("name_stage"),
            "status": props.get("status"),
            "docs_stage": props.get("docs_stage"),
            "superseded_from_stage": props.get("superseded_from_stage"),
            "unit": props.get("unit"),
            "edit_mode": props.get("edit_mode"),
            "edit_status": props.get("edit_status"),
            "claim_token": props.get("claim_token"),
            "binding_lock_token": props.get("binding_lock_token"),
            "property_present": {
                key: key in props
                for key in (
                    "name_stage",
                    "status",
                    "docs_stage",
                    "superseded_from_stage",
                    "unit",
                )
            },
            "control_present": CONTROL_PRESENT in props,
            "control_absent": CONTROL_ABSENT in props,
        }
        out[row["id"]] = entry
    return out


def read_source_bindings(gc: GraphClient) -> dict[str, object]:
    """Bind every family source to the live names holding it, before the move.

    A source bound to more than one live name, or to none, is the failure the
    move must not create, so the reading enumerates all holders -- live and
    retired -- rather than only the one this node cares about.
    """
    rows = list(
        gc.query(
            """
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            WHERE src.id IN $ids
            RETURN src.id AS source_id,
                   collect({
                     name: sn.id,
                     stage: sn.name_stage,
                     status: sn.status
                   }) AS holders,
                   src.produced_sn_id AS scalar_target
            ORDER BY src.id
            """,
            ids=FAMILY_SOURCES,
        )
    )
    out: dict[str, object] = {}
    for row in rows:
        holders = [dict(h) for h in row["holders"]]
        live = [h for h in holders if h.get("stage") in LIVE_STAGES]
        out[row["source_id"]] = {
            "holders": holders,
            "live_holders": [h["name"] for h in live],
            "live_holder_count": len(live),
            "scalar_target": row["scalar_target"],
            "scalar_matches_live_holder": (
                len(live) == 1 and row["scalar_target"] == live[0]["name"]
            ),
        }
    missing = sorted(set(FAMILY_SOURCES) - set(out))
    out["_missing_sources"] = missing
    return out


def parse_name(name: str) -> dict[str, object]:
    """Parse one name through the project's own ISN grammar and read it back."""
    ok, detail = _isn_round_trip_ok(name)
    entry: dict[str, object] = {"round_trip_ok": ok, "detail": detail}
    try:
        ir = _isn_parser.parse(name).ir
    except Exception as exc:  # noqa: BLE001 - the refusal is the evidence
        entry["parse_error"] = f"{exc.__class__.__name__}: {exc}"
        return entry
    entry["segments"] = ir.model_dump(exclude_none=True)
    return entry


def writer_quiescence(gc: GraphClient) -> dict[str, object]:
    """Count the graph's live writer reservations.

    A signed apply needs a quiet graph.  Claims and binding locks are the
    reservations a concurrent writer holds while it composes or migrates, so a
    nonzero count here means the apply must wait.
    """
    rows = list(
        gc.query(
            """
            CALL () {
              MATCH (sn:StandardName)
              WHERE sn.claim_token IS NOT NULL OR sn.binding_lock_token IS NOT NULL
              RETURN count(sn) AS claimed_names
            }
            CALL () {
              MATCH (src:StandardNameSource)
              WHERE src.claim_token IS NOT NULL
              RETURN count(src) AS claimed_sources
            }
            RETURN claimed_names, claimed_sources
            """
        )
    )
    row = dict(rows[0]) if rows else {}
    return {
        "claimed_names": row.get("claimed_names"),
        "claimed_sources": row.get("claimed_sources"),
        "quiet": row.get("claimed_names") == 0 and row.get("claimed_sources") == 0,
    }


def revive_previews() -> dict[str, object]:
    """Run the sanctioned fold-revival owner in preview mode, per donor.

    ``supersede_into(old, into)`` is the owner that restores a superseded
    identity: folding a live name into a superseded tombstone re-occupies the
    tombstone instead of leaving the carried sources with no live name.  It is
    exercised here only as a preview; the donor set is every retired member of
    the family, which is the whole set of names a fold could legitimately
    absorb.
    """
    previews: dict[str, object] = {}
    for donor in ("distance_at_outboard_midplane", "radial_gap"):
        try:
            previews[donor] = supersede_into(donor, TARGET, dry_run=True)
        except Exception as exc:  # noqa: BLE001 - the refusal is the evidence
            previews[donor] = {"raised": f"{exc.__class__.__name__}: {exc}"}
    return previews


def main() -> int:
    with GraphClient() as gc:
        payload = {
            "names": read_names(gc),
            "source_bindings": read_source_bindings(gc),
            "isn_grammar": {
                "target": parse_name(TARGET),
                "holder": parse_name(HOLDER),
            },
            "writer_quiescence": writer_quiescence(gc),
        }
    payload["revive_previews"] = revive_previews()
    payload["negative_control"] = {
        "assertion": (
            "the pre-apply readback must report the three measurement sources "
            "on gap_at_outboard_midplane and must fail any assertion that they "
            "are held by the revived name"
        ),
        "three_measurement_sources_on_holder": sorted(
            source
            for source, entry in payload["source_bindings"].items()  # type: ignore[union-attr]
            if isinstance(entry, dict)
            and entry.get("live_holders") == [HOLDER]
            and source != FAMILY_SOURCES[0]
        ),
    }
    print(json.dumps(payload, indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

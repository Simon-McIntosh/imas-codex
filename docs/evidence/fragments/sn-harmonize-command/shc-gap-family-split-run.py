#!/usr/bin/env python
"""Split the gap family: revive the measurement distance, then move its sources.

``gap_at_outboard_midplane`` holds four data-dictionary sources. One of them —
the separatrix separation, ``dd:summary/boundary/distance_inner_outer_separatrices/value``
— is what the name means, and stays. The other three are midplane measurement
distances that the name was never about, and the ruling moves them to
``radial_separatrix_distance_at_outboard_midplane``, which parses on ISN's new
``separatrix_distance`` base.

That target is a tombstone, and the sanctioned source mover refuses a retired
successor, so the revival comes first. The revival goes through the signed
repair envelope rather than through an owner code change: one ``set_properties``
row returns the identity to a live stage and a non-terminal catalog status, with
the ruling recorded as its authority.

Nothing here builds Cypher. The revival is ``apply_signed_manifest`` over an
authority built by ``build_repair_authority``; the move is
``retarget_standard_name_sources``. Modes, in the order a run uses them:

    read      the before-state, with an existence control on every property
    build     emit the signed authority artifact
    preview   dry-run the authority and return its digest
    apply     commit it under the preview's digest
    move      retarget the three measurement sources
    after     the done-when readback of both names and all four sources
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.cascade import _isn_parser, _isn_round_trip_ok
from imas_codex.standard_names.edit import supersede_into
from imas_codex.standard_names.provenance_lifecycle import (
    retarget_standard_name_sources,
)
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

TARGET = "radial_separatrix_distance_at_outboard_midplane"
HOLDER = "gap_at_outboard_midplane"
FAMILY = [TARGET, HOLDER, "distance_at_outboard_midplane", "radial_gap"]

#: What each name means, and which source carries it. The split's whole point is
#: that one name held four sources of two different kinds.
SEPARATRIX_SOURCE = "dd:summary/boundary/distance_inner_outer_separatrices/value"
MEASUREMENT_SOURCES = [
    "dd:langmuir_probes/reciprocating/plunge/distance_separatrix_midplane",
    "dd:langmuir_probes/embedded/distance_separatrix_midplane",
    "dd:thomson_scattering/channel/distance_separatrix_midplane",
]
FAMILY_SOURCES = [SEPARATRIX_SOURCE, *MEASUREMENT_SOURCES]

#: Properties every StandardName carries, used as the read's positive control.
CONTROL_PRESENT = "id"
#: A property no StandardName carries, used as the negative control: the same
#: read must report this one as absent on every node.
CONTROL_ABSENT = "this_property_is_never_set_by_any_writer"
LIVE_STAGES = ("pending", "drafted", "reviewed", "accepted", "approved", "refining")

AUTHORITY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "shc-gap-family-split-manifest.json",
)
AUTHORITY_MODE = "external_reviewed"
REVIVE_OPERATION = "revive_superseded_identity"
REVIVE_OPERATION_ID = "revive-outboard-midplane-separatrix-distance"
REVIVE_REASON = (
    "revive radial_separatrix_distance_at_outboard_midplane so the midplane "
    "measurement distances can move off gap_at_outboard_midplane; authority is "
    "comment c-s4-lead-ruling-gap-split on plan sn-harmonize-command section 4"
)
MOVE_OPERATION = "source_migration_manifest"
MOVE_REASON = (
    "the midplane measurement distances are not the outboard-midplane gap; the "
    "ruling in c-s4-lead-ruling-gap-split moves them to the revived "
    "radial_separatrix_distance_at_outboard_midplane"
)
RUN_ID = os.environ.get(
    "SHC_GAP_FAMILY_RUN_ID", "r-20261018T194720843228-shc-gap-family-split"
)


def read_names(gc: GraphClient) -> dict[str, object]:
    """Read the family's lifecycle scalars with an existence control each.

    Every property is read from the node's ``properties()`` map, so a null
    value and a property no writer ever set cannot be confused: a bare property
    access reports both as null, and only the membership tells them apart.
    """
    rows = list(
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IN $ids
            RETURN sn.id AS id, properties(sn) AS props
            ORDER BY sn.id
            """,
            ids=FAMILY,
        )
    )
    out: dict[str, object] = {}
    for row in rows:
        props = dict(row["props"])
        out[row["id"]] = {
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
    return out


def read_source_bindings(gc: GraphClient) -> dict[str, object]:
    """Bind every family source to the names holding it, live and retired.

    A source held by two live names, or by none, is the failure this node must
    not create, so the reading enumerates every holder and reports the scalar
    mirror beside the edge: the two agree in a consistent graph and diverge in
    the middle of a bad migration.
    """
    rows = list(
        gc.query(
            """
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            WHERE src.id IN $ids
            RETURN src.id AS source_id,
                   collect({name: sn.id, stage: sn.name_stage,
                            status: sn.status}) AS holders,
                   src.produced_sn_id AS scalar_target
            ORDER BY src.id
            """,
            ids=FAMILY_SOURCES,
        )
    )
    out: dict[str, object] = {}
    for row in rows:
        holders = [dict(holder) for holder in row["holders"]]
        live = [holder for holder in holders if holder.get("stage") in LIVE_STAGES]
        out[row["source_id"]] = {
            "holders": holders,
            "live_holders": [holder["name"] for holder in live],
            "live_holder_count": len(live),
            "scalar_target": row["scalar_target"],
            "scalar_matches_live_holder": (
                len(live) == 1 and row["scalar_target"] == live[0]["name"]
            ),
        }
    out["_missing_sources"] = sorted(set(FAMILY_SOURCES) - set(out))
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

    A signed apply reads the whole participant closure inside its transaction,
    so a concurrent writer composing on the same family would make the digest
    the preview returns describe a state the apply no longer sees. Claims and
    binding locks are that concurrency, so the reading is taken before each
    write and recorded.
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
    """Preview the fold-revival owner that was refused for this identity.

    Kept in the record because the refusal is what routes the revival through
    the signed envelope instead: an owner that cannot restore a tombstone is
    not evidence that the identity is unrestorable, but it is evidence that the
    revival needs authorisation rather than a flag.
    """
    previews: dict[str, object] = {}
    for donor in ("distance_at_outboard_midplane", "radial_gap"):
        try:
            previews[donor] = supersede_into(donor, TARGET, dry_run=True)
        except Exception as exc:  # noqa: BLE001 - the refusal is the evidence
            previews[donor] = {"raised": f"{exc.__class__.__name__}: {exc}"}
    return previews


def read_state() -> dict:
    """The whole before- or after-state in one reading."""
    with GraphClient() as gc:
        state = {
            "uri": gc.uri,
            "names": read_names(gc),
            "source_bindings": read_source_bindings(gc),
            "isn_grammar": {
                "target": parse_name(TARGET),
                "holder": parse_name(HOLDER),
            },
            "writer_quiescence": writer_quiescence(gc),
        }
    return state


def after_state_assertion(state: dict) -> dict[str, object]:
    """Whether a reading is the split's post-state, source by source.

    This is the assertion the pre-state readback must fail: it names, for every
    family source, the one live name expected to hold it.
    """
    expected = {
        **dict.fromkeys(MEASUREMENT_SOURCES, TARGET),
        SEPARATRIX_SOURCE: HOLDER,
    }
    bindings = state["source_bindings"]
    per_source = {}
    for source, name in sorted(expected.items()):
        entry = bindings.get(source) or {}
        per_source[source] = {
            "expected_live_holder": name,
            "observed_live_holders": entry.get("live_holders"),
            "ok": entry.get("live_holders") == [name]
            and entry.get("scalar_matches_live_holder") is True,
        }
    names = state["names"]
    target_live = names.get(TARGET, {}).get("name_stage") in LIVE_STAGES
    holder_live = names.get(HOLDER, {}).get("name_stage") in LIVE_STAGES
    holds = {
        "target_docs_stage": names.get(TARGET, {}).get("docs_stage"),
        "holder_docs_stage": names.get(HOLDER, {}).get("docs_stage"),
    }
    return {
        "per_source": per_source,
        "revived_name_is_live": target_live,
        "holder_is_live": holder_live,
        "documentation": holds,
        "ok": target_live
        and holder_live
        and all(item["ok"] for item in per_source.values()),
    }


def build_row() -> dict:
    """One signed row returning the tombstone to a live lifecycle state.

    The three values move together because they are one decision: the name axis
    leaves the terminal stage, the catalog axis stops excluding the spelling
    from export, and the recorded prior stage — which for this identity is
    itself the terminal one — is cleared so nothing downstream reads a live
    name as having been superseded from a superseded state.
    """
    return {
        "id": TARGET,
        "identity": {"id": TARGET, "kind": "standard_name", "target_id": TARGET},
        "participants": [{"id": TARGET, "kind": "node", "graph_label": "StandardName"}],
        "mutations": [
            {
                "id": f"{TARGET}:revive",
                "order": 0,
                "kind": "set_properties",
                "participant_id": TARGET,
                "arguments": {
                    "properties": {
                        "name_stage": "drafted",
                        "status": "draft",
                        "superseded_from_stage": None,
                    }
                },
            }
        ],
        "guards": [
            {
                "id": f"{TARGET}:out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [TARGET],
            }
        ],
        "orphan_policy": "refuse",
        "selection": {
            "id": "artifact-rows",
            "mode": "exact_complete_signed_cohort",
            "predicate": "artifact-rows",
        },
    }


def build_authority() -> dict:
    return {
        "operation_id": REVIVE_OPERATION_ID,
        "authority_mode": AUTHORITY_MODE,
        "orphan_policy": "refuse",
        "selection": {
            "id": "artifact-rows",
            "mode": "exact_complete_signed_cohort",
            "predicate": "artifact-rows",
        },
        "receipt_policy": {
            "id": "one-per-revived-identity",
            "operation": REVIVE_OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "rows": [build_row()],
    }


def write_authority() -> dict:
    built = build_repair_authority(build_authority())
    with open(AUTHORITY_PATH, "wb") as handle:
        handle.write(built.content)
    return {
        "authority_path": AUTHORITY_PATH,
        "authority_file_sha256": built.file_sha256,
        "authority_payload_sha256": built.payload_sha256,
        "rows": 1,
    }


def run_authority(apply: bool, digest: str | None) -> dict:
    raw = open(AUTHORITY_PATH, "rb").read()
    result = apply_signed_manifest(
        AUTHORITY_PATH,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(json.loads(raw)),
        reason=REVIVE_REASON,
        apply=apply,
        manifest_sha256=digest if apply else None,
        run_id=RUN_ID,
    )
    return {
        key: value
        for key, value in result.items()
        if key not in {"manifest", "admitted_actions"}
    }


def move_sources() -> dict:
    """Move the three measurement sources onto the revived name.

    The move names every source and its expected current target, so it is a
    compare-and-set on exactly this cohort: a source that has moved since the
    readback makes it fail rather than quietly migrate a different set.
    """
    with GraphClient() as gc:
        quiet = writer_quiescence(gc)
        moved = retarget_standard_name_sources(
            gc,
            HOLDER,
            TARGET,
            operation=MOVE_OPERATION,
            reason=MOVE_REASON,
            origin="signed_manifest",
            run_id=RUN_ID,
            source_ids=MEASUREMENT_SOURCES,
            expected_current_bindings=dict.fromkeys(MEASUREMENT_SOURCES, HOLDER),
        )
    return {"moved": moved, "expected": len(MEASUREMENT_SOURCES), "quiescence": quiet}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "read"
    if mode == "read":
        state = read_state()
        state["negative_control"] = after_state_assertion(state)
        state["revive_previews"] = revive_previews()
        print(json.dumps(state, indent=2, default=str, sort_keys=True))
    elif mode == "build":
        print(json.dumps(write_authority(), indent=2))
    elif mode == "preview":
        print(json.dumps(run_authority(False, None), indent=2, default=str))
    elif mode == "apply":
        digest = sys.argv[2]
        print(json.dumps(run_authority(True, digest), indent=2, default=str))
    elif mode == "move":
        print(json.dumps(move_sources(), indent=2, default=str))
    elif mode == "after":
        state = read_state()
        state["after_state_assertion"] = after_state_assertion(state)
        print(json.dumps(state, indent=2, default=str, sort_keys=True))
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Record the retirement of the superseded refine products no successor is named for.

The live cohort is ten superseded standard names whose recorded history names a
predecessor through a `refine` or `backfill_refine` change but whose history
names no live successor: they carry no outgoing `HAS_SUCCESSOR` edge and no
incoming `REFINED_FROM` edge. Each gets a signed retirement change (a
`supersede` mutation with no signed successor) and stays terminal at
`superseded`, as the source-less superseded names did.

Every graph write goes through `apply_signed_manifest`: the authority is built
by `build_repair_authority` (the owner of the `imas-codex.repair-authority.v1`
wire artifact), previewed, applied under the preview's digest, then read back
and replayed. One control row — an accepted name carrying a live structural
child — must be refused in the preview and left untouched by the apply.

Modes: read | build | preview | apply | readback | replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict

from imas_codex.graph import GraphClient
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

TEN = [
    "beam_area_of_neutral_beam_injector",
    "equilibrium_weight_of_poloidal_magnetic_field_probe",
    "initial_incident_wave_stokes_parameter_at_line_of_sight",
    "ion_vibrational_level",
    "radial_outline_of_plasma_facing_component",
    "radial_plasma_momentum_source",
    "ratio_of_neutral_hydrogenic_density_to_difference_of_total_neutral_hydrogenic_density_and_neutral_hydrogenic_density",
    "time_derivative_of_bremsstrahlung_count_at_detector_pixel",
    "toroidal_plasma_beta",
    "wave_power_at_measurement_position",
]

SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
RETIRE_OPERATION = "supersede_exhausted_orphan"
RETIRE_OPERATION_ID = (
    "record-unlinked-refine-product-retirement-that-history-left-unrecorded"
)
REASON = (
    "the refine product's recorded history names a predecessor and no live "
    "successor; record the retirement it lacks and leave it terminal at superseded"
)
AUTHORITY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sib-retire-unlinked-refine-products-authority.json",
)
RUN_ID = os.environ.get(
    "SIB_RUN_ID", "r-20261015T042445570608-sib-retire-unlinked-refine-products"
)


def build_row(name_id: str) -> dict:
    return {
        "id": name_id,
        "identity": {"id": name_id, "kind": "standard_name", "target_id": name_id},
        "participants": [
            {"id": name_id, "kind": "node", "graph_label": "StandardName"}
        ],
        "mutations": [
            {
                "id": f"{name_id}:supersede",
                "order": 0,
                "kind": "supersede",
                "participant_id": name_id,
            }
        ],
        "guards": [
            {
                "id": "structural-legitimacy",
                "kind": "semantic_authority",
                "implementation": "structural-legitimacy",
                "participant_ids": [name_id],
            },
            {
                "id": "out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [name_id],
            },
        ],
        "orphan_policy": "refuse",
        "selection": dict(SELECTION),
    }


def _control_id(gc: GraphClient) -> str | None:
    rows = gc.query(
        "MATCH (sn:StandardName {name_stage:'accepted'}) "
        "WHERE COUNT { (c:StandardName)-[:HAS_PARENT]->(sn) } >= 1 "
        "RETURN sn.id AS id ORDER BY id LIMIT 1"
    )
    return rows[0]["id"] if rows else None


def read_cohort() -> dict:
    """Re-read the ten and stop if any has moved out of the unlinked class."""
    out: dict = {}
    with GraphClient() as gc:
        out["uri"] = gc.uri
        rows = gc.query(
            "UNWIND $ids AS id "
            "MATCH (sn:StandardName {id:id}) "
            "RETURN sn.id AS id, sn.name_stage AS stage, sn.status AS status, "
            "COUNT { (sn)-[:HAS_SUCCESSOR]->(:StandardName) } AS out_succ, "
            "COUNT { ()-[:REFINED_FROM]->(sn) } AS in_refined "
            "ORDER BY id",
            ids=TEN,
        )
        by_id = {r["id"]: r for r in rows}
        moved = [
            i
            for i in TEN
            if i not in by_id
            or by_id[i]["stage"] != "superseded"
            or int(by_id[i]["out_succ"]) != 0
            or int(by_id[i]["in_refined"]) != 0
        ]
        out["cohort"] = [
            {
                "id": i,
                "present": i in by_id,
                "name_stage": by_id.get(i, {}).get("stage"),
                "status": by_id.get(i, {}).get("status"),
                "out_has_successor": int(by_id.get(i, {}).get("out_succ") or 0),
                "in_refined_from": int(by_id.get(i, {}).get("in_refined") or 0),
            }
            for i in TEN
        ]
        out["moved"] = moved
        out["all_unlinked_superseded"] = not moved

        preds = gc.query(
            "UNWIND $ids AS id "
            "MATCH (sn:StandardName {id:id})-[:HAS_INTERNAL_CHANGE]->"
            "(c:StandardNameChange) "
            "WHERE c.operation IN ['refine','backfill_refine',"
            "'supersede_legacy_spelling'] "
            "RETURN sn.id AS id, c.operation AS op, c.from_name AS from_name, "
            "c.to_name AS to_name ORDER BY id, op",
            ids=TEN,
        )
        hist: dict[str, list] = defaultdict(list)
        for p in preds:
            hist[p["id"]].append(
                {
                    "operation": p["op"],
                    "from_name": p["from_name"],
                    "to_name": p["to_name"],
                }
            )
        out["history"] = {i: hist.get(i, []) for i in TEN}

        out["control"] = _control_id(gc)
    return out


def build_authority() -> dict:
    with GraphClient() as gc:
        control = _control_id(gc)
    rows = [build_row(i) for i in TEN]
    rows.append(build_row(control))
    return {
        "operation_id": RETIRE_OPERATION_ID,
        "authority_mode": "external_reviewed",
        "orphan_policy": "refuse",
        "selection": dict(SELECTION),
        "receipt_policy": {
            "id": "one-per-retired-identity",
            "operation": RETIRE_OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "rows": rows,
    }


def write_authority() -> dict:
    spec = build_authority()
    built = build_repair_authority(spec)
    with open(AUTHORITY_PATH, "wb") as handle:
        handle.write(built.content)
    emitted = json.loads(built.content)
    return {
        "authority_file_sha256": built.file_sha256,
        "authority_payload_sha256": built.payload_sha256,
        "rows": len(emitted["rows"]),
        "control": spec["rows"][-1]["id"],
        "row_ids": [r["id"] for r in spec["rows"]],
    }


def run(apply: bool, digest: str | None) -> dict:
    raw = open(AUTHORITY_PATH, "rb").read()
    auth = json.loads(raw)
    file_sha = hashlib.sha256(raw).hexdigest()
    payload_sha = signed_payload_sha256(auth)
    res = apply_signed_manifest(
        AUTHORITY_PATH,
        authority_file_sha256=file_sha,
        authority_payload_sha256=payload_sha,
        reason=REASON,
        apply=apply,
        manifest_sha256=digest if apply else None,
        run_id=RUN_ID,
    )
    return {k: res[k] for k in res if k not in {"manifest", "admitted_actions"}}


def readback() -> dict:
    raw = open(AUTHORITY_PATH, "rb").read()
    file_sha = hashlib.sha256(raw).hexdigest()
    auth = json.loads(raw)
    expected = {r["id"] for r in auth["rows"][:-1]}
    control = auth["rows"][-1]["id"]
    with GraphClient() as gc:
        receipts = gc.query(
            "MATCH (c:StandardNameChange {authority_file_sha256:$fs}) "
            "RETURN c.row_id AS row_id, c.operation AS op, "
            "c.manifest_sha256 AS digest",
            fs=file_sha,
        )
        per: dict[str, int] = defaultdict(int)
        for c in receipts:
            per[c["row_id"]] += 1
        stage = gc.query(
            "UNWIND $ids AS id "
            "MATCH (sn:StandardName {id:id}) "
            "RETURN count(sn) AS n, "
            "count(CASE WHEN sn.name_stage='superseded' THEN 1 END) AS superseded",
            ids=sorted(expected),
        )[0]
        control_row = gc.query(
            "MATCH (sn:StandardName {id:$id}) RETURN sn.name_stage AS stage",
            id=control,
        )
    return {
        "authority_file_sha256": file_sha,
        "manifest_sha256": sorted({str(c["digest"]) for c in receipts}),
        "receipt_operations": sorted({str(c["op"]) for c in receipts}),
        "receipt_names": len(per),
        "expected_names": len(expected),
        "receipts_per_name_min": min(per.values()) if per else 0,
        "receipts_per_name_max": max(per.values()) if per else 1,
        "names_without_receipt": sorted(expected - set(per)),
        "still_superseded": int(stage["superseded"]),
        "control_stage": control_row[0]["stage"] if control_row else None,
        "control_receipts": per.get(control, 0),
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "read"
    if mode == "read":
        print(json.dumps(read_cohort(), indent=2, default=str))
    elif mode == "build":
        print(json.dumps(write_authority(), indent=2))
    elif mode == "preview":
        print(json.dumps(run(False, None), indent=2, default=str)[:4000])
    elif mode in {"apply", "replay"}:
        print(json.dumps(run(True, sys.argv[2]), indent=2, default=str)[:4000])
    elif mode == "readback":
        print(json.dumps(readback(), indent=2, default=str))
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()

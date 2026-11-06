"""Give the retired magnetic-field-average name one signed successor edge.

The predecessor ``flux_surface_average_magnetic_field_magnitude`` lost its DD
path ``equilibrium/time_slice/profiles_1d/b_average`` at DD 4. An earlier
retirement receipted the loss without recording where the quantity went. The
lead ruled the name superseded onto the accepted identity that holds the
renamed path ``b_field_average``, and the graph's supersession lineage is the
receipt-stamped ``HAS_SUCCESSOR`` edge.

Every write goes through its owning writer: ``apply_signed_manifest`` under an
authority built by ``build_repair_authority``, previewed before it is applied.
Nothing is written by hand Cypher and no other name is touched.

Modes: read | authority | preview | apply | readback | gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from imas_codex.graph import GraphClient
from imas_codex.standard_names.export import (
    _RELEASE_IDENTITY_HOLDS,
    _SHIPPING_NAME_STAGES,
)
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)
from imas_codex.standard_names.source_invariant import find_sourceless_live_names

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_PATH = HERE / "sib-magnetic-successor-edge-authority.json"

RUN_ID: str | None = None

PREDECESSOR = "flux_surface_average_magnetic_field_magnitude"
SUCCESSOR = "flux_surface_averaged_magnetic_field_magnitude"

SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
OPERATION_ID = "give-retired-name-one-signed-successor"
SUPERSEDE_OPERATION = "link_signed_successor"
REASON = (
    "the Data Dictionary removed the name's path and the renamed quantity is "
    "held by the accepted identity; record the supersession with a signed "
    "successor edge so the retired identity resolves to the live name"
)
ROW_ID = f"{PREDECESSOR}=>{SUCCESSOR}"


def read() -> dict[str, Any]:
    """Re-derive this node's rows from the live graph."""
    with GraphClient() as gc:
        names = [
            dict(row)
            for row in gc.query(
                """
                UNWIND [$old, $new] AS id
                MATCH (sn:StandardName {id: id})
                OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                OPTIONAL MATCH (sn)-[:HAS_SUCCESSOR]->(target:StandardName)
                OPTIONAL MATCH (sn)-[:REFINED_FROM]->(parent:StandardName)
                RETURN sn.id AS id,
                       sn.name_stage AS stage, sn.status AS status,
                       sn.validation_status AS validation,
                       sn.source_paths AS source_paths,
                       collect(DISTINCT src.id + '|' + coalesce(src.status, '?'))
                         AS sources,
                       collect(DISTINCT target.id) AS has_successor_ids,
                       collect(DISTINCT parent.id) AS refined_from_ids
                ORDER BY id
                """,
                old=PREDECESSOR,
                new=SUCCESSOR,
            )
        ]
        receipts = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName {id: $old})-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
                RETURN c.id AS id, c.operation AS operation,
                       c.from_name AS from_name, c.to_name AS to_name,
                       toString(c.changed_at) AS changed_at
                ORDER BY changed_at
                """,
                old=PREDECESSOR,
            )
        ]
    for row in names:
        row["sources"] = sorted(row["sources"])
        row["refined_from_ids"] = sorted(row["refined_from_ids"])
        row["has_successor_ids"] = sorted(row["has_successor_ids"])
    return {"names": names, "receipts": receipts}


def _row() -> dict[str, Any]:
    return {
        "id": ROW_ID,
        "identity": {"id": ROW_ID, "kind": "standard_name", "target_id": PREDECESSOR},
        "participants": [
            {"id": PREDECESSOR, "kind": "node", "graph_label": "StandardName"},
            {"id": SUCCESSOR, "kind": "node", "graph_label": "StandardName"},
        ],
        "selection": dict(SELECTION),
        "mutations": [
            {
                "id": f"{PREDECESSOR}:supersede",
                "order": 0,
                "kind": "supersede",
                "participant_id": PREDECESSOR,
                "arguments": {"successor_id": SUCCESSOR},
            }
        ],
        "guards": [
            {
                "id": f"{ROW_ID}:structural-legitimacy",
                "kind": "semantic_authority",
                "implementation": "structural-legitimacy",
                "participant_ids": [PREDECESSOR],
            },
            {
                "id": f"{ROW_ID}:out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [PREDECESSOR, SUCCESSOR],
            },
        ],
        "orphan_policy": "refuse",
    }


def build_authority() -> dict[str, Any]:
    row = _row()
    spec = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": OPERATION_ID,
        "authority_mode": "external_reviewed",
        "rows": [row],
        "repair_rows": [row["id"]],
        "selection": dict(SELECTION),
        "receipt_policy": {
            "id": "one-per-superseded-identity",
            "operation": SUPERSEDE_OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    built = build_repair_authority(spec)
    AUTHORITY_PATH.write_bytes(built.content)
    return {
        "path": str(AUTHORITY_PATH.relative_to(ROOT)),
        "rows": [row["id"]],
        "file_sha256": hashlib.sha256(AUTHORITY_PATH.read_bytes()).hexdigest(),
        "payload_sha256": built.payload_sha256,
    }


def signed(apply: bool, digest: str | None) -> dict[str, Any]:
    raw = AUTHORITY_PATH.read_bytes()
    payload = json.loads(raw)
    result = apply_signed_manifest(
        AUTHORITY_PATH,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(payload),
        reason=REASON,
        apply=apply,
        manifest_sha256=digest if apply else None,
        run_id=RUN_ID,
    )
    return {
        k: v for k, v in result.items() if k not in {"manifest", "admitted_actions"}
    }


def successor_state() -> dict[str, Any]:
    with GraphClient() as gc:
        edges = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName {id: $old})-[edge:HAS_SUCCESSOR]->(target:StandardName)
                RETURN target.id AS successor_id, edge.receipt_id AS receipt_id,
                       toString(edge.signed_at) AS ptr
                """,
                old=PREDECESSOR,
            )
        ]
        receipt = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName {id: $old})-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
                WHERE c.to_name = $new
                RETURN c.id AS id, c.operation AS operation, c.from_name AS from_name,
                       c.to_name AS to_name, c.manifest_sha256 AS manifest_sha256,
                       toString(c.changed_at) AS changed_at
                """,
                old=PREDECESSOR,
                new=SUCCESSOR,
            )
        ]
        snapshots = [
            dict(row)
            for row in gc.query(
                """
                UNWIND [$old, $new] AS id
                MATCH (sn:StandardName {id: id})
                OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                RETURN sn.id AS id, sn.name_stage AS stage, sn.status AS status,
                       sn.validation_status AS validation,
                       sn.superseded_from_stage AS superseded_from_stage,
                       toString(sn.updated_at) AS updated_at,
                       collect(DISTINCT src.id + '|' + coalesce(src.status, '?'))
                         AS sources
                ORDER BY id
                """,
                old=PREDECESSOR,
                new=SUCCESSOR,
            )
        ]
        counts = [
            dict(row)
            for row in gc.query(
                """
                MATCH (sn:StandardName {id: $old})
                OPTIONAL MATCH (sn)-[edge:HAS_SUCCESSOR]->(t:StandardName)
                RETURN count(edge) AS successor_edges,
                       collect(t.id) AS successors
                """,
                old=PREDECESSOR,
            )
        ]
    for row in snapshots:
        row["sources"] = sorted(row["sources"])
    return {
        "successor_edges": edges,
        "receipts": receipt,
        "name_snapshots": snapshots,
        "successor_count": counts,
    }


def gate() -> dict[str, Any]:
    """Compute the export gate's blocking set through the gate's own predicate."""
    with GraphClient() as gc:
        live_ids = [
            str(row["id"])
            for row in gc.query(
                """
                MATCH (sn:StandardName)
                WHERE coalesce(sn.name_stage, '') IN ['accepted', 'approved']
                RETURN sn.id AS id ORDER BY id
                """
            )
        ]
        sourceless = find_sourceless_live_names(gc, live_ids)
        stages = {
            str(row["id"]): str(row["name_stage"])
            for row in gc.query(
                "UNWIND $ids AS id MATCH (sn:StandardName {id: id}) "
                "RETURN sn.id AS id, sn.name_stage AS name_stage",
                ids=sourceless,
            )
        }
    shipping = sorted(
        name for name in sourceless if stages.get(name) in _SHIPPING_NAME_STAGES
    )
    blocking = sorted(name for name in shipping if name not in _RELEASE_IDENTITY_HOLDS)
    return {
        # The shipping-stage set before the release holds are subtracted is the
        # control that the predicate seen an empty blocking set only because
        # the holds removed it, not because it saw nothing at all.
        "shipping_sourceless_before_holds": len(shipping),
        "held_sourceless": sorted(
            name for name in shipping if name in _RELEASE_IDENTITY_HOLDS
        ),
        "sourceless_live_count": len(sourceless),
        "sourceless_live_ids": sourceless,  # noqa: F821 - see next line
        "sourceless_ids": sourceless,
        "stages": stages,
        "blocking": blocking,
        "blocking_count": len(blocking),
        "predecessor_sourceless": PREDECESSOR in sourceless,
        "successor_sourceless": SUCCESSOR in sourceless,
        "release_hold_count": len(_RELEASE_IDENTITY_HOLDS),
    }


def main() -> int:
    global RUN_ID
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["read", "authority", "preview", "apply", "readback", "gate"],
    )
    parser.add_argument("--run-id", default=os.environ.get("SIB_RUN_ID"))
    parser.add_argument("--digest", default=None)
    args = parser.parse_args()
    if not args.run_id:
        raise SystemExit("--run-id is required (or set SIB_RUN_ID)")
    RUN_ID = args.run_id
    fn = {
        "read": read,
        "authority": build_authority,
        "preview": lambda: signed(False, None),
        "apply": lambda: signed(True, args.digest),
        "readback": successor_state,
        "gate": gate,
    }[args.mode]
    print(json.dumps(fn(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

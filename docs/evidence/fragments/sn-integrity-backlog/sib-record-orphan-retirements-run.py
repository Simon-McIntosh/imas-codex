"""Record the retirement that superseded source-less names never had recorded.

Read-only census plus the signed-manifest write path. The census re-reads the
population of superseded StandardName nodes that point to no successor, splits
the class that carries no supersession event, and reports the two branches
separately. The write builds one signed authority that stamps a retirement
change (a `supersede` mutation with no signed successor) onto every name in the
source-less branch, previews it under `apply_signed_manifest`, applies under the
preview's digest, then reads back and replays. Every graph write goes through
`apply_signed_manifest`; there is no other write.

Modes: read | build | preview | apply | readback | replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict

from imas_codex.graph import GraphClient
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

CENSUS_MATCH = (
    "MATCH (sn:StandardName {name_stage:'superseded'}) "
    "WHERE NOT (sn)-[:HAS_SUCCESSOR]->(:StandardName) "
    "AND NOT ()-[:REFINED_FROM]->(sn) "
)
SUPERSESION_OPS = {
    "supersede_exhausted_orphan",
    "refine",
    "backfill_refine",
    "supersede_legacy_spelling",
    "rename",
    "human_edit",
    "retire_signed_dual_authority_target",
    "retire_signed_provenance_orphan",
    "retire_ineligible_standard_name_sources",
}
SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
RETIRE_OPERATION = "supersede_exhausted_orphan"
RETIRE_OPERATION_ID = (
    "record-orphan-retirement-that-supersession-census-found-unrecorded"
)
REASON = (
    "the supersession census found no supersession event on this source-less "
    "superseded identity; record the orphan retirement it lacks and leave it "
    "terminal at superseded"
)
AUTHORITY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sib-record-orphan-retirements-authority.json",
)
RUN_ID = os.environ.get(
    "SIB_RUN_ID", "r-20261014T200851870928-sib-record-orphan-retirements"
)


def _authority_header() -> dict:
    return {
        "authority_mode": "external_reviewed",
        "operation_id": RETIRE_OPERATION_ID,
        "orphan_policy": "refuse",
        "receipt_policy": {
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "id": "one-per-retired-identity",
            "link_participant_kind": "standard_name",
            "operation": RETIRE_OPERATION,
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "schema": "imas-codex.repair-authority.v1",
        "selection": SELECTION,
    }


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
        "selection": SELECTION,
    }


def read_split() -> dict:
    out: dict = {}
    with GraphClient() as gc:
        out["uri"] = gc.uri
        out["population"] = gc.query(
            "MATCH (sn:StandardName {name_stage:'superseded'}) RETURN count(sn) AS n"
        )[0]["n"]
        out["excluded_out_has_successor_only"] = gc.query(
            "MATCH (sn:StandardName {name_stage:'superseded'}) "
            "WHERE (sn)-[:HAS_SUCCESSOR]->(:StandardName) "
            "AND NOT ()-[:REFINED_FROM]->(sn) RETURN count(DISTINCT sn) AS n"
        )[0]["n"]
        out["excluded_in_refined_from_only"] = gc.query(
            "MATCH (sn:StandardName {name_stage:'superseded'}) "
            "WHERE ()-[:REFINED_FROM]->(sn) "
            "AND NOT (sn)-[:HAS_SUCCESSOR]->(:StandardName) "
            "RETURN count(DISTINCT sn) AS n"
        )[0]["n"]
        out["excluded_both"] = gc.query(
            "MATCH (x:StandardName {name_stage:'superseded'}) "
            "WHERE (x)-[:HAS_SUCCESSOR]->(:StandardName) "
            "AND ()-[:REFINED_FROM]->(x) RETURN count(DISTINCT x) AS n"
        )[0]["n"]
        out["excluded_total"] = gc.query(
            "MATCH (sn:StandardName {name_stage:'superseded'}) "
            "WHERE (sn)-[:HAS_SUCCESSOR]->(:StandardName) "
            "OR ()-[:REFINED_FROM]->(sn) RETURN count(DISTINCT sn) AS n"
        )[0]["n"]
        out["census"] = gc.query(CENSUS_MATCH + "RETURN count(DISTINCT sn) AS n")[0][
            "n"
        ]
        out["population_sum_check"] = out["excluded_total"] + out["census"]

        changes = gc.query(
            CENSUS_MATCH + "MATCH (sn)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange) "
            "RETURN sn.id AS id, c.operation AS op"
        )
        ops: dict[str, set] = defaultdict(set)
        for c in changes:
            ops[c["id"]].add(str(c["op"]))

        census_ids = [r["id"] for r in gc.query(CENSUS_MATCH + "RETURN sn.id AS id")]
        no_supersession = [
            i for i in census_ids if not (ops.get(i, set()) & SUPERSESION_OPS)
        ]
        out["no_supersession_class"] = len(no_supersession)

        produced = {
            r["id"]
            for r in gc.query(
                "UNWIND $ids AS id "
                "MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id:id}) "
                "RETURN DISTINCT id",
                ids=no_supersession_ids(no_supersession),
            )
        }
        source_less = sorted(i for i in no_supersession if i not in produced)
        with_source = sorted(i for i in no_supersession if i in produced)
        out["source_less"] = source_less
        out["source_less_count"] = len(source_less)
        out["with_source_count"] = len(with_source)

        control = gc.query(
            "MATCH (sn:StandardName {name_stage:'accepted'}) "
            "WHERE COUNT { (c:StandardName)-[:HAS_PARENT]->(sn) } >= 1 "
            "RETURN sn.id AS id ORDER BY id LIMIT 1"
        )
        out["control"] = control[0]["id"] if control else None

        for wid in (
            "etendue_of_spectrometer_channel",
            "x_direction_unit_vector_of_sensor",
        ):
            row = gc.query(
                "MATCH (sn:StandardName {id:$id}) RETURN sn.name_stage AS stage",
                id=wid,
            )
            out.setdefault("witnesses", {})[wid] = {
                "name_stage": row[0]["stage"] if row else None,
                "in_source_less": wid in set(source_less),
            }
    return out


def no_supersession_ids(ids: list[str]) -> list[str]:
    return ids


def split() -> tuple[list[str], str]:
    with GraphClient() as gc:
        changes = gc.query(
            CENSUS_MATCH + "MATCH (sn)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange) "
            "RETURN sn.id AS id, c.operation AS op"
        )
        ops: dict[str, set] = defaultdict(set)
        for c in changes:
            ops[c["id"]].add(str(c["op"]))
        census_ids = [r["id"] for r in gc.query(CENSUS_MATCH + "RETURN sn.id AS id")]
        no_sup = [i for i in census_ids if not (ops.get(i, set()) & SUPERSESION_OPS)]
        produced = {
            r["id"]
            for r in gc.query(
                "UNWIND $ids AS id "
                "MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id:id}) "
                "RETURN DISTINCT id",
                ids=no_sup,
            )
        }
        source_less = sorted(i for i in no_sup if i not in produced)
        control = gc.query(
            "MATCH (sn:StandardName {name_stage:'accepted'}) "
            "WHERE COUNT { (c:StandardName)-[:HAS_PARENT]->(sn) } >= 1 "
            "RETURN sn.id AS id ORDER BY id LIMIT 1"
        )
        return source_less, (control[0]["id"] if control else "")


def build_authority() -> dict:
    source_less, control = split()
    rows = [build_row(i) for i in source_less]
    rows.append(build_row(control))
    auth = _authority_header()
    auth["repair_rows"] = [r["id"] for r in rows]
    auth["rows"] = rows
    auth["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": signed_payload_sha256(auth),
    }
    return auth


def write_authority() -> dict:
    auth = build_authority()
    blob = json.dumps(auth, indent=2).encode()
    with open(AUTHORITY_PATH, "wb") as handle:
        handle.write(blob)
    return {
        "authority_file_sha256": hashlib.sha256(blob).hexdigest(),
        "authority_payload_sha256": signed_payload_sha256(auth),
        "rows": len(auth["rows"]),
    }


def preview(apply: bool, digest: str | None) -> dict:
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
    return {k: res[k] for k in res if k != "manifest"}


def readback() -> dict:
    raw = open(AUTHORITY_PATH, "rb").read()
    auth = json.loads(raw)
    file_sha = hashlib.sha256(raw).hexdigest()
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
        digests = sorted({str(c["digest"]) for c in receipts})
        ops = sorted({str(c["op"]) for c in receipts})
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
        "manifest_sha256": digests,
        "receipt_operations": ops,
        "receipt_names": len(per),
        "expected_names": len(expected),
        "receipts_per_name_min": min(per.values()) if per else 0,
        "receipts_per_name_max": max(per.values()) if per else 0,
        "names_without_receipt": sorted(expected - set(per)),
        "still_superseded": int(stage["superseded"]),
        "control_stage": control_row[0]["stage"] if control_row else None,
        "control_receipts": per.get(control, 0),
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "read"
    if mode == "read":
        print(json.dumps(read_split(), indent=2, default=str))
    elif mode == "build":
        print(json.dumps(write_authority(), indent=2))
    elif mode == "preview":
        print(json.dumps(preview(False, None), indent=2, default=str)[:4000])
    elif mode == "apply":
        digest = sys.argv[2]
        print(json.dumps(preview(True, digest), indent=2, default=str)[:4000])
    elif mode == "readback":
        print(json.dumps(readback(), indent=2, default=str))
    elif mode == "replay":
        digest = sys.argv[2]
        print(json.dumps(preview(True, digest), indent=2, default=str)[:4000])
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()

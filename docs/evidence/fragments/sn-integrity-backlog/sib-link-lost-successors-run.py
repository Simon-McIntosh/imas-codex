"""Inspect lineage and apply a signed successor authority with readback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

AUTHORITY = Path(__file__).with_name("sib-link-lost-successors-authority.json")
TARGETS = {
    "atomic_count_of_ion_state": "atomic_count",
    "minimum_of_safety_factor": "minimum_safety_factor",
    "parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift": "parallel_neutral_state_velocity_due_to_diamagnetic_drift",
    "toroidal_neutral_state_momentum_diffusivity": "toroidal_neutral_internal_state_momentum_diffusion_coefficient",
}
CONTROL = (
    "flux_surface_averaged_deuterium_tritium_density",
    "flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary",
)
SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}


def inspect() -> None:
    with GraphClient() as graph:
        population = graph.query(
            """
            MATCH (name:StandardName {name_stage: 'superseded'})
            RETURN count(name) AS population,
                   count(CASE WHEN EXISTS {(name)-[:HAS_SUCCESSOR]->(:StandardName)}
                         OR EXISTS {(:StandardName)-[:REFINED_FROM]->(name)}
                         THEN 1 END) AS excluded,
                   count(CASE WHEN NOT EXISTS {(name)-[:HAS_SUCCESSOR]->(:StandardName)}
                         AND NOT EXISTS {(:StandardName)-[:REFINED_FROM]->(name)}
                         THEN 1 END) AS census
            """
        )[0]
        rows = graph.query(
            """
            MATCH (name:StandardName {name_stage: 'superseded'})
            WHERE NOT EXISTS {(name)-[:HAS_SUCCESSOR]->(:StandardName)}
              AND NOT EXISTS {(:StandardName)-[:REFINED_FROM]->(name)}
            OPTIONAL MATCH (name)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
            WITH name, collect(properties(change)) AS changes
            WHERE any(change IN changes WHERE change.operation IN
                      ['supersede_legacy_spelling', 'refine', 'backfill_refine'])
            RETURN name.id AS id, properties(name) AS properties, changes
            ORDER BY id LIMIT 100
            """
        )
        refine_rows = [
            row
            for row in rows
            if any(
                change.get("operation") in {"refine", "backfill_refine"}
                for change in row["changes"]
            )
            and not any(
                change.get("operation") == "supersede_exhausted_orphan"
                for change in row["changes"]
            )
        ]
        legacy_rows = [
            row
            for row in rows
            if any(
                change.get("operation") == "supersede_legacy_spelling"
                for change in row["changes"]
            )
            and row not in refine_rows
        ]
        assert len(refine_rows) == 12, len(refine_rows)
        assert len(legacy_rows) == 2, len(legacy_rows)
        result = []
        for row in refine_rows + legacy_rows:
            name_id = row["id"]
            lineage = graph.query(
                """
                MATCH (name:StandardName {id: $name_id})
                OPTIONAL MATCH (name)-[:REFINED_FROM*1..12]->(ancestor:StandardName)
                WITH name, collect(DISTINCT {id: ancestor.id,
                     stage: ancestor.name_stage}) AS ancestors
                OPTIONAL MATCH (descendant:StandardName)-[:REFINED_FROM*1..12]->(name)
                RETURN ancestors,
                       collect(DISTINCT {id: descendant.id,
                            stage: descendant.name_stage}) AS descendants
                """,
                name_id=name_id,
            )[0]
            result.append(
                {
                    "id": name_id,
                    "name_stage": row["properties"].get("name_stage"),
                    "status": row["properties"].get("status"),
                    "description": row["properties"].get("description"),
                    "source_paths": row["properties"].get("source_paths"),
                    "changes": row["changes"],
                    **lineage,
                }
            )
        ids = [row["id"] for row in result]
        cited_by = graph.query(
            """
            MATCH (change:StandardNameChange)
            WHERE change.from_name IN $ids
              AND change.to_name <> change.from_name
            MATCH (owner:StandardName)-[:HAS_INTERNAL_CHANGE]->(change)
            RETURN change.from_name AS predecessor,
                   change.to_name AS to_name, owner.id AS owner,
                   owner.name_stage AS owner_stage, change.operation AS operation,
                   change.reason AS reason
            ORDER BY predecessor, owner LIMIT 100
            """,
            ids=[*ids, "particle_atomic_count"],
        )
        target_ids = sorted(
            {
                change["row_id"].split("=>", 1)[1]
                for row in result
                for change in row["changes"]
                if change.get("operation") == "supersede_legacy_spelling"
                and "=>" in change.get("row_id", "")
            }
        )
        targets = graph.query(
            """
            UNWIND $ids AS id
            OPTIONAL MATCH (name:StandardName {id: id})
            RETURN id, name.name_stage AS name_stage, name.status AS status,
                   count {(name)-[:HAS_SUCCESSOR]->(:StandardName)} AS successors
            ORDER BY id LIMIT 20
            """,
            ids=target_ids,
        )
        control = graph.query(
            """
            MATCH (name:StandardName {id: 'etendue_of_spectrometer_channel'})
            RETURN name.name_stage AS stage, name.source_paths AS source_paths,
                   count {(source:StandardNameSource)-[:PRODUCED_NAME]->(name)
                          WHERE coalesce(source.status, '') <> 'stale'} AS producers,
                   count {(child:StandardName)-[:HAS_PARENT]->(name)
                          WHERE child.name_stage <> 'superseded'} AS children,
                   [(name)-[edge:HAS_SUCCESSOR]->(next:StandardName) |
                     {id: next.id, target_stage: next.name_stage,
                      target_source_paths: next.source_paths,
                      receipt_id: edge.receipt_id}] AS successors
            """
        )
        print(
            json.dumps(
                {
                    "population": population,
                    "rows": result,
                    "cited_by": cited_by,
                    "named_targets": targets,
                    "control": control,
                },
                indent=2,
                default=str,
            )
        )


def authority_row(source: str, target: str) -> dict:
    row_id = f"{source}=>{target}"
    source_participant = {"id": source, "kind": "node", "graph_label": "StandardName"}
    return {
        "id": row_id,
        "identity": {"id": row_id, "kind": "standard_name", "target_id": source},
        "participants": [
            source_participant,
            {"id": target, "kind": "node", "graph_label": "StandardName"},
        ],
        "selection": SELECTION,
        "mutations": [
            {
                "id": f"{source}:supersede",
                "order": 0,
                "kind": "supersede",
                "participant_id": source,
                "arguments": {"successor_id": target},
            }
        ],
        "guards": [
            {
                "id": f"{row_id}:structural-legitimacy",
                "kind": "semantic_authority",
                "implementation": "structural-legitimacy",
                "participant_ids": [source],
            },
            {
                "id": f"{row_id}:out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [source, target],
            },
        ],
        "orphan_policy": "refuse",
    }


def write_authority() -> None:
    rows = [authority_row(source, target) for source, target in sorted(TARGETS.items())]
    rows.append(authority_row(*CONTROL))
    authority = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "recover-recorded-successor-lineage",
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": SELECTION,
        "receipt_policy": {
            "id": "one-per-superseded-identity",
            "operation": "link_signed_successor",
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": signed_payload_sha256(authority),
    }
    AUTHORITY.write_text(json.dumps(authority, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "path": str(AUTHORITY),
                "rows": len(rows),
                "file_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
                "payload_sha256": authority["signature"]["sha256"],
            },
            indent=2,
        )
    )


def inspect_controls() -> None:
    with GraphClient() as graph:
        rows = graph.query(
            """
            MATCH (name:StandardName {name_stage: 'superseded'})
                  -[edge:HAS_SUCCESSOR]->(successor:StandardName)
            RETURN name.id AS id, successor.id AS successor,
                   name.source_paths AS source_paths,
                   successor.source_paths AS successor_source_paths,
                   count {(child:StandardName)-[:HAS_PARENT]->(name)
                          WHERE child.name_stage <> 'superseded'
                            AND coalesce(child.status, '') <> 'superseded'}
                        AS live_children,
                   edge.receipt_id AS receipt_id
            ORDER BY id LIMIT 30
            """
        )
        print(json.dumps(rows, indent=2, default=str))


def signed_call(*, apply: bool, digest: str | None = None) -> None:
    raw = AUTHORITY.read_bytes()
    authority = json.loads(raw)
    receipt = apply_signed_manifest(
        AUTHORITY,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(authority),
        reason="Recover successor links named by recorded lineage evidence",
        apply=apply,
        manifest_sha256=digest,
    )
    print(json.dumps(receipt, indent=2, default=str))


def readback(digest: str, before_excluded: int) -> None:
    with GraphClient() as graph:
        rows = graph.query(
            """
            UNWIND $ids AS id
            MATCH (name:StandardName {id: id})
            OPTIONAL MATCH (name)-[edge:HAS_SUCCESSOR]->(successor:StandardName)
            WITH name, id, collect({target: successor.id,
                                    receipt_id: edge.receipt_id}) AS edges
            OPTIONAL MATCH (name)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange {
              manifest_sha256: $digest, operation: 'link_signed_successor'
            })
            RETURN id, edges, collect(change.id) AS change_ids
            ORDER BY id LIMIT 10
            """,
            ids=[*TARGETS, CONTROL[0]],
            digest=digest,
        )
        counts = graph.query(
            """
            MATCH (name:StandardName {name_stage: 'superseded'})
            RETURN count(name) AS population,
                   count(CASE WHEN EXISTS {(name)-[:HAS_SUCCESSOR]->(:StandardName)}
                         OR EXISTS {(:StandardName)-[:REFINED_FROM]->(name)}
                         THEN 1 END) AS excluded
            """
        )[0]
        assert counts["excluded"] == before_excluded + len(TARGETS), counts
        for row in rows:
            if row["id"] == CONTROL[0]:
                assert len(row["edges"]) == 1, row
                assert row["edges"][0]["target"] == CONTROL[1], row
                assert row["edges"][0]["receipt_id"] is None, row
                assert not row["change_ids"], row
            else:
                assert len(row["edges"]) == len(row["change_ids"]) == 1, row
                assert row["edges"][0]["target"] == TARGETS[row["id"]], row
                assert row["edges"][0]["receipt_id"] == row["change_ids"][0], row
                assert digest in row["change_ids"][0], row
        assert len(rows) == len(TARGETS) + 1, len(rows)
        print(
            json.dumps(
                {"counts": counts, "before_excluded": before_excluded, "rows": rows},
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "inspect",
            "inspect-controls",
            "authority",
            "preview",
            "apply",
            "readback",
            "replay",
        ],
    )
    parser.add_argument("--digest")
    parser.add_argument("--before-excluded", type=int)
    args = parser.parse_args()
    if args.action == "inspect":
        inspect()
    elif args.action == "inspect-controls":
        inspect_controls()
    elif args.action == "authority":
        write_authority()
    elif args.action in {"preview", "replay"}:
        signed_call(apply=False)
    elif args.action == "apply":
        assert args.digest
        signed_call(apply=True, digest=args.digest)
    else:
        assert args.digest and args.before_excluded is not None
        readback(args.digest, args.before_excluded)


if __name__ == "__main__":
    main()

"""Inspect and sign dispositions for sourced superseded names."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

AUTHORITY = Path(__file__).with_name("sib-restore-produced-names-authority.json")

LINKS = {
    "accumulated_helium_4_prefill_count_due_to_gas_injection": "accumulated_helium_4_count_due_to_gas_injection",
    "alpha_angle": "alpha_angle_of_iron_core_segment",
    "density_at_separatrix": "flux_surface_averaged_ion_density_at_plasma_boundary",
    "diamagnetic_heat_current_density_due_to_viscosity": "diamagnetic_current_density_due_to_heat_viscosity",
    "inner_radius_of_passive_structure": "inner_radius_of_passive_loop",
    "ion_power_due_to_recombination": "plasma_power_of_divertor_due_to_recombination",
    "parallel_convection_velocity": "parallel_electron_energy_convection_velocity",
    "parallel_total_ion_energy_diffusivity": "parallel_ion_energy_diffusivity",
    "poloidal_diffusivity": "poloidal_neutral_energy_diffusivity",
    "poloidal_heat_current_density_due_to_viscosity": "poloidal_current_density_due_to_heat_viscosity",
    "poloidal_momentum_diffusivity": "poloidal_neutral_internal_state_momentum_diffusion_coefficient",
    "radial_ion_state_effective_momentum_convection_velocity": "radial_ion_charge_state_momentum_convection_velocity",
    "vertical_line_of_sight": "vertical_coordinate_of_pellet_path_point",
    "x_coordinate_of_ferritic_insert": "x_coordinate_of_ferritic_element_centroid",
    "toroidal_diamagnetic_current_density": "toroidal_current_density_due_to_diamagnetic_drift",
}

RETIRE = (
    "toroidal_current_density_due_to_viscosity",
    "vertical_coordinate_of_diagnostic_component_centre",
    "x1_coordinate_of_diagnostic_aperture",
    "height_of_diagnostic_aperture",
    "major_length_of_coil_conductor_element",
    "normalized_parallel_perturbed_heat_flux",
    "normalized_perpendicular_perturbed_energy",
    "radial_derivative_of_coefficient_of_flux_surface",
    "energy_flux_at_first_wall",
    "ion_state_particle_convection_velocity",
)
SPLIT = RETIRE[:3]
SPLIT_ACCEPTED = {
    "toroidal_current_density_due_to_viscosity": (
        "toroidal_current_density_due_to_heat_viscosity",
        "toroidal_current_density_due_to_parallel_viscosity",
        "toroidal_current_density_due_to_perpendicular_viscosity",
    ),
    "vertical_coordinate_of_diagnostic_component_centre": (
        "vertical_coordinate_of_line_of_sight",
        "vertical_coordinate_of_measurement_position",
        "z_coordinate_of_ferritic_element_centroid",
    ),
    "x1_coordinate_of_diagnostic_aperture": (
        "first_local_tangential_coordinate_of_filter",
        "first_local_tangential_coordinate_of_optical_element",
        "first_local_tangential_coordinate_of_reflectometer_antenna",
    ),
}
SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
OPERATION = "resolve_sourced_supersession"


def authority_row(name_id: str, successor_id: str | None = None) -> dict:
    row_id = f"{name_id}=>{successor_id}" if successor_id else name_id
    participants = [{"id": name_id, "kind": "node", "graph_label": "StandardName"}]
    if successor_id:
        participants.append(
            {"id": successor_id, "kind": "node", "graph_label": "StandardName"}
        )
    mutation = {
        "id": f"{name_id}:supersede",
        "order": 0,
        "kind": "supersede",
        "participant_id": name_id,
    }
    if successor_id:
        mutation["arguments"] = {"successor_id": successor_id}
    return {
        "id": row_id,
        "identity": {"id": row_id, "kind": "standard_name", "target_id": name_id},
        "participants": participants,
        "mutations": [mutation],
        "guards": [
            {
                "id": f"{row_id}:structural-legitimacy",
                "kind": "semantic_authority",
                "implementation": "structural-legitimacy",
                "participant_ids": [name_id],
            },
            {
                "id": f"{row_id}:out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [name_id, successor_id]
                if successor_id
                else [name_id],
            },
        ],
        "orphan_policy": "refuse",
    }


def retirement_reason() -> str:
    return (
        "Record the sourced supersession disposition: link one accepted name "
        "produced by the same source where it exists, or retire a name with "
        "no single successor. Split retirements: "
        + "; ".join(
            f"{name} split into {', '.join(SPLIT_ACCEPTED[name])}" for name in SPLIT
        )
        + ". The other seven retired names have no accepted counterpart."
    )


def build() -> None:
    state = inspect()
    rows = [authority_row(old, new) for old, new in sorted(LINKS.items())]
    rows.extend(authority_row(name) for name in sorted(RETIRE))
    rows.append(authority_row(state["control"][0]["id"]))
    specification = {
        "authority_mode": "external_reviewed",
        "operation_id": "resolve-produced-supersession",
        "orphan_policy": "refuse",
        "receipt_policy": {
            "cardinality": "per_target",
            "id": "one-per-sourced-disposition",
            "operation": OPERATION,
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "rows": rows,
    }
    built = build_repair_authority(specification)
    AUTHORITY.write_bytes(built.content)
    print(
        json.dumps(
            {
                "authority": str(AUTHORITY),
                "rows": len(rows),
                "links": len(LINKS),
                "retirements": len(RETIRE),
                "control": state["control"][0]["id"],
                "file_sha256": built.file_sha256,
                "payload_sha256": built.payload_sha256,
                "reason": retirement_reason(),
            },
            indent=2,
        )
    )


def signed_call(*, apply: bool, digest: str | None = None) -> None:
    raw = AUTHORITY.read_bytes()
    authority = json.loads(raw)
    if apply:
        inspect()
    receipt = apply_signed_manifest(
        AUTHORITY,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(authority),
        reason=retirement_reason(),
        apply=apply,
        manifest_sha256=digest,
        run_id=os.environ.get("REPAIR_RUN_ID"),
    )
    print(json.dumps(receipt, indent=2, default=str))


def readback(digest: str, before_population: int, before_successorless: int) -> None:
    control = json.loads(AUTHORITY.read_bytes())["rows"][-1]["id"]
    ids = sorted(set(LINKS) | set(RETIRE) | {control})
    with GraphClient() as graph:
        counts = graph.query("""
            MATCH (name:StandardName {name_stage: 'superseded'})
            RETURN count(name) AS population,
                   count(CASE WHEN NOT EXISTS {
                     (name)-[:HAS_SUCCESSOR]->(:StandardName)
                   } AND NOT EXISTS {
                     (:StandardName)-[:REFINED_FROM]->(name)
                   } THEN 1 END) AS successorless
        """)[0]
        rows = graph.query(
            """
            UNWIND $ids AS id
            MATCH (name:StandardName {id: id})
            OPTIONAL MATCH (name)-[edge:HAS_SUCCESSOR]->(successor:StandardName)
            WITH name, id, collect(CASE WHEN edge IS NULL THEN null ELSE {
              target: successor.id, receipt_id: edge.receipt_id,
              signed_at: edge.signed_at
            } END) AS edges
            OPTIONAL MATCH (name)-[:HAS_INTERNAL_CHANGE]->
              (change:StandardNameChange {manifest_sha256: $digest})
            WITH name, id, edges, collect(CASE WHEN change IS NULL THEN null ELSE {
              id: change.id, operation: change.operation,
              reason: change.reason, row_id: change.row_id
            } END) AS receipts
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(name)
            RETURN id, name.name_stage AS stage, name.status AS status,
                   edges, receipts, collect(DISTINCT source.id) AS source_ids
            ORDER BY id LIMIT 30
        """,
            ids=ids,
            digest=digest,
        )
    print(
        json.dumps(
            {
                "before": {
                    "population": before_population,
                    "successorless": before_successorless,
                },
                "after": counts,
                "rows": rows,
            },
            indent=2,
            default=str,
        )
    )
    if counts != {
        "population": before_population,
        "successorless": before_successorless - len(LINKS),
    }:
        raise RuntimeError("the population or successor-less delta changed")
    if len(rows) != len(ids):
        raise RuntimeError("readback did not see every authority participant")
    for row in rows:
        name_id = row["id"]
        if name_id == control:
            if row["stage"] != "accepted" or row["receipts"]:
                raise RuntimeError("the refused control row changed")
            continue
        if row["stage"] != "superseded" or row["status"] != "superseded":
            raise RuntimeError(f"{name_id} lost terminal status")
        if not row["source_ids"] or len(row["receipts"]) != 1:
            raise RuntimeError(f"{name_id} lost its source or receipt")
        receipt = row["receipts"][0]
        if receipt["operation"] != OPERATION or digest not in receipt["id"]:
            raise RuntimeError(f"{name_id} has the wrong receipt")
        if name_id in LINKS:
            if (
                len(row["edges"]) != 1
                or row["edges"][0]
                != {
                    "target": LINKS[name_id],
                    "receipt_id": receipt["id"],
                    "signed_at": row["edges"][0]["signed_at"],
                }
                or row["edges"][0]["signed_at"] is None
            ):
                raise RuntimeError(f"{name_id} has the wrong successor edge")
        elif row["edges"]:
            raise RuntimeError(f"{name_id} gained a successor while retiring")


def inspect() -> dict:
    ids = sorted(set(LINKS) | set(RETIRE))
    if len(ids) != 25 or len(LINKS) != 15 or len(RETIRE) != 10:
        raise RuntimeError("the disposition roster has duplicate or missing names")
    with GraphClient() as graph:
        counts = graph.query("""
            MATCH (name:StandardName {name_stage: 'superseded'})
            RETURN count(name) AS population,
                   count(CASE WHEN NOT EXISTS {
                     (name)-[:HAS_SUCCESSOR]->(:StandardName)
                   } AND NOT EXISTS {
                     (:StandardName)-[:REFINED_FROM]->(name)
                   } THEN 1 END) AS successorless
        """)[0]
        cohort = graph.query(
            """
            UNWIND $ids AS id
            MATCH (name:StandardName {id: id})
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(name)
            WITH name, id, collect(DISTINCT source.id) AS sources
            OPTIONAL MATCH (name)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
            RETURN id, name.name_stage AS stage, name.status AS status,
                   name.superseded_from_stage AS prior_stage, sources,
                   collect(DISTINCT change.operation) AS operations,
                   COUNT { (name)-[:HAS_SUCCESSOR]->(:StandardName) } AS successors,
                   COUNT { (:StandardName)-[:REFINED_FROM]->(name) } AS refinements
            ORDER BY id LIMIT 30
        """,
            ids=ids,
        )
        pairs = graph.query(
            """
            UNWIND $pairs AS pair
            MATCH (old:StandardName {id: pair.old})
            MATCH (new:StandardName {id: pair.new})
            OPTIONAL MATCH (shared:StandardNameSource)-[:PRODUCED_NAME]->(old)
            WHERE (shared)-[:PRODUCED_NAME]->(new)
            RETURN pair.old AS old, pair.new AS new,
                   new.name_stage AS new_stage, new.status AS new_status,
                   collect(DISTINCT shared.id) AS shared_source_ids
            ORDER BY old LIMIT 20
        """,
            pairs=[{"old": old, "new": new} for old, new in LINKS.items()],
        )
        retirements = graph.query(
            """
            UNWIND $ids AS id
            MATCH (old:StandardName {id: id})
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(old)
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->
                           (accepted:StandardName {name_stage: 'accepted'})
            RETURN id, collect(DISTINCT accepted.id) AS shared_accepted_names
            ORDER BY id LIMIT 12
        """,
            ids=list(RETIRE),
        )
        control = graph.query("""
            MATCH (name:StandardName {name_stage: 'accepted'})
            WHERE COUNT {
              (child:StandardName)-[:HAS_PARENT]->(name)
              WHERE child.name_stage <> 'superseded'
                AND coalesce(child.status, '') <> 'superseded'
            } > 0
            RETURN name.id AS id, name.name_stage AS stage,
                   COUNT {
                     (child:StandardName)-[:HAS_PARENT]->(name)
                     WHERE child.name_stage <> 'superseded'
                       AND coalesce(child.status, '') <> 'superseded'
                   } AS live_children
            ORDER BY id LIMIT 1
        """)
    result = {
        "counts": counts,
        "cohort": cohort,
        "pairs": pairs,
        "retirements": retirements,
        "control": control,
    }
    print(json.dumps(result, indent=2, default=str))
    if len(cohort) != 25 or {row["id"] for row in cohort} != set(ids):
        raise RuntimeError("the sourced cohort no longer matches the ruling")
    if any(
        row["stage"] != "superseded"
        or not row["sources"]
        or row["successors"] != 0
        or row["refinements"] != 0
        or set(row["operations"]) - {"realign_grammar_segments"}
        for row in cohort
    ):
        raise RuntimeError("a source identity gained lineage or changed stage")
    if len(pairs) != 15 or any(
        row["new_stage"] != "accepted" or not row["shared_source_ids"] for row in pairs
    ):
        raise RuntimeError("an accepted successor no longer shares a producer")
    if (
        len(retirements) != 10
        or any(
            set(row["shared_accepted_names"]) != set(SPLIT_ACCEPTED[row["id"]])
            for row in retirements
            if row["id"] in SPLIT
        )
        or any(
            row["shared_accepted_names"]
            for row in retirements
            if row["id"] not in SPLIT
        )
    ):
        raise RuntimeError("the accepted split counterparts have changed")
    if len(control) != 1 or control[0]["live_children"] < 1:
        raise RuntimeError("the refusal control lacks a live child")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["inspect", "build", "preview", "apply", "readback", "replay"]
    )
    parser.add_argument("--digest")
    parser.add_argument("--before-population", type=int)
    parser.add_argument("--before-successorless", type=int)
    args = parser.parse_args()
    if args.action == "inspect":
        inspect()
    elif args.action == "build":
        build()
    elif args.action in {"preview", "replay"}:
        signed_call(apply=False)
    elif args.action == "apply":
        if not args.digest:
            raise SystemExit("apply requires --digest")
        signed_call(apply=True, digest=args.digest)
    elif args.action == "readback":
        if (
            not args.digest
            or args.before_population is None
            or args.before_successorless is None
        ):
            raise SystemExit("readback requires digest and baseline counts")
        readback(args.digest, args.before_population, args.before_successorless)


if __name__ == "__main__":
    main()

"""Apply reviewed DD source moves and a signed successor supersession.

Each phase leaves a complete log in the run directory. The authority phase
must be committed before the matching preview and apply phases.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import signed_manifest as signed
from imas_codex.standard_names.repair_authority import build_repair_authority

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
AUTHORITY = HERE / "sib-rebind-removed-path-names-authority.json"
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261106T222618682261-sib-rebind-removed-path-names"
)
NAMES = {
    "radial_coordinate_of_flux_surface",
    "radial_coordinate_of_launching_position",
    "toroidal_current_density_due_to_distribution_function_driven",
    "deuterium_tritium_flux",
    "flux_surface_average_magnetic_field_magnitude",
    "flux_surface_averaged_magnetic_field_magnitude",
}
EXCHANGES = [
    (
        "radial_coordinate_of_flux_surface",
        "gyrokinetics/normalizing_quantities/r",
        "gyrokinetics_local/normalizing_quantities/r",
    ),
    (
        "radial_coordinate_of_launching_position",
        "ec_launchers/launcher/launching_position/r",
        "ec_launchers/beam/launching_position/r",
    ),
    (
        "toroidal_current_density_due_to_distribution_function_driven",
        "distributions/distribution/profiles_1d/current_tor",
        "distributions/distribution/profiles_1d/current_phi",
    ),
    (
        "deuterium_tritium_flux",
        "summary/fusion/neutron_fluxes/dt/total/value",
        "summary/fusion/neutron_rates/dt/total/value",
    ),
    (
        "toroidal_current_density_due_to_distribution_function_driven",
        "distributions/distribution/profiles_2d/current_tor",
        "distributions/distribution/profiles_2d/current_phi",
    ),
]
OLD = "flux_surface_average_magnetic_field_magnitude"
SUCCESSOR = "flux_surface_averaged_magnetic_field_magnitude"
REASON = (
    "Move each reviewed name's producing DD source to its documented successor path"
)


def snapshot(gc: GraphClient) -> dict:
    rows = gc.query(
        """
        MATCH (n:StandardName)
        OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n)
        OPTIONAL MATCH (n)-[link:REFINED_FROM|HAS_SUCCESSOR]-(peer:StandardName)
        RETURN n.id AS id, properties(n) AS properties,
               collect(DISTINCT s.id) AS sources,
               collect(DISTINCT {type:type(link), peer:peer.id}) AS links
        ORDER BY id
        """
    )
    result = {
        row["id"]: json.loads(json.dumps(row, sort_keys=True, default=str))
        for row in rows
    }
    if not NAMES <= result.keys() or len(result) < 1000:
        raise RuntimeError("standard-name population read is incomplete")
    return result


def targeted(gc: GraphClient) -> list[dict]:
    return gc.query(
        """
        UNWIND $names AS id
        MATCH (n:StandardName {id:id})
        OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n)
        OPTIONAL MATCH (s)-[:FROM_DD_PATH]->(dd:IMASNode)
        RETURN id, n.name_stage AS stage, n.status AS status,
               n.source_paths AS paths,
               collect(DISTINCT {source:s.id, status:s.status,
                                 dd:dd.id, dd_status:dd.lifecycle_status}) AS producers
        ORDER BY id
        """,
        names=sorted(NAMES),
    )


def blocking(gc: GraphClient) -> list[str]:
    rows = gc.query(
        """
        MATCH (n:StandardName)
        WHERE n.name_stage IN ['accepted', 'approved']
        OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n)
        WITH n, collect(s.status) AS statuses
        WHERE size(statuses) = 0 OR all(status IN statuses WHERE status = 'stale')
        RETURN n.id AS id ORDER BY id
        """
    )
    return [row["id"] for row in rows]


def authority(phase: str) -> dict:
    if phase in {"exchange_first", "exchange_second"}:
        selected = EXCHANGES[:4] if phase == "exchange_first" else EXCHANGES[4:]
        rows = [
            {
                "id": name,
                "name_id": name,
                "given_source_id": "dd:" + old,
                "taken_source_id": "dd:" + new,
                "receiving_name_id": None,
            }
            for name, old, new in selected
        ]
        spec = {
            "operation_id": signed._SOURCE_EXCHANGE_OPERATION,
            "authority_mode": "external_reviewed",
            "rows": rows,
            "receipt_policy": {
                "id": "one-per-logical-change",
                "operation": "exchange_dd_producing_source",
                "cardinality": "per_target",
                "link_participant_kind": "standard_name",
                "replay_projection": ["manifest_sha256", "row_id"],
            },
            "orphan_policy": "refuse",
        }
    elif phase == "supersede":
        row_id = OLD + "=>" + SUCCESSOR
        spec = {
            "operation_id": "retire-duplicate-magnetic-field-average",
            "authority_mode": "external_reviewed",
            "rows": [
                {
                    "id": row_id,
                    "identity": {
                        "id": row_id,
                        "kind": "standard_name",
                        "target_id": OLD,
                    },
                    "participants": [
                        {"id": OLD, "kind": "node", "graph_label": "StandardName"},
                        {
                            "id": SUCCESSOR,
                            "kind": "node",
                            "graph_label": "StandardName",
                        },
                    ],
                    "mutations": [
                        {
                            "id": OLD + ":supersede",
                            "order": 0,
                            "kind": "supersede",
                            "participant_id": OLD,
                            "arguments": {"successor_id": SUCCESSOR},
                        }
                    ],
                    "guards": [
                        {
                            "id": row_id + ":structural-legitimacy",
                            "kind": "semantic_authority",
                            "implementation": "structural-legitimacy",
                            "participant_ids": [OLD],
                        },
                        {
                            "id": row_id + ":out-of-allowlist-immutability",
                            "kind": "collateral_immutability",
                            "implementation": "out-of-allowlist-immutability",
                            "participant_ids": [OLD, SUCCESSOR],
                        },
                    ],
                    "orphan_policy": "refuse",
                }
            ],
            "receipt_policy": {
                "id": "one-per-retired-identity",
                "operation": "link_signed_successor",
                "cardinality": "per_target",
                "link_participant_kind": "standard_name",
                "replay_projection": ["manifest_sha256", "row_id"],
            },
            "orphan_policy": "refuse",
        }
    else:
        raise ValueError(phase)
    built = build_repair_authority(spec)
    AUTHORITY.write_bytes(built.content)
    return {
        "phase": phase,
        "path": str(AUTHORITY),
        "file_sha256": built.file_sha256,
        "payload_sha256": built.payload_sha256,
        "rows": built.artifact.repair_rows,
    }


def signed_call(phase: str, *, apply: bool, manifest_sha256: str | None = None) -> dict:
    raw = AUTHORITY.read_bytes()
    document = json.loads(raw)
    if phase in {"exchange_first", "exchange_second"}:
        adapter = signed._SOURCE_EXCHANGE_ADAPTER
        kwargs = {
            "authority_adapter": adapter,
            "guard_set": signed._SOURCE_EXCHANGE_GUARDS,
        }
    elif phase == "supersede":
        kwargs = {}
    else:
        raise ValueError(phase)
    with GraphClient() as gc:
        result = signed.apply_signed_manifest(
            AUTHORITY,
            authority_file_sha256=hashlib.sha256(raw).hexdigest(),
            authority_payload_sha256=signed.signed_payload_sha256(document),
            reason=REASON
            if phase != "supersede"
            else "Retire the duplicate spelling in favour of the accepted identity for averaged magnetic-field magnitude",
            apply=apply,
            manifest_sha256=manifest_sha256,
            run_id=RUN.name,
            gc=gc,
            **kwargs,
        )
    return result


def operate(action: str, phase: str | None) -> dict:
    if action == "baseline":
        with GraphClient() as gc:
            result = {"names": targeted(gc), "blocking": blocking(gc)}
            (RUN / "before-names.json").write_text(
                json.dumps(snapshot(gc), sort_keys=True, default=str)
            )
        (RUN / "before.json").write_text(json.dumps(result, indent=2, default=str))
        return result
    if action == "authority":
        assert phase is not None
        return authority(phase)
    if action == "control":
        specification = json.loads(AUTHORITY.read_text())
        specification.pop("signature")
        specification["rows"][0]["taken_source_id"] = (
            "dd:equilibrium/time_slice/profiles_1d/b_field_average"
        )
        built = build_repair_authority(specification)
        control_path = RUN / "held-source-control-authority.json"
        control_path.write_bytes(built.content)
        with GraphClient() as gc:
            result = signed.apply_signed_manifest(
                control_path,
                authority_adapter=signed._SOURCE_EXCHANGE_ADAPTER,
                authority_file_sha256=built.file_sha256,
                authority_payload_sha256=built.payload_sha256,
                guard_set=signed._SOURCE_EXCHANGE_GUARDS,
                reason="Prove that a source held by another name is refused",
                apply=False,
                gc=gc,
            )
        if (
            result.get("outcome") != "refused"
            or result.get("changed") != 0
            or not any(
                item.get("reason") == "taken source already has a live owner"
                for item in result.get("refusals", [])
            )
        ):
            raise RuntimeError(f"held-source control did not refuse: {result}")
        (RUN / "held-source-control.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        return result
    if action == "preview":
        assert phase is not None
        result = signed_call(phase, apply=False)
        (RUN / f"{phase}-preview.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        return result
    if action == "apply":
        assert phase is not None
        preview = json.loads((RUN / f"{phase}-preview.json").read_text())
        if preview.get("outcome") not in {"would_apply", "would_supersede"}:
            raise RuntimeError(f"preview refused: {preview.get('refusals')}")
        result = signed_call(
            phase, apply=True, manifest_sha256=preview["manifest_sha256"]
        )
        (RUN / f"{phase}-receipt.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        return result
    if action == "readback":
        with GraphClient() as gc:
            after = snapshot(gc)
            before = json.loads((RUN / "before-names.json").read_text())
            other_changed = sorted(
                id
                for id in before.keys() | after.keys()
                if id not in NAMES and before.get(id) != after.get(id)
            )
            result = {
                "names": targeted(gc),
                "blocking": blocking(gc),
                "other_names_changed": other_changed,
                "names_before": len(before),
                "names_after": len(after),
            }
        (RUN / "after.json").write_text(json.dumps(result, indent=2, default=str))
        return result
    raise ValueError(action)


def main() -> None:
    action = sys.argv[1]
    phase = sys.argv[2] if len(sys.argv) > 2 else None
    logs = RUN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    label = action + ("-" + phase if phase else "")
    log = logs / f"{label}.log"
    with log.open("w") as stream:
        import subprocess

        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        stream.write(f"# revision: {revision}\n# tree: {ROOT}\n")
        stream.write(f"# command: {' '.join(sys.argv)}\n")
        stream.flush()
        try:
            result = operate(action, phase)
            stream.write(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
            )
            stream.write("EXIT=0\n")
        except Exception:
            stream.write(traceback.format_exc())
            stream.write("EXIT=1\n")
            raise
    print(log)


if __name__ == "__main__":
    main()

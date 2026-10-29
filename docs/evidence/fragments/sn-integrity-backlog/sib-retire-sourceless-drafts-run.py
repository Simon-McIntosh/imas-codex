"""Retire the live drafts that carry no producing DD source.

Population: every live (non-superseded) standard name that
``find_sourceless_live_names`` reports, excluding derived parents (the function
exempts those) and the six names held from release in ``export.py``.

For each name the runner looks for the live name that now holds its published
catalog path or its recorded source.  A holder must itself carry a producing
(non-stale) source on that path, so a path that only a sourceless sibling names
has no holder.  Where a holder exists the retirement carries a signed successor
edge; where none does it carries the retirement receipt alone.

Modes: read | authority | preview | apply --digest | readback | negative.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.source_invariant import find_sourceless_live_names
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

HERE = Path(__file__).resolve().parent
AUTHORITY = HERE / "sib-retire-sourceless-drafts-authority.json"
POPULATION = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261029T083805766546-sib-retire-sourceless-drafts/population.json"
)

HELD = {
    "coolant_absorbed_energy_accumulated_of_plasma_facing_component",
    "count_normalized_at_detector_pixel",
    "electrostatic_potential_at_outboard_midplane",
    "electrostatic_potential_at_wall",
    "root_mean_square_spectral_width_of_spectrometer_channel",
    "volume_averaged_electron_number_density_over_scrape_off_layer",
}

SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
RETIRE_OPERATION = "supersede_exhausted_orphan"
OPERATION_ID = "retire-sourceless-live-drafts-with-recorded-successor"
REASON = (
    "the draft carries no producing DD source; retire it with a signed receipt "
    "and a successor edge to the live name holding its recorded source"
)

LIVE = """
MATCH (n:StandardName)
WHERE coalesce(n.name_stage,'') <> 'superseded'
  AND coalesce(n.status,'') <> 'superseded'
RETURN n.id AS id ORDER BY id
"""

RECORDED = """
UNWIND $ids AS id
MATCH (n:StandardName {id: id})
OPTIONAL MATCH (n)<-[:PRODUCED_NAME]-(s:StandardNameSource)
WITH n, collect(DISTINCT s.source_id) AS source_ids
RETURN n.id AS id, n.name_stage AS stage,
       coalesce(n.source_paths, []) AS source_paths, source_ids
"""

HOLDERS = """
UNWIND $paths AS path
MATCH (h:StandardName)<-[:PRODUCED_NAME]-(s:StandardNameSource {source_id: path})
WHERE coalesce(s.status,'') <> 'stale'
  AND coalesce(h.name_stage,'') <> 'superseded'
  AND coalesce(h.status,'') <> 'superseded'
RETURN path, h.id AS holder, h.name_stage AS holder_stage ORDER BY path, holder
"""


def _bare(path: str) -> str:
    return path[3:] if path.startswith("dd:") else path


def read_population() -> dict:
    with GraphClient() as graph:
        live = [r["id"] for r in graph.query(LIVE)]
        sourceless = find_sourceless_live_names(graph, live)
        recorded = graph.query(RECORDED, ids=sourceless)
        paths = sorted(
            {
                _bare(p)
                for row in recorded
                for p in list(row["source_paths"]) + list(row["source_ids"] or [])
                if p
            }
        )
        holders = graph.query(HOLDERS, paths=paths)
    holder_map: dict[str, list[str]] = {}
    for row in holders:
        holder_map.setdefault(row["path"], []).append(row["holder"])

    rows = []
    for row in recorded:
        name = row["id"]
        paths = sorted(
            {
                _bare(p)
                for p in list(row["source_paths"]) + list(row["source_ids"] or [])
                if p
            }
        )
        held_by = sorted(
            {
                holder
                for path in paths
                for holder in holder_map.get(path, [])
                if holder != name and holder not in sourceless
            }
        )
        rows.append(
            {
                "id": name,
                "stage": row["stage"],
                "recorded_paths": paths,
                "recorded_sources": [s for s in (row["source_ids"] or []) if s],
                "holders": held_by,
                "held_from_release": name in HELD,
                "in_scope": row["stage"] == "exhausted" and name not in HELD,
            }
        )
    summary = {
        "live_names": len(live),
        "sourceless_live": len(sourceless),
        "by_stage": {
            stage: sum(1 for r in rows if r["stage"] == stage)
            for stage in sorted({r["stage"] for r in rows})
        },
        "exhausted_in_scope": sum(1 for r in rows if r["in_scope"]),
        "with_holder": sum(1 for r in rows if r["in_scope"] and r["holders"]),
        "held_from_release": sorted(r["id"] for r in rows if r["held_from_release"]),
    }
    result = {"summary": summary, "rows": rows}
    POPULATION.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def _row(name: str, successor: str | None) -> dict:
    mutation = {
        "id": f"{name}:supersede",
        "order": 0,
        "kind": "supersede",
        "participant_id": name,
    }
    if successor:
        mutation["arguments"] = {"successor_id": successor}
    participants = [{"id": name, "kind": "node", "graph_label": "StandardName"}]
    if successor:
        participants.append(
            {"id": successor, "kind": "node", "graph_label": "StandardName"}
        )
    return {
        "id": name,
        "identity": {"id": name, "kind": "standard_name", "target_id": name},
        "participants": participants,
        "mutations": [mutation],
        "guards": [
            {
                "id": f"{name}:structural-legitimacy",
                "kind": "semantic_authority",
                "implementation": "structural-legitimacy",
                "participant_ids": [name],
            },
            {
                "id": f"{name}:out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [p["id"] for p in participants],
            },
        ],
        "orphan_policy": "refuse",
        "selection": dict(SELECTION),
    }


def build_authority() -> dict:
    population = json.dumps(
        json.loads(POPULATION.read_text()), sort_keys=True
    ) if POPULATION.exists() else None
    if population is None:
        raise SystemExit("run the read mode first")
    rows = json.loads(POPULATION.read_text())["rows"]
    scoped = [r for r in rows if r["in_scope"]]
    built_rows = [
        _row(r["id"], r["holders"][0] if r["holders"] else None) for r in scoped
    ]
    with GraphClient() as graph:
        control = graph.query(
            "MATCH (n:StandardName {name_stage:'accepted'}) "
            "WHERE COUNT { (c:StandardName)-[:HAS_PARENT]->(n) } >= 1 "
            "RETURN n.id AS id ORDER BY id LIMIT 1"
        )[0]["id"]
    built_rows.append(_row(control, None))
    spec = {
        "operation_id": OPERATION_ID,
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
        "rows": built_rows,
    }
    built = build_repair_authority(spec)
    AUTHORITY.write_bytes(built.content)
    return {
        "authority": str(AUTHORITY),
        "rows": len(spec["rows"]),
        "with_successor": sum(1 for r in scoped if r["holders"]),
        "control": control,
        "authority_file_sha256": built.file_sha256,
    }


def _call(apply: bool, digest: str | None) -> dict:
    raw = AUTHORITY.read_bytes()
    authority = json.loads(raw)
    receipt = apply_signed_manifest(
        AUTHORITY,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(authority),
        reason=REASON,
        apply=apply,
        manifest_sha256=digest,
    )
    return {k: v for k, v in receipt.items() if k not in {"manifest", "admitted_actions"}}


def readback() -> dict:
    population = json.loads(POPULATION.read_text())
    expected = sorted(r["id"] for r in population["rows"] if r["in_scope"])
    raw = AUTHORITY.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    with GraphClient() as graph:
        receipts = graph.query(
            "MATCH (c:StandardNameChange {authority_file_sha256:$fs}) "
            "RETURN c.row_id AS row_id, c.operation AS op, c.manifest_sha256 AS digest",
            fs=file_sha,
        )
        stage = graph.query(
            "UNWIND $ids AS id MATCH (n:StandardName {id:id}) "
            "RETURN count(n) AS present, "
            "count(CASE WHEN n.name_stage='superseded' THEN 1 END) AS superseded",
            ids=expected,
        )[0]
        successors = graph.query(
            "UNWIND $ids AS id MATCH (n:StandardName {id:id}) "
            "UNWIND (CASE WHEN (n)-[:HAS_SUCCESSOR]->(:StandardName) "
            "THEN [(n)-[e:HAS_SUCCESSOR]->(s) | {id:id, successor:s.id, receipt:e.receipt_id}] "
            "ELSE [null] END) AS row "
            "WITH row WHERE row IS NOT NULL RETURN row ORDER BY row.id",
            ids=expected,
        )
        live = [r["id"] for r in graph.query(LIVE)]
        remaining = find_sourceless_live_names(graph, live)
    per: dict[str, int] = {}
    for row in receipts:
        per[row["row_id"]] = per.get(row["row_id"], 0) + 1
    return {
        "authority_file_sha256": file_sha,
        "expected": len(expected),
        "present": int(stage["present"]),
        "still_superseded": int(stage["superseded"]),
        "receipt_names": len(per),
        "receipts_per_name_min": min(per.values()) if per else 0,
        "receipts_per_name_max": max(per.values()) if per else 0,
        "names_without_receipt": sorted(set(expected) - set(per)),
        "successor_edges": successors,
        "manifest_sha256": sorted({str(r["digest"]) for r in receipts}),
        "remaining_sourceless_live": len(remaining),
        "remaining_names": remaining,
    }


def negative() -> dict:
    """A preview whose authority payload no longer matches its digest must refuse."""
    raw = AUTHORITY.read_bytes()
    authority = json.loads(raw)
    tampered = dict(authority)
    tampered["rows"] = authority["rows"][:-1]
    try:
        apply_signed_manifest(
            AUTHORITY,
            authority_file_sha256=hashlib.sha256(raw).hexdigest(),
            authority_payload_sha256=signed_payload_sha256(tampered),
            reason=REASON,
            apply=False,
        )
    except Exception as exc:  # the refusal is the evidence
        return {"refused": True, "error": f"{type(exc).__name__}: {exc}"}
    return {"refused": False, "error": None}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "read"
    if mode == "read":
        print(json.dumps(read_population()["summary"], indent=1))
    elif mode == "authority":
        print(json.dumps(build_authority(), indent=1))
    elif mode in {"preview", "replay"}:
        print(json.dumps(_call(False, None), indent=1, default=str)[:6000])
    elif mode == "apply":
        print(json.dumps(_call(True, sys.argv[2]), indent=1, default=str)[:6000])
    elif mode == "readback":
        result = readback()
        result.pop("remaining_names", None)
        print(json.dumps(result, indent=1, default=str)[:8000])
    elif mode == "negative":
        print(json.dumps(negative(), indent=1))
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
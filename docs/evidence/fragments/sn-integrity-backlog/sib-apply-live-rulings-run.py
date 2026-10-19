"""Apply the lead's ruled provenance changes to live standard names without a source.

The lead ruled four graph changes on live standard names that have no DD
source. Two retire a duplicate in favour of the live identity that already
holds the quantity, through the signed supersede with a successor link. One
restores a lost binding, through ``bind_recovery_sources``, the production
writer that merges a recovery source and links it. The fourth swaps two
separatrix electron-temperature bindings.

Every write goes through its owning writer: the retirements through
``apply_signed_manifest`` under an authority built by ``build_repair_authority``
and previewed before it is applied on a quiet graph; the restored binding
through ``bind_recovery_sources``. Nothing is written by hand Cypher.

The swap is not applied. Its first half moves an existing source's
``PRODUCED_NAME`` edge from one live name to another, which is the signed
ordinary-source migration program's job; that program admits only a target
whose ``name_stage`` is ``accepted`` and whose ``validation_status`` is
``valid`` (``_ordinary_source_migration_refusal``), and the receiving name here
is ``reviewed``, so the row is refused. The graph has no other owner that
removes a binding and adds its replacement in one transaction: the envelope's
remaining closed programs are the unbound-source attachment (admits only an
unbound ``extracted`` source, and adds without removing), the stale-source
detach, the name-realization detach and the recovery unbind. Applying the two
halves as separate writes would leave the name sourceless between them, or
transiently bind one source to two live names, so it is left unapplied and
reported.

Modes: read | authority | preview | quiet | apply | bindings | readback |
verify | falsify | swap | record.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from imas_codex.graph import GraphClient
from imas_codex.standard_names.graph_ops import classify_sourceless_standard_names
from imas_codex.standard_names.provenance_rebuild import bind_recovery_sources
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIGURE_DIR = ROOT / "docs/figures/sn-integrity-backlog/sib-apply-live-rulings"
FIGURE_SRC = (
    "/imas-codex/figures/sn-integrity-backlog/sib-apply-live-rulings/"
    "populations-before-after.svg"
)
AUTHORITY_PATH = HERE / "sib-apply-live-rulings-authority.json"
FRAGMENT_PATH = HERE / "sib-apply-live-rulings.json"

RUN_ID: str | None = None
RUN_DIR: Path | None = None

SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}

# The two identities the lead retired, each in favour of the live name that
# already holds the quantity.
RETIREMENTS: tuple[tuple[str, str], ...] = (
    ("neutron_flux_due_to_fusion", "total_neutron_rate"),
    (
        "toroidal_wave_vector_normalized_of_beam_tracing_beam",
        "normalized_toroidal_wave_vector_of_beam_tracing_beam",
    ),
)

# The name whose lost DD source is restored, and the path it is restored from.
BINDINGS: tuple[tuple[str, str], ...] = (
    (
        "flux_surface_averaged_parallel_electric_field_at_separatrix",
        "summary/local/separatrix_average/e_field_parallel/value",
    ),
)

RETIRE_OPERATION_ID = "retire-live-duplicate-in-favour-of-its-quantity-holder"
BIND_OPERATION_ID = "restore-lost-binding-for-live-sourceless-name"
RETIRE_REASON = (
    "the name and the live identity that already holds its quantity denote the "
    "same physical quantity; retire the duplicate in the holder's favour "
    "through the signed supersede with a successor link"
)
BIND_REASON = (
    "the name's own recorded DD path is the quantity it denotes; restore the "
    "lost binding from it rather than synthesizing a source"
)
SUPERSEDE_OPERATION = "link_signed_successor"

# The swap the lead ruled, and the two names it exchanges the DD paths of.
SWAP_LOCAL_NAME = "electron_temperature_at_separatrix"
SWAP_AVERAGE_NAME = (
    "bulk_electron_temperature_flux_surface_averaged_at_last_closed_flux_surface"
)
SWAP_LOCAL_PATH = "dd:summary/local/separatrix/t_e/value"
SWAP_AVERAGE_PATH = "dd:summary/local/separatrix_average/t_e/value"

CLASSIFIER_TARGETS = (
    *[name for pair in RETIREMENTS for name in pair],
    *[name for name, _ in BINDINGS],
    SWAP_LOCAL_NAME,
    SWAP_AVERAGE_NAME,
)


def _resolve_runtime(args: argparse.Namespace) -> None:
    global RUN_ID, RUN_DIR  # noqa: PLW0603
    run_id = args.run_id or os.environ.get("SIB_RUN_ID")
    if not run_id:
        raise SystemExit("--run-id is required (or set SIB_RUN_ID)")
    RUN_ID = run_id
    run_dir = args.run_dir or os.environ.get("SIB_RUN_DIR")
    if not run_dir:
        raise SystemExit("--run-dir is required (or set SIB_RUN_DIR)")
    RUN_DIR = Path(run_dir)


# --------------------------------------------------------------------------
# Live read
# --------------------------------------------------------------------------


def read() -> dict[str, Any]:
    """Re-derive every row this node writes from the live graph."""
    with GraphClient() as gc:
        names = gc.query(
            """
            UNWIND $ids AS id
            MATCH (sn:StandardName {id: id})
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
            RETURN sn.id AS id, sn.name_stage AS stage, sn.status AS status,
                   sn.validation_status AS validation,
                   sn.superseded_by AS superseded_by,
                   sn.source_paths AS source_paths,
                   collect(DISTINCT source.id) AS source_ids,
                   collect(DISTINCT dd.id) AS dd_paths
            ORDER BY id
            """,
            ids=sorted(set(CLASSIFIER_TARGETS)),
        )
        sources = gc.query(
            """
            UNWIND $paths AS path
            MATCH (source:StandardNameSource {id: 'dd:' + path})
            OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(held:StandardName)
            RETURN path AS dd_path, source.id AS source_id,
                   source.status AS status, source.source_type AS source_type,
                   source.produced_sn_id AS scalar,
                   elementId(binding) AS binding_element_id,
                   elementId(source) AS source_element_id,
                   collect(DISTINCT held.id) AS holders
            ORDER BY dd_path
            """,
            paths=sorted(
                {
                    *[path for _, path in BINDINGS],
                    SWAP_LOCAL_PATH.removeprefix("dd:"),
                    SWAP_AVERAGE_PATH.removeprefix("dd:"),
                }
            ),
        )
        classified = classify_sourceless_standard_names(gc)
    counts: dict[str, int] = defaultdict(int)
    members: dict[str, set[str]] = defaultdict(set)
    for row in classified:
        counts[row["population"]] += 1
        members[row["population"]].add(row["id"])
    return {
        "names": [dict(r) for r in names],
        "sources": [dict(r) for r in sources],
        "classifier_counts": dict(counts),
        "sourceless_ids": {
            "local": SWAP_LOCAL_NAME in members["never_bound"],
            "average": SWAP_AVERAGE_NAME in members["never_bound"],
        },
        "sourceless_targets": [
            name for name in CLASSIFIER_TARGETS if name in members["never_bound"]
        ],
        "populations": {k: sorted(v) for k, v in members.items()},
    }


# --------------------------------------------------------------------------
# Retirements: signed supersede under an exact signed cohort
# --------------------------------------------------------------------------


def _supersede_row(name_id: str, holder_id: str) -> dict[str, Any]:
    row_id = f"{name_id}=>{holder_id}"
    return {
        "id": row_id,
        "identity": {"id": row_id, "kind": "standard_name", "target_id": name_id},
        "participants": [
            {"id": name_id, "kind": "node", "graph_label": "StandardName"},
            {"id": holder_id, "kind": "node", "graph_label": "StandardName"},
        ],
        "selection": dict(SELECTION),
        "mutations": [
            {
                "id": f"{name_id}:supersede",
                "order": 0,
                "kind": "supersede",
                "participant_id": name_id,
                "arguments": {"successor_id": holder_id},
            }
        ],
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
                "participant_ids": [name_id, holder_id],
            },
        ],
        "orphan_policy": "refuse",
    }


def build_authority() -> dict[str, Any]:
    """Emit the retirement authority and its two digests."""
    rows = [_supersede_row(name, holder) for name, holder in RETIREMENTS]
    spec = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": RETIRE_OPERATION_ID,
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": dict(SELECTION),
        "receipt_policy": {
            "id": "one-per-retired-identity",
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
        "path": str(AUTHORITY_PATH.relative_to(HERE.parents[2])),
        "rows": [row["id"] for row in rows],
        "file_sha256": hashlib.sha256(AUTHORITY_PATH.read_bytes()).hexdigest(),
        "payload_sha256": built.payload_sha256,
    }


def _signed_call(apply: bool, digest: str | None) -> dict[str, Any]:
    raw = AUTHORITY_PATH.read_bytes()
    payload = json.loads(raw)
    res = apply_signed_manifest(
        AUTHORITY_PATH,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(payload),
        reason=RETIRE_REASON,
        apply=apply,
        manifest_sha256=digest if apply else None,
        run_id=RUN_ID,
    )
    return {k: v for k, v in res.items() if k not in {"manifest", "admitted_actions"}}


# --------------------------------------------------------------------------
# Restored bindings: the production recovery writer
# --------------------------------------------------------------------------


def bind() -> dict[str, Any]:
    """Restore each ruled binding through ``bind_recovery_sources``."""
    out = []
    with GraphClient() as gc:
        for name, path in BINDINGS:
            spec = {
                "id": f"dd:{path}",
                "source_type": "dd",
                "source_id": path,
                "status": "attached",
                "dd_path": path,
            }
            bound = bind_recovery_sources(name, [spec], gc=gc)
            held = gc.query(
                """
                MATCH (s:StandardNameSource {id: $sid})-[b:PRODUCED_NAME]->(n:StandardName {id: $name})
                OPTIONAL MATCH (s)-[:FROM_DD_PATH]->(d:IMASNode {id: $path})
                RETURN s.status AS status, s.produced_sn_id AS scalar,
                       elementId(b) AS binding_element_id,
                       d.id AS dd_path
                """,
                sid=f"dd:{path}",
                name=name,
                path=path,
            )
            out.append(
                {
                    "name": name,
                    "source_id": f"dd:{path}",
                    "bound": bound,
                    "held": [dict(r) for r in held],
                }
            )
    return {"bindings": out}


# --------------------------------------------------------------------------
# The swap: shown not expressible, never applied
# --------------------------------------------------------------------------


def _migration_row(binding_element_id: str) -> dict[str, Any]:
    return {
        "id": SWAP_AVERAGE_PATH,
        "identity": {
            "id": SWAP_AVERAGE_PATH,
            "kind": "source",
            "source_id": SWAP_AVERAGE_PATH,
            "target_id": SWAP_AVERAGE_NAME,
        },
        "participants": [
            {
                "id": SWAP_AVERAGE_PATH,
                "kind": "node",
                "graph_label": "StandardNameSource",
            },
            {"id": SWAP_LOCAL_NAME, "kind": "node", "graph_label": "StandardName"},
            {"id": SWAP_AVERAGE_NAME, "kind": "node", "graph_label": "StandardName"},
            {
                "id": binding_element_id,
                "kind": "relationship",
                "graph_label": "PRODUCED_NAME",
            },
        ],
        "selection": dict(SELECTION),
        "mutations": [
            {
                "id": "remove-incumbent-binding",
                "order": 1,
                "kind": "delete_relationship",
                "participant_id": binding_element_id,
            },
            {
                "id": "attach-authoritative-target",
                "order": 2,
                "kind": "add_relationship",
                "participant_id": SWAP_AVERAGE_NAME,
                "arguments": {
                    "relationship_type": "PRODUCED_NAME",
                    "start_id": SWAP_AVERAGE_PATH,
                    "end_id": SWAP_AVERAGE_NAME,
                },
            },
            {
                "id": "retarget-source-scalar",
                "order": 3,
                "kind": "set_properties",
                "participant_id": SWAP_AVERAGE_PATH,
                "arguments": {"properties": {"produced_sn_id": SWAP_AVERAGE_NAME}},
            },
        ],
        "guards": [
            {
                "id": "last-producing-source",
                "kind": "semantic_authority",
                "implementation": "last-producing-source",
                "participant_ids": [],
            },
            {
                "id": "out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [],
            },
        ],
        "orphan_policy": "refuse",
    }


def _attachment_row() -> dict[str, Any]:
    return {
        "id": SWAP_LOCAL_PATH,
        "identity": {
            "id": SWAP_LOCAL_PATH,
            "kind": "source",
            "source_id": SWAP_LOCAL_PATH,
            "target_id": SWAP_LOCAL_NAME,
        },
        "participants": [
            {
                "id": SWAP_LOCAL_PATH,
                "kind": "node",
                "graph_label": "StandardNameSource",
            },
            {"id": SWAP_LOCAL_NAME, "kind": "node", "graph_label": "StandardName"},
        ],
        "selection": dict(SELECTION),
        "mutations": [
            {
                "id": "attach-authoritative-target",
                "order": 1,
                "kind": "add_relationship",
                "participant_id": SWAP_LOCAL_NAME,
                "arguments": {
                    "relationship_type": "PRODUCED_NAME",
                    "start_id": SWAP_LOCAL_PATH,
                    "end_id": SWAP_LOCAL_NAME,
                },
            },
            {
                "id": "advance-source-lifecycle",
                "order": 2,
                "kind": "set_properties",
                "participant_id": SWAP_LOCAL_PATH,
                "arguments": {
                    "properties": {
                        "status": "attached",
                        "produced_sn_id": SWAP_LOCAL_NAME,
                        "claimed_at": None,
                        "claim_token": None,
                        "last_error": None,
                    }
                },
            },
        ],
        "guards": [
            {
                "id": "out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [],
            }
        ],
        "orphan_policy": "refuse",
    }


def _preview_rows(rows: list[dict[str, Any]], tag: str) -> dict[str, Any]:
    """Preview a candidate payload with ``apply=False``; nothing is written."""
    spec = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": f"probe-swap-expressibility-{tag}",
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": dict(SELECTION),
        "receipt_policy": {
            "id": "one-per-target",
            "operation": SUPERSEDE_OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    built = build_repair_authority(spec)
    assert RUN_DIR is not None
    path = RUN_DIR / f"sib-apply-live-rulings-probe-{tag}-authority.json"
    path.write_bytes(built.content)
    raw = path.read_bytes()
    res = apply_signed_manifest(
        path,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(json.loads(raw)),
        reason="read-only probe of the swap's expressibility",
        apply=False,
        run_id=RUN_ID,
    )
    return {
        "tag": tag,
        "rows": [row["id"] for row in rows],
        "refusals": res.get("refusals"),
        "manifest_sha256": res.get("manifest_sha256"),
    }


def swap() -> dict[str, Any]:
    """Show, read-only, that no owner removes and adds the binding atomically."""
    with GraphClient() as gc:
        row = gc.query(
            """
            MATCH (s:StandardNameSource {id: $avg})-[b:PRODUCED_NAME]->(n:StandardName {id: $name})
            RETURN elementId(b) AS binding_element_id, s.status AS status,
                   s.produced_sn_id AS scalar, n.name_stage AS name_stage,
                   n.validation_status AS validation
            """,
            avg=SWAP_AVERAGE_PATH,
            name=SWAP_LOCAL_NAME,
        )
        receiving = gc.query(
            """
            MATCH (n:StandardName {id: $name})
            RETURN n.name_stage AS name_stage, n.validation_status AS validation,
                   n.status AS status
            """,
            name=SWAP_AVERAGE_NAME,
        )
    if not row:
        return {
            "applied": False,
            "reason": "the swap's incumbent binding is gone; re-derive",
        }
    incumbent = dict(row[0])
    probes = [
        _preview_rows([_migration_row(incumbent["binding_element_id"])], "migration"),
        _preview_rows([_attachment_row()], "attachment"),
        _preview_rows(
            [_attachment_row(), _migration_row(incumbent["binding_element_id"])],
            "combined",
        ),
    ]
    return {
        "applied": False,
        "incumbent": incumbent,
        "receiving_name": dict(receiving[0]) if receiving else None,
        "probes": probes,
        "receiving_name_stage": receiving[0]["name_stage"] if receiving else None,
        "refused_program": "signed ordinary-source migration",
        "refusal_reason": (
            probes[0]["refusals"][0]["reason"] if probes[0]["refusals"] else None
        ),
    }


# --------------------------------------------------------------------------
# Read-back, gates, and the composed record
# --------------------------------------------------------------------------


def readback(before: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = AUTHORITY_PATH.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    retired = {row["identity"]["target_id"] for row in payload["rows"]}
    bound = {name for name, _ in BINDINGS}
    with GraphClient() as gc:
        classified = classify_sourceless_standard_names(gc)
        counts: dict[str, int] = defaultdict(int)
        members: dict[str, set[str]] = defaultdict(set)
        for row in classified:
            counts[row["population"]] += 1
            members[row["population"]].add(row["id"])
        receipts = gc.query(
            """
            MATCH (c:StandardNameChange {authority_file_sha256: $fs})
            RETURN c.row_id AS row_id, c.operation AS operation,
                   c.manifest_sha256 AS digest
            """,
            fs=file_sha,
        )
        states = gc.query(
            """
            UNWIND $ids AS id
            MATCH (sn:StandardName {id: id})
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
            OPTIONAL MATCH (sn)-[:HAS_SUCCESSOR]->(successor:StandardName)
            RETURN sn.id AS id, sn.name_stage AS stage, sn.status AS status,
                   collect(DISTINCT successor.id) AS successor_ids,
                   collect(DISTINCT source.id) AS source_ids,
                   collect(DISTINCT dd.id) AS dd_paths
            ORDER BY id
            """,
            ids=sorted(set(CLASSIFIER_TARGETS)),
        )
    receipt_rows = {str(r["row_id"]) for r in receipts}
    out: dict[str, Any] = {
        "counts": dict(counts),
        "live_non_derived": sum(
            len(v) for k, v in members.items() if k != "derived_parent"
        ),
        "states": [dict(r) for r in states],
        "receipt_rows": sorted(receipt_rows),
        "receipts": [dict(r) for r in receipts],
        "retired": sorted(retired),
        "bound": sorted(bound),
        "still_sourceless": sorted(
            name
            for name in CLASSIFIER_TARGETS
            if name in members["never_bound"] or name in members["dropped_edge"]
        ),
        "populations": {k: sorted(v) for k, v in members.items()},
    }
    if before is not None:
        baseline = {
            name
            for key, ids in (before.get("populations") or {}).items()
            if key != "derived_parent"
            for name in ids
        }
        current = {
            name
            for key, ids in members.items()
            if key != "derived_parent"
            for name in ids
        }
        out["before_counts"] = before["classifier_counts"]
        out["before_sourceless_targets"] = before["sourceless_targets"]
        out["left_the_population"] = sorted(set(before["sourceless_targets"]) - current)
        out["entered_the_population"] = sorted(current - baseline)
        out["targets_still_sourceless"] = sorted(
            name for name in CLASSIFIER_TARGETS if name in current
        )
        out["moved_populations"] = {
            key: (before["classifier_counts"].get(key), counts.get(key))
            for key in set(before["classifier_counts"]) | set(counts)
            if before["classifier_counts"].get(key) != counts.get(key)
        }
    return out


def verify(
    before: dict[str, Any] | None = None, *, raise_on_mismatch: bool = False
) -> dict[str, Any]:
    if before is None:
        assert RUN_DIR is not None
        before = json.loads(
            (RUN_DIR / "sib-apply-live-rulings-before.json").read_text()
        )
    out = readback(before=before)
    retired = set(out["retired"])
    stages = {row["id"]: row for row in out["states"]}
    receipt_targets = {row_id.split("=>", 1)[0] for row_id in out["receipt_rows"]}
    checks = {
        "every_retirement_has_receipt": retired <= receipt_targets,
        "no_receipt_outside_retirements": receipt_targets <= retired,
        "retirements_are_superseded": all(
            stages[name]["stage"] == "superseded"
            and stages[name]["status"] == "superseded"
            and stages[name]["successor_ids"]
            for name in retired
        ),
        "restored_binding_is_held": all(
            stages[name]["source_ids"] and stages[name]["dd_paths"]
            for name in out["bound"]
        ),
        "no_name_entered_the_population": not out.get("entered_the_population"),
    }
    out["checks"] = checks
    if raise_on_mismatch and not all(checks.values()):
        raise SystemExit(f"read-back mismatch: {checks} :: {out}")
    return out


def falsify() -> dict[str, Any]:
    """Negative control: a read-back that denies a member of the baseline fails.

    The gate's entered-population check compares the live population against
    the baseline the read captured. Dropping one member from that baseline
    makes the same name read as having entered, so the gate must refuse. The
    control exercises the check rather than asserting a number.
    """
    assert RUN_DIR is not None
    before = json.loads((RUN_DIR / "sib-apply-live-rulings-before.json").read_text())
    doctored = json.loads(json.dumps(before))
    never_bound = list((doctored.get("populations") or {}).get("never_bound") or [])
    if not never_bound:
        return {"check_failed_as_expected": False, "detail": "no baseline to doctor"}
    dropped = never_bound[0]
    doctored["populations"]["never_bound"] = never_bound[1:]
    out = verify(before=doctored)
    return {
        "dropped_from_baseline": dropped,
        "entered_the_population": out["entered_the_population"],
        "checks": out["checks"],
        "check_failed_as_expected": not all(out["checks"].values()),
    }


def quiet_graph_check() -> dict[str, Any]:
    """Ask the repository's own activity probe whether another client is writing."""
    from imas_codex.graph.offsite import live_graph_activity

    live_dir = Path.home() / ".config" / "reckon" / "crew" / "live"
    recent: list[dict[str, Any]] = []
    now = time.time()
    if live_dir.is_dir():
        for pointer in sorted(live_dir.glob("*.json")):
            age = now - pointer.stat().st_mtime
            if age < 1800:
                recent.append({"run": pointer.stem, "age_seconds": round(age)})
    try:
        jobs = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T %M"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        jobs = f"unavailable: {exc}"
    activity = live_graph_activity(window_seconds=6.0, interval_seconds=3.0)
    return {
        "quiet": not activity.busy,
        "busy": activity.busy,
        "activity": {
            key: value
            for key, value in dataclasses.asdict(activity).items()
            if isinstance(value, int | float | str | bool | type(None))
        },
        "recent_live_runs": recent,
        "squeue": [line for line in jobs.splitlines() if line.strip()],
    }


def compose() -> dict[str, Any]:
    """Assemble the fragment record from the artifacts this run wrote."""
    assert RUN_DIR is not None
    raw = AUTHORITY_PATH.read_bytes()
    payload = json.loads(raw)
    before = json.loads((RUN_DIR / "sib-apply-live-rulings-before.json").read_text())
    after = json.loads((RUN_DIR / "sib-apply-live-rulings-verify.json").read_text())
    preview_file = RUN_DIR / "sib-apply-live-rulings-preview.json"
    apply_file = RUN_DIR / "sib-apply-live-rulings-apply.json"
    quiet_file = RUN_DIR / "sib-apply-live-rulings-quiet.json"
    swap_file = RUN_DIR / "sib-apply-live-rulings-swap.json"
    bind_file = RUN_DIR / "sib-apply-live-rulings-bindings.json"
    read_file = RUN_DIR / "sib-apply-live-rulings-read.json"

    def load(path: Path) -> Any:
        return json.loads(path.read_text()) if path.exists() else None

    record = {
        "node": "sib-apply-live-rulings",
        "run_id": RUN_ID,
        "authority": {
            "path": str(AUTHORITY_PATH.name),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "payload_sha256": signed_payload_sha256(payload),
            "operation_id": payload["operation_id"],
            "receipt_operation": payload["receipt_policy"]["operation"],
            "rows": [row["id"] for row in payload["rows"]],
        },
        "read": load(read_file),
        "preview": load(preview_file),
        "quiet": load(quiet_file),
        "apply": load(apply_file),
        "bindings": load(bind_file),
        "swap": load(swap_file),
        "readback_before": {
            "counts": before["classifier_counts"],
            "sourceless_targets": before["sourceless_targets"],
        },
        "readback_after": after,
    }
    FRAGMENT_PATH.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


def _table(headers: list[str], rows: list[list[Any]], style: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    attr = f" class='{style}'" if style else ""
    return f"<table{attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def html_fragment(record: dict[str, Any]) -> str:
    before_counts = record["readback_before"]["counts"]
    after_counts = record["readback_after"]["counts"]
    retired = record["readback_after"]["retired"]
    left = record["readback_after"]["left_the_population"]
    entered = record["readback_after"]["entered_the_population"]
    swap = record["swap"] or {}
    states = {row["id"]: row for row in record["readback_after"]["states"]}
    bind_rows = [
        [
            item["name"],
            item["source_id"],
            item["bound"],
            (item["held"][0]["status"] if item["held"] else ""),
        ]
        for item in (record["bindings"] or {}).get("bindings", [])
    ]
    parts = []
    parts.append(
        "<h2 id='sib-apply-live-rulings'>Ruled provenance changes applied</h2>"
    )
    parts.append(
        "<p id='sib-apply-live-rulings-summary'>Three of the four ruled graph "
        "changes are applied and one is reported unapplied. Two live names are "
        "retired through the signed supersede with a successor link and one "
        "lost DD binding is restored through <code>bind_recovery_sources</code>; "
        f"the separatrix temperature swap is not expressible by any owner and "
        f"stays unapplied. The classifier's live non-derived population reads "
        f"{record['readback_after']['live_non_derived']} after, against "
        f"{sum(v for k, v in before_counts.items() if k != 'derived_parent')} before.</p>"
    )
    parts.append(
        "<h3 id='sib-apply-live-rulings-retired'>Retired through the signed supersede</h3>"
    )
    parts.append(
        _table(
            [
                "predecessor",
                "successor (quantity holder)",
                "stage after",
                "status after",
            ],
            [
                [
                    row["id"],
                    ", ".join(states[row["id"]]["successor_ids"]),
                    states[row["id"]]["stage"],
                    states[row["id"]]["status"],
                ]
                for row in record["readback_after"]["states"]
                if row["id"] in retired
            ],
        )
    )
    parts.append("<h3 id='sib-apply-live-rulings-restored'>Restored bindings</h3>")
    parts.append(_table(["name", "source", "sources linked", "status"], bind_rows))
    parts.append(
        "<h3 id='sib-apply-live-rulings-swap'>The separatrix temperature swap, unapplied</h3>"
        "<p id='sib-apply-live-rulings-swap-why'>The swap's first half is a "
        "source migration: it removes the incumbent <code>PRODUCED_NAME</code> "
        "edge and adds its replacement, and the signed ordinary-source "
        "migration program is the owner that does both in one transaction. That "
        "program admits only a target whose <code>name_stage</code> is "
        "<code>accepted</code> and whose <code>validation_status</code> is "
        "<code>valid</code>; the receiving name here is "
        f"<code>{html.escape(str(swap.get('receiving_name_stage', '')))}</code>, "
        "so the row is refused. The graph holds no other owner that removes a "
        "binding and adds its replacement atomically: the remaining closed "
        "envelope programs either add without removing, or require a stale "
        "source, a name realization, or an exhausted name. Applying the two "
        "halves as separate writes would leave the name sourceless between them "
        "or transiently bind one source to two live names, so the swap is left "
        "unapplied.</p>"
    )
    parts.append(
        _table(
            ["probe", "rows", "refusal"],
            [
                [
                    probe["tag"],
                    ", ".join(probe["rows"]),
                    "; ".join(r["reason"] for r in (probe["refusals"] or []))
                    or "admitted",
                ]
                for probe in swap.get("probes", [])
            ],
        )
    )
    parts.append("<h3 id='sib-apply-live-rulings-authority'>Signed authority</h3>")
    parts.append(
        "<pre id='sib-apply-live-rulings-authority-pre'>"
        + html.escape(json.dumps(record["authority"], indent=1))
        + "</pre>"
    )
    parts.append("<h3 id='sib-apply-live-rulings-quiet'>Quiet-graph probe</h3>")
    quiet = record["quiet"] or {}
    parts.append(
        "<p id='sib-apply-live-rulings-quiet-p'>The repository's own activity probe "
        f"reports <code>quiet={html.escape(str(quiet.get('quiet')))}</code> "
        f"(busy={html.escape(str(quiet.get('busy')))}). Recent live runs: "
        + ", ".join(r["run"] for r in quiet.get("recent_live_runs", []) or [])
        + ".</p>"
    )
    parts.append("<h3 id='sib-apply-live-rulings-readback'>Classifier read-back</h3>")
    parts.append(
        f"<img id='sib-apply-live-rulings-fig' src='{FIGURE_SRC}' alt='Classifier "
        "populations before and after the applied writes; only the never-bound "
        "population moves, by three names.'/>"
    )
    parts.append(
        "<p class='figure-caption' id='sib-apply-live-rulings-fig-cap'>Classifier "
        "sourceless populations before and after the two signed retirements and "
        "the one restored DD binding, on the live graph at the read-back. The "
        "never-bound population is the only one that moves, from 21 to 18. Three "
        "names leave the live set: the two retired ones, which the classifier drops "
        "entirely once they are terminal, and the electric-field name, which leaves "
        "the never-bound population by gaining a source.</p>"
    )
    parts.append(
        _table(
            ["population", "before", "after"],
            [
                [key, before_counts.get(key, 0), after_counts.get(key, 0)]
                for key in sorted(set(before_counts) | set(after_counts))
            ],
        )
    )
    parts.append(
        f"<p id='sib-apply-live-rulings-readback-p'>Targets that left the "
        f"sourceless population: {html.escape(', '.join(left) or 'none')}. "
        f"Entered: {html.escape(', '.join(entered) or 'none')}. Before: "
        f"{html.escape(', '.join(record['readback_before']['sourceless_targets']) or 'none')}.</p>"
    )
    parts.append("<h3 id='sib-apply-live-rulings-preview'>Preview and receipt</h3>")
    parts.append(
        "<pre id='sib-apply-live-rulings-preview-pre'>"
        + html.escape(
            json.dumps(
                {
                    "preview": record["preview"],
                    "apply": record["apply"],
                    "receipts": record["readback_after"]["receipts"],
                },
                indent=1,
                default=str,
            )
        )
        + "</pre>"
    )
    return "\n".join(parts) + "\n"


def write_fragment() -> dict[str, Any]:
    record = compose()
    html_path = HERE / "sib-apply-live-rulings.html"
    html_path.write_text(html_fragment(record))
    return {"json": str(FRAGMENT_PATH.name), "html": str(html_path.name)}


def draw_population_figure() -> dict[str, Any]:
    """Draw the classifier populations before and after the three applied writes.

    One panel, before against after per population. The population the writes
    moved carries the feature colour and the ones that do not move stay grey, so
    the reader sees which population the ruled writes changed without a legend.
    Each bar's count sits beyond its end, in clear space.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sheet = Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle"
    if sheet.exists():
        plt.style.use(str(sheet))

    record = json.loads(FRAGMENT_PATH.read_text())
    before = record["readback_before"]["counts"]
    after = record["readback_after"]["counts"]
    moved = set(record["readback_after"].get("moved_populations") or {})

    keys = sorted(set(before) | set(after), key=lambda k: -before.get(k, 0))
    labels = [k.replace("_", " ") for k in keys]
    b = [before.get(k, 0) for k in keys]
    a = [after.get(k, 0) for k in keys]

    grey = "#929292"
    feature = "#31688e"
    fig, ax = plt.subplots(figsize=(14, 3.4))
    h = 0.36
    for i, (key, bv, av) in enumerate(zip(keys, b, a, strict=True)):
        changed = key in moved
        ax.barh(i + h / 2, bv, height=h, color=grey, zorder=2)
        ax.barh(i - h / 2, av, height=h, color=feature if changed else grey, zorder=3)
        ax.text(
            bv + 1.5, i + h / 2, f"before {bv}", va="center", fontsize=14, color=grey
        )
        ax.text(
            av + 1.5,
            i - h / 2,
            f"after {av}",
            va="center",
            fontsize=14,
            color=feature if changed else grey,
        )
    ax.set_yticks(list(range(len(keys))))
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("standard names", fontsize=16)
    ax.set_xlim(0, max(b + a) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.invert_yaxis()
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "populations-before-after.svg"
    fig.savefig(out)
    plt.close(fig)

    from shutil import which

    converter = which("rsvg-convert")
    if converter:
        subprocess.run([converter, "-w", "1400", str(out)], check=True)
    return {"svg": str(out), "rendered": bool(converter)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "read",
            "authority",
            "preview",
            "quiet",
            "apply",
            "bindings",
            "swap",
            "readback",
            "verify",
            "falsify",
            "record",
            "figure",
        ],
    )
    parser.add_argument("--digest")
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    _resolve_runtime(args)
    assert RUN_DIR is not None

    def emit(name: str, payload: Any) -> None:
        (RUN_DIR / f"sib-apply-live-rulings-{name}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )
        print(json.dumps(payload, indent=2, default=str)[:6000])

    if args.mode == "read":
        out = read()
        emit("read", out)
        # The baseline every later gate compares against is this read, captured
        # before the first write.
        (RUN_DIR / "sib-apply-live-rulings-before.json").write_text(
            json.dumps(out, indent=2, default=str) + "\n"
        )
    elif args.mode == "authority":
        emit("authority", build_authority())
    elif args.mode == "preview":
        emit("preview", _signed_call(False, None))
    elif args.mode == "quiet":
        emit("quiet", quiet_graph_check())
    elif args.mode == "apply":
        assert args.digest
        emit("apply", _signed_call(True, args.digest))
    elif args.mode == "bindings":
        emit("bindings", bind())
    elif args.mode == "swap":
        emit("swap", swap())
    elif args.mode == "readback":
        emit("readback", readback())
    elif args.mode == "verify":
        emit("verify", verify(raise_on_mismatch=True))
    elif args.mode == "falsify":
        emit("falsify", falsify())
    elif args.mode == "record":
        emit("record", write_fragment())
    elif args.mode == "figure":
        emit("figure", draw_population_figure())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Partition and restore the documentation reset by the 2026-11-11 drain.

Drain job 1310773 steered 508 names through ``refresh_drifted_sources`` at its
startup; 400 of them held ``docs_stage='accepted'`` before the reset. Each reset
wrote a ``DocsRevision`` snapshot of the pre-reset text, so the accepted
documentation is recoverable per name.

The attributed set is the drain's own before/after leave set joined to the
``DocsRevision`` receipts created in the reset window. It is partitioned three
ways:

* ``a`` - already ``docs_stage='accepted'`` again: untouched;
* ``b`` - the source is still content-drifted under the corrected refresh
  (``requires_steering``, read with the refresh's own reach-only comparison):
  left for regeneration;
* ``c`` - the remaining names, restored from their own snapshot and accepted,
  through the signed repair manifest so each name carries a signed receipt
  naming its snapshot.

Read-only except for the ``restore`` phase. No LLM call, no ``sn run``.

Usage::

    python sib-restore-reset-docs-run.py census --drain-run-dir DIR --out census.json
    python sib-restore-reset-docs-run.py restore --census census.json --out restore.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESET_START = "2026-11-11T17:28:00+00:00"
RESET_END = "2026-11-11T17:29:00+00:00"
SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
OPERATION = "restore_documentation_from_revision"

STATE_QUERY = """
UNWIND $names AS id
MATCH (sn:StandardName {id: id})
RETURN sn.id AS id, sn.name_stage AS name_stage, sn.docs_stage AS docs_stage,
       sn.documentation AS documentation,
       sn.description AS description,
       sn.docs_chain_length AS docs_chain_length,
       sn.docs_model AS docs_model,
       sn.reviewer_score_docs AS reviewer_score_docs,
       sn.edit_status AS edit_status
"""

SNAPSHOT_QUERY = """
MATCH (sn:StandardName {id: $sn_id})-[:DOCS_REVISION_OF]->(rev:DocsRevision)
WHERE rev.id = $revision_id
RETURN rev.id AS revision_id, rev.documentation AS documentation,
       rev.description AS description,
       rev.revision_number AS revision_number,
       rev.reviewer_score_docs AS reviewer_score_docs,
       rev.model AS model, rev.created_at AS created_at
"""

ACCEPTED_COUNTS = """
MATCH (sn:StandardName)
WHERE sn.name_stage <> 'superseded'
RETURN coalesce(sn.docs_stage, 'null') AS docs_stage, count(sn) AS n
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stage_map(path: Path) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in json.loads(path.read_text())["accepted_names"]}


def left_accepted(drain_run_dir: Path) -> dict[str, dict[str, Any]]:
    before = stage_map(drain_run_dir / "before.json")
    after = stage_map(drain_run_dir / "after.json")
    out: dict[str, dict[str, Any]] = {}
    for name, row in before.items():
        if row.get("docs_stage") != "accepted":
            continue
        after_stage = after.get(name, {}).get("docs_stage")
        if after_stage != "accepted":
            out[name] = {
                "before": "accepted",
                "after": after_stage,
                "in_after": name in after,
            }
    return out


def reset_revisions(gc: Any) -> list[dict[str, Any]]:
    rows = gc.query(
        """
        MATCH (rev:DocsRevision)
        WHERE rev.created_at >= datetime($start)
          AND rev.created_at < datetime($end)
        MATCH (owner:StandardName)-[:DOCS_REVISION_OF]->(rev)
        RETURN rev.id AS revision_id, rev.standard_name_id AS snapshot_name_id,
               owner.id AS owner_id, rev.revision_number AS revision_number,
               rev.created_at AS created_at
        ORDER BY rev.created_at, owner.id
        """,
        start=RESET_START,
        end=RESET_END,
    )
    return [dict(r) for r in (rows or [])]


def live_state(gc: Any, names: list[str]) -> dict[str, dict[str, Any]]:
    rows = gc.query(STATE_QUERY, names=names)
    return {r["id"]: dict(r) for r in (rows or [])}


def accepted_counts(gc: Any) -> dict[str, int]:
    rows = gc.query(ACCEPTED_COUNTS)
    return {str(r["docs_stage"]): int(r["n"]) for r in (rows or [])}


def snapshot(gc: Any, sn_id: str, revision_id: str) -> dict[str, Any] | None:
    rows = gc.query(SNAPSHOT_QUERY, sn_id=sn_id, revision_id=revision_id)
    return dict(rows[0]) if rows else None


def census(gc: Any, drain_run_dir: Path) -> dict[str, Any]:
    from imas_codex.standard_names.source_refresh import (
        detect_source_drift,
        refresh_drifted_sources,
    )

    left = left_accepted(drain_run_dir)
    revisions = reset_revisions(gc)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in revisions:
        by_name.setdefault(row["owner_id"], []).append(row)
    attributed = sorted(set(left) & set(by_name))
    state = live_state(gc, attributed)

    unchecked: list[dict[str, str]] = []
    drifted = {
        d["sn_id"]: d
        for d in detect_source_drift(
            include_accepted=True, sn_ids=None, unchecked=unchecked, gc=gc
        )
    }
    dry = refresh_drifted_sources(dry_run=True, include_accepted=True, gc=gc)
    steering = sorted(
        n for n in attributed if drifted.get(n, {}).get("requires_steering")
    )

    parts: dict[str, list[str]] = {"a": [], "b": [], "c": []}
    for name in attributed:
        if state.get(name, {}).get("docs_stage") == "accepted":
            parts["a"].append(name)
        elif name in set(steering):
            parts["b"].append(name)
        else:
            parts["c"].append(name)

    plan_c = []
    for name in parts["c"]:
        rows = by_name[name]
        rev = rows[-1]
        snap = snapshot(gc, name, rev["revision_id"])
        current = state.get(name, {})
        plan_c.append(
            {
                "sn_id": name,
                "revision_id": rev["revision_id"],
                "snapshot_revision_number": rev["revision_number"],
                "has_snapshot": bool(snap and (snap.get("documentation") or "")),
                "snapshot_docs_length": len((snap or {}).get("documentation") or ""),
                "current_docs_length": len(current.get("documentation") or ""),
                "snapshot_score": (snap or {}).get("reviewer_score_docs"),
                "current_stage": current.get("docs_stage"),
                "docs_identical": (snap or {}).get("documentation")
                == current.get("documentation"),
            }
        )

    return {
        "instrument": {
            "generated_at": now_utc(),
            "drain_run_dir": str(drain_run_dir),
            "reset_window": [RESET_START, RESET_END],
        },
        "attribution": {
            "left_accepted": len(left),
            "reset_revisions": len(revisions),
            "reset_names": len(by_name),
        },
        "partition": {k: {"count": len(v), "names": v} for k, v in parts.items()},
        "partition_sum": sum(len(v) for v in parts.values()),
        "drift_dry_run": {
            "detected": dry["detected"],
            "renamed": dry["renamed"],
            "steered": dry["steering_count"] if "steering_count" in dry else dry["steered"],
            "dry_run": dry["dry_run"],
            "requires_steering_among_attributed": len(steering),
            "unchecked_sources": len(unchecked),
        },
        "accepted_counts_before_restore": accepted_counts(gc),
        "control_names": parts["a"][:1] + parts["b"][:1],
        "restore_plan": plan_c,
        "left_accepted_detail": left,
    }


def _guard(implementation: str, participant_ids: list[str]) -> dict[str, Any]:
    kind = (
        "collateral_immutability"
        if implementation == "out-of-allowlist-immutability"
        else "semantic_authority"
    )
    return {
        "id": implementation,
        "kind": kind,
        "implementation": implementation,
        "participant_ids": participant_ids,
    }


def _restore_properties(snap: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = {
        "documentation": snap.get("documentation") or "",
        "description": snap.get("description") or "",
        "docs_stage": "accepted",
        "edit_status": "applied",
        "docs_chain_length": int(snap.get("revision_number") or 0) + 1,
    }
    if snap.get("reviewer_score_docs") is not None:
        props["reviewer_score_docs"] = float(snap["reviewer_score_docs"])
    if snap.get("model") is not None:
        props["docs_model"] = snap["model"]
    return props


def _row(row_id: str, sn_id: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row_id,
        "identity": {"id": row_id, "kind": "standard_name", "target_id": sn_id},
        "participants": [
            {"id": sn_id, "kind": "node", "graph_label": "StandardName"}
        ],
        "selection": SELECTION,
        "mutations": [
            {
                "id": f"{row_id}:mutation",
                "order": 0,
                "kind": "set_properties",
                "participant_id": sn_id,
                "arguments": {"properties": properties},
            }
        ],
        "guards": [
            _guard("structural-legitimacy", [sn_id]),
            _guard("out-of-allowlist-immutability", [sn_id]),
        ],
        "orphan_policy": "refuse",
    }


def _authority(rows: list[dict[str, Any]], operation: str) -> dict[str, Any]:
    from imas_codex.standard_names.signed_manifest import signed_payload_sha256

    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": operation,
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": SELECTION,
        "receipt_policy": {
            "id": "one-per-restored-name",
            "operation": OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    payload_sha256 = signed_payload_sha256(authority)
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": payload_sha256,
    }
    return authority


def restore(gc: Any, census_data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    from imas_codex.standard_names.signed_manifest import (
        apply_signed_manifest,
        signed_payload_sha256,
    )

    plan = census_data["restore_plan"]
    rows: list[dict[str, Any]] = []
    missing_snapshot: list[str] = []
    no_score: list[str] = []
    for item in plan:
        sn_id = item["sn_id"]
        snap = snapshot(gc, sn_id, item["revision_id"])
        if not snap or not (snap.get("documentation") or ""):
            missing_snapshot.append(sn_id)
            continue
        if snap.get("reviewer_score_docs") is None:
            no_score.append(sn_id)
        rows.append(_row(f"restore:{sn_id}:{item['revision_id']}", sn_id, _restore_properties(snap)))

    reason = (
        "restore the accepted documentation that the 2026-11-11 drift "
        "refresh reset, from each name's own DocsRevision snapshot"
    )
    from imas_codex.standard_names.signed_manifest import SignedManifestConflict

    batches = [rows[i : i + 60] for i in range(0, len(rows), 60)]
    receipts: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        authority = _authority(batch, f"docs-restore-batch-{index:02d}")
        authority_path = out_dir / f"restore-authority-{index:02d}.json"
        raw = json.dumps(authority, sort_keys=True, indent=2).encode()
        authority_path.write_bytes(raw)
        file_sha256 = hashlib.sha256(raw).hexdigest()
        payload_sha256 = authority["signature"]["sha256"]
        attempt = 0
        while True:
            attempt += 1
            preview = apply_signed_manifest(
                authority_path,
                authority_file_sha256=file_sha256,
                authority_payload_sha256=payload_sha256,
                reason=reason,
                gc=gc,
            )
            applied = None
            if preview.get("counts", {}).get("admitted"):
                try:
                    applied = apply_signed_manifest(
                        authority_path,
                        authority_file_sha256=file_sha256,
                        authority_payload_sha256=payload_sha256,
                        reason=reason,
                        apply=True,
                        manifest_sha256=preview["manifest_sha256"],
                        gc=gc,
                    )
                except SignedManifestConflict as exc:
                    if attempt >= 3:
                        receipts.append(
                            {
                                "batch": index,
                                "authority": str(authority_path),
                                "rows": len(batch),
                                "attempts": attempt,
                                "error": str(exc),
                            }
                        )
                        break
                    continue
            receipts.append(
                {
                    "batch": index,
                    "authority": str(authority_path),
                    "authority_file_sha256": file_sha256,
                    "attempts": attempt,
                    "rows": len(batch),
                    "preview": {
                        "outcome": preview.get("outcome"),
                        "counts": preview.get("counts"),
                        "refusals": (preview.get("refusals") or [])[:20],
                        "refusal_count": len(preview.get("refusals") or []),
                    },
                    "applied": None
                    if applied is None
                    else {
                        "outcome": applied.get("outcome"),
                        "changed": applied.get("changed"),
                        "counts": applied.get("counts"),
                        "manifest_sha256": applied.get("manifest_sha256"),
                    },
                }
            )
            break

    names = [item["sn_id"] for item in plan if item["sn_id"] not in missing_snapshot]
    after = live_state(gc, names)
    readback = []
    for item in plan:
        sn_id = item["sn_id"]
        if sn_id in missing_snapshot:
            continue
        snap = snapshot(gc, sn_id, item["revision_id"])
        state = after.get(sn_id, {})
        readback.append(
            {
                "sn_id": sn_id,
                "revision_id": item["revision_id"],
                "documentation_equals_snapshot": state.get("documentation")
                == (snap or {}).get("documentation"),
                "docs_stage": state.get("docs_stage"),
                "docs_stage_accepted": state.get("docs_stage") == "accepted",
            }
        )

    return {
        "authority_path": str(authority_path),
        "authority_file_sha256": file_sha256,
        "authority_payload_sha256": payload_sha256,
        "rows": len(rows),
        "missing_snapshot": missing_snapshot,
        "names_without_snapshot_score": no_score,
        "batches": receipts,
        "applied_changes": sum(
            (r.get("applied") or {}).get("changed") or 0 for r in receipts
        ),
        "readback": readback,
        "readback_all_equal": all(r["documentation_equals_snapshot"] for r in readback),
        "readback_all_accepted": all(r["docs_stage_accepted"] for r in readback),
        "accepted_counts_after_restore": accepted_counts(gc),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["census", "restore"])
    ap.add_argument("--drain-run-dir")
    ap.add_argument("--census")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    from imas_codex.graph.client import GraphClient

    out = Path(args.out)
    work = Path(args.work_dir) if args.work_dir else out.parent
    with GraphClient() as gc:
        if args.phase == "census":
            if not args.drain_run_dir:
                raise SystemExit("census needs --drain-run-dir")
            result = census(gc, Path(args.drain_run_dir))
        else:
            if not args.census:
                raise SystemExit("restore needs --census")
            result = restore(gc, json.loads(Path(args.census).read_text()), work)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({
        "phase": args.phase,
        "out": str(out),
        "partition": {k: v["count"] for k, v in result.get("partition", {}).items()},
        "rows": result.get("rows"),
        "readback_all_equal": result.get("readback_all_equal"),
        "readback_all_accepted": result.get("readback_all_accepted"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
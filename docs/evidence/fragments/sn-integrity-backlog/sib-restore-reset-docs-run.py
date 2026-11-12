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
from datetime import UTC, datetime
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
    return datetime.now(UTC).isoformat(timespec="seconds")


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
            "steered": dry["steering_count"]
            if "steering_count" in dry
            else dry["steered"],
            "dry_run": dry["dry_run"],
            "requires_steering_among_attributed": len(steering),
            "unchecked_sources": len(unchecked),
        },
        "accepted_counts_before_restore": accepted_counts(gc),
        "control_names": parts["a"][:1] + parts["b"][:1],
        "restore_plan": plan_c,
        "left_accepted_detail": left,
    }


def restore(gc: Any, census_data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Restore each partition-(c) name from its own snapshot, in signed cohorts.

    The mutation is ``signed_manifest``'s docs-revision-restore kind, which
    writes only the documentation fields a ``DocsRevision`` carries plus
    ``docs_stage='accepted'``, so it is admitted on names that own a live
    structural child where ``set_properties`` is refused.
    """
    from imas_codex.standard_names.signed_manifest import (
        _DOCS_RESTORE_ADAPTER,
        _DOCS_RESTORE_GUARDS,
        _DOCS_RESTORE_MUTATION,
        apply_signed_manifest,
        signed_payload_sha256,
    )

    plan = census_data["restore_plan"]
    names = [item["sn_id"] for item in plan]
    stage = {
        row["id"]: row["docs_stage"]
        for row in gc.query(
            "UNWIND $names AS id MATCH (sn:StandardName {id: id}) "
            "RETURN sn.id AS id, sn.docs_stage AS docs_stage",
            names=names,
        )
    }
    todo = [item for item in plan if stage.get(item["sn_id"]) != "accepted"]
    rows = [
        {
            "id": f"docs:{item['sn_id']}",
            "sn_id": item["sn_id"],
            "snapshot_id": item["revision_id"],
            "selection": SELECTION,
        }
        for item in todo
    ]
    reason = (
        "restore the accepted documentation that the 2026-11-11 drift "
        "refresh reset, from each name's own DocsRevision snapshot"
    )
    receipts: list[dict[str, Any]] = []
    for index, offset in enumerate(range(0, len(rows), 90)):
        batch = rows[offset : offset + 90]
        authority: dict[str, Any] = {
            "schema": "imas-codex.docs-revision-restore.v1",
            "operation_id": OPERATION,
            "rows": batch,
            "selection": SELECTION,
        }
        authority["signature"] = {
            "canonicalization": "json-sort-keys-v1",
            "sha256": signed_payload_sha256(authority),
        }
        authority_path = out_dir / f"restore-authority-{index:02d}.json"
        raw = json.dumps(authority, sort_keys=True, indent=2).encode()
        authority_path.write_bytes(raw)
        file_sha256 = hashlib.sha256(raw).hexdigest()
        payload_sha256 = authority["signature"]["sha256"]
        preview = apply_signed_manifest(
            authority_path,
            authority_file_sha256=file_sha256,
            authority_payload_sha256=payload_sha256,
            authority_adapter=_DOCS_RESTORE_ADAPTER,
            mutation_kind=_DOCS_RESTORE_MUTATION,
            guard_set=_DOCS_RESTORE_GUARDS,
            reason=reason,
            gc=gc,
        )
        entry: dict[str, Any] = {
            "batch": index,
            "authority": str(authority_path),
            "authority_file_sha256": file_sha256,
            "rows": len(batch),
            "preview": preview.get("outcome"),
            "refusals": (preview.get("refusals") or [])[:20],
        }
        if not preview.get("refusals"):
            applied = apply_signed_manifest(
                authority_path,
                authority_file_sha256=file_sha256,
                authority_payload_sha256=payload_sha256,
                authority_adapter=_DOCS_RESTORE_ADAPTER,
                mutation_kind=_DOCS_RESTORE_MUTATION,
                guard_set=_DOCS_RESTORE_GUARDS,
                reason=reason,
                apply=True,
                manifest_sha256=preview["manifest_sha256"],
                gc=gc,
            )
            entry["applied"] = applied.get("outcome")
            entry["changed"] = applied.get("changed")
            entry["docs_revision_ids"] = applied.get("docs_revision_ids")
        receipts.append(entry)

    after = live_state(gc, names)
    readback = []
    for item in plan:
        snap = snapshot(gc, item["sn_id"], item["revision_id"])
        state = after.get(item["sn_id"], {})
        readback.append(
            {
                "sn_id": item["sn_id"],
                "revision_id": item["revision_id"],
                "documentation_equals_snapshot": (state.get("documentation") or "")
                == ((snap or {}).get("documentation") or ""),
                "docs_stage": state.get("docs_stage"),
                "docs_stage_accepted": state.get("docs_stage") == "accepted",
            }
        )

    return {
        "cohorts": receipts,
        "changed_total": sum(entry.get("changed", 0) for entry in receipts),
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
    print(
        json.dumps(
            {
                "phase": args.phase,
                "out": str(out),
                "partition": {
                    k: v["count"] for k, v in result.get("partition", {}).items()
                },
                "rows": result.get("rows"),
                "readback_all_equal": result.get("readback_all_equal"),
                "readback_all_accepted": result.get("readback_all_accepted"),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Preview, apply and audit the exact recorded separatrix naming cohort.

The authority fixes identities and source paths. Mutations use the owning
rename writer; every apply is preceded by a persisted dry-run and followed by
a receipt read keyed to the writer's returned run id.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import edit

HERE = Path(__file__).resolve().parent
AUTHORITY = HERE / "sib-separatrix-reclaim-apply-authority.json"
PRIOR = HERE / "sib-separatrix-suffix-apply-authority.json"
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T185046746594-sib-separatrix-reclaim-apply"
)
BARE = "toroidal_tritium_velocity"
REASON = (
    "Use X_at_plasma_boundary for a local separatrix value and "
    "flux_surface_averaged_X_at_plasma_boundary for its surface average. "
    "Reclaim only when superseded intermediates lead to this identity, "
    "or the holder has a different DD locus from the tombstone and this identity."
)
TRITIUM_PATHS = [
    "summary/local/pedestal/velocity_phi/tritium/value",
    "summary/local/separatrix/velocity_phi/tritium/value",
]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def snapshot(gc, ids):
    rows = gc.query(
        """
      UNWIND $ids AS id MATCH (n:StandardName {id:id})
      OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n)
      OPTIONAL MATCH (n)-[:HAS_REVIEW]->(v:StandardNameReview)
      RETURN id, n.name_stage AS stage, n.status AS status,
        n.source_paths AS paths, n.description AS description,
        n.reviewer_score_name AS score, n.unit AS unit, n.run_id AS run_id,
        collect(DISTINCT s.id) AS sources, collect(DISTINCT v.id) AS reviews
      ORDER BY id
    """,
        ids=sorted(set(ids)),
    )
    return {r["id"]: dict(r) for r in rows}


def lineage(gc, name):
    return gc.query(
        """
      MATCH (root:StandardName {id:$id})
      OPTIONAL MATCH p=(root)<-[:REFINED_FROM*1..16]-(tip:StandardName)
      WHERE tip.name_stage <> 'superseded'
        AND all(n IN nodes(p)[0..-1] WHERE n.name_stage='superseded')
      RETURN root.id AS root, root.source_paths AS root_paths,
        [n IN nodes(p) | {id:n.id,stage:n.name_stage,paths:n.source_paths}] AS chain
      LIMIT 100
    """,
        id=name,
    )


def prepare():
    original = json.loads(PRIOR.read_text())["names"]
    assert len(original) == 98
    with GraphClient() as gc:
        state = snapshot(
            gc,
            [n for r in original for n in (r["id"], r["target"])]
            + [BARE + "_at_pedestal_top"],
        )
        assert len(state) > 98 and "effective_charge_at_separatrix" in state
        rows = []
        for r in original:
            if r["id"] == r["target"] or state[r["id"]]["stage"] == "superseded":
                continue
            row = {k: r[k] for k in ("id", "target", "paths", "locus")}
            row["before"] = state[r["id"]]
            row["target_before"] = state.get(r["target"])
            row["lineage"] = lineage(gc, r["target"])
            row["preview"] = asdict(
                edit.apply_edit(
                    target=r["id"],
                    rename=r["target"],
                    reason=REASON,
                    origin="agent",
                    scope="only_self",
                    dry_run=True,
                    gc=gc,
                )
            )
            row["action"] = (
                "tritium_split"
                if r["id"] == BARE
                else ("report" if row["preview"]["blocked"] else "reclaim")
            )
            rows.append(row)
        result = {
            "run_id": RUN.name,
            "observed_at": datetime.now(UTC).isoformat(),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "authority": {
                "convention": REASON,
                "clarification": "Follow superseded intermediates to the live tip; electron-temperature closes on the renaming name. Generated models are expected ignored provisioning.",
                "origin": "explicit continuation instruction",
            },
            "census": original,
            "before": state,
            "operations": rows,
            "tritium": {
                "retire": BARE,
                "compose": [BARE + "_at_pedestal_top", BARE + "_at_plasma_boundary"],
            },
            "spend_ceiling_usd": 2.0,
        }
        write(RUN / "authority-full-preview.json", result)
        result["census"] = [
            {k: r[k] for k in ("id", "target", "paths", "locus")} for r in original
        ]
        for item in state.values():
            reviews = item.pop("reviews")
            item["review_count"] = len(reviews)
            item["review_ids_sha256"] = hashlib.sha256(
                json.dumps(sorted(reviews)).encode()
            ).hexdigest()
        write(AUTHORITY, result)
        print(
            json.dumps(
                {
                    "operations": [
                        {k: r[k] for k in ("id", "target", "action")} for r in rows
                    ]
                },
                indent=2,
            ),
            flush=True,
        )


def apply_one(name):
    a = json.loads(AUTHORITY.read_text())
    row = next(r for r in a["operations"] if r["id"] == name)
    assert row["action"] == "reclaim", row
    key = hashlib.sha256(name.encode()).hexdigest()[:16]
    receipt_path = RUN / f"reclaim-{key}.json"
    assert not receipt_path.exists(), "read an existing receipt rather than replaying"
    with GraphClient() as gc:
        before = snapshot(gc, [name, row["target"]])
        preview = asdict(
            edit.apply_edit(
                target=name,
                rename=row["target"],
                reason=REASON,
                origin="agent",
                scope="only_self",
                dry_run=True,
                gc=gc,
            )
        )
        evidence = {
            "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
            "before": before,
            "preview": preview,
        }
        write(RUN / f"preview-{key}.json", evidence)
        assert not preview["blocked"], preview
        result = asdict(
            edit.apply_edit(
                target=name,
                rename=row["target"],
                reason=REASON,
                origin="agent",
                scope="only_self",
                dry_run=False,
                gc=gc,
            )
        )
        evidence["apply"] = result
        write(receipt_path, evidence)
        assert result["applied"] and not result["blocked"], result
        evidence["receipts"] = gc.query(
            "MATCH (c:StandardNameChange {run_id:$run}) RETURN properties(c) AS receipt",
            run=result["run_id"],
        )
        evidence["after"] = snapshot(gc, [name, row["target"]])
        write(receipt_path, evidence)
        receipts = [
            r["receipt"]
            for r in evidence["receipts"]
            if r["receipt"].get("operation") == "reclaim_spelling"
        ]
        assert len(receipts) == 1, evidence["receipts"]
        assert json.loads(receipts[0]["reason"])["case"] in (
            "closing_lineage",
            "different_locus",
        )
        after = evidence["after"][row["target"]]
        assert after["stage"] != "superseded" and after["run_id"] == result["run_id"]
        assert set(before[name]["sources"]) <= set(after["sources"])
        assert set(before[row["target"]]["reviews"]) <= set(after["reviews"])
        print(
            json.dumps(
                {
                    "from": name,
                    "to": row["target"],
                    "receipt": receipts[0]["id"],
                    "run_id": result["run_id"],
                    "case": json.loads(receipts[0]["reason"])["case"],
                    "sources": after["sources"],
                },
                default=str,
            ),
            flush=True,
        )


def tritium_authority():
    from imas_codex.standard_names.repair_authority import build_repair_authority

    guards = [
        {
            "id": name,
            "kind": "collateral_immutability"
            if name == "out-of-allowlist-immutability"
            else "semantic_authority",
            "implementation": name,
            "participant_ids": [BARE],
        }
        for name in (
            "structural-legitimacy",
            "no-live-producing-source",
            "no-live-structural-child",
            "out-of-allowlist-immutability",
        )
    ]
    built = build_repair_authority(
        {
            "operation_id": "retire-unlocated-tritium-stub",
            "authority_mode": "external_reviewed",
            "orphan_policy": "refuse",
            "receipt_policy": {
                "id": "one-per-retired-identity",
                "operation": "retire_unlocated_tritium_stub",
                "cardinality": "per_target",
                "link_participant_kind": "standard_name",
                "replay_projection": ["manifest_sha256", "row_id"],
            },
            "rows": [
                {
                    "id": BARE,
                    "identity": {
                        "id": BARE,
                        "kind": "standard_name",
                        "target_id": BARE,
                    },
                    "participants": [
                        {
                            "id": BARE,
                            "kind": "node",
                            "graph_label": "StandardName",
                        }
                    ],
                    "mutations": [
                        {
                            "id": "retire-stub",
                            "order": 0,
                            "kind": "supersede",
                            "participant_id": BARE,
                        }
                    ],
                    "guards": guards,
                    "orphan_policy": "refuse",
                }
            ],
        }
    )
    from imas_codex.graph.models import RepairGuard, RepairMutation, RepairParticipant

    for row in json.loads(built.content)["rows"]:
        for key, model in (
            ("participants", RepairParticipant),
            ("mutations", RepairMutation),
            ("guards", RepairGuard),
        ):
            for item in row[key]:
                model.model_validate(item)
    authority = json.loads(AUTHORITY.read_text())
    authority["tritium"]["retirement_authority"] = json.loads(built.content)
    authority["tritium"]["retirement_file_sha256"] = built.file_sha256
    authority["tritium"]["source_paths"] = TRITIUM_PATHS
    authority["tritium"]["composition"] = (
        "Retry exactly these two failed sources and compose each location through the existing claimed-source pipeline, with run-attributed cost bounded by USD 2. Preserve any tombstone refusal."
    )
    authority["applied_reclaims"] = []
    for receipt_path in sorted(RUN.glob("reclaim-*.json")):
        receipt = json.loads(receipt_path.read_text())
        authority["applied_reclaims"].append(
            {
                "apply": receipt["apply"],
                "receipts": receipt["receipts"],
                "after": receipt["after"],
            }
        )
    write(AUTHORITY, authority)
    print(
        json.dumps(
            {
                "retirement_authority_sha256": built.file_sha256,
                "applied_reclaims": len(authority["applied_reclaims"]),
            }
        ),
        flush=True,
    )


async def tritium():
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.graph_store.graph_audits import (
        create_sn_run_open,
        finalize_sn_run,
    )
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        retry_failed_sources,
        set_source_compose_hint,
    )
    from imas_codex.standard_names.pool_adapter import _seed_explicit_paths
    from imas_codex.standard_names.signed_manifest import (
        apply_signed_manifest,
        signed_payload_sha256,
    )
    from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
        compose_batch,
    )

    authority = json.loads(AUTHORITY.read_text())
    retire = authority["tritium"]["retirement_authority"]
    retire_path = RUN / "tritium-retirement-authority.json"
    retire_path.write_text(json.dumps(retire, sort_keys=True, indent=2) + "\n")
    assert (
        hashlib.sha256(retire_path.read_bytes()).hexdigest()
        == authority["tritium"]["retirement_file_sha256"]
    )
    kwargs = {
        "authority_file_sha256": hashlib.sha256(retire_path.read_bytes()).hexdigest(),
        "authority_payload_sha256": signed_payload_sha256(retire),
        "reason": "The unlocated stub mixes pedestal-top and separatrix values. Retire it and compose the two independently located quantities.",
        "run_id": RUN.name,
    }
    preview = apply_signed_manifest(retire_path, **kwargs)
    write(RUN / "tritium-retirement-preview.json", preview)
    print(
        "retirement_preview="
        + json.dumps(
            {
                k: v
                for k, v in preview.items()
                if k not in ("manifest", "admitted_actions")
            },
            default=str,
        ),
        flush=True,
    )
    assert not preview.get("refusals"), preview
    receipt = apply_signed_manifest(
        retire_path, apply=True, manifest_sha256=preview["manifest_sha256"], **kwargs
    )
    write(RUN / "tritium-retirement-receipt.json", receipt)
    print(
        "retirement_receipt="
        + json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("manifest", "admitted_actions")
            },
            default=str,
        ),
        flush=True,
    )
    with GraphClient() as gc:
        state = snapshot(gc, [BARE])
        assert state[BARE]["stage"] == "superseded", state
        retry_preview = retry_failed_sources(
            TRITIUM_PATHS, reason=kwargs["reason"], dry_run=True, gc=gc
        )
        write(RUN / "tritium-retry-preview.json", retry_preview)
        assert retry_preview["eligible"] == 2, retry_preview
        retry = retry_failed_sources(TRITIUM_PATHS, reason=kwargs["reason"], gc=gc)
        write(RUN / "tritium-retry-receipt.json", retry)
        assert retry["retried"] == 2, retry
        for path, target in zip(
            TRITIUM_PATHS, authority["tritium"]["compose"], strict=True
        ):
            hint = f"Compose {target}: the local toroidal component of tritium ion fluid velocity at this path's own location. Keep the pedestal-top value separate from the plasma-boundary value. This is a local value, not a flux-surface average."
            preview = set_source_compose_hint(
                path, hint=hint, reason=kwargs["reason"], dry_run=True, gc=gc
            )
            write(RUN / ("hint-preview-" + target + ".json"), preview)
            assert preview["eligible"], preview
            applied = set_source_compose_hint(
                path, hint=hint, reason=kwargs["reason"], gc=gc
            )
            write(RUN / ("hint-receipt-" + target + ".json"), applied)
            assert applied["applied"], applied
    create_sn_run_open(
        RUN.name,
        started_at=datetime.now(UTC),
        cost_limit=2.0,
        invocation="scoped tritium composition",
        invocation_scope=json.dumps(TRITIUM_PATHS),
    )
    mgr = BudgetManager(total_budget=2.0, run_id=RUN.name)
    await mgr.start()
    outcome = {"run_id": RUN.name, "sources": TRITIUM_PATHS}
    try:
        items = await _seed_explicit_paths(TRITIUM_PATHS)
        write(RUN / "tritium-claimed-items.json", items)
        assert len(items) == 2, items
        for item, target in zip(
            sorted(items, key=lambda r: r["path"]),
            authority["tritium"]["compose"],
            strict=True,
        ):
            item["id"] = "dd:" + item["path"]
            item["name_hint"] = target
            item["edit_reason"] = kwargs["reason"]
            item["edit_origin"] = "human"
        outcome["composed"] = await compose_batch(items, mgr, asyncio.Event())
    finally:
        outcome["ledger_drained"] = await mgr.drain_pending(raise_on_failure=True)
        outcome["spend_usd"] = await mgr.get_total_spent(force_refresh=True)
        write(RUN / "tritium-compose-result.json", outcome)
        finalize_sn_run(
            RUN.name,
            status="completed" if "composed" in outcome else "failed",
            cost_spent=outcome["spend_usd"],
            stopped_at=datetime.now(UTC),
        )
    print("composition=" + json.dumps(outcome), flush=True)
    assert outcome["spend_usd"] <= 2.0, outcome


async def compose_held_sources():
    """Finish composition from persisted claims after input-shape repair."""
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.graph_store.graph_audits import finalize_sn_run
    from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
        compose_batch,
    )

    authority = json.loads(AUTHORITY.read_text())
    items = json.loads((RUN / "tritium-claimed-items.json").read_text())
    assert sorted(item["path"] for item in items) == sorted(TRITIUM_PATHS)
    for item, target in zip(
        sorted(items, key=lambda r: r["path"]),
        authority["tritium"]["compose"],
        strict=True,
    ):
        item["id"] = "dd:" + item["path"]
        item["name_hint"] = target
        item["edit_reason"] = (
            "Compose separate local tritium velocities at pedestal top and plasma boundary."
        )
        item["edit_origin"] = "human"
    mgr = BudgetManager(total_budget=2.0, run_id=RUN.name)
    await mgr.start()
    assert await mgr.get_total_spent(force_refresh=True) == 0.0
    previous = RUN / "tritium-compose-result.json"
    (RUN / "tritium-compose-input-refusal.json").write_bytes(previous.read_bytes())
    result = {"run_id": RUN.name, "sources": TRITIUM_PATHS}
    try:
        result["composed"] = await compose_batch(items, mgr, asyncio.Event())
    finally:
        result["ledger_drained"] = await mgr.drain_pending(raise_on_failure=True)
        result["spend_usd"] = await mgr.get_total_spent(force_refresh=True)
        write(previous, result)
        finalize_sn_run(
            RUN.name,
            status="completed" if "composed" in result else "failed",
            cost_spent=result["spend_usd"],
            stopped_at=datetime.now(UTC),
        )
    print("composition=" + json.dumps(result), flush=True)
    assert result["spend_usd"] <= 2.0, result


def readback():
    authority = json.loads(AUTHORITY.read_text())
    census = authority["census"]
    ids = sorted(
        {n for row in census for n in (row["id"], row["target"])}
        | set(authority["tritium"]["compose"])
    )
    receipts = [
        json.loads(path.read_text()) for path in sorted(RUN.glob("reclaim-*.json"))
    ]
    changed = [r["apply"]["successor"] for r in receipts]
    paths = sorted(
        {p.removeprefix("dd:") for r in census for p in r["paths"]} | set(TRITIUM_PATHS)
    )
    with GraphClient() as gc:
        state = snapshot(gc, ids)
        assert len(state) >= 98 and len(census) == 98
        sources = gc.query(
            """
          UNWIND $paths AS path
          OPTIONAL MATCH (s:StandardNameSource {id:'dd:'+path})
          OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(n:StandardName)
          RETURN path,s.id AS id,s.status AS status,s.last_error AS last_error,
            collect(CASE WHEN n IS NULL THEN null ELSE {id:n.id,stage:n.name_stage} END) AS names
        """,
            paths=paths,
        )
        source_by_path = {r["path"]: r for r in sources}
        assert source_by_path["summary/local/separatrix/zeff/value"]["id"]
        cycle_rows = gc.query(
            """
          UNWIND $ids AS id MATCH (n:StandardName {id:id})
          RETURN id, EXISTS { MATCH (n)-[:REFINED_FROM*1..64]->(n) } AS cycle,
            COUNT { (n)-[:REFINED_FROM]->(:StandardName) } AS predecessor_edges
        """,
            ids=ids,
        )
        assert any(r["predecessor_edges"] for r in cycle_rows), (
            "lineage instrument sees no known edge"
        )
        cycles = sorted(r["id"] for r in cycle_rows if r["cycle"])
        changed_cycles = sorted(set(cycles) & set(changed))
        costs = gc.query(
            """
          MATCH (r:SNRun {id:$id})
          OPTIONAL MATCH (c:LLMCost {run_id:$id})
          RETURN r.id AS run_id,count(c) AS events,count(c.llm_cost) AS priced_events,
            coalesce(sum(c.llm_cost),0.0) AS usd,r.status AS status
        """,
            id=RUN.name,
        )
        remaining = []
        conforms = []
        for row in census:
            old = state.get(row["id"])
            target = state.get(row["target"])
            live_old = old and old["stage"] != "superseded"
            live_target = target and target["stage"] != "superseded"
            lexical = bool(
                (live_old and row["id"] == row["target"])
                or (not live_old and live_target)
            )
            bindings = all(
                any(
                    n["id"] == row["target"] and n["stage"] != "superseded"
                    for n in source_by_path[p.removeprefix("dd:")]["names"]
                )
                for p in row["paths"]
            )
            result = {
                **row,
                "conforms": lexical,
                "all_recorded_sources_on_target": bindings,
            }
            conforms.append(result)
            if not lexical:
                result["lineage"] = lineage(gc, row["target"])
                remaining.append(result)
        tritium_state = {
            n: state.get(n) for n in [BARE, *authority["tritium"]["compose"]]
        }
        produced = [
            any(
                n["id"] == target and n["stage"] != "superseded"
                for n in source_by_path[path]["names"]
            )
            for path, target in zip(
                TRITIUM_PATHS, authority["tritium"]["compose"], strict=True
            )
        ]
        retirement = gc.query(
            "MATCH (c:StandardNameChange {run_id:$id}) WHERE c.operation=$operation RETURN properties(c) AS receipt",
            id=RUN.name,
            operation="retire_unlocated_tritium_stub",
        )
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "census_count": 98,
        "conforming": sum(r["conforms"] for r in conforms),
        "conforming_with_all_recorded_sources": sum(
            r["conforms"] and r["all_recorded_sources_on_target"] for r in conforms
        ),
        "non_conforming": remaining,
        "census_readback": conforms,
        "sources": sources,
        "tritium": tritium_state,
        "tritium_producing_sources": produced,
        "retirement_receipts": retirement,
        "costs": costs,
        "cycle_names_in_cohort": cycles,
        "cycles_through_reclaimed_names": changed_cycles,
        "reclaim_receipts": len(receipts),
    }
    result["complete"] = bool(
        len(receipts) == 7
        and not changed_cycles
        and all(produced)
        and retirement
        and costs
        and costs[0]["usd"] <= 2.0
    )
    write(RUN / "readback.json", result)
    print(
        "SUMMARY "
        + json.dumps(
            {
                k: result[k]
                for k in (
                    "census_count",
                    "conforming",
                    "conforming_with_all_recorded_sources",
                    "reclaim_receipts",
                    "tritium_producing_sources",
                    "cycle_names_in_cohort",
                    "cycles_through_reclaimed_names",
                    "costs",
                    "complete",
                )
            }
        ),
        flush=True,
    )
    assert not changed_cycles, changed_cycles
    assert (
        costs
        and costs[0]["events"] == costs[0]["priced_events"]
        and costs[0]["usd"] <= 2.0
    ), costs
    assert result["complete"], (
        "Read-back records an incomplete tritium outcome; inspect the source-specific refusals."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=[
            "prepare",
            "apply",
            "apply-all",
            "tritium-authority",
            "tritium",
            "readback",
            "compose-held",
        ],
    )
    p.add_argument("--name")
    args = p.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"# revision: {revision}; tree: {Path.cwd()}; command: "
        + shlex.join(["uv", "run", "--no-sync", "python", *sys.argv]),
        flush=True,
    )
    print("module=" + edit.__file__, flush=True)
    print("cwd=" + str(Path.cwd().resolve()), flush=True)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "apply":
        apply_one(args.name)
    elif args.mode == "tritium-authority":
        tritium_authority()
    elif args.mode == "tritium":
        asyncio.run(tritium())
    elif args.mode == "readback":
        readback()
    elif args.mode == "compose-held":
        asyncio.run(compose_held_sources())
    else:
        authority = json.loads(AUTHORITY.read_text())
        for row in authority["operations"]:
            if row["action"] == "reclaim":
                apply_one(row["id"])


if __name__ == "__main__":
    main()

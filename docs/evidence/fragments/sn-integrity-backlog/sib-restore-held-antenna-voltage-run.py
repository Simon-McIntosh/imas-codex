"""Return ``voltage_of_diagnostic_antenna`` to the state it held before its retirement.

The name was retired on 2026-10-29T08:57:38Z by a signed ``supersede`` row of the
``sib-retire-sourceless-drafts`` manifest. Its recorded prior stage is
``superseded_from_stage='exhausted'``; the retiring run read ``status='draft'``,
``source_paths=[]`` and no producer immediately before it wrote. The lead's
2026-10-29 ruling keeps the other 130 retirements and restores this one, because
§38 holds it from release.

The repository has no command that un-supersedes a tombstone on its own: ``sn
restore`` reconstructs an absent identity from an archive, and ``sn supersede
--into`` revives a tombstone only as the target of a fold that carries a source
onto it. A revival restores what the repository itself restores in that fold: the
recorded prior stage, and ``status='draft'`` because the tombstone's
``superseded`` status keeps a live identity out of export (edit.py
``_fold_revival_stage``). The signed path here writes exactly those two
properties, once, through ``apply_signed_manifest``.

The preview refuses any authority whose rows name an identity other than the one
authorised. The overreach this node answers was a 133-row authority applied where
19 were approved, so the scope check is the guard this write needs.

Modes: state | snapshot | authority | preview | apply --digest | readback | negative
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "voltage_of_diagnostic_antenna"
AUTHORITY = HERE / "sib-restore-held-antenna-voltage-authority.json"
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261029T164430537469-sib-restore-held-antenna-voltage"
)
STATE = RUN / "state.json"
BEFORE = RUN / "names-before.json"

# The stage and catalog status the retirement replaced. The stage is the node's
# own recorded prior stage; the status is what the retiring run read before it
# wrote, and what a revival writes back.
PRIOR = {"name_stage": "exhausted", "status": "draft"}

SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
OPERATION_ID = "restore-retired-name-to-recorded-prior-stage"
REASON = (
    "the retirement of this name is reversed by lead ruling; restore the stage "
    "and catalog status it held before the signed retirement"
)
RESTORE_OPERATION = "restore_prior_stage"


def _graph():
    from imas_codex.graph.client import GraphClient

    return GraphClient()


def _row(name: str, properties: dict | None = None) -> dict:
    return {
        "id": name,
        "identity": {"id": name, "kind": "standard_name", "target_id": name},
        "participants": [{"id": name, "kind": "node", "graph_label": "StandardName"}],
        "mutations": [
            {
                "id": f"{name}:restore-prior-stage",
                "order": 0,
                "kind": "set_properties",
                "participant_id": name,
                "arguments": {"properties": dict(properties or PRIOR)},
            }
        ],
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
                "participant_ids": [name],
            },
        ],
        "orphan_policy": "refuse",
        "selection": dict(SELECTION),
    }


def read_state() -> dict:
    with _graph() as graph:
        node = graph.query(
            """
            MATCH (n:StandardName {id: $id})
            RETURN n.id AS id, n.name_stage AS stage, n.status AS status,
                   n.superseded_from_stage AS prior_stage,
                   n.source_paths AS source_paths,
                   n.superseded_by AS superseded_by,
                   n.docs_stage AS docs_stage,
                   n.validation_status AS validation_status
            """,
            id=NAME,
        )
        if not node:
            raise SystemExit(f"{NAME} is absent from the live graph")
        edges = graph.query(
            """
            MATCH (n:StandardName {id: $id})-[r]->(m)
            RETURN type(r) AS type, count(*) AS n ORDER BY type
            """,
            id=NAME,
        )
        incoming = graph.query(
            """
            MATCH (n:StandardName {id: $id})<-[r]-(m)
            RETURN type(r) AS type, count(*) AS n ORDER BY type
            """,
            id=NAME,
        )
        receipts = graph.query(
            """
            MATCH (c:StandardNameChange)
            WHERE c.row_id = $id
            RETURN c.id AS id, c.operation AS operation,
                   c.manifest_sha256 AS manifest, c.changed_at AS changed_at
            ORDER BY c.changed_at
            """,
            id=NAME,
        )
        successor = graph.query(
            """
            MATCH (n:StandardName {id: $id})-[:HAS_SUCCESSOR]->(s:StandardName)
            RETURN s.id AS successor
            """,
            id=NAME,
        )
        producers = graph.query(
            """
            MATCH (n:StandardName {id: $id})<-[:PRODUCED_NAME]-(s:StandardNameSource)
            RETURN s.source_id AS source_id, s.status AS status ORDER BY source_id
            """,
            id=NAME,
        )
    state = {
        "id": NAME,
        "observed": dict(node[0]),
        "edges": [dict(r) for r in edges],
        "incoming": [dict(r) for r in incoming],
        "change_receipts": [dict(r) for r in receipts],
        "successor_edges": [dict(r) for r in successor],
        "producers": [dict(r) for r in producers],
        "prior_state": {
            "name_stage": "exhausted",
            "status": "draft",
            "source_paths": [],
            "producers": 0,
            "successor": None,
            "source": (
                "the sib-retire-sourceless-drafts run stream's pre-write read of "
                "2026-10-29 (stage exhausted, status draft, producers 0, "
                "source_paths []), and the node's own superseded_from_stage="
                "'exhausted' written by the retirement"
            ),
        },
    }
    RUN.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, default=str) + "\n")
    return state


def _names_digest(rows: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(f"{r['id']}\t{r['stage']}\t{r['status']}" for r in rows).encode()
    ).hexdigest()


def snapshot_names() -> dict:
    """Digest every live name's lifecycle pair, before the write."""
    with _graph() as graph:
        rows = graph.query(
            """
            MATCH (n:StandardName)
            RETURN n.id AS id, coalesce(n.name_stage, '') AS stage,
                   coalesce(n.status, '') AS status
            ORDER BY id
            """
        )
    payload = {"count": len(rows), "digest": _names_digest(rows)}
    (RUN / "names-before.json").write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def build_authority(rows: list[dict], path: Path) -> dict:
    from imas_codex.standard_names.repair_authority import build_repair_authority

    spec = {
        "operation_id": OPERATION_ID,
        "authority_mode": "external_reviewed",
        "orphan_policy": "refuse",
        "selection": dict(SELECTION),
        "receipt_policy": {
            "id": "one-per-restored-identity",
            "operation": RESTORE_OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "rows": rows,
    }
    built = build_repair_authority(spec)
    path.write_bytes(built.content)
    return {
        "authority": str(path),
        "rows": len(rows),
        "row_ids": [r["id"] for r in rows],
        "authority_file_sha256": built.file_sha256,
    }


def _scope_refusal(authority: dict) -> str | None:
    rows = authority.get("rows") or []
    named = {str(r.get("id")) for r in rows}
    participants = {
        str(p.get("id")) for r in rows for p in (r.get("participants") or [])
    }
    outside = sorted((named | participants) - {NAME})
    if outside:
        return (
            "restore scope refused: this authority restores exactly "
            f"{NAME!r}; it also names {outside}"
        )
    if len(rows) != 1:
        return (
            f"restore scope refused: exactly one row is admitted; this has {len(rows)}"
        )
    return None


def _call(apply: bool, digest: str | None, path: Path = AUTHORITY) -> dict:
    from imas_codex.standard_names.signed_manifest import (
        apply_signed_manifest,
        signed_payload_sha256,
    )

    raw = path.read_bytes()
    authority = json.loads(raw)
    refusal = _scope_refusal(authority)
    if refusal is not None:
        return {"status": "refused", "refusal": refusal, "authority": str(path)}
    receipt = apply_signed_manifest(
        path,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(authority),
        reason=REASON,
        apply=apply,
        manifest_sha256=digest,
    )
    return {
        k: v for k, v in receipt.items() if k not in {"manifest", "admitted_actions"}
    }


def readback() -> dict:
    """The post-write state, both receipts, and whether any other name moved."""
    from imas_codex.standard_names.source_invariant import find_sourceless_live_names

    raw = AUTHORITY.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    with _graph() as graph:
        node = graph.query(
            """
            MATCH (n:StandardName {id: $id})
            RETURN n.name_stage AS stage, n.status AS status,
                   n.superseded_from_stage AS prior_stage,
                   n.source_paths AS source_paths
            """,
            id=NAME,
        )[0]
        successor = graph.query(
            """
            MATCH (n:StandardName {id: $id})-[:HAS_SUCCESSOR]->(s:StandardName)
            RETURN s.id AS successor
            """,
            id=NAME,
        )
        receipts = graph.query(
            """
            MATCH (c:StandardNameChange)
            WHERE c.row_id = $id
            RETURN c.id AS id, c.operation AS operation,
                   c.manifest_sha256 AS manifest, c.changed_at AS changed_at,
                   c.authority_file_sha256 AS authority
            ORDER BY c.changed_at
            """,
            id=NAME,
        )
        rows = graph.query(
            """
            MATCH (n:StandardName)
            RETURN n.id AS id, coalesce(n.name_stage, '') AS stage,
                   coalesce(n.status, '') AS status
            ORDER BY id
            """
        )
        # Every identity this authority wrote. One row is the whole scope, so
        # the engine's own ledger is the proof that no other name moved.
        written = graph.query(
            """
            MATCH (c:StandardNameChange)
            WHERE c.authority_file_sha256 = $sha
            RETURN c.row_id AS row_id, c.operation AS operation
            ORDER BY row_id
            """,
            sha=file_sha,
        )
        live = [r["id"] for r in rows]
        sourceless = find_sourceless_live_names(graph, live)
    digest = _names_digest(rows)
    before = json.loads(BEFORE.read_text()) if BEFORE.exists() else {}
    return {
        "authority_file_sha256": file_sha,
        "state": dict(node),
        "successor_edges": [dict(r) for r in successor],
        "change_receipts": [dict(r) for r in receipts],
        "identities_written_by_this_authority": [dict(r) for r in written],
        "live_names": len(rows),
        "names_digest": digest,
        "names_digest_before": before.get("digest"),
        "names_count_before": before.get("count"),
        "names_digest_moved": None
        if not before.get("digest")
        else before["digest"] != digest,
        "named_live": NAME in live,
        "sourceless_live_count": len(sourceless),
        "sourceless_includes_name": NAME in sourceless,
    }


def gate() -> dict:
    """Show the release gate holding the restored name, and the source audit.

    The identity hold is read before any eligibility test, so a restored name
    leaves through the release-hold path whatever its stage. The source audit
    counts a live name with no producer, so the measurement is that this name
    lands in its dormant set with no blocking source loss.
    """
    from imas_codex.standard_names.export import (
        _RELEASE_IDENTITY_HOLDS,
        _classify_export_population,
    )
    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )

    with _graph() as graph:
        node = graph.query(
            "MATCH (n:StandardName {id: $id}) RETURN n.name_stage AS name_stage",
            id=NAME,
        )[0]
        eligible, excluded = _classify_export_population(
            [{"id": NAME, "name_stage": node["name_stage"]}],
            domain=None,
            names_only=False,
        )
        audit = classify_sourceless_standard_names(graph)
    mine = [row for row in audit if row["id"] == NAME]
    losses = [
        row
        for row in audit
        if row["population"] in {"dropped_edge", "never_bound"}
        and row["name_stage"] != "exhausted"
        and row["id"] not in _RELEASE_IDENTITY_HOLDS
    ]
    return {
        "hold_declared": list(_RELEASE_IDENTITY_HOLDS[NAME]),
        "export_eligible_rows": eligible,
        "export_exclusions": [
            dict(e.__dict__) if hasattr(e, "__dict__") else e for e in excluded
        ],
        "export_exclusion_count": len(excluded),
        "source_audit_row": [dict(r) for r in mine],
        "blocking_source_losses": len(losses),
        "blocking_source_loss_ids": sorted(row["id"] for row in losses),
    }


def negative() -> dict:
    """A second identity in the authority must be refused by the preview."""
    second = "voltage_of_spectrometer_channel"
    path = RUN / "negative-two-name-authority.json"
    built = build_authority(
        [_row(NAME), _row(second, {"validation_status": "valid"})], path
    )
    return {"built": built, "preview": _call(False, None, path)}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "state"
    if mode == "all":
        print("## readback")
        print(json.dumps(readback(), indent=1, default=str)[:8000])
        print("## gate")
        print(json.dumps(gate(), indent=1, default=str)[:8000])
        print("## negative")
        print(json.dumps(negative(), indent=1, default=str)[:4000])
    elif mode == "state":
        print(json.dumps(read_state(), indent=1, default=str))
    elif mode == "snapshot":
        print(json.dumps(snapshot_names(), indent=1))
    elif mode == "authority":
        print(json.dumps(build_authority([_row(NAME)], AUTHORITY), indent=1))
    elif mode == "preview":
        print(json.dumps(_call(False, None), indent=1, default=str)[:8000])
    elif mode == "apply":
        print(json.dumps(_call(True, sys.argv[2]), indent=1, default=str)[:8000])
    elif mode == "readback":
        print(json.dumps(readback(), indent=1, default=str)[:8000])
    elif mode == "gate":
        print(json.dumps(gate(), indent=1, default=str)[:8000])
    elif mode == "negative":
        print(json.dumps(negative(), indent=1, default=str)[:4000])
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()

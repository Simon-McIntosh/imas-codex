"""Give the wall potential its DD source back and retire the outboard-midplane name.

Phases, each writing its own log whose final line is a real ``EXIT=``:

``read``        Read the four identities, their producers and the three DD
                paths; carry a positive and a negative control on the same read.
``exchange``    Show that the ruled source move is not expressible through the
                ``dd-source-exchange`` adapter. Both attempts are previews.
``authorities`` Build the retirement authority through ``build_repair_authority``
                (never by hand) and record the migration's compare-and-set
                manifest beside the exchange refusals.
``preview``     Preview the retirement to ``would_apply``, and refuse a stale
                authority and a stale compare-and-set before the real writes.
``apply``       Apply the retirement under its preview digest, then move the
                wall path through ``retarget_standard_name_sources``.
``readback``    Re-read the four identities, the successor edge, both receipts
                and the sourceless classifier.
``harmonize``   ``sn harmonize --family <locus> --report --json``; never applied.
``record``      Compose the fragment HTML from the artifacts on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261028T202206786051-shc-locus-wall-rebind-midplane-retire"
)
LOGS = RUN / "logs"
PYTHON = Path("/home/ITER/mcintos/Code/imas-codex/.venv/bin/python")

FRAGMENT_HTML = HERE / "shc-locus-wall-rebind-midplane-retire.html"
EXCHANGE_AUTHORITY = (
    HERE / "shc-locus-wall-rebind-midplane-retire-exchange-authority.json"
)
RETIREMENT_AUTHORITY = (
    HERE / "shc-locus-wall-rebind-midplane-retire-retirement-authority.json"
)

BASE = "electrostatic_potential"
WALL = "electrostatic_potential_at_wall"
MIDPLANE = "electrostatic_potential_at_outboard_midplane"
SUCCESSOR = "electrostatic_potential_at_langmuir_probe_position"
WALL_PATH = "dd:wall/description_ggd/ggd/phi_potential/values"
#: The six bindings the base keeps; the wall path is the seventh.
KEPT_PATHS = [
    "dd:core_profiles/profiles_1d/phi_potential",
    "dd:edge_profiles/ggd/phi_potential/values",
    "dd:edge_profiles/profiles_1d/phi_potential",
    "dd:mhd/ggd/phi_potential/values",
    "dd:plasma_profiles/ggd/phi_potential/values",
    "dd:plasma_profiles/profiles_1d/phi_potential",
]
PLUNGE_PATHS = [
    "dd:langmuir_probes/reciprocating/plunge/potential_plasma",
    "dd:langmuir_probes/reciprocating/plunge/v_plasma",
]
FAMILY = [BASE, WALL, MIDPLANE, SUCCESSOR]

SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
RETIRE_OPERATION_ID = "retire-outboard-midplane-in-favour-of-langmuir-probe-position"
RETIRE_OPERATION = "link_signed_successor"
RETIRE_ROW_ID = f"{MIDPLANE}=>{SUCCESSOR}"
RETIRE_REASON = (
    "the DD records a reciprocating-probe plunge, not an outboard-midplane "
    "location; retire the exhausted identity onto the name that already holds "
    "the plunge potential, keeping its paid review history"
)
MIGRATE_REASON = (
    "a potential stored on the wall GGD is the potential at the wall; the ruling "
    "moves wall/description_ggd/ggd/phi_potential/values to "
    "electrostatic_potential_at_wall"
)
RUN_ID = os.environ.get(
    "SHC_LOCUS_RUN_ID", "r-20261028T202206786051-shc-locus-wall-rebind-midplane-retire"
)


class Log:
    """Append-only log whose final line is one real ``EXIT=`` line."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("w")

    def write(self, line: str) -> None:
        self._fh.write(str(line).rstrip("\n") + "\n")
        self._fh.flush()

    def block(self, text: str) -> None:
        for line in str(text).splitlines():
            self.write(line)

    def close(self, code: int) -> None:
        self.write(f"EXIT={code}")
        self._fh.close()


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _client():
    from imas_codex.graph import GraphClient

    return GraphClient()


# ── phase: read ───────────────────────────────────────────────────────────
NAMES_QUERY = """
// SHC_LOCUS_READ_NAMES
UNWIND $ids AS id
MATCH (sn:StandardName {id: id})
RETURN id,
       sn.name_stage AS name_stage,
       sn.status AS status,
       sn.validation_status AS validation_status,
       sn.unit AS unit,
       sn.reviewer_score_docs AS docs_score,
       sn.source_paths AS source_paths,
       sn.claim_token AS claim_token,
       'source_paths' IN keys(sn) AS control_present,
       'shc_never_written_property' IN keys(sn) AS control_absent,
       [(sn)<-[:PRODUCED_NAME]-(s:StandardNameSource) |
         {id: s.id, status: s.status, source_type: s.source_type,
          claim_token: s.claim_token, produced_sn_id: s.produced_sn_id}]
         AS producers,
       [(sn)-[:HAS_HOLD]->(h:ReviewHold) WHERE h.status = 'open' |
         {id: h.id, axis: h.axis}] AS open_holds,
       COUNT { (sn)-[:HAS_SUCCESSOR]->(:StandardName) } AS out_successors,
       [(sn)-[:HAS_SUCCESSOR]->(n:StandardName) | n.id] AS successors,
       COUNT { (child:StandardName)-[:HAS_PARENT]->(sn) } AS parents
ORDER BY id
"""

PATHS_QUERY = """
// SHC_LOCUS_READ_PATHS
UNWIND $paths AS p
OPTIONAL MATCH (s:StandardNameSource {id: p})
OPTIONAL MATCH (dd:IMASNode {id: substring(p, 3)})
RETURN p,
       s IS NOT NULL AS source_present,
       s.status AS source_status,
       s.produced_sn_id AS scalar_target,
       [(s)-[:PRODUCED_NAME]->(n:StandardName) | n.id] AS holders,
       dd IS NOT NULL AS dd_present,
       dd.lifecycle_status AS dd_lifecycle_status,
       [(dd)-[:HAS_STANDARD_NAME]->(n:StandardName) | n.id] AS projections
ORDER BY p
"""


def read_state(gc) -> dict:
    names = [dict(row) for row in gc.query(NAMES_QUERY, ids=FAMILY)]
    paths = [
        dict(row) for row in gc.query(PATHS_QUERY, paths=[WALL_PATH, *PLUNGE_PATHS])
    ]
    by_id = {row["id"]: row for row in names}
    paths_by_id = {row["p"]: row for row in paths}
    bound = sorted(source["id"] for source in by_id.get(BASE, {}).get("producers", []))
    return {
        "names": names,
        "paths": paths,
        "by_id": by_id,
        "paths_by_id": paths_by_id,
        "base_bindings": bound,
        "base_binding_count": len(bound),
        "kept_missing": sorted(set(KEPT_PATHS) - set(bound)),
        "wall_held_by": paths_by_id.get(WALL_PATH, {}).get("holders") or [],
        "missing_names": sorted(set(FAMILY) - set(by_id)),
    }


def check_state(state: dict) -> list[str]:
    """Return precondition failures; empty means the ruled state holds."""
    by_id = state["by_id"]
    paths = state["paths_by_id"]
    if state["missing_names"]:
        return [f"names absent from the graph: {state['missing_names']!r}"]
    problems: list[str] = []
    if by_id[BASE]["name_stage"] != "accepted":
        problems.append(f"{BASE} stage is {by_id[BASE]['name_stage']!r}")
    if state["base_binding_count"] != 7:
        problems.append(
            f"{BASE} has {state['base_binding_count']} producing sources, expected 7"
        )
    if state["kept_missing"]:
        problems.append(
            f"{BASE} is missing expected bindings {state['kept_missing']!r}"
        )
    if state["wall_held_by"] != [BASE]:
        problems.append(f"wall path holders are {state['wall_held_by']!r}")
    if by_id[WALL]["producers"]:
        problems.append(f"{WALL} already has producers")
    if by_id[MIDPLANE]["name_stage"] not in {"exhausted", "superseded"}:
        problems.append(f"{MIDPLANE} stage is {by_id[MIDPLANE]['name_stage']!r}")
    if by_id[MIDPLANE]["producers"]:
        problems.append(f"{MIDPLANE} already has producers")
    if by_id[MIDPLANE]["out_successors"]:
        problems.append(f"{MIDPLANE} already carries a successor")
    if by_id[MIDPLANE]["parents"]:
        problems.append(f"{MIDPLANE} still has {by_id[MIDPLANE]['parents']} children")
    if by_id[SUCCESSOR]["name_stage"] != "accepted":
        problems.append(f"{SUCCESSOR} stage is {by_id[SUCCESSOR]['name_stage']!r}")
    for name_id in FAMILY:
        row = by_id[name_id]
        if row["claim_token"] is not None:
            problems.append(f"{name_id} carries claim token {row['claim_token']!r}")
        if row["open_holds"]:
            problems.append(f"{name_id} has open holds {row['open_holds']!r}")
        for producer in row["producers"]:
            if producer.get("claim_token") is not None:
                problems.append(
                    f"{producer['id']} carries claim token {producer['claim_token']!r}"
                )
    for path in (WALL_PATH, *PLUNGE_PATHS):
        if not paths[path]["source_present"]:
            problems.append(f"source node for {path} is absent")
    if paths[WALL_PATH].get("dd_lifecycle_status") == "removed":
        problems.append("the wall DD path is removed from the current DD")
    return problems


def phase_read() -> int:
    log = Log(LOGS / "read.log")
    log.write(f"# command: {sys.executable} {Path(__file__).name} --phase read")
    log.write(f"# worktree: {WORKTREE}")
    log.write("# writes nothing: reads the live graph and reports")
    rc = 0
    gc = _client()
    try:
        state = read_state(gc)
        for key in ("names", "paths"):
            log.write(f"{key}={_json(state[key])}")
        log.write(f"base_binding_count={state['base_binding_count']}")
        problems = check_state(state)
        controls = {
            "rows_read": len(state["names"]),
            "positive_control_rows": sum(
                1 for row in state["names"] if row["control_present"]
            ),
            "negative_control_rows": sum(
                1 for row in state["names"] if row["control_absent"]
            ),
        }
        log.write(f"read_controls={_json(controls)}")
        if controls["positive_control_rows"] != len(FAMILY):
            problems.append("positive read control did not see source_paths everywhere")
        if controls["negative_control_rows"] != 0:
            problems.append("negative read control saw an impossible property")
        log.write(f"precondition_problems={problems}")
        log.write("VERDICT=" + ("proceed" if not problems else "blocked"))
        (LOGS / "read-state.json").write_text(_json(state) + "\n")
        if problems:
            rc = 2
    except Exception as exc:  # noqa: BLE001
        log.write(f"read_error={type(exc).__name__}: {exc}")
        rc = 2
    finally:
        gc.close()
    log.close(rc)
    return rc


# ── phase: exchange (the adapter's applicability, previewed) ──────────────
def exchange_authority(rows: list[dict]) -> tuple[bytes, str, str]:
    from imas_codex.standard_names import signed_manifest as operator
    from imas_codex.standard_names.repair_authority import build_repair_authority

    built = build_repair_authority(
        {
            "operation_id": operator._SOURCE_EXCHANGE_OPERATION,
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
    )
    return built.content, built.file_sha256, built.payload_sha256


def phase_exchange() -> int:
    from imas_codex.standard_names import signed_manifest as operator

    log = Log(LOGS / "exchange.log")
    log.write(f"# command: {sys.executable} {Path(__file__).name} --phase exchange")
    log.write("# previews only: every call passes apply=False and writes nothing")
    rc = 0
    attempts: list[dict] = []
    tries = {
        "transfer_without_replacement": {
            "id": BASE,
            "name_id": BASE,
            "given_source_id": WALL_PATH,
            "receiving_name_id": WALL,
            "selection": dict(SELECTION),
        },
        "transfer_with_a_held_replacement": {
            "id": BASE,
            "name_id": BASE,
            "given_source_id": WALL_PATH,
            "receiving_name_id": WALL,
            "taken_source_id": "dd:edge_profiles/ggd/phi_potential/values",
            "selection": dict(SELECTION),
        },
    }
    gc = _client()
    try:
        for label, row in tries.items():
            attempt: dict = {"label": label, "row": row}
            try:
                content, file_sha, payload_sha = exchange_authority([row])
                path = LOGS / f"exchange-{label}.json"
                path.write_bytes(content)
                attempt["authority_path"] = str(path)
                result = operator.apply_signed_manifest(
                    path,
                    authority_adapter=operator._SOURCE_EXCHANGE_ADAPTER,
                    authority_file_sha256=file_sha,
                    authority_payload_sha256=payload_sha,
                    guard_set=operator._SOURCE_EXCHANGE_GUARDS,
                    reason="show the adapter's treatment of a pure source transfer",
                    apply=False,
                    gc=gc,
                )
                attempt["outcome"] = result.get("outcome")
                attempt["refusals"] = result.get("refusals")
            except Exception as exc:  # noqa: BLE001
                attempt["outcome"] = "refused"
                attempt["refusal_text"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
            log.write(f"attempt_{label}={_json(attempt)}")
        accepted = [a for a in attempts if a.get("outcome") == "would_apply"]
        log.write(f"exchange_attempts_admitted={len(accepted)}")
        log.write(
            "verdict="
            + ("transfer-not-expressible" if not accepted else "transfer-expressible")
        )
        (LOGS / "exchange-attempts.json").write_text(_json(attempts) + "\n")
        if accepted:
            rc = 3
    except Exception as exc:  # noqa: BLE001
        log.write(f"exchange_error={type(exc).__name__}: {exc}")
        rc = 2
    finally:
        gc.close()
    log.close(rc)
    return rc


# ── phase: authorities ────────────────────────────────────────────────────
def retirement_row() -> dict:
    """One signed supersede that names the successor the ruling chose."""
    return {
        "id": RETIRE_ROW_ID,
        "identity": {
            "id": RETIRE_ROW_ID,
            "kind": "standard_name",
            "target_id": MIDPLANE,
        },
        "participants": [
            {"id": MIDPLANE, "kind": "node", "graph_label": "StandardName"},
            {"id": SUCCESSOR, "kind": "node", "graph_label": "StandardName"},
        ],
        "selection": dict(SELECTION),
        "mutations": [
            {
                "id": f"{MIDPLANE}:supersede",
                "order": 0,
                "kind": "supersede",
                "participant_id": MIDPLANE,
                "arguments": {"successor_id": SUCCESSOR},
            }
        ],
        "guards": [
            {
                "id": f"{RETIRE_ROW_ID}:structural-legitimacy",
                "kind": "semantic_authority",
                "implementation": "structural-legitimacy",
                "participant_ids": [MIDPLANE],
            },
            {
                "id": f"{RETIRE_ROW_ID}:out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [MIDPLANE, SUCCESSOR],
            },
        ],
        "orphan_policy": "refuse",
    }


def retirement_spec() -> dict:
    return {
        "operation_id": RETIRE_OPERATION_ID,
        "authority_mode": "external_reviewed",
        "orphan_policy": "refuse",
        "selection": dict(SELECTION),
        "receipt_policy": {
            "id": "one-per-superseded-identity",
            "operation": RETIRE_OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "rows": [retirement_row()],
    }


def phase_authorities() -> int:
    from imas_codex.standard_names.repair_authority import build_repair_authority

    log = Log(LOGS / "authorities.log")
    log.write(f"# command: {sys.executable} {Path(__file__).name} --phase authorities")
    rc = 0
    try:
        built = build_repair_authority(retirement_spec())
        RETIREMENT_AUTHORITY.write_bytes(built.content)
        emitted = json.loads(built.content)
        assert emitted["signature"]["sha256"] == built.payload_sha256
        retirement = {
            "path": str(RETIREMENT_AUTHORITY),
            "file_sha256": built.file_sha256,
            "payload_sha256": built.payload_sha256,
            "row_ids": [row["id"] for row in emitted["rows"]],
            "successor": SUCCESSOR,
        }
        log.write(f"retirement_authority={_json(retirement)}")

        migration_manifest = {
            "expected_current_bindings": {WALL_PATH: BASE},
            "new_name": WALL,
            "old_name": BASE,
            "operation": "source_migration_manifest",
            "source_ids": [WALL_PATH],
        }
        migration_sha = hashlib.sha256(
            json.dumps(
                migration_manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        attempts = json.loads((LOGS / "exchange-attempts.json").read_text())
        record = {
            "node": "shc-locus-wall-rebind-midplane-retire",
            "ruled_direction": f"{BASE} -> {WALL}",
            "source_id": WALL_PATH,
            "adapter_named_by_the_ruling": "dd-source-exchange",
            "adapter_outcomes": [a.get("outcome") for a in attempts],
            "adapter_verdict": "transfer-not-expressible",
            "adapter_attempts": attempts,
            "applied_writer": (
                "imas_codex.standard_names.provenance_lifecycle."
                "retarget_standard_name_sources"
            ),
            "migration_manifest": migration_manifest,
            "migration_manifest_sha256": migration_sha,
        }
        EXCHANGE_AUTHORITY.write_text(_json(record) + "\n")
        log.write(
            "exchange_authority="
            + _json({k: v for k, v in record.items() if k != "adapter_attempts"})
        )
        (LOGS / "authorities.json").write_text(
            _json({"retirement": retirement, "migration": migration_manifest}) + "\n"
        )
    except Exception as exc:  # noqa: BLE001
        log.write(f"authorities_error={type(exc).__name__}: {exc}")
        rc = 2
    log.close(rc)
    return rc


# ── phase: preview (with the declared stale-authority control) ────────────
def preview_retirement(gc, path: Path) -> dict:
    from imas_codex.standard_names.signed_manifest import (
        apply_signed_manifest,
        signed_payload_sha256,
    )

    raw = path.read_bytes()
    return apply_signed_manifest(
        path,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(json.loads(raw)),
        reason=RETIRE_REASON,
        apply=False,
        run_id=RUN_ID,
        gc=gc,
    )


def phase_preview() -> int:
    from imas_codex.standard_names.provenance_lifecycle import (
        retarget_standard_name_sources,
    )

    log = Log(LOGS / "preview.log")
    log.write(f"# command: {sys.executable} {Path(__file__).name} --phase preview")
    log.write("# previews only: apply=False on the retirement, and a CAS preflight")
    rc = 0
    gc = _client()
    try:
        preview = preview_retirement(gc, RETIREMENT_AUTHORITY)
        log.write(
            "retirement_preview="
            + _json({k: v for k, v in preview.items() if k != "manifest"})
        )
        (LOGS / "retirement-preview.json").write_text(_json(preview) + "\n")
        if preview.get("outcome") != "would_apply":
            log.write("retirement preview did not report would_apply")
            log.close(2)
            return 2

        # declared control: a stale authority must be refused by the preview.
        stale = Log(LOGS / "stale-authority-refusal.log")
        stale_path = LOGS / "retirement-authority-stale.json"
        raw = json.loads(RETIREMENT_AUTHORITY.read_text())
        raw["rows"][0]["mutations"][0]["arguments"]["successor_id"] = BASE
        stale_path.write_text(_json(raw) + "\n")
        try:
            preview_retirement(gc, stale_path)
        except Exception as exc:  # noqa: BLE001
            stale.write(
                "stale_authority: rows[0].mutations[0].arguments.successor_id "
                f"-> {BASE} (a live name with bound sources, not the ruled successor)"
            )
            stale.write(f"refused={type(exc).__name__}: {exc}")
            stale.close(1)
            log.write("stale_authority_control=refused (EXIT=1)")
        else:
            stale.write("preview accepted a stale authority")
            stale.close(3)
            log.write("stale_authority_control=ACCEPTED-STALE")
            rc = 3

        # the migration's compare-and-set, refused with the wrong predecessor.
        cas = Log(LOGS / "migration-stale-cas-refusal.log")
        try:
            retarget_standard_name_sources(
                gc,
                WALL,
                BASE,
                operation="source_migration_manifest",
                reason="negative control: the ruling's direction, reversed",
                origin="signed_manifest",
                run_id=RUN_ID,
                source_ids=[WALL_PATH],
                expected_current_bindings={WALL_PATH: WALL},
            )
        except Exception as exc:  # noqa: BLE001
            cas.write("stale_cas: old_name=WALL new_name=BASE expected=WALL")
            cas.write(f"refused={type(exc).__name__}: {exc}")
            cas.close(1)
            log.write("stale_cas_control=refused (EXIT=1)")
        else:
            cas.write("migration accepted a wrong compare-and-set")
            cas.close(3)
            log.write("stale_cas_control=ACCEPTED-STALE")
            rc = 3
    except Exception as exc:  # noqa: BLE001
        log.write(f"preview_error={type(exc).__name__}: {exc}")
        rc = 2
    finally:
        gc.close()
    log.close(rc)
    return rc


# ── phase: apply ──────────────────────────────────────────────────────────
def phase_apply() -> int:
    from imas_codex.standard_names.provenance_lifecycle import (
        retarget_standard_name_sources,
    )
    from imas_codex.standard_names.signed_manifest import (
        apply_signed_manifest,
        signed_payload_sha256,
    )

    log = Log(LOGS / "apply.log")
    log.write(f"# command: {sys.executable} {Path(__file__).name} --phase apply")
    rc = 0
    gc = _client()
    try:
        preview = json.loads((LOGS / "retirement-preview.json").read_text())
        digest = preview["manifest_sha256"]
        log.write(f"retirement_manifest_sha256={digest}")
        raw = RETIREMENT_AUTHORITY.read_bytes()
        applied = apply_signed_manifest(
            RETIREMENT_AUTHORITY,
            authority_file_sha256=hashlib.sha256(raw).hexdigest(),
            authority_payload_sha256=signed_payload_sha256(json.loads(raw)),
            reason=RETIRE_REASON,
            apply=True,
            manifest_sha256=digest,
            run_id=RUN_ID,
            gc=gc,
        )
        log.write(
            "retirement_apply="
            + _json({k: v for k, v in applied.items() if k != "manifest"})
        )
        (LOGS / "retirement-receipt.json").write_text(_json(applied) + "\n")
        if applied.get("outcome") not in {"applied", "already_applied"}:
            log.write("retirement apply did not report applied")
            log.close(2)
            return 2

        moved = retarget_standard_name_sources(
            gc,
            BASE,
            WALL,
            operation="source_migration_manifest",
            reason=MIGRATE_REASON,
            origin="signed_manifest",
            run_id=RUN_ID,
            source_ids=[WALL_PATH],
            expected_current_bindings={WALL_PATH: BASE},
        )
        log.write(f"migration_moved={moved}")
        (LOGS / "migration-receipt.json").write_text(
            _json(
                {
                    "moved": moved,
                    "source_id": WALL_PATH,
                    "from_name": BASE,
                    "to_name": WALL,
                    "operation": "source_migration_manifest",
                    "run_id": RUN_ID,
                }
            )
            + "\n"
        )
        if moved != 1:
            rc = 2
    except Exception as exc:  # noqa: BLE001
        log.write(f"apply_error={type(exc).__name__}: {exc}")
        rc = 2
    finally:
        gc.close()
    log.close(rc)
    return rc


# ── phase: readback ──────────────────────────────────────────────────────
def _classifier(gc) -> dict:
    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )

    rows = classify_sourceless_standard_names(gc)
    return {
        "population": len(rows),
        "flagged": sorted(row["id"] for row in rows if row["id"] in {WALL, MIDPLANE}),
        "base_flagged": any(row["id"] == BASE for row in rows),
    }


def readback(gc) -> dict:
    state = read_state(gc)
    by_id = state["by_id"]
    wall_receipts = gc.query(
        """
        // SHC_LOCUS_READ_WALL_RECEIPTS
        MATCH (sn:StandardName {id: $id})-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
        RETURN DISTINCT c.operation AS operation, c.id AS change_id,
               c.manifest_sha256 AS manifest_sha256, c.run_id AS run_id
        ORDER BY operation, change_id
        """,
        id=WALL,
    )
    retire_receipts = gc.query(
        """
        // SHC_LOCUS_READ_RETIRE_RECEIPTS
        MATCH (sn:StandardName {id: $id})-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
        RETURN c.operation AS operation, c.id AS change_id,
               c.manifest_sha256 AS manifest_sha256, c.run_id AS run_id
        ORDER BY change_id
        """,
        id=MIDPLANE,
    )
    successor = gc.query(
        """
        // SHC_LOCUS_READ_SUCCESSOR
        MATCH (sn:StandardName {id: $id})
        OPTIONAL MATCH (sn)-[e:HAS_SUCCESSOR]->(n:StandardName)
        RETURN sn.name_stage AS stage, sn.status AS status,
               collect({target: n.id, target_stage: n.name_stage,
                        receipt_id: e.receipt_id}) AS edges
        """,
        id=MIDPLANE,
    )[0]
    return {
        "base_producers": sorted(s["id"] for s in by_id[BASE]["producers"]),
        "wall_producers": sorted(s["id"] for s in by_id[WALL]["producers"]),
        "base_source_paths": by_id[BASE]["source_paths"],
        "wall_source_paths": by_id[WALL]["source_paths"],
        "wall_projections": state["paths_by_id"][WALL_PATH]["projections"],
        "wall_scalar_target": state["paths_by_id"][WALL_PATH]["scalar_target"],
        "midplane": {
            "name_stage": successor["stage"],
            "status": successor["status"],
            "successors": [e for e in successor["edges"] if e["target"]],
        },
        "classifier": _classifier(gc),
        "wall_receipts": wall_receipts,
        "retire_receipts": retire_receipts,
        "state": state,
    }


def check_readback(result: dict) -> list[str]:
    problems: list[str] = []
    if result["wall_producers"] != [WALL_PATH]:
        problems.append(f"{WALL} producing sources are {result['wall_producers']!r}")
    if result["base_producers"] != sorted(KEPT_PATHS):
        problems.append(f"{BASE} producing sources are {result['base_producers']!r}")
    if sorted(result["base_source_paths"] or []) != sorted(KEPT_PATHS):
        problems.append(f"{BASE} source_paths are {result['base_source_paths']!r}")
    if sorted(result["wall_source_paths"] or []) != [WALL_PATH]:
        problems.append(f"{WALL} source_paths are {result['wall_source_paths']!r}")
    if result["wall_projections"] != [WALL]:
        problems.append(f"wall path projections are {result['wall_projections']!r}")
    if result["wall_scalar_target"] != WALL:
        problems.append(
            f"wall source scalar target is {result['wall_scalar_target']!r}"
        )
    if result["midplane"]["name_stage"] != "superseded":
        problems.append(f"{MIDPLANE} stage is {result['midplane']['name_stage']!r}")
    edges = result["midplane"]["successors"]
    if len(edges) != 1 or edges[0]["target"] != SUCCESSOR:
        problems.append(f"{MIDPLANE} successor edges are {edges!r}")
    elif not edges[0].get("receipt_id"):
        problems.append("successor edge carries no receipt id")
    classifier = result["classifier"]
    if classifier["flagged"]:
        problems.append(
            f"the classifier still lists {classifier['flagged']!r} as sourceless"
        )
    if classifier["base_flagged"]:
        problems.append(f"the classifier now lists {BASE} as sourceless")
    if not any(
        row["operation"] == "source_migration_manifest"
        for row in result["wall_receipts"]
    ):
        problems.append("no source_migration_manifest receipt on the receiving name")
    if not any(
        row["operation"] == RETIRE_OPERATION for row in result["retire_receipts"]
    ):
        problems.append("no signed-retirement receipt on the retired name")
    return problems


def phase_readback() -> int:
    log = Log(LOGS / "readback.log")
    log.write(f"# command: {sys.executable} {Path(__file__).name} --phase readback")
    rc = 0
    gc = _client()
    try:
        result = readback(gc)
        log.write(
            "readback=" + _json({k: v for k, v in result.items() if k != "state"})
        )
        problems = check_readback(result)
        log.write(f"readback_problems={problems}")
        (LOGS / "readback.json").write_text(_json(result) + "\n")

        # the read-back must be able to fail: doctor one expectation.
        control = Log(LOGS / "readback-doctored-refusal.log")
        doctored = json.loads(json.dumps(result, default=str))
        doctored["wall_producers"] = [WALL_PATH, KEPT_PATHS[0]]
        doctor_problems = check_readback(doctored)
        if doctor_problems:
            control.write("doctored_expectation: the wall name holds two paths")
            control.write(f"refused={doctor_problems}")
            control.close(1)
            log.write("doctored_readback_control=refused (EXIT=1)")
        else:
            control.write("read-back accepted a doctored expectation")
            control.close(3)
            log.write("doctored_readback_control=ACCEPTED-DOCTORED")
            rc = 3
        if problems:
            rc = 2
    except Exception as exc:  # noqa: BLE001
        log.write(f"readback_error={type(exc).__name__}: {exc}")
        rc = 2
    finally:
        gc.close()
    log.close(rc)
    return rc


# ── phase: harmonize ─────────────────────────────────────────────────────
def phase_harmonize() -> int:
    log = Log(LOGS / "harmonize.log")
    cmd = [
        str(PYTHON),
        "-m",
        "imas_codex.cli",
        "sn",
        "harmonize",
        "--family",
        BASE,
        "--report",
        "--json",
    ]
    log.write(f"# command: {' '.join(cmd)}")
    log.write(f"# worktree: {WORKTREE}")
    log.write("# report mode only: writes nothing to the graph and is not applied")
    rc = 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKTREE)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKTREE),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        log.write(f"harmonize_exit={proc.returncode}")
        log.block(proc.stdout)
        if proc.stderr.strip():
            log.block("stderr:")
            log.block(proc.stderr)
        (LOGS / "harmonize-output.json").write_text(proc.stdout)
        rc = 0 if proc.returncode == 0 else 2
    except Exception as exc:  # noqa: BLE001
        log.write(f"harmonize_error={type(exc).__name__}: {exc}")
        rc = 2
    log.close(rc)
    return rc


# ── phase: record ────────────────────────────────────────────────────────
def _read(path: Path, limit: int = 12000) -> str:
    text = path.read_text() if path.exists() else "<missing>"
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]\n"


def phase_record() -> int:
    log = Log(LOGS / "record.log")
    rc = 0
    try:
        result = json.loads((LOGS / "readback.json").read_text())
        attempts = json.loads((LOGS / "exchange-attempts.json").read_text())
        migration = json.loads((LOGS / "authorities.json").read_text())["migration"]
        receipt = json.loads((LOGS / "retirement-receipt.json").read_text())
        preview = json.loads((LOGS / "retirement-preview.json").read_text())
        harmonize_text = _read(LOGS / "harmonize.log")
        stale = _read(LOGS / "stale-authority-refusal.log")
        cas = _read(LOGS / "migration-stale-cas-refusal.log")
        doctored = _read(LOGS / "readback-doctored-refusal.log")

        def pre(value: object) -> str:
            return html.escape(json.dumps(value, indent=2, sort_keys=True, default=str))

        names_html = "\n".join(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td></tr>".format(
                html.escape(row["id"]),
                html.escape(str(row["name_stage"])),
                len(row["producers"]),
                html.escape(str(row["docs_score"])),
                "yes" if row["open_holds"] else "no",
            )
            for row in result["state"]["names"]
        )
        attempt_html = "\n".join(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                html.escape(attempt["label"]),
                html.escape(str(attempt.get("outcome"))),
                html.escape(
                    attempt.get("refusal_text")
                    or "; ".join(
                        f"{r['row_id']}: {r['reason']}"
                        for r in (attempt.get("refusals") or [])
                        if isinstance(r, dict)
                    )
                    or "(no refusal recorded)"
                ),
            )
            for attempt in attempts
        )
        edges = result["midplane"]["successors"]
        edge_text = (
            ", ".join(
                "<code>" + html.escape(str(e["target"])) + "</code>" for e in edges
            )
            or "none"
        )
        page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="reckon-type" content="evidence-fragment">
<meta name="plan-slug" content="sn-harmonize-command">
<meta name="node-id" content="shc-locus-wall-rebind-midplane-retire">
<title>The wall potential gets its DD source back; the outboard-midplane name retires</title>
</head>
<body>
<main>
<h2 id="shc-locus-wall-rebind-midplane-retire-ruling">The ruling</h2>
<p id="shc-locus-wall-rebind-midplane-retire-ruling-p">Both names were published
with a DD source: <code>electrostatic_potential_at_wall</code> from
<code>wall/description_ggd/ggd/phi_potential/values</code>, and the midplane name
from <code>langmuir_probes/reciprocating/plunge/potential_plasma</code>, which DD 4
renamed <code>v_plasma</code>. The graph imported them with no source edge, so
both read as sourceless. The lead ruled that the wall path moves to
<code>electrostatic_potential_at_wall</code>, and that
<code>electrostatic_potential_at_outboard_midplane</code> is retired onto
<code>electrostatic_potential_at_langmuir_probe_position</code>.</p>

<h2 id="shc-locus-wall-rebind-midplane-retire-before">State before</h2>
<table id="shc-locus-wall-rebind-midplane-retire-before-table">
<caption>Identity, producer count, documentation score and open holds, read live before any write</caption>
<thead><tr><th>name</th><th>name stage</th><th>producers</th><th>docs score</th><th>open hold</th></tr></thead>
<tbody>
{names_html}
</tbody>
</table>

<h2 id="shc-locus-wall-rebind-midplane-retire-exchange">Why the source move is a migration, not an exchange</h2>
<p id="shc-locus-wall-rebind-midplane-retire-exchange-p">The
<code>dd-source-exchange</code> adapter swaps one source for another: the giving
name hands over one source <em>and takes</em> a second, unheld one, while the
receiving name takes the first. A pure transfer, where the giving name keeps its
other six bindings and takes nothing, has no row shape in that adapter. Both
attempts below are previews and wrote nothing: the first carries no replacement
source, the second offers a source the giver already holds.</p>
<table id="shc-locus-wall-rebind-midplane-retire-exchange-table">
<caption>Exchange previews against the ruled direction</caption>
<thead><tr><th>attempt</th><th>outcome</th><th>refusal</th></tr></thead>
<tbody>
{attempt_html}
</tbody>
</table>
<p id="shc-locus-wall-rebind-midplane-retire-migration">The move was therefore
applied by <code>retarget_standard_name_sources</code>, the production writer
that owns an explicit compare-and-set source cohort and makes
<code>PRODUCED_NAME</code>, its scalar mirror, the upstream
<code>HAS_STANDARD_NAME</code> projection and both names'
<code>source_paths</code> agree in one transaction. Its manifest:</p>
<pre id="shc-locus-wall-rebind-midplane-retire-migration-pre">{pre(migration)}</pre>

<h2 id="shc-locus-wall-rebind-midplane-retire-authorities">Authorities and previews</h2>
<p id="shc-locus-wall-rebind-midplane-retire-authorities-p">The retirement
authority was built by <code>build_repair_authority</code>, previewed to
<code>would_apply</code>, and applied under that preview's digest.</p>
<pre id="shc-locus-wall-rebind-midplane-retire-preview-pre">{pre({k: v for k, v in preview.items() if k != "manifest"})}</pre>
<pre id="shc-locus-wall-rebind-midplane-retire-receipt-pre">{pre({k: v for k, v in receipt.items() if k != "manifest"})}</pre>

<h2 id="shc-locus-wall-rebind-midplane-retire-controls">The refusals this run had to earn</h2>
<p id="shc-locus-wall-rebind-midplane-retire-controls-p">A preview that accepts
everything shows nothing. A doctored retirement authority naming the wrong
successor is refused, the migration's compare-and-set is refused when its
expected binding is wrong, and the read-back check is refused when a doctored
expectation is fed to it.</p>
<pre id="shc-locus-wall-rebind-midplane-retire-stale-pre">{html.escape(stale)}</pre>
<pre id="shc-locus-wall-rebind-midplane-retire-cas-pre">{html.escape(cas)}</pre>
<pre id="shc-locus-wall-rebind-midplane-retire-doctored-pre">{html.escape(doctored)}</pre>

<h2 id="shc-locus-wall-rebind-midplane-retire-readback">Read-back</h2>
<p id="shc-locus-wall-rebind-midplane-retire-readback-p">
<code>electrostatic_potential_at_wall</code> has exactly one producing source,
<code>{html.escape(WALL_PATH)}</code>, with the DD path's
<code>HAS_STANDARD_NAME</code> projection and the source's scalar mirror both
naming it. <code>electrostatic_potential</code> keeps the other six.
<code>electrostatic_potential_at_outboard_midplane</code> is at
<code>{html.escape(str(result["midplane"]["name_stage"]))}</code> with
{len(edges)} successor edge{"" if len(edges) == 1 else "s"} to {edge_text}.
The sourceless classifier lists {len(result["classifier"]["flagged"])} of the two
ruled names and does not list the base.</p>
<pre id="shc-locus-wall-rebind-midplane-retire-readback-pre">{pre({k: v for k, v in result.items() if k != "state"})}</pre>

<h2 id="shc-locus-wall-rebind-midplane-retire-harmonize">Harmonize report for the family</h2>
<p id="shc-locus-wall-rebind-midplane-retire-harmonize-p"><code>sn harmonize
--family electrostatic_potential --report --json</code> was run in report mode
and not applied.</p>
<pre id="shc-locus-wall-rebind-midplane-retire-harmonize-pre">{html.escape(harmonize_text)}</pre>
</main>
</body>
</html>
"""
        FRAGMENT_HTML.write_text(page)
        log.write(f"wrote {FRAGMENT_HTML}")
    except Exception as exc:  # noqa: BLE001
        log.write(f"record_error={type(exc).__name__}: {exc}")
        rc = 2
    log.close(rc)
    return rc


PHASES = {
    "read": phase_read,
    "exchange": phase_exchange,
    "authorities": phase_authorities,
    "preview": phase_preview,
    "apply": phase_apply,
    "readback": phase_readback,
    "harmonize": phase_harmonize,
    "record": phase_record,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    return PHASES[args.phase]()


if __name__ == "__main__":
    raise SystemExit(main())

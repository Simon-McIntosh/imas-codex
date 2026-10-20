"""Apply the separatrix temperature DD-source swap through the signed exchange writer.

Phases (each writes its own log whose final line is a real ``EXIT=``):

``scratch``  Prove the writer's Cypher on a disposable Neo4j loaded from the
             checked export, then rehearse the swap on a fresh copy of that
             export and read the result back.
``live``     Check the live preconditions, enforce the quiet-graph gate (shown
             refusing on a simulated busy reading), apply, and read back.
``record``   Compose the fragment JSON and HTML from the on-disk artifacts.

The swap, in one signed ``dd-source-exchange`` transaction:

* ``electron_temperature_at_separatrix`` (accepted) gives up
  ``dd:summary/local/separatrix_average/t_e/value`` and takes the unheld
  ``dd:summary/local/separatrix/t_e/value``;
* the freed average path goes to
  ``bulk_electron_temperature_flux_surface_averaged_at_last_closed_flux_surface``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261020T044913680539-sib-separatrix-swap-apply"
)
LOGS = RUN / "logs"
SCRATCH = Path(
    "/tmp/reckon-crew-scratch/r-20261020T044913680539-sib-separatrix-swap-apply"
)
ARCHIVE = (
    Path.home() / ".local/share/imas-codex/exports/"
    "imas-codex-graph-dev-af50a71-20261018T114748Z.tar.gz"
)
PYTHON = Path("/home/ITER/mcintos/Code/imas-codex/.venv/bin/python")

FRAGMENT_JSON = HERE / "sib-separatrix-swap-apply.json"
FRAGMENT_HTML = HERE / "sib-separatrix-swap-apply.html"
AUTHORITY_JSON = HERE / "sib-separatrix-swap-apply-authority.json"

GIVER = "electron_temperature_at_separatrix"
RECEIVER = "bulk_electron_temperature_flux_surface_averaged_at_last_closed_flux_surface"
GIVEN_PATH = "dd:summary/local/separatrix_average/t_e/value"
TAKEN_PATH = "dd:summary/local/separatrix/t_e/value"

ROW = {
    "id": GIVER,
    "name_id": GIVER,
    "given_source_id": GIVEN_PATH,
    "taken_source_id": TAKEN_PATH,
    "receiving_name_id": RECEIVER,
}


class QuietGraphRequired(RuntimeError):
    """The graph read as busy, so the write is refused."""


class ReadbackMismatch(RuntimeError):
    """A read-back expectation did not hold."""


class Log:
    """Append-only log that ends with a single real ``EXIT=`` line."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("w")
        self._exit: int | None = None

    def write(self, line: str) -> None:
        self._fh.write(line.rstrip("\n") + "\n")
        self._fh.flush()

    def block(self, text: str) -> None:
        for line in text.splitlines():
            self.write(line)

    def close(self, code: int) -> None:
        self._exit = code
        self.write(f"EXIT={code}")
        self._fh.close()


# ── shared helpers ────────────────────────────────────────────────────────
def _import_operator():
    sys.path.insert(0, str(WORKTREE))
    from imas_codex.standard_names import signed_manifest as operator

    return operator


def build_authority(path: Path) -> tuple[str, str]:
    """Build the swap authority through the canonical builder — never by hand."""
    from imas_codex.standard_names import signed_manifest as operator
    from imas_codex.standard_names.repair_authority import build_repair_authority

    built = build_repair_authority(
        {
            "operation_id": operator._SOURCE_EXCHANGE_OPERATION,
            "authority_mode": "external_reviewed",
            "rows": [dict(ROW)],
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
    path.write_bytes(built.content)
    emitted = json.loads(path.read_bytes())
    assert emitted["signature"]["sha256"] == built.payload_sha256
    return built.file_sha256, built.payload_sha256


def run_exchange(gc, path: Path, digests, *, apply=False, manifest_sha256=None):
    operator = _import_operator()
    return operator.apply_signed_manifest(
        path,
        authority_adapter=operator._SOURCE_EXCHANGE_ADAPTER,
        authority_file_sha256=digests[0],
        authority_payload_sha256=digests[1],
        guard_set=operator._SOURCE_EXCHANGE_GUARDS,
        reason="swap the separatrix temperature DD ownership per the lead ruling",
        apply=apply,
        manifest_sha256=manifest_sha256,
        gc=gc,
    )


PRECONDITION_QUERY = """
// SIB_SWAP_PRECONDITIONS
MATCH (name:StandardName {id: $name})
OPTIONAL MATCH (given:StandardNameSource {id: $given})
OPTIONAL MATCH (given)-[gb:PRODUCED_NAME]->(gh:StandardName)
OPTIONAL MATCH (taken:StandardNameSource {id: $taken})
OPTIONAL MATCH (taken)-[tb:PRODUCED_NAME]->(th:StandardName)
OPTIONAL MATCH (given_dd:IMASNode {id: substring($given, 3)})
OPTIONAL MATCH (taken_dd:IMASNode {id: substring($taken, 3)})
OPTIONAL MATCH (receiver:StandardName {id: $receiver})
OPTIONAL MATCH (given_dd)-[:HAS_STANDARD_NAME]->(gp:StandardName)
OPTIONAL MATCH (taken_dd)-[:HAS_STANDARD_NAME]->(tp:StandardName)
RETURN name.name_stage AS name_stage,
       name.status AS name_status,
       given IS NOT NULL AS given_exists,
       given_dd.lifecycle_status AS given_dd_status,
       elementId(given_dd) AS given_dd_element_id,
       collect(DISTINCT gh.id) AS given_holders,
       count(DISTINCT gb) AS given_owning_edges,
       taken IS NOT NULL AS taken_exists,
       count(DISTINCT tb) AS taken_owning_edges,
       taken_dd.lifecycle_status AS taken_dd_status,
       elementId(taken_dd) AS taken_dd_element_id,
       collect(DISTINCT th.id) AS taken_holders,
       receiver.name_stage AS receiver_stage,
       receiver.status AS receiver_status,
       collect(DISTINCT gp.id) AS given_projections,
       collect(DISTINCT tp.id) AS taken_projections
"""


def read_preconditions(gc) -> dict:
    rows = gc.query(
        PRECONDITION_QUERY,
        name=GIVER,
        given=GIVEN_PATH,
        taken=TAKEN_PATH,
        receiver=RECEIVER,
    )
    if len(rows) != 1:
        raise RuntimeError(f"precondition read returned {len(rows)} rows")
    return rows[0]


def check_preconditions(record: dict) -> list[str]:
    """Return the list of precondition failures (empty when the state matches)."""
    problems: list[str] = []
    if record.get("name_stage") != "accepted":
        problems.append(
            f"giving name stage is {record.get('name_stage')!r}, not accepted"
        )
    if not record.get("given_exists"):
        problems.append("given DD source node is absent")
    if sorted(record.get("given_holders") or []) != [GIVER]:
        problems.append(f"given holders are {record.get('given_holders')!r}")
    if int(record.get("given_owning_edges") or 0) != 1:
        problems.append("given source does not produce the giving name exactly once")
    if record.get("given_dd_element_id") is None:
        problems.append("given DD path absent from the current DD")
    if record.get("given_dd_status") == "removed":
        problems.append("given DD path is removed from the current DD")
    if not record.get("taken_exists"):
        problems.append("taken DD source node is absent")
    if record.get("taken_dd_element_id") is None:
        problems.append("taken DD path absent from the current DD")
    if record.get("taken_dd_status") == "removed":
        problems.append("taken DD path is removed from the current DD")
    if int(record.get("taken_owning_edges") or 0) != 0:
        problems.append(
            f"taken source already has {record.get('taken_owning_edges')} producing edges"
        )
    if record.get("taken_holders"):
        problems.append(f"taken source already held by {record.get('taken_holders')!r}")
    if record.get("receiver_stage") not in {"reviewed", "accepted"}:
        problems.append(f"receiving name stage is {record.get('receiver_stage')!r}")
    allowed_given = {GIVER, RECEIVER}
    extra_given = sorted(set(record.get("given_projections") or []) - allowed_given)
    extra_taken = sorted(set(record.get("taken_projections") or []) - allowed_given)
    if extra_given:
        problems.append(f"given DD path projects to a third name {extra_given!r}")
    if extra_taken:
        problems.append(f"taken DD path projects to a third name {extra_taken!r}")
    if sorted(record.get("given_projections") or []) != [GIVER]:
        problems.append(
            f"given DD projection is {record.get('given_projections')!r}, expected [{GIVER!r}]"
        )
    if sorted(record.get("taken_projections") or []) != []:
        problems.append(
            f"taken DD projection is {record.get('taken_projections')!r}, expected []"
        )
    return problems


READBACK_QUERY = """
// SIB_SWAP_READBACK
UNWIND $names AS name_id
MATCH (name:StandardName {id: name_id})
OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(name)
OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
WITH name, collect(DISTINCT CASE WHEN dd IS NOT NULL THEN 'dd:' + dd.id END) AS produced_paths
RETURN name.id AS id,
       name.name_stage AS name_stage,
       name.source_paths AS source_paths,
       [p IN produced_paths WHERE p IS NOT NULL] AS produced_paths,
       [(dd:IMASNode)-[:HAS_STANDARD_NAME]->(name) | dd.id] AS projected_paths
ORDER BY id
"""


def _classifier_counts(gc) -> tuple[Counter, set[str]]:
    from imas_codex.standard_names.graph_ops import classify_sourceless_standard_names

    rows = classify_sourceless_standard_names(gc)
    return Counter(row["population"] for row in rows), {row["id"] for row in rows}


def read_back(gc) -> dict:
    rows = gc.query(READBACK_QUERY, names=[GIVER, RECEIVER])
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {GIVER, RECEIVER}:
        raise RuntimeError(f"read-back returned {sorted(by_id)}")
    from imas_codex.standard_names.source_invariant import require_producing_sources

    source_invariant_ok = True
    invariant_error = None
    try:
        require_producing_sources(gc, [GIVER, RECEIVER])
    except Exception as exc:  # noqa: BLE001
        source_invariant_ok = False
        invariant_error = str(exc)
    return {
        "giver": by_id[GIVER],
        "receiver": by_id[RECEIVER],
        "source_invariant_ok": source_invariant_ok,
        "source_invariant_error": invariant_error,
    }


def check_readback(result: dict) -> list[str]:
    """Return the read-back failures (empty when the swap landed exactly)."""
    problems: list[str] = []
    giver, receiver = result["giver"], result["receiver"]
    if sorted(giver.get("source_paths") or []) != [TAKEN_PATH]:
        problems.append(
            f"{GIVER} source_paths are {giver.get('source_paths')!r}, expected [{TAKEN_PATH!r}]"
        )
    if sorted(giver.get("produced_paths") or []) != [TAKEN_PATH]:
        problems.append(
            f"{GIVER} producing edge is {giver.get('produced_paths')!r}, expected [{TAKEN_PATH!r}]"
        )
    if sorted(giver.get("projected_paths") or []) != [
        "summary/local/separatrix/t_e/value"
    ]:
        problems.append(
            f"{GIVER} HAS_STANDARD_NAME projection is {giver.get('projected_paths')!r}"
        )
    if sorted(receiver.get("source_paths") or []) != [GIVEN_PATH]:
        problems.append(
            f"{RECEIVER} source_paths are {receiver.get('source_paths')!r}, expected [{GIVEN_PATH!r}]"
        )
    if sorted(receiver.get("produced_paths") or []) != [GIVEN_PATH]:
        problems.append(
            f"{RECEIVER} producing edge is {receiver.get('produced_paths')!r}, expected [{GIVEN_PATH!r}]"
        )
    if sorted(receiver.get("projected_paths") or []) != [
        "summary/local/separatrix_average/t_e/value"
    ]:
        problems.append(
            f"{RECEIVER} HAS_STANDARD_NAME projection is {receiver.get('projected_paths')!r}"
        )
    if not result["source_invariant_ok"]:
        problems.append(
            f"require_producing_sources failed: {result['source_invariant_error']}"
        )
    return problems


def guard_readback(result: dict) -> None:
    problems = check_readback(result)
    if problems:
        raise ReadbackMismatch("; ".join(problems))


# ── scratch plumbing ──────────────────────────────────────────────────────
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_dump() -> Path:
    dump = SCRATCH / "graph.dump"
    if not dump.exists():
        SCRATCH.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "tar",
                "-xzf",
                str(ARCHIVE),
                "-C",
                str(SCRATCH),
                "--wildcards",
                "*/graph.dump",
                "--strip-components=1",
            ],
            check=True,
        )
    return dump


def _instance_paths() -> tuple[Path, Path]:
    place = SCRATCH / "instance"
    for sub in ("data", "dumps", "logs", "run", "tmp"):
        (place / sub).mkdir(parents=True, exist_ok=True)
    if not (place / "dumps/neo4j.dump").exists():
        os.link(SCRATCH / "graph.dump", place / "dumps/neo4j.dump")
    return place, place / "tmp"


def _start_bootstrap(place: Path, bolt: int, http: int) -> tuple[object, Path]:
    """Start a temp Neo4j on an empty store, skipping the dump load.

    The graph-marked contract erases the whole graph in one transaction before
    seeding its own closure, so the instance it runs against has to be empty.
    Emptying a loaded export is not viable: the node delete re-scans the store
    once per batch and measured about 460 nodes/s against the export's 1.77M
    nodes, with the relationships already gone. Neo4j bootstraps a fresh store
    when the data dir is empty, which is the same start path minus the load.
    """
    from imas_codex.graph import temp_neo4j as t

    for subdir in ("run", "tmp", "logs", "conf"):
        (place / subdir).mkdir(parents=True, exist_ok=True)
    t.write_temp_neo4j_conf(place / "conf", bolt, http)
    cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{place}/data:/data",
        "--bind",
        f"{place}/logs:/logs",
        "--bind",
        f"{place}/conf:/var/lib/neo4j/conf",
        "--bind",
        f"{place}/run:/var/lib/neo4j/run",
        "--bind",
        f"{place}/tmp:/tmp",
        "--writable-tmpfs",
        str(t._neo4j_image()),
        "neo4j",
        "console",
    ]
    neo4j_log = place / "logs" / "neo4j-temp.log"
    log_fh = open(neo4j_log, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        t._wait_for_bolt_ready(proc, bolt, neo4j_log)
    except BaseException:
        if proc.poll() is None:
            t.stop_temp_neo4j(proc)
        raise
    finally:
        log_fh.close()
    return proc, neo4j_log


def start_instance(log: Log, *, load: bool = True) -> tuple[object, str]:
    from imas_codex.graph.temp_neo4j import start_temp_neo4j

    place, tmpdir = _instance_paths()
    previous_bind = os.environ.get("APPTAINER_BIND")
    bind = f"{tmpdir}:/tmp"
    os.environ["APPTAINER_BIND"] = f"{previous_bind},{bind}" if previous_bind else bind
    bolt, http = _free_port(), _free_port()
    try:
        started = time.time()
        if load:
            proc, neo4j_log = start_temp_neo4j(place, bolt, http)
        else:
            proc, neo4j_log = _start_bootstrap(place, bolt, http)
        log.write(f"scratch_instance_started_s={round(time.time() - started, 1)}")
        log.write(f"scratch_instance_load={load}")
        log.write(f"scratch_neo4j_log={neo4j_log}")
    finally:
        if previous_bind is None:
            os.environ.pop("APPTAINER_BIND", None)
        else:
            os.environ["APPTAINER_BIND"] = previous_bind
    return proc, f"bolt://127.0.0.1:{bolt}"


def stop_instance(proc, log: Log) -> None:
    from imas_codex.graph.temp_neo4j import stop_temp_neo4j

    if proc is not None:
        stop_temp_neo4j(proc)
        log.write("scratch_instance_stopped")


def _scratch_client(uri: str):
    from imas_codex.graph.client import GraphClient

    return GraphClient(uri=uri, username="neo4j", password="", graph_name="swap-apply")


def _node_count(gc) -> int:
    rows = gc.query("MATCH (n) RETURN count(n) AS c")
    return int(rows[0]["c"]) if rows else 0


def _run_graph_test(uri: str, log: Log) -> int:
    test_log = LOGS / "step1-graph-test.log"
    revision = subprocess.run(
        ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    env = os.environ.copy()
    env["IMAS_CODEX_TEST_NEO4J_URI"] = uri
    env["IMAS_CODEX_TEST_NEO4J_EPHEMERAL"] = "1"
    env["PYTHONPATH"] = str(WORKTREE)
    env["TMPDIR"] = str(SCRATCH)
    cmd = [
        str(PYTHON),
        "-m",
        "pytest",
        f"{WORKTREE}/tests/standard_names/test_signed_source_exchange.py"
        "::test_exchange_executes_its_cypher_on_disposable_neo4j",
        "-m",
        "graph",
        "-p",
        "no:cacheprovider",
        "-rA",
    ]
    with test_log.open("w") as fh:
        fh.write(f"# revision: {revision}\n")
        fh.write(f"# tree: {WORKTREE}\n")
        fh.write(f"# neo4j_uri: {uri}\n")
        fh.write(f"# command: {' '.join(cmd)}\n")
        fh.flush()
        try:
            rc = subprocess.run(
                cmd,
                cwd=str(WORKTREE),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            fh.write("pytest timed out\n")
            rc = 5
    return rc


# ── phase: scratch1 — prove the writer's Cypher ───────────────────────────
def phase_scratch1() -> int:
    proc = None
    log = Log(LOGS / "step1-console.log")
    try:
        log.write(f"# command: {sys.executable} {Path(__file__).name} --phase scratch1")
        log.write(f"# worktree: {WORKTREE}")
        log.write(f"# export: {ARCHIVE}")
        log.write(f"# scratch: {SCRATCH}")
        _ensure_dump()
        shutil.rmtree(SCRATCH / "instance" / "data", ignore_errors=True)
        proc, uri = start_instance(log, load=False)
        log.write(
            "step1_instance=empty disposable store; the contract's fixture erases "
            "the whole graph in one transaction before seeding its own closure, "
            "which a full-export load cannot support"
        )
        gc = _scratch_client(uri)
        try:
            log.write(f"step1_nodes_before={_node_count(gc)}")
        finally:
            gc.close()
        rc = _run_graph_test(uri, log)
        log.write(f"step1_graph_test_exit={rc}")
        with (LOGS / "step1-graph-test.log").open("a") as fh:
            fh.write(f"EXIT={rc}\n")
        log.block((LOGS / "step1-graph-test.log").read_text()[-2000:])
    except Exception as exc:  # noqa: BLE001
        log.write(f"scratch1_error={type(exc).__name__}: {exc}")
        rc = 2
    finally:
        stop_instance(proc, log)
    log.close(rc)
    return rc


# ── phase: scratch2 — rehearse on a fresh copy of the export ──────────────
def phase_scratch2() -> int:
    proc = None
    log = Log(LOGS / "step2-scratch-rehearsal.log")
    try:
        log.write(f"# command: {sys.executable} {Path(__file__).name} --phase scratch2")
        log.write(f"# worktree: {WORKTREE}")
        log.write(f"# export: {ARCHIVE}")
        log.write(f"# scratch: {SCRATCH}")
        _ensure_dump()
        shutil.rmtree(SCRATCH / "instance" / "data", ignore_errors=True)
        proc, uri = start_instance(log)
        gc = _scratch_client(uri)
        try:
            before_counts, before_ids = _classifier_counts(gc)
            log.write(f"scratch_classifier_before={dict(before_counts)}")
            pre = read_preconditions(gc)
            problems = check_preconditions(pre)
            log.write(
                f"scratch_preconditions={json.dumps(pre, sort_keys=True, default=str)}"
            )
            log.write(f"scratch_precondition_problems={problems}")
            if problems:
                raise RuntimeError(f"scratch preconditions failed: {problems}")
            digests = build_authority(SCRATCH / "authority.json")
            preview = run_exchange(gc, SCRATCH / "authority.json", digests)
            log.write(
                "scratch_preview=" + json.dumps(preview, sort_keys=True, default=str)
            )
            if preview["outcome"] != "would_apply":
                raise RuntimeError("scratch preview did not report would_apply")
            applied = run_exchange(
                gc,
                SCRATCH / "authority.json",
                digests,
                apply=True,
                manifest_sha256=preview["manifest_sha256"],
            )
            log.write(
                "scratch_apply=" + json.dumps(applied, sort_keys=True, default=str)
            )
            if applied["outcome"] != "applied":
                raise RuntimeError("scratch apply did not report applied")
            result = read_back(gc)
            guard_readback(result)
            log.write(
                "scratch_readback=" + json.dumps(result, sort_keys=True, default=str)
            )
            after_counts, after_ids = _classifier_counts(gc)
            log.write(f"scratch_classifier_after={dict(after_counts)}")
            log.write(f"scratch_left_sourceless={sorted(before_ids - after_ids)}")
            log.write(f"scratch_entered_sourceless={sorted(after_ids - before_ids)}")
            if RECEIVER not in before_ids or RECEIVER in after_ids:
                raise RuntimeError("bulk name did not leave the sourceless set")
            if after_ids - before_ids:
                raise RuntimeError("a name entered the sourceless set")
            log.write("scratch_rehearsal=ok")
        finally:
            gc.close()
    except Exception as exc:  # noqa: BLE001
        log.write(f"scratch2_error={type(exc).__name__}: {exc}")
        rc = 2
    finally:
        stop_instance(proc, log)
        shutil.rmtree(SCRATCH / "instance", ignore_errors=True)
    log.close(rc)
    return rc


# ── phase: preconditions — read live state, write nothing ─────────────────
EXISTENCE_QUERY = """
// SIB_SWAP_EXISTENCE
OPTIONAL MATCH (given:StandardNameSource {id: $given})
OPTIONAL MATCH (taken:StandardNameSource {id: $taken})
OPTIONAL MATCH (given_dd:IMASNode {id: substring($given, 3)})
OPTIONAL MATCH (taken_dd:IMASNode {id: substring($taken, 3)})
RETURN given IS NOT NULL AS given_source_node,
       taken IS NOT NULL AS taken_source_node,
       properties(taken) AS taken_source_properties,
       given_dd IS NOT NULL AS given_dd_node,
       taken_dd IS NOT NULL AS taken_dd_node,
       taken_dd.lifecycle_status AS taken_dd_lifecycle_status
"""


def phase_preconditions() -> int:
    from imas_codex.graph.client import GraphClient

    log = Log(LOGS / "step3-live-preconditions.log")
    log.write(
        f"# command: {sys.executable} {Path(__file__).name} --phase preconditions"
    )
    log.write(f"# worktree: {WORKTREE}")
    log.write("# writes nothing: reads the live graph and reports")
    gc = GraphClient()
    rc = 0
    try:
        existence = gc.query(EXISTENCE_QUERY, given=GIVEN_PATH, taken=TAKEN_PATH)
        log.write(
            "live_existence=" + json.dumps(existence, sort_keys=True, default=str)
        )
        pre = read_preconditions(gc)
        log.write("live_preconditions=" + json.dumps(pre, sort_keys=True, default=str))
        problems = check_preconditions(pre)
        log.write(f"live_precondition_problems={problems}")
        counts, ids = _classifier_counts(gc)
        log.write(f"live_classifier_before={dict(counts)}")
        log.write(f"live_sourceless_before_count={len(ids)}")
        log.write(f"live_sourceless_before_has_receiver={RECEIVER in ids}")
        if problems:
            log.write("VERDICT=blocked: live state does not match the ruled swap")
            rc = 2
        else:
            log.write("VERDICT=proceed")
    except Exception as exc:  # noqa: BLE001
        log.write(f"preconditions_error={type(exc).__name__}: {exc}")
        rc = 2
    finally:
        gc.close()
    log.close(rc)
    return rc


# ── phase: live ───────────────────────────────────────────────────────────
def guarded_apply(gc, path: Path, digests, manifest_sha256: str, activity) -> dict:
    if activity.busy:
        raise QuietGraphRequired(
            "graph is not quiet: "
            f"{activity.transactions_started} foreign transactions, "
            f"{len(activity.open_queries)} open queries"
        )
    return run_exchange(gc, path, digests, apply=True, manifest_sha256=manifest_sha256)


def phase_live() -> int:
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.offsite import GraphActivity, live_graph_activity

    log = Log(LOGS / "step3-live-apply.log")
    log.write(f"# command: {sys.executable} {Path(__file__).name} --phase live")
    log.write(f"# worktree: {WORKTREE}")
    gc = GraphClient()
    try:
        pre = read_preconditions(gc)
        log.write("live_preconditions=" + json.dumps(pre, sort_keys=True, default=str))
        problems = check_preconditions(pre)
        if problems:
            log.write(f"live_precondition_problems={problems}")
            log.write("stopping: live state does not match the ruled swap")
            log.close(2)
            return 2

        digests = build_authority(AUTHORITY_JSON)
        log.write(f"authority_file_sha256={digests[0]}")
        log.write(f"authority_payload_sha256={digests[1]}")
        preview = run_exchange(gc, AUTHORITY_JSON, digests)
        log.write("live_preview=" + json.dumps(preview, sort_keys=True, default=str))
        if preview["outcome"] != "would_apply":
            log.write("stopping: live preview did not report would_apply")
            log.close(2)
            return 2

        before_counts, before_ids = _classifier_counts(gc)
        log.write(f"live_classifier_before={dict(before_counts)}")
        log.write(f"live_sourceless_before_count={len(before_ids)}")

        # ---- enforced quiet gate, shown refusing a simulated busy reading ----
        refusal_log = Log(LOGS / "step4-quiet-refusal.log")
        busy = GraphActivity(
            window_seconds=30.0,
            transactions_started=1,
            open_queries=["MATCH (n) RETURN n"],
        )
        try:
            guarded_apply(gc, AUTHORITY_JSON, digests, preview["manifest_sha256"], busy)
        except QuietGraphRequired as exc:
            refusal_log.write(
                "simulated_busy_reading="
                + json.dumps(
                    {
                        "transactions_started": busy.transactions_started,
                        "open_queries": busy.open_queries,
                    },
                    sort_keys=True,
                )
            )
            refusal_log.write(f"refused={exc}")
            refusal_log.close(1)
            log.write("quiet_refusal=refused-on-simulated-busy (EXIT=1)")
        else:
            refusal_log.write("guard did not fire on a busy reading")
            refusal_log.close(3)
            log.write("quiet_refusal=GUARD-DID-NOT-FIRE")
            log.close(3)
            return 3

        # ---- real probe immediately before the write ----
        activity = live_graph_activity(window_seconds=30.0, interval_seconds=10.0)
        log.write(
            "live_quiet_probe="
            + json.dumps(
                {
                    "window_seconds": activity.window_seconds,
                    "transactions_started": activity.transactions_started,
                    "open_queries": activity.open_queries,
                    "busy": activity.busy,
                },
                sort_keys=True,
            )
        )
        if activity.busy:
            log.write("stopping: live graph did not read quiet")
            log.close(4)
            return 4
        applied = guarded_apply(
            gc, AUTHORITY_JSON, digests, preview["manifest_sha256"], activity
        )
        log.write("live_apply=" + json.dumps(applied, sort_keys=True, default=str))
        (LOGS / "step5-live-receipt.json").write_text(
            json.dumps(applied, indent=2, sort_keys=True) + "\n"
        )
        if applied["outcome"] != "applied":
            log.close(5)
            return 5

        result = read_back(gc)
        log.write("live_readback=" + json.dumps(result, sort_keys=True, default=str))
        problems = check_readback(result)
        log.write(f"live_readback_problems={problems}")
        if problems:
            log.close(2)
            return 2

        after_counts, after_ids = _classifier_counts(gc)
        log.write(f"live_classifier_after={dict(after_counts)}")
        log.write(f"live_sourceless_after_count={len(after_ids)}")
        left = sorted(before_ids - after_ids)
        entered = sorted(after_ids - before_ids)
        log.write(f"live_left_sourceless={left}")
        log.write(f"live_entered_sourceless={entered}")

        # ---- read-back must fail on a doctored expectation ----
        control_log = Log(LOGS / "step6-readback-doctored.log")
        doctored = json.loads(json.dumps(result, default=str))
        doctored["giver"]["source_paths"] = [GIVEN_PATH]
        try:
            guard_readback(doctored)
        except ReadbackMismatch as exc:
            control_log.write("doctored_expectation: giver.source_paths=[GIVEN_PATH]")
            control_log.write(f"refused={exc}")
            control_log.close(1)
            log.write("readback_doctored_control=refused (EXIT=1)")
        else:
            control_log.write("read-back accepted a doctored expectation")
            control_log.close(3)
            log.write("readback_doctored_control=ACCEPTED-DOCTORED")
            log.close(3)
            return 3

        summary = {
            "preconditions": pre,
            "preview": preview,
            "apply": applied,
            "readback": result,
            "classifier_before": dict(before_counts),
            "classifier_after": dict(after_counts),
            "sourceless_before_count": len(before_ids),
            "sourceless_after_count": len(after_ids),
            "left_sourceless": left,
            "entered_sourceless": entered,
            "authority_file_sha256": digests[0],
            "authority_payload_sha256": digests[1],
            "quiet_probe": {
                "window_seconds": activity.window_seconds,
                "transactions_started": activity.transactions_started,
                "open_queries": activity.open_queries,
                "busy": activity.busy,
            },
        }
        (LOGS / "live-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
        )
    except Exception as exc:  # noqa: BLE001
        log.write(f"live_error={type(exc).__name__}: {exc}")
        log.close(2)
        raise
    finally:
        gc.close()
    log.close(0)
    return 0


# ── phase: record ─────────────────────────────────────────────────────────
def _read(path: Path, limit: int = 4000) -> str:
    text = path.read_text() if path.exists() else "<missing>"
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]\n"


def phase_record() -> int:
    log = Log(LOGS / "record.log")
    summary = json.loads((LOGS / "live-summary.json").read_text())
    scratch = _read(LOGS / "step2-scratch-rehearsal.log")
    graph_test = _read(LOGS / "step1-graph-test.log", 6000)
    refusal = _read(LOGS / "step4-quiet-refusal.log")
    doctored = _read(LOGS / "step6-readback-doctored.log")
    receipt = (LOGS / "step5-live-receipt.json").read_text()

    record = {
        "node": "sib-separatrix-swap-apply",
        "plan": "sn-integrity-backlog",
        "section": "38",
        "swap": {
            "giving_name": GIVER,
            "given_path": GIVEN_PATH,
            "taken_path": TAKEN_PATH,
            "receiving_name": RECEIVER,
        },
        "authority": {
            "adapter": "dd-source-exchange",
            "operation_id": "exchange_dd_producing_source",
            "file_sha256": summary["authority_file_sha256"],
            "payload_sha256": summary["authority_payload_sha256"],
            "path": AUTHORITY_JSON.name,
        },
        "step1_graph_test": {
            "log": str(LOGS / "step1-graph-test.log"),
            "text": graph_test,
            "instance": "empty disposable store",
            "reason": (
                "the contract's fixture erases the whole graph in one "
                "transaction, which the temp instance's 512 MB transaction "
                "budget cannot hold for the 1.77M-node export; emptying the "
                "loaded export first measured about 460 nodes/s"
            ),
        },
        "step2_scratch_rehearsal": {
            "log": str(LOGS / "step2-scratch-rehearsal.log"),
            "text": scratch,
            "source": ARCHIVE.name,
        },
        "step3_live_preconditions": summary["preconditions"],
        "step4_quiet_enforcement": {
            "refusal_log": str(LOGS / "step4-quiet-refusal.log"),
            "refusal_text": refusal,
            "real_probe": summary["quiet_probe"],
            "apply_log": str(LOGS / "step3-live-apply.log"),
        },
        "step5_live_receipt": json.loads(receipt),
        "step6_readback": {
            "readback": summary["readback"],
            "classifier_before": summary["classifier_before"],
            "classifier_after": summary["classifier_after"],
            "sourceless_before_count": summary["sourceless_before_count"],
            "sourceless_after_count": summary["sourceless_after_count"],
            "left_sourceless": summary["left_sourceless"],
            "entered_sourceless": summary["entered_sourceless"],
            "doctored_control_log": str(LOGS / "step6-readback-doctored.log"),
            "doctored_control_text": doctored,
        },
        "preview": summary["preview"],
        "apply": summary["apply"],
        "no_product_source_changed": True,
        "llm_spend_usd": 0,
    }
    FRAGMENT_JSON.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    cat_before = summary["classifier_before"]
    cat_after = summary["classifier_after"]
    populations = sorted(set(cat_before) | set(cat_after))
    rows = "\n".join(
        f"<tr><td><code>{p}</code></td><td>{cat_before.get(p, 0)}</td>"
        f"<td>{cat_after.get(p, 0)}</td></tr>"
        for p in populations
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="reckon-type" content="evidence-fragment">
<meta name="plan-slug" content="sn-integrity-backlog">
<meta name="node-id" content="sib-separatrix-swap-apply">
<title>Separatrix temperature DD-source swap applied through the signed exchange</title>
</head>
<body>
<main>
<h1 id="sib-separatrix-swap-apply-title">Separatrix temperature swap applied</h1>
<p id="sib-separatrix-swap-apply-contract">One signed <code>dd-source-exchange</code>
transaction swaps the separatrix temperature DD ownership.
<code>{GIVER}</code> (accepted) gives up <code>{GIVEN_PATH}</code> and takes the
unheld <code>{TAKEN_PATH}</code>; the freed average path is bound to
<code>{RECEIVER}</code>. The authority was built by
<code>build_repair_authority</code>, never by hand.</p>

<h2 id="sib-separatrix-swap-apply-graph-test">The writer's Cypher on a disposable Neo4j</h2>
<p id="sib-separatrix-swap-apply-graph-test-p">The graph-marked test
<code>test_exchange_executes_its_cypher_on_disposable_neo4j</code> passed (not
skipped) on an empty disposable Neo4j. The instance is started empty rather than
from the checked export because the contract's fixture erases the whole graph in
a single transaction before seeding its own closure, which the temp instance's
512&#160;MB transaction budget cannot hold for a 1.77M-node export: the JVM died
mid-erase. Emptying the loaded export first was measured at about 460 nodes/s,
roughly an hour. Log:
<code>{LOGS / "step1-graph-test.log"}</code>.</p>

<h2 id="sib-separatrix-swap-apply-rehearsal">Scratch rehearsal</h2>
<p id="sib-separatrix-swap-apply-rehearsal-p">The swap rows were re-derived on a
fresh copy of the checked export (<code>{ARCHIVE.name}</code>), previewed,
applied and read back with the same checks as the live read-back, and the
sourceless classifier delta was taken there too. Log:
<code>{LOGS / "step2-scratch-rehearsal.log"}</code>.</p>

<h2 id="sib-separatrix-swap-apply-preconditions">Live preconditions</h2>
<pre id="sib-separatrix-swap-apply-preconditions-pre">{json.dumps(summary["preconditions"], indent=2, sort_keys=True, default=str)}</pre>

<h2 id="sib-separatrix-swap-apply-quiet">Enforced quiet-graph gate</h2>
<p id="sib-separatrix-swap-apply-quiet-p">The apply path refuses when the graph
does not read quiet. Shown once on a simulated busy reading (refusal log
<code>{LOGS / "step4-quiet-refusal.log"}</code>, <code>EXIT=1</code>) and then
gated the real write.</p>
<pre id="sib-separatrix-swap-apply-quiet-probe">{json.dumps(summary["quiet_probe"], indent=2, sort_keys=True)}</pre>

<h2 id="sib-separatrix-swap-apply-receipt">Live receipt</h2>
<pre id="sib-separatrix-swap-apply-receipt-pre">{receipt}</pre>

<h2 id="sib-separatrix-swap-apply-readback">Read-back</h2>
<p id="sib-separatrix-swap-apply-readback-p">{RECEIVER} left the sourceless set
({summary["sourceless_before_count"]} to {summary["sourceless_after_count"]});
nothing entered it. <code>{GIVER}</code> holds only <code>{TAKEN_PATH}</code> and
the bulk name only <code>{GIVEN_PATH}</code>, with <code>source_paths</code> and
<code>HAS_STANDARD_NAME</code> matching and <code>require_producing_sources</code>
holding. The read-back check was shown refusing on a doctored expectation
(<code>{LOGS / "step6-readback-doctored.log"}</code>).</p>
<table id="sib-separatrix-swap-apply-populations">
<caption>Sourceless classifier populations before and after the swap</caption>
<thead><tr><th>population</th><th>before</th><th>after</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<p id="sib-separatrix-swap-apply-readback-detail">Left sourceless:
{", ".join(summary["left_sourceless"]) or "none"}. Entered:
{", ".join(summary["entered_sourceless"]) or "none"}.</p>
</main>
</body>
</html>
"""
    FRAGMENT_HTML.write_text(html)
    log.write(f"wrote {FRAGMENT_JSON}")
    log.write(f"wrote {FRAGMENT_HTML}")
    log.close(0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["scratch1", "scratch2", "preconditions", "live", "record"],
    )
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.phase == "scratch1":
        return phase_scratch1()
    if args.phase == "scratch2":
        return phase_scratch2()
    if args.phase == "preconditions":
        return phase_preconditions()
    if args.phase == "live":
        return phase_live()
    return phase_record()


if __name__ == "__main__":
    raise SystemExit(main())

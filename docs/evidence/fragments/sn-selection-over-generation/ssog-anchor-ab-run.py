"""Measure paired family harmonization from one archived graph snapshot."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
from imas_codex.settings import (
    get_model,
    get_openrouter_pricing,
    get_sn_review_docs_models,
)
from imas_codex.standard_names.families import (
    fetch_pattern_families,
    score_family_anchor,
)
from imas_codex.standard_names.harmonize import (
    _record_family_anchor,
    group_signature,
    measure_doc_call_cost,
    plan_family_applies,
    report_family_rows,
    resolve_family_anchor,
)

ARCHIVE = (
    Path.home()
    / ".local/share/imas-codex/exports/imas-codex-graph-dev-af50a71-20261018T114748Z.tar.gz"
)
RUN_DIR = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261018T102210638794-ssog-anchor-ab"
)
RESULTS = Path(
    os.environ.get(
        "ANCHOR_RESULTS_PATH",
        str(Path(__file__).with_name("ssog-anchor-ab-results.json")),
    )
)
ROSTERS = Path("imas_codex/standard_names/manifests/reviews")
SPEND_LIMIT = 50.0
PREVIOUS_SPEND = 0.021194586
RULE_REGISTERED_AT = "2026-10-18T10:45:18.869980+00:00"
PRIOR_DOC_CALL_COST = 0.41403
PRIOR_COST_EVIDENCE = (
    "docs/evidence/fragments/sn-harmonize-command/shc-apply-scope-estimates.html"
)
VARIATION_AXES = (
    "subject",
    "component",
    "coordinate",
    "position",
    "geometry",
    "region",
    "device",
)
DEADLINE = datetime.fromisoformat("2026-10-18T23:40:19+00:00")


def apply_cap(planned_calls: int) -> float:
    if planned_calls == 3:
        return 1.98
    return round(planned_calls * 0.674, 2)


def rotated_doc_call_cost(gc: GraphClient, measured_calls: int) -> dict:
    """Reprice snapshot documentation usage for the active model seats."""
    from datetime import timedelta

    docs_model = get_model("sn-docs")
    refine_model = get_model("sn-refine")
    review_models = get_sn_review_docs_models()
    if "gpt-6" not in docs_model or not any("gpt-6" in m for m in review_models):
        raise RuntimeError("Documentation and review breaker seats are not GPT-6")
    since = (datetime.now(UTC) - timedelta(days=14)).isoformat()
    rows = gc.query(
        "MATCH (c:LLMCost) WHERE c.phase IN $phases "
        "AND c.llm_at >= datetime($since) AND c.run_id IS NOT NULL "
        "WITH c.run_id AS run_id, max(c.llm_at) AS newest "
        "ORDER BY newest DESC LIMIT 200 "
        "WITH collect(run_id) AS run_ids "
        "MATCH (c:LLMCost) WHERE c.run_id IN run_ids AND c.phase IN $phases "
        "RETURN c.run_id AS run_id, c.phase AS phase, c.llm_model AS model, "
        "c.llm_cost AS cost, c.llm_tokens_in AS tokens_in, "
        "c.llm_tokens_out AS tokens_out, "
        "c.llm_tokens_cached_read AS cached_read, "
        "c.llm_tokens_cached_write AS cached_write, "
        "c.standard_name_ids AS names LIMIT 20001",
        phases=["generate_docs", "refine_docs", "review_docs"],
        since=since,
    )
    if len(rows) == 20001:
        raise RuntimeError("Documentation receipt read reached its row bound")
    names = {name for row in rows for name in (row["names"] or [])}
    if len(names) != measured_calls or not rows:
        raise RuntimeError(
            "Repricing read did not recover measured documentation calls"
        )
    per_model: dict[str, dict[str, float | int]] = {}
    old_total = 0.0
    rotated_total = 0.0
    changed = 0
    for row in rows:
        old = row["model"]
        phase = row["phase"]
        if phase == "generate_docs":
            new = docs_model
        elif phase == "refine_docs":
            new = refine_model
        elif phase == "review_docs" and "gpt-5" in old:
            new = next(m for m in review_models if "gpt-6" in m)
        else:
            new = old
        actual = float(row["cost"] or 0.0)
        old_total += actual
        if new == old:
            projected = actual
        else:
            price = get_openrouter_pricing(new)
            if not price:
                raise RuntimeError(f"No active price for {new}")
            tokens_in = int(row["tokens_in"] or 0)
            tokens_out = int(row["tokens_out"] or 0)
            cached_read = int(row["cached_read"] or 0)
            cached_write = int(row["cached_write"] or 0)
            uncached = max(0, tokens_in - cached_read - cached_write)
            projected = (
                uncached * float(price["prompt"])
                + tokens_out * float(price["completion"])
                + cached_read * float(price.get("cache_read") or price["prompt"])
                + cached_write * float(price.get("cache_write") or price["prompt"])
            ) / 1_000_000 + float(price.get("request") or 0)
            changed += 1
        rotated_total += projected
        entry = per_model.setdefault(new, {"rows": 0, "usd": 0.0})
        entry["rows"] += 1
        entry["usd"] += projected
    return {
        "calls": measured_calls,
        "runs": len({row["run_id"] for row in rows}),
        "receipt_rows": len(rows),
        "repriced_rows": changed,
        "historical_usd": old_total,
        "rotated_usd": rotated_total,
        "rate_usd": rotated_total / measured_calls,
        "per_model": per_model,
        "seats": {
            "sn-docs": docs_model,
            "sn-refine": refine_model,
            "sn-review.docs": review_models,
        },
    }


def now() -> str:
    return datetime.now(UTC).isoformat()


def save(data: dict) -> None:
    RESULTS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def require_time() -> None:
    if datetime.now(UTC) >= DEADLINE:
        raise RuntimeError("Worker deadline reached")


def graph_count(gc: GraphClient) -> int:
    rows = gc.query("MATCH (n) RETURN count(n) AS total")
    return int(rows[0]["total"])


def prove_isolation(
    gc: GraphClient, live_uri: str, scratch_uri: str, arm: str, data: dict, log: Path
) -> None:
    """Show a scratch-only node and remove it before measuring a family."""
    scratch_count = graph_count(gc)
    with GraphClient(uri=live_uri) as live:
        live_count = graph_count(live)
        token = str(uuid.uuid4())
        gc.query("CREATE (p:ScratchProbe {id: $token}) RETURN p.id AS id", token=token)
        marked_count = graph_count(gc)
        scratch_probe = gc.query(
            "MATCH (p:ScratchProbe {id: $token}) RETURN count(p) AS n", token=token
        )[0]["n"]
        live_probe = live.query(
            "MATCH (p:ScratchProbe {id: $token}) RETURN count(p) AS n", token=token
        )[0]["n"]
        gc.query("MATCH (p:ScratchProbe {id: $token}) DELETE p", token=token)
        restored_count = graph_count(gc)
    proof = {
        "arm": arm,
        "scratch_uri": scratch_uri,
        "scratch_nodes": scratch_count,
        "live_uri": live_uri,
        "live_nodes": live_count,
        "marked_scratch_nodes": marked_count,
        "scratch_probe": scratch_probe,
        "live_probe": live_probe,
        "restored_scratch_nodes": restored_count,
        "at": now(),
        "neo4j_log": str(log),
    }
    data.setdefault("isolation_proof", []).append(proof)
    save(data)
    print(
        f"isolation {arm}: scratch {scratch_uri} {scratch_count}->{marked_count}->{restored_count}; live {live_uri} {live_count}; marker scratch={scratch_probe} live={live_probe}",
        flush=True,
    )
    if (
        scratch_count <= 0
        or marked_count != scratch_count + 1
        or scratch_probe != 1
        or live_probe != 0
        or restored_count != scratch_count
    ):
        raise RuntimeError("Scratch-only marker contrast failed")


def west_roster() -> tuple[set[str], list[dict]]:
    names: set[str] = set()
    evidence: list[dict] = []
    for path in sorted(ROSTERS.glob("*.sn_names.yaml")):
        doc = yaml.safe_load(path.read_text())
        members = set(doc.get("names") or [])
        names.update(members)
        evidence.append({"path": str(path), "names": len(members)})
    if not names:
        raise RuntimeError("WEST review roster search found no names")
    return names, evidence


def kind(family: dict, traits: dict[str, dict]) -> str:
    varied = [
        axis
        for axis in VARIATION_AXES
        if len(
            {str(traits[member["id"]].get(axis) or "") for member in family["members"]}
        )
        > 1
    ]
    return "+".join(varied) or "unclassified"


async def scored_candidates(families: list[dict], data: dict) -> list[dict]:
    semaphore = asyncio.Semaphore(4)
    results: list[dict] = []

    async def score(family: dict) -> dict:
        async with semaphore:
            require_time()
            members = family["members"]
            heuristic = resolve_family_anchor(members)
            judged = await score_family_anchor(
                None, None, None, members, pattern="|".join(family["invariant"])
            )
            return {
                "seed": sorted(m["id"] for m in members)[0],
                "physical_base": family["physical_base"],
                "kind": family["kind"],
                "members": sorted(m["id"] for m in members),
                "heuristic_anchor": heuristic,
                "jev_anchor": judged.choice,
                "jev_cost": judged.cost,
                "jev_asked": list(judged.asked),
                "jev_scored": len(judged.expected),
                "jev_reason": judged.reason,
            }

    for batch_start in range(0, len(families), 8):
        batch = await asyncio.gather(
            *(score(f) for f in families[batch_start : batch_start + 8])
        )
        results.extend(batch)
        data["scored_candidates"] = results
        save(data)
        print(f"scored {len(results)}/{len(families)} candidate families", flush=True)
    return results


def select(candidates: list[dict]) -> list[dict]:
    differing = [
        row
        for row in candidates
        if row["jev_anchor"]
        and row["heuristic_anchor"]
        and row["jev_anchor"] != row["heuristic_anchor"]
    ]
    differing.sort(
        key=lambda row: (len(row["members"]), row["physical_base"], row["seed"])
    )
    picked: list[dict] = []
    represented: set[str] = set()
    for row in differing:
        if len(row["members"]) <= 4 and row["kind"] not in represented:
            picked.append(row)
            represented.add(row["kind"])
            if len(picked) == 10:
                return picked
    for row in differing:
        if row not in picked:
            picked.append(row)
            if len(picked) == 10:
                break
    return picked


def make_instance(root: Path, dump: Path, label: str, port: int):
    place = root / label
    for subdir in ("data", "dumps", "logs", "run", "tmp"):
        (place / subdir).mkdir(parents=True, exist_ok=True)
    os.link(dump, place / "dumps/neo4j.dump")
    previous_bind = os.environ.get("APPTAINER_BIND")
    tmp_bind = f"{place / 'tmp'}:/tmp"
    os.environ["APPTAINER_BIND"] = (
        f"{previous_bind},{tmp_bind}" if previous_bind else tmp_bind
    )
    try:
        proc, log = start_temp_neo4j(place, port, port + 1000)
    finally:
        if previous_bind is None:
            os.environ.pop("APPTAINER_BIND", None)
        else:
            os.environ["APPTAINER_BIND"] = previous_bind
    return place, proc, log


def cli(seed: str, uri: str, *flags: str) -> list[dict] | None:
    command = [
        "uv",
        "run",
        "imas-codex",
        "sn",
        "harmonize",
        "--family",
        seed,
        *flags,
    ]
    env = dict(
        os.environ,
        UV_NO_SYNC="1",
        NEO4J_URI=uri,
        UV_PROJECT_ENVIRONMENT="/home/ITER/mcintos/Code/imas-codex/.venv",
        PYTHONPATH=str(Path.cwd()),
    )
    print("command: " + " ".join(command), flush=True)
    remaining = (DEADLINE - datetime.now(UTC)).total_seconds()
    if "--json" not in flags:
        completed = subprocess.run(
            command, env=env, check=False, timeout=max(1, remaining)
        )
        if completed.returncode:
            raise RuntimeError(
                f"Harmonize command failed for {seed}: {completed.returncode}"
            )
        return None
    from imas_codex.standard_names import harmonize

    arm = os.environ.get("ANCHOR_ARM", "unknown")
    log_path = RUN_DIR / f"apply-{arm}-{seed}.log"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    with log_path.open("w") as stream:
        stream.write(
            f"revision={revision} tree={Path.cwd()} command={' '.join(command)}\n"
        )
        stream.write(
            f"module.__file__={harmonize.__file__}\nresolved_cwd={Path.cwd().resolve()}\n"
        )
        stream.flush()
        try:
            completed = subprocess.run(
                command,
                env=env,
                check=False,
                timeout=max(1, remaining),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        except subprocess.TimeoutExpired:
            stream.write("EXIT=124\n")
            raise
        stream.write(f"EXIT={completed.returncode}\n")
    print(f"apply log: {log_path} EXIT={completed.returncode}", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"Harmonize command failed for {seed}: {completed.returncode}; {log_path}"
        )
    lines = log_path.read_text().splitlines()
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(parsed, list)
            and parsed
            and isinstance(parsed[0], dict)
            and "result" in parsed[0]
        ):
            return parsed
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].lstrip().startswith("[{"):
            try:
                parsed = json.loads("\n".join(lines[index:-1]))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list) and parsed and "result" in parsed[0]:
                return parsed
    raise RuntimeError(f"Harmonize returned no JSON receipt for {seed}: {log_path}")


def scores(gc: GraphClient, seed: str) -> dict[str, float | None]:
    family = fetch_pattern_families(gc, seeds=[seed])
    if len(family) != 1:
        raise RuntimeError(f"Expected one family for {seed}, got {len(family)}")
    return {m["id"]: m.get("reviewer_score_docs") for m in family[0]["members"]}


def measure_arm(
    gc: GraphClient, uri: str, selected: list[dict], arm: str, data: dict
) -> None:
    for row in selected:
        require_time()
        seed = row["seed"]
        plan = plan_family_applies(gc, seeds=[seed])
        if len(plan) != 1 or plan[0]["decision"] != "apply":
            raise RuntimeError(f"Family {seed} no longer plans one apply: {plan}")
        if arm == "jev":
            _record_family_anchor(gc, row["members"], row["jev_anchor"])
            plan = plan_family_applies(gc, seeds=[seed])
            if plan[0]["anchor"] != row["jev_anchor"]:
                raise RuntimeError(f"Anchor pin did not resolve for {seed}")
        before = report_family_rows(gc, seeds=[seed])[0]
        before_scores = scores(gc, seed)
        ceiling = min(
            apply_cap(int(plan[0]["planned_calls"])),
            SPEND_LIMIT - data.get("budget_spend_usd", data["spend_usd"]) - 0.02,
        )
        if ceiling <= 0:
            raise RuntimeError("No budget remains before next apply")
        ledger_baseline_rows = int(
            gc.query("MATCH (c:LLMCost) RETURN count(c) AS n")[0]["n"]
        )
        if ledger_baseline_rows == 0:
            raise RuntimeError("LLMCost receipt query missed known snapshot rows")
        started_at = now()
        started = time.monotonic()
        try:
            outcome = cli(
                seed, uri, "--apply", "--cost-limit", f"{ceiling:.2f}", "--json"
            )
        except Exception as exc:
            ended_at = now()
            ledger = gc.query(
                "MATCH (c:LLMCost) "
                "WHERE c.llm_at >= datetime($started) AND c.llm_at <= datetime($ended) "
                "RETURN count(c) AS calls, coalesce(sum(c.llm_cost), 0.0) AS spend",
                started=started_at,
                ended=ended_at,
            )[0]
            partial_spend = float(ledger["spend"])
            data["spend_usd"] += partial_spend
            if "confirmed_spend_usd" in data:
                data["confirmed_spend_usd"] += partial_spend
            if "budget_spend_usd" in data:
                data["budget_spend_usd"] += partial_spend
            data.setdefault("failed_applies", []).append(
                {
                    "arm": arm,
                    "seed": seed,
                    "anchor": plan[0]["anchor"],
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "ledger_calls": int(ledger["calls"]),
                    "ledger_spend_usd": partial_spend,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            save(data)
            raise
        elapsed = time.monotonic() - started
        ended_at = now()
        after = report_family_rows(gc, seeds=[seed])[0]
        after_scores = scores(gc, seed)
        summary = outcome[0]["result"]
        run_ids = summary["run_ids"]
        if not run_ids:
            raise RuntimeError(f"Apply for {seed} returned no scope run IDs")
        ledger = gc.query(
            "MATCH (c:LLMCost) "
            "WHERE c.llm_at >= datetime($started) AND c.llm_at <= datetime($ended) "
            "RETURN count(c) AS calls, coalesce(sum(c.llm_cost), 0.0) AS spend, "
            "collect(DISTINCT c.run_id) AS session_ids",
            started=started_at,
            ended=ended_at,
        )[0]
        model_receipts = gc.query(
            "MATCH (c:LLMCost) "
            "WHERE c.llm_at >= datetime($started) AND c.llm_at <= datetime($ended) "
            "RETURN c.llm_model AS model, count(c) AS calls, "
            "sum(c.llm_cost) AS spend ORDER BY model",
            started=started_at,
            ended=ended_at,
        )
        calls = int(ledger["calls"])
        spend = float(ledger["spend"])
        reported = float(summary["total_cost"])
        if reported > 0 and (calls == 0 or abs(spend - reported) > 0.02):
            raise RuntimeError(
                f"Apply receipt mismatch for {seed}: ledger={ledger}, reported={reported}"
            )
        receipt = {
            "harmonize_summary": summary,
            "scope_run_ids": run_ids,
            "ledger_window": {"started": started_at, "ended": ended_at},
            "ledger_baseline_rows": ledger_baseline_rows,
            "ledger_calls": calls,
            "ledger_spend_usd": spend,
            "ledger_session_ids": ledger["session_ids"],
            "per_model": model_receipts,
        }
        data["spend_usd"] += reported
        if "confirmed_spend_usd" in data:
            data["confirmed_spend_usd"] += reported
        if "budget_spend_usd" in data:
            data["budget_spend_usd"] += reported
        data.setdefault("arms", {}).setdefault(arm, []).append(
            {
                "seed": seed,
                "anchor": plan[0]["anchor"],
                "members": row["members"],
                "lint_before": before["lint_total"],
                "lint_after": after["lint_total"],
                "lint_before_classes": before["lint"],
                "lint_after_classes": after["lint"],
                "scores_before": before_scores,
                "scores_after": after_scores,
                "planned_calls": plan[0]["planned_calls"],
                "calls": calls,
                "receipt": receipt,
                "spend_usd": reported,
                "wall_seconds": elapsed,
                "stamped": plan_family_applies(gc, seeds=[seed])[0]["decision"]
                == "harmonized",
            }
        )
        save(data)
        print(
            f"{arm} {seed}: lint {before['lint_total']}->{after['lint_total']}, receipt ${reported:.4f}, {elapsed:.1f}s",
            flush=True,
        )


def arm_worker(arm: str) -> None:
    selected = json.loads((RUN_DIR / "arm-input.json").read_text())
    data = {"arm": arm, "spend_usd": 0.0, "arms": {arm: []}}
    save(data)
    with GraphClient() as gc:
        measure_arm(gc, os.environ["NEO4J_URI"], selected, arm, data)


def start_arm_worker(uri: str) -> tuple[subprocess.Popen, Path, object]:
    log_path = RUN_DIR / "arm-jev.log"
    stream = log_path.open("w")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    stream.write(
        f"revision={revision} tree={Path.cwd()} command=uv run python {Path(__file__).resolve()} --arm-worker jev\n"
    )
    stream.write(
        f"module.__file__={Path(__file__).resolve()}\nresolved_cwd={Path.cwd().resolve()}\n"
    )
    stream.flush()
    env = dict(
        os.environ,
        UV_NO_SYNC="1",
        NEO4J_URI=uri,
        ANCHOR_ARM="jev",
        ANCHOR_RESULTS_PATH=str(RUN_DIR / "arm-jev-results.json"),
        UV_PROJECT_ENVIRONMENT="/home/ITER/mcintos/Code/imas-codex/.venv",
        PYTHONPATH=str(Path.cwd()),
    )
    proc = subprocess.Popen(
        ["uv", "run", "python", str(Path(__file__).resolve()), "--arm-worker", "jev"],
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"Jev arm launched pid={proc.pid}, log={log_path}", flush=True)
    return proc, log_path, stream


def resume_applies() -> None:
    data = json.loads(RESULTS.read_text())
    if data["source_export"] != str(ARCHIVE):
        raise RuntimeError("Cached selection belongs to another graph export")
    selected = data["selected"]
    if len(selected) != 10 or len(data["dry_runs"]) != 10:
        raise RuntimeError("Ten selected families and dry runs are required")
    if data["adoption_rule"]["registered_at"] != RULE_REGISTERED_AT:
        raise RuntimeError("Pre-registered adoption rule changed")
    initial_spend = float(data["spend_usd"])
    cap_envelope = initial_spend + 2 * sum(
        apply_cap(item["calls_per_arm"]) for item in data["dry_runs"]
    )
    if cap_envelope >= SPEND_LIMIT:
        raise RuntimeError(f"Apply caps ${cap_envelope:.4f} reach ${SPEND_LIMIT:.2f}")
    data.update(
        spend_limit_usd=SPEND_LIMIT,
        cap_envelope_usd=cap_envelope,
        resumed_at=now(),
        ruling="2026-10-18 lead: USD 42.00 total including prior spend; all ten paired families",
        arms={},
        isolation_proof=[],
        stopped_instances=[],
        scratch_removed=None,
        verdict=None,
        outcome_status="applying",
    )
    data.pop("failure", None)
    save(data)
    scratch = Path(tempfile.mkdtemp(prefix="anchor-compare-", dir=os.environ["TMPDIR"]))
    data["scratch_tree"] = str(scratch)
    save(data)
    live_uri = GraphClient().uri
    if live_uri.startswith(("bolt://127.0.0.1", "bolt://localhost")):
        raise RuntimeError("Live URI unexpectedly resolves to loopback")
    instances = []
    child_proc = None
    child_stream = None
    try:
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            member = next(item for item in archive if item.name.endswith("/graph.dump"))
            dump = scratch / "neo4j.dump"
            with archive.extractfile(member) as source, dump.open("wb") as target:
                shutil.copyfileobj(source, target)
        print(f"extracted dump: {dump.stat().st_size} bytes", flush=True)
        for arm, port in (("heuristic", 27697), ("jev", 27698)):
            require_time()
            place, proc, log = make_instance(scratch, dump, arm, port)
            instances.append((arm, place, proc))
            uri = f"bolt://127.0.0.1:{port}"
            with GraphClient(uri=uri) as gc:
                prove_isolation(gc, live_uri, uri, arm, data, log)
        (RUN_DIR / "arm-input.json").write_text(json.dumps(selected, indent=2) + "\n")
        child_proc, child_log, child_stream = start_arm_worker("bolt://127.0.0.1:27698")
        heuristic_uri = "bolt://127.0.0.1:27697"
        os.environ["NEO4J_URI"] = heuristic_uri
        os.environ["ANCHOR_ARM"] = "heuristic"
        with GraphClient(uri=heuristic_uri) as gc:
            measure_arm(gc, heuristic_uri, selected, "heuristic", data)
        remaining = (DEADLINE - datetime.now(UTC)).total_seconds()
        child_proc.wait(timeout=max(1, remaining))
        child_stream.write(f"EXIT={child_proc.returncode}\n")
        child_stream.close()
        child_stream = None
        if child_proc.returncode:
            raise RuntimeError(f"Jev arm failed: {child_log}")
        jev_data = json.loads((RUN_DIR / "arm-jev-results.json").read_text())
        jev_rows = jev_data.get("arms", {}).get("jev", [])
        if len(jev_rows) != 10:
            raise RuntimeError(f"Jev arm recorded {len(jev_rows)} families")
        data["arms"]["jev"] = jev_rows
        data["spend_usd"] += jev_data["spend_usd"]
        data["arm_worker_log"] = str(child_log)
        if data["spend_usd"] >= SPEND_LIMIT:
            raise RuntimeError("Combined spend reached the authorized ceiling")
        heur = {row["seed"]: row for row in data["arms"]["heuristic"]}
        jev = {row["seed"]: row for row in jev_rows}
        lower = []
        missing = []
        for seed, row in jev.items():
            for member, value in row["scores_after"].items():
                baseline = heur[seed]["scores_after"].get(member)
                if value is None or baseline is None:
                    missing.append({"seed": seed, "member": member})
                elif value < baseline:
                    lower.append(
                        {
                            "seed": seed,
                            "member": member,
                            "heuristic": baseline,
                            "jev": value,
                        }
                    )
        heur_lint = sum(row["lint_after"] for row in heur.values())
        jev_lint = sum(row["lint_after"] for row in jev.values())
        data["verdict"] = {
            "adopt_jev": jev_lint < heur_lint and not lower and not missing,
            "heuristic_lint_after": heur_lint,
            "jev_lint_after": jev_lint,
            "lower_scores": lower,
            "missing_comparisons": missing,
            "decided_at": now(),
        }
        data["outcome_status"] = "complete"
        save(data)
    except Exception as exc:
        data["failure"] = {"at": now(), "type": type(exc).__name__, "message": str(exc)}
        data["outcome_status"] = "blocked"
        save(data)
        raise
    finally:
        if child_proc is not None and child_proc.poll() is None:
            os.killpg(os.getpgid(child_proc.pid), signal.SIGTERM)
            try:
                child_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(child_proc.pid), signal.SIGKILL)
                child_proc.wait(timeout=10)
        if child_stream is not None:
            child_stream.write(
                f"EXIT={child_proc.returncode if child_proc else 'unknown'}\n"
            )
            child_stream.close()
        for arm, place, proc in instances:
            stop_temp_neo4j(proc)
            data["stopped_instances"].append(
                {"arm": arm, "tree": str(place), "at": now()}
            )
        shutil.rmtree(scratch)
        data["scratch_removed"] = {"tree": str(scratch), "at": now()}
        save(data)


def run_filtered_pairs() -> None:
    from imas_codex.standard_names import harmonize

    module_path = str(Path(harmonize.__file__).resolve())
    if not module_path.startswith("/home/ITER/mcintos/Code/imas-codex/"):
        raise RuntimeError(f"Driver imported non-main harmonize code: {module_path}")
    data = json.loads(RESULTS.read_text())
    if data["source_export"] != str(ARCHIVE):
        raise RuntimeError("Cached anchor scores belong to another export")
    if data["adoption_rule"]["registered_at"] != RULE_REGISTERED_AT:
        raise RuntimeError("Pre-registered adoption rule changed")
    confirmed_prior = float(data["spend_usd"])
    unreceipted_reserve = float(data["budget_reservation"]["usd"])
    data["previous_selected"] = data["selected"]
    data["previous_isolation_proof"] = data.get("isolation_proof", [])
    data.update(
        resumed_at=now(),
        measurement_module=module_path,
        measurement_cwd=str(Path.cwd().resolve()),
        model_seats={
            "sn-docs": get_model("sn-docs"),
            "sn-refine": get_model("sn-refine"),
            "sn-review.docs": get_sn_review_docs_models(),
        },
        spend_limit_usd=SPEND_LIMIT,
        confirmed_spend_usd=confirmed_prior,
        budget_spend_usd=confirmed_prior + unreceipted_reserve,
        budget_reservation={
            "reason": "one unreceipted Jev call at the snapshot dry-run rate",
            "usd": unreceipted_reserve,
        },
        selected=[],
        dry_runs=[],
        arms={},
        isolation_proof=[],
        stopped_instances=[],
        scratch_removed=None,
        verdict=None,
        outcome_status="selecting",
        paired_family_count=0,
        failed_applies=[],
    )
    data.pop("failure", None)
    save(data)
    west, roster_evidence = west_roster()
    data["west_search"] = {"files": roster_evidence, "distinct_names": len(west)}
    save(data)
    scratch = Path(tempfile.mkdtemp(prefix="anchor-compare-", dir=os.environ["TMPDIR"]))
    data["scratch_tree"] = str(scratch)
    save(data)
    live_uri = GraphClient().uri
    if live_uri.startswith(("bolt://127.0.0.1", "bolt://localhost")):
        raise RuntimeError("Live URI unexpectedly resolves to loopback")
    instances = []
    try:
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            member = next(item for item in archive if item.name.endswith("/graph.dump"))
            dump = scratch / "neo4j.dump"
            with archive.extractfile(member) as source, dump.open("wb") as target:
                shutil.copyfileobj(source, target)
        print(f"extracted dump: {dump.stat().st_size} bytes", flush=True)
        heur_place, heur_proc, heur_log = make_instance(
            scratch, dump, "heuristic", 27697
        )
        instances.append(("heuristic", heur_place, heur_proc))
        heur_uri = "bolt://127.0.0.1:27697"
        with GraphClient(uri=heur_uri) as heur_gc:
            prove_isolation(heur_gc, live_uri, heur_uri, "heuristic", data, heur_log)
            families = fetch_pattern_families(heur_gc, min_size=1)
            base_eligible = []
            exclusions = {"too_small": 0, "west_roster": 0, "harmonized": 0}
            for family in families:
                members = family["members"]
                ids = {m["id"] for m in members}
                if len(ids) < 3:
                    exclusions["too_small"] += 1
                elif ids & west:
                    exclusions["west_roster"] += 1
                elif {
                    m.get("harmonized_group_signature")
                    for m in members
                    if m.get("harmonized_group_signature")
                } == {group_signature(members)}:
                    exclusions["harmonized"] += 1
                else:
                    base_eligible.append(family)
            member_ids = sorted(
                {m["id"] for family in base_eligible for m in family["members"]}
            )
            statuses = heur_gc.query(
                "UNWIND $ids AS sid MATCH (m:StandardName {id: sid}) "
                "RETURN m.id AS id, m.validation_status AS validation_status, "
                "m.docs_stage AS docs_stage, m.name_stage AS name_stage",
                ids=member_ids,
            )
            by_id = {row["id"]: row for row in statuses}
            if len(by_id) != len(member_ids):
                raise RuntimeError("Validation read omitted known family members")
            if not any(row["validation_status"] for row in statuses) or not any(
                row["docs_stage"] or row["name_stage"] for row in statuses
            ):
                raise RuntimeError("Validation read saw no known status values")
            excluded = {"quarantined": 0, "docs_exhausted": 0, "name_exhausted": 0}
            eligible = []
            removed = []
            for family in base_eligible:
                reasons = set()
                for member in family["members"]:
                    row = by_id[member["id"]]
                    if row["validation_status"] == "quarantined":
                        reasons.add("quarantined")
                    if row["docs_stage"] == "exhausted":
                        reasons.add("docs_exhausted")
                    if row["name_stage"] == "exhausted":
                        reasons.add("name_exhausted")
                if reasons:
                    removed.append(
                        {
                            "seed": min(m["id"] for m in family["members"]),
                            "reasons": sorted(reasons),
                        }
                    )
                    for reason in reasons:
                        excluded[reason] += 1
                else:
                    eligible.append(family)
            data["candidate_pool"] = {
                "all_families": len(families),
                "before_validation": len(base_eligible),
                "eligible": len(eligible),
                "exclusions": exclusions,
                "validation_removed": len(removed),
                "validation_exclusions": excluded,
            }
            data["validation_search"] = {
                "at": now(),
                "read_from": heur_uri,
                "properties": ["validation_status", "docs_stage", "name_stage"],
                "member_rows": len(statuses),
                "removed_families": removed,
            }
            save(data)
            print(f"candidate pool: {data['candidate_pool']}", flush=True)
            cached = {row["seed"]: row for row in data["scored_candidates"]}
            candidates = []
            for family in eligible:
                ids = sorted(m["id"] for m in family["members"])
                row = cached.get(ids[0])
                if row is None or row["members"] != ids:
                    raise RuntimeError(f"Cached anchor score missing for {ids[0]}")
                candidates.append(row)
            data["same_anchor_count"] = sum(
                row["jev_anchor"] == row["heuristic_anchor"]
                for row in candidates
                if row["jev_anchor"]
            )
            data["unscored_count"] = sum(
                row["jev_anchor"] is None for row in candidates
            )
            selected = select(candidates)
            if len(selected) != 10:
                raise RuntimeError(
                    f"Only {len(selected)} differing-anchor families remain"
                )
            data["selected"] = selected
            estimate = measure_doc_call_cost(heur_gc)
            if estimate is None:
                raise RuntimeError("Snapshot lacks a measured documentation-call rate")
            rotated = rotated_doc_call_cost(heur_gc, estimate.calls)
            rate = rotated["rate_usd"]
            data["dry_run_estimate_basis"] = {
                "source": "snapshot LLMCost token usage repriced for active model seats",
                "runs": estimate.runs,
                "calls": estimate.calls,
                "historical_rate_usd": estimate.mean_usd,
                "rotated": rotated,
                "rate_usd": rate,
            }
            data["dry_run_estimate_usd_per_call"] = rate
            data["dry_runs"] = []
            for row in selected:
                cli(row["seed"], heur_uri, "--apply", "--dry-run")
                planned = plan_family_applies(heur_gc, seeds=[row["seed"]])[0]
                if planned["decision"] != "apply" or planned["planned_calls"] != len(
                    row["members"]
                ):
                    raise RuntimeError(
                        f"Unexpected dry-run plan for {row['seed']}: {planned}"
                    )
                data["dry_runs"].append(
                    {
                        "seed": row["seed"],
                        "calls_per_arm": planned["planned_calls"],
                        "estimate_per_arm_usd": planned["planned_calls"] * rate,
                    }
                )
                save(data)
            estimated = 2 * sum(
                item["estimate_per_arm_usd"] for item in data["dry_runs"]
            )
            data["dry_run_estimate_total_usd"] = estimated
            data["cap_envelope_usd"] = data["budget_spend_usd"] + 2 * sum(
                apply_cap(item["calls_per_arm"]) for item in data["dry_runs"]
            )
            save(data)
            if (
                data["budget_spend_usd"] + estimated >= SPEND_LIMIT
                or data["cap_envelope_usd"] >= SPEND_LIMIT
            ):
                raise RuntimeError(
                    f"Dry-run estimate or apply cap reaches USD {SPEND_LIMIT:.2f}"
                )
            jev_place, jev_proc, jev_log = make_instance(scratch, dump, "jev", 27698)
            instances.append(("jev", jev_place, jev_proc))
            jev_uri = "bolt://127.0.0.1:27698"
            with GraphClient(uri=jev_uri) as jev_gc:
                prove_isolation(jev_gc, live_uri, jev_uri, "jev", data, jev_log)
                prove_isolation(
                    heur_gc, live_uri, heur_uri, "heuristic", data, heur_log
                )
                data["outcome_status"] = "applying"
                save(data)
                for row in selected:
                    require_time()
                    for arm, uri, gc in (
                        ("heuristic", heur_uri, heur_gc),
                        ("jev", jev_uri, jev_gc),
                    ):
                        os.environ["NEO4J_URI"] = uri
                        os.environ["ANCHOR_ARM"] = arm
                        measure_arm(gc, uri, [row], arm, data)
                    data["paired_family_count"] += 1
                    save(data)
        heur = {row["seed"]: row for row in data["arms"]["heuristic"]}
        jev = {row["seed"]: row for row in data["arms"]["jev"]}
        if len(heur) != 10 or len(jev) != 10:
            raise RuntimeError("Ten complete families are required in each arm")
        lower = []
        missing = []
        for seed, row in jev.items():
            for member, value in row["scores_after"].items():
                baseline = heur[seed]["scores_after"].get(member)
                if value is None or baseline is None:
                    missing.append({"seed": seed, "member": member})
                elif value < baseline:
                    lower.append(
                        {
                            "seed": seed,
                            "member": member,
                            "heuristic": baseline,
                            "jev": value,
                        }
                    )
        heur_lint = sum(row["lint_after"] for row in heur.values())
        jev_lint = sum(row["lint_after"] for row in jev.values())
        data["verdict"] = {
            "adopt_jev": jev_lint < heur_lint and not lower and not missing,
            "heuristic_lint_after": heur_lint,
            "jev_lint_after": jev_lint,
            "lower_scores": lower,
            "missing_comparisons": missing,
            "decided_at": now(),
        }
        data["outcome_status"] = "complete"
        save(data)
    except Exception as exc:
        data["failure"] = {"at": now(), "type": type(exc).__name__, "message": str(exc)}
        data["outcome_status"] = "blocked"
        save(data)
        raise
    finally:
        for arm, place, proc in instances:
            stop_temp_neo4j(proc)
            data["stopped_instances"].append(
                {"arm": arm, "tree": str(place), "at": now()}
            )
        shutil.rmtree(scratch)
        data["scratch_removed"] = {"tree": str(scratch), "at": now()}
        save(data)


def main() -> None:
    data: dict = {
        "source_export": str(ARCHIVE),
        "started_at": now(),
        "spend_limit_usd": SPEND_LIMIT,
        "previous_spend_usd": PREVIOUS_SPEND,
        "spend_usd": PREVIOUS_SPEND,
        "adoption_rule": {
            "registered_at": RULE_REGISTERED_AT,
            "text": "Adopt the Jev anchor only if its arm has fewer total lint findings after harmonization across the families and no member's documentation review score is lower than that same member's score in the heuristic arm.",
        },
    }
    save(data)
    west, roster_evidence = west_roster()
    data["west_search"] = {"files": roster_evidence, "distinct_names": len(west)}
    save(data)
    scratch = Path(tempfile.mkdtemp(prefix="anchor-compare-", dir=os.environ["TMPDIR"]))
    data["scratch_tree"] = str(scratch)
    save(data)
    print(f"scratch tree: {scratch}", flush=True)
    live_uri = GraphClient().uri
    if live_uri.startswith("bolt://127.0.0.1") or live_uri.startswith(
        "bolt://localhost"
    ):
        raise RuntimeError("Live URI unexpectedly resolves to loopback")
    active_procs = []
    child_proc = None
    child_stream = None
    try:
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            member = next(item for item in archive if item.name.endswith("/graph.dump"))
            dump = scratch / "neo4j.dump"
            with archive.extractfile(member) as source, dump.open("wb") as target:
                shutil.copyfileobj(source, target)
        print(f"extracted dump: {dump.stat().st_size} bytes", flush=True)
        for arm, port in (("heuristic", 27697),):
            require_time()
            place, active_proc, log = make_instance(scratch, dump, arm, port)
            active_procs.append(active_proc)
            uri = f"bolt://127.0.0.1:{port}"
            os.environ["NEO4J_URI"] = uri
            with GraphClient() as gc:
                prove_isolation(gc, live_uri, uri, arm, data, log)
                if arm == "heuristic":
                    families = fetch_pattern_families(gc, min_size=1)
                    eligible = []
                    exclusions = {"too_small": 0, "harmonized": 0, "west_roster": 0}
                    for family in families:
                        ids = {m["id"] for m in family["members"]}
                        if len(ids) < 3:
                            exclusions["too_small"] += 1
                        elif ids & west:
                            exclusions["west_roster"] += 1
                        elif {
                            m.get("harmonized_group_signature")
                            for m in family["members"]
                            if m.get("harmonized_group_signature")
                        } == {group_signature(family["members"])}:
                            exclusions["harmonized"] += 1
                        else:
                            eligible.append(family)
                    data["candidate_pool"] = {
                        "all_families": len(families),
                        "eligible": len(eligible),
                        "exclusions": exclusions,
                    }
                    save(data)
                    print(f"candidate pool: {data['candidate_pool']}", flush=True)
                    ids = sorted(
                        {
                            member["id"]
                            for family in eligible
                            for member in family["members"]
                        }
                    )
                    trait_rows = gc.query(
                        "UNWIND $ids AS sid MATCH (m:StandardName {id: sid}) "
                        "RETURN m.id AS id, m.subject AS subject, m.component AS component, "
                        "m.coordinate AS coordinate, m.position AS position, m.geometry AS geometry, "
                        "m.region AS region, m.device AS device",
                        ids=ids,
                    )
                    traits = {row["id"]: row for row in trait_rows}
                    if len(traits) != len(ids):
                        raise RuntimeError(
                            "Variation-axis read omitted eligible members"
                        )
                    for family in eligible:
                        family["kind"] = kind(family, traits)
                    if all(family["kind"] == "unclassified" for family in eligible):
                        raise RuntimeError(
                            "Variation-axis classification saw no nonuniform property"
                        )
                    data["variation_kind_counts"] = {
                        label: sum(family["kind"] == label for family in eligible)
                        for label in sorted({family["kind"] for family in eligible})
                    }
                    save(data)
                    candidates = asyncio.run(scored_candidates(eligible, data))
                    data["same_anchor_count"] = sum(
                        row["jev_anchor"] == row["heuristic_anchor"]
                        for row in candidates
                        if row["jev_anchor"]
                    )
                    data["unscored_count"] = sum(
                        row["jev_anchor"] is None for row in candidates
                    )
                    data["scorer_spend_usd"] = sum(
                        row["jev_cost"] for row in candidates
                    )
                    data["spend_usd"] = PREVIOUS_SPEND + data["scorer_spend_usd"]
                    selected = select(candidates)
                    data["selected"] = selected
                    if len(selected) < 10:
                        data["selection_shortfall"] = (
                            f"Only {len(selected)} eligible differing-anchor families"
                        )
                    estimate = measure_doc_call_cost(gc)
                    if estimate is None:
                        rate = PRIOR_DOC_CALL_COST
                        data["dry_run_estimate_basis"] = {
                            "source": "prior measured documentation-call rate; snapshot CLI reports unavailable",
                            "path": PRIOR_COST_EVIDENCE,
                            "rate_usd": rate,
                        }
                    else:
                        rate = estimate.mean_usd
                        data["dry_run_estimate_basis"] = {
                            "source": "snapshot LLMCost rows",
                            "runs": estimate.runs,
                            "calls": estimate.calls,
                            "rate_usd": rate,
                        }
                    data["dry_run_estimate_usd_per_call"] = rate
                    data["dry_runs"] = []
                    for row in selected:
                        cli(row["seed"], uri, "--apply", "--dry-run")
                        planned = plan_family_applies(gc, seeds=[row["seed"]])[0]
                        data["dry_runs"].append(
                            {
                                "seed": row["seed"],
                                "calls_per_arm": planned["planned_calls"],
                                "estimate_per_arm_usd": planned["planned_calls"] * rate,
                            }
                        )
                        save(data)
                    estimated = 2 * sum(
                        item["estimate_per_arm_usd"] for item in data["dry_runs"]
                    )
                    data["dry_run_estimate_total_usd"] = estimated
                    cap_envelope = (
                        PREVIOUS_SPEND
                        + data["scorer_spend_usd"]
                        + 2
                        * sum(
                            apply_cap(item["calls_per_arm"])
                            for item in data["dry_runs"]
                        )
                    )
                    data["cap_envelope_usd"] = cap_envelope
                    save(data)
                    if (
                        estimated + PREVIOUS_SPEND + data["scorer_spend_usd"]
                        >= SPEND_LIMIT
                        or cap_envelope >= SPEND_LIMIT
                    ):
                        raise RuntimeError(
                            f"Dry-run estimate ${estimated:.2f} or cap ${cap_envelope:.2f} reaches budget"
                        )
                    jev_place, jev_proc, jev_log = make_instance(
                        scratch, dump, "jev", 27698
                    )
                    active_procs.append(jev_proc)
                    jev_uri = "bolt://127.0.0.1:27698"
                    with GraphClient(uri=jev_uri) as jev_gc:
                        prove_isolation(jev_gc, live_uri, jev_uri, "jev", data, jev_log)
                    (RUN_DIR / "arm-input.json").write_text(
                        json.dumps(selected, indent=2) + "\n"
                    )
                    child_proc, child_log, child_stream = start_arm_worker(jev_uri)
                    os.environ["NEO4J_URI"] = uri
                    os.environ["ANCHOR_ARM"] = "heuristic"
                    measure_arm(gc, uri, selected, arm, data)
                    remaining = (DEADLINE - datetime.now(UTC)).total_seconds()
                    child_proc.wait(timeout=max(1, remaining))
                    child_stream.write(f"EXIT={child_proc.returncode}\n")
                    child_stream.close()
                    child_stream = None
                    if child_proc.returncode:
                        raise RuntimeError(f"Jev arm failed: {child_log}")
                    jev_data = json.loads(
                        (RUN_DIR / "arm-jev-results.json").read_text()
                    )
                    jev_rows = jev_data.get("arms", {}).get("jev", [])
                    if len(jev_rows) != len(selected):
                        raise RuntimeError(
                            f"Jev arm recorded {len(jev_rows)} of {len(selected)} families"
                        )
                    data.setdefault("arms", {})["jev"] = jev_rows
                    data["spend_usd"] += jev_data["spend_usd"]
                    data["arm_worker_log"] = str(child_log)
                    save(data)
            for finished_arm, finished_place, finished_proc in (
                ("heuristic", place, active_proc),
                ("jev", jev_place, jev_proc),
            ):
                stop_temp_neo4j(finished_proc)
                data.setdefault("stopped_instances", []).append(
                    {"arm": finished_arm, "tree": str(finished_place), "at": now()}
                )
            save(data)
        heur = {row["seed"]: row for row in data["arms"]["heuristic"]}
        jev = {row["seed"]: row for row in data["arms"]["jev"]}
        heur_lint = sum(row["lint_after"] for row in heur.values())
        jev_lint = sum(row["lint_after"] for row in jev.values())
        lower = []
        missing = []
        for seed, row in jev.items():
            for member, value in row["scores_after"].items():
                baseline = heur[seed]["scores_after"].get(member)
                if value is None or baseline is None:
                    missing.append({"seed": seed, "member": member})
                elif value < baseline:
                    lower.append(
                        {
                            "seed": seed,
                            "member": member,
                            "heuristic": baseline,
                            "jev": value,
                        }
                    )
        data["verdict"] = {
            "adopt_jev": jev_lint < heur_lint and not lower and not missing,
            "heuristic_lint_after": heur_lint,
            "jev_lint_after": jev_lint,
            "lower_scores": lower,
            "missing_comparisons": missing,
            "decided_at": now(),
        }
        save(data)
    finally:
        if child_proc is not None and child_proc.poll() is None:
            print(
                f"stopping own Jev arm process group pid={child_proc.pid}", flush=True
            )
            os.killpg(os.getpgid(child_proc.pid), signal.SIGTERM)
            try:
                child_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print(
                    f"Jev arm pid={child_proc.pid} survived SIGTERM; escalating",
                    flush=True,
                )
                os.killpg(os.getpgid(child_proc.pid), signal.SIGKILL)
                child_proc.wait(timeout=10)
        if child_stream is not None:
            child_stream.write(
                f"EXIT={child_proc.returncode if child_proc else 'unknown'}\n"
            )
            child_stream.close()
        for process in active_procs:
            stop_temp_neo4j(process)
        shutil.rmtree(scratch)
        data["scratch_removed"] = {"tree": str(scratch), "at": now()}
        save(data)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--arm-worker":
        arm_worker(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "--resume-applies":
        resume_applies()
    elif len(sys.argv) == 2 and sys.argv[1] == "--filtered-pairs":
        run_filtered_pairs()
    else:
        main()

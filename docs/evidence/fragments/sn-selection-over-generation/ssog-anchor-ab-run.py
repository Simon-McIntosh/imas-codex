"""Measure paired family harmonization from one archived graph snapshot."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
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
    / ".local/share/imas-codex/exports/imas-codex-graph-dev-ff4d0f1-20261012T035511Z.tar.gz"
)
RUN_DIR = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261018T102210638794-ssog-anchor-ab"
)
RESULTS = Path(__file__).with_name("ssog-anchor-ab-results.json")
ROSTERS = Path("imas_codex/standard_names/manifests/reviews")
SPEND_LIMIT = 30.0
DEADLINE = datetime.fromisoformat("2026-10-18T11:22:36+00:00")


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


def kind(family: dict) -> str:
    labels = ("physical", "geometric", "transformation", "process", "object")
    values = family["invariant"]
    return (
        "+".join(label for label, value in zip(labels, values, strict=True) if value)
        or "unclassified"
    )


async def scored_candidates(families: list[dict], data: dict) -> list[dict]:
    semaphore = asyncio.Semaphore(4)
    results: list[dict] = []

    async def score(family: dict) -> dict:
        async with semaphore:
            require_time()
            members = family["members"]
            heuristic = resolve_family_anchor(members)
            judged = await score_family_anchor(
                None, None, None, members, pattern=kind(family)
            )
            return {
                "seed": sorted(m["id"] for m in members)[0],
                "physical_base": family["physical_base"],
                "kind": kind(family),
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
        if row["kind"] not in represented:
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
    proc, log = start_temp_neo4j(place, port, port + 1000)
    return place, proc, log


def cli(seed: str, uri: str, *flags: str) -> None:
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
    completed = subprocess.run(command, env=env, check=False, timeout=max(1, remaining))
    if completed.returncode:
        raise RuntimeError(
            f"Harmonize command failed for {seed}: {completed.returncode}"
        )


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
        ceiling = min(1.49, SPEND_LIMIT - data["spend_usd"] - 0.02)
        if ceiling <= 0:
            raise RuntimeError("No budget remains before next apply")
        started = time.monotonic()
        cli(seed, uri, "--apply", "--cost-limit", f"{ceiling:.2f}", "--json")
        elapsed = time.monotonic() - started
        after = report_family_rows(gc, seeds=[seed])[0]
        after_scores = scores(gc, seed)
        receipt = gc.query(
            "MATCH (r:SNRun) WHERE r.started_at IS NOT NULL "
            "OPTIONAL MATCH (c:LLMCost {for_run: r.id}) "
            "RETURN r.id AS id, r.cost_spent AS spend, r.started_at AS started, "
            "count(c) AS calls, sum(c.llm_cost) AS llm_spend "
            "ORDER BY r.started_at DESC LIMIT 1"
        )
        receipt = [{**item, "started": str(item["started"])} for item in receipt]
        spend = float(receipt[0].get("spend") or 0) if receipt else 0.0
        data["spend_usd"] += spend
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
                "receipt": receipt,
                "spend_usd": spend,
                "wall_seconds": elapsed,
                "stamped": plan_family_applies(gc, seeds=[seed])[0]["decision"]
                == "harmonized",
            }
        )
        save(data)
        print(
            f"{arm} {seed}: lint {before['lint_total']}->{after['lint_total']}, receipt ${spend:.4f}, {elapsed:.1f}s",
            flush=True,
        )


def main() -> None:
    data: dict = {
        "source_export": str(ARCHIVE),
        "started_at": now(),
        "spend_limit_usd": SPEND_LIMIT,
        "spend_usd": 0.0,
        "adoption_rule": {
            "registered_at": now(),
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
    active_proc = None
    try:
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            member = next(item for item in archive if item.name.endswith("/graph.dump"))
            dump = scratch / "neo4j.dump"
            with archive.extractfile(member) as source, dump.open("wb") as target:
                shutil.copyfileobj(source, target)
        print(f"extracted dump: {dump.stat().st_size} bytes", flush=True)
        for arm, port in (("heuristic", 27697), ("jev", 27698)):
            require_time()
            place, active_proc, log = make_instance(scratch, dump, arm, port)
            uri = f"bolt://127.0.0.1:{port}"
            os.environ["NEO4J_URI"] = uri
            with GraphClient() as gc:
                scratch_count = graph_count(gc)
                with GraphClient(uri=live_uri) as live:
                    live_count = graph_count(live)
                data.setdefault("isolation_proof", []).append(
                    {
                        "arm": arm,
                        "scratch_uri": uri,
                        "scratch_nodes": scratch_count,
                        "live_uri": live_uri,
                        "live_nodes": live_count,
                        "at": now(),
                        "neo4j_log": str(log),
                    }
                )
                save(data)
                print(
                    f"isolation {arm}: scratch {uri} {scratch_count}; live {live_uri} {live_count}",
                    flush=True,
                )
                if scratch_count == live_count or scratch_count <= 0:
                    raise RuntimeError("Scratch/live count contrast failed")
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
                    candidates = asyncio.run(scored_candidates(eligible, data))
                    data["same_anchor_count"] = sum(
                        row["jev_anchor"] == row["heuristic_anchor"]
                        for row in candidates
                        if row["jev_anchor"]
                    )
                    data["unscored_count"] = sum(
                        row["jev_anchor"] is None for row in candidates
                    )
                    selected = select(candidates)
                    data["selected"] = selected
                    if len(selected) < 10:
                        data["selection_shortfall"] = (
                            f"Only {len(selected)} eligible differing-anchor families"
                        )
                    estimate = measure_doc_call_cost(gc)
                    if estimate is None:
                        raise RuntimeError(
                            "Snapshot has no measured documentation call cost for dry-run estimate"
                        )
                    data["dry_run_estimate_usd_per_call"] = estimate.mean_usd
                    data["dry_runs"] = []
                    for row in selected:
                        cli(row["seed"], uri, "--apply", "--dry-run")
                        planned = plan_family_applies(gc, seeds=[row["seed"]])[0]
                        data["dry_runs"].append(
                            {
                                "seed": row["seed"],
                                "calls_per_arm": planned["planned_calls"],
                                "estimate_per_arm_usd": planned["planned_calls"]
                                * estimate.mean_usd,
                            }
                        )
                        save(data)
                    estimated = 2 * sum(
                        item["estimate_per_arm_usd"] for item in data["dry_runs"]
                    )
                    data["dry_run_estimate_total_usd"] = estimated
                    save(data)
                    if estimated >= SPEND_LIMIT:
                        raise RuntimeError(
                            f"Dry-run estimate ${estimated:.2f} reaches budget"
                        )
                else:
                    selected = data["selected"]
                measure_arm(gc, uri, selected, arm, data)
            stop_temp_neo4j(active_proc)
            active_proc = None
            data.setdefault("stopped_instances", []).append(
                {"arm": arm, "tree": str(place), "at": now()}
            )
            save(data)
        heur = {row["seed"]: row for row in data["arms"]["heuristic"]}
        jev = {row["seed"]: row for row in data["arms"]["jev"]}
        heur_lint = sum(row["lint_after"] for row in heur.values())
        jev_lint = sum(row["lint_after"] for row in jev.values())
        lower = []
        for seed, row in jev.items():
            for member, value in row["scores_after"].items():
                baseline = heur[seed]["scores_after"].get(member)
                if value is not None and baseline is not None and value < baseline:
                    lower.append(
                        {
                            "seed": seed,
                            "member": member,
                            "heuristic": baseline,
                            "jev": value,
                        }
                    )
        data["verdict"] = {
            "adopt_jev": jev_lint < heur_lint and not lower,
            "heuristic_lint_after": heur_lint,
            "jev_lint_after": jev_lint,
            "lower_scores": lower,
            "decided_at": now(),
        }
        save(data)
    finally:
        if active_proc is not None:
            stop_temp_neo4j(active_proc)
        shutil.rmtree(scratch)
        data["scratch_removed"] = {"tree": str(scratch), "at": now()}
        save(data)


if __name__ == "__main__":
    main()

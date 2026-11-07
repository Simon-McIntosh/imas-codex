"""Read graph state and verify the evidence chain around a catalog run."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.graph.offsite import live_graph_activity
from imas_codex.standard_names.export import run_export
from imas_codex.standard_names.graph_store.graph_supersession import (
    classify_sourceless_standard_names,
)
from imas_codex.standard_names.source_invariant import find_sourceless_live_names


def read_back(
    staging_dir: Path,
    *,
    graph_factory: Any = GraphClient,
    export_gate: Any = run_export,
    source_invariant: Any = find_sourceless_live_names,
    source_classifier: Any = classify_sourceless_standard_names,
) -> dict[str, Any]:
    """Read graph counts and consume the export owner's gate-only report."""
    with graph_factory() as graph:
        graph_nodes = graph.query("MATCH (n) RETURN count(n) AS total")[0]["total"]
        counts = graph.query(
            "MATCH (n:StandardName) RETURN count(n) AS total, "
            "count(n.id) AS with_id, "
            "sum(CASE WHEN n.name_stage = 'accepted' THEN 1 ELSE 0 END) AS accepted"
        )[0]
        if (
            not counts["total"]
            or counts["with_id"] != counts["total"]
            or not 0 <= counts["accepted"] <= counts["total"] <= graph_nodes
        ):
            raise RuntimeError(f"StandardName count control failed: {counts}")

        ids = [
            row["id"] for row in graph.query("MATCH (n:StandardName) RETURN n.id AS id")
        ]
        if len(ids) != counts["total"] or len(set(ids)) != counts["total"]:
            raise RuntimeError("StandardName id enumeration disagrees with count")
        sourceless_ids = set(source_invariant(graph, ids))
        source_population = [
            row
            for row in source_classifier(graph)
            if row["population"] in {"dropped_edge", "never_bound"}
        ]

    report = export_gate(staging_dir, gate_only=True)
    source_gates = [gate for gate in report.gate_results if gate.gate == "source_audit"]
    if len(source_gates) > 1:
        raise RuntimeError("export returned duplicate source-audit gates")
    blocked = sorted({issue["name"] for gate in source_gates for issue in gate.issues})
    drafts = sorted({row["name"] for row in report.sourceless_draft_names})
    if set(drafts) - sourceless_ids:
        raise RuntimeError("export drafts disagree with the source invariant")
    if set(blocked) - {row["id"] for row in source_population}:
        raise RuntimeError("export blockers fall outside the source audit")
    return {
        "graph_nodes_total": graph_nodes,
        "standard_names_total": counts["total"],
        "accepted": counts["accepted"],
        "sourceless_live_drafts": drafts,
        "export_source_audit_blocking_set": blocked,
        "sourceless_invariant_total": len(sourceless_ids),
        "source_audit_population_total": len(source_population),
    }


def require_clear_export(snapshot: dict[str, Any]) -> None:
    """Refuse a release when the export owner's source gate finds losses."""
    if names := snapshot["export_source_audit_blocking_set"]:
        raise RuntimeError(f"export source audit blocks {len(names)} name(s): {names}")


def test_read_back_contract() -> None:
    """Exercise counts, draft reporting, owner delegation, and refusal."""
    calls: list[tuple[Path, bool]] = []

    class Graph:
        def __enter__(self) -> "Graph":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def query(self, cypher: str) -> list[dict[str, Any]]:
            if cypher == "MATCH (n) RETURN count(n) AS total":
                return [{"total": 9}]
            if "count(n.id)" in cypher:
                return [{"total": 2, "with_id": 2, "accepted": 1}]
            if "RETURN n.id AS id" in cypher:
                return [{"id": "accepted_name"}, {"id": "orphan_draft"}]
            raise AssertionError(f"unexpected query: {cypher}")

    def owner(staging_dir: Path, *, gate_only: bool) -> SimpleNamespace:
        calls.append((staging_dir, gate_only))
        return SimpleNamespace(
            gate_results=[],
            sourceless_draft_names=[{"name": "orphan_draft"}],
        )

    staging = Path("unused-staging")
    snapshot = read_back(
        staging,
        graph_factory=Graph,
        export_gate=owner,
        source_invariant=lambda graph, ids: ["orphan_draft"],
        source_classifier=lambda graph: [
            {"id": "orphan_draft", "population": "never_bound"}
        ],
    )
    assert calls == [(staging, True)]
    assert snapshot == {
        "graph_nodes_total": 9,
        "standard_names_total": 2,
        "accepted": 1,
        "sourceless_live_drafts": ["orphan_draft"],
        "export_source_audit_blocking_set": [],
        "sourceless_invariant_total": 1,
        "source_audit_population_total": 1,
    }
    require_clear_export(snapshot)

    def blocked_owner(staging_dir: Path, *, gate_only: bool) -> SimpleNamespace:
        calls.append((staging_dir, gate_only))
        return SimpleNamespace(
            gate_results=[
                SimpleNamespace(gate="source_audit", issues=[{"name": "accepted_name"}])
            ],
            sourceless_draft_names=[{"name": "orphan_draft"}],
        )

    blocked = read_back(
        staging,
        graph_factory=Graph,
        export_gate=blocked_owner,
        source_invariant=lambda graph, ids: ["orphan_draft", "accepted_name"],
        source_classifier=lambda graph: [
            {"id": "orphan_draft", "population": "never_bound"},
            {"id": "accepted_name", "population": "dropped_edge"},
        ],
    )
    assert calls == [(staging, True), (staging, True)]
    assert blocked["export_source_audit_blocking_set"] == ["accepted_name"]
    try:
        require_clear_export(blocked)
    except RuntimeError as error:
        assert "accepted_name" in str(error)
    else:
        raise AssertionError("a planted export source loss did not refuse")


def self_check() -> dict[str, str]:
    test_read_back_contract()
    return {"counts_and_draft": "passed", "owner_gate": "passed", "refusal": "passed"}


def verify_baseline(
    log_path: Path, stream_path: Path, source_commit: str, main_revision: str
) -> dict[str, Any]:
    """Reconstruct the source of a baseline whose old header named only main."""
    log_text = log_path.read_text()
    lines = log_text.splitlines()
    if not lines[0].startswith(f"# revision={main_revision} ") or lines[-1] != "EXIT=0":
        raise RuntimeError("baseline log header or completion is unreadable")
    baseline = json.loads("\n".join(lines[1:-1]))
    if baseline["standard_names_total"] <= 0 or baseline["accepted"] <= 0:
        raise RuntimeError("baseline population control did not fire")

    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    relative_path = Path(__file__).resolve().relative_to(root)
    script_path = root / relative_path
    absent_at_main = subprocess.run(
        ["git", "cat-file", "-e", f"{main_revision}:{relative_path}"],
        capture_output=True,
    )
    if absent_at_main.returncode == 0:
        raise RuntimeError("the baseline header unexpectedly owns the script")
    blob = subprocess.run(
        ["git", "rev-parse", f"{source_commit}:{relative_path}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    events = [json.loads(line) for line in stream_path.read_text().splitlines()]
    completed = [
        (index, event["item"])
        for index, event in enumerate(events)
        if event.get("type") == "item.completed"
    ]
    observed = [
        (index, item)
        for index, item in completed
        if item.get("type") == "command_execution"
        and item.get("aggregated_output") == log_text
    ]
    if len(observed) != 1:
        raise RuntimeError("baseline bytes have no unique stream observation")
    observed_index, observed_item = observed[0]
    commands = [
        (index, item)
        for index, item in completed
        if index < observed_index
        and item.get("type") == "command_execution"
        and item.get("exit_code") == 0
        and str(log_path) in item.get("command", "")
        and f"python {relative_path} read-back" in item.get("command", "")
    ]
    if not commands:
        raise RuntimeError("baseline stream has no successful read-back command")
    command_index, command_item = commands[-1]
    edits = [
        (index, item)
        for index, item in completed
        if index < command_index
        and item.get("type") == "file_change"
        and any(
            change.get("path") == str(script_path) for change in item.get("changes", [])
        )
    ]
    if not edits:
        raise RuntimeError("baseline script update is absent from the stream")
    _, edit_item = edits[-1]
    commits = [
        (index, item)
        for index, item in completed
        if index > observed_index
        and item.get("type") == "command_execution"
        and item.get("exit_code") == 0
        and "git commit" in item.get("command", "")
        and source_commit[:9] in item.get("aggregated_output", "")
    ]
    if not commits:
        raise RuntimeError("the script commit is absent after baseline execution")
    commit_index, commit_item = commits[0]
    if any(
        command_index < index < commit_index
        and item.get("type") == "file_change"
        and any(
            change.get("path") == str(script_path) for change in item.get("changes", [])
        )
        for index, item in completed
    ):
        raise RuntimeError("script changed after baseline execution")
    formats = [
        item
        for index, item in completed
        if command_index < index < commit_index
        and item.get("type") == "command_execution"
        and "ruff format" in item.get("command", "")
        and str(relative_path) in item.get("command", "")
    ]
    if not formats or any(
        "left unchanged" not in item.get("aggregated_output", "") for item in formats
    ):
        raise RuntimeError("formatting did not preserve the baseline script")
    return {
        "baseline_log_sha256": hashlib.sha256(log_text.encode()).hexdigest(),
        "runner_commit": source_commit,
        "runner_blob": blob,
        "product_revision": main_revision,
        "stream_items": [
            edit_item["id"],
            command_item["id"],
            observed_item["id"],
            commit_item["id"],
        ],
        "standard_names_total": baseline["standard_names_total"],
        "accepted": baseline["accepted"],
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["self-check"]:
        print(json.dumps(self_check(), indent=2, sort_keys=True))
    elif sys.argv[1:] == ["activity"]:
        print(json.dumps(asdict(live_graph_activity()), indent=2, sort_keys=True))
    elif len(sys.argv) == 3 and sys.argv[1] == "read-back":
        print(json.dumps(read_back(Path(sys.argv[2])), indent=2, sort_keys=True))
    elif len(sys.argv) == 6 and sys.argv[1] == "verify-baseline":
        print(
            json.dumps(
                verify_baseline(
                    Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        raise SystemExit(
            "usage: run.py self-check|activity|read-back STAGING_DIR|"
            "verify-baseline LOG STREAM SCRIPT_COMMIT MAIN_REVISION"
        )

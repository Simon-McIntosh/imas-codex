"""Stopping the SLURM-hosted graph never touches its store before it exits.

``scancel`` returns at once while Neo4j keeps flushing its shutdown
checkpoint. Lock cleanup replaces every ``neostore*`` inode with a ``dd`` copy,
so running it inside that window sends the remaining flushes to the replaced
inode and tears the store. Every call that could reach the real scheduler, a
compute node or a local process is patched here.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import click
import pytest

import iter_codex.cli.services as services
from iter_codex.graph.neo4j_ops import Neo4jOperation

RUNNING_JOB = {
    "job_name": "codex-neo4j",
    "job_id": "4242",
    "state": "RUNNING",
    "time": "1-00:00:00",
    "node": "gpu-node",
    "gres": "",
    "cpus": "4",
}


@pytest.fixture
def scheduler(monkeypatch):
    """A fake scheduler that lists the job for a given number of polls."""
    state = {"listed_polls": 2, "polls": 0, "commands": [], "squeue_errors": 0}

    def run_remote(cmd, timeout=30, check=False):
        state["commands"].append(cmd)
        if cmd.startswith("scancel"):
            return ""
        if "squeue" in cmd and '--format="%A"' in cmd:
            if state["squeue_errors"]:
                state["squeue_errors"] -= 1
                raise subprocess.CalledProcessError(1, cmd)
            state["polls"] += 1
            return "4242\n" if state["polls"] <= state["listed_polls"] else ""
        raise AssertionError(f"unexpected remote command: {cmd}")

    monkeypatch.setattr(services, "_run_remote", run_remote)
    monkeypatch.setattr(
        services, "_get_service_job", lambda name, strict=False: dict(RUNNING_JOB)
    )
    monkeypatch.setattr(services, "_term_neo4j_on_node", lambda node, **_kw: None)
    # No process holds the store once the job has gone.
    monkeypatch.setattr(services, "_run_on_node", lambda node, cmd, timeout=30: "")
    monkeypatch.setattr(services.time, "sleep", lambda _s: None)
    return state


def test_stop_returns_only_after_the_job_leaves_the_scheduler(scheduler):
    node = services.stop_neo4j_job(timeout=60)

    assert node == "gpu-node"
    assert "scancel 4242" in scheduler["commands"]
    # listed twice after scancel, gone on the third poll
    assert scheduler["polls"] == 3


def test_stop_raises_while_the_job_is_still_listed(scheduler, monkeypatch):
    scheduler["listed_polls"] = 10**6
    clock = iter(float(t) for t in range(74, 1_000_074, 7))
    monkeypatch.setattr(services.time, "monotonic", lambda: next(clock))
    node_commands = []
    monkeypatch.setattr(
        services,
        "_run_on_node",
        lambda node, cmd, timeout=30: node_commands.append(cmd) or "",
    )

    with pytest.raises(click.ClickException, match="still listed"):
        services.stop_neo4j_job(timeout=180)

    # The wait ended in a raise, so nothing reached the store files or locks.
    assert node_commands == []
    assert not any(
        token in cmd
        for cmd in scheduler["commands"]
        for token in ("rm ", "dd ", "kill")
    )


def test_store_release_raises_while_a_process_still_holds_it(monkeypatch):
    holder = {"pid": "31337", "age": "42", "cmdline": "java Neo4jCommunity"}
    monkeypatch.setattr(
        services, "_select_processes", lambda node, **kwargs: ([holder], [])
    )
    clock = iter(float(t) for t in range(74, 1_000_074, 7))
    monkeypatch.setattr(services.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(services.time, "sleep", lambda _s: None)

    with pytest.raises(click.ClickException, match="still held"):
        services._wait_for_store_release("gpu-node", timeout=180)


def test_an_unanswered_squeue_counts_as_still_listed(scheduler):
    scheduler["listed_polls"] = 0
    scheduler["squeue_errors"] = 3

    assert services.stop_neo4j_job(timeout=60) == "gpu-node"
    # three failed polls did not end the wait; the first answer did
    assert scheduler["polls"] == 1


def test_a_strict_job_lookup_raises_when_squeue_fails(monkeypatch):
    def failing(cmd, timeout=30, check=False):
        if check:
            raise subprocess.CalledProcessError(1, cmd)
        return "slurm_load_jobs error"

    monkeypatch.setattr(services, "_run_remote", failing)

    assert services._get_service_job("codex-neo4j") is None
    with pytest.raises(subprocess.CalledProcessError):
        services._get_service_job("codex-neo4j", strict=True)


def test_export_stop_cleans_locks_only_after_the_job_has_exited(monkeypatch):
    events = []
    monkeypatch.setattr(services, "_is_graph_compute_target", lambda: True)
    monkeypatch.setattr(
        services, "stop_neo4j_job", lambda: events.append("exited") or "gpu-node"
    )
    monkeypatch.setattr(
        services, "_clean_neo4j_locks", lambda node: events.append(f"clean {node}")
    )
    monkeypatch.setattr(
        services,
        "stop_local_store_holders",
        lambda: events.append("local holders signalled"),
    )
    monkeypatch.setattr(
        "iter_codex.graph.neo4j_ops.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    Neo4jOperation._stop_neo4j(SimpleNamespace(profile=SimpleNamespace(name="codex")))

    assert events == ["exited", "clean gpu-node", "local holders signalled"]


def test_a_failed_stop_stops_the_operation_before_any_cleanup(monkeypatch):
    cleaned = []

    def stop_fails():
        raise click.ClickException("Neo4j job 4242 is still listed")

    monkeypatch.setattr(services, "_is_graph_compute_target", lambda: True)
    monkeypatch.setattr(services, "stop_neo4j_job", stop_fails)
    monkeypatch.setattr(services, "_clean_neo4j_locks", cleaned.append)
    monkeypatch.setattr(
        "iter_codex.graph.neo4j_ops.subprocess.run",
        lambda *args, **kwargs: pytest.fail("ran a local command after a failed stop"),
    )

    with pytest.raises(click.ClickException, match="still listed"):
        Neo4jOperation._stop_neo4j(
            SimpleNamespace(profile=SimpleNamespace(name="codex"))
        )
    assert cleaned == []


def test_lock_cleanup_refuses_while_a_process_holds_the_store(monkeypatch):
    scripts = []

    def run_on_node(node, cmd, timeout=30):
        scripts.append(cmd)
        if "SERVICE_STORE" in cmd:
            return "HOLD|31337|42|java org.neo4j.server.Neo4jCommunity\n"
        return "locks_cleaned\n"

    monkeypatch.setattr(services, "_run_on_node", run_on_node)

    with pytest.raises(click.ClickException, match="still running on gpu-node"):
        services._clean_neo4j_locks("gpu-node")

    assert not any("pkill" in cmd for cmd in scripts)


def test_the_store_holder_scan_never_selects_by_name_pattern(monkeypatch):
    scripts = []
    monkeypatch.setattr(
        services,
        "_run_on_node",
        lambda node, cmd, timeout=30: scripts.append(cmd) or "",
    )
    services._clean_neo4j_locks("gpu-node")

    scan = scripts[0]
    assert "pgrep" not in scan
    assert "pkill" not in scan
    assert "SERVICE_STORE" in scan

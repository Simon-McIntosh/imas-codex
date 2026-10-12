"""Stopping the SLURM-hosted graph never touches its store before it exits.

``scancel`` returns at once while Neo4j keeps flushing its shutdown
checkpoint. Lock cleanup replaces every ``neostore*`` inode with a ``dd`` copy,
so running it inside that window sends the remaining flushes to the replaced
inode and tears the store. Every call that could reach the real scheduler, a
compute node or a local process is patched here.
"""

from __future__ import annotations

import re
import subprocess
from types import SimpleNamespace

import click
import pytest

import imas_codex.cli.services as services
from imas_codex.graph.neo4j_ops import Neo4jOperation

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
    monkeypatch.setattr(services, "_term_neo4j_on_node", lambda node: None)
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
    clock = iter(float(t) for t in range(0, 10**6, 50))
    monkeypatch.setattr(services.time, "monotonic", lambda: next(clock))

    with pytest.raises(click.ClickException, match="still listed"):
        services.stop_neo4j_job(timeout=180)


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
        "imas_codex.graph.neo4j_ops.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    Neo4jOperation._stop_neo4j(SimpleNamespace(profile=SimpleNamespace(name="codex")))

    assert events == ["exited", "clean gpu-node"]


def test_a_failed_stop_stops_the_operation_before_any_cleanup(monkeypatch):
    cleaned = []

    def stop_fails():
        raise click.ClickException("Neo4j job 4242 is still listed")

    monkeypatch.setattr(services, "_is_graph_compute_target", lambda: True)
    monkeypatch.setattr(services, "stop_neo4j_job", stop_fails)
    monkeypatch.setattr(services, "_clean_neo4j_locks", cleaned.append)
    monkeypatch.setattr(
        "imas_codex.graph.neo4j_ops.subprocess.run",
        lambda *args, **kwargs: pytest.fail("ran a local command after a failed stop"),
    )

    with pytest.raises(click.ClickException, match="still listed"):
        Neo4jOperation._stop_neo4j(
            SimpleNamespace(profile=SimpleNamespace(name="codex"))
        )
    assert cleaned == []


def test_lock_cleanup_refuses_while_a_neo4j_jvm_is_alive(monkeypatch):
    scripts = []

    def run_on_node(node, cmd, timeout=30):
        scripts.append(cmd)
        return "neo4j_alive 31337\n"

    monkeypatch.setattr(services, "_run_on_node", run_on_node)

    with pytest.raises(click.ClickException, match="still running on gpu-node"):
        services._clean_neo4j_locks("gpu-node")

    script = scripts[0]
    guard = script.index("pgrep")
    assert guard < script.index("rm -f")
    assert guard < script.index("dd if=")


def test_the_store_holder_pattern_matches_the_jvms_and_not_its_own_script(
    monkeypatch,
):
    pattern = re.compile(services._NEO4J_STORE_HOLDERS)
    java = "/opt/java/openjdk/bin/java -cp /var/lib/neo4j/lib/*"
    assert pattern.search(f"{java} org.neo4j.server.Neo4jCommunity --home-dir=/x")
    assert pattern.search(f"{java} org.neo4j.cli.AdminTool database dump neo4j")

    # the remote shell's argv carries the cleanup script verbatim, so a pattern
    # that matched its own text would find the shell and never report clean
    scripts = []
    monkeypatch.setattr(
        services,
        "_run_on_node",
        lambda node, cmd, timeout=30: scripts.append(cmd) or "locks_cleaned\n",
    )
    services._clean_neo4j_locks("gpu-node")
    assert not pattern.search(scripts[0])

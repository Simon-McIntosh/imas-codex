"""Stopping a service signals only the pids attributable to that service.

The selector is the graph store path or the service's own scheduler job id.
A fake /proc tree is walked by the real scan script, so these tests exercise
the selection rather than a description of it: a second Neo4j on another
store and a shell whose command line names the service are both left alone.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click
import pytest

import iter_codex.cli.services as services


def _add_process(
    proc,
    pid: int,
    cmdline: str,
    *,
    cwd=None,
    open_path=None,
    cgroup: str | None = None,
) -> None:
    entry = proc / str(pid)
    entry.mkdir()
    (entry / "cmdline").write_bytes(cmdline.encode() + b"\0")
    if cwd is not None:
        (entry / "cwd").symlink_to(cwd)
    if open_path is not None:
        (entry / "fd").mkdir()
        (entry / "fd" / "3").symlink_to(open_path)
    if cgroup is not None:
        (entry / "cgroup").write_text(cgroup + "\n")


@pytest.fixture
def proc_table(tmp_path):
    """A fake process table with one service Neo4j, one other, one shell."""
    proc = tmp_path / "proc"
    proc.mkdir()
    store = tmp_path / "store"
    (store / "data" / "databases").mkdir(parents=True)
    other = tmp_path / "other-store"
    (other / "data" / "databases").mkdir(parents=True)

    _add_process(
        proc,
        1001,
        "java org.neo4j.server.Neo4jCommunity --home-dir=/x",
        cwd=store,
        open_path=store / "data" / "databases" / "neostore",
    )
    # Another project's Neo4j: same program, different store.
    _add_process(
        proc,
        2002,
        "java org.neo4j.server.Neo4jCommunity --home-dir=/y",
        cwd=other,
        open_path=other / "data" / "databases" / "neostore",
    )
    # A shell whose own command line names the service.
    _add_process(proc, 3003, "bash -c echo Neo4jCommunity", cwd=tmp_path)
    return {"proc": proc, "store": store, "other": other}


def _scan(proc_table, **kwargs):
    cmd = services._process_scan_command(**kwargs)
    env = dict(os.environ)
    env["PROC_ROOT"] = str(proc_table["proc"])
    completed = subprocess.run(
        ["bash", "-c", cmd], env=env, capture_output=True, text=True, check=True
    )
    return services._parse_process_lines(completed.stdout)


def test_scan_selects_only_processes_holding_this_store(proc_table):
    holders, left_alone = _scan(proc_table, store=str(proc_table["store"]))

    assert [p["pid"] for p in holders] == ["1001"]
    assert "1001" not in [p["pid"] for p in left_alone]


def test_another_store_is_left_alone_with_its_reason(proc_table):
    holders, left_alone = _scan(proc_table, store=str(proc_table["store"]))
    pip = {p["pid"]: p for p in left_alone}

    assert "2002" in pip
    assert "does not hold" in pip["2002"]["reason"]


def test_a_shell_naming_the_service_is_not_a_holder(proc_table):
    holders, left_alone = _scan(proc_table, store=str(proc_table["store"]))

    assert "3003" not in [p["pid"] for p in holders]
    assert "3003" in [p["pid"] for p in left_alone]


def test_job_cgroup_attributes_a_process_without_a_store_match(proc_table):
    _add_process(
        proc_table["proc"],
        4004,
        "uvicorn embed",
        cgroup="/slurm/uid_1000/job_4242/step_0",
    )

    holders, _left = _scan(proc_table, store="", job_id="4242", word="embed")

    assert [p["pid"] for p in holders] == ["4004"]


@pytest.fixture
def stop_recording(monkeypatch, proc_table):
    """Route the real scan at the fake table and record every signal."""
    calls: list[str] = []
    alive: dict[str, set[str]] = {"pids": set()}
    killed = {"done": False}
    clock = {"t": 0.0}

    def monotonic():
        clock["t"] += 20.0
        return clock["t"]

    def run_shell(node, cmd, timeout=30):
        if "SERVICE_STORE" in cmd:
            # Run the real scan against the fake table.
            scan = services._process_scan_command(store=str(proc_table["store"]))
            env = dict(os.environ)
            env["PROC_ROOT"] = str(proc_table["proc"])
            return subprocess.run(
                ["bash", "-c", scan], env=env, capture_output=True, text=True
            ).stdout
        calls.append(cmd)
        if "kill -KILL" in cmd:
            killed["done"] = True
            return ""
        if "kill -0" in cmd:
            if killed["done"]:
                return ""
            return "\n".join(sorted(alive["pids"]))
        return ""

    monkeypatch.setattr(services, "_run_shell", run_shell)
    monkeypatch.setattr(services.time, "sleep", lambda _s: None)
    monkeypatch.setattr(services.time, "monotonic", monotonic)
    return {"calls": calls, "alive": alive}


def test_stop_signals_exactly_the_listed_pid(stop_recording):
    services._kill_neo4j_on_node("gpu-node")

    joined = "\n".join(stop_recording["calls"])
    assert "kill -TERM 1001" in joined or "kill -TERM 1001" in joined.replace("if ", "")
    assert "2002" not in joined
    assert "3003" not in joined


def test_escalation_reaches_only_a_listed_survivor(stop_recording):
    stop_recording["alive"]["pids"] = {"1001"}

    services._kill_neo4j_on_node("gpu-node")

    joined = "\n".join(stop_recording["calls"])
    assert "kill -KILL 1001" in joined
    assert "kill -KILL 2002" not in joined


def test_embed_stop_signals_only_the_embed_job_cgroup(monkeypatch, proc_table):
    """Embed has no store path, so only its own scheduler job is attributable."""
    _add_process(
        proc_table["proc"],
        4004,
        "uvicorn embed.server:app",
        cgroup="/slurm/uid_1000/job_4242/step_0",
    )
    # Same words, no job attribution: indistinguishable from another service.
    _add_process(proc_table["proc"], 5005, "uvicorn embed.server:app")

    scripts: list[str] = []

    def run_shell(node, cmd, timeout=30):
        if "SERVICE_STORE" in cmd:
            scan = services._process_scan_command(store="", job_id="4242", word="embed")
            env = dict(os.environ)
            env["PROC_ROOT"] = str(proc_table["proc"])
            return subprocess.run(
                ["bash", "-c", scan], env=env, capture_output=True, text=True
            ).stdout
        scripts.append(cmd)
        return ""

    monkeypatch.setattr(services, "_run_shell", run_shell)
    monkeypatch.setattr(
        services,
        "_get_service_job",
        lambda name, strict=False: {"job_id": "4242", "node": "gpu-node"},
    )

    services._kill_embed_orphans("gpu-node")

    joined = "\n".join(scripts)
    assert "kill -TERM 4004" in joined
    assert "4004" in joined
    assert "5005" not in joined


def test_embed_stop_reports_nothing_when_nothing_is_attributable(monkeypatch):
    signalled: list[str] = []
    monkeypatch.setattr(services, "_get_service_job", lambda name, strict=False: None)
    monkeypatch.setattr(
        services, "_select_recorded_pid", lambda node, path, *, word: ([], [])
    )
    monkeypatch.setattr(
        services, "_terminate_listed_pids", lambda *a, **kw: signalled.append(str(a))
    )

    services._kill_embed_orphans("gpu-node")

    assert signalled == []


def test_embed_stop_signals_the_recorded_pid_when_no_job_is_listed(monkeypatch):
    signalled: list[str] = []
    monkeypatch.setattr(services, "_get_service_job", lambda name, strict=False: None)
    monkeypatch.setattr(
        services,
        "_select_recorded_pid",
        lambda node, path, *, word: (
            [{"pid": "7777", "age": "12", "cmdline": "uvicorn iter-codex embed"}],
            [],
        ),
    )
    monkeypatch.setattr(
        services, "_terminate_listed_pids", lambda *a, **kw: signalled.append(str(a))
    )

    services._kill_embed_orphans("gpu-node")

    assert signalled and "7777" in signalled[0]


def test_a_recorded_pid_is_selected_against_a_fake_proc_table(tmp_path, proc_table):
    pid_file = tmp_path / "codex-embed-host.pid"
    pid_file.write_text("7777\n")
    _add_process(proc_table["proc"], 7777, "uvicorn iter_codex embed", cwd=tmp_path)

    cmd = services._recorded_pid_command(str(pid_file), word="embed")
    env = dict(os.environ)
    env["PROC_ROOT"] = str(proc_table["proc"])
    out = subprocess.run(
        ["bash", "-c", cmd], env=env, capture_output=True, text=True, check=True
    ).stdout

    holders, _left = services._parse_process_lines(out)
    assert [p["pid"] for p in holders] == ["7777"]


def test_a_recycled_recorded_pid_is_left_alone(tmp_path, proc_table):
    pid_file = tmp_path / "codex-embed-host.pid"
    pid_file.write_text("8888\n")
    _add_process(proc_table["proc"], 8888, "sleep 3000", cwd=tmp_path)

    cmd = services._recorded_pid_command(str(pid_file), word="embed")
    env = dict(os.environ)
    env["PROC_ROOT"] = str(proc_table["proc"])
    out = subprocess.run(
        ["bash", "-c", cmd], env=env, capture_output=True, text=True, check=True
    ).stdout

    holders, left_alone = services._parse_process_lines(out)
    assert holders == []
    assert [p["pid"] for p in left_alone] == ["8888"]


def test_the_default_store_path_reaches_the_scan_expanded():
    cmd = services._process_scan_command(store=services._NEO4J_STORE_BASE)

    assert "$HOME" not in cmd
    assert os.path.expanduser("~/.local/share/iter-codex/neo4j") in cmd


def test_a_process_under_the_default_store_is_selected(proc_table):
    """The production default must match a real cwd, not a string ending in $HOME."""
    default = Path(os.path.expanduser("~/.local/share/iter-codex/neo4j"))
    _add_process(proc_table["proc"], 6006, "java Neo4jCommunity", cwd=default / "data")

    holders, _left = _scan(proc_table, store=services._NEO4J_STORE_BASE)

    assert "6006" in [p["pid"] for p in holders]


def test_login_embed_stop_targets_the_pid_its_launcher_recorded(monkeypatch):
    scripts: list[str] = []

    def run_remote(cmd, timeout=30, check=False):
        if cmd.startswith("cat "):
            return "4321\n"
        scripts.append(cmd)
        return "PID 4321 (7s) surveyed\nSIGTERM -> 4321\nPID 4321 is not running\n"

    monkeypatch.setattr(services, "_run_remote", run_remote)

    services._kill_login_embed()

    signal_script = "\n".join(scripts)
    assert "kill -TERM" in signal_script
    assert "4321" in signal_script
    assert "pgrep" not in signal_script
    assert "pkill" not in signal_script

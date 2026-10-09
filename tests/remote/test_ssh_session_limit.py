"""Process-local SSH session admission for facility hosts."""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import pytest

from imas_codex.remote import executor, ssh_worker, tools
from imas_codex.remote.ssh_worker import (
    SSHWorker,
    SSHWorkerPool,
    acquire_host_session,
    configure_host_session_limit,
    host_session_limit,
)


def test_limited_host_serializes_calls_and_unlimited_host_overlaps(monkeypatch):
    monkeypatch.setattr(executor, "is_local_host", lambda host: False)
    monkeypatch.setattr(executor, "_ensure_ssh_healthy_once", lambda host: None)
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda facility: (
            {"ssh_host": facility, "ssh_max_sessions": 1}
            if facility == "limited"
            else {"ssh_host": facility}
        ),
    )

    lock = Lock()
    active = 0
    peak = 0

    def fake_run(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        sleep(0.05)
        with lock:
            active -= 1
        return type("Result", (), {"stdout": "ok", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    for host, expected_peak in (("limited", 1), ("unlimited", 2)):
        resolved = tools._resolve_ssh_host(host)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(
                pool.map(
                    lambda _, host=resolved: executor.run_command("true", host),
                    range(2),
                )
            )
        assert outputs == ["ok", "ok"]
        assert peak == expected_peak
        peak = 0


@pytest.mark.asyncio
async def test_async_and_threaded_calls_share_the_host_limit(monkeypatch):
    host = "mixed-host"
    configure_host_session_limit(host, 1)
    monkeypatch.setattr(executor, "is_local_host", lambda host: False)
    monkeypatch.setattr(executor, "_ensure_ssh_healthy_once", lambda host: None)
    lock = Lock()
    active = 0
    peak = 0

    def enter():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)

    def leave():
        nonlocal active
        with lock:
            active -= 1

    def fake_run(*args, **kwargs):
        enter()
        sleep(0.05)
        leave()
        return type("Result", (), {"stdout": "ok", "stderr": "", "returncode": 0})()

    class FakeProcess:
        returncode = 0

        async def communicate(self, data):
            enter()
            await asyncio.sleep(0.05)
            leave()
            return b"ok", b""

    async def fake_subprocess(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", fake_subprocess)
    for target, limited in ((host, True), ("mixed-unlimited", False)):
        await asyncio.gather(
            executor.async_run_python_script("scan_directories.py", ssh_host=target),
            asyncio.to_thread(
                executor.run_python_script, "scan_directories.py", ssh_host=target
            ),
            asyncio.to_thread(
                executor.run_script_via_stdin, "echo ok", ssh_host=target
            ),
            asyncio.to_thread(executor.run_command, "true", ssh_host=target),
        )
        if limited:
            assert peak == 1
        else:
            assert peak >= 2
        peak = 0


@pytest.mark.asyncio
async def test_open_pool_owns_one_session_and_handles_per_call_scripts(monkeypatch):
    host = "pooled-host"
    configure_host_session_limit(host, 1)
    monkeypatch.setattr(executor, "is_local_host", lambda host: False)
    monkeypatch.setattr(executor, "_ensure_ssh_healthy_once", lambda host: None)

    async def fake_start(self, timeout=30.0):
        self._session_permit = await acquire_host_session(self.ssh_host)
        self._ready = True

    async def fake_close(self):
        self._ready = False
        self._release_session()

    async def fake_execute(self, script, stdin_data="{}", timeout=60.0):
        return json.dumps({"returncode": 0, "stdout": "pooled", "stderr": ""})

    monkeypatch.setattr(SSHWorker, "start", fake_start)
    monkeypatch.setattr(SSHWorker, "close", fake_close)
    monkeypatch.setattr(SSHWorker, "alive", property(lambda self: self._ready))
    monkeypatch.setattr(SSHWorker, "execute", fake_execute)
    monkeypatch.setattr(
        executor.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("opened a second SSH session"),
    )

    pool = SSHWorkerPool(host, max_workers=4)
    assert pool.max_workers == 1
    await pool.start()
    try:
        assert len(pool._workers) == 1
        assert (
            await executor.async_run_python_script("scan_directories.py", ssh_host=host)
            == "pooled"
        )
        assert (
            await asyncio.to_thread(
                executor.run_python_script, "scan_directories.py", ssh_host=host
            )
            == "pooled"
        )
        assert (
            await asyncio.to_thread(executor.run_command, "true", ssh_host=host)
            == "pooled"
        )
        assert (
            await asyncio.to_thread(
                executor.run_script_via_stdin, "echo ok", ssh_host=host
            )
            == "pooled"
        )
    finally:
        await pool.close()


def test_facility_declares_one_session():
    from imas_codex.discovery.base.facility import (
        get_facility,
        validate_facility_config,
    )

    assert get_facility("jt-60sa")["ssh_max_sessions"] == 1
    assert validate_facility_config("jt-60sa") == []
    assert tools._resolve_ssh_host("jt-60sa") == "jt-60sa"
    assert host_session_limit("jt-60sa") == 1


@pytest.mark.asyncio
async def test_pool_bridge_executes_a_command_over_its_worker(monkeypatch):
    host = "bridge-host"
    configure_host_session_limit(host, 1)
    monkeypatch.setattr(executor, "is_local_host", lambda host: False)
    monkeypatch.setattr(executor, "_ensure_ssh_healthy_once", lambda host: None)
    original_create = asyncio.create_subprocess_exec
    launches = []

    async def local_worker(*args, **kwargs):
        launches.append(args)
        return await original_create(
            sys.executable, "-u", "-c", ssh_worker._WORKER_SCRIPT, **kwargs
        )

    monkeypatch.setattr(ssh_worker.asyncio, "create_subprocess_exec", local_worker)
    monkeypatch.setattr(
        executor.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("opened a second SSH session"),
    )
    pool = SSHWorkerPool(host, max_workers=3)
    await pool.start()
    try:
        assert pool.max_workers == 1
        assert (
            await asyncio.to_thread(executor.run_command, "printf bridge-ok", host)
            == "bridge-ok"
        )
        assert (
            await asyncio.to_thread(executor.run_command, "cat", host, 2)
            == "(no output)"
        )
        assert (
            await asyncio.to_thread(
                executor.run_script_via_stdin, "printf stdin-ok", host
            )
            == "stdin-ok"
        )
        assert len(launches) == 1
    finally:
        await pool.close()

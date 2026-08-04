"""Unit tests for imas_codex.graph.temp_neo4j module.

Tests for temporary Neo4j instance management extracted from graph_cli.py.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock, call

import click
import pytest

from imas_codex.graph import temp_neo4j as sut

# ============================================================================
# IMAS_DD_LABELS
# ============================================================================


class TestImasDdLabels:
    """Tests for the IMAS_DD_LABELS constant."""

    def test_not_empty(self):
        assert len(sut.IMAS_DD_LABELS) > 0

    def test_contains_core_labels(self):
        assert "DDVersion" in sut.IMAS_DD_LABELS
        assert "IDS" in sut.IMAS_DD_LABELS
        assert "IMASNode" in sut.IMAS_DD_LABELS


# ============================================================================
# write_temp_neo4j_conf
# ============================================================================


class TestWriteTempNeo4jConf:
    """Tests for write_temp_neo4j_conf()."""

    def test_creates_conf_file(self, tmp_path):
        conf = sut.write_temp_neo4j_conf(tmp_path, 7688, 7475)
        assert conf.exists()
        content = conf.read_text()
        assert "7688" in content
        assert "7475" in content

    def test_disables_auth(self, tmp_path):
        conf = sut.write_temp_neo4j_conf(tmp_path, 7688, 7475)
        content = conf.read_text()
        assert "auth_enabled=false" in content

    def test_sets_compute_safe_default_memory_limits(self, tmp_path, monkeypatch):
        monkeypatch.delenv(sut._TEMP_NEO4J_METASPACE_ENV, raising=False)
        conf = sut.write_temp_neo4j_conf(tmp_path, 7688, 7475)
        content = conf.read_text()
        assert "heap" in content
        assert "pagecache" in content
        assert "server.jvm.additional=-XX:MaxDirectMemorySize=256m" in content
        assert "server.jvm.additional=-XX:MaxMetaspaceSize=768m" in content
        assert "MaxMetaspaceSize=128m" not in content

    def test_environment_and_explicit_metaspace_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv(sut._TEMP_NEO4J_METASPACE_ENV, "1G")
        environment_conf = sut.write_temp_neo4j_conf(tmp_path, 7688, 7475)
        assert "MaxMetaspaceSize=1g" in environment_conf.read_text()

        explicit_conf = sut.write_temp_neo4j_conf(
            tmp_path, 7688, 7475, metaspace_limit="512m"
        )
        assert "MaxMetaspaceSize=512m" in explicit_conf.read_text()

    @pytest.mark.parametrize("value", ["256m", "512M", "1g", "2G"])
    def test_metaspace_boundaries_are_accepted(self, value):
        options = sut._temp_neo4j_jvm_options(value)
        assert options[-1] == f"-XX:MaxMetaspaceSize={value.lower()}"

    @pytest.mark.parametrize("value", ["", "0m", "255m", "2049m", "3g", "768", "many"])
    def test_invalid_or_unsafe_metaspace_overrides_are_rejected(self, value):
        with pytest.raises(click.ClickException, match="TEMP_NEO4J_MAX_METASPACE"):
            sut._temp_neo4j_jvm_options(value)


class TestBoltReadiness:
    """Tests for process-aware temporary Neo4j readiness."""

    def test_start_surfaces_child_exit_without_launching_neo4j(
        self, tmp_path, monkeypatch
    ):
        for name in ("data", "dumps", "logs"):
            (tmp_path / name).mkdir()
        monkeypatch.setattr(sut, "_neo4j_image", lambda: Path("neo4j.sif"))
        monkeypatch.setattr(
            sut.subprocess,
            "run",
            Mock(return_value=Mock(returncode=0, stderr="")),
        )
        proc = Mock()
        proc.pid = 4321
        proc.poll.return_value = 31
        popen = Mock(return_value=proc)
        monkeypatch.setattr(sut.subprocess, "Popen", popen)

        with pytest.raises(click.ClickException, match="exit code 31"):
            sut.start_temp_neo4j(tmp_path, 27687, 27474)

        popen.assert_called_once()
        proc.wait.assert_not_called()

    def test_child_death_reports_exit_code_and_recent_log(self, tmp_path):
        log_path = tmp_path / "neo4j.log"
        log_path.write_text("earlier\nclass metadata exhausted\n")
        proc = Mock()
        proc.poll.return_value = 17

        with pytest.raises(click.ClickException) as caught:
            sut._wait_for_bolt_ready(proc, 27687, log_path)

        message = str(caught.value)
        assert "exit code 17" in message
        assert "class metadata exhausted" in message

    def test_child_death_after_failed_probe_does_not_sleep(self, tmp_path, monkeypatch):
        log_path = tmp_path / "neo4j.log"
        log_path.write_text("fatal startup error\n")
        proc = Mock()
        proc.poll.side_effect = [None, 23]
        monkeypatch.setattr(sut, "_bolt_port_ready", lambda _port: False)
        sleep = Mock()
        monkeypatch.setattr(sut.time, "sleep", sleep)

        with pytest.raises(click.ClickException, match="exit code 23"):
            sut._wait_for_bolt_ready(proc, 27687, log_path)

        sleep.assert_not_called()

    def test_returns_as_soon_as_bolt_accepts_connections(self, tmp_path, monkeypatch):
        log_path = tmp_path / "neo4j.log"
        proc = Mock()
        proc.poll.return_value = None
        readiness = iter([False, True])
        monkeypatch.setattr(sut, "_bolt_port_ready", lambda _port: next(readiness))
        sleep = Mock()
        monkeypatch.setattr(sut.time, "sleep", sleep)
        monotonic = iter([10.0, 10.25])
        monkeypatch.setattr(sut.time, "monotonic", lambda: next(monotonic))

        sut._wait_for_bolt_ready(
            proc, 27687, log_path, timeout_seconds=5, poll_seconds=0.5
        )

        sleep.assert_called_once_with(0.5)

    def test_timeout_reports_recent_log_context(self, tmp_path, monkeypatch):
        log_path = tmp_path / "neo4j.log"
        log_path.write_text("still initializing\n")
        proc = Mock()
        proc.poll.return_value = None
        monkeypatch.setattr(sut, "_bolt_port_ready", lambda _port: False)
        monotonic = iter([5.0, 6.0])
        monkeypatch.setattr(sut.time, "monotonic", lambda: next(monotonic))

        with pytest.raises(click.ClickException) as caught:
            sut._wait_for_bolt_ready(
                proc, 27687, log_path, timeout_seconds=1, poll_seconds=0.5
            )

        message = str(caught.value)
        assert "Bolt readiness within 1 seconds" in message
        assert "still initializing" in message


class TestStopTempNeo4j:
    """Tests for bounded temporary Neo4j teardown."""

    def test_already_failed_child_returns_without_waiting(self):
        proc = Mock()
        proc.poll.return_value = 9
        proc.wait.side_effect = AssertionError("wait must not be called")

        sut.stop_temp_neo4j(proc)

        proc.wait.assert_not_called()

    def test_live_child_escalates_with_bounded_waits(self, monkeypatch):
        proc = Mock()
        proc.pid = 4321
        proc.poll.return_value = None
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="neo4j", timeout=15),
            0,
        ]
        monkeypatch.setattr(sut.os, "getpgid", lambda _pid: 7654)
        killpg = Mock()
        monkeypatch.setattr(sut.os, "killpg", killpg)

        sut.stop_temp_neo4j(proc)

        assert killpg.call_args_list == [
            call(7654, signal.SIGTERM),
            call(7654, signal.SIGKILL),
        ]
        assert proc.wait.call_args_list == [call(timeout=15), call(timeout=10)]

    def test_unresponsive_child_raises_after_final_bounded_wait(self, monkeypatch):
        proc = Mock()
        proc.pid = 4321
        proc.poll.return_value = None
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="neo4j", timeout=15),
            subprocess.TimeoutExpired(cmd="neo4j", timeout=10),
        ]
        monkeypatch.setattr(sut.os, "getpgid", lambda _pid: 7654)
        monkeypatch.setattr(sut.os, "killpg", Mock())

        with pytest.raises(click.ClickException, match="did not exit after SIGKILL"):
            sut.stop_temp_neo4j(proc)

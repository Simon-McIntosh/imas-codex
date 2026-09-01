"""Unit tests for imas_codex.graph.neo4j_ops module.

Tests for Neo4j operation infrastructure extracted from graph_cli.py.
All subprocess calls and filesystem access are mocked.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from imas_codex.graph.neo4j_ops import (
    RECOVERY_DIR,
    backup_graph_dump,
    check_stale_neo4j_process,
    parse_dump_error,
    write_data_presence_marker,
)

# ============================================================================
# check_stale_neo4j_process
# ============================================================================


class TestCheckStaleNeo4j:
    """Tests for check_stale_neo4j_process()."""

    def test_no_pid_file(self, tmp_path):
        """No PID file returns (False, None)."""
        is_stale, info = check_stale_neo4j_process(tmp_path)
        assert is_stale is False
        assert info is None

    def test_stale_pid(self, tmp_path, monkeypatch):
        """PID file for dead process is cleaned up and returns (False, None)."""
        pid_file = tmp_path / "neo4j.pid"
        pid_file.write_text("999999")

        # Mock os.kill to raise ProcessLookupError (process doesn't exist)
        def mock_kill(pid, sig):
            raise ProcessLookupError()

        monkeypatch.setattr("os.kill", mock_kill)
        is_stale, info = check_stale_neo4j_process(tmp_path)
        assert is_stale is False
        assert info is None
        # PID file should have been cleaned up
        assert not pid_file.exists()

    def test_process_owned_by_another_user(self, tmp_path, monkeypatch):
        """Process owned by another user is flagged as stale."""
        pid_file = tmp_path / "neo4j.pid"
        pid_file.write_text("999999")

        def mock_kill(pid, sig):
            raise PermissionError()

        monkeypatch.setattr("os.kill", mock_kill)
        is_stale, info = check_stale_neo4j_process(tmp_path)
        assert is_stale is True
        assert "999999" in info


# ============================================================================
# write_data_presence_marker
# ============================================================================


class TestWriteDataPresenceMarker:
    """Tests for write_data_presence_marker()."""

    def test_creates_recovery_dir(self, tmp_path, monkeypatch):
        """Marker writing creates a timestamped recovery directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "databases").mkdir()
        (data_dir / "databases" / "store.db").write_text("test")

        # Point RECOVERY_DIR to temp
        monkeypatch.setattr(
            "imas_codex.graph.neo4j_ops.RECOVERY_DIR",
            tmp_path / "recovery",
        )

        result = write_data_presence_marker("test-backup", data_dir=data_dir)
        assert result is not None
        assert result.exists()


# ============================================================================
# backup_graph_dump
# ============================================================================


class TestBackupGraphDump:
    """Tests for backup_graph_dump()."""

    def test_creates_nonempty_recoverable_artifact(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "neo4j"
        profile = SimpleNamespace(name="test", data_dir=data_dir)
        output = tmp_path / "backups" / "test.dump"

        monkeypatch.setattr(
            "imas_codex.graph.profiles.resolve_neo4j",
            lambda: profile,
        )
        monkeypatch.setattr(
            "imas_codex.graph.profiles.BACKUPS_DIR",
            output.parent,
        )

        def create_dump(_profile, dumps_dir):
            (dumps_dir / "neo4j.dump").write_bytes(b"recoverable graph dump")

        monkeypatch.setattr(
            "imas_codex.graph.neo4j_ops.run_neo4j_dump",
            create_dump,
        )

        result = backup_graph_dump(output=output)

        assert result == output
        assert result.stat().st_size > 0
        assert result.read_bytes() == b"recoverable graph dump"

    def test_rejects_empty_dump_artifact(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "neo4j"
        profile = SimpleNamespace(name="test", data_dir=data_dir)
        output = tmp_path / "backups" / "test.dump"

        monkeypatch.setattr(
            "imas_codex.graph.profiles.resolve_neo4j",
            lambda: profile,
        )
        monkeypatch.setattr(
            "imas_codex.graph.profiles.BACKUPS_DIR",
            output.parent,
        )

        def create_empty_dump(_profile, dumps_dir):
            (dumps_dir / "neo4j.dump").touch()

        monkeypatch.setattr(
            "imas_codex.graph.neo4j_ops.run_neo4j_dump",
            create_empty_dump,
        )

        with pytest.raises(click.ClickException, match="is empty"):
            backup_graph_dump(output=output)

        assert not output.exists()


# ============================================================================
# parse_dump_error
# ============================================================================


class TestParseDumpError:
    """Tests for parse_dump_error()."""

    def test_lock_detected(self):
        msg, is_lock = parse_dump_error("Error: database is in use by another process")
        assert is_lock is True

    def test_filelockexception(self):
        msg, is_lock = parse_dump_error(
            "org.neo4j.kernel.FileLockException: lock on store"
        )
        assert is_lock is True

    def test_generic_error(self):
        msg, is_lock = parse_dump_error("Caused by: java.io.IOException: disk full")
        assert is_lock is False
        assert "Caused by" in msg

    def test_dump_failed_for_databases(self):
        msg, is_lock = parse_dump_error("Dump failed for databases: 'neo4j'")
        assert is_lock is True

    def test_unable_to_find_store_id(self):
        msg, is_lock = parse_dump_error("Unable to find store id")
        assert is_lock is False
        assert "store id" in msg

    def test_empty_stderr(self):
        msg, is_lock = parse_dump_error("")
        assert is_lock is False
        assert msg == "Unknown error"

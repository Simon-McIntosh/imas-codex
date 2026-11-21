"""Agreement between a bare graph export and the backup currency instrument.

``graph export`` writes an archive into the export directory when no explicit
output path is given, while the currency instrument used to read only the
backup directory — so an archive taken as briefed landed where the instrument
never looked and read as ``no_backup``. These tests pin the agreement: the
instrument scans both directories, and its verdict is the archive's existence
and verified contents rather than an age against live data.
"""

from __future__ import annotations

import io
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from iter_codex.graph import neo4j_ops


def _write_recovery_archive(path: Path, timestamp: float) -> Path:
    """Write a real gzip tar carrying a non-empty graph.dump member."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"recoverable graph dump"
    member = tarfile.TarInfo(f"{path.stem}/graph.dump")
    member.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(payload))
    os.utime(path, (timestamp, timestamp))
    return path


def _write_empty_dump_archive(path: Path, timestamp: float) -> Path:
    """Write a gzip tar whose graph.dump member is empty (not restorable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    member = tarfile.TarInfo("graph.dump")
    member.size = 0
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(b""))
    os.utime(path, (timestamp, timestamp))
    return path


def _write_file(path: Path, timestamp: float, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.utime(path, (timestamp, timestamp))
    return path


def _configure_paths(
    monkeypatch,
    backups_dir: Path,
    exports_dir: Path,
    data_dir: Path,
) -> None:
    monkeypatch.setattr("iter_codex.graph.profiles.BACKUPS_DIR", backups_dir)
    monkeypatch.setattr("iter_codex.graph.dirs.EXPORTS_DIR", exports_dir)
    monkeypatch.setattr(
        "iter_codex.graph.profiles.resolve_neo4j",
        lambda: SimpleNamespace(data_dir=data_dir),
    )


def test_bare_export_in_exports_dir_registers_as_backup(
    tmp_path: Path, monkeypatch
) -> None:
    """A bare export (no ``-o``) writes here; the instrument must find it."""
    backups_dir = tmp_path / "backups"
    exports_dir = tmp_path / "exports"
    data_dir = tmp_path / "neo4j"
    archive = _write_recovery_archive(
        exports_dir / "iter-codex-graph-abc1234.tar.gz", 100.0
    )
    _write_file(data_dir / "data" / "store", 200.0, b"live")
    _configure_paths(monkeypatch, backups_dir, exports_dir, data_dir)

    currency = neo4j_ops.get_backup_currency()

    assert currency.status == "current"
    assert currency.backup_path == archive
    assert currency.backup_size_bytes == archive.stat().st_size


def test_misfiled_archive_outside_scanned_dirs_reads_no_backup(
    tmp_path: Path, monkeypatch
) -> None:
    """An archive the instrument does not scan is indistinguishable from none."""
    stray_dir = tmp_path / "elsewhere"
    backups_dir = tmp_path / "backups"
    exports_dir = tmp_path / "exports"
    data_dir = tmp_path / "neo4j"
    _write_recovery_archive(stray_dir / "archive.tar.gz", 100.0)
    _write_file(data_dir / "data" / "store", 200.0, b"live")
    _configure_paths(monkeypatch, backups_dir, exports_dir, data_dir)

    currency = neo4j_ops.get_backup_currency()

    assert currency.status == "no_backup"
    assert currency.backup_path is None
    assert currency.backup_size_bytes is None
    assert currency.backup_modified_at is None
    assert currency.age_seconds is None


def test_fresh_archive_reads_current_despite_newer_live_writes(
    tmp_path: Path, monkeypatch
) -> None:
    """A fresh checkpoint previously read stale; existence is now the verdict."""
    backups_dir = tmp_path / "backups"
    exports_dir = tmp_path / "exports"
    data_dir = tmp_path / "neo4j"
    archive = _write_recovery_archive(backups_dir / "archive.tar.gz", 100.0)
    # The restart writes into the live tree after the archive is sealed, so
    # age_seconds 0 is unreachable for the export that brought the service
    # back — the measured fresh-checkpoint lag was ~23 s.
    _write_file(data_dir / "data" / "store", 100.0 + 22.956782, b"live")
    _configure_paths(monkeypatch, backups_dir, exports_dir, data_dir)

    currency = neo4j_ops.get_backup_currency()

    assert currency.status == "current"
    assert currency.backup_path == archive
    assert currency.age_seconds == pytest.approx(22.956782)


def test_archive_with_empty_dump_member_reads_no_backup(
    tmp_path: Path, monkeypatch
) -> None:
    """A gzip tar is not a recovery archive unless it carries real contents."""
    backups_dir = tmp_path / "backups"
    exports_dir = tmp_path / "exports"
    data_dir = tmp_path / "neo4j"
    _write_empty_dump_archive(exports_dir / "empty.tar.gz", 100.0)
    _write_file(data_dir / "data" / "store", 200.0, b"live")
    _configure_paths(monkeypatch, backups_dir, exports_dir, data_dir)

    currency = neo4j_ops.get_backup_currency()

    assert currency.status == "no_backup"
    assert currency.backup_path is None
    assert currency.age_seconds is None

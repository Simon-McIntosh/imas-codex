"""Currency status and lag semantics after recovery artifacts are identified."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

from imas_codex.graph import neo4j_ops


def _file_with_mtime(path: Path, timestamp: float, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (timestamp, timestamp))
    return path


def _recovery_archive_with_mtime(
    path: Path, timestamp: float, content: bytes = b"graph data"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    member = tarfile.TarInfo(f"{path.stem}/graph.dump")
    member.size = len(content)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(content))
    os.utime(path, (timestamp, timestamp))
    return path


def _configure_paths(monkeypatch, backups_dir: Path, data_dir: Path) -> None:
    monkeypatch.setattr("imas_codex.graph.profiles.BACKUPS_DIR", backups_dir)
    monkeypatch.setattr(
        "imas_codex.graph.profiles.resolve_neo4j",
        lambda: SimpleNamespace(data_dir=data_dir),
    )


def test_reports_no_backup_separately_from_staleness(tmp_path, monkeypatch):
    backups_dir = tmp_path / "backups"
    data_dir = tmp_path / "neo4j"
    live_file = _file_with_mtime(data_dir / "data" / "store", 300.0)
    _configure_paths(monkeypatch, backups_dir, data_dir)

    result = neo4j_ops.get_backup_currency()

    assert result.status == "no_backup"
    assert result.backup_path is None
    assert result.live_path == live_file
    assert result.age_seconds is None


def test_reports_newest_backup_age_against_newest_live_file(tmp_path, monkeypatch):
    backups_dir = tmp_path / "backups"
    data_dir = tmp_path / "neo4j"
    _recovery_archive_with_mtime(backups_dir / "older.tar.gz", 100.0)
    newest_backup = _recovery_archive_with_mtime(backups_dir / "newer.tar.gz", 200.0)
    _file_with_mtime(data_dir / "data" / "older", 250.0)
    newest_live = _file_with_mtime(data_dir / "data" / "nested" / "newer", 325.0)
    _configure_paths(monkeypatch, backups_dir, data_dir)

    result = neo4j_ops.get_backup_currency()

    assert result.status == "stale"
    assert result.backup_path == newest_backup
    assert result.live_path == newest_live
    assert result.age_seconds == 125.0


def test_reports_zero_lag_when_backup_is_newer_than_live_data(tmp_path, monkeypatch):
    backups_dir = tmp_path / "backups"
    data_dir = tmp_path / "neo4j"
    newest_backup = _recovery_archive_with_mtime(backups_dir / "newer.tar.gz", 400.0)
    newest_live = _file_with_mtime(data_dir / "data" / "store", 325.0)
    _configure_paths(monkeypatch, backups_dir, data_dir)

    result = neo4j_ops.get_backup_currency()

    assert result.status == "current"
    assert result.backup_path == newest_backup
    assert result.live_path == newest_live
    assert result.age_seconds == 0.0


def test_offsite_currency_reports_newest_full_registry_copy(tmp_path, monkeypatch):
    data_dir = tmp_path / "neo4j"
    live_file = _file_with_mtime(data_dir / "data" / "store", 500.0)
    _configure_paths(monkeypatch, tmp_path / "backups", data_dir)
    monkeypatch.setattr(
        "imas_codex.graph.ghcr.list_package_versions",
        lambda registry, token, pkg_name: [
            {
                "name": "sha256:older",
                "created_at": "1970-01-01T00:01:40Z",
                "metadata": {"container": {"tags": ["v5.2.0"]}},
            },
            {
                "name": "sha256:newer",
                "created_at": "1970-01-01T00:03:20Z",
                "metadata": {"container": {"tags": ["v5.3.0-rc6", "latest"]}},
            },
        ],
    )

    result = neo4j_ops.get_offsite_currency("ghcr.io/example")

    assert result.status == "stale"
    assert result.offsite_ref == "ghcr.io/example/imas-codex-graph:v5.3.0-rc6"
    assert result.live_path == live_file
    assert result.age_seconds == 300.0


def test_offsite_currency_is_current_when_registry_copy_is_newer(tmp_path, monkeypatch):
    data_dir = tmp_path / "neo4j"
    _file_with_mtime(data_dir / "data" / "store", 200.0)
    _configure_paths(monkeypatch, tmp_path / "backups", data_dir)
    monkeypatch.setattr(
        "imas_codex.graph.ghcr.list_package_versions",
        lambda registry, token, pkg_name: [
            {
                "name": "sha256:newer",
                "created_at": "1970-01-01T00:05:00Z",
                "metadata": {"container": {"tags": ["fresh"]}},
            }
        ],
    )

    result = neo4j_ops.get_offsite_currency("ghcr.io/example")

    assert result.status == "current"
    assert result.age_seconds == 0.0


def test_offsite_currency_reports_no_offsite_copy(tmp_path, monkeypatch):
    data_dir = tmp_path / "neo4j"
    live_file = _file_with_mtime(data_dir / "data" / "store", 200.0)
    _configure_paths(monkeypatch, tmp_path / "backups", data_dir)
    monkeypatch.setattr(
        "imas_codex.graph.ghcr.list_package_versions",
        lambda registry, token, pkg_name: [],
    )

    result = neo4j_ops.get_offsite_currency("ghcr.io/example")

    assert result.status == "no_offsite"
    assert result.offsite_ref is None
    assert result.live_path == live_file
    assert result.age_seconds is None

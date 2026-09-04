"""Recovery-archive selection for graph backup currency."""

from __future__ import annotations

import io
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from imas_codex.graph import neo4j_ops


def _write_recovery_archive(path: Path, timestamp: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"recoverable graph dump"
    member = tarfile.TarInfo("imas-codex-graph/graph.dump")
    member.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(payload))
    os.utime(path, (timestamp, timestamp))
    return path


def _write_file(path: Path, timestamp: float, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.utime(path, (timestamp, timestamp))
    return path


def test_currency_uses_stale_recovery_archive_beside_newer_trial_dump(
    tmp_path: Path, monkeypatch
) -> None:
    backups_dir = tmp_path / "backups"
    data_dir = tmp_path / "neo4j"
    stale_archive = _write_recovery_archive(
        backups_dir / "imas-codex-graph.tar.gz", 100.0
    )
    trial_dump = _write_file(backups_dir / "offsite-trial.dump", 300.0, b"x" * 4_748)
    live_file = _write_file(data_dir / "data" / "checkpoint", 400.0, b"live")
    monkeypatch.setattr("imas_codex.graph.profiles.BACKUPS_DIR", backups_dir)
    monkeypatch.setattr(
        "imas_codex.graph.profiles.resolve_neo4j",
        lambda: SimpleNamespace(data_dir=data_dir),
    )

    currency = neo4j_ops.get_backup_currency()

    assert currency.status == "stale"
    assert trial_dump.stat().st_size == 4_748
    assert currency.backup_path == stale_archive
    assert currency.backup_size_bytes == stale_archive.stat().st_size
    assert currency.backup_modified_at == datetime.fromtimestamp(100.0, tz=UTC)
    assert currency.live_path == live_file
    assert currency.age_seconds == 300.0


def test_currency_does_not_treat_a_nonarchive_as_a_backup(
    tmp_path: Path, monkeypatch
) -> None:
    backups_dir = tmp_path / "backups"
    data_dir = tmp_path / "neo4j"
    trial_dump = _write_file(backups_dir / "offsite-trial.dump", 300.0, b"x" * 4_748)
    live_file = _write_file(data_dir / "data" / "checkpoint", 400.0, b"live")
    monkeypatch.setattr("imas_codex.graph.profiles.BACKUPS_DIR", backups_dir)
    monkeypatch.setattr(
        "imas_codex.graph.profiles.resolve_neo4j",
        lambda: SimpleNamespace(data_dir=data_dir),
    )

    currency = neo4j_ops.get_backup_currency()

    assert currency.status == "no_backup"
    assert trial_dump.stat().st_size == 4_748
    assert currency.backup_path is None
    assert currency.backup_size_bytes is None
    assert currency.backup_modified_at is None
    assert currency.live_path == live_file
    assert currency.age_seconds is None


def test_offsite_currency_names_full_graph_package_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "neo4j"
    live_file = _write_file(data_dir / "data" / "checkpoint", 400.0, b"live")
    monkeypatch.setattr(
        "imas_codex.graph.profiles.resolve_neo4j",
        lambda: SimpleNamespace(data_dir=data_dir),
    )
    calls: list[tuple[str, str | None, str]] = []

    def list_versions(registry: str, token: str | None, pkg_name: str) -> list[dict]:
        calls.append((registry, token, pkg_name))
        return [
            {
                "name": "sha256:full-archive",
                "created_at": "1970-01-01T00:01:40Z",
                "size_in_bytes": 2_458_442_456,
                "metadata": {"container": {"tags": ["verified"]}},
            }
        ]

    monkeypatch.setattr("imas_codex.graph.ghcr.list_package_versions", list_versions)

    currency = neo4j_ops.get_offsite_currency("ghcr.io/example", "token")

    assert calls == [("ghcr.io/example", "token", "imas-codex-graph")]
    assert currency.status == "stale"
    assert currency.offsite_ref == "ghcr.io/example/imas-codex-graph:verified"
    assert currency.offsite_size_bytes == 2_458_442_456
    assert currency.offsite_modified_at == datetime.fromtimestamp(100.0, tz=UTC)
    assert currency.live_path == live_file
    assert currency.age_seconds == 300.0

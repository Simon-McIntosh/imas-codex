"""Archive filenames identify both their code and creation time."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from imas_codex.cli.graph.data import graph_export
from imas_codex.graph.neo4j_ops import backup_graph_dump


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 1, 12, 51, 9, tzinfo=tz or UTC)


def test_default_archive_names_share_revision_and_utc_timestamp(tmp_path, monkeypatch):
    """Export archives and recovery dumps use the same immutable stamp."""
    profile = SimpleNamespace(
        name="codex",
        host="localhost",
        data_dir=tmp_path / "neo4j",
    )
    exports_dir = tmp_path / "exports"
    backups_dir = tmp_path / "backups"
    source_dump = tmp_path / "source.dump"
    profile.data_dir.mkdir()
    exports_dir.mkdir()
    source_dump.write_bytes(b"graph data")

    git_info = {
        "commit": "6fc745cf0123456789abcdef0123456789abcdef",
        "commit_short": "6fc745c",
        "tag": None,
    }
    monkeypatch.setattr(
        "imas_codex.graph.profiles.resolve_neo4j",
        lambda: profile,
    )
    monkeypatch.setattr(
        "imas_codex.graph.dirs.ensure_exports_dir",
        lambda: exports_dir,
    )
    monkeypatch.setattr(
        "imas_codex.graph.profiles.BACKUPS_DIR",
        backups_dir,
    )
    monkeypatch.setattr(
        "imas_codex.graph.remote.is_remote_location",
        lambda _host: False,
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.data.get_git_info",
        lambda: git_info,
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.data._measure_dump_scope",
        lambda _dump: {"nodes": 0, "relationships": 0, "labels": {}},
    )
    monkeypatch.setattr(
        "imas_codex.graph.ghcr.get_git_info",
        lambda: git_info,
    )
    monkeypatch.setattr("imas_codex.graph.neo4j_ops.datetime", _FrozenDateTime)

    def create_dump(_profile, dumps_dir: Path) -> None:
        (dumps_dir / "neo4j.dump").write_bytes(b"recoverable graph dump")

    monkeypatch.setattr(
        "imas_codex.graph.neo4j_ops.run_neo4j_dump",
        create_dump,
    )

    graph_export.callback(
        output=None,
        no_restart=False,
        facilities=(),
        without_dd=False,
        dd_only=False,
        local=False,
        source_dump=str(source_dump),
        version_label=None,
    )
    backup_path = backup_graph_dump()

    archive_stamp = "dev-6fc745c-20260901T125109Z"
    assert (exports_dir / f"imas-codex-graph-{archive_stamp}.tar.gz").is_file()
    assert backup_path == backups_dir / f"codex-{archive_stamp}.dump"

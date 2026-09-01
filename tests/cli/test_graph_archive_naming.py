"""Archive filenames identify both their code and creation time."""

from __future__ import annotations

import tarfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from imas_codex.cli.graph import registry as graph_registry
from imas_codex.cli.graph.data import graph_export
from imas_codex.graph.neo4j_ops import backup_graph_dump
from imas_codex.graph.remote import (
    build_remote_export_script,
    build_remote_push_script,
    build_remote_release_push_script,
)


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
    export_path = exports_dir / f"imas-codex-graph-{archive_stamp}.tar.gz"
    assert export_path.is_file()
    with tarfile.open(export_path, "r:gz") as archive:
        assert archive.getnames()[0] == f"imas-codex-graph-{archive_stamp}"
    assert backup_path == backups_dir / f"codex-{archive_stamp}.dump"


def test_remote_archive_producers_use_caller_supplied_stamped_names():
    """Remote scripts preserve one caller-resolved archive identity."""
    stamp = "dev-6fc745c-20260901T125109Z"
    full_dir = f"imas-codex-graph-{stamp}"
    full_archive = f"{full_dir}.tar.gz"
    dd_dir = f"imas-codex-graph-dd-{stamp}"
    dd_archive = f"{dd_dir}.tar.gz"
    facility_dir = f"imas-codex-graph-tcv-{stamp}"
    facility_archive = f"{facility_dir}.tar.gz"

    export_script = build_remote_export_script(
        "codex",
        archive_name=full_archive,
        archive_dir_name=full_dir,
    )
    push_script = build_remote_push_script(
        "codex",
        "ghcr.io/example/imas-codex-graph:v1.0.0",
        version_tag="v1.0.0",
        git_commit="6fc745cf0123456789abcdef0123456789abcdef",
        archive_name=full_archive,
        archive_dir_name=full_dir,
    )
    dd_push_script = build_remote_push_script(
        "codex",
        "ghcr.io/example/imas-codex-graph-dd:v1.0.0",
        version_tag="v1.0.0",
        git_commit="6fc745cf0123456789abcdef0123456789abcdef",
        dd_only=True,
        archive_name=dd_archive,
        archive_dir_name=dd_dir,
    )
    release_script = build_remote_release_push_script(
        "codex",
        "ghcr.io/example/imas-codex-graph:v1.0.0",
        dd_artifact_ref="ghcr.io/example/imas-codex-graph-dd:v1.0.0",
        facility_artifact_refs={"tcv": "ghcr.io/example/imas-codex-graph-tcv:v1.0.0"},
        version_tag="v1.0.0",
        git_commit="6fc745cf0123456789abcdef0123456789abcdef",
        archive_name=full_archive,
        archive_dir_name=full_dir,
        dd_archive_name=dd_archive,
        dd_archive_dir_name=dd_dir,
        facility_archive_names={"tcv": facility_archive},
        facility_archive_dir_names={"tcv": facility_dir},
    )

    for script in (export_script, push_script, dd_push_script, release_script):
        assert "push-$$" not in script
        assert "export-$$" not in script
    for script in (export_script, push_script, release_script):
        assert f"/{full_archive}" in script
        assert f'TMPDIR/{full_dir}"' in script
    assert f"/{dd_archive}" in release_script
    assert f"/{facility_archive}" in release_script
    assert f"/{dd_archive}" in dd_push_script
    assert f'--archive-dir-name "{dd_dir}"' in dd_push_script
    assert f'--archive-dir-name "{dd_dir}"' in release_script
    assert f'--archive-dir-name "{facility_dir}"' in release_script


def test_graph_push_temporary_archive_uses_shared_stamp(monkeypatch):
    """The local push layer uses the same immutable archive identity."""
    git_info = {
        "commit": "6fc745cf0123456789abcdef0123456789abcdef",
        "commit_short": "6fc745c",
        "is_fork": False,
        "remote_owner": "example",
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(graph_registry, "get_git_info", lambda: git_info)
    monkeypatch.setattr(graph_registry, "_ensure_fresh_version", lambda: "1.0.0")
    monkeypatch.setattr(
        graph_registry,
        "get_version_tag",
        lambda *_args, **_kwargs: "1.0.0.dev1-r1",
    )
    monkeypatch.setattr(graph_registry, "get_registry", lambda *_args: "ghcr.io/x")
    monkeypatch.setattr(graph_registry, "require_oras", lambda: None)
    monkeypatch.setattr(graph_registry, "login_to_ghcr", lambda _token: None)
    monkeypatch.setattr(graph_registry, "get_local_graph_manifest", lambda: {})
    monkeypatch.setattr(graph_registry, "save_local_graph_manifest", lambda _data: None)
    monkeypatch.setattr(graph_registry, "_save_dev_revision", lambda *_args: None)
    monkeypatch.setattr(graph_registry, "_dispatch_graph_quality", lambda *_args: None)
    monkeypatch.setattr(
        "imas_codex.graph.profiles.resolve_neo4j",
        lambda: SimpleNamespace(name="codex", host="localhost"),
    )
    monkeypatch.setattr(
        "imas_codex.graph.remote.is_remote_location", lambda _host: False
    )

    def invoke_export(_runner, _command, args):
        captured["export_args"] = args
        output_path = Path(args[args.index("-o") + 1])
        output_path.write_bytes(b"archive")
        return SimpleNamespace(exit_code=0, exception=None, output="")

    def capture_push(command, **_kwargs):
        captured["push_command"] = command

    monkeypatch.setattr("click.testing.CliRunner.invoke", invoke_export)
    monkeypatch.setattr(
        "imas_codex.cli.graph_progress.run_oras_with_progress", capture_push
    )
    monkeypatch.setattr(graph_registry, "__version__", "1.0.0")
    monkeypatch.setattr("imas_codex.graph.neo4j_ops.datetime", _FrozenDateTime)

    graph_registry.graph_push.callback(
        dev=True,
        registry=None,
        token=None,
        dry_run=False,
        facilities=(),
        without_dd=False,
        dd_only=False,
        message=None,
        verbose=False,
        version_tag_override=None,
        source_dump=None,
    )

    archive_name = "imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz"
    export_args = captured["export_args"]
    push_command = captured["push_command"]
    assert Path(export_args[export_args.index("-o") + 1]).name == archive_name
    assert push_command[3] == f"{archive_name}:application/gzip"

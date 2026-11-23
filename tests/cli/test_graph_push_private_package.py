"""Every graph push carrying facility content lands in a private package.

Each push path builds an archive whose own manifest states what it holds, and
the push reads that statement to decide whether the package must read back
private. The registry, the dump and the ORAS calls are faked here, so no
registry call leaves the process and no push is made.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from iter_codex.cli.graph.registry import _push_cycle_archive, graph_push
from iter_codex.graph.ghcr import dump_holds_facility_content

GIT_INFO = {
    "commit": "a" * 40,
    "commit_short": "aaaaaaa",
    "tag": None,
    "is_dirty": False,
    "is_fork": False,
    "remote_owner": None,
}

FACILITY_LABELS = {"Facility": 1, "FacilitySignal": 12, "GraphMeta": 1}
DD_LABELS = {"StandardName": 1, "DataDictionary": 1, "Unit": 1}


def _add(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    bundle.addfile(member, io.BytesIO(payload))


def _archive(path: Path, label_counts, *, with_dump: bool = False) -> None:
    """Write an archive carrying the manifest an export records."""
    manifest = json.dumps(
        {
            "version": "1.0",
            "git_commit": GIT_INFO["commit"],
            "node_count": 13,
            "relationship_count": 7,
            "label_counts": label_counts,
        }
    ).encode()
    with tarfile.open(path, "w:gz") as bundle:
        if with_dump:
            _add(bundle, "iter-codex-graph-stamp/graph.dump", b"dump")
        _add(bundle, "iter-codex-graph-stamp/manifest.json", manifest)


def _manifestless_archive(path: Path) -> None:
    """An archive that records no scope at all."""
    with tarfile.open(path, "w:gz") as bundle:
        _add(bundle, "iter-codex-graph-stamp/graph.dump", b"dump")


def _export_command(label_counts):
    """A stand-in for ``graph export`` that writes the archive it was given."""

    @click.command(context_settings={"ignore_unknown_options": True})
    @click.option("--output", "-o", "output", required=True)
    @click.argument("rest", nargs=-1, type=click.UNPROCESSED)
    def export(output, rest):
        _archive(Path(output), label_counts)

    return export


class _Progress:
    def __init__(self, *args, **kwargs):
        self.phases = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_total_phases(self, count):
        self.phases = count

    def start_phase(self, *args, **kwargs):
        pass

    def complete_phase(self, *args, **kwargs):
        pass

    def fail_phase(self, *args, **kwargs):
        pass


def _fake_registry(monkeypatch, *, visibility, after_set="private"):
    """A registry that reports *visibility* and answers *after_set* to a set."""
    from iter_codex.graph import ghcr

    state = {"visibility": visibility, "sets": []}

    def _get(*args, **kwargs):
        return state["visibility"]

    def _set(registry, pkg_name, vis, token=None):
        state["sets"].append(vis)
        state["visibility"] = after_set
        return after_set

    monkeypatch.setattr(ghcr, "get_package_visibility", _get)
    monkeypatch.setattr(ghcr, "set_package_visibility", _set)
    return state


def _profile(tmp_path, *, host):
    from iter_codex.graph.profiles import Neo4jProfile

    return Neo4jProfile(
        name="codex",
        location="local" if host == "localhost" else "remote",
        host=host,
        uri=f"bolt://{host}:7687",
        username="neo4j",
        password="secret",
        bolt_port=7687,
        http_port=7474,
        data_dir=tmp_path,
    )


def _prepare_local(
    monkeypatch, tmp_path, label_counts, *, visibility, after_set="private"
):
    """Wire a local push path whose archive is written by a fake export."""
    from iter_codex.cli.graph import registry as registry_mod

    calls = {"push": [], "tag": [], "manifest": []}
    profile = _profile(tmp_path, host="localhost")

    monkeypatch.setattr(
        "iter_codex.graph.profiles.resolve_neo4j", lambda *a, **k: profile
    )
    monkeypatch.setattr(
        "iter_codex.graph.remote.is_remote_location", lambda host: False
    )
    monkeypatch.setattr(registry_mod, "get_git_info", lambda: dict(GIT_INFO))
    monkeypatch.setattr(registry_mod, "require_clean_git", lambda info: None)
    monkeypatch.setattr(
        registry_mod, "get_registry", lambda info, forced: "ghcr.io/owner"
    )
    monkeypatch.setattr(registry_mod, "_ensure_fresh_version", lambda: "1.0.0")
    monkeypatch.setattr(
        registry_mod,
        "get_version_tag",
        lambda info, dev=False, version_override=None: "dev-aaaaaaa",
    )
    monkeypatch.setattr(registry_mod, "require_oras", lambda: None)
    monkeypatch.setattr(registry_mod, "login_to_ghcr", lambda token: None)
    monkeypatch.setattr(registry_mod, "_dispatch_graph_quality", lambda *a: None)
    monkeypatch.setattr(registry_mod, "get_local_graph_manifest", lambda: {})
    monkeypatch.setattr(
        registry_mod, "save_local_graph_manifest", lambda m: calls["manifest"].append(m)
    )
    monkeypatch.setattr(
        "iter_codex.cli.graph.data.graph_export", _export_command(label_counts)
    )
    monkeypatch.setattr("iter_codex.cli.graph_progress.GraphProgress", _Progress)
    monkeypatch.setattr(
        "iter_codex.cli.graph_progress.run_oras_with_progress",
        lambda cmd, **kwargs: calls["push"].append(cmd),
    )

    def _tag_run(cmd, *a, **k):
        calls["tag"].append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(registry_mod.subprocess, "run", _tag_run)
    calls["registry"] = _fake_registry(
        monkeypatch, visibility=visibility, after_set=after_set
    )
    return calls


def _prepare_remote(monkeypatch, tmp_path, *, visibility, after_set="private"):
    """Wire the remote push path so the script and its stream are recorded."""
    from iter_codex.cli.graph import registry as registry_mod

    calls = {"scripts": [], "builds": [], "manifest": []}
    profile = _profile(tmp_path, host="compute.example")

    monkeypatch.setattr(
        "iter_codex.graph.profiles.resolve_neo4j", lambda *a, **k: profile
    )
    monkeypatch.setattr("iter_codex.graph.remote.is_remote_location", lambda host: True)
    monkeypatch.setattr("iter_codex.graph.remote.remote_check_oras", lambda host: True)
    monkeypatch.setattr(registry_mod, "get_git_info", lambda: dict(GIT_INFO))
    monkeypatch.setattr(registry_mod, "require_clean_git", lambda info: None)
    monkeypatch.setattr(
        registry_mod, "get_registry", lambda info, forced: "ghcr.io/owner"
    )
    monkeypatch.setattr(registry_mod, "_ensure_fresh_version", lambda: "1.0.0")
    monkeypatch.setattr(
        registry_mod,
        "get_version_tag",
        lambda info, dev=False, version_override=None: "dev-aaaaaaa",
    )
    monkeypatch.setattr(registry_mod, "require_oras", lambda: None)
    monkeypatch.setattr(registry_mod, "login_to_ghcr", lambda token: None)
    monkeypatch.setattr(registry_mod, "_dispatch_graph_quality", lambda *a: None)
    monkeypatch.setattr(registry_mod, "get_local_graph_manifest", lambda: {})
    monkeypatch.setattr(
        registry_mod, "save_local_graph_manifest", lambda m: calls["manifest"].append(m)
    )
    monkeypatch.setattr(registry_mod, "_resolve_scheduler", lambda p: "none")
    monkeypatch.setattr(registry_mod, "_resolve_partition", lambda p: None)
    monkeypatch.setattr("iter_codex.cli.graph_progress.GraphProgress", _Progress)

    def _build(*args, **kwargs):
        calls["builds"].append(kwargs)
        return "remote script"

    monkeypatch.setattr("iter_codex.graph.remote.build_remote_push_script", _build)

    def _stream(script, host, **kwargs):
        calls["scripts"].append(script)
        return "SIZE=1.0M\n"

    monkeypatch.setattr(
        "iter_codex.cli.graph_progress.remote_operation_streaming", _stream
    )
    calls["registry"] = _fake_registry(
        monkeypatch, visibility=visibility, after_set=after_set
    )
    return calls


def test_full_graph_push_refuses_a_package_that_stays_public(monkeypatch, tmp_path):
    """A facility-bearing package whose set to private does not take effect fails."""
    calls = _prepare_local(
        monkeypatch, tmp_path, FACILITY_LABELS, visibility="public", after_set="public"
    )

    result = CliRunner().invoke(graph_push, [])

    assert result.exit_code != 0, result.output
    assert "must not be published publicly" in result.output
    assert calls["registry"]["sets"] == ["private"]
    assert calls["tag"] == []
    assert calls["manifest"] == []


def test_full_graph_push_sets_a_public_package_private_then_continues(
    monkeypatch, tmp_path
):
    """A public package holding facility content is made private, then verified."""
    calls = _prepare_local(
        monkeypatch, tmp_path, FACILITY_LABELS, visibility="public", after_set="private"
    )

    result = CliRunner().invoke(graph_push, [])

    assert result.exit_code == 0, result.output
    assert calls["registry"]["sets"] == ["private"]
    assert "private (registry-reported)" in result.output
    assert calls["tag"] == [
        ["oras", "tag", "ghcr.io/owner/iter-codex-graph:dev-aaaaaaa", "latest"]
    ]
    assert calls["manifest"]


def test_full_graph_push_continues_when_the_package_is_already_private(
    monkeypatch, tmp_path
):
    calls = _prepare_local(monkeypatch, tmp_path, FACILITY_LABELS, visibility="private")

    result = CliRunner().invoke(graph_push, [])

    assert result.exit_code == 0, result.output
    assert calls["registry"]["sets"] == []
    assert "private (registry-reported)" in result.output
    assert calls["tag"] == [
        ["oras", "tag", "ghcr.io/owner/iter-codex-graph:dev-aaaaaaa", "latest"]
    ]
    assert calls["manifest"]


def test_facility_filtered_push_refuses_a_package_that_stays_public(
    monkeypatch, tmp_path
):
    calls = _prepare_local(
        monkeypatch, tmp_path, FACILITY_LABELS, visibility="public", after_set="public"
    )

    result = CliRunner().invoke(graph_push, ["--facility", "iter"])

    assert result.exit_code != 0, result.output
    assert "must not be published publicly" in result.output
    assert calls["registry"]["sets"] == ["private"]
    assert calls["tag"] == []


def test_dd_only_push_never_queries_visibility(monkeypatch, tmp_path):
    from iter_codex.graph import ghcr

    calls = _prepare_local(monkeypatch, tmp_path, DD_LABELS, visibility="public")
    monkeypatch.setattr(
        ghcr,
        "get_package_visibility",
        lambda *a, **k: pytest.fail("visibility read for a DD-only package"),
    )
    monkeypatch.setattr(
        ghcr,
        "set_package_visibility",
        lambda *a, **k: pytest.fail("visibility write for a DD-only package"),
    )

    result = CliRunner().invoke(graph_push, ["--dd-only"])

    assert result.exit_code == 0, result.output
    assert calls["tag"]


def test_cycle_push_refuses_a_package_that_stays_public(monkeypatch, tmp_path):
    calls = _prepare_local(
        monkeypatch, tmp_path, FACILITY_LABELS, visibility="public", after_set="public"
    )
    cycle_archive = tmp_path / "cycle.tar.gz"
    _archive(cycle_archive, FACILITY_LABELS, with_dump=True)

    with pytest.raises(click.ClickException):
        _push_cycle_archive(
            cycle_archive,
            "20260101T000000Z",
            registry="ghcr.io/owner",
            token=None,
        )

    assert calls["registry"]["sets"] == ["private"]
    assert calls["tag"] == []
    assert calls["manifest"] == []


def test_cycle_push_records_the_push_when_the_package_is_private(monkeypatch, tmp_path):
    calls = _prepare_local(monkeypatch, tmp_path, FACILITY_LABELS, visibility="private")
    cycle_archive = tmp_path / "cycle.tar.gz"
    _archive(cycle_archive, FACILITY_LABELS, with_dump=True)

    reference = _push_cycle_archive(
        cycle_archive,
        "20260101T000000Z",
        registry="ghcr.io/owner",
        token=None,
    )

    assert reference.endswith("iter-codex-graph:20260101T000000Z-r1")
    assert calls["manifest"]


def test_remote_push_refuses_a_package_that_stays_public_before_tagging(
    monkeypatch, tmp_path
):
    calls = _prepare_remote(
        monkeypatch, tmp_path, visibility="public", after_set="public"
    )

    result = CliRunner().invoke(graph_push, [])

    assert result.exit_code != 0, result.output
    assert "must not be published publicly" in result.output
    assert calls["registry"]["sets"] == ["private"]
    assert calls["builds"][0]["tag_latest"] is False
    assert calls["scripts"] == ["remote script"]


def test_remote_push_sets_a_public_package_private_then_tags(monkeypatch, tmp_path):
    calls = _prepare_remote(
        monkeypatch, tmp_path, visibility="public", after_set="private"
    )

    result = CliRunner().invoke(graph_push, [])

    assert result.exit_code == 0, result.output
    assert calls["registry"]["sets"] == ["private"]
    assert calls["builds"][0]["tag_latest"] is False
    assert len(calls["scripts"]) == 2
    assert "oras tag" in calls["scripts"][1]
    assert "latest" in calls["scripts"][1]


def test_remote_push_tags_after_a_private_report(monkeypatch, tmp_path):
    calls = _prepare_remote(monkeypatch, tmp_path, visibility="private")

    result = CliRunner().invoke(graph_push, [])

    assert result.exit_code == 0, result.output
    assert calls["builds"][0]["tag_latest"] is False
    assert len(calls["scripts"]) == 2
    assert "oras tag" in calls["scripts"][1]
    assert "latest" in calls["scripts"][1]


def test_the_decision_reads_the_archive_scope_not_the_push_flags():
    assert dump_holds_facility_content(FACILITY_LABELS) is True
    assert dump_holds_facility_content(DD_LABELS) is False
    # A scope that cannot be read cannot clear a facility package.
    assert dump_holds_facility_content(None) is True
    assert dump_holds_facility_content({}) is True


def test_an_archive_without_a_manifest_is_treated_as_facility_bearing(
    monkeypatch, tmp_path
):
    calls = _prepare_local(
        monkeypatch, tmp_path, FACILITY_LABELS, visibility="public", after_set="public"
    )

    @click.command(context_settings={"ignore_unknown_options": True})
    @click.option("--output", "-o", "output", required=True)
    @click.argument("rest", nargs=-1, type=click.UNPROCESSED)
    def export(output, rest):
        _manifestless_archive(Path(output))

    monkeypatch.setattr("iter_codex.cli.graph.data.graph_export", export)

    result = CliRunner().invoke(graph_push, [])

    assert result.exit_code != 0, result.output
    assert calls["registry"]["sets"] == ["private"]
    assert calls["tag"] == []

"""Archive manifests describe the graph stored in the adjacent dump."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from imas_codex.cli.graph.data import graph_export


def _read_archive_manifest(archive: Path) -> dict:
    with tarfile.open(archive, "r:gz") as tar:
        member = next(
            item for item in tar.getmembers() if item.name.endswith("manifest.json")
        )
        manifest_file = tar.extractfile(member)
        assert manifest_file is not None
        return json.load(manifest_file)


@pytest.fixture
def export_graph(tmp_path, monkeypatch):
    profile = SimpleNamespace(
        name="codex",
        host="localhost",
        data_dir=tmp_path / "neo4j",
    )
    profile.data_dir.mkdir()
    source_dump = tmp_path / "source.dump"
    source_dump.write_bytes(b"graph dump")

    monkeypatch.setattr("imas_codex.graph.profiles.resolve_neo4j", lambda: profile)
    monkeypatch.setattr(
        "imas_codex.graph.remote.is_remote_location", lambda _host: False
    )
    monkeypatch.setattr("imas_codex.graph.ghcr.require_apptainer", lambda: None)
    monkeypatch.setattr(
        "imas_codex.cli.graph.data.get_git_info",
        lambda: {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "commit_short": "0123456",
            "tag": None,
        },
    )

    def run(
        *,
        counts: dict,
        facilities: tuple[str, ...] = (),
        measure=None,
    ) -> dict:
        archive = tmp_path / f"archive-{len(list(tmp_path.glob('archive-*')))}.tar.gz"
        monkeypatch.setattr(
            "imas_codex.cli.graph.data._measure_dump_scope",
            measure or (lambda dump_path: counts),
            raising=False,
        )
        graph_export.callback(
            output=str(archive),
            no_restart=False,
            facilities=facilities,
            without_dd=False,
            dd_only=False,
            local=False,
            source_dump=str(source_dump),
            version_label="test",
        )
        return _read_archive_manifest(archive)

    return run


def test_unfiltered_archive_manifest_records_dump_scope(export_graph):
    manifest = export_graph(
        counts={
            "nodes": 11,
            "relationships": 7,
            "labels": {"Facility": 2, "Shared": 3},
        }
    )

    assert manifest["node_count"] == 11
    assert manifest["relationship_count"] == 7
    assert manifest["label_counts"] == {"Facility": 2, "Shared": 3}


def test_filtered_archive_manifest_records_post_filter_scope(export_graph, monkeypatch):
    filter_finished = False

    def filter_dump(source: Path, facility: str, output: Path) -> None:
        nonlocal filter_finished
        assert facility == "tcv"
        assert source == output
        filter_finished = True

    def measured_counts(dump_path: Path) -> dict:
        assert filter_finished
        return {
            "nodes": 4,
            "relationships": 2,
            "labels": {"Facility": 1, "Shared": 2},
        }

    monkeypatch.setattr("imas_codex.cli.graph.data._create_facility_dump", filter_dump)
    manifest = export_graph(
        counts={
            "nodes": 4,
            "relationships": 2,
            "labels": {"Facility": 1, "Shared": 2},
        },
        facilities=("tcv",),
        measure=measured_counts,
    )

    assert manifest["node_count"] == 4
    assert manifest["relationship_count"] == 2
    assert manifest["label_counts"] == {"Facility": 1, "Shared": 2}

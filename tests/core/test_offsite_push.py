"""Count-gated full-graph offsite push cycles."""

from __future__ import annotations

import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from imas_codex.graph.neo4j_ops import OffsiteCurrency
from imas_codex.graph.offsite import (
    GraphCensus,
    OffsiteCountMismatch,
    OffsitePushFailed,
    run_offsite_push_cycle,
)


def _currency(status: str) -> OffsiteCurrency:
    return OffsiteCurrency(
        status=status,
        offsite_ref="ghcr.io/example/imas-codex-graph:existing",
        offsite_modified_at=datetime(2026, 7, 7, tzinfo=UTC),
        live_path=Path("/graph/data/store"),
        live_modified_at=datetime(2026, 9, 1, tzinfo=UTC),
        age_seconds=0.0 if status == "current" else 4_838_400.0,
    )


def _archive(tmp_path: Path, census: GraphCensus) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "graph.dump").write_bytes(b"graph archive")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "node_count": census.node_count,
                "relationship_count": census.relationship_count,
                "label_counts": census.label_counts,
            }
        )
    )
    archive = tmp_path / "graph.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(root, arcname="imas-codex-graph-test")
    return archive


def test_current_copy_records_no_op_without_export(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "imas_codex.graph.offsite.graph_archive_stamp", lambda created_at: "stamp"
    )
    exported = False

    def export(stamp: str) -> Path:
        nonlocal exported
        exported = True
        raise AssertionError(stamp)

    result = run_offsite_push_cycle(
        currency=_currency("current"),
        export_archive=export,
        push_archive=lambda archive, stamp: "unreachable",
        census=lambda: GraphCensus(1, 1, {"Facility": 1}),
        receipt_dir=tmp_path / "receipts",
    )

    assert result.outcome == "no_op"
    assert result.archive_bytes == 0
    assert not exported
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["outcome"] == "no_op"
    assert receipt["archive_bytes"] == 0


def test_matching_archive_is_pushed_and_receipted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "imas_codex.graph.offsite.graph_archive_stamp", lambda created_at: "stamp"
    )
    counts = GraphCensus(11, 7, {"Facility": 2, "Shared": 3})
    archive = _archive(tmp_path, counts)
    pushed: list[tuple[Path, str]] = []

    result = run_offsite_push_cycle(
        currency=_currency("stale"),
        export_archive=lambda stamp: archive,
        push_archive=lambda path, stamp: (
            pushed.append((path, stamp)) or "ghcr.io/example/imas-codex-graph:stamp"
        ),
        census=lambda: counts,
        receipt_dir=tmp_path / "receipts",
    )

    assert result.outcome == "pushed"
    assert pushed == [(archive, "stamp")]
    assert result.archive_bytes == archive.stat().st_size
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["counts_match"] is True
    assert receipt["wall_time_seconds"] >= 0
    assert receipt["archive_bytes"] == archive.stat().st_size


def test_count_mismatch_refuses_push_and_records_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "imas_codex.graph.offsite.graph_archive_stamp", lambda created_at: "stamp"
    )
    archive = _archive(tmp_path, GraphCensus(10, 7, {"Facility": 2}))
    pushed = False

    def push(path: Path, stamp: str) -> str:
        nonlocal pushed
        pushed = True
        return "unreachable"

    with pytest.raises(OffsiteCountMismatch) as caught:
        run_offsite_push_cycle(
            currency=_currency("stale"),
            export_archive=lambda stamp: archive,
            push_archive=push,
            census=lambda: GraphCensus(11, 7, {"Facility": 2}),
            receipt_dir=tmp_path / "receipts",
        )

    assert not pushed
    receipt = json.loads(caught.value.result.receipt_path.read_text())
    assert receipt["outcome"] == "refused"
    assert receipt["counts_match"] is False
    assert receipt["live_census"]["node_count"] == 11
    assert receipt["archive_census"]["node_count"] == 10


def test_upload_failure_records_archive_and_error_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "imas_codex.graph.offsite.graph_archive_stamp", lambda created_at: "stamp"
    )
    counts = GraphCensus(11, 7, {"Facility": 2})
    archive = _archive(tmp_path, counts)

    def fail_upload(path: Path, stamp: str) -> str:
        assert path == archive
        assert stamp == "stamp"
        raise RuntimeError("oras unavailable during upload preparation")

    with pytest.raises(OffsitePushFailed) as caught:
        run_offsite_push_cycle(
            currency=_currency("stale"),
            export_archive=lambda stamp: archive,
            push_archive=fail_upload,
            census=lambda: counts,
            receipt_dir=tmp_path / "receipts",
        )

    receipt = json.loads(caught.value.result.receipt_path.read_text())
    assert receipt["outcome"] == "failed"
    assert receipt["archive_stamp"] == "stamp"
    assert receipt["archive_path"] == str(archive)
    assert receipt["archive_bytes"] == archive.stat().st_size
    assert receipt["wall_time_seconds"] >= 0
    assert receipt["counts_match"] is True
    assert receipt["error"] == (
        "RuntimeError: oras unavailable during upload preparation"
    )

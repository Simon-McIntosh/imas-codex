"""Count-gated full-graph offsite push cycles."""

from __future__ import annotations

import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from imas_codex.graph.neo4j_ops import OffsiteCurrency
from imas_codex.graph.offsite import (
    GraphActivity,
    GraphCensus,
    OffsiteCountMismatch,
    OffsitePushFailed,
    live_graph_activity,
    run_offsite_push_cycle,
)

IDLE = GraphActivity(window_seconds=600.0, transactions_started=0, open_queries=[])


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
        activity=lambda: IDLE,
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
            activity=lambda: IDLE,
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
            activity=lambda: IDLE,
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


def _not_reached(*_args):
    raise AssertionError("the cycle went past the guard")


def test_a_copy_under_a_week_old_is_not_due_and_leaves_the_graph_running(tmp_path):
    currency = _currency("stale")
    result = run_offsite_push_cycle(
        currency=currency,
        export_archive=_not_reached,
        push_archive=_not_reached,
        census=_not_reached,
        activity=_not_reached,
        receipt_dir=tmp_path / "receipts",
        now=lambda: datetime(2026, 7, 9, tzinfo=UTC),
    )

    assert result.outcome == "not_due"
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["outcome"] == "not_due"
    assert receipt["archive_path"] is None


def test_a_due_push_defers_while_other_clients_work_on_the_graph(tmp_path):
    busy = GraphActivity(
        window_seconds=600.0,
        transactions_started=41,
        open_queries=["MATCH (s:FacilitySignal) SET s.status = $status"],
    )
    result = run_offsite_push_cycle(
        currency=_currency("stale"),
        export_archive=_not_reached,
        push_archive=_not_reached,
        census=_not_reached,
        activity=lambda: busy,
        receipt_dir=tmp_path / "receipts",
    )

    assert result.outcome == "deferred"
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["outcome"] == "deferred"
    assert receipt["activity"]["busy"] is True
    assert receipt["activity"]["transactions_started"] == 41


class _ProbeGraph:
    """Answers SHOW TRANSACTIONS with a scripted sequence of probe rows."""

    def __init__(self, probes):
        self.probes = iter(probes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher):
        assert cypher.startswith("SHOW TRANSACTIONS")
        return next(self.probes)


def _own(tx: int) -> dict:
    return {
        "transactionId": f"neo4j-transaction-{tx}",
        "currentQuery": "SHOW TRANSACTIONS YIELD",
    }


def _activity(monkeypatch, probes) -> GraphActivity:
    monkeypatch.setattr(
        "imas_codex.graph.client.GraphClient", lambda: _ProbeGraph(probes)
    )
    return live_graph_activity(
        window_seconds=120.0, interval_seconds=60.0, sleep=lambda _s: None
    )


def test_probes_alone_measure_an_idle_graph(monkeypatch):
    observed = _activity(monkeypatch, [[_own(500)], [_own(501)], [_own(502)]])

    assert observed.transactions_started == 0
    assert not observed.busy


def test_transactions_between_probes_mark_the_graph_busy(monkeypatch):
    observed = _activity(monkeypatch, [[_own(500)], [_own(504)], [_own(505)]])

    assert observed.transactions_started == 3
    assert observed.busy


def test_a_transaction_seen_running_marks_the_graph_busy(monkeypatch):
    reader = {
        "transactionId": "neo4j-transaction-499",
        "currentQuery": "MATCH (n) RETURN n",
    }
    observed = _activity(monkeypatch, [[_own(500), reader], [_own(501)], [_own(502)]])

    assert observed.transactions_started == 0
    assert observed.open_queries == ["MATCH (n) RETURN n"]
    assert observed.busy

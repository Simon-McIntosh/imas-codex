"""Verified full-graph offsite push receipts."""

from __future__ import annotations

import json
import tarfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from imas_codex.graph.dirs import DATA_BASE_DIR
from imas_codex.graph.neo4j_ops import OffsiteCurrency, graph_archive_stamp

OFFSITE_RECEIPTS_DIR = DATA_BASE_DIR / "offsite-push" / "receipts"
PUSH_INTERVAL_SECONDS = 7 * 86_400.0
ACTIVITY_WINDOW_SECONDS = 600.0
ACTIVITY_INTERVAL_SECONDS = 60.0
_ACTIVITY_PROBE = (
    "SHOW TRANSACTIONS YIELD transactionId, currentQuery "
    "RETURN transactionId, currentQuery"
)


@dataclass(frozen=True, slots=True)
class GraphCensus:
    """Counts that identify the graph scope represented by an archive."""

    node_count: int
    relationship_count: int
    label_counts: dict[str, int]

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> GraphCensus:
        """Read the required census fields from an archive manifest."""
        try:
            return cls(
                node_count=int(manifest["node_count"]),
                relationship_count=int(manifest["relationship_count"]),
                label_counts={
                    str(label): int(count)
                    for label, count in manifest["label_counts"].items()
                },
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("archive manifest has no valid graph census") from exc

    def normalized(self) -> GraphCensus:
        """Return a stable representation for equality and receipts."""
        return GraphCensus(
            node_count=self.node_count,
            relationship_count=self.relationship_count,
            label_counts=dict(sorted(self.label_counts.items())),
        )


@dataclass(frozen=True, slots=True)
class GraphActivity:
    """Work other clients ran on the graph during one observation window.

    Transaction ids are one sequence across every session, so the gap
    between the first and last probe counts the transactions anyone else
    started in between, reads included; ``open_queries`` holds the other
    transactions seen running at any probe.
    """

    window_seconds: float
    transactions_started: int
    open_queries: list[str]

    @property
    def busy(self) -> bool:
        return self.transactions_started > 0 or bool(self.open_queries)


@dataclass(frozen=True, slots=True)
class OffsitePushResult:
    """Outcome and durable receipt path for one scheduled cycle."""

    outcome: Literal["no_op", "not_due", "deferred", "pushed", "refused", "failed"]
    receipt_path: Path
    archive_ref: str | None
    archive_bytes: int
    wall_time_seconds: float


class OffsiteCountMismatch(RuntimeError):
    """The exported archive does not represent the live push-time graph."""

    def __init__(self, result: OffsitePushResult):
        super().__init__(
            f"archive census does not match the live graph; receipt: "
            f"{result.receipt_path}"
        )
        self.result = result


class OffsitePushFailed(RuntimeError):
    """An archive passed census verification but could not be uploaded."""

    def __init__(self, result: OffsitePushResult, error: str):
        super().__init__(
            f"offsite upload failed: {error}; receipt: {result.receipt_path}"
        )
        self.result = result


def live_graph_census() -> GraphCensus:
    """Measure nodes, relationships, and labels through the graph client."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as graph:
        totals = graph.get_stats()
        label_rows = graph.query(
            "MATCH (n) UNWIND labels(n) AS label "
            "RETURN label, count(*) AS count ORDER BY label"
        )
        labels = {row["label"]: row["count"] for row in label_rows}
    return GraphCensus(
        node_count=int(totals["nodes"]),
        relationship_count=int(totals["relationships"]),
        label_counts={str(label): int(count) for label, count in labels.items()},
    ).normalized()


def live_graph_activity(
    window_seconds: float = ACTIVITY_WINDOW_SECONDS,
    interval_seconds: float = ACTIVITY_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> GraphActivity:
    """Watch the graph for other clients' transactions over one window.

    Every probe runs in the same session and is itself one transaction, so
    probes are subtracted from the id gap; an idle graph measures zero.
    """
    from imas_codex.graph.client import GraphClient

    probes = max(2, int(window_seconds // interval_seconds) + 1)
    first_id = last_id = 0
    open_queries: list[str] = []
    with GraphClient() as graph:
        for probe in range(probes):
            if probe:
                sleep(interval_seconds)
            own_id = None
            for row in graph.query(_ACTIVITY_PROBE):
                query = row["currentQuery"] or ""
                if query.startswith("SHOW TRANSACTIONS"):
                    own_id = int(str(row["transactionId"]).rsplit("-", 1)[1])
                elif query.strip():
                    open_queries.append(query.strip()[:120])
            if own_id is None:
                raise RuntimeError("SHOW TRANSACTIONS did not list its own probe")
            if probe == 0:
                first_id = own_id
            last_id = own_id
    return GraphActivity(
        window_seconds=(probes - 1) * interval_seconds,
        transactions_started=max(0, last_id - first_id - (probes - 1)),
        open_queries=open_queries,
    )


def read_archive_manifest(archive_path: Path) -> dict[str, Any]:
    """Read the single manifest adjacent to the dump in a graph archive."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and Path(member.name).name == "manifest.json"
        ]
        if len(members) != 1:
            raise ValueError(f"expected one archive manifest, found {len(members)}")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError("archive manifest is unreadable")
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("archive manifest is not a JSON object")
    return payload


def _receipt_payload(
    *,
    outcome: str,
    stamp: str,
    started_at: datetime,
    completed_at: datetime,
    wall_time_seconds: float,
    archive_ref: str | None,
    archive_path: Path | None,
    archive_bytes: int,
    currency: OffsiteCurrency,
    live_census: GraphCensus | None,
    archive_census: GraphCensus | None,
    error: str | None = None,
    activity: GraphActivity | None = None,
) -> dict[str, Any]:
    return {
        "schema": "imas-codex.offsite-push-receipt",
        "outcome": outcome,
        "archive_stamp": stamp,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "wall_time_seconds": round(wall_time_seconds, 6),
        "archive_ref": archive_ref,
        "archive_path": str(archive_path) if archive_path else None,
        "archive_bytes": archive_bytes,
        "currency": {
            "status": currency.status,
            "offsite_ref": currency.offsite_ref,
            "offsite_modified_at": (
                currency.offsite_modified_at.isoformat()
                if currency.offsite_modified_at
                else None
            ),
            "live_path": str(currency.live_path) if currency.live_path else None,
            "live_modified_at": (
                currency.live_modified_at.isoformat()
                if currency.live_modified_at
                else None
            ),
            "age_seconds": currency.age_seconds,
        },
        "live_census": asdict(live_census) if live_census else None,
        "archive_census": asdict(archive_census) if archive_census else None,
        "counts_match": (
            live_census.normalized() == archive_census.normalized()
            if live_census and archive_census
            else None
        ),
        "error": error,
        "activity": ({**asdict(activity), "busy": activity.busy} if activity else None),
    }


def _held_cycle(
    outcome: Literal["not_due", "deferred"],
    stamp: str,
    started_at: datetime,
    started: float,
    currency: OffsiteCurrency,
    receipt_dir: Path,
    clock: Callable[[], datetime],
    monotonic: Callable[[], float],
    observed: GraphActivity | None,
) -> OffsitePushResult:
    """Write the receipt of a cycle that left the graph running."""
    elapsed = monotonic() - started
    payload = _receipt_payload(
        outcome=outcome,
        stamp=stamp,
        started_at=started_at,
        completed_at=clock(),
        wall_time_seconds=elapsed,
        archive_ref=currency.offsite_ref,
        archive_path=None,
        archive_bytes=0,
        currency=currency,
        live_census=None,
        archive_census=None,
        activity=observed,
    )
    receipt = _write_receipt(receipt_dir, stamp, payload)
    return OffsitePushResult(outcome, receipt, currency.offsite_ref, 0, elapsed)


def _write_receipt(
    receipt_dir: Path,
    stamp: str,
    payload: dict[str, Any],
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{stamp}.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    return receipt_path


def run_offsite_push_cycle(
    *,
    currency: OffsiteCurrency,
    export_archive: Callable[[str], Path],
    push_archive: Callable[[Path, str], str],
    census: Callable[[], GraphCensus] = live_graph_census,
    activity: Callable[[], GraphActivity] = live_graph_activity,
    push_interval_seconds: float = PUSH_INTERVAL_SECONDS,
    receipt_dir: Path = OFFSITE_RECEIPTS_DIR,
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> OffsitePushResult:
    """Run one no-op, not-due, deferred or push cycle and write its receipt.

    The timer fires daily. A push falls due once the newest offsite copy is
    ``push_interval_seconds`` old, and a due push waits for a day on which
    no other client is working on the graph: exporting stops the database,
    and writes landing between the census and the stop make the archive
    disagree with the census, so a busy graph could never pass anyway.
    """
    clock = now or (lambda: datetime.now(UTC))
    started_at = clock()
    started = monotonic()
    stamp = graph_archive_stamp(created_at=started_at)

    if currency.status == "current":
        completed_at = clock()
        elapsed = monotonic() - started
        payload = _receipt_payload(
            outcome="no_op",
            stamp=stamp,
            started_at=started_at,
            completed_at=completed_at,
            wall_time_seconds=elapsed,
            archive_ref=currency.offsite_ref,
            archive_path=None,
            archive_bytes=0,
            currency=currency,
            live_census=None,
            archive_census=None,
        )
        receipt = _write_receipt(receipt_dir, stamp, payload)
        return OffsitePushResult("no_op", receipt, currency.offsite_ref, 0, elapsed)

    offsite_age = (
        (started_at - currency.offsite_modified_at).total_seconds()
        if currency.offsite_modified_at
        else None
    )
    if offsite_age is not None and offsite_age < push_interval_seconds:
        return _held_cycle(
            "not_due",
            stamp,
            started_at,
            started,
            currency,
            receipt_dir,
            clock,
            monotonic,
            None,
        )

    observed = activity()
    if observed.busy:
        return _held_cycle(
            "deferred",
            stamp,
            started_at,
            started,
            currency,
            receipt_dir,
            clock,
            monotonic,
            observed,
        )

    live_census = census().normalized()
    archive_path = export_archive(stamp)
    archive_bytes = archive_path.stat().st_size
    archive_census = GraphCensus.from_manifest(
        read_archive_manifest(archive_path)
    ).normalized()

    if archive_census != live_census:
        completed_at = clock()
        elapsed = monotonic() - started
        payload = _receipt_payload(
            outcome="refused",
            stamp=stamp,
            started_at=started_at,
            completed_at=completed_at,
            wall_time_seconds=elapsed,
            archive_ref=None,
            archive_path=archive_path,
            archive_bytes=archive_bytes,
            currency=currency,
            live_census=live_census,
            archive_census=archive_census,
        )
        receipt = _write_receipt(receipt_dir, stamp, payload)
        result = OffsitePushResult("refused", receipt, None, archive_bytes, elapsed)
        raise OffsiteCountMismatch(result)

    try:
        archive_ref = push_archive(archive_path, stamp)
    except Exception as exc:
        completed_at = clock()
        elapsed = monotonic() - started
        error = f"{type(exc).__name__}: {exc}"
        payload = _receipt_payload(
            outcome="failed",
            stamp=stamp,
            started_at=started_at,
            completed_at=completed_at,
            wall_time_seconds=elapsed,
            archive_ref=None,
            archive_path=archive_path,
            archive_bytes=archive_bytes,
            currency=currency,
            live_census=live_census,
            archive_census=archive_census,
            error=error,
        )
        receipt = _write_receipt(receipt_dir, stamp, payload)
        result = OffsitePushResult("failed", receipt, None, archive_bytes, elapsed)
        raise OffsitePushFailed(result, error) from exc

    completed_at = clock()
    elapsed = monotonic() - started
    payload = _receipt_payload(
        outcome="pushed",
        stamp=stamp,
        started_at=started_at,
        completed_at=completed_at,
        wall_time_seconds=elapsed,
        archive_ref=archive_ref,
        archive_path=archive_path,
        archive_bytes=archive_bytes,
        currency=currency,
        live_census=live_census,
        archive_census=archive_census,
    )
    receipt = _write_receipt(receipt_dir, stamp, payload)
    return OffsitePushResult("pushed", receipt, archive_ref, archive_bytes, elapsed)

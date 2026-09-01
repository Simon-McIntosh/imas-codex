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
class OffsitePushResult:
    """Outcome and durable receipt path for one scheduled cycle."""

    outcome: Literal["no_op", "pushed", "refused"]
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
    }


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
    receipt_dir: Path = OFFSITE_RECEIPTS_DIR,
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> OffsitePushResult:
    """Run one no-op-or-push cycle and write its acceptance receipt."""
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

    archive_ref = push_archive(archive_path, stamp)
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

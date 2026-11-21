"""Group a private archive catalogue and optionally project existing graph members.

Invoke with ``uv run python -m iter_codex.remote.catalogue_grouping --help``.
The receipt contains private selectors and field sets and must stay private.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from iter_codex.discovery.signals.archive_grouping import (
    ArchiveGrouping,
    group_private_catalogue,
)
from iter_codex.remote.catalogue_scan import write_receipt


def run_grouping(
    *,
    catalogue: Path,
    store: Path,
    facility: str,
    receipt_path: Path,
    persist_graph: bool = False,
) -> dict:
    """Return a durable result for both the first pass and its idempotent repeat."""
    if catalogue.resolve() == store.resolve():
        raise ValueError("grouping store must be separate from the source catalogue")
    started = time.monotonic()
    receipt = {
        "status": "running",
        "facility": facility,
        "catalogue": str(catalogue.resolve()),
        "store": str(store.resolve()),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "started_at": datetime.now(UTC).isoformat(),
    }
    write_receipt(receipt_path, receipt)
    graph = grouping = None
    try:
        signals = []
        if persist_graph:
            from iter_codex.discovery.signals.archive_grouping_graph import (
                snapshot_archive_signals,
            )
            from iter_codex.graph.client import GraphClient

            graph = GraphClient()
            signals = snapshot_archive_signals(graph, facility)
        grouping = ArchiveGrouping(store)
        receipt["first"] = group_private_catalogue(
            catalogue, grouping, facility, graph_signals=signals
        )
        write_receipt(receipt_path, receipt)
        # Replay the same snapshot, not a fresh read of a growing metadata store.
        receipt["repeat"] = grouping.rebuild(
            facility, grouping.snapshot_records(facility)
        )
        if any(
            receipt["repeat"][key]
            for key in ("groups_added", "control_groups_added", "patterns_added")
        ):
            raise RuntimeError(
                "repeated grouping added component groups, control groups or patterns"
            )
        if graph is not None:
            from iter_codex.discovery.signals.archive_grouping_graph import (
                graph_group_counts,
                persist_archive_grouping,
            )

            receipt["graph"] = persist_archive_grouping(
                graph, grouping, facility, signals
            )
            receipt["graph_counts"] = graph_group_counts(graph, facility)
            receipt["graph_repeat"] = persist_archive_grouping(
                graph, grouping, facility, signals
            )
            receipt["graph_counts_repeat"] = graph_group_counts(graph, facility)
            if receipt["graph"] != receipt["graph_repeat"]:
                raise RuntimeError("graph grouping repeat changed coverage")
            if receipt["graph_counts"] != receipt["graph_counts_repeat"]:
                raise RuntimeError("graph grouping repeat changed group counts")
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if grouping is not None:
            grouping.close()
        if graph is not None:
            graph.close()
        receipt.update(
            elapsed_seconds=time.monotonic() - started,
            finished_at=datetime.now(UTC).isoformat(),
        )
        write_receipt(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--facility", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--persist-graph", action="store_true")
    args = parser.parse_args(argv)
    run_grouping(
        catalogue=args.catalogue,
        store=args.store,
        facility=args.facility,
        receipt_path=args.receipt,
        persist_graph=args.persist_graph,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

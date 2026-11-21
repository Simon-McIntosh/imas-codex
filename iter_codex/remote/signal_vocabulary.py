"""Assign the closed signal vocabulary to one facility's cohort and store it.

Invoke with ``uv run python -m iter_codex.remote.signal_vocabulary --help``.
The receipt carries signal identities and the words their sources used, so it is
private: write it under the private cache, never into the repository.

The cohort is the plant-context-assessed catalogue, and every source is read from
what the graph already holds — the linked signal-list rows, the controls
metadata a signal carries, and the archive's stored metadata document. No archive
or document host is contacted, and no credential is read.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from iter_codex.discovery.signals.vocabulary_graph import (
    apply_assignment,
    assign_cohort,
    snapshot_cohort,
    unit_agreement,
    verify_assignment,
)
from iter_codex.graph.client import GraphClient


def run_assignment(*, facility: str, receipt_path: Path, dry_run: bool = False) -> dict:
    """Assign, optionally write, verify, and prove a repeat changes nothing."""
    receipt: dict = {
        "facility": facility,
        "taken_at": datetime.now(UTC).isoformat(),
        "archive_contacted": False,
        "document_host_contacted": False,
    }
    started = datetime.now(UTC)
    with GraphClient() as graph:
        signals, rows = snapshot_cohort(graph, facility)
        receipt["cohort_signals"] = len(signals)
        receipt["linked_rows"] = len(rows)
        first = assign_cohort(signals, rows)
        receipt["summary"] = first["summary"]
        receipt["changed"] = first["changed"]
        receipt["unit_agreement"] = unit_agreement(signals, first["ledger"])
        if dry_run:
            receipt["status"] = "dry-run"
        else:
            receipt["write"] = apply_assignment(graph, facility, first["pending"])
            receipt["verify"] = verify_assignment(graph, facility, first["ledger"])
            if receipt["verify"]["mismatched"]:
                raise RuntimeError("Stored vocabulary does not match its assignment")
            second_signals, second_rows = snapshot_cohort(graph, facility)
            second = assign_cohort(second_signals, second_rows)
            receipt["repeat"] = {"changed": second["changed"]}
            if second["changed"]:
                raise RuntimeError("A repeated assignment changed a signal")
            receipt["status"] = "complete"
    receipt["elapsed_seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facility", default="iter")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="assign and report without writing to the graph",
    )
    arguments = parser.parse_args(argv)
    receipt = run_assignment(
        facility=arguments.facility,
        receipt_path=arguments.receipt,
        dry_run=arguments.dry_run,
    )
    print(json.dumps({key: receipt[key] for key in ("status", "cohort_signals")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

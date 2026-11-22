"""Build a facility's field-code table from its catalogue and store it privately.

Invoke with ``uv run python -m iter_codex.remote.field_code_table --help``. The
table's rows name real archive codes and the meanings read for them, so the
receipt and the table it writes are private: write both under the private cache,
never into the repository.

The meanings are declared in the facility's private configuration under
``field_code_meanings``; the roles are read from the sources the table records.
A build is deterministic, so a second build changes no entry, and that is
asserted rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from iter_codex.discovery.base.facility import get_facility_infrastructure
from iter_codex.discovery.signals.field_codes_graph import (
    build_from_graph,
    meanings_from_config,
)
from iter_codex.graph.client import GraphClient

MEANINGS_KEY = "field_code_meanings"


def build(*, facility: str, receipt_path: Path, dry_run: bool = False) -> dict:
    """Build the table twice, prove it is stable, and write the private output."""
    started = datetime.now(UTC)
    private = get_facility_infrastructure(facility) or {}
    entries = private.get(MEANINGS_KEY) or []
    meanings = meanings_from_config(entries)
    receipt: dict = {
        "facility": facility,
        "taken_at": started.isoformat(),
        "declared_meanings": len(meanings),
        "archive_contacted": False,
        "document_host_contacted": False,
    }
    with GraphClient() as graph:
        first = build_from_graph(graph, facility, meanings)
        table = first["table"]
        second = build_from_graph(graph, facility, meanings)
        first_rows = [entry.as_dict() for entry in table.entries]
        second_rows = [entry.as_dict() for entry in second["table"].entries]
        receipt["repeat"] = {
            "entries_changed": sum(
                1 for a, b in zip(first_rows, second_rows, strict=False) if a != b
            )
            + abs(len(first_rows) - len(second_rows))
        }
        if receipt["repeat"]["entries_changed"]:
            raise RuntimeError("A repeated build changed an entry")
        receipt["summary"] = first["summary"]
        receipt["entries"] = first_rows
        receipt["conflicts"] = [item.as_dict() for item in table.conflicts]
        receipt["unestablished"] = list(table.unestablished)
        receipt["status"] = "dry-run" if dry_run else "complete"
    receipt["elapsed_seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facility", default="iter")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    receipt = build(
        facility=arguments.facility,
        receipt_path=arguments.receipt,
        dry_run=arguments.dry_run,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "distinct_codes": receipt["summary"]["distinct_codes"],
                "entries": receipt["summary"]["entries"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

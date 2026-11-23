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
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from iter_codex.discovery.signals.field_codes_graph import (
    build_from_graph,
    meanings_from_graph,
)
from iter_codex.discovery.signals.vocabulary_graph import (
    apply_assignment,
    assign_cohort,
    newly_decided,
    pilot_comparison,
    snapshot_catalogue,
    snapshot_cohort,
    unit_agreement,
    verify_assignment,
)
from iter_codex.graph.client import GraphClient


def sample_for_reading(items: list[dict], *, per_source: int = 10) -> dict:
    """Choose the reading sample deterministically, ten per credited source.

    The draw is a sorted, evenly spaced stride over each source's items, so the
    same assignment yields the same sample and its digest can be recorded before
    the reading begins. A source with fewer than ten items contributes all of
    them.
    """
    grouped: dict[str, list[dict]] = {}
    for item in items:
        grouped.setdefault(item["source"], []).append(item)
    sample: list[dict] = []
    counts: dict[str, int] = {}
    for source in sorted(grouped):
        members = sorted(grouped[source], key=lambda entry: entry["id"])
        if len(members) <= per_source:
            chosen = members
        else:
            step = (len(members) - 1) / (per_source - 1)
            seen: set[int] = set()
            chosen = []
            for position in range(per_source):
                index = round(position * step)
                if index not in seen:
                    seen.add(index)
                    chosen.append(members[index])
        counts[source] = len(chosen)
        sample.extend(chosen)
    encoded = json.dumps(sample, sort_keys=True, separators=(",", ":"))
    return {
        "per_source_limit": per_source,
        "counts": counts,
        "sample": sample,
        "digest": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def validate_readings(entries: list[dict], expected_ids: set[str]) -> dict:
    """Refuse a file that is not a reading of the sample it names.

    A reading writes one verdict and one reason per item, read by a person. A
    file that omits an item, carries an empty reason, repeats one reason for
    every item, or uses a verdict outside the three the reading admits is not a
    reading, however plausible its counts.
    """
    verdicts = {"right", "wrong", "unverifiable"}
    if not entries:
        raise ValueError("a reading file with no entries is not a reading")
    seen = set()
    reasons = []
    for entry in entries:
        signal_id = entry.get("id")
        if signal_id not in expected_ids:
            raise ValueError(f"reading names an item outside the sample: {signal_id!r}")
        if signal_id in seen:
            raise ValueError(f"reading repeats an item: {signal_id!r}")
        seen.add(signal_id)
        verdict = entry.get("verdict")
        if verdict not in verdicts:
            raise ValueError(
                f"reading verdict {verdict!r} is not one of {sorted(verdicts)}"
            )
        reason = str(entry.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"reading for {signal_id!r} has no written reason")
        reasons.append(reason)
    missing = expected_ids - seen
    if missing:
        raise ValueError(f"reading omits {len(missing)} sampled items")
    if len(set(reasons)) == 1:
        raise ValueError(
            "one identical reason for every item is a script, not a reading"
        )
    return {
        "items": len(entries),
        "right": sum(1 for entry in entries if entry["verdict"] == "right"),
        "wrong": sum(1 for entry in entries if entry["verdict"] == "wrong"),
        "unverifiable": sum(
            1 for entry in entries if entry["verdict"] == "unverifiable"
        ),
        "reason_variety": len(set(reasons)),
    }


def run_catalogue_assignment(
    *, facility: str, receipt_path: Path, dry_run: bool = False
) -> dict:
    """Assign the whole catalogue, with the field-code table deciding the role.

    The table is built from the graph's own sources and the meanings the graph
    holds, so the run contacts no archive or document host. The pilot's
    before-and-after comparison is read from the values the graph already
    stores, and every changed value is listed so it can be read by hand.
    """
    receipt: dict = {
        "facility": facility,
        "mode": "catalogue",
        "taken_at": datetime.now(UTC).isoformat(),
        "archive_contacted": False,
        "document_host_contacted": False,
    }
    started = datetime.now(UTC)
    with GraphClient() as graph:
        meanings = meanings_from_graph(graph, facility)
        built = build_from_graph(graph, facility, meanings)
        table = built["table"]
        receipt["field_code_table"] = built["summary"]
        signals, rows = snapshot_catalogue(graph, facility)
        receipt["catalogue_signals"] = len(signals)
        receipt["linked_rows"] = len(rows)
        first = assign_cohort(signals, rows, table=table)
        receipt["summary"] = first["summary"]
        receipt["changed"] = first["changed"]
        receipt["unit_agreement"] = unit_agreement(signals, first["ledger"])
        pilot = [signal for signal in signals if signal.get("plant_context_state")]
        receipt["pilot_signals"] = len(pilot)
        receipt["pilot_comparison"] = pilot_comparison(pilot, first["ledger"])
        newly_decided_roles = newly_decided(signals, first["ledger"], "role")
        receipt["newly_decided_roles"] = len(newly_decided_roles)
        receipt["readings_sample"] = sample_for_reading(newly_decided_roles)
        if dry_run:
            receipt["status"] = "dry-run"
        else:
            receipt["write"] = apply_assignment(graph, facility, first["pending"])
            receipt["verify"] = verify_assignment(
                graph, facility, first["ledger"], snapshot=snapshot_catalogue
            )
            if receipt["verify"]["mismatched"]:
                raise RuntimeError("Stored vocabulary does not match its assignment")
            second_signals, second_rows = snapshot_catalogue(graph, facility)
            second = assign_cohort(second_signals, second_rows, table=table)
            receipt["repeat"] = {"changed": second["changed"]}
            if second["changed"]:
                raise RuntimeError("A repeated assignment changed a signal")
            receipt["status"] = "complete"
    receipt["elapsed_seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


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
    parser.add_argument(
        "--catalogue",
        action="store_true",
        help="assign every catalogued signal, with the field-code table deciding the role",
    )
    arguments = parser.parse_args(argv)
    run = run_catalogue_assignment if arguments.catalogue else run_assignment
    receipt = run(
        facility=arguments.facility,
        receipt_path=arguments.receipt,
        dry_run=arguments.dry_run,
    )
    keys = (
        ("status", "catalogue_signals")
        if arguments.catalogue
        else ("status", "cohort_signals")
    )
    print(json.dumps({key: receipt[key] for key in keys}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Resumable metadata-only pulse catalogue snapshots and graph imports.

Run with ``uv run python -m iter_codex.remote.pulse_catalogue --help``.
Real identities and source descriptions belong only in the private snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iter_codex.graph.models import ArchiveEndpoint, Pulse, PulseCategory
from iter_codex.remote.catalogue_scan import write_receipt
from iter_codex.remote.codac_uda import ArchiveReader


def collect_pulses(
    reader: ArchiveReader, path: Path, *, pulse_limit: int = 100000
) -> dict:
    """Resume completed category listings without treating partial work as complete."""
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "archive": reader.archive,
            "facility": reader.facility,
            "categories": reader.pulse_categories(),
            "listings": {},
            "complete": False,
        }
    )
    if state["archive"] != reader.archive or state["facility"] != reader.facility:
        raise ValueError("pulse snapshot belongs to a different archive")
    state["complete"] = False
    write_receipt(path, state)
    for category in state["categories"]:
        if category in state["listings"]:
            continue
        pulses = reader.pulses(category, limit=pulse_limit)
        category_id = f"{reader.archive}:{category}"
        if any(category_id not in (pulse.categories or []) for pulse in pulses):
            raise ValueError("archive returned a pulse outside the requested category")
        state["listings"][category] = [
            pulse.model_dump(mode="json") for pulse in pulses
        ]
        write_receipt(path, state)
    state["complete"] = True
    write_receipt(path, state)
    return state


def snapshot_models(state: dict) -> tuple[list[PulseCategory], list[Pulse]]:
    """Validate completeness before allowing any graph writes."""
    if not state.get("complete") or set(state["listings"]) != set(state["categories"]):
        raise ValueError("pulse snapshot is incomplete")
    categories = [
        PulseCategory(
            id=f"{state['archive']}:{name}",
            facility_id=state["facility"],
            archive=state["archive"],
            name=name,
        )
        for name in state["categories"]
    ]
    unique = {}
    for category, listing in state["listings"].items():
        category_id = f"{state['archive']}:{category}"
        for values in listing:
            pulse = Pulse.model_validate(values)
            if category_id not in (pulse.categories or []):
                raise ValueError(
                    "snapshot pulse does not belong to its listed category"
                )
            if pulse.id in unique and unique[pulse.id] != pulse:
                raise ValueError("archive pulse listings disagree")
            unique[pulse.id] = pulse
    return categories, list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--facility", default="iter")
    parser.add_argument("--role", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--import-graph", action="store_true")
    parser.add_argument(
        "--kind-rules", type=Path, help="Private YAML evidence rules for category kinds"
    )
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(args.config.read_text())
    if "data_systems" not in config:
        config = {"data_systems": {"codac_uda": config}}
    reader = ArchiveReader(args.facility, args.role, config=config)
    state = collect_pulses(reader, args.snapshot)
    categories, pulses = snapshot_models(state)
    from iter_codex.discovery.signals.archive_pulses import (
        PulseKindRule,
        classify_categories,
        import_pulses,
        pulse_distribution,
        pulse_metadata_counts,
    )

    if args.kind_rules:
        rules = [
            PulseKindRule.model_validate(rule)
            for rule in yaml.safe_load(args.kind_rules.read_text())
        ]
        categories = classify_categories(categories, pulses, rules)
    receipt = pulse_distribution(categories, pulses)
    if args.import_graph:
        from iter_codex.graph.client import GraphClient

        archive = ArchiveEndpoint(
            id=reader.archive,
            facility_id=reader.facility,
            data_system="codac_uda",
            role=reader.role,
            excluded=False,
        )
        with GraphClient() as graph:
            receipt = import_pulses(graph, archive, categories, pulses)
    receipt.update(pulse_metadata_counts(categories, pulses))
    write_receipt(args.receipt, receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

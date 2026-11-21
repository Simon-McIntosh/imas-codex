"""Measure a pilot graph population separately from the private metadata scan."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from iter_codex.remote.catalogue_scan import write_receipt


def measure_population(facility: str, output: Path) -> dict:
    """Read an existing nonempty pilot, without a graph mutation statement."""
    from iter_codex.graph.client import GraphClient

    with GraphClient() as graph:
        rows = graph.query(
            "MATCH (s:FacilitySignal {facility_id:$facility}) "
            "RETURN count(s) AS signals,count(s.facility_id) AS with_facility, "
            "count(DISTINCT s.id) AS identities",
            facility=facility,
        )
    if (
        len(rows) != 1
        or not rows[0]["signals"]
        or rows[0]["with_facility"] != rows[0]["signals"]
    ):
        raise RuntimeError("pilot count requires a nonempty facility-property control")
    receipt = {
        "facility": facility,
        **rows[0],
        "observed_at": datetime.now(UTC).isoformat(),
    }
    write_receipt(output, receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facility", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    measure_population(args.facility, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

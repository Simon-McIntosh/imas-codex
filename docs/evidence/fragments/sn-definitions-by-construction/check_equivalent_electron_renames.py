"""Check equivalent-electron rename evidence against the live graph, read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iter_codex.graph.client import GraphClient
from iter_codex.standard_names.source_reads import (
    count_live_names_without_source,
    read_source_paths_by_name,
)

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "equivalent-electron-renames.json"

READ_IDENTITIES = """
UNWIND $ids AS wanted
OPTIONAL MATCH (sn:StandardName {id: wanted})
RETURN wanted AS requested_id,
       sn.id AS id,
       sn.description AS description,
       sn.unit AS unit,
       sn.status AS status,
       sn.name_stage AS name_stage,
       sn.docs_stage AS docs_stage,
       sn.reviewer_score_name AS score_name,
       sn.reviewer_score_docs AS score_docs
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument(
        "--expected-count",
        type=int,
        help="override the recorded population size for a negative control",
    )
    args = parser.parse_args()

    try:
        document = json.loads(args.evidence.read_text())
        records = document["names"]
        expected_count = (
            document["population"]["count"]
            if args.expected_count is None
            else args.expected_count
        )
        failures: list[str] = []

        if len(records) != expected_count:
            failures.append(
                f"population: evidence has {len(records)} rows, expected {expected_count}"
            )

        live_ids = [row["live_identity"]["id"] for row in records]
        if len(live_ids) != len(set(live_ids)):
            failures.append("population: duplicate live identities in evidence")

        spend_total = round(sum(float(row["cost"]["spend_usd"]) for row in records), 6)
        if abs(spend_total - float(document["spend_total_usd"])) > 1e-6:
            failures.append(
                f"spend: rows total {spend_total:.6f}, "
                f"evidence says {document['spend_total_usd']}"
            )
        if spend_total > float(document["spend_ceiling_usd"]):
            failures.append(
                f"spend: USD {spend_total:.6f} exceeds the recorded ceiling"
            )

        with GraphClient() as graph:
            identity_rows = graph.query(READ_IDENTITIES, ids=live_ids) or []
            actual = {row["requested_id"]: row for row in identity_rows}
            sources = read_source_paths_by_name(graph, live_ids)

            for row in records:
                old_id = row["old_identity"]["id"]
                expected = row["live_identity"]["id"]
                found = actual.get(expected)
                new_identity = row.get("new_identity")
                if new_identity is not None and new_identity.get("id") != expected:
                    failures.append(
                        f"{old_id}: new identity does not match live identity {expected!r}"
                    )
                if new_identity is None and old_id != expected:
                    failures.append(
                        f"{old_id}: unchanged identity points to {expected!r}"
                    )
                if not found or found.get("id") != expected:
                    failures.append(f"{old_id}: live identity {expected!r} is absent")
                    continue

                after = row["after"]
                for field in ("description", "unit"):
                    if found.get(field) != after.get(field):
                        failures.append(
                            f"{old_id}: {field} is {found.get(field)!r}, "
                            f"evidence says {after.get(field)!r}"
                        )
                for field, key in (
                    ("name_stage", "name_stage"),
                    ("docs_stage", "docs_stage"),
                ):
                    if found.get(field) != after.get(key):
                        failures.append(
                            f"{old_id}: {field} is {found.get(field)!r}, "
                            f"evidence says {after.get(key)!r}"
                        )

                for field, key in (
                    ("score_name", "score_name"),
                    ("score_docs", "score_docs"),
                ):
                    wanted = after.get(key)
                    observed = found.get(field)
                    if wanted is None and observed is None:
                        continue
                    if (
                        wanted is None
                        or observed is None
                        or abs(float(wanted) - float(observed)) > 1e-6
                    ):
                        failures.append(
                            f"{old_id}: {field} is {observed!r}, evidence says {wanted!r}"
                        )

                wanted_sources = sorted(after.get("source_bindings", []))
                actual_sources = sorted(sources.get(expected, []))
                if wanted_sources != actual_sources:
                    failures.append(
                        f"{old_id}: sources are {actual_sources!r}, "
                        f"evidence says {wanted_sources!r}"
                    )
                if new_identity is not None:
                    original_sources = sorted(
                        row["old_identity"].get("source_bindings", [])
                    )
                    if original_sources != actual_sources:
                        failures.append(
                            f"{old_id}: rename changed source bindings from "
                            f"{original_sources!r} to {actual_sources!r}"
                        )
                    if not found.get("id").endswith("equivalent_electron") and (
                        "equivalent_electron" not in (found.get("id") or "")
                    ):
                        failures.append(
                            f"{old_id}: live identity {found.get('id')!r} does not "
                            "carry equivalent_electron"
                        )
                    if found.get("name_stage") != "accepted" and not row.get("held_by"):
                        failures.append(
                            f"{old_id}: live identity {found.get('id')!r} is at "
                            f"name_stage {found.get('name_stage')!r} and the record "
                            "names no check holding it"
                        )
                elif row.get("outcome") in {"collision", "declined_source"} and not (
                    row.get("left_reason")
                ):
                    failures.append(f"{old_id}: left in place with no recorded reason")

            sourceless = count_live_names_without_source(graph)["count"]
            expected_sourceless = document["sourceless"]["count"]
            if sourceless != expected_sourceless:
                failures.append(
                    f"sourceless: graph has {sourceless}, "
                    f"evidence says {expected_sourceless}"
                )
            if sourceless > int(document["sourceless"]["allowance"]):
                failures.append(
                    f"sourceless: graph has {sourceless}, above the "
                    f"{document['sourceless']['allowance']} allowance"
                )

        for failure in failures:
            print(f"FAIL {failure}")
        print(
            f"checked {len(records)} rows, {len(live_ids)} live identities, "
            f"{sourceless} live names without a source"
        )
        code = 1 if failures else 0
        print(f"EXIT={code}")
        return code
    except Exception as error:  # an unreadable receipt is a failed check
        print(f"FAIL {type(error).__name__}: {error}")
        print("EXIT=1")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

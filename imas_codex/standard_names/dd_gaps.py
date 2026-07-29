"""Data Dictionary defect reporting and registry provenance.

``DDGap`` nodes are evidence. Creating one never changes source eligibility,
unit resolution, attachment policy, or any other pipeline behavior. Existing
curated registries remain the enforcement authority until a human triages a
report and changes the relevant registry through its normal workflow.
"""

from __future__ import annotations

import fnmatch
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import DDGap, DDGapKind, DDGapStatus
from imas_codex.units.dd_unit_exceptions import load_exceptions


def _enum_value(value: str, enum_type: type) -> str:
    """Validate and return one generated-enum value."""
    try:
        return enum_type(value).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(
            f"invalid {enum_type.__name__} {value!r}; choose {allowed}"
        ) from exc


def _gap_id(path: str, kind: str) -> str:
    return f"dd_gap:{path}:{kind}"


def _prepare_reports(reports: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Validate reports and aggregate their node/edge batches."""
    observed_at = datetime.now(UTC).isoformat()
    nodes: dict[str, dict[str, Any]] = {}
    relationships: dict[tuple[str, str], dict[str, Any]] = {}

    for report in reports:
        path = str(report.get("path") or "").strip()
        reason = str(report.get("reason") or "").strip()
        if not path:
            raise ValueError("DD-gap path must be non-empty")
        if not reason:
            raise ValueError(f"DD-gap report for {path!r} requires a reason")

        kind = _enum_value(str(report.get("kind") or ""), DDGapKind)
        status = _enum_value(
            str(report.get("status") or DDGapStatus.flagged.value),
            DDGapStatus,
        )
        gap_id = _gap_id(path, kind)
        node = nodes.setdefault(
            gap_id,
            DDGap(
                id=gap_id,
                path=path,
                kind=kind,
                status=status,
                example_count=0,
                upstream_url=report.get("upstream_url"),
                resolved_dd_version=report.get("resolved_dd_version"),
                registry_backend=report.get("registry_backend"),
                affected_path_count=report.get("affected_path_count"),
            ).model_dump(mode="json"),
        )
        node["example_count"] += 1

        source_path = str(report.get("source_path") or path).strip()
        relationships[(source_path, gap_id)] = {
            "source_path": source_path,
            "gap_id": gap_id,
            "reason": reason,
            "reporter": str(report.get("reporter") or "human"),
            "observed_at": str(report.get("observed_at") or observed_at),
        }

    return list(nodes.values()), list(relationships.values())


@retry_on_deadlock()
def write_dd_gaps(
    reports: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Persist human or audit DD-defect reports without changing behavior."""
    if not reports:
        return {"reported": 0, "relationships": 0, "ids": [], "dry_run": dry_run}

    nodes, relationships = _prepare_reports(reports)
    result = {
        "reported": len(nodes),
        "relationships": len(relationships),
        "ids": [node["id"] for node in nodes],
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (gap:DDGap {id: b.id})
            ON CREATE SET gap.first_seen_at = datetime(),
                          gap.example_count = 0
            SET gap.path = b.path,
                gap.kind = b.kind,
                gap.example_count =
                    coalesce(gap.example_count, 0) + b.example_count,
                gap.last_seen_at = datetime(),
                gap.status = CASE
                    WHEN gap.status IS NULL OR gap.status = 'flagged'
                    THEN b.status
                    ELSE gap.status
                END,
                gap.upstream_url =
                    coalesce(b.upstream_url, gap.upstream_url),
                gap.resolved_dd_version =
                    coalesce(b.resolved_dd_version, gap.resolved_dd_version),
                gap.registry_backend =
                    coalesce(b.registry_backend, gap.registry_backend),
                gap.affected_path_count =
                    coalesce(b.affected_path_count, gap.affected_path_count)
            """,
            batch=nodes,
        )
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (node:IMASNode {id: b.source_path})
            MATCH (gap:DDGap {id: b.gap_id})
            MERGE (node)-[report:HAS_DD_GAP]->(gap)
            SET report.reason = b.reason,
                report.reporter = b.reporter,
                report.observed_at = datetime(b.observed_at)
            """,
            batch=relationships,
        )
    return result


def _registry_inventory(
    current_paths: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build idempotent DDGap nodes and path edges from the unit registry."""
    now = datetime.now(UTC).isoformat()
    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    entries = load_exceptions()["dd_unit_bugs"]
    for entry in entries:
        pattern = str(entry["path"])
        kind = (
            DDGapKind.self_contradiction.value
            if entry.get("correct_in_graph")
            else DDGapKind.unit_defect.value
        )
        matches = [path for path in current_paths if fnmatch.fnmatchcase(path, pattern)]
        gap_id = _gap_id(pattern, kind)
        nodes.append(
            {
                "id": gap_id,
                "path": pattern,
                "kind": kind,
                "status": DDGapStatus.registered_exception.value,
                "registry_backend": "dd_unit_exceptions",
                "affected_path_count": len(matches),
                "triaged_at": now,
                "upstream_url": None,
            }
        )
        relationships.extend(
            {
                "source_path": path,
                "gap_id": gap_id,
                "reason": str(entry["reason"]),
                "reporter": "registry_backfill",
                "observed_at": now,
            }
            for path in matches
        )

    for entry in entries:
        upstream_url = str(entry.get("upstream_url") or "").strip()
        if not upstream_url:
            continue
        pattern = str(entry["path"])
        kind = _enum_value(str(entry["upstream_kind"]), DDGapKind)
        matches = [path for path in current_paths if fnmatch.fnmatchcase(path, pattern)]
        gap_id = _gap_id(pattern, kind)
        nodes.append(
            {
                "id": gap_id,
                "path": pattern,
                "kind": kind,
                "status": DDGapStatus.upstream_issue.value,
                "registry_backend": None,
                "affected_path_count": len(matches),
                "triaged_at": now,
                "upstream_url": upstream_url,
            }
        )
        relationships.extend(
            {
                "source_path": path,
                "gap_id": gap_id,
                "reason": str(entry["reason"]),
                "reporter": "registry_backfill",
                "observed_at": now,
            }
            for path in matches
        )
    return nodes, relationships


@retry_on_deadlock()
def sync_dd_unit_exception_gaps(*, dry_run: bool = False) -> dict[str, Any]:
    """Mirror the curated unit-exception record into provenance-only DD gaps."""
    with GraphClient() as gc:
        current_paths = [
            str(row["id"])
            for row in gc.query("MATCH (node:IMASNode) RETURN node.id AS id")
        ]
        nodes, relationships = _registry_inventory(current_paths)
        result = {
            "registry_entries": len(load_exceptions()["dd_unit_bugs"]),
            "reported": len(nodes),
            "relationships": len(relationships),
            "matched_paths": len({item["source_path"] for item in relationships}),
            "dry_run": dry_run,
        }
        if dry_run:
            return result

        gc.query(
            """
            UNWIND $batch AS b
            MERGE (gap:DDGap {id: b.id})
            ON CREATE SET gap.first_seen_at = datetime(),
                          gap.last_seen_at = datetime(),
                          gap.triaged_at = datetime(b.triaged_at),
                          gap.example_count = 1
            SET gap.path = b.path,
                gap.kind = b.kind,
                gap.status = b.status,
                gap.registry_backend = b.registry_backend,
                gap.affected_path_count = b.affected_path_count,
                gap.upstream_url = b.upstream_url
            """,
            batch=nodes,
        )
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (node:IMASNode {id: b.source_path})
            MATCH (gap:DDGap {id: b.gap_id})
            MERGE (node)-[report:HAS_DD_GAP]->(gap)
            ON CREATE SET report.observed_at = datetime(b.observed_at)
            SET report.reason = b.reason,
                report.reporter = b.reporter
            """,
            batch=relationships,
        )
        return result


def get_dd_gap_stats(gc: GraphClient | None = None) -> dict[str, Any]:
    """Return status and kind counts for the CLI status surface."""
    if gc is None:
        with GraphClient() as owned:
            return get_dd_gap_stats(owned)
    rows = list(
        gc.query(
            """
            MATCH (gap:DDGap)
            RETURN gap.status AS status, gap.kind AS kind, count(*) AS count
            ORDER BY status, kind
            """
        )
    )

    by_status: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    for row in rows:
        by_status[str(row["status"])] += int(row["count"])
        by_kind[str(row["kind"])] += int(row["count"])
    return {
        "total": sum(by_status.values()),
        "by_status": dict(by_status),
        "by_kind": dict(by_kind),
    }

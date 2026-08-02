"""Data Dictionary defect evidence, triage, and release reconciliation.

``DDGap`` nodes are observational. Reporting evidence never changes source
eligibility, unit resolution, attachment policy, or any other pipeline behavior.
Human dispositions use a separate compare-and-set API, and the curated unit
registry remains the sole unit-enforcement authority.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import (
    DDGapEvidenceRule,
    DDGapKind,
    DDGapStatus,
)
from imas_codex.units.dd_unit_exceptions import canonical_or_none, load_exceptions


class DDGapTransitionConflict(RuntimeError):
    """The fact no longer has the lifecycle state expected by the caller."""


class DDGapRegistrySyncConflict(RuntimeError):
    """Registry identity reconciliation is ambiguous or changed during apply."""


_DISPOSITION_FIELDS = frozenset(
    {
        "status",
        "registry_backend",
        "upstream_url",
        "resolved_dd_version",
        "triaged_at",
        "triage_actor",
        "triage_reason",
        "status_changed_at",
        "status_changed_by",
        "status_change_reason",
        "validation_evidence",
    }
)

_REPORT_FIELDS = frozenset(
    {
        "path",
        "source_path",
        "kind",
        "reason",
        "reporter",
        "observed_at",
        "observed_dd_version",
        "observed_value",
        "expected_value",
        "evidence_rule",
        "reference_path",
        "reference_value",
    }
)

_ALLOWED_TRANSITIONS = {
    DDGapStatus.flagged.value: frozenset(
        {
            DDGapStatus.triaged.value,
            DDGapStatus.registered_exception.value,
            DDGapStatus.upstream_issue.value,
            DDGapStatus.rejected.value,
        }
    ),
    DDGapStatus.triaged.value: frozenset(
        {
            DDGapStatus.registered_exception.value,
            DDGapStatus.upstream_issue.value,
            DDGapStatus.rejected.value,
            DDGapStatus.resolved_upstream.value,
        }
    ),
    DDGapStatus.registered_exception.value: frozenset(
        {DDGapStatus.resolved_upstream.value}
    ),
    DDGapStatus.upstream_issue.value: frozenset({DDGapStatus.resolved_upstream.value}),
}

_RECONCILABLE_STATUSES = (
    DDGapStatus.triaged.value,
    DDGapStatus.registered_exception.value,
    DDGapStatus.upstream_issue.value,
)

_RELEASE_VALIDATORS = {
    (
        DDGapKind.unit_defect.value,
        DDGapEvidenceRule.unit_equals_expected.value,
    ): "unit",
    (
        DDGapKind.self_contradiction.value,
        DDGapEvidenceRule.unit_equals_expected.value,
    ): "unit",
}


def _enum_value(value: str, enum_type: type) -> str:
    """Validate and return one generated-enum value."""
    try:
        return enum_type(value).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(
            f"invalid {enum_type.__name__} {value!r}; choose {allowed}"
        ) from exc


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _observation_time(value: Any) -> str:
    text = _optional_text(value) or datetime.now(UTC).isoformat()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid DD-gap observed_at {text!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("DD-gap observed_at must include a UTC offset")
    return text


def _gap_id(path: str, kind: str) -> str:
    return f"dd_gap:{path}:{kind}"


def _validate_gap_id(gap_id: str) -> str:
    """Validate one exact DD-gap identity without querying the graph."""
    clean_id = _optional_text(gap_id)
    prefix = "dd_gap:"
    if not clean_id or not clean_id.startswith(prefix):
        raise ValueError("DD-gap id must use the exact 'dd_gap:{path}:{kind}' form")
    path, separator, kind = clean_id.removeprefix(prefix).rpartition(":")
    if not separator or not path:
        raise ValueError("DD-gap id must use the exact 'dd_gap:{path}:{kind}' form")
    _enum_value(kind, DDGapKind)
    return clean_id


def _filter_sequence(
    values: Sequence[str] | None,
    *,
    label: str,
    enum_type: type | None = None,
    exact_paths: bool = False,
) -> list[str]:
    """Normalize one explicit read filter while rejecting scalar strings."""
    if values is None:
        return []
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence, not a bare string")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{label} must contain strings")
        clean = _optional_text(value)
        if not clean:
            raise ValueError(f"{label} cannot contain empty values")
        if exact_paths and ("*" in clean or "?" in clean):
            raise ValueError(f"{label} must contain exact paths, not patterns")
        normalized.append(_enum_value(clean, enum_type) if enum_type else clean)
    return sorted(set(normalized))


def _evidence_snapshot(fact: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize every mutable field that defines one reviewed evidence set."""
    raw_observation_ids = fact.get("observation_ids")
    if raw_observation_ids is None:
        raw_observation_ids = [item["id"] for item in (fact.get("observations") or [])]
    return {
        "path": _optional_text(fact.get("path")) or "",
        "kind": _optional_text(fact.get("kind")) or "",
        "observed_dd_version": _optional_text(fact.get("observed_dd_version")) or "",
        "observed_value": _optional_text(fact.get("observed_value")) or "",
        "expected_value": _optional_text(fact.get("expected_value")) or "",
        "evidence_rule": _optional_text(fact.get("evidence_rule")) or "",
        "reference_path": _optional_text(fact.get("reference_path")) or "",
        "reference_value": _optional_text(fact.get("reference_value")) or "",
        "registry_backend": _optional_text(fact.get("registry_backend")) or "",
        "source_paths": sorted(str(item) for item in (fact.get("source_paths") or [])),
        "observation_ids": sorted(str(item) for item in raw_observation_ids),
        "example_count": int(fact.get("example_count") or 0),
        "first_seen_at": fact.get("first_seen_at"),
        "last_seen_at": fact.get("last_seen_at"),
    }


def _evidence_token(fact: Mapping[str, Any]) -> str:
    """Return a stable token for the exact evidence snapshot an operator saw."""
    snapshot = _evidence_snapshot(fact)
    serializable = {
        **snapshot,
        "first_seen_at": str(snapshot["first_seen_at"] or ""),
        "last_seen_at": str(snapshot["last_seen_at"] or ""),
    }
    digest = hashlib.sha256(
        json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"dd-gap-evidence:{digest}"


def _observation_id(payload: Mapping[str, Any]) -> str:
    identity = {
        key: payload.get(key)
        for key in (
            "gap_id",
            "source_path",
            "reason",
            "reporter",
            "observed_dd_version",
            "observed_value",
            "expected_value",
            "evidence_rule",
            "reference_path",
            "reference_value",
        )
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"dd_gap_observation:{digest}"


def _prepare_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate flag-only evidence and build content-addressed observations."""
    batch: list[dict[str, Any]] = []
    for report in reports:
        forbidden = sorted(
            _DISPOSITION_FIELDS.intersection(report) | (set(report) - _REPORT_FIELDS)
        )
        if forbidden:
            fields = ", ".join(forbidden)
            raise ValueError(
                "DD-gap reporting is evidence-only; disposition fields are "
                f"rejected: {fields}"
            )

        path = _optional_text(report.get("path"))
        reason = _optional_text(report.get("reason"))
        if not path:
            raise ValueError("DD-gap path must be non-empty")
        if not reason:
            raise ValueError(f"DD-gap report for {path!r} requires a reason")

        kind = _enum_value(str(report.get("kind") or ""), DDGapKind)
        source_path = _optional_text(report.get("source_path")) or path
        reporter = _optional_text(report.get("reporter")) or "automated"
        evidence_rule = _optional_text(report.get("evidence_rule"))
        if evidence_rule:
            evidence_rule = _enum_value(evidence_rule, DDGapEvidenceRule)
        reference_path = _optional_text(report.get("reference_path"))
        reference_value = _optional_text(report.get("reference_value"))
        if bool(reference_path) != bool(reference_value):
            raise ValueError(
                "DD-gap reference_path and reference_value must be supplied together"
            )

        item: dict[str, Any] = {
            "gap_id": _gap_id(path, kind),
            "path": path,
            "kind": kind,
            "source_path": source_path,
            "reason": reason,
            "reporter": reporter,
            "observed_at": _observation_time(report.get("observed_at")),
            "observed_dd_version": _optional_text(report.get("observed_dd_version")),
            "observed_value": _optional_text(report.get("observed_value")),
            "expected_value": _optional_text(report.get("expected_value")),
            "evidence_rule": evidence_rule,
            "reference_path": reference_path,
            "reference_value": reference_value,
        }
        item["observation_id"] = _observation_id(item)
        batch.append(item)
    return batch


def _validate_exact_paths(gc: GraphClient, paths: set[str]) -> None:
    rows = gc.query(
        """
        UNWIND $paths AS path
        MATCH (node:IMASNode {id: path})
        RETURN node.id AS id
        """,
        paths=sorted(paths),
    )
    found = {str(row["id"]) for row in rows}
    missing = sorted(paths - found)
    if missing:
        raise ValueError(
            "DD-gap evidence references missing exact IMASNode path(s): "
            + ", ".join(missing)
        )


_WRITE_EVIDENCE_QUERY = """
UNWIND $batch AS b
MATCH (node:IMASNode {id: b.source_path})
MERGE (gap:DDGap {id: b.gap_id})
ON CREATE SET gap.first_seen_at = datetime(b.observed_at),
              gap.status = 'flagged',
              gap.example_count = 0
SET gap.path = b.path,
    gap.kind = b.kind,
    gap.first_seen_at = CASE
        WHEN gap.first_seen_at IS NULL
          OR datetime(b.observed_at) < gap.first_seen_at
        THEN datetime(b.observed_at) ELSE gap.first_seen_at END,
    gap.last_seen_at = CASE
        WHEN gap.last_seen_at IS NULL
          OR datetime(b.observed_at) > gap.last_seen_at
        THEN datetime(b.observed_at) ELSE gap.last_seen_at END,
    gap.observed_dd_version = coalesce(b.observed_dd_version,
                                       gap.observed_dd_version),
    gap.observed_value = coalesce(b.observed_value, gap.observed_value),
    gap.expected_value = coalesce(b.expected_value, gap.expected_value),
    gap.evidence_rule = coalesce(b.evidence_rule, gap.evidence_rule),
    gap.reference_path = coalesce(b.reference_path, gap.reference_path),
    gap.reference_value = coalesce(b.reference_value, gap.reference_value)
MERGE (observation:DDGapObservation {id: b.observation_id})
ON CREATE SET observation.dd_gap_id = b.gap_id,
              observation.source_path = b.source_path,
              observation.reason = b.reason,
              observation.reporter = b.reporter,
              observation.observed_dd_version = b.observed_dd_version,
              observation.observed_value = b.observed_value,
              observation.expected_value = b.expected_value,
              observation.evidence_rule = b.evidence_rule,
              observation.reference_path = b.reference_path,
              observation.reference_value = b.reference_value,
              observation.first_observed_at = datetime(b.observed_at)
SET observation.first_observed_at = CASE
        WHEN datetime(b.observed_at) < observation.first_observed_at
        THEN datetime(b.observed_at) ELSE observation.first_observed_at END,
    observation.last_observed_at = CASE
        WHEN observation.last_observed_at IS NULL
          OR datetime(b.observed_at) > observation.last_observed_at
        THEN datetime(b.observed_at) ELSE observation.last_observed_at END
MERGE (gap)-[:HAS_OBSERVATION]->(observation)
MERGE (node)-[report:HAS_DD_GAP]->(gap)
ON CREATE SET report.reason = b.reason,
              report.reporter = b.reporter,
              report.first_observed_at = datetime(b.observed_at)
SET report.first_observed_at = CASE
        WHEN datetime(b.observed_at) < report.first_observed_at
        THEN datetime(b.observed_at) ELSE report.first_observed_at END,
    report.last_observed_at = CASE
        WHEN report.last_observed_at IS NULL
          OR datetime(b.observed_at) > report.last_observed_at
        THEN datetime(b.observed_at) ELSE report.last_observed_at END
WITH DISTINCT gap, report, observation
CALL {
    WITH gap
    MATCH (gap)-[:HAS_OBSERVATION]->(evidence:DDGapObservation)
    RETURN count(DISTINCT evidence) AS evidence_count
}
SET gap.example_count = evidence_count
WITH collect(DISTINCT gap.id) AS ids,
     count(DISTINCT report) AS relationships,
     count(DISTINCT observation) AS observations
RETURN size(ids) AS reported, relationships, observations, ids
"""


@retry_on_deadlock()
def write_dd_gaps(
    reports: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Persist flag-only DD evidence without accepting lifecycle dispositions."""
    if not reports:
        return {
            "reported": 0,
            "relationships": 0,
            "observations": 0,
            "ids": [],
            "dry_run": dry_run,
        }

    batch = _prepare_reports(reports)
    ids = sorted({str(item["gap_id"]) for item in batch})
    exact_paths = {str(item["source_path"]) for item in batch}
    exact_paths.update(
        str(item["reference_path"]) for item in batch if item.get("reference_path")
    )

    with GraphClient() as gc:
        _validate_exact_paths(gc, exact_paths)
        if dry_run:
            return {
                "reported": len(ids),
                "relationships": len(
                    {(item["source_path"], item["gap_id"]) for item in batch}
                ),
                "observations": len({str(item["observation_id"]) for item in batch}),
                "ids": ids,
                "dry_run": True,
            }

        rows = gc.query(_WRITE_EVIDENCE_QUERY, batch=batch)
        row = rows[0] if rows else {}
        return {
            "reported": int(row.get("reported", 0)),
            "relationships": int(row.get("relationships", 0)),
            "observations": int(row.get("observations", 0)),
            "ids": sorted(str(value) for value in row.get("ids", [])),
            "dry_run": False,
        }


def _registry_inventory(
    current_paths: list[str],
    observed_dd_version: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build DDGap facts and structured observations from the unit registry."""
    now = datetime.now(UTC).isoformat()
    nodes: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

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
                "observed_dd_version": observed_dd_version,
                "observed_value": str(entry["dd_unit"]),
                "expected_value": str(entry["correct_unit"]),
                "evidence_rule": DDGapEvidenceRule.unit_equals_expected.value,
                "upstream_url": None,
            }
        )
        for path in matches:
            observation = {
                "source_path": path,
                "gap_id": gap_id,
                "reason": str(entry["reason"]),
                "reporter": "registry_backfill",
                "observed_at": now,
                "observed_dd_version": observed_dd_version,
                "observed_value": str(entry["dd_unit"]),
                "expected_value": str(entry["correct_unit"]),
                "evidence_rule": DDGapEvidenceRule.unit_equals_expected.value,
                "reference_path": None,
                "reference_value": None,
            }
            observation["observation_id"] = _observation_id(observation)
            observations.append(observation)

    for entry in entries:
        upstream_url = _optional_text(entry.get("upstream_url"))
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
                "observed_dd_version": observed_dd_version,
                "observed_value": str(entry["dd_unit"]),
                "expected_value": str(entry["correct_unit"]),
                "evidence_rule": DDGapEvidenceRule.unit_equals_expected.value,
                "upstream_url": upstream_url,
            }
        )
        for path in matches:
            observation = {
                "source_path": path,
                "gap_id": gap_id,
                "reason": str(entry["reason"]),
                "reporter": "registry_backfill",
                "observed_at": now,
                "observed_dd_version": observed_dd_version,
                "observed_value": str(entry["dd_unit"]),
                "expected_value": str(entry["correct_unit"]),
                "evidence_rule": DDGapEvidenceRule.unit_equals_expected.value,
                "reference_path": None,
                "reference_value": None,
            }
            observation["observation_id"] = _observation_id(observation)
            observations.append(observation)
    return nodes, observations


_SYNC_REGISTRY_QUERY = """
UNWIND $nodes AS b
MERGE (gap:DDGap {id: b.id})
ON CREATE SET gap.first_seen_at = datetime(),
              gap.last_seen_at = datetime(),
              gap.triaged_at = datetime(b.triaged_at),
              gap.status = b.status,
              gap.example_count = 0
SET gap.path = b.path,
    gap.kind = b.kind,
    gap.registry_backend = coalesce(gap.registry_backend, b.registry_backend),
    gap.affected_path_count = b.affected_path_count,
    gap.upstream_url = coalesce(gap.upstream_url, b.upstream_url),
    gap.observed_dd_version = coalesce(gap.observed_dd_version,
                                       b.observed_dd_version),
    gap.observed_value = coalesce(gap.observed_value, b.observed_value),
    gap.expected_value = coalesce(gap.expected_value, b.expected_value),
    gap.evidence_rule = coalesce(gap.evidence_rule, b.evidence_rule)
WITH collect(DISTINCT gap.id) AS node_ids
CALL {
    WITH $observations AS observations
    UNWIND observations AS b
    MATCH (node:IMASNode {id: b.source_path})
    MATCH (gap:DDGap {id: b.gap_id})
    MERGE (observation:DDGapObservation {id: b.observation_id})
    ON CREATE SET observation.dd_gap_id = b.gap_id,
                  observation.source_path = b.source_path,
                  observation.reason = b.reason,
                  observation.reporter = b.reporter,
                  observation.observed_dd_version = b.observed_dd_version,
                  observation.observed_value = b.observed_value,
                  observation.expected_value = b.expected_value,
                  observation.evidence_rule = b.evidence_rule,
                  observation.reference_path = b.reference_path,
                  observation.reference_value = b.reference_value,
                  observation.first_observed_at = datetime(b.observed_at),
                  observation.last_observed_at = datetime(b.observed_at)
    MERGE (gap)-[:HAS_OBSERVATION]->(observation)
    MERGE (node)-[report:HAS_DD_GAP]->(gap)
    ON CREATE SET report.reason = b.reason,
                  report.reporter = b.reporter,
                  report.first_observed_at = datetime(b.observed_at),
                  report.last_observed_at = datetime(b.observed_at)
    WITH DISTINCT gap, report, observation
    CALL {
        WITH gap
        MATCH (gap)-[:HAS_OBSERVATION]->(evidence:DDGapObservation)
        RETURN count(DISTINCT evidence) AS evidence_count
    }
    SET gap.example_count = evidence_count
    RETURN count(DISTINCT report) AS relationships,
           count(DISTINCT observation) AS observation_count
}
RETURN size(node_ids) AS reported, relationships,
       observation_count AS observations, node_ids AS ids
"""


_REGISTRY_FACTS_QUERY = """
MATCH (gap:DDGap)
OPTIONAL MATCH (node:IMASNode)-[:HAS_DD_GAP]->(gap)
WITH gap, collect(DISTINCT node.id) AS source_paths
OPTIONAL MATCH (gap)-[:HAS_OBSERVATION]->(observation:DDGapObservation)
WITH gap, source_paths, collect(DISTINCT observation.id) AS observation_ids
OPTIONAL MATCH (gap)-[:HAS_STATE_CHANGE]->(change:DDGapStateChange)
RETURN gap.id AS id,
       gap.path AS path,
       gap.kind AS kind,
       gap.status AS status,
       gap.registry_backend AS registry_backend,
       gap.upstream_url AS upstream_url,
       gap.resolved_dd_version AS resolved_dd_version,
       gap.triaged_at AS triaged_at,
       gap.triage_actor AS triage_actor,
       gap.triage_reason AS triage_reason,
       gap.status_changed_at AS status_changed_at,
       gap.status_changed_by AS status_changed_by,
       gap.status_change_reason AS status_change_reason,
       gap.validation_evidence AS validation_evidence,
       gap.first_seen_at AS first_seen_at,
       gap.last_seen_at AS last_seen_at,
       gap.example_count AS example_count,
       gap.affected_path_count AS affected_path_count,
       gap.observed_dd_version AS observed_dd_version,
       gap.observed_value AS observed_value,
       gap.expected_value AS expected_value,
       gap.evidence_rule AS evidence_rule,
       gap.reference_path AS reference_path,
       gap.reference_value AS reference_value,
       source_paths,
       observation_ids,
       collect(DISTINCT change.id) AS state_change_ids
ORDER BY gap.id
"""


def _registry_identity(
    fact: Mapping[str, Any], source_paths: Sequence[str]
) -> tuple[str, str, str, tuple[str, ...]]:
    """Return the conservative registry identity independent of gap kind."""
    return (
        _optional_text(fact.get("registry_backend")) or "",
        _optional_text(fact.get("path")) or "",
        _optional_text(fact.get("upstream_url")) or "",
        tuple(sorted(str(path) for path in source_paths)),
    )


def _registry_sync_snapshot(fact: Mapping[str, Any]) -> dict[str, Any]:
    """Capture every lifecycle, evidence, and link field guarded during sync."""
    evidence = _evidence_snapshot(fact)
    return {
        **evidence,
        "id": str(fact["id"]),
        "status": _optional_text(fact.get("status")) or "",
        "upstream_url": _optional_text(fact.get("upstream_url")) or "",
        "resolved_dd_version": _optional_text(fact.get("resolved_dd_version")) or "",
        "triaged_at": fact.get("triaged_at"),
        "triage_actor": _optional_text(fact.get("triage_actor")) or "",
        "triage_reason": _optional_text(fact.get("triage_reason")) or "",
        "status_changed_at": fact.get("status_changed_at"),
        "status_changed_by": _optional_text(fact.get("status_changed_by")) or "",
        "status_change_reason": _optional_text(fact.get("status_change_reason")) or "",
        "validation_evidence": _optional_text(fact.get("validation_evidence")) or "",
        "affected_path_count": int(fact.get("affected_path_count") or 0),
        "state_change_ids": sorted(
            str(item) for item in (fact.get("state_change_ids") or [])
        ),
    }


def _registry_sync_token(fact: Mapping[str, Any]) -> str:
    """Identify the exact registry fact state used to plan a rewrite."""
    snapshot = _registry_sync_snapshot(fact)
    serializable = {
        key: str(value or "") if key.endswith("_at") else value
        for key, value in snapshot.items()
    }
    digest = hashlib.sha256(
        json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"dd-gap-registry-sync:{digest}"


def _registry_sync_plan(
    nodes: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    existing_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify registry identities without mutating or guessing intent."""
    paths_by_id: dict[str, set[str]] = {}
    for observation in observations:
        paths_by_id.setdefault(str(observation["gap_id"]), set()).add(
            str(observation["source_path"])
        )

    existing_by_id = {str(fact["id"]): fact for fact in existing_facts}
    existing_by_identity: dict[tuple[str, str, str, tuple[str, ...]], list[dict]] = {}
    for fact in existing_facts:
        identity = _registry_identity(fact, fact.get("source_paths") or [])
        existing_by_identity.setdefault(identity, []).append(fact)

    creates: list[str] = []
    updates: list[str] = []
    reclassifications: list[dict[str, Any]] = []
    manual_required: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda item: str(item["id"])):
        target_id = str(node["id"])
        intended_paths = sorted(paths_by_id.get(target_id, set()))
        intended_identity = _registry_identity(node, intended_paths)
        target = existing_by_id.get(target_id)
        candidates = [
            fact
            for fact in existing_by_identity.get(intended_identity, [])
            if str(fact["id"]) != target_id
        ]
        related_candidates = [
            fact
            for fact in existing_facts
            if str(fact["id"]) != target_id
            and (
                _optional_text(fact.get("registry_backend")) or "",
                _optional_text(fact.get("path")) or "",
                _optional_text(fact.get("upstream_url")) or "",
            )
            == intended_identity[:3]
        ]

        if target is not None:
            target_identity = _registry_identity(
                target, target.get("source_paths") or []
            )
            if target_identity != intended_identity:
                manual_required.append(
                    {
                        "id": target_id,
                        "reason": "target id belongs to different registry evidence",
                    }
                )
            elif candidates:
                manual_required.append(
                    {
                        "id": target_id,
                        "reason": "target id collides with another matching registry fact",
                        "candidate_ids": sorted(str(item["id"]) for item in candidates),
                    }
                )
            else:
                updates.append(target_id)
            continue

        if len(candidates) > 1:
            manual_required.append(
                {
                    "id": target_id,
                    "reason": "multiple facts match the authoritative registry identity",
                    "candidate_ids": sorted(str(item["id"]) for item in candidates),
                }
            )
        elif len(candidates) == 1:
            old = candidates[0]
            reclassifications.append(
                {
                    "old_id": str(old["id"]),
                    "new_id": target_id,
                    "old_kind": _optional_text(old.get("kind")) or "",
                    "new_kind": str(node["kind"]),
                    "expected_sync_token": _registry_sync_token(old),
                    "expected": _registry_sync_snapshot(old),
                    "target": dict(node),
                }
            )
        elif related_candidates:
            manual_required.append(
                {
                    "id": target_id,
                    "reason": "registry evidence path set differs from existing fact",
                    "candidate_ids": sorted(
                        str(item["id"]) for item in related_candidates
                    ),
                }
            )
        else:
            creates.append(target_id)

    return {
        "create": creates,
        "update": updates,
        "reclassify": reclassifications,
        "manual_required": manual_required,
    }


_RECLASSIFY_REGISTRY_FACT_QUERY = """
MATCH (gap:DDGap {id: $old_id})
WHERE NOT EXISTS { MATCH (:DDGap {id: $new_id}) }
  AND coalesce(gap.path, '') = $expected_path
  AND coalesce(gap.kind, '') = $expected_kind
  AND coalesce(gap.status, '') = $expected_status
  AND coalesce(gap.registry_backend, '') = $expected_registry_backend
  AND coalesce(gap.upstream_url, '') = $expected_upstream_url
  AND coalesce(gap.resolved_dd_version, '') = $expected_resolved_dd_version
  AND coalesce(gap.triage_actor, '') = $expected_triage_actor
  AND coalesce(gap.triage_reason, '') = $expected_triage_reason
  AND coalesce(gap.status_changed_by, '') = $expected_status_changed_by
  AND coalesce(gap.status_change_reason, '') = $expected_status_change_reason
  AND coalesce(gap.validation_evidence, '') = $expected_validation_evidence
  AND coalesce(gap.observed_dd_version, '') = $expected_observed_dd_version
  AND coalesce(gap.observed_value, '') = $expected_observed_value
  AND coalesce(gap.expected_value, '') = $expected_expected_value
  AND coalesce(gap.evidence_rule, '') = $expected_evidence_rule
  AND coalesce(gap.reference_path, '') = $expected_reference_path
  AND coalesce(gap.reference_value, '') = $expected_reference_value
  AND coalesce(gap.example_count, 0) = $expected_example_count
  AND coalesce(gap.affected_path_count, 0) = $expected_affected_path_count
  AND ((gap.first_seen_at IS NULL AND $expected_first_seen_at IS NULL)
       OR gap.first_seen_at = $expected_first_seen_at)
  AND ((gap.last_seen_at IS NULL AND $expected_last_seen_at IS NULL)
       OR gap.last_seen_at = $expected_last_seen_at)
  AND ((gap.triaged_at IS NULL AND $expected_triaged_at IS NULL)
       OR gap.triaged_at = $expected_triaged_at)
  AND ((gap.status_changed_at IS NULL AND $expected_status_changed_at IS NULL)
       OR gap.status_changed_at = $expected_status_changed_at)
OPTIONAL MATCH (source:IMASNode)-[:HAS_DD_GAP]->(gap)
WITH gap, collect(DISTINCT source.id) AS source_paths
WHERE size(source_paths) = size($expected_source_paths)
  AND all(path IN source_paths WHERE path IN $expected_source_paths)
  AND all(path IN $expected_source_paths WHERE path IN source_paths)
OPTIONAL MATCH (gap)-[:HAS_OBSERVATION]->(observation:DDGapObservation)
WITH gap, source_paths, collect(DISTINCT observation) AS observation_nodes
WHERE size(observation_nodes) = size($expected_observation_ids)
  AND all(item IN observation_nodes WHERE item.id IN $expected_observation_ids)
  AND all(id IN $expected_observation_ids
          WHERE any(item IN observation_nodes WHERE item.id = id))
OPTIONAL MATCH (gap)-[:HAS_STATE_CHANGE]->(prior_change:DDGapStateChange)
WITH gap, source_paths, observation_nodes,
     collect(DISTINCT prior_change) AS prior_changes
WHERE size(prior_changes) = size($expected_state_change_ids)
  AND all(item IN prior_changes WHERE item.id IN $expected_state_change_ids)
  AND all(id IN $expected_state_change_ids
          WHERE any(item IN prior_changes WHERE item.id = id))
SET gap.id = $new_id,
    gap.path = $target_path,
    gap.kind = $target_kind,
    gap.registry_backend = $target_registry_backend,
    gap.affected_path_count = $target_affected_path_count,
    gap.observed_dd_version = $target_observed_dd_version,
    gap.observed_value = $target_observed_value,
    gap.expected_value = $target_expected_value,
    gap.evidence_rule = $target_evidence_rule
CREATE (gap)-[:HAS_STATE_CHANGE]->(change:DDGapStateChange {
    id: 'dd_gap_state_change:' + randomUUID(),
    dd_gap_id: $new_id,
    from_status: gap.status,
    to_status: gap.status,
    actor: 'registry-sync',
    reason: 'authoritative registry identity changed classification from '
            + $old_id + ' to ' + $new_id,
    changed_at: datetime($changed_at)
})
WITH gap, source_paths, observation_nodes, prior_changes, change
MATCH (verified:DDGap {id: $new_id})
WHERE verified = gap
  AND NOT EXISTS { MATCH (:DDGap {id: $old_id}) }
RETURN gap.id AS id,
       size(source_paths) AS source_path_count,
       size(observation_nodes) AS observation_count,
       size(prior_changes) + 1 AS state_change_count,
       change.id AS state_change_id
"""


_VERIFY_REGISTRY_SYNC_QUERY = """
UNWIND $expected AS item
OPTIONAL MATCH (gap:DDGap {id: item.id})
OPTIONAL MATCH (node:IMASNode)-[:HAS_DD_GAP]->(gap)
WITH item, gap, collect(DISTINCT node.id) AS source_paths
RETURN item.id AS id,
       gap.id IS NOT NULL AS exists,
       gap.kind AS kind,
       source_paths
ORDER BY id
"""


@retry_on_deadlock()
def sync_dd_unit_exception_gaps(*, dry_run: bool = False) -> dict[str, Any]:
    """Mirror curated unit exceptions into provenance without changing behavior."""
    registry_entries = load_exceptions()["dd_unit_bugs"]
    with GraphClient() as gc:
        versions = gc.query(
            "MATCH (version:DDVersion {is_current: true}) RETURN version.id AS id"
        )
        if len(versions) != 1:
            raise ValueError(
                "DD-gap registry sync requires exactly one current DDVersion"
            )
        observed_dd_version = str(versions[0]["id"])
        current_paths = [
            str(row["id"])
            for row in gc.query("MATCH (node:IMASNode) RETURN node.id AS id")
        ]
        nodes, observations = _registry_inventory(current_paths, observed_dd_version)
        existing_facts = [dict(row) for row in gc.query(_REGISTRY_FACTS_QUERY)]
        plan = _registry_sync_plan(nodes, observations, existing_facts)
        public_reclassifications = [
            {
                key: item[key]
                for key in (
                    "old_id",
                    "new_id",
                    "old_kind",
                    "new_kind",
                    "expected_sync_token",
                )
            }
            for item in plan["reclassify"]
        ]
        intended = {
            "registry_entries": len(registry_entries),
            "reported": len({node["id"] for node in nodes}),
            "relationships": len(
                {(item["source_path"], item["gap_id"]) for item in observations}
            ),
            "observations": len({str(item["observation_id"]) for item in observations}),
            "matched_paths": len({str(item["source_path"]) for item in observations}),
            "create": plan["create"],
            "update": plan["update"],
            "reclassify": public_reclassifications,
            "manual_required": plan["manual_required"],
            "dry_run": dry_run,
        }
        if dry_run:
            return intended

        if plan["manual_required"]:
            raise DDGapRegistrySyncConflict(
                "registry identity sync requires manual resolution: "
                + json.dumps(plan["manual_required"], sort_keys=True)
            )

        paths_by_id: dict[str, set[str]] = {}
        for observation in observations:
            paths_by_id.setdefault(str(observation["gap_id"]), set()).add(
                str(observation["source_path"])
            )

        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                current_facts = {
                    str(row["id"]): dict(row) for row in tx.run(_REGISTRY_FACTS_QUERY)
                }
                for item in plan["reclassify"]:
                    old_id = str(item["old_id"])
                    current = current_facts.get(old_id)
                    if (
                        current is None
                        or _registry_sync_token(current) != item["expected_sync_token"]
                    ):
                        raise DDGapRegistrySyncConflict(
                            f"registry fact {old_id!r} changed after preflight"
                        )

                    expected = item["expected"]
                    target = item["target"]
                    migrated = [
                        dict(row)
                        for row in tx.run(
                            _RECLASSIFY_REGISTRY_FACT_QUERY,
                            old_id=old_id,
                            new_id=item["new_id"],
                            expected_path=expected["path"],
                            expected_kind=expected["kind"],
                            expected_status=expected["status"],
                            expected_registry_backend=expected["registry_backend"],
                            expected_upstream_url=expected["upstream_url"],
                            expected_resolved_dd_version=expected[
                                "resolved_dd_version"
                            ],
                            expected_triaged_at=expected["triaged_at"],
                            expected_triage_actor=expected["triage_actor"],
                            expected_triage_reason=expected["triage_reason"],
                            expected_status_changed_at=expected["status_changed_at"],
                            expected_status_changed_by=expected["status_changed_by"],
                            expected_status_change_reason=expected[
                                "status_change_reason"
                            ],
                            expected_validation_evidence=expected[
                                "validation_evidence"
                            ],
                            expected_first_seen_at=expected["first_seen_at"],
                            expected_last_seen_at=expected["last_seen_at"],
                            expected_example_count=expected["example_count"],
                            expected_affected_path_count=expected[
                                "affected_path_count"
                            ],
                            expected_observed_dd_version=expected[
                                "observed_dd_version"
                            ],
                            expected_observed_value=expected["observed_value"],
                            expected_expected_value=expected["expected_value"],
                            expected_evidence_rule=expected["evidence_rule"],
                            expected_reference_path=expected["reference_path"],
                            expected_reference_value=expected["reference_value"],
                            expected_source_paths=expected["source_paths"],
                            expected_observation_ids=expected["observation_ids"],
                            expected_state_change_ids=expected["state_change_ids"],
                            target_path=target["path"],
                            target_kind=target["kind"],
                            target_registry_backend=target["registry_backend"],
                            target_affected_path_count=target["affected_path_count"],
                            target_observed_dd_version=target["observed_dd_version"],
                            target_observed_value=target["observed_value"],
                            target_expected_value=target["expected_value"],
                            target_evidence_rule=target["evidence_rule"],
                            changed_at=datetime.now(UTC).isoformat(),
                        )
                    ]
                    if len(migrated) != 1:
                        raise DDGapRegistrySyncConflict(
                            f"registry fact {old_id!r} no longer matches reviewed "
                            "lifecycle evidence and links"
                        )

                rows = [
                    dict(row)
                    for row in tx.run(
                        _SYNC_REGISTRY_QUERY,
                        nodes=nodes,
                        observations=observations,
                    )
                ]
                expected_nodes = [
                    {
                        "id": str(node["id"]),
                        "kind": str(node["kind"]),
                        "source_paths": sorted(paths_by_id.get(str(node["id"]), set())),
                    }
                    for node in nodes
                ]
                verified = [
                    dict(row)
                    for row in tx.run(
                        _VERIFY_REGISTRY_SYNC_QUERY,
                        expected=expected_nodes,
                    )
                ]
                verified_by_id = {str(row["id"]): row for row in verified}
                invalid = [
                    item["id"]
                    for item in expected_nodes
                    if not verified_by_id.get(item["id"], {}).get("exists")
                    or str(verified_by_id[item["id"]].get("kind") or "") != item["kind"]
                    or sorted(
                        str(path)
                        for path in (
                            verified_by_id[item["id"]].get("source_paths") or []
                        )
                    )
                    != item["source_paths"]
                ]
                old_ids = [str(item["old_id"]) for item in plan["reclassify"]]
                stale_ids = [
                    str(row["id"])
                    for row in tx.run(
                        "MATCH (gap:DDGap) WHERE gap.id IN $ids "
                        "RETURN gap.id AS id ORDER BY id",
                        ids=old_ids,
                    )
                ]
                if invalid or stale_ids:
                    raise DDGapRegistrySyncConflict(
                        "registry identity verification failed: "
                        f"invalid={sorted(invalid)} stale={sorted(stale_ids)}"
                    )
                tx.commit()
            except BaseException:
                try:
                    tx.rollback()
                except Exception:
                    tx.close()
                raise

        row = rows[0] if rows else {}
        return {
            "registry_entries": len(registry_entries),
            "reported": int(row.get("reported", 0)),
            "relationships": int(row.get("relationships", 0)),
            "observations": int(row.get("observations", 0)),
            "matched_paths": intended["matched_paths"],
            "create": plan["create"],
            "update": plan["update"],
            "reclassify": public_reclassifications,
            "manual_required": [],
            "dry_run": False,
        }


def _validate_transition_request(
    *,
    expected_status: str,
    new_status: str,
    actor: str,
    reason: str,
    upstream_url: str | None,
    resolved_dd_version: str | None,
    registry_backend: str | None,
    validation_evidence: str | None,
) -> tuple[str, str, str, str]:
    expected = _enum_value(expected_status, DDGapStatus)
    target = _enum_value(new_status, DDGapStatus)
    clean_actor = _optional_text(actor)
    clean_reason = _optional_text(reason)
    if not clean_actor:
        raise ValueError("DD-gap transition requires a non-empty actor")
    if not clean_reason:
        raise ValueError("DD-gap transition requires a non-empty reason")
    if target not in _ALLOWED_TRANSITIONS.get(expected, frozenset()):
        raise ValueError(f"invalid DD-gap transition {expected!r} -> {target!r}")
    if target == DDGapStatus.registered_exception.value and not _optional_text(
        registry_backend
    ):
        raise ValueError("registered_exception requires registry_backend provenance")
    if target == DDGapStatus.upstream_issue.value:
        parsed = urlparse(_optional_text(upstream_url) or "")
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("upstream_issue requires an absolute HTTPS URL")
    if target == DDGapStatus.resolved_upstream.value:
        if not _optional_text(resolved_dd_version):
            raise ValueError("resolved_upstream requires an exact published DD version")
        if not _optional_text(validation_evidence):
            raise ValueError("resolved_upstream requires validation_evidence")
    return expected, target, clean_actor, clean_reason


_TRANSITION_QUERY = """
MATCH (gap:DDGap {id: $gap_id})
WHERE gap.status = $expected_status
  AND coalesce(gap.path, '') = $evidence_path
  AND coalesce(gap.kind, '') = $evidence_kind
  AND coalesce(gap.observed_dd_version, '') = $evidence_observed_dd_version
  AND coalesce(gap.observed_value, '') = $evidence_observed_value
  AND coalesce(gap.expected_value, '') = $evidence_expected_value
  AND coalesce(gap.evidence_rule, '') = $evidence_rule
  AND coalesce(gap.reference_path, '') = $evidence_reference_path
  AND coalesce(gap.reference_value, '') = $evidence_reference_value
  AND coalesce(gap.registry_backend, '') = $evidence_registry_backend
OPTIONAL MATCH (source:IMASNode)-[:HAS_DD_GAP]->(gap)
WITH gap, collect(DISTINCT source.id) AS current_source_paths
OPTIONAL MATCH (gap)-[:HAS_OBSERVATION]->(observation:DDGapObservation)
WITH gap, current_source_paths,
     collect(DISTINCT observation.id) AS current_observation_ids
WHERE size(current_source_paths) = size($evidence_source_paths)
  AND all(path IN current_source_paths WHERE path IN $evidence_source_paths)
  AND all(path IN $evidence_source_paths WHERE path IN current_source_paths)
  AND size(current_observation_ids) = size($evidence_observation_ids)
  AND all(id IN current_observation_ids WHERE id IN $evidence_observation_ids)
  AND all(id IN $evidence_observation_ids WHERE id IN current_observation_ids)
  AND coalesce(gap.example_count, 0) = $evidence_example_count
  AND ((gap.first_seen_at IS NULL AND $evidence_first_seen_at IS NULL)
       OR gap.first_seen_at = $evidence_first_seen_at)
  AND ((gap.last_seen_at IS NULL AND $evidence_last_seen_at IS NULL)
       OR gap.last_seen_at = $evidence_last_seen_at)
CALL {
    WITH gap
    WITH gap WHERE $new_status <> 'resolved_upstream'
    RETURN true AS published_version_exists
    UNION
    WITH gap
    MATCH (version:DDVersion {id: $resolved_dd_version})
    WHERE $new_status = 'resolved_upstream'
    RETURN version.id IS NOT NULL AS published_version_exists
}
WITH gap, gap.status AS from_status, published_version_exists
SET gap.status = $new_status,
    gap.triaged_at = datetime($changed_at),
    gap.triage_actor = $actor,
    gap.triage_reason = $reason,
    gap.status_changed_at = datetime($changed_at),
    gap.status_changed_by = $actor,
    gap.status_change_reason = $reason,
    gap.registry_backend = coalesce($registry_backend, gap.registry_backend),
    gap.upstream_url = coalesce($upstream_url, gap.upstream_url),
    gap.resolved_dd_version = coalesce($resolved_dd_version,
                                        gap.resolved_dd_version),
    gap.validation_evidence = coalesce($validation_evidence,
                                       gap.validation_evidence)
CREATE (gap)-[:HAS_STATE_CHANGE]->(change:DDGapStateChange {
    id: 'dd_gap_state_change:' + randomUUID(),
    dd_gap_id: gap.id,
    from_status: from_status,
    to_status: $new_status,
    actor: $actor,
    reason: $reason,
    changed_at: datetime($changed_at),
    upstream_url: $upstream_url,
    resolved_dd_version: $resolved_dd_version,
    validation_evidence: $validation_evidence
})
RETURN gap.id AS id, from_status, gap.status AS status
"""


@retry_on_deadlock()
def transition_dd_gap(
    gap_id: str,
    *,
    expected_status: str,
    new_status: str,
    actor: str,
    reason: str,
    expected_evidence_token: str,
    upstream_url: str | None = None,
    resolved_dd_version: str | None = None,
    registry_backend: str | None = None,
    validation_evidence: str | None = None,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Apply one human-authorized transition with status and evidence CAS."""
    clean_gap_id = _validate_gap_id(gap_id)
    clean_evidence_token = _optional_text(expected_evidence_token)
    if not clean_evidence_token:
        raise ValueError("DD-gap transition requires expected_evidence_token")
    expected, target, clean_actor, clean_reason = _validate_transition_request(
        expected_status=expected_status,
        new_status=new_status,
        actor=actor,
        reason=reason,
        upstream_url=upstream_url,
        resolved_dd_version=resolved_dd_version,
        registry_backend=registry_backend,
        validation_evidence=validation_evidence,
    )
    if gc is None:
        with GraphClient() as owned:
            return transition_dd_gap(
                clean_gap_id,
                expected_status=expected,
                new_status=target,
                actor=clean_actor,
                reason=clean_reason,
                expected_evidence_token=clean_evidence_token,
                upstream_url=upstream_url,
                resolved_dd_version=resolved_dd_version,
                registry_backend=registry_backend,
                validation_evidence=validation_evidence,
                gc=owned,
            )

    fact = get_dd_gap(clean_gap_id, gc=gc)
    if fact is None:
        raise DDGapTransitionConflict(f"DD gap {clean_gap_id!r} does not exist")
    current_evidence_token = str(fact["evidence_token"])
    if current_evidence_token != clean_evidence_token:
        raise DDGapTransitionConflict(
            f"DD gap {clean_gap_id!r} evidence changed; expected "
            f"{clean_evidence_token!r}, found {current_evidence_token!r}"
        )
    evidence = _evidence_snapshot(fact)
    rows = gc.query(
        _TRANSITION_QUERY,
        gap_id=clean_gap_id,
        expected_status=expected,
        new_status=target,
        actor=clean_actor,
        reason=clean_reason,
        evidence_path=evidence["path"],
        evidence_kind=evidence["kind"],
        evidence_observed_dd_version=evidence["observed_dd_version"],
        evidence_observed_value=evidence["observed_value"],
        evidence_expected_value=evidence["expected_value"],
        evidence_rule=evidence["evidence_rule"],
        evidence_reference_path=evidence["reference_path"],
        evidence_reference_value=evidence["reference_value"],
        evidence_registry_backend=evidence["registry_backend"],
        evidence_source_paths=evidence["source_paths"],
        evidence_observation_ids=evidence["observation_ids"],
        evidence_example_count=evidence["example_count"],
        evidence_first_seen_at=evidence["first_seen_at"],
        evidence_last_seen_at=evidence["last_seen_at"],
        changed_at=datetime.now(UTC).isoformat(),
        upstream_url=_optional_text(upstream_url),
        resolved_dd_version=_optional_text(resolved_dd_version),
        registry_backend=_optional_text(registry_backend),
        validation_evidence=_optional_text(validation_evidence),
    )
    if not rows:
        raise DDGapTransitionConflict(
            f"DD gap {clean_gap_id!r} did not match expected {expected!r} status, "
            "reviewed evidence, or published resolution version"
        )
    return dict(rows[0])


def build_unit_release_facts(
    rows: list[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Normalize raw DD parser rows into exact-path reconciliation facts."""
    facts: dict[str, dict[str, str]] = {}
    for row in rows:
        path = _optional_text(row.get("path") or row.get("id"))
        if not path:
            raise ValueError("DD unit release fact requires an exact path")
        unit = _optional_text(row.get("unit") or row.get("units"))
        if path in facts and facts[path].get("unit") != unit:
            raise ValueError(f"conflicting DD unit release facts for {path}")
        facts[path] = {"unit": unit or ""}
    return facts


def _unit_matches_expected(actual: Any, expected: Any) -> bool:
    actual_unit = canonical_or_none(_optional_text(actual))
    expected_unit = canonical_or_none(_optional_text(expected))
    return actual_unit is not None and actual_unit == expected_unit


_RECONCILE_QUERY = """
MATCH (version:DDVersion {id: $dd_version})
WHERE NOT $require_current OR version.is_current = true
UNWIND $batch AS item
MATCH (gap:DDGap {id: item.id})
WHERE gap.status = item.expected_status
  AND coalesce(gap.path, '') = item.path
  AND coalesce(gap.kind, '') = item.kind
  AND coalesce(gap.observed_dd_version, '') = item.observed_dd_version
  AND coalesce(gap.observed_value, '') = item.observed_value
  AND coalesce(gap.expected_value, '') = item.expected_value
  AND coalesce(gap.evidence_rule, '') = item.evidence_rule
  AND coalesce(gap.reference_path, '') = item.reference_path
  AND coalesce(gap.reference_value, '') = item.reference_value
  AND coalesce(gap.registry_backend, '') = item.registry_backend
OPTIONAL MATCH (source:IMASNode)-[:HAS_DD_GAP]->(gap)
WITH version, gap, gap.status AS from_status, item,
     collect(DISTINCT source.id) AS current_source_paths
OPTIONAL MATCH (gap)-[:HAS_OBSERVATION]->(observation:DDGapObservation)
WITH version, gap, from_status, item, current_source_paths,
     collect(DISTINCT observation.id) AS current_observation_ids
WHERE size(current_source_paths) = size(item.source_paths)
  AND all(path IN current_source_paths WHERE path IN item.source_paths)
  AND all(path IN item.source_paths WHERE path IN current_source_paths)
  AND size(current_observation_ids) = size(item.observation_ids)
  AND all(id IN current_observation_ids WHERE id IN item.observation_ids)
  AND all(id IN item.observation_ids WHERE id IN current_observation_ids)
  AND coalesce(gap.example_count, 0) = item.example_count
  AND ((gap.first_seen_at IS NULL AND item.first_seen_at IS NULL)
       OR gap.first_seen_at = item.first_seen_at)
  AND ((gap.last_seen_at IS NULL AND item.last_seen_at IS NULL)
       OR gap.last_seen_at = item.last_seen_at)
SET gap.status = 'resolved_upstream',
    gap.triaged_at = datetime($changed_at),
    gap.triage_actor = $actor,
    gap.triage_reason = item.validation_evidence,
    gap.status_changed_at = datetime($changed_at),
    gap.status_changed_by = $actor,
    gap.status_change_reason = item.validation_evidence,
    gap.resolved_dd_version = version.id,
    gap.validation_evidence = item.validation_evidence
CREATE (gap)-[:HAS_STATE_CHANGE]->(change:DDGapStateChange {
    id: 'dd_gap_state_change:' + randomUUID(),
    dd_gap_id: gap.id,
    from_status: from_status,
    to_status: 'resolved_upstream',
    actor: $actor,
    reason: item.validation_evidence,
    changed_at: datetime($changed_at),
    resolved_dd_version: version.id,
    validation_evidence: item.validation_evidence
})
RETURN gap.id AS id
"""


@retry_on_deadlock()
def reconcile_dd_gaps(
    dd_version: str,
    release_facts: Mapping[str, Mapping[str, Any] | str],
    *,
    require_current: bool = True,
    dry_run: bool = False,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Resolve only DD gaps mechanically proven corrected by a published release.

    ``release_facts`` must contain raw declarations parsed from the named DD
    release. Stored ``IMASNode`` values are deliberately not used because the
    unit registry may have corrected that cache before upstream publication.
    """
    if gc is None:
        with GraphClient() as owned:
            return reconcile_dd_gaps(
                dd_version,
                release_facts,
                require_current=require_current,
                dry_run=dry_run,
                gc=owned,
            )

    versions = gc.query(
        """
        MATCH (version:DDVersion {id: $dd_version})
        RETURN version.id AS id, version.is_current AS is_current
        """,
        dd_version=dd_version,
    )
    if len(versions) != 1:
        raise ValueError(f"{dd_version!r} is not an exact published DD version")
    if require_current and not bool(versions[0].get("is_current")):
        raise ValueError(f"published DD version {dd_version!r} is not current")

    gaps = gc.query(
        """
        MATCH (gap:DDGap)
        WHERE gap.status IN $statuses
        OPTIONAL MATCH (node:IMASNode)-[:HAS_DD_GAP]->(gap)
        WITH gap, collect(DISTINCT node.id) AS source_paths
        OPTIONAL MATCH (gap)-[:HAS_OBSERVATION]->(observation:DDGapObservation)
        RETURN gap.id AS id, gap.path AS path, gap.kind AS kind,
               gap.status AS status, gap.expected_value AS expected_value,
               gap.example_count AS example_count,
               gap.first_seen_at AS first_seen_at,
               gap.last_seen_at AS last_seen_at,
               gap.observed_dd_version AS observed_dd_version,
               gap.observed_value AS observed_value,
               gap.evidence_rule AS evidence_rule,
               gap.reference_path AS reference_path,
               gap.reference_value AS reference_value,
               gap.registry_backend AS registry_backend,
               source_paths,
               collect(DISTINCT observation.id) AS observation_ids
        ORDER BY id
        """,
        statuses=list(_RECONCILABLE_STATUSES),
    )

    candidates: list[dict[str, Any]] = []
    manual_required: list[dict[str, str]] = []
    unchanged: list[str] = []
    registry_candidates: set[str] = set()
    for gap in gaps:
        gap_id = str(gap["id"])
        kind = str(gap.get("kind") or "")
        rule = str(gap.get("evidence_rule") or "")
        fact_field = _RELEASE_VALIDATORS.get((kind, rule))
        if fact_field is None:
            manual_required.append(
                {"id": gap_id, "reason": f"unsupported predicate for {kind}"}
            )
            continue

        expected = _optional_text(gap.get("expected_value"))
        if expected is None:
            manual_required.append(
                {"id": gap_id, "reason": "structured expected_value is missing"}
            )
            continue
        source_paths = sorted(
            str(path) for path in (gap.get("source_paths") or []) if path
        )
        if not source_paths:
            manual_required.append(
                {"id": gap_id, "reason": "no exact evidence paths are linked"}
            )
            continue

        actual_values: list[Any] = []
        missing_path: str | None = None
        for path in source_paths:
            fact = release_facts.get(path)
            if fact is None:
                missing_path = path
                break
            if isinstance(fact, Mapping):
                actual_values.append(fact.get(fact_field))
            else:
                actual_values.append(fact)
        if missing_path:
            manual_required.append(
                {
                    "id": gap_id,
                    "reason": f"release facts missing exact path {missing_path}",
                }
            )
            continue

        if fact_field == "unit" and all(
            _unit_matches_expected(actual, expected) for actual in actual_values
        ):
            validation = f"{dd_version} raw unit equals {expected!r} on " + ", ".join(
                source_paths
            )
            evidence = _evidence_snapshot(gap)
            candidates.append(
                {
                    "id": gap_id,
                    "expected_status": str(gap["status"]),
                    **evidence,
                    "validation_evidence": validation,
                }
            )
            if gap.get("registry_backend"):
                registry_candidates.add(gap_id)
        else:
            unchanged.append(gap_id)

    result: dict[str, Any] = {
        "dd_version": dd_version,
        "evaluated": len(gaps),
        "resolved": 0,
        "would_resolve": [item["id"] for item in candidates],
        "manual_required": manual_required,
        "unchanged": unchanged,
        "conflicts": [],
        "stale_registry_entries": [],
        "dry_run": dry_run,
    }
    if dry_run or not candidates:
        return result

    rows = gc.query(
        _RECONCILE_QUERY,
        dd_version=dd_version,
        require_current=require_current,
        batch=candidates,
        actor="dd-release-reconcile",
        changed_at=datetime.now(UTC).isoformat(),
    )
    resolved_ids = sorted(str(row["id"]) for row in rows)
    candidate_ids = {item["id"] for item in candidates}
    result["resolved"] = len(resolved_ids)
    result["would_resolve"] = []
    result["conflicts"] = sorted(candidate_ids - set(resolved_ids))
    result["stale_registry_entries"] = sorted(
        registry_candidates.intersection(resolved_ids)
    )
    return result


_LIST_DD_GAPS_QUERY = """
MATCH (gap:DDGap)
WHERE size($statuses) = 0 OR gap.status IN $statuses
WITH gap
WHERE size($kinds) = 0 OR gap.kind IN $kinds
OPTIONAL MATCH (node:IMASNode)-[:HAS_DD_GAP]->(gap)
WITH gap, collect(DISTINCT node.id) AS source_paths
WHERE size($path_ids) = 0
   OR any(path_id IN source_paths WHERE path_id IN $path_ids)
OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(name:StandardName)
WHERE source.source_type = 'dd' AND source.source_id IN source_paths
WITH gap, source_paths, collect(DISTINCT name.id) AS affected_name_ids
WHERE size($name_ids) = 0
   OR any(name_id IN affected_name_ids WHERE name_id IN $name_ids)
OPTIONAL MATCH (gap)-[:HAS_OBSERVATION]->(observation:DDGapObservation)
RETURN gap.id AS id,
       gap.path AS path,
       gap.kind AS kind,
       gap.status AS status,
       gap.example_count AS example_count,
       gap.first_seen_at AS first_seen_at,
       gap.last_seen_at AS last_seen_at,
       gap.observed_dd_version AS observed_dd_version,
       gap.observed_value AS observed_value,
       gap.expected_value AS expected_value,
       gap.evidence_rule AS evidence_rule,
       gap.reference_path AS reference_path,
       gap.reference_value AS reference_value,
       source_paths,
       affected_name_ids,
       gap.upstream_url AS upstream_url,
       gap.registry_backend AS registry_backend,
       gap.resolved_dd_version AS resolved_dd_version,
       collect(DISTINCT observation.id) AS observation_ids,
       size(source_paths) AS affected_path_count
ORDER BY gap.id
"""


def list_dd_gaps(
    *,
    statuses: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
    path_ids: Sequence[str] | None = None,
    name_ids: Sequence[str] | None = None,
    gc: GraphClient | None = None,
) -> list[dict[str, Any]]:
    """List lifecycle facts through deterministic, exact read filters.

    ``path_ids`` matches the exact IMAS paths carrying evidence, not the
    possibly patterned canonical ``DDGap.path``. ``name_ids`` matches names
    produced from those exact DD sources. All filters are sequences so callers
    cannot accidentally turn one string into a character-wise filter.
    """
    clean_statuses = _filter_sequence(statuses, label="statuses", enum_type=DDGapStatus)
    clean_kinds = _filter_sequence(kinds, label="kinds", enum_type=DDGapKind)
    clean_path_ids = _filter_sequence(path_ids, label="path_ids", exact_paths=True)
    clean_name_ids = _filter_sequence(name_ids, label="name_ids")
    if gc is None:
        with GraphClient() as owned:
            return list_dd_gaps(
                statuses=clean_statuses,
                kinds=clean_kinds,
                path_ids=clean_path_ids,
                name_ids=clean_name_ids,
                gc=owned,
            )
    rows = gc.query(
        _LIST_DD_GAPS_QUERY,
        statuses=clean_statuses,
        kinds=clean_kinds,
        path_ids=clean_path_ids,
        name_ids=clean_name_ids,
    )
    result = [dict(row) for row in rows]
    for row in result:
        row["source_paths"] = sorted(
            str(item) for item in (row.get("source_paths") or [])
        )
        row["affected_name_ids"] = sorted(
            str(item) for item in (row.get("affected_name_ids") or [])
        )
        row["observation_ids"] = sorted(
            str(item) for item in (row.get("observation_ids") or [])
        )
        row["evidence_token"] = _evidence_token(row)
    return sorted(result, key=lambda row: str(row["id"]))


def get_dd_gap(
    gap_id: str,
    *,
    gc: GraphClient | None = None,
) -> dict[str, Any] | None:
    """Return one exact lifecycle fact with evidence and state-change history."""
    clean_id = _validate_gap_id(gap_id)
    if gc is None:
        with GraphClient() as owned:
            return get_dd_gap(clean_id, gc=owned)

    facts = gc.query(
        """
        MATCH (gap:DDGap {id: $gap_id})
        OPTIONAL MATCH (node:IMASNode)-[:HAS_DD_GAP]->(gap)
        WITH gap, collect(DISTINCT node.id) AS source_paths
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(name:StandardName)
        WHERE source.source_type = 'dd' AND source.source_id IN source_paths
        RETURN gap.id AS id,
               gap.path AS path,
               gap.kind AS kind,
               gap.status AS status,
               gap.example_count AS example_count,
               gap.first_seen_at AS first_seen_at,
               gap.last_seen_at AS last_seen_at,
               gap.observed_dd_version AS observed_dd_version,
               gap.observed_value AS observed_value,
               gap.expected_value AS expected_value,
               gap.evidence_rule AS evidence_rule,
               gap.reference_path AS reference_path,
               gap.reference_value AS reference_value,
               gap.triaged_at AS triaged_at,
               gap.triage_actor AS triage_actor,
               gap.triage_reason AS triage_reason,
               gap.status_changed_at AS status_changed_at,
               gap.status_changed_by AS status_changed_by,
               gap.status_change_reason AS status_change_reason,
               gap.upstream_url AS upstream_url,
               gap.registry_backend AS registry_backend,
               gap.resolved_dd_version AS resolved_dd_version,
               gap.validation_evidence AS validation_evidence,
               source_paths,
               collect(DISTINCT name.id) AS affected_name_ids,
               size(source_paths) AS affected_path_count
        """,
        gap_id=clean_id,
    )
    if not facts:
        return None

    observations = gc.query(
        """
        MATCH (:DDGap {id: $gap_id})-[:HAS_OBSERVATION]->(item:DDGapObservation)
        RETURN item.id AS id,
               item.source_path AS source_path,
               item.reason AS reason,
               item.reporter AS reporter,
               item.observed_dd_version AS observed_dd_version,
               item.observed_value AS observed_value,
               item.expected_value AS expected_value,
               item.evidence_rule AS evidence_rule,
               item.reference_path AS reference_path,
               item.reference_value AS reference_value,
               item.first_observed_at AS first_observed_at,
               item.last_observed_at AS last_observed_at
        ORDER BY item.first_observed_at, item.id
        """,
        gap_id=clean_id,
    )
    state_changes = gc.query(
        """
        MATCH (:DDGap {id: $gap_id})-[:HAS_STATE_CHANGE]->(item:DDGapStateChange)
        RETURN item.id AS id,
               item.from_status AS from_status,
               item.to_status AS to_status,
               item.actor AS actor,
               item.reason AS reason,
               item.changed_at AS changed_at,
               item.upstream_url AS upstream_url,
               item.resolved_dd_version AS resolved_dd_version,
               item.validation_evidence AS validation_evidence
        ORDER BY item.changed_at, item.id
        """,
        gap_id=clean_id,
    )
    result = dict(facts[0])
    result["source_paths"] = sorted(str(item) for item in result["source_paths"])
    result["affected_name_ids"] = sorted(
        str(item) for item in result["affected_name_ids"]
    )
    result["observations"] = sorted(
        (dict(row) for row in observations),
        key=lambda row: (str(row.get("first_observed_at") or ""), str(row["id"])),
    )
    result["state_changes"] = sorted(
        (dict(row) for row in state_changes),
        key=lambda row: (str(row.get("changed_at") or ""), str(row["id"])),
    )
    result["evidence_token"] = _evidence_token(result)
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

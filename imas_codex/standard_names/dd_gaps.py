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
from collections.abc import Mapping
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
        intended = {
            "registry_entries": len(registry_entries),
            "reported": len({node["id"] for node in nodes}),
            "relationships": len(
                {(item["source_path"], item["gap_id"]) for item in observations}
            ),
            "observations": len({str(item["observation_id"]) for item in observations}),
            "matched_paths": len({str(item["source_path"]) for item in observations}),
            "dry_run": dry_run,
        }
        if dry_run:
            return intended

        rows = gc.query(
            _SYNC_REGISTRY_QUERY,
            nodes=nodes,
            observations=observations,
        )
        row = rows[0] if rows else {}
        return {
            "registry_entries": len(registry_entries),
            "reported": int(row.get("reported", 0)),
            "relationships": int(row.get("relationships", 0)),
            "observations": int(row.get("observations", 0)),
            "matched_paths": intended["matched_paths"],
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
    upstream_url: str | None = None,
    resolved_dd_version: str | None = None,
    registry_backend: str | None = None,
    validation_evidence: str | None = None,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Apply one human-authorized lifecycle transition with status CAS."""
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
                gap_id,
                expected_status=expected,
                new_status=target,
                actor=clean_actor,
                reason=clean_reason,
                upstream_url=upstream_url,
                resolved_dd_version=resolved_dd_version,
                registry_backend=registry_backend,
                validation_evidence=validation_evidence,
                gc=owned,
            )

    rows = gc.query(
        _TRANSITION_QUERY,
        gap_id=gap_id,
        expected_status=expected,
        new_status=target,
        actor=clean_actor,
        reason=clean_reason,
        changed_at=datetime.now(UTC).isoformat(),
        upstream_url=_optional_text(upstream_url),
        resolved_dd_version=_optional_text(resolved_dd_version),
        registry_backend=_optional_text(registry_backend),
        validation_evidence=_optional_text(validation_evidence),
    )
    if not rows:
        raise DDGapTransitionConflict(
            f"DD gap {gap_id!r} did not have expected {expected!r} status "
            "or the published resolution version does not exist"
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
WITH version, gap, gap.status AS from_status, item
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
        RETURN gap.id AS id, gap.path AS path, gap.kind AS kind,
               gap.status AS status, gap.expected_value AS expected_value,
               gap.evidence_rule AS evidence_rule,
               gap.registry_backend AS registry_backend,
               collect(DISTINCT node.id) AS source_paths
        ORDER BY id
        """,
        statuses=list(_RECONCILABLE_STATUSES),
    )

    candidates: list[dict[str, str]] = []
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
            candidates.append(
                {
                    "id": gap_id,
                    "expected_status": str(gap["status"]),
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

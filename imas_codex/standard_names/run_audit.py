"""Bounded, typed postflight evidence for one exact standard-name run."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

_TARGET_EVIDENCE_QUERY = """
// EXACT_STANDARD_NAME_TARGET_EVIDENCE
MATCH (target:StandardName {id: $name_id})
CALL (target) {
  OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(target)
  WITH target, source
  OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(backing:IMASNode)
  WITH target, source, backing
  OPTIONAL MATCH (backing)-[:HAS_UNIT]->(backing_unit:Unit)
  WITH target, source, backing,
       collect(DISTINCT backing_unit.id) AS backing_unit_ids
  OPTIONAL MATCH (backing)-[projection:HAS_STANDARD_NAME]->(target)
  WITH source, backing, backing_unit_ids,
       collect(DISTINCT elementId(projection)) AS projection_ids
  RETURN collect(DISTINCT CASE WHEN source IS NULL THEN null ELSE {
    source_id: source.id,
    source_type: source.source_type,
    source_status: source.status,
    produced_sn_id: source.produced_sn_id,
    dd_snapshot: source.dd_version,
    dd_snapshot_pinned: source.dd_snapshot_pinned,
    dd_unit: source.dd_unit,
    backing_id: backing.id,
    backing_unit: backing.unit,
    backing_unit_ids: backing_unit_ids,
    projection_ids: projection_ids,
    per_path_cocos_label: backing.cocos_transformation_type,
    west: source.id IN $west_source_ids,
    fixture: source.id STARTS WITH $fixture_source_id_prefix
             OR source.id STARTS WITH 'fixture:'
             OR source.id STARTS WITH 'test:'
             OR source.id STARTS WITH 'signals:test:'
  } END) AS sources
}
CALL (target) {
  OPTIONAL MATCH (target)-[:HAS_UNIT]->(target_unit:Unit)
  RETURN collect(DISTINCT target_unit.id) AS target_units
}
CALL () {
  OPTIONAL MATCH (version:DDVersion {is_current: true})
  OPTIONAL MATCH (version)-[:HAS_COCOS]->(cocos:COCOS)
  WITH version, collect(DISTINCT cocos.id) AS cocos_ids
  RETURN collect(DISTINCT CASE WHEN version IS NULL THEN null ELSE {
    id: version.id,
    cocos: version.cocos,
    cocos_ids: cocos_ids
  } END) AS current_versions
}
RETURN {
  element_id: elementId(target),
  id: target.id,
  run_id: target.run_id,
  last_run_id: target.last_run_id,
  name_stage: target.name_stage,
  docs_stage: target.docs_stage,
  status: target.status,
  validation_status: target.validation_status,
  reviewer_score_name: target.reviewer_score_name,
  reviewer_score_docs: target.reviewer_score_docs,
  origin: target.origin,
  edit_status: target.edit_status,
  unit: target.unit,
  dd_version: target.dd_version,
  cocos: target.cocos,
  cocos_transformation_type: target.cocos_transformation_type,
  chain_length: target.chain_length,
  docs_chain_length: target.docs_chain_length,
  refine_name_count: target.refine_name_count,
  generate_docs_count: target.generate_docs_count,
  review_docs_count: target.review_docs_count,
  refine_docs_count: target.refine_docs_count
} AS target,
sources, target_units, current_versions
"""


_RUN_EVIDENCE_QUERY = """
// EXACT_STANDARD_NAME_RUN_EVIDENCE
MATCH (run:SNRun)
WHERE run.id STARTS WITH $run_id_prefix
  AND run.started_at >= datetime($launched_at)
  AND run.started_at <= datetime($completed_at)
CALL (run) {
  OPTIONAL MATCH (cost:LLMCost)-[:FOR_RUN]->(run)
  WITH cost
  WHERE cost IS NULL
     OR cost.llm_at IS NULL
     OR (cost.llm_at >= datetime($launched_at)
         AND cost.llm_at <= datetime($completed_at))
  WITH collect(CASE WHEN cost IS NULL THEN null ELSE {
         id: cost.id,
         run_id: cost.run_id,
         pool: cost.pool,
         phase: cost.phase,
         event_type: cost.event_type,
         cycle: cost.cycle,
         sn_ids: cost.sn_ids,
         llm_cost: cost.llm_cost,
         overspend: cost.overspend,
         llm_at: cost.llm_at
       } END) AS costs,
       coalesce(sum(coalesce(cost.llm_cost, 0.0)), 0.0) AS ledger_cost,
       coalesce(sum(coalesce(cost.overspend, 0.0)), 0.0) AS overspend_cost,
       count(cost) AS cost_events
  RETURN costs, ledger_cost, overspend_cost, cost_events
}
CALL () {
  OPTIONAL MATCH (target:StandardName {id: $name_id})
                 -[:HAS_REVIEW]->(review:StandardNameReview)
  WITH review
  WHERE review IS NULL
     OR review.reviewed_at IS NULL
     OR (review.reviewed_at >= datetime($launched_at)
         AND review.reviewed_at <= datetime($completed_at))
  WITH collect(CASE WHEN review IS NULL THEN null ELSE {
         id: review.id,
         review_axis: review.review_axis,
         cycle_index: review.cycle_index,
         review_group_id: review.review_group_id,
         resolution_role: review.resolution_role,
         resolution_method: review.resolution_method,
         score: review.score,
         is_canonical: review.is_canonical,
         reviewed_at: review.reviewed_at,
         llm_at: review.llm_at
       } END) AS reviews,
       count(review) AS review_count
  RETURN reviews, review_count
}
RETURN {
  id: run.id,
  status: run.status,
  stop_reason: run.stop_reason,
  started_at: run.started_at,
  stopped_at: run.stopped_at,
  ended_at: run.ended_at,
  cost_spent: run.cost_spent,
  cost_limit: run.cost_limit,
  cost_total: run.cost_total,
  cost_is_exact: run.cost_is_exact,
  events_total: run.events_total
} AS run,
costs, ledger_cost, overspend_cost, cost_events, reviews, review_count
"""


_DELTA_EVIDENCE_QUERY = """
// EXACT_STANDARD_NAME_DELTA_EVIDENCE
MATCH (target:StandardName {id: $name_id})
CALL (target) {
  OPTIONAL MATCH (target)-[:REFINED_FROM]->(predecessor:StandardName)
  RETURN collect(DISTINCT predecessor.id) AS predecessor_ids
}
CALL (target) {
  OPTIONAL MATCH (successor:StandardName)-[:REFINED_FROM]->(target)
  WHERE successor.run_id = $scope_uuid
     OR successor.last_run_id STARTS WITH $run_id_prefix
  RETURN collect(DISTINCT successor.id) AS refined_successor_ids
}
CALL (target) {
  OPTIONAL MATCH (target)-[:DOCS_REVISION_OF]->(revision:DocsRevision)
  WHERE revision.created_at IS NULL
     OR (revision.created_at >= datetime($launched_at)
         AND revision.created_at <= datetime($completed_at))
  RETURN collect(DISTINCT revision.id) AS docs_revision_ids
}
CALL (target) {
  OPTIONAL MATCH (target)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
  WHERE change.changed_at IS NULL
     OR (change.changed_at >= datetime($launched_at)
         AND change.changed_at <= datetime($completed_at))
  RETURN collect(DISTINCT change.id) AS internal_change_ids
}
RETURN predecessor_ids, refined_successor_ids,
       docs_revision_ids, internal_change_ids
"""


class ExactStandardNameRunAuditReceipt(BaseModel):
    """Machine-readable evidence and verdict for one bounded pipeline run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    passed: bool = False
    diagnostics: list[str] = Field(default_factory=list)
    query_count: int = 0
    name_id: str
    scope_uuid: str
    run_id_prefix: str
    launched_at: str
    completed_at: str
    raw_rows: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    target_name_stage: str | None = None
    target_docs_stage: str | None = None
    target_status: str | None = None
    target_score_name: Decimal | None = None
    target_score_docs: Decimal | None = None
    target_run_id: str | None = None
    target_last_run_id: str | None = None
    target_protected: bool = False

    review_count: int = 0
    review_cycles: list[int] = Field(default_factory=list)
    review_resolutions: list[str] = Field(default_factory=list)

    run_id: str | None = None
    run_status: str | None = None
    run_stop_reason: str | None = None
    run_open: bool = False
    pool_counts: dict[str, int] = Field(default_factory=dict)
    ledger_cost: Decimal = Decimal("0")
    cumulative_cost: Decimal = Decimal("0")
    cost_limit: Decimal | None = None
    cost_is_exact: bool | None = None
    overspend_cost: Decimal = Decimal("0")
    overspent: bool = False

    source_ids: list[str] = Field(default_factory=list)
    source_count: int = 0
    backing_ids: list[str] = Field(default_factory=list)
    backing_count: int = 0
    projection_count: int = 0
    target_units: list[str] = Field(default_factory=list)
    backing_units: list[str] = Field(default_factory=list)
    dd_snapshot_versions: list[str] = Field(default_factory=list)
    current_dd_versions: list[str] = Field(default_factory=list)
    global_cocos: list[int] = Field(default_factory=list)
    per_path_cocos_labels: dict[str, str | None] = Field(default_factory=dict)
    west: bool = False
    fixture: bool = False

    predecessor_ids: list[str] = Field(default_factory=list)
    predecessor_count: int = 0
    refined_successor_ids: list[str] = Field(default_factory=list)
    refined_successor_count: int = 0
    docs_revision_ids: list[str] = Field(default_factory=list)
    docs_revision_count: int = 0
    internal_change_ids: list[str] = Field(default_factory=list)
    new_name_ids: list[str] = Field(default_factory=list)
    new_name_count: int = 0


def _bounded_datetime(value: datetime | str, *, field_name: str) -> datetime:
    if isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO 8601 datetime") from exc
    else:
        parsed = value
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _primitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    if isinstance(value, Decimal | str | int | float | bool) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _sorted_strings(values: Any) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None})


def _populate_receipt(receipt: ExactStandardNameRunAuditReceipt) -> None:
    target_rows = receipt.raw_rows.get("target", [])
    run_rows = receipt.raw_rows.get("run", [])
    delta_rows = receipt.raw_rows.get("deltas", [])

    if len(target_rows) != 1:
        receipt.diagnostics.append(
            f"exact target identity resolved to {len(target_rows)} rows"
        )
    else:
        row = target_rows[0]
        target = row.get("target") or {}
        receipt.target_name_stage = target.get("name_stage")
        receipt.target_docs_stage = target.get("docs_stage")
        receipt.target_status = target.get("status")
        receipt.target_score_name = _decimal(target.get("reviewer_score_name"))
        receipt.target_score_docs = _decimal(target.get("reviewer_score_docs"))
        receipt.target_run_id = target.get("run_id")
        receipt.target_last_run_id = target.get("last_run_id")
        receipt.target_protected = bool(
            target.get("origin") == "catalog_edit"
            or target.get("name_stage") == "approved"
        )

        sources = row.get("sources") or []
        receipt.source_ids = _sorted_strings(
            source.get("source_id") for source in sources
        )
        receipt.source_count = len(receipt.source_ids)
        receipt.backing_ids = _sorted_strings(
            source.get("backing_id") for source in sources
        )
        receipt.backing_count = len(receipt.backing_ids)
        receipt.projection_count = sum(
            len(source.get("projection_ids") or []) for source in sources
        )
        receipt.target_units = _sorted_strings(row.get("target_units"))
        receipt.backing_units = _sorted_strings(
            unit
            for source in sources
            for unit in [
                source.get("backing_unit"),
                *(source.get("backing_unit_ids") or []),
            ]
        )
        receipt.dd_snapshot_versions = _sorted_strings(
            source.get("dd_snapshot") for source in sources
        )
        receipt.per_path_cocos_labels = {
            str(source["backing_id"]): source.get("per_path_cocos_label")
            for source in sources
            if source.get("backing_id") is not None
        }
        receipt.west = any(bool(source.get("west")) for source in sources)
        receipt.fixture = any(bool(source.get("fixture")) for source in sources)
        current_versions = row.get("current_versions") or []
        receipt.current_dd_versions = _sorted_strings(
            version.get("id") for version in current_versions
        )
        receipt.global_cocos = sorted(
            {
                int(cocos)
                for version in current_versions
                for cocos in [version.get("cocos"), *(version.get("cocos_ids") or [])]
                if cocos is not None
            }
        )
        if receipt.target_run_id != receipt.scope_uuid:
            receipt.diagnostics.append(
                "target run provenance does not match exact scope"
            )
        if len(receipt.current_dd_versions) != 1:
            receipt.diagnostics.append(
                "current DD version identity is missing or ambiguous"
            )
        if receipt.global_cocos != [17]:
            receipt.diagnostics.append("current DD catalog COCOS is not exactly 17")
        if receipt.target_protected or receipt.west or receipt.fixture:
            receipt.diagnostics.append("target closure intersects protected state")

    if len(run_rows) != 1:
        receipt.diagnostics.append(
            f"bounded run identity resolved to {len(run_rows)} rows"
        )
    else:
        row = run_rows[0]
        run = row.get("run") or {}
        receipt.run_id = run.get("id")
        receipt.run_status = run.get("status")
        receipt.run_stop_reason = run.get("stop_reason")
        receipt.run_open = receipt.run_status == "started" or not run.get("ended_at")
        receipt.ledger_cost = _decimal(row.get("ledger_cost")) or Decimal("0")
        receipt.cumulative_cost = _decimal(run.get("cost_spent")) or Decimal("0")
        receipt.cost_limit = _decimal(run.get("cost_limit"))
        receipt.cost_is_exact = run.get("cost_is_exact")
        receipt.overspend_cost = _decimal(row.get("overspend_cost")) or Decimal("0")
        receipt.overspent = receipt.overspend_cost > 0
        costs = row.get("costs") or []
        for cost in costs:
            pool = str(cost.get("pool") or cost.get("phase") or "unknown")
            receipt.pool_counts[pool] = receipt.pool_counts.get(pool, 0) + 1
        reviews = row.get("reviews") or []
        receipt.review_count = int(row.get("review_count") or 0)
        receipt.review_cycles = sorted(
            {
                int(review["cycle_index"])
                for review in reviews
                if review.get("cycle_index") is not None
            }
        )
        receipt.review_resolutions = _sorted_strings(
            review.get("resolution_method") for review in reviews
        )
        run_total = _decimal(run.get("cost_total"))
        if run_total is not None and run_total != receipt.ledger_cost:
            receipt.diagnostics.append("run cost total differs from exact ledger sum")
        if receipt.run_open:
            receipt.diagnostics.append("selected run remains open")
        if receipt.cost_is_exact is False:
            receipt.diagnostics.append("selected run reports inexact cost")
        if receipt.overspent:
            receipt.diagnostics.append("selected run contains overspend")

    if len(delta_rows) != 1:
        receipt.diagnostics.append(
            f"exact target delta query resolved to {len(delta_rows)} rows"
        )
    else:
        row = delta_rows[0]
        receipt.predecessor_ids = _sorted_strings(row.get("predecessor_ids"))
        receipt.predecessor_count = len(receipt.predecessor_ids)
        receipt.refined_successor_ids = _sorted_strings(
            row.get("refined_successor_ids")
        )
        receipt.refined_successor_count = len(receipt.refined_successor_ids)
        receipt.docs_revision_ids = _sorted_strings(row.get("docs_revision_ids"))
        receipt.docs_revision_count = len(receipt.docs_revision_ids)
        receipt.internal_change_ids = _sorted_strings(row.get("internal_change_ids"))
        receipt.new_name_ids = list(receipt.refined_successor_ids)
        receipt.new_name_count = len(receipt.new_name_ids)

    receipt.passed = not receipt.diagnostics


def audit_exact_standard_name_run(
    name_id: str,
    scope_uuid: str,
    run_id_or_prefix: str,
    launched_at: datetime | str,
    completed_at: datetime | str,
    *,
    gc: GraphClient | None = None,
) -> ExactStandardNameRunAuditReceipt:
    """Audit one exact name and one bounded emitted run without caller Cypher.

    All evidence reads are parameterized and query-count constant with respect
    to graph size. Query failures return the evidence already captured instead
    of discarding it or raising past the receipt boundary.
    """
    normalized_name = name_id.strip()
    normalized_scope = scope_uuid.strip()
    normalized_run = run_id_or_prefix.strip()
    launch = _bounded_datetime(launched_at, field_name="launched_at")
    completion = _bounded_datetime(completed_at, field_name="completed_at")
    if not normalized_name or not normalized_scope or not normalized_run:
        raise ValueError("name_id, scope_uuid, and run_id_or_prefix are required")
    if completion < launch:
        raise ValueError("completed_at must not precede launched_at")

    receipt = ExactStandardNameRunAuditReceipt(
        name_id=normalized_name,
        scope_uuid=normalized_scope,
        run_id_prefix=normalized_run,
        launched_at=launch.isoformat(),
        completed_at=completion.isoformat(),
    )
    from imas_codex.standard_names.grammar_segment_reconciliation import (
        _FIXTURE_SOURCE_ID_PREFIX,
        _west_source_ids,
    )

    params = {
        "name_id": normalized_name,
        "scope_uuid": normalized_scope,
        "run_id_prefix": normalized_run,
        "launched_at": launch.isoformat(),
        "completed_at": completion.isoformat(),
        "west_source_ids": sorted(_west_source_ids()),
        "fixture_source_id_prefix": _FIXTURE_SOURCE_ID_PREFIX,
    }
    queries = (
        ("target", _TARGET_EVIDENCE_QUERY),
        ("run", _RUN_EVIDENCE_QUERY),
        ("deltas", _DELTA_EVIDENCE_QUERY),
    )
    if gc is None:
        from imas_codex.graph.client import GraphClient

        manager = GraphClient()
    else:
        manager = nullcontext(gc)
    with manager as client:
        assert client is not None
        for group, query in queries:
            receipt.query_count += 1
            try:
                rows = client.query(query, **params)
            except Exception as exc:
                receipt.diagnostics.append(
                    f"{group} evidence query failed: {type(exc).__name__}: {exc}"
                )
                break
            receipt.raw_rows[group] = _primitive(rows)

    _populate_receipt(receipt)
    return receipt


__all__ = ["ExactStandardNameRunAuditReceipt", "audit_exact_standard_name_run"]

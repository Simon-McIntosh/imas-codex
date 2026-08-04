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
sources, target_units
"""


_DD_EVIDENCE_QUERY = """
// EXACT_STANDARD_NAME_DD_EVIDENCE
MATCH (version:DDVersion {id: $dd_version})
OPTIONAL MATCH (version)-[:HAS_COCOS]->(cocos:COCOS)
WITH version, collect(DISTINCT cocos.id) AS cocos_ids
RETURN {
  id: version.id,
  is_current: version.is_current,
  cocos: version.cocos,
  cocos_ids: cocos_ids
} AS version
"""


_RUN_EVIDENCE_QUERY = """
// EXACT_STANDARD_NAME_RUN_EVIDENCE
MATCH (run:SNRun)
WHERE run.id STARTS WITH $run_id_prefix
  AND datetime(toString(run.started_at)) >= datetime($launched_at)
  AND datetime(toString(run.started_at)) <= datetime($completed_at)
CALL (run) {
  OPTIONAL MATCH (cost:LLMCost)-[:FOR_RUN]->(run)
  WITH run, cost
  WHERE cost IS NULL
     OR cost.llm_at IS NULL
     OR (datetime(toString(cost.llm_at)) >= datetime($launched_at)
         AND datetime(toString(cost.llm_at)) <= datetime($completed_at))
  WITH run, collect(CASE WHEN cost IS NULL THEN null ELSE {
         id: cost.id,
         run_id: cost.run_id,
         for_run: cost.for_run,
         linked_run_id: run.id,
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
CALL (run, costs) {
  OPTIONAL MATCH (target:StandardName {id: $name_id})
                 -[:HAS_REVIEW]->(review:StandardNameReview)
  WITH run, costs, target, review
  WHERE review IS NULL
     OR review.reviewed_at IS NULL
     OR (datetime(toString(review.reviewed_at)) >= datetime($launched_at)
         AND datetime(toString(review.reviewed_at)) <= datetime($completed_at))
  WITH review, run, costs, target,
       CASE review.review_axis
         WHEN 'names' THEN 'review_name'
         WHEN 'docs' THEN 'review_docs'
       END AS review_pool,
       CASE WHEN review.cycle_index IS NULL THEN null
            ELSE 'c' + toString(review.cycle_index)
       END AS review_cycle
  WITH review, run,
       [cost IN costs WHERE review_pool IS NOT NULL
          AND review_cycle IS NOT NULL
          AND cost.run_id = run.id
          AND target.id IN coalesce(cost.sn_ids, [])
          AND cost.pool = review_pool
          AND cost.cycle = review_cycle | cost.id] AS linked_cost_ids
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
         llm_at: review.llm_at,
         linked_run_id: run.id,
         linked_cost_ids: linked_cost_ids
       } END) AS reviews,
       count(review) AS review_count
  RETURN reviews, review_count
}
CALL (run) {
  OPTIONAL MATCH (target:StandardName {id: $name_id})
  OPTIONAL MATCH (target)-[:REFINED_FROM]->(predecessor:StandardName)
  WITH run, target, collect(DISTINCT predecessor.id) AS predecessor_ids
  OPTIONAL MATCH (successor:StandardName)-[:REFINED_FROM]->(target)
  WHERE successor.run_id = $scope_run_id
    AND successor.last_run_id = run.id
  WITH run, target, predecessor_ids,
       collect(DISTINCT successor.id) AS refined_successor_ids
  OPTIONAL MATCH (target)-[:DOCS_REVISION_OF]->(revision:DocsRevision)
  WHERE revision.created_at IS NULL
     OR (datetime(toString(revision.created_at)) >= datetime($launched_at)
         AND datetime(toString(revision.created_at)) <= datetime($completed_at))
  WITH run, target, predecessor_ids, refined_successor_ids,
       collect(DISTINCT revision.id) AS docs_revision_ids
  OPTIONAL MATCH (target)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
  WHERE change.changed_at IS NULL
     OR (datetime(toString(change.changed_at)) >= datetime($launched_at)
         AND datetime(toString(change.changed_at)) <= datetime($completed_at))
  RETURN predecessor_ids, refined_successor_ids, docs_revision_ids,
         collect(DISTINCT change.id) AS internal_change_ids
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
costs, ledger_cost, overspend_cost, cost_events, reviews, review_count,
predecessor_ids, refined_successor_ids, docs_revision_ids, internal_change_ids
"""


class ExactStandardNameRunAuditReceipt(BaseModel):
    """Machine-readable evidence and verdict for one bounded pipeline run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    passed: bool = False
    diagnostics: list[str] = Field(default_factory=list)
    query_count: int = 0
    name_id: str
    scope_run_id: str
    run_id_prefix: str
    launched_at: str
    completed_at: str
    raw_rows: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    target_name_stage: str | None = None
    target_docs_stage: str | None = None
    target_status: str | None = None
    target_score_name: Decimal | None = None
    target_score_docs: Decimal | None = None
    target_scope_run_id: str | None = None
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
    cost_total: Decimal | None = None
    cost_is_exact: bool | None = None
    events_total: int | None = None
    cost_event_count: int = 0
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
    dd_rows = receipt.raw_rows.get("dd", [])
    run_rows = receipt.raw_rows.get("run", [])

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
        receipt.target_scope_run_id = target.get("run_id")
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
        if receipt.target_scope_run_id != receipt.scope_run_id:
            receipt.diagnostics.append(
                "target run provenance does not match exact scope"
            )
        if receipt.target_protected or receipt.west or receipt.fixture:
            receipt.diagnostics.append("target closure intersects protected state")

    if len(receipt.dd_snapshot_versions) != 1:
        receipt.diagnostics.append(
            "source DD snapshot identity is missing or ambiguous"
        )
    if len(dd_rows) != 1:
        receipt.diagnostics.append(
            f"exact DD snapshot identity resolved to {len(dd_rows)} rows"
        )
    else:
        version = dd_rows[0].get("version") or {}
        version_id = version.get("id")
        receipt.current_dd_versions = _sorted_strings([version_id])
        receipt.global_cocos = sorted(
            {
                int(cocos)
                for cocos in [version.get("cocos"), *(version.get("cocos_ids") or [])]
                if cocos is not None
            }
        )
        if receipt.dd_snapshot_versions != receipt.current_dd_versions:
            receipt.diagnostics.append("DD evidence does not match source snapshot")
        if version.get("is_current") is not True:
            receipt.diagnostics.append("source DD snapshot is not current")
        if receipt.global_cocos != [17]:
            receipt.diagnostics.append("current DD catalog COCOS is not exactly 17")

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
        if not receipt.run_id or not receipt.run_id.startswith(receipt.run_id_prefix):
            receipt.diagnostics.append("run evidence does not match supplied identity")
        try:
            run_started = _bounded_datetime(
                str(run.get("started_at")), field_name="run.started_at"
            )
            launch = _bounded_datetime(receipt.launched_at, field_name="launched_at")
            completion = _bounded_datetime(
                receipt.completed_at, field_name="completed_at"
            )
        except ValueError:
            receipt.diagnostics.append("run evidence has an invalid start timestamp")
        else:
            if not launch <= run_started <= completion:
                receipt.diagnostics.append("run evidence falls outside supplied bounds")
        costs = row.get("costs") or []
        event_costs = [_decimal(cost.get("llm_cost")) for cost in costs]
        if any(cost is None for cost in event_costs):
            receipt.diagnostics.append("cost evidence contains a missing event cost")
        receipt.ledger_cost = sum(
            (cost for cost in event_costs if cost is not None), Decimal("0")
        )
        receipt.cumulative_cost = _decimal(run.get("cost_spent")) or Decimal("0")
        receipt.cost_limit = _decimal(run.get("cost_limit"))
        receipt.cost_total = _decimal(run.get("cost_total"))
        receipt.cost_is_exact = run.get("cost_is_exact")
        receipt.events_total = (
            int(run["events_total"]) if run.get("events_total") is not None else None
        )
        receipt.cost_event_count = int(row.get("cost_events") or 0)
        receipt.overspend_cost = _decimal(row.get("overspend_cost")) or Decimal("0")
        receipt.overspent = receipt.overspend_cost > 0
        cost_ids = [str(cost.get("id")) for cost in costs if cost.get("id") is not None]
        unrelated_costs = [
            cost
            for cost in costs
            if cost.get("run_id") != receipt.run_id
            or cost.get("linked_run_id") != receipt.run_id
        ]
        if unrelated_costs:
            receipt.diagnostics.append(
                "cost evidence is not linked to the selected run"
            )
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
        costs_by_id = {
            str(cost["id"]): cost for cost in costs if cost.get("id") is not None
        }
        review_pools = {"names": "review_name", "docs": "review_docs"}

        def review_costs_match(review: dict[str, Any]) -> bool:
            linked_cost_ids = [
                str(cost_id) for cost_id in review.get("linked_cost_ids") or []
            ]
            review_pool = review_pools.get(str(review.get("review_axis")))
            cycle_index = review.get("cycle_index")
            if not linked_cost_ids or review_pool is None or cycle_index is None:
                return False
            review_cycle = f"c{cycle_index}"
            for cost_id in linked_cost_ids:
                cost = costs_by_id.get(cost_id)
                if (
                    cost is None
                    or cost.get("run_id") != receipt.run_id
                    or cost.get("linked_run_id") != receipt.run_id
                    or cost.get("pool") != review_pool
                    or cost.get("cycle") != review_cycle
                    or receipt.name_id
                    not in {str(name_id) for name_id in cost.get("sn_ids") or []}
                ):
                    return False
            return True

        unrelated_reviews = [
            review
            for review in reviews
            if review.get("linked_run_id") != receipt.run_id
            or not review_costs_match(review)
        ]
        if unrelated_reviews:
            receipt.diagnostics.append(
                "review evidence is not linked to the selected run"
            )
        review_ids = [
            str(review.get("id")) for review in reviews if review.get("id") is not None
        ]
        if not (receipt.review_count == len(reviews) == len(set(review_ids))):
            receipt.diagnostics.append("review count differs from review evidence")
        if receipt.cost_total is None:
            receipt.diagnostics.append("selected run has no exact cost total")
        elif receipt.cost_total != receipt.ledger_cost:
            receipt.diagnostics.append("run cost total differs from exact ledger sum")
        if receipt.cumulative_cost != receipt.ledger_cost:
            receipt.diagnostics.append(
                "run cumulative cost differs from exact ledger sum"
            )
        if receipt.events_total is None:
            receipt.diagnostics.append("selected run has no event total")
        elif not (
            receipt.events_total
            == receipt.cost_event_count
            == len(costs)
            == len(set(cost_ids))
        ):
            receipt.diagnostics.append("run event total differs from cost evidence")
        if receipt.run_open:
            receipt.diagnostics.append("selected run remains open")
        if receipt.cost_is_exact is not True:
            receipt.diagnostics.append("selected run reports inexact cost")
        if receipt.overspent:
            receipt.diagnostics.append("selected run contains overspend")

        row = run_rows[0]
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
    scope_run_id: str,
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
    normalized_scope = scope_run_id.strip()
    normalized_run = run_id_or_prefix.strip()
    launch = _bounded_datetime(launched_at, field_name="launched_at")
    completion = _bounded_datetime(completed_at, field_name="completed_at")
    if not normalized_name or not normalized_scope or not normalized_run:
        raise ValueError("name_id, scope_run_id, and run_id_or_prefix are required")
    if completion < launch:
        raise ValueError("completed_at must not precede launched_at")

    receipt = ExactStandardNameRunAuditReceipt(
        name_id=normalized_name,
        scope_run_id=normalized_scope,
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
        "scope_run_id": normalized_scope,
        "run_id_prefix": normalized_run,
        "launched_at": launch.isoformat(),
        "completed_at": completion.isoformat(),
        "west_source_ids": sorted(_west_source_ids()),
        "fixture_source_id_prefix": _FIXTURE_SOURCE_ID_PREFIX,
    }
    if gc is None:
        from imas_codex.graph.client import GraphClient

        manager = GraphClient()
    else:
        manager = nullcontext(gc)
    with manager as client:
        assert client is not None
        receipt.query_count += 1
        try:
            target_rows = client.query(_TARGET_EVIDENCE_QUERY, **params)
        except Exception as exc:
            receipt.diagnostics.append(
                f"target evidence query failed: {type(exc).__name__}: {exc}"
            )
        else:
            receipt.raw_rows["target"] = _primitive(target_rows)

        snapshot_versions = {
            str(source["dd_snapshot"])
            for row in receipt.raw_rows.get("target", [])
            for source in row.get("sources") or []
            if source.get("dd_snapshot") is not None
        }
        if len(snapshot_versions) == 1:
            params["dd_version"] = next(iter(snapshot_versions))
            receipt.query_count += 1
            try:
                dd_rows = client.query(_DD_EVIDENCE_QUERY, **params)
            except Exception as exc:
                receipt.diagnostics.append(
                    f"dd evidence query failed: {type(exc).__name__}: {exc}"
                )
            else:
                receipt.raw_rows["dd"] = _primitive(dd_rows)

        receipt.query_count += 1
        try:
            run_rows = client.query(_RUN_EVIDENCE_QUERY, **params)
        except Exception as exc:
            receipt.diagnostics.append(
                f"run evidence query failed: {type(exc).__name__}: {exc}"
            )
        else:
            receipt.raw_rows["run"] = _primitive(run_rows)

    _populate_receipt(receipt)
    return receipt


__all__ = ["ExactStandardNameRunAuditReceipt", "audit_exact_standard_name_run"]

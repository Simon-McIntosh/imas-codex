"""Fail-closed evidence for authorizing one exact paid name refinement."""

from __future__ import annotations

from contextlib import nullcontext
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from imas_codex.settings import get_dd_version
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.graph_ops import REFINE_NAME_ELIGIBILITY_WHERE

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient


_EXACT_STANDARD_NAME_PREFLIGHT_QUERY = f"""
// EXACT_STANDARD_NAME_PREFLIGHT
OPTIONAL MATCH (candidate:StandardName {{id: $name_id}})
WITH collect(candidate) AS target_matches
CALL (target_matches) {{
  UNWIND target_matches AS sn
  WITH sn
  WHERE {REFINE_NAME_ELIGIBILITY_WHERE}
  RETURN count(sn) AS refine_action_count,
         collect(elementId(sn)) AS refine_action_element_ids
}}
CALL (target_matches) {{
  WITH head(target_matches) AS target
  OPTIONAL MATCH (source:StandardNameSource)-[produced:PRODUCED_NAME]->(target)
  WITH target, source, produced
  OPTIONAL MATCH (source)-[from_dd:FROM_DD_PATH]->(backing:IMASNode)
  WITH target, source, produced,
       collect(DISTINCT backing) AS backings,
       collect(DISTINCT elementId(from_dd)) AS from_dd_edge_ids
  UNWIND CASE WHEN backings = [] THEN [null] ELSE backings END AS backing
  OPTIONAL MATCH (backing)-[projection:HAS_STANDARD_NAME]->(target)
  OPTIONAL MATCH (backing)-[backing_unit_edge:HAS_UNIT]->(backing_unit:Unit)
  WITH source, produced, backing, from_dd_edge_ids,
       collect(DISTINCT elementId(projection)) AS projection_edge_ids,
       collect(DISTINCT backing_unit.id) AS backing_unit_ids,
       collect(DISTINCT elementId(backing_unit_edge)) AS backing_unit_edge_ids
  RETURN collect(CASE WHEN source IS NULL THEN null ELSE {{
    element_id: elementId(source),
    id: source.id,
    source_id: source.source_id,
    source_type: source.source_type,
    status: source.status,
    produced_sn_id: source.produced_sn_id,
    produced_edge_id: elementId(produced),
    claimed_at: source.claimed_at,
    claim_token: source.claim_token,
    drain_scope_id: source.drain_scope_id,
    drain_scope_claimed_at: source.drain_scope_claimed_at,
    drain_claim_scope_id: source.drain_claim_scope_id,
    dd_version: source.dd_version,
    dd_snapshot_pinned: source.dd_snapshot_pinned,
    dd_unit: source.dd_unit,
    dd_path: source.dd_path,
    backing_id: backing.id,
    backing_unit: backing.unit,
    backing_unit_ids: backing_unit_ids,
    backing_unit_edge_ids: backing_unit_edge_ids,
    from_dd_edge_ids: from_dd_edge_ids,
    projection_edge_ids: projection_edge_ids,
    cocos_label: backing.cocos_transformation_type
  }} END) AS sources
}}
CALL (target_matches) {{
  WITH head(target_matches) AS target
  OPTIONAL MATCH (target)-[target_unit_edge:HAS_UNIT]->(target_unit:Unit)
  RETURN collect(DISTINCT target_unit.id) AS target_unit_ids,
         collect(DISTINCT elementId(target_unit_edge)) AS target_unit_edge_ids
}}
CALL (target_matches) {{
  WITH head(target_matches) AS target
  OPTIONAL MATCH (target)-[:REFINED_FROM]->(predecessor:StandardName)
  WITH target, collect(DISTINCT predecessor.id) AS predecessor_ids
  OPTIONAL MATCH (successor:StandardName)-[:REFINED_FROM]->(target)
  WITH target, predecessor_ids,
       collect(DISTINCT successor.id) AS successor_ids
  OPTIONAL MATCH (target)-[:REFINED_FROM*0..]->(prior:StandardName)
  WITH target, predecessor_ids, successor_ids,
       collect(DISTINCT prior) AS prior_lineage
  OPTIONAL MATCH (later:StandardName)-[:REFINED_FROM*1..]->(target)
  WITH predecessor_ids, successor_ids, prior_lineage,
       collect(DISTINCT later) AS later_lineage
  WITH predecessor_ids, successor_ids,
       prior_lineage + later_lineage AS refinement_lineage
  UNWIND CASE WHEN refinement_lineage = [] THEN [null]
              ELSE refinement_lineage END AS member
  RETURN predecessor_ids, successor_ids,
         collect(DISTINCT CASE
           WHEN member.name_stage IN ['accepted', 'approved']
             OR member.status = 'active'
             OR member.origin = 'catalog_edit'
           THEN member.id
         END) AS accepted_or_protected_lineage_ids
}}
CALL (target_matches) {{
  WITH head(target_matches) AS target
  OPTIONAL MATCH (target)-[:HAS_PARENT*0..]->(ancestor:StandardName)
  WITH target, collect(DISTINCT ancestor) AS ancestors
  OPTIONAL MATCH (descendant:StandardName)-[:HAS_PARENT*1..]->(target)
  WITH ancestors, collect(DISTINCT descendant) AS descendants
  WITH ancestors + descendants AS structural_lineage
  UNWIND CASE WHEN structural_lineage = [] THEN [null]
              ELSE structural_lineage END AS member
  OPTIONAL MATCH (protected_source:StandardNameSource)-[:PRODUCED_NAME]->(member)
  WITH protected_source
  WHERE protected_source.id IN $west_source_ids
     OR protected_source.id STARTS WITH $fixture_source_id_prefix
     OR protected_source.id STARTS WITH 'fixture:'
     OR protected_source.id STARTS WITH 'test:'
     OR protected_source.id STARTS WITH 'signals:test:'
  RETURN collect(DISTINCT protected_source.id) AS protected_source_ids
}}
OPTIONAL MATCH (catalog:DDVersion {{id: $dd_version}})
WITH target_matches, refine_action_count, refine_action_element_ids,
     sources, target_unit_ids, target_unit_edge_ids,
     predecessor_ids, successor_ids, accepted_or_protected_lineage_ids,
     protected_source_ids, collect(catalog) AS catalog_matches
CALL (catalog_matches) {{
  UNWIND catalog_matches AS catalog
  OPTIONAL MATCH (catalog)-[catalog_cocos_edge:HAS_COCOS]->(catalog_cocos:COCOS)
  RETURN collect(DISTINCT catalog_cocos.id) AS catalog_cocos_ids,
         collect(DISTINCT elementId(catalog_cocos_edge)) AS catalog_cocos_edge_ids
}}
RETURN [target IN target_matches | {{
         element_id: elementId(target),
         id: target.id,
         name_stage: target.name_stage,
         docs_stage: target.docs_stage,
         status: target.status,
         validation_status: target.validation_status,
         reviewer_score_name: target.reviewer_score_name,
         chain_length: target.chain_length,
         review_resubmit_count: target.review_resubmit_count,
         origin: target.origin,
         edit_mode: target.edit_mode,
         edit_status: target.edit_status,
         unit: target.unit,
         dd_version: target.dd_version,
         cocos: target.cocos,
         cocos_transformation_type: target.cocos_transformation_type,
         source_paths: target.source_paths,
         run_id: target.run_id,
         last_run_id: target.last_run_id,
         claimed_at: target.claimed_at,
         claim_token: target.claim_token,
         drain_scope_id: target.drain_scope_id,
         drain_scope_claimed_at: target.drain_scope_claimed_at,
         drain_claim_scope_id: target.drain_claim_scope_id
       }}] AS targets,
       refine_action_count, refine_action_element_ids,
       sources, target_unit_ids, target_unit_edge_ids,
       predecessor_ids, successor_ids, accepted_or_protected_lineage_ids,
       protected_source_ids,
       [catalog IN catalog_matches | {{
         element_id: elementId(catalog),
         id: catalog.id,
         status: catalog.status,
         is_current: catalog.is_current,
         cocos: catalog.cocos
       }}] AS catalogs,
       catalog_cocos_ids, catalog_cocos_edge_ids
"""


class ExactStandardNamePreflightReceipt(BaseModel):
    """Machine-readable authorization evidence for one exact refinement."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    passed: bool = False
    diagnostics: list[str] = Field(default_factory=list)
    query_count: int = 0
    name_id: str
    dd_version: str
    min_score: Decimal
    rotation_cap: int
    requested_cost_ceiling: Decimal
    cumulative_spend: Decimal
    authorized_budget: Decimal
    budget_remaining_before: Decimal
    budget_remaining_after: Decimal
    raw_evidence: dict[str, Any] = Field(default_factory=dict)

    identity_count: int = 0
    refine_action_count: int = 0
    target_name_stage: str | None = None
    target_status: str | None = None
    target_validation_status: str | None = None
    target_score: Decimal | None = None
    target_chain_length: int | None = None
    target_edit_mode: str | None = None
    target_origin: str | None = None
    target_run_id: str | None = None
    target_claim_fields: dict[str, Any] = Field(default_factory=dict)

    source_ids: list[str] = Field(default_factory=list)
    source_count: int = 0
    backing_ids: list[str] = Field(default_factory=list)
    backing_count: int = 0
    projection_count: int = 0
    target_units: list[str] = Field(default_factory=list)
    backing_units: list[str] = Field(default_factory=list)
    per_path_cocos_labels: dict[str, str | None] = Field(default_factory=dict)
    catalog_cocos: list[int] = Field(default_factory=list)
    predecessor_ids: list[str] = Field(default_factory=list)
    successor_ids: list[str] = Field(default_factory=list)
    accepted_or_protected_lineage_ids: list[str] = Field(default_factory=list)
    protected_source_ids: list[str] = Field(default_factory=list)


def _primitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    if isinstance(value, Decimal | str | int | float | bool) or value is None:
        return value
    return str(value)


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _sorted_strings(values: Any) -> list[str]:
    return sorted({str(value) for value in values or [] if value is not None})


def _claim_fields(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "claimed_at",
        "claim_token",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
    )
    return {field: record.get(field) for field in fields}


def _populate_receipt(receipt: ExactStandardNamePreflightReceipt) -> None:
    evidence = receipt.raw_evidence
    targets = evidence.get("targets") or []
    receipt.identity_count = len(targets)
    receipt.refine_action_count = int(evidence.get("refine_action_count") or 0)
    if receipt.identity_count != 1:
        receipt.diagnostics.append(
            f"exact StandardName identity resolved to {receipt.identity_count} rows"
        )
    else:
        target = targets[0]
        receipt.target_name_stage = target.get("name_stage")
        receipt.target_status = target.get("status")
        receipt.target_validation_status = target.get("validation_status")
        receipt.target_score = _decimal(target.get("reviewer_score_name"))
        receipt.target_chain_length = int(target.get("chain_length") or 0)
        receipt.target_edit_mode = target.get("edit_mode")
        receipt.target_origin = target.get("origin")
        receipt.target_run_id = target.get("run_id")
        receipt.target_claim_fields = _claim_fields(target)

        if receipt.target_name_stage != "reviewed":
            receipt.diagnostics.append("target name stage is not reviewed")
        if receipt.target_score is None:
            receipt.diagnostics.append("target has no name-review score")
        elif receipt.target_score >= receipt.min_score:
            receipt.diagnostics.append("target name-review score is not below minimum")
        if receipt.target_chain_length >= receipt.rotation_cap:
            receipt.diagnostics.append(
                "target refinement chain reached the rotation cap"
            )
        if (
            receipt.target_edit_mode == "rename"
            and int(target.get("review_resubmit_count") or 0) >= receipt.rotation_cap
        ):
            receipt.diagnostics.append("pinned rename exhausted its re-review budget")
        if receipt.target_origin == "derived":
            receipt.diagnostics.append("derived names are structurally fixed")
        if any(value is not None for value in receipt.target_claim_fields.values()):
            receipt.diagnostics.append("target has an active worker or drain lease")

    if receipt.refine_action_count != 1:
        receipt.diagnostics.append(
            f"exact refine_name action cardinality is {receipt.refine_action_count}"
        )

    sources = evidence.get("sources") or []
    receipt.source_ids = _sorted_strings(source.get("id") for source in sources)
    receipt.source_count = len(receipt.source_ids)
    receipt.backing_ids = _sorted_strings(
        source.get("backing_id") for source in sources
    )
    receipt.backing_count = len(receipt.backing_ids)
    receipt.projection_count = sum(
        len(source.get("projection_edge_ids") or []) for source in sources
    )
    receipt.target_units = _sorted_strings(evidence.get("target_unit_ids"))
    receipt.backing_units = _sorted_strings(
        unit
        for source in sources
        for unit in [
            source.get("backing_unit"),
            *(source.get("backing_unit_ids") or []),
        ]
    )
    receipt.per_path_cocos_labels = {
        str(source["backing_id"]): source.get("cocos_label")
        for source in sources
        if source.get("backing_id") is not None
    }

    if receipt.source_count == 0:
        receipt.diagnostics.append("target has no producing source")
    if len(sources) != receipt.source_count:
        receipt.diagnostics.append("producing source identity is ambiguous")
    if receipt.backing_count != receipt.source_count:
        receipt.diagnostics.append("source and backing DD path cardinality differ")
    if receipt.projection_count != receipt.source_count:
        receipt.diagnostics.append("backing DD projection closure is incomplete")

    target = targets[0] if len(targets) == 1 else {}
    target_source_paths = _sorted_strings(target.get("source_paths"))
    source_backing_ids: list[str] = []
    for source in sources:
        backing_id = source.get("backing_id")
        if backing_id is not None:
            source_backing_ids.append(str(backing_id))
        if source.get("source_type") != "dd":
            receipt.diagnostics.append(f"{source.get('id')}: source is not DD-backed")
        if source.get("id") != f"dd:{backing_id}":
            receipt.diagnostics.append(
                f"{source.get('id')}: DD source identity mismatch"
            )
        if source.get("source_id") != backing_id or source.get("dd_path") != backing_id:
            receipt.diagnostics.append(f"{source.get('id')}: backing DD path mismatch")
        if source.get("produced_sn_id") != receipt.name_id:
            receipt.diagnostics.append(
                f"{source.get('id')}: produced-name mirror mismatch"
            )
        if len(source.get("from_dd_edge_ids") or []) != 1:
            receipt.diagnostics.append(
                f"{source.get('id')}: FROM_DD_PATH is not singular"
            )
        if len(source.get("projection_edge_ids") or []) != 1:
            receipt.diagnostics.append(
                f"{source.get('id')}: projection is not singular"
            )
        if any(value is not None for value in _claim_fields(source).values()):
            receipt.diagnostics.append(
                f"{source.get('id')}: source has an active lease"
            )
        if source.get("dd_snapshot_pinned") is not True:
            receipt.diagnostics.append(f"{source.get('id')}: DD snapshot is not pinned")
        if source.get("dd_version") != receipt.dd_version:
            receipt.diagnostics.append(
                f"{source.get('id')}: source DD version is not current"
            )

        dd_unit = source.get("dd_unit")
        backing_unit = source.get("backing_unit")
        raw_backing_unit_ids = source.get("backing_unit_ids") or []
        backing_unit_ids = _sorted_strings(raw_backing_unit_ids)
        backing_unit_edges = source.get("backing_unit_edge_ids") or []
        if dd_unit is None:
            receipt.diagnostics.append(f"{source.get('id')}: source DD unit is missing")
        if backing_unit is None:
            receipt.diagnostics.append(
                f"{source.get('id')}: backing unit property is missing"
            )
        if len(raw_backing_unit_ids) != 1 or raw_backing_unit_ids[0] is None:
            receipt.diagnostics.append(
                f"{source.get('id')}: backing unit relationship is missing or ambiguous"
            )
        if len(backing_unit_ids) != 1 or len(backing_unit_edges) != 1:
            receipt.diagnostics.append(
                f"{source.get('id')}: backing unit is not singular"
            )
        if not (
            dd_unit is not None
            and backing_unit is not None
            and len(raw_backing_unit_ids) == 1
            and raw_backing_unit_ids[0] is not None
            and dd_unit == backing_unit == raw_backing_unit_ids[0]
        ):
            receipt.diagnostics.append(
                f"{source.get('id')}: source/backing unit mismatch"
            )

    if sorted(source_backing_ids) != target_source_paths:
        receipt.diagnostics.append("target source-path mirror differs from DD closure")
    target_unit_edges = evidence.get("target_unit_edge_ids") or []
    raw_target_unit_ids = evidence.get("target_unit_ids") or []
    target_unit = target.get("unit")
    if target_unit is None:
        receipt.diagnostics.append("target unit property is missing")
    if len(raw_target_unit_ids) != 1 or raw_target_unit_ids[0] is None:
        receipt.diagnostics.append("target unit relationship is missing or ambiguous")
    if len(receipt.target_units) != 1 or len(target_unit_edges) != 1:
        receipt.diagnostics.append("target unit relationship is not singular")
    if not (
        target_unit is not None
        and len(raw_target_unit_ids) == 1
        and raw_target_unit_ids[0] is not None
        and target_unit == raw_target_unit_ids[0]
    ):
        receipt.diagnostics.append("target unit property and relationship differ")
    if receipt.target_units and receipt.backing_units != receipt.target_units:
        receipt.diagnostics.append("target and backing DD units differ")

    if target.get("dd_version") != receipt.dd_version:
        receipt.diagnostics.append("target DD version is not current")
    catalogs = evidence.get("catalogs") or []
    catalog_cocos_ids = evidence.get("catalog_cocos_ids") or []
    receipt.catalog_cocos = sorted(
        {
            int(value)
            for value in [
                *(catalog.get("cocos") for catalog in catalogs),
                *catalog_cocos_ids,
            ]
            if value is not None
        }
    )
    if len(catalogs) != 1 or catalogs[0].get("id") != receipt.dd_version:
        receipt.diagnostics.append(
            "configured DD catalog identity is missing or ambiguous"
        )
    elif catalogs[0].get("is_current") is not True:
        receipt.diagnostics.append("configured DD catalog is not current")
    if (
        receipt.catalog_cocos != [17]
        or len(catalog_cocos_ids) != 1
        or len(evidence.get("catalog_cocos_edge_ids") or []) != 1
    ):
        receipt.diagnostics.append("global DD catalog COCOS is not exactly 17")

    labels = list(receipt.per_path_cocos_labels.values())
    unique_labels = set(labels)
    if len(unique_labels) > 1:
        receipt.diagnostics.append("backing DD paths disagree on COCOS label")
    elif unique_labels:
        backing_label = next(iter(unique_labels))
        if target.get("cocos_transformation_type") != backing_label:
            receipt.diagnostics.append(
                "target did not preserve the per-path COCOS label"
            )
        if backing_label is not None and str(target.get("cocos")) != "17":
            receipt.diagnostics.append("COCOS-sensitive target does not carry COCOS 17")

    receipt.predecessor_ids = _sorted_strings(evidence.get("predecessor_ids"))
    receipt.successor_ids = _sorted_strings(evidence.get("successor_ids"))
    receipt.accepted_or_protected_lineage_ids = _sorted_strings(
        evidence.get("accepted_or_protected_lineage_ids")
    )
    receipt.protected_source_ids = _sorted_strings(evidence.get("protected_source_ids"))
    if len(receipt.predecessor_ids) > 1:
        receipt.diagnostics.append("target has ambiguous refinement predecessors")
    if receipt.successor_ids:
        receipt.diagnostics.append("target already has a refined successor")
    if receipt.accepted_or_protected_lineage_ids:
        receipt.diagnostics.append(
            "refinement lineage intersects accepted or protected state"
        )
    if receipt.protected_source_ids:
        receipt.diagnostics.append(
            "structural lineage intersects WEST or fixture sources"
        )

    if receipt.requested_cost_ceiling <= 0:
        receipt.diagnostics.append("requested cost ceiling must be positive")
    if receipt.cumulative_spend < 0 or receipt.authorized_budget < 0:
        receipt.diagnostics.append("budget values must be non-negative")
    if receipt.budget_remaining_after < 0:
        receipt.diagnostics.append("requested cost ceiling exceeds authorized budget")

    receipt.passed = not receipt.diagnostics


def audit_exact_standard_name_preflight(
    name_id: str,
    *,
    requested_cost_ceiling: Decimal | float | str,
    cumulative_spend: Decimal | float | str,
    authorized_budget: Decimal | float | str,
    min_score: Decimal | float | str = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    dd_version: str | None = None,
    gc: GraphClient | None = None,
) -> ExactStandardNamePreflightReceipt:
    """Return one-query evidence for a paid exact ``refine_name`` decision.

    ``run_id`` and ``last_run_id`` are returned as durable provenance but never
    interpreted as leases. Only worker and bounded-drain claim fields can block.
    A query failure is represented in the receipt rather than escaping without
    a machine-readable refusal.
    """
    normalized_name = name_id.strip()
    configured_dd = (dd_version or get_dd_version()).strip()
    if not normalized_name:
        raise ValueError("name_id is required")
    if not configured_dd:
        raise ValueError("dd_version is required")
    if rotation_cap <= 0:
        raise ValueError("rotation_cap must be positive")

    ceiling = Decimal(str(requested_cost_ceiling))
    spent = Decimal(str(cumulative_spend))
    budget = Decimal(str(authorized_budget))
    threshold = Decimal(str(min_score))
    if not all(value.is_finite() for value in (ceiling, spent, budget, threshold)):
        raise ValueError("cost and score inputs must be finite")
    receipt = ExactStandardNamePreflightReceipt(
        name_id=normalized_name,
        dd_version=configured_dd,
        min_score=threshold,
        rotation_cap=rotation_cap,
        requested_cost_ceiling=ceiling,
        cumulative_spend=spent,
        authorized_budget=budget,
        budget_remaining_before=budget - spent,
        budget_remaining_after=budget - spent - ceiling,
    )

    from imas_codex.standard_names.grammar_segment_reconciliation import (
        _FIXTURE_SOURCE_ID_PREFIX,
        _west_source_ids,
    )

    params = {
        "name_id": normalized_name,
        "dd_version": configured_dd,
        "min_score": float(threshold),
        "rotation_cap": rotation_cap,
        "west_source_ids": sorted(_west_source_ids()),
        "fixture_source_id_prefix": _FIXTURE_SOURCE_ID_PREFIX,
    }
    manager = nullcontext(gc) if gc is not None else _graph_client()
    with manager as client:
        assert client is not None
        receipt.query_count = 1
        try:
            rows = client.query(_EXACT_STANDARD_NAME_PREFLIGHT_QUERY, **params)
        except Exception as exc:
            receipt.diagnostics.append(
                f"preflight evidence query failed: {type(exc).__name__}: {exc}"
            )
        else:
            primitive_rows = _primitive(rows)
            receipt.raw_evidence = (
                primitive_rows[0]
                if len(primitive_rows) == 1
                else {"returned_rows": primitive_rows}
            )
            if len(primitive_rows) != 1:
                receipt.diagnostics.append(
                    f"preflight query returned {len(primitive_rows)} receipt rows"
                )

    _populate_receipt(receipt)
    return receipt


def _graph_client() -> GraphClient:
    from imas_codex.graph.client import GraphClient

    return GraphClient()


__all__ = [
    "ExactStandardNamePreflightReceipt",
    "audit_exact_standard_name_preflight",
]

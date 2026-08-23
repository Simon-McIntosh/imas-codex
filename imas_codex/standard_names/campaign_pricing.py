"""Fail-closed, model-free pricing for name-review campaigns.

The projector renders the production name-review request for every supplied
identity and prices the complete reviewer chain before any dispatch is
possible.  The first two reviewer seats are mandatory; a third seat is priced
as worst-case disagreement escalation.  A caller also supplies a bound on the
number of identities that may enter refinement.  Those identities are selected
by highest projected refinement cost, so the bound cannot be made optimistic by
cohort ordering.

This module deliberately contains no model-call function.  Graph reads used to
assemble production prompt context are allowed; provider calls and graph writes
are not.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imas_codex.llm.prompt_loader import load_prompt_config, render_prompt
from imas_codex.standard_names.budget import (
    BudgetExposureUnknown,
    model_provider_exposure,
)
from imas_codex.standard_names.models import (
    RefinedName,
    StandardNameQualityReviewNameOnlyBatch,
)


class CampaignCostCeilingExceeded(RuntimeError):
    """Raised before dispatch when a campaign exceeds its spending authority."""


class CampaignPricingUnknown(RuntimeError):
    """Raised when any paid route or request cannot be priced safely."""


@dataclass(frozen=True)
class CampaignPricingPolicy:
    """Bounded production policy used for one campaign projection."""

    reviewer_models: tuple[str, ...]
    refine_model: str
    refine_escalation_model: str
    refinement_rotations: int
    max_refinement_names: int
    provider_attempts: int = 1
    escalation_critique_chars: int = 4_000
    refinement_feedback_chars: int = 800
    refinement_context_chars: int = 8_000
    fanout_enabled: bool = False
    fanout_baseline_cost_cap_usd: float = 0.0
    fanout_escalation_cost_cap_usd: float = 0.0

    @classmethod
    def production(
        cls,
        *,
        cohort_size: int,
        max_refinement_names: int | None = None,
    ) -> CampaignPricingPolicy:
        """Read the live names-review, refine, escalation, and fan-out policy."""
        from imas_codex.settings import get_model, get_sn_review_names_models
        from imas_codex.standard_names.defaults import DEFAULT_REFINE_ROTATIONS
        from imas_codex.standard_names.fanout import load_settings

        models = tuple(get_sn_review_names_models())
        if not models:
            raise CampaignPricingUnknown("names-review policy has no reviewer seat")
        fanout = load_settings()
        fanout_enabled = bool(fanout.enabled and fanout.sites.get("refine_name"))
        return cls(
            reviewer_models=models,
            refine_model=get_model("sn-refine"),
            refine_escalation_model=get_model("sn-escalation"),
            refinement_rotations=DEFAULT_REFINE_ROTATIONS,
            max_refinement_names=(
                cohort_size if max_refinement_names is None else max_refinement_names
            ),
            refinement_context_chars=4 * fanout.evidence_token_cap_baseline,
            fanout_enabled=fanout_enabled,
            fanout_baseline_cost_cap_usd=(
                fanout.fanout_max_charge_per_cycle_baseline if fanout_enabled else 0.0
            ),
            fanout_escalation_cost_cap_usd=(
                fanout.fanout_max_charge_per_cycle_escalation if fanout_enabled else 0.0
            ),
        )


@dataclass(frozen=True)
class CampaignCostLine:
    """One independently attributable component of campaign exposure."""

    phase: str
    call_count: int
    projected_cost_usd: float
    conditional: bool


@dataclass(frozen=True)
class CampaignProjection:
    """A zero-call receipt for the maximum bounded campaign exposure."""

    cohort_size: int
    projected_call_count: int
    projected_cost_usd: float
    mandatory_call_count: int
    mandatory_cost_usd: float
    conditional_call_count: int
    conditional_cost_usd: float
    refinement_name_bound: int
    lines: tuple[CampaignCostLine, ...]


@dataclass(frozen=True)
class CampaignProjectionScenario:
    """One ceiling decision with normalized per-call and per-name exposure."""

    projection: CampaignProjection
    cost_ceiling_usd: float
    projected_cost_per_call_usd: float
    projected_cost_per_name_usd: float
    admitted: bool


@dataclass(frozen=True)
class CampaignProjectionRange:
    """Minimum-escalation and bounded worst-case decisions for one cohort."""

    minimum_escalation: CampaignProjectionScenario
    worst_case: CampaignProjectionScenario


@dataclass(frozen=True)
class ExactCampaignCohorts:
    """Live exact identities and the read-only counter proof around resolution."""

    catalog_import_candidates: int
    catalog_import_recovered_ids: tuple[str, ...]
    catalog_import_rescore: tuple[dict[str, Any], ...]
    redraw: tuple[dict[str, Any], ...]
    standard_name_count_before: int
    standard_name_count_after: int


@dataclass(frozen=True)
class _PreparedNameRequest:
    """One fully rendered production review request and its enriched item."""

    item: dict[str, Any]
    base_messages: list[dict[str, Any]]
    escalation_messages: list[dict[str, Any]]


_ARCHIVE_RECOVERED_CATALOG_IDS = frozenset(
    {"normalized_collisionality", "thermal_ion_density"}
)

_COHORT_RETURN = """
RETURN sn.id AS id,
       sn.name AS name,
       sn.description AS description,
       sn.documentation AS documentation,
       sn.kind AS kind,
       sn.unit AS unit,
       sn.tags AS tags,
       sn.physics_domain AS physics_domain,
       coalesce(sn.chain_length, 0) AS chain_length,
       sn.name_stage AS name_stage,
       sn.origin AS origin,
       sn.edit_mode AS edit_mode,
       sn.name_hint AS name_hint,
       sn.docs_hint AS docs_hint,
       sn.edit_reason AS edit_reason,
       sn.edit_origin AS edit_origin,
       sn.physical_base AS physical_base,
       sn.geometry AS geometry,
       sn.grammar_parse_version AS grammar_parse_version,
       sn.source_paths AS source_paths,
       [(source:StandardNameSource)-[:PRODUCED_NAME]->(sn) | {
           id: source.id,
           source_type: source.source_type,
           source_id: source.source_id,
           status: source.status,
           description: source.description,
           physics_domain: source.physics_domain,
           compose_hint: source.compose_hint,
           compose_hint_reason: source.compose_hint_reason,
           dd_path: source.dd_path,
           dd_version: source.dd_version,
           dd_documentation: source.dd_documentation,
           dd_snapshot_pinned: source.dd_snapshot_pinned,
           dd_parent_path: source.dd_parent_path,
           dd_parent_documentation: source.dd_parent_documentation,
           dd_data_type: source.dd_data_type,
           dd_unit: source.dd_unit,
           dd_coordinates: source.dd_coordinates,
           dd_lifecycle_status: source.dd_lifecycle_status,
           enhanced_description: source.enhanced_description
       }] AS source_bindings
ORDER BY sn.id
"""


def redraw_identities_from_census(census_path: str | Path) -> tuple[str, ...]:
    """Read the complete redraw-eligible identity table from its evidence file."""
    text = Path(census_path).read_text(encoding="utf-8")
    match = re.search(
        r"^### redraw-eligible \((\d+)\)\s*$\n(?P<table>.*?)(?=^### )",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise CampaignPricingUnknown("redraw-eligible census section is absent")
    identities = tuple(
        row_match.group(1)
        for row in match.group("table").splitlines()
        if (row_match := re.match(r"^\| `([^`]+)` \|", row)) is not None
    )
    expected_count = int(match.group(1))
    if len(identities) != expected_count or len(set(identities)) != expected_count:
        raise CampaignPricingUnknown(
            "redraw-eligible census identities do not match its declared count"
        )
    return identities


def resolve_exact_campaign_cohorts(
    redraw_census_path: str | Path,
) -> ExactCampaignCohorts:
    """Resolve both review cohorts live while proving the graph count unchanged."""
    from imas_codex.graph.client import GraphClient

    redraw_ids = redraw_identities_from_census(redraw_census_path)
    with GraphClient() as gc:
        before_rows = gc.query(
            "MATCH (sn:StandardName) RETURN count(sn) AS standard_name_count"
        )
        catalog_rows = list(
            gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.name_stage = 'accepted'
                  AND sn.validation_status = 'valid'
                  AND sn.docs_stage = 'accepted'
                  AND sn.origin = 'catalog_edit'
                  AND sn.reviewer_model_name IS NULL
                  AND NOT EXISTS {
                    MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
                    WHERE review.review_axis = 'names'
                  }
                """
                + _COHORT_RETURN
            )
        )
        redraw_rows = list(
            gc.query(
                "MATCH (sn:StandardName) WHERE sn.id IN $identities\n" + _COHORT_RETURN,
                identities=list(redraw_ids),
            )
        )
        after_rows = gc.query(
            "MATCH (sn:StandardName) RETURN count(sn) AS standard_name_count"
        )

    if not before_rows or not after_rows:
        raise CampaignPricingUnknown("StandardName counter query returned no row")
    before = int(before_rows[0]["standard_name_count"])
    after = int(after_rows[0]["standard_name_count"])
    if before != after:
        raise CampaignPricingUnknown(
            f"StandardName count changed during cohort resolution: {before} -> {after}"
        )

    catalog_by_id = {str(row["id"]): row for row in catalog_rows}
    if len(catalog_by_id) != len(catalog_rows):
        raise CampaignPricingUnknown(
            "catalog-import cohort contains duplicate identities"
        )
    recovered_present = tuple(
        sorted(_ARCHIVE_RECOVERED_CATALOG_IDS.intersection(catalog_by_id))
    )
    catalog_rescore = tuple(
        catalog_by_id[identity]
        for identity in sorted(catalog_by_id)
        if identity not in _ARCHIVE_RECOVERED_CATALOG_IDS
    )

    redraw_by_id = {str(row["id"]): row for row in redraw_rows}
    missing_redraw = sorted(set(redraw_ids).difference(redraw_by_id))
    if missing_redraw:
        raise CampaignPricingUnknown(
            "redraw census identities missing from the live graph: "
            + ", ".join(missing_redraw)
        )
    if set(catalog_by_id).intersection(redraw_by_id):
        raise CampaignPricingUnknown("catalog-import and redraw cohorts overlap")

    return ExactCampaignCohorts(
        catalog_import_candidates=len(catalog_rows),
        catalog_import_recovered_ids=recovered_present,
        catalog_import_rescore=catalog_rescore,
        redraw=tuple(redraw_by_id[identity] for identity in redraw_ids),
        standard_name_count_before=before,
        standard_name_count_after=after,
    )


def _load_review_context(item: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """Load the neighbour and scored-example context used by production review."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.context import fetch_review_neighbours
    from imas_codex.standard_names.example_loader import load_review_examples

    neighbours = fetch_review_neighbours(item)
    with GraphClient() as gc:
        examples = load_review_examples(
            gc,
            physics_domains=[item.get("physics_domain") or ""],
            axis="name",
        )
    return neighbours, examples


def _prior_review_payload(char_count: int) -> list[dict[str, Any]]:
    """Return a bounded two-review payload for the production escalator prompt."""
    critique = "x" * char_count
    scores = {
        "grammar": 10,
        "semantic": 10,
        "convention": 10,
        "completeness": 10,
    }
    return [
        {
            "role": role,
            "model": model,
            "score": 0.5,
            "scores": scores,
            "comments": critique,
            "comments_per_dim": dict.fromkeys(scores, critique),
        }
        for role, model in (("primary", "reviewer-0"), ("secondary", "reviewer-1"))
    ]


def _prepare_name_requests(
    cohort: Sequence[dict[str, Any]],
    *,
    escalation_critique_chars: int,
) -> list[_PreparedNameRequest]:
    """Render production-enriched name-review requests for the exact cohort."""
    from imas_codex.standard_names.context import (
        _build_enum_lists,
        build_compose_context,
    )
    from imas_codex.standard_names.workers import _enrich_name_review_items

    items = [dict(item) for item in cohort]
    _enrich_name_review_items(items)
    compose_context = build_compose_context()
    enum_lists = _build_enum_lists()
    prepared: list[_PreparedNameRequest] = []
    for item in items:
        if not item.get("id"):
            raise CampaignPricingUnknown("campaign item has no standard-name identity")
        neighbours, examples = _load_review_context(item)
        context = {
            **compose_context,
            "items": [item],
            **neighbours,
            **enum_lists,
            "review_scored_examples": examples,
            "prior_reviews": [],
        }
        system_prompt = render_prompt("sn/review_names_system", context)
        user_prompt = render_prompt("sn/review_names_user", context)
        escalation_context = {
            **context,
            "prior_reviews": _prior_review_payload(escalation_critique_chars),
        }
        escalation_user_prompt = render_prompt(
            "sn/review_names_user", escalation_context
        )
        prepared.append(
            _PreparedNameRequest(
                item=item,
                base_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                escalation_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": escalation_user_prompt},
                ],
            )
        )
    return prepared


def _refinement_messages(
    item: dict[str, Any],
    *,
    feedback_chars: int,
    context_chars: int,
) -> list[dict[str, Any]]:
    """Render a bounded production refine request from the enriched review item."""
    from imas_codex.standard_names.context import build_compose_context

    feedback = "x" * feedback_chars
    refine_item = {
        **item,
        "reviewer_comments_name": feedback,
        "reviewer_comments_per_dim_name": json.dumps(
            dict.fromkeys(
                ("grammar", "semantic", "convention", "completeness"), feedback
            ),
            sort_keys=True,
        ),
    }
    try:
        composition_rules = load_prompt_config("sn_composition_rules").get(
            "composition_rules", []
        )
    except Exception as exc:
        raise CampaignPricingUnknown(
            "refinement composition rules unavailable"
        ) from exc
    context = {
        **build_compose_context(),
        "item": refine_item,
        "chain_history": list(item.get("chain_history") or []),
        "chain_length": int(item.get("chain_length") or 0),
        "hybrid_neighbours": list(item.get("semantic_comparators") or []),
        "fanout_evidence": "x" * context_chars,
        "compose_scored_examples": list(item.get("compose_scored_examples") or []),
        "vocab_gap_detail": item.get("vocab_gap_detail"),
        "validation_issues": item.get("validation_issues") or None,
        "composition_rules": composition_rules,
    }
    system_prompt = render_prompt("sn/refine_name_system", context)
    user_prompt = render_prompt("sn/refine_name_user", context)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _price_request(
    model: str,
    messages: list[dict[str, Any]],
    *,
    response_model: type[Any],
    provider_attempts: int,
) -> float:
    """Price one rendered request, preserving fail-closed pricing semantics."""
    try:
        cost = model_provider_exposure(
            model,
            messages,
            response_model=response_model,
            provider_attempts=provider_attempts,
        )
    except (BudgetExposureUnknown, ValueError, KeyError) as exc:
        raise CampaignPricingUnknown(f"unpriced campaign route {model}") from exc
    if not math.isfinite(cost) or cost <= 0:
        raise CampaignPricingUnknown(f"campaign route {model} projected no cost")
    return cost


def _validate_policy(policy: CampaignPricingPolicy, cohort_size: int) -> None:
    if not policy.reviewer_models or len(policy.reviewer_models) > 3:
        raise CampaignPricingUnknown("names-review policy requires one to three seats")
    if policy.provider_attempts < 1:
        raise CampaignPricingUnknown("provider attempts are not positively bounded")
    if policy.refinement_rotations < 0:
        raise CampaignPricingUnknown("refinement rotations cannot be negative")
    if not 0 <= policy.max_refinement_names <= cohort_size:
        raise CampaignPricingUnknown("refinement-name bound is outside the cohort")
    numeric_bounds = (
        policy.escalation_critique_chars,
        policy.refinement_feedback_chars,
        policy.refinement_context_chars,
    )
    if any(value < 0 for value in numeric_bounds):
        raise CampaignPricingUnknown("prompt payload bounds cannot be negative")
    if policy.fanout_enabled and (
        policy.fanout_baseline_cost_cap_usd <= 0
        or policy.fanout_escalation_cost_cap_usd <= 0
    ):
        raise CampaignPricingUnknown("enabled refinement fan-out has no positive cap")


def project_name_review_campaign(
    cohort: Sequence[dict[str, Any]],
    *,
    cost_ceiling_usd: float | None = None,
    policy: CampaignPricingPolicy | None = None,
) -> CampaignProjection:
    """Project and admit a names-axis campaign without issuing model calls.

    ``max_refinement_names`` is a maximum population, not a sampled fraction.
    The projector prices refinement for the most expensive rendered requests in
    the cohort.  The third reviewer seat and every refinement/fan-out line are
    conditional exposure, but they are included in the admission total.
    """
    cohort_items = tuple(cohort)
    if not cohort_items:
        raise CampaignPricingUnknown("campaign cohort is empty")
    active_policy = policy or CampaignPricingPolicy.production(
        cohort_size=len(cohort_items)
    )
    _validate_policy(active_policy, len(cohort_items))
    if cost_ceiling_usd is not None and (
        not math.isfinite(cost_ceiling_usd) or cost_ceiling_usd < 0
    ):
        raise ValueError("cost ceiling must be finite and non-negative")

    prepared = _prepare_name_requests(
        cohort_items,
        escalation_critique_chars=active_policy.escalation_critique_chars,
    )
    mandatory_review_cost = 0.0
    escalation_review_cost = 0.0
    base_models = active_policy.reviewer_models[:2]
    escalation_models = active_policy.reviewer_models[2:3]
    for request in prepared:
        mandatory_review_cost += sum(
            _price_request(
                model,
                request.base_messages,
                response_model=StandardNameQualityReviewNameOnlyBatch,
                provider_attempts=active_policy.provider_attempts,
            )
            for model in base_models
        )
        escalation_review_cost += sum(
            _price_request(
                model,
                request.escalation_messages,
                response_model=StandardNameQualityReviewNameOnlyBatch,
                provider_attempts=active_policy.provider_attempts,
            )
            for model in escalation_models
        )

    per_item_refinement: list[float] = []
    for request in prepared:
        messages = _refinement_messages(
            request.item,
            feedback_chars=active_policy.refinement_feedback_chars,
            context_chars=active_policy.refinement_context_chars,
        )
        rotations = active_policy.refinement_rotations
        base_rotations = max(0, rotations - 1)
        refinement_cost = 0.0
        if base_rotations:
            refinement_cost += (
                _price_request(
                    active_policy.refine_model,
                    messages,
                    response_model=RefinedName,
                    provider_attempts=active_policy.provider_attempts,
                )
                * base_rotations
            )
        if rotations:
            refinement_cost += _price_request(
                active_policy.refine_escalation_model,
                messages,
                response_model=RefinedName,
                provider_attempts=active_policy.provider_attempts,
            )
        per_item_refinement.append(refinement_cost)
    refinement_model_cost = sum(
        sorted(per_item_refinement, reverse=True)[: active_policy.max_refinement_names]
    )

    refinement_calls = (
        active_policy.max_refinement_names * active_policy.refinement_rotations
    )
    fanout_calls = refinement_calls if active_policy.fanout_enabled else 0
    fanout_cost = 0.0
    if active_policy.fanout_enabled and active_policy.refinement_rotations:
        fanout_cost = active_policy.max_refinement_names * (
            max(0, active_policy.refinement_rotations - 1)
            * active_policy.fanout_baseline_cost_cap_usd
            + active_policy.fanout_escalation_cost_cap_usd
        )

    lines = (
        CampaignCostLine(
            phase="review_base",
            call_count=len(prepared) * len(base_models),
            projected_cost_usd=mandatory_review_cost,
            conditional=False,
        ),
        CampaignCostLine(
            phase="review_escalation",
            call_count=len(prepared) * len(escalation_models),
            projected_cost_usd=escalation_review_cost,
            conditional=True,
        ),
        CampaignCostLine(
            phase="refine_name",
            call_count=refinement_calls,
            projected_cost_usd=refinement_model_cost,
            conditional=True,
        ),
        CampaignCostLine(
            phase="refine_fanout",
            call_count=fanout_calls,
            projected_cost_usd=fanout_cost,
            conditional=True,
        ),
    )
    mandatory_calls = lines[0].call_count
    conditional_calls = sum(line.call_count for line in lines[1:])
    conditional_cost = sum(line.projected_cost_usd for line in lines[1:])
    total_cost = mandatory_review_cost + conditional_cost
    if not math.isfinite(total_cost) or total_cost <= 0:
        raise CampaignPricingUnknown("campaign projection is not finite and positive")
    projection = CampaignProjection(
        cohort_size=len(prepared),
        projected_call_count=mandatory_calls + conditional_calls,
        projected_cost_usd=total_cost,
        mandatory_call_count=mandatory_calls,
        mandatory_cost_usd=mandatory_review_cost,
        conditional_call_count=conditional_calls,
        conditional_cost_usd=conditional_cost,
        refinement_name_bound=active_policy.max_refinement_names,
        lines=lines,
    )
    if cost_ceiling_usd is not None and total_cost > cost_ceiling_usd:
        raise CampaignCostCeilingExceeded(
            f"projected names-review campaign ${total_cost:.6f} is above the "
            f"${cost_ceiling_usd:.6f} ceiling"
        )
    return projection


def _scenario(
    projection: CampaignProjection,
    *,
    cost_ceiling_usd: float,
) -> CampaignProjectionScenario:
    return CampaignProjectionScenario(
        projection=projection,
        cost_ceiling_usd=cost_ceiling_usd,
        projected_cost_per_call_usd=(
            projection.projected_cost_usd / projection.projected_call_count
        ),
        projected_cost_per_name_usd=(
            projection.projected_cost_usd / projection.cohort_size
        ),
        admitted=projection.projected_cost_usd <= cost_ceiling_usd,
    )


def project_name_review_campaign_range(
    cohort: Sequence[dict[str, Any]],
    *,
    cost_ceiling_usd: float,
    policy: CampaignPricingPolicy | None = None,
) -> CampaignProjectionRange:
    """Price minimum escalation and bounded worst case from one exact render.

    Minimum escalation includes every configured reviewer seat but no refine or
    refine fan-out calls.  With the production three-seat chain this is three
    calls per name.  Worst case includes the full bounded refinement exposure.
    """
    if not math.isfinite(cost_ceiling_usd) or cost_ceiling_usd < 0:
        raise ValueError("cost ceiling must be finite and non-negative")
    worst = project_name_review_campaign(cohort, policy=policy)
    review_lines = worst.lines[:2]
    minimum_calls = sum(line.call_count for line in review_lines)
    minimum_cost = sum(line.projected_cost_usd for line in review_lines)
    minimum = CampaignProjection(
        cohort_size=worst.cohort_size,
        projected_call_count=minimum_calls,
        projected_cost_usd=minimum_cost,
        mandatory_call_count=worst.mandatory_call_count,
        mandatory_cost_usd=worst.mandatory_cost_usd,
        conditional_call_count=review_lines[1].call_count,
        conditional_cost_usd=review_lines[1].projected_cost_usd,
        refinement_name_bound=0,
        lines=review_lines,
    )
    return CampaignProjectionRange(
        minimum_escalation=_scenario(minimum, cost_ceiling_usd=cost_ceiling_usd),
        worst_case=_scenario(worst, cost_ceiling_usd=cost_ceiling_usd),
    )


def projection_as_json(projection: CampaignProjection) -> str:
    """Return a stable machine-readable projection receipt."""
    return json.dumps(
        {
            "cohort_size": projection.cohort_size,
            "projected_call_count": projection.projected_call_count,
            "projected_cost_usd": projection.projected_cost_usd,
            "mandatory_call_count": projection.mandatory_call_count,
            "mandatory_cost_usd": projection.mandatory_cost_usd,
            "conditional_call_count": projection.conditional_call_count,
            "conditional_cost_usd": projection.conditional_cost_usd,
            "refinement_name_bound": projection.refinement_name_bound,
            "lines": [line.__dict__ for line in projection.lines],
        },
        sort_keys=True,
        separators=(",", ":"),
    )

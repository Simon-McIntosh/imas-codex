"""Compose member descriptions from one component and one field-pattern description.

Groups come from the archive grouping already stored on ``PlantComponent`` and
``ArchiveFieldPattern``; context comes from the plant-context import's rows,
function-category meanings, plant structures and document references. The
component description is written from the context linked to its member signals,
the pattern description says what the pattern's field codes mean where a linked
row states the meaning, and each member description is composed from the two
together with the closed vocabulary section 11 decided for that signal.

A signal with no linked context is never described from its name: it is marked
``awaiting_context`` and counted. A member of a component whose description
failed is marked ``failed`` and carries no composed description, so a group
failure cannot leak text into a member.

Member descriptions are written through the discovery pipeline's single
description writer, which stores the text, its vector and its hash together and
carries the member's status and cost update in the same statement.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.discovery.signals.enrichment import (
    context_items,
    filter_statements,
    get_enrichment_options,
)
from imas_codex.discovery.signals.vocabulary import (
    QUANTITY_TERMS,
    ROLE_TERMS,
    UNIT_SPELLINGS,
)
from imas_codex.graph.client import GraphClient
from imas_codex.llm.prompt_loader import render_prompt

DESCRIBED = "described"
AWAITING_CONTEXT = "awaiting_context"
FAILED = "failed"

GROUP_KINDS = ("component", "field_pattern")

VOCABULARY_MEMBERS: dict[str, frozenset[str]] = {
    "quantity": frozenset(QUANTITY_TERMS),
    "unit": frozenset(UNIT_SPELLINGS),
    "role": frozenset(ROLE_TERMS),
}

COHORT_QUERY = """
MATCH (s:FacilitySignal {facility_id: $facility})
WHERE s.plant_context_state IS NOT NULL
RETURN s.id AS id, s.accessor AS accessor, s.status AS status,
       s.plant_context_state AS context_state,
       s.archive_component AS component,
       s.archive_control_node AS control_node,
       s.field_code AS field_code,
       s.vocabulary_quantity AS quantity,
       s.vocabulary_quantity_state AS quantity_state,
       s.vocabulary_unit AS unit,
       s.vocabulary_unit_state AS unit_state,
       s.vocabulary_role AS role,
       s.vocabulary_role_state AS role_state
ORDER BY s.id
"""

COMPONENT_QUERY = """
MATCH (c:PlantComponent {id: $component, facility_id: $facility})
OPTIONAL MATCH (c)-[:COMPONENT_FUNCTION_CATEGORY]->(category:PlantFunctionCategory)
OPTIONAL MATCH (c)-[:COMPONENT_PLANT_STRUCTURE]->(structure:PlantStructure)
RETURN c.code AS code, c.component_category AS category_code,
       c.component_sequence AS sequence,
       c.archive_field_pattern AS field_pattern,
       c.archive_category_pattern AS category_pattern,
       category.description AS function_category,
       category.code AS function_category_code,
       structure.description AS plant_structure
"""

COMPONENT_CONTEXT_QUERY = """
MATCH (s:FacilitySignal {facility_id: $facility})
      -[:IN_ARCHIVE_COMPONENT]->(c:PlantComponent {id: $component})
MATCH (s)-[:HAS_PLANT_CONTEXT]->(:SignalContextLink)
      -[:CONTEXT_SOURCE_ROW]->(r:PlantSignalRow)
OPTIONAL MATCH (r)-[:FROM_PLANT_DOCUMENT]->(d:PlantContextDocument)
RETURN DISTINCT r.id AS row_id, r.description AS description,
       r.field_code AS row_field, s.field_code AS signal_field,
       d.description AS document, d.role AS document_role
ORDER BY row_id
"""

PATTERN_QUERY = """
MATCH (p:ArchiveFieldPattern {id: $pattern, facility_id: $facility})
RETURN p.fields AS fields, p.component_category AS category_code,
       p.key_kind AS key_kind
"""

PATTERN_CONTEXT_QUERY = """
MATCH (p:ArchiveFieldPattern {id: $pattern, facility_id: $facility})
      <-[:HAS_ARCHIVE_FIELD_PATTERN]-(c:PlantComponent)
      <-[:IN_ARCHIVE_COMPONENT]-(s:FacilitySignal {facility_id: $facility})
MATCH (s)-[:HAS_PLANT_CONTEXT]->(:SignalContextLink)
      -[:CONTEXT_SOURCE_ROW]->(r:PlantSignalRow)
RETURN DISTINCT r.field_code AS row_field, r.description AS description,
       s.field_code AS signal_field
ORDER BY row_field, description
"""

PILOT_COMPONENTS_QUERY = """
MATCH (c:PlantComponent {facility_id: $facility})
WHERE c.id IN $components
OPTIONAL MATCH (c)-[:HAS_ARCHIVE_FIELD_PATTERN]->(p:ArchiveFieldPattern)
RETURN c.id AS component, p.id AS pattern
"""


COMPONENT_DESCRIPTION_WRITE = """
MATCH (c:PlantComponent {id: $component, facility_id: $facility})
SET c.description = $text,
    c.description_source = 'group_context',
    c.description_updated_at = datetime()
RETURN count(c) AS written
"""

PATTERN_DESCRIPTION_WRITE = """
MATCH (p:ArchiveFieldPattern {id: $pattern, facility_id: $facility})
SET p.description = $text,
    p.description_source = 'group_context',
    p.description_updated_at = datetime()
RETURN count(p) AS written
"""

GROUP_STATE_WRITE = """
UNWIND $rows AS row
MATCH (s:FacilitySignal {id: row.id, facility_id: $facility})
SET s.group_enrichment_state = row.group_enrichment_state,
    s.group_enrichment_reason = row.group_enrichment_reason
FOREACH (_ IN CASE WHEN row.status IS NULL THEN [] ELSE [1] END |
    SET s.status = row.status, s.claimed_at = null, s.claim_token = null
)
RETURN count(s) AS written
"""


@dataclass(frozen=True)
class PilotMember:
    """One pilot signal with the context and vocabulary it already carries."""

    id: str
    accessor: str
    status: str | None
    context_state: str
    component: str | None
    control_node: str | None
    field_code: str | None
    quantity: str | None
    quantity_state: str | None
    unit: str | None
    unit_state: str | None
    role: str | None
    role_state: str | None

    @property
    def has_context(self) -> bool:
        return self.context_state in ("exact", "derived")


@dataclass(frozen=True)
class GroupRequest:
    """One group the model describes from its supplied context items."""

    group_id: str
    kind: str
    context: tuple[str, ...]


@dataclass(frozen=True)
class GroupOutcome:
    """One call's delivered description, its cost and its refusals."""

    text: str = ""
    fields: Mapping[str, str] = field(default_factory=dict)
    cost: float = 0.0
    tokens: int = 0
    drops: int = 0
    calls: int = 0
    error: str | None = None

    @property
    def delivered(self) -> bool:
        return bool(self.text.strip()) and self.error is None


@dataclass
class GroupPlan:
    """The pilot cohort, its groups and the members each group reaches."""

    members: list[PilotMember]
    components: dict[str, GroupRequest]
    patterns: dict[str, GroupRequest]
    component_pattern: dict[str, str]
    component_members: dict[str, list[PilotMember]]
    pattern_members: dict[str, list[PilotMember]]
    all_components: dict[str, list[PilotMember]] = field(default_factory=dict)
    all_patterns: dict[str, str] = field(default_factory=dict)


def decisions(member: PilotMember) -> dict[str, str]:
    """The closed-vocabulary values this signal's sources decided."""
    decided = {}
    for axis, value, state in (
        ("quantity", member.quantity, member.quantity_state),
        ("unit", member.unit, member.unit_state),
        ("role", member.role, member.role_state),
    ):
        if state == "decided" and value in VOCABULARY_MEMBERS[axis]:
            decided[axis] = value
    return decided


def compose_member_description(
    component_text: str,
    pattern_text: str,
    field_meaning: str,
    member: PilotMember,
) -> str:
    """Compose one member description from the two group descriptions.

    Both group descriptions are written from linked context; the member's own
    closed-vocabulary values are stated only where a source decided them, so an
    undecided or conflicting axis contributes nothing.
    """
    field_sentence = (field_meaning or pattern_text).strip()
    parts = [component_text.strip(), field_sentence]
    decided = decisions(member)
    if decided:
        parts.append(
            " ".join(
                f"{axis.capitalize()}: {value}." for axis, value in decided.items()
            )
        )
    return " ".join(part for part in parts if part).strip()


@dataclass
class ComposedPlan:
    """What one run writes, and the counts its landing record reports."""

    component_texts: dict[str, str]
    pattern_texts: dict[str, str]
    described: list[dict[str, Any]]
    awaiting: list[dict[str, Any]]
    failed: list[dict[str, Any]]
    component_states: dict[str, str]
    pattern_states: dict[str, str]
    calls: int = 0
    cost: float = 0.0
    tokens: int = 0
    drops: int = 0


def compose_plan(
    plan: GroupPlan,
    component_outcomes: Mapping[str, GroupOutcome],
    pattern_outcomes: Mapping[str, GroupOutcome],
) -> ComposedPlan:
    """Decide every group's and member's state from the calls that came back.

    A component whose description failed fails every member it groups, whatever
    its pattern returned: a member cannot be composed from a description that
    was never delivered. The same holds for a pattern, because a member is
    composed from the two.
    """
    component_texts = {
        group_id: outcome.text.strip()
        for group_id, outcome in component_outcomes.items()
        if outcome.delivered
    }
    pattern_texts = {
        group_id: outcome.text.strip()
        for group_id, outcome in pattern_outcomes.items()
        if outcome.delivered
    }
    pattern_fields = {
        group_id: dict(outcome.fields)
        for group_id, outcome in pattern_outcomes.items()
        if outcome.delivered
    }
    component_states, pattern_states = {}, {}
    described, awaiting, failed = [], [], []

    for group_id in plan.components:
        component_states[group_id] = (
            DESCRIBED if group_id in component_texts else FAILED
        )
    for group_id in plan.patterns:
        pattern_states[group_id] = DESCRIBED if group_id in pattern_texts else FAILED

    for member in plan.members:
        if not member.has_context:
            awaiting.append(
                {
                    "id": member.id,
                    "state": AWAITING_CONTEXT,
                    "reason": "No linked source row or component context",
                }
            )
            continue
        component = member.component
        if component is None or component not in plan.components:
            failed.append(
                {
                    "id": member.id,
                    "state": FAILED,
                    "reason": "Linked context has no component group",
                }
            )
            continue
        if component not in component_texts:
            failed.append(
                {
                    "id": member.id,
                    "state": FAILED,
                    "reason": "Component description was not delivered",
                }
            )
            continue
        pattern = plan.component_pattern.get(component)
        if pattern is None or pattern not in plan.patterns:
            failed.append(
                {
                    "id": member.id,
                    "state": FAILED,
                    "reason": "Component has no field-pattern group",
                }
            )
            continue
        if pattern not in pattern_texts:
            failed.append(
                {
                    "id": member.id,
                    "state": FAILED,
                    "reason": "Field-pattern description was not delivered",
                }
            )
            continue
        field_meaning = pattern_fields.get(pattern, {}).get(member.field_code or "", "")
        described.append(
            {
                "id": member.id,
                "state": DESCRIBED,
                "reason": None,
                "description": compose_member_description(
                    component_texts[component],
                    pattern_texts[pattern],
                    field_meaning,
                    member,
                ),
            }
        )

    cost = sum(
        outcome.cost
        for outcome in (*component_outcomes.values(), *pattern_outcomes.values())
    )
    tokens = sum(
        outcome.tokens
        for outcome in (*component_outcomes.values(), *pattern_outcomes.values())
    )
    drops = sum(
        outcome.drops
        for outcome in (*component_outcomes.values(), *pattern_outcomes.values())
    )
    return ComposedPlan(
        component_texts=component_texts,
        pattern_texts=pattern_texts,
        described=described,
        awaiting=awaiting,
        failed=failed,
        component_states=component_states,
        pattern_states=pattern_states,
        calls=sum(
            outcome.calls
            for outcome in (*component_outcomes.values(), *pattern_outcomes.values())
        ),
        cost=cost,
        tokens=tokens,
        drops=drops,
    )


def context_lines(
    *,
    function_category: str | None,
    function_category_code: str | None,
    plant_structure: str | None,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] = (),
    document_titles: Sequence[tuple[str | None, str | None]] = (),
) -> tuple[str, ...]:
    """Build the numbered context lines a group prompt cites.

    A row contributes only the field it names, so a sibling channel's row cannot
    be read as evidence about another field.
    """
    lines = []
    if fields:
        lines.append("pattern fields: " + ", ".join(sorted(str(f) for f in fields)))
    if function_category:
        code = f" ({function_category_code})" if function_category_code else ""
        lines.append(f"function category meaning: {function_category}{code}")
    if plant_structure:
        lines.append(f"plant structure: {plant_structure}")
    for row in rows:
        field_code = row.get("row_field") or row.get("signal_field")
        text = (row.get("description") or "").strip()
        if not text:
            continue
        label = f"source field {field_code}" if field_code else "source row"
        lines.append(f"{label}: {text}")
    for title, role in document_titles:
        if title:
            suffix = f" ({role})" if role else ""
            lines.append(f"source document: {title}{suffix}")
    return tuple(dict.fromkeys(lines))


def _component_document_titles(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str | None, str | None]]:
    return list(
        dict.fromkeys((row.get("document"), row.get("document_role")) for row in rows)
    )


def build_group_plan(graph, facility: str) -> GroupPlan:
    """Read the pilot cohort, its component groups and its field patterns."""
    records = graph.query(COHORT_QUERY, facility=facility)
    if not records:
        raise RuntimeError("Pilot cohort is empty; plant context was not imported")
    members = [PilotMember(**record) for record in records]
    component_members: dict[str, list[PilotMember]] = {}
    all_components: dict[str, list[PilotMember]] = {}
    for member in members:
        if member.component:
            all_components.setdefault(member.component, []).append(member)
        if member.has_context and member.component:
            component_members.setdefault(member.component, []).append(member)
    all_patterns: dict[str, str] = {}
    component_ids = sorted(all_components)
    for offset in range(0, len(component_ids), 500):
        rows = graph.query(
            PILOT_COMPONENTS_QUERY,
            facility=facility,
            components=component_ids[offset : offset + 500],
        )
        for row in rows:
            if row["pattern"]:
                all_patterns[row["component"]] = row["pattern"]

    components: dict[str, GroupRequest] = {}
    component_pattern: dict[str, str] = {}
    patterns: dict[str, GroupRequest] = {}
    pattern_members: dict[str, list[PilotMember]] = {}
    for component_id, component_members_ in component_members.items():
        detail = graph.query(COMPONENT_QUERY, facility=facility, component=component_id)
        if len(detail) != 1:
            raise RuntimeError(f"Component {component_id} is missing or duplicated")
        detail = detail[0]
        rows = graph.query(
            COMPONENT_CONTEXT_QUERY, facility=facility, component=component_id
        )
        components[component_id] = GroupRequest(
            group_id=component_id,
            kind="component",
            context=context_lines(
                function_category=detail["function_category"],
                function_category_code=detail["function_category_code"],
                plant_structure=detail["plant_structure"],
                rows=rows,
                document_titles=_component_document_titles(rows),
            ),
        )
        pattern_id = detail["field_pattern"]
        if not pattern_id:
            continue
        component_pattern[component_id] = pattern_id
        pattern_members.setdefault(pattern_id, []).extend(component_members_)
        if pattern_id in patterns:
            continue
        pattern = graph.query(PATTERN_QUERY, facility=facility, pattern=pattern_id)
        if len(pattern) != 1:
            raise RuntimeError(f"Field pattern {pattern_id} is missing or duplicated")
        pattern = pattern[0]
        pattern_rows = graph.query(
            PATTERN_CONTEXT_QUERY, facility=facility, pattern=pattern_id
        )
        patterns[pattern_id] = GroupRequest(
            group_id=pattern_id,
            kind="field_pattern",
            context=context_lines(
                function_category=None,
                function_category_code=None,
                plant_structure=None,
                rows=pattern_rows,
                fields=pattern["fields"] or (),
            ),
        )

    return GroupPlan(
        members=members,
        components=components,
        patterns=patterns,
        component_pattern=component_pattern,
        component_members=component_members,
        pattern_members=pattern_members,
        all_components=all_components,
        all_patterns=all_patterns,
    )


def _group_prompt(
    request: GroupRequest, sentence_limit: int
) -> tuple[list[dict[str, str]], dict[str, str]]:
    from imas_codex.discovery.signals.enrichment import GroupDescriptionBatch

    system = render_prompt(
        "signals/group_description",
        {
            "kind": request.kind,
            "group_id": request.group_id,
            "sentence_limit": sentence_limit,
            "output_schema": GroupDescriptionBatch.model_json_schema(),
        },
    )
    items = context_items("\n".join(request.context))
    user = "\n".join(f"[{key}] {text}" for key, text in items.items())
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], items


async def _describe_once(
    request: GroupRequest,
    *,
    model: str,
    call: Callable,
    messages: list[dict[str, str]],
    items: Mapping[str, str],
    review: bool,
    review_thinking: bool,
    recovery_attempts: int,
    call_kwargs: Mapping[str, Any],
) -> GroupOutcome:
    """One draft, optionally the seat's review pass, and its citations."""
    from imas_codex.discovery.signals.enrichment import (
        GroupDescriptionBatch,
        recovered_call,
        review_draft_statements,
    )

    def validate_draft(batch) -> None:
        rows = [row for row in batch.results if row.group_id == request.group_id]
        if not rows and len(batch.results) == 1:
            rows = batch.results
        if len(rows) != 1:
            raise ValueError("draft must describe the requested group exactly once")

    # One retry for a transient lane fault, as the group path always had: a
    # hiccup must not cost a group the run's only attempt.
    counters = {"draft_retries": 0, "review_retries": 0, "transient_retries": 0}
    try:
        batch, cost, tokens = await recovered_call(
            call,
            attempts=recovery_attempts,
            transient_attempts=1,
            validate=validate_draft,
            counters=counters,
            counter="draft_retries",
            model=model,
            messages=messages,
            response_model=GroupDescriptionBatch,
            **call_kwargs,
        )
    except Exception as exc:
        return GroupOutcome(
            cost=float(getattr(exc, "cost", 0) or 0.0),
            tokens=int(getattr(exc, "tokens", 0) or 0),
            calls=1 + counters["draft_retries"] + counters["transient_retries"],
            error=f"{type(exc).__name__}: {exc}",
        )
    calls = 1 + counters["draft_retries"] + counters["transient_retries"]
    cost, tokens = float(cost or 0.0), int(tokens or 0)
    rows = [row for row in batch.results if row.group_id == request.group_id]
    if not rows and len(batch.results) == 1:
        rows = batch.results
    kept, drops = filter_statements(rows[0].statements, items)
    reviewed_statements: list[str] | None = None
    if review and kept:
        review_kwargs = dict(call_kwargs)
        review_kwargs["reasoning_effort"] = "high" if review_thinking else None
        review_kwargs["temperature"] = 0
        try:
            (
                kept_by_index,
                review_cost,
                review_tokens,
                review_drops,
            ) = await review_draft_statements(
                call=call,
                model=model,
                items=items,
                drafts={1: rows[0].statements},
                attempts=recovery_attempts,
                counters=counters,
                counter="review_retries",
                review_kwargs=review_kwargs,
            )
        except Exception as exc:
            return GroupOutcome(
                cost=cost + float(getattr(exc, "cost", 0) or 0.0),
                tokens=tokens + int(getattr(exc, "tokens", 0) or 0),
                drops=drops,
                calls=calls + 1 + counters["review_retries"],
                error=f"Review failed: {type(exc).__name__}: {exc}",
            )
        calls += 1 + counters["review_retries"]
        cost += review_cost
        tokens += review_tokens
        drops += review_drops
        reviewed_statements = [text for _, text in kept_by_index.get(1, [])]
    text = " ".join(
        reviewed_statements if reviewed_statements is not None else kept
    ).strip()
    if not text:
        return GroupOutcome(
            cost=cost,
            tokens=tokens,
            drops=drops,
            calls=calls,
            error="No statement was supported by a supplied context item",
        )
    return GroupOutcome(
        text=text,
        fields=dict(rows[0].fields),
        cost=cost,
        tokens=tokens,
        drops=drops,
        calls=calls,
    )


async def describe_group(
    request: GroupRequest,
    *,
    model: str,
    call: Callable,
    sentence_limit: int = 2,
    review: bool = False,
    review_thinking: bool = False,
    attempts: int = 0,
    recovery_attempts: int = 0,
    **call_kwargs: Any,
) -> GroupOutcome:
    """Draft, optionally review, and refuse an unsupported description.

    The seat's own review pass checks the drafted statements against the same
    supplied items. A draft that leaves nothing supported is re-asked once with
    the failure named, mirroring the seat's validation re-ask; every attempt's
    calls, cost and tokens are accumulated. A group that still delivers nothing
    is reported failed and its members inherit that.
    """
    messages, items = _group_prompt(request, sentence_limit)
    total_cost, total_tokens, total_calls, total_drops = 0.0, 0, 0, 0
    last_error = None
    for attempt in range(max(0, attempts) + 1):
        outcome = await _describe_once(
            request,
            model=model,
            call=call,
            messages=messages,
            items=items,
            review=review,
            review_thinking=review_thinking,
            recovery_attempts=recovery_attempts,
            call_kwargs=call_kwargs,
        )
        total_cost += outcome.cost
        total_tokens += outcome.tokens
        total_calls += outcome.calls
        total_drops += outcome.drops
        if outcome.delivered:
            return GroupOutcome(
                text=outcome.text,
                fields=outcome.fields,
                cost=total_cost,
                tokens=total_tokens,
                drops=total_drops,
                calls=total_calls,
            )
        last_error = outcome.error
        if attempt < attempts:
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": "The previous answer delivered no supported "
                    "statement. Cite the supplied context items by their "
                    "bracketed identifiers and state only what they say.",
                },
            ]
    return GroupOutcome(
        cost=total_cost,
        tokens=total_tokens,
        drops=total_drops,
        calls=total_calls,
        error=last_error,
    )


async def describe_groups(
    requests: Sequence[GroupRequest],
    *,
    model: str,
    call: Callable,
    max_workers: int = 8,
    sentence_limit: int = 2,
    review: bool = False,
    review_thinking: bool = False,
    attempts: int = 0,
    recovery_attempts: int = 0,
    **call_kwargs,
) -> dict[str, GroupOutcome]:
    """Describe groups with bounded concurrency, retaining every outcome."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    limit = min(max_workers, 8)
    gate = asyncio.Semaphore(limit)

    async def one(request):
        async with gate:
            return request.group_id, await describe_group(
                request,
                model=model,
                call=call,
                sentence_limit=sentence_limit,
                review=review,
                review_thinking=review_thinking,
                attempts=attempts,
                recovery_attempts=recovery_attempts,
                **call_kwargs,
            )

    return dict(await asyncio.gather(*(one(request) for request in requests)))


def _apportion(cost: float, count: int) -> float:
    return cost / count if count and cost else 0.0


def apply_composed_plan(
    graph,
    facility: str,
    plan: GroupPlan,
    composed: ComposedPlan,
    *,
    writer: Callable | None = None,
) -> dict:
    """Write the group descriptions and every member's decided state."""
    from imas_codex.discovery.signals.parallel import write_signal_descriptions

    writer = writer or write_signal_descriptions
    written_components = 0
    for component_id, text in composed.component_texts.items():
        result = graph.query(
            COMPONENT_DESCRIPTION_WRITE,
            component=component_id,
            facility=facility,
            text=text,
        )
        if not result or result[0]["written"] != 1:
            raise RuntimeError(f"Component description was not written: {component_id}")
        written_components += 1
    written_patterns = 0
    for pattern_id, text in composed.pattern_texts.items():
        result = graph.query(
            PATTERN_DESCRIPTION_WRITE,
            pattern=pattern_id,
            facility=facility,
            text=text,
        )
        if not result or result[0]["written"] != 1:
            raise RuntimeError(f"Pattern description was not written: {pattern_id}")
        written_patterns += 1

    rows = []
    for item in composed.described:
        member = next(m for m in plan.members if m.id == item["id"])
        cost = _apportion(composed.cost, len(composed.described))
        rows.append(
            {
                "id": member.id,
                "description": item["description"],
                "status": "enriched",
                "enrichment_source": "group_composition",
                "group_enrichment_state": DESCRIBED,
                "group_enrichment_reason": None,
                "llm_cost": cost,
                "enriched_at": datetime.now(UTC),
                "claimed_at": None,
                "claim_token": None,
            }
        )
    written_members = writer(rows, gc=graph) if rows else 0

    failed_rows = [
        {
            "id": item["id"],
            "status": "failed",
            "group_enrichment_state": FAILED,
            "group_enrichment_reason": item["reason"],
            "claimed_at": None,
            "claim_token": None,
        }
        for item in composed.failed
    ]
    written_failed = _write_states(graph, facility, failed_rows) if failed_rows else 0

    awaiting_rows = [
        {
            "id": item["id"],
            "group_enrichment_state": AWAITING_CONTEXT,
            "group_enrichment_reason": item["reason"],
        }
        for item in composed.awaiting
    ]
    written_awaiting = (
        _write_states(graph, facility, awaiting_rows) if awaiting_rows else 0
    )

    return {
        "components_written": written_components,
        "patterns_written": written_patterns,
        "members_described": written_members,
        "members_failed": written_failed,
        "members_awaiting_context": written_awaiting,
    }


def _write_states(graph, facility: str, rows: Sequence[Mapping[str, Any]]) -> int:
    """Write only the state properties, leaving every description untouched."""
    written = 0
    for offset in range(0, len(rows), 500):
        batch = rows[offset : offset + 500]
        result = graph.query(
            GROUP_STATE_WRITE,
            facility=facility,
            rows=batch,
        )
        if not result or result[0]["written"] != len(batch):
            raise RuntimeError("Signal group state was not fully written")
        written += result[0]["written"]
    return written


def summarise(plan: GroupPlan, composed: ComposedPlan) -> dict[str, Any]:
    """Count every state so the three member counts sum to the pilot total."""
    component_total = len(plan.all_components) or len(plan.components)
    component_described = sum(
        1 for state in composed.component_states.values() if state == DESCRIBED
    )
    component_attempted = len(plan.components)
    component_failed = component_attempted - component_described
    component_awaiting = component_total - component_attempted
    pattern_total = len(set(plan.all_patterns.values())) or len(plan.patterns)
    pattern_described = sum(
        1 for state in composed.pattern_states.values() if state == DESCRIBED
    )
    pattern_failed = len(plan.patterns) - pattern_described
    pattern_awaiting = pattern_total - len(plan.patterns)
    values: dict[str, dict[str, int]] = {
        axis: dict.fromkeys(sorted(members), 0)
        for axis, members in VOCABULARY_MEMBERS.items()
    }
    invalid = 0
    members = {member.id: member for member in plan.members}
    for item in composed.described:
        decided = decisions(members[item["id"]])
        for axis, value in decided.items():
            values[axis][value] += 1
    for member in plan.members:
        for axis, value, state in (
            ("quantity", member.quantity, member.quantity_state),
            ("unit", member.unit, member.unit_state),
            ("role", member.role, member.role_state),
        ):
            if state == "decided" and value not in VOCABULARY_MEMBERS[axis]:
                invalid += 1
    described = len(composed.described)
    awaiting = len(composed.awaiting)
    failed = len(composed.failed)
    return {
        "signals_total": len(plan.members),
        "signals_described": described,
        "signals_awaiting_context": awaiting,
        "signals_failed": failed,
        "signals_sum": described + awaiting + failed,
        "components_total": component_total,
        "components_described": component_described,
        "components_awaiting_context": component_awaiting,
        "components_failed": component_failed,
        "patterns_total": pattern_total,
        "patterns_described": pattern_described,
        "patterns_awaiting_context": pattern_awaiting,
        "patterns_failed": pattern_failed,
        "calls": composed.calls,
        "cost_usd": composed.cost,
        "tokens": composed.tokens,
        "statements_dropped": composed.drops,
        "vocabulary": values,
        "invalid_vocabulary_values": invalid,
    }


async def enrich_pilot_groups(
    *,
    facility: str,
    system: str | None = None,
    graph=None,
    call: Callable | None = None,
    max_workers: int = 8,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Describe every pilot component and pattern, then compose its members.

    ``system`` is accepted for the command surface's symmetry with the plant
    context import; the cohort is already the assessed pilot system, so it is
    used only in the reported scope name.
    """
    from imas_codex.settings import get_model, get_reasoning_effort

    owns_graph = graph is None
    graph = graph or GraphClient()
    try:
        plan = build_group_plan(graph, facility)
        if dry_run:
            return {
                "scope": system,
                "signals_total": len(plan.members),
                "components_total": len(plan.components),
                "patterns_total": len(plan.patterns),
                "dry_run": True,
            }
        model = get_model("discovery-describe")
        options = get_enrichment_options(model)
        if call is None:
            from imas_codex.discovery.base.llm import acall_llm_structured

            call = acall_llm_structured
        call_kwargs = {"reasoning_effort": get_reasoning_effort("discovery-describe")}
        call_kwargs.update(options.call_kwargs())
        if options.temperature is not None:
            call_kwargs["temperature"] = options.temperature
        requests = list(plan.components.values()) + list(plan.patterns.values())
        outcomes = await describe_groups(
            requests,
            model=model,
            call=call,
            max_workers=min(max_workers, options.max_workers or 8),
            sentence_limit=options.description_sentences or 2,
            review=options.review,
            review_thinking=options.review_thinking,
            attempts=options.recovery_attempts,
            recovery_attempts=options.recovery_attempts,
            **call_kwargs,
        )
        component_outcomes = {
            group_id: outcome
            for group_id, outcome in outcomes.items()
            if group_id in plan.components
        }
        pattern_outcomes = {
            group_id: outcome
            for group_id, outcome in outcomes.items()
            if group_id in plan.patterns
        }
        composed = compose_plan(plan, component_outcomes, pattern_outcomes)
        written = apply_composed_plan(graph, facility, plan, composed)
        return {
            "scope": system,
            "signals_total": len(plan.members),
            "components_total": len(plan.components),
            "patterns_total": len(plan.patterns),
            "dry_run": False,
            **summarise(plan, composed),
            **written,
        }
    finally:
        if owns_graph:
            graph.close()

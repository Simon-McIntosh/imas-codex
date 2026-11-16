"""Seat-scoped controls for signal enrichment requests and output contracts."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from imas_codex.discovery.signals.models import (
    ContextQuality,
    SignalEnrichmentBatch,
    SignalEnrichmentResult,
)
from imas_codex.llm.prompt_loader import render_prompt


class EnrichmentOptions(BaseModel):
    """Opt-in local controls; unset seats retain their existing request shape."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    batch_size: int = Field(default=20, alias="batch-size", ge=1, le=200)
    schema_constraint: bool = Field(default=False, alias="schema-constraint")
    json_mode: bool = Field(default=False, alias="json-mode")
    thinking: bool | None = None
    max_tokens: int | None = Field(default=None, alias="max-tokens", ge=1)
    fitted_prompt: bool = Field(default=False, alias="fitted-prompt")
    validation_reask: bool = Field(default=False, alias="validation-reask")
    grounding: bool = False
    claim_evidence: bool = Field(default=False, alias="claim-evidence")
    review: bool = False
    review_thinking: bool = Field(default=False, alias="review-thinking")
    temperature: float | None = Field(default=None, ge=0, le=2)
    description_sentences: int | None = Field(
        default=None, alias="description-sentences", ge=1, le=10
    )

    @model_validator(mode="after")
    def check_output_constraint(self):
        if self.schema_constraint and self.json_mode:
            raise ValueError("choose either schema-constraint or json-mode")
        return self

    def call_kwargs(self) -> dict:
        """Send only enabled controls, retaining shared defaults otherwise."""
        result = {}
        if self.schema_constraint:
            result["local_schema_constraint"] = True
        if self.json_mode:
            result["local_json_mode"] = True
        if self.validation_reask:
            result["validation_reask"] = True
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.thinking is not None:
            result["reasoning_effort"] = "high" if self.thinking else None
        return result


def get_enrichment_options(model: str) -> EnrichmentOptions:
    """Read the description seat's signal controls for its local model only."""
    from imas_codex.settings import _get_section

    if not model.startswith("local/"):
        return EnrichmentOptions()
    seat = _get_section("discovery-describe")
    if model != seat.get("model", model):
        return EnrichmentOptions()
    return EnrichmentOptions.model_validate(seat.get("signal-enrichment", {}))


def fit_enrichment_prompt(baseline: str) -> str:
    """Retain the full task and append an explicit, schema-derived contract."""
    schema = SignalEnrichmentBatch.model_json_schema()
    example = SignalEnrichmentBatch(
        results=[
            SignalEnrichmentResult(
                signal_index=1,
                physics_domain="plant_systems",
                name="Cooling circuit temperature",
                description="Measured temperature of the cooling circuit.",
                context_quality=ContextQuality.high,
                keywords=["temperature", "cooling"],
            )
        ]
    )
    return render_prompt(
        "signals/enrichment_output",
        {
            "baseline": baseline,
            "domain_values": schema["$defs"]["PhysicsDomain"]["enum"],
            "diagnostic_values": schema["$defs"]["DiagnosticCategory"]["enum"],
            "quality_values": [quality.value for quality in ContextQuality],
            "worked_example": example.model_dump_json(indent=2),
            "output_schema": schema,
        },
    )


class CitedStatement(BaseModel):
    """One description statement and the supplied items supporting it."""

    text: str
    context_items: list[str] = Field(default_factory=list)


class CitedEnrichmentResult(SignalEnrichmentResult):
    statements: list[CitedStatement] = Field(default_factory=list)


class CitedEnrichmentBatch(BaseModel):
    results: list[CitedEnrichmentResult]


class ReviewedStatement(CitedStatement):
    draft_index: int


class ReviewedDescription(BaseModel):
    signal_index: int
    statements: list[ReviewedStatement]


class DescriptionReview(BaseModel):
    results: list[ReviewedDescription]


class GroundedEnrichmentBatch(SignalEnrichmentBatch):
    """Counts accompany the result without becoming graph properties."""

    evidence_drops: int = 0
    review_drops: int = 0
    review_revisions: int = 0
    length_drops: int = 0


def context_items(text: str) -> dict[str, str]:
    """Give each nonempty supplied context line a stable call-local identity."""
    return {
        f"context-{index}": line
        for index, line in enumerate(text.splitlines(), 1)
        if line.strip()
    }


def filter_statements(
    statements: list[CitedStatement], items: dict[str, str]
) -> tuple[list[str], int]:
    """Refuse absent citations; item existence alone does not prove entailment."""
    kept = []
    for statement in statements:
        references = [
            item[1:-1] if item.startswith("[") and item.endswith("]") else item
            for item in statement.context_items
        ]
        if (
            statement.text.strip()
            and references
            and all(item in items for item in references)
        ):
            kept.append(statement.text.strip())
    return kept, len(statements) - len(kept)


def description_statements(text: str) -> list[str]:
    """Split prose at sentence boundaries for review and the length bound."""
    import re

    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


async def enrich_signal_batch(*, model, messages, options, call, **kwargs):
    """Draft, optionally verify, and return only the supported description text.

    Evidence validation checks references, while the separate local review and
    blinded reading assess whether the cited text actually supports the claim.
    A failed review refuses the entire batch rather than delivering its draft.
    """
    import copy
    import json

    active = options.grounding or options.claim_evidence or options.review
    if active and not model.startswith("local/"):
        raise ValueError("grounded signal enrichment requires a local model")
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if not active and not options.description_sentences:
        return await call(
            model=model,
            messages=messages,
            response_model=SignalEnrichmentBatch,
            **kwargs,
        )
    messages = copy.deepcopy(messages)
    user = next(message for message in messages if message["role"] == "user")
    items = context_items(user["content"])
    response_model = (
        CitedEnrichmentBatch if options.claim_evidence else SignalEnrichmentBatch
    )
    if options.grounding or options.claim_evidence or options.description_sentences:
        system = next(message for message in messages if message["role"] == "system")
        system["content"] = render_prompt(
            "signals/enrichment_grounding",
            {
                "baseline": system["content"],
                "grounding": options.grounding,
                "claim_evidence": options.claim_evidence,
                "sentence_limit": options.description_sentences,
                "output_schema": response_model.model_json_schema(),
            },
        )
    if options.claim_evidence:
        user["content"] = "\n".join(f"[{key}] {text}" for key, text in items.items())
    draft, cost, tokens = await call(
        model=model, messages=messages, response_model=response_model, **kwargs
    )
    result = GroundedEnrichmentBatch(results=[])
    draft_statements = {}
    for row in draft.results:
        if options.claim_evidence:
            statements, dropped = filter_statements(row.statements, items)
            result.evidence_drops += dropped
        else:
            statements = description_statements(row.description)
        draft_statements[row.signal_index] = statements
        values = row.model_dump(exclude={"statements"})
        values["description"] = " ".join(statements)
        result.results.append(SignalEnrichmentResult.model_validate(values))
    if options.review:
        review_kwargs = dict(kwargs)
        review_kwargs["reasoning_effort"] = "high" if options.review_thinking else None
        review_kwargs["temperature"] = 0
        review_messages = [
            {
                "role": "system",
                "content": render_prompt(
                    "signals/enrichment_review",
                    {"output_schema": DescriptionReview.model_json_schema()},
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"context_items": items, "drafts": draft_statements},
                    ensure_ascii=False,
                ),
            },
        ]
        reviewed, review_cost, review_tokens = await call(
            model=model,
            messages=review_messages,
            response_model=DescriptionReview,
            **review_kwargs,
        )
        cost += review_cost
        tokens += review_tokens
        by_index = {}
        for row in reviewed.results:
            if row.signal_index in by_index:
                raise ValueError("review returned a duplicate signal index")
            by_index[row.signal_index] = row
        if set(by_index) != set(draft_statements):
            raise ValueError("review must cover every drafted signal exactly once")
        for row in result.results:
            original = draft_statements[row.signal_index]
            kept = []
            seen = set()
            for statement in by_index[row.signal_index].statements:
                if (
                    not 1 <= statement.draft_index <= len(original)
                    or statement.draft_index in seen
                ):
                    raise ValueError(
                        "review must refer to unique existing draft statements"
                    )
                seen.add(statement.draft_index)
                valid, dropped = filter_statements([statement], items)
                result.evidence_drops += dropped
                if valid:
                    kept.append((statement.draft_index, valid[0]))
                    result.review_revisions += (
                        valid[0] != original[statement.draft_index - 1]
                    )
            result.review_drops += len(original) - len(kept)
            row.description = " ".join(text for _, text in sorted(kept))
    for row in result.results:
        if options.description_sentences:
            statements = description_statements(row.description)
            result.length_drops += max(
                0, len(statements) - options.description_sentences
            )
            row.description = " ".join(statements[: options.description_sentences])
        if not row.description.strip():
            row.context_quality = ContextQuality.low
    return result, cost, tokens

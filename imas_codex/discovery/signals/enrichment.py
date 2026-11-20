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
    max_workers: int | None = Field(default=None, alias="max-workers", ge=1, le=64)
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
    recovery_attempts: int = Field(default=0, alias="recovery-attempts", ge=0, le=2)
    source_fallback: bool = Field(default=False, alias="source-fallback")
    review_if_unsupported: bool = Field(default=False, alias="review-if-unsupported")
    review_batch_size: int = Field(default=1, alias="review-batch-size", ge=1, le=8)
    temperature: float | None = Field(default=None, ge=0, le=2)
    description_sentences: int | None = Field(
        default=None, alias="description-sentences", ge=1, le=10
    )

    @model_validator(mode="after")
    def check_output_constraint(self):
        if self.schema_constraint and self.json_mode:
            raise ValueError("choose either schema-constraint or json-mode")
        if (
            self.review_if_unsupported or self.review_batch_size > 1
        ) and not self.review:
            raise ValueError("conditional or batched review requires review")
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


class GroupStatement(BaseModel):
    """One statement about a group, with the supplied items supporting it."""

    text: str
    context_items: list[str] = Field(default_factory=list)


class GroupDescription(BaseModel):
    """One component or field-pattern description and its per-field meanings."""

    group_id: str
    statements: list[GroupStatement] = Field(default_factory=list)
    fields: dict[str, str] = Field(default_factory=dict)


class GroupDescriptionBatch(BaseModel):
    results: list[GroupDescription] = Field(default_factory=list)


class GroundedEnrichmentBatch(SignalEnrichmentBatch):
    """Counts accompany the result without becoming graph properties."""

    evidence_drops: int = 0
    review_drops: int = 0
    review_revisions: int = 0
    length_drops: int = 0
    draft_retries: int = 0
    review_retries: int = 0
    review_skips: int = 0
    fallback_indices: list[int] = Field(default_factory=list)
    fallback_context_items: dict[int, list[str]] = Field(default_factory=dict)


def signal_source_items(items: dict[str, str]) -> dict[int, dict[str, tuple[str, str]]]:
    """Bind catalogue fields to their target, excluding shared retrieval text."""
    import re

    sources = {}
    index = 1
    in_signal = not any(
        re.match(r"### Signal \d+\s*$", line) for line in items.values()
    )
    in_fence = False
    for key, line in items.items():
        if line.startswith("```"):
            in_fence = not in_fence
            in_signal = False
            continue
        if in_fence:
            continue
        match = re.match(r"### Signal (\d+)\s*$", line)
        if match:
            index, in_signal = int(match[1]), True
            sources.setdefault(index, {})
        elif line.startswith(("## ", "**")):
            in_signal = False
        elif in_signal:
            for field in ("source_description", "unit"):
                prefix = field + ": "
                if line.startswith(prefix) and line[len(prefix) :].strip():
                    sources.setdefault(index, {})[field] = (
                        key,
                        line[len(prefix) :].strip(),
                    )
    return sources


def supported_source_statement(text: str, source: dict) -> bool:
    """Skip review only for a literal catalogue label of this exact signal.

    Token overlap and citation existence cannot establish entailment. This
    conservative shortcut admits only punctuation/whitespace normalization.
    """
    import re

    def normalize(value):
        return re.sub(r"\s+", " ", value.strip().rstrip(".!?")).casefold()

    label = source.get("source_description", (None, ""))[1]
    return bool(label) and normalize(text) == normalize(label)


async def recovered_call(call, *, attempts, validate, counters, counter, **kwargs):
    """Retry only the failed structured call, retaining its consumed telemetry."""
    from imas_codex.discovery.base.llm import (
        LLMStructuredCallError,
        ProviderBudgetExhausted,
    )

    cost, tokens = 0.0, 0
    for attempt in range(attempts + 1):
        try:
            value, call_cost, call_tokens = await call(**kwargs)
            cost += call_cost
            tokens += call_tokens
            validate(value)
            return value, cost, tokens
        except ProviderBudgetExhausted:
            raise
        except ValueError as exc:
            if isinstance(exc, LLMStructuredCallError):
                if exc.telemetry_failure:
                    raise
            cost += getattr(exc, "cost", 0) or 0
            tokens += getattr(exc, "tokens", 0) or 0
            if attempt == attempts:
                raise
            counters[counter] += 1
            kwargs = dict(kwargs)
            kwargs["messages"] = [
                *kwargs["messages"],
                {
                    "role": "user",
                    "content": "The response failed validation:\n"
                    + str(exc)
                    + "\nReturn corrected JSON for the same supplied signals and context.",
                },
            ]


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

    statements = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        part = part.strip()
        if not part:
            continue
        if statements and re.search(
            r"\b(?:ref|no|e\.g|i\.e|etc|vs|fig|eq|inj|gr)\.$", statements[-1], re.I
        ):
            statements[-1] += " " + part
        else:
            statements.append(part)
    return statements


async def enrich_signal_batch(*, model, messages, options, call, **kwargs):
    """Draft, optionally verify, and return only the supported description text.

    Evidence validation checks references, while the separate local review and
    blinded reading assess whether the cited text actually supports the claim.
    A failed review refuses the entire batch rather than delivering its draft.
    """
    import copy
    import json

    active = (
        options.grounding
        or options.claim_evidence
        or options.review
        or options.source_fallback
        or options.recovery_attempts
    )
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
    sources = signal_source_items(items)
    counters = {"draft_retries": 0, "review_retries": 0}
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

    def validate_draft(value):
        if options.recovery_attempts:
            expected = set(sources)
            indices = [row.signal_index for row in value.results]
            if (
                not expected
                or set(indices) != expected
                or len(indices) != len(expected)
            ):
                raise ValueError("draft must cover every supplied signal exactly once")

    draft, cost, tokens = await recovered_call(
        call,
        attempts=options.recovery_attempts,
        validate=validate_draft,
        counters=counters,
        counter="draft_retries",
        model=model,
        messages=messages,
        response_model=response_model,
        **kwargs,
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
    review_statements = {
        index: statements
        for index, statements in draft_statements.items()
        if not (
            options.review_if_unsupported
            and statements
            and all(
                supported_source_statement(text, sources.get(index, {}))
                for text in statements
            )
        )
    }
    result.review_skips = (
        len(draft_statements) - len(review_statements) if options.review else 0
    )
    if options.review and review_statements:
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
                    {"context_items": items, "drafts": review_statements},
                    ensure_ascii=False,
                ),
            },
        ]

        def validate_review(value):
            indices = [row.signal_index for row in value.results]
            if set(indices) != set(review_statements) or len(indices) != len(
                review_statements
            ):
                raise ValueError("review must cover every drafted signal exactly once")
            for row in value.results:
                seen = set()
                for statement in row.statements:
                    if (
                        not 1
                        <= statement.draft_index
                        <= len(review_statements[row.signal_index])
                        or statement.draft_index in seen
                    ):
                        raise ValueError(
                            "review must refer to unique existing draft statements"
                        )
                    seen.add(statement.draft_index)

        review_call = call
        if options.review_batch_size > 1:
            from imas_codex.discovery.signals.enrichment_review import batched_review

            async def review_call(**request):
                return await batched_review(call, options.review_batch_size, **request)

        reviewed, review_cost, review_tokens = await recovered_call(
            review_call,
            attempts=options.recovery_attempts,
            validate=validate_review,
            counters=counters,
            counter="review_retries",
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
        if set(by_index) != set(review_statements):
            raise ValueError("review must cover every drafted signal exactly once")
        for row in result.results:
            if row.signal_index not in review_statements:
                continue
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
            if options.source_fallback:
                source = sources.get(row.signal_index, {})
                label = source.get("source_description")
                if label is None:
                    raise ValueError("no authoritative source label for fallback")
                row.description = description_statements(label[1])[0]
                result.fallback_indices.append(row.signal_index)
                result.fallback_context_items[row.signal_index] = [label[0]]
    result.draft_retries = counters["draft_retries"]
    result.review_retries = counters["review_retries"]
    return result, cost, tokens

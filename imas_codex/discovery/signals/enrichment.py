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
    return EnrichmentOptions.model_validate(
        _get_section("discovery-describe").get("signal-enrichment", {})
    )


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

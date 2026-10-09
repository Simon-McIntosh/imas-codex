"""The mapping system prompts carry every field of the model they ask for.

The local lane sends no response_format, so a field missing from the prompt is
a field the model has to guess. A guessed name fails Pydantic validation and
the whole batch is dropped.
"""

import pytest

from imas_codex.ids.models import (
    AssemblyConfig,
    EscalationFlag,
    SignalMappingBatch,
    SignalMappingEntry,
    UnmappedSignal,
)
from imas_codex.llm.prompt_loader import render_prompt


@pytest.mark.parametrize(
    ("prompt", "models"),
    [
        (
            "mapping/signal_mapping_system",
            (SignalMappingBatch, SignalMappingEntry, UnmappedSignal, EscalationFlag),
        ),
        ("mapping/assembly_system", (AssemblyConfig,)),
    ],
)
def test_system_prompt_names_every_response_field(prompt, models):
    rendered = render_prompt(prompt)
    for model in models:
        for name, field in model.model_fields.items():
            marker = "required" if field.is_required() else "optional"
            assert f"**{name}**" in rendered, f"{prompt} omits {model.__name__}.{name}"
            line = next(line for line in rendered.splitlines() if f"**{name}**" in line)
            assert marker in line, f"{prompt} does not mark {name} as {marker}"

"""Golden test: candidate schema in prompt must match StandardNameCandidate model.

This test is the regression gate for prompt ↔ schema drift.  It parses
the ``### Candidate Schema`` block in ``generate_name_system.md`` and
asserts the documented field set is exactly the union of
``StandardNameCandidate.model_fields`` and ``GrammarSegments.model_fields``.

Any addition or removal of a field in either the Pydantic model or the
prompt **must** be mirrored in the other; this test will catch the drift
at CI time.
"""

from __future__ import annotations

import re

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR
from imas_codex.standard_names.models import GrammarSegments, StandardNameCandidate

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _extract_candidate_schema_fields(prompt_text: str) -> set[str]:
    """Extract field names from the ``### Candidate Schema`` section.

    The section documents both top-level fields and segment fields.
    We capture the backtick-delimited field name from each ``- `field`:``
    line, excluding the ``segments`` wrapper itself (which is structural,
    not a data field).
    """
    # Locate the section
    marker = "### Candidate Schema"
    idx = prompt_text.find(marker)
    if idx == -1:
        raise ValueError(f"'{marker}' section not found in prompt text")

    # Slice from the marker to the next heading or end-of-file
    rest = prompt_text[idx + len(marker) :]
    # Next heading: a line starting with '#'
    next_heading = re.search(r"(?m)^#{1,4}\s", rest)
    block = rest[: next_heading.start()] if next_heading else rest

    # Extract field names: lines matching ``- `field_name`:``
    fields: set[str] = set()
    for m in re.finditer(r"^-\s+`([a-z_]+)`[^:]*:", block, re.MULTILINE):
        fields.add(m.group(1))

    if not fields:
        raise ValueError(
            "No fields parsed from Candidate Schema block — check the prompt format"
        )
    # Remove 'segments' — it's a structural wrapper, not a data field
    fields.discard("segments")
    return fields


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestCandidateSchemaMatchesPromptSpec:
    """Prompt-documented fields must exactly match the Pydantic model."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        path = PROMPTS_DIR / "sn" / "generate_name_system.md"
        self.raw = path.read_text(encoding="utf-8")
        # Union of top-level fields (minus 'segments' wrapper) + segment fields
        top_fields = set(StandardNameCandidate.model_fields.keys()) - {"segments"}
        segment_fields = set(GrammarSegments.model_fields.keys())
        self.model_fields = top_fields | segment_fields

    def test_candidate_schema_matches_prompt_spec(self) -> None:
        """Field set in prompt == union of top-level + segment model fields."""
        prompt_fields = _extract_candidate_schema_fields(self.raw)
        assert prompt_fields == self.model_fields, (
            f"Schema drift detected!\n"
            f"  In prompt but not model: {prompt_fields - self.model_fields}\n"
            f"  In model but not prompt: {self.model_fields - prompt_fields}"
        )


# --------------------------------------------------------------------------
# Regression guard: structured-output schemas must stay within the Anthropic /
# OpenRouter per-object property limit.  RefinedName.segments silently grew to
# 14 properties and every Anthropic structured call with it failed at runtime
# with "Schema is too complex" — invisible until a live bench.  Pin the
# invariant so a future field addition fails in CI, not in production.
# --------------------------------------------------------------------------

# OpenRouter's undocumented cap for Anthropic structured output is ~13
# properties per JSON-Schema object ($defs item).  Stay at or below it.
_ANTHROPIC_MAX_PROPS_PER_OBJECT = 13


def _max_object_property_count(schema: dict) -> tuple[int, str]:
    """Return (max properties on any object, the object's title) in *schema*.

    Walks the root schema plus every ``$defs`` entry — an Anthropic structured
    call rejects a schema when ANY single object exceeds the cap, so the
    per-object maximum is what matters, not the total field count.
    """
    worst = (0, "<root>")
    objects = [("<root>", schema)]
    for name, defn in (schema.get("$defs") or {}).items():
        objects.append((name, defn))
    for name, obj in objects:
        n = len(obj.get("properties") or {})
        if n > worst[0]:
            worst = (n, name)
    return worst


@pytest.mark.parametrize(
    "model_import",
    [
        "RefinedName",
        "StandardNameCandidate",
        "StandardNameCandidateResponse",
    ],
)
def test_llm_schema_object_within_anthropic_property_limit(model_import):
    """Every object in an LLM response-model's validation schema stays ≤ the
    Anthropic per-object property cap, so Anthropic structured output does not
    fail with 'Schema is too complex'."""
    import importlib

    models_mod = importlib.import_module("imas_codex.standard_names.models")
    model_cls = getattr(models_mod, model_import, None)
    if model_cls is None:
        pytest.skip(f"{model_import} not present")
    schema = model_cls.model_json_schema()  # validation mode = the LLM's schema
    count, obj_name = _max_object_property_count(schema)
    assert count <= _ANTHROPIC_MAX_PROPS_PER_OBJECT, (
        f"{model_import} object '{obj_name}' has {count} properties, exceeding "
        f"the Anthropic structured-output cap of {_ANTHROPIC_MAX_PROPS_PER_OBJECT}"
        f" — Anthropic calls will fail with 'Schema is too complex'. Split the "
        f"object or derive a field (see GrammarSegments.projection_shape)."
    )

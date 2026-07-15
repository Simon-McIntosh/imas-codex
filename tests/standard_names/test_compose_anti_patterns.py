"""Compose-prompt instrument/hardware anti-pattern hardening tests.

Verifies the three diagnostic-IDS anti-patterns (instrument prefix carry-over,
suffix-form for component, compound hardware identifiers) are present in
the system prompt (generate_name_system.md) and the user prompts
(generate_name_dd.md, generate_name_dd_names.md).

The system prompt teaches these as one ``ANTI-PATTERN REFERENCE`` section plus
the ``Hardware / instrument tokens`` field-choice rule; the user-prompt tables
tag each row with a descriptive rule name. Rotation stage labels were dropped
per the naming-hygiene rule (source must not leak plan/stage identifiers), so
these tests assert the surviving bad/good example pairs and the descriptive
rule tags, never a stage label.

These are content/structural assertions — they do not call the LLM.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt


def _load(name: str) -> str:
    return (PROMPTS_DIR / "sn" / name).read_text(encoding="utf-8")


# Concrete real bad/good pairs from the diagnostic-IDS rotation.
ANTI_BAD_EXAMPLES = (
    "x_ray_crystal_spectrometer_pixel_photon_energy_lower_bound",
    "halo_region_parallel_energy_due_to_heat_flux",
    "z_coordinate_of_sensor_direction_unit_vector",
)
ANTI_GOOD_EXAMPLES = (
    "lower_bound_photon_energy",
    "parallel_halo_energy",
    "z_direction_unit_vector",
)
ANTI_HARDWARE_PROPERTY_EXEMPLAR = "area_of_rogowski_coil"
LOCAL_AXIS_BAD_EXAMPLE = "vertical_front_surface_radius_of_optical_element"
LOCAL_AXIS_GOOD_EXAMPLE = "second_local_tangential_front_surface_radius_of_reflector"


@pytest.mark.parametrize("filename", ["generate_name_system.md"])
class TestSystemPromptInstrumentGallery:
    """The instrument anti-pattern entries must appear in the system prompt."""

    def test_gallery_heading(self, filename: str) -> None:
        raw = _load(filename)
        assert "ANTI-PATTERN REFERENCE" in raw

    @pytest.mark.parametrize("bad", ANTI_BAD_EXAMPLES)
    def test_bad_example_present(self, filename: str, bad: str) -> None:
        assert bad in _load(filename)

    @pytest.mark.parametrize("good", ANTI_GOOD_EXAMPLES)
    def test_good_example_present(self, filename: str, good: str) -> None:
        assert good in _load(filename)

    def test_hardware_property_exception(self, filename: str) -> None:
        # The gallery must carve out the hardware-property exception so the
        # generator does not over-strip instrument tokens.
        raw = _load(filename)
        assert ANTI_HARDWARE_PROPERTY_EXEMPLAR in raw
        assert "intrinsic" in raw.lower()

    def test_local_axis_semantic_replacement(self, filename: str) -> None:
        raw = _load(filename)
        assert "first_local_tangential_coordinate" in raw
        assert "second_local_tangential_coordinate" in raw
        assert "do not emit `x1_coordinate` / `x2_coordinate`" in raw


@pytest.mark.parametrize(
    "filename", ["generate_name_dd.md", "generate_name_dd_names.md"]
)
class TestUserPromptAntiPatternRows:
    """The user-facing per-batch tables must carry the descriptive rule tags."""

    def test_anti_pattern_tags_in_table(self, filename: str) -> None:
        raw = _load(filename)
        assert "Instrument-prefix carry-over" in raw
        assert "Suffix-form for component" in raw
        assert "Compound hardware identifiers" in raw
        # The plan/stage labels must NOT leak back in.
        assert "W38" not in raw

    @pytest.mark.parametrize(
        "bad,good",
        list(zip(ANTI_BAD_EXAMPLES, ANTI_GOOD_EXAMPLES, strict=True)),
    )
    def test_concrete_pairs(self, filename: str, bad: str, good: str) -> None:
        raw = _load(filename)
        assert bad in raw
        assert good in raw


def test_dd_prompt_carries_reflector_local_axis_regression_pair() -> None:
    raw = _load("generate_name_dd.md")
    assert LOCAL_AXIS_BAD_EXAMPLE in raw
    assert LOCAL_AXIS_GOOD_EXAMPLE in raw


class TestSystemPromptInstrumentPlacement:
    """Anti-patterns must come AFTER the static schema/grammar block (includes)
    but BEFORE the curated-examples / dynamic context — i.e. inside the static
    system-prompt cache layer, not inside the dynamic user prompt."""

    def test_compose_system_after_includes(self) -> None:
        raw = _load("generate_name_system.md")
        # The anti-pattern gallery must appear after the prelude includes;
        # trailing tail includes (e.g. _coordinate_conventions.md) may
        # legitimately appear later.
        prelude_end = raw.index('{% include "sn/_exemplars_name_only.md" %}')
        gallery_pos = raw.index("ANTI-PATTERN REFERENCE")
        assert gallery_pos > prelude_end

    def test_compose_system_before_curated_examples(self) -> None:
        raw = _load("generate_name_system.md")
        gallery_pos = raw.index("ANTI-PATTERN REFERENCE")
        examples_pos = raw.index("## Curated Examples")
        assert gallery_pos < examples_pos


class TestSystemPromptRendersWithDefaultContext:
    """Render the system prompt with a representative compose context and
    confirm the instrument anti-pattern examples survive Jinja rendering."""

    @pytest.fixture
    def context(self) -> dict:
        from imas_codex.standard_names.context import build_compose_context

        return build_compose_context()

    def test_compose_system_renders(self, context: dict) -> None:
        rendered = render_prompt("sn/generate_name_system", context)
        assert "ANTI-PATTERN REFERENCE" in rendered
        for bad in ANTI_BAD_EXAMPLES:
            assert bad in rendered
        for good in ANTI_GOOD_EXAMPLES:
            assert good in rendered

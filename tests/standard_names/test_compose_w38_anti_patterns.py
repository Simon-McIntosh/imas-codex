"""Compose-prompt instrument/hardware anti-pattern hardening tests.

Verifies the three diagnostic-IDS anti-patterns (instrument prefix carry-over,
suffix-form for component, compound hardware identifiers) are present in
the system prompt (generate_name_system.md) and the user prompts
(generate_name_dd.md, generate_name_dd_names.md).

The consolidated system prompt teaches these as one ``ANTI-PATTERN REFERENCE``
section plus the ``Hardware / instrument tokens`` field-choice rule; rotation
stage labels (``W38-A1``, ``EMW-1``) were dropped per the naming-hygiene rule
(source must not leak plan/stage identifiers).  These tests assert the
surviving bad/good example pairs and decision rules, not the stage labels.

These are content/structural assertions — they do not call the LLM.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt


def _load(name: str) -> str:
    return (PROMPTS_DIR / "sn" / name).read_text(encoding="utf-8")


# Concrete real bad/good pairs from the diagnostic-IDS rotation.
W38_BAD_EXAMPLES = (
    "x_ray_crystal_spectrometer_pixel_photon_energy_lower_bound",
    "halo_region_parallel_energy_due_to_heat_flux",
    "z_coordinate_of_sensor_direction_unit_vector",
)
W38_GOOD_EXAMPLES = (
    "photon_energy_lower_bound",
    "parallel_halo_energy",
    "z_direction_unit_vector",
)
W38_HARDWARE_PROPERTY_EXEMPLAR = "area_of_rogowski_coil"


@pytest.mark.parametrize("filename", ["generate_name_system.md"])
class TestSystemPromptInstrumentGallery:
    """The instrument anti-pattern entries must appear in the system prompt."""

    def test_gallery_heading(self, filename: str) -> None:
        raw = _load(filename)
        assert "ANTI-PATTERN REFERENCE" in raw

    @pytest.mark.parametrize("bad", W38_BAD_EXAMPLES)
    def test_bad_example_present(self, filename: str, bad: str) -> None:
        assert bad in _load(filename)

    @pytest.mark.parametrize("good", W38_GOOD_EXAMPLES)
    def test_good_example_present(self, filename: str, good: str) -> None:
        assert good in _load(filename)

    def test_hardware_property_exception(self, filename: str) -> None:
        # The gallery must carve out the hardware-property exception so the
        # generator does not over-strip instrument tokens.
        raw = _load(filename)
        assert W38_HARDWARE_PROPERTY_EXEMPLAR in raw
        assert "intrinsic" in raw.lower()


@pytest.mark.parametrize(
    "filename", ["generate_name_dd.md", "generate_name_dd_names.md"]
)
class TestUserPromptW38TableRows:
    """The user-facing per-batch tables must include W38 row entries."""

    def test_w38_tags_in_table(self, filename: str) -> None:
        raw = _load(filename)
        assert "W38-A1" in raw
        assert "W38-A2" in raw
        assert "W38-A3" in raw

    @pytest.mark.parametrize(
        "bad,good",
        list(zip(W38_BAD_EXAMPLES, W38_GOOD_EXAMPLES, strict=True)),
    )
    def test_concrete_pairs(self, filename: str, bad: str, good: str) -> None:
        raw = _load(filename)
        assert bad in raw
        assert good in raw


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
        for bad in W38_BAD_EXAMPLES:
            assert bad in rendered
        for good in W38_GOOD_EXAMPLES:
            assert good in rendered

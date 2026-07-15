"""Tests verifying IMAS coordinate convention guidance is rendered in SN system prompts.

Regression guard: if the shared fragment path breaks or the include is removed,
these tests catch it before bad documentation is generated.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def rendered_compose_system() -> str:
    """Render sn/generate_name_system with full grammar context."""
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import (
        build_compose_context,
        clear_context_cache,
    )

    clear_context_cache()
    context = build_compose_context()
    return render_prompt("sn/generate_name_system", context)


@pytest.fixture()
def rendered_enrich_system() -> str:
    """Render sn/generate_docs_system (static — no dynamic context needed)."""
    from imas_codex.llm.prompt_loader import render_prompt

    return render_prompt("sn/generate_docs_system", {})


@pytest.fixture()
def rendered_review_system() -> str:
    """Render the name reviewer with the same closed grammar context."""
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_compose_context

    return render_prompt("sn/review_names_system", build_compose_context())


class TestCoordinateGuidanceInComposeSystem:
    """Coordinate convention guidance must appear in the compose system prompt."""

    def test_right_handed_phrase_present(self, rendered_compose_system: str) -> None:
        """The phrase 'right-handed' must appear (establishes handedness fact)."""
        assert "right-handed" in rendered_compose_system

    def test_cylindrical_tuple_present(self, rendered_compose_system: str) -> None:
        r"""The explicit (R, \phi, Z) tuple must appear."""
        assert r"(R, \phi, Z)" in rendered_compose_system

    def test_prohibition_phrase_present(self, rendered_compose_system: str) -> None:
        """A prohibition against vague coordinate phrases must be stated."""
        assert (
            "NEVER" in rendered_compose_system
            or "do NOT" in rendered_compose_system
            or "Never" in rendered_compose_system
        ), "Expected a prohibition (NEVER / Never / do NOT) in the compose prompt"

    def test_vague_phrase_named_as_bad_example(
        self, rendered_compose_system: str
    ) -> None:
        """The vague phrase 'standard cylindrical' must appear as a bad example."""
        assert "standard cylindrical" in rendered_compose_system

    def test_coordinate_conventions_section_heading(
        self, rendered_compose_system: str
    ) -> None:
        """The coordinate conventions section heading must be present."""
        assert "Coordinate Convention" in rendered_compose_system or (
            "IMAS coordinate" in rendered_compose_system
        )

    def test_flux_coordinate_guidance_present(
        self, rendered_compose_system: str
    ) -> None:
        """Flux coordinates must be mentioned explicitly (no 'the standard')."""
        assert "flux coordinate" in rendered_compose_system.lower()

    def test_directional_frame_semantics_present(
        self, rendered_compose_system: str
    ) -> None:
        assert "`radial`" in rendered_compose_system
        assert "means only the cylindrical" in rendered_compose_system
        assert "flux_surface_normal" in rendered_compose_system
        assert "perpendicular` remains relative to the" in rendered_compose_system

    def test_local_tangent_semantics_present(
        self, rendered_compose_system: str
    ) -> None:
        assert "first_local_tangential_coordinate" in rendered_compose_system
        assert "second_local_tangential_coordinate" in rendered_compose_system
        assert "never reinterpret the second tangent as" in rendered_compose_system

    def test_coordinate_guidance_in_static_prefix(
        self, rendered_compose_system: str
    ) -> None:
        """Coordinate convention block must sit in the static cacheable prefix.

        Both the coordinate naming rule ('ABSOLUTE RULE') and the basis-tuple
        convention include must appear before the dynamic Curated Examples loop
        so the static prompt prefix (and the OpenRouter cache hit) is maximised.
        """
        coord_idx = rendered_compose_system.find(r"(R, \phi, Z)")
        abs_rule_idx = rendered_compose_system.find("ABSOLUTE RULE")
        examples_idx = rendered_compose_system.find("## Curated Examples")
        assert coord_idx != -1, r"(R, \phi, Z) not found in compose prompt"
        assert abs_rule_idx != -1, "ABSOLUTE RULE not found in compose prompt"
        assert examples_idx != -1, "Curated Examples not found in compose prompt"
        assert coord_idx < examples_idx, (
            "Coordinate convention include must be in the static prefix "
            "(before Curated Examples)"
        )
        assert abs_rule_idx < examples_idx, (
            "Coordinate naming rule must be in the static prefix"
        )


class TestCoordinateGuidanceInEnrichSystem:
    """Coordinate convention guidance must appear in the enrich system prompt."""

    def test_right_handed_phrase_present(self, rendered_enrich_system: str) -> None:
        """The phrase 'right-handed' must appear."""
        assert "right-handed" in rendered_enrich_system

    def test_cylindrical_tuple_present(self, rendered_enrich_system: str) -> None:
        r"""The explicit (R, \phi, Z) tuple must appear."""
        assert r"(R, \phi, Z)" in rendered_enrich_system

    def test_prohibition_phrase_present(self, rendered_enrich_system: str) -> None:
        """A prohibition against vague coordinate phrases must be stated."""
        assert (
            "NEVER" in rendered_enrich_system
            or "do NOT" in rendered_enrich_system
            or "Never" in rendered_enrich_system
        ), "Expected a prohibition (NEVER / Never / do NOT) in the enrich prompt"

    def test_vague_phrase_named_as_bad_example(
        self, rendered_enrich_system: str
    ) -> None:
        """The vague phrase 'standard cylindrical' must appear as a bad example."""
        assert "standard cylindrical" in rendered_enrich_system

    def test_coordinate_conventions_section_heading(
        self, rendered_enrich_system: str
    ) -> None:
        """The coordinate conventions section heading must be present."""
        assert "Coordinate Convention" in rendered_enrich_system or (
            "IMAS coordinate" in rendered_enrich_system
        )

    def test_coordinate_guidance_before_documentation_template(
        self, rendered_enrich_system: str
    ) -> None:
        """Coordinate convention block must precede the Documentation Template section.

        Ensures the conventions are injected before the per-batch dynamic content
        so the static prompt prefix is maximised for caching.
        """
        coord_idx = rendered_enrich_system.find(r"(R, \phi, Z)")
        template_idx = rendered_enrich_system.find("Documentation Template")
        assert coord_idx != -1, r"(R, \phi, Z) not found in enrich prompt"
        assert template_idx != -1, "Documentation Template not found in enrich prompt"
        assert coord_idx < template_idx, (
            "Coordinate convention block must precede Documentation Template"
        )

    def test_cartesian_basis_mentioned(self, rendered_enrich_system: str) -> None:
        """Cartesian basis guidance for sensor vectors must be present."""
        assert (
            r"\hat{x}" in rendered_enrich_system
            or "Cartesian" in rendered_enrich_system
        )


class TestCoordinateGuidanceInReviewSystem:
    """The independent critic must reject the same frame regressions."""

    def test_radial_and_flux_surface_normal_are_distinct(
        self, rendered_review_system: str
    ) -> None:
        assert "`radial` is only cylindrical" in rendered_review_system
        assert "cross-flux-surface vector projections require" in rendered_review_system

    def test_local_x2_cannot_be_verticalised(self, rendered_review_system: str) -> None:
        assert "`x1_coordinate` / `x2_coordinate`" in rendered_review_system
        assert "silently verticalised" in rendered_review_system

"""Prompt regressions for lossless ordered operator expressions."""

from __future__ import annotations

import pytest

pytest.importorskip("imas_standard_names")

from imas_standard_names import compose, parse  # noqa: E402

from imas_codex.llm.prompt_loader import render_prompt  # noqa: E402
from imas_codex.standard_names.context import (  # noqa: E402
    build_compose_context,
    clear_context_cache,
)

_METRIC_NAMES = {
    "gm1": "flux_surface_averaged_inverse_of_square_of_major_radius",
    "gm5": "flux_surface_averaged_square_of_magnetic_field_magnitude",
    "gm6": (
        "flux_surface_averaged_ratio_of"
        "_square_of_toroidal_flux_coordinate_gradient_magnitude"
        "_to_square_of_magnetic_field_magnitude"
    ),
}


@pytest.fixture
def rendered_prompt() -> str:
    clear_context_cache()
    rendered = render_prompt(
        "sn/generate_name_system",
        context=build_compose_context(),
    )
    clear_context_cache()
    return rendered


@pytest.mark.parametrize("name", _METRIC_NAMES.values())
def test_metric_examples_strictly_round_trip(name: str) -> None:
    """Every metric example endorsed by the prompt is strict-valid."""
    assert compose(parse(name, strict=True).ir) == name


def test_prompt_contains_exact_metric_decompositions(rendered_prompt: str) -> None:
    """The three DD aliases carry their complete outer-to-inner decomposition."""
    for alias, name in _METRIC_NAMES.items():
        assert f"DD `{alias}`" in rendered_prompt
        assert name in rendered_prompt

    assert (
        'operators=[{"token":"flux_surface_averaged"},'
        '{"token":"inverse"},{"token":"square"}]'
    ) in rendered_prompt
    assert (
        'operators=[{"token":"flux_surface_averaged"},'
        '{"token":"square"},{"token":"magnitude"}]'
    ) in rendered_prompt
    assert (
        '{"token":"ratio","secondary_operand":"square_of_magnetic_field_magnitude"}'
    ) in rendered_prompt
    assert "numerator and denominator each contain the authored" in rendered_prompt


def test_prompt_preserves_authored_order_semantics(rendered_prompt: str) -> None:
    """Precedence validates a chain without normalizing equal-precedence order."""
    assert "never sorts or rewrites the authored chain" in rendered_prompt
    assert "Equal-precedence chains retain their authored order" in rendered_prompt
    assert "do not emit an explicit `inverse`→`square` chain" not in rendered_prompt


def test_prompt_renders_qualifier_categories_in_validator_order(
    rendered_prompt: str,
) -> None:
    """The context's ordered categories are visible without a codex copy."""
    context = build_compose_context()
    categories = context["grammar"]["vocabularies"]["qualifier_categories"]

    positions = [rendered_prompt.index(f"**{category}**") for category in categories]
    assert positions == sorted(positions)

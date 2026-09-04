"""Rendering guard for property-specific attachment guidance."""

from imas_codex.llm.prompt_loader import render_prompt


def test_equal_axis_leaf_contrasts_render() -> None:
    """The DD composer sees all three property-specific attachment contrasts."""
    rendered = render_prompt(
        "sn/generate_name_dd",
        {
            "items": [],
            "nearby_existing_names": [],
            "reference_exemplars": [],
        },
    )
    normalized = " ".join(rendered.split())
    contrasts = (
        (
            "`boundary/geometric_axis/z` may produce only "
            "`vertical_coordinate_of_geometric_axis`, while "
            "`boundary/dr_dz_zero_point/z` must use an exact registered identity "
            "for the `dr/dz = 0` landmark or emit `vocab_gap`; never attach the "
            "landmark to the geometric-axis coordinate."
        ),
        (
            "`ece/channel/position/z` may produce only "
            "`vertical_coordinate_of_measurement_position`, while "
            "`ece/channel/delta_position_suprathermal/z` must retain its `delta` "
            "and `suprathermal` semantics through an exact registered carrier or "
            "emit `vocab_gap`; never attach the delta as the absolute measurement "
            "position."
        ),
        (
            "`boundary/outline/z` may produce "
            "`vertical_outline_of_plasma_boundary`: this is the positive "
            "property-specific control, and it does not license any other `z` "
            "leaf to share that attachment."
        ),
    )

    for contrast in contrasts:
        assert contrast in normalized

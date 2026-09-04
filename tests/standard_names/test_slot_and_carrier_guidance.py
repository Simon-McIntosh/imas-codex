"""Rendering guard for grammar-slot and carrier-identity guidance."""

from imas_codex.llm.prompt_loader import render_prompt


def test_slot_and_carrier_rules_render() -> None:
    """The DD composer sees both slot ownership and carrier identity rules."""
    rendered = render_prompt(
        "sn/generate_name_dd",
        {
            "items": [],
            "nearby_existing_names": [],
            "reference_exemplars": [],
        },
    )
    normalized = " ".join(rendered.split())

    slot_rule = (
        "A token registered as a `geometric_base` MUST be emitted as "
        '`base_token` with `base_kind: "geometry"`; never emit it as a '
        "`physical_base` or report it missing from `physical_base`."
    )
    carrier_rule = (
        "Each distinct carrier keeps its own name qualified by that carrier, "
        "using an exact registered identity or a `vocab_gap` when the required "
        "carrier token is unavailable."
    )

    assert slot_rule in normalized
    assert "displacement is a registered geometric_base token" in normalized
    assert carrier_rule in normalized
    assert "currently carries 24 bound sources" in normalized
    assert "a delta or displacement never binds to an absolute position identity" in (
        normalized
    )

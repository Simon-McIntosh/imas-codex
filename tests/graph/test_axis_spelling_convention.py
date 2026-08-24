"""Schema-derived spelling invariants for Standard Name review axes."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from imas_codex.graph.schema import GraphSchema

_SCHEMA_PATH = (
    Path(__file__).parents[2] / "imas_codex" / "schemas" / "standard_name.yaml"
)
_AXIS_SLOT_FAMILIES: dict[str, tuple[re.Pattern[str], Callable[[str], str]]] = {
    "reviewer_score": (
        re.compile(r"^reviewer_score_(?P<axis>[a-z][a-z0-9_]*)$"),
        lambda axis: f"reviewer_score_{axis}",
    ),
    "reviewed_at": (
        re.compile(r"^reviewed_(?P<axis>[a-z][a-z0-9_]*)_at$"),
        lambda axis: f"reviewed_{axis}_at",
    ),
    "reviewer_model": (
        re.compile(r"^reviewer_model_(?P<axis>[a-z][a-z0-9_]*)$"),
        lambda axis: f"reviewer_model_{axis}",
    ),
}


def _review_axis_values(schema: GraphSchema) -> tuple[str, set[str]]:
    """Return the enum and values selected by the schema's review-axis slot."""
    enums = schema.get_enums()
    declarations = {
        details["type"]
        for class_name in schema.node_labels
        for slot_name, details in schema.get_all_slots(class_name).items()
        if slot_name == "review_axis" and details["type"] in enums
    }
    assert len(declarations) == 1, (
        f"Expected one enum-backed review_axis declaration, got {sorted(declarations)}"
    )
    enum_name = declarations.pop()
    return enum_name, set(enums[enum_name])


def _paired_axis_slots(
    schema: GraphSchema,
) -> dict[str, dict[str, dict[str, str]]]:
    """Collect axis suffixes represented in every aggregate slot family."""
    paired: dict[str, dict[str, dict[str, str]]] = {}
    for class_name in schema.node_labels:
        slots = schema.get_all_slots(class_name)
        by_family: dict[str, dict[str, str]] = {}
        for family, (pattern, _) in _AXIS_SLOT_FAMILIES.items():
            by_family[family] = {
                match.group("axis"): slot_name
                for slot_name in slots
                if (match := pattern.fullmatch(slot_name)) is not None
            }

        common_suffixes = set.intersection(
            *(set(axis_slots) for axis_slots in by_family.values())
        )
        if common_suffixes:
            paired[class_name] = {
                family: {suffix: axis_slots[suffix] for suffix in common_suffixes}
                for family, axis_slots in by_family.items()
            }
    return paired


def test_review_axis_values_match_paired_slot_suffixes() -> None:
    """Each permissible axis spelling must construct every paired slot name."""
    schema = GraphSchema(_SCHEMA_PATH)
    enum_name, axis_values = _review_axis_values(schema)
    paired_slots = _paired_axis_slots(schema)

    assert paired_slots, "LinkML declares no fully paired review-axis slot families"
    paired_suffixes = {
        suffix
        for family_slots in paired_slots.values()
        for suffix in family_slots["reviewer_score"]
    }
    enum_only = sorted(axis_values - paired_suffixes)
    slot_only = sorted(paired_suffixes - axis_values)
    mismatch = ", ".join(
        f"{enum_value!r} versus _{slot_suffix}"
        for enum_value, slot_suffix in zip(enum_only, slot_only, strict=False)
    )
    assert axis_values == paired_suffixes, (
        f"{enum_name} values and paired LinkML slot suffixes use different "
        f"spellings: {mismatch}; permissible values={sorted(axis_values)}, "
        f"slot suffixes={sorted(paired_suffixes)}"
    )

    for class_name, family_slots in paired_slots.items():
        for axis_value in sorted(axis_values):
            for family, (_, build_slot_name) in _AXIS_SLOT_FAMILIES.items():
                expected = build_slot_name(axis_value)
                actual = family_slots[family].get(axis_value)
                assert actual == expected, (
                    f"{class_name}.{family} does not satisfy slot == "
                    f"prefix + axis value for {axis_value!r}: "
                    f"expected {expected!r}, got {actual!r}"
                )

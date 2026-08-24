"""Schema-owned naming convention for StandardName foreign keys."""

from __future__ import annotations

import re
from pathlib import Path

from imas_codex.graph.schema import GraphSchema

_SCHEMA_PATH = (
    Path(__file__).parents[2] / "imas_codex" / "schemas" / "standard_name.yaml"
)
_STANDARD_NAME_FOREIGN_KEY = re.compile(r"^(?:sn|standard_name)_ids?$")
_RETIRED_REVIEW_RESOLUTION_FIELD = "docs_review_resolution_method"


def _standard_name_foreign_keys(
    schema: GraphSchema,
) -> list[tuple[str, str, bool]]:
    """Return every generic StandardName foreign key and its cardinality."""
    return sorted(
        (
            label,
            slot_name,
            bool(details.get("multivalued")),
        )
        for label in schema.node_labels
        for slot_name, details in schema.get_all_slots(label).items()
        if _STANDARD_NAME_FOREIGN_KEY.fullmatch(slot_name)
        and "standardname" in details.get("description", "").lower()
    )


def _classes_declaring_slot(schema: GraphSchema, slot_name: str) -> list[str]:
    """Return every LinkML class inducing a slot, including abstract classes."""
    return sorted(
        class_name
        for class_name in schema._view.all_classes()
        if slot_name
        in {slot.name for slot in schema._view.class_induced_slots(class_name)}
    )


def test_standard_name_foreign_keys_follow_cardinality_convention() -> None:
    """All generic StandardName foreign keys use the cardinality-derived spelling."""
    schema = GraphSchema(_SCHEMA_PATH)
    foreign_keys = _standard_name_foreign_keys(schema)
    assert foreign_keys, "LinkML declares no generic StandardName foreign keys"

    spellings = sorted({slot_name for _, slot_name, _ in foreign_keys})
    violations = [
        (
            label,
            slot_name,
            "standard_name_ids" if multivalued else "standard_name_id",
            "multivalued" if multivalued else "scalar",
        )
        for label, slot_name, multivalued in foreign_keys
        if slot_name != ("standard_name_ids" if multivalued else "standard_name_id")
    ]
    retired_owners = _classes_declaring_slot(schema, _RETIRED_REVIEW_RESOLUTION_FIELD)

    failures: list[str] = []
    if violations:
        rendered = "; ".join(
            f"{label}.{actual} -> {expected} ({cardinality})"
            for label, actual, expected, cardinality in violations
        )
        failures.append(
            "StandardName foreign keys use inconsistent spellings "
            f"{spellings}; LinkML cardinality requires standard_name_id for "
            "scalar keys and standard_name_ids for multivalued keys. "
            f"Violations: {rendered}"
        )
    if retired_owners:
        failures.append(
            f"Retired field {_RETIRED_REVIEW_RESOLUTION_FIELD} is still declared "
            f"by LinkML classes {retired_owners}"
        )

    assert not failures, "\n".join(failures)

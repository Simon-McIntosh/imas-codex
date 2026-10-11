"""The assembler writes what a binding's transform returns.

A binding's transform expression carries any unit conversion: compose_transform
writes it there and validation refuses an identity transform across differing
units. The assembler must not convert the transformed value a second time.
"""

import logging
from types import SimpleNamespace

from imas_codex.ids.assembler import IDSAssembler


def _assembler(ids_name: str) -> IDSAssembler:
    assembler = IDSAssembler.__new__(IDSAssembler)
    assembler.ids_name = ids_name
    return assembler


def _binding(target_id: str, expression: str, **units) -> SimpleNamespace:
    return SimpleNamespace(
        source_id="jt-60sa:source",
        source_property="value",
        target_id=target_id,
        transform_expression=expression,
        source_units=units.get("source_units"),
        target_units=units.get("target_units"),
    )


def test_converting_transform_is_applied_once_to_a_scalar():
    entry = SimpleNamespace()
    binding = _binding(
        "pf_active/coil/current",
        "convert_units(value, 'kA', 'A')",
        source_units="kA",
        target_units="A",
    )
    _assembler("pf_active")._apply_mappings(entry, {"value": 2.0}, [binding], "coil")
    assert entry.current == 2000.0


def test_field_outside_the_nested_path_is_reported(caplog):
    entry = SimpleNamespace()
    binding = _binding("wall/description_2d/vessel/unit/annular/outline/r", "value")
    with caplog.at_level(logging.WARNING):
        _assembler("wall")._apply_mappings(
            entry,
            {"value": 1.0},
            [binding],
            "description_2d",
            {"nested_path": "limiter.unit"},
        )
    assert "outside the section's nested path" in caplog.text

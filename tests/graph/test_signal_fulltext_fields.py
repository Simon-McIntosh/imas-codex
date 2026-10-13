"""Guard the widened ``facility_signal_text`` index definition.

The signal full-text index is defined once, as a LinkML annotation on
``FacilitySignal`` in ``imas_codex/schemas/facility.yaml``. A searcher's words
live in more than the identifier, so the index must cover the catalogue wording,
alias code, unit and source prose beside the name and node path — a physicist
copies those from a logbook or a wiki table. This module reads that definition
straight from the schema, with no graph, and fails if any field it must carry is
dropped.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "imas_codex" / "schemas" / "facility.yaml"
)

INDEX_NAME = "facility_signal_text"

# What the index must cover: the identifier and prose a signal is known by, plus
# the catalogue wording, alias code, unit and source prose a searcher types.
REQUIRED_PROPERTIES: tuple[str, ...] = (
    "name",
    "description",
    "node_path",
    "keywords",
    "aliases",
    "unit",
    "source_description",
)


def _facility_signal_fulltext(schema_path: Path) -> tuple[str, list[str]]:
    """Read the ``FacilitySignal`` full-text index definition from the schema."""
    document = yaml.safe_load(schema_path.read_text())
    annotation = document["classes"]["FacilitySignal"]["annotations"]["fulltext_index"]
    index_name, _, properties = annotation["value"].partition(":")
    return index_name, [p.strip() for p in properties.split(",") if p.strip()]


def test_facility_signal_text_index_covers_searchable_fields():
    """The signal full-text index carries every field a searcher's words reach."""
    index_name, properties = _facility_signal_fulltext(_SCHEMA_PATH)
    assert index_name == INDEX_NAME
    missing = [name for name in REQUIRED_PROPERTIES if name not in properties]
    assert not missing, f"{INDEX_NAME} index is missing indexed fields: {missing}"

"""Guard the widened ``facility_signal_text`` index definition.

The signal full-text index is defined once, as a LinkML annotation on
``FacilitySignal`` in ``imas_codex/schemas/facility.yaml``, and materialised by
the generated context that ``GraphClient`` reads. A searcher's words live in more
than the identifier, so the index must cover the catalogue wording, alias code,
unit and source prose beside the name and node path. The schema definition and
its generated form are each read through their owner — no graph — and both must
carry every field, so a dropped annotation or a stale regeneration fails here.
"""

from __future__ import annotations

from pathlib import Path

from imas_codex.graph.schema import GraphSchema
from imas_codex.graph.schema_context_data import FULLTEXT_INDEXES

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


def _missing(properties: list[str]) -> list[str]:
    """The required fields absent from ``properties``."""
    return [name for name in REQUIRED_PROPERTIES if name not in properties]


def _schema_index_properties() -> list[str]:
    """Read the ``facility_signal_text`` properties from the LinkML schema."""
    indexes = {
        name: props
        for name, _label, props in GraphSchema(_SCHEMA_PATH).fulltext_indexes
    }
    return indexes[INDEX_NAME]


def test_schema_index_covers_searchable_fields():
    """The schema's index definition carries every field a searcher's words reach."""
    missing = _missing(_schema_index_properties())
    assert not missing, (
        f"{INDEX_NAME} schema definition is missing indexed fields: {missing}"
    )


def test_generated_index_covers_searchable_fields():
    """The generated context the client builds the index from carries the fields."""
    _label, properties = FULLTEXT_INDEXES[INDEX_NAME]
    missing = _missing(properties)
    assert not missing, (
        f"{INDEX_NAME} generated definition is missing indexed fields: {missing}"
    )

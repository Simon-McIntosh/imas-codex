"""Composition follows the strict parser's runtime-owned segment order."""

from imas_codex.standard_names.consolidation import consolidate_candidates
from imas_codex.standard_names.grammar_adapter import (
    is_canonical_name,
    normalize_canonical_name,
    parse_canonical_name,
)


def test_composed_segment_order_is_normalized_before_consolidation() -> None:
    composer_output = "volumetric_electron_source_rate"
    composed = normalize_canonical_name(composer_output).name
    result = consolidate_candidates(
        [
            {
                "id": composed,
                "source_id": "dd:core_sources/source/electrons/particles",
                "source_types": ["dd"],
                "source_paths": ["dd:core_sources/source/electrons/particles"],
                "unit": "m^-3.s^-1",
                "kind": "quantity",
            }
        ]
    )

    assert composed == "electron_volumetric_source_rate"
    assert parse_canonical_name(composed).name == composed
    assert not is_canonical_name(composer_output)
    assert [candidate["id"] for candidate in result.approved] == [
        "electron_volumetric_source_rate"
    ]


def test_consolidation_normalizes_a_parser_reorderable_candidate() -> None:
    result = consolidate_candidates(
        [
            {
                "id": "volumetric_electron_source_rate",
                "source_id": "dd:core_sources/source/electrons/particles",
                "source_types": ["dd"],
                "source_paths": ["dd:core_sources/source/electrons/particles"],
                "unit": "m^-3.s^-1",
                "kind": "quantity",
            }
        ]
    )

    assert [candidate["id"] for candidate in result.approved] == [
        "electron_volumetric_source_rate"
    ]

"""Tests for the lossless public-grammar adapter."""

from __future__ import annotations

import pytest

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.grammar_adapter import (  # noqa: E402
    compose_canonical_ir,
    is_canonical_name,
    parse_canonical_name,
)
from imas_codex.standard_names.models import GrammarSegments  # noqa: E402

METRIC_NAMES = (
    "flux_surface_averaged_inverse_of_square_of_major_radius",
    "flux_surface_averaged_square_of_magnetic_field_magnitude",
    (
        "flux_surface_averaged_ratio_of"
        "_square_of_toroidal_flux_coordinate_gradient_magnitude"
        "_to_square_of_magnetic_field_magnitude"
    ),
)


@pytest.mark.parametrize("name", METRIC_NAMES)
def test_ordered_metric_trees_round_trip_losslessly(name: str) -> None:
    parsed = parse_canonical_name(name)

    assert compose_canonical_ir(parsed.ir) == name


def test_unary_tree_order_is_outermost_first() -> None:
    parsed = parse_canonical_name(METRIC_NAMES[0])

    assert [operator.op for operator in parsed.ir.operators] == [
        "flux_surface_averaged",
        "inverse",
        "square",
    ]


def test_binary_operands_retain_their_nested_operator_order() -> None:
    parsed = parse_canonical_name(METRIC_NAMES[2])
    binary = next(
        operator for operator in parsed.ir.operators if operator.kind.value == "binary"
    )

    assert [operator.op for operator in binary.args[0].operators] == [
        "square",
        "magnitude",
    ]
    assert [operator.op for operator in binary.args[1].operators] == [
        "square",
        "magnitude",
    ]


def test_invalid_operator_precedence_is_rejected() -> None:
    assert not is_canonical_name("gradient_of_maximum_of_pressure")

    with pytest.raises(ValueError, match="precedence"):
        parse_canonical_name("gradient_of_maximum_of_pressure")


def test_flat_projection_failure_does_not_invalidate_ordered_tree() -> None:
    from imas_standard_names.grammar import parse_standard_name

    name = METRIC_NAMES[0]
    with pytest.raises(ValueError):
        parse_standard_name(name)

    assert is_canonical_name(name)


@pytest.mark.parametrize(
    ("segments", "expected"),
    [
        (
            {
                "base_token": "radius",
                "base_kind": "quantity",
                "qualifiers": ["major"],
                "operators": [
                    {"token": "flux_surface_averaged"},
                    {"token": "inverse"},
                    {"token": "square"},
                ],
            },
            METRIC_NAMES[0],
        ),
        (
            {
                "base_token": "magnetic_field",
                "base_kind": "quantity",
                "operators": [
                    {"token": "flux_surface_averaged"},
                    {"token": "square"},
                    {"token": "magnitude"},
                ],
            },
            METRIC_NAMES[1],
        ),
        (
            {
                "base_token": "toroidal_flux_coordinate_gradient",
                "base_kind": "quantity",
                "operators": [
                    {"token": "flux_surface_averaged"},
                    {
                        "token": "ratio",
                        "secondary_operand": "square_of_magnetic_field_magnitude",
                    },
                    {"token": "square"},
                    {"token": "magnitude"},
                ],
            },
            METRIC_NAMES[2],
        ),
    ],
)
def test_structured_segments_generate_exact_metric_names(
    segments: dict, expected: str
) -> None:
    assert GrammarSegments(**segments).compose_name() == expected


def test_binary_derivation_reads_real_operands_not_placeholder_base() -> None:
    from imas_codex.standard_names.derivation import derive_edges

    outer = [
        edge for edge in derive_edges(METRIC_NAMES[2]) if edge.edge_type == "HAS_PARENT"
    ]
    assert len(outer) == 1
    assert outer[0].props["operator"] == "flux_surface_averaged"

    parents = [
        edge
        for edge in derive_edges(outer[0].to_name)
        if edge.edge_type == "HAS_PARENT"
    ]

    assert [(edge.props["role"], edge.to_name) for edge in parents] == [
        (
            "a",
            "square_of_toroidal_flux_coordinate_gradient_magnitude",
        ),
        ("b", "square_of_magnetic_field_magnitude"),
    ]


def test_ordered_metric_name_does_not_enter_grammar_retry() -> None:
    from imas_codex.standard_names.workers import _grammar_round_trip_failures

    class Candidate:
        source_id = "metric"

        def compose_name(self) -> str:
            return METRIC_NAMES[0]

    assert _grammar_round_trip_failures([Candidate()]) == ([], [])


def test_ordered_intermediate_parent_passes_admission_identity() -> None:
    from imas_codex.standard_names.parents import (
        _has_structural_specificity,
        _has_valid_standard_name_identity,
    )

    name = "inverse_of_square_of_major_radius"

    assert _has_valid_standard_name_identity(name)[0]
    assert _has_structural_specificity(name)[0]

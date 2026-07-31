"""Tests for the lossless public-grammar adapter."""

from __future__ import annotations

import pytest

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.grammar_adapter import (  # noqa: E402
    ParsedCanonicalName,
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

ISOTOPE_RATIO = (
    "ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density"
    "_and_neutral_density_of_isotope"
)
DOUBLED_ISOTOPE_LOCUS = (
    "ratio_of_neutral_density_to_difference_of_total_neutral_density"
    "_and_neutral_density_of_isotope_of_isotope"
)


def _binary_ir(operator: str, left: str, right: str, separator: str):
    """Build a binary public IR from two strict canonical operand names."""
    from imas_standard_names import StandardNameIR

    return StandardNameIR.model_validate(
        {
            "operators": [
                {
                    "kind": "binary",
                    "op": operator,
                    "args": [
                        parse_canonical_name(left).ir,
                        parse_canonical_name(right).ir,
                    ],
                    "separator": separator,
                }
            ],
            "base": {"token": "placeholder", "kind": "quantity"},
        }
    )


def _nested_isotope_ratio_ir():
    """Return the intended n_i / (n_total - n_i) recursive expression."""
    from imas_standard_names import StandardNameIR

    difference = _binary_ir(
        "difference",
        "total_neutral_density",
        "neutral_density_of_isotope",
        "and",
    )
    return StandardNameIR.model_validate(
        {
            "operators": [
                {
                    "kind": "binary",
                    "op": "ratio",
                    "args": [
                        parse_canonical_name("neutral_density_of_isotope").ir,
                        difference,
                    ],
                    "separator": "to",
                }
            ],
            "base": {"token": "placeholder", "kind": "quantity"},
        }
    )


def _normalized_radial_ir():
    """Return the public IR before its flat normalized-axis canonicalization."""
    from imas_standard_names import StandardNameIR

    data = parse_canonical_name("radial_electric_field").ir.model_dump(mode="python")
    data["operators"] = [
        {
            "kind": "unary_prefix",
            "op": "normalized",
            "bare_prefix": True,
        }
    ]
    return StandardNameIR.model_validate(data)


def _with_inverse(ir, *, outer: bool):
    """Add inverse outside or inside an existing public operator chain."""
    from imas_standard_names import StandardNameIR

    data = ir.model_dump(mode="python")
    inverse = {
        "kind": "unary_prefix",
        "op": "inverse",
        "bare_prefix": False,
    }
    data["operators"] = (
        [inverse, *data["operators"]] if outer else [*data["operators"], inverse]
    )
    return StandardNameIR.model_validate(data)


def _strict_parser_preserves(ir) -> bool:
    """Report whether the installed public parser preserves this exact tree."""
    from imas_standard_names import compose, parse

    try:
        parsed = parse(compose(ir), strict=True).ir
    except ValueError:
        return False
    return parsed.model_dump(mode="json") == ir.model_dump(mode="json")


def _strict_parser_rejects(name: str) -> bool:
    """Report rejection using only the installed public ISN grammar."""
    from imas_standard_names import compose, parse

    try:
        parsed = parse(name, strict=True)
    except ValueError:
        return True
    return compose(parsed.ir) != name


_INTENDED_ISOTOPE_IR = _nested_isotope_ratio_ir()
_ISN_PRESERVES_NESTED_ISOTOPE_SCOPE = _strict_parser_preserves(_INTENDED_ISOTOPE_IR)
_ISN_REJECTS_DOUBLED_ISOTOPE_LOCUS = _strict_parser_rejects(DOUBLED_ISOTOPE_LOCUS)


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


def test_structured_locus_stays_on_binary_operand_a(monkeypatch) -> None:
    """The candidate's base fields qualify operand A, not the expression."""
    from imas_codex.standard_names import grammar_adapter

    captured = {}

    def capture(ir) -> str:
        captured["ir"] = ir
        return "captured_binary_expression"

    monkeypatch.setattr(grammar_adapter, "compose_canonical_ir", capture)
    segments = GrammarSegments(
        base_token="density",
        base_kind="quantity",
        qualifiers=["neutral"],
        locus_token="isotope",
        locus_relation="of",
        locus_type="entity",
        operators=[{"token": "ratio", "secondary_operand": "density"}],
    )

    assert segments.compose_name() == "captured_binary_expression"
    binary = captured["ir"].operators[0]
    assert captured["ir"].locus is None
    assert binary.args[0].locus.token == "isotope"
    assert binary.args[1].locus is None


def test_semantic_guard_rejects_binary_operand_scope_drift() -> None:
    """Reject a synthetic dependency result that hoists a leaf locus.

    The test-owned ``ParsedCanonicalName`` simulates a dependency parser
    returning the candidate's canonical surface while changing recursive
    scope.  It is not a valid public parse result.
    """
    from imas_codex.standard_names import grammar_adapter

    intended = _binary_ir(
        "ratio",
        "neutral_density_of_isotope",
        "total_neutral_density_of_isotope",
        "to",
    )
    shifted = _binary_ir(
        "ratio", "neutral_density_of_isotope", "total_neutral_density", "to"
    )
    shifted_data = shifted.model_dump(mode="python")
    shifted_data["locus"] = parse_canonical_name("density_of_isotope").ir.locus
    from imas_standard_names import StandardNameIR, compose

    shifted = StandardNameIR.model_validate(shifted_data)
    name = compose(intended)
    simulated_dependency_result = ParsedCanonicalName(
        name=name,
        ir=shifted,
    )

    with pytest.raises(ValueError, match="changes the semantic IR"):
        grammar_adapter._require_semantic_ir(
            intended,
            simulated_dependency_result.ir,
            name=simulated_dependency_result.name,
        )


def test_public_flat_projection_normalization_remains_supported() -> None:
    assert compose_canonical_ir(_normalized_radial_ir()) == (
        "normalized_radial_electric_field"
    )


def test_innermost_flat_projection_normalization_preserves_outer_operator() -> None:
    intended = _with_inverse(_normalized_radial_ir(), outer=True)

    assert compose_canonical_ir(intended) == (
        "inverse_of_normalized_radial_electric_field"
    )


def test_flat_normalization_cannot_cross_an_inner_operator(monkeypatch) -> None:
    """Absorbing an outer token across an inner operator changes precedence."""
    from imas_standard_names import StandardNameIR

    from imas_codex.standard_names import grammar_adapter

    intended = _with_inverse(_normalized_radial_ir(), outer=False)
    parsed_data = intended.model_dump(mode="python")
    parsed_data["operators"] = parsed_data["operators"][1:]
    parsed_data["projection"]["axis"] = "normalized_radial"
    parsed = StandardNameIR.model_validate(parsed_data)
    monkeypatch.setattr(
        grammar_adapter,
        "parse_canonical_name",
        lambda name: ParsedCanonicalName(name=name, ir=parsed),
    )

    with pytest.raises(ValueError, match="changes the semantic IR"):
        grammar_adapter.compose_canonical_ir(intended)


@pytest.mark.parametrize("drift", ["operator", "projection", "qualifier"])
def test_unrelated_flat_semantic_drift_is_rejected(monkeypatch, drift) -> None:
    """The compatibility path cannot erase or invent unrelated semantics."""
    from imas_standard_names import StandardNameIR, compose

    from imas_codex.standard_names import grammar_adapter

    intended = _normalized_radial_ir()
    parsed_data = intended.model_dump(mode="python")
    if drift == "operator":
        parsed_data["operators"] = [
            {
                "kind": "unary_prefix",
                "op": "inverse",
                "bare_prefix": False,
            }
        ]
    elif drift == "projection":
        parsed_data["operators"] = []
        parsed_data["projection"]["axis"] = "normalized_toroidal"
    else:
        parsed_data["operators"] = []
        parsed_data["projection"]["axis"] = "normalized_radial"
        parsed_data["qualifiers"] = [{"token": "electron"}]
    parsed = StandardNameIR.model_validate(parsed_data)

    monkeypatch.setattr(
        grammar_adapter,
        "parse_canonical_name",
        lambda name: ParsedCanonicalName(name=name, ir=parsed),
    )

    with pytest.raises(ValueError, match="changes the semantic IR"):
        grammar_adapter.compose_canonical_ir(intended)
    assert compose(intended) == "normalized_radial_electric_field"


def test_numerator_only_locus_composes_on_operand_a() -> None:
    segments = GrammarSegments(
        base_token="density",
        base_kind="quantity",
        qualifiers=["neutral"],
        locus_token="isotope",
        locus_relation="of",
        locus_type="entity",
        operators=[{"token": "ratio", "secondary_operand": "density"}],
    )

    assert segments.compose_name() == "ratio_of_neutral_density_of_isotope_to_density"


@pytest.mark.xfail(
    condition=not _strict_parser_preserves(
        _binary_ir(
            "ratio",
            "neutral_density_of_isotope",
            "total_neutral_density_of_isotope",
            "to",
        )
    ),
    reason="installed ISN does not yet preserve the terminal operand locus",
    strict=True,
)
def test_loci_on_both_binary_leaves_round_trip_semantically() -> None:
    segments = GrammarSegments(
        base_token="density",
        base_kind="quantity",
        qualifiers=["neutral"],
        locus_token="isotope",
        locus_relation="of",
        locus_type="entity",
        operators=[
            {
                "token": "ratio",
                "secondary_operand": "total_neutral_density_of_isotope",
            }
        ],
    )

    assert segments.compose_name() == (
        "ratio_of_neutral_density_of_isotope_to_total_neutral_density_of_isotope"
    )


@pytest.mark.xfail(
    condition=not _ISN_PRESERVES_NESTED_ISOTOPE_SCOPE,
    reason="installed ISN does not yet preserve nested terminal operand scope",
    strict=True,
)
def test_nested_ratio_difference_round_trips_semantically() -> None:
    segments = GrammarSegments(
        base_token="density",
        base_kind="quantity",
        qualifiers=["neutral"],
        locus_token="isotope",
        locus_relation="of",
        locus_type="entity",
        operators=[
            {
                "token": "ratio",
                "secondary_operand": (
                    "difference_of_total_neutral_density_and_neutral_density_of_isotope"
                ),
            }
        ],
    )

    assert segments.compose_name() == ISOTOPE_RATIO


@pytest.mark.xfail(
    condition=not _ISN_REJECTS_DOUBLED_ISOTOPE_LOCUS,
    reason="installed ISN does not yet reject repeated terminal locus scope",
    strict=True,
)
def test_doubled_terminal_locus_is_not_canonical() -> None:
    assert not is_canonical_name(DOUBLED_ISOTOPE_LOCUS)


@pytest.mark.parametrize(
    ("segments", "expected"),
    [
        (
            {
                "base_token": "temperature",
                "base_kind": "quantity",
                "qualifiers": ["electron"],
                "operators": [{"token": "inverse"}],
            },
            "inverse_of_electron_temperature",
        ),
        (
            {
                "base_token": "radius",
                "base_kind": "quantity",
                "qualifiers": ["major"],
                "operators": [{"token": "ratio", "secondary_operand": "minor_radius"}],
            },
            "ratio_of_major_radius_to_minor_radius",
        ),
    ],
)
def test_ordinary_operator_composition_is_unchanged(segments, expected) -> None:
    assert GrammarSegments(**segments).compose_name() == expected


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


def test_benchmark_resolves_ordered_segments_losslessly() -> None:
    from imas_codex.standard_names.benchmark import _resolve_name

    candidate = {
        "segments": {
            "base_token": "radius",
            "base_kind": "quantity",
            "qualifiers": ["major"],
            "operators": [
                {"token": "flux_surface_averaged"},
                {"token": "inverse"},
                {"token": "square"},
            ],
        }
    }

    assert _resolve_name(candidate) == METRIC_NAMES[0]

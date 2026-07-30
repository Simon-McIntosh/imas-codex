"""Ordered ISN trees stay lossless at graph and export boundaries."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imas_codex.standard_names.export import _run_gate_b
from imas_codex.standard_names.graph_ops import (
    _parse_grammar,
    _write_grammar_decomposition,
    _write_standard_name_edges,
    write_standard_names,
)

INVERSE_SQUARE_RADIUS = "flux_surface_averaged_inverse_of_square_of_major_radius"
SQUARE_FIELD_MAGNITUDE = "flux_surface_averaged_square_of_magnetic_field_magnitude"
GRADIENT_FIELD_RATIO = (
    "ratio_of_square_of_toroidal_flux_coordinate_gradient"
    "_to_square_of_magnetic_field_magnitude"
)


def _operator_batch(gc: MagicMock) -> list[dict]:
    for call in gc.query.call_args_list:
        cypher = call.args[0]
        if "SET r.operator" in cypher:
            return call.kwargs["batch"]
    raise AssertionError("HAS_PARENT operator write was not emitted")


def _column_write(gc: MagicMock) -> dict:
    for call in gc.query.call_args_list:
        if "SET sn.physical_base" in call.args[0]:
            return call.kwargs
    raise AssertionError("grammar compatibility columns were not written")


def test_unary_ir_projects_outermost_operator_and_leaf_base() -> None:
    inverse = _parse_grammar(INVERSE_SQUARE_RADIUS)
    magnitude = _parse_grammar(SQUARE_FIELD_MAGNITUDE)

    assert inverse["transformation"] == "flux_surface_averaged"
    assert inverse["physical_base"] == "radius"
    assert magnitude["transformation"] == "flux_surface_averaged"
    assert magnitude["physical_base"] == "magnetic_field"


def test_binary_ir_uses_primary_operand_instead_of_placeholder() -> None:
    projected = _parse_grammar(GRADIENT_FIELD_RATIO)

    assert projected["transformation"] == "ratio"
    assert projected["physical_base"] == "toroidal_flux_coordinate_gradient"
    assert projected["physical_base"] != "placeholder"


def test_decomposition_keeps_columns_for_ordered_unary_tree() -> None:
    gc = MagicMock()
    gc.query.return_value = []

    gaps = _write_grammar_decomposition(gc, [INVERSE_SQUARE_RADIUS])

    assert gaps == []
    written = _column_write(gc)
    assert written["transformation"] == "flux_surface_averaged"
    assert written["physical_base"] == "radius"


def test_graph_writer_expands_complete_unary_operator_order() -> None:
    gc = MagicMock()
    gc.query.return_value = []

    with patch(
        "imas_codex.standard_names.graph_ops._filter_admissible_parents",
        side_effect=lambda batch, _gc, **_kwargs: batch,
    ):
        _write_standard_name_edges(gc, [{"id": INVERSE_SQUARE_RADIUS}])

    edges = _operator_batch(gc)
    by_child = {edge["from_name"]: edge for edge in edges}
    assert by_child[INVERSE_SQUARE_RADIUS]["operator"] == "flux_surface_averaged"
    assert by_child[INVERSE_SQUARE_RADIUS]["to_name"] == (
        "inverse_of_square_of_major_radius"
    )
    assert by_child["inverse_of_square_of_major_radius"]["operator"] == "inverse"
    assert by_child["inverse_of_square_of_major_radius"]["to_name"] == (
        "square_of_major_radius"
    )
    assert by_child["square_of_major_radius"]["operator"] == "square"
    assert by_child["square_of_major_radius"]["to_name"] == "major_radius"


def test_graph_writer_expands_postfix_operator_in_authored_position() -> None:
    gc = MagicMock()
    gc.query.return_value = []

    with patch(
        "imas_codex.standard_names.graph_ops._filter_admissible_parents",
        side_effect=lambda batch, _gc, **_kwargs: batch,
    ):
        _write_standard_name_edges(gc, [{"id": SQUARE_FIELD_MAGNITUDE}])

    edges = _operator_batch(gc)
    by_child = {edge["from_name"]: edge for edge in edges}
    assert by_child[SQUARE_FIELD_MAGNITUDE]["operator"] == "flux_surface_averaged"
    assert by_child["square_of_magnetic_field_magnitude"]["operator"] == "square"
    assert by_child["magnetic_field_magnitude"]["operator"] == "magnitude"
    assert by_child["magnetic_field_magnitude"]["operator_kind"] == "unary_postfix"


def test_graph_writer_expands_both_binary_operand_trees() -> None:
    gc = MagicMock()
    gc.query.return_value = []

    with patch(
        "imas_codex.standard_names.graph_ops._filter_admissible_parents",
        side_effect=lambda batch, _gc, **_kwargs: batch,
    ):
        _write_standard_name_edges(gc, [{"id": GRADIENT_FIELD_RATIO}])

    edges = _operator_batch(gc)
    root_edges = [edge for edge in edges if edge["from_name"] == GRADIENT_FIELD_RATIO]
    assert {edge["role"] for edge in root_edges} == {"a", "b"}
    assert {edge["to_name"] for edge in root_edges} == {
        "square_of_toroidal_flux_coordinate_gradient",
        "square_of_magnetic_field_magnitude",
    }
    nested = {(edge["from_name"], edge["operator"], edge["to_name"]) for edge in edges}
    assert (
        "square_of_toroidal_flux_coordinate_gradient",
        "square",
        "toroidal_flux_coordinate_gradient",
    ) in nested
    assert (
        "square_of_magnetic_field_magnitude",
        "square",
        "magnetic_field_magnitude",
    ) in nested
    assert ("magnetic_field_magnitude", "magnitude", "magnetic_field") in nested


def test_graph_write_gate_accepts_ordered_tree() -> None:
    gc = MagicMock()
    gc.query.return_value = []

    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient") as graph_client,
        patch(
            "imas_codex.standard_names.protection.filter_protected",
            side_effect=lambda names, **_kwargs: (names, []),
        ),
        patch(
            "imas_codex.standard_names.graph_ops._filter_admissible_parents",
            side_effect=lambda batch, _gc, **_kwargs: batch,
        ),
    ):
        graph_client.return_value.__enter__.return_value = gc
        graph_client.return_value.__exit__.return_value = False
        written = write_standard_names([{"id": INVERSE_SQUARE_RADIUS, "unit": "m^-2"}])

    assert written == 1
    assert any(
        call.kwargs.get("batch", [{}])[0].get("id") == INVERSE_SQUARE_RADIUS
        for call in gc.query.call_args_list
        if call.kwargs.get("batch")
    )


def test_export_gate_accepts_lossless_ordered_trees() -> None:
    candidates = [
        {"id": name, "cocos": 17, "links": []}
        for name in (
            INVERSE_SQUARE_RADIUS,
            SQUARE_FIELD_MAGNITUDE,
            GRADIENT_FIELD_RATIO,
        )
    ]

    result = _run_gate_b(candidates, cocos_convention=17)

    assert result.passed
    assert not [
        issue for issue in result.issues if issue["type"] == "grammar_parse_failure"
    ]

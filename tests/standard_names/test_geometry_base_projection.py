"""Regression coverage for strict-IR geometry base projection."""

import pytest
from imas_standard_names import compose, parse

from imas_codex.standard_names.graph_ops import _segments_from_ir


@pytest.mark.parametrize(
    "name",
    [
        "toroidal_coordinate_of_shatter_cone",
        "toroidal_coordinate_of_reflectometer_antenna",
    ],
)
def test_coordinate_geometry_base_projects_to_geometric_column(name: str) -> None:
    parsed = parse(name, strict=True)

    assert compose(parsed.ir) == name
    assert parsed.ir.base.kind is type(parsed.ir.base.kind).GEOMETRY

    columns = _segments_from_ir(parsed.ir)

    assert columns["geometric_base"] == "coordinate"
    assert columns["physical_base"] is None
    assert columns["coordinate"] == "toroidal"

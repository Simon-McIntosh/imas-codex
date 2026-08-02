"""Write-boundary guards for semantic source attachments."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.attachment_audit import guard_source_pairings
from imas_codex.standard_names.workers import _is_attachment_consistent


@pytest.mark.parametrize(
    "primitive",
    ["thick_line", "outline", "rectangle", "oblique", "arcs_of_circle"],
)
def test_solid_hardware_geometry_cannot_source_optical_locus(primitive: str) -> None:
    path = f"pf_active/coil/element/geometry/{primitive}/r"

    accepted, reason = _is_attachment_consistent(
        path,
        "radial_coordinate_of_line_of_sight",
        dd_unit="m",
        sn_unit="m",
    )

    assert not accepted
    assert "geometry representation mismatch" in reason


def test_optical_path_cannot_source_solid_geometry_locus() -> None:
    accepted, reason = _is_attachment_consistent(
        "bremsstrahlung_visible/channel/line_of_sight/first_point/r",
        "radial_coordinate_of_rectangle",
        dd_unit="m",
        sn_unit="m",
    )

    assert not accepted
    assert "geometry representation mismatch" in reason


def test_valid_sightline_coordinate_remains_accepted() -> None:
    accepted, reason = _is_attachment_consistent(
        "bremsstrahlung_visible/channel/line_of_sight/first_point/r",
        "radial_coordinate_of_line_of_sight",
        dd_unit="m",
        sn_unit="m",
    )

    assert accepted, reason


def test_valid_hardware_geometry_coordinate_remains_accepted() -> None:
    accepted, reason = _is_attachment_consistent(
        "pf_active/coil/element/geometry/rectangle/r",
        "radial_coordinate_of_rectangle",
        dd_unit="m",
        sn_unit="m",
    )

    assert accepted, reason


def test_pairing_guard_preserves_existing_and_rejects_only_fresh_conflict() -> None:
    class FakeGraph:
        def query(self, _cypher: str, **_params):
            return [
                {
                    "source_id": "dd:existing",
                    "source_type": "dd",
                    "dd_path": "pf_active/coil/element/geometry/rectangle/r",
                    "dd_unit": "m",
                    "sn_unit": "m",
                    "already_bound": True,
                    "existing_dd_paths": [],
                    "name_stage": "drafted",
                },
                {
                    "source_id": "dd:fresh",
                    "source_type": "dd",
                    "dd_path": "pf_active/coil/element/geometry/rectangle/r",
                    "dd_unit": "m",
                    "sn_unit": "m",
                    "already_bound": False,
                    "existing_dd_paths": [],
                    "name_stage": "drafted",
                },
            ]

    result = guard_source_pairings(
        FakeGraph(),
        "radial_coordinate_of_line_of_sight",
        ["dd:fresh", "dd:existing"],
    )

    assert result.accepted_source_ids == ("dd:existing",)
    assert [item.source_node_id for item in result.rejected] == ["dd:fresh"]


def test_bulk_writer_does_not_materialize_rejected_dd_pairing() -> None:
    from imas_codex.standard_names.graph_ops import write_standard_names

    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def query(self, cypher: str, **params):
            self.calls.append((cypher, params))
            if "existing_dd_paths" in cypher:
                return [
                    {
                        "id": "radial_coordinate_of_line_of_sight",
                        "source_id": ("pf_active/coil/element/geometry/rectangle/r"),
                        "dd_unit": "m",
                        "sn_unit": "m",
                        "existing_dd_paths": [],
                    }
                ]
            return []

    gc = FakeGraph()
    written = write_standard_names(
        [
            {
                "id": "radial_coordinate_of_line_of_sight",
                "source_types": ["dd"],
                "source_id": "pf_active/coil/element/geometry/rectangle/r",
                "unit": "m",
            }
        ],
        gc=gc,
    )

    projection_calls = [
        params
        for cypher, params in gc.calls
        if "MERGE (src)-[:HAS_STANDARD_NAME]->(sn)" in cypher
    ]
    assert written == 1
    assert projection_calls == [{"batch": []}]

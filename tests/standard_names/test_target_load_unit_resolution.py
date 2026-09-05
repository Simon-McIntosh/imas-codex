"""Graph storage contract for the divertor-target-load unit resolution.

Four DD sources feed ``maximum_of_energy_flux_at_divertor_target``: two wall
``global_quantities`` paths declare ``W`` for what is documented as a peak
areal power density, while two sibling paths for the identical quantity
(``divertors/divertor/target/power_flux_peak`` and
``summary/local/divertor_target/power_flux_peak/value``) declare ``W.m^-2``.
The DD contradicts itself on one physical quantity, so the two ``W`` paths
carry a governed resolution rather than a rename derived from a majority vote.
"""

from datetime import UTC, datetime

import pytest

from imas_codex.graph.models import (
    DDResolutionField,
    DDResolutionStatus,
    DDResolutionValueKind,
)
from imas_codex.standard_names.dd_resolutions import (
    TARGET_LOAD_RESOLUTION_PATHS,
    DDResolutionGraphPortConflict,
    DDResolutionManifest,
    DDResolutionRecord,
    DDResolutionValue,
    active_dd_resolution,
    target_load_resolution_records,
)


def _existing_target_load(path: str) -> DDResolutionRecord:
    return DDResolutionRecord(
        id=f"dd_resolution:existing:{path}",
        gap_id=f"dd_gap:{path}:unit_defect",
        path=path,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        observed=DDResolutionValue(kind=DDResolutionValueKind.string, value="W"),
        effective=DDResolutionValue(kind=DDResolutionValueKind.string, value="W.m^-2"),
        reason="A peak areal power density is W.m^-2, not total power W.",
        recorded_by="test operator",
        recorded_at=datetime(2026, 9, 5, tzinfo=UTC),
        upstream_reference="none-yet",
        retiring_release="none-yet",
        state=DDResolutionStatus.active,
    )


def test_target_load_cohort_is_exactly_the_two_wall_paths() -> None:
    assert set(TARGET_LOAD_RESOLUTION_PATHS) == {
        "wall/global_quantities/power_density_inner_target_max",
        "wall/global_quantities/power_density_outer_target_max",
    }


def test_cohort_builder_adds_both_paths_when_none_exist() -> None:
    records = target_load_resolution_records(
        manifest=DDResolutionManifest(resolutions=()),
        recorded_by="test operator",
        recorded_at=datetime(2026, 9, 5, tzinfo=UTC),
        reason="A peak areal power density is W.m^-2, not total power W.",
    )
    assert len(records) == 2
    assert {record["path"] for record in records} == set(TARGET_LOAD_RESOLUTION_PATHS)
    assert all(record["published_value"] == '"W"' for record in records)
    assert all(record["effective_value"] == '"W.m^-2"' for record in records)
    assert all(record["evidence"].endswith(":unit_defect") for record in records)
    assert all(record["upstream_reference"] == "none-yet" for record in records)


def test_cohort_builder_is_idempotent_against_matching_existing_records() -> None:
    manifest = DDResolutionManifest(
        resolutions=tuple(
            _existing_target_load(path) for path in TARGET_LOAD_RESOLUTION_PATHS
        )
    )
    records = target_load_resolution_records(
        manifest=manifest,
        recorded_by="test operator",
        recorded_at=datetime(2026, 9, 5, tzinfo=UTC),
        reason="A peak areal power density is W.m^-2, not total power W.",
    )
    assert records == ()


def test_cohort_builder_refuses_conflicting_existing_authority() -> None:
    path = TARGET_LOAD_RESOLUTION_PATHS[0]
    conflicting = DDResolutionRecord(
        id="dd_resolution:existing:conflict",
        gap_id=f"dd_gap:{path}:unit_defect",
        path=path,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        observed=DDResolutionValue(kind=DDResolutionValueKind.string, value="W"),
        effective=DDResolutionValue(kind=DDResolutionValueKind.string, value="keV"),
        reason="wrong effective unit",
        recorded_by="test operator",
        recorded_at=datetime(2026, 9, 5, tzinfo=UTC),
        upstream_reference="none-yet",
        retiring_release="none-yet",
        state=DDResolutionStatus.active,
    )
    with pytest.raises(DDResolutionGraphPortConflict, match=path):
        target_load_resolution_records(
            manifest=DDResolutionManifest(resolutions=(conflicting,)),
            recorded_by="test operator",
            recorded_at=datetime(2026, 9, 5, tzinfo=UTC),
            reason="A peak areal power density is W.m^-2, not total power W.",
        )


@pytest.mark.graph
def test_live_target_load_cohort_has_exact_edges_and_units() -> None:
    """The registered resolution must survive: both wall paths stay W.m^-2.

    This is the regression gate for the rename-unit-authority agreement: if
    either wall path's resolution or corrected unit disappears, the rename
    derivation in imas_codex.standard_names.edit will refuse again.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as graph:
        rows = graph.query(
            """
            UNWIND $paths AS path
            MATCH (node:IMASNode {id: path})-[:BRIDGED_BY]->(resolution:DDResolution)
            MATCH (resolution)-[:EVIDENCED_BY]->(gap:DDGap)
            MATCH (resolution)-[:FOR_DD_VERSION]->(version:DDVersion)
            OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
            RETURN path, node.unit AS unit, collect(DISTINCT unit.id) AS unit_ids,
                   count(DISTINCT resolution) AS resolutions,
                   count(DISTINCT gap) AS gaps,
                   count(DISTINCT version) AS versions,
                   collect(DISTINCT gap.kind) AS gap_kinds,
                   collect(DISTINCT gap.status) AS gap_statuses
            ORDER BY path
            """,
            paths=list(TARGET_LOAD_RESOLUTION_PATHS),
        )
    assert len(rows) == 2
    for row in rows:
        assert row["unit"] == "W.m^-2"
        assert row["unit_ids"] == ["W.m^-2"]
        assert (row["resolutions"], row["gaps"], row["versions"]) == (1, 1, 1)
        assert row["gap_kinds"] == ["unit_defect"]
        assert row["gap_statuses"] == ["flagged"]

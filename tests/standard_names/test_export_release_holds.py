"""Release-authority holds are identity-bearing export exclusions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import patch

from imas_codex.standard_names.export import (
    _RELEASE_IDENTITY_HOLDS,
    _classify_export_population,
    run_export,
)
from tests.standard_names.conftest import clean_source_audit_graph

_DD_SOURCELESS_HELD_IDENTITIES = {
    "electrostatic_potential_at_outboard_midplane",
    "electrostatic_potential_at_wall",
    "root_mean_square_spectral_width_of_spectrometer_channel",
    "count_normalized_at_detector_pixel",
    "coolant_absorbed_energy_accumulated_of_plasma_facing_component",
    "volume_averaged_electron_number_density_over_scrape_off_layer",
}
_HELD_IDENTITIES_TO_REASONS = {
    "fast_ion_charge_state_power_at_inside_flux_surface": (
        "release_hold_dd_recipient_unresolved"
    ),
    "toroidal_coordinate_of_field_map_grid": (
        "release_hold_field_map_grid_vocabulary_unresolved"
    ),
    "neutron_flux_due_to_fusion": "release_hold_documentation_not_accepted",
    "radial_neutral_internal_state_momentum_flux": (
        "release_hold_dual_bound_source_conflict"
    ),
    "voltage_of_diagnostic_antenna": "release_hold_exhausted_antenna_identity",
    "voltage_of_ece_channel": "release_hold_missing_reviewed_successor",
} | dict.fromkeys(_DD_SOURCELESS_HELD_IDENTITIES, "release_hold_no_dd_source")
_INCLUDED_IDENTITIES = {
    "tendency_of_total_thermal_plasma_internal_energy",
    "radial_neutral_state_momentum_flux",
}


@clean_source_audit_graph
class _ReadOnlyGraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        return []


def _candidate(name: str) -> dict:
    return {
        "id": name,
        "name_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "_validation_observed_at": "2026-09-09T00:00:00Z",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "_has_docs_review": True,
        "_has_winning_docs_review": True,
        "docs_review_quorum_shortfall": None,
        "reviewer_score_name": 0.95,
        "description": f"Description for {name}.",
        "documentation": f"Documentation for {name}.",
        "kind": "scalar",
        "unit": "1",
        "physics_domain": "general",
        "links": [],
        # The topology the population query projects onto every candidate: a
        # Data Dictionary extraction reached this name directly. Without
        # producer evidence the eligibility classifier withholds the candidate,
        # so a fixture that stands for a real population row carries it.
        "_has_dd_source_binding": True,
        "_has_derived_producer": False,
        "_has_non_derived_producer": True,
        "_has_live_child": False,
        "_is_parent": False,
    }


def _run_fixture_export(staging_dir: Path, population: list[dict]):
    with (
        patch(
            "imas_codex.standard_names.export._fetch_export_population",
            return_value=population,
        ),
        patch(
            "imas_codex.graph.client.GraphClient",
            return_value=_ReadOnlyGraphClient(),
        ),
        patch(
            "imas_codex.standard_names.export._validate_entry",
            side_effect=lambda entry: entry,
        ),
        patch(
            "imas_codex.standard_names.export._fetch_deprecation_stubs",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
            return_value=([], set()),
        ),
        patch("imas_codex.standard_names.export._write_domain_yaml"),
    ):
        return run_export(
            staging_dir,
            min_score=0.85,
            skip_gate=True,
            force=True,
            include_sources=False,
        )


def test_each_held_identity_is_excluded_with_its_reason() -> None:
    population = [_candidate(name) for name in _HELD_IDENTITIES_TO_REASONS]

    eligible, excluded = _classify_export_population(
        population,
        domain=None,
        names_only=False,
    )

    assert eligible == []
    assert {
        record.standard_name_id: record.reason for record in excluded
    } == _HELD_IDENTITIES_TO_REASONS
    assert {
        name: reason for name, (reason, _) in _RELEASE_IDENTITY_HOLDS.items()
    } == _HELD_IDENTITIES_TO_REASONS
    assert {record.stage for record in excluded} == {"release_authority"}


def test_dd_sourceless_holds_name_the_missing_measure() -> None:
    """Six live names with no source in the active DD are held from release.

    No Data Dictionary path measures them, so no binding can be made and the
    hold records why. The reason is shared because the condition is one
    condition; the detail is what a reviewer reads to see the measure is
    absent from the DD rather than the name being unreviewed.
    """
    for name in _DD_SOURCELESS_HELD_IDENTITIES:
        reason, detail = _RELEASE_IDENTITY_HOLDS[name]
        assert reason == "release_hold_no_dd_source"
        assert "Data Dictionary path" in detail


def test_explicitly_included_identities_remain_eligible() -> None:
    population = [_candidate(name) for name in _INCLUDED_IDENTITIES]

    eligible, excluded = _classify_export_population(
        population,
        domain=None,
        names_only=False,
    )

    assert {candidate["id"] for candidate in eligible} == _INCLUDED_IDENTITIES
    assert excluded == []


def test_release_holds_close_the_identity_ledger(tmp_path: Path) -> None:
    population = [
        *(_candidate(name) for name in _HELD_IDENTITIES_TO_REASONS),
        *(_candidate(name) for name in _INCLUDED_IDENTITIES),
    ]

    report = _run_fixture_export(tmp_path, population)
    rows = {row["reason"]: row for row in report.to_dict()["exclusion_ledger"]}

    assert report.all_gates_passed
    assert report.total_candidates == len(population)
    assert set(report.exported_names) == _INCLUDED_IDENTITIES
    assert set(rows) == set(_HELD_IDENTITIES_TO_REASONS.values())
    assert {reason: row["count"] for reason, row in rows.items()} == dict(
        Counter(_HELD_IDENTITIES_TO_REASONS.values())
    )
    assert report.exported_count + sum(row["count"] for row in rows.values()) == len(
        population
    )

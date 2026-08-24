"""Release-authority holds are identity-bearing export exclusions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from imas_codex.standard_names.export import (
    _RELEASE_IDENTITY_HOLDS,
    _classify_export_population,
    run_export,
)

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
}
_INCLUDED_IDENTITIES = {
    "tendency_of_total_thermal_plasma_internal_energy",
    "radial_neutral_state_momentum_flux",
}


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
        "validation_status": "valid",
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


def test_each_held_identity_has_its_own_exclusion_reason() -> None:
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
    assert len({record.reason for record in excluded}) == len(
        _HELD_IDENTITIES_TO_REASONS
    )
    assert {record.stage for record in excluded} == {"release_authority"}


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
    assert all(row["count"] == 1 for row in rows.values())
    assert report.exported_count + sum(row["count"] for row in rows.values()) == len(
        population
    )

"""Flux-surface-reduction gate: family eligibility + live-name audit.

The ISN grammar rejects flux-surface reduction operators
(``flux_surface_averaged``, ``maximum/minimum_over_flux_surface``) applied
to a base flagged ``constant_on_flux_surface`` (a flux function — the
reduction is a no-op). Graph-side, ``find_flux_surface_reduction_violations``
flags any live name that survives from before the gate; harmonize's family
grouping admits the reduction operators so a reduced/unreduced pair
docs-harmonizes together.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.standard_names.audits import (
    find_flux_surface_reduction_violations,
)
from imas_codex.standard_names.harmonize import (
    _FAMILY_OPERATOR_KINDS,
    _FAMILY_REDUCTION_OPERATORS,
)


class TestFamilyReductionOperators:
    def test_reduction_operators_registered(self):
        assert set(_FAMILY_REDUCTION_OPERATORS) == {
            "flux_surface_averaged",
            "maximum_over_flux_surface",
            "minimum_over_flux_surface",
        }

    def test_broad_prefix_class_stays_out_of_kinds(self):
        # The kind filter must not blanket-admit unary_prefix — only the
        # named reduction operators join via the operator-token filter.
        assert "unary_prefix" not in _FAMILY_OPERATOR_KINDS


class TestFindFluxSurfaceReductionViolations:
    def _gc(self, rows):
        gc = MagicMock()
        gc.query.return_value = rows
        return gc

    def test_flags_reduction_of_flux_function(self):
        gc = self._gc(
            [
                {
                    "id": "flux_surface_averaged_safety_factor_at_plasma_boundary",
                    "name_stage": "accepted",
                },
                {
                    "id": "flux_surface_averaged_electron_density_at_plasma_boundary",
                    "name_stage": "accepted",
                },
            ]
        )
        violations = find_flux_surface_reduction_violations(gc=gc)
        assert [v["id"] for v in violations] == [
            "flux_surface_averaged_safety_factor_at_plasma_boundary"
        ]
        assert "constant on a flux surface" in violations[0]["reason"]
        gc.close.assert_not_called()

    def test_clean_catalog_returns_empty(self):
        gc = self._gc(
            [
                {
                    "id": "flux_surface_averaged_electron_temperature_at_plasma_boundary",
                    "name_stage": "accepted",
                }
            ]
        )
        assert find_flux_surface_reduction_violations(gc=gc) == []

    def test_unparseable_names_are_not_this_gate(self):
        gc = self._gc(
            [{"id": "flux_surface_averaged_zzz_nonsense", "name_stage": "draft"}]
        )
        assert find_flux_surface_reduction_violations(gc=gc) == []

"""Tests for the _parent_supports_uncertainty_index semantic gate (Phase C).

Verifies that mint_error_siblings() skips uncertainty_index_of_<P> siblings
when the parent name or unit is semantically unsuitable, while still minting
upper/lower uncertainty siblings unconditionally.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: produce a pass-through ISN grammar mock so integration tests don't
# depend on the imas_standard_names package parsing specific name strings.
# ---------------------------------------------------------------------------


def _make_isn_passthrough():
    """Return a pair of (parse_mock, compose_mock) that echo input unchanged."""

    def _parse(name: str):
        mock_result = MagicMock()
        mock_result.ir = name
        return mock_result

    def _compose(ir):
        return ir  # identity

    return _parse, _compose


# ---------------------------------------------------------------------------
# Unit tests for _parent_supports_uncertainty_index
# ---------------------------------------------------------------------------


class TestParentSupportsUncertaintyIndex:
    """Direct unit tests for the gate helper function."""

    def test_allow_temperature(self):
        """The policy gate returns False for every parent.

        An uncertainty index is bookkeeping rather than a physical quantity,
        so Rule 6 closes the gate unconditionally — including for dimensional
        scalars, which an earlier gate allowed.
        """
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("electron_temperature", "eV") is False

    def test_allow_current(self):
        """Policy gate: a dimensional scalar (A) is blocked by Rule 6."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("plasma_current", "A") is False

    def test_allow_ion_density(self):
        """Policy gate: a dimensional scalar (m^-3) is blocked by Rule 6."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("ion_density", "m^-3") is False

    def test_deny_process_term(self):
        """Name containing _due_to_ → denied (process attribution)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert (
            _parent_supports_uncertainty_index("power_due_to_thermalization", "W")
            is False
        )

    def test_deny_caused_by_pattern(self):
        """Name containing caused_by_ → denied (process attribution)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert (
            _parent_supports_uncertainty_index("energy_caused_by_radiation", "J")
            is False
        )

    def test_deny_dimensionless_empty(self):
        """Empty unit string → denied (dimensionless)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("safety_factor", "") is False

    def test_deny_unit_one(self):
        """Unit '1' → denied (explicitly dimensionless)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("safety_factor", "1") is False

    def test_deny_unit_none(self):
        """Unit None → denied (no unit = dimensionless)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("some_quantity", None) is False

    def test_deny_unit_dash(self):
        """Unit '-' → denied (dimensionless dash convention)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("some_quantity", "-") is False

    def test_deny_status_suffix(self):
        """Name ending in _status → denied (categorical field)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("plasma_status", "") is False

    def test_deny_type_suffix(self):
        """Name ending in _type → denied (categorical field)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("ion_type", "") is False

    def test_deny_index_suffix(self):
        """Name ending in _index with dimensionless unit → denied."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("phase_index", "") is False

    def test_deny_id_suffix(self):
        """Name ending in _id → denied (identifier field)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("node_id", "") is False

    def test_deny_label_suffix(self):
        """Name ending in _label → denied (categorical label)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("grid_label", "") is False

    def test_deny_constant_prefix(self):
        """Name starting with constant_ → denied (data-type descriptor)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("constant_float_value", "m") is False

    def test_deny_generic_prefix(self):
        """Name starting with generic_ → denied (data-type descriptor)."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert _parent_supports_uncertainty_index("generic_quantity", "Pa") is False


# ---------------------------------------------------------------------------
# Integration tests for mint_error_siblings (gate wired in)
# ---------------------------------------------------------------------------


class TestMintErrorSiblingsGate:
    """Integration tests verifying the gate is applied inside mint_error_siblings."""

    def test_mint_skips_denied_parent(self):
        """Process-term parent → no uncertainty_index sibling produced."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        parse_mock, compose_mock = _make_isn_passthrough()

        with (
            patch(
                "imas_standard_names.grammar.parser.parse",
                side_effect=parse_mock,
            ),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=compose_mock,
            ),
        ):
            siblings = mint_error_siblings(
                "power_due_to_thermalization",
                error_node_ids=[
                    "fast_particles/power_due_to_thermalization_error_index",
                ],
                unit="W",
                physics_domain="heating",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert not any("uncertainty_index" in sid for sid in ids), (
            f"Expected no uncertainty_index sibling, got: {ids}"
        )

    def test_mint_allows_approved_parent(self):
        """uncertainty_index is not produced for any parent.

        Rule 6 blocks the sibling even for a dimensional parent (eV), which
        an earlier gate allowed.  upper/lower uncertainty siblings are still
        produced (not gated).
        """
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        parse_mock, compose_mock = _make_isn_passthrough()

        with (
            patch(
                "imas_standard_names.grammar.parser.parse",
                side_effect=parse_mock,
            ),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=compose_mock,
            ),
        ):
            siblings = mint_error_siblings(
                "electron_temperature",
                error_node_ids=[
                    "core_profiles/profiles_1d/electrons/temperature_error_index",
                ],
                unit="eV",
                physics_domain="transport",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert not any("uncertainty_index" in sid for sid in ids), (
            f"uncertainty_index siblings must not be produced, got: {ids}"
        )

    def test_upper_lower_not_blocked_for_denied_parent(self):
        """Gate only applies to _error_index; upper/lower always pass through."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        parse_mock, compose_mock = _make_isn_passthrough()

        with (
            patch(
                "imas_standard_names.grammar.parser.parse",
                side_effect=parse_mock,
            ),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=compose_mock,
            ),
        ):
            siblings = mint_error_siblings(
                "power_due_to_thermalization",
                error_node_ids=[
                    "fast_particles/power_due_to_thermalization_error_upper",
                    "fast_particles/power_due_to_thermalization_error_lower",
                    "fast_particles/power_due_to_thermalization_error_index",
                ],
                unit="W",
                physics_domain="heating",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        # upper and lower pass through, index is blocked
        assert len(siblings) == 2, f"Expected 2 siblings (upper+lower), got: {ids}"
        assert any("upper_uncertainty" in sid for sid in ids)
        assert any("lower_uncertainty" in sid for sid in ids)
        assert not any("uncertainty_index" in sid for sid in ids)

    def test_dimensionless_unit_blocks_index_only(self):
        """Dimensionless unit blocks uncertainty_index but not upper/lower."""
        from imas_codex.standard_names.error_siblings import mint_error_siblings

        parse_mock, compose_mock = _make_isn_passthrough()

        with (
            patch(
                "imas_standard_names.grammar.parser.parse",
                side_effect=parse_mock,
            ),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=compose_mock,
            ),
        ):
            siblings = mint_error_siblings(
                "safety_factor",
                error_node_ids=[
                    "x/safety_factor_error_upper",
                    "x/safety_factor_error_lower",
                    "x/safety_factor_error_index",
                ],
                unit="1",  # dimensionless
                physics_domain="equilibrium",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert len(siblings) == 2, f"Expected 2 siblings (upper+lower), got: {ids}"
        assert not any("uncertainty_index" in sid for sid in ids)


class TestUncertaintyIndexNeverLeaksViaErrorSiblings:
    """uncertainty_index_of_* must not reach the graph via error_siblings.

    The error_siblings pipeline mints uncertainty_index_of_<parent>
    deterministically from parent names + error_node_ids, so it bypasses the
    extract_deny gate, which only applies to DD path extraction.  A gate keyed
    on dimensionality lets the sibling through for dimensional parents such as
    a current density (A.m^-2).  Rule 6 in _parent_supports_uncertainty_index
    therefore returns False unconditionally, blocking every
    uncertainty_index_of_* sibling regardless of the parent's unit.
    """

    def test_current_density_component_blocked(self):
        """A current-density component must not produce uncertainty_index."""
        from unittest.mock import MagicMock, patch

        from imas_codex.standard_names.error_siblings import mint_error_siblings

        def _parse(name):
            m = MagicMock()
            m.ir = name
            return m

        with (
            patch("imas_standard_names.grammar.parser.parse", side_effect=_parse),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=lambda ir: ir,
            ),
        ):
            siblings = mint_error_siblings(
                "vertical_inertial_current_density",
                error_node_ids=[
                    "edge_profiles/ggd/j_inertial/z_error_index",
                ],
                unit="A.m^-2",
                physics_domain="edge_plasma_physics",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert not any("uncertainty_index" in sid for sid in ids), (
            f"uncertainty_index_of_vertical_inertial_"
            f"current_density must not be generated, got: {ids}"
        )

    def test_diamagnetic_current_density_blocked(self):
        """A diamagnetic current-density component is blocked."""
        from unittest.mock import MagicMock, patch

        from imas_codex.standard_names.error_siblings import mint_error_siblings

        def _parse(name):
            m = MagicMock()
            m.ir = name
            return m

        with (
            patch("imas_standard_names.grammar.parser.parse", side_effect=_parse),
            patch(
                "imas_standard_names.grammar.render.compose",
                side_effect=lambda ir: ir,
            ),
        ):
            siblings = mint_error_siblings(
                "radial_diamagnetic_current_density",
                error_node_ids=[
                    "edge_profiles/ggd/j_diamagnetic/radial_error_index",
                ],
                unit="A.m^-2",
                physics_domain="edge_plasma_physics",
                cocos_type=None,
                cocos_version=None,
                dd_version="4.0.0",
            )

        ids = [s["id"] for s in siblings]
        assert not any("uncertainty_index" in sid for sid in ids), (
            f"uncertainty_index_of_radial_diamagnetic_"
            f"current_density must not be generated, got: {ids}"
        )

    def test_geometry_dimension_prefix_blocked(self):
        """Rule 5 regression: length_of_* parents must not produce uncertainty_index."""
        from imas_codex.standard_names.error_siblings import (
            _parent_supports_uncertainty_index,
        )

        assert (
            _parent_supports_uncertainty_index("length_of_magnetic_field_probe", "m")
            is False
        )
        assert (
            _parent_supports_uncertainty_index(
                "major_radius_of_magnetic_field_probe", "m"
            )
            is False
        )
        assert (
            _parent_supports_uncertainty_index("minor_radius_of_plasma_boundary", "m")
            is False
        )

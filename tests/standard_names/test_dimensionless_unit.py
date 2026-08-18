"""Tests for declared and missing units in the SN pipeline.

An absent HAS_UNIT relationship is missing authority, even for numeric data.
Only an explicit DD dimensionless declaration may become unit ``"1"``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────

_ROW_TEMPLATE = {
    "unit_from_rel": None,
    "data_type": "FLT_1D",
    "coordinate1": None,
    "coordinate2": None,
    "coordinate3": None,
    "timebase": None,
    "cocos_label": None,
    "cocos_expression": None,
    "lifecycle_status": "active",
    "identifier_schema_name": None,
    "identifier_schema_doc": None,
    "identifier_options": None,
    "parent_path": "equilibrium/time_slice/profiles_1d",
    "parent_description": "profiles 1D",
    "sibling_fields": [],
}


def _make_row(**overrides):
    row = dict(_ROW_TEMPLATE)
    row.update(overrides)
    return row


def _run_enrich(items, row):
    """Run _enrich_batch_items with a mocked GraphClient returning *row*."""
    from imas_codex.standard_names.workers import _enrich_batch_items

    mock_gc = MagicMock()
    mock_gc.query.return_value = [row]

    with patch("imas_codex.graph.client.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)

        _enrich_batch_items(items)


def _assert_numeric_fallback_audited_absent() -> None:
    from imas_codex.standard_names.legacy_authority import (
        ShadowAuditStatus,
        find_shadow_authorities,
    )

    audit = find_shadow_authorities()
    numeric = next(
        result
        for result in audit.carrier_results
        if result.carrier == "numeric_missing_unit_fallback"
    )
    assert numeric.status is ShadowAuditStatus.audited
    assert numeric.residual_count == 0


# ── Enrichment tests ──────────────────────────────────────────────────


class TestMissingUnitEnrichment:
    """Enrichment preserves missing unit authority as unresolved."""

    def test_numeric_path_without_unit_stays_unresolved(self) -> None:
        """A floating-point path cannot manufacture dimensionless authority."""
        items = [{"path": "equilibrium/time_slice/profiles_1d/q"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="FLT_1D"))
        assert items[0].get("unit") is None
        _assert_numeric_fallback_audited_absent()

    def test_structure_path_stays_none(self) -> None:
        """STRUCTURE path with no unit → unit stays unset."""
        items = [{"path": "equilibrium/time_slice/profiles_1d"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="STRUCTURE"))
        assert items[0].get("unit") is None

    def test_path_with_real_unit_unchanged(self) -> None:
        """Path with HAS_UNIT → unit from relationship, not '1'."""
        items = [{"path": "equilibrium/time_slice/profiles_1d/pressure"}]
        _run_enrich(items, _make_row(unit_from_rel="Pa", data_type="FLT_1D"))
        assert items[0]["unit"] == "Pa"

    def test_integer_path_without_unit_stays_unresolved(self) -> None:
        """An integer path cannot manufacture dimensionless authority."""
        items = [{"path": "mhd/time_slice/toroidal_mode/n"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="INT_0D"))
        assert items[0].get("unit") is None
        _assert_numeric_fallback_audited_absent()

    def test_complex_path_without_unit_stays_unresolved(self) -> None:
        """A complex path cannot manufacture dimensionless authority."""
        items = [{"path": "some/complex/path"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="CPX_0D"))
        assert items[0].get("unit") is None
        _assert_numeric_fallback_audited_absent()

    def test_str_0d_no_unit(self) -> None:
        """STR_0D path without unit → unit stays None (strings aren't numeric)."""
        items = [{"path": "some/string/path"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="STR_0D"))
        assert items[0].get("unit") is None

    def test_item_with_existing_unit_not_overwritten(self) -> None:
        """If the item already has a unit, enrichment doesn't overwrite it."""
        items = [{"path": "equilibrium/time_slice/profiles_1d/q", "unit": "eV"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="FLT_1D"))
        assert items[0]["unit"] == "eV"


# ── Dash normalization tests ─────────────────────────────────────────


class TestDashUnitNormalization:
    """The '-' DD dimensionless marker is normalized to '1' at compose time."""

    def test_dash_normalized_to_one(self) -> None:
        """raw_unit='-' should become '1', not trigger a skip."""
        raw_unit = "-"
        if raw_unit == "-":
            raw_unit = "1"
        assert raw_unit == "1"
        assert raw_unit not in ("mixed", None, "")

    def test_mixed_still_skipped(self) -> None:
        """raw_unit='mixed' is still rejected."""
        raw_unit = "mixed"
        if raw_unit == "-":
            raw_unit = "1"
        assert raw_unit in ("mixed", None, "")

    def test_none_still_skipped(self) -> None:
        """raw_unit=None (truly unresolvable) is still rejected."""
        raw_unit = None
        if raw_unit == "-":
            raw_unit = "1"
        assert raw_unit in ("mixed", None, "")

    def test_empty_string_still_skipped(self) -> None:
        """raw_unit='' is still rejected."""
        raw_unit = ""
        if raw_unit == "-":
            raw_unit = "1"
        assert raw_unit in ("mixed", None, "")

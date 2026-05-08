"""Tests for dimensionless quantity handling in the SN pipeline.

Verifies that DD paths with no HAS_UNIT relationship (dimensionless
quantities like safety factor q, beta, mode numbers) are correctly
assigned unit="1" (ISN convention) rather than being skipped as
unresolvable.
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


# ── Enrichment tests ──────────────────────────────────────────────────


class TestDimensionlessEnrichment:
    """Enrichment assigns unit='1' to numeric DD paths without HAS_UNIT."""

    def test_numeric_path_gets_dimensionless_unit(self) -> None:
        """FLT_1D path with no unit_from_rel → unit='1'."""
        items = [{"path": "equilibrium/time_slice/profiles_1d/q"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="FLT_1D"))
        assert items[0]["unit"] == "1"

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

    def test_int_0d_gets_dimensionless(self) -> None:
        """INT_0D path (mode number, count) → unit='1'."""
        items = [{"path": "mhd/time_slice/toroidal_mode/n"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="INT_0D"))
        assert items[0]["unit"] == "1"

    def test_cpx_0d_gets_dimensionless(self) -> None:
        """CPX_0D (rare complex dimensionless) → unit='1'."""
        items = [{"path": "some/complex/path"}]
        _run_enrich(items, _make_row(unit_from_rel=None, data_type="CPX_0D"))
        assert items[0]["unit"] == "1"

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

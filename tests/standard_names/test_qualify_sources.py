"""Tests for the _qualify_sources pipeline function in dd.py."""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names.sources.dd import _qualify_sources


def _row(path: str, **kw: object) -> dict:
    return {
        "path": path,
        "data_type": kw.pop("data_type", "FLT_1D"),
        "unit": kw.pop("unit", "Pa"),
        "description": kw.pop("description", "Test quantity"),
        **kw,
    }


class TestQualifySources:
    """Test the _qualify_sources pipeline function."""

    def test_eligible_rows_kept(self) -> None:
        rows = [
            _row("equilibrium/time_slice/profiles_1d/psi", unit="Wb"),
            _row("core_profiles/profiles_1d/electrons/temperature", unit="eV"),
        ]
        result = _qualify_sources(rows, write_skipped=False)
        assert len(result) == 2

    def test_ineligible_rows_removed(self) -> None:
        rows = [
            _row("equilibrium/time_slice/profiles_1d/psi", unit="Wb"),
            _row("core_instant_changes/change/density"),
            _row("core_profiles/profiles_1d/electrons/temperature", unit="eV"),
        ]
        result = _qualify_sources(rows, write_skipped=False)
        assert len(result) == 2
        # The ineligible one (core_instant_changes) is removed
        paths = [r["path"] for r in result]
        assert "core_instant_changes/change/density" not in paths

    def test_mixed_unit_rejected(self) -> None:
        rows = [
            _row("some/path/value", unit="mixed"),
            _row("equilibrium/time_slice/profiles_1d/psi", unit="Wb"),
        ]
        result = _qualify_sources(rows, write_skipped=False)
        assert len(result) == 1
        assert result[0]["path"] == "equilibrium/time_slice/profiles_1d/psi"

    def test_string_type_rejected(self) -> None:
        rows = [
            _row("core_profiles/profiles_1d/label", data_type="STR_0D"),
            _row("equilibrium/time_slice/profiles_1d/psi", unit="Wb"),
        ]
        result = _qualify_sources(rows, write_skipped=False)
        assert len(result) == 1

    def test_process_path_rejected(self) -> None:
        rows = [
            _row("edge_transport/model/ggd/process/density"),
            _row("equilibrium/time_slice/profiles_1d/psi", unit="Wb"),
        ]
        result = _qualify_sources(rows, write_skipped=False)
        assert len(result) == 1

    def test_empty_input(self) -> None:
        result = _qualify_sources([], write_skipped=False)
        assert result == []

    def test_all_rejected(self) -> None:
        rows = [
            _row("core_instant_changes/change/density"),
            _row("core_profiles/profiles_1d/label", data_type="STR_0D"),
        ]
        result = _qualify_sources(rows, write_skipped=False)
        assert result == []

    def test_skip_records_written(self) -> None:
        """Verify skip records are passed to write_skipped_sources."""
        rows = [
            _row("core_instant_changes/change/density"),
        ]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources",
            return_value=1,
        ) as mock_write:
            result = _qualify_sources(rows, write_skipped=True)
            assert result == []
            assert mock_write.called
            records = mock_write.call_args[0][0]
            assert len(records) == 1
            assert records[0]["skip_reason"] == "duplicate_ids"

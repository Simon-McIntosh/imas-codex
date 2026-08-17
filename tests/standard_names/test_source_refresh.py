"""Tests for the DD source-drift refresh (idempotent re-refine on source change)."""

from __future__ import annotations

from imas_codex.standard_names import source_refresh as sr


class _FakeGC:
    """Minimal GraphClient stand-in: returns canned rows from ``query``."""

    def __init__(self, rows):
        self._rows = rows
        self.writes: list[tuple[str, dict]] = []

    def query(self, cypher, **kw):
        # Record writes; return canned rows for reads.
        if "SET" in cypher and "RETURN" not in cypher:
            self.writes.append((cypher, kw))
            return []
        return self._rows

    def close(self):
        pass


class _CaptureGC:
    """Capture a snapshot query and report one stamped name."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def query(self, cypher, **kw):
        self.calls.append((cypher, kw))
        if "RETURN sn.id AS sn_id" in cypher:
            return [
                {
                    "sn_id": "gas_flow",
                    "path": "gas_injection/pipe/flow_rate",
                    "unit": "s^-1",
                    "documentation": "Particle flow rate.",
                }
            ]
        return []


def test_norm_collapses_none_and_strips():
    assert sr._norm(None) == ""
    assert sr._norm("  x ") == "x"
    assert sr._norm("W.m^-2") == "W.m^-2"
    assert sr._norm(3) == "3"


def test_format_reason_reports_precise_delta():
    reason = sr._format_reason(
        "neutral_energy_flux_at_wall",
        [{"field": "units", "old": "m^-2.s^-1", "new": "W.m^-2"}],
    )
    assert "units" in reason
    assert "m^-2.s^-1" in reason and "W.m^-2" in reason
    assert "source-refresh" in reason  # framed as targeted refresh, not rewrite


def test_format_reason_truncates_long_documentation():
    long_old = "A" * 400
    reason = sr._format_reason(
        "x", [{"field": "documentation", "old": long_old, "new": "B"}]
    )
    assert "…" in reason  # long doc is truncated for the steering reason


def _row(
    old_unit, new_unit, old_doc="d", new_doc="d", old_path="wall/x", new_path="wall/x"
):
    return {
        "sn_id": "some_name",
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "old_unit": old_unit,
        "new_unit": new_unit,
        "old_doc": old_doc,
        "new_doc": new_doc,
        "old_path": old_path,
        "new_path": new_path,
    }


def test_detect_drift_units_change():
    out = sr.detect_source_drift(gc=_FakeGC([_row("m^-2.s^-1", "W.m^-2")]))
    assert len(out) == 1
    assert out[0]["deltas"] == [{"field": "units", "old": "m^-2.s^-1", "new": "W.m^-2"}]
    assert out[0]["renamed"] is False


def test_detect_drift_documentation_change():
    out = sr.detect_source_drift(
        gc=_FakeGC([_row("W.m^-2", "W.m^-2", "old doc", "new doc")])
    )
    assert len(out) == 1
    assert [d["field"] for d in out[0]["deltas"]] == ["documentation"]


def test_detect_drift_path_rename():
    out = sr.detect_source_drift(
        gc=_FakeGC(
            [
                _row(
                    "W.m^-2",
                    "W.m^-2",
                    old_path="x/torque_fast_tor",
                    new_path="x/torque_fast_phi",
                )
            ]
        )
    )
    assert len(out) == 1
    assert out[0]["renamed"] is True
    assert out[0]["deltas"][0]["field"] == "source_path"
    assert out[0]["new_path"] == "x/torque_fast_phi"


def test_format_reason_rename_labelled():
    reason = sr._format_reason(
        "x", [{"field": "source_path", "old": "a/tor", "new": "a/phi"}]
    )
    assert "path renamed" in reason


def test_detect_drift_ignores_whitespace_only_change():
    # normalised-equal values must not be reported as drift (idempotency guard)
    out = sr.detect_source_drift(gc=_FakeGC([_row("W.m^-2 ", "W.m^-2", "d ", " d")]))
    assert out == []


def test_refresh_no_drift_is_noop():
    # No rows -> nothing detected, nothing steered (safe on every run).
    summary = sr.refresh_drifted_sources(gc=_FakeGC([]), dry_run=True)
    assert summary["detected"] == 0
    assert summary["steered"] == 0


def test_stamp_source_snapshot_targets_only_gas_flow_cache():
    gc = _CaptureGC()

    stamped = sr.stamp_source_snapshots(["gas_flow"], gc=gc)

    assert stamped == 1
    assert len(gc.calls) == 2
    cypher, params = gc.calls[0]
    assert "sn.id IN $sn_ids" in cypher
    assert params["sn_ids"] == ["gas_flow"]
    write, write_params = gc.calls[1]
    assert "sn.source_unit = update.unit" in write
    [snapshot] = write_params["updates"]
    assert snapshot["sn_id"] == "gas_flow"
    assert snapshot["path"] == "gas_injection/pipe/flow_rate"
    assert snapshot["unit"] == "s^-1"
    assert snapshot["documentation"] == "Particle flow rate."
    assert snapshot["raw_unit"] == "s^-1"
    assert snapshot["raw_documentation"] == "Particle flow rate."
    assert snapshot["resolution_ids"] == []
    assert snapshot["converged_ids"] == []
    assert snapshot["manifest_digest"].startswith("sha256:")
    assert snapshot["resolution_marker"] == "resolved-dd-context"
    assert "sn.source_raw_unit = update.raw_unit" in write
    assert "sn.source_dd_resolution_marker = update.resolution_marker" in write

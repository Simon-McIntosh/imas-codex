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
    reason = sr._format_reason("x", [{"field": "documentation", "old": long_old, "new": "B"}])
    assert "…" in reason  # long doc is truncated for the steering reason


def _row(old_unit, new_unit, old_doc="d", new_doc="d"):
    return {
        "sn_id": "some_name",
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "old_unit": old_unit,
        "new_unit": new_unit,
        "old_doc": old_doc,
        "new_doc": new_doc,
        "source_id": "wall/x/values",
    }


def test_detect_drift_units_change():
    out = sr.detect_source_drift(gc=_FakeGC([_row("m^-2.s^-1", "W.m^-2")]))
    assert len(out) == 1
    assert out[0]["deltas"] == [
        {"field": "units", "old": "m^-2.s^-1", "new": "W.m^-2"}
    ]


def test_detect_drift_documentation_change():
    out = sr.detect_source_drift(gc=_FakeGC([_row("W.m^-2", "W.m^-2", "old doc", "new doc")]))
    assert len(out) == 1
    assert [d["field"] for d in out[0]["deltas"]] == ["documentation"]


def test_detect_drift_ignores_whitespace_only_change():
    # normalised-equal values must not be reported as drift (idempotency guard)
    out = sr.detect_source_drift(gc=_FakeGC([_row("W.m^-2 ", "W.m^-2", "d ", " d")]))
    assert out == []


def test_refresh_no_drift_is_noop():
    # No rows -> nothing detected, nothing steered (safe on every run).
    summary = sr.refresh_drifted_sources(gc=_FakeGC([]), dry_run=True)
    assert summary["detected"] == 0
    assert summary["steered"] == 0

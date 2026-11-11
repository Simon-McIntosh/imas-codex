"""A DD path move that ``RENAMED_TO`` lineage explains is a rename, not a steer.

The startup drift refresh used to treat every ``source_path`` difference as
documentation drift, so a path the Data Dictionary had merely renamed reset an
accepted name's documentation for no content change. These tests pin the three
distinctions the refresh must keep: a lineage-only move re-stamps without
steering and counts as a rename, a lineage move that also changed the unit still
steers, and a path change no lineage explains is still a path change.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.standard_names import source_refresh as sr

SNAPSHOT_PATH = "equilibrium/time_slice/profiles_1d/psi"
TERMINAL_PATH = "equilibrium/time_slice/profiles_1d/psi_axis"


class _FakeGC:
    """Graph double: canned rows for reads, recorded writes for ``SET``."""

    def __init__(self, rows):
        self._rows = rows
        self.queries: list[tuple[str, dict]] = []
        self.writes: list[tuple[str, dict]] = []

    def query(self, cypher, **kw):
        self.queries.append((cypher, kw))
        if "SET" in cypher and "RETURN" not in cypher:
            self.writes.append((cypher, kw))
            return []
        if "sn.source_path IS NULL" in cypher:
            return []  # nothing unstamped: the baseline pass writes nothing
        if "n.id AS path" in cypher:
            # Re-stamp read: the resolved terminal the snapshot is written from.
            return [
                {
                    "sn_id": "poloidal_flux_axis",
                    "path": TERMINAL_PATH,
                    "unit": "W.m^-2",
                    "documentation": "Flux.",
                }
            ]
        return self._rows

    def close(self):
        pass


def _row(
    *,
    old_path: str = SNAPSHOT_PATH,
    new_path: str = TERMINAL_PATH,
    anchor_id: str | None = None,
    old_unit: str = "W.m^-2",
    new_unit: str = "W.m^-2",
    old_doc: str = "Flux.",
    new_doc: str = "Flux.",
    lineage: bool = False,
) -> dict:
    resolved = sr._resolved_source_context(new_path, new_unit, new_doc)
    return {
        "sn_id": "poloidal_flux_axis",
        "source_id": f"dd:{new_path if anchor_id is None else anchor_id}",
        "anchor_id": new_path if anchor_id is None else anchor_id,
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "old_unit": old_unit,
        "new_unit": new_unit,
        "old_doc": old_doc,
        "new_doc": new_doc,
        "old_path": old_path,
        "new_path": new_path,
        "old_raw_unit": resolved["raw_dd_context"]["unit"],
        "old_raw_doc": resolved["raw_dd_context"]["documentation"],
        "old_resolution_ids": resolved["dd_resolution_ids"],
        "old_converged_ids": resolved["dd_resolution_converged_ids"],
        "old_manifest_digest": resolved["dd_resolution_manifest_digest"],
        "old_resolution_marker": resolved["_dd_resolution_marker"],
        "lineage_rename_explains": lineage,
    }


def _steer_recorder(monkeypatch) -> list[str]:
    steered: list[str] = []

    def _record(**kw):
        steered.append(kw["target"])
        return MagicMock(blocked=None)

    monkeypatch.setattr("imas_codex.standard_names.edit.apply_edit", _record)
    return steered


def test_lineage_only_move_steers_no_docs_reset_and_counts_one_rename(monkeypatch):
    # The anchor already resolves to the terminal path; only the stored
    # snapshot lags behind the rename. That is exactly the shape that reset
    # accepted documentation on the startup refresh.
    graph = _FakeGC([_row(old_path=SNAPSHOT_PATH, new_path=TERMINAL_PATH, lineage=True)])
    steered = _steer_recorder(monkeypatch)

    summary = sr.refresh_drifted_sources(gc=graph)

    assert steered == []
    assert summary["detected"] == 1
    assert summary["steered"] == 0
    assert summary["renamed"] == 1
    assert summary["restamped"] == 1
    # The snapshot is re-stamped to the terminal path, so the next run is a
    # no-op rather than a second false drift.
    stamped = [kw["updates"][0]["path"] for _, kw in graph.writes if kw.get("updates")]
    assert stamped == [TERMINAL_PATH]


def test_lineage_only_move_delta_is_flagged_as_a_rename():
    out = sr.detect_source_drift(
        gc=_FakeGC([_row(old_path=SNAPSHOT_PATH, new_path=TERMINAL_PATH, lineage=True)])
    )

    [drifted] = out
    assert drifted["renamed"] is True
    assert drifted["requires_steering"] is False
    [delta] = drifted["deltas"]
    assert delta == {
        "field": "source_path",
        "old": SNAPSHOT_PATH,
        "new": TERMINAL_PATH,
        "rename": True,
    }


def test_lineage_move_plus_unit_change_still_steers(monkeypatch):
    graph = _FakeGC(
        [
            _row(
                old_path=SNAPSHOT_PATH,
                new_path=TERMINAL_PATH,
                old_unit="m^-2.s^-1",
                new_unit="W.m^-2",
                lineage=True,
            )
        ]
    )
    steered = _steer_recorder(monkeypatch)

    summary = sr.refresh_drifted_sources(gc=graph)

    assert steered == ["poloidal_flux_axis"]
    assert summary["steered"] == 1
    assert summary["renamed"] == 1


def test_path_change_with_no_lineage_still_steers(monkeypatch):
    graph = _FakeGC(
        [
            _row(
                old_path="equilibrium/time_slice/profiles_1d/psi",
                new_path="equilibrium/time_slice/profiles_1d/j_phi",
                anchor_id="equilibrium/time_slice/profiles_1d/psi",
                lineage=False,
            )
        ]
    )
    steered = _steer_recorder(monkeypatch)

    summary = sr.refresh_drifted_sources(gc=graph)

    assert steered == ["poloidal_flux_axis"]
    assert summary["steered"] == 1


def test_detect_query_asks_the_graph_whether_lineage_explains_the_move():
    # The flag must come from the graph, not from the row's shape: a double
    # that supplies the key would keep every test above green if the query
    # stopped asking.
    graph = _FakeGC([])

    sr.detect_source_drift(gc=graph)

    detect_queries = [c for c, _ in graph.queries if "n.id AS new_path" in c]
    assert len(detect_queries) == 1
    assert "lineage_rename_explains" in detect_queries[0]
    assert "RENAMED_TO*1..12" in detect_queries[0]
    assert "sn.source_path" in detect_queries[0]
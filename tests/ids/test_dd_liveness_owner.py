"""DD path liveness is decided by the search layer's version clause.

``_dd_version_clause`` in ``imas_codex.tools.graph_search`` is the one owner of
the "active at this DD version" rule, and ``dd_path_lifecycles`` renders its
filter through it rather than re-implementing the comparison in Python. The
owner requires an ``INTRODUCED_IN`` edge, so a path with no recorded
introduction is not part of the version. An unset DD version — ``None`` or a
blank string — disables the filter and leaves every requested path live, which
matters because the mapping pipeline passes ``state.dd_version or ""`` when a
run carries no version.
"""

from __future__ import annotations

from imas_codex.ids import graph_ops
from imas_codex.ids.graph_ops import (
    dd_path_lifecycles,
    dd_path_live_at,
)
from imas_codex.tools.graph_search import _dd_version_clause

DD_VERSION = "4.1.1"
CONDUCTOR = "tf/coil/conductor/current/data"
COIL = "tf/coil/current/data"


class _RecordingGraph:
    """A graph stub that records each statement and replays fixed rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.statements: list[str] = []

    def query(self, statement: str, **params: object) -> list[dict]:
        self.statements.append(statement)
        return list(self._rows)


def _rows(*, live: bool) -> list[dict]:
    return [
        {"id": COIL, "introduced": "3.22.0", "deprecated": None, "live": live},
        {
            "id": CONDUCTOR,
            "introduced": "3.22.0",
            "deprecated": "3.42.0",
            "live": live,
        },
    ]


def test_filter_is_rendered_through_dd_version_clause(monkeypatch):
    seen: list[tuple[str, object]] = []
    real = graph_ops._dd_version_clause

    def spy(alias="p", dd_version=None, params=None):
        seen.append((alias, dd_version))
        return real(alias, dd_version, params)

    monkeypatch.setattr(graph_ops, "_dd_version_clause", spy)
    gc = _RecordingGraph(_rows(live=True))

    dd_path_lifecycles(gc, [COIL, CONDUCTOR], DD_VERSION)

    assert seen == [("p", DD_VERSION)]
    statement = gc.statements[0]
    # The emitted filter is the owner's own fragment, not a re-derivation.
    assert _dd_version_clause("p", DD_VERSION, {}) in statement
    assert "INTRODUCED_IN" in statement
    assert "DEPRECATED_IN" in statement


def test_empty_dd_version_leaves_every_path_live_without_raising():
    # The predicate that a blank version must not reach a parser through.
    assert dd_path_live_at("3.22.0", "3.42.0", "") is True
    assert dd_path_live_at("3.22.0", "3.42.0", "   ") is True

    gc = _RecordingGraph(_rows(live=True))
    lifecycles = dd_path_lifecycles(gc, [COIL, CONDUCTOR], "")

    assert set(lifecycles) == {COIL, CONDUCTOR}
    assert all(life.live for life in lifecycles.values())
    # No version predicate is rendered, so the projection is unconditionally
    # true and every surviving row is live.
    assert "EXISTS" not in gc.statements[0]
    assert _dd_version_clause("p", "", {}) == ""


def test_missing_introduction_is_not_live():
    # The owner's semantics: without an INTRODUCED_IN edge the path does not
    # belong to the version, so liveness must not default to present.
    rows = [{"id": COIL, "introduced": None, "deprecated": None, "live": None}]
    gc = _RecordingGraph(rows)

    lifecycles = dd_path_lifecycles(gc, [COIL], DD_VERSION)

    assert lifecycles[COIL].live is False
    assert lifecycles[COIL].reason is not None

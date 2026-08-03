"""Read-only startup visibility for stale grammar compatibility projections.

Bare-name segment columns are deterministic projections of the canonical name
identifier. Startup audits that invariant and routes drift to the governed,
manifest-bound operator instead of performing an untracked graph mutation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imas_codex.standard_names.graph_ops import (
    _GRAMMAR_SEGMENT_COLUMNS,
    _parse_grammar,
    reconcile_grammar_segments,
)


def test_reconcile_grammar_segments_returns_count():
    """The reconcile reports how many names it realigned (contract shape)."""
    # Pure-shape guard that does not require a live graph: the module exposes
    # the reconcile and the segment-column authority it realigns against.
    assert "position" in _GRAMMAR_SEGMENT_COLUMNS
    assert callable(reconcile_grammar_segments)


def test_startup_compatibility_call_is_read_only_and_surfaces_governed_work():
    """Startup audits projection drift but cannot mutate the live catalog."""
    name = "electron_density"
    parsed = _parse_grammar(name)
    row = {"id": name, **dict.fromkeys(_GRAMMAR_SEGMENT_COLUMNS)}
    graph = MagicMock()
    graph.query.return_value = [row]
    graph_client = MagicMock()
    graph_client.return_value.__enter__.return_value = graph

    with patch("imas_codex.standard_names.graph_ops.GraphClient", graph_client):
        result = reconcile_grammar_segments()

    assert result == {
        "names_realigned": 0,
        "names_planned": 1,
        "governed_apply_required": True,
    }
    assert parsed["physical_base"] == "density"
    assert graph.query.call_count == 1
    assert "SET " not in graph.query.call_args.args[0]

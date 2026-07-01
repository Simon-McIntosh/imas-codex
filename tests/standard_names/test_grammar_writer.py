"""Plan 40 Phase 1 — writer behaviour tests for ``_write_grammar_decomposition``.

Mocked GraphClient — does not require a live Neo4j. Verifies:

- T-A1/A3: per-segment columns always populated when parser succeeds
- T-A6: re-writing with narrower grammar clears stale columns
- T-A7: parser error → segment columns cleared, no fallback flag written
  (non-compliance is recorded by ``validation_status`` at validate time)
- T-A3: open-vocab ``physical_base`` populates the column even when no
  GrammarToken corpus exists in the graph (token_version is ``None``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.graph_ops import _write_grammar_decomposition


def _mock_gc_with_token_version(version: str | None = "0.7.0") -> MagicMock:
    """Build a mock GraphClient where _resolve_grammar_token_version returns *version*."""
    gc = MagicMock()
    gc.query = MagicMock(return_value=[])
    return gc


def _column_setters(mock_gc: MagicMock) -> list[dict]:
    """Return the kwargs of every gc.query call that wrote per-segment columns."""
    out = []
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if "SET sn.physical_base" in cypher:
            out.append(c[1])
    return out


def test_columns_always_set_when_parser_succeeds() -> None:
    """`_write_grammar_decomposition` writes bare-name columns from parser output."""
    gc = _mock_gc_with_token_version("0.7.0")
    with patch(
        "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
        return_value="0.7.0",
    ):
        gaps = _write_grammar_decomposition(gc, ["electron_temperature"])

    assert gaps == []
    setters = _column_setters(gc)
    assert len(setters) == 1
    kwargs = setters[0]
    assert kwargs["sn_id"] == "electron_temperature"
    assert kwargs["physical_base"] == "temperature"
    assert kwargs["subject"] == "electron"
    # Unset segments coerced to None
    assert kwargs["component"] is None
    assert kwargs["coordinate"] is None


def test_columns_cleared_when_parser_narrowed() -> None:
    """Re-writing with a name that has no `component` writes None for that column."""
    gc = _mock_gc_with_token_version("0.7.0")
    with patch(
        "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
        return_value="0.7.0",
    ):
        # First call: name with `component=x`
        _write_grammar_decomposition(gc, ["x_magnetic_field_at_outboard_midplane"])
        # Second call: simpler name with no component
        _write_grammar_decomposition(gc, ["electron_temperature"])

    setters = _column_setters(gc)
    assert len(setters) == 2
    assert setters[0]["component"] == "x"
    assert setters[0]["physical_base"] == "magnetic_field"
    # Second SET clears component to None
    assert setters[1]["component"] is None
    assert setters[1]["physical_base"] == "temperature"


def test_parser_error_clears_columns_without_flag() -> None:
    """Parser exception → SET clears segment columns; no fallback flag written.

    A name the ISN parser rejects has its per-segment columns nulled and is
    left unparseable in the graph. Its non-compliance is recorded
    authoritatively by ``validation_status='quarantined'`` at validate time
    (the single-pipeline gate), never by a separate ``grammar_parse_fallback``
    flag.
    """
    gc = _mock_gc_with_token_version("0.7.0")

    def _boom(_name: str):
        raise ValueError("synthetic parser failure")

    with (
        patch(
            "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
            return_value="0.7.0",
        ),
        patch("imas_standard_names.grammar.parse_standard_name", side_effect=_boom),
    ):
        gaps = _write_grammar_decomposition(gc, ["definitely_not_a_real_name"])

    assert gaps == []
    # The reject path issues a SET that nulls the segment columns …
    clear_calls = [
        c for c in gc.query.call_args_list if "physical_base = null" in c[0][0]
    ]
    assert len(clear_calls) == 1
    assert clear_calls[0][1]["sn_id"] == "definitely_not_a_real_name"
    # … and never writes a grammar_parse_fallback flag.
    assert not any(
        "grammar_parse_fallback" in c[0][0] for c in gc.query.call_args_list
    )


def test_open_vocab_physical_base_populates_column_with_no_edge() -> None:
    """Even when no GrammarToken corpus exists, columns are still written
    from the parser output (open-vocab capture)."""
    gc = _mock_gc_with_token_version(None)
    with patch(
        "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
        return_value=None,
    ):
        _write_grammar_decomposition(gc, ["electron_temperature"])

    setters = _column_setters(gc)
    assert len(setters) == 1
    assert setters[0]["physical_base"] == "temperature"
    assert setters[0]["subject"] == "electron"
    # No edge-write Cypher executed (token_version is None → bail before edges).
    edge_calls = [
        c
        for c in gc.query.call_args_list
        if "MERGE (sn)-[r:HAS_SEGMENT]" in c[0][0]
        or "FOREACH (_ IN CASE WHEN t IS NOT NULL" in c[0][0]
    ]
    assert edge_calls == []


def test_idempotent_on_repeated_calls() -> None:
    """Calling twice on the same name yields identical column SETs."""
    gc = _mock_gc_with_token_version("0.7.0")
    with patch(
        "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
        return_value="0.7.0",
    ):
        _write_grammar_decomposition(gc, ["electron_temperature"])
        _write_grammar_decomposition(gc, ["electron_temperature"])

    setters = _column_setters(gc)
    assert len(setters) == 2
    assert setters[0] == setters[1]

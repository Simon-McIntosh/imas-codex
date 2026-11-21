"""Closed-vocabulary filters on the served signal search.

A quantity or vocabulary-role filter must reach both ranking streams and must
refuse an unknown member rather than return an empty report. The graph is a
recording double, so the assertions read the statements the search really built.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iter_codex.llm.search_tools import (
    VOCABULARY_QUANTITIES,
    VOCABULARY_ROLES,
    _search_signals,
)


def _graph() -> MagicMock:
    graph = MagicMock()
    graph.query = MagicMock(return_value=[])
    return graph


def _encoder() -> MagicMock:
    encoder = MagicMock()
    encoder.embed_texts = MagicMock(return_value=[[0.1] * 8])
    return encoder


def _statements(graph: MagicMock) -> list[str]:
    return [call[0][0] for call in graph.query.call_args_list]


def test_an_unknown_quantity_is_refused_with_the_member_list():
    graph = _graph()
    report = _search_signals(
        "pressure at the outlet",
        "demo",
        quantity="not_a_quantity",
        gc=graph,
        encoder=_encoder(),
    )
    assert "Unknown quantity" in report
    assert "pressure" in report
    assert graph.query.call_args_list == []


def test_an_unknown_vocabulary_role_is_refused_with_the_member_list():
    graph = _graph()
    report = _search_signals(
        "pressure at the outlet",
        "demo",
        vocabulary_role="measured",
        gc=graph,
        encoder=_encoder(),
    )
    assert "Unknown vocabulary_role" in report
    assert "setpoint" in report
    assert graph.query.call_args_list == []


def test_a_quantity_filter_reaches_both_ranking_streams():
    graph = _graph()
    _search_signals(
        "pressure at the outlet",
        "demo",
        quantity="pressure",
        gc=graph,
        encoder=_encoder(),
    )
    statements = _statements(graph)
    vector = next(s for s in statements if "facility_signal_desc_embedding" in s)
    keyword = next(s for s in statements if "facility_signal_text" in s)
    assert "s.vocabulary_quantity = $quantity" in vector
    assert "s.vocabulary_quantity = $quantity" in keyword


def test_a_vocabulary_role_filter_reaches_both_ranking_streams():
    graph = _graph()
    _search_signals(
        "water temperature",
        "demo",
        vocabulary_role="setpoint",
        gc=graph,
        encoder=_encoder(),
    )
    statements = _statements(graph)
    vector = next(s for s in statements if "facility_signal_desc_embedding" in s)
    keyword = next(s for s in statements if "facility_signal_text" in s)
    assert "s.vocabulary_role = $vocabulary_role" in vector
    assert "s.vocabulary_role = $vocabulary_role" in keyword


def test_a_quantity_and_a_role_combine_with_and():
    graph = _graph()
    _search_signals(
        "valve position",
        "demo",
        quantity="valve_position",
        vocabulary_role="measurement",
        gc=graph,
        encoder=_encoder(),
    )
    vector = next(
        s for s in _statements(graph) if "facility_signal_desc_embedding" in s
    )
    assert "s.vocabulary_quantity = $quantity" in vector
    assert "s.vocabulary_role = $vocabulary_role" in vector


def test_a_plant_listing_refuses_the_vocabulary_filters():
    with pytest.raises(ValueError, match="plant-code filters only"):
        _search_signals(
            None,
            "demo",
            quantity="pressure",
            plant_system="DEMO-SYSTEM",
            gc=_graph(),
        )


def test_the_refusal_lists_are_the_schema_vocabularies():
    assert len(VOCABULARY_QUANTITIES) == 16
    assert set(VOCABULARY_ROLES) == {
        "measurement",
        "setpoint",
        "state",
        "alarm",
        "command",
    }

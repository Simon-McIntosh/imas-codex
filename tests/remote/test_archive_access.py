"""Shell metadata accessor paging with invented category and signal names."""

from unittest.mock import Mock

import pytest

from iter_codex.remote.archive_access import (
    find_categories,
    find_signals,
    list_pulses,
)


@pytest.mark.parametrize(
    ("accessor", "argument", "parameter"),
    [
        (find_categories, "Demo magnet", "text"),
        (list_pulses, "demo:codac_uda:slow:DEMO:TRIAL", "category_id"),
        (find_signals, "DEMO-PLANT:TEMP", "name"),
    ],
)
def test_metadata_accessors_page_and_bind_literals(accessor, argument, parameter):
    graph = Mock()
    item = {"id": "demo:entry", "name": "DEMO:ENTRY"}
    graph.query.side_effect = [[{"total": 5}], [item]]
    page = accessor(graph, argument, facility="demo", limit=2, offset=2)
    assert page == {"items": [item], "total": 5, "next_offset": 4}
    count, listing = graph.query.call_args_list
    assert count.kwargs == {"facility": "demo", parameter: argument}
    assert listing.kwargs == {
        "facility": "demo",
        parameter: argument,
        "limit": 2,
        "offset": 2,
    }
    assert argument not in count.args[0]
    assert argument not in listing.args[0]
    assert "SKIP $offset LIMIT $limit" in listing.args[0]


@pytest.mark.parametrize("text", ["not present", "*", "' OR true //"])
def test_unmatched_category_is_explicitly_empty(text):
    graph = Mock()
    graph.query.side_effect = [[{"total": 0}], []]
    assert find_categories(graph, text) == {
        "items": [],
        "total": 0,
        "next_offset": None,
    }
    assert "CONTAINS toLower($text)" in graph.query.call_args.args[0]
    assert graph.query.call_args.kwargs["text"] == text


def test_pulse_listing_preserves_open_end_and_description():
    graph = Mock()
    pulse = {
        "id": "demo:codac_uda:slow:DEMO:TRIAL/nested/alpha",
        "name": "DEMO:TRIAL/nested/alpha",
        "archive": "demo:codac_uda:slow",
        "start_ns": 100,
        "end_ns": None,
        "description": "Demonstration open pulse",
    }
    graph.query.side_effect = [[{"total": 1}], [pulse]]
    page = list_pulses(graph, "demo:codac_uda:slow:DEMO:TRIAL")
    assert page == {"items": [pulse], "total": 1, "next_offset": None}
    statement = graph.query.call_args.args[0]
    assert "IN_PULSE_CATEGORY" in statement
    assert "p.end_ns AS end_ns" in statement
    assert "p.description AS description" in statement


@pytest.mark.parametrize("accessor", [find_categories, find_signals, list_pulses])
@pytest.mark.parametrize(
    "options",
    [
        {"limit": 0},
        {"limit": 101},
        {"limit": True},
        {"offset": -1},
        {"offset": 0.5},
    ],
)
def test_bad_page_never_queries_graph(accessor, options):
    graph = Mock()
    with pytest.raises(ValueError, match="page"):
        accessor(graph, "DEMO", **options)
    graph.query.assert_not_called()


@pytest.mark.parametrize("accessor", [find_categories, find_signals, list_pulses])
@pytest.mark.parametrize("argument", ["", "  ", None])
def test_empty_lookup_never_becomes_unbounded_listing(accessor, argument):
    graph = Mock()
    with pytest.raises(ValueError):
        accessor(graph, argument)
    graph.query.assert_not_called()

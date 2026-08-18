"""Runtime coverage for DD rows whose unit declaration is absent."""

from unittest.mock import MagicMock, patch

from imas_codex.standard_names.workers import (
    _DD_CONTEXT_QUERY,
    _enrich_batch_items,
)

_NUMERIC_PATH = "equilibrium/time_slice/profiles_1d/q"


def _enrich_from_exact_facts(row: dict) -> dict:
    item = {"path": _NUMERIC_PATH, "dd_version": row["dd_version"]}
    graph = MagicMock()
    graph.query.side_effect = lambda query, **_params: (
        [row] if query == _DD_CONTEXT_QUERY else []
    )
    with patch("imas_codex.graph.client.GraphClient") as graph_client:
        graph_client.return_value.__enter__.return_value = graph
        graph_client.return_value.__exit__.return_value = False
        _enrich_batch_items([item])
    return item


def test_numeric_row_without_unit_stays_unresolved() -> None:
    item = _enrich_from_exact_facts(
        {
            "dd_version": "4.1.1",
            "raw_unit": None,
            "unit_relations": [],
            "unit_from_rel": None,
            "data_type": "FLT_1D",
        }
    )

    assert item["path"] == _NUMERIC_PATH
    assert item["dd_version"] == "4.1.1"
    assert item["raw_dd_context"]["unit"] is None
    assert item["raw_dd_context"]["data_type"] == "FLT_1D"
    assert item.get("unit") is None


def test_exact_graph_query_carries_all_unit_applicability_facts() -> None:
    assert "current_dd_version.id AS dd_version" in _DD_CONTEXT_QUERY
    assert "n.unit AS raw_unit" in _DD_CONTEXT_QUERY
    assert "unit_relations" in _DD_CONTEXT_QUERY
    assert "n.data_type AS data_type" in _DD_CONTEXT_QUERY


def test_declared_unit_relationship_remains_authoritative() -> None:
    item = _enrich_from_exact_facts(
        {
            "dd_version": "4.1.1",
            "raw_unit": None,
            "unit_relations": ["Pa"],
            "unit_from_rel": "Pa",
            "data_type": "FLT_1D",
        }
    )

    assert item["raw_dd_context"]["unit"] == "Pa"
    assert item["unit"] == "Pa"

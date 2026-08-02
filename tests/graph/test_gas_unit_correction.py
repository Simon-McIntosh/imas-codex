"""Exact-scope contracts for correcting the SPI gas-flow DD units."""

from __future__ import annotations

import fnmatch

import pytest

from imas_codex.graph.dd_graph_ops import reconcile_dd_unit_corrections
from imas_codex.graph.dd_lifecycle import dd_path_index
from imas_codex.units import resolve_dd_unit
from imas_codex.units.dd_unit_exceptions import load_exceptions

GAS_FLOW_PATHS = (
    "spi/injector/fragmentation_gas/flow_rate",
    "spi/injector/propellant_gas/flow_rate",
)


class _ScopeProbe:
    """Return one out-of-scope row to exercise the defensive Python fence."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def query(self, cypher: str, **params):
        self.calls.append((cypher, params))
        if "RETURN n.id AS path" in cypher:
            return [
                {"path": GAS_FLOW_PATHS[0], "unit": "s^-1"},
                {"path": GAS_FLOW_PATHS[1], "unit": "s^-1"},
                {"path": "ece/channel/position/psi", "unit": "W"},
            ]
        return []


def _gas_flow_entry() -> dict:
    entries = [
        entry
        for entry in load_exceptions()["dd_unit_bugs"]
        if entry["path"] == "spi/injector/*_gas/flow_rate"
    ]
    assert len(entries) == 1
    return entries[0]


def test_registry_rewrites_the_two_gas_flow_paths() -> None:
    entry = _gas_flow_entry()

    assert entry["dd_unit"] == "s^-1"
    assert entry["correct_unit"] == "Pa.m^3.s^-1"
    assert entry["correct_in_graph"] is True
    assert [resolve_dd_unit(path, "s^-1") for path in GAS_FLOW_PATHS] == [
        "Pa.m^3.s^-1",
        "Pa.m^3.s^-1",
    ]


def test_registry_glob_expands_to_exactly_the_two_current_dd_paths() -> None:
    paths, _ = dd_path_index()
    matched = sorted(
        path for path in paths if fnmatch.fnmatchcase(path, _gas_flow_entry()["path"])
    )

    assert matched == sorted(GAS_FLOW_PATHS)


def test_exact_scope_excludes_an_unrelated_registered_correction() -> None:
    gc = _ScopeProbe()

    result = reconcile_dd_unit_corrections(gc, path_ids=GAS_FLOW_PATHS)

    assert result == {"checked": 2, "corrected": 2}
    read_query, read_params = gc.calls[0]
    assert "n.id IN $path_ids" in read_query
    assert set(read_params["path_ids"]) == set(GAS_FLOW_PATHS)

    write_query, write_params = gc.calls[1]
    assert {item["path"] for item in write_params["items"]} == set(GAS_FLOW_PATHS)
    assert "ece/channel/position/psi" not in {
        item["path"] for item in write_params["items"]
    }
    assert "OPTIONAL MATCH (n)-[r:HAS_UNIT]->(:Unit)" in write_query
    assert "DELETE r" in write_query
    assert "SET n.unit = item.expected" in write_query
    assert "MERGE (n)-[:HAS_UNIT]->(u)" in write_query


def test_empty_exact_scope_is_a_noop() -> None:
    gc = _ScopeProbe()

    result = reconcile_dd_unit_corrections(gc, path_ids=[])

    assert result == {"checked": 0, "corrected": 0}
    assert gc.calls == []


def test_one_string_is_rejected_instead_of_becoming_a_character_scope() -> None:
    gc = _ScopeProbe()

    with pytest.raises(TypeError, match="collection of exact paths"):
        reconcile_dd_unit_corrections(gc, path_ids=GAS_FLOW_PATHS[0])

    assert gc.calls == []


def test_unrelated_paths_and_correct_units_pass_through() -> None:
    assert resolve_dd_unit(GAS_FLOW_PATHS[0], "Pa.m^3.s^-1") == "Pa.m^3.s^-1"
    assert resolve_dd_unit("gas_injection/valve/flow_rate", "s^-1") == "s^-1"

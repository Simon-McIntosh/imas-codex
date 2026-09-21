"""Nearby-name comparator allocation across compose batches."""

import ast
import inspect

import pytest

from imas_codex.standard_names import workers
from imas_codex.standard_names.workers import _collect_nearby_name_comparators


@pytest.mark.parametrize("item_count", [31, 50])
def test_every_item_receives_a_comparator_before_budget_refill(
    item_count: int,
) -> None:
    items = [
        {
            "path": f"equilibrium/time_slice/item_{index}",
            "description": f"distinct description {index}",
        }
        for index in range(item_count)
    ]

    def search(description: str, *, k: int) -> list[dict]:
        index = description.rsplit(" ", 1)[-1]
        results = [
            {"id": f"item_{index}_primary"},
            {"id": "shared_comparator"},
            *({"id": f"item_{index}_extra_{offset}"} for offset in range(3)),
        ]
        return results[:k]

    nearby = _collect_nearby_name_comparators(items, search=search)
    nearby_ids = {result["id"] for result in nearby}
    expected_primary_ids = {f"item_{index}_primary" for index in range(item_count)}

    assert expected_primary_ids <= nearby_ids
    assert len(nearby_ids) == max(30, item_count)
    assert len(nearby_ids) == len(nearby)


def test_item_without_search_results_uses_no_comparator_share() -> None:
    items = [{"description": f"item {index}"} for index in range(25)]

    def search(description: str, *, k: int) -> list[dict]:
        index = int(description.rsplit(" ", 1)[-1])
        if index == 12:
            return []
        return [{"id": f"item_{index}_{offset}"} for offset in range(k)]

    nearby = _collect_nearby_name_comparators(items, search=search)
    nearby_ids = [result["id"] for result in nearby]

    assert all(f"item_{index}_0" in nearby_ids for index in range(25) if index != 12)
    assert not any(result_id.startswith("item_12_") for result_id in nearby_ids)
    assert len(nearby_ids) == 30
    assert len(nearby_ids) == len(set(nearby_ids))


def test_compose_paths_offload_comparator_searches() -> None:
    tree = ast.parse(inspect.getsource(workers))
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    direct_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_collect_nearby_name_comparators"
    ]
    offloaded_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
        and node.func.attr == "to_thread"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_collect_nearby_name_comparators"
    ]

    assert direct_calls == []
    assert len(offloaded_calls) == 2
    assert all(isinstance(parents[id(call)], ast.Await) for call in offloaded_calls)

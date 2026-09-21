"""Nearby-name comparator allocation across compose batches."""

from imas_codex.standard_names.workers import _collect_nearby_name_comparators


def test_every_item_receives_a_comparator_before_budget_refill() -> None:
    items = [
        {
            "path": f"equilibrium/time_slice/item_{index}",
            "description": f"distinct description {index}",
        }
        for index in range(25)
    ]
    searched: list[str] = []

    def search(description: str, *, k: int) -> list[dict]:
        searched.append(description)
        index = description.rsplit(" ", 1)[-1]
        results = [
            {"id": f"item_{index}_primary"},
            {"id": "shared_comparator"},
            *({"id": f"item_{index}_extra_{offset}"} for offset in range(3)),
        ]
        return results[:k]

    nearby = _collect_nearby_name_comparators(items, search=search)
    nearby_ids = [result["id"] for result in nearby]

    assert searched == [item["description"] for item in items]
    assert all(f"item_{index}_primary" in nearby_ids for index in range(25))
    assert len(nearby_ids) == 30
    assert len(nearby_ids) == len(set(nearby_ids))


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

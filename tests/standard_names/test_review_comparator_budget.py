"""Unit-anchored comparator budget: per-candidate share and overall bound."""

from __future__ import annotations

from imas_codex.standard_names.review.enrichment import build_neighborhood_context


def _catalog_name(
    name_id: str,
    *,
    unit: str,
    physical_base: str,
) -> dict:
    return {
        "id": name_id,
        "description": name_id.replace("_", " "),
        "kind": "scalar",
        "unit": unit,
        "physical_base": physical_base,
        "name_stage": "accepted",
        "review_tier": "good",
    }


def _candidate(name_id: str, *, unit: str, physical_base: str) -> dict:
    return {
        "id": name_id,
        "description": name_id.replace("_", " "),
        "kind": "scalar",
        "unit": unit,
        "physical_base": physical_base,
    }


def _unit_anchored_counts(comparators: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for comparator in comparators:
        if comparator["comparison_basis"] != "same_unit_different_physical_base":
            continue
        unit = comparator["unit"]
        counts[unit] = counts.get(unit, 0) + 1
    return counts


def test_every_candidate_receives_a_unit_anchored_comparator(monkeypatch) -> None:
    """One candidate on a common unit must not absorb the whole budget.

    Three candidates on distinct units against a catalog dense on the first
    unit only.  A first-come budget spends every slot on unit ``m`` and leaves
    ``s`` and ``eV`` with none, though each holds three eligible comparators;
    the per-candidate share hands each candidate a slot before the remainder is
    spent.
    """
    monkeypatch.setattr(
        "imas_codex.standard_names.search.search_standard_names_vector",
        lambda _query, *, k: [],
    )
    candidates = [
        _candidate("candidate_m", unit="m", physical_base="alpha"),
        _candidate("candidate_s", unit="s", physical_base="beta"),
        _candidate("candidate_eV", unit="eV", physical_base="gamma"),
    ]
    catalog = [
        _catalog_name(f"common_m_{index:02d}", unit="m", physical_base="m_alt")
        for index in range(40)
    ]
    catalog += [
        _catalog_name(f"catalog_s_{index}", unit="s", physical_base=f"s_alt_{index}")
        for index in range(3)
    ]
    catalog += [
        _catalog_name(f"catalog_eV_{index}", unit="eV", physical_base=f"eV_alt_{index}")
        for index in range(3)
    ]

    comparators = build_neighborhood_context(
        {"names": candidates, "cluster": None},
        [*candidates, *catalog],
        k=10,
    )

    counts = _unit_anchored_counts(comparators)
    # unit_cap = max(10, 3) = 10; allowance = 10 // 3 = 3 per candidate, then
    # the remainder returns to the first candidate.
    assert counts == {"m": 4, "s": 3, "eV": 3}, counts
    assert all(count >= 1 for count in counts.values())


def test_returned_neighbourhood_respects_the_stated_bound(monkeypatch) -> None:
    """The returned length is capped at unit_cap + semantic_cap."""

    def search(query: str, *, k: int) -> list[dict]:
        tag = query.rsplit(" ", 1)[-1]
        return [
            _catalog_name(
                f"semantic_{tag}_{index}", unit="1", physical_base=f"peer_{tag}"
            )
            for index in range(5)
        ]

    monkeypatch.setattr(
        "imas_codex.standard_names.search.search_standard_names_vector", search
    )
    batch_size = 15
    candidates = [
        _candidate(f"candidate_{index}", unit=f"unit_{index}", physical_base="cand")
        for index in range(batch_size)
    ]
    catalog = [*candidates]
    for index in range(batch_size):
        catalog.extend(
            _catalog_name(
                f"existing_{index}_{slot}",
                unit=f"unit_{index}",
                physical_base=f"existing_base_{slot}",
            )
            for slot in range(5)
        )

    comparators = build_neighborhood_context(
        {"names": candidates, "cluster": None},
        catalog,
        k=10,
    )

    # unit_cap = max(10, 15) = 15 and semantic_cap = min(15, 60) = 15, with
    # disjoint ids on each channel, so the stated bound is saturated.
    assert len(comparators) == 30


def test_large_catalog_saturation_lands_on_the_bound(monkeypatch) -> None:
    """A catalog that saturates both channels returns exactly the stated bound.

    Sixty-one candidates, each on its own unit, against a catalog carrying two
    hundred accepted cross-base names per unit.  The unit-anchored channel
    fills its share of 61 from the catalog and the semantic channel its share
    of 60 from a wide result set, over disjoint identity sets, so the returned
    length sits at ``unit_cap + semantic_cap`` — 121 at ``k=10`` over a batch
    above the shared helper's 60-item ceiling.
    """

    def search(query: str, *, k: int) -> list[dict]:
        tag = query.rsplit(" ", 1)[-1]
        return [
            _catalog_name(
                f"semantic_{tag}_{slot}", unit="1", physical_base=f"peer_{tag}"
            )
            for slot in range(5)
        ]

    monkeypatch.setattr(
        "imas_codex.standard_names.search.search_standard_names_vector", search
    )
    batch_size = 61
    candidates = [
        _candidate(f"candidate_{index}", unit=f"unit_{index}", physical_base="cand")
        for index in range(batch_size)
    ]
    catalog = [*candidates]
    for index in range(batch_size):
        catalog.extend(
            _catalog_name(
                f"existing_{index}_{slot}",
                unit=f"unit_{index}",
                physical_base=f"existing_base_{slot}",
            )
            for slot in range(200)
        )

    comparators = build_neighborhood_context(
        {"names": candidates, "cluster": None},
        catalog,
        k=10,
    )

    unit_cap = max(10, batch_size)
    semantic_cap = min(unit_cap, 60)
    assert len(comparators) == unit_cap + semantic_cap == 121


def test_over_supplied_channel_is_closed_at_the_terminal_bound(monkeypatch) -> None:
    """The terminal cap holds when a channel delivers past its own share.

    Every catalog-driven case lands on the bound by construction: the
    unit-anchored channel is capped at ``unit_cap``, the semantic channel at
    ``semantic_cap``, and their sum is the terminal bound, so neither channel
    can carry the pair past it.  The terminal cap is the guard for the case
    those two caps do not cover — a channel delivering above its share — so the
    semantic channel is supplied at 40 comparators here, above its share of 15,
    and the returned length must still be ``unit_cap + semantic_cap``.  Without
    the terminal cap the excess is returned.
    """
    over_supplied = [
        _catalog_name(f"semantic_{index}", unit="1", physical_base=f"peer_{index}")
        for index in range(40)
    ]
    monkeypatch.setattr(
        "imas_codex.standard_names.workers._collect_nearby_name_comparators",
        lambda _items, **_kwargs: list(over_supplied),
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.search.search_standard_names_vector",
        lambda _query, *, k: [],
    )
    candidates = [
        _candidate(f"candidate_{index}", unit=f"unit_{index}", physical_base="cand")
        for index in range(15)
    ]
    catalog = [*candidates]
    for index in range(15):
        catalog.extend(
            _catalog_name(
                f"existing_{index}_{slot}",
                unit=f"unit_{index}",
                physical_base=f"existing_base_{slot}",
            )
            for slot in range(5)
        )

    comparators = build_neighborhood_context(
        {"names": candidates, "cluster": None},
        catalog,
        k=10,
    )

    # unit_cap = max(10, 15) = 15, semantic_cap = min(15, 60) = 15.
    unit_anchored = sum(
        1
        for comparator in comparators
        if comparator["comparison_basis"] == "same_unit_different_physical_base"
    )
    assert len(comparators) == 30
    assert unit_anchored == 15

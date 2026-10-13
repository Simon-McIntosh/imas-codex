"""Contract for the declared physics-domain selector and its pure callers."""

from itertools import permutations
from unittest.mock import patch

import pytest

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.standard_names import domain_priority, domain_ranking
from imas_codex.standard_names.consolidation import _resolve_physics_domains


def test_declared_order_covers_vocabulary():
    assert set(domain_ranking.FALLBACK_RANK_TABLE) == {d.value for d in PhysicsDomain}
    assert len(domain_ranking.FALLBACK_RANK_TABLE) == len(PhysicsDomain)
    assert domain_ranking.FALLBACK_RANK_TABLE[-1] == "general"


def test_permutations_and_duplicates_choose_same_domain():
    domains = ("general", "transport", "equilibrium")
    for ordering in permutations(domains):
        assert domain_ranking.select_primary_domain(ordering) == "equilibrium"
        assert (
            domain_ranking.select_primary_domain((*ordering, *ordering))
            == "equilibrium"
        )


def test_empty_input_returns_none():
    assert domain_ranking.select_primary_domain([]) is None


def test_unknown_domain_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown_domain"):
        domain_ranking.select_primary_domain(["transport", "unknown_domain"])


def test_selection_never_opens_graph():
    with patch(
        "imas_codex.graph.client.GraphClient",
        side_effect=AssertionError("graph opened"),
    ):
        assert (
            domain_ranking.select_primary_domain(["general", "transport"])
            == "transport"
        )


def test_pure_callers_use_selector(monkeypatch):
    calls = []

    def sentinel(domains):
        calls.append(set(domains))
        return "sentinel"

    monkeypatch.setattr(domain_ranking, "select_primary_domain", sentinel)
    assert domain_priority.pick_primary_domain(["transport", "general"]) == "sentinel"
    assert domain_priority.domain_key(["transport", "general"]) == "sentinel"
    assert domain_priority.domain_key("transport") == "sentinel"
    assert _resolve_physics_domains(["general", "transport"]) == (
        "sentinel",
        ["general", "transport"],
    )
    assert calls == [
        {"transport", "general"},
        {"transport", "general"},
        {"transport"},
        {"transport", "general"},
    ]


def test_pure_callers_ignore_arrival_order():
    domains = ["general", "transport"]
    assert domain_priority.pick_primary_domain(domains) == "transport"
    assert domain_priority.domain_key(domains) == "transport"
    assert _resolve_physics_domains(domains) == ("transport", domains)

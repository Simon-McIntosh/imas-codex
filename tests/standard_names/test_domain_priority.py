"""Tests for domain coercion used by catalog and review callers."""

import pytest

from imas_codex.standard_names import domain_priority


def test_pick_primary_uses_declared_order():
    assert domain_priority.pick_primary_domain(["general", "transport"]) == "transport"
    assert domain_priority.pick_primary_domain(["transport", "general"]) == "transport"
    assert domain_priority.pick_primary_domain([]) is None


def test_pick_primary_refuses_unknown():
    with pytest.raises(ValueError, match="unknown_domain"):
        domain_priority.pick_primary_domain(["unknown_domain", "transport"])


def test_domain_key_none_returns_fallback():
    assert domain_priority.domain_key(None) == "unknown"
    assert domain_priority.domain_key(None, fallback="x") == "x"


def test_domain_key_empty_string_returns_fallback():
    assert domain_priority.domain_key("") == "unknown"
    assert domain_priority.domain_key("   ") == "unknown"


def test_domain_key_string_returns_trimmed():
    assert domain_priority.domain_key("equilibrium") == "equilibrium"
    assert domain_priority.domain_key("  transport  ") == "transport"


def test_domain_key_list_picks_declared_priority():
    assert domain_priority.domain_key(["general", "transport"]) == "transport"
    assert domain_priority.domain_key(["transport", "general"]) == "transport"


def test_domain_key_unknown_is_refused():
    with pytest.raises(ValueError, match="unknown_domain"):
        domain_priority.domain_key(["unknown_domain", "transport"])


def test_domain_key_list_with_blanks_returns_fallback():
    assert domain_priority.domain_key(["", None, "  "]) == "unknown"


def test_domain_key_unexpected_type_returns_fallback():
    assert domain_priority.domain_key(42) == "unknown"
    assert domain_priority.domain_key({"x": 1}) == "unknown"


def test_domain_list_none_empty():
    assert domain_priority.domain_list(None) == []
    assert domain_priority.domain_list("") == []
    assert domain_priority.domain_list("   ") == []


def test_domain_list_string_promoted_to_list():
    assert domain_priority.domain_list("transport") == ["transport"]
    assert domain_priority.domain_list("  equilibrium  ") == ["equilibrium"]


def test_domain_list_filters_blanks():
    assert domain_priority.domain_list(["a", "", "b", None, " ", "c"]) == [
        "a",
        "b",
        "c",
    ]

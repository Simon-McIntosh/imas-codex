"""The physics-domain guidance file must track the ISN vocabulary it is keyed by.

The guidance file (``imas_codex/definitions/physics/domains.yaml``) gives each
physics domain and each category a scope, and each domain the quantity kinds it
covers and the exclusions that separate it from a named neighbour. It is keyed
by ISN's values, so it must not fall behind that vocabulary: a domain or a
category added upstream without an entry here would be classified with no
guidance at all, and nothing would say so. These tests make that a failure.
"""

from __future__ import annotations

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.definitions.physics import (
    load_domain_guidance,
    load_isn_physics_domains,
)
from imas_codex.llm.prompt_loader import _provide_physics_domains


def test_domain_keys_are_exactly_the_enum_values() -> None:
    """Every PhysicsDomain value has an entry, and no other key exists.

    A domain added to the enum upstream without guidance added here fails this,
    which is the point: the file must not fall behind the vocabulary.
    """
    guidance = load_domain_guidance()
    assert set(guidance["domains"]) == {member.value for member in PhysicsDomain}


def test_category_keys_are_exactly_the_isn_categories() -> None:
    """Every category the ISN vocabulary uses has a scope, and no other key exists.

    Categories are read from ISN rather than frozen here, so a category added
    upstream fails this until the file carries its scope.
    """
    isn_categories = {
        meta["category"]
        for meta in load_isn_physics_domains().values()
        if meta["category"]
    }
    assert isn_categories, (
        "ISN vocabulary read empty — the check cannot see its subject"
    )
    guidance = load_domain_guidance()
    assert set(guidance["categories"]) == isn_categories


def test_every_domain_carries_scope_includes_and_excludes() -> None:
    """Each domain entry is complete: a scope, quantity kinds, and exclusions."""
    for name, entry in load_domain_guidance()["domains"].items():
        assert entry.get("scope", "").strip(), f"{name}: empty scope"
        assert entry.get("includes"), f"{name}: no includes"
        assert entry.get("excludes"), f"{name}: no excludes"


def test_every_excluded_neighbour_is_a_domain_value() -> None:
    """An exclusion names the neighbour domain that owns the quantity instead.

    A neighbour that is not a PhysicsDomain value points a reader at a domain
    that does not exist, so the rule cannot be followed.
    """
    values = {member.value for member in PhysicsDomain}
    for name, entry in load_domain_guidance()["domains"].items():
        for exclusion in entry.get("excludes", []):
            assert exclusion.get("neighbour") in values, (
                f"{name} excludes {exclusion.get('neighbour')!r}, not a PhysicsDomain value"
            )
            assert exclusion.get("quantity", "").strip(), (
                f"{name}: exclusion without a quantity"
            )
            assert exclusion.get("rule", "").strip(), (
                f"{name}: exclusion without a rule"
            )


def test_provider_returns_the_guidance_per_domain_and_category() -> None:
    """The prompt provider serves the guidance, keyed by value and by category."""
    provided = _provide_physics_domains()
    domains = provided["physics_domains"]
    by_value = {d["value"]: d for d in domains}
    assert set(by_value) == {member.value for member in PhysicsDomain}
    for value, entry in by_value.items():
        assert entry["scope"].strip(), f"{value}: provider dropped the scope"
        assert entry["includes"], f"{value}: provider dropped includes"
        assert entry["excludes"], f"{value}: provider dropped excludes"
        assert entry["description"], f"{value}: provider dropped the ISN description"
        assert entry["category"], f"{value}: provider dropped the ISN category"

    categories = provided["physics_domain_categories"]
    isn_categories = {
        meta["category"]
        for meta in load_isn_physics_domains().values()
        if meta["category"]
    }
    assert {c["value"] for c in categories} == isn_categories
    for category in categories:
        assert category["scope"].strip(), f"{category['value']}: empty category scope"

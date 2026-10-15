"""The physics-domain guidance file must track the ISN vocabulary it is keyed by.

The guidance file (``imas_codex/definitions/physics/domains.yaml``) gives each
physics domain and each category a scope, and each domain the quantity kinds it
covers and the exclusions that separate it from a named neighbour. It is keyed
by ISN's values, so it must not fall behind that vocabulary: a domain or a
category added upstream without an entry here would be classified with no
guidance at all, and nothing would say so. These tests make that a failure.

A domain is judged on the physics a path describes, so guidance must not claim
an IDS hierarchy as a reason to take a domain. A path filed under a container
takes the domain of the quantity it names, whatever the container is called.
"""

from __future__ import annotations

import json
import re

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.definitions.physics import (
    IDS_DOMAINS_FILE,
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


def test_each_excluded_quantity_and_neighbour_appears_once_per_domain() -> None:
    """One rule must carry all guidance for a quantity and its neighbour."""
    for name, entry in load_domain_guidance()["domains"].items():
        seen: set[tuple[str, str]] = set()
        for exclusion in entry.get("excludes", []):
            pair = (exclusion["quantity"], exclusion["neighbour"])
            assert pair not in seen, f"{name}: duplicate exclusion {pair!r}"
            seen.add(pair)


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


#: An IDS named as a path hierarchy in a guidance clause, as in ``summary/*``.
_PATH_PATTERN = re.compile(r"\b([a-z][a-z0-9_]*)/\*")

#: A clause that makes a container — rather than the quantity it carries — decide.
_CONTAINER_DECIDES = re.compile(
    r"\b(?:container|wrapper|interface\s+data\s+structure|ids)\b[^.]{0,80}?"
    r"\b(?:decides?|determines?|governs?|sets?)\b[^.]{0,60}?\bdomain\b",
    re.IGNORECASE | re.DOTALL,
)

_DATASET_DOMAIN = "data_management"


def _ids_domain_mapping() -> dict[str, str | None]:
    """The IDS-to-domain map, so a path-pattern claim can be checked against it."""
    payload = json.loads(IDS_DOMAINS_FILE.read_text(encoding="utf-8"))
    return payload["ids_domain_mappings"]


def _guidance_clauses(entry: dict) -> list[tuple[str, str]]:
    """Every prose clause of one domain or category entry, with its location."""
    clauses = [("scope", entry.get("scope", ""))]
    for i, include in enumerate(entry.get("includes") or []):
        clauses.append((f"includes[{i}]", str(include)))
    for i, exclusion in enumerate(entry.get("excludes") or []):
        for key in ("quantity", "rule"):
            clauses.append((f"excludes[{i}].{key}", exclusion.get(key, "")))
    return clauses


def test_data_management_claims_no_ids_it_does_not_own() -> None:
    """The dataset domain names as datasets only the IDSs the map gives it.

    A quantity filed under a physics IDS keeps the domain of the physics it
    names, so claiming that IDS here would hand every path beneath it to the
    dataset's domain by container. ``summary`` holds physical quantities, so a
    ``summary/*`` claim on this domain is exactly the defect this rejects; the
    IDS map is what settles the question.

    The map does not settle it the other way: a physics domain may legitimately
    name an IDS whose map entry is another domain, because there the claim is
    about the physics of those paths (a spectroscopy domain reading ``mse``).
    This test therefore binds only the dataset domain, whose subject is the
    dataset and never the physics it records.
    """
    includes = load_domain_guidance()["domains"][_DATASET_DOMAIN]["includes"]
    claimed = {
        ids_name: clause
        for clause in map(str, includes)
        for ids_name in _PATH_PATTERN.findall(clause)
    }
    assert claimed, "no IDS path pattern found — the check cannot see its subject"

    mapping = _ids_domain_mapping()
    for ids_name, clause in claimed.items():
        assert mapping.get(ids_name) == _DATASET_DOMAIN, (
            f"{_DATASET_DOMAIN} claims `{ids_name}/*` by path pattern, but the IDS "
            f"map places {ids_name} in {mapping.get(ids_name)!r}: {clause!r}"
        )


def test_no_rule_lets_a_container_decide_the_domain() -> None:
    """A quantity takes the domain of the physics it names, not of its container.

    Where a clause lets the container decide, every path filed beneath it takes
    the container's domain whatever it measures — an elongation under a summary
    container reading as a data product rather than as equilibrium physics. No
    clause of any domain or category may say so.
    """
    guidance = load_domain_guidance()
    offenders = [
        (name, location, match.group(0))
        for section in ("categories", "domains")
        for name, entry in guidance[section].items()
        for location, clause in _guidance_clauses(entry)
        if (match := _CONTAINER_DECIDES.search(clause or ""))
    ]
    assert not offenders, (
        "a container is made to decide the domain rather than the quantity it "
        f"carries: {offenders}"
    )

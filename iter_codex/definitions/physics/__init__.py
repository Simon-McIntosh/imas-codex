"""Physics domain definitions for IDS categorization.

``domains.yaml`` holds classification guidance for the physics domains: a scope
for each domain and each category, the quantity kinds a domain covers, and the
exclusions that separate a domain from a named neighbour. It is not a LinkML
schema and it repeats none of the ISN vocabulary — the canonical domain values,
descriptions and categories live in imas-standard-names.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Classification guidance keyed by the ISN physics domain and category values.
DOMAINS_SCHEMA = Path(__file__).parent / "domains.yaml"

# Version-independent IDS to physics domain mappings (covers all DD versions)
IDS_DOMAINS_FILE = Path(__file__).parent / "ids_domains.json"


@lru_cache(maxsize=1)
def _read_domains_file() -> dict[str, Any]:
    """The parsed ``domains.yaml``, or an empty mapping when unreadable."""
    try:
        data = yaml.safe_load(DOMAINS_SCHEMA.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_domain_guidance() -> dict[str, Any]:
    """Load the physics-domain classification guidance.

    Returns the ``categories`` and ``domains`` mappings from ``domains.yaml``.
    Each category entry holds a ``scope`` describing the first-level choice it
    answers; each domain entry holds a ``scope``, an ``includes`` list of
    quantity kinds, and an ``excludes`` list of ``{quantity, neighbour, rule}``
    entries naming the neighbour domain that owns the quantity instead.

    Returns empty mappings when the file is absent or unreadable, so a caller
    degrades to bare domain values rather than failing.
    """
    data = _read_domains_file()
    return {
        "categories": data.get("categories") or {},
        "domains": data.get("domains") or {},
    }


def _declared_precedence() -> list[str]:
    """The domains of ``domains.yaml``'s ``precedence`` block, band order kept."""
    order: list[str] = []
    for band in _read_domains_file().get("precedence") or []:
        order.extend(band.get("domains") or [])
    return order


def _refuse_invalid_precedence(order: list[str], vocabulary: dict[str, Any]) -> None:
    """Refuse an order that misses, repeats or invents a domain of *vocabulary*.

    The precedence is keyed by ISN's domain values, so an order that does not
    hold exactly those values — once each — points a classifier at a domain that
    does not exist or leaves one unranked. A vocabulary read empty cannot judge
    the order, so the check stands down rather than refusing a good order.
    """
    if not vocabulary:
        return
    known = set(vocabulary)
    declared = set(order)
    missing = sorted(known - declared)
    unknown = sorted(declared - known)
    repeated = sorted(domain for domain in declared if order.count(domain) > 1)
    if missing or unknown or repeated:
        raise ValueError(
            "domain precedence does not match the ISN vocabulary: "
            f"missing={missing} unknown={unknown} repeated={repeated}"
        )


@lru_cache(maxsize=1)
def load_domain_precedence() -> list[str]:
    """The declared precedence over the physics domains, earliest first.

    This is the one order the standard-name selector and the DD path classifier
    both read: when more than one domain's scope fits a quantity, the one
    earliest here wins. Bands and their rationale live in ``domains.yaml``.
    """
    order = _declared_precedence()
    _refuse_invalid_precedence(order, load_isn_physics_domains())
    return order


@lru_cache(maxsize=1)
def load_isn_physics_domains() -> dict[str, dict[str, Any]]:
    """Load the canonical ISN vocabulary: description, category and IDS per domain.

    ISN owns the domain values, their one-line descriptions and their categories
    in ``imas_standard_names/grammar/vocabularies/physics_domains.yml``. This
    reads that file rather than repeating it, so the guidance and the vocabulary
    cannot drift. Returns an empty mapping when the file cannot be read.
    """
    try:
        data = (
            yaml.safe_load(_isn_physics_domains_path().read_text(encoding="utf-8"))
            or {}
        )
    except (OSError, yaml.YAMLError):
        return {}
    domains = data.get("physics_domains") or {}
    return {
        name: {
            "description": meta.get("description", ""),
            "category": meta.get("category", ""),
            "ids": list(meta.get("ids") or []),
        }
        for name, meta in domains.items()
    }


def _isn_physics_domains_path() -> Path:
    """Locate the ISN physics-domain vocabulary file in the installed package."""
    import imas_standard_names

    return (
        Path(imas_standard_names.__file__).parent
        / "grammar"
        / "vocabularies"
        / "physics_domains.yml"
    )


__all__ = [
    "DOMAINS_SCHEMA",
    "IDS_DOMAINS_FILE",
    "load_domain_guidance",
    "load_domain_precedence",
    "load_isn_physics_domains",
]

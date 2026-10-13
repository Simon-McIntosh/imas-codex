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
    try:
        data = yaml.safe_load(DOMAINS_SCHEMA.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {"categories": {}, "domains": {}}
    return {
        "categories": data.get("categories") or {},
        "domains": data.get("domains") or {},
    }


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
    "load_isn_physics_domains",
]

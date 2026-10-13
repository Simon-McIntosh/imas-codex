"""Coerce physics domains for catalog and review callers."""

from __future__ import annotations

from collections.abc import Iterable

from imas_codex.standard_names import domain_ranking


def pick_primary_domain(domains: Iterable[str]) -> str | None:
    """Select the primary domain from the declared domain order."""
    return domain_ranking.select_primary_domain(domains)


def domain_key(value, fallback: str = "unknown") -> str:
    """Coerce a scalar or list of domains to one grouping key."""
    if value is None:
        return fallback
    if isinstance(value, str):
        domains = [value.strip()] if value.strip() else []
    elif isinstance(value, list | tuple):
        domains = [d.strip() for d in value if isinstance(d, str) and d.strip()]
    else:
        return fallback
    return domain_ranking.select_primary_domain(domains) or fallback


def domain_list(value) -> list[str]:
    """Coerce a possibly-scalar ``physics_domain`` value to a list of strings.

    Returns ``[]`` for None/empty. Trims whitespace and drops empty items.
    """
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list | tuple):
        return [d.strip() for d in value if isinstance(d, str) and d.strip()]
    return []

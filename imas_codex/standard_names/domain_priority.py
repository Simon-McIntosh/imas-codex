"""Coerce physics domains for catalog and review callers."""

from __future__ import annotations

from imas_codex.standard_names import domain_ranking


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

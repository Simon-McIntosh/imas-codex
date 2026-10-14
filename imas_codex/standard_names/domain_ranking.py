"""Declared, stable order for a standard name's physics domain."""

from __future__ import annotations

from collections.abc import Iterable

from imas_codex.definitions.physics import load_domain_precedence

# The declared precedence over the physics domains, read from the guidance file
# (imas_codex/definitions/physics/domains.yaml). The rank is the position in
# this order — lower index = more central. This is one of the two readers of
# that order; the DD path classifier is the other, so a name and a path settle
# the same overlap the same way.
FALLBACK_RANK_TABLE: tuple[str, ...] = tuple(load_domain_precedence())

_RANK_INDEX = {domain: rank for rank, domain in enumerate(FALLBACK_RANK_TABLE)}


def select_primary_domain(domains: Iterable[str]) -> str | None:
    """Select the highest-ranked member of a domain set, or ``None`` if empty.

    All supplied values must belong to the declared vocabulary. The answer
    depends only on the set, so duplicates and arrival order have no effect.
    """
    unique = set(domains)
    unknown = unique - _RANK_INDEX.keys()
    if unknown:
        raise ValueError(f"Unknown physics domain: {sorted(map(str, unknown))}")
    return min(unique, key=_RANK_INDEX.__getitem__) if unique else None


def maybe_promote_domain(current: str | None, candidate: str | None) -> str | None:
    """Compatibility adapter for graph writers pending set-based rewiring."""
    return select_primary_domain(domain for domain in (current, candidate) if domain)


def merge_source_domains(existing: list[str] | None, *new: str | None) -> list[str]:
    """Append *new* domains to *existing*, deduplicate, preserve order.

    ``None`` and empty values are skipped.  Existing order is preserved;
    new domains are appended in argument order.  Used to maintain the
    ``source_domains`` accumulator on :class:`StandardName` nodes.
    """
    out: list[str] = []
    seen: set[str] = set()
    for d in list(existing or []) + list(new):
        if not d:
            continue
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


__all__ = [
    "FALLBACK_RANK_TABLE",
    "select_primary_domain",
    "maybe_promote_domain",
    "merge_source_domains",
]

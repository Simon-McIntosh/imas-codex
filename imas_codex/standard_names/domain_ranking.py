"""Declared, stable order for a standard name's physics domain."""

from __future__ import annotations

from collections.abc import Iterable

# Declared ordering of PhysicsDomain enum members; the
# rank is the position in this tuple (lower index = more central).
#
# Design intent (lower rank = higher priority):
#   - Plasma core physics first (equilibrium, transport, MHD)
#   - Heating & current drive next
#   - Edge / SOL / divertor
#   - Diagnostics grouped (magnetics first as backbone)
#   - Operational / engineering / metadata last
#   - 'general' acts as the lowest-priority (most generic) domain.
FALLBACK_RANK_TABLE: tuple[str, ...] = (
    "equilibrium",
    "magnetohydrodynamics",
    "transport",
    "core_plasma_physics",
    "turbulence",
    "gyrokinetics",
    "fast_particles",
    "runaway_electrons",
    "auxiliary_heating",
    "current_drive",
    "waves",
    "edge_plasma_physics",
    "plasma_wall_interactions",
    "divertor_physics",
    "fueling",
    "magnetic_field_diagnostics",
    "plasma_measurement_diagnostics",
    "particle_measurement_diagnostics",
    "electromagnetic_wave_diagnostics",
    "radiation_measurement_diagnostics",
    "mechanical_measurement_diagnostics",
    "spectroscopy",
    "neutronics",
    "plasma_control",
    "plasma_initiation",
    "machine_operations",
    "magnetic_field_systems",
    "structural_components",
    "plant_systems",
    "data_management",
    "computational_workflow",
    "general",
)

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

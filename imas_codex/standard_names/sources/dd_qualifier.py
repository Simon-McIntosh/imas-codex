"""DD source qualifier: determines which DD paths should receive standard names.

Consolidates all DD-specific qualification logic into a single qualifier:

- **Structural checks** (Python predicates): string types, error fields,
  placeholder containers, process slots, duplicate IDSs.
- **Declarative deny rules** (YAML): geometry primitives, GGD metadata,
  temporal coordinates, boolean flags — loaded from ``config/extract_deny.yaml``.
- **Unit eligibility**: mixed units, unparseable units.

The YAML deny list handles stable policy rules that change infrequently.
Python predicates handle structural invariants that require code logic.
Both produce the same ``Qualification`` result type.
"""

from __future__ import annotations

import re

from imas_codex.standard_names.extract_deny import match_deny_rule
from imas_codex.standard_names.sources.base import (
    ELIGIBLE,
    Qualification,
    SourceCandidate,
    not_physical,
    skip,
)

# Suffixes indicating error companion fields.
_ERROR_SUFFIXES: tuple[str, ...] = ("_error_upper", "_error_lower", "_error_index")

# Pattern matching placeholder / generic-container leaf names.
_PLACEHOLDER_RE = re.compile(
    r"^constant_(float|integer|boolean|string)_value$"
    r"|^generic_(float|integer)$"
)

# Path segments indicating configurable-meaning slots.
_CONFIGURABLE_SEGMENTS: tuple[str, ...] = ("process",)


def qualify_dd(candidate: SourceCandidate) -> Qualification:
    """Qualify a DD path for standard name generation.

    Runs checks in cost order: cheap structural checks first, then the
    YAML deny list (which loads and caches rules on first call).

    Args:
        candidate: Normalized DD source candidate.

    Returns:
        ``Qualification`` with eligible=True if the path should proceed
        to LLM composition, or a skip/not_physical result with reason codes.
    """
    # --- Structural checks (Python predicates) ---

    # S0: String-typed leaves — names, descriptions, identifiers.
    if candidate.value_type and candidate.value_type.startswith("STR"):
        return skip(
            "string_data_type",
            f"String-typed leaf ({candidate.value_type}) — not a physical quantity.",
        )

    # S1: Duplicate IDS — core_instant_changes duplicates core_profiles.
    ids_name = candidate.metadata.get("ids_name", "")
    if ids_name == "core_instant_changes":
        return skip(
            "duplicate_ids",
            "core_instant_changes duplicates core_profiles quantities "
            "with change_in_* prefixes.",
        )

    # S2: Error companion fields — defensive catch for any that pass
    # the node_category gate.
    path = candidate.source_id
    if any(suffix in path for suffix in _ERROR_SUFFIXES):
        return skip(
            "error_companion_field",
            f"Error companion field ({path.rsplit('/', 1)[-1]}).",
        )

    # S3: Placeholder / generic-container leaves.
    leaf = path.rsplit("/", 1)[-1] if "/" in path else path
    if _PLACEHOLDER_RE.match(leaf):
        return skip(
            "placeholder_container",
            f"Generic container leaf ({leaf}) — describes data type, not physics.",
        )

    # S4: Configurable-meaning process slots.
    hierarchy = candidate.hierarchy
    if any(seg in _CONFIGURABLE_SEGMENTS for seg in hierarchy):
        return skip(
            "configurable_meaning",
            "Path inside a /process/ structure — concrete quantity is "
            "determined by sibling identifier at runtime.",
        )

    # S5: Unit eligibility — mixed units are ineligible by definition.
    if candidate.unit == "mixed":
        return skip(
            "dd_unit_mixed_non_standard",
            "DD unit is 'mixed' — heterogeneous dimensions are "
            "non-standard and ineligible for standard names.",
        )

    # S6: Unparseable units.
    if candidate.unit and _is_unparseable_unit(candidate.unit):
        return skip(
            "dd_unit_unresolvable",
            f"Unit '{candidate.unit}' is not a valid SI expression.",
        )

    # --- Declarative deny rules (YAML) ---
    # Build node_attrs dict for attribute-predicate rules (data_type,
    # units, documentation).
    node_attrs = {
        "data_type": candidate.value_type,
        "units": candidate.unit,
        "documentation": candidate.documentation,
    }
    rule = match_deny_rule(path, node_attrs=node_attrs)
    if rule is not None:
        status = (
            not_physical("", "").status
            if rule.status == "not_physical_quantity"
            else skip("", "").status
        )
        return Qualification(
            status=status,
            reason_code=rule.skip_reason,
            reason_detail=rule.reason,
        )

    return ELIGIBLE


def _is_unparseable_unit(unit: str) -> bool:
    """Return True if *unit* cannot be parsed as a valid SI expression."""
    if not unit or not unit.strip():
        return True
    unit = unit.strip()
    # Dimensionless sentinels are valid.
    if unit in ("1", "dimensionless", "-", "none"):
        return False
    # Whitespace in unit string.
    if re.search(r"\s", unit):
        return True
    # Non-numeric exponents (e.g., m^dimension).
    if re.search(r"\^[a-zA-Z]", unit):
        return True
    # Attempt pint parse.
    try:
        from imas_codex.units import normalize_unit_symbol

        normalize_unit_symbol(unit)
    except Exception:
        return True
    return False

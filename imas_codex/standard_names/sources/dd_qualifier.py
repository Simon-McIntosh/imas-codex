"""DD source qualifier: determines which DD paths should receive standard names.

Structural Python predicates only — no YAML deny lists, no glob patterns.
Each check identifies a class of DD path that is structurally un-nameable
(string data, error fields, coordinate axes, etc.). All semantic quality
judgments (geometry specificity, engineering vs physics) are delegated to
the LLM at compose time via the skip mechanism in the compose prompt.

Design principle: checks must be justifiable from the *nature* of the data,
not from historical LLM scoring. If the LLM generates bad names for a path,
the fix is better prompting or review — not denying the source.
"""

from __future__ import annotations

import re

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

# Local coordinate frame unit vector segments — abstract reference frame
# definitions (dimensionless basis vectors), not measurable quantities.
_UNIT_VECTOR_SEGMENTS: tuple[str, ...] = (
    "x1_unit_vector",
    "x2_unit_vector",
    "x3_unit_vector",
)

# Documentation phrases that indicate boolean configuration flags
# rather than physical quantities.
_FLAG_PHRASES: tuple[str, ...] = (
    "flag = 1",
    "flag = 0",
    "flag if",
    "1 if",
    "0 if not",
)


def qualify_dd(candidate: SourceCandidate) -> Qualification:
    """Qualify a DD path for standard name generation.

    Runs structural checks in cost order (cheap string checks first).
    All semantic quality judgments are delegated to the LLM compose step.

    Args:
        candidate: Normalized DD source candidate.

    Returns:
        ``Qualification`` with eligible=True if the path should proceed
        to LLM composition, or a skip/not_physical result with reason codes.
    """
    # --- Structural checks (Python predicates) ---

    # S0: String-typed leaves — names, descriptions, identifiers.
    # Match STR_0D / STR_1D (string scalars/arrays) via the trailing underscore —
    # NOT STRUCTURE / STRUCT_ARRAY, which also begin "STR" but are signal
    # containers admitted by the leaf-invariant signal signature.
    if candidate.value_type and candidate.value_type.startswith("STR_"):
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

    # S7: Temporal coordinate arrays — dimension axes for time-varying
    # data, not physics quantities. Top-level <ids>/time (depth 1) is
    # exempt; nested time paths (depth >= 3 segments) with coordinate
    # category are filtered.
    node_category = candidate.metadata.get("node_category", "")
    if leaf == "time" and len(hierarchy) >= 3 and node_category == "coordinate":
        return skip(
            "temporal_coordinate",
            "Nested time coordinate array — dimension axis for "
            "time-varying data, not a physical quantity.",
        )

    # S8: Local coordinate frame unit vectors — abstract reference frame
    # basis vector definitions (x1/x2/x3_unit_vector), not measurable.
    if any(seg in hierarchy for seg in _UNIT_VECTOR_SEGMENTS):
        return skip(
            "local_coordinate_frame",
            "Local coordinate frame unit vector component — abstract "
            "reference frame definition, not a measurable quantity.",
        )

    # S9: GGD grid topology subtree — structural mesh metadata defining
    # spaces, subsets, coordinate types. Physics values live under ggd/*
    # (not grid_ggd/*).
    if "grid_ggd" in hierarchy:
        return skip(
            "ggd_structural_metadata",
            "GGD grid topology subtree (grid_ggd) — structural mesh "
            "metadata, not a physical observable.",
        )

    # S10: GGD grid back-reference indices — integer bookkeeping fields
    # inside ggd/* that reference grid definitions.
    if "ggd" in hierarchy and leaf in ("grid_index", "grid_subset_index"):
        return skip(
            "ggd_structural_metadata",
            f"GGD grid back-reference ({leaf}) — structural bookkeeping "
            "index into grid_ggd topology, not a physical quantity.",
        )

    # S11: Configuration flags — boolean switches (INT_0D/INT_1D, no
    # units, flag-style documentation) that describe calculation setup.
    if _is_configuration_flag(candidate):
        return not_physical(
            "configuration_flag",
            "Boolean configuration flag (INT data type, no units, "
            "flag-style documentation) — describes setup, not a quantity.",
        )

    return ELIGIBLE


def _is_configuration_flag(candidate: SourceCandidate) -> bool:
    """Return True if *candidate* looks like a boolean configuration flag."""
    if candidate.value_type not in ("INT_0D", "INT_1D"):
        return False
    if candidate.unit and candidate.unit not in ("", "-", "none"):
        return False
    doc = (candidate.documentation or "").lower()
    return any(phrase in doc for phrase in _FLAG_PHRASES)


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

"""Family detection and axis ordering for standard name generation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from imas_standard_names.grammar.model_types import Component, GeometricBase

# Right-handed cylindrical coordinate ordering: R̂ × φ̂ = Ẑ
# This is the standard tokamak convention. NOT alphabetical (R, Z, φ).
AXIS_ORDER: dict[str, int] = {
    # Cylindrical — right-handed (R, φ, Z)
    "radial": 0,
    "r": 0,
    "major_radius": 0,
    "toroidal": 1,
    "phi": 1,
    "toroidal_angle": 1,
    "vertical": 2,
    "z": 2,
    # Poloidal (after cylindrical triplet)
    "poloidal": 3,
    "theta": 3,
    # Field-aligned
    "parallel": 4,
    "perpendicular": 5,
    "normal": 6,
    "tangential": 7,
    "binormal": 8,
    # Normalized variants inherit parent ordering
    "normalized_radial": 0,
    "normalized_toroidal": 1,
    "normalized_vertical": 2,
    "normalized_poloidal": 3,
    "normalized_parallel": 4,
    "normalized_perpendicular": 5,
    # Cartesian — right-handed (x, y, z); z already defined above at 2
    "x": 0,
    "y": 1,
}

# ---------------------------------------------------------------------------
# Axis suffix mapping (built from ISN Component enum)
# ---------------------------------------------------------------------------

# All ISN Component values
_COMPONENT_AXES = {c.value for c in Component}

# Map DD suffixes to ISN Component axes. Identity mappings for every Component
# value come first; the explicit DD short-form mappings then override them.
# ``z`` is frame-dependent: the default here is ``vertical`` (cylindrical Z /
# machine-vertical), and must win over the ``z`` Component's identity mapping —
# :func:`_classify_suffix` overrides it to ``z`` when the node is Cartesian.
_SUFFIX_TO_AXIS: dict[str, str] = {_axis: _axis for _axis in _COMPONENT_AXES}
_SUFFIX_TO_AXIS.update(
    {
        "r": "radial",
        "z": "vertical",
        "phi": "toroidal",
        "tor": "toroidal",
        "pol": "poloidal",
    }
)

# Axis tokens that mark a vector node's coordinate frame. A node carrying a
# cylindrical member is cylindrical (z → vertical); a purely Cartesian node
# is Cartesian (z → z). Standalone / ambiguous nodes default to cylindrical.
_CYLINDRICAL_AXES = frozenset({"radial", "toroidal", "poloidal"})
_CARTESIAN_AXES = frozenset({"x", "y"})

# GeometricBase values for parent path matching
_GEOMETRIC_BASES = {g.value for g in GeometricBase}

# ---------------------------------------------------------------------------
# DD derivative map
# ---------------------------------------------------------------------------

# Known DD derivative patterns → ISN decomposition (numerator, denominator)
DD_DERIVATIVE_MAP: dict[str, tuple[str, str]] = {
    "darea_dpsi": ("area", "poloidal_magnetic_flux"),
    "dpressure_dpsi": ("pressure", "poloidal_magnetic_flux"),
    "dvolume_dpsi": ("volume", "poloidal_magnetic_flux"),
    "dpsi_drho_tor": ("poloidal_magnetic_flux", "normalised_toroidal_flux_coordinate"),
    "darea_drho_tor": ("area", "normalised_toroidal_flux_coordinate"),
    "dvolume_drho_tor": ("volume", "normalised_toroidal_flux_coordinate"),
    "dc_dr_minor_norm": ("triangularity_upper", "normalised_minor_radius"),
    "delongation_dr_minor_norm": ("elongation", "normalised_minor_radius"),
    "dgeometric_axis_r_dr_minor": ("geometric_axis_radial_position", "minor_radius"),
    "dgeometric_axis_z_dr_minor": ("geometric_axis_vertical_position", "minor_radius"),
    "ds_dr_minor_norm": ("squareness", "normalised_minor_radius"),
}

# Regex for generic d{X}_d{Y} patterns
_DERIVATIVE_RE = re.compile(r"^d(.+)_d(.+)$")

_FALLBACK_ORDER = 99  # Unknown axes sort last


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FamilyMember:
    """A single member of a vector/geometric/derivative family."""

    dd_path: str
    axis: str  # e.g., "radial", "toroidal", "r", "z", "phi"
    unit: str  # SI unit string
    suffix: str  # DD leaf name (e.g., "j_tor", "r", "darea_dpsi")


@dataclass
class VectorFamily:
    """A group of DD paths that form a vector/geometric/derivative set."""

    parent_path: str  # DD structural parent path
    family_type: str  # "physical_vector" | "geometric_coordinate" | "derivative"
    members: list[FamilyMember] = field(default_factory=list)
    parent_name: str | None = None  # Deterministic ISN parent name (if derivable)
    unit_uniform: bool = True  # Whether all members share a single unit
    units: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def _is_z_leaf(suffix: str) -> bool:
    """Whether a DD leaf suffix is the vertical/Cartesian ``z`` axis leaf."""
    return suffix == "z" or suffix.endswith("_z")


def _classify_suffix(suffix: str, frame: str = "cylindrical") -> str | None:
    """Map a DD leaf suffix to an ISN axis name, or None.

    ``frame`` selects the reading of a ``z`` leaf: ``"cartesian"`` names it the
    ``z`` axis (third member of an x, y, z triple); ``"cylindrical"`` (the
    default) names it ``vertical`` (cylindrical Z / machine-vertical).
    """
    # Direct lookup
    axis: str | None = None
    if suffix in _SUFFIX_TO_AXIS:
        axis = _SUFFIX_TO_AXIS[suffix]
    else:
        # Check if suffix ends with a known axis (e.g. "j_tor" → "tor" → "toroidal")
        for short, mapped in _SUFFIX_TO_AXIS.items():
            if suffix.endswith(f"_{short}"):
                axis = mapped
                break
    if frame == "cartesian" and _is_z_leaf(suffix):
        return "z"
    return axis


def _frame_from_axes(axes: set[str | None]) -> str:
    """Infer a vector node's coordinate frame from its members' axis tokens.

    A node with any cylindrical member (radial/toroidal/poloidal) is
    cylindrical; a purely Cartesian node (x/y) is Cartesian; anything else
    (a lone z leaf, unknown axes) defaults to cylindrical.
    """
    if axes & _CYLINDRICAL_AXES:
        return "cylindrical"
    if axes & _CARTESIAN_AXES:
        return "cartesian"
    return "cylindrical"


def _extract_derivative_denominator(suffix: str) -> str | None:
    """Extract the denominator from a d{X}_d{Y} pattern, or None."""
    if suffix in DD_DERIVATIVE_MAP:
        return DD_DERIVATIVE_MAP[suffix][1]
    m = _DERIVATIVE_RE.match(suffix)
    if m:
        return m.group(2)
    return None


def detect_families(items: list[dict]) -> list[VectorFamily]:
    """Detect vector, geometric, and derivative families from DD paths.

    Parameters
    ----------
    items : list[dict]
        Each dict must have ``path`` (str) and ``unit`` (str) keys.

    Returns
    -------
    list[VectorFamily]
        Families with 2+ members.
    """
    # Group items by DD structural parent
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        path = item["path"]
        if "/" not in path:
            continue
        parent = path.rsplit("/", 1)[0]
        groups[parent].append(item)

    families: list[VectorFamily] = []

    for parent_path, group_items in groups.items():
        # Extract the parent name (last segment of parent path)
        parent_name = (
            parent_path.rsplit("/", 1)[-1] if "/" in parent_path else parent_path
        )
        is_geometric = parent_name in _GEOMETRIC_BASES

        # --- Axis-based detection (physical_vector / geometric_coordinate) ---
        # Read the node's frame from the sibling leaf set first (cylindrical vs
        # Cartesian) so a ``z`` leaf is named ``vertical`` (cylindrical Z) or
        # ``z`` (Cartesian third axis) consistently across the family.
        frame = _frame_from_axes(
            {_classify_suffix(item["path"].rsplit("/", 1)[-1]) for item in group_items}
        )
        axis_members: list[FamilyMember] = []
        for item in group_items:
            suffix = item["path"].rsplit("/", 1)[-1]
            axis = _classify_suffix(suffix, frame=frame)
            if axis is not None:
                axis_members.append(
                    FamilyMember(
                        dd_path=item["path"],
                        axis=axis,
                        unit=item.get("unit", ""),
                        suffix=suffix,
                    )
                )

        if len(axis_members) >= 2:
            units = {m.unit for m in axis_members}
            family_type = "geometric_coordinate" if is_geometric else "physical_vector"
            families.append(
                VectorFamily(
                    parent_path=parent_path,
                    family_type=family_type,
                    members=axis_members,
                    parent_name=parent_name if is_geometric else None,
                    unit_uniform=len(units) == 1,
                    units=units,
                )
            )
            continue  # Don't double-classify

        # --- Derivative-based detection ---
        deriv_by_denom: dict[str, list[FamilyMember]] = defaultdict(list)
        for item in group_items:
            suffix = item["path"].rsplit("/", 1)[-1]
            denom = _extract_derivative_denominator(suffix)
            if denom is not None:
                deriv_by_denom[denom].append(
                    FamilyMember(
                        dd_path=item["path"],
                        axis=denom,
                        unit=item.get("unit", ""),
                        suffix=suffix,
                    )
                )

        for _denom, members in deriv_by_denom.items():
            if len(members) >= 2:
                units = {m.unit for m in members}
                families.append(
                    VectorFamily(
                        parent_path=parent_path,
                        family_type="derivative",
                        members=members,
                        parent_name=None,
                        unit_uniform=len(units) == 1,
                        units=units,
                    )
                )

    return families


def sort_by_axis_convention(
    items: list[dict],
    axis_key: str = "axis",
) -> list[dict]:
    """Sort items by physics-conventional axis ordering.

    Uses right-handed coordinate conventions:
    - Cylindrical: R, φ, Z (NOT alphabetical R, Z, φ)
    - Field-aligned: parallel, perpendicular
    - Cartesian: x, y, z

    Parameters
    ----------
    items : list[dict]
        Items to sort. Each must have a key given by axis_key.
    axis_key : str
        Key in each item dict that holds the axis/component name.

    Returns
    -------
    list[dict]
        New list sorted by axis convention. Original list unchanged.
    """
    return sorted(
        items,
        key=lambda item: AXIS_ORDER.get(
            str(item.get(axis_key, "")).lower(), _FALLBACK_ORDER
        ),
    )

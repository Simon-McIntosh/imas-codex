"""Family detection and axis ordering for standard name generation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from imas_standard_names.grammar.model_types import Component, GeometricBase

from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
)

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

# DD structural primitives whose direct children jointly describe one object.
_GEOMETRY_PRIMITIVES = frozenset(
    {"annulus", "arcs_of_circle", "oblique", "outline", "rectangle"}
)

_GEOMETRY_ERROR_SUFFIXES = ("_error_lower", "_error_upper", "_error_index")

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


@dataclass
class GeometryField:
    """One direct DD field describing a geometric object."""

    dd_path: str
    suffix: str
    unit: str


@dataclass
class GeometryFamily:
    """All direct DD fields that jointly describe one geometric primitive."""

    parent_path: str
    primitive: str
    members: list[GeometryField] = field(default_factory=list)


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


def detect_geometry_families(items: list[dict]) -> list[GeometryFamily]:
    """Detect complete field groups for named DD geometric primitives.

    A family's identity comes from its structural parent's primitive name, not
    from the suffixes of its children. Direct error-bound and uncertainty-index
    leaves are excluded. A primitive containing only axis-classified leaves is
    already represented by :func:`detect_families` and is not duplicated here.

    Parameters
    ----------
    items : list[dict]
        Each dict must have ``path`` (str); ``unit`` (str) is optional.

    Returns
    -------
    list[GeometryFamily]
        Shape-keyed families with at least two geometric fields and at least
        one field that the axis-only detector cannot see.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        path = item["path"]
        if "/" not in path:
            continue
        parent_path = path.rsplit("/", 1)[0]
        groups[parent_path].append(item)

    families: list[GeometryFamily] = []
    for parent_path, group_items in groups.items():
        primitive = parent_path.rsplit("/", 1)[-1]
        if primitive not in _GEOMETRY_PRIMITIVES:
            continue

        members = [
            GeometryField(
                dd_path=item["path"],
                suffix=item["path"].rsplit("/", 1)[-1],
                unit=item.get("unit", ""),
            )
            for item in group_items
            if not item["path"].rsplit("/", 1)[-1].endswith(_GEOMETRY_ERROR_SUFFIXES)
        ]
        if len(members) < 2:
            continue
        if all(_classify_suffix(member.suffix) is not None for member in members):
            continue

        families.append(
            GeometryFamily(
                parent_path=parent_path,
                primitive=primitive,
                members=members,
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


# ---------------------------------------------------------------------------
# Pattern families — grammar-identical cohorts
# ---------------------------------------------------------------------------

# The name lifecycle stage excluded from every family read.
_LIVE_STAGE_EXCLUSION = "superseded"

#: Segments that fix a name's identity. Two live names share a pattern family
#: only when every one of these is equal. ``transformation`` carries the
#: operator token (``flux_surface_averaged``, ``time_derivative`` …), which is
#: deliberately NOT a pattern axis: a reduction operator and its unreduced
#: parent therefore land in different pattern families and are joined only by
#: the explicit ``HAS_PARENT`` route.
#:
#: ``object`` carries the entity locus — the ISN locus registry's ``entity``
#: type (``antenna_strap``, ``bolometer`` …). An entity names a distinct
#: object, so ``radial_coordinate_of_<owner>`` over many owners is grammar
#: ancestry, not prose parallelism: two entity owners never share a family.
#: See :data:`PATTERN_VARIATION_AXES` for the locus types that do vary.
PATTERN_INVARIANT_SEGMENTS: tuple[str, ...] = (
    "physical_base",
    "geometric_base",
    "transformation",
    "process",
    "object",
)

#: Segments along which two names may differ and stay family-mates. An empty
#: segment is a value of its axis, so a name with no locus shares a signature
#: with its locus-qualified variants. ``component`` and ``coordinate`` name the
#: same ISN axis (they are parser aliases) and are stored in either column, so
#: both vary together. ``position``, ``geometry`` and ``region`` carry the
#: locus variation axis: the ISN locus registry's ``position``, ``geometry``
#: and ``region`` types place one quantity somewhere and are prose-parallel,
#: so ``electron_temperature`` and ``electron_temperature_at_magnetic_axis``
#: are family-mates. Only the ``entity`` type is invariant (see
#: :data:`PATTERN_INVARIANT_SEGMENTS`).
PATTERN_VARIATION_AXES: tuple[str, ...] = (
    "subject",
    "component",
    "coordinate",
    "position",
    "geometry",
    "region",
    "device",
)

#: Every grammar segment that participates in a pattern family.
PATTERN_SEGMENTS: tuple[str, ...] = PATTERN_INVARIANT_SEGMENTS + PATTERN_VARIATION_AXES


def _invariant_projection(alias: str) -> str:
    """Cypher projecting *alias*'s invariant segments as comparable strings."""
    return ",\n".join(
        f"     coalesce({alias}.{seg}, '') AS {seg}"
        for seg in PATTERN_INVARIANT_SEGMENTS
    )


def _invariant_equality(alias: str) -> str:
    """Cypher equality predicates binding *alias* to a projected seed."""
    return "\n  AND ".join(
        f"coalesce({alias}.{seg}, '') = {seg}" for seg in PATTERN_INVARIANT_SEGMENTS
    )


DOCS_STAGE_ACCEPTED = "accepted"
"""``docs_stage`` value marking a member whose documentation is accepted."""


def is_placeholder_description(description: str | None) -> bool:
    """True when *description* is absent or the deterministic-parent sentinel."""
    return (
        not description or description == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
    )


def select_anchor(
    parent_id: str | None,
    parent_docs_stage: str | None,
    parent_description: str | None,
    members: list[dict[str, Any]],
) -> str | None:
    """Select the documentation anchor for a family.

    One rule, shared by the docs prompt and the harmonize command, so both
    readers name the same member. Priority order:
      1. The parent itself, if ``docs_stage == 'accepted'`` and its
         description is real (non-placeholder).
      2. The member with the highest ``reviewer_score_docs`` among members
         with ``docs_stage == 'accepted'`` — the best signal we have that a
         member already meets the docs standard (accepted-but-unscored breaks
         ties deterministically by id).
      3. Last resort (no accepted member): a non-placeholder member chosen
         deterministically by id — NEVER by longest description, which
         propagates a verbose, non-canonical opening into the family.

    The anchor deliberately carries NO vocabulary or phrasing knowledge: the
    canonical opening a family should converge on is defined by the docs
    prompt/context, not hardcoded here. Score + determinism is the data-driven
    proxy — a member that already meets the standard scores well.

    Returns ``None`` when no candidate qualifies (family is flagged
    ``deferred`` by the caller).
    """
    if (
        parent_id
        and parent_docs_stage == DOCS_STAGE_ACCEPTED
        and not is_placeholder_description(parent_description)
    ):
        return parent_id

    accepted = [
        m
        for m in members
        if m.get("docs_stage") == DOCS_STAGE_ACCEPTED
        and not is_placeholder_description(m.get("description"))
    ]
    scored = [m for m in accepted if m.get("reviewer_score_docs") is not None]
    if scored:
        best = max(
            scored,
            key=lambda m: (m.get("reviewer_score_docs") or 0.0, m.get("id") or ""),
        )
        return best.get("id")
    if accepted:
        # Accepted but unscored — deterministic by id (NOT longest description).
        return min(accepted, key=lambda m: m.get("id") or "").get("id")

    non_placeholder = [
        m for m in members if not is_placeholder_description(m.get("description"))
    ]
    if non_placeholder:
        # Last resort — deterministic by id (NOT longest description); picking
        # by length propagates a verbose, non-canonical opening to the family.
        return min(non_placeholder, key=lambda m: m.get("id") or "").get("id")

    return None


# Rows for the seed's live pattern family: the seed fixes the invariant segment
# tuple, and every live name with the same tuple is a member. The invariant
# columns are derived from :data:`PATTERN_INVARIANT_SEGMENTS` so that constant
# is the single definition; a bounded ``:StandardName`` scan with equality
# predicates.
_PATTERN_FAMILY_MEMBERS_QUERY = f"""
MATCH (seed:StandardName {{id: $sn_id}})
WHERE coalesce(seed.name_stage, '') <> $superseded
WITH
{_invariant_projection("seed")}
MATCH (m:StandardName)
WHERE {_invariant_equality("m")}
  AND coalesce(m.name_stage, '') <> $superseded
RETURN m.id AS id,
       m.description AS description,
       m.documentation AS documentation,
       coalesce(m.docs_stage, '') AS docs_stage,
       m.reviewer_score_docs AS reviewer_score_docs,
       coalesce(m.physical_base, '') AS physical_base,
       coalesce(m.subject, '') AS subject,
       coalesce(m.component, '') AS component,
       coalesce(m.coordinate, '') AS coordinate,
       coalesce(m.object, '') AS object,
       coalesce(m.position, '') AS position,
       coalesce(m.geometry, '') AS geometry,
       coalesce(m.region, '') AS region,
       coalesce(m.device, '') AS device,
       coalesce(m.name_stage, '') AS name_stage
ORDER BY m.id
"""

# Families (live names grouped by invariant signature) with at least
# ``$min_members`` members. The grouping keys are the same single constant.
_PATTERN_FAMILY_CENSUS_QUERY = f"""
MATCH (m:StandardName)
WHERE coalesce(m.name_stage, '') <> $superseded
WITH
{_invariant_projection("m")},
     count(m) AS n
WHERE n >= $min_members
RETURN count(*) AS families, sum(n) AS members
"""


def _segment_value(segments: dict[str, Any], name: str) -> str:
    """One segment as a comparable string; an absent/empty segment is ``""``."""
    value = segments.get(name)
    return "" if value is None else str(value)


def pattern_signature(segments: dict[str, Any]) -> tuple[str, ...]:
    """Return the invariant pattern signature of a name's grammar segments.

    Two live names are family-mates iff their signatures are equal; their
    parses then differ only along :data:`PATTERN_VARIATION_AXES`. An absent
    or empty segment contributes ``""`` — it is a value of its axis, so a
    parent with no locus shares a signature with its locus-qualified children.
    An ``entity`` locus (the ``object`` segment) is invariant, so two entity
    owners never share a signature.
    """
    return tuple(_segment_value(segments, seg) for seg in PATTERN_INVARIANT_SEGMENTS)


def fetch_pattern_family(sn_id: str, gc: Any) -> list[dict[str, Any]]:
    """Return the live pattern-family rows of *sn_id* (the seed included).

    The family is every live (non-superseded) StandardName sharing the seed's
    invariant segment tuple, ordered by id. Returns ``[]`` when the seed is
    absent or superseded. The seed's own row is always present, so a name with
    no axis sibling yields a family of one. Never raises on a graph fault —
    the caller decides.
    """
    return [
        dict(row)
        for row in (
            gc.query(
                _PATTERN_FAMILY_MEMBERS_QUERY,
                sn_id=sn_id,
                superseded=_LIVE_STAGE_EXCLUSION,
            )
            or []
        )
    ]


def count_pattern_families(gc: Any, *, min_members: int = 3) -> dict[str, int]:
    """Count live pattern families with at least *min_members* members.

    Returns ``{"families": n, "members": m}`` where ``members`` is the total
    membership of those families.
    """
    rows = list(
        gc.query(
            _PATTERN_FAMILY_CENSUS_QUERY,
            min_members=min_members,
            superseded=_LIVE_STAGE_EXCLUSION,
        )
        or []
    )
    if not rows:
        return {"families": 0, "members": 0}
    row = rows[0]
    return {"families": row.get("families") or 0, "members": row.get("members") or 0}


def _family_member_collect() -> str:
    """Cypher collecting one family's member rows with the report columns.

    The documentation columns drive anchor selection and drift; the lint
    columns (unit, COCOS type, physics domain, position, links, link status)
    are what the review lint reads, so an enumerating caller hands it rows it
    can score without a second read.
    """
    return """collect({
        id: m.id,
        description: m.description,
        documentation: m.documentation,
        docs_stage: m.docs_stage,
        reviewer_score_docs: m.reviewer_score_docs,
        harmonized_group_signature: m.harmonized_group_signature,
        unit: m.unit,
        cocos_transformation_type: m.cocos_transformation_type,
        physics_domain: m.physics_domain,
        position: m.position,
        links: m.links,
        link_status: m.link_status
    }) AS members"""


def _pattern_family_rows_query(*, seeded: bool, node_filter: str) -> str:
    """Cypher enumerating pattern families, keyed on the invariant segments.

    *seeded* groups only the families that contain one of ``$seeds`` (two
    hops: the seeds' signatures, then every live name sharing one). Otherwise
    every live family is grouped. *node_filter* is an extra predicate on the
    member alias ``m``, so a caller narrows the population without restating
    the grouping.

    The grouping is the invariant segment tuple that
    :data:`_PATTERN_FAMILY_MEMBERS_QUERY` keys a single family on — the same
    :func:`_invariant_projection` / :func:`_invariant_equality` primitives —
    so the two spellings cannot disagree about what makes a family.
    """
    segments = ", ".join(PATTERN_INVARIANT_SEGMENTS)
    projection = _invariant_projection("m")
    collect = _family_member_collect()
    extra = f"\n  AND {node_filter}" if node_filter else ""
    if seeded:
        seed_projection = _invariant_projection("seed")
        equality = _invariant_equality("m")
        return f"""
MATCH (seed:StandardName)
WHERE seed.id IN $seeds AND coalesce(seed.name_stage, '') <> $superseded
WITH DISTINCT
{seed_projection}
MATCH (m:StandardName)
WHERE coalesce(m.name_stage, '') <> $superseded
  AND {equality}{extra}
WITH
{projection},
     {collect}
RETURN {segments}, members
"""
    return f"""
MATCH (m:StandardName)
WHERE coalesce(m.name_stage, '') <> $superseded{extra}
WITH
{projection},
     {collect}
RETURN {segments}, members
"""


def fetch_pattern_families(
    gc: Any,
    *,
    seeds: list[str] | None = None,
    node_filter: str = "",
    min_size: int = 1,
) -> list[dict[str, Any]]:
    """Enumerate live pattern families grouped by invariant grammar signature.

    Pass *seeds* to keep only the families containing at least one seed id;
    ``None`` enumerates every live family with at least *min_size* members.
    *node_filter* is an extra predicate on the member alias, so a caller
    selects a population without restating the grouping.

    Returns one dict per family: ``invariant`` (the tuple of invariant
    segment values), ``physical_base``, ``members`` (member rows carrying the
    documentation and lint columns) and ``n``. Read-only: it never writes.
    """
    if seeds is not None and not seeds:
        return []
    rows = (
        gc.query(
            _pattern_family_rows_query(
                seeded=seeds is not None, node_filter=node_filter
            ),
            seeds=sorted(set(seeds or [])),
            superseded=_LIVE_STAGE_EXCLUSION,
        )
        or []
    )

    families: list[dict[str, Any]] = []
    for row in rows:
        members = [dict(m) for m in (row.get("members") or [])]
        if len(members) < min_size:
            continue
        families.append(
            {
                "invariant": tuple(
                    row.get(seg) or "" for seg in PATTERN_INVARIANT_SEGMENTS
                ),
                "physical_base": row.get("physical_base") or "",
                "members": members,
                "n": len(members),
            }
        )
    families.sort(key=lambda f: (f["physical_base"], f["invariant"]))
    return families


# The base physical quantity a family is expressed over — the row whose id is
# the family's ``physical_base``. It is a member of the family only when the
# family's ``transformation`` is empty; an operator-reduced family
# (``flux_surface_averaged`` …) therefore looks its base up separately.
_FAMILY_BASE_QUERY = """
MATCH (b:StandardName {id: $base_id})
WHERE coalesce(b.name_stage, '') <> $superseded
RETURN b.id AS id,
       b.description AS description,
       b.documentation AS documentation,
       coalesce(b.docs_stage, '') AS docs_stage,
       b.reviewer_score_docs AS reviewer_score_docs
"""


def fetch_family_base(base_id: str, gc: Any) -> dict[str, Any]:
    """Return the base-quantity row for *base_id*, or ``{}`` when absent.

    The returned dict carries the documentation columns the sibling-family
    context reads (``description``, ``documentation``, ``docs_stage``,
    ``reviewer_score_docs``); every variation segment is absent, which the
    caller reads as the unqualified base.
    """
    rows = list(
        gc.query(
            _FAMILY_BASE_QUERY,
            base_id=base_id,
            superseded=_LIVE_STAGE_EXCLUSION,
        )
        or []
    )
    return dict(rows[0]) if rows else {}

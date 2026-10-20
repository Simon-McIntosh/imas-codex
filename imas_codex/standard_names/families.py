"""Family detection and axis ordering for standard name generation."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
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


def dd_source_exists(alias: str) -> str:
    """Cypher predicate: *alias* is bound to a live Data Dictionary source.

    A name row carries the flag under ``has_dd_source``. The producing edge is
    ``PRODUCED_NAME`` from a ``StandardNameSource`` of type ``dd``, and a source
    the catalog has retired (``status = 'stale'``) no longer produces anything,
    so it does not count. One definition, used by every family query that
    reports the flag, so no two selectors can disagree about what "has a DD
    source" means.
    """
    return (
        f"EXISTS {{ MATCH ({alias})<-[:PRODUCED_NAME]-"
        "(dd:StandardNameSource {source_type: 'dd'}) "
        "WHERE coalesce(dd.status, '') <> 'stale' }"
    )


def member_has_dd_source(row: Mapping[str, Any]) -> bool:
    """Whether *row* may anchor a family: it is entailed by a DD source.

    A family takes its documentation template from its anchor, so the rewrite
    the anchor sets for its members is only checkable against source text when
    the anchor has a Data Dictionary source of its own. A name with none — a
    derived parent folded from other evidence, or a name whose only binding is
    a facility signal — has no source its template can be verified against, and
    would then template members it holds no textual authority over.

    The flag is projected by every family query in this module, but a row can
    still reach a selector through a caller that rebuilds it — ``harmonize``
    maps query rows to member dicts — so an absent flag is read as no DD
    source, not as unknown, and the filter fails closed. A caller that carries
    the flag through states it; one that drops it disqualifies the member
    rather than silently licensing an unverifiable template.
    """
    return bool(row.get("has_dd_source"))


def select_anchor(
    parent_id: str | None,
    parent_docs_stage: str | None,
    parent_description: str | None,
    members: list[dict[str, Any]],
    *,
    parent_has_dd_source: bool | None = None,
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

    Every candidate must also be entailed by a Data Dictionary source
    (:func:`member_has_dd_source`): a member with none sets a template its
    rewrite cannot be checked against, and would then template members it has
    no textual authority over. The rule applies to the parent as well, and to
    the parent the caller supplies a state for: when the parent is one of
    ``members`` its own row decides, and otherwise ``parent_has_dd_source``
    does — ``True`` only, so a caller that does not report the parent's source
    leaves the accepted-parent rule off rather than firing it on an
    unverifiable parent. A family in which no member qualifies has no anchor at
    all, and the caller records it ``deferred``.

    The anchor deliberately carries NO vocabulary or phrasing knowledge: the
    canonical opening a family should converge on is defined by the docs
    prompt/context, not hardcoded here. Score + determinism is the data-driven
    proxy — a member that already meets the standard scores well.

    Returns ``None`` when no candidate qualifies (family is flagged
    ``deferred`` by the caller).
    """
    parent_row = next((m for m in members if m.get("id") == parent_id), None)
    parent_sourced = (
        member_has_dd_source(parent_row)
        if parent_row is not None
        else parent_has_dd_source is True
    )
    if (
        parent_id
        and parent_sourced
        and parent_docs_stage == DOCS_STAGE_ACCEPTED
        and not is_placeholder_description(parent_description)
    ):
        return parent_id

    eligible = [m for m in members if member_has_dd_source(m)]
    accepted = [
        m
        for m in eligible
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
        m for m in eligible if not is_placeholder_description(m.get("description"))
    ]
    if non_placeholder:
        # Last resort — deterministic by id (NOT longest description); picking
        # by length propagates a verbose, non-canonical opening to the family.
        return min(non_placeholder, key=lambda m: m.get("id") or "").get("id")

    return None


# ---------------------------------------------------------------------------
# Judged anchor: the member whose documentation best templates the family
# ---------------------------------------------------------------------------

#: Score question name the judge answers with an expected score and a per-level
#: distribution. A name, not sent to the model — code reads the answer under it.
TEMPLATE_FITNESS_QUESTION = "template_fitness"

#: Score levels, low first, each a concrete situation. The level numbers scale
#: a member's documentation by how well it would template the family; the judge
#: supplies only the position, as its expectation over the levels.
TEMPLATE_FITNESS_LEVELS: tuple[str, ...] = (
    "Names or describes a quantity other than the family's, or contradicts the "
    "family's pattern — templating from it would mislead every other member.",
    "Specific to this member alone: no opening or structure that could carry "
    "over to the family's other members.",
    "Reusable in parts: states the family's shared quantity but departs from "
    "the house style in terminology or structure, so a member templated from "
    "it would need rewriting.",
    "A good template: states the family's quantity and follows the house style, "
    "with only minor member-specific detail a sibling would carry harmlessly.",
    "An ideal template: states the family's quantity, follows the house style "
    "and its fill-in slots exactly, and carries no member-specific detail that "
    "would mislead a sibling.",
)

#: Judgements in flight at once within one family's scoring.
_ANCHOR_SCORE_CONCURRENCY = 8

#: Wall-time ceiling for one family's scoring; ``None`` waits for every call.
_ANCHOR_SCORE_BUDGET_SECONDS = 60.0

#: Decisions service tag, shared with the other standard-names stages.
_ANCHOR_SCORE_SERVICE = "standard-names"

#: Pipeline-cost step these calls are booked to.
_ANCHOR_SCORE_COST_STEP = "sn_anchor_score"


@dataclass
class AnchorScore:
    """One family's members scored as the family's documentation template.

    ``expected`` and ``distribution`` hold only the members a call scored
    successfully, keyed by member id; a member whose call failed or was
    unscored is absent from both rather than read as a zero. ``choice`` is the
    highest expected score, or ``None`` when no member was scored — an empty
    outcome is never replaced by a default anchor. ``asked`` names the members
    one call was made for; ``cost`` is their summed reported spend. ``reason``
    is set only when the family was refused before any call — no eligible
    member, or no member yields a house-style template — so an empty outcome
    can say why rather than reading as a family the judge happened to score
    nothing for.
    """

    pattern: str
    expected: dict[str, float] = field(default_factory=dict)
    distribution: dict[str, dict[str, float]] = field(default_factory=dict)
    choice: str | None = None
    asked: tuple[str, ...] = ()
    cost: float = 0.0
    reason: str | None = None


def template_fitness_question() -> dict[str, dict[str, Any]]:
    """The Score question every eligible member of a family is asked.

    Code builds the question; the model supplies only the judgement. One
    question, asked once per eligible member, so the scores are comparable
    across the family.
    """
    return {
        TEMPLATE_FITNESS_QUESTION: {
            "type": "score",
            "instructions": (
                "How well would this member's documentation serve as the "
                "template for the whole family? Judge the opening and the "
                "structure a sibling's documentation would inherit from it, "
                "against the family's pattern and the house-style template."
            ),
            "criteria": list(TEMPLATE_FITNESS_LEVELS),
        }
    }


def _house_style_template(pattern_name: str) -> str:
    """The house-style openings the grammar renders for this family's pattern.

    Reuses the accessor the docs prompt renders its conventions block through,
    so the template the judge is shown is the one the family is written to —
    never a copy that could drift from the registry.
    """
    from imas_codex.standard_names.context import conventions_context_for

    selected = conventions_context_for(pattern_name) or {}
    lines: list[str] = []
    for entry in selected.get("templates") or []:
        documentation = str(entry.get("documentation") or "").strip()
        note = str(entry.get("note") or "").strip()
        if documentation:
            lines.append(documentation)
        if note and note not in lines:
            lines.append(note)
    return "\n".join(lines)


def _eligible_anchor_candidate(row: Mapping[str, Any]) -> bool:
    """Whether a member is one :func:`select_anchor` would consider.

    The same rule ``select_anchor`` applies to a member — a real
    (non-placeholder) description, entailed by a DD source
    (:func:`member_has_dd_source`) — plus the documentation a template needs: a
    member with no documentation has nothing to template from.
    """
    return (
        member_has_dd_source(row)
        and not is_placeholder_description(row.get("description"))
        and bool(str(row.get("documentation") or "").strip())
    )


def _anchor_score_candidates(
    parent_id: str | None,
    parent_description: str | None,
    parent_documentation: str | None,
    parent_docs_stage: str | None,
    members: Sequence[Mapping[str, Any]],
    *,
    parent_has_dd_source: bool | None = None,
) -> list[dict[str, Any]]:
    """The family's eligible members as uniform rows, parent first.

    The parent, when it names a distinct node, is folded into the same row
    shape as the members, so one eligibility rule and one ``select_anchor``
    ranking serve the whole family. A parent already present in ``members`` is
    not duplicated. A folded parent carries ``parent_has_dd_source``, since the
    caller's query is what read the edge; it is absent unless the caller states
    it, which leaves the parent ineligible rather than assumed sourced.
    """
    member_ids = {m.get("id") for m in members}
    rows: list[dict[str, Any]] = []
    if parent_id and parent_id not in member_ids:
        rows.append(
            {
                "id": parent_id,
                "description": parent_description,
                "documentation": parent_documentation,
                "docs_stage": parent_docs_stage,
                "reviewer_score_docs": None,
                "has_dd_source": parent_has_dd_source is True,
            }
        )
    rows.extend(dict(m) for m in members)
    return [row for row in rows if _eligible_anchor_candidate(row)]


#: Reason recorded when the family has no member a call could be made for.
_NO_ELIGIBLE_REASON = "no member is eligible as the family's template"

#: Reason recorded when no family member yields a house-style template. The
#: family is refused rather than asked, because a template-less state asks the
#: judge to compare openings against nothing.
_NO_TEMPLATE_REASON = (
    "no family member yields a house-style template for the family's pattern"
)


def _resolve_house_style_template(
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    """The house-style template any member of the family yields, or ``""``.

    Every member shares the family's pattern, so one member's conventions are
    the family's. The rows are tried in order (the parent leads them) and the
    first id the grammar can parse wins: an id it cannot parse — a legacy or
    unresolvable name — returns no conventions, and reading the template from
    that row alone would silently leave the family's state template empty.
    """
    for row in candidates:
        member_id = str(row.get("id") or "")
        if member_id:
            template = _house_style_template(member_id)
            if template:
                return template
    return ""


async def score_family_anchor(
    parent_id: str | None,
    parent_docs_stage: str | None,
    parent_description: str | None,
    members: Sequence[Mapping[str, Any]],
    *,
    parent_documentation: str | None = None,
    parent_has_dd_source: bool | None = None,
    pattern: str = "",
    model: str | None = None,
    concurrency: int = _ANCHOR_SCORE_CONCURRENCY,
    budget_seconds: float | None = _ANCHOR_SCORE_BUDGET_SECONDS,
    cost: Any | None = None,
) -> AnchorScore:
    """Score each eligible family member as the family's documentation template.

    For one family, one Jev Score per eligible member answers how well that
    member's documentation would serve as the template for the whole family,
    given the member's name, description and documentation, the family's
    pattern, and the house-style template the grammar renders for that pattern.
    The template is read from the family's own members — the first whose id the
    grammar parses — because every member shares the family's pattern; a family
    whose members all fail to parse yields no template and is refused with a
    reason rather than asked without one. The member with the highest expected
    score is the candidate anchor; a tie falls to :func:`select_anchor`'s own
    ranking over the tied members, with the family's parent passed through when
    it is tied too, so the accepted-parent rule the heuristic applies to the
    whole family applies here as well and the judged choice never contradicts
    the heuristic on equal evidence.

    This is a scorer, not a caller of the member rows' anchor: nothing here
    changes a recorded anchor or a caller's behaviour. A failed or unscored
    call leaves its member unscored, and a family with no scored member returns
    no choice rather than a default.

    Args:
        parent_id: The family's base-quantity node, or ``None`` when the family
            is grouped by pattern with no parent node.
        parent_docs_stage: The parent's ``docs_stage``, read by the tie-break.
        parent_description: The parent's description, read by the tie-break.
        members: The family's member rows. Each carries ``id``, ``description``
            and ``documentation``; ``docs_stage`` and ``reviewer_score_docs``
            feed the tie-break where present.
        parent_documentation: The parent's documentation, so a parent not in
            ``members`` can still be scored as the family's template.
        pattern: The family's pattern — its invariant segment signature —
            carried in the state for the judge. The template accessor is called
            with the family's member ids, not with the pattern.
        model: Decisions model id; defaults to the ``sn-judge`` seat.
        concurrency: Maximum judgements in flight at once.
        budget_seconds: Wall-time ceiling for the family; ``None`` waits.
        cost: Optional run cost tracker; the batch's summed cost is booked to
            it under the scorer's step.

    Returns:
        An :class:`AnchorScore` naming the scored members, their expected
        scores and distributions, the chosen member (or ``None``), the members
        one call was made for, and the summed reported spend. A family refused
        before any call carries a ``reason`` and no choice.
    """
    candidates = _anchor_score_candidates(
        parent_id,
        parent_description,
        parent_documentation,
        parent_docs_stage,
        members,
        parent_has_dd_source=parent_has_dd_source,
    )
    asked = tuple(str(row["id"]) for row in candidates if row.get("id"))
    if not candidates:
        return AnchorScore(pattern=pattern, reason=_NO_ELIGIBLE_REASON)

    if model is None:
        from imas_codex.settings import get_model

        model = get_model("sn-judge")

    template = _resolve_house_style_template(candidates)
    if not template:
        return AnchorScore(pattern=pattern, reason=_NO_TEMPLATE_REASON)

    states = [
        {
            "member": {
                "name": row["id"],
                "description": str(row.get("description") or ""),
                "documentation": str(row.get("documentation") or ""),
            },
            "family_pattern": pattern,
            "house_style_template": template,
        }
        for row in candidates
    ]

    from imas_codex.discovery.base.judgment import decide_batch

    results, batch_cost = await decide_batch(
        states,
        template_fitness_question(),
        model=model,
        service=_ANCHOR_SCORE_SERVICE,
        concurrency=concurrency,
        budget_seconds=budget_seconds,
        cost=cost,
        step=_ANCHOR_SCORE_COST_STEP,
    )

    expected: dict[str, float] = {}
    distribution: dict[str, dict[str, float]] = {}
    for row, result in zip(candidates, results, strict=True):
        if result is None:
            continue
        answer = result[0].get(TEMPLATE_FITNESS_QUESTION) or {}
        score = answer.get("score")
        if score is None:
            continue
        member_id = str(row["id"])
        expected[member_id] = float(score)
        distribution[member_id] = dict(answer.get("probabilities") or {})

    choice: str | None = None
    if expected:
        top = max(expected.values())
        tied = [row for row in candidates if expected.get(str(row["id"])) == top]
        # Ties fall to select_anchor's own ranking over the tied members, so the
        # judged choice never contradicts the heuristic on equal evidence. The
        # family's parent is passed through when it is itself one of the tied
        # members, so its accepted-parent rule fires exactly as it would over
        # the whole family; when it is not tied, the ranking runs without it and
        # can never name a member the judge did not score.
        if parent_id and parent_id in {str(row["id"]) for row in tied}:
            choice = select_anchor(
                parent_id, parent_docs_stage, parent_description, tied
            )
        else:
            choice = select_anchor(None, None, None, tied)

    return AnchorScore(
        pattern=pattern,
        expected=expected,
        distribution=distribution,
        choice=choice,
        asked=asked,
        cost=batch_cost,
    )


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
       {dd_source_exists("m")} AS has_dd_source,
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
    return (
        """collect({
        id: m.id,
        description: m.description,
        documentation: m.documentation,
        docs_stage: m.docs_stage,
        reviewer_score_docs: m.reviewer_score_docs,
        has_dd_source: """
        + dd_source_exists("m")
        + """,
        harmonized_group_signature: m.harmonized_group_signature,
        harmonized_anchor: m.harmonized_anchor,
        unit: m.unit,
        cocos_transformation_type: m.cocos_transformation_type,
        physics_domain: m.physics_domain,
        position: m.position,
        links: m.links,
        link_status: m.link_status
    }) AS members"""
    )


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
_FAMILY_BASE_QUERY = f"""
MATCH (b:StandardName {{id: $base_id}})
WHERE coalesce(b.name_stage, '') <> $superseded
RETURN b.id AS id,
       b.description AS description,
       b.documentation AS documentation,
       coalesce(b.docs_stage, '') AS docs_stage,
       b.reviewer_score_docs AS reviewer_score_docs,
       {dd_source_exists("b")} AS has_dd_source
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

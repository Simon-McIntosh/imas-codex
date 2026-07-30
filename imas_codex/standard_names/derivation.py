"""Structural edge derivation for StandardName nodes.

Pure logic module — no graph access, no I/O.  Given a single
StandardName id string, ``derive_edges`` peels the outermost ISN
grammar operator/projection and returns the corresponding
``HAS_PARENT``, ``HAS_ERROR``, or ``HAS_LOCUS`` edge descriptor.

Recursion is structural: when the inner StandardName is itself written
to the graph, *its* derivation runs and emits *its* own edge.  We never
peel more than one layer here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from imas_standard_names import StandardNameIR, compose, parse

from imas_codex.standard_names.grammar_adapter import (
    compose_canonical_ir,
    parse_canonical_name,
)

logger = logging.getLogger(__name__)

# Uncertainty prefix operators → HAS_ERROR error_type label
_UNCERTAINTY_OPS: dict[str, str] = {
    "upper_uncertainty": "upper",
    "lower_uncertainty": "lower",
    "uncertainty_index": "index",
}


@dataclass(frozen=True)
class DerivedEdge:
    """A single derived structural edge between two StandardName ids."""

    edge_type: str  # "HAS_PARENT", "HAS_ERROR", or "HAS_LOCUS"
    from_name: str  # source StandardName id
    to_name: str  # target StandardName id
    props: dict  # edge properties (operator, operator_kind, …)


def _strip_outer(ir: StandardNameIR) -> StandardNameIR:
    """Return *ir* with the outermost operator removed.

    All other fields (projection, qualifiers, base, locus, mechanism)
    are preserved unchanged.
    """
    return ir.model_copy(update={"operators": ir.operators[1:]})


def _strip_projection(ir: StandardNameIR) -> StandardNameIR:
    """Return *ir* with the projection cleared.

    All other fields (operators, qualifiers, base, locus, mechanism)
    are preserved unchanged.
    """
    return ir.model_copy(update={"projection": None})


def _parse_structural_ir(name: str) -> StandardNameIR:
    """Parse exact structure, tolerating legacy names only for graph inspection.

    Strict parsing remains the validity authority. Existing graph identities
    can predate a semantic gate (for example an unqualified generic base), so
    edge derivation falls back to the public non-strict parser while still
    requiring exact lossless compose equality. This fallback never admits,
    persists, or validates a candidate.
    """
    try:
        return parse_canonical_name(name).ir
    except ValueError:
        result = parse(name)
        if compose(result.ir) != name:
            raise ValueError(f"legacy structural parse changed {name!r}") from None
        return result.ir


def _compose_structural_ir(ir: StandardNameIR) -> str:
    """Compose an edge endpoint with strict-first legacy structural fallback."""
    try:
        return compose_canonical_ir(ir)
    except ValueError:
        name = compose(ir)
        if compose(parse(name).ir) != name:
            raise ValueError(f"legacy structural compose changed {name!r}") from None
        return name


def derive_edges(name: str) -> list[DerivedEdge]:
    """Return derived structural edges for a single StandardName id.

    Pure function.  The ISN parser is the sole source of structural truth.
    Names the parser cannot parse produce no edges (leaf treatment).

    Edge types produced:

    - ``HAS_PARENT``: projection / operator / coordinate decomposition.
    - ``HAS_ERROR``: uncertainty siblings.
    - ``HAS_LOCUS``: locus grouping — names sharing the same ISN
      locus token (e.g. ``magnetic_axis``, ``plasma_boundary``).

    Parameters
    ----------
    name:
        StandardName id string (e.g. ``"maximum_of_temperature"``).

    Returns
    -------
    list[DerivedEdge]
        Zero or more edges depending on the outermost IR shape and
        locus qualification.  Returns ``[]`` for unparseable names
        and leaf names with no locus qualifier.

    Self-loop guard
    ---------------
    Edges where ``from_name == to_name`` are dropped unconditionally.
    Such an edge can never be structurally meaningful — a name cannot
    be its own argument / error / locus — and used to surface only as
    a downstream symptom: ISNC `validate_catalog` raises
    `graphlib.CycleError: nodes are in a cycle, ['x', 'x']` when the
    catalog builder topologically sorts ``HAS_ARGUMENT`` edges, and
    that crash is the only place we noticed. The defect is the ISN
    parser/composer pair occasionally round-tripping to the same name
    on a postfix operator (e.g. ``minimum_magnetic_field_magnitude``
    → strip ``magnitude`` → compose → ``minimum_magnetic_field_magnitude``).
    We catch it here so it cannot reach the graph regardless of which
    upstream parser version is in use.
    """
    try:
        ir = _parse_structural_ir(name)
    except Exception:
        return []

    structural = _derive_structural(name, ir)
    locus = _locus_check(name, ir)
    return _drop_self_loops(name, structural + locus)


def _drop_self_loops(name: str, edges: list[DerivedEdge]) -> list[DerivedEdge]:
    """Filter edges whose endpoint coincides with the source name."""
    out: list[DerivedEdge] = []
    for e in edges:
        if e.from_name == e.to_name:
            logger.debug(
                "derive_edges dropping self-loop %s edge for %r",
                e.edge_type,
                name,
            )
            continue
        out.append(e)
    return out


def _derive_structural(name: str, ir: StandardNameIR) -> list[DerivedEdge]:
    """Derive HAS_PARENT / HAS_ERROR edges from the IR parse tree."""

    # --- Outermost operator ---
    if ir.operators:
        op = ir.operators[0]

        if op.kind.value == "binary":
            # Binary: two HAS_PARENT edges, one per argument
            try:
                a = _compose_structural_ir(op.args[0])
                b = _compose_structural_ir(op.args[1])
            except Exception as exc:
                logger.debug("derive_edges compose failed for %r: %s", name, exc)
                return []
            return [
                DerivedEdge(
                    "HAS_PARENT",
                    name,
                    a,
                    {
                        "operator": op.op,
                        "operator_kind": "binary",
                        "role": "a",
                        "separator": op.separator,
                    },
                ),
                DerivedEdge(
                    "HAS_PARENT",
                    name,
                    b,
                    {
                        "operator": op.op,
                        "operator_kind": "binary",
                        "role": "b",
                        "separator": op.separator,
                    },
                ),
            ]

        # Unary prefix / postfix: strip outer and compose inner
        stripped = _strip_outer(ir)
        try:
            inner = _compose_structural_ir(stripped)
        except Exception as exc:
            logger.debug("derive_edges compose failed for %r: %s", name, exc)
            return []

        if op.kind.value == "unary_prefix" and op.op in _UNCERTAINTY_OPS:
            # Uncertainty prefix — direction inverts: inner → name
            return [
                DerivedEdge(
                    "HAS_ERROR",
                    inner,
                    name,
                    {"error_type": _UNCERTAINTY_OPS[op.op]},
                )
            ]

        return [
            DerivedEdge(
                "HAS_PARENT",
                name,
                inner,
                {
                    "operator": op.op,
                    "operator_kind": op.kind.value,
                },
            )
        ]

    # --- Projection (component) without operator ---
    if ir.projection is not None:
        stripped = _strip_projection(ir)
        try:
            inner = _compose_structural_ir(stripped)
        except Exception as exc:
            logger.debug("derive_edges compose failed for %r: %s", name, exc)
            return []

        # Round-trip guard: verify that the inner name is a genuine
        # standalone name, not a compound fragment.  ISN v0.8.0rc1
        # projection branch can produce nonsensical inner names from
        # compound qualifier structures.
        #
        # We check two things:
        # 1. The inner name must round-trip through the IR parser.
        # 2. The inner name must be a simple base (no locus qualifier),
        #    because a locus like "of_measurement_position" attached to a
        #    geometric base produces edges to generic parents that group
        #    unrelated names together.
        try:
            inner_ir = _parse_structural_ir(inner)
            # Reject if inner has a locus (qualifier) — the edge would
            # point to a compound parent like "coordinate_of_measurement_position"
            if inner_ir.locus is not None:
                logger.debug(
                    "derive_edges rejecting projection edge for %r: "
                    "inner=%r has locus qualifier",
                    name,
                    inner,
                )
                return []
        except Exception:
            # Inner name doesn't round-trip — spurious projection
            logger.debug(
                "derive_edges projection inner parse failed for %r: inner=%r",
                name,
                inner,
            )
            return []

        return [
            DerivedEdge(
                "HAS_PARENT",
                name,
                inner,
                {
                    "operator": "component",
                    "operator_kind": "projection",
                    "axis": ir.projection.axis,
                    "shape": ir.projection.shape.value,
                },
            )
        ]

    # --- Qualifier layer ---
    # Peel the outermost qualifier and link to the residual. This is the
    # natural structural parent for shape-asymmetry / role qualifiers that
    # ISN does not model as operators: e.g.
    #     upper_elongation_of_plasma_boundary
    #         → HAS_PARENT → elongation_of_plasma_boundary  (qualifier=upper)
    #         and then that name's own derivation peels the locus →
    #     elongation_of_plasma_boundary
    #         → HAS_PARENT → elongation                     (locus=plasma_boundary)
    # The recursion is structural — we only ever peel ONE layer per call.
    # Without this layer the parent shortcut in dataset.py jumped straight
    # to `ir.base.token` (`elongation`), losing the boundary locus and
    # grouping upper/lower boundary elongation with unrelated flux-surface
    # elongation under a generic root.
    if ir.qualifiers:
        stripped = ir.model_copy(update={"qualifiers": ir.qualifiers[1:]})
        try:
            inner = _compose_structural_ir(stripped)
        except Exception as exc:
            logger.debug(
                "derive_edges qualifier-peel compose failed for %r: %s", name, exc
            )
            return []
        return [
            DerivedEdge(
                "HAS_PARENT",
                name,
                inner,
                {
                    "operator": ir.qualifiers[0].token,
                    "operator_kind": "qualifier",
                },
            )
        ]

    # --- Locus layer ---
    # No qualifier, no operator, no projection — but the name carries a
    # locus suffix (``_of_<locus>``, ``_at_<locus>``). Peel the locus and
    # link to the residual base. This is what builds the
    #     elongation_of_plasma_boundary → elongation
    # leg of the chain. The locus relation (``of`` vs ``at``) is not
    # carried on the edge — keeping the edge schema minimal lets ISN's
    # ``ArgumentRef`` accept ``operator_kind="locus"`` without a per-
    # relation field; the relation is recoverable from the source
    # name's parse if anything ever needs it.
    if ir.locus is not None:
        stripped = ir.model_copy(update={"locus": None})
        try:
            inner = _compose_structural_ir(stripped)
        except Exception as exc:
            logger.debug("derive_edges locus-peel compose failed for %r: %s", name, exc)
            return []
        return [
            DerivedEdge(
                "HAS_PARENT",
                name,
                inner,
                {
                    "operator": ir.locus.token,
                    "operator_kind": "locus",
                },
            )
        ]

    # Leaf: no operator, no projection, no qualifier, no locus
    # — try geometric coordinate as a last resort.
    return _geometric_coordinate_check(name)


def _locus_check(name: str, ir: StandardNameIR | None = None) -> list[DerivedEdge]:
    """Detect locus-qualified names and emit HAS_LOCUS grouping edges.

    Uses the structured IR parser to extract locus information from names
    like ``major_radius_of_magnetic_axis`` (locus=magnetic_axis, relation=of)
    or ``safety_factor_at_normalized_poloidal_flux`` (locus=normalized_poloidal_flux,
    relation=at).

    Names sharing the same locus token form a **locus family** — they
    describe different physical quantities measured at the same location.
    The ``HAS_LOCUS`` edge groups them by linking each name to a shared
    ``Locus`` node (e.g. ``magnetic_axis``, ``plasma_boundary``).

    Returns ``[]`` when the name has no locus qualifier.
    """
    if ir is None:
        try:
            ir = _parse_structural_ir(name)
        except Exception:
            return []

    locus = ir.locus
    if locus is None or not locus.token:
        return []

    return [
        DerivedEdge(
            "HAS_LOCUS",
            name,
            locus.token,
            {
                "locus_token": locus.token,
                "locus_relation": str(locus.relation.value),
            },
        )
    ]


def _geometric_coordinate_check(name: str) -> list[DerivedEdge]:
    """Detect geometric coordinate names via the model-level ISN parser.

    ISN grammar uses strict short form for axis projections:

    * **Physical vector**: ``{axis}_{base}`` (e.g. ``toroidal_magnetic_field``)
      → ``component`` slot populated, handled by the projection branch.
    * **Geometric coordinate**: ``{axis}_{geometric_base}`` →
      ``coordinate`` slot populated (e.g. ``radial_position``).

    The IR parser often cannot parse geometric coordinates
    (``radial_position`` raises ``ParseError``), but the model-level
    ``parse_standard_name`` succeeds.  This function uses the model
    parser as a last-resort check and emits a ``HAS_PARENT`` edge
    with ``operator_kind="coordinate"`` when a coordinate slot is found.

    Returns ``[]`` when the name is not a geometric coordinate (true leaf).
    """
    try:
        from imas_standard_names.grammar import parse_standard_name

        parsed = parse_standard_name(name)
    except Exception:
        return []

    if parsed is None or parsed.coordinate is None:
        return []

    # Determine the parent (inner) name
    inner_name: str | None = None
    if parsed.geometric_base is not None:
        inner_name = (
            str(parsed.geometric_base.value)
            if hasattr(parsed.geometric_base, "value")
            else str(parsed.geometric_base)
        )
    elif parsed.physical_base is not None:
        # Edge case: toroidal_angle has physical_base='angle', no geometric_base.
        # Guard: reject compound physical_base containing '_of_' — this signals
        # the parser failed to decompose a compound (e.g. a removed locus token
        # absorbed into physical_base).
        pb = str(parsed.physical_base)
        if "_of_" in pb:
            return []
        inner_name = pb

    if not inner_name:
        return []

    axis_value = (
        str(parsed.coordinate.value)
        if hasattr(parsed.coordinate, "value")
        else str(parsed.coordinate)
    )

    # Round-trip validation: only create edges for simple forms where the
    # name is exactly "{axis}_{inner_name}".  For compound names like
    # "vertical_coordinate_of_first_point_of_line_of_sight", ISN's
    # geometric_base captures only the base token ("coordinate"), losing
    # the qualifier.  Creating an edge to a generic parent would group
    # unrelated coordinates together.
    expected = f"{axis_value}_{inner_name}"
    if name != expected:
        return []

    return [
        DerivedEdge(
            "HAS_PARENT",
            name,
            inner_name,
            {
                "operator": "coordinate",
                "operator_kind": "coordinate",
                "axis": axis_value,
                "shape": "coordinate",
            },
        )
    ]

"""Admission gate for structural-parent StandardName nodes.

Pure logic + topology queries.  Decides whether a derived-parent name
proposed by ``derive_edges`` deserves to exist in the graph as a
StandardName placeholder, or whether the inbound ``HAS_PARENT`` edge
should be dropped entirely.

Two-clause admission (see plan D1):

- **Clause A — structural specificity.** The candidate's ISN IR carries
  at least one of: non-empty qualifiers, projection, locus, non-empty
  operators, or a mechanism.  Bare-base names (``pressure``, ``density``,
  …) fail this clause.

- **Clause B — vector-like topology.** The candidate already has
  ``HAS_PARENT`` children with ``operator_kind='projection'`` along ≥2
  distinct axes.  Catches true vector parents (``magnetic_field``,
  ``electric_field``) whose name strings are bare bases but whose
  algebraic content makes them first-class SNs.

Admit if either clause holds; reject otherwise.

The gate is **callable without a real graph** for Clause-A testing.
Clause B requires a topology lookup, dispatched through a small
protocol so tests can substitute a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from imas_standard_names.grammar import parser as _isn_parser
from imas_standard_names.grammar.parser import ParseError as _ParseError


@dataclass(frozen=True)
class AdmissionResult:
    """Outcome of an admission-gate evaluation.

    Attributes
    ----------
    admit:
        ``True`` if the candidate parent passes the gate.
    reason:
        Human-readable explanation (used for audit logs and CLI output).
    clause:
        ``"A"`` (structural specificity), ``"B"`` (vector-like
        topology), or ``None`` (rejected — neither clause).
    """

    admit: bool
    reason: str
    clause: str | None  # "A" | "B" | None


class _TopologyProbe(Protocol):
    """Minimal interface used by Clause B.

    Concrete production callers pass a ``GraphClient`` (which has a
    ``query`` method); tests can pass a stub returning canned rows.
    """

    def query(self, cypher: str, **params): ...  # pragma: no cover


def _has_structural_specificity(name: str) -> tuple[bool, str]:
    """Clause A — does the IR carry anything beyond the bare base?

    Returns ``(admit, reason)``.  Names the ISN parser cannot parse
    are rejected (no specificity claim possible).
    """
    try:
        parsed = _isn_parser.parse(name)
        ir = parsed.ir
    except _ParseError as exc:
        return False, f"ISN parse failed: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"ISN parse failed: {exc.__class__.__name__}: {exc}"

    if ir.qualifiers:
        tokens = ",".join(q.token for q in ir.qualifiers)
        return True, f"has qualifiers [{tokens}]"
    if ir.operators:
        ops = ",".join(getattr(o, "op", str(o)) for o in ir.operators)
        return True, f"has operators [{ops}]"
    if ir.projection is not None:
        axis = getattr(ir.projection, "axis", "?")
        return True, f"has projection (axis={axis})"
    if ir.locus is not None:
        tok = getattr(ir.locus, "token", "?")
        return True, f"has locus ({tok})"
    if ir.mechanism is not None:
        tok = getattr(ir.mechanism, "token", "?")
        return True, f"has mechanism ({tok})"
    return False, "bare base — no qualifier, locus, projection, operator, or mechanism"


def _has_vector_like_topology(name: str, gc: _TopologyProbe) -> tuple[bool, str]:
    """Clause B — does the candidate already have multi-axis projection children?

    A parent is "vector-like" when ≥2 distinct-axis projection children
    point at it via HAS_PARENT.  This catches ``magnetic_field`` from
    ``radial_magnetic_field`` + ``toroidal_magnetic_field`` even though
    the name string itself is a pure base.

    Returns ``(admit, reason)``.
    """
    cypher = """
        MATCH (child:StandardName)-[r:HAS_PARENT]->(p:StandardName {id: $name})
        WHERE r.operator_kind = 'projection' AND r.axis IS NOT NULL
        RETURN collect(DISTINCT r.axis) AS axes
    """
    try:
        rows = list(gc.query(cypher, name=name))
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"topology query failed: {exc.__class__.__name__}: {exc}"

    if not rows:
        return False, "no projection children"

    axes = rows[0].get("axes") or []
    if len(axes) >= 2:
        return (
            True,
            f"vector-like ({len(axes)} distinct projection axes: {sorted(axes)})",
        )
    if len(axes) == 1:
        return False, f"only one projection axis ({axes[0]}) — not multi-axis"
    return False, "no projection children"


def is_admissible_parent_name(
    name: str, gc: _TopologyProbe | None = None
) -> AdmissionResult:
    """Decide whether *name* deserves a StandardName placeholder + HAS_PARENT edge.

    Parameters
    ----------
    name:
        Candidate parent StandardName id (typically the ``to_name`` of a
        ``DerivedEdge`` from ``derive_edges``).
    gc:
        Graph client used for Clause-B topology lookup.  When ``None``,
        only Clause A is evaluated; useful for pure unit tests.

    Returns
    -------
    AdmissionResult
        Decision plus reason and which clause matched.
    """
    admit_a, reason_a = _has_structural_specificity(name)
    if admit_a:
        return AdmissionResult(admit=True, reason=reason_a, clause="A")

    if gc is None:
        return AdmissionResult(
            admit=False, reason=reason_a + " (no graph probe for clause B)", clause=None
        )

    admit_b, reason_b = _has_vector_like_topology(name, gc)
    if admit_b:
        return AdmissionResult(admit=True, reason=reason_b, clause="B")

    return AdmissionResult(
        admit=False,
        reason=f"clause A: {reason_a}; clause B: {reason_b}",
        clause=None,
    )


def recompute_parent_kind(name: str, gc: _TopologyProbe) -> str:
    """Topology-driven kind for an admitted parent (Phase 2 helper).

    Returns the canonical kind based on the parent's HAS_PARENT children:

    - ``vector`` if ≥2 distinct-axis projection children exist
    - ``tensor`` if the name contains ``_tensor``
    - ``eigenfunction`` if the name contains ``_eigenfunction``
    - ``spectrum`` if the name ends with ``_spectrum``
    - ``complex`` if the name contains ``real_part`` / ``imaginary_part``
    - ``scalar`` otherwise

    Topology beats the string pattern when both fire (a name like
    ``foo_spectrum`` with multi-axis projections still returns
    ``vector`` — projection children are the stronger signal).
    """
    cypher = """
        MATCH (child:StandardName)-[r:HAS_PARENT]->(p:StandardName {id: $name})
        WHERE r.operator_kind = 'projection' AND r.axis IS NOT NULL
        RETURN count(DISTINCT r.axis) AS n
    """
    try:
        n_axes = list(gc.query(cypher, name=name))[0]["n"]
    except (IndexError, KeyError, Exception):  # pragma: no cover - defensive
        n_axes = 0

    if n_axes >= 2:
        return "vector"
    if "_tensor" in name:
        return "tensor"
    if "_eigenfunction" in name:
        return "eigenfunction"
    if name.endswith("_spectrum"):
        return "spectrum"
    if "real_part" in name or "imaginary_part" in name:
        return "complex"
    return "scalar"

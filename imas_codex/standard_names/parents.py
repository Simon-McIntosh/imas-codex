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


def is_single_child_shadow(name: str, gc: _TopologyProbe) -> tuple[bool, str]:
    """Suppression veto — is *name* just a less-specific shadow of one child?

    A derived parent must earn its existence by generalising over **multiple**
    specific names.  When a candidate parent has exactly one ``HAS_PARENT``
    child, and that child is a live pipeline-origin name sourced from a DD
    path (via ``HAS_STANDARD_NAME``), the parent contributes no grouping
    value: it is merely a less-specific spelling of that single accepted
    sibling sourced from the same path (e.g. ``radius_of_divertor_target``
    shadowing ``major_radius_of_divertor_target`` from
    ``divertors/divertor/target/tile/surface_outline/r``).  Materialising it
    produces a second accepted name competing for the same source — the
    Class-B duplicate.

    The veto fires only when **all** of the following hold:

    - the candidate has exactly **one** distinct ``HAS_PARENT`` child;
    - that child is sourced from at least one DD path the candidate does not
      independently own (the parent has no DD source of its own that differs
      from the child's);
    - the child is a live, non-superseded name (``name_stage`` not in
      ``{superseded, exhausted}``) of pipeline / catalog origin (i.e. it is a
      real specific name, not itself a derived placeholder).

    Genuine shared parents survive untouched: ``temperature`` parenting both
    ``electron_temperature`` and ``ion_temperature`` has ≥2 distinct children,
    so the single-child condition fails and the veto does not fire.

    Returns ``(suppress, reason)``.
    """
    cypher = """
        MATCH (child:StandardName)-[:HAS_PARENT]->(p:StandardName {id: $name})
        WITH p, collect(DISTINCT child) AS children
        WHERE size(children) = 1
        WITH p, children[0] AS child
        WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted']
          AND coalesce(child.origin, 'pipeline') <> 'derived'
        OPTIONAL MATCH (csrc:IMASNode)-[:HAS_STANDARD_NAME]->(child)
        OPTIONAL MATCH (psrc:IMASNode)-[:HAS_STANDARD_NAME]->(p)
        WITH child,
             collect(DISTINCT csrc.id) AS child_sources,
             collect(DISTINCT psrc.id) AS parent_sources
        RETURN child.id AS child_id, child_sources, parent_sources
    """
    try:
        rows = list(gc.query(cypher, name=name))
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"shadow probe failed: {exc.__class__.__name__}: {exc}"

    if not rows:
        return False, "not a single-child parent"

    row = rows[0]
    child_id = row.get("child_id")
    child_sources = set(row.get("child_sources") or [])
    parent_sources = set(row.get("parent_sources") or [])

    if not child_id:
        return False, "single child but missing id"
    if not child_sources:
        # The lone child carries no DD source — cannot claim source-equivalence.
        return False, "single child has no DD source"
    # If the parent independently owns a DD source the child does not, it is a
    # real sourced name in its own right, not a pure shadow — keep it.
    if parent_sources - child_sources:
        return False, "parent independently sourced — not a shadow"

    return (
        True,
        f"single-child shadow of {child_id} "
        f"(shared DD source: {sorted(child_sources)[:2]})",
    )


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
        Graph client used for Clause-B topology lookup and the single-child
        shadow veto.  When ``None``, only Clause A is evaluated; useful for
        pure unit tests.

    Returns
    -------
    AdmissionResult
        Decision plus reason and which clause matched.
    """
    admit_a, reason_a = _has_structural_specificity(name)

    # Suppression veto (Class-B): even a structurally-specific candidate is
    # rejected when it is merely a less-specific shadow of a single accepted
    # sibling sourced from the same DD path. Requires a graph probe; skipped
    # in pure-logic (gc is None) callers.
    if gc is not None:
        suppress, suppress_reason = is_single_child_shadow(name, gc)
        if suppress:
            return AdmissionResult(
                admit=False,
                reason=f"suppressed: {suppress_reason}",
                clause=None,
            )

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

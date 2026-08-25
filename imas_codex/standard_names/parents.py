"""Admission gate for structural-parent StandardName nodes.

Pure logic + topology queries.  Decides whether a derived-parent name
proposed by ``derive_edges`` deserves to exist in the graph as a
StandardName placeholder, or whether the inbound ``HAS_PARENT`` edge
should be dropped entirely.

Two-clause admission:

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
from typing import Any, Protocol

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.standard_names.grammar_adapter import (
    compose_canonical_ir,
    parse_canonical_name,
)


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


def _has_valid_standard_name_identity(name: str) -> tuple[bool, str]:
    """Validate *name* through ISN's public semantic model."""
    try:
        parse_canonical_name(name)
    except Exception as exc:
        return False, f"ISN identity invalid: {exc}"
    return True, "valid ISN identity"


def is_algebraic_decomposition_edge(child_name: str, parent_name: str) -> bool:
    """Return whether strict public IR requires ``child_name → parent_name``.

    Operator and projection applications are lossless expression-tree nodes,
    even when only one source-backed name currently uses them.  They must not
    be mistaken for semantic shadows merely because their graph fan-in is one.
    Qualifier and locus peels deliberately do not qualify here.
    """
    try:
        ir = parse_canonical_name(child_name).ir
        if ir.operators:
            outer = ir.operators[0]
            if len(outer.args) == 2:
                candidates = outer.args
            else:
                candidates = [ir.model_copy(update={"operators": ir.operators[1:]})]
        elif ir.projection is not None:
            candidates = [ir.model_copy(update={"projection": None})]
        else:
            return False
        return any(
            compose_canonical_ir(candidate) == parent_name for candidate in candidates
        )
    except (TypeError, ValueError):
        return False


def _has_algebraic_decomposition_child(
    name: str, gc: _TopologyProbe
) -> tuple[bool, str]:
    """Return whether graph topology proves *name* is a strict algebraic node."""
    try:
        rows = list(
            gc.query(
                """
                MATCH (child:StandardName)-[:HAS_PARENT]->
                      (:StandardName {id: $name})
                RETURN collect(DISTINCT child.id) AS child_ids
                """,
                name=name,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"algebraic child probe failed: {exc.__class__.__name__}: {exc}"

    child_ids = {
        child_id
        for row in rows
        for child_id in (
            list(row.get("child_ids") or [])
            + ([row["child_id"]] if row.get("child_id") else [])
        )
    }
    for child_id in child_ids:
        if is_algebraic_decomposition_edge(child_id, name):
            return True, f"required algebraic decomposition of {child_id}"
    return False, "no strict algebraic decomposition child"


def _has_structural_specificity(name: str) -> tuple[bool, str]:
    """Clause A — does the IR carry anything beyond the bare base?

    Returns ``(admit, reason)``.  Names the ISN parser cannot parse
    are rejected (no specificity claim possible).
    """
    try:
        ir = parse_canonical_name(name).ir
    except ValueError as exc:
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
    if is_algebraic_decomposition_edge(child_id, name):
        return False, "single child requires this algebraic decomposition target"
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
    valid_identity, identity_reason = _has_valid_standard_name_identity(name)
    if not valid_identity:
        return AdmissionResult(
            admit=False,
            reason=identity_reason,
            clause=None,
        )

    admit_a, reason_a = _has_structural_specificity(name)

    # A strict, lossless operator/projection peel is itself the structural
    # reason for this target to exist. This also covers canonical bare leaves,
    # whose parent identity carries no standalone qualifier or projection.
    if gc is not None:
        algebraic, algebraic_reason = _has_algebraic_decomposition_child(name, gc)
        if algebraic:
            return AdmissionResult(
                admit=True,
                reason=algebraic_reason,
                clause="A",
            )

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
    """Topology-driven kind for an admitted parent.

    Returns the canonical kind based on the parent's HAS_PARENT children:

    - ``vector`` if ≥2 distinct-axis projection children exist
    - ``tensor`` if the name contains ``_tensor``
    - ``complex`` if the name contains ``real_part`` / ``imaginary_part``
    - ``scalar`` otherwise (including eigenfunction / spectrum names —
      semantic categories, structurally scalar; kind mirrors the ISN
      catalog Kind enum exactly)

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
    if "real_part" in name or "imaginary_part" in name:
        return "complex"
    return "scalar"


def _replay_described_parent_authorities(gc: _TopologyProbe) -> dict[str, Any]:
    """Authorize accepted, described parents grounded by accepted children."""
    from imas_codex.standard_names import graph_ops
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    candidates = list(
        gc.query(
            """
            MATCH (parent:StandardName)
            WHERE parent.name_stage = 'accepted'
              AND parent.origin = 'derived'
              AND parent.validation_status = 'valid'
              AND parent.reviewer_score_name IS NULL
              AND parent.description IS NOT NULL
              AND parent.description <> $placeholder
              AND NOT (parent)-[:HAS_STRUCTURAL_AUTHORITY]->(
                :StructuralNameAuthority)
            RETURN parent.id AS id
            ORDER BY parent.id
            """,
            placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
    )
    replayed: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for candidate in candidates:
        parent_id = str(candidate["id"])
        locked = list(
            gc.query(
                """
                MATCH (parent:StandardName {id: $parent_id})
                SET parent._structural_authority_replay_lock = true
                REMOVE parent._structural_authority_replay_lock
                WITH parent
                WHERE parent.name_stage = 'accepted'
                  AND parent.origin = 'derived'
                  AND parent.validation_status = 'valid'
                  AND parent.reviewer_score_name IS NULL
                  AND parent.description IS NOT NULL
                  AND parent.description <> $placeholder
                  AND NOT (parent)-[:HAS_STRUCTURAL_AUTHORITY]->(
                    :StructuralNameAuthority)
                RETURN count(parent) AS locked
                """,
                parent_id=parent_id,
                placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
            )
        )
        if not locked or int(locked[0].get("locked") or 0) != 1:
            refused.append(
                {
                    "parent_id": parent_id,
                    "reason": "eligibility changed before authority replay",
                    "live_child_ids": [],
                }
            )
            continue

        snapshot = graph_ops._structural_authority_snapshot(gc, parent_id)
        if snapshot is None:
            refused.append(
                {
                    "parent_id": parent_id,
                    "reason": "parent disappeared before authority replay",
                    "live_child_ids": [],
                }
            )
            continue
        children = list(snapshot.get("children") or [])
        live_child_ids = [str(child["id"]) for child in children]
        accepted_child_ids = [
            str(child["id"])
            for child in children
            if child.get("name_stage") == "accepted"
        ]
        if not accepted_child_ids:
            refused.append(
                {
                    "parent_id": parent_id,
                    "reason": "no accepted children",
                    "live_child_ids": live_child_ids,
                }
            )
            continue

        accepted_child_element_ids = [
            str(child["element_id"])
            for child in children
            if child.get("name_stage") == "accepted"
        ]
        grounding_lock = list(
            gc.query(
                """
                UNWIND $child_element_ids AS child_element_id
                MATCH (child:StandardName)
                WHERE elementId(child) = child_element_id
                  AND child.name_stage = 'accepted'
                SET child._structural_authority_grounding_lock = true
                REMOVE child._structural_authority_grounding_lock
                RETURN count(child) AS locked
                """,
                child_element_ids=accepted_child_element_ids,
            )
        )
        if not grounding_lock or int(grounding_lock[0].get("locked") or 0) != len(
            accepted_child_element_ids
        ):
            refused.append(
                {
                    "parent_id": parent_id,
                    "reason": "accepted child grounding changed before replay",
                    "live_child_ids": live_child_ids,
                }
            )
            continue

        record = graph_ops._structural_authority_record(snapshot, accepting=False)
        if not graph_ops._persist_structural_authority(
            gc,
            record,
            parent_updates={},
        ):
            raise graph_ops.StructuralAuthorityConflict(
                f"derived parent {parent_id!r} changed before authority replay"
            )
        replayed.append(
            {
                "parent_id": parent_id,
                "authority_id": record["id"],
                "child_ids": list(record["child_ids"]),
                "accepted_child_ids": accepted_child_ids,
            }
        )

    return {
        "schema": "imas-codex.described-parent-authority-replay",
        "candidate_count": len(candidates),
        "replayed_count": len(replayed),
        "refused_count": len(refused),
        "replayed": replayed,
        "refused": refused,
    }


@retry_on_deadlock()
def replay_described_parent_authorities(
    *, gc: _TopologyProbe | None = None
) -> dict[str, Any]:
    """Write structural authority for already-accepted described parents.

    The historical marker is not treated as authority. Eligibility is derived
    from the current accepted lifecycle, valid state, real description, and
    exact live child closure. At least one child must itself be accepted; a
    parent grounded only by non-accepted children is reported without mutation.

    Each successful row reuses the same signed, content-addressed authority
    record as parent enrichment. Selection, parent locking, child snapshot, and
    authority persistence share one transaction when this function owns the
    graph client. An injected query handle is intended for tests or a
    caller-owned transaction.
    """
    if gc is not None:
        return _replay_described_parent_authorities(gc)

    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_ops import _TransactionQuery

    with GraphClient() as client, client.session() as session:
        transaction = session.begin_transaction()
        try:
            result = _replay_described_parent_authorities(
                _TransactionQuery(transaction)
            )
            transaction.commit()
            return result
        except Exception:
            transaction.rollback()
            raise

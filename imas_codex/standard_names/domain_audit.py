"""Read-only audit of every live standard name's physics domain.

A standard name's physics domain is not authored: it is derived from the DD
graph. ``IMASNode.physics_domain`` is the one source of truth, a name's
``source_domains`` is exactly the set of its current DD sources' domains, and
its ``physics_domain`` is the single domain ``select_primary_domain`` (see
``domain_ranking``) picks from that set. A name with no DD source — a derived
parent — takes the union of its children's domains through the same function.

Nothing enforces any of that at write time across every writer that touches a
name's domain, so the graph drifts: a name keeps a scalar the derivation would
not choose, an edge set that disagrees with its own ``source_domains``, or a
stale value from a source that has since been detached. This module re-asks the
derivation's question of every live name and reports the ones that disagree. It
is the measuring instrument the repair node runs against, so its counts now are
the "before" figure.

**Four checks**, one per property the derivation ties together:

1. ``physics_domain`` is set.
2. ``physics_domain`` equals ``select_primary_domain(expected)``.
3. ``source_domains`` equals the expected set — the domains of the name's
   current DD sources, or, for a derived parent with no DD source, the union of
   its live children's expected sets.
4. The ``HAS_PHYSICS_DOMAIN`` edges are exactly ``source_domains``, and every
   edge names the domain it points at. A name with no edge has an empty edge
   set rather than a placeholder, so absence and an unnameable target are
   reported as the different findings they are.

Sets are compared, never lists: the order a source arrived in carries no
meaning, and neither does the order domains sit in the graph property.

Read-only. It opens a graph client only when the caller does not supply one, and
never writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imas_codex.standard_names.domain_ranking import select_primary_domain

__all__ = [
    "LIVE_NAME_STAGES",
    "CHECK_PHYSICS_DOMAIN_SET",
    "CHECK_PHYSICS_DOMAIN_DERIVED",
    "CHECK_SOURCE_DOMAINS_EXPECTED",
    "CHECK_DOMAIN_EDGES_SOURCE_DOMAINS",
    "DomainCheck",
    "NameDomainState",
    "DomainAuditResult",
    "audit_domains",
]

#: The name stages a "live" name holds. A superseded or exhausted name is not
#: live: its edges are the deliberate provenance record, so there is no claim
#: for the derivation to contradict. Carried by the ``name_stage`` property on
#: :class:`StandardName` (``standard_name.yaml``, ``name_stage:`` range
#: ``NameStage``).
LIVE_NAME_STAGES: frozenset[str] = frozenset(
    {"accepted", "reviewed", "drafted", "pending"}
)

CHECK_PHYSICS_DOMAIN_SET = "physics_domain_unset"
CHECK_PHYSICS_DOMAIN_DERIVED = "physics_domain_not_derived"
CHECK_SOURCE_DOMAINS_EXPECTED = "source_domains_not_expected"
CHECK_DOMAIN_EDGES_SOURCE_DOMAINS = "edges_not_source_domains"

#: Every live name with everything the four checks need, in one read:
#:
#: * ``dd_paths`` / ``dd_domains`` — the name's current DD sources through
#:   ``StandardName <-PRODUCED_NAME- StandardNameSource -FROM_DD_PATH->
#:   IMASNode``; a name with no DD source is a derived parent.
#: * ``child_ids`` — the live children through ``(child)-[:HAS_PARENT]->(sn)``,
#:   whose expected sets a derived parent unions.
#: * ``edge_domains`` / ``edge_count`` — the ``HAS_PHYSICS_DOMAIN`` targets,
#:   matched without a label so an edge to something that is not a
#:   ``PhysicsDomain`` still surfaces. A name with no such edge gets an empty
#:   list, never a placeholder: absence and an unnamed target are different
#:   findings and ``edge_count`` minus the collected ids tells them apart.
#:
#: The row set is the live names — a bounded label scan over ``StandardName``
#: filtered by ``name_stage``; each walk is one hop.
_LIVE_NAMES_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN $live
CALL (sn) {
  OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
  OPTIONAL MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
  RETURN collect(DISTINCT dd.id) AS dd_paths,
         [domain IN collect(DISTINCT dd.physics_domain)
            WHERE domain IS NOT NULL] AS dd_domains
}
CALL (sn) {
  OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
  WHERE child.name_stage IN $live
  RETURN collect(DISTINCT child.id) AS child_ids
}
CALL (sn) {
  OPTIONAL MATCH (sn)-[:HAS_PHYSICS_DOMAIN]->(edge)
  RETURN collect(DISTINCT edge.id) AS edge_domains,
         count(edge) AS edge_count
}
RETURN sn.id AS sn_id,
       sn.name_stage AS name_stage,
       sn.physics_domain AS physics_domain,
       sn.source_domains AS source_domains,
       dd_paths,
       dd_domains,
       child_ids,
       edge_domains,
       edge_count
ORDER BY sn_id
"""


@dataclass(frozen=True)
class DomainCheck:
    """One check's failing ids, with the question it asks."""

    key: str
    description: str
    names: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.key,
            "description": self.description,
            "count": self.count,
            "names": list(self.names),
        }


@dataclass(frozen=True)
class NameDomainState:
    """One live name's stored domain state beside the derivation's answer."""

    sn_id: str
    name_stage: str | None
    physics_domain: str | None
    source_domains: frozenset[str]
    expected: frozenset[str]
    edge_domains: frozenset[str]
    derived_domain: str | None
    derived_domain_error: str | None
    dd_source_count: int
    is_derived_parent: bool
    unnamed_edge_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sn_id": self.sn_id,
            "name_stage": self.name_stage,
            "physics_domain": self.physics_domain,
            "source_domains": sorted(self.source_domains),
            "expected": sorted(self.expected),
            "edge_domains": sorted(self.edge_domains),
            "derived_domain": self.derived_domain,
            "derived_domain_error": self.derived_domain_error,
            "dd_source_count": self.dd_source_count,
            "is_derived_parent": self.is_derived_parent,
            "unnamed_edge_count": self.unnamed_edge_count,
        }


@dataclass
class DomainAuditResult:
    """Outcome of one read-only domain audit over the live names."""

    live_names: int
    states: tuple[NameDomainState, ...]
    checks: dict[str, DomainCheck]

    def counts(self) -> dict[str, int]:
        return {key: check.count for key, check in self.checks.items()}

    def state_for(self, sn_id: str) -> NameDomainState | None:
        for state in self.states:
            if state.sn_id == sn_id:
                return state
        return None

    def largest_source_sets(self, limit: int = 10) -> list[NameDomainState]:
        """The names drawing on the most domains, most first, ties by id."""
        ordered = sorted(
            self.states, key=lambda state: (-len(state.expected), state.sn_id)
        )
        return ordered[:limit]

    def as_dict(self) -> dict[str, Any]:
        return {
            "live_names": self.live_names,
            "checks": {key: check.as_dict() for key, check in self.checks.items()},
            "counts": self.counts(),
        }


def _expected_domains(
    by_id: dict[str, dict[str, Any]],
    sn_id: str,
    memo: dict[str, frozenset[str]],
    visiting: frozenset[str],
) -> frozenset[str]:
    """The set a name's ``source_domains`` must equal, derived from its sources.

    A name with DD sources takes their domains. A name with none is a derived
    parent and unions its live children's expected sets through the same rule.
    The recursion is guarded so a cycle in ``HAS_PARENT`` (which the graph does
    not forbid) terminates: a name already on the current path contributes
    nothing rather than recursing forever.
    """
    if sn_id in memo:
        return memo[sn_id]
    row = by_id[sn_id]
    if row.get("dd_paths"):
        result = frozenset(row.get("dd_domains") or [])
    elif sn_id in visiting:
        result = frozenset()
    else:
        union: set[str] = set()
        for child_id in row.get("child_ids") or []:
            if child_id in by_id:
                union |= _expected_domains(by_id, child_id, memo, visiting | {sn_id})
        result = frozenset(union)
    memo[sn_id] = result
    return result


def _derived_domain(expected: frozenset[str]) -> tuple[str | None, str | None]:
    """The one domain the declaration selects from *expected*, or why it refuses.

    ``select_primary_domain`` refuses a set carrying a value outside the
    declared vocabulary. That is a real finding — an unclassified or misspelt
    domain reaching a name — not an audit crash, so the refusal is returned
    beside the (absent) answer and the caller records it as a failure.
    """
    try:
        return select_primary_domain(expected), None
    except ValueError as exc:
        return None, str(exc)


def _build_result(rows: list[dict[str, Any]]) -> DomainAuditResult:
    """Score every live name in *rows* against the four checks."""
    by_id = {row["sn_id"]: row for row in rows}
    memo: dict[str, frozenset[str]] = {}

    states: list[NameDomainState] = []
    unset: list[str] = []
    not_derived: list[str] = []
    wrong_sources: list[str] = []
    wrong_edges: list[str] = []

    for sn_id in sorted(by_id):
        row = by_id[sn_id]
        expected = _expected_domains(by_id, sn_id, memo, frozenset())
        stored_primary = row.get("physics_domain") or None
        stored_sources = frozenset(row.get("source_domains") or [])
        edges = frozenset(row.get("edge_domains") or [])
        derived, derivation_error = _derived_domain(expected)
        dd_paths = row.get("dd_paths") or []
        unnamed_edges = max(0, (row.get("edge_count") or 0) - len(edges))

        states.append(
            NameDomainState(
                sn_id=sn_id,
                name_stage=row.get("name_stage"),
                physics_domain=stored_primary,
                source_domains=stored_sources,
                expected=expected,
                edge_domains=edges,
                derived_domain=derived,
                derived_domain_error=derivation_error,
                dd_source_count=len(dd_paths),
                is_derived_parent=not dd_paths,
                unnamed_edge_count=unnamed_edges,
            )
        )

        if not stored_primary:
            unset.append(sn_id)
        if stored_primary != derived:
            not_derived.append(sn_id)
        if stored_sources != expected:
            wrong_sources.append(sn_id)
        if edges != stored_sources or unnamed_edges:
            wrong_edges.append(sn_id)

    checks = {
        CHECK_PHYSICS_DOMAIN_SET: DomainCheck(
            key=CHECK_PHYSICS_DOMAIN_SET,
            description=("physics_domain is set on the live name"),
            names=tuple(unset),
        ),
        CHECK_PHYSICS_DOMAIN_DERIVED: DomainCheck(
            key=CHECK_PHYSICS_DOMAIN_DERIVED,
            description=(
                "physics_domain equals select_primary_domain(expected), the "
                "derivation's answer for the name's expected source set"
            ),
            names=tuple(not_derived),
        ),
        CHECK_SOURCE_DOMAINS_EXPECTED: DomainCheck(
            key=CHECK_SOURCE_DOMAINS_EXPECTED,
            description=(
                "source_domains equals the expected set — the name's current "
                "DD sources' domains, or its live children's union for a "
                "derived parent"
            ),
            names=tuple(wrong_sources),
        ),
        CHECK_DOMAIN_EDGES_SOURCE_DOMAINS: DomainCheck(
            key=CHECK_DOMAIN_EDGES_SOURCE_DOMAINS,
            description=(
                "HAS_PHYSICS_DOMAIN edges are exactly source_domains, and "
                "every edge names the domain it points at"
            ),
            names=tuple(wrong_edges),
        ),
    }
    return DomainAuditResult(live_names=len(by_id), states=tuple(states), checks=checks)


def audit_domains(gc: Any | None = None) -> DomainAuditResult:
    """Score every live name's domain against its DD sources. Read-only.

    Args:
        gc: An open ``GraphClient``, or None to open and close one.
    """
    own = gc is None
    if own:
        from imas_codex.graph.client import GraphClient

        client: Any = GraphClient()
    else:
        client = gc
    try:
        rows = [
            dict(row)
            for row in client.query(_LIVE_NAMES_QUERY, live=sorted(LIVE_NAME_STAGES))
        ]
    finally:
        if own:
            client.close()
    return _build_result(rows)

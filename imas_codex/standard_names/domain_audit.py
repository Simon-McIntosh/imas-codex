"""Read-only audit of every live standard name's physics domain.

A standard name's physics domain is not authored: ``source_domains`` is exactly
the set of its current DD sources' domains, and ``physics_domain`` is the
recorded correction or, failing that, the decision stored on the name by the one
writer in ``domain_decision``. A name with no DD source — a derived parent —
takes the union of its children's domains.

Nothing enforces any of that at write time across every writer that touches a
name's domain, so the graph drifts: a name keeps a scalar no stored decision
names, an edge set that disagrees with its own ``source_domains``, or a stale
value from a source that has since been detached. This module checks every live
name against the decision it stored and reports the ones that disagree, together
with the names that are undecided and why. It is the measuring instrument the
repair node runs against, so its counts now are the "before" figure.

**Four checks**, one per property the derivation ties together:

1. ``physics_domain`` is set.
2. ``physics_domain`` equals the stored decision or the recorded correction.
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

__all__ = [
    "LIVE_NAME_STAGES",
    "CHECK_PHYSICS_DOMAIN_SET",
    "CHECK_PHYSICS_DOMAIN_DECISION",
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
CHECK_PHYSICS_DOMAIN_DECISION = "physics_domain_not_stored_decision"
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
       sn.domain_decision_choice AS decision_choice,
       sn.domain_decision_candidates AS decision_candidates,
       sn.domain_correction AS correction,
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
    stored_choice: str | None
    correction: str | None
    decided_domain: str | None
    undecided_reason: str | None
    decision_candidates: tuple[str, ...]
    dd_source_count: int
    is_derived_parent: bool
    unnamed_edge_count: int

    @property
    def undecided(self) -> bool:
        return self.decided_domain is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sn_id": self.sn_id,
            "name_stage": self.name_stage,
            "physics_domain": self.physics_domain,
            "source_domains": sorted(self.source_domains),
            "expected": sorted(self.expected),
            "edge_domains": sorted(self.edge_domains),
            "stored_choice": self.stored_choice,
            "correction": self.correction,
            "decided_domain": self.decided_domain,
            "undecided_reason": self.undecided_reason,
            "decision_candidates": list(self.decision_candidates),
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

    @property
    def undecided(self) -> tuple[NameDomainState, ...]:
        """Live names with neither a stored decision nor a recorded correction."""
        return tuple(state for state in self.states if state.undecided)

    def undecided_reasons(self) -> dict[str, str]:
        return {state.sn_id: state.undecided_reason or "" for state in self.undecided}

    @property
    def undecided_share(self) -> float:
        """The undecided names' share of the live names, 0.0 when none are live."""
        if not self.live_names:
            return 0.0
        return len(self.undecided) / self.live_names

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
            "undecided": {
                "count": len(self.undecided),
                "share": self.undecided_share,
                "reasons": self.undecided_reasons(),
            },
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


def _decided_domain(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """The domain a name's stored decision or recorded correction names.

    A recorded correction stands above the stored decision, the order the one
    writer applies. A name with neither — an undecided name, whose call failed
    or whose probability fell below the floor — has no decided domain and is
    reported with the reason it is undecided.
    """
    correction = row.get("correction") or None
    if correction:
        return correction, None
    choice = row.get("decision_choice") or None
    if choice:
        return choice, None
    return None, "no stored decision and no recorded correction"


def _build_result(rows: list[dict[str, Any]]) -> DomainAuditResult:
    """Score every live name in *rows* against the four checks."""
    by_id = {row["sn_id"]: row for row in rows}
    memo: dict[str, frozenset[str]] = {}

    states: list[NameDomainState] = []
    unset: list[str] = []
    not_decided: list[str] = []
    wrong_sources: list[str] = []
    wrong_edges: list[str] = []

    for sn_id in sorted(by_id):
        row = by_id[sn_id]
        expected = _expected_domains(by_id, sn_id, memo, frozenset())
        stored_primary = row.get("physics_domain") or None
        stored_sources = frozenset(row.get("source_domains") or [])
        edges = frozenset(row.get("edge_domains") or [])
        decided, undecided_reason = _decided_domain(row)
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
                stored_choice=row.get("decision_choice") or None,
                correction=row.get("correction") or None,
                decided_domain=decided,
                undecided_reason=undecided_reason,
                decision_candidates=tuple(row.get("decision_candidates") or ()),
                dd_source_count=len(dd_paths),
                is_derived_parent=not dd_paths,
                unnamed_edge_count=unnamed_edges,
            )
        )

        if not stored_primary:
            unset.append(sn_id)
        if stored_primary != decided:
            not_decided.append(sn_id)
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
        CHECK_PHYSICS_DOMAIN_DECISION: DomainCheck(
            key=CHECK_PHYSICS_DOMAIN_DECISION,
            description=(
                "physics_domain equals the stored decision or the recorded "
                "correction, the one writer's order"
            ),
            names=tuple(not_decided),
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

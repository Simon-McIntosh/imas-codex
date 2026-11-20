"""Compare a plant-only graph package against the scope it was exported from.

A plant package is built by filtering a graph dump, and the filter is the only
place that can silently keep too much or drop too little: a facility filter
that keeps another facility's signals, or a kind whose deletion leaves the
package short, produces a server that answers wrongly without an error. This
module holds the comparison so the same rule measures the shared graph and the
package built from it, and a test can show the refusal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# Facility-less kinds a plant package is allowed to hold. ``Facility`` is the
# facility node the server resolves by id and ``GraphMeta`` is what the server
# reads at startup; every other facility-less kind belongs to the data
# dictionary or to content the plant tools never read.
PLANT_KEEP_LABELS: frozenset[str] = frozenset({"Facility", "GraphMeta"})


@dataclass(frozen=True)
class PlantScope:
    """The measured scope of a graph, in the terms a plant package is checked."""

    #: Nodes owned by the facility, by label.
    node_labels: dict[str, int] = field(default_factory=dict)
    #: Relationships between two nodes owned by the facility, by type.
    relationship_types: dict[str, int] = field(default_factory=dict)
    #: Nodes owned by some other facility.
    foreign_nodes: int = 0
    #: Nodes owned by no facility, by label.
    facilityless_labels: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_counts(
        cls,
        *,
        node_rows: list[dict],
        relationship_rows: list[dict],
        foreign_nodes: int,
        facilityless_rows: list[dict],
    ) -> PlantScope:
        """Build a scope from label/type rows returned by graph queries."""
        return cls(
            node_labels={r["label"]: r["count"] for r in node_rows},
            relationship_types={r["type"]: r["count"] for r in relationship_rows},
            foreign_nodes=foreign_nodes,
            facilityless_labels={r["label"]: r["count"] for r in facilityless_rows},
        )

    def to_dict(self) -> dict:
        """Return the scope as JSON-serialisable data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PlantScope:
        """Rebuild a scope from :meth:`to_dict` data."""
        return cls(
            node_labels=dict(data.get("node_labels", {})),
            relationship_types=dict(data.get("relationship_types", {})),
            foreign_nodes=int(data.get("foreign_nodes", 0)),
            facilityless_labels=dict(data.get("facilityless_labels", {})),
        )


def compare_plant_scope(expected: PlantScope, actual: PlantScope) -> list[str]:
    """Return every way *actual* is not the plant package *expected* describes.

    An empty list means the package holds the same facility-owned content, as
    many nodes of each kind and as many relationships of each type, holds no
    node of another facility, and holds no facility-less kind beyond
    :data:`PLANT_KEEP_LABELS`.
    """
    violations: list[str] = []

    for kind, count in sorted(expected.node_labels.items()):
        got = actual.node_labels.get(kind)
        if got != count:
            violations.append(
                f"node kind {kind}: package has {got!r}, shared graph has {count}"
            )
    for kind in sorted(set(actual.node_labels) - set(expected.node_labels)):
        violations.append(
            f"node kind {kind}: package has {actual.node_labels[kind]}, "
            "shared graph has none for this facility"
        )

    for kind, count in sorted(expected.relationship_types.items()):
        got = actual.relationship_types.get(kind)
        if got != count:
            violations.append(
                f"relationship {kind}: package has {got!r}, shared graph has {count}"
            )
    for kind in sorted(
        set(actual.relationship_types) - set(expected.relationship_types)
    ):
        violations.append(
            f"relationship {kind}: package has "
            f"{actual.relationship_types[kind]}, shared graph has none"
        )

    if actual.foreign_nodes:
        violations.append(
            f"foreign nodes: package holds {actual.foreign_nodes} node(s) of "
            "another facility"
        )

    unexpected = sorted(set(actual.facilityless_labels) - PLANT_KEEP_LABELS)
    for kind in unexpected:
        violations.append(
            f"facility-less kind {kind}: package holds "
            f"{actual.facilityless_labels[kind]} node(s), which the plant "
            "tools do not read"
        )
    for kind in sorted(PLANT_KEEP_LABELS):
        want = expected.facilityless_labels.get(kind, 0)
        got = actual.facilityless_labels.get(kind, 0)
        if want != got:
            violations.append(
                f"facility-less kind {kind}: package has {got}, shared graph has {want}"
            )

    return violations


def measure_plant_scope(query, facility: str) -> PlantScope:
    """Measure one graph in the terms a plant package is checked.

    *query* is a callable taking a Cypher statement and parameters and
    returning row dicts, so the export path can measure the graph it loads and
    the package it writes through the same rule.
    """
    node_rows = query(
        "MATCH (n) WHERE n.facility_id = $f UNWIND labels(n) AS l "
        "WITH l RETURN l AS label, count(*) AS count ORDER BY label",
        f=facility,
    )
    relationship_rows = query(
        "MATCH (a)-[r]->(b) WHERE a.facility_id = $f AND b.facility_id = $f "
        "RETURN type(r) AS type, count(*) AS count ORDER BY type",
        f=facility,
    )
    foreign_nodes = query(
        "MATCH (n) WHERE n.facility_id IS NOT NULL AND n.facility_id <> $f "
        "RETURN count(n) AS n",
        f=facility,
    )[0]["n"]
    facilityless_rows = query(
        "MATCH (n) WHERE n.facility_id IS NULL AND (NOT n:Facility OR n.id = $f) "
        "UNWIND labels(n) AS l "
        "WITH l RETURN l AS label, count(*) AS count ORDER BY label",
        f=facility,
    )
    return PlantScope.from_counts(
        node_rows=node_rows,
        relationship_rows=relationship_rows,
        foreign_nodes=foreign_nodes,
        facilityless_rows=facilityless_rows,
    )


def measure_plant_scope_json(bolt_uri: str, facility: str, output: str) -> None:
    """Measure a graph over Bolt and write the scope as JSON (``-`` = stdout).

    This is the entry point the SLURM filter script calls, so the package a
    dump build writes is measured by the same code the caller compares.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(bolt_uri, auth=None)
    try:
        with driver.session() as session:

            def query(cypher: str, **params) -> list[dict]:
                return [dict(row) for row in session.run(cypher, **params)]

            scope = measure_plant_scope(query, facility)
    finally:
        driver.close()
    payload = json.dumps(scope.to_dict())
    if output == "-":
        print(payload)
    else:
        with open(output, "w") as handle:
            handle.write(payload)


class PlantPackageMismatch(Exception):
    """A package does not hold what the graph it was exported from holds."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        super().__init__(
            "plant package does not match its export: " + "; ".join(violations)
        )


def verify_plant_package(expected: PlantScope, actual: PlantScope) -> None:
    """Raise :class:`PlantPackageMismatch` unless *actual* is the package of *expected*."""
    violations = compare_plant_scope(expected, actual)
    if violations:
        raise PlantPackageMismatch(violations)

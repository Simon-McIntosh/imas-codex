"""Compare a plant-only graph package against the scope it was exported from.

A plant package is built by filtering a graph dump, and the filter is the only
place that can silently keep too much or drop too little: a facility filter
that keeps another facility's signals, or a kind whose deletion leaves the
package short, produces a server that answers wrongly without an error. This
module holds the comparison so the same rule measures the shared graph and the
package built from it, and a test can show the refusal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

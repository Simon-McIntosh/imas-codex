"""Live-graph ceilings for repaired Standard Name integrity classes.

The constants below are ratchets, not desired steady-state counts. A later graph
repair may lower them, but new rows must not silently raise them. Every failure
prints the identities needed to inspect the regrowth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

pytestmark = pytest.mark.graph


# Live graph measurement taken 2026-08-20.
INTEGRITY_CEILING_MEASUREMENT_DATE = "2026-08-20"
MAX_SOURCES_WITH_MULTIPLE_LIVE_TARGETS = 23
MAX_STALE_SOURCES_WITH_LIVE_BINDINGS = 3
MAX_UNSOURCED_LIVE_NAMES_WITHOUT_LIVE_CHILDREN = 36
MAX_EXPLICIT_AXIS_BINDINGS_TO_GENERIC_PARENTS = 1

_KNOWN_AXIS_RESIDUE = {
    "source_id": "dd:langmuir_probes/reciprocating/plunge/mach_number_parallel",
    "parent_id": "mach_number",
    "axis_child_id": "parallel_mach_number",
    "axis": "parallel",
}

_MULTIPLE_LIVE_TARGETS_QUERY = """
MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(target:StandardName)
WHERE NOT (target.name_stage IN ['superseded', 'exhausted'])
WITH source, collect(DISTINCT target.id) AS live_targets
WHERE size(live_targets) > 1
RETURN source.id AS source_id, live_targets
ORDER BY source_id
"""

_STALE_LIVE_BINDINGS_QUERY = """
MATCH (source:StandardNameSource {status: 'stale'})
      -[:PRODUCED_NAME]->(target:StandardName)
WHERE NOT (target.name_stage IN ['superseded', 'exhausted'])
WITH source, collect(DISTINCT target.id) AS live_targets
RETURN source.id AS source_id, live_targets
ORDER BY source_id
"""

_UNSOURCED_WITHOUT_LIVE_CHILD_QUERY = """
MATCH (name:StandardName)
WHERE NOT (name.name_stage IN ['superseded', 'exhausted'])
  AND NOT EXISTS {
    MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(name)
  }
  AND NOT EXISTS {
    MATCH (child:StandardName)-[:HAS_PARENT]->(name)
    WHERE NOT (child.name_stage IN ['superseded', 'exhausted'])
  }
RETURN name.id AS name_id
ORDER BY name_id
"""

_EXPLICIT_AXIS_TO_GENERIC_PARENT_QUERY = """
MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(parent:StandardName)
MATCH (axis_child:StandardName)-[projection:HAS_PARENT]->(parent)
WHERE source.source_type = 'dd'
  AND source.status IN ['composed', 'attached']
  AND source.produced_sn_id = parent.id
  AND NOT (parent.name_stage IN ['superseded', 'exhausted'])
  AND NOT (axis_child.name_stage IN ['superseded', 'exhausted'])
  AND projection.operator_kind = 'projection'
  AND projection.axis IS NOT NULL
  AND (source.source_id ENDS WITH '/' + parent.id + '_' + projection.axis
       OR source.source_id ENDS WITH '/' + projection.axis + '_' + parent.id)
RETURN DISTINCT source.id AS source_id,
       parent.id AS parent_id,
       axis_child.id AS axis_child_id,
       projection.axis AS axis
ORDER BY source_id, parent_id, axis_child_id
"""


@pytest.fixture(scope="module")
def sn_graph(graph_client):
    """Use the shared live graph, skipping when its SN corpus is too thin."""
    rows = graph_client.query(
        "MATCH (sn:StandardName {name_stage: 'accepted'}) RETURN count(sn) AS count"
    )
    accepted = rows[0]["count"] if rows else 0
    if accepted < 10:
        pytest.skip(
            f"Graph has only {accepted} accepted StandardName nodes (<10); "
            "populate via `sn run` before judging integrity ceilings."
        )
    return graph_client


def _rows(graph_client: Any, query: str) -> list[dict[str, Any]]:
    return [dict(row) for row in graph_client.query(query)]


def _details(
    rows: Sequence[Mapping[str, Any]],
    identity_fields: Sequence[str],
) -> str:
    return "; ".join(
        ", ".join(f"{field}={row.get(field)!r}" for field in identity_fields)
        for row in rows
    )


def _assert_at_most(
    rows: Sequence[Mapping[str, Any]],
    ceiling: int,
    label: str,
    identity_fields: Sequence[str],
) -> None:
    assert len(rows) <= ceiling, (
        f"{label} measured {len(rows)}, above the {ceiling} ceiling measured "
        f"{INTEGRITY_CEILING_MEASUREMENT_DATE}. Offending identities: "
        f"{_details(rows, identity_fields)}"
    )


def test_sources_with_multiple_live_targets_do_not_regrow(sn_graph):
    rows = _rows(sn_graph, _MULTIPLE_LIVE_TARGETS_QUERY)
    _assert_at_most(
        rows,
        MAX_SOURCES_WITH_MULTIPLE_LIVE_TARGETS,
        "StandardNameSource nodes with multiple live targets",
        ("source_id", "live_targets"),
    )


def test_stale_sources_with_live_bindings_do_not_regrow(sn_graph):
    rows = _rows(sn_graph, _STALE_LIVE_BINDINGS_QUERY)
    _assert_at_most(
        rows,
        MAX_STALE_SOURCES_WITH_LIVE_BINDINGS,
        "stale StandardNameSource nodes retaining live bindings",
        ("source_id", "live_targets"),
    )


def test_unsourced_live_names_without_live_children_do_not_regrow(sn_graph):
    rows = _rows(sn_graph, _UNSOURCED_WITHOUT_LIVE_CHILD_QUERY)
    _assert_at_most(
        rows,
        MAX_UNSOURCED_LIVE_NAMES_WITHOUT_LIVE_CHILDREN,
        "live StandardName nodes with neither a source nor a live structural child",
        ("name_id",),
    )


def test_explicit_axis_paths_do_not_bind_new_generic_parents(sn_graph):
    rows = _rows(sn_graph, _EXPLICIT_AXIS_TO_GENERIC_PARENT_QUERY)
    _assert_at_most(
        rows,
        MAX_EXPLICIT_AXIS_BINDINGS_TO_GENERIC_PARENTS,
        "explicit-axis DD sources bound to a generic structural parent",
        ("source_id", "parent_id", "axis_child_id", "axis"),
    )

    unexpected = [row for row in rows if row != _KNOWN_AXIS_RESIDUE]
    assert not unexpected, (
        "The only permitted measured residue is "
        "dd:langmuir_probes/reciprocating/plunge/mach_number_parallel bound to "
        "mach_number while parallel_mach_number is its live projection child. "
        f"Unexpected identities: {_details(unexpected, tuple(_KNOWN_AXIS_RESIDUE))}"
    )

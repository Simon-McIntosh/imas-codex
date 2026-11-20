"""The group-enrichment statements parse and its cohort selector sees data."""

from __future__ import annotations

import pytest

from imas_codex.discovery.signals.group_enrichment import (
    COHORT_QUERY,
    COMPONENT_CONTEXT_QUERY,
    COMPONENT_QUERY,
    PATTERN_CONTEXT_QUERY,
    PATTERN_QUERY,
)
from imas_codex.graph.client import GraphClient

STATEMENTS = (
    COHORT_QUERY,
    COMPONENT_QUERY,
    COMPONENT_CONTEXT_QUERY,
    PATTERN_QUERY,
    PATTERN_CONTEXT_QUERY,
)

PARAMETERS = {
    "facility": "iter",
    "component": "iter:plant_component:DEMO-UNIT:QAZ0001",
    "pattern": "iter:archive_field_pattern:invented",
}


@pytest.mark.graph
@pytest.mark.parametrize("statement", STATEMENTS)
def test_group_statement_parses(statement):
    with GraphClient() as gc:
        gc.query("EXPLAIN " + statement, **PARAMETERS)


@pytest.mark.graph
def test_cohort_selector_sees_a_plant_context_cohort():
    with GraphClient() as gc:
        total = gc.query(
            "MATCH (s:FacilitySignal {facility_id: $facility}) "
            "WHERE s.plant_context_state IS NOT NULL RETURN count(s) AS total",
            facility="iter",
        )[0]["total"]
        if not total:
            pytest.skip("no plant-context cohort in this graph")
        rows = gc.query(COHORT_QUERY, facility="iter")
    assert len(rows) == total
    assert all(row["context_state"] in {"exact", "derived", "absent"} for row in rows)

"""Every group-enrichment statement parses and its cohort selector sees data."""

from __future__ import annotations

import pytest

from iter_codex.discovery.signals import group_enrichment as module
from iter_codex.graph.client import GraphClient

COMPONENT = "iter:plant_component:DEMO-UNIT:QAZ0001"
PATTERN = "iter:archive_field_pattern:invented"
STATE_ROW = {
    "id": "iter:general/codac_uda/DEMO-UNIT:QAZ0001-TEMP",
    "group_enrichment_state": "described",
    "group_enrichment_reason": None,
    "status": None,
}

#: Every statement the module issues, by the name it is defined under, with the
#: parameters its call site supplies. A statement constant added without an
#: entry here fails ``test_every_statement_constant_is_gated``.
STATEMENTS = {
    "COHORT_QUERY": (module.COHORT_QUERY, {"facility": "iter"}),
    "COMPONENT_QUERY": (
        module.COMPONENT_QUERY,
        {"facility": "iter", "component": COMPONENT},
    ),
    "COMPONENT_CONTEXT_QUERY": (
        module.COMPONENT_CONTEXT_QUERY,
        {"facility": "iter", "component": COMPONENT},
    ),
    "PATTERN_QUERY": (module.PATTERN_QUERY, {"facility": "iter", "pattern": PATTERN}),
    "PATTERN_CONTEXT_QUERY": (
        module.PATTERN_CONTEXT_QUERY,
        {"facility": "iter", "pattern": PATTERN},
    ),
    "PILOT_COMPONENTS_QUERY": (
        module.PILOT_COMPONENTS_QUERY,
        {"facility": "iter", "components": [COMPONENT]},
    ),
    "COMPONENT_DESCRIPTION_WRITE": (
        module.COMPONENT_DESCRIPTION_WRITE,
        {"facility": "iter", "component": COMPONENT, "text": "Invented description."},
    ),
    "PATTERN_DESCRIPTION_WRITE": (
        module.PATTERN_DESCRIPTION_WRITE,
        {"facility": "iter", "pattern": PATTERN, "text": "Invented description."},
    ),
    "GROUP_STATE_WRITE": (
        module.GROUP_STATE_WRITE,
        {"facility": "iter", "rows": [STATE_ROW]},
    ),
}


@pytest.mark.graph
@pytest.mark.parametrize("name", sorted(STATEMENTS))
def test_group_statement_parses(name):
    statement, parameters = STATEMENTS[name]
    with GraphClient() as gc:
        gc.query("EXPLAIN " + statement, **parameters)


@pytest.mark.graph
def test_every_statement_constant_is_gated():
    declared = {
        name
        for name in dir(module)
        if name.isupper() and name.endswith(("_QUERY", "_WRITE"))
    }
    assert declared == set(STATEMENTS), sorted(declared ^ set(STATEMENTS))


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
        rows = gc.query(module.COHORT_QUERY, facility="iter")
    assert len(rows) == total
    assert all(row["context_state"] in {"exact", "derived", "absent"} for row in rows)

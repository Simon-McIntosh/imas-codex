"""A plant package must hold the facility's content and nothing else.

These checks are what the export run compares the built package against, so
each failure below names a package that would answer wrongly: a foreign
facility's signals, a facility-less kind the plant tools never read, or a kind
whose count differs from the graph it was exported from.
"""

from __future__ import annotations

from imas_codex.graph.plant_scope import PlantScope, compare_plant_scope


def _shared() -> PlantScope:
    return PlantScope(
        node_labels={"FacilitySignal": 120, "Pulse": 7, "PulseCategory": 2},
        relationship_types={"IN_PULSE_CATEGORY": 7, "AT_CONTROL_NODE": 90},
        foreign_nodes=0,
        facilityless_labels={"Facility": 1, "GraphMeta": 1},
    )


def test_matching_package_reports_no_violation():
    assert compare_plant_scope(_shared(), _shared()) == []


def test_one_omitted_kind_fails_the_equality():
    package = PlantScope(
        node_labels={"FacilitySignal": 120, "PulseCategory": 2},
        relationship_types={"IN_PULSE_CATEGORY": 7, "AT_CONTROL_NODE": 90},
        facilityless_labels={"Facility": 1, "GraphMeta": 1},
    )
    violations = compare_plant_scope(_shared(), package)
    assert violations == ["node kind Pulse: package has None, shared graph has 7"]


def test_relationship_type_shortfall_fails_the_equality():
    package = PlantScope(
        node_labels={"FacilitySignal": 120, "Pulse": 7, "PulseCategory": 2},
        relationship_types={"IN_PULSE_CATEGORY": 7, "AT_CONTROL_NODE": 89},
        facilityless_labels={"Facility": 1, "GraphMeta": 1},
    )
    assert compare_plant_scope(_shared(), package) == [
        "relationship AT_CONTROL_NODE: package has 89, shared graph has 90"
    ]


def test_another_facility_left_in_fails_the_foreign_count():
    package = PlantScope(
        node_labels={"FacilitySignal": 120, "Pulse": 7, "PulseCategory": 2},
        relationship_types={"IN_PULSE_CATEGORY": 7, "AT_CONTROL_NODE": 90},
        foreign_nodes=4,
        facilityless_labels={"Facility": 1, "GraphMeta": 1},
    )
    assert compare_plant_scope(_shared(), package) == [
        "foreign nodes: package holds 4 node(s) of another facility"
    ]


def test_facilityless_kind_the_tools_do_not_read_is_a_violation():
    package = PlantScope(
        node_labels={"FacilitySignal": 120, "Pulse": 7, "PulseCategory": 2},
        relationship_types={"IN_PULSE_CATEGORY": 7, "AT_CONTROL_NODE": 90},
        facilityless_labels={"Facility": 1, "GraphMeta": 1, "Unit": 12},
    )
    assert compare_plant_scope(_shared(), package) == [
        "facility-less kind Unit: package holds 12 node(s), which the plant "
        "tools do not read"
    ]


def test_kept_facilityless_kind_is_counted_not_assumed():
    package = PlantScope(
        node_labels={"FacilitySignal": 120, "Pulse": 7, "PulseCategory": 2},
        relationship_types={"IN_PULSE_CATEGORY": 7, "AT_CONTROL_NODE": 90},
        facilityless_labels={"Facility": 4, "GraphMeta": 1},
    )
    assert compare_plant_scope(_shared(), package) == [
        "facility-less kind Facility: package has 4, shared graph has 1"
    ]


def test_kind_absent_from_the_shared_graph_is_not_silently_accepted():
    package = PlantScope(
        node_labels={
            "FacilitySignal": 120,
            "Pulse": 7,
            "PulseCategory": 2,
            "IMASNode": 61000,
        },
        relationship_types={"IN_PULSE_CATEGORY": 7, "AT_CONTROL_NODE": 90},
        facilityless_labels={"Facility": 1, "GraphMeta": 1},
    )
    assert compare_plant_scope(_shared(), package) == [
        "node kind IMASNode: package has 61000, shared graph has none for this facility"
    ]

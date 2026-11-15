"""Candidate archive-name interpretation using invented selectors only."""

from unittest.mock import Mock

import pytest

from imas_codex.discovery.signals.scanners.codac_uda import CodacUDAScanner
from imas_codex.discovery.signals.variable_name_graph import (
    parse_archive_catalogue,
    persist_name_parses,
)
from imas_codex.discovery.signals.variable_names import (
    PARSE_PROVENANCE,
    parse_variable_name,
)


def test_scanner_parses_candidate_fields():
    selector = "DEMO-COOL-LOOP-PUMP:PCV9001-MEAS"
    signal = CodacUDAScanner._signal(
        "demo",
        "demo:archive",
        "demo:access",
        {
            "name": selector,
            "metadata_state": "available",
            "archive_metadata": {"type": {"datatypes": []}, "full": []},
        },
        "example",
    )
    assert signal.accessor == selector
    assert getattr(signal, "plant_system_code", None) == "DEMO"
    assert signal.plant_component_code == "DEMO-COOL-LOOP-PUMP"
    assert signal.control_function == "PCV9001"
    assert signal.variable_token == "MEAS"
    assert signal.name_parse_outcome == "parsed"
    assert signal.name_parse_rule_status == "unconfirmed"


# Explicit expected tokens permit a manual check without a live catalogue.
@pytest.mark.parametrize(
    "selector,system,component,function,variable",
    [
        (
            "DEMO-COOL-LOOP-PUMP:PCV9001-MEAS",
            "DEMO",
            "DEMO-COOL-LOOP-PUMP",
            "PCV9001",
            "MEAS",
        ),
        (
            "DEMO-COOL-LOOP-PUMP:PCV9001-TARGET",
            "DEMO",
            "DEMO-COOL-LOOP-PUMP",
            "PCV9001",
            "TARGET",
        ),
        (
            "DEMO-COOL-LOOP-PUMP:PCV9001-DRIVE_EN",
            "DEMO",
            "DEMO-COOL-LOOP-PUMP",
            "PCV9001",
            "DRIVE_EN",
        ),
        (
            "DEMO-COOL-LOOP-PUMP:PCV9001-ALARM0",
            "DEMO",
            "DEMO-COOL-LOOP-PUMP",
            "PCV9001",
            "ALARM0",
        ),
        (
            "DEMO-COOL-LOOP-PUMP:PCV9001-LIMIT_HIGH",
            "DEMO",
            "DEMO-COOL-LOOP-PUMP",
            "PCV9001",
            "LIMIT_HIGH",
        ),
        (
            "SYN-HEAT-BRANCH-UNIT:TCV8002-MEAS",
            "SYN",
            "SYN-HEAT-BRANCH-UNIT",
            "TCV8002",
            "MEAS",
        ),
        (
            "SYN-HEAT-BRANCH-UNIT:TCV8002-TARGET",
            "SYN",
            "SYN-HEAT-BRANCH-UNIT",
            "TCV8002",
            "TARGET",
        ),
        (
            "SYN-HEAT-BRANCH-UNIT:TCV8002-DRIVE_EN",
            "SYN",
            "SYN-HEAT-BRANCH-UNIT",
            "TCV8002",
            "DRIVE_EN",
        ),
        (
            "SYN-HEAT-BRANCH-UNIT:TCV8002-ALARM0",
            "SYN",
            "SYN-HEAT-BRANCH-UNIT",
            "TCV8002",
            "ALARM0",
        ),
        (
            "SYN-HEAT-BRANCH-UNIT:TCV8002-LIMIT_HIGH",
            "SYN",
            "SYN-HEAT-BRANCH-UNIT",
            "TCV8002",
            "LIMIT_HIGH",
        ),
        (
            "TEST-GAS-TRAIN-VALVE:FCV7003-MEAS",
            "TEST",
            "TEST-GAS-TRAIN-VALVE",
            "FCV7003",
            "MEAS",
        ),
        (
            "TEST-GAS-TRAIN-VALVE:FCV7003-TARGET",
            "TEST",
            "TEST-GAS-TRAIN-VALVE",
            "FCV7003",
            "TARGET",
        ),
        (
            "TEST-GAS-TRAIN-VALVE:FCV7003-DRIVE_EN",
            "TEST",
            "TEST-GAS-TRAIN-VALVE",
            "FCV7003",
            "DRIVE_EN",
        ),
        (
            "TEST-GAS-TRAIN-VALVE:FCV7003-ALARM0",
            "TEST",
            "TEST-GAS-TRAIN-VALVE",
            "FCV7003",
            "ALARM0",
        ),
        (
            "TEST-GAS-TRAIN-VALVE:FCV7003-LIMIT_HIGH",
            "TEST",
            "TEST-GAS-TRAIN-VALVE",
            "FCV7003",
            "LIMIT_HIGH",
        ),
        (
            "MOCK-POWER-FEED-SWITCH:ICV6004-MEAS",
            "MOCK",
            "MOCK-POWER-FEED-SWITCH",
            "ICV6004",
            "MEAS",
        ),
        (
            "MOCK-POWER-FEED-SWITCH:ICV6004-TARGET",
            "MOCK",
            "MOCK-POWER-FEED-SWITCH",
            "ICV6004",
            "TARGET",
        ),
        (
            "MOCK-POWER-FEED-SWITCH:ICV6004-DRIVE_EN",
            "MOCK",
            "MOCK-POWER-FEED-SWITCH",
            "ICV6004",
            "DRIVE_EN",
        ),
        (
            "MOCK-POWER-FEED-SWITCH:ICV6004-ALARM0",
            "MOCK",
            "MOCK-POWER-FEED-SWITCH",
            "ICV6004",
            "ALARM0",
        ),
        (
            "MOCK-POWER-FEED-SWITCH:ICV6004-LIMIT_HIGH",
            "MOCK",
            "MOCK-POWER-FEED-SWITCH",
            "ICV6004",
            "LIMIT_HIGH",
        ),
    ],
)
def test_twenty_hand_checked_invented_parses(
    selector, system, component, function, variable
):
    parsed = parse_variable_name(selector)
    assert parsed.selector == selector
    assert parsed.outcome == "parsed"
    assert (
        parsed.plant_system_code,
        parsed.plant_component_code,
        parsed.control_function,
        parsed.variable_token,
    ) == (system, component, function, variable)
    hierarchy = parsed.components("example")
    assert hierarchy[-1].code == component
    assert hierarchy[0].code == system
    assert hierarchy[0].parent_component is None
    assert [item.parent_component for item in hierarchy[1:]] == [
        item.id for item in hierarchy[:-1]
    ]
    assert parsed.properties("example")["name_parse_rule_status"] == "unconfirmed"
    assert parsed.properties("example")["name_parse_provenance"] == PARSE_PROVENANCE


@pytest.mark.parametrize(
    "selector",
    [
        "arbitrary text",
        "demo-COOL:PCV9001-MEAS",
        "DEMO--COOL:PCV9001-MEAS",
        "DEMO-COOL:PCV9001-MEAS:EXTRA",
        " DEMO-COOL:PCV9001-MEAS",
        "",
        "DEMO:TEMP",
    ],
)
def test_nonconforming_name_has_no_component(selector):
    parsed = parse_variable_name(selector)
    assert parsed.outcome == "not_conforming"
    assert parsed.plant_system_code is None
    assert parsed.plant_component_code is None
    assert parsed.control_function is None
    assert parsed.variable_token is None
    assert parsed.components("demo") == []
    assert parsed.properties("demo")["candidate_plant_component"] is None
    graph = Mock()
    graph.query.return_value = [{"written": 1}]
    persist_name_parses(
        graph,
        [
            {
                "id": "demo:signal",
                "facility": "demo",
                "selector": selector,
                "properties": {"candidate_plant_component": "invented:unsupported"},
            }
        ],
    )
    written = graph.query.call_args.kwargs["items"][0]
    assert written["properties"]["candidate_plant_component"] is None
    assert written["components"] == []


@pytest.mark.parametrize(
    "tail,function,variable",
    [
        ("", None, None),
        ("PCV9001", "PCV9001", None),
        ("PCV9001-", "PCV9001", None),
        ("UNKNOWN-MEAS", None, "MEAS"),
        ("PCV9001-bad suffix", "PCV9001", None),
        ("PCV9001-MEAS-EXTRA", "PCV9001", None),
    ],
)
def test_partial_parse_retains_only_recognised_tokens(tail, function, variable):
    parsed = parse_variable_name("DEMO-COOL:" + tail)
    assert parsed.outcome == "partly_parsed"
    assert parsed.plant_component_code == "DEMO-COOL"
    assert parsed.control_function == function
    assert parsed.variable_token == variable
    assert parsed.reason


def test_write_refuses_changed_or_missing_selector():
    graph = Mock()
    graph.query.return_value = [{"written": 0}]
    with pytest.raises(RuntimeError, match="not acknowledged"):
        persist_name_parses(
            graph,
            [{"id": "demo:missing", "facility": "demo", "selector": "arbitrary text"}],
        )


def test_catalogue_pass_is_not_limited_to_one_cohort_or_facility():
    graph = Mock()
    graph.query.side_effect = [
        [{"total": 2, "selectors": 2}],
        [
            {
                "id": "alpha:one",
                "facility": "alpha",
                "selector": "DEMO-COOL:PCV9001-MEAS",
            }
        ],
        [{"id": "beta:two", "facility": "beta", "selector": "arbitrary text"}],
        [],
        [{"written": 1}],
        [{"written": 1}],
        [{"total": 2}],
    ]
    receipt = parse_archive_catalogue(graph, batch_size=1)
    assert receipt == {
        "total": 2,
        "written": 2,
        "outcomes": {"parsed": 1, "partly_parsed": 0, "not_conforming": 1},
    }
    assert all("cohort" not in call.kwargs for call in graph.query.call_args_list)


@pytest.mark.parametrize(
    "control", [{"total": 0, "selectors": 0}, {"total": 2, "selectors": 1}]
)
def test_catalogue_pass_refuses_empty_or_missing_selector_measurement(control):
    graph = Mock()
    graph.query.return_value = [control]
    with pytest.raises(RuntimeError, match="empty or contains missing selectors"):
        parse_archive_catalogue(graph)
    assert graph.query.call_count == 1


@pytest.mark.parametrize("stage", ["snapshot", "write"])
def test_catalogue_pass_refuses_population_changes(stage):
    graph = Mock()
    graph.query.side_effect = [
        [{"total": 1, "selectors": 1}],
        [{"id": "demo:one", "facility": "demo", "selector": "arbitrary text"}],
        [],
        [{"written": 1}],
        [{"total": 2}],
    ]
    if stage == "snapshot":
        graph.query.side_effect = [[{"total": 2, "selectors": 2}], []]
    with pytest.raises(RuntimeError, match="catalogue changed"):
        parse_archive_catalogue(graph)

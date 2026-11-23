"""Naming-convention boundaries and graph parse inputs using invented names."""

from unittest.mock import Mock

import pytest

from iter_codex.discovery.signals.scanners.codac_uda import CodacUDAScanner
from iter_codex.discovery.signals.variable_name_graph import (
    parse_archive_catalogue,
    persist_name_parses,
    snapshot_archive_signals,
)
from iter_codex.discovery.signals.variable_names import (
    PARSE_PROVENANCE,
    parse_variable_name,
)


def test_component_identity_includes_control_path_category_and_sequence():
    first = parse_variable_name("DEMO-UNIT:QAZ0001-ABCD")
    second = parse_variable_name("DEMO-UNIT:QAZ0002-ABCD")
    other_path = parse_variable_name("DEMO-NEXT:QAZ0001-ABCD")
    assert first.outcome == "component_variable"
    assert first.plant_component_code == "DEMO-UNIT:QAZ0001"
    assert len({item.plant_component_code for item in (first, second, other_path)}) == 3
    assert first.component_category == "QAZ"
    assert first.component_sequence == "0001"
    assert first.field_code == "ABCD"
    assert first.field_discriminator is None
    component = first.components("demo")[0]
    assert component.control_node == "demo:control_node:DEMO-UNIT"
    assert component.component_category == "QAZ"
    assert component.description is None


@pytest.mark.parametrize("depth", range(1, 6))
def test_control_hierarchy_has_one_to_five_levels(depth):
    path = "-".join(("DEMO", "UNIT", "SUB1", "LEAF", "ZONE")[:depth])
    parsed = parse_variable_name(path + ":QAZ0001-ABCD")
    nodes = parsed.control_nodes("demo")
    assert [node.level for node in nodes] == list(range(1, depth + 1))
    assert nodes[-1].code == path
    assert nodes[0].parent_control_node is None
    assert [node.parent_control_node for node in nodes[1:]] == [
        node.id for node in nodes[:-1]
    ]
    assert len(parsed.components("demo")) == 1


@pytest.mark.parametrize(
    "selector",
    [
        "DEMO-UNIT:STATUS",
        "DEMO-UNIT:1234",
        "DEMO-UNIT:QAZ0001",
        "DEMO-UNIT:QAZ0001-ABCDE",
        "DEMO-UNIT:QAZ0001-ABCD-ABCDE",
        "DEMO-UNIT:QAZ00001-ABCD",
        "DEMO-UNIT:QAZZ0001-ABCD",
        "DEMO-UNIT:" + "X" * 24,
    ],
)
def test_valid_function_variable_has_a_control_node_and_no_component(selector):
    parsed = parse_variable_name(selector)
    assert parsed.outcome == "function_variable"
    assert parsed.control_path == "DEMO-UNIT"
    assert parsed.variable_id == selector.split(":")[1]
    assert parsed.plant_component_code is None
    assert parsed.field_code is None
    assert parsed.component_category is None
    assert parsed.components("demo") == []
    props = parsed.properties("demo")
    assert props["control_node"] == "demo:control_node:DEMO-UNIT"
    assert "candidate_plant_component" not in props


@pytest.mark.parametrize(
    "selector",
    [
        "unparseable fixture",
        "",
        "DEMO-UNIT:QAZ0001-ABCD:EXTRA",
        "demo-UNIT:QAZ0001-ABCD",
        "DEMO--UNIT:QAZ0001-ABCD",
        "DEMOX-UNIT:QAZ0001-ABCD",
        "DEMO-UNIT-SUB1-LEAF-ZONE-EXTR:QAZ0001-ABCD",
        "DEMO-UNIT:",
        "DEMO-UNIT:HAS_SPACE ",
        "DEMO-UNIT:HAS_UNDERSCORE",
        "DEMO-UNIT:" + "X" * 25,
    ],
)
def test_nonconforming_variable_has_no_inherited_identity(selector):
    parsed = parse_variable_name(selector)
    assert parsed.outcome == "not_conforming"
    assert parsed.control_path is None
    assert parsed.components("demo") == parsed.control_nodes("demo") == []
    assert parsed.properties("demo")["control_node"] is None
    graph = Mock()
    graph.query.return_value = [{"written": 1}]
    persist_name_parses(
        graph, [{"id": "demo:signal", "facility": "demo", "selector": selector}]
    )
    item = graph.query.call_args.kwargs["items"][0]
    assert item["components"] == item["control_nodes"] == []
    assert "candidate_plant_component" not in item["properties"]
    assert item["properties"]["name_parse_reason"]


@pytest.mark.parametrize(
    "tail,field,discriminator",
    [
        ("QAZ0001-ABCD", "ABCD", None),
        ("QAZ0001-ABCD1", "ABCD", "1"),
        ("QAZ0001-ABCD1234", "ABCD", "1234"),
        ("QAZ0001-ABCD1Z9", "ABCD", "1Z9"),
        ("QAZ0001-ABCD-Z", "ABCD", "Z"),
        ("QAZ0001-ABCD-Z9X8", "ABCD", "Z9X8"),
        ("Q1-A", "A", None),
        ("QZ01-ABC0", "ABC", "0"),
    ],
)
def test_component_fields_exclude_the_optional_discriminator(
    tail, field, discriminator
):
    parsed = parse_variable_name("DEMO-UNIT:" + tail)
    assert parsed.outcome == "component_variable"
    assert parsed.field_code == field
    assert parsed.field_discriminator == discriminator


@pytest.mark.parametrize("index", range(20))
def test_twenty_invented_component_parses(index):
    selector = f"DEMO-UNIT:QAZ{index:04d}-ABCD-Z9"
    parsed = parse_variable_name(selector)
    assert parsed.plant_component_code == f"DEMO-UNIT:QAZ{index:04d}"
    assert parsed.component_sequence == f"{index:04d}"
    assert parsed.variable_id == f"QAZ{index:04d}-ABCD-Z9"
    assert parsed.field_code == "ABCD"
    assert parsed.field_discriminator == "Z9"
    assert parsed.properties("demo")["name_parse_provenance"] == PARSE_PROVENANCE


def test_scanner_uses_the_convention_parser():
    selector = "DEMO-UNIT:QAZ0001-ABCD"
    signal = CodacUDAScanner._signal(
        "demo",
        "demo:archive",
        "demo:access",
        {
            "name": selector,
            "metadata_state": "available",
            "archive_metadata": {"type": {"datatypes": []}, "full": []},
        },
        "fixture",
    )
    assert signal.accessor == selector
    assert signal.name_parse_outcome == "component_variable"
    assert signal.plant_component_code == "DEMO-UNIT:QAZ0001"
    assert signal.control_path == "DEMO-UNIT"
    assert signal.component_category == "QAZ"
    assert signal.field_code == "ABCD"


def test_parse_write_refuses_missing_acknowledgement():
    graph = Mock()
    graph.query.return_value = [{"written": 0}]
    with pytest.raises(RuntimeError, match="not acknowledged"):
        persist_name_parses(
            graph,
            [
                {
                    "id": "demo:missing",
                    "facility": "demo",
                    "selector": "unparseable fixture",
                }
            ],
        )


@pytest.mark.parametrize(
    "control", [{"total": 0, "selectors": 0}, {"total": 2, "selectors": 1}]
)
def test_snapshot_guard_refuses_missing_selector_or_required_population(control):
    graph = Mock()
    graph.query.return_value = [control]
    with pytest.raises(RuntimeError, match="empty or contains missing selectors"):
        snapshot_archive_signals(graph, require_nonempty=True)
    assert graph.query.call_count == 1


def test_catalogue_parser_reuses_guard_and_counts_all_outcomes():
    graph = Mock()
    graph.query.side_effect = [
        [{"total": 3, "selectors": 3}],
        [
            {
                "id": "alpha:one",
                "facility": "alpha",
                "selector": "DEMO-UNIT:QAZ0001-ABCD",
            },
            {"id": "beta:two", "facility": "beta", "selector": "DEMO-UNIT:STATUS"},
            {"id": "beta:three", "facility": "beta", "selector": "unparseable fixture"},
        ],
        [],
        [{"total": 3, "selectors": 3}],
        [{"written": 3}],
        [{"total": 3, "selectors": 3}],
    ]
    assert parse_archive_catalogue(graph) == {
        "total": 3,
        "written": 3,
        "outcomes": {
            "component_variable": 1,
            "function_variable": 1,
            "not_conforming": 1,
        },
    }
    assert graph.query.call_args_list[0].kwargs["facility"] is None


@pytest.mark.parametrize("stage", ["snapshot", "write"])
def test_catalogue_guard_refuses_population_changes(stage):
    graph = Mock()
    graph.query.side_effect = [
        [{"total": 1, "selectors": 1}],
        [{"id": "demo:one", "facility": "demo", "selector": "unparseable fixture"}],
        [],
        [{"total": 1, "selectors": 1}],
        [{"written": 1}],
        [{"total": 2, "selectors": 2}],
    ]
    if stage == "snapshot":
        graph.query.side_effect = [[{"total": 2, "selectors": 2}], []]
    with pytest.raises(RuntimeError, match="catalogue changed"):
        parse_archive_catalogue(graph)

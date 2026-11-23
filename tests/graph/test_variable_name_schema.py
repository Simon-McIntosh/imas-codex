"""Control hierarchy and component identity remain separate schema entities."""

from iter_codex.graph.models import ControlNode, FacilitySignal, PlantComponent
from iter_codex.graph.schema import GraphSchema


def test_control_and_component_relationships_and_roundtrip():
    schema = GraphSchema()
    expected = {
        "ControlNode": {
            ("facility_id", "Facility", "AT_FACILITY"),
            ("parent_control_node", "ControlNode", "PARENT_CONTROL_NODE"),
        },
        "PlantComponent": {
            ("facility_id", "Facility", "AT_FACILITY"),
            ("control_node", "ControlNode", "AT_CONTROL_NODE"),
        },
        "FacilitySignal": {
            ("archive_component", "PlantComponent", "IN_ARCHIVE_COMPONENT"),
            ("control_node", "ControlNode", "AT_CONTROL_NODE"),
        },
    }
    for label, relationships in expected.items():
        assert relationships <= {
            (r.slot_name, r.to_class, r.cypher_type)
            for r in schema.get_relationships_from(label)
        }
    control = ControlNode(
        id="demo:control_node:DEMO", facility_id="demo", code="DEMO", level=1
    )
    component = PlantComponent(
        id="demo:plant_component:DEMO:QAZ0001",
        facility_id="demo",
        code="DEMO:QAZ0001",
        control_node=control.id,
    )
    assert PlantComponent.model_validate_json(component.model_dump_json()) == component
    signal = FacilitySignal(
        id="demo:example",
        facility_id="demo",
        accessor="DEMO:STATUS",
        name_parse_outcome="function_variable",
        control_node=control.id,
    )
    assert FacilitySignal.model_validate_json(signal.model_dump_json()) == signal

from pathlib import Path

import yaml

_ROOT = Path(__file__).parents[2]


def _schema(name: str) -> dict:
    return yaml.safe_load((_ROOT / "imas_codex" / "schemas" / name).read_text())


def test_resolution_schema_has_typed_lifecycle_and_immutable_receipt() -> None:
    schema = _schema("standard_name.yaml")
    enums = schema["enums"]
    classes = schema["classes"]

    assert set(enums["DDResolutionStatus"]["permissible_values"]) == {
        "active",
        "retired_upstream",
        "withdrawn",
        "superseded",
    }
    assert set(enums["DDResolutionValueKind"]["permissible_values"]) == {
        "string",
        "string_list",
        "null",
    }
    assert set(enums["DDResolutionField"]["permissible_values"]) == {
        "unit",
        "documentation",
        "data_type",
        "node_type",
        "physics_domain",
        "cocos_transformation_type",
        "cocos_transformation_expression",
        "coordinates",
        "lifecycle_status",
        "lifecycle_version",
    }
    receipt = classes["DDResolutionStateChange"]["attributes"]
    assert receipt["id"]["identifier"] is True
    assert receipt["expected_manifest_digest"]["required"] is True
    assert receipt["expected_evidence_token"]["required"] is True
    assert receipt["graph_snapshot_token"]["required"] is True


def test_resolution_graph_record_carries_bridge_provenance() -> None:
    schema = _schema("standard_name.yaml")
    attributes = schema["classes"]["DDResolution"]["attributes"]

    for field in (
        "id",
        "path",
        "dd_version",
        "field",
        "published_kind",
        "published_value",
        "effective_kind",
        "effective_value",
        "reason",
        "recorded_by",
        "recorded_at",
        "upstream_reference",
        "upstream_commit_reference",
        "retiring_release",
        "source_manifest_digest",
        "status",
        "corrected_node",
        "evidence",
        "for_dd_version",
    ):
        assert attributes[field]["required"] is True
    assert attributes["for_dd_version"]["range"] == "string"
    assert (
        attributes["for_dd_version"]["annotations"]["relationship_type"]
        == "FOR_DD_VERSION"
    )
    assert attributes["for_dd_version"]["annotations"]["target_label"] == "DDVersion"


def test_raw_dd_node_and_resolution_declare_bridge_chain() -> None:
    standard_name = _schema("standard_name.yaml")
    imas_dd = _schema("imas_dd.yaml")

    node_slot = imas_dd["classes"]["IMASNode"]["attributes"]["dd_resolutions"]
    assert node_slot["range"] == "string"
    assert node_slot["annotations"] == {
        "relationship_type": "BRIDGED_BY",
        "target_label": "DDResolution",
    }

    evidence_slot = standard_name["classes"]["DDResolution"]["attributes"]["evidence"]
    assert evidence_slot["range"] == "string"
    assert evidence_slot["annotations"] == {
        "relationship_type": "EVIDENCED_BY",
        "target_label": "DDGap",
    }


def test_raw_dd_node_has_no_hidden_effective_resolution_fields() -> None:
    imas_dd = _schema("imas_dd.yaml")
    attributes = imas_dd["classes"]["IMASNode"]["attributes"]

    assert "effective_unit" not in attributes
    assert "resolved_unit" not in attributes
    assert "effective_documentation" not in attributes


def test_review_candidates_are_not_graph_writable_schema_types() -> None:
    schema = _schema("standard_name.yaml")

    assert "DDResolutionCandidate" not in schema["classes"]
    assert "DDResolutionCandidateManifest" not in schema["classes"]
    for class_definition in schema["classes"].values():
        for attribute in class_definition.get("attributes", {}).values():
            assert attribute.get("range") not in {
                "DDResolutionCandidate",
                "DDResolutionCandidateManifest",
            }

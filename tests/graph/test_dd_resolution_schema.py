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


def test_resolution_graph_mirror_carries_exact_authority_and_drift_digest() -> None:
    schema = _schema("standard_name.yaml")
    attributes = schema["classes"]["DDResolution"]["attributes"]

    for field in (
        "id",
        "gap_id",
        "path",
        "dd_version",
        "field",
        "observed_kind",
        "observed_value",
        "observed_hash",
        "effective_kind",
        "effective_value",
        "observation_ids",
        "evidence_token",
        "approved_by",
        "approved_at",
        "approval_receipt",
        "upstream_url",
        "upstream_ref",
        "manifest_digest",
        "status",
    ):
        assert attributes[field]["required"] is True
    assert attributes["for_dd_version"]["range"] == "DDVersion"
    assert (
        attributes["for_dd_version"]["annotations"]["relationship_type"]
        == "FOR_DD_VERSION"
    )
    assert (
        attributes["observations"]["annotations"]["relationship_type"]
        == "SUPPORTED_BY_OBSERVATION"
    )


def test_gap_and_raw_dd_node_link_to_resolution_mirror() -> None:
    standard_name = _schema("standard_name.yaml")
    imas_dd = _schema("imas_dd.yaml")

    gap_slot = standard_name["classes"]["DDGap"]["attributes"]["resolutions"]
    assert gap_slot["range"] == "DDResolution"
    assert gap_slot["annotations"]["relationship_type"] == "HAS_RESOLUTION"

    node_slot = imas_dd["classes"]["IMASNode"]["attributes"]["dd_resolutions"]
    assert node_slot["range"] == "DDResolution"
    assert node_slot["annotations"]["relationship_type"] == "HAS_DD_RESOLUTION"


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

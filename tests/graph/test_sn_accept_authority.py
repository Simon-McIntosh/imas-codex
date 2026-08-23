"""Accepted StandardName authority invariants.

An accepted name is justified in one of two ways: its own name-axis reviewer
score, or a durable structural authority naming the exact reviewed children
whose deterministic grammar peel entails the parent. The descriptive
``structural-inheritance`` reviewer marker is not authority by itself.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.graph.schema import GraphSchema

pytestmark = pytest.mark.graph

_STANDARD_NAME = "StandardName"
_REVIEW = "StandardNameReview"
_STRUCTURAL_AUTHORITY = "StructuralNameAuthority"
_REPAIR_AUTHORITY_ROW = "RepairAuthorityRow"
_STRUCTURAL_MARKER = "structural-inheritance"


def _slot_name(schema: GraphSchema, class_name: str, range_name: str) -> str:
    """Return the unique schema slot on a class with the requested range."""
    matches = [
        name
        for name, details in schema.get_all_slots(class_name).items()
        if details["type"] == range_name
    ]
    assert len(matches) == 1, (
        f"Expected one {class_name} slot with range {range_name}, got {matches}"
    )
    return matches[0]


def _relationship_type(schema: GraphSchema, class_name: str, slot_name: str) -> str:
    """Resolve a relationship type from the LinkML slot declaration."""
    matches = [
        relationship.cypher_type
        for relationship in schema.relationships
        if relationship.from_class == class_name and relationship.slot_name == slot_name
    ]
    assert len(matches) == 1, (
        f"Expected one relationship for {class_name}.{slot_name}, got {matches}"
    )
    return matches[0]


def _property_with_range(
    schema: GraphSchema, class_name: str, range_name: str, name_fragment: str
) -> str:
    """Resolve one scalar property by schema range and semantic name fragment."""
    matches = [
        name
        for name, details in schema.get_all_slots(class_name).items()
        if details["type"] == range_name
        and name_fragment in name
        and not details.get("relationship")
    ]
    assert len(matches) == 1, (
        f"Expected one {class_name} {range_name} property containing "
        f"{name_fragment!r}, got {matches}"
    )
    return matches[0]


def _accepted_count(graph_client: Any, label: str, stage_property: str) -> int:
    rows = graph_client.query(
        f"MATCH (sn:{label}) "
        f"WHERE sn.{stage_property} = $accepted "
        "RETURN count(sn) AS count",
        accepted="accepted",
    )
    return rows[0]["count"] if rows else 0


def test_structural_authority_contract_is_schema_declared(schema: GraphSchema) -> None:
    """The structural record extends signed authority and names its children."""
    assert _STRUCTURAL_AUTHORITY in schema.node_labels
    authority_class = schema._view.get_class(_STRUCTURAL_AUTHORITY)
    assert str(authority_class.is_a) == _REPAIR_AUTHORITY_ROW

    name_authority_slot = _slot_name(schema, _STANDARD_NAME, _STRUCTURAL_AUTHORITY)
    assert (
        _relationship_type(schema, _STANDARD_NAME, name_authority_slot)
        == "HAS_STRUCTURAL_AUTHORITY"
    )

    child_slot = _slot_name(schema, _STRUCTURAL_AUTHORITY, _STANDARD_NAME)
    assert (
        _relationship_type(schema, _STRUCTURAL_AUTHORITY, child_slot)
        == "ENTAILED_FROM_CHILD"
    )

    required = set(schema.get_required_fields(_STRUCTURAL_AUTHORITY))
    assert {
        "identity",
        "signatures",
        "participants",
        "selection",
        "mutations",
        "guards",
        "orphan_policy",
        "accepted_name_id",
        "child_ids",
        "children",
        "entailment_mechanism",
        "created_at",
        "code_identity",
        "schema_identity",
    } <= required

    child_ids = schema._view.induced_slot("child_ids", _STRUCTURAL_AUTHORITY)
    assert child_ids.multivalued
    assert child_ids.list_elements_ordered


def test_accepted_names_have_review_or_structural_authority(
    graph_client: Any, schema: GraphSchema
) -> None:
    """Every accepted name has its own score or a child-set authority record."""
    stage_property = _property_with_range(
        schema, _STANDARD_NAME, "NameStage", "name_stage"
    )
    score_property = _property_with_range(
        schema, _STANDARD_NAME, "float", "reviewer_score_name"
    )
    reviewer_property = _property_with_range(
        schema, _STANDARD_NAME, "string", "reviewer_model_name"
    )
    review_axis_property = _property_with_range(
        schema, _REVIEW, "StandardNameReviewMode", "review_axis"
    )
    assert "names" in schema.get_enums()["StandardNameReviewMode"]

    authority_slot = _slot_name(schema, _STANDARD_NAME, _STRUCTURAL_AUTHORITY)
    authority_relationship = _relationship_type(schema, _STANDARD_NAME, authority_slot)

    accepted = _accepted_count(graph_client, _STANDARD_NAME, stage_property)
    if accepted < 10:
        pytest.skip(
            f"Graph has only {accepted} accepted StandardName nodes (<10); "
            "populate the Standard Name corpus before checking authority."
        )

    rows = graph_client.query(
        f"""
        MATCH (sn:{_STANDARD_NAME})
        WHERE sn.{stage_property} = $accepted
        OPTIONAL MATCH (sn)-[:{authority_relationship}]->
                       (authority:{_STRUCTURAL_AUTHORITY})
        WITH sn, count(authority) > 0 AS has_structural_authority
        RETURN count(sn) AS accepted,
               sum(CASE WHEN sn.{score_property} IS NOT NULL
                        THEN 1 ELSE 0 END) AS scored,
               sum(CASE WHEN has_structural_authority
                        THEN 1 ELSE 0 END) AS structurally_authorized,
               sum(CASE WHEN sn.{score_property} IS NULL
                          AND NOT has_structural_authority
                        THEN 1 ELSE 0 END) AS residual,
               sum(CASE WHEN sn.{score_property} IS NULL
                          AND sn.{reviewer_property} = $structural_marker
                          AND NOT has_structural_authority
                        THEN 1 ELSE 0 END) AS bare_marker_only
        """,
        accepted="accepted",
        structural_marker=_STRUCTURAL_MARKER,
    )
    summary = rows[0]

    residual_rows = graph_client.query(
        f"""
        MATCH (sn:{_STANDARD_NAME})
        WHERE sn.{stage_property} = $accepted
          AND sn.{score_property} IS NULL
          AND NOT (sn)-[:{authority_relationship}]->
                  (:{_STRUCTURAL_AUTHORITY})
        RETURN sn.id AS id, sn.{reviewer_property} AS reviewer_marker
        ORDER BY id
        LIMIT 25
        """,
        accepted="accepted",
    )

    assert summary["residual"] == 0, (
        "Accepted-name authority invariant failed: "
        f"accepted={summary['accepted']}, scored={summary['scored']}, "
        f"structurally_authorized={summary['structurally_authorized']}, "
        f"residual={summary['residual']}. "
        f"Bare {_STRUCTURAL_MARKER!r} markers without authority="
        f"{summary['bare_marker_only']}; this is the structural-authority "
        "backfill size. First residual identities: "
        + ", ".join(row["id"] for row in residual_rows)
        + f". Schema name-review axis property: {_REVIEW}.{review_axis_property}."
    )

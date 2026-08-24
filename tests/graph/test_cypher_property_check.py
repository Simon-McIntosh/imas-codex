"""Specification for schema-checked Cypher property literals."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from imas_codex.graph.schema import GraphSchema

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARD_NAME_SCHEMA = REPO_ROOT / "imas_codex" / "schemas" / "standard_name.yaml"


def _checker_module() -> ModuleType:
    try:
        return importlib.import_module("imas_codex.graph.cypher_property_check")
    except ModuleNotFoundError as exc:
        if exc.name != "imas_codex.graph.cypher_property_check":
            raise
        raise AssertionError(
            "repository Cypher-property checking is not implemented: "
            "imas_codex.graph.cypher_property_check is missing"
        ) from None


def _review_axis_slot(schema: GraphSchema) -> tuple[str, str]:
    matches = [
        (label, slot_name)
        for label in schema.node_labels
        for slot_name, metadata in schema.get_all_slots(label).items()
        if metadata["type"] == "StandardNameReviewMode"
    ]
    assert len(matches) == 1, (
        "LinkML must expose exactly one property with the review-mode range; "
        f"found {matches}"
    )
    return matches[0]


def _audit(checker: ModuleType, *args: Any, **kwargs: Any) -> Any:
    audit = getattr(checker, "audit_cypher_properties", None)
    assert callable(audit), (
        "repository Cypher-property checking is not implemented: "
        "audit_cypher_properties is missing"
    )
    return audit(*args, **kwargs)


def test_misspelled_property_literal_is_reported_from_linkml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    schema = GraphSchema(STANDARD_NAME_SCHEMA)
    label, property_name = _review_axis_slot(schema)
    misspelled = f"{property_name}_misspelled"
    fixture = tmp_path / "query_fixture.py"
    fixture.write_text(
        "QUERY = '''\n"
        f"MATCH (review:{label})\n"
        f"WHERE review.{misspelled} = 'docs'\n"
        "RETURN count(review) AS count\n"
        "'''\n",
        encoding="utf-8",
    )

    requested_labels: list[str] = []
    get_all_slots = schema.get_all_slots

    def recording_get_all_slots(requested_label: str):
        requested_labels.append(requested_label)
        return get_all_slots(requested_label)

    monkeypatch.setattr(schema, "get_all_slots", recording_get_all_slots)
    report = _audit(_checker_module(), fixture, schemas=(schema,))

    assert report.checked_properties == 1
    assert len(report.violations) == 1
    violation = report.violations[0]
    assert violation.label == label
    assert violation.property_name == misspelled
    assert violation.path == fixture
    assert label in requested_labels, (
        "the declared-property universe must be read from GraphSchema.get_all_slots"
    )


def test_repository_cypher_literals_have_declared_properties() -> None:
    checker = _checker_module()
    schemas = tuple(
        GraphSchema(schema_path)
        for schema_path in sorted((REPO_ROOT / "imas_codex" / "schemas").glob("*.yaml"))
    )
    assert schemas
    assert any(schema.node_labels for schema in schemas), (
        "the LinkML class universe must be populated before a zero-violation result "
        "can be trusted"
    )

    report = _audit(checker, REPO_ROOT / "imas_codex", schemas=schemas)

    assert report.checked_properties > 0, (
        "a zero-violation result is not evidence unless properties were checked"
    )
    assert not report.violations, (
        "Cypher properties absent from their labels' LinkML slots:\n"
        + "\n".join(str(violation) for violation in report.violations)
    )
    incomplete_allowlist = [
        entry
        for entry in report.allowlisted
        if not entry.path or not entry.line or not entry.reason.strip()
    ]
    assert not incomplete_allowlist, (
        "every statically unresolved Cypher property needs a path, line, and reason: "
        f"{incomplete_allowlist}"
    )

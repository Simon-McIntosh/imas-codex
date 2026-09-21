"""The report-only mutation classifier reads clauses, not identifier parts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from imas_codex.cli.sn import _cypher_mutation_clause
from imas_codex.standard_names import protection

#: Line of the string literal opening the automatic-deletion protection query.
_PROTECTION_QUERY_LINE = 100


def _query_literal_at(path: Path, line: int) -> str:
    """Return the string literal passed to a ``.query(...)`` call at ``line``."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "query":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        value = node.args[0].value
        if isinstance(value, str) and node.args[0].lineno == line:
            return value
    raise AssertionError(f"no query string literal at {path}:{line}")


def _protection_query() -> str:
    return _query_literal_at(Path(protection.__file__), _PROTECTION_QUERY_LINE)


def test_read_only_protection_query_is_not_classified_as_a_mutation() -> None:
    query = _protection_query()
    # Guard the fixture: the identifier that drives the defect must be present,
    # otherwise this test could pass because the query changed, not the reader.
    assert "catalog_merge_commit_sha" in query
    assert _cypher_mutation_clause(query) is None


def test_identifier_part_named_merge_is_not_a_mutation() -> None:
    query = "MATCH (sn:StandardName) RETURN sn.catalog_merge_commit_sha, sn.exported_at"
    assert _cypher_mutation_clause(query) is None


@pytest.mark.parametrize(
    ("keyword", "statement"),
    [
        ("CREATE", "CREATE (sn:StandardName {id: $id})"),
        ("MERGE", "MERGE (sn:StandardName {id: $id})"),
        (
            "SET",
            "MATCH (sn:StandardName {id: $id}) SET sn.name_stage = 'approved'",
        ),
        ("DELETE", "MATCH (sn:StandardName {id: $id}) DELETE sn"),
        ("REMOVE", "MATCH (sn:StandardName {id: $id}) REMOVE sn.name_stage"),
        ("DROP", "DROP CONSTRAINT uniqueness_standard_name_id IF EXISTS"),
    ],
)
def test_genuine_mutation_statements_keep_their_clause(
    keyword: str, statement: str
) -> None:
    assert _cypher_mutation_clause(statement) == keyword


def test_first_mutation_clause_is_reported_in_document_order() -> None:
    statement = "MERGE (sn:StandardName {id: $id}) SET sn.name_stage = 'approved'"
    assert _cypher_mutation_clause(statement) == "MERGE"
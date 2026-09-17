"""The approval unwind restores every property the approval wrote.

The production Cypher of :func:`mark_catalog_name_approved` and
:func:`undo_approval` runs here against a real Cypher engine, inside one
explicit transaction that is rolled back before the test returns. The
statements execute for real, so a ``SET`` dropped from either function's
statements makes this test fail, while the transaction leaves nothing behind.

The identity is purpose-made and created inside that same transaction, so the
demonstration costs no graph state whether it passes or fails.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from imas_codex.graph import GraphClient
from imas_codex.standard_names.promote import (
    mark_catalog_name_approved,
    undo_approval,
)

pytestmark = pytest.mark.graph

APPROVAL_FIELDS = (
    "catalog_approved_at",
    "catalog_pr_number",
    "catalog_pr_url",
    "catalog_merge_commit_sha",
    "catalog_reviewer_actor",
)

RESTAMPED = "updated_at"

PR_NUMBER = 4242
PR_URL = "https://github.com/iterorganization/imas-standard-names-catalog/pull/4242"
MERGE_COMMIT = "0" * 40


class _TransactionGraph:
    """Runs each production statement on an explicit, never-committed transaction."""

    def __init__(self, tx: Any) -> None:
        self._tx = tx

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return [dict(record) for record in self._tx.run(cypher, **params)]

    def close(self) -> None:
        pass


def _properties(tx: Any, name: str) -> dict[str, Any]:
    record = tx.run(
        "MATCH (sn:StandardName {id: $id}) RETURN properties(sn) AS props",
        id=name,
    ).single()
    return dict(record["props"])


def _comparable(props: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in props.items() if key != RESTAMPED}


def _differing_keys(first: dict[str, Any], second: dict[str, Any]) -> set[str]:
    return {
        key for key in set(first) | set(second) if first.get(key) != second.get(key)
    }


def test_undo_approval_restores_every_property_the_approval_wrote() -> None:
    gc = GraphClient()
    name = f"unwind_proof_{uuid.uuid4().hex[:12]}"
    try:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                graph = _TransactionGraph(tx)
                tx.run(
                    "CREATE (sn:StandardName {id: $id, name: $id, "
                    "name_stage: 'accepted', docs_stage: 'accepted', "
                    "status: 'draft', validation_status: 'valid'})",
                    id=name,
                )
                before = _properties(tx, name)
                assert before["name_stage"] == "accepted"
                assert before["status"] == "draft"

                written = mark_catalog_name_approved(
                    name,
                    catalog_pr_number=PR_NUMBER,
                    catalog_pr_url=PR_URL,
                    catalog_merge_commit_sha=MERGE_COMMIT,
                    catalog_reviewer_actor="unwind-proof-reviewer",
                    gc=graph,
                )
                assert written is True

                approved = _properties(tx, name)
                for field in APPROVAL_FIELDS:
                    assert approved[field] is not None, field
                assert approved["name_stage"] == "approved"
                assert approved["status"] == "active"

                report = undo_approval(pr_number=PR_NUMBER, batch=[name], gc=graph)
                assert report.demoted == [name]

                after = _properties(tx, name)
                for field in APPROVAL_FIELDS:
                    assert after.get(field) is None, (
                        f"{field} survived the unwind: {after[field]!r}"
                    )
                assert after["name_stage"] == "accepted"
                assert after["status"] == "draft"

                assert _comparable(after) == _comparable(before)
                assert _differing_keys(before, after) == {RESTAMPED}

                one_left_set = dict(after)
                one_left_set["catalog_pr_number"] = PR_NUMBER
                assert _comparable(one_left_set) != _comparable(before)
            finally:
                tx.rollback()
    finally:
        gc.close()

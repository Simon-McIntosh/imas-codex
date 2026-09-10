"""The edits scope selects exactly the live identities carrying an open edit.

``sn run --edits`` must make every pool claim — and the dry-run selection
count that previews it — pick only StandardName identities whose
``edit_status`` is ``'open'``.  A scope that selects everything is not a
scope, so an otherwise-eligible identity WITHOUT an open edit must be
invisible to the edits-scoped surfaces while remaining visible to the
unscoped run.

Two surfaces are pinned here:

- The pending-count surface, ``_compute_pool_progress`` from the CLI —
  the number printed by ``sn run --only review --edits --dry-run``.  The
  fake applies the edits predicate only WHEN the code's generated Cypher
  carries it, so a regression that drops the predicate from the
  edits-scoped query changes the counted set and the assertions fail.
- The claim surface, the seed statement ``claim_review_name_batch`` builds
  — what the run actually asks the graph to claim.

Driven against in-memory node sets; no live Neo4j.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import pytest

from imas_codex.cli.sn import _compute_pool_progress
from imas_codex.standard_names import graph_ops


def _eligible_node(**overrides: Any) -> dict[str, Any]:
    """A name the review pool would otherwise claim: drafted, valid, given
    a real description and a non-derived origin."""
    node: dict[str, Any] = {
        "name_stage": "drafted",
        "validation_status": "valid",
        "description": "a real, non-placeholder description",
        "origin": "pipeline",
        "edit_status": None,
    }
    node.update(overrides)
    return node


class _ProgressGraph:
    """Text-aware fake serving ``_compute_pool_progress``' aggregate query.

    ``review_name`` pending counts the in-memory node set that satisfies
    the review eligibility, restricted to ``edit_status='open'`` exactly
    when the generated Cypher carries that predicate.  The test therefore
    measures the code's own query, not a reimplementation of it.
    """

    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.nodes = nodes

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        eligible = [
            node
            for node in self.nodes.values()
            if node.get("name_stage") == "drafted"
            and node.get("validation_status") == "valid"
            and node.get("description")
            and node.get("origin") != "derived"
        ]
        if "coalesce(sn.edit_status, '') = 'open'" in cypher:
            eligible = [node for node in eligible if node.get("edit_status") == "open"]
        return [
            {
                "generate_name": 0,
                "review_name": len(eligible),
                "refine_name": 0,
                "generate_docs": 0,
                "review_docs": 0,
                "refine_docs": 0,
                "enrich_parents": 0,
                "generate_name_done": 0,
                "review_name_done": 0,
                "refine_name_done": 0,
                "generate_docs_done": 0,
                "review_docs_done": 0,
                "refine_docs_done": 0,
                "enrich_parents_done": 0,
            }
        ]


def _pending_review(cypher_aware: _ProgressGraph, *, edits_only: bool) -> int:
    progress = _compute_pool_progress(
        cypher_aware,
        domains=None,
        rotation_cap=3,
        min_score=0.8,
        edits_only=edits_only,
    )
    return progress["review_name"]["pending"]


class TestEditsScopeSelectsOpenEditIdentities:
    def test_open_edit_identity_is_selected(self) -> None:
        graph = _ProgressGraph(
            {"pending_successor": _eligible_node(edit_status="open")}
        )
        assert _pending_review(graph, edits_only=True) == 1

    def test_mixed_set_counts_only_the_open_identity(self) -> None:
        graph = _ProgressGraph(
            {
                "pending_successor": _eligible_node(edit_status="open"),
                "reviewed_name": _eligible_node(edit_status=None),
            }
        )
        # The scope narrows the very set the unscoped run would see: with
        # both identities present the edits run counts only the open one.
        assert _pending_review(graph, edits_only=True) == 1
        assert _pending_review(graph, edits_only=False) == 2


class TestEditsScopeExcludesNoOpenEditIdentities:
    def test_identity_without_open_edit_is_not_selected(self) -> None:
        graph = _ProgressGraph({"reviewed_name": _eligible_node(edit_status=None)})
        assert _pending_review(graph, edits_only=True) == 0
        # The same identity is visible to the unscoped run — it is the
        # scope that excludes it, not the identity itself.
        assert _pending_review(graph, edits_only=False) == 1


class _RecordingTransaction:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.closed = False

    def run(self, statement: str, **parameters: Any) -> list[dict[str, Any]]:
        self.statements.append(statement)
        referenced = set(re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", statement))
        missing = referenced - parameters.keys()
        if missing:
            raise RuntimeError(f"missing Cypher parameters: {sorted(missing)}")
        if "RETURN c.id AS _cluster_id" in statement:
            return [{"_cluster_id": None, "_unit": None, "_physics_domain": None}]
        return [{"id": "pending_successor", "claim_token": parameters.get("token")}]

    def commit(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _RecordingSession:
    def __init__(self, transaction: _RecordingTransaction) -> None:
        self.transaction = transaction

    def __enter__(self) -> _RecordingSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def begin_transaction(self) -> _RecordingTransaction:
        return self.transaction


class _RecordingGraph:
    def __init__(self, transaction: _RecordingTransaction) -> None:
        self.transaction = transaction

    def __enter__(self) -> _RecordingGraph:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def query(self, statement: str, **parameters: Any) -> list[dict[str, Any]]:
        # Winner re-read after the claim: the claimed item still owns its token.
        ids = parameters.get("ids") or []
        return [{"id": item_id, "claim_seq": 1} for item_id in ids]

    def session(self) -> _RecordingSession:
        return _RecordingSession(self.transaction)


@pytest.mark.parametrize("edits_only", [True, False])
def test_review_claim_seed_carries_exactly_the_edits_predicate(
    edits_only: bool,
) -> None:
    """The claim the run actually makes carries the open-edit predicate when
    scoped and omits it when not — a scope that selected everything would
    never emit it."""
    transaction = _RecordingTransaction()
    graph = _RecordingGraph(transaction)
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        graph_ops.claim_review_name_batch(edits_only=edits_only, batch_size=1)
    seed = transaction.statements[0]
    assert ("coalesce(sn.edit_status, '') = 'open'" in seed) is edits_only


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

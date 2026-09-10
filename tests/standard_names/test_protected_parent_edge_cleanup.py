"""Superseded-child parent-edge cleanup consults the target's protection.

``rederive_structural_edges`` deletes operator-bearing ``HAS_PARENT`` edges
originating from superseded/exhausted children so dead names stop propping up
zombie parents. Those edges are also provenance for their parent: an edge INTO
a protected identity — recorded spend, ratification, a catalog binding — must
survive the pass even though the child that carries it is dead. A cleanup that
preserves the protected edge while still deleting ordinary dead edges is the
fix; a cleanup that stops deleting anything is not.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from imas_codex.standard_names import graph_ops

PROTECTED_PARENT = "protected_parent"
PROTECTED_CHILD = "superseded_child_of_protected"
UNPROTECTED_PARENT = "zombie_parent"
UNPROTECTED_CHILD = "superseded_child_of_zombie"


def _scripted_graph(*, dead_pairs: list[tuple[str, str]]) -> MagicMock:
    """A GraphClient double that serves the reconcile pass's own queries.

    The live-name discovery returns one live name so the pass proceeds past
    its early exit; the dead-edge candidate query returns *dead_pairs* (each
    ``(child, parent)``); the pair-filtered deletion reports no rows back.
    """
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = None

    def _query(cypher: str, **params) -> list[dict[str, object]]:
        if "RETURN sn.id AS id" in cypher and "EDGE_IDENTITY" not in cypher:
            return [{"id": "electron_temperature"}]
        if "RETURN c.id AS child, p.id AS parent" in cypher:
            return [
                {"child": child, "parent": parent}
                for child, parent in dead_pairs
            ]
        if "RETURN count(r)" in cypher:
            return [{"n": len(dead_pairs)}]
        return []

    graph.query = MagicMock(side_effect=_query)
    return graph


def _deletion_pairs(graph: MagicMock) -> list[tuple[str, str]]:
    """The ``(child, parent)`` pairs the pass proposed for HAS_PARENT deletion.

    An unfiltered ``DELETE r`` (no pair params) reports nothing here, which is
    how the pre-fix behavior reads as deleting the protected edge too.
    """
    for call in graph.query.call_args_list:
        if call.kwargs.get("pairs") and "HAS_PARENT" in call.args[0]:
            return [(pair["child"], pair["parent"]) for pair in call.kwargs["pairs"]]
    return []


def _drive_rederive(
    graph: MagicMock, *, protected_parents: set[str]
) -> tuple[dict[str, int], MagicMock]:
    """Run the structural redrive against the scripted graph.

    ``automatic_deletion_protections`` is the write path's protection question;
    it is spied (creating the attribute if the pre-fix code never imports it,
    so the test reproduces the defect rather than erroring out) and made to
    answer that exactly *protected_parents* carry durable authority.
    """
    with (
        patch.object(graph_ops, "GraphClient", return_value=graph),
        patch(
            "imas_codex.standard_names.graph_ops.automatic_deletion_protections",
            side_effect=lambda _gc, name_ids: {
                name_id: "unchanged_ratification"
                for name_id in name_ids
                if name_id in protected_parents
            },
            create=True,
        ) as consult_protection,
        patch.object(
            graph_ops, "_write_standard_name_edges", return_value=set()
        ),
        patch.object(graph_ops, "_rewire_has_parent_off_superseded", return_value=0),
    ):
        result = graph_ops.rederive_structural_edges()
    return result, consult_protection


def test_cleanup_consults_protection_before_deleting_a_superseded_child_edge():
    """An edge whose target carries protection survives the pass.

    The delete-path guard must ask the same protection question the write path
    asks, over the candidate edge targets, and must not remove an edge INTO a
    protected identity. Before the guard, the cleanup consulted nothing and the
    protected edge was swept with the dead edge — that state fails this test
    because the protection question is never asked.
    """
    graph = _scripted_graph(
        dead_pairs=[
            (PROTECTED_CHILD, PROTECTED_PARENT),
            (UNPROTECTED_CHILD, UNPROTECTED_PARENT),
        ]
    )

    result, consult_protection = _drive_rederive(
        graph, protected_parents={PROTECTED_PARENT}
    )

    assert consult_protection.call_count == 1
    assert consult_protection.call_args[0][1] == sorted(
        [PROTECTED_PARENT, UNPROTECTED_PARENT]
    )
    assert (PROTECTED_CHILD, PROTECTED_PARENT) not in _deletion_pairs(graph)
    assert result["dead_edges_cleared"] == 1


def test_cleanup_still_deletes_an_ordinary_dead_superseded_child_edge():
    """An ordinary dead edge to an unprotected target is still deleted.

    The same pass that preserves the protected edge must keep its legitimate
    job: a dead child whose parent carries no durable authority is reaped as
    before. A cleanup that stops deleting anything fails this assertion.
    """
    graph = _scripted_graph(
        dead_pairs=[
            (PROTECTED_CHILD, PROTECTED_PARENT),
            (UNPROTECTED_CHILD, UNPROTECTED_PARENT),
        ]
    )

    result, _consult_protection = _drive_rederive(
        graph, protected_parents={PROTECTED_PARENT}
    )

    assert _deletion_pairs(graph) == [(UNPROTECTED_CHILD, UNPROTECTED_PARENT)]
    assert result["dead_edges_cleared"] == 1

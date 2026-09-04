"""Deletion-ledger and conservative provenance-repair contracts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _no_approved_names():
    """Make the approval-backed protection lookup explicit for this suite."""
    with patch(
        "imas_codex.standard_names.protection._fetch_catalog_edit_names",
        return_value=set(),
    ):
        yield


def _queries(gc: MagicMock) -> list[str]:
    return [call.args[0] for call in gc.query.call_args_list]


def test_derived_parent_delete_records_change_atomically() -> None:
    """The parent and its deletion event are written in one graph statement."""
    from imas_codex.standard_names.graph_ops import _delete_derived_parent_nodes

    gc = MagicMock()
    gc.query.return_value = [{"deleted": 1}]

    assert _delete_derived_parent_nodes(gc, ["parent_name"]) == 1

    delete_query = next(
        q
        for q in _queries(gc)
        if "CREATE (change:StandardNameChange" in q and "DETACH DELETE sn" in q
    )
    assert "CREATE (change:StandardNameChange" in delete_query
    assert "operation: $deletion_operation" in delete_query
    assert "derived_sources" in delete_query
    assert "mirror_sources" in delete_query
    assert gc.query.call_args.kwargs["deletion_operation"] == "remove_derived_parent"


def test_skeleton_delete_records_change_atomically() -> None:
    """Placeholder cleanup keeps a durable explanation after the node is gone."""
    from imas_codex.standard_names.graph_ops import write_standard_names

    main = MagicMock()
    main.__enter__.return_value = main
    main.query.return_value = []
    sweep = MagicMock()
    sweep.__enter__.return_value = sweep
    sweep.query.return_value = [{"swept": 1}]

    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        side_effect=[main, sweep],
    ):
        write_standard_names(
            [
                {
                    "id": "electron_pressure",
                    "description": "A test quantity",
                    "documentation": None,
                    "kind": "scalar",
                    "unit": None,
                    "source_types": ["dd"],
                    "source_id": None,
                    "physics_domain": None,
                }
            ]
        )

    query = sweep.query.call_args.args[0]
    assert "CREATE (change:StandardNameChange" in query
    assert "DETACH DELETE sn" in query
    assert sweep.query.call_args.kwargs["deletion_operation"] == (
        "remove_skeleton_placeholder"
    )


def test_scoped_clear_records_each_deletion_in_same_statement() -> None:
    """A selected-name clear remains atomic with its deletion ledger writes."""
    from imas_codex.standard_names import graph_ops

    gc = MagicMock()
    gc.__enter__.return_value = gc

    def query(cypher: str, **_kwargs):
        if "count(" in cypher:
            return [{"n": 2}]
        return []

    gc.query.side_effect = query
    with patch.object(graph_ops, "GraphClient", return_value=gc):
        graph_ops.clear_standard_names(path_allowlist=["equilibrium/a"])

    delete_query = next(q for q in _queries(gc) if "DETACH DELETE sn" in q)
    assert "DELETE rel" in delete_query
    assert "CREATE (change:StandardNameChange" in delete_query
    assert delete_query.index("CREATE (change:StandardNameChange") < (
        delete_query.index("DETACH DELETE sn")
    )
    delete_call = next(
        call for call in gc.query.call_args_list if "DETACH DELETE sn" in call.args[0]
    )
    assert delete_call.kwargs["deletion_operation"] == "clear_selected_name"


def test_subsystem_clear_ledgers_names_but_preserves_change_rows() -> None:
    """A full pipeline wipe deletes names without deleting its explanation."""
    from imas_codex.standard_names import graph_ops

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.return_value = [{"n": 1}]

    with patch.object(graph_ops, "GraphClient", return_value=gc):
        graph_ops.clear_sn_subsystem()

    name_delete = next(
        call for call in gc.query.call_args_list if "DETACH DELETE sn" in call.args[0]
    )
    assert "CREATE (change:StandardNameChange" in name_delete.args[0]
    assert name_delete.kwargs["deletion_operation"] == "clear_subsystem_name"
    assert not any(
        "MATCH (change:StandardNameChange)" in q and "DELETE change" in q
        for q in _queries(gc)
    )


def test_compaction_ledgers_delete_with_owned_content() -> None:
    """Compaction records the deletion beside its atomic owned-content cleanup."""
    from imas_codex.standard_names.provenance_lifecycle import (
        compact_unapproved_superseded,
    )

    gc = MagicMock()
    gc.query.side_effect = [
        [
            {
                "name": "old_name",
                "prior_stage": "drafted",
                "tips": ["live_name"],
                "source_count": 1,
                "safe_to_compact": True,
            }
        ],
        [{"moved": 1}],
        [{"deleted": 1}],
    ]

    manifest = compact_unapproved_superseded(gc, apply=True)

    assert manifest[0]["compacted"] is True
    delete_query = next(q for q in _queries(gc) if "DETACH DELETE old" in q)
    assert "CREATE (change:StandardNameChange" in delete_query
    assert "FOREACH (item IN reviews | DETACH DELETE item)" in delete_query
    assert "FOREACH (item IN revisions | DETACH DELETE item)" in delete_query
    delete_call = next(
        call for call in gc.query.call_args_list if "DETACH DELETE old" in call.args[0]
    )
    assert delete_call.kwargs["deletion_operation"] == "compact_unapproved_name"


def test_missing_change_targets_are_classified_without_deleting_history() -> None:
    """Missing targets are reported as explained deletion or historical residue."""
    from imas_codex.standard_names.provenance_lifecycle import (
        classify_missing_change_targets,
    )

    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "sn-change:known",
            "to_name": "removed",
            "operation": "clear_selected_name",
            "classification": "explained_deletion",
        },
        {
            "id": "sn-change:legacy",
            "to_name": "old_target",
            "operation": "rename",
            "classification": "legacy_unexplained",
        },
    ]

    report = classify_missing_change_targets(gc)

    assert report["total"] == 2
    assert report["explained_deletion"] == 1
    assert report["legacy_unexplained"] == 1
    assert report["rows"][1]["id"] == "sn-change:legacy"
    query = gc.query.call_args.args[0]
    assert "DETACH DELETE" not in query

"""Receipt-backed fold revivals regain a draft catalog status."""

from unittest.mock import MagicMock

from imas_codex.standard_names.graph_store.graph_audits import reconcile_catalog_status


def test_reconcile_revived_status_requires_a_matching_fold_receipt() -> None:
    graph = MagicMock()
    graph.query.side_effect = [
        [{"changed": 0}],
        [{"changed": 0}],
        [{"changed": 1}],
        [{"changed": 0}],
    ]

    result = reconcile_catalog_status(gc=graph)

    assert result == {
        "drafted": 0,
        "revived": 1,
        "superseded": 0,
        "quarantined": 0,
        "deprecated": 0,
        "total_changed": 1,
    }
    assert graph.query.call_count == 4
    terminal_stage = graph.query.call_args_list[0].args[0]
    exhausted_stage = graph.query.call_args_list[1].args[0]
    revived = graph.query.call_args_list[2].args[0]
    assert "sn.name_stage = 'superseded'" in terminal_stage
    assert "sn.name_stage = 'exhausted'" in exhausted_stage
    assert "sn.name_stage IS NOT NULL" in revived
    assert "sn.name_stage <> 'superseded'" in revived
    assert "sn.name_stage <> 'exhausted'" in revived
    assert "sn.status = 'superseded'" in revived
    assert "AND EXISTS {" in revived
    assert "(sn)-[:HAS_INTERNAL_CHANGE]->" in revived
    assert "(change:StandardNameChange {operation: 'fold_identity'})" in revived
    assert "change.to_name = sn.id" in revived
    assert "SET sn.status = 'draft'" in revived

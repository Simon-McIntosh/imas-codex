"""A detachment written during a run must record that run's identifier.

``reconcile_attachment_consistency`` writes a ``StandardNameChange`` with
``operation='detach_inconsistent_attachment'`` for every attachment it strips.
Unlike the documentation-refine path (``operation='refine'``), which has
carried ``run_id`` on every change it writes since the field existed, the
detach path never threaded a run identifier through — so every detachment
recorded by it is untraceable to the run that performed it. These tests cover
the threading (``run_id`` reaches the persisted change) and the census that
measures how much of the existing record is still unattributed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.standard_names.attachment_audit import (
    count_unattributed_detachments,
    reconcile_attachment_consistency,
)

_DD_PATH = "summary/boundary/strike_point_inner_z/value"
_SN_ID = "z_image_up_unit_vector_of_camera"


def _row(dd_path: str, sn_id: str, *, name_stage: str = "drafted") -> dict:
    return {
        "source_node_id": f"dd:{dd_path}",
        "dd_path": dd_path,
        "sn_id": sn_id,
        "name_stage": name_stage,
        "origin": "pipeline",
        "dd_unit": None,
        "sn_unit": None,
        "other_live_names": 0,
    }


def _client(rows: list[dict]) -> MagicMock:
    """A client answering the audit read with *rows* and every write with a count."""
    from imas_codex.standard_names import attachment_audit as mod

    def _query(q: str, **params):
        if "AS other_live_names" in q:
            if (want := params.get("sn_id")) is not None:
                return [r for r in rows if r["sn_id"] == want]
            return rows
        if q == mod._DETACH_QUERY:
            return [{"detached": len(params.get("items") or [])}]
        return []

    gc = MagicMock()
    gc.query.side_effect = _query
    return gc


def test_detach_records_the_run_that_performed_it() -> None:
    """The recorded change carries the caller's ``run_id``, not ``None``."""
    gc = _client([_row(_DD_PATH, _SN_ID)])

    result = reconcile_attachment_consistency(gc, run_id="sn-run:pytest-attribution")

    assert result.detached == 1
    change_calls = [
        call
        for call in gc.query.call_args_list
        if "CREATE (change:StandardNameChange" in call.args[0]
    ]
    assert len(change_calls) == 1
    assert change_calls[0].kwargs["run_id"] == "sn-run:pytest-attribution"
    assert change_calls[0].kwargs["operation"] == "detach_inconsistent_attachment"


def test_detach_without_a_run_scope_still_records_none_explicitly() -> None:
    """No run scope is still an explicit, threaded absence — not a dropped arg."""
    gc = _client([_row(_DD_PATH, _SN_ID)])

    reconcile_attachment_consistency(gc)

    change_calls = [
        call
        for call in gc.query.call_args_list
        if "CREATE (change:StandardNameChange" in call.args[0]
    ]
    assert len(change_calls) == 1
    assert change_calls[0].kwargs["run_id"] is None


def test_census_counts_only_unattributed_detach_changes() -> None:
    """The census reads the graph directly and ignores other operations/run states."""
    gc = MagicMock()
    gc.query.return_value = [{"n": 508}]

    assert count_unattributed_detachments(gc) == 508

    query, params = gc.query.call_args
    assert "detach_inconsistent_attachment" in query[0]
    assert "run_id IS NULL" in query[0]
    assert params == {}


def test_census_reports_zero_once_every_detachment_is_attributed() -> None:
    gc = MagicMock()
    gc.query.return_value = [{"n": 0}]

    assert count_unattributed_detachments(gc) == 0

"""A minted identity always carries a catalog status, and a null is healed.

Every StandardName the pipeline creates must leave its creation with
``status='draft'`` — null is outside the enumerated catalog-status vocabulary
and blocks the export gate. Three write-side sites are covered here: the
first-generation binding-reservation mint, the refine-successor mint, and the
ordinary governed write, which must heal a row whose status is already null
rather than leaving it.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

_GO = "imas_codex.standard_names.graph_ops"


def _mock_graph():
    """A mock GraphClient: empty query results, a usable session/transaction.

    ``gc.query`` answers every read with no rows, and the transaction used by
    ``persist_refined_name`` returns a single persisted successor row, which
    is all that function consumes from it.
    """
    mock_gc = MagicMock()
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)
    mock_gc.query.return_value = []

    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_tx.closed = True
    mock_tx.run.return_value = [{"new_name": "refined", "old_name": "original"}]
    mock_session.begin_transaction.return_value = mock_tx
    mock_gc.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_gc.session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_gc, mock_session, mock_tx


def _cypher_matching(gc_or_tx, fragment: str) -> str:
    """Return the first executed Cypher containing *fragment*."""
    for call_item in gc_or_tx.query.call_args_list:
        if call_item.args and fragment in call_item.args[0]:
            return call_item.args[0]
    raise AssertionError(f"no executed query contained {fragment!r}")


def _run_matching(run_mock, fragment: str) -> str:
    """Return the first ``tx.run`` Cypher containing *fragment*."""
    for call_item in run_mock.call_args_list:
        if call_item.args and fragment in call_item.args[0]:
            return call_item.args[0]
    raise AssertionError(f"no transaction run contained {fragment!r}")


def _assigned(cypher: str, target: str, value: str) -> bool:
    """True when *cypher* assigns ``<target> = <value>`` at a SET site.

    Cypher assignments are column-aligned in the source, so the comparison is
    whitespace-insensitive rather than literal.
    """
    return re.search(
        re.escape(target) + r"\s*=\s*" + re.escape(value), cypher
    ) is not None


def test_binding_reservation_mints_target_with_status_draft() -> None:
    """A first-generation mint through the binding reservation sets status."""
    from imas_codex.standard_names.graph_ops import _lock_claimed_name_bindings

    gc, _session, _tx = _mock_graph()
    batch = [
        {
            "sns_id": "dd:equilibrium/time_slice/profiles_1d/psi",
            "claim_token": "tok-1",
            "claim_seq": 1,
            "sn_id": "poloidal_flux",
            "source_type": "dd",
            "unit": "Wb",
        }
    ]

    _lock_claimed_name_bindings(
        gc,
        batch,
        allow_missing=True,
        allow_own_pending_reservation=True,
    )

    cypher = _cypher_matching(gc, "MERGE (target:StandardName {id: b.sn_id})")
    assert _assigned(cypher, "target.status", "'draft'")


def test_refine_successor_minted_with_status_draft() -> None:
    """A refine-successor creation through persist_refined_name sets status."""
    from imas_codex.standard_names.graph_ops import persist_refined_name

    mock_gc, _session, mock_tx = _mock_graph()

    with patch(f"{_GO}.GraphClient", return_value=mock_gc):
        persist_refined_name(
            old_name="original",
            new_name="refined",
            description="refined description",
            run_id="run-1",
        )

    cypher = _run_matching(mock_tx.run, "MERGE (new:StandardName {id: $new_name})")
    assert _assigned(cypher, "new.status", "'draft'")


def test_derived_parent_materialization_sets_status_draft() -> None:
    """A bootstrapped derived-parent mint also carries the catalog status.

    Derived parents are minted by the structural write path; leaving their
    ``status`` unset reproduces the null the export gate refuses, so the
    materializer's SET clause fills it the same way every other write does.
    """
    from imas_codex.standard_names.graph_ops import _materialize_derived_parent_rows

    gc, _session, _tx = _mock_graph()
    rows = [
        {
            "parent_id": "magnetic_field",
            "child_data": [
                {
                    "id": "x_magnetic_field",
                    "unit": "T",
                    "edge_kinds": ["projection"],
                }
            ],
            "edge_kinds": ["projection"],
            "authorized_unit": "T",
            "description": "(deterministic parent)",
        }
    ]

    _materialize_derived_parent_rows(gc, rows, bootstrap_missing=True)

    cypher = _cypher_matching(gc, "MERGE (parent:StandardName {id: $parent_id})")
    assert _assigned(cypher, "parent.status", "coalesce(parent.status, 'draft')")


def test_ordinary_write_heals_null_status_and_sets_draft_on_create() -> None:
    """write_standard_names sets draft on create and fills a pre-existing null.

    The plain ``SET`` clause carries ``status = coalesce(status, 'draft')`` so
    a governed write over a row whose status is already null repairs it — the
    sanctioned backfill — while a non-null status survives untouched.
    """
    from imas_codex.standard_names.graph_ops import write_standard_names

    gc, _session, _tx = _mock_graph()
    names = [
        {
            "id": "electron_temperature",
            "source_id": "core_profiles/profiles_1d/electrons/temperature",
            "source_types": ["dd"],
            "kind": "scalar",
            "unit": "eV",
        }
    ]

    with (
        patch(f"{_GO}.GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.protection.filter_protected",
            side_effect=lambda names, **kwargs: (names, []),
        ),
    ):
        write_standard_names(names, gc=gc)

    cypher = _cypher_matching(gc, "MERGE (sn:StandardName {id: b.id})")
    assert _assigned(cypher, "sn.status", "'draft'")
    assert _assigned(cypher, "sn.status", "coalesce(sn.status, 'draft')")

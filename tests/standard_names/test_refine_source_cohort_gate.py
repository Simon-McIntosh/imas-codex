"""Refined-name persistence requires authoritative source provenance."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.attachment_audit import AttachmentPairingGuardResult
from imas_codex.standard_names.graph_ops import (
    RefinedNamePersistenceRefusal,
    RefinedNamePersistenceRefusalReason,
    persist_refined_name,
)


def test_empty_authoritative_source_cohort_refuses_successor() -> None:
    transaction = MagicMock()

    def run(cypher: str, **params):
        if "// REFINE_ATOMIC_PREFLIGHT" in cypher:
            return [
                {
                    "old_name": params["old_name"],
                    "new_name": params["new_name"],
                    "source_ids": [],
                    "source_edit_mode": None,
                    "source_name_hint": None,
                    "source_docs_hint": None,
                    "source_edit_reason": None,
                    "source_edit_origin": None,
                    "source_edit_scope": None,
                    "source_edit_status": None,
                    "source_edit_requested_at": None,
                    "source_edit_override_edits": None,
                    "source_edit_include_accepted": None,
                    "effective_edit_mode": None,
                    "effective_name_hint": None,
                    "effective_docs_hint": None,
                    "effective_edit_reason": None,
                    "effective_edit_origin": None,
                    "effective_edit_scope": None,
                    "effective_edit_status": None,
                    "effective_edit_requested_at": None,
                    "effective_edit_override_edits": None,
                    "effective_edit_include_accepted": None,
                }
            ]
        if "MERGE (new)-[:REFINED_FROM]->(old)" in cypher:
            return [{"old_name": params["old_name"], "new_name": params["new_name"]}]
        raise AssertionError(f"unexpected transaction query: {cypher}")

    transaction.run.side_effect = run
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = False
    session = MagicMock()
    session.begin_transaction.return_value = transaction
    graph.session.return_value.__enter__.return_value = session
    graph.session.return_value.__exit__.return_value = False

    admitted = AttachmentPairingGuardResult((), ())
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph),
        patch(
            "imas_codex.standard_names.attachment_audit.guard_source_pairings",
            return_value=admitted,
        ) as guard,
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "retarget_standard_name_sources",
            return_value=0,
        ) as retarget,
        patch(
            "imas_codex.standard_names.provenance_lifecycle.record_standard_name_change"
        ) as record,
        patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter"),
    ):
        with pytest.raises(RefinedNamePersistenceRefusal) as refusal:
            persist_refined_name(
                old_name="electron_density",
                new_name="volume_averaged_electron_density",
                description="Electron density averaged over volume.",
                unit="m^-3",
            )

    assert (
        refusal.value.reason
        is RefinedNamePersistenceRefusalReason.AUTHORITATIVE_SOURCE_COHORT_EMPTY
    )
    assert transaction.run.call_count == 1
    transaction.rollback.assert_called_once_with()
    transaction.commit.assert_not_called()
    guard.assert_not_called()
    retarget.assert_not_called()
    record.assert_not_called()

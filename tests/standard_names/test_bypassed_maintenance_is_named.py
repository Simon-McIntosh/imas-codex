"""Scoped maintenance bypasses remain deliberate and leave a receipt."""

from __future__ import annotations

import logging

import pytest

from imas_codex.standard_names.loop import summary_table
from tests.standard_names.test_scoped_global_maintenance import _run_loop

_LOOP_LOGGER = "imas_codex.standard_names.loop"

_BYPASSED_PASSES = (
    "reconcile_standard_name_sources",
    "reconcile_vocab_gaps",
    "revive_unit_skipped_sources",
    "retry_vocab_gap_sources_on_grammar_change",
    "reconcile_provenance",
    "reconcile_source_status_liveness",
    "retire_unreachable_hint_edits",
    "reconcile_grammar_segments",
    "reconcile_catalog_status",
    "reconcile_reviewable_name_stage",
    "reconcile_standard_name_cocos_links",
    "reconcile_dd_unit_corrections",
    "reconcile_standard_name_unit_edges",
    "reconcile_standard_name_dd_edges",
    "reconcile_standard_name_source_paths",
    "refresh_drifted_sources",
    "promote_stranded_reviewed",
    "rederive_structural_edges",
    "seed_parent_sources",
    "normalize_derived_parent_lifecycle",
    "structural_accept_derived_parents",
    "reconcile_orphan_parent_sources",
    "release_all_orphan_claims",
    "resolve_doc_links",
)


@pytest.mark.asyncio
async def test_scoped_run_names_every_bypassed_maintenance_pass(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=_LOOP_LOGGER)

    summary, *_ = await _run_loop(skip_global_maintenance=True)

    expected = list(_BYPASSED_PASSES)
    assert summary.bypassed_maintenance_passes == expected
    assert summary_table(summary)["bypassed_maintenance_passes"] == expected

    messages = [record.getMessage() for record in caplog.records]
    for function_name in expected:
        assert (
            f"run_sn_pools: global maintenance bypassed — {function_name}" in messages
        )

    summary_message = next(
        message
        for message in messages
        if message.startswith("run_sn_pools: bypassed global maintenance summary")
    )
    assert f"count={len(expected)}" in summary_message
    assert ", ".join(expected) in summary_message

"""An edits-scoped ``sn run`` skips global maintenance when asked.

``sn run`` accepts ``--skip-global-maintenance`` only for a run already bounded
by ``--focus/--batch``, ``--name`` or ``--scope-run-id`` — the case it was
written for is a run scoped to an edit's own run id, whose edits must not
escalate into a graph-wide reconcile pass.  The flag is carried by
``_run_sn_cmd``: it suppresses the CLI's own graph-wide startup steps and is
forwarded to ``run_sn_pools``, which is where the mutating maintenance set is
silenced.

The hop pinned here is the forwarding inside ``_run_sn_cmd`` — the CLI's
``--scope-run-id`` wiring into ``_run_sn_cmd`` is covered elsewhere, and the
loop's handling of ``skip_global_maintenance`` is covered by the scoped
maintenance suite.  Removing the forwarding leaves ``run_sn_pools`` on the
default ``False`` and this test fails on a known-present control.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from imas_codex.cli.sn import _run_sn_cmd

_EDIT_RUN_ID = "sn-edit-scope-probe"


def _run_discovery(_config, async_main):
    """Drive the pool orchestrator without the discovery harness."""
    return asyncio.run(async_main(asyncio.Event(), MagicMock()))


def test_edits_scoped_run_skips_global_maintenance() -> None:
    """A run scoped to an edit forwards the bypass and skips global startup."""
    pools = AsyncMock(return_value=MagicMock(stop_reason="completed"))

    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch("imas_codex.cli.sn._require_terminal_drain"),
        patch(
            "imas_codex.cli.sn._configured_local_compose_required",
            return_value=False,
        ),
        patch("imas_codex.cli.sn._auto_sync_grammar") as grammar_sync,
        patch("imas_codex.cli.sn._note_pipeline_version_drift") as drift_note,
        patch(
            "imas_codex.cli.discover.common.use_rich_output",
            return_value=False,
        ),
        patch("imas_codex.cli.discover.common.setup_logging"),
        patch(
            "imas_codex.cli.discover.common.run_discovery",
            side_effect=_run_discovery,
        ),
        patch("imas_codex.standard_names.loop.run_sn_pools", new=pools),
        patch(
            "imas_codex.standard_names.loop.summary_table",
            return_value={"stop_reason": "completed"},
        ),
    ):
        row = _run_sn_cmd(
            cost_limit=5.0,
            per_domain_limit=None,
            dry_run=False,
            quiet=True,
            source="dd",
            scope_run_id=_EDIT_RUN_ID,
            edits_only=True,
            skip_global_maintenance=True,
        )

    assert row == {"stop_reason": "completed"}
    # The CLI's own graph-wide startup steps are the first maintenance a scoped
    # run must skip; they run only when the flag is unset.
    grammar_sync.assert_not_called()
    drift_note.assert_not_called()
    # Known-present controls: the recorder reached the orchestrator seam with
    # the edit's scope before the bypass is judged.
    pools.assert_awaited_once()
    kwargs = pools.await_args.kwargs
    assert kwargs["scope_run_id"] == _EDIT_RUN_ID
    assert kwargs["edits_only"] is True
    assert kwargs["skip_global_maintenance"] is True

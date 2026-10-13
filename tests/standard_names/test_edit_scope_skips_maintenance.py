"""An edits-scoped ``sn run`` skips global maintenance when asked.

``sn run`` accepts ``--skip-global-maintenance`` only for a run already bounded
by ``--focus/--batch``, ``--name`` or ``--scope-run-id`` — the case it was
written for is a run scoped to an edit's own run id, whose edits must not
escalate into a graph-wide reconcile pass.

The bypass is judged at two seams and this test observes both by driving the
real ``sn run`` command:

* ``sn_run`` owns the CLI's graph-wide startup steps and gates them on
  ``skip_global_maintenance``; the command runs them only when the flag is
  unset.
* ``_run_sn_cmd`` forwards the flag to ``run_sn_pools``, where the mutating
  maintenance set is silenced.

Removing either behaviour turns its assertion red: the startup steps run
unconditionally, or ``run_sn_pools`` falls back to the default ``False``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn

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
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--scope-run-id",
                _EDIT_RUN_ID,
                "--edits",
                "--skip-global-maintenance",
                "--quiet",
            ],
        )

    assert result.exit_code == 0, result.output
    # The CLI's own graph-wide startup steps are the first maintenance a scoped
    # run must skip; sn_run runs them only when the flag is unset.
    grammar_sync.assert_not_called()
    drift_note.assert_not_called()
    # Known-present controls: the run reached the orchestrator seam carrying
    # the edit's scope, the bypass, and the edits predicate, before the skip
    # can be judged.
    pools.assert_awaited_once()
    kwargs = pools.await_args.kwargs
    assert kwargs["scope_run_id"] == _EDIT_RUN_ID
    assert kwargs["edits_only"] is True
    assert kwargs["skip_global_maintenance"] is True

"""Tests for ``sn run`` embedding preflight checks."""

from __future__ import annotations

from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import _require_embed_ready, _run_sn_cmd, sn


def test_require_embed_ready_allows_healthy_embed() -> None:
    """Healthy embedding service should not block ``sn run``."""
    with patch(
        "imas_codex.discovery.base.services.embed_health_check",
        return_value=(True, "iter"),
    ):
        _require_embed_ready("sn run")


def test_require_embed_ready_raises_actionable_error() -> None:
    """Unhealthy embedding service should fail fast with recovery guidance."""
    with patch(
        "imas_codex.discovery.base.services.embed_health_check",
        return_value=(False, "server unavailable"),
    ):
        with pytest.raises(click.ClickException) as exc_info:
            _require_embed_ready("sn run")

    message = str(exc_info.value)
    assert "Embedding server is required for `sn run`" in message
    assert "embed status" in message
    assert "embed start" in message


def test_run_sn_cmd_checks_embed_before_starting() -> None:
    """The pool orchestrator must preflight embed health before running."""
    with (
        patch("imas_codex.cli.sn._require_embed_ready") as require_embed,
        patch("imas_codex.cli.discover.common.use_rich_output", return_value=False),
        patch("imas_codex.cli.discover.common.setup_logging", return_value=None),
        patch(
            "imas_codex.cli.discover.common.run_discovery",
            return_value={"summary": None},
        ),
    ):
        _run_sn_cmd(
            cost_limit=1.0,
            time_limit=None,
            per_domain_limit=None,
            dry_run=False,
            quiet=True,
        )

    require_embed.assert_called_once_with("sn run")


def test_run_sn_cmd_skips_embed_preflight_for_dry_run() -> None:
    """Dry runs should not require a live embedding service."""
    with (
        patch("imas_codex.cli.sn._require_embed_ready") as require_embed,
        patch("imas_codex.cli.discover.common.use_rich_output", return_value=False),
        patch("imas_codex.cli.discover.common.setup_logging", return_value=None),
        patch(
            "imas_codex.cli.discover.common.run_discovery",
            return_value={"summary": None},
        ),
    ):
        _run_sn_cmd(
            cost_limit=1.0,
            time_limit=None,
            per_domain_limit=None,
            dry_run=True,
            quiet=True,
        )

    require_embed.assert_not_called()


def test_focus_run_preflights_before_graph_mutation() -> None:
    """Focused runs must fail before any graph mutation when embed is down."""
    runner = CliRunner()
    with (
        patch(
            "imas_codex.cli.sn._require_embed_ready",
            side_effect=click.ClickException("embed down"),
        ),
        patch("imas_codex.graph.client.GraphClient") as graph_client,
        patch("imas_codex.cli.sn._run_sn_cmd") as run_pools,
    ):
        result = runner.invoke(
            sn,
            [
                "run",
                "--skip-clear-gate",
                "--focus",
                "equilibrium/time_slice/global_quantities/ip",
            ],
        )

    assert result.exit_code == 1
    assert "embed down" in result.output
    graph_client.assert_not_called()
    run_pools.assert_not_called()


def test_signals_reset_preflights_before_reset_mutation() -> None:
    """Signals-source reset paths must fail before reset mutations."""
    runner = CliRunner()
    with (
        patch(
            "imas_codex.cli.sn._require_embed_ready",
            side_effect=click.ClickException("embed down"),
        ),
        patch(
            "imas_codex.standard_names.graph_ops.clear_standard_names"
        ) as clear_names,
        patch(
            "imas_codex.standard_names.graph_ops.reset_standard_names"
        ) as reset_names,
    ):
        result = runner.invoke(
            sn,
            [
                "run",
                "--skip-clear-gate",
                "--source",
                "signals",
                "--facility",
                "tcv",
                "--reset-to",
                "extracted",
            ],
        )

    assert result.exit_code == 1
    assert "embed down" in result.output
    clear_names.assert_not_called()
    reset_names.assert_not_called()

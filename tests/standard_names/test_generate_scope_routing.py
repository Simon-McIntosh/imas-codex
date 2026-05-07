"""Tests for ``sn generate`` scope-routing rules.

Validates that the CLI routes to the rotator or pool adapter
based on the source type.

No LLM calls — all external dependencies are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn


@pytest.fixture()
def runner():
    return CliRunner()


# Patch target for the rotator entrypoint in the CLI module.
_ROTATOR = "imas_codex.cli.sn._run_sn_loop_cmd"
# The single-pass path reaches ``run_explicit_paths`` via a local import
# inside the CLI function.  Patching the source module blocks it
# effectively because the import resolves each invocation.
_SINGLE_PASS = "imas_codex.standard_names.pool_adapter.run_explicit_paths"


async def _async_noop(*args, **kwargs):  # pragma: no cover - test helper
    return None


class TestScopeRouting:
    """``sn generate`` routes to rotator vs single-pass correctly."""

    def test_default_routes_to_rotator(self, runner):
        """No --source override → rotator."""
        with patch(_ROTATOR) as mock_rot:
            result = runner.invoke(sn, ["run", "-c", "0.01", "-q"])
            assert mock_rot.called, f"Rotator not called. Output: {result.output}"

    def test_physics_domain_routes_to_rotator(self, runner):
        """--domain without signals source → rotator (scoped single-domain)."""
        with patch(_ROTATOR) as mock_rot:
            result = runner.invoke(
                sn,
                ["run", "--domain", "equilibrium", "-c", "0.01", "-q"],
            )
            assert mock_rot.called, (
                f"Rotator not called with --domain. Output: {result.output}"
            )

    def test_signals_source_routes_to_single_pass(self, runner):
        """--source signals → pool adapter (rotator is DD-only)."""
        with patch(_ROTATOR) as mock_rot, patch(_SINGLE_PASS, side_effect=_async_noop):
            runner.invoke(
                sn,
                [
                    "run",
                    "--source",
                    "signals",
                    "--facility",
                    "tcv",
                    "-c",
                    "0.01",
                    "-q",
                ],
            )
            assert not mock_rot.called, (
                "Rotator should NOT be called with --source signals"
            )

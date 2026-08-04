"""Test ``sn --help`` omits removed bare reconcile, link, and seed verbs.

The legacy standalone verbs are removed. Governed maintenance commands may use
more specific names with mandatory manifest and audit inputs.
"""

from __future__ import annotations

import re

from click.testing import CliRunner

from imas_codex.cli.sn import sn


class TestSnHelpNoLegacyVerbs:
    """sn --help must not show removed verbs."""

    def test_no_reconcile_in_help(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert result.exit_code == 0
        command_lines = result.output.partition("Commands:")[2].splitlines()
        listed_commands = {
            match.group(1)
            for line in command_lines
            if (match := re.match(r"^  ([a-z0-9][a-z0-9-]*)\s{2,}", line))
        }
        assert "reconcile" not in listed_commands
        assert "reconcile-grammar-segments" in listed_commands

    def test_no_resolve_links_as_command(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert result.exit_code == 0
        # Check there's no "resolve-links" or "link" as a top-level subcommand
        lines = result.output.splitlines()
        command_lines = [ln.strip() for ln in lines if ln.strip().startswith("resolve")]
        # Should be empty — resolve-links is no longer a verb
        assert not command_lines, f"Found resolve-links as a command: {command_lines}"

    def test_no_seed_command(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert result.exit_code == 0
        lines = result.output.splitlines()
        command_lines = [ln.strip() for ln in lines if ln.strip().startswith("seed")]
        assert not command_lines, f"Found seed as a command: {command_lines}"

    def test_reconcile_verb_is_error(self):
        """Invoking ``sn reconcile`` should fail (unknown command)."""
        runner = CliRunner()
        result = runner.invoke(sn, ["reconcile"])
        assert result.exit_code != 0

    def test_resolve_links_verb_is_error(self):
        """Invoking ``sn resolve-links`` should fail (unknown command)."""
        runner = CliRunner()
        result = runner.invoke(sn, ["resolve-links"])
        assert result.exit_code != 0

    def test_seed_verb_is_error(self):
        """Invoking ``sn seed`` should fail (unknown command)."""
        runner = CliRunner()
        result = runner.invoke(sn, ["seed"])
        assert result.exit_code != 0

"""Test ``sn --help`` lists the catalog workflow CLI verbs.

Verifies that ``preview``, ``release``, and ``import`` appear as
subcommands in the ``sn`` group help. The standalone ``export`` verb was
folded into ``sn release --export-only``.
"""

from __future__ import annotations

from click.testing import CliRunner

from imas_codex.cli.sn import sn


class TestSnHelpNewVerbs:
    """sn --help must list the catalog workflow verbs."""

    def _get_help(self) -> str:
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert result.exit_code == 0
        return result.output

    def test_export_command_retired(self):
        # Folded into `sn release --export-only`.
        runner = CliRunner()
        result = runner.invoke(sn, ["export", "--help"])
        assert result.exit_code != 0

    def test_preview_in_help(self):
        assert "preview" in self._get_help()

    def test_release_in_help(self):
        assert "release" in self._get_help()

    def test_import_in_help(self):
        assert "import" in self._get_help()

    def test_publish_not_in_help(self):
        assert "publish" not in self._get_help()

    def test_release_export_only_shows_staging(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "--help"])
        assert result.exit_code == 0
        assert "--export-only" in result.output
        assert "--staging" in result.output

    def test_preview_help_shows_port(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["preview", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_release_help_shows_message(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["release", "--help"])
        assert result.exit_code == 0
        assert "--message" in result.output
        assert "--bump" in result.output
        assert "--final" in result.output

    def test_import_help_shows_override_flags(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["import", "--help"])
        assert result.exit_code == 0
        assert "--accept-unit-override" in result.output

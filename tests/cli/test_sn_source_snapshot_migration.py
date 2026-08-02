"""CLI safety coverage for bounded source maintenance."""

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


def test_reviewer_help_names_only_configured_profiles() -> None:
    runner = CliRunner()

    run_help = runner.invoke(sn, ["run", "--help"])
    review_help = runner.invoke(sn, ["review", "--help"])

    assert run_help.exit_code == 0, run_help.output
    assert review_help.exit_code == 0, review_help.output
    assert "quality-cost-balanced" in run_help.output
    assert "quality-cost-balanced" in review_help.output
    assert "--reviewer-profile pilot" not in run_help.output
    assert "--reviewer-profile pilot" not in review_help.output


def test_bounded_drain_refuses_explicit_single_reviewer_profile() -> None:
    with (
        patch(
            "imas_codex.settings.get_sn_review_profile_models",
            return_value=["openrouter/anthropic/one-reviewer"],
        ),
        patch(
            "imas_codex.standard_names.sources_manifest.resolve_batch_token"
        ) as resolve,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--drain-batch",
                "bounded.json",
                "--reviewer-profile",
                "opus-only",
                "--dry-run",
            ],
        )

    assert result.exit_code != 0
    assert "at least two reviewer models" in result.output
    resolve.assert_not_called()


def test_bounded_drain_refuses_environment_single_reviewer_profile() -> None:
    with (
        patch(
            "imas_codex.settings.get_sn_review_profile_models",
            return_value=["openrouter/anthropic/one-reviewer"],
        ),
        patch(
            "imas_codex.standard_names.sources_manifest.resolve_batch_token"
        ) as resolve,
    ):
        result = CliRunner().invoke(
            sn,
            ["run", "--drain-batch", "bounded.json", "--dry-run"],
            env={"IMAS_CODEX_SN_REVIEW_PROFILE": "opus-only"},
        )

    assert result.exit_code != 0
    assert "at least two reviewer models" in result.output
    resolve.assert_not_called()

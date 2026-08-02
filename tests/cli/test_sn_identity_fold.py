"""CLI coverage for folding one standard-name identity into another."""

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


def _fold_result(*, prior_stage: str, already_superseded: bool) -> dict[str, object]:
    return {
        "ok": True,
        "old_id": "old_identity",
        "into_id": "canonical_identity",
        "old_prior_stage": prior_stage,
        "already_superseded": already_superseded,
        "sources_carried": 0,
        "sources_would_strand": 0,
        "attachments_rejected": 0,
    }


def test_supersede_reports_actual_prior_stage_in_dry_run() -> None:
    result = _fold_result(prior_stage="reviewed", already_superseded=False)

    with patch(
        "imas_codex.standard_names.edit.supersede_into", return_value=result
    ) as supersede_into:
        invocation = CliRunner().invoke(
            sn,
            ["supersede", "old_identity", "--into", "canonical_identity", "--dry-run"],
        )

    assert invocation.exit_code == 0, invocation.output
    assert (
        "old prior stage: reviewed → superseded "
        "(superseded_from_stage=reviewed)" in invocation.output
    )
    assert "superseded_from_stage=accepted" not in invocation.output
    supersede_into.assert_called_once_with(
        "old_identity", "canonical_identity", dry_run=True
    )


def test_supersede_idempotent_replay_reports_preserved_prior_stage() -> None:
    result = _fold_result(prior_stage="exhausted", already_superseded=True)

    with patch(
        "imas_codex.standard_names.edit.supersede_into", return_value=result
    ) as supersede_into:
        invocation = CliRunner().invoke(
            sn, ["supersede", "old_identity", "--into", "canonical_identity"]
        )

    assert invocation.exit_code == 0, invocation.output
    assert "already superseded" in invocation.output
    assert (
        "old prior stage: exhausted → superseded "
        "(superseded_from_stage=exhausted)" in invocation.output
    )
    assert "superseded_from_stage=accepted" not in invocation.output
    supersede_into.assert_called_once_with(
        "old_identity", "canonical_identity", dry_run=False
    )

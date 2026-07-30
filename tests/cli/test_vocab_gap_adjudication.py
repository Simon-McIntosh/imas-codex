"""The vocabulary editorial command is dry-run first and explicit on writes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def test_vocab_adjudicate_defaults_to_dry_run(tmp_path: Path) -> None:
    from imas_codex.cli.sn import sn

    artifact = tmp_path / "decisions.json"
    artifact.write_text("{}")
    batch = MagicMock()
    result_payload = {
        "rows": 159,
        "changed": 159,
        "unchanged": 0,
        "counts": {"add": 18, "fold": 27, "reject": 114},
        "grammar_signature": "abc123",
        "grammar_version": "0.8",
        "dry_run": True,
    }
    with (
        patch(
            "imas_codex.standard_names.vocab_adjudication.load_vocab_gap_adjudications",
            return_value=batch,
        ) as load,
        patch(
            "imas_codex.standard_names.vocab_adjudication.apply_vocab_gap_adjudications",
            return_value=result_payload,
        ) as apply,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "vocab-adjudicate",
                "--actor",
                "catalog review",
                str(artifact),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "dry-run: rows=159 changed=159 unchanged=0" in result.output
    assert "add=18 fold=27 reject=114" in result.output
    load.assert_called_once_with(artifact)
    apply.assert_called_once_with(batch, actor="catalog review", dry_run=True)


def test_vocab_adjudicate_apply_is_explicit(tmp_path: Path) -> None:
    from imas_codex.cli.sn import sn

    artifact = tmp_path / "decisions.json"
    artifact.write_text("{}")
    payload = {
        "rows": 1,
        "changed": 1,
        "unchanged": 0,
        "counts": {"add": 0, "fold": 1, "reject": 0},
        "grammar_signature": "abc123",
        "grammar_version": "0.8",
        "dry_run": False,
    }
    with (
        patch(
            "imas_codex.standard_names.vocab_adjudication.load_vocab_gap_adjudications",
            return_value=MagicMock(),
        ),
        patch(
            "imas_codex.standard_names.vocab_adjudication.apply_vocab_gap_adjudications",
            return_value=payload,
        ) as apply,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "vocab-adjudicate",
                "--actor",
                "catalog review",
                "--apply",
                str(artifact),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "applied: rows=1 changed=1 unchanged=0" in result.output
    assert apply.call_args.kwargs["dry_run"] is False


def test_vocab_adjudicate_reset_requires_a_reason() -> None:
    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(
        sn,
        [
            "vocab-adjudicate",
            "--actor",
            "catalog review",
            "--reset-signature",
            "old-signature",
        ],
    )
    assert result.exit_code != 0
    assert "--reason is required" in result.output

"""Tests for ``sn run`` CLI flag wiring.

Verifies that:
- The rotation-control flags (--min-score, --rotation-cap, --escalation-model,
  --review-name-backlog-cap, --review-docs-backlog-cap) are accepted and
  forwarded with the correct defaults.
- Retired flags (--regen, --enrich, --target, --skip-regen, --skip-enrich,
  --name-only-batch-size, --docs-status, --docs-batch-size) have been
  removed and raise Click UsageError when supplied.
- Without ``--pool`` filter, all 6 pools start (default behaviour).
- ``--physics-domain`` is forwarded.

No graph or LLM access — ``_run_sn_cmd`` is patched to capture kwargs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.defaults import (
    DEFAULT_ESCALATION_MODEL,
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
    REVIEW_DOCS_BACKLOG_CAP,
    REVIEW_NAME_BACKLOG_CAP,
)


def test_release_refine_claim_help_describes_fenced_dry_run() -> None:
    result = CliRunner().invoke(
        sn, ["release-refine-claim", "--help"], terminal_width=200
    )

    assert result.exit_code == 0
    assert "zero-write" in result.output
    assert "dry run" in result.output
    assert "exact claim token" in result.output
    assert "--manifest-sha256" in result.output
    assert "refining" in result.output


def test_release_refine_claim_default_emits_json_dry_run() -> None:
    receipt = {
        "outcome": "would_release",
        "dry_run": True,
        "eligible": True,
        "would_release": 1,
        "changed": 0,
        "manifest_sha256": "a" * 64,
    }
    with patch(
        "imas_codex.standard_names.graph_ops.release_stranded_refine_claim",
        return_value=receipt,
    ) as release:
        result = CliRunner().invoke(
            sn,
            [
                "release-refine-claim",
                "electron_temperature",
                "--claim-token",
                "exact-token",
                "--expected-stage",
                "refining",
                "--scope-run-id",
                "exact-scope",
            ],
        )

    assert result.exit_code == 0
    assert result.output.strip() == (
        '{"changed":0,"dry_run":true,"eligible":true,'
        '"manifest_sha256":"' + "a" * 64 + '","outcome":"would_release",'
        '"would_release":1}'
    )
    release.assert_called_once_with(
        "electron_temperature",
        claim_token="exact-token",
        expected_stage="refining",
        scope_run_id="exact-scope",
        apply=False,
        manifest_sha256=None,
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--apply"], "--apply requires --scope-run-id"),
        (
            ["--apply", "--scope-run-id", "exact-scope"],
            "--apply requires --manifest-sha256",
        ),
    ],
)
def test_release_refine_claim_apply_requires_scope_and_manifest(
    extra: list[str], message: str
) -> None:
    result = CliRunner().invoke(
        sn,
        [
            "release-refine-claim",
            "electron_temperature",
            "--claim-token",
            "exact-token",
            "--expected-stage",
            "refining",
            *extra,
        ],
    )

    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.parametrize(
    "args",
    [
        [
            "release-refine-claim",
            "electron_temperature",
            "--expected-stage",
            "refining",
            "--apply",
        ],
        [
            "release-refine-claim",
            "electron_temperature",
            "--claim-token",
            "exact-token",
            "--expected-stage",
            "reviewed",
        ],
    ],
)
def test_release_refine_claim_rejects_missing_token_or_wrong_stage(
    args: list[str],
) -> None:
    result = CliRunner().invoke(sn, args)

    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_ARGS = [
    "run",
    "--skip-clear-gate",  # skip gate so we don't need a running graph
    "--dry-run",  # prevent actual pipeline start
]


def _invoke(*extra_args: str) -> tuple[object, dict | None]:
    """Invoke ``sn run`` with *extra_args* appended to _MINIMAL_ARGS.

    Returns ``(result, captured_kwargs)`` where *captured_kwargs* is the
    kwargs dict passed to ``_run_sn_cmd`` (or ``None`` if not reached).
    """
    runner = CliRunner()
    captured: dict = {}

    def _fake_loop_cmd(**kwargs):
        captured.update(kwargs)

    with patch("imas_codex.cli.sn._run_sn_cmd", side_effect=_fake_loop_cmd):
        result = runner.invoke(
            sn, _MINIMAL_ARGS + list(extra_args), catch_exceptions=False
        )

    return result, captured if captured else None


def test_required_docs_admission_flag_is_forwarded_for_exact_recovery_pool() -> None:
    result, captured = _invoke(
        "--only",
        "review_docs",
        "--scope-run-id",
        "priced-scope",
        "--require-docs-admission",
    )

    assert result.exit_code == 0
    assert captured is not None
    assert captured["require_docs_admission"] is True
    assert captured["scope_run_id"] == "priced-scope"


@pytest.mark.parametrize(
    "args",
    [
        ("--require-docs-admission",),
        (
            "--require-docs-admission",
            "--scope-run-id",
            "priced-scope",
            "--only",
            "review_name",
        ),
        (
            "--require-docs-admission",
            "--scope-run-id",
            "priced-scope",
            "--only",
            "review_docs",
            "--focus",
            "path/to/name",
        ),
    ],
)
def test_required_docs_admission_rejects_ambiguous_invocation(
    args: tuple[str, ...],
) -> None:
    result, _captured = _invoke(*args)

    assert result.exit_code == 2
    assert "--require-docs-admission" in result.output


# ---------------------------------------------------------------------------
# 1. test_default_min_score
# ---------------------------------------------------------------------------


def test_default_min_score():
    """Without --min-score the default comes from DEFAULT_MIN_SCORE."""
    _, kwargs = _invoke()
    assert kwargs is not None, "_run_sn_cmd was not called"
    assert kwargs["min_score"] == DEFAULT_MIN_SCORE, (
        f"Expected DEFAULT_MIN_SCORE={DEFAULT_MIN_SCORE}, got {kwargs['min_score']}"
    )


# ---------------------------------------------------------------------------
# 2. test_min_score_override
# ---------------------------------------------------------------------------


def test_min_score_override():
    """--min-score 0.85 is forwarded to _run_sn_cmd."""
    _, kwargs = _invoke("--min-score", "0.85")
    assert kwargs is not None
    assert abs(kwargs["min_score"] - 0.85) < 1e-9, (
        f"Expected 0.85, got {kwargs['min_score']}"
    )


# ---------------------------------------------------------------------------
# 3. test_default_rotation_cap
# ---------------------------------------------------------------------------


def test_default_rotation_cap():
    """Without --rotation-cap the default comes from DEFAULT_REFINE_ROTATIONS."""
    _, kwargs = _invoke()
    assert kwargs is not None
    assert kwargs["rotation_cap"] == DEFAULT_REFINE_ROTATIONS, (
        f"Expected {DEFAULT_REFINE_ROTATIONS}, got {kwargs['rotation_cap']}"
    )


# ---------------------------------------------------------------------------
# 4. test_rotation_cap_override
# ---------------------------------------------------------------------------


def test_rotation_cap_override():
    """--rotation-cap 5 is forwarded correctly."""
    _, kwargs = _invoke("--rotation-cap", "5")
    assert kwargs is not None
    assert kwargs["rotation_cap"] == 5, f"Expected 5, got {kwargs['rotation_cap']}"


# ---------------------------------------------------------------------------
# 5. test_escalation_model_default
# ---------------------------------------------------------------------------


def test_escalation_model_default():
    """Without --escalation-model the default is DEFAULT_ESCALATION_MODEL."""
    _, kwargs = _invoke()
    assert kwargs is not None
    assert kwargs["escalation_model"] == DEFAULT_ESCALATION_MODEL, (
        f"Expected {DEFAULT_ESCALATION_MODEL!r}, got {kwargs['escalation_model']!r}"
    )


# ---------------------------------------------------------------------------
# 6. test_escalation_model_override
# ---------------------------------------------------------------------------


def test_escalation_model_override():
    """--escalation-model <value> is forwarded correctly."""
    custom_model = "openrouter/anthropic/claude-sonnet-4-5"
    _, kwargs = _invoke("--escalation-model", custom_model)
    assert kwargs is not None
    assert kwargs["escalation_model"] == custom_model, (
        f"Expected {custom_model!r}, got {kwargs['escalation_model']!r}"
    )


# ---------------------------------------------------------------------------
# 7. test_backlog_caps_defaults_and_overrides
# ---------------------------------------------------------------------------


def test_backlog_caps_defaults():
    """Without explicit backlog caps, defaults are forwarded."""
    _, kwargs = _invoke()
    assert kwargs is not None
    assert kwargs["review_name_backlog_cap"] == REVIEW_NAME_BACKLOG_CAP, (
        f"Expected {REVIEW_NAME_BACKLOG_CAP}, got {kwargs['review_name_backlog_cap']}"
    )
    assert kwargs["review_docs_backlog_cap"] == REVIEW_DOCS_BACKLOG_CAP, (
        f"Expected {REVIEW_DOCS_BACKLOG_CAP}, got {kwargs['review_docs_backlog_cap']}"
    )


def test_backlog_caps_overrides():
    """--review-name-backlog-cap and --review-docs-backlog-cap are forwarded."""
    _, kwargs = _invoke(
        "--review-name-backlog-cap",
        "50",
        "--review-docs-backlog-cap",
        "75",
    )
    assert kwargs is not None
    assert kwargs["review_name_backlog_cap"] == 50, (
        f"Expected 50, got {kwargs['review_name_backlog_cap']}"
    )
    assert kwargs["review_docs_backlog_cap"] == 75, (
        f"Expected 75, got {kwargs['review_docs_backlog_cap']}"
    )


# ---------------------------------------------------------------------------
# 8. test_no_obsolete_flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obsolete_flag",
    [
        "--regen",
        "--enrich",
        "--target",
        "--skip-regen",
        "--skip-enrich",
        "--name-only-batch-size",
        "--docs-status",
        "--docs-batch-size",
    ],
)
def test_no_obsolete_flags(obsolete_flag: str):
    """Obsolete flags raise a Click UsageError (no such option)."""
    runner = CliRunner()
    # Provide a dummy value to avoid "option requires an argument" masking the
    # "no such option" error for flags that take a value.
    args = _MINIMAL_ARGS + [obsolete_flag, "dummy"]
    with patch("imas_codex.cli.sn._run_sn_cmd"):
        result = runner.invoke(sn, args, catch_exceptions=True)
    assert result.exit_code != 0, (
        f"Obsolete flag {obsolete_flag!r} should have caused a non-zero exit"
    )
    assert "no such option" in (result.output or "").lower() or result.exit_code == 2, (
        f"Expected 'no such option' error for {obsolete_flag!r}, "
        f"got exit_code={result.exit_code}, output={result.output!r}"
    )


# ---------------------------------------------------------------------------
# 9. test_default_targets_all_six_pools
# ---------------------------------------------------------------------------


def test_default_targets_all_six_pools():
    """Without a pool filter, _run_sn_cmd is called (all 6 pools start).

    Verifies:
    - ``skip_generate`` is False (generate_name + refine_name active)
    - ``skip_review`` is False (review_name + review_docs active)
    - generate_docs and refine_docs pools are always active in loop mode
    """
    _, kwargs = _invoke()
    assert kwargs is not None, "_run_sn_cmd was not called"
    # No pool-skipping flags were provided, so all pools are enabled
    assert not kwargs.get("skip_generate", True), (
        "generate_name pool should be active by default"
    )
    assert not kwargs.get("skip_review", True), (
        "review pools should be active by default"
    )


# ---------------------------------------------------------------------------
# 10. test_physics_domain_filter
# ---------------------------------------------------------------------------


def test_physics_domain_filter():
    """--domain is forwarded as domains to _run_sn_cmd."""
    _, kwargs = _invoke("--domain", "equilibrium")
    assert kwargs is not None
    assert "equilibrium" in kwargs.get("domains", ()), (
        f"Expected 'equilibrium' in domains, got {kwargs.get('domains')!r}"
    )

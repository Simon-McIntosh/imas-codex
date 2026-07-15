"""Tests for the ``sn edit`` CLI and the mirrored ``edit_standard_name`` MCP tool.

Both surfaces are thin wrappers over
:func:`imas_codex.standard_names.edit.apply_edit` — the engine itself is
tested elsewhere. Both the CLI and the MCP tool import ``apply_edit``
function-locally (a module-level import from either surface closes an
import cycle: edit -> graph_ops -> discovery -> cli.logging -> cli/__init__
register_commands -> cli.sn), so these tests patch the single source,
``imas_codex.standard_names.edit.apply_edit``, rather than a
surface-local name. They verify argument wiring, mutual-exclusion /
mandatory-reason validation, and rendering of the returned ``EditPlan``.

No graph or LLM access — the autouse ``_block_live_graph`` fixture in
``conftest.py`` would raise if a real ``GraphClient`` were touched.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.edit import EditPlan


def _plan(**overrides) -> EditPlan:
    base = {
        "target": "electron_temperature",
        "mode": "hint",
        "axis": "name",
        "scope": "only_self",
        "entry": "generate",
        "successor": None,
        "cascade_planned": [],
        "blocked": None,
        "actions": ["hint attached to 'electron_temperature' (axis=name)"],
        "applied": True,
    }
    base.update(overrides)
    return EditPlan(**base)


# ---------------------------------------------------------------------------
# CLI: mode / reason validation
# ---------------------------------------------------------------------------


def test_no_mode_flag_is_usage_error():
    runner = CliRunner()
    result = runner.invoke(sn, ["edit", "electron_temperature", "--reason", "because"])
    assert result.exit_code != 0
    assert "exactly one of" in (result.output or "").lower()


def test_two_mode_flags_is_usage_error():
    runner = CliRunner()
    result = runner.invoke(
        sn,
        [
            "edit",
            "electron_temperature",
            "--hint",
            "clarify",
            "--rename",
            "ion_temperature",
            "--reason",
            "because",
        ],
    )
    assert result.exit_code != 0
    assert "exactly one of" in (result.output or "").lower()


def test_missing_reason_is_usage_error():
    runner = CliRunner()
    result = runner.invoke(sn, ["edit", "electron_temperature", "--hint", "clarify"])
    assert result.exit_code != 0
    assert "--reason" in (result.output or "")


def test_blank_reason_is_usage_error():
    runner = CliRunner()
    result = runner.invoke(
        sn,
        [
            "edit",
            "electron_temperature",
            "--hint",
            "clarify",
            "--reason",
            "   ",
        ],
    )
    assert result.exit_code != 0
    assert "--reason" in (result.output or "")


# ---------------------------------------------------------------------------
# CLI: scope mapping
# ---------------------------------------------------------------------------


def test_scope_self_maps_to_only_self():
    runner = CliRunner()
    with patch(
        "imas_codex.standard_names.edit.apply_edit", return_value=_plan()
    ) as mock_apply:
        result = runner.invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--hint",
                "clarify",
                "--reason",
                "because",
                "--scope",
                "self",
                "--stage-only",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_apply.assert_called_once()
    assert mock_apply.call_args.kwargs["scope"] == "only_self"


def test_scope_family_and_subtree_pass_through():
    runner = CliRunner()
    for scope_flag, expected in (("family", "family"), ("subtree", "subtree")):
        with patch(
            "imas_codex.standard_names.edit.apply_edit", return_value=_plan()
        ) as mock_apply:
            result = runner.invoke(
                sn,
                [
                    "edit",
                    "electron_temperature",
                    "--hint",
                    "clarify",
                    "--reason",
                    "because",
                    "--scope",
                    scope_flag,
                    "--stage-only",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_apply.call_args.kwargs["scope"] == expected


def test_default_scope_is_none():
    """No --scope means apply_edit resolves the default itself."""
    runner = CliRunner()
    with patch(
        "imas_codex.standard_names.edit.apply_edit", return_value=_plan()
    ) as mock_apply:
        result = runner.invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--hint",
                "clarify",
                "--reason",
                "because",
                "--stage-only",
            ],
        )
    assert result.exit_code == 0, result.output
    assert mock_apply.call_args.kwargs["scope"] is None


# ---------------------------------------------------------------------------
# CLI: rendering
# ---------------------------------------------------------------------------


def test_dry_run_banner_rendered():
    plan = _plan(applied=False, entry="review_name", mode="rename", axis="name")
    runner = CliRunner()
    with patch("imas_codex.standard_names.edit.apply_edit", return_value=plan):
        result = runner.invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--rename",
                "ion_temperature",
                "--reason",
                "because",
                "--dry-run",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    with patch(
        "imas_codex.standard_names.edit.apply_edit", return_value=plan
    ) as mock_apply:
        runner.invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--rename",
                "ion_temperature",
                "--reason",
                "because",
                "--dry-run",
            ],
        )
    assert mock_apply.call_args.kwargs["dry_run"] is True


def test_blocked_plan_exits_2_with_message():
    plan = _plan(
        blocked="renaming the shared segment would desync 3 sibling(s)",
        applied=False,
        actions=[],
    )
    runner = CliRunner()
    with patch("imas_codex.standard_names.edit.apply_edit", return_value=plan):
        result = runner.invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--rename",
                "ion_temperature",
                "--reason",
                "because",
            ],
        )
    assert result.exit_code == 2
    assert "desync 3 sibling" in result.output
    assert "BLOCKED" in result.output


def test_successful_apply_prints_successor_and_followthrough_hint():
    plan = _plan(
        mode="rename",
        axis="name",
        entry="review_name",
        successor="ion_temperature",
        applied=True,
    )
    runner = CliRunner()
    with patch("imas_codex.standard_names.edit.apply_edit", return_value=plan):
        result = runner.invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--rename",
                "ion_temperature",
                "--reason",
                "because",
                "--stage-only",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "ion_temperature" in result.output
    assert "sn run" in result.output
    assert "sn status" in result.output


def test_cascade_table_rendered_for_family_scope():
    plan = _plan(
        mode="rename",
        scope="family",
        entry="review_name",
        successor="ion_temperature",
        cascade_planned=[{"from": "electron_temperature_a", "to": "ion_temperature_a"}],
    )
    runner = CliRunner()
    with patch("imas_codex.standard_names.edit.apply_edit", return_value=plan):
        result = runner.invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--rename",
                "ion_temperature",
                "--reason",
                "because",
                "--scope",
                "family",
                "--stage-only",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "electron_temperature_a" in result.output
    assert "ion_temperature_a" in result.output


def test_value_error_from_engine_becomes_usage_error():
    runner = CliRunner()
    with patch(
        "imas_codex.standard_names.edit.apply_edit",
        side_effect=ValueError("rename mode requires a non-empty `rename` value"),
    ):
        result = runner.invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--rename",
                "ion_temperature",
                "--reason",
                "because",
            ],
        )
    assert result.exit_code != 0
    assert "non-empty" in (result.output or "")


def test_help_works_without_graph():
    runner = CliRunner()
    result = runner.invoke(sn, ["edit", "--help"])
    assert result.exit_code == 0
    assert "--hint" in result.output
    assert "--rename" in result.output
    assert "--docs" in result.output
    assert "--reason" in result.output
    assert "--axis" in result.output
    assert "--scope" in result.output
    assert "--dry-run" in result.output


# ---------------------------------------------------------------------------
# MCP tool: _edit_standard_name
# ---------------------------------------------------------------------------


class TestEditStandardNameTool:
    def test_returns_dict_with_expected_keys(self):
        from imas_codex.llm.sn_tools import _edit_standard_name

        plan = _plan()
        with patch("imas_codex.standard_names.edit.apply_edit", return_value=plan):
            result = _edit_standard_name(
                "electron_temperature", "because", hint="clarify"
            )
        assert isinstance(result, dict)
        for key in ("mode", "entry", "blocked", "target", "successor", "applied"):
            assert key in result
        assert "summary" in result

    def test_passes_origin_agent(self):
        from imas_codex.llm.sn_tools import _edit_standard_name

        with patch(
            "imas_codex.standard_names.edit.apply_edit", return_value=_plan()
        ) as mock_apply:
            _edit_standard_name("electron_temperature", "because", hint="clarify")
        assert mock_apply.call_args.kwargs["origin"] == "agent"

    def test_scope_mapping(self):
        from imas_codex.llm.sn_tools import _edit_standard_name

        with patch(
            "imas_codex.standard_names.edit.apply_edit", return_value=_plan()
        ) as mock_apply:
            _edit_standard_name(
                "electron_temperature", "because", hint="clarify", scope="self"
            )
        assert mock_apply.call_args.kwargs["scope"] == "only_self"

    def test_invalid_scope_returns_error_dict(self):
        from imas_codex.llm.sn_tools import _edit_standard_name

        result = _edit_standard_name(
            "electron_temperature", "because", hint="clarify", scope="bogus"
        )
        assert "error" in result

    def test_value_error_from_engine_returns_error_dict(self):
        from imas_codex.llm.sn_tools import _edit_standard_name

        with patch(
            "imas_codex.standard_names.edit.apply_edit",
            side_effect=ValueError("apply_edit requires a non-empty reason"),
        ):
            result = _edit_standard_name("electron_temperature", "because")
        assert "error" in result
        assert "non-empty reason" in result["error"]

    def test_blocked_plan_summary_mentions_blocked(self):
        from imas_codex.llm.sn_tools import _edit_standard_name

        plan = _plan(blocked="target not found", applied=False)
        with patch("imas_codex.standard_names.edit.apply_edit", return_value=plan):
            result = _edit_standard_name("unknown_name", "because", hint="clarify")
        assert result["blocked"] == "target not found"
        assert "BLOCKED" in result["summary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

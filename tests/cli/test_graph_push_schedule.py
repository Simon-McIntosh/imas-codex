"""The graph push command owns scheduled verified cycles."""

from __future__ import annotations

from click.testing import CliRunner

from imas_codex.cli.graph import graph
from imas_codex.cli.graph.registry import graph_push


def test_push_help_lists_schedule_and_cycle() -> None:
    result = CliRunner().invoke(graph_push, ["--help"])

    assert result.exit_code == 0
    help_lines = result.output.splitlines()
    assert any(
        "--schedule" in line and "Submit the weekly self-resubmitting push job." in line
        for line in help_lines
    )
    assert any(
        "--cycle" in line and "Run one verified full-graph push cycle." in line
        for line in help_lines
    )


def test_graph_has_no_separate_offsite_push_command() -> None:
    result = CliRunner().invoke(graph, ["offsite-push", "--help"])

    assert result.exit_code != 0
    assert "No such command 'offsite-push'" in result.output


def test_schedule_and_cycle_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(graph_push, ["--schedule", "--cycle"])

    assert result.exit_code != 0
    assert "--schedule and --cycle are mutually exclusive" in result.output

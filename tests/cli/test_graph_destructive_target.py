"""Target identity guards for destructive graph commands."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.graph.data import graph_clear, graph_load
from imas_codex.graph.profiles import resolve_neo4j


@pytest.mark.parametrize(
    ("command", "args", "operation"),
    [
        (graph_load, ["unused.tar.gz", "scratch"], "load"),
        (graph_clear, ["scratch", "--force"], "clear"),
    ],
)
def test_destructive_command_refuses_target_mismatch(
    tmp_path: Path,
    command,
    args: list[str],
    operation: str,
) -> None:
    """A scratch-targeted command cannot act through the live graph link."""
    live_dir = tmp_path / "codex"
    live_dir.mkdir()
    sentinel = live_dir / "live-data"
    sentinel.write_text("must remain")
    profile = SimpleNamespace(
        name="codex",
        data_dir=live_dir,
        host=None,
        http_port=7474,
    )
    if command is graph_load:
        archive = tmp_path / "unused.tar.gz"
        archive.touch()
        args[0] = str(archive)

    with (
        patch(
            "imas_codex.graph.profiles.resolve_neo4j",
            return_value=profile,
        ),
        patch("imas_codex.cli.graph.data.backup_graph_dump") as backup,
        patch("imas_codex.cli.graph.data.is_neo4j_running", return_value=True),
        patch("imas_codex.graph.client.GraphClient.from_profile") as client,
    ):
        result = CliRunner().invoke(command, args)

    assert result.exit_code != 0
    assert (
        f"Refusing to {operation} graph 'scratch': active graph is 'codex'"
        in result.output
    )
    backup.assert_not_called()
    client.assert_not_called()
    assert sentinel.read_text() == "must remain"


def test_clear_succeeds_when_target_matches_active_graph() -> None:
    """A matching explicit name permits the existing clear operation."""
    profile = SimpleNamespace(name="scratch", host=None, http_port=7474)
    graph_client = MagicMock()
    graph_client.get_stats.return_value = {"nodes": 3, "relationships": 2}
    graph_client.drop_all.return_value = 3

    with (
        patch(
            "imas_codex.graph.profiles.resolve_neo4j",
            return_value=profile,
        ),
        patch("imas_codex.cli.graph.data.is_neo4j_running", return_value=True),
        patch(
            "imas_codex.graph.client.GraphClient.from_profile",
            return_value=graph_client,
        ),
    ):
        result = CliRunner().invoke(graph_clear, ["scratch", "--force"])

    assert result.exit_code == 0, result.output
    assert "Clearing graph [scratch]" in result.output
    assert "Cleared 3 nodes from [scratch]" in result.output
    graph_client.drop_all.assert_called_once_with()


def test_destructive_commands_require_target_name(tmp_path: Path) -> None:
    """Both destructive commands reject invocations without a target name."""
    runner = CliRunner()
    archive = tmp_path / "unused.tar.gz"
    archive.touch()

    clear_result = runner.invoke(graph_clear, [])
    load_result = runner.invoke(graph_load, [str(archive)])

    assert clear_result.exit_code == 2
    assert "Missing argument 'TARGET'" in clear_result.output
    assert load_result.exit_code == 2
    assert "Missing argument 'TARGET'" in load_result.output


def test_resolve_neo4j_signature_remains_connection_only() -> None:
    """The shared profile resolver remains unchanged for read-only callers."""
    assert inspect.signature(resolve_neo4j) == inspect.Signature(
        parameters=[
            inspect.Parameter(
                "auto_tunnel",
                inspect.Parameter.KEYWORD_ONLY,
                default=True,
                annotation="bool",
            )
        ],
        return_annotation="Neo4jProfile",
    )

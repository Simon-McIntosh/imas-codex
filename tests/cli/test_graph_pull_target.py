"""Target identity guard for registry graph pulls."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.graph import registry


def test_pull_requires_target_name() -> None:
    """A pull cannot begin without an explicitly named graph target."""
    result = CliRunner().invoke(registry.graph_pull, [])

    assert result.exit_code == 2
    assert "Missing argument 'TARGET'" in result.output


def test_pull_refuses_target_mismatch_before_destructive_steps() -> None:
    """A pull aimed away from the active graph refuses before doing work."""
    profile = SimpleNamespace(name="codex")

    with (
        patch("imas_codex.graph.profiles.resolve_neo4j", return_value=profile),
        patch.object(registry, "get_git_info") as git_info,
        patch.object(registry, "backup_graph_dump") as backup,
        patch.object(registry, "require_oras") as require_oras,
        patch("imas_codex.cli.graph.data.graph_load") as graph_load,
    ):
        result = CliRunner().invoke(
            registry.graph_pull,
            ["scratch", "--version", "test", "--force"],
        )

    assert result.exit_code != 0
    assert "Refusing to pull graph 'scratch': active graph is 'codex'" in result.output
    git_info.assert_not_called()
    backup.assert_not_called()
    require_oras.assert_not_called()
    graph_load.assert_not_called()


def test_pull_forwards_target_to_local_load(tmp_path: Path) -> None:
    """The matched target remains explicit at the nested load boundary."""
    profile = SimpleNamespace(name="scratch", host=None, data_dir=tmp_path / "data")
    nested_runner = MagicMock()
    nested_runner.invoke.return_value = SimpleNamespace(exit_code=0, output="")
    archive = MagicMock()
    archive.extractall.side_effect = lambda destination: Path(destination).mkdir()

    def create_archive(args: list[str], **_kwargs) -> None:
        output_dir = Path(args[args.index("-o") + 1])
        (output_dir / "graph.tar.gz").touch()

    runner = CliRunner()
    with (
        patch("imas_codex.graph.profiles.resolve_neo4j", return_value=profile),
        patch("imas_codex.graph.remote.is_remote_location", return_value=False),
        patch.object(registry, "get_git_info", return_value={}),
        patch.object(registry, "get_registry", return_value="registry.example"),
        patch.object(registry, "get_package_name", return_value="graph"),
        patch.object(registry, "require_oras"),
        patch.object(registry, "check_graph_exists", return_value=False),
        patch.object(registry, "login_to_ghcr"),
        patch(
            "imas_codex.cli.graph_progress.run_oras_with_progress",
            side_effect=create_archive,
        ),
        patch("imas_codex.cli.graph_progress.GraphProgress"),
        patch("click.testing.CliRunner", return_value=nested_runner),
        patch.object(registry.tarfile, "open") as tar_open,
    ):
        tar_open.return_value.__enter__.return_value = archive
        result = runner.invoke(
            registry.graph_pull,
            ["scratch", "--version", "test", "--force", "--no-backup"],
        )

    assert result.exit_code == 0, result.output
    nested_runner.invoke.assert_called_once()
    load_args = nested_runner.invoke.call_args.args[1]
    assert Path(load_args[0]).name == "graph.tar.gz"
    assert load_args[1:] == ["scratch", "--force"]

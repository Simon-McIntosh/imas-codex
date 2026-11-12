"""CLI interface for IMAS Codex Server.

Modular CLI structure with command groups split by functionality. Each command
module is imported when that command runs, so starting one command does not pay
for every other command's imports.
"""

# Warning filters are set in imas_codex/__init__.py at package import time
# to ensure they're applied before any aiohttp imports.

import importlib
import logging

import click

from imas_codex import __version__

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ModuleNotFoundError:
    pass

# Configure logging
logger = logging.getLogger(__name__)


# name -> (module, attribute, help shown in the top-level listing).
# The help text sits here because `--help` must not import the module.
_COMMANDS: dict[str, tuple[str, str, str]] = {
    "config": (
        "imas_codex.cli.config_cli",
        "config",
        "Manage private facility configs, secrets, and local-host settings.",
    ),
    "credentials": (
        "imas_codex.cli.credentials",
        "credentials",
        "Manage service credentials.",
    ),
    "discover": (
        "imas_codex.cli.discover",
        "discover",
        "Discover facility resources with graph-led exploration.",
    ),
    "embed": ("imas_codex.cli.embed", "embed", "Manage GPU embedding server."),
    "facilities": (
        "imas_codex.cli.facilities",
        "facilities",
        "Manage facility configurations.",
    ),
    "graph": ("imas_codex.cli.graph", "graph", "Manage graph database lifecycle."),
    "host": (
        "imas_codex.cli.host",
        "host",
        "Node load monitoring and SSH target management.",
    ),
    "hpc": (
        "imas_codex.cli.compute",
        "hpc",
        "Manage SLURM HPC jobs for heavy workloads.",
    ),
    "imas": (
        "imas_codex.cli.imas_dd",
        "imas",
        "IMAS data dictionary, mappings, and IDS assembly.",
    ),
    "llm": (
        "imas_codex.cli.llm_cli",
        "llm",
        "Manage LiteLLM proxy server for centralized LLM routing.",
    ),
    "release": (
        "imas_codex.cli.release",
        "release",
        "State-machine release: semantic version bumps, graph publishing, tagging.",
    ),
    "serve": ("imas_codex.cli.serve", "serve", "Start the IMAS Codex MCP server."),
    "sn": (
        "imas_codex.cli.sn",
        "sn",
        "Standard name generation and management.",
    ),
    "tools": (
        "imas_codex.cli.tools",
        "tools",
        "Manage development tools and Python environment.",
    ),
    "tunnel": (
        "imas_codex.cli.tunnel",
        "tunnel",
        "Manage SSH tunnels to remote services.",
    ),
}


class DeferredCommandGroup(click.Group):
    """A command group that imports a command module only when needed.

    ``list_commands`` and ``format_commands`` answer from the table alone, so
    ``--help`` exits without importing any command module. ``get_command``
    imports the module the first time the command is resolved and caches the
    result in the command's normal ``commands`` mapping.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(_COMMANDS) | set(self.commands))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        registered = self.commands.get(cmd_name)
        if registered is not None:
            return registered
        target = _COMMANDS.get(cmd_name)
        if target is None:
            return None
        module_name, attribute, _ = target
        command = getattr(importlib.import_module(module_name), attribute)
        self.commands[cmd_name] = command
        return command

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        names = []
        for name in self.list_commands(ctx):
            resolved = self.commands.get(name)
            if resolved is not None and resolved.hidden:
                continue
            names.append(name)
        if not names:
            return
        limit = formatter.width - 6 - max(len(name) for name in names)
        rows = []
        for name in names:
            resolved = self.commands.get(name)
            if resolved is not None:
                rows.append((name, resolved.get_short_help_str(limit)))
            else:
                rows.append((name, _COMMANDS[name][2]))
        with formatter.section("Commands"):
            formatter.write_dl(rows)


# Create the main CLI group
@click.group(cls=DeferredCommandGroup, invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    help="Show the imas-codex version and exit.",
)
@click.pass_context
def main(ctx: click.Context, version: bool) -> None:
    """IMAS Codex - AI-enhanced MCP servers for fusion data.

    Use subcommands to start servers or manage data:

    \b
      imas-codex serve              Start the MCP server
      imas-codex graph start        Start Neo4j graph database
      imas-codex embed start        Start GPU embedding server
      imas-codex llm start          Start LiteLLM proxy
      imas-codex tunnel start       Start SSH tunnels
      imas-codex config private     Manage private facility YAML
      imas-codex imas dd build       Build/update IMAS DD graph
      imas-codex facilities list    List configured facilities
    """
    if version:
        click.echo(__version__)
        ctx.exit()

    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


__all__ = ["main"]

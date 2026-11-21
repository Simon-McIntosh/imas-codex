"""Graph CLI — Neo4j lifecycle, data operations, and registry."""

from __future__ import annotations

import click


@click.group()
def graph() -> None:
    """Manage graph database lifecycle.

    \b
    Server:
      iter-codex graph start               Start Neo4j (SLURM/systemd/local)
      iter-codex graph stop                Stop Neo4j
      iter-codex graph status              Show service, backup, and SLURM status

    \b
    Setup:
      iter-codex graph init NAME           Create a new graph
      iter-codex graph list                List local graph instances
      iter-codex graph switch NAME         Select data via the active symlink
      iter-codex graph shell               Interactive Cypher REPL

    \b
    Archive & Registry:
      iter-codex graph export              Export graph to archive
      iter-codex graph load ARCHIVE TARGET Load archive into the named active graph
      iter-codex graph push                Push archive to GHCR
      iter-codex graph pull TARGET         Fetch + load into the named active graph
      iter-codex graph fetch               Download archive (no load)
      iter-codex graph tags                List GHCR versions
      iter-codex graph prune               Remove GHCR tags

    \b
    Maintenance:
      iter-codex graph clear TARGET        Clear the named active graph
      iter-codex graph secure              Rotate Neo4j password

    \b
    Standard Names grammar is auto-synced by ``iter-codex sn run`` (at
    startup) and ``iter-codex sn clear`` (post-wipe re-seed).
    """
    pass


def _register_graph_commands() -> None:
    """Register all graph subcommands from split modules."""
    from iter_codex.cli.graph.data import (
        graph_clear,
        graph_export,
        graph_facility_group,
        graph_init,
        graph_list,
        graph_load,
        graph_secure,
        graph_switch,
    )
    from iter_codex.cli.graph.registry import (
        graph_fetch,
        graph_prune,
        graph_pull,
        graph_push,
        graph_tags,
    )
    from iter_codex.cli.graph.server import (
        graph_profiles,
        graph_shell,
        graph_start,
        graph_status,
        graph_stop,
    )

    # Server
    graph.add_command(graph_start, "start")
    graph.add_command(graph_stop, "stop")
    graph.add_command(graph_status, "status")
    graph.add_command(graph_shell, "shell")
    graph.add_command(graph_profiles, "profiles")

    # Data / lifecycle
    graph.add_command(graph_secure, "secure")
    graph.add_command(graph_export, "export")
    graph.add_command(graph_load, "load")
    graph.add_command(graph_init, "init")
    graph.add_command(graph_switch, "switch")
    graph.add_command(graph_list, "list")
    graph.add_command(graph_clear, "clear")
    graph.add_command(graph_facility_group, "facility")

    # Registry
    graph.add_command(graph_push, "push")
    graph.add_command(graph_fetch, "fetch")
    graph.add_command(graph_pull, "pull")
    graph.add_command(graph_tags, "tags")
    graph.add_command(graph_prune, "prune")


_register_graph_commands()

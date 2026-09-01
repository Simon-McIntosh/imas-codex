"""Graph CLI — Neo4j lifecycle, data operations, and registry."""

from __future__ import annotations

import click


@click.group()
def graph() -> None:
    """Manage graph database lifecycle.

    \b
    Server:
      imas-codex graph start               Start Neo4j (SLURM/systemd/local)
      imas-codex graph stop                Stop Neo4j
      imas-codex graph status              Show service, backup, and SLURM status

    \b
    Setup:
      imas-codex graph init NAME           Create a new graph
      imas-codex graph list                List local graph instances
      imas-codex graph switch NAME         Select data via the active symlink
      imas-codex graph shell               Interactive Cypher REPL

    \b
    Archive & Registry:
      imas-codex graph export              Export graph to archive
      imas-codex graph load ARCHIVE TARGET Load archive into the named active graph
      imas-codex graph push                Push archive to GHCR
      imas-codex graph pull                Fetch + load from GHCR
      imas-codex graph fetch               Download archive (no load)
      imas-codex graph tags                List GHCR versions
      imas-codex graph prune               Remove GHCR tags

    \b
    Maintenance:
      imas-codex graph clear TARGET        Clear the named active graph
      imas-codex graph secure              Rotate Neo4j password

    \b
    Standard Names grammar is auto-synced by ``imas-codex sn run`` (at
    startup) and ``imas-codex sn clear`` (post-wipe re-seed).
    """
    pass


def _register_graph_commands() -> None:
    """Register all graph subcommands from split modules."""
    from imas_codex.cli.graph.data import (
        graph_clear,
        graph_export,
        graph_facility_group,
        graph_init,
        graph_list,
        graph_load,
        graph_secure,
        graph_switch,
    )
    from imas_codex.cli.graph.offsite import graph_offsite_push
    from imas_codex.cli.graph.registry import (
        graph_fetch,
        graph_prune,
        graph_pull,
        graph_push,
        graph_tags,
    )
    from imas_codex.cli.graph.server import (
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
    graph.add_command(graph_offsite_push, "offsite-push")


_register_graph_commands()

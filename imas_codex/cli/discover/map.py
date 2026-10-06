"""Candidate judgment command: route signal sources to DD candidate targets.

Claims enriched sources whose Data Dictionary candidates have not been judged,
routes each to the IDSs most likely to hold its values, retrieves candidates
within those IDSs, judges them with the decisions model, and records the route
and candidate edges on the graph. A source that already carries a
``candidate_route`` is skipped; re-judging it is a two-step operation — clear
its candidates first with ``imas-codex discover clear FACILITY -d map``.
"""

from __future__ import annotations

import logging
import time

import click

logger = logging.getLogger(__name__)


def clear_facility_candidates(facility: str) -> dict[str, int]:
    """Remove every MAPPING_CANDIDATE edge and candidate_route for a facility."""
    from imas_codex.graph import GraphClient
    from imas_codex.ids.graph_ops import clear_candidates

    with GraphClient() as gc:
        return clear_candidates(facility, gc)


@click.command("map")
@click.argument("facility")
@click.option(
    "--physics-domain",
    "-d",
    "physics_domains",
    multiple=True,
    help="Restrict to a physics domain (repeatable).",
)
@click.option(
    "--ids",
    "-i",
    "ids_filter",
    multiple=True,
    help="Restrict the routed IDSs to this set (repeatable).",
)
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=5.0,
    help="Maximum LLM spend in USD",
)
@click.option(
    "--signal-limit",
    "-n",
    type=int,
    default=None,
    help="Maximum sources to judge",
)
@click.option(
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum runtime in minutes (e.g., 5). Halts when time expires.",
)
def map_candidates(
    facility: str,
    physics_domains: tuple[str, ...],
    ids_filter: tuple[str, ...],
    cost_limit: float,
    signal_limit: int | None,
    time_limit: int | None,
) -> None:
    """Judge DD candidates for a facility's signal sources.

    Claims enriched sources whose candidates are unjudged, routes each to its
    most probable IDSs, retrieves candidates within those IDSs, judges them and
    records the route and candidate edges. Re-judging a source is controlled by
    clearing its candidates first.

    \b
    Examples:
      imas-codex discover map jet
      imas-codex discover map jet -d magnetics
      imas-codex discover map jet -i equilibrium -i core_profiles -c 2.0
      imas-codex discover map jet -n 50 --time 10
    """
    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.discovery.base.facility import get_facility

    use_rich = use_rich_output()
    console = setup_logging("map", facility, use_rich)
    log_print = make_log_print("map", console)

    try:
        config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    log_print(f"\n[bold]Candidate Discovery: {facility}[/bold]")
    if physics_domains:
        log_print(f"  Physics domains: {', '.join(physics_domains)}")
    if ids_filter:
        log_print(f"  IDS filter: {', '.join(ids_filter)}")
    log_print(f"  Cost limit: ${cost_limit:.2f}")
    if signal_limit:
        log_print(f"  Source limit: {signal_limit}")
    if time_limit is not None:
        log_print(f"  Time limit: {time_limit} min")
    log_print("")

    deadline = time.time() + time_limit * 60 if time_limit is not None else None

    disc_config = DiscoveryConfig(
        domain="map",
        facility=facility,
        facility_config=config,
        model_section="mapping-candidates",
        display=None,
        check_graph=True,
        check_embed=True,
        check_model=False,
        check_ssh=False,
        check_auth=False,
    )

    async def async_main(stop_event, service_monitor):
        from imas_codex.ids.workers import (
            CandidateDiscoveryState,
            run_candidate_engine,
        )

        engine_state = CandidateDiscoveryState(
            facility=facility,
            domains=list(physics_domains),
            ids_filter=list(ids_filter),
            source_limit=signal_limit,
            cost_limit=cost_limit,
        )
        if deadline is not None:
            engine_state.deadline = deadline

        def _on_progress(detail, stats, stream_items=None):
            log_print(f"  {detail}")

        await run_candidate_engine(
            engine_state, stop_event=stop_event, on_progress=_on_progress
        )

        return {
            "sources_judged": engine_state.sources_judged,
            "candidates_written": engine_state.candidates_written,
            "cost": engine_state.cost.total_usd,
        }

    try:
        result = run_discovery(disc_config, async_main)
    except KeyboardInterrupt:
        log_print("\n[yellow]Discovery interrupted by user[/yellow]")
        raise SystemExit(130) from None
    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from e

    log_print(
        f"\n  [green]{result['sources_judged']} sources judged, "
        f"{result['candidates_written']} candidates written[/green]"
    )
    log_print(f"  [dim]Cost: ${result['cost']:.2f}[/dim]")
    log_print("\n[green]Candidate discovery complete.[/green]")

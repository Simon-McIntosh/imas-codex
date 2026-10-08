"""Candidate judgment stage: route signal sources to DD candidate targets.

Claims enriched sources whose Data Dictionary candidates have not been judged,
routes each to the IDSs most likely to hold its values, retrieves candidates
within those IDSs, judges them with the decisions model, and records the route
and candidate edges on the graph. A source that already carries a
``candidate_route`` is skipped; re-judging it is a two-step operation — clear
its candidates first with ``imas-codex discover clear FACILITY -d map``.

The stage drains the sources the signals stage seeded, so it has no seeding
half of its own. :func:`run_candidates_stage` is the stage the sequence runner
calls; the ``map`` click command is a thin wrapper that builds the stage
options and delegates to it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import click

from imas_codex.cli.discover.common import resolve_focus_items

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidatesStageOptions:
    """The settled discover options the candidates stage accepts.

    Field names are the settled spellings the sequence command uses, so a
    domain's options read the same wherever the sequence passes them.
    ``focus`` names SignalSource identities or a manifest listing them.
    ``topic`` is carried for a stage that steers a scorer.
    """

    physics_domain: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()
    cost_limit: float = 5.0
    limit: int | None = None
    time_limit: int | None = None
    scan_only: bool = False
    flush: bool = False
    focus: tuple[str, ...] = ()
    topic: str | None = None


def clear_facility_candidates(facility: str) -> dict[str, int]:
    """Remove every MAPPING_CANDIDATE edge and candidate_route for a facility."""
    from imas_codex.graph import GraphClient
    from imas_codex.ids.graph_ops import clear_candidates

    with GraphClient() as gc:
        return clear_candidates(facility, gc)


def _validate_focus(facility: str, ids: list[str]) -> None:
    """Refuse names that do not identify a source at this facility."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (n:SignalSource {facility_id: $facility}) "
            "WHERE n.id IN $ids RETURN n.id AS id",
            facility=facility,
            ids=ids,
        )
    found = {row["id"] for row in rows}
    missing = [source_id for source_id in ids if source_id not in found]
    if missing:
        raise click.UsageError(
            "--focus item filter names unknown SignalSource id(s): "
            + ", ".join(missing)
        )


def run_candidates_stage(
    facility: str, options: CandidatesStageOptions
) -> dict[str, float]:
    """Judge DD candidates for a facility's signal sources.

    The candidates stage drains work the signals stage seeded and seeds
    nothing itself, so ``--scan-only`` has no half to run and returns without
    touching the graph. ``--flush`` and the default both run the draining
    worker. ``--focus`` restricts claims to named SignalSource identities;
    ``--limit`` caps the sources judged this run.
    """
    focus_ids = resolve_focus_items(options.focus)
    if focus_ids:
        _validate_focus(facility, focus_ids)

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

    log_print(f"\n[bold]Candidate Discovery: {facility}[/bold]")

    # Candidates drains only. A --scan-only pass therefore has no seeding half
    # to run; it reports nothing to do rather than seeding and judging anyway.
    if options.scan_only and not options.flush:
        log_print("  Nothing to seed: candidates drains the signals stage's sources.")
        return {"sources_judged": 0, "candidates_written": 0, "cost": 0.0}

    try:
        config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    if options.physics_domain:
        log_print(f"  Physics domains: {', '.join(options.physics_domain)}")
    if options.ids:
        log_print(f"  IDS filter: {', '.join(options.ids)}")
    if focus_ids:
        log_print(f"  SignalSource focus: {', '.join(focus_ids)}")
    log_print(f"  Cost limit: ${options.cost_limit:.2f}")
    if options.limit:
        log_print(f"  Source limit: {options.limit}")
    if options.time_limit is not None:
        log_print(f"  Time limit: {options.time_limit} min")
    if options.topic:
        log_print(f"  Topic: {options.topic}")
    log_print("")

    deadline = (
        time.time() + options.time_limit * 60
        if options.time_limit is not None
        else None
    )

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
            domains=list(options.physics_domain),
            ids_filter=list(options.ids),
            focus_ids=focus_ids,
            source_limit=options.limit,
            cost_limit=options.cost_limit,
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
    return result


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
@click.option(
    "--focus",
    multiple=True,
    help="Restrict claims to SignalSource ids or a YAML manifest listing items.",
)
def map_candidates(
    facility: str,
    physics_domains: tuple[str, ...],
    ids_filter: tuple[str, ...],
    cost_limit: float,
    signal_limit: int | None,
    time_limit: int | None,
    focus: tuple[str, ...],
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
      imas-codex discover map jet --focus source-id
    """
    options = CandidatesStageOptions(
        physics_domain=tuple(physics_domains),
        ids=tuple(ids_filter),
        cost_limit=cost_limit,
        limit=signal_limit,
        time_limit=time_limit,
        focus=focus,
    )
    run_candidates_stage(facility, options)

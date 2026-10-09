"""Signals discovery stage: facility-agnostic signal scanning and enrichment.

Dispatches to registered scanner plugins based on facility config data_systems.
Scanner plugins handle facility-specific enumeration (TDI, PPF, EDAS, MDSplus,
IMAS, device XML), while shared infrastructure handles LLM enrichment and
validation. Wiki content is used as enrichment context rather than a
user-selectable scanner.

The discovery logic lives in :func:`run_signals_stage`, which takes a facility
and a frozen :class:`SignalsStageOptions`. The click command is a thin wrapper
that builds the options and calls the stage function. ``--scan-only`` selects
the seeding half (enumerate work, judge nothing); ``--flush`` selects the
draining half (enrich and check the nodes the seeding half created).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import click

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalsStageOptions:
    """Settled options for the signals discovery stage.

    Field names follow the settled discover option surface: ``scan_only`` and
    ``flush`` select the seeding and draining halves, ``topic`` is the
    free-text steer the enricher reads, ``focus`` names signals or signal
    sources and scopes every claim to them, and ``limit`` caps items.
    """

    scan_only: bool = False
    flush: bool = False
    topic: str | None = None
    focus: tuple[str, ...] = ()
    limit: int | None = None
    cost_limit: float = 5.0
    time_limit: int | None = None
    rescan: bool = False
    scanners: str | None = None
    categories: str | None = None
    reference_shot: int | None = None
    enrich_workers: int = 8
    check_workers: int = 4
    reset_to: str | None = None


def _validate_focus(facility: str, focus_items: list[str]) -> None:
    """Refuse focus items that name no signal or source at this facility.

    A focus item may name a FacilitySignal by id or accessor, a SignalSource
    by id, or a source array by a ``data_source_path`` segment (such as
    ``magPbTC10``). Anything that resolves to none of those is a typo the
    claim would silently honour by selecting nothing, so it is refused up
    front with the offending identities named.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (n) WHERE (n:FacilitySignal OR n:SignalSource) "
            "AND n.facility_id = $facility "
            "AND (n.id IN $ids OR n.accessor IN $ids "
            "OR ANY(segment IN split(coalesce(n.data_source_path, ''), '/') "
            "WHERE segment IN $ids)) "
            "RETURN n.id AS id, n.accessor AS accessor, "
            "n.data_source_path AS data_source_path",
            facility=facility,
            ids=focus_items,
        )
    matched = {row["id"] for row in rows}
    matched.update(row["accessor"] for row in rows if row["accessor"])
    for row in rows:
        if row["data_source_path"]:
            matched.update(
                segment
                for segment in row["data_source_path"].split("/")
                if segment in focus_items
            )
    missing = [item for item in focus_items if item not in matched]
    if missing:
        raise click.UsageError(
            "--focus names unknown signal or source id(s) at "
            f"{facility}: " + ", ".join(missing)
        )


def run_signals_stage(facility: str, options: SignalsStageOptions) -> dict:
    """Run the signals discovery stage for a facility.

    Builds the engine config and calls ``run_discovery``; returns the run's
    result dict (counts, cost, elapsed seconds). ``--focus`` names signals or
    ``SignalSource`` identities (or a manifest of them) and scopes every claim
    to those; an item that names nothing at the facility is a usage error.
    """
    from imas_codex.cli.discover.common import resolve_focus_items

    focus_items = resolve_focus_items(options.focus)
    if focus_items:
        _validate_focus(facility, focus_items)

    # Auto-detect rich output
    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        ensure_remote_environment,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.signals.scanners.base import (
        get_scanners_for_facility,
        list_scanners,
    )

    use_rich = use_rich_output()
    console = setup_logging("signals", facility, use_rich)
    log_print = make_log_print("signals", console)

    try:
        config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    ssh_host = config.get("ssh_host")
    if not ssh_host:
        log_print(f"[red]No SSH host configured for {facility}[/red]")
        raise SystemExit(1)

    # Scanning and checking run on the facility host; a draining run reads
    # only the graph and the model, so it opens no session there.
    if not options.flush:
        ensure_remote_environment(config)

    # Resolve scanner types
    if options.scanners:
        scanner_types = [s.strip() for s in options.scanners.split(",")]
        # Validate requested scanner types exist
        available = list_scanners()
        invalid = [s for s in scanner_types if s not in available]
        if invalid:
            log_print(
                f"[red]Unknown scanner types: {invalid}. Available: {available}[/red]"
            )
            raise SystemExit(1)
    else:
        # Auto-detect from facility config
        scanner_instances = get_scanners_for_facility(facility)
        scanner_types = [s.scanner_type for s in scanner_instances]

    if not scanner_types:
        log_print(
            f"[red]No data sources configured for {facility}.[/red]\n"
            "Configure data_systems in facility YAML or specify --scanners."
        )
        raise SystemExit(1)

    # Resolve reference shot from config if not specified
    reference_shot = options.reference_shot
    data_systems = config.get("data_systems", {})
    if reference_shot is None:
        for source_config in data_systems.values():
            if isinstance(source_config, dict):
                ref = source_config.get("reference_shot") or source_config.get(
                    "reference_pulse"
                )
                if ref:
                    reference_shot = int(ref)
                    break

    category_list = (
        [c.strip() for c in options.categories.split(",") if c.strip()]
        if options.categories
        else None
    )

    # Handle --reset-to: reset signals back to the target state. The reset
    # takes the same scanner, category and focus scope as the claims that
    # follow it, so a scoped run never resets rows it will not then process.
    if options.reset_to:
        from imas_codex.discovery.base.reset import SIGNAL_RESET_SPECS, reset_to_status
        from imas_codex.discovery.signals.parallel import (
            build_category_predicate,
            build_focus_predicate,
        )

        spec = SIGNAL_RESET_SPECS[options.reset_to]
        extra_filter = ""
        extra_params: dict = {}
        if options.scanners:
            extra_filter = "AND n.discovery_source IN $sources"
            extra_params["sources"] = scanner_types
        if category_list:
            extra_filter += f" AND {build_category_predicate('n')}"
            extra_params["categories"] = category_list
        if focus_items:
            extra_filter += f" {build_focus_predicate('n', focus_items)}"
            extra_params["focus_items"] = focus_items

        reset_count = reset_to_status(
            spec, facility, extra_filter=extra_filter, extra_params=extra_params
        )
        scope_parts = []
        if options.scanners:
            scope_parts.append(f"scanner: {options.scanners}")
        if category_list:
            scope_parts.append(f"categories: {', '.join(category_list)}")
        if focus_items:
            scope_parts.append(f"focus: {', '.join(focus_items)}")
        scope = f" ({'; '.join(scope_parts)})" if scope_parts else ""
        log_print(
            f"[yellow]Reset {reset_count} signals to '{options.reset_to}'{scope}[/yellow]"
        )

    log_print(f"\n[bold]Signal Discovery: {facility}[/bold]")
    log_print(f"  Scanners: {', '.join(scanner_types)}")
    log_print(f"  SSH host: {ssh_host}")
    if reference_shot:
        log_print(f"  Reference shot: {reference_shot}")
    log_print(f"  Cost limit: ${options.cost_limit:.2f}")
    if options.limit:
        log_print(f"  Signal limit: {options.limit}")
    if options.time_limit is not None:
        log_print(f"  Time limit: {options.time_limit} min")
    if options.topic:
        log_print(f"  Topic: {options.topic}")
    if category_list:
        log_print(f"  Categories: {', '.join(category_list)}")
    if options.rescan:
        log_print("  Mode: rescan")
    if options.reset_to:
        log_print(f"  Mode: reset-to {options.reset_to}")
    log_print(
        f"  Workers: {options.enrich_workers} enrich, {options.check_workers} check"
    )
    log_print("")

    try:
        from imas_codex.discovery.signals.parallel import run_parallel_data_discovery

        # Compute deadline from time limit
        deadline: float | None = None
        if options.time_limit is not None:
            deadline = time.time() + (options.time_limit * 60)

        sig_logger = logging.getLogger("imas_codex.discovery.signals")

        # Build display for rich mode
        display = None
        if use_rich:
            from imas_codex.discovery.signals.progress import DataProgressDisplay

            display = DataProgressDisplay(
                facility=facility,
                cost_limit=options.cost_limit,
                signal_limit=options.limit,
                focus=options.topic or "",
                console=console,
                discover_only=options.scan_only,
                enrich_only=options.flush,
            )

        # Custom async graph refresh for signals (uses update_from_graph with kwargs)
        async def signals_graph_refresh():
            from imas_codex.discovery.signals.parallel import (
                get_data_discovery_stats,
            )

            stats = await asyncio.to_thread(
                get_data_discovery_stats,
                facility,
                scanner_types,
            )
            if stats and display:
                display.update_from_graph(
                    total_signals=stats.get("total", 0),
                    signals_discovered=stats.get("discovered", 0),
                    signals_enriched=stats.get("enriched", 0),
                    signals_checked=stats.get("checked", 0),
                    signals_skipped=stats.get("skipped", 0),
                    signals_failed=stats.get("failed", 0),
                    pending_enrich=stats.get("pending_enrich", 0),
                    pending_check=stats.get("pending_check", 0),
                    accumulated_cost=stats.get("accumulated_cost", 0.0),
                    signal_sources=stats.get("signal_sources", 0),
                    grouped_signals=stats.get("grouped_signals", 0),
                )

        disc_config = DiscoveryConfig(
            domain="signals",
            facility=facility,
            facility_config=config,
            model_section="discovery-describe",
            display=display,
            check_graph=True,
            check_embed=not options.scan_only,
            check_model=not options.scan_only,
            check_ssh=not options.flush,
            check_auth=False,
            graph_refresh_interval=2.0,
            graph_refresh_fn=signals_graph_refresh if use_rich else None,
            suppress_loggers=[
                "imas_codex.embeddings",
                "imas_codex.discovery.signals",
            ],
        )

        # Callbacks — wire to display or logging
        if display:

            def on_scan(msg, stats, results=None):
                display.update_scan(msg, stats, results)

            def on_extract(msg, stats, results=None):
                display.update_extract(msg, stats, results)

            def on_promote(msg, stats, results=None):
                display.update_promote(msg, stats, results)

            def on_enrich(msg, stats, results=None):
                display.update_enrich(msg, stats, results)

            def on_check(msg, stats, results=None):
                display.update_check(msg, stats, results)

            def on_worker_status(worker_group):
                display.update_worker_status(worker_group)
        else:

            def on_scan(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("SEED: %s", msg)

            def on_extract(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("EXTRACT: %s", msg)

            def on_promote(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("PROMOTE: %s", msg)

            def on_enrich(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("ENRICH: %s", msg)

            def on_check(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("CHECK: %s", msg)

            on_worker_status = None

        async def async_main(stop_event, service_monitor):
            return await run_parallel_data_discovery(
                facility=facility,
                ssh_host=ssh_host,
                scanner_types=scanner_types,
                reference_shot=reference_shot,
                cost_limit=options.cost_limit,
                signal_limit=options.limit,
                focus=options.topic,
                focus_items=focus_items or None,
                categories=category_list,
                discover_only=options.scan_only,
                enrich_only=options.flush,
                deadline=deadline,
                num_enrich_workers=options.enrich_workers,
                num_check_workers=options.check_workers,
                on_discover_progress=on_scan,
                on_extract_progress=on_extract,
                on_promote_progress=on_promote,
                on_enrich_progress=on_enrich,
                on_check_progress=on_check,
                on_worker_status=on_worker_status,
                stop_event=stop_event,
            )

        def on_complete(result):
            if display:
                try:
                    from imas_codex.discovery.signals.parallel import (
                        get_data_discovery_stats,
                    )

                    final_stats = get_data_discovery_stats(facility, scanner_types)
                    if final_stats:
                        display.update_from_graph(
                            total_signals=final_stats.get("total", 0),
                            signals_discovered=final_stats.get("discovered", 0),
                            signals_enriched=final_stats.get("enriched", 0),
                            signals_checked=final_stats.get("checked", 0),
                            signals_skipped=final_stats.get("skipped", 0),
                            signals_failed=final_stats.get("failed", 0),
                            pending_enrich=final_stats.get("pending_enrich", 0),
                            pending_check=final_stats.get("pending_check", 0),
                            accumulated_cost=final_stats.get("accumulated_cost", 0.0),
                        )
                except Exception:
                    pass

        result = run_discovery(disc_config, async_main, on_complete=on_complete)

        # Final output
        scanned = result.get("scanned", 0)
        enriched = result.get("enriched", 0)
        checked = result.get("checked", 0)
        cost = result.get("cost", 0)
        elapsed = result.get("elapsed_seconds", 0)

        log_print(
            f"\n  [green]{scanned} scanned, {enriched} enriched, "
            f"{checked} checked[/green]"
        )
        log_print(f"  [dim]Cost: ${cost:.2f}, Time: {elapsed:.1f}s[/dim]")

    except KeyboardInterrupt:
        log_print("\n[yellow]Discovery interrupted by user[/yellow]")
        from imas_codex.remote.executor import cleanup_ssh_on_exit

        cleanup_ssh_on_exit()
        raise SystemExit(130) from None
    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from e

    log_print("\n[green]Signal discovery complete.[/green]")
    return result

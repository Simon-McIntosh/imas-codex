"""Code discovery command: Parallel code scanning, scoring, and ingestion."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import click

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeStageOptions:
    """Options shared by the code command and discovery sequence."""

    min_score: float | None = None
    limit: int = 100
    focus: tuple[str, ...] = ()
    topic: str | None = None
    cost_limit: float = 5.0
    scan_workers: int = 2
    triage_workers: int = 2
    enrich_workers: int = 2
    score_workers: int = 1
    code_workers: int = 1
    scan_only: bool = False
    flush: bool = False
    time_limit: int | None = None
    verbose: bool = False
    rescan: bool = False
    triage_batch_size: int | None = None
    rejudge_ingested: bool = False
    reset_to: str | None = None


def run_code_stage(facility: str, options: CodeStageOptions) -> None:
    """Run code discovery with item scope and independent seed/drain halves.

    The code engine disables its primary draining workers for scan-only runs.
    A zero scan worker count disables its seed phase for flush runs while
    retaining triage, enrichment, scoring, ingestion, and linking.
    """
    if options.scan_only and options.flush:
        raise click.UsageError("--scan-only and --flush are mutually exclusive")

    min_score = options.min_score
    max_paths = options.limit
    focus = options.topic
    path_prefixes = options.focus
    cost_limit = options.cost_limit
    scan_workers = 0 if options.flush else options.scan_workers
    triage_workers = options.triage_workers
    enrich_workers = options.enrich_workers
    score_workers = options.score_workers
    code_workers = options.code_workers
    scan_only = options.scan_only
    score_only = False
    time_limit = options.time_limit
    verbose = options.verbose
    rescan = options.rescan
    triage_batch_size = options.triage_batch_size
    rejudge_ingested = options.rejudge_ingested
    reset_to = options.reset_to
    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        ensure_remote_environment,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.base.services import llm_health_check_with_decisions
    from imas_codex.settings import get_path_scan_threshold

    if min_score is None:
        min_score = get_path_scan_threshold()

    use_rich = use_rich_output()
    console = setup_logging("code", facility, use_rich, verbose=verbose)
    log_print = make_log_print("code", console)

    try:
        facility_config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    if rejudge_ingested:
        # A re-judge reads the file's text from its stored CodeChunks, so it
        # needs neither the facility hop nor the scan/triage/enrich stages: it
        # takes the ingested files that carry no recorded answer and rewrites
        # the answer in place. The reset that clears a fresh request's judgments
        # is the explicit --reset-to ingested step (owner: reset_to_status), run
        # once when a re-judge is requested; a plain --rejudge-ingested leaves
        # the recorded answers alone so a following pass resumes on the files a
        # prior pass left unanswered rather than restarting the whole set.
        from imas_codex.cli.shutdown import safe_asyncio_run
        from imas_codex.discovery.base.reset import CODE_RESET_SPECS, reset_to_status
        from imas_codex.discovery.code.workers import rejudge_ingested_files

        prefixes = list(path_prefixes) or None
        if reset_to == "ingested":
            cleared = reset_to_status(
                CODE_RESET_SPECS["ingested"], facility, path_prefixes=prefixes
            )
            log_print(
                f"[yellow]Cleared the content judgment of {cleared} ingested "
                "file(s) for re-judging[/yellow]"
            )
        result = safe_asyncio_run(
            rejudge_ingested_files(
                facility, path_prefixes=prefixes, cost_limit=cost_limit
            )
        )
        log_print(
            f"\n  [green]{result['rejudged']} ingested file(s) re-judged, "
            f"{len(result['below_gate'])} below the admission gates[/green]"
        )
        log_print(
            f"  [dim]Cost: ${result['cost']:.3f}, batches: {result['batches']}[/dim]"
        )
        for path in result["below_gate"]:
            log_print(f"  [dim]below gate: {path}[/dim]")
        return

    ssh_host = facility_config.get("ssh_host")
    if not ssh_host:
        log_print(f"[red]No SSH host configured for {facility}[/red]")
        raise SystemExit(1)

    ensure_remote_environment(facility_config)

    if rescan:
        from imas_codex.discovery.code.graph_ops import set_files_scan_after

        set_files_scan_after(facility)
        log_print(
            "[yellow]Rescan enabled — previously scanned paths will be re-processed[/yellow]"
        )

    # Handle --reset-to: reset code files to a target state
    if reset_to:
        from imas_codex.discovery.base.reset import CODE_RESET_SPECS, reset_to_status

        spec = CODE_RESET_SPECS[reset_to]
        reset_count = reset_to_status(
            spec, facility, path_prefixes=list(path_prefixes) or None
        )
        if reset_count > 0:
            log_print(
                f"[yellow]Reset {reset_count} file(s) to '{reset_to}' for reprocessing[/yellow]"
            )
        else:
            log_print(f"[dim]No files to reset to '{reset_to}'[/dim]")

    deadline: float | None = None
    if time_limit is not None:
        deadline = time.time() + (time_limit * 60)

    try:
        from imas_codex.discovery.code.parallel import run_parallel_code_discovery

        # Build display (or None for plain mode)
        display = None
        if use_rich:
            from imas_codex.discovery.code.progress import FileProgressDisplay

            display = FileProgressDisplay(
                facility=facility,
                cost_limit=cost_limit,
                focus=focus or "",
                console=console,
                scan_only=scan_only,
                score_only=score_only,
                min_score=min_score,
            )

        def _llm_check() -> tuple[bool, str]:
            return llm_health_check_with_decisions("discovery-score")

        disc_config = DiscoveryConfig(
            domain="code",
            facility=facility,
            facility_config=facility_config,
            model_section="discovery-score",
            display=display,
            check_graph=True,
            check_embed=not scan_only and not score_only,
            check_model=False,  # the llm row is registered below, probing both seats
            check_ssh=True,
            check_auth=False,
            extra_service_checks=(
                []
                if scan_only
                else [
                    (
                        "llm",
                        _llm_check,
                        {"poll_interval": 60.0, "critical": False},
                    )
                ]
            ),
            suppress_loggers=[
                "imas_codex.embeddings",
                "imas_codex.discovery.code.scanner",
                "imas_codex.discovery.code.enrichment",
                "imas_codex.discovery.code.scorer",
                "imas_codex.discovery.code.graph_ops",
            ],
            verbose=verbose,
        )

        # Callbacks — wire to display or logging
        file_logger = logging.getLogger("imas_codex.discovery.code")

        if display:

            def on_scan(msg, stats, results=None):
                display.update_scan(msg, stats, results)

            def on_triage(msg, stats, results=None):
                display.update_triage(msg, stats, results)

            def on_score(msg, stats, results=None):
                display.update_score(msg, stats, results)

            def on_code(msg, stats, results=None):
                display.update_code(msg, stats, results)

            def on_enrich(msg, stats, results=None):
                display.update_enrich(msg, stats, results)

            def on_embed(msg, stats, results=None):
                display.update_embed(msg, stats, results)

            def on_worker_status(worker_group):
                display.update_worker_status(worker_group)
        else:
            log_print(f"\n[bold]Code Discovery: {facility}[/bold]")
            log_print(f"  SSH host: {ssh_host}")
            log_print(f"  Min score: {min_score}")
            log_print(f"  Cost limit: ${cost_limit:.2f}")
            if focus:
                log_print(f"  Focus: {focus}")
            log_print("")

            def on_scan(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("SCAN: %s", msg)

            def on_triage(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("TRIAGE: %s", msg)

            def on_score(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("SCORE: %s", msg)

            def on_code(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("CODE: %s", msg)

            def on_enrich(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("ENRICH: %s", msg)

            def on_embed(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("EMBED: %s", msg)

            on_worker_status = None

        async def async_main(stop_event, service_monitor):
            return await run_parallel_code_discovery(
                facility=facility,
                ssh_host=ssh_host,
                cost_limit=cost_limit,
                min_score=min_score,
                max_paths=max_paths,
                focus=focus,
                path_prefixes=list(path_prefixes) or None,
                num_scan_workers=scan_workers,
                num_triage_workers=triage_workers,
                num_enrich_workers=enrich_workers,
                num_score_workers=score_workers,
                num_code_workers=code_workers,
                scan_only=scan_only,
                score_only=score_only,
                **(
                    {
                        "triage_batch_size": triage_batch_size,
                    }
                    if triage_batch_size is not None
                    else {}
                ),
                deadline=deadline,
                on_scan_progress=on_scan,
                on_triage_progress=on_triage,
                on_score_progress=on_score,
                on_enrich_progress=on_enrich,
                on_code_progress=on_code,
                on_embed_progress=on_embed,
                on_worker_status=on_worker_status,
                stop_event=stop_event,
            )

        result = run_discovery(disc_config, async_main)

        # Final output
        scanned = result.get("scanned", 0)
        scored = result.get("scored", 0)
        code_ingested = result.get("code_ingested", 0)
        cost = result.get("cost", 0)
        elapsed = result.get("elapsed_seconds", 0)

        log_print(
            f"\n  [green]{scanned} scanned, {scored} scored, "
            f"{code_ingested} code ingested[/green]"
        )
        log_print(f"  [dim]Cost: ${cost:.2f}, Time: {elapsed:.1f}s[/dim]")

    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e

    log_print("\n[green]Code discovery complete.[/green]")

"""Document discovery command: scan, fetch images, VLM captioning."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import click

from imas_codex.cli.discover.common import reset_to_option, resolve_focus_items

logger = logging.getLogger(__name__)


def _validate_focus(facility: str, prefixes: list[str]) -> None:
    """Refuse path prefixes that match no image Document at this facility."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $prefixes AS prefix
            OPTIONAL MATCH (d:Document {facility_id: $facility, document_type: 'image'})
            WHERE d.path STARTS WITH prefix
            RETURN prefix, count(d) AS matches
            """,
            facility=facility,
            prefixes=prefixes,
        )
    found = {row["prefix"] for row in rows if row["matches"]}
    missing = [prefix for prefix in prefixes if prefix not in found]
    if missing:
        raise click.UsageError(
            "--focus item filter names unknown Document path(s): " + ", ".join(missing)
        )


@dataclass(frozen=True)
class DocumentsOptions:
    """Settled options for the documents discovery stage.

    The click command builds one of these and hands it to
    :func:`run_documents_stage`, which is also what the discover sequence
    runner calls directly. Field names match the settled spellings rather
    than the retired per-domain flags.
    """

    min_score: float = 0.5
    limit: int = 50
    cost_limit: float = 2.0
    workers: int = 2
    vlm_workers: int = 1
    store_bytes: bool = False
    scan_only: bool = False
    flush: bool = False
    topic: str | None = None
    focus: tuple[str, ...] = ()
    time_limit: int | None = None
    verbose: bool = False
    reset_to: str | None = None


def run_documents_stage(facility: str, options: DocumentsOptions) -> None:
    """Run the documents discovery stage for one facility.

    The stage has two halves. The seeding half enumerates document and image
    files and creates ``Document`` nodes. The draining half fetches images and
    runs VLM captioning and scoring over the nodes the seed half created.

    ``--scan-only`` runs the seeding half and stops. ``--flush`` runs the
    draining half without seeding. Neither flag runs both halves in order.

    Args:
        facility: Facility id to discover documents for.
        options: Settled stage options.

    Raises:
        click.UsageError: When a focused path matches no Document or when
            ``--scan-only`` and ``--flush`` are combined.
    """
    if options.scan_only and options.flush:
        raise click.UsageError("--scan-only and --flush are mutually exclusive")
    path_prefixes = resolve_focus_items(options.focus)
    if path_prefixes:
        _validate_focus(facility, path_prefixes)

    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.discovery.base.facility import get_facility

    use_rich = use_rich_output()
    console = setup_logging(
        "documents", facility, use_rich=use_rich, verbose=options.verbose
    )
    log_print = make_log_print("documents", console)

    try:
        config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    ssh_host = config.get("ssh_host")
    if not ssh_host:
        log_print(f"[red]No SSH host configured for {facility}[/red]")
        raise SystemExit(1)

    # Handle --reset-to: reset documents to a target state
    if options.reset_to:
        from imas_codex.discovery.base.reset import (
            DOCUMENT_RESET_SPECS,
            reset_to_status,
        )

        spec = DOCUMENT_RESET_SPECS[options.reset_to]
        reset_count = reset_to_status(spec, facility)
        if reset_count > 0:
            log_print(
                f"[yellow]Reset {reset_count} document(s) to "
                f"'{options.reset_to}' for reprocessing[/yellow]"
            )
        else:
            log_print(f"[dim]No documents to reset to '{options.reset_to}'[/dim]")

    deadline: float | None = None
    if options.time_limit is not None:
        deadline = time.time() + (options.time_limit * 60)

    try:
        # Seeding half: scan for document files (synchronous). Skipped by
        # --flush, which drains work a previous seed half already created.
        if not options.flush:
            from imas_codex.discovery.documents.scanner import (
                scan_facility_documents,
            )

            log_print(f"\n[bold]Document Discovery: {facility}[/bold]")
            log_print(f"  SSH host: {ssh_host}")
            log_print(f"  Min score: {options.min_score}")
            log_print(f"  Cost limit: ${options.cost_limit:.2f}")
            if options.topic:
                log_print(f"  Topic: {options.topic}")
            log_print("")

            scan_stats = scan_facility_documents(
                facility=facility,
                min_score=options.min_score,
                max_paths=options.limit,
                ssh_host=ssh_host,
            )

            log_print(
                f"  [green]Scanned: {scan_stats['new_files']} new documents "
                f"in {scan_stats['total_paths']} paths[/green]"
            )

            if options.scan_only:
                log_print("\n[green]Document scan complete (--scan-only).[/green]")
                return

        # Draining half: process images (fetch + VLM captioning) via harness
        from imas_codex.discovery.documents.pipeline import (
            DocumentDiscoveryState,
            run_document_discovery,
        )

        # Create state externally so the display can observe it
        state = DocumentDiscoveryState(
            facility=facility,
            ssh_host=ssh_host,
            cost_limit=options.cost_limit,
            min_score=options.min_score,
            deadline=deadline,
            store_images=options.store_bytes,
            scan_only=False,
            focus=options.topic,
            path_prefixes=tuple(path_prefixes) if path_prefixes else None,
        )

        # Build display for rich mode
        display = None
        if use_rich:
            from imas_codex.discovery.base.progress import (
                DataDrivenProgressDisplay,
                StageDisplaySpec,
            )

            display = DataDrivenProgressDisplay(
                facility=facility,
                cost_limit=options.cost_limit,
                stages=[
                    StageDisplaySpec(
                        "FETCH", "bold blue", "image", "image_stats", "image_phase"
                    ),
                    StageDisplaySpec(
                        "VLM",
                        "bold magenta",
                        "vlm",
                        "image_score_stats",
                        "image_score_phase",
                    ),
                ],
                console=console,
                focus=options.topic or "",
                title_suffix="Document Discovery",
            )
            display.set_engine_state(state)

        disc_config = DiscoveryConfig(
            domain="documents",
            facility=facility,
            facility_config=config,
            model_section="discovery-vision",
            display=display,
            check_graph=False,
            check_embed=False,
            check_ssh=False,
            verbose=options.verbose,
        )

        async def async_main(stop_event, service_monitor):
            return await run_document_discovery(
                state,
                num_image_workers=options.workers,
                num_vlm_workers=options.vlm_workers,
                stop_event=stop_event,
                on_worker_status=(display.update_worker_status if display else None),
            )

        result = run_discovery(disc_config, async_main)

        if not display:
            fetched = result.get("images_fetched", 0)
            captioned = result.get("images_captioned", 0)
            cost = result.get("cost", 0)
            elapsed = result.get("elapsed_seconds", 0)

            log_print(
                f"\n  [green]{fetched} images fetched, {captioned} captioned[/green]"
            )
            log_print(f"  [dim]Cost: ${cost:.2f}, Time: {elapsed:.1f}s[/dim]")

    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        if options.verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e

    log_print("\n[green]Document discovery complete.[/green]")


@click.command()
@click.argument("facility")
@click.option(
    "--min-score",
    type=float,
    default=0.5,
    help="Minimum FacilityPath score to include (default: 0.5)",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="Maximum number of items to process (default: 50)",
)
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=2.0,
    help="Maximum VLM spend in USD (default: 2.0)",
)
@click.option(
    "--workers",
    type=int,
    default=2,
    help="Number of parallel image fetch workers (default: 2)",
)
@click.option(
    "--vlm-workers",
    type=int,
    default=1,
    help="Number of parallel VLM captioning workers (default: 1)",
)
@click.option(
    "--store-bytes",
    is_flag=True,
    default=False,
    help="Keep image bytes in graph after VLM scoring (default: clear)",
)
@click.option(
    "--scan-only",
    is_flag=True,
    help="Only scan for document files, skip image processing",
)
@click.option(
    "--flush",
    is_flag=True,
    help="Only process image Documents a previous scan created, skip scanning",
)
@click.option(
    "--topic",
    type=str,
    default=None,
    help="Free-text steer for VLM scoring (e.g. 'diagnostics', 'equilibrium')",
)
@click.option(
    "--focus",
    "focus",
    multiple=True,
    help="Restrict to Document path prefixes",
)
@click.option(
    "--time",
    "time_limit",
    default=None,
    type=int,
    help="Maximum runtime in minutes",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@reset_to_option("documents")
def documents(
    facility: str,
    min_score: float,
    limit: int,
    cost_limit: float,
    workers: int,
    vlm_workers: int,
    store_bytes: bool,
    scan_only: bool,
    flush: bool,
    topic: str | None,
    focus: tuple[str, ...],
    time_limit: int | None,
    verbose: bool,
    reset_to: str | None = None,
) -> None:
    """Discover documents and images from scored facility paths.

    Scans for document files (PDF, Markdown, notebooks) and images
    (PNG, JPG, SVG, etc.) in scored FacilityPaths. Images are fetched,
    downsampled, and optionally captioned with a VLM.

    \b
    Pipeline stages:
      SCAN:    SSH enumerate document + image files, create Document nodes
      FETCH:   Download images via SCP, create Image nodes
      CAPTION: VLM captioning and relevance scoring

    \b
    Examples:
      imas-codex discover documents tcv
      imas-codex discover documents tcv --scan-only
      imas-codex discover documents tcv --flush
      imas-codex discover documents tcv -c 1.0 --vlm-workers 2
      imas-codex discover documents tcv --topic diagnostics
    """
    run_documents_stage(
        facility,
        DocumentsOptions(
            min_score=min_score,
            limit=limit,
            cost_limit=cost_limit,
            workers=workers,
            vlm_workers=vlm_workers,
            store_bytes=store_bytes,
            scan_only=scan_only,
            flush=flush,
            topic=topic,
            focus=focus,
            time_limit=time_limit,
            verbose=verbose,
            reset_to=reset_to,
        ),
    )

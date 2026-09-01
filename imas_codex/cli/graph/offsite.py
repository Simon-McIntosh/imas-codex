"""Scheduled, count-verified full-graph offsite pushes."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import click
from click.testing import CliRunner

from imas_codex.graph.ghcr import get_git_info, get_registry
from imas_codex.graph.neo4j_ops import (
    OffsiteCurrency,
    get_offsite_currency,
    graph_archive_stamp,
)
from imas_codex.graph.offsite import OffsiteCountMismatch, run_offsite_push_cycle

_JOB_NAME = "codex-offsite-push"
_WEEKLY_DELAY = "now+7days"


def _scheduled_command() -> str:
    """Return the job body that runs once and submits its next occurrence."""
    return (
        "uv run --no-sync imas-codex graph offsite-push\n"
        "cycle_status=$?\n"
        f'sbatch --begin={_WEEKLY_DELAY} "$0"\n'
        "schedule_status=$?\n"
        'if [ "$cycle_status" -ne 0 ]; then exit "$cycle_status"; fi\n'
        'exit "$schedule_status"'
    )


def _schedule_preview() -> str:
    """Render the material SLURM script lines shown by dry-run."""
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={_JOB_NAME}\n"
        "#SBATCH --cpus-per-task=2\n"
        "#SBATCH --mem=8G\n"
        "# partition and log path are resolved by the graph service helper\n"
        "cd $HOME/Code/imas-codex\n"
        "source .env 2>/dev/null || true\n"
        f"{_scheduled_command()}"
    )


def _submit_schedule() -> None:
    from imas_codex.cli.services import _submit_service_job

    _submit_service_job(
        _JOB_NAME,
        _scheduled_command(),
        cpus=2,
        mem="8G",
    )


def _invoke(command, args: list[str], operation: str) -> None:
    result = CliRunner().invoke(command, args)
    if result.exit_code == 0:
        if result.output:
            click.echo(result.output, nl=not result.output.endswith("\n"))
        return
    detail = result.output.strip()
    if not detail and result.exception:
        detail = f"{type(result.exception).__name__}: {result.exception}"
    raise click.ClickException(f"{operation} failed: {detail}")


def _export_archive(stamp: str) -> Path:
    from imas_codex.cli.graph.data import graph_export
    from imas_codex.graph.dirs import ensure_exports_dir

    archive = ensure_exports_dir() / f"imas-codex-graph-{stamp}.tar.gz"
    _invoke(
        graph_export,
        [
            "--output",
            str(archive),
            "--version-label",
            stamp,
            "--archive-dir-name",
            f"imas-codex-graph-{stamp}",
        ],
        "graph export",
    )
    return archive


def _extract_dump(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = [
            member
            for member in bundle.getmembers()
            if member.isfile() and Path(member.name).name == "graph.dump"
        ]
        if len(members) != 1:
            raise click.ClickException(
                f"Expected one graph.dump in {archive}, found {len(members)}"
            )
        stream = bundle.extractfile(members[0])
        if stream is None:
            raise click.ClickException(f"Cannot read graph.dump in {archive}")
        with target.open("wb") as output:
            shutil.copyfileobj(stream, output)


def _push_archive(
    archive: Path,
    stamp: str,
    *,
    registry: str,
    token: str | None,
) -> str:
    from imas_codex.cli.graph.registry import graph_push

    with tempfile.TemporaryDirectory() as temporary:
        source_dump = Path(temporary) / "graph.dump"
        _extract_dump(archive, source_dump)
        version_tag = f"{stamp}-r1"
        args = [
            "--dev",
            "--version",
            version_tag,
            "--registry",
            registry,
            "--source-dump",
            str(source_dump),
            "--message",
            "Scheduled full-graph offsite recovery point",
        ]
        if token:
            args.extend(["--token", token])
        _invoke(graph_push, args, "graph push")
    return f"{registry}/imas-codex-graph:{version_tag}"


def _decision_text(status: str, age_seconds: float | None) -> str:
    if status == "current":
        return "no-op: live graph has not changed since the newest offsite copy"
    if status == "no_offsite":
        return "push: no full-graph offsite copy exists"
    days = (age_seconds or 0.0) / 86_400
    return f"push: newest full-graph offsite copy is {days:.3f} days stale"


def _read_currency(
    registry: str, token: str | None
) -> tuple[OffsiteCurrency, str | None]:
    """Read registry currency, recovering from an invalid stored token."""
    try:
        return get_offsite_currency(registry, token), token
    except click.ClickException as exc:
        if "401" not in exc.format_message():
            raise

    credential = subprocess.run(
        ["gh", "auth", "token"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    fallback = credential.stdout.strip()
    if credential.returncode != 0 or not fallback:
        raise click.ClickException(
            "The configured GHCR token was rejected and gh has no usable credential."
        )
    return get_offsite_currency(registry, fallback), fallback


@click.command("offsite-push")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN", hidden=True)
@click.option(
    "--schedule",
    is_flag=True,
    help="Submit the weekly self-resubmitting SLURM job.",
)
@click.option("--dry-run", is_flag=True, help="Show the decision and job script.")
def graph_offsite_push(
    registry: str | None,
    token: str | None,
    schedule: bool,
    dry_run: bool,
) -> None:
    """Run or schedule one verified full-graph offsite push cycle."""
    target_registry = get_registry(get_git_info(), registry)
    currency, token = _read_currency(target_registry, token)
    click.echo(f"Decision: {_decision_text(currency.status, currency.age_seconds)}")

    if dry_run:
        click.echo(
            f"Archive stamp: {graph_archive_stamp()} "
            "(a fresh UTC stamp is generated when the cycle runs)"
        )
        click.echo("\n[DRY RUN] SLURM script:")
        click.echo(_schedule_preview())
        click.echo("\n[DRY RUN] No archive was exported or pushed.")
        return

    if schedule:
        _submit_schedule()
        click.echo(
            "Scheduled one immediate offsite cycle; each job submits its successor "
            "for seven days later."
        )
        return

    try:
        result = run_offsite_push_cycle(
            currency=currency,
            export_archive=_export_archive,
            push_archive=lambda archive, stamp: _push_archive(
                archive,
                stamp,
                registry=target_registry,
                token=token,
            ),
        )
    except OffsiteCountMismatch as exc:
        raise click.ClickException(str(exc)) from exc

    if result.outcome == "no_op":
        click.echo(f"No-op receipt: {result.receipt_path}")
    else:
        click.echo(f"Pushed: {result.archive_ref}")
        click.echo(
            f"Archive: {result.archive_bytes} bytes in {result.wall_time_seconds:.3f} s"
        )
        click.echo(f"Receipt: {result.receipt_path}")

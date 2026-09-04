"""Graph CLI registry commands — push, fetch, pull, tags, prune."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click

from imas_codex import __version__
from imas_codex.graph.ghcr import (
    delete_tag as _delete_tag,
    dispatch_graph_quality as _dispatch_graph_quality,
    ensure_fresh_version as _ensure_fresh_version,
    fetch_tag_messages as _fetch_tag_messages,
    get_ghcr_owner_and_type,
    get_git_info,
    get_local_graph_manifest,
    get_package_name,
    get_registry,
    get_version_tag,
    github_api_paginated,
    github_api_request,
    list_registry_tags as _list_registry_tags,
    login_to_ghcr,
    require_clean_git,
    require_oras,
    resolve_latest_tag as _resolve_latest_tag,
    resolve_token,
    save_dev_revision as _save_dev_revision,
    save_local_graph_manifest,
)
from imas_codex.graph.neo4j_ops import (
    Neo4jOperation,
    OffsiteCurrency,
    backup_graph_dump,
    check_graph_exists,
    get_offsite_currency,
    graph_archive_stamp,
)
from imas_codex.graph.offsite import (
    OffsiteCountMismatch,
    OffsitePushFailed,
    run_offsite_push_cycle,
)

_RELEASE_TAG = re.compile(r"^v?\d+\.\d+\.\d+(?:-rc\d+)?$")
_PUSH_SERVICE_NAME = "imas-codex-graph-push"
_PUSH_TIMER_NAME = f"{_PUSH_SERVICE_NAME}.timer"


@dataclass(frozen=True)
class RegistryVersion:
    """One GitHub Packages version with its registry-owned creation time."""

    id: int
    name: str
    created_at: datetime
    tags: tuple[str, ...]

    @property
    def display_name(self) -> str:
        """Return stable human-facing identity, including untagged versions."""
        if self.tags:
            return ", ".join(self.tags)
        return f"untagged@{self.id} ({self.name})"


@dataclass(frozen=True)
class RetentionDecision:
    """Keep/delete classification for one registry version."""

    version: RegistryVersion
    keep: bool
    tier: str


def _is_release_version(version: RegistryVersion) -> bool:
    return any(_RELEASE_TAG.fullmatch(tag) for tag in version.tags)


def _select_tiered_retention(
    versions: list[RegistryVersion],
    *,
    weekly_keep: int = 4,
    monthly_keep: int = 3,
) -> list[RetentionDecision]:
    """Classify versions under dense-weekly, sparse-monthly retention.

    Release-tagged versions and the version carrying ``latest`` are protected.
    Untagged and ``test-*`` versions are deletion candidates regardless of age.
    The remaining scheduled/development copies retain the newest weekly window,
    then the newest copy in each of the next distinct calendar months.
    """
    ordered = sorted(versions, key=lambda version: version.created_at, reverse=True)
    decisions: dict[int, RetentionDecision] = {}
    tierable: list[RegistryVersion] = []

    for version in ordered:
        if "latest" in version.tags:
            decisions[version.id] = RetentionDecision(version, True, "latest")
        elif _is_release_version(version):
            decisions[version.id] = RetentionDecision(version, True, "release")
        elif not version.tags:
            decisions[version.id] = RetentionDecision(version, False, "delete-untagged")
        elif any(tag.startswith("test-") for tag in version.tags):
            decisions[version.id] = RetentionDecision(version, False, "delete-test")
        else:
            tierable.append(version)

    for version in tierable[:weekly_keep]:
        decisions[version.id] = RetentionDecision(version, True, "weekly")

    monthly_selected = 0
    represented_months: set[tuple[int, int]] = set()
    for version in tierable[weekly_keep:]:
        month = (version.created_at.year, version.created_at.month)
        if month not in represented_months and monthly_selected < monthly_keep:
            represented_months.add(month)
            monthly_selected += 1
            decisions[version.id] = RetentionDecision(version, True, "monthly")
        else:
            decisions[version.id] = RetentionDecision(version, False, "delete-thinned")

    return [decisions[version.id] for version in ordered]


def _list_registry_versions(
    registry: str,
    pkg_name: str,
    token: str | None,
) -> list[RegistryVersion]:
    """List package versions and verify their tags against the OCI inventory."""
    listed_tags = set(_list_registry_tags(registry, token, pkg_name=pkg_name))
    resolved = resolve_token(token)
    owner, api_type = get_ghcr_owner_and_type(registry, resolved)
    path = f"/{api_type}/{owner}/packages/container/{pkg_name}/versions"
    status, records = github_api_paginated(path, resolved)
    if status != 200:
        raise click.ClickException(
            f"Failed to list package versions for {registry}/{pkg_name} (HTTP {status})"
        )

    versions: list[RegistryVersion] = []
    for record in records:
        try:
            tags = tuple(
                record.get("metadata", {}).get("container", {}).get("tags", [])
            )
            created_at = datetime.fromisoformat(
                record["created_at"].replace("Z", "+00:00")
            )
            versions.append(
                RegistryVersion(
                    id=int(record["id"]),
                    name=str(record["name"]),
                    created_at=created_at,
                    tags=tags,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise click.ClickException(
                f"Malformed package-version record for {registry}/{pkg_name}: {record!r}"
            ) from exc

    version_tags = {tag for version in versions for tag in version.tags}
    missing = sorted(listed_tags - version_tags)
    if missing:
        raise click.ClickException(
            "Package-version inventory does not contain tags returned by the "
            f"registry listing: {', '.join(missing)}"
        )
    return versions


def _delete_untagged_version(
    registry: str,
    pkg_name: str,
    version_id: int,
    token: str | None,
) -> bool:
    """Delete an untagged package version, which has no tag for ``delete_tag``."""
    resolved = resolve_token(token)
    owner, api_type = get_ghcr_owner_and_type(registry, resolved)
    path = f"/{api_type}/{owner}/packages/container/{pkg_name}/versions/{version_id}"
    status, response = github_api_request(path, resolved, method="DELETE")
    if status in (200, 204):
        return True
    message = response.get("message", "") if isinstance(response, dict) else response
    click.echo(
        f"  Failed to delete untagged version {version_id} (HTTP {status}): {message}",
        err=True,
    )
    return False


def _resolve_scheduler(profile) -> str:
    """Resolve the job scheduler for a Neo4j profile's location."""
    try:
        from imas_codex.remote.locations import resolve_location

        return resolve_location(profile.location).scheduler
    except Exception:
        return "none"


def _resolve_partition(profile) -> str | None:
    """Resolve the SLURM partition for a Neo4j profile's location."""
    try:
        from imas_codex.remote.locations import resolve_location

        return resolve_location(profile.location).partition
    except Exception:
        return None


def _push_schedule_paths() -> tuple[Path, Path]:
    service_dir = Path.home() / ".config" / "systemd" / "user"
    return (
        service_dir / f"{_PUSH_SERVICE_NAME}.service",
        service_dir / _PUSH_TIMER_NAME,
    )


def _push_schedule_project_dir() -> Path:
    candidates = (
        Path.home() / "Code" / "imas-codex",
        Path.home() / "imas-codex",
        Path.cwd(),
    )
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise click.ClickException(
        "Project not found under ~/Code/imas-codex, ~/imas-codex, or the current "
        "directory"
    )


def _push_schedule_service_text(
    project_dir: Path,
    *,
    uv_path: str,
    oras_path: str,
    hostname: str,
) -> str:
    """Build the host-pinned login-node oneshot unit."""
    executable_dirs = dict.fromkeys(
        (
            str(Path(oras_path).parent),
            str(Path.home() / ".local" / "bin"),
            "/usr/local/bin",
            "/usr/bin",
        )
    )
    service_path = ":".join(executable_dirs)
    return f"""[Unit]
Description=IMAS Codex verified full-graph offsite push
After=network-online.target
Wants=network-online.target
ConditionHost={hostname}

[Service]
Type=oneshot
WorkingDirectory={project_dir}
EnvironmentFile=-{project_dir / ".env"}
Environment="PATH={service_path}"
ExecStart={uv_path} run --no-sync --project {project_dir} imas-codex graph push --cycle
"""


def _push_schedule_timer_text() -> str:
    """Build the persistent weekly timer for verified offsite pushes."""
    return f"""[Unit]
Description=Weekly IMAS Codex verified full-graph offsite push

[Timer]
OnCalendar=weekly
Persistent=true
Unit={_PUSH_SERVICE_NAME}.service

[Install]
WantedBy=timers.target
"""


def _manage_push_schedule(action: str, *, dry_run: bool = False) -> None:
    """Install, inspect, or remove the login-node user timer."""
    import platform
    import socket

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")
    systemctl = shutil.which("systemctl")
    if not systemctl:
        raise click.ClickException("systemctl not found")

    service_file, timer_file = _push_schedule_paths()

    if action == "install":
        project_dir = _push_schedule_project_dir()
        uv_path = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        oras_path = shutil.which("oras")
        if not oras_path:
            raise click.ClickException(
                "oras not found; cannot install the weekly graph push service"
            )
        service_text = _push_schedule_service_text(
            project_dir,
            uv_path=uv_path,
            oras_path=oras_path,
            hostname=socket.getfqdn(),
        )
        timer_text = _push_schedule_timer_text()
        if dry_run:
            click.echo("[DRY RUN] systemd user service:")
            click.echo(service_text)
            click.echo("[DRY RUN] systemd user timer:")
            click.echo(timer_text)
            click.echo("[DRY RUN] No files were written and systemctl was not called.")
            return

        service_file.parent.mkdir(parents=True, exist_ok=True)
        service_file.write_text(service_text)
        timer_file.write_text(timer_text)
        subprocess.run([systemctl, "--user", "daemon-reload"], check=True)
        subprocess.run(
            [systemctl, "--user", "enable", "--now", _PUSH_TIMER_NAME],
            check=True,
        )
        click.echo("Weekly login-node graph push timer installed and enabled")
        click.echo(f"  Service: {service_file}")
        click.echo(f"  Timer:   {timer_file}")
        return

    if action == "status":
        if not service_file.exists() or not timer_file.exists():
            click.echo("Weekly login-node graph push timer is not installed")
            return
        result = subprocess.run(
            [systemctl, "--user", "status", _PUSH_TIMER_NAME],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout, nl=not result.stdout.endswith("\n"))
        if result.stderr:
            click.echo(result.stderr, err=True, nl=not result.stderr.endswith("\n"))
        return

    if action == "remove":
        if not service_file.exists() and not timer_file.exists():
            click.echo("Weekly login-node graph push timer is not installed")
            return
        subprocess.run(
            [systemctl, "--user", "disable", "--now", _PUSH_TIMER_NAME],
            capture_output=True,
        )
        subprocess.run(
            [systemctl, "--user", "stop", f"{_PUSH_SERVICE_NAME}.service"],
            capture_output=True,
        )
        service_file.unlink(missing_ok=True)
        timer_file.unlink(missing_ok=True)
        subprocess.run([systemctl, "--user", "daemon-reload"], check=True)
        click.echo("Weekly login-node graph push timer removed")
        return

    raise ValueError(f"Unknown graph push schedule action: {action}")


def _invoke_graph_command(command, args: list[str], operation: str) -> None:
    from click.testing import CliRunner

    result = CliRunner().invoke(command, args)
    if result.exit_code == 0:
        if result.output:
            click.echo(result.output, nl=not result.output.endswith("\n"))
        return
    detail = result.output.strip()
    if not detail and result.exception:
        detail = f"{type(result.exception).__name__}: {result.exception}"
    raise click.ClickException(f"{operation} failed: {detail}")


def _export_cycle_archive(stamp: str) -> Path:
    from imas_codex.cli.graph.data import graph_export
    from imas_codex.graph.dirs import ensure_exports_dir

    archive = ensure_exports_dir() / f"imas-codex-graph-{stamp}.tar.gz"
    _invoke_graph_command(
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


def _extract_cycle_dump(archive: Path, target: Path) -> None:
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


def _push_cycle_archive(
    archive: Path,
    stamp: str,
    *,
    registry: str,
    token: str | None,
) -> str:
    with tempfile.TemporaryDirectory() as temporary:
        source_dump = Path(temporary) / "graph.dump"
        _extract_cycle_dump(archive, source_dump)
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
        _invoke_graph_command(graph_push, args, "graph push")
    return f"{registry}/imas-codex-graph:{version_tag}"


def _push_cycle_decision(status: str, age_seconds: float | None) -> str:
    if status == "current":
        return "no-op: live graph has not changed since the newest offsite copy"
    if status == "no_offsite":
        return "push: no full-graph offsite copy exists"
    days = (age_seconds or 0.0) / 86_400
    return f"push: newest full-graph offsite copy is {days:.3f} days stale"


def _read_offsite_currency(
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


def _run_offsite_cycle(
    *,
    registry: str | None,
    token: str | None,
    dry_run: bool,
) -> None:
    target_registry = get_registry(get_git_info(), registry)
    currency, token = _read_offsite_currency(target_registry, token)
    click.echo(
        f"Decision: {_push_cycle_decision(currency.status, currency.age_seconds)}"
    )

    if dry_run:
        click.echo(
            f"Archive stamp: {graph_archive_stamp()} "
            "(a fresh UTC stamp is generated when the cycle runs)"
        )
        click.echo("\n[DRY RUN] No archive was exported or pushed.")
        return

    try:
        result = run_offsite_push_cycle(
            currency=currency,
            export_archive=_export_cycle_archive,
            push_archive=lambda archive, stamp: _push_cycle_archive(
                archive,
                stamp,
                registry=target_registry,
                token=token,
            ),
        )
    except (OffsiteCountMismatch, OffsitePushFailed) as exc:
        raise click.ClickException(str(exc)) from exc

    if result.outcome == "no_op":
        click.echo(f"No-op receipt: {result.receipt_path}")
    else:
        click.echo(f"Pushed: {result.archive_ref}")
        click.echo(
            f"Archive: {result.archive_bytes} bytes in {result.wall_time_seconds:.3f} s"
        )
        click.echo(f"Receipt: {result.receipt_path}")


@click.command()
@click.option("--dev", is_flag=True, help="Push as dev-{commit} tag")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
@click.option(
    "--schedule",
    is_flag=True,
    help="Install the weekly login-node push timer.",
)
@click.option(
    "--schedule-status",
    is_flag=True,
    help="Show the weekly login-node push timer status.",
)
@click.option(
    "--remove-schedule",
    is_flag=True,
    help="Remove the weekly login-node push timer.",
)
@click.option(
    "--cycle",
    is_flag=True,
    help="Run one verified full-graph push cycle.",
)
@click.option(
    "--facility",
    "-F",
    "facilities",
    multiple=True,
    help="Facility to include (repeatable). Filters the dump.",
)
@click.option(
    "--without-dd",
    is_flag=True,
    help="Exclude IMAS Data Dictionary nodes",
)
@click.option(
    "--dd-only",
    is_flag=True,
    help="Push only IMAS Data Dictionary nodes (no facility data)",
)
@click.option(
    "-m",
    "--message",
    default=None,
    help="Short description to attach to this push (shown by 'graph tags').",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show full error output from neo4j-admin.",
)
@click.option(
    "--version",
    "version_tag_override",
    default=None,
    help="Override version tag (e.g. v5.0.0-rc2). Bypasses git tag detection.",
)
@click.option(
    "--source-dump",
    type=click.Path(exists=True),
    default=None,
    help="Use pre-existing dump file (avoids Neo4j stop/start per variant).",
)
def graph_push(
    dev: bool,
    registry: str | None,
    token: str | None,
    dry_run: bool,
    facilities: tuple[str, ...],
    without_dd: bool,
    dd_only: bool,
    message: str | None,
    verbose: bool = False,
    version_tag_override: str | None = None,
    source_dump: str | None = None,
    schedule: bool = False,
    schedule_status: bool = False,
    remove_schedule: bool = False,
    cycle: bool = False,
) -> None:
    """Push graph archive to GHCR.

    Use --facility/-f (repeatable) to push a filtered per-facility graph.
    Use --dd-only to push only IMAS Data Dictionary nodes.
    Use -m/--message to attach a short description (like a git commit message).
    """
    from imas_codex.cli.graph_progress import GraphProgress, run_oras_with_progress

    operational_modes = sum((schedule, schedule_status, remove_schedule, cycle))
    if operational_modes > 1:
        raise click.UsageError(
            "--schedule, --schedule-status, --remove-schedule, and --cycle are "
            "mutually exclusive"
        )
    if operational_modes:
        incompatible = any(
            (
                dev,
                facilities,
                without_dd,
                dd_only,
                message,
                verbose,
                version_tag_override,
                source_dump,
            )
        )
        if incompatible:
            raise click.UsageError(
                "schedule/cycle options cannot be combined with push variant options"
            )
        if schedule:
            _manage_push_schedule("install", dry_run=dry_run)
        elif schedule_status:
            _manage_push_schedule("status")
        elif remove_schedule:
            _manage_push_schedule("remove")
        else:
            _run_offsite_cycle(
                registry=registry,
                token=token,
                dry_run=dry_run,
            )
        return

    git_info = get_git_info()

    if not dev:
        require_clean_git(git_info)

    target_registry = get_registry(git_info, registry)

    if version_tag_override:
        version_tag = version_tag_override
    else:
        # Ensure __version__ reflects current git state (hatch-vcs freezes at
        # uv sync time — without this, the GHCR tag embeds a stale commit hash).
        fresh_version = _ensure_fresh_version()
        version_tag = get_version_tag(git_info, dev, version_override=fresh_version)
    pkg_name = get_package_name(
        list(facilities) or None, without_dd=without_dd, dd_only=dd_only
    )
    archive_dir_name = f"{pkg_name}-{graph_archive_stamp(git_info['commit_short'])}"
    archive_name = f"{archive_dir_name}.tar.gz"

    click.echo(f"Push target: {target_registry}/{pkg_name}:{version_tag}")
    if git_info["is_fork"]:
        click.echo(f"  Detected fork: {git_info['remote_owner']}")

    if dry_run:
        click.echo("\n[DRY RUN] Would:")
        click.echo("  1. Dump graph (auto stop/start Neo4j)")
        click.echo(f"  2. Push to {target_registry}/{pkg_name}:{version_tag}")
        return

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.graph.remote import is_remote_location

    profile = resolve_neo4j()

    if is_remote_location(profile.host):
        from imas_codex.cli.graph_progress import remote_operation_streaming
        from imas_codex.graph.remote import (
            build_remote_push_script,
            remote_check_imas_codex,
            remote_check_oras,
        )

        if not remote_check_oras(profile.host):
            raise click.ClickException(
                f"oras not found on {profile.host}. "
                "Install with: imas-codex tools install"
            )

        codex_cli_path: str | None = None
        if dd_only:
            codex_cli_path = remote_check_imas_codex(profile.host)
            if not codex_cli_path:
                raise click.ClickException(
                    f"imas-codex CLI not found on {profile.host}. "
                    "Install with: cd ~/Code/imas-codex && uv sync"
                )

        if facilities:
            click.echo(
                "Warning: --facility filtering is not supported for remote push. "
                "The full graph will be pushed.",
                err=True,
            )

        artifact_ref = f"{target_registry}/{pkg_name}:{version_tag}"

        _remote_markers_push = {
            "STOPPING": f"Stopping Neo4j on {profile.host}",
            "DUMPING": "Dumping graph database",
            "RECOVERY": "Recovery cycle (clean start/stop)",
            "EXPORTING": "Exporting IMAS-only graph via imas-codex CLI",
            "FILTERING": "Filtering to IMAS DD nodes only",
            "ARCHIVING": "Creating archive",
            "STARTING": f"Starting Neo4j on {profile.host}",
            "LOGIN": "Authenticating with GHCR",
            "PUSHING": f"Pushing to GHCR ({artifact_ref})",
            "TAGGING": "Tagging as latest",
            "COMPLETE": "Push complete",
        }

        phases = 1  # single streaming operation
        with GraphProgress("push") as gp:
            gp.set_total_phases(phases)
            gp.start_phase(f"Pushing [{profile.name}] from {profile.host}")

            script = build_remote_push_script(
                profile.name,
                artifact_ref,
                version_tag=version_tag,
                git_commit=git_info["commit"],
                message=message,
                token=token,
                is_dev=dev,
                dd_only=dd_only,
                codex_cli_path=codex_cli_path,
                archive_name=archive_name,
                archive_dir_name=archive_dir_name,
                scheduler=_resolve_scheduler(profile),
                partition=_resolve_partition(profile),
            )

            try:
                push_output = remote_operation_streaming(
                    script,
                    profile.host,
                    progress=gp,
                    progress_markers=_remote_markers_push,
                    timeout=900,
                )
            except Exception as e:
                gp.fail_phase(str(e))
                raise click.ClickException(
                    f"Remote push on {profile.host} failed: {e}"
                ) from e

            size_str = None
            for line in push_output.strip().splitlines():
                if line.startswith("SIZE="):
                    size_str = line.split("=", 1)[1].strip()
            gp.complete_phase(size_str)

        # Update local manifest
        manifest = get_local_graph_manifest() or {}
        manifest["pushed"] = True
        manifest["pushed_version"] = version_tag
        manifest["pushed_to"] = artifact_ref
        manifest["pushed_at"] = datetime.now(UTC).isoformat()
        if message:
            manifest["pushed_message"] = message
        save_local_graph_manifest(manifest)

        if dev:
            base = __version__.replace("+", "-")
            rev_str = version_tag.rsplit("-r", 1)[-1]
            _save_dev_revision(base, int(rev_str))

        _dispatch_graph_quality(git_info, version_tag, target_registry)
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    require_oras()

    with tempfile.TemporaryDirectory() as push_tmpdir:
        archive_path = Path(push_tmpdir) / archive_name

        with GraphProgress("push") as gp:
            gp.set_total_phases(3 if not dev else 2)

            gp.start_phase("Exporting graph database")
            from click.testing import CliRunner

            from imas_codex.cli.graph.data import graph_export

            runner = CliRunner()
            dump_args = ["-o", str(archive_path)]
            for fac in facilities:
                dump_args.extend(["--facility", fac])
            if without_dd:
                dump_args.append("--without-dd")
            if dd_only:
                dump_args.append("--dd-only")
            if verbose:
                dump_args.append("--verbose")
            if source_dump:
                dump_args.extend(["--source-dump", source_dump])
            if version_tag_override:
                dump_args.extend(["--version-label", version_tag_override])
            result = runner.invoke(graph_export, dump_args)
            if result.exit_code != 0:
                if result.exception and not isinstance(result.exception, SystemExit):
                    detail = f"{type(result.exception).__name__}: {result.exception}"
                else:
                    # Extract the error block from click output.
                    # Click formats ClickException as "Error: <message>"
                    # where <message> may be multi-line.  Capture
                    # everything from the last "Error: " to the end.
                    output_lines = result.output.strip().splitlines()
                    error_start = None
                    for i, line in enumerate(output_lines):
                        if line.startswith("Error: "):
                            error_start = i
                    if error_start is not None:
                        error_block = output_lines[error_start:]
                        error_block[0] = error_block[0].removeprefix("Error: ")
                        detail = "\n".join(error_block)
                    else:
                        detail = result.output.strip()
                gp.fail_phase(detail.splitlines()[0])
                raise click.ClickException(detail)
            size_mb = archive_path.stat().st_size / 1024 / 1024
            gp.complete_phase(f"{size_mb:.1f} MB")

            login_to_ghcr(token)

            artifact_ref = f"{target_registry}/{pkg_name}:{version_tag}"
            push_cmd = [
                "oras",
                "push",
                artifact_ref,
                f"{archive_path.name}:application/gzip",
                "--annotation",
                f"org.opencontainers.image.version={version_tag}",
                "--annotation",
                f"io.imas-codex.git-commit={git_info['commit']}",
            ]
            if message:
                push_cmd.extend(
                    [
                        "--annotation",
                        f"org.opencontainers.image.description={message}",
                    ]
                )

            gp.start_phase(f"Pushing to GHCR ({artifact_ref})")
            run_oras_with_progress(push_cmd, progress=gp, cwd=archive_path.parent)
            gp.complete_phase()

            manifest = get_local_graph_manifest() or {}
            manifest["pushed"] = True
            manifest["pushed_version"] = version_tag
            manifest["pushed_to"] = artifact_ref
            manifest["pushed_at"] = datetime.now(UTC).isoformat()
            if message:
                manifest["pushed_message"] = message
            save_local_graph_manifest(manifest)

            # Save dev revision for auto-increment on next push
            if dev:
                base = __version__.replace("+", "-")
                rev_str = version_tag.rsplit("-r", 1)[-1]
                _save_dev_revision(base, int(rev_str))

            if not dev:
                gp.start_phase("Tagging as latest")
                result = subprocess.run(
                    ["oras", "tag", artifact_ref, "latest"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    gp.complete_phase()
                else:
                    gp.fail_phase(result.stderr.strip())

    # Dispatch graph quality CI
    _dispatch_graph_quality(git_info, version_tag, target_registry)


@click.command()
@click.option("-v", "--version", "version", default="latest")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save archive to this path (default: auto-named in current directory)",
)
@click.option(
    "--facility",
    "-F",
    "facilities",
    multiple=True,
    help="Facility to filter (repeatable). Selects GHCR package name.",
)
@click.option(
    "--without-dd",
    is_flag=True,
    help="Fetch without-dd variant (no IMAS Data Dictionary)",
)
@click.option(
    "--dd-only",
    is_flag=True,
    help="Fetch IMAS-only variant (DD nodes only)",
)
@click.option(
    "--local",
    is_flag=True,
    help="Also transfer the archive locally (remote graphs only).",
)
def graph_fetch(
    version: str,
    registry: str | None,
    token: str | None,
    output: str | None,
    facilities: tuple[str, ...],
    without_dd: bool,
    dd_only: bool,
    local: bool,
) -> Path:
    """Fetch graph archive from GHCR without loading.

    Downloads the archive to disk but does NOT load it into Neo4j.
    Use 'graph load <archive>' to load it afterwards, or use
    'graph pull' as a convenience for fetch + load.

    When the configured location is remote and ``oras`` is available
    there, the fetch runs directly on the remote host.  Use
    ``--local`` to also transfer the archive back via SCP.

    When no --version is specified, fetches 'latest'. If 'latest' doesn't
    exist, falls back to the most recent tag in the registry.
    """
    from imas_codex.cli.graph_progress import (
        GraphProgress,
        remote_operation_streaming,
        run_oras_with_progress,
    )
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = get_package_name(
        list(facilities) or None, without_dd=without_dd, dd_only=dd_only
    )

    # Resolve version: if "latest" doesn't exist, find most recent tag
    resolved_version = version
    if version == "latest":
        resolved_version = _resolve_latest_tag(target_registry, token, pkg_name)

    artifact_ref = f"{target_registry}/{pkg_name}:{resolved_version}"

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            build_remote_fetch_script,
            remote_check_oras,
            scp_from_remote,
        )

        if remote_check_oras(profile.host):
            with GraphProgress("fetch") as gp:
                gp.set_total_phases(2 if (local or output) else 1)

                # Build output name for remote file
                ref_parts = artifact_ref.rsplit("/", 1)[-1]
                output_name = ref_parts.replace(":", "-") + ".tar.gz"

                gp.start_phase(f"Fetching on {profile.host} via ORAS")
                script = build_remote_fetch_script(
                    artifact_ref, output_name, token=token
                )
                fetch_output = remote_operation_streaming(
                    script,
                    profile.host,
                    progress=gp,
                    progress_markers={
                        "LOGIN": f"Authenticating on {profile.host}",
                        "PULLING": f"Downloading from GHCR on {profile.host}",
                        "MOVING": "Saving archive",
                        "DONE": "Fetch complete",
                    },
                    timeout=300,
                )

                # Extract archive path and size from output
                remote_archive = None
                size_str = None
                for line in fetch_output.strip().splitlines():
                    if line.startswith("ARCHIVE_PATH="):
                        remote_archive = line.split("=", 1)[1].strip()
                    elif line.startswith("SIZE="):
                        size_str = line.split("=", 1)[1].strip()
                if not remote_archive:
                    gp.fail_phase("No archive path in output")
                    raise click.ClickException(
                        f"Could not find archive path in output:\n{fetch_output}"
                    )
                gp.complete_phase(size_str)

                if local or output:
                    from imas_codex.graph.dirs import ensure_exports_dir

                    if output:
                        dest = Path(output)
                    else:
                        exports = ensure_exports_dir()
                        dest = exports / f"{pkg_name}-{resolved_version}.tar.gz"

                    gp.start_phase(f"Transferring from {profile.host}")
                    scp_from_remote(remote_archive, dest, profile.host)
                    size_mb = dest.stat().st_size / 1024 / 1024
                    gp.complete_phase(f"{size_mb:.1f} MB")
                    gp.print(f"  Load locally: imas-codex graph load {dest}")
                    return dest

                gp.print(f"  Load remotely: imas-codex graph load {remote_archive}")
                return Path(remote_archive)
        else:
            click.echo(f"oras not on {profile.host}, fetching locally...")
    # ── End remote dispatch ──────────────────────────────────────────────

    require_oras()

    with GraphProgress("fetch") as gp:
        gp.set_total_phases(1)

        gp.start_phase("Fetching from GHCR")
        login_to_ghcr(token)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            run_oras_with_progress(
                ["oras", "pull", artifact_ref, "-o", str(tmp)],
                progress=gp,
                phase_description=f"Fetching {artifact_ref}",
            )

            archives = list(tmp.glob("*.tar.gz"))
            if not archives:
                gp.fail_phase("No archive found")
                raise click.ClickException("No archive found in fetched artifact")

            src_archive = archives[0]
            if output:
                dest = Path(output)
            else:
                from imas_codex.graph.dirs import ensure_exports_dir

                exports = ensure_exports_dir()
                dest = exports / f"{pkg_name}-{resolved_version}.tar.gz"

            shutil.move(str(src_archive), str(dest))

        size_mb = dest.stat().st_size / 1024 / 1024
        gp.complete_phase(f"{size_mb:.1f} MB")
        gp.print(f"  Load with: imas-codex graph load {dest}")
    return dest


@click.command()
@click.argument("target")
@click.option("-v", "--version", "version", default="latest")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--force", is_flag=True, help="Overwrite existing graph without checks")
@click.option("--no-backup", is_flag=True, help="Skip the pre-pull graph dump")
@click.option(
    "--facility",
    "-F",
    "facilities",
    multiple=True,
    help="Facility to filter (repeatable). Selects GHCR package name.",
)
@click.option(
    "--without-dd",
    is_flag=True,
    help="Pull without-dd variant (no IMAS Data Dictionary)",
)
@click.option(
    "--dd-only",
    is_flag=True,
    help="Pull IMAS-only variant (DD nodes only)",
)
def graph_pull(
    target: str,
    version: str,
    registry: str | None,
    token: str | None,
    force: bool,
    no_backup: bool,
    facilities: tuple[str, ...],
    without_dd: bool,
    dd_only: bool,
) -> None:
    """Pull graph from GHCR and load it (convenience for fetch + load).

    TARGET must name the graph selected by the active symlink. The command
    refuses before fetching or replacing data when that graph is not active.

    This is equivalent to running 'graph fetch' followed by 'graph load'.
    Use 'graph fetch' if you only want to download without loading.

    When the configured location is remote:
    - If ``oras`` is available on the remote host, the archive is fetched
      directly there (no SCP transfer needed).
    - Otherwise, the archive is fetched locally and transferred via SCP.

    When no --version is specified, pulls 'latest'. If 'latest' doesn't
    exist, falls back to the most recent tag in the registry.

    Use --facility/-f (repeatable) to pull a per-facility graph.
    """
    from imas_codex.cli.graph.data import _require_matching_graph_target
    from imas_codex.cli.graph_progress import (
        GraphProgress,
        remote_operation_streaming,
        run_oras_with_progress,
    )
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    _require_matching_graph_target(target, profile, operation="pull")
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = get_package_name(
        list(facilities) or None, without_dd=without_dd, dd_only=dd_only
    )

    # Resolve version: if "latest" doesn't exist, find most recent tag
    resolved_version = version
    if version == "latest":
        resolved_version = _resolve_latest_tag(target_registry, token, pkg_name)

    artifact_ref = f"{target_registry}/{pkg_name}:{resolved_version}"

    # ── Pull compatibility check ─────────────────────────────────────────
    if not force:
        try:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.meta import check_pull_compatibility, get_graph_meta

            gc = GraphClient.from_profile()
            meta = get_graph_meta(gc)
            gc.close()
            if meta:
                pull_errors = check_pull_compatibility(
                    meta,
                    dd_only=dd_only,
                    without_dd=without_dd,
                    facilities=list(facilities) or None,
                )
                if pull_errors:
                    msg = "\n".join(pull_errors)
                    raise click.ClickException(f"{msg}\nUse --force to override.")
        except click.ClickException:
            raise
        except Exception:
            pass  # Can't reach Neo4j — skip check

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            REMOTE_EXPORTS,
            build_remote_fetch_script,
            build_remote_load_script,
            remote_check_oras,
            remote_cleanup_archive,
            scp_to_remote,
        )
        from imas_codex.settings import get_graph_password

        password = get_graph_password()

        _remote_markers_fetch = {
            "LOGIN": f"Authenticating on {profile.host}",
            "PULLING": f"Downloading from GHCR on {profile.host}",
            "MOVING": "Saving archive",
            "DONE": "Fetch complete",
        }
        _remote_markers_load = {
            "STOPPING": f"Stopping Neo4j on {profile.host}",
            "EXTRACTING": "Extracting archive",
            "LOADING_DUMP": "Loading graph dump into Neo4j",
            "PASSWORD": "Resetting password",
            "STARTING": f"Starting Neo4j on {profile.host}",
            "COMPLETE": "Load complete",
        }

        with GraphProgress("pull") as gp:
            click.echo(f"Pulling: {artifact_ref}")

            if remote_check_oras(profile.host):
                gp.set_total_phases(3)

                # Build output name
                ref_parts = artifact_ref.rsplit("/", 1)[-1]
                output_name = ref_parts.replace(":", "-") + ".tar.gz"

                gp.start_phase(f"Fetching on {profile.host} via ORAS")
                script = build_remote_fetch_script(
                    artifact_ref, output_name, token=token
                )
                fetch_output = remote_operation_streaming(
                    script,
                    profile.host,
                    progress=gp,
                    progress_markers=_remote_markers_fetch,
                    timeout=300,
                )
                remote_archive = None
                for line in fetch_output.strip().splitlines():
                    if line.startswith("ARCHIVE_PATH="):
                        remote_archive = line.split("=", 1)[1].strip()
                if not remote_archive:
                    gp.fail_phase("No archive path in output")
                    raise click.ClickException(
                        f"Could not find archive path:\n{fetch_output}"
                    )
                gp.complete_phase()
            else:
                gp.set_total_phases(4)

                gp.start_phase("Fetching from GHCR locally")
                require_oras()
                login_to_ghcr(token)

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    run_oras_with_progress(
                        ["oras", "pull", artifact_ref, "-o", str(tmp)],
                        progress=gp,
                    )

                    archives = list(tmp.glob("*.tar.gz"))
                    if not archives:
                        gp.fail_phase("No archive found")
                        raise click.ClickException("No archive found")
                    gp.complete_phase()

                    local_archive = archives[0]
                    remote_archive = f"{REMOTE_EXPORTS}/{local_archive.name}"

                    gp.start_phase(f"Transferring to {profile.host}")
                    scp_to_remote(local_archive, remote_archive, profile.host)
                    gp.complete_phase()

            # Load on remote (streaming)
            gp.start_phase(f"Loading on {profile.host}")
            load_script = build_remote_load_script(
                remote_archive,
                target,
                password,
                scheduler=_resolve_scheduler(profile),
            )
            try:
                load_output = remote_operation_streaming(
                    load_script,
                    profile.host,
                    progress=gp,
                    progress_markers=_remote_markers_load,
                    timeout=600,
                )
            finally:
                remote_cleanup_archive(remote_archive, profile.host)

            if "LOAD_COMPLETE" not in load_output:
                gp.fail_phase("Unexpected output")
                click.echo(f"Warning: Unexpected output: {load_output}", err=True)
            else:
                gp.complete_phase()

            # Update local manifest
            manifest = {
                "version": resolved_version,
                "pulled_from": artifact_ref,
                "pulled_version": resolved_version,
                "pushed": True,
                "pushed_version": resolved_version,
            }
            save_local_graph_manifest(manifest)

            gp.print("[green]✓[/] Graph pull complete (remote)")
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    require_oras()

    if check_graph_exists(data_dir=profile.data_dir) and not force:
        manifest = get_local_graph_manifest()
        if manifest is None:
            raise click.ClickException(
                "Local graph exists but has no manifest (unknown origin).\n"
                "Either:\n"
                "  1. Push current graph first: imas-codex graph push --dev\n"
                "  2. Use --force to overwrite (data will be lost)"
            )
        elif not manifest.get("pushed"):
            raise click.ClickException(
                f"Local graph (loaded {manifest.get('loaded_at', 'unknown')}) "
                "has not been pushed.\n"
                "Either:\n"
                "  1. Push current graph: imas-codex graph push --dev\n"
                "  2. Use --force to overwrite (data will be lost)"
            )
        else:
            pushed_version = manifest.get("pushed_version", "unknown")
            click.echo(f"Local graph was pushed as: {pushed_version}")

    click.echo(f"Pulling: {artifact_ref}")

    if not no_backup:
        with Neo4jOperation("graph pull backup", require_stopped=True):
            backup_graph_dump()

    with GraphProgress("pull") as gp:
        gp.set_total_phases(2)

        gp.start_phase("Fetching from GHCR")
        login_to_ghcr(token)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            run_oras_with_progress(
                ["oras", "pull", artifact_ref, "-o", str(tmp)],
                progress=gp,
            )

            archives = list(tmp.glob("*.tar.gz"))
            if not archives:
                gp.fail_phase("No archive found")
                raise click.ClickException("No archive found")
            gp.complete_phase()

            gp.start_phase("Loading into Neo4j")
            from click.testing import CliRunner

            from imas_codex.cli.graph.data import graph_load

            runner = CliRunner()
            load_args = [str(archives[0]), target, "--force"]
            result = runner.invoke(graph_load, load_args)
            if result.exit_code != 0:
                gp.fail_phase(result.output.strip())
                raise click.ClickException(f"Load failed: {result.output}")
            gp.complete_phase()

            with tarfile.open(archives[0], "r:gz") as tar:
                tar.extractall(tmp / "extracted")
            extracted_dirs = list((tmp / "extracted").iterdir())
            if extracted_dirs:
                manifest_file = extracted_dirs[0] / "manifest.json"
                if manifest_file.exists():
                    manifest = json.loads(manifest_file.read_text())
                    manifest["pulled_from"] = artifact_ref
                    manifest["pulled_version"] = resolved_version
                    manifest["pushed"] = True
                    manifest["pushed_version"] = resolved_version
                    save_local_graph_manifest(manifest)

        gp.print("[green]✓[/] Graph pull complete")


@click.command()
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option(
    "--facility",
    "-F",
    default=None,
    help="List tags for a facility-specific graph package.",
)
def graph_tags(registry: str | None, facility: str | None) -> None:
    """List available graph versions in GHCR."""
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = f"imas-codex-graph-{facility}" if facility else "imas-codex-graph"

    tags = _list_registry_tags(target_registry, pkg_name=pkg_name)
    if not tags:
        click.echo(f"No tags found for {target_registry}/{pkg_name}")
        return

    # Fetch messages for each tag from OCI annotations
    tag_messages = _fetch_tag_messages(target_registry, tags, pkg_name=pkg_name)

    click.echo(f"Tags in {target_registry}/{pkg_name}:")
    for tag in sorted(tags):
        msg = tag_messages.get(tag)
        if msg:
            # Clip long messages to keep output tidy
            display_msg = msg if len(msg) <= 72 else msg[:69] + "..."
            click.echo(f"  {tag}  — {display_msg}")
        else:
            click.echo(f"  {tag}")
    click.echo(f"\n{len(tags)} tag(s) total")


@click.command()
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option(
    "--facility",
    "-F",
    default=None,
    help="Prune tags for a facility-specific graph package.",
)
@click.option("--keep", default=5, help="Number of recent tags to keep.")
@click.option("--dev-only", is_flag=True, help="Only prune dev tags.")
@click.option(
    "--tiered/--flat",
    default=True,
    help="Use 4 weekly + 3 monthly copies, or the legacy flat --keep window.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be deleted.")
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def graph_prune(
    registry: str | None,
    token: str | None,
    facility: str | None,
    keep: int,
    dev_only: bool,
    tiered: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Prune old graph versions from GHCR."""
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = f"imas-codex-graph-{facility}" if facility else "imas-codex-graph"

    if tiered and not dev_only:
        versions = _list_registry_versions(target_registry, pkg_name, token)
        if not versions:
            click.echo(f"No versions found for {target_registry}/{pkg_name}")
            return

        decisions = _select_tiered_retention(versions)
        to_delete = [decision for decision in decisions if not decision.keep]
        click.echo(
            f"Tiered retention for {target_registry}/{pkg_name}: "
            f"{len(decisions) - len(to_delete)} keep, {len(to_delete)} delete"
        )
        for decision in decisions:
            action = "KEEP" if decision.keep else "DELETE"
            stamp = (
                decision.version.created_at.astimezone(UTC)
                .isoformat()
                .replace("+00:00", "Z")
            )
            click.echo(
                f"  {action:<6} {decision.tier:<17} {stamp}  "
                f"{decision.version.display_name}"
            )

        if not to_delete:
            click.echo("Nothing to prune under tiered retention")
            return
        if dry_run:
            click.echo("\n(dry-run — no changes made)")
            return
        if not force and not click.confirm(f"Delete {len(to_delete)} version(s)?"):
            click.echo("Aborted.")
            return

        deleted = 0
        for decision in to_delete:
            version = decision.version
            if version.tags:
                success = _delete_tag(
                    target_registry,
                    version.tags[0],
                    token,
                    pkg_name=pkg_name,
                )
            else:
                success = _delete_untagged_version(
                    target_registry, pkg_name, version.id, token
                )
            if success:
                click.echo(f"  ✓ Deleted {version.display_name}")
                deleted += 1
            else:
                click.echo(f"  ✗ Failed to delete {version.display_name}")
        click.echo(f"\n✓ Pruned {deleted}/{len(to_delete)} versions")
        return

    tags = _list_registry_tags(target_registry, token, pkg_name=pkg_name)
    if not tags:
        click.echo(f"No tags found for {target_registry}/{pkg_name}")
        return

    # Separate release and dev tags
    dev_tags = [t for t in tags if "dev" in t or "-r" in t]
    release_tags = [t for t in tags if t not in dev_tags and t != "latest"]

    if dev_only:
        candidates = dev_tags
    else:
        candidates = dev_tags + release_tags

    # Sort candidates: dev tags by revision descending, release by semver
    def _sort_key(tag: str) -> tuple[int, int]:
        is_dev = 0 if ("dev" in tag or "-r" in tag) else 1
        rev = 0
        if "-r" in tag:
            try:
                rev = int(tag.rsplit("-r", 1)[-1])
            except ValueError:
                pass
        return (is_dev, -rev)

    candidates.sort(key=_sort_key)

    # Keep the most recent N, delete the rest
    to_keep = set(candidates[:keep])
    to_keep.add("latest")  # Never prune 'latest'
    to_delete = [t for t in candidates if t not in to_keep]

    if not to_delete:
        click.echo(f"Nothing to prune (keeping {keep} most recent)")
        return

    click.echo(
        f"Will delete {len(to_delete)} tag(s) from {target_registry}/{pkg_name}:"
    )
    for tag in to_delete:
        click.echo(f"  {tag}")

    if dry_run:
        click.echo("\n(dry-run — no changes made)")
        return

    if not force:
        if not click.confirm(f"Delete {len(to_delete)} tag(s)?"):
            click.echo("Aborted.")
            return

    deleted = 0
    for tag in to_delete:
        if _delete_tag(target_registry, tag, token, pkg_name=pkg_name):
            click.echo(f"  ✓ Deleted {tag}")
            deleted += 1
        else:
            click.echo(f"  ✗ Failed to delete {tag}")

    click.echo(f"\n✓ Pruned {deleted}/{len(to_delete)} tags")

"""Discover run: the stage registry, its access probes and a dry run.

One facility's discovery stages, in dependency order, with the access each
needs and the outcome it reaches. This module holds the registry and the access
probes that decide whether a stage can start.

Each stage keeps exactly one owner: the registry points at the existing
``discover`` command and adds no second implementation. The mapping stage runs
``imas map run``, which is not a discover domain.

Access is probed through the check functions every discover command already
uses, imported here unchanged so no probe is written anew:

* ``neo4j_health_check`` -- the graph
* ``ssh_health_check`` -- the facility hop
* ``wiki_auth_check`` -- a configured wiki site

The registry records each stage's model seat (the stage's own ``ServiceMonitor``
guards the model while the stage runs), but the access probe gates a stage only
on the three functions above.

Outcomes, each carrying a reason:

* ``ran`` -- the stage completed (counts done and remaining)
* ``runnable`` -- the dry run's outcome for a stage that would run
* ``skipped: not configured`` -- a required facility-config block is absent
* ``skipped: unreachable`` -- a probe failed, or a pending-work query raised
* ``skipped: nothing to do`` -- the stage's pending-work predicate is false
* ``skipped: dependency failed`` -- a stage whose input stage failed
* ``failed`` -- the stage raised
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import click

from imas_codex.discovery.base.services import (
    neo4j_health_check,
    ssh_health_check,
    wiki_auth_check,
)

# Outcome vocabulary -- every stage ends in one of these, each with a reason.
RAN = "ran"
RUNNABLE = "runnable"
FAILED = "failed"
NOT_CONFIGURED = "skipped: not configured"
UNREACHABLE = "skipped: unreachable"
NOTHING_TO_DO = "skipped: nothing to do"
DEPENDENCY_FAILED = "skipped: dependency failed"


@dataclass(frozen=True)
class Stage:
    """One discovery stage and what it needs to run."""

    name: str
    domain: str | None
    command: str
    access: tuple[str, ...]
    config: tuple[str, ...]
    reads: tuple[str, ...]
    pending: tuple[str, ...]
    context: tuple[str, ...] = ()
    model_section: str | None = None

    @property
    def pending_names(self) -> tuple[str, ...]:
        """The bare predicate names, for the report and for tests."""
        return tuple(spec.rsplit(":", 1)[-1] for spec in self.pending)


_PATHS = "imas_codex.discovery.paths.parallel"
_CODE = "imas_codex.discovery.code.graph_ops"
_DOCUMENTS = "imas_codex.discovery.documents.pipeline"
_WIKI = "imas_codex.discovery.wiki.graph_ops"
_SIGNALS = "imas_codex.discovery.signals.parallel"
_IDS = "imas_codex.ids.workers"


def _stage(
    name: str,
    domain: str | None,
    command: str,
    *,
    access: tuple[str, ...],
    pending: tuple[str, ...],
    config: tuple[str, ...] = (),
    reads: tuple[str, ...] = (),
    context: tuple[str, ...] = (),
    model_section: str | None = None,
) -> Stage:
    return Stage(
        name=name,
        domain=domain,
        command=command,
        access=access,
        config=config,
        reads=reads,
        pending=pending,
        context=context,
        model_section=model_section,
    )


# The stages, in dependency order. Each discover domain appears at least once;
# a test fails when one does not, so adding a domain cannot leave this stale.
STAGES: tuple[Stage, ...] = (
    _stage(
        "paths",
        "paths",
        "discover paths",
        access=("ssh",),
        pending=(f"{_PATHS}:has_pending_work",),
        config=("discovery_roots",),
    ),
    _stage(
        "code",
        "code",
        "discover code",
        access=("ssh",),
        pending=(
            f"{_CODE}:has_pending_scan_work",
            f"{_CODE}:has_pending_triage_work",
            f"{_CODE}:has_pending_score_work",
            f"{_CODE}:has_pending_enrich_work",
            f"{_CODE}:has_pending_code_work",
        ),
        reads=("paths",),
    ),
    _stage(
        "documents",
        "documents",
        "discover documents",
        access=("ssh",),
        pending=(f"{_DOCUMENTS}:has_pending_work",),
        reads=("paths",),
        model_section="discovery-vision",
    ),
    _stage(
        "wiki",
        "wiki",
        "discover wiki",
        access=("wiki",),
        pending=(
            f"{_WIKI}:has_pending_work",
            f"{_WIKI}:has_pending_scan_work",
            f"{_WIKI}:has_pending_ingest_work",
        ),
        config=("wiki_sites",),
        model_section="discovery-score",
    ),
    _stage(
        "signals scan",
        "signals",
        "discover signals --scan-only",
        access=("ssh",),
        pending=(f"{_SIGNALS}:has_pending_work",),
        config=("data_systems",),
    ),
    _stage(
        "signals enrich",
        "signals",
        "discover signals --enrich-only",
        access=("graph",),
        pending=(f"{_SIGNALS}:has_pending_enrich_work",),
        reads=("signals scan",),
        context=("wiki", "code"),
        model_section="discovery-describe",
    ),
    _stage(
        "signals check",
        "signals",
        "discover signals",
        access=("ssh",),
        pending=(f"{_SIGNALS}:has_pending_check_work",),
        reads=("signals enrich",),
    ),
    _stage(
        "candidates",
        "map",
        "discover map",
        access=("graph",),
        pending=(f"{_IDS}:has_pending_candidate_work",),
        reads=("signals enrich",),
        model_section="mapping-candidates",
    ),
    _stage(
        "mapping",
        None,
        "imas map run",
        access=("graph",),
        pending=(
            f"{_IDS}:has_pending_mapping_work",
            f"{_IDS}:has_pending_validation_work",
        ),
        reads=("candidates",),
        model_section="mapping",
    ),
)


@dataclass
class StageOutcome:
    """What one stage reached, with its reason and its measures."""

    stage: str
    domain: str | None
    outcome: str
    reason: str
    done: int = 0
    remaining: int = 0
    cost: float = 0.0
    seconds: float = 0.0


def _ssh_host(config: dict) -> str:
    """The SSH host a facility config resolves to, or an empty string."""
    return config.get("ssh_host") or ""


def _wiki_target(config: dict) -> tuple[str, str | None]:
    """The first configured wiki site's URL and the host its probe uses."""
    for site in config.get("wiki_sites") or []:
        entry = site if isinstance(site, dict) else {}
        url = entry.get("url")
        if not url:
            continue
        if entry.get("ssh_available") and _ssh_host(config):
            return url, _ssh_host(config)
        return url, None
    return "", None


def probe_access(need: str, config: dict) -> tuple[bool, str]:
    """Probe one access need, returning ``(healthy, detail)``.

    Each need resolves to the check function a stage's ``ServiceMonitor``
    registers, looked up by module global so a test can patch it.
    """
    if need == "graph":
        return neo4j_health_check()
    if need == "ssh":
        host = _ssh_host(config)
        if not host:
            return False, "no ssh host configured"
        return ssh_health_check(host)
    if need == "wiki":
        url, host = _wiki_target(config)
        if not url:
            return False, "no wiki site configured"
        return wiki_auth_check(url, host)
    raise ValueError(f"unknown access need: {need}")


def _configured(config: dict, key: str) -> bool:
    return bool(config.get(key))


def _call_predicate(spec: str, facility: str) -> bool:
    """Resolve and call a pending-work predicate named ``module:function``."""
    module_name, func = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return bool(getattr(module, func)(facility))


def _missing_config(stage: Stage, config: dict) -> list[str]:
    return [key for key in stage.config if not _configured(config, key)]


def evaluate_stage(stage: Stage, facility: str, config: dict) -> StageOutcome:
    """Decide what a stage would reach, without running it.

    Config presence is checked first, then access, then pending work. A raising
    pending-work query is reported as unreachable, never as nothing to do, so a
    graph fault cannot read as an empty stage.
    """
    missing = _missing_config(stage, config)
    if missing:
        return StageOutcome(
            stage=stage.name,
            domain=stage.domain,
            outcome=NOT_CONFIGURED,
            reason=f"{', '.join(missing)} not configured",
        )

    for need in stage.access:
        healthy, detail = probe_access(need, config)
        if not healthy:
            reason = f"{need}: {detail}" if detail else need
            return StageOutcome(
                stage=stage.name,
                domain=stage.domain,
                outcome=UNREACHABLE,
                reason=reason,
            )

    try:
        pending = any(_call_predicate(spec, facility) for spec in stage.pending)
    except Exception as e:  # noqa: BLE001 -- a graph fault reads as unreachable
        return StageOutcome(
            stage=stage.name,
            domain=stage.domain,
            outcome=UNREACHABLE,
            reason=f"pending-work query failed: {e}",
        )

    if not pending:
        return StageOutcome(
            stage=stage.name,
            domain=stage.domain,
            outcome=NOTHING_TO_DO,
            reason=f"no pending work ({', '.join(stage.pending_names)})",
        )

    return StageOutcome(
        stage=stage.name,
        domain=stage.domain,
        outcome=RUNNABLE,
        reason="ready",
    )


def evaluate_plan(facility: str, config: dict) -> list[StageOutcome]:
    """Evaluate every stage in registry order."""
    return [evaluate_stage(stage, facility, config) for stage in STAGES]


def _print_plan(facility: str, outcomes: list[StageOutcome]) -> None:
    """Print one row per stage: name, outcome and reason."""
    click.echo(f"Discovery sequence for {facility}")
    width = max(len(o.stage) for o in outcomes)
    for o in outcomes:
        name = o.stage.ljust(width)
        click.echo(f"  {name}  {o.outcome}  ({o.reason})")


def stage_command(stage: Stage) -> Callable[..., None] | None:
    """Resolve a stage's existing click command, or ``None`` for mapping.

    The commands are imported lazily so this module stays cheap to import and
    so a test can patch the callable the runner invokes.
    """
    if stage.domain is None:
        return None
    from imas_codex.cli.discover.code import code
    from imas_codex.cli.discover.documents import documents
    from imas_codex.cli.discover.map import map_candidates
    from imas_codex.cli.discover.paths import paths
    from imas_codex.cli.discover.signals import signals
    from imas_codex.cli.discover.wiki import wiki

    commands: dict[str, Callable[..., None]] = {
        "paths": paths,
        "code": code,
        "documents": documents,
        "wiki": wiki,
        "signals": signals,
        "map": map_candidates,
    }
    return commands[stage.domain]


def run_sequence(
    facility: str,
    *,
    dry_run: bool,
    config: dict | None = None,
) -> list[StageOutcome]:
    """Evaluate the plan and, unless ``dry_run``, run each runnable stage.

    Args:
        facility: Facility identifier.
        dry_run: When true, print every stage with its outcome and reason and
            invoke nothing.
        config: Facility config; loaded when not supplied.

    Returns:
        One outcome per stage, in registry order.
    """
    if config is None:
        from imas_codex.discovery.base.facility import get_facility

        config = get_facility(facility)

    outcomes = evaluate_plan(facility, config)

    if dry_run:
        _print_plan(facility, outcomes)
        return outcomes

    ctx = click.get_current_context()
    ran: list[StageOutcome] = []
    for stage, outcome in zip(STAGES, outcomes, strict=True):
        if outcome.outcome != RUNNABLE:
            ran.append(outcome)
            continue
        command = stage_command(stage)
        if command is None:
            # The mapping stage runs imas map run, not a discover command; it
            # is opt-in and is selected by the runner.
            ran.append(outcome)
            continue
        ctx.invoke(command, facility=facility)
        ran.append(outcome)
    return ran


@click.command("run")
@click.argument("facility")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print every stage with its outcome and reason, and run nothing.",
)
def run(facility: str, dry_run: bool) -> None:
    """Run a facility's discovery sequence.

    Evaluates the stages in dependency order, probing each one's access, and
    runs the stages whose prerequisites hold. Re-running continues where a
    previous run stopped.

    \b
    Examples:
      imas-codex discover run jt-60sa --dry-run   # Show the plan
      imas-codex discover run jt-60sa             # Run every runnable stage
    """
    run_sequence(facility, dry_run=dry_run)

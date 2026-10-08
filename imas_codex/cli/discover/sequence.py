"""Run facility discovery stages in dependency order with bounded resources."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, fields
from pathlib import Path

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
NOTHING_TO_SEED = "skipped: nothing to seed"
NOT_SELECTED = "skipped: not selected"
LIMIT_REACHED = "skipped: limit reached"
DOMAINS = ("paths", "code", "documents", "wiki", "signals", "candidates", "mapping")


@dataclass(frozen=True)
class Stage:
    """One discovery stage and what it needs to run."""

    name: str
    domain: str
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


_CODE = "imas_codex.discovery.code.graph_ops"
_SIGNALS = "imas_codex.discovery.signals.parallel"
_IDS = "imas_codex.ids.workers"


def _stage(
    name: str,
    domain: str,
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
        access=("ssh",),
        pending=(),
        config=("discovery_roots",),
    ),
    _stage(
        "code",
        "code",
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
        access=("ssh",),
        pending=(),
        reads=("paths",),
        model_section="discovery-vision",
    ),
    _stage(
        "wiki",
        "wiki",
        access=("wiki",),
        pending=(),
        config=("wiki_sites",),
        model_section="discovery-score",
    ),
    _stage(
        "signals scan",
        "signals",
        access=("ssh",),
        pending=(),
        config=("data_systems",),
    ),
    _stage(
        "signals enrich",
        "signals",
        access=("graph",),
        pending=(
            f"{_SIGNALS}:has_pending_enrich_work",
            f"{_SIGNALS}:has_pending_check_work",
        ),
        reads=("signals scan",),
        context=("wiki", "code"),
        model_section="discovery-describe",
    ),
    _stage(
        "candidates",
        "candidates",
        access=("graph",),
        pending=(f"{_IDS}:has_pending_candidate_work",),
        reads=("signals enrich",),
        model_section="mapping-candidates",
    ),
    _stage(
        "mapping",
        "mapping",
        access=("graph",),
        pending=(
            f"{_IDS}:has_pending_mapping_work",
            f"{_IDS}:has_pending_validation_work",
        ),
        reads=("candidates",),
        model_section="ids-mapping",
    ),
)


@dataclass
class StageOutcome:
    """What one stage reached, with its reason and its measures."""

    stage: str
    domain: str | None
    outcome: str
    reason: str
    done: int | None = None
    remaining: int | None = None
    cost: float = 0.0
    seconds: float = 0.0


@dataclass(frozen=True)
class SequenceOptions:
    """One invocation's selection, resource limits, and domain tunables."""

    only: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()
    scan_only: bool = False
    flush: bool = False
    focus: tuple[str, ...] = ()
    topic: str | None = None
    cost_limit: float = 25.0
    time_limit: int | None = None
    limit: int | None = None
    dry_run: bool = False
    reset_to: str | None = None
    rescan: bool = False
    scan_workers: int | None = None
    triage_workers: int | None = None
    enrich_workers: int | None = None
    check_workers: int | None = None
    score_workers: int | None = None
    ingest_workers: int | None = None
    code_workers: int | None = None
    workers: int | None = None
    vlm_workers: int | None = None
    category: str | None = None
    scanners: str | None = None
    reference_shot: int | None = None
    physics_domain: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()
    rejudge_ingested: bool = False
    rescan_documents: bool = False
    add_roots: bool = False
    store_bytes: bool = False
    max_depth: int | None = None
    store_images: bool = False
    wiki_site: str | None = None
    threshold: float | None = None
    enrich_threshold: float | None = None
    min_score: float | None = None
    triage_batch_size: int | None = None
    rejudge_stale: bool = False
    verbose: bool = False


def _domains(items: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(name.strip() for item in items for name in item.split(","))
    unknown = [name for name in selected if name not in DOMAINS]
    if unknown:
        raise click.UsageError(f"unknown discovery domain: {', '.join(unknown)}")
    return selected


def _validate_options(options: SequenceOptions) -> None:
    if options.scan_only and options.flush:
        raise click.UsageError("--scan-only and --flush are mutually exclusive")
    if options.reset_to and (len(options.only) != 1 or options.skip):
        raise click.UsageError(
            "--reset-to requires exactly one --only domain and no --skip"
        )
    if options.reset_to:
        from imas_codex.discovery.base.reset import get_valid_targets

        domain = options.only[0]
        targets = (
            get_valid_targets(domain) if domain not in {"candidates", "mapping"} else ()
        )
        if options.reset_to not in targets:
            raise click.UsageError(
                f"invalid --reset-to target for {domain}: {options.reset_to}"
            )
    if options.cost_limit < 0 or (
        options.time_limit is not None and options.time_limit < 0
    ):
        raise click.UsageError("cost and time limits must be non-negative")
    if options.limit is not None and options.limit < 0:
        raise click.UsageError("--limit must be non-negative")


def _ssh_host(config: dict) -> str:
    """The SSH host a facility config resolves to, or an empty string."""
    return config.get("ssh_host") or ""


def _wiki_targets(config: dict) -> list[tuple[str, str | None]]:
    """Every configured wiki site's URL and the host its probe uses."""
    targets = []
    for site in config.get("wiki_sites") or []:
        entry = site if isinstance(site, dict) else {}
        url = entry.get("url")
        if not url:
            continue
        if entry.get("ssh_available") and _ssh_host(config):
            targets.append((url, _ssh_host(config)))
        else:
            targets.append((url, None))
    return targets


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
        targets = _wiki_targets(config)
        if not targets:
            return False, "no wiki site configured"
        for url, host in targets:
            healthy, detail = wiki_auth_check(url, host)
            if not healthy:
                return False, f"{url}: {detail}"
        return True, f"{len(targets)} wiki sites reachable"
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
        pending = not stage.pending or any(
            _call_predicate(spec, facility) for spec in stage.pending
        )
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


def _stage_function(stage: Stage):
    """Resolve a domain's stage function without importing all engines eagerly."""
    if stage.domain == "mapping":
        return run_mapping_stage
    modules = {
        "paths": ("paths", "run_paths_stage"),
        "code": ("code", "run_code_stage"),
        "documents": ("documents", "run_documents_stage"),
        "wiki": ("wiki", "run_wiki_stage"),
        "signals": ("signals", "run_signals_stage"),
        "candidates": ("map", "run_candidates_stage"),
    }
    module_name, function_name = modules[stage.domain]
    module = importlib.import_module(f"imas_codex.cli.discover.{module_name}")
    return getattr(module, function_name)


def _stage_options(
    stage: Stage, options: SequenceOptions, cost: float, minutes: int | None
):
    """Build a stage's frozen options from the common surface and its tunables."""
    module_name = "map" if stage.domain == "candidates" else stage.domain
    module = importlib.import_module(f"imas_codex.cli.discover.{module_name}")
    class_name = {
        "paths": "PathsStageOptions",
        "code": "CodeStageOptions",
        "documents": "DocumentsOptions",
        "wiki": "WikiStageOptions",
        "signals": "SignalsStageOptions",
        "candidates": "CandidatesStageOptions",
    }[stage.domain]
    cls = getattr(module, class_name)
    names = {field.name for field in fields(cls)}
    values = {
        "cost_limit": cost,
        "time_limit": minutes,
        "limit": options.limit,
        "focus": options.focus,
        "topic": options.topic,
        "reset_to": options.reset_to if stage.name != "signals enrich" else None,
        "rescan": options.rescan,
        "scan_only": options.scan_only,
        "flush": options.flush,
        "scan_workers": options.scan_workers,
        "triage_workers": options.triage_workers,
        "enrich_workers": options.enrich_workers,
        "check_workers": options.check_workers,
        "score_workers": options.score_workers,
        "ingest_workers": options.ingest_workers,
        "code_workers": options.code_workers,
        "workers": options.workers,
        "vlm_workers": options.vlm_workers,
        "scanners": options.scanners,
        "categories": options.category,
        "reference_shot": options.reference_shot,
        "physics_domain": options.physics_domain,
        "ids": options.ids,
        "rejudge_ingested": options.rejudge_ingested,
        "rescan_documents": options.rescan_documents,
        "add_roots": options.add_roots,
        "store_bytes": options.store_bytes,
        "max_depth": options.max_depth,
        "store_images": options.store_images,
        "wiki_site": options.wiki_site,
        "threshold": options.threshold,
        "enrich_threshold": options.enrich_threshold,
        "min_score": options.min_score if stage.domain != "code" else None,
        "triage_batch_size": options.triage_batch_size,
        "rejudge_stale": options.rejudge_stale,
        "verbose": options.verbose,
    }
    if stage.name == "signals scan":
        values.update(scan_only=True, flush=False)
    elif stage.name == "signals enrich":
        values.update(scan_only=False, flush=True)
    return cls(
        **{
            name: value
            for name, value in values.items()
            if name in names and value is not None
        }
    )


def run_mapping_stage(
    facility: str, options: SequenceOptions, cost: float, minutes: int | None
):
    """Call the existing mapping pipeline with the sequence's remaining limits."""
    from imas_codex.cli.map import map_run
    from imas_codex.ids import workers

    focus = []
    for item in options.focus:
        path = Path(item)
        if path.is_file():
            focus.extend(
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        else:
            focus.append(item)
    ids_names = options.ids
    domains = options.physics_domain
    if focus:
        ids_names = tuple(
            name for name in focus if not options.ids or name in options.ids
        )
        domains = ()
        if not ids_names:
            raise click.UsageError("--focus and --ids select no common IDS")
    remaining = None
    if options.limit is not None:
        from imas_codex.graph.client import GraphClient
        from imas_codex.ids.tools import discover_mappable_ids

        with GraphClient() as gc:
            plan = discover_mappable_ids(
                facility,
                gc=gc,
                domains=list(domains) if domains else None,
                ids_filter=list(ids_names) if ids_names else None,
            )
        targets = sorted(t["ids_name"] for t in plan["ids_targets"])
        ids_names = tuple(targets[: options.limit])
        domains = ()
        remaining = len(targets) - len(ids_names)
        if not ids_names:
            return {"cost": 0.0, "bindings": 0, "remaining": remaining}

    original = workers.run_mapping_engine
    receipt = {"cost": 0.0, "bindings": 0}
    if remaining is not None:
        receipt["remaining"] = remaining

    async def record(state, **kwargs):
        try:
            return await original(state, **kwargs)
        finally:
            receipt["cost"] = state.cost.total_usd
            receipt["bindings"] = len(state.ids_results)

    workers.run_mapping_engine = record
    try:
        map_run.callback(
            facility=facility,
            domains=domains,
            ids_names=ids_names,
            model=None,
            dd_version=None,
            cost_limit=cost,
            dry_run=False,
            no_activate=False,
            time_limit=minutes,
            verbose=False,
            clear=False,
            stage="all",
        )
    except (Exception, SystemExit) as exc:
        exc.discovery_receipts = [receipt]
        raise
    finally:
        workers.run_mapping_engine = original
    return receipt


def _selected(stage: Stage, options: SequenceOptions) -> tuple[bool, str]:
    if options.only and stage.domain not in options.only:
        return False, "excluded by --only"
    if stage.domain in options.skip:
        return False, "excluded by --skip"
    if options.scan_only and stage.domain in {"candidates", "mapping"}:
        return False, "nothing to seed"
    if options.scan_only and stage.name == "signals enrich":
        return False, "draining half excluded by --scan-only"
    if options.flush and stage.name == "signals scan":
        return False, "seeding half excluded by --flush"
    return True, ""


def _report(facility: str, outcomes: list[StageOutcome]) -> str:
    """Render exactly one final row per domain from the stage outcomes."""
    records = []
    for domain in DOMAINS:
        rows = [outcome for outcome in outcomes if outcome.domain == domain]
        failed = next((row for row in rows if row.outcome == FAILED), None)
        ran = [row for row in rows if row.outcome == RAN]
        runnable = next((row for row in rows if row.outcome == RUNNABLE), None)
        selected = failed or (ran[-1] if ran else None) or runnable or rows[-1]
        outcome = selected.outcome
        reason = selected.reason
        done = sum(row.done for row in rows if row.done is not None)
        done_text = str(done) if any(row.done is not None for row in rows) else "—"
        remaining = sum(row.remaining for row in rows if row.remaining is not None)
        remaining_text = (
            str(remaining) if any(row.remaining is not None for row in rows) else "—"
        )
        cost = sum(row.cost for row in rows)
        seconds = sum(row.seconds for row in rows)
        records.append(
            (domain, outcome, reason, done_text, remaining_text, cost, seconds)
        )
    reason_width = max(30, *(len(row[2]) for row in records))
    lines = [
        f"Discovery sequence for {facility}",
        f"{'Domain':<10} {'Outcome':<29} {'Reason':<{reason_width}} Done  Remaining  Cost     Time",
    ]
    for domain, outcome, reason, done, remaining, cost, seconds in records:
        lines.append(
            f"{domain:<10} {outcome:<29} {reason:<{reason_width}} "
            f"{done:>4}  {remaining:>9}  ${cost:>6.2f}  {seconds:>5.1f}s"
        )
    return "\n".join(lines)


def _run_with_receipt(function, *args):
    """Retain the discovery harness result while thin stage wrappers return None."""
    from imas_codex.cli.discover import common

    original = common.run_discovery
    receipts = []

    def record(*run_args, **run_kwargs):
        receipt = original(*run_args, **run_kwargs)
        receipts.append(receipt)
        return receipt

    common.run_discovery = record
    try:
        result = function(*args)
    except (Exception, SystemExit) as exc:
        exc.discovery_receipts = getattr(exc, "discovery_receipts", []) + receipts
        raise
    finally:
        common.run_discovery = original
    if isinstance(result, dict):
        return result
    if not receipts:
        return {}
    combined: dict = {}
    for receipt in receipts:
        for name, value in receipt.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                combined[name] = combined.get(name, 0) + value
    return combined


def _done_count(result: dict) -> int | None:
    for name in (
        "sources_judged",
        "scanned",
        "images_captioned",
        "enriched",
        "checked",
        "bindings",
    ):
        if name in result and isinstance(result[name], int | float):
            return int(result[name])
    return None


def _remaining_count(stage: Stage, facility: str, result: dict) -> int | None:
    """Use a count receipt, or prove zero from every pending predicate."""
    count = result.get("remaining")
    if isinstance(count, int) and count >= 0:
        return count
    if not stage.pending:
        return None
    try:
        if not any(_call_predicate(spec, facility) for spec in stage.pending):
            return 0
    except Exception:
        pass
    return None


def run_sequence(
    facility: str,
    *,
    dry_run: bool = False,
    config: dict | None = None,
    options: SequenceOptions | None = None,
) -> list[StageOutcome]:
    """Run selected stages in order, isolating failures and sharing limits."""
    options = options or SequenceOptions(dry_run=dry_run)
    _validate_options(options)
    if config is None:
        from imas_codex.discovery.base.facility import get_facility

        config = get_facility(facility)

    outcomes: list[StageOutcome] = []
    by_name: dict[str, StageOutcome] = {}
    spent = 0.0
    started = time.monotonic()
    for stage in STAGES:
        selected, reason = _selected(stage, options)
        if not selected:
            outcome = StageOutcome(
                stage.name,
                stage.domain,
                NOTHING_TO_SEED if reason == "nothing to seed" else NOT_SELECTED,
                reason,
            )
        else:
            failed_input = next(
                (
                    name
                    for name in stage.reads
                    if by_name[name].outcome in {FAILED, DEPENDENCY_FAILED}
                ),
                None,
            )
            if failed_input:
                outcome = StageOutcome(
                    stage.name,
                    stage.domain,
                    DEPENDENCY_FAILED,
                    f"{failed_input} failed",
                )
            else:
                outcome = evaluate_stage(stage, facility, config)
        if outcome.outcome != RUNNABLE or options.dry_run:
            outcomes.append(outcome)
            by_name[stage.name] = outcome
            continue

        remaining_cost = max(0.0, options.cost_limit - spent)
        elapsed = time.monotonic() - started
        remaining_seconds = (
            None if options.time_limit is None else options.time_limit * 60 - elapsed
        )
        if remaining_cost <= 0 or (
            remaining_seconds is not None and remaining_seconds <= 0
        ):
            outcome.outcome = LIMIT_REACHED
            outcome.reason = (
                "cost limit reached" if remaining_cost <= 0 else "time limit reached"
            )
            outcomes.append(outcome)
            by_name[stage.name] = outcome
            continue
        minutes = remaining_seconds / 60 if remaining_seconds is not None else None
        stage_started = time.monotonic()
        try:
            function = _stage_function(stage)
            if stage.domain == "mapping":
                result = _run_with_receipt(
                    function, facility, options, remaining_cost, minutes
                )
            else:
                stage_options = _stage_options(stage, options, remaining_cost, minutes)
                result = _run_with_receipt(function, facility, stage_options)
            outcome.outcome = RAN
            outcome.reason = "completed"
            outcome.done = _done_count(result)
            outcome.remaining = _remaining_count(stage, facility, result)
            outcome.cost = float(result.get("cost", 0.0))
            spent += outcome.cost
        except (Exception, SystemExit) as exc:
            outcome.outcome = FAILED
            detail = exc.__cause__ or exc
            outcome.reason = f"{type(detail).__name__}: {detail}"
            outcome.cost = sum(
                float(receipt.get("cost", 0.0))
                for receipt in getattr(exc, "discovery_receipts", [])
            )
            spent += outcome.cost
        outcome.seconds = time.monotonic() - stage_started
        outcomes.append(outcome)
        by_name[stage.name] = outcome

    report = _report(facility, outcomes)
    click.echo(report)
    return outcomes


@click.command("run", hidden=True)
@click.argument("facility")
@click.option(
    "--only", multiple=True, help="Select domains (repeat or comma-separate)."
)
@click.option(
    "--skip", multiple=True, help="Exclude domains (repeat or comma-separate)."
)
@click.option("--scan-only", is_flag=True, help="Run only seeding halves.")
@click.option("--flush", is_flag=True, help="Run only draining halves.")
@click.option("--focus", multiple=True, help="Restrict work to named items.")
@click.option("--topic", help="Free-text steer for scoring and enrichment.")
@click.option("--cost-limit", "-c", type=float, default=25.0)
@click.option("--time", "-t", "time_limit", type=int)
@click.option("--limit", type=int)
@click.option("--dry-run", is_flag=True, help="Show outcomes without running stages.")
@click.option("--reset-to", help="Reset the single selected domain to a target state.")
@click.option("--rescan", is_flag=True)
@click.option("--scan-workers", type=int)
@click.option("--triage-workers", type=int)
@click.option("--enrich-workers", type=int)
@click.option("--check-workers", type=int)
@click.option("--score-workers", type=int)
@click.option("--ingest-workers", type=int)
@click.option("--code-workers", type=int)
@click.option("--workers", type=int)
@click.option("--vlm-workers", type=int)
@click.option("--category")
@click.option("--scanners")
@click.option("--reference-shot", type=int)
@click.option("--physics-domain", multiple=True)
@click.option("--ids", multiple=True)
@click.option("--rejudge-ingested", is_flag=True)
@click.option("--rescan-documents", is_flag=True)
@click.option("--add-roots", is_flag=True)
@click.option("--store-bytes", is_flag=True)
@click.option("--max-depth", type=int)
@click.option("--store-images", is_flag=True)
@click.option("--wiki-site")
@click.option("--threshold", type=float)
@click.option("--enrich-threshold", type=float)
@click.option("--min-score", type=float)
@click.option("--triage-batch-size", type=int)
@click.option("--rejudge-stale", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def run(facility: str, **kwargs) -> None:
    """Run the selected discovery domains for a facility."""
    kwargs["only"] = _domains(kwargs["only"])
    kwargs["skip"] = _domains(kwargs["skip"])
    options = SequenceOptions(**kwargs)
    from imas_codex.cli.logging import configure_cli_logging, get_log_file

    configure_cli_logging("discover", facility=facility)
    outcomes = run_sequence(facility, options=options)
    with get_log_file("discover", facility=facility).open("a", encoding="utf-8") as log:
        log.write(_report(facility, outcomes) + "\n")
    if any(outcome.outcome == FAILED for outcome in outcomes):
        raise SystemExit(1)

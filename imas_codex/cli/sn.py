"""Standard name generation commands."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import click
from rich.console import Console

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.standard_names.defaults import (
    DEFAULT_ESCALATION_MODEL,
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
    REVIEW_DOCS_BACKLOG_CAP,
    REVIEW_NAME_BACKLOG_CAP,
)

logger = logging.getLogger(__name__)
console = Console()


def _suppress_console_handlers() -> None:
    """Remove or silence all console StreamHandlers across all loggers.

    LiteLLM registers loggers with CamelCase names (``LiteLLM``,
    ``LiteLLM Proxy``, ``LiteLLM Router``) that leak DEBUG messages
    to stderr.  A targeted name list can't keep up with third-party
    loggers, so this sweeps **every** registered logger.
    """
    for name in list(logging.Logger.manager.loggerDict):
        lg = logging.getLogger(name)
        for handler in list(lg.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                lg.removeHandler(handler)

    # Root logger — remove any StreamHandler that isn't a FileHandler
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            root.removeHandler(handler)


class _SpaceSplitMultiple(click.Option):
    """Click option that splits quoted space-separated values.

    ``--focus "a b c" --focus d`` produces ``('a', 'b', 'c', 'd')``.
    Each flag invocation may contain whitespace-separated tokens that are
    flattened into the final tuple.
    """

    def type_cast_value(self, ctx: click.Context, value: Any) -> tuple[str, ...]:
        if not value:
            return ()
        flat: list[str] = []
        for v in value:
            flat.extend(v.split())
        return tuple(flat)


def _check_local_llm() -> tuple[bool, str]:
    """Probe the local GPU model server configured for SN compose.

    Sends an **authenticated** ``GET {api-base}/models`` — vLLM launched
    with ``--api-key`` returns 401 on every unauthenticated route, so the
    probe must carry the key from ``api-key-env`` or a healthy server is
    misreported as unreachable.

    Returns ``(healthy, detail)``:
      - healthy   → detail is the served model's short name (e.g.
        ``"deepseek-v4-flash"``)
      - unhealthy → detail is a concise reason: ``"down"`` (connection
        refused), ``"timeout"``, ``"unreachable"``, ``"auth error"``
        (server up, key rejected), ``"key missing"`` (server up, no key
        in the environment), or ``"HTTP <code>"``.
    """
    import os
    import urllib.error
    import urllib.request

    from imas_codex.settings import get_model_config

    cfg = get_model_config("sn-compose")
    api_base = cfg.get("api_base")
    if not api_base:
        return False, "not configured"

    model_label = (cfg.get("model") or "").rsplit("/", 1)[-1] or "local"
    key_env = cfg.get("api_key_env") or ""
    key = os.getenv(key_env, "") if key_env else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    url = f"{api_base.rstrip('/')}/models"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, model_label
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        # The server responded — it is up.  4xx here is a key problem,
        # not an availability problem.
        if exc.code in (401, 403):
            return False, "auth error" if key else "key missing"
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        reason = str(exc.reason).lower()
        if "connection refused" in reason or "actively refused" in reason:
            return False, "down"
        if "timed out" in reason or "timeout" in reason:
            return False, "timeout"
        return False, "unreachable"
    except Exception:
        return False, "unreachable"


def _check_openrouter() -> tuple[bool, str]:
    """Probe OpenRouter key validity and remaining credit.

    The SN pipeline routes docs, refine, and the review quorum through
    OpenRouter regardless of where compose runs, so this check is always
    registered.  Uses the free ``/auth/key`` endpoint (no completion
    request, no cost).

    Returns ``(healthy, detail)``:
      - healthy   → ``"ok"``, with remaining credit appended when the
        account reports a limit (e.g. ``"ok $123"``)
      - unhealthy → ``"no key"``, ``"no credit"``, ``"rate limited"``,
        ``"auth error"``, ``"unreachable"``, or ``"HTTP <code>"``.
    """
    import json
    import os
    import urllib.error
    import urllib.request

    key = os.getenv("OPENROUTER_API_KEY_IMAS_CODEX")
    if not key:
        return False, "no key"
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            data = json.loads(resp.read()).get("data") or {}
            remaining = data.get("limit_remaining")
            if isinstance(remaining, int | float):
                return True, f"ok ${remaining:,.0f}"
            return True, "ok"
    except urllib.error.HTTPError as exc:
        if exc.code == 402:
            return False, "no credit"
        if exc.code == 429:
            return False, "rate limited"
        if exc.code == 401:
            return False, "auth error"
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError:
        return False, "unreachable"
    except Exception:
        return False, "unreachable"


def _require_embed_ready(command_label: str) -> None:
    """Raise a user-facing error when the embedding service is unavailable."""
    from imas_codex.discovery.base.services import embed_health_check

    healthy, detail = embed_health_check()
    if healthy:
        return
    raise click.ClickException(
        f"Embedding server is required for `{command_label}` but is unavailable"
        f" ({detail or 'unknown error'}). "
        "Run `uv run imas-codex embed status` and, if needed, "
        "`uv run imas-codex embed start`."
    )


_PHYSICS_DOMAIN_CHOICE = click.Choice(
    [d.value for d in PhysicsDomain], case_sensitive=False
)


@click.group()
def sn() -> None:
    """Standard name generation and management.

    \b
    Pipeline:
      sn run --source dd [--domain NAME ...]
      sn run --source signals --facility NAME
      sn status

    \b
    Catalog workflow:
      sn release --export-only            # graph → staging YAML
      sn preview                          # auto-export + local MkDocs
      sn release -m "msg"                 # auto-export + tag RC + push
      sn release --final -m "msg"         # finalize RC → stable
      sn release status                   # show ISNC state and tags
      sn import                           # ISNC YAML → graph

    \b
    Housekeeping:
      sn clear | sn prune | sn bench
    """
    pass


def _split_whitespace(
    ctx: click.Context, param: click.Parameter, value: tuple[str, ...]
) -> tuple[str, ...]:
    """Split each value on whitespace so ``--domain "a b"`` works."""
    out: list[str] = []
    for v in value or ():
        out.extend(v.split())
    return tuple(out)


def _compute_pool_progress(
    gc: object,
    domains: list[str] | None,
    rotation_cap: int,
    min_score: float,
    scope_run_id: str | None = None,
) -> dict[str, dict[str, int]]:
    """Return per-pool ``{"pending": int, "done": int}`` from one query.

    ``pending`` mirrors the ``claim_*_batch`` predicates (work the loop
    can still claim).  ``done`` counts work already persisted in the
    graph — across *all* runs — so the display can show true pipeline
    position instead of restarting at 0 % each session:

    - ``generate_name``: sources processed (composed/attached/vocab_gap/failed)
    - ``review_name``:   names with a reviewer score
    - ``refine_name``:   refined name nodes (``chain_length > 0``)
    - ``generate_docs``: names whose docs left ``pending``
    - ``review_docs``:   names with a docs reviewer score
    - ``refine_docs``:   total docs refine operations (Σ ``docs_chain_length``)
    - ``enrich_parents``: derived parents whose placeholder description has been
      replaced (``parent_enriched_at`` set)

    Keys: ``generate_name``, ``review_name``, ``refine_name``,
    ``generate_docs``, ``review_docs``, ``refine_docs``, ``enrich_parents``.

    Parameters
    ----------
    gc:
        An open :class:`~imas_codex.graph.client.GraphClient` session.
    domains:
        When non-empty, restrict counts to these physics domains.
        ``physics_domain`` on ``StandardName`` is a *string*, so the
        filter uses ``sn.physics_domain IN $domains``.
    rotation_cap:
        Maximum chain depth — mirrors ``claim_refine_name_batch``.
    min_score:
        Reviewer threshold — mirrors ``claim_refine_name_batch``.
    scope_run_id:
        When set (``--focus`` mode), restrict counts to sources/names
        with ``run_id = $scope_run_id``.  Without this filter the
        pending count can see stale sources from previous runs that
        the scoped claim query will never pick up, causing the exit
        watchdog to spin forever.
    """
    domain_filter_sn = "AND sn.physics_domain IN $domains" if domains else ""
    domain_filter_src = "AND s.physics_domain IN $domains" if domains else ""
    scope_filter_src = "AND s.run_id = $scope_run_id" if scope_run_id else ""
    scope_filter_sn = "AND sn.run_id = $scope_run_id" if scope_run_id else ""

    query = f"""
    CALL {{
      MATCH (s:StandardNameSource {{status: 'extracted'}})
      WHERE NOT (s)-[:PRODUCED_NAME]->(:StandardName)
        {domain_filter_src}
        {scope_filter_src}
      RETURN count(s) AS generate_name
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.name_stage = 'drafted'
        AND NOT (sn.name_stage IN ['superseded', 'exhausted'])
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS review_name
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.name_stage = 'reviewed'
        AND sn.reviewer_score_name IS NOT NULL
        AND sn.reviewer_score_name < $min_score
        AND coalesce(sn.chain_length, 0) < $rotation_cap
        AND NOT (sn.name_stage IN ['superseded', 'exhausted'])
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS refine_name
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.name_stage = 'accepted'
        AND sn.docs_stage = 'pending'
        AND NOT (sn.name_stage IN ['superseded', 'exhausted'])
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS generate_docs
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.docs_stage = 'drafted'
        AND NOT (sn.name_stage IN ['superseded', 'exhausted'])
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS review_docs
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.docs_stage = 'reviewed'
        AND sn.reviewer_score_docs IS NOT NULL
        AND sn.reviewer_score_docs < $min_score
        AND coalesce(sn.docs_chain_length, 0) < $rotation_cap
        AND NOT (sn.name_stage IN ['superseded', 'exhausted'])
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS refine_docs
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.origin = 'derived'
        AND sn.description = $parent_desc_placeholder
        AND EXISTS {{ MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
            WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted'] }}
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS enrich_parents
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.origin = 'derived'
        AND sn.parent_enriched_at IS NOT NULL
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS enrich_parents_done
    }}
    CALL {{
      MATCH (s:StandardNameSource)
      WHERE s.status IN ['composed', 'attached', 'vocab_gap', 'failed']
        {domain_filter_src}
        {scope_filter_src}
      RETURN count(s) AS generate_name_done
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.reviewer_score_name IS NOT NULL
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS review_name_done
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE coalesce(sn.chain_length, 0) > 0
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS refine_name_done
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.docs_stage IS NOT NULL AND sn.docs_stage <> 'pending'
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS generate_docs_done
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.reviewer_score_docs IS NOT NULL
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN count(sn) AS review_docs_done
    }}
    CALL {{
      MATCH (sn:StandardName)
      WHERE coalesce(sn.docs_chain_length, 0) > 0
        {domain_filter_sn}
        {scope_filter_sn}
      RETURN coalesce(sum(sn.docs_chain_length), 0) AS refine_docs_done
    }}
    RETURN generate_name, review_name, refine_name,
           generate_docs, review_docs, refine_docs, enrich_parents,
           generate_name_done, review_name_done, refine_name_done,
           generate_docs_done, review_docs_done, refine_docs_done,
           enrich_parents_done
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    params: dict[str, object] = {
        "rotation_cap": rotation_cap,
        "min_score": min_score,
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    }
    if domains:
        params["domains"] = list(domains)
    if scope_run_id:
        params["scope_run_id"] = scope_run_id
    pools = (
        "generate_name",
        "review_name",
        "refine_name",
        "generate_docs",
        "review_docs",
        "refine_docs",
        "enrich_parents",
    )
    rows = list(gc.query(query, **params))  # type: ignore[attr-defined]
    if not rows:
        return {k: {"pending": 0, "done": 0} for k in pools}
    r = rows[0]
    return {
        k: {"pending": int(r.get(k, 0)), "done": int(r.get(f"{k}_done", 0) or 0)}
        for k in pools
    }


def _compute_pool_pending(
    gc: object,
    domains: list[str] | None,
    rotation_cap: int,
    min_score: float,
    scope_run_id: str | None = None,
) -> dict[str, int]:
    """Per-pool pending counts mirroring ``claim_*_batch`` predicates.

    Thin projection of :func:`_compute_pool_progress` — used by the run
    loop's exit watchdog and pool weighting, which only need the
    claimable backlog.
    """
    progress = _compute_pool_progress(
        gc,
        domains=domains,
        rotation_cap=rotation_cap,
        min_score=min_score,
        scope_run_id=scope_run_id,
    )
    return {k: v["pending"] for k, v in progress.items()}


def _run_sn_cmd(
    *,
    cost_limit: float,
    time_limit: float | None = None,
    per_domain_limit: int | None,
    dry_run: bool,
    quiet: bool,
    domains: tuple[str, ...] = (),
    verbose: bool = False,
    min_score: float | None = None,
    rotation_cap: int | None = None,
    escalation_model: str | None = None,
    review_name_backlog_cap: int | None = None,
    review_docs_backlog_cap: int | None = None,
    skip_generate: bool = False,
    skip_review: bool = False,
    names_only: bool = False,
    docs_only: bool = False,
    flush: bool = False,
    source: str = "dd",
    override_edits: list[str] | None = None,
    only: str | None = None,
    max_sources: int | None = None,
    scope_run_id: str | None = None,
) -> None:
    """Execute the pool-based SN orchestrator with Rich progress display.

    Uses the ``run_discovery()`` harness for 3-press shutdown,
    periodic ticker, graph-refresh, and service monitoring.
    Falls back to plain-mode logging when Rich is unavailable.
    """
    import uuid as _uuid

    from rich.console import Console
    from rich.table import Table

    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.standard_names.loop import (
        run_sn_pools,
        summary_table,
    )

    run_id = str(_uuid.uuid4())
    use_rich = not quiet and not dry_run and use_rich_output()

    # Pre-suppress console output BEFORE any heavy imports/inits so
    # rich-mode startup is silent.  Sweeps ALL registered loggers to
    # catch CamelCase LiteLLM names and any other third-party leaks.
    if use_rich:
        _suppress_console_handlers()

    # Build Rich display or fall back to plain logging
    display = None
    cli_console: Console | None = None
    _on_event: Callable[[dict[str, Any]], None] | None = None

    # Shared progress callable for both Rich and headless modes.  One
    # Cypher query returns pending (mirroring claim predicates) and done
    # (cross-run graph baseline) per pool; a 1 s cache serves both the
    # display ticker and the loop's exit watchdog.
    from imas_codex.graph.client import GraphClient as _GC

    _domains_list: list[str] | None = list(domains) if domains else None
    _rc = rotation_cap if rotation_cap is not None else 3
    _ms = min_score if min_score is not None else DEFAULT_MIN_SCORE
    _scope_run_id = scope_run_id
    _POOLS = (
        "generate_name",
        "review_name",
        "refine_name",
        "generate_docs",
        "review_docs",
        "refine_docs",
        "enrich_parents",
    )

    _progress_cache: dict[str, tuple[float, dict[str, dict[str, int]]]] = {
        "v": (0.0, {})
    }

    def _pool_progress_fn() -> dict[str, dict[str, int]]:
        """Cached per-pool {pending, done} counts (1 s TTL)."""
        import time as _t

        now = _t.monotonic()
        ts, val = _progress_cache["v"]
        if not val or (now - ts) > 1.0:
            try:
                with _GC() as gc:
                    val = _compute_pool_progress(
                        gc,
                        domains=_domains_list,
                        rotation_cap=_rc,
                        min_score=_ms,
                        scope_run_id=_scope_run_id,
                    )
            except Exception:
                val = {k: {"pending": 0, "done": 0} for k in _POOLS}
            _progress_cache["v"] = (now, val)
        return val

    def _pool_pending_fn() -> dict[str, int]:
        return {k: v["pending"] for k, v in _pool_progress_fn().items()}

    if use_rich:
        cli_console = Console()
        setup_logging("sn", "sn-compose", use_rich=True, verbose=verbose)

        # Cost gauge: graph-backed when available, else returns 0.0
        try:
            from imas_codex.standard_names.graph_ops import (
                aggregate_spend_for_run,
            )

            def _cost_fn() -> float:
                return aggregate_spend_for_run(run_id)

        except ImportError:

            def _cost_fn() -> float:
                return 0.0

        from imas_codex.standard_names.display import SN6PoolDisplay

        display = SN6PoolDisplay(
            cost_limit=cost_limit,
            console=cli_console,
            progress_fn=_pool_progress_fn,
            accumulated_cost_fn=_cost_fn,
            flush=flush,
        )
        _on_event = display.on_event
    else:
        setup_logging("sn", "sn-compose", use_rich=False, verbose=verbose)
        cli_console = Console(quiet=quiet)
        if not quiet:
            cli_console.print(
                f"[bold]SN pipeline[/bold] "
                f"(budget=${cost_limit:.2f}"
                f"{f', time={time_limit:.0f}m' if time_limit else ''}"
                f"{f', min_score={min_score}' if min_score is not None else ''}"
                f"{', dry-run' if dry_run else ''})"
            )

    if not dry_run:
        _require_embed_ready("sn run")

    # Build harness config — the SERVERS row mirrors the endpoints this
    # run actually uses: graph, the local embedding server, the local GPU
    # compose endpoint (when [sn-compose].api-base is set), and OpenRouter
    # (docs / refine / review quorum).  The LiteLLM proxy check is skipped —
    # SN routes around the proxy.
    _llm_checks: list[tuple[str, Any, dict[str, Any]]] = []
    if use_rich and not dry_run:
        from imas_codex.settings import get_model_config as _gmc

        if _gmc("sn-compose").get("api_base"):
            _llm_checks.append(
                ("gpu", _check_local_llm, {"poll_interval": 30.0, "critical": False})
            )
        _llm_checks.append(
            (
                "openrouter",
                _check_openrouter,
                {"poll_interval": 60.0, "critical": False},
            )
        )

    disc_config = DiscoveryConfig(
        domain="standard-names",
        facility="sn",
        facility_config={},  # SN has no facility YAML
        display=display,
        check_graph=not dry_run,
        check_embed=not dry_run,
        check_ssh=False,
        check_auth=False,
        check_model=False,  # proxy check is misleading — SN uses direct bypass
        verbose=verbose,
        suppress_loggers=[
            "imas_codex.standard_names",
            "imas_codex.graph",
            "imas_codex.embeddings",
            "imas_codex.discovery",
            "imas_codex.llm",
            "imas_codex.remote",
            "imas_codex.cli",
            "litellm",
            "httpx",
            "httpcore",
            "openai",
            "urllib3",
            "neo4j",
        ]
        if use_rich
        else [],
        extra_service_checks=_llm_checks,
    )

    async def async_main(stop_event, service_monitor):
        summary = await run_sn_pools(
            cost_limit=cost_limit,
            time_limit_s=time_limit * 60 if time_limit else None,
            min_score=min_score,
            rotation_cap=rotation_cap,
            escalation_model=escalation_model,
            review_name_backlog_cap=review_name_backlog_cap,
            review_docs_backlog_cap=review_docs_backlog_cap,
            source=source,
            domains=domains,
            max_sources=max_sources,
            stop_event=stop_event,
            pending_fn=_pool_pending_fn,
            on_event=_on_event,
            display=display,
            scope_run_id=scope_run_id,
            names_only=names_only,
            docs_only=docs_only,
            flush=flush,
            skip_review=skip_review,
        )
        return {"summary": summary}

    result = run_discovery(disc_config, async_main)
    summary = result.get("summary")
    if summary is None:
        return

    row = summary_table(summary)

    if quiet:
        if row.get("stop_reason") == "provider_budget_exhausted":
            raise SystemExit(4)
        return

    # Print summary table (in both rich and plain mode, after display exits)
    out_console = cli_console or Console()
    table = Table(title=f"Run {row['run_id'][:8]}…")
    table.add_column("field", style="cyan")
    table.add_column("value", style="white")
    for key in (
        "stop_reason",
        "cost_spent",
        "cost_limit",
        "names_composed",
        "names_enriched",
        "names_reviewed",
        "names_regenerated",
        "elapsed_s",
    ):
        if key in row:
            table.add_row(key, str(row[key]))
    out_console.print(table)

    # Surface provider-exhaustion as an actionable error and non-zero exit:
    # this is an external (account-level) issue, not a code bug, but the
    # operator needs to know the run did not complete its intended work.
    if row.get("stop_reason") == "provider_budget_exhausted":
        out_console.print(
            "\n[red bold]Upstream LLM provider exhausted.[/red bold]\n"
            "  The configured OpenRouter account has insufficient credits "
            "for the docs / review models.\n"
            "  Top up the account or raise the spending cap, then re-run "
            "[bold]sn run[/bold] to resume the queued work.\n"
            "  Existing partial progress is preserved in the graph; the "
            "next run picks up where this one stopped."
        )
        raise SystemExit(4)


def _execute_rename_cascade(
    *,
    rename_spec: str,
    dry_run: bool,
    override_edits: list[str],
    include_accepted: bool,
) -> None:
    """Run the parent-rename cascade and exit.

    Parses ``OLD:NEW``, invokes
    :func:`imas_codex.standard_names.cascade.rename_cascade`, and reports
    the plan (or applied changes) to the console.  Raises ``SystemExit``
    on usage / cascade error.
    """
    if ":" not in rename_spec:
        raise click.UsageError(
            f"--rename expects 'OLD:NEW' (got --rename {rename_spec})"
        )
    old_name, _, new_name = rename_spec.partition(":")
    old_name = old_name.strip()
    new_name = new_name.strip()
    if not old_name or not new_name:
        raise click.UsageError("--rename: both OLD and NEW must be non-empty")

    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.cascade import rename_cascade

    # The override_edits flag in sn run is per-name; for the cascade
    # operation we treat it as a boolean (override any catalog_edit).
    override_flag = bool(override_edits)

    with GraphClient() as gc:
        result = rename_cascade(
            gc,
            old_name,
            new_name,
            dry_run=dry_run,
            override_edits=override_flag,
            include_accepted=include_accepted,
        )

    mode = (
        "[bold yellow]DRY RUN[/bold yellow]"
        if result.dry_run
        else "[bold green]APPLIED[/bold green]"
    )
    console.print(
        f"\n{mode} rename cascade: [cyan]{old_name}[/cyan] → [cyan]{new_name}[/cyan]"
    )
    console.print(f"  descendants discovered: {result.total_descendants}")
    console.print(f"  planned renames:        {len(result.renamed)}")
    console.print(f"  skipped (independent):  {len(result.skipped)}")
    console.print(f"  conflicts:              {len(result.conflicts)}")

    if result.conflicts:
        console.print("\n[bold red]Conflicts (cascade aborted):[/bold red]")
        for c in result.conflicts:
            console.print(f"  - {c}")
        raise SystemExit(2)

    if result.renamed:
        console.print("\n[bold]Renames:[/bold]")
        for r in result.renamed[:50]:
            console.print(f"  [dim]{r['from']}[/dim] → [green]{r['to']}[/green]")
        if len(result.renamed) > 50:
            console.print(f"  ... and {len(result.renamed) - 50} more")

    if result.skipped:
        console.print("\n[bold]Skipped (independent identity):[/bold]")
        for s in result.skipped[:20]:
            console.print(f"  [dim]{s['name']}[/dim] — {s['reason']}")
        if len(result.skipped) > 20:
            console.print(f"  ... and {len(result.skipped) - 20} more")


def _check_pipeline_clear_gate() -> None:
    """Check whether the pipeline version has changed since the last SNRun.

    Queries the graph for the most recent ``SNRun`` node that has a
    ``pipeline_hash`` set.  If the stored composite hash differs from the
    current one **and** there are ``StandardName`` nodes generated after
    that run, print a warning banner and raise ``SystemExit(1)``.

    Best-effort: if the graph is unreachable or the import fails, the
    function returns silently so it never blocks a legitimate first run.
    """
    try:
        import json as _json

        from rich.console import Console as _Console

        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.pipeline_version import (
            compute_pipeline_hash,
            diff_pipeline_hashes,
        )
    except ImportError:
        return  # bootstrap — skip gate

    try:
        current = compute_pipeline_hash()
        current_composite = current["_composite"]

        with GraphClient() as gc:
            # Fetch most recent SNRun that recorded a pipeline_hash
            rows = list(
                gc.query(
                    """
                    MATCH (r:SNRun)
                    WHERE r.pipeline_hash IS NOT NULL
                    RETURN r.pipeline_hash          AS composite,
                           r.pipeline_hash_detail   AS detail,
                           r.started_at             AS started_at,
                           r.id                     AS run_id
                    ORDER BY r.started_at DESC
                    LIMIT 1
                    """
                )
            )
            if not rows:
                return  # no prior run with hash — fresh graph, skip gate

            row = rows[0]
            prev_composite = row["composite"]
            if prev_composite == current_composite:
                return  # no change

            # Hashes differ — check whether there are generated names
            name_count_rows = list(
                gc.query("MATCH (sn:StandardName) RETURN count(sn) AS n")
            )
            name_count = name_count_rows[0]["n"] if name_count_rows else 0
            if name_count == 0:
                return  # empty graph — nothing to protect

            # Compute which keys changed for a useful message
            prev_detail: dict[str, str] = {}
            if row["detail"]:
                try:
                    prev_detail = _json.loads(row["detail"])
                except Exception:  # noqa: BLE001
                    pass
            changed_keys = diff_pipeline_hashes(prev_detail, current)

            _Console(stderr=True).print(
                "\n[bold yellow]⚠  Pipeline version changed since last cycle.[/bold yellow]\n"
                f"   Previous composite hash : [dim]{prev_composite}[/dim]\n"
                f"   Current  composite hash : [dim]{current_composite}[/dim]\n"
                f"   Keys that changed       : [yellow]{', '.join(changed_keys) or '(detail unavailable)'}[/yellow]\n"
                f"   Existing generated names: [cyan]{name_count}[/cyan]\n\n"
                "   Recommendation: run the following command before continuing:\n"
                "     [bold]imas-codex sn clear --all --force --include-sources[/bold]\n\n"
                "   To bypass this check and continue anyway:\n"
                "     [bold]imas-codex sn run --skip-clear-gate ...[/bold]\n"
            )
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        return  # graph unreachable or error — skip gate silently


def _auto_sync_grammar(*, quiet: bool = False) -> None:
    """Idempotently sync the ISN grammar spec into the graph if stale.

    Compares the graph's active ``ISNGrammarVersion`` against the installed
    ``imas_standard_names`` version. When they differ (or no grammar is
    present), runs :func:`sync_isn_grammar_to_graph` so the running pipeline
    always composes against the installed grammar. A no-op when already in
    sync.

    Best-effort: any failure (graph unreachable, ISN missing) is logged and
    swallowed so it degrades gracefully rather than crashing a run.
    """
    try:
        from imas_standard_names import __version__ as isn_version

        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.grammar_sync import sync_isn_grammar_to_graph
    except Exception:  # noqa: BLE001 — ISN absent / import failure
        logger.debug("auto grammar sync skipped (import failed)", exc_info=True)
        return

    try:
        with GraphClient() as gc:
            rows = list(
                gc.query(
                    "MATCH (v:ISNGrammarVersion {active: true}) "
                    "RETURN v.version AS version LIMIT 1"
                )
                or []
            )
            active = rows[0]["version"] if rows else None
            if active == isn_version:
                return  # already in sync — no-op
            sync_isn_grammar_to_graph(gc=gc)
        if not quiet:
            console.print(
                f"[dim]Grammar synced to ISN {isn_version}"
                f" (was {active or 'unset'}).[/dim]"
            )
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning("auto grammar sync failed (continuing): %s", exc)


@sn.command("run")
@click.option(
    "--source",
    type=click.Choice(["dd", "signals"]),
    default="dd",
    show_default=True,
    help="Source to extract candidates from",
)
@click.option(
    "--domain",
    "-d",
    "domains",
    multiple=True,
    callback=_split_whitespace,
    help=(
        "Physics domain(s) to seed. Repeatable; whitespace-separated values "
        "also accepted. Default: seed all eligible domains from DD."
    ),
)
@click.option(
    "--facility",
    type=str,
    default=None,
    help="Facility ID (required for signals source)",
)
@click.option(
    "-c",
    "--cost-limit",
    type=float,
    default=5.0,
    help="Maximum LLM cost in USD",
)
@click.option(
    "-t",
    "--time",
    "time_limit",
    type=float,
    default=None,
    help="Maximum runtime in minutes (e.g., 5). Pipeline shuts down gracefully when time expires.",
)
@click.option("--dry-run", is_flag=True, help="Preview extraction without LLM calls")
@click.option(
    "--force", is_flag=True, help="Re-generate names for already-named sources"
)
@click.option(
    "--revalidate",
    is_flag=True,
    help=(
        "Before extraction, sweep StandardName nodes with validation_status='pending' "
        "in the current scope (source/domain/ids filters) and clear validated_at so "
        "validate_worker re-runs ISN checks against the current grammar. Use after "
        "an ISN vocab/grammar update to clear legacy quarantines without a full regen."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of DD paths to process",
)
@click.option(
    "--max-sources",
    "max_sources",
    type=int,
    default=None,
    help=(
        "Cap on total StandardNameSource nodes to seed across all domains. "
        "Prevents runaway queue growth when auto-seeding without --domain."
    ),
)
@click.option(
    "--compose-model",
    type=str,
    default=None,
    help="LLM model for name composition (default: reasoning model)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")
@click.option(
    "--focus",
    "focus_paths",
    multiple=True,
    cls=_SpaceSplitMultiple,
    default=(),
    help=(
        "Focus on specific DD paths for full-pipeline processing. "
        "Runs the complete 6-pool pipeline (generate → review → refine → docs) "
        "scoped to only these items via run_id filtering. "
        "Accepts multiple --focus flags and/or quoted space-separated values "
        '(e.g., --focus "path/a path/b" --focus path/c).'
    ),
)
@click.option(
    "--reset-to",
    type=click.Choice(["extracted", "drafted"]),
    default=None,
    help=(
        "Reset standard names before generating. "
        "'extracted' clears matching SN nodes (full re-run); "
        "'drafted' resets existing drafted names (re-compose only)."
    ),
)
@click.option(
    "--from-model",
    type=str,
    default=None,
    help=(
        "Regenerate names produced by a specific model (substring match). "
        "Example: --from-model gemini matches 'google/gemini-3.1-flash-lite-preview'. "
        "Implies --force."
    ),
)
@click.option(
    "--reset-only",
    is_flag=True,
    default=False,
    help=(
        "Perform --reset-to cleanup then exit without running generation. "
        "Requires --reset-to. Useful for housekeeping without recomposing."
    ),
)
@click.option(
    "--since",
    type=str,
    default=None,
    help=(
        "Only reset/regenerate names with generated_at >= this ISO timestamp "
        "(e.g. '2026-04-19T10:00'). Combines with --reset-to and filters."
    ),
)
@click.option(
    "--before",
    type=str,
    default=None,
    help=(
        "Only reset/regenerate names with generated_at < this ISO timestamp. "
        "Combines with --since for a window."
    ),
)
@click.option(
    "--below-score",
    type=float,
    default=None,
    help=(
        "Only reset/regenerate names with reviewer_score_name < this value "
        "(0.0-1.0 scale). Requires prior `sn review` run."
    ),
)
@click.option(
    "--tier",
    type=str,
    default=None,
    help=(
        "Only reset/regenerate names with review_tier in this comma-separated "
        "list (e.g. 'poor,inadequate'). Requires prior `sn review` run."
    ),
)
@click.option(
    "--retry-quarantined",
    is_flag=True,
    default=False,
    help=("Shortcut: select names with validation_status=quarantined for regen."),
)
@click.option(
    "--retry-skipped",
    is_flag=True,
    default=False,
    help=(
        "Include StandardNameSource records with status=skipped in re-extraction "
        "(their underlying DD paths get re-queued). Useful after unit override "
        "table updates. (status='skipped' will be added in Phase B — it is OK "
        "for this flag to be a no-op today; Phase B will wire it up.)"
    ),
)
@click.option(
    "--retry-vocab-gap",
    is_flag=True,
    default=False,
    help=(
        "Select names with validation_status=quarantined AND a vocab_gap cause "
        "(or StandardNameSource.status=vocab_gap) for regen after ISN vocab "
        "updates."
    ),
)
@click.option(
    "--min-score",
    "min_score",
    type=float,
    default=DEFAULT_MIN_SCORE,
    show_default=True,
    help=(
        "Reviewer-score threshold for the refine pools.  Names / docs with a "
        "score below this value are routed to refine_name / refine_docs.  "
        "Sourced from ``defaults.DEFAULT_MIN_SCORE`` (0.80) when not provided."
    ),
)
@click.option(
    "--rotation-cap",
    "rotation_cap",
    type=int,
    default=DEFAULT_REFINE_ROTATIONS,
    show_default=True,
    help=(
        "Maximum REFINED_FROM / DOCS_REVISION_OF chain depth before a name "
        "is marked exhausted.  Sourced from "
        "``defaults.DEFAULT_REFINE_ROTATIONS`` (3) when not provided."
    ),
)
@click.option(
    "--escalation-model",
    "escalation_model",
    type=str,
    default=DEFAULT_ESCALATION_MODEL,
    show_default=True,
    help=(
        "Higher-capability model used on the final refine attempt "
        "(chain_length == rotation_cap - 1).  Sourced from "
        "``defaults.DEFAULT_ESCALATION_MODEL`` when not provided."
    ),
)
@click.option(
    "--review-name-backlog-cap",
    "review_name_backlog_cap",
    type=int,
    default=REVIEW_NAME_BACKLOG_CAP,
    show_default=True,
    help=(
        "Maximum pending review_name items before generate_name / refine_name "
        "pause.  Sourced from ``defaults.REVIEW_NAME_BACKLOG_CAP`` (200)."
    ),
)
@click.option(
    "--review-docs-backlog-cap",
    "review_docs_backlog_cap",
    type=int,
    default=REVIEW_DOCS_BACKLOG_CAP,
    show_default=True,
    help=(
        "Maximum pending review_docs items before generate_docs / refine_docs "
        "pause.  Sourced from ``defaults.REVIEW_DOCS_BACKLOG_CAP`` (200)."
    ),
)
@click.option(
    "--skip-review",
    is_flag=True,
    default=False,
    help="Skip the review phase (6-dimensional scoring).",
)
@click.option(
    "--names-only",
    "names_only",
    is_flag=True,
    default=False,
    help=(
        "Skip all docs pools (generate_docs, review_docs, refine_docs). "
        "Use for faster name-only iteration cycles at lower cost."
    ),
)
@click.option(
    "--docs-only",
    "docs_only",
    is_flag=True,
    default=False,
    help=(
        "Run ONLY the docs pools (generate_docs, review_docs, refine_docs) on "
        "already name-accepted names; skip name compose/review and auto-seeding. "
        "Use for budget-capped docs rotations."
    ),
)
@click.option(
    "--flush",
    is_flag=True,
    default=False,
    help=(
        "Drain existing work without composing new names. "
        "Skips auto-seeding and the generate_name pool; only review, "
        "refine, and docs pools run. Incompatible with --focus."
    ),
)
@click.option(
    "--only",
    "only_phase",
    type=click.Choice(
        [
            "reconcile",
            "extract",
            "compose",
            "validate",
            "consolidate",
            "persist",
            "review",
            "link",
        ],
        case_sensitive=False,
    ),
    default=None,
    help=(
        "Run only this phase — all others are skipped. "
        "extract/compose/validate/consolidate/persist select the generate phase."
    ),
)
@click.option(
    "--override-edits",
    multiple=True,
    help=(
        "Standard name IDs to bypass pipeline protection for. "
        "Allows overwriting catalog-edited fields on these names only. "
        "Repeatable: --override-edits foo --override-edits bar."
    ),
)
@click.option(
    "--skip-clear-gate",
    is_flag=True,
    default=False,
    help=(
        "Bypass the pipeline-version change check. Normally ``sn run`` "
        "exits non-zero when prompt files or ISN vocab have changed since "
        "the last SNRun node and there are existing generated names. "
        "Pass this flag to suppress the gate and continue anyway."
    ),
)
@click.option(
    "--reviewer-profile",
    "reviewer_profile",
    type=click.Choice(
        ["default", "pilot", "opus-only", "haiku-only"], case_sensitive=False
    ),
    default="default",
    show_default=True,
    envvar="IMAS_CODEX_SN_REVIEW_PROFILE",
    help=(
        "Reviewer model chain profile for the review phase. "
        "'default' → Opus+GPT-5.4+Sonnet (3-model RD-quorum). "
        "'pilot' → Haiku×2+Opus arbiter (~85%% cost reduction). "
        "'opus-only' → single Opus reviewer. "
        "'haiku-only' → single Haiku reviewer (cheapest). "
        "Also read from IMAS_CODEX_SN_REVIEW_PROFILE env var."
    ),
)
@click.option(
    "--rename",
    "rename_spec",
    type=str,
    default=None,
    help=(
        "Rename a StandardName and cascade through HAS_PARENT descendants. "
        "Format: 'OLD:NEW' (e.g. 'elongation:elongation_of_closed_flux_surface'). "
        "Short-circuits the normal pool loop — no LLM work is performed. "
        "Use --dry-run to preview, --override-edits to override catalog-edited "
        "descendants, and --include-accepted to override accepted descendants."
    ),
)
@click.option(
    "--include-accepted",
    is_flag=True,
    default=False,
    help=(
        "When used with --rename, allow renaming descendants whose "
        "name_stage is 'accepted'. Without this flag, the cascade "
        "refuses to touch catalog-authoritative names."
    ),
)
@click.argument("paths", nargs=-1)
def sn_run(
    source: str,
    domains: tuple[str, ...],
    facility: str | None,
    cost_limit: float,
    time_limit: float | None,
    dry_run: bool,
    force: bool,
    limit: int | None,
    max_sources: int | None,
    compose_model: str | None,
    verbose: bool,
    quiet: bool,
    focus_paths: tuple[str, ...],
    paths: tuple[str, ...],
    reset_to: str | None,
    from_model: str | None,
    revalidate: bool,
    reset_only: bool,
    since: str | None,
    before: str | None,
    below_score: float | None,
    tier: str | None,
    retry_quarantined: bool,
    retry_skipped: bool,
    retry_vocab_gap: bool,
    min_score: float,
    rotation_cap: int,
    escalation_model: str,
    review_name_backlog_cap: int,
    review_docs_backlog_cap: int,
    skip_review: bool,
    names_only: bool,
    docs_only: bool,
    flush: bool,
    only_phase: str | None,
    override_edits: tuple[str, ...],
    skip_clear_gate: bool,
    reviewer_profile: str,
    rename_spec: str | None,
    include_accepted: bool,
) -> None:
    """Generate standard names from a source.

    \b
    Scope routing:
      - Default: all-pool completion loop (all 6 pools concurrent)
      - With --focus: full 6-pool pipeline scoped to specific DD paths

    \b
    Focus paths can be provided as trailing arguments or via --focus:
      imas-codex sn run eq/path/a eq/path/b eq/path/c       # positional (simplest)
      imas-codex sn run --focus "path/a path/b" --focus c    # quoted space-sep
      imas-codex sn run --focus path/a --focus path/b        # repeated flags

    \b
    Examples:
      imas-codex sn run -c 50                                 # all 6 pools, full run
      imas-codex sn run --domain equilibrium -c 5             # scoped to one domain
      imas-codex sn run --domain equilibrium --domain transport  # two domains
      imas-codex sn run --domain "equilibrium transport" --dry-run  # same, space-sep
      imas-codex sn run --source signals --facility tcv --domain magnetics
      imas-codex sn run --names-only -c 5 eq/time_slice/profiles_1d/psi eq/time_slice/profiles_1d/q
      imas-codex sn run --focus eq/.../psi --focus eq/.../q   # debug multiple paths
      imas-codex sn run --reset-to drafted --reset-only
      imas-codex sn run --reset-to drafted --below-score 0.6 --reset-only
      imas-codex sn run --only link                   # resolve links only
      imas-codex sn run --override-edits foo --override-edits bar  # bypass protection on foo, bar
      imas-codex sn run --reviewer-profile pilot -c 5  # use cheap Haiku+Opus reviewer
      imas-codex sn run --min-score 0.85 --rotation-cap 5    # tighter thresholds
    """
    import os as _os

    # --- Rename short-circuit ---
    # When --rename OLD:NEW is provided, run the parent-rename cascade and
    # exit immediately.  No LLM work is performed; the pool loop is bypassed.
    if rename_spec is not None:
        _execute_rename_cascade(
            rename_spec=rename_spec,
            dry_run=dry_run,
            override_edits=list(override_edits),
            include_accepted=include_accepted,
        )
        return

    # --- Reviewer profile: propagate via env var so the review pipeline picks
    # it up automatically wherever it reads get_sn_review_names_models() /
    # get_sn_review_disagreement_threshold().
    reviewer_profile = reviewer_profile.lower()
    if reviewer_profile != "default":
        _os.environ["IMAS_CODEX_SN_REVIEW_PROFILE"] = reviewer_profile

    # --- Pipeline-version clear gate ---
    # Check if prompt/vocab/code has changed since the last SNRun.
    # Exits non-zero with a warning banner unless --skip-clear-gate is set
    # or there are no existing generated names (fresh graph).
    if not skip_clear_gate and not dry_run:
        _check_pipeline_clear_gate()

    # --- Apply --only overrides ---
    if only_phase:
        from imas_codex.standard_names.turn import skip_flags_from_only

        overrides = skip_flags_from_only(only_phase)
        if overrides.get("skip_generate", False):
            # When --only skips generate, also skip related pre-processing
            force = False
        if overrides.get("skip_review", False):
            skip_review = True
        # skip_generate handled via the overrides dict below
        skip_generate_from_only = overrides.get("skip_generate", False)
    else:
        skip_generate_from_only = False

    # Flatten --focus values (handled by _SpaceSplitMultiple.type_cast_value).
    # Merge with trailing positional paths argument.
    flat_focus = list(focus_paths) + list(paths)

    # Coerce override_edits tuple to list for downstream
    _override_edits = list(override_edits) if override_edits else None

    # ── Validate --flush constraints ──────────────────────────────────
    if flush and flat_focus:
        raise click.UsageError("--flush and --focus are mutually exclusive")
    if docs_only and names_only:
        raise click.UsageError("--docs-only and --names-only are mutually exclusive")

    # ── --reset-only: execute reset then exit (all source types) ──────
    if reset_only:
        if reset_to is None:
            raise click.UsageError("--reset-only requires --reset-to")
        if not dry_run:
            source_arg = "dd" if source == "dd" else "signals"
            ids_filter: str | None = None
            _tiers = [t.strip() for t in tier.split(",")] if tier else None
            _validation_status: str | None = None
            if retry_quarantined:
                _validation_status = "quarantined"
            _reset_filter_kwargs: dict[str, Any] = {
                "since": since,
                "before": before,
                "below_score": below_score,
                "tiers": _tiers,
                "validation_status": _validation_status,
            }
            from imas_codex.standard_names.graph_ops import (
                clear_standard_names,
                reset_standard_names,
            )

            if reset_to == "extracted":
                n = clear_standard_names(
                    source_filter=source_arg,
                    ids_filter=ids_filter,
                    include_accepted=include_accepted,
                    **_reset_filter_kwargs,
                )
                console.print(
                    f"[yellow]--reset-to extracted:[/yellow] cleared {n} SN nodes"
                )
            elif reset_to == "drafted":
                # With --retry-quarantined the targets sit at any live stage
                # (reviewed/refining/accepted); select by the filter set and
                # re-stage them to 'drafted' for recompose.
                n = reset_standard_names(
                    from_stage=None if retry_quarantined else "drafted",
                    to_stage="drafted" if retry_quarantined else None,
                    include_accepted=include_accepted,
                    source_filter=source_arg,
                    ids_filter=ids_filter,
                    **_reset_filter_kwargs,
                )
                console.print(
                    f"[yellow]--reset-to drafted:[/yellow] reset {n} SN nodes"
                )
        console.print(
            "[green]--reset-only:[/green] reset complete, exiting without generation"
        )
        return

    if not dry_run:
        _require_embed_ready("sn run")
        # Auto-sync the ISN grammar into the graph when the active grammar
        # version differs from the installed ISN package. Idempotent — a
        # no-op when already in sync — and best-effort (a sync failure logs
        # and continues rather than crashing the run). Runs once at startup,
        # before any pool launches; never on the per-name hot path.
        _auto_sync_grammar(quiet=quiet)

    # ── --focus routing: full 6-pool pipeline scoped by run_id ────────
    if flat_focus:
        import uuid as _uuid

        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        scope_run_id = str(_uuid.uuid4())

        # 1. Clear stale run_ids from previous focused runs.
        with GraphClient() as gc:
            gc.query(
                "MATCH (sn:StandardName) WHERE sn.run_id IS NOT NULL "
                "SET sn.run_id = NULL"
            )
            gc.query(
                "MATCH (sns:StandardNameSource) WHERE sns.run_id IS NOT NULL "
                "SET sns.run_id = NULL"
            )

        # 2. Seed StandardNameSource nodes for each focused path.
        sources = []
        for path in flat_focus:
            sources.append(
                {
                    "id": f"dd:{path}",
                    "source_type": "dd",
                    "source_id": path,
                    "dd_path": path,
                    "batch_key": "focus",
                    "status": "extracted",
                    "description": "",
                }
            )
        written = merge_standard_name_sources(sources, force=True)
        if not quiet:
            click.echo(f"Seeded {written} focus source(s) (run_id={scope_run_id[:8]}…)")

        # 3. Post-stamp run_id on the seeded SNS nodes.
        sns_ids = [f"dd:{p}" for p in flat_focus]
        with GraphClient() as gc:
            gc.query(
                "UNWIND $ids AS sid "
                "MATCH (sns:StandardNameSource {id: sid}) "
                "SET sns.run_id = $run_id",
                ids=sns_ids,
                run_id=scope_run_id,
            )

        # 4. Force-reset any existing StandardNames for these paths.
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS sid
                MATCH (sns:StandardNameSource {id: sid})-[:PRODUCED_NAME]->(sn:StandardName)
                SET sn.name_stage = 'pending',
                    sn.docs_stage = 'pending',
                    sn.run_id = $run_id,
                    sn.reviewed_name_at = null,
                    sn.reviewed_docs_at = null,
                    sn.reviewer_score_name = null,
                    sn.reviewer_score_docs = null,
                    sn.claim_token = null,
                    sn.claimed_at = null
                """,
                ids=sns_ids,
                run_id=scope_run_id,
            )

        # 5. Route through the pool orchestrator with scope_run_id.
        _run_sn_cmd(
            cost_limit=cost_limit,
            time_limit=time_limit,
            per_domain_limit=limit,
            dry_run=dry_run,
            quiet=quiet,
            domains=domains,
            verbose=verbose,
            min_score=min_score,
            rotation_cap=rotation_cap,
            escalation_model=escalation_model,
            review_name_backlog_cap=review_name_backlog_cap,
            review_docs_backlog_cap=review_docs_backlog_cap,
            skip_generate=skip_generate_from_only,
            skip_review=skip_review,
            names_only=names_only,
            docs_only=docs_only,
            source=source,
            override_edits=_override_edits,
            only=only_phase,
            max_sources=max_sources,
            scope_run_id=scope_run_id,
        )
        return

    # Handle --reset-to BEFORE scope routing so it applies to BOTH the DD
    # pool-orchestrator path and the signals single-pass path. (This block
    # previously sat AFTER the use_pools early-return, making --reset-to a
    # silent no-op on the default DD path.) clear_standard_names re-seeds the
    # orphaned StandardNameSources to 'extracted' (its Step E) so the pool
    # orchestrator's extract phase re-composes them.
    if reset_to is not None and not dry_run:
        _tiers = [t.strip() for t in tier.split(",")] if tier else None
        _reset_filter_kwargs: dict[str, Any] = {
            "since": since,
            "before": before,
            "below_score": below_score,
            "tiers": _tiers,
            "validation_status": "quarantined" if retry_quarantined else None,
        }
        source_arg = "dd" if source == "dd" else "signals"
        from imas_codex.standard_names.graph_ops import (
            clear_standard_names,
            reset_standard_names,
        )

        if reset_to == "extracted":
            # --reset-to extracted deletes matching SN nodes so they recompose
            # from their (authoritative) DD source. clear_standard_names deletes
            # only drafted-stage nodes unless include_accepted is set — thread
            # the flag through so a migration can re-derive accepted/quarantined
            # names (e.g. sn run --retry-quarantined --reset-to extracted
            # --include-accepted).
            n = clear_standard_names(
                source_filter=source_arg,
                include_accepted=include_accepted,
                **_reset_filter_kwargs,
            )
            console.print(
                f"[yellow]--reset-to extracted:[/yellow] cleared {n} SN nodes"
            )
        elif reset_to == "drafted":
            n = reset_standard_names(
                from_stage=None if retry_quarantined else "drafted",
                to_stage="drafted" if retry_quarantined else None,
                include_accepted=include_accepted,
                source_filter=source_arg,
                **_reset_filter_kwargs,
            )
            console.print(f"[yellow]--reset-to drafted:[/yellow] reset {n} SN nodes")

    # Scope-routing: default (DD source) → pool orchestrator.
    # Runs all 6 pools concurrently, sampling globally from the available
    # pool of StandardNameSource / StandardName nodes.
    # --domain is forwarded to scope the extract_phase seeding only;
    # the pools themselves are domain-agnostic.
    use_pools = source == "dd"

    if use_pools:
        _run_sn_cmd(
            cost_limit=cost_limit,
            time_limit=time_limit,
            per_domain_limit=limit,
            dry_run=dry_run,
            quiet=quiet,
            domains=domains,
            verbose=verbose,
            min_score=min_score,
            rotation_cap=rotation_cap,
            escalation_model=escalation_model,
            review_name_backlog_cap=review_name_backlog_cap,
            review_docs_backlog_cap=review_docs_backlog_cap,
            skip_generate=skip_generate_from_only,
            skip_review=skip_review,
            names_only=names_only,
            docs_only=docs_only,
            flush=flush,
            source=source,
            override_edits=_override_edits,
            only=only_phase,
            max_sources=max_sources,
        )
        return

    # --ids has been removed from this command; scope narrowing is domain-based
    # so it works uniformly across DD and facility-signals sources.
    ids_filter_sp: str | None = None

    # Single-pass path uses a scalar domain_filter; derive from --domain tuple.
    domain_filter: str | None = domains[0] if len(domains) == 1 else None

    # Validate: signals source requires facility
    if source == "signals" and not facility:
        raise click.UsageError("--facility is required when --source is signals")

    # --from-model implies --force (selecting by model only makes sense for regeneration)
    if from_model:
        force = True

    # Log Phase B/C flags that are pending wire-up
    if retry_skipped:
        logger.info("--retry-skipped set (pending Phase B wire-up)")
    if retry_vocab_gap:
        logger.info("--retry-vocab-gap set (pending Phase B wire-up)")

    # Handle --revalidate: clear validated_at on pending SNs in current scope so
    # validate_worker re-runs ISN checks without a full regen. Safe with any source.
    if revalidate and not dry_run:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            where_clauses = [
                "sn.validation_status = 'pending'",
                "sn.validated_at IS NOT NULL",
            ]
            params: dict[str, Any] = {}
            if domain_filter:
                where_clauses.append("sn.physics_domain = $domain")
                params["domain"] = domain_filter
            if source == "dd":
                where_clauses.append(
                    "EXISTS { MATCH (sn)<-[:HAS_STANDARD_NAME]-(:IMASNode) }"
                )
            elif source == "signals":
                where_clauses.append(
                    "EXISTS { MATCH (sn)<-[:HAS_STANDARD_NAME]-(:FacilitySignal) }"
                )
            q = f"""
                MATCH (sn:StandardName)
                WHERE {" AND ".join(where_clauses)}
                WITH sn, sn.id AS id
                SET sn.validated_at = NULL, sn.claimed_at = NULL, sn.claim_token = NULL
                RETURN count(sn) AS n
            """
            rows = list(gc.query(q, **params))
            n = rows[0]["n"] if rows else 0
            console.print(
                f"[yellow]--revalidate:[/yellow] cleared validated_at on {n} pending SN node(s)"
            )

    from imas_codex.discovery.base.llm import set_litellm_offline_env

    set_litellm_offline_env()

    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )

    use_rich = use_rich_output()
    console_obj = setup_logging("sn", "sn", use_rich, verbose=verbose)
    log_print = make_log_print("sn", console_obj)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Determine effective facility for state
    effective_facility = facility if source == "signals" else "dd"

    log_print("\n[bold]Standard Name Build[/bold]")
    log_print(f"  Source: {source}")
    if domain_filter:
        log_print(f"  Domain filter: {domain_filter}")
    if facility:
        log_print(f"  Facility: {facility}")
    if dry_run:
        log_print("  Mode: dry run")
    if force:
        log_print("  Force: re-generating all names")
    if from_model:
        log_print(f"  From model: {from_model} (substring match)")
    if limit:
        log_print(f"  Limit: {limit} paths")
    if compose_model:
        log_print(f"  Compose model: {compose_model}")
    log_print(f"  Cost limit: ${cost_limit:.2f}")
    log_print("")

    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.pool_adapter import run_explicit_paths
    from imas_codex.standard_names.state import StandardNameBuildState

    # Build progress display
    display = None
    if use_rich and not quiet:
        try:
            from imas_codex.standard_names.progress import StandardNameProgressDisplay

            display = StandardNameProgressDisplay(
                source=source,
                console=console_obj,
                cost_limit=cost_limit,
                mode_label="DRY RUN" if dry_run else None,
            )
        except Exception:
            logger.debug("Could not create progress display", exc_info=True)

    # Resolve name-only batch size from pyproject default when unspecified.
    name_only: bool = False
    name_only_batch_size: int = 50
    try:
        from imas_codex.settings import _get_section

        name_only_batch_size = int(
            _get_section("sn-compose").get("name-only-batch-size", 50)
        )
    except Exception:
        pass

    state = StandardNameBuildState(
        facility=effective_facility,
        source=source,
        ids_filter=ids_filter_sp,
        domain_filter=domain_filter,
        facility_filter=facility,
        cost_limit=cost_limit,
        dry_run=dry_run,
        force=force,
        regen=min_score is not None,
        min_score=min_score,
        limit=limit,
        compose_model=compose_model,
        from_model=from_model,
        name_only=name_only,
        name_only_batch_size=name_only_batch_size,
        budget_manager=BudgetManager(cost_limit),
    )

    if display:
        display.set_engine_state(state)

    async def _run(stop_event, service_monitor):
        if service_monitor:
            state.service_monitor = service_monitor
        await run_explicit_paths(
            state,
            stop_event=stop_event,
            on_worker_status=display.on_worker_status if display else None,
        )
        return state.stats

    config = DiscoveryConfig(
        facility=effective_facility,
        domain="sn",
        facility_config={},
        display=display,
        check_graph=True,
        check_embed=False,
        check_ssh=False,
        check_auth=source != "dd",  # signals source might need auth
        check_model=not dry_run,
        model_section="sn-compose",
        suppress_loggers=[
            "imas_codex.standard_names",
        ],
        verbose=verbose,
    )

    result = run_discovery(config, _run)

    # Print summary
    if result:
        extracted = result.get("extract_count", 0)
        composed = result.get("generate_name_count", 0)
        attached = result.get("attachments", 0)
        validated = result.get("validate_valid", 0)
        compose_cost = result.get("compose_cost", 0.0)
        compose_model_name = result.get("compose_model", "")
        parts = [
            f"Extracted: {extracted}",
            f"Composed: {composed}",
        ]
        if attached:
            parts.append(f"Attached: {attached}")
        parts.append(f"Validated: {validated}")
        if compose_cost > 0:
            parts.append(f"Cost: ${compose_cost:.4f}")
        log_print(", ".join(parts))
        if compose_model_name:
            log_print(f"Model: {compose_model_name}")
        if dry_run:
            log_print("(dry run — no LLM calls or graph writes)")


@sn.command("bench")
@click.option(
    "--models",
    type=str,
    multiple=True,
    help="Model(s) to benchmark. Repeat for multiple or use commas. "
    "Defaults to [sn-benchmark].compose-models.",
)
@click.option(
    "--max-candidates",
    type=int,
    default=None,
    help="Limit to first N reference paths (default: all 54)",
)
@click.option(
    "--runs",
    type=int,
    default=1,
    help="Runs per model for consistency check",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="LLM temperature (0.0 for reproducibility)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="JSON report output path",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--reviewer-model",
    type=str,
    default=None,
    help="Single judge model for quality scoring (backward compat). "
    "Defaults to [sn-benchmark].reviewer-model.",
)
@click.option(
    "--reviewer-models",
    type=str,
    multiple=True,
    help="Reviewer model(s) for multi-reviewer matrix. Repeat or use commas. "
    "Defaults to [sn-benchmark].reviewer-models.",
)
@click.option(
    "--include-docs/--names-only",
    "include_docs",
    default=False,
    help="--include-docs: benchmark names + docs. "
    "--names-only: benchmark name generation only (default).",
)
@click.option(
    "--physics",
    is_flag=True,
    help="Run the physics-correctness judge (gold-set gated).",
)
@click.option(
    "--gold-set",
    "gold_set",
    type=click.Path(exists=True),
    default=None,
    help="Path to human gold labels (research/physics_bench_gold.json).",
)
@click.option(
    "--physics-judge-model",
    "physics_judge_model",
    default=None,
    help="Judge model (default: the calibrated sn-benchmark reviewer model, opus-4.8).",
)
@click.option(
    "--reasoning-effort",
    "reasoning_effort",
    type=click.Choice(["low", "medium", "high", "max", "none"]),
    default=None,
    help="Override the compose reasoning effort (default: [sn-compose] config). "
    "Use to scan the effort axis with a fixed model: vary this and compare "
    "accuracy vs speed vs cost across runs.",
)
@click.option(
    "--review-reasoning-effort",
    "review_reasoning_effort",
    type=click.Choice(["low", "medium", "high", "max", "none"]),
    default=None,
    help="Override the REVIEWER reasoning effort (default: provider default). "
    "Use to scan the review-effort axis with a fixed reviewer set: measure "
    "bad-name catching vs cost across effort levels.",
)
@click.option(
    "--rescore",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Load an EXISTING benchmark report JSON and run ONLY the compose-time "
    "description-scoring pass on its stored candidates, then merge + write back "
    "(to --output if given, else in place). Does not re-run composition or "
    "names review.",
)
def sn_bench(
    models: tuple[str, ...],
    max_candidates: int | None,
    runs: int,
    temperature: float,
    output: str | None,
    verbose: bool,
    reviewer_model: str | None,
    reviewer_models: tuple[str, ...],
    include_docs: bool,
    physics: bool,
    gold_set: str | None,
    physics_judge_model: str | None,
    reasoning_effort: str | None,
    review_reasoning_effort: str | None,
    rescore: str | None,
) -> None:
    """Benchmark LLM models on standard name generation.

    Uses a fixed reference dataset of 54 curated DD paths for reproducible
    cross-model comparison. Results include grammar validity, reference
    overlap, reviewer quality scores, cost, and speed.

    When --models is omitted, loads the model list from
    [tool.imas-codex.sn-benchmark].compose-models in pyproject.toml.

    \b
    Examples:
      imas-codex sn bench
      imas-codex sn bench --max-candidates 5
      imas-codex sn bench --models anthropic/claude-sonnet-4.6 --models openai/gpt-5.4
      imas-codex sn bench --models anthropic/claude-sonnet-4.6,openai/gpt-5.4
      imas-codex sn bench --include-docs
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # --- Standalone description rescore of an existing report ---
    if rescore:
        from pathlib import Path

        from imas_codex.cli.utils import run_async
        from imas_codex.standard_names.benchmark import (
            BenchmarkReport,
            render_comparison_table,
            rescore_descriptions_report,
        )

        src_path = Path(rescore)
        report = BenchmarkReport.from_json(src_path.read_text())

        # Optional reviewer override for the rescore pass
        if reviewer_models:
            rev_list = []
            for m in reviewer_models:
                rev_list.extend(part.strip() for part in m.split(",") if part.strip())
            report.config.reviewer_models = rev_list
            report.config.reviewer_model = rev_list[0] if rev_list else None
        elif reviewer_model:
            report.config.reviewer_models = [reviewer_model]
            report.config.reviewer_model = reviewer_model

        rev_display = ", ".join(m.split("/")[-1] for m in report.config.reviewer_models)
        console.print("[bold]SN Benchmark — description rescore[/bold]")
        console.print(f"  Source report: {src_path}")
        console.print(f"  Reviewer(s): {rev_display}")
        console.print()

        report = run_async(rescore_descriptions_report(report))
        render_comparison_table(report)

        dest = Path(output) if output else src_path
        report.save_atomic(str(dest))
        console.print(f"\n[green]Report saved:[/green] {dest}")
        return

    from imas_codex.settings import (
        get_sn_benchmark_compose_models,
        get_sn_benchmark_reviewer_model,
        get_sn_benchmark_reviewer_models,
    )

    # Resolve model list: CLI flag → pyproject.toml → built-in defaults
    if models:
        # Support both --models a,b and --models a --models b
        model_list = []
        for m in models:
            model_list.extend(part.strip() for part in m.split(",") if part.strip())
    else:
        model_list = get_sn_benchmark_compose_models()

    if not model_list:
        raise click.UsageError(
            "No models configured. Pass --models or set "
            "[tool.imas-codex.sn-benchmark].compose-models in pyproject.toml."
        )

    # Resolve reviewer models: --reviewer-models → --reviewer-model → pyproject
    if reviewer_models:
        rev_list = []
        for m in reviewer_models:
            rev_list.extend(part.strip() for part in m.split(",") if part.strip())
    elif reviewer_model:
        rev_list = [reviewer_model]
    else:
        rev_list = get_sn_benchmark_reviewer_models()

    # Keep backward-compat reviewer_model as first in list
    rev_model_single = rev_list[0] if rev_list else get_sn_benchmark_reviewer_model()

    from imas_codex.standard_names.benchmark import (
        BenchmarkConfig,
        render_comparison_table,
        run_benchmark,
    )
    from imas_codex.standard_names.benchmark_reference import REFERENCE_NAMES

    names_only = not include_docs
    total_ref = len(REFERENCE_NAMES)
    effective_max = max_candidates if max_candidates else total_ref

    # Resolve output path early so run_benchmark can save incrementally
    from pathlib import Path

    if output is None:
        bench_dir = Path.home() / ".local" / "share" / "imas-codex" / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
        out_path = bench_dir / f"sn_benchmark_{ts}.json"
    else:
        out_path = Path(output)

    # "none" → explicit no-reasoning-budget; map to None-effort sentinel handled
    # downstream (config.reasoning_effort stays the literal string the override
    # passes through to acall_llm_structured).
    config = BenchmarkConfig(
        models=model_list,
        max_candidates=effective_max,
        runs_per_model=runs,
        temperature=temperature,
        reviewer_model=rev_model_single,
        reviewer_models=rev_list,
        names_only=names_only,
        output_path=str(out_path),
        physics_judge=physics,
        gold_set_path=gold_set,
        physics_judge_model=physics_judge_model,
        reasoning_effort=reasoning_effort,
        review_reasoning_effort=review_reasoning_effort,
    )

    mode_str = "names + docs" if include_docs else "names-only"
    rev_display = ", ".join(m.split("/")[-1] for m in rev_list)
    console.print("[bold]SN Benchmark[/bold]")
    console.print(f"  Models: {', '.join(model_list)}")
    console.print(f"  Reference paths: {min(effective_max, total_ref)}/{total_ref}")
    console.print(f"  Runs per model: {runs}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Reviewer(s): {rev_display}")
    console.print(f"  Mode: {mode_str}")
    console.print(f"  Output: {out_path}")
    console.print()

    from imas_codex.cli.utils import run_async

    # Gold-set calibration gate: validate the judge before trusting its scores.
    if physics and gold_set:
        import json as _json
        from functools import partial

        from imas_codex.settings import get_sn_benchmark_reviewer_model
        from imas_codex.standard_names.physics_judge import (
            judge_name_physics,
            run_calibration_gate,
        )

        gold = _json.loads(open(gold_set).read())
        jmodel = physics_judge_model or get_sn_benchmark_reviewer_model()
        cal = run_async(
            run_calibration_gate(gold, partial(judge_name_physics, model=jmodel))
        )
        console.print(
            f"[bold]Judge calibration:[/] trusted={cal['trusted']} "
            f"overall={cal['overall_agreement']} "
            f"hardcase={cal['hardcase_errors_caught']}/{cal['hardcase_errors_total']}"
        )
        if not cal["trusted"]:
            console.print(
                "[red]Judge FAILED calibration — physics scores are advisory; "
                f"hardcase misses: {cal['hardcase_misses']}. "
                "Fall back to human scoring.[/]"
            )

    report = run_async(run_benchmark(config))

    # Display comparison table
    render_comparison_table(report)

    # Display provenance
    prov = report.provenance
    if prov.codex_version or prov.codex_commit:
        console.print(
            f"\nProvenance: codex={prov.codex_version} ({prov.codex_commit}), "
            f"ISN={prov.isn_version}, DD={prov.dd_version}"
        )
    if report.dataset_hash:
        console.print(f"Dataset hash: {report.dataset_hash}")
    console.print(
        f"Source paths ({report.extraction_count}): "
        + ", ".join(report.extraction_source_ids[:5])
        + ("…" if len(report.extraction_source_ids) > 5 else "")
    )

    # Final atomic save (run_benchmark already saved incrementally)
    report.save_atomic(str(out_path))
    console.print(f"\n[green]Report saved:[/green] {out_path}")


@sn.command("coverage")
@click.option(
    "--physics-domain",
    "physics_domain",
    type=_PHYSICS_DOMAIN_CHOICE,
    default=None,
    help="Restrict eligibility counts to this physics domain.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit machine-readable JSON instead of rich tables.",
)
def sn_coverage(physics_domain: str | None, as_json: bool) -> None:
    """Pre-run coverage report: how many names do we expect to mint?

    Prints a three-section report:

    \b
      1. DD extract eligibility — leaves admitted by the B3' invariant.
      2. Already-minted coverage — existing StandardName catalog.
      3. Work remaining — uncovered paths + rough LLM cost estimate.

    \b
    Examples:
      imas-codex sn coverage
      imas-codex sn coverage --physics-domain equilibrium
      imas-codex sn coverage --json | jq .to_compose
    """
    from imas_codex.standard_names.coverage import compute_coverage

    try:
        report = compute_coverage(physics_domain=physics_domain)
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]Error computing coverage:[/red] {exc}")
        raise SystemExit(1) from exc

    if as_json:
        click.echo(report.to_json())
        return

    # --- Rich output --------------------------------------------------------
    from rich.table import Table

    scope_label = (
        f"[bold cyan]{physics_domain}[/bold cyan]"
        if physics_domain
        else "[bold cyan]all domains[/bold cyan]"
    )
    console.print(f"\n[bold]SN Coverage Report[/bold] — scope: {scope_label}\n")

    # Section 1: DD eligibility
    elig_table = Table(
        title="1 · DD Extract Eligibility (B3' invariant)", show_header=True
    )
    elig_table.add_column("Metric", style="cyan")
    elig_table.add_column("Count", justify="right")
    elig_table.add_row(
        "[bold]Total eligible leaves[/bold]", f"[bold]{report.eligible_total:,}[/bold]"
    )
    elig_table.add_row(
        "  … with HAS_ERROR edges (B9 parents)", f"{report.eligible_with_errors:,}"
    )
    console.print(elig_table)

    cat_table = Table(title="By node_category", show_header=True)
    cat_table.add_column("node_category", style="dim")
    cat_table.add_column("Count", justify="right")
    for k, v in sorted(report.eligible_by_category.items(), key=lambda x: -x[1]):
        cat_table.add_row(k, f"{v:,}")
    console.print(cat_table)

    nt_table = Table(title="By node_type", show_header=True)
    nt_table.add_column("node_type", style="dim")
    nt_table.add_column("Count", justify="right")
    for k, v in sorted(report.eligible_by_node_type.items(), key=lambda x: -x[1]):
        nt_table.add_row(k, f"{v:,}")
    console.print(nt_table)

    # Only show domain table when not filtered (it's redundant when filtered)
    if not physics_domain:
        dom_table = Table(title="By physics_domain (top 15)", show_header=True)
        dom_table.add_column("physics_domain", style="dim")
        dom_table.add_column("Count", justify="right")
        for k, v in sorted(report.eligible_by_domain.items(), key=lambda x: -x[1])[:15]:
            dom_table.add_row(k, f"{v:,}")
        console.print(dom_table)

    # Section 2: Already-minted
    console.print()
    minted_table = Table(title="2 · Already-Minted Coverage", show_header=True)
    minted_table.add_column("Metric", style="cyan")
    minted_table.add_column("Count", justify="right")
    minted_table.add_row(
        "[bold]Total StandardName nodes[/bold]", f"[bold]{report.sn_total:,}[/bold]"
    )
    minted_table.add_row(
        "  Error-sibling names (deterministic:dd_error_modifier)",
        f"{report.error_siblings_minted:,}",
    )
    minted_table.add_row(
        "  IMASNodes covered (HAS_STANDARD_NAME)", f"{report.covered_parents:,}"
    )
    console.print(minted_table)

    ps_table = Table(title="By name_stage", show_header=True)
    ps_table.add_column("name_stage", style="dim")
    ps_table.add_column("Count", justify="right")
    for k, v in sorted(report.sn_by_name_stage.items(), key=lambda x: -x[1]):
        ps_table.add_row(k, f"{v:,}")
    console.print(ps_table)

    vs_table = Table(title="By validation_status", show_header=True)
    vs_table.add_column("validation_status", style="dim")
    vs_table.add_column("Count", justify="right")
    for k, v in sorted(report.sn_by_validation_status.items(), key=lambda x: -x[1]):
        vs_table.add_row(k, f"{v:,}")
    console.print(vs_table)

    # Section 3: Work remaining
    console.print()
    work_table = Table(title="3 · Work Remaining", show_header=True)
    work_table.add_column("Metric", style="cyan")
    work_table.add_column("Value", justify="right")
    work_table.add_row(
        "[bold]To compose (uncovered leaves)[/bold]",
        f"[bold]{report.to_compose:,}[/bold]",
    )
    work_table.add_row("  … with HAS_ERROR edges", f"{report.to_compose_with_errors:,}")
    work_table.add_row(
        "  Expected error siblings (3×)", f"{report.expected_error_siblings:,}"
    )

    if report.cost_per_name is not None:
        work_table.add_row(
            "Avg cost/name (from SNRun telemetry)",
            f"${report.cost_per_name:.5f}",
        )
        if report.estimated_compose_cost is not None:
            work_table.add_row(
                "Estimated total compose cost",
                f"${report.estimated_compose_cost:.2f}",
            )
    else:
        work_table.add_row(
            "Cost estimate",
            "[dim]unknown — no prior SNRun telemetry[/dim]",
        )
    console.print(work_table)
    console.print()


@sn.command("status")
def sn_status() -> None:
    """Show standard name statistics."""
    from imas_codex.graph.client import GraphClient

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (sn:StandardName)
                RETURN count(sn) AS total,
                       count(CASE WHEN 'dd' IN sn.source_types THEN 1 END) AS from_dd,
                       count(CASE WHEN 'signals' IN sn.source_types THEN 1 END) AS from_signals,
                       count(CASE WHEN 'derived' IN sn.source_types THEN 1 END) AS from_derived
            """
            )
            row = next(iter(result), None)
            if row:
                console.print(f"[bold]Standard Names:[/bold] {row['total']}")
                console.print(f"  From DD: {row['from_dd']}")
                console.print(f"  From signals: {row['from_signals']}")
                console.print(f"  From derived: {row['from_derived']}")
            else:
                console.print("No standard names in graph")

            # Validation status breakdown
            vstatus_result = gc.query(
                """
                MATCH (sn:StandardName)
                RETURN coalesce(sn.validation_status, 'unset') AS status,
                       count(sn) AS cnt
                ORDER BY cnt DESC
            """
            )
            if vstatus_result:
                from rich.table import Table as RichTable

                console.print()
                console.print("[bold]Validation Status[/bold]")
                vtable = RichTable(show_header=True)
                vtable.add_column("Status")
                vtable.add_column("Count", justify="right")
                for vrow in vstatus_result:
                    vtable.add_row(vrow["status"], str(vrow["cnt"]))
                console.print(vtable)

            # Name stage breakdown
            name_stage_rows = list(
                gc.query("""
                MATCH (sn:StandardName)
                RETURN sn.name_stage AS stage, count(*) AS n
                ORDER BY n DESC
            """)
            )
            if name_stage_rows:
                console.print()
                console.print("[bold]Name Stage[/bold]")
                ns_table = RichTable(show_header=True)
                ns_table.add_column("Stage")
                ns_table.add_column("Count", justify="right")
                for ns_row in name_stage_rows:
                    ns_table.add_row(ns_row["stage"] or "—", str(ns_row["n"]))
                console.print(ns_table)

                # Acceptance rate
                ns = {r["stage"] or "—": r["n"] for r in name_stage_rows}
                accepted = ns.get("accepted", 0)
                superseded = ns.get("superseded", 0)
                total_incl = sum(ns.values())
                total_excl = total_incl - superseded
                rate_excl = 100 * accepted / max(total_excl, 1)
                rate_incl = 100 * accepted / max(total_incl, 1)
                console.print(
                    f"Acceptance rate (excl. superseded): "
                    f"[bold]{rate_excl:.1f}%[/bold] ({accepted} / {total_excl})"
                )
                console.print(
                    f"Acceptance rate (incl. superseded): "
                    f"[bold]{rate_incl:.1f}%[/bold] ({accepted} / {total_incl})"
                )

            # Docs stage breakdown
            docs_stage_rows = list(
                gc.query("""
                MATCH (sn:StandardName)
                RETURN sn.docs_stage AS stage, count(*) AS n
                ORDER BY n DESC
            """)
            )
            if docs_stage_rows:
                console.print()
                console.print("[bold]Docs Stage[/bold]")
                ds_table = RichTable(show_header=True)
                ds_table.add_column("Stage")
                ds_table.add_column("Count", justify="right")
                for ds_row in docs_stage_rows:
                    ds_table.add_row(ds_row["stage"] or "—", str(ds_row["n"]))
                console.print(ds_table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

    # StandardNameSource status
    from rich.table import Table

    from imas_codex.standard_names.graph_ops import get_standard_name_source_stats

    source_stats = get_standard_name_source_stats()
    if source_stats:
        console.print()
        console.print("[bold]StandardNameSource Pipeline Status[/bold]")
        source_table = Table(show_header=True)
        source_table.add_column("Status")
        source_table.add_column("Count", justify="right")
        total = 0
        for status_name in [
            "extracted",
            "composed",
            "attached",
            "vocab_gap",
            "failed",
            "stale",
            "skipped",
        ]:
            count = source_stats.get(status_name, 0)
            total += count
            if count > 0:
                source_table.add_row(status_name, str(count))
        source_table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
        console.print(source_table)

    # Skipped sources breakdown by reason (DD unit overrides, etc.)
    try:
        from imas_codex.standard_names.graph_ops import get_skipped_source_counts

        skip_counts = get_skipped_source_counts()
    except Exception as exc:  # pragma: no cover — graph connection issues
        skip_counts = {}
        logger.debug("Could not fetch skipped source counts: %s", exc)

    if skip_counts:
        console.print()
        console.print("[bold]Skipped sources (by reason)[/bold]")
        skip_table = Table(show_header=True)
        skip_table.add_column("Skip Reason")
        skip_table.add_column("Count", justify="right")
        skip_total = 0
        for reason, count in skip_counts.items():
            skip_table.add_row(reason, str(count))
            skip_total += count
        skip_table.add_row("[bold]Total[/bold]", f"[bold]{skip_total}[/bold]")
        console.print(skip_table)

    # Latest SNRun (from sn run rotator)
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rr_rows = list(
                gc.query(
                    """
                MATCH (rr:SNRun)
                RETURN rr.id AS id,
                       rr.started_at AS started_at,
                       rr.stopped_at AS stopped_at,
                       rr.stop_reason AS stop_reason,
                       rr.elapsed_s AS elapsed_s,
                       rr.cost_spent AS cost_spent,
                       rr.cost_limit AS cost_limit,
                       rr.names_composed AS names_composed,
                       rr.names_enriched AS names_enriched,
                       rr.names_reviewed AS names_reviewed,
                       rr.names_regenerated AS names_regenerated,
                       rr.domains_touched AS domains_touched
                ORDER BY rr.started_at DESC
                LIMIT 1
                """
                )
            )
    except Exception as exc:  # pragma: no cover
        rr_rows = []
        logger.debug("Could not fetch latest SNRun: %s", exc)

    if rr_rows:
        rr = rr_rows[0]
        console.print()
        console.print("[bold]Latest Rotation (sn run)[/bold]")
        rr_table = Table(show_header=True)
        rr_table.add_column("Field")
        rr_table.add_column("Value")
        rr_table.add_row("id", str(rr["id"]))
        rr_table.add_row("started_at", str(rr["started_at"]))
        rr_table.add_row("stopped_at", str(rr["stopped_at"] or "—"))
        rr_table.add_row("stop_reason", str(rr["stop_reason"] or "—"))
        _elapsed = rr.get("elapsed_s")
        if _elapsed is not None:
            _es = float(_elapsed)
            if _es >= 3600:
                _elapsed_str = f"{int(_es // 3600)}h {int((_es % 3600) // 60)}m"
            elif _es >= 60:
                _elapsed_str = f"{int(_es // 60)}m {int(_es % 60)}s"
            else:
                _elapsed_str = f"{_es:.1f}s"
            rr_table.add_row("elapsed", _elapsed_str)
        rr_table.add_row(
            "cost",
            f"${float(rr['cost_spent'] or 0):.4f} / ${float(rr['cost_limit'] or 0):.2f}",
        )
        rr_table.add_row("names_composed", str(rr["names_composed"] or 0))
        rr_table.add_row("names_enriched", str(rr["names_enriched"] or 0))
        rr_table.add_row("names_reviewed", str(rr["names_reviewed"] or 0))
        rr_table.add_row("names_regenerated", str(rr["names_regenerated"] or 0))
        console.print(rr_table)

    # --- Linking integrity & review cost --------------------------------
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            integrity = next(
                iter(
                    gc.query(
                        """
                        MATCH (sn:StandardName)
                        WITH count(sn) AS total_sn,
                             count(CASE WHEN NOT (sn)<-[:PRODUCED_NAME]-()
                                         AND sn.model <> 'deterministic:dd_error_modifier'
                                         AND sn.name_stage <> 'superseded'
                                   THEN 1 END) AS orphan_sn,
                             count(CASE WHEN sn.model = 'deterministic:dd_error_modifier'
                                   THEN 1 END) AS error_siblings
                        MATCH (s:StandardNameSource)
                        WITH total_sn, orphan_sn, error_siblings,
                             count(CASE WHEN s.status IN ['composed','attached']
                                         AND NOT (s)-[:PRODUCED_NAME]->()
                                   THEN 1 END) AS orphan_src
                        RETURN total_sn, orphan_sn, orphan_src, error_siblings
                        """
                    )
                ),
                None,
            )
    except Exception as exc:  # pragma: no cover — graph connection issues
        integrity = None
        logger.debug("Could not fetch linking integrity: %s", exc)

    if integrity:
        console.print()
        console.print("[bold]Linking Integrity[/bold]")
        li_table = Table(show_header=True)
        li_table.add_column("Metric")
        li_table.add_column("Count", justify="right")
        li_table.add_row(
            "Orphan StandardName (no PRODUCED_NAME edge, excl. error siblings & superseded)",
            str(integrity.get("orphan_sn", 0)),
        )
        li_table.add_row(
            "Error-sibling StandardNames (deterministic, no source link expected)",
            str(integrity.get("error_siblings", 0)),
        )
        li_table.add_row(
            "Orphan composed/attached source (no PRODUCED_NAME edge)",
            str(integrity.get("orphan_src", 0)),
        )
        console.print(li_table)

    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            review_totals = next(
                iter(
                    gc.query(
                        """
                        MATCH (r:StandardNameReview)
                        RETURN count(r) AS total_reviews,
                               count(r.llm_cost) AS reviews_with_cost,
                               coalesce(sum(r.llm_cost), 0.0) AS total_cost,
                               coalesce(sum(r.llm_tokens_in), 0) AS total_tokens_in,
                               coalesce(sum(r.llm_tokens_out), 0) AS total_tokens_out
                        """
                    )
                ),
                None,
            )
            cost_by_model = list(
                gc.query(
                    """
                    MATCH (r:StandardNameReview)
                    WHERE r.llm_cost IS NOT NULL
                    RETURN r.llm_model AS model,
                           count(r) AS n,
                           sum(r.llm_cost) AS cost_usd,
                           sum(r.llm_tokens_in) AS tokens_in,
                           sum(r.llm_tokens_out) AS tokens_out
                    ORDER BY cost_usd DESC
                    """
                )
            )
    except Exception as exc:  # pragma: no cover
        review_totals = None
        cost_by_model = []
        logger.debug("Could not fetch review cost: %s", exc)

    if review_totals:
        console.print()
        console.print("[bold]Review Cost[/bold]")
        rc_table = Table(show_header=True)
        rc_table.add_column("Metric")
        rc_table.add_column("Value", justify="right")
        total_reviews = review_totals.get("total_reviews", 0) or 0
        with_cost = review_totals.get("reviews_with_cost", 0) or 0
        rc_table.add_row("Review nodes", str(total_reviews))
        rc_table.add_row(
            "With cost recorded",
            f"{with_cost} ({100 * with_cost / max(total_reviews, 1):.1f}%)",
        )
        rc_table.add_row(
            "Total cost (USD)", f"${float(review_totals.get('total_cost') or 0):.4f}"
        )
        rc_table.add_row(
            "Total tokens in", str(review_totals.get("total_tokens_in") or 0)
        )
        rc_table.add_row(
            "Total tokens out", str(review_totals.get("total_tokens_out") or 0)
        )
        console.print(rc_table)

    if cost_by_model:
        console.print()
        console.print("[bold]Review Cost by Reviewer Model[/bold]")
        cm_table = Table(show_header=True)
        cm_table.add_column("Model")
        cm_table.add_column("Reviews", justify="right")
        cm_table.add_column("Cost (USD)", justify="right")
        cm_table.add_column("Tokens in", justify="right")
        cm_table.add_column("Tokens out", justify="right")
        for row in cost_by_model:
            cm_table.add_row(
                str(row.get("model") or "—"),
                str(row.get("n") or 0),
                f"${float(row.get('cost_usd') or 0):.4f}",
                str(row.get("tokens_in") or 0),
                str(row.get("tokens_out") or 0),
            )
        console.print(cm_table)


# =============================================================================
# Export / Preview / Release / Import — catalog workflow
# =============================================================================


def _run_export_cli(
    *,
    staging: str | None,
    min_score: float,
    include_unreviewed: bool,
    min_description_score: float | None,
    force: bool,
    skip_gate: bool,
    gate_only: bool,
    gate_scope: str,
    domain: str | None,
    override_edits: tuple[str, ...],
    include_sources: bool,
    names_only: bool,
) -> None:
    """Run the graph→staging export leg and render the CLI summary.

    Shared by ``sn release --export-only`` (graph → staging YAML, then
    stop — no tag/push). Reads StandardName nodes, applies quality gates
    (A/B/C/D), and writes YAML files to
    ``<staging>/standard_names/<domain>/<name>.yml``.

    Exit codes mirror the export semantics: 1 on blocking gate failure,
    2 on precondition (FileExists) error, 3 on internal error.
    """
    from pathlib import Path

    from rich.table import Table

    from imas_codex.settings import get_sn_staging_dir
    from imas_codex.standard_names.export import run_export

    staging_path = Path(staging) if staging else get_sn_staging_dir()
    staging_path.mkdir(parents=True, exist_ok=True)

    edits_list = list(override_edits) if override_edits else None

    console.print("\n[bold]Standard Name Export[/bold]")
    console.print(f"  Staging: {staging_path}")
    console.print(f"  Min score: {min_score}")
    if domain:
        console.print(f"  Domain: {domain}")
    if gate_only:
        console.print("  Mode: [yellow]gate-only (no YAML output)[/yellow]")
    if skip_gate:
        console.print("  Gates: [yellow]skipped[/yellow]")
    if not include_sources:
        console.print("  Sources: [dim]excluded (--no-include-sources)[/dim]")
    if edits_list:
        console.print(f"  Override edits: {', '.join(edits_list)}")
    console.print("")

    try:
        report = run_export(
            staging_dir=staging_path,
            min_score=min_score,
            include_unreviewed=include_unreviewed,
            min_description_score=min_description_score,
            domain=domain,
            force=force,
            skip_gate=skip_gate,
            gate_only=gate_only,
            gate_scope=gate_scope,
            override_edits=edits_list,
            include_sources=include_sources,
            names_only=names_only,
        )
    except FileExistsError as exc:
        console.print(f"[red]Precondition failure:[/red] {exc}")
        console.print("[dim]Use --force to overwrite.[/dim]")
        raise SystemExit(2) from exc
    except Exception as exc:
        console.print(f"[red]Export error:[/red] {exc}")
        raise SystemExit(3) from exc

    # ── Summary table ──────────────────────────────────────
    table = Table(title="Export Summary")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="white")
    table.add_row("total candidates", str(report.total_candidates))
    table.add_row("exported", str(report.exported_count))
    table.add_row("excluded (below score)", str(report.excluded_below_score))
    table.add_row("excluded (unreviewed)", str(report.excluded_unreviewed))
    table.add_row("excluded (domain)", str(report.excluded_by_domain))
    console.print(table)

    # ── Gate results ───────────────────────────────────────
    if report.gate_results:
        console.print("\n[bold]Gate Results[/bold]")
        for gr in report.gate_results:
            status = "[green]PASS[/green]" if gr.passed else "[red]FAIL[/red]"
            if gr.skipped:
                status = "[dim]SKIP[/dim]"
            issues = f" ({len(gr.issues)} issue(s))" if gr.issues else ""
            console.print(f"  {gr.gate}: {status}{issues}")

    # ── Divergence entries ─────────────────────────────────
    if report.divergence_entries:
        console.print(
            f"\n[yellow]Divergence:[/yellow] {len(report.divergence_entries)} entries"
        )
        for de in report.divergence_entries[:10]:
            console.print(f"  ~ {de.name} ({de.field}): {de.detail}")
        if len(report.divergence_entries) > 10:
            console.print(f"  ... and {len(report.divergence_entries) - 10} more")

    # ── Exit code ──────────────────────────────────────────
    if not report.all_gates_passed:
        failed_gates = [g.gate for g in report.gate_results if not g.passed]
        console.print(
            f"\n[red]✗ Blocking gate failure(s):[/red] {', '.join(failed_gates)}"
        )
        raise SystemExit(1)

    console.print("\n[green]✓ Export complete[/green]")


@sn.command("preview")
@click.option(
    "--staging",
    type=click.Path(),
    default=None,
    help="Staging directory to preview (default: ~/.cache/imas-codex/staging)",
)
@click.option(
    "--export/--no-export",
    "do_export",
    default=True,
    help="Run sn export before serving (default: on). Use --no-export to serve an existing staging dir.",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port for the local preview server (default: 8000)",
)
@click.option(
    "--host",
    type=str,
    default=None,
    help=(
        "Host to bind to (default: 127.0.0.1). Pass 0.0.0.0 to expose "
        "the dev server to other machines on the network — useful when "
        "previewing over an SSH tunnel that other collaborators on the "
        "same cluster can reach."
    ),
)
def sn_preview(
    staging: str | None,
    port: int | None,
    host: str | None,
    do_export: bool,
) -> None:
    """Preview standard names via ISN catalog-site.

    \b
    Exports from graph and launches a local MkDocs dev server.
    Press Ctrl-C to stop. Use --no-export to serve an existing
    staging directory without re-exporting.

    \b
    SSH tunnel (required for remote access):
      ssh -L 8000:localhost:8000 <host>

    \b
    Examples:
      imas-codex sn preview
      imas-codex sn preview --no-export
      imas-codex sn preview --staging ./staging --no-export
      imas-codex sn preview --port 9090
    """
    from pathlib import Path

    from imas_codex.settings import get_sn_staging_dir
    from imas_codex.standard_names.preview import run_preview

    staging_path = Path(staging) if staging else get_sn_staging_dir()

    if do_export:
        from imas_codex.standard_names.export import run_export

        staging_path.mkdir(parents=True, exist_ok=True)
        console.print(f"  Exporting to [cyan]{staging_path}[/cyan]...")
        try:
            report = run_export(staging_dir=staging_path, force=True)
        except Exception as exc:
            console.print(f"[red]Export error:[/red] {exc}")
            raise SystemExit(3) from exc
        console.print(f"  Exported [green]{report.exported_count}[/green] names\n")

    catalog = staging_path / "catalog.yml"
    if not catalog.is_file():
        console.print(
            f"[red]No catalog.yml found at {staging_path}[/red]\n"
            "  Run [bold]sn export[/bold] first, or remove [bold]--no-export[/bold] flag."
        )
        raise SystemExit(2)

    console.print("\n[bold]Standard Name Preview[/bold]")
    console.print(f"  Staging: {staging_path}")
    if host:
        console.print(f"  Host: {host}")
    if port:
        console.print(f"  Port: {port}")
    console.print("")

    try:
        handle = run_preview(str(staging_path), port=port, host=host)
    except FileNotFoundError as exc:
        console.print(f"[red]Precondition failure:[/red] {exc}")
        raise SystemExit(2) from exc
    except Exception as exc:
        console.print(f"[red]Preview error:[/red] {exc}")
        raise SystemExit(3) from exc

    if handle.process is None:
        console.print(
            "[red]Could not start preview server.[/red]\n"
            "Ensure imas-standard-names is installed: "
            "uv pip install imas-standard-names"
        )
        raise SystemExit(3)

    console.print(f"  Preview URL: [link={handle.url}]{handle.url}[/link]")
    console.print("  Press [bold]Ctrl-C[/bold] to stop.\n")

    try:
        handle.process.wait()
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down preview server...[/dim]")
    finally:
        handle.stop()


@sn.command("release")
@click.argument("action", required=False, default=None)
@click.option(
    "-m",
    "--message",
    type=str,
    default=None,
    help="Release message (used for git tag annotation and commit)",
)
@click.option(
    "--bump",
    type=click.Choice(["major", "minor", "patch"], case_sensitive=False),
    default=None,
    help="Version bump type. Required when on a stable tag to start a new series.",
)
@click.option(
    "--final",
    "is_final",
    is_flag=True,
    help="Finalize current RC to stable release. Pushes to upstream by default.",
)
@click.option(
    "--remote",
    type=str,
    default=None,
    help="Git remote to push to (default: origin for RC, upstream for final)",
)
@click.option(
    "--isnc",
    type=click.Path(),
    default=None,
    help="Path to ISNC git checkout (default: auto-discover)",
)
@click.option(
    "--staging",
    type=click.Path(),
    default=None,
    help="Staging directory (default: ~/.cache/imas-codex/staging)",
)
@click.option(
    "--skip-export",
    is_flag=True,
    help="Skip auto-export (use existing staging content). For custom filtering, run 'sn release --export-only' first.",
)
@click.option(
    "--skip-gate",
    is_flag=True,
    help="Skip export quality gates (ISN validation). Use during development.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate and report without making changes",
)
@click.option(
    "--names-only",
    "names_only",
    is_flag=True,
    help="Export names without requiring accepted docs (skip docs_stage gate)",
)
@click.option(
    "--export-only",
    is_flag=True,
    help=(
        "Run only the graph→staging export leg and stop (no tag/push). "
        "Use the export-scoping flags below to control what is written; "
        "then 'sn release --skip-export -m ...' to publish the staged content."
    ),
)
@click.option(
    "--min-score",
    type=float,
    default=0.65,
    show_default=True,
    help="[export] Minimum reviewer_score_name for inclusion",
)
@click.option(
    "--include-unreviewed",
    is_flag=True,
    help="[export] Include names without a reviewer_score_name",
)
@click.option(
    "--min-description-score",
    type=float,
    default=None,
    help="[export] Secondary threshold on description sub-score",
)
@click.option(
    "--gate-only",
    is_flag=True,
    help="[export] Run quality gates and report without emitting YAML",
)
@click.option(
    "--gate-scope",
    type=click.Choice(["all", "a", "b", "c", "d"], case_sensitive=False),
    default="all",
    show_default=True,
    help="[export] Which gates to run",
)
@click.option(
    "--domain",
    type=str,
    default=None,
    help="[export] Filter export to a single physics domain",
)
@click.option(
    "--override-edits",
    type=str,
    multiple=True,
    help="[export] Name(s) to reset from catalog_edit to pipeline origin (repeatable; or 'all')",
)
@click.option(
    "--include-sources/--no-include-sources",
    default=True,
    show_default=True,
    help="[export] Populate sources field in each entry with graph provenance (debug aid)",
)
@click.option(
    "--force",
    is_flag=True,
    help="[export] Overwrite non-empty staging directory without prompting (--export-only)",
)
def sn_release(
    action: str | None,
    message: str | None,
    bump: str | None,
    is_final: bool,
    remote: str | None,
    isnc: str | None,
    staging: str | None,
    skip_export: bool,
    skip_gate: bool,
    dry_run: bool,
    names_only: bool,
    export_only: bool,
    min_score: float,
    include_unreviewed: bool,
    min_description_score: float | None,
    gate_only: bool,
    gate_scope: str,
    domain: str | None,
    override_edits: tuple[str, ...],
    include_sources: bool,
    force: bool,
) -> None:
    """Release standard names to the ISNC catalog.

    \b
    Auto-exports from graph, publishes to ISNC, tags, and pushes.
    RC releases go to origin (fork) by default; final releases
    go to upstream. The state machine follows the same pattern
    as codex and ISN releases.

    \b
    Use ACTION 'status' to show current ISNC release state:
      imas-codex sn release status

    \b
    SSH tunnel (for verifying GitHub Pages after release):
      ssh -L 8000:localhost:8000 <host>

    \b
    For custom export filtering, run 'sn release --export-only' (with the
    [export] scoping flags) to stage the YAML, then use --skip-export to
    publish that existing staging content.

    \b
    Examples:
      imas-codex sn release status
      imas-codex sn release --bump minor -m "Initial catalog release"
      imas-codex sn release -m "Fix electron_temperature docs"
      imas-codex sn release --final -m "Production release v1.0.0"
      imas-codex sn release --dry-run --bump minor -m "Test release"
      imas-codex sn release --skip-export -m "Re-release with fixes"
      imas-codex sn release --export-only                       # graph → staging only
      imas-codex sn release --export-only --domain equilibrium --gate-only
      imas-codex sn release --export-only --min-score 0.8 --force
    """
    from pathlib import Path

    from rich.table import Table

    from imas_codex.settings import get_sn_isnc_dir, get_sn_staging_dir

    # ── --export-only: run the graph→staging export leg and stop ──
    # No ISNC / tag / push — just the export, with full export-scoping
    # flags. Exit codes match the export semantics (1 gate fail, 2
    # precondition, 3 internal).
    if export_only:
        if action is not None:
            raise click.ClickException(
                "--export-only does not take an ACTION argument."
            )
        _run_export_cli(
            staging=staging,
            min_score=min_score,
            include_unreviewed=include_unreviewed,
            min_description_score=min_description_score,
            force=force,
            skip_gate=skip_gate,
            gate_only=gate_only,
            gate_scope=gate_scope,
            domain=domain,
            override_edits=override_edits,
            include_sources=include_sources,
            names_only=names_only,
        )
        return

    # ── Resolve ISNC path ─────────────────────────────────
    if isnc:
        isnc_path = Path(isnc)
    else:
        resolved = get_sn_isnc_dir()
        if resolved is None:
            console.print(
                "[red]ISNC not found.[/red] Set [bold]IMAS_CODEX_SN_ISNC[/bold] env var "
                "or clone imas-standard-names-catalog as a sibling directory."
            )
            raise SystemExit(2)
        isnc_path = resolved

    # ── Status subcommand ─────────────────────────────────
    if action == "status":
        from imas_codex.standard_names.catalog_release import get_release_status

        info = get_release_status(isnc_path)
        console.print("\n[bold]ISNC Release Status[/bold]")
        console.print(f"  Path: {info['isnc_path']}")
        console.print(f"  State: {info['state'] or '[dim]no releases yet[/dim]'}")
        if info["tag"]:
            console.print(f"  Latest tag: {info['tag']}")
            if info["commits_since"]:
                console.print(f"  Commits since: {info['commits_since']}")
        if info.get("isn_version"):
            console.print(f"  ISN dep: {info['isn_version']}")
        if info.get("remotes"):
            for name, url in info["remotes"].items():
                console.print(f"  Remote ({name}): {url}")
        pages = info.get("pages_enabled")
        if pages is not None:
            status = "[green]yes[/green]" if pages else "[red]no[/red]"
            console.print(f"  GitHub Pages: {status}")

        # Show available commands based on state
        console.print("\n[bold]Available commands:[/bold]")
        state = info["state"]
        if state is None:
            console.print("  sn release --bump minor -m 'Initial release'")
        elif state == "stable":
            console.print("  sn release --bump minor -m 'New feature release'")
            console.print("  sn release --bump patch -m 'Bug fix release'")
        else:
            console.print(f"  sn release -m 'Next RC'  (→ next RC of {info['tag']})")
            console.print("  sn release --final -m 'Finalize'  (→ stable)")
        console.print()
        return

    if action is not None:
        raise click.ClickException(
            f"Unknown action '{action}'. Only 'status' is supported."
        )

    # ── Validate message ──────────────────────────────────
    if not message:
        raise click.ClickException("Release message required: -m / --message")

    # ── Resolve staging ───────────────────────────────────
    staging_path = Path(staging) if staging else get_sn_staging_dir()

    # ── Display ───────────────────────────────────────────
    console.print("\n[bold]Standard Name Release[/bold]")
    console.print(f"  ISNC: {isnc_path}")
    console.print(f"  Staging: {staging_path}")
    if bump:
        console.print(f"  Bump: {bump}")
    if is_final:
        console.print("  Mode: [green]final release[/green]")
    if skip_export:
        console.print("  Export: [yellow]skipped[/yellow]")
    if names_only:
        console.print("  Export: [cyan]names-only (skip docs gate)[/cyan]")
    if dry_run:
        console.print("  Mode: [yellow]dry run[/yellow]")
    console.print("")

    # ── Run release ───────────────────────────────────────
    from imas_codex.standard_names.catalog_release import run_release

    try:
        export_kwargs = {}
        if skip_gate:
            export_kwargs["skip_gate"] = True
        if names_only:
            export_kwargs["names_only"] = True

        report = run_release(
            isnc_path=isnc_path,
            message=message,
            staging_dir=staging_path,
            bump=bump,
            final=is_final,
            remote=remote,
            dry_run=dry_run,
            skip_export=skip_export,
            export_kwargs=export_kwargs or None,
        )
    except Exception as exc:
        console.print(f"[red]Release error:[/red] {exc}")
        raise SystemExit(3) from exc

    # ── Errors ────────────────────────────────────────────
    if report.errors:
        console.print(f"[red]Errors: {len(report.errors)}[/red]")
        for err in report.errors[:10]:
            console.print(f"  - {err}")
        if len(report.errors) > 10:
            console.print(f"  ... and {len(report.errors) - 10} more")
        raise SystemExit(2)

    # ── Summary ───────────────────────────────────────────
    table = Table(title="Release Summary")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="white")
    table.add_row("version", report.version)
    table.add_row("git tag", report.git_tag)
    table.add_row("remote", report.remote)
    table.add_row("exported", str(report.export_count))
    table.add_row("files copied", str(report.files_copied))
    table.add_row("commit SHA", report.commit_sha or "(no changes)")
    table.add_row("pushed", "yes" if report.pushed else "no")
    table.add_row("dry run", "yes" if report.dry_run else "no")
    console.print(table)

    if report.pushed:
        console.print(f"\n[green]✓ Released {report.git_tag} → {report.remote}[/green]")
    elif report.dry_run:
        console.print(
            f"\n[yellow]✓ Dry run complete — would release {report.git_tag}[/yellow]"
        )
    else:
        console.print(f"\n[green]✓ Tagged {report.git_tag} (not pushed)[/green]")


@sn.command("import")
@click.option(
    "--isnc",
    type=click.Path(),
    default=None,
    help="Path to ISNC repository root (default: auto-discover)",
)
@click.option(
    "--accept-unit-override",
    is_flag=True,
    help="Accept unit mismatches against DD values",
)
@click.option(
    "--dry-run", is_flag=True, help="Parse and validate without writing to graph"
)
def sn_import(
    isnc: str | None,
    accept_unit_override: bool,
    dry_run: bool,
) -> None:
    """Import reviewed catalog entries from ISNC into the graph.

    \b
    Reads YAML files from the ISNC standard_names/ subtree, validates
    them, derives grammar fields, applies diff-based origin tracking,
    and MERGEs into the graph with name_stage='accepted'.

    \b
    Examples:
      imas-codex sn import
      imas-codex sn import --isnc ../imas-standard-names-catalog
      imas-codex sn import --dry-run
    """
    from pathlib import Path

    from rich.table import Table

    from imas_codex.settings import get_sn_isnc_dir
    from imas_codex.standard_names.catalog_import import run_import

    if isnc:
        isnc_path = Path(isnc)
    else:
        resolved = get_sn_isnc_dir()
        if resolved is None:
            console.print(
                "[red]ISNC not found.[/red] Set [bold]IMAS_CODEX_SN_ISNC[/bold] env var "
                "or clone imas-standard-names-catalog as a sibling directory."
            )
            raise SystemExit(2)
        isnc_path = resolved

    console.print("\n[bold]Standard Name Import[/bold]")
    console.print(f"  ISNC: {isnc_path}")
    if accept_unit_override:
        console.print("  Unit override: [yellow]accepted[/yellow]")
    if dry_run:
        console.print("  Mode: [yellow]dry run[/yellow]")
    console.print("")

    try:
        report = run_import(
            catalog_dir=isnc_path,
            dry_run=dry_run,
            accept_unit_override=accept_unit_override,
        )
    except Exception as exc:
        console.print(f"[red]Import error:[/red] {exc}")
        raise SystemExit(3) from exc

    # ── Errors ─────────────────────────────────────────────
    if report.errors:
        console.print(f"[red]Errors: {len(report.errors)}[/red]")
        for err in report.errors[:10]:
            console.print(f"  - {err}")
        if len(report.errors) > 10:
            console.print(f"  ... and {len(report.errors) - 10} more")

    # ── Summary table ──────────────────────────────────────
    action = "Would import" if dry_run else "Imported"
    table = Table(title=f"Import Summary ({action})")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="white")
    table.add_row("imported", str(report.imported))
    table.add_row("created", str(report.created))
    table.add_row("updated", str(report.updated))
    table.add_row("skipped", str(report.skipped))
    table.add_row("errors", str(len(report.errors)))
    if report.catalog_commit_sha:
        table.add_row("catalog SHA", report.catalog_commit_sha[:12])
    if report.pr_numbers:
        table.add_row("PR numbers", ", ".join(f"#{n}" for n in report.pr_numbers))
    table.add_row("watermark advanced", "yes" if report.watermark_advanced else "no")
    console.print(table)

    if dry_run and report.entries:
        console.print("\n[bold]Preview:[/bold]")
        for entry in report.entries[:20]:
            units = f" [{entry.get('unit', '')}]" if entry.get("unit") else ""
            console.print(f"  - {entry.get('id', '?')}{units}")
        if len(report.entries) > 20:
            console.print(f"  ... and {len(report.entries) - 20} more")

    if report.errors and not dry_run:
        raise SystemExit(2)

    console.print(f"\n[green]✓ {action}: {report.imported} entries[/green]")


@sn.command("clear")
@click.option("--dry-run", is_flag=True, help="Preview without modifying the graph")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--no-comment-export",
    "no_comment_export",
    is_flag=True,
    default=False,
    help="Skip the pre-clear Review comment export to JSONL.",
)
@click.option(
    "--no-reseed",
    "no_reseed",
    is_flag=True,
    default=False,
    help="Skip the post-clear ISN grammar re-seed.",
)
def sn_clear(
    dry_run: bool, force: bool, no_comment_export: bool, no_reseed: bool
) -> None:
    """Wipe every Standard Name the pipeline has produced.

    Deletes the six pipeline-output labels: StandardName, StandardNameReview,
    StandardNameSource, VocabGap, SNRun, LLMCost. ISN grammar nodes
    (GrammarToken, GrammarSegment, GrammarTemplate, ISNGrammarVersion)
    are ISN-authoritative reference data and are automatically re-seeded
    from the installed ``imas_standard_names`` package after the wipe.
    Pass ``--no-reseed`` to leave the grammar nodes untouched.

    For scoped deletes (by status, source, IDS, score tier, …) use
    ``sn prune`` instead.

    Before deleting, any existing StandardNameReview nodes are exported to a JSONL
    file in ``research/`` so reviewer feedback survives across clear
    cycles.  Pass ``--no-comment-export`` to skip the dump (e.g. in
    automated tests).

    \b
    Examples:
      imas-codex sn clear --dry-run    # Preview the wipe
      imas-codex sn clear --force      # Full wipe + grammar re-seed
      imas-codex sn clear --force --no-reseed  # Wipe without re-seeding grammar
    """
    import datetime

    from imas_codex.standard_names.graph_ops import clear_sn_subsystem

    try:
        preview = clear_sn_subsystem(dry_run=True)
        total = sum(preview.values())
        if total == 0:
            console.print("No SN pipeline nodes to delete.")
            return

        console.print("[bold]SN pipeline wipe preview:[/bold]")
        for label, n in preview.items():
            if n:
                console.print(f"  {label}: {n}")
        console.print(f"[bold]Total: {total}[/bold]")

        if dry_run:
            return

        if not force:
            click.confirm(
                f"This will delete {total} SN pipeline nodes. Continue?",
                abort=True,
            )

        # Pre-clear comment export (Phase F)
        if not no_comment_export and preview.get("StandardNameReview", 0) > 0:
            import pathlib

            from imas_codex.standard_names.graph_ops import export_review_comments

            ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            export_dir = pathlib.Path("research")
            export_path = export_dir / f"comments-{ts}.jsonl"
            try:
                n_exported = export_review_comments(export_path)
                if n_exported:
                    console.print(
                        f"[dim]Exported {n_exported} StandardNameReview records → {export_path}[/dim]"
                    )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Comment export skipped: {exc}[/yellow]")

        deleted = clear_sn_subsystem(dry_run=False)
        total_deleted = sum(deleted.values())
        console.print(f"[green]Deleted {total_deleted} nodes[/green]")
        for label, n in deleted.items():
            if n:
                console.print(f"  {label}: {n}")

        # Re-seed the ISN grammar so the graph is immediately usable for a
        # fresh pipeline run. Best-effort — a sync failure is logged and the
        # clear still succeeds.
        if not no_reseed:
            from imas_codex.standard_names.grammar_sync import (
                sync_isn_grammar_to_graph,
            )

            try:
                report = sync_isn_grammar_to_graph()
                console.print(
                    f"[green]Re-seeded ISN grammar → {report.isn_version}"
                    f" ({report.segments} segments, {report.templates} templates)"
                    f"[/green]"
                )
            except Exception as exc:  # noqa: BLE001 — clear already succeeded
                console.print(f"[yellow]Grammar re-seed skipped: {exc}[/yellow]")
    except click.Abort:
        raise
    except Exception as e:
        console.print(f"[red]Clear error:[/red] {e}")
        raise SystemExit(1) from e


@sn.command("prune")
@click.option(
    "--stage",
    default=None,
    help="Delete names with this name_stage (e.g. drafted)",
)
@click.option(
    "--all",
    "prune_all",
    is_flag=True,
    help="Delete all standard names (still respects --include-accepted)",
)
@click.option(
    "--source",
    type=click.Choice(["dd", "signals"]),
    default=None,
    help="Filter by source ('dd' or 'signals')",
)
@click.option("--ids", "ids_filter", default=None, help="Filter to specific IDS")
@click.option(
    "--include-accepted",
    is_flag=True,
    help="Also delete accepted names (dangerous — use with care)",
)
@click.option("--dry-run", is_flag=True, help="Preview without modifying the graph")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--include-sources",
    is_flag=True,
    default=False,
    help="Also delete StandardNameSource nodes matching the same scope",
)
def sn_prune(
    stage: str | None,
    prune_all: bool,
    source: str | None,
    ids_filter: str | None,
    include_accepted: bool,
    dry_run: bool,
    force: bool,
    include_sources: bool,
) -> None:
    """Delete a subset of StandardName nodes (scoped by filters).

    Relationship-first safety model: HAS_STANDARD_NAME edges are removed
    before deleting nodes; scoped deletes only remove orphaned nodes.
    Review nodes attached to pruned StandardNames are deleted alongside
    them; a final sweep removes any orphan StandardNameReview nodes left by prior
    runs.

    Use this for targeted cleanup while iterating on generation. For a
    full subsystem wipe (all nodes + grammar re-seed), use ``sn clear``.

    \b
    Examples:
      imas-codex sn prune --stage drafted
      imas-codex sn prune --all --source dd --ids equilibrium
      imas-codex sn prune --all --include-accepted --dry-run
    """
    if not stage and not prune_all:
        raise click.UsageError("Provide --stage <value> or --all to select names.")

    stage_filter = None if prune_all else ([stage] if stage else None)

    from imas_codex.standard_names.graph_ops import clear_standard_names

    try:
        # Always preview first
        count = clear_standard_names(
            stage_filter=stage_filter,
            source_filter=source,
            ids_filter=ids_filter,
            include_accepted=include_accepted,
            dry_run=True,
        )

        if count == 0:
            console.print("No matching StandardName nodes to delete.")
            return

        # Build scope description for the confirmation message
        scope_parts: list[str] = []
        if stage:
            scope_parts.append(f"stage={stage}")
        if source:
            scope_parts.append(f"source={source}")
        if ids_filter:
            scope_parts.append(f"ids={ids_filter}")
        if include_accepted:
            scope_parts.append("including accepted")
        scope = f" ({', '.join(scope_parts)})" if scope_parts else ""

        if dry_run:
            console.print(f"Would delete {count} StandardName node(s){scope}")
            return

        if not force:
            click.confirm(
                f"This will delete {count} StandardName node(s){scope}. Continue?",
                abort=True,
            )

        deleted = clear_standard_names(
            stage_filter=stage_filter,
            source_filter=source,
            ids_filter=ids_filter,
            include_accepted=include_accepted,
            dry_run=False,
        )
        console.print(f"Deleted {deleted} StandardName node(s)")

        if include_sources:
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                sns_where_clauses = []
                sns_params: dict = {}
                if source:
                    sns_where_clauses.append("sns.source_type = $source_type")
                    sns_params["source_type"] = source
                if ids_filter:
                    sns_where_clauses.append("sns.ids_name = $ids_filter")
                    sns_params["ids_filter"] = ids_filter
                where_clause = (
                    "WHERE " + " AND ".join(sns_where_clauses)
                    if sns_where_clauses
                    else ""
                )
                count_result = gc.query(
                    f"MATCH (sns:StandardNameSource) {where_clause} RETURN count(sns) AS count",
                    **sns_params,
                )
                sns_count = count_result[0]["count"] if count_result else 0
                if sns_count > 0:
                    gc.query(
                        f"MATCH (sns:StandardNameSource) {where_clause} DETACH DELETE sns",
                        **sns_params,
                    )
                    console.print(f"  Deleted {sns_count} StandardNameSource nodes")

    except click.Abort:
        raise
    except Exception as e:
        console.print(f"[red]Prune error:[/red] {e}")
        raise SystemExit(1) from e


@sn.command("review")
@click.option("--ids", default=None, help="Scope to names linked to specific IDS")
@click.option(
    "--physics-domain",
    "domain",
    type=_PHYSICS_DOMAIN_CHOICE,
    default=None,
    help="Scope to physics domain.",
)
@click.option(
    "--stage",
    "stage_filter",
    default="drafted",
    help="Filter by name_stage (default: drafted)",
)
@click.option(
    "--unreviewed",
    is_flag=True,
    help="Only names with no reviewer_score_name or stale review",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force re-review of already-scored names",
)
@click.option(
    "--models",
    "models_override",
    default=None,
    help=(
        "Ad-hoc override for the reviewer list as comma-separated model "
        "ids. Overrides [sn.review.names].models or [sn.review.docs].models "
        "depending on --target."
    ),
)
@click.option(
    "--batch-size",
    type=int,
    default=15,
    help="Max names per batch (hard cap: 25)",
)
@click.option(
    "--neighborhood",
    type=int,
    default=10,
    help="Similar names for context",
)
@click.option(
    "-c",
    "--cost-limit",
    type=float,
    default=5.0,
    help="Max LLM spend in USD",
)
@click.option(
    "--dry-run", is_flag=True, help="Run Layer 1 audits, show batch plan, no LLM calls"
)
@click.option("--skip-audit", is_flag=True, help="Skip Layer 1 audits (debug only)")
@click.option("--concurrency", type=int, default=8, help="Parallel review batches")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--target",
    type=click.Choice(["names", "docs"], case_sensitive=False),
    default="names",
    show_default=True,
    help=(
        "Which review rubric to apply. 'names' → 4-dim name rubric "
        "(grammar/semantic/convention/completeness, /80). 'docs' → 4-dim "
        "docs rubric (description_quality/documentation_quality/"
        "completeness/physics_accuracy, /80). A lower-fidelity target will "
        "not overwrite a higher-fidelity prior review unless --force."
    ),
)
@click.option(
    "--reviewer-profile",
    "reviewer_profile",
    type=click.Choice(
        ["default", "pilot", "opus-only", "haiku-only"], case_sensitive=False
    ),
    default="default",
    show_default=True,
    envvar="IMAS_CODEX_SN_REVIEW_PROFILE",
    help=(
        "Reviewer model chain profile. "
        "'default' → Opus+GPT-5.4+Sonnet (3-model RD-quorum, $0.027/name). "
        "'pilot' → Haiku×2+Opus arbiter ($0.004/name, ~85%% cost reduction). "
        "'opus-only' → single Opus reviewer (no quorum). "
        "'haiku-only' → single Haiku reviewer (cheapest). "
        "Overridden by --models if both are specified. "
        "Also read from IMAS_CODEX_SN_REVIEW_PROFILE env var."
    ),
)
def sn_review(
    ids: str | None,
    domain: str | None,
    stage_filter: str,
    unreviewed: bool,
    force: bool,
    models_override: str | None,
    batch_size: int,
    neighborhood: int,
    cost_limit: float,
    dry_run: bool,
    skip_audit: bool,
    concurrency: int,
    verbose: bool,
    target: str,
    reviewer_profile: str,
) -> None:
    """Review standard names with 3-layer pipeline.

    \b
    Layer 1: Deterministic audits (embedding, lint, links, duplicates)
    Layer 2: Batched LLM quality scoring with neighborhood context
    Layer 3: Cross-batch consolidation and summary report

    \b
    Examples:
      imas-codex sn review --unreviewed --cost-limit 5.0
      imas-codex sn review --ids equilibrium --dry-run
      imas-codex sn review --force --physics-domain magnetics
      imas-codex sn review --target names --unreviewed
      imas-codex sn review --target docs --physics-domain equilibrium
      imas-codex sn review --reviewer-profile pilot --unreviewed -c 2.0
    """
    import asyncio

    from imas_codex.cli.discover.common import setup_logging
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.review.state import StandardNameReviewState

    setup_logging("sn", "sn-review", use_rich=False, verbose=verbose)

    # Resolve --target.
    target_normalized = target.lower()
    # Downstream state uses a derived name_only boolean.
    name_only = target_normalized == "names"

    # Enforce batch-size cap
    batch_size = min(batch_size, 25)

    # Load reviewer list (N>=1). Priority: --models > --reviewer-profile > config.
    from imas_codex.settings import (
        get_sn_review_disagreement_threshold,
        get_sn_review_docs_models,
        get_sn_review_names_models,
        get_sn_review_profile_models,
        get_sn_review_profile_threshold,
    )

    reviewer_profile = reviewer_profile.lower()
    if models_override:
        # Ad-hoc --models takes precedence over profile.
        review_models = [m.strip() for m in models_override.split(",") if m.strip()]
        disagreement_threshold = get_sn_review_disagreement_threshold()
    elif reviewer_profile != "default":
        # Explicit non-default profile selected.
        review_models = get_sn_review_profile_models(reviewer_profile)
        disagreement_threshold = get_sn_review_profile_threshold(reviewer_profile)
    elif target_normalized == "names":
        review_models = get_sn_review_names_models()
        disagreement_threshold = get_sn_review_disagreement_threshold()
    else:
        review_models = get_sn_review_docs_models()
        disagreement_threshold = get_sn_review_disagreement_threshold()

    # Build state
    state = StandardNameReviewState(
        facility="dd",
        cost_limit=cost_limit,
        ids_filter=ids,
        domain_filter=domain,
        stage_filter=stage_filter,
        unreviewed_only=unreviewed,
        force_review=force,
        skip_audit=skip_audit,
        review_model=(review_models[0] if review_models else None),
        batch_size=batch_size,
        neighborhood_k=neighborhood,
        concurrency=concurrency,
        dry_run=dry_run,
        name_only=name_only,
        target=target_normalized,
        budget_manager=BudgetManager(cost_limit),
        review_models=review_models,
        disagreement_threshold=disagreement_threshold,
    )

    async def _run() -> None:
        # Layer 1: Audits (on full catalog, unless --skip-audit)
        if not skip_audit:
            console.print(
                "[bold]Layer 1:[/bold] Running deterministic audits on full catalog…"
            )
            from imas_codex.graph.client import GraphClient

            def _load_catalog() -> list[dict]:
                with GraphClient() as gc:
                    rows = gc.query(
                        """
                        MATCH (sn:StandardName)
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        RETURN sn.id AS id, sn.description AS description,
                               sn.documentation AS documentation,
                               sn.kind AS kind,
                               coalesce(u.id, sn.unit) AS unit,
                               sn.tags AS tags, sn.links AS links,
                               sn.source_paths AS source_paths,
                               sn.physical_base AS physical_base,
                               sn.subject AS subject,
                               sn.component AS component,
                               sn.coordinate AS coordinate,
                               sn.position AS position,
                               sn.process AS process,
                               sn.cocos_transformation_type AS cocos_transformation_type,
                               sn.physics_domain AS physics_domain,
                               sn.name_stage AS name_stage,
                               sn.reviewer_score_name AS reviewer_score,
                               sn.review_input_hash AS review_input_hash,
                               sn.embedding AS embedding,
                               sn.review_tier AS review_tier,
                               sn.link_status AS link_status,
                               sn.source_types AS source_types,
                               sn.geometric_base AS geometric_base
                        """
                    )
                    return [dict(r) for r in rows] if rows else []

            all_names = await asyncio.to_thread(_load_catalog)

            if not all_names:
                console.print("[yellow]No standard names found in graph[/yellow]")
                return

            console.print(f"  Loaded {len(all_names)} standard names")

            from imas_codex.standard_names.review.audits import run_all_audits

            state.audit_report = await asyncio.to_thread(run_all_audits, all_names)
            state.all_names = all_names

            # Print audit summary
            ar = state.audit_report
            console.print(
                f"  Embeddings: {ar.embedding.missing_count} missing, "
                f"{ar.embedding.stale_count} stale, "
                f"{ar.embedding.refreshed_count} refreshed"
            )
            console.print(f"  Lint findings: {len(ar.lint_findings)}")
            console.print(f"  Link issues: {len(ar.link_findings)}")
            console.print(f"  Duplicate components: {len(ar.duplicate_components)}")

        if dry_run:
            # In dry-run mode, show batch plan but don't run LLM
            console.print("\n[bold]Dry run:[/bold] Showing batch plan (no LLM calls)")

            from imas_codex.graph.client import GraphClient
            from imas_codex.standard_names.review.enrichment import (
                group_into_review_batches,
                reconstruct_clusters_batch,
            )

            # Apply filters to get target names
            targets = list(state.all_names) if state.all_names else []
            if stage_filter:
                targets = [n for n in targets if n.get("name_stage") == stage_filter]
            if ids:
                targets = [
                    n
                    for n in targets
                    if any(
                        p.startswith(ids + "/") for p in (n.get("source_paths") or [])
                    )
                ]
            if domain:
                targets = [n for n in targets if n.get("physics_domain") == domain]
            if unreviewed:
                from imas_codex.standard_names.review.audits import (
                    compute_review_input_hash,
                )

                targets = [
                    n
                    for n in targets
                    if n.get("reviewer_score_name") is None
                    or n.get("review_input_hash") != compute_review_input_hash(n)
                ]

            console.print(f"  Targets for review: {len(targets)} names")

            if targets:
                try:

                    def _get_clusters() -> dict:
                        with GraphClient() as gc:
                            return reconstruct_clusters_batch(targets, gc)

                    clusters = await asyncio.to_thread(_get_clusters)
                    batches = group_into_review_batches(
                        targets,
                        clusters,
                        max_batch_size=batch_size,
                    )
                    console.print(f"  Would create {len(batches)} review batches:")
                    for i, b in enumerate(batches[:10]):
                        n_names = len(b.get("names", []))
                        tokens = b.get("estimated_tokens", 0)
                        console.print(
                            f"    Batch {i + 1}: {n_names} names, ~{tokens} tokens"
                            f" — {b.get('group_key', 'unknown')}"
                        )
                    if len(batches) > 10:
                        console.print(f"    … and {len(batches) - 10} more batches")
                except Exception as exc:
                    console.print(
                        f"  [yellow]Could not compute batch plan: {exc}[/yellow]"
                    )
            return

        # Layer 2: Batched LLM Review
        console.print("\n[bold]Layer 2:[/bold] Running batched LLM review…")

        from imas_codex.standard_names.review.pipeline import run_sn_review_engine

        stop_event = asyncio.Event()
        await run_sn_review_engine(state, stop_event=stop_event)

        # Layer 3: Consolidation
        console.print("\n[bold]Layer 3:[/bold] Running cross-batch consolidation…")
        from imas_codex.standard_names.review.consolidation import run_consolidation

        summary = run_consolidation(state)

        # Print summary report
        console.print("\n[bold]═══ Review Summary ═══[/bold]")
        scored_info = f"  Scored: {summary.total_scored} / {summary.total_catalog_size}"
        scored_info += f" names ({summary.coverage_pct:.1f}%)"
        if summary.total_unscored > 0:
            scored_info += f"  [yellow]({summary.total_unscored} unscored)[/yellow]"
        console.print(scored_info)
        console.print(f"  LLM cost: ${summary.total_cost:.4f}")

        if summary.tier_distribution:
            tier_str = ", ".join(
                f"{t}: {c}" for t, c in sorted(summary.tier_distribution.items())
            )
            console.print(f"  Tier distribution: {tier_str}")

        if summary.duplicate_candidates:
            console.print(
                f"  Duplicate candidates: {len(summary.duplicate_candidates)}"
            )
            for dc in summary.duplicate_candidates[:3]:
                console.print(f"    {dc.names} (sim={dc.max_similarity:.3f})")

        if summary.drift_warnings:
            console.print(f"  Convention drift warnings: {len(summary.drift_warnings)}")
            for dw in summary.drift_warnings[:3]:
                console.print(
                    f"    [{dw.physics_domain}] {dw.drift_type}: {dw.detail[:80]}"
                )

        if summary.outliers:
            console.print(f"  Score outliers: {len(summary.outliers)}")
            for ol in summary.outliers[:5]:
                console.print(
                    f"    {ol.name_id}: {ol.score:.2f}"
                    f" (z={ol.z_score:.1f}, {ol.recommendation})"
                )

        if summary.lowest_scorers:
            console.print("  Lowest scorers:")
            for ls in summary.lowest_scorers[:5]:
                console.print(
                    f"    {ls.get('id', '?')}: {ls.get('reviewer_score_name', 0):.2f}"
                    f" ({ls.get('review_tier', '?')})"
                )

        # Budget summary
        if state.budget_manager:
            bs = state.budget_manager.summary
            console.print(
                f"  Budget: ${bs['total_spent']:.4f} used of"
                f" ${bs['total_budget']:.2f} ({bs['batch_count']} batches)"
            )

    asyncio.run(_run())

"""Per-pool streaming display for the 6-pool SN pipeline.

Six pools rendered individually using the canonical ``BaseProgressDisplay``
layout (full-width panel, per-worker streaming via ``PipelineRowConfig``):

    DRAFT         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━────────      720  36%
    electron_temperature_in_core_plasma                          1.5/s
    → e_temp_core                                                $1.20
    REVIEW NAME   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━────────      640  32%
    e_temp_core                                                  2.1/s
    0.83  "Good grammar; documentation could be…"                $0.85
    ...
    ─────────────────────────────────────────────────────────────
    TIME  ━━━━━━━━━━━━────────────────  18m 32s  ETA 2h 14m
    COST  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  $4.55  ETC $9.20

Inherits ``BaseProgressDisplay`` for full-width rendering, canonical
HEADER → SERVERS → PIPELINE → RESOURCES layout, and service-monitor
integration.  Per-item streaming uses ``PipelineRowConfig`` (3-line
per pool) with the standard ``build_pipeline_section`` renderer.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text

from imas_codex.discovery.base.progress import (
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    WorkerStats,
    build_pipeline_section,
    build_resource_section,
    compute_parallel_eta,
)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

#: Pool names in display order (all 7 internal pools).
POOL_ORDER: tuple[str, ...] = (
    "generate_name",
    "enrich_parents",
    "review_name",
    "refine_name",
    "generate_docs",
    "review_docs",
    "refine_docs",
)

#: Display rows — each maps to one or more internal pools.
#: The display merges 7 pools into 4 visual rows.
DISPLAY_ROWS: tuple[str, ...] = (
    "generate_name",  # merges generate_name + refine_name + enrich_parents
    "review_name",  # review_name only
    "generate_docs",  # merges generate_docs + refine_docs
    "review_docs",  # review_docs only
)

#: Mapping: display row → internal pools that feed it.
#: ``enrich_parents`` (derived-parent description synthesis) is a name-axis
#: producer like generate_name/refine_name, so it folds into the GENERATE NAME
#: row rather than claiming its own row.
DISPLAY_POOL_MAP: dict[str, tuple[str, ...]] = {
    "generate_name": ("generate_name", "refine_name", "enrich_parents"),
    "review_name": ("review_name",),
    "generate_docs": ("generate_docs", "refine_docs"),
    "review_docs": ("review_docs",),
}

#: Display labels for the 4 merged rows.
DISPLAY_LABELS: dict[str, str] = {
    "generate_name": "GENERATE NAME",
    "review_name": "REVIEW NAME",
    "generate_docs": "GENERATE DOCS",
    "review_docs": "REVIEW DOCS",
}

#: Rich styles for the 4 display rows.
DISPLAY_STYLES: dict[str, str] = {
    "generate_name": "bold magenta",
    "review_name": "bold yellow",
    "generate_docs": "bold cyan",
    "review_docs": "bold yellow",
}

#: Display labels — short labels that fit the canonical LABEL_WIDTH (12).
POOL_LABELS: dict[str, str] = {
    "generate_name": "DRAFT NAME",
    "enrich_parents": "PARENT DESC",
    "review_name": "REVIEW NAME",
    "refine_name": "REFINE NAME",
    "generate_docs": "DRAFT DOCS",
    "review_docs": "REVIEW DOCS",
    "refine_docs": "REFINE DOCS",
}

#: Legacy long labels (upper-case, underscore-separated).
#: Used by legacy :func:`render_pool_panel` for backward compat.
_LEGACY_POOL_LABELS: dict[str, str] = {
    "generate_name": "GENERATE_NAME",
    "enrich_parents": "ENRICH_PARENTS",
    "review_name": "REVIEW_NAME",
    "refine_name": "REFINE_NAME",
    "generate_docs": "GENERATE_DOCS",
    "review_docs": "REVIEW_DOCS",
    "refine_docs": "REFINE_DOCS",
}

#: Rich styles per pool.
POOL_STYLES: dict[str, str] = {
    "generate_name": "bold magenta",
    "enrich_parents": "magenta",
    "review_name": "bold yellow",
    "refine_name": "magenta",
    "generate_docs": "bold cyan",
    "review_docs": "bold yellow",
    "refine_docs": "cyan",
}

#: Maximum streamed items kept per pool (legacy compat).
STREAM_MAXLEN = 3

#: Review-score color thresholds.
SCORE_GREEN_THRESHOLD = 0.85
SCORE_YELLOW_THRESHOLD = 0.65


# ═══════════════════════════════════════════════════════════════════════
# Event → PipelineRowConfig field mapping (per-pool)
# ═══════════════════════════════════════════════════════════════════════


def _map_generate_name(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a generate_name event to PipelineRowConfig stream fields."""
    source = ev.get("source", ev.get("dd_path", ""))
    name = ev.get("name", "")
    return {
        "primary_text": str(source),
        "primary_text_style": "dim",
        "description": f"→ {name}" if name else "",
    }


def _map_enrich_parents(ev: dict[str, Any]) -> dict[str, Any]:
    """Map an enrich_parents event to PipelineRowConfig stream fields."""
    name = str(ev.get("name", ""))
    desc = str(ev.get("description", ""))
    return {
        "primary_text": name,
        "primary_text_style": "magenta",
        "description": f"⌁ {desc[:80]}" if desc else "⌁ enriched parent",
    }


def _map_review_name(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a review_name event to PipelineRowConfig stream fields."""
    name = str(ev.get("name", ""))
    raw_score = ev.get("score")
    sc = float(raw_score) if raw_score is not None else None
    comment = str(ev.get("comment", ""))
    return {
        "primary_text": name,
        "primary_text_style": "white",
        "score_value": sc,
        "description": f'"{comment}"' if comment else "",
    }


def _map_refine_name(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a refine_name event to PipelineRowConfig stream fields."""
    old = str(ev.get("old_name", ""))
    chain = int(ev.get("chain_length", 0))
    escalated = bool(ev.get("escalated", False))
    new = str(ev.get("new_name", ""))
    model = str(ev.get("model", ""))
    if escalated:
        desc = f"(chain={chain}) → escalating to {model}"
    else:
        desc = f"(chain={chain}) → {new}"
    return {
        "primary_text": old,
        "primary_text_style": "white",
        "description": desc,
    }


def _map_generate_docs(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a generate_docs event to PipelineRowConfig stream fields."""
    name = str(ev.get("name", ""))
    desc = str(ev.get("description", ""))
    return {
        "primary_text": name,
        "primary_text_style": "white",
        "description": f'"{desc}"' if desc else "",
    }


def _map_review_docs(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a review_docs event to PipelineRowConfig stream fields."""
    return _map_review_name(ev)


def _map_refine_docs(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a refine_docs event to PipelineRowConfig stream fields."""
    name = str(ev.get("name", ""))
    rev = int(ev.get("revision", 0))
    desc = str(ev.get("description", ""))
    return {
        "primary_text": name,
        "primary_text_style": "white",
        "description": f'(rev={rev}) "{desc}"' if desc else f"(rev={rev})",
    }


def _map_embed_name(ev: dict[str, Any]) -> dict[str, Any]:
    """Map an embed_name event to PipelineRowConfig stream fields."""
    name = str(ev.get("name", ""))
    return {
        "primary_text": name,
        "primary_text_style": "green",
        "description": "✓ embedded",
    }


#: Registry mapping pool name → event-to-stream-fields mapper.
_EVENT_MAPPERS: dict[str, Any] = {
    "generate_name": _map_generate_name,
    "enrich_parents": _map_enrich_parents,
    "review_name": _map_review_name,
    "refine_name": _map_refine_name,
    "generate_docs": _map_generate_docs,
    "review_docs": _map_review_docs,
    "refine_docs": _map_refine_docs,
}

# ═══════════════════════════════════════════════════════════════════════
# Legacy per-item renderers (kept for backward compat / test imports)
# ═══════════════════════════════════════════════════════════════════════


def score_color(score: float) -> str:
    """Return Rich style name for a reviewer score.

    ≥ 0.85 → green, 0.65–0.85 → yellow, < 0.65 → red.
    """
    if score >= SCORE_GREEN_THRESHOLD:
        return "green"
    if score >= SCORE_YELLOW_THRESHOLD:
        return "yellow"
    return "red"


def _clip(text: str, maxlen: int) -> str:
    """Clip text with ellipsis if exceeding *maxlen*."""
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 1] + "…"


def format_item_generate_name(item: dict[str, Any]) -> Text:
    """Render a GENERATE_NAME per-item line."""
    source = item.get("source", item.get("dd_path", ""))
    name = item.get("name", "")
    line = Text("    ")
    line.append(_clip(str(source), 40), style="dim")
    line.append("  →  ", style="white")
    line.append(str(name), style="bold white")
    return line


def format_item_review_name(item: dict[str, Any]) -> Text:
    """Render a REVIEW_NAME per-item line."""
    name = str(item.get("name", ""))
    raw_score = item.get("score", 0.0)
    sc = float(raw_score) if raw_score is not None else 0.0
    comment = str(item.get("comment", ""))
    line = Text("    ")
    line.append(_clip(name, 30), style="white")
    line.append("  ")
    line.append(f"{sc:.2f}", style=score_color(sc))
    if comment:
        line.append(f'  "{_clip(comment, 80)}"', style="dim")
    return line


def format_item_refine_name(item: dict[str, Any]) -> Text:
    """Render a REFINE_NAME per-item line."""
    old = str(item.get("old_name", ""))
    chain = int(item.get("chain_length", 0))
    escalated = bool(item.get("escalated", False))
    new = str(item.get("new_name", ""))
    model = str(item.get("model", ""))
    line = Text("    ")
    line.append(_clip(old, 30), style="white")
    line.append(f" (chain={chain})", style="dim")
    if escalated:
        line.append(f" → escalating to {model}", style="bold red")
    else:
        line.append(" → ", style="white")
        line.append(new, style="bold white")
    return line


def format_item_generate_docs(item: dict[str, Any]) -> Text:
    """Render a GENERATE_DOCS per-item line."""
    name = str(item.get("name", ""))
    desc = str(item.get("description", ""))
    line = Text("    ")
    line.append(_clip(name, 30), style="white")
    if desc:
        line.append(f'  "{_clip(desc, 100)}"', style="dim")
    return line


def format_item_review_docs(item: dict[str, Any]) -> Text:
    """Render a REVIEW_DOCS per-item line."""
    return format_item_review_name(item)


def format_item_refine_docs(item: dict[str, Any]) -> Text:
    """Render a REFINE_DOCS per-item line."""
    name = str(item.get("name", ""))
    rev = int(item.get("revision", 0))
    desc = str(item.get("description", ""))
    line = Text("    ")
    line.append(_clip(name, 30), style="white")
    line.append(f" (rev={rev})", style="dim")
    if desc:
        line.append(f'  "{_clip(desc, 100)}"', style="dim")
    return line


def format_item_embed_name(item: dict[str, Any]) -> Text:
    """Render an EMBED_NAME per-item line."""
    name = str(item.get("name", ""))
    line = Text("    ")
    line.append(_clip(name, 40), style="green")
    line.append("  ✓ embedded", style="dim")
    return line


#: Registry of per-item formatters by pool name (legacy).
ITEM_FORMATTERS: dict[str, Any] = {
    "generate_name": format_item_generate_name,
    "review_name": format_item_review_name,
    "refine_name": format_item_refine_name,
    "generate_docs": format_item_generate_docs,
    "review_docs": format_item_review_docs,
    "refine_docs": format_item_refine_docs,
}


# ═══════════════════════════════════════════════════════════════════════
# Pool state dataclass (legacy — kept for backward compat / tests)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PoolDisplayState:
    """Observable state for a single pool's display panel.

    All fields are updated by the pool loop or watchers; the display
    reads them on each tick.
    """

    name: str
    completed: int = 0
    total: int = 0
    #: Claimable backlog from the graph (refreshed via ``refresh_progress``).
    pending: int = 0
    cost: float = 0.0
    start_time: float = field(default_factory=time.time)

    #: Latest streamed items (newest at right / bottom).
    items: deque = field(default_factory=lambda: deque(maxlen=STREAM_MAXLEN))

    #: Throttle state — set when backlog exceeds cap.
    throttled: bool = False
    throttle_reason: str = ""

    #: Timestamp of the last completed item (for stall detection).
    last_completion_at: float | None = None

    #: Events received *this session* (not including graph baseline).
    #: Used for rate calculation so the display doesn't show a fake rate
    #: derived from historical graph counts divided by session elapsed time.
    _events_this_run: int = 0

    def add_item(self, item: dict[str, Any]) -> None:
        """Push a streamed item into the display deque."""
        self.items.append(item)

    @property
    def rate(self) -> float | None:
        """Items per second (this-run average, excluding graph baseline)."""
        elapsed = time.time() - self.start_time
        if elapsed <= 0 or self._events_this_run <= 0:
            return None
        return self._events_this_run / elapsed

    @property
    def remaining(self) -> int:
        return max(self.pending, self.total - self.completed, 0)

    @property
    def ratio(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(self.completed / self.total, 1.0)

    @property
    def pct(self) -> float:
        return self.ratio * 100.0


# ═══════════════════════════════════════════════════════════════════════
# Legacy rendering utilities (kept for backward compat / tests)
# ═══════════════════════════════════════════════════════════════════════

#: Legacy label column width (superseded by LABEL_WIDTH from base).
LABEL_COL = 16

#: Legacy progress bar width (superseded by terminal-responsive bar_width).
BAR_WIDTH = 36


def make_bar(ratio: float, width: int = BAR_WIDTH) -> str:
    """Create a thin Unicode progress bar.

    ``━`` for filled, ``─`` for empty.
    """
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)
    return "━" * filled + "─" * (width - filled)


def format_time_value(seconds: float) -> str:
    """Format a duration as ``Xh Ym``, ``Xm Ys``, or ``Xs``."""
    if seconds < 0:
        return "--"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s" if s else f"{m}m"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m" if m else f"{h}h"


def compute_eta(pools: list[PoolDisplayState]) -> float | None:
    """ETA in seconds based on aggregate throughput."""
    total_remaining = sum(p.remaining for p in pools)
    if total_remaining <= 0:
        return 0.0
    total_completed = sum(p.completed for p in pools)
    if not pools:
        return None
    earliest = min(p.start_time for p in pools)
    elapsed = time.time() - earliest
    if elapsed <= 0 or total_completed <= 0:
        return None
    throughput = total_completed / elapsed
    return total_remaining / throughput


def compute_etc(pools: list[PoolDisplayState]) -> float | None:
    """Estimated Total Cost: current_cost + cost_per_item × remaining."""
    total_cost = sum(p.cost for p in pools)
    total_completed = sum(p.completed for p in pools)
    total_remaining = sum(p.remaining for p in pools)
    if total_completed <= 0 or total_remaining <= 0:
        return None
    cost_per_item = total_cost / total_completed
    return total_cost + cost_per_item * total_remaining


def render_pool_panel(state: PoolDisplayState) -> Text:
    """Render one pool's block using legacy custom layout.

    .. deprecated:: Use ``SN6PoolDisplay._build_pipeline_section`` instead.
    """
    pool_name = state.name
    label = _LEGACY_POOL_LABELS.get(pool_name, pool_name.upper())
    style = POOL_STYLES.get(pool_name, "white")

    header = Text()
    if state.throttled:
        header.append(f"  {label}", style=style)
        header.append(f" [paused: {state.throttle_reason}]", style="bold red")
    else:
        header.append(f"  {label}", style=style)

    label_len = len(header.plain)
    pad = max(1, LABEL_COL - label_len)
    header.append(" " * pad)
    filled_count = int(BAR_WIDTH * state.ratio)
    header.append("━" * filled_count, style="green")
    header.append("─" * (BAR_WIDTH - filled_count), style="dim")
    header.append("  ")
    header.append(f"{state.completed}/{state.total}", style="white")
    header.append(f"  {state.pct:.0f}%", style="dim")
    r = state.rate
    if r is not None:
        header.append(f"  {r:.1f}/s", style="dim")
    if state.cost > 0:
        header.append(f"  ${state.cost:.2f}", style="green")

    result = Text()
    result.append_text(header)
    formatter = ITEM_FORMATTERS.get(pool_name)
    if formatter and state.items:
        for item in state.items:
            result.append("\n")
            result.append_text(formatter(item))
    return result


def render_footer(
    pools: list[PoolDisplayState],
    *,
    cost_limit: float = 0.0,
    graph_latency_ms: float | None = None,
    llm_latency_s: float | None = None,
    graph_host: str = "graph",
    llm_host: str = "llm",
) -> Text:
    """Render the footer block using legacy custom layout.

    .. deprecated:: Use ``SN6PoolDisplay._build_resources_section`` instead.
    """
    footer = Text()
    sep_width = LABEL_COL + BAR_WIDTH + 30
    footer.append("─" * sep_width, style="dim")
    earliest = min(p.start_time for p in pools) if pools else time.time()
    elapsed = time.time() - earliest
    eta = compute_eta(pools)
    footer.append("\n  TIME", style="bold white")
    pad = max(1, LABEL_COL - 6)
    footer.append(" " * pad)
    total_expected = elapsed + (eta if eta and eta > 0 else 0)
    time_ratio = elapsed / total_expected if total_expected > 0 else 1.0
    bar_filled = int(BAR_WIDTH * min(time_ratio, 1.0))
    footer.append("━" * bar_filled, style="blue")
    footer.append("─" * (BAR_WIDTH - bar_filled), style="dim")
    footer.append(f"  {format_time_value(elapsed)}", style="white")
    if eta is not None and eta > 0:
        footer.append(f"  ETA {format_time_value(eta)}", style="dim")
    total_cost = sum(p.cost for p in pools)
    if total_cost > 0 or cost_limit > 0:
        etc = compute_etc(pools)
        footer.append("\n  COST", style="bold white")
        pad = max(1, LABEL_COL - 6)
        footer.append(" " * pad)
        cost_ratio = total_cost / cost_limit if cost_limit > 0 else 0.0
        bar_filled = int(BAR_WIDTH * min(cost_ratio, 1.0))
        cost_color = (
            "green" if cost_ratio < 0.5 else ("yellow" if cost_ratio < 0.8 else "red")
        )
        footer.append("━" * bar_filled, style=cost_color)
        footer.append("─" * (BAR_WIDTH - bar_filled), style="dim")
        footer.append(f"  ${total_cost:.2f}", style="white")
        if etc is not None:
            footer.append(f"  ETC ${etc:.2f}", style="dim")
        if cost_limit > 0:
            footer.append(f"  CAP ${cost_limit:.2f}", style="dim")
    server_parts: list[str] = []
    if graph_latency_ms is not None:
        server_parts.append(f"{graph_host} (avg {graph_latency_ms:.0f}ms)")
    if llm_latency_s is not None:
        server_parts.append(f"{llm_host} (avg {llm_latency_s:.1f}s)")
    if server_parts:
        footer.append("\n  SERVERS", style="bold white")
        pad = max(1, LABEL_COL - 9)
        footer.append(" " * pad)
        footer.append("  ".join(server_parts), style="dim")
    return footer


def render_full_display(
    pools: dict[str, PoolDisplayState],
    *,
    cost_limit: float = 0.0,
    graph_latency_ms: float | None = None,
    llm_latency_s: float | None = None,
    graph_host: str = "graph",
    llm_host: str = "llm",
) -> Text:
    """Compose legacy display from 6 pool panels + footer.

    .. deprecated:: Use ``SN6PoolDisplay`` (canonical BaseProgressDisplay
        subclass) instead.
    """
    display = Text()
    display.append("  Standard Name Pipeline\n", style="bold white")
    pool_list = [pools[name] for name in POOL_ORDER if name in pools]
    for i, pstate in enumerate(pool_list):
        if i > 0:
            display.append("\n")
        display.append_text(render_pool_panel(pstate))
    display.append("\n")
    display.append_text(
        render_footer(
            pool_list,
            cost_limit=cost_limit,
            graph_latency_ms=graph_latency_ms,
            llm_latency_s=llm_latency_s,
            graph_host=graph_host,
            llm_host=llm_host,
        )
    )
    return display


# ═══════════════════════════════════════════════════════════════════════
# SN6PoolDisplay — canonical BaseProgressDisplay subclass
# ═══════════════════════════════════════════════════════════════════════


class SN6PoolDisplay(BaseProgressDisplay):
    """Full-width 6-pool display using the canonical ``BaseProgressDisplay``.

    Internally tracks 6 pools (generate_name, review_name, refine_name,
    generate_docs, review_docs, refine_docs) but renders only 4 display
    rows by merging generate+refine into combined rows.

    Usage::

        display = SN6PoolDisplay(
            cost_limit=5.0,
            progress_fn=_progress_fn,
            accumulated_cost_fn=_cost_fn,
        )
        display.on_event({"pool": "review_name", "name": "e_temp", "score": 0.85, ...})

    ``progress_fn`` returns ``{pool: {"pending": int, "done": int}}`` —
    pending mirrors the claim predicates, done counts work already
    persisted in the graph (prior runs included).  This seeds the bars
    with cross-run pipeline position instead of starting at 0 % every
    session.

    The ``on_event`` method is the callback wired into worker pools.
    Each call pushes an item into the corresponding pool's
    ``WorkerStats.stream_queue`` and increments its counter.
    """

    def __init__(
        self,
        *,
        cost_limit: float = 0.0,
        console: Any | None = None,
        progress_fn: Any | None = None,
        accumulated_cost_fn: Any | None = None,
        flush: bool = False,
    ) -> None:
        super().__init__(
            facility="sn",
            cost_limit=cost_limit,
            console=console,
            title_suffix="Standard Name",
        )
        self._progress_fn = progress_fn
        self._accumulated_cost_fn = accumulated_cost_fn
        #: Flush mode: generate_name is gated, so its (large) source
        #: backlog is excluded from row totals, ETA, and ETC projection.
        self._flush = flush

        #: Authoritative spend ledger from the run's BudgetManager.
        #: Per-pool ``on_event`` payloads systematically undercount real
        #: spend — fanout sub-charges, L6 grammar-retry charges, and
        #: retry-accumulated cost inside ``acall_llm_structured`` bill the
        #: budget ledger without emitting a matching display event.  When
        #: the run wires its ledger here (via :meth:`set_budget_ledger`),
        #: the COST gauge and final summary reconcile to it instead of to
        #: summed event payloads.  Falls back to event sums when unset
        #: (tests / dry-run with no manager).
        self._budget_phase_spent: dict[str, float] | None = None
        self._budget_total: float | None = None

        # Per-pool observable state (7 pools).
        # PoolDisplayState tracks completed/total/cost; WorkerStats drives
        # the canonical streaming display via stream_queue.
        self.pools: dict[str, PoolDisplayState] = {
            name: PoolDisplayState(name=name) for name in POOL_ORDER
        }
        self._pool_stats: dict[str, WorkerStats] = {
            name: WorkerStats() for name in POOL_ORDER
        }

        # Per-pool batch accumulators for smooth streaming.
        # Accumulate stream items within a burst; tick() flushes them with
        # last_batch_time so StreamQueue can apply adaptive rate pacing.
        # This matches the discovery pipeline pattern (parallel.py sets
        # stats.last_batch_time; progress.py passes it to stream_queue.add).
        self._pending_stream: dict[str, list] = {name: [] for name in POOL_ORDER}
        self._batch_start: dict[str, float] = {}  # pool → first-event timestamp

    # ── Header ─────────────────────────────────────────────────────────

    def _header_mode_label(self) -> str | None:
        """Show FLUSH in the header when draining without generation."""
        return "FLUSH" if self._flush else None

    # ── Event callback (wired into workers) ───────────────────────────

    def on_event(self, ev: dict[str, Any]) -> None:
        """Push a per-item event into the display.

        Called by workers after each successful persist.  Thread-safe
        because deque.append is atomic in CPython.

        Events are accumulated in ``_pending_stream`` and flushed in
        :meth:`tick` with ``last_batch_time`` so the StreamQueue can
        apply adaptive rate pacing (matching the discovery pipeline pattern).

        Args:
            ev: Event dict with at minimum ``"pool"`` key matching one
                of :data:`POOL_ORDER`.  Additional keys are pool-specific.
        """
        pool_name = ev.get("pool", "")
        state = self.pools.get(pool_name)
        if state is None:
            return
        state.add_item(ev)
        state.completed += 1
        state._events_this_run += 1
        state.last_completion_at = time.time()
        cost = ev.get("cost", 0.0)
        if cost:
            state.cost += float(cost)

        # Accumulate stream items for batch-aware flushing in tick().
        # Record batch start time on first event in a burst.
        ws = self._pool_stats.get(pool_name)
        if ws is not None:
            mapper = _EVENT_MAPPERS.get(pool_name)
            if mapper:
                stream_item = mapper(ev)
                if pool_name not in self._batch_start:
                    self._batch_start[pool_name] = time.time()
                self._pending_stream[pool_name].append(stream_item)
            ws.processed = state.completed
            ws.total = max(state.total, state.completed)
            ws.cost = state.cost

    # ── Graph progress refresh ────────────────────────────────────────

    def refresh_progress(self) -> None:
        """Refresh pool completed/pending counts from the graph callback.

        ``progress_fn`` returns ``{pool: {"pending": int, "done": int}}``.
        The graph is the source of truth: ``completed`` is the graph-side
        done count (which includes prior runs and concurrent sessions),
        ratcheted against optimistic ``on_event`` increments so the bar
        never steps backwards between graph refreshes.  ``total`` is
        ``completed + pending`` and tracks the live backlog in both
        directions (work can be added or drained).
        """
        if self._progress_fn is None:
            return
        try:
            counts = self._progress_fn()
        except Exception:
            return

        for pool_name in POOL_ORDER:
            state = self.pools.get(pool_name)
            entry = counts.get(pool_name) if isinstance(counts, dict) else None
            if state is None or not isinstance(entry, dict):
                continue
            pending = int(entry.get("pending", 0))
            done = int(entry.get("done", 0))
            state.pending = pending
            state.completed = max(state.completed, done)
            state.total = state.completed + pending

            # Sync WorkerStats for canonical rendering.
            ws = self._pool_stats.get(pool_name)
            if ws is not None:
                ws.processed = state.completed
                ws.total = max(state.total, state.completed)
                ws.cost = state.cost

    # ── Budget ledger reconciliation ──────────────────────────────────

    def set_budget_ledger(
        self,
        *,
        phase_spent: dict[str, float] | None = None,
        total: float | None = None,
    ) -> None:
        """Wire the run's authoritative spend ledger into the display.

        The per-pool ``on_event`` payloads only carry the cost the worker
        chose to surface for a single emitted event.  Several charge paths
        bill the :class:`~imas_codex.standard_names.budget.BudgetManager`
        ledger *without* a matching display event — fanout sub-charges,
        grammar-retry charges, and retry-accumulated cost inside
        ``acall_llm_structured``.  Summing event payloads therefore
        systematically *undercounts* real spend (the dangerous direction:
        it hides money).  The budget ledger sees every charge.

        Call this with ``BudgetManager.phase_spent`` (per-pool USD, keyed
        by :data:`POOL_ORDER` names) and an authoritative total
        (``BudgetManager.spent`` or the graph-reconciled run total) so the
        COST gauge and the final summary report what was actually billed.

        Args:
            phase_spent: Per-pool spend dict keyed by pool name.  Folds
                fanout / retry sub-charges into their parent phase
                automatically (that is how the ledger already tags them).
            total: Authoritative total spend in USD.  Defaults to the sum
                of ``phase_spent`` when omitted.
        """
        if phase_spent is not None:
            self._budget_phase_spent = dict(phase_spent)
        if total is not None:
            self._budget_total = float(total)
        elif phase_spent is not None and self._budget_total is None:
            self._budget_total = float(sum(phase_spent.values()))

    def _ledger_total(self) -> float | None:
        """Authoritative total spend, or ``None`` when no ledger is wired."""
        if self._budget_total is not None:
            return self._budget_total
        if self._budget_phase_spent is not None:
            return float(sum(self._budget_phase_spent.values()))
        return None

    # ── Canonical display methods (override BaseProgressDisplay) ──────

    def _row_pending(self, sub_pools: tuple[str, ...]) -> int:
        """Aggregate claimable backlog for a display row.

        In flush mode the generate_name pool is gated, so its (large)
        source backlog is excluded — only work the run can actually
        claim counts toward the row total.
        """
        return sum(
            self.pools[p].pending
            for p in sub_pools
            if not (self._flush and p == "generate_name")
        )

    def _build_pipeline_section(self) -> Text:
        """Build pipeline section using canonical PipelineRowConfig.

        Renders 4 display rows by aggregating sub-pools according to
        :data:`DISPLAY_POOL_MAP`.  Each row shows graph-truth progress
        (``done/total`` including prior runs), the live backlog, and the
        most recent processed name (sticky until replaced).
        """
        rows: list[PipelineRowConfig] = []
        for display_row in DISPLAY_ROWS:
            sub_pools = DISPLAY_POOL_MAP[display_row]
            label = DISPLAY_LABELS[display_row]
            style = DISPLAY_STYLES[display_row]

            # Aggregate stats from sub-pools.
            completed = sum(self.pools[p].completed for p in sub_pools)
            pending = self._row_pending(sub_pools)
            total = max(completed + pending, 1)
            cost = sum(self.pools[p].cost for p in sub_pools)

            # Combined rate from sub-pools.
            rates = [
                self.pools[p].rate for p in sub_pools if self.pools[p].rate is not None
            ]
            rate = sum(rates) if rates else None

            events_this_run = sum(self.pools[p]._events_this_run for p in sub_pools)

            # Stream item: pick from first sub-pool that has one.
            primary_text = ""
            primary_text_style = "white"
            description = ""
            score_value: float | None = None
            for p in sub_pools:
                ws = self._pool_stats[p]
                si = ws._current_stream_item
                if si:
                    primary_text = si.get("primary_text", "")
                    primary_text_style = si.get("primary_text_style", "white")
                    description = si.get("description", "")
                    _sv = si.get("score_value")
                    if isinstance(_sv, int | float):
                        score_value = float(_sv)
                    break

            # Idle rows announce their backlog instead of a bare "idle".
            idle_label = f"{pending:,} queued" if pending > 0 else "idle"

            rows.append(
                PipelineRowConfig(
                    name=label,
                    style=style,
                    completed=completed,
                    total=total,
                    rate=rate,
                    cost=cost if cost > 0 else None,
                    primary_text=primary_text,
                    primary_text_style=primary_text_style,
                    description=description,
                    score_value=score_value,
                    show_total=True,
                    idle_label=idle_label,
                    is_processing=(events_this_run > 0 and pending > 0),
                    is_complete=pending == 0 and completed > 0,
                )
            )
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build TIME + COST resource gauges using pipeline-aware ETC.

        Replaces the old per-pool independent projection with a
        hybrid pipeline-flow model that accounts for upstream work
        flowing through downstream pools.
        """
        from imas_codex.standard_names.cost_model import (
            compute_cycle_estimates,
            compute_pipeline_etc,
            detect_stall,
            resolve_pool_cpi,
        )

        # Event-sum cost undercounts (fanout / retry sub-charges emit no
        # display event); reconcile the COST gauge to the budget ledger
        # when the run has wired it in, so mid-run and final agree with
        # the cap.  Fall back to the event sum when no ledger is present.
        event_cost = sum(p.cost for p in self.pools.values())
        total_cost = self._ledger_total()
        if total_cost is None:
            total_cost = event_cost

        # ETA: parallel ETA across pools with remaining work.  Pools that
        # have a backlog but no session rate yet would otherwise silently
        # drop out of the estimate (the old behavior — ETA 55s while
        # thousands of items were queued).  Their pending work is folded
        # in via the aggregate throughput of the pools that *are* rated,
        # so the estimate is honest even early in a run.  Flush mode
        # excludes the gated generate_name backlog.
        work_items: list[tuple[int, float | None]] = []
        unrated_pending = 0
        rated_total_rate = 0.0
        for pool_name in POOL_ORDER:
            if self._flush and pool_name == "generate_name":
                continue
            state = self.pools[pool_name]
            remaining = state.remaining
            if remaining <= 0:
                continue
            if state.rate is not None and state.rate > 0:
                work_items.append((remaining, state.rate))
                rated_total_rate += state.rate
            else:
                unrated_pending += remaining

        eta = compute_parallel_eta(work_items)
        if unrated_pending > 0 and rated_total_rate > 0:
            unrated_eta = unrated_pending / rated_total_rate
            eta = max(eta or 0.0, unrated_eta)

        # Accumulated cost from graph (cross-run).
        accumulated = total_cost
        if self._accumulated_cost_fn:
            try:
                graph_cost = self._accumulated_cost_fn()
                accumulated = graph_cost + total_cost
            except Exception:
                pass

        # --- Pipeline-aware ETC ---
        projected: float | None = None
        stalled = False
        try:
            from imas_codex.standard_names.graph_ops import (
                query_historical_cpi,
                query_pipeline_buckets,
            )

            buckets = query_pipeline_buckets()
            if self._flush:
                # Flush gates generation: un-started sources are out of
                # scope, so ETC projects only the in-flight names.
                from dataclasses import replace as _dc_replace

                buckets = _dc_replace(buckets, a_sources=0)

            # Build CycleEstimates from this-run pool counters.
            gn = self.pools["generate_name"]
            rn = self.pools["review_name"]
            rfn = self.pools["refine_name"]
            rd = self.pools["review_docs"]
            rfd = self.pools["refine_docs"]

            cycles = compute_cycle_estimates(
                refine_name_done=rfn.completed,
                name_review_first_pass_done=rn.completed,
                refine_docs_done=rfd.completed,
                docs_review_first_pass_done=rd.completed,
                # accepted_count: names reviewed with score >= threshold
                accepted_count=max(rn.completed - rfn.completed, 0),
                total_completed_name_stage=rn.completed,
                sources_attempted=gn.total if gn.total > 0 else 0,
                names_drafted=gn.completed,
            )

            # Resolve CPI per pool.
            historical = query_historical_cpi()

            # Build sibling CPI fallback map.
            # refine_name ≈ 1.0 × review_name; refine_docs ≈ 1.0 × review_docs
            _sibling_map: dict[str, str] = {
                "refine_name": "review_name",
                "refine_docs": "review_docs",
            }

            cpis: dict[str, Any] = {}
            for pool_name in POOL_ORDER:
                state = self.pools[pool_name]
                sibling_pool = _sibling_map.get(pool_name)
                sibling_cpi: float | None = None
                if sibling_pool and sibling_pool in cpis:
                    sibling_cpi = cpis[sibling_pool].value

                cpis[pool_name] = resolve_pool_cpi(
                    pool=pool_name,
                    observed_cost=state.cost,
                    observed_completed=state.completed,
                    historical=historical,
                    sibling_cpi=sibling_cpi,
                )

            projected = compute_pipeline_etc(
                buckets=buckets,
                cycles=cycles,
                cpis=cpis,
                accumulated_cost=accumulated,
            )

            # Stall detection (gated generate_name never counts as stalled).
            pool_pending = {
                name: self.pools[name].remaining
                for name in POOL_ORDER
                if not (self._flush and name == "generate_name")
            }
            pool_last_at = {
                name: self.pools[name].last_completion_at for name in POOL_ORDER
            }
            stalled = detect_stall(pool_pending, pool_last_at, time.time())
        except Exception:
            # Fallback: no projection if graph queries fail.
            projected = None

        # Format ETC for display.
        # When stalled, suppress ETC (canonical renderer can't render "∞").
        etc_value: float | None = projected
        if stalled:
            etc_value = None

        config = ResourceConfig(
            elapsed=self.elapsed,
            eta=eta,
            run_cost=total_cost if total_cost > 0 else None,
            cost_limit=self.cost_limit if self.cost_limit > 0 else None,
            accumulated_cost=accumulated,
            etc=etc_value,
        )
        return build_resource_section(config, self.gauge_width)

    # ── Tick (called by run_discovery ticker task) ────────────────────

    def tick(self) -> None:
        """Periodic refresh: flush pending batches, drain stream queues, repaint.

        Flushing pending batches (accumulated in :meth:`on_event`) with
        ``last_batch_time`` gives StreamQueue adaptive rate pacing: items
        spread out over the next expected batch gap rather than cycling
        through instantly.  Matches the discovery pipeline pattern where
        ``stats.last_batch_time`` is set after each batch and passed to
        ``stream_queue.add()``.
        """
        now = time.time()

        # Flush pending stream items with batch timing.
        for pool_name, pending in self._pending_stream.items():
            if not pending:
                continue
            ws = self._pool_stats.get(pool_name)
            if ws is None:
                continue
            batch_start = self._batch_start.pop(pool_name, now)
            last_batch_time = now - batch_start
            ws.stream_queue.add(pending, last_batch_time=last_batch_time)
            pending.clear()

        # Drain stream queues → _current_stream_item for each pool.
        # The last item is sticky: SN pools complete items at multi-second
        # cadence, and the last processed name carries more information
        # than reverting to a blank "processing..." row.  Items are only
        # replaced, never cleared.
        for ws in self._pool_stats.values():
            item = ws.stream_queue.pop()
            if item is not None:
                ws._current_stream_item = item

        self.refresh_progress()
        self._refresh()

    # ── Harness compatibility ─────────────────────────────────────────

    def refresh_from_graph(self, facility: str) -> None:
        """Called by run_discovery graph-refresh task."""
        self.refresh_progress()
        self._refresh()

    def print_summary(self) -> None:
        """Print a compact summary after the run completes.

        COST figures reconcile to the authoritative budget ledger (wired
        via :meth:`set_budget_ledger`) rather than to summed ``on_event``
        payloads, which systematically undercount real spend (fanout /
        retry sub-charges emit no display event).  COMPLETED COUNTS still
        come from the live per-pool counters — those are correct.  When no
        ledger is wired (tests / dry-run), fall back to the event sum.
        """
        # Per-pool spend: ledger when present, else the live event sum.
        ledger = self._budget_phase_spent

        def _pool_cost(name: str) -> float:
            if ledger is not None:
                return float(ledger.get(name, 0.0))
            return self.pools[name].cost

        total_items = sum(p.completed for p in self.pools.values())
        total_cost = self._ledger_total()
        if total_cost is None:
            total_cost = sum(p.cost for p in self.pools.values())

        if total_items > 0 or total_cost > 0:
            self.console.print()
            for name in POOL_ORDER:
                p = self.pools[name]
                if p.completed > 0:
                    parts = [
                        f"  {POOL_LABELS[name]}: {p.completed:,}",
                    ]
                    pool_cost = _pool_cost(name)
                    if pool_cost > 0:
                        parts.append(f"${pool_cost:.2f}")
                    self.console.print("  ".join(parts))
            if total_cost > 0:
                self.console.print(f"  TOTAL COST: ${total_cost:.2f}")

    def on_worker_status(self, group: Any) -> None:
        """Callback for worker status updates (harness compat)."""
        self.update_worker_status(group)


__all__ = [
    "BAR_WIDTH",
    "DISPLAY_LABELS",
    "DISPLAY_POOL_MAP",
    "DISPLAY_ROWS",
    "DISPLAY_STYLES",
    "ITEM_FORMATTERS",
    "LABEL_COL",
    "POOL_LABELS",
    "POOL_ORDER",
    "POOL_STYLES",
    "PoolDisplayState",
    "SCORE_GREEN_THRESHOLD",
    "SCORE_YELLOW_THRESHOLD",
    "SN6PoolDisplay",
    "STREAM_MAXLEN",
    "compute_eta",
    "compute_etc",
    "format_item_generate_docs",
    "format_item_generate_name",
    "format_item_refine_docs",
    "format_item_refine_name",
    "format_item_review_docs",
    "format_item_review_name",
    "format_time_value",
    "make_bar",
    "render_footer",
    "render_full_display",
    "render_pool_panel",
    "score_color",
]

"""Benchmark runner for comparing LLM models on standard name generation.

Extracts a fixed dataset, runs it through multiple models, validates
output via grammar round-trip, and compares against a reference set.
Produces a :class:`BenchmarkReport` with per-model metrics suitable
for Rich table display and JSON export.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_standard_names.grammar import (
    compose_standard_name,
    parse_standard_name,
)

from imas_codex.standard_names.models import StandardNameComposeBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    models: list[str]
    source: str = "dd"
    ids_filter: str | None = None
    domain_filter: str | None = None
    facility: str | None = None
    max_candidates: int = 50
    runs_per_model: int = 1
    temperature: float = 0.0  # pinned for reproducibility
    reviewer_model: str | None = None  # frontier model for quality scoring
    force: bool = False  # re-run over already-processed paths
    names_only: bool = True  # docs benchmarking not yet implemented


@dataclass
class ModelResult:
    """Results from running one model."""

    model: str
    candidates: list[dict] = field(default_factory=list)
    grammar_valid_count: int = 0
    grammar_invalid_count: int = 0
    fields_consistent_count: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    names_per_minute: float = 0.0
    cost_per_name: float = 0.0
    skipped_count: int = 0
    batch_errors: int = 0
    # Quality against reference set
    reference_overlap: int = 0
    reference_total: int = 0
    reference_precision: float = 0.0
    reference_recall: float = 0.0
    # Quality scoring (reviewer model)
    quality_scores: list[dict] = field(default_factory=list)
    quality_distribution: dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    avg_doc_length: float = 0.0
    avg_fields_populated: float = 0.0
    # Per-dimension score averages
    avg_grammar_score: float = 0.0
    avg_semantic_score: float = 0.0
    avg_convention_score: float = 0.0
    avg_completeness_score: float = 0.0
    # Prompt-cache statistics
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # Reviewer cost and timing breakdown
    reviewer_cost: float = 0.0
    compose_elapsed_seconds: float = 0.0
    review_elapsed_seconds: float = 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    config: BenchmarkConfig
    results: list[ModelResult]
    reference_names: list[str]
    extraction_count: int = 0
    timestamp: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> BenchmarkReport:
        """Deserialize from JSON string."""
        raw = json.loads(data)
        config = BenchmarkConfig(**raw["config"])
        results = [ModelResult(**r) for r in raw["results"]]
        return cls(
            config=config,
            results=results,
            reference_names=raw["reference_names"],
            extraction_count=raw.get("extraction_count", 0),
            timestamp=raw.get("timestamp", ""),
        )


# ---------------------------------------------------------------------------
# Grammar validation
# ---------------------------------------------------------------------------


def _get_segment_fields(candidate: dict) -> dict:
    """Extract segment fields from a candidate dict.

    Handles both nested format (``{segments: {base_token: ...}}``)
    and legacy flat format (``{base_token: ...}``).
    """
    segs = candidate.get("segments")
    if isinstance(segs, dict):
        return segs
    return candidate


def _resolve_name(candidate: dict) -> str:
    """Resolve the composed standard name from a candidate dict.

    Tries ``standard_name`` key first (legacy), then composes from
    IR segment fields via ISN.
    """
    name = candidate.get("standard_name", "") or candidate.get("id", "")
    if name:
        return name

    segs = _get_segment_fields(candidate)
    base_token = segs.get("base_token")
    if not base_token:
        return ""

    try:
        from imas_standard_names.grammar.ir import (
            BaseKind,
            QuantityOrCarrier,
            StandardNameIR,
        )
        from imas_standard_names.grammar.render import compose

        base = QuantityOrCarrier(
            token=base_token,
            kind=BaseKind(segs.get("base_kind", "quantity")),
        )
        qualifiers_raw = segs.get("qualifiers") or []
        qualifiers = None
        if qualifiers_raw:
            from imas_standard_names.grammar.ir import Qualifier

            qualifiers = [Qualifier(token=q) for q in qualifiers_raw]

        ir = StandardNameIR(base=base, qualifiers=qualifiers or [])
        return compose(ir)
    except Exception:
        return ""


def validate_candidate(candidate: dict) -> tuple[bool, bool]:
    """Validate a single candidate via grammar round-trip.

    Returns:
        (grammar_valid, fields_consistent) tuple.
        grammar_valid: True if the name parses and round-trips.
        fields_consistent: True if IR compose produces the same name.
    """
    segs = _get_segment_fields(candidate)
    name = _resolve_name(candidate)

    grammar_valid = False
    fields_consistent = False

    # Check grammar round-trip
    try:
        parsed = parse_standard_name(name)
        normalized = compose_standard_name(parsed)
        grammar_valid = True  # parse+compose succeeded
    except Exception:
        return False, False

    # Check IR consistency: if candidate has IR fields, compose and compare
    try:
        base_token = segs.get("base_token")
        if base_token:
            from imas_standard_names.grammar.ir import (
                BaseKind,
                QuantityOrCarrier,
                StandardNameIR,
            )
            from imas_standard_names.grammar.render import compose

            base_kind = segs.get("base_kind", "quantity")
            base = QuantityOrCarrier(token=base_token, kind=BaseKind(base_kind))

            qualifiers = None
            qualifiers_raw = segs.get("qualifiers") or []
            if qualifiers_raw:
                from imas_standard_names.grammar.ir import Qualifier

                qualifiers = [Qualifier(token=q) for q in qualifiers_raw]

            ir = StandardNameIR(base=base, qualifiers=qualifiers or [])
            from_ir = compose(ir)
            fields_consistent = from_ir == normalized
    except Exception:
        pass

    return grammar_valid, fields_consistent


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------


def compare_to_reference(
    candidates: list[dict],
    reference: dict[str, dict],
) -> tuple[int, int, float, float]:
    """Compare model output against the reference set.

    Args:
        candidates: List of candidate dicts with source_id and standard_name.
        reference: REFERENCE_NAMES dict mapping source_path → {name, fields}.

    Returns:
        (overlap, ref_total, precision, recall) tuple.
        overlap: Number of candidates whose standard_name matches reference.
        ref_total: Total entries in reference set.
        precision: overlap / len(candidates) if candidates else 0.
        recall: overlap / ref_total if ref_total else 0.
    """
    # Build lookup from source_id → generated name
    generated = {}
    for c in candidates:
        sid = c.get("source_id", "")
        generated[sid] = _resolve_name(c)

    overlap = 0
    ref_total = len(reference)
    for path, ref_entry in reference.items():
        if path in generated:
            # Normalize both for comparison
            gen_name = generated[path]
            ref_name = ref_entry["name"]
            try:
                gen_parsed = parse_standard_name(gen_name)
                gen_normalized = compose_standard_name(gen_parsed)
            except Exception:
                gen_normalized = gen_name

            try:
                ref_parsed = parse_standard_name(ref_name)
                ref_normalized = compose_standard_name(ref_parsed)
            except Exception:
                ref_normalized = ref_name

            if gen_normalized == ref_normalized:
                overlap += 1

    n_candidates = len(candidates)
    precision = overlap / n_candidates if n_candidates else 0.0
    recall = overlap / ref_total if ref_total else 0.0

    return overlap, ref_total, precision, recall


# ---------------------------------------------------------------------------
# Quality tier labels
# ---------------------------------------------------------------------------


async def score_with_reviewer(
    candidates: list[dict],
    reviewer_model: str,
    target: str = "names",
) -> tuple[list[dict], float]:
    """Score candidates using the production-fidelity review pipeline.

    Uses the split system/user prompt pair (``sn/review_names_system`` +
    ``sn/review_names_user``) with full graph context: K3 scored calibration
    examples, vector/same-base/same-path neighbour comparators, and grammar
    vocabulary.

    target="names" (default) — uses the 4-dimensional ``sn/review_names``
        rubric (grammar, semantic, convention, completeness; 0-80 total,
        normalised to 0-1).

    Returns (reviews, total_cost) where reviews is a list of dicts with:
    name, quality_tier, score, and per-dimension scores keyed ``<dim>_score``.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import (
        build_compose_context,
        fetch_review_neighbours,
    )
    from imas_codex.standard_names.example_loader import load_review_examples
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnlyBatch,
    )

    if target != "names":
        raise ValueError(f"Unknown review target {target!r}; expected 'names'.")

    response_model: type = StandardNameQualityReviewNameOnlyBatch

    # Get compose context (grammar enums + shared include variables)
    compose_ctx = build_compose_context()

    # Load K3 scored calibration examples from graph
    review_scored_examples: list[dict] = []
    try:
        with GraphClient() as gc:
            review_scored_examples = load_review_examples(
                gc, physics_domains=[], axis="name"
            )
    except Exception as exc:
        logger.debug("Could not load review examples: %s", exc)

    # Build system prompt (static, cached across batches)
    system_context = {
        **compose_ctx,
        "review_scored_examples": review_scored_examples,
    }
    system_prompt = render_prompt("sn/review_names_system", system_context)

    # Process in batches of 10
    all_reviews: list[dict] = []
    total_reviewer_cost: float = 0.0
    for i in range(0, len(candidates), 10):
        batch = candidates[i : i + 10]

        # Build per-candidate items with neighbour context
        batch_items = []
        for c in batch:
            name = _resolve_name(c)
            item: dict[str, Any] = {
                "standard_name": name,
                "source_id": c.get("source_id", ""),
                "description": c.get("description", "") or "",
                "documentation": (c.get("documentation", "") or "")[:500],
                "unit": c.get("unit", "N/A"),
                "kind": c.get("kind", "N/A"),
                "source_paths": c.get("source_paths", []),
            }

            # Fetch review neighbours from live graph
            try:
                neighbours = fetch_review_neighbours(
                    {
                        "id": name,
                        "name": name,
                        "description": c.get("description", ""),
                        "physical_base": (
                            c.get("base_token")
                            or (c.get("segments") or {}).get("base_token")
                        ),
                        "source_paths": c.get("source_paths", []),
                    }
                )
                item.update(neighbours)
            except Exception:
                item["vector_neighbours"] = []
                item["same_base_neighbours"] = []
                item["same_path_neighbours"] = []

            batch_items.append(item)

        # Build user prompt with full context
        user_context = {
            **compose_ctx,
            "review_scored_examples": review_scored_examples,
            "items": batch_items,
            "existing_names": [],
            "batch_context": "",
            "nearby_existing_names": [],
            "audit_findings": [],
            # Flatten neighbour lists for template — production injects
            # these at the batch level from the first item's context
            "vector_neighbours": batch_items[0].get("vector_neighbours", [])
            if batch_items
            else [],
            "same_base_neighbours": batch_items[0].get("same_base_neighbours", [])
            if batch_items
            else [],
            "same_path_neighbours": batch_items[0].get("same_path_neighbours", [])
            if batch_items
            else [],
        }

        try:
            user_prompt = render_prompt("sn/review_names_user", user_context)
        except Exception as exc:
            logger.warning("Failed to render review user prompt: %s", exc)
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result, cost, _ = await acall_llm_structured(
                model=reviewer_model,
                messages=messages,
                response_model=response_model,
                service="standard-names",
            )
            total_reviewer_cost += cost
            for r in result.reviews:
                review_dict: dict[str, Any] = {
                    "name": r.standard_name,
                    "quality_tier": r.scores.tier,
                    "score": r.scores.score,
                    "grammar_score": r.scores.grammar,
                    "semantic_score": r.scores.semantic,
                    "convention_score": r.scores.convention,
                    "completeness_score": r.scores.completeness,
                    "reasoning": r.reasoning,
                }
                all_reviews.append(review_dict)
        except Exception as e:
            logger.warning("Reviewer scoring failed for batch: %s", e)

    return all_reviews, total_reviewer_cost


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    config: BenchmarkConfig,
    extraction_batches: list[dict] | None = None,
) -> BenchmarkReport:
    """Run the benchmark across all configured models.

    Args:
        config: Benchmark configuration.
        extraction_batches: Pre-extracted candidate batches (list of dicts
            with items grouped by IDS). If None, extracts from graph.

    Returns:
        BenchmarkReport with per-model results.
    """
    from imas_codex.standard_names.benchmark_reference import REFERENCE_NAMES

    # --- 1. Extract candidates (same for all models) ---
    if extraction_batches is None:
        extraction_batches = _extract_candidates(config)

    # Flatten items for counting
    all_items = []
    for batch in extraction_batches:
        all_items.extend(batch.get("items", []))

    # Limit to max_candidates
    if len(all_items) > config.max_candidates:
        all_items = all_items[: config.max_candidates]
        # Rebuild batches with limited items
        extraction_batches = _rebuild_batches(extraction_batches, config.max_candidates)

    logger.info(
        "Benchmark: %d extraction items across %d batches",
        len(all_items),
        len(extraction_batches),
    )

    # --- 2. Run each model ---
    results: list[ModelResult] = []
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_compose_context

    context = build_compose_context()
    context["compose_scored_examples"] = []
    system_prompt = render_prompt("sn/generate_name_system", context)

    for model in config.models:
        logger.info("Benchmarking model: %s", model)
        model_result = await _run_model(
            model=model,
            extraction_batches=extraction_batches,
            config=config,
            reference=REFERENCE_NAMES,
            system_prompt=system_prompt,
            context=context,
        )
        model_result.compose_elapsed_seconds = model_result.elapsed_seconds
        results.append(model_result)

    # --- 2b. Reviewer scoring (mandatory — scores ARE the benchmark) ---
    if not config.reviewer_model:
        raise ValueError(
            "reviewer_model is required for benchmarking. "
            "Reviewer scores are the quality metric."
        )

    for result in results:
        if not result.candidates:
            continue
        t_review_start = time.monotonic()
        reviews, reviewer_cost = await score_with_reviewer(
            result.candidates,
            config.reviewer_model,
            target="names",
        )
        result.review_elapsed_seconds = round(time.monotonic() - t_review_start, 2)
        result.reviewer_cost = reviewer_cost
        result.quality_scores = reviews

        # Review completeness check
        n_candidates = len(result.candidates)
        n_reviews = len(reviews)
        if n_reviews < n_candidates:
            logger.warning(
                "Model %s: only %d/%d candidates reviewed "
                "(partial reviews may bias scores)",
                result.model,
                n_reviews,
                n_candidates,
            )

        # Compute distribution
        for r in reviews:
            tier = r.get("quality_tier", "unknown")
            result.quality_distribution[tier] = (
                result.quality_distribution.get(tier, 0) + 1
            )
        if reviews:
            result.avg_quality_score = sum(r.get("score", 0) for r in reviews) / len(
                reviews
            )

            # Per-dimension averages
            for dim in (
                "grammar",
                "semantic",
                "convention",
                "completeness",
            ):
                key = f"{dim}_score"
                vals = [r[key] for r in reviews if key in r and r[key] > 0]
                if vals:
                    setattr(
                        result,
                        f"avg_{key}",
                        sum(vals) / len(vals),
                    )

        # Compute doc length and field coverage metrics
        docs = [c.get("documentation", "") or "" for c in result.candidates]
        result.avg_doc_length = sum(len(d) for d in docs) / len(docs) if docs else 0.0

        all_fields = {
            "base_token",
            "qualifiers",
            "projection_axis",
            "locus_token",
            "process_token",
            "operator_token",
        }
        field_counts = []
        for c in result.candidates:
            populated = 0
            if c.get("base_token"):
                populated += 1
            if c.get("qualifiers"):
                populated += 1
            if c.get("projection_axis"):
                populated += 1
            if c.get("locus_token"):
                populated += 1
            if c.get("process_token"):
                populated += 1
            if c.get("operator_token"):
                populated += 1
            field_counts.append(populated / len(all_fields))
        result.avg_fields_populated = (
            sum(field_counts) / len(field_counts) if field_counts else 0.0
        )

    # --- 3. Build report ---
    report = BenchmarkReport(
        config=config,
        results=results,
        reference_names=list(REFERENCE_NAMES.keys()),
        extraction_count=len(all_items),
        timestamp=datetime.now(tz=UTC).isoformat(),
    )
    return report


def _extract_candidates(config: BenchmarkConfig) -> list[dict]:
    """Extract candidates from the graph DB.

    Returns list of batch dicts with keys: group_key, items, existing_names, context.
    """
    from imas_codex.standard_names.sources.dd import extract_dd_candidates

    batches = extract_dd_candidates(
        ids_filter=config.ids_filter,
        domain_filter=config.domain_filter,
        limit=config.max_candidates,
        force=config.force,
        write_skipped=False,
    )

    # Convert ExtractionBatch to plain dicts
    result = []
    for batch in batches:
        result.append(
            {
                "group_key": batch.group_key,
                "items": batch.items,
                "existing_names": list(batch.existing_names),
                "context": batch.context,
            }
        )
    return result


def _rebuild_batches(batches: list[dict], max_items: int) -> list[dict]:
    """Rebuild batches capping total items at max_items."""
    result = []
    count = 0
    for batch in batches:
        items = batch.get("items", [])
        remaining = max_items - count
        if remaining <= 0:
            break
        if len(items) > remaining:
            items = items[:remaining]
        result.append({**batch, "items": items})
        count += len(items)
    return result


async def _run_model(
    model: str,
    extraction_batches: list[dict],
    config: BenchmarkConfig,
    reference: dict[str, dict],
    system_prompt: str,
    context: dict[str, Any],
) -> ModelResult:
    """Run a single model across all extraction batches."""
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt

    result = ModelResult(model=model)
    all_candidates: list[dict] = []

    t0 = time.monotonic()

    for _run_idx in range(config.runs_per_model):
        for batch in extraction_batches:
            items = batch.get("items", [])
            if not items:
                continue

            group_key = batch.get("group_key", "unknown")
            existing = set(batch.get("existing_names", []))

            # Build user prompt context — mirrors workers.py pattern
            user_context = {
                "items": items,
                "ids_name": group_key,
                "existing_names": sorted(existing)[:200],
                "cluster_context": batch.get("context", ""),
            }

            try:
                user_prompt = render_prompt(
                    "sn/generate_name_dd", {**context, **user_context}
                )
            except Exception:
                logger.warning("Failed to render prompt for batch %s", group_key)
                result.batch_errors += 1
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # GPT-5.x models don't support temperature=0.0
            temp = config.temperature
            if "gpt-5" in model and temp == 0.0:
                temp = None  # let the provider use its default

            try:
                llm_response = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    temperature=temp,
                    service="standard-names",
                )
                llm_result, cost, tokens = llm_response
                result.total_cost += cost
                result.total_tokens += tokens
                result.cache_read_tokens += getattr(
                    llm_response, "cache_read_tokens", 0
                )
                result.cache_creation_tokens += getattr(
                    llm_response, "cache_creation_tokens", 0
                )
                logger.debug(
                    "Batch %s: cost=%.4f tokens=%d",
                    group_key,
                    cost,
                    tokens,
                )

                # Collect candidates
                for c in llm_result.candidates:
                    all_candidates.append(c.model_dump())
                result.skipped_count += len(llm_result.skipped)

            except Exception as exc:
                logger.warning(
                    "LLM call failed for model %s batch %s: %s",
                    model,
                    group_key,
                    exc,
                )
                result.batch_errors += 1

    elapsed = time.monotonic() - t0
    result.elapsed_seconds = round(elapsed, 2)
    result.candidates = all_candidates

    # --- Validate grammar ---
    valid = 0
    invalid = 0
    fields_ok = 0
    for c in all_candidates:
        g_valid, f_consistent = validate_candidate(c)
        if g_valid:
            valid += 1
        else:
            invalid += 1
        if f_consistent:
            fields_ok += 1

    result.grammar_valid_count = valid
    result.grammar_invalid_count = invalid
    result.fields_consistent_count = fields_ok

    # --- Derived metrics ---
    n = len(all_candidates)
    if elapsed > 0 and n > 0:
        result.names_per_minute = round(n / elapsed * 60, 1)
    if n > 0:
        result.cost_per_name = round(result.total_cost / n, 6)

    # --- Reference comparison ---
    overlap, ref_total, precision, recall = compare_to_reference(
        all_candidates, reference
    )
    result.reference_overlap = overlap
    result.reference_total = ref_total
    result.reference_precision = round(precision, 4)
    result.reference_recall = round(recall, 4)

    logger.info(
        "Model %s: %d names, %d valid, %d invalid, $%.4f cost, %.1f names/min",
        model,
        n,
        valid,
        invalid,
        result.total_cost,
        result.names_per_minute,
    )

    return result


# ---------------------------------------------------------------------------
# Rich table rendering
# ---------------------------------------------------------------------------


def render_comparison_table(report: BenchmarkReport) -> None:
    """Render a Rich comparison table to stdout."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Check if any result has quality scores
    has_quality = any(r.quality_scores for r in report.results)

    table = Table(
        title="SN Benchmark Results",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Model", style="bold")
    table.add_column("Names", justify="right")
    table.add_column("Valid %", justify="right")
    table.add_column("Fields %", justify="right")
    table.add_column("Ref Match", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Names/min", justify="right")
    table.add_column("$/name", justify="right")
    table.add_column("Cache %", justify="right")
    table.add_column("Errors", justify="right")
    if has_quality:
        table.add_column("Avg Quality", justify="right")
        table.add_column("Gram", justify="right")
        table.add_column("Sem", justify="right")
        table.add_column("Conv", justify="right")
        table.add_column("Compl", justify="right")

    for r in report.results:
        n = len(r.candidates)
        valid_pct = f"{r.grammar_valid_count / n * 100:.0f}%" if n else "—"
        fields_pct = f"{r.fields_consistent_count / n * 100:.0f}%" if n else "—"
        ref_match = (
            f"{r.reference_overlap}/{r.reference_total}" if r.reference_total else "—"
        )
        cost_str = f"${r.total_cost:.4f}" if r.total_cost > 0 else "—"
        speed_str = f"{r.names_per_minute:.0f}" if r.names_per_minute > 0 else "—"
        cpn_str = f"${r.cost_per_name:.4f}" if r.cost_per_name > 0 else "—"
        cache_total = r.cache_read_tokens + r.cache_creation_tokens
        cache_pct = (
            f"{r.cache_read_tokens / cache_total * 100:.0f}%"
            if cache_total > 0
            else "—"
        )
        err_str = str(r.batch_errors) if r.batch_errors > 0 else "0"

        row_data = [
            r.model,
            str(n),
            valid_pct,
            fields_pct,
            ref_match,
            cost_str,
            speed_str,
            cpn_str,
            cache_pct,
            err_str,
        ]

        if has_quality:
            qual_str = f"{r.avg_quality_score:.2f}" if r.quality_scores else "—"
            gram_str = f"{r.avg_grammar_score:.1f}" if r.quality_scores else "—"
            sem_str = f"{r.avg_semantic_score:.1f}" if r.quality_scores else "—"
            conv_str = f"{r.avg_convention_score:.1f}" if r.quality_scores else "—"
            compl_str = f"{r.avg_completeness_score:.1f}" if r.quality_scores else "—"
            row_data.extend([qual_str, gram_str, sem_str, conv_str, compl_str])

        table.add_row(*row_data)

    console.print()
    console.print(table)

    # Quality distribution table (when reviewer was used)
    if has_quality:
        qual_table = Table(
            title="Quality Distribution",
            show_header=True,
            header_style="bold magenta",
        )
        qual_table.add_column("Model", style="bold")
        qual_table.add_column("Outstanding", justify="right")
        qual_table.add_column("Good", justify="right")
        qual_table.add_column("Inadequate", justify="right")
        qual_table.add_column("Poor", justify="right")

        for r in report.results:
            if r.quality_scores:
                dist = r.quality_distribution
                n_reviews = len(r.quality_scores)

                qual_table.add_row(
                    r.model,
                    f"{dist.get('outstanding', 0)} ({dist.get('outstanding', 0) / n_reviews * 100:.0f}%)"
                    if n_reviews
                    else "—",
                    f"{dist.get('good', 0)} ({dist.get('good', 0) / n_reviews * 100:.0f}%)"
                    if n_reviews
                    else "—",
                    f"{dist.get('inadequate', 0)} ({dist.get('inadequate', 0) / n_reviews * 100:.0f}%)"
                    if n_reviews
                    else "—",
                    f"{dist.get('poor', 0)} ({dist.get('poor', 0) / n_reviews * 100:.0f}%)"
                    if n_reviews
                    else "—",
                )

        console.print()
        console.print(qual_table)

    # Summary line
    reviewer_str = (
        f" | Reviewer: {report.config.reviewer_model}"
        if report.config.reviewer_model
        else ""
    )

    # Cost summary
    total_compose_cost = sum(r.total_cost for r in report.results)
    total_reviewer_cost = sum(r.reviewer_cost for r in report.results)
    total_cost = total_compose_cost + total_reviewer_cost
    total_compose_elapsed = sum(r.compose_elapsed_seconds for r in report.results)
    total_review_elapsed = sum(r.review_elapsed_seconds for r in report.results)
    total_elapsed = total_compose_elapsed + total_review_elapsed

    console.print("\n[bold]Cost Summary[/bold]")
    console.print(f"  Compose cost:  ${total_compose_cost:.4f}")
    console.print(f"  Reviewer cost: ${total_reviewer_cost:.4f}")
    console.print(f"  [bold]Total cost:  ${total_cost:.4f}[/bold]")
    console.print(
        f"  Compose time:  {total_compose_elapsed:.1f}s ({total_compose_elapsed / 60:.1f}min)"
    )
    console.print(
        f"  Review time:   {total_review_elapsed:.1f}s ({total_review_elapsed / 60:.1f}min)"
    )
    console.print(
        f"  Total wall-clock: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)"
    )
    console.print(f"  Models evaluated: {len(report.results)}")

    console.print(
        f"\n[dim]Extraction: {report.extraction_count} items | "
        f"Temperature: {report.config.temperature}{reviewer_str} | "
        f"Timestamp: {report.timestamp}[/dim]"
    )

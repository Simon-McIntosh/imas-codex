"""Benchmark runner for comparing LLM models on standard name generation.

Extracts a fixed dataset, runs it through multiple models, validates
output via grammar round-trip, and compares against a reference set.
Produces a :class:`BenchmarkReport` with per-model metrics suitable
for Rich table display and JSON export.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
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
    max_candidates: int = 54
    runs_per_model: int = 1
    temperature: float = 0.0
    reviewer_models: list[str] = field(default_factory=list)
    names_only: bool = True
    output_path: str | None = None
    # Legacy fields kept for backward-compatible deserialization
    source: str = "dd"
    ids_filter: str | None = None
    domain_filter: str | None = None
    facility: str | None = None
    force: bool = True
    reviewer_model: str | None = None  # legacy single-reviewer compat


@dataclass
class ReviewerScores:
    """Scores from a single reviewer model for one compose model's output."""

    reviewer_model: str
    target: str = "names"  # "names" or "docs"
    scores: list[dict] = field(default_factory=list)
    avg_score: float = 0.0
    distribution: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    elapsed_seconds: float = 0.0
    error: str = ""
    # Per-dimension averages (names: grammar/semantic/convention/completeness)
    # (docs: description_quality/documentation_quality/completeness/physics_accuracy)
    dim_averages: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelResult:
    """Results from running one model."""

    model: str
    status: str = "pending"  # pending | composed | reviewed | completed | failed
    error: str = ""
    compose_error: str = ""
    review_error: str = ""
    docs_error: str = ""
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
    attachment_count: int = 0
    vocab_gap_count: int = 0
    batch_errors: int = 0
    # Quality against reference set
    reference_overlap: int = 0
    reference_total: int = 0
    reference_precision: float = 0.0
    reference_recall: float = 0.0
    # Multi-reviewer results (new — matrix evaluation)
    name_reviewer_results: list[ReviewerScores] = field(default_factory=list)
    docs_reviewer_results: list[ReviewerScores] = field(default_factory=list)
    # Compose-time description scoring (per-judge ReviewerScores; the four
    # description dims live in each ReviewerScores.dim_averages — no flat
    # avg_*_score fields, to avoid collision with the docs physics_accuracy dim)
    description_reviewer_results: list[ReviewerScores] = field(default_factory=list)
    avg_description_score: float = 0.0
    description_reviewer_cost: float = 0.0
    description_review_elapsed_seconds: float = 0.0
    # Legacy single-reviewer fields (populated from first reviewer for compat)
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
    # Docs quality scoring
    docs_quality_scores: list[dict] = field(default_factory=list)
    docs_quality_distribution: dict[str, int] = field(default_factory=dict)
    avg_docs_quality_score: float = 0.0
    # Docs per-dimension averages
    avg_description_quality_score: float = 0.0
    avg_documentation_quality_score: float = 0.0
    avg_docs_completeness_score: float = 0.0
    avg_physics_accuracy_score: float = 0.0
    # Docs cost/timing
    docs_compose_cost: float = 0.0
    docs_compose_elapsed_seconds: float = 0.0
    docs_review_elapsed_seconds: float = 0.0
    docs_reviewer_cost: float = 0.0


@dataclass
class BenchmarkProvenance:
    """Version provenance for reproducibility."""

    codex_version: str = ""
    codex_commit: str = ""
    isn_version: str = ""
    dd_version: str = ""

    @classmethod
    def capture(cls) -> BenchmarkProvenance:
        """Capture current versions from installed packages and config."""
        codex_ver = codex_commit = isn_ver = dd_ver = ""
        try:
            from importlib.metadata import version

            codex_ver = version("imas-codex")
        except Exception:
            pass
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                codex_commit = result.stdout.strip()
        except Exception:
            pass
        try:
            from importlib.metadata import version

            isn_ver = version("imas-standard-names")
        except Exception:
            pass
        try:
            from imas_codex.settings import get_dd_version

            dd_ver = get_dd_version()
        except Exception:
            pass
        return cls(
            codex_version=codex_ver,
            codex_commit=codex_commit,
            isn_version=isn_ver,
            dd_version=dd_ver,
        )


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    config: BenchmarkConfig
    results: list[ModelResult]
    reference_names: list[str]
    extraction_count: int = 0
    extraction_source_ids: list[str] = field(default_factory=list)
    dataset_hash: str = ""
    timestamp: str = ""
    provenance: BenchmarkProvenance = field(default_factory=BenchmarkProvenance)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)

    def save_atomic(self, path: str) -> None:
        """Write report to *path* atomically (tmp + os.replace).

        Crash-safe: a partial write never corrupts the previous version.
        """
        import tempfile
        from pathlib import Path

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(dest.parent), suffix=".tmp", prefix=dest.stem
        )
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(self.to_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, str(dest))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def from_json(cls, data: str) -> BenchmarkReport:
        """Deserialize from JSON string."""
        raw = json.loads(data)
        # Filter unknown config keys for backward compatibility
        known_config_keys = {
            f.name for f in BenchmarkConfig.__dataclass_fields__.values()
        }
        config_data = {k: v for k, v in raw["config"].items() if k in known_config_keys}
        config = BenchmarkConfig(**config_data)

        results = []
        known_result_keys = {f.name for f in ModelResult.__dataclass_fields__.values()}
        for r in raw["results"]:
            # Deserialize nested ReviewerScores lists
            name_revs = [
                ReviewerScores(**rs) for rs in r.pop("name_reviewer_results", [])
            ]
            docs_revs = [
                ReviewerScores(**rs) for rs in r.pop("docs_reviewer_results", [])
            ]
            desc_revs = [
                ReviewerScores(**rs) for rs in r.pop("description_reviewer_results", [])
            ]
            filtered = {k: v for k, v in r.items() if k in known_result_keys}
            mr = ModelResult(
                **filtered,
                name_reviewer_results=name_revs,
                docs_reviewer_results=docs_revs,
                description_reviewer_results=desc_revs,
            )
            results.append(mr)

        prov_data = raw.get("provenance", {})
        provenance = (
            BenchmarkProvenance(**prov_data) if prov_data else BenchmarkProvenance()
        )
        return cls(
            config=config,
            results=results,
            reference_names=raw["reference_names"],
            extraction_count=raw.get("extraction_count", 0),
            extraction_source_ids=raw.get("extraction_source_ids", []),
            dataset_hash=raw.get("dataset_hash", ""),
            timestamp=raw.get("timestamp", ""),
            provenance=provenance,
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

    if target == "names":
        from imas_codex.standard_names.models import (
            StandardNameQualityReviewNameOnlyBatch,
        )

        response_model: type = StandardNameQualityReviewNameOnlyBatch
        system_template = "sn/review_names_system"
        user_template = "sn/review_names_user"
        dim_keys = ("grammar", "semantic", "convention", "completeness")
    elif target == "docs":
        from imas_codex.standard_names.models import (
            StandardNameQualityReviewDocsBatch,
        )

        response_model = StandardNameQualityReviewDocsBatch
        system_template = "sn/review_docs_system"
        user_template = "sn/review_docs_user"
        dim_keys = (
            "description_quality",
            "documentation_quality",
            "completeness",
            "physics_accuracy",
        )
    else:
        raise ValueError(
            f"Unknown review target {target!r}; expected 'names' or 'docs'."
        )

    # Get compose context (grammar enums + shared include variables)
    compose_ctx = build_compose_context()

    # Load K3 scored calibration examples from graph
    review_scored_examples: list[dict] = []
    review_axis = "name" if target == "names" else "docs"
    try:
        with GraphClient() as gc:
            review_scored_examples = load_review_examples(
                gc, physics_domains=[], axis=review_axis
            )
    except Exception as exc:
        logger.debug("Could not load review examples: %s", exc)

    # Build system prompt (static, cached across batches)
    system_context = {
        **compose_ctx,
        "review_scored_examples": review_scored_examples,
    }
    system_prompt = render_prompt(system_template, system_context)

    # Process in batches of 10
    all_reviews: list[dict] = []
    total_reviewer_cost: float = 0.0
    for i in range(0, len(candidates), 10):
        batch = candidates[i : i + 10]

        # Build per-candidate items with neighbour context
        batch_items = []
        for c in batch:
            name = _resolve_name(c)
            if target == "docs":
                item: dict[str, Any] = {
                    "id": name,
                    "standard_name": name,
                    "description": c.get("docs_description", c.get("description", "")),
                    "documentation": c.get("documentation", ""),
                    "unit": c.get("unit", "N/A"),
                    "kind": c.get("kind", "N/A"),
                    "source_paths": c.get("source_paths", []),
                }
            else:
                item = {
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

        # Build user prompt — docs template expects single `item`, names
        # template iterates over `items` list.
        if target == "docs":
            # Single-item template: render and call per candidate
            for item_ctx in batch_items:
                user_context = {
                    **compose_ctx,
                    "review_scored_examples": review_scored_examples,
                    "item": item_ctx,
                }
                try:
                    user_prompt = render_prompt(user_template, user_context)
                except Exception as exc:
                    logger.warning(
                        "Failed to render docs review prompt for %s: %s",
                        item_ctx.get("id", "?"),
                        exc,
                    )
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
                        }
                        for dim in dim_keys:
                            review_dict[f"{dim}_score"] = getattr(r.scores, dim, 0)
                        if hasattr(r, "reasoning"):
                            review_dict["reasoning"] = r.reasoning
                        all_reviews.append(review_dict)
                except Exception as e:
                    logger.warning(
                        "Docs review failed for %s: %s", item_ctx.get("id", "?"), e
                    )
        else:
            # Batch template: render with items list
            user_context = {
                **compose_ctx,
                "review_scored_examples": review_scored_examples,
                "items": batch_items,
                "existing_names": [],
                "batch_context": "",
                "nearby_existing_names": [],
                "audit_findings": [],
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
                user_prompt = render_prompt(user_template, user_context)
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
                    }
                    for dim in dim_keys:
                        review_dict[f"{dim}_score"] = getattr(r.scores, dim, 0)
                    if hasattr(r, "reasoning"):
                        review_dict["reasoning"] = r.reasoning
                    all_reviews.append(review_dict)
            except Exception as e:
                logger.warning("Reviewer scoring failed for batch: %s", e)

    return all_reviews, total_reviewer_cost


async def score_descriptions(
    candidates: list[dict],
    reviewer_model: str,
) -> tuple[list[dict], float]:
    """Score each candidate's SHORT compose-time ``description``.

    Parallel in style to :func:`score_with_reviewer` but scores the
    one-line compose-time ``description`` (NOT ``docs_description``) on a
    4-dimensional 0–20 rubric: physics_accuracy, specificity, consistency,
    concision. Unlike the names review, this pass is **self-contained** — it
    scores against the candidate's own DD context (unit, kind, source_paths,
    physics_domain) and its companion name, with no graph-neighbour fetch.
    That keeps the standalone ``--rescore`` path offline.

    Returns (reviews, total_cost) where each review dict carries:
    ``name``, ``quality_tier``, ``score``, ``reasoning`` and the four
    per-dimension ``<dim>_score`` keys.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewDescriptionBatch,
    )

    dim_keys = ("physics_accuracy", "specificity", "consistency", "concision")

    # Static system prompt (cached across batches via schema_needs injection).
    system_prompt = render_prompt("sn/review_description_system", {})

    all_reviews: list[dict] = []
    total_cost: float = 0.0
    for i in range(0, len(candidates), 10):
        batch = candidates[i : i + 10]
        batch_items = []
        for c in batch:
            name = _resolve_name(c)
            batch_items.append(
                {
                    "standard_name": name,
                    "source_id": c.get("source_id", ""),
                    "description": c.get("description", "") or "",
                    "unit": c.get("unit", "N/A"),
                    "kind": c.get("kind", "N/A"),
                    "physics_domain": c.get("physics_domain", ""),
                    "source_paths": c.get("source_paths", []),
                }
            )

        try:
            user_prompt = render_prompt(
                "sn/review_description_user", {"items": batch_items}
            )
        except Exception as exc:
            logger.warning("Failed to render description review prompt: %s", exc)
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result, cost, _ = await acall_llm_structured(
                model=reviewer_model,
                messages=messages,
                response_model=StandardNameQualityReviewDescriptionBatch,
                service="standard-names",
            )
            total_cost += cost
            for r in result.reviews:
                review_dict: dict[str, Any] = {
                    "name": r.standard_name,
                    "quality_tier": r.scores.tier,
                    "score": r.scores.score,
                    "reasoning": r.reasoning,
                }
                for dim in dim_keys:
                    review_dict[f"{dim}_score"] = getattr(r.scores, dim, 0)
                all_reviews.append(review_dict)
        except Exception as e:
            logger.warning("Description scoring failed for batch: %s", e)

    return all_reviews, total_cost


# ---------------------------------------------------------------------------
# Docs generation for benchmark
# ---------------------------------------------------------------------------


async def generate_docs_for_candidates(
    candidates: list[dict],
    model: str,
    context: dict[str, Any],
) -> tuple[list[dict], float, float]:
    """Generate documentation for benchmark candidates.

    Returns (updated_candidates, total_cost, elapsed_seconds).
    Each candidate dict is updated with 'docs_description' and 'documentation' keys.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.models import GeneratedDocs

    system_prompt = render_prompt("sn/generate_docs_system", context)
    total_cost = 0.0
    t0 = time.monotonic()

    for c in candidates:
        name = _resolve_name(c)

        # Build item context for the docs generation prompt
        item = {
            "name": name,
            "unit": c.get("unit", ""),
            "kind": c.get("kind", "scalar"),
            "physics_domain": c.get("physics_domain", ""),
            "description": c.get("description", ""),
            "source_paths": c.get("source_paths", []),
            # Minimal context — no enrichment for benchmark simplicity
            "reviewer_score_name": None,
            "reviewer_comments_name": "",
            "chain_history": [],
        }

        try:
            user_prompt = render_prompt(
                "sn/generate_docs_user", {**context, "item": item}
            )
        except Exception as exc:
            logger.warning("Failed to render docs prompt for %s: %s", name, exc)
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # GPT-5.x models don't support temperature=0.0
        temp = 0.0
        if "gpt-5" in model:
            temp = None  # type: ignore[assignment]

        try:
            result, cost, _ = await acall_llm_structured(
                model=model,
                messages=messages,
                response_model=GeneratedDocs,
                temperature=temp,
                service="standard-names",
            )
            total_cost += cost
            c["docs_description"] = result.description
            c["documentation"] = result.documentation
        except Exception as exc:
            logger.warning(
                "Docs generation failed for %s with %s: %s", name, model, exc
            )
            c["docs_description"] = ""
            c["documentation"] = ""

    elapsed = time.monotonic() - t0
    return candidates, total_cost, round(elapsed, 2)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    config: BenchmarkConfig,
    extraction_batches: list[dict] | None = None,
) -> BenchmarkReport:
    """Run the benchmark across all configured models.

    Each model is processed as a complete unit (compose → review → optional
    docs) before moving to the next.  After each model completes (or fails),
    an incremental report is saved to ``config.output_path``.  If the
    process is killed, the report contains all fully-processed models.

    Args:
        config: Benchmark configuration.
        extraction_batches: Pre-extracted candidate batches (list of dicts
            with items grouped by IDS). If None, extracts from graph.

    Returns:
        BenchmarkReport with per-model results.
    """
    from imas_codex.standard_names.benchmark_reference import REFERENCE_NAMES

    # --- Fail-fast: at least one reviewer model is required ---
    reviewer_models = config.reviewer_models
    if not reviewer_models:
        # Backward compat: single reviewer_model → list of one
        if config.reviewer_model:
            reviewer_models = [config.reviewer_model]
        else:
            raise ValueError(
                "reviewer_models is required for benchmarking. "
                "Reviewer scores are the quality metric."
            )

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
        extraction_batches = _rebuild_batches(extraction_batches, config.max_candidates)

    logger.info(
        "Benchmark: %d extraction items across %d batches",
        len(all_items),
        len(extraction_batches),
    )

    # Record source IDs for cross-model identity assertion
    extraction_source_ids = [
        item.get("source_id", item.get("id", "")) for item in all_items
    ]

    # Compute deterministic dataset hash from ordered source IDs
    dataset_hash = hashlib.sha256(
        "\n".join(sorted(extraction_source_ids)).encode()
    ).hexdigest()[:16]

    # --- Pre-build shared prompt context (reused across models) ---
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_compose_context

    context = build_compose_context()
    context["compose_scored_examples"] = []
    system_prompt = render_prompt("sn/generate_name_system", context)

    provenance = BenchmarkProvenance.capture()
    reference_names_list = list(REFERENCE_NAMES.keys())

    def _build_partial_report(results_so_far: list[ModelResult]) -> BenchmarkReport:
        return BenchmarkReport(
            config=config,
            results=results_so_far,
            reference_names=reference_names_list,
            extraction_count=len(all_items),
            extraction_source_ids=extraction_source_ids,
            dataset_hash=dataset_hash,
            timestamp=datetime.now(tz=UTC).isoformat(),
            provenance=provenance,
        )

    def _save_incremental(results_so_far: list[ModelResult]) -> None:
        if not config.output_path:
            return
        try:
            report = _build_partial_report(results_so_far)
            report.save_atomic(config.output_path)
            logger.info(
                "Incremental save: %d/%d models → %s",
                len(results_so_far),
                len(config.models),
                config.output_path,
            )
        except Exception as exc:
            logger.error("Failed to save incremental report: %s", exc)

    # --- 2. Per-model loop: compose → review → (docs) → save ---
    results: list[ModelResult] = []

    for model_idx, model in enumerate(config.models):
        logger.info(
            "Benchmarking model %d/%d: %s",
            model_idx + 1,
            len(config.models),
            model,
        )

        # --- 2a. Compose phase ---
        try:
            model_result = await _run_model(
                model=model,
                extraction_batches=extraction_batches,
                config=config,
                reference=REFERENCE_NAMES,
                system_prompt=system_prompt,
                context=context,
            )
            model_result.compose_elapsed_seconds = model_result.elapsed_seconds
            model_result.status = "composed"
        except Exception as exc:
            logger.error("Model %s compose failed: %s", model, exc)
            model_result = ModelResult(
                model=model, status="failed", compose_error=str(exc), error=str(exc)
            )
            results.append(model_result)
            _save_incremental(results)
            continue

        # --- 2b. Review names phase (all reviewer models in parallel) ---
        if model_result.candidates:

            async def _review_names(
                rev_model: str,
                _candidates: list = model_result.candidates,
                _model: str = model,
            ) -> ReviewerScores:
                rs = ReviewerScores(reviewer_model=rev_model, target="names")
                try:
                    t0 = time.monotonic()
                    revs, cost = await score_with_reviewer(
                        _candidates, rev_model, target="names"
                    )
                    rs.elapsed_seconds = round(time.monotonic() - t0, 2)
                    rs.cost = cost
                    rs.scores = revs
                    _apply_reviewer_scores_metrics(rs, revs, "names")
                    logger.info(
                        "  ✓ names review by %s: avg=%.3f, $%.4f, %.1fs",
                        rev_model.split("/")[-1],
                        rs.avg_score,
                        cost,
                        rs.elapsed_seconds,
                    )
                except Exception as exc:
                    logger.error(
                        "Model %s name review by %s failed: %s",
                        _model,
                        rev_model,
                        exc,
                    )
                    rs.error = str(exc)
                return rs

            name_review_tasks = [_review_names(rm) for rm in reviewer_models]
            model_result.name_reviewer_results = list(
                await asyncio.gather(*name_review_tasks)
            )

            # Populate legacy single-reviewer fields from first successful reviewer
            first_ok = next(
                (r for r in model_result.name_reviewer_results if not r.error), None
            )
            if first_ok:
                model_result.quality_scores = first_ok.scores
                model_result.reviewer_cost = sum(
                    r.cost for r in model_result.name_reviewer_results
                )
                model_result.review_elapsed_seconds = sum(
                    r.elapsed_seconds for r in model_result.name_reviewer_results
                )
                _apply_review_metrics(model_result, first_ok.scores)
                model_result.status = "reviewed"
            else:
                model_result.review_error = "All reviewer models failed for names"

            # --- 2b'. Description scoring phase (same reviewer matrix) ---
            async def _review_descriptions(
                rev_model: str,
                _candidates: list = model_result.candidates,
                _model: str = model,
            ) -> ReviewerScores:
                rs = ReviewerScores(reviewer_model=rev_model, target="descriptions")
                try:
                    t0 = time.monotonic()
                    revs, cost = await score_descriptions(_candidates, rev_model)
                    rs.elapsed_seconds = round(time.monotonic() - t0, 2)
                    rs.cost = cost
                    rs.scores = revs
                    _apply_reviewer_scores_metrics(rs, revs, "descriptions")
                    logger.info(
                        "  ✓ description review by %s: avg=%.3f, $%.4f, %.1fs",
                        rev_model.split("/")[-1],
                        rs.avg_score,
                        cost,
                        rs.elapsed_seconds,
                    )
                except Exception as exc:
                    logger.error(
                        "Model %s description review by %s failed: %s",
                        _model,
                        rev_model,
                        exc,
                    )
                    rs.error = str(exc)
                return rs

            desc_review_tasks = [_review_descriptions(rm) for rm in reviewer_models]
            model_result.description_reviewer_results = list(
                await asyncio.gather(*desc_review_tasks)
            )
            _apply_description_aggregate(model_result)

        # --- 2c. Docs phase (when not names-only) ---
        if not config.names_only:
            try:
                valid_candidates = [
                    c for c in model_result.candidates if validate_candidate(c)[0]
                ]
                if valid_candidates:
                    # Generate docs (once per compose model — shared across reviewers)
                    _, docs_cost, docs_elapsed = await generate_docs_for_candidates(
                        valid_candidates, model, context
                    )
                    model_result.docs_compose_cost = docs_cost
                    model_result.docs_compose_elapsed_seconds = docs_elapsed

                    # Review docs with all reviewer models in parallel
                    async def _review_docs(
                        rev_model: str,
                        _candidates: list = valid_candidates,
                        _model: str = model,
                    ) -> ReviewerScores:
                        rs = ReviewerScores(reviewer_model=rev_model, target="docs")
                        try:
                            t0 = time.monotonic()
                            dr, dc = await score_with_reviewer(
                                _candidates, rev_model, target="docs"
                            )
                            rs.elapsed_seconds = round(time.monotonic() - t0, 2)
                            rs.cost = dc
                            rs.scores = dr
                            _apply_reviewer_scores_metrics(rs, dr, "docs")
                            logger.info(
                                "  ✓ docs review by %s: avg=%.3f, $%.4f, %.1fs",
                                rev_model.split("/")[-1],
                                rs.avg_score,
                                dc,
                                rs.elapsed_seconds,
                            )
                        except Exception as exc:
                            logger.error(
                                "Model %s docs review by %s failed: %s",
                                _model,
                                rev_model,
                                exc,
                            )
                            rs.error = str(exc)
                        return rs

                    docs_review_tasks = [_review_docs(rm) for rm in reviewer_models]
                    model_result.docs_reviewer_results = list(
                        await asyncio.gather(*docs_review_tasks)
                    )

                    # Populate legacy fields from first successful reviewer
                    first_docs_ok = next(
                        (r for r in model_result.docs_reviewer_results if not r.error),
                        None,
                    )
                    if first_docs_ok:
                        model_result.docs_quality_scores = first_docs_ok.scores
                        model_result.docs_reviewer_cost = sum(
                            r.cost for r in model_result.docs_reviewer_results
                        )
                        model_result.docs_review_elapsed_seconds = sum(
                            r.elapsed_seconds
                            for r in model_result.docs_reviewer_results
                        )
                        _apply_docs_review_metrics(model_result, first_docs_ok.scores)
            except Exception as exc:
                logger.error("Model %s docs phase failed: %s", model, exc)
                model_result.docs_error = str(exc)

        # Mark final status
        if not model_result.compose_error:
            model_result.status = "completed"
        model_result.error = " | ".join(
            filter(
                None,
                [
                    model_result.compose_error,
                    model_result.review_error,
                    model_result.docs_error,
                ],
            )
        )

        results.append(model_result)
        _save_incremental(results)

    return _build_partial_report(results)


def _apply_reviewer_scores_metrics(
    rev: ReviewerScores, reviews: list[dict], target: str
) -> None:
    """Compute and set metrics on a ReviewerScores instance in-place."""
    if not reviews:
        return

    # Average score
    rev.avg_score = sum(r.get("score", 0) for r in reviews) / len(reviews)

    # Tier distribution
    for r in reviews:
        tier = r.get("quality_tier", "unknown")
        rev.distribution[tier] = rev.distribution.get(tier, 0) + 1

    # Per-dimension averages
    if target == "names":
        dims = ("grammar", "semantic", "convention", "completeness")
    elif target == "descriptions":
        dims = ("physics_accuracy", "specificity", "consistency", "concision")
    else:
        dims = (
            "description_quality",
            "documentation_quality",
            "completeness",
            "physics_accuracy",
        )
    for dim in dims:
        key = f"{dim}_score"
        vals = [r[key] for r in reviews if key in r and r[key] > 0]
        if vals:
            rev.dim_averages[dim] = round(sum(vals) / len(vals), 2)


def _apply_review_metrics(result: ModelResult, reviews: list[dict]) -> None:
    """Compute and set name-review metrics on *result* in-place."""
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

    # Tier distribution
    for r in reviews:
        tier = r.get("quality_tier", "unknown")
        result.quality_distribution[tier] = result.quality_distribution.get(tier, 0) + 1

    if reviews:
        result.avg_quality_score = sum(r.get("score", 0) for r in reviews) / len(
            reviews
        )
        for dim in ("grammar", "semantic", "convention", "completeness"):
            key = f"{dim}_score"
            vals = [r[key] for r in reviews if key in r and r[key] > 0]
            if vals:
                setattr(result, f"avg_{key}", sum(vals) / len(vals))

    # Doc-length and field-coverage (name-phase metadata)
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
        populated = sum(
            1 for f in all_fields if c.get(f) or (c.get("segments") or {}).get(f)
        )
        field_counts.append(populated / len(all_fields))
    result.avg_fields_populated = (
        sum(field_counts) / len(field_counts) if field_counts else 0.0
    )


def _apply_description_aggregate(result: ModelResult) -> None:
    """Aggregate compose-time description scores across judges in-place.

    Sets ``avg_description_score`` to the mean over successful judges, plus
    the summed reviewer cost and elapsed time. Per-dimension averages remain
    on each judge's ``ReviewerScores.dim_averages``.
    """
    judges = result.description_reviewer_results
    ok = [r for r in judges if not r.error]
    if ok:
        result.avg_description_score = round(sum(r.avg_score for r in ok) / len(ok), 4)
    result.description_reviewer_cost = sum(r.cost for r in judges)
    result.description_review_elapsed_seconds = sum(r.elapsed_seconds for r in judges)


def _apply_docs_review_metrics(result: ModelResult, docs_reviews: list[dict]) -> None:
    """Compute and set docs-review metrics on *result* in-place."""
    for r in docs_reviews:
        tier = r.get("quality_tier", "unknown")
        result.docs_quality_distribution[tier] = (
            result.docs_quality_distribution.get(tier, 0) + 1
        )
    if docs_reviews:
        result.avg_docs_quality_score = sum(
            r.get("score", 0) for r in docs_reviews
        ) / len(docs_reviews)
        for dim in (
            "description_quality",
            "documentation_quality",
            "completeness",
            "physics_accuracy",
        ):
            key = f"{dim}_score"
            vals = [r[key] for r in docs_reviews if key in r and r[key] > 0]
            if vals:
                setattr(result, f"avg_{key}", sum(vals) / len(vals))


def _extract_candidates(config: BenchmarkConfig) -> list[dict]:
    """Extract candidates using the fixed reference dataset.

    Uses REFERENCE_NAMES paths as explicit inputs, enriched through the
    same production pipeline (graph context, clustering, batching).
    This ensures benchmark reproducibility — same paths every run.

    Returns list of batch dicts with keys: group_key, items, existing_names, context.
    """
    from imas_codex.standard_names.benchmark_reference import REFERENCE_NAMES
    from imas_codex.standard_names.sources.dd import extract_dd_candidates

    # Select reference paths, capped by max_candidates
    reference_paths = list(REFERENCE_NAMES.keys())
    if config.max_candidates and len(reference_paths) > config.max_candidates:
        reference_paths = reference_paths[: config.max_candidates]

    logger.info(
        "Benchmark extraction: %d/%d reference paths",
        len(reference_paths),
        len(REFERENCE_NAMES),
    )

    batches = extract_dd_candidates(
        explicit_paths=reference_paths,
        force=True,  # always force — benchmark paths are pre-curated
        write_skipped=False,  # no graph side effects
    )

    # Fail fast if paths are missing from the graph
    extracted_paths = set()
    for batch in batches:
        for item in batch.items:
            extracted_paths.add(item.get("path", item.get("source_id", "")))
    missing = set(reference_paths) - extracted_paths
    if missing:
        logger.warning(
            "Benchmark: %d/%d reference paths missing from graph: %s",
            len(missing),
            len(reference_paths),
            sorted(missing)[:5],
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

                # Collect candidates — merge extraction context
                for c_idx, c in enumerate(llm_result.candidates):
                    candidate = c.model_dump()
                    # Compose the standard name from segments
                    try:
                        segs = _get_segment_fields(candidate)
                        candidate["standard_name"] = compose_standard_name(segs)
                    except Exception:
                        candidate["standard_name"] = ""
                    # Carry forward extraction context for docs generation
                    if c_idx < len(items):
                        src = items[c_idx]
                        for key in ("source_paths", "physics_domain", "unit"):
                            if key in src and key not in candidate:
                                candidate[key] = src[key]
                    all_candidates.append(candidate)
                result.skipped_count += len(llm_result.skipped)
                result.attachment_count += len(llm_result.attachments)
                result.vocab_gap_count += len(getattr(llm_result, "vocab_gaps", []))

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


async def rescore_descriptions_report(
    report: BenchmarkReport,
) -> BenchmarkReport:
    """Run ONLY the compose-time description-scoring pass on an existing report.

    Loads the stored candidates from each :class:`ModelResult` (they carry the
    compose-time ``description``), scores them with every reviewer model in
    ``report.config.reviewer_models`` (each judge independently), and merges
    the new ``description_reviewer_results`` + ``avg_description_score`` back
    onto the report in-place. Does NOT re-run composition or names review.

    Returns the same (mutated) report for convenience.
    """
    reviewer_models = report.config.reviewer_models
    if not reviewer_models and report.config.reviewer_model:
        reviewer_models = [report.config.reviewer_model]
    if not reviewer_models:
        raise ValueError(
            "No reviewer_models on the report config; cannot rescore descriptions."
        )

    for model_result in report.results:
        if not model_result.candidates:
            continue

        async def _review_descriptions(
            rev_model: str,
            _candidates: list = model_result.candidates,
            _model: str = model_result.model,
        ) -> ReviewerScores:
            rs = ReviewerScores(reviewer_model=rev_model, target="descriptions")
            try:
                t0 = time.monotonic()
                revs, cost = await score_descriptions(_candidates, rev_model)
                rs.elapsed_seconds = round(time.monotonic() - t0, 2)
                rs.cost = cost
                rs.scores = revs
                _apply_reviewer_scores_metrics(rs, revs, "descriptions")
                logger.info(
                    "  ✓ [rescore] description review by %s for %s: avg=%.3f, $%.4f",
                    rev_model.split("/")[-1],
                    _model,
                    rs.avg_score,
                    cost,
                )
            except Exception as exc:
                logger.error(
                    "[rescore] %s description review by %s failed: %s",
                    _model,
                    rev_model,
                    exc,
                )
                rs.error = str(exc)
            return rs

        tasks = [_review_descriptions(rm) for rm in reviewer_models]
        model_result.description_reviewer_results = list(await asyncio.gather(*tasks))
        _apply_description_aggregate(model_result)

    return report


# ---------------------------------------------------------------------------
# Rich table rendering
# ---------------------------------------------------------------------------


def render_comparison_table(report: BenchmarkReport) -> None:
    """Render Rich comparison tables to stdout.

    Shows compose metrics, quality scores per reviewer (matrix when
    multiple reviewers), and cost/timing summary.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # --- Compose metrics table ---
    has_quality = any(r.quality_scores for r in report.results)
    has_multi_rev = any(len(r.name_reviewer_results) > 1 for r in report.results)
    has_desc = any(r.description_reviewer_results for r in report.results)

    table = Table(
        title="SN Benchmark — Compose Results",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Model", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Names", justify="right")
    table.add_column("Skip/Att/Gap", justify="right")
    table.add_column("Valid %", justify="right")
    table.add_column("Fields %", justify="right")
    table.add_column("Ref Match", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Names/min", justify="right")
    table.add_column("$/name", justify="right")
    table.add_column("Cache %", justify="right")
    table.add_column("Errors", justify="right")
    if has_desc:
        table.add_column("Desc", justify="right")
    if has_quality and not has_multi_rev:
        # Single-reviewer: inline scores in compose table
        table.add_column("Avg Quality", justify="right")
        table.add_column("Gram", justify="right")
        table.add_column("Sem", justify="right")
        table.add_column("Conv", justify="right")
        table.add_column("Compl", justify="right")

    for r in report.results:
        n = len(r.candidates)
        status_colours = {
            "completed": "green",
            "composed": "yellow",
            "reviewed": "cyan",
            "failed": "red",
            "pending": "dim",
        }
        colour = status_colours.get(r.status, "white")
        status_str = f"[{colour}]{r.status}[/{colour}]"
        sag_str = f"{r.skipped_count}/{r.attachment_count}/{r.vocab_gap_count}"
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
            status_str,
            str(n),
            sag_str,
            valid_pct,
            fields_pct,
            ref_match,
            cost_str,
            speed_str,
            cpn_str,
            cache_pct,
            err_str,
        ]

        if has_desc:
            desc_str = (
                f"{r.avg_description_score:.2f}"
                if r.description_reviewer_results
                and any(not jr.error for jr in r.description_reviewer_results)
                else "—"
            )
            row_data.append(desc_str)

        if has_quality and not has_multi_rev:
            qual_str = f"{r.avg_quality_score:.2f}" if r.quality_scores else "—"
            gram_str = f"{r.avg_grammar_score:.1f}" if r.quality_scores else "—"
            sem_str = f"{r.avg_semantic_score:.1f}" if r.quality_scores else "—"
            conv_str = f"{r.avg_convention_score:.1f}" if r.quality_scores else "—"
            compl_str = f"{r.avg_completeness_score:.1f}" if r.quality_scores else "—"
            row_data.extend([qual_str, gram_str, sem_str, conv_str, compl_str])

        table.add_row(*row_data)

    console.print()
    console.print(table)

    # --- Multi-reviewer names quality matrix ---
    if has_multi_rev:
        _render_reviewer_matrix(console, report, "names")

    # --- Single-reviewer quality distribution (legacy) ---
    elif has_quality:
        _render_quality_distribution(console, report)

    # --- Docs quality ---
    has_docs = any(r.docs_quality_scores for r in report.results)
    has_multi_docs_rev = any(len(r.docs_reviewer_results) > 1 for r in report.results)
    if has_multi_docs_rev:
        _render_reviewer_matrix(console, report, "docs")
    elif has_docs:
        _render_docs_table(console, report)

    # --- Reviewer agreement (when multiple reviewers) ---
    if has_multi_rev:
        _render_reviewer_agreement(console, report, "names")
    if has_multi_docs_rev:
        _render_reviewer_agreement(console, report, "docs")

    # --- Cost summary ---
    _render_cost_summary(console, report)


def _render_reviewer_matrix(console: Any, report: BenchmarkReport, target: str) -> None:
    """Render a compose_model × reviewer_model quality matrix."""
    from rich.table import Table

    title = "Names" if target == "names" else "Docs"
    results_attr = (
        "name_reviewer_results" if target == "names" else "docs_reviewer_results"
    )

    # Collect all reviewer models across all results
    all_reviewers: list[str] = []
    seen: set[str] = set()
    for r in report.results:
        for rv in getattr(r, results_attr, []):
            if rv.reviewer_model not in seen:
                all_reviewers.append(rv.reviewer_model)
                seen.add(rv.reviewer_model)

    if not all_reviewers:
        return

    dims = (
        ("grammar", "semantic", "convention", "completeness")
        if target == "names"
        else (
            "description_quality",
            "documentation_quality",
            "completeness",
            "physics_accuracy",
        )
    )
    dim_short = {
        "grammar": "Gr",
        "semantic": "Se",
        "convention": "Cv",
        "completeness": "Cp",
        "description_quality": "Desc",
        "documentation_quality": "Doc",
        "physics_accuracy": "Phys",
    }

    table = Table(
        title=f"{title} Quality Matrix (compose × reviewer)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Compose Model", style="bold")
    for rev in all_reviewers:
        short_rev = rev.split("/")[-1]
        table.add_column(f"{short_rev}\nAvg", justify="right")
        for d in dims:
            table.add_column(dim_short.get(d, d[:4]), justify="right")
        table.add_column("Cost", justify="right")

    for r in report.results:
        rev_map = {rv.reviewer_model: rv for rv in getattr(r, results_attr, [])}
        row: list[str] = [r.model]
        for rev in all_reviewers:
            rv = rev_map.get(rev)
            if rv and not rv.error:
                row.append(f"{rv.avg_score:.3f}")
                for d in dims:
                    val = rv.dim_averages.get(d, 0)
                    row.append(f"{val:.1f}" if val else "—")
                row.append(f"${rv.cost:.3f}")
            else:
                error_str = rv.error[:20] if rv and rv.error else "—"
                row.extend([error_str] + ["—"] * len(dims) + ["—"])
        table.add_row(*row)

    console.print()
    console.print(table)


def _render_reviewer_agreement(
    console: Any, report: BenchmarkReport, target: str
) -> None:
    """Show pairwise rank correlation between reviewers."""
    from rich.table import Table

    results_attr = (
        "name_reviewer_results" if target == "names" else "docs_reviewer_results"
    )
    title = "Names" if target == "names" else "Docs"

    # Build reviewer → compose_model → avg_score mapping
    reviewer_scores: dict[str, dict[str, float]] = {}
    for r in report.results:
        for rv in getattr(r, results_attr, []):
            if rv.error:
                continue
            reviewer_scores.setdefault(rv.reviewer_model, {})[r.model] = rv.avg_score

    reviewers = list(reviewer_scores.keys())
    if len(reviewers) < 2:
        return

    # Compute Spearman rank correlation for each pair
    table = Table(
        title=f"{title} Reviewer Agreement (Spearman ρ)",
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Reviewer", style="bold")
    for rev in reviewers:
        table.add_column(rev.split("/")[-1], justify="right")

    for rev_a in reviewers:
        row = [rev_a.split("/")[-1]]
        for rev_b in reviewers:
            if rev_a == rev_b:
                row.append("1.000")
                continue
            # Find shared compose models
            shared = sorted(set(reviewer_scores[rev_a]) & set(reviewer_scores[rev_b]))
            if len(shared) < 3:
                row.append("—")
                continue
            # Spearman rank correlation
            scores_a = [reviewer_scores[rev_a][m] for m in shared]
            scores_b = [reviewer_scores[rev_b][m] for m in shared]
            rho = _spearman_rho(scores_a, scores_b)
            row.append(f"{rho:.3f}")
        table.add_row(*row)

    console.print()
    console.print(table)


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda iv: iv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True))
    return 1.0 - 6.0 * d_sq / (n * (n * n - 1))


def _render_quality_distribution(console: Any, report: BenchmarkReport) -> None:
    """Render single-reviewer quality distribution table."""
    from rich.table import Table

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
                _dist_cell(dist, "outstanding", n_reviews),
                _dist_cell(dist, "good", n_reviews),
                _dist_cell(dist, "inadequate", n_reviews),
                _dist_cell(dist, "poor", n_reviews),
            )

    console.print()
    console.print(qual_table)


def _dist_cell(dist: dict[str, int], tier: str, total: int) -> str:
    count = dist.get(tier, 0)
    if total <= 0:
        return "—"
    return f"{count} ({count / total * 100:.0f}%)"


def _render_docs_table(console: Any, report: BenchmarkReport) -> None:
    """Render single-reviewer docs quality table."""
    from rich.table import Table

    docs_table = Table(
        title="Docs Quality Results",
        show_header=True,
        header_style="bold green",
    )
    docs_table.add_column("Model", style="bold")
    docs_table.add_column("Docs Scored", justify="right")
    docs_table.add_column("Avg Docs Q", justify="right")
    docs_table.add_column("Desc", justify="right")
    docs_table.add_column("Docs", justify="right")
    docs_table.add_column("Compl", justify="right")
    docs_table.add_column("Physics", justify="right")
    docs_table.add_column("Docs Cost", justify="right")

    for r in report.results:
        if r.docs_quality_scores:
            n_docs = len(r.docs_quality_scores)
            docs_table.add_row(
                r.model,
                str(n_docs),
                f"{r.avg_docs_quality_score:.2f}",
                f"{r.avg_description_quality_score:.1f}",
                f"{r.avg_documentation_quality_score:.1f}",
                f"{r.avg_docs_completeness_score:.1f}",
                f"{r.avg_physics_accuracy_score:.1f}",
                f"${r.docs_compose_cost + r.docs_reviewer_cost:.4f}",
            )
        else:
            docs_table.add_row(r.model, "—", "—", "—", "—", "—", "—", "—")

    console.print()
    console.print(docs_table)


def _render_cost_summary(console: Any, report: BenchmarkReport) -> None:
    """Render cost and timing summary."""
    # Determine reviewer label
    reviewer_models = report.config.reviewer_models
    if not reviewer_models and report.config.reviewer_model:
        reviewer_models = [report.config.reviewer_model]
    reviewer_str = (
        f" | Reviewers: {', '.join(m.split('/')[-1] for m in reviewer_models)}"
        if reviewer_models
        else ""
    )

    total_compose_cost = sum(r.total_cost for r in report.results)
    total_reviewer_cost = sum(r.reviewer_cost for r in report.results)
    total_docs_compose_cost = sum(r.docs_compose_cost for r in report.results)
    total_docs_reviewer_cost = sum(r.docs_reviewer_cost for r in report.results)
    total_cost = (
        total_compose_cost
        + total_reviewer_cost
        + total_docs_compose_cost
        + total_docs_reviewer_cost
    )
    total_compose_elapsed = sum(r.compose_elapsed_seconds for r in report.results)
    total_review_elapsed = sum(r.review_elapsed_seconds for r in report.results)
    total_docs_compose_elapsed = sum(
        r.docs_compose_elapsed_seconds for r in report.results
    )
    total_docs_review_elapsed = sum(
        r.docs_review_elapsed_seconds for r in report.results
    )
    total_elapsed = (
        total_compose_elapsed
        + total_review_elapsed
        + total_docs_compose_elapsed
        + total_docs_review_elapsed
    )

    has_docs = any(r.docs_quality_scores for r in report.results)

    console.print("\n[bold]Cost Summary[/bold]")
    console.print(f"  Compose cost:  ${total_compose_cost:.4f}")
    console.print(f"  Reviewer cost: ${total_reviewer_cost:.4f}")
    if has_docs:
        console.print(f"  Docs compose cost:  ${total_docs_compose_cost:.4f}")
        console.print(f"  Docs reviewer cost: ${total_docs_reviewer_cost:.4f}")
    console.print(f"  [bold]Total cost:  ${total_cost:.4f}[/bold]")
    console.print(
        f"  Compose time:  {total_compose_elapsed:.1f}s ({total_compose_elapsed / 60:.1f}min)"
    )
    console.print(
        f"  Review time:   {total_review_elapsed:.1f}s ({total_review_elapsed / 60:.1f}min)"
    )
    if has_docs:
        console.print(
            f"  Docs compose time:  {total_docs_compose_elapsed:.1f}s ({total_docs_compose_elapsed / 60:.1f}min)"
        )
        console.print(
            f"  Docs review time:   {total_docs_review_elapsed:.1f}s ({total_docs_review_elapsed / 60:.1f}min)"
        )
    console.print(
        f"  Total wall-clock: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)"
    )
    console.print(f"  Models evaluated: {len(report.results)}")
    completed = sum(1 for r in report.results if r.status == "completed")
    failed = sum(1 for r in report.results if r.status == "failed")
    partial = len(report.results) - completed - failed
    parts = [f"{completed} completed"]
    if partial:
        parts.append(f"{partial} partial")
    if failed:
        parts.append(f"{failed} failed")
    console.print(f"  Status: {', '.join(parts)}")

    prov = report.provenance
    prov_str = ""
    if prov.codex_version or prov.codex_commit:
        prov_str = (
            f" | codex={prov.codex_version}@{prov.codex_commit}"
            f" ISN={prov.isn_version} DD={prov.dd_version}"
        )

    console.print(
        f"\n[dim]Extraction: {report.extraction_count} items | "
        f"Temperature: {report.config.temperature}{reviewer_str} | "
        f"Timestamp: {report.timestamp}{prov_str}[/dim]"
    )

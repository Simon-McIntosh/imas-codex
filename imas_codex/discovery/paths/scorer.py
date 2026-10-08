"""LLM-based directory triage.

This module implements the triage phase of graph-led discovery:
1. Query graph for scanned but untriaged paths
2. Build batched prompts with directory context (parent/sibling scores,
   tree structure, file types, quality indicators)
3. Call LLM with structured output schema for reliable parsing
4. Trust the LLM's scores and expansion decisions directly

The LLM sees all relevant context in the prompt and makes calibrated
scoring and expansion decisions. The code applies only structural
overrides (git repos, data containers) that encode facts the LLM
cannot verify.

Retry logic handles rate limiting (OpenRouter "Overloaded" errors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from imas_codex.discovery.base.llm import suppress_litellm_noise
from imas_codex.discovery.base.scoring import PATH_SCORE_DIMENSIONS, max_composite
from imas_codex.discovery.paths.models import (
    DirectoryEvidence,
    PathDescriptionBatch,
    ResourcePurpose,
    TriageBatch,
    TriagedBatch,
    TriagedDirectory,
    parse_path_purpose,
)
from imas_codex.settings import get_model, get_path_scan_threshold, get_reasoning_effort

logger = logging.getLogger(__name__)

# Suppress litellm noise on import (print-based + logger-based)
suppress_litellm_noise()

# Per-purpose score names — canonical list from shared scoring module
PURPOSE_SCORE_NAMES = PATH_SCORE_DIMENSIONS

# Scan is a broad discovery gate: visualization and documentation alone do not
# imply that code or measured data should be scanned.
SCAN_FACETS = tuple(
    field
    for field in PURPOSE_SCORE_NAMES
    if field not in {"score_visualization", "score_documentation"}
)
PATH_EXPAND_THRESHOLD = 0.50


def path_scan_relevance(scores: dict[str, float]) -> float:
    """Return the strongest discovery facet in a path judgment."""
    return max((scores.get(field, 0.0) for field in SCAN_FACETS), default=0.0)


def path_judgment_fields(
    answers: dict[str, Any], model: str, *, prefix: str = "score"
) -> dict[str, Any]:
    """Validate and flatten a complete typed Jev answer for graph storage."""
    questions = build_path_judgment_questions()
    missing = set(questions) - set(answers)
    if missing:
        raise ValueError(f"Path judgment missing answers: {sorted(missing)}")
    purpose = answers["path_purpose"]
    options = list(questions["path_purpose"]["criteria"])
    choice = purpose["choice"]
    if choice not in options:
        raise ValueError(f"Unknown path purpose: {choice}")
    fields: dict[str, Any] = {
        "path_purpose": choice,
        "path_purpose_probs": [
            float(purpose["probabilities"].get(x, 0)) for x in options
        ],
        "path_purpose_confidence": float(purpose["confidence"]),
        "children_worth_listing": float(answers["children_worth_listing"]["noul"]),
        "judgment_model": model,
    }
    scores: dict[str, float] = {}
    for name in PURPOSE_SCORE_NAMES:
        answer = answers[name]
        levels = len(questions[name]["criteria"])
        value = float(answer["score"]) / (levels - 1)
        scores[name] = value
        stored = name if prefix == "score" else name.replace("score_", "triage_")
        fields[stored] = value
        fields[f"{stored}_probs"] = [
            float(answer["probabilities"].get(str(i), 0)) for i in range(levels)
        ]
        fields[f"{stored}_confidence"] = float(answer["confidence"])
    fields["scan_relevance"] = path_scan_relevance(scores)
    fields["should_expand"] = fields["children_worth_listing"] >= PATH_EXPAND_THRESHOLD
    fields["should_enrich"] = fields["scan_relevance"] >= get_path_scan_threshold()
    return fields


async def rejudge_stale_paths(
    facility: str, prefix: str, *, limit: int = 25
) -> tuple[int, float]:
    """Refresh stored judgments under one prefix from graph evidence alone."""
    import uuid

    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.base.judgment import judge_rows
    from imas_codex.graph import GraphClient

    model = get_model("discovery-relevance")
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        claimed = list(
            gc.query(
                """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE (p.path = $prefix OR p.path STARTS WITH $child_prefix)
              AND p.status IN ['triaged', 'scored', 'explored']
              AND (p.judgment_model IS NULL OR p.judgment_model <> $model)
              AND p.claimed_at IS NULL
            WITH p ORDER BY rand() LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            RETURN properties(p) AS row
            """,
                facility=facility,
                prefix=prefix.rstrip("/"),
                child_prefix=prefix.rstrip("/") + "/",
                model=model,
                limit=limit,
                token=token,
            )
        )
    rows = [dict(item["row"]) for item in claimed]
    if not rows:
        return 0, 0.0

    def apply(answered, _cost):
        items = []
        for row, answers, paid in answered:
            fields = path_judgment_fields(
                answers,
                model,
                prefix="score" if row["status"] in {"scored", "explored"} else "triage",
            )
            if row["status"] in {"scored", "explored"}:
                fields["score_composite"] = fields["scan_relevance"]
            else:
                fields["triage_composite"] = fields["scan_relevance"]
            fields["score_cost"] = float(row.get("score_cost") or 0) + paid
            items.append({"id": row["id"], "fields": fields})
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (p:FacilityPath {id: item.id, claim_token: $token})
                SET p += item.fields, p.claimed_at = null, p.claim_token = null
                """,
                items=items,
                token=token,
            )
        return len(items)

    try:
        facility_config = get_facility(facility)
        count, cost, failed = await judge_rows(
            rows,
            lambda row: build_path_judgment_state(row, facility, facility_config),
            build_path_judgment_questions,
            apply,
            model=model,
            service="facility-discovery",
        )
    finally:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (p:FacilityPath {claim_token: $token})
                SET p.claimed_at = null, p.claim_token = null
                """,
                token=token,
            )
    if failed:
        logger.warning("Path re-judge left %d unanswered rows", len(failed))
    return count or 0, cost


# Concrete directory situations for each stored FacilityPath facet. The
# schema supplies the question's subject; these examples anchor its levels.
_FACET_SITUATIONS = {
    "score_modeling_code": ("simulation input or launcher", "physics solver source"),
    "score_analysis_code": (
        "analysis launcher or configuration",
        "diagnostic processing or reconstruction source",
    ),
    "score_operations_code": (
        "control-system configuration",
        "DAQ, timing or feedback source",
    ),
    "score_modeling_data": (
        "model run manifest",
        "simulation output or parameter scan",
    ),
    "score_experimental_data": (
        "shot index or data catalogue",
        "measured shot records or database",
    ),
    "score_data_access": (
        "data-system configuration",
        "measured-data reader or converter source",
    ),
    "score_workflow": ("batch configuration", "orchestration or processing scripts"),
    "score_visualization": ("plot configuration", "plotting or rendering source"),
    "score_documentation": ("documentation index", "READMEs, tutorials or papers"),
    "score_imas": (
        "IMAS configuration or IDS references",
        "facility-to-IDS mapping source",
    ),
    "score_convention": (
        "unit or coordinate references",
        "sign, unit or COCOS conversion source",
    ),
}


def build_path_judgment_questions() -> dict[str, Any]:
    """Ask Jev for the schema's exclusive purpose and every stored path facet."""
    from imas_codex.graph.schema import get_schema

    schema = get_schema()
    purposes = schema.get_enum_with_descriptions("PathPurpose") or []
    choices = {
        item["value"]: item["description"]
        for item in purposes
        if item["value"] != "empty"
    }
    choices["other"] = "A purpose not covered by the listed directory categories"
    slots = schema.get_all_slots("FacilityPath")
    questions: dict[str, Any] = {
        "path_purpose": {
            "type": "choice",
            "instructions": "Which single category best describes the directory at `directory.path`?",
            "criteria": choices,
        }
    }
    for field in PURPOSE_SCORE_NAMES:
        if field not in slots or field not in _FACET_SITUATIONS:
            raise ValueError(f"Missing FacilityPath judgment definition: {field}")
        supporting, direct = _FACET_SITUATIONS[field]
        questions[field] = {
            "type": "score",
            "instructions": f"Does the directory at `directory.path` contain {slots[field]['description'].lower()}? Judge the directory's own contents; a container may be worth listing for its children without having this facet itself.",
            "criteria": [
                "No evidence of this content in the directory.",
                f"Contains {supporting}.",
                f"Contains {direct}.",
                f"Is the primary directory for {direct}.",
            ],
        }
    questions["children_worth_listing"] = {
        "type": "noul",
        "instructions": "Would listing the immediate child directories of `directory.path` likely reveal facility-specific code, measured data, machine descriptions or useful documentation? Judge children separately from this directory's own content. A container can be worth listing even when its own facet Scores are low.",
        "criteria": {
            "true": "listing children is likely to reveal relevant facility content",
            "false": "children are absent or unlikely to contain relevant facility content",
        },
    }
    return questions


def build_path_judgment_state(
    path_row: dict[str, Any], facility_id: str, facility_config: dict[str, Any]
) -> dict[str, Any]:
    """Present the directory evidence used by path scoring as one Jev state."""
    import json

    from imas_codex.discovery.code.scorer import facility_relevance_block

    def decoded(value: Any, fallback: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return fallback
        return value if value is not None else fallback

    return {
        "facility": facility_relevance_block(facility_id, facility_config),
        "directory": {
            "path": path_row["path"],
            "depth": path_row.get("depth"),
            "total_files": path_row.get("total_files") or 0,
            "total_dirs": path_row.get("total_dirs") or 0,
            "file_type_counts": decoded(path_row.get("file_type_counts"), {}),
            "child_names": decoded(path_row.get("child_names"), []),
            "tree_context": (path_row.get("tree_context") or "")[:3000],
            "has_readme": bool(path_row.get("has_readme")),
            "has_makefile": bool(path_row.get("has_makefile")),
            "has_git": bool(path_row.get("has_git")),
            "vcs_type": path_row.get("vcs_type"),
            "patterns_detected": decoded(path_row.get("patterns_detected"), []),
            "numeric_dir_ratio": path_row.get("numeric_dir_ratio") or 0,
            "description": (path_row.get("description") or "")[:1500],
            "enrichment": {
                "total_bytes": path_row.get("total_bytes"),
                "total_lines": path_row.get("total_lines"),
                "language_breakdown": decoded(path_row.get("language_breakdown"), {}),
                "pattern_categories": decoded(path_row.get("pattern_categories"), []),
                "read_matches": path_row.get("read_matches"),
                "write_matches": path_row.get("write_matches"),
                "is_multiformat": path_row.get("is_multiformat"),
            },
        },
    }


def combined_score(
    scores: dict[str, float],
    input_data: dict[str, Any],
    purpose: ResourcePurpose,
) -> float:
    """Compute combined score = max of per-dimension LLM scores.

    Delegates to the shared ``max_composite`` function.

    Args:
        scores: Dict of per-dimension scores (score_modeling_code, score_imas, etc.)
        input_data: Directory info dict (available for future use)
        purpose: Classified purpose (available for future use)

    Returns:
        Combined score (0.0-1.0)
    """
    return max_composite(scores)


@dataclass
class DirectoryTriager:
    """Score directories using LLM with grounded evidence.

    Implements:
    1. Batch prompt construction from DirStats
    2. LLM evidence collection via LiteLLM/OpenRouter
    3. Deterministic combined scoring from LLM dimension scores
    4. Frontier expansion logic

    Args:
        model: Model name (None = use "score" task model from config)
        facility: Facility ID for sampling calibration examples

    Example:
        triager = DirectoryTriager(facility="tcv")
        batch = triager.triage_batch(
            directories=[...],
            focus="equilibrium codes",
            threshold=0.7,
        )
    """

    model: str | None = None
    facility: str | None = None

    def __post_init__(self):
        """Initialize model from config if not provided."""
        if self.model is None:
            self.model = get_model("discovery-triage")

    def triage_batch(
        self,
        directories: list[dict[str, Any]],
        focus: str | None = None,
        threshold: float = 0.7,
    ) -> TriagedBatch:
        """Score a batch of directories using LLM with structured output.

        Args:
            directories: List of directory info dicts with:
              - path: str
              - total_files: int
              - total_dirs: int
              - has_readme: bool
              - has_makefile: bool
              - has_git: bool
              - file_type_counts: dict (optional)
              - patterns_detected: list (optional)
            focus: Natural language focus query (e.g., "equilibrium codes")
            threshold: Min score to expand (0.0-1.0)

        Returns:
            TriagedBatch with results and cost
        """
        from imas_codex.discovery.base.llm import call_llm_structured

        if not directories:
            return TriagedBatch(
                triaged_dirs=[],
                total_cost=0.0,
                model=self.model,
                tokens_used=0,
            )

        # Load prompt template
        system_prompt = self._build_system_prompt(focus)
        user_prompt = self._build_user_prompt(directories)

        # Call LLM with shared retry+parse loop (retries on both API
        # errors and JSON/validation errors from truncated responses).
        # Model-aware token limits applied automatically.
        batch, cost, total_tokens = call_llm_structured(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=TriageBatch,
            service="facility-discovery",
            reasoning_effort=get_reasoning_effort("discovery-triage"),
        )

        # Calculate cost per path for tracking
        cost_per_path = cost / len(directories) if directories else 0.0

        # Map parsed results to TriagedDirectory objects
        triaged_dirs = self._map_triaged_directories(
            batch, directories, threshold, cost_per_path
        )

        return TriagedBatch(
            triaged_dirs=triaged_dirs,
            total_cost=cost,
            model=self.model,
            tokens_used=total_tokens,
        )

    async def async_triage_batch(
        self,
        directories: list[dict[str, Any]],
        focus: str | None = None,
        threshold: float = 0.7,
    ) -> TriagedBatch:
        """Describe paths with the language model and judge them with Jev."""
        import json

        from imas_codex.discovery.base.facility import get_facility
        from imas_codex.discovery.base.judgment import judge_rows
        from imas_codex.discovery.base.llm import acall_llm_structured

        if not directories:
            return TriagedBatch([], 0.0, self.model or "", 0)

        batch, description_cost, tokens = await acall_llm_structured(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Describe each directory in one factual sentence. Return only its path and description; do not score, classify, or decide whether to explore it.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"focus": focus, "directories": directories}, default=str
                    ),
                },
            ],
            response_model=PathDescriptionBatch,
            service="facility-discovery",
            reasoning_effort=get_reasoning_effort("discovery-triage"),
        )
        descriptions = {item.path: item.description for item in batch.results}
        if set(descriptions) != {row["path"] for row in directories}:
            raise ValueError("Path descriptions do not match the claimed directories")
        rows = [
            {**row, "description": descriptions[row["path"]]} for row in directories
        ]
        facility_config = get_facility(self.facility) if self.facility else {}
        model = get_model("discovery-relevance")

        def apply(answered, _cost):
            triaged = []
            for row, answers, paid in answered:
                fields = path_judgment_fields(answers, model, prefix="triage")
                file_types = row.get("file_type_counts") or {}
                if isinstance(file_types, str):
                    file_types = json.loads(file_types)
                evidence = DirectoryEvidence(
                    code_indicators=[
                        key
                        for key in file_types
                        if key.lower() in {"py", "f", "f90", "c", "cpp", "h"}
                    ],
                    data_indicators=[
                        key
                        for key in file_types
                        if key.lower() in {"h5", "nc", "dat", "mat"}
                    ],
                    doc_indicators=["README"] if row.get("has_readme") else [],
                )
                purpose = fields["path_purpose"]
                scores = {
                    name: fields[name.replace("score_", "triage_")]
                    for name in PURPOSE_SCORE_NAMES
                }
                triaged.append(
                    TriagedDirectory(
                        path=row["path"],
                        path_purpose=purpose,
                        description=row["description"],
                        evidence=evidence,
                        **scores,
                        score=fields["scan_relevance"],
                        should_expand=fields["should_expand"],
                        should_enrich=fields["should_enrich"],
                        score_cost=paid + description_cost / len(rows),
                        judgments=fields,
                    )
                )
            return triaged

        triaged, judgment_cost, failed = await judge_rows(
            rows,
            lambda row: build_path_judgment_state(
                row, self.facility or "", facility_config
            ),
            build_path_judgment_questions,
            apply,
            model=model,
            service="facility-discovery",
        )
        if failed:
            raise ValueError(f"Path judgments incomplete for {len(failed)} directories")
        return TriagedBatch(
            triaged or [], description_cost + judgment_cost, model, tokens
        )

    def _build_system_prompt(self, focus: str | None = None) -> str:
        """Build system prompt for directory triage.

        Uses render_prompt() for proper Jinja2 rendering with schema context.
        The triage.md prompt is dynamic=true, so schema-derived values
        (path_purposes, score_dimensions) are injected automatically.

        Injects:
        - focus: optional natural language focus
        - dimension_calibration: examples at 5 score levels per dimension
        """
        from imas_codex.discovery.paths.frontier import (
            sample_dimension_calibration_examples,
        )
        from imas_codex.llm.prompt_loader import render_prompt

        # Build context for template rendering
        context: dict[str, Any] = {}

        # Add focus if provided
        if focus:
            context["focus"] = focus

        # Add dimension calibration examples (5 levels x 11 dimensions x 3 examples)
        # This provides calibration for the LLM to understand
        # what scores have historically been assigned at each level.
        # Results are cached with 5-minute TTL to avoid redundant graph queries.
        # phase='triage' draws from triaged peers (1st-pass dimensions).
        dimension_calibration = sample_dimension_calibration_examples(
            facility=self.facility,
            per_level=5,
            tolerance=0.1,
            phase="triage",
        )
        # Only add if we have meaningful data
        has_calibration = any(
            any(examples for examples in dim_levels.values())
            for dim_levels in dimension_calibration.values()
        )
        if has_calibration:
            context["dimension_calibration"] = dimension_calibration

        # Use render_prompt for proper Jinja2 rendering with schema context
        return render_prompt("paths/triage", context)

    def _build_user_prompt(self, directories: list[dict[str, Any]]) -> str:
        """Build user prompt with directories to score.

        Includes child file/directory names for context - these are
        critical for the LLM to infer purpose from naming conventions.

        Also injects parent/sibling context from the graph to enable
        relative scoring decisions.
        """
        import json as json_module

        from imas_codex.discovery.paths.frontier import get_hierarchy_context

        # Query hierarchy context for all paths in the batch
        paths = [d["path"] for d in directories]
        hierarchy = {}
        if self.facility:
            try:
                hierarchy = get_hierarchy_context(self.facility, paths)
            except Exception:
                logger.debug("Failed to get hierarchy context", exc_info=True)

        lines = [
            "Score these directories.",
            "(In Contents below, entries ending with / are subdirectories, "
            "others are files.)\n",
        ]

        for i, d in enumerate(directories, 1):
            # Full path is critical context - shown prominently
            lines.append(f"\n## Directory {i}")
            lines.append(f"Path: {d['path']}")

            # Depth info
            depth = d.get("depth")
            if depth is not None:
                lines.append(f"Depth: {depth}")

            # Add DirStats
            lines.append(
                f"Files: {d.get('total_files', 0)}, Dirs: {d.get('total_dirs', 0)}"
            )

            file_types = d.get("file_type_counts")
            if file_types:
                if isinstance(file_types, str):
                    try:
                        file_types = json_module.loads(file_types)
                    except json_module.JSONDecodeError:
                        file_types = {}
                lines.append(f"File types: {file_types}")

            # Quality indicators on one line
            quality = []
            if d.get("has_readme"):
                quality.append("README")
            if d.get("has_makefile"):
                quality.append("Makefile")
            vcs = d.get("vcs_type")
            if vcs:
                accessible = d.get("vcs_remote_accessible")
                if accessible is True:
                    quality.append(f".{vcs} (remote accessible)")
                elif accessible is False:
                    quality.append(f".{vcs} (remote inaccessible)")
                else:
                    quality.append(f".{vcs}")
            elif d.get("has_git"):
                quality.append(".git")
            if quality:
                lines.append(f"Quality: {', '.join(quality)}")

            patterns = d.get("patterns_detected", [])
            if patterns:
                lines.append(f"Patterns: {', '.join(patterns)}")

            # Numeric directory ratio warning (shot folders, run dirs)
            numeric_ratio = d.get("numeric_dir_ratio") or 0
            if numeric_ratio > 0.5:
                lines.append(
                    f"⚠️ DATA CONTAINER: {numeric_ratio:.0%} of subdirs are "
                    f"numeric (shot IDs/runs). Set should_expand=false."
                )

            # Parent/sibling context from graph
            ctx = hierarchy.get(d["path"])
            if ctx:
                parent = ctx.get("parent")
                if parent and parent.get("score") is not None:
                    lines.append(
                        f"Parent: {parent['path']} → "
                        f"{parent.get('purpose', '?')} "
                        f"(score: {parent['score']:.2f})"
                    )

                siblings = ctx.get("siblings", [])
                if siblings:
                    sib_strs = []
                    for s in siblings[:6]:
                        basename = s["path"].rstrip("/").split("/")[-1]
                        sib_strs.append(
                            f"{basename}={s.get('purpose', '?')}"
                            f"({s.get('score', 0):.1f})"
                        )
                    lines.append(f"Scored siblings: {', '.join(sib_strs)}")

            # Prefer tree context over flat child_names (shows hierarchy)
            tree_context = d.get("tree_context")
            if tree_context:
                lines.append("Structure (eza --tree):")
                lines.append(f"```\n{tree_context}\n```")
            else:
                # Fall back to flat child names
                child_names = d.get("child_names")
                if child_names:
                    if isinstance(child_names, str):
                        try:
                            child_names = json_module.loads(child_names)
                        except json_module.JSONDecodeError:
                            child_names = []
                    if child_names:
                        names_to_show = child_names[:30]
                        lines.append(f"Contents: {', '.join(names_to_show)}")

        lines.append(
            "\n\nReturn results for each directory in order. "
            "The response format is enforced by the schema."
        )

        return "\n".join(lines)

    def _map_triaged_directories(
        self,
        batch: TriageBatch,
        directories: list[dict[str, Any]],
        threshold: float,
        cost_per_path: float = 0.0,
    ) -> list[TriagedDirectory]:
        """Map parsed TriageBatch to TriagedDirectory objects.

        JSON parsing and sanitization are handled by call_llm_structured().
        This method applies combined scoring, expansion logic, and enrichment
        decisions to the already-validated Pydantic model.

        Args:
            batch: Parsed TriageBatch from LLM response
            directories: Input directory info dicts
            threshold: Minimum score to expand
            cost_per_path: LLM cost per path (batch_cost / batch_size)

        Returns:
            List of TriagedDirectory objects with scores and cost_per_path set
        """
        import json as json_module

        results = batch.results

        triaged = []
        for i, result in enumerate(results[: len(directories)]):
            path = directories[i]["path"]

            # Clamp per-purpose scores (should already be valid from schema)
            scores = {
                "score_modeling_code": max(
                    0.0, min(1.0, getattr(result, "score_modeling_code", 0.0))
                ),
                "score_analysis_code": max(
                    0.0, min(1.0, getattr(result, "score_analysis_code", 0.0))
                ),
                "score_operations_code": max(
                    0.0, min(1.0, getattr(result, "score_operations_code", 0.0))
                ),
                "score_modeling_data": max(
                    0.0, min(1.0, getattr(result, "score_modeling_data", 0.0))
                ),
                "score_experimental_data": max(
                    0.0, min(1.0, getattr(result, "score_experimental_data", 0.0))
                ),
                "score_data_access": max(
                    0.0, min(1.0, getattr(result, "score_data_access", 0.0))
                ),
                "score_workflow": max(
                    0.0, min(1.0, getattr(result, "score_workflow", 0.0))
                ),
                "score_visualization": max(
                    0.0, min(1.0, getattr(result, "score_visualization", 0.0))
                ),
                "score_documentation": max(
                    0.0, min(1.0, getattr(result, "score_documentation", 0.0))
                ),
                "score_imas": max(0.0, min(1.0, getattr(result, "score_imas", 0.0))),
            }

            # Convert Pydantic enum to graph ResourcePurpose
            purpose = parse_path_purpose(result.path_purpose.value)

            # Build evidence from input data (not LLM response - schema simplified)
            file_types = directories[i].get("file_type_counts") or {}
            if isinstance(file_types, str):
                try:
                    file_types = json_module.loads(file_types)
                except (json_module.JSONDecodeError, TypeError):
                    file_types = {}
            code_exts = {"py", "f90", "f", "cpp", "c", "h", "jl", "m", "pro", "idl"}
            data_exts = {"nc", "h5", "hdf5", "csv", "dat", "mat"}
            evidence = DirectoryEvidence(
                code_indicators=[ext for ext in file_types if ext.lower() in code_exts],
                data_indicators=[ext for ext in file_types if ext.lower() in data_exts],
                doc_indicators=["README"] if directories[i].get("has_readme") else [],
                imas_indicators=[],  # Filled by enrichment worker
                physics_indicators=[],  # Filled by enrichment worker
                quality_indicators=[
                    name
                    for name, flag in [
                        ("has_readme", directories[i].get("has_readme")),
                        ("has_makefile", directories[i].get("has_makefile")),
                        ("has_git", directories[i].get("has_git")),
                    ]
                    if flag
                ],
            )

            # Compute grounded score from per-purpose scores and input data
            combined = combined_score(scores, directories[i], purpose)

            # Pass through LLM decisions directly — structural overrides
            # (VCS accessibility, data containers) are applied in
            # mark_paths_triaged() at persistence time.
            should_expand = result.should_expand
            should_enrich = result.should_enrich
            enrich_skip_reason = result.enrich_skip_reason

            # terminal_reason is NULL for LLM-scored paths - reason is derivable
            # from has_git, path_purpose, score. Only set for non-derivable cases
            # (access_denied, empty, parent_terminal, etc.)

            triaged_dir = TriagedDirectory(
                path=path,
                path_purpose=purpose,
                description=result.description,
                evidence=evidence,
                score_modeling_code=scores["score_modeling_code"],
                score_analysis_code=scores["score_analysis_code"],
                score_operations_code=scores["score_operations_code"],
                score_modeling_data=scores["score_modeling_data"],
                score_experimental_data=scores["score_experimental_data"],
                score_data_access=scores["score_data_access"],
                score_workflow=scores["score_workflow"],
                score_visualization=scores["score_visualization"],
                score_documentation=scores["score_documentation"],
                score_imas=scores["score_imas"],
                score=combined,
                should_expand=should_expand,
                should_enrich=should_enrich,
                keywords=result.keywords[:5] if result.keywords else [],
                physics_domain=result.physics_domain,
                expansion_reason=result.expansion_reason,
                skip_reason=result.skip_reason,
                enrich_skip_reason=enrich_skip_reason,
                score_cost=cost_per_path,
            )

            triaged.append(triaged_dir)

        return triaged

    def _parse_response(
        self,
        response_text: str,
        directories: list[dict[str, Any]],
        threshold: float,
    ) -> list[TriagedDirectory]:
        """Parse unstructured LLM response (legacy fallback)."""
        import json

        try:
            # Extract JSON from response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")

            json_str = response_text[json_start:json_end]
            results = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback: return empty scores for all (use container as neutral purpose)
            return [
                TriagedDirectory(
                    path=d["path"],
                    path_purpose=ResourcePurpose.container,
                    description="Parse error",
                    evidence=DirectoryEvidence(),
                    score=0.0,
                    should_expand=False,
                    skip_reason=f"LLM response parse failed: {e}",
                )
                for d in directories
            ]

        triaged = []
        for i, result in enumerate(results[: len(directories)]):
            path = directories[i]["path"]

            # Extract and clamp per-purpose scores
            scores = {
                "score_modeling_code": max(
                    0.0, min(1.0, float(result.get("score_modeling_code", 0.0)))
                ),
                "score_analysis_code": max(
                    0.0, min(1.0, float(result.get("score_analysis_code", 0.0)))
                ),
                "score_operations_code": max(
                    0.0, min(1.0, float(result.get("score_operations_code", 0.0)))
                ),
                "score_modeling_data": max(
                    0.0, min(1.0, float(result.get("score_modeling_data", 0.0)))
                ),
                "score_experimental_data": max(
                    0.0, min(1.0, float(result.get("score_experimental_data", 0.0)))
                ),
                "score_data_access": max(
                    0.0, min(1.0, float(result.get("score_data_access", 0.0)))
                ),
                "score_workflow": max(
                    0.0, min(1.0, float(result.get("score_workflow", 0.0)))
                ),
                "score_visualization": max(
                    0.0, min(1.0, float(result.get("score_visualization", 0.0)))
                ),
                "score_documentation": max(
                    0.0, min(1.0, float(result.get("score_documentation", 0.0)))
                ),
                "score_imas": max(0.0, min(1.0, float(result.get("score_imas", 0.0)))),
            }

            # Parse purpose
            purpose = parse_path_purpose(result.get("path_purpose", "unknown"))

            # Build evidence from input data (not LLM response - schema simplified)
            file_types = directories[i].get("file_type_counts") or {}
            if isinstance(file_types, str):
                try:
                    file_types = json.loads(file_types)
                except (json.JSONDecodeError, TypeError):
                    file_types = {}
            code_exts = {"py", "f90", "f", "cpp", "c", "h", "jl", "m", "pro", "idl"}
            data_exts = {"nc", "h5", "hdf5", "csv", "dat", "mat"}
            evidence = DirectoryEvidence(
                code_indicators=[ext for ext in file_types if ext.lower() in code_exts],
                data_indicators=[ext for ext in file_types if ext.lower() in data_exts],
                doc_indicators=["README"] if directories[i].get("has_readme") else [],
                imas_indicators=[],  # Filled by enrichment worker
                physics_indicators=[],  # Filled by enrichment worker
                quality_indicators=[
                    name
                    for name, flag in [
                        ("has_readme", directories[i].get("has_readme")),
                        ("has_makefile", directories[i].get("has_makefile")),
                        ("has_git", directories[i].get("has_git")),
                    ]
                    if flag
                ],
            )

            # Compute combined score from input data
            combined = combined_score(scores, directories[i], purpose)

            # Pass through LLM decisions directly — structural overrides
            # applied in mark_paths_triaged() at persistence time.
            should_expand = result.get("should_expand", False)
            should_enrich = result.get("should_enrich", True)
            enrich_skip_reason = result.get("enrich_skip_reason")

            triaged_dir = TriagedDirectory(
                path=path,
                path_purpose=purpose,
                description=result.get("description", ""),
                evidence=evidence,
                score_modeling_code=scores["score_modeling_code"],
                score_analysis_code=scores["score_analysis_code"],
                score_operations_code=scores["score_operations_code"],
                score_modeling_data=scores["score_modeling_data"],
                score_experimental_data=scores["score_experimental_data"],
                score_data_access=scores["score_data_access"],
                score_workflow=scores["score_workflow"],
                score_visualization=scores["score_visualization"],
                score_documentation=scores["score_documentation"],
                score_imas=scores["score_imas"],
                score=combined,
                should_expand=should_expand,
                should_enrich=should_enrich,
                keywords=result.get("keywords", [])[:5],  # Cap at 5
                physics_domain=result.get("physics_domain"),
                expansion_reason=result.get("expansion_reason"),
                skip_reason=result.get("skip_reason"),
                enrich_skip_reason=enrich_skip_reason,
            )

            triaged.append(triaged_dir)

        return triaged

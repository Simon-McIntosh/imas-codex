"""Directory description and typed path judgments for graph-led discovery.

The language model writes factual descriptions. Jev classifies purpose and
judges whether children are worth listing. Code derives path decisions from
those answers. Facet answers are stored separately from the path gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from imas_codex.discovery.base.llm import suppress_litellm_noise
from imas_codex.discovery.base.scoring import PATH_SCORE_DIMENSIONS, max_composite
from imas_codex.discovery.paths.models import (
    DirectoryEvidence,
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
CODE_BEARING_PURPOSES = frozenset(
    {
        "modeling_code",
        "analysis_code",
        "operations_code",
        "data_access",
        "workflow",
        "visualization",
        "software_project",
        "test_suite",
    }
)
DATA_PURPOSES = frozenset({"experimental_data", "modeling_data"})
SKIPPED_PURPOSES = frozenset({"archive", "build_artifact", "system", "empty_directory"})


def path_scan_relevance(scores: dict[str, float]) -> float:
    """Return the strongest discovery facet in a path judgment."""
    return max((scores.get(field, 0.0) for field in SCAN_FACETS), default=0.0)


def path_category_gate(
    probabilities: dict[str, float], children_worth_listing: float
) -> tuple[float, float]:
    """Rank scanning and expansion from purpose probabilities and child evidence."""
    scan = sum(probabilities.get(purpose, 0.0) for purpose in CODE_BEARING_PURPOSES)
    expand = max(probabilities.get("container", 0.0), children_worth_listing)
    return scan, expand


def path_judgment_fields(
    answers: dict[str, Any],
    model: str,
    *,
    prefix: str = "score",
    scan_threshold: float | None = None,
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
    for name in PURPOSE_SCORE_NAMES:
        answer = answers[name]
        levels = len(questions[name]["criteria"])
        value = float(answer["score"]) / (levels - 1)
        stored = name if prefix == "score" else name.replace("score_", "triage_")
        fields[stored] = value
        fields[f"{stored}_probs"] = [
            float(answer["probabilities"].get(str(i), 0)) for i in range(levels)
        ]
        fields[f"{stored}_confidence"] = float(answer["confidence"])
    distribution = dict(zip(options, fields["path_purpose_probs"], strict=True))
    scan_relevance, _ = path_category_gate(
        distribution, fields["children_worth_listing"]
    )
    restricted = choice in DATA_PURPOSES | SKIPPED_PURPOSES
    fields["scan_relevance"] = 0.0 if restricted else scan_relevance
    fields["should_expand"] = not restricted and (
        choice == "container"
        or fields["children_worth_listing"] >= PATH_EXPAND_THRESHOLD
    )
    minimum = get_path_scan_threshold() if scan_threshold is None else scan_threshold
    fields["should_enrich"] = not restricted and fields["scan_relevance"] >= minimum
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
    from imas_codex.discovery.paths.enrichment import facility_access_matches

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
            "description": (
                path_row["description"][:1500]
                if path_row.get("description") is not None
                else None
            ),
            "enrichment": {
                "total_bytes": path_row.get("total_bytes"),
                "total_lines": path_row.get("total_lines"),
                "language_breakdown": decoded(path_row.get("language_breakdown"), {}),
                "pattern_categories": decoded(path_row.get("pattern_categories"), []),
                "facility_data_access_matches": facility_access_matches(
                    path_row.get("pattern_categories")
                ),
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
    """Describe scanned directories and judge their path facets with Jev."""

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
        threshold: float | None = None,
    ) -> TriagedBatch:
        """Run the same description and Jev path as asynchronous triage."""
        import asyncio

        return asyncio.run(self.async_triage_batch(directories, focus, threshold))

    async def async_triage_batch(
        self,
        directories: list[dict[str, Any]],
        focus: str | None = None,
        threshold: float | None = None,
    ) -> TriagedBatch:
        """Describe paths with the language model and judge them with Jev."""
        import json

        from imas_codex.discovery.base.facility import get_facility
        from imas_codex.discovery.base.judgment import judge_rows
        from imas_codex.discovery.paths.description import describe_paths

        if not directories:
            return TriagedBatch([], 0.0, self.model or "", 0)

        try:
            batch, description_cost, tokens = await describe_paths(
                directories,
                model=self.model,
                focus=focus,
                reasoning_effort=get_reasoning_effort("discovery-triage"),
            )
            descriptions = {item.path: item.description for item in batch.results}
        except Exception:
            logger.warning(
                "Path description failed; judging metadata without it", exc_info=True
            )
            descriptions, description_cost, tokens = {}, 0.0, 0
        rows = [
            {**row, "description": descriptions.get(row["path"])} for row in directories
        ]
        facility_config = get_facility(self.facility) if self.facility else {}
        model = get_model("discovery-relevance")

        def apply(answered, _cost):
            triaged = []
            for row, answers, paid in answered:
                fields = path_judgment_fields(
                    answers, model, prefix="triage", scan_threshold=threshold
                )
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

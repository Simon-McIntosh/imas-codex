"""Two-stage scoring for discovered CodeFiles.

Pass 1 (Triage): a decisions model answers six typed questions about the file
from minimal context — parent directory description, filename, sibling names
and the facility's data-access patterns.  A file's relevance is the largest of
the four scope probabilities (loads, processes, describes, maps_to_imas).
Files whose relevance reaches the triage threshold proceed to enrichment.

Pass 2 (Score): the content scorer's LLM call still writes descriptions and
search facets from enrichment evidence, and a content-arm decision records the
file's content relevance.  Ingest admits a scored file whose content relevance
reaches the ingest threshold.

Both decision arms share the same eight relevance fields and the persistence
helper below; the stage field records which arm produced the stored values.

Lifecycle: discovered → triaged → (enrich) → scored → ingested | skipped
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.discovery.base.scoring import (
    CODE_SCORE_DIMENSIONS,
    CodeScoreFields,
    max_composite,
)
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score dimensions — canonical list from shared scoring module
# ---------------------------------------------------------------------------

SCORE_DIMENSION_NAMES = CODE_SCORE_DIMENSIONS

# The four scope questions. A file's relevance is the largest noul over these.
SCOPE_NOULS = (
    "loads_diagnostic_data",
    "processes_diagnostic_signals",
    "describes_machine_or_diagnostics",
    "maps_to_imas",
)

# decision question name -> CodeFile relevance field
_RELEVANCE_FIELDS = {
    "loads_diagnostic_data": "relevance_loads",
    "processes_diagnostic_signals": "relevance_processes",
    "describes_machine_or_diagnostics": "relevance_describes",
    "maps_to_imas": "relevance_imas",
    "is_simulation": "relevance_simulation",
}

# The Cypher that reads a CodeFile's relevance as the max of the four scope
# fields.  Shared so the triage threshold, the ingest threshold and the
# highest-relevance-first ordering all measure the same quantity.
CODE_RELEVANCE_EXPR = (
    "reduce(max = 0.0, x IN "
    "[sf.relevance_loads, sf.relevance_processes, "
    "sf.relevance_describes, sf.relevance_imas] "
    "| CASE WHEN coalesce(x, 0.0) > max THEN x ELSE max END)"
)


# ---------------------------------------------------------------------------
# Decisions questions and state
# ---------------------------------------------------------------------------


def build_triage_questions() -> dict[str, Any]:
    """Render the decisions questions template into a questions mapping.

    The six questions and their wording are declared in
    ``imas_codex/llm/prompts/code/triage.md`` and rendered through
    :func:`render_prompt`; this loads the rendered JSON body.
    """
    import json

    from imas_codex.llm.prompt_loader import render_prompt

    return json.loads(render_prompt("code/triage", {}))


def facility_relevance_block(facility_id: str, facility_config: dict) -> dict[str, Any]:
    """Build the facility block injected into a decision's state.

    Reads the facility's ``data_access_patterns`` — the same source the wiki
    scorer injects — so the facility YAML stays the one owner of that config.
    """
    d = (facility_config or {}).get("data_access_patterns") or {}
    return {
        "id": facility_id,
        "primary_data_system": d.get("primary_method"),
        "data_access_tools": d.get("key_tools") or [],
        "data_access_code_patterns": d.get("code_import_patterns") or [],
        "data_organization": (d.get("tree_organization") or "")[:500],
        "signal_naming": (d.get("signal_naming") or "")[:500],
    }


def build_triage_state(
    file_row: dict,
    facility_id: str,
    facility_config: dict,
    *,
    with_content: bool = False,
) -> dict[str, Any]:
    """Build the state a code decision judges.

    Carries the facility block plus the file's path, language, directory,
    directory description and sibling names.  The content arm additionally
    carries ``content_head`` — the first 1500 characters of the file preview.
    """
    state: dict[str, Any] = {
        "facility": facility_relevance_block(facility_id, facility_config),
        "file": {
            "path": file_row["path"],
            "language": file_row.get("language") or "unknown",
            "directory": file_row.get("parent_path") or "",
            "directory_description": file_row.get("parent_description") or "",
            "siblings": file_row.get("sibling_names") or [],
        },
    }
    if with_content:
        state["file"]["content_head"] = (file_row.get("preview_text") or "")[:1500]
    return state


def triage_relevance(answers: dict[str, Any]) -> float:
    """A file's relevance: the largest of the four scope nouls."""
    return max(float(answers[name]["noul"]) for name in SCOPE_NOULS)


def _relevance_item(
    sf_id: str,
    answers: dict[str, Any],
    *,
    stage: str,
    model: str | None,
    cost: float,
) -> dict[str, Any]:
    """Build the persisted relevance fields from one decision's answers."""
    item: dict[str, Any] = {
        "id": sf_id,
        "score_cost": cost,
        "relevance_stage": stage,
        "relevance_model": model or "",
        "relevance_role": (answers.get("role") or {}).get("choice") or "",
    }
    for question, field in _RELEVANCE_FIELDS.items():
        answer = answers.get(question) or {}
        item[field] = round(float(answer.get("noul", 0.0)), 4)
    return item


def _relevance_set_clause() -> str:
    """The Cypher ``SET`` fragment writing every relevance field from an item."""
    return """,
                    sf.relevance_loads = item.relevance_loads,
                    sf.relevance_processes = item.relevance_processes,
                    sf.relevance_describes = item.relevance_describes,
                    sf.relevance_imas = item.relevance_imas,
                    sf.relevance_simulation = item.relevance_simulation,
                    sf.relevance_role = item.relevance_role,
                    sf.relevance_stage = item.relevance_stage,
                    sf.relevance_model = item.relevance_model"""


# ---------------------------------------------------------------------------
# Dynamic calibration (same architecture as paths/frontier.py)
# ---------------------------------------------------------------------------

_calibration_cache: dict[str, tuple[float, dict]] = {}
_CALIBRATION_TTL_SECONDS = 300.0  # 5 minutes — matches LLM provider ephemeral cache TTL


def sample_code_dimension_calibration(
    facility: str | None = None,
    per_level: int = 3,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Sample calibration examples per score dimension at 5 levels.

    Draws ``score_*`` from scored-only CodeFiles (the cohort that passed triage
    and enrichment), so scoring calibrates among peers.  Cached with a 5-minute
    TTL — stable within a batch, evolving over time.

    Returns:
        Nested dict: dimension -> level -> list of examples.
        Each example: path, facility, score, purpose, description.
    """
    global _calibration_cache  # noqa: PLW0603

    cache_key = f"score:{facility}:{per_level}"
    now = time.monotonic()

    if cache_key in _calibration_cache:
        cached_time, cached_data = _calibration_cache[cache_key]
        if (now - cached_time) < _CALIBRATION_TTL_SECONDS:
            return cached_data

    samples = _fetch_code_dimension_calibration(facility, per_level)
    _calibration_cache[cache_key] = (now, samples)
    return samples


def _fetch_code_dimension_calibration(
    facility: str | None,
    per_level: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Fetch dimension calibration from scored CodeFile nodes (uncached)."""
    status_clause = "cf.status IN ['scored', 'ingested']"

    buckets: list[tuple[str, float, float]] = [
        ("lowest", 0.0, 0.15),
        ("low", 0.10, 0.30),
        ("medium", 0.40, 0.60),
        ("high", 0.70, 0.90),
        ("highest", 0.90, 1.01),
    ]

    samples: dict[str, dict[str, list[dict[str, Any]]]] = {}

    with GraphClient() as gc:
        for dim, graph_prop in zip(
            SCORE_DIMENSION_NAMES, SCORE_DIMENSION_NAMES, strict=True
        ):
            samples[dim] = {}

            for level_name, min_score, max_score in buckets:
                target = (min_score + max_score) / 2
                result = gc.query(
                    f"""
                    MATCH (cf:CodeFile)
                    WHERE {status_clause}
                        AND cf.{graph_prop} >= $min_score
                        AND cf.{graph_prop} < $max_score
                        AND cf.{graph_prop} IS NOT NULL
                    RETURN cf.path AS path,
                           cf.facility_id AS facility,
                           cf.{graph_prop} AS score,
                           cf.score_reason AS description
                    ORDER BY
                        CASE WHEN cf.facility_id = $facility
                             THEN 0 ELSE 1 END,
                        abs(cf.{graph_prop} - $target) ASC,
                        cf.id ASC
                    LIMIT $limit
                    """,
                    min_score=min_score,
                    max_score=max_score,
                    target=target,
                    facility=facility or "",
                    limit=per_level,
                )

                samples[dim][level_name] = [
                    {
                        "path": r["path"],
                        "facility": r["facility"],
                        "score": round(r["score"], 2),
                        "purpose": "code file",
                        "description": r["description"] or "",
                    }
                    for r in result
                ]

    return samples


# ---------------------------------------------------------------------------
# Triage models (retained for the module's public re-exports)
# ---------------------------------------------------------------------------


class FileTriageResult(CodeScoreFields):
    """Per-dimension scoring result shape, retained for module re-exports.

    The code pipeline no longer produces these; triage now asks a decisions
    model for typed relevance judgements.  The class is kept so
    ``imas_codex.discovery.code`` keeps importing cleanly until the package
    re-created exports are retired.
    """

    path: str = Field(description="The file path (echo from input)")
    description: str = Field(
        default="",
        description="Brief description of what the file likely contains (1 sentence)",
    )

    @property
    def triage_composite(self) -> float:
        """Composite = max of all dimension scores."""
        return max_composite(self.get_score_dict())


class FileTriageBatch(BaseModel):
    """Batch of triage results (retained for module re-exports)."""

    results: list[FileTriageResult]


# ---------------------------------------------------------------------------
# Score models (full scoring with enrichment evidence)
# ---------------------------------------------------------------------------


class FileScoreResult(CodeScoreFields):
    """Full scoring result with enrichment evidence.

    Inherits 9 score dimensions from CodeScoreFields.
    """

    path: str = Field(description="The file path (echo from input)")
    file_category: str = Field(
        description="code, document, notebook, config, data, or other"
    )
    description: str = Field(
        default="",
        description="Brief summary of what the file likely contains (1 sentence)",
    )

    @property
    def score_composite(self) -> float:
        """Composite = max of all dimension scores."""
        return max_composite(self.get_score_dict())


class FileScoreBatch(BaseModel):
    """Batch of file scoring results from LLM."""

    results: list[FileScoreResult]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_score_system_prompt(
    facility: str | None = None,
    focus: str | None = None,
) -> str:
    """Build scorer system prompt with dimension calibration."""
    from imas_codex.llm.prompt_loader import render_prompt

    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus

    dimension_calibration = sample_code_dimension_calibration(
        facility=facility, per_level=5
    )
    has_calibration = any(
        any(examples for examples in dim_levels.values())
        for dim_levels in dimension_calibration.values()
    )
    if has_calibration:
        context["dimension_calibration"] = dimension_calibration

    return render_prompt("code/scorer", context)


def _build_score_user_prompt(file_groups: list[dict[str, Any]]) -> str:
    """Build scorer user prompt -- enrichment evidence + preview text."""
    lines = ["Score these files using their enrichment evidence and content preview.\n"]

    for i, group in enumerate(file_groups, 1):
        parent_path = group.get("parent_path", "unknown")
        parent_desc = group.get("parent_description") or ""

        lines.append(f"\n## Directory {i}: {parent_path}")
        if parent_desc:
            lines.append(f"Directory description: {parent_desc}")

        lines.append("\nFiles:")
        for f in group.get("files", []):
            lang = f.get("language") or "unknown"
            line_count = f.get("line_count") or 0
            patterns = f.get("pattern_categories") or {}
            total_matches = f.get("total_pattern_matches") or 0
            preview = f.get("preview_text") or ""

            parts = [f"\n  ### {f['path']} ({lang}, {line_count} lines)"]

            if total_matches > 0:
                pattern_str = ", ".join(f"{k}: {v}" for k, v in patterns.items() if v)
                parts.append(
                    f"  Pattern matches: {pattern_str} (total: {total_matches})"
                )

            if preview:
                truncated = preview[:500]
                if len(preview) > 500:
                    truncated += "..."
                parts.append(f"  Content preview:\n  ```\n  {truncated}\n  ```")

            lines.append("\n".join(parts))

    return "\n".join(lines)


def _group_files_by_parent(
    files: list[dict],
    include_siblings: bool = False,
) -> list[dict[str, Any]]:
    """Group files by their parent FacilityPath.

    When ``include_siblings=True``, queries the graph for ALL CodeFile
    names under each parent directory.  This gives the triage decision
    neighborhood context -- seeing what other files exist alongside the
    ones being judged.

    Returns list of group dicts with parent context and file lists.
    """
    groups: dict[str, dict[str, Any]] = {}

    for f in files:
        parent_id = f.get("parent_path_id", "unknown")
        if parent_id not in groups:
            groups[parent_id] = {
                "parent_path_id": parent_id,
                "parent_path": f.get("parent_path") or "unknown",
                "parent_description": f.get("parent_description") or "",
                "files": [],
                "sibling_names": [],
            }

        file_entry: dict[str, Any] = {
            "id": f["id"],
            "path": f["path"],
            "language": f.get("language") or "unknown",
        }
        for key in (
            "line_count",
            "pattern_categories",
            "total_pattern_matches",
            "preview_text",
        ):
            if key in f:
                file_entry[key] = f[key]

        groups[parent_id]["files"].append(file_entry)

    # Fetch sibling file names for triage context
    if include_siblings and groups:
        parent_ids = list(groups.keys())
        with GraphClient() as gc:
            rows = gc.query(
                """
                UNWIND $parent_ids AS pid
                MATCH (cf:CodeFile)-[:IN_DIRECTORY]->(fp:FacilityPath {id: pid})
                RETURN pid AS parent_id, cf.path AS path
                """,
                parent_ids=parent_ids,
            )
            for row in rows:
                pid = row["parent_id"]
                if pid in groups:
                    groups[pid]["sibling_names"].append(row["path"])

    return list(groups.values())


# ---------------------------------------------------------------------------
# Graph persistence
# ---------------------------------------------------------------------------


def apply_triage_results(
    decisions: list[dict[str, Any]],
    file_id_map: dict[str, str],
    threshold: float | None = None,
    cost_total: float = 0.0,
) -> dict[str, Any]:
    """Persist names-arm decision relevance and set the triage outcome.

    A file whose relevance reaches ``threshold`` becomes ``triaged`` and
    proceeds to enrichment; otherwise it becomes ``skipped`` with a skip
    reason naming its role and the highest role probabilities.

    Args:
        decisions: One dict per judged file with ``path``, ``answers``,
            ``model`` and ``cost`` keys.
        file_id_map: Mapping from path to CodeFile ID.
        threshold: Minimum relevance to pass.
            Defaults to ``get_code_triage_threshold()``.
        cost_total: Total decision cost for the batch, distributed per file.

    Returns dict with triaged, skipped counts and triaged_ids.
    """
    if threshold is None:
        from imas_codex.settings import get_code_triage_threshold

        threshold = get_code_triage_threshold()

    matched_count = sum(1 for d in decisions if file_id_map.get(d["path"]))
    cost_per_file = cost_total / matched_count if matched_count > 0 else 0.0

    triaged_items = []
    skipped_items = []

    for decision in decisions:
        sf_id = file_id_map.get(decision["path"])
        if not sf_id:
            continue
        answers = decision["answers"]
        relevance = triage_relevance(answers)
        item = _relevance_item(
            sf_id,
            answers,
            stage="name",
            model=decision.get("model"),
            cost=decision.get("cost", cost_per_file),
        )
        if relevance >= threshold:
            triaged_items.append(item)
        else:
            probabilities = (answers.get("role") or {}).get("probabilities") or {}
            ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
            top = ", ".join(f"{name}={prob:.2f}" for name, prob in ranked[:2])
            skipped_items.append(
                {
                    **item,
                    "reason": (
                        f"role={item['relevance_role']}; relevance={relevance:.2f}; {top}"
                    ),
                }
            )

    set_clause = _relevance_set_clause()
    with GraphClient() as gc:
        if triaged_items:
            gc.query(
                f"""
                UNWIND $items AS item
                MATCH (sf:CodeFile {{id: item.id}})
                SET sf.status = 'triaged',
                    sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost,
                    sf.triaged_at = datetime(),
                    sf.claimed_at = null{set_clause}
                """,
                items=triaged_items,
            )

        if skipped_items:
            gc.query(
                f"""
                UNWIND $items AS item
                MATCH (sf:CodeFile {{id: item.id}})
                SET sf.status = 'skipped',
                    sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost,
                    sf.skip_reason = item.reason,
                    sf.triaged_at = datetime(),
                    sf.claimed_at = null{set_clause}
                """,
                items=skipped_items,
            )

    return {
        "triaged": len(triaged_items),
        "skipped": len(skipped_items),
        "triaged_ids": [item["id"] for item in triaged_items],
    }


def apply_file_scores(
    results: list[FileScoreResult],
    file_id_map: dict[str, str],
    content_decisions: list[dict[str, Any]],
    batch_cost: float = 0.0,
    content_cost: float = 0.0,
) -> dict[str, int]:
    """Persist a batch's score fields together with its content relevance.

    A file reaches ``scored`` only when its content decision succeeded: the
    dimension scores, the description and the content-arm relevance fields are
    one write.  The ingest claim requires ``relevance_stage='content'``, so
    admitting a file on a name-arm relevance or a stale status is not possible.
    A file whose content decision failed is left at its prior status and
    unclaimed, so a later score pass reclaims and retries it.

    Args:
        results: Score results from LLM.
        file_id_map: Mapping from path to CodeFile ID.
        content_decisions: One dict per file whose content decision succeeded,
            with ``path``, ``answers``, ``model`` and ``cost`` keys.
        batch_cost: Total description-call cost, distributed across the files
            actually scored.
        content_cost: Total content-decision cost, distributed across the files
            actually scored.

    Returns:
        Dict with ``scored`` and ``deferred`` counts.  A deferred file is one
        whose content decision failed and was left for a later pass.
    """
    decision_by_path = {d["path"]: d for d in content_decisions}

    matched = [r for r in results if file_id_map.get(r.path)]
    scored_count = sum(1 for r in matched if r.path in decision_by_path)
    score_cost_per_file = batch_cost / scored_count if scored_count > 0 else 0.0
    content_cost_per_file = content_cost / scored_count if scored_count > 0 else 0.0

    scored_items = []
    for result in matched:
        decision = decision_by_path.get(result.path)
        if decision is None:
            continue
        sf_id = file_id_map[result.path]
        item = _relevance_item(
            sf_id,
            decision["answers"],
            stage="content",
            model=decision.get("model"),
            cost=0.0,
        )
        item["score_cost"] = score_cost_per_file + content_cost_per_file
        item["score_composite"] = round(result.score_composite, 4)
        item["score_reason"] = result.description
        item["file_category"] = result.file_category
        item["score_modeling_code"] = result.score_modeling_code
        item["score_analysis_code"] = result.score_analysis_code
        item["score_operations_code"] = result.score_operations_code
        item["score_data_access"] = result.score_data_access
        item["score_workflow"] = result.score_workflow
        item["score_visualization"] = result.score_visualization
        item["score_documentation"] = result.score_documentation
        item["score_imas"] = result.score_imas
        item["score_convention"] = result.score_convention
        scored_items.append(item)

    if scored_items:
        set_clause = _relevance_set_clause()
        with GraphClient() as gc:
            gc.query(
                f"""
                UNWIND $items AS item
                MATCH (sf:CodeFile {{id: item.id}})
                SET sf.status = 'scored',
                    sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost,
                    sf.score_composite = item.score_composite,
                    sf.score_reason = item.score_reason,
                    sf.file_category = item.file_category,
                    sf.score_modeling_code = item.score_modeling_code,
                    sf.score_analysis_code = item.score_analysis_code,
                    sf.score_operations_code = item.score_operations_code{set_clause},
                    sf.score_data_access = item.score_data_access,
                    sf.score_workflow = item.score_workflow,
                    sf.score_visualization = item.score_visualization,
                    sf.score_documentation = item.score_documentation,
                    sf.score_imas = item.score_imas,
                    sf.score_convention = item.score_convention,
                    sf.scored_at = datetime(),
                    sf.claimed_at = null
                """,
                items=scored_items,
            )

    return {
        "scored": len(scored_items),
        "deferred": len(matched) - len(scored_items),
    }

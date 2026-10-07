"""Two-stage relevance for discovered CodeFiles, judged by a decisions model.

Pass 1 (Triage): the names arm asks six typed questions about the file from
minimal context — parent directory description, filename, sibling names and
the facility's data-access patterns.  A file's relevance is the largest of the
four scope probabilities (loads, processes, describes, maps_to_imas).  Files
whose relevance reaches the triage threshold proceed to enrichment.

Pass 2 (Score): the content arm asks the same six questions plus a graded
relevance Score and four facet Scores, judged from the file's preview and its
pattern evidence.  Beside each judgement a local model writes a one-sentence
description, and only for a file whose content relevance reaches the ingest
threshold.  Ingest admits a scored file whose content relevance reaches the
ingest threshold.

``score_composite`` is the one stored owner of a file's relevance: both arms
write it as the largest of the four scope probabilities of that decision.  The
names arm records it with ``relevance_stage='name'`` and the content arm
overwrites it with ``relevance_stage='content'``.  The stage field records
which arm produced the stored values.

Lifecycle: discovered → triaged → (enrich) → scored → ingested | skipped
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.discovery.base.reset import (
    CODE_RELEVANCE_FIELDS as RELEVANCE_FIELDS,
)
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score dimensions — the four Jev code facets stored on CodeFile
# ---------------------------------------------------------------------------

SCORE_DIMENSION_NAMES = [
    "score_data_access",
    "score_signal_processing",
    "score_machine_description",
    "score_imas_mapping",
]

# The four scope questions. A file's relevance is the largest noul over these.
SCOPE_NOULS = (
    "loads_diagnostic_data",
    "processes_diagnostic_signals",
    "describes_machine_or_diagnostics",
    "maps_to_imas",
)

# ``RELEVANCE_FIELDS`` is the registry of every relevance field a CodeFile
# carries, owned by ``discovery/base/reset.py`` so a reset and the decision arms
# that write these fields read the same list.  It lives there because that
# module is imported before the code package and importing the code package
# from ``reset.py`` would close an import cycle.


def _relevance_field(name: str) -> str:
    """Return ``name`` spelled by the registry, refusing an unregistered field."""
    if name not in RELEVANCE_FIELDS:
        raise KeyError(f"{name!r} is not a registered relevance field")
    return name


# decision question name -> CodeFile relevance field.  Each field is resolved
# through the registry, so a field the reset clears and the field the decision
# arm writes are the same string and cannot drift.
_RELEVANCE_FIELDS = {
    "loads_diagnostic_data": _relevance_field("relevance_loads"),
    "processes_diagnostic_signals": _relevance_field("relevance_processes"),
    "describes_machine_or_diagnostics": _relevance_field("relevance_describes"),
    "maps_to_imas": _relevance_field("relevance_imas"),
    "is_simulation": _relevance_field("relevance_simulation"),
}

# The decision arm that produced a file's stored relevance, recorded on
# ``relevance_stage``.  ``name`` is the names arm, which sets ``triaged``;
# ``content`` is the content arm, which admits a file to ingestion.
RELEVANCE_STAGE_NAME = "name"
RELEVANCE_STAGE_CONTENT = "content"


def relevance_predicate(
    alias: str,
    stage: str,
    threshold_param: str = "$min_relevance",
    facet_threshold_param: str | None = None,
) -> str:
    """The stage-and-relevance predicate every claim and has-work site shares.

    Requires *alias*'s stored relevance to come from *stage* — the arm that set
    the file's status — and compares its ``score_composite`` against
    *threshold_param*.  Rendered from one owner, a file's status and the
    relevance that carried it cannot drift apart between the claim, the
    has-work check and the count.

    When *facet_threshold_param* is given, the predicate also admits a file
    whose strongest facet reaches it.  A content-stage file whose composite is
    below the ingest gate can still carry the machine description or signal
    processing a mapper needs, so the clause reads the four facet fields the
    content arm writes.  It renders from :data:`ADMISSION_FACET_FIELDS`, the
    same list :func:`content_facet_relevance` reads, so the Cypher and Python
    arms cannot drift.
    """
    base = f"{alias}.relevance_stage = {stage!r}"
    if facet_threshold_param is None:
        return f"{base} AND {alias}.score_composite >= {threshold_param}"
    facet_clause = " OR ".join(
        f"{alias}.{field} >= {facet_threshold_param}"
        for field in ADMISSION_FACET_FIELDS
    )
    return (
        f"{base} AND ({alias}.score_composite >= {threshold_param} OR {facet_clause})"
    )


# The content arm's graded Score questions and the CodeFile field each fills.
# The stored value is the Score divided by its top level, so it lies in 0-1;
# ``relevance_grade`` keeps its own 0-4 scale.
FACET_QUESTION_FIELDS = {
    "data_access_depth": "score_data_access",
    "signal_processing_depth": "score_signal_processing",
    "machine_description_depth": "score_machine_description",
    "imas_mapping_depth": "score_imas_mapping",
}
RELEVANCE_GRADE_QUESTION = "relevance_grade"

# The facet fields the admission clause reads, in the order the content arm
# writes them.  Both the Cypher clause in :func:`relevance_predicate` and the
# Python value in :func:`content_facet_relevance` read this one list, so a
# facet the clause admits and a facet the description choice admits are the
# same four values.
ADMISSION_FACET_FIELDS = tuple(FACET_QUESTION_FIELDS.values())

# The role enum's fixed order. ``relevance_role_probs`` follows it, so a
# reader can compare two files' role distributions index by index.
RELEVANCE_ROLE_ORDER = (
    "diagnostic_data_access",
    "signal_processing",
    "machine_description",
    "imas_mapping",
    "simulation_or_solver",
    "visualization",
    "control_or_operations",
    "infrastructure_or_utility",
)


def _ordered_probs(probabilities: Any, keys: Any) -> list[float]:
    """A distribution as a list of floats ordered by *keys*.

    Missing levels read as 0.0, so the stored list always covers every offered
    level in the schema's fixed order however sparsely the model answered.
    """
    source = probabilities if isinstance(probabilities, dict) else {}
    out: list[float] = []
    for key in keys:
        value = source.get(key)
        if value is None:
            value = source.get(str(key), 0.0)
        out.append(round(float(value or 0.0), 4))
    return out


# ---------------------------------------------------------------------------
# Decisions questions and state
# ---------------------------------------------------------------------------


def build_triage_questions(with_content: bool = False) -> dict[str, Any]:
    """Render the decisions questions template into a questions mapping.

    The questions and their wording are declared in
    ``imas_codex/llm/prompts/code/triage.md`` and rendered through
    :func:`render_prompt`.  The names arm takes the six identity questions;
    the content arm additionally takes the graded relevance Score and the four
    facet Scores.
    """
    import json

    from imas_codex.llm.prompt_loader import render_prompt

    return json.loads(render_prompt("code/triage", {"with_content": with_content}))


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
    carries ``content_head`` — the first 1500 characters of the file preview —
    and the file's pattern evidence.
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
        state["file"]["pattern_evidence"] = {
            "categories": file_row.get("pattern_categories"),
            "total_matches": file_row.get("total_pattern_matches"),
            "line_count": file_row.get("line_count"),
        }
    return state


def scope_relevance(nouls: dict[str, float]) -> float:
    """A file's relevance: the largest of the four scope nouls."""
    return max(float(nouls[name]) for name in SCOPE_NOULS)


def triage_relevance(answers: dict[str, Any]) -> float:
    """A file's relevance from one decision's answers."""
    return scope_relevance({name: float(answers[name]["noul"]) for name in SCOPE_NOULS})


@lru_cache(maxsize=1)
def _content_top_levels() -> dict[str, int]:
    """The highest level index the content arm offers per Score question.

    Read from the rendered questions template so the stored facet value is
    always divided by the level count the model was actually shown.
    """
    questions = build_triage_questions(with_content=True)
    return {
        name: max(len(question.get("criteria") or []) - 1, 0)
        for name, question in questions.items()
        if question.get("type") == "score"
    }


def content_facet_relevance(answers: dict[str, Any]) -> float:
    """A file's strongest facet from one content decision's answers.

    Each facet answer's Score divided by its top level, over the four facet
    questions :data:`ADMISSION_FACET_FIELDS` names — the same values the
    admission clause compares against the stored facet fields.  ``answers`` is
    a content-arm decision, which carries a ``score`` per facet; the names arm
    does not, so a name-arm answer yields 0.
    """
    top_levels = _content_top_levels()
    best = 0.0
    for question in FACET_QUESTION_FIELDS:
        answer = answers.get(question) or {}
        top = top_levels.get(question, 0)
        score = float(answer.get("score", 0.0) or 0.0)
        best = max(best, score / top if top > 0 else 0.0)
    return best


def content_admits(
    answers: dict[str, Any],
    composite_threshold: float,
    facet_threshold: float,
) -> bool:
    """Whether a content decision admits a file to ingestion.

    True when the content-arm composite reaches *composite_threshold*, or when
    the strongest facet reaches *facet_threshold* — the Python mirror of the
    clause :func:`relevance_predicate` renders, so the description choice and
    the ingest claim agree.
    """
    return (
        triage_relevance(answers) >= composite_threshold
        or content_facet_relevance(answers) >= facet_threshold
    )


def _relevance_item(
    sf_id: str,
    answers: dict[str, Any],
    *,
    stage: str,
    model: str | None,
    cost: float,
) -> dict[str, Any]:
    """Build the persisted relevance fields from one decision's answers.

    Both arms write ``score_composite`` — the largest of the four scope
    probabilities — so the stored relevance and the gate never drift apart.
    The content arm additionally writes the graded relevance and the four facet
    Scores, each divided by its top level, and every judgement's distribution
    and confidence beside its value.
    """
    item: dict[str, Any] = {
        "id": sf_id,
        "score_cost": cost,
        "score_composite": round(triage_relevance(answers), 4),
        "relevance_stage": stage,
        "relevance_model": model or "",
        "relevance_role": (answers.get("role") or {}).get("choice") or "",
    }
    for question, field in _RELEVANCE_FIELDS.items():
        answer = answers.get(question) or {}
        item[field] = round(float(answer.get("noul", 0.0)), 4)

    role = answers.get("role") or {}
    item["relevance_role_probs"] = _ordered_probs(
        role.get("probabilities"), RELEVANCE_ROLE_ORDER
    )
    item["relevance_role_confidence"] = round(
        float(role.get("confidence", 0.0) or 0.0), 4
    )

    if stage == "content":
        # A content-stage answer set must carry every content question.  An
        # absent answer is not a score of zero: it means the question was never
        # asked, and writing it as a zero persists the absence as a judgement.
        missing = [
            question
            for question in (*FACET_QUESTION_FIELDS, RELEVANCE_GRADE_QUESTION)
            if question not in answers
        ]
        if missing:
            raise ValueError(
                f"content-stage answers for {sf_id!r} lack "
                f"{', '.join(sorted(missing))}; the content question set "
                "was not asked, so its answers cannot be recorded"
            )
        top_levels = _content_top_levels()
        for question, field in FACET_QUESTION_FIELDS.items():
            answer = answers.get(question) or {}
            top = top_levels.get(question, 0)
            score = float(answer.get("score", 0.0) or 0.0)
            item[field] = round(score / top, 4) if top > 0 else 0.0
            item[f"{field}_probs"] = _ordered_probs(
                answer.get("probabilities"), range(top + 1)
            )
            item[f"{field}_confidence"] = round(
                float(answer.get("confidence", 0.0) or 0.0), 4
            )
        grade = answers.get(RELEVANCE_GRADE_QUESTION) or {}
        grade_top = top_levels.get(RELEVANCE_GRADE_QUESTION, 0)
        item["relevance_grade"] = round(float(grade.get("score", 0.0) or 0.0), 4)
        item["relevance_grade_probs"] = _ordered_probs(
            grade.get("probabilities"), range(grade_top + 1)
        )
        item["relevance_grade_confidence"] = round(
            float(grade.get("confidence", 0.0) or 0.0), 4
        )
    return item


def _relevance_set_clause(*, include_content: bool = False) -> str:
    """The Cypher ``SET`` fragment writing every relevance field from an item."""
    clause = """,
                    sf.score_composite = item.score_composite,
                    sf.relevance_loads = item.relevance_loads,
                    sf.relevance_processes = item.relevance_processes,
                    sf.relevance_describes = item.relevance_describes,
                    sf.relevance_imas = item.relevance_imas,
                    sf.relevance_simulation = item.relevance_simulation,
                    sf.relevance_role = item.relevance_role,
                    sf.relevance_role_probs = item.relevance_role_probs,
                    sf.relevance_role_confidence = item.relevance_role_confidence,
                    sf.relevance_stage = item.relevance_stage,
                    sf.relevance_model = item.relevance_model"""
    if include_content:
        for field in (*FACET_QUESTION_FIELDS.values(), "relevance_grade"):
            clause += (
                f",\n                    sf.{field} = item.{field}"
                f",\n                    sf.{field}_probs = item.{field}_probs"
                f",\n                    sf.{field}_confidence = item.{field}_confidence"
            )
    return clause


# stored CodeFile field -> the scope question it carries the noul for
_STORED_SCOPE_FIELDS = {
    "loads_diagnostic_data": "relevance_loads",
    "processes_diagnostic_signals": "relevance_processes",
    "describes_machine_or_diagnostics": "relevance_describes",
    "maps_to_imas": "relevance_imas",
}


def _stored_composites(rows: list[dict[str, Any]]):
    for row in rows:
        nouls = {
            question: float(row.get(field) or 0.0)
            for question, field in _STORED_SCOPE_FIELDS.items()
        }
        yield (
            row["id"],
            row.get("stage"),
            float(row.get("stored") or 0.0),
            round(scope_relevance(nouls), 4),
        )


def recompute_stored_composites() -> dict[str, int]:
    """Rewrite ``score_composite`` on every staged CodeFile from its nouls.

    Both arms write ``score_composite`` as the largest of the four scope nouls,
    so a staged file whose stored composite is not that maximum disagrees with
    the rule ``_relevance_item`` applies.  This walks every CodeFile carrying a
    ``relevance_stage`` — both the ``name`` and ``content`` arms — and rewrites
    the field with the value ``scope_relevance`` produces, reporting how many
    rows disagreed before the write and how many still do after it.
    """
    fields = ", ".join(
        f"cf.{field} AS {field}" for field in _STORED_SCOPE_FIELDS.values()
    )
    select = f"""
        MATCH (cf:CodeFile)
        WHERE cf.relevance_stage IS NOT NULL
        RETURN cf.id AS id, cf.relevance_stage AS stage,
               cf.score_composite AS stored, {fields}
    """
    with GraphClient() as gc:
        before = list(_stored_composites(gc.query(select)))
        updates = [
            {"id": file_id, "composite": composite}
            for file_id, _stage, _stored, composite in before
        ]
        if updates:
            gc.query(
                """
                UNWIND $updates AS u
                MATCH (cf:CodeFile {id: u.id})
                SET cf.score_composite = u.composite
                """,
                updates=updates,
            )
        after = list(_stored_composites(gc.query(select)))

    def _differ(rows) -> int:
        return sum(
            1 for _id, _s, stored, composite in rows if abs(stored - composite) > 1e-9
        )

    return {
        "total": len(before),
        "differed_before": _differ(before),
        "differed_after": _differ(after),
        "content": sum(1 for _id, stage, _s, _c in before if stage == "content"),
        "name": sum(1 for _id, stage, _s, _c in before if stage == "name"),
    }


# ---------------------------------------------------------------------------
# Score models (description-only; relevance comes from the decisions model)
# ---------------------------------------------------------------------------


class FileScoreResult(BaseModel):
    """Description-only score result.

    The file's relevance and facet values come from the content-arm decision,
    not from this response; the local model writes only the one-sentence
    description, and only for a file the content decision admitted.
    """

    path: str = Field(description="The file path (echo from input)")
    description: str = Field(
        default="",
        description="Brief summary of what the file likely contains (1 sentence)",
    )


class FileScoreBatch(BaseModel):
    """Batch of file descriptions from the local model."""

    results: list[FileScoreResult]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_score_system_prompt(
    facility: str | None = None,
    focus: str | None = None,
) -> str:
    """Build the description-only system prompt for the local model."""
    from imas_codex.llm.prompt_loader import render_prompt

    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus

    return render_prompt("code/scorer", context)


def _build_score_user_prompt(file_groups: list[dict[str, Any]]) -> str:
    """Build the description-only user prompt — evidence + preview text."""
    lines = ["Describe these files using their evidence and content preview.\n"]

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


def apply_name_relevance(
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
    """Persist a batch's content relevance together with its descriptions.

    A file reaches ``scored`` when its content decision succeeded: the
    content-arm relevance and facet fields, the description and the status are
    one write.  The ingest claim requires ``relevance_stage='content'``, so
    admitting a file on a name-arm relevance or a stale status is not possible.
    A file whose content decision failed is left at its prior status and
    unclaimed, so a later score pass reclaims and retries it.

    The local model writes a description only for a file whose content
    relevance reaches the ingest threshold, so ``results`` covers that subset;
    every file with a successful content decision is still marked ``scored``.

    Args:
        results: Description-only score results from the local model.
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
    description_by_path = {r.path: r.description for r in results}

    matched = [d for d in content_decisions if file_id_map.get(d["path"])]
    scored_count = len(matched)
    score_cost_per_file = batch_cost / scored_count if scored_count > 0 else 0.0
    content_cost_per_file = content_cost / scored_count if scored_count > 0 else 0.0

    scored_items = []
    for decision in matched:
        sf_id = file_id_map[decision["path"]]
        item = _relevance_item(
            sf_id,
            decision["answers"],
            stage="content",
            model=decision.get("model"),
            cost=score_cost_per_file + content_cost_per_file,
        )
        item["score_reason"] = description_by_path.get(decision["path"], "")
        scored_items.append(item)

    if scored_items:
        set_clause = _relevance_set_clause(include_content=True)
        with GraphClient() as gc:
            gc.query(
                f"""
                UNWIND $items AS item
                MATCH (sf:CodeFile {{id: item.id}})
                SET sf.status = 'scored',
                    sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost,
                    sf.score_reason = coalesce(item.score_reason, sf.score_reason){set_clause},
                    sf.scored_at = datetime(),
                    sf.claimed_at = null
                """,
                items=scored_items,
            )

    return {
        "scored": len(scored_items),
        "deferred": len(content_decisions) - len(scored_items),
    }

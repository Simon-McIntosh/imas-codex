"""Graph storage operations for the standard-name pipeline."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from collections import Counter, defaultdict
from typing import Any

from imas_codex.standard_names.domain_decision import (
    DomainDecision,
    _undecided,
    write_name_domain,
)
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

from . import graph_foundation as foundation

logger = logging.getLogger(__name__)


def _stored_domain_decisions(
    gc: Any, sn_ids: list[str]
) -> dict[str, DomainDecision | None]:
    """The stored decisions for these names, keyed by every name the graph holds.

    A name the graph holds with no stored choice maps to ``None``: it has no
    decision to route through, so its current value is kept and the gap is
    recorded by the caller. A name the graph does not hold is absent, so a
    caller writing a batch never asks the one writer about a node that is not
    there.
    """
    if not sn_ids:
        return {}
    rows = gc.query(
        """
        UNWIND $ids AS id
        MATCH (sn:StandardName {id: id})
        RETURN sn.id AS id,
               sn.domain_decision_candidates AS candidates,
               sn.domain_decision_choice AS choice,
               sn.domain_decision_probability AS probability,
               sn.domain_decision_model AS model,
               sn.domain_decision_at AS decided_at
        """,
        ids=list(sn_ids),
    )
    decided: dict[str, DomainDecision | None] = {}
    for row in rows:
        choice = row.get("choice")
        if not choice:
            decided[str(row["id"])] = None
            continue
        decided[str(row["id"])] = DomainDecision(
            name=str(row["id"]),
            candidates=tuple(row.get("candidates") or ()),
            choice=str(choice),
            probability=row.get("probability"),
            model=row.get("model"),
            decided_at=row.get("decided_at")
            if isinstance(row.get("decided_at"), str)
            else None,
        )
    return decided


def get_source_name_mapping(*, rich: bool = False, services: Any) -> dict[str, dict]:
    """Return mapping of source_id → previous standard name details.

    Used by extract_worker in --force mode to inject per-path
    previous_name context so the LLM can improve on prior names.

    Args:
        rich: If True, return full SN metadata including documentation,
            links, and all linked DD paths. Used by --paths mode.

    Returns:
        Dict mapping source entity ID to dict with keys:
        name, description, kind, name_stage (and more if rich=True).
        If a source has multiple names, prefers the accepted one.
    """
    if rich:
        return services._get_rich_source_name_mapping()
    with services.GraphClient() as gc:
        results = gc.query(
            "\n            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)\n            RETURN src.id AS source_id,\n                   sn.id AS name,\n                   sn.description AS description,\n                   sn.kind AS kind,\n                   sn.name_stage AS name_stage\n        "
        )
        mapping: dict[str, dict] = {}
        for r in results:
            sid = r["source_id"]
            if sid not in mapping or r.get("name_stage") == "accepted":
                mapping[sid] = {
                    "name": r["name"],
                    "description": r.get("description"),
                    "kind": r.get("kind"),
                    "name_stage": r.get("name_stage"),
                }
        return mapping


def _get_rich_source_name_mapping(*, services: Any) -> dict[str, dict]:
    """Full SN metadata with documentation + all linked DD paths."""
    with services.GraphClient() as gc:
        results = gc.query(
            "\n            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)\n            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)\n            OPTIONAL MATCH (other_src)-[:HAS_STANDARD_NAME]->(sn)\n            WHERE other_src <> src\n            RETURN src.id AS source_id,\n                   sn.id AS name,\n                   sn.description AS description,\n                   sn.documentation AS documentation,\n                   sn.kind AS kind,\n                   sn.links AS links,\n                   sn.name_stage AS name_stage,\n                   sn.reviewer_score_name AS reviewer_score,\n                   sn.review_tier AS review_tier,\n                   sn.validation_issues AS validation_issues,\n                   u.id AS unit,\n                   collect(DISTINCT other_src.id) AS linked_dd_paths\n        "
        )
        mapping: dict[str, dict] = {}
        for r in results:
            sid = r["source_id"]
            if sid not in mapping or r.get("name_stage") == "accepted":
                mapping[sid] = {
                    "name": r["name"],
                    "description": r.get("description"),
                    "documentation": r.get("documentation"),
                    "kind": r.get("kind"),
                    "links": r.get("links"),
                    "name_stage": r.get("name_stage"),
                    "reviewer_score": r.get("reviewer_score"),
                    "review_tier": r.get("review_tier"),
                    "validation_issues": r.get("validation_issues"),
                    "unit": r.get("unit"),
                    "linked_dd_paths": [
                        p for p in r.get("linked_dd_paths") or [] if p != sid
                    ],
                }
        return mapping


def fetch_low_score_sources(
    *,
    min_score: float | None = None,
    domain: str | None = None,
    ids: str | None = None,
    limit: int | None = None,
    source_type: str = "dd",
    services: Any,
) -> list[dict[str, Any]]:
    """Enumerate sources whose linked StandardName has ``reviewer_score_name < min_score``.

    Walks ``(:IMASNode|:FacilitySignal)-[:HAS_STANDARD_NAME]->(:StandardName)``
    and returns the originating source IDs along with the reviewer feedback
    needed to prompt a targeted regeneration. Used by the extract worker's
    regen mode (``sn run --min-score F``) to re-queue the exact sources
    whose names scored below the threshold.

    Only reviewed names are considered (``reviewer_score_name IS NOT NULL``), so
    unreviewed names are never pulled into regen. Duplicates per source_id
    are collapsed, keeping the lowest-scoring entry ("worst critique wins").

    Args:
        min_score: Reviewer-score threshold (0-1). Names with
            ``reviewer_score_name < min_score`` are returned. Required for any
            results; None short-circuits to an empty list.
        domain: Optional physics-domain filter applied to the StandardName.
        ids: Optional IDS-name filter (DD source only).
        limit: Optional cap on rows returned (ordered by worst score first).
        source_type: ``"dd"`` or ``"signals"``.

    Returns:
        List of dicts with ``source_id``, ``source_type``, ``review_feedback``.
    """
    if min_score is None:
        return []
    if source_type == "dd":
        match_clause = "MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)"
    elif source_type == "signals":
        match_clause = (
            "MATCH (src:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)"
        )
    else:
        raise ValueError(f"source_type must be 'dd' or 'signals', got {source_type!r}")
    where_clauses = [
        "sn.reviewer_score_name IS NOT NULL",
        "sn.reviewer_score_name < $min_score",
        "coalesce(sn.regen_count, 0) < 1",
    ]
    params: dict[str, Any] = {"min_score": float(min_score)}
    if domain:
        where_clauses.append("sn.physics_domain = $domain")
        params["domain"] = domain
    if ids and source_type == "dd":
        match_clause += "\n            MATCH (src)-[:IN_IDS]->(ids_node:IDS)"
        where_clauses.append("ids_node.id = $ids")
        params["ids"] = ids
    query = f"\n        {match_clause}\n        WHERE {' AND '.join(where_clauses)}\n        RETURN src.id AS source_id,\n               sn.id AS previous_name,\n               sn.description AS previous_description,\n               sn.documentation AS previous_documentation,\n               sn.reviewer_score_name AS reviewer_score,\n               sn.review_tier AS review_tier,\n               sn.reviewer_comments_name AS reviewer_comments,\n               sn.reviewer_scores_name AS reviewer_scores_json,\n               sn.validation_status AS validation_status\n        ORDER BY coalesce(sn.reviewer_score_name, 1.0) ASC, src.id ASC\n    "
    if limit is not None and limit > 0:
        query += "\n        LIMIT $limit"
        params["limit"] = int(limit)
    with services.GraphClient() as gc:
        rows = list(gc.query(query, **params))
    by_source: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        sid = row.get("source_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores: dict[str, Any] | None = None
        if scores_json:
            try:
                scores = json.loads(scores_json)
            except (TypeError, ValueError):
                scores = None
        feedback = {
            "previous_name": row.get("previous_name"),
            "previous_description": row.get("previous_description"),
            "previous_documentation": row.get("previous_documentation"),
            "reviewer_score": row.get("reviewer_score"),
            "review_tier": row.get("review_tier"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores,
            "validation_status": row.get("validation_status"),
        }
        existing = by_source.get(sid)
        if existing is None:
            by_source[sid] = feedback
            order.append(sid)
        else:
            prev = existing.get("reviewer_score")
            cur = feedback.get("reviewer_score")
            if prev is None or (cur is not None and cur < prev):
                by_source[sid] = feedback
    return [
        {
            "source_id": sid,
            "source_type": source_type,
            "review_feedback": by_source[sid],
        }
        for sid in order
    ]


def fetch_review_feedback_for_sources(
    source_ids: list[str] | set[str] | None, *, services: Any
) -> dict[str, dict[str, Any]]:
    """Fetch prior reviewer feedback for a batch of StandardNameSource ids.

    Used by the compose worker when a regeneration run is invoked with
    ``--min-score F``. Each returned dict carries enough context
    to let the LLM understand what the previous reviewer objected to and
    adjust the new candidate accordingly.

    Args:
        source_ids: Iterable of source node ids (e.g. ``dd:equilibrium/...``
            or ``signals:tcv:...``). ``None`` or empty input returns ``{}``
            without hitting the graph.

    Returns:
        ``{source_id: feedback_dict}`` where feedback_dict has keys:

        - ``previous_name`` (str | None): prior standard-name id
        - ``previous_description`` (str | None)
        - ``previous_documentation`` (str | None)
        - ``reviewer_score`` (float | None): composite 0–1 score
        - ``review_tier`` (str | None): ``outstanding|good|inadequate|poor``
        - ``reviewer_comments`` (str | None): free-form reviewer critique
        - ``reviewer_scores`` (dict | None): parsed name-axis dimensional scores
        - ``validation_status`` (str | None): graph lifecycle state at
          fetch time

        A source's link to its name is read over ``PRODUCED_NAME`` — the
        authority — with ``HAS_STANDARD_NAME`` unioned in for callers that pass
        an upstream ``IMASNode``/``FacilitySignal`` id instead. Reading only
        ``HAS_STANDARD_NAME`` silently returned nothing for every
        ``StandardNameSource`` id, since that edge originates from the DD node
        (keyed on the bare path) and never from the source node (keyed
        ``dd:<path>``) — so an operator's ``sn edit --hint`` never reached the
        compose prompt.

        An entry is returned when the name carries prior review data OR an open
        edit. A hint is the operator's intent and does not depend on a review
        having happened, so requiring ``reviewer_score_name`` would drop the
        steering on exactly the names an edit just staged for regeneration.
        Sources with neither are omitted (a cold start; the caller skips
        feedback injection).
    """
    if not source_ids:
        return {}
    ids = sorted({sid for sid in source_ids if sid})
    if not ids:
        return {}
    with services.GraphClient() as gc:
        rows = gc.query(
            "\n            UNWIND $ids AS source_id\n            MATCH (src {id: source_id})\n            CALL {\n                WITH src\n                MATCH (src)-[:PRODUCED_NAME]->(sn:StandardName)\n                RETURN sn\n              UNION\n                WITH src\n                MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)\n                RETURN sn\n            }\n            WITH source_id, sn\n            WHERE sn.reviewer_score_name IS NOT NULL\n               OR coalesce(sn.edit_mode, '') <> ''\n            RETURN source_id AS source_id,\n                   sn.id AS previous_name,\n                   sn.description AS previous_description,\n                   sn.documentation AS previous_documentation,\n                   sn.reviewer_score_name AS reviewer_score,\n                   sn.review_tier AS review_tier,\n                   sn.reviewer_comments_name AS reviewer_comments,\n                   sn.reviewer_scores_name AS reviewer_scores_json,\n                   sn.reviewer_suggested_name AS reviewer_suggested_name,\n                   sn.reviewer_suggestion_justification_name\n                       AS reviewer_suggestion_justification,\n                   sn.validation_status AS validation_status,\n                   sn.edit_mode AS edit_mode,\n                   sn.name_hint AS name_hint,\n                   sn.docs_hint AS docs_hint,\n                   sn.edit_reason AS edit_reason,\n                   sn.edit_origin AS edit_origin\n            ",
            ids=ids,
        )
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("source_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores_dict: dict[str, Any] | None = None
        if scores_json:
            try:
                scores_dict = json.loads(scores_json)
            except (TypeError, ValueError):
                scores_dict = None
        new_entry = {
            "previous_name": row.get("previous_name"),
            "previous_description": row.get("previous_description"),
            "previous_documentation": row.get("previous_documentation"),
            "reviewer_score": row.get("reviewer_score"),
            "review_tier": row.get("review_tier"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores_dict,
            "reviewer_suggested_name": row.get("reviewer_suggested_name") or None,
            "reviewer_suggestion_justification": row.get(
                "reviewer_suggestion_justification"
            )
            or None,
            "validation_status": row.get("validation_status"),
            "edit_mode": row.get("edit_mode"),
            "name_hint": row.get("name_hint"),
            "docs_hint": row.get("docs_hint"),
            "edit_reason": row.get("edit_reason"),
            "edit_origin": row.get("edit_origin"),
        }
        existing = mapping.get(sid)
        if existing is None:
            mapping[sid] = new_entry
            continue
        prev = existing.get("reviewer_score")
        cur = new_entry.get("reviewer_score")
        if prev is None or (cur is not None and cur < prev):
            mapping[sid] = new_entry
    return mapping


def fetch_reviewer_history_for_sources(
    source_ids: list[str] | set[str] | None, *, services: Any
) -> dict[str, dict[str, Any]]:
    """Fetch full StandardNameReview-node history per source for compose prompt enrichment.

    For each source that currently links to a StandardName with ≥1
    StandardNameReview node, returns:

    - ``latest``: the most recent StandardNameReview (score, comment ≤800 chars, model).
    - ``prior_themes``: n-gram theme extraction from older StandardNameReview comments,
      rendered as ``[{theme, count, example}]``.

    This complements :func:`fetch_review_feedback_for_sources` which injects
    only the denormalised aggregates from the StandardName node itself.  The
    history variant drills into *all* StandardNameReview nodes for richer compose context.

    Args:
        source_ids: Iterable of source-node ids (e.g. ``dd:equilibrium/...``).
            ``None`` or empty input returns ``{}`` without hitting the graph.

    Returns:
        ``{source_id: history_dict}`` where history_dict has keys:

        - ``latest`` (dict): ``{score, comment, model}`` of newest review.
        - ``prior_themes`` (list[dict]): theme summaries from older reviews,
          each ``{theme, count, example}``.  Empty when only one review exists.
    """
    if not source_ids:
        return {}
    ids = sorted({sid for sid in source_ids if sid})
    if not ids:
        return {}
    with services.GraphClient() as gc:
        rows = gc.query(
            "\n            UNWIND $ids AS source_id\n            MATCH (src {id: source_id})-[:HAS_STANDARD_NAME]->(sn:StandardName)\n                  -[:HAS_REVIEW]->(r:StandardNameReview)\n            RETURN source_id,\n                   sn.id AS sn_id,\n                   r.score AS score,\n                   r.comments AS full_comment,\n                   r.model AS model,\n                   r.reviewed_at AS ts\n            ORDER BY source_id, ts DESC\n            ",
            ids=ids,
        )
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = row.get("source_id")
        if sid:
            by_source[sid].append(row)
    mapping: dict[str, dict[str, Any]] = {}
    for sid, reviews in by_source.items():
        if not reviews:
            continue
        latest = reviews[0]
        comment_text = latest.get("full_comment") or ""
        history: dict[str, Any] = {
            "latest": {
                "score": latest.get("score"),
                "comment": comment_text[:800],
                "model": latest.get("model"),
            },
            "prior_themes": [],
        }
        if len(reviews) > 1:
            prior_comments = [
                r.get("full_comment") or ""
                for r in reviews[1:]
                if r.get("full_comment")
            ]
            if prior_comments:
                from imas_codex.standard_names.review.themes import (
                    _extract_themes_from_texts,
                )

                themes = _extract_themes_from_texts(prior_comments, top_n=5)
                for theme in themes:
                    example = ""
                    theme_words = set(theme.lower().split())
                    for c in prior_comments:
                        c_lower = c.lower()
                        if any(w in c_lower for w in theme_words):
                            example = c[:160]
                            break
                    history["prior_themes"].append(
                        {"theme": theme, "count": 1, "example": example}
                    )
        mapping[sid] = history
    return mapping


def fetch_docs_review_feedback_for_standard_names(
    sn_ids: list[str] | set[str] | None, *, services: Any
) -> dict[str, dict[str, Any]]:
    """Fetch prior docs-axis reviewer feedback keyed by StandardName id.

    Used by the enrich contextualise worker when re-enriching SNs that
    already have a docs-axis review on file.  Lets the LLM target the
    specific weaknesses (description quality, documentation_quality,
    completeness, physics_accuracy) the reviewer flagged previously
    instead of running blind.

    This is the docs-axis analogue of
    :func:`fetch_review_feedback_for_sources` — it keys on SN id (since
    enrich operates on SN nodes directly, not on sources) and pulls the
    ``*_docs`` reviewer fields plus ``validation_issues``.

    Args:
        sn_ids: Iterable of StandardName ids. ``None`` or empty returns
            ``{}`` without hitting the graph.

    Returns:
        ``{sn_id: feedback_dict}`` where feedback_dict has keys:

        - ``reviewer_score`` (float | None): docs-axis 0–1 score.
        - ``reviewer_comments`` (str | None): free-form docs critique.
        - ``reviewer_scores`` (dict | None): parsed docs-axis dimensional
          scores (description_quality / documentation_quality /
          completeness / physics_accuracy).
        - ``validation_issues`` (list[str] | None): tagged ISN validation
          issue strings, when present.

        Only SNs with ``reviewer_score_docs IS NOT NULL`` are returned —
        cold-start enrichments are silently omitted.
    """
    if not sn_ids:
        return {}
    ids = sorted({sid for sid in sn_ids if sid})
    if not ids:
        return {}
    with services.GraphClient() as gc:
        rows = gc.query(
            "\n            UNWIND $ids AS sn_id\n            MATCH (sn:StandardName {id: sn_id})\n            WHERE sn.reviewer_score_docs IS NOT NULL\n            RETURN sn_id AS sn_id,\n                   sn.reviewer_score_docs AS reviewer_score,\n                   sn.reviewer_comments_docs AS reviewer_comments,\n                   sn.reviewer_scores_docs AS reviewer_scores_json,\n                   sn.validation_issues AS validation_issues\n            ",
            ids=ids,
        )
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("sn_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores_dict: dict[str, Any] | None = None
        if scores_json:
            try:
                scores_dict = json.loads(scores_json)
            except (TypeError, ValueError):
                scores_dict = None
        issues = row.get("validation_issues")
        if isinstance(issues, str):
            issues = [issues]
        elif issues is None:
            issues = None
        mapping[sid] = {
            "reviewer_score": row.get("reviewer_score"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores_dict,
            "validation_issues": issues if issues else None,
        }
    return mapping


def fetch_docs_review_feedback_for_sns(
    sn_ids: list[str] | set[str] | None, *, services: Any
) -> dict[str, dict[str, Any]]:
    """Deprecated alias of :func:`fetch_docs_review_feedback_for_standard_names`."""
    import warnings

    warnings.warn(
        "fetch_docs_review_feedback_for_sns is deprecated; use fetch_docs_review_feedback_for_standard_names.",
        DeprecationWarning,
        stacklevel=2,
    )
    return services.fetch_docs_review_feedback_for_standard_names(sn_ids)


def _filter_admissible_parents(
    co_batch: list[dict[str, Any]],
    gc: Any,
    *,
    full_rebuild: bool = False,
    services: Any,
) -> list[dict[str, Any]]:
    """Drop HAS_PARENT edges whose target fails the admission gate.

    Two-clause gate (see the admission gate in ``parents.py``):
      A. IR specificity — keep if the target name has qualifiers, locus,
         operators, projection, or mechanism in its ISN parse.
      B. Vector-like topology — keep bare-base targets if they have or
         will have ≥2 distinct-axis projection children. Considers
         both already-in-graph projections and the current batch.

    Skips legacy entries: if a parent has ``origin='catalog_edit'`` or
    is already ``name_stage='accepted'``, the edge is preserved unless the
    parent is superseded.

    ``full_rebuild`` (set by ``rederive_structural_edges``, which passes the
    COMPLETE set of live names) makes admission **batch-authoritative**: Clause
    B axes and the single-child shadow veto are evaluated from the batch alone,
    ignoring transient graph topology. This is essential for idempotency — the
    stale, pre-reconcile graph children would otherwise flip the shadow-veto
    verdict for qualifier single-child parents (admitted one startup, vetoed
    and reaped the next), an unbounded mint/reap oscillation. With the batch as
    the sole source of truth, the verdict is a pure function of the live-name
    set, so the re-derivation reaches a fixpoint.
    """
    from imas_codex.standard_names.parents import (
        is_admissible_parent_name,
        is_algebraic_decomposition_edge,
        superseded_parent_names,
    )

    if not co_batch:
        return co_batch
    targets = {e["to_name"] for e in co_batch}
    batch_axes: dict[str, set[str]] = {}
    batch_children: dict[str, set[str]] = {}
    for e in co_batch:
        if e.get("operator_kind") == "projection" and e.get("axis"):
            batch_axes.setdefault(e["to_name"], set()).add(e["axis"])
        if e.get("from_name"):
            batch_children.setdefault(e["to_name"], set()).add(e["from_name"])
    graph_axes: dict[str, set[str]] = {t: set() for t in targets}
    graph_children: dict[str, set[str]] = {t: set() for t in targets}
    shadow_info: dict[str, dict[str, Any]] = {}
    legacy_protected: set[str] = set()
    try:
        rows = list(
            gc.query(
                "\n                UNWIND $names AS nm\n                OPTIONAL MATCH (child:StandardName)-[r:HAS_PARENT]->(p:StandardName {id: nm})\n                  WHERE r.operator_kind = 'projection' AND r.axis IS NOT NULL\n                WITH nm, collect(DISTINCT r.axis) AS axes\n                OPTIONAL MATCH (anychild:StandardName)-[:HAS_PARENT]->(p3:StandardName {id: nm})\n                WITH nm, axes, collect(DISTINCT anychild) AS children\n                OPTIONAL MATCH (psrc:IMASNode)-[:HAS_STANDARD_NAME]->(p4:StandardName {id: nm})\n                WITH nm, axes, children, collect(DISTINCT psrc.id) AS parent_sources\n                OPTIONAL MATCH (csrc:IMASNode)-[:HAS_STANDARD_NAME]->(lone)\n                  WHERE size(children) = 1 AND lone = children[0]\n                WITH nm, axes, children, parent_sources,\n                     collect(DISTINCT csrc.id) AS lone_child_sources\n                OPTIONAL MATCH (p2:StandardName {id: nm})\n                RETURN nm AS name, axes,\n                       [c IN children | c.id] AS child_ids,\n                       CASE WHEN size(children) = 1\n                            THEN children[0].id ELSE null END AS lone_child_id,\n                       CASE WHEN size(children) = 1\n                            THEN coalesce(children[0].name_stage, '')\n                            ELSE null END AS lone_child_stage,\n                       CASE WHEN size(children) = 1\n                            THEN coalesce(children[0].origin, 'pipeline')\n                            ELSE null END AS lone_child_origin,\n                       parent_sources,\n                       lone_child_sources,\n                       p2.origin AS origin,\n                       p2.name_stage AS name_stage\n                ",
                names=list(targets),
            )
        )
    except Exception:
        logger.exception("Parent-admission topology probe failed; aborting reconcile")
        raise
    superseded_targets = superseded_parent_names(gc, targets, lifecycle_rows=rows)
    for r in rows:
        nm = r.get("name")
        if not nm:
            continue
        graph_axes[nm] = set(r.get("axes") or [])
        graph_children[nm] = {c for c in r.get("child_ids") or [] if c}
        if r.get("origin") == "catalog_edit" or r.get("name_stage") == "accepted":
            legacy_protected.add(nm)
        if r.get("lone_child_id"):
            shadow_info[nm] = {
                "child_id": r.get("lone_child_id"),
                "child_stage": r.get("lone_child_stage") or "",
                "child_origin": r.get("lone_child_origin") or "pipeline",
                "child_sources": {s for s in r.get("lone_child_sources") or [] if s},
                "parent_sources": {s for s in r.get("parent_sources") or [] if s},
            }
    if full_rebuild:
        graph_axes = {t: set() for t in targets}
        graph_children = {t: set() for t in targets}
        shadow_info = {}
    lone_pairs = []
    for target in targets:
        all_children = graph_children.get(target, set()) | batch_children.get(
            target, set()
        )
        if len(all_children) == 1 and target not in shadow_info:
            lone_pairs.append({"target": target, "child": next(iter(all_children))})
    if lone_pairs:
        try:
            srows = list(
                gc.query(
                    "\n                    UNWIND $pairs AS pr\n                    MATCH (child:StandardName {id: pr.child})\n                    OPTIONAL MATCH (csrc:IMASNode)-[:HAS_STANDARD_NAME]->(child)\n                    OPTIONAL MATCH (psrc:IMASNode)-[:HAS_STANDARD_NAME]->\n                          (:StandardName {id: pr.target})\n                    RETURN pr.target AS target, pr.child AS child,\n                           coalesce(child.name_stage, '') AS child_stage,\n                           coalesce(child.origin, 'pipeline') AS child_origin,\n                           collect(DISTINCT csrc.id) AS child_sources,\n                           collect(DISTINCT psrc.id) AS parent_sources\n                    ",
                    pairs=lone_pairs,
                )
            )
        except Exception as exc:
            logger.warning("Batch shadow-source probe failed: %s; skipping veto", exc)
            srows = []
        for r in srows:
            shadow_info[r["target"]] = {
                "child_id": r["child"],
                "child_stage": r.get("child_stage") or "",
                "child_origin": r.get("child_origin") or "pipeline",
                "child_sources": {s for s in r.get("child_sources") or [] if s},
                "parent_sources": {s for s in r.get("parent_sources") or [] if s},
            }

    def _is_single_child_shadow(target: str) -> tuple[bool, str]:
        """Batch-aware single-child shadow veto (mirrors parents.is_single_child_shadow).

        Combines graph children with the children proposed in THIS batch so a
        parent that genuinely groups ≥2 names is never vetoed, even before the
        sibling edges are written.
        """
        all_children = graph_children.get(target, set()) | batch_children.get(
            target, set()
        )
        if len(all_children) != 1:
            return (False, "not a single-child parent")
        info = shadow_info.get(target)
        if not info:
            return (False, "single child source probe returned no match")
        if is_algebraic_decomposition_edge(info["child_id"], target):
            return (False, "single child requires this algebraic decomposition target")
        if info["child_stage"] in ("superseded", "exhausted"):
            return (False, "lone child superseded/exhausted")
        if info["child_origin"] == "derived":
            return (False, "lone child is itself derived")
        child_sources = info["child_sources"]
        parent_sources = info["parent_sources"]
        if not child_sources:
            return (False, "lone child has no DD source")
        if parent_sources - child_sources:
            return (False, "parent independently sourced — not a shadow")
        return (True, f"single-child shadow of {info['child_id']}")

    def _admit(target: str) -> tuple[bool, str]:
        if target in superseded_targets:
            return (False, "superseded parent")
        if target in legacy_protected:
            return (True, "legacy (catalog_edit or accepted)")
        algebraic_children = graph_children.get(target, set()) | batch_children.get(
            target, set()
        )
        if any(
            is_algebraic_decomposition_edge(child, target)
            for child in algebraic_children
        ):
            return (True, "strict algebraic decomposition target")
        suppress, suppress_reason = _is_single_child_shadow(target)
        if suppress:
            return (False, f"suppressed: {suppress_reason}")
        result = is_admissible_parent_name(target, gc=None)
        if result.admit:
            return (True, f"clause A: {result.reason}")
        combined = graph_axes.get(target, set()) | batch_axes.get(target, set())
        if len(combined) >= 2:
            return (True, f"clause B: vector-like ({len(combined)} axes)")
        return (False, result.reason)

    kept: list[dict[str, Any]] = []
    dropped_count = 0
    dropped_targets: set[str] = set()
    for e in co_batch:
        admit, reason = _admit(e["to_name"])
        if admit:
            kept.append(e)
        else:
            dropped_count += 1
            dropped_targets.add(e["to_name"])
    if dropped_count:
        logger.info(
            "Parent admission gate: dropped %d HAS_PARENT edges to %d inadmissible targets (sample: %s)",
            dropped_count,
            len(dropped_targets),
            sorted(dropped_targets)[:5],
        )
    return kept


def _emit_magnitude_of_edges(
    names: list[dict[str, Any]], gc: Any, *, services: Any
) -> None:
    """Emit MAGNITUDE_OF edges for ``magnitude_of_<X>`` names.

    Passive — only fires when:
      1. The name being written matches the pattern ``magnitude_of_<X>``.
      2. ``<X>`` already exists in the graph as a ``kind='vector'`` SN.
      3. ``<X>`` is an admitted parent (not catalog-only metadata).

    Never speculatively creates magnitude SNs; the magnitude must already
    be sourced from a DD path or facility signal.
    """
    candidates: list[dict[str, str]] = []
    for n in names:
        name_id = n.get("id")
        if not name_id or not name_id.startswith("magnitude_of_"):
            continue
        vector = name_id[len("magnitude_of_") :]
        if not vector:
            continue
        candidates.append({"magnitude": name_id, "vector": vector})
    if not candidates:
        return
    gc.query(
        "\n        UNWIND $rows AS r\n        MATCH (m:StandardName {id: r.magnitude})\n        MATCH (v:StandardName {id: r.vector})\n        WHERE v.kind = 'vector'\n        MERGE (m)-[:MAGNITUDE_OF]->(v)\n        ",
        rows=candidates,
    )


def _bootstrap_missing_derived_parent_targets(
    gc: Any,
    parent_edges: list[dict[str, Any]],
    missing_parent_ids: set[str],
    *,
    services: Any,
) -> set[str]:
    """Fully materialize absent admitted parents before writing their edges.

    Parents are prepared from already-authorized child nodes, one dependency
    layer at a time. The canonical materializer performs unit derivation and
    strict identity validation; a row it cannot prove is left absent and its
    relationships are filtered by the caller.
    """
    remaining = set(missing_parent_ids)
    materialized: set[str] = set()
    edges_by_parent: dict[str, list[dict[str, Any]]] = {}
    for edge in parent_edges:
        if edge["to_name"] in remaining:
            edges_by_parent.setdefault(edge["to_name"], []).append(edge)
    while remaining:
        pairs = [
            {
                "parent_id": parent_id,
                "child_id": edge["from_name"],
                "operator_kind": edge.get("operator_kind"),
            }
            for parent_id in sorted(remaining)
            for edge in edges_by_parent.get(parent_id, [])
        ]
        child_rows = list(
            gc.query(
                "\n                // STANDARD_NAME_PARENT_BOOTSTRAP_CHILDREN\n                UNWIND $pairs AS pair\n                MATCH (child:StandardName {id: pair.child_id})\n                RETURN pair.parent_id AS parent_id,\n                       child.id AS id,\n                       child.name_stage AS name_stage,\n                       child.unit AS unit,\n                       child.cocos_transformation_type AS cocos,\n                       child.physics_domain AS physics_domain,\n                       child.kind AS kind,\n                       pair.operator_kind AS op_kind\n                ",
                pairs=pairs,
            )
        )
        by_parent: dict[str, list[dict[str, Any]]] = {}
        for row in child_rows:
            by_parent.setdefault(str(row["parent_id"]), []).append(dict(row))
        progressed = False
        for parent_id in sorted(remaining):
            expected_children = {
                edge["from_name"] for edge in edges_by_parent.get(parent_id, [])
            }
            child_data = by_parent.get(parent_id, [])
            if {str(row["id"]) for row in child_data} != expected_children:
                continue
            inherited_units = {
                str(row["unit"])
                for row in child_data
                if row.get("unit") and row.get("op_kind") != "binary"
            }
            authorized_unit = (
                next(iter(inherited_units)) if len(inherited_units) == 1 else None
            )
            child_by_id = {str(child["id"]): child for child in child_data}
            bootstrap_edges = []
            for edge in edges_by_parent.get(parent_id, []):
                child = child_by_id[edge["from_name"]]
                bootstrap_edges.append(
                    {
                        **edge,
                        "expected_name_stage": child.get("name_stage"),
                        "expected_unit": child.get("unit"),
                        "expected_cocos": child.get("cocos"),
                        "expected_physics_domain": child.get("physics_domain"),
                        "expected_kind": child.get("kind"),
                    }
                )
            seeded = services._materialize_derived_parent_rows(
                gc,
                [
                    {
                        "parent_id": parent_id,
                        "child_data": child_data,
                        "edge_kinds": [
                            edge.get("operator_kind")
                            for edge in edges_by_parent.get(parent_id, [])
                        ],
                        "authorized_unit": authorized_unit,
                        "bootstrap_edges": bootstrap_edges,
                    }
                ],
                bootstrap_missing=True,
            )
            if seeded == 1:
                materialized.add(parent_id)
                remaining.remove(parent_id)
                progressed = True
        if not progressed:
            break
    if remaining:
        logger.warning(
            "Refused %d missing derived-parent target(s) without complete authority: %s",
            len(remaining),
            ", ".join(sorted(remaining)[:5]),
        )
    return materialized


def _write_standard_name_edges(
    gc: Any,
    names: list[dict[str, Any]],
    *,
    full_rebuild: bool = False,
    expand_closure: bool = True,
    services: Any,
) -> set[str]:
    """Emit all structural edges for a batch of StandardName nodes.

    Called as a tail pass **after** all nodes in the batch have been
    written. Forward-reference parents are admitted and fully materialized
    before relationship persistence. No relationship writer creates a
    ``StandardName`` endpoint.

    Ordered operator trees are expanded to closure in this pass.  Each
    ``derive_edges`` call peels exactly one outer layer; newly discovered
    structural parents are queued until every ordered unary/binary branch has
    been projected.  This makes the graph lossless even when intermediate
    parent nodes have not yet appeared in a source batch.

    Handles the following edge types:

    - ``HAS_PARENT``: derived from the ISN grammar parser (one layer
      per call: unary prefix/postfix, binary, or projection).
    - ``HAS_ERROR``: uncertainty siblings — direction inverted relative to
      the derivation (inner → uncertainty form).
    - ``HAS_PREDECESSOR``: from ``predecessor`` or ``deprecates`` field.
    - Successor edges are written by signed supersede mutations.
    - ``IN_CLUSTER``: from ``primary_cluster_id`` field.
    - ``HAS_PHYSICS_DOMAIN``: from ``physics_domain`` field.

    Parameters
    ----------
    gc:
        Active ``GraphClient`` context (already open — do not open a new
        one inside this function).
    names:
        List of name dicts, each containing at minimum ``id``.  All other
        fields are optional; missing fields produce no edges.
    expand_closure:
        Also reconcile structural parent identities discovered while deriving
        the requested names.  Disable for exact-name repair callers.

    Returns
    -------
    The finite set of authorized ``StandardName`` identities involved in this
    relation projection.
    """
    from imas_codex.standard_names.derivation import derive_edges

    co_batch: list[dict[str, Any]] = []
    he_batch: list[dict[str, Any]] = []
    geo_batch: list[dict[str, Any]] = []
    pending = [n["id"] for n in names if n.get("id")]
    processed: set[str] = set()
    while pending:
        name_id = pending.pop()
        if name_id in processed:
            continue
        processed.add(name_id)
        for edge in derive_edges(name_id):
            if edge.edge_type == "HAS_PARENT":
                co_batch.append(
                    {
                        "from_name": edge.from_name,
                        "to_name": edge.to_name,
                        "operator": edge.props.get("operator"),
                        "operator_kind": edge.props.get("operator_kind"),
                        "role": edge.props.get("role"),
                        "separator": edge.props.get("separator"),
                        "axis": edge.props.get("axis"),
                        "shape": edge.props.get("shape"),
                    }
                )
                if expand_closure and edge.to_name not in processed:
                    pending.append(edge.to_name)
            elif edge.edge_type == "HAS_ERROR":
                he_batch.append(
                    {
                        "from_name": edge.from_name,
                        "to_name": edge.to_name,
                        "error_type": edge.props.get("error_type"),
                    }
                )
                if expand_closure and edge.from_name not in processed:
                    pending.append(edge.from_name)
            elif edge.edge_type == "HAS_LOCUS":
                geo_batch.append(
                    {
                        "from_name": edge.from_name,
                        "locus_token": edge.props.get("locus_token"),
                        "locus_relation": edge.props.get("locus_relation"),
                    }
                )
    co_batch = services._filter_admissible_parents(
        co_batch, gc, full_rebuild=full_rebuild
    )
    explicit_ids = {n["id"] for n in names if n.get("id")}
    endpoint_ids = set(explicit_ids)
    for batch in (co_batch, he_batch):
        for row in batch:
            endpoint_ids.update((row["from_name"], row["to_name"]))
    endpoint_ids.update(row["from_name"] for row in geo_batch if row.get("from_name"))
    pred_batch: list[dict[str, str]] = []
    for n in names:
        name_id = n.get("id")
        if not name_id:
            continue
        predecessor = n.get("predecessor") or n.get("deprecates")
        if predecessor:
            pred_batch.append({"from_name": name_id, "to_name": predecessor})
            endpoint_ids.add(predecessor)
    discovered = list(
        gc.query(
            "\n            // STANDARD_NAME_EDGE_IDENTITY_DISCOVERY\n            UNWIND $ids AS requested\n            OPTIONAL MATCH (sn:StandardName {id: requested})\n            RETURN sn.id AS id, sn.name_stage AS name_stage\n            ",
            ids=sorted(endpoint_ids),
        )
    )
    if endpoint_ids and (not discovered):
        lifecycle_ids = set(endpoint_ids)
    else:
        lifecycle_ids = {
            str(row["id"])
            for row in discovered
            if row.get("id")
            and ("name_stage" not in row or row.get("name_stage") is not None)
        }
    unmaterialized_parent_ids = {
        row["to_name"] for row in co_batch if row["to_name"] not in lifecycle_ids
    }
    materialized_ids = services._bootstrap_missing_derived_parent_targets(
        gc, co_batch, unmaterialized_parent_ids
    )
    authorized_ids = explicit_ids | lifecycle_ids | materialized_ids
    admitted_parent_ids = {row["to_name"] for row in co_batch}
    structural_ids = explicit_ids | admitted_parent_ids & (
        lifecycle_ids | materialized_ids
    )
    co_batch = [
        row
        for row in co_batch
        if row["from_name"] in structural_ids and row["to_name"] in structural_ids
    ]
    he_batch = [
        row
        for row in he_batch
        if row["from_name"] in authorized_ids and row["to_name"] in authorized_ids
    ]
    geo_batch = [row for row in geo_batch if row["from_name"] in structural_ids]
    pred_batch = [
        row
        for row in pred_batch
        if row["from_name"] in authorized_ids and row["to_name"] in authorized_ids
    ]
    desired_by_child: dict[str, set[str]] = {}
    for e in co_batch:
        if e.get("from_name"):
            desired_by_child.setdefault(e["from_name"], set()).add(e["to_name"])
    recon = [
        {"child": child, "keep": sorted(desired_by_child.get(child, set()))}
        for child in sorted(processed)
    ]
    if recon:
        gc.query(
            "\n            UNWIND $recon AS rc\n            MATCH (c:StandardName {id: rc.child})-[r:HAS_PARENT]->(p:StandardName)\n            WHERE r.operator_kind IS NOT NULL AND NOT (p.id IN rc.keep)\n            DELETE r\n            ",
            recon=recon,
        )
    if co_batch:
        gc.query(
            "\n            UNWIND $batch AS b\n            MATCH (src:StandardName {id: b.from_name})\n            MATCH (tgt:StandardName {id: b.to_name})\n            MERGE (src)-[r:HAS_PARENT]->(tgt)\n            SET r.operator      = b.operator,\n                r.operator_kind = b.operator_kind,\n                r.role          = b.role,\n                r.separator     = b.separator,\n                r.axis          = b.axis,\n                r.shape         = b.shape\n            WITH tgt, b.operator_kind AS ok\n            WHERE tgt.name_stage IS NULL OR tgt.name_stage = 'pending'\n            WITH tgt, ok\n            WHERE ok IN ['projection', 'coordinate', 'unary_postfix']\n            SET tgt.updated_at = datetime(), tgt.needs_composition = true\n            ",
            batch=co_batch,
        )
        services._emit_magnitude_of_edges(
            [{"id": name_id} for name_id in processed], gc
        )
    if he_batch:
        gc.query(
            "\n            UNWIND $batch AS b\n            MATCH (src:StandardName {id: b.from_name})\n            MATCH (tgt:StandardName {id: b.to_name})\n            MERGE (src)-[r:HAS_ERROR]->(tgt)\n            SET r.error_type = b.error_type\n            ",
            batch=he_batch,
        )
    if geo_batch:
        gc.query(
            "\n            UNWIND $batch AS b\n            MATCH (src:StandardName {id: b.from_name})\n            MERGE (loc:Locus {id: b.locus_token})\n            MERGE (src)-[r:HAS_LOCUS]->(loc)\n            SET r.locus_token = b.locus_token,\n                r.locus_relation = b.locus_relation\n            ",
            batch=geo_batch,
        )
    if pred_batch:
        gc.query(
            "\n            UNWIND $batch AS b\n            MATCH (src:StandardName {id: b.from_name})\n            MATCH (tgt:StandardName {id: b.to_name})\n            MERGE (src)-[:HAS_PREDECESSOR]->(tgt)\n            ",
            batch=pred_batch,
        )
    cluster_batch = [
        {"sn_id": n["id"], "cluster_id": n["primary_cluster_id"]}
        for n in names
        if n.get("id") and n.get("primary_cluster_id")
    ]
    if cluster_batch:
        gc.query(
            "\n            UNWIND $batch AS b\n            MATCH (sn:StandardName {id: b.sn_id})\n            MERGE (c:IMASSemanticCluster {id: b.cluster_id})\n            MERGE (sn)-[:IN_CLUSTER]->(c)\n            ",
            batch=cluster_batch,
        )
    name_ids = [str(n["id"]) for n in names if n.get("id")]
    domain_batch = []
    for sn_id, stored in _stored_domain_decisions(gc, name_ids).items():
        write_name_domain(
            gc,
            sn_id,
            stored
            or _undecided(
                sn_id, (), "no stored Jev decision; the name keeps its current domain"
            ),
        )
        domain_batch.append({"sn_id": sn_id})
    candidate_name_ids = {n["id"] for n in names if n.get("id")}
    for batch in (co_batch, he_batch, pred_batch):
        for row in batch:
            candidate_name_ids.update((row["from_name"], row["to_name"]))
    candidate_name_ids.update(
        row["from_name"] for row in geo_batch if row.get("from_name")
    )
    candidate_name_ids.update(row["sn_id"] for row in cluster_batch if row.get("sn_id"))
    candidate_name_ids.update(row["sn_id"] for row in domain_batch if row.get("sn_id"))
    return candidate_name_ids


_structural_edge_reconcile_apply = functools.partial(
    apply_signed_manifest,
    {},
    authority_adapter="structural-edge-reconcile",
    mutation_kind="recompute-canonical-structural-edges",
    guard_set=(
        "exact-request-scope",
        "canonical-structural-derivation",
        "out-of-allowlist-immutability",
    ),
    reason="exact structural-edge reconciliation for requested Standard Names",
    apply=True,
)


def _clear_superseded_child_derived_parent_edges(gc: Any, *, services: Any) -> int:
    """Delete derived HAS_PARENT edges FROM superseded/exhausted children.

    A dead (superseded/exhausted) child must not keep propping up a zombie
    parent, so its derived edges are ordinarily reaped. But the edge is also
    provenance for the identity it points INTO: deleting an incoming edge
    strips that identity of a relationship it carries. An edge whose target
    holds durable authority — recorded spend, ratification, a catalog binding
    — therefore survives even though the child that carries it is dead. The
    write path asks the same question of its candidates via
    :func:`automatic_deletion_protections`; this cleanup asks it of the edge's
    targets before deleting. Returns the number of edges actually deleted.
    """
    rows = gc.query(
        "\n        MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName)\n        WHERE coalesce(c.name_stage, '') IN ['superseded', 'exhausted']\n          AND r.operator_kind IS NOT NULL\n        RETURN c.id AS child, p.id AS parent\n        "
    )
    pairs = [
        (str(row["child"]), str(row["parent"]))
        for row in rows or []
        if row.get("child") and row.get("parent")
    ]
    if not pairs:
        return 0
    protected_parents = set(
        services.automatic_deletion_protections(
            gc, sorted({parent for _c, parent in pairs})
        )
    )
    to_delete = [
        {"child": child, "parent": parent}
        for child, parent in pairs
        if parent not in protected_parents
    ]
    if not to_delete:
        return 0
    gc.query(
        "\n        UNWIND $pairs AS pair\n        MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName)\n        WHERE c.id = pair.child AND p.id = pair.parent\n          AND coalesce(c.name_stage, '') IN ['superseded', 'exhausted']\n          AND r.operator_kind IS NOT NULL\n        DELETE r\n        ",
        pairs=to_delete,
    )
    return len(to_delete)


def rederive_structural_edges(*, services: Any) -> dict[str, int]:
    """Reconcile HAS_PARENT/HAS_ERROR edges to the current derivation of LIVE names.

    Structure is defined by **live** names only. Superseded/exhausted names are
    dead (refined away) and must not contribute structural parents: deriving
    edges from a dead name re-mints a parent that the childless reaper then
    deletes — an unbounded mint/reap churn across startups. So this:

    1. derives + reconciles edges for non-superseded/-exhausted names only
       (``_write_standard_name_edges`` deletes each child's stale derived edges
       and writes the current admitted set — idempotent);
    2. deletes any derived (``operator_kind``-bearing) HAS_PARENT edge still
       originating FROM a superseded/exhausted name, so dead names stop
       propping up zombie parents (those parents become reapable, then drain)
       — unless the edge's target carries durable authority, in which case the
       edge is kept as provenance (``_clear_superseded_child_derived_parent_edges``);
    3. migrates inbound HAS_PARENT edges off superseded PARENTS to their live
       successor (``_rewire_has_parent_off_superseded``, which skips
       projection-of-self successors).

    The result is a fixpoint: re-running on a stable live name set mints no new
    parents and orphans none. Returns processed/dead-edge/migrated counts.
    """
    with services.GraphClient() as gc:
        live = gc.query(
            "MATCH (sn:StandardName) WHERE sn.name_stage IS NOT NULL AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested']) RETURN sn.id AS id"
        )
        names = [{"id": r["id"]} for r in live if r.get("id")]
        if not names:
            return {"processed": 0, "dead_edges_cleared": 0, "migrated": 0}
        services._write_standard_name_edges(gc, names, full_rebuild=True)
        dead_cleared = services._clear_superseded_child_derived_parent_edges(gc)
        migrated = services._rewire_has_parent_off_superseded(gc)
    logger.info(
        "rederive_structural_edges: processed %d live names, cleared %d dead-name edges, migrated %d HAS_PARENT edges",
        len(names),
        dead_cleared,
        migrated,
    )
    return {
        "processed": len(names),
        "dead_edges_cleared": dead_cleared,
        "migrated": migrated,
    }


def _rewire_has_parent_off_superseded(gc: Any, *, services: Any) -> int:
    """Rewire HAS_PARENT edges from superseded parents to their successors.

    A name that supersedes another carries a ``REFINED_FROM`` edge back
    to its predecessor. We walk the chain to find the live tip (the SN
    at the head whose ``name_stage`` is not ``superseded``) and migrate
    each inbound ``HAS_PARENT`` edge to that tip, preserving edge
    properties. The child's current ``derive_edges`` result must authorize
    both the tip identity and the complete edge-property map; refinement
    lineage cannot change grammar topology by itself. If no authorized live
    tip exists, the edge is left alone so manual review can investigate.

    Self-loop guard: if a refine step renamed the *child* into the same
    name as its (now-superseded) parent — e.g. ``minimum_magnetic_field``
    got refined into ``minimum_magnetic_field_magnitude`` while
    ``minimum_magnetic_field_magnitude`` was already its HAS_PARENT
    parent — naive migration produces ``child -[HAS_PARENT]-> child``,
    a structural impossibility that crashes ISNC's
    ``validate_catalog`` topological sort with
    ``CycleError: nodes are in a cycle, ['x', 'x']``. In that case the
    edge is deleted outright; the new structural parent (if any) will
    be re-emitted next time ``derive_edges`` runs.

    Returns the number of HAS_PARENT edges actually migrated (delete-
    only self-loops are reported under ``deleted_self_loops`` in the
    logger but excluded from the migrated count).
    """
    from imas_codex.standard_names.derivation import derive_edges

    child_rows = list(
        gc.query(
            "\n            MATCH (child:StandardName)-[:HAS_PARENT]->(old:StandardName)\n            WHERE old.name_stage = 'superseded'\n            RETURN DISTINCT child.id AS child_id\n            "
        )
    )
    authorized_parent_edges: list[dict[str, Any]] = []
    for row in child_rows:
        child_id = row.get("child_id")
        if not child_id:
            continue
        for edge in derive_edges(str(child_id)):
            if edge.edge_type != "HAS_PARENT":
                continue
            authorized_parent_edges.append(
                {
                    "child_id": edge.from_name,
                    "parent_id": edge.to_name,
                    "edge_properties": {
                        key: value
                        for key, value in edge.props.items()
                        if value is not None
                    },
                }
            )
    if not authorized_parent_edges:
        return 0
    result = list(
        gc.query(
            "\n            MATCH (child:StandardName)-[c:HAS_PARENT]->(old:StandardName)\n            WHERE old.name_stage = 'superseded'\n            MATCH path = (tip:StandardName)-[:REFINED_FROM*1..]->(old)\n            WHERE tip.name_stage <> 'superseded'\n              // Migrate ONLY to a true rename of `old` — a successor with the\n              // SAME identity segments (physical/geometric base, subject,\n              // component). A REFINED_FROM \"successor\" that ADDS a qualifier/\n              // projection or CHANGES the subject (e.g. ion_convection_velocity\n              // -> radial_electron_convection_velocity, or major_radius ->\n              // major_radius_of_flux_surface) is a more-specific CHILD, not a\n              // rename; migrating `old`'s children onto it recreates the\n              // domain/locus-mismatched tangle this function exists to repair.\n              // The correct remedy for those is to un-supersede `old` (it is the\n              // legitimate generic parent), so leave the edge in place here.\n              // Conservative on NULL columns: if identity cannot be confirmed\n              // equal, do not migrate.\n              AND coalesce(tip.physical_base, '')  = coalesce(old.physical_base, '')\n              AND coalesce(tip.geometric_base, '') = coalesce(old.geometric_base, '')\n              AND coalesce(tip.subject, '')        = coalesce(old.subject, '')\n              AND coalesce(tip.component, '')       = coalesce(old.component, '')\n              AND any(authorized IN $authorized_parent_edges\n                      WHERE authorized.child_id = child.id\n                        AND authorized.parent_id = tip.id\n                        AND authorized.edge_properties = properties(c))\n            WITH old, child, c, properties(c) AS props, tip,\n                 length(path) AS hops\n            ORDER BY hops DESC\n            WITH old, child, c, props, head(collect(tip)) AS tip\n            WHERE tip IS NOT NULL\n            DELETE c\n            // Drop the edge if migration would produce a self-loop;\n            // any legitimate structural parent for `child` will be\n            // re-emitted by the next derive_edges pass.\n            WITH tip, child, props\n            WHERE tip.id <> child.id\n              // Re-assert current derivation authority at the mutation\n              // boundary. Refinement lineage alone never authorizes moving a\n              // grammar edge onto a target the child does not derive.\n              AND any(authorized IN $authorized_parent_edges\n                      WHERE authorized.child_id = child.id\n                        AND authorized.parent_id = tip.id\n                        AND authorized.edge_properties = props)\n            MERGE (child)-[c_new:HAS_PARENT]->(tip)\n            SET c_new = props\n            RETURN count(c_new) AS migrated\n            ",
            authorized_parent_edges=authorized_parent_edges,
        )
    )
    if not result:
        return 0
    return int(result[0].get("migrated") or 0)


def _parse_parent_grammar(name_id: str, *, services: Any) -> dict[str, str | None]:
    """Attempt ISN parse on a parent name to extract grammar fields.

    Returns a dict with compatibility segment keys. On strict parse failure,
    all values are None. Uses the same lossless IR authority as
    ``_parse_grammar`` and ``_write_grammar_decomposition``.
    """
    try:
        from imas_standard_names import compose, parse

        result = parse(name_id, strict=True)
        if compose(result.ir) != name_id:
            return dict.fromkeys(services._GRAMMAR_SEGMENT_COLUMNS)
        return services._segments_from_ir(result.ir)
    except ImportError:
        logger.debug("ISN unavailable — no grammar for parent %s", name_id)
    except Exception:
        logger.debug("ISN strict grammar rejected parent %s", name_id)
    return dict.fromkeys(services._GRAMMAR_SEGMENT_COLUMNS)


def _query_seedable_derived_parents(
    gc: Any, *, state_where: str, services: Any
) -> list[dict[str, Any]]:
    """Return structurally seedable derived-parent candidates."""
    return list(
        gc.query(
            f"\n            MATCH (parent:StandardName)\n            WHERE (parent.origin IS NULL OR parent.origin = 'derived')\n              AND ({state_where})\n            MATCH (child)-[comp:HAS_PARENT]->(parent)\n            WHERE comp.operator_kind IN\n                  ['projection', 'coordinate', 'unary_postfix']\n            WITH parent,\n                 collect(DISTINCT child) AS children\n            MATCH (any_child)-[any_comp:HAS_PARENT]->(parent)\n            WITH parent, children,\n                 collect(any_comp.operator_kind) AS edge_kinds,\n                 count(any_child) AS total_children,\n                 count(CASE WHEN any_child.name_stage IS NOT NULL\n                       THEN 1 END) AS composed_children\n            WHERE total_children = composed_children\n              AND total_children >= 1\n            UNWIND children AS child\n            OPTIONAL MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(child)\n            OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(imas:IMASNode)\n            OPTIONAL MATCH (child)-[cedge:HAS_PARENT]->(parent)\n            WITH parent, edge_kinds,\n                 collect(DISTINCT {{\n                     id: child.id,\n                     unit: child.unit,\n                     cocos: child.cocos_transformation_type,\n                     physics_domain: child.physics_domain,\n                     kind: child.kind,\n                     op_kind: cedge.operator_kind\n                 }}) AS child_data,\n                 collect(DISTINCT imas.id) AS dd_paths\n            RETURN parent.id AS parent_id,\n                   parent.origin AS origin,\n                   parent.name_stage AS name_stage,\n                   child_data,\n                   dd_paths,\n                   edge_kinds\n            "
        )
    )


def _query_legacy_repairable_derived_parents(
    gc: Any, *, state_where: str, services: Any
) -> list[dict[str, Any]]:
    """Return already-admitted legacy derived parents eligible for lifecycle repair."""
    return list(
        gc.query(
            f"\n            MATCH (parent:StandardName)\n            WHERE (parent.origin IS NULL OR parent.origin = 'derived')\n              AND ({state_where})\n            MATCH (child)-[comp:HAS_PARENT]->(parent)\n            WITH parent,\n                 collect(DISTINCT child) AS children,\n                 collect(comp.operator_kind) AS edge_kinds,\n                 count(child) AS total_children,\n                 count(CASE WHEN child.name_stage IS NOT NULL\n                       THEN 1 END) AS composed_children,\n                 count(CASE WHEN comp.operator_kind IN\n                       ['projection', 'coordinate', 'unary_postfix']\n                       THEN 1 END) AS seedable_edges\n            WHERE total_children = composed_children\n              AND total_children >= 1\n              AND seedable_edges = 0\n            UNWIND children AS child\n            OPTIONAL MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(child)\n            OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(imas:IMASNode)\n            OPTIONAL MATCH (child)-[cedge:HAS_PARENT]->(parent)\n            WITH parent, edge_kinds,\n                 collect(DISTINCT {{\n                     id: child.id,\n                     unit: child.unit,\n                     cocos: child.cocos_transformation_type,\n                     physics_domain: child.physics_domain,\n                     kind: child.kind,\n                     op_kind: cedge.operator_kind\n                 }}) AS child_data,\n                 collect(DISTINCT imas.id) AS dd_paths\n            RETURN parent.id AS parent_id,\n                   parent.origin AS origin,\n                   parent.name_stage AS name_stage,\n                   child_data,\n                   dd_paths,\n                   edge_kinds\n            "
        )
    )


def _query_derived_parents_for_admission_cleanup(
    gc: Any, *, services: Any
) -> list[str]:
    """Return materialized derived-parent ids to re-check for admission.

    Accepted nodes can predate the current admission gate. Pending nodes can be
    born before their first batch child has a graph edge, when the shadow veto
    has no topology to inspect. Both states must pass the current gate before
    lifecycle materialization or structural acceptance can promote them.
    Restrict to ``origin='derived'`` and exclude identities carrying durable
    authority before admission is evaluated.  The deletion path repeats the
    authority check as a backstop against changes between selection and write.

    Re-check regardless of ``docs_stage``. A single-child shadow can otherwise
    survive either pending materialization or an accepted names-only run.
    """
    rows = gc.query(
        "\n        MATCH (parent:StandardName)\n        WHERE parent.origin = 'derived'\n          AND parent.name_stage IN ['pending', 'accepted']\n        MATCH (:StandardName)-[:HAS_PARENT]->(parent)\n        RETURN DISTINCT parent.id AS parent_id\n        "
    )
    candidate_ids = [str(r["parent_id"]) for r in rows if r.get("parent_id")]
    return services.filter_automatic_deletion_candidates(gc, candidate_ids)


def _delete_derived_parent_nodes(
    gc: Any,
    parent_ids: list[str],
    *,
    manifest_sha256: str | None = None,
    reason: str | None = None,
    expected_producer_ids_by_parent: dict[str, list[str]] | None = None,
    services: Any,
) -> int:
    """Delete only unprotected structural placeholders with recovery material.

    Admission determines whether scaffolding is warranted. It is not a licence
    to delete a physics identity, so this path requires the explicit
    ``needs_composition`` placeholder marker and refuses a complete batch when
    it contains publication evidence or an unusual candidate volume.
    """
    parent_ids = sorted(set(parent_ids))
    if not parent_ids:
        return 0
    if len(parent_ids) > foundation.DERIVED_PARENT_CLEANUP_DELETION_LIMIT:
        raise services.DerivedParentCleanupRefusal(
            f"Refused structural derived-parent cleanup: {len(parent_ids)} candidate identities exceeds the {foundation.DERIVED_PARENT_CLEANUP_DELETION_LIMIT}-identity ceiling"
        )
    try:
        services.refuse_protected_automatic_deletion(
            gc, parent_ids, operation="structural derived-parent cleanup"
        )
    except services.ProtectedDeletionError as exc:
        raise services.DerivedParentCleanupRefusal(str(exc)) from exc
    deleted = 0
    deletion_clause = services.deletion_change_cypher("sn")
    deletion_params = services.deletion_change_params(
        "remove_derived_parent",
        reason=reason
        or "structural derived parent no longer satisfies lifecycle admission",
    )
    for parent_id in parent_ids:
        expected_producer_ids = (
            sorted(set(expected_producer_ids_by_parent[parent_id]))
            if expected_producer_ids_by_parent is not None
            and parent_id in expected_producer_ids_by_parent
            else None
        )
        rows = list(
            gc.query(
                f"\n                MATCH (sn:StandardName {{id: $parent_id}})\n                WHERE sn.needs_composition = true\n                CALL (sn) {{\n                  OPTIONAL MATCH (producer:StandardNameSource)\n                  WHERE producer.produced_sn_id = sn.id\n                     OR EXISTS {{ (producer)-[:PRODUCED_NAME]->(sn) }}\n                     OR (producer.source_type = 'derived'\n                         AND producer.source_id = sn.id)\n                  RETURN collect(DISTINCT producer.id) AS current_producer_ids\n                }}\n                OPTIONAL MATCH (derived_source:StandardNameSource\n                  {{source_type: 'derived', source_id: $parent_id}})\n                OPTIONAL MATCH (mirror_source:StandardNameSource)\n                WHERE mirror_source.produced_sn_id = $parent_id\n                OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(rv:StandardNameReview)\n                OPTIONAL MATCH (sn)-[:DOCS_REVISION_OF]->(dr:DocsRevision)\n                WITH sn,\n                     current_producer_ids,\n                     collect(DISTINCT derived_source) AS derived_sources,\n                     collect(DISTINCT mirror_source) AS mirror_sources,\n                     collect(DISTINCT rv) AS reviews,\n                     collect(DISTINCT dr) AS revisions\n                WHERE $expected_producer_ids IS NULL\n                   OR (size(current_producer_ids) = size($expected_producer_ids)\n                       AND all(id IN current_producer_ids\n                               WHERE id IN $expected_producer_ids)\n                       AND all(id IN $expected_producer_ids\n                               WHERE id IN current_producer_ids))\n                {deletion_clause}\n                SET change.manifest_sha256 = $stub_manifest_sha256\n                FOREACH (source IN mirror_sources |\n                  SET source.produced_sn_id = null)\n                FOREACH (source IN derived_sources | DETACH DELETE source)\n                FOREACH (r IN reviews | DETACH DELETE r)\n                FOREACH (d IN revisions | DETACH DELETE d)\n                DETACH DELETE sn\n                RETURN 1 AS deleted\n                ",
                parent_id=parent_id,
                expected_producer_ids=expected_producer_ids,
                stub_manifest_sha256=manifest_sha256,
                **deletion_params,
            )
        )
        if rows:
            deleted += int(rows[0].get("deleted", 0))
    return deleted


def sweep_orphaned_docs_revisions(gc: Any, *, services: Any) -> int:
    """Delete DocsRevision snapshots no longer linked to any StandardName.

    A ``DocsRevision`` is owned by exactly one ``StandardName`` via
    ``(sn)-[:DOCS_REVISION_OF]->(dr)``. When the owning name is reaped its
    revisions become unreachable scaffolding. Although ``_delete_derived_parent_nodes``
    now deletes a parent's revisions inline, this sweep is the idempotent
    backstop: it clears any revision orphaned by an earlier reap (or a future
    deletion path that forgets to). Runs every startup so orphaned docs content
    can never accumulate. Returns the number deleted.
    """
    rows = list(
        gc.query(
            "\n            MATCH (dr:DocsRevision)\n            WHERE NOT EXISTS { (:StandardName)-[:DOCS_REVISION_OF]->(dr) }\n            WITH dr LIMIT 50000\n            DETACH DELETE dr\n            RETURN count(dr) AS n\n            "
        )
    )
    n = int(rows[0]["n"]) if rows else 0
    if n:
        logger.info("sweep_orphaned_docs_revisions: deleted %d orphaned revisions", n)
    return n


def _derived_parent_source_metadata(parent_id: str, *, services: Any) -> dict[str, str]:
    """Build the distinct structural provenance identity for a derived parent.

    A common child DD path is evidence about the children, not a realization of
    the parent. Structural sources therefore neither reuse a ``dd:<path>``
    identity nor carry a ``FROM_DD_PATH`` relationship.
    """
    return {
        "source_node_id": f"derived:{parent_id}",
        "source_type": "derived",
        "source_id": parent_id,
        "batch_key": "derived_parent",
    }


def _validate_derived_parent_identity(
    parent_id: str, *, kind: str, unit: str, services: Any
) -> list[str]:
    """Validate a parent name and unit through ISN's public authority surface."""
    try:
        from imas_standard_names.grammar import compose, parse
        from imas_standard_names.models import create_standard_name_entry
        from imas_standard_names.validation.semantic import run_semantic_checks

        from imas_codex.standard_names.kind_derivation import to_isn_kind

        parsed = parse(parent_id, strict=True)
        if compose(parsed.ir) != parent_id:
            return ["strict grammar round-trip changed the parent name"]
        entry = create_standard_name_entry(
            {"name": parent_id, "kind": to_isn_kind(kind), "unit": unit}, name_only=True
        )
        return [
            issue
            for issue in run_semantic_checks({parent_id: entry})
            if " ERROR - " in issue
        ]
    except Exception as exc:
        return [str(exc)]


def _eligible_derived_parent_unit_children(
    parent_id: str, child_data: list[dict[str, Any]], *, services: Any
) -> list[dict[str, Any]]:
    """Select children whose units authoritatively constrain a parent."""
    eligible = [child for child in child_data if child.get("op_kind") != "binary"]
    normalization_markers = ("normalized", "normalised")
    if not any(marker in parent_id.split("_") for marker in normalization_markers):
        eligible = [
            child
            for child in eligible
            if not any(
                marker in str(child.get("id") or "").split("_")
                for marker in normalization_markers
            )
        ]
    return eligible


def _materialize_derived_parent_rows(
    gc: Any,
    parents: list[dict[str, Any]],
    *,
    infer_kind_from_existing_topology: bool = False,
    rejected_pending_parent_ids: list[str] | None = None,
    bootstrap_missing: bool = False,
    services: Any,
) -> int:
    """Materialize structurally eligible derived parents onto the docs lifecycle."""
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )
    from imas_codex.standard_names.parents import recompute_parent_kind

    seeded = 0
    for row in parents:
        parent_id = row["parent_id"]
        child_data = row.get("child_data") or []
        edge_kinds = row.get("edge_kinds") or []
        authorized_unit = row.get("authorized_unit")
        if authorized_unit is None:
            eligible_unit_children = services._eligible_derived_parent_unit_children(
                parent_id, child_data
            )
            eligible_units = {
                str(child["unit"])
                for child in eligible_unit_children
                if child.get("unit")
            }
            if len(eligible_units) == 1:
                authorized_unit = next(iter(eligible_units))
        is_geometric = any(k == "coordinate" for k in edge_kinds)
        if infer_kind_from_existing_topology:
            kind = recompute_parent_kind(parent_id, gc)
        else:
            seedable_edge_kinds = {"projection", "coordinate", "unary_postfix"}
            if authorized_unit is None and (
                not any(k in seedable_edge_kinds for k in edge_kinds)
            ):
                logger.debug(
                    "Parent %s has only non-seedable edges (%s) — skipping",
                    parent_id,
                    edge_kinds,
                )
                continue
            is_magnitude_parent = all(k == "unary_postfix" for k in edge_kinds)
            kind = "scalar" if is_magnitude_parent else "vector"
        unit_children = services._eligible_derived_parent_unit_children(
            parent_id, child_data
        )
        child_units = {c["unit"] for c in unit_children if c.get("unit")}
        if authorized_unit is not None:
            unit = str(authorized_unit)
        elif is_geometric:
            unit = None
        elif len(child_units) > 1:
            logger.warning(
                "Parent %s has children with heterogeneous units: %s — skipping (needs manual review)",
                parent_id,
                child_units,
            )
            continue
        else:
            unit = next(iter(child_units)) if child_units else None
        child_cocos = {c["cocos"] for c in unit_children if c.get("cocos")}
        cocos = next(iter(child_cocos)) if len(child_cocos) == 1 else None
        child_domain_list = [
            c["physics_domain"] for c in child_data if c.get("physics_domain")
        ]
        if child_domain_list:
            _domain_counts = Counter(child_domain_list)
            _top = max(_domain_counts.values())
            physics_domain = sorted(
                (d for d, n in _domain_counts.items() if n == _top)
            )[0]
            source_domains = sorted(set(child_domain_list))
        else:
            physics_domain = None
            source_domains = []
        grammar_fields = services._parse_parent_grammar(parent_id)
        props: dict[str, Any] = {
            "parent_id": parent_id,
            "unit": unit,
            "cocos_transformation_type": cocos,
            "physics_domain": physics_domain,
            "source_domains": source_domains,
            "kind": kind,
            "description": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
            "is_geometric_coordinate": is_geometric,
        }
        props.update(grammar_fields)
        props.update(services._derived_parent_source_metadata(parent_id))
        props["bootstrap_edges"] = row.get("bootstrap_edges") or []
        if not unit:
            logger.warning(
                "Parent %s has no authoritative unit — skipping materialization",
                parent_id,
            )
            continue
        validation_issues = services._validate_derived_parent_identity(
            parent_id, kind=kind, unit=unit
        )
        if validation_issues:
            logger.warning(
                "Parent %s failed the name-validity oracle — skipping: %s",
                parent_id,
                "; ".join(validation_issues),
            )
            if (
                rejected_pending_parent_ids is not None
                and row.get("origin") == "derived"
                and (row.get("name_stage") == "pending")
            ):
                rejected_pending_parent_ids.append(parent_id)
            continue
        parent_match = (
            "UNWIND $bootstrap_edges AS authorized_edge MATCH (authorized_child:StandardName {id: authorized_edge.from_name}) WHERE ((authorized_child.name_stage IS NULL AND authorized_edge.expected_name_stage IS NULL) OR authorized_child.name_stage = authorized_edge.expected_name_stage) AND ((authorized_child.unit IS NULL AND authorized_edge.expected_unit IS NULL) OR authorized_child.unit = authorized_edge.expected_unit) AND ((authorized_child.kind IS NULL AND authorized_edge.expected_kind IS NULL) OR authorized_child.kind = authorized_edge.expected_kind) AND ((authorized_child.physics_domain IS NULL AND authorized_edge.expected_physics_domain IS NULL) OR authorized_child.physics_domain = authorized_edge.expected_physics_domain) AND ((authorized_child.cocos_transformation_type IS NULL AND authorized_edge.expected_cocos IS NULL) OR authorized_child.cocos_transformation_type = authorized_edge.expected_cocos) WITH collect({child: authorized_child, edge: authorized_edge}) AS authorized_children WHERE size(authorized_children) = size($bootstrap_edges) MERGE (parent:StandardName {id: $parent_id}) ON CREATE SET parent.updated_at = datetime(), parent.created_at = datetime()"
            if bootstrap_missing
            else "MATCH (parent:StandardName {id: $parent_id}) WHERE EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(parent) }"
        )
        parent_scope = (
            "WITH parent, authorized_children" if bootstrap_missing else "WITH parent"
        )
        materialize_cypher = "\n            __PARENT_MATCH__\n            SET parent.name_stage = CASE\n                    // Already name-reviewed → stays accepted (idempotent).\n                    WHEN parent.reviewer_score_name IS NOT NULL THEN 'accepted'\n                    // Unreviewed but review-ready (real description + embedding)\n                    // → route through REVIEW_NAME before it can earn docs.\n                    WHEN trim(coalesce(parent.description, '')) <> ''\n                         AND parent.description <> $description\n                         AND parent.embedding IS NOT NULL THEN 'drafted'\n                    // Unreviewed with only the placeholder description → hold at\n                    // accepted; the docs gate (reviewer_score_name IS NOT NULL)\n                    // keeps it out of docs until enrichment + review fill it in.\n                    ELSE 'accepted'\n                END,\n                parent.docs_stage = coalesce(parent.docs_stage, 'pending'),\n                parent.validation_status = coalesce(parent.validation_status, 'valid'),\n                parent.origin = 'derived',\n                parent.kind = $kind,\n                parent.unit = coalesce($unit, parent.unit),\n                parent.cocos_transformation_type = $cocos_transformation_type,\n                parent.cocos = CASE\n                    WHEN $cocos_transformation_type IS NULL THEN null\n                    ELSE parent.cocos\n                END,\n                parent.physics_domain =\n                    coalesce($physics_domain, parent.physics_domain),\n                parent.description = CASE\n                    WHEN trim(coalesce(parent.description, '')) = ''\n                    THEN $description\n                    ELSE parent.description\n                END,\n                parent.needs_composition = null,\n                parent.chain_length = coalesce(parent.chain_length, 0),\n                parent.docs_chain_length = coalesce(parent.docs_chain_length, 0),\n                parent.physical_base =\n                    coalesce($physical_base, parent.physical_base),\n                parent.geometric_base =\n                    coalesce($geometric_base, parent.geometric_base),\n                parent.subject =\n                    coalesce($subject, parent.subject),\n                parent.transformation =\n                    coalesce($transformation, parent.transformation),\n                parent.component =\n                    coalesce($component, parent.component),\n                parent.coordinate =\n                    coalesce($coordinate, parent.coordinate),\n                parent.position =\n                    coalesce($position, parent.position),\n                parent.process =\n                    coalesce($process, parent.process),\n                parent.device =\n                    coalesce($device, parent.device),\n                parent.region =\n                    coalesce($region, parent.region),\n                parent.aggregation =\n                    coalesce($aggregation, parent.aggregation),\n                parent.orbit =\n                    coalesce($orbit, parent.orbit),\n                parent.population =\n                    coalesce($population, parent.population),\n                parent.object =\n                    coalesce($object, parent.object),\n                parent.geometry =\n                    coalesce($geometry, parent.geometry),\n                parent.is_geometric_coordinate =\n                    coalesce(parent.is_geometric_coordinate,\n                             $is_geometric_coordinate),\n                parent.status = coalesce(parent.status, 'draft')\n            __PARENT_SCOPE__\n            CALL (parent) {\n                OPTIONAL MATCH (parent)-[stored_cocos:HAS_COCOS]->(:COCOS)\n                WITH collect(stored_cocos) AS stored_cocos_edges\n                FOREACH (stored_cocos IN CASE\n                    WHEN $cocos_transformation_type IS NULL\n                    THEN stored_cocos_edges ELSE []\n                END | DELETE stored_cocos)\n                RETURN size(stored_cocos_edges) AS stored_cocos_edge_count\n            }\n            __PARENT_SCOPE__\n            MERGE (sns:StandardNameSource {id: $source_node_id})\n            ON CREATE SET sns.created_at = datetime(),\n                          sns.attempt_count = 0\n            SET sns.status = 'composed',\n                sns.source_type = $source_type,\n                sns.source_id = $source_id,\n                sns.batch_key = coalesce(sns.batch_key, $batch_key),\n                sns.description = CASE\n                    WHEN trim(coalesce(sns.description, '')) = ''\n                    THEN parent.description\n                    ELSE sns.description\n                END,\n                sns.physics_domain = coalesce(parent.physics_domain, sns.physics_domain),\n                sns.composed_at = coalesce(sns.composed_at, datetime()),\n                sns.produced_sn_id = parent.id,\n                sns.claimed_at = null,\n                sns.claim_token = null\n            MERGE (sns)-[:PRODUCED_NAME]->(parent)\n            ".replace(
            "__PARENT_MATCH__", parent_match
        ).replace("__PARENT_SCOPE__", parent_scope)
        if bootstrap_missing:
            materialize_cypher += "\n            WITH parent, authorized_children\n            OPTIONAL MATCH (parent)-[old_unit:HAS_UNIT]->(:Unit)\n            DELETE old_unit\n            WITH parent, authorized_children\n            MERGE (unit:Unit {id: $unit})\n            MERGE (parent)-[:HAS_UNIT]->(unit)\n            WITH parent, authorized_children\n            UNWIND authorized_children AS binding\n            WITH parent, binding.child AS child, binding.edge AS edge\n            MERGE (child)-[relation:HAS_PARENT]->(parent)\n            SET relation.operator = edge.operator,\n                relation.operator_kind = edge.operator_kind,\n                relation.role = edge.role,\n                relation.separator = edge.separator,\n                relation.axis = edge.axis,\n                relation.shape = edge.shape\n            RETURN parent.id AS parent_id\n            "
        result = list(gc.query(materialize_cypher, **props))
        if bootstrap_missing and (
            not any(returned.get("parent_id") == parent_id for returned in result)
        ):
            logger.warning(
                "Parent %s bootstrap authority changed before persistence — refusing",
                parent_id,
            )
            continue
        if unit and (not bootstrap_missing):
            gc.query(
                "\n                MATCH (sn:StandardName {id: $parent_id})\n                WHERE EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(sn) }\n                OPTIONAL MATCH (sn)-[r:HAS_UNIT]->(:Unit)\n                DELETE r\n                WITH sn\n                MERGE (u:Unit {id: $unit})\n                MERGE (sn)-[:HAS_UNIT]->(u)\n                ",
                parent_id=parent_id,
                unit=unit,
            )
        stored = _stored_domain_decisions(gc, [parent_id])
        if parent_id in stored:
            write_name_domain(
                gc,
                parent_id,
                stored[parent_id]
                or _undecided(
                    parent_id,
                    source_domains,
                    "a derived parent has no stored Jev decision; it keeps the "
                    "domain its children give it",
                ),
            )
        seeded += 1
    return seeded


class _ParentMaterializationCapture:
    """Capture sanctioned materializer parameters without graph access."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append({"cypher": cypher, "params": params})
        return []


def _materialize_derived_parent_rows_batched(
    gc: Any,
    parents: list[dict[str, Any]],
    *,
    event_by_parent: dict[str, dict[str, Any]] | None = None,
    services: Any,
) -> int:
    """Run the sanctioned parent materializer in two cohort-wide queries."""
    capture = services._ParentMaterializationCapture()
    prepared_count = services._materialize_derived_parent_rows(capture, parents)
    if prepared_count != len(parents):
        raise RuntimeError("authorized derived-parent materialization row was dropped")
    prepared_rows = [
        dict(call["params"])
        for call in capture.calls
        if "source_node_id" in call["params"]
    ]
    if len(prepared_rows) != prepared_count:
        raise RuntimeError("derived-parent materialization preparation is incomplete")
    rows = []
    for properties in prepared_rows:
        parent_id = str(properties["parent_id"])
        properties["event"] = (event_by_parent or {}).get(parent_id)
        rows.append(properties)
    materialized = gc.query(
        "\n        // STRUCTURAL_CLOSURE_BATCH_MATERIALIZE\n        UNWIND $rows AS row\n        MATCH (parent:StandardName {id: row.parent_id})\n        WHERE EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(parent) }\n        SET parent.updated_at = datetime(), parent.name_stage = CASE\n                WHEN parent.reviewer_score_name IS NOT NULL THEN 'accepted'\n                WHEN trim(coalesce(parent.description, '')) <> ''\n                     AND parent.description <> row.description\n                     AND parent.embedding IS NOT NULL THEN 'drafted'\n                ELSE 'accepted'\n            END,\n            parent.docs_stage = coalesce(parent.docs_stage, 'pending'),\n            parent.validation_status = coalesce(parent.validation_status, 'valid'),\n            parent.origin = 'derived',\n            parent.kind = row.kind,\n            parent.unit = coalesce(row.unit, parent.unit),\n            parent.cocos_transformation_type = row.cocos_transformation_type,\n            parent.cocos = CASE\n                WHEN row.cocos_transformation_type IS NULL THEN null\n                ELSE parent.cocos\n            END,\n            parent.physics_domain =\n                    coalesce(row.physics_domain, parent.physics_domain),\n            parent.description = CASE\n                WHEN trim(coalesce(parent.description, '')) = ''\n                THEN row.description ELSE parent.description\n            END,\n            parent.needs_composition = null,\n            parent.chain_length = coalesce(parent.chain_length, 0),\n            parent.docs_chain_length = coalesce(parent.docs_chain_length, 0),\n            parent.physical_base = coalesce(row.physical_base, parent.physical_base),\n            parent.geometric_base = coalesce(row.geometric_base, parent.geometric_base),\n            parent.subject = coalesce(row.subject, parent.subject),\n            parent.transformation = coalesce(row.transformation, parent.transformation),\n            parent.component = coalesce(row.component, parent.component),\n            parent.coordinate = coalesce(row.coordinate, parent.coordinate),\n            parent.position = coalesce(row.position, parent.position),\n            parent.process = coalesce(row.process, parent.process),\n            parent.device = coalesce(row.device, parent.device),\n            parent.region = coalesce(row.region, parent.region),\n            parent.aggregation = coalesce(row.aggregation, parent.aggregation),\n            parent.orbit = coalesce(row.orbit, parent.orbit),\n            parent.population = coalesce(row.population, parent.population),\n            parent.object = coalesce(row.object, parent.object),\n            parent.geometry = coalesce(row.geometry, parent.geometry),\n            parent.is_geometric_coordinate =\n                coalesce(parent.is_geometric_coordinate,\n                         row.is_geometric_coordinate),\n            parent.status = coalesce(parent.status, 'draft')\n        WITH parent, row\n        CALL (parent, row) {\n            OPTIONAL MATCH (parent)-[stored_cocos:HAS_COCOS]->(:COCOS)\n            WITH collect(stored_cocos) AS stored_cocos_edges\n            FOREACH (stored_cocos IN CASE\n                WHEN row.cocos_transformation_type IS NULL\n                THEN stored_cocos_edges ELSE []\n            END | DELETE stored_cocos)\n            RETURN size(stored_cocos_edges) AS stored_cocos_edge_count\n        }\n        MERGE (source:StandardNameSource {id: row.source_node_id})\n          ON CREATE SET source.created_at = datetime(), source.attempt_count = 0\n        SET source.status = 'composed',\n            source.source_type = row.source_type,\n            source.source_id = row.source_id,\n            source.batch_key = coalesce(source.batch_key, row.batch_key),\n            source.description = CASE\n                WHEN trim(coalesce(source.description, '')) = ''\n                THEN parent.description ELSE source.description\n            END,\n            source.physics_domain =\n                coalesce(parent.physics_domain, source.physics_domain),\n            source.composed_at = coalesce(source.composed_at, datetime()),\n            source.produced_sn_id = parent.id,\n            source.claimed_at = null,\n            source.claim_token = null\n        MERGE (source)-[:PRODUCED_NAME]->(parent)\n        FOREACH (_ IN CASE WHEN row.event IS NULL THEN [] ELSE [1] END |\n          CREATE (change:StandardNameChange)\n          SET change = row.event\n          CREATE (parent)-[:HAS_INTERNAL_CHANGE]->(change)\n        )\n        WITH parent, row\n        OPTIONAL MATCH (parent)-[internal_change:HAS_INTERNAL_CHANGE]->\n                       (change:StandardNameChange {id: row.event.id})\n        RETURN collect(DISTINCT parent.id) AS ids,\n               collect(DISTINCT change.id) AS event_ids,\n               collect(DISTINCT CASE\n                   WHEN change IS NULL THEN null\n                   ELSE {event_id: change.id, parent_id: parent.id}\n               END) AS event_links,\n               count(DISTINCT change) AS event_count,\n               count(DISTINCT internal_change) AS event_link_count\n        ",
        rows=rows,
    )
    linked_units = gc.query(
        "\n        // STRUCTURAL_CLOSURE_BATCH_PARENT_UNITS\n        UNWIND $rows AS row\n        MATCH (parent:StandardName {id: row.parent_id})\n        WHERE EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(parent) }\n        OPTIONAL MATCH (parent)-[old_unit:HAS_UNIT]->(:Unit)\n        DELETE old_unit\n        WITH parent, row\n        MERGE (unit:Unit {id: row.unit})\n        MERGE (parent)-[:HAS_UNIT]->(unit)\n        RETURN collect(parent.id) AS ids\n        ",
        rows=rows,
    )
    expected = {str(row["parent_id"]) for row in rows}
    expected_event_links = {
        (str(row["event"]["id"]), str(row["parent_id"]))
        for row in rows
        if row["event"] is not None
    }
    expected_event_ids = {event_id for event_id, _ in expected_event_links}
    materialized_ids = (
        set(materialized[0].get("ids") or []) if len(materialized) == 1 else set()
    )
    materialized_event_ids = (
        set(materialized[0].get("event_ids") or []) if len(materialized) == 1 else set()
    )
    materialized_event_links = (
        {
            (str(link["event_id"]), str(link["parent_id"]))
            for link in materialized[0].get("event_links") or []
        }
        if len(materialized) == 1
        else set()
    )
    event_count = (
        int(materialized[0].get("event_count") or 0) if len(materialized) == 1 else 0
    )
    event_link_count = (
        int(materialized[0].get("event_link_count") or 0)
        if len(materialized) == 1
        else 0
    )
    unit_ids = (
        set(linked_units[0].get("ids") or []) if len(linked_units) == 1 else set()
    )
    if (
        materialized_ids != expected
        or unit_ids != expected
        or materialized_event_ids != expected_event_ids
        or (materialized_event_links != expected_event_links)
        or (event_count != len(expected_event_ids))
        or (event_link_count != len(expected_event_links))
    ):
        raise RuntimeError("batched derived-parent materialization cardinality changed")
    stored_by_parent = _stored_domain_decisions(gc, sorted(expected))
    for row in rows:
        parent_id = str(row["parent_id"])
        if parent_id not in stored_by_parent:
            continue
        write_name_domain(
            gc,
            parent_id,
            stored_by_parent[parent_id]
            or _undecided(
                parent_id,
                row.get("source_domains") or (),
                "a derived parent has no stored Jev decision; it keeps the "
                "domain its children give it",
            ),
        )
    return len(expected)


_normalization_peel_unit_repair_apply = functools.partial(
    apply_signed_manifest,
    {},
    authority_adapter="normalization-peel-unit-repair",
    mutation_kind="clear-normalization-peel-parent-unit",
    guard_set=(
        "corrected-normalization-peel-unit-authority",
        "out-of-allowlist-immutability",
    ),
    reason="clear mis-inherited units from normalization-peel parents",
    apply=True,
)


def stamp_standard_name_units(
    gc: Any,
    unit_by_name: dict[str, str],
    *,
    reason: str,
    origin: str = "unit_authority_repair",
    services: Any,
) -> list[str]:
    """Apply reviewed unit corrections with one durable ledger row per name.

    This is the sanctioned repair instrument for a small, explicitly
    adjudicated set.  It does not infer units: callers supply the reviewed
    mapping, and ISN canonicalizes every value before any graph write.  The
    scalar, ``HAS_UNIT`` edge, and ``StandardNameChange`` record are updated in
    one transaction.  Missing targets are rejected before mutation.
    """
    if not unit_by_name:
        return []
    if not reason.strip():
        raise ValueError("a non-empty unit-repair reason is required")
    from imas_standard_names import canonical_unit

    rows = [
        {"id": name, "unit": canonical_unit(unit)}
        for name, unit in sorted(unit_by_name.items())
    ]
    existing = {
        str(row["id"])
        for row in gc.query(
            "\n            UNWIND $ids AS id\n            MATCH (sn:StandardName {id: id})\n            RETURN sn.id AS id\n            ",
            ids=[row["id"] for row in rows],
        )
    }
    missing = [row["id"] for row in rows if row["id"] not in existing]
    if missing:
        raise ValueError(
            "cannot stamp units on missing StandardName nodes: " + ", ".join(missing)
        )
    changed = list(
        gc.query(
            "\n            UNWIND $rows AS row\n            MATCH (sn:StandardName {id: row.id})\n            OPTIONAL MATCH (sn)-[old_unit:HAS_UNIT]->(:Unit)\n            WITH sn, row, collect(old_unit) AS old_units\n            FOREACH (edge IN old_units | DELETE edge)\n            CREATE (change:StandardNameChange {\n              id: 'sn-change:' + randomUUID(),\n              from_name: sn.id,\n              to_name: sn.id,\n              operation: 'repair_unit_authority',\n              reason: $reason,\n              origin: $origin,\n              changed_at: datetime(),\n              internal: true\n            })\n            MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(change)\n            SET sn.unit = row.unit,\n                sn.updated_at = datetime()\n            MERGE (unit:Unit {id: row.unit})\n            MERGE (sn)-[:HAS_UNIT]->(unit)\n            RETURN sn.id AS id\n            ORDER BY id\n            ",
            rows=rows,
            reason=reason,
            origin=origin,
        )
    )
    return [str(row["id"]) for row in changed]


def seed_parent_sources(gc: Any | None = None, *, services: Any) -> int:
    """Fully populate parent StandardName nodes from their children.

    Selects placeholder parents structurally: any ``StandardName`` node
    with ``name_stage IS NULL`` that is the target of at least one
    HAS_PARENT edge whose ``operator_kind`` is seedable
    (``projection``, ``coordinate``, ``unary_postfix``).

    This deliberately does NOT require the ``needs_composition`` flag.
    That flag is set by ``_write_standard_name_edges`` at edge-write
    time, but the seedable-edge-kinds set has grown over time
    (``unary_postfix`` was added later); placeholders created before
    the set expanded were never flagged and would otherwise stay
    orphaned even though the pipeline now considers them seedable.
    Matching structurally lets the pipeline self-heal on every run.

    Parent names are DETERMINISTIC — the name itself is grammar-derived
    from the components' shared base (e.g. ``magnetic_field`` is the
    only sensible parent of ``poloidal_magnetic_field`` /
    ``toroidal_magnetic_field`` / …). The name has no alternative to
    refine to: REVIEW_NAME produces noise scores on the placeholder
    description, REFINE_NAME has nothing to write. We therefore skip
    the name-axis pipeline entirely for deterministic parents — the
    quality-bearing work happens on the description+docs axis.

    1. Waits until ALL children of a parent have a ``name_stage`` set.
    2. Copies unit and cocos from children (asserts uniformity).
    3. Sets ``name_stage='accepted'`` directly. The children are
       already REVIEW_NAME-validated standard names; their structural
       parent inherits that validity by construction.
    4. Stamps ``origin='deterministic'`` so the semantic-sim gate,
       REFINE_NAME claim, and pool-pending counters all know to leave
       this node alone on the name axis.
    5. Sets ``description = DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER``
       so the export-time guard can refuse to publish until
       ``GENERATE_DOCS`` rewrites it with LLM-quality content.
    6. Sets ``docs_stage='pending'`` so the parent enters
       ``GENERATE_DOCS`` → ``REVIEW_DOCS`` immediately. The
       description+docs go through full RD-quorum review on the docs
       axis — that is the quality gate for these names.
    7. Creates a ``StandardNameSource`` with ``status='composed'`` for audit.
    8. Runs ISN parse to populate grammar fields.
    9. Clears ``needs_composition`` on the parent.

    Returns the number of parents fully populated.
    """
    _own_gc = gc is None
    if _own_gc:
        gc = services.GraphClient().__enter__()
    try:
        parents = services._query_seedable_derived_parents(
            gc, state_where="parent.name_stage IS NULL"
        )
        if not parents:
            return 0
        seeded = services._materialize_derived_parent_rows(gc, parents)
        logger.info("seed_parent_sources: populated %d parent SNs", seeded)
        return seeded
    finally:
        if _own_gc:
            gc.__exit__(None, None, None)


def normalize_derived_parent_lifecycle(gc: Any | None = None, *, services: Any) -> int:
    """Repair eligible derived parents onto the docs lifecycle, idempotently."""
    _own_gc = gc is None
    if _own_gc:
        gc = services.GraphClient().__enter__()
    try:
        from imas_codex.standard_names.parents import is_admissible_parent_name

        state_where = "parent.name_stage IS NULL OR parent.name_stage = 'pending' OR (parent.name_stage = 'accepted' AND parent.docs_stage IS NULL)"
        seedable_parents = services._query_seedable_derived_parents(
            gc, state_where=state_where
        )
        legacy_parents = services._query_legacy_repairable_derived_parents(
            gc, state_where=state_where
        )
        accepted_unit_gap_where = "parent.origin = 'derived' AND parent.name_stage = 'accepted' AND parent.docs_stage = 'accepted' AND parent.unit IS NULL"
        accepted_seedable_unit_gaps = services._query_seedable_derived_parents(
            gc, state_where=accepted_unit_gap_where
        )
        accepted_legacy_unit_gaps = services._query_legacy_repairable_derived_parents(
            gc, state_where=accepted_unit_gap_where
        )
        cleanup_candidates = services._query_derived_parents_for_admission_cleanup(gc)
        invalid_parents = []
        for parent_id in cleanup_candidates:
            result = is_admissible_parent_name(parent_id, gc)
            if not result.admit:
                invalid_parents.append(parent_id)
        deleted = services._delete_derived_parent_nodes(gc, invalid_parents)
        if deleted:
            logger.info(
                "normalize_derived_parent_lifecycle: deleted %d inadmissible derived parents",
                deleted,
            )
        if invalid_parents:
            invalid_parent_set = set(invalid_parents)
            seedable_parents = [
                row
                for row in seedable_parents
                if row["parent_id"] not in invalid_parent_set
            ]
            legacy_parents = [
                row
                for row in legacy_parents
                if row["parent_id"] not in invalid_parent_set
            ]
            accepted_seedable_unit_gaps = [
                row
                for row in accepted_seedable_unit_gaps
                if row["parent_id"] not in invalid_parent_set
            ]
            accepted_legacy_unit_gaps = [
                row
                for row in accepted_legacy_unit_gaps
                if row["parent_id"] not in invalid_parent_set
            ]
        reaped_parent_ids: set[str] = set()
        while True:
            childless_candidates = [
                r["id"]
                for r in gc.query(
                    "\n                    MATCH (p:StandardName {origin: 'derived'})\n                    WHERE NOT EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(p) }\n                    RETURN p.id AS id\n                    "
                )
                or []
                if r["id"] not in reaped_parent_ids
            ]
            childless = services.filter_automatic_deletion_candidates(
                gc, childless_candidates
            )
            if not childless:
                break
            reaped_parent_ids.update(childless)
            reaped = services._delete_derived_parent_nodes(gc, childless)
            if reaped:
                deleted += reaped
                logger.info(
                    "normalize_derived_parent_lifecycle: reaped %d childless derived parents",
                    reaped,
                )
        if reaped_parent_ids:
            seedable_parents = [
                row
                for row in seedable_parents
                if row["parent_id"] not in reaped_parent_ids
            ]
            legacy_parents = [
                row
                for row in legacy_parents
                if row["parent_id"] not in reaped_parent_ids
            ]
            accepted_seedable_unit_gaps = [
                row
                for row in accepted_seedable_unit_gaps
                if row["parent_id"] not in reaped_parent_ids
            ]
            accepted_legacy_unit_gaps = [
                row
                for row in accepted_legacy_unit_gaps
                if row["parent_id"] not in reaped_parent_ids
            ]
        services.sweep_orphaned_docs_revisions(gc)
        if (
            not seedable_parents
            and (not legacy_parents)
            and (not accepted_seedable_unit_gaps)
            and (not accepted_legacy_unit_gaps)
        ):
            return deleted
        rejected_pending_parent_ids: list[str] = []
        repaired = services._materialize_derived_parent_rows(
            gc,
            seedable_parents,
            rejected_pending_parent_ids=rejected_pending_parent_ids,
        )
        repaired += services._materialize_derived_parent_rows(
            gc,
            legacy_parents,
            infer_kind_from_existing_topology=True,
            rejected_pending_parent_ids=rejected_pending_parent_ids,
        )
        repaired += services._materialize_derived_parent_rows(
            gc,
            accepted_seedable_unit_gaps,
            rejected_pending_parent_ids=rejected_pending_parent_ids,
        )
        repaired += services._materialize_derived_parent_rows(
            gc,
            accepted_legacy_unit_gaps,
            infer_kind_from_existing_topology=True,
            rejected_pending_parent_ids=rejected_pending_parent_ids,
        )
        rejected_pending_parent_ids = list(dict.fromkeys(rejected_pending_parent_ids))
        if rejected_pending_parent_ids:
            rejected = services._delete_derived_parent_nodes(
                gc, rejected_pending_parent_ids
            )
            deleted += rejected
            if rejected:
                logger.info(
                    "normalize_derived_parent_lifecycle: deleted %d identity-invalid pending derived parents",
                    rejected,
                )
        logger.info(
            "normalize_derived_parent_lifecycle: repaired %d derived parent SNs",
            repaired,
        )
        return repaired + deleted
    finally:
        if _own_gc:
            gc.__exit__(None, None, None)


LIFECYCLELESS_STUB_MANIFEST_SCHEMA = "imas-codex.lifecycleless-stub-manifest"

LIFECYCLELESS_STUB_RECEIPT_SCHEMA = "imas-codex.lifecycleless-stub-receipt"

_LIFECYCLELESS_STUB_LOCK_QUERY = (
    "MATCH (stub:StandardName) "
    "WHERE stub.name_stage IS NULL "
    "AND (stub.status IS NULL OR stub.status = 'draft') "
    "AND stub.origin IS NULL "
    "SET stub._lifecycleless_reconcile_lock = true "
    "REMOVE stub._lifecycleless_reconcile_lock "
    "RETURN count(stub) AS locked"
)


class LifecyclelessStubConflict(RuntimeError):
    """A lifecycle-less StandardName cohort changed or lacks authority."""


_READ_LIFECYCLELESS_STUBS_QUERY = """
// LIFECYCLELESS_STUB_AUTHORITY_READ
MATCH (stub:StandardName)
WHERE stub.name_stage IS NULL
  AND (stub.status IS NULL OR stub.status = 'draft')
  AND stub.origin IS NULL
CALL (stub) {
  OPTIONAL MATCH (child:StandardName)-[edge:HAS_PARENT]->(stub)
  RETURN collect(CASE WHEN child IS NULL THEN null ELSE {
    id: child.id,
    name_stage: child.name_stage,
    unit: child.unit,
    cocos: child.cocos_transformation_type,
    physics_domain: child.physics_domain,
    kind: child.kind,
    op_kind: edge.operator_kind,
    operator: edge.operator,
    role: edge.role,
    separator: edge.separator,
    axis: edge.axis,
    shape: edge.shape
  } END) AS child_data,
  collect(CASE WHEN edge IS NULL THEN null ELSE edge.operator_kind END) AS edge_kinds
}
CALL (stub) {
  OPTIONAL MATCH (stub)-[edge]-(related)
  RETURN collect(CASE WHEN edge IS NULL THEN null ELSE {
    direction: CASE WHEN startNode(edge) = stub THEN 'out' ELSE 'in' END,
    type: type(edge),
    properties: properties(edge),
    related_labels: labels(related),
    related_properties: properties(related)
  } END) AS incident_relationships
}
CALL (stub) {
  OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(stub)
  WHERE source.source_type = 'dd' OR EXISTS { (source)-[:FROM_DD_PATH]->(:IMASNode) }
  OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(bound:StandardName)
  OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
  WITH source, dd, collect(DISTINCT {
    id: bound.id,
    name_stage: bound.name_stage,
    validation_status: bound.validation_status
  }) AS binding_targets
  RETURN collect(CASE WHEN source IS NULL THEN null ELSE {
    source_id: source.id,
    expected_status: source.status,
    expected_scalar: source.produced_sn_id,
    expected_bindings: [target IN binding_targets WHERE target.id IS NOT NULL |
                        target.id],
    binding_targets: [target IN binding_targets WHERE target.id IS NOT NULL |
                      target],
    dd_path: dd.id,
    unit: coalesce(dd.unit, source.dd_unit, dd['units'])
  } END) AS dd_sources
}
CALL (stub) {
  OPTIONAL MATCH (source:StandardNameSource)
  WHERE source.produced_sn_id = stub.id
     OR EXISTS { (source)-[:PRODUCED_NAME]->(stub) }
     OR (source.source_type = 'derived' AND source.source_id = stub.id)
  CALL (source) {
    OPTIONAL MATCH (source)-[edge]-(related)
    RETURN collect(CASE WHEN edge IS NULL THEN null ELSE {
      direction: CASE WHEN startNode(edge) = source THEN 'out' ELSE 'in' END,
      type: type(edge), properties: properties(edge),
      related_labels: labels(related),
      related_properties: properties(related)
    } END) AS incident
  }
  RETURN collect(CASE WHEN source IS NULL THEN null ELSE {
    properties: properties(source), incident: incident
  } END) AS source_closure,
  collect(DISTINCT source.id) AS producer_ids
}
CALL (stub) {
  OPTIONAL MATCH (stub)-[:HAS_REVIEW]->(review:StandardNameReview)
  CALL (review) {
    OPTIONAL MATCH (review)-[edge]-(related)
    RETURN collect(CASE WHEN edge IS NULL THEN null ELSE {
      direction: CASE WHEN startNode(edge) = review THEN 'out' ELSE 'in' END,
      type: type(edge), properties: properties(edge),
      related_labels: labels(related),
      related_properties: properties(related)
    } END) AS incident
  }
  RETURN collect(CASE WHEN review IS NULL THEN null ELSE {
    properties: properties(review), incident: incident
  } END) AS review_closure
}
CALL (stub) {
  OPTIONAL MATCH (stub)-[:DOCS_REVISION_OF]->(revision:DocsRevision)
  CALL (revision) {
    OPTIONAL MATCH (revision)-[edge]-(related)
    RETURN collect(CASE WHEN edge IS NULL THEN null ELSE {
      direction: CASE WHEN startNode(edge) = revision THEN 'out' ELSE 'in' END,
      type: type(edge), properties: properties(edge),
      related_labels: labels(related),
      related_properties: properties(related)
    } END) AS incident
  }
  RETURN collect(CASE WHEN revision IS NULL THEN null ELSE {
    properties: properties(revision), incident: incident
  } END) AS revision_closure
}
RETURN stub.id AS id, properties(stub) AS properties,
       child_data, edge_kinds, dd_sources, incident_relationships,
       source_closure, producer_ids, review_closure, revision_closure
ORDER BY id
"""


def _canonicalize_lifecycleless_manifest_value(value: Any, *, services: Any) -> Any:
    """Canonicalize nested graph collections for stable authority hashes."""
    normalized = services._authority_json_value(value)
    if isinstance(normalized, dict):
        return {
            key: services._canonicalize_lifecycleless_manifest_value(item)
            for key, item in sorted(normalized.items())
        }
    if isinstance(normalized, list):
        items = [
            services._canonicalize_lifecycleless_manifest_value(item)
            for item in normalized
        ]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    return normalized


def _derived_parent_unit_from_eligible_children(
    parent_id: str, child_data: list[dict[str, Any]], *, services: Any
) -> str | None:
    """Apply the sanctioned binary and normalization unit exclusions."""
    eligible = services._eligible_derived_parent_unit_children(parent_id, child_data)
    units = {str(child["unit"]) for child in eligible if child.get("unit")}
    return next(iter(units)) if len(units) == 1 else None


def _spurious_stub_binding_authority(
    name_id: str,
    dd_sources: list[dict[str, Any]],
    producer_ids: list[str],
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Return accepted-sibling authority only for the complete producer set."""
    authorities: list[dict[str, Any]] = []
    for source in dd_sources:
        bindings = sorted(set(source.get("expected_bindings") or ()))
        targets = [dict(target) for target in source.get("binding_targets") or ()]
        accepted_siblings = [
            target
            for target in targets
            if target.get("id") != name_id
            and target.get("name_stage") == "accepted"
            and (target.get("validation_status") == "valid")
        ]
        if (
            not source.get("source_id")
            or not source.get("expected_status")
            or (not source.get("dd_path"))
            or (not source.get("unit"))
            or (len(bindings) != 2)
            or (name_id not in bindings)
            or (sorted(target.get("id") for target in targets) != bindings)
            or (len(accepted_siblings) != 1)
            or (source.get("expected_scalar") not in (None, accepted_siblings[0]["id"]))
        ):
            return []
        sibling = accepted_siblings[0]
        authorities.append(
            {
                "source_id": source["source_id"],
                "expected_status": source["expected_status"],
                "expected_scalar": source.get("expected_scalar"),
                "expected_bindings": bindings,
                "authoritative_target_id": sibling["id"],
                "authoritative_name_stage": sibling["name_stage"],
                "authoritative_validation_status": sibling["validation_status"],
            }
        )
    expected_producer_ids = sorted(set(producer_ids))
    qualifying_source_ids = sorted(
        {str(authority["source_id"]) for authority in authorities}
    )
    if (
        len(authorities) != len(dd_sources)
        or len(authorities) != len(qualifying_source_ids)
        or qualifying_source_ids != expected_producer_ids
    ):
        return []
    for authority in authorities:
        authority["expected_producer_ids"] = expected_producer_ids
    return authorities


def _partition_lifecycleless_stub_rows(
    rows: list[dict[str, Any]],
    *,
    admission_by_id: dict[str, Any] | None = None,
    services: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Partition a fresh lifecycle-less cohort without inventing authority."""
    partitions: dict[str, list[dict[str, Any]]] = {
        "materialize-as-derived-parent": [],
        "delete-as-dead-link-stub": [],
        "rebind-source": [],
        "refused": [],
    }
    for raw in rows:
        row = services._authority_json_value(dict(raw))
        name_id = str(row.get("id") or "").strip()
        children = [dict(child) for child in row.get("child_data") or []]
        dd_sources = [dict(source) for source in row.get("dd_sources") or []]
        producer_ids = sorted(set(row.get("producer_ids") or ()))
        row["id"] = name_id
        if not name_id:
            row["refusal_reason"] = "missing StandardName identity"
            partitions["refused"].append(row)
            continue
        if dd_sources:
            spurious_authority = (
                services._spurious_stub_binding_authority(
                    name_id, dd_sources, producer_ids
                )
                if not children
                else []
            )
            if spurious_authority:
                row["spurious_binding_authority"] = spurious_authority
                partitions["delete-as-dead-link-stub"].append(row)
                continue
            incomplete = [
                source
                for source in dd_sources
                if not source.get("source_id")
                or not source.get("expected_status")
                or source.get("expected_scalar") != name_id
                or (sorted(source.get("expected_bindings") or []) != [name_id])
                or (not source.get("dd_path"))
                or (not source.get("unit"))
            ]
            if incomplete:
                row["refusal_reason"] = "incomplete DD source or unit authority"
                partitions["refused"].append(row)
            elif (
                children
                and services._derived_parent_unit_from_eligible_children(
                    name_id, children
                )
                is None
            ):
                row["refusal_reason"] = "no parent-owned unit authority"
                partitions["refused"].append(row)
            elif (
                children
                and admission_by_id is not None
                and (not admission_by_id[name_id].admit)
            ):
                row["refusal_reason"] = (
                    f"structural parent admission refused: {admission_by_id[name_id].reason}"
                )
                partitions["refused"].append(row)
            else:
                partitions["rebind-source"].append(row)
            continue
        if children:
            authorized_unit = services._derived_parent_unit_from_eligible_children(
                name_id, children
            )
            if any(child.get("name_stage") is None for child in children):
                row["refusal_reason"] = "incomplete child lifecycle or unit authority"
                partitions["refused"].append(row)
            elif authorized_unit is None:
                row["refusal_reason"] = "no parent-owned unit authority"
                partitions["refused"].append(row)
            elif admission_by_id is not None and (not admission_by_id[name_id].admit):
                row["refusal_reason"] = (
                    f"structural parent admission refused: {admission_by_id[name_id].reason}"
                )
                partitions["refused"].append(row)
            else:
                row["parent_id"] = name_id
                row["bootstrap_edges"] = [
                    {
                        "from_name": child["id"],
                        "to_name": name_id,
                        "operator": child.get("operator"),
                        "operator_kind": child.get("op_kind"),
                        "role": child.get("role"),
                        "separator": child.get("separator"),
                        "axis": child.get("axis"),
                        "shape": child.get("shape"),
                        "expected_name_stage": child.get("name_stage"),
                        "expected_unit": child.get("unit"),
                        "expected_cocos": child.get("cocos"),
                        "expected_physics_domain": child.get("physics_domain"),
                        "expected_kind": child.get("kind"),
                    }
                    for child in children
                ]
                partitions["materialize-as-derived-parent"].append(row)
            continue
        partitions["delete-as-dead-link-stub"].append(row)
    return partitions


def _lifecycleless_stub_manifest(
    partitions: dict[str, list[dict[str, Any]]], *, services: Any
) -> dict[str, Any]:
    return {
        "schema": LIFECYCLELESS_STUB_MANIFEST_SCHEMA,
        "operation": "reconcile_lifecycleless_standard_name_stubs",
        "rows": {
            action: services._canonicalize_lifecycleless_manifest_value(items)
            for action, items in sorted(partitions.items())
        },
    }


def _reconcile_spurious_stub_source_scalars(
    transaction: Any, partitions: dict[str, list[dict[str, Any]]], *, services: Any
) -> int:
    """CAS null source mirrors to the accepted sibling signed by the manifest."""
    repairs = [
        {"stub_id": row["id"], **authority}
        for row in partitions["delete-as-dead-link-stub"]
        for authority in row.get("spurious_binding_authority") or ()
    ]
    if not repairs:
        return 0
    rows = list(
        transaction.run(
            "\n            UNWIND $repairs AS repair\n            MATCH (source:StandardNameSource {id: repair.source_id})\n            MATCH (stub:StandardName {id: repair.stub_id})\n            MATCH (accepted:StandardName {id: repair.authoritative_target_id})\n            WHERE source.status = repair.expected_status\n              AND source.claimed_at IS NULL\n              AND source.claim_token IS NULL\n              AND ((repair.expected_scalar IS NULL\n                    AND source.produced_sn_id IS NULL)\n                   OR source.produced_sn_id = repair.expected_scalar)\n              AND stub.name_stage IS NULL\n              AND (stub.status IS NULL OR stub.status = 'draft')\n              AND stub.origin IS NULL\n              AND accepted.name_stage = repair.authoritative_name_stage\n              AND accepted.validation_status =\n                  repair.authoritative_validation_status\n              AND EXISTS { (source)-[:PRODUCED_NAME]->(stub) }\n              AND EXISTS { (source)-[:PRODUCED_NAME]->(accepted) }\n            CALL (stub) {\n              OPTIONAL MATCH (producer:StandardNameSource)\n              WHERE producer.produced_sn_id = stub.id\n                 OR EXISTS { (producer)-[:PRODUCED_NAME]->(stub) }\n                 OR (producer.source_type = 'derived'\n                     AND producer.source_id = stub.id)\n              RETURN collect(DISTINCT producer.id) AS current_producer_ids\n            }\n            CALL (source) {\n              MATCH (source)-[:PRODUCED_NAME]->(bound:StandardName)\n              RETURN collect(DISTINCT bound.id) AS current_bindings\n            }\n            WITH repair, source, accepted, current_bindings,\n                 current_producer_ids\n            WHERE size(current_producer_ids) = size(repair.expected_producer_ids)\n              AND all(id IN current_producer_ids\n                      WHERE id IN repair.expected_producer_ids)\n              AND all(id IN repair.expected_producer_ids\n                      WHERE id IN current_producer_ids)\n              AND size(current_bindings) = size(repair.expected_bindings)\n              AND all(id IN current_bindings WHERE id IN repair.expected_bindings)\n              AND all(id IN repair.expected_bindings WHERE id IN current_bindings)\n            SET source.produced_sn_id = accepted.id\n            RETURN count(*) AS repairs,\n                   count(DISTINCT CASE WHEN repair.expected_scalar IS NULL\n                                      THEN source END) AS sources\n            ",
            repairs=repairs,
        )
    )
    repaired = int(rows[0].get("repairs", 0)) if rows else 0
    if repaired != len(repairs):
        raise services.LifecyclelessStubConflict(
            "accepted sibling authority changed before stub deletion"
        )
    return int(rows[0].get("sources", 0))


_lifecycleless_stub_reconcile_apply = functools.partial(
    apply_signed_manifest,
    {},
    authority_adapter="lifecycleless-stub-reconcile",
    mutation_kind="reconcile-lifecycleless-stub",
    guard_set=(
        "materialize-derived-parent",
        "delete-dead-link-stub",
        "rebind-source",
        "refused",
    ),
    reason="exact lifecycle-less StandardName stub reconciliation",
)

_lifecycleless_stub_reconcile_signature = inspect.signature(
    _lifecycleless_stub_reconcile_apply
)

_lifecycleless_stub_reconcile_parameters = (
    _lifecycleless_stub_reconcile_signature.parameters
)

_SKELETON_PLACEHOLDER_PREDICATE = """
sn.created_at IS NULL
  AND sn.generated_at IS NULL
  AND sn.validation_status IS NULL
  AND sn.unit IS NULL
  AND sn.kind IS NULL
  AND sn.needs_composition IS NULL
  AND coalesce(sn.name_stage, '') IN ['', 'pending']
  AND NOT EXISTS { ()-[:HAS_PARENT]->(sn) }
  AND NOT EXISTS { ()-[:HAS_ERROR]->(sn) }
  AND NOT EXISTS { ()-[:HAS_STANDARD_NAME]->(sn) }
  AND NOT EXISTS {
      (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
  }
  AND NOT EXISTS {
      MATCH (source:StandardNameSource)
      WHERE source.produced_sn_id = sn.id
  }
"""


def _query_skeleton_placeholders_for_cleanup(
    gc: Any, candidate_ids: set[str], *, services: Any
) -> list[str]:
    """Return relationship endpoints that positively match placeholder state."""
    if not candidate_ids:
        return []
    rows = gc.query(
        f"\n        // STANDARD_NAME_SKELETON_PLACEHOLDER_SELECTION\n        MATCH (sn:StandardName)\n        WHERE sn.id IN $candidate_ids\n          AND {_SKELETON_PLACEHOLDER_PREDICATE}\n        RETURN sn.id AS candidate_id\n        ",
        candidate_ids=sorted(candidate_ids),
    )
    return sorted(
        {str(row["candidate_id"]) for row in rows or [] if row.get("candidate_id")}
    )

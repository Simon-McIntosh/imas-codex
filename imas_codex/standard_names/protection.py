"""Symmetric ownership guards for standard-name graph writes.

Prevents the codex LLM pipeline from overwriting editorial content
that has passed the catalog approval gate.
All writers of protected fields call ``filter_protected()`` before
persisting to the graph.

Catalog writers have the inverse constraint: review results and their
authority records belong to the pipeline.  They call
``refuse_pipeline_authority_loss()`` before issuing a write.  Unlike the
editorial filter, this guard refuses the complete batch instead of silently
removing fields from it.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Fields that are catalog-authoritative after approval.
#: Pipeline writers must not overwrite these without override=True.
PROTECTED_FIELDS: frozenset[str] = frozenset(
    {
        "description",
        "documentation",
        "kind",
        "links",
        "status",
        "deprecates",
        "superseded_by",
        "validity_domain",
        "constraints",
    }
)

#: Scalar review projections owned by the pipeline, separated by review axis.
#: Catalog writers may neither clear nor replace an existing value.
PIPELINE_AUTHORITY_FIELDS: frozenset[str] = frozenset(
    {
        "reviewer_score_name",
        "reviewer_model_name",
        "reviewer_score_docs",
        "reviewer_model_docs",
    }
)

#: Relationship slots carrying the terminal review and structural authority.
#: These are LinkML slot names, not raw relationship-type spellings, so catalog
#: payload validation follows the schema-owned write shape.
PIPELINE_AUTHORITY_RELATIONSHIPS: frozenset[str] = frozenset(
    {"reviews", "structural_authorities"}
)


class PipelineAuthorityError(RuntimeError):
    """A catalog write cannot prove that pipeline authority is preserved."""


class ProtectedDeletionError(RuntimeError):
    """An automatic deletion targeted an identity carrying durable authority."""


def materialize_llm_cost_name_edges(gc: Any) -> int:
    """Materialize positive spend-to-name authority once per graph pass."""
    if vars(gc).get("_llm_cost_name_edges_materialized", False):
        return 0
    rows = gc.query(
        """
        MATCH (cost:LLMCost)
        WHERE coalesce(cost.llm_cost, 0.0) > 0.0
          AND size(coalesce(cost.standard_name_ids, [])) > 0
        UNWIND cost.standard_name_ids AS name_id
        MATCH (sn:StandardName {id: name_id})
        MERGE (cost)-[:FOR_STANDARD_NAME]->(sn)
        RETURN count(*) AS linked
        """
    )
    gc._llm_cost_name_edges_materialized = True
    return int(rows[0]["linked"]) if rows else 0


def automatic_deletion_protections(gc: Any, name_ids: list[str]) -> dict[str, str]:
    """Return durable authority evidence that bars automatic deletion.

    A derived-parent cleanup may ask whether structural scaffolding is still
    warranted.  It must not use that answer to withdraw a name that a catalog
    cut, ratification, or approval has made durable authority.  The result is
    keyed by identity so callers can name the exact evidence that refused a
    deletion.
    """
    names = sorted({name for name in name_ids if name})
    if not names:
        return {}
    materialize_llm_cost_name_edges(gc)
    rows = gc.query(
        """
        UNWIND $names AS name
        MATCH (sn:StandardName {id: name})
        OPTIONAL MATCH (sn)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
        WITH sn, collect(DISTINCT change.operation) AS operations
        OPTIONAL MATCH (cost:LLMCost)-[:FOR_STANDARD_NAME]->(sn)
        WITH sn, operations, sum(coalesce(cost.llm_cost, 0.0)) AS recorded_spend
        WITH sn, operations, recorded_spend,
             [reason IN [
               CASE WHEN sn.name_stage = 'approved'
                    THEN 'name_stage=approved' END,
               CASE WHEN sn.catalog_pr_number IS NOT NULL
                    THEN 'catalog_pr_number' END,
               CASE WHEN sn.catalog_merge_commit_sha IS NOT NULL
                    THEN 'catalog_merge_commit_sha' END,
               CASE WHEN sn.catalog_commit_sha IS NOT NULL
                    THEN 'catalog_commit_sha' END,
               CASE WHEN sn.exported_at IS NOT NULL
                    THEN 'exported_at' END,
               CASE WHEN 'unchanged_ratification' IN operations
                    THEN 'unchanged_ratification' END,
               CASE WHEN 'content_edit' IN operations
                    THEN 'content_edit' END,
               CASE WHEN recorded_spend > 0.0
                    THEN 'recorded_llm_spend_usd=' + toString(recorded_spend) END
             ] WHERE reason IS NOT NULL] AS reasons
        WHERE size(reasons) > 0
        RETURN sn.id AS id, reasons, recorded_spend
        """,
        names=names,
    )
    return {
        str(row["id"]): ", ".join(sorted(str(reason) for reason in row["reasons"]))
        for row in (rows or [])
        if row.get("id") and row.get("reasons")
    }


def filter_automatic_deletion_candidates(gc: Any, name_ids: list[str]) -> list[str]:
    """Return deletion candidates that carry no durable authority evidence.

    Candidate selection is the routine path: protected identities are not
    proposed for deletion.  Direct deleters must still call
    :func:`refuse_protected_automatic_deletion` so authority added between
    selection and mutation remains a whole-batch refusal.
    """
    candidates = sorted({name_id for name_id in name_ids if name_id})
    protected_ids = set(automatic_deletion_protections(gc, candidates))
    return [name_id for name_id in candidates if name_id not in protected_ids]


def refuse_protected_automatic_deletion(
    gc: Any, name_ids: list[str], *, operation: str
) -> None:
    """Refuse a whole automatic batch when any identity is protected."""
    protected_reasons = automatic_deletion_protections(gc, name_ids)
    _filtered, protected_ids = filter_protected(
        [{"id": name_id, "status": "delete"} for name_id in name_ids],
        protected_names=set(protected_reasons),
    )
    if protected_ids:
        details = ", ".join(
            f"{name_id} ({protected_reasons[name_id]})" for name_id in protected_ids
        )
        raise ProtectedDeletionError(
            f"Refused {operation} automatic deletion for protected identity: " + details
        )


class RecommendedNameError(RuntimeError):
    """An apply targeted a name already recommended into a catalog request."""


def _frozen_roster_requests(reviews_dir: Any | None) -> dict[str, str]:
    """Map each name of an opened frozen review roster to its request evidence.

    A frozen review roster under ``manifests/reviews/`` is the reproducible
    batch identity carried through export → PR → merge. Once the roster has
    recorded the candidate pull request it recomposed (``pr_number``/``pr_url``),
    its members are text a physicist has been asked to review. Folding the
    request back promotes those members to ``name_stage=approved``, which
    :func:`recommended_name_protections` reads from the graph; until then the
    roster itself is the only record that they are committed.

    Returns ``{standard_name_id: "candidate_request=<url>"}``. Reads files, not
    the graph; a missing or unreadable roster contributes no evidence.
    """
    from pathlib import Path

    import yaml

    from imas_codex.standard_names.catalog_release import default_reviews_dir

    directory = Path(reviews_dir) if reviews_dir else default_reviews_dir()
    evidence: dict[str, str] = {}
    if not directory.is_dir():
        return evidence
    for path in sorted(directory.glob("*.sn_names.yaml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(doc, dict):
            continue
        if not (doc.get("pr_number") or doc.get("pr_url")):
            # No candidate request recorded: the batch was never recommended.
            continue
        pr_ref = doc.get("pr_url") or f"pr_number={doc.get('pr_number')}"
        for name in doc.get("names") or []:
            evidence.setdefault(str(name), f"candidate_request={pr_ref}")
    return evidence


def recommended_name_protections(
    gc: Any,
    name_ids: list[str],
    *,
    reviews_dir: Any | None = None,
) -> dict[str, str]:
    """Return evidence that a name is recommended into a catalog request.

    Three classes make a name committed, so a documentation rewrite would
    change text a physicist has been asked to review, or has already reviewed:

    * a member of a frozen review roster whose candidate request is open;
    * a name at ``name_stage=approved`` (accepted through a merged request);
    * a name at ``name_stage=contested`` (edited by a reviewer during one).

    The result is keyed by identity so a caller can name the exact evidence
    that refused a move. Reads the roster files once and the graph once.
    """
    names = sorted({name for name in name_ids if name})
    if not names:
        return {}
    evidence: dict[str, set[str]] = {}
    wanted = set(names)
    for name_id, reason in _frozen_roster_requests(reviews_dir).items():
        if name_id in wanted:
            evidence.setdefault(name_id, set()).add(reason)
    rows = (
        gc.query(
            """
            UNWIND $names AS name
            MATCH (sn:StandardName {id: name})
            RETURN sn.id AS id, sn.name_stage AS name_stage
            """,
            names=names,
        )
        or []
    )
    for row in rows:
        stage = row.get("name_stage")
        if stage == "approved":
            evidence.setdefault(str(row["id"]), set()).add("name_stage=approved")
        elif stage == "contested":
            evidence.setdefault(str(row["id"]), set()).add("name_stage=contested")
    return {
        name_id: ", ".join(sorted(reasons))
        for name_id, reasons in evidence.items()
        if reasons
    }


def _authority_value(value: Any) -> Any:
    """Normalize relationship collections without changing scalar meaning."""
    if isinstance(value, list | tuple | set | frozenset):
        return tuple(sorted(str(item) for item in value))
    return value


def _has_authority(value: Any) -> bool:
    """Return whether an existing scalar or relationship records authority."""
    if isinstance(value, list | tuple | set | frozenset):
        return bool(value)
    return value is not None


def refuse_pipeline_authority_loss(
    items: list[dict[str, Any]],
    *,
    current_by_id: Mapping[str, Mapping[str, Any]] | None,
    identity_key: str = "id",
) -> list[dict[str, Any]]:
    """Refuse catalog payloads that clear or replace pipeline authority.

    ``current_by_id`` must come from a successful graph read immediately before
    the catalog write.  Passing ``None`` is therefore a refusal rather than an
    empty-state fallback.  Omitted authority keys mean "leave unchanged";
    explicitly supplied keys must equal the authoritative graph value whenever
    one already exists.

    The returned batch is a shallow copy and the input is never mutated.
    """
    if current_by_id is None:
        raise PipelineAuthorityError(
            "Refused catalog write: pipeline-authoritative graph state "
            "could not be read"
        )

    authority_keys = PIPELINE_AUTHORITY_FIELDS | PIPELINE_AUTHORITY_RELATIONSHIPS
    violations: list[str] = []
    copied: list[dict[str, Any]] = []

    for item in items:
        name_id = item.get(identity_key)
        if not name_id:
            raise PipelineAuthorityError(
                "Refused catalog write: pipeline-authoritative comparison "
                f"requires a non-empty {identity_key!r}"
            )

        copied.append(copy.copy(item))
        current = current_by_id.get(str(name_id))
        if current is None:
            continue

        for key in sorted(authority_keys & item.keys()):
            existing = current.get(key)
            if not _has_authority(existing):
                continue
            if _authority_value(item[key]) != _authority_value(existing):
                violations.append(f"{name_id}.{key}")

    if violations:
        raise PipelineAuthorityError(
            "Refused catalog write that would null or replace "
            "pipeline-authoritative provenance: " + ", ".join(violations)
        )

    return copied


def filter_protected(
    items: list[dict[str, Any]],
    *,
    override: bool = False,
    override_names: set[str] | None = None,
    protected_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Strip protected editorial fields from approved items.

    Parameters
    ----------
    items:
        Dicts to filter. Each must have an ``"id"`` key (the standard name).
    override:
        When ``True``, bypass protection — all fields pass through.
    override_names:
        Selective override — set of standard name IDs that should bypass
        protection even if they have passed approval. Other names remain
        protected. Ignored when ``override=True``.
    protected_names:
        Pre-fetched set of standard name IDs whose ``name_stage`` is
        ``'approved'``. If ``None``, queries the graph to determine protection
        status. Callers in hot loops should pre-fetch.

    Returns
    -------
    tuple of (filtered_items, skipped_names):
        - ``filtered_items``: new list with protected fields stripped from
          approved items. Non-protected fields pass through. Items that have
          not passed approval remain unchanged.
        - ``skipped_names``: list of item IDs that had fields stripped.

    Notes
    -----
    Does not mutate the input list or its dicts.
    """
    if override:
        return items, []

    if protected_names is None:
        protected_names = _fetch_catalog_edit_names(
            [it["id"] for it in items if "id" in it]
        )

    # Selective per-name override: remove explicitly overridden names
    # from the protected set so their fields pass through.
    if override_names:
        protected_names = protected_names - override_names

    filtered: list[dict[str, Any]] = []
    skipped: list[str] = []

    for item in items:
        name_id = item.get("id", "")
        if name_id in protected_names:
            stripped = {k: v for k, v in item.items() if k not in PROTECTED_FIELDS}
            if len(stripped) < len(item):
                skipped.append(name_id)
                logger.warning(
                    "Stripped %d protected field(s) from catalog-edited name '%s'",
                    len(item) - len(stripped),
                    name_id,
                )
            filtered.append(stripped)
        else:
            # Shallow copy to avoid mutating caller's dict
            filtered.append(copy.copy(item))

    return filtered, skipped


def _fetch_catalog_edit_names(name_ids: list[str]) -> set[str]:
    """Query graph for names that have passed the approval gate."""
    if not name_ids:
        return set()
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $names AS name
            MATCH (sn:StandardName {id: name})
            WHERE sn.name_stage = 'approved'
            RETURN sn.id AS id
            """,
            names=name_ids,
        )
        return {r["id"] for r in (rows or [])}

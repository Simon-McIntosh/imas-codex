"""Symmetric ownership guards for standard-name graph writes.

Prevents the codex LLM pipeline from overwriting editorial content
that was manually curated via a catalog PR (origin=catalog_edit).
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

#: Fields that are catalog-authoritative when origin=catalog_edit.
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
    """Strip protected editorial fields from catalog-edited items.

    Parameters
    ----------
    items:
        Dicts to filter. Each must have an ``"id"`` key (the standard name).
    override:
        When ``True``, bypass protection — all fields pass through.
    override_names:
        Selective override — set of standard name IDs that should bypass
        protection even if they have ``origin='catalog_edit'``.  Other
        names remain protected.  Ignored when ``override=True``.
    protected_names:
        Pre-fetched set of standard name IDs whose ``origin`` is
        ``'catalog_edit'``. If ``None``, queries the graph to determine
        protection status. Callers in hot loops should pre-fetch.

    Returns
    -------
    tuple of (filtered_items, skipped_names):
        - ``filtered_items``: new list with protected fields stripped from
          catalog-edited items. Non-protected fields pass through. Items
          without ``origin`` or with ``origin='pipeline'`` pass unchanged.
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
    """Query graph for curator-owned or PR-approved names."""
    if not name_ids:
        return set()
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = gc.query(
                """
                UNWIND $names AS name
                MATCH (sn:StandardName {id: name})
                WHERE sn.origin = 'catalog_edit' OR sn.name_stage = 'approved'
                RETURN sn.id AS id
                """,
                names=name_ids,
            )
            return {r["id"] for r in (rows or [])}
    except Exception:
        logger.warning(
            "Failed to query catalog_edit names — treating all as pipeline",
            exc_info=True,
        )
        return set()

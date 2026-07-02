"""Deterministic drift detection and sibling-set assembly for SN docs.

Standard names are generated per-name, so sibling families (names sharing
a parent via ``HAS_PARENT``, or sharing a ``physical_base`` when parentless)
can drift apart in documentation structure even though they describe
structurally related quantities. This module computes a deterministic
drift metric over sibling descriptions and assembles a harmonization
worklist — families that are both large enough and drifted enough to be
worth re-aligning.

This module is read-only against the graph: it never writes StandardName
properties. See :mod:`imas_codex.cli.sn` (``sn harmonize --report``) for
the CLI surface.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from imas_codex.graph.models import NameStage
from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
)

#: Operator kinds eligible for parent-based family grouping.
_FAMILY_OPERATOR_KINDS: tuple[str, ...] = ("projection", "qualifier", "coordinate")

#: docs_stage value indicating an accepted documentation revision.
_DOCS_STAGE_ACCEPTED = "accepted"

_TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+")


def doc_sig(desc: str, n: int = 6) -> str:
    """Deterministic signature over the first *n* word-tokens of *desc*.

    Tokens are extracted with a simple alpha/digit regex, lower-cased,
    and digit tokens are collapsed to ``'#'`` so numeric variation (e.g.
    axis indices) does not itself count as drift. Returns a space-joined
    signature string; empty/short descriptions yield a shorter signature.
    """
    tokens = _TOKEN_RE.findall((desc or "").lower())
    normalized = ["#" if tok.isdigit() else tok for tok in tokens[:n]]
    return " ".join(normalized)


def _is_placeholder(description: str | None) -> bool:
    return (
        not description or description == DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
    )


def drift(members: list[dict[str, Any]], *, n: int = 6) -> float:
    """Structural drift over a family's member descriptions.

    ``drift = 1 - (largest cohort size / member count)`` where a cohort is
    a set of members sharing the same :func:`doc_sig`. A uniform family
    (all members share one signature) has drift 0.0; a family with no two
    alike has drift ``1 - 1/n``.
    """
    if not members:
        return 0.0
    sigs = [doc_sig(m.get("description") or "", n) for m in members]
    counts: dict[str, int] = {}
    for sig in sigs:
        counts[sig] = counts.get(sig, 0) + 1
    max_cohort = max(counts.values())
    return 1.0 - (max_cohort / len(members))


def group_signature(members: list[dict[str, Any]]) -> str:
    """Stable, content-sensitive signature for a family's membership + docs.

    Changes whenever the member set changes (ids added/removed) or any
    member's description/documentation text changes. Order-invariant over
    the member list — members are sorted by id before hashing.
    """
    ordered = sorted(members, key=lambda m: m.get("id") or "")
    ids_part = ",".join(m.get("id") or "" for m in ordered)
    per_member_hashes = []
    for m in ordered:
        content = (m.get("description") or "") + "\x1f" + (m.get("documentation") or "")
        per_member_hashes.append(hashlib.sha256(content.encode("utf-8")).hexdigest())
    payload = ids_part + "|" + "|".join(sorted(per_member_hashes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_anchor(
    parent_id: str | None,
    parent_docs_stage: str | None,
    parent_description: str | None,
    members: list[dict[str, Any]],
) -> str | None:
    """Select the documentation anchor for a family.

    Priority order:
      1. The parent itself, if ``docs_stage == 'accepted'`` and its
         description is real (non-placeholder).
      2. The member with the highest ``reviewer_score_docs`` among members
         with ``docs_stage == 'accepted'``.
      3. The member with the longest non-placeholder description.

    Returns ``None`` when no candidate qualifies (family is flagged
    ``deferred`` by the caller).
    """
    if (
        parent_id
        and parent_docs_stage == _DOCS_STAGE_ACCEPTED
        and not _is_placeholder(parent_description)
    ):
        return parent_id

    accepted = [
        m
        for m in members
        if m.get("docs_stage") == _DOCS_STAGE_ACCEPTED
        and not _is_placeholder(m.get("description"))
    ]
    scored = [m for m in accepted if m.get("reviewer_score_docs") is not None]
    if scored:
        best = max(
            scored,
            key=lambda m: (m.get("reviewer_score_docs") or 0.0, m.get("id") or ""),
        )
        return best.get("id")
    if accepted:
        # Accepted but unscored — still a valid anchor; prefer longest desc.
        best = max(
            accepted,
            key=lambda m: (len(m.get("description") or ""), m.get("id") or ""),
        )
        return best.get("id")

    non_placeholder = [m for m in members if not _is_placeholder(m.get("description"))]
    if non_placeholder:
        best = max(
            non_placeholder,
            key=lambda m: (len(m.get("description") or ""), m.get("id") or ""),
        )
        return best.get("id")

    return None


def _top_divergent_member(
    members: list[dict[str, Any]], anchor_sig: str | None
) -> str | None:
    """Return the member id whose doc_sig differs most from the anchor's.

    Simple heuristic: prefer any member whose signature differs from the
    modal (largest-cohort) signature; ties broken by id.
    """
    if not members:
        return None
    sigs = {m.get("id"): doc_sig(m.get("description") or "") for m in members}
    counts: dict[str, int] = {}
    for sig in sigs.values():
        counts[sig] = counts.get(sig, 0) + 1
    if not counts:
        return None
    modal_sig = max(
        counts.items(), key=lambda kv: (kv[1], kv[0] != (anchor_sig or ""))
    )[0]
    divergent = sorted(
        mid for mid, sig in sigs.items() if sig != modal_sig and mid is not None
    )
    return divergent[0] if divergent else None


def _row_to_member(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "description": row.get("description"),
        "documentation": row.get("documentation"),
        "docs_stage": row.get("docs_stage"),
        "reviewer_score_docs": row.get("reviewer_score_docs"),
        "operator_kind": row.get("operator_kind"),
    }


def _resolve_gc(gc: Any | None) -> tuple[Any, bool]:
    if gc is not None:
        return gc, False
    from imas_codex.graph.client import GraphClient

    return GraphClient(), True


def _close_if_owned(gc: Any, own_gc: bool) -> None:
    if own_gc:
        try:
            gc.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public: assemble_family
# ---------------------------------------------------------------------------


def assemble_family(seed_name: str, gc: Any | None = None) -> dict[str, Any] | None:
    """Assemble the full live sibling family for *seed_name*.

    Looks up the seed's parent (via ``HAS_PARENT`` with a family-eligible
    ``operator_kind``); if found, returns the parent's full sibling set.
    Falls back to grouping by ``physical_base`` when the seed has no such
    parent edge.

    Returns a dict with keys ``parent`` (id or ``None``), ``grouping``
    (``"parent"`` or ``"physical_base"``), ``operator_kinds`` (sorted list
    of distinct kinds seen among members), ``members`` (list of member
    dicts), and ``anchor`` (selected anchor id or ``None``). Returns
    ``None`` if the seed itself is not found or is superseded.
    """
    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        seed_rows = (
            resolved_gc.query(
                """
                MATCH (c:StandardName {id: $seed})
                WHERE coalesce(c.name_stage, '') <> $superseded
                RETURN c.id AS id, c.physical_base AS physical_base
                """,
                seed=seed_name,
                superseded=NameStage.superseded.value,
            )
            or []
        )
        if not seed_rows:
            return None

        parent_rows = (
            resolved_gc.query(
                """
                MATCH (c:StandardName {id: $seed})-[r:HAS_PARENT]->(p:StandardName)
                WHERE r.operator_kind IN $kinds
                RETURN p.id AS parent_id
                LIMIT 1
                """,
                seed=seed_name,
                kinds=list(_FAMILY_OPERATOR_KINDS),
            )
            or []
        )

        if parent_rows and parent_rows[0].get("parent_id"):
            parent_id = parent_rows[0]["parent_id"]
            member_rows = (
                resolved_gc.query(
                    """
                    MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName {id: $parent_id})
                    WHERE r.operator_kind IN $kinds
                      AND coalesce(c.name_stage, '') <> $superseded
                      AND c.description IS NOT NULL
                    RETURN c.id AS id, c.description AS description,
                           c.documentation AS documentation,
                           c.docs_stage AS docs_stage,
                           c.reviewer_score_docs AS reviewer_score_docs,
                           r.operator_kind AS operator_kind
                    """,
                    parent_id=parent_id,
                    kinds=list(_FAMILY_OPERATOR_KINDS),
                    superseded=NameStage.superseded.value,
                )
                or []
            )
            parent_info_rows = (
                resolved_gc.query(
                    """
                    MATCH (p:StandardName {id: $parent_id})
                    RETURN p.docs_stage AS docs_stage, p.description AS description
                    """,
                    parent_id=parent_id,
                )
                or []
            )
            parent_info = parent_info_rows[0] if parent_info_rows else {}
            members = [_row_to_member(r) for r in member_rows]
            kinds_seen = sorted(
                {m["operator_kind"] for m in members if m["operator_kind"]}
            )
            anchor = select_anchor(
                parent_id,
                parent_info.get("docs_stage"),
                parent_info.get("description"),
                members,
            )
            return {
                "parent": parent_id,
                "grouping": "parent",
                "operator_kinds": kinds_seen,
                "members": members,
                "anchor": anchor,
            }

        physical_base = seed_rows[0].get("physical_base")
        if not physical_base:
            return {
                "parent": None,
                "grouping": "physical_base",
                "operator_kinds": [],
                "members": [_row_to_member(seed_rows[0])],
                "anchor": None,
            }

        member_rows = (
            resolved_gc.query(
                """
                MATCH (c:StandardName)
                WHERE c.physical_base = $pb
                  AND coalesce(c.name_stage, '') <> $superseded
                  AND c.description IS NOT NULL
                  AND NOT EXISTS {
                    MATCH (c)-[r:HAS_PARENT]->(:StandardName)
                    WHERE r.operator_kind IN $kinds
                  }
                RETURN c.id AS id, c.description AS description,
                       c.documentation AS documentation,
                       c.docs_stage AS docs_stage,
                       c.reviewer_score_docs AS reviewer_score_docs
                """,
                pb=physical_base,
                kinds=list(_FAMILY_OPERATOR_KINDS),
                superseded=NameStage.superseded.value,
            )
            or []
        )
        members = [_row_to_member(r) for r in member_rows]
        anchor = select_anchor(None, None, None, members)
        return {
            "parent": None,
            "grouping": "physical_base",
            "operator_kinds": [],
            "members": members,
            "anchor": anchor,
            "physical_base": physical_base,
        }
    finally:
        _close_if_owned(resolved_gc, own_gc)


# ---------------------------------------------------------------------------
# Public: build_worklist
# ---------------------------------------------------------------------------


def build_worklist(
    gc: Any | None = None,
    *,
    min_drift: float = 0.5,
    min_size: int = 3,
    include_parentless: bool = False,
) -> list[dict[str, Any]]:
    """Build the harmonization worklist ranked by (drift desc, n desc).

    Enumerates families grouped by shared ``HAS_PARENT`` parent (family-
    eligible ``operator_kind`` only), plus — when *include_parentless* is
    True — a fallback bucket of parentless names grouped by
    ``physical_base``. Filters to families with ``n >= min_size`` and
    ``drift >= min_drift``, excluding families whose
    ``harmonized_group_signature`` already matches the current signature
    (a missing property is treated as "never harmonized").

    Each worklist entry is a dict with keys: ``parent`` (id or ``None``),
    ``grouping``, ``operator_kinds``, ``n``, ``drift``,
    ``drift_secondary`` (drift over an 8-token signature), ``docs_accepted``
    (count of members with ``docs_stage == 'accepted'``), ``anchor`` (id or
    ``None``), ``deferred`` (bool — True when anchor is ``None``),
    ``top_divergent_member``, ``members`` (list of member ids),
    ``group_signature`` (current signature), ``physical_base`` (only for
    the parentless bucket).
    """
    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        family_rows = (
            resolved_gc.query(
                """
                MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName)
                WHERE r.operator_kind IN $kinds
                  AND coalesce(c.name_stage, '') <> $superseded
                  AND c.description IS NOT NULL
                RETURN p.id AS parent_id,
                       p.docs_stage AS parent_docs_stage,
                       p.description AS parent_description,
                       p.harmonized_group_signature AS harmonized_group_signature,
                       collect({
                           id: c.id,
                           description: c.description,
                           documentation: c.documentation,
                           docs_stage: c.docs_stage,
                           reviewer_score_docs: c.reviewer_score_docs,
                           operator_kind: r.operator_kind
                       }) AS members
                """,
                kinds=list(_FAMILY_OPERATOR_KINDS),
                superseded=NameStage.superseded.value,
            )
            or []
        )

        worklist: list[dict[str, Any]] = []
        for row in family_rows:
            members = list(row.get("members") or [])
            n = len(members)
            if n < min_size:
                continue
            d = drift(members, n=6)
            if d < min_drift:
                continue
            d_secondary = drift(members, n=8)
            sig = group_signature(members)
            existing_sig = row.get("harmonized_group_signature")
            if existing_sig is not None and existing_sig == sig:
                continue

            parent_id = row.get("parent_id")
            anchor = select_anchor(
                parent_id,
                row.get("parent_docs_stage"),
                row.get("parent_description"),
                members,
            )
            docs_accepted = sum(
                1 for m in members if m.get("docs_stage") == _DOCS_STAGE_ACCEPTED
            )
            kinds_seen = sorted(
                {m.get("operator_kind") for m in members if m.get("operator_kind")}
            )
            worklist.append(
                {
                    "parent": parent_id,
                    "grouping": "parent",
                    "operator_kinds": kinds_seen,
                    "n": n,
                    "drift": d,
                    "drift_secondary": d_secondary,
                    "docs_accepted": docs_accepted,
                    "anchor": anchor,
                    "deferred": anchor is None,
                    "top_divergent_member": _top_divergent_member(
                        members, doc_sig(anchor and _anchor_desc(members, anchor) or "")
                    ),
                    "members": sorted(m.get("id") for m in members if m.get("id")),
                    "group_signature": sig,
                }
            )

        if include_parentless:
            parentless_rows = (
                resolved_gc.query(
                    """
                    MATCH (c:StandardName)
                    WHERE c.physical_base IS NOT NULL
                      AND coalesce(c.name_stage, '') <> $superseded
                      AND c.description IS NOT NULL
                      AND NOT EXISTS {
                        MATCH (c)-[r:HAS_PARENT]->(:StandardName)
                        WHERE r.operator_kind IN $kinds
                      }
                    RETURN c.physical_base AS physical_base,
                           collect({
                               id: c.id,
                               description: c.description,
                               documentation: c.documentation,
                               docs_stage: c.docs_stage,
                               reviewer_score_docs: c.reviewer_score_docs,
                               harmonized_group_signature: c.harmonized_group_signature
                           }) AS members
                    """,
                    kinds=list(_FAMILY_OPERATOR_KINDS),
                    superseded=NameStage.superseded.value,
                )
                or []
            )
            for row in parentless_rows:
                members = list(row.get("members") or [])
                n = len(members)
                if n < min_size:
                    continue
                d = drift(members, n=6)
                if d < min_drift:
                    continue
                d_secondary = drift(members, n=8)
                sig = group_signature(members)
                # Parentless families have no single parent node to stamp;
                # use the first member's signature property as a proxy —
                # coalesce (missing) means "never harmonized".
                existing_sigs = {
                    m.get("harmonized_group_signature")
                    for m in members
                    if m.get("harmonized_group_signature") is not None
                }
                if existing_sigs == {sig}:
                    continue

                anchor = select_anchor(None, None, None, members)
                docs_accepted = sum(
                    1 for m in members if m.get("docs_stage") == _DOCS_STAGE_ACCEPTED
                )
                worklist.append(
                    {
                        "parent": None,
                        "grouping": "physical_base",
                        "physical_base": row.get("physical_base"),
                        "operator_kinds": [],
                        "n": n,
                        "drift": d,
                        "drift_secondary": d_secondary,
                        "docs_accepted": docs_accepted,
                        "anchor": anchor,
                        "deferred": anchor is None,
                        "top_divergent_member": _top_divergent_member(members, None),
                        "members": sorted(m.get("id") for m in members if m.get("id")),
                        "group_signature": sig,
                    }
                )

        worklist.sort(
            key=lambda f: (
                -f["drift"],
                -f["n"],
                f.get("parent") or f.get("physical_base") or "",
            )
        )
        return worklist
    finally:
        _close_if_owned(resolved_gc, own_gc)


def _anchor_desc(members: list[dict[str, Any]], anchor_id: str) -> str:
    for m in members:
        if m.get("id") == anchor_id:
            return m.get("description") or ""
    return ""


__all__ = [
    "doc_sig",
    "drift",
    "group_signature",
    "select_anchor",
    "assemble_family",
    "build_worklist",
]

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
_FAMILY_OPERATOR_KINDS: tuple[str, ...] = (
    "projection",
    "qualifier",
    "coordinate",
    "locus",
)

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
         with ``docs_stage == 'accepted'`` — the best signal we have that a
         member already meets the docs standard (accepted-but-unscored breaks
         ties deterministically by id).
      3. Last resort (no accepted member): a non-placeholder member chosen
         deterministically by id — NEVER by longest description, which
         propagates a verbose, non-canonical opening into the family.

    The anchor deliberately carries NO vocabulary or phrasing knowledge: the
    canonical opening a family should converge on is defined by the docs
    prompt/context, not hardcoded here. Score + determinism is the data-driven
    proxy — a member that already meets the standard scores well.

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
        # Accepted but unscored — deterministic by id (NOT longest description).
        return min(accepted, key=lambda m: m.get("id") or "").get("id")

    non_placeholder = [m for m in members if not _is_placeholder(m.get("description"))]
    if non_placeholder:
        # Last resort — deterministic by id (NOT longest description); picking
        # by length propagates a verbose, non-canonical opening to the family.
        return min(non_placeholder, key=lambda m: m.get("id") or "").get("id")

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


# ---------------------------------------------------------------------------
# Harmonization apply: mark + stamp orchestration
# ---------------------------------------------------------------------------

_LINK_RE = re.compile(r"\[([^\]]+)\]\(name:([a-z0-9_]+)\)")


def lint_links(gc: Any | None = None) -> list[dict[str, Any]]:
    """Scan accepted docs for markdown links whose text names a DIFFERENT
    standard name than their target resolves to.

    A link ``[label](name:target)`` is flagged when the snake_cased label is
    itself an existing StandardName id AND differs from ``target`` — the
    text promises one quantity while the href resolves to another (a real
    physics error class found across sibling families). Human-readable
    labels that aren't themselves ids are fine and skipped.

    Returns ``[{member, label, target}]`` sorted by member id.
    """
    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        rows = (
            resolved_gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.docs_stage = 'accepted'
                  AND sn.documentation IS NOT NULL
                RETURN sn.id AS id, sn.documentation AS documentation
                """
            )
            or []
        )
        all_ids = {
            r["id"]
            for r in (
                resolved_gc.query("MATCH (sn:StandardName) RETURN sn.id AS id") or []
            )
        }
    finally:
        _close_if_owned(resolved_gc, own_gc)

    findings: list[dict[str, Any]] = []
    for row in rows:
        for label, target in _LINK_RE.findall(row.get("documentation") or ""):
            label_id = label.strip().lower().replace(" ", "_").replace("-", "_")
            if label_id in all_ids and label_id != target:
                findings.append({"member": row["id"], "label": label, "target": target})
    return sorted(findings, key=lambda f: f["member"])


def mark_members_for_regen(
    member_ids: list[str],
    *,
    gc: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Mark an explicit member set for docs regeneration (the apply step).

    Snapshots each member's current docs to a DocsRevision, resets its docs
    pipeline to ``pending`` and stamps a fresh scope ``run_id`` so a
    subsequent ``sn run --scope-run-id <id> --docs-only --flush`` regenerates
    ONLY these names. Returns ``{"run_id", "eligible", "reset"}``.
    """
    import uuid

    from imas_codex.standard_names.graph_ops import reset_standard_name_docs

    run_id = str(uuid.uuid4())
    out = reset_standard_name_docs(
        sn_ids=member_ids,
        run_id=None if dry_run else run_id,
        dry_run=dry_run,
    )
    return {"run_id": run_id, **out}


def stamp_harmonized(
    family_parents: list[str],
    *,
    gc: Any | None = None,
) -> dict[str, int]:
    """Recompute + stamp the idempotency signature for the given families.

    For each parent id, reassembles the live family, recomputes the current
    group signature, and stamps ``harmonized_at`` +
    ``harmonized_group_signature`` on members AND parent — but only when
    every live member is docs-accepted (defer otherwise).

    Returns ``{"stamped": n, "deferred": m}``.
    """
    from imas_codex.standard_names.graph_ops import stamp_harmonized_families

    resolved_gc, own_gc = _resolve_gc(gc)
    families: list[dict[str, Any]] = []
    deferred = 0
    try:
        for parent_id in family_parents:
            rows = (
                resolved_gc.query(
                    """
                    MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName {id: $pid})
                    WHERE r.operator_kind IN $kinds
                      AND coalesce(c.name_stage, '') <> 'superseded'
                      AND c.description IS NOT NULL
                    RETURN c.id AS id, c.description AS description,
                           c.documentation AS documentation,
                           c.docs_stage AS docs_stage
                    """,
                    pid=parent_id,
                    kinds=list(_FAMILY_OPERATOR_KINDS),
                )
                or []
            )
            members = [dict(r) for r in rows]
            if not members or any(
                m.get("docs_stage") != _DOCS_STAGE_ACCEPTED for m in members
            ):
                deferred += 1
                continue
            families.append(
                {
                    "parent": parent_id,
                    "members": [m["id"] for m in members],
                    "signature": group_signature(members),
                }
            )
    finally:
        _close_if_owned(resolved_gc, own_gc)

    stamped = stamp_harmonized_families(families)
    return {"stamped": stamped, "deferred": deferred}


__all__ = [
    "doc_sig",
    "lint_links",
    "mark_members_for_regen",
    "mark_families_for_regen",
    "restamp_harmonized_families",
    "stamp_harmonized",
    "drift",
    "group_signature",
    "select_anchor",
    "assemble_family",
    "build_worklist",
]


def mark_families_for_regen(
    family_parents: list[str],
    *,
    gc: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Mark whole families for docs regeneration by parent id.

    Resolves each parent's live family (members via the family operator
    kinds, plus the parent itself), then resets the docs pipeline for the
    full set via :func:`mark_members_for_regen`. Returns that function's
    result plus ``member_ids`` and ``unknown_parents``.
    """
    resolved_gc, own_gc = _resolve_gc(gc)
    member_ids: list[str] = []
    unknown: list[str] = []
    try:
        for parent_id in family_parents:
            rows = (
                resolved_gc.query(
                    """
                    MATCH (p:StandardName {id: $pid})
                    OPTIONAL MATCH (c:StandardName)-[r:HAS_PARENT]->(p)
                    WHERE r.operator_kind IN $kinds
                      AND coalesce(c.name_stage, '') <> $superseded
                    RETURN p.id AS parent_id,
                           [x IN collect(c.id) WHERE x IS NOT NULL] AS kids
                    """,
                    pid=parent_id,
                    kinds=list(_FAMILY_OPERATOR_KINDS),
                    superseded=NameStage.superseded.value,
                )
                or []
            )
            if not rows:
                unknown.append(parent_id)
                continue
            member_ids.append(rows[0]["parent_id"])
            member_ids.extend(rows[0]["kids"] or [])
    finally:
        _close_if_owned(resolved_gc, own_gc)

    member_ids = sorted(set(member_ids))
    out = mark_members_for_regen(member_ids, dry_run=dry_run)
    return {**out, "member_ids": member_ids, "unknown_parents": unknown}


def restamp_harmonized_families(gc: Any | None = None) -> dict[str, int]:
    """Reconcile family idempotency signatures against current graph state.

    The automatic steady-state half of harmonization bookkeeping: every
    family whose live members are ALL docs-accepted gets its
    ``harmonized_at`` + ``harmonized_group_signature`` refreshed when the
    stored signature is missing or stale (a member joined or left, or a
    member's docs changed and re-passed review). Families with any
    non-accepted member are left alone — they stamp on a later run once
    their members clear the docs gate. Runs in every ``sn run`` post-drain
    reconcile; a no-op when nothing changed.

    Returns ``{"restamped": n, "unchanged": m, "not_ready": k}``.
    """
    from imas_codex.standard_names.graph_ops import stamp_harmonized_families

    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        rows = (
            resolved_gc.query(
                """
                MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName)
                WHERE r.operator_kind IN $kinds
                  AND coalesce(c.name_stage, '') <> $superseded
                  AND c.description IS NOT NULL
                WITH p, collect({
                    id: c.id,
                    description: c.description,
                    documentation: c.documentation,
                    docs_stage: c.docs_stage
                }) AS members
                WHERE size(members) >= 2
                RETURN p.id AS parent_id,
                       p.harmonized_group_signature AS stored_signature,
                       members
                """,
                kinds=list(_FAMILY_OPERATOR_KINDS),
                superseded=NameStage.superseded.value,
            )
            or []
        )
    finally:
        _close_if_owned(resolved_gc, own_gc)

    to_stamp: list[dict[str, Any]] = []
    unchanged = 0
    not_ready = 0
    for row in rows:
        members = list(row.get("members") or [])
        if any(m.get("docs_stage") != _DOCS_STAGE_ACCEPTED for m in members):
            not_ready += 1
            continue
        sig = group_signature(members)
        if row.get("stored_signature") == sig:
            unchanged += 1
            continue
        to_stamp.append(
            {
                "parent": row["parent_id"],
                "members": [m["id"] for m in members],
                "signature": sig,
            }
        )

    restamped = stamp_harmonized_families(to_stamp) if to_stamp else 0
    return {"restamped": restamped, "unchanged": unchanged, "not_ready": not_ready}

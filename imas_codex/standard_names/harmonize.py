"""Deterministic drift detection and sibling-set assembly for SN docs.

Standard names are generated per-name, so sibling families (names joined by
a family-eligible ``HAS_PARENT`` edge, or — when parentless — a shared
pattern signature over the invariant grammar segments) can drift apart in
documentation structure even though they describe structurally related
quantities. This module computes a deterministic drift metric over sibling
descriptions and assembles a harmonization worklist — families that are both
large enough and drifted enough to be worth re-aligning.

This module is read-only against the graph: it never writes StandardName
properties. See :mod:`imas_codex.cli.sn` (``sn harmonize --report``) for
the CLI surface.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from imas_codex.graph.models import NameStage
from imas_codex.standard_names.context import DOCUMENTATION_FAMILY_OPERATOR_KINDS
from imas_codex.standard_names.families import (
    DOCS_STAGE_ACCEPTED,
    PATTERN_INVARIANT_SEGMENTS,
    _invariant_equality,
    _invariant_projection,
    fetch_pattern_family,
    select_anchor,
)

#: Operator kinds eligible for parent-based family grouping.
_FAMILY_OPERATOR_KINDS = DOCUMENTATION_FAMILY_OPERATOR_KINDS

#: Averaging-reduction operator tokens whose HAS_PARENT edges also join a
#: family even though their kind is unary_prefix (excluded above so the
#: broad prefix class — normalized, time_derivative, … — stays out).
#: These pair a reduced quantity with its unreduced parent (e.g.
#: flux_surface_averaged_X_at_plasma_boundary with X_at_plasma_boundary),
#: which must docs-harmonize together or their descriptions drift.
_FAMILY_REDUCTION_OPERATORS: tuple[str, ...] = (
    "flux_surface_averaged",
    "maximum_over_flux_surface",
    "minimum_over_flux_surface",
)

_TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+")

#: Member predicate for the parentless bucket: a name with a base quantity
#: and a description and no operator parent is grouped by its pattern family.
_PARENTLESS_MEMBER_FILTER = (
    "m.physical_base IS NOT NULL AND m.description IS NOT NULL "
    "AND NOT EXISTS { MATCH (m)-[:HAS_PARENT]->(:StandardName) }"
)


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
        "operator": row.get("operator"),
    }


def _is_documentation_family_member(member: dict[str, Any]) -> bool:
    """Return whether structural edge metadata proves a safe docs cohort."""
    kind = member.get("operator_kind")
    operator = member.get("operator")
    return kind in _FAMILY_OPERATOR_KINDS or operator in _FAMILY_REDUCTION_OPERATORS


def _documentation_family_members(
    members: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop explicitly ineligible structural edges from a query result."""
    return [member for member in members if _is_documentation_family_member(member)]


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
    """Assemble the full live family for *seed_name*.

    Primary route: the seed's parent via ``HAS_PARENT`` with a family-eligible
    ``operator_kind`` (or a reduction operator). The operator axis is not a
    pattern axis, so operator-derived children are joined to their parent only
    here.

    Fallback: the seed's pattern family — every live name whose grammar parses
    identically except along
    :data:`~imas_codex.standard_names.families.PATTERN_VARIATION_AXES`.

    Returns a dict with keys ``parent`` (id or ``None``), ``grouping``
    (``"parent"`` or ``"pattern"``), ``operator_kinds`` (sorted list of
    distinct kinds seen among members), ``members`` (list of member dicts),
    and ``anchor`` (selected anchor id or ``None``). Returns ``None`` if the
    seed itself is not found or is superseded.
    """
    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        seed_rows = (
            resolved_gc.query(
                """
                MATCH (c:StandardName {id: $seed})
                WHERE coalesce(c.name_stage, '') <> $superseded
                RETURN c.id AS id
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
                RETURN p.id AS parent_id,
                       r.operator_kind AS operator_kind,
                       r.operator AS operator
                ORDER BY p.id
                """,
                seed=seed_name,
            )
            or []
        )
        eligible_parent_rows = _documentation_family_members(list(parent_rows))

        if eligible_parent_rows and eligible_parent_rows[0].get("parent_id"):
            parent_id = eligible_parent_rows[0]["parent_id"]
            member_rows = (
                resolved_gc.query(
                    """
                    MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName {id: $parent_id})
                    WHERE (r.operator_kind IN $kinds OR r.operator IN $reduction_ops)
                      AND coalesce(c.name_stage, '') <> $superseded
                      AND c.description IS NOT NULL
                    RETURN c.id AS id, c.description AS description,
                           c.documentation AS documentation,
                           c.docs_stage AS docs_stage,
                           c.reviewer_score_docs AS reviewer_score_docs,
                           r.operator_kind AS operator_kind,
                           r.operator AS operator
                    """,
                    parent_id=parent_id,
                    kinds=list(_FAMILY_OPERATOR_KINDS),
                    reduction_ops=list(_FAMILY_REDUCTION_OPERATORS),
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
            members = _documentation_family_members(
                [_row_to_member(r) for r in member_rows]
            )
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

        if parent_rows:
            return None

        rows = fetch_pattern_family(seed_name, resolved_gc)
        members = [_row_to_member(r) for r in rows]
        if not members:
            return None
        anchor = select_anchor(None, None, None, members)
        return {
            "parent": None,
            "grouping": "pattern",
            "operator_kinds": [],
            "members": members,
            "anchor": anchor,
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
    True — a bucket of parentless names grouped by the same pattern family
    the sibling-family context reads (the invariant segment tuple).
    Filters to families with ``n >= min_size`` and
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
                WHERE (r.operator_kind IN $kinds OR r.operator IN $reduction_ops)
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
                           operator_kind: r.operator_kind,
                           operator: r.operator
                       }) AS members
                """,
                kinds=list(_FAMILY_OPERATOR_KINDS),
                reduction_ops=list(_FAMILY_REDUCTION_OPERATORS),
                superseded=NameStage.superseded.value,
            )
            or []
        )

        worklist: list[dict[str, Any]] = []
        for row in family_rows:
            members = _documentation_family_members(list(row.get("members") or []))
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
                1 for m in members if m.get("docs_stage") == DOCS_STAGE_ACCEPTED
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
            for family in fetch_pattern_families(
                resolved_gc,
                node_filter=_PARENTLESS_MEMBER_FILTER,
                min_size=min_size,
            ):
                members = family["members"]
                d = drift(members, n=6)
                if d < min_drift:
                    continue
                d_secondary = drift(members, n=8)
                sig = group_signature(members)
                # Parentless families have no single parent node to stamp;
                # use the members' signature property as a proxy —
                # absent (coalesced missing) means "never harmonized".
                existing_sigs = {
                    m.get("harmonized_group_signature")
                    for m in members
                    if m.get("harmonized_group_signature") is not None
                }
                if existing_sigs == {sig}:
                    continue

                anchor = select_anchor(None, None, None, members)
                docs_accepted = sum(
                    1 for m in members if m.get("docs_stage") == DOCS_STAGE_ACCEPTED
                )
                worklist.append(
                    {
                        "parent": None,
                        "grouping": "pattern",
                        "physical_base": family["physical_base"],
                        "operator_kinds": [],
                        "n": len(members),
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


def _family_member_collect() -> str:
    """Cypher collecting one family's member rows with the report columns.

    The documentation columns drive anchor selection and drift; the lint
    columns (unit, COCOS type, physics domain, position, links, link status)
    are what the review lint reads, so the report hands it rows it can score
    without a second read.
    """
    return """collect({
        id: m.id,
        description: m.description,
        documentation: m.documentation,
        docs_stage: m.docs_stage,
        reviewer_score_docs: m.reviewer_score_docs,
        harmonized_group_signature: m.harmonized_group_signature,
        unit: m.unit,
        cocos_transformation_type: m.cocos_transformation_type,
        physics_domain: m.physics_domain,
        position: m.position,
        links: m.links,
        link_status: m.link_status
    }) AS members"""


def _pattern_family_rows_query(*, seeded: bool, node_filter: str) -> str:
    """Cypher enumerating pattern families, keyed on the invariant segments.

    *seeded* groups only the families that contain one of ``$seeds`` (two
    hops: the seeds' signatures, then every live name sharing one). Otherwise
    every live family is grouped. *node_filter* is an extra predicate on the
    member alias ``m``, so a caller narrows the population without restating
    the grouping — the fragment that used to be a second definition of a
    family in this module.
    """
    segments = ", ".join(PATTERN_INVARIANT_SEGMENTS)
    projection = _invariant_projection("m")
    collect = _family_member_collect()
    extra = f"\n  AND {node_filter}" if node_filter else ""
    if seeded:
        seed_projection = _invariant_projection("seed")
        equality = _invariant_equality("m")
        return f"""
MATCH (seed:StandardName)
WHERE seed.id IN $seeds AND coalesce(seed.name_stage, '') <> $superseded
WITH DISTINCT
{seed_projection}
MATCH (m:StandardName)
WHERE coalesce(m.name_stage, '') <> $superseded
  AND {equality}{extra}
WITH
{projection},
     {collect}
RETURN {segments}, members
"""
    return f"""
MATCH (m:StandardName)
WHERE coalesce(m.name_stage, '') <> $superseded{extra}
WITH
{projection},
     {collect}
RETURN {segments}, members
"""


def fetch_pattern_families(
    gc: Any | None = None,
    *,
    seeds: list[str] | None = None,
    node_filter: str = "",
    min_size: int = 1,
) -> list[dict[str, Any]]:
    """Enumerate live pattern families grouped by invariant grammar signature.

    The one family-enumeration mechanism in this module. Pass *seeds* to keep
    only the families containing at least one seed id; ``None`` enumerates
    every live family with at least *min_size* members. *node_filter* is an
    extra predicate on the member alias, so a caller selects a population
    without restating the grouping.

    Returns one dict per family: ``invariant`` (the tuple of invariant
    segment values), ``physical_base``, ``members`` (member rows carrying the
    documentation and lint columns) and ``n``. Read-only: it never writes.
    """
    if seeds is not None and not seeds:
        return []
    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        rows = (
            resolved_gc.query(
                _pattern_family_rows_query(
                    seeded=seeds is not None, node_filter=node_filter
                ),
                seeds=sorted(set(seeds or [])),
                superseded=NameStage.superseded.value,
            )
            or []
        )
    finally:
        _close_if_owned(resolved_gc, own_gc)

    families: list[dict[str, Any]] = []
    for row in rows:
        members = [dict(m) for m in (row.get("members") or [])]
        if len(members) < min_size:
            continue
        families.append(
            {
                "invariant": tuple(
                    row.get(seg) or "" for seg in PATTERN_INVARIANT_SEGMENTS
                ),
                "physical_base": row.get("physical_base") or "",
                "members": members,
                "n": len(members),
            }
        )
    families.sort(key=lambda f: (f["physical_base"], f["invariant"]))
    return families


def concept_member_ids(concept: str, gc: Any | None = None) -> list[str]:
    """Live standard-name ids whose grammar selects registry concept *concept*."""
    from imas_standard_names.grammar.terms import load_conventions, select_conventions

    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        rows = (
            resolved_gc.query(
                """
                MATCH (m:StandardName)
                WHERE coalesce(m.name_stage, '') <> $superseded
                RETURN m.id AS id
                """,
                superseded=NameStage.superseded.value,
            )
            or []
        )
    finally:
        _close_if_owned(resolved_gc, own_gc)

    registry = load_conventions()
    selected: list[str] = []
    for row in rows:
        name_id = row.get("id")
        if not name_id:
            continue
        try:
            conventions = select_conventions(name_id, registry=registry)
        except Exception:
            continue
        if any(concept_obj.token == concept for concept_obj in conventions.concepts):
            selected.append(name_id)
    return sorted(selected)


def report_family_rows(
    gc: Any | None = None,
    *,
    seeds: list[str] | None = None,
    min_size: int = 1,
) -> list[dict[str, Any]]:
    """Assemble the read-only harmonization report rows for families in scope.

    One row per pattern family: the member ids, the anchor, the drift metric
    and the lint findings by class, sorted by (drift desc, size desc, base).
    The lint is the composition of
    :func:`~imas_codex.standard_names.review.audits.run_lint_checks` plus
    :func:`~imas_codex.standard_names.review.audits.run_link_integrity` — the
    report never calls ``run_all_audits``, whose embedding preflight writes.
    Nothing here writes to the graph.
    """
    from imas_codex.standard_names.review.audits import (
        run_link_integrity,
        run_lint_checks,
    )

    families = fetch_pattern_families(gc, seeds=seeds, min_size=min_size)
    in_scope = [member for family in families for member in family["members"]]

    by_name: dict[str, dict[str, int]] = {}
    for finding in run_lint_checks(in_scope):
        counts = by_name.setdefault(finding.name_id, {})
        counts[finding.finding_type] = counts.get(finding.finding_type, 0) + 1
    for link_finding in run_link_integrity(in_scope):
        counts = by_name.setdefault(link_finding.name_id, {})
        key = f"link_{link_finding.finding_type}"
        counts[key] = counts.get(key, 0) + 1

    rows: list[dict[str, Any]] = []
    for family in families:
        members = family["members"]
        lint: dict[str, int] = {}
        for member in members:
            for finding_class, count in by_name.get(member.get("id"), {}).items():
                lint[finding_class] = lint.get(finding_class, 0) + count
        rows.append(
            {
                "physical_base": family["physical_base"],
                "invariant": list(family["invariant"]),
                "members": sorted(m.get("id") for m in members if m.get("id")),
                "n": len(members),
                "anchor": select_anchor(None, None, None, members),
                "drift": drift(members),
                "lint": dict(sorted(lint.items())),
                "lint_total": sum(lint.values()),
            }
        )
    rows.sort(key=lambda r: (-r["drift"], -r["n"], r["physical_base"]))
    return rows


def _anchor_desc(members: list[dict[str, Any]], anchor_id: str) -> str:
    for m in members:
        if m.get("id") == anchor_id:
            return m.get("description") or ""
    return ""


# ---------------------------------------------------------------------------
# Harmonization apply: mark + stamp orchestration
# ---------------------------------------------------------------------------


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
                    WHERE (r.operator_kind IN $kinds OR r.operator IN $reduction_ops)
                      AND coalesce(c.name_stage, '') <> 'superseded'
                      AND c.description IS NOT NULL
                    RETURN c.id AS id, c.description AS description,
                           c.documentation AS documentation,
                           c.docs_stage AS docs_stage,
                           r.operator_kind AS operator_kind,
                           r.operator AS operator
                    """,
                    pid=parent_id,
                    kinds=list(_FAMILY_OPERATOR_KINDS),
                    reduction_ops=list(_FAMILY_REDUCTION_OPERATORS),
                )
                or []
            )
            members = _documentation_family_members([dict(r) for r in rows])
            if not members or any(
                m.get("docs_stage") != DOCS_STAGE_ACCEPTED for m in members
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
    "mark_members_for_regen",
    "restamp_harmonized_families",
    "stamp_harmonized",
    "drift",
    "group_signature",
    "select_anchor",
    "assemble_family",
    "build_worklist",
    "fetch_pattern_families",
    "concept_member_ids",
    "report_family_rows",
]


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
                WHERE (r.operator_kind IN $kinds OR r.operator IN $reduction_ops)
                  AND coalesce(c.name_stage, '') <> $superseded
                  AND c.description IS NOT NULL
                WITH p, collect({
                    id: c.id,
                    description: c.description,
                    documentation: c.documentation,
                    docs_stage: c.docs_stage,
                    operator_kind: r.operator_kind,
                    operator: r.operator
                }) AS members
                WHERE size(members) >= 2
                RETURN p.id AS parent_id,
                       p.harmonized_group_signature AS stored_signature,
                       members
                """,
                kinds=list(_FAMILY_OPERATOR_KINDS),
                reduction_ops=list(_FAMILY_REDUCTION_OPERATORS),
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
        members = _documentation_family_members(list(row.get("members") or []))
        if len(members) < 2:
            continue
        if any(m.get("docs_stage") != DOCS_STAGE_ACCEPTED for m in members):
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

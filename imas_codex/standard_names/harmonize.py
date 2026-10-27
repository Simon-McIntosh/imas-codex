"""Deterministic drift detection and sibling-set assembly for SN docs.

Standard names are generated per-name, so sibling families (names joined by
a family-eligible ``HAS_PARENT`` edge, or — when parentless — a shared
pattern signature over the invariant grammar segments) can drift apart in
documentation structure even though they describe structurally related
quantities. This module computes a deterministic drift metric over sibling
descriptions and assembles a harmonization worklist — families that are both
large enough and drifted enough to be worth re-aligning.

The report surface is read-only against the graph and never writes
StandardName properties. The apply surface writes two things and no more:
each member's docs pipeline is reset so a later ``sn run`` regenerates it
(:func:`mark_members_for_regen`), and a converged family's sibling links are
made reciprocal (:func:`reconcile_family_reverse_links`) through the
``graph_ops`` field writer that already persists them. See
:mod:`imas_codex.cli.sn` (``sn harmonize --report``) for the CLI surface.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from imas_codex.graph.models import NameStage
from imas_codex.standard_names.context import DOCUMENTATION_FAMILY_OPERATOR_KINDS
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.doc_links import (
    find_name_links,
    find_name_references,
)

# The family-enumeration query is owned by families.py; re-exported so the
# worklist test can compare the query build_worklist issues against it.
from imas_codex.standard_names.families import (
    DOCS_STAGE_ACCEPTED,
    _pattern_family_rows_query as _pattern_family_rows_query,
    dd_source_exists,
    fetch_pattern_families,
    fetch_pattern_family,
    select_anchor,
)

logger = logging.getLogger(__name__)

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
        "has_dd_source": row.get("has_dd_source"),
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
                    f"""
                    MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName {{id: $parent_id}})
                    WHERE (r.operator_kind IN $kinds OR r.operator IN $reduction_ops)
                      AND coalesce(c.name_stage, '') <> $superseded
                      AND c.description IS NOT NULL
                    RETURN c.id AS id, c.description AS description,
                           c.documentation AS documentation,
                           c.docs_stage AS docs_stage,
                           c.reviewer_score_docs AS reviewer_score_docs,
                           r.operator_kind AS operator_kind,
                           r.operator AS operator,
                           {dd_source_exists("c")} AS has_dd_source
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
                    f"""
                    MATCH (p:StandardName {{id: $parent_id}})
                    RETURN p.docs_stage AS docs_stage,
                           p.description AS description,
                           {dd_source_exists("p")} AS has_dd_source
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
                parent_has_dd_source=parent_info.get("has_dd_source"),
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
                f"""
                MATCH (c:StandardName)-[r:HAS_PARENT]->(p:StandardName)
                WHERE (r.operator_kind IN $kinds OR r.operator IN $reduction_ops)
                  AND coalesce(c.name_stage, '') <> $superseded
                  AND c.description IS NOT NULL
                RETURN p.id AS parent_id,
                       p.docs_stage AS parent_docs_stage,
                       p.description AS parent_description,
                       {dd_source_exists("p")} AS parent_has_dd_source,
                       p.harmonized_group_signature AS harmonized_group_signature,
                       collect({{
                           id: c.id,
                           description: c.description,
                           documentation: c.documentation,
                           docs_stage: c.docs_stage,
                           reviewer_score_docs: c.reviewer_score_docs,
                           operator_kind: r.operator_kind,
                           operator: r.operator,
                           has_dd_source: {dd_source_exists("c")}
                       }}) AS members
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
                parent_has_dd_source=row.get("parent_has_dd_source"),
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


#: Prefix of the lexical lint's cross-name finding ids. A position-form
#: conflict is a property of a physics domain's names as a set, not of any one
#: name, so ``run_lexical_lint`` keys it on ``[domain:<d>]`` — an id that
#: matches no member and must be attributed to the families in that domain.
_DOMAIN_FINDING_PREFIX = "[domain:"


def _domain_scoped_finding(name_id: str) -> str | None:
    """The physics domain of a domain-scoped finding id, or ``None``.

    Returns ``None`` for an ordinary per-name finding id, which the report
    attributes by member id as before.
    """
    if name_id.startswith(_DOMAIN_FINDING_PREFIX) and name_id.endswith("]"):
        return name_id[len(_DOMAIN_FINDING_PREFIX) : -1]
    return None


#: Dead-link resolution reads the whole catalog, not the scope under report:
#: a target is dead only when no ``StandardName`` anywhere carries it. The read
#: has exactly one definition, ``graph_ops.get_existing_standard_names``, called
#: here for both callers — with a client it rides that session, without one the
#: owner opens its own. Bounded either way: the read names its row set.
def known_name_ids(gc: Any | None) -> set[str]:
    """Every ``StandardName`` id the graph holds, for dead-link resolution."""
    from imas_codex.standard_names.graph_store.graph_claim_scope import (
        get_existing_standard_names,
    )

    return get_existing_standard_names(gc)


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
    Nothing here writes to the graph. Link-target deadness resolves against
    every name.

    A lint finding scoped to a physics domain rather than to one name — the
    lexical lint's cross-name position-form findings carry the synthetic id
    ``[domain:<d>]`` — matches no member id, so it is attributed here to the
    families in scope whose members belong to that domain (row key
    ``domain_lint``). The scope total counts each distinct finding exactly
    once: a domain finding belongs to every qualifying family row but is
    counted in the scope total at the first family in report order, so summing
    the rows' ``lint_total`` is the number of distinct findings.
    """
    from imas_codex.standard_names.domain_priority import domain_key
    from imas_codex.standard_names.review.audits import (
        run_link_integrity,
        run_lint_checks,
    )

    families = fetch_pattern_families(gc, seeds=seeds, min_size=min_size)
    in_scope = [member for family in families for member in family["members"]]
    # A link target is dead only when the graph holds no such name. The scoped
    # members alone are the report's subject, not the test for deadness: many
    # valid references point outside a family.
    known_ids = known_name_ids(gc)

    by_name: dict[str, dict[str, int]] = {}
    # (domain, class) → count, one entry per domain-scoped finding.
    by_domain: dict[tuple[str, str], int] = {}
    for finding in run_lint_checks(in_scope):
        domain = _domain_scoped_finding(finding.name_id)
        if domain is not None:
            key = (domain, f"domain_{finding.finding_type}")
            by_domain[key] = by_domain.get(key, 0) + 1
            continue
        counts = by_name.setdefault(finding.name_id, {})
        counts[finding.finding_type] = counts.get(finding.finding_type, 0) + 1
    for link_finding in run_link_integrity(in_scope, known_ids=known_ids):
        counts = by_name.setdefault(link_finding.name_id, {})
        key = f"link_{link_finding.finding_type}"
        counts[key] = counts.get(key, 0) + 1

    rows: list[dict[str, Any]] = []
    family_domains: list[set[str]] = []
    for family in families:
        members = family["members"]
        domains = {
            domain_key(m.get("physics_domain"), fallback="") for m in members
        } - {""}
        lint: dict[str, int] = {}
        for member in members:
            for finding_class, count in by_name.get(member.get("id"), {}).items():
                lint[finding_class] = lint.get(finding_class, 0) + count
        domain_lint = {
            finding_class: count
            for (domain, finding_class), count in by_domain.items()
            if domain in domains
        }
        family_domains.append(domains)
        rows.append(
            {
                "physical_base": family["physical_base"],
                "invariant": list(family["invariant"]),
                "members": sorted(m.get("id") for m in members if m.get("id")),
                "n": len(members),
                "anchor": resolve_family_anchor(members),
                "drift": drift(members),
                "lint": dict(sorted(lint.items())),
                "domain_lint": dict(sorted(domain_lint.items())),
                "lint_total": sum(lint.values()),
            }
        )

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            -rows[i]["drift"],
            -rows[i]["n"],
            rows[i]["physical_base"],
        ),
    )
    rows = [rows[i] for i in order]
    order_domains = [family_domains[i] for i in order]

    # Count each domain finding once: at the first family in report order that
    # belongs to its domain. Every finding of one domain lands at that family,
    # so the summed total stays the number of distinct findings.
    owner_of: dict[str, int] = {}
    for idx, domains in enumerate(order_domains):
        for domain in domains:
            owner_of.setdefault(domain, idx)
    for (domain, finding_class), count in sorted(by_domain.items()):
        owner = owner_of.get(domain)
        if owner is None:
            continue
        rows[owner]["lint"][finding_class] = (
            rows[owner]["lint"].get(finding_class, 0) + count
        )
        rows[owner]["lint_total"] += count
    for row in rows:
        row["lint"] = dict(sorted(row["lint"].items()))
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

    from imas_codex.standard_names.graph_store.graph_docs_persistence import (
        reset_standard_name_docs,
    )

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
    from imas_codex.standard_names.graph_store.graph_docs_persistence import (
        stamp_harmonized_families,
    )

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


# ── Pattern-family apply: the anchor-gated campaign ─────────────────────────
# A pattern family has no parent node, so its idempotency mark lives on the
# members themselves (``harmonized_group_signature``). The apply is one
# CampaignRunner per family with an explicit two-batch plan: the anchor alone,
# then the remaining members. See ``plan_family_applies`` / ``apply_families``.

# ── Cost of a documented name, measured from the ledger ─────────────────────
# The dry run has to state a cost before any call is placed. It measures that
# cost from what recent runs actually spent rather than from a fixed guess: the
# per-name spend of the documentation seat — the regenerations a harmonize
# apply places and the reviews that follow them — read from the ``LLMCost``
# rows the pools record.

#: The documentation-seat phases a regeneration cycle bills. A planned call
#: regenerates one member's documentation and pays the review that follows it,
#: so the cycle's spend is the summed cost of these phases for that name.
DOC_CALL_PHASES: tuple[str, ...] = ("generate_docs", "refine_docs", "review_docs")

#: How far back the dry run measures, and how many recent runs it reads. The
#: read is bounded by the number of runs, never by a row count taken before the
#: per-run aggregation: a global cap cuts a run mid-way, so its cost and the
#: names it covered are reported for only the rows the cap happened to keep.
_DOC_COST_WINDOW_DAYS = 14
_DOC_COST_RUN_CAP = 200

#: The most rows one run may contribute before the read is treated as no
#: longer bounded. The window and the run cap already bound the read; this is
#: the stated per-run bound the reader checks every returned run against, so a
#: run that outgrows it is reported rather than silently truncated.
_DOC_COST_ROWS_PER_RUN = 20000

#: The recent runs are selected first, bounded by ``$run_cap``; only then is
#: every row of those runs aggregated. No row cap is applied before the
#: grouping, so a run contributes all of its rows whatever its size.
_DOC_COST_QUERY = """
MATCH (c:LLMCost)
WHERE c.phase IN $phases AND c.llm_at >= datetime($since)
  AND c.run_id IS NOT NULL
WITH c.run_id AS run_id, max(c.llm_at) AS newest
ORDER BY newest DESC
LIMIT $run_cap
WITH collect(run_id) AS run_ids
MATCH (c:LLMCost)
WHERE c.run_id IN run_ids AND c.phase IN $phases
  AND c.run_id IS NOT NULL AND c.llm_at >= datetime($since)
WITH c.run_id AS run_id, sum(c.llm_cost) AS run_cost,
     collect(c.standard_name_ids) AS name_lists, count(c) AS rows
RETURN run_id AS run_id, run_cost AS run_cost, name_lists AS name_lists,
       rows AS rows
"""


@dataclass(frozen=True)
class DocCallCostEstimate:
    """The measured cost of one documentation call, and its basis.

    ``runs`` is how many recorded runs the measurement spans; ``calls`` is how
    many distinct names those runs documented (one planned call regenerates one
    name); ``total_usd`` is the documentation-seat spend those calls accounted
    for. ``mean_usd`` is the cost per call the dry run estimates from.
    """

    runs: int
    calls: int
    total_usd: float

    @property
    def mean_usd(self) -> float:
        return self.total_usd / self.calls if self.calls else 0.0


def measure_doc_call_cost(
    gc: Any,
    *,
    window_days: int = _DOC_COST_WINDOW_DAYS,
    run_cap: int = _DOC_COST_RUN_CAP,
) -> DocCallCostEstimate | None:
    """Measure the cost of one documentation call from the recent ledger.

    Reads the ``LLMCost`` rows for the documentation-seat phases
    (:data:`DOC_CALL_PHASES`) recorded within *window_days* across the
    *run_cap* most recent runs, aggregating every row of each of those runs,
    and returns their spend per documented name. Every run's row count is
    checked against :data:`_DOC_COST_ROWS_PER_RUN`, so a run large enough to
    make the read unbounded is reported rather than silently cut. ``None``
    when the ledger holds no such measurement, so the caller states the
    absence rather than falling back to a guess.
    """
    since = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
    rows = gc.query(
        _DOC_COST_QUERY,
        phases=list(DOC_CALL_PHASES),
        since=since,
        run_cap=run_cap,
    )
    runs = 0
    calls: set[str] = set()
    total_usd = 0.0
    for row in rows or []:
        if row.get("run_id") is None:
            continue
        counted = int(row.get("rows") or 0)
        if counted > _DOC_COST_ROWS_PER_RUN:
            raise ValueError(
                f"run {row['run_id']!r} contributes {counted} LLMCost rows, "
                f"over the per-run bound of {_DOC_COST_ROWS_PER_RUN}; the read "
                f"is unbounded and the run's spend would be understated"
            )
        runs += 1
        total_usd += float(row.get("run_cost") or 0.0)
        for name_list in row.get("name_lists") or []:
            calls.update(name_list or [])
    if runs == 0 or not calls:
        return None
    return DocCallCostEstimate(runs=runs, calls=len(calls), total_usd=total_usd)


#: The member property a started apply stamps with the anchor it chose. It
#: rides the family's own run state — the same member nodes carry ``run_id``
#: and ``harmonized_group_signature`` — so a replan interrupted mid-run reuses
#: the recorded anchor instead of re-ranking a reset member's null score.
_FAMILY_ANCHOR_PROPERTY = "harmonized_anchor"


def pinned_family_anchor(members: list[dict[str, Any]]) -> str | None:
    """The anchor a prior apply recorded for this family, or ``None``.

    One ``harmonized_anchor`` shared by the members is the family's recorded
    choice. A family whose members disagree, or carry none, has no pin and
    selects normally. The pin is released by the family's harmonization stamp
    and by nothing else: a forced apply waives the recommended-name refusal,
    it does not move the run onto a different anchor.
    """
    recorded = {
        m.get(_FAMILY_ANCHOR_PROPERTY)
        for m in members
        if m.get(_FAMILY_ANCHOR_PROPERTY)
    }
    return next(iter(recorded)) if len(recorded) == 1 else None


def resolve_family_anchor(
    members: list[dict[str, Any]],
    *,
    member_ids: list[str] | None = None,
    signature: str | None = None,
) -> str | None:
    """The anchor an apply would use for this family.

    The single definition of what anchor a family shows: ``select_anchor``'s
    live ranking, replaced by the family's recorded pin unless the family is
    already stamped harmonized. The stamp releases the pin, so a converged
    family reranks and an interrupted one keeps the anchor its run recorded.
    The report and the dry run both resolve a family's anchor here, so the
    anchor an operator reads is the one an apply would start on.
    ``member_ids`` and ``signature`` are accepted so a caller that already
    derived them from the same members is not made to recompute.
    """
    anchor = select_anchor(None, None, None, members)
    if member_ids is None:
        member_ids = sorted(m.get("id") for m in members if m.get("id"))
    if signature is None:
        signature = group_signature(members)
    stored = {
        m.get("harmonized_group_signature")
        for m in members
        if m.get("harmonized_group_signature") is not None
    }
    if not (member_ids and stored == {signature}):
        pinned = pinned_family_anchor(members)
        if pinned is not None and pinned in member_ids:
            anchor = pinned
    return anchor


def _record_family_anchor(gc: Any, member_ids: list[str], anchor_id: str) -> None:
    """Stamp the chosen anchor on the family's members as an apply starts."""
    gc.query(
        f"UNWIND $ids AS sid MATCH (sn:StandardName {{id: sid}}) "
        f"SET sn.{_FAMILY_ANCHOR_PROPERTY} = $anchor",
        ids=list(member_ids),
        anchor=anchor_id,
    )


def _clear_family_anchor(gc: Any, member_ids: list[str]) -> None:
    """Release the pinned anchor once the family is stamped harmonized."""
    gc.query(
        f"UNWIND $ids AS sid MATCH (sn:StandardName {{id: sid}}) "
        f"REMOVE sn.{_FAMILY_ANCHOR_PROPERTY}",
        ids=list(member_ids),
    )


# ── Post-convergence reverse links ──────────────────────────────────────────
# A member's ``links`` are derived from its own documentation, and each member's
# documentation is regenerated from its own prompt in which the family's sibling
# block is one of many inputs. Two members told to link "at least one adjacent
# sibling" therefore need not choose each other, so a converged family can carry
# one-way sibling links: a ``name:m`` in one member's documentation with no
# reverse in the sibling's. This step makes the sibling relation symmetric once
# the family has converged: it adds the reverse of a sibling link, and removes a
# link only when its target is not carried by the catalog at all, so no reverse
# could exist. A ``name:`` link out of the family is a legitimate cross-family
# reference and is left alone — the step reconciles the sibling relation, it does
# not police the rest of a member's references.

#: Prefix of a link that names a standard name. Every other link (a ``dd:``
#: path, an external URL) carries no standard-name target and is left alone.
_NAME_LINK_PREFIX = "name:"


def _name_link_target(link: Any) -> str:
    """The standard-name id a ``name:`` link points at, else ``""``."""
    if isinstance(link, str) and link.startswith(_NAME_LINK_PREFIX):
        return link[len(_NAME_LINK_PREFIX) :]
    return ""


def _drop_doc_reference(documentation: str, target: str) -> str:
    """Remove the reference to *target* from documentation prose.

    The derived link index is rebuilt from the documentation, so dropping a
    link from the index alone would let the next derivation restore it. The
    spellings the reference parser reads are found by that parser's own owner
    — :func:`doc_links.find_name_links` for a markdown link and
    :func:`doc_links.find_name_references` for a bare ``name:`` reference — so
    no local pattern can drift from the syntax the derivation accepts. A
    markdown link keeps its visible text.
    """
    if not documentation:
        return documentation or ""
    links = [link for link in find_name_links(documentation) if link.target == target]
    for link in reversed(links):
        documentation = (
            documentation[: link.start] + link.label + documentation[link.end :]
        )
    inline = [
        reference
        for reference in find_name_references(documentation)
        if reference.syntax == "inline_name" and reference.name == target
    ]
    for reference in reversed(inline):
        documentation = (
            documentation[: reference.start] + documentation[reference.end :]
        )
    return documentation


def reconcile_family_reverse_links(
    gc: Any,
    member_ids: list[str],
    *,
    write_fn: Any | None = None,
    refuse_protected: bool = False,
) -> dict[str, Any]:
    """Make every family sibling link reciprocal across a converged family.

    Reads the members named by *member_ids* and writes each member's ``links``
    so that:

    * a ``name:`` link to a family sibling is kept, and that sibling links
      back — a sibling the member's links reach gains the reverse when it lacks
      it. A reverse is added only between family siblings; a reference to a name
      outside the family is never made reciprocal, because the two are not
      siblings and no such relation is being asserted;
    * a ``name:`` link whose target no standard name in the catalog carries is
      dropped, from the derived index and from the documentation prose, because
      no reverse could ever exist for it;
    * a ``name:`` link to a live name outside the family is carried through
      unchanged. It is a legitimate cross-family reference, not a broken
      sibling link, and dropping it would destroy a valid reference the member
      wrote;
    * every non-``name:`` link — a ``dd:`` path, an external URL — is carried
      through unchanged, kept as ``links`` and untouched in the documentation.

    A link is added to the member's documentation as well as to the derived
    ``links``, since the index is derived from the prose and an index edited
    alone would lose the addition at the next derivation.

    A member whose ``name_stage`` is ``approved`` is catalog-edited: its
    editorial fields belong to the catalog, so the reconciliation computed for
    it is not written, and its ``description``, ``documentation`` and ``links``
    stay exactly as the catalog left them. Such a member's row is withheld from
    the writer entirely, because the writer does not *skip* a protected row — it
    strips that row's fields and writes each now-absent field as empty, which
    would clear the very documentation protection exists to keep. The withheld
    ids are returned under ``"protected"`` so the caller can report them rather
    than let a partial write look complete.

    With ``refuse_protected``, a protected row in the pending batch prevents
    writes to every member. Finalization uses this after its scope preflight.

    The write goes through the ``graph_ops`` function that already persists
    these fields — ``write_enrichment_results`` — called from here rather than
    re-issued; ``write_fn`` injects that writer for a test. Nothing is written
    when the family's sibling links are already symmetric, so the step is
    idempotent.

    Returns ``{"members", "added", "removed", "written", "protected"}``.
    """
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        write_enrichment_results,
    )

    members = sorted({mid for mid in member_ids if mid})
    if not members:
        return {"members": 0, "added": 0, "removed": 0, "written": 0}

    rows = (
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IN $ids
            RETURN sn.id AS id, sn.links AS links,
                   sn.documentation AS documentation,
                   sn.description AS description,
                   sn.validity_domain AS validity_domain,
                   sn.constraints AS constraints
            """,
            ids=members,
        )
        or []
    )
    state = {row["id"]: dict(row) for row in rows if row.get("id")}
    if not state:
        return {"members": 0, "added": 0, "removed": 0, "written": 0}
    present = sorted(state)

    # A name: link is dead only when no StandardName anywhere carries it, so
    # deadness resolves against the catalog rather than against the family. The
    # read names its row set: the members plus every target they reference.
    referenced = sorted(
        {
            target
            for row in state.values()
            for link in (row.get("links") or [])
            if (target := _name_link_target(link))
        }
    )
    live_ids = {
        row["id"]
        for row in (
            gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.id IN $ids
                RETURN sn.id AS id
                """,
                ids=sorted(set(present) | set(referenced)),
            )
            or []
        )
        if row.get("id")
    }

    sibling = set(present)
    linked = {
        mid: {
            target
            for link in (state[mid].get("links") or [])
            if (target := _name_link_target(link)) in sibling
        }
        for mid in present
    }
    inbound: dict[str, set[str]] = {mid: set() for mid in present}
    for source, targets in linked.items():
        for target in targets:
            inbound[target].add(source)

    # The counts are accumulated per member and summed over the rows the writer
    # is actually given, because a member whose row is withheld is not written
    # and counting its additions and removals would report a change that did
    # not land.
    batch: list[dict[str, Any]] = []
    row_added: dict[str, int] = {}
    row_removed: dict[str, int] = {}
    for mid in present:
        row = state[mid]
        original_links = list(row.get("links") or [])
        original_docs = row.get("documentation") or ""
        kept: list[str] = []
        documentation = original_docs
        removed_here = 0
        for link in original_links:
            target = _name_link_target(link)
            if not target:
                kept.append(link)
                continue
            if target in live_ids:
                if link not in kept:
                    kept.append(link)
                continue
            removed_here += 1
            documentation = _drop_doc_reference(documentation, target)

        additions = [
            source
            for source in sorted(inbound[mid])
            if f"{_NAME_LINK_PREFIX}{source}" not in kept
        ]
        if not additions and kept == original_links and documentation == original_docs:
            continue
        for source in additions:
            kept.append(f"{_NAME_LINK_PREFIX}{source}")
        row_added[mid] = len(additions)
        row_removed[mid] = removed_here
        if additions:
            refs = ", ".join(f"[{s}](name:{s})" for s in additions)
            documentation = f"{documentation.rstrip()}\n\nRelated quantities: {refs}."
        batch.append(
            {
                "id": mid,
                "description": row.get("description"),
                "documentation": documentation,
                "links": kept,
                "validity_domain": row.get("validity_domain"),
                "constraints": row.get("constraints"),
            }
        )

    # Deliver the result through the writer, but never hand it a protected row.
    # filter_protected does not skip a catalog-edited name: it strips that
    # name's editorial fields and returns the stripped row, and the writer turns
    # each absent field into an empty one (sn.documentation = "", sn.links =
    # null). Passing the row through would therefore clear the documentation and
    # links protection exists to keep. The protected rows are dropped before the
    # writer sees them, and their ids returned, so a partially written family
    # cannot read as a whole one.
    protected: list[str] = []
    written = 0
    writable: list[dict[str, Any]] = []
    if batch:
        from imas_codex.standard_names.protection import filter_protected

        _, protected = filter_protected(batch)
        if not (refuse_protected and protected):
            withheld = set(protected)
            writable = [row for row in batch if row["id"] not in withheld]
            written = int((write_fn or write_enrichment_results)(writable) or 0)
    return {
        "members": len(present),
        "added": sum(row_added.get(row["id"], 0) for row in writable),
        "removed": sum(row_removed.get(row["id"], 0) for row in writable),
        "written": written,
        "protected": sorted(protected),
    }


_NAME_STAGE_QUERY = """
UNWIND $ids AS sid
MATCH (sn:StandardName {id: sid})
RETURN sn.id AS id, coalesce(sn.name_stage, '') AS name_stage
"""


def _name_stages(gc: Any, ids: Sequence[str]) -> dict[str, str]:
    """Return the stored name stage of each live name in *ids*.

    A family's member rows carry the documentation columns but not the name
    stage, and the stage decides whether the documentation pools can claim a
    member at all, so the planner reads it once for the whole roster.
    """
    if not ids:
        return {}
    rows = gc.query(_NAME_STAGE_QUERY, ids=list(ids)) or []
    return {str(row.get("id")): str(row.get("name_stage") or "") for row in rows}


def _split_name_accepted(
    member_ids: Sequence[str], stages: dict[str, str]
) -> tuple[list[str], list[dict[str, str]]]:
    """Split a family roster into the docs-eligible ids and the skipped ones.

    A docs-only campaign never writes the name axis, so a member that is not
    name-accepted can never be claimed by the documentation pools: including it
    would collect nothing from the drain and then report the untouched stage as
    drift. It is skipped and reported with the stage it carries, so the family's
    outcome names it instead of blaming the run for it.
    """
    eligible: list[str] = []
    skipped: list[dict[str, str]] = []
    for mid in member_ids:
        # A stage the read could not answer for is not evidence of anything, so
        # the member is planned as before rather than silently dropped.
        stage = stages.get(mid)
        if stage is None or stage == "" or stage == "accepted":
            eligible.append(mid)
        else:
            skipped.append({"id": mid, "name_stage": stage})
    return eligible, skipped


def plan_family_applies(
    gc: Any | None = None,
    *,
    seeds: list[str] | None = None,
    min_size: int = 1,
) -> list[dict[str, Any]]:
    """Read-only plan of the anchor-gated applies over the families in scope.

    One plan per pattern family: its anchor, its members, and the batch
    sequence an apply would run — the anchor alone, then the other members.
    The anchor is the family's recorded choice when an earlier apply started
    one and the family is not yet stamped harmonized; otherwise it is
    ``select_anchor``'s live ranking. A family whose members already carry the
    current ``harmonized_group_signature`` is ``harmonized`` and plans no
    batch; a family with no selectable anchor, or whose anchor is not
    name-accepted, is ``deferred``. Members outside ``accepted`` are listed
    under ``skipped`` with their name stage and never enter a batch. Writes
    nothing.
    """
    from imas_codex.standard_names.campaign import anchor_gated_batches

    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        families = fetch_pattern_families(resolved_gc, seeds=seeds, min_size=min_size)
        roster = sorted(
            {
                m.get("id")
                for family in families
                for m in family["members"]
                if m.get("id")
            }
        )
        stages = _name_stages(resolved_gc, roster)
    finally:
        _close_if_owned(resolved_gc, own_gc)

    plans: list[dict[str, Any]] = []
    for family in families:
        members = family["members"]
        member_ids = sorted(m.get("id") for m in members if m.get("id"))
        signature = group_signature(members)
        eligible_ids, skipped = _split_name_accepted(member_ids, stages)
        stored = {
            m.get("harmonized_group_signature")
            for m in members
            if m.get("harmonized_group_signature") is not None
        }
        # A family not yet stamped harmonized reuses the anchor its run
        # recorded, even after that member's reset left it mid-pipeline and
        # another accepted member outscores it. A stamped family selects
        # normally; only the stamp releases the pin.
        anchor = resolve_family_anchor(
            members, member_ids=member_ids, signature=signature
        )
        if member_ids and stored == {signature}:
            decision, batches = "harmonized", []
        elif anchor is None or anchor not in eligible_ids:
            decision, batches = "deferred", []
        else:
            rest = [mid for mid in eligible_ids if mid != anchor]
            decision, batches = "apply", anchor_gated_batches(anchor, rest)
        plans.append(
            {
                "physical_base": family["physical_base"],
                "invariant": list(family["invariant"]),
                "anchor": anchor,
                "members": member_ids,
                "docs_eligible": eligible_ids,
                "skipped": skipped,
                "n": len(member_ids),
                "group_signature": signature,
                "harmonized": decision == "harmonized",
                "decision": decision,
                "batches": batches,
                "planned_calls": sum(len(b) for b in batches),
                # The pre-reset documentation-axis score per member, captured
                # from the same read that planned the family. The apply
                # compares each member's post-drain score against it.
                "member_scores": {
                    m.get("id"): m.get("reviewer_score_docs")
                    for m in members
                    if m.get("id")
                },
            }
        )
    return plans


def docs_score_regressions(
    before: dict[str, float | None],
    after: dict[str, float | None],
    *,
    threshold: float = DEFAULT_MIN_SCORE,
) -> dict[str, list[dict[str, Any]]]:
    """Compare documentation-axis scores before and after an apply.

    ``before``/``after`` each map a member's id to its ``reviewer_score_docs``
    (``None`` when the name carries no stored score — never reviewed). Returns
    two name-keyed lists, each entry ``{"id", "before", "after"}``:

    * ``below_threshold`` — a member whose after score is present and below
      *threshold* (the acceptance score the review pipeline accepts on);
    * ``fell`` — a member whose after score is present and lower than a
      present before score.

    A member is reported only on an after score it actually carries: a name
    whose rerun did not reach review is neither below threshold nor a fall, so
    the two lists answer "did making them agree make any worse" without
    inventing a score for a name that has none.
    """
    below: list[dict[str, Any]] = []
    fell: list[dict[str, Any]] = []
    for member_id in sorted(set(before) | set(after)):
        after_score = after.get(member_id)
        if after_score is None:
            continue
        before_score = before.get(member_id)
        if after_score < threshold:
            below.append(
                {"id": member_id, "before": before_score, "after": after_score}
            )
        if before_score is not None and after_score < before_score:
            fell.append({"id": member_id, "before": before_score, "after": after_score})
    return {"below_threshold": below, "fell": fell}


def _family_docs_scores(gc: Any, member_ids: list[str]) -> dict[str, float | None]:
    """Post-drain documentation-axis score per member of the family.

    Re-reads the family through the same pattern-family query the plan uses,
    so the comparison adds no new query shape. A member absent from the read
    is omitted, which :func:`docs_score_regressions` treats as an unscored
    name rather than a fall.
    """
    if not member_ids:
        return {}
    families = fetch_pattern_families(gc, seeds=list(member_ids[:1]))
    if not families:
        return {}
    return {
        m.get("id"): m.get("reviewer_score_docs")
        for m in families[0]["members"]
        if m.get("id")
    }


def _stamp_applied_family(gc: Any, anchor_id: str) -> bool:
    """Re-read a converged family and stamp its current group signature."""
    from imas_codex.standard_names.graph_store.graph_docs_persistence import (
        stamp_harmonized_families,
    )

    families = fetch_pattern_families(gc, seeds=[anchor_id])
    if not families:
        return False
    members = families[0]["members"]
    member_ids = sorted(m.get("id") for m in members if m.get("id"))
    if not member_ids:
        return False
    stamped = bool(
        stamp_harmonized_families(
            [
                {
                    "parent": None,
                    "members": member_ids,
                    "signature": group_signature(members),
                }
            ]
        )
    )
    # The family is now harmonized, so the anchor that carried its run has done
    # its work: release it and let the next apply select normally.
    if stamped:
        _clear_family_anchor(gc, member_ids)
    return stamped


class HarmonizeFinalizeError(ValueError):
    """A family cannot be finalized without another documentation pass."""


def _finalize_member_rows(gc: Any, member_ids: list[str]) -> list[dict[str, Any]]:
    """Read the current lifecycle, prose and links of an exact family roster."""
    return list(
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IN $ids
            RETURN sn.id AS id, sn.docs_stage AS docs_stage,
                   sn.validation_status AS validation_status,
                   sn.description AS description,
                   sn.documentation AS documentation, sn.links AS links,
                   sn.harmonized_group_signature AS harmonized_group_signature
            """,
            ids=member_ids,
        )
        or []
    )


def _missing_family_reverse_links(rows: list[dict[str, Any]]) -> list[str]:
    """Name one-way links between members in the current readback."""
    links = {
        row["id"]: {
            target
            for link in row.get("links") or []
            if (target := _name_link_target(link))
        }
        for row in rows
    }
    return [
        f"{source} -> {target}"
        for source, targets in sorted(links.items())
        for target in sorted(targets & links.keys())
        if source not in links[target]
    ]


def finalize_families(
    gc: Any,
    *,
    seeds: list[str] | None = None,
    min_size: int = 1,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Reconcile and stamp drained, clean families without running a campaign."""
    from imas_codex.standard_names.prose_policy import banned_prose_findings
    from imas_codex.standard_names.protection import recommended_name_protections

    plans = plan_family_applies(gc, seeds=seeds, min_size=min_size)
    if seeds and not plans:
        raise HarmonizeFinalizeError("No live family contains the requested seed")
    member_ids = sorted({mid for plan in plans for mid in plan["members"]})
    protected = recommended_name_protections(gc, member_ids)
    refusals = [
        f"{member_id}: protected ({reason})"
        for member_id, reason in sorted(protected.items())
    ]
    before_rows: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for plan in plans:
        ids = plan["members"]
        rows = _finalize_member_rows(gc, ids)
        before_rows[tuple(ids)] = rows
        by_id = {row["id"]: row for row in rows}
        for member_id in ids:
            row = by_id.get(member_id)
            if row is None:
                refusals.append(f"{member_id}: missing from graph")
                continue
            if row.get("docs_stage") != "accepted":
                refusals.append(
                    f"{member_id}: docs_stage={row.get('docs_stage')!r}, expected 'accepted'"
                )
            if row.get("validation_status") == "quarantined":
                refusals.append(f"{member_id}: validation_status='quarantined'")
            text = f"{row.get('description') or ''}\n{row.get('documentation') or ''}"
            findings = banned_prose_findings(text)
            classes = [name for name, count in findings.items() if count]
            if classes:
                refusals.append(f"{member_id}: banned prose ({', '.join(classes)})")
    if refusals:
        raise HarmonizeFinalizeError("Finalization refused: " + "; ".join(refusals))

    results: list[dict[str, Any]] = []
    for plan in plans:
        result = {"physical_base": plan["physical_base"], "members": plan["members"]}
        if dry_run:
            results.append(
                {
                    **result,
                    "dry_run": True,
                    "stamped": False,
                    "missing_reverse_pairs": len(
                        _missing_family_reverse_links(
                            before_rows[tuple(plan["members"])]
                        )
                    ),
                    "planned_calls": 0,
                }
            )
            continue
        links = reconcile_family_reverse_links(
            gc, plan["members"], refuse_protected=True
        )
        if links.get("protected"):
            raise HarmonizeFinalizeError(
                "Finalization refused: protected members were not reconciled: "
                + ", ".join(links["protected"])
            )
        rows = _finalize_member_rows(gc, plan["members"])
        if {row["id"] for row in rows} != set(plan["members"]):
            raise HarmonizeFinalizeError(
                "Family membership changed during finalization"
            )
        missing = _missing_family_reverse_links(rows)
        if missing:
            raise HarmonizeFinalizeError(
                "Reverse links remain missing: " + ", ".join(missing)
            )
        if not _stamp_applied_family(gc, plan["anchor"] or plan["members"][0]):
            raise HarmonizeFinalizeError("Family stamp was not written")
        stamped_rows = _finalize_member_rows(gc, plan["members"])
        if {row["id"] for row in stamped_rows} != set(plan["members"]):
            raise HarmonizeFinalizeError("Family membership changed after stamping")
        signature = group_signature(stamped_rows)
        unstamped = [
            row["id"]
            for row in stamped_rows
            if row.get("harmonized_group_signature") != signature
        ]
        if unstamped:
            raise HarmonizeFinalizeError(
                "Current family signature missing on: " + ", ".join(unstamped)
            )
        replanned = plan_family_applies(gc, seeds=[plan["members"][0]])
        if (
            len(replanned) != 1
            or replanned[0]["decision"] != "harmonized"
            or replanned[0]["planned_calls"] != 0
        ):
            raise HarmonizeFinalizeError("Stamped family still plans model calls")
        results.append(
            {
                **result,
                "stamped": True,
                "missing_reverse_pairs": 0,
                "planned_calls": 0,
                "reverse_links": links,
            }
        )
    return results


def apply_guard_targets(plans: list[dict[str, Any]]) -> list[str]:
    """The names an anchored apply would rewrite: every ``apply`` family's members.

    The recommended-name guard runs over exactly this set, so both the real
    apply and a dry run that reports it derive the targets here rather than
    restating the selection.
    """
    return sorted(
        {
            mid
            for fam_plan in plans
            if fam_plan["decision"] == "apply"
            for mid in fam_plan["members"]
        }
    )


def preview_recommended_refusals(
    gc: Any,
    plans: list[dict[str, Any]],
    *,
    reviews_dir: Any | None = None,
) -> list[dict[str, str]]:
    """Per plan, the protected members an ``apply`` would refuse, with evidence.

    One entry per plan in order: the ``{member_id: evidence}`` map of that
    plan's members recommended into a catalog request (empty when none). Reads
    the same evidence as :func:`refuse_recommended_members`, in a single query,
    and writes nothing — so a dry run can report the refusal the real apply
    would raise without raising it.
    """
    from imas_codex.standard_names.protection import recommended_name_protections

    protections = recommended_name_protections(
        gc, apply_guard_targets(plans), reviews_dir=reviews_dir
    )
    preview: list[dict[str, str]] = []
    for fam_plan in plans:
        held = {
            mid: protections[mid]
            for mid in fam_plan["members"]
            if fam_plan["decision"] == "apply" and mid in protections
        }
        preview.append(held)
    return preview


def refuse_recommended_members(
    gc: Any,
    member_ids: list[str],
    *,
    force: str | None = None,
    reviews_dir: Any | None = None,
) -> dict[str, str]:
    """Refuse an apply over names already recommended into a catalog request.

    Returns the per-request evidence when *force* is given (the caller records
    the reason), and raises :class:`RecommendedNameError` naming every
    committed member when it is not. An empty result means nothing is in one
    of the covered classes. Reads the graph and the frozen rosters through
    :func:`~imas_codex.standard_names.protection.recommended_name_protections`.
    """
    from imas_codex.standard_names.protection import (
        RecommendedNameError,
        recommended_name_protections,
    )

    protections = recommended_name_protections(gc, member_ids, reviews_dir=reviews_dir)
    if protections and force is None:
        details = "; ".join(
            f"{name_id} ({reason})" for name_id, reason in sorted(protections.items())
        )
        raise RecommendedNameError(
            "Refused harmonize apply for names recommended into a catalog "
            "pull request: " + details
        )
    return protections


def _record_forced_recommended(
    gc: Any, protections: dict[str, str], reason: str, run_id: str
) -> None:
    """Stamp the override reason on each forced name's internal change log.

    The reason and the evidence it overrode are recorded under the apply's own
    scope run id, so the override is auditable against the run that made it.
    """
    from imas_codex.standard_names.provenance_lifecycle import (
        record_standard_name_change,
    )

    for name_id, evidence in sorted(protections.items()):
        record_standard_name_change(
            gc,
            name_id,
            name_id,
            operation="harmonize_forced_recommended",
            reason=f"{reason} (overrode {evidence})",
            origin="harmonize",
            run_id=run_id,
        )


def _family_budget_drain(
    base_drain: Any,
    ceiling: float,
    *,
    cost_fn: Any | None = None,
) -> Any:
    """Wrap *base_drain* so one family's batches share a single cost cap.

    The runner drains each batch with the budget's ``per_batch_cost_cap``,
    which on its own would hand the anchor batch and the member batch the
    whole cap independently. This wrapper instead gives every batch the
    family's *remaining* budget — ``ceiling`` minus what the earlier batches
    already spent — so the anchor spends what it needs and the members receive
    exactly what it leaves, and the family stops at ``ceiling`` across all
    batches. The drain's measured spend already includes the review that
    follows each regeneration, so the cap covers that too.

    A batch whose drain cannot measure its own spend (returns ``None``) is
    charged ``cost_fn(run_id)`` — the same figure the runner falls back to for
    that batch — so a batch that cannot report its cost still counts against
    the family's remaining budget, and the family cannot exceed ``ceiling`` by
    one.

    ``ceiling`` is the budget's ``campaign_cost_ceiling`` — the same field the
    runner halts on between batches — so there is one family ceiling, not a
    second counter beside it.
    """
    from imas_codex.standard_names.campaign import default_cost

    fallback = cost_fn or default_cost
    spent = 0.0

    def drain(run_id: str, batch_cost_cap: float) -> float | None:
        nonlocal spent
        remaining = max(ceiling - spent, 0.0)
        cost = base_drain(run_id, min(batch_cost_cap, remaining))
        if cost is None:
            cost = fallback(run_id)
        spent += float(cost)
        return cost

    return drain


def apply_families(
    gc: Any | None = None,
    *,
    seeds: list[str] | None = None,
    min_size: int = 1,
    drain_fn: Any,
    plan: list[dict[str, Any]] | None = None,
    budget: Any | None = None,
    thresholds: Any | None = None,
    runner_factory: Any | None = None,
    runner_kwargs: dict[str, Any] | None = None,
    force: str | None = None,
    reviews_dir: Any | None = None,
) -> dict[str, Any]:
    """Run the anchor-gated apply over each family in scope.

    One :class:`~imas_codex.standard_names.campaign.CampaignRunner` per family
    marked ``apply``: the anchor drained as the first batch, the remaining
    members as the second. The single-member anchor batch accepts at rate 1.0
    when the anchor reaches documentation acceptance and 0.0 otherwise, so the
    runner's convergence gate halts the family before the member batch is ever
    marked or drained when the anchor was not accepted — the members are then
    never reset and no DocsRevision snapshot is taken. A family that converges
    is stamped with its new ``harmonized_group_signature`` so a repeated apply
    over the unchanged scope plans nothing.

    ``drain_fn`` and the entries of ``runner_kwargs`` are forwarded to the
    runner unchanged, so the orchestration is drivable against a fixture graph.
    """
    from imas_codex.standard_names.campaign import (
        CampaignBudget,
        CampaignRunner,
        CampaignSpec,
        ConvergenceThresholds,
        default_cost,
    )

    resolved_gc, own_gc = _resolve_gc(gc)
    budget = budget or CampaignBudget()
    thresholds = thresholds or ConvergenceThresholds()
    plans = (
        plan
        if plan is not None
        else plan_family_applies(
            resolved_gc,
            seeds=seeds,
            min_size=min_size,
        )
    )
    spec = CampaignSpec(raw="harmonize-pattern-family")
    kwargs = dict(runner_kwargs or {})

    # The recommended-name guard runs BEFORE any family is marked: a reset
    # rewrites text a reviewer was asked to read, so the whole apply is refused
    # (or the override reason recorded) before the first batch touches the graph.
    targets = apply_guard_targets(plans)
    guards: dict[str, str] = {}
    override_run_id: str | None = None
    try:
        guards = refuse_recommended_members(
            resolved_gc, targets, force=force, reviews_dir=reviews_dir
        )
        if guards and force is not None:
            import uuid

            override_run_id = str(uuid.uuid4())
            _record_forced_recommended(resolved_gc, guards, force, override_run_id)
    except Exception:
        _close_if_owned(resolved_gc, own_gc)
        raise

    applied: list[dict[str, Any]] = []
    try:
        for fam_plan in plans:
            if fam_plan["decision"] != "apply":
                applied.append({**fam_plan, "applied": False, "stamped": False})
                continue
            # Anchor the family before its first batch resets anything: the
            # recorded anchor is what a replan mid-run reuses, so it must
            # precede the run rather than follow it. Released on the stamp.
            _record_family_anchor(resolved_gc, fam_plan["members"], fam_plan["anchor"])
            # The anchor batch and the member batch draw on one family cap:
            # each batch is drained against what the family has left, not the
            # whole cap again.
            family_drain = _family_budget_drain(
                drain_fn,
                budget.campaign_cost_ceiling,
                cost_fn=kwargs.get("cost_fn") or default_cost,
            )
            runner = (
                runner_factory(spec)
                if runner_factory is not None
                else CampaignRunner(spec, budget, thresholds)
            )
            result = runner.run(
                gc=resolved_gc,
                target_ids=fam_plan["members"],
                drain_fn=family_drain,
                batch_plan=fam_plan["batches"],
                **kwargs,
            )
            stamped = False
            reverse_links: dict[str, Any] | None = None
            if not result.halted:
                # Convergence is the first moment the family's member links are
                # final for this run, and the last moment before the signature
                # is stamped over their documentation — so the reverse-link
                # reconciliation runs here and the stamp covers its result.
                reverse_links = reconcile_family_reverse_links(
                    resolved_gc, fam_plan["members"]
                )
                if reverse_links.get("protected"):
                    logger.warning(
                        "Family '%s': %d member(s) are catalog-edited, so their "
                        "reconciled links were not written: %s",
                        fam_plan["anchor"],
                        len(reverse_links["protected"]),
                        ", ".join(reverse_links["protected"]),
                    )
                stamped = _stamp_applied_family(resolved_gc, fam_plan["anchor"])
            after_scores = _family_docs_scores(resolved_gc, fam_plan["members"])
            regressions = docs_score_regressions(
                fam_plan.get("member_scores") or {}, after_scores
            )
            applied.append(
                {
                    **fam_plan,
                    "applied": True,
                    "result": result,
                    "stamped": stamped,
                    "reverse_links": reverse_links,
                    "scores_after": after_scores,
                    "score_regressions": regressions,
                }
            )
    finally:
        _close_if_owned(resolved_gc, own_gc)
    return {
        "plans": plans,
        "applied": applied,
        "recommended_guards": guards,
        "force_reason": force if guards else None,
        "force_run_id": override_run_id,
    }


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
    "concept_member_ids",
    "report_family_rows",
    "measure_doc_call_cost",
    "DocCallCostEstimate",
    "plan_family_applies",
    "apply_families",
    "finalize_families",
    "HarmonizeFinalizeError",
    "reconcile_family_reverse_links",
    "docs_score_regressions",
    "apply_guard_targets",
    "preview_recommended_refusals",
    "refuse_recommended_members",
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
    from imas_codex.standard_names.graph_store.graph_docs_persistence import (
        stamp_harmonized_families,
    )

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

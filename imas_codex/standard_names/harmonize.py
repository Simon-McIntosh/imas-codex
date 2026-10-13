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
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

# The family-enumeration query is owned by families.py; re-exported so the
# worklist test can compare the query build_worklist issues against it.
from imas_codex.standard_names.families import (
    DOCS_STAGE_ACCEPTED,
    _pattern_family_rows_query as _pattern_family_rows_query,
    fetch_pattern_families,
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
#: a target is dead only when no ``StandardName`` anywhere carries it. The
#: report's scope stays the scope of what is reported. Bounded, because the
#: read names its row set (every live name id).
_NAME_IDS_QUERY = "MATCH (sn:StandardName) RETURN sn.id AS id LIMIT 200000"


def _known_name_ids(gc: Any) -> set[str]:
    """Every ``StandardName`` id the graph holds, for dead-link resolution."""
    rows = gc.query(_NAME_IDS_QUERY) or []
    return {row["id"] for row in rows if row.get("id")}


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
    known_ids = _known_name_ids(gc) if gc is not None else None

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
                "anchor": select_anchor(None, None, None, members),
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


# ── Pattern-family apply: the anchor-gated campaign ─────────────────────────
# A pattern family has no parent node, so its idempotency mark lives on the
# members themselves (``harmonized_group_signature``). The apply is one
# CampaignRunner per family with an explicit two-batch plan: the anchor alone,
# then the remaining members. See ``plan_family_applies`` / ``apply_families``.

#: Rough per-name estimate for the dry-run cost figure. The pool drain reports
#: its own measured spend; this only gives ``--dry-run`` a number before any
#: call is placed.
ESTIMATED_DOC_CALL_COST_USD = 0.05


def plan_family_applies(
    gc: Any | None = None,
    *,
    seeds: list[str] | None = None,
    min_size: int = 1,
) -> list[dict[str, Any]]:
    """Read-only plan of the anchor-gated applies over the families in scope.

    One plan per pattern family: its anchor (``select_anchor``), its members,
    and the batch sequence an apply would run — the anchor alone, then the
    other members. A family whose members already carry the current
    ``harmonized_group_signature`` is ``harmonized`` and plans no batch; a
    family with no selectable anchor is ``deferred``. Writes nothing.
    """
    from imas_codex.standard_names.campaign import anchor_gated_batches

    resolved_gc, own_gc = _resolve_gc(gc)
    try:
        families = fetch_pattern_families(resolved_gc, seeds=seeds, min_size=min_size)
    finally:
        _close_if_owned(resolved_gc, own_gc)

    plans: list[dict[str, Any]] = []
    for family in families:
        members = family["members"]
        member_ids = sorted(m.get("id") for m in members if m.get("id"))
        signature = group_signature(members)
        stored = {
            m.get("harmonized_group_signature")
            for m in members
            if m.get("harmonized_group_signature") is not None
        }
        anchor = select_anchor(None, None, None, members)
        if member_ids and stored == {signature}:
            decision, batches = "harmonized", []
        elif anchor is None:
            decision, batches = "deferred", []
        else:
            rest = [mid for mid in member_ids if mid != anchor]
            decision, batches = "apply", anchor_gated_batches(anchor, rest)
        plans.append(
            {
                "physical_base": family["physical_base"],
                "invariant": list(family["invariant"]),
                "anchor": anchor,
                "members": member_ids,
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
    from imas_codex.standard_names.graph_ops import stamp_harmonized_families

    families = fetch_pattern_families(gc, seeds=[anchor_id])
    if not families:
        return False
    members = families[0]["members"]
    member_ids = sorted(m.get("id") for m in members if m.get("id"))
    if not member_ids:
        return False
    return bool(
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
    )

    resolved_gc, own_gc = _resolve_gc(gc)
    budget = budget or CampaignBudget()
    thresholds = thresholds or ConvergenceThresholds()
    plans = (
        plan
        if plan is not None
        else plan_family_applies(resolved_gc, seeds=seeds, min_size=min_size)
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
            runner = (
                runner_factory(spec)
                if runner_factory is not None
                else CampaignRunner(spec, budget, thresholds)
            )
            result = runner.run(
                gc=resolved_gc,
                target_ids=fam_plan["members"],
                drain_fn=drain_fn,
                batch_plan=fam_plan["batches"],
                **kwargs,
            )
            stamped = False
            if not result.halted:
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
    "ESTIMATED_DOC_CALL_COST_USD",
    "plan_family_applies",
    "apply_families",
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

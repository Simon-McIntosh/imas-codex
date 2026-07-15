"""Rename-cascade machinery for parent → descendant propagation.

When a parent StandardName is renamed (e.g. ``elongation`` →
``elongation_of_closed_flux_surface``), descendants that derive their
identity from the parent name through grammar structure must follow
(``upper_elongation`` → ``upper_elongation_of_closed_flux_surface``).
The information needed for this cascade is **already encoded** on
``HAS_PARENT`` edges — the ``operator``, ``operator_kind``, ``role``,
``separator``, ``axis`` and ``shape`` properties record exactly how the
child name relates to the parent.  This module provides the operation.

Cascade rules by ``operator_kind`` (plan D10):

==================  =========== ============================================
``operator_kind``   Cascades?   Renaming rule
==================  =========== ============================================
``qualifier``        Yes         ``{operator}_{new_parent_name}``
``locus``            Yes         ``{new_parent_name}_{relation}_{locus}``
``unary_prefix``     Yes         ``{operator}_{new_parent_name}``
``unary_postfix``    Yes         ``{new_parent_name}_{operator}``
``binary`` (role a)  Yes         ``{op}_of_{new_parent_name}_{sep}_{other}``
``binary`` (role b)  Yes         ``{op}_of_{other}_{sep}_{new_parent_name}``
``projection``       **No**      Component children have independent identity
``coordinate``       **No**      Same as projection — axis renders independently
==================  =========== ============================================

The operation is all-or-nothing: every candidate is parsed and composed
through ISN's grammar (``parse_standard_name`` round-trip) before any
write touches the graph.  If any candidate fails validation, the entire
cascade aborts with no graph state mutated.

Safety knobs (mirror ``sn run --reset-to`` / ``sn prune``):

- ``override_edits`` is required to rename any descendant with
  ``origin = 'catalog_edit'`` — the catalog is the authoritative editor
  for those entries and silently rewriting them is destructive.
- ``include_accepted`` is required to rename any descendant with
  ``name_stage = 'accepted'`` — these names are committed catalog
  entries and the operator should opt-in explicitly.

Audit log: every rename writes one line to
``~/.local/share/imas-codex/logs/parents_rename.log`` for traceability.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from imas_standard_names.grammar import parser as _isn_parser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeResult:
    """Outcome of a :func:`rename_cascade` invocation.

    Attributes
    ----------
    old_name:
        The root parent id that was renamed.
    new_name:
        The replacement parent id.
    renamed:
        List of ``{"from": old_id, "to": new_id}`` entries, one per
        descendant whose id changed (root included).  In ``dry_run``
        mode this is the *planned* rename list.
    skipped:
        Descendants intentionally left untouched (projection edges,
        independent-identity children).  Each entry is
        ``{"name": id, "reason": str}``.
    conflicts:
        Reasons the cascade aborted (or would abort in dry-run mode).
        Empty list means the cascade is safe to apply.
    total_descendants:
        Total number of descendants discovered via ``HAS_PARENT*`` walk
        (regardless of whether they cascade).
    dry_run:
        ``True`` when no write took place.
    """

    old_name: str
    new_name: str
    renamed: list[dict[str, str]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    total_descendants: int = 0
    dry_run: bool = True


# ---------------------------------------------------------------------------
# Edge-rule dispatch (pure logic — no graph)
# ---------------------------------------------------------------------------


def _cascade_target_name(
    edge_props: dict[str, Any],
    new_parent_name: str,
    other_arg_name: str | None = None,
    locus_relation: str | None = None,
    locus_token: str | None = None,
) -> str | None:
    """Compute the new child id given the edge properties and renamed parent.

    Returns ``None`` for edges that do **not** cascade (projection,
    coordinate).  Returns a string for every operator_kind that does.

    Parameters
    ----------
    edge_props:
        The ``HAS_PARENT`` edge property dict (``operator``,
        ``operator_kind``, optionally ``role``, ``separator``).
    new_parent_name:
        The renamed parent's new id.
    other_arg_name:
        For binary edges, the id of the *other* argument (the one whose
        role differs from the renamed parent's).  Ignored for non-binary
        kinds.
    locus_relation / locus_token:
        Required for ``operator_kind == 'locus'`` edges.  Recovered from
        the original child name's parse — the edge schema stores only
        the locus token under ``operator``; the relation is rebuilt at
        cascade-time from the child's IR.
    """
    op_kind = edge_props.get("operator_kind")
    operator = edge_props.get("operator")

    if op_kind in ("projection", "coordinate"):
        # Projections render axis_inner — the child is independent of
        # the parent's exact name string.  Do not cascade.
        return None

    if op_kind == "qualifier":
        if not operator:
            return None
        return f"{operator}_{new_parent_name}"

    if op_kind == "locus":
        # Locus edges carry the locus token in ``operator``.  The
        # relation (``of`` / ``at`` / ``over``) is not stored on the
        # edge — derivation drops it because it's recoverable from the
        # child name's parse.  We pass it through explicitly here so
        # the caller (which already has the child name) can compute it
        # once and hand it down.
        if not locus_relation or not locus_token:
            return None
        return f"{new_parent_name}_{locus_relation}_{locus_token}"

    if op_kind == "unary_prefix":
        if not operator:
            return None
        return f"{operator}_of_{new_parent_name}"

    if op_kind == "unary_postfix":
        if not operator:
            return None
        return f"{new_parent_name}_{operator}"

    if op_kind == "binary":
        role = edge_props.get("role")
        separator = edge_props.get("separator")
        if not operator or not separator or role not in ("a", "b"):
            return None
        if other_arg_name is None:
            # Cannot rename a binary edge without the partner argument.
            return None
        if role == "a":
            return f"{operator}_of_{new_parent_name}_{separator}_{other_arg_name}"
        # role == "b"
        return f"{operator}_of_{other_arg_name}_{separator}_{new_parent_name}"

    # Unknown kind — leave alone.
    return None


# ---------------------------------------------------------------------------
# ISN round-trip validation
# ---------------------------------------------------------------------------


def _isn_round_trip_ok(name: str) -> tuple[bool, str]:
    """Validate that ``name`` parses and composes back to itself.

    The cascade rules above are pure string concatenation — the only
    way to know whether the result is valid grammar is to feed it
    through ISN's parser and check the round-trip identity.
    """
    try:
        parsed = _isn_parser.parse(name)
    except Exception as exc:
        return False, f"parse failed: {exc.__class__.__name__}: {exc}"
    try:
        composed = _isn_parser.compose(parsed.ir)
    except Exception as exc:
        return False, f"compose failed: {exc.__class__.__name__}: {exc}"
    if composed != name:
        return False, f"round-trip mismatch ({name!r} → {composed!r})"
    return True, "ok"


# ---------------------------------------------------------------------------
# Locus relation recovery (parses the original child name)
# ---------------------------------------------------------------------------


def _recover_locus_parts(child_id: str) -> tuple[str | None, str | None]:
    """Extract (relation, token) from a locus-qualified child id.

    Returns ``(None, None)`` when the child name has no locus or fails
    to parse.  The caller treats that as "cannot cascade this edge."
    """
    try:
        parsed = _isn_parser.parse(child_id)
    except Exception:
        return None, None
    locus = parsed.ir.locus
    if locus is None or not locus.token:
        return None, None
    return str(locus.relation.value), str(locus.token)


def parent_segment_of_child(
    edge_props: dict[str, Any],
    child_name: str,
    other_arg_name: str | None = None,
) -> str | None:
    """Extract the parent-derived substring from a cascade child's own id.

    The inverse of :func:`_cascade_target_name`: given the same edge
    properties and a *child's current id*, recover the substring that
    plays the role of ``new_parent_name`` in the forward formula.  Used
    by the edit engine's shared-base guard to detect when a requested
    leaf rename actually touches the segment owned by the parent (and
    therefore should cascade to siblings rather than desync from them).

    Returns ``None`` when *child_name* does not match the template
    implied by *edge_props* (e.g. the operator itself changed too, or
    the edge kind does not cascade).
    """
    op_kind = edge_props.get("operator_kind")
    operator = edge_props.get("operator")

    if op_kind in ("projection", "coordinate"):
        return None

    if op_kind == "qualifier":
        if not operator:
            return None
        prefix = f"{operator}_"
        if not child_name.startswith(prefix):
            return None
        return child_name[len(prefix) :] or None

    if op_kind == "unary_prefix":
        if not operator:
            return None
        prefix = f"{operator}_of_"
        if not child_name.startswith(prefix):
            return None
        return child_name[len(prefix) :] or None

    if op_kind == "unary_postfix":
        if not operator:
            return None
        suffix = f"_{operator}"
        if not child_name.endswith(suffix):
            return None
        return child_name[: -len(suffix)] or None

    if op_kind == "locus":
        relation, token = _recover_locus_parts(child_name)
        if relation is None or token is None:
            return None
        suffix = f"_{relation}_{token}"
        if not child_name.endswith(suffix):
            return None
        return child_name[: -len(suffix)] or None

    if op_kind == "binary":
        role = edge_props.get("role")
        separator = edge_props.get("separator")
        if (
            not operator
            or not separator
            or role not in ("a", "b")
            or not other_arg_name
        ):
            return None
        if role == "a":
            prefix = f"{operator}_of_"
            suffix = f"_{separator}_{other_arg_name}"
            if not (child_name.startswith(prefix) and child_name.endswith(suffix)):
                return None
            middle = child_name[len(prefix) : -len(suffix)]
            return middle or None
        # role == "b"
        prefix = f"{operator}_of_{other_arg_name}_{separator}_"
        if not child_name.startswith(prefix):
            return None
        return child_name[len(prefix) :] or None

    return None


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


def _audit_log_path() -> Path:
    """Path to the rename audit log.

    Honours ``XDG_DATA_HOME`` (falls back to ``~/.local/share``) so the
    log lands in the same directory as ``sn run``'s rotating logs.  The
    directory is created if missing.
    """
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        root = Path(base)
    else:
        root = Path.home() / ".local" / "share"
    log_dir = root / "imas-codex" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "parents_rename.log"


def _write_audit_lines(
    old_name: str,
    new_name: str,
    renames: list[dict[str, str]],
    *,
    dry_run: bool,
    log_path: Path | None = None,
) -> None:
    """Append one audit line per rename to the rotating log file.

    Format::

        <iso8601> mode=<commit|dry-run> root=<old>→<new> from=<x> to=<y>
    """
    if not renames:
        return
    path = log_path or _audit_log_path()
    ts = datetime.now(UTC).isoformat()
    mode = "dry-run" if dry_run else "commit"
    try:
        with path.open("a", encoding="utf-8") as fh:
            for r in renames:
                fh.write(
                    f"{ts} mode={mode} root={old_name}->{new_name} "
                    f"from={r['from']} to={r['to']}\n"
                )
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("rename_cascade audit log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Topology probe (testable via mocks)
# ---------------------------------------------------------------------------


class _GraphProbe(Protocol):
    """Minimal graph-client interface needed by ``rename_cascade``."""

    def query(
        self, cypher: str, **params: Any
    ) -> list[dict[str, Any]]: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Shared descendant walk + per-edge resolution (used by both entry points)
# ---------------------------------------------------------------------------


def _walk_and_resolve_cascade(
    gc: _GraphProbe,
    root_id: str,
    root_new_name: str,
    *,
    override_edits: bool,
    include_accepted: bool,
) -> tuple[dict[str, str], list[dict[str, str]], list[str], int]:
    """Walk the ``HAS_PARENT*`` subtree rooted at *root_id* and resolve
    the cascade rename for every descendant.

    Pure topology + rule dispatch — no root-level validation (existence,
    ISN round-trip, collision) and no write.  Shared by :func:`rename_cascade`
    (root itself is being renamed) and :func:`cascade_descendants_of` (root
    has already been renamed; only its descendants remain to resolve).

    Returns ``(rename_plan, skipped, conflicts, total_descendants)`` where
    ``rename_plan`` is seeded with ``{root_id: root_new_name}``.
    """
    # Walk descendants depth-first via HAS_PARENT*.  We need the per-edge
    # properties for each direct child of every node in the subtree.  The
    # DB walks edges; we walk the result rows.
    rows = list(
        gc.query(
            """
            MATCH (parent:StandardName {id: $old})
            OPTIONAL MATCH path = (parent)<-[:HAS_PARENT*1..]-(d:StandardName)
            WITH parent, d, path
            WHERE d IS NOT NULL
            WITH DISTINCT d
            RETURN d.id AS id,
                   d.origin AS origin,
                   d.name_stage AS name_stage
            """,
            old=root_id,
        )
    )
    descendants_meta: dict[str, dict[str, Any]] = {
        r["id"]: {
            "origin": r.get("origin"),
            "name_stage": r.get("name_stage"),
        }
        for r in rows
        if r.get("id")
    }
    total_descendants = len(descendants_meta)

    # For each direct parent-child relationship in the subtree, pull the
    # edge properties so we can dispatch the cascade rule.
    edge_rows = list(
        gc.query(
            """
            MATCH (parent:StandardName {id: $old})
            OPTIONAL MATCH (child)-[r:HAS_PARENT]->(target)
            WHERE (target = parent) OR EXISTS {
                MATCH (target)-[:HAS_PARENT*]->(parent)
            }
            RETURN child.id AS child_id,
                   target.id AS target_id,
                   r.operator AS operator,
                   r.operator_kind AS operator_kind,
                   r.role AS role,
                   r.separator AS separator,
                   r.axis AS axis,
                   r.shape AS shape
            """,
            old=root_id,
        )
    )

    # Group edges by child for processing.  A child may have multiple
    # incoming HAS_PARENT edges to different siblings in the subtree
    # (e.g. binary operators with two parents).
    edges_by_child: dict[str, list[dict[str, Any]]] = {}
    for er in edge_rows:
        cid = er.get("child_id")
        tid = er.get("target_id")
        if not cid or not tid:
            continue
        edges_by_child.setdefault(cid, []).append(er)

    # Plan per-descendant renames.
    rename_plan: dict[str, str] = {root_id: root_new_name}
    skipped: list[dict[str, str]] = []
    conflicts: list[str] = []

    # Iterate descendants in topological-ish order (depth-first via the
    # graph walk's row order is sufficient — every child references its
    # parent by id, so we just need to know the parent's new name when
    # we compute the child's).  Multiple passes ensure transitive
    # closure: each pass propagates any newly-renamed parent into its
    # direct children.
    pending: set[str] = set(descendants_meta.keys())
    for _safety_iter in range(max(2, total_descendants + 2)):
        progress = False
        for child_id in sorted(pending):
            edges = edges_by_child.get(child_id, [])
            if not edges:
                # No incoming edge within the subtree — independent name.
                skipped.append(
                    {"name": child_id, "reason": "no HAS_PARENT edge in subtree"}
                )
                pending.discard(child_id)
                progress = True
                continue

            # Find the single anchor edge — the one whose target is in
            # the rename_plan (i.e. parent already known to be renamed).
            anchor: dict[str, Any] | None = None
            for er in edges:
                tid = er.get("target_id")
                if tid in rename_plan:
                    anchor = er
                    break
            if anchor is None:
                # No parent renamed yet — skip this round.
                continue

            target_id = anchor.get("target_id")
            new_parent_name = rename_plan[target_id]
            op_kind = anchor.get("operator_kind")

            # Projections don't cascade.
            if op_kind in ("projection", "coordinate"):
                skipped.append(
                    {
                        "name": child_id,
                        "reason": f"operator_kind={op_kind} has independent identity",
                    }
                )
                pending.discard(child_id)
                progress = True
                continue

            # Binary edges need the *other* argument's id.  In the
            # general case the other arg is another StandardName in the
            # graph, and may itself be a descendant being renamed.  We
            # look it up via edges_by_child as well: the same child has
            # two HAS_PARENT edges, one per arg role.
            other_arg_name: str | None = None
            if op_kind == "binary":
                role = anchor.get("role")
                other_role = "b" if role == "a" else "a"
                for er in edges:
                    if (
                        er.get("operator_kind") == "binary"
                        and er.get("operator") == anchor.get("operator")
                        and er.get("role") == other_role
                    ):
                        other_arg_name = er.get("target_id")
                        # Substitute renamed id if applicable.
                        if other_arg_name in rename_plan:
                            other_arg_name = rename_plan[other_arg_name]
                        break
                if other_arg_name is None:
                    conflicts.append(
                        f"binary edge for {child_id!r}: cannot locate the "
                        f"role={other_role!r} partner argument"
                    )
                    pending.discard(child_id)
                    progress = True
                    continue

            # Locus edges need to recover the relation token (``of`` /
            # ``at``) and locus token from the *original* child id.
            locus_relation: str | None = None
            locus_token: str | None = None
            if op_kind == "locus":
                locus_relation, locus_token = _recover_locus_parts(child_id)
                if locus_relation is None or locus_token is None:
                    conflicts.append(
                        f"locus edge for {child_id!r}: cannot recover locus "
                        "(relation, token) from child id parse"
                    )
                    pending.discard(child_id)
                    progress = True
                    continue

            new_child = _cascade_target_name(
                anchor,
                new_parent_name,
                other_arg_name=other_arg_name,
                locus_relation=locus_relation,
                locus_token=locus_token,
            )
            if new_child is None:
                # The dispatcher returned None for a kind we expected to
                # cascade — surface as a conflict so the operator sees it.
                conflicts.append(
                    f"cannot compute cascade for {child_id!r} "
                    f"(operator_kind={op_kind!r}; edge props insufficient)"
                )
                pending.discard(child_id)
                progress = True
                continue

            # ISN round-trip on the candidate.
            rt_ok, rt_reason = _isn_round_trip_ok(new_child)
            if not rt_ok:
                conflicts.append(
                    f"ISN round-trip failed for cascade "
                    f"{child_id!r} → {new_child!r}: {rt_reason}"
                )
                pending.discard(child_id)
                progress = True
                continue

            # Safety checks on the child node itself.
            meta = descendants_meta.get(child_id, {})
            if meta.get("origin") == "catalog_edit" and not override_edits:
                conflicts.append(
                    f"{child_id!r} has origin='catalog_edit'; "
                    "pass override_edits=True to rename anyway"
                )
                pending.discard(child_id)
                progress = True
                continue
            if meta.get("name_stage") == "accepted" and not include_accepted:
                conflicts.append(
                    f"{child_id!r} has name_stage='accepted'; "
                    "pass include_accepted=True to rename anyway"
                )
                pending.discard(child_id)
                progress = True
                continue

            rename_plan[child_id] = new_child
            pending.discard(child_id)
            progress = True

        if not pending:
            break
        if not progress:
            # Pending children can't be resolved (probably orphaned by
            # a broken topology).  Report and stop.
            for orphan in sorted(pending):
                conflicts.append(
                    f"{orphan!r} unreachable in cascade — no parent in plan"
                )
            break

    return rename_plan, skipped, conflicts, total_descendants


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rename_cascade(
    gc: _GraphProbe,
    old_name: str,
    new_name: str,
    *,
    dry_run: bool = True,
    override_edits: bool = False,
    include_accepted: bool = False,
    audit_log_path: Path | None = None,
) -> CascadeResult:
    """Rename ``old_name`` → ``new_name`` and cascade through descendants.

    Walks the inbound ``HAS_PARENT`` subgraph rooted at ``old_name``,
    computes the per-edge cascade rule for each descendant, and either
    reports the plan (``dry_run=True``) or applies it in a single Neo4j
    transaction (``dry_run=False``).

    See module docstring for cascade-rule table and safety semantics.

    Parameters
    ----------
    gc:
        A graph client exposing a ``query(cypher, **params)`` method.
        Tests may inject a stub.
    old_name, new_name:
        Root rename — the parent whose id changes.
    dry_run:
        When ``True`` (default), no write is issued.  Returned
        ``CascadeResult.renamed`` lists planned changes.
    override_edits:
        Required to rename any descendant with ``origin='catalog_edit'``.
        Without this flag, the presence of such descendants is reported
        as a conflict and the cascade refuses to proceed.
    include_accepted:
        Required to rename any descendant with
        ``name_stage='accepted'``.  Mirrors the safety discipline
        of ``sn run --reset-to`` and ``sn prune``.
    audit_log_path:
        Override for the audit log file (tests pass a tempfile).
    """
    if old_name == new_name:
        return CascadeResult(
            old_name=old_name,
            new_name=new_name,
            conflicts=["old_name == new_name (no-op rename)"],
            dry_run=dry_run,
        )

    # ── 1. Validate the root name round-trips (the new name must be
    #       valid ISN grammar; otherwise the whole operation is invalid).
    rt_ok, rt_reason = _isn_round_trip_ok(new_name)
    if not rt_ok:
        return CascadeResult(
            old_name=old_name,
            new_name=new_name,
            conflicts=[f"new root name fails ISN round-trip: {rt_reason}"],
            dry_run=dry_run,
        )

    # ── 2. Confirm the root exists in the graph and gather its safety
    #       attributes.  Also check the destination id isn't already in
    #       use by a different node.
    root_info = list(
        gc.query(
            """
            OPTIONAL MATCH (root:StandardName {id: $old})
            OPTIONAL MATCH (target:StandardName {id: $new})
            RETURN
                CASE WHEN root IS NULL THEN false ELSE true END AS root_exists,
                root.origin AS origin,
                root.name_stage AS name_stage,
                CASE WHEN target IS NULL THEN false ELSE true END AS target_exists
            """,
            old=old_name,
            new=new_name,
        )
    )
    if not root_info or not root_info[0].get("root_exists"):
        return CascadeResult(
            old_name=old_name,
            new_name=new_name,
            conflicts=[f"root StandardName {old_name!r} not found in graph"],
            dry_run=dry_run,
        )
    if root_info[0].get("target_exists"):
        return CascadeResult(
            old_name=old_name,
            new_name=new_name,
            conflicts=[
                f"destination StandardName {new_name!r} already exists "
                "(collision with rename root)"
            ],
            dry_run=dry_run,
        )

    # ── 3-5. Walk the subtree and resolve every descendant's cascade rename.
    rename_plan, skipped, conflicts, total_descendants = _walk_and_resolve_cascade(
        gc,
        old_name,
        new_name,
        override_edits=override_edits,
        include_accepted=include_accepted,
    )

    # ── 6. Collision check — none of the planned new ids may match an
    #       existing StandardName (excluding self-rename of the root).
    new_ids = [v for k, v in rename_plan.items() if v != k]
    if new_ids:
        collision_rows = list(
            gc.query(
                """
                UNWIND $ids AS nid
                OPTIONAL MATCH (sn:StandardName {id: nid})
                WITH nid, sn
                WHERE sn IS NOT NULL
                RETURN nid AS id
                """,
                ids=new_ids,
            )
        )
        for r in collision_rows:
            cid = r.get("id")
            if cid is None:
                continue
            # Allow the root collision case (target_exists was already
            # rejected); other in-tree collisions are conflicts.
            if cid in rename_plan.values() and cid not in rename_plan.keys():
                conflicts.append(
                    f"planned new id {cid!r} collides with existing StandardName"
                )

    # ── 7. Materialise the (from → to) list for the result.
    renamed_list: list[dict[str, str]] = []
    for from_id, to_id in rename_plan.items():
        if from_id == to_id:
            continue
        renamed_list.append({"from": from_id, "to": to_id})

    # If anything went wrong, do not write — return the diagnostic.
    if conflicts:
        return CascadeResult(
            old_name=old_name,
            new_name=new_name,
            renamed=renamed_list,
            skipped=skipped,
            conflicts=conflicts,
            total_descendants=total_descendants,
            dry_run=dry_run,
        )

    # ── 8. Audit log every rename (even dry-run, so the operator can
    #       inspect what *would* happen).
    _write_audit_lines(
        old_name,
        new_name,
        renamed_list,
        dry_run=dry_run,
        log_path=audit_log_path,
    )

    if dry_run or not renamed_list:
        return CascadeResult(
            old_name=old_name,
            new_name=new_name,
            renamed=renamed_list,
            skipped=skipped,
            conflicts=[],
            total_descendants=total_descendants,
            dry_run=True,
        )

    # ── 9. Apply.  Single transaction via UNWIND; node identity is
    #       preserved across the SET id=... rewrite so the HAS_PARENT
    #       edges follow automatically.
    gc.query(
        """
        UNWIND $renames AS r
        MATCH (sn:StandardName {id: r.from})
        SET sn.id = r.to
        """,
        renames=renamed_list,
    )
    from imas_codex.standard_names.provenance_lifecycle import (
        record_standard_name_change,
        refresh_renamed_source_mirrors,
    )

    refresh_renamed_source_mirrors(gc, renamed_list)
    for rename in renamed_list:
        record_standard_name_change(
            gc,
            rename["from"],
            rename["to"],
            operation="cascade",
        )

    logger.info(
        "rename_cascade applied: root=%s→%s descendants_renamed=%d skipped=%d",
        old_name,
        new_name,
        len(renamed_list),
        len(skipped),
    )

    return CascadeResult(
        old_name=old_name,
        new_name=new_name,
        renamed=renamed_list,
        skipped=skipped,
        conflicts=[],
        total_descendants=total_descendants,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# Post-acceptance descendant cascade (edit engine)
# ---------------------------------------------------------------------------


def cascade_descendants_of(
    gc: _GraphProbe,
    successor_id: str,
    old_root: str,
    new_root: str,
    *,
    dry_run: bool = True,
    override_edits: bool = False,
    include_accepted: bool = False,
    audit_log_path: Path | None = None,
) -> CascadeResult:
    """Apply the descendant cascade after a root rename has already landed.

    Called from ``persist_reviewed_name``'s edit-acceptance hook once a
    rename-mode ``imas-codex sn edit`` proposal with ``edit_scope`` in
    (``family``, ``subtree``) is accepted: :func:`~imas_codex.standard_names
    .graph_ops.persist_refined_name` has already re-pointed the root's
    former children's ``HAS_PARENT`` edges at *successor_id* (whose id is
    already *new_root* — the root rename is the reviewed decision), but the
    descendants themselves still carry ids derived from *old_root*.  This
    walks the *live* subtree rooted at *successor_id* and applies the same
    per-edge cascade rules as :func:`rename_cascade`, without repeating the
    root rename (already committed).

    ``old_root`` is used only for audit-log labelling (``root=old→new``) —
    the descendant walk and rule dispatch are keyed entirely off the live
    topology rooted at *successor_id*, which no longer references
    *old_root* at all.

    Descendants never individually re-enter LLM review — the root rename
    was the reviewed decision; this applies its consequences atomically.
    """
    if new_root != successor_id:
        return CascadeResult(
            old_name=old_root,
            new_name=new_root,
            conflicts=[
                f"successor_id {successor_id!r} does not match new_root {new_root!r}"
            ],
            dry_run=dry_run,
        )

    exists_rows = list(
        gc.query(
            "MATCH (s:StandardName {id: $id}) RETURN count(s) AS n",
            id=successor_id,
        )
    )
    if not exists_rows or not exists_rows[0].get("n"):
        return CascadeResult(
            old_name=old_root,
            new_name=new_root,
            conflicts=[f"successor {successor_id!r} not found in graph"],
            dry_run=dry_run,
        )

    rename_plan, skipped, conflicts, total_descendants = _walk_and_resolve_cascade(
        gc,
        successor_id,
        new_root,
        override_edits=override_edits,
        include_accepted=include_accepted,
    )

    # Collision check — identical to rename_cascade's step 6, excluding the
    # trivial root self-mapping (successor_id == new_root already).
    new_ids = [v for k, v in rename_plan.items() if v != k]
    if new_ids:
        collision_rows = list(
            gc.query(
                """
                UNWIND $ids AS nid
                OPTIONAL MATCH (sn:StandardName {id: nid})
                WITH nid, sn
                WHERE sn IS NOT NULL
                RETURN nid AS id
                """,
                ids=new_ids,
            )
        )
        for r in collision_rows:
            cid = r.get("id")
            if cid is None:
                continue
            if cid in rename_plan.values() and cid not in rename_plan.keys():
                conflicts.append(
                    f"planned new id {cid!r} collides with existing StandardName"
                )

    renamed_list: list[dict[str, str]] = [
        {"from": from_id, "to": to_id}
        for from_id, to_id in rename_plan.items()
        if from_id != to_id
    ]

    if conflicts:
        return CascadeResult(
            old_name=old_root,
            new_name=new_root,
            renamed=renamed_list,
            skipped=skipped,
            conflicts=conflicts,
            total_descendants=total_descendants,
            dry_run=dry_run,
        )

    _write_audit_lines(
        old_root,
        new_root,
        renamed_list,
        dry_run=dry_run,
        log_path=audit_log_path,
    )

    if dry_run or not renamed_list:
        return CascadeResult(
            old_name=old_root,
            new_name=new_root,
            renamed=renamed_list,
            skipped=skipped,
            conflicts=[],
            total_descendants=total_descendants,
            dry_run=True,
        )

    gc.query(
        """
        UNWIND $renames AS r
        MATCH (sn:StandardName {id: r.from})
        SET sn.id = r.to
        """,
        renames=renamed_list,
    )

    logger.info(
        "cascade_descendants_of applied: root=%s (was %s) descendants_renamed=%d "
        "skipped=%d",
        new_root,
        old_root,
        len(renamed_list),
        len(skipped),
    )

    return CascadeResult(
        old_name=old_root,
        new_name=new_root,
        renamed=renamed_list,
        skipped=skipped,
        conflicts=[],
        total_descendants=total_descendants,
        dry_run=False,
    )

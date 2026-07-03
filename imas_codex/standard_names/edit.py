"""The ``sn edit`` engine — steered proposals ride the SN pipeline.

``imas-codex sn edit <sn>`` lets a human or agent attach a proposal to a
``StandardName`` that rides the normal generate → review → score pipeline
instead of hand-editing graph text directly.  Three modes:

- **hint** — a steering direction is injected into generate/refine prompts;
  the pipeline still composes the candidate.  Re-enters at ``generate``.
- **rename** — a full replacement name skips generation and rides straight
  into name review.  Re-enters at ``review_name``.
- **docs** — full replacement documentation skips generation and rides
  straight into docs review.  Re-enters at ``review_docs``.

Locked decisions (see the SN edit-engine plan):

- Edit-steering fields are scalar properties on ``StandardName`` (see
  ``imas_codex.graph.models``: ``EditMode``, ``EditOrigin``, ``EditStatus``,
  ``EditScope``).
- Review receives ONLY the edit reason (intent) — never the proposal
  pre-approved. The reviewer still independently scores the candidate.
- A shared-base leaf edit (renaming a segment a leaf's siblings also carry)
  BLOCKS without explicit ``scope=family``.
- Cascade descendants never individually re-enter LLM review — the ROOT
  rename is the reviewed decision; descendants follow atomically once it is
  accepted (see :func:`imas_codex.standard_names.graph_ops
  .persist_reviewed_name` and :func:`imas_codex.standard_names.cascade
  .cascade_descendants_of`).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_standard_names.grammar import parser as _isn_parser

from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import EditMode, EditOrigin, EditScope, EditStatus
from imas_codex.standard_names.cascade import (
    _isn_round_trip_ok,
    parent_segment_of_child,
    rename_cascade,
)
from imas_codex.standard_names.graph_ops import (
    persist_refined_docs,
    persist_refined_name,
    reset_standard_name_docs,
)

#: name_stage values eligible for a direct rename (superseded is handled
#: separately: eligible only when it has no successor).
_RENAME_ELIGIBLE_STAGES = frozenset({"accepted", "reviewed", "exhausted", "drafted"})

_ENTRY_BY_MODE = {"rename": "review_name", "docs": "review_docs", "hint": "generate"}


@dataclass(frozen=True)
class EditPlan:
    """Outcome of an :func:`apply_edit` invocation.

    Attributes
    ----------
    target:
        The StandardName id the edit was requested against (the id the
        caller passed in — for a family-scoped leaf rename this may differ
        from the node actually re-entering the pipeline; see ``actions``).
    mode:
        ``"hint" | "rename" | "docs"``.
    axis:
        ``"name" | "docs" | "both"``.
    scope:
        ``"only_self" | "family" | "subtree"``.
    entry:
        Pipeline stage the edit re-enters at: ``"generate" | "review_name"
        | "review_docs"``.
    successor:
        The new drafted StandardName id (rename mode only — a new node
        identity). ``None`` for docs/hint modes (same-id, in-place) or when
        blocked/dry-run.
    cascade_planned:
        ``[{"from": ..., "to": ...}]`` staged descendant renames (subtree /
        family scope only). Populated even in ``dry_run``.
    blocked:
        Human-readable refusal reason, or ``None`` if the edit is valid.
    actions:
        Human-readable action lines — drives ``--dry-run`` CLI output.
    applied:
        ``False`` for ``dry_run`` or ``blocked`` outcomes.
    """

    target: str
    mode: str
    axis: str
    scope: str
    entry: str
    successor: str | None
    cascade_planned: list[dict[str, str]] = field(default_factory=list)
    blocked: str | None = None
    actions: list[str] = field(default_factory=list)
    applied: bool = False


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _base_token(name: str) -> str | None:
    """The ISN grammar's physical base token for *name*, or ``None`` if it
    fails to parse (treated conservatively by callers — "cannot prove the
    base is unchanged")."""
    try:
        return _isn_parser.parse(name).ir.base.token
    except Exception:
        return None


def _blocked(
    target: str,
    mode: str,
    axis: str,
    scope: str,
    message: str,
    *,
    extra_actions: list[str] | None = None,
) -> EditPlan:
    actions = list(extra_actions or []) + [message]
    return EditPlan(
        target=target,
        mode=mode,
        axis=axis,
        scope=scope,
        entry=_ENTRY_BY_MODE[mode],
        successor=None,
        cascade_planned=[],
        blocked=message,
        actions=actions,
        applied=False,
    )


def _fetch_target(gc: GraphClient, sn_id: str) -> dict[str, Any] | None:
    """Fetch the fields ``apply_edit`` needs to validate + persist an edit."""
    rows = gc.query(
        """
        // EDIT_FETCH_TARGET
        MATCH (sn:StandardName {id: $id})
        OPTIONAL MATCH (succ:StandardName)-[:REFINED_FROM]->(sn)
        WITH sn, succ IS NOT NULL AS has_successor
        RETURN sn.name_stage AS name_stage,
               sn.docs_stage AS docs_stage,
               sn.description AS description,
               sn.documentation AS documentation,
               sn.docs_model AS docs_model,
               sn.docs_generated_at AS docs_generated_at,
               sn.kind AS kind,
               sn.unit AS unit,
               sn.physics_domain AS physics_domain,
               sn.tags AS tags,
               coalesce(sn.chain_length, 0) AS chain_length,
               has_successor,
               EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(sn) } AS has_children
        """,
        id=sn_id,
    )
    if not rows:
        return None
    return dict(rows[0])


def _stamp_edit_fields(
    gc: GraphClient,
    sn_id: str,
    *,
    edit_mode: str,
    name_hint: str | None,
    docs_hint: str | None,
    edit_reason: str,
    edit_origin: str,
    edit_scope: str,
    edit_status: str,
) -> None:
    gc.query(
        """
        // EDIT_STAMP_FIELDS
        MATCH (sn:StandardName {id: $id})
        SET sn.edit_mode         = $edit_mode,
            sn.name_hint         = $name_hint,
            sn.docs_hint         = $docs_hint,
            sn.edit_reason       = $edit_reason,
            sn.edit_origin       = $edit_origin,
            sn.edit_scope        = $edit_scope,
            sn.edit_status       = $edit_status,
            sn.edit_requested_at = $edit_requested_at
        """,
        id=sn_id,
        edit_mode=edit_mode,
        name_hint=name_hint,
        docs_hint=docs_hint,
        edit_reason=edit_reason,
        edit_origin=edit_origin,
        edit_scope=edit_scope,
        edit_status=edit_status,
        edit_requested_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_edit(
    *,
    target: str,
    hint: str | None = None,
    rename: str | None = None,
    docs: str | None = None,
    reason: str,
    axis: str | None = None,
    scope: str | None = None,
    origin: str = "human",
    dry_run: bool = False,
    gc: GraphClient | None = None,
) -> EditPlan:
    """Attach a steered edit proposal to a StandardName.

    Exactly one of ``hint``/``rename``/``docs`` selects the mode. ``reason``
    is mandatory (injected into the review gate to neutralise the
    reviewer's revert-to-original pull). Raises :class:`ValueError` for
    malformed calls (wrong argument combination, missing reason, invalid
    axis/scope/origin); returns an :class:`EditPlan` with ``blocked`` set
    for runtime graph-state refusals (unknown target, ineligible stage,
    shared-base desync, cascade conflicts).
    """
    provided = [
        name for name, val in (("hint", hint), ("rename", rename), ("docs", docs)) if val
    ]
    if len(provided) != 1:
        raise ValueError(
            "apply_edit requires exactly one of hint, rename, or docs "
            f"(got: {provided or 'none'})"
        )
    mode = provided[0]

    if not reason or not reason.strip():
        raise ValueError("apply_edit requires a non-empty reason")

    if origin not in (EditOrigin.human.value, EditOrigin.agent.value):
        raise ValueError(f"origin must be 'human' or 'agent', got {origin!r}")

    if mode == "rename":
        axis = "name"
    elif mode == "docs":
        axis = "docs"
    else:  # hint
        if axis is None:
            axis = "name"
        if axis not in ("name", "docs", "both"):
            raise ValueError(f"axis={axis!r} invalid for hint mode (name|docs|both)")

    if scope is not None and scope not in (
        EditScope.only_self.value,
        EditScope.family.value,
        EditScope.subtree.value,
    ):
        raise ValueError(f"scope={scope!r} invalid (only_self|family|subtree)")

    owns_gc = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        target_row = _fetch_target(gc, target)
        if target_row is None:
            return _blocked(
                target, mode, axis, scope or EditScope.only_self.value,
                f"target StandardName {target!r} not found",
            )

        is_parent = bool(target_row.get("has_children"))
        if scope is None:
            scope = EditScope.subtree.value if is_parent else EditScope.only_self.value

        if mode == "rename":
            return _apply_rename(
                gc,
                target=target,
                target_row=target_row,
                new_name=rename,
                reason=reason,
                origin=origin,
                scope=scope,
                is_parent=is_parent,
                dry_run=dry_run,
            )
        if mode == "docs":
            return _apply_docs(
                gc,
                target=target,
                target_row=target_row,
                new_docs=docs,
                reason=reason,
                origin=origin,
                scope=scope,
                dry_run=dry_run,
            )
        return _apply_hint(
            gc,
            target=target,
            target_row=target_row,
            hint=hint,
            axis=axis,
            reason=reason,
            origin=origin,
            scope=scope,
            dry_run=dry_run,
        )
    finally:
        if owns_gc:
            gc.close()


# ---------------------------------------------------------------------------
# Rename mode
# ---------------------------------------------------------------------------


def _apply_rename(
    gc: GraphClient,
    *,
    target: str,
    target_row: dict[str, Any],
    new_name: str | None,
    reason: str,
    origin: str,
    scope: str,
    is_parent: bool,
    dry_run: bool,
) -> EditPlan:
    if not new_name:
        raise ValueError("rename mode requires a non-empty `rename` value")

    # 1. ISN round-trip guard on the literal requested name.
    rt_ok, rt_reason = _isn_round_trip_ok(new_name)
    if not rt_ok:
        return _blocked(
            target, "rename", "name", scope,
            f"new name fails ISN grammar round-trip: {rt_reason}",
        )

    # 2. Collision check on the literal requested id — it will eventually
    #    become a live StandardName id whether created directly (only_self)
    #    or via cascade (family-mapped).
    coll = gc.query(
        "// EDIT_CHECK_COLLISION\nMATCH (sn:StandardName {id: $id}) RETURN count(sn) AS n",
        id=new_name,
    )
    if coll and coll[0].get("n"):
        return _blocked(
            target, "rename", "name", scope,
            f"a StandardName {new_name!r} already exists",
        )

    # 3. Shared-base guard + family→parent mapping (leaf targets only — a
    #    parent target's own siblings are unaffected by renaming it, since
    #    the subtree cascade only touches ITS descendants).
    #
    #    Gate on the ISN grammar's physical base token, not on template
    #    string-matching: changing only the qualifier/operator token (e.g.
    #    electron_temperature → upper_temperature) leaves the base
    #    ("temperature") untouched and is always safe in place, even with
    #    siblings present — the base is what siblings actually share.
    refine_root_old = target
    refine_root_new = new_name
    mapped_from_leaf = False
    actions: list[str] = []

    base_changed = _base_token(target) != _base_token(new_name)

    if not is_parent and base_changed:
        edges = gc.query(
            """
            // EDIT_FETCH_PARENT_EDGES
            MATCH (child:StandardName {id: $id})-[r:HAS_PARENT]->(parent:StandardName)
            RETURN parent.id AS parent_id, r.operator AS operator,
                   r.operator_kind AS operator_kind, r.role AS role,
                   r.separator AS separator
            """,
            id=target,
        )
        for edge in edges:
            other_arg = None
            if edge.get("operator_kind") == "binary":
                other_role = "b" if edge.get("role") == "a" else "a"
                other_arg = next(
                    (
                        e.get("parent_id")
                        for e in edges
                        if e.get("operator_kind") == "binary"
                        and e.get("operator") == edge.get("operator")
                        and e.get("role") == other_role
                    ),
                    None,
                )
            old_part = parent_segment_of_child(edge, target, other_arg)
            if old_part is None:
                continue
            new_part = parent_segment_of_child(edge, new_name, other_arg)
            if new_part == old_part:
                continue  # only the operator/qualifier token changed — safe

            parent_id = edge.get("parent_id")
            sib_rows = gc.query(
                """
                // EDIT_FETCH_SIBLINGS
                MATCH (sib:StandardName)-[:HAS_PARENT]->(parent:StandardName {id: $parent_id})
                WHERE sib.id <> $target_id AND sib.id CONTAINS $substring
                RETURN count(sib) AS n
                """,
                parent_id=parent_id,
                target_id=target,
                substring=old_part,
            )
            sib_count = sib_rows[0].get("n", 0) if sib_rows else 0
            if not sib_count:
                continue  # no siblings to desync — safe to rename this leaf in place

            if scope != EditScope.family.value:
                return _blocked(
                    target, "rename", "name", scope,
                    f"renaming the shared segment {old_part!r} would desync "
                    f"{sib_count} sibling(s) under parent {parent_id!r} — "
                    "use --scope family",
                )
            if old_part != parent_id:
                return _blocked(
                    target, "rename", "name", scope,
                    f"cannot map family-scope rename onto parent {parent_id!r} — "
                    f"the edited segment {old_part!r} does not match the parent's "
                    "id (topology inconsistency)",
                )
            if new_part is None:
                return _blocked(
                    target, "rename", "name", scope,
                    f"cannot map family-scope rename onto parent {parent_id!r} — "
                    "the requested new name does not follow the same cascade "
                    "template as the current name",
                )
            refine_root_old, refine_root_new = parent_id, new_part
            mapped_from_leaf = True
            actions.append(
                f"family scope maps to parent {parent_id!r} — mapped rename: "
                f"{parent_id!r} → {new_part!r} (subtree)"
            )
            break

    # 4. Eligible-stage check — applies to whichever node is actually being
    #    refined (the mapped parent, or the target itself).
    root_row = target_row
    if mapped_from_leaf:
        root_row = _fetch_target(gc, refine_root_old)
        if root_row is None:
            return _blocked(
                target, "rename", "name", scope,
                f"mapped parent {refine_root_old!r} not found",
                extra_actions=actions,
            )

    root_stage = root_row.get("name_stage")
    root_has_successor = bool(root_row.get("has_successor"))
    if root_stage == "superseded":
        if root_has_successor:
            return _blocked(
                target, "rename", "name", scope,
                f"{refine_root_old!r} is superseded and has a successor — "
                "edit the successor instead",
                extra_actions=actions,
            )
    elif root_stage not in _RENAME_ELIGIBLE_STAGES:
        return _blocked(
            target, "rename", "name", scope,
            f"{refine_root_old!r} has name_stage={root_stage!r} — not eligible "
            "for rename (must be accepted/reviewed/exhausted/drafted, or "
            "superseded with no successor)",
            extra_actions=actions,
        )

    # 5. Plan the descendant cascade now (dry-run) — conflicts refuse the
    #    whole edit, all-or-nothing. Even a childless root plans cleanly
    #    (no descendants to resolve).
    cascade_planned: list[dict[str, str]] = []
    if scope in (EditScope.family.value, EditScope.subtree.value):
        plan_result = rename_cascade(
            gc,
            old_name=refine_root_old,
            new_name=refine_root_new,
            dry_run=True,
            override_edits=True,
            include_accepted=True,
        )
        if plan_result.conflicts:
            return _blocked(
                target, "rename", "name", scope,
                "cascade plan conflict: " + "; ".join(plan_result.conflicts),
                extra_actions=actions,
            )
        cascade_planned = [
            r for r in plan_result.renamed if r["from"] != refine_root_old
        ]

    if dry_run:
        actions.append(
            f"[dry-run] would rename {refine_root_old!r} → {refine_root_new!r}"
            f" ({len(cascade_planned)} descendant(s) staged)"
            if cascade_planned
            else f"[dry-run] would rename {refine_root_old!r} → {refine_root_new!r}"
        )
        return EditPlan(
            target=target,
            mode="rename",
            axis="name",
            scope=scope,
            entry="review_name",
            successor=None,
            cascade_planned=cascade_planned,
            blocked=None,
            actions=actions,
            applied=False,
        )

    # 6. Apply: enter REVIEW_NAME by creating the refined successor node.
    gc.query(
        """
        // EDIT_SET_REFINING
        MATCH (sn:StandardName {id: $id})
        SET sn.name_stage = 'refining'
        """,
        id=refine_root_old,
    )
    result = persist_refined_name(
        old_name=refine_root_old,
        new_name=refine_root_new,
        description=root_row.get("description") or "",
        kind=root_row.get("kind") or "scalar",
        unit=root_row.get("unit"),
        physics_domain=root_row.get("physics_domain"),
        tags=root_row.get("tags") or [],
        old_chain_length=root_row.get("chain_length") or 0,
        model="sn-edit",
        reason=reason,
        edit_mode=EditMode.rename.value,
        name_hint=refine_root_new,
        edit_reason=reason,
        edit_origin=origin,
        edit_scope=scope,
        edit_status=EditStatus.open.value,
        edit_requested_at=_now_iso(),
    )
    successor = result["new_name"]
    actions.append(
        f"renamed {refine_root_old!r} → {successor!r}, entering name review "
        "(edit_status=open)"
    )
    return EditPlan(
        target=target,
        mode="rename",
        axis="name",
        scope=scope,
        entry="review_name",
        successor=successor,
        cascade_planned=cascade_planned,
        blocked=None,
        actions=actions,
        applied=True,
    )


# ---------------------------------------------------------------------------
# Docs mode
# ---------------------------------------------------------------------------


def _apply_docs(
    gc: GraphClient,
    *,
    target: str,
    target_row: dict[str, Any],
    new_docs: str | None,
    reason: str,
    origin: str,
    scope: str,
    dry_run: bool,
) -> EditPlan:
    if not new_docs:
        raise ValueError("docs mode requires a non-empty `docs` value")

    name_stage = target_row.get("name_stage")
    has_successor = bool(target_row.get("has_successor"))
    if name_stage == "superseded" and has_successor:
        return _blocked(
            target, "docs", "docs", scope,
            f"{target!r} is superseded and has a successor — edit the "
            "successor instead",
        )
    if name_stage != "accepted":
        return _blocked(
            target, "docs", "docs", scope,
            f"target name_stage={name_stage!r} — docs edits require an "
            "accepted name (name_stage='accepted')",
        )

    actions = [f"docs replacement queued for {target!r}"]
    if dry_run:
        actions.append("[dry-run] no writes performed")
        return EditPlan(
            target=target,
            mode="docs",
            axis="docs",
            scope=scope,
            entry="review_docs",
            successor=None,
            cascade_planned=[],
            blocked=None,
            actions=actions,
            applied=False,
        )

    token = str(uuid.uuid4())
    gc.query(
        """
        // EDIT_CLAIM_FOR_DOCS_REFINE
        MATCH (sn:StandardName {id: $id})
        WHERE sn.name_stage = 'accepted'
        SET sn.docs_stage = 'refining', sn.claim_token = $token,
            sn.claimed_at = datetime()
        """,
        id=target,
        token=token,
    )
    result = persist_refined_docs(
        sn_id=target,
        claim_token=token,
        description=target_row.get("description") or "",
        documentation=new_docs,
        model="sn-edit",
        current_description=target_row.get("description") or "",
        current_documentation=target_row.get("documentation") or "",
        current_model=target_row.get("docs_model"),
        current_generated_at=target_row.get("docs_generated_at"),
    )
    if result.get("docs_chain_length", -1) < 0:
        return _blocked(
            target, "docs", "docs", scope,
            "docs claim raced — target left docs_stage='refining'; retry the edit",
            extra_actions=actions,
        )

    _stamp_edit_fields(
        gc,
        target,
        edit_mode=EditMode.docs.value,
        name_hint=None,
        docs_hint=new_docs,
        edit_reason=reason,
        edit_origin=origin,
        edit_scope=scope,
        edit_status=EditStatus.open.value,
    )
    actions.append(
        f"docs refined in place (revision={result.get('revision_id')}), "
        "entering docs review (edit_status=open)"
    )
    return EditPlan(
        target=target,
        mode="docs",
        axis="docs",
        scope=scope,
        entry="review_docs",
        successor=None,
        cascade_planned=[],
        blocked=None,
        actions=actions,
        applied=True,
    )


# ---------------------------------------------------------------------------
# Hint mode
# ---------------------------------------------------------------------------


def _apply_hint(
    gc: GraphClient,
    *,
    target: str,
    target_row: dict[str, Any],
    hint: str | None,
    axis: str,
    reason: str,
    origin: str,
    scope: str,
    dry_run: bool,
) -> EditPlan:
    if not hint:
        raise ValueError("hint mode requires a non-empty `hint` value")

    name_stage = target_row.get("name_stage")
    has_successor = bool(target_row.get("has_successor"))
    if name_stage == "superseded" and has_successor:
        return _blocked(
            target, "hint", axis, scope,
            f"{target!r} is superseded and has a successor — edit the "
            "successor instead",
        )

    actions = [f"hint attached to {target!r} (axis={axis})"]
    if dry_run:
        actions.append("[dry-run] no writes performed")
        return EditPlan(
            target=target,
            mode="hint",
            axis=axis,
            scope=scope,
            entry="generate",
            successor=None,
            cascade_planned=[],
            blocked=None,
            actions=actions,
            applied=False,
        )

    name_hint_value = hint if axis in ("name", "both") else None
    docs_hint_value = hint if axis in ("docs", "both") else None
    _stamp_edit_fields(
        gc,
        target,
        edit_mode=EditMode.hint.value,
        name_hint=name_hint_value,
        docs_hint=docs_hint_value,
        edit_reason=reason,
        edit_origin=origin,
        edit_scope=scope,
        edit_status=EditStatus.open.value,
    )

    if axis in ("name", "both"):
        src_rows = gc.query(
            """
            // EDIT_RESET_SOURCES
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: $id})
            SET src.status = 'extracted', src.claimed_at = null,
                src.claim_token = null, src.attempt_count = 0
            RETURN src.id AS id
            """,
            id=target,
        )
        actions.append(
            f"reset {len(src_rows)} producing source(s) to status='extracted' "
            "for regeneration"
        )

    if axis in ("docs", "both"):
        docs_result = reset_standard_name_docs(sn_ids=[target])
        actions.append(
            f"docs reset for regeneration (eligible={docs_result['eligible']}, "
            f"reset={docs_result['reset']})"
        )

    return EditPlan(
        target=target,
        mode="hint",
        axis=axis,
        scope=scope,
        entry="generate",
        successor=None,
        cascade_planned=[],
        blocked=None,
        actions=actions,
        applied=True,
    )

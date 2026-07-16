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

Validation-parity call graph (edit entry → accepted, gate by gate)
------------------------------------------------------------------
Every edit-origin artifact clears exactly the gates a pipeline-generated
name clears — there is no privileged accept path:

- **rename** — ``apply_edit`` → ``_apply_rename``:
  1. ISN grammar round-trip on the literal requested name
     (``cascade._isn_round_trip_ok``) — same round-trip the validate gate
     and the cascade apply run; a grammar-invalid name is refused up front.
  2. id-collision check against the live graph.
  3. shared-base / sibling desync guard (``--scope family`` mapping).
  4. ``persist_refined_name`` mints the ``drafted`` successor, then
     ``_stamp_successor_validation`` runs the FULL name-admission gate
     (:func:`imas_codex.standard_names.workers.validate_name_candidate`:
     grammar round-trip + ISN Pydantic/semantic/structural/canonical/
     description layers + L3 audits) and stamps ``validation_status``.  A
     quarantined successor is skipped with a 0.0 review by the review
     worker and can never reach ``accepted`` — identical to a quarantined
     pipeline candidate.
  5. name review (``review_name`` pool → ``persist_reviewed_name``) scores
     it; ``score >= min_score`` accepts.
  6. on acceptance of a ``family``/``subtree`` rename, the descendant
     cascade is preflighted (round-trip + uniqueness of every descendant
     id) BEFORE the acceptance commits; any conflict refuses the
     acceptance (``name_stage`` stays ``reviewed``) and renames nothing.
- **docs** — ``apply_edit`` → ``_apply_docs``: ``name_stage='accepted'`` +
  ``docs_stage in (accepted, exhausted)`` preconditions, then
  ``protection.filter_protected`` (catalog-edit docs require
  ``--override-edits``), then ``persist_refined_docs`` → the ``review_docs``
  pool scores the replacement — the same docs gate a pipeline docs
  candidate rides.
- **hint** — ``apply_edit`` → ``_apply_hint``: resets the producing
  sources / docs so the name is REGENERATED through the ``generate_name`` /
  ``generate_docs`` pools; it therefore rides the pipeline's own gates by
  construction (nothing edit-specific to validate).

``--scope family`` semantics: a shared-segment leaf edit is PROMOTED to its
parent and the parent's rename cascades through the **full** ``HAS_PARENT``
subtree (every descendant that embeds the segment), not just the leaf's
direct siblings.  Full-subtree breadth is grammar-required: a grandchild
such as ``time_derivative_of_upper_elongation`` embeds the same base as
``elongation`` and would desync into invalid grammar if only direct
siblings were renamed.  (The ``EditScope.family`` enum docstring in
``imas_codex.graph.models`` still reads "sibling StandardNames" — that text
is stale; the behaviour is full-subtree and the enum doc should be
corrected to match.)
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
    tombstone_supersede_into,
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
    run_id:
        The ``sn-edit-<UTC timestamp>`` scope stamp written onto the touched
        SN (rename mode: the drafted successor; docs/hint modes: the
        target). Lets an operator run a surgical pool rotation over just
        this edit (``sn run --scope-run-id <id>``) instead of opening the
        whole backlog. ``None`` for ``dry_run`` or ``blocked`` outcomes —
        nothing was stamped.
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
    run_id: str | None = None


@dataclass(frozen=True)
class InlineReviewResult:
    """Per-successor outcome of an inline review (:func:`run_inline_review`).

    Attributes
    ----------
    id:
        StandardName id the review scored.
    name_stage / docs_stage:
        Final lifecycle stage after the scoped review rotation
        (``accepted`` / ``reviewed`` / ``exhausted`` / …).
    edit_status:
        Edit lifecycle after review (``applied`` when accepted, ``exhausted``
        when the refine cap was reached below threshold, ``open`` when still
        mid-rotation — see :class:`imas_codex.graph.models.EditStatus`).
    reviewer_score_name / reviewer_score_docs:
        The winning reviewer score on the relevant axis, or ``None`` if the
        axis was not scored.
    accepted:
        ``True`` iff the relevant axis reached ``accepted`` — the gate was
        cleared with no privileged path. A below-threshold or exhausted
        successor reports ``accepted=False``; the score is the signal.
    """

    id: str
    name_stage: str | None
    docs_stage: str | None
    edit_status: str | None
    reviewer_score_name: float | None
    reviewer_score_docs: float | None
    accepted: bool


@dataclass(frozen=True)
class InlineReviewOutcome:
    """Result of running the review pipeline inline after ``sn edit`` staging.

    Attributes
    ----------
    ran:
        ``False`` when there was nothing to review (the edit did not apply,
        was blocked, or carries no scope stamp) — the caller staged only.
    run_id:
        The ``sn-edit-<ts>`` scope the review claimed against.
    cost:
        LLM spend (USD) of the inline review, from the run's authoritative
        ledger.
    stop_reason:
        Why the scoped run stopped (``no_eligible_work`` on a clean drain,
        ``budget_exhausted`` if the cost cap bound it, …).
    results:
        One :class:`InlineReviewResult` per touched successor (rename: the
        successor + any cascade descendants; docs/hint: the target).
    """

    ran: bool
    run_id: str | None
    cost: float
    stop_reason: str | None
    results: list[InlineReviewResult] = field(default_factory=list)

    @property
    def all_accepted(self) -> bool:
        """``True`` iff the review ran and every touched successor landed."""
        return self.ran and bool(self.results) and all(r.accepted for r in self.results)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_id() -> str:
    """A fresh ``sn-edit-<UTC compact timestamp>`` scope stamp.

    Passed to ``sn run --scope-run-id`` to restrict pool claims to exactly
    the SN(s) this edit touched.
    """
    return f"sn-edit-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _base_token(name: str) -> str | None:
    """The ISN grammar's physical base token for *name*, or ``None`` if it
    fails to parse (treated conservatively by callers — "cannot prove the
    base is unchanged")."""
    try:
        return _isn_parser.parse(name).ir.base.token
    except Exception:
        return None


def _grammar_segment_props(name: str) -> dict[str, str]:
    """Parsed ISN segment properties for a grammar-valid name.

    Stamped on a rename successor so the review gate scores grammar from
    the actual registered decomposition instead of reverse-engineering the
    vocabulary — a reviewer guessing at unregistered base tokens is part of
    the revert-to-original pull this tool exists to neutralize.
    """
    props: dict[str, str] = {}
    try:
        ir = _isn_parser.parse(name).ir
    except Exception:
        return props
    if ir.base is not None:
        props["physical_base"] = ir.base.token
    if ir.locus is not None:
        props["geometry"] = ir.locus.token
    try:
        from imas_standard_names import __version__ as _isn_version

        props["grammar_parse_version"] = _isn_version
    except Exception:
        pass
    return props


def _stamp_successor_validation(
    gc: GraphClient, successor: str, root_row: dict[str, Any]
) -> None:
    """Run the pipeline name-admission gate on a rename successor and stamp it.

    Overwrites the provisional ``validation_status`` persist_refined_name
    seeds so an edit-origin name is judged by exactly the gate a
    pipeline-generated candidate passes (grammar round-trip, ISN Pydantic /
    semantic / structural / canonical / description layers, L3 audits).  A
    quarantined result cannot reach ``accepted`` — the review worker skips
    quarantined names with a 0.0 score.
    """
    from imas_codex.standard_names.workers import validate_name_candidate

    entry = {
        "id": successor,
        "kind": root_row.get("kind") or "scalar",
        "unit": root_row.get("unit"),
        "description": root_row.get("description") or "",
        "physics_domain": root_row.get("physics_domain"),
        "cocos_transformation_type": root_row.get("cocos_transformation_type"),
        "source_paths": root_row.get("source_paths") or [],
    }
    issues, _summary, status = validate_name_candidate(entry)
    gc.query(
        """
        // EDIT_STAMP_VALIDATION
        MATCH (sn:StandardName {id: $id})
        SET sn.validation_status = $status,
            sn.validation_issues = $issues,
            sn.validated_at = datetime()
        """,
        id=successor,
        status=status,
        issues=issues,
    )


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
               sn.origin AS origin,
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
    run_id: str,
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
            sn.edit_requested_at = $edit_requested_at,
            sn.run_id            = $run_id
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
        run_id=run_id,
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
    override_edits: bool = False,
    include_accepted: bool = False,
    refine: bool = True,
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

    ``override_edits`` / ``include_accepted`` (rename mode, family/subtree
    scope only) are the operator's opt-in to let the descendant cascade
    rename descendants that are catalog-edited (``origin='catalog_edit'``)
    or committed (``name_stage='accepted'``) respectively. Both default to
    ``False`` — without them, such descendants surface as cascade conflicts
    in the dry-run plan and block the edit rather than being silently
    clobbered. The recorded values are re-read at acceptance time so the
    post-review cascade reproduces exactly the operator's choice.

    ``refine`` (default ``True``) declares whether the attached proposal may
    be automatically refined (its wording rewritten) if review scores it
    below threshold. ``sn merge`` attaches human-approved catalog wording
    with ``refine=False`` so the proposal is stamped ``edit_refine=false`` —
    a durable review-only marker recording that the wording must be scored
    as-is and never silently mutated. It does not alter the attach itself;
    the merge caller enforces the accept-or-quarantine outcome.
    """
    provided = [
        name
        for name, val in (("hint", hint), ("rename", rename), ("docs", docs))
        if val
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
                target,
                mode,
                axis,
                scope or EditScope.only_self.value,
                f"target StandardName {target!r} not found",
            )

        is_parent = bool(target_row.get("has_children"))
        if scope is None:
            scope = EditScope.subtree.value if is_parent else EditScope.only_self.value

        if mode == "rename":
            plan = _apply_rename(
                gc,
                target=target,
                target_row=target_row,
                new_name=rename,
                reason=reason,
                origin=origin,
                scope=scope,
                is_parent=is_parent,
                override_edits=override_edits,
                include_accepted=include_accepted,
                dry_run=dry_run,
            )
        elif mode == "docs":
            plan = _apply_docs(
                gc,
                target=target,
                target_row=target_row,
                new_docs=docs,
                reason=reason,
                origin=origin,
                scope=scope,
                override_edits=override_edits,
                dry_run=dry_run,
            )
        else:
            plan = _apply_hint(
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

        # Durable review-only marker: when the caller disables auto-refine
        # (``sn merge`` folding human-approved wording), stamp the touched
        # node so its provenance records that the wording must be scored
        # as-is and never silently rewritten.
        if not refine and plan.applied and plan.blocked is None:
            stamped = plan.successor or target
            gc.query(
                """
                // EDIT_STAMP_REVIEW_ONLY
                MATCH (sn:StandardName {id: $id})
                SET sn.edit_refine = false
                """,
                id=stamped,
            )
        return plan
    finally:
        if owns_gc:
            gc.close()


def reclassify_domain(
    name: str,
    domain: str,
    *,
    reason: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reassign a standard name's physics domain with recorded provenance.

    ``physics_domain`` is not a compose/review axis, so ``--rename``/``--docs``
    cannot express it; this is the supported operation for a semantic-subject
    domain move. It SETs ``physics_domain`` and ``source_domains`` on the node
    and records a ``StandardNameChange`` (operation ``reclassify_domain``) so
    the move is traceable without leaking into public outputs.

    Returns ``{"ok": bool, ...}`` with the before/after domains; on refusal
    ``ok`` is False with a ``reason``.
    """
    from imas_codex.core.physics_domain import PhysicsDomain
    from imas_codex.standard_names.provenance_lifecycle import (
        record_standard_name_change,
    )

    name = (name or "").strip()
    reason = (reason or "").strip()
    if not name:
        return {"ok": False, "reason": "a standard name is required"}
    if not reason:
        return {"ok": False, "reason": "--reason is mandatory for a domain move"}
    try:
        target_domain = PhysicsDomain(domain).value
    except ValueError:
        valid = ", ".join(d.value for d in PhysicsDomain)
        return {
            "ok": False,
            "reason": f"unknown physics domain {domain!r}; one of: {valid}",
        }

    with GraphClient() as gc:
        rows = list(
            gc.query(
                "MATCH (n:StandardName {id: $id}) "
                "RETURN n.physics_domain AS domain, n.source_domains AS sources, "
                "n.name_stage AS stage",
                id=name,
            )
        )
        if not rows:
            return {"ok": False, "reason": f"standard name {name!r} not found"}
        before = rows[0]
        result = {
            "ok": True,
            "name": name,
            "stage": before["stage"],
            "from_domain": before["domain"],
            "to_domain": target_domain,
            "dry_run": dry_run,
        }
        if dry_run or before["domain"] == target_domain:
            result["noop"] = before["domain"] == target_domain
            return result
        gc.query(
            "MATCH (n:StandardName {id: $id}) "
            "SET n.physics_domain = $domain, n.source_domains = [$domain]",
            id=name,
            domain=target_domain,
        )
        result["change_id"] = record_standard_name_change(
            gc,
            name,
            name,
            operation="reclassify_domain",
            reason=reason,
            origin="catalog_edit",
        )
        return result


def supersede_into(
    old: str,
    into: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fold ``old`` into an already-existing accepted name ``into``.

    The rename path refuses a rename onto an existing id, and the source-keyed
    supersede only retires predecessors sharing one source — so folding a name
    into an existing canonical name (or re-pointing a name onto a restored
    tombstoned id) has no supported operation. This is that operation: it
    tombstones ``old`` and threads the ``REFINED_FROM`` lineage so the export
    boundary emits a ``status: deprecated`` stub pointing at ``into``.

    Thin validation wrapper over
    :func:`imas_codex.standard_names.graph_ops.tombstone_supersede_into` — it
    normalises the ids and forwards; all guards (target must be accepted, no
    self-fold, no cycle, idempotent re-stamp) live in the graph op so the CLI
    and any programmatic caller share one enforcement point.

    Returns the graph op's result dict: ``{"ok": bool, ...}`` — on refusal
    ``ok`` is False with a ``reason``.
    """
    old = (old or "").strip()
    into = (into or "").strip()
    if not old or not into:
        return {"ok": False, "reason": "both old and target names are required"}
    return tombstone_supersede_into(old, into, dry_run=dry_run)


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
    override_edits: bool,
    include_accepted: bool,
    dry_run: bool,
) -> EditPlan:
    if not new_name:
        raise ValueError("rename mode requires a non-empty `rename` value")

    # 1. ISN round-trip guard on the literal requested name.
    rt_ok, rt_reason = _isn_round_trip_ok(new_name)
    if not rt_ok:
        return _blocked(
            target,
            "rename",
            "name",
            scope,
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
            target,
            "rename",
            "name",
            scope,
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
                    target,
                    "rename",
                    "name",
                    scope,
                    f"renaming the shared segment {old_part!r} would desync "
                    f"{sib_count} sibling(s) under parent {parent_id!r} — "
                    "use --scope family",
                )
            if old_part != parent_id:
                return _blocked(
                    target,
                    "rename",
                    "name",
                    scope,
                    f"cannot map family-scope rename onto parent {parent_id!r} — "
                    f"the edited segment {old_part!r} does not match the parent's "
                    "id (topology inconsistency)",
                )
            if new_part is None:
                return _blocked(
                    target,
                    "rename",
                    "name",
                    scope,
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
                target,
                "rename",
                "name",
                scope,
                f"mapped parent {refine_root_old!r} not found",
                extra_actions=actions,
            )

    root_stage = root_row.get("name_stage")
    root_has_successor = bool(root_row.get("has_successor"))
    if root_stage == "superseded":
        if root_has_successor:
            return _blocked(
                target,
                "rename",
                "name",
                scope,
                f"{refine_root_old!r} is superseded and has a successor — "
                "edit the successor instead",
                extra_actions=actions,
            )
    elif root_stage not in _RENAME_ELIGIBLE_STAGES:
        return _blocked(
            target,
            "rename",
            "name",
            scope,
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
            override_edits=override_edits,
            include_accepted=include_accepted,
        )
        if plan_result.conflicts:
            return _blocked(
                target,
                "rename",
                "name",
                scope,
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
    run_id = _new_run_id()
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
        run_id=run_id,
        edit_mode=EditMode.rename.value,
        name_hint=refine_root_new,
        edit_reason=reason,
        edit_origin=origin,
        edit_scope=scope,
        edit_status=EditStatus.open.value,
        edit_requested_at=_now_iso(),
        edit_override_edits=override_edits,
        edit_include_accepted=include_accepted,
    )
    successor = result["new_name"]

    # Stamp the parsed ISN segment decomposition on the successor so the
    # review gate sees the verified grammar fields instead of guessing at
    # the registered vocabulary.
    seg_props = _grammar_segment_props(successor)
    if seg_props:
        gc.query(
            "MATCH (sn:StandardName {id: $id}) SET sn += $props",
            id=successor,
            props=seg_props,
        )

    # Gate parity: a rename mints a brand-new name string that never rode the
    # generate pool's admission gate.  persist_refined_name stamps a
    # provisional validation_status='valid' (its default for a refine
    # rotation); run the SAME gate a pipeline-generated candidate passes so a
    # grammar-valid-but-semantically/structurally-invalid replacement is
    # quarantined here and can never reach 'accepted' (the review worker
    # persists a 0.0 review for quarantined names). No privileged path.
    _stamp_successor_validation(gc, successor, root_row)

    actions.append(
        f"renamed {refine_root_old!r} → {successor!r}, entering name review "
        f"(edit_status=open, run_id={run_id})"
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
        run_id=run_id,
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
    override_edits: bool,
    dry_run: bool,
) -> EditPlan:
    if not new_docs:
        raise ValueError("docs mode requires a non-empty `docs` value")

    name_stage = target_row.get("name_stage")
    has_successor = bool(target_row.get("has_successor"))
    if name_stage == "superseded" and has_successor:
        return _blocked(
            target,
            "docs",
            "docs",
            scope,
            f"{target!r} is superseded and has a successor — edit the "
            "successor instead",
        )
    if name_stage != "accepted":
        return _blocked(
            target,
            "docs",
            "docs",
            scope,
            f"target name_stage={name_stage!r} — docs edits require an "
            "accepted name (name_stage='accepted')",
        )

    # Docs-edit claim precondition (parity with the docs pipeline's own
    # eligibility): documentation may only be re-opened once the docs axis
    # has settled — accepted (published) or exhausted (refine cap reached).
    # A name still drafting/refining/pending docs is mid-flight; steering it
    # would race the docs pool.
    docs_stage = target_row.get("docs_stage")
    if docs_stage not in ("accepted", "exhausted"):
        return _blocked(
            target,
            "docs",
            "docs",
            scope,
            f"target docs_stage={docs_stage!r} — docs edits require the docs "
            "axis to have settled (docs_stage in accepted/exhausted)",
        )

    # Catalog-edit protection: `documentation` is a catalog-authoritative
    # field (see protection.PROTECTED_FIELDS). Editing the documentation of a
    # name curated via a catalog PR (origin='catalog_edit') runs the SAME
    # filter the pipeline writers run — it strips the write unless the
    # operator explicitly overrides. Route through filter_protected so there
    # is one protection decision, not a parallel one.
    from imas_codex.standard_names.protection import filter_protected

    is_catalog_edit = target_row.get("origin") == "catalog_edit"
    _filtered, _skipped = filter_protected(
        [{"id": target, "documentation": new_docs}],
        override=override_edits,
        protected_names={target} if is_catalog_edit else set(),
    )
    if _skipped:
        return _blocked(
            target,
            "docs",
            "docs",
            scope,
            f"{target!r} is catalog-edited (origin='catalog_edit') — its "
            "documentation is catalog-authoritative; pass --override-edits to "
            "steer it anyway",
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
    run_id = _new_run_id()
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
        run_id=run_id,
    )
    if result.get("docs_chain_length", -1) < 0:
        return _blocked(
            target,
            "docs",
            "docs",
            scope,
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
        run_id=run_id,
    )
    actions.append(
        f"docs refined in place (revision={result.get('revision_id')}), "
        f"entering docs review (edit_status=open, run_id={run_id})"
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
        run_id=run_id,
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
            target,
            "hint",
            axis,
            scope,
            f"{target!r} is superseded and has a successor — edit the "
            "successor instead",
        )

    # A name-axis hint steers regeneration, which is driven by the target's
    # producing StandardNameSource(s). A derived/structural name has none —
    # resetting zero sources is a silent no-op that leaves edit_status stuck
    # 'open' forever. Block with an actionable alternative instead.
    if axis in ("name", "both"):
        src_count = gc.query(
            """
            // EDIT_COUNT_PRODUCING_SOURCES
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: $id})
            RETURN count(src) AS n
            """,
            id=target,
        )
        if not (src_count and src_count[0].get("n")):
            return _blocked(
                target,
                "hint",
                axis,
                scope,
                f"{target!r} has no producing StandardNameSource (it is a "
                "derived/structural name) — a name-axis hint cannot regenerate "
                "it. Use `--rename` to propose a replacement name, or "
                "`--axis docs` to steer only its documentation.",
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

    run_id = _new_run_id()
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
        run_id=run_id,
    )

    if axis in ("name", "both"):
        # Stamp the reset sources with the edit's run_id so an inline review
        # scoped to this run (run_inline_review → scope_run_id) claims exactly
        # the regenerated candidate — a scope filter on StandardNameSource.run_id
        # only matches when the source carries the stamp.
        src_rows = gc.query(
            """
            // EDIT_RESET_SOURCES
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: $id})
            SET src.status = 'extracted', src.claimed_at = null,
                src.claim_token = null, src.attempt_count = 0,
                src.run_id = $run_id
            RETURN src.id AS id
            """,
            id=target,
            run_id=run_id,
        )
        actions.append(
            f"reset {len(src_rows)} producing source(s) to status='extracted' "
            "for regeneration"
        )

    if axis in ("docs", "both"):
        docs_result = reset_standard_name_docs(sn_ids=[target], run_id=run_id)
        actions.append(
            f"docs reset for regeneration (eligible={docs_result['eligible']}, "
            f"reset={docs_result['reset']})"
        )

    actions.append(f"scope stamp run_id={run_id}")
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
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Inline review — land a staged edit in one command
# ---------------------------------------------------------------------------


def _inline_review_ids(plan: EditPlan) -> list[str]:
    """The StandardName ids an inline review over *plan* should report on.

    A rename lands its own successor plus every cascade descendant it staged;
    docs / hint edits settle the target in place.
    """
    if plan.mode == "rename" and plan.successor:
        ids = [plan.successor]
        ids += [c["to"] for c in plan.cascade_planned if c.get("to")]
        return ids
    return [plan.target]


def _collect_inline_outcomes(
    gc: GraphClient, ids: list[str], *, axis: str
) -> list[InlineReviewResult]:
    """Read the post-review state of *ids* and classify each as accepted or not.

    Acceptance is judged on the axis the edit steered: a rename/hint-name edit
    accepts when ``name_stage='accepted'``; a docs edit accepts when
    ``docs_stage='accepted'``.  No score comparison here — the review pool has
    already applied the gate and written the stage; this only surfaces it.
    """
    rows = gc.query(
        """
        // EDIT_INLINE_COLLECT_OUTCOMES
        MATCH (sn:StandardName)
        WHERE sn.id IN $ids
        RETURN sn.id AS id,
               sn.name_stage AS name_stage,
               sn.docs_stage AS docs_stage,
               sn.edit_status AS edit_status,
               sn.reviewer_score_name AS reviewer_score_name,
               sn.reviewer_score_docs AS reviewer_score_docs
        """,
        ids=ids,
    )
    by_id = {r["id"]: r for r in rows}
    results: list[InlineReviewResult] = []
    for _id in ids:
        r = by_id.get(_id, {})
        name_stage = r.get("name_stage")
        docs_stage = r.get("docs_stage")
        accepted = (
            docs_stage == "accepted" if axis == "docs" else name_stage == "accepted"
        )
        results.append(
            InlineReviewResult(
                id=_id,
                name_stage=name_stage,
                docs_stage=docs_stage,
                edit_status=r.get("edit_status"),
                reviewer_score_name=r.get("reviewer_score_name"),
                reviewer_score_docs=r.get("reviewer_score_docs"),
                accepted=accepted,
            )
        )
    return results


def _run_scoped_pipeline(
    *,
    run_id: str,
    skip_generate: bool,
    cost_limit: float,
    min_score: float | None,
    rotation_cap: int | None,
    pending_fn: Any | None,
) -> Any:
    """Drive :func:`run_sn_pools` scoped to a single edit's ``run_id``.

    Runs the SAME six-pool orchestrator a normal ``sn run`` uses, so the
    inline review clears exactly the pool's gates (P2 parity) — there is no
    edit-privileged accept path.  ``scope_run_id`` restricts every pool claim
    to the SN(s) this edit stamped, so the review never touches the backlog.

    ``skip_generate`` is ``True`` for rename/docs edits (their candidate is
    already composed — this is ``--only review`` semantics: review + refine
    pools run, generation does not) and ``False`` for hint edits (which reset
    the producing sources for regeneration and therefore need the generate
    pool too).  The clear-gate footgun does not apply: it is a CLI-level guard
    on the ``sn run`` command, not part of ``run_sn_pools``, so an inline
    review — which calls ``run_sn_pools`` directly — never trips it.
    """
    import asyncio

    from imas_codex.standard_names.loop import run_sn_pools

    async def _main() -> Any:
        return await run_sn_pools(
            cost_limit=cost_limit,
            min_score=min_score,
            rotation_cap=rotation_cap,
            scope_run_id=run_id,
            skip_generate=skip_generate,
            pending_fn=pending_fn,
        )

    return asyncio.run(_main())


def run_inline_review(
    plan: EditPlan,
    *,
    cost_limit: float,
    min_score: float | None = None,
    rotation_cap: int | None = None,
    pending_fn: Any | None = None,
    gc: GraphClient | None = None,
) -> InlineReviewOutcome:
    """Review a just-staged ``sn edit`` inline, scoped to its ``run_id``.

    After :func:`apply_edit` stages a successor, this runs the review pipeline
    over exactly that edit's scope and reports whether it landed — so a single
    ``sn edit`` invocation stages *and* reviews, with no follow-up ``sn run``.

    The gate is honoured with no exception: the scoped pool scores the
    successor and writes ``name_stage``/``docs_stage`` itself; a below-threshold
    or refine-exhausted successor stays un-accepted and is reported as such
    (``accepted=False`` with its score).  A failed review is a result, not an
    error to paper over — the caller decides how to signal it.

    Returns an :class:`InlineReviewOutcome`.  When *plan* did not apply (dry-run,
    blocked, or no ``run_id``), returns ``ran=False`` with no results — nothing
    was staged to review.
    """
    if not (plan.applied and plan.run_id and plan.blocked is None):
        return InlineReviewOutcome(
            ran=False, run_id=plan.run_id, cost=0.0, stop_reason=None, results=[]
        )

    # rename/docs edits ride --only review (the candidate is composed); a hint
    # edit reset its sources and must regenerate, so keep the generate pool.
    skip_generate = plan.entry in ("review_name", "review_docs")

    summary = _run_scoped_pipeline(
        run_id=plan.run_id,
        skip_generate=skip_generate,
        cost_limit=cost_limit,
        min_score=min_score,
        rotation_cap=rotation_cap,
        pending_fn=pending_fn,
    )

    owns_gc = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        results = _collect_inline_outcomes(gc, _inline_review_ids(plan), axis=plan.axis)
    finally:
        if owns_gc:
            gc.close()

    return InlineReviewOutcome(
        ran=True,
        run_id=plan.run_id,
        cost=float(getattr(summary, "cost_spent", 0.0) or 0.0),
        stop_reason=getattr(summary, "stop_reason", None),
        results=results,
    )

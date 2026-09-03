"""The ``sn edit`` engine — steered proposals ride the SN pipeline.

``imas-codex sn edit <sn>`` lets a human or agent attach a proposal to a
``StandardName`` that rides the normal generate → review → score pipeline
instead of hand-editing graph text directly. Four modes:

- **hint** — a steering direction is injected into generate/refine prompts;
  the pipeline still composes the candidate.  Re-enters at ``generate``.
- **rename** — a full replacement name skips generation and rides straight
  into name review.  Re-enters at ``review_name``.
- **docs** — full replacement documentation skips generation and rides
  straight into docs review.  Re-enters at ``review_docs``.
- **kind** — an exact in-place structural repair derived from the unchanged
  identity, validated through ISN, previewed before an explicit apply, and
  recorded without changing already-reviewed wording or lifecycle stages.

Locked decisions (see the SN edit-engine plan):

- Edit-steering fields are scalar properties on ``StandardName`` (see
  ``imas_codex.graph.models``: ``EditMode``, ``EditOrigin``, ``EditStatus``,
  ``EditScope``).
- Review receives ONLY the edit reason (intent) — never the proposal
  pre-approved. The reviewer still independently scores the candidate.
- A shared-base leaf edit (renaming a segment a leaf's siblings also carry)
  blocks.  It is never promoted automatically to a structural parent; a
  parent-scoped edit must be requested and reviewed explicitly.
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
  3. shared-base / sibling desync guard (no leaf-to-parent promotion).
  4. ``persist_refined_name`` mints the ``drafted`` successor, then
     ``_stamp_successor_validation`` runs the FULL name-admission gate
     (:func:`imas_codex.standard_names.workers.validate_name_candidate`:
     grammar round-trip + ISN Pydantic/semantic/structural/canonical/
     description layers + post-generation audits) and stamps
     ``validation_status``.  A
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

``--scope family`` never widens a leaf edit through ``HAS_PARENT``.  Grammar
ancestry groups structural forms but does not prove that their sources share
an owner, representation, or repair intent.  Cascade-bearing parent edits are
reviewed only when the parent itself is the explicit target.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_standard_names.grammar import parser as _isn_parser
from neo4j.time import DateTime

from imas_codex.discovery.base.claims import retry_on_deadlock
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
from imas_codex.standard_names.kind_derivation import derive_kind

#: name_stage values eligible for a direct rename (superseded is handled
#: separately: eligible only when it has no successor; the stages below the
#: review entry point are handled by :func:`_stranded_rename_refusal`).
_RENAME_ELIGIBLE_STAGES = frozenset({"accepted", "reviewed", "exhausted", "drafted"})

#: name_stage values that sit below the review entry stage. A name resting
#: here has either not been minted yet or has been re-staged by a reseed.
_UNMINTED_NAME_STAGES = frozenset({"", "pending"})

#: StandardNameSource statuses whose composition has already been consumed —
#: the generate pool claims 'extracted' sources only, so a name produced
#: exclusively by sources in this set is never re-minted.
_LIVE_SOURCE_STATUSES = frozenset({"composed", "attached"})

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
        ``[{"from": ..., "to": ...}]`` descendant renames the root rename
        implies (subtree / family scope only). This is a *deferred* plan, not
        an outcome: nothing here is written when the edit is applied. The
        descendants keep their current ids until the successor reaches
        ``accepted``, at which point the acceptance hook re-walks the live
        subtree and applies the cascade in one transaction. A root that never
        reaches ``accepted`` leaves every entry here unperformed. Populated
        identically in ``dry_run``, which is why it must never be surfaced as
        completed work.
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
    semantic / structural / canonical / description layers, post-generation
    audits).  A
    quarantined result cannot reach ``accepted`` — the name-review claim
    predicate requires ``validation_status='valid'``, so a quarantined name is
    never claimed for review and never scored.
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


def _stranded_rename_refusal(
    sn_id: str, stage: str | None, row: dict[str, Any]
) -> str | None:
    """Refusal text for a rename whose root sits outside the eligible stages.

    Returns ``None`` when the rename is admitted. Exactly one class of
    ineligible stage is admitted: a *stranded* name — one resting below the
    review entry stage that no pool can ever move, and whose identity is
    nonetheless real. Such a name has no other repair vehicle:

    * the generate pool claims ``StandardNameSource`` at ``status='extracted'``,
      so a name whose sources are all composed/attached is never re-minted;
    * the review pool claims ``name_stage='drafted'``, so nothing lifts it;
    * ``reconcile_reviewable_name_stage`` lifts only ``validation_status='valid'``
      names, and a quarantined name is excluded from the review claim anyway,
      so advancing one to 'drafted' would simply re-strand it there.

    The conditions below each hold back a name whose repair is something other
    than a rename:

    no ``name_stage`` at all
        The refine hand-off compare-and-sets the predecessor on its concrete
        stage, so a bare node cannot be handed off; it has to be minted first.
    ``has_successor``
        A refine already converged this name onto a successor; the successor
        carries the concept forward and is the node to edit. This generalises
        the superseded-with-successor guard to a name whose stage label was
        reset out from under it by a reseed.
    ``origin='derived'``
        A structural parent is named by the grammar peel, and the enrich-parents
        pool lifts it out of 'pending'; it is not renamed by hand.
    no live source
        Nothing produced this name, so there is no identity to rename.
    no description
        The name was never composed. Steering generation with a hint is the
        repair; renaming an empty placeholder only mints a second empty one.
    """
    if (stage or "") not in _UNMINTED_NAME_STAGES:
        return (
            f"{sn_id!r} has name_stage={stage!r} — not eligible for rename "
            "(must be accepted/reviewed/exhausted/drafted, or superseded with "
            "no successor)"
        )
    if not stage:
        return (
            f"{sn_id!r} carries no name_stage — not eligible for rename (the "
            "refine hand-off compares against a concrete predecessor stage, so "
            "a bare node has to be minted before it can be renamed)"
        )
    if row.get("has_successor"):
        return (
            f"{sn_id!r} rests at name_stage={stage!r} but a successor was "
            "already refined from it — edit the successor instead"
        )
    if (row.get("origin") or "") == "derived":
        return (
            f"{sn_id!r} is a derived structural parent at name_stage={stage!r} "
            "— not eligible for rename (its name follows from the grammar peel "
            "over its children)"
        )
    if not row.get("has_live_source"):
        return (
            f"{sn_id!r} has name_stage={stage!r} and no live source — not "
            "eligible for rename (nothing produced this name, so there is no "
            "identity to rename)"
        )
    if not (row.get("description") or "").strip():
        return (
            f"{sn_id!r} has name_stage={stage!r} and no description — not "
            "eligible for rename (it was never composed; steer generation with "
            "a hint instead)"
        )
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
               sn.origin AS origin,
               sn.tags AS tags,
               coalesce(sn.chain_length, 0) AS chain_length,
               has_successor,
               EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(sn) } AS has_children,
               EXISTS {
                 MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                 WHERE coalesce(src.source_type, '') <> 'derived'
                   AND coalesce(src.status, '') IN $live_source_statuses
               } AS has_live_source
        """,
        live_source_statuses=sorted(_LIVE_SOURCE_STATUSES),
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
            sn.run_id            = $run_id,
            // A name-steering edit is new information: the evidence that
            // further refinement is futile was gathered without it, so the
            // rotation budget and its diagnosis are refunded. A docs-only
            // edit says nothing about the name axis and leaves both alone.
            sn.refine_attempts = CASE WHEN $name_hint IS NULL
                                      THEN sn.refine_attempts ELSE 0 END,
            sn.refine_stop_reason = CASE WHEN $name_hint IS NULL
                                         THEN sn.refine_stop_reason ELSE null END,
            sn.refine_stopped_at = CASE WHEN $name_hint IS NULL
                                        THEN sn.refine_stopped_at ELSE null END,
            sn.refine_collision_name = CASE
                WHEN $name_hint IS NULL
                THEN sn.refine_collision_name ELSE null END
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
    below threshold. ``sn approve`` attaches human-approved catalog wording
    with ``refine=False`` so the proposal is stamped ``edit_refine=false`` —
    a durable review-only marker recording that the wording must be scored
    as-is and never silently mutated. It does not alter the attach itself;
    the approval caller enforces the accept-or-quarantine outcome.
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
        # (``sn approve`` folding human-approved wording), stamp the touched
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


def _validate_kind_candidate(name: str, kind: str, unit: str | None) -> list[str]:
    """Return hard pinned-ISN findings for one proposed kind assignment."""
    try:
        from imas_standard_names.models import create_standard_name_entry
        from imas_standard_names.validation import run_semantic_checks

        from imas_codex.standard_names.kind_derivation import to_isn_kind

        candidate: dict[str, str] = {
            "name": name,
            "kind": to_isn_kind(kind),
        }
        if unit:
            candidate["unit"] = unit
        elif kind != "metadata":
            return [f"{kind} entry requires an authoritative unit"]
        entry = create_standard_name_entry(candidate, name_only=True)
        return [
            issue
            for issue in run_semantic_checks({name: entry})
            if " WARNING - " not in issue and " INFO - " not in issue
        ]
    except Exception as exc:  # ISN exposes parse and model-validation failures.
        return [str(exc)]


def reclassify_kind(
    name: str,
    kind: str,
    *,
    reason: str,
    override_edits: bool = False,
    apply: bool = False,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Repair one stored structural kind through a validated, audited CAS.

    Kind is mechanically derived from the unchanged identity, so callers may
    only request the value returned by :func:`derive_kind`. The default is a
    zero-write preview; ``apply=True`` is an explicit second step. Reviewed
    wording and lifecycle stages are preserved because neither changes, while
    catalog-owned fields retain the standard ``--override-edits`` protection.
    """
    from imas_codex.standard_names.protection import filter_protected

    name = (name or "").strip()
    requested_kind = (kind or "").strip()
    reason = (reason or "").strip()
    if not name:
        return {"ok": False, "reason": "a standard name is required"}
    if not reason:
        return {"ok": False, "reason": "--reason is mandatory for a kind repair"}

    derived_kind = derive_kind(name)
    if requested_kind != derived_kind:
        return {
            "ok": False,
            "reason": (
                f"requested kind {requested_kind!r} disagrees with "
                f"derive_kind({name!r})={derived_kind!r}"
            ),
        }

    owns_gc = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        rows = list(
            gc.query(
                """
                // KIND_RECLASSIFY_FETCH
                MATCH (n:StandardName {id: $id})
                RETURN n.kind AS kind, n.unit AS unit, n.origin AS origin,
                       n.name_stage AS name_stage, n.docs_stage AS docs_stage,
                       n.validation_status AS validation_status
                """,
                id=name,
            )
        )
        if not rows:
            return {"ok": False, "reason": f"standard name {name!r} not found"}
        before = dict(rows[0])

        protected_names = (
            {name}
            if before.get("origin") == "catalog_edit"
            or before.get("name_stage") == "approved"
            else set()
        )
        filtered, skipped = filter_protected(
            [{"id": name, "kind": requested_kind}],
            override=override_edits,
            protected_names=protected_names,
        )
        if skipped or "kind" not in filtered[0]:
            return {
                "ok": False,
                "reason": (
                    f"{name!r} has a catalog-protected kind; pass "
                    "--override-edits to authorize this exact field repair"
                ),
            }

        hard_findings = _validate_kind_candidate(
            name, requested_kind, before.get("unit")
        )
        if hard_findings:
            return {
                "ok": False,
                "reason": "pinned ISN rejected the proposed kind: "
                + "; ".join(hard_findings),
                "hard_findings": hard_findings,
            }

        result: dict[str, Any] = {
            "ok": True,
            "name": name,
            "from_kind": before.get("kind"),
            "to_kind": requested_kind,
            "unit": before.get("unit"),
            "name_stage": before.get("name_stage"),
            "docs_stage": before.get("docs_stage"),
            "validation_status": before.get("validation_status"),
            "dry_run": not apply,
            "noop": before.get("kind") == requested_kind,
            "hard_findings": [],
        }
        if not apply or result["noop"]:
            return result

        change_id = f"sn-change:{uuid.uuid4()}"
        changed_at = _now_iso()
        run_id = "sn-edit-kind-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        applied_rows = list(
            gc.query(
                """
                // KIND_RECLASSIFY_APPLY
                MATCH (n:StandardName {id: $id})
                WHERE n.kind = $from_kind
                  AND (n.unit = $unit
                       OR (n.unit IS NULL AND $unit IS NULL))
                  AND (n.name_stage = $name_stage
                       OR (n.name_stage IS NULL AND $name_stage IS NULL))
                  AND (n.docs_stage = $docs_stage
                       OR (n.docs_stage IS NULL AND $docs_stage IS NULL))
                  AND (n.validation_status = $validation_status
                       OR (n.validation_status IS NULL
                           AND $validation_status IS NULL))
                SET n.kind = $to_kind
                CREATE (change:StandardNameChange {
                  id: $change_id,
                  from_name: $id,
                  to_name: $id,
                  operation: 'reclassify_kind',
                  reason: $reason,
                  origin: 'catalog_edit',
                  run_id: $run_id,
                  changed_at: datetime($changed_at),
                  internal: true
                })
                MERGE (n)-[:HAS_INTERNAL_CHANGE]->(change)
                RETURN n.kind AS kind, n.unit AS unit,
                       n.name_stage AS name_stage, n.docs_stage AS docs_stage,
                       n.validation_status AS validation_status,
                       change.id AS change_id
                """,
                id=name,
                from_kind=before.get("kind"),
                to_kind=requested_kind,
                unit=before.get("unit"),
                name_stage=before.get("name_stage"),
                docs_stage=before.get("docs_stage"),
                validation_status=before.get("validation_status"),
                change_id=change_id,
                reason=reason,
                run_id=run_id,
                changed_at=changed_at,
            )
        )
        if len(applied_rows) != 1:
            return {
                "ok": False,
                "reason": "kind repair compare-and-set lost; graph state changed",
            }
        applied_row = dict(applied_rows[0])
        result.update(
            dry_run=False,
            change_id=applied_row["change_id"],
            unit=applied_row["unit"],
            name_stage=applied_row["name_stage"],
            docs_stage=applied_row["docs_stage"],
            validation_status=applied_row["validation_status"],
        )
        return result
    finally:
        if owns_gc:
            gc.close()


_FOLD_LIVE_STAGES = frozenset(
    {"pending", "drafted", "reviewed", "accepted", "approved", "refining"}
)
_FOLD_PREDECESSOR_STAGES = frozenset(
    {"pending", "drafted", "reviewed", "accepted", "exhausted"}
)
_FOLD_REASON = "fold parser-invalid duplicate into accepted authoritative identity"
_FOLD_RECEIPT_TYPE = "standard_name_identity_fold"
_FOLD_RECEIPT_SCHEMA = 1

_FOLD_SNAPSHOT_QUERY = """
// ATOMIC_FOLD_SNAPSHOT
MATCH (old:StandardName {id: $old_id}),
      (target:StandardName {id: $into_id})
CALL (old, target) {
  MATCH (source:StandardNameSource)
  WHERE source.produced_sn_id IN [old.id, target.id]
     OR EXISTS { (source)-[:PRODUCED_NAME]->(old) }
     OR EXISTS { (source)-[:PRODUCED_NAME]->(target) }
     OR EXISTS {
       MATCH (source)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
       WHERE EXISTS { (backing)-[:HAS_STANDARD_NAME]->(old) }
          OR EXISTS { (backing)-[:HAS_STANDARD_NAME]->(target) }
     }
  OPTIONAL MATCH (scalar_target:StandardName {id: source.produced_sn_id})
  WITH source, scalar_target
  RETURN collect({
    id: source.id,
    element_id: elementId(source),
    labels: labels(source),
    properties: properties(source),
    scalar_target: CASE WHEN scalar_target IS NULL THEN null ELSE {
      element_id: elementId(scalar_target),
      labels: labels(scalar_target),
      properties: properties(scalar_target),
      target_id: scalar_target.id,
      target_stage: scalar_target.name_stage
    } END,
    bindings: [(source)-[binding:PRODUCED_NAME]->(bound:StandardName) | {
      element_id: elementId(binding),
      properties: properties(binding),
      target_element_id: elementId(bound),
      target_labels: labels(bound),
      target_properties: properties(bound),
      target_id: bound.id,
      target_stage: bound.name_stage
    }],
    backing_refs: [(source)-[owner:FROM_DD_PATH|FROM_SIGNAL]->(backing) | {
      element_id: elementId(owner),
      properties: properties(owner),
      type: type(owner),
      backing_element_id: elementId(backing),
      backing_id: backing.id
    }]
  }) AS sources
}
CALL (old, target) {
  MATCH (backing)
  WHERE (backing:IMASNode OR backing:FacilitySignal)
    AND (
      EXISTS { (backing)-[:HAS_STANDARD_NAME]->(old) }
      OR EXISTS { (backing)-[:HAS_STANDARD_NAME]->(target) }
      OR EXISTS {
        MATCH (source:StandardNameSource)
              -[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
        WHERE source.produced_sn_id IN [old.id, target.id]
           OR EXISTS { (source)-[:PRODUCED_NAME]->(old) }
           OR EXISTS { (source)-[:PRODUCED_NAME]->(target) }
      }
    )
  RETURN collect({
    id: backing.id,
    element_id: elementId(backing),
    labels: labels(backing),
    properties: properties(backing),
    owners: [(owner:StandardNameSource)
             -[owner_link:FROM_DD_PATH|FROM_SIGNAL]->(backing) | {
      source_id: owner.id,
      source_element_id: elementId(owner),
      relationship_element_id: elementId(owner_link),
      relationship_properties: properties(owner_link),
      relationship_type: type(owner_link)
    }],
    projections: [(backing)-[projection:HAS_STANDARD_NAME]
                  ->(projected:StandardName) | {
      element_id: elementId(projection),
      properties: properties(projection),
      target_element_id: elementId(projected),
      target_labels: labels(projected),
      target_properties: properties(projected),
      target_id: projected.id,
      target_stage: projected.name_stage
    }],
    units: [(backing)-[unit_link:HAS_UNIT]->(unit:Unit) | {
      element_id: elementId(unit_link),
      properties: properties(unit_link),
      unit_element_id: elementId(unit),
      unit_labels: labels(unit),
      unit_id: unit.id,
      unit_properties: properties(unit)
    }]
  }) AS backings
}
CALL (old, target) {
  OPTIONAL MATCH (start)-[relationship]->(end)
  WHERE start = old OR start = target OR end = old OR end = target
  WITH relationship, start, end
  WHERE relationship IS NOT NULL
  RETURN collect(DISTINCT {
    element_id: elementId(relationship),
    type: type(relationship),
    start_element_id: elementId(startNode(relationship)),
    end_element_id: elementId(endNode(relationship)),
    start_id: start.id,
    end_id: end.id,
    start_labels: labels(start),
    end_labels: labels(end),
    other_element_id: CASE
      WHEN start = old OR start = target THEN elementId(end)
      ELSE elementId(start)
    END,
    properties: properties(relationship)
  }) AS relationships
}
CALL (old, target) {
  MATCH (review:StandardNameReview)
  WHERE review.standard_name_id IN [old.id, target.id]
     OR EXISTS { (old)-[:HAS_REVIEW]->(review) }
     OR EXISTS { (target)-[:HAS_REVIEW]->(review) }
  RETURN collect({
    element_id: elementId(review),
    labels: labels(review),
    properties: properties(review),
    owners: [(owner:StandardName)-[link:HAS_REVIEW]->(review) | {
      owner_id: owner.id,
      element_id: elementId(link),
      properties: properties(link)
    }]
  }) AS reviews
}
CALL (old, target) {
  OPTIONAL MATCH (owner:StandardName)-[link:DOCS_REVISION_OF]
                 ->(revision:DocsRevision)
  WHERE owner = old OR owner = target
  WITH link, revision
  WHERE link IS NOT NULL
  RETURN collect({
    element_id: elementId(revision),
    labels: labels(revision),
    properties: properties(revision),
    owners: [(owner:StandardName)-[owner_link:DOCS_REVISION_OF]->(revision) | {
      owner_id: owner.id,
      element_id: elementId(owner_link),
      properties: properties(owner_link)
    }]
  }) AS revisions
}
CALL (old, target) {
  OPTIONAL MATCH (owner:StandardName)-[link:HAS_INTERNAL_CHANGE]
                 ->(change:StandardNameChange)
  WHERE owner = old OR owner = target
  WITH link, change
  WHERE link IS NOT NULL
  RETURN collect(DISTINCT {
    element_id: elementId(change),
    labels: labels(change),
    properties: properties(change),
    owners: [(owner:StandardName)-[owner_link:HAS_INTERNAL_CHANGE]->(change) | {
      owner_id: owner.id,
      element_id: elementId(owner_link),
      properties: properties(owner_link)
    }]
  }) AS changes
}
WITH old, target, sources, backings, relationships, reviews, revisions, changes,
     [(old)-[link:HAS_UNIT]->(unit:Unit) | {
       element_id: elementId(link), properties: properties(link),
       unit_element_id: elementId(unit), unit_labels: labels(unit),
       unit_id: unit.id, unit_properties: properties(unit)
     }] AS old_units,
     [(target)-[link:HAS_UNIT]->(unit:Unit) | {
       element_id: elementId(link), properties: properties(link),
       unit_element_id: elementId(unit), unit_labels: labels(unit),
       unit_id: unit.id, unit_properties: properties(unit)
     }] AS target_units
RETURN elementId(old) AS old_element_id,
       elementId(target) AS target_element_id,
       labels(old) AS old_labels,
       labels(target) AS target_labels,
       properties(old) AS old_properties,
       properties(target) AS target_properties,
       EXISTS { (old)-[:REFINED_FROM*1..]->(target) } AS cycle,
       sources,
       backings,
       relationships,
       reviews,
       revisions,
       changes,
       old_units,
       target_units
"""

_FOLD_LOCK_QUERY = """
// ATOMIC_FOLD_LOCK
MATCH (participant)
WHERE elementId(participant) IN $element_ids AND participant.id IS NOT NULL
SET participant.id = participant.id
RETURN count(participant) AS locked
"""

_FOLD_EVENT_QUERY = """
// ATOMIC_FOLD_EVENT
MATCH (old:StandardName {id: $old_id}),
      (target:StandardName {id: $into_id})
WHERE elementId(old) = $old_element_id
  AND elementId(target) = $target_element_id
CREATE (change:StandardNameChange {
  id: $change_id,
  from_name: $old_id,
  to_name: $into_id,
  operation: 'fold_identity',
  reason: $receipt,
  origin: 'catalog_edit',
  run_id: $run_id,
  changed_at: datetime($changed_at),
  internal: true
})
MERGE (old)-[:HAS_INTERNAL_CHANGE]->(change)
MERGE (target)-[:HAS_INTERNAL_CHANGE]->(change)
RETURN change.id AS change_id
"""

_FOLD_SOURCE_MUTATION_QUERY = """
// ATOMIC_FOLD_MOVE_SOURCES
MATCH (target:StandardName {id: $into_id})
WHERE elementId(target) = $target_element_id
CALL (target) {
  UNWIND $sources AS expected
  MATCH (source:StandardNameSource {id: expected.id})
  WHERE elementId(source) = expected.element_id
  OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(bound:StandardName)
  WHERE elementId(binding) IN expected.remove_binding_element_ids
  WITH source, target, collect(binding) AS bindings
  FOREACH (binding IN bindings | DELETE binding)
  CREATE (source)-[:PRODUCED_NAME]->(target)
  SET source.produced_sn_id = target.id
  RETURN count(DISTINCT source) AS sources_moved
}
CALL (target) {
  UNWIND $backings AS expected
  MATCH (backing)
  WHERE elementId(backing) = expected.element_id
  OPTIONAL MATCH (backing)-[projection:HAS_STANDARD_NAME]
                 ->(bound:StandardName)
  WHERE elementId(projection) IN expected.remove_projection_element_ids
  WITH backing, target, expected, collect(projection) AS projections
  FOREACH (projection IN projections | DELETE projection)
  CREATE (backing)-[:HAS_STANDARD_NAME]->(target)
  SET backing.standard_name_id = CASE
    WHEN expected.has_standard_name_id THEN target.id
    ELSE backing.standard_name_id
  END
  RETURN count(DISTINCT backing) AS projections_moved
}
RETURN sources_moved, projections_moved
"""

_FOLD_NAME_MUTATION_QUERY = """
// ATOMIC_FOLD_MUTATE_NAMES
MATCH (old:StandardName {id: $old_id}),
      (target:StandardName {id: $into_id})
WHERE elementId(old) = $old_element_id
  AND elementId(target) = $target_element_id
SET old.superseded_from_stage = $predecessor_stage,
    old.name_stage = 'superseded',
    old.claim_token = null,
    old.claimed_at = null,
    old.source_paths = [],
    old.edit_status = CASE
      WHEN old.edit_status = 'open' THEN 'applied'
      ELSE old.edit_status
    END,
    target.source_paths = $target_paths,
    target.name_stage = coalesce($target_revived_stage, target.name_stage)
MERGE (target)-[:REFINED_FROM]->(old)
RETURN old.name_stage AS old_stage,
       old.superseded_from_stage AS predecessor_stage,
       target.name_stage AS target_stage
"""

_FOLD_POSTFLIGHT_QUERY = _FOLD_SNAPSHOT_QUERY.replace(
    "// ATOMIC_FOLD_SNAPSHOT", "// ATOMIC_FOLD_POSTFLIGHT", 1
)


class _FoldTransactionQuery:
    """Expose a caller-owned Neo4j transaction to attachment guards."""

    def __init__(self, transaction: Any) -> None:
        self._transaction = transaction

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in self._transaction.run(cypher, **params)]


def _fold_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _fold_json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_fold_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_fold_json_safe(item) for item in value), key=repr)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _fold_cas_value(value: Any, *, key: str = "") -> Any:
    """Encode graph state deterministically without erasing scalar types."""
    if isinstance(value, dict):
        return [
            "mapping",
            [
                [str(name), _fold_cas_value(item, key=str(name))]
                for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ],
        ]
    if isinstance(value, list | tuple):
        items = [_fold_cas_value(item) for item in value]
        if key == "labels" or value and all(isinstance(item, dict) for item in value):
            items.sort(
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
            )
        return [type(value).__name__, items]
    if isinstance(value, set | frozenset):
        items = [_fold_cas_value(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
        return [type(value).__name__, items]
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", repr(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    scalar_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if hasattr(value, "iso_format"):
        return [scalar_type, value.iso_format()]
    if hasattr(value, "isoformat"):
        return [scalar_type, value.isoformat()]
    return [scalar_type, str(value)]


def _fold_cas_signature(value: Any) -> str:
    encoded = json.dumps(
        _fold_cas_value(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fold_sort_key(value: Any) -> str:
    return json.dumps(_fold_json_safe(value), sort_keys=True, separators=(",", ":"))


def _fold_normalize(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(name): _fold_normalize(item, key=str(name))
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        normalized = [_fold_normalize(item) for item in value]
        if (
            key == "labels"
            or normalized
            and all(isinstance(item, dict) for item in normalized)
        ):
            normalized.sort(key=_fold_sort_key)
        return normalized
    return _fold_json_safe(value)


def _fold_snapshot(
    transaction: Any,
    old: str,
    into: str,
    *,
    query: str = _FOLD_SNAPSHOT_QUERY,
) -> dict[str, Any] | None:
    rows = list(
        transaction.run(
            query,
            old_id=old,
            into_id=into,
            live_stages=sorted(_FOLD_LIVE_STAGES),
        )
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise RuntimeError("fold identity pair is not unique")
    raw_snapshot = dict(rows[0])
    snapshot = _fold_normalize(raw_snapshot)
    snapshot["_cas_signature"] = _fold_cas_signature(raw_snapshot)
    snapshot["fold_events"] = [
        change
        for change in snapshot.get("changes") or []
        if (change.get("properties") or {}).get("operation") == "fold_identity"
        and (change.get("properties") or {}).get("from_name") == old
        and (change.get("properties") or {}).get("to_name") == into
    ]
    return snapshot


def _fold_binding_ids(source: dict[str, Any]) -> list[str]:
    return [
        binding["target_id"]
        for binding in source.get("bindings") or []
        if binding.get("target_id")
    ]


def _fold_backing_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        backing["element_id"]: backing
        for backing in snapshot.get("backings") or []
        if backing.get("element_id")
    }


def _fold_old_backings(
    snapshot: dict[str, Any], old_sources: list[dict[str, Any]], old: str
) -> list[dict[str, Any]]:
    referenced = {
        ref["backing_element_id"]
        for source in old_sources
        for ref in source.get("backing_refs") or []
        if ref.get("backing_element_id")
    }
    return [
        backing
        for backing in snapshot.get("backings") or []
        if backing.get("element_id") in referenced
        or old
        in {
            projection.get("target_id")
            for projection in backing.get("projections") or []
        }
    ]


def _fold_source_rows(snapshot: dict[str, Any], old: str) -> list[dict[str, Any]]:
    old_projected = {
        backing["element_id"]
        for backing in snapshot.get("backings") or []
        if old
        in {
            projection.get("target_id")
            for projection in backing.get("projections") or []
        }
    }
    return [
        source
        for source in snapshot.get("sources") or []
        if (source.get("properties") or {}).get("produced_sn_id") == old
        or old in _fold_binding_ids(source)
        or any(
            ref.get("backing_element_id") in old_projected
            for ref in source.get("backing_refs") or []
        )
    ]


def _fold_refusal(reason: str) -> dict[str, Any]:
    return {"ok": False, "reason": reason}


def _fold_unit_authority(
    properties: dict[str, Any], unit_edges: list[dict[str, Any]], label: str
) -> tuple[str | None, str | None]:
    if len(unit_edges) > 1:
        return None, f"{label} has multiple HAS_UNIT authorities"
    scalar = properties.get("unit")
    edge = unit_edges[0].get("unit_id") if unit_edges else None
    if scalar and edge:
        from imas_codex.units.dd_unit_exceptions import canonical_or_none

        scalar_canonical = canonical_or_none(str(scalar))
        edge_canonical = canonical_or_none(str(edge))
        agree = (
            scalar_canonical == edge_canonical
            if scalar_canonical is not None and edge_canonical is not None
            else str(scalar) == str(edge)
        )
        if not agree:
            return None, f"{label} scalar unit disagrees with HAS_UNIT authority"
    return (str(edge or scalar) if edge or scalar else None), None


def _derive_rename_unit(
    gc: GraphClient, sn_id: str, predecessor_unit: str | None
) -> str | None:
    """Return unanimous DD unit authority for a renamed successor.

    A rename changes the asserted quantity, so the predecessor's unit is not
    successor authority. Every DD source migrated by the rename is resolved
    through its backing node and cardinality-one ``HAS_UNIT`` edge. The rename
    refuses incomplete or conflicting DD authority rather than selecting one
    source arbitrarily. Names without DD sources retain their existing unit;
    their signal-derived unit policy is outside this DD-specific boundary.
    """
    rows = gc.query(
        """
        // EDIT_DERIVE_RENAME_UNIT
        MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->
              (:StandardName {id: $id})
        WHERE source.source_type = 'dd'
        OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(node:IMASNode)
        OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
        WITH source, node, collect(DISTINCT unit.id) AS relationship_units
        RETURN source.id AS source_id,
               node.id AS dd_path,
               node.unit AS dd_unit,
               relationship_units AS dd_relationship_units
        ORDER BY source.id, node.id
        """,
        id=sn_id,
    )
    if not rows:
        return predecessor_unit

    from imas_codex.units.dd_unit_exceptions import canonical_or_none

    authorities: list[tuple[str, str]] = []
    for row in rows:
        source_id = str(row.get("source_id") or "<unknown>")
        dd_path = row.get("dd_path")
        if not dd_path:
            raise ValueError(
                f"rename unit derivation refused: DD source {source_id!r} "
                "has no backing IMASNode"
            )
        unit, refusal = _fold_unit_authority(
            {"unit": row.get("dd_unit")},
            [{"unit_id": value} for value in (row.get("dd_relationship_units") or [])],
            f"DD source {source_id!r}",
        )
        if refusal:
            raise ValueError(f"rename unit derivation refused: {refusal}")
        if unit is None:
            raise ValueError(
                f"rename unit derivation refused: DD source {source_id!r} "
                "has no unit authority"
            )
        canonical = canonical_or_none(unit)
        authorities.append((canonical or f"raw:{unit}", unit))

    canonical_units = {canonical for canonical, _unit in authorities}
    if len(canonical_units) != 1:
        units = sorted({unit for _canonical, unit in authorities})
        raise ValueError(
            "rename unit derivation refused: DD source cohort disagrees on "
            f"unit authority: {units}"
        )
    return authorities[0][1]


def _fold_target_paths(
    snapshot: dict[str, Any],
    old_sources: list[dict[str, Any]],
    old_backings: list[dict[str, Any]],
    into: str,
) -> list[str]:
    old_source_ids = {source["id"] for source in old_sources}
    old_backing_ids = {backing["element_id"] for backing in old_backings}
    paths: set[str] = set()
    for backing in snapshot.get("backings") or []:
        projected = {
            projection.get("target_id")
            for projection in backing.get("projections") or []
        }
        if backing.get("element_id") not in old_backing_ids and into not in projected:
            continue
        labels = set(backing.get("labels") or [])
        if "IMASNode" in labels:
            paths.add("dd:" + backing["id"])
        elif "FacilitySignal" in labels:
            paths.add(backing["id"])
    for source in snapshot.get("sources") or []:
        if source["id"] not in old_source_ids and into not in _fold_binding_ids(source):
            continue
        properties = source.get("properties") or {}
        if (
            not source.get("backing_refs")
            and properties.get("source_type") == "derived"
        ):
            paths.add(properties.get("source_id") or source["id"])
    return sorted(path for path in paths if path)


def _fold_target_successor_names(
    relationships: list[dict[str, Any]], target_element_id: str
) -> set[str]:
    """Names with a direct REFINED_FROM edge onto ``target_element_id``."""
    return {
        relationship.get("start_id")
        for relationship in relationships
        if relationship.get("type") == "REFINED_FROM"
        and relationship.get("end_element_id") == target_element_id
    }


_FOLD_CHAIN_MAX_DEPTH = 64


def _fold_chain_walk(
    transaction: Any, snapshot: dict[str, Any], old: str, into: str
) -> tuple[bool, str | None]:
    """Walk ``into``'s REFINED_FROM descendants for a straight chain to ``old``.

    A chain of any depth whose every live descendant is ``old`` closes onto
    ``into`` as its own root identity — each hop is confirmed superseded
    before the walk descends through it, so an intermediate that is still
    live stops the walk exactly as a direct live successor would. A branch
    that leads to a different name anywhere along the way keeps the target
    spelling load-bearing, so the walk refuses instead of descending into it.

    Returns ``(True, None)`` when the chain closes on ``old``. Returns
    ``(False, reason)`` when a live successor — direct or found while
    descending — is not ``old``. Returns ``(False, None)`` when ``into`` has
    no successor lineage at all, leaving the caller's other free-identity
    checks (sources, parent, child) to decide.
    """
    element_id = snapshot["target_element_id"]
    relationships = snapshot.get("relationships") or []
    visited = {into}
    for _ in range(_FOLD_CHAIN_MAX_DEPTH):
        successors = _fold_target_successor_names(relationships, element_id)
        if successors == {old}:
            return True, None
        live = sorted(successors - {old})
        if len(successors) != 1 or not live:
            if live:
                return False, (
                    f"target {into!r} is superseded and has successor lineage: "
                    + ", ".join(live)
                )
            return False, None
        (candidate,) = live
        if candidate in visited:
            return False, None
        candidate_snapshot = _fold_snapshot(transaction, candidate, into)
        if candidate_snapshot is None or (
            candidate_snapshot["old_properties"].get("name_stage") != "superseded"
        ):
            return False, (
                f"target {into!r} is superseded and has successor lineage: {candidate}"
            )
        visited.add(candidate)
        element_id = candidate_snapshot["old_element_id"]
        relationships = candidate_snapshot.get("relationships") or []
    return False, f"target {into!r} successor lineage exceeds walk depth"


def _fold_tombstone_target_reason(
    transaction: Any, snapshot: dict[str, Any], old: str, into: str
) -> tuple[str | None, bool]:
    """Refuse a tombstoned fold target that is not a free identity.

    A tombstoned identity can be re-occupied only when nothing reads it: no
    recorded successor and no successor lineage other than a chain of any
    depth that closes back onto the name now being folded into it, no
    sources bound to or projected onto it, and neither a parent nor a child.
    A target whose only live descendant, at the end of that chain, is the
    name being folded is a straight-line refinement chain closing on itself,
    not a third-party identity being taken, so that case alone is admitted —
    provided every intermediate on the way is itself superseded, with no
    other lineage of its own. Any other live descendant, or a chain that
    passes through a still-live intermediate, makes the spelling
    load-bearing, so the fold refuses and the caller folds into whatever
    holds the live meaning instead.

    Returns ``(reason, closing_straight_chain)``.
    """
    target_properties = snapshot["target_properties"]
    successor = target_properties.get("superseded_by")
    if successor:
        return (
            f"target {into!r} is superseded and records successor {successor!r} — "
            "fold into the successor instead"
        ), False
    closing_straight_chain, chain_reason = _fold_chain_walk(
        transaction, snapshot, old, into
    )
    if chain_reason:
        return chain_reason, False
    target_sources = _fold_source_rows(snapshot, into)
    if target_sources:
        return (
            f"target {into!r} is superseded and still carries "
            f"{len(target_sources)} source(s)"
        ), closing_straight_chain
    target_element_id = snapshot["target_element_id"]
    for relationship in snapshot.get("relationships") or []:
        if relationship.get("type") != "HAS_PARENT":
            continue
        if relationship.get("start_element_id") == target_element_id:
            return (
                f"target {into!r} is superseded and still has parent "
                f"{relationship.get('end_id')!r}"
            ), closing_straight_chain
        if relationship.get("end_element_id") == target_element_id:
            return (
                f"target {into!r} is superseded and still has child "
                f"{relationship.get('start_id')!r}"
            ), closing_straight_chain
    return None, closing_straight_chain


#: A recorded pre-tombstone stage that revival caps before re-entering the
#: pipeline: acceptance is a gate the review pool clears, never a side
#: effect of re-occupying a dead spelling, so a target that was 'accepted'
#: right before it was superseded is revived one hop short of that, at
#: 'reviewed' — eligible for refine_name pickup, exactly the pipeline entry
#: an ordinary below-threshold review outcome would leave it at.
_FOLD_REVIVAL_STAGE_CAP = {"accepted": "reviewed"}


def _fold_revival_stage(target_properties: dict[str, Any]) -> str:
    """The name_stage a tombstoned fold target is revived to.

    A fold onto a free tombstone re-occupies a dead identity rather than
    leaving the carried source's data-dictionary path with no live standard
    name at all. ``superseded_from_stage`` already records the stage the
    target held immediately before it was superseded, so revival restores
    exactly that — the state that was lost — capped through
    :data:`_FOLD_REVIVAL_STAGE_CAP` so revival always re-enters the review
    pipeline rather than granting acceptance outright. A target tombstoned
    before this field existed has no recorded prior stage; it revives to
    'drafted', the ordinary unreviewed entry point.
    """
    stage = target_properties.get("superseded_from_stage") or "drafted"
    return _FOLD_REVIVAL_STAGE_CAP.get(stage, stage)


def _fold_guard_reason(
    transaction: Any, snapshot: dict[str, Any], old: str, into: str
) -> str | None:
    old_properties = snapshot["old_properties"]
    target_properties = snapshot["target_properties"]
    old_stage = old_properties.get("name_stage")
    if old_stage != "superseded" and old_stage not in _FOLD_PREDECESSOR_STAGES:
        return f"name {old!r} has unsupported predecessor stage {old_stage!r}"
    target_stage = target_properties.get("name_stage")
    closing_straight_chain = False
    if target_stage == "superseded":
        tombstone_reason, closing_straight_chain = _fold_tombstone_target_reason(
            transaction, snapshot, old, into
        )
        if tombstone_reason:
            return tombstone_reason
    elif target_stage != "accepted":
        return f"target {into!r} is name_stage={target_stage!r}, not 'accepted'"
    elif target_properties.get("validation_status") != "valid":
        return (
            f"target {into!r} is validation_status="
            f"{target_properties.get('validation_status')!r}, not 'valid'"
        )
    if snapshot.get("cycle") and not closing_straight_chain:
        return (
            f"{old!r} already descends from {into!r} (REFINED_FROM cycle) — cannot fold"
        )

    successors = [
        relationship
        for relationship in snapshot.get("relationships") or []
        if relationship.get("type") == "REFINED_FROM"
        and relationship.get("end_element_id") == snapshot["old_element_id"]
    ]
    direct = [
        relationship
        for relationship in successors
        if relationship.get("start_element_id") == snapshot["target_element_id"]
    ]
    if len(direct) > 1:
        return f"name {old!r} has duplicate target successor lineage"
    if any(relationship not in direct for relationship in successors):
        return f"name {old!r} has another successor lineage; fold is ambiguous"

    for label, properties in (("old", old_properties), ("target", target_properties)):
        if (
            properties.get("claim_token") is not None
            or properties.get("claimed_at") is not None
        ):
            return f"{label} name is actively claimed"
    parseable, detail = _isn_round_trip_ok(into)
    if not parseable:
        return f"target {into!r} fails strict ISN parse/round-trip: {detail}"
    target_unit, unit_reason = _fold_unit_authority(
        target_properties, snapshot.get("target_units") or [], f"target {into!r}"
    )
    if unit_reason:
        return unit_reason

    for source in snapshot.get("sources") or []:
        properties = source.get("properties") or {}
        if (
            properties.get("claim_token") is not None
            or properties.get("claimed_at") is not None
        ):
            return f"source {source['id']!r} is actively claimed"
        if (
            properties.get("source_type") in {"dd", "signals"}
            and len(source.get("backing_refs") or []) != 1
        ):
            return f"source {source['id']!r} has ambiguous backing cardinality"

    old_sources = _fold_source_rows(snapshot, old)
    old_backings = _fold_old_backings(snapshot, old_sources, old)
    allowed = {old, into}
    for source in old_sources:
        scalar_target = source.get("scalar_target") or {}
        if (
            scalar_target.get("target_id") not in allowed
            and scalar_target.get("target_stage") in _FOLD_LIVE_STAGES
        ):
            return (
                f"source {source['id']!r} has a third live scalar target: "
                f"{scalar_target['target_id']}"
            )
        third_live = sorted(
            {
                binding["target_id"]
                for binding in source.get("bindings") or []
                if binding.get("target_id") not in allowed
                and binding.get("target_stage") in _FOLD_LIVE_STAGES
            }
        )
        if third_live:
            return f"source {source['id']!r} has third live bindings: " + ", ".join(
                third_live
            )
    source_ids = {source["id"] for source in snapshot.get("sources") or []}
    for backing in old_backings:
        owners = backing.get("owners") or []
        if len(owners) != 1 or owners[0].get("source_id") not in source_ids:
            return f"backing {backing['id']!r} has ambiguous owner cardinality"
        third_live = sorted(
            {
                projection["target_id"]
                for projection in backing.get("projections") or []
                if projection.get("target_id") not in allowed
                and projection.get("target_stage") in _FOLD_LIVE_STAGES
            }
        )
        if third_live:
            return (
                f"backing {backing['id']!r} has third live projections: "
                + ", ".join(third_live)
            )
        _, backing_unit_reason = _fold_unit_authority(
            backing.get("properties") or {},
            backing.get("units") or [],
            f"backing {backing['id']!r}",
        )
        if backing_unit_reason:
            return backing_unit_reason

    existing_paths = sorted(
        backing["id"]
        for backing in snapshot.get("backings") or []
        if "IMASNode" in set(backing.get("labels") or [])
        and into
        in {
            projection.get("target_id")
            for projection in backing.get("projections") or []
        }
    )
    from imas_codex.standard_names.workers import _is_attachment_consistent

    backings = _fold_backing_map(snapshot)
    for source in old_sources:
        for reference in source.get("backing_refs") or []:
            backing = backings[reference["backing_element_id"]]
            if "IMASNode" not in set(backing.get("labels") or []):
                continue
            dd_unit, _ = _fold_unit_authority(
                backing.get("properties") or {},
                backing.get("units") or [],
                f"backing {backing['id']!r}",
            )
            ok, reason = _is_attachment_consistent(
                backing["id"],
                into,
                existing_sources=tuple(existing_paths),
                dd_unit=dd_unit,
                sn_unit=target_unit,
            )
            if not ok:
                return f"source {source['id']!r} attachment refused: {reason}"
            if backing["id"] not in existing_paths:
                existing_paths.append(backing["id"])
    return None


def _fold_relationship_semantics(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in {"element_id", "other_element_id"}
    }


def _fold_link_semantics(
    records: list[dict[str, Any]], *, node_element_id: bool = False
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        item = deepcopy(record)
        if not node_element_id:
            item.pop("element_id", None)
        for nested_key in (
            "owners",
            "bindings",
            "backing_refs",
            "projections",
            "units",
        ):
            for nested in item.get(nested_key) or []:
                nested.pop("element_id", None)
                nested.pop("relationship_element_id", None)
        result.append(item)
    return _fold_normalize(result)


def _fold_receipt_summary(reason: str) -> dict[str, Any] | None:
    try:
        receipt = json.loads(reason)
    except (TypeError, ValueError):
        return None
    return {
        "receipt_type": receipt.get("receipt_type"),
        "schema_version": receipt.get("schema_version"),
        "run_id": receipt.get("run_id"),
        "old_id": receipt.get("old_id"),
        "into_id": receipt.get("into_id"),
    }


def _fold_verification_state(
    snapshot: dict[str, Any], *, fold_change_id: str | None = None
) -> dict[str, Any]:
    changes = []
    for change in snapshot.get("changes") or []:
        item = deepcopy(change)
        item.pop("element_id", None)
        for owner in item.get("owners") or []:
            owner.pop("element_id", None)
        properties = item.get("properties") or {}
        if properties.get("id") == fold_change_id:
            properties["reason"] = _fold_receipt_summary(properties.get("reason") or "")
        changes.append(item)
    excluded = {
        "PRODUCED_NAME",
        "HAS_STANDARD_NAME",
        "REFINED_FROM",
        "HAS_REVIEW",
        "DOCS_REVISION_OF",
        "HAS_INTERNAL_CHANGE",
        "HAS_UNIT",
    }
    return _fold_normalize(
        {
            "names": {
                "old": snapshot["old_properties"],
                "target": snapshot["target_properties"],
            },
            "sources": _fold_link_semantics(
                snapshot.get("sources") or [], node_element_id=True
            ),
            "backings": _fold_link_semantics(
                snapshot.get("backings") or [], node_element_id=True
            ),
            "lineage": [
                _fold_relationship_semantics(relationship)
                for relationship in snapshot.get("relationships") or []
                if relationship.get("type") == "REFINED_FROM"
            ],
            "other_relationships": [
                _fold_relationship_semantics(relationship)
                for relationship in snapshot.get("relationships") or []
                if relationship.get("type") not in excluded
            ],
            "reviews": _fold_link_semantics(
                snapshot.get("reviews") or [], node_element_id=True
            ),
            "revisions": _fold_link_semantics(
                snapshot.get("revisions") or [], node_element_id=True
            ),
            "changes": changes,
            "old_units": _fold_link_semantics(snapshot.get("old_units") or []),
            "target_units": _fold_link_semantics(snapshot.get("target_units") or []),
        }
    )


def _fold_expected_state(
    snapshot: dict[str, Any],
    old: str,
    into: str,
    predecessor_stage: str,
    old_sources: list[dict[str, Any]],
    old_backings: list[dict[str, Any]],
    target_paths: list[str],
    *,
    change_id: str,
    run_id: str,
    changed_at: str,
    include_domain_mutation: bool,
    target_revived_stage: str | None = None,
) -> dict[str, Any]:
    expected = _fold_verification_state(snapshot)
    # A tombstoned target is revived to target_revived_stage (computed by
    # the caller only when the snapshot read it as 'superseded'); any other
    # target keeps the stage the snapshot read — the fold never touches an
    # already-live target's lifecycle.
    target_stage = target_revived_stage or snapshot["target_properties"].get(
        "name_stage"
    )
    old_source_ids = {source["id"] for source in old_sources}
    old_backing_ids = {backing["element_id"] for backing in old_backings}
    if include_domain_mutation:
        old_properties = expected["names"]["old"]
        old_properties.update(
            {
                "name_stage": "superseded",
                "superseded_from_stage": predecessor_stage,
                "source_paths": [],
            }
        )
        old_properties.pop("claim_token", None)
        old_properties.pop("claimed_at", None)
        if old_properties.get("edit_status") == "open":
            old_properties["edit_status"] = "applied"
        expected["names"]["target"]["source_paths"] = target_paths
        if target_revived_stage is not None:
            expected["names"]["target"]["name_stage"] = target_revived_stage
        target_reference = {
            "target_element_id": snapshot["target_element_id"],
            "target_labels": snapshot["target_labels"],
            "target_properties": expected["names"]["target"],
            "target_id": into,
            "target_stage": target_stage,
        }
        for source in expected["sources"]:
            if (source.get("scalar_target") or {}).get("target_id") == into:
                source["scalar_target"] = {
                    "element_id": snapshot["target_element_id"],
                    "labels": snapshot["target_labels"],
                    "properties": expected["names"]["target"],
                    "target_id": into,
                    "target_stage": target_stage,
                }
            for binding in source["bindings"]:
                if binding.get("target_id") == into:
                    binding.update(target_reference)
            if source["id"] not in old_source_ids:
                continue
            source["properties"]["produced_sn_id"] = into
            source["scalar_target"] = {
                "element_id": snapshot["target_element_id"],
                "labels": snapshot["target_labels"],
                "properties": expected["names"]["target"],
                "target_id": into,
                "target_stage": target_stage,
            }
            source["bindings"] = [
                binding
                for binding in source["bindings"]
                if binding.get("target_id") not in {old, into}
            ] + [{"properties": {}, **target_reference}]
        for backing in expected["backings"]:
            for projection in backing["projections"]:
                if projection.get("target_id") == into:
                    projection.update(target_reference)
            if backing["element_id"] not in old_backing_ids:
                continue
            if "standard_name_id" in backing["properties"]:
                backing["properties"]["standard_name_id"] = into
            backing["projections"] = [
                projection
                for projection in backing["projections"]
                if projection.get("target_id") not in {old, into}
            ] + [{"properties": {}, **target_reference}]
        direct_lineage = [
            relationship
            for relationship in expected["lineage"]
            if relationship.get("start_id") == into
            and relationship.get("end_id") == old
        ]
        if not direct_lineage:
            expected["lineage"].append(
                {
                    "type": "REFINED_FROM",
                    "start_element_id": snapshot["target_element_id"],
                    "end_element_id": snapshot["old_element_id"],
                    "start_id": into,
                    "end_id": old,
                    "start_labels": snapshot["target_labels"],
                    "end_labels": snapshot["old_labels"],
                    "properties": {},
                }
            )
    event = {
        "labels": ["StandardNameChange"],
        "properties": {
            "id": change_id,
            "from_name": old,
            "to_name": into,
            "operation": "fold_identity",
            "reason": {
                "receipt_type": _FOLD_RECEIPT_TYPE,
                "schema_version": _FOLD_RECEIPT_SCHEMA,
                "run_id": run_id,
                "old_id": old,
                "into_id": into,
            },
            "origin": "catalog_edit",
            "run_id": run_id,
            "changed_at": changed_at,
            "internal": True,
        },
        "owners": [
            {"owner_id": old, "properties": {}},
            {"owner_id": into, "properties": {}},
        ],
    }
    expected["changes"].append(event)
    return _fold_normalize(expected)


def _fold_receipt(
    snapshot: dict[str, Any],
    old: str,
    into: str,
    predecessor_stage: str,
    old_sources: list[dict[str, Any]],
    old_backings: list[dict[str, Any]],
    target_paths: list[str],
    *,
    change_id: str,
    run_id: str,
    changed_at: str,
    target_revived_stage: str | None = None,
) -> tuple[str, dict[str, Any]]:
    source_ids = sorted(source["id"] for source in old_sources)
    backing_ids = sorted(backing["id"] for backing in old_backings)
    receipt_snapshot = deepcopy(snapshot)
    receipt_snapshot.pop("_cas_signature", None)
    receipt = {
        "receipt_type": _FOLD_RECEIPT_TYPE,
        "schema_version": _FOLD_RECEIPT_SCHEMA,
        "mechanism": _FOLD_REASON,
        "run_id": run_id,
        "change_id": change_id,
        "old_id": old,
        "into_id": into,
        "predecessor_stage": predecessor_stage,
        "source_ids": source_ids,
        "backing_ids": backing_ids,
        "source_count": len(source_ids),
        "projection_count": len(backing_ids),
        "before": receipt_snapshot,
        "expected_after": _fold_expected_state(
            snapshot,
            old,
            into,
            predecessor_stage,
            old_sources,
            old_backings,
            target_paths,
            change_id=change_id,
            run_id=run_id,
            changed_at=changed_at,
            include_domain_mutation=True,
            target_revived_stage=target_revived_stage,
        ),
    }
    return json.dumps(receipt, sort_keys=True, separators=(",", ":")), receipt


def _fold_participant_ids(snapshot: dict[str, Any]) -> list[str]:
    participants = {snapshot["old_element_id"], snapshot["target_element_id"]}
    for key in ("sources", "backings", "reviews", "revisions", "changes"):
        participants.update(
            record["element_id"]
            for record in snapshot.get(key) or []
            if record.get("element_id")
        )
    participants.update(
        relationship["other_element_id"]
        for relationship in snapshot.get("relationships") or []
        if relationship.get("other_element_id")
    )
    for source in snapshot.get("sources") or []:
        scalar_target = source.get("scalar_target") or {}
        if scalar_target.get("element_id"):
            participants.add(scalar_target["element_id"])
        for binding in source.get("bindings") or []:
            if binding.get("target_element_id"):
                participants.add(binding["target_element_id"])
        for reference in source.get("backing_refs") or []:
            if reference.get("backing_element_id"):
                participants.add(reference["backing_element_id"])
    for backing in snapshot.get("backings") or []:
        for owner in backing.get("owners") or []:
            if owner.get("source_element_id"):
                participants.add(owner["source_element_id"])
        for projection in backing.get("projections") or []:
            if projection.get("target_element_id"):
                participants.add(projection["target_element_id"])
        for unit in backing.get("units") or []:
            if unit.get("unit_element_id"):
                participants.add(unit["unit_element_id"])
    for key in ("old_units", "target_units"):
        for unit in snapshot.get(key) or []:
            if unit.get("unit_element_id"):
                participants.add(unit["unit_element_id"])
    return sorted(participants)


def _fold_idempotent_result(
    snapshot: dict[str, Any], old: str, into: str, *, dry_run: bool
) -> dict[str, Any]:
    events = snapshot.get("fold_events") or []
    if len(events) != 1:
        return _fold_refusal("superseded identity has missing or ambiguous fold ledger")
    event = events[0]
    properties = event.get("properties") or {}
    try:
        receipt = json.loads(properties.get("reason") or "")
    except (TypeError, ValueError):
        return _fold_refusal("superseded identity has an unreadable fold receipt")
    required = {
        "receipt_type",
        "schema_version",
        "mechanism",
        "run_id",
        "change_id",
        "old_id",
        "into_id",
        "predecessor_stage",
        "source_ids",
        "backing_ids",
        "source_count",
        "projection_count",
        "before",
        "expected_after",
    }
    if set(receipt) != required:
        return _fold_refusal("superseded identity fold receipt has schema drift")
    if (
        receipt.get("receipt_type") != _FOLD_RECEIPT_TYPE
        or receipt.get("schema_version") != _FOLD_RECEIPT_SCHEMA
        or receipt.get("mechanism") != _FOLD_REASON
        or receipt.get("change_id") != properties.get("id")
        or receipt.get("run_id") != properties.get("run_id")
    ):
        return _fold_refusal("superseded identity fold receipt has typed-ledger drift")
    if receipt.get("old_id") != old or receipt.get("into_id") != into:
        return _fold_refusal(
            "superseded identity fold receipt targets a different pair"
        )
    if (
        len(receipt["source_ids"]) != receipt["source_count"]
        or len(receipt["backing_ids"]) != receipt["projection_count"]
    ):
        return _fold_refusal("fold receipt cardinalities are inconsistent")
    current = _fold_verification_state(snapshot, fold_change_id=properties["id"])
    if current != receipt.get("expected_after"):
        return _fold_refusal("superseded identity graph drifted from its receipt")
    return {
        "ok": True,
        "old_id": old,
        "into_id": into,
        "old_prior_stage": receipt["predecessor_stage"],
        "already_superseded": True,
        "sources_carried": receipt["source_count"],
        "sources_would_strand": 0,
        "projections_carried": receipt["projection_count"],
        "attachments_rejected": 0,
        "attachments_detached": 0,
        "change_id": properties["id"],
        "run_id": properties["run_id"],
        "dry_run": dry_run,
    }


@retry_on_deadlock()
def supersede_into(
    old: str,
    into: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Atomically fold one identity into an accepted authoritative target.

    The transaction snapshots every identity-bearing scalar and relationship,
    writes a typed before/expected-after receipt, locks the snapshotted nodes by
    assigning their immutable identifiers to themselves, rechecks the complete
    state, applies only the declared fold fields and edges, then compares the
    complete semantic post-state with the receipt before commit.
    """
    old = (old or "").strip()
    into = (into or "").strip()
    if not old or not into:
        return _fold_refusal("both old and target names are required")
    if old == into:
        return _fold_refusal("old and target are the same name")

    with GraphClient() as graph:
        with graph.session() as session:
            transaction = session.begin_transaction()
            try:
                snapshot = _fold_snapshot(transaction, old, into)
                if snapshot is None:
                    transaction.rollback()
                    return _fold_refusal(
                        f"name {old!r} or target {into!r} was not found"
                    )
                guard_reason = _fold_guard_reason(transaction, snapshot, old, into)
                if guard_reason:
                    transaction.rollback()
                    return _fold_refusal(guard_reason)
                if snapshot["old_properties"].get("name_stage") == "superseded":
                    result = _fold_idempotent_result(
                        snapshot, old, into, dry_run=dry_run
                    )
                    transaction.rollback()
                    return result

                old_sources = _fold_source_rows(snapshot, old)
                old_backings = _fold_old_backings(snapshot, old_sources, old)
                predecessor_stage = snapshot["old_properties"]["name_stage"]
                target_revived_stage = (
                    _fold_revival_stage(snapshot["target_properties"])
                    if snapshot["target_properties"].get("name_stage") == "superseded"
                    else None
                )
                target_paths = _fold_target_paths(
                    snapshot, old_sources, old_backings, into
                )
                change_id = f"sn-change:{uuid.uuid4()}"
                run_id = f"sn-fold:{uuid.uuid4()}"
                changed_at = DateTime.from_native(datetime.now(UTC)).iso_format()
                receipt_text, receipt = _fold_receipt(
                    snapshot,
                    old,
                    into,
                    predecessor_stage,
                    old_sources,
                    old_backings,
                    target_paths,
                    change_id=change_id,
                    run_id=run_id,
                    changed_at=changed_at,
                    target_revived_stage=target_revived_stage,
                )
                would_strand = sum(
                    1
                    for source in old_sources
                    if all(
                        binding.get("target_id") == old
                        or binding.get("target_stage") not in _FOLD_LIVE_STAGES
                        for binding in source.get("bindings") or []
                    )
                )
                result = {
                    "ok": True,
                    "old_id": old,
                    "into_id": into,
                    "old_prior_stage": predecessor_stage,
                    "already_superseded": False,
                    "sources_carried": receipt["source_count"],
                    "sources_would_strand": would_strand,
                    "projections_carried": receipt["projection_count"],
                    "attachments_rejected": 0,
                    "attachments_detached": 0,
                    "run_id": run_id,
                    "dry_run": dry_run,
                }
                if dry_run:
                    result["mutation_plan"] = {
                        "predecessor_stage": predecessor_stage,
                        "source_ids": receipt["source_ids"],
                        "backing_ids": receipt["backing_ids"],
                        "target_paths": target_paths,
                        "lineage": {"from": into, "to": old},
                        "change_operation": "fold_identity",
                        "run_id": run_id,
                    }
                    transaction.rollback()
                    return result

                participants = _fold_participant_ids(snapshot)
                lock_rows = list(
                    transaction.run(_FOLD_LOCK_QUERY, element_ids=participants)
                )
                locked = int(dict(lock_rows[0]).get("locked") or 0) if lock_rows else 0
                if locked != len(participants):
                    raise RuntimeError("fold participant set changed before locking")
                locked_snapshot = _fold_snapshot(transaction, old, into)
                if locked_snapshot is None or locked_snapshot.get(
                    "_cas_signature"
                ) != snapshot.get("_cas_signature"):
                    raise RuntimeError(
                        "fold graph state changed after preflight snapshot"
                    )

                event_rows = list(
                    transaction.run(
                        _FOLD_EVENT_QUERY,
                        old_id=old,
                        into_id=into,
                        old_element_id=snapshot["old_element_id"],
                        target_element_id=snapshot["target_element_id"],
                        change_id=change_id,
                        run_id=run_id,
                        receipt=receipt_text,
                        changed_at=changed_at,
                    )
                )
                if not event_rows:
                    raise RuntimeError(
                        "fold identity pair changed before event creation"
                    )
                if len(event_rows) != 1:
                    raise RuntimeError("fold event cardinality was not exactly one")

                from imas_codex.standard_names.attachment_audit import (
                    guard_source_pairings,
                )

                guarded = guard_source_pairings(
                    _FoldTransactionQuery(transaction),
                    into,
                    [source["id"] for source in old_sources],
                )
                if guarded.rejected or set(guarded.accepted_source_ids) != {
                    source["id"] for source in old_sources
                }:
                    detail = "; ".join(item.reason for item in guarded.rejected)
                    raise RuntimeError(
                        "attachment authority changed after fold locks: "
                        + (detail or "source set was not admitted")
                    )

                source_rows = list(
                    transaction.run(
                        _FOLD_SOURCE_MUTATION_QUERY,
                        old_id=old,
                        into_id=into,
                        target_element_id=snapshot["target_element_id"],
                        sources=[
                            {
                                "id": source["id"],
                                "element_id": source["element_id"],
                                "remove_binding_element_ids": [
                                    binding["element_id"]
                                    for binding in source.get("bindings") or []
                                    if binding.get("target_id") in {old, into}
                                ],
                            }
                            for source in old_sources
                        ],
                        backings=[
                            {
                                "element_id": backing["element_id"],
                                "has_standard_name_id": "standard_name_id"
                                in backing["properties"],
                                "remove_projection_element_ids": [
                                    projection["element_id"]
                                    for projection in backing.get("projections") or []
                                    if projection.get("target_id") in {old, into}
                                ],
                            }
                            for backing in old_backings
                        ],
                    )
                )
                if len(source_rows) != 1:
                    raise RuntimeError("fold source migration returned no receipt")
                source_result = dict(source_rows[0])
                moved_sources = int(source_result.get("sources_moved") or 0)
                moved_projections = int(source_result.get("projections_moved") or 0)
                if moved_sources != receipt["source_count"]:
                    raise RuntimeError("fold source set changed during migration")
                if moved_projections != receipt["projection_count"]:
                    raise RuntimeError("fold projection set changed during migration")

                name_rows = list(
                    transaction.run(
                        _FOLD_NAME_MUTATION_QUERY,
                        old_id=old,
                        into_id=into,
                        old_element_id=snapshot["old_element_id"],
                        target_element_id=snapshot["target_element_id"],
                        predecessor_stage=predecessor_stage,
                        target_paths=target_paths,
                        target_revived_stage=target_revived_stage,
                    )
                )
                if len(name_rows) != 1:
                    raise RuntimeError("fold name state changed during mutation")

                post_snapshot = _fold_snapshot(
                    transaction, old, into, query=_FOLD_POSTFLIGHT_QUERY
                )
                if (
                    post_snapshot is None
                    or _fold_verification_state(post_snapshot, fold_change_id=change_id)
                    != receipt["expected_after"]
                ):
                    raise RuntimeError("fold postflight exact-state proof did not hold")
                transaction.commit()
                result["change_id"] = change_id
                result["receipt_counts"] = {
                    "sources": moved_sources,
                    "projections": moved_projections,
                    "lineage": 1,
                    "changes": 1,
                }
                return result
            except BaseException:
                with suppress(Exception):
                    transaction.rollback()
                raise


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

    # 3. Shared-base guard (leaf targets only — a
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

            if scope == EditScope.family.value:
                return _blocked(
                    target,
                    "rename",
                    "name",
                    scope,
                    f"renaming the shared segment {old_part!r} reaches "
                    f"{sib_count} sibling(s) under parent {parent_id!r}; a leaf "
                    "edit cannot be promoted to its parent automatically — target "
                    "the parent explicitly with a semantics-preserving cohort",
                )
            actions.append(
                f"keeping the base change local to leaf {target!r}; "
                f"{sib_count} sibling(s) under {parent_id!r} remain untouched"
            )

    # 4. Eligible-stage check — applies to whichever node is actually being
    #    refined (the mapped parent, or the target itself).
    root_row = target_row

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
        refusal = _stranded_rename_refusal(refine_root_old, root_stage, root_row)
        if refusal is not None:
            return _blocked(
                target,
                "rename",
                "name",
                scope,
                refusal,
                extra_actions=actions,
            )
        actions.append(
            f"{refine_root_old!r} is stranded at name_stage={root_stage!r} with "
            "live sources — admitting the rename as its only repair vehicle"
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

    successor_unit = _derive_rename_unit(gc, refine_root_old, root_row.get("unit"))

    if dry_run:
        actions.append(
            f"[dry-run] would rename {refine_root_old!r} → {refine_root_new!r}"
            f" ({len(cascade_planned)} descendant(s) would then await "
            f"{refine_root_new!r} reaching accepted)"
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
    result = persist_refined_name(
        old_name=refine_root_old,
        new_name=refine_root_new,
        description=root_row.get("description") or "",
        kind=derive_kind(refine_root_new),
        unit=successor_unit,
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
        expected_old_stage=root_stage,
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
    successor_row = {**root_row, "unit": successor_unit}
    _stamp_successor_validation(gc, successor, successor_row)

    actions.append(
        f"renamed {refine_root_old!r} → {successor!r}, entering name review "
        f"(edit_status=open, run_id={run_id})"
    )
    if cascade_planned:
        actions.append(
            f"{len(cascade_planned)} descendant(s) unchanged and deferred — "
            f"they are renamed only once {successor!r} reaches accepted; if it "
            "is withheld or exhausted they keep their current ids"
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

    if axis in ("name", "both") and name_stage in ("accepted", "exhausted"):
        return _blocked(
            target,
            "hint",
            axis,
            scope,
            f"{target!r} is name_stage={name_stage!r} and cannot re-enter name "
            "generation from a hint. Use `--rename` to propose the complete "
            "replacement name. For an exhausted but otherwise sound name, use "
            "`sn rescore` to request a fresh review of the same name.",
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
    inline review clears exactly the pool's gates — there is no
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


def rescore_name(
    sn_id: str,
    *,
    cost_limit: float = 1.0,
    stage_only: bool = False,
    min_score: float | None = None,
    rotation_cap: int | None = None,
    pending_fn: Any | None = None,
    dry_run: bool = False,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Recover a stranded name and re-score it with a fresh review quorum.

    Reverts an ``exhausted`` / ``reviewed`` name to ``'drafted'`` (stamped with
    a fresh scope run_id) and then — unless *stage_only* — runs the review
    pipeline scoped to exactly that name, so the operator gets a fresh
    score/outcome back rather than a queue state. This is the ``sn rescore``
    backend and mirrors :func:`run_inline_review`'s scoped pattern.

    ``stage_only=True`` performs only the drafted transition (no review) — the
    escape hatch for when the embedding service is down; a later ``sn run``
    picks the drafted name up. ``dry_run=True`` reports the intended transition
    without writing.

    Returns a dict with ``ok`` (bool) and, on success, ``prior_stage``,
    ``run_id``, ``reviewed`` (bool), and ``outcome`` (an
    :class:`InlineReviewOutcome` or ``None`` when not reviewed). On refusal,
    ``ok`` is ``False`` with a ``reason``.
    """
    from imas_codex.standard_names.graph_ops import stage_name_for_rescore

    run_id = f"sn-rescore-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    staged = stage_name_for_rescore(sn_id, run_id=run_id, dry_run=dry_run)
    if not staged.get("ok"):
        return staged

    reviewed = not (stage_only or dry_run)
    result: dict[str, Any] = {
        "ok": True,
        "sn_id": sn_id,
        "prior_stage": staged.get("prior_stage"),
        "run_id": run_id,
        "reviewed": reviewed,
        "outcome": None,
    }
    if dry_run:
        return result

    validation = _run_rescore_validation(sn_id)
    result["validation"] = validation
    requarantined = set(validation.get("requarantined_ids") or ())
    if sn_id in requarantined or int(validation.get("quarantined", 0) or 0) > 0:
        from imas_codex.standard_names.graph_ops import (
            restore_name_after_failed_rescore,
        )

        restored = restore_name_after_failed_rescore(
            sn_id,
            run_id=run_id,
            prior_stage=str(staged.get("prior_stage")),
        )
        result.update(
            {
                "ok": False,
                "reviewed": False,
                "restored": restored,
                "reason": (
                    f"{sn_id!r} failed deterministic revalidation and was "
                    "re-quarantined before review; its prior terminal state "
                    f"was {'restored' if restored else 'not restored'}"
                ),
            }
        )
        return result

    if not reviewed:
        return result

    # skip_generate: the name already exists (it is being re-scored, not
    # regenerated) — run the review (+refine) pools scoped to this run_id only.
    summary = _run_scoped_pipeline(
        run_id=run_id,
        skip_generate=True,
        cost_limit=cost_limit,
        min_score=min_score,
        rotation_cap=rotation_cap,
        pending_fn=pending_fn,
    )

    owns_gc = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        results = _collect_inline_outcomes(gc, [sn_id], axis="name")
    finally:
        if owns_gc:
            gc.close()

    result["outcome"] = InlineReviewOutcome(
        ran=True,
        run_id=run_id,
        cost=float(getattr(summary, "cost_spent", 0.0) or 0.0),
        stop_reason=getattr(summary, "stop_reason", None),
        results=results,
    )
    return result


_ACCEPTED_REVIEW_RESTAGE_SNAPSHOT_QUERY = """
// ACCEPTED_REVIEW_RESTAGE_SNAPSHOT
UNWIND $name_ids AS requested_id
OPTIONAL MATCH (sn:StandardName {id: requested_id})
WITH requested_id, collect(sn) AS matches
RETURN requested_id,
       [sn IN matches | {
           element_id: elementId(sn),
           properties: properties(sn),
           outgoing: [(sn)-[relationship]->(other) | {
               element_id: elementId(relationship),
               relationship_type: type(relationship),
               properties: properties(relationship),
               other_element_id: elementId(other),
               other_id: other.id,
               other_labels: labels(other)
           }],
           incoming: [(other)-[relationship]->(sn) | {
               element_id: elementId(relationship),
               relationship_type: type(relationship),
               properties: properties(relationship),
               other_element_id: elementId(other),
               other_id: other.id,
               other_labels: labels(other)
           }]
       }] AS matches
ORDER BY requested_id
"""

_ACCEPTED_REVIEW_RESTAGE_LOCK_QUERY = """
// ACCEPTED_REVIEW_RESTAGE_LOCK
UNWIND $targets AS target
MATCH (sn:StandardName)
WHERE elementId(sn) = target.element_id AND sn.id = target.id
SET sn.id = sn.id
RETURN collect(sn.id) AS locked_ids
"""

_ACCEPTED_REVIEW_RESTAGE_MUTATION_QUERY = """
// ACCEPTED_REVIEW_RESTAGE_MUTATION
UNWIND $targets AS target
MATCH (sn:StandardName)
WHERE elementId(sn) = target.element_id
  AND sn.id = target.id
  AND sn.name_stage = 'accepted'
  AND sn.validation_status = 'valid'
  AND sn.reviewer_score_name IS NULL
  AND sn.claim_token IS NULL
  AND sn.claimed_at IS NULL
  AND sn.drain_scope_id IS NULL
  AND sn.drain_scope_claimed_at IS NULL
  AND sn.drain_claim_scope_id IS NULL
SET sn.name_stage = 'drafted', sn.run_id = $run_id
RETURN collect(sn.id) AS staged_ids
"""

_ACCEPTED_REVIEW_BINDING_TYPES = (
    "HAS_STANDARD_NAME",
    "HAS_UNIT",
    "HAS_COCOS",
)


def _accepted_review_run_id(name_ids: list[str]) -> str:
    encoded = json.dumps(name_ids, separators=(",", ":")).encode()
    return f"sn-review-restage-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _accepted_review_refusal(
    name_ids: list[str],
    reason: str,
    *,
    run_id: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    return {
        "schema": "imas-codex.accepted-review-restage-receipt",
        "schema_version": 1,
        "outcome": "refused",
        "dry_run": dry_run,
        "run_id": run_id,
        "requested": len(name_ids),
        "would_stage": 0,
        "staged": 0,
        "rows": [],
        "reason": reason,
    }


def _accepted_review_snapshot(
    transaction: Any, name_ids: list[str]
) -> dict[str, dict[str, Any]]:
    rows = list(
        transaction.run(
            _ACCEPTED_REVIEW_RESTAGE_SNAPSHOT_QUERY,
            name_ids=name_ids,
        )
    )
    snapshots: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        requested_id = str(row["requested_id"])
        snapshots[requested_id] = {
            "requested_id": requested_id,
            "matches": _fold_normalize(row.get("matches") or []),
        }
    return snapshots


def _accepted_review_relationship_state(
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name_id, snapshot in snapshots.items():
        matches = snapshot.get("matches") or []
        if len(matches) != 1:
            continue
        match = matches[0]
        state[name_id] = {
            "outgoing": match.get("outgoing") or [],
            "incoming": match.get("incoming") or [],
        }
    return _fold_normalize(state)


def _accepted_review_relationship_counts(
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts = dict.fromkeys(_ACCEPTED_REVIEW_BINDING_TYPES, 0)
    for relationship_state in _accepted_review_relationship_state(snapshots).values():
        for direction in ("outgoing", "incoming"):
            for relationship in relationship_state[direction]:
                relationship_type = relationship.get("relationship_type")
                if relationship_type in counts:
                    counts[relationship_type] += 1
    return counts


def _accepted_review_expected_post(
    snapshots: dict[str, dict[str, Any]],
    staged_ids: set[str],
    run_id: str,
) -> dict[str, dict[str, Any]]:
    expected = deepcopy(snapshots)
    for name_id in staged_ids:
        properties = expected[name_id]["matches"][0]["properties"]
        properties["name_stage"] = "drafted"
        properties["run_id"] = run_id
    return _fold_normalize(expected)


@retry_on_deadlock()
def restage_accepted_names_for_review(
    name_ids: Iterable[str],
    *,
    include_accepted: bool = False,
    dry_run: bool = True,
    run_id: str | None = None,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Atomically move an exact accepted-unscored cohort to name review.

    Every requested identity must be accepted, valid, name-unscored, and free
    of worker and drain claims. The only applied property changes are
    ``name_stage='drafted'`` and one deterministic cohort ``run_id``; the
    spelling, review fields, DD authority, and every relationship are preserved.
    The ordinary ``REVIEW_NAME`` pool then decides whether each unchanged
    identity earns acceptance again.

    ``include_accepted=True`` is mandatory because applying this transition
    temporarily removes the names from export eligibility until the quorum
    accepts them again. The default is a zero-write dry run. Replaying the same
    exact cohort after a successful apply is idempotent and stages nothing.
    """
    normalized_ids = sorted(str(name_id).strip() for name_id in name_ids)
    if not normalized_ids or any(not name_id for name_id in normalized_ids):
        return _accepted_review_refusal(
            normalized_ids,
            "at least one non-empty StandardName identity is required",
            dry_run=dry_run,
        )
    if len(set(normalized_ids)) != len(normalized_ids):
        return _accepted_review_refusal(
            normalized_ids,
            "duplicate StandardName identities are not permitted",
            dry_run=dry_run,
        )
    resolved_run_id = (run_id or "").strip() or _accepted_review_run_id(normalized_ids)
    if not include_accepted:
        return _accepted_review_refusal(
            normalized_ids,
            "accepted-name restaging requires --include-accepted",
            run_id=resolved_run_id,
            dry_run=dry_run,
        )

    owns_gc = gc is None
    graph = gc or GraphClient()
    try:
        with graph.session() as session:
            transaction = session.begin_transaction()
            try:
                before = _accepted_review_snapshot(transaction, normalized_ids)
                if set(before) != set(normalized_ids):
                    transaction.rollback()
                    missing = sorted(set(normalized_ids).difference(before))
                    return _accepted_review_refusal(
                        normalized_ids,
                        "snapshot omitted requested identities: " + ", ".join(missing),
                        run_id=resolved_run_id,
                        dry_run=dry_run,
                    )

                eligible_ids: set[str] = set()
                idempotent_ids: set[str] = set()
                refusals: list[dict[str, str]] = []
                targets: list[dict[str, str]] = []
                for name_id in normalized_ids:
                    matches = before[name_id]["matches"]
                    if len(matches) != 1:
                        reason = (
                            "missing StandardName"
                            if not matches
                            else "ambiguous StandardName identity"
                        )
                        refusals.append({"id": name_id, "reason": reason})
                        continue
                    match = matches[0]
                    properties = match["properties"]
                    is_claim_free = all(
                        properties.get(field) is None
                        for field in (
                            "claim_token",
                            "claimed_at",
                            "drain_scope_id",
                            "drain_scope_claimed_at",
                            "drain_claim_scope_id",
                        )
                    )
                    is_eligible = (
                        properties.get("name_stage") == "accepted"
                        and properties.get("validation_status") == "valid"
                        and properties.get("reviewer_score_name") is None
                        and is_claim_free
                    )
                    is_idempotent = (
                        properties.get("name_stage") == "drafted"
                        and properties.get("validation_status") == "valid"
                        and properties.get("reviewer_score_name") is None
                        and properties.get("run_id") == resolved_run_id
                        and is_claim_free
                    )
                    if is_eligible:
                        eligible_ids.add(name_id)
                        targets.append(
                            {"id": name_id, "element_id": match["element_id"]}
                        )
                    elif is_idempotent:
                        idempotent_ids.add(name_id)
                    else:
                        refusals.append(
                            {
                                "id": name_id,
                                "reason": (
                                    "row is not accepted-valid-null-score and "
                                    "claim-free"
                                ),
                            }
                        )

                if refusals:
                    transaction.rollback()
                    receipt = _accepted_review_refusal(
                        normalized_ids,
                        "one or more rows failed the exact restage precondition",
                        run_id=resolved_run_id,
                        dry_run=dry_run,
                    )
                    receipt["refused_rows"] = refusals
                    return receipt

                relationship_state_before = _accepted_review_relationship_state(before)
                binding_counts_before = _accepted_review_relationship_counts(before)
                rows = [
                    {
                        "id": name_id,
                        "before_stage": before[name_id]["matches"][0]["properties"].get(
                            "name_stage"
                        ),
                        "after_stage": "drafted",
                        "changed": name_id in eligible_ids,
                    }
                    for name_id in normalized_ids
                ]
                base_receipt: dict[str, Any] = {
                    "schema": "imas-codex.accepted-review-restage-receipt",
                    "schema_version": 1,
                    "dry_run": dry_run,
                    "run_id": resolved_run_id,
                    "requested": len(normalized_ids),
                    "would_stage": len(eligible_ids),
                    "staged": 0,
                    "idempotent": len(idempotent_ids),
                    "rows": rows,
                    "binding_counts_before": binding_counts_before,
                    "binding_counts_after": binding_counts_before,
                    "relationship_count_before": sum(
                        len(state[direction])
                        for state in relationship_state_before.values()
                        for direction in ("outgoing", "incoming")
                    ),
                    "relationship_count_after": sum(
                        len(state[direction])
                        for state in relationship_state_before.values()
                        for direction in ("outgoing", "incoming")
                    ),
                    "relationship_signature_before": _fold_cas_signature(
                        relationship_state_before
                    ),
                    "relationship_signature_after": _fold_cas_signature(
                        relationship_state_before
                    ),
                    "reviewer_scores_written": 0,
                }
                if dry_run:
                    transaction.rollback()
                    base_receipt["outcome"] = (
                        "idempotent" if not eligible_ids else "would_apply"
                    )
                    return base_receipt
                if not eligible_ids:
                    transaction.rollback()
                    base_receipt["outcome"] = "idempotent"
                    return base_receipt

                lock_rows = list(
                    transaction.run(
                        _ACCEPTED_REVIEW_RESTAGE_LOCK_QUERY,
                        targets=targets,
                    )
                )
                locked_ids = (
                    sorted(dict(lock_rows[0]).get("locked_ids") or [])
                    if len(lock_rows) == 1
                    else []
                )
                if locked_ids != sorted(eligible_ids):
                    raise RuntimeError("accepted restage lock cardinality changed")
                locked = _accepted_review_snapshot(transaction, normalized_ids)
                if _fold_cas_signature(locked) != _fold_cas_signature(before):
                    raise RuntimeError("accepted restage cohort drifted after locking")

                mutation_rows = list(
                    transaction.run(
                        _ACCEPTED_REVIEW_RESTAGE_MUTATION_QUERY,
                        targets=targets,
                        run_id=resolved_run_id,
                    )
                )
                staged_ids = (
                    sorted(dict(mutation_rows[0]).get("staged_ids") or [])
                    if len(mutation_rows) == 1
                    else []
                )
                if staged_ids != sorted(eligible_ids):
                    raise RuntimeError("accepted restage compare-and-set refused a row")

                after = _accepted_review_snapshot(transaction, normalized_ids)
                expected_after = _accepted_review_expected_post(
                    before, eligible_ids, resolved_run_id
                )
                if _fold_normalize(after) != expected_after:
                    raise RuntimeError("accepted restage post-state proof failed")
                relationship_state_after = _accepted_review_relationship_state(after)
                binding_counts_after = _accepted_review_relationship_counts(after)
                if relationship_state_after != relationship_state_before:
                    raise RuntimeError(
                        "accepted restage changed an identity relationship"
                    )
                if binding_counts_after != binding_counts_before:
                    raise RuntimeError(
                        "accepted restage changed authority binding counts"
                    )

                transaction.commit()
                base_receipt.update(
                    {
                        "outcome": "applied",
                        "staged": len(staged_ids),
                        "binding_counts_after": binding_counts_after,
                        "relationship_signature_after": _fold_cas_signature(
                            relationship_state_after
                        ),
                    }
                )
                return base_receipt
            except BaseException:
                with suppress(Exception):
                    transaction.rollback()
                raise
    finally:
        if owns_gc:
            graph.close()


def _run_rescore_validation(sn_id: str) -> dict[str, Any]:
    """Run the LLM-free admission gate for one freshly staged rescore."""
    from imas_codex.cli.utils import run_async
    from imas_codex.standard_names.workers import drain_validation_for_ids

    return run_async(drain_validation_for_ids([sn_id]))

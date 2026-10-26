"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.review.holds import (
    sync_holds,
)

from . import graph_foundation as foundation

logger = logging.getLogger(__name__)


def _quorum_admits_acceptance(
    resolution_method: str | None,
    reviewer_chain_size: int | None,
) -> str | None:
    """Return why a review cannot carry an acceptance, or ``None`` when it can.

    A reviewer chain that loses seats mid-run still produces a score — the
    surviving seat's — and that score is indistinguishable from a quorate one
    once it reaches the stage decision.  Acceptance therefore has to ask how
    the score was reached, not just how high it is.
    """
    if resolution_method in foundation.get_services().QUORATE_RESOLUTION_METHODS:
        return None
    if resolution_method is None:
        return "review carried no resolution method"
    if resolution_method == "max_cycles_reached":
        if (reviewer_chain_size or 0) < foundation.get_services()._ESCALATOR_CHAIN_SIZE:
            return None
        return (
            "blind seats disagreed and the escalator seat did not resolve them "
            f"(method={resolution_method})"
        )
    return f"fewer reviewer seats scored than the chain defines (method={resolution_method})"


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _authority_json_value(value: Any) -> Any:
    """Convert driver values to deterministic JSON-compatible values."""
    if isinstance(value, dict):
        return {
            str(key): foundation.get_services()._authority_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [foundation.get_services()._authority_json_value(item) for item in value]
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _authority_payload_hash(value: Any) -> str:
    payload = foundation.get_services().json.dumps(
        foundation.get_services()._authority_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return foundation.get_services().hashlib.sha256(payload.encode("utf-8")).hexdigest()


@retry_on_deadlock()
def persist_reviewed_name(
    *,
    sn_id: str,
    claim_token: str,
    claim_seq: int | None = None,
    score: float,
    scores: dict[str, Any] | None = None,
    comments: str | None = None,
    comments_per_dim: dict[str, Any] | None = None,
    model: str,
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    llm_cost: float | None = None,
    llm_tokens_in: int | None = None,
    llm_tokens_out: int | None = None,
    llm_tokens_cached_read: int | None = None,
    llm_tokens_cached_write: int | None = None,
    llm_at: str | None = None,
    llm_service: str | None = None,
    run_id: str | None = None,
    skip_review_node: bool = False,
    resolution_method: str | None = None,
    reviewer_chain_size: int | None = None,
    quorum_exempt: bool = False,
) -> str:
    """Persist name-review results and transition ``name_stage``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. Compute target stage:
       - ``'accepted'`` if ``score >= min_score`` and no docs hold is open
       - ``'exhausted'`` if a positive ``refine_attempts`` count has reached
         ``rotation_cap`` and the score is below min_score (every rotation
         spent, including the escalated final one)
       - ``'reviewed'`` otherwise (eligible for refine_name pickup; the last
         remaining rotation routes through the escalation seat in
         process_refine_name_batch)
    3. Record a quorum shortfall whenever the reviewer chain did not reach a
       verdict — see ``resolution_method`` below.  A would-be acceptance is
       withheld and holds at ``'reviewed'``; a below-threshold score keeps its
       stage but is skipped by ``REFINE_NAME_ELIGIBILITY_WHERE``, so neither
       decision is acted on until ``sn rescore`` re-reviews the name.
    4. SET reviewer fields, ``name_stage``, clear claim state.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    score:
        Normalised review score (0-1).
    scores:
        Per-dimension sub-scores dict (written as JSON to
        ``reviewer_scores_name``).
    comments:
        Free-text reviewer comments (written to ``reviewer_comments_name``).
    comments_per_dim:
        Per-dimension comments dict (written as JSON to
        ``reviewer_comments_per_dim_name``).
    model:
        LLM model slug used for this review.
    min_score:
        Acceptance threshold.
    rotation_cap:
        Maximum chain depth before exhaustion (same value used by
        :func:`claim_refine_name_batch`).
    resolution_method:
        How the reviewer chain reached this score, as recorded by the
        RD-quorum loop.  A quorate method is required before the score is
        acted on in either direction — accepting the name or spending a
        refine rotation to rewrite it.  Omitting it withholds both, so a
        caller that publishes names must state how the score was reached.
    reviewer_chain_size:
        Number of seats the configured chain defines.  Distinguishes a
        two-seat chain's terminal disagreement, which is its designed
        outcome, from an escalator chain whose escalator never answered.
    quorum_exempt:
        Skip the quorum gate for an acceptance carrying a different
        authority than the reviewer chain — a merged catalog PR.  Never set
        it on a pipeline review.

    Returns
    -------
    str
        The new ``name_stage`` value (``'accepted'``, ``'reviewed'``, or
        ``'exhausted'``).  Returns ``''`` when the token did not match
        (no-op).
    """
    import json as _json

    # Read current chain_length — needed for exhaustion check.
    # Relax claim_token guard: accept token match OR cleared token (orphan
    # sweep may have nullified claim_token while the RD-quorum review was
    # still in flight).  The name_stage='drafted' check is the real CAS
    # guard — a name can only transition OUT of drafted once.
    with foundation.get_services().GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.name_stage = 'drafted'
              AND (($claim_seq IS NULL
                    AND (sn.claim_token = $token OR sn.claim_token IS NULL))
                   OR (sn.claim_token = $token AND sn.claim_seq = $claim_seq))
            RETURN coalesce(sn.chain_length, 0) AS chain_length,
                   coalesce(sn.refine_attempts, coalesce(sn.chain_length, 0))
                       AS refine_attempts,
                   sn.refine_attempts IS NOT NULL AS refine_attempts_recorded,
                   sn.validation_status AS validation_status,
                   sn.edit_status AS edit_status,
                   sn.edit_scope AS edit_scope,
                   sn.edit_mode AS edit_mode,
                   sn.edit_override_edits AS edit_override_edits,
                   sn.edit_include_accepted AS edit_include_accepted,
                   sn.id AS current_id
            """,
            id=sn_id,
            token=claim_token,
            claim_seq=claim_seq,
        )

    if not rows:
        logger.debug(
            "persist_reviewed_name: stage/token mismatch for %s (token=%s) — no-op",
            sn_id,
            claim_token[:8],
        )
        return ""

    chain_length: int = int(rows[0]["chain_length"])
    refine_attempts: int = int(rows[0].get("refine_attempts") or chain_length)
    # Lightweight stage-decision fixtures written before the durable counter
    # may omit this projection. Real Neo4j rows always carry it, including
    # ``False`` for a missing property.
    attempt_state_known = "refine_attempts_recorded" in rows[0]
    refine_attempts_recorded = bool(rows[0].get("refine_attempts_recorded", True))
    has_refine_attempt = refine_attempts_recorded and refine_attempts > 0
    edit_status_before: str | None = rows[0].get("edit_status")
    edit_scope_before: str | None = rows[0].get("edit_scope")
    edit_mode_before: str | None = rows[0].get("edit_mode")
    # Cascade-authorization opt-in recorded at edit time (coalesce to False —
    # a rename edit that did not opt in must NOT silently clobber protected
    # (catalog_edit) or accepted descendants when the root rename is accepted).
    edit_override_before: bool = bool(rows[0].get("edit_override_edits"))
    edit_include_before: bool = bool(rows[0].get("edit_include_accepted"))
    current_id: str | None = rows[0].get("current_id")

    # A same-spelling review authorizes no rename and therefore has no
    # descendant cascade to preflight or apply.  Run ids cannot establish this:
    # a later scoped drain legitimately replaces them.  Compare the spelling
    # submitted to the quorum with the id read under the stage/token guard, while
    # retaining cascade semantics for an explicit rename successor.
    from imas_codex.standard_names.rescore import review_preserves_identity

    identity_preserving_review = review_preserves_identity(
        reviewed_spelling=sn_id,
        current_id=current_id,
        edit_mode=edit_mode_before,
    )

    # ── Stage decision ────────────────────────────────────────────────
    # Score is canonical (rubric-driven 0–1).
    # Exhaustion is decided by the rotations SPENT, not by the depth of the
    # persisted chain: rotations charged on attempts that never produced a
    # successor are exactly the ones a lineage-depth test cannot see, and a
    # name below threshold with no rotations left is terminal — leaving it at
    # 'reviewed' would advertise a refinement that can never be claimed. A
    # missing or zero attempt counter proves no refinement has yet been spent,
    # so it cannot establish exhaustion even when a caller supplies a zero
    # cap. A name still holding budget stays 'reviewed'; its last rotation
    # routes through the escalation seat in process_refine_name_batch.
    if score >= min_score:
        target_stage = "accepted"
    elif has_refine_attempt and refine_attempts >= rotation_cap:
        target_stage = "exhausted"
    else:
        target_stage = "reviewed"

    # ── Quorum gate ───────────────────────────────────────────────────
    # A score high enough to accept is only as good as the seats that
    # produced it.  When seats drop out mid-run the surviving seat's score
    # arrives here looking exactly like a quorate one, so a provider outage
    # would otherwise publish names on a single opinion.  Hold the name at
    # 'reviewed' with the shortfall recorded: the score and its comments
    # survive, and `sn rescore` re-reviews it once the seats are healthy.
    # `quorum_exempt` is for accepts that carry a different authority than
    # the reviewer chain (a merged catalog PR), never for pipeline reviews.
    # The marker is recorded for every non-quorate review, not only for one
    # that would have accepted.  A score below the threshold decides the
    # opposite thing — that the NAME needs rewriting — and it is no more
    # trustworthy for having come from seats that never agreed.  Acting on it
    # spends a paid refine rotation, and a rotation can rewrite a sound name
    # into a candidate that terminates it.  ``REFINE_NAME_ELIGIBILITY_WHERE``
    # therefore skips any name carrying this marker; ``sn rescore`` is the
    # recovery, and the next quorate review clears it by writing null here.
    quorum_shortfall: str | None = None
    if not quorum_exempt:
        quorum_shortfall = foundation.get_services()._quorum_admits_acceptance(
            resolution_method, reviewer_chain_size
        )
        if quorum_shortfall is not None:
            if target_stage == "accepted":
                target_stage = "reviewed"
                logger.warning(
                    "persist_reviewed_name: refusing acceptance of %s at "
                    "score %.3f — %s",
                    sn_id,
                    score,
                    quorum_shortfall,
                )
            else:
                logger.warning(
                    "persist_reviewed_name: parking %s at %s without a refine "
                    "rotation (score %.3f) — %s",
                    sn_id,
                    target_stage,
                    score,
                    quorum_shortfall,
                )

    grammar_valid = True
    grammar_issue: str | None = None
    # Lightweight stage-decision unit fixtures may omit the graph status field;
    # real Neo4j rows always carry the projected key, including when its value
    # is null. The durable write path therefore always runs this gate.
    if "validation_status" in rows[0]:
        try:
            from imas_codex.standard_names.grammar_adapter import parse_canonical_name

            parse_canonical_name(sn_id)
        except Exception as exc:
            grammar_valid = False
            grammar_issue = f"[strict_grammar] {str(exc)[:240]}"
            target_stage = (
                "exhausted"
                if has_refine_attempt or not attempt_state_known
                else "reviewed"
            )
            logger.warning(
                "persist_reviewed_name: quarantining grammar-invalid name %s: %s",
                sn_id,
                exc,
            )
    grammar_identity = foundation.get_services()._parse_grammar(sn_id)

    # ── Cascade atomicity preflight (edit rename acceptance) ─────────────
    # A family/subtree rename edit accepts the ROOT and cascades every
    # descendant id atomically.  Validate the WHOLE descendant cascade
    # (ISN grammar round-trip + uniqueness of every new id) BEFORE the
    # acceptance is written — so a constraint collision or a grammar-invalid
    # descendant id refuses the acceptance itself rather than leaving the
    # root accepted with a half-applied cascade.  On conflict the root is
    # NOT accepted (it stays reviewable) and no descendant is renamed:
    # nothing persisted.  The auto-commit query primitive cannot span the
    # acceptance and the rename in one Neo4j transaction, so atomicity is
    # achieved by gating the accept on a clean preflight; the residual
    # accept→apply window is the same TOCTOU the pool path carries.
    cascade_preflight_conflicts: list[str] = []
    if (
        target_stage == "accepted"
        and not identity_preserving_review
        and edit_scope_before in ("family", "subtree")
    ):
        from imas_codex.standard_names.cascade import cascade_descendants_of

        with foundation.get_services().GraphClient() as gc:
            pred_rows = gc.query(
                """
                MATCH (sn:StandardName {id: $id})-[:REFINED_FROM]->(pred:StandardName)
                RETURN pred.id AS pred_id
                """,
                id=sn_id,
            )
            old_root_pf = pred_rows[0]["pred_id"] if pred_rows else sn_id
            preflight = cascade_descendants_of(
                gc,
                successor_id=sn_id,
                old_root=old_root_pf,
                new_root=sn_id,
                dry_run=True,
                override_edits=edit_override_before,
                include_accepted=edit_include_before,
            )
        if preflight.conflicts:
            cascade_preflight_conflicts = preflight.conflicts
            # Refuse the acceptance — the reviewed decision cannot land
            # while its atomic consequences (the descendant cascade) are
            # invalid. Route back to 'reviewed' so an operator can resolve
            # the collision; nothing is accepted and nothing is renamed.
            target_stage = "reviewed"
            logger.warning(
                "persist_reviewed_name: %s scored acceptance but its "
                "descendant cascade has %d conflict(s) — refusing acceptance "
                "(name_stage=reviewed, no rename applied): %s",
                sn_id,
                len(preflight.conflicts),
                preflight.conflicts,
            )

    # ── Edit lifecycle decision ──────────────────────────────────────
    # Mirrors the `open → applied | exhausted | rejected` lifecycle
    # documented on EditStatus. A 'reviewed' outcome leaves the edit
    # 'open' — it rides the refine_name loop (persist_refined_name
    # propagates the still-open edit fields onto the next rotation).
    new_edit_status = edit_status_before
    if edit_status_before == "open":
        if target_stage == "accepted":
            new_edit_status = "applied"
        elif target_stage == "exhausted":
            new_edit_status = "exhausted"
        else:
            new_edit_status = "open"

    scores_json = _json.dumps(scores) if scores is not None else None
    comments_per_dim_json = (
        _json.dumps(comments_per_dim) if comments_per_dim is not None else None
    )

    with foundation.get_services().GraphClient() as gc:
        write_rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.name_stage = 'drafted'
              AND (($claim_seq IS NULL
                    AND (sn.claim_token = $token OR sn.claim_token IS NULL))
                   OR (sn.claim_token = $token AND sn.claim_seq = $claim_seq))
            SET sn.updated_at = datetime(), sn.reviewer_score_name        = $score,
                sn.reviewer_scores_name       = $scores_json,
                sn.reviewer_comments_name     = $comments,
                sn.reviewer_comments_per_dim_name = $comments_per_dim_json,
                sn.reviewer_model_name        = $model,
                sn.reviewed_name_at           = datetime(),
                sn.name_stage                 = $target_stage,
                sn.edit_status                = $new_edit_status,
                sn.claim_token                = null,
                sn.claimed_at                 = null,
                sn.validation_status = CASE
                  WHEN $grammar_issue IS NOT NULL THEN 'quarantined'
                  ELSE sn.validation_status
                END,
                sn.validation_issues = CASE
                  WHEN $grammar_issue IS NULL THEN sn.validation_issues
                  ELSE coalesce(sn.validation_issues, []) + [$grammar_issue]
                END,
                sn.review_quorum_shortfall    = $quorum_shortfall,
                sn.review_quorum_shortfall_at = CASE
                  WHEN $quorum_shortfall IS NULL THEN null
                  ELSE datetime()
                END,
                sn.review_resolution_method   = $resolution_method,
                sn.refine_stop_reason = CASE
                  WHEN $target_stage <> 'exhausted' THEN sn.refine_stop_reason
                  WHEN NOT $grammar_valid THEN $grammar_invalid_reason
                  ELSE $attempts_exhausted_reason
                END,
                sn.refine_stopped_at = CASE
                  WHEN $target_stage <> 'exhausted' THEN sn.refine_stopped_at
                  ELSE datetime()
                END
            SET sn += $grammar_identity
            RETURN sn.id AS id
            """,
            id=sn_id,
            token=claim_token,
            claim_seq=claim_seq,
            score=score,
            scores_json=scores_json,
            comments=comments,
            comments_per_dim_json=comments_per_dim_json,
            model=model,
            target_stage=target_stage,
            new_edit_status=new_edit_status,
            grammar_valid=grammar_valid,
            grammar_issue=grammar_issue,
            grammar_identity=grammar_identity,
            quorum_shortfall=quorum_shortfall,
            resolution_method=resolution_method,
            grammar_invalid_reason=foundation.get_services().RefineStopReason.grammar_invalid.value,
            attempts_exhausted_reason=foundation.get_services().RefineStopReason.attempts_exhausted.value,
        )

    # A concurrent reviewer already transitioned this node out of 'drafted'
    # between our readback and this SET — our review is a duplicate; report a
    # no-op so the worker does not double-count or emit a spurious event.
    if not write_rows:
        logger.debug(
            "persist_reviewed_name: %s already transitioned out of 'drafted' "
            "(concurrent reviewer won) — no-op",
            sn_id,
        )
        return ""

    logger.info(
        "persist_reviewed_name: %s → name_stage=%s "
        "(score=%.3f, rotations=%d/%d, chain=%d)",
        sn_id,
        target_stage,
        score,
        refine_attempts,
        rotation_cap,
        chain_length,
    )

    # Acceptance refused by the cascade preflight — record why on the node so
    # an operator can resolve the collision. Nothing was accepted, nothing
    # renamed; the name stays reviewable.
    if cascade_preflight_conflicts:
        with foundation.get_services().GraphClient() as gc:
            gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                SET sn.updated_at = datetime(), sn.validation_issues = coalesce(sn.validation_issues, [])
                    + [c IN $conflicts | '[edit_cascade] ' + c]
                """,
                id=sn_id,
                conflicts=cascade_preflight_conflicts,
            )

    # ── Apply the staged descendant cascade atomically on edit acceptance ──
    # scope='family'|'subtree' means sn_id is the (possibly parent-mapped)
    # rename root of an `imas-codex sn edit` proposal — accepting it here is
    # the reviewed decision; descendants never individually re-enter LLM
    # review, they follow the root atomically. The cascade was already
    # preflighted clean above (no conflicts), so this apply reproduces that
    # validated plan against the LIVE subtree.
    if (
        target_stage == "accepted"
        and not identity_preserving_review
        and new_edit_status == "applied"
        and edit_scope_before in ("family", "subtree")
    ):
        from imas_codex.standard_names.cascade import cascade_descendants_of

        with foundation.get_services().GraphClient() as gc:
            pred_rows = gc.query(
                """
                MATCH (sn:StandardName {id: $id})-[:REFINED_FROM]->(pred:StandardName)
                RETURN pred.id AS pred_id
                """,
                id=sn_id,
            )
            old_root = pred_rows[0]["pred_id"] if pred_rows else sn_id
            cascade_result = cascade_descendants_of(
                gc,
                successor_id=sn_id,
                old_root=old_root,
                new_root=sn_id,
                dry_run=False,
                override_edits=edit_override_before,
                include_accepted=edit_include_before,
            )
            if cascade_result.conflicts:
                # The preflight was clean, so a conflict here is a genuine
                # concurrent-write race between preflight and apply. Record it
                # for operator follow-up; the acceptance already landed.
                logger.warning(
                    "persist_reviewed_name: edit accepted for %s but the "
                    "descendant cascade raced to %d conflict(s) after a clean "
                    "preflight — descendants left unrenamed: %s",
                    sn_id,
                    len(cascade_result.conflicts),
                    cascade_result.conflicts,
                )
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $id})
                    SET sn.updated_at = datetime(), sn.validation_issues = coalesce(sn.validation_issues, [])
                        + [c IN $conflicts | '[edit_cascade] ' + c]
                    """,
                    id=sn_id,
                    conflicts=cascade_result.conflicts,
                )
            else:
                logger.info(
                    "persist_reviewed_name: edit-cascade applied for %s — "
                    "%d descendant(s) renamed",
                    sn_id,
                    len(cascade_result.renamed),
                )

    # ── Write StandardNameReview node + HAS_REVIEW edge ─────────────────
    # The single-reviewer worker path was previously SETting reviewer_*
    # fields on the SN node only — no Review node was ever created.
    # Mirror the RD-quorum schema: cycle_index=0, role='primary',
    # canonical=True, fresh review_group_id per call.
    #
    # When ``skip_review_node`` is True the caller (pool RD-quorum
    # worker) is responsible for writing per-cycle Review nodes itself
    # and only needs the SN-side stage transition + axis-slot mirror.
    if not skip_review_node:
        try:
            import uuid as _uuid
            from datetime import datetime as _dt

            _now_iso = llm_at or _dt.now(foundation.get_services().UTC).isoformat()
            _group_id = str(_uuid.uuid4())
            _review_id = f"{sn_id}:name:{_group_id}:0"
            # Map normalised score (0-1) to tier name per the rubric.
            if score >= 0.85:
                _tier = "outstanding"
            elif score >= 0.60:
                _tier = "good"
            elif score >= 0.40:
                _tier = "inadequate"
            else:
                _tier = "poor"
            foundation.get_services().write_reviews(
                [
                    {
                        "id": _review_id,
                        "standard_name_id": sn_id,
                        "model": model,
                        "reviewer_model": model,
                        "model_family": "other",
                        "is_canonical": True,
                        "score": float(score),
                        "scores_json": scores_json or "{}",
                        "tier": _tier,
                        "comments": comments or "",
                        "comments_per_dim_json": comments_per_dim_json,
                        "suggested_name": "",
                        "suggestion_justification": "",
                        "reviewed_at": _now_iso,
                        "review_axis": "name",
                        "cycle_index": 0,
                        "review_group_id": _group_id,
                        "resolution_role": "primary",
                        "resolution_method": resolution_method,
                        "llm_model": model,
                        "llm_cost": llm_cost,
                        "llm_tokens_in": llm_tokens_in,
                        "llm_tokens_out": llm_tokens_out,
                        "llm_tokens_cached_read": llm_tokens_cached_read,
                        "llm_tokens_cached_write": llm_tokens_cached_write,
                        "llm_at": _now_iso,
                        "llm_service": llm_service or "standard-names",
                    }
                ],
            )
        except Exception:
            # Don't let review-node bookkeeping fail the stage transition.
            logger.exception(
                "persist_reviewed_name: failed to write StandardNameReview for %s",
                sn_id,
            )

    # Async counter bump — live progress visibility for ``sn status``
    foundation.get_services().bump_sn_run_counter(run_id, "names_reviewed")

    return target_stage


@retry_on_deadlock()
def claim_review_docs_batch(
    facility: str | None = None,
    batch_size: int = foundation.DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = foundation._CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for docs review.

    Eligibility: ``docs_stage = 'drafted'`` AND ``claimed_at IS NULL``.

    Does NOT transition stage at claim time — stage remains ``'drafted'``
    until :func:`persist_reviewed_docs` writes the final outcome.  Only
    ``claim_token`` and ``claimed_at`` are set so the orphan sweep can
    recover stuck claims.

    Returns claimed items as dicts with keys:
    ``id``, ``name``, ``description``, ``documentation``, ``kind``,
    ``unit``, ``tags``, ``physics_domain``, ``docs_chain_length``,
    ``claim_token``.
    """
    # Name-form-vetted gate (same invariant as generate_docs): a real
    # name-review score OR a structurally-accepted derived parent. CURATIVE-SCOPE
    # EXCEPTION: under a scope_run_id (curative family-docs wave) the names are
    # already-accepted and operator-authorised for re-docs, so the name-score
    # gate is dropped inside the scope (mirrors claim_generate_docs_batch) —
    # otherwise derived-leaf families draft docs that can never be reviewed.
    # The same operator authorisation holds under edits_only: an open sn-edit
    # is an explicit steered proposal on a name whose form is already
    # authoritative (catalog-imported names carry no reviewer_score_name), so
    # without this exemption staged docs edits are permanently unclaimable.
    score_gate = (
        ""
        if (scope_run_id or edits_only)
        else (
            " AND (sn.reviewer_score_name IS NOT NULL"
            " OR (coalesce(sn.origin, '') = 'derived'"
            "     AND EXISTS { MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)"
            "       WHERE NOT coalesce(kid.name_stage, '') IN"
            "       ['superseded', 'exhausted', 'contested'] }))"
        )
    )
    where = (
        "(sn.docs_stage = 'drafted' OR "
        "($drain_scope_id IS NOT NULL AND sn.docs_stage = 'reviewed' "
        "AND sn.reviewer_score_docs >= $min_score))"
        " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
        + score_gate
    )
    if facility is not None:
        where += " AND sn.facility = $facility"
        query_params: dict[str, foundation.get_services().Any] = {"facility": facility}
    else:
        query_params = {}
    query_params.update(drain_scope_id=drain_scope_id, min_score=min_score)
    items = foundation.get_services()._claim_sn_atomic(
        eligibility_where=where,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.name AS name"
            ", sn.tags AS tags"
            ", coalesce(sn.docs_chain_length, 0) AS docs_chain_length"
            ", sn.docs_stage AS docs_stage"
            ", sn.edit_mode AS edit_mode"
            ", sn.name_hint AS name_hint"
            ", sn.docs_hint AS docs_hint"
            ", sn.edit_reason AS edit_reason"
            ", sn.edit_origin AS edit_origin"
            ", sn.origin AS origin"
            ", sn.run_id AS run_id"
            ", sn.source_paths AS source_paths"
            ", [(source:StandardNameSource)-[:PRODUCED_NAME]->(sn) | {"
            "     id: source.id,"
            "     source_type: source.source_type,"
            "     source_id: source.source_id,"
            "     dd_path: source.dd_path,"
            "     dd_version: source.dd_version"
            " }] AS source_bindings"
        ),
        domain=domain,
        scope_run_id=scope_run_id,
        drain_scope_id=drain_scope_id,
        edits_only=edits_only,
        restage_review_axis="docs" if drain_scope_id else None,
    )
    # Drop claim-race losers BEFORE the reviewer quorum spends LLM calls (see
    # _verify_docs_claim_winners). review_docs is gated on docs_stage='drafted'.
    return foundation.get_services()._verify_docs_claim_winners(
        items, eligible_stage="drafted"
    )


@retry_on_deadlock()
def persist_reviewed_docs(
    *,
    sn_id: str,
    claim_token: str,
    claim_seq: int | None = None,
    score: float,
    scores: dict[str, Any] | None = None,
    comments: str | None = None,
    comments_per_dim: dict[str, Any] | None = None,
    review_holds: list[dict[str, str]] | None = None,
    checked_hold_kinds: set[str] | None = None,
    model: str,
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    llm_cost: float | None = None,
    llm_tokens_in: int | None = None,
    llm_tokens_out: int | None = None,
    llm_tokens_cached_read: int | None = None,
    llm_tokens_cached_write: int | None = None,
    llm_at: str | None = None,
    llm_service: str | None = None,
    run_id: str | None = None,
    skip_review_node: bool = False,
    resolution_method: str | None = None,
    reviewer_chain_size: int | None = None,
) -> str:
    """Persist docs-review results and transition ``docs_stage``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. Compute target stage:
       - ``'accepted'`` if ``score >= min_score`` (score-canonical)
       - ``'exhausted'`` if ``docs_chain_length >= rotation_cap``
         (cap reached, no further refine — the escalated final
         attempt at chain == rotation_cap-1 has already been spent)
       - ``'reviewed'`` otherwise (eligible for refine_docs pickup;
         at docs_chain_length == rotation_cap-1 this routes through
         the Opus escalator in process_refine_docs_batch)
    3. Record a quorum shortfall whenever the configured reviewer chain did
       not reach a verdict. A would-be acceptance is withheld at
       ``'reviewed'``; a below-threshold stage decision is retained, but the
       marker prevents refinement from spending another rotation.
    4. Raise and clear the checks' holds under the verified claim lock.
       A held passing document stays reviewed, including at the rotation cap.
    5. SET reviewer_docs fields, ``docs_stage``, clear claim state and commit
       the score and hold changes together.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    score:
        Normalised review score (0-1).
    scores:
        Per-dimension sub-scores dict (written as JSON to
        ``reviewer_scores_docs``).
    comments:
        Free-text reviewer comments (written to ``reviewer_comments_docs``).
    comments_per_dim:
        Per-dimension comments dict (written as JSON to
        ``reviewer_comments_per_dim_docs``).
    review_holds:
        Findings from checks already run by the worker. They are written only
        after this claim has been locked and verified.
    checked_hold_kinds:
        Kinds checked by this review. Open findings of those kinds absent from
        ``review_holds`` are cleared in the same transaction.
    model:
        LLM model slug used for this review.
    min_score:
        Acceptance threshold.
    rotation_cap:
        Maximum chain depth before exhaustion (same value used by
        :func:`claim_refine_docs_batch`).
    resolution_method:
        How the RD-quorum loop reached this score. Missing metadata is treated
        as a shortfall, so historical score-only calls fail closed.
    reviewer_chain_size:
        Number of seats the configured docs reviewer chain defines. This
        preserves the canonical two-seat terminal policy while distinguishing
        a three-seat chain whose escalator did not answer.

    Returns
    -------
    str
        The new ``docs_stage`` value (``'accepted'``, ``'reviewed'``, or
        ``'exhausted'``).  Returns ``''`` when the token did not match
        (no-op).
    """
    import json as _json

    quorum_shortfall = foundation.get_services()._quorum_admits_acceptance(
        resolution_method, reviewer_chain_size
    )
    pending_holds = list(review_holds or [])
    checked_kinds = set(checked_hold_kinds or ())
    scores_json = _json.dumps(scores) if scores is not None else None

    # The preflight SET takes the name's write lock. A stale claimant cannot
    # mutate holds, and the final score and hold changes commit together.
    with foundation.get_services().GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            graph = foundation.get_services()._TransactionQuery(tx)
            try:
                rows = graph.query(
                    """
                    MATCH (sn:StandardName {id: $id})
                    WHERE (($claim_seq IS NULL
                            AND (sn.claim_token = $token OR sn.claim_token IS NULL))
                           OR (sn.claim_token = $token
                               AND sn.claim_seq = $claim_seq))
                      AND sn.docs_stage = 'drafted'
                      AND sn.name_stage = 'accepted'
                    SET sn.updated_at = datetime()
                    RETURN coalesce(sn.docs_chain_length, 0) AS docs_chain_length,
                           sn.documentation AS documentation,
                           sn.edit_status AS edit_status
                    """,
                    id=sn_id,
                    token=claim_token,
                    claim_seq=claim_seq,
                )
                if not rows:
                    tx.rollback()
                    logger.debug(
                        "persist_reviewed_docs: stage/token mismatch for %s "
                        "(token=%s) — no-op",
                        sn_id,
                        claim_token[:8],
                    )
                    return ""

                docs_chain_length = int(rows[0]["docs_chain_length"])
                documentation = rows[0].get("documentation")
                edit_status_before = rows[0].get("edit_status")

                if score >= min_score:
                    target_stage = "accepted"
                elif docs_chain_length >= rotation_cap:
                    target_stage = "exhausted"
                else:
                    target_stage = "reviewed"
                if quorum_shortfall is not None and target_stage == "accepted":
                    target_stage = "reviewed"

                # Link findings join the worker findings in this exact review
                # transaction. A failed scan did not check the link kind.
                if target_stage == "accepted":
                    try:
                        mismatches = foundation.get_services()._doc_link_mismatches(
                            graph, documentation
                        )
                    except Exception:
                        logger.debug(
                            "persist_reviewed_docs: link-mismatch scan failed for %s "
                            "(non-fatal; accepting)",
                            sn_id,
                            exc_info=True,
                        )
                        mismatches = []
                    else:
                        checked_kinds.add("link_target_mismatch")
                    if mismatches:
                        note = (
                            "link text names a different standard name than its target "
                            "resolves to — make each label match its target id (or "
                            "reword to plain prose): " + "; ".join(mismatches)
                        )
                        comments = f"{comments}\n{note}" if comments else note
                        comments_per_dim = {
                            **(comments_per_dim or {}),
                            "link_integrity": note,
                        }
                        pending_holds.append(
                            {
                                "kind": "link_target_mismatch",
                                "finding": note,
                                "evidence": "; ".join(mismatches),
                                "raised_by": "doc-link-target-check",
                            }
                        )
                        logger.info(
                            "persist_reviewed_docs: %s held — %d mismatched doc link(s)",
                            sn_id,
                            len(mismatches),
                        )

                sync_holds(sn_id, "docs", checked_kinds, pending_holds, graph=graph)

                if (
                    target_stage == "accepted"
                    and not foundation.get_services().docs_accepted(
                        score, min_score, sn_id, graph=graph
                    )
                ):
                    # Keep a held passing document triageable even at the cap.
                    target_stage = "reviewed"

                if target_stage == "accepted":
                    foundation.get_services()._normalize_bare_doc_links(
                        graph, sn_id=sn_id, documentation=documentation
                    )

                new_edit_status = edit_status_before
                if edit_status_before == "open" and target_stage == "exhausted":
                    new_edit_status = "exhausted"
                stage_assignments = (
                    foundation.get_services().docs_acceptance_set()
                    if target_stage == "accepted"
                    else "sn.docs_stage = $target_stage, sn.edit_status = $new_edit_status"
                )

                comments_per_dim_json = (
                    _json.dumps(comments_per_dim)
                    if comments_per_dim is not None
                    else None
                )
                write_rows = graph.query(
                    f"""
                    MATCH (sn:StandardName {{id: $id}})
                    WHERE (($claim_seq IS NULL
                            AND (sn.claim_token = $token OR sn.claim_token IS NULL))
                           OR (sn.claim_token = $token
                               AND sn.claim_seq = $claim_seq))
                      AND sn.name_stage = 'accepted'
                      AND sn.docs_stage = 'drafted'
                    SET sn.updated_at = datetime(), sn.reviewer_score_docs = $score,
                        sn.reviewer_scores_docs = $scores_json,
                        sn.reviewer_comments_docs = $comments,
                        sn.reviewer_comments_per_dim_docs = $comments_per_dim_json,
                        sn.reviewer_model_docs = $model,
                        sn.docs_review_quorum_shortfall = $quorum_shortfall,
                        sn.docs_review_quorum_shortfall_at =
                            CASE WHEN $quorum_shortfall IS NULL
                                 THEN null ELSE datetime() END,
                        sn.reviewed_docs_at = datetime(),
                        {stage_assignments},
                        sn.claim_token = null,
                        sn.claimed_at = null
                    RETURN sn.id AS id
                    """,
                    id=sn_id,
                    token=claim_token,
                    claim_seq=claim_seq,
                    score=score,
                    scores_json=scores_json,
                    comments=comments,
                    comments_per_dim_json=comments_per_dim_json,
                    model=model,
                    quorum_shortfall=quorum_shortfall,
                    target_stage=target_stage,
                    new_edit_status=new_edit_status,
                )
                if not write_rows:
                    tx.rollback()
                    logger.debug(
                        "persist_reviewed_docs: %s transitioned before the "
                        "claim-locked write — no-op",
                        sn_id,
                    )
                    return ""
                tx.commit()
            except BaseException:
                tx.rollback()
                raise
            finally:
                tx.close()

    logger.info(
        "persist_reviewed_docs: %s → docs_stage=%s "
        "(score=%.3f, chain=%d/%d, resolution=%s, shortfall=%s)",
        sn_id,
        target_stage,
        score,
        docs_chain_length,
        rotation_cap,
        resolution_method,
        quorum_shortfall,
    )

    # ── Write StandardNameReview node + HAS_REVIEW edge (Finding 1 fix) ──
    # When ``skip_review_node`` is True the caller (pool RD-quorum
    # worker) writes per-cycle Review nodes itself.
    if not skip_review_node:
        try:
            import uuid as _uuid
            from datetime import datetime as _dt

            _now_iso = llm_at or _dt.now(foundation.get_services().UTC).isoformat()
            _group_id = str(_uuid.uuid4())
            _review_id = f"{sn_id}:docs:{_group_id}:0"
            if score >= 0.85:
                _tier = "outstanding"
            elif score >= 0.60:
                _tier = "good"
            elif score >= 0.40:
                _tier = "inadequate"
            else:
                _tier = "poor"
            foundation.get_services().write_reviews(
                [
                    {
                        "id": _review_id,
                        "standard_name_id": sn_id,
                        "model": model,
                        "reviewer_model": model,
                        "model_family": "other",
                        "is_canonical": True,
                        "score": float(score),
                        "scores_json": scores_json or "{}",
                        "tier": _tier,
                        "comments": comments or "",
                        "comments_per_dim_json": comments_per_dim_json,
                        "suggested_name": "",
                        "suggestion_justification": "",
                        "reviewed_at": _now_iso,
                        "review_axis": "docs",
                        "cycle_index": 0,
                        "review_group_id": _group_id,
                        "resolution_role": "primary",
                        "resolution_method": resolution_method,
                        "llm_model": model,
                        "llm_cost": llm_cost,
                        "llm_tokens_in": llm_tokens_in,
                        "llm_tokens_out": llm_tokens_out,
                        "llm_tokens_cached_read": llm_tokens_cached_read,
                        "llm_tokens_cached_write": llm_tokens_cached_write,
                        "llm_at": _now_iso,
                        "llm_service": llm_service or "standard-names",
                    }
                ],
            )
        except Exception:
            logger.exception(
                "persist_reviewed_docs: failed to write StandardNameReview for %s",
                sn_id,
            )

    # Async counter bump — live progress visibility for ``sn status``
    foundation.get_services().bump_sn_run_counter(run_id, "names_reviewed")

    return target_stage


STAGE_DOCS_FOR_RESCORE_QUERY = """
MATCH (sn:StandardName {id: $id})
WHERE sn.name_stage = 'accepted'
  AND sn.docs_stage IN ['accepted', 'reviewed', 'exhausted']
  AND sn.claim_token IS NULL
  AND sn.claimed_at IS NULL
  AND (NOT $require_unscoped OR sn.run_id IS NULL)
  AND sn.drain_scope_id IS NULL
  AND sn.drain_scope_claimed_at IS NULL
  AND sn.drain_claim_scope_id IS NULL
WITH sn, sn.docs_stage AS prior_stage
FOREACH (_ IN CASE WHEN $dry_run THEN [] ELSE [1] END |
    SET sn.updated_at = datetime(), sn.docs_stage = 'drafted',
        sn.reviewer_score_docs = null,
        sn.reviewer_scores_docs = null,
        sn.reviewer_comments_docs = null,
        sn.reviewer_comments_per_dim_docs = null,
        sn.reviewer_model_docs = null,
        sn.docs_review_quorum_shortfall = null,
        sn.docs_review_quorum_shortfall_at = null,
        sn.reviewed_docs_at = null,
        sn.run_id = $run_id
)
RETURN prior_stage AS prior_stage,
       sn.description AS description,
       sn.documentation AS documentation
"""

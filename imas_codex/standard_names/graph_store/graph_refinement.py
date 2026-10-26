"""Graph storage operations for the standard-name pipeline.

Refined name identity, rename and embedding persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import suppress
from enum import StrEnum
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
)

from .graph_foundation import (
    _CLAIM_TIMEOUT_SECONDS,
    DEFAULT_POOL_BATCH_SIZE,
    _TransactionQuery,
)

logger = logging.getLogger(__name__)


def _docs_rescore_ineligible(sn_id: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": (
            f"{sn_id!r} is not docs-rescore eligible "
            f"(name_stage={state.get('name_stage')!r}, "
            f"docs_stage={state.get('docs_stage')!r}, "
            f"claimed={bool(state.get('claim_token') or state.get('claimed_at'))}, "
            "drain_scoped="
            f"{any(state.get(field) is not None for field in ('drain_scope_id', 'drain_scope_claimed_at', 'drain_claim_scope_id'))})"
        ),
    }


def stage_docs_for_rescore(
    sn_id: str,
    *,
    run_id: str,
    dry_run: bool = False,
    services: Any,
) -> dict[str, Any]:
    """Stage one exact documentation record for a fresh quorum review.

    This is the compare-and-set recovery primitive for docs whose stored
    aggregate decision lacks sufficient reviewer authority. It accepts only a
    name whose name axis is accepted, whose docs are in an operator-recoverable
    terminal stage, and which has no active claim. The documentation text,
    description, refinement depth, and ``StandardNameReview`` history are
    preserved. Only the docs stage and aggregate docs-review decision fields
    are reset, and ``run_id`` fences the next ``review_docs`` claim to this
    exact record.

    Returns ``{"ok": True, ...}`` on an eligible transition, or a fail-closed
    diagnostic when the record is absent, live, claimed, or not name-accepted.
    A dry run executes the same eligibility predicate without mutation.
    """
    if not run_id.strip():
        return {"ok": False, "reason": "run_id must be non-empty"}

    with services.GraphClient() as gc:
        rows = gc.query(
            services.STAGE_DOCS_FOR_RESCORE_QUERY,
            id=sn_id,
            run_id=run_id,
            dry_run=dry_run,
            require_unscoped=False,
        )
        if not rows:
            current = gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                RETURN sn.name_stage AS name_stage,
                       sn.docs_stage AS docs_stage,
                       sn.claim_token AS claim_token,
                       sn.claimed_at AS claimed_at,
                       sn.drain_scope_id AS drain_scope_id,
                       sn.drain_scope_claimed_at AS drain_scope_claimed_at,
                       sn.drain_claim_scope_id AS drain_claim_scope_id
                """,
                id=sn_id,
            )
            if not current:
                return {"ok": False, "reason": f"name {sn_id!r} not found"}
            state = current[0]
            return _docs_rescore_ineligible(sn_id, state)

    return {
        "ok": True,
        "sn_id": sn_id,
        "prior_stage": rows[0]["prior_stage"],
        "run_id": run_id,
        "dry_run": dry_run,
    }


REFINE_NAME_ATTEMPTS_SPENT = (
    "coalesce(sn.refine_attempts, coalesce(sn.chain_length, 0))"
)
"""Rotations already spent on a name.

Reads the durable attempt counter, falling back to the lineage depth for a
name minted before the counter existed — every persisted successor cost at
least one attempt, so the fallback never grants budget that was already spent.
"""


REFINE_NAME_ELIGIBILITY_WHERE = (
    "sn.name_stage = 'reviewed'"
    " AND sn.reviewer_score_name IS NOT NULL"
    " AND sn.reviewer_score_name < $min_score"
    # Budget is charged per CLAIMED attempt, not per persisted successor: a
    # proposal the graph cannot accept (the identity is taken, the persistence
    # fence refuses it, the candidate is ungrammatical) leaves chain_length at
    # zero, so gating on lineage depth lets such a name re-claim and re-bill on
    # every poll for as long as the run lasts.
    f" AND {REFINE_NAME_ATTEMPTS_SPENT} < $rotation_cap"
    " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
    # A pinned rename that has already spent its re-review budget rests at
    # 'reviewed' — refine must not re-claim it (it is never rewritten, only
    # resubmitted; see resubmit_pinned_rename_for_review). Under the cap it
    # is claimed so the resubmit-to-review can fire.
    " AND NOT (coalesce(sn.edit_mode, '') = 'rename'"
    "          AND coalesce(sn.review_resubmit_count, 0) >= $rotation_cap)"
    # Derived parents are structurally fixed and have no refinement target.
    " AND coalesce(sn.origin, '') <> 'derived'"
    # A score the reviewer seats never agreed on is not evidence that the NAME
    # needs rewriting, so it must not buy a paid refine rotation — a rotation
    # spent on it can drive a sound name into a rewrite and terminate it.
    # The name rests here until `sn rescore` re-reviews it against a healthy
    # chain, which clears the marker.
    " AND sn.review_quorum_shortfall IS NULL"
)
"""Canonical graph predicate for name-refinement eligibility."""


@retry_on_deadlock()
def _charge_refine_name_attempts(
    items: list[dict[str, Any]],
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Spend one refinement rotation on every claimed name.

    Fenced on the claim token and the ``'refining'`` stage, so only the worker
    that owns the row charges it. Each surviving item carries the committed
    ``refine_attempts`` value, which numbers this attempt: the caller uses it
    to decide escalation and to report progress as attempt k of the cap.

    Items whose charge does not commit lost the claim between verification and
    this write; they are dropped so no model call is made against a fence
    another worker now holds. An item carrying no claim token has no fence to
    charge or to drop against, so it passes through uncharged — the same
    tolerance the claim-race verifier applies.
    """
    if not items:
        return []
    charged: dict[str, int] = {}
    by_token: dict[str, list[str]] = {}
    for item in items:
        token = item.get("claim_token")
        if token:
            by_token.setdefault(str(token), []).append(item["id"])
    if not by_token:
        return items
    with services.GraphClient() as gc:
        for token, sn_ids in by_token.items():
            rows = gc.query(
                """
                UNWIND $sn_ids AS sid
                MATCH (sn:StandardName {id: sid})
                WHERE sn.claim_token = $token
                  AND sn.name_stage = 'refining'
                SET sn.updated_at = datetime(), sn.refine_attempts = coalesce(
                        sn.refine_attempts, coalesce(sn.chain_length, 0)
                    ) + 1
                RETURN sn.id AS id, sn.refine_attempts AS refine_attempts
                """,
                sn_ids=sn_ids,
                token=token,
            )
            for row in rows:
                charged[row["id"]] = int(row["refine_attempts"])
    kept: list[dict[str, Any]] = []
    for item in items:
        if not item.get("claim_token"):
            kept.append(item)
            continue
        attempts = charged.get(item["id"])
        if attempts is None:
            logger.debug(
                "claim_refine_name_batch: %s lost its claim before the attempt "
                "was charged — dropping",
                item["id"],
            )
            continue
        item["refine_attempts"] = attempts
        kept.append(item)
    return kept


@retry_on_deadlock()
def claim_refine_name_batch(
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name refinement (Option B chain creation).

    Eligibility: ``name_stage = 'reviewed'`` AND
    ``reviewer_score_name < min_score`` AND ``chain_length < rotation_cap``.

    The claim atomically transitions ``name_stage`` from ``'reviewed'``
    to ``'refining'`` via :func:`_claim_sn_atomic`.

    After claiming, each item is enriched with REFINED_FROM chain history
    via :func:`~imas_codex.standard_names.chain_history.name_chain_history`.

    Returns claimed items as dicts with chain_history appended.
    """
    from imas_codex.standard_names.chain_history import name_chain_history

    items = services._claim_sn_atomic(
        eligibility_where=REFINE_NAME_ELIGIBILITY_WHERE,
        query_params={"min_score": min_score, "rotation_cap": rotation_cap},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.reviewer_score_name AS reviewer_score_name"
            ", sn.reviewer_comments_per_dim_name"
            "     AS reviewer_comments_per_dim_name"
            ", sn.chain_length AS chain_length"
            ", sn.refine_stop_reason AS refine_stop_reason"
            ", sn.refine_collision_name AS refine_collision_name"
            ", sn.name_stage AS name_stage"
            ", sn.source_paths AS source_paths"
            ", sn.tags AS tags"
            ", sn.vocab_gap_detail AS vocab_gap_detail"
            ", sn.validation_issues AS validation_issues"
            ", sn.edit_mode AS edit_mode"
            ", sn.name_hint AS name_hint"
            ", sn.docs_hint AS docs_hint"
            ", sn.edit_reason AS edit_reason"
            ", sn.edit_origin AS edit_origin"
            ", sn.run_id AS scope_run_id"
        ),
        stage_field="name_stage",
        to_stage="refining",
        domain=domain,
        scope_run_id=scope_run_id,
        drain_scope_id=drain_scope_id,
        edits_only=edits_only,
    )

    # Drop claim-race losers BEFORE the (paid) refine LLM call. The claim
    # transitions name_stage 'reviewed'→'refining', but the shared seed's
    # MATCH-bound row + lock-serialised SET still let concurrent replicas
    # each transition the same node; only the claim_token winner truly owns
    # it. See _verify_name_claim_winners.
    items = services._verify_name_claim_winners(items, eligible_stage="refining")

    # Charge the attempt budget once ownership is proven and before any model
    # call, so the spend is durable across a crash or an orphan sweep and a
    # race loser — which never calls a model — is not billed for the winner's
    # attempt. A row that cannot be charged has lost its claim in the interim
    # and is dropped rather than refined under a stale fence.
    items = _charge_refine_name_attempts(items, services=services)

    # Enrich each claimed item with its REFINED_FROM chain history and build
    # a unified prior_reviews list that also includes the current node's own
    # reviewer feedback.  Without this, a name cycled >1 time loses the
    # intermediate reviewer verdict when the next refine prompt is rendered.
    for item in items:
        item["chain_history"] = name_chain_history(item["id"])
        raw = item.get("reviewer_comments_per_dim_name") or "{}"
        try:
            per_dim: dict = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (ValueError, TypeError):
            per_dim = {}
        current_review: dict[str, Any] = {
            "name": item["id"],
            "model": item.get("model", "unknown"),
            "reviewer_score": item.get("reviewer_score_name"),
            "reviewer_comments_per_dim": per_dim,
        }
        item["prior_reviews"] = item["chain_history"] + [current_review]

    return items


# =============================================================================
# Shared embedding helper for individual StandardName nodes
# =============================================================================


def _embed_single_standard_name(
    sn_id: str, description: str | None, *, services: Any
) -> None:
    """Compute and persist embedding for a single StandardName node.

    Uses the same "name — description" format as
    :func:`persist_generated_name_batch`.  Failures are logged but never
    propagated — the SN is still usable without an embedding.
    """
    try:
        from imas_codex.embeddings.description import embed_descriptions_batch

        embed_text = f"{sn_id} — {description}" if description else sn_id
        items: list[dict[str, Any]] = [{"id": sn_id, "_embed_text": embed_text}]
        embed_descriptions_batch(items, text_field="_embed_text")
        vec = items[0].get("embedding")
        if vec is not None:
            with services.GraphClient() as gc:
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $id})
                    SET sn.updated_at = datetime(), sn.embedding    = $embedding,
                        sn.embedded_at  = datetime()
                    """,
                    id=sn_id,
                    embedding=vec,
                )
            logger.debug("_embed_single_standard_name: embedded %s", sn_id)
        else:
            logger.warning(
                "_embed_single_standard_name: embedding returned None for %s",
                sn_id,
            )
    except Exception:
        logger.warning(
            "Failed to embed StandardName %s — will retry later",
            sn_id,
            exc_info=True,
        )


# =============================================================================
# Embed worker — claim / release / persist for dedicated embed pool
# =============================================================================


def _compute_embed_hash(sn_id: str, description: str | None) -> str:
    """Compute truncated SHA-256 hash of the embed text.

    Format: ``"name — description"`` when description is available,
    otherwise just the name.  Truncated to 16 hex chars.
    """
    text = f"{sn_id} — {description}" if description else sn_id
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@retry_on_deadlock()
def claim_embed_batch(
    limit: int = 50,
    scope_run_id: str | None = None,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes needing (re-)embedding.

    Targets nodes where:
    - ``embedding IS NULL`` or ``embed_text_hash IS NULL``
    - Not already claimed for embedding (or claim is stale >5min)

    Uses ``embed_claimed_at`` / ``embed_claim_token`` which are
    independent of the main ``claimed_at`` / ``claim_token`` used
    for stage transitions.
    """
    token = str(uuid.uuid4())
    scope_where = ""
    scope_params: dict[str, Any] = {}
    if scope_run_id:
        scope_where = "AND sn.run_id = $scope_run_id"
        scope_params["scope_run_id"] = scope_run_id
    with services.GraphClient() as gc:
        gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE (sn.embedding IS NULL OR sn.embed_text_hash IS NULL)
              AND (sn.embed_claimed_at IS NULL
                   OR sn.embed_claimed_at < datetime() - duration('PT5M'))
              {scope_where}
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.updated_at = datetime(),
                sn.embed_claimed_at = datetime(),
                sn.embed_claim_token = $token
            """,
            limit=limit,
            token=token,
            **scope_params,
        )
        return list(
            gc.query(
                """
                MATCH (sn:StandardName {embed_claim_token: $token})
                RETURN sn.id AS id,
                       sn.description AS description,
                       $token AS claim_token
                """,
                token=token,
            )
        )


def release_embed_claims(
    sn_ids: list[str],
    claim_token: str,
    *,
    services: Any,
) -> int:
    """Release embed claims IFF token matches.

    Called on error to unlock nodes for other embed workers.
    """
    if not sn_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid, embed_claim_token: $token})
            SET sn.updated_at = datetime(), sn.embed_claimed_at = null,
                sn.embed_claim_token = null
            RETURN count(sn) AS released
            """,
            ids=sn_ids,
            token=claim_token,
        )
        return result[0]["released"] if result else 0


def persist_embed_batch(items: list[dict[str, Any]], *, services: Any) -> int:
    """Write embedding vectors and metadata for a batch of StandardNames.

    Each item must have ``id``, ``embedding``, and ``embed_text_hash``.
    Clears embed claim fields on success.
    """
    if not items:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $items AS item
            MATCH (sn:StandardName {id: item.id})
            SET sn.updated_at = datetime(), sn.embedding = item.embedding,
                sn.embedded_at = datetime(),
                sn.embed_text_hash = item.embed_text_hash,
                sn.embed_failed_at = null,
                sn.embed_claimed_at = null,
                sn.embed_claim_token = null
            RETURN count(sn) AS written
            """,
            items=[
                {
                    "id": it["id"],
                    "embedding": it["embedding"],
                    "embed_text_hash": it["embed_text_hash"],
                }
                for it in items
            ],
        )
        return result[0]["written"] if result else 0


# =============================================================================
# Persist — refine_name (Option B: new node + REFINED_FROM + edge migration)
# =============================================================================


class RefinedNamePersistenceRefusalReason(StrEnum):
    """Exact mechanism that prevented a refined successor write."""

    PREDECESSOR_MISSING = "predecessor_missing"
    PREDECESSOR_STAGE = "predecessor_stage_changed"
    CLAIM_LOST = "claim_lost"
    AUTHORITATIVE_SOURCE_COHORT_EMPTY = "authoritative_source_cohort_empty"
    SUCCESSOR_LIFECYCLE = "successor_lifecycle_collision"
    SUCCESSOR_SCOPE = "successor_scope_collision"
    SUCCESSOR_NOT_PERSISTED = "successor_not_persisted"


class RefinedNamePersistenceRefusal(RuntimeError):
    """Typed refusal carrying both the proposed and existing identities."""

    def __init__(
        self,
        *,
        old_name: str,
        proposed_name: str,
        reason: RefinedNamePersistenceRefusalReason,
        existing_name: str | None = None,
    ) -> None:
        self.old_name = old_name
        self.proposed_name = proposed_name
        self.reason = reason
        self.existing_name = existing_name
        detail = f" existing={existing_name!r}" if existing_name else ""
        super().__init__(
            f"refined successor refused: predecessor={old_name!r} "
            f"proposed={proposed_name!r} reason={reason.value}{detail}"
        )


def _meaning_preservation_assertion(
    *,
    reason: str | None,
    asserted_by: str | None,
) -> tuple[str, str] | None:
    """Validate an explicit assertion that a rename preserves meaning."""
    if reason is None and asserted_by is None:
        return None

    normalized_reason = (reason or "").strip()
    normalized_actor = (asserted_by or "").strip()
    if not normalized_reason:
        raise ValueError(
            "a meaning-preservation assertion requires a substantive reason"
        )
    if not normalized_actor:
        raise ValueError(
            "a meaning-preservation assertion requires the asserting actor"
        )
    return normalized_reason, normalized_actor


def _predecessor_document_identity(name: str, documentation: str) -> str:
    """Return the content-addressed identity of one predecessor document."""
    digest = hashlib.sha256(documentation.encode("utf-8")).hexdigest()
    return f"standard-name-document:{name}:sha256:{digest}"


def rename_preserves_meaning(
    old_name: str,
    new_name: str,
    *,
    assertion_reason: str | None = None,
    asserted_by: str | None = None,
) -> bool:
    """True when two spellings denote the same quantity.

    The grammar's intermediate representation is the meaning; the string is
    one rendering of it. Exact IR equality therefore preserves meaning. A
    locus-token substitution also preserves meaning when the locus relation,
    type, qualifiers, and value stay fixed and every non-locus part of the IR
    is unchanged.

    A spelling either side cannot parse yields no evidence of sameness, and
    absence of evidence answers False here: carrying an accepted document
    across a rename that might have changed the meaning would publish a
    statement nobody wrote about the successor. A caller may instead provide
    a reasoned, attributed assertion. That explicit authority is distinct from
    grammar inference and must be recorded by the persistence operation.
    """
    assertion = _meaning_preservation_assertion(
        reason=assertion_reason,
        asserted_by=asserted_by,
    )
    if assertion is not None:
        return True

    from imas_standard_names.grammar import parser as isn_parser

    try:
        old_ir = isn_parser.parse(old_name).ir
        new_ir = isn_parser.parse(new_name).ir
        if old_ir == new_ir:
            return True

        old_locus = old_ir.locus
        new_locus = new_ir.locus
        if (
            old_locus is None
            or new_locus is None
            or old_locus.token == new_locus.token
            or old_locus.relation != new_locus.relation
            or old_locus.type != new_locus.type
            or old_locus.qualifiers != new_locus.qualifiers
            or old_locus.value != new_locus.value
        ):
            return False

        return (
            old_ir.model_copy(
                update={
                    "locus": old_locus.model_copy(update={"token": new_locus.token})
                }
            )
            == new_ir
        )
    except Exception:
        return False


def _classify_refined_name_persistence_refusal(
    tx: Any,
    *,
    old_name: str,
    new_name: str,
    expected_old_stage: str | None,
    expected_claim_token: str | None,
    edit_mode: str | None,
) -> RefinedNamePersistenceRefusal:
    """Read the failed transaction predicate and return its exact refusal."""
    rows = list(
        tx.run(
            """
            OPTIONAL MATCH (old:StandardName {id: $old_name})
            OPTIONAL MATCH (existing:StandardName {id: $new_name})
            RETURN old IS NOT NULL AS old_exists,
                   old.name_stage AS old_stage,
                   old.claim_token AS old_claim_token,
                   old.drain_scope_id AS old_drain_scope_id,
                   existing.id AS existing_id,
                   existing.name_stage AS existing_stage,
                   existing.origin AS existing_origin,
                   existing.drain_scope_id AS existing_drain_scope_id,
                   existing.drain_scope_claimed_at
                       AS existing_drain_scope_claimed_at,
                   (old.drain_scope_id IS NULL
                    OR existing.drain_scope_id IS NULL
                    OR existing.drain_scope_id = old.drain_scope_id
                    OR existing.drain_scope_claimed_at IS NULL
                    OR existing.drain_scope_claimed_at < datetime()
                         - duration('PT600S')) AS successor_scope_compatible
            """,
            old_name=old_name,
            new_name=new_name,
        )
    )
    state = dict(rows[0]) if rows else {}
    existing_name = state.get("existing_id")
    if not state.get("old_exists"):
        reason = RefinedNamePersistenceRefusalReason.PREDECESSOR_MISSING
    else:
        required_stage = expected_old_stage or "refining"
        if state.get("old_stage") != required_stage:
            reason = RefinedNamePersistenceRefusalReason.PREDECESSOR_STAGE
        elif (
            expected_old_stage is None
            and expected_claim_token is not None
            and state.get("old_claim_token") != expected_claim_token
        ):
            reason = RefinedNamePersistenceRefusalReason.CLAIM_LOST
        elif existing_name:
            reusable = (
                edit_mode is None
                and (state.get("existing_stage") or "") in {"", "pending"}
                and (state.get("existing_origin") or "") != "derived"
            )
            if not reusable:
                reason = RefinedNamePersistenceRefusalReason.SUCCESSOR_LIFECYCLE
            else:
                reason = (
                    RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED
                    if state.get("successor_scope_compatible")
                    else RefinedNamePersistenceRefusalReason.SUCCESSOR_SCOPE
                )
        else:
            reason = RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED
    return RefinedNamePersistenceRefusal(
        old_name=old_name,
        proposed_name=new_name,
        reason=reason,
        existing_name=existing_name,
    )


@retry_on_deadlock()
def persist_refined_name(
    *,
    old_name: str,
    new_name: str,
    description: str,
    kind: str = "scalar",
    unit: str | None = None,
    physics_domain: str | None = None,
    source_domains: list[str] | None = None,
    tags: list[str] | None = None,
    old_chain_length: int = 0,
    model: str = "unknown",
    reason: str = "",
    escalated: bool = False,
    run_id: str | None = None,
    edit_mode: str | None = None,
    name_hint: str | None = None,
    docs_hint: str | None = None,
    edit_reason: str | None = None,
    edit_origin: str | None = None,
    edit_scope: str | None = None,
    edit_status: str | None = None,
    edit_requested_at: str | None = None,
    edit_override_edits: bool | None = None,
    edit_include_accepted: bool | None = None,
    expected_old_stage: str | None = None,
    expected_claim_token: str | None = None,
    meaning_preservation_reason: str | None = None,
    meaning_preservation_asserted_by: str | None = None,
    services: Any,
) -> dict[str, str]:
    """Persist a refined StandardName as a NEW node with source-edge migration.

    Since ``StandardName.id`` is the name string, refining a name produces a
    new node identity. In a single transaction:

    1. Compare-and-set the predecessor stage and claim, then provision a
       drafted successor carrying the unit and semantic fields the attachment
       guard needs. When the two spellings parse to the same grammar IR the
       rename is a re-rendering, so the successor also inherits the
       predecessor's documentation and its whole docs lifecycle
       (``docs_stage``, ``docs_chain_length``, ``docs_model``,
       ``docs_generated_at``) — the description already carried across
       unchanged, and the two prose fields describe one meaning. Any other
       rename leaves the docs axis at ``pending`` with no text, because the
       predecessor's document was written about a different meaning. A caller
       may explicitly assert preserved meaning only with a substantive reason
       and asserting actor; that route writes a separate provenance receipt.
    2. Require the full authoritative source set to pass the attachment guard.
    3. Create lineage, supersede the predecessor, and migrate every source
       edge, scalar, upstream projection, and source-path cache.
    4. Record one internal rename/refine event and commit all writes together.

    Returns ``{"new_name": <new_id>, "old_name": <old_id>}``.

    Raises ``ValueError`` if ``new_name == old_name`` — self-referential
    refinement would create a ``REFINED_FROM`` self-loop, or if any source is
    inconsistent with the successor. An empty authoritative source cohort on
    an automated refine, a failed stage/claim CAS, partial source migration,
    or ledger failure also rolls the complete transaction back.

    Edit-steering fields (``edit_mode``, ``name_hint``, ``docs_hint``,
    ``edit_reason``, ``edit_origin``, ``edit_scope``, ``edit_status``,
    ``edit_requested_at``) are set on the new successor node when passed
    explicitly (the ``imas-codex sn edit`` rename-mode caller).  When the
    caller passes none of them, this function instead checks whether the
    predecessor (``old_name``) carries a still-open edit
    (``edit_status='open'``) and, if so, copies its edit fields forward —
    a refine rotation of an edited name must not silently drop the
    steering + reason that got it there.
    """
    if new_name == old_name:
        raise ValueError(
            f"Refine produced identical name '{new_name}' — "
            f"cannot create self-referential REFINED_FROM edge"
        )
    from imas_codex.standard_names.kind_derivation import derive_kind

    derived_kind = derive_kind(new_name)
    if derived_kind != kind:
        logger.info(
            "Refined successor %s kind corrected from %s to %s",
            new_name,
            kind,
            derived_kind,
        )
    kind = derived_kind
    new_chain_length = old_chain_length + 1

    inherit_open_edit = (
        edit_mode is None and edit_reason is None and edit_status is None
    )
    meaning_assertion = _meaning_preservation_assertion(
        reason=meaning_preservation_reason,
        asserted_by=meaning_preservation_asserted_by,
    )
    meaning_preserved = rename_preserves_meaning(old_name, new_name)
    if meaning_assertion is not None:
        meaning_preserved = rename_preserves_meaning(
            old_name,
            new_name,
            assertion_reason=meaning_preservation_reason,
            asserted_by=meaning_preservation_asserted_by,
        )

    escalation_set = ""
    if escalated:
        escalation_set = ", new.refine_name_escalated_at = datetime()"

    with services.GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                preflight = list(
                    tx.run(
                        f"""
                        // REFINE_ATOMIC_PREFLIGHT
                        MATCH (old:StandardName {{id: $old_name}})
                        WHERE old.name_stage = coalesce(
                            $expected_old_stage, 'refining'
                        )
                          AND (
                            ($expected_old_stage IS NOT NULL
                             AND old.claim_token IS NULL
                             AND old.claimed_at IS NULL)
                            OR ($expected_old_stage IS NULL
                                AND ($expected_claim_token IS NULL
                                     OR old.claim_token = $expected_claim_token))
                          )
                        OPTIONAL MATCH (existing:StandardName {{id: $new_name}})
                        WITH old, existing
                        WHERE existing IS NULL
                           OR ($edit_mode IS NULL
                               AND coalesce(existing.name_stage, '') IN ['', 'pending']
                               AND coalesce(existing.origin, '') <> 'derived'
                               AND (old.drain_scope_id IS NULL
                                    OR existing.drain_scope_id IS NULL
                                    OR existing.drain_scope_id = old.drain_scope_id
                                    OR existing.drain_scope_claimed_at IS NULL
                                    OR existing.drain_scope_claimed_at < datetime()
                                         - duration('PT600S')))
                        WITH old, existing,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_mode ELSE $edit_mode END
                               AS successor_edit_mode,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.name_hint ELSE $name_hint END
                               AS successor_name_hint,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.docs_hint ELSE $docs_hint END
                               AS successor_docs_hint,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_reason ELSE $edit_reason END
                               AS successor_edit_reason,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_origin ELSE $edit_origin END
                               AS successor_edit_origin,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_scope ELSE $edit_scope END
                               AS successor_edit_scope,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_status ELSE $edit_status END
                               AS successor_edit_status,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_requested_at
                                  ELSE $edit_requested_at END
                               AS successor_edit_requested_at,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_override_edits
                                  ELSE $edit_override_edits END
                               AS successor_edit_override_edits,
                             CASE WHEN $inherit_open_edit
                                        AND old.edit_status = 'open'
                                  THEN old.edit_include_accepted
                                  ELSE $edit_include_accepted END
                               AS successor_edit_include_accepted
                        SET old.updated_at = datetime(), old.superseded_from_stage = coalesce(
                                old.superseded_from_stage, old.name_stage
                            ),
                            old.name_stage = 'refining'
                        MERGE (new:StandardName {{id: $new_name}})
                        ON CREATE SET new.updated_at = datetime(),
                          new.name_stage        = 'drafted',
                          new.status            = 'draft',
                          // Documentation is prose written about a meaning,
                          // so it travels with the meaning and not with the
                          // spelling. A re-rendering keeps the predecessor's
                          // text and its whole docs lifecycle — the identity
                          // moved, nothing it documents did. A rename that
                          // states something else starts the docs axis over,
                          // because publishing the old document under the new
                          // meaning would assert what nobody wrote.
                          new.docs_stage        = CASE
                            WHEN $meaning_preserved
                             AND coalesce(old.documentation, '') <> ''
                            THEN coalesce(old.docs_stage, 'pending')
                            ELSE 'pending' END,
                          new.documentation     = CASE
                            WHEN $meaning_preserved THEN old.documentation
                            ELSE null END,
                          new.docs_model        = CASE
                            WHEN $meaning_preserved THEN old.docs_model
                            ELSE null END,
                          new.docs_generated_at = CASE
                            WHEN $meaning_preserved THEN old.docs_generated_at
                            ELSE null END,
                          new.validation_status = 'valid',
                          new.origin            = 'pipeline',
                          new.chain_length      = $new_chain_length,
                          // The budget follows the identity lineage, not the
                          // node: a successor born of the predecessor's last
                          // rotation has none left, so the cap bounds the
                          // whole chain instead of resetting at every link.
                          new.refine_attempts   = coalesce(
                            old.refine_attempts, $new_chain_length
                          ),
                          new.docs_chain_length = CASE
                            WHEN $meaning_preserved
                            THEN coalesce(old.docs_chain_length, 0)
                            ELSE 0 END,
                          new.description       = $description,
                          new.kind              = $kind,
                          new.unit              = $unit,
                          new.physics_domain    = $physics_domain,
                          new.source_domains    = $source_domains,
                          new.tags              = $tags,
                          new.model             = $model,
                          new.created_at        = datetime(),
                          new.generated_at      = datetime(),
                          new.refine_reason     = $reason,
                          new.run_id            = $run_id,
                          new.drain_scope_id     = old.drain_scope_id,
                          new.drain_scope_claimed_at =
                            old.drain_scope_claimed_at,
                          new.drain_scope_paths = old.drain_scope_paths,
                          new.edit_mode         = successor_edit_mode,
                          new.name_hint         = successor_name_hint,
                          new.docs_hint         = successor_docs_hint,
                          new.edit_reason       = successor_edit_reason,
                          new.edit_origin       = successor_edit_origin,
                          new.edit_scope        = successor_edit_scope,
                          new.edit_status       = successor_edit_status,
                          new.edit_requested_at = successor_edit_requested_at,
                          new.edit_override_edits = successor_edit_override_edits,
                          new.edit_include_accepted = successor_edit_include_accepted,
                          new.embed_text_hash   = null
                          {escalation_set}
                        ON MATCH SET
                          new.updated_at = datetime(),
                          new.name_stage = CASE
                            WHEN coalesce(new.name_stage, '') IN ['', 'pending']
                             AND coalesce(new.origin, '') <> 'derived'
                            THEN 'drafted' ELSE new.name_stage END,
                          new.origin = CASE
                            WHEN coalesce(new.name_stage, '') IN ['', 'pending']
                             AND coalesce(new.origin, '') <> 'derived'
                            THEN 'pipeline' ELSE new.origin END,
                          new.edit_mode = successor_edit_mode,
                          new.name_hint = successor_name_hint,
                          new.docs_hint = successor_docs_hint,
                          new.edit_reason = successor_edit_reason,
                          new.edit_origin = successor_edit_origin,
                          new.edit_scope = successor_edit_scope,
                          new.edit_status = successor_edit_status,
                          new.edit_requested_at = successor_edit_requested_at,
                          new.edit_override_edits = successor_edit_override_edits,
                          new.edit_include_accepted = successor_edit_include_accepted,
                          new.refine_attempts = coalesce(
                            old.refine_attempts, $new_chain_length),
                          new.drain_scope_id = coalesce(
                            old.drain_scope_id, new.drain_scope_id),
                          new.drain_scope_claimed_at = CASE
                            WHEN old.drain_scope_id IS NOT NULL
                            THEN old.drain_scope_claimed_at
                            ELSE new.drain_scope_claimed_at END,
                          new.drain_scope_paths = CASE
                            WHEN old.drain_scope_id IS NOT NULL
                            THEN old.drain_scope_paths
                            ELSE new.drain_scope_paths END,
                          new.embed_text_hash = null
                        WITH old, new, successor_edit_mode,
                             successor_name_hint, successor_docs_hint,
                             successor_edit_reason, successor_edit_origin,
                             successor_edit_scope, successor_edit_status,
                             successor_edit_requested_at,
                             successor_edit_override_edits,
                             successor_edit_include_accepted
                        OPTIONAL MATCH (source:StandardNameSource)
                        WHERE (source)-[:PRODUCED_NAME]->(old)
                           OR source.produced_sn_id = old.id
                        RETURN old.id AS old_name, new.id AS new_name,
                               [source_id IN collect(DISTINCT source.id)
                                WHERE source_id IS NOT NULL] AS source_ids,
                               old.edit_mode AS source_edit_mode,
                               old.name_hint AS source_name_hint,
                               old.docs_hint AS source_docs_hint,
                               old.edit_reason AS source_edit_reason,
                               old.edit_origin AS source_edit_origin,
                               old.edit_scope AS source_edit_scope,
                               old.edit_status AS source_edit_status,
                               old.edit_requested_at AS source_edit_requested_at,
                               old.edit_override_edits
                                 AS source_edit_override_edits,
                               old.edit_include_accepted
                                 AS source_edit_include_accepted,
                               old.documentation AS source_documentation,
                               successor_edit_mode AS effective_edit_mode,
                               successor_name_hint AS effective_name_hint,
                               successor_docs_hint AS effective_docs_hint,
                               successor_edit_reason AS effective_edit_reason,
                               successor_edit_origin AS effective_edit_origin,
                               successor_edit_scope AS effective_edit_scope,
                               successor_edit_status AS effective_edit_status,
                               successor_edit_requested_at
                                 AS effective_edit_requested_at,
                               successor_edit_override_edits
                                 AS effective_edit_override_edits,
                               successor_edit_include_accepted
                                 AS effective_edit_include_accepted
                        """,
                        old_name=old_name,
                        new_name=new_name,
                        expected_old_stage=expected_old_stage,
                        expected_claim_token=expected_claim_token,
                        inherit_open_edit=inherit_open_edit,
                        meaning_preserved=meaning_preserved,
                        new_chain_length=new_chain_length,
                        description=description,
                        kind=kind,
                        unit=unit,
                        physics_domain=physics_domain,
                        source_domains=source_domains
                        or ([physics_domain] if physics_domain else []),
                        tags=tags or [],
                        model=model,
                        reason=reason,
                        run_id=run_id,
                        edit_mode=edit_mode,
                        name_hint=name_hint,
                        docs_hint=docs_hint,
                        edit_reason=edit_reason,
                        edit_origin=edit_origin,
                        edit_scope=edit_scope,
                        edit_status=edit_status,
                        edit_requested_at=edit_requested_at,
                        edit_override_edits=edit_override_edits,
                        edit_include_accepted=edit_include_accepted,
                    )
                )
                if not preflight:
                    raise _classify_refined_name_persistence_refusal(
                        tx,
                        old_name=old_name,
                        new_name=new_name,
                        expected_old_stage=expected_old_stage,
                        expected_claim_token=expected_claim_token,
                        edit_mode=edit_mode,
                    )

                preflight_row = dict(preflight[0])
                authoritative_cohort_observed = "source_ids" in preflight_row
                candidate_source_ids = sorted(
                    set(preflight_row.get("source_ids") or [])
                )
                edit_mode = preflight_row.get("effective_edit_mode", edit_mode)
                name_hint = preflight_row.get("effective_name_hint", name_hint)
                docs_hint = preflight_row.get("effective_docs_hint", docs_hint)
                edit_reason = preflight_row.get("effective_edit_reason", edit_reason)
                edit_origin = preflight_row.get("effective_edit_origin", edit_origin)
                edit_scope = preflight_row.get("effective_edit_scope", edit_scope)
                edit_status = preflight_row.get("effective_edit_status", edit_status)
                edit_requested_at = preflight_row.get(
                    "effective_edit_requested_at", edit_requested_at
                )
                edit_override_edits = preflight_row.get(
                    "effective_edit_override_edits", edit_override_edits
                )
                edit_include_accepted = preflight_row.get(
                    "effective_edit_include_accepted", edit_include_accepted
                )
                if (
                    authoritative_cohort_observed
                    and not candidate_source_ids
                    and not edit_mode
                ):
                    raise RefinedNamePersistenceRefusal(
                        old_name=old_name,
                        proposed_name=new_name,
                        reason=(
                            RefinedNamePersistenceRefusalReason.AUTHORITATIVE_SOURCE_COHORT_EMPTY
                        ),
                    )
                source_edit_state = {
                    field: preflight_row.get(f"source_{field}")
                    for field in (
                        "edit_mode",
                        "name_hint",
                        "docs_hint",
                        "edit_reason",
                        "edit_origin",
                        "edit_scope",
                        "edit_status",
                        "edit_requested_at",
                        "edit_override_edits",
                        "edit_include_accepted",
                    )
                }
                from imas_codex.standard_names.attachment_audit import (
                    guard_source_pairings,
                )

                query_handle = _TransactionQuery(tx)
                guarded = guard_source_pairings(
                    query_handle, new_name, candidate_source_ids
                )
                if guarded.rejected or set(guarded.accepted_source_ids) != set(
                    candidate_source_ids
                ):
                    rejected = (
                        ", ".join(
                            f"{item.source_node_id}: {item.reason}"
                            for item in guarded.rejected
                        )
                        or "the pairing guard did not admit every source"
                    )
                    raise ValueError(
                        f"refined-name attachment rejected; rename rolled back: "
                        f"{rejected}"
                    )

                result = list(
                    tx.run(
                        """
                        // Finalize the provisionally guarded successor.
                        MATCH (new:StandardName {id: $new_name}),
                              (old:StandardName {id: $old_name})
                        WHERE old.name_stage = 'refining'
                          AND (old.edit_mode = $source_edit_mode OR
                               (old.edit_mode IS NULL AND
                                $source_edit_mode IS NULL))
                          AND (old.name_hint = $source_name_hint OR
                               (old.name_hint IS NULL AND
                                $source_name_hint IS NULL))
                          AND (old.docs_hint = $source_docs_hint OR
                               (old.docs_hint IS NULL AND
                                $source_docs_hint IS NULL))
                          AND (old.edit_reason = $source_edit_reason OR
                               (old.edit_reason IS NULL AND
                                $source_edit_reason IS NULL))
                          AND (old.edit_origin = $source_edit_origin OR
                               (old.edit_origin IS NULL AND
                                $source_edit_origin IS NULL))
                          AND (old.edit_scope = $source_edit_scope OR
                               (old.edit_scope IS NULL AND
                                $source_edit_scope IS NULL))
                          AND (old.edit_status = $source_edit_status OR
                               (old.edit_status IS NULL AND
                                $source_edit_status IS NULL))
                          AND (old.edit_requested_at =
                               $source_edit_requested_at OR
                               (old.edit_requested_at IS NULL AND
                                $source_edit_requested_at IS NULL))
                          AND (old.edit_override_edits =
                               $source_edit_override_edits OR
                               (old.edit_override_edits IS NULL AND
                                $source_edit_override_edits IS NULL))
                          AND (old.edit_include_accepted =
                               $source_edit_include_accepted OR
                               (old.edit_include_accepted IS NULL AND
                                $source_edit_include_accepted IS NULL))
                        MERGE (new)-[:REFINED_FROM]->(old)

                        // 3. Mark old as superseded, clear claim
                        //
                        // Record whether the predecessor had reached the
                        // published bar so the export boundary can decide
                        // whether to emit a deprecation stub. The atomic
                        // preflight has flipped name_stage to 'refining', so
                        // the prior name-axis stage is carried by
                        // superseded_from_stage; the durable "was published"
                        // signal that survives that flip is docs_stage —
                        // a name only reaches consumers once its docs review
                        // has accepted it. A pipeline refine of a merely
                        // 'reviewed' name (never published) records a
                        // non-accepted sentinel, so no stub is emitted for it.
                        SET old.updated_at = datetime(), new.updated_at = datetime(), old.name_stage  = 'superseded',
                            old.superseded_from_stage = coalesce(
                                old.superseded_from_stage,
                                CASE WHEN coalesce(old.docs_stage, '') = 'accepted'
                                     THEN 'accepted' ELSE 'refining' END),
                            old.claim_token = null,
                            old.claimed_at  = null,
                            // A refine stays in the predecessor's scope: a
                            // scoped drain (--focus/--batch) binds names by
                            // run_id, so a successor minted with no run_id (the
                            // refine worker's run_id is unset in the pool path)
                            // would fall out of scope — the scoped review pool
                            // could never claim it and the drain would report
                            // no_eligible_work while the refined name sits
                            // stranded 'drafted'. Inherit the predecessor's
                            // run_id when the successor has none.
                            new.run_id = coalesce(new.run_id, old.run_id),
                            new.drain_scope_id = coalesce(
                                old.drain_scope_id, new.drain_scope_id),
                            new.drain_scope_claimed_at = CASE
                                WHEN old.drain_scope_id IS NOT NULL
                                THEN old.drain_scope_claimed_at
                                ELSE new.drain_scope_claimed_at END,
                            new.drain_scope_paths = CASE
                                WHEN old.drain_scope_id IS NOT NULL
                                THEN old.drain_scope_paths
                                ELSE new.drain_scope_paths END,
                            // Close the predecessor's edit lifecycle: a
                            // still-open steer was already carried forward onto
                            // the successor (see the copy-forward read above), so
                            // its intended change is realized by this rename.
                            // Leaving 'open' on a superseded (terminal) node
                            // orphans the edit — it can never resolve because the
                            // predecessor is no longer reviewable. Reconcile to
                            // 'applied', mirroring supersede_prior_source_names.
                            old.edit_status = CASE
                                WHEN coalesce(old.edit_status, '') = 'open'
                                THEN 'applied' ELSE old.edit_status END

                        // 4. Migrate inbound HAS_PARENT edges
                        // (child)-[:HAS_PARENT]->(old) → (child)-[:HAS_PARENT]->(new)
                        // Preserves edge properties so the SPA's parent
                        // widget points at the live successor, not the
                        // superseded predecessor.
                        // Note: MERGE inside FOREACH cannot reference map
                        // properties as node patterns (CYPHER_5 restriction).
                        // Use CALL subquery + UNWIND to bind child as a
                        // proper node variable before the MERGE.
                        WITH DISTINCT new, old
                        OPTIONAL MATCH (child)-[c_old:HAS_PARENT]->(old)
                        WITH new, old,
                             collect({
                                 child: child,
                                 props: properties(c_old),
                                 rel:   c_old
                             }) AS comp_links
                        FOREACH (link IN comp_links | DELETE link.rel)
                        WITH new, old, comp_links
                        CALL {
                            WITH new, comp_links
                            UNWIND comp_links AS link
                            WITH link.child AS ch, link.props AS p, new
                            WHERE ch IS NOT NULL AND ch.id <> new.id
                            MERGE (ch)-[c_new:HAS_PARENT]->(new)
                            SET   c_new = p
                        }

                        // 5. Inherit HAS_UNIT and IN_CLUSTER edges from the
                        // predecessor so that downstream claim-expand
                        // (which scopes by cluster+unit) can pick up the
                        // refined node. Without this, chain>0 SNs are
                        // statistically excluded from review-pool expand.
                        WITH DISTINCT new, old
                        OPTIONAL MATCH (old)-[:HAS_UNIT]->(u:Unit)
                        FOREACH (uu IN CASE WHEN u IS NULL THEN [] ELSE [u] END |
                            MERGE (new)-[:HAS_UNIT]->(uu))
                        WITH DISTINCT new, old
                        OPTIONAL MATCH (old)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
                        FOREACH (cc IN CASE WHEN c IS NULL THEN [] ELSE [c] END |
                            MERGE (new)-[:IN_CLUSTER]->(cc))

                        WITH DISTINCT new, old
                        RETURN new.id AS new_name, old.id AS old_name
                        """,
                        new_name=new_name,
                        old_name=old_name,
                        new_chain_length=new_chain_length,
                        description=description,
                        kind=kind,
                        unit=unit,
                        physics_domain=physics_domain,
                        source_domains=source_domains
                        or ([physics_domain] if physics_domain else []),
                        tags=tags or [],
                        model=model,
                        reason=reason,
                        run_id=run_id,
                        edit_mode=edit_mode,
                        name_hint=name_hint,
                        docs_hint=docs_hint,
                        edit_reason=edit_reason,
                        edit_origin=edit_origin,
                        edit_scope=edit_scope,
                        edit_status=edit_status,
                        edit_requested_at=edit_requested_at,
                        edit_override_edits=edit_override_edits,
                        edit_include_accepted=edit_include_accepted,
                        **{
                            f"source_{field}": value
                            for field, value in source_edit_state.items()
                        },
                    )
                )
                if not result:
                    raise RefinedNamePersistenceRefusal(
                        old_name=old_name,
                        proposed_name=new_name,
                        reason=(
                            RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED
                        ),
                        existing_name=new_name,
                    )

                from imas_codex.standard_names.provenance_lifecycle import (
                    record_standard_name_change,
                    retarget_standard_name_sources,
                )

                moved = retarget_standard_name_sources(
                    query_handle,
                    old_name,
                    new_name,
                    operation="human_edit" if edit_mode else "refine",
                    reason=edit_reason or reason,
                    origin=edit_origin,
                    run_id=run_id,
                    record_change=False,
                    enforce_consistency=False,
                    source_ids=candidate_source_ids,
                    expected_current_bindings=dict.fromkeys(
                        candidate_source_ids, old_name
                    ),
                    _allow_empty_noop=(
                        not authoritative_cohort_observed or bool(edit_mode)
                    ),
                )
                if moved != len(candidate_source_ids):
                    raise RuntimeError(
                        "semantic source set changed during atomic rename: "
                        f"guarded {len(candidate_source_ids)}, moved {moved}"
                    )
                if edit_mode:
                    from imas_codex.standard_names.source_invariant import (
                        require_producing_sources,
                    )

                    require_producing_sources(query_handle, [new_name])
                record_standard_name_change(
                    query_handle,
                    old_name,
                    new_name,
                    operation="human_edit" if edit_mode else "refine",
                    reason=edit_reason or reason,
                    origin=edit_origin,
                    run_id=run_id,
                )
                if meaning_assertion is not None:
                    assertion_reason, asserted_by = meaning_assertion
                    predecessor_document = str(
                        preflight_row.get("source_documentation") or ""
                    )
                    record_standard_name_change(
                        query_handle,
                        _predecessor_document_identity(old_name, predecessor_document),
                        new_name,
                        operation="semantics_preserving_rename",
                        reason=assertion_reason,
                        origin=asserted_by,
                        run_id=run_id,
                    )
                tx.commit()
            except BaseException:
                with suppress(Exception):
                    tx.rollback()
                raise

    if result:
        row = dict(result[0])
        logger.debug(
            "persist_refined_name: %s → %s (chain_length=%d)",
            old_name,
            new_name,
            new_chain_length,
        )

        # Async counter bump — live progress visibility for ``sn status``
        services.bump_sn_run_counter(run_id, "names_regenerated")
        return row

    raise RefinedNamePersistenceRefusal(
        old_name=old_name,
        proposed_name=new_name,
        reason=RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED,
        existing_name=new_name,
    )


@retry_on_deadlock()
def carry_accepted_documentation_across_rename(
    *,
    predecessor_name: str,
    successor_name: str,
    reason: str,
    asserted_by: str,
    services: Any,
) -> dict[str, Any]:
    """Carry an accepted document after an explicitly equivalent rename.

    This is the repair route for a rename that has already committed. It
    requires the recorded ``REFINED_FROM`` lineage, a superseded predecessor
    with non-empty accepted documentation, and either a pristine pending docs
    axis or an exact prior carry. The assertion is written atomically as an
    internal change whose source value is the predecessor document's
    content-addressed identity.

    The operation is idempotent for the same predecessor document and refuses
    to overwrite any independently authored successor documentation.
    """
    assertion = _meaning_preservation_assertion(
        reason=reason,
        asserted_by=asserted_by,
    )
    if assertion is None:  # pragma: no cover - both arguments are required
        raise ValueError("a meaning-preservation assertion is required")
    assertion_reason, assertion_actor = assertion

    with services.GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                rows = list(
                    tx.run(
                        """
                        MATCH (new:StandardName {id: $successor_name})
                              -[:REFINED_FROM]->
                              (old:StandardName {id: $predecessor_name})
                        RETURN old.name_stage AS predecessor_name_stage,
                               old.docs_stage AS predecessor_docs_stage,
                               old.documentation AS predecessor_documentation,
                               old.docs_chain_length
                                 AS predecessor_docs_chain_length,
                               old.docs_model AS predecessor_docs_model,
                               toString(old.docs_generated_at)
                                 AS predecessor_docs_generated_at,
                               new.docs_stage AS successor_docs_stage,
                               new.documentation AS successor_documentation,
                               new.docs_chain_length
                                 AS successor_docs_chain_length,
                               new.docs_model AS successor_docs_model,
                               toString(new.docs_generated_at)
                                 AS successor_docs_generated_at,
                               new.name_stage AS successor_name_stage,
                               new.status AS successor_status
                        """,
                        predecessor_name=predecessor_name,
                        successor_name=successor_name,
                    )
                )
                if len(rows) != 1:
                    raise ValueError(
                        "the requested identities are not one recorded rename: "
                        f"{predecessor_name!r} -> {successor_name!r}"
                    )

                before = dict(rows[0])
                documentation = str(before.get("predecessor_documentation") or "")
                if before.get("predecessor_name_stage") != "superseded":
                    raise ValueError("the predecessor is not superseded")
                if (
                    before.get("predecessor_docs_stage") != "accepted"
                    or not documentation
                ):
                    raise ValueError(
                        "the predecessor does not hold non-empty accepted documentation"
                    )

                predecessor_chain_length = int(
                    before.get("predecessor_docs_chain_length") or 0
                )
                successor_chain_length = int(
                    before.get("successor_docs_chain_length") or 0
                )
                already_carried = (
                    before.get("successor_docs_stage")
                    == before.get("predecessor_docs_stage")
                    and str(before.get("successor_documentation") or "")
                    == documentation
                    and successor_chain_length == predecessor_chain_length
                    and before.get("successor_docs_model")
                    == before.get("predecessor_docs_model")
                    and before.get("successor_docs_generated_at")
                    == before.get("predecessor_docs_generated_at")
                )
                pristine_pending = (
                    before.get("successor_docs_stage") == "pending"
                    and not str(before.get("successor_documentation") or "")
                    and successor_chain_length == 0
                    and before.get("successor_docs_model") is None
                    and before.get("successor_docs_generated_at") is None
                )
                if not already_carried and not pristine_pending:
                    raise ValueError(
                        "the successor documentation axis is not pristine and "
                        "does not match the predecessor"
                    )

                document_identity = _predecessor_document_identity(
                    predecessor_name,
                    documentation,
                )
                document_sha256 = document_identity.rsplit(":", 1)[-1]
                receipt_id = f"sn-change:{
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        '|'.join(
                            (
                                'semantics_preserving_rename',
                                document_identity,
                                successor_name,
                            )
                        ),
                    )
                }"
                mutation = list(
                    tx.run(
                        """
                        MATCH (new:StandardName {id: $successor_name})
                              -[:REFINED_FROM]->
                              (old:StandardName {id: $predecessor_name})
                        WHERE old.name_stage = 'superseded'
                          AND old.docs_stage = 'accepted'
                          AND old.documentation = $documentation
                          AND coalesce(old.docs_chain_length, 0)
                              = $docs_chain_length
                          AND coalesce(old.docs_model, '')
                              = coalesce($docs_model, '')
                          AND coalesce(toString(old.docs_generated_at), '')
                              = coalesce($docs_generated_at, '')
                          AND (
                            (new.docs_stage = 'pending'
                             AND coalesce(new.documentation, '') = ''
                             AND coalesce(new.docs_chain_length, 0) = 0
                             AND new.docs_model IS NULL
                             AND new.docs_generated_at IS NULL)
                            OR
                            (new.docs_stage = old.docs_stage
                             AND new.documentation = old.documentation
                             AND coalesce(new.docs_chain_length, 0)
                                 = coalesce(old.docs_chain_length, 0)
                             AND coalesce(new.docs_model, '')
                                 = coalesce(old.docs_model, '')
                             AND coalesce(toString(new.docs_generated_at), '')
                                 = coalesce(toString(old.docs_generated_at), ''))
                          )
                        WITH old, new,
                             new.docs_stage = old.docs_stage
                             AND new.documentation = old.documentation
                             AND coalesce(new.docs_chain_length, 0)
                                 = coalesce(old.docs_chain_length, 0)
                             AND coalesce(new.docs_model, '')
                                 = coalesce(old.docs_model, '')
                             AND coalesce(toString(new.docs_generated_at), '')
                                 = coalesce(toString(old.docs_generated_at), '')
                               AS already_carried
                        FOREACH (_ IN CASE WHEN already_carried THEN [] ELSE [1] END |
                          SET new.updated_at = datetime(),
                              new.docs_stage = old.docs_stage,
                              new.documentation = old.documentation,
                              new.docs_chain_length = coalesce(
                                old.docs_chain_length, 0),
                              new.docs_model = old.docs_model,
                              new.docs_generated_at = old.docs_generated_at)
                        MERGE (change:StandardNameChange {id: $receipt_id})
                        ON CREATE SET change.from_name = $document_identity,
                                      change.to_name = $successor_name,
                                      change.operation = $operation,
                                      change.reason = $reason,
                                      change.origin = $asserted_by,
                                      change.changed_at = datetime(),
                                      change.internal = true
                        WITH new, change, already_carried
                        WHERE change.from_name = $document_identity
                          AND change.to_name = $successor_name
                          AND change.operation = $operation
                          AND change.reason = $reason
                          AND change.origin = $asserted_by
                        MERGE (new)-[:HAS_INTERNAL_CHANGE]->(change)
                        RETURN NOT already_carried AS changed,
                               new.name_stage AS name_stage,
                               new.status AS status,
                               new.docs_stage AS docs_stage,
                               size(new.documentation) AS documentation_length,
                               new.docs_model AS docs_model,
                               toString(new.docs_generated_at)
                                 AS docs_generated_at
                        """,
                        predecessor_name=predecessor_name,
                        successor_name=successor_name,
                        documentation=documentation,
                        docs_chain_length=predecessor_chain_length,
                        docs_model=before.get("predecessor_docs_model"),
                        docs_generated_at=before.get("predecessor_docs_generated_at"),
                        receipt_id=receipt_id,
                        document_identity=document_identity,
                        operation="semantics_preserving_rename",
                        reason=assertion_reason,
                        asserted_by=assertion_actor,
                    )
                )
                if len(mutation) != 1:
                    raise RuntimeError(
                        "rename documentation or its assertion receipt changed "
                        "during the transaction"
                    )
                tx.commit()
                result = dict(mutation[0])
                result.update(
                    {
                        "predecessor_name": predecessor_name,
                        "successor_name": successor_name,
                        "predecessor_document_identity": document_identity,
                        "predecessor_document_sha256": document_sha256,
                        "receipt_id": receipt_id,
                        "reason": assertion_reason,
                        "asserted_by": assertion_actor,
                    }
                )
                return result
            except BaseException:
                with suppress(Exception):
                    tx.rollback()
                raise


_GENERATED_SUPERSESSION_PREFLIGHT_QUERY = """
// GENERATED_SUPERSESSION_PREFLIGHT
UNWIND $pairs AS pr
OPTIONAL MATCH (src:IMASNode {id: pr.source_id})
OPTIONAL MATCH (new:StandardName {id: pr.new_name})
// The recomposed source itself, resolved through its DD edge rather than by
// rebuilding the composite id. One DD path can anchor several sources — a
// renamed path keeps a source under its superseded spelling alongside the
// current one — so the successor binding this pass just wrote is what singles
// the recomposed source out of them. Without it every alias reaches the
// predecessor, none of them can be told apart, and the batch aborts.
OPTIONAL MATCH (trigger:StandardNameSource)-[:FROM_DD_PATH]->(src)
WHERE EXISTS { (trigger)-[:PRODUCED_NAME]->(new) }
OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME]->(old:StandardName)
WHERE old.id <> pr.new_name
  AND NOT coalesce(old.name_stage, '') IN
      ['superseded', 'exhausted', 'contested', 'approved']
  AND old.catalog_pr_number IS NULL
  AND coalesce(old.origin, 'pipeline') <> 'derived'
// Skip self and any case where old already descends from new
// along the REFINED_FROM chain (would form a cycle).
  AND new.id <> old.id
  AND NOT (old)-[:REFINED_FROM*1..]->(new)
WITH pr, src, new, trigger, old
OPTIONAL MATCH (source:StandardNameSource)
WHERE old IS NOT NULL
  AND ((source)-[:PRODUCED_NAME]->(old)
       OR source.produced_sn_id = old.id
       OR (source)-[:PRODUCED_NAME]->(new))
// A live successor edge is what a composer emits, so it is the one fact that
// says this pairing was judged. Everything else reaches the predecessor only
// through the predecessor.
WITH pr, src, new, trigger, old, source,
     EXISTS { (source)-[:PRODUCED_NAME]->(new) } AS judged
RETURN pr.source_id AS requested_source_id,
       pr.new_name AS new_name,
       src IS NOT NULL AS requested_source_exists,
       new IS NOT NULL AS successor_exists,
       trigger.id AS trigger_source_id,
       old.id AS old_name,
       old.name_stage AS old_stage,
       [source_id IN collect(DISTINCT
          CASE WHEN judged THEN source.id END)
        WHERE source_id IS NOT NULL] AS judged_source_ids,
       [source_id IN collect(DISTINCT
          CASE WHEN judged THEN null ELSE source.id END)
        WHERE source_id IS NOT NULL] AS retained_source_ids
"""

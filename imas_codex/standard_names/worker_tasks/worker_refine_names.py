"""Refine-name batch processor and its deterministic failure classifiers.

The successor-chain refine processor walks a name's ``REFINED_FROM`` history,
escalates on the last rotation, runs one structured model call and persists the
survivor; the classifiers beside it decide whether a failure is terminal.

Every collaborator that lives on the entrypoint module — the logger, the
hybrid-neighbour and DD-context enrichment helpers, the budget exposure
helpers and the telemetry extractor, and the grammar-failure marker table owned
by the compose-batch module — is reached through the batch processor's
positional-only ``_worker_dependency`` resolver, so a patch on
``workers.<name>`` stays visible inside the moved code.  Only standard-library
modules are imported directly; ``RefineStopReason`` comes from the graph model
module, and ``BudgetManager`` is imported for typing alone.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from imas_codex.graph.models import RefineStopReason

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager


class RefineCandidateValidationError(ValueError):
    """A refined candidate failed deterministic strict grammar validation."""


class RefineScopeContinuityError(RuntimeError):
    """A refine batch cannot prove one coherent successor scope."""


def _refine_batch_scope_id(
    batch: list[dict[str, Any]], scope_run_id: str | None
) -> str | None:
    """Validate and return the invocation's explicit focused scope, if any.

    ``BudgetManager.run_id`` identifies the audit run that pays for and counts
    the work. It is not the graph scope selected by ``--focus``. Only an
    explicit invocation scope activates focused continuity; durable run IDs on
    rows in an ordinary unscoped backlog are historical provenance and do not
    classify the invocation.

    In focused mode, the atomic claim read-back must prove that every
    predecessor carries the exact requested scope. Missing, blank, or
    mismatched claim metadata refuses before any model call or persistence.
    """
    if scope_run_id is None:
        return None
    if (
        not isinstance(scope_run_id, str)
        or not scope_run_id.strip()
        or scope_run_id != scope_run_id.strip()
    ):
        raise RefineScopeContinuityError(
            "refine invocation carries a blank or non-canonical scope identity"
        )

    for item in batch:
        claimed_scope = item.get("scope_run_id")
        if claimed_scope != scope_run_id:
            raise RefineScopeContinuityError(
                f"refine claim for {item.get('id')!r} carries scope "
                f"{claimed_scope!r}, expected {scope_run_id!r}"
            )
    return scope_run_id


def _compose_refined_candidate_name(result_obj: Any) -> str:
    """Compose one refined candidate name and type deterministic failures."""
    from imas_standard_names import ParseError
    from pydantic import ValidationError

    try:
        return result_obj.name
    except (ParseError, ValidationError, ValueError) as exc:
        raise RefineCandidateValidationError(
            f"refined candidate failed strict grammar validation: {exc}"
        ) from exc


def _is_refine_grammar_failure(_worker_dependency, exc: BaseException) -> bool:
    """Return True if *exc* is a deterministic grammar/validation refine failure.

    Such failures (e.g. the LLM proposing an unregistered qualifier token) are
    a normal failed-refine outcome, not a crash: they must be routed to the
    exhaust path so the item stops re-claiming and re-burning paid budget.
    """
    if isinstance(exc, RefineCandidateValidationError):
        return True
    msg = str(exc).lower()
    return any(
        marker in msg for marker in _worker_dependency("_GRAMMAR_FAILURE_MARKERS")
    )


#: Markers within the deterministic set that identify a MISSING TOKEN rather
#: than a malformed composition. The distinction is the operator's next move:
#: a missing token is admitted on the ISN side, while a malformed composition
#: is a prompt or model problem in this repository.
_VOCABULARY_GAP_MARKERS: tuple[str, ...] = ("not a registered",)


def _refine_failure_stop_reason(
    exc: BaseException, *, terminal: bool
) -> RefineStopReason:
    """Classify a refine failure into the reason recorded on the name."""
    if not terminal:
        return RefineStopReason.transient_failure
    msg = str(exc).lower()
    if any(marker in msg for marker in _VOCABULARY_GAP_MARKERS):
        return RefineStopReason.vocabulary_gap
    return RefineStopReason.grammar_invalid


def _is_refine_docs_failure(exc: BaseException) -> bool:
    """Return True if *exc* is a deterministic ``RefinedDocs`` validation failure.

    A docs-refine that produces docs violating the ``RefinedDocs`` schema
    (e.g. consistently over/under the length bounds) is deterministic for a
    given item.  Reverting it to 'reviewed' would re-claim and re-burn paid
    budget forever, so route it to the docs-exhaust path instead.
    """
    return "validation error for refineddocs" in str(exc).lower()


#: Return the rotation an unchanged resubmission did not spend.
#:
#: The refine pool charges a rotation when it CLAIMS a name, before any model
#: call, so the charge buys an improvement attempt. A pinned rename
#: (``edit_mode = 'rename'``) is never rewritten — the pool routes the same
#: name back to a fresh review quorum — so that charge buys nothing. Left
#: alone it walks ``refine_attempts`` to the rotation cap, and the cap is also
#: the eligibility gate, so a name whose only consumer is the re-review loop
#: stops being claimed at 3 of 3. Fenced on the same claim token and
#: ``'refining'`` stage the resubmission checks, so the rotation can only be
#: returned while the claim that charged it is still live.
_RETURN_UNSPENT_REFINE_ATTEMPT = """
MATCH (sn:StandardName {id: $sn_id})
WHERE sn.claim_token = $token AND sn.name_stage = 'refining'
SET sn.refine_attempts = CASE WHEN coalesce(sn.refine_attempts, 0) > 0
                              THEN sn.refine_attempts - 1
                              ELSE sn.refine_attempts END
RETURN coalesce(sn.refine_attempts, 0) AS refine_attempts
"""


def _return_unspent_refine_attempt(
    _worker_dependency, /, gc: Any, *, sn_id: str, token: str
) -> int | None:
    """Give back the rotation an unchanged resubmission did not spend.

    Returns the counter the name is left holding, or ``None`` when the fence
    did not match — a concurrent sweep or another worker already moved the
    row, in which case this pool no longer owns the charge and must not
    rewrite the counter under its own token.
    """
    logger = _worker_dependency("logger")
    try:
        rows = gc.query(
            _RETURN_UNSPENT_REFINE_ATTEMPT,
            sn_id=sn_id,
            token=token,
        )
    except Exception:
        logger.debug(
            "refine_name: could not return the unspent rotation for %s",
            sn_id,
            exc_info=True,
        )
    else:
        try:
            if rows:
                return int(rows[0].get("refine_attempts") or 0)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            # A best-effort return must never cost the item its resubmission.
            logger.debug("refine_name: unreadable rotation counter for %s", sn_id)
    return None


async def process_refine_name_batch(
    _worker_dependency,
    /,
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    scope_run_id: str | None = None,
) -> int:
    """Process a batch of StandardNames through successor-chain refinement.

    For each item in the batch:
    1. Walk the REFINED_FROM chain via ``chain_history`` (already enriched).
    2. Escalate when the claim charged the last rotation the cap allows.
    3. Optionally fan out targeted DD context (gated).
    4. Call LLM to produce a refined name (``RefinedName`` response model).
    5. Run the deterministic name-key dup guard before persisting.
    6. Persist via ``persist_refined_name`` (new node + edge migration).
    7. Close a failed attempt with :func:`stop_refine_name_attempt`, which
       records why it failed and parks the name when the reason is decided or
       the budget is spent.

    Returns count of items successfully processed.
    """
    import asyncio as _asyncio

    logger = _worker_dependency("logger")

    from imas_codex.discovery.base.llm import (
        LLMStructuredCallError,
        ProviderBudgetExhausted,
        acall_llm_structured,
    )
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.canonical import find_name_key_duplicate
    from imas_codex.standard_names.defaults import (
        DEFAULT_ESCALATION_MODEL,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.fanout import (
        CandidateContext,
        FanoutScope,
        assign_arm,
        load_settings as load_fanout_settings,
        run_fanout,
        should_trigger_fanout,
    )
    from imas_codex.standard_names.fanout.dispatcher import proposer_exposure
    from imas_codex.standard_names.graph_ops import (
        RefinedNamePersistenceRefusal,
        RefinedNamePersistenceRefusalReason,
        bump_sn_run_counter,
        persist_refined_name,
        resubmit_pinned_rename_for_review,
        stop_refine_name_attempt,
    )
    from imas_codex.standard_names.models import RefinedName

    rotation_cap = DEFAULT_REFINE_ROTATIONS
    focus_scope_id = _refine_batch_scope_id(batch, scope_run_id)
    processed = 0

    async def _stop_attempt(
        item: dict[str, Any],
        reason: RefineStopReason,
        *,
        detail: str = "",
        collision_name: str | None = None,
    ) -> str:
        """Close this attempt on the graph, returning the stage written."""
        try:
            return await _asyncio.to_thread(
                stop_refine_name_attempt,
                sn_id=item["id"],
                token=item.get("claim_token") or "",
                reason=reason.value,
                detail=detail,
                collision_name=collision_name,
                rotation_cap=rotation_cap,
            )
        except Exception:
            logger.debug(
                "refine_name: could not record stop for %s (reason=%s)",
                item["id"],
                reason.value,
                exc_info=True,
            )
            return ""

    # ── GraphClient lifecycle ────────────────────────────────────────
    # One client per cycle, reused by hybrid-neighbour search,
    # run_fanout, dup guard, and Fanout-node telemetry writes.
    # ``persist_refined_name`` opens its own client (different
    # transaction lifecycle) — that is intentional and unchanged.
    fanout_settings = load_fanout_settings()

    with GraphClient() as gc:
        for item in batch:
            if stop_event.is_set():
                break

            sn_id = item["id"]
            chain_length = item.get("chain_length", 0) or 0
            chain_history = item.get("chain_history", [])

            # ── Pinned-rename guard (never rewrite an operator's name) ──
            # A rename edit carries a fixed, operator-chosen name string.
            # Rewriting it is meaningless and destructive: re-emitting the
            # identical name trips the self-referential-refine guard and
            # decomposing a lexicalised base trips grammar validation — both
            # wrongly exhaust a correct name. Resubmit it to a fresh review
            # quorum instead (bounded), never spend an LLM rewrite on it.
            if (item.get("edit_mode") or "") == "rename":
                token = item.get("claim_token") or ""
                # The claim charged a rotation on the way in. Detect the
                # unchanged case — this step resubmits the name the item
                # already carries: no candidate is produced, so the rotation
                # has bought nothing and must not be spent. Without this a
                # pinned name walks to the cap on re-review alone, and the cap
                # is the eligibility gate, so it stops being claimed.
                attempts_left = _return_unspent_refine_attempt(
                    _worker_dependency, gc, sn_id=sn_id, token=token
                )
                try:
                    outcome = await _asyncio.to_thread(
                        resubmit_pinned_rename_for_review,
                        sn_id=sn_id,
                        token=token,
                        rotation_cap=rotation_cap,
                    )
                except Exception:
                    logger.debug(
                        "refine_name: pinned-rename resubmit failed for %s", sn_id
                    )
                    outcome = ""
                logger.info(
                    "refine_name: pinned rename %s not rewritten — %s "
                    "(rotation returned, %s remaining)",
                    sn_id,
                    outcome or "no-op",
                    attempts_left if attempts_left is not None else "unknown",
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": sn_id,
                            "old_name": sn_id,
                            "outcome": f"pinned_rename_{outcome or 'noop'}",
                            "model": "none",
                            "cost": 0.0,
                            # The rotation this resubmission did not spend.
                            # A caller that sums ``refine_attempts`` from
                            # events must not count it.
                            "refine_attempts": attempts_left,
                        }
                    )
                continue

            # ── Escalation decision ───────────────────────────────────
            # Keyed on the rotations SPENT, which the claim has just charged,
            # so the last one runs on the escalation seat even when nothing
            # this name proposed has ever persisted — the exact case the
            # lineage-depth test never reached.
            attempt = int(item.get("refine_attempts") or chain_length + 1)
            escalate = attempt >= rotation_cap
            logger.info(
                "refine_name: %s attempt %d/%d%s",
                sn_id,
                attempt,
                rotation_cap,
                (
                    f" (previous stop: {item['refine_stop_reason']})"
                    if item.get("refine_stop_reason")
                    else ""
                ),
            )
            if escalate:
                model = DEFAULT_ESCALATION_MODEL
            else:
                # Dedicated refine seat: a flash-lite tier lifted critiqued
                # names at ~5% against ~42% for the compose tier, so refine
                # owns its own model choice rather than sharing the compose
                # seat's.
                model = get_model("sn-refine")

            # ── Build prompt context ──────────────────────────────────
            path = item.get("source_paths", [""])[0] if item.get("source_paths") else ""

            # ── Load scored compose examples for refinement context ────
            from imas_codex.standard_names.example_loader import load_compose_examples

            try:
                _item_domain = item.get("physics_domain") or ""

                def _load_compose_examples(domain: str = _item_domain) -> list:
                    with GraphClient() as _gc:
                        return load_compose_examples(
                            _gc, physics_domains=[domain], axis="name"
                        )

                compose_scored_examples = await _asyncio.to_thread(
                    _load_compose_examples
                )
            except Exception:
                logger.debug(
                    "refine_name: scored-example load failed for %s",
                    sn_id,
                    exc_info=True,
                )
                compose_scored_examples = []

            # ── Parse vocab_gap_detail from JSON string (best-effort) ─────
            raw_vgd = item.get("vocab_gap_detail")
            vocab_gap_detail: dict | None = None
            if raw_vgd:
                try:
                    import json as _json

                    vocab_gap_detail = (
                        _json.loads(raw_vgd) if isinstance(raw_vgd, str) else raw_vgd
                    )
                except (ValueError, TypeError):
                    logger.debug(
                        "refine_name: could not parse vocab_gap_detail for %s", sn_id
                    )

            # validation_issues is stored as a list on the node directly
            validation_issues: list[str] | None = item.get("validation_issues") or None

            # Merge the cached compose context so the system prompt's
            # ``_grammar_reference.md`` include renders the closed-vocabulary
            # token map (``closed_vocab_full`` etc.). Without it the refiner
            # rewrites names with no vocabulary reference — the empty-grammar-
            # block silent gap. ``build_compose_context`` is ``_CONTEXT_CACHE``-
            # backed, so this is a dict lookup after the first call.
            from imas_codex.standard_names.context import build_compose_context

            prompt_context: dict[str, Any] = {
                **build_compose_context(),
                "item": item,
                "chain_history": chain_history,
                "chain_length": chain_length,
                "hybrid_neighbours": [],
                "fanout_evidence": "",
                "compose_scored_examples": compose_scored_examples,
                "vocab_gap_detail": vocab_gap_detail,
                "validation_issues": validation_issues,
            }
            from imas_codex.standard_names.review.cascade import (
                failed_check_instructions,
            )

            failed_checks = failed_check_instructions(item, "name")
            if failed_checks:
                prompt_context["failed_checks"] = failed_checks

            # Attempt hybrid neighbour search (best-effort).  Uses the
            # cycle-scoped ``gc`` — no fresh client.
            try:
                neighbours = _worker_dependency("_hybrid_search_neighbours")(gc, path)
                prompt_context["hybrid_neighbours"] = [
                    {"path": n.get("path", ""), "description": n.get("description")}
                    for n in neighbours
                ]
            except Exception:
                logger.debug("Hybrid neighbour search failed for %s", sn_id)

            # ── DD path context (keywords, clusters, version history) ─
            if path:
                try:
                    _worker_dependency("_enrich_dd_path_context")(gc, item, path)
                except Exception:
                    logger.debug(
                        "refine_name: dd_path_context failed for %s",
                        sn_id,
                        exc_info=True,
                    )

            # ── Fan-out trigger gate ─────────────────────────────────
            # A routed refine fires only when one of its named failures is
            # a retrieval check at or above the calibrated threshold.
            check_trigger: dict[str, Any] = {}
            if failed_checks:
                from imas_codex.standard_names.review.checks import load_review_checks

                check_trigger = {
                    "checks": [
                        check
                        for check in load_review_checks()[0]
                        if check.axis == "name"
                    ],
                    "check_results": {
                        check["id"]: {"answered": True, "failure_probability": 1.0}
                        for check in failed_checks
                    },
                    "check_threshold": fanout_settings.refine_trigger_check_threshold,
                }
            fanout_eligible, reviewer_excerpt = should_trigger_fanout(**check_trigger)

            # ── Optional fan-out ─────────────────────────────────────
            lease = None
            fanout_evidence = ""
            fanout_run_id: str | None = None
            fanout_arm: str | None = None
            if (
                fanout_eligible
                and fanout_settings.enabled
                and fanout_settings.sites.get("refine_name", False)
            ):
                fanout_arm = assign_arm(
                    sn_id,
                    chain_length,
                    arm_percent=fanout_settings.refine_fanout_arm_percent,
                )
                import uuid as _uuid

                fanout_run_id = str(_uuid.uuid4())
                physics_dom = item.get("physics_domain")
                if isinstance(physics_dom, list):
                    physics_dom = physics_dom[0] if physics_dom else None
                ids_filter = item.get("ids_name") or None
                candidate_ctx = CandidateContext(
                    sn_id=sn_id,
                    name=sn_id,
                    path=path,
                    description=item.get("description") or "",
                    physics_domain=physics_dom or "",
                    chain_length=chain_length,
                )
                scope = FanoutScope(
                    physics_domain=physics_dom,
                    ids_filter=ids_filter,
                )
                try:
                    fanout_maximum = proposer_exposure(
                        candidate=candidate_ctx,
                        reviewer_excerpt=reviewer_excerpt,
                        scope=scope,
                        settings=fanout_settings,
                    )
                    lease = mgr.reserve(fanout_maximum, phase="refine_name")
                    if lease is None:
                        continue
                    fanout_evidence = await run_fanout(
                        site="refine_name",
                        candidate=candidate_ctx,
                        reviewer_excerpt=reviewer_excerpt,
                        scope=scope,
                        gc=gc,
                        parent_lease=lease,
                        settings=fanout_settings,
                        arm=fanout_arm,
                        escalate=escalate,
                        fanout_run_id=fanout_run_id,
                    )
                except Exception:
                    logger.exception(
                        "fan-out failed for %s — proceeding with empty evidence",
                        sn_id,
                    )
                    fanout_evidence = ""
                finally:
                    if lease is not None:
                        lease.release_unused()
                        lease = None
            prompt_context["fanout_evidence"] = fanout_evidence

            # Load composition rules for system prompt
            from imas_codex.llm.prompt_loader import load_prompt_config as _lpc

            try:
                _rules_cfg = _lpc("sn_composition_rules")
                prompt_context["composition_rules"] = _rules_cfg.get(
                    "composition_rules", []
                )
            except Exception:
                prompt_context["composition_rules"] = []

            user_prompt = render_prompt("sn/refine_name_user", prompt_context)
            try:
                system_prompt = render_prompt("sn/refine_name_system", prompt_context)
            except Exception:
                logger.debug("refine_name: system prompt render failed for %s", sn_id)
                system_prompt = None

            # ── LLM call ──────────────────────────────────────────────
            _messages = (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if system_prompt
                else [{"role": "user", "content": user_prompt}]
            )
            maximum_exposure = _worker_dependency("model_provider_exposure")(
                model,
                _messages,
                response_model=RefinedName,
                provider_attempts=1,
            )
            lease = mgr.reserve(maximum_exposure, phase="refine_name")
            if lease is None:
                continue
            original_reservation = lease.reserved
            cost = 0.0
            llm_tokens_in = 0
            llm_tokens_out = 0
            llm_tokens_cached_read = 0
            llm_tokens_cached_write = 0
            llm_charge_recorded = False
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=_messages,
                    response_model=RefinedName,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-refine"),
                    before_attempt=_worker_dependency("bind_attempt_exposure")(
                        lease, model, _messages, response_model=RefinedName
                    ),
                )

                # acall_llm_structured returns an LLMResult that still supports
                # legacy tuple unpacking. Normalize telemetry from the real
                # result object and fall back only for older tuple-shaped tests.
                result_obj, cost, _tokens = llm_out
                (
                    llm_tokens_in,
                    llm_tokens_out,
                    llm_tokens_cached_read,
                    llm_tokens_cached_write,
                ) = _worker_dependency("_extract_llm_telemetry")(
                    llm_out,
                    parsed_obj=result_obj,
                    tokens_fallback=_tokens,
                )

                # Charge cost to lease.  Stamp ``batch_id`` with the
                # ``fanout_run_id`` whenever fan-out fired so the
                # ``Fanout`` ↔ ``LLMCost`` join works for both arms.
                if lease:
                    _event = LLMCostEvent(
                        model=model,
                        tokens_in=llm_tokens_in,
                        tokens_out=llm_tokens_out,
                        tokens_cached_read=llm_tokens_cached_read,
                        tokens_cached_write=llm_tokens_cached_write,
                        sn_ids=(sn_id,),
                        phase=(
                            "refine_name+fanout" if fanout_run_id else "refine_name"
                        ),
                        service="standard-names",
                        batch_id=fanout_run_id,
                    )
                    lease.charge_event(cost, _event)
                    llm_charge_recorded = True

                # Strict composition can fail even after the response model
                # validates.  Cost is already recorded against the stable
                # predecessor before this derived accessor is evaluated.
                candidate_name = _compose_refined_candidate_name(result_obj)

                # ── Deterministic name-key dup guard ──────────────
                # Name-key lookup AFTER the final retry's
                # candidate, BEFORE persisting.  On hit we drop the
                # candidate and emit a ``dup_prevented`` log line.
                # Excludes ``old_name`` so the chain's predecessor is
                # never treated as a self-collision.
                dup_id = None
                try:
                    dup_id = find_name_key_duplicate(
                        gc,
                        candidate_name,
                        exclude=sn_id,
                    )
                except Exception:
                    logger.debug(
                        "dup guard failed for %s — proceeding with persist",
                        sn_id,
                    )
                if dup_id:
                    # The identity the refiner is reaching for is already
                    # held. Refinement may not take an occupied identity —
                    # folding into it migrates sources and belongs to
                    # `sn edit` — and the same prompt re-proposes the same
                    # identity on every cycle, across vendors. Park the name
                    # with the collision recorded rather than spending the
                    # rotations that remain on a decided outcome.
                    stage = await _stop_attempt(
                        item,
                        RefineStopReason.successor_collision,
                        detail=f"proposed {candidate_name} which already exists",
                        collision_name=dup_id,
                    )
                    collision_event = {
                        "pool": "refine_name",
                        "name": sn_id,
                        "old_name": sn_id,
                        "proposed_name": candidate_name,
                        "existing_name": dup_id,
                        "duplicate_of": dup_id,
                        "refusal_reason": "successor_collision",
                        "attempt": attempt,
                        "rotation_cap": rotation_cap,
                        "resulting_stage": stage,
                        "outcome": "dup_prevented",
                        "model": model,
                        "cost": cost,
                    }
                    logger.warning(
                        "refine_name_persistence_refused %s",
                        json.dumps(collision_event, sort_keys=True),
                    )
                    if on_event is not None:
                        on_event(collision_event)
                    continue

                # ── Persist ───────────────────────────────────────────
                await _asyncio.to_thread(
                    persist_refined_name,
                    old_name=sn_id,
                    new_name=candidate_name,
                    description=_worker_dependency("normalize_description_text")(
                        result_obj.description
                    ),
                    kind=result_obj.kind,
                    unit=item.get("unit"),
                    physics_domain=(
                        (
                            (item.get("physics_domain") or [None])[0]
                            if isinstance(item.get("physics_domain"), list)
                            else item.get("physics_domain")
                        )
                        or None
                    ),
                    source_domains=(
                        item.get("source_domains")
                        if isinstance(item.get("source_domains"), list)
                        else None
                    ),
                    old_chain_length=chain_length,
                    model=model,
                    reason=result_obj.reason,
                    escalated=escalate,
                    # A focused successor inherits the predecessor's durable
                    # scope inside persist_refined_name's transaction when
                    # run_id is null. The manager id remains reserved for the
                    # SNRun that paid for this invocation. Unscoped runs keep
                    # their established audit-id stamping behavior.
                    run_id=mgr.run_id if focus_scope_id is None else None,
                    expected_claim_token=item.get("claim_token"),
                )
                if focus_scope_id is not None:
                    bump_sn_run_counter(mgr.run_id, "names_regenerated")
                processed += 1
                logger.info(
                    "refine_name: %s → %s (chain_length=%d, model=%s)",
                    sn_id,
                    candidate_name,
                    chain_length + 1,
                    model,
                )

                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": candidate_name,
                            "old_name": sn_id,
                            "new_name": candidate_name,
                            "chain_length": chain_length + 1,
                            "escalated": escalate,
                            "model": model,
                            "cost": cost,
                            "fanout_run_id": fanout_run_id,
                            "fanout_arm": fanout_arm,
                            "original_reservation": original_reservation,
                            "llm_tokens_in": llm_tokens_in,
                            "llm_tokens_out": llm_tokens_out,
                            "llm_tokens_cached_read": llm_tokens_cached_read,
                            "llm_tokens_cached_write": llm_tokens_cached_write,
                        }
                    )

            except Exception as exc:
                if isinstance(exc, LLMStructuredCallError | ProviderBudgetExhausted):
                    cost = exc.cost
                    llm_tokens_in = exc.input_tokens
                    llm_tokens_out = exc.output_tokens
                    llm_tokens_cached_read = exc.cache_read_tokens
                    llm_tokens_cached_write = exc.cache_creation_tokens
                    if not llm_charge_recorded:
                        llm_charge_recorded = _worker_dependency(
                            "charge_billable_exception"
                        )(
                            lease,
                            exc,
                            model=model,
                            sn_ids=(sn_id,),
                            phase=(
                                "refine_name+fanout" if fanout_run_id else "refine_name"
                            ),
                            batch_id=fanout_run_id,
                        )
                if isinstance(exc, ProviderBudgetExhausted):
                    raise
                _exc_str = str(exc)
                if isinstance(exc, RefinedNamePersistenceRefusal):
                    claim_lost = exc.reason in {
                        RefinedNamePersistenceRefusalReason.CLAIM_LOST,
                        RefinedNamePersistenceRefusalReason.PREDECESSOR_STAGE,
                        RefinedNamePersistenceRefusalReason.PREDECESSOR_MISSING,
                    }
                    # A successor the persistence fence will never provision
                    # under that name is as decided as a name-key collision;
                    # a lost claim or a contended drain scope is not, so those
                    # keep the remaining rotations.
                    lifecycle_collision = (
                        exc.reason
                        is RefinedNamePersistenceRefusalReason.SUCCESSOR_LIFECYCLE
                    )
                    stage = await _stop_attempt(
                        item,
                        (
                            RefineStopReason.successor_lifecycle_collision
                            if lifecycle_collision
                            else RefineStopReason.transient_failure
                        ),
                        detail=f"{exc.reason.value}: proposed {exc.proposed_name}",
                        collision_name=exc.existing_name,
                    )
                    refusal_event = {
                        "pool": "refine_name",
                        "name": sn_id,
                        "old_name": sn_id,
                        "proposed_name": exc.proposed_name,
                        "existing_name": exc.existing_name,
                        "refusal_reason": exc.reason.value,
                        "attempt": attempt,
                        "rotation_cap": rotation_cap,
                        "resulting_stage": stage,
                        "outcome": "claim_lost" if claim_lost else "persist_refused",
                        "model": model,
                        "cost": cost,
                    }
                    logger.warning(
                        "refine_name_persistence_refused %s",
                        json.dumps(refusal_event, sort_keys=True),
                    )
                    if on_event is not None:
                        on_event(refusal_event)
                    continue
                # A grammar/validation failure (the LLM proposed an
                # ungrammatical name — e.g. an unregistered qualifier token) or
                # a self-referential refine is a NORMAL failed-refine outcome,
                # not a crash: the model keeps producing the same output for
                # this item.  Mark it exhausted (terminal) instead of reverting
                # to 'reviewed', which would re-claim and re-charge it on a paid
                # model every cycle — an infinite paid loop.  Log terminal
                # failures at WARNING; only genuinely unexpected errors get the
                # ERROR + traceback.
                is_terminal = _is_refine_grammar_failure(_worker_dependency, exc)
                if is_terminal:
                    logger.warning(
                        "refine_name failed (deterministic, marking exhausted) "
                        "for %s: %s",
                        sn_id,
                        _exc_str[:200],
                    )
                else:
                    logger.exception("refine_name failed for %s", sn_id)
                stage = await _stop_attempt(
                    item,
                    _refine_failure_stop_reason(exc, terminal=is_terminal),
                    detail=_exc_str[:500],
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "refine_name",
                            "name": sn_id,
                            "old_name": sn_id,
                            "outcome": "refine_failed",
                            "attempt": attempt,
                            "rotation_cap": rotation_cap,
                            "resulting_stage": stage,
                            "model": model,
                            "cost": cost,
                            "llm_tokens_in": llm_tokens_in,
                            "llm_tokens_out": llm_tokens_out,
                            "llm_tokens_cached_read": llm_tokens_cached_read,
                            "llm_tokens_cached_write": llm_tokens_cached_write,
                        }
                    )
            finally:
                # Always return the unused portion of the lease to the pool.
                # Without this the unspent remainder leaks every iteration
                # and the pool exhausts at ~25 % of cost_limit.
                if lease is not None:
                    try:
                        lease.release_unused()
                    except Exception:
                        pass

    return processed


# How many exact-source bindings a name-review context carries. The
# successor-chain refine bounds that same context, so the limit is owned by the
# refine classifiers and read by the review path through the entrypoint.
_NAME_REVIEW_SOURCE_CONTEXT_LIMIT = 8

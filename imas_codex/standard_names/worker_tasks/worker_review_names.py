"""RD-quorum name review: prompt assembly, cycle chain and persistence.

``process_review_name_batch`` prepares each drafted name's review context, runs
the configured reviewer chain through the calibrated cascade and persists the
resulting cycle rows; ``_run_rd_quorum_cycles`` is the chain itself, deciding
how many seats survive, whether a quorum was reached, and which review is
canonical. The deferral tally beside them records names whose quorum was
incomplete so a run summary can surface the shortfall instead of advancing a
name on one review.

Every collaborator that lives on the entrypoint module — the logger, the
batch-enrichment helper and the source-prefix stripper, the budget exposure
helpers, the DD-gap persistence helpers and the quorum persistence wrapper —
is reached through the batch processor's positional-only ``_worker_dependency``
resolver, so a patch on ``workers.<name>`` stays visible inside the moved code
and the persisting wrapper a caller sees is the entrypoint's own object, which
is what the panel audit recognises structurally. Only standard-library modules
are imported directly; ``BudgetManager`` is imported for typing alone.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from imas_codex.standard_names import vocabulary_definitions as vocabulary_source
from imas_codex.standard_names.defaults import DEFAULT_ESCALATION_MODEL

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager


def _review_source_binding_sort_key(binding: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable order for exact-source review context."""
    return tuple(
        str(binding.get(field) or "")
        for field in (
            "id",
            "source_id",
            "dd_path",
            "dd_version",
            "dd_documentation",
            "dd_parent_path",
            "dd_parent_documentation",
        )
    )


def _enrich_name_review_items(
    _worker_dependency, /, items: list[dict[str, Any]]
) -> None:
    """Attach exact source, canonical DD, and lossless grammar review context.

    Immutable leaf and parent definitions come from each exact
    ``StandardNameSource`` binding projected by the claim.  Rich relational
    context is added through the same batch enrichment used by composition;
    mutable graph enrichment never replaces the pinned source text.
    """
    logger = _worker_dependency("logger")
    strip_dd_prefix = _worker_dependency("strip_dd_prefix")
    _enrich_batch_items = _worker_dependency("_enrich_batch_items")
    _name_review_source_context_limit = _worker_dependency(
        "_NAME_REVIEW_SOURCE_CONTEXT_LIMIT"
    )
    from imas_codex.standard_names.graph_ops import strict_review_grammar_context

    stubs: list[dict[str, Any]] = []
    owners: list[dict[str, Any]] = []

    for item in items:
        try:
            item.update(strict_review_grammar_context(item["id"]))
        except (KeyError, TypeError, ValueError):
            logger.debug(
                "review_name: strict grammar context unavailable for %s",
                item.get("id"),
                exc_info=True,
            )

        raw_bindings = item.get("source_bindings")
        # Claim results always carry this key.  Its absence identifies direct
        # unit-test/legacy callers and avoids an unnecessary graph round-trip.
        if raw_bindings is None:
            continue
        ordered_bindings = sorted(
            (dict(binding) for binding in (raw_bindings or []) if binding),
            key=_review_source_binding_sort_key,
        )
        bindings: list[dict[str, Any]] = []
        seen_binding_ids: set[str] = set()
        for binding in ordered_bindings:
            binding_id = str(
                binding.get("id")
                or binding.get("source_id")
                or binding.get("dd_path")
                or ""
            )
            if binding_id in seen_binding_ids:
                continue
            seen_binding_ids.add(binding_id)
            bindings.append(binding)
        item["source_bindings"] = bindings
        dd_bindings = [
            binding for binding in bindings if binding.get("source_type") == "dd"
        ]
        if not dd_bindings:
            continue

        item["unpinned_source_count"] = sum(
            binding.get("dd_snapshot_pinned") is not True for binding in dd_bindings
        )
        context_bindings = dd_bindings[:_name_review_source_context_limit]
        omitted_count = len(dd_bindings) - len(context_bindings)
        if omitted_count:
            item["source_context_omitted"] = omitted_count

        source_paths = list(
            dict.fromkeys(
                strip_dd_prefix(
                    binding.get("dd_path") or binding.get("source_id") or ""
                )
                for binding in context_bindings
                if binding.get("dd_path") or binding.get("source_id")
            )
        )
        if source_paths:
            item["source_paths"] = source_paths
            item["source_id"] = source_paths[0]

        source_doc_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for binding in context_bindings:
            source_id = strip_dd_prefix(
                binding.get("dd_path") or binding.get("source_id") or ""
            )
            pinned = binding.get("dd_snapshot_pinned") is True
            dd_version = binding.get("dd_version") or ""
            from imas_codex.standard_names.dd_resolutions import resolve_dd_row

            context = resolve_dd_row(
                {
                    "path": source_id,
                    "unit": binding.get("dd_unit") or item.get("unit") or None,
                    "documentation": binding.get("dd_documentation"),
                    "data_type": binding.get("dd_data_type"),
                    "coordinates": binding.get("dd_coordinates") or (),
                    "lifecycle_status": binding.get("dd_lifecycle_status"),
                    "lifecycle_version": binding.get("dd_lifecycle_version"),
                },
                dd_version=dd_version,
            )
            binding.update(context.as_pipeline_item())
            unit = context.unit or ""
            description = (
                binding.get("enhanced_description") or binding.get("description") or ""
            )
            documentation = context.documentation or ""
            group_key = (
                pinned,
                dd_version,
                unit,
                documentation,
                description,
            )
            group = source_doc_groups.setdefault(
                group_key,
                {
                    "id": source_id,
                    "ids": [],
                    "unit": unit,
                    "description": description,
                    "documentation": documentation,
                    "dd_version": dd_version,
                    "snapshot_pinned": pinned,
                },
            )
            if source_id and source_id not in group["ids"]:
                group["ids"].append(source_id)
        item["dd_source_docs"] = list(source_doc_groups.values())

        parent_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for binding in context_bindings:
            parent_path = binding.get("dd_parent_path") or ""
            documentation = binding.get("dd_parent_documentation") or ""
            if not parent_path and not documentation:
                continue
            pinned = binding.get("dd_snapshot_pinned") is True
            dd_version = binding.get("dd_version") or ""
            if parent_path:
                from imas_codex.standard_names.dd_resolutions import resolve_dd_row

                parent_context = resolve_dd_row(
                    {"path": parent_path, "documentation": documentation},
                    dd_version=dd_version,
                )
                documentation = parent_context.documentation or ""
            group_key = (pinned, dd_version, documentation)
            group = parent_groups.setdefault(
                group_key,
                {
                    "path": parent_path,
                    "paths": [],
                    "documentation": documentation,
                    "source_ids": [],
                    "dd_version": dd_version,
                    "snapshot_pinned": pinned,
                },
            )
            if parent_path and parent_path not in group["paths"]:
                group["paths"].append(parent_path)
            source_id = binding.get("source_id") or ""
            if source_id and source_id not in group["source_ids"]:
                group["source_ids"].append(source_id)
        parent_contexts = list(parent_groups.values())
        if parent_contexts:
            item["dd_parent_contexts"] = parent_contexts
        source_hints = [
            {
                "source_id": binding.get("source_id") or "",
                "hint": binding.get("compose_hint") or "",
                "reason": binding.get("compose_hint_reason") or "",
            }
            for binding in context_bindings
            if binding.get("compose_hint")
        ]
        if source_hints:
            item["source_hints"] = source_hints

        if source_paths:
            stubs.append(
                {
                    "path": source_paths[0],
                    "description": item.get("description"),
                    "documentation": item["dd_source_docs"][0]["documentation"],
                    "unit": item["dd_source_docs"][0]["unit"],
                    "physics_domain": item.get("physics_domain"),
                }
            )
            owners.append(item)

    if not stubs:
        return
    try:
        _enrich_batch_items(stubs)
    except Exception:
        logger.debug("review_name: compose-parity DD enrichment failed", exc_info=True)
        return

    protected = {"path", "description", "documentation", "unit", "physics_domain"}
    for item, stub in zip(owners, stubs, strict=True):
        for key, value in stub.items():
            if key == "cross_ids_paths":
                item["semantic_comparators"] = [
                    {"path": path, "basis": "semantic_cluster"} for path in value
                ]
                continue
            if key not in protected and key not in item:
                item[key] = value


# In-process tally of RD-quorum reviews deferred because a ≥2-model quorum did
# NOT complete — a secondary reviewer failed (throttled or empty response), so
# only cycle 0 (or nothing) survived. Keyed by (run_id, review_axis). Accepting
# such a name on the single surviving review would silently advance it on ONE
# review when the profile demanded ≥2 — a critical correctness failure under
# provider throttling. Instead the review is deferred (the claim is released
# back to ``drafted``) and counted here so the run summary can surface the
# shortfall rather than let acceptance quietly degrade to single-model.
_quorum_incomplete_deferrals: dict[tuple[str, str], int] = {}


def _record_quorum_incomplete(run_id: str | None, review_axis: str) -> None:
    """Record one incomplete-quorum deferral for the (run_id, axis) tally."""
    key = (run_id or "", review_axis)
    _quorum_incomplete_deferrals[key] = _quorum_incomplete_deferrals.get(key, 0) + 1


def _unfunded_cause(unfunded: list[tuple[int, str, float]]) -> str:
    """One cause line naming every seat the run budget refused to fund.

    Renders each refused reservation as ``c<idx> <model> $<exposure>`` so an
    operator can see at a glance which seat a run's cost limit or reviewer
    replica count starved, rather than debugging the reviewer chain for a
    provider outage that never happened.
    """
    seats = ", ".join(
        f"c{idx} {model} ${exposure:.4f}" for idx, model, exposure in unfunded
    )
    return f"budget refused {len(unfunded)} seat(s): {seats}"


def quorum_incomplete_snapshot(run_id: str | None) -> dict[str, int]:
    """Return ``{review_axis: deferrals}`` for *run_id* (empty when none)."""
    rid = run_id or ""
    return {
        axis: n for (r, axis), n in _quorum_incomplete_deferrals.items() if r == rid
    }


def reset_quorum_incomplete(run_id: str | None = None) -> None:
    """Clear incomplete-quorum tallies (all runs, or just *run_id*)."""
    if run_id is None:
        _quorum_incomplete_deferrals.clear()
        return
    for key in [k for k in _quorum_incomplete_deferrals if k[0] == (run_id or "")]:
        del _quorum_incomplete_deferrals[key]


async def _run_rd_quorum_cycles(
    _worker_dependency,
    /,
    *,
    sn_id: str,
    review_axis: str,
    response_model: Any,
    user_prompt: str,
    system_prompt: str,
    models: list[str],
    disagreement_threshold: float,
    rubric_dims: tuple[str, ...],
    phase: str,
    acall_llm_structured: Callable[..., Any],
    lease: Any = None,
    budget_manager: Any = None,
    reasoning_effort: str | None = None,
    escalation_reasoning_effort: str | None = None,
    run_id: str | None = None,
    escalation_prompt_factory: Callable[[list[dict[str, Any]]], str] | None = None,
    review_group_id: str | None = None,
) -> dict[str, Any] | None:
    """Run the configured RD-quorum reviewer chain for a single StandardName.

    Calls each configured model in turn (cycle 0 = primary, cycle 1 =
    secondary, cycle 2 = escalator). Cycle 1 runs when ≥2 models are
    configured and the primary survived — a lost primary already puts the
    two-review quorum out of reach, so the secondary would be billed for a
    review the completeness guard must discard. Cycle 2 runs only when (a) ≥3
    models are configured AND (b) any rubric dimension differs between cycles 0
    and 1 by more than *disagreement_threshold* (after normalising 0–20 scores
    to 0–1).

    Persists no graph state directly — the returned ``records`` list is
    written by :func:`_persisting_quorum_cycles`, and the SN-side stage
    transition belongs to the caller.

    Each cycle's cost is charged to *lease* via a fresh
    :class:`~imas_codex.standard_names.budget.LLMCostEvent` tagged with
    the cycle id (``c0``/``c1``/``c2``).

    Returns
    -------
    dict | None
        ``None`` when no cycle produced a parseable result (caller
        should release the claim). Otherwise a dict with keys:

        - ``records`` — list of Review record dicts (one per cycle) ready
          for :func:`~imas_codex.standard_names.graph_ops.write_reviews`.
          Share a single ``review_group_id``.
        - ``winning_score`` — float 0-1, the consensus score that should
          mirror onto the SN axis slot.
        - ``winning_scores`` — dict[str, float] of per-dim consensus.
        - ``winning_comments`` — str.
        - ``winning_comments_per_dim`` — dict | None.
        - ``canonical_model`` — str, the model attribution for the SN
          axis slot (always cycle 0's model — the chain anchor).
        - ``resolution_method`` — one of
          ``single_review`` / ``quorum_consensus`` /
          ``authoritative_escalation`` / ``max_cycles_reached``.
        - ``total_cost`` — float, sum of all cycles.
        - ``total_tokens_in`` / ``total_tokens_out`` — int.
    """
    logger = _worker_dependency("logger")
    model_provider_exposure = _worker_dependency("model_provider_exposure")
    bind_attempt_exposure = _worker_dependency("bind_attempt_exposure")
    charge_billable_exception = _worker_dependency("charge_billable_exception")
    import uuid as _uuid
    from datetime import datetime as _dt

    from imas_codex.standard_names.budget import BudgetExceeded, LLMCostEvent

    review_group_id = review_group_id or str(_uuid.uuid4())
    cycles: list[dict[str, Any]] = []  # parsed cycle results (cycle_index 0..N)
    # Seats the run budget refused to fund, as (cycle index, model, exposure).
    # A refused reservation calls no provider and raises nothing, so without
    # this record an underfunded run is indistinguishable from a provider
    # outage: both surface only as a short quorum.
    unfunded: list[tuple[int, str, float]] = []
    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    async def _run_cycle(
        cycle_idx: int,
        model: str,
        *,
        cycle_user_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """Single LLM call. Returns parsed cycle dict or None on failure."""
        nonlocal total_cost, total_tokens_in, total_tokens_out
        # The escalator cycle (idx >= 2) fires only on a flagged disagreement,
        # so it gets a higher reasoning budget on just the contested items; the
        # base cycles run at the cheaper base effort (review is the cost-
        # dominant phase). Falls back to the base effort when unset.
        cycle_effort = (
            escalation_reasoning_effort
            if (cycle_idx >= 2 and escalation_reasoning_effort is not None)
            else reasoning_effort
        )
        cycle_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cycle_user_prompt or user_prompt},
        ]
        cycle_lease = lease
        owns_lease = False
        if budget_manager is not None:
            maximum_exposure = model_provider_exposure(
                model,
                cycle_messages,
                response_model=response_model,
                provider_attempts=1,
            )
            cycle_lease = budget_manager.reserve(maximum_exposure, phase=phase)
            if cycle_lease is None:
                unfunded.append((cycle_idx, model, maximum_exposure))
                logger.warning(
                    "rd_quorum %s cycle %d unfunded for %s (model=%s): the run "
                    "budget could not reserve $%.4f of pre-launch exposure — "
                    "the reviewer was never called",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                    maximum_exposure,
                )
                return None
            owns_lease = True
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=cycle_messages,
                response_model=response_model,
                service="standard-names",
                reasoning_effort=cycle_effort,
                before_attempt=bind_attempt_exposure(
                    cycle_lease, model, cycle_messages, response_model=response_model
                ),
            )
            result_obj, cost, _tokens = llm_out
        except Exception as exc:
            charge_billable_exception(
                cycle_lease,
                exc,
                model=model,
                sn_ids=(sn_id,),
                phase=phase,
            )
            logger.exception(
                "rd_quorum %s cycle %d failed for %s (model=%s)",
                review_axis,
                cycle_idx,
                sn_id,
                model,
            )
            if owns_lease:
                cycle_lease.release_unused()
            return None

        # Unwrap batch response: prompts ask for {"reviews": [...]}; we pass
        # one item at a time so reviews[0] is the per-item ReviewItem object.
        # A wrapper whose `reviews` is empty/None is a malformed/empty response
        # — treat it as a FAILED cycle (return None) rather than falling through
        # to `.scores` on the bare wrapper. That access raises AttributeError,
        # gets swallowed to score=0.0/tier='poor', and would persist a spurious
        # canonical review that drives a good name into refine/exhausted.
        _missing = object()
        reviews_list = getattr(result_obj, "reviews", _missing)
        if reviews_list is not _missing:
            if not reviews_list:
                logger.warning(
                    "rd_quorum %s cycle %d returned empty/no reviews for %s "
                    "(model=%s) — failed cycle",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                )
                if owns_lease:
                    cycle_lease.release_unused()
                return None
            try:
                result_obj = reviews_list[0]
            except (IndexError, TypeError):
                logger.warning(
                    "rd_quorum %s cycle %d returned unusable reviews for %s "
                    "(model=%s) — failed cycle",
                    review_axis,
                    cycle_idx,
                    sn_id,
                    model,
                )
                if owns_lease:
                    cycle_lease.release_unused()
                return None

        tokens_in = int(getattr(llm_out, "input_tokens", 0) or 0)
        tokens_out = int(getattr(llm_out, "output_tokens", 0) or 0)
        cached_read = int(getattr(llm_out, "cache_read_tokens", 0) or 0)
        cached_write = int(getattr(llm_out, "cache_creation_tokens", 0) or 0)

        total_cost += cost
        total_tokens_in += tokens_in
        total_tokens_out += tokens_out

        if cycle_lease is not None:
            try:
                cycle_lease.charge_event(
                    cost,
                    LLMCostEvent(
                        model=model,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        tokens_cached_read=cached_read,
                        tokens_cached_write=cached_write,
                        sn_ids=(sn_id,),
                        cycle=f"c{cycle_idx}",
                        phase=phase,
                        service="standard-names",
                    ),
                )
            except BudgetExceeded:
                raise
            except Exception:
                logger.debug(
                    "rd_quorum charge_event failed for %s c%d",
                    sn_id,
                    cycle_idx,
                    exc_info=True,
                )
        if owns_lease:
            cycle_lease.release_unused()

        # Extract score and per-dim scores.
        try:
            score = float(result_obj.scores.score)
        except Exception:
            score = 0.0
        try:
            scores_dict = result_obj.scores.model_dump()
        except Exception:
            scores_dict = {}

        comments = getattr(result_obj, "reasoning", None) or ""
        comments_per_dim = None
        try:
            if getattr(result_obj, "comments", None) is not None:
                comments_per_dim = result_obj.comments.model_dump()
        except Exception:
            comments_per_dim = None
        dd_gaps = list(getattr(result_obj, "dd_gaps", None) or [])

        return {
            "cycle_index": cycle_idx,
            "model": model,
            "score": score,
            "scores": scores_dict,
            "comments": comments,
            "comments_per_dim": comments_per_dim,
            "cost": cost,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cached_read": cached_read,
            "cached_write": cached_write,
            "dd_gaps": dd_gaps,
        }

    # ── Cycle 0 (primary, blind) ───────────────────────────────────────
    c0 = await _run_cycle(0, models[0])
    if c0 is not None:
        cycles.append(c0)

    # ── Cycle 1 (secondary, blind) — runs whenever ≥ 2 models ──────────
    # A multi-seat profile demands two surviving base reviews, and the
    # escalator fires only when both base cycles already succeeded. So once
    # the primary is lost the quorum can no longer be reached, and calling the
    # secondary buys a review that the completeness guard below is obliged to
    # discard — paid tokens with no reachable outcome. Stop here instead.
    c1 = None
    if len(models) >= 2 and c0 is not None:
        c1 = await _run_cycle(1, models[1])
        if c1 is not None:
            cycles.append(c1)

    # ── Quorum-completeness guard ──────────────────────────────────────
    # A profile with ≥2 models demands ≥2 successful reviews. If a secondary
    # reviewer failed (throttled / empty response) only cycle 0 — or nothing —
    # survived. A single surviving review must NOT be accepted for a name the
    # profile intended for multi-model adjudication, or throttling would
    # silently advance names on ONE review. Defer instead: return None so the
    # caller releases the claim back to ``drafted`` (same path it already takes
    # for a total failure), and count the deferral so the run summary can
    # surface the shortfall. Cycle 2 is not part of this guard: two base seats
    # that AGREE are a complete quorum, but two that DISAGREE are exactly the
    # case the escalator exists to break — a disagreeing pair whose escalator
    # seat the budget refuses to fund is an INCOMPLETE quorum, and the
    # disagreement branch below defers it rather than publishing a mean.
    intended_model_count = len(models)
    successful_cycles = len(cycles)
    if intended_model_count >= 2 and successful_cycles < 2:
        # Name the cause. A seat the budget refused to fund never reached its
        # provider, so reporting only the shortfall sends an operator hunting a
        # provider outage when the fix is the run's cost limit or its reviewer
        # replica count.
        if unfunded:
            cause = _unfunded_cause(unfunded)
        else:
            cause = "reviewer call failed"
        logger.warning(
            "rd_quorum %s incomplete: %d/%d reviews succeeded for %s — "
            "deferring, NOT accepting on a single review (%s)",
            review_axis,
            successful_cycles,
            intended_model_count,
            sn_id,
            cause,
        )
        _record_quorum_incomplete(run_id, review_axis)
        return None

    if not cycles:
        # Single-model profile whose only cycle failed — nothing to persist.
        return None

    # ── Per-dimension disagreement (cycle 2 gate) ──────────────────────
    disagreement = False
    if c0 is not None and c1 is not None:
        for dim in rubric_dims:
            s0 = float(c0["scores"].get(dim, 0)) / 20.0
            s1 = float(c1["scores"].get(dim, 0)) / 20.0
            if abs(s0 - s1) > disagreement_threshold:
                disagreement = True
                break

    # ── Cycle 2 (escalator) ────────────────────────────────────────────
    c2 = None
    if disagreement and len(models) >= 3:
        prior_reviews = [
            {
                "role": role,
                "model": review["model"],
                "score": review["score"],
                "scores_json": json.dumps(review["scores"], sort_keys=True),
                "reasoning": review["comments"],
                "comments_per_dim_json": json.dumps(
                    review["comments_per_dim"] or {}, sort_keys=True
                ),
            }
            for role, review in (("primary", c0), ("secondary", c1))
            if review is not None
        ]
        if escalation_prompt_factory is not None:
            escalation_user_prompt = escalation_prompt_factory(prior_reviews)
        else:
            critique_text = "\n".join(
                f"- {review['role']} ({review['model']}): "
                f"scores={review['scores_json']}; {review['reasoning']}"
                for review in prior_reviews
            )
            escalation_user_prompt = (
                f"{user_prompt}\n\n## Prior reviewer critiques for authoritative "
                f"escalation\n{critique_text}"
            )
        c2 = await _run_cycle(
            2,
            models[2],
            cycle_user_prompt=escalation_user_prompt,
        )
        if c2 is not None:
            cycles.append(c2)

    # ── Determine winning score + resolution method ────────────────────
    canonical_model = models[0]  # SN axis attribution always cycle-0 model
    if len(cycles) == 1:
        # Reachable only for a single-model profile (len(models) == 1): a
        # ≥2-model profile with fewer than 2 successful cycles is deferred by
        # the quorum-completeness guard above and never reaches here, so a
        # single review is never a silent degradation of a multi-model quorum.
        winning = cycles[0]
        resolution_method = "single_review"
        winning_score = float(winning["score"])
        winning_scores = dict(winning["scores"])
        winning_comments = winning["comments"]
        winning_comments_per_dim = winning["comments_per_dim"]
    elif c2 is not None:
        # Escalator authoritative
        resolution_method = "authoritative_escalation"
        winning_score = float(c2["score"])
        winning_scores = dict(c2["scores"])
        winning_comments = c2["comments"]
        winning_comments_per_dim = c2["comments_per_dim"]
    else:
        # 2 cycles only (or escalator skipped/failed) — mean of c0 + c1
        winning_score = (float(c0["score"]) + float(c1["score"])) / 2.0
        winning_scores = {}
        for dim in rubric_dims:
            v0 = float(c0["scores"].get(dim, 0))
            v1 = float(c1["scores"].get(dim, 0))
            winning_scores[dim] = (v0 + v1) / 2.0
        # Merge comments preserving both reviewers' reasoning
        c0_comments = c0["comments"] or ""
        c1_comments = c1["comments"] or ""
        if c0_comments and c1_comments and c0_comments != c1_comments:
            winning_comments = f"[Primary] {c0_comments}\n[Secondary] {c1_comments}"
        else:
            winning_comments = c1_comments or c0_comments
        winning_comments_per_dim = c1["comments_per_dim"] or c0["comments_per_dim"]
        if disagreement and unfunded:
            # Both base seats succeeded but DISAGREE, and the run budget refused
            # to fund the escalator seat that exists to break the tie. A mean of
            # two disagreeing blind seats is not a verdict, and publishing it
            # under max_cycles_reached would make a funding shortfall
            # indistinguishable from a profile with no tie-breaker configured.
            # Defer exactly as an incomplete quorum does: release the claim,
            # name the refused seats, and count the deferral.
            cause = _unfunded_cause(unfunded)
            logger.warning(
                "rd_quorum %s deferring disputed name %s — tie-breaker seat "
                "could not be funded (%s)",
                review_axis,
                sn_id,
                cause,
            )
            _record_quorum_incomplete(run_id, review_axis)
            return None
        if disagreement:
            # Disputed but no escalator available
            resolution_method = "max_cycles_reached"
        else:
            resolution_method = "quorum_consensus"

    # ── Build Review records ───────────────────────────────────────────
    role_by_idx = {0: "primary", 1: "secondary", 2: "escalator"}
    method_by_idx = {0: None, 1: None, 2: None}
    if resolution_method == "single_review":
        method_by_idx[0] = "single_review"
    elif resolution_method == "authoritative_escalation":
        method_by_idx[2] = "authoritative_escalation"
    elif resolution_method == "max_cycles_reached":
        method_by_idx[1] = "max_cycles_reached"
    elif resolution_method == "quorum_consensus":
        method_by_idx[1] = "quorum_consensus"

    now_iso = _dt.utcnow().isoformat()
    records: list[dict[str, Any]] = []
    for c in cycles:
        idx = c["cycle_index"]
        score = float(c["score"])
        if score >= 0.85:
            tier = "outstanding"
        elif score >= 0.60:
            tier = "good"
        elif score >= 0.40:
            tier = "inadequate"
        else:
            tier = "poor"
        scores_json = json.dumps(c["scores"]) if c["scores"] else "{}"
        comments_per_dim_json = (
            json.dumps(c["comments_per_dim"]) if c["comments_per_dim"] else None
        )
        records.append(
            {
                "id": f"{sn_id}:{review_axis}:{review_group_id}:{idx}",
                "standard_name_id": sn_id,
                "model": c["model"],
                "reviewer_model": c["model"],
                "model_family": "other",
                "is_canonical": idx == 0,
                "score": score,
                "scores_json": scores_json,
                "tier": tier,
                "comments": c["comments"] or "",
                "comments_per_dim_json": comments_per_dim_json,
                "suggested_name": "",
                "suggestion_justification": "",
                "reviewed_at": now_iso,
                "review_axis": review_axis,
                "cycle_index": idx,
                "review_group_id": review_group_id,
                "resolution_role": role_by_idx.get(idx, "primary"),
                "resolution_method": method_by_idx.get(idx),
                "llm_model": c["model"],
                "llm_cost": c["cost"],
                "llm_tokens_in": c["tokens_in"],
                "llm_tokens_out": c["tokens_out"],
                "llm_tokens_cached_read": c["cached_read"],
                "llm_tokens_cached_write": c["cached_write"],
                "llm_at": now_iso,
                "llm_service": "standard-names",
            }
        )

    return {
        "records": records,
        "winning_score": float(winning_score),
        "winning_scores": winning_scores,
        "winning_comments": winning_comments,
        "winning_comments_per_dim": winning_comments_per_dim,
        "canonical_model": canonical_model,
        "resolution_method": resolution_method,
        "total_cost": total_cost,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "review_group_id": review_group_id,
        "disagreement": disagreement,
        "dd_gaps": [gap for cycle in cycles for gap in cycle["dd_gaps"]],
    }


async def _persisting_quorum_cycles(
    _worker_dependency, /, **kwargs: Any
) -> dict[str, Any] | None:
    """Run the quorum cycles and persist their records before returning.

    The calibrated cascade records its shadow route after the incumbent's rows,
    so the incumbent writes its own reviews here instead of handing them back to
    the caller. A cycle chain that produced no reviews returns ``None`` and
    writes nothing.
    """
    run_rd_quorum_cycles = _worker_dependency("_run_rd_quorum_cycles")
    result = await run_rd_quorum_cycles(**kwargs)
    if result is not None:
        from imas_codex.standard_names.graph_ops import write_reviews

        await asyncio.to_thread(write_reviews, result["records"])
    return result


_closed_vocabulary_definitions = vocabulary_source.vocabulary_definitions


def semantic_gate_name_text(name: str) -> str:
    """Return the name-side text the semantic gate scores a description against.

    The gate is a cosine between a name and its description at a floor
    calibrated on multi-word identifiers. A name that is a single vocabulary
    token carries too little text to reach that floor at any description, so
    the coupling rejects the vocabulary's own bare tokens by construction
    rather than on merit. Where the closed vocabulary defines the token, its
    definition is the prose the description should agree with, so score
    against that instead of against the identifier.

    The substitution stays scoped to a name that parses to exactly one token:
    a qualifier carries no definition, and a composed definition of a
    multi-word name scores good names down onto the floor. Every other name —
    multi-token, unparseable, or a token the vocabulary leaves undefined —
    keeps the identifier text, so an unknown token still fails closed.
    """
    try:
        from imas_standard_names.grammar.parser import parse

        ir = parse(name, strict=True).ir
    except Exception:
        return name
    if ir.operators or ir.qualifiers or ir.projection or ir.locus or ir.mechanism:
        return name
    return vocabulary_source.vocabulary_definitions().get(ir.base.token) or name


async def process_review_name_batch(
    _worker_dependency,
    /,
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of drafted StandardNames for name review (RD-quorum).

    For each item in the batch:

    1. Build a name-axis review prompt via ``render_prompt("sn/review_names_user", ...)``.
    2. Run the configured RD-quorum chain
       (``[sn.review.names].models``):

       - **Cycle 0** (primary, blind) — always runs.
       - **Cycle 1** (secondary, blind) — runs when ≥ 2 models configured.
       - **Cycle 2** (escalator) — runs only when per-dimension disagreement
         between cycles 0 and 1 exceeds the configured threshold AND ≥ 3
         models are configured.
    3. Persist each cycle as a separate ``StandardNameReview`` node with
       proper ``cycle_index`` (0/1/2), ``review_group_id`` (UUID shared
       across cycles for the same SN), ``resolution_role``
       (primary/secondary/escalator) and ``resolution_method``
       (single_review / quorum_consensus / authoritative_escalation /
       max_cycles_reached).
    4. Persist via :func:`~imas_codex.standard_names.graph_ops.persist_reviewed_name`
       with ``skip_review_node=True`` (review nodes already written) and the
       *winning* score so the SN axis slots mirror the consensus and the
       state machine transitions ``name_stage`` to ``'accepted'``,
       ``'reviewed'`` or ``'exhausted'``.
    5. Update review aggregates so ``review_count`` /
       ``review_disagreement`` reflect the multi-cycle group.
    6. On LLM error: release the claim via
       :func:`~imas_codex.standard_names.graph_ops.release_review_names_failed_claims`.

    Returns count of items successfully reviewed.
    """
    logger = _worker_dependency("logger")
    # Callers substitute this enricher to reach the review paths without a
    # graph, so it is resolved through the entrypoint rather than called here.
    enrich_name_review_items = _worker_dependency("_enrich_name_review_items")
    _persist_dd_gap_evidence = _worker_dependency("_persist_dd_gap_evidence")
    _review_dd_source_bindings = _worker_dependency("_review_dd_source_bindings")
    _persisting_quorum_cycles = _worker_dependency("_persisting_quorum_cycles")

    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_sn_review_disagreement_threshold,
        get_sn_review_escalation_reasoning_effort,
        get_sn_review_names_models,
        get_sn_review_reasoning_effort,
    )
    from imas_codex.standard_names.defaults import (
        DEFAULT_MIN_SCORE,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_ops import (
        persist_reviewed_name,
        release_review_names_failed_claims,
        update_review_aggregates,
    )
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnlyBatch,
    )

    # ── Resolve review model chain ─────────────────────────────────────
    try:
        review_models = get_sn_review_names_models()
    except (ValueError, IndexError):
        review_models = []
    if not review_models:
        review_models = [DEFAULT_ESCALATION_MODEL]

    try:
        disagreement_threshold = get_sn_review_disagreement_threshold()
    except Exception:
        disagreement_threshold = 0.20

    await _asyncio.to_thread(enrich_name_review_items, batch)

    processed = 0

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""

        # ── Skip review for quarantined names ──────────────────────────
        # Names that failed ISN 3-layer validation already have a low-tier
        # verdict — running RD-quorum on them is pure waste. Persist a
        # zero-score review (cycle 0 only, single_review) directly.
        if item.get("validation_status") == "quarantined":
            try:
                persist_reviewed_name(
                    sn_id=sn_id,
                    claim_token=claim_token,
                    score=0.0,
                    scores={
                        "grammar": 0.0,
                        "semantic": 0.0,
                        "convention": 0.0,
                        "completeness": 0.0,
                    },
                    comments="quarantined: skipped reviewer LLM call",
                    comments_per_dim=None,
                    model="(skipped: quarantined)",
                    llm_cost=0.0,
                    llm_tokens_in=0,
                    llm_tokens_out=0,
                    llm_tokens_cached_read=0,
                    llm_tokens_cached_write=0,
                    llm_service="standard-names",
                    run_id=mgr.run_id,
                )
                processed += 1
                if on_event:
                    on_event(
                        {
                            "type": "review_name_skipped_quarantined",
                            "sn_id": sn_id,
                        }
                    )
            except Exception:
                logger.debug(
                    "review_name: persist failed for quarantined %s",
                    sn_id,
                    exc_info=True,
                )
            continue

        # ── Semantic similarity gate (pre-LLM) ────────────────────────
        # Compute cosine similarity between name-as-text and description.
        # Names below the critical threshold are semantically ambiguous
        # (e.g. "co_passing_density" — density of what?).  Skip the
        # expensive LLM review and persist a synthetic low score as a
        # non-winning diagnostic verdict: the gate's resolution method
        # records a review_quorum_shortfall, and a refine claim requires
        # that shortfall to be null — so the row is unclaimable by the
        # refine_name pool.
        #
        # The name-side text comes from ``semantic_gate_name_text``: a name
        # that is a single defined vocabulary token is scored against the
        # vocabulary's definition of that token, because one token cannot
        # reach a floor calibrated on multi-word identifiers. Every other
        # name keeps the identifier text.
        #
        # EXCEPT for deterministic parents: their description is the
        # canonical placeholder until ``GENERATE_DOCS`` runs, so the
        # cosine similarity is meaningless. Cosine sim between a
        # registered base token (``magnetic_field``) and a generic
        # placeholder string is always low — the gate would fire on
        # every deterministic parent that ever reached this point.
        # Deterministic parents auto-accept on the name axis anyway
        # (``seed_parent_sources`` writes ``name_stage='accepted'``);
        # this skip is defensive in case any drift puts one at
        # ``drafted``.
        from imas_codex.standard_names.audits import semantic_similarity_check
        from imas_codex.standard_names.defaults import (
            SEMANTIC_SIM_CRITICAL,
            SEMANTIC_SIM_GATE_MODEL,  # noqa: F811
            SEMANTIC_SIM_GATE_RESOLUTION_METHOD,  # noqa: F811
            SEMANTIC_SIM_SYNTHETIC_SCORE,
            SEMANTIC_SIM_WARNING,
        )

        sem_sim: float | None = None
        sem_issues: list[str] = []
        is_derived = item.get("origin") == "derived"
        if is_derived:
            # ── Desc-name similarity gate for derived parents ─────────────
            # For derived parents, the description may have been seeded from
            # a placeholder or a DD-derived string that doesn't align with
            # the name.  Compute desc_name_similarity and, if below
            # threshold, route to REFINE_DOCS instead of name scoring.
            # This keeps description-quality failures off the name axis.
            #
            # Unlike the general semantic_similarity_check (which uses the
            # quarantine-oriented SEMANTIC_SIM_CRITICAL threshold and routes
            # to REFINE_NAME), this gate uses the dedicated
            # desc_name_similarity_threshold and routes to REFINE_DOCS.
            from imas_codex.standard_names.desc_name_sim import (
                compute_desc_name_similarity,
                should_route_to_refine_docs,
            )

            desc_sim: float | None = None
            try:
                desc_sim = await _asyncio.to_thread(
                    compute_desc_name_similarity,
                    sn_id,
                    item.get("description") or "",
                )
            except Exception:
                logger.debug(
                    "review_name: desc_name_similarity failed for derived %s",
                    sn_id,
                    exc_info=True,
                )

            if desc_sim is not None and should_route_to_refine_docs(desc_sim):
                logger.info(
                    "review_name: desc-name sim gate FIRED for derived %s "
                    "(sim=%.3f) — routing to REFINE_DOCS",
                    sn_id,
                    desc_sim,
                )
                from imas_codex.standard_names.graph_ops import mark_for_refine_docs

                try:
                    mark_for_refine_docs(
                        sn_id,
                        desc_name_similarity=desc_sim,
                        claim_token=claim_token,
                    )
                    processed += 1
                    if on_event:
                        on_event(
                            {
                                "type": "review_name_desc_sim_gate_routed_docs",
                                "sn_id": sn_id,
                                "desc_name_similarity": desc_sim,
                            }
                        )
                except Exception:
                    logger.debug(
                        "review_name: mark_for_refine_docs failed for %s",
                        sn_id,
                        exc_info=True,
                    )
                continue

            # Below threshold not triggered (sim is high enough, or embed
            # failed — treat as "gate not applicable"). Log and fall through
            # to normal name scoring.
            if desc_sim is not None:
                logger.debug(
                    "review_name: derived %s desc-name sim=%.3f (gate clear)",
                    sn_id,
                    desc_sim,
                )
        else:
            try:
                sem_sim, sem_issues = await _asyncio.to_thread(
                    semantic_similarity_check,
                    semantic_gate_name_text(sn_id),
                    item.get("description") or "",
                )
            except Exception:
                logger.debug(
                    "review_name: semantic_similarity_check failed for %s",
                    sn_id,
                    exc_info=True,
                )

        # Persist semantic_sim on the node regardless of outcome
        if sem_sim is not None:
            try:
                from imas_codex.graph.client import GraphClient as _GC

                def _store_sim(_id=sn_id, _sim=sem_sim) -> None:
                    with _GC() as gc:
                        gc.query(
                            "MATCH (sn:StandardName {id: $id}) "
                            "SET sn.semantic_sim = $sim, sn.updated_at = datetime()",
                            id=_id,
                            sim=_sim,
                        )

                await _asyncio.to_thread(_store_sim)
            except Exception:
                pass  # best-effort

        # Critical: skip LLM review and record a non-winning diagnostic verdict.
        # The resulting quorum shortfall prevents both acceptance and a refine
        # claim until the lifecycle owns a deliberate route for this evidence.
        if sem_sim is not None and sem_sim < SEMANTIC_SIM_CRITICAL:
            logger.info(
                "review_name: semantic gate FAILED for %s (sim=%.3f < %.2f) — "
                "recording non-winning verdict",
                sn_id,
                sem_sim,
                SEMANTIC_SIM_CRITICAL,
            )
            try:
                persist_reviewed_name(
                    sn_id=sn_id,
                    claim_token=claim_token,
                    score=SEMANTIC_SIM_SYNTHETIC_SCORE,
                    scores={"semantic": float(sem_sim)},
                    comments=(
                        f"semantic_similarity_gate: sim={sem_sim:.3f} below "
                        f"critical {SEMANTIC_SIM_CRITICAL:.2f}. Name is "
                        f"semantically ambiguous — a reader cannot determine "
                        f"the measured quantity from the name alone."
                    ),
                    comments_per_dim={
                        "semantic": (
                            f"Name-to-description embedding similarity is "
                            f"{sem_sim:.3f}, well below the {SEMANTIC_SIM_CRITICAL:.2f} "
                            f"threshold. The name does not stand alone."
                        ),
                    },
                    model=SEMANTIC_SIM_GATE_MODEL,
                    llm_cost=0.0,
                    llm_tokens_in=0,
                    llm_tokens_out=0,
                    llm_tokens_cached_read=0,
                    llm_tokens_cached_write=0,
                    llm_service="standard-names",
                    run_id=mgr.run_id,
                    resolution_method=SEMANTIC_SIM_GATE_RESOLUTION_METHOD,
                )
                processed += 1
                if on_event:
                    on_event(
                        {
                            "type": "review_name_semantic_gate_failed",
                            "sn_id": sn_id,
                            "semantic_sim": sem_sim,
                        }
                    )
            except Exception:
                logger.debug(
                    "review_name: persist failed for semantic-gated %s",
                    sn_id,
                    exc_info=True,
                )
            continue

        # Inject warning into item context for the reviewer
        if sem_sim is not None and sem_sim < SEMANTIC_SIM_WARNING:
            sem_warning_note = (
                f"⚠️ SEMANTIC AMBIGUITY WARNING: Name-to-description "
                f"embedding similarity is {sem_sim:.3f} (threshold "
                f"{SEMANTIC_SIM_WARNING:.2f}). Verify that the name is "
                f"self-describing — can someone determine the measured "
                f"quantity from the name alone?"
            )
            item["semantic_warning"] = sem_warning_note

        # ── Build prompt context ───────────────────────────────────────
        from imas_codex.standard_names.context import (
            _build_enum_lists,
            fetch_review_neighbours,
        )

        try:
            neighbours = fetch_review_neighbours(item)
        except Exception:
            logger.debug(
                "review_name: neighbour fetch failed for %s", sn_id, exc_info=True
            )
            neighbours = {
                "vector_neighbours": [],
                "same_base_neighbours": [],
                "same_path_neighbours": [],
            }

        # ── Load scored examples for reviewer calibration ──────────────
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.example_loader import load_review_examples

        try:
            _item_domain = item.get("physics_domain") or ""

            def _load_review_examples(domain: str = _item_domain) -> list:
                with GraphClient() as _gc:
                    return load_review_examples(
                        _gc, physics_domains=[domain], axis="name"
                    )

            review_scored_examples = await _asyncio.to_thread(_load_review_examples)
        except Exception:
            logger.debug(
                "review_name: scored-example load failed for %s", sn_id, exc_info=True
            )
            review_scored_examples = []

        # The reviewer must see the SAME closed token registry the composer
        # sees (closed_vocab_full + operators_full): a valid name built on a
        # less-common but registered base (etendue, opacity, …) otherwise reads
        # as an unknown base and is scored down, so it never accepts. The
        # registry is static and lands in the cached system-prompt prefix.
        from imas_codex.standard_names.context import build_compose_context

        prompt_context: dict[str, Any] = {
            **build_compose_context(),
            "items": [item],
            **neighbours,
            **_build_enum_lists(),
            "review_scored_examples": review_scored_examples,
            "prior_reviews": [],
        }
        try:
            user_prompt = render_prompt("sn/review_names_user", prompt_context)
            system_prompt = render_prompt("sn/review_names_system", prompt_context)
        except Exception:
            logger.debug("review_name: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Review the standard name: {item.get('id', sn_id)}\n"
                f"Description: {item.get('description', '')}"
            )
            system_prompt = (
                "You are a quality reviewer for IMAS standard names in "
                "fusion plasma physics."
            )

        def _render_escalation_prompt(
            prior_reviews: list[dict[str, Any]],
            base_context: dict[str, Any] = prompt_context,
        ) -> str:
            return render_prompt(
                "sn/review_names_user",
                {**base_context, "prior_reviews": prior_reviews},
            )

        # ── RD-quorum cycles ──────────────────────────────────────────
        lease = None
        try:
            from functools import partial

            from imas_codex.standard_names.review.cascade import adjudicate_review

            quorum = await adjudicate_review(
                "name",
                item,
                review_models,
                partial(
                    _persisting_quorum_cycles,
                    sn_id=sn_id,
                    review_axis="name",
                    response_model=StandardNameQualityReviewNameOnlyBatch,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    disagreement_threshold=disagreement_threshold,
                    rubric_dims=("grammar", "semantic", "convention", "completeness"),
                    lease=lease,
                    budget_manager=mgr,
                    phase="review_name",
                    acall_llm_structured=acall_llm_structured,
                    reasoning_effort=get_sn_review_reasoning_effort(),
                    escalation_reasoning_effort=get_sn_review_escalation_reasoning_effort(),
                    run_id=mgr.run_id,
                    escalation_prompt_factory=_render_escalation_prompt,
                ),
                mgr=mgr,
            )

            if quorum is None:
                # No cycles produced a valid result — release claim.
                if claim_token:
                    try:
                        await _asyncio.to_thread(
                            release_review_names_failed_claims,
                            sn_ids=[sn_id],
                            claim_token=claim_token,
                            from_stage="drafted",
                            to_stage="drafted",
                        )
                    except Exception:
                        logger.debug(
                            "release_review_names_failed_claims also failed for %s",
                            sn_id,
                        )
                continue

            # ── Stage transition with winning score ────────────────────
            new_stage = await _asyncio.to_thread(
                persist_reviewed_name,
                sn_id=sn_id,
                claim_token=claim_token,
                claim_seq=item.get("claim_seq"),
                score=quorum["winning_score"],
                scores=quorum["winning_scores"],
                comments=quorum["winning_comments"],
                comments_per_dim=quorum["winning_comments_per_dim"],
                model=quorum["canonical_model"],
                min_score=DEFAULT_MIN_SCORE,
                rotation_cap=DEFAULT_REFINE_ROTATIONS,
                llm_cost=quorum["total_cost"],
                llm_tokens_in=quorum["total_tokens_in"],
                llm_tokens_out=quorum["total_tokens_out"],
                llm_tokens_cached_read=0,
                llm_tokens_cached_write=0,
                llm_service="standard-names",
                run_id=mgr.run_id,
                skip_review_node=True,
                resolution_method=quorum["resolution_method"],
                reviewer_chain_size=len(review_models),
            )
            if new_stage:
                source_versions = _review_dd_source_bindings(item)
                await _asyncio.to_thread(
                    _persist_dd_gap_evidence,
                    quorum["dd_gaps"],
                    set(source_versions),
                    phase="review_name",
                    reporter="review-name",
                    source_versions=source_versions,
                )

            # ── Update aggregates so review_count / disagreement reflect group ─
            try:
                await _asyncio.to_thread(update_review_aggregates, [sn_id])
            except Exception:
                logger.debug(
                    "update_review_aggregates failed for %s", sn_id, exc_info=True
                )

            if new_stage:
                processed += 1
                _comment_log = (quorum["winning_comments"] or "")[:80]
                logger.info(
                    "review_name: %s → %s (score=%.3f, cycles=%d, method=%s) %s",
                    sn_id,
                    new_stage,
                    quorum["winning_score"],
                    len(quorum["records"]),
                    quorum["resolution_method"],
                    _comment_log,
                )
                if on_event is not None:
                    on_event(
                        {
                            "pool": "review_name",
                            "name": sn_id,
                            "score": quorum["winning_score"],
                            "comment": quorum["winning_comments"] or "",
                            "stage": new_stage,
                            "model": quorum["canonical_model"],
                            "cost": quorum["total_cost"],
                            "cycles": len(quorum["records"]),
                            "resolution_method": quorum["resolution_method"],
                        }
                    )
            else:
                logger.debug("review_name: %s persist no-op (token mismatch?)", sn_id)

        except Exception:
            logger.exception("review_name failed for %s", sn_id)
            token = item.get("claim_token") or ""
            if token:
                try:
                    await _asyncio.to_thread(
                        release_review_names_failed_claims,
                        sn_ids=[sn_id],
                        claim_token=token,
                        from_stage="drafted",
                        to_stage="drafted",
                    )
                except Exception:
                    logger.debug(
                        "release_review_names_failed_claims also failed for %s",
                        sn_id,
                    )
        finally:
            # Always return the unused portion of the lease to the pool.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


# Query text for the generate_docs enrichment readers. They live here because
# the review and docs reviewers share one prompt-context shape; the readers
# resolve them from this module.

_DD_PATH_CONTEXT_QUERY = """
MATCH (n:IMASNode {id: $path})
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
WHERE c.scope IN ['global', 'domain']
WITH n, collect(DISTINCT {
    label: c.label, description: c.description, scope: c.scope
})[0..3] AS clusters
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (vc:IMASNodeChange)-[:FOR_IMAS_PATH]->(n)
WHERE vc.change_type IN [
    'path_added', 'cocos_transformation_type', 'sign_convention',
    'units', 'path_renamed', 'definition_clarification'
]
WITH n, clusters, parent,
     collect(DISTINCT {change_id: vc.id, change_type: vc.change_type}) AS changes
RETURN n.keywords AS keywords,
       n.documentation AS dd_documentation,
       n.description AS dd_description,
       n.unit AS dd_units,
       parent.description AS parent_description,
       clusters,
       changes
"""

_DOCS_GEN_ENRICH_QUERY = """
MATCH (sn:StandardName {id: $sn_id})
OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(imas:IMASNode)
RETURN sn.source_paths AS source_paths,
       sn.cocos_transformation_type AS cocos_label,
       sn.origin AS origin,
       collect(DISTINCT {
           id: imas.id,
           documentation: coalesce(imas.documentation, ''),
           description: coalesce(imas.description, ''),
           alias: imas.name,
           unit: coalesce(imas.unit, '')
       }) AS dd_nodes
"""

_DOCS_GEN_COCOS_PARAMS_QUERY = """
MATCH (dv:DDVersion {id: $ver})-[:HAS_COCOS]->(c:COCOS)
RETURN properties(c) AS cocos_params
"""

# Children grounding for a derived-parent docs item. A derived parent is an
# abstraction over these concrete instances; generate_docs must GENERALISE over
# them (describe the shared quantity), not over-specialise to any one child.
_DOCS_GEN_PARENT_CHILDREN_QUERY = """
MATCH (c:StandardName)-[:HAS_PARENT]->(p:StandardName {id: $sn_id})
WHERE NOT coalesce(c.name_stage, '') IN ['superseded', 'exhausted', 'contested']
RETURN c.id AS name,
       c.description AS description,
       c.unit AS unit,
       c.physics_domain AS physics_domain
ORDER BY c.id LIMIT 12
"""

_DOCS_GEN_NEARBY_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage IN ['accepted', 'approved']
  AND sn.physics_domain = $domain
  AND sn.id <> $sn_id
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS id,
       sn.description AS description,
       sn.kind AS kind,
       coalesce(u.id, sn.unit) AS unit
LIMIT 20
"""

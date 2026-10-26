"""Documentation worker tasks and their explicit entrypoint collaborators."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager


async def process_enrich_parents_batch(
    _worker_dependency,
    /,
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Synthesize real descriptions for placeholder derived parents.

    A derived parent materialised by ``_materialize_derived_parent_rows`` keeps
    the deterministic placeholder description until something writes a real one.
    That placeholder deadlocks it: REVIEW_NAME drops it (no real description),
    so it never earns a ``reviewer_score_name``, so generate_docs drops it too.
    This worker breaks the deadlock — for each claimed parent:

    1. Fetch the parent's live children (the concrete instances it abstracts
       over) for grounding.
    2. Render the children-grounded enrichment prompt and call
       ``get_model("sn-parent-enrich")`` (compose-tier; generate_docs rewrites
       the full documentation downstream).
    3. Charge the budget lease.
    4. Embed the synthesised description locally (free) so REVIEW_NAME's
       semantic-similarity check has an embedding immediately.
    5. Persist via :func:`persist_enriched_parent` (writes description +
       embedding, routes ``name_stage`` to ``'drafted'``).

    Returns the count of parents successfully enriched.
    """
    bind_attempt_exposure = _worker_dependency("bind_attempt_exposure")
    charge_billable_exception = _worker_dependency("charge_billable_exception")
    model_provider_exposure = _worker_dependency("model_provider_exposure")
    normalize_description_text = _worker_dependency("normalize_description_text")
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.embeddings.description import embed_description
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.graph_ops import (
        fetch_derived_parent_children,
        persist_enriched_parent,
        release_enrich_parents_claims,
    )
    from imas_codex.standard_names.models import EnrichedParentDescription

    model = get_model("sn-parent-enrich")
    processed = 0

    # ── Fetch grounding children for the whole batch (one round-trip) ─────
    try:
        children_by_parent = await _asyncio.to_thread(
            fetch_derived_parent_children, [item["id"] for item in batch]
        )
    except Exception:
        logger.debug(
            "process_enrich_parents_batch: children fetch failed", exc_info=True
        )
        children_by_parent = {}

    # Static grammar/context block (cached after first render) so the system
    # prompt's grammar include renders the closed-vocabulary map.
    _base_ctx = build_compose_context()

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        # The claim returns id as the StandardName id (== the name string).
        item.setdefault("name", sn_id)
        children = children_by_parent.get(sn_id, [])

        # A parent with no live children cannot be grounded — skip (the claim
        # gate already requires ≥1 child, so this is defensive). Release it.
        if not children:
            logger.debug("enrich_parents: %s has no live children — releasing", sn_id)
            try:
                await _asyncio.to_thread(
                    release_enrich_parents_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                pass
            continue

        prompt_context: dict[str, Any] = {
            **_base_ctx,
            "item": item,
            "children": children,
        }
        try:
            user_prompt = render_prompt("sn/enrich_parent_user", prompt_context)
        except Exception:
            logger.debug("enrich_parents: user prompt render failed for %s", sn_id)
            _child_lines = "\n".join(
                f"- {c.get('name')}: {c.get('description') or ''}" for c in children
            )
            user_prompt = (
                f"Describe the derived parent standard name '{sn_id}' "
                f"(unit: {item.get('unit') or '—'}, kind: {item.get('kind') or 'scalar'}) "
                f"by generalising over its children:\n{_child_lines}"
            )
        try:
            system_prompt = render_prompt("sn/enrich_parent_system", prompt_context)
        except Exception:
            logger.debug("enrich_parents: system prompt render failed for %s", sn_id)
            system_prompt = None

        _messages = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if system_prompt
            else [{"role": "user", "content": user_prompt}]
        )
        maximum_exposure = model_provider_exposure(
            model,
            _messages,
            response_model=EnrichedParentDescription,
            provider_attempts=1,
        )
        lease = mgr.reserve(maximum_exposure, phase="enrich_parents")
        if lease is None:
            continue

        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=EnrichedParentDescription,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-parent-enrich"),
                before_attempt=bind_attempt_exposure(
                    lease,
                    model,
                    _messages,
                    response_model=EnrichedParentDescription,
                ),
            )
            result_obj, cost, _tokens = llm_out

            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=(getattr(llm_out, "cache_read_tokens", 0) or 0),
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=(sn_id,),
                    phase="enrich_parents",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            description = normalize_description_text(
                (result_obj.description or "").strip()
            )

            # Embed locally (free) so REVIEW_NAME has an embedding immediately.
            embedding = await _asyncio.to_thread(embed_description, description)

            await _asyncio.to_thread(
                persist_enriched_parent,
                sn_id=sn_id,
                claim_token=claim_token,
                description=description,
                embedding=embedding,
                model=model,
                run_id=mgr.run_id,
            )
            processed += 1

            if on_event is not None:
                on_event(
                    {
                        "pool": "enrich_parents",
                        "name": sn_id,
                        "description": description,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception as _enrich_exc:
            charge_billable_exception(
                lease,
                _enrich_exc,
                model=model,
                sn_ids=(sn_id,),
                phase="enrich_parents",
            )
            if "token mismatch or node not found" in str(_enrich_exc):
                logger.warning(
                    "enrich_parents claim lost for %s (reclaimed mid-enrich) "
                    "— will retry on a later pass",
                    sn_id,
                )
            else:
                logger.exception("enrich_parents failed for %s", sn_id)
            try:
                await _asyncio.to_thread(
                    release_enrich_parents_claims,
                    sn_ids=[sn_id],
                    claim_token=claim_token,
                )
            except Exception:
                logger.debug("release_enrich_parents_claims also failed for %s", sn_id)
        finally:
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


_DOCS_REVIEW_VOLATILE_ITEM_FIELDS = frozenset(
    {
        "claim_token",
        "claim_seq",
        "claimed_at",
        "docs_stage",
        "run_id",
        "reviewer_score_docs",
        "reviewer_scores_docs",
        "reviewer_comments_docs",
        "reviewer_comments_per_dim_docs",
        "reviewer_model_docs",
        "docs_review_quorum_shortfall",
        "docs_review_quorum_shortfall_at",
        "reviewed_docs_at",
    }
)

"""Documentation worker tasks and their explicit entrypoint collaborators."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager


async def process_refine_docs_batch(
    _worker_dependency,
    /,
    batch: list[dict[str, Any]],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process a batch of reviewed StandardNames for docs refinement.

    For each item in the batch:

    1. Read ``docs_chain_length``, decide model (escalation on final attempt).
    2. Render prompt with current docs, reviewer feedback, and chain history.
    3. Call LLM to produce refined docs (``RefinedDocs`` response model).
    4. Persist via ``persist_refined_docs`` — snapshots old docs into
       ``DocsRevision``, updates SN in-place, advances chain.
    5. On failure: release claims via ``release_refine_docs_failed_claims``.

    Returns count of items successfully processed.
    """
    _documentation_with_registry_form = _worker_dependency(
        "_documentation_with_registry_form"
    )
    _is_refine_docs_failure = _worker_dependency("_is_refine_docs_failure")
    bind_attempt_exposure = _worker_dependency("bind_attempt_exposure")
    charge_billable_exception = _worker_dependency("charge_billable_exception")
    model_provider_exposure = _worker_dependency("model_provider_exposure")
    normalize_description_text = _worker_dependency("normalize_description_text")
    import asyncio as _asyncio

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import conventions_context_for
    from imas_codex.standard_names.defaults import (
        DEFAULT_ESCALATION_MODEL,
        DEFAULT_REFINE_ROTATIONS,
    )
    from imas_codex.standard_names.graph_store.graph_claim_release import (
        _mark_refine_docs_exhausted,
    )
    from imas_codex.standard_names.graph_store.graph_docs_persistence import (
        persist_refined_docs,
        release_refine_docs_failed_claims,
    )
    from imas_codex.standard_names.models import RefinedDocs

    rotation_cap = DEFAULT_REFINE_ROTATIONS
    processed = 0

    for item in batch:
        if stop_event.is_set():
            break

        sn_id = item["id"]
        claim_token = item.get("claim_token") or ""
        docs_chain_length = item.get("docs_chain_length", 0) or 0
        docs_chain_history = item.get("docs_chain_history", [])

        # ── Escalation decision ───────────────────────────────────
        escalate = docs_chain_length >= rotation_cap - 1
        if escalate:
            model = DEFAULT_ESCALATION_MODEL
        else:
            # Dedicated refine seat — see the refine_name comment for why
            # refine owns its own model choice rather than sharing compose's.
            model = get_model("sn-refine")

        # ── Build prompt context ──────────────────────────────────
        prompt_context: dict[str, Any] = {
            "sn_name": sn_id,
            "description": item.get("description", ""),
            "documentation": item.get("documentation", ""),
            "kind": item.get("kind", "scalar"),
            "unit": item.get("unit", ""),
            "physics_domain": item.get("physics_domain", ""),
            "docs_chain_length": docs_chain_length,
            "docs_chain_history": docs_chain_history,
            "reviewer_score_docs": item.get("reviewer_score_docs"),
            "reviewer_comments_per_dim_docs": item.get(
                "reviewer_comments_per_dim_docs"
            ),
            "dd_paths": [],
            # Expert edit steering (imas-codex sn edit) — the refine_docs_user
            # template reads these as top-level context vars, not item.*,
            # because this prompt_context is a hand-built whitelist rather
            # than an `item` merge.
            "docs_hint": item.get("docs_hint"),
            "edit_reason": item.get("edit_reason"),
            "edit_origin": item.get("edit_origin"),
        }
        from imas_codex.standard_names.review.cascade import failed_check_instructions

        failed_checks = failed_check_instructions(item, "docs")
        if failed_checks:
            prompt_context["failed_checks"] = failed_checks
        prompt_context["conventions"] = conventions_context_for(sn_id)

        # Best-effort DD path enrichment
        from imas_codex.standard_names.dd_resolutions import DDResolutionError

        try:
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                dd_rows = gc.query(
                    """
                    MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName {id: $sn_id})
                    WHERE n:IMASNode
                    RETURN n.id AS path, n.ids AS ids,
                           n.unit AS unit,
                           n.documentation AS documentation,
                           n.description AS description
                    LIMIT 5
                    """,
                    sn_id=sn_id,
                )
                from imas_codex.settings import get_dd_version
                from imas_codex.standard_names.dd_resolutions import resolve_dd_rows

                raw_rows = [dict(row) for row in dd_rows]
                batch = resolve_dd_rows(raw_rows, dd_version=get_dd_version())
                if batch.refusals:
                    logger.warning(
                        "Refine context omitted %d changed DD fields: %s",
                        len(batch.refusals),
                        ", ".join(
                            f"{refusal.path}:{refusal.field.value}"
                            for refusal in batch.refusals
                        ),
                    )
                prompt_context["dd_paths"] = [
                    {
                        **raw_rows[resolved.row_index],
                        **resolved.context.as_pipeline_item(),
                    }
                    for resolved in batch.resolved
                ]
        except DDResolutionError:
            raise
        except Exception:
            logger.debug("refine_docs: DD path enrichment failed for %s", sn_id)

        # Parent-aware refine: inject children (origin='derived' only) so the
        # rewrite GENERALISES over them rather than re-hugging one child — the
        # parent-aware reviewer dock that routes single-child parents here is
        # for over-specialisation, which only children-grounding can fix.
        try:
            from imas_codex.graph.client import GraphClient as _GCRefKids
            from imas_codex.standard_names.defaults import (
                DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER as _PH_REF,
            )

            with _GCRefKids() as gc:
                krows = gc.query(
                    """
                    MATCH (p:StandardName {id: $sn_id}) WHERE p.origin = 'derived'
                    MATCH (c:StandardName)-[:HAS_PARENT]->(p)
                    WHERE NOT coalesce(c.name_stage, '') IN ['superseded', 'exhausted', 'contested']
                    WITH c ORDER BY c.id
                    RETURN c.id AS name,
                           CASE WHEN c.description = $ph THEN null
                                ELSE c.description END AS description,
                           c.unit AS unit, c.physics_domain AS physics_domain
                    LIMIT 12
                    """,
                    sn_id=sn_id,
                    ph=_PH_REF,
                )
                kids = [dict(r) for r in krows]
                if kids:
                    prompt_context["derived_children"] = kids
        except Exception:
            logger.debug("refine_docs: derived children fetch failed for %s", sn_id)

        # Sibling-family context so the rewrite converges on the family's
        # parallel documentation template rather than drifting further.
        try:
            from imas_codex.standard_names.context import fetch_sibling_family

            family = fetch_sibling_family(sn_id)
            if family and family.get("siblings"):
                prompt_context["sibling_family"] = family
        except Exception:
            logger.debug("refine_docs: sibling family fetch failed for %s", sn_id)

        # Locus-defining cross-link context, matching generated documentation.
        try:
            from imas_codex.standard_names.context import locus_context_for

            prompt_context["locus_context"] = locus_context_for(sn_id)
        except Exception:
            logger.debug("refine_docs: locus context fetch failed for %s", sn_id)

        try:
            user_prompt = render_prompt("sn/refine_docs_user", prompt_context)
        except Exception:
            logger.exception("refine_docs: prompt render failed for %s", sn_id)
            user_prompt = (
                f"Refine the documentation for standard name: {sn_id}\n"
                f"Current description: {item.get('description', '')}\n"
                f"Current documentation: {item.get('documentation', '')}\n"
                f"Reviewer feedback: {item.get('reviewer_comments_docs', '')}"
            )

        try:
            system_prompt = render_prompt("sn/refine_docs_system", prompt_context)
        except Exception:
            logger.debug("refine_docs: system prompt render failed for %s", sn_id)
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
            response_model=RefinedDocs,
            provider_attempts=1,
        )
        lease = mgr.reserve(maximum_exposure, phase="refine_docs")
        if lease is None:
            continue

        # ── LLM call ──────────────────────────────────────────────
        try:
            llm_out = await acall_llm_structured(
                model=model,
                messages=_messages,
                response_model=RefinedDocs,
                service="standard-names",
                reasoning_effort=get_reasoning_effort("sn-refine"),
                before_attempt=bind_attempt_exposure(
                    lease, model, _messages, response_model=RefinedDocs
                ),
            )

            result_obj, cost, _tokens = llm_out

            # Charge cost to lease
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
                    phase="refine_docs",
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # ── Persist ───────────────────────────────────────────
            await _asyncio.to_thread(
                persist_refined_docs,
                sn_id=sn_id,
                claim_token=claim_token,
                description=normalize_description_text(result_obj.description),
                documentation=_documentation_with_registry_form(
                    sn_id, result_obj.documentation, prompt_context["conventions"]
                ),
                model=model,
                current_description=item.get("description") or "",
                current_documentation=item.get("documentation") or "",
                current_model=item.get("docs_model"),
                current_generated_at=(
                    str(item["docs_generated_at"])
                    if item.get("docs_generated_at")
                    else None
                ),
                reviewer_score_to_snapshot=item.get("reviewer_score_docs"),
                reviewer_comments_to_snapshot=item.get("reviewer_comments_docs"),
                reviewer_comments_per_dim_to_snapshot=item.get(
                    "reviewer_comments_per_dim_docs"
                ),
                run_id=mgr.run_id,
            )
            processed += 1

            desc_preview = result_obj.description[:80]
            logger.info(
                "\033[35mrefine_docs\033[0m: %s — %s (chain=%d, model=%s)",
                sn_id,
                desc_preview,
                docs_chain_length + 1,
                model,
            )

            if on_event is not None:
                on_event(
                    {
                        "pool": "refine_docs",
                        "name": sn_id,
                        "description": result_obj.description,
                        "revision": docs_chain_length + 1,
                        "model": model,
                        "cost": cost,
                    }
                )

        except Exception as exc:
            charge_billable_exception(
                lease,
                exc,
                model=model,
                sn_ids=(sn_id,),
                phase="refine_docs",
            )
            # A deterministic ``RefinedDocs`` validation failure (docs that
            # consistently violate the schema length bounds) is a NORMAL
            # failed-refine outcome, not a crash — the model keeps producing
            # the same invalid docs for this item.  Mark it exhausted instead
            # of reverting to 'reviewed', which would re-claim and re-charge it
            # on a paid model every cycle (an infinite paid loop).  Log such
            # failures at WARNING; only unexpected errors get ERROR + traceback.
            is_terminal = _is_refine_docs_failure(exc)
            if is_terminal:
                logger.warning(
                    "refine_docs failed (deterministic, marking exhausted) for %s: %s",
                    sn_id,
                    str(exc)[:200],
                )
            else:
                logger.exception("refine_docs failed for %s", sn_id)
            try:
                if is_terminal:
                    await _asyncio.to_thread(
                        _mark_refine_docs_exhausted,
                        sn_id=sn_id,
                        token=claim_token,
                        error_msg=str(exc)[:500],
                    )
                else:
                    await _asyncio.to_thread(
                        release_refine_docs_failed_claims,
                        sn_ids=[sn_id],
                        claim_token=claim_token,
                    )
            except Exception:
                logger.debug(
                    "release/exhaust refine_docs also failed for %s",
                    sn_id,
                )
            if on_event is not None:
                on_event(
                    {
                        "pool": "refine_docs",
                        "name": sn_id,
                        "outcome": "refine_failed",
                        "model": model,
                        "cost": 0.0,
                    }
                )
        finally:
            # Always return the unused portion of the lease to the pool.
            # Without this the unspent remainder leaks every iteration and
            # the pool exhausts at ~25 % of cost_limit.
            if lease is not None:
                try:
                    lease.release_unused()
                except Exception:
                    pass

    return processed


async def process_embed_batch(
    _worker_dependency,
    /,
    items: list[dict[str, Any]],
    mgr: Any,
    stop_event: asyncio.Event,
    *,
    on_event: Callable | None = None,
) -> int:
    """Batch-embed StandardName nodes.

    No LLM cost — uses embedding server only.  Does not charge budget.
    """
    from imas_codex.embeddings.description import embed_descriptions_batch
    from imas_codex.standard_names.graph_store.graph_refinement import (
        _compute_embed_hash,
        persist_embed_batch,
    )

    if stop_event.is_set():
        return 0

    # Build embed texts
    for item in items:
        desc = item.get("description")
        sn_id = item["id"]
        item["_embed_text"] = f"{sn_id} — {desc}" if desc else sn_id
        item["embed_text_hash"] = _compute_embed_hash(sn_id, desc)

    # Batch embed (no LLM cost — uses embedding server)
    await asyncio.to_thread(embed_descriptions_batch, items, text_field="_embed_text")

    # Filter to those that got embeddings
    to_persist = [
        {
            "id": it["id"],
            "embedding": it["embedding"],
            "embed_text_hash": it["embed_text_hash"],
        }
        for it in items
        if it.get("embedding") is not None
    ]

    if to_persist:
        written = await asyncio.to_thread(persist_embed_batch, to_persist)
    else:
        written = 0

    # Emit events for display
    if on_event:
        for it in to_persist:
            on_event(
                {
                    "pool": "embed_name",
                    "name": it["id"],
                }
            )

    return written

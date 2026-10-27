"""Pool-mode compose batch processors (generate-name).

Every collaborator that lives on the entrypoint module — including the budget
and provenance helpers — is reached through the two functions' positional-only
``_worker_dependency`` resolver, so a patch on ``workers.<name>`` stays visible
inside the moved code.  Only standard-library modules are imported directly;
``BudgetManager`` is imported for typing alone.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager


async def compose_batch(
    _worker_dependency,
    /,
    batch: list[dict],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    regen: bool = False,
    compose_model: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Shared implementation for compose and regen pool batch processors.

    Takes a list of pre-claimed items (dicts with ``path``, ``description``,
    ``physics_domain``, etc.) and runs the compose pipeline:
    prompt → LLM → grammar validate → persist.

    **Batch-scope domain context:**
    Domain vocabulary is derived from the *batch items* rather than a
    run-scoped ``state.domain_filter``, so pooled-mode batches get domain
    context even without ``--physics-domain``.

    **Soft drain on stop_event:**
    Checks ``stop_event.is_set()`` before the LLM call and returns early
    (after persisting any in-flight results) if the event fires.

    Args:
        batch: Pre-claimed source items from the graph claim query.
        mgr: Shared :class:`BudgetManager` for cost tracking.
        stop_event: Cooperative shutdown signal.
        regen: If True, set ``regen_increment`` on persisted candidates.
        compose_model: Optional configured model override for this run. When
            absent, the production ``sn-compose`` seat is resolved normally.

    Returns:
        Count of successfully composed/regenerated candidates.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import (
        get_compose_self_refine,
        get_model,
        get_reasoning_effort,
    )
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameComposeBatch

    if not batch:
        return 0

    # Claims are already verified by claim_generate_name_batch. Re-verify here
    # as well so direct processor callers cannot bypass fencing before a paid
    # compose call.
    from imas_codex.standard_names.graph_store.graph_claim_release import (
        _verify_source_claim_winners,
    )

    batch = await asyncio.to_thread(_verify_source_claim_winners, batch)
    if not batch:
        return 0

    # Claim read-back carries name-side edit steering as scalar fields. The
    # compose template renders those fields through review_feedback, so nest
    # them here for pooled generation. Exact-source compose_hint stays separate
    # because its lifecycle and authority boundary belong to the source node.
    for item in batch:
        if item.get("name_hint") and item.get("edit_reason"):
            item["review_feedback"] = {
                "previous_name": item.get("previous_name"),
                "reviewer_score": None,
                **(item.get("review_feedback") or {}),
                "name_hint": item["name_hint"],
                "edit_reason": item["edit_reason"],
                "edit_origin": item.get("edit_origin"),
            }

    model = compose_model or get_model("sn-compose")
    context = build_compose_context()

    # ── Batch-scope domain context ─────────────────────────────────────
    def _scalar_domain(d: object) -> str | None:
        """Normalise physics_domain to a scalar string (handles list or str)."""
        if isinstance(d, list):
            return d[0] if d else None
        return d  # type: ignore[return-value]

    domains_in_batch = sorted(
        {
            _scalar_domain(item.get("physics_domain"))
            for item in batch
            if item.get("physics_domain")
        }
        - {None}
    )

    from imas_codex.standard_names.context import build_domain_vocabulary_preseed

    domain_vocab_parts: list[str] = []
    for dom in domains_in_batch:
        vocab = await asyncio.to_thread(build_domain_vocabulary_preseed, dom)
        if vocab:
            domain_vocab_parts.append(vocab)
    context["domain_vocabulary"] = "\n".join(domain_vocab_parts)

    # ── Reviewer-theme extraction (batch-scoped) ──────────────────────
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    for dom in domains_in_batch:
        themes = await asyncio.to_thread(extract_reviewer_themes, dom)
        reviewer_themes.extend(themes)
    context["reviewer_themes"] = reviewer_themes[:12]

    # ── Scored-example injection ───────────────────────────────────────
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_compose_examples

    def _load_scored_examples() -> list[dict]:
        with GraphClient() as gc:
            return load_compose_examples(
                gc, physics_domains=domains_in_batch, axis="name"
            )

    compose_scored_examples = await asyncio.to_thread(_load_scored_examples)
    context["compose_scored_examples"] = compose_scored_examples

    # ── System prompt (cached per pool lifetime) ───────────────────────
    system_prompt = render_prompt("sn/generate_name_system", context)

    # ── Enrich items with DD context ───────────────────────────────────
    def _enrich():
        _worker_dependency("_enrich_batch_items")(batch)

    await asyncio.to_thread(_enrich)

    # ── Search nearby existing names (per-item) ──────────────────────────
    # For each batch item, search using the item's description (or path
    # leaf) to get semantically relevant nearby names. Deduplicate across
    # items, give each item a share, and cap at 30 total.
    group_key = batch[0].get("path", "").split("/")[0] if batch else ""
    nearby = await asyncio.to_thread(
        _worker_dependency("_collect_nearby_name_comparators"), batch
    )
    nearby, _rerank_cost = await _worker_dependency("_reranked_nearby_comparators")(
        mgr, batch, nearby, phase="regen" if regen else "generate_name"
    )

    # ── IDS context ────────────────────────────────────────────────────
    ids_names = sorted(
        {
            item["path"].split("/")[0]
            for item in batch
            if item.get("path") and "/" in item["path"]
        }
    )
    ids_contexts: list[dict] = []
    for iname in ids_names:
        info = _worker_dependency("_enrich_ids_context")(iname)
        if info:
            ids_contexts.append({"ids_name": iname, **info})

    # ── Pre-LLM stop check ────────────────────────────────────────────
    if stop_event.is_set():
        return 0

    # ── Reference exemplars — superseded by graph-backed
    # ``compose_scored_examples`` loaded above. The active prompt
    # template ``sn/generate_name_dd`` no longer renders a
    # ``reference_exemplars`` block, so
    # the per-batch ANN search has been retired here. The linear path
    # still uses ``_search_reference_exemplars`` for the names-only
    # template ``sn/generate_name_dd_names``.

    # ── Cluster-aware existing-names roster ───────────────────────────
    # Build name-only roster from any existing_name field carried on batch
    # items plus a graph query for SNs in the same IMASSemanticCluster.
    existing_names_set: set[str] = {
        item.get("existing_name") for item in batch if item.get("existing_name")
    }
    existing_names_set.discard(None)
    try:
        from imas_codex.graph.client import GraphClient

        path_ids = [item.get("path") for item in batch if item.get("path")]
        if path_ids:

            def _cluster_roster() -> list[str]:
                with GraphClient() as gc:
                    rows = gc.query(
                        """
                        UNWIND $paths AS p
                        MATCH (n:IMASNode {id: p})-[:IN_SEMANTIC_CLUSTER]->(cl)
                        MATCH (other:IMASNode)-[:IN_SEMANTIC_CLUSTER]->(cl)
                        MATCH (other)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                        WHERE coalesce(sn.validation_status, '') <> 'quarantined'
                          AND coalesce(sn.name_stage, '') <> 'superseded'
                        RETURN DISTINCT sn.id AS id
                        LIMIT 200
                        """,
                        paths=path_ids,
                    )
                    return [r["id"] for r in (rows or []) if r.get("id")]

            cluster_names = await asyncio.to_thread(_cluster_roster)
            existing_names_set.update(cluster_names)
    except Exception:
        logger.debug("Cluster-aware existing_names lookup failed", exc_info=True)

    # ── Accepted family siblings into the prompt ───────────────────────
    # A family member that belongs to a detected vector/geometric family
    # already carries its accepted siblings (see
    # attach_family_accepted_siblings / _enrich_batch_items). Fold those
    # spellings into each item's EXISTING neighbour block — the template
    # renders hybrid_neighbours as reusable ``name:`` entries — and into the
    # existing-name roster, so the composer follows the family's established
    # spelling instead of inventing one. No parallel prompt section is added.
    for item in batch:
        sibs = item.get("family_accepted_siblings")
        if not sibs or not item.get("path"):
            continue
        for s in sibs:
            existing_names_set.add(s["name"])
        neighbours = item.setdefault("hybrid_neighbours", [])
        seen_tags = {n.get("tag") for n in neighbours}
        for s in sibs:
            tag = f"name:{s['name']}"
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            neighbours.append(
                {
                    "tag": tag,
                    "path": s.get("path"),
                    "ids": "",
                    "unit": s.get("unit", ""),
                    "physics_domain": s.get("physics_domain", ""),
                    "doc_short": (
                        f"{s.get('description') or ''} (family axis: "
                        f"{s.get('axis') or ''})"
                    ).strip(),
                    "cocos_label": "",
                    "lifecycle": "accepted",
                    "node_type": "",
                    "score": None,
                    "_family_sibling": True,
                }
            )

    existing_names = sorted(existing_names_set)[:200]

    # ── Render user prompt ─────────────────────────────────────────────
    user_context = {
        "items": batch,
        "ids_name": group_key,
        "ids_contexts": ids_contexts,
        "existing_names": existing_names,
        "cluster_context": "",
        "nearby_existing_names": nearby,
        "cocos_version": batch[0].get("cocos_version") if batch else None,
        "dd_version": batch[0].get("dd_version") if batch else None,
    }
    user_prompt = render_prompt("sn/generate_name_dd", {**context, **user_context})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # ── Budget reservation ─────────────────────────────────────────────
    # Seed the lease with the first compose attempt; each further attempt and
    # each per-candidate grammar retry binds its own exposure before it runs.
    _max_retries = _worker_dependency("_retry_attempts")()
    # Token-reuse hits from the FINAL attempt (carried to the VocabGap stamp).
    _token_reuse_hits: dict[tuple[str, str], Any] = {}
    _editorial_guidance = ""
    _editorial_should_retry = False
    maximum_exposure = _worker_dependency("model_provider_exposure")(
        model,
        messages,
        response_model=StandardNameComposeBatch,
        provider_attempts=1,
    )
    phase_tag = "regen" if regen else "generate_name"
    lease = mgr.reserve(maximum_exposure, phase=phase_tag)
    if lease is None:
        # Signal the refusal rather than returning a count, matching the
        # review path. Returning would leave the batch's sources claimed with
        # nothing to release them until the claim ages out, so every replica
        # would spin on empty claims while the pool showed pending work — a
        # stall, not a budget verdict. Raising routes through the pool's
        # release hook, which frees the sources for the next attempt, and each
        # refused reservation advances the saturation counter so the run exits
        # promptly on the reason that actually stopped it.
        from imas_codex.standard_names.budget import BudgetExceeded

        raise BudgetExceeded(
            f"{phase_tag} batch of {len(batch)} has no reservable provider "
            f"exposure (${maximum_exposure:.4f} required)"
        )

    try:
        # ── Bounded retry loop for failed compositions ─────────────────
        # Mirror of the linear path's retry: on grammar parse
        # failure across any candidate, re-enrich items with expanded DD
        # context and re-prompt.  Bounded by _retry_attempts().
        _total_compose_cost = 0.0
        _total_tokens_in = 0
        _total_tokens_out = 0
        _total_cache_read = 0
        _total_cache_creation = 0
        _collected_dd_gaps: list[Any] = []

        for _compose_attempt in range(_max_retries + 1):
            try:
                llm_out = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    service="standard-names",
                    reasoning_effort=get_reasoning_effort("sn-compose"),
                    before_attempt=_worker_dependency("bind_attempt_exposure")(
                        lease,
                        model,
                        messages,
                        response_model=StandardNameComposeBatch,
                    ),
                )
            except (ValueError, Exception) as compose_exc:
                _worker_dependency("charge_billable_exception")(
                    lease,
                    compose_exc,
                    model=model,
                    batch_id=group_key,
                    phase=phase_tag,
                )
                _exc_str = str(compose_exc)
                if "not a registered" in _exc_str:
                    logger.warning(
                        "Pool %s: vocab gap validation error — marking "
                        "%d sources as vocab_gap: %s",
                        phase_tag,
                        len(batch),
                        _exc_str[:200],
                    )
                    _worker_dependency("_mark_vocab_gap_sources")(batch, _exc_str, "dd")
                    return 0
                raise
            result, cost, tokens = llm_out
            _worker_dependency("_sanitize_compose_result_sources")(
                result,
                _worker_dependency("_claimed_source_ids")(batch),
                phase=phase_tag,
            )
            _collected_dd_gaps.extend(result.dd_gaps)

            # Accumulate token/cost totals across retries
            _total_compose_cost += cost
            _attempt_tokens_in = getattr(llm_out, "input_tokens", 0) or 0
            _attempt_tokens_out = getattr(llm_out, "output_tokens", 0) or 0
            _attempt_cache_r = getattr(llm_out, "cache_read_tokens", 0) or 0
            _attempt_cache_w = getattr(llm_out, "cache_creation_tokens", 0) or 0
            _total_tokens_in += _attempt_tokens_in
            _total_tokens_out += _attempt_tokens_out
            _total_cache_read += _attempt_cache_r
            _total_cache_creation += _attempt_cache_w

            # Charge this attempt's cost immediately
            if lease:

                def _safe_sn_ids(_r=result) -> tuple[str, ...]:
                    ids: list[str] = []
                    for c in _r.candidates if _r else []:
                        try:
                            ids.append(c.compose_name())
                        except Exception:
                            ids.append(f"<compose-error:{c.segments.base_token}>")
                    return tuple(ids)

                _event = LLMCostEvent(
                    model=model,
                    tokens_in=_attempt_tokens_in,
                    tokens_out=_attempt_tokens_out,
                    tokens_cached_read=_attempt_cache_r,
                    tokens_cached_write=_attempt_cache_w,
                    sn_ids=_safe_sn_ids(),
                    batch_id=group_key,
                    phase=phase_tag,
                    service="standard-names",
                )
                lease.charge_event(cost, _event)

            # Quick grammar round-trip check on all candidates, with the
            # per-failure repair advice the retry needs to converge.
            _grammar_failures, _grammar_advice = _worker_dependency(
                "_grammar_round_trip_failures"
            )(result.candidates)

            # Component-token reuse check (free local embeddings): flag any
            # candidate-new token semantically near a registered same-segment
            # token so the agent gets a chained chance to reuse-or-confirm.
            _token_reuse_hits = await asyncio.to_thread(
                _worker_dependency("_compute_token_reuse_hits"), result.vocab_gaps
            )
            _editorial_guidance, _editorial_should_retry = await asyncio.to_thread(
                _worker_dependency("_active_editorial_gap_guidance"), result.vocab_gaps
            )

            if (
                not _grammar_failures
                and not _token_reuse_hits
                and not _editorial_should_retry
            ) or (_compose_attempt >= _max_retries):
                break

            # Re-enrich items with expanded hybrid search for retry
            logger.info(
                "Pool %s: composition retry %d/%d: %d grammar failures (%s), "
                "%d token-reuse hits, editorial_retry=%s — "
                "re-composing with expanded DD context",
                phase_tag,
                _compose_attempt + 1,
                _max_retries,
                len(_grammar_failures),
                ", ".join(_grammar_failures[:3]),
                len(_token_reuse_hits),
                _editorial_should_retry,
            )

            def _re_enrich_expanded():
                from imas_codex.graph.client import GraphClient

                with GraphClient() as gc:
                    batch_tuples = [
                        (
                            item.get("path"),
                            item.get("description"),
                            item.get("physics_domain"),
                        )
                        for item in batch
                        if item.get("path")
                    ]
                    if batch_tuples:
                        hybrid_results = _worker_dependency(
                            "_hybrid_search_neighbours_batch"
                        )(
                            gc,
                            batch_tuples,
                            search_k=_worker_dependency("_retry_k_expansion")(),
                        )
                        item_idx = 0
                        for item in batch:
                            if not item.get("path"):
                                continue
                            if hybrid_results[item_idx]:
                                item["hybrid_neighbours"] = hybrid_results[item_idx]
                            item_idx += 1

            # Only re-enrich when grammar failed; a token-reuse-only retry needs
            # no expanded neighbour context (the agent just reuses-or-confirms).
            if _grammar_failures:
                await asyncio.to_thread(_re_enrich_expanded)

            _reason_parts = []
            if _grammar_failures:
                _reason_parts.append(
                    _worker_dependency("_build_grammar_retry_reason")(
                        _grammar_failures, _grammar_advice
                    )
                )
            if _token_reuse_hits:
                _reason_parts.append(
                    _worker_dependency("_build_token_reuse_retry_reason")(
                        _token_reuse_hits
                    )
                )
            if _editorial_guidance:
                _reason_parts.append(_editorial_guidance)
            _retry_reason = "\n\n".join(_reason_parts)
            retry_render_ctx = {
                **context,
                **user_context,
                "retry_reason": _retry_reason,
            }
            user_prompt = render_prompt("sn/generate_name_dd", retry_render_ctx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        # The paid response can return after the stale-claim window. Fence it
        # again before source status, attachment, or candidate graph writes.
        current_claims = await asyncio.to_thread(
            _verify_source_claim_winners,
            batch,
            settle_seconds=0,
        )
        batch = current_claims
        _worker_dependency("_sanitize_compose_result_sources")(
            result,
            _worker_dependency("_claimed_source_ids")(batch),
            phase=phase_tag,
        )

        # Use accumulated totals for per-candidate cost attribution
        cost = _total_compose_cost
        tokens_in = _total_tokens_in
        tokens_out = _total_tokens_out
        tokens_cache_r = _total_cache_read
        tokens_cache_w = _total_cache_creation

        # ── Build candidates ───────────────────────────────────────────
        candidates: list[dict] = []
        disposition_winner_ids: set[str] = set()
        wlog = _worker_dependency(
            "logger"
        )  # plain logger; pool path has no WorkerLogAdapter
        source_kind = "dd"  # pool-mode generate-name is DD-only today
        # Deterministic provenance collapse: one canonical name per base quantity
        # so estimator facets (measured/reconstructed/reference) never drift apart.
        _prov_canon = _worker_dependency("provenance_canonical_names")(
            result.candidates, log=wlog
        )
        for c in result.candidates:
            source_item = next(
                (item for item in batch if item.get("path") == c.source_id),
                None,
            )
            # User invariant (AGENTS.md "Unit safety"): the LLM never
            # decides units; the DD source must provide one.
            # - "1": dimensionless (ISN convention) — valid
            # - "-": DD dimensionless marker — normalize to "1"
            # - "mixed": heterogeneous dimensions — skip
            # - None/empty: enrichment couldn't resolve — skip
            raw_unit = source_item.get("unit") if source_item else None
            if raw_unit == "-":
                raw_unit = "1"  # normalize DD marker to ISN convention
            if raw_unit in ("mixed", None, ""):
                _skip_reason = (
                    "dd_unit_mixed_non_standard"
                    if raw_unit == "mixed"
                    else "dd_unit_unresolvable"
                )
                wlog.warning(
                    "compose(pool): %s, skipping source=%s raw_unit=%r",
                    _skip_reason,
                    c.source_id,
                    raw_unit,
                )
                if c.source_id:
                    try:
                        skip_winners = _worker_dependency("_persist_claimed_skips")(
                            [c.source_id],
                            batch,
                            source_type=source_kind,
                            reason=_skip_reason,
                            detail=str(raw_unit),
                        )
                        disposition_winner_ids.update(skip_winners)
                    except Exception as exc:
                        wlog.debug(
                            "Failed to mark pool source %s skipped: %s",
                            c.source_id,
                            exc,
                        )
                continue
            unit = raw_unit

            # A name's domain is its stored Jev decision, set by the one writer
            # after the batch is persisted; the DD source's own domain is not
            # carried onto the name here.
            physics_domain = None
            source_domains: list[str] = []

            cocos_type = source_item.get("cocos_label") if source_item else None

            # Shape-parameter leaves (triangularity/elongation/squareness) are
            # only meaningful *of* a surface; force the surface locus from the
            # source DD path when the composer left it bare so the leaf is
            # surface-explicit and the boundary/profile siblings de-conflate.
            _worker_dependency("_inject_shape_parameter_surface")(c, c.source_id, wlog)

            # Pre-validation gate — reject malformed LLM output
            # before MERGE.  Mark the source as 'failed' so it is not
            # re-claimed forever.
            name_id = _worker_dependency("normalize_spelling")(c.compose_name())
            # Provenance collapse: estimator facets of one quantity share the
            # canonical name (MERGE identity) regardless of per-facet drift.
            _pterm, _pbase = _worker_dependency("detect_value_provenance")(
                c.source_id or ""
            )
            if _pterm and _pbase in _prov_canon:
                name_id = _prov_canon[_pbase]
            _well_formed, _reject_reason = _worker_dependency(
                "is_well_formed_candidate"
            )(name_id)
            if not _well_formed:
                wlog.warning(
                    "Pool %s: pre-validation reject %r (%s)",
                    phase_tag,
                    name_id[:80],
                    _reject_reason,
                )
                if c.source_id:
                    try:
                        failed = _worker_dependency(
                            "_mark_source_prevalidation_failed"
                        )(
                            c.source_id,
                            source_type=source_kind,
                            candidate_name=name_id,
                            reason=_reject_reason or "malformed",
                            claim_token=(source_item or {}).get("claim_token"),
                            claim_seq=(source_item or {}).get("claim_seq"),
                        )
                        if failed:
                            disposition_winner_ids.add(f"{source_kind}:{c.source_id}")
                    except Exception:
                        wlog.debug("Failed to mark source %s as failed", c.source_id)
                continue

            # Non-nameable coordinate/infrastructure gate: a bare timestamp,
            # latency, counter, or metadata token is not a physics observable.
            # Composing it produces a doomed bare name that fails the semantic
            # gate and burns review + every refine rotation before exhausting.
            # Route the source to ``skipped`` (terminal) so it never re-claims.
            _non_nameable, _skip_reason = _worker_dependency(
                "is_non_nameable_coordinate"
            )(name_id)
            if _non_nameable:
                wlog.warning(
                    "Pool %s: non-nameable skip %r (%s)",
                    phase_tag,
                    name_id[:80],
                    _skip_reason,
                )
                if c.source_id:
                    try:
                        skip_winners = _worker_dependency("_persist_claimed_skips")(
                            [c.source_id],
                            batch,
                            source_type=source_kind,
                            reason=_skip_reason or "non_nameable_coordinate",
                            detail=f"bare non-nameable token: {name_id}",
                        )
                        disposition_winner_ids.update(skip_winners)
                    except Exception:
                        wlog.debug("Failed to mark source %s skipped", c.source_id)
                continue

            # Deterministic source<->name consistency gate (tense + state
            # resolution). The LLM sometimes emits a species-level sibling
            # for a state-resolved source (e.g. thermal_neutral_density
            # sourced from .../neutral/state/density_thermal) -- drop the
            # pairing so the source retries rather than persisting a
            # mis-attributed name. Same predicate the auto-attach path uses.
            if c.source_id:
                _paths = [
                    c.source_id,
                    *(p for p in (c.dd_paths or []) if p != c.source_id),
                ]
                _ok, _why = True, ""
                for _p in _paths:
                    _ok, _why = _worker_dependency("_is_attachment_consistent")(
                        _p, name_id, existing_sources=[q for q in _paths if q != _p]
                    )
                    if not _ok:
                        break
                if not _ok:
                    wlog.warning(
                        "Source-name consistency reject: %r (%s)",
                        name_id[:80],
                        _why,
                    )
                    continue

            grammar_failed = False
            try:
                from imas_codex.standard_names.grammar_adapter import (
                    parse_canonical_name,
                )

                parse_canonical_name(name_id)
            except Exception as gram_exc:
                grammar_failed = True
                wlog.debug(
                    "Pool %s: grammar parse failed for %r — attempting retry",
                    phase_tag,
                    name_id,
                )

                # Single-shot grammar-failure retry.
                try:
                    retry_name, _r_cost, _r_ti, _r_to = await _worker_dependency(
                        "_grammar_retry"
                    )(
                        name_id,
                        str(gram_exc),
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
                    )
                    if lease and _r_cost > 0:
                        _r_event = LLMCostEvent(
                            model=model,
                            tokens_in=_r_ti,
                            tokens_out=_r_to,
                            sn_ids=(name_id,),
                            batch_id=f"{group_key}-grammar-retry",
                            phase=phase_tag,
                            service="standard-names",
                        )
                        lease.charge_event(_r_cost, _r_event)
                    if retry_name and retry_name != name_id:
                        try:
                            parse_canonical_name(retry_name)
                            name_id = retry_name
                            grammar_failed = False
                            wlog.info(
                                "Pool %s: grammar retry succeeded %r → %r",
                                phase_tag,
                                c.compose_name(),
                                name_id,
                            )
                        except Exception:
                            wlog.debug(
                                "Pool %s: retry result still un-parseable",
                                phase_tag,
                            )
                except Exception as retry_exc:
                    _worker_dependency("charge_billable_exception")(
                        lease,
                        retry_exc,
                        model=model,
                        sn_ids=(name_id,),
                        batch_id=f"{group_key}-grammar-retry",
                        phase=phase_tag,
                    )
                    wlog.debug(
                        "Pool %s: grammar retry failed for %r",
                        phase_tag,
                        name_id,
                    )

            cand_description = _worker_dependency("normalize_description_text")(
                c.description or ""
            )

            # ── Free local self-refine pass (default off) ─────────────────
            # After compose + grammar normalization and BEFORE persist (so
            # before the paid review quorum), let the LOCAL compose model
            # critique its own name + description and emit an improved
            # candidate. Improve-or-no-op; the rewrite is re-validated and
            # discarded if it fails grammar. OFF = byte-identical to the path
            # without this block.
            #
            # The pass discards its LLMResult, so it can only run where the
            # call is free: on a paid route its spend would leave no cost
            # record and no reservation. Refuse rather than spend silently.
            if not grammar_failed and get_compose_self_refine():
                from imas_codex.discovery.base.llm import _is_local_model

                if not _is_local_model(model):
                    raise RuntimeError(
                        "compose self-refine records no cost and reserves no "
                        f"budget, so it cannot run on the metered route {model}"
                    )
                try:
                    _ref_name, _ref_desc = await _worker_dependency(
                        "_self_refine_candidate"
                    )(
                        name_id,
                        cand_description,
                        c.segments.model_dump() if hasattr(c, "segments") else None,
                        source_item,
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
                    )
                    if _ref_name != name_id:
                        wlog.info(
                            "Pool %s: self-refine %r → %r",
                            phase_tag,
                            name_id,
                            _ref_name,
                        )
                    name_id = _ref_name
                    cand_description = _ref_desc
                except Exception:
                    wlog.debug(
                        "Pool %s: self-refine failed for %r — keeping original",
                        phase_tag,
                        name_id,
                    )

            cand = {
                "id": name_id,
                "source_types": [source_kind],
                "source_id": c.source_id,
                "description": cand_description,
                "kind": c.kind,
                "source_paths": [
                    _worker_dependency("encode_source_path")(source_kind, p)
                    for p in (c.dd_paths or [])
                ],
                "reason": c.reason,
                "unit": unit,
                "physics_domain": physics_domain,
                "source_domains": source_domains,
                "cocos_transformation_type": cocos_type,
                "cocos": batch[0].get("cocos_version") if batch else None,
                "dd_version": batch[0].get("dd_version") if batch else None,
                "source_claim_token": (source_item or {}).get("claim_token"),
                "source_claim_seq": (source_item or {}).get("claim_seq"),
                **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
            }

            if regen:
                cand["regen_increment"] = True

            candidates.append(cand)

            # ── Deterministic error-sibling minting ──────────────────
            if (
                not grammar_failed
                and source_item
                and source_item.get("has_errors")
                and source_item.get("error_node_ids")
            ):
                try:
                    from imas_codex.standard_names.error_siblings import (
                        mint_error_siblings,
                    )

                    siblings = mint_error_siblings(
                        name_id,
                        error_node_ids=source_item["error_node_ids"],
                        unit=unit,
                        physics_domain=physics_domain,
                        cocos_type=cocos_type,
                        cocos_version=batch[0].get("cocos_version") if batch else None,
                        dd_version=batch[0].get("dd_version") if batch else None,
                    )
                    if siblings:
                        for s in siblings:
                            s["_from_error_sibling"] = True
                        candidates.extend(siblings)
                        wlog.debug(
                            "Pool %s: minted %d error siblings for %r",
                            phase_tag,
                            len(siblings),
                            name_id,
                        )
                except Exception:
                    wlog.debug(
                        "Pool %s: error-sibling minting failed for %r",
                        phase_tag,
                        name_id,
                        exc_info=True,
                    )

        # ── Per-candidate cost / token attribution ────────────────────
        # Pro-rata across LLM-generated candidates only — error siblings
        # are deterministic and carry no LLM cost.
        from datetime import UTC, datetime

        _llm_cands = [c for c in candidates if not c.get("_from_error_sibling")]
        n_cands = max(len(_llm_cands), 1)
        _llm_at = datetime.now(UTC).isoformat()
        for cand in candidates:
            if cand.get("_from_error_sibling"):
                continue
            cand["llm_cost"] = cost / n_cands
            cand["llm_model"] = model
            cand["llm_service"] = "standard-names"
            cand["llm_at"] = _llm_at
            cand["llm_tokens_in"] = tokens_in // n_cands
            cand["llm_tokens_out"] = tokens_out // n_cands
            cand["llm_tokens_cached_read"] = tokens_cache_r // n_cands
            cand["llm_tokens_cached_write"] = tokens_cache_w // n_cands

        # ── Inline audits — populate validation_status ────────────────
        try:
            from imas_codex.standard_names.audits import run_audits

            for cand in candidates:
                if cand.get("_from_error_sibling"):
                    cand.setdefault("validation_status", "valid")
                    continue
                src_paths = cand.get("source_paths") or []
                src_path = (
                    _worker_dependency("strip_dd_prefix")(src_paths[0])
                    if src_paths
                    else None
                )
                try:
                    audit_issues = run_audits(
                        candidate=cand,
                        existing_sns_in_domain=None,
                        source_path=src_path,
                        source_cocos_type=cand.get("cocos_transformation_type"),
                    )
                except Exception:
                    audit_issues = []
                if cand.get("_grammar_retry_exhausted"):
                    audit_issues = list(audit_issues) + [
                        "audit:grammar_retry_exhausted"
                    ]
                cand["validation_issues"] = audit_issues
                cand["validation_status"] = (
                    "quarantined"
                    if _worker_dependency("_is_quarantined")(list(audit_issues), {})
                    else "valid"
                )
        except Exception:
            wlog.debug("Pool %s: audits failed", phase_tag, exc_info=True)
            for cand in candidates:
                cand.setdefault("validation_status", "valid")

        # Strip private flag before persist
        for cand in candidates:
            cand.pop("_grammar_retry_exhausted", None)

        # Re-check exact token+sequence ownership immediately before any graph
        # mutation. A worker that lost its source claim while the LLM was
        # running must not persist any part of that delayed response.
        current_claims = await asyncio.to_thread(
            _verify_source_claim_winners,
            batch,
            settle_seconds=0,
        )
        _worker_dependency("_sanitize_compose_result_sources")(
            result,
            _worker_dependency("_claimed_source_ids")(current_claims),
            phase=phase_tag,
        )
        current_source_ids = _worker_dependency("_claimed_source_ids")(current_claims)
        candidates = _worker_dependency(
            "_filter_persist_candidates_to_claimed_sources"
        )(
            candidates,
            current_source_ids,
            phase=phase_tag,
        )

        # ── Vocab gaps — persist + clear sources ──────────────────────
        if result.vocab_gaps:
            gap_dicts = [
                {
                    "source_id": vg.source_id,
                    "segment": vg.segment,
                    "token": vg.token,
                    "reason": vg.reason,
                }
                for vg in result.vocab_gaps
            ]
            # Stamp the compose-time token-reuse adjudication: gaps re-emitted
            # after being shown a near-synonym registered token are
            # distinct_confirmed; the rest are unchecked.
            _worker_dependency("_stamp_dedup_decision")(gap_dicts, _token_reuse_hits)
            gap_dicts, gap_winner_ids = await asyncio.to_thread(
                _worker_dependency("_consume_claimed_vocab_gaps"),
                gap_dicts,
                batch,
                source_type=source_kind,
                return_winner_ids=True,
            )
            disposition_winner_ids.update(gap_winner_ids)
            wlog.debug("Pool %s: persisted %d vocab gaps", phase_tag, len(gap_dicts))
        # ── Attachments — write edges + clear source claims ───────────
        if result.attachments:
            try:
                attach_counts = await asyncio.to_thread(
                    _worker_dependency("_process_attachments_core"),
                    result.attachments,
                    wlog,
                    source_claims=batch,
                    return_winner_ids=True,
                )
                disposition_winner_ids.update(attach_counts["winner_ids"])
                wlog.info(
                    "Pool %s: attached %d (rejected %d)",
                    phase_tag,
                    attach_counts.get("accepted", 0),
                    attach_counts.get("rejected", 0),
                )
            except Exception:
                wlog.debug(
                    "Pool %s: attachment processing failed",
                    phase_tag,
                    exc_info=True,
                )
        # ── Skipped sources — clear claims so they don't loop ─────────
        if result.skipped:
            try:
                skipped_winner_ids = await asyncio.to_thread(
                    _worker_dependency("_persist_claimed_skips"),
                    list(result.skipped),
                    batch,
                    source_type=source_kind,
                    reason="compose_model_skipped",
                )
                disposition_winner_ids.update(skipped_winner_ids)
                wlog.debug(
                    "Pool %s: marked %d sources as skipped",
                    phase_tag,
                    len(skipped_winner_ids),
                )
            except Exception:
                wlog.debug(
                    "Pool %s: skipped source-status update failed",
                    phase_tag,
                    exc_info=True,
                )

        # ── Persist ────────────────────────────────────────────────────
        if candidates:
            persisted = await asyncio.to_thread(
                _worker_dependency("_persist_description_checked_candidates"),
                candidates,
                source_type=source_kind,
                phase=phase_tag,
                compose_model=model,
                dd_version=batch[0].get("dd_version"),
                cocos_version=batch[0].get("cocos_version"),
                run_id=mgr.run_id,
            )
            # Only the exact ID-returning persistence contract proves which
            # source claims won the CAS. A legacy count cannot authorize a
            # source-specific evidence mutation.
            winner_ids = persisted if isinstance(persisted, list) else []
            winner_id_set = set(winner_ids)
            disposition_winner_ids.update(winner_ids)
            if isinstance(persisted, list):
                candidates = [
                    candidate
                    for candidate in candidates
                    if f"{source_kind}:{candidate.get('source_id')}" in winner_id_set
                ]
                written = len(candidates)
            else:
                written = int(persisted)
            logger.debug("Pool %s: persisted %d candidates", phase_tag, written)

            # A persisted name's domain is its stored Jev decision, set by the
            # one writer rather than carried from the source item.
            if isinstance(persisted, list) and candidates:
                await _worker_dependency("_write_composed_name_domains")(
                    candidates,
                    {str(c.get("source_id")) for c in candidates},
                )

            # Derive auto-gaps only from finalized source winners. A losing
            # source cannot authorize a gap mutation.
            if candidates:
                try:
                    auto_gaps = _worker_dependency("_auto_detect_physical_base_gaps")(
                        candidates
                    )
                    if auto_gaps:
                        from imas_codex.standard_names.graph_store.graph_catalog_reset import (
                            write_vocab_gaps,
                        )

                        await asyncio.to_thread(
                            write_vocab_gaps,
                            auto_gaps,
                            source_kind,
                            skip_segment_filter=True,
                        )
                except Exception:
                    wlog.debug(
                        "Pool %s: physical_base gap detection failed",
                        phase_tag,
                        exc_info=True,
                    )

            # ── Emit per-item events ──────────────────────────────────
            if on_event is not None:
                for cand in candidates:
                    on_event(
                        {
                            "pool": "generate_name",
                            "name": cand["id"],
                            "source": (cand.get("source_paths") or [""])[0],
                            "dd_path": (cand.get("source_paths") or [""])[0],
                            "model": model,
                            "cost": cand.get("llm_cost", 0.0),
                        }
                    )

        committed_source_ids = {
            winner.split(":", 1)[1]
            for winner in disposition_winner_ids
            if isinstance(winner, str) and winner.startswith(f"{source_kind}:")
        }
        source_versions = {
            source_id: item.get("dd_version")
            for item in batch
            if (source_id := (item.get("path") or item.get("source_id")))
            in committed_source_ids
        }
        await asyncio.to_thread(
            _worker_dependency("_persist_dd_gap_evidence"),
            _collected_dd_gaps,
            committed_source_ids,
            phase=phase_tag,
            reporter="compose",
            source_versions=source_versions,
        )

        return written if candidates else 0

    except Exception:
        logger.exception("Pool %s: compose batch failed", phase_tag)
        raise
    finally:
        if lease:
            lease.release_unused()


async def process_generate_name_batch(
    _worker_dependency,
    /,
    batch: list[dict],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    compose_model: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Pool-mode generate-name batch processor.

    Takes pre-claimed source items (dicts from the graph claim query),
    generates standard names via LLM, and persists results.

    Returns count of items successfully processed.
    """
    return await _worker_dependency("compose_batch")(
        batch,
        mgr,
        stop_event,
        regen=False,
        compose_model=compose_model,
        on_event=on_event,
    )


# Substrings that identify a deterministic grammar / vocab / schema failure
# in an LLM-proposed refined name.  These are NORMAL failed-refine outcomes
# (the model proposed an ungrammatical name), not crashes — the item is
# routed to the exhaust path rather than re-claimed and re-charged forever.
_GRAMMAR_FAILURE_MARKERS: tuple[str, ...] = (
    "not a registered",  # unregistered grammar token (base/qualifier/axis)
    "kind must be one of",  # legacy schema enum violation message
    "input should be",  # pydantic Literal enum violation (deterministic)
    "self-referential refined_from",  # refine produced an identical name
    "validation error for refinedname",  # pydantic RefinedName validation
    # ISN StandardName grammar composition raised inside the refine compose
    # step (compose_standard_name → StandardName.model_validate): the model
    # proposed a name whose tokens/structure are not grammatical (invalid
    # operator token, unregistered base, malformed transformation). Same
    # output every cycle, so terminal — not a crash.
    "validation error for standardname",
    "invalid operator token",  # deterministic fused-operator/coordinate misuse
)

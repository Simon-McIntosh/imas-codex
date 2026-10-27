"""Compose, validate, consolidate, and persist worker tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re as _re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetLease
    from imas_codex.standard_names.sources.base import ExtractionBatch
    from imas_codex.standard_names.state import StandardNameBuildState


async def compose_worker(
    _worker_dependency, /, state: StandardNameBuildState, **_kwargs
) -> None:
    """LLM-generate standard names from extracted batches.

    Uses acall_llm_structured() with system/user prompt split for
    prompt caching.  Runs batches concurrently with semaphore.
    Results stored in ``state.composed``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_compose_worker")

    total_items = sum(len(b.items) for b in state.extracted)

    if state.dry_run:
        wlog.info("Dry run — skipping composition for %d items", total_items)
        state.compose_stats.total = total_items
        state.compose_stats.processed = total_items
        state.stats["compose_skipped"] = True
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    if not state.extracted:
        wlog.info("No batches to compose — skipping")
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model, get_reasoning_effort
    from imas_codex.standard_names.budget import LLMCostEvent
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameComposeBatch

    model = state.compose_model or get_model("sn-compose")
    context = build_compose_context()

    # Enrich batch items with rich DD context (coordinate specs, COCOS, siblings,
    # cross-IDS paths, version history)
    def _enrich_all_batches():
        for batch in state.extracted:
            _worker_dependency("_enrich_batch_items")(batch.items)

    await asyncio.to_thread(_enrich_all_batches)
    wlog.info("Enriched batch items with DD context")

    # Propagate COCOS metadata to state (for downstream phases)
    if state.extracted:
        first_batch = state.extracted[0]
        state.dd_version = first_batch.dd_version
        state.cocos_version = first_batch.cocos_version
        state.cocos_params = first_batch.cocos_params

    # Pre-fetch IDS-level context for each unique IDS across batches
    ids_context_cache: dict[str, dict | None] = {}

    def _prefetch_ids_context():
        for batch in state.extracted:
            # Derive IDS names from item paths (first segment)
            ids_names = {
                item["path"].split("/")[0]
                for item in batch.items
                if item.get("path") and "/" in item["path"]
            }
            for ids_name in ids_names:
                if ids_name not in ids_context_cache:
                    ids_context_cache[ids_name] = _worker_dependency(
                        "_enrich_ids_context"
                    )(ids_name)

    await asyncio.to_thread(_prefetch_ids_context)
    wlog.info("Fetched IDS context for %d IDS(s)", len(ids_context_cache))

    # Render system prompt once (cached via prompt caching)
    # Inject COCOS context for system prompt (all batches share one DD version)
    if state.extracted:
        first_batch = state.extracted[0]
        if first_batch.cocos_version:
            context["cocos_version"] = first_batch.cocos_version
            context["dd_version"] = first_batch.dd_version
            if first_batch.cocos_params:
                context["cocos_sigma_bp"] = first_batch.cocos_params.get("sigma_bp")
                context["cocos_e_bp"] = first_batch.cocos_params.get("e_bp")
                context["cocos_sigma_r_phi_z"] = first_batch.cocos_params.get(
                    "sigma_r_phi_z"
                )
                context["cocos_sigma_rho_theta_phi"] = first_batch.cocos_params.get(
                    "sigma_rho_theta_phi"
                )

    # --- Domain-vocabulary pre-seeding ---
    # Inject validated domain vocabulary into system prompt context
    from imas_codex.standard_names.context import build_domain_vocabulary_preseed

    domain_vocab = ""
    if state.domain_filter:
        domain_vocab = await asyncio.to_thread(
            build_domain_vocabulary_preseed, state.domain_filter
        )
        if domain_vocab:
            wlog.info("Injected domain vocabulary preseed for %s", state.domain_filter)
    context["domain_vocabulary"] = domain_vocab

    # --- Reviewer-theme extraction ---
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    reviewer_themes: list[str] = []
    if state.domain_filter:
        reviewer_themes = await asyncio.to_thread(
            extract_reviewer_themes, state.domain_filter
        )
        if reviewer_themes:
            wlog.info(
                "Extracted %d reviewer themes for %s",
                len(reviewer_themes),
                state.domain_filter,
            )
    context["reviewer_themes"] = reviewer_themes

    # --- Scored-example injection ---
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_compose_examples

    # Derive physics_domains from domain_filter and batch items
    batch_domains: list[str] = []
    if state.domain_filter:
        batch_domains = [state.domain_filter]
    else:
        # Collect unique domains from all batch items
        _domains = {
            item.get("physics_domain")
            for batch in state.extracted
            for item in batch.items
            if item.get("physics_domain")
        }
        batch_domains = sorted(_domains)

    def _load_scored_examples() -> list[dict]:
        with GraphClient() as gc:
            return load_compose_examples(gc, physics_domains=batch_domains, axis="name")

    compose_scored_examples = await asyncio.to_thread(_load_scored_examples)
    if compose_scored_examples:
        wlog.info(
            "Loaded %d scored examples for compose (domains=%s)",
            len(compose_scored_examples),
            batch_domains or "all",
        )
    context["compose_scored_examples"] = compose_scored_examples

    # --- NC rules injection ---
    # Load naming-consistency rules from YAML so the system prompt
    # {% include "_nc_rules.md" %} block renders the full rule set.
    from imas_codex.llm.prompt_loader import load_prompt_config

    try:
        _rules_cfg = load_prompt_config("sn_composition_rules")
        context["composition_rules"] = _rules_cfg.get("composition_rules", [])
    except Exception:
        wlog.debug("NC rules YAML not found — skipping injection")
        context["composition_rules"] = []

    system_prompt = render_prompt("sn/generate_name_system", context)

    wlog.info(
        "Composing standard names for %d items in %d batches (model=%s)",
        total_items,
        len(state.extracted),
        model,
    )
    state.compose_stats.total = total_items

    from imas_codex.settings import get_compose_concurrency

    async def _compose_batch_body(
        batch: ExtractionBatch, lease_box: list[BudgetLease]
    ) -> list[dict] | None:
        _collect_nearby_name_comparators = _worker_dependency(
            "_collect_nearby_name_comparators"
        )
        _rerank_comparators_for_worker = _worker_dependency(
            "_rerank_comparators_for_worker"
        )
        # Search for nearby existing names with a bounded share per item.
        nearby = await asyncio.to_thread(_collect_nearby_name_comparators, batch.items)
        nearby = await _rerank_comparators_for_worker(state, batch.items, nearby)

        # IDS-level context — collect for each IDS present in batch
        ids_names = sorted(
            {
                item["path"].split("/")[0]
                for item in batch.items
                if item.get("path") and "/" in item["path"]
            }
        )
        ids_contexts = []
        for iname in ids_names:
            info = ids_context_cache.get(iname)
            if info:
                ids_contexts.append({"ids_name": iname, **info})

        # Pre-render COCOS guidance for items
        if batch.cocos_params:
            from imas_codex.standard_names.context import render_cocos_guidance

            for item in batch.items:
                cocos_label = item.get("cocos_label")
                if cocos_label:
                    item["cocos_guidance"] = render_cocos_guidance(
                        cocos_label, batch.cocos_params
                    )

        # --- Rate-quantity detection ---
        # When DD documentation indicates a rate/time-derivative, inject a
        # hard constraint so the LLM uses tendency_of_/change_in_ prefix
        # and writes a consistent description (prevents instant_change_*
        # names and name/description verb drift).

        _RATE_DOC_PATTERNS = _re.compile(
            r"\b(instantaneous change|signed change|rate of change"
            r"|time derivative|per unit time|instant change|d/dt"
            r"|tendency of|time-rate)\b",
            _re.IGNORECASE,
        )
        for item in batch.items:
            haystack = " ".join(
                str(item.get(k, "") or "") for k in ("description", "documentation")
            )
            if _RATE_DOC_PATTERNS.search(haystack):
                item["rate_hint"] = True

        # --- Reference SN few-shot retrieval ---
        # Synthesize query from first 3 path descriptions
        reference_exemplars: list[dict] = []
        try:
            from imas_codex.standard_names.search import (
                search_standard_names_with_documentation,
            )

            desc_snippets = [
                item.get("description", "")
                for item in batch.items[:3]
                if item.get("description")
            ]
            if desc_snippets:
                synth_query = "; ".join(desc_snippets)
                # Exclude names already in this batch's candidate IDs
                batch_ids = [
                    item.get("path", "").replace("/", "_") for item in batch.items
                ]
                reference_exemplars = await asyncio.to_thread(
                    search_standard_names_with_documentation,
                    synth_query,
                    k=5,
                    exclude_ids=batch_ids,
                )
        except Exception:
            wlog.debug("Reference exemplar search failed", exc_info=True)

        user_context = {
            "items": batch.items,
            "ids_name": batch.group_key,
            "ids_contexts": ids_contexts,
            "existing_names": sorted(batch.existing_names)[:200],
            "cluster_context": batch.context,
            "nearby_existing_names": nearby,
            "reference_exemplars": reference_exemplars,
            "cocos_version": batch.cocos_version,
            "dd_version": batch.dd_version,
        }
        # Name-only batches render a leaner user prompt
        # that trades per-item cluster siblings / COCOS blocks / sibling
        # fields for a "identify natural sub-groups, then name" directive.
        # System prompt and the per-candidate grammar-retry / revision
        # logic are unchanged so
        # prompt caching and grammar safety stay intact.
        prompt_template = (
            "sn/generate_name_dd_names"
            if batch.mode == "names"
            else "sn/generate_name_dd"
        )
        user_prompt = render_prompt(prompt_template, {**context, **user_context})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        lease = None
        if state.budget_manager:
            maximum_exposure = _worker_dependency("model_provider_exposure")(
                model,
                messages,
                response_model=StandardNameComposeBatch,
                provider_attempts=1,
            )
            phase_tag = _worker_dependency("_compose_budget_phase")(state)
            lease = state.budget_manager.reserve(maximum_exposure, phase=phase_tag)
            if lease is None:
                return None
            lease_box.append(lease)

        # --- Bounded retry loop for failed compositions ---
        # Accumulate full LLMResult fields across retries so the batch's
        # per-candidate cost/token attribution is accurate.  Compose is
        # unified with review/enrich here: every LLM call site extracts
        # the same cost/token fields from LLMResult and writes them to
        # the graph for single-source cost observability.
        _total_compose_cost = 0.0
        _total_tokens_in = 0
        _total_tokens_out = 0
        _total_cache_read = 0
        _total_cache_creation = 0
        _collected_dd_gaps: list[Any] = []
        _max_retries = _worker_dependency("_retry_attempts")()
        # Token-reuse hits from the FINAL attempt: candidate-new tokens the
        # agent re-emitted after being shown a near-synonym registered token.
        # Carried to the post-loop VocabGap stamp as distinct_confirmed.
        _token_reuse_hits: dict[tuple[str, str], Any] = {}
        _editorial_guidance = ""
        _editorial_should_retry = False
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
                    batch_id=batch.group_key,
                    phase=getattr(state, "budget_phase_tag", "") or "generate_name",
                )
                _exc_str = str(compose_exc)
                if "not a registered" in _exc_str:
                    logger.warning(
                        "Compose: vocab gap validation error — marking "
                        "%d sources as vocab_gap: %s",
                        len(batch),
                        _exc_str[:200],
                    )
                    _worker_dependency("_mark_vocab_gap_sources")(
                        batch.items, _exc_str, "dd"
                    )
                    return []
                raise
            result, cost, tokens = llm_out
            _worker_dependency("_sanitize_compose_result_sources")(
                result,
                _worker_dependency("_claimed_source_ids")(batch.items),
                phase="generate_name",
            )
            _collected_dd_gaps.extend(result.dd_gaps)
            _total_compose_cost += cost
            _total_tokens_in += getattr(llm_out, "input_tokens", 0) or 0
            _total_tokens_out += getattr(llm_out, "output_tokens", 0) or 0
            _total_cache_read += getattr(llm_out, "cache_read_tokens", 0) or 0
            _total_cache_creation += getattr(llm_out, "cache_creation_tokens", 0) or 0

            # Charge the exact completed-call cost against the exposure that
            # was reserved before the provider request started.
            if lease:
                _event = LLMCostEvent(
                    model=model,
                    tokens_in=getattr(llm_out, "input_tokens", 0) or 0,
                    tokens_out=getattr(llm_out, "output_tokens", 0) or 0,
                    tokens_cached_read=getattr(llm_out, "cache_read_tokens", 0) or 0,
                    tokens_cached_write=(
                        getattr(llm_out, "cache_creation_tokens", 0) or 0
                    ),
                    sn_ids=tuple(
                        c.compose_name() for c in (result.candidates if result else [])
                    ),
                    batch_id=batch.group_key,
                    phase=getattr(state, "budget_phase_tag", "") or "generate_name",
                    service="standard-names",
                )
                _charge = lease.charge_event(cost, _event)
                if state.loop_stats is not None:
                    state.loop_stats.processed += max(1, len(result.candidates))
                    state.loop_stats.cost += cost
                    # Stream one item per candidate so the user sees names
                    # rotate through with grammar composition.
                    _items: list[dict[str, Any]] = []
                    for _cand in result.candidates[:50]:
                        _segs = []
                        if _cand.base_token:
                            _segs.append(f"base={_cand.base_token}")
                        for _q in _cand.qualifiers:
                            _segs.append(f"qualifier={_q}")
                        if _cand.projection_axis:
                            _segs.append(f"projection={_cand.projection_axis}")
                        if _cand.process_token:
                            _segs.append(f"process={_cand.process_token}")
                        for _operator in _cand.operators:
                            _segs.append(f"operator={_operator.token}")
                        _desc = (
                            "  ".join(_segs) if _segs else ((_cand.reason or "")[:80])
                        )
                        try:
                            _composed = _cand.compose_name()
                        except Exception:
                            _composed = f"[IR error] base={_cand.base_token}"
                        _items.append(
                            {
                                "primary_text": _composed,
                                "primary_text_style": "white",
                                "description": _desc,
                            }
                        )
                    if not _items:
                        _items.append(
                            {
                                "primary_text": (
                                    f"sn={_event.sn_ids[0]}"
                                    if _event.sn_ids
                                    else f"batch={_event.batch_id}"
                                ),
                                "primary_text_style": "white",
                                "description": batch.group_key,
                            }
                        )
                    state.loop_stats.stream_queue.add(_items)
                if _charge.overspend > 0:
                    wlog.warning(
                        "Compose batch %s overspent reservation by $%.4f "
                        "(batch cost $%.4f); budget tracking will report overrun",
                        batch.group_key,
                        _charge.overspend,
                        cost,
                    )

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
            wlog.info(
                "Composition retry %d/%d: %d grammar failures (%s), "
                "%d token-reuse hits, editorial_retry=%s — "
                "re-composing with expanded DD context",
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
                        for item in batch.items
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
                        for item in batch.items:
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
            user_prompt = render_prompt(prompt_template, retry_render_ctx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        state.compose_stats.cost += _total_compose_cost
        state.compose_stats.processed += len(batch.items)
        state.compose_stats.record_batch(len(batch.items))

        candidates = []
        # Deterministic provenance collapse (see pooled path): one canonical
        # name per base quantity so estimator facets never drift apart.
        _prov_canon = _worker_dependency("provenance_canonical_names")(
            result.candidates, log=wlog
        )
        for c in result.candidates:
            # Find the matching source item to get authoritative fields
            source_item = next(
                (item for item in batch.items if item.get("path") == c.source_id),
                None,
            )
            # Inject unit from DD (authoritative, not LLM output).
            #
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
                    "compose: %s, skipping source=%s raw_unit=%r",
                    _skip_reason,
                    c.source_id,
                    raw_unit,
                )
                state.compose_stats.errors += 1
                continue
            unit = raw_unit

            # A name's domain is its stored Jev decision, set by the one writer
            # after the batch is persisted. The DD source's own domain is not
            # carried onto the name here.
            physics_domain = None
            source_domains: list[str] = []

            # Inject COCOS metadata from DD (authoritative, like unit)
            cocos_type = source_item.get("cocos_label") if source_item else None

            # Shape-parameter leaves (triangularity/elongation/squareness) are
            # only meaningful *of* a surface; force the surface locus from the
            # source DD path when the composer left it bare so the leaf is
            # surface-explicit and the boundary/profile siblings de-conflate.
            _worker_dependency("_inject_shape_parameter_surface")(c, c.source_id, wlog)

            # Normalize name via grammar round-trip BEFORE persist
            # to avoid duplicate nodes if validate would rename
            name_id = _worker_dependency("normalize_spelling")(c.compose_name())
            # Provenance collapse: estimator facets of one quantity share the
            # canonical name (MERGE identity) regardless of per-facet drift.
            _pterm, _pbase = _worker_dependency("detect_value_provenance")(
                c.source_id or ""
            )
            if _pterm and _pbase in _prov_canon:
                name_id = _prov_canon[_pbase]

            # Pre-validation gate — reject malformed LLM output
            # before it reaches MERGE.
            _well_formed, _reject_reason = _worker_dependency(
                "is_well_formed_candidate"
            )(name_id)
            if not _well_formed:
                state.stats["compose_pre_validation_rejects"] = (
                    state.stats.get("compose_pre_validation_rejects", 0) + 1
                )
                wlog.warning(
                    "Pre-validation reject: %r (%s)",
                    name_id[:80],
                    _reject_reason,
                )
                # Mark source as failed (if identifiable)
                if c.source_id and not state.dry_run:
                    try:
                        _prefix = "dd" if state.source == "dd" else "signals"
                        _worker_dependency("_mark_source_prevalidation_failed")(
                            c.source_id,
                            source_type=_prefix,
                            candidate_name=name_id,
                            reason=_reject_reason or "malformed",
                        )
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
                state.stats["compose_non_nameable_skips"] = (
                    state.stats.get("compose_non_nameable_skips", 0) + 1
                )
                wlog.warning(
                    "Non-nameable skip: %r (%s)",
                    name_id[:80],
                    _skip_reason,
                )
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
                    state.stats["compose_consistency_rejects"] = (
                        state.stats.get("compose_consistency_rejects", 0) + 1
                    )
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
                wlog.debug("Grammar parse failed for %r — attempting retry", name_id)

                # --- Grammar-failure re-prompt (single retry) ---
                state.grammar_retries += 1
                try:
                    retry_name, _l6_cost, _l6_ti, _l6_to = await _worker_dependency(
                        "_grammar_retry"
                    )(
                        name_id,
                        str(gram_exc),
                        model,
                        acall_llm_structured,
                        reasoning_effort=get_reasoning_effort("sn-compose"),
                    )
                    if lease and _l6_cost > 0:
                        _l6_event = LLMCostEvent(
                            model=model,
                            tokens_in=_l6_ti,
                            tokens_out=_l6_to,
                            sn_ids=(name_id,),
                            batch_id=f"{batch.group_key}-grammar-retry",
                            phase=(
                                getattr(state, "budget_phase_tag", "")
                                or "generate_name"
                            ),
                            service="standard-names",
                        )
                        lease.charge_event(_l6_cost, _l6_event)
                        if state.loop_stats is not None:
                            state.loop_stats.cost += _l6_cost
                            state.loop_stats.stream_queue.add(
                                [
                                    {
                                        "primary_text": f"sn={name_id}",
                                        "primary_text_style": "white",
                                        "description": f"{batch.group_key}-grammar-retry",
                                    }
                                ]
                            )
                    if retry_name and retry_name != name_id:
                        # Verify the retry result actually parses
                        parse_canonical_name(retry_name)
                        name_id = retry_name
                        grammar_failed = False
                        state.grammar_retries_succeeded += 1
                        wlog.info(
                            "Grammar retry succeeded: %r → %r",
                            c.compose_name(),
                            name_id,
                        )
                except Exception as retry_exc:
                    _worker_dependency("charge_billable_exception")(
                        lease,
                        retry_exc,
                        model=model,
                        sn_ids=(name_id,),
                        batch_id=f"{batch.group_key}-grammar-retry",
                        phase=getattr(state, "budget_phase_tag", "") or "generate_name",
                    )
                    wlog.debug("Grammar retry also failed for %r", name_id)

            candidates.append(
                {
                    "id": name_id,
                    "source_types": ["dd"] if state.source == "dd" else ["signals"],
                    "source_id": c.source_id,
                    "kind": c.kind,
                    "source_paths": [
                        _worker_dependency("encode_source_path")(
                            "dd" if state.source == "dd" else "signals", p
                        )
                        for p in (c.dd_paths or [])
                    ],
                    "reason": c.reason,
                    "unit": unit,
                    "physics_domain": physics_domain,
                    "source_domains": source_domains,
                    "cocos_transformation_type": cocos_type,
                    "cocos": batch.cocos_version,
                    "dd_version": batch.dd_version,
                    "source_claim_token": (source_item or {}).get("claim_token"),
                    "source_claim_seq": (source_item or {}).get("claim_seq"),
                    # Track grammar-retry exhaustion
                    **({"_grammar_retry_exhausted": True} if grammar_failed else {}),
                }
            )

            # --- Mint error siblings deterministically ---
            # If the parent has HAS_ERROR edges, mint uncertainty
            # modifier siblings without LLM calls.
            if (
                not grammar_failed
                and source_item
                and source_item.get("has_errors")
                and source_item.get("error_node_ids")
            ):
                from imas_codex.standard_names.error_siblings import (
                    mint_error_siblings,
                )

                siblings = mint_error_siblings(
                    name_id,
                    error_node_ids=source_item["error_node_ids"],
                    unit=unit,
                    physics_domain=physics_domain,
                    cocos_type=cocos_type,
                    cocos_version=batch.cocos_version,
                    dd_version=batch.dd_version,
                )
                if siblings:
                    for s in siblings:
                        s["_from_error_sibling"] = True
                    candidates.extend(siblings)
                    state.error_siblings_minted = getattr(
                        state, "error_siblings_minted", 0
                    ) + len(siblings)
                    wlog.debug(
                        "Minted %d error siblings for parent %r",
                        len(siblings),
                        name_id,
                    )

        # Collect vocab gaps and persist immediately
        if result.vocab_gaps:
            gap_dicts = []
            for vg in result.vocab_gaps:
                gap_dict = {
                    "source_id": vg.source_id,
                    "segment": vg.segment,
                    "token": vg.token,
                    "reason": vg.reason,
                }
                state.stats.setdefault("vocab_gaps", []).append(gap_dict)
                gap_dicts.append(gap_dict)

            # Stamp the compose-time token-reuse adjudication: gaps re-emitted
            # after being shown a near-synonym registered token are
            # distinct_confirmed; the rest are unchecked.
            _worker_dependency("_stamp_dedup_decision")(gap_dicts, _token_reuse_hits)

            wlog.warning(
                "Retained %d legacy compose vocabulary gaps in memory only; "
                "graph persistence requires fenced pool claims",
                len(gap_dicts),
            )

        # Auto-detect novel physical_base tokens in composed candidates.
        # These are surfaced as VocabGap nodes for ISN review without
        # requiring the LLM to emit explicit vocab_gap exits.
        if candidates:
            auto_gaps = _worker_dependency("_auto_detect_physical_base_gaps")(
                candidates
            )
            if auto_gaps:
                # Dedupe against LLM-emitted gaps (by source_id+segment+token)
                existing_keys = {
                    (g["source_id"], g["segment"], g["token"])
                    for g in state.stats.get("vocab_gaps", [])
                }
                novel = [
                    g
                    for g in auto_gaps
                    if (g["source_id"], g["segment"], g["token"]) not in existing_keys
                ]
                if novel:
                    state.stats.setdefault("vocab_gaps", []).extend(novel)
                    wlog.info(
                        "Retained %d legacy physical-base gaps in memory only",
                        len(novel),
                    )
        if result.attachments:
            _worker_dependency("_process_attachments")(result.attachments, state, wlog)
            if not state.dry_run:
                _worker_dependency("_update_sources_after_attach")(
                    result.attachments, state.source, wlog
                )

        # --- GRAPH-STATE-MACHINE: persist immediately per batch ---
        # This ensures completed batches survive cost-limit cancellation.
        if candidates:
            from datetime import UTC, datetime

            from imas_codex.standard_names.graph_store.graph_catalog import (
                persist_generated_name_batch,
            )

            # Pro-rata per-candidate cost/token attribution.  One LLM
            # batch call produces N candidates; divide cost and tokens
            # so SUM(sn.llm_cost) aggregates back to the batch total and
            # callers can extract spend directly from the graph instead
            # of scraping logs.  Error-sibling candidates minted
            # deterministically below are excluded from the denominator
            # (they carry no LLM cost).
            _llm_candidates = [
                c for c in candidates if not c.get("_from_error_sibling")
            ]
            _n = max(len(_llm_candidates), 1)
            _pro_rata_cost = _total_compose_cost / _n
            _pro_rata_in = _total_tokens_in // _n
            _pro_rata_out = _total_tokens_out // _n
            _pro_rata_cache_r = _total_cache_read // _n
            _pro_rata_cache_w = _total_cache_creation // _n
            _llm_at = datetime.now(UTC).isoformat()
            for c in candidates:
                if c.get("_from_error_sibling"):
                    continue
                c["llm_cost"] = _pro_rata_cost
                c["llm_model"] = model
                c["llm_service"] = "standard-names"
                c["llm_at"] = _llm_at
                c["llm_tokens_in"] = _pro_rata_in
                c["llm_tokens_out"] = _pro_rata_out
                c["llm_tokens_cached_read"] = _pro_rata_cache_r
                c["llm_tokens_cached_write"] = _pro_rata_cache_w

            # Tag regen-path candidates so persist increments regen_count
            if state.regen:
                for c in candidates:
                    c["regen_increment"] = True

            persisted = await asyncio.to_thread(
                persist_generated_name_batch,
                candidates,
                compose_model=model,
                dd_version=batch.dd_version,
                cocos_version=batch.cocos_version,
                run_id=getattr(state.budget_manager, "run_id", None)
                if state.budget_manager
                else None,
                return_winner_ids=True,
            )
            winner_ids = persisted if isinstance(persisted, list) else []
            winner_source_ids = {
                winner.split(":", 1)[1]
                for winner in winner_ids
                if isinstance(winner, str) and ":" in winner
            }
            if not state.dry_run:
                await asyncio.to_thread(
                    _worker_dependency("_persist_dd_gap_evidence"),
                    _collected_dd_gaps,
                    winner_source_ids,
                    phase="generate_name",
                    reporter="compose",
                    observed_dd_version=batch.dd_version,
                )
                await _worker_dependency("_write_composed_name_domains")(
                    candidates, winner_source_ids
                )
            written = len(winner_ids)
            wlog.debug("Persisted %d names from batch %s", written, batch.group_key)

        # Update StandardNameSource nodes to composed status
        if candidates and not state.dry_run:
            _worker_dependency("_update_sources_after_compose")(
                candidates, state.source, wlog
            )

        wlog.info(
            "Batch %s: %d composed, %d attached, %d skipped (cost=$%.4f)",
            batch.group_key,
            len(result.candidates),
            len(result.attachments),
            len(result.skipped),
            cost,
        )
        # Stream batch completion to progress display
        attach_label = (
            f"+{len(result.attachments)} attached  " if result.attachments else ""
        )
        state.compose_stats.stream_queue.add(
            [
                {
                    "primary_text": batch.group_key,
                    "description": (
                        f"{len(result.candidates)} names  {attach_label}${cost:.3f}"
                    ),
                }
            ]
        )
        return candidates

    # --- Polling-based work distribution (42-polling-workers) ---
    # N independent workers poll an asyncio.Queue for batches. Each worker
    # reserves budget before processing; on failure the batch is re-enqueued
    # and the worker sleeps before retrying — no silent drops.  This replaces
    # the previous asyncio.gather fan-out + Semaphore pattern.
    batch_queue: asyncio.Queue = asyncio.Queue()
    for _q_batch in state.extracted:
        batch_queue.put_nowait(_q_batch)

    composed: list[dict] = []
    errors = 0
    _MAX_BUDGET_RETRIES = 5

    async def _compose_polling_worker(worker_id: int) -> None:
        nonlocal errors
        budget_retries = 0

        while not state.should_stop():
            try:
                batch = batch_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            lease_box: list[BudgetLease] = []

            try:
                candidates = await _compose_batch_body(batch, lease_box)
                if candidates is None:
                    budget_retries += 1
                    if (
                        budget_retries > _MAX_BUDGET_RETRIES
                        or state.budget_manager.exhausted()
                    ):
                        wlog.info(
                            "Worker %d: budget exhausted — stopping",
                            worker_id,
                        )
                        break
                    # Re-enqueue batch and wait for other workers to release
                    batch_queue.put_nowait(batch)
                    wlog.debug(
                        "Worker %d: budget reserve failed for %s, "
                        "re-enqueuing (retry %d/%d)",
                        worker_id,
                        batch.group_key,
                        budget_retries,
                        _MAX_BUDGET_RETRIES,
                    )
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    continue
                if candidates:
                    composed.extend(candidates)
                budget_retries = 0  # Reset on success
            except Exception as exc:
                errors += 1
                wlog.warning(
                    "Worker %d: batch %s failed: %s",
                    worker_id,
                    batch.group_key,
                    exc,
                )
            finally:
                if lease_box:
                    lease_box[0].release_unused()

    n_compose_workers = get_compose_concurrency()
    await asyncio.gather(
        *[_compose_polling_worker(i) for i in range(n_compose_workers)]
    )

    state.composed = composed
    state.compose_stats.errors = errors

    attached = state.stats.get("attachments", 0)
    wlog.info(
        "Composition complete: %d composed, %d attached, %d errors (cost=$%.4f)",
        len(composed),
        attached,
        errors,
        state.compose_stats.cost,
    )

    # --- Batch-size telemetry ---
    # Report the distribution of items per batch and the name-only mode
    # indicator so rotation summaries can compare name-only vs default
    # throughput without bespoke log scraping.
    if state.extracted:
        sizes = [len(b.items) for b in state.extracted]
        total_items_in_batches = sum(sizes)
        name_only_batches = sum(
            1 for b in state.extracted if getattr(b, "mode", "default") == "names"
        )
        singleton_count = sum(1 for s in sizes if s == 1)
        wlog.info(
            "Batch telemetry: %d batches (%d name_only), total_items=%d, "
            "mean=%.2f, min=%d, max=%d, singletons=%d (%.1f%%), cost_per_batch=$%.4f",
            len(sizes),
            name_only_batches,
            total_items_in_batches,
            total_items_in_batches / len(sizes) if sizes else 0.0,
            min(sizes) if sizes else 0,
            max(sizes) if sizes else 0,
            singleton_count,
            100.0 * singleton_count / len(sizes) if sizes else 0.0,
            state.compose_stats.cost / len(sizes) if sizes else 0.0,
        )
        state.stats["compose_batches"] = len(sizes)
        state.stats["compose_batches_name_only"] = name_only_batches
        state.stats["compose_batch_mean_size"] = (
            total_items_in_batches / len(sizes) if sizes else 0.0
        )
        state.stats["compose_batch_singleton_pct"] = (
            100.0 * singleton_count / len(sizes) if sizes else 0.0
        )

    state.stats["generate_name_count"] = len(composed)
    state.stats["compose_errors"] = errors
    state.stats["compose_cost"] = state.compose_stats.cost
    state.stats["compose_model"] = model
    state.stats["grammar_retries"] = state.grammar_retries
    state.stats["grammar_retries_succeeded"] = state.grammar_retries_succeeded
    state.stats["opus_revisions_attempted"] = state.opus_revisions_attempted
    state.stats["opus_revisions_accepted"] = state.opus_revisions_accepted

    state.compose_stats.freeze_rate()
    state.compose_phase.mark_done()


def _validate_via_isn(
    _worker_dependency,
    /,
    entry: dict,
) -> tuple[list[str], dict]:
    """Construct ISN Pydantic model and collect ALL validation issues.

    Returns:
        (issues: list[str], layer_summary: dict)

    Compose is always name-only (ADR-1): validation uses ISN's name-only
    model. This function is purely an annotator — it never rejects entries.
    Parseability is checked upstream by the grammar round-trip in
    validate_worker. This function attaches quality annotations.
    """
    from pydantic import ValidationError

    issues: list[str] = []
    summary = {
        "pydantic": {"passed": True, "error_count": 0},
        "semantic": {"issue_count": 0, "skipped": False},
        "description": {"issue_count": 0},
    }

    # Map codex dict keys to ISN model fields
    from imas_codex.standard_names.kind_derivation import to_isn_kind

    # Name-only ISN variants forbid dd_paths and physics_domain — those
    # fields only exist on the full (non-name-only) model. Compose is
    # always name-only (ADR-1), so we omit them here. DD-path provenance
    # is retained on the codex graph node via `source_paths`.
    isn_dict: dict[str, Any] = {
        "name": entry.get("id", ""),
        "kind": to_isn_kind(entry.get("kind", "scalar")),
    }
    # ISN metadata kind forbids unit field entirely
    if isn_dict["kind"] != "metadata":
        # Invariant: by the time we build the ISN dict, the unit must come
        # from DD — the dd_unit_unresolvable skip above filters invalid units
        # before this path is reachable. Defensively assert so a regression
        # surfaces in tests rather than silently pollute the catalog
        # with `unit="1"`.
        unit = entry.get("unit")
        assert unit, (
            f"ISN dict missing unit for {entry.get('id')!r}; "
            "dd_unit_unresolvable skip invariant violation"
        )
        isn_dict["unit"] = unit

    # Layer 1: Pydantic model construction (fires 18 validators)
    model = None
    try:
        from imas_standard_names.models import create_standard_name_entry

        model = create_standard_name_entry(isn_dict, name_only=True)
    except ValidationError as e:
        summary["pydantic"]["passed"] = False
        summary["pydantic"]["error_count"] = len(e.errors())
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            issues.append(f"[pydantic:{field}] {err['msg']}")
    except Exception as e:
        # Non-validation errors (import issues, etc.) — don't crash
        summary["pydantic"]["passed"] = False
        summary["pydantic"]["error_count"] = 1
        issues.append(f"[pydantic:unknown] {e}")

    # Layer 2: Semantic checks (only if model constructed)
    if model is not None:
        try:
            from imas_standard_names.validation.semantic import run_semantic_checks

            sem_issues = run_semantic_checks({isn_dict["name"]: model})
            summary["semantic"]["issue_count"] = len(sem_issues)
            issues.extend(f"[semantic] {i}" for i in sem_issues)
        except Exception as e:
            summary["semantic"]["skipped"] = True
            issues.append(f"[semantic] check failed: {e}")
    else:
        summary["semantic"]["skipped"] = True

    # Layer 2b: Structural checks (cross-entry consistency, normally run
    # at catalog-load time by ISN's YamlStore). Single-entry invocation
    # surfaces the per-name structural defects (e.g. mode-with-no-base,
    # forbidden-token use) at compose/validate time rather than waiting
    # for catalog assembly. Empty result is the expected pass case.
    summary["structural"] = {"issue_count": 0, "skipped": False}
    if model is not None:
        try:
            from imas_standard_names.validation.structural import (
                run_structural_checks,
            )

            struct_issues = run_structural_checks({isn_dict["name"]: model})
            summary["structural"]["issue_count"] = len(struct_issues)
            issues.extend(f"[structural] {i}" for i in struct_issues)
        except Exception as e:
            summary["structural"]["skipped"] = True
            issues.append(f"[structural] check failed: {e}")
    else:
        summary["structural"]["skipped"] = True

    # Layer 2c: Codex-side canonical-token + preposition checks. Runs even
    # when the ISN model fails to construct, because the violation is
    # determined from the candidate name itself rather than the model.
    from imas_codex.standard_names.audits import canonical_locus_check

    canon_issues = canonical_locus_check({"id": isn_dict.get("name", "")})
    summary["canonical"] = {"issue_count": len(canon_issues)}
    issues.extend(f"[canonical] {i}" for i in canon_issues)

    # Layer 3: Description quality
    try:
        from imas_standard_names.validation.description import validate_description

        desc_issues = validate_description(isn_dict)
        summary["description"]["issue_count"] = len(desc_issues)
        issues.extend(f"[description] {i['message']}" for i in desc_issues)
    except Exception as e:
        issues.append(f"[description] check failed: {e}")

    return issues, summary


def _is_quarantined(
    _worker_dependency, /, issues: list[str], layer_summary: dict
) -> bool:
    """Determine whether validation issues are critical (quarantine the name).

    Critical failures that make a name unusable for publication:
    - Grammar round-trip failure (``parse_error:`` prefix)
    - Pydantic validation failure (layer 1 did not pass)
    - Empty or missing description (no ``id`` or empty string)
    - Invalid kind value
    - Critical audit failures (latex_def_check, synonym_check, multi_subject_check)
    - Grammar retry exhausted
    - ISN semantic/structural ERROR-level issues (ISN's own catalog publish gate
      hard-fails on these, so a name carrying one can never be published)

    Non-critical issues (semantic WARNING/INFO hints, description quality hints,
    non-critical audits) do NOT trigger quarantine — they are advisory.
    """
    # Grammar round-trip failures are always critical
    if any(i.startswith("parse_error:") for i in issues):
        return True

    # ISN semantic/structural ERROR-level issues are catalog-blocking: ISN's
    # publish-time validator (imas_standard_names, run by the ISNC gate) hard-
    # fails on them, so a name carrying one can never be published. Mirror that
    # authority here — an ISN-invalid name (e.g. a bare geometric quantity such
    # as 'coordinate' that lacks the required object/geometry/position
    # qualifier) must not reach validation_status='valid' and slip into the
    # export set. ISN emits an explicit 'ERROR -' severity marker for hard
    # failures; WARNING/INFO-level semantic hints stay advisory.
    for i in issues:
        if (i.startswith("[semantic]") or i.startswith("[structural]")) and (
            " ERROR - " in i
        ):
            return True

    # Grammar ambiguity is also critical — the name can't be reliably parsed
    if any("grammar:ambiguity" in i for i in issues):
        return True

    # Pydantic validation failure (model construction failed)
    pydantic = layer_summary.get("pydantic", {})
    if not pydantic.get("passed", True):
        return True

    # Grammar retry exhausted
    if any("audit:grammar_retry_exhausted" in i for i in issues):
        return True

    # Critical audit failures
    from imas_codex.standard_names.audits import has_critical_audit_failure

    if has_critical_audit_failure(issues):
        return True

    return False


def validate_name_candidate(
    _worker_dependency, /, entry: dict[str, Any]
) -> tuple[list[str], dict, str]:
    """Run the full name-admission gate on a single candidate and classify it.

    This is the ONE gate every newly-minted StandardName passes before it can
    reach ``accepted`` — the same checks a pipeline-generated candidate clears:
    grammar round-trip, the ISN Pydantic/semantic/structural/canonical/
    description layers (:func:`_validate_via_isn`), and the post-generation
    audits (:func:`run_audits`).  :func:`_is_quarantined` then decides whether
    the accumulated issues are critical.

    Both the pool ``generate_name`` path (inline audit) and the legacy
    ``validate_worker`` classify names this way; the ``sn edit`` rename path
    calls this so an operator-supplied replacement name rides *exactly* the
    same gate rather than being stamped ``valid`` on a privileged path.

    Returns ``(issues, layer_summary, validation_status)`` where
    ``validation_status`` is ``"valid"`` or ``"quarantined"``.
    """
    name = entry.get("id", "")
    is_derived_parent = entry.get("origin") == "derived"
    derived_children = entry.get("children") or []
    from imas_codex.standard_names.audits import derived_parent_structural_check

    try:
        from imas_standard_names.grammar import (
            StandardName,
            compose_standard_name,
        )

        from imas_codex.standard_names.grammar_adapter import parse_canonical_name

        # Grammar round-trip validates parsability.
        parse_canonical_name(name)

        # Fields-consistency check (best-effort — never rejects here).
        fields_dict = {}
        for fk in _worker_dependency("_GRAMMAR_FIELDS"):
            val = entry.get(fk)
            if val:
                fields_dict[fk] = val
        if fields_dict:
            try:
                sn_fields = _worker_dependency("_convert_fields_to_grammar")(
                    fields_dict
                )
                if sn_fields:
                    sn = StandardName(**sn_fields)
                    compose_standard_name(sn)
            except Exception:
                pass

        # ISN three-layer validation (annotate).
        issues, layer_summary = _worker_dependency("_validate_via_isn")(entry)

        # Post-generation audits.
        try:
            from imas_codex.standard_names.audits import run_audits

            source_path = None
            source_paths = entry.get("source_paths") or []
            if source_paths:
                source_path = _worker_dependency("strip_dd_prefix")(source_paths[0])
            audit_issues = run_audits(
                candidate=entry,
                existing_sns_in_domain=None,
                source_path=source_path,
                source_cocos_type=entry.get("cocos_transformation_type"),
            )
            if audit_issues:
                issues.extend(audit_issues)
        except Exception:
            logger.debug("validate_name_candidate: audits failed for %r", name)

        if entry.get("_grammar_retry_exhausted"):
            issues.append("audit:grammar_retry_exhausted")

        # A derived parent that DOES round-trip still owes its structural
        # contract — an orphan parent that happens to parse generalises nothing.
        if is_derived_parent:
            issues.extend(derived_parent_structural_check(name, derived_children))

        status = (
            "quarantined"
            if _worker_dependency("_is_quarantined")(issues, layer_summary)
            else "valid"
        )
        return issues, layer_summary, status
    except Exception as exc:
        exc_msg = str(exc).lower()
        issues = []
        if "component" in exc_msg and "coordinate" in exc_msg:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")
        if is_derived_parent:
            issues.extend(derived_parent_structural_check(name, derived_children))
        return issues, {}, "quarantined"


async def drain_validation_backlog(
    _worker_dependency, /, batch_size: int = 50
) -> dict[str, int]:
    """Drain every unvalidated StandardName through the admission gate.

    Standalone claim→validate→mark loop over ``claim_names_for_validation``
    (any name with a description and ``validated_at`` null). Deterministic and
    LLM-free — safe to run as a maintenance pass after ``--revalidate`` clears
    stamps, without composing or reviewing anything. Returns
    ``{"validated": n, "quarantined": m}``.
    """
    from imas_codex.standard_names.graph_store.graph_catalog_reset import (
        claim_names_for_validation,
        mark_names_validated,
        release_validation_claims,
    )

    totals = {"validated": 0, "quarantined": 0}
    while True:
        token, items = await asyncio.to_thread(claim_names_for_validation, batch_size)
        if not items:
            break
        try:
            results: list[dict[str, Any]] = []
            for entry in items:
                issues, layer_summary, status = _worker_dependency(
                    "validate_name_candidate"
                )(entry)
                if status == "quarantined":
                    totals["quarantined"] += 1
                results.append(
                    {
                        "id": entry.get("id", ""),
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            totals["validated"] += marked
        except Exception:
            logger.warning(
                "drain_validation_backlog: batch failed — releasing claims",
                exc_info=True,
            )
            await asyncio.to_thread(release_validation_claims, token)
            raise
    return totals


_SCOPED_VALIDATION_CLAIM_TIMEOUT = "PT300S"


def claim_ids_for_validation(
    _worker_dependency, /, ids: Sequence[str], limit: int
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim named StandardNames for a scoped ISN re-validation.

    The id-scoped counterpart of ``claim_names_for_validation``: only nodes in
    *ids* whose ``validated_at`` is null are claimed, so a campaign re-stamp
    never touches names outside its batch. Returns ``(token, items)`` where the
    token must be passed to ``mark_names_validated`` or
    ``release_validation_claims``.
    """
    import uuid

    from imas_codex.graph.client import GraphClient

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IN $ids
              AND sn.description IS NOT NULL
              AND sn.validated_at IS NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token, sn.updated_at = datetime()
            """,
            ids=list(ids),
            limit=limit,
            token=token,
            timeout=_worker_dependency("_SCOPED_VALIDATION_CLAIM_TIMEOUT"),
        )
        results = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(src)
            OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
            WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted', 'contested']
            RETURN sn.id AS id, sn.description AS description,
                   sn.documentation AS documentation, sn.kind AS kind,
                   sn.unit AS unit, sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.object AS object,
                   sn.physics_domain AS physics_domain,
                   sn.origin AS origin,
                   collect(DISTINCT src.id) AS source_ids,
                   collect(DISTINCT child.id) AS children
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


async def drain_validation_for_ids(
    _worker_dependency, /, ids: Sequence[str], *, batch_size: int = 50
) -> dict[str, Any]:
    """Re-run the deterministic ISN audit on an explicit id scope.

    Id-scoped counterpart of :func:`drain_validation_backlog`. Claims only the
    named nodes whose ``validated_at`` is null — a campaign clears the stamp on
    a batch before calling this — runs the shared admission gate
    (:func:`validate_name_candidate`), and re-stamps ``validation_issues`` /
    ``validation_status`` / ``validated_at``. LLM-free; a genuine defect (e.g. a
    unit inconsistency) re-quarantines instead of washing to 'valid'. Returns
    ``{"validated": n, "quarantined": m, "requarantined_ids": [...],
    "cleared_ids": [...]}``.
    """
    from imas_codex.standard_names.graph_store.graph_catalog_reset import (
        mark_names_validated,
        release_validation_claims,
    )

    ids = list(ids)
    totals: dict[str, Any] = {
        "validated": 0,
        "quarantined": 0,
        "requarantined_ids": [],
        "cleared_ids": [],
    }
    if not ids:
        return totals
    while True:
        token, items = await asyncio.to_thread(
            _worker_dependency("claim_ids_for_validation"), ids, batch_size
        )
        if not items:
            break
        try:
            results: list[dict[str, Any]] = []
            for entry in items:
                issues, layer_summary, status = _worker_dependency(
                    "validate_name_candidate"
                )(entry)
                sid = entry.get("id", "")
                if status == "quarantined":
                    totals["quarantined"] += 1
                    totals["requarantined_ids"].append(sid)
                else:
                    totals["cleared_ids"].append(sid)
                results.append(
                    {
                        "id": sid,
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            totals["validated"] += marked
        except Exception:
            logger.warning(
                "drain_validation_for_ids: batch failed — releasing claims",
                exc_info=True,
            )
            await asyncio.to_thread(release_validation_claims, token)
            raise
    return totals


async def validate_worker(
    _worker_dependency, /, state: StandardNameBuildState, **_kwargs
) -> None:
    """Validate composed names via ISN grammar checks (claim loop).

    Graph-primary: claims unvalidated StandardName nodes, runs ISN
    three-layer validation + grammar round-trip, marks results on graph.
    Follows the claim/mark/release pattern from discovery workers.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_validate_worker")

    # Initialize finalize progress (3 steps: validate, consolidate, persist)
    state.finalize_stats.total = 3
    state.finalize_stats.status_text = "validating…"

    if state.dry_run:
        wlog.info("Dry run — skipping validation")
        count = sum(len(b.items) for b in state.extracted) if state.extracted else 0
        state.validate_stats.total = count
        state.validate_stats.processed = count
        state.stats["validate_skipped"] = True
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    from imas_codex.standard_names.graph_store.graph_catalog_reset import (
        claim_names_for_validation,
        mark_names_validated,
        release_validation_claims,
    )

    _BATCH_SIZE = 50

    total_valid = 0
    total_invalid = 0
    idle_count = 0
    _MAX_IDLE = 5
    _BATCH_SIZE = 50

    wlog.info("Starting validation claim loop")

    while not state.stop_requested:
        # Claim a batch from graph
        token, items = await asyncio.to_thread(claim_names_for_validation, _BATCH_SIZE)

        if not items:
            idle_count += 1
            if idle_count >= _MAX_IDLE:
                wlog.info("No more unvalidated names — exiting")
                break
            await asyncio.sleep(2.0)
            continue
        idle_count = 0

        wlog.debug("Claimed %d names for validation (token=%s)", len(items), token[:8])

        # Process the claimed batch
        try:
            results: list[dict[str, Any]] = []
            batch_invalid = 0

            for entry in items:
                name = entry.get("id", "")
                # Single shared admission gate (see validate_name_candidate) —
                # the same classification the pool generate_name path and the
                # sn edit rename path apply, so every name is judged identically.
                issues, layer_summary, status = _worker_dependency(
                    "validate_name_candidate"
                )(entry)
                state.audits_run += 1
                if any(i.startswith("audit:") or i.startswith("[") for i in issues):
                    state.audits_failed += 1
                if status == "quarantined":
                    batch_invalid += 1
                    wlog.debug("Validation quarantined %r: %s", name, issues[:2])
                results.append(
                    {
                        "id": name,
                        "validation_issues": issues,
                        "validation_layer_summary": json.dumps(layer_summary),
                        "validation_status": status,
                    }
                )

            # Mark results on graph (token-verified)
            marked = await asyncio.to_thread(mark_names_validated, token, results)
            total_valid += marked
            total_invalid += batch_invalid
            state.validate_stats.processed += marked

            wlog.info(
                "Validated batch: %d marked, %d errors",
                marked,
                batch_invalid,
            )
            state.validate_stats.record_batch(marked)

        except Exception:
            wlog.warning("Validation batch failed — releasing claims", exc_info=True)
            await asyncio.to_thread(release_validation_claims, token)

    state.stats["validate_valid"] = total_valid
    state.stats["validate_invalid"] = total_invalid
    state.stats["audits_run"] = state.audits_run
    state.stats["audits_failed"] = state.audits_failed
    state.validate_stats.errors = total_invalid
    state.validate_stats.freeze_rate()
    state.validate_phase.mark_done()
    state.finalize_stats.processed = 1
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "validate",
                "description": f"{total_valid} valid  {total_invalid} invalid",
            }
        ]
    )


def _convert_fields_to_grammar(_worker_dependency, /, fields: dict) -> dict:
    """Convert string field values to grammar enum instances."""
    from imas_standard_names.grammar import (
        BinaryOperator,
        Component,
        GeometricBase,
        Object,
        Position,
        Process,
        Subject,
        Transformation,
    )

    enum_map = {
        "subject": Subject,
        "component": Component,
        "coordinate": Component,
        "position": Position,
        "process": Process,
        "transformation": Transformation,
        "geometric_base": GeometricBase,
        "object": Object,
        "binary_operator": BinaryOperator,
    }

    sn_fields: dict = {}
    for k, v in fields.items():
        if k == "physical_base":
            sn_fields[k] = v
        elif k in enum_map:
            sn_fields[k] = enum_map[k](v)
    return sn_fields


async def consolidate_worker(
    _worker_dependency, /, state: StandardNameBuildState, **_kwargs
) -> None:
    """Cross-batch consolidation: dedup, conflict detection, coverage accounting.

    Graph-primary: queries all validated StandardNames from graph, runs
    consolidation analysis, marks approved names with ``consolidated_at``.
    Read-only query (no claims needed — single-pass batch analysis).
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_consolidate_worker")

    state.finalize_stats.status_text = "consolidating…"

    if state.dry_run:
        wlog.info("Dry run — skipping consolidation")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    from imas_codex.standard_names.consolidation import consolidate_candidates
    from imas_codex.standard_names.graph_store.graph_catalog_reset import (
        get_validated_names,
        mark_names_consolidated,
    )

    # Always read from graph — this is the primary data source
    validated = await asyncio.to_thread(
        get_validated_names,
        ids_filter=getattr(state, "ids_filter", None),
    )

    if not validated:
        wlog.info("No validated names to consolidate — skipping")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    wlog.info("Consolidating %d validated candidates from graph", len(validated))
    state.consolidate_stats.total = len(validated)

    result = await asyncio.to_thread(consolidate_candidates, validated)

    # Mark approved names with consolidated_at on graph
    approved_ids = [e["id"] for e in result.approved if e.get("id")]
    if approved_ids:
        marked = await asyncio.to_thread(mark_names_consolidated, approved_ids)
        wlog.info("Marked %d names as consolidated", marked)

    # Log results
    wlog.info(
        "Consolidation: %d approved, %d conflicts, %d coverage gaps, %d reused",
        len(result.approved),
        len(result.conflicts),
        len(result.coverage_gaps),
        len(result.reused),
    )

    # Record stats
    state.stats["consolidation"] = result.stats
    if result.conflicts:
        for conflict in result.conflicts:
            wlog.warning(
                "Conflict: %s (%s) — %s",
                conflict.standard_name,
                conflict.conflict_type,
                conflict.details,
            )
    if result.coverage_gaps:
        wlog.info("Coverage gaps: %d unmapped source paths", len(result.coverage_gaps))

    state.consolidate_stats.processed = len(validated)
    state.consolidate_stats.freeze_rate()
    state.consolidate_phase.mark_done()
    state.finalize_stats.processed = 2
    conflicts_count = len(result.conflicts) if result.conflicts else 0
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "consolidate",
                "description": (
                    f"{len(result.approved)} names  {conflicts_count} conflicts"
                ),
            }
        ]
    )


async def persist_worker(
    _worker_dependency, /, state: StandardNameBuildState, **_kwargs
) -> None:
    """Compute embeddings for consolidated StandardNames (claim loop).

    Graph-primary: claims unembedded StandardNames, computes embeddings,
    writes results back to graph. Names are already persisted by compose —
    this worker handles the embedding enrichment pass.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_persist_worker")

    state.finalize_stats.status_text = "embedding…"

    if state.dry_run:
        wlog.info("Dry run — skipping embedding")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    from imas_codex.standard_names.graph_store.graph_catalog_reset import (
        claim_names_for_embedding,
        mark_names_embedded,
        release_embedding_claims,
    )

    total_embedded = 0
    idle_count = 0
    _MAX_IDLE = 5
    _BATCH_SIZE = 100

    wlog.info("Starting embedding claim loop")

    while not state.stop_requested:
        # Claim a batch from graph
        token, items = await asyncio.to_thread(claim_names_for_embedding, _BATCH_SIZE)

        if not items:
            idle_count += 1
            if idle_count >= _MAX_IDLE:
                wlog.info("No more names needing embedding — exiting")
                break
            await asyncio.sleep(2.0)
            continue
        idle_count = 0

        wlog.debug("Claimed %d names for embedding (token=%s)", len(items), token[:8])

        try:
            from imas_codex.embeddings.description import embed_descriptions_batch

            enriched = await asyncio.to_thread(embed_descriptions_batch, items)
            embed_batch = [
                {"id": e["id"], "embedding": e["embedding"]}
                for e in enriched
                if e.get("embedding")
            ]

            marked = await asyncio.to_thread(mark_names_embedded, token, embed_batch)
            total_embedded += marked

            wlog.info("Embedded batch: %d names", marked)
            state.persist_stats.processed += marked
            state.persist_stats.record_batch(marked)

        except Exception:
            wlog.warning("Embedding batch failed — releasing claims", exc_info=True)
            await asyncio.to_thread(release_embedding_claims, token)

    # Post-success cleanup: detach stale HAS_STANDARD_NAME for targeted paths
    # Only runs when --force/--paths regenerated names for specific paths
    if state.force and total_embedded > 0 and state.extracted:
        new_name_ids: set[str] = set()
        source_paths: set[str] = set()

        # Collect from graph — names we just embedded
        from imas_codex.graph.client import GraphClient

        def _get_recent_names():
            with GraphClient() as gc:
                results = gc.query(
                    """
                    MATCH (sn:StandardName)
                    WHERE sn.embedded_at IS NOT NULL
                      AND sn.name_stage = 'drafted'
                    RETURN sn.id AS id, sn.source_paths AS source_paths
                    """
                )
                for r in results:
                    new_name_ids.add(r["id"])
                    for p in r["source_paths"] or []:
                        source_paths.add(p)

        await asyncio.to_thread(_get_recent_names)

        if source_paths and new_name_ids:

            def _cleanup_stale():
                # Strip dd: prefix for IMASNode.id lookup
                bare_paths = [
                    _worker_dependency("strip_dd_prefix")(p) for p in source_paths
                ]
                with GraphClient() as gc:
                    result = list(
                        gc.query(
                            """
                            UNWIND $paths AS path
                            MATCH (n:IMASNode {id: path})-[r:HAS_STANDARD_NAME]->(sn:StandardName)
                            WHERE NOT (sn.id IN $keep_names)
                              AND coalesce(sn.name_stage, 'drafted') = 'drafted'
                            DELETE r
                            RETURN count(r) AS detached
                            """,
                            paths=bare_paths,
                            keep_names=list(new_name_ids),
                        )
                    )
                    return result[0]["detached"] if result else 0

            detached = await asyncio.to_thread(_cleanup_stale)
            if detached:
                wlog.info("Cleaned %d stale HAS_STANDARD_NAME relationships", detached)
                state.stats["stale_detached"] = detached

    state.stats["persist_embedded"] = total_embedded
    wlog.info("Persist complete: %d embedded", total_embedded)
    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()
    state.finalize_stats.processed = 3
    state.finalize_stats.status_text = "done"
    state.finalize_stats.stream_queue.add(
        [
            {
                "primary_text": "persist",
                "description": f"{total_embedded} embedded",
            }
        ]
    )
    state.finalize_stats.freeze_rate()

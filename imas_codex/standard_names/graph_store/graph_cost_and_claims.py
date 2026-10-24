"""Graph storage operations for the standard-name pipeline."""

from __future__ import annotations

from datetime import UTC
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
)

from .graph_foundation import _CLAIM_TIMEOUT_SECONDS


def _normalize_pool(phase: str, *, services: Any) -> str:
    """Map a raw phase string to a canonical pool name."""
    return services._PHASE_TO_POOL.get(phase, phase)


@retry_on_deadlock()
def record_llm_cost(
    *,
    run_id: str,
    phase: str,
    cycle: str | None = None,
    sn_ids: list[str] | None = None,
    model: str,
    cost: float,
    tokens_in: int,
    tokens_out: int,
    tokens_cached_read: int = 0,
    tokens_cached_write: int = 0,
    service: str = "standard-names",
    batch_id: str | None = None,
    overspend: float = 0.0,
    llm_at: Any | None = None,
    services: Any,
) -> str:
    """Write an atomic ``LLMCost`` node and ``FOR_RUN`` edge.

    **Idempotency contract**: ``id`` is a deterministic UUID-5 over
    ``(run_id, phase, batch_id, model, llm_at_iso, cost, tokens_in,
    tokens_out)``.  The node is written with ``CREATE`` (not MERGE).
    If a uniqueness constraint violation fires (duplicate id), the
    exception is swallowed — the previous write already succeeded.

    Returns:
        The deterministic ``id`` string.
    """
    if llm_at is None:
        llm_at = services.datetime.now(UTC)
    llm_at_iso = llm_at.isoformat() if not isinstance(llm_at, str) else llm_at
    id_seed = f"{run_id}|{phase}|{batch_id}|{model}|{llm_at_iso}|{cost}|{tokens_in}|{tokens_out}"
    spend_id = str(services.uuid.uuid5(services.uuid.NAMESPACE_DNS, id_seed))
    sn_ids_clean = list(sn_ids) if sn_ids else []
    if (
        model == "test-model"
        and services.os.environ.get("IMAS_CODEX_ALLOW_TEST_MODEL") != "1"
    ):
        services.logger.warning(
            "record_llm_cost: refusing model='test-model' (run=%s phase=%s) — fixture leak guard. Mock the write or set IMAS_CODEX_ALLOW_TEST_MODEL=1 against a test-scoped graph profile (never `codex`).",
            run_id,
            phase,
        )
        return spend_id
    cypher = "\n        CREATE (c:LLMCost {\n            id: $id,\n            run_id: $run_id,\n            phase: $phase,\n            pool: $pool,\n            event_type: $event_type,\n            cycle: $cycle,\n            standard_name_ids: $standard_name_ids,\n            batch_id: $batch_id,\n            overspend: $overspend,\n            llm_model: $llm_model,\n            llm_cost: $llm_cost,\n            llm_tokens_in: $llm_tokens_in,\n            llm_tokens_out: $llm_tokens_out,\n            llm_tokens_cached_read: $llm_tokens_cached_read,\n            llm_tokens_cached_write: $llm_tokens_cached_write,\n            llm_service: $llm_service,\n            llm_at: datetime($llm_at),\n            for_run: $run_id\n        })\n        WITH c\n        MATCH (rr:SNRun {id: $run_id})\n        MERGE (c)-[:FOR_RUN]->(rr)\n    "
    params = {
        "id": spend_id,
        "run_id": run_id,
        "phase": phase,
        "pool": services._normalize_pool(phase),
        "event_type": phase,
        "cycle": cycle,
        "standard_name_ids": sn_ids_clean,
        "batch_id": batch_id,
        "overspend": round(overspend, 6),
        "llm_model": model,
        "llm_cost": round(cost, 6),
        "llm_tokens_in": tokens_in,
        "llm_tokens_out": tokens_out,
        "llm_tokens_cached_read": tokens_cached_read,
        "llm_tokens_cached_write": tokens_cached_write,
        "llm_service": service,
        "llm_at": llm_at_iso,
    }
    try:
        with services.GraphClient() as gc:
            gc.query(cypher, **params)
    except Exception as exc:
        from neo4j.exceptions import ConstraintError

        if isinstance(exc, ConstraintError):
            services.logger.debug(
                "record_llm_cost: duplicate id=%s (idempotent), skipping", spend_id
            )
        else:
            raise
    return spend_id


def aggregate_spend_for_run(run_id: str, *, services: Any) -> float:
    """Return total LLM cost for a run by summing ``LLMCost`` nodes."""
    with services.GraphClient() as gc:
        result = gc.query(
            "MATCH (c:LLMCost {run_id: $run_id}) RETURN coalesce(sum(c.llm_cost), 0.0) AS total",
            run_id=run_id,
        )
        return float(result[0]["total"]) if result else 0.0


def aggregate_spend_per_phase(run_id: str, *, services: Any) -> dict[str, float]:
    """Return ``{phase: total_cost}`` for a run."""
    with services.GraphClient() as gc:
        rows = gc.query(
            "MATCH (c:LLMCost {run_id: $run_id}) RETURN c.phase AS phase, sum(c.llm_cost) AS total ORDER BY phase",
            run_id=run_id,
        )
        return {r["phase"]: float(r["total"]) for r in rows}


def aggregate_calls_per_phase(run_id: str, *, services: Any) -> dict[str, int]:
    """Return ``{phase: call_count}`` for a run, the spend owner's own rows.

    The call count is the row count behind
    :func:`aggregate_spend_per_phase`, so a run's calls and its per-phase
    spend are read from one row set under the same exact-run-id predicate.
    """
    with services.GraphClient() as gc:
        rows = gc.query(
            "MATCH (c:LLMCost {run_id: $run_id}) RETURN c.phase AS phase, count(*) AS calls ORDER BY phase",
            run_id=run_id,
        )
        return {r["phase"]: int(r["calls"]) for r in rows}


def aggregate_spend_per_name(run_id: str, *, services: Any) -> dict[str, float]:
    """Return ``{sn_id: apportioned_cost}`` for a run.

    Per-name cost share is ``llm_cost / size(standard_name_ids)`` — each name
    in the ``standard_name_ids`` list gets an equal share. Rows with an empty
    list are skipped (for example, audit calls with no names).
    """
    with services.GraphClient() as gc:
        rows = gc.query(
            "\n            MATCH (c:LLMCost {run_id: $run_id})\n            WHERE size(c.standard_name_ids) > 0\n            UNWIND c.standard_name_ids AS sn_id\n            RETURN sn_id,\n                   sum(c.llm_cost / size(c.standard_name_ids)) AS apportioned\n            ",
            run_id=run_id,
        )
        return {r["sn_id"]: float(r["apportioned"]) for r in rows}


def update_sn_per_phase_costs(run_id: str, *, services: Any) -> int:
    """Push aggregated per-name costs into ``StandardName.llm_cost_*`` fields.

    For each ``LLMCost`` row in the run, apportions
    ``llm_cost / size(standard_name_ids)`` to each name, grouped by phase.
    Then writes the per-phase totals and overall ``llm_cost`` onto the
    ``StandardName`` node.

    Returns:
        Number of ``StandardName`` nodes updated.
    """
    with services.GraphClient() as gc:
        rows = gc.query(
            "\n            MATCH (c:LLMCost {run_id: $run_id})\n            WHERE size(c.standard_name_ids) > 0\n            UNWIND c.standard_name_ids AS sn_id\n            RETURN sn_id, c.phase AS phase,\n                   sum(c.llm_cost / size(c.standard_name_ids)) AS apportioned\n            ",
            run_id=run_id,
        )
    if not rows:
        return 0
    per_name: dict[str, dict[str, float]] = {}
    for r in rows:
        sn_id = r["sn_id"]
        phase = r["phase"]
        cost = float(r["apportioned"])
        if sn_id not in per_name:
            per_name[sn_id] = {}
        field = services._PHASE_TO_SN_COST_FIELD.get(phase)
        if field:
            per_name[sn_id][field] = per_name[sn_id].get(field, 0.0) + cost
    updated_ids: set[str] = set()
    with services.GraphClient() as gc:
        for sn_id, fields in per_name.items():
            total_cost = sum(fields.values())
            set_parts = ["sn.llm_cost = $total"]
            params: dict[str, services.Any] = {
                "sn_id": sn_id,
                "total": round(total_cost, 6),
            }
            for field_name, field_cost in fields.items():
                set_parts.append(f"sn.{field_name} = ${field_name}")
                params[field_name] = round(field_cost, 6)
            result = gc.query(
                "MATCH (sn:StandardName {id: $sn_id}) SET "
                + ", ".join(set_parts)
                + " RETURN sn.id AS id",
                **params,
            )
            if result:
                updated_ids.add(sn_id)
    return len(updated_ids)


DEFAULT_POOL_BATCH_SIZE = 25

_PIPELINE_FROZEN_NAME_STAGES = ("approved", "contested")

_MAX_COMPOSE_CLAIM_ATTEMPTS = 5


def _claim_sn_atomic(
    *,
    eligibility_where: str,
    query_params: dict[str, Any],
    batch_size: int,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    extra_return_fields: str = "",
    stage_field: str | None = None,
    to_stage: str | None = None,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    restage_review_axis: str | None = None,
    seed_extra_where: str = "",
    seed_with_extras: str = "",
    seed_order_by: str = "rand()",
    services: Any,
) -> list[dict[str, Any]]:
    """Single-transaction stage-aware claim for StandardName nodes.

    All steps — seed, expand, and read-back — execute inside **one** Neo4j
    transaction.  If any step fails, the entire transaction rolls back and
    no partial claim state leaks into the graph.

    Callers should wrap this with ``@retry_on_deadlock()`` so that a
    ``TransientError`` (deadlock) causes the whole function — including
    token generation — to retry cleanly.

    Parameters
    ----------
    eligibility_where:
        Cypher WHERE fragment applied to ``sn`` (must NOT include ``WHERE``
        keyword itself).  May reference parameters in *query_params* via
        ``$name`` syntax.
    query_params:
        Extra Cypher parameters consumed by *eligibility_where*.
    batch_size:
        Maximum batch size (seed + expand).
    timeout_seconds:
        Stale-claim recovery window.
    extra_return_fields:
        Additional ``RETURN`` columns appended after the common set.
        Must start with a comma if non-empty, e.g.
        ``", sn.enriched_at AS enriched_at"``.
    stage_field:
        Optional stage property to transition at claim time
        (``'name_stage'`` or ``'docs_stage'``).  When *None* (default),
        no stage transition occurs — only ``claim_token`` and
        ``claimed_at`` are written.
    to_stage:
        Target stage value for the transition (e.g. ``'refining'``).
        Ignored when *stage_field* is *None*.
    domain:
        Optional physics domain to scope the claim.  When set, only
        StandardName nodes with ``physics_domain = $domain`` are
        considered eligible.  Used by ``sn run --physics-domain`` to
        prevent concurrent domain-specific runs from competing for
        each other's items.
    edits_only:
        When ``True``, restrict eligibility to StandardName nodes carrying
        ``edit_status = 'open'`` — the pending successors of a ``sn edit``.
        Composed exactly like *scope_run_id*'s ``scope_where`` and ANDed
        alongside it; the common use is *edits_only* alone (an edit is
        already focused, so no ``--focus``/DD-path is required).
    seed_extra_where:
        Optional additional Cypher WHERE conditions injected into the seed
        step after *eligibility_where* and *domain_where*.  Must begin
        with ``AND`` if non-empty.  Used to implement ordering-specific
        eligibility filters (e.g. parent-first escape hatches).
    seed_with_extras:
        Optional extra WITH columns computed before the ORDER BY in the
        seed step.  Must begin with a comma if non-empty, e.g.
        ``", CASE WHEN ... END AS priority"``.  Used alongside
        *seed_order_by* to express multi-column ordering.
    seed_order_by:
        ORDER BY expression for the seed step.  Defaults to ``rand()``
        (uniform random selection).  Override to implement priority-based
        claim ordering, e.g. ``"priority ASC, rand()"``.

    Returns
    -------
    list[dict]
        Claimed items as plain dicts.  Empty list when no eligible seed
        exists.
    """
    token = str(services.uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"
    params: dict[str, services.Any] = {
        **query_params,
        "token": token,
        "cutoff": cutoff,
        "frozen_name_stages": list(services._PIPELINE_FROZEN_NAME_STAGES),
    }
    stage_set = ""
    if stage_field and to_stage:
        stage_set = f", sn.{stage_field} = $to_stage"
        params["to_stage"] = to_stage
    if restage_review_axis:
        if restage_review_axis not in {"name", "docs"}:
            raise ValueError("restage_review_axis must be 'name' or 'docs'")
        stage_property = f"{restage_review_axis}_stage"
        score_property = f"reviewer_score_{restage_review_axis}"
        stage_set += f", sn.{stage_property} = CASE WHEN sn.{stage_property} = 'reviewed' AND sn.{score_property} >= $min_score THEN 'drafted' ELSE sn.{stage_property} END"
    if drain_scope_id:
        stage_set += ", sn.drain_claim_scope_id = $drain_scope_id"
        params["drain_scope_id"] = drain_scope_id
    domain_where = ""
    if domain:
        domain_where = " AND sn.physics_domain = $domain"
        params["domain"] = domain
    scope_where = ""
    if scope_run_id:
        scope_where += " AND sn.run_id = $scope_run_id"
        params["scope_run_id"] = scope_run_id
    if drain_scope_id:
        scope_where += " AND sn.drain_scope_id = $drain_scope_id"
        params["drain_scope_id"] = drain_scope_id
    if edits_only:
        scope_where += " AND coalesce(sn.edit_status, '') = 'open'"
    with services.GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                seed_result = list(
                    tx.run(
                        f"\n                        MATCH (sn:StandardName)\n                        WHERE {eligibility_where}\n                          AND NOT (coalesce(sn.name_stage, '')\n                                   IN $frozen_name_stages)\n                          AND (sn.claimed_at IS NULL\n                               OR sn.claimed_at < datetime()\n                                    - duration($cutoff))\n                          {domain_where}\n                          {scope_where}\n                          {seed_extra_where}\n                        WITH sn{seed_with_extras}\n                        ORDER BY {seed_order_by} LIMIT 1\n                        SET sn.updated_at = datetime(),\n                            sn.claimed_at = datetime(),\n                            sn.claim_token = $token,\n                            sn.claim_seq = coalesce(sn.claim_seq, 0) + 1\n                            {stage_set}\n                        WITH sn\n                        OPTIONAL MATCH (sn)-[:IN_CLUSTER]\n                            ->(c:IMASSemanticCluster)\n                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)\n                        RETURN c.id AS _cluster_id, u.id AS _unit,\n                               sn.physics_domain AS _physics_domain\n                        ",
                        **params,
                    )
                )
                if not seed_result:
                    tx.close()
                    return []
                seed = dict(seed_result[0])
                cluster_id = seed.get("_cluster_id")
                unit = seed.get("_unit")
                physics_domain = services._scalar_domain(seed.get("_physics_domain"))
                expand_limit = batch_size - 1
                if expand_limit > 0:
                    expand_params: dict[str, services.Any] = {
                        **params,
                        "expand_limit": expand_limit,
                    }
                    if cluster_id is not None and unit is not None:
                        expand_params.update(cluster_id=cluster_id, unit=unit)
                        tx.run(
                            f"\n                            MATCH (sn:StandardName)\n                            WHERE {eligibility_where}\n                              AND NOT (coalesce(sn.name_stage, '')\n                                       IN $frozen_name_stages)\n                              AND sn.claimed_at IS NULL\n                              {scope_where}\n                            MATCH (sn)-[:IN_CLUSTER]\n                                ->(:IMASSemanticCluster\n                                    {{id: $cluster_id}})\n                            MATCH (sn)-[:HAS_UNIT]\n                                ->(:Unit {{id: $unit}})\n                            WITH sn LIMIT $expand_limit\n                            SET sn.updated_at = datetime(),\n                                sn.claimed_at = datetime(),\n                                sn.claim_token = $token,\n                                sn.claim_seq = coalesce(sn.claim_seq, 0) + 1\n                                {stage_set}\n                            ",
                            **expand_params,
                        )
                    elif cluster_id is not None:
                        expand_params["cluster_id"] = cluster_id
                        tx.run(
                            f"\n                            MATCH (sn:StandardName)\n                            WHERE {eligibility_where}\n                              AND NOT (coalesce(sn.name_stage, '')\n                                       IN $frozen_name_stages)\n                              AND sn.claimed_at IS NULL\n                              {scope_where}\n                            MATCH (sn)-[:IN_CLUSTER]\n                                ->(:IMASSemanticCluster\n                                    {{id: $cluster_id}})\n                            WITH sn LIMIT $expand_limit\n                            SET sn.updated_at = datetime(),\n                                sn.claimed_at = datetime(),\n                                sn.claim_token = $token,\n                                sn.claim_seq = coalesce(sn.claim_seq, 0) + 1\n                                {stage_set}\n                            ",
                            **expand_params,
                        )
                    elif physics_domain is not None and unit is not None:
                        expand_params.update(fallback_domain=physics_domain, unit=unit)
                        tx.run(
                            f"\n                            MATCH (sn:StandardName)\n                            WHERE {eligibility_where}\n                              AND NOT (coalesce(sn.name_stage, '')\n                                       IN $frozen_name_stages)\n                              AND sn.claimed_at IS NULL\n                              {scope_where}\n                              AND sn.physics_domain = $fallback_domain\n                            MATCH (sn)-[:HAS_UNIT]\n                                ->(:Unit {{id: $unit}})\n                            WITH sn LIMIT $expand_limit\n                            SET sn.updated_at = datetime(),\n                                sn.claimed_at = datetime(),\n                                sn.claim_token = $token,\n                                sn.claim_seq = coalesce(sn.claim_seq, 0) + 1\n                                {stage_set}\n                            ",
                            **expand_params,
                        )
                    elif physics_domain is not None:
                        expand_params["fallback_domain"] = physics_domain
                        tx.run(
                            f"\n                            MATCH (sn:StandardName)\n                            WHERE {eligibility_where}\n                              AND NOT (coalesce(sn.name_stage, '')\n                                       IN $frozen_name_stages)\n                              AND sn.claimed_at IS NULL\n                              {scope_where}\n                              AND sn.physics_domain = $fallback_domain\n                            WITH sn LIMIT $expand_limit\n                            SET sn.updated_at = datetime(),\n                                sn.claimed_at = datetime(),\n                                sn.claim_token = $token,\n                                sn.claim_seq = coalesce(sn.claim_seq, 0) + 1\n                                {stage_set}\n                            ",
                            **expand_params,
                        )
                results = list(
                    tx.run(
                        f"\n                        MATCH (sn:StandardName {{claim_token: $token}})\n                        WHERE $drain_scope_id IS NULL\n                           OR sn.drain_scope_id = $drain_scope_id\n                        WITH sn\n                        WHERE NOT (coalesce(sn.name_stage, '')\n                                   IN $frozen_name_stages)\n                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)\n                        OPTIONAL MATCH (sn)-[:IN_CLUSTER]\n                            ->(c:IMASSemanticCluster)\n                        RETURN sn.id AS id,\n                               sn.description AS description,\n                               sn.documentation AS documentation,\n                               sn.kind AS kind,\n                               coalesce(u.id, sn.unit) AS unit,\n                               c.id AS cluster_id,\n                               sn.physics_domain AS physics_domain,\n                               sn.validation_status\n                                   AS validation_status,\n                               sn.claim_token AS claim_token,\n                               sn.claim_seq AS claim_seq\n                               {extra_return_fields}\n                        ",
                        token=token,
                        drain_scope_id=drain_scope_id,
                        frozen_name_stages=list(services._PIPELINE_FROZEN_NAME_STAGES),
                    )
                )
                items = [dict(r) for r in results]
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    services.logger.debug(
        "_claim_sn_atomic: claimed %d (token=%s)", len(items), token[:8]
    )
    return items


@retry_on_deadlock()
def claim_generate_name_batch(
    facility: str | None = None,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim StandardNameSource nodes (status='extracted') for name generation.

    Seed-and-expand: one random seed is claimed, then up to
    ``batch_size - 1`` additional sources sharing the same
    ``(cluster_id, unit)`` (via ``FROM_DD_PATH``→``IN_CLUSTER`` /
    ``HAS_UNIT``).  When no cluster exists the fallback key is
    ``(physics_domain, unit)``.

    All three steps (seed, expand, read-back) execute inside a **single**
    Neo4j transaction so that no partial claim state leaks on deadlock
    retry. Each selected source takes a write lock through a transient
    property before claim eligibility is re-checked; only the transaction
    that still sees an eligible source increments its durable counters.

    Parameters
    ----------
    facility:
        Optional facility id to restrict claims to signal-backed sources
        from that facility.  ``None`` means all sources.
    batch_size:
        Maximum batch size including the seed.
    timeout_seconds:
        Stale-claim recovery window.
    domain:
        Optional physics domain to restrict claims to DD-backed sources
        whose linked IMASNode has ``physics_domain = domain``.  When set,
        the seed step only claims items from that domain, so concurrent
        ``--physics-domain`` runs do not compete for each other's seeds.

    Returns
    -------
    list[dict]
        Claimed sources as dicts with keys ``id``, ``source_id``,
        ``source_type``, ``batch_key``, ``description``.
    """
    token = str(services.uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"
    facility_where = ""
    facility_where_sns2 = ""
    extra_params: dict[str, services.Any] = {}
    extra_params["frozen_name_stages"] = list(services._PIPELINE_FROZEN_NAME_STAGES)
    if facility:
        facility_where = "AND (sns.source_type = 'dd' OR EXISTS {  MATCH (sns)-[:FROM_SIGNAL]->(:FacilitySignal)    -[:AT_FACILITY]->(:Facility {id: $facility})})"
        facility_where_sns2 = "AND (sns2.source_type = 'dd' OR EXISTS {  MATCH (sns2)-[:FROM_SIGNAL]->(:FacilitySignal)    -[:AT_FACILITY]->(:Facility {id: $facility})})"
        extra_params["facility"] = facility
    domain_where = ""
    if domain:
        domain_where = "AND (NOT (sns.source_type = 'dd') OR EXISTS {  MATCH (sns)-[:FROM_DD_PATH]->(n:IMASNode)  WHERE n.physics_domain = $domain})"
        extra_params["domain"] = domain
    scope_sns_where = ""
    scope_sns2_where = ""
    if scope_run_id:
        scope_sns_where += " AND sns.run_id = $scope_run_id"
        scope_sns2_where += " AND sns2.run_id = $scope_run_id"
        extra_params["scope_run_id"] = scope_run_id
    if drain_scope_id:
        scope_sns_where += " AND sns.drain_scope_id = $drain_scope_id AND sns.drain_scope_actionable = true"
        scope_sns2_where += " AND sns2.drain_scope_id = $drain_scope_id AND sns2.drain_scope_actionable = true"
        extra_params["drain_scope_id"] = drain_scope_id
    claim_owner_set = (
        ", sns.drain_claim_scope_id = $drain_scope_id" if drain_scope_id else ""
    )
    if edits_only:
        scope_sns_where += " AND coalesce(sns.edit_status, '') = 'open'"
        scope_sns2_where += " AND coalesce(sns2.edit_status, '') = 'open'"
    with services.GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                seed_result = list(
                    tx.run(
                        f"\n                        MATCH (sns:StandardNameSource)\n                        WHERE sns.status = 'extracted'\n                          AND NOT EXISTS {{\n                            MATCH (sns)-[:PRODUCED_NAME]->(frozen:StandardName)\n                            WHERE frozen.name_stage IN $frozen_name_stages\n                          }}\n                          AND coalesce(sns.attempt_count, 0) < $max_attempts\n                          AND (sns.claimed_at IS NULL\n                               OR sns.claimed_at < datetime()\n                                    - duration($cutoff))\n                          {facility_where}\n                          {domain_where}\n                          {scope_sns_where}\n                        MATCH (sns)-[:FROM_DD_PATH]->(imas0:IMASNode)\n                        WITH DISTINCT coalesce(imas0.physics_domain, 'general') AS pd\n                        WITH pd ORDER BY rand() LIMIT 1\n                        MATCH (sns2:StandardNameSource)\n                              -[:FROM_DD_PATH]->(imas2:IMASNode)\n                        WHERE sns2.status = 'extracted'\n                          AND NOT EXISTS {{\n                            MATCH (sns2)-[:PRODUCED_NAME]->(frozen:StandardName)\n                            WHERE frozen.name_stage IN $frozen_name_stages\n                          }}\n                          AND coalesce(sns2.attempt_count, 0) < $max_attempts\n                          AND (sns2.claimed_at IS NULL\n                               OR sns2.claimed_at < datetime()\n                                    - duration($cutoff))\n                          AND coalesce(imas2.physics_domain, 'general') = pd\n                          {facility_where_sns2}\n                          {scope_sns2_where}\n                        WITH sns2 ORDER BY rand() LIMIT 1\n                        SET sns2._claim_lock = true\n                        REMOVE sns2._claim_lock\n                        WITH sns2\n                        WHERE sns2.status = 'extracted'\n                          AND NOT EXISTS {{\n                            MATCH (sns2)-[:PRODUCED_NAME]->(frozen:StandardName)\n                            WHERE frozen.name_stage IN $frozen_name_stages\n                          }}\n                          AND coalesce(sns2.attempt_count, 0) < $max_attempts\n                          AND (sns2.claimed_at IS NULL\n                               OR sns2.claimed_at < datetime()\n                                    - duration($cutoff))\n                        SET sns2.claimed_at = datetime(),\n                            sns2.claim_token = $token,\n                            sns2.claim_seq = coalesce(sns2.claim_seq, 0) + 1,\n                            sns2.attempt_count = coalesce(sns2.attempt_count, 0) + 1\n                            {claim_owner_set.replace('sns.', 'sns2.')}\n                        WITH sns2 AS sns\n                        OPTIONAL MATCH (sns)-[:FROM_DD_PATH]\n                            ->(imas:IMASNode)\n                        OPTIONAL MATCH (imas)-[:IN_CLUSTER]\n                            ->(c:IMASSemanticCluster)\n                        OPTIONAL MATCH (imas)-[:HAS_UNIT]->(u:Unit)\n                        OPTIONAL MATCH (sns)-[:FROM_SIGNAL]\n                            ->(sig:FacilitySignal)\n                        RETURN c.id AS _cluster_id,\n                               CASE WHEN u IS NOT NULL THEN u.id\n                                    WHEN sig IS NOT NULL\n                                    THEN sig.unit\n                                    ELSE null END AS _unit,\n                               CASE WHEN imas IS NOT NULL\n                                    THEN imas.physics_domain\n                                    WHEN sig IS NOT NULL\n                                    THEN sig.physics_domain\n                                    ELSE null END\n                                        AS _physics_domain,\n                               sns.batch_key AS _batch_key\n                        ",
                        token=token,
                        cutoff=cutoff,
                        max_attempts=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
                        **extra_params,
                    )
                )
                if not seed_result:
                    tx.close()
                    return []
                seed = dict(seed_result[0])
                cluster_id = seed.get("_cluster_id")
                unit = seed.get("_unit")
                physics_domain = seed.get("_physics_domain")
                batch_key = seed.get("_batch_key")
                expand_limit = batch_size - 1
                if expand_limit > 0:
                    expanded = False
                    if cluster_id is not None and unit is not None:
                        tx.run(
                            f"\n                            MATCH (sns:StandardNameSource)\n                            WHERE sns.status = 'extracted'\n                              AND NOT EXISTS {{\n                                MATCH (sns)-[:PRODUCED_NAME]->(frozen:StandardName)\n                                WHERE frozen.name_stage IN $frozen_name_stages\n                              }}\n                              AND coalesce(sns.attempt_count, 0) < $max_attempts\n                              AND sns.claimed_at IS NULL\n                              {facility_where}\n                              {scope_sns_where}\n                            MATCH (sns)-[:FROM_DD_PATH]\n                                ->(imas:IMASNode)\n                            MATCH (imas)-[:IN_CLUSTER]\n                                ->(:IMASSemanticCluster\n                                    {{id: $cluster_id}})\n                            MATCH (imas)-[:HAS_UNIT]\n                                ->(:Unit {{id: $unit}})\n                            WITH sns ORDER BY rand() LIMIT $expand_limit\n                            WITH sns ORDER BY sns.id\n                            SET sns._claim_lock = true\n                            REMOVE sns._claim_lock\n                            WITH sns\n                            WHERE sns.status = 'extracted'\n                              AND NOT EXISTS {{\n                                MATCH (sns)-[:PRODUCED_NAME]->(frozen:StandardName)\n                                WHERE frozen.name_stage IN $frozen_name_stages\n                              }}\n                              AND coalesce(sns.attempt_count, 0) < $max_attempts\n                              AND (sns.claimed_at IS NULL\n                                   OR sns.claimed_at < datetime()\n                                        - duration($cutoff))\n                            SET sns.claimed_at = datetime(),\n                                sns.claim_token = $token,\n                                sns.claim_seq = coalesce(sns.claim_seq, 0) + 1,\n                                sns.attempt_count = coalesce(sns.attempt_count, 0) + 1\n                                {claim_owner_set}\n                            ",
                            token=token,
                            cluster_id=cluster_id,
                            unit=unit,
                            expand_limit=expand_limit,
                            cutoff=cutoff,
                            max_attempts=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
                            **extra_params,
                        )
                        expanded = True
                    elif physics_domain is not None and unit is not None:
                        tx.run(
                            f"\n                            MATCH (sns:StandardNameSource)\n                            WHERE sns.status = 'extracted'\n                              AND NOT EXISTS {{\n                                MATCH (sns)-[:PRODUCED_NAME]->(frozen:StandardName)\n                                WHERE frozen.name_stage IN $frozen_name_stages\n                              }}\n                              AND coalesce(sns.attempt_count, 0) < $max_attempts\n                              AND sns.claimed_at IS NULL\n                              {facility_where}\n                              {scope_sns_where}\n                            MATCH (sns)-[:FROM_DD_PATH]\n                                ->(imas:IMASNode)\n                            WHERE imas.physics_domain\n                                = $fallback_domain\n                            MATCH (imas)-[:HAS_UNIT]\n                                ->(:Unit {{id: $unit}})\n                            WITH sns ORDER BY rand() LIMIT $expand_limit\n                            WITH sns ORDER BY sns.id\n                            SET sns._claim_lock = true\n                            REMOVE sns._claim_lock\n                            WITH sns\n                            WHERE sns.status = 'extracted'\n                              AND NOT EXISTS {{\n                                MATCH (sns)-[:PRODUCED_NAME]->(frozen:StandardName)\n                                WHERE frozen.name_stage IN $frozen_name_stages\n                              }}\n                              AND coalesce(sns.attempt_count, 0) < $max_attempts\n                              AND (sns.claimed_at IS NULL\n                                   OR sns.claimed_at < datetime()\n                                        - duration($cutoff))\n                            SET sns.claimed_at = datetime(),\n                                sns.claim_token = $token,\n                                sns.claim_seq = coalesce(sns.claim_seq, 0) + 1,\n                                sns.attempt_count = coalesce(sns.attempt_count, 0) + 1\n                                {claim_owner_set}\n                            ",
                            token=token,
                            fallback_domain=physics_domain,
                            unit=unit,
                            expand_limit=expand_limit,
                            cutoff=cutoff,
                            max_attempts=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
                            **extra_params,
                        )
                        expanded = True
                    if not expanded and batch_key:
                        tx.run(
                            f"\n                            MATCH (sns:StandardNameSource)\n                            WHERE sns.status = 'extracted'\n                              AND NOT EXISTS {{\n                                MATCH (sns)-[:PRODUCED_NAME]->(frozen:StandardName)\n                                WHERE frozen.name_stage IN $frozen_name_stages\n                              }}\n                              AND coalesce(sns.attempt_count, 0) < $max_attempts\n                              AND sns.claimed_at IS NULL\n                              AND sns.batch_key = $batch_key\n                              {facility_where}\n                              {scope_sns_where}\n                            WITH sns ORDER BY rand() LIMIT $expand_limit\n                            WITH sns ORDER BY sns.id\n                            SET sns._claim_lock = true\n                            REMOVE sns._claim_lock\n                            WITH sns\n                            WHERE sns.status = 'extracted'\n                              AND NOT EXISTS {{\n                                MATCH (sns)-[:PRODUCED_NAME]->(frozen:StandardName)\n                                WHERE frozen.name_stage IN $frozen_name_stages\n                              }}\n                              AND coalesce(sns.attempt_count, 0) < $max_attempts\n                              AND (sns.claimed_at IS NULL\n                                   OR sns.claimed_at < datetime()\n                                        - duration($cutoff))\n                            SET sns.claimed_at = datetime(),\n                                sns.claim_token = $token,\n                                sns.claim_seq = coalesce(sns.claim_seq, 0) + 1,\n                                sns.attempt_count = coalesce(sns.attempt_count, 0) + 1\n                                {claim_owner_set}\n                            ",
                            token=token,
                            batch_key=batch_key,
                            expand_limit=expand_limit,
                            cutoff=cutoff,
                            max_attempts=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
                            **extra_params,
                        )
                results = list(
                    tx.run(
                        "\n                        MATCH (sns:StandardNameSource\n                               {claim_token: $token})\n                        WHERE $drain_scope_id IS NULL\n                           OR sns.drain_scope_id = $drain_scope_id\n                        OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(im:IMASNode)\n                        OPTIONAL MATCH (sns)-[:FROM_SIGNAL]->(fs:FacilitySignal)\n                        CALL (sns) {\n                          OPTIONAL MATCH (sns)-[:PRODUCED_NAME]\n                              ->(hinted:StandardName)\n                          WHERE hinted.name_hint IS NOT NULL\n                            AND hinted.edit_reason IS NOT NULL\n                            AND hinted.edit_status = 'open'\n                          WITH hinted\n                          ORDER BY hinted.edit_requested_at DESC, hinted.id ASC\n                          RETURN head(collect(hinted.id)) AS previous_name,\n                                 head(collect(hinted.name_hint)) AS name_hint,\n                                 head(collect(hinted.edit_reason)) AS edit_reason,\n                                 head(collect(hinted.edit_origin)) AS edit_origin\n                        }\n                        RETURN sns.id AS id,\n                               sns.source_id AS source_id,\n                               sns.source_type AS source_type,\n                               sns.batch_key AS batch_key,\n                               sns.description AS description,\n                               coalesce(sns.physics_domain, im.physics_domain, fs.physics_domain) AS physics_domain,\n                               sns.claim_token AS claim_token,\n                               sns.claim_seq AS claim_seq,\n                               sns.attempt_count AS attempt_count,\n                               sns.compose_hint AS compose_hint,\n                               sns.compose_hint_reason AS compose_hint_reason,\n                               sns.compose_hint_status AS compose_hint_status,\n                               sns.compose_hint_requested_at\n                                   AS compose_hint_requested_at,\n                               sns.compose_hint_consumed_at\n                                   AS compose_hint_consumed_at,\n                               previous_name,\n                               name_hint,\n                               edit_reason,\n                               edit_origin\n                        ",
                        token=token,
                        drain_scope_id=drain_scope_id,
                    )
                )
                items = [dict(r) for r in results]
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    items = services._verify_source_claim_winners(items)
    services.logger.debug(
        "claim_generate_name_batch: claimed %d (token=%s)", len(items), token[:8]
    )
    return items


@retry_on_deadlock()
def claim_enrich_seed_and_expand(
    min_score_threshold: float = 0.0,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes that lack documentation enrichment.

    Eligibility: ``validation_status = 'valid'`` AND ``enriched_at IS NULL``.

    Returns claimed items as dicts.
    """
    where = "sn.validation_status = 'valid' AND sn.enriched_at IS NULL"
    params: dict[str, services.Any] = {}
    return services._claim_sn_atomic(
        eligibility_where=where,
        query_params=params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", sn.enriched_at AS enriched_at",
    )


@retry_on_deadlock()
def claim_review_names_seed_and_expand(
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    min_score: float = DEFAULT_MIN_SCORE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name-axis review scoring.

    Eligibility: ``reviewed_name_at IS NULL`` AND
    ``validation_status = 'valid'``.

    Review/refine exclusivity: names with
    ``reviewer_score_name < min_score`` are reserved for the regeneration pool
    and excluded here via
    ``coalesce(sn.reviewer_score_name, 1.0) >= $min_score``.
    """
    where = "sn.reviewed_name_at IS NULL AND sn.validation_status = 'valid' AND coalesce(sn.reviewer_score_name, 1.0) >= $min_score"
    return services._claim_sn_atomic(
        eligibility_where=where,
        query_params={"min_score": min_score},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", sn.reviewer_score_name AS reviewer_score_name, sn.reviewed_name_at AS reviewed_name_at",
    )


REVIEW_NAME_ELIGIBILITY_WHERE = (
    "(sn.name_stage = 'drafted' OR "
    "($drain_scope_id IS NOT NULL AND sn.name_stage = 'reviewed' "
    "AND sn.reviewer_score_name >= $min_score))"
    " AND sn.validation_status = 'valid'"
    " AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
    " AND sn.description IS NOT NULL"
    " AND sn.description <> $parent_desc_placeholder"
    " AND coalesce(sn.origin, '') <> 'derived'"
    " AND ($facility IS NULL OR sn.facility = $facility)"
)


@retry_on_deadlock()
def claim_review_name_batch(
    facility: str | None = None,
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name review.

    Eligibility: ``name_stage = 'drafted'``, ``validation_status = 'valid'``,
    and ``claimed_at IS NULL``.

    Does NOT transition stage at claim time — stage remains ``'drafted'``
    until :func:`persist_reviewed_name` writes the final outcome.  Only
    ``claim_token`` and ``claimed_at`` are set so the orphan sweep can
    recover stuck claims.

    Returns claimed items as dicts with keys:
    ``id``, ``name``, ``description``, ``documentation``, ``kind``,
    ``unit``, ``tags``, ``physics_domain``, ``chain_length``,
    ``claim_token``, ``source_paths``, and ``source_bindings``.  Each source
    binding carries the immutable DD snapshot captured on the exact
    ``StandardNameSource`` that produced the name.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    query_params: dict[str, services.Any] = {
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        "drain_scope_id": drain_scope_id,
        "min_score": min_score,
        "facility": facility,
    }
    items = services._claim_sn_atomic(
        eligibility_where=services.REVIEW_NAME_ELIGIBILITY_WHERE,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", sn.name AS name, sn.tags AS tags, coalesce(sn.chain_length, 0) AS chain_length, sn.name_stage AS name_stage, sn.origin AS origin, sn.edit_mode AS edit_mode, sn.name_hint AS name_hint, sn.docs_hint AS docs_hint, sn.edit_reason AS edit_reason, sn.edit_origin AS edit_origin, sn.physical_base AS physical_base, sn.geometry AS geometry, sn.grammar_parse_version AS grammar_parse_version, sn.source_paths AS source_paths, [(source:StandardNameSource)-[:PRODUCED_NAME]->(sn) | {     id: source.id,     source_type: source.source_type,     source_id: source.source_id,     status: source.status,     description: source.description,     physics_domain: source.physics_domain,     compose_hint: source.compose_hint,     compose_hint_reason: source.compose_hint_reason,     dd_path: source.dd_path,     dd_version: source.dd_version,     dd_documentation: source.dd_documentation,     dd_snapshot_pinned: source.dd_snapshot_pinned,     dd_parent_path: source.dd_parent_path,     dd_parent_documentation: source.dd_parent_documentation,     dd_data_type: source.dd_data_type,     dd_unit: source.dd_unit,     dd_coordinates: source.dd_coordinates,     dd_lifecycle_status: source.dd_lifecycle_status,     enhanced_description: source.enhanced_description }] AS source_bindings",
        domain=domain,
        scope_run_id=scope_run_id,
        drain_scope_id=drain_scope_id,
        edits_only=edits_only,
        restage_review_axis="name" if drain_scope_id else None,
    )
    return services._verify_name_claim_winners(items, eligible_stage="drafted")


QUORATE_RESOLUTION_METHODS: frozenset[str] = frozenset(
    {
        "quorum_consensus",
        "authoritative_escalation",
        "jev_cascade_accept",
        "jev_cascade_refine",
        "jev_cascade_chat_review",
        "jev_cascade_chat_below_reject",
    }
)

_ESCALATOR_CHAIN_SIZE = 3

"""Graph storage operations for the standard-name pipeline."""

from __future__ import annotations

from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
)

from .graph_foundation import _CLAIM_TIMEOUT_SECONDS


@retry_on_deadlock()
def claim_generate_docs_batch(
    batch_size: int = 25,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes ready for generate_docs.

    Eligibility: ``name_stage = 'accepted'`` AND ``docs_stage = 'pending'``
    AND ``claimed_at IS NULL`` (or stale).

    The claim does NOT transition ``docs_stage`` — that happens in
    :func:`persist_generated_docs` so that a failed worker leaves the node
    cleanly at ``'pending'``.

    Each claimed item is enriched with REFINED_FROM chain history via
    :func:`~imas_codex.standard_names.chain_history.name_chain_history`
    and the name-review feedback fields so the LLM understands why this
    name was accepted.

    Returns claimed items as dicts.
    """
    from imas_codex.standard_names.chain_history import name_chain_history

    score_gate = (
        ""
        if scope_run_id or edits_only
        else " AND (sn.reviewer_score_name IS NOT NULL OR (coalesce(sn.origin, '') = 'derived'     AND EXISTS { MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)       WHERE NOT coalesce(kid.name_stage, '') IN       ['superseded', 'exhausted', 'contested'] }))"
    )
    where = (
        "sn.name_stage = 'accepted' AND sn.docs_stage = 'pending' AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])"
        + score_gate
    )
    items = services._claim_sn_atomic(
        eligibility_where=where,
        query_params={},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", sn.tags AS tags, sn.reviewer_score_name AS reviewer_score_name, sn.reviewer_comments_name AS reviewer_comments_name, sn.chain_length AS chain_length, sn.docs_stage AS docs_stage, sn.name_stage AS name_stage, sn.edit_mode AS edit_mode, sn.name_hint AS name_hint, sn.docs_hint AS docs_hint, sn.edit_reason AS edit_reason, sn.edit_origin AS edit_origin",
        domain=domain,
        scope_run_id=scope_run_id,
        drain_scope_id=drain_scope_id,
        edits_only=edits_only,
        seed_extra_where="AND NOT EXISTS { MATCH (sn)-[:HAS_PARENT]->(parent:StandardName) WHERE parent.docs_stage = 'pending' AND parent.claimed_at IS NOT NULL }",
        seed_with_extras=", CASE WHEN EXISTS { ()-[:HAS_PARENT]->(sn) } THEN 0 ELSE 1 END AS _docs_priority",
        seed_order_by="_docs_priority ASC, rand()",
    )
    items = services._verify_docs_claim_winners(items, eligible_stage="pending")
    for item in items:
        item["chain_history"] = name_chain_history(item["id"])
    services.logger.debug("claim_generate_docs_batch: claimed %d", len(items))
    return items


@retry_on_deadlock()
def persist_generated_docs(
    *,
    sn_id: str,
    claim_token: str,
    description: str,
    documentation: str,
    model: str,
    run_id: str | None = None,
    services: Any,
) -> str:
    """Persist generate_docs results and transition ``docs_stage`` to ``'drafted'``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. SET ``description``, ``documentation``, ``docs_stage = 'drafted'``,
       ``docs_chain_length = 0``, ``docs_model``, ``docs_generated_at``,
       clear ``claim_token`` and ``claimed_at``.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    description:
        Short description (1–3 sentences) from the LLM.
    documentation:
        Rich markdown documentation from the LLM.
    model:
        LLM model identifier used for generation.

    Returns the new ``docs_stage`` value (``'drafted'``).
    Raises :exc:`ValueError` if token verification fails (no matching node).
    """
    description = services._sanitize_doc_text(description)
    documentation = services._sanitize_doc_text(documentation)
    with services.GraphClient() as gc:
        links = services._extract_links_from_docs(documentation, self_name=sn_id)
        link_status = services._compute_link_status(links) if links else None
        result = gc.query(
            "\n            MATCH (sn:StandardName {id: $sn_id})\n            WHERE sn.claim_token = $token\n              AND sn.name_stage = 'accepted'\n            SET sn.updated_at = datetime(), sn.description      = $description,\n                sn.documentation    = $documentation,\n                sn.docs_stage       = 'drafted',\n                sn.docs_chain_length = 0,\n                sn.docs_model       = $model,\n                sn.docs_generated_at = datetime(),\n                sn.generate_docs_count = coalesce(sn.generate_docs_count, 0) + 1,\n                sn.claim_token      = null,\n                sn.claimed_at       = null,\n                sn.links            = $links,\n                sn.link_status      = $link_status,\n                sn.embed_text_hash  = null\n            RETURN sn.docs_stage AS docs_stage\n            ",
            sn_id=sn_id,
            token=claim_token,
            description=description,
            documentation=documentation,
            model=model,
            links=links,
            link_status=link_status,
        )
        if not result:
            raise ValueError(
                f"persist_generated_docs: token mismatch or node not found for {sn_id!r}"
            )
        from imas_codex.standard_names.source_refresh import stamp_source_snapshots

        stamp_source_snapshots(sn_ids=[sn_id], gc=gc)
    new_stage: str = result[0]["docs_stage"]
    services.logger.debug(
        "persist_generated_docs: %s → docs_stage=%s", sn_id, new_stage
    )
    services.bump_sn_run_counter(run_id, "names_enriched")
    return new_stage


@retry_on_deadlock()
def release_generate_docs_claims(
    *, sn_ids: list[str], claim_token: str, services: Any
) -> int:
    """Release generate_docs claims IFF token matches.

    Called after successful persist to clear any nodes whose claim state
    was not already cleared inside :func:`persist_generated_docs`
    (e.g., skipped items in a multi-item batch).

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            "\n            UNWIND $ids AS sid\n            MATCH (n:StandardName {id: sid})\n            WHERE n.claim_token = $token\n            SET n.updated_at = datetime(), n.claimed_at = null, n.claim_token = null\n            RETURN count(n) AS released\n            ",
            ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        services.logger.debug(
            "release_generate_docs_claims: %d/%d released (token=%s) — remainder already cleared or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_generate_docs_failed_claims(
    *, sn_ids: list[str], claim_token: str, services: Any
) -> int:
    """Release generate_docs claims after LLM or processing failure.

    Token-verified: only clears ``claim_token`` and ``claimed_at`` on nodes
    where ``claim_token = $token``.  Does NOT change ``docs_stage`` (it
    stays at ``'pending'`` since the claim never transitioned it).

    Returns the count of nodes released.
    """
    if not sn_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            "\n            UNWIND $sn_ids AS sid\n            MATCH (sn:StandardName {id: sid})\n            WHERE sn.claim_token = $token\n            SET sn.updated_at = datetime(), sn.claim_token = null,\n                sn.claimed_at  = null\n            RETURN count(sn) AS released\n            ",
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        services.logger.debug(
            "release_generate_docs_failed_claims: %d/%d released (token=%s) — remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def claim_refine_docs_batch(
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    batch_size: int = 25,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for docs refinement.

    Eligibility: ``docs_stage = 'reviewed'`` AND
    ``(reviewer_score_docs < min_score OR an open docs hold)`` AND
    ``docs_chain_length < rotation_cap``.

    The claim atomically transitions ``docs_stage`` from ``'reviewed'``
    to ``'refining'`` via :func:`_claim_sn_atomic`.

    After claiming, each item is enriched with DOCS_REVISION_OF chain
    history via :func:`~imas_codex.standard_names.chain_history.docs_chain_history`.

    Returns claimed items as dicts with docs_chain_history appended.
    """
    from imas_codex.standard_names.chain_history import docs_chain_history

    score_gate = (
        ""
        if scope_run_id or edits_only
        else " AND (sn.reviewer_score_name IS NOT NULL OR (coalesce(sn.origin, '') = 'derived'     AND EXISTS { MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)       WHERE NOT coalesce(kid.name_stage, '') IN       ['superseded', 'exhausted', 'contested'] }))"
    )
    where = (
        "sn.docs_stage = 'reviewed' AND sn.reviewer_score_docs IS NOT NULL AND (sn.reviewer_score_docs < $min_score OR "
        + services.has_open_docs_hold_where()
        + f") AND coalesce(sn.docs_chain_length, 0) < $rotation_cap AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested']) AND {services.docs_review_eligibility_where()} AND sn.docs_review_quorum_shortfall IS NULL"
        + score_gate
    )
    query_params: dict[str, services.Any] = {
        "min_score": min_score,
        "rotation_cap": rotation_cap,
        **services.docs_review_eligibility_params(),
    }
    items = services._claim_sn_atomic(
        eligibility_where=where,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", sn.tags AS tags, sn.docs_stage AS docs_stage, sn.docs_chain_length AS docs_chain_length, sn.docs_model AS docs_model, sn.docs_generated_at AS docs_generated_at, sn.reviewer_score_docs AS reviewer_score_docs, sn.reviewer_comments_per_dim_docs     AS reviewer_comments_per_dim_docs, sn.reviewer_comments_docs AS reviewer_comments_docs, sn.edit_mode AS edit_mode, sn.name_hint AS name_hint, sn.docs_hint AS docs_hint, sn.edit_reason AS edit_reason, sn.edit_origin AS edit_origin",
        stage_field="docs_stage",
        to_stage="refining",
        domain=domain,
        scope_run_id=scope_run_id,
        drain_scope_id=drain_scope_id,
        edits_only=edits_only,
    )
    items = services._verify_docs_claim_winners(items, eligible_stage="refining")
    for item in items:
        item["docs_chain_history"] = docs_chain_history(item["id"], limit=5)
        raw = item.get("reviewer_comments_per_dim_docs") or "{}"
        try:
            per_dim: dict = (
                services.json.loads(raw) if isinstance(raw, str) else raw or {}
            )
        except (ValueError, TypeError):
            per_dim = {}
        current_review: dict[str, services.Any] = {
            "model": item.get("docs_model", "unknown"),
            "reviewer_score": item.get("reviewer_score_docs"),
            "reviewer_comments_per_dim": per_dim,
            "documentation": item.get("documentation", ""),
        }
        for hold in services.open_holds(item["id"], "docs"):
            per_dim[f"hold:{hold['kind']}:{hold['finding_key'][:12]}"] = hold["finding"]
        current_review["reviewer_comments_per_dim"] = per_dim
        item["prior_docs_reviews"] = item["docs_chain_history"] + [current_review]
    services.logger.debug("claim_refine_docs_batch: claimed %d", len(items))
    return items


@retry_on_deadlock()
def persist_refined_docs(
    *,
    sn_id: str,
    claim_token: str,
    description: str,
    documentation: str,
    model: str,
    current_description: str,
    current_documentation: str,
    current_model: str | None = None,
    current_generated_at: str | None = None,
    reviewer_score_to_snapshot: float | None = None,
    reviewer_comments_to_snapshot: str | None = None,
    reviewer_comments_per_dim_to_snapshot: str | None = None,
    run_id: str | None = None,
    services: Any,
) -> dict[str, Any]:
    """Persist a refined docs revision with DocsRevision snapshot.

    Single-transaction:

    1. Verify ``claim_token`` + ``docs_stage = 'refining'``.
    2. CREATE ``DocsRevision`` snapshot of CURRENT state (before this refine):
       ``id = "{sn_id}#rev-{docs_chain_length}"`` (deterministic key).
    3. CREATE ``(sn)-[:DOCS_REVISION_OF]->(rev)`` edge.
    4. SET new description/documentation on SN, advance chain, clear claim.
    5. Clear ``reviewer_*_docs`` fields on the SN (new docs need fresh review).

    Returns ``{"docs_chain_length": <new>, "revision_id": <id>}``.
    Returns ``{"docs_chain_length": -1, "revision_id": ""}`` on token/stage
    mismatch (no-op).
    """
    description = services._sanitize_doc_text(description)
    documentation = services._sanitize_doc_text(documentation)
    with services.GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                result = list(
                    tx.run(
                        "\n                        // 1. Match + verify\n                        MATCH (sn:StandardName {id: $sn_id})\n                        WHERE sn.claim_token = $token\n                          AND sn.docs_stage = 'refining'\n                          AND sn.name_stage = 'accepted'\n                        WITH sn, coalesce(sn.docs_chain_length, 0) AS cur_chain\n\n                        // 2. Create DocsRevision snapshot (deterministic id)\n                        WITH sn, cur_chain,\n                             $sn_id + '#rev-' + toString(cur_chain) AS rev_id\n                        MERGE (rev:DocsRevision {id: rev_id})\n                        ON CREATE SET\n                          rev.standard_name_id                = $sn_id,\n                          rev.revision_number                = cur_chain,\n                          rev.description                    = $cur_desc,\n                          rev.documentation                  = $cur_doc,\n                          rev.model                          = $cur_model,\n                          rev.generated_at                   = $cur_gen_at,\n                          rev.reviewer_score_docs            = $snap_score,\n                          rev.reviewer_comments_docs         = $snap_comments,\n                          rev.reviewer_comments_per_dim_docs = $snap_comments_dim,\n                          rev.created_at                     = datetime()\n\n                        // 3. Link SN → revision\n                        WITH sn, rev, cur_chain\n                        MERGE (sn)-[:DOCS_REVISION_OF]->(rev)\n\n                        // 4. Update SN with new docs + advance chain\n                        WITH sn, rev, cur_chain\n                        SET sn.updated_at = datetime(), sn.description       = $new_desc,\n                            sn.documentation     = $new_doc,\n                            sn.docs_stage        = 'drafted',\n                            sn.docs_chain_length = cur_chain + 1,\n                            sn.docs_model        = $model,\n                            sn.docs_generated_at = datetime(),\n                            sn.refine_docs_count = coalesce(sn.refine_docs_count, 0) + 1,\n                            sn.claim_token       = null,\n                            sn.claimed_at        = null,\n                            // 5. Clear reviewer_*_docs — new docs need fresh review\n                            sn.reviewer_score_docs            = null,\n                            sn.reviewer_scores_docs           = null,\n                            sn.reviewer_comments_per_dim_docs = null,\n                            sn.reviewer_comments_docs         = null,\n                            sn.reviewer_model_docs            = null,\n                            sn.reviewed_docs_at               = null\n\n                        RETURN cur_chain + 1 AS docs_chain_length,\n                               rev.id        AS revision_id\n                        ",
                        sn_id=sn_id,
                        token=claim_token,
                        cur_desc=current_description or "",
                        cur_doc=current_documentation or "",
                        cur_model=current_model,
                        cur_gen_at=current_generated_at,
                        snap_score=reviewer_score_to_snapshot,
                        snap_comments=reviewer_comments_to_snapshot,
                        snap_comments_dim=reviewer_comments_per_dim_to_snapshot,
                        new_desc=description,
                        new_doc=documentation,
                        model=model,
                    )
                )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    if result:
        row = dict(result[0])
        services.logger.debug(
            "persist_refined_docs: %s (chain_length=%d, rev=%s)",
            sn_id,
            row["docs_chain_length"],
            row["revision_id"],
        )
        with services.GraphClient() as gc:
            from imas_codex.standard_names.source_refresh import stamp_source_snapshots

            stamp_source_snapshots(sn_ids=[sn_id], gc=gc)
            gc.query(
                "\n                MATCH (sn:StandardName {id: $id})\n                SET sn.updated_at = datetime(), sn.embed_text_hash = null\n                ",
                id=sn_id,
            )
        services.bump_sn_run_counter(run_id, "names_regenerated")
        services._record_persist_outcome(run_id, "refine_docs", persisted=True)
        return row
    services._record_persist_outcome(run_id, "refine_docs", persisted=False)
    services.logger.debug(
        "persist_refined_docs: no-op for %s (token/stage mismatch)", sn_id
    )
    return {"docs_chain_length": -1, "revision_id": ""}


@retry_on_deadlock()
def reset_standard_name_docs(
    *,
    sn_ids: list[str],
    run_id: str | None = None,
    dry_run: bool = False,
    services: Any,
) -> dict[str, int]:
    """Reset the docs pipeline for an explicit set of StandardNames.

    The family-harmonization mark step: each eligible member has its
    current docs snapshotted to a :class:`DocsRevision` (same deterministic
    ``{id}#rev-{chain}`` scheme as :func:`persist_refined_docs`, advancing
    ``docs_chain_length`` so revision ids never collide), then its
    ``docs_stage`` is reset to ``'pending'`` and all reviewer-docs state is
    cleared so the generate_docs → review_docs gate re-runs from scratch.
    ``description``/``documentation`` are left in place — the generate_docs
    prompt treats them as untrusted scaffolding and rewrites them.

    Eligible: ``name_stage='accepted'`` AND ``docs_stage`` in
    accepted/exhausted/reviewed/drafted. Names already ``pending`` (or not
    in the docs pipeline) are skipped — they regenerate anyway.

    Args:
        sn_ids: Explicit StandardName ids to reset.
        run_id: When set, stamped onto ``sn.run_id`` so a subsequent
            pool run with ``scope_run_id=run_id`` claims ONLY these names
            (leaving the global docs-pending backlog untouched).
        dry_run: Report the eligible count without writing.

    Returns:
        ``{"eligible": n, "reset": n_or_0}``.
    """
    if not sn_ids:
        return {"eligible": 0, "reset": 0}
    eligibility = "\n        UNWIND $sn_ids AS sid\n        MATCH (sn:StandardName {id: sid})\n        WHERE sn.name_stage = 'accepted'\n          AND coalesce(sn.docs_stage, '')\n              IN ['accepted', 'exhausted', 'reviewed', 'drafted']\n    "
    with services.GraphClient() as gc:
        rows = gc.query(eligibility + " RETURN count(sn) AS eligible", sn_ids=sn_ids)
        eligible: int = rows[0]["eligible"] if rows else 0
        if dry_run or eligible == 0:
            return {"eligible": eligible, "reset": 0}
        result = gc.query(
            eligibility
            + "\n            WITH sn, coalesce(sn.docs_chain_length, 0) AS cur_chain,\n                 (sn.documentation IS NOT NULL AND sn.documentation <> '')\n                 AS has_docs\n\n            // The chain counter can lag stored history after an authority\n            // import. Derive the append position from both sources so an\n            // existing deterministic id is never reused.\n            OPTIONAL MATCH (prior:DocsRevision)\n            WHERE has_docs\n              AND prior.id STARTS WITH sn.id + '#rev-'\n            WITH sn, cur_chain, has_docs,\n                 max(toInteger(split(prior.id, '#rev-')[1]))\n                 AS latest_revision\n            WITH sn, cur_chain, has_docs,\n                 CASE\n                   WHEN latest_revision IS NULL OR cur_chain > latest_revision\n                   THEN cur_chain\n                   ELSE latest_revision + 1\n                 END AS revision_number\n\n            // Snapshot current docs (when present) exactly like a refine\n            // pass would, so the pre-harmonization text stays recoverable.\n            FOREACH (_ IN CASE WHEN has_docs THEN [1] ELSE [] END |\n              CREATE (rev:DocsRevision {\n                id: sn.id + '#rev-' + toString(revision_number)\n              })\n              SET\n                rev.standard_name_id                = sn.id,\n                rev.revision_number                = revision_number,\n                rev.description                    = coalesce(sn.description, ''),\n                rev.documentation                  = coalesce(sn.documentation, ''),\n                rev.model                          = sn.docs_model,\n                rev.generated_at                   = sn.docs_generated_at,\n                rev.reviewer_score_docs            = sn.reviewer_score_docs,\n                rev.reviewer_comments_docs         = sn.reviewer_comments_docs,\n                rev.reviewer_comments_per_dim_docs = sn.reviewer_comments_per_dim_docs,\n                rev.created_at                     = datetime()\n              MERGE (sn)-[:DOCS_REVISION_OF]->(rev)\n            )\n\n            SET sn.docs_stage        = 'pending',\n                sn.docs_chain_length =\n                    CASE WHEN has_docs THEN revision_number + 1 ELSE cur_chain END,\n                sn.docs_model        = null,\n                sn.docs_generated_at = null,\n                sn.claim_token       = null,\n                sn.claimed_at        = null,\n                sn.reviewer_score_docs            = null,\n                sn.reviewer_scores_docs           = null,\n                sn.reviewer_comments_per_dim_docs = null,\n                sn.reviewer_comments_docs         = null,\n                sn.reviewer_model_docs            = null,\n                sn.reviewed_docs_at               = null,\n                sn.embed_text_hash                = null,\n                sn.run_id = CASE WHEN $run_id IS NULL\n                                 THEN sn.run_id ELSE $run_id END\n            RETURN count(sn) AS reset\n            ",
            sn_ids=sn_ids,
            run_id=run_id,
        )
        if run_id:
            gc.query(
                "\n                UNWIND $sn_ids AS sid\n                MATCH (sn:StandardName {id: sid})\n                WHERE sn.name_stage = 'accepted'\n                  AND coalesce(sn.docs_stage, 'pending') = 'pending'\n                SET sn.updated_at = datetime(), sn.run_id = $run_id\n                ",
                sn_ids=sn_ids,
                run_id=run_id,
            )
    reset: int = result[0]["reset"] if result else 0
    services.logger.info(
        "reset_standard_name_docs: %d/%d reset to docs pending (run_id=%s)",
        reset,
        len(sn_ids),
        run_id,
    )
    return {"eligible": eligible, "reset": reset}


@retry_on_deadlock()
def stamp_harmonized_families(families: list[dict[str, Any]], *, services: Any) -> int:
    """Stamp the family idempotency scalars on fully docs-accepted families.

    For each family dict (``{"parent": id|None, "members": [ids],
    "signature": str}``) whose live members are ALL ``docs_stage='accepted'``,
    set ``harmonized_at`` + ``harmonized_group_signature`` on every member
    AND on the shared parent (the harmonize worklist reads the stored
    signature off the parent node). Families with any non-accepted member
    are skipped (stamp after the next rotation).

    Returns the number of families stamped.
    """
    stamped = 0
    with services.GraphClient() as gc:
        for fam in families:
            member_ids = fam.get("members") or []
            signature = fam.get("signature") or ""
            if not member_ids or not signature:
                continue
            rows = gc.query(
                "\n                UNWIND $ids AS sid\n                MATCH (sn:StandardName {id: sid})\n                WITH collect(sn) AS members,\n                     sum(CASE WHEN sn.docs_stage = 'accepted' THEN 0 ELSE 1 END)\n                     AS not_accepted\n                WHERE not_accepted = 0 AND size(members) = size($ids)\n                FOREACH (sn IN members |\n                  SET sn.updated_at = datetime(), sn.harmonized_at = datetime(),\n                      sn.harmonized_group_signature = $sig\n                )\n                RETURN size(members) AS stamped_members\n                ",
                ids=member_ids,
                sig=signature,
            )
            if rows and rows[0].get("stamped_members"):
                stamped += 1
                parent_id = fam.get("parent")
                if parent_id:
                    gc.query(
                        "\n                        MATCH (p:StandardName {id: $pid})\n                        SET p.updated_at = datetime(), p.harmonized_at = datetime(),\n                            p.harmonized_group_signature = $sig\n                        ",
                        pid=parent_id,
                        sig=signature,
                    )
    return stamped


@retry_on_deadlock()
def release_refine_docs_claims(
    *, sn_ids: list[str], claim_token: str, services: Any
) -> int:
    """Release refine-docs claims, reverting ``docs_stage`` to ``'reviewed'``.

    Token-and-stage verified: only reverts nodes where
    ``claim_token = $token AND docs_stage = 'refining'``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            "\n            UNWIND $sn_ids AS sid\n            MATCH (sn:StandardName {id: sid})\n            WHERE sn.claim_token = $token\n              AND sn.docs_stage = 'refining'\n            SET sn.updated_at = datetime(), sn.docs_stage  = 'reviewed',\n                sn.claim_token = null,\n                sn.claimed_at  = null\n            RETURN count(sn) AS released\n            ",
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        services.logger.debug(
            "release_refine_docs_claims: %d/%d released (token=%s) — remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_refine_docs_failed_claims(
    *, sn_ids: list[str], claim_token: str, services: Any
) -> int:
    """Release refine-docs claims after LLM or processing failure.

    Token-and-stage verified: only reverts nodes where
    ``claim_token = $token AND docs_stage = 'refining'``.  Prevents
    late-release from clobbering an SN already swept by orphan recovery.

    Returns the count of nodes released.
    """
    if not sn_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            "\n            UNWIND $sn_ids AS sid\n            MATCH (sn:StandardName {id: sid})\n            WHERE sn.claim_token = $token\n              AND sn.docs_stage = 'refining'\n            SET sn.updated_at = datetime(), sn.docs_stage  = 'reviewed',\n                sn.claim_token = null,\n                sn.claimed_at  = null\n            RETURN count(sn) AS released\n            ",
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        services.logger.debug(
            "release_refine_docs_failed_claims: %d/%d released (token=%s) — remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def mark_for_refine_docs(
    sn_id: str,
    *,
    desc_name_similarity: float,
    claim_token: str,
    synthetic_score: float = 0.3,
    services: Any,
) -> bool:
    """Route a derived parent to REFINE_DOCS by writing a synthetic docs-review score.

    Called from the REVIEW_NAME pre-step when ``origin='derived'`` AND
    ``desc_name_similarity < threshold``.  The node's description is
    considered misaligned with the name; we skip name scoring and instead
    push the item into the REFINE_DOCS pool.

    Mechanics:

    1. Verify ``claim_token`` so no other worker has claimed this node.
    2. Transition ``name_stage`` back to the pre-review state that allows
       the node to wait until ``name_stage='accepted'`` is later confirmed.
       Because ``seed_parent_sources`` already set ``name_stage='accepted'``
       for admitted parents, we restore that here so the node can still
       reach ``GENERATE_DOCS``.
    3. Set ``docs_stage='reviewed'`` with a synthetic ``reviewer_score_docs``
       below ``min_score`` so ``claim_refine_docs_batch`` picks it up.
    4. Store ``desc_name_similarity`` on the node for audit / reviewer context.
    5. Clear the name-review claim token.

    Returns ``True`` if the node was updated, ``False`` if the claim_token
    no longer matches (race condition — caller should skip).

    Args:
        sn_id: StandardName node id.
        desc_name_similarity: Cosine similarity value to store on the node.
        claim_token: Token set by the REVIEW_NAME claim step; used to
            confirm ownership before writing.
        synthetic_score: ``reviewer_score_docs`` value written (default
            ``0.30``, well below any ``min_score``).
    """
    with services.GraphClient() as gc:
        result = gc.query(
            "\n            MATCH (sn:StandardName {id: $id})\n            WHERE sn.claim_token = $token\n            SET sn.updated_at = datetime(), sn.desc_name_similarity  = $sim,\n                sn.docs_stage            = 'reviewed',\n                sn.reviewer_score_docs   = $synthetic_score,\n                sn.reviewer_model_docs   = '(desc_name_similarity_gate)',\n                sn.reviewer_comments_docs =\n                    '(desc-name similarity ' + toString($sim) +\n                    ' below threshold; routed to REFINE_DOCS)',\n                sn.name_stage            = 'accepted',\n                sn.claimed_at            = null,\n                sn.claim_token           = null\n            RETURN count(sn) AS updated\n            ",
            id=sn_id,
            token=claim_token,
            sim=desc_name_similarity,
            synthetic_score=synthetic_score,
        )
    updated = result[0]["updated"] if result else 0
    services.logger.debug(
        "mark_for_refine_docs: %s sim=%.3f updated=%d",
        sn_id,
        desc_name_similarity,
        updated,
    )
    return bool(updated)

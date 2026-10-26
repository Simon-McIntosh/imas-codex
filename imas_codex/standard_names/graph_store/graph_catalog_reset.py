"""Graph storage operations for the standard-name pipeline."""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock

logger = logging.getLogger(__name__)


@retry_on_deadlock()
def write_vocab_gaps(
    gaps: list[dict[str, Any]],
    source_type: str = "dd",
    *,
    skip_segment_filter: bool = False,
    gc: Any | None = None,
    services: Any,
) -> int:
    """Persist VocabGap nodes and HAS_STANDARD_NAME_VOCAB_GAP relationships.

    Each gap dict has: source_id, segment, token, reason.

    Deduplicates VocabGap nodes by id (vocab_gap:{segment}:{token}).
    Creates HAS_STANDARD_NAME_VOCAB_GAP relationships from source entities with
    per-source reason as a relationship property.

    Args:
        gaps: List of gap dicts.
        source_type: Source type ('dd' or 'signals').
        skip_segment_filter: When ``True``, bypass the segment filter.
            Used by auto-VocabGap detection — we still want to
            track novel tokens for ISN review.

    Returns the number of VocabGap nodes written.
    """
    if not gaps:
        return 0

    from datetime import UTC

    if not skip_segment_filter:
        from imas_codex.standard_names.segments import filter_closed_segment_gaps

        # Drop gaps on pseudo segments (grammar_ambiguity) — not missing tokens.
        gaps, dropped = filter_closed_segment_gaps(gaps)
        if dropped:
            from collections import Counter

            drop_hist = Counter(g.get("segment") for g in dropped)
            logger.info(
                "write_vocab_gaps: skipped %d gaps on pseudo segments (%s)",
                len(dropped),
                ", ".join(f"{seg}={n}" for seg, n in drop_hist.most_common()),
            )
        if not gaps:
            return 0

    now = services.datetime.now(UTC).isoformat()

    # Classify gaps using the ISN-backed classify_gap() function
    from imas_codex.graph.models import VocabGapDedupDecision
    from imas_codex.standard_names.segments import (
        NON_ACTIONABLE_GAP_CATEGORIES,
        classify_gap,
        is_valid_segment,
    )

    settled_dedup_decisions = [
        VocabGapDedupDecision.reuse_confirmed.value,
        VocabGapDedupDecision.distinct_confirmed.value,
        VocabGapDedupDecision.checked_no_reuse.value,
    ]

    # Build deduplicated gap nodes and relationship batch
    gap_nodes: dict[str, dict] = {}
    rel_batch: list[dict] = []

    for g in gaps:
        segment = g["segment"]
        token = g["token"]
        gap_id = f"vocab_gap:{segment}:{token}"

        if gap_id not in gap_nodes:
            # Validate segment exists in ISN grammar
            if not is_valid_segment(segment):
                logger.warning(
                    "write_vocab_gaps: skipping gap with unknown segment "
                    "'%s' for token '%s'",
                    segment,
                    token,
                )
                continue

            category, actual_segments = classify_gap(segment, token)

            # Filter non-actionable gaps — these are not genuine vocabulary
            # deficiencies but LLM mis-classifications or decomposable compounds.
            # Shared with the source-marking path so the two never drift.
            if category in NON_ACTIONABLE_GAP_CATEGORIES:
                logger.debug(
                    "write_vocab_gaps: skipping %s gap %s (token '%s', segment '%s')",
                    category,
                    gap_id,
                    token,
                    segment,
                )
                continue

            gap_nodes[gap_id] = {
                "id": gap_id,
                "segment": segment,
                "token": token,
                "example_count": 0,
                "category": category,
                "actual_segments": actual_segments,
                "nearest_token": None,
                "nearest_similarity": None,
                "dedup_decision": None,
            }
        gap_nodes[gap_id]["example_count"] += 1

        # Carry the compose-time token-reuse adjudication. A settled decision
        # from an earlier observation in this batch is monotonic.
        _decision = g.get("dedup_decision")
        _current_decision = gap_nodes[gap_id]["dedup_decision"]
        if _decision == VocabGapDedupDecision.reuse_confirmed.value or (
            _decision and _current_decision not in settled_dedup_decisions
        ):
            gap_nodes[gap_id]["dedup_decision"] = _decision
            if _decision == VocabGapDedupDecision.distinct_confirmed.value:
                gap_nodes[gap_id]["nearest_token"] = g.get("nearest_token")
                gap_nodes[gap_id]["nearest_similarity"] = g.get("nearest_similarity")

        rel_batch.append(
            {
                "gap_id": gap_id,
                "evidence_id": f"vocab-gap-evidence:{services.uuid.uuid4()}",
                "segment": segment,
                "token": token,
                "source_id": g["source_id"],
                "reason": g.get("reason", ""),
                "observed_at": now,
            }
        )

    with (
        services.nullcontext(gc)
        if gc is not None
        else services.GraphClient() as write_gc
    ):
        # MERGE VocabGap nodes — increment count, update timestamps
        write_gc.query(
            """
            UNWIND $batch AS b
            MERGE (vg:VocabGap {id: b.id})
            SET vg.segment = b.segment,
                vg.token = b.token,
                vg.example_count = coalesce(vg.example_count, 0) + b.example_count,
                vg.category = b.category,
                vg.actual_segments = b.actual_segments,
                vg.first_seen_at = coalesce(vg.first_seen_at, datetime()),
                vg.last_seen_at = datetime()
            // Settled token-reuse decisions are monotonic. A fresh observation
            // may fill an unset/unchecked decision, but cannot erase a
            // mechanical reuse verdict or a completed distinctness check.
            FOREACH (_ IN CASE
                WHEN b.dedup_decision IS NOT NULL
                     AND NOT (coalesce(vg.dedup_decision, '')
                              IN $settled_dedup_decisions)
                THEN [1] ELSE [] END |
                SET vg.dedup_decision = b.dedup_decision)
            FOREACH (_ IN CASE
                WHEN b.dedup_decision = $distinct_decision
                     AND vg.dedup_decision = $distinct_decision
                THEN [1] ELSE [] END |
                SET vg.nearest_token = b.nearest_token,
                    vg.nearest_similarity = b.nearest_similarity)
            """,
            batch=list(gap_nodes.values()),
            settled_dedup_decisions=settled_dedup_decisions,
            distinct_decision=VocabGapDedupDecision.distinct_confirmed.value,
        )

        # Preserve every observation independently of the normalized node and
        # the convenience relationship properties below.
        write_gc.query(
            """
            UNWIND $batch AS b
            MATCH (vg:VocabGap {id: b.gap_id})
            CREATE (e:VocabGapEvidence {
                id: b.evidence_id,
                source_id: b.source_id,
                vocab_gap_id: b.gap_id,
                segment: b.segment,
                token: b.token,
                source_type: $source_type,
                description: b.reason,
                reason: b.reason,
                observed_at: datetime(b.observed_at)
            })
            MERGE (vg)-[:HAS_EVIDENCE]->(e)
            SET vg.evidence = coalesce(vg.evidence, []) + b.evidence_id
            """,
            batch=rel_batch,
            source_type=source_type,
        )

        # Create HAS_STANDARD_NAME_VOCAB_GAP relationships from the underlying
        # DD path / facility signal entity (the DD-catalog view).
        entity_label = "IMASNode" if source_type == "dd" else "FacilitySignal"
        write_gc.query(
            f"""
            UNWIND $batch AS b
            MATCH (vg:VocabGap {{id: b.gap_id}})
            MATCH (src:{entity_label} {{id: b.source_id}})
            MERGE (src)-[r:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
            SET r.reason = b.reason,
                r.observed_at = datetime(b.observed_at)
            """,
            batch=rel_batch,
        )

        # And a source-first link from the StandardNameSource itself — the
        # canonical, one-hop "why is this source blocked?" edge that reconcile
        # traverses. Its id is uniform across source types ({type}:{source_id}).
        write_gc.query(
            """
            UNWIND $batch AS b
            MATCH (vg:VocabGap {id: b.gap_id})
            MATCH (sns:StandardNameSource {id: $prefix + b.source_id})
            MERGE (sns)-[r:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
            SET r.reason = b.reason,
                r.observed_at = datetime(b.observed_at)
            """,
            batch=rel_batch,
            prefix=f"{source_type}:",
        )

    written = len(gap_nodes)
    logger.info("Wrote %d VocabGap nodes from %d gap reports", written, len(gaps))
    return written


def fetch_vocab_gap_adjudications(
    gaps: list[dict[str, Any]], gc: Any | None = None, *, services: Any
) -> list[dict[str, Any]]:
    """Read active editorial decisions for the exact gaps in one compose result."""
    identities = sorted(
        {
            f"vocab_gap:{gap.get('segment')}:{gap.get('token')}"
            for gap in gaps
            if gap.get("segment") and gap.get("token") is not None
        }
    )
    if not identities:
        return []

    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            UNWIND $ids AS gap_id
            MATCH (vg:VocabGap {id: gap_id, editorial_active: true})
            RETURN vg.id AS id, vg.segment AS segment, vg.token AS token,
                   vg.editorial_disposition AS disposition,
                   vg.editorial_target AS target,
                   vg.editorial_reason AS reason,
                   vg.editorial_grammar_signature AS grammar_signature,
                   vg.editorial_grammar_version AS grammar_version
            ORDER BY vg.segment, vg.token
            """,
            ids=identities,
        )
        return [dict(row) for row in rows]
    finally:
        if own:
            client.close()


_CLAIM_TIMEOUT = "PT300S"  # 5 minutes — matches DEFAULT_CLAIM_TIMEOUT_SECONDS


@retry_on_deadlock()
def claim_names_for_validation(
    limit: int = 50, *, services: Any
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim unvalidated StandardNames for ISN validation.

    Returns ``(token, items)`` where *token* must be passed to
    ``mark_names_validated`` or ``release_validation_claims``.
    """
    import uuid

    token = str(uuid.uuid4())
    with services.GraphClient() as gc:
        # Step 1: claim with random ordering and unique token
        # Gate: any StandardName with a description that hasn't been validated.
        # No stage filter — names with NULL name_stage must also be validated.
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.description IS NOT NULL
              AND sn.validated_at IS NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.updated_at = datetime(), sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            limit=limit,
            token=token,
            timeout=_CLAIM_TIMEOUT,
        )
        # Step 2: verify — only our token
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


def mark_names_validated(
    token: str, results: list[dict[str, Any]], *, services: Any
) -> int:
    """Write validation results and release claims atomically.

    Each result dict must have ``id``, ``validation_issues`` (list[str]),
    ``validation_layer_summary`` (JSON string), and ``validation_status``
    (``"valid"`` or ``"quarantined"``).
    Token-verified: only updates nodes still claimed by this token.
    """
    if not results:
        return 0
    batch = []
    for r in results:
        batch.append(
            {
                "id": r["id"],
                "issues": r.get("validation_issues") or [],
                "summary": services._ensure_json(r.get("validation_layer_summary")),
                "validation_status": r.get("validation_status", "valid"),
            }
        )
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id, claim_token: $token})
            SET sn.updated_at = datetime(), sn.validated_at = datetime(),
                sn.validation_issues = b.issues,
                sn.validation_layer_summary = b.summary,
                sn.validation_status = b.validation_status,
                sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS marked
            """,
            batch=batch,
            token=token,
        )
        return result[0]["marked"] if result else 0


def release_validation_claims(token: str, *, services: Any) -> int:
    """Release validation claims on error. Token-verified."""
    with services.GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.updated_at = datetime(), sn.claimed_at = null, sn.claim_token = null
            RETURN count(sn) AS released
            """,
            token=token,
        )
        return result[0]["released"] if result else 0


@retry_on_deadlock()
def claim_names_for_embedding(
    limit: int = 100, *, services: Any
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim validated StandardNames needing embedding.

    Returns ``(token, items)`` with ``id`` and ``description``.
    """
    import uuid

    token = str(uuid.uuid4())
    with services.GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted', 'contested']
              AND sn.validated_at IS NOT NULL
              AND sn.embedding IS NULL
              AND sn.description IS NOT NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.updated_at = datetime(), sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            limit=limit,
            token=token,
            timeout=_CLAIM_TIMEOUT,
        )
        results = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            RETURN sn.id AS id, sn.description AS description
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


def mark_names_embedded(
    token: str, embed_batch: list[dict[str, Any]], *, services: Any
) -> int:
    """Write embeddings and release claims. Token-verified.

    Each item in *embed_batch* must have ``id`` and ``embedding``.
    """
    if not embed_batch:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id, claim_token: $token})
            SET sn.updated_at = datetime(), sn.embedding = b.embedding,
                sn.embedded_at = datetime(),
                sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS marked
            """,
            batch=[
                {"id": e["id"], "embedding": e["embedding"]}
                for e in embed_batch
                if e.get("embedding")
            ],
            token=token,
        )
        return result[0]["marked"] if result else 0


def release_embedding_claims(token: str, *, services: Any) -> int:
    """Release embedding claims on error. Token-verified."""
    with services.GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.updated_at = datetime(), sn.claimed_at = null, sn.claim_token = null
            RETURN count(sn) AS released
            """,
            token=token,
        )
        return result[0]["released"] if result else 0


def get_validated_names(
    ids_filter: str | None = None, limit: int = 2000, *, services: Any
) -> list[dict[str, Any]]:
    """Query all validated StandardNames for consolidation analysis.

    Read-only — no claims needed since consolidation is a batch analysis.
    Returns drafted names that have ``validated_at`` set and
    ``validation_status`` = ``'valid'``.
    """
    where_parts = [
        "sn.name_stage = 'drafted'",
        "sn.validated_at IS NOT NULL",
        "sn.validation_status = 'valid'",
    ]
    params: dict[str, Any] = {"limit": limit}

    if ids_filter:
        where_parts.append("ANY(p IN sn.source_paths WHERE p STARTS WITH $ids_prefix)")
        from imas_codex.standard_names.source_paths import ids_prefix_for_source_paths

        params["ids_prefix"] = ids_prefix_for_source_paths(ids_filter)

    where_clause = " AND ".join(where_parts)

    with services.GraphClient() as gc:
        results = gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {where_clause}
            OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(src)
            WITH sn, collect(DISTINCT src.id) AS source_ids
            RETURN sn.id AS id, sn.description AS description,
                   sn.documentation AS documentation, sn.kind AS kind,
                   sn.unit AS unit, sn.links AS links,
                   sn.source_paths AS source_paths,
                   source_ids
            LIMIT $limit
            """,
            **params,
        )
        return [dict(r) for r in results]


def mark_names_consolidated(name_ids: list[str], *, services: Any) -> int:
    """Mark names as consolidated (approved by cross-batch analysis)."""
    if not name_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            SET sn.updated_at = datetime(), sn.consolidated_at = datetime()
            RETURN count(sn) AS marked
            """,
            ids=name_ids,
        )
        return result[0]["marked"] if result else 0


def reset_standard_names(
    *,
    from_stage: str | None = "drafted",
    to_stage: str | None = None,
    include_accepted: bool = False,
    source_filter: str | None = None,
    ids_filter: str | None = None,
    path_allowlist: list[str] | None = None,
    dry_run: bool = False,
    since: str | None = None,
    before: str | None = None,
    below_score: float | None = None,
    tiers: list[str] | None = None,
    validation_status: str | None = None,
    services: Any,
) -> int:
    """Reset StandardName nodes to allow re-processing.

    Clears transient fields (embedding, embedded_at, model, generated_at)
    and removes HAS_STANDARD_NAME, HAS_UNIT, and
    HAS_COCOS relationships for matching nodes.

    Parameters
    ----------
    from_stage:
        Only reset nodes with this ``name_stage`` (default ``"drafted"``).
        ``None`` selects any live stage (terminal ``superseded`` /
        ``exhausted`` stages are always excluded; ``accepted`` requires
        ``include_accepted=True``).
    to_stage:
        Target ``name_stage`` after reset.  ``None`` (default) clears fields
        only without changing the stage.
    include_accepted:
        Required to reset ``name_stage='accepted'`` names — these are
        committed catalog entries and the operator must opt in explicitly.
    source_filter:
        Restrict to nodes whose ``source_types`` contains ``"dd"`` or
        ``"signals"``.
    ids_filter:
        Restrict to nodes whose HAS_STANDARD_NAME source path starts with this
        IDS name (matched via ``IMASNode -[:HAS_STANDARD_NAME]-> sn``).
    path_allowlist:
        Restrict to nodes attached to an IMASNode whose ``src.id`` is *exactly*
        in this list (matched via ``IMASNode -[:HAS_STANDARD_NAME]-> sn`` with
        ``src.id IN $path_allowlist``).  Unlike ``ids_filter`` this is an
        exact-path allowlist, not a prefix — names on any other path are left
        untouched.  Combines with ``ids_filter`` (both predicates apply).
    dry_run:
        Return the count of matching nodes without modifying anything.
    since:
        Only reset names with ``generated_at >= this`` ISO timestamp.
    before:
        Only reset names with ``generated_at < this`` ISO timestamp.
    below_score:
        Only reset names with ``reviewer_score < this`` value (0.0–1.0).
    tiers:
        Only reset names with ``review_tier`` in this list.
    validation_status:
        Only reset names with this ``validation_status`` value.

    Returns
    -------
    Number of nodes reset (or that would be reset in dry-run mode).
    """
    with services.GraphClient() as gc:
        params: dict[str, Any] = {}
        where_clauses = [
            "NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted', 'contested']"
        ]
        if from_stage is not None:
            where_clauses.append("sn.name_stage = $from_stage")
            params["from_stage"] = from_stage
        if not include_accepted and from_stage != "accepted":
            where_clauses.append("coalesce(sn.name_stage, '') <> 'accepted'")

        if source_filter:
            where_clauses.append("$source_filter IN sn.source_types")
            params["source_filter"] = source_filter

        if since:
            where_clauses.append("sn.generated_at >= datetime($since)")
            params["since"] = since
        if before:
            where_clauses.append("sn.generated_at < datetime($before)")
            params["before"] = before
        if below_score is not None:
            where_clauses.append("sn.reviewer_score_name < $below_score")
            params["below_score"] = below_score
        if tiers:
            where_clauses.append("sn.review_tier IN $tiers")
            params["tiers"] = tiers
        if validation_status:
            where_clauses.append("sn.validation_status = $validation_status")
            params["validation_status"] = validation_status

        where = " AND ".join(where_clauses)

        # Build the optional IMASNode-source scope. ids_filter is a prefix
        # match; path_allowlist is an exact-path membership test. Either (or
        # both) forces selection to go through the HAS_STANDARD_NAME join.
        src_clauses: list[str] = []
        if ids_filter:
            params["ids_prefix"] = ids_filter + "/"
            src_clauses.append("src.id STARTS WITH $ids_prefix")
        # Fail-closed: an empty allowlist selects the scoped branch and
        # matches NOTHING (``src.id IN []``). Treating ``[]`` as falsy here
        # would fall through to the unscoped branch and reset the whole graph
        # — the 1863-name-wipe shape. ``None`` means "no allowlist scope".
        if path_allowlist is not None:
            params["path_allowlist"] = list(path_allowlist)
            src_clauses.append("src.id IN $path_allowlist")
        use_src_join = bool(src_clauses)
        src_where = " AND ".join(src_clauses)

        if use_src_join:
            count_cypher = f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {where}
                AND {src_where}
                RETURN count(DISTINCT sn) AS n
            """
        else:
            count_cypher = f"""
                MATCH (sn:StandardName)
                WHERE {where}
                RETURN count(sn) AS n
            """

        result = gc.query(count_cypher, **params)
        count = result[0]["n"] if result else 0
        logger.info(
            "reset_standard_names: %d nodes match (from_stage=%s, source=%s, ids=%s)",
            count,
            from_stage,
            source_filter,
            ids_filter,
        )

        if dry_run or count == 0:
            return count

        if use_src_join:
            # Collect matching SN ids first, then operate on them. This keeps
            # every subsequent mutation scoped to exactly the names attached to
            # the in-scope source paths — names on other paths are untouched.
            collect_cypher = f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {where}
                AND {src_where}
                RETURN DISTINCT sn.id AS sn_id
            """
            rows = gc.query(collect_cypher, **params)
            sn_ids = [r["sn_id"] for r in rows]
            reset_params: dict[str, Any] = {"sn_ids": sn_ids}
            node_match = "MATCH (sn:StandardName) WHERE sn.id IN $sn_ids"
        else:
            reset_params = dict(params)
            node_match = f"MATCH (sn:StandardName) WHERE {where}"

        # Clear transient fields, optionally set a new stage
        if to_stage is not None:
            set_clause = (
                "sn.embedding = null, sn.embedded_at = null, sn.model = null, "
                "sn.generated_at = null, "
                "sn.cocos_transformation_type = null, sn.cocos = null, sn.dd_version = null, "
                "sn.name_stage = $to_stage, sn.updated_at = datetime()"
            )
            reset_params["to_stage"] = to_stage
        else:
            set_clause = (
                "sn.embedding = null, sn.embedded_at = null, sn.model = null, "
                "sn.generated_at = null, "
                "sn.cocos_transformation_type = null, sn.cocos = null, sn.dd_version = null, "
                "sn.updated_at = datetime()"
            )

        # Remove HAS_STANDARD_NAME, HAS_UNIT, and HAS_COCOS relationships and
        # clear the transient fields in a SINGLE statement so the reset is
        # atomic: a mid-sequence failure can no longer leave a node with its
        # edges stripped but its fields (or target stage) untouched, or vice
        # versa. Each DELETE runs in its own unit CALL subquery so the outer
        # cardinality stays one row per node and the SET applies exactly once.
        gc.query(
            f"""
            {node_match}
            CALL {{
                WITH sn
                OPTIONAL MATCH (sn)<-[r:HAS_STANDARD_NAME]-()
                DELETE r
            }}
            CALL {{
                WITH sn
                OPTIONAL MATCH (sn)-[r:HAS_UNIT]->()
                DELETE r
            }}
            CALL {{
                WITH sn
                OPTIONAL MATCH (sn)-[r:HAS_COCOS]->()
                DELETE r
            }}
            SET {set_clause}
            """,
            **reset_params,
        )

    logger.info("Reset %d StandardName nodes", count)
    return count


def clear_standard_names(
    *,
    stage_filter: list[str] | None = None,
    source_filter: str | None = None,
    ids_filter: str | None = None,
    path_allowlist: list[str] | None = None,
    include_accepted: bool = False,
    dry_run: bool = False,
    since: str | None = None,
    before: str | None = None,
    below_score: float | None = None,
    tiers: list[str] | None = None,
    validation_status: str | None = None,
    services: Any,
) -> int:
    """Delete StandardName nodes and their relationships.

    Safety model (relationship-first):

    1. If ``ids_filter`` or ``source_filter`` is set, delete matching
       ``HAS_STANDARD_NAME`` relationships first.
    2. Then delete ``StandardName`` nodes that have zero remaining
       ``HAS_STANDARD_NAME`` edges.

    With no ``stage_filter`` every live stage is eligible except
    ``accepted`` (requires ``include_accepted=True``); the terminal
    ``superseded`` / ``exhausted`` stages are never deleted unless
    explicitly listed (they carry REFINED_FROM chain history).

    Parameters
    ----------
    stage_filter:
        List of ``name_stage`` values to delete.  ``None`` (default)
        selects all live stages, gated by ``include_accepted``.
    source_filter:
        Restrict to nodes whose ``source_types`` contains ``"dd"`` or
        ``"signals"``.
    ids_filter:
        Delete only names linked to an IMASNode whose id starts with this IDS
        name.  Relationships are removed first; nodes become orphans and are
        then deleted.
    path_allowlist:
        Delete only names linked to an IMASNode whose ``src.id`` is *exactly*
        in this list (``src.id IN $path_allowlist``).  Unlike ``ids_filter``
        this is an exact-path allowlist, not a prefix.  Relationships are
        removed first (relationship-first delete), so names attached to any
        other path — including accepted catalog names — keep their edges and
        are never deleted.  Combines with ``ids_filter``.
    include_accepted:
        Required to delete ``name_stage='accepted'`` names — these are
        committed catalog entries and the operator must opt in explicitly.
    dry_run:
        Return the count of nodes that would be deleted without modifying
        anything.
    since:
        Only clear names with ``generated_at >= this`` ISO timestamp.
    before:
        Only clear names with ``generated_at < this`` ISO timestamp.
    below_score:
        Only clear names with ``reviewer_score < this`` value (0.0–1.0).
    tiers:
        Only clear names with ``review_tier`` in this list.
    validation_status:
        Only clear names with this ``validation_status`` value.

    Returns
    -------
    Number of nodes deleted (or that would be deleted in dry-run mode).
    """
    effective_stages = list(stage_filter) if stage_filter else []
    # include_accepted only AUTHORIZES an explicitly-listed 'accepted' stage;
    # it must NEVER inject 'accepted' into a stage list the operator did not
    # name. ``--stage drafted --include-accepted`` deletes drafted names only —
    # not accepted catalog entries. Without the opt-in, an explicitly-listed
    # 'accepted' is dropped as a protection.
    if stage_filter is not None and not include_accepted:
        effective_stages = [s for s in effective_stages if s != "accepted"]

    with services.GraphClient() as gc:
        params: dict[str, Any] = {}
        sn_where_clauses: list[str] = []
        if stage_filter is None:
            sn_where_clauses.append(
                "NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted', 'contested']"
            )
            if not include_accepted:
                sn_where_clauses.append("coalesce(sn.name_stage, '') <> 'accepted'")
        else:
            params["stages"] = effective_stages
            sn_where_clauses.append("sn.name_stage IN $stages")

        if source_filter:
            sn_where_clauses.append("$source_filter IN sn.source_types")
            params["source_filter"] = source_filter

        if since:
            sn_where_clauses.append("sn.generated_at >= datetime($since)")
            params["since"] = since
        if before:
            sn_where_clauses.append("sn.generated_at < datetime($before)")
            params["before"] = before
        if below_score is not None:
            sn_where_clauses.append("sn.reviewer_score_name < $below_score")
            params["below_score"] = below_score
        if tiers:
            sn_where_clauses.append("sn.review_tier IN $tiers")
            params["tiers"] = tiers
        if validation_status:
            sn_where_clauses.append("sn.validation_status = $validation_status")
            params["validation_status"] = validation_status

        sn_where = " AND ".join(sn_where_clauses) if sn_where_clauses else "true"

        # Build the optional IMASNode-source scope. ids_filter is a prefix
        # match; path_allowlist is an exact-path membership test. Either (or
        # both) forces the relationship-first delete path so names on other
        # paths keep their edges and survive.
        src_clauses: list[str] = []
        if ids_filter:
            params["ids_prefix"] = ids_filter + "/"
            src_clauses.append("src.id STARTS WITH $ids_prefix")
        # Fail-closed: an empty allowlist selects the scoped branch and
        # matches NOTHING (``src.id IN []``). Treating ``[]`` as falsy here
        # would fall through to the unscoped branch and delete across the
        # whole graph — the 1863-name-wipe shape. ``None`` = no allowlist.
        if path_allowlist is not None:
            params["path_allowlist"] = list(path_allowlist)
            src_clauses.append("src.id IN $path_allowlist")
        use_src_join = bool(src_clauses)
        src_where = " AND ".join(src_clauses)
        # Same scope predicate rebound to an ``o`` producer, for the
        # out-of-scope-producer guard below.
        out_scope_where = src_where.replace("src.", "o.")

        exact_reset_parent_where = """
            (
                parent.origin = 'derived'
                OR (
                    parent.origin IS NULL
                    AND parent.transformation IS NOT NULL
                )
            )
            AND coalesce(parent.name_stage, '') IN ['', 'pending']
            AND NOT (parent.id IN reset_candidate_ids)
            AND parent.claimed_at IS NULL
            AND parent.claim_token IS NULL
            AND all(key IN keys(parent)
                    WHERE key IN $reset_skeleton_parent_keys)
            AND NOT EXISTS {
                MATCH (other)-[parent_rel]-(parent)
                WHERE NOT (
                    (
                        type(parent_rel) = 'HAS_PARENT'
                        AND parent_rel.operator_kind IS NOT NULL
                        AND other.id IN reset_candidate_ids
                        AND startNode(parent_rel) = other
                        AND endNode(parent_rel) = parent
                    )
                    OR (
                        type(parent_rel) = 'PRODUCED_NAME'
                        AND other:StandardNameSource
                        AND other.id = 'derived:' + parent.id
                        AND startNode(parent_rel) = other
                        AND endNode(parent_rel) = parent
                    )
                    OR (
                        type(parent_rel) = 'HAS_PARENT'
                        AND parent_rel.operator_kind IS NOT NULL
                        AND other:StandardName
                        AND startNode(parent_rel) = parent
                        AND endNode(parent_rel) = other
                    )
                    OR (
                        type(parent_rel) = 'HAS_LOCUS'
                        AND other:Locus
                        AND startNode(parent_rel) = parent
                        AND endNode(parent_rel) = other
                        AND parent_rel.locus_token = other.id
                        AND parent_rel.locus_relation IS NOT NULL
                    )
                    OR (
                        type(parent_rel) = 'HAS_SEGMENT'
                        AND other:GrammarToken
                        AND startNode(parent_rel) = parent
                        AND endNode(parent_rel) = other
                        AND parent_rel.segment = other.segment
                        AND parent[parent_rel.segment] = other.value
                    )
                    OR (
                        type(parent_rel) IN $reset_skeleton_segment_edge_types
                        AND other:GrammarToken
                        AND startNode(parent_rel) = parent
                        AND endNode(parent_rel) = other
                        AND parent[
                            toLower(substring(type(parent_rel), 4))
                        ] = other.value
                    )
                )
            }
            AND NOT EXISTS {
                MATCH (mirror_source:StandardNameSource)
                WHERE mirror_source.produced_sn_id = parent.id
                  AND mirror_source.id <> 'derived:' + parent.id
            }
        """
        exact_reset_source_where = """
            derived_source IS NULL OR (
                derived_source.source_type = 'derived'
                AND derived_source.source_id = parent.id
                AND coalesce(derived_source.batch_key, 'derived_parent') =
                    'derived_parent'
                AND coalesce(derived_source.attempt_count, 0) = 0
                AND derived_source.claimed_at IS NULL
                AND derived_source.claim_token IS NULL
                AND derived_source.produced_sn_id = parent.id
                AND (
                    (
                        derived_source.status IS NULL
                        AND all(key IN keys(derived_source)
                                WHERE key IN
                                      $reset_skeleton_minimal_source_keys)
                    )
                    OR (
                        derived_source.status = 'composed'
                        AND derived_source.created_at IS NOT NULL
                        AND derived_source.composed_at IS NOT NULL
                        AND all(key IN keys(derived_source)
                                WHERE key IN
                                      $reset_skeleton_composed_source_keys)
                    )
                )
                AND NOT EXISTS {
                    MATCH (derived_source)-[source_rel]-()
                    WHERE NOT (
                        type(source_rel) = 'PRODUCED_NAME'
                        AND endNode(source_rel) = parent
                    )
                }
            )
        """
        if path_allowlist is not None:
            params["reset_skeleton_parent_keys"] = [
                "id",
                "origin",
                "name_stage",
                "needs_composition",
                "claimed_at",
                "claim_token",
                *services._GRAMMAR_SEGMENT_COLUMNS,
            ]
            params["reset_skeleton_segment_edge_types"] = [
                rel_type for _segment, rel_type in services._SEGMENT_EDGE_TYPES
            ]
            params["reset_skeleton_minimal_source_keys"] = [
                "id",
                "source_type",
                "source_id",
                "batch_key",
                "attempt_count",
                "produced_sn_id",
                "claimed_at",
                "claim_token",
            ]
            params["reset_skeleton_composed_source_keys"] = [
                *params["reset_skeleton_minimal_source_keys"],
                "status",
                "created_at",
                "composed_at",
            ]

        if use_src_join:
            # Count the names that will ACTUALLY be deleted, not merely the ones
            # matched in scope: a name matched via an in-scope edge SURVIVES if
            # it also has a producer outside the scope (relationship-first
            # delete keeps that edge). Counting bare in-scope matches overcounts
            # such survivors, so the reported/dry-run number would exceed the
            # real deletions. Restrict to names whose producers are ALL in
            # scope (no out-of-scope producer → truly orphaned after the edge
            # delete).
            if path_allowlist is not None:
                count_cypher = f"""
                    MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                    WHERE {sn_where}
                    AND {src_where}
                    WITH DISTINCT sn
                    WHERE NOT EXISTS {{
                        MATCH (o:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
                        WHERE NOT ({out_scope_where})
                    }}
                    WITH collect(sn) AS reset_candidates
                    WITH reset_candidates,
                         [candidate IN reset_candidates | candidate.id]
                             AS reset_candidate_ids
                    CALL {{
                        WITH reset_candidates, reset_candidate_ids
                        UNWIND reset_candidates AS candidate
                        OPTIONAL MATCH (candidate)-[candidate_parent_edge:HAS_PARENT]->
                                       (candidate_parent:StandardName)
                        WHERE candidate_parent_edge.operator_kind IS NOT NULL
                        WITH reset_candidate_ids,
                             collect(DISTINCT candidate_parent) AS candidate_parents
                        UNWIND CASE WHEN size(candidate_parents) = 0
                            THEN [null] ELSE candidate_parents END AS parent
                        WITH DISTINCT reset_candidate_ids, parent
                        WHERE parent IS NOT NULL
                          AND {exact_reset_parent_where}
                        OPTIONAL MATCH (derived_source:StandardNameSource)
                        WHERE derived_source.id = 'derived:' + parent.id
                        WITH parent, derived_source
                        WHERE {exact_reset_source_where}
                        RETURN count(DISTINCT parent) AS skeleton_count
                    }}
                    RETURN size(reset_candidates) + skeleton_count AS n,
                           size(reset_candidates) AS candidate_count,
                           skeleton_count
                """
            else:
                count_cypher = f"""
                    MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                    WHERE {sn_where}
                    AND {src_where}
                    WITH DISTINCT sn
                    WHERE NOT EXISTS {{
                        MATCH (o:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
                        WHERE NOT ({out_scope_where})
                    }}
                    RETURN count(sn) AS n
                """
        else:
            count_cypher = f"""
                MATCH (sn:StandardName)
                WHERE {sn_where}
                RETURN count(sn) AS n
            """

        result = gc.query(count_cypher, **params)
        count = result[0]["n"] if result else 0
        logger.info(
            "clear_standard_names: %d nodes match (stages=%s, source=%s, ids=%s)",
            count,
            effective_stages or "all-live",
            source_filter,
            ids_filter,
        )

        if dry_run or count == 0:
            return count

        # Delete the in-scope reviews and StandardName nodes in a SINGLE
        # statement so the delete is atomic: a mid-sequence failure can no
        # longer leave a name with its producer edge stripped but the node
        # (or its review) still present, dropping the name out of every future
        # regeneration while orphaning its review.
        #
        # DETACH DELETE on the StandardName alone would orphan its
        # StandardNameReview (HAS_REVIEW goes StandardName -> StandardNameReview),
        # so the review is deleted in the same statement inside a unit CALL
        # subquery.
        if use_src_join:
            # Relationship-first, scope-safe: remove the in-scope producer
            # edges, then (orphan-guard) delete only the names that have NO
            # remaining HAS_STANDARD_NAME edge. Neo4j inserts an Eager barrier
            # between the DELETE and the NOT EXISTS read so the guard sees
            # post-delete state — a name still attached to any out-of-scope
            # path survives with its review intact (survivor-review handling:
            # the review is deleted only for names that actually go away).
            # Verified against a live graph in
            # tests/standard_names/test_clear_atomicity_graph.py.
            deletion_clause = services.deletion_change_cypher("sn")
            deletion_params = services.deletion_change_params(
                "clear_selected_name",
                reason="selected name removed by a scoped pipeline clear",
            )
            if path_allowlist is not None:
                # An exact reset can delete the only child of a structural
                # parent that relationship projection created as a bare
                # scaffold. Carry only those former direct parents through the
                # candidate delete and retire a parent in this SAME statement
                # when it remains a null-lifecycle skeleton. The property-key,
                # relationship, claim, and source-identity fences make this a
                # cleanup of reset-created scaffolding rather than a general
                # derived-parent reaper.
                skeleton_params = services.deletion_change_params(
                    "remove_skeleton_placeholder",
                    reason=(
                        "exact source reset removed the scaffold's only "
                        "structural child"
                    ),
                )
                # The scaffold receipt is a different operation from the
                # candidate's, so it namespaces its parameters and shares the
                # statement rather than writing a bare change row.
                skeleton_clause = services.deletion_change_cypher(
                    "parent", param_prefix="reset_skeleton_"
                )
                rows = gc.query(
                    f"""
                    MATCH (src:IMASNode)-[rel:HAS_STANDARD_NAME]->(sn:StandardName)
                    WHERE {sn_where}
                    AND {src_where}
                    OPTIONAL MATCH (sn)-[candidate_parent_edge:HAS_PARENT]->
                                   (candidate_parent:StandardName)
                    WHERE candidate_parent_edge.operator_kind IS NOT NULL
                    WITH sn,
                         collect(DISTINCT rel) AS reset_relationships,
                         collect(DISTINCT candidate_parent.id)
                             AS candidate_parent_ids
                    FOREACH (rel IN reset_relationships | DELETE rel)
                    WITH sn, candidate_parent_ids
                    WHERE NOT EXISTS {{ MATCH ()-[:HAS_STANDARD_NAME]->(sn) }}
                    {deletion_clause}
                    WITH sn, change, candidate_parent_ids,
                         sn.id AS reset_candidate_id
                    CALL {{
                        WITH sn
                        OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
                        DETACH DELETE r
                    }}
                    DETACH DELETE sn
                    WITH collect(DISTINCT change) AS candidate_changes,
                         collect(DISTINCT reset_candidate_id)
                             AS reset_candidate_ids,
                         reduce(
                             all_parent_ids = [],
                             parent_ids IN collect(candidate_parent_ids) |
                                 all_parent_ids + parent_ids
                         ) AS candidate_parent_ids
                    CALL {{
                        WITH reset_candidate_ids, candidate_parent_ids
                        UNWIND candidate_parent_ids AS parent_id
                        WITH DISTINCT reset_candidate_ids, parent_id
                        MATCH (parent:StandardName {{id: parent_id}})
                        WHERE {exact_reset_parent_where}
                        OPTIONAL MATCH (derived_source:StandardNameSource)
                        WHERE derived_source.id = 'derived:' + parent.id
                        WITH parent, derived_source
                        WHERE {exact_reset_source_where}
                        {skeleton_clause}
                        FOREACH (source IN CASE
                            WHEN derived_source IS NULL THEN [] ELSE [derived_source]
                        END | DETACH DELETE source)
                        DETACH DELETE parent
                        RETURN count(DISTINCT parent) AS skeleton_count
                    }}
                    RETURN size(candidate_changes) + skeleton_count AS n,
                           size(candidate_changes) AS candidate_count,
                           skeleton_count
                    """,
                    **params,
                    **deletion_params,
                    **{
                        f"reset_skeleton_{key}": value
                        for key, value in skeleton_params.items()
                    },
                )
                count = int(rows[0]["n"]) if rows else 0
            else:
                gc.query(
                    f"""
                    MATCH (src:IMASNode)-[rel:HAS_STANDARD_NAME]->(sn:StandardName)
                    WHERE {sn_where}
                    AND {src_where}
                    DELETE rel
                    WITH DISTINCT sn
                    WHERE NOT EXISTS {{ MATCH ()-[:HAS_STANDARD_NAME]->(sn) }}
                    {deletion_clause}
                    WITH sn, change
                    CALL {{
                        WITH sn
                        OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
                        DETACH DELETE r
                    }}
                    DETACH DELETE sn
                    """,
                    **params,
                    **deletion_params,
                )
        else:
            deletion_clause = services.deletion_change_cypher("sn")
            deletion_params = services.deletion_change_params(
                "clear_selected_name",
                reason="selected name removed by a pipeline clear",
            )
            gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE {sn_where}
                {deletion_clause}
                WITH sn, change
                CALL {{
                    WITH sn
                    OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
                    DETACH DELETE r
                }}
                DETACH DELETE sn
                """,
                **params,
                **deletion_params,
            )

        # Sweep any StandardNameReview node whose owner is absent.
        orphan_result = gc.query(
            """
            MATCH (r:StandardNameReview)
            WHERE NOT EXISTS { MATCH (:StandardName)-[:HAS_REVIEW]->(r) }
            WITH r LIMIT 10000
            DETACH DELETE r
            RETURN count(r) AS n
            """
        )
        orphan_count = orphan_result[0]["n"] if orphan_result else 0
        if orphan_count:
            logger.info("Swept %d orphaned StandardNameReview nodes", orphan_count)

        # Reset orphaned sources. Deleting a StandardName strands its
        # StandardNameSource at 'composed'/'attached' — a status the generate
        # pool never claims — so without this reset the source silently drops
        # out of all future regeneration. Orphan check (no remaining
        # PRODUCED_NAME edge) is inherently scope-correct: sources whose names
        # survived a filtered clear keep their status. The produced_sn_id
        # scalar mirror is nulled in the same operation so it never outlives
        # the deleted name (PRODUCED_NAME edge is the single source of truth).
        reset_result = gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status IN ['composed', 'attached']
            AND NOT (sns)-[:PRODUCED_NAME]->(:StandardName)
            SET sns.status = 'extracted',
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.produced_sn_id = null
            RETURN count(sns) AS n
            """
        )
        reset_count = reset_result[0]["n"] if reset_result else 0
        if reset_count:
            logger.info(
                "Reset %d orphaned StandardNameSource nodes to 'extracted'",
                reset_count,
            )

    logger.info("Deleted %d StandardName nodes", count)
    return count


def clear_sn_subsystem(*, dry_run: bool = False, services: Any) -> dict[str, int]:
    """Wipe every Standard Name the pipeline has produced.

    Deletes the six labels owned by the SN pipeline output:

    * ``StandardName`` — the generated names
    * ``StandardNameReview`` — RD-quorum review records
    * ``StandardNameSource`` — per-path extraction tracking
    * ``DocsRevision`` — refine_docs snapshot history (orphans without parent SN)
    * ``VocabGap`` — grammar vocabulary gap reports
    * ``SNRun`` — run audit / rotation memory

    ``LLMCost`` rows are an all-time record of money spent, independent of
    the mutable pipeline state cleared here. They remain in place even when
    every identity they name has been removed.

    **Grammar nodes** (``GrammarToken``, ``GrammarSegment``,
    ``GrammarTemplate``, ``ISNGrammarVersion``) are ISN-authoritative
    reference data and are never touched here. They stay in the graph so
    the vocabulary is immediately available for the next ``sn run``; the
    ``sn clear`` CLI re-seeds them from the installed ISN release after a
    wipe, and ``sn run`` auto-syncs at startup when the version differs.

    Parameters
    ----------
    dry_run:
        Count matching nodes without modifying the graph.

    Returns
    -------
    Dict mapping node label to deleted count. In dry-run mode,
    values are the current counts that would be deleted.
    """
    counts: dict[str, int] = {}
    labels = (
        "StandardName",
        "StandardNameReview",
        "StandardNameSource",
        "DocsRevision",
        "VocabGap",
        "SNRun",
    )

    with services.GraphClient() as gc:

        def _count(label: str) -> int:
            r = gc.query(f"MATCH (n:{label}) RETURN count(n) AS n")
            return r[0]["n"] if r else 0

        for label in labels:
            counts[label] = _count(label)

        if dry_run:
            return counts

        # Delete order is significant: StandardNameReview BEFORE StandardName so
        # orphan StandardNameReview nodes can't linger if HAS_STANDARD_NAME edges
        # are missing. DETACH DELETE handles remaining
        # edges on each pass. At SN-pipeline scale (~thousands of
        # nodes total) a single DETACH DELETE per label is sub-second.
        gc.query("MATCH (r:StandardNameReview) DETACH DELETE r")
        deletion_clause = services.deletion_change_cypher("sn")
        gc.query(
            f"""
            MATCH (sn:StandardName)
            {deletion_clause}
            DETACH DELETE sn
            RETURN count(sn) AS deleted
            """,
            **services.deletion_change_params(
                "clear_subsystem_name",
                reason="standard-name pipeline subsystem cleared",
            ),
        )
        gc.query("MATCH (s:StandardNameSource) DETACH DELETE s")
        gc.query("MATCH (d:DocsRevision) DETACH DELETE d")
        gc.query("MATCH (v:VocabGap) DETACH DELETE v")
        gc.query("MATCH (rr:SNRun) DETACH DELETE rr")

    total = sum(counts.values())
    logger.info("clear_sn_subsystem: deleted %d nodes (%s)", total, counts)

    return counts


_MAX_LINK_RETRIES = 5

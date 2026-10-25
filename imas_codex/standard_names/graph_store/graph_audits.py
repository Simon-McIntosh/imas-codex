"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
)
from imas_codex.standard_names.review.holds import (
    docs_acceptance_where,
)
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

from . import graph_foundation as foundation
from .graph_foundation import _CLAIM_TIMEOUT_SECONDS


@retry_on_deadlock()
def claim_compose_sources(
    *,
    limit: int = 15,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> tuple[str, list[dict]]:
    """Claim extracted StandardNameSource nodes for composition (polling).

    Unlike :func:`claim_standard_name_source_batch` this does NOT filter by
    ``batch_key`` — it claims any extracted source. Workers use this in a
    polling loop to pick up the next available batch of work regardless of
    which batch grouping the extract phase produced.

    Uses the standard ``ORDER BY rand()`` + ``claim_token`` two-step verify
    pattern from the discovery pipelines.

    Returns ``(token, claimed_sources)`` where each source dict has keys
    ``id``, ``source_id``, ``source_type``, ``batch_key``, ``description``.
    """
    token = str(foundation.services.uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"

    with foundation.services.GraphClient() as gc:
        # Step 1: Atomically claim with random ordering + unique token
        gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status = 'extracted'
              AND (sns.claimed_at IS NULL
                   OR sns.claimed_at < datetime() - duration($cutoff))
            WITH sns ORDER BY rand() LIMIT $limit
            SET sns.claimed_at = datetime(),
                sns.claim_token = $token
            """,
            limit=limit,
            token=token,
            cutoff=cutoff,
        )
        # Step 2: Verify — only our token
        claimed = list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description
                """,
                token=token,
            )
        )

    foundation.services.logger.debug(
        "claim_compose_sources: requested %d, won %d (token=%s)",
        limit,
        len(claimed),
        token[:8],
    )
    return token, [dict(r) for r in claimed]


def count_extracted_for_domain(domain: str, source: str = "dd") -> int:
    """Count StandardNameSource nodes already extracted for a physics domain.

    Used by :func:`run_sn_pools` to decide whether the domain-specific
    extract seed step is needed before starting the pools.

    Args:
        domain: Physics domain string (e.g. ``"turbulence"``).
        source: Source type — ``"dd"`` (DD paths) or ``"signals"``.

    Returns:
        Number of extracted sources whose backing entity belongs to *domain*.
    """
    if source == "dd":
        with foundation.services.GraphClient() as gc:
            result = gc.query(
                """
                MATCH (sns:StandardNameSource)-[:FROM_DD_PATH]->(n:IMASNode)
                WHERE sns.status = 'extracted'
                  AND n.physics_domain = $domain
                RETURN count(sns) AS cnt
                """,
                domain=domain,
            )
            return result[0]["cnt"] if result else 0
    return 0


def count_eligible_compose_sources(
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> int:
    """Count StandardNameSource nodes eligible for composition.

    Returns the number of extracted, unclaimed (or stale-claimed) sources.
    Used by polling workers for drain detection — when this returns 0 and
    no active leases remain, the compose phase is complete.
    """
    cutoff = f"PT{timeout_seconds}S"
    with foundation.services.GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status = 'extracted'
              AND (sns.claimed_at IS NULL
                   OR sns.claimed_at < datetime() - duration($cutoff))
            RETURN count(sns) AS cnt
            """,
            cutoff=cutoff,
        )
        return result[0]["cnt"] if result else 0


@retry_on_deadlock()
def claim_review_names(
    name_ids: list[str],
    *,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> tuple[str, list[str]]:
    """Claim specific StandardName nodes for review scoring.

    Only claims names from *name_ids* that are still eligible — not already
    claimed by another worker (unless the claim is stale).

    Uses the same ``claim_token`` two-step verify pattern as the compose
    claim functions.

    Returns ``(token, actually_claimed_ids)``.
    """
    if not name_ids:
        return "", []

    token = str(foundation.services.uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"

    with foundation.services.GraphClient() as gc:
        # Step 1: Atomically claim unclaimed/stale-claimed names
        gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            WHERE sn.validation_status = 'valid'
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($cutoff))
            SET sn.updated_at = datetime(), sn.claimed_at = datetime(),
                sn.claim_token = $token
            """,
            ids=name_ids,
            token=token,
            cutoff=cutoff,
        )
        # Step 2: Verify — only our token
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            WHERE sn.validation_status = 'valid'
            RETURN sn.id AS id
            """,
            token=token,
        )
        claimed = [r["id"] for r in result] if result else []

    foundation.services.logger.debug(
        "claim_review_names: requested %d, won %d (token=%s)",
        len(name_ids),
        len(claimed),
        token[:8],
    )
    return token, claimed


def release_review_claims(token: str) -> int:
    """Release all StandardName claims held by this token.

    Clears ``claimed_at`` and ``claim_token`` without changing any other
    fields.  Used for error recovery — released names become eligible for
    other workers.

    Returns count of released names.
    """
    if not token:
        return 0

    with foundation.services.GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.updated_at = datetime(), sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS affected
            """,
            token=token,
        )
        return result[0]["affected"] if result else 0


_CLAIM_HEARTBEAT_SECONDS = _CLAIM_TIMEOUT_SECONDS // 3


def refresh_name_claims(sn_ids: list[str], claim_token: str) -> int:
    """Refresh the lease on StandardName nodes still held by *claim_token*.

    Compare-and-set on ``claim_token``: only nodes that STILL bear this token
    have their ``claimed_at`` bumped to now. A node whose lease already
    expired and was re-claimed by a peer (its ``claim_token`` overwritten) is
    left untouched — the heartbeat never steals back a claim it has lost. This
    is the liveness half of the claim lease: the claim SEED is a compare-and-set
    (two-step token verify); this keeps a legitimately-held claim from expiring
    under a batch that outruns the TTL.

    Returns the number of nodes whose lease was refreshed.
    """
    if not claim_token or not sn_ids:
        return 0
    with foundation.services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid, claim_token: $token})
            SET sn.claimed_at = datetime(), sn.updated_at = datetime()
            RETURN count(sn) AS refreshed
            """,
            ids=list(sn_ids),
            token=claim_token,
        )
        return result[0]["refreshed"] if result else 0


PROMOTE_STRANDED_NAME_WHERE = (
    "sn.name_stage = 'reviewed' "
    "AND sn.reviewer_score_name >= $min_score "
    "AND coalesce(sn.edit_status, '') <> 'open' "
    "AND coalesce(sn.validation_status, '') <> 'quarantined' "
    "AND sn.review_quorum_shortfall IS NULL"
)


PROMOTE_STRANDED_DOCS_WHERE = (
    "sn.docs_stage = 'reviewed' "
    f"AND {docs_acceptance_where()} "
    "AND sn.name_stage = 'accepted' "
    "AND coalesce(sn.validation_status, '') <> 'quarantined' "
    "AND {docs_review_eligibility_where} "
    "AND sn.docs_review_quorum_shortfall IS NULL"
)


def promote_stranded_reviewed(
    min_score: float = DEFAULT_MIN_SCORE,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Promote names stranded at ``'reviewed'`` whose stored score now clears
    the active threshold.

    A name is scored once and its stage is decided against the threshold in
    force AT THAT TIME. When the acceptance threshold is later lowered, names
    that scored between the old and new thresholds are stuck at
    ``name_stage='reviewed'`` (or ``docs_stage='reviewed'``): the refine pool
    only claims names BELOW the threshold, so a name whose stored score already
    clears the current threshold is never re-touched and never accepted. This
    idempotent pass flips those to ``'accepted'`` on both axes.

    Guards (never promote a name that must go through the normal path):

    * ``reviewer_score_* >= min_score`` — only names that genuinely clear the
      CURRENT threshold; below-threshold names are left for refine.
    * name axis excludes ``edit_status='open'`` — a name carrying an unapplied
      edit (a rename / family-or-subtree cascade) must be accepted through
      :func:`persist_reviewed_name`, which applies the edit and cascades
      descendants atomically; a bare stage flip here would strand that cascade.
    * both axes exclude ``validation_status='quarantined'``.
    * docs axis requires ``name_stage='accepted'`` (docs is a post-name axis).
    * docs axis excludes an open documentation hold even when its judged score
      passes; an accepted open edit is marked applied with the stage write.
    * name axis excludes a recorded ``review_quorum_shortfall`` — a name held
      at ``'reviewed'`` because its reviewer chain never reached a verdict
      looks identical to one stranded by a lowered threshold, and its stored
      score can clear the bar while the seats behind it disagreed. Promoting
      on that score publishes exactly what the quorum gate withheld, so the
      marker must be honoured here as well as in
      :func:`persist_reviewed_name`.

    Idempotent: a second run matches nothing (the names are already accepted).

    Returns ``{"name": n_name, "docs": n_docs}`` — the number promoted on each
    axis (or that would be promoted in ``dry_run``).
    """
    name_where = foundation.services.PROMOTE_STRANDED_NAME_WHERE
    docs_where = foundation.services.PROMOTE_STRANDED_DOCS_WHERE
    with foundation.services.GraphClient() as gc:
        eligibility_params = foundation.services.docs_review_eligibility_params()
        if dry_run:
            n_name = gc.query(
                f"MATCH (sn:StandardName) WHERE {name_where} RETURN count(sn) AS n",
                min_score=min_score,
            )
            n_docs = gc.query(
                f"MATCH (sn:StandardName) WHERE {docs_where} RETURN count(sn) AS n",
                min_score=min_score,
                **eligibility_params,
            )
            return {
                "name": n_name[0]["n"] if n_name else 0,
                "docs": n_docs[0]["n"] if n_docs else 0,
            }
        # Name axis first so a name promoted here is eligible for the docs-axis
        # promotion in the same pass.
        r_name = gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {name_where}
            SET sn.updated_at = datetime(), sn.name_stage = 'accepted'
            RETURN count(sn) AS n
            """,
            min_score=min_score,
        )
        r_docs = gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {docs_where}
            SET sn.updated_at = datetime(), {foundation.services.docs_acceptance_set()}
            RETURN count(sn) AS n
            """,
            min_score=min_score,
            **eligibility_params,
        )
    promoted = {
        "name": r_name[0]["n"] if r_name else 0,
        "docs": r_docs[0]["n"] if r_docs else 0,
    }
    if promoted["name"] or promoted["docs"]:
        foundation.services.logger.info(
            "promote_stranded_reviewed: promoted %d name(s) + %d docs to "
            "accepted (stored score >= %.3f threshold)",
            promoted["name"],
            promoted["docs"],
            min_score,
        )
    return promoted


DOCS_AXIS_REDERIVE_NAME_WHERE = (
    "sn.name_stage = 'accepted' AND coalesce(sn.validation_status, '') <> 'quarantined'"
)


_DOCS_AXIS_MIRROR_AGREES = (
    "sn.docs_stage = 'accepted' "
    "AND sn.reviewer_score_docs IS NOT NULL "
    "AND abs(sn.reviewer_score_docs - winner.score) <= 0.000000001"
)


_DOCS_AXIS_MIRROR_DISAGREES = (
    "sn.docs_stage <> 'accepted' "
    "OR sn.reviewer_score_docs IS NULL "
    "OR abs(sn.reviewer_score_docs - winner.score) > 0.000000001"
)


def _docs_axis_rederive_query(*, dry_run: bool, ids: list[str] | None) -> str:
    """Build the docs-axis re-derivation statement.

    The statement reuses :func:`_docs_review_winner_query_body` so the selection
    rule has exactly one definition; a second traversal here would be a second
    source of truth for which surviving review is authoritative.
    """
    scope = (
        "UNWIND $ids AS sid\nMATCH (sn:StandardName {id: sid})\n"
        if ids is not None
        else "MATCH (sn:StandardName)\n"
    )
    if dry_run:
        tail = (
            f"RETURN count(sn) AS with_winner,\n"
            f"       sum(CASE WHEN {foundation.services._DOCS_AXIS_MIRROR_DISAGREES} THEN 1 ELSE 0 END)"
            f" AS repair,\n"
            f"       sum(CASE WHEN {foundation.services._DOCS_AXIS_MIRROR_AGREES} THEN 1 ELSE 0 END)"
            f" AS agree"
        )
    else:
        tail = (
            "SET sn.updated_at = datetime(), sn.docs_stage = 'accepted',\n"
            "    sn.reviewer_score_docs = winner.score\n"
            "RETURN count(sn) AS repair"
        )
    return (
        f"{scope}"
        f"WHERE {foundation.services.DOCS_AXIS_REDERIVE_NAME_WHERE}\n"
        f"CALL {{\n  WITH sn\n{foundation.services._docs_review_winner_query_body()}\n}}\n"
        f"WITH sn, winner\n"
        + (
            f"WHERE {foundation.services._DOCS_AXIS_MIRROR_DISAGREES}\n"
            if not dry_run
            else ""
        )
        + tail
    )


def reconcile_docs_axis_from_reviews(
    *,
    dry_run: bool = True,
    ids: list[str] | None = None,
    gc: Any | None = None,
) -> dict[str, int]:
    """Re-derive ``docs_stage`` and ``reviewer_score_docs`` from surviving docs reviews.

    **Selection rule:** a name's docs axis is taken from its winning docs-axis
    review — the highest-ranked surviving docs review group (a canonical group
    ahead of a non-canonical one, then the newest group, then the newest
    winning record inside it), as defined once by
    :func:`_docs_review_winner_query_body`.

    A name is repaired only when its name axis is already ``'accepted'``, it is
    not quarantined, and its stored mirror disagrees with that winner — its
    ``docs_stage`` is not ``'accepted'``, its stored score is null, or it
    carries a different score. **A name with no surviving docs-axis review at
    all is refused:** the winner traversal matches nothing for it, so it cannot
    reach the update, and it is left exactly as it is. That refusal is
    load-bearing — a pass that inferred acceptance from the absence of a review
    would manufacture the false-acceptance projections this catalog already
    contains, where a permissive scalar points at documentation no reviewer
    ever scored.

    This is the repeatable mechanism the release path is missing: a lifecycle
    scalar regressed by unrelated work is restored from the review edges that
    survived, instead of being repaired by hand against the same evidence.

    Idempotent: a repaired row now agrees with its winner and a second run
    matches nothing.

    Parameters
    ----------
    dry_run:
        When true (the default) nothing is written and the counts describe
        what a real run would do. ``agree`` counts rows whose stored mirror
        already equals the winner, ``repair`` those that would be re-derived,
        and ``with_winner`` every row in scope carrying a surviving docs-axis
        review.
    ids:
        Restrict the pass to these StandardName ids. Used to measure the rule
        against a named cohort; omitted, the pass covers the whole label.
    gc:
        Optional ``GraphClient``. One is opened when omitted.

    Returns
    -------
    dict[str, int]
        ``{"agree", "repair", "with_winner", "refused_no_review"}`` in
        ``dry_run``, or ``{"repaired"}`` after a real run.
    """
    eligibility = foundation.services.docs_review_eligibility_params()
    params: dict[str, foundation.services.Any] = dict(eligibility)
    if ids is not None:
        params["ids"] = list(ids)
    if gc is None:
        with foundation.services.GraphClient() as graph:
            rows = graph.query(
                foundation.services._docs_axis_rederive_query(dry_run=dry_run, ids=ids),
                **params,
            )
    else:
        rows = gc.query(
            foundation.services._docs_axis_rederive_query(dry_run=dry_run, ids=ids),
            **params,
        )
    if dry_run:
        row = dict(rows[0]) if rows else {}
        with_winner = int(row.get("with_winner") or 0)
        repair = int(row.get("repair") or 0)
        agree = int(row.get("agree") or 0)
        refused = len(ids) - with_winner if ids is not None else 0
        return {
            "agree": agree,
            "repair": repair,
            "with_winner": with_winner,
            "refused_no_review": refused,
        }
    repaired = int(rows[0]["repair"]) if rows else 0
    if repaired:
        foundation.services.logger.info(
            "reconcile_docs_axis_from_reviews: repaired %d name(s)", repaired
        )
    return {"repaired": repaired}


def _standard_name_source_upstream_present_cypher(source_variable: str) -> str:
    """Return the reconciliation predicate for a source's live upstream entity."""
    if source_variable not in {"sns", "source"}:
        raise ValueError("unsupported StandardNameSource Cypher variable")
    return """
        CASE
          WHEN __SOURCE__.source_type = 'dd' THEN EXISTS {
            MATCH (upstream:IMASNode {id: __SOURCE__.source_id})
            WHERE coalesce(upstream.lifecycle_status, '') <> 'removed'
          }
          WHEN __SOURCE__.source_type = 'signals' THEN EXISTS {
            MATCH (:FacilitySignal {id: __SOURCE__.source_id})
          }
          ELSE false
        END
        """.replace("__SOURCE__", source_variable)


def reconcile_standard_name_sources(source_type: str = "dd") -> dict:
    """Post-rebuild reconciliation of StandardNameSource nodes.

    1. Re-links sources to DD paths/signals that still exist
    2. Marks sources as stale if their upstream entity is gone
    3. Revives stale sources if their entity reappears

    Returns dict with counts: {relinked, stale_marked, revived}.
    """
    upstream_present = (
        foundation.services._standard_name_source_upstream_present_cypher("sns")
    )
    with foundation.services.GraphClient() as gc:
        if source_type == "dd":
            # Find stale: sources whose DD path no longer exists, or whose
            # node the DD-version lifecycle stamped removed (absent from the
            # current DD) — such sources must never re-enter the queue.
            stale = list(
                gc.query(
                    """
                    MATCH (sns:StandardNameSource {source_type: 'dd'})
                    WHERE (
                        NOT (sns)-[:FROM_DD_PATH]->(:IMASNode)
                        OR NOT (
                    """
                    + upstream_present
                    + """
                        )
                    )
                    AND NOT (sns.status = 'stale')
                    RETURN sns.id AS id
                    """
                )
            )
            stale_ids = [r["id"] for r in stale]
            stale_count = foundation.services.mark_sources_stale(stale_ids)

            # Revive: stale sources whose DD path now exists again
            revived = gc.query(
                """
                MATCH (sns:StandardNameSource
                       {source_type: 'dd', status: 'stale'})
                WHERE
                """
                + upstream_present
                + """
                MATCH (imas:IMASNode {id: sns.source_id})
                WHERE coalesce(imas.lifecycle_status, '') <> 'removed'
                SET sns.status = 'extracted',
                    sns.claimed_at = null,
                    sns.claim_token = null
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
                RETURN count(sns) AS count
                """
            )
            revived_count = revived[0]["count"] if revived else 0

            # Re-link: ensure FROM_DD_PATH exists for non-stale sources
            relinked = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'dd'})
                WHERE NOT (sns.status = 'stale')
                  AND NOT (sns)-[:FROM_DD_PATH]->()
                MATCH (imas:IMASNode {id: sns.source_id})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
                RETURN count(sns) AS count
                """
            )
            relinked_count = relinked[0]["count"] if relinked else 0
        else:
            # Signal reconciliation (same pattern, different relationships)
            stale = list(
                gc.query(
                    """
                    MATCH (sns:StandardNameSource {source_type: 'signals'})
                    WHERE (NOT (sns)-[:FROM_SIGNAL]->(:FacilitySignal)
                           OR NOT (
                    """
                    + upstream_present
                    + """
                           ))
                    AND NOT (sns.status = 'stale')
                    RETURN sns.id AS id
                    """
                )
            )
            stale_ids = [r["id"] for r in stale]
            stale_count = foundation.services.mark_sources_stale(stale_ids)

            revived = gc.query(
                """
                MATCH (sns:StandardNameSource
                       {source_type: 'signals', status: 'stale'})
                WHERE
                """
                + upstream_present
                + """
                MATCH (sig:FacilitySignal {id: sns.source_id})
                SET sns.status = 'extracted',
                    sns.claimed_at = null,
                    sns.claim_token = null
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
                RETURN count(sns) AS count
                """
            )
            revived_count = revived[0]["count"] if revived else 0

            relinked = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'signals'})
                WHERE NOT (sns.status = 'stale')
                  AND NOT (sns)-[:FROM_SIGNAL]->()
                MATCH (sig:FacilitySignal {id: sns.source_id})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
                RETURN count(sns) AS count
                """
            )
            relinked_count = relinked[0]["count"] if relinked else 0

    return {
        "stale_marked": stale_count,
        "revived": revived_count,
        "relinked": relinked_count,
    }


def fetch_manifest_source_release_rows(
    dd_paths: list[str], gc: Any | None = None
) -> list[dict[str, Any]]:
    """Resolve every manifest path to its current terminal name identity.

    Review-cohort minting intentionally selects only live names.  Release
    accounting needs a wider projection: the exact source membership plus the
    identity at the end of each source's successor chain, even when that name is
    exhausted or otherwise unaccepted.  Skipped sources retain their durable
    refusal so they can be reported as documented non-nameable rows.
    """
    paths = list(dict.fromkeys(dd_paths))
    if not paths:
        return []

    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        source_rows = client.query(
            """
            UNWIND $paths AS path
            OPTIONAL MATCH (source:StandardNameSource {id: 'dd:' + path})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(direct:StandardName)
            RETURN path AS source_path,
                   source.status AS source_status,
                   source.skip_reason AS skip_reason,
                   source.skip_reason_detail AS skip_reason_detail,
                   source.last_error AS last_error,
                   source.produced_sn_id AS produced_sn_id,
                   collect(DISTINCT direct.id) AS direct_ids
            ORDER BY source_path
            """,
            paths=paths,
        )
        ambiguous_bindings: list[str] = []
        for row in source_rows or []:
            direct_ids = sorted(row.get("direct_ids") or [])
            produced_id = row.get("produced_sn_id")
            if len(direct_ids) > 1 and produced_id not in direct_ids:
                ambiguous_bindings.append(
                    f"{row['source_path']} has direct targets {direct_ids!r} "
                    f"but produced_sn_id is {produced_id!r}"
                )
        if ambiguous_bindings:
            detail = "; ".join(ambiguous_bindings)
            raise ValueError(
                "ambiguous manifest source binding; reconcile the source before "
                f"release: {detail}"
            )
        seed_ids = sorted(
            {
                name_id
                for row in source_rows or []
                for name_id in [
                    row.get("produced_sn_id"),
                    *(row.get("direct_ids") or []),
                ]
                if name_id
            }
        )
        terminal_by_seed: dict[str, dict[str, foundation.services.Any]] = {}
        if seed_ids:
            successor_rows = client.query(
                """
                MATCH (start:StandardName)
                WHERE start.id IN $seed_ids
                OPTIONAL MATCH path=(start)-[:HAS_SUCCESSOR*0..]->(target:StandardName)
                WHERE NOT (target)-[:HAS_SUCCESSOR]->(:StandardName)
                RETURN start.id AS start_id, target.id AS target_id,
                       target.name_stage AS terminal_stage,
                       length(path) AS depth
                ORDER BY start_id, depth DESC, target_id
                """,
                seed_ids=seed_ids,
            )
            for row in successor_rows or []:
                terminal_by_seed.setdefault(str(row["start_id"]), dict(row))

        resolved: list[dict[str, foundation.services.Any]] = []
        for row in source_rows or []:
            direct_ids = sorted(row.get("direct_ids") or [])
            produced_id = row.get("produced_sn_id")
            seed_id = produced_id if produced_id in direct_ids else None
            if seed_id is None and len(direct_ids) == 1:
                seed_id = direct_ids[0]
            if seed_id is None and produced_id:
                seed_id = produced_id
            terminal = terminal_by_seed.get(str(seed_id)) if seed_id else None
            last_error = str(row.get("last_error") or "").strip()
            skip_reason = str(row.get("skip_reason") or "").strip()
            skip_detail = str(row.get("skip_reason_detail") or "").strip()
            skip_cause = ": ".join(part for part in [skip_reason, skip_detail] if part)
            # A terminal error is the current cause and wins over an older skip
            # record. Otherwise preserve the deterministic skip verdict and its
            # detail, and distinguish recorded silence from lost transcription.
            non_nameable_reason = last_error or skip_cause or "cause not recorded"
            if terminal or seed_id:
                non_nameable_reason = ""
            resolved.append(
                {
                    "source_path": str(row["source_path"]),
                    "source_status": row.get("source_status"),
                    "standard_name_id": (
                        terminal.get("target_id") if terminal else seed_id
                    ),
                    "terminal_stage": (
                        terminal.get("terminal_stage") if terminal else None
                    ),
                    "non_nameable_reason": non_nameable_reason,
                }
            )
        return resolved
    finally:
        if own:
            client.close()


def reconcile_source_status_liveness(
    gc: Any | None = None,
    *,
    source_ids: list[str] | None = None,
) -> dict[str, int]:
    """Make source lifecycle status agree with its current live target.

    ``composed`` and ``attached`` are terminal only while the source still
    points at a non-terminal ``StandardName``. A refine/supersede/delete path
    that loses the target can otherwise leave the source permanently excluded
    from generation. Conversely, a source with a live ``PRODUCED_NAME`` edge
    may still say ``extracted``/``failed`` after an interrupted write.

    The repair preserves the historical ``composed`` versus ``attached``
    distinction whenever it is already valid. Sources with a live target but
    another non-stale status become ``attached``, except when any live target
    carries an open name hint: that source must stay eligible for the
    regeneration requested by the edit. Sources claiming completion with no
    live target return to ``extracted`` with a fresh attempt budget. Terminal
    target edges, exact DD projections, source-path cache entries, and scalar
    mirrors are cleared in the latter case. Already-extracted DD sources with
    no produced edge receive the same projection/cache cleanup when their
    single exact backing path proves ownership of a terminal projection.
    Upstream-stale sources and active extracted claims are never revived or
    modified here.

    ``source_ids`` optionally restricts every repair direction to exact source
    identities; the default remains the whole graph. Idempotent once status,
    edge, and scalar liveness agree.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        live_rows = client.query(
            """
            MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            WHERE NOT (coalesce(sn.name_stage, '') IN
                       ['superseded', 'exhausted', 'contested'])
              AND ($source_ids IS NULL OR sns.id IN $source_ids)
              AND NOT (sns.status IN ['composed', 'attached', 'stale'])
              AND sns.claim_token IS NULL
              AND NOT EXISTS {
                MATCH (sns)-[:PRODUCED_NAME]->(hint:StandardName)
                WHERE NOT (coalesce(hint.name_stage, '') IN
                           ['superseded', 'exhausted', 'contested'])
                  AND coalesce(hint.edit_mode, '') = 'hint'
                  AND coalesce(hint.edit_status, '') = 'open'
                  AND hint.name_hint IS NOT NULL
              }
            SET sns.status = 'attached',
                sns.produced_sn_id = sn.id,
                sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(DISTINCT sns) AS n
            """,
            source_ids=source_ids,
        )
        live_realigned = live_rows[0]["n"] if live_rows else 0

        orphan_rows = client.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status IN ['composed', 'attached']
              AND ($source_ids IS NULL OR sns.id IN $source_ids)
              AND NOT EXISTS {
                MATCH (sns)-[:PRODUCED_NAME]->(live:StandardName)
                WHERE NOT (coalesce(live.name_stage, '') IN
                           ['superseded', 'exhausted', 'contested'])
              }
            OPTIONAL MATCH (sns)-[r:PRODUCED_NAME]->(terminal:StandardName)
            WITH sns, collect(DISTINCT r) AS stale_edges,
                 collect(DISTINCT terminal) AS terminal_targets
            OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(dd:IMASNode)
            WITH sns, stale_edges, terminal_targets,
                 collect(DISTINCT dd) AS dd_nodes
            WITH sns, stale_edges, terminal_targets,
                 CASE
                   WHEN size(dd_nodes) = 1
                    AND sns.source_type = 'dd'
                    AND sns.id = 'dd:' + sns.source_id
                    AND dd_nodes[0].id = sns.source_id
                   THEN dd_nodes[0] ELSE null
                 END AS exact_dd
            UNWIND CASE
              WHEN terminal_targets = [] THEN [null]
              ELSE terminal_targets
            END AS terminal
            OPTIONAL MATCH
                (exact_dd)-[projection:HAS_STANDARD_NAME]->(terminal)
            WITH sns, stale_edges, terminal_targets, terminal, projection,
                 EXISTS {
                      MATCH (other:StandardNameSource)-[:FROM_DD_PATH]->(exact_dd)
                      WHERE other <> sns
                        AND other.status IN ['composed', 'attached']
                        AND EXISTS {
                          MATCH (other)-[:PRODUCED_NAME]->(terminal)
                        }
                        AND EXISTS {
                          MATCH (other)-[:PRODUCED_NAME]->(other_live:StandardName)
                          WHERE NOT (coalesce(other_live.name_stage, '') IN
                                     $terminal_stages)
                        }
                 } AS backed_by_other
            WITH sns, stale_edges, terminal_targets, terminal, projection,
                 backed_by_other,
                 terminal IS NOT NULL
                   AND NOT backed_by_other
                   AND sns.id IN coalesce(terminal.source_paths, [])
                     AS had_source_path,
                 CASE
                   WHEN projection IS NOT NULL AND NOT backed_by_other
                   THEN projection ELSE null
                 END AS removable_projection
            FOREACH (target IN CASE
              WHEN terminal IS NULL OR backed_by_other THEN []
              ELSE [terminal] END |
              SET target.source_paths = [
                path IN coalesce(target.source_paths, [])
                WHERE path <> sns.id
              ]
            )
            WITH sns, stale_edges,
                 collect(DISTINCT removable_projection) AS stale_projections,
                 sum(CASE WHEN had_source_path THEN 1 ELSE 0 END)
                     AS source_paths_removed
            FOREACH (edge IN stale_edges | DELETE edge)
            FOREACH (edge IN stale_projections | DELETE edge)
            SET sns.status = 'extracted',
                sns.attempt_count = 0,
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.produced_sn_id = null,
                sns.composed_at = null
            RETURN count(DISTINCT sns) AS n,
                   sum(size(stale_edges)) AS edges,
                   sum(size(stale_projections)) AS projections,
                   sum(source_paths_removed) AS source_paths
            """,
            terminal_stages=sorted(foundation.services._TERMINAL_BINDING_NAME_STAGES),
            source_ids=source_ids,
        )
        orphaned_reset = orphan_rows[0]["n"] if orphan_rows else 0
        terminal_edges_dropped = (
            int(orphan_rows[0].get("edges") or 0) if orphan_rows else 0
        )
        terminal_projections_dropped = (
            int(orphan_rows[0].get("projections") or 0) if orphan_rows else 0
        )
        terminal_source_paths_dropped = (
            int(orphan_rows[0].get("source_paths") or 0) if orphan_rows else 0
        )

        projection_rows = client.query(
            """
            MATCH (sns:StandardNameSource {source_type: 'dd', status: 'extracted'})
            WHERE sns.claim_token IS NULL
              AND ($source_ids IS NULL OR sns.id IN $source_ids)
              AND NOT (sns)-[:PRODUCED_NAME]->(:StandardName)
            MATCH (sns)-[:FROM_DD_PATH]->(dd:IMASNode)
            WITH sns, collect(DISTINCT dd) AS dd_nodes
            WHERE size(dd_nodes) = 1
              AND sns.id = 'dd:' + sns.source_id
              AND dd_nodes[0].id = sns.source_id
            WITH sns, dd_nodes[0] AS dd
            MATCH (dd)-[projection:HAS_STANDARD_NAME]->(terminal:StandardName)
            WHERE coalesce(terminal.name_stage, '') IN $terminal_stages
              AND sns.id IN coalesce(terminal.source_paths, [])
            WITH sns, terminal, projection,
                 EXISTS {
                   MATCH (other:StandardNameSource)-[:FROM_DD_PATH]->(dd)
                   WHERE other <> sns
                     AND other.status IN ['composed', 'attached']
                     AND EXISTS {
                       MATCH (other)-[:PRODUCED_NAME]->(terminal)
                     }
                     AND EXISTS {
                       MATCH (other)-[:PRODUCED_NAME]->(other_live:StandardName)
                       WHERE NOT (coalesce(other_live.name_stage, '') IN
                                  $terminal_stages)
                     }
                 } AS backed_by_other
            WITH sns, terminal, projection, backed_by_other
            WHERE NOT backed_by_other
               OR sns.composed_at IS NOT NULL
               OR sns.produced_sn_id IS NOT NULL
               OR sns.claimed_at IS NOT NULL
            FOREACH (target IN CASE
              WHEN backed_by_other THEN [] ELSE [terminal] END |
              SET target.source_paths = [
                path IN coalesce(target.source_paths, [])
                WHERE path <> sns.id
              ]
            )
            FOREACH (edge IN CASE
              WHEN backed_by_other THEN [] ELSE [projection] END |
              DELETE edge
            )
            WITH sns, count(DISTINCT CASE
                   WHEN NOT backed_by_other THEN terminal
                 END) AS terminal_targets,
                 count(DISTINCT CASE
                   WHEN NOT backed_by_other THEN projection
                 END) AS projections
            SET sns.claimed_at = null,
                sns.produced_sn_id = null,
                sns.composed_at = null
            RETURN count(DISTINCT sns) AS n,
                   sum(terminal_targets) AS terminal_targets,
                   sum(projections) AS projections
            """,
            terminal_stages=sorted(foundation.services._TERMINAL_BINDING_NAME_STAGES),
            source_ids=source_ids,
        )
        projection_ghosts_reset = projection_rows[0]["n"] if projection_rows else 0
        ghost_projections_dropped = (
            int(projection_rows[0].get("projections") or 0) if projection_rows else 0
        )
        ghost_source_paths_dropped = (
            int(projection_rows[0].get("terminal_targets") or 0)
            if projection_rows
            else 0
        )
    finally:
        if own:
            client.close()

    if live_realigned or orphaned_reset or projection_ghosts_reset:
        foundation.services.logger.info(
            "reconcile_source_status_liveness: realigned %d source(s) with a "
            "live target; returned %d source(s) without one to extracted "
            "(%d stale edge(s), %d projection(s), and %d source-path entry(s) "
            "dropped); cleaned %d projection-only ghost source(s) "
            "(%d projection(s), %d source-path entry(s))",
            live_realigned,
            orphaned_reset,
            terminal_edges_dropped,
            terminal_projections_dropped,
            terminal_source_paths_dropped,
            projection_ghosts_reset,
            ghost_projections_dropped,
            ghost_source_paths_dropped,
        )
    return {
        "live_realigned": live_realigned,
        "orphaned_reset": orphaned_reset,
        "terminal_edges_dropped": terminal_edges_dropped,
        "terminal_projections_dropped": terminal_projections_dropped,
        "terminal_source_paths_dropped": terminal_source_paths_dropped,
        "projection_ghosts_reset": projection_ghosts_reset,
        "ghost_projections_dropped": ghost_projections_dropped,
        "ghost_source_paths_dropped": ghost_source_paths_dropped,
    }


def retire_unreachable_hint_edits(gc: Any | None = None) -> int:
    """Reject name hints whose target cannot re-enter name generation.

    A name-axis hint is generation steering, not a replacement name. On an
    already accepted or exhausted target no name-stage claim consumes that
    hint, so leaving ``edit_status='open'`` creates a permanent edit backlog.
    New requests are refused by :func:`imas_codex.standard_names.edit._apply_hint`;
    this startup repair closes rows written before that gate existed.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.edit_status = 'open'
              AND sn.name_hint IS NOT NULL
              AND sn.name_stage IN ['accepted', 'exhausted']
            SET sn.updated_at = datetime(), sn.edit_status = 'rejected',
                sn.reviewer_comments_name =
                    trim(coalesce(sn.reviewer_comments_name, '')
                    + ' [edit_rejected] name hints require a target that can '
                    + 're-enter generation; use a full rename proposal')
            RETURN count(sn) AS n
            """
        )
        retired = rows[0]["n"] if rows else 0
    finally:
        if own:
            client.close()
    if retired:
        foundation.services.logger.info(
            "retire_unreachable_hint_edits: rejected %d terminal name-hint edit(s)",
            retired,
        )
    return retired


def reconcile_vocab_gaps() -> dict[str, int]:
    """Re-validate all VocabGap nodes against the current ISN vocabulary.

    Performs three passes on every VocabGap node:

    1. **Delete false positives** — token now exists in the reported segment
       (ISN was upgraded since the gap was written).
    2. **Delete invalid-segment gaps** — segment name is not in ISN grammar.
    3. **Reclassify surviving gaps** — recompute ``category`` and
       ``actual_segments`` against the current ISN installation.  A gap
       previously ``absent`` may become ``wrong_slot_placement`` if ISN added
       the token in a different segment.

    Returns dict with counts:
        checked, deleted_false_positive, deleted_invalid_segment,
        deleted_open_segment, reclassified, remaining.
    """
    from imas_codex.standard_names.segments import (
        classify_gap,
        known_segments,
    )

    segs = known_segments()
    if segs is None:
        foundation.services.logger.warning(
            "reconcile_vocab_gaps: ISN unavailable — skipping reconciliation"
        )
        return {"checked": 0, "skipped": True}

    with foundation.services.GraphClient() as gc:
        all_gaps = list(
            gc.query(
                """
                MATCH (vg:VocabGap)
                RETURN vg.id AS id, vg.segment AS segment,
                       vg.token AS token, vg.category AS category
                """
            )
        )

    if not all_gaps:
        return {"checked": 0, "remaining": 0}

    to_delete: list[str] = []
    to_update: list[dict] = []
    stats: dict[str, int] = {
        "checked": len(all_gaps),
        "deleted_false_positive": 0,
        "deleted_invalid_segment": 0,
        "deleted_open_segment": 0,
        "reclassified": 0,
    }

    for gap in all_gaps:
        segment, token = gap["segment"], gap["token"]
        new_category, new_actual = classify_gap(segment, token)

        if new_category == "false_positive":
            to_delete.append(gap["id"])
            stats["deleted_false_positive"] += 1
        elif new_category == "invalid_segment":
            to_delete.append(gap["id"])
            stats["deleted_invalid_segment"] += 1
        elif new_category == "open_segment":
            to_delete.append(gap["id"])
            stats["deleted_open_segment"] += 1
        elif new_category != gap.get("category"):
            to_update.append(
                {
                    "id": gap["id"],
                    "category": new_category,
                    "actual_segments": new_actual,
                }
            )
            stats["reclassified"] += 1

    with foundation.services.GraphClient() as gc:
        if to_delete:
            gc.query(
                """
                UNWIND $ids AS gap_id
                MATCH (vg:VocabGap {id: gap_id})
                DETACH DELETE vg
                """,
                ids=to_delete,
            )
            foundation.services.logger.info(
                "reconcile_vocab_gaps: deleted %d false-positive/invalid gaps",
                len(to_delete),
            )

        if to_update:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (vg:VocabGap {id: b.id})
                SET vg.category = b.category,
                    vg.actual_segments = b.actual_segments,
                    vg.reconciled_at = datetime()
                """,
                batch=to_update,
            )
            foundation.services.logger.info(
                "reconcile_vocab_gaps: reclassified %d gaps",
                len(to_update),
            )

        # Sources parked at 'vocab_gap' whose blocking gaps have all been
        # resolved (no remaining source-first HAS_STANDARD_NAME_VOCAB_GAP edge)
        # revert to 'extracted' so the next generate pool retries them under the
        # upgraded vocabulary. Without this, a vocabulary fix never un-blocks the
        # sources that reported the gap. Uses the direct source→VocabGap edge; a
        # deleted gap node removes it via DETACH, a reclassified one keeps it.
        retried = gc.query(
            """
            MATCH (sns:StandardNameSource {status: 'vocab_gap'})
            WHERE NOT (sns)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(:VocabGap)
            SET sns.status = 'extracted',
                sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS n
            """
        )
        rows = list(retried) if retried else []
        stats["sources_retried"] = rows[0].get("n", 0) if rows else 0
        if stats["sources_retried"]:
            foundation.services.logger.info(
                "reconcile_vocab_gaps: reset %d vocab_gap sources to 'extracted'",
                stats["sources_retried"],
            )

    stats["remaining"] = stats["checked"] - len(to_delete)
    return stats


def triage_vocab_gaps(
    gc: Any | None = None,
    *,
    stale_before: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sort every stored ``VocabGap`` into one lifecycle bucket, with provenance.

    Run AFTER :func:`reconcile_vocab_gaps` (categories current) and after the
    mechanical reuse verdicts are stamped — this pass only reads what those
    wrote and derives the bucket:

    - ``add`` / ``fold`` / ``reject`` — an active editorial disposition,
      projected directly without changing the observation category;
    - ``reuse`` — a mechanical check resolved the token to a registered one
      (``dedup_decision = 'reuse_confirmed'``);
    - ``rule_violation`` — the grammar already forbids the construction
      (ordinal sample points, bare DD field spellings);
    - ``composable`` — expressible from registered grammar
      (``decomposable`` / ``wrong_slot_placement`` / ``ambiguous_known_token``);
    - ``retired_stale`` — still ``absent`` but last proposed before
      ``stale_before``: no current composer reproduces it, so it is history,
      not demand;
    - ``genuine`` — ``absent`` and recently reproduced: the real vocabulary
      demand a rotation decision reads.

    Nothing is deleted; each node gets ``triage`` + ``triaged_at`` and keeps
    every field that led to its bucket. Returns the counts and the ``genuine``
    (segment, token, example_count) list, newest first.
    """
    from imas_codex.graph.models import VocabGapDisposition

    editorial_buckets = tuple(item.value for item in VocabGapDisposition)
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (vg:VocabGap)
            RETURN vg.id AS id, vg.segment AS segment, vg.token AS token,
                   vg.category AS category, vg.dedup_decision AS dedup,
                   CASE WHEN vg.editorial_active = true
                        THEN vg.editorial_disposition ELSE null END
                       AS editorial_disposition,
                   toString(vg.last_seen_at) AS last_seen,
                   vg.example_count AS n
            """
        )
        buckets: dict[str, list[dict[str, foundation.services.Any]]] = {
            **{value: [] for value in editorial_buckets},
            "reuse": [],
            "rule_violation": [],
            "composable": [],
            "retired_stale": [],
            "genuine": [],
        }
        for r in rows:
            if r.get("editorial_disposition") in editorial_buckets:
                bucket = r["editorial_disposition"]
            elif r.get("dedup") == "reuse_confirmed":
                bucket = "reuse"
            elif r.get("category") == "rule_violation":
                bucket = "rule_violation"
            elif r.get("category") in (
                "decomposable",
                "wrong_slot_placement",
                "ambiguous_known_token",
            ):
                bucket = "composable"
            elif (r.get("last_seen") or "") < stale_before:
                bucket = "retired_stale"
            else:
                bucket = "genuine"
            buckets[bucket].append(r)

        if not dry_run:
            client.query(
                """
                UNWIND $items AS it
                MATCH (vg:VocabGap {id: it.id})
                SET vg.triage = it.bucket, vg.triaged_at = datetime()
                """,
                items=[
                    {"id": r["id"], "bucket": b}
                    for b, rs in buckets.items()
                    for r in rs
                ],
            )
        genuine = sorted(
            (
                {
                    "segment": r["segment"],
                    "token": r["token"],
                    "example_count": r.get("n") or 0,
                    "last_seen": r.get("last_seen"),
                }
                for r in buckets["genuine"]
            ),
            key=lambda x: (x["last_seen"] or "", x["example_count"]),
            reverse=True,
        )
        return {
            "checked": len(rows),
            "counts": {
                b: len(rs)
                for b, rs in buckets.items()
                if rs or b not in editorial_buckets
            },
            "genuine": genuine,
            "dry_run": dry_run,
        }
    finally:
        if own:
            client.close()


_UNIT_SKIP_REASON_PREFIX = "dd_unit_"


def revive_unit_skipped_sources(gc: Any | None = None) -> dict[str, int]:
    """Re-evaluate unit-derived skips against the CURRENT unit resolver.

    ``_apply_unit_overrides`` removes a DD row whose effective unit fails the
    parse and records the exclusion durably as
    ``skip_reason='dd_unit_unresolvable'`` (status ``'skipped'``). That verdict
    is a snapshot of the resolver at extraction time: when the resolver later
    learns the unit — a dimensionless sentinel is a real unit, a count
    pseudo-unit maps to ``1``, a symbol is reordered into canonical form — the
    source stays excluded because nothing revisits it.

    This pass asks the extractor's own question again, through the extractor's
    own code (``resolve_unit`` + ``_is_unparseable_dd_unit``), for every source
    still blocked by a unit skip, and returns the ones that now parse to
    ``'extracted'`` so the generate pool picks them up. The unit is read the way
    extraction reads it — the ``HAS_UNIT`` edge first, the ``unit`` property as
    fallback — so the verdict cannot diverge from what a fresh extraction would
    decide.

    Scope is deliberately narrow:

    * only ``status='skipped'`` — a source that already advanced carries
      ``skip_reason`` as an audit crumb, and reviving it would rewind progress;
    * only ``skip_reason`` in the ``dd_unit_*`` family — non-unit skips are
      correct, permanent eligibility exclusions;
    * never a source whose DD node is gone or ``lifecycle_status='removed'``;
    * a source whose unit still fails (or whose path an explicit skip rule
      matches) is left exactly as it was.

    ``attempt_count`` is reset because the recorded attempts were spent under
    the broken resolver — a revived source that stays over the seeding cap
    would be revived into a queue that never claims it.

    Idempotent: a revived source no longer carries the skip, so a second pass
    at an unchanged resolver revives nothing.

    Returns dict: {checked, revived}.
    """
    from imas_codex.standard_names.sources.dd import _is_unparseable_dd_unit
    from imas_codex.standard_names.unit_overrides import resolve_unit

    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        candidates = list(
            client.query(
                """
                MATCH (sns:StandardNameSource {status: 'skipped'})
                      -[:FROM_DD_PATH]->(n:IMASNode)
                WHERE sns.skip_reason STARTS WITH $prefix
                  AND coalesce(n.lifecycle_status, '') <> 'removed'
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                WITH sns, n, collect(DISTINCT u.id) AS unit_rels
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       CASE WHEN size(unit_rels) = 1 THEN unit_rels[0]
                            WHEN size(unit_rels) > 1
                            THEN coalesce(n.unit, unit_rels[0])
                            ELSE null END AS unit_from_rel,
                       n.unit AS node_unit
                """,
                prefix=foundation.services._UNIT_SKIP_REASON_PREFIX,
            )
        )

        revivable: list[str] = []
        for row in candidates:
            path = row.get("source_id") or ""
            raw_unit = row.get("unit_from_rel") or row.get("node_unit") or None
            effective, meta = resolve_unit(path, raw_unit)
            if meta and meta.get("rule") == "skip":
                continue  # an explicit rule still skips this path
            final = effective if (meta and meta.get("rule") == "override") else raw_unit
            if not final or _is_unparseable_dd_unit(final):
                continue
            revivable.append(row["id"])

        revived = 0
        if revivable:
            rows = client.query(
                """
                UNWIND $ids AS sns_id
                MATCH (sns:StandardNameSource {id: sns_id})
                SET sns.status = 'extracted',
                    sns.skip_reason = null,
                    sns.skip_reason_detail = null,
                    sns.attempt_count = 0,
                    sns.claimed_at = null,
                    sns.claim_token = null
                RETURN count(sns) AS revived
                """,
                ids=revivable,
            )
            revived = rows[0]["revived"] if rows else 0
    finally:
        if own:
            client.close()

    if revived:
        foundation.services.logger.info(
            "revive_unit_skipped_sources: revived %d of %d unit-skipped "
            "source(s) whose unit the current resolver parses",
            revived,
            len(candidates),
        )
    return {"checked": len(candidates), "revived": revived}


def _isn_grammar_context() -> dict[str, Any] | None:
    """Return the live ISN grammar context, or None when ISN is unavailable."""
    try:
        from imas_standard_names import get_grammar_context

        return get_grammar_context()
    except Exception:  # noqa: BLE001 — ISN absent or failing to load
        return None


@functools.lru_cache(maxsize=1)
def isn_vocabulary_signature() -> str | None:
    """Return a digest identifying the ISN grammar vocabulary in force.

    The identity is derived from the vocabulary ITSELF — every segment with its
    token list, plus the operator registry — not from a package version string.
    A version is the wrong key: a fork or an editable install can change the
    vocabulary without moving the version, and a version bump that touches no
    vocabulary would pointlessly re-open every parked source. ISN owns the
    vocabulary; this only hashes what ``get_grammar_context()`` reports.

    Returns ``None`` when ISN is unavailable, so callers can no-op rather than
    stamp a meaningless identity.
    """
    ctx = foundation.services._isn_grammar_context()
    if not ctx:
        return None

    sections = ctx.get("vocabulary_sections") or []
    payload: list[tuple[str, tuple[str, ...]]] = []
    for section in sections:
        segment = section.get("segment")
        if not segment:
            continue
        payload.append((str(segment), tuple(sorted(section.get("tokens") or ()))))
    operators = (ctx.get("grammar") or {}).get("vocabularies", {}).get("operators", {})
    payload.append(("operators", tuple(sorted(operators))))

    if not payload:
        return None
    payload.sort()
    return foundation.services.hashlib.sha256(
        foundation.services.json.dumps(payload).encode()
    ).hexdigest()[:16]


def retry_vocab_gap_sources_on_grammar_change(
    gc: Any | None = None,
) -> dict[str, int]:
    """Grant a parked ``vocab_gap`` source one retry per vocabulary change.

    ``reconcile_vocab_gaps`` un-parks a source only once it has no remaining
    blocking ``VocabGap`` — i.e. only when the *exact token the composer asked
    for* turns out to exist. But that request is frequently the wrong spelling
    of a capability the vocabulary already expresses, or now expresses, another
    way (a request for ``h_98_factor`` / ``confinement`` is answered by
    ``confinement_enhancement_factor`` + ``ipb98y2``). Keyed on the requested
    token, such a source is stranded permanently by construction.

    So the retry key is the vocabulary as a whole. Each source records the
    vocabulary digest in force when it was parked
    (``vocab_gap_grammar_signature``); a source whose stamp differs from the
    live digest — including an unstamped source parked before the stamp
    existed — is returned to ``'extracted'`` for exactly ONE fresh compose
    attempt, and re-stamped with the current digest. The blocking gap edges are
    dropped so the next attempt reports what actually blocks it now rather than
    inheriting a stale request; the ``VocabGap`` nodes themselves survive (they
    are the ISN-facing worklist and are owned by ``reconcile_vocab_gaps``).
    ``attempt_count`` is reset because the prior attempts were spent under a
    vocabulary that could not express the quantity.

    Idempotent: after one pass at a fixed vocabulary every remaining parked
    source carries the current digest, so a second pass revives nothing. A
    genuinely inexpressible quantity re-parks with the current stamp and waits
    for the next vocabulary change — one retry per change, never a loop.

    Returns dict: {checked, revived} (plus ``skipped`` when ISN is unavailable).
    """
    signature = foundation.services.isn_vocabulary_signature()
    if signature is None:
        foundation.services.logger.warning(
            "retry_vocab_gap_sources_on_grammar_change: ISN vocabulary "
            "unavailable — skipping retry pass"
        )
        return {"checked": 0, "revived": 0, "skipped": True}

    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (sns:StandardNameSource {status: 'vocab_gap'})
            RETURN count(sns) AS checked
            """
        )
        checked = rows[0]["checked"] if rows else 0

        revived_rows = client.query(
            """
            MATCH (sns:StandardNameSource {status: 'vocab_gap'})
            WHERE coalesce(sns.vocab_gap_grammar_signature, '') <> $signature
            SET sns.status = 'extracted',
                sns.vocab_gap_grammar_signature = $signature,
                sns.skip_reason = null,
                sns.skip_reason_detail = null,
                sns.attempt_count = 0,
                sns.claimed_at = null,
                sns.claim_token = null
            WITH sns
            OPTIONAL MATCH (sns)-[r:HAS_STANDARD_NAME_VOCAB_GAP]->(:VocabGap)
            DELETE r
            RETURN count(DISTINCT sns) AS revived
            """,
            signature=signature,
        )
        revived = revived_rows[0]["revived"] if revived_rows else 0
    finally:
        if own:
            client.close()

    if revived:
        foundation.services.logger.info(
            "retry_vocab_gap_sources_on_grammar_change: returned %d of %d "
            "parked source(s) to 'extracted' for one retry under vocabulary %s",
            revived,
            checked,
            signature,
        )
    return {"checked": checked, "revived": revived}


def reconcile_provenance() -> dict[str, int]:
    """Realign StandardNameSource provenance metadata with live StandardNames.

    The ``produced_sn_id`` scalar mirrors the ``PRODUCED_NAME`` edge, but on
    name supersede/delete/reset only the edge is removed — the scalar mirror
    is orphaned and silently points at a now-deleted name. Likewise a
    ``derived_parent`` scaffolding source can outlive the parent StandardName
    it was minted for. Both are audit pollution, not pipeline blockers, so
    this idempotent sweep is safe to run every rotation.

    Three passes:

    0. **Reattach live desyncs** — any ``StandardNameSource`` whose
       ``produced_sn_id`` scalar names a LIVE ``StandardName`` but whose
       ``PRODUCED_NAME`` edge is missing has the edge MERGEd back (the scalar
       is the recovery mirror; a missing edge is silent provenance loss).
    1. **NULL stale scalars** — any ``StandardNameSource.produced_sn_id`` whose
       target ``StandardName`` no longer exists is set to null.
    2. **Delete orphaned derived-parent scaffolding** — any edge-less derived
       ``StandardNameSource {batch_key: 'derived_parent'}`` whose intended
       ``StandardName(id = source_id)`` no longer exists is DETACH-DELETEd.
       A live ``PRODUCED_NAME`` edge is authoritative after a cascade retargets
       the source while preserving its immutable ``source_id``.

    Returns dict with counts: {edges_reattached, scalars_cleared,
    orphan_sources_deleted}.
    """
    with foundation.services.GraphClient() as gc:
        # Pass 0: heal live scalar/missing-edge desyncs before cleanup. Disjoint
        # from pass 1 (that only touches scalars whose target no longer exists).
        edges_reattached = foundation.services.reattach_produced_name_edges(gc=gc)

        scalars = gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.produced_sn_id IS NOT NULL
              AND NOT EXISTS {
                  MATCH (sn:StandardName {id: sns.produced_sn_id})
              }
            SET sns.produced_sn_id = null
            RETURN count(sns) AS n
            """
        )
        scalars_cleared = scalars[0]["n"] if scalars else 0

        orphans = gc.query(
            """
            MATCH (sns:StandardNameSource {batch_key: 'derived_parent'})
            WHERE sns.source_type = 'derived'
              AND NOT EXISTS {
                  MATCH (sns)-[:PRODUCED_NAME]->(:StandardName)
              }
              AND NOT EXISTS {
                  MATCH (sn:StandardName {id: sns.source_id})
              }
            DETACH DELETE sns
            RETURN count(sns) AS n
            """
        )
        orphan_sources_deleted = orphans[0]["n"] if orphans else 0

    if edges_reattached or scalars_cleared or orphan_sources_deleted:
        foundation.services.logger.info(
            "reconcile_provenance: reattached %d missing PRODUCED_NAME edge(s), "
            "cleared %d stale produced_sn_id scalar(s), "
            "deleted %d orphaned derived-parent source(s)",
            edges_reattached,
            scalars_cleared,
            orphan_sources_deleted,
        )

    return {
        "edges_reattached": edges_reattached,
        "scalars_cleared": scalars_cleared,
        "orphan_sources_deleted": orphan_sources_deleted,
    }


def reconcile_catalog_edit_origins(
    gc: Any | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Remove catalog authority from identities without catalog evidence.

    ``StandardName.origin`` describes how an identity entered the graph.  A
    catalog edit is a stronger claim: it says a catalog request supplied the
    authoritative edit.  Keep that claim only when the identity carries a
    catalog request number, approval timestamp, or approved lifecycle stage.

    The export/import provenance path historically used ``catalog_edit`` as a
    fallback for every imported identity.  For the unearned residue, recover
    the strongest provenance still present in the graph: a derived source
    wins over a regular DD or signal source, and an identity with no such
    evidence has the stale property removed.  The operation is idempotent and
    records one non-catalog reconciliation event per changed identity.

    Returns counts for ``eligible``, ``pipeline``, ``derived``, ``unset``, and
    ``changed``.  With ``dry_run=True`` no graph mutation occurs.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        classification = client.query(
            """
            MATCH (sn:StandardName {origin: 'catalog_edit'})
            WHERE sn.catalog_pr_number IS NULL
              AND sn.catalog_approved_at IS NULL
              AND coalesce(sn.name_stage, '') <> 'approved'
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            WITH sn, collect(DISTINCT source.source_type) AS source_types
            WITH CASE
                   WHEN 'derived' IN source_types THEN 'derived'
                   WHEN any(kind IN source_types
                            WHERE kind IN ['dd', 'signals']) THEN 'pipeline'
                   ELSE null
                 END AS target_origin
            RETURN target_origin, count(*) AS changed
            ORDER BY target_origin
            """
        )
        counts = {
            "eligible": sum(int(row.get("changed") or 0) for row in classification),
            "pipeline": sum(
                int(row.get("changed") or 0)
                for row in classification
                if row.get("target_origin") == "pipeline"
            ),
            "derived": sum(
                int(row.get("changed") or 0)
                for row in classification
                if row.get("target_origin") == "derived"
            ),
            "unset": sum(
                int(row.get("changed") or 0)
                for row in classification
                if row.get("target_origin") is None
            ),
            "changed": 0,
        }
        if dry_run or not counts["eligible"]:
            return counts

        changed = client.query(
            """
            MATCH (sn:StandardName {origin: 'catalog_edit'})
            WHERE sn.catalog_pr_number IS NULL
              AND sn.catalog_approved_at IS NULL
              AND coalesce(sn.name_stage, '') <> 'approved'
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            WITH sn, collect(DISTINCT source.source_type) AS source_types
            WITH sn, CASE
                       WHEN 'derived' IN source_types THEN 'derived'
                       WHEN any(kind IN source_types
                                WHERE kind IN ['dd', 'signals']) THEN 'pipeline'
                       ELSE null
                     END AS target_origin
            SET sn.origin = target_origin, sn.updated_at = datetime()
            CREATE (change:StandardNameChange {
              id: 'sn-change:' + randomUUID(),
              from_name: sn.id,
              to_name: sn.id,
              operation: 'reconcile_catalog_edit_origin',
              reason: 'catalog_edit origin had no catalog request evidence',
              origin: 'lineage_reconciliation',
              changed_at: datetime(),
              internal: true
            })
            MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(change)
            RETURN count(sn) AS changed
            """
        )
        counts["changed"] = int(changed[0].get("changed") or 0) if changed else 0
    finally:
        if own:
            client.close()

    if counts["changed"]:
        foundation.services.logger.info(
            "reconcile_catalog_edit_origins: reassigned %d identity origin(s) "
            "(%d pipeline, %d derived, %d unset)",
            counts["changed"],
            counts["pipeline"],
            counts["derived"],
            counts["unset"],
        )
    return counts


def reconcile_standard_name_cocos_links(gc: Any | None = None) -> dict[str, int]:
    """Link COCOS-dependent standard names to the catalog's COCOS convention.

    A standard name with a ``cocos_transformation_type`` (psi_like, ip_like, …)
    is COCOS-dependent: its sign depends on the convention. The catalog follows
    a single convention — the current DD version's (DDv4 → COCOS 17) — so every
    such name should carry the integer ``cocos`` and a ``HAS_COCOS`` edge to
    that COCOS singleton. COCOS-independent names (no transformation type) are
    left unlinked by design.

    Idempotent and re-runnable: sets the ``cocos`` scalar where null and MERGEs
    the ``HAS_COCOS`` edge where missing, across every origin (pipeline, derived
    parents, catalog edits). No-op once the invariant holds.

    Returns dict: {convention, scalars_set, edges_created}.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        row = client.query(
            "MATCH (v:DDVersion {is_current: true}) RETURN v.cocos AS cocos"
        )
        convention = (
            row[0]["cocos"] if row and row[0].get("cocos") is not None else None
        )
        if convention is None:
            foundation.services.logger.debug(
                "reconcile_standard_name_cocos_links: no current DD COCOS convention — skipping"
            )
            return {"convention": 0, "scalars_set": 0, "edges_created": 0}

        scalars = client.query(
            """
            MATCH (s:StandardName)
            WHERE s.cocos_transformation_type IS NOT NULL AND s.cocos IS NULL
            SET s.updated_at = datetime(), s.cocos = $conv
            RETURN count(s) AS n
            """,
            conv=convention,
        )
        scalars_set = scalars[0]["n"] if scalars else 0

        edges = client.query(
            """
            MATCH (s:StandardName)
            WHERE s.cocos_transformation_type IS NOT NULL
              AND s.cocos IS NOT NULL
              AND NOT (s)-[:HAS_COCOS]->(:COCOS)
            MATCH (c:COCOS {id: s.cocos})
            MERGE (s)-[:HAS_COCOS]->(c)
            RETURN count(s) AS n
            """
        )
        edges_created = edges[0]["n"] if edges else 0
    finally:
        if own:
            client.close()

    if scalars_set or edges_created:
        foundation.services.logger.info(
            "reconcile_standard_name_cocos_links: convention=COCOS %s, set %d cocos scalar(s), "
            "created %d HAS_COCOS edge(s)",
            convention,
            scalars_set,
            edges_created,
        )
    return {
        "convention": convention,
        "scalars_set": scalars_set,
        "edges_created": edges_created,
    }


def reconcile_standard_name_unit_edges(gc: Any | None = None) -> dict[str, int]:
    """Realign each name's ``HAS_UNIT`` edge set with its own ``unit`` scalar.

    ``StandardName`` declares ``HAS_UNIT`` with cardinality *one*, and every
    writer that sets a unit already drops the pre-existing edge before merging
    the canonical one, so a unit CORRECTION cannot strand the old edge beside
    the new. A name whose unit was last written before that self-heal existed
    still carries the residue — and if it has since reached a terminal stage no
    writer will ever revisit it, making the stale edge permanent. This is the
    net for exactly that case.

    The ``unit`` scalar is authoritative: it is what the catalog export, the
    reviewer prompt, and the attachment guard's dimensionality rule all read. A
    name carrying two units has no single dimensionality, so the guard will
    admit sources of *either* — which is how physically distinct quantities
    collapse onto one name.

    Names with no declared unit are left alone: their edge is the only unit
    information they carry, and awaiting a unit is not a cardinality violation.
    Applies at every stage, terminal included, since the cardinality invariant
    is structural rather than a claim about live physics.

    Idempotent: no-op once every name's edge set is exactly its scalar.

    Returns dict: {names_realigned, edges_dropped, edges_created}.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (sn:StandardName)-[:HAS_UNIT]->(u:Unit)
            WHERE sn.unit IS NOT NULL AND sn.unit <> ''
            WITH sn, collect(DISTINCT u.id) AS edges
            WHERE NOT (size(edges) = 1 AND edges[0] = sn.unit)
            RETURN sn.id AS id, sn.unit AS unit, edges AS edges
            ORDER BY id
            """
        )
        if not rows:
            return {"names_realigned": 0, "edges_dropped": 0, "edges_created": 0}

        dropped = sum(1 for r in rows for e in r["edges"] if e != r["unit"])
        created = sum(1 for r in rows if r["unit"] not in r["edges"])

        client.query(
            """
            UNWIND $items AS item
            MATCH (sn:StandardName {id: item.id})
            OPTIONAL MATCH (sn)-[r:HAS_UNIT]->(stale:Unit)
            WHERE stale.id <> item.unit
            DELETE r
            WITH DISTINCT sn, item
            MERGE (u:Unit {id: item.unit})
            MERGE (sn)-[:HAS_UNIT]->(u)
            """,
            items=[{"id": r["id"], "unit": r["unit"]} for r in rows],
        )
    finally:
        if own:
            client.close()

    foundation.services.logger.info(
        "reconcile_standard_name_unit_edges: realigned %d name(s) — dropped %d "
        "stale unit edge(s), created %d missing edge(s)",
        len(rows),
        dropped,
        created,
    )
    return {
        "names_realigned": len(rows),
        "edges_dropped": dropped,
        "edges_created": created,
    }


def reconcile_constraint_fit_artifacts(gc: Any | None = None) -> dict[str, int]:
    """Reclassify constraint fit bookkeeping that still looks nameable.

    Constraint ``weight`` leaves and reconstructed-value branches describe how
    an equilibrium fit was performed, not independent physical quantities.  DD
    graphs built before that distinction was available retain ``quantity`` on
    those leaves, admitting them to standard-name extraction.  Select only
    leaf nodes below a ``constraints`` subtree whose terminal segment is
    ``weight`` or whose path contains a reconstructed-value segment.

    Idempotent: the selector matches only ``quantity`` nodes, so every repaired
    node drops out after its category becomes ``fit_artifact``.

    Returns dict: {nodes_reclassified}.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (node:IMASNode {node_category: 'quantity'})
            WHERE node.is_leaf = true
              AND node.id CONTAINS '/constraints/'
              AND (
                last(split(node.id, '/')) = 'weight'
                OR any(
                  segment IN split(node.id, '/')
                  WHERE segment = 'reconstructed'
                     OR segment ENDS WITH '_reconstructed'
                )
              )
            SET node.node_category = 'fit_artifact'
            RETURN count(node) AS reclassified
            """
        )
    finally:
        if own:
            client.close()

    reclassified = int(rows[0].get("reclassified", 0)) if rows else 0
    if reclassified:
        foundation.services.logger.info(
            "reconcile_constraint_fit_artifacts: reclassified %d constraint "
            "fit-artifact leaf node(s)",
            reclassified,
        )
    return {"nodes_reclassified": reclassified}


_TERMINAL_NAME_STAGES = ["superseded", "exhausted", "contested"]


def reconcile_standard_name_source_paths(gc: Any | None = None) -> dict[str, int]:
    """Materialize each live name's ``source_paths`` scalar from its edges.

    ``source_paths`` is a denormalised cache written once at compose time (the
    union of DD paths sent to the composer) and updated by NO edge-mutating
    path — not refine (``persist_refined_name`` migrates HAS_STANDARD_NAME
    edges but never touches the scalar), not prune (the sever path deletes
    edges but leaves the scalar), and not ``reconcile_standard_name_sources``
    (which reconciles FROM_DD_PATH edges only). So the scalar goes stale the
    moment any mapping is remapped or pruned, mis-driving consumers that trust
    it. This reconcile makes the edges the source of truth: it recomputes the
    scalar as the sorted, deduped union of the canonical-URI forms of the live
    provenance edges —

    - ``'dd:' + imas.id`` for each ``HAS_STANDARD_NAME`` IMASNode source, and
    - ``src.id`` for each non-``derived`` ``PRODUCED_NAME`` StandardNameSource

    — while preserving any existing ``derived:`` scalar entries (structural
    provenance, not a source entity). This mirrors exactly the edge→scalar
    mapping the edge-integrity invariant asserts, so the invariant guards it in
    CI. Scoped to non-terminal names; idempotent (no-op once the scalar equals
    the union). Any HAS_STANDARD_NAME edges backfilled later are picked up on
    the next run automatically — edge completeness is a separate concern this
    function deliberately does not judge.

    Returns dict: {names_reconciled}.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (sn:StandardName)
            WHERE NOT (sn.name_stage IN $terminal)
            OPTIONAL MATCH (imas:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
            WITH sn, collect(DISTINCT 'dd:' + imas.id) AS hsn
            OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            WHERE src.source_type <> 'derived' AND src.id IS NOT NULL
            WITH sn, hsn, collect(DISTINCT src.id) AS prod
            RETURN sn.id AS id,
                   coalesce(sn.source_paths, []) AS current,
                   [p IN hsn WHERE p IS NOT NULL AND p <> 'dd:'] AS hsn_paths,
                   prod AS produced_paths
            """,
            terminal=foundation.services._TERMINAL_NAME_STAGES,
        )
        updates = []
        for r in rows:
            current = list(r["current"])
            derived_keep = [p for p in current if p.startswith("derived:")]
            union = sorted(
                set(derived_keep) | set(r["hsn_paths"]) | set(r["produced_paths"])
            )
            if union != current:
                updates.append({"id": r["id"], "paths": union})
        if updates:
            client.query(
                """
                UNWIND $updates AS u
                MATCH (sn:StandardName {id: u.id})
                SET sn.source_paths = u.paths, sn.updated_at = datetime()
                """,
                updates=updates,
            )
    finally:
        if own:
            client.close()

    if updates:
        foundation.services.logger.info(
            "reconcile_standard_name_source_paths: materialized source_paths "
            "from live edges on %d name(s)",
            len(updates),
        )
    return {"names_reconciled": len(updates)}


def reconcile_catalog_status(gc: Any | None = None) -> dict[str, int]:
    """Maintain the catalog lifecycle without assuming publication authority.

    ``status`` is distinct from the pipeline's ``name_stage``: it gates what a
    catalog may publish. New and historical names without a catalog status are
    drafts. Superseded identities carry the successor-aware terminal catalog
    status, while exhausted identities remain drafts because they never entered
    a catalog. Exhaustion is a pipeline outcome, not a validation result, so
    this reconcile never changes ``validation_status``.

    Approval is the sole writer of ``'active'``. Accordingly, this reconcile
    makes superseded pipeline identities ``'superseded'``, repairs every
    exhausted identity to ``status='draft'``, restores receipt-backed revived
    identities to ``'draft'``, and makes any remaining null status ``'draft'``.
    It never writes ``'deprecated'``; that transition
    belongs only to withdrawal of a name that was actually active. Terminal
    transitions run first so each changed name is counted once.

    Returns dict: {drafted, revived, superseded, quarantined, deprecated,
    total_changed}.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        superseded_rows = client.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.name_stage = 'superseded'
              AND (sn.status IS NULL OR sn.status = 'draft')
            SET sn.status = 'superseded', sn.updated_at = datetime()
            RETURN count(sn) AS changed
            """
        )
        exhausted_rows = client.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.name_stage = 'exhausted'
              AND coalesce(sn.status, '') <> 'draft'
            SET sn.status = 'draft', sn.updated_at = datetime()
            RETURN count(sn) AS changed
            """
        )
        revived_rows = client.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.name_stage IS NOT NULL
              AND sn.name_stage <> 'superseded'
              AND sn.name_stage <> 'exhausted'
              AND sn.status = 'superseded'
              AND EXISTS {
                  MATCH (sn)-[:HAS_INTERNAL_CHANGE]->
                        (change:StandardNameChange {operation: 'fold_identity'})
                  WHERE change.to_name = sn.id
              }
            SET sn.status = 'draft', sn.updated_at = datetime()
            RETURN count(sn) AS changed
            """
        )
        drafted_rows = client.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.status IS NULL
            SET sn.status = 'draft', sn.updated_at = datetime()
            RETURN count(sn) AS changed
            """
        )
    finally:
        if own:
            client.close()

    result = {
        "drafted": (drafted_rows[0]["changed"] if drafted_rows else 0)
        + (exhausted_rows[0]["changed"] if exhausted_rows else 0),
        "revived": revived_rows[0]["changed"] if revived_rows else 0,
        "superseded": superseded_rows[0]["changed"] if superseded_rows else 0,
        "quarantined": 0,
        "deprecated": 0,
    }
    result["total_changed"] = sum(result.values())
    if result["total_changed"]:
        foundation.services.logger.info(
            "reconcile_catalog_status: set %d name(s) to draft, revived %d, "
            "%d to superseded; exhausted names remain validation-neutral; "
            "deprecated writes: 0",
            result["drafted"],
            result["revived"],
            result["superseded"],
        )
    return result


def reconcile_reviewable_name_stage(gc: Any | None = None) -> dict[str, int]:
    """Advance source-backed pipeline names stranded below the review entry stage.

    The name review pool claims ``name_stage='drafted'``. A name minted by the
    normal compose path always reaches 'drafted' (its finalize is an
    unconditional ``SET``), but a refine that converges onto a name string that
    already exists as an un-reviewed placeholder — a locus/parent scaffold at
    ``name_stage='pending'`` or a bare node with no stage — historically kept
    that placeholder stage (its successor init was ``ON CREATE`` only). The
    refine still migrates the real ``PRODUCED_NAME`` / ``HAS_STANDARD_NAME``
    sources onto it, so the result is a valid, source-backed quantity that can
    never be reviewed.

    :func:`persist_refined_name` now advances such a successor at write time;
    this reconcile heals any name already stranded by the earlier behaviour and
    is an idempotent safety net for any other path that could leave a produced
    name below 'drafted'. A name qualifies when it is at ``name_stage`` null or
    ``'pending'``, is ``validation_status='valid'``, is not a structural
    ``derived`` parent (those are reviewed structurally, not by the name
    quorum), and carries at least one non-``derived`` produced source (a real
    DD/signal origin). Once advanced the name is ``'drafted'`` and no longer
    matches — the pass is a no-op thereafter.

    The ``'valid'`` requirement is narrow but load-bearing:
    :data:`REVIEW_NAME_ELIGIBILITY_WHERE` demands ``'valid'`` too, so advancing
    a quarantined name here would move it from stranded-at-``'pending'`` to
    stranded-at-``'drafted'`` without making it claimable by anything. A
    quarantined name cannot re-enter the pool loop at any stage — review never
    claims it, so it never reaches ``'reviewed'`` and refine never claims it
    either. Its repair vehicle is ``sn edit --rename``, which mints a fresh,
    separately validated successor rather than advancing a stage.

    A null description withholds the advance for exactly the same reason: the
    review predicate demands prose, so promoting a prose-less name would move
    it from stranded-at-``'pending'`` to stranded-at-``'drafted'`` and buy
    nothing. Such a name needs the opposite repair, and
    :func:`reconcile_descriptionless_composed_names` runs from here to apply
    it, reporting its own counts to the log.

    Returns dict: {names_advanced}.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            WHERE coalesce(sn.name_stage, '') IN ['', 'pending']
              AND coalesce(sn.origin, '') <> 'derived'
              AND coalesce(sn.validation_status, '') = 'valid'
              AND coalesce(src.source_type, '') <> 'derived'
              AND sn.description IS NOT NULL
              AND trim(sn.description) <> ''
            WITH DISTINCT sn
            SET sn.updated_at = datetime(), sn.name_stage = 'drafted',
                sn.origin     = 'pipeline'
            RETURN count(sn) AS advanced
            """
        )
        foundation.services.reconcile_descriptionless_composed_names(gc=client)
        foundation.services.reconcile_sourceless_pipeline_names(gc=client)
        foundation.services.reconcile_constraint_fit_artifacts(gc=client)
    finally:
        if own:
            client.close()

    advanced = rows[0]["advanced"] if rows else 0
    if advanced:
        foundation.services.logger.info(
            "reconcile_reviewable_name_stage: advanced %d stranded "
            "source-backed name(s) to 'drafted' for review",
            advanced,
        )
    return {"names_advanced": advanced}


def reconcile_descriptionless_composed_names(
    gc: Any | None = None,
) -> dict[str, int]:
    """Re-open a composed name that carries no description at all.

    :data:`REVIEW_NAME_ELIGIBILITY_WHERE` demands a non-null description, so a
    name whose description is null is never claimed for name review — and
    compose cannot refill the prose either, because the source that produced it
    already reads ``composed`` or ``attached`` while the compose pool claims
    only ``extracted``. The pair is terminally stuck while still presenting as
    live work, which is how an unscoped drain comes to exit reporting no
    eligible work graph-wide.

    The repair returns both sides to a state the pipeline can act on: the name
    drops to ``'pending'``, and each bound non-derived source returns to
    ``'extracted'`` with a fresh attempt budget, its produced edge released and
    its identity dropped from the name's ``source_paths`` cache. Releasing the
    edge rather than keeping it is what makes the repair survive to the next
    run — a re-opened source still pointing at a live target would be realigned
    straight back to ``'attached'`` by
    :func:`reconcile_source_status_liveness`, re-stranding the pair. If compose
    then lands on the same spelling the node is reused, the prose is written and
    the finalize advances it; if it lands elsewhere the ``'pending'`` node keeps
    no false claim on the review pool.

    Idempotent: once the sources are freed no ``composed``/``attached``
    non-derived source points at the name, so a second pass matches nothing.
    Sources under an active claim are left alone — the compose worker holding
    them owns the write.

    Returns dict: {names_reopened, sources_reopened}.
    """
    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (src:StandardNameSource)-[produced:PRODUCED_NAME]->
                  (sn:StandardName)
            WHERE coalesce(sn.name_stage, '') IN ['', 'pending', 'drafted']
              AND (sn.description IS NULL OR trim(sn.description) = '')
              AND coalesce(sn.origin, '') <> 'derived'
              AND coalesce(src.source_type, '') <> 'derived'
              AND src.status IN ['composed', 'attached']
              AND src.claim_token IS NULL
            WITH sn, collect(DISTINCT src) AS sources,
                 collect(DISTINCT produced) AS produced_edges
            SET sn.updated_at = datetime(), sn.name_stage = 'pending',
                sn.source_paths = [
                  path IN coalesce(sn.source_paths, [])
                  WHERE NOT path IN [source IN sources | source.id]
                ]
            FOREACH (edge IN produced_edges | DELETE edge)
            FOREACH (source IN sources |
              SET source.status         = 'extracted',
                  source.attempt_count  = 0,
                  source.claimed_at     = null,
                  source.claim_token    = null,
                  source.produced_sn_id = null,
                  source.composed_at    = null
            )
            RETURN count(DISTINCT sn) AS names,
                   sum(size(sources)) AS sources_freed
            """
        )
    finally:
        if own:
            client.close()

    names_reopened = int(rows[0].get("names") or 0) if rows else 0
    sources_reopened = int(rows[0].get("sources_freed") or 0) if rows else 0
    if names_reopened:
        foundation.services.logger.info(
            "reconcile_descriptionless_composed_names: re-opened %d name(s) "
            "carrying no description — %d source(s) returned to 'extracted' "
            "so compose rewrites the missing prose",
            names_reopened,
            sources_reopened,
        )
    return {
        "names_reopened": names_reopened,
        "sources_reopened": sources_reopened,
    }


def reconcile_sourceless_pipeline_names(gc: Any | None = None) -> dict[str, int]:
    """Reattach the detached source of a live pipeline name when the reason lapsed.

    A consistency detach (:func:`reconcile_attachment_consistency`) strips a
    source whose pairing the compose guard then rejected, returns the freed
    source to ``'extracted'`` and records the event as a
    ``'detach_inconsistent_attachment'`` ``StandardNameChange``. The name itself
    is left live but with no producing source — a provenance orphan. If the
    rejection was transient — the guard judged the pairing under an older
    grammar (e.g. a retired operator spelling whose semantics the audit now
    resolves, or a stale DD unit) — nothing ever re-pairs the two, because the
    freed source sits at ``'extracted'`` where no pool revisits an already live
    name.

    This reconcile closes that gap. For every live pipeline-origin name with no
    producing source it reads the recorded detachment(s), re-asks the guard
    under the CURRENT grammar, and:

    * reattaches the source when the guard now accepts it — the provenance edge,
      the DD-side projection, the source's produced state and the name's
      ``source_paths`` entry are restored, and the re-pair is itself recorded as
      a ``StandardNameChange``;
    * leaves the source alone and surfaces the name as a blocker when the
      reason still holds — a catalog-authoritative name must not silently keep
      a provenance the data no longer justifies;
    * reports a name whose detachment path is no longer on record as
      unrecoverable rather than guessing a replacement source.

    Only sources in the exact rewound state the detach produced are candidates:
    a ``'dd'`` source at ``'extracted'`` carrying no claim, no ``produced_sn_id``
    and no edge to a different live name. A source that has since composed or
    attached elsewhere is current authority for that target and is never
    reclaimed. Status mirrors the compose path (``'composed'``, ``produced_sn_id``
    set); the liveness reconcile that follows startup rewrites it to
    ``'attached'`` as a side effect.

    Idempotent: a name that gains a source drops out of the selector, so a
    second pass matches nothing.

    Returns dict: {sourceless, reattached, names_reattached, blockers,
    unrecoverable}.
    """
    from imas_codex.standard_names.attachment_audit import _attachment_consistency
    from imas_codex.standard_names.provenance_lifecycle import (
        record_standard_name_change,
    )

    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    terminal = foundation.services._TERMINAL_NAME_STAGES
    try:
        names = client.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.origin = 'pipeline'
              AND NOT coalesce(sn.name_stage, '') IN $terminal
              AND NOT (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            OPTIONAL MATCH (sn)-[:HAS_INTERNAL_CHANGE]->(ch:StandardNameChange)
            WHERE ch.operation = 'detach_inconsistent_attachment'
            WITH sn, collect(DISTINCT ch.from_name) AS detached_paths
            RETURN sn.id AS sn_id, sn.name_stage AS stage,
                   [p IN detached_paths WHERE p IS NOT NULL] AS detached_paths
            ORDER BY sn.id
            """,
            terminal=terminal,
        )
        sourceless = len(names)
        reattached = 0
        names_reattached = 0
        blockers: list[str] = []
        unrecoverable: list[str] = []
        for row in names:
            sn_id = row.get("sn_id")
            if not sn_id:
                continue
            paths = row.get("detached_paths") or []
            if not paths:
                unrecoverable.append(sn_id)
                continue
            name_gained_source = False
            for dd_path in sorted({p for p in paths if p}):
                source_id = f"dd:{dd_path}"
                probe = client.query(
                    """
                    MATCH (sn:StandardName {id: $sn_id})
                    OPTIONAL MATCH (sns:StandardNameSource {id: $source_id})
                    OPTIONAL MATCH (dd:IMASNode {id: $dd_path})
                    OPTIONAL MATCH (dd)-[:HAS_UNIT]->(du:Unit)
                    RETURN sn.unit AS sn_unit,
                           sns.id AS source_id,
                           sns.status AS status,
                           sns.claim_token AS claim_token,
                           sns.claimed_at AS claimed_at,
                           sns.produced_sn_id AS produced_sn_id,
                           coalesce(du.id, dd.unit) AS dd_unit,
                           EXISTS {
                             MATCH (sns)-[:PRODUCED_NAME]->(other:StandardName)
                             WHERE NOT coalesce(other.name_stage, '') IN $terminal
                           } AS bound_to_other_live
                    """,
                    sn_id=sn_id,
                    source_id=source_id,
                    dd_path=dd_path,
                    terminal=terminal,
                )
                p = probe[0] if probe else {}
                ok, reason = _attachment_consistency(
                    dd_path,
                    sn_id,
                    dd_unit=p.get("dd_unit"),
                    sn_unit=p.get("sn_unit"),
                )
                if not ok:
                    # The recorded reason still holds under the current
                    # grammar — do not reattach; the name stays a blocker.
                    continue
                # The reason has lapsed. Reattach only if the source is still
                # in the exact rewound state the detach produced — gone,
                # claimed, composed elsewhere, or authority for another live
                # target means the pair is no longer recoverable here.
                if (
                    p.get("source_id") is None
                    or p.get("status") != "extracted"
                    or p.get("claim_token") is not None
                    or p.get("claimed_at") is not None
                    or p.get("produced_sn_id") is not None
                    or p.get("bound_to_other_live")
                ):
                    continue
                written = client.query(
                    """
                    MATCH (sns:StandardNameSource {id: $source_id})
                    MATCH (sn:StandardName {id: $sn_id})
                    WHERE sns.status = 'extracted'
                      AND sns.claim_token IS NULL
                      AND sns.claimed_at IS NULL
                      AND sns.produced_sn_id IS NULL
                      AND NOT EXISTS {
                        MATCH (sns)-[:PRODUCED_NAME]->(other:StandardName)
                        WHERE NOT coalesce(other.name_stage, '') IN $terminal
                      }
                      AND NOT (sns)-[:PRODUCED_NAME]->(sn)
                    OPTIONAL MATCH (dd:IMASNode {id: $dd_path})
                    MERGE (sns)-[:PRODUCED_NAME]->(sn)
                    FOREACH (_ IN CASE WHEN dd IS NULL THEN [] ELSE [1] END |
                      MERGE (dd)-[:HAS_STANDARD_NAME]->(sn))
                    SET sn.updated_at = datetime(), sns.status = 'composed',
                        sns.produced_sn_id = sn.id,
                        sns.composed_at = datetime(),
                        sns.claimed_at = null,
                        sns.claim_token = null,
                        sn.source_paths = CASE
                          WHEN $uri IN coalesce(sn.source_paths, [])
                          THEN sn.source_paths
                          ELSE coalesce(sn.source_paths, []) + $uri END
                    RETURN count(sn) AS n
                    """,
                    sn_id=sn_id,
                    source_id=source_id,
                    dd_path=dd_path,
                    uri=source_id,
                    terminal=terminal,
                )
                count = int(written[0]["n"]) if written else 0
                if count:
                    reattached += 1
                    name_gained_source = True
                    try:
                        record_standard_name_change(
                            client,
                            dd_path,
                            sn_id,
                            operation="reattach_lapsed_detachment",
                            reason=(
                                "guard now accepts the pairing under the "
                                "current grammar"
                            ),
                            origin="sourceless_reconcile",
                        )
                    except (
                        Exception
                    ):  # pragma: no cover - audit crumb must not block the heal
                        foundation.services.logger.debug(
                            "Failed to record reattachment of %s",
                            dd_path,
                            exc_info=True,
                        )
            if not name_gained_source:
                # A recorded detachment exists but no source was restored:
                # the reason holds, or the pair is no longer recoverable.
                blockers.append(sn_id)
            else:
                names_reattached += 1
    finally:
        if own:
            client.close()

    if reattached:
        foundation.services.logger.info(
            "reconcile_sourceless_pipeline_names: reattached %d detached "
            "source(s) to %d live pipeline name(s) whose detachment reason no "
            "longer holds",
            reattached,
            names_reattached,
        )
    if blockers:
        foundation.services.logger.warning(
            "reconcile_sourceless_pipeline_names: %d live pipeline name(s) "
            "stay sourceless because their recorded detachment reason still "
            "holds under the current grammar: %s",
            len(blockers),
            "; ".join(blockers)[:1200],
        )
    if unrecoverable:
        foundation.services.logger.warning(
            "reconcile_sourceless_pipeline_names: %d live pipeline name(s) "
            "have no recorded detached source to re-pair and are surfaced "
            "for review: %s",
            len(unrecoverable),
            "; ".join(unrecoverable)[:1200],
        )
    return {
        "sourceless": sourceless,
        "reattached": reattached,
        "names_reattached": names_reattached,
        "blockers": len(blockers),
        "unrecoverable": len(unrecoverable),
    }


def reconcile_standard_name_dd_edges(gc: Any | None = None) -> dict[str, int]:
    """Materialize the DD-side ``HAS_STANDARD_NAME`` edge from provenance.

    The ``(:IMASNode)-[:HAS_STANDARD_NAME]->(:StandardName)`` edge is the
    codex-internal 1-hop navigation index over the DD: given a name, every DD
    path that realizes it; given a path, its canonical name. It was written
    imperatively — one edge per name, to the single representative source that
    seeded it (``create_nodes``). When consolidation collapses N same-concept
    DD sources into one name, the other N-1 keep their
    ``PRODUCED_NAME``/``FROM_DD_PATH`` provenance but never receive a DD-side
    edge, so the index reaches only a fraction of the paths the provenance
    already asserts.

    This makes the edge what it structurally is — a materialized projection of
    the per-source provenance — and reconciles it, symmetric with
    :func:`reconcile_standard_name_source_paths` on the SN-side scalar. For
    every ``(s)-[:PRODUCED_NAME]->(sn)`` on a non-terminal name where
    ``(s)-[:FROM_DD_PATH]->(dd)`` and ``dd`` is SN-eligible
    (``node_category ∈ SN_SOURCE_CATEGORIES``), MERGE the edge only when the
    shared attachment guard admits the pair.

    **Full gate (drop-and-triage).** Unit disagreement, semantic-shape conflict,
    geometry-representation conflict, and sibling composition conflict all
    reject a fresh projection. A DD path with **no** unit edge remains a DD
    completeness gap rather than an attachment defect. Candidate order is
    deterministic so the compose-style sibling rule retains the same accepted
    representative on every pass.

    Idempotent: the match excludes pairs already carrying the edge, so a second
    run creates zero. Provenance stays the sole authority — no orphan edge is
    ever authored for a node with no producing source.

    Returns dict: {edges_created, pairs_dropped}.
    """
    from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
    from imas_codex.standard_names.workers import _is_attachment_consistent

    own = gc is None
    client = foundation.services.GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            WHERE NOT (sn.name_stage IN $terminal)
            MATCH (s)-[:FROM_DD_PATH]->(dd:IMASNode)
            WHERE dd.node_category IN $categories
              AND NOT (dd)-[:HAS_STANDARD_NAME]->(sn)
            OPTIONAL MATCH (dd)-[:HAS_UNIT]->(du:Unit)
            OPTIONAL MATCH (bound:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
            RETURN DISTINCT dd.id AS dd_path, sn.id AS sn_id, sn.id AS name,
                   sn.unit AS sn_unit, du.id AS dd_unit,
                   collect(DISTINCT bound.id) AS existing_dd_paths
            """,
            terminal=foundation.services._TERMINAL_NAME_STAGES,
            categories=list(SN_SOURCE_CATEGORIES),
        )
        attach: list[dict[str, str]] = []
        dropped: list[dict[str, foundation.services.Any]] = []
        accepted_by_name: dict[str, list[str]] = {}
        for r in sorted(rows, key=lambda row: (row["sn_id"], row["dd_path"])):
            accepted = accepted_by_name.setdefault(
                r["sn_id"], sorted(r.get("existing_dd_paths") or [])
            )
            ok, reason = _is_attachment_consistent(
                r["dd_path"],
                r["sn_id"],
                existing_sources=tuple(accepted),
                dd_unit=r["dd_unit"],
                sn_unit=r["sn_unit"],
            )
            if ok:
                attach.append({"dd_path": r["dd_path"], "sn_id": r["sn_id"]})
                accepted.append(r["dd_path"])
            else:
                dropped.append({**r, "reason": reason})
        if attach:
            client.query(
                """
                UNWIND $pairs AS p
                MATCH (dd:IMASNode {id: p.dd_path})
                MATCH (sn:StandardName {id: p.sn_id})
                MERGE (dd)-[:HAS_STANDARD_NAME]->(sn)
                """,
                pairs=attach,
            )
    finally:
        if own:
            client.close()

    if attach:
        foundation.services.logger.info(
            "reconcile_standard_name_dd_edges: materialized %d HAS_STANDARD_NAME "
            "edge(s) from provenance",
            len(attach),
        )
    if dropped:
        # A rejected pair is a wrong name or source, never an edge to force.
        # Surface a bounded sample for attachment triage.
        foundation.services.logger.warning(
            "reconcile_standard_name_dd_edges: dropped %d provenance pair(s) on "
            "attachment consistency (route to attachment triage). First few: %s",
            len(dropped),
            "; ".join(
                f"{d['name']} ↮ {d['dd_path']} ({d['reason']})" for d in dropped[:10]
            ),
        )
    return {"edges_created": len(attach), "pairs_dropped": len(dropped)}


_SEGMENT_REALIGN_OPERATION = "realign_grammar_segments"


def reconcile_grammar_segments() -> dict[str, int]:
    """Realign every name's grammar segment columns with its canonical id.

    The bare-name segments (``position``, ``component``, ``subject``, …) are a
    deterministic function of the canonical name id via the ISN parser. A name
    written by an out-of-grammar path — e.g. the since-removed bulk catalog
    import — can carry stale segments that disagree with its own id: an
    ``…_at_pedestal_top`` name storing ``position='pedestal'`` is the observed
    case. Such a name no longer recomposes to itself, and a re-composition from
    its DD leaf would mint a divergent ``_at_pedestal`` near-duplicate.

    This idempotent sweep re-parses every live name through the authoritative
    ISN parser and realigns any drifted segment column to the parse. Names the
    ISN grammar cannot parse are skipped (their segments are owned by the
    quarantine path, never wiped here). Safe to run every rotation.

    A superseded or exhausted identity is a tombstone: rewriting its stored
    segments as a side effect of a routine sweep changes a historical record a
    published catalog may still resolve through, for no benefit to the live
    graph. The sweep quarantines every terminal-stage identity rather than
    touching it; :func:`realign_grammar_segments_for_name` is the deliberate,
    operator-named route that reaches a tombstoned identity when a repair to
    it is actually intended.

    Kind is another deterministic projection of the canonical name. The same
    maintenance step refreshes a stale stored kind and records every change in
    the internal ledger so publication and graph state converge on one rule.

    Returns ``{"names_realigned": n}``; the stable result shape remains scoped
    to grammar segments while kind refresh reports through its own log entry.
    """
    from imas_codex.standard_names.ledger import LIVE_NAME

    cols = foundation.services._GRAMMAR_SEGMENT_COLUMNS
    select = ", ".join(f"sn.{c} AS {c}" for c in cols)
    set_clause = (
        ", ".join(f"sn.{c} = b.{c}" for c in cols) + ", sn.updated_at = datetime()"
    )
    batch: list[dict[str, foundation.services.Any]] = []
    with foundation.services.GraphClient() as gc:
        rows = gc.query(
            f"MATCH (sn:StandardName) WHERE {LIVE_NAME} RETURN sn.id AS id, {select}"
        )
        for r in rows:
            parsed = foundation.services._parse_grammar(r["id"])
            # A successful ISN parse always yields a physical_base; an all-None
            # parse means the model rejected the name — leave its stored
            # segments untouched rather than wiping them.
            if not parsed.get("physical_base"):
                continue
            if any(parsed.get(c) != r.get(c) for c in cols):
                batch.append({"id": r["id"], **{c: parsed.get(c) for c in cols}})
        if batch:
            gc.query(
                f"UNWIND $batch AS b MATCH (sn:StandardName {{id: b.id}}) "
                f"SET {set_clause} "
                f"CREATE (change:StandardNameChange {{"
                f"  id: 'sn-change:' + randomUUID(),"
                f"  from_name: sn.id, to_name: sn.id,"
                f"  operation: '{foundation.services._SEGMENT_REALIGN_OPERATION}',"
                f"  reason: 'stored grammar segments disagreed with the "
                f"canonical-name parse', origin: 'graph_reconcile',"
                f"  changed_at: datetime(), internal: true}}) "
                f"MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(change)",
                batch=batch,
            )
    if batch:
        foundation.services.logger.info(
            "reconcile_grammar_segments: realigned %d name(s) to their canonical "
            "id parse",
            len(batch),
        )
    foundation.services.reconcile_standard_name_kinds()
    return {"names_realigned": len(batch)}


def realign_grammar_segments_for_name(
    name: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Realign one name's grammar segment columns with its canonical id.

    The name-scoped counterpart of :func:`reconcile_grammar_segments`: the
    same parse authority and the same write, narrowed to a single id and
    offering a dry run so the drift can be read before it is repaired. The
    sweep only runs as a rotation-wide maintenance step, which leaves a single
    known-drifted name unreachable between rotations; this is that route.

    Every lifecycle stage is in range. A superseded or exhausted identity
    carries the same drifted columns as a live one and is repaired rather than
    refused, and the ``StandardNameChange`` written beside the repair is what
    keeps a rewrite of a tombstoned identity reviewable.

    Args:
        name: Canonical StandardName id to re-parse and realign.
        dry_run: Report the drift without writing to the graph.

    Returns:
        ``{"ok": True, "name", "stage", "dry_run", "drift", "noop"}`` where
        ``drift`` maps each disagreeing segment column to its ``stored`` and
        ``parsed`` values, or ``{"ok": False, "reason": ...}`` for a refusal.
    """
    cols = foundation.services._GRAMMAR_SEGMENT_COLUMNS
    name = (name or "").strip()
    if not name:
        return {"ok": False, "reason": "a standard name is required"}

    select = ", ".join(f"sn.{c} AS {c}" for c in cols)
    with foundation.services.GraphClient() as gc:
        rows = list(
            gc.query(
                "MATCH (sn:StandardName {id: $id}) "
                f"RETURN sn.name_stage AS stage, {select}",
                id=name,
            )
        )
        if not rows:
            return {"ok": False, "reason": f"standard name {name!r} not found"}

        stored = rows[0]
        parsed = foundation.services._parse_grammar(name)
        # A successful ISN parse always yields a physical_base; an all-None
        # parse means the grammar rejected the name, whose segments belong to
        # the quarantine path and are never wiped from here.
        if not parsed.get("physical_base"):
            return {
                "ok": False,
                "reason": (
                    f"the ISN grammar cannot parse {name!r}; its segments are "
                    "owned by the quarantine path, not by this repair"
                ),
            }

        drift = {
            column: {"stored": stored.get(column), "parsed": parsed.get(column)}
            for column in cols
            if parsed.get(column) != stored.get(column)
        }
        result: dict[str, foundation.services.Any] = {
            "ok": True,
            "name": name,
            "stage": stored.get("stage"),
            "dry_run": dry_run,
            "drift": drift,
            "noop": not drift,
            # First segment column, reported on every outcome so an aligned
            # name still shows the value the parse agreed on.
            "physical_base": parsed.get("physical_base"),
        }
        if dry_run or not drift:
            return result

        change_id = f"sn-change:{foundation.services.uuid.uuid4()}"
        set_clause = ", ".join(f"sn.{c} = seg.{c}" for c in cols)
        mutation = list(
            gc.query(
                f"""
                UNWIND [$segments] AS seg
                MATCH (sn:StandardName {{id: $id}})
                SET {set_clause},
                    sn.updated_at = datetime()
                CREATE (change:StandardNameChange {{
                  id: $change_id, from_name: sn.id, to_name: sn.id,
                  operation: $operation, reason: $reason,
                  origin: 'catalog_edit', changed_at: datetime($changed_at),
                  internal: true
                }})
                MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(change)
                RETURN sn.id AS id
                """,
                id=name,
                segments={column: parsed.get(column) for column in cols},
                change_id=change_id,
                operation=foundation.services._SEGMENT_REALIGN_OPERATION,
                reason=foundation.services._segment_drift_reason(drift),
                changed_at=foundation.services.datetime.now(
                    foundation.services.UTC
                ).isoformat(),
            )
        )
        if [row.get("id") for row in mutation] != [name]:
            raise RuntimeError(
                "standard name changed or disappeared during segment realignment"
            )
        result["change_id"] = change_id
        return result


def _segment_drift_reason(drift: dict[str, dict[str, Any]]) -> str:
    """Render a segment drift map as the ledger reason for its repair."""
    moves = ", ".join(
        f"{column}: {values['stored']!r} -> {values['parsed']!r}"
        for column, values in sorted(drift.items())
    )
    return f"stored segments disagreed with the canonical-name parse ({moves})"


def reconcile_standard_name_kinds(gc: Any | None = None) -> dict[str, int]:
    """Refresh live names whose stored kind disagrees with ``derive_kind``.

    The canonical identity is the authority for structural kind. Every changed
    value is written with a linked ``StandardNameChange`` in the same query;
    matching rows are omitted from the write entirely. A compare-and-set guard
    prevents a concurrent update from being overwritten after the read.
    """
    from imas_codex.standard_names.kind_derivation import derive_kind
    from imas_codex.standard_names.ledger import LIVE_NAME

    own = gc is None
    client = gc or foundation.services.GraphClient()
    try:
        rows = client.query(
            f"MATCH (sn:StandardName) WHERE {LIVE_NAME} "
            "RETURN sn.id AS id, sn.kind AS stored_kind"
        )
        updates = [
            {
                "id": str(row["id"]),
                "stored_kind": row.get("stored_kind"),
                "derived_kind": derive_kind(str(row["id"])),
            }
            for row in rows
        ]
        updates = [row for row in updates if row["stored_kind"] != row["derived_kind"]]
        changed = (
            list(
                client.query(
                    """
                    UNWIND $updates AS row
                    MATCH (sn:StandardName {id: row.id})
                    WHERE sn.kind = row.stored_kind
                       OR (sn.kind IS NULL AND row.stored_kind IS NULL)
                    CREATE (change:StandardNameChange {
                      id: 'sn-change:' + randomUUID(),
                      from_name: row.stored_kind,
                      to_name: row.derived_kind,
                      operation: 'refresh_derived_kind',
                      reason: 'stored kind disagreed with canonical-name derivation',
                      origin: 'graph_reconcile',
                      changed_at: datetime(),
                      internal: true
                    })
                    MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(change)
                    SET sn.kind = row.derived_kind,
                        sn.updated_at = datetime()
                    RETURN sn.id AS id
                    ORDER BY id
                    """,
                    updates=updates,
                )
            )
            if updates
            else []
        )
    finally:
        if own:
            client.close()

    if changed:
        foundation.services.logger.info(
            "reconcile_standard_name_kinds: refreshed %d stored kind(s) from "
            "canonical-name derivation",
            len(changed),
        )
    return {"names_refreshed": len(changed)}


def _open_graph_client() -> GraphClient:
    """Create the graph client used by manifest-backed compatibility exports."""
    return foundation.services.GraphClient().__enter__()


_error_sibling_reconcile_apply = functools.partial(
    apply_signed_manifest,
    {},
    authority_adapter="error-sibling-reconcile",
    mutation_kind="set_properties",
    guard_set=(
        "recognized-error-sibling-parent-absence",
        "out-of-allowlist-immutability",
    ),
    reason="orphaned error sibling (parent name deleted)",
    apply=True,
    client_factory=_open_graph_client,
)


def get_standard_name_source_stats(
    source_type: str | None = None,
) -> dict[str, int]:
    """Get StandardNameSource status counts.

    Returns dict mapping status → count. Optionally filtered by source_type.
    """
    with foundation.services.GraphClient() as gc:
        where = ""
        params: dict = {}
        if source_type:
            where = "WHERE sns.source_type = $source_type"
            params["source_type"] = source_type
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource) {where}
            RETURN sns.status AS status, count(sns) AS count
            """,
            **params,
        )
        return {r["status"]: r["count"] for r in result}


def write_run_provenance(
    name_ids: list[str],
    run_id: str,
) -> int:
    """Stamp run provenance fields on StandardName nodes.

    Sets ``last_run_id`` and ``last_run_at`` on every name touched by
    the current ``sn run`` invocation.

    Args:
        name_ids: StandardName ids to stamp.
        run_id: UUID string identifying this ``sn run`` invocation.

    Returns:
        Number of nodes updated.
    """
    if not name_ids:
        return 0

    now = foundation.services.datetime.now(foundation.services.UTC).isoformat()

    with foundation.services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            SET sn.updated_at = datetime(), sn.last_run_id = $rid,
                sn.last_run_at = datetime($ts)
            RETURN count(sn) AS n
            """,
            ids=name_ids,
            rid=run_id,
            ts=now,
        )
        return result[0]["n"] if result else 0


def export_review_comments(
    output_path: str | Path,
    *,
    domain: str | None = None,
) -> int:
    """Dump StandardNameReview node comment data to a JSONL file before ``sn clear``.

    Queries all ``StandardNameReview`` nodes (optionally filtered by the parent
    ``StandardName.physics_domain``) and writes one JSON record per
    line to *output_path*.  Each record contains:

    * ``name`` — ``StandardName.id``
    * ``domain`` — ``StandardName.physics_domain``
    * ``reviewer_model`` — the model that produced the review
    * ``score`` — numeric score (0–1)
    * ``comments_per_dim`` — parsed dict of per-dimension comments
    * ``comments`` — full free-text comment string
    * ``review_axis`` — "name" or "docs"
    * ``generated_at`` — ``StandardName.generated_at`` ISO string
    * ``reviewed_at`` — ``StandardNameReview.reviewed_at`` ISO string

    Parameters
    ----------
    output_path:
        Destination file path.  Parent directories are created if absent.
    domain:
        When provided, restrict to reviews on names with this
        ``physics_domain``.

    Returns
    -------
    Number of StandardNameReview records written (0 if none found).
    """
    output_path = foundation.services.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params: dict[str, foundation.services.Any] = {}
    where_clauses: list[str] = []
    if domain:
        where_clauses.append("sn.physics_domain = $domain")
        params["domain"] = domain

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cypher = f"""
        MATCH (sn:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview)
        {where}
        RETURN sn.id AS name,
               sn.physics_domain AS domain,
               r.reviewer_model AS reviewer_model,
               r.score AS score,
               r.comments_per_dim_json AS comments_per_dim_json,
               r.comments AS comments,
               r.review_axis AS review_axis,
               sn.generated_at AS generated_at,
               r.reviewed_at AS reviewed_at
        ORDER BY sn.id, r.reviewed_at
    """

    with foundation.services.GraphClient() as gc:
        rows = gc.query(cypher, **params)

    if not rows:
        foundation.services.logger.info(
            "export_review_comments: no StandardNameReview nodes found (domain=%s)",
            domain,
        )
        return 0

    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            # Parse comments_per_dim_json if it's a string
            cpd_raw = row.get("comments_per_dim_json")
            if isinstance(cpd_raw, str):
                try:
                    cpd = foundation.services.json.loads(cpd_raw)
                except (foundation.services.json.JSONDecodeError, ValueError):
                    cpd = {}
            elif isinstance(cpd_raw, dict):
                cpd = cpd_raw
            else:
                cpd = {}

            record: dict[str, foundation.services.Any] = {
                "name": row.get("name"),
                "domain": row.get("domain"),
                "reviewer_model": row.get("reviewer_model"),
                "score": row.get("score"),
                "comments_per_dim": cpd,
                "comments": row.get("comments"),
                "review_axis": row.get("review_axis"),
                "generated_at": str(row["generated_at"])
                if row.get("generated_at")
                else None,
                "reviewed_at": str(row["reviewed_at"])
                if row.get("reviewed_at")
                else None,
            }
            fh.write(foundation.services.json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    foundation.services.logger.info(
        "export_review_comments: wrote %d records to %s (domain=%s)",
        count,
        output_path,
        domain,
    )
    return count


_PHASE_TO_STANDARD_NAME_COST_FIELD: dict[str, str] = {
    "generate_name": "llm_cost_generate_name",
    "review_name": "llm_cost_review_name",
    "refine_name": "llm_cost_refine_name",
    "generate_docs": "llm_cost_generate_docs",
    "review_docs": "llm_cost_review_docs",
    "refine_docs": "llm_cost_refine_docs",
}


def create_sn_run_open(
    run_id: str,
    *,
    started_at: Any,
    cost_limit: float,
    min_score: float | None = None,
    invocation: str | None = None,
    invocation_flags: str | None = None,
    invocation_scope: str | None = None,
) -> None:
    """Pre-create an ``SNRun`` node with ``status='started'``.

    Called at the START of ``run_sn_pools`` so that ``(LLMCost)-[:FOR_RUN]->
    (SNRun)`` edges have a target from the first LLM call onward.

    Uses MERGE so repeated calls (e.g. after a retry) are safe.
    Sets ``created_at`` explicitly so the timestamp is never NULL.

    The invocation triple (command line, resolved knobs, resolved scope) is
    written at creation rather than at finalization so it survives a run that
    is killed before it can close itself — which is exactly the run whose
    scope a reader most needs to reconstruct.  See
    :mod:`imas_codex.standard_names.run_invocation`.
    """
    from imas_codex.graph.models import SNRun

    run = SNRun(
        id=run_id,
        started_at=started_at,
        cost_limit=round(cost_limit, 6),
        cost_spent=0.0,
        cost_total=0.0,
        events_total=0,
        min_score=min_score,
        status="started",
        cost_is_exact=True,
        invocation=invocation,
        invocation_flags=invocation_flags,
        invocation_scope=invocation_scope,
    )
    props = run.model_dump(mode="json")
    try:
        with foundation.services.GraphClient() as gc:
            gc.create_nodes("SNRun", [props])
            # Set created_at via Cypher datetime() for Neo4j-native timestamp
            gc.query(
                "MATCH (rr:SNRun {id: $run_id}) SET rr.created_at = datetime()",
                run_id=run_id,
            )
    except Exception as exc:
        foundation.services.logger.error(
            "Failed to pre-create SNRun %s: %s", run_id, exc
        )
        raise


def bump_sn_run_counter(
    run_id: str | None,
    counter: str,
    delta: int = 1,
) -> None:
    """Atomically increment an SNRun counter (best-effort, non-blocking).

    Uses ``coalesce(run.X, 0) + $delta`` so concurrent bumps accumulate
    correctly — Neo4j's single-node write lock serialises the read-modify-
    write within the same transaction, preventing lost updates.

    Parameters
    ----------
    run_id:
        SNRun node id.  If ``None``, the bump is silently skipped (allows
        callers outside a run context to work unchanged).
    counter:
        Property name on the SNRun node to increment (e.g.
        ``"names_composed"``, ``"names_reviewed"``).
    delta:
        Amount to add (default 1).
    """
    if not run_id or delta <= 0:
        return

    _ALLOWED = {
        "names_composed",
        "names_reviewed",
        "names_regenerated",
        "names_enriched",
    }
    if counter not in _ALLOWED:
        foundation.services.logger.warning(
            "bump_sn_run_counter: unknown counter %r — skipping", counter
        )
        return

    try:
        with foundation.services.GraphClient() as gc:
            gc.query(
                f"MATCH (rr:SNRun {{id: $run_id}}) "
                f"SET rr.{counter} = coalesce(rr.{counter}, 0) + $delta",
                run_id=run_id,
                delta=delta,
            )
    except Exception as exc:  # pragma: no cover — defensive
        foundation.services.logger.debug(
            "bump_sn_run_counter(%s, %s, %d) failed: %s",
            run_id,
            counter,
            delta,
            exc,
        )


def update_sn_run_progress(
    run_id: str,
    *,
    cost_spent: float,
    cost_total: float | None = None,
    events_total: int | None = None,
) -> None:
    """Periodic in-progress sync of ``SNRun`` telemetry fields.

    Mirrors the running spend total onto the graph node so ``imas-codex sn
    status`` reflects real progress even when a run is interrupted or
    crashes before :func:`finalize_sn_run` runs.  Best-effort: any graph
    error is logged at DEBUG and swallowed so we never poison the loop.

    When *cost_total* and *events_total* are provided (from the BudgetManager
    batch counter), they are written as absolute values alongside
    ``last_heartbeat = datetime()``.
    """
    set_parts = ["rr.cost_spent = $cost_spent", "rr.last_heartbeat = datetime()"]
    params: dict[str, object] = {
        "run_id": run_id,
        "cost_spent": round(float(cost_spent), 6),
    }
    if cost_total is not None:
        set_parts.append("rr.cost_total = $cost_total")
        params["cost_total"] = round(float(cost_total), 6)
    if events_total is not None:
        set_parts.append("rr.events_total = $events_total")
        params["events_total"] = int(events_total)

    try:
        with foundation.services.GraphClient() as gc:
            gc.query(
                "MATCH (rr:SNRun {id: $run_id}) SET " + ", ".join(set_parts),
                **params,
            )
    except Exception as exc:  # pragma: no cover — defensive
        foundation.services.logger.debug(
            "update_sn_run_progress(%s) failed: %s", run_id, exc
        )


def finalize_sn_run(
    run_id: str,
    *,
    status: str,
    cost_spent: float,
    cost_is_exact: bool = True,
    stopped_at: Any,
    elapsed_s: float | None = None,
    **summary_fields: Any,
) -> None:
    """Update an existing ``SNRun`` node at run end.

    Uses ``MATCH + SET`` (not CREATE) — the node must already exist
    (created by :func:`create_sn_run_open`).

    Aggregates ``cost_total`` and ``events_total`` from LLMCost children
    so the SNRun is an authoritative mirror of the LLMCost ledger.
    ``ended_at`` is set to the current graph timestamp.
    ``elapsed_s`` records wall-clock seconds between *started_at* and
    *stopped_at*; callers should pass the pre-computed value so finalization
    never needs to re-read the start timestamp from the graph.

    ``summary_fields`` may contain any other ``SNRun`` property such as
    ``domains_touched``, ``stop_reason``, ``pipeline_hash``,
    ``names_composed``, ``names_enriched``, etc.
    """
    # First, aggregate cost_total and events_total from LLMCost children
    agg_cost: float = 0.0
    agg_events: int = 0
    try:
        with foundation.services.GraphClient() as gc:
            agg_rows = list(
                gc.query(
                    "MATCH (c:LLMCost {run_id: $rid}) "
                    "RETURN sum(c.llm_cost) AS cost, count(c) AS events",
                    rid=run_id,
                )
            )
            if agg_rows:
                agg_cost = float(agg_rows[0].get("cost") or 0.0)
                agg_events = int(agg_rows[0].get("events") or 0)
    except Exception as exc:  # pragma: no cover — defensive
        foundation.services.logger.warning(
            "finalize_sn_run: LLMCost aggregation failed for %s: %s", run_id, exc
        )

    set_clauses = [
        "rr.status = $status",
        "rr.cost_spent = $cost_spent",
        "rr.cost_is_exact = $cost_is_exact",
        "rr.stopped_at = datetime($stopped_at)",
        "rr.ended_at = datetime()",
        "rr.cost_total = $cost_total",
        "rr.events_total = $events_total",
    ]
    params: dict[str, foundation.services.Any] = {
        "run_id": run_id,
        "status": status,
        "cost_spent": round(cost_spent, 6),
        "cost_is_exact": cost_is_exact,
        "stopped_at": stopped_at
        if isinstance(stopped_at, str)
        else stopped_at.isoformat(),
        "cost_total": round(agg_cost, 6),
        "events_total": agg_events,
    }

    if elapsed_s is not None:
        set_clauses.append("rr.elapsed_s = $elapsed_s")
        params["elapsed_s"] = round(elapsed_s, 3)

    for key, value in summary_fields.items():
        set_clauses.append(f"rr.{key} = ${key}")
        params[key] = value

    cypher = (
        "MATCH (rr:SNRun {id: $run_id}) "
        "SET " + ", ".join(set_clauses) + " "
        "RETURN rr.id AS id"
    )
    try:
        with foundation.services.GraphClient() as gc:
            result = gc.query(cypher, **params)
            if not result:
                foundation.services.logger.warning(
                    "finalize_sn_run: no SNRun found with id=%s", run_id
                )
    except Exception as exc:  # pragma: no cover — defensive
        foundation.services.logger.warning(
            "Failed to finalize SNRun %s: %s", run_id, exc
        )


def mark_orphaned_standard_name_runs_stale(
    *,
    current_run_id: str | None = None,
    max_age_hours: float = 6.0,
) -> int:
    """Finalize SNRun rows that never reached a terminal status.

    ``finalize_sn_run`` runs inside ``run_sn_pools``' ``finally`` block, so a
    clean/exception/SIGINT exit always closes the run. A run left at the
    open ``status='started'``/``'running'`` status therefore means the process
    died before Python could finalize (hard kill, OOM, node loss) — the row is
    an orphan that no live process will ever close.

    This sweep marks such rows ``status='stale'`` with
    ``stop_reason='orphaned_no_finalize'`` so ``sn status`` and provenance
    stop reporting them as in-flight. A run is only considered orphaned when
    its most recent liveness signal (``last_heartbeat``, then ``created_at``,
    then ``started_at``) is older than *max_age_hours* — an active run heartbeats via
    :func:`update_sn_run_progress`, so a generous threshold never touches a
    genuinely-running peer. The current run (``current_run_id``) is always
    excluded.

    Idempotent: a second call matches nothing (the rows are now ``'stale'``).
    Returns the number of runs marked stale.
    """
    with foundation.services.GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (rr:SNRun)
            WHERE rr.status IN ['started', 'running']
              AND ($current_run_id IS NULL OR rr.id <> $current_run_id)
              AND coalesce(
                    rr.last_heartbeat, rr.created_at, rr.started_at, rr.stopped_at
                  )
                  < datetime() - duration({hours: $max_age})
            SET rr.status = 'stale',
                rr.stop_reason = 'orphaned_no_finalize',
                rr.cost_is_exact = false,
                rr.ended_at = datetime()
            RETURN rr.id AS id
            """,
            current_run_id=current_run_id,
            max_age=max_age_hours,
        )
    marked = len(rows or [])
    if marked:
        foundation.services.logger.info(
            "mark_orphaned_standard_name_runs_stale: marked %d orphaned SNRun(s) stale "
            "(no finalize, older than %.1fh)",
            marked,
            max_age_hours,
        )
    return marked


def normalize_stored_standard_name_kinds() -> int:
    """Collapse any stored ``StandardName.kind`` outside the ISN Kind enum.

    The ISN catalog ``Kind`` enum (imas_standard_names) is the authority for
    the closed kind vocabulary. Legacy graph data written before the local
    vocabulary was unified with ISN may still carry a retired kind
    (``eigenfunction`` / ``spectrum``); this sweep rewrites each offending node
    to the value the ISN discriminator accepts via :func:`to_isn_kind` — the
    single source of truth for the collapse (retired kinds map to ``scalar``).

    Idempotent: a second call matches nothing (every node now holds an
    in-enum kind). Returns the number of nodes rewritten.
    """
    from imas_standard_names.models import Kind

    from imas_codex.standard_names.kind_derivation import to_isn_kind

    allowed = [k.value for k in Kind]
    with foundation.services.GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.kind IS NOT NULL AND NOT sn.kind IN $allowed
            RETURN sn.id AS id, sn.kind AS kind
            """,
            allowed=allowed,
        )
        updates = [
            {"id": r["id"], "kind": to_isn_kind(r["kind"])} for r in (rows or [])
        ]
        if updates:
            gc.query(
                """
                UNWIND $updates AS u
                MATCH (sn:StandardName {id: u.id})
                SET sn.kind = u.kind, sn.updated_at = datetime()
                """,
                updates=updates,
            )
    rewritten = len(updates)
    if rewritten:
        foundation.services.logger.info(
            "normalize_stored_standard_name_kinds: collapsed %d node(s) with a "
            "retired/off-enum kind to the ISN-canonical value",
            rewritten,
        )
    return rewritten


def backfill_sn_run_telemetry() -> list[dict[str, Any]]:
    """One-shot backfill of cost_total/events_total on existing SNRun nodes.

    Idempotent: only touches nodes where cost_total IS NULL.
    Aggregates from LLMCost children and writes the result.
    Returns a list of dicts with id, cost, events for each patched node.
    """
    cypher = """
    MATCH (r:SNRun)
    WHERE r.cost_total IS NULL
    OPTIONAL MATCH (c:LLMCost {run_id: r.id})
    WITH r, sum(c.llm_cost) AS cost_sum, count(c) AS event_count
    SET r.cost_total = coalesce(r.cost_total, cost_sum, 0.0),
        r.events_total = coalesce(r.events_total, event_count, 0)
    RETURN r.id AS id, r.cost_total AS cost, r.events_total AS events
    """
    with foundation.services.GraphClient() as gc:
        return list(gc.query(cypher))


_PHASE_TO_POOL: dict[str, str] = {
    "generate": "compose",
    "generate_name": "compose",
    "compose": "compose",
    "regen": "refine_name",
    "refine_name": "refine_name",
    "refine_docs": "refine_docs",
    "validate": "validate",
    "validate_name": "validate",
    "review_names": "review",
    "review_docs": "review",
    "review": "review",
    "enrich": "enrich",
    "enrich_links": "enrich",
}

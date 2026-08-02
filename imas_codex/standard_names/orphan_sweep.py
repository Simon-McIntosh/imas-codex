"""Stage-aware orphan recovery for the SN refine pipeline.

Periodically reverts StandardName nodes stuck in transient stages (e.g.
'refining') that have stale claimed_at timestamps. Mirrors the discovery
CLI orphan-sweep pattern but applies stage-aware predecessor reverts.

Predecessor-stage mapping
-------------------------
``name_stage='refining'``  →  revert to ``'reviewed'``
``docs_stage='refining'``  →  revert to ``'reviewed'``

For defense in depth, also clear stale ``claim_token``/``claimed_at``
on non-refining StandardName and StandardNameSource nodes.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Final

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep queries — (label, cypher) pairs. Each RETURN alias is ``n``.
# ---------------------------------------------------------------------------

_SWEEP_QUERIES: Final[list[tuple[str, str]]] = [
    (
        "name_refining",
        """
        MATCH (sn:StandardName)
        WHERE sn.name_stage = 'refining'
          AND (sn.claimed_at IS NULL
               OR sn.claimed_at < datetime() - duration({seconds: $timeout_s}))
        SET sn.name_stage = 'reviewed',
            sn.claim_token = null,
            sn.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
    (
        "docs_refining",
        """
        MATCH (sn:StandardName)
        WHERE sn.docs_stage = 'refining'
          AND (sn.claimed_at IS NULL
               OR sn.claimed_at < datetime() - duration({seconds: $timeout_s}))
        SET sn.docs_stage  = 'reviewed',
            sn.claim_token = null,
            sn.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
    (
        "stale_token_sn",
        """
        MATCH (sn:StandardName)
        WHERE sn.claim_token IS NOT NULL
          AND sn.claimed_at IS NOT NULL
          AND sn.claimed_at < datetime() - duration({seconds: $timeout_s})
          AND NOT sn.name_stage = 'refining'
          AND NOT sn.docs_stage = 'refining'
        SET sn.claim_token = null,
            sn.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
    (
        "stale_token_source",
        """
        MATCH (s:StandardNameSource)
        WHERE s.claim_token IS NOT NULL
          AND s.claimed_at IS NOT NULL
          AND s.claimed_at < datetime() - duration({seconds: $timeout_s})
        SET s.claim_token = null,
            s.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
    (
        # Terminal-classify compose sources that have exhausted their claim
        # budget.  A source whose batch repeatedly fails (LLM omits it, the
        # batch errors, or its candidate is grammar-rejected) returns to
        # ``status='extracted'`` and is re-claimed forever — each claim bumps
        # ``attempt_count`` (claim_generate_name_batch) but nothing ever
        # transitions it out of the claimable pool, so ``total_processed`` stays
        # flat and the run wedges on the residue (full-DD build 2026-06-20).
        # Marking it ``failed`` here (counter-agnostic: ``failed`` is not
        # ``extracted``, so every pending/idle counter drops it) lets the idle
        # watchdog exit cleanly instead of the stall-guard.  Revived by
        # ``--reset-to extracted`` (clears attempt_count) once compose improves.
        "compose_attempt_cap",
        """
        MATCH (s:StandardNameSource)
        WHERE s.status = 'extracted'
          AND coalesce(s.attempt_count, 0) >= $max_compose_attempts
        SET s.status     = 'failed',
            s.failed_at  = datetime(),
            s.last_error = 'compose claim-attempt cap reached',
            s.claim_token = null,
            s.claimed_at  = null
        RETURN count(*) AS n
        """,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _orphan_sweep_tick(*, timeout_s: int) -> dict[str, int]:
    """Run one full orphan-sweep pass (synchronous).

    Executes all four sweep queries in separate transactions and returns
    a mapping of sweep label → number of nodes reverted/cleared.

    Args:
        timeout_s: Age threshold in seconds.  Nodes whose ``claimed_at``
            is older than ``timeout_s`` seconds are considered orphaned.

    Returns:
        ``{"name_refining": n, "docs_refining": n, "stale_token_sn": n,
        "stale_token_source": n}``
    """
    from imas_codex.standard_names.graph_ops import _MAX_COMPOSE_CLAIM_ATTEMPTS

    counts: dict[str, int] = {}
    with GraphClient() as gc:
        for label, cypher in _SWEEP_QUERIES:
            rows = gc.query(
                cypher,
                timeout_s=timeout_s,
                max_compose_attempts=_MAX_COMPOSE_CLAIM_ATTEMPTS,
            )
            counts[label] = rows[0]["n"] if rows else 0
    return counts


async def run_orphan_sweep_loop(
    *,
    interval_s: int,
    timeout_s: int,
    stop_event: asyncio.Event,
) -> None:
    """Background coroutine that periodically reverts orphaned claims.

    Loops every *interval_s* seconds, delegating the actual DB work to
    :func:`_orphan_sweep_tick` via ``asyncio.to_thread``.  Exits cleanly
    when *stop_event* is set (checked both before each tick and during the
    sleep phase).

    Args:
        interval_s: How often to run a sweep pass, in seconds.
        timeout_s: Age threshold passed to :func:`_orphan_sweep_tick`.
        stop_event: Cooperative shutdown signal; shared with worker pools.
    """
    logger.info(
        "Orphan sweep loop started (interval=%ds, timeout=%ds)",
        interval_s,
        timeout_s,
    )

    while not stop_event.is_set():
        try:
            counts = await asyncio.to_thread(
                _orphan_sweep_tick,
                timeout_s=timeout_s,
            )
            total = sum(counts.values())
            if total:
                logger.warning(
                    "Orphan sweep reverted %d claims: %s",
                    total,
                    counts,
                )
            else:
                logger.debug("Orphan sweep: no stuck claims found")
        except Exception:  # noqa: BLE001
            logger.exception("Orphan sweep tick failed; continuing")

        # Seed parent component sources for newly tagged parents
        try:
            from imas_codex.standard_names.graph_ops import seed_parent_sources

            parent_count = await asyncio.to_thread(seed_parent_sources)
            if parent_count:
                logger.info("Seeded %d parent component sources", parent_count)
        except Exception:  # noqa: BLE001
            logger.exception("Parent source seeding failed; continuing")

        # Sleep for interval_s, but wake early if stop_event fires.
        try:
            await asyncio.wait_for(
                asyncio.shield(stop_event.wait()),
                timeout=interval_s,
            )
        except TimeoutError:
            pass  # Normal path — interval elapsed, loop again.

    logger.info("Orphan sweep loop stopped")


def refresh_manifest_drain_scope(
    drain_scope_id: str, *, gc: Any | None = None
) -> dict[str, int]:
    """Refresh only one drain lease without changing worker claim ownership."""
    own = gc is None
    client = GraphClient() if own else gc
    try:
        rows = client.query(
            """
            CALL {
              MATCH (s:StandardNameSource {drain_scope_id: $scope_id})
              SET s.drain_scope_claimed_at = datetime()
              RETURN count(s) AS sources
            }
            CALL {
              MATCH (sn:StandardName {drain_scope_id: $scope_id})
              SET sn.drain_scope_claimed_at = datetime()
              RETURN count(sn) AS names
            }
            RETURN sources, names
            """,
            scope_id=drain_scope_id,
        )
        return dict(rows[0]) if rows else {"sources": 0, "names": 0}
    finally:
        if own:
            client.close()


def recover_manifest_drain_scope(
    drain_scope_id: str,
    *,
    scope_timeout_s: int,
    worker_timeout_s: int,
    paths: list[str] | None = None,
    gc: Any | None = None,
) -> dict[str, int]:
    """Recover the exact-path portion of one expired drain lease.

    Recovery is all-or-nothing for the lease: if any scoped node has a fresh
    drain heartbeat, this function performs no writes.  A stale scoped node in
    a refining stage is reverted only when its independent worker claim is also
    stale or absent.  Fresh external claim tokens are never cleared.
    """
    scope_cutoff = f"PT{scope_timeout_s}S"
    worker_cutoff = f"PT{worker_timeout_s}S"
    own = gc is None
    client = GraphClient() if own else gc
    try:
        source_ids = [f"dd:{path}" for path in paths] if paths is not None else None
        with client.session() as session:
            tx = session.begin_transaction()
            try:
                probe = list(
                    tx.run(
                        """
                        MATCH (source:StandardNameSource)
                        WHERE source.drain_scope_id = $scope_id
                          AND ($source_ids IS NULL OR source.id IN $source_ids)
                        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(name:StandardName)
                        OPTIONAL MATCH (name)-[:HAS_PARENT*1..]->(parent:StandardName)
                        WITH collect(DISTINCT source) + collect(DISTINCT name)
                             + collect(DISTINCT parent) AS candidates
                        UNWIND [node IN candidates
                                WHERE node IS NOT NULL
                                  AND node.drain_scope_id = $scope_id] AS node
                        RETURN count(node) AS total,
                               count(CASE
                                 WHEN node.drain_scope_claimed_at IS NULL
                                   OR node.drain_scope_claimed_at < datetime()
                                        - duration($scope_cutoff)
                                 THEN 1 END) AS stale
                        """,
                        scope_id=drain_scope_id,
                        scope_cutoff=scope_cutoff,
                        source_ids=source_ids,
                    )
                )
                total = int(probe[0]["total"]) if probe else 0
                stale = int(probe[0]["stale"]) if probe else 0
                if total == 0 or stale != total:
                    tx.rollback()
                    return {"sources": 0, "names": 0, "refining_reverted": 0}

                names = list(
                    tx.run(
                        """
                        MATCH (source:StandardNameSource)
                        WHERE source.drain_scope_id = $scope_id
                          AND ($source_ids IS NULL OR source.id IN $source_ids)
                        MATCH (source)-[:PRODUCED_NAME]->(sn:StandardName)
                        WHERE sn.drain_scope_id = $scope_id
                        OPTIONAL MATCH (sn)-[:HAS_PARENT*1..]->(parent:StandardName)
                        WITH collect(DISTINCT sn) + collect(DISTINCT parent) AS names
                        UNWIND [entry IN names WHERE entry IS NOT NULL
                                AND entry.drain_scope_id = $scope_id] AS sn
                        WITH sn,
                          (sn.name_stage = 'refining'
                           OR sn.docs_stage = 'refining') AS was_refining,
                          (sn.claim_token IS NULL OR sn.claimed_at IS NULL
                           OR sn.claimed_at < datetime()
                                - duration($worker_cutoff)) AS worker_stale
                        SET sn.name_stage = CASE
                              WHEN sn.name_stage = 'refining' AND worker_stale
                              THEN 'reviewed' ELSE sn.name_stage END,
                            sn.docs_stage = CASE
                              WHEN sn.docs_stage = 'refining' AND worker_stale
                              THEN 'reviewed' ELSE sn.docs_stage END,
                            sn.claim_token = CASE WHEN worker_stale THEN null
                              ELSE sn.claim_token END,
                            sn.claimed_at = CASE WHEN worker_stale THEN null
                              ELSE sn.claimed_at END
                        REMOVE sn.drain_scope_id, sn.drain_scope_claimed_at
                        RETURN count(sn) AS names,
                               count(CASE
                                 WHEN was_refining AND worker_stale THEN 1 END)
                                 AS refining_reverted
                        """,
                        scope_id=drain_scope_id,
                        worker_cutoff=worker_cutoff,
                        source_ids=source_ids,
                    )
                )
                sources = list(
                    tx.run(
                        """
                        MATCH (s:StandardNameSource {drain_scope_id: $scope_id})
                        WHERE $source_ids IS NULL OR s.id IN $source_ids
                        REMOVE s.drain_scope_id, s.drain_scope_claimed_at,
                               s.drain_scope_actionable
                        RETURN count(s) AS sources
                        """,
                        scope_id=drain_scope_id,
                        source_ids=source_ids,
                    )
                )
                tx.commit()
                return {
                    "sources": int(sources[0]["sources"]) if sources else 0,
                    "names": int(names[0]["names"]) if names else 0,
                    "refining_reverted": (
                        int(names[0]["refining_reverted"]) if names else 0
                    ),
                }
            except BaseException:
                with suppress(Exception):
                    tx.rollback()
                raise
    finally:
        if own:
            client.close()


def find_expired_manifest_drain_scopes(
    paths: list[str], *, scope_timeout_s: int, gc: Any | None = None
) -> list[str]:
    """Return expired scope owners intersecting the exact DD source set."""
    source_ids = [f"dd:{path}" for path in paths]
    own = gc is None
    client = GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (source:StandardNameSource)
            WHERE source.id IN $source_ids
              AND source.drain_scope_id IS NOT NULL
              AND (source.drain_scope_claimed_at IS NULL
                   OR source.drain_scope_claimed_at < datetime()
                        - duration($scope_cutoff))
            RETURN DISTINCT source.drain_scope_id AS scope_id
            ORDER BY scope_id
            """,
            source_ids=source_ids,
            scope_cutoff=f"PT{scope_timeout_s}S",
        )
        return [row["scope_id"] for row in rows]
    finally:
        if own:
            client.close()


async def run_manifest_drain_heartbeat_loop(
    *, drain_scope_id: str, interval_s: int, stop_event: asyncio.Event
) -> None:
    """Refresh a bounded drain lease until cooperative shutdown."""
    while not stop_event.is_set():
        await asyncio.to_thread(refresh_manifest_drain_scope, drain_scope_id)
        try:
            await asyncio.wait_for(
                asyncio.shield(stop_event.wait()), timeout=interval_s
            )
        except TimeoutError:
            pass

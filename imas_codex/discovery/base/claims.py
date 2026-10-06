"""Common claim coordination for parallel discovery engines.

All discovery modules (paths, wiki, signals, files) use the same claim pattern:

1. Atomically SET claimed_at = datetime() on unclaimed/stale nodes
2. Process the claimed nodes
3. On completion: update status (claimed_at cleared implicitly or explicitly)
4. On error: release claim via SET claimed_at = null

Stale claims (older than timeout) are automatically recovered by other workers,
making the system safe for parallel execution across CLI instances.

Anti-deadlock pattern for claim functions:
- ORDER BY rand() to avoid deterministic lock ordering collisions
- claim_token (UUID) two-step verify to handle race conditions
- @retry_on_deadlock decorator for transient Neo4j deadlock errors

Usage::

    from imas_codex.discovery.base.claims import (
        DEFAULT_CLAIM_TIMEOUT_SECONDS,
        release_claim,
        release_claims_batch,
        reset_stale_claims,
        retry_on_deadlock,
    )

    @retry_on_deadlock()
    def claim_items(facility, limit=10):
        token = str(uuid.uuid4())
        with GraphClient() as gc:
            gc.query("... ORDER BY rand() LIMIT $limit SET n.claim_token = $token ...", ...)
            return list(gc.query("MATCH (n {claim_token: $token}) RETURN ...", ...))
"""

from __future__ import annotations

import functools
import logging
import random
import time
import uuid
from collections.abc import Mapping
from typing import Any

from neo4j.exceptions import TransientError

logger = logging.getLogger(__name__)

DEFAULT_CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes

# Retry configuration for Neo4j transient errors (deadlocks)
MAX_RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 0.1  # seconds
RETRY_MAX_DELAY = 2.0  # seconds


def retry_on_deadlock(
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
):
    """Decorator to retry functions on Neo4j transient errors (e.g., deadlocks).

    Uses exponential backoff with jitter to reduce contention.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TransientError as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.5)
                        sleep_time = delay + jitter
                        logger.debug(
                            "%s: transient error (attempt %d/%d), "
                            "retrying in %.2fs: %s",
                            func.__name__,
                            attempt + 1,
                            max_attempts,
                            sleep_time,
                            e,
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.warning(
                            "%s: transient error after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            e,
                        )
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def reset_stale_claims(
    label: str,
    facility: str,
    *,
    timeout_seconds: int = DEFAULT_CLAIM_TIMEOUT_SECONDS,
    facility_field: str = "facility_id",
    claimed_field: str = "claimed_at",
    silent: bool = False,
) -> int:
    """Release claims older than timeout_seconds for a node type.

    Uses timeout-based recovery so multiple CLI instances can run
    concurrently without wiping each other's active claims.

    Args:
        label: Node label (e.g., ``"CodeFile"``, ``"FacilityPath"``)
        facility: Facility ID
        timeout_seconds: Age threshold for orphaned claims
        facility_field: Property containing facility ID
        claimed_field: Property name for the claim timestamp
        silent: Suppress logging

    Returns:
        Number of claims released
    """
    from imas_codex.graph import GraphClient

    cutoff = f"PT{timeout_seconds}S"
    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (n:{label} {{{facility_field}: $facility}})
            WHERE n.{claimed_field} IS NOT NULL
              AND (n.{claimed_field} < datetime() - duration($cutoff)
                   OR n.{claimed_field} > datetime())
            SET n.{claimed_field} = null
            RETURN count(n) AS reset_count
            """,
            facility=facility,
            cutoff=cutoff,
        )
        count = result[0]["reset_count"] if result else 0

    if count and not silent:
        logger.info(
            "Released %d orphaned %s claims older than %ds for %s",
            count,
            label,
            timeout_seconds,
            facility,
        )

    return count


def release_claim(
    label: str,
    node_id: str,
    *,
    claimed_field: str = "claimed_at",
    token_field: str | None = None,
) -> None:
    """Release claim on a single node by clearing its claim timestamp.

    Args:
        label: Node label (e.g., ``"CodeFile"``)
        node_id: Node ID to release
        claimed_field: Property holding the claim timestamp
        token_field: Claim-token property to clear as well, when set
    """
    from imas_codex.graph import GraphClient

    clears = [f"n.{claimed_field} = null"]
    if token_field:
        clears.append(f"n.{token_field} = null")

    try:
        with GraphClient() as gc:
            gc.query(
                f"""
                MATCH (n:{label} {{id: $id}})
                SET {", ".join(clears)}
                """,
                id=node_id,
            )
    except Exception as e:
        logger.warning("Failed to release %s claim for %s: %s", label, node_id, e)


def release_claims_batch(
    label: str,
    node_ids: list[str],
    *,
    claimed_field: str = "claimed_at",
    token_field: str | None = None,
) -> int:
    """Release claims on multiple nodes by clearing their claim timestamps.

    Args:
        label: Node label (e.g., ``"CodeFile"``)
        node_ids: Node IDs to release
        claimed_field: Property holding the claim timestamp
        token_field: Claim-token property to clear as well, when set

    Returns:
        Number of claims released
    """
    from imas_codex.graph import GraphClient

    if not node_ids:
        return 0

    clears = [f"n.{claimed_field} = null"]
    if token_field:
        clears.append(f"n.{token_field} = null")

    try:
        with GraphClient() as gc:
            result = gc.query(
                f"""
                UNWIND $ids AS nid
                MATCH (n:{label} {{id: nid}})
                WHERE n.{claimed_field} IS NOT NULL
                SET {", ".join(clears)}
                RETURN count(n) AS released
                """,
                ids=node_ids,
            )
            return result[0]["released"] if result else 0
    except Exception as e:
        logger.warning("Failed to release %s claims: %s", label, e)
        return 0


# =============================================================================
# Generic claim / pending routines
# =============================================================================
# Discovery domains differ only in the node label they claim, the status
# predicate a row must satisfy, the claim property names and the fields they
# read back. ``claim_batch`` and ``has_pending`` take exactly those as
# arguments, so a domain's claim and has-work helpers carry no query text of
# their own beyond their predicate.


@retry_on_deadlock()
def claim_batch(
    label: str,
    *,
    facility: str,
    status_predicate: str = "TRUE",
    status_params: Mapping[str, Any] | None = None,
    batch_size: int = 20,
    facility_field: str = "facility_id",
    claimed_field: str = "claimed_at",
    token_field: str = "claim_token",
    return_fields: str = "n.id AS id",
    return_clause: str = "",
    timeout_seconds: int = DEFAULT_CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim a random batch of unclaimed or stale rows and read them back.

    Sets ``claimed_field`` and ``token_field`` on up to ``batch_size`` rows
    matching ``status_predicate`` whose claim is absent or older than
    ``timeout_seconds``, then returns the rows carrying this call's token. The
    token two-step verify and ``ORDER BY rand()`` are the anti-deadlock pattern
    shared by every domain; ``retry_on_deadlock`` retries a transient error.

    Args:
        label: Node label to claim (e.g., ``"SignalSource"``).
        facility: Facility ID the rows belong to.
        status_predicate: Cypher predicate a row must satisfy to be claimable,
            written against the node variable ``n``.
        status_params: Named parameters bound by ``status_predicate``.
        batch_size: Maximum rows to claim.
        facility_field: Property holding the facility ID.
        claimed_field: Property holding the claim timestamp.
        token_field: Property holding this claim's token.
        return_fields: ``RETURN`` projection read back for each claimed row.
        return_clause: Optional Cypher fragment (e.g. an ``OPTIONAL MATCH``)
            inserted between the token match and the ``RETURN``.
        timeout_seconds: Age at which an existing claim may be reclaimed.

    Returns:
        One mapping per claimed row, projected by ``return_fields``.
    """
    from imas_codex.graph import GraphClient

    cutoff = f"PT{timeout_seconds}S"
    token = str(uuid.uuid4())
    params: dict[str, Any] = {
        "facility": facility,
        "batch_size": batch_size,
        "cutoff": cutoff,
        "token": token,
    }
    if status_params:
        params.update(status_params)

    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (n:{label} {{{facility_field}: $facility}})
            WHERE {status_predicate}
              AND (n.{claimed_field} IS NULL
                   OR n.{claimed_field} < datetime() - duration($cutoff))
            WITH n ORDER BY rand() LIMIT $batch_size
            SET n.{claimed_field} = datetime(),
                n.{token_field} = $token
            """,
            **params,
        )

        result = gc.query(
            f"""
            MATCH (n:{label} {{{facility_field}: $facility,
                                 {token_field}: $token}})
            {return_clause}
            RETURN {return_fields}
            """,
            facility=facility,
            token=token,
        )
        return list(result)


def has_pending(
    label: str,
    *,
    facility: str,
    status_predicate: str = "TRUE",
    status_params: Mapping[str, Any] | None = None,
    facility_field: str = "facility_id",
) -> bool:
    """Return whether any row matching ``status_predicate`` remains unclaimed-upon.

    The predicate carries no claim clause: it names the work state itself, so
    the answer is independent of which rows currently hold a claim.
    """
    from imas_codex.graph import GraphClient

    params: dict[str, Any] = {"facility": facility}
    if status_params:
        params.update(status_params)

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (n:{label} {{{facility_field}: $facility}})
            WHERE {status_predicate}
            RETURN count(n) > 0 AS has_work
            """,
            **params,
        )
        return result[0]["has_work"] if result else False

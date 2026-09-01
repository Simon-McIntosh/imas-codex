"""Execution policy for exact-identity Standard Name rescores.

A rescore is one new name-review quorum draw.  It deliberately excludes every
producer and refinement pool: a below-threshold result is returned on the same
node instead of authorizing a successor identity.
"""

from __future__ import annotations

from typing import Protocol


class _NamedPool(Protocol):
    name: str


_RESCORE_RUN_PREFIX = "sn-rescore-"
_RESCORE_POOL = "review_name"


def is_rescore_scope(scope_run_id: str | None) -> bool:
    """Return whether *scope_run_id* identifies the exact rescore operator."""
    return bool(scope_run_id and scope_run_id.startswith(_RESCORE_RUN_PREFIX))


def apply_rescore_pool_contract[PoolT: _NamedPool](
    pools: list[PoolT], *, scope_run_id: str | None
) -> list[PoolT]:
    """Restrict an exact rescore scope to its mandatory name-review pool."""
    if not is_rescore_scope(scope_run_id):
        return pools
    selected = [pool for pool in pools if pool.name == _RESCORE_POOL]
    if len(selected) != 1:
        raise ValueError("rescore scope requires exactly one review_name pool")
    return selected


def classify_rescore_budget_stop(
    *, scope_run_id: str | None, budget_saturated: bool
) -> str | None:
    """Classify an unfundable mandatory rescore quorum as exhausted budget."""
    if is_rescore_scope(scope_run_id) and budget_saturated:
        return "budget_exhausted"
    return None

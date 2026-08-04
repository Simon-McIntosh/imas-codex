"""Phase constants and ``--only`` skip-flag helper for ``sn run``.

Provides:
- ``TURN_PHASES`` — valid ``--only`` choices for CLI validation.
- ``skip_flags_from_only()`` — derives per-pool skip flags from a phase selection.
"""

from __future__ import annotations

from imas_codex.standard_names.pool_registry import POOL_NAMES

# Maps an --only value to the set of turn-level phases to keep running.
# Everything outside the set is skipped.
_ONLY_TO_ACTIVE: dict[str, set[str]] = {
    "reconcile": {"reconcile"},
    # 'attach' is a focused one-shot: run_sn_pools short-circuits to the
    # DD-edge + source_paths reconcile only. Like reconcile, it runs no pools.
    "attach": {"reconcile"},
    "extract": {"generate", "enrich"},
    "compose": {"generate", "enrich"},
    "validate": {"generate", "enrich"},
    "consolidate": {"generate", "enrich"},
    "persist": {"generate", "enrich"},
    "enrich": {"enrich"},
    "review": {"review_names", "review_docs"},
    "review_names": {"review_names"},
    "review_docs": {"review_docs"},
    "link": {"link"},
}

# A single-quorum action needs a pool boundary, not the broader name-review
# axis. Derive the eligible pool identifiers from the canonical pool registry;
# the action vocabulary is policy, while the pool-name universe remains owned
# by ``pool_registry.POOL_NAMES``.
_NAME_REVIEW_ACTIONS = frozenset({"review", "refine"})
_EXACT_POOL_SELECTORS: dict[str, str] = {
    pool_name: pool_name
    for pool_name in POOL_NAMES
    if pool_name.endswith("_name")
    and pool_name.removesuffix("_name") in _NAME_REVIEW_ACTIONS
}

# Valid --only choices (CLI enforces this set). Broad phase selectors preserve
# their historical multi-pool meaning; exact selectors name one worker pool.
TURN_PHASES: tuple[str, ...] = (*_ONLY_TO_ACTIVE, *_EXACT_POOL_SELECTORS)


def exact_pool_from_only(only_phase: str | None) -> str | None:
    """Resolve an exact pool selector while preserving broad phase selectors."""
    if only_phase is None or only_phase in _ONLY_TO_ACTIVE:
        return None
    try:
        return _EXACT_POOL_SELECTORS[only_phase]
    except KeyError as exc:
        raise ValueError(f"unknown --only selector: {only_phase}") from exc


def skip_flags_from_only(only_phase: str | None) -> dict[str, bool]:
    """Derive per-phase skip flags from an ``--only`` selection.

    Returns a dict of ``skip_*`` keys that should be set to ``True``
    when *only_phase* is active.  When *only_phase* is ``None``,
    returns an empty dict (no overrides).
    """
    if only_phase is None:
        return {}

    if only_phase in _EXACT_POOL_SELECTORS:
        active = {"review_names"}
    else:
        active = _ONLY_TO_ACTIVE.get(only_phase, set())
    return {
        "skip_generate": "generate" not in active,
        "skip_enrich": "enrich" not in active,
        "skip_review": "review_names" not in active and "review_docs" not in active,
        "skip_regen": "generate" not in active,
    }

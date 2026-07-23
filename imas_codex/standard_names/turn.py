"""Phase constants and ``--only`` skip-flag helper for ``sn run``.

Provides:
- ``TURN_PHASES`` — valid ``--only`` choices for CLI validation.
- ``skip_flags_from_only()`` — derives per-pool skip flags from a phase selection.
"""

from __future__ import annotations

# Valid --only phase choices (CLI enforces this set).
TURN_PHASES: tuple[str, ...] = (
    "reconcile",
    "attach",
    "extract",
    "compose",
    "validate",
    "consolidate",
    "persist",
    "review",
    "review_names",
    "review_docs",
    "link",
)

# Maps an --only value to the set of turn-level phases to keep running.
# Everything outside the set is skipped.
_ONLY_TO_ACTIVE: dict[str, set[str]] = {
    "reconcile": {"reconcile"},
    # 'attach' is a focused one-shot: run_sn_pools short-circuits to the
    # DD-edge + source_paths reconcile only. Like reconcile, it runs no pools.
    "attach": {"reconcile"},
    "extract": {"generate"},
    "compose": {"generate"},
    "validate": {"generate"},
    "consolidate": {"generate"},
    "persist": {"generate"},
    "review": {"review_names", "review_docs"},
    "review_names": {"review_names"},
    "review_docs": {"review_docs"},
    "link": {"link"},
}


def skip_flags_from_only(only_phase: str | None) -> dict[str, bool]:
    """Derive per-phase skip flags from an ``--only`` selection.

    Returns a dict of ``skip_*`` keys that should be set to ``True``
    when *only_phase* is active.  When *only_phase* is ``None``,
    returns an empty dict (no overrides).
    """
    if only_phase is None:
        return {}

    active = _ONLY_TO_ACTIVE.get(only_phase, set())
    return {
        "skip_generate": "generate" not in active,
        "skip_enrich": "generate" not in active,  # enrich follows generate
        "skip_review": "review_names" not in active and "review_docs" not in active,
        "skip_regen": "generate" not in active,
    }

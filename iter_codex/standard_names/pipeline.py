"""Which maintenance a scoped ``sn run`` may perform.

A run given a scope identifier is a request about that scope's names.  The
pipeline's graph-wide maintenance passes rewrite rows the caller never named —
source bindings, catalog statuses, document links, derived parents — so a
scoped run that starts them changes names outside its scope.  The case that
mattered was a review of one edit's names, which returned 24 unrelated names'
sources to the pool.

The CLI resolves that decision here, once, after the scope flags have become
concrete focus paths, exact names and a run id.  Keeping the rule in one
function lets the command and its tests judge the same policy instead of
repeating the condition, and makes a scope without the maintenance bypass
impossible to request by omission.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: ``--only`` selectors that are themselves graph-wide maintenance.  They have
#: no notion of a scoped subset, so a scoped run refuses rather than quietly
#: performing the global pass.
GRAPH_WIDE_ONLY_PHASES = frozenset({"attach", "reconcile", "validate", "link"})


@dataclass(frozen=True)
class ScopeMaintenancePolicy:
    """The maintenance a run may perform, given the scope it was asked for."""

    scoped: bool
    bypass_global_maintenance: bool
    refusal: str | None


def resolve_scope_maintenance(
    *,
    scope_run_id: str | None = None,
    focus_paths: Sequence[str] = (),
    exact_names: Sequence[str] = (),
    edits_only: bool = False,
    only_phase: str | None = None,
    reset_to: str | None = None,
    revalidate: bool = False,
) -> ScopeMaintenancePolicy:
    """Decide whether *this* run may write outside its scope.

    An unscoped run is the catalogue-wide run and keeps the full maintenance
    sequence.  Any scope identifier — a run id, focus paths, exact names, or
    the edits predicate — promises the run is about those names, so the
    graph-wide passes are bypassed whether or not ``--skip-global-maintenance``
    was also given.

    Actions that cannot be bounded by a scope are refused with a reason rather
    than silently widened: a maintenance-only ``--only`` selector, and the
    graph-wide rewrites ``--reset-to`` (without focus paths to bound it) and
    ``--revalidate``.
    """
    scoped = bool(scope_run_id or focus_paths or exact_names or edits_only)
    if not scoped:
        return ScopeMaintenancePolicy(
            scoped=False,
            bypass_global_maintenance=False,
            refusal=None,
        )

    if only_phase in GRAPH_WIDE_ONLY_PHASES:
        return ScopeMaintenancePolicy(
            scoped=True,
            bypass_global_maintenance=False,
            refusal=(
                f"--only {only_phase} is a graph-wide maintenance pass and "
                "cannot be combined with a scoped run; drop the scope "
                "(--scope-run-id/--focus/--name/--edits) or the --only phase"
            ),
        )

    if reset_to is not None and not focus_paths:
        return ScopeMaintenancePolicy(
            scoped=True,
            bypass_global_maintenance=False,
            refusal=(
                "--reset-to without --focus/--batch rewrites every matching "
                "name in the graph, so it cannot run under a scope; add "
                "--focus/--batch to bound it, or drop the scope"
            ),
        )

    if revalidate:
        return ScopeMaintenancePolicy(
            scoped=True,
            bypass_global_maintenance=False,
            refusal=(
                "--revalidate clears the whole validation backlog, so it "
                "cannot run under a scope; drop the scope or the --revalidate "
                "flag"
            ),
        )

    return ScopeMaintenancePolicy(
        scoped=True,
        bypass_global_maintenance=True,
        refusal=None,
    )

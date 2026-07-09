"""Idempotent source-drift refresh — re-refine names when their DD source changes.

Wired into ``sn run``. Each standard name records the DD-source snapshot (unit +
documentation) it was last built against, stored as scalar properties on the
``StandardName`` node (``source_unit``, ``source_documentation``). Every run
compares that snapshot to the *live* ``IMASNode`` the name is anchored to and,
for any drift, steers a **refine** pass (via :func:`apply_edit`, hint mode,
``axis='docs'``) whose reason carries the precise DD delta — the existing
refine/review pools then rewrite the docs against the corrected source, with the
current name/docs and the original ISN context already supplied by those prompts.
After steering, the snapshot is re-stamped, so a subsequent run with no further
DD change claims nothing (idempotent).

The refresh only ever *steers existing* names — it never composes fresh names.
Names in a terminal state (``superseded``) are skipped. Because the change signal
is a content comparison (not a timestamp), the pass is safe to run on every
``sn run`` and picks up exactly the changes introduced by a new DD version (or
any other edit to the anchoring ``IMASNode``).
"""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

#: Source fields compared for drift. Each maps a StandardName snapshot property
#: to the live IMASNode property it mirrors.
_TRACKED_FIELDS: tuple[tuple[str, str, str], ...] = (
    # (label, StandardName snapshot prop, IMASNode live prop)
    ("units", "source_unit", "unit"),
    ("documentation", "source_documentation", "documentation"),
)


def _norm(v: Any) -> str:
    """Normalise a value for comparison (None/empty collapse to '')."""
    return (v or "").strip() if isinstance(v, str) else ("" if v is None else str(v))


def stamp_source_snapshots(
    sn_ids: list[str] | None = None,
    *,
    only_unstamped: bool = False,
    gc: GraphClient | None = None,
) -> int:
    """Record the current DD-source snapshot on names (baseline / re-stamp).

    Sets ``source_unit`` and ``source_documentation`` on each targeted name from
    the live ``IMASNode`` its producing ``StandardNameSource`` points at. With
    ``sn_ids=None`` targets every live, source-linked name. With
    ``only_unstamped=True`` targets only names that lack a snapshot — the
    self-bootstrapping baseline that stops a fresh install from mass-refining
    (a name is only ever detected as drifted once it has a baseline to drift
    from). Returns the number of names stamped.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        clauses = ["sn.name_stage <> 'superseded'"]
        if sn_ids is not None:
            clauses.append("sn.id IN $sn_ids")
        if only_unstamped:
            clauses.append("sn.source_unit IS NULL")
        where = " AND ".join(clauses)
        rows = gc.query(
            f"""
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            MATCH (n:IMASNode {{id: src.source_id}})
            WHERE {where}
            WITH sn, head(collect(n)) AS n
            SET sn.source_unit = n.unit,
                sn.source_documentation = n.documentation
            RETURN count(sn) AS c
            """,
            sn_ids=sn_ids,
        )
        n = rows[0]["c"] if rows else 0
        logger.info(
            "stamp_source_snapshots: stamped %d name(s)%s",
            n,
            " (unstamped baseline)" if only_unstamped else "",
        )
        return n
    finally:
        if owns:
            gc.close()


def detect_source_drift(
    *, include_accepted: bool = True, gc: GraphClient | None = None
) -> list[dict[str, Any]]:
    """Return live, source-linked names whose DD-source snapshot has drifted.

    A name has drifted when any tracked field (unit, documentation) on the live
    ``IMASNode`` differs from the snapshot recorded on the name. Names with no
    snapshot yet (never stamped) are NOT reported as drifted — they are baselined
    by :func:`stamp_source_snapshots` first, so a fresh install does not mass-refine.

    Each result carries ``sn_id``, ``name``, the current pipeline stages, and a
    ``deltas`` list of ``{field, old, new}`` for the fields that changed.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        stage_filter = "" if include_accepted else "AND sn.docs_stage <> 'accepted'"
        rows = gc.query(
            f"""
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            MATCH (n:IMASNode {{id: src.source_id}})
            WHERE sn.name_stage <> 'superseded' {stage_filter}
              AND sn.source_unit IS NOT NULL
            WITH sn, head(collect(n)) AS n
            WHERE coalesce(sn.source_unit,'') <> coalesce(n.unit,'')
               OR coalesce(sn.source_documentation,'') <> coalesce(n.documentation,'')
            RETURN sn.id AS sn_id, sn.name_stage AS name_stage,
                   sn.docs_stage AS docs_stage,
                   sn.source_unit AS old_unit, n.unit AS new_unit,
                   sn.source_documentation AS old_doc, n.documentation AS new_doc,
                   n.id AS source_id
            ORDER BY sn.id
            """,
        )
        drifted: list[dict[str, Any]] = []
        for r in rows:
            deltas = []
            if _norm(r["old_unit"]) != _norm(r["new_unit"]):
                deltas.append(
                    {"field": "units", "old": r["old_unit"], "new": r["new_unit"]}
                )
            if _norm(r["old_doc"]) != _norm(r["new_doc"]):
                deltas.append(
                    {
                        "field": "documentation",
                        "old": r["old_doc"],
                        "new": r["new_doc"],
                    }
                )
            if not deltas:
                continue
            drifted.append(
                {
                    "sn_id": r["sn_id"],
                    "name_stage": r["name_stage"],
                    "docs_stage": r["docs_stage"],
                    "source_id": r["source_id"],
                    "deltas": deltas,
                }
            )
        return drifted
    finally:
        if owns:
            gc.close()


def _format_reason(sn_id: str, deltas: list[dict[str, Any]]) -> str:
    """Human/LLM-readable steering reason describing the precise DD change."""
    parts = []
    for d in deltas:
        old = _norm(d["old"]) or "—"
        new = _norm(d["new"]) or "—"
        if d["field"] == "documentation":
            old = (old[:160] + "…") if len(old) > 160 else old
            new = (new[:160] + "…") if len(new) > 160 else new
        parts.append(f"{d['field']}: {old!r} → {new!r}")
    return (
        "DD source drift — the anchoring Data Dictionary path changed "
        f"({'; '.join(parts)}). Refresh this name's documentation to reflect the "
        "corrected source; preserve the established intent and family phrasing. "
        "This is a targeted source-refresh, not a rewrite."
    )


def refresh_drifted_sources(
    *,
    run_id: str | None = None,
    dry_run: bool = False,
    include_accepted: bool = True,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Detect DD-source drift and steer a refine pass for each drifted name.

    For every drifted name a docs-axis ``apply_edit`` (hint mode, ``origin='agent'``)
    is attached carrying the precise DD delta as its reason, resetting the docs into
    the refine/review queue; the name's snapshot is then re-stamped so the change is
    not re-detected on the next run (idempotent). Returns a summary dict with the
    drifted names and how many were steered. A no-op (``steered=0``) when nothing
    drifted, so it is safe on every ``sn run``.
    """
    # Local import avoids a module-load cycle (edit.py imports graph_ops heavily).
    from imas_codex.standard_names.edit import apply_edit

    owns = gc is None
    gc = gc or GraphClient()
    try:
        # Self-bootstrapping baseline: any name without a snapshot is stamped to
        # the current source first, so it is never reported as drifted on the run
        # that first sees it (no mass-refine on deploy). Skipped under dry-run.
        baselined = 0 if dry_run else stamp_source_snapshots(only_unstamped=True, gc=gc)
        drifted = detect_source_drift(include_accepted=include_accepted, gc=gc)
        summary: dict[str, Any] = {
            "baselined": baselined,
            "detected": len(drifted),
            "steered": 0,
            "blocked": [],
            "names": [d["sn_id"] for d in drifted],
            "dry_run": dry_run,
        }
        if not drifted:
            logger.info("refresh_drifted_sources: no DD-source drift detected")
            return summary
        logger.info(
            "refresh_drifted_sources: %d name(s) drifted%s",
            len(drifted),
            " (dry-run)" if dry_run else "",
        )
        for d in drifted:
            reason = _format_reason(d["sn_id"], d["deltas"])
            if dry_run:
                logger.info("  would refresh %s — %s", d["sn_id"], reason)
                continue
            plan = apply_edit(
                target=d["sn_id"],
                hint=(
                    "The Data Dictionary source this name derives from has changed. "
                    "Update the documentation to reflect the corrected source."
                ),
                reason=reason,
                axis="docs",
                origin="agent",
                include_accepted=include_accepted,
                gc=gc,
            )
            if getattr(plan, "blocked", None):
                summary["blocked"].append({"sn_id": d["sn_id"], "why": plan.blocked})
                logger.warning("  refresh blocked for %s: %s", d["sn_id"], plan.blocked)
                continue
            # Re-stamp so this exact change is not re-detected next run (idempotent).
            stamp_source_snapshots([d["sn_id"]], gc=gc)
            summary["steered"] += 1
        return summary
    finally:
        if owns:
            gc.close()

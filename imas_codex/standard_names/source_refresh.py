"""Idempotent source-drift refresh — re-refine names when their DD source changes.

Wired into ``sn run``. Each standard name records the DD-source snapshot it was
last built against as typed properties on the ``StandardName``. The snapshot
retains effective values alongside raw values, applied and converged resolution
identities, the authority digest, and its typed marker. Every run resolves the
name's producing ``StandardNameSource`` to the *current* ``IMASNode`` (following
``RENAMED_TO`` lineage to the latest identity of the path) and compares the live
node to the complete snapshot. Effective unit, documentation, or path drift
steers a **refine** pass (via :func:`apply_edit`, hint mode, ``axis='docs'``)
whose reason carries the precise delta. Provenance-only drift is re-stamped
without steering documentation. On a rename, the source's ``source_id`` is
re-pointed to the new path. Re-stamping makes an unchanged subsequent run
idempotent.

Captures exactly the changes recorded in the DD graph — unit/documentation edits
(a new DD version) and ``RENAMED_TO`` path moves. It only ever *steers existing*
names; it never composes fresh names. Names in a terminal state (``superseded``)
are skipped.
"""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

#: Resolve a bound ``(o:IMASNode)`` (the source's original anchor) to the current
#: terminal node, following ``RENAMED_TO`` lineage to the latest identity. Also
#: collapses multiple producing sources per name to one. Leaves ``sn`` and ``n``
#: (the live node) bound.
_RESOLVE_TERMINAL = """
    OPTIONAL MATCH (o)-[:RENAMED_TO*1..12]->(t:IMASNode)
    WHERE NOT exists { (t)-[:RENAMED_TO]->() }
    WITH sn, o, head(collect(t)) AS t
    WITH sn, coalesce(t, o) AS n
    WITH sn, head(collect(n)) AS n
"""


def _norm(v: Any) -> str:
    """Normalise a value for comparison (None/empty collapse to '')."""
    return (v or "").strip() if isinstance(v, str) else ("" if v is None else str(v))


def _norm_ids(value: Any) -> tuple[str, ...]:
    """Normalize stored resolution identifiers for stable drift comparison."""
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(sorted(str(item) for item in value if item))


def _resolved_source_context(
    path: str, unit: str | None, documentation: str | None
) -> dict[str, Any]:
    """Read graph source fields with their typed DD-resolution provenance."""
    from imas_codex.settings import get_dd_version
    from imas_codex.standard_names.dd_resolutions import resolve_dd_row

    return resolve_dd_row(
        {"path": path, "unit": unit, "documentation": documentation},
        dd_version=get_dd_version(),
    ).as_pipeline_item()


def stamp_source_snapshots(
    sn_ids: list[str] | None = None,
    *,
    only_unstamped: bool = False,
    gc: GraphClient | None = None,
) -> int:
    """Record the current DD-source snapshot on names (baseline / re-stamp).

    Sets ``source_unit``, ``source_documentation`` and ``source_path`` on each
    targeted name from the *live* ``IMASNode`` its producing ``StandardNameSource``
    resolves to (via ``RENAMED_TO`` lineage). With ``sn_ids=None`` targets every
    live, source-linked name. With ``only_unstamped=True`` targets only names that
    lack a snapshot — the self-bootstrapping baseline that stops a fresh install
    from mass-refining. Returns the number of names stamped.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        clauses = ["sn.name_stage <> 'superseded'"]
        if sn_ids is not None:
            clauses.append("sn.id IN $sn_ids")
        if only_unstamped:
            clauses.append("sn.source_path IS NULL")
        where = " AND ".join(clauses)
        rows = gc.query(
            f"""
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            MATCH (o:IMASNode {{id: src.source_id}})
            WHERE {where}
            {_RESOLVE_TERMINAL}
            RETURN sn.id AS sn_id, n.id AS path, n.unit AS unit,
                   n.documentation AS documentation
            """,
            sn_ids=sn_ids,
        )
        updates = []
        for row in rows:
            resolved = _resolved_source_context(
                row["path"], row.get("unit"), row.get("documentation")
            )
            published = resolved["published_dd_context"]
            updates.append(
                {
                    "sn_id": row["sn_id"],
                    "path": row["path"],
                    "unit": resolved["unit"],
                    "documentation": resolved["documentation"],
                    "raw_unit": published["unit"],
                    "raw_documentation": published["documentation"],
                    "published_dd_context": published,
                    "resolution_ids": resolved["dd_resolution_ids"],
                    "converged_ids": resolved["dd_resolution_converged_ids"],
                    "manifest_digest": resolved["dd_resolution_manifest_digest"],
                    "resolution_marker": resolved["_dd_resolution_marker"],
                }
            )
        if updates:
            gc.query(
                """
                UNWIND $updates AS update
                MATCH (sn:StandardName {id: update.sn_id})
                SET sn.source_unit = update.unit,
                    sn.source_documentation = update.documentation,
                    sn.source_path = update.path,
                    sn.source_raw_unit = update.raw_unit,
                    sn.source_raw_documentation = update.raw_documentation,
                    sn.source_dd_resolution_ids = update.resolution_ids,
                    sn.source_dd_resolution_converged_ids = update.converged_ids,
                    sn.source_dd_resolution_manifest_digest = update.manifest_digest,
                    sn.source_dd_resolution_marker = update.resolution_marker
                """,
                updates=updates,
            )
        n = len(updates)
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

    A name has drifted when the live ``IMASNode`` it resolves to (following
    ``RENAMED_TO``) differs from the effective snapshot or when its raw DD
    context, resolution identities, authority digest, or typed marker changes.
    Names with no snapshot yet are NOT reported — they are baselined by
    :func:`stamp_source_snapshots` first, so a fresh install does not mass-refine.

    Each result carries ``sn_id``, the pipeline stages, ``new_path`` (the resolved
    terminal id), whether documentation steering is required, and a ``deltas``
    list of ``{field, old, new}`` entries. Provenance-only drift is visible but
    does not require documentation steering.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        stage_filter = "" if include_accepted else "AND sn.docs_stage <> 'accepted'"
        rows = gc.query(
            f"""
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
            MATCH (o:IMASNode {{id: src.source_id}})
            WHERE sn.name_stage <> 'superseded' {stage_filter}
              AND sn.source_path IS NOT NULL
            {_RESOLVE_TERMINAL}
            RETURN sn.id AS sn_id, sn.name_stage AS name_stage,
                   sn.docs_stage AS docs_stage,
                   sn.source_unit AS old_unit, n.unit AS new_unit,
                   sn.source_documentation AS old_doc, n.documentation AS new_doc,
                   sn.source_path AS old_path, n.id AS new_path,
                   sn.source_raw_unit AS old_raw_unit,
                   sn.source_raw_documentation AS old_raw_doc,
                   sn.source_dd_resolution_ids AS old_resolution_ids,
                   sn.source_dd_resolution_converged_ids AS old_converged_ids,
                   sn.source_dd_resolution_manifest_digest AS old_manifest_digest,
                   sn.source_dd_resolution_marker AS old_resolution_marker
            ORDER BY sn.id
            """,
        )
        drifted: list[dict[str, Any]] = []
        for r in rows:
            resolved = _resolved_source_context(
                r["new_path"], r.get("new_unit"), r.get("new_doc")
            )
            r = {
                **r,
                "new_unit": resolved["unit"],
                "new_doc": resolved["documentation"],
            }
            deltas = []
            if _norm(r["old_path"]) != _norm(r["new_path"]):
                deltas.append(
                    {"field": "source_path", "old": r["old_path"], "new": r["new_path"]}
                )
            if _norm(r["old_unit"]) != _norm(r["new_unit"]):
                deltas.append(
                    {"field": "units", "old": r["old_unit"], "new": r["new_unit"]}
                )
            if _norm(r["old_doc"]) != _norm(r["new_doc"]):
                deltas.append(
                    {"field": "documentation", "old": r["old_doc"], "new": r["new_doc"]}
                )
            published = resolved["published_dd_context"]
            if _norm(r.get("old_raw_unit")) != _norm(published.get("unit")):
                deltas.append(
                    {
                        "field": "raw_unit",
                        "old": r.get("old_raw_unit"),
                        "new": published.get("unit"),
                    }
                )
            if _norm(r.get("old_raw_doc")) != _norm(published.get("documentation")):
                deltas.append(
                    {
                        "field": "raw_documentation",
                        "old": r.get("old_raw_doc"),
                        "new": published.get("documentation"),
                    }
                )
            if _norm_ids(r.get("old_resolution_ids")) != _norm_ids(
                resolved["dd_resolution_ids"]
            ):
                deltas.append(
                    {
                        "field": "resolution_ids",
                        "old": r.get("old_resolution_ids") or [],
                        "new": resolved["dd_resolution_ids"],
                    }
                )
            if _norm_ids(r.get("old_converged_ids")) != _norm_ids(
                resolved["dd_resolution_converged_ids"]
            ):
                deltas.append(
                    {
                        "field": "converged_resolution_ids",
                        "old": r.get("old_converged_ids") or [],
                        "new": resolved["dd_resolution_converged_ids"],
                    }
                )
            if _norm(r.get("old_manifest_digest")) != _norm(
                resolved["dd_resolution_manifest_digest"]
            ):
                deltas.append(
                    {
                        "field": "resolution_manifest_digest",
                        "old": r.get("old_manifest_digest"),
                        "new": resolved["dd_resolution_manifest_digest"],
                    }
                )
            if _norm(r.get("old_resolution_marker")) != _norm(
                resolved["_dd_resolution_marker"]
            ):
                deltas.append(
                    {
                        "field": "resolution_marker",
                        "old": r.get("old_resolution_marker"),
                        "new": resolved["_dd_resolution_marker"],
                    }
                )
            if not deltas:
                continue
            requires_steering = any(
                delta["field"] in {"source_path", "units", "documentation"}
                for delta in deltas
            )
            drifted.append(
                {
                    "sn_id": r["sn_id"],
                    "name_stage": r["name_stage"],
                    "docs_stage": r["docs_stage"],
                    "new_path": r["new_path"],
                    "resolved_dd_context": resolved,
                    "renamed": any(d["field"] == "source_path" for d in deltas),
                    "requires_steering": requires_steering,
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
        label = "path renamed" if d["field"] == "source_path" else d["field"]
        parts.append(f"{label}: {old!r} → {new!r}")
    return (
        "DD source drift — the anchoring Data Dictionary path changed "
        f"({'; '.join(parts)}). Refresh this name's documentation to reflect the "
        "corrected source; preserve the established intent and family phrasing. "
        "This is a targeted source-refresh, not a rewrite."
    )


def _repoint_source(sn_id: str, new_path: str, *, gc: GraphClient) -> None:
    """Re-anchor a renamed source's ``source_id`` to the new DD path."""
    gc.query(
        """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: $id})
        MATCH (o:IMASNode {id: src.source_id})-[:RENAMED_TO*1..12]->(t:IMASNode {id: $new})
        SET src.source_id = $new
        """,
        id=sn_id,
        new=new_path,
    )


def refresh_drifted_sources(
    *,
    dry_run: bool = False,
    include_accepted: bool = True,
    scope_run_id: str | None = None,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Detect DD-source drift and steer a refine pass for each drifted name.

    Effective content drift attaches a docs-axis ``apply_edit`` carrying the
    precise DD delta and resets the docs into the refine/review queue. A rename
    also re-points the source to the new path. Provenance-only drift re-stamps
    without steering. When *scope_run_id* is supplied, each successfully reset
    name is attached to that active run so its docs pools can prioritize the
    work without weakening the ordinary global claim gate. Returns a summary
    dict; an unchanged source is a no-op.
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
            "renamed": sum(1 for d in drifted if d["renamed"]),
            "steered": 0,
            "restamped": 0,
            "blocked": [],
            "names": [d["sn_id"] for d in drifted],
            "dry_run": dry_run,
        }
        if not drifted:
            logger.info("refresh_drifted_sources: no DD-source drift detected")
            return summary
        logger.info(
            "refresh_drifted_sources: %d name(s) drifted (%d renamed)%s",
            len(drifted),
            summary["renamed"],
            " (dry-run)" if dry_run else "",
        )
        for d in drifted:
            if dry_run:
                logger.info(
                    "  would %s %s",
                    "steer and re-stamp" if d["requires_steering"] else "re-stamp",
                    d["sn_id"],
                )
                continue
            if not d["requires_steering"]:
                stamp_source_snapshots([d["sn_id"]], gc=gc)
                summary["restamped"] += 1
                continue
            reason = _format_reason(d["sn_id"], d["deltas"])
            # Re-anchor a renamed source before refining so the delta is coherent.
            if d["renamed"]:
                _repoint_source(d["sn_id"], d["new_path"], gc=gc)
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
            if scope_run_id:
                gc.query(
                    """
                    MATCH (sn:StandardName {id: $id})
                    SET sn.run_id = $scope_run_id
                    """,
                    id=d["sn_id"],
                    scope_run_id=scope_run_id,
                )
            # Re-stamp so this exact change is not re-detected next run (idempotent).
            stamp_source_snapshots([d["sn_id"]], gc=gc)
            summary["restamped"] += 1
            summary["steered"] += 1
        return summary
    finally:
        if owns:
            gc.close()

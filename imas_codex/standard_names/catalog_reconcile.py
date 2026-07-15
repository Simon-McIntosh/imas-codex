"""Diff-by-id reconciler — restore the graph ledger from the published catalog.

The published catalog is a COMPLETE, restorable projection of the graph
provenance ledger.  Restoring the graph from it is a diff-by-id RECONCILE,
never a node-recreating import:

- every entry is MATCHed to its existing ``StandardName`` BY ID — a missing
  node is reported, never recreated (the reconciler issues no
  ``MERGE (sn:StandardName`` node-creation);
- scalar editorial fields (description / documentation / unit) are updated
  only where the catalog and graph differ;
- each entry's ``sources:`` block is replayed through the provenance-rebuild
  binder to restore ``StandardNameSource`` + ``PRODUCED_NAME`` +
  ``FROM_DD_PATH`` / ``FROM_SIGNAL`` (every write gated on the name existing).

This is the sanctioned restore path.  ``sn merge`` is the PR-merge diff path
that folds reviewed curator edits from a merged catalog PR back into the
ledger; it must NOT be used to bootstrap or restore the graph.  This
reconciler is the only path that rebuilds provenance from a published catalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from imas_codex.standard_names.catalog_import import _normalize_field
from imas_codex.standard_names.provenance_rebuild import (
    bind_recovery_sources,
    recovery_sources_from_entries,
)

logger = logging.getLogger(__name__)

#: Scalar editorial fields the reconciler restores where catalog and graph
#: differ.  Structural / computed fields (arguments, error_variants, link
#: topology) are rebuilt from graph edges on export, not here.
_SCALAR_FIELDS: tuple[str, ...] = ("description", "documentation", "unit")

#: Read-only fetch of the current scalar state, keyed by id.  OPTIONAL MATCH
#: so a missing node returns a null-id row (reported as missing, not created).
_FETCH_SCALAR_STATE = """
    UNWIND $ids AS id
    OPTIONAL MATCH (sn:StandardName {id: id})
    RETURN sn.id AS id,
           sn.description AS description,
           sn.documentation AS documentation,
           sn.unit AS unit
"""

#: Apply scalar deltas by id.  MATCH (never MERGE) — a missing node is not in
#: the batch (filtered upstream), so this only ever updates existing nodes.
_APPLY_SCALAR_DELTAS = """
    UNWIND $batch AS b
    MATCH (sn:StandardName {id: b.id})
    SET sn.description = b.description,
        sn.documentation = b.documentation,
        sn.unit = b.unit
"""


@dataclass
class ReconcileReport:
    """Summary of a catalog→graph reconcile run."""

    matched: int = 0  # entries matched to an existing graph node
    updated: int = 0  # entries with a scalar delta applied (or would-be, dry_run)
    sources_bound: int = 0  # StandardNameSource nodes rebound
    missing: list[str] = field(default_factory=list)  # ids with no graph node
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


def _load_catalog_entries(
    catalog_dir: Path,
    report: ReconcileReport,
) -> list[dict[str, Any]]:
    """Load catalog entries from the working tree (per-domain list layout).

    Unlike ``catalog_import``, the ``sources:`` block is preserved — it is the
    provenance projection the reconciler replays.  Returns every named entry
    dict across ``standard_names/*.yml``.
    """
    sn_dir = catalog_dir / "standard_names"
    if not sn_dir.is_dir():
        sn_dir = catalog_dir

    yaml_files = sorted(
        p for p in sn_dir.rglob("*") if p.suffix in (".yml", ".yaml") and p.is_file()
    )

    entries: list[dict[str, Any]] = []
    for path in yaml_files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            report.errors.append(f"{path.name}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if isinstance(entry, dict) and entry.get("name"):
                entries.append(entry)
    return entries


def _fetch_scalar_state(gc: Any, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch current scalar field values for *ids*, keyed by id.

    A row whose ``sn.id`` is null (OPTIONAL MATCH miss) is dropped — the name
    is then absent from the result and treated as missing by the caller.
    """
    if not ids:
        return {}
    rows = gc.query(_FETCH_SCALAR_STATE, ids=ids)
    result: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if row.get("id"):
            result[row["id"]] = dict(row)
    return result


def _scalar_differs(existing: dict[str, Any], entry: dict[str, Any]) -> bool:
    """True when any scalar field differs (through :func:`_normalize_field`)."""
    for field_name in _SCALAR_FIELDS:
        if _normalize_field(existing.get(field_name)) != _normalize_field(
            entry.get(field_name)
        ):
            return True
    return False


def reconcile_catalog(
    catalog_dir: str | Path,
    *,
    dry_run: bool = False,
    gc: Any | None = None,
) -> ReconcileReport:
    """Reconcile the graph ledger against the published catalog by id.

    Parameters
    ----------
    catalog_dir:
        Path to the ISN catalog repository root (containing ``standard_names/``).
    dry_run:
        Compute and report the would-be deltas without writing to the graph.
    gc:
        Optional open ``GraphClient``.  When omitted, one is opened and closed
        for the duration of the call.

    Returns
    -------
    ReconcileReport with matched / updated / sources_bound counts and the list
    of catalog entries with no matching graph node (``missing``).
    """
    catalog_dir = Path(catalog_dir)
    report = ReconcileReport(dry_run=dry_run)

    entries = _load_catalog_entries(catalog_dir, report)
    if not entries:
        return report

    from imas_codex.graph.client import GraphClient

    owns = gc is None
    gc = gc or GraphClient()
    try:
        ids = [e["name"] for e in entries]
        state = _fetch_scalar_state(gc, ids)

        matched_names: set[str] = set()
        write_batch: list[dict[str, Any]] = []
        for entry in entries:
            name = entry["name"]
            existing = state.get(name)
            if existing is None:
                # Missing node — reported, never recreated (restore is a diff,
                # not a bootstrap; a name absent from the graph is a signal).
                report.missing.append(name)
                continue
            matched_names.add(name)
            report.matched += 1
            if _scalar_differs(existing, entry):
                write_batch.append(
                    {
                        "id": name,
                        "description": entry.get("description"),
                        "documentation": entry.get("documentation"),
                        "unit": entry.get("unit"),
                    }
                )

        report.updated = len(write_batch)
        if write_batch and not dry_run:
            gc.query(_APPLY_SCALAR_DELTAS, batch=write_batch)

        # Replay each matched entry's sources: block through the shared binder.
        recovered = recovery_sources_from_entries(entries)
        for name, specs in recovered.items():
            if name not in matched_names or dry_run:
                continue
            report.sources_bound += bind_recovery_sources(name, specs, gc=gc)
            from imas_codex.standard_names.provenance_lifecycle import (
                bind_sources_exclusively,
            )

            bind_sources_exclusively(gc, name, [spec["id"] for spec in specs])

        logger.info(
            "reconcile_catalog: matched=%d updated=%d sources_bound=%d "
            "missing=%d dry_run=%s",
            report.matched,
            report.updated,
            report.sources_bound,
            len(report.missing),
            dry_run,
        )
        return report
    finally:
        if owns:
            gc.close()

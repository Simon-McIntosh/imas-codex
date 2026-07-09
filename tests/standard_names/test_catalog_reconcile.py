"""Diff-by-id catalog reconciler — the sanctioned graph restore path.

The published catalog is a complete, restorable projection of the graph
provenance ledger.  Restoring the graph from it is a diff-by-id RECONCILE:

- every entry is MATCHed to its existing StandardName BY ID — a missing node
  is reported, never recreated (no ``MERGE (sn:StandardName`` node-creation);
- scalar editorial fields (description / documentation / unit) are updated
  only where the catalog and graph differ;
- each entry's ``sources:`` block is replayed through the provenance-rebuild
  binder (``bind_recovery_sources``) to restore StandardNameSource edges.

All tests mock ``gc.query`` / patch ``bind_recovery_sources`` — no Neo4j
instance is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from imas_codex.standard_names.catalog_reconcile import reconcile_catalog

_BIND = "imas_codex.standard_names.catalog_reconcile.bind_recovery_sources"


def _make_catalog(
    tmp_path: Path, entries: list[dict], domain: str = "equilibrium"
) -> Path:
    """Write a per-domain catalog YAML and return the catalog root."""
    root = tmp_path / "isnc"
    sn_dir = root / "standard_names"
    sn_dir.mkdir(parents=True)
    (sn_dir / f"{domain}.yml").write_text(yaml.safe_dump(entries))
    return root


def _gc_with_state(
    graph_rows: list[dict], captured: list[str] | None = None
) -> MagicMock:
    """Mock GraphClient returning ``graph_rows`` for the scalar-state fetch."""

    def _query(cypher: str, **params):
        if captured is not None:
            captured.append(cypher)
        if "UNWIND $ids AS id" in cypher:
            return graph_rows
        return []

    gc = MagicMock()
    gc.query = MagicMock(side_effect=_query)
    return gc


# ---------------------------------------------------------------------------
# Sources are rebuilt via bind_recovery_sources
# ---------------------------------------------------------------------------


def test_rebuilds_sources_from_yaml_block_via_binder(tmp_path: Path) -> None:
    """A matched entry's ``sources:`` block is replayed through
    ``bind_recovery_sources`` with the parsed source specs."""
    entry = {
        "name": "elongation_of_plasma_boundary",
        "kind": "scalar",
        "unit": "1",
        "description": "Elongation of the plasma boundary.",
        "documentation": "Ratio of vertical to horizontal extent.",
        "sources": [
            {
                "id": "dd:equilibrium/time_slice/boundary/elongation",
                "dd_path": "equilibrium/time_slice/boundary/elongation",
                "status": "attached",
            }
        ],
    }
    root = _make_catalog(tmp_path, [entry])
    graph_rows = [
        {
            "id": "elongation_of_plasma_boundary",
            "description": "Elongation of the plasma boundary.",
            "documentation": "Ratio of vertical to horizontal extent.",
            "unit": "1",
        }
    ]
    gc = _gc_with_state(graph_rows)

    bind_calls: list[tuple[str, list[dict]]] = []

    def _spy_bind(name_id, specs, *, gc):  # noqa: ANN001
        bind_calls.append((name_id, specs))
        return len(specs)

    with patch(_BIND, side_effect=_spy_bind):
        report = reconcile_catalog(root, gc=gc)

    assert len(bind_calls) == 1, "sources block must be replayed exactly once"
    name_id, specs = bind_calls[0]
    assert name_id == "elongation_of_plasma_boundary"
    assert specs[0]["id"] == "dd:equilibrium/time_slice/boundary/elongation"
    assert specs[0]["source_type"] == "dd"
    assert specs[0]["dd_path"] == "equilibrium/time_slice/boundary/elongation"
    assert report.sources_bound == 1


def test_no_sources_block_binds_nothing(tmp_path: Path) -> None:
    """An entry without a ``sources:`` block triggers no source binding."""
    entry = {
        "name": "plasma_current",
        "kind": "scalar",
        "unit": "A",
        "description": "Total toroidal plasma current.",
        "documentation": "The total toroidal current.",
    }
    root = _make_catalog(tmp_path, [entry])
    graph_rows = [
        {
            "id": "plasma_current",
            "description": "Total toroidal plasma current.",
            "documentation": "The total toroidal current.",
            "unit": "A",
        }
    ]
    gc = _gc_with_state(graph_rows)

    with patch(_BIND) as m_bind:
        report = reconcile_catalog(root, gc=gc)

    assert not m_bind.called
    assert report.sources_bound == 0


# ---------------------------------------------------------------------------
# Match-by-id: never recreates a StandardName node
# ---------------------------------------------------------------------------


def test_matches_by_id_and_never_creates_a_node(tmp_path: Path) -> None:
    """A scalar delta is applied via MATCH...SET — never a node-creating MERGE."""
    entry = {
        "name": "electron_temperature",
        "kind": "scalar",
        "unit": "eV",
        "description": "Electron temperature (revised).",
        "documentation": "Te from Thomson scattering.",
    }
    root = _make_catalog(tmp_path, [entry], domain="core_plasma_physics")
    graph_rows = [
        {
            "id": "electron_temperature",
            # Stale description → a delta the reconciler must apply.
            "description": "Electron temperature.",
            "documentation": "Te from Thomson scattering.",
            "unit": "eV",
        }
    ]
    captured: list[str] = []
    gc = _gc_with_state(graph_rows, captured)

    report = reconcile_catalog(root, gc=gc)

    for cypher in captured:
        assert "MERGE (sn:StandardName" not in cypher, (
            "reconciler must never recreate a StandardName node — it matches "
            f"by id only. Offending Cypher:\n{cypher}"
        )
    assert any(
        "MATCH (sn:StandardName {id: b.id})" in c and "SET sn.description" in c
        for c in captured
    ), "the scalar delta must be applied via a MATCH...SET (by id)"
    assert report.matched == 1
    assert report.updated == 1


def test_missing_name_is_reported_not_recreated(tmp_path: Path) -> None:
    """An entry with no graph node is recorded in ``missing`` and not created."""
    entry = {
        "name": "ghost_quantity",
        "kind": "scalar",
        "unit": "1",
        "description": "A name absent from the graph.",
        "documentation": "Should not be recreated by reconcile.",
    }
    root = _make_catalog(tmp_path, [entry])
    # OPTIONAL MATCH miss → sn.id null row.
    graph_rows = [
        {"id": None, "description": None, "documentation": None, "unit": None}
    ]
    captured: list[str] = []
    gc = _gc_with_state(graph_rows, captured)

    report = reconcile_catalog(root, gc=gc)

    assert report.missing == ["ghost_quantity"]
    assert report.matched == 0
    assert report.updated == 0
    for cypher in captured:
        assert "MERGE (sn:StandardName" not in cypher


# ---------------------------------------------------------------------------
# dry_run performs no writes
# ---------------------------------------------------------------------------


def test_dry_run_performs_no_writes(tmp_path: Path) -> None:
    """dry_run reports would-be counts but issues no SET and binds no sources."""
    entry = {
        "name": "elongation_of_plasma_boundary",
        "kind": "scalar",
        "unit": "1",
        "description": "New description.",
        "documentation": "New documentation.",
        "sources": [
            {
                "id": "dd:equilibrium/time_slice/boundary/elongation",
                "dd_path": "equilibrium/time_slice/boundary/elongation",
                "status": "attached",
            }
        ],
    }
    root = _make_catalog(tmp_path, [entry])
    graph_rows = [
        {
            "id": "elongation_of_plasma_boundary",
            "description": "Stale description.",
            "documentation": "Stale documentation.",
            "unit": "1",
        }
    ]
    captured: list[str] = []
    gc = _gc_with_state(graph_rows, captured)

    with patch(_BIND) as m_bind:
        report = reconcile_catalog(root, gc=gc, dry_run=True)

    assert not m_bind.called, "dry_run must not bind sources"
    assert not any("SET sn.description" in c for c in captured), (
        "dry_run must not issue a scalar SET write"
    )
    assert report.dry_run is True
    # Would-be accounting is still reported.
    assert report.matched == 1
    assert report.updated == 1
    assert report.sources_bound == 0

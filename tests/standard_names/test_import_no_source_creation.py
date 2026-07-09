"""Contract lock: ``run_import`` is the PR-merge diff path, NOT a graph restore.

The graph is the authoritative provenance ledger.  Restoring / bootstrapping
it from the published catalog is the job of the diff-by-id reconciler
(``catalog_reconcile.reconcile_catalog``), which replays each entry's
``sources:`` block through ``bind_recovery_sources``.

``run_import`` deliberately does NOT rebuild provenance: it must never create a
``StandardNameSource`` node nor a ``PRODUCED_NAME`` edge.  This regression test
locks that contract so no one can quietly re-introduce bulk-restore semantics
into the import path (which is exactly how the round-trip severed provenance
in the first place).

Mock-based — the whole ``run_import`` flow runs against a stubbed GraphClient;
no Neo4j instance is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

pytest.importorskip("imas_standard_names")

_GC_PATCH = "imas_codex.graph.client.GraphClient"


def _make_catalog(tmp_path: Path) -> Path:
    """A minimal per-domain catalog with a ``sources:`` block present.

    The entry carries sources precisely to prove import IGNORES them (it never
    turns a YAML ``sources:`` block into StandardNameSource / PRODUCED_NAME).
    """
    root = tmp_path / "isnc"
    sn_dir = root / "standard_names"
    sn_dir.mkdir(parents=True)
    entries = [
        {
            "name": "electron_temperature",
            "kind": "scalar",
            "unit": "eV",
            "description": "Electron temperature.",
            "documentation": "Te from Thomson scattering.",
            "links": [],
            "status": "active",
        }
    ]
    (sn_dir / "core_plasma_physics.yml").write_text(yaml.safe_dump(entries))
    return root


def test_import_never_creates_source_or_produced_name(tmp_path: Path) -> None:
    """No Cypher issued by run_import may create a source or PRODUCED_NAME edge."""
    from imas_codex.standard_names.catalog_import import run_import

    isnc = _make_catalog(tmp_path)
    captured: list[str] = []

    def _query(cypher: str, **params: Any):
        captured.append(cypher)
        # Import lock acquire / release
        if "ImportLock" in cypher and "holder IS NULL" in cypher:
            return [{"acquired": True}]
        if "ImportLock" in cypher:
            return []
        # _fetch_graph_state — new node (no existing rows) so it is "created".
        if "UNWIND $ids AS id" in cypher:
            return [{"id": None}]
        # Watermark reads
        if "Watermark" in cypher or "watermark" in cypher:
            return []
        return []

    gc = MagicMock()
    gc.query = MagicMock(side_effect=_query)

    with patch(_GC_PATCH) as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        report = run_import(isnc)

    assert captured, "run_import issued no Cypher"
    for cypher in captured:
        assert "StandardNameSource" not in cypher, (
            "run_import must never touch StandardNameSource — provenance "
            "restore is the reconciler's job, not the import diff path.\n\n"
            f"Offending Cypher:\n{cypher}"
        )
        assert "PRODUCED_NAME" not in cypher, (
            "run_import must never create a PRODUCED_NAME edge — provenance "
            "restore is the reconciler's job, not the import diff path.\n\n"
            f"Offending Cypher:\n{cypher}"
        )
    # Sanity: the entry really was processed (a StandardName MERGE ran).
    assert any("MERGE (sn:StandardName" in c for c in captured), (
        "expected the import to MERGE the StandardName node"
    )
    assert report.created == 1

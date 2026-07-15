"""Export emits only the public semantic StandardNameSource projection.

Operational ids, statuses and source types remain internal. DD sources expose
their pinned DD snapshot plus graph-held semantic context.

The forbidden flat fields (``source_paths`` / ``dd_paths``) are a separate
concern and stay stripped — these tests only cover the structured block.

All tests mock ``gc.query`` — no Neo4j instance is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.export import _fetch_sources_for_entry


def _gc_returning(rows: list[dict]) -> MagicMock:
    gc = MagicMock()
    gc.query = MagicMock(return_value=rows)
    return gc


def test_query_projects_semantic_provenance_only() -> None:
    captured: list[str] = []
    gc = MagicMock()
    gc.query = MagicMock(side_effect=lambda cypher, **kw: captured.append(cypher) or [])
    _fetch_sources_for_entry(gc, "electron_temperature")

    assert captured, "_fetch_sources_for_entry issued no query"
    cypher = captured[0]
    assert "src.source_type AS source_type" not in cypher
    assert "src.status AS status" not in cypher
    assert "src.id AS source_id" not in cypher
    assert "src.dd_version AS dd_version" in cypher
    assert "src.dd_snapshot_pinned AS dd_snapshot_pinned" in cypher
    assert "src.dd_parent_documentation AS parent_documentation" in cypher
    assert "src.dd_data_type AS data_type" in cypher
    assert "src.dd_coordinates AS coordinates" in cypher
    assert "dd.documentation" not in cypher
    assert "dd.description" not in cypher
    assert "src.provenance AS semantic_facet" in cypher


def test_emits_source_type_and_provenance_when_present() -> None:
    """A dd source with source_type + provenance emits both keys."""
    rows = [
        {
            "source_id": "dd:equilibrium/time_slice/profiles_1d/psi",
            "dd_path": "equilibrium/time_slice/profiles_1d/psi",
            "dd_version": "4.1.0",
            "dd_snapshot_pinned": True,
            "leaf_documentation": "Poloidal magnetic flux.",
            "parent_path": "equilibrium/time_slice/profiles_1d",
            "parent_documentation": "One-dimensional equilibrium profiles.",
            "data_type": "FLT_1D",
            "unit": "Wb",
            "coordinates": ["rho_tor_norm"],
            "lifecycle_status": "active",
            "lifecycle_version": "3.0.0",
            "enhanced_description": "Physics-aware flux context.",
            "enhancement_kind": "llm",
            "signal_id": None,
            "status": "attached",
            "source_type": "dd",
            "semantic_facet": "reconstructed",
        }
    ]
    sources = _fetch_sources_for_entry(_gc_returning(rows), "poloidal_magnetic_flux")

    assert sources is not None and len(sources) == 1
    src = sources[0]
    assert src["dd_path"] == "equilibrium/time_slice/profiles_1d/psi"
    assert src["dd_version"] == "4.1.0"
    assert src["dd_documentation_url"] == (
        "https://imas-data-dictionary.readthedocs.io/en/4.1.0/generated/ids/"
        "equilibrium.html#equilibrium-time_slice-profiles_1d-psi"
    )
    assert src["dd_documentation"] == {
        "leaf": "Poloidal magnetic flux.",
        "parent_path": "equilibrium/time_slice/profiles_1d",
        "parent": "One-dimensional equilibrium profiles.",
        "data_type": "FLT_1D",
        "unit": "Wb",
        "coordinates": ["rho_tor_norm"],
        "lifecycle_status": "active",
        "lifecycle_version": "3.0.0",
    }
    assert src["enhanced_context"] == {
        "description": "Physics-aware flux context.",
        "kind": "llm",
    }
    assert src["semantic_facet"] == "reconstructed"
    assert "signal_id" not in src  # null → omitted


def test_omits_non_semantic_source_without_path() -> None:
    rows = [
        {
            "source_id": "derived:foo_bar",
            "dd_path": None,
            "dd_version": None,
            "dd_snapshot_pinned": None,
            "signal_id": None,
            "status": "composed",
            "source_type": "derived",
            "semantic_facet": None,
        }
    ]
    sources = _fetch_sources_for_entry(_gc_returning(rows), "foo_bar")

    assert sources is None


def test_signal_source_carries_signal_id_and_type() -> None:
    """A facility-signal source emits signal_id + source_type='signals'."""
    rows = [
        {
            "source_id": "west:magnetics/ip",
            "dd_path": None,
            "dd_version": None,
            "dd_snapshot_pinned": None,
            "signal_id": "west:magnetics/ip",
            "status": "attached",
            "source_type": "signals",
            "semantic_facet": None,
        }
    ]
    sources = _fetch_sources_for_entry(_gc_returning(rows), "plasma_current")

    src = sources[0]
    assert src["signal_id"] == "west:magnetics/ip"
    assert "source_type" not in src
    assert "dd_path" not in src


def test_refuses_dd_source_without_pinned_version() -> None:
    rows = [
        {
            "dd_path": "equilibrium/time_slice/profiles_1d/psi",
            "dd_version": None,
            "signal_id": None,
        }
    ]
    with pytest.raises(ValueError, match="no pinned dd_version"):
        _fetch_sources_for_entry(_gc_returning(rows), "poloidal_magnetic_flux")


def test_refuses_dd_source_without_immutable_snapshot_marker() -> None:
    rows = [
        {
            "dd_path": "equilibrium/time_slice/profiles_1d/psi",
            "dd_version": "4.1.0",
            "dd_snapshot_pinned": False,
            "signal_id": None,
        }
    ]
    with pytest.raises(ValueError, match="no provable immutable snapshot"):
        _fetch_sources_for_entry(_gc_returning(rows), "poloidal_magnetic_flux")


def test_no_sources_returns_none() -> None:
    """A name with no PRODUCED_NAME source yields None (no sources key)."""
    assert _fetch_sources_for_entry(_gc_returning([]), "unlinked_name") is None

"""Export emits the full StandardNameSource projection (lossless ledger).

The ``sources:`` block written per catalog entry is the sanctioned,
restorable projection of the provenance ledger.  For a restore to be
lossless it must carry, per source, the fields the reconciler needs to
rebuild the ``StandardNameSource`` node: ``id`` + (``dd_path`` | ``signal_id``)
+ ``status`` + ``source_type`` (always) + ``provenance`` (when non-null).

The forbidden flat fields (``source_paths`` / ``dd_paths``) are a separate
concern and stay stripped — these tests only cover the structured block.

All tests mock ``gc.query`` — no Neo4j instance is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.standard_names.export import _fetch_sources_for_entry


def _gc_returning(rows: list[dict]) -> MagicMock:
    gc = MagicMock()
    gc.query = MagicMock(return_value=rows)
    return gc


def test_query_projects_source_type_and_provenance() -> None:
    """The Cypher must RETURN src.source_type and src.provenance."""
    captured: list[str] = []
    gc = MagicMock()
    gc.query = MagicMock(side_effect=lambda cypher, **kw: captured.append(cypher) or [])
    _fetch_sources_for_entry(gc, "electron_temperature")

    assert captured, "_fetch_sources_for_entry issued no query"
    cypher = captured[0]
    assert "src.source_type AS source_type" in cypher, (
        f"query must project source_type for a lossless ledger:\n{cypher}"
    )
    assert "src.provenance AS provenance" in cypher, (
        f"query must project provenance for a lossless ledger:\n{cypher}"
    )


def test_emits_source_type_and_provenance_when_present() -> None:
    """A dd source with source_type + provenance emits both keys."""
    rows = [
        {
            "source_id": "dd:equilibrium/time_slice/profiles_1d/psi",
            "dd_path": "equilibrium/time_slice/profiles_1d/psi",
            "signal_id": None,
            "status": "attached",
            "source_type": "dd",
            "provenance": "dd-extract-2026-07",
        }
    ]
    sources = _fetch_sources_for_entry(_gc_returning(rows), "poloidal_magnetic_flux")

    assert sources is not None and len(sources) == 1
    src = sources[0]
    assert src["id"] == "dd:equilibrium/time_slice/profiles_1d/psi"
    assert src["dd_path"] == "equilibrium/time_slice/profiles_1d/psi"
    assert src["status"] == "attached"
    assert src["source_type"] == "dd"
    assert src["provenance"] == "dd-extract-2026-07"
    assert "signal_id" not in src  # null → omitted


def test_emits_source_type_but_omits_null_provenance() -> None:
    """source_type is always emitted; a null provenance is omitted."""
    rows = [
        {
            "source_id": "derived:foo_bar",
            "dd_path": None,
            "signal_id": None,
            "status": "composed",
            "source_type": "derived",
            "provenance": None,
        }
    ]
    sources = _fetch_sources_for_entry(_gc_returning(rows), "foo_bar")

    assert sources is not None and len(sources) == 1
    src = sources[0]
    assert src["source_type"] == "derived"
    assert "provenance" not in src, (
        "a null provenance must not be emitted into the sources block"
    )


def test_signal_source_carries_signal_id_and_type() -> None:
    """A facility-signal source emits signal_id + source_type='signals'."""
    rows = [
        {
            "source_id": "west:magnetics/ip",
            "dd_path": None,
            "signal_id": "west:magnetics/ip",
            "status": "attached",
            "source_type": "signals",
            "provenance": None,
        }
    ]
    sources = _fetch_sources_for_entry(_gc_returning(rows), "plasma_current")

    src = sources[0]
    assert src["signal_id"] == "west:magnetics/ip"
    assert src["source_type"] == "signals"
    assert "dd_path" not in src


def test_no_sources_returns_none() -> None:
    """A name with no PRODUCED_NAME source yields None (no sources key)."""
    assert _fetch_sources_for_entry(_gc_returning([]), "unlinked_name") is None

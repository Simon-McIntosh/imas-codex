"""Fast unit tests for source_paths scalar materialization.

``reconcile_standard_name_source_paths`` recomputes each live name's
denormalised ``source_paths`` scalar as the sorted, deduped union of its live
provenance edges (``'dd:'+imas.id`` for HAS_STANDARD_NAME, ``src.id`` for
non-derived PRODUCED_NAME), preserving existing ``derived:`` entries. These
tests mock the GraphClient so they run without a live graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _run_reconcile(read_rows: list[dict]) -> list[dict] | None:
    """Invoke the reconcile with a mocked GraphClient; return the SET batch.

    The mock's ``query`` returns ``read_rows`` on the first (read) call and an
    empty list on the write call, capturing the write's ``updates`` payload.
    """
    captured: dict[str, list[dict] | None] = {"updates": None}
    calls = {"n": 0}

    def fake_query(cypher: str, **params):
        calls["n"] += 1
        if "SET sn.source_paths = u.paths" in cypher:
            captured["updates"] = params.get("updates")
            return []
        return read_rows

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(side_effect=fake_query)

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=mock_gc):
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_source_paths,
        )

        result = reconcile_standard_name_source_paths()

    return captured["updates"], result


def test_materializes_sorted_union_of_edges() -> None:
    """The scalar becomes the sorted, deduped union of both edge sides."""
    rows = [
        {
            "id": "electron_temperature",
            "current": ["dd:zzz/old/stale"],  # stale entry, no backing edge
            "hsn_paths": ["dd:core_profiles/profiles_1d/electrons/temperature"],
            "produced_paths": ["dd:edge_profiles/profiles_1d/electrons/temperature"],
        }
    ]
    updates, result = _run_reconcile(rows)
    assert result == {"names_reconciled": 1}
    assert updates == [
        {
            "id": "electron_temperature",
            "paths": [
                "dd:core_profiles/profiles_1d/electrons/temperature",
                "dd:edge_profiles/profiles_1d/electrons/temperature",
            ],
        }
    ]


def test_preserves_derived_entries() -> None:
    """Existing ``derived:`` scalar entries are retained, not dropped."""
    rows = [
        {
            "id": "some_derived_parent",
            "current": ["derived:grammar_peel:some_parent", "dd:stale/path"],
            "hsn_paths": ["dd:live/path"],
            "produced_paths": [],
        }
    ]
    updates, _ = _run_reconcile(rows)
    assert updates == [
        {
            "id": "some_derived_parent",
            # sorted: "dd:…" precedes "derived:…" ("dd" < "de")
            "paths": ["dd:live/path", "derived:grammar_peel:some_parent"],
        }
    ]


def test_idempotent_when_scalar_equals_union() -> None:
    """No write when the scalar already equals the sorted edge union."""
    rows = [
        {
            "id": "plasma_current",
            "current": ["dd:a", "dd:b"],  # already sorted union
            "hsn_paths": ["dd:a", "dd:b"],
            "produced_paths": ["dd:a"],
        }
    ]
    updates, result = _run_reconcile(rows)
    assert updates is None  # SET batch never issued
    assert result == {"names_reconciled": 0}


def test_deduplicates_across_both_edge_sides() -> None:
    """A path present on both edge sides appears once."""
    rows = [
        {
            "id": "n",
            "current": [],
            "hsn_paths": ["dd:x"],
            "produced_paths": ["dd:x"],
        }
    ]
    updates, _ = _run_reconcile(rows)
    assert updates == [{"id": "n", "paths": ["dd:x"]}]

"""Tests for DD context injection into the generate_docs worker.

Covers:
- _enrich_for_docs_gen populates source_paths, dd_source_docs, dd_aliases
- _enrich_for_docs_gen gracefully skips items with no graph data
- _nearby_names_for_docs_gen returns accepted SNs from same domain
- generate_docs_user.md template renders source_paths, dd_source_docs,
  nearby_existing_names when provided (template-binding smoke test)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

# =============================================================================
# Helpers
# =============================================================================

_WORKERS_MOD = "imas_codex.standard_names.workers"
_GC_CLASS = "imas_codex.graph.client.GraphClient"


def _mock_gc(query_side_effect: list[list[dict]] | None = None) -> Any:
    """Return a mock GraphClient whose .query() returns successive result sets."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    if query_side_effect is not None:
        gc.query = MagicMock(side_effect=query_side_effect)
    else:
        gc.query = MagicMock(return_value=[])
    return gc


@contextmanager
def _patch_gc(gc: Any):
    with patch(_GC_CLASS, return_value=gc):
        yield


# =============================================================================
# _enrich_for_docs_gen — unit tests (mocked GraphClient)
# =============================================================================


def test_enrich_populates_source_paths():
    """source_paths are stripped of dd: prefix and stored on item."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _mock_gc(
        query_side_effect=[
            # first call: _DOCS_GEN_ENRICH_QUERY for electron_temperature
            [
                {
                    "source_paths": [
                        "dd:core_profiles/profiles_1d/electrons/temperature"
                    ],
                    "dd_nodes": [
                        {
                            "id": "core_profiles/profiles_1d/electrons/temperature",
                            "documentation": "Electron temperature profile.",
                            "description": "Te profile",
                            "alias": "Te",
                            "unit": "eV",
                        }
                    ],
                }
            ],
        ]
    )

    item: dict[str, Any] = {
        "id": "electron_temperature",
        "description": "Electron temperature",
        "physics_domain": "core_profiles",
    }

    # Patch _search_nearby_names and _related_path_neighbours to avoid graph I/O
    with (
        patch(f"{_WORKERS_MOD}._search_nearby_names", return_value=[]),
        patch(f"{_WORKERS_MOD}._related_path_neighbours", return_value=[]),
    ):
        _enrich_for_docs_gen(gc, [item])

    assert "source_paths" in item, "source_paths should be populated"
    assert item["source_paths"] == [
        "core_profiles/profiles_1d/electrons/temperature"
    ], "dd: prefix should be stripped"


def test_enrich_populates_dd_source_docs_and_aliases():
    """dd_source_docs and dd_aliases are populated from linked IMASNodes."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _mock_gc(
        query_side_effect=[
            [
                {
                    "source_paths": [
                        "dd:core_profiles/profiles_1d/electrons/temperature"
                    ],
                    "dd_nodes": [
                        {
                            "id": "core_profiles/profiles_1d/electrons/temperature",
                            "documentation": "Electron temperature in plasma core.",
                            "description": "Te",
                            "alias": "Te",
                            "unit": "eV",
                        }
                    ],
                }
            ],
        ]
    )

    item: dict[str, Any] = {
        "id": "electron_temperature",
        "description": "Electron temperature",
    }

    with (
        patch(f"{_WORKERS_MOD}._search_nearby_names", return_value=[]),
        patch(f"{_WORKERS_MOD}._related_path_neighbours", return_value=[]),
    ):
        _enrich_for_docs_gen(gc, [item])

    assert "dd_source_docs" in item, "dd_source_docs should be populated"
    assert item["dd_source_docs"][0]["id"] == (
        "core_profiles/profiles_1d/electrons/temperature"
    )
    assert "dd_aliases" in item, "dd_aliases should be populated"
    assert "Te" in item["dd_aliases"]


def test_enrich_populates_nearest_peers():
    """nearest_peers are populated when _search_nearby_names returns results."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _mock_gc(
        query_side_effect=[
            # enrich query: empty dd_nodes, no source_paths
            [{"source_paths": [], "dd_nodes": []}],
        ]
    )

    peers_mock = [
        {
            "id": "ion_temperature",
            "description": "Ion temperature",
            "unit": "eV",
            "physics_domain": "core_profiles",
        },
    ]

    item: dict[str, Any] = {
        "id": "electron_temperature",
        "description": "Electron temperature",
    }

    with patch(f"{_WORKERS_MOD}._search_nearby_names", return_value=peers_mock):
        _enrich_for_docs_gen(gc, [item])

    assert "nearest_peers" in item, "nearest_peers should be populated"
    assert item["nearest_peers"][0]["tag"] == "name:ion_temperature"


def test_enrich_skips_item_with_no_id():
    """Items without an id key are silently skipped."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _mock_gc()
    item: dict[str, Any] = {"description": "orphan item"}

    _enrich_for_docs_gen(gc, [item])

    gc.query.assert_not_called()
    assert "source_paths" not in item


def test_enrich_handles_empty_graph_result():
    """Empty graph result leaves item without enrichment keys (no crash)."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _mock_gc(query_side_effect=[[]])  # empty result set

    item: dict[str, Any] = {"id": "missing_sn", "description": "some description"}

    with patch(f"{_WORKERS_MOD}._search_nearby_names", return_value=[]):
        _enrich_for_docs_gen(gc, [item])

    assert "source_paths" not in item
    assert "dd_source_docs" not in item


def test_enrich_excludes_self_from_nearest_peers():
    """The SN being enriched must not appear in nearest_peers."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    gc = _mock_gc(query_side_effect=[[{"source_paths": [], "dd_nodes": []}]])

    peers_mock = [
        {"id": "electron_temperature", "description": "Self", "unit": "eV"},
        {"id": "ion_temperature", "description": "Ion temp", "unit": "eV"},
    ]

    item: dict[str, Any] = {"id": "electron_temperature", "description": "Te"}

    with patch(f"{_WORKERS_MOD}._search_nearby_names", return_value=peers_mock):
        _enrich_for_docs_gen(gc, [item])

    peer_tags = [p["tag"] for p in item.get("nearest_peers", [])]
    assert "name:electron_temperature" not in peer_tags, "Self must be excluded"
    assert "name:ion_temperature" in peer_tags


# =============================================================================
# _nearby_names_for_docs_gen — unit tests
# =============================================================================


def test_nearby_names_returns_accepted_sns():
    """_nearby_names_for_docs_gen returns list of dicts from the graph."""
    from imas_codex.standard_names.workers import _nearby_names_for_docs_gen

    gc = _mock_gc(
        query_side_effect=[
            [
                {
                    "id": "ion_temperature",
                    "description": "Ion temperature.",
                    "kind": "scalar",
                    "unit": "eV",
                }
            ],
        ]
    )

    items = [
        {"id": "electron_temperature", "physics_domain": "core_profiles"},
    ]

    nearby = _nearby_names_for_docs_gen(gc, items)

    assert len(nearby) == 1
    assert nearby[0]["id"] == "ion_temperature"


def test_nearby_names_excludes_batch_items():
    """Items in the batch are not included in nearby_existing_names."""
    from imas_codex.standard_names.workers import _nearby_names_for_docs_gen

    gc = _mock_gc(
        query_side_effect=[
            [
                {
                    "id": "electron_temperature",  # same as batch item
                    "description": "Self.",
                    "kind": "scalar",
                    "unit": "eV",
                },
                {
                    "id": "ion_temperature",
                    "description": "Ion temperature.",
                    "kind": "scalar",
                    "unit": "eV",
                },
            ],
        ]
    )

    items = [{"id": "electron_temperature", "physics_domain": "core_profiles"}]

    nearby = _nearby_names_for_docs_gen(gc, items)

    ids = [n["id"] for n in nearby]
    assert "electron_temperature" not in ids, "Batch item must be excluded"
    assert "ion_temperature" in ids


# =============================================================================
# Template binding smoke test — no graph required
# =============================================================================


def test_generate_docs_user_template_renders_source_paths():
    """generate_docs_user template renders source_paths and dd_source_docs."""
    from imas_codex.llm.prompt_loader import render_prompt

    context = {
        "item": {
            "id": "electron_temperature",
            "name": "electron_temperature",
            "unit": "eV",
            "kind": "scalar",
            "physics_domain": "core_profiles",
            "description": "Electron temperature in plasma.",
            "reviewer_score_name": 0.85,
            "reviewer_comments_name": "Good name.",
            "chain_history": [],
            "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
            "dd_source_docs": [
                {
                    "id": "core_profiles/profiles_1d/electrons/temperature",
                    "documentation": "Temperature of electrons in core profile.",
                    "unit": "eV",
                }
            ],
            "dd_aliases": ["Te"],
            "nearest_peers": [
                {
                    "tag": "name:ion_temperature",
                    "unit": "eV",
                    "physics_domain": "core_profiles",
                    "doc_short": "Ion temperature profile.",
                    "cocos_label": "",
                }
            ],
        },
        "chain_history": [],
        "nearby_existing_names": [
            {
                "id": "electron_density",
                "description": "Electron number density.",
                "kind": "scalar",
                "unit": "m^-3",
            }
        ],
    }

    rendered = render_prompt("sn/generate_docs_user", context)

    # source_paths section
    assert "core_profiles/profiles_1d/electrons/temperature" in rendered, (
        "Rendered prompt must contain the DD path"
    )

    # dd_source_docs section
    assert "Temperature of electrons in core profile" in rendered, (
        "Rendered prompt must contain DD source documentation"
    )

    # dd_aliases section
    assert "Te" in rendered, "Rendered prompt must mention DD alias"

    # nearest_peers section
    assert "name:ion_temperature" in rendered, (
        "Rendered prompt must list nearest peer SNs"
    )

    # nearby_existing_names section
    assert "electron_density" in rendered, (
        "Rendered prompt must list nearby existing names in same domain"
    )


def test_generate_docs_user_template_renders_without_dd_context():
    """generate_docs_user template renders cleanly when DD context is absent."""
    from imas_codex.llm.prompt_loader import render_prompt

    context = {
        "item": {
            "id": "electron_temperature",
            "name": "electron_temperature",
            "unit": "eV",
            "kind": "scalar",
            "physics_domain": "core_profiles",
            "description": "Electron temperature.",
            "reviewer_score_name": None,
            "reviewer_comments_name": None,
            "chain_history": [],
        },
        "chain_history": [],
        "nearby_existing_names": [],
    }

    # Should not raise
    rendered = render_prompt("sn/generate_docs_user", context)
    assert "electron_temperature" in rendered
    # DD sections should be absent (not even their headers)
    assert "IMAS DD Paths" not in rendered
    assert "DD Source Documentation" not in rendered


# =============================================================================
# Documentation-family safety
# =============================================================================


class _SiblingFamilyGC:
    """Return a canned family row while recording the read query."""

    def __init__(self, row: dict[str, Any] | None):
        self.row = row
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, **params: Any):
        self.calls.append((cypher, params))
        return [self.row] if self.row is not None else []


def _sibling(name: str, operator_kind: str) -> dict[str, Any]:
    return {
        "id": name,
        "description": f"Description for {name}",
        "documentation": f"Documentation for {name}",
        "docs_stage": "accepted",
        "reviewer_score_docs": 0.9,
        "operator_kind": operator_kind,
        "axis": None,
    }


def test_locus_ancestry_does_not_create_automatic_docs_cohort():
    """A large locus peel is grammar ancestry, not prose parallelism."""
    from imas_codex.standard_names.context import fetch_sibling_family

    siblings = [
        _sibling(f"radial_coordinate_of_distinct_owner_{index}", "locus")
        for index in range(34)
    ]
    gc = _SiblingFamilyGC(
        {
            "parent_id": "radial_coordinate",
            "parent_description": "Radial coordinate.",
            "parent_documentation": "The radial coordinate locates a point.",
            "parent_docs_stage": "accepted",
            "member_operator_kind": "locus",
            "sibs": siblings,
        }
    )

    family = fetch_sibling_family("radial_coordinate_of_control_surface", gc=gc)

    assert family is None
    assert len(siblings) + 1 == 35
    assert "locus" not in gc.calls[0][1]["family_kinds"]


def test_safe_projection_family_remains_available_to_docs():
    """Axis projections remain useful parallel-structure comparators."""
    from imas_codex.standard_names.context import fetch_sibling_family

    gc = _SiblingFamilyGC(
        {
            "parent_id": "mode_number",
            "parent_description": "Mode number components.",
            "parent_documentation": "Mode numbers label periodic structure.",
            "parent_docs_stage": "accepted",
            "member_operator_kind": "projection",
            "sibs": [
                _sibling("poloidal_mode_number", "projection"),
                _sibling("toroidal_mode_number", "projection"),
            ],
        }
    )

    family = fetch_sibling_family("radial_mode_number", gc=gc)

    assert family is not None
    assert [member["name"] for member in family["siblings"]] == [
        "poloidal_mode_number",
        "toroidal_mode_number",
    ]
    assert set(gc.calls[0][1]["family_kinds"]) == {
        "coordinate",
        "projection",
        "qualifier",
    }


def test_source_owners_are_not_automatic_prose_family_members():
    """Attached owners never become sibling members or identity evidence."""
    from imas_codex.standard_names.context import fetch_sibling_family

    gc = _SiblingFamilyGC(None)
    source_owners = [f"outline_owner_{index}" for index in range(10)]

    assert fetch_sibling_family("vertical_outline", gc=gc) is None
    assert len(source_owners) == 10
    cypher = gc.calls[0][0]
    assert "StandardNameSource" not in cypher
    assert "PRODUCED_NAME" not in cypher

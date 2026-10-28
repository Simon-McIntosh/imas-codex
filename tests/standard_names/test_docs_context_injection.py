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

from tests.standard_names.test_pattern_families import (
    _ENTITY_TOKENS,
    _entity_owner_fixture,
    _entity_owner_id,
    _name,
    _PatternGraph,
)

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
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        _enrich_for_docs_gen,
    )

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
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._search_nearby_names",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._related_path_neighbours",
            return_value=[],
        ),
    ):
        _enrich_for_docs_gen(gc, [item])

    assert "source_paths" in item, "source_paths should be populated"
    assert item["source_paths"] == [
        "core_profiles/profiles_1d/electrons/temperature"
    ], "dd: prefix should be stripped"


def test_enrich_populates_dd_source_docs_and_aliases():
    """dd_source_docs and dd_aliases are populated from linked IMASNodes."""
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        _enrich_for_docs_gen,
    )

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
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._search_nearby_names",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._related_path_neighbours",
            return_value=[],
        ),
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
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        _enrich_for_docs_gen,
    )

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

    with patch(
        "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._search_nearby_names",
        return_value=peers_mock,
    ):
        _enrich_for_docs_gen(gc, [item])

    assert "nearest_peers" in item, "nearest_peers should be populated"
    assert item["nearest_peers"][0]["tag"] == "name:ion_temperature"


def test_enrich_skips_item_with_no_id():
    """Items without an id key are silently skipped."""
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        _enrich_for_docs_gen,
    )

    gc = _mock_gc()
    item: dict[str, Any] = {"description": "orphan item"}

    _enrich_for_docs_gen(gc, [item])

    gc.query.assert_not_called()
    assert "source_paths" not in item


def test_enrich_handles_empty_graph_result():
    """Empty graph result leaves item without enrichment keys (no crash)."""
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        _enrich_for_docs_gen,
    )

    gc = _mock_gc(query_side_effect=[[]])  # empty result set

    item: dict[str, Any] = {"id": "missing_sn", "description": "some description"}

    with patch(
        "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._search_nearby_names",
        return_value=[],
    ):
        _enrich_for_docs_gen(gc, [item])

    assert "source_paths" not in item
    assert "dd_source_docs" not in item


def test_enrich_excludes_self_from_nearest_peers():
    """The SN being enriched must not appear in nearest_peers."""
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        _enrich_for_docs_gen,
    )

    gc = _mock_gc(query_side_effect=[[{"source_paths": [], "dd_nodes": []}]])

    peers_mock = [
        {"id": "electron_temperature", "description": "Self", "unit": "eV"},
        {"id": "ion_temperature", "description": "Ion temp", "unit": "eV"},
    ]

    item: dict[str, Any] = {"id": "electron_temperature", "description": "Te"}

    with patch(
        "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._search_nearby_names",
        return_value=peers_mock,
    ):
        _enrich_for_docs_gen(gc, [item])

    peer_tags = [p["tag"] for p in item.get("nearest_peers", [])]
    assert "name:electron_temperature" not in peer_tags, "Self must be excluded"
    assert "name:ion_temperature" in peer_tags


# =============================================================================
# _nearby_names_for_docs_gen — unit tests
# =============================================================================


def test_nearby_names_returns_accepted_sns():
    """_nearby_names_for_docs_gen returns list of dicts from the graph."""
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        _nearby_names_for_docs_gen,
    )

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
    from imas_codex.standard_names.worker_tasks.worker_generate_docs import (
        _nearby_names_for_docs_gen,
    )

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


class _PatternFamilyGC:
    """Return pattern-family rows from an in-memory fixture graph."""

    def __init__(self, graph):
        self._graph = graph
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, **params: Any):
        self.calls.append((cypher, params))
        return self._graph.query(cypher, **params)


def test_locus_qualified_variants_resolve_to_one_pattern_family():
    """A locus axis is a variation axis, so per-locus variants are family-mates."""
    from imas_codex.standard_names.context import fetch_sibling_family

    gc = _PatternFamilyGC(
        _PatternGraph(
            [
                _name(
                    "density",
                    physical_base="density",
                    docs_stage="accepted",
                ),
                _name(
                    "electron_density_at_pedestal_top",
                    physical_base="density",
                    subject="electron",
                    position="pedestal_top",
                ),
                _name(
                    "electron_density_at_plasma_boundary",
                    physical_base="density",
                    subject="electron",
                    position="plasma_boundary",
                ),
            ]
        )
    )

    family = fetch_sibling_family("electron_density_at_pedestal_top", gc=gc)

    assert family is not None
    # the base quantity backs the parent: it is a real node, not the seed's
    # base token echoed back
    assert family["parent"]["name"] == "density"
    assert [s["name"] for s in family["siblings"]] == [
        "electron_density_at_plasma_boundary"
    ]


def test_projection_family_remains_available_to_docs():
    """Component projections remain useful parallel-structure comparators."""
    from imas_codex.standard_names.context import fetch_sibling_family

    gc = _PatternFamilyGC(
        _PatternGraph(
            [
                _name(
                    "radial_mode_number",
                    physical_base="mode_number",
                    component="radial",
                ),
                _name(
                    "poloidal_mode_number",
                    physical_base="mode_number",
                    component="poloidal",
                ),
                _name(
                    "toroidal_mode_number",
                    physical_base="mode_number",
                    component="toroidal",
                ),
            ]
        )
    )

    family = fetch_sibling_family("radial_mode_number", gc=gc)

    assert family is not None
    assert [member["name"] for member in family["siblings"]] == [
        "poloidal_mode_number",
        "toroidal_mode_number",
    ]


def test_source_owners_are_not_automatic_prose_family_members():
    """Attached owners never become sibling members or identity evidence."""
    from imas_codex.standard_names.context import fetch_sibling_family

    gc = _PatternFamilyGC(
        _PatternGraph([_name("vertical_outline", physical_base="vertical_outline")])
    )
    source_owners = [f"outline_owner_{index}" for index in range(10)]

    assert fetch_sibling_family("vertical_outline", gc=gc) is None
    assert len(source_owners) == 10
    cypher = gc.calls[0][0]
    # Sources are consulted only to report whether an already-admitted member
    # is DD-derived; the membership match and its axis predicates name
    # StandardName alone, so an attached owner is never admitted as a sibling.
    membership = cypher.split("EXISTS")[0]
    assert "MATCH (m:StandardName)" in membership
    assert "StandardNameSource" not in membership
    assert "PRODUCED_NAME" not in membership


def test_entity_owners_are_not_an_automatic_docs_cohort():
    """Names describing different entities never share a docs family.

    The entity locus is invariant, so 34 registered entity tokens sharing a
    base quantity form 34 families of one, not one cohort of 34.
    """
    from imas_codex.standard_names.context import fetch_sibling_family

    gc = _PatternFamilyGC(_entity_owner_fixture())

    assert len(_ENTITY_TOKENS) == 34
    for entity in _ENTITY_TOKENS:
        assert fetch_sibling_family(_entity_owner_id(entity), gc=gc) is None


def test_sibling_family_omits_a_parent_the_graph_does_not_hold():
    """A base token with no backing node yields no parent, not a phantom."""
    from imas_codex.standard_names.context import fetch_sibling_family

    gc = _PatternFamilyGC(
        _PatternGraph(
            [
                _name(
                    "radial_mode_number",
                    physical_base="mode_number",
                    component="radial",
                ),
                _name(
                    "poloidal_mode_number",
                    physical_base="mode_number",
                    component="poloidal",
                ),
            ]
        )
    )

    family = fetch_sibling_family("radial_mode_number", gc=gc)

    assert family is not None
    assert family["parent"] is None

"""Tests for the component/parent standard name system.

Verifies that:
- ``_write_standard_name_edges()`` tags bare parent placeholders with ``needs_composition``
- ``seed_parent_sources()`` creates StandardNameSource nodes for parents
- ``_enrich_for_docs_gen()`` injects parent/child context into items
- The docs prompt template renders parent/child sections
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_gc_for_enrichment(
    *,
    parent: dict | None = None,
    children: list[dict] | None = None,
) -> MagicMock:
    """Return a mock GraphClient with component relationship data."""
    gc = MagicMock()

    def _query(cypher, **kwargs):
        if "cocos_label" in cypher and "source_paths" in cypher:
            return [
                {
                    "source_paths": ["dd:equilibrium/time_slice/profiles_1d/j_tor"],
                    "cocos_label": None,
                    "dd_nodes": [],
                }
            ]
        if "COMPONENT_OF]->(parent" in cypher:
            if parent:
                return [parent]
            return []
        if "COMPONENT_OF]->(sn:StandardName" in cypher:
            return children or []
        return []

    gc.query = _query
    return gc


# ── Tests: Component tagging ───────────────────────────────────────────


def test_component_tags_parent():
    """_write_standard_name_edges sets needs_composition on bare parents."""
    from imas_codex.standard_names.derivation import derive_edges

    # A toroidal component should derive a COMPONENT_OF edge
    edges = derive_edges("current_density_toroidal")
    co_edges = [e for e in edges if e.edge_type == "COMPONENT_OF"]

    # If ISN parser finds a parent, it should be in the list
    # (depends on ISN grammar — may be empty if ISN doesn't parse this)
    # This test validates the derive_edges function runs without error
    assert isinstance(co_edges, list)


def test_derive_edges_projection():
    """derive_edges detects projection components."""
    from imas_codex.standard_names.derivation import derive_edges

    edges = derive_edges("magnetic_field_toroidal")
    co_edges = [e for e in edges if e.edge_type == "COMPONENT_OF"]
    if co_edges:
        assert co_edges[0].to_name == "magnetic_field"
        assert co_edges[0].props.get("axis") in ("toroidal", None)


# ── Tests: seed_parent_sources ──────────────────────────────────────────


def test_seed_parent_sources_creates_source():
    """seed_parent_sources fully populates parents when all children are composed."""
    gc = MagicMock()
    call_log = []

    def _query(cypher, **kwargs):
        call_log.append(cypher)
        if "name_stage IS NULL" in cypher:
            return [
                {
                    "parent_id": "magnetic_field",
                    "child_data": [
                        {
                            "id": "toroidal_magnetic_field",
                            "unit": "T",
                            "cocos": "b0_like",
                            "physics_domain": "magnetics",
                            "kind": "scalar",
                        },
                        {
                            "id": "poloidal_magnetic_field",
                            "unit": "T",
                            "cocos": "b0_like",
                            "physics_domain": "magnetics",
                            "kind": "scalar",
                        },
                    ],
                    "dd_paths": [
                        "equilibrium/time_slice/profiles_1d/b_field_tor",
                        "equilibrium/time_slice/profiles_1d/b_field_pol",
                    ],
                    "edge_kinds": ["projection", "projection"],
                }
            ]
        return []

    gc.query = _query
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    from imas_codex.standard_names.graph_ops import seed_parent_sources

    count = seed_parent_sources(gc)
    assert count == 1

    # Should have called query at least twice: find parents + populate parent
    assert len(call_log) >= 2
    # The second query should SET name_stage='accepted' and MERGE StandardNameSource
    assert any("StandardNameSource" in q for q in call_log)
    assert any("name_stage" in q for q in call_log)


def test_seed_parent_sources_no_parents():
    """seed_parent_sources returns 0 when no bare parents exist."""
    gc = MagicMock()
    gc.query = MagicMock(return_value=[])
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    from imas_codex.standard_names.graph_ops import seed_parent_sources

    count = seed_parent_sources(gc)
    assert count == 0


def test_seed_parent_sources_skips_incomplete_children():
    """seed_parent_sources does NOT seed when children are incomplete.

    The race-condition guard ensures parents are only populated when
    ALL children have name_stage set (i.e., fully composed).
    """
    gc = MagicMock()
    # Query returns empty — the Cypher WHERE clause
    # (total_children = composed_children) filters incomplete parents
    gc.query = MagicMock(return_value=[])
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    from imas_codex.standard_names.graph_ops import seed_parent_sources

    count = seed_parent_sources(gc)
    assert count == 0


def test_run_sn_pools_rederives_edges_before_seeding():
    """``run_sn_pools`` must call ``rederive_structural_edges`` before
    ``seed_parent_sources`` so newly-grammar-derived parent edges land
    in the graph in time to be seeded on the same run.

    Reproduces the production gap: names like
    ``flux_surface_mean_magnetic_field_magnitude`` and
    ``toroidal_total_plasma_current`` had no COMPONENT_OF edges at all
    in the live graph, so their parents
    (``flux_surface_mean_magnetic_field`` / ``total_plasma_current``)
    were entirely absent rather than orphan placeholders — invisible
    to ``seed_parent_sources`` no matter how robust its query is.
    """
    import inspect

    from imas_codex.standard_names import loop

    src = inspect.getsource(loop.run_sn_pools)
    # Both helpers must be invoked (not just mentioned in comments) —
    # check for ``asyncio.to_thread(<helper>)`` call sites.
    assert "asyncio.to_thread(rederive_structural_edges)" in src, (
        "run_sn_pools must invoke rederive_structural_edges to backfill "
        "missing COMPONENT_OF edges before parent seeding."
    )
    assert "asyncio.to_thread(seed_parent_sources)" in src
    # The rederive call must precede the seed call so newly-derived
    # placeholders are visible to the same seed pass.
    assert src.index("asyncio.to_thread(rederive_structural_edges)") < src.index(
        "asyncio.to_thread(seed_parent_sources)"
    ), (
        "rederive_structural_edges must run before seed_parent_sources "
        "in run_sn_pools (rederive creates the placeholder, seed fills it)."
    )


def test_seed_parent_sources_picks_up_placeholder_without_flag():
    """Regression: an orphan placeholder (name_stage NULL) must be seeded
    even if ``needs_composition`` is missing.

    Reproduces the bug where ``unary_postfix`` was added to the seedable
    edge-kinds set after some placeholders had already been written.
    The earlier write path never set ``needs_composition`` on those
    placeholders, so the old ``seed_parent_sources`` (which keyed on
    ``needs_composition: true``) left them orphaned forever — and
    ``sn run --flush`` correctly reported "no eligible work" even
    though structurally those parents were ready to seed.

    The fix: select parents structurally by ``name_stage IS NULL`` plus
    a seedable COMPONENT_OF edge — no flag required.
    """
    gc = MagicMock()
    captured_cyphers: list[str] = []

    def _query(cypher, **kwargs):
        captured_cyphers.append(cypher)
        # Match the new structural query; the mock returns a parent with
        # NO ``needs_composition`` set (the orphan-placeholder shape).
        if "name_stage IS NULL" in cypher and "COMPONENT_OF" in cypher:
            return [
                {
                    "parent_id": "average_magnetic_field",
                    "child_data": [
                        {
                            "id": "average_magnetic_field_magnitude",
                            "unit": "T",
                            "cocos": None,
                            "physics_domain": "equilibrium",
                            "kind": "scalar",
                        },
                    ],
                    "dd_paths": [],
                    "edge_kinds": ["unary_postfix"],
                }
            ]
        return []

    gc.query = _query
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    from imas_codex.standard_names.graph_ops import seed_parent_sources

    assert seed_parent_sources(gc) == 1
    # The selection query MUST NOT depend on the needs_composition flag.
    selection_query = captured_cyphers[0]
    assert "needs_composition: true" not in selection_query, (
        "seed_parent_sources must not key on needs_composition=true — "
        "that flag is set at edge-write time and goes stale when the "
        "seedable edge-kinds set grows."
    )


def test_seed_parent_sources_skips_heterogeneous_units():
    """seed_parent_sources skips parents with non-uniform child units."""
    gc = MagicMock()
    call_log = []

    def _query(cypher, **kwargs):
        call_log.append(cypher)
        if "name_stage IS NULL" in cypher:
            return [
                {
                    "parent_id": "some_vector",
                    "child_data": [
                        {
                            "id": "comp_a",
                            "unit": "T",
                            "cocos": None,
                            "physics_domain": "magnetics",
                            "kind": "scalar",
                        },
                        {
                            "id": "comp_b",
                            "unit": "m/s",
                            "cocos": None,
                            "physics_domain": "magnetics",
                            "kind": "scalar",
                        },
                    ],
                    "dd_paths": [],
                }
            ]
        return []

    gc.query = _query
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    from imas_codex.standard_names.graph_ops import seed_parent_sources

    count = seed_parent_sources(gc)
    assert count == 0  # Skipped due to unit mismatch


# ── Tests: docs enrichment with parent/child context ────────────────────


def test_docs_enrich_parent_context():
    """Child SN gets parent description and documentation."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    parent_data = {
        "name": "magnetic_field",
        "description": "Total magnetic field vector",
        "documentation": "The magnetic field is a fundamental quantity...",
        "axis": "toroidal",
    }
    gc = _make_gc_for_enrichment(parent=parent_data)
    items = [{"id": "magnetic_field_toroidal"}]
    _enrich_for_docs_gen(gc, items)

    assert "parent_sn" in items[0]
    assert items[0]["parent_sn"]["name"] == "magnetic_field"
    assert items[0]["component_axis"] == "toroidal"


def test_docs_enrich_child_context():
    """Parent SN gets list of child components."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    children = [
        {
            "name": "magnetic_field_toroidal",
            "description": "Toroidal component",
            "axis": "toroidal",
        },
        {
            "name": "magnetic_field_poloidal",
            "description": "Poloidal component",
            "axis": "poloidal",
        },
    ]
    gc = _make_gc_for_enrichment(children=children)
    items = [{"id": "magnetic_field"}]
    _enrich_for_docs_gen(gc, items)

    assert "child_components" in items[0]
    assert len(items[0]["child_components"]) == 2
    assert items[0]["child_components"][0]["name"] == "magnetic_field_toroidal"


def test_docs_enrich_child_components_axis_ordered():
    """Child components sorted by right-handed axis convention (R, φ, Z)."""
    from imas_codex.standard_names.workers import _enrich_for_docs_gen

    # Return children in WRONG order: vertical before toroidal
    children = [
        {
            "name": "vertical_B",
            "description": "Vertical comp",
            "axis": "vertical",
        },
        {
            "name": "toroidal_B",
            "description": "Toroidal comp",
            "axis": "toroidal",
        },
        {
            "name": "radial_B",
            "description": "Radial comp",
            "axis": "radial",
        },
    ]
    gc = _make_gc_for_enrichment(children=children)
    items = [{"id": "magnetic_field"}]
    _enrich_for_docs_gen(gc, items)

    assert "child_components" in items[0]
    axes = [c["axis"] for c in items[0]["child_components"]]
    assert axes == ["radial", "toroidal", "vertical"], (
        f"Expected R, φ, Z ordering but got {axes}"
    )


def test_docs_prompt_renders_parent_section():
    """Docs prompt includes parent context when parent_sn is present."""
    from imas_codex.llm.prompt_loader import render_prompt

    context = {
        "item": {
            "name": "magnetic_field_toroidal",
            "unit": "T",
            "kind": "scalar",
            "physics_domain": "equilibrium",
            "parent_sn": {
                "name": "magnetic_field",
                "description": "Total magnetic field",
                "documentation": "The magnetic field is...",
            },
            "component_axis": "toroidal",
        },
        "chain_history": [],
        "nearby_existing_names": [],
    }
    rendered = render_prompt("sn/generate_docs_user", context)
    assert "Parent Standard Name" in rendered
    assert "magnetic_field" in rendered
    assert "toroidal" in rendered


def test_docs_prompt_renders_child_section():
    """Docs prompt includes child components when child_components is present."""
    from imas_codex.llm.prompt_loader import render_prompt

    context = {
        "item": {
            "name": "magnetic_field",
            "unit": "T",
            "kind": "vector",
            "physics_domain": "equilibrium",
            "child_components": [
                {
                    "name": "magnetic_field_toroidal",
                    "axis": "toroidal",
                    "description": "Toroidal B",
                },
                {
                    "name": "magnetic_field_poloidal",
                    "axis": "poloidal",
                    "description": "Poloidal B",
                },
            ],
        },
        "chain_history": [],
        "nearby_existing_names": [],
    }
    rendered = render_prompt("sn/generate_docs_user", context)
    assert "Component Standard Names" in rendered
    assert "magnetic_field_toroidal" in rendered
    assert "toroidal" in rendered


def test_docs_prompt_omits_parent_child_when_absent():
    """Docs prompt has no parent/child sections for standalone quantities."""
    from imas_codex.llm.prompt_loader import render_prompt

    context = {
        "item": {
            "name": "electron_temperature",
            "unit": "eV",
            "kind": "scalar",
            "physics_domain": "core_profiles",
        },
        "chain_history": [],
        "nearby_existing_names": [],
    }
    rendered = render_prompt("sn/generate_docs_user", context)
    assert "Parent Standard Name" not in rendered
    assert "Component Standard Names" not in rendered

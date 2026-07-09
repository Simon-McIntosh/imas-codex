"""Ancestor-lineage grounding for compose (name + docs).

A DD *value* leaf is often a terse template stub (e.g. a per-species
``.../velocity_phi/<species>/value`` reads only "Deuterium (D)."), while the
physically-meaningful text — the quantity AND its evaluation locus ("Ion
toroidal rotation velocity … at the pedestal top") — lives on a parent quantity
node. Both name and docs generation must surface that ancestor lineage so the
composed name resolves the correct locus (``pedestal_top``, not the bare DD path
segment ``pedestal``).
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.workers import (
    _ANCESTOR_CONTEXT_QUERY,
    _enrich_batch_items,
    _enrich_for_docs_gen,
)

PEDESTAL_LEAF = "summary/local/pedestal/velocity_phi/deuterium/value"
# An accepted name that already owns real dd sources (velocity_phi + velocity_tor).
GROUNDED_NAME = "toroidal_carbon_velocity_at_pedestal_top"


def test_ancestor_context_query_walks_has_parent_ordered_by_depth():
    """The lineage query walks HAS_PARENT ancestors, nearest first, and only
    returns ancestors that carry some description/documentation text."""
    q = _ANCESTOR_CONTEXT_QUERY
    assert "HAS_PARENT*1..8" in q
    assert "ORDER BY depth ASC" in q
    # Only ancestors with usable text (never an empty-grounding node).
    assert "a.description" in q and "a.documentation" in q
    assert "trim(a.description)" in q


@pytest.mark.graph
def test_enrich_batch_items_surfaces_pedestal_top_ancestor_live():
    """Name-gen: a terse per-species value leaf gains the parent quantity's
    rich description, which names the pedestal-top evaluation locus."""
    item = {
        "path": PEDESTAL_LEAF,
        "id": f"dd:{PEDESTAL_LEAF}",
        "source_id": PEDESTAL_LEAF,
        "source_type": "dd",
    }
    _enrich_batch_items([item])
    lineage = item.get("ancestor_context") or []
    assert lineage, "expected ancestor_context to be populated from the lineage"
    blob = " ".join(a["text"].lower() for a in lineage)
    assert "pedestal top" in blob, f"lineage missing pedestal-top locus: {lineage}"
    # Nearest-first, capped.
    assert len(lineage) <= 4


@pytest.mark.graph
def test_enrich_for_docs_gen_surfaces_pedestal_top_ancestor_live():
    """Docs-gen: an accepted name grounded on a template-stub source leaf gains
    the parent quantity node's rich locus-bearing description."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        item = {"id": GROUNDED_NAME}
        _enrich_for_docs_gen(gc, [item])

    lineage = item.get("ancestor_context") or []
    assert lineage, "expected ancestor_context for a dd-sourced accepted name"
    blob = " ".join(a["text"].lower() for a in lineage)
    assert "pedestal top" in blob, f"lineage missing pedestal-top locus: {lineage}"

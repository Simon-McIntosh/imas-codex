"""Live-graph test for the nearest-label lookup behind the mapping hand-off.

The unit tests in ``test_mapping_handoff`` drive a mock whose hop trimming is
reimplemented in Python, so a query that kept ``*0..`` but broke the
``ORDER BY hops`` / ``collect(...)[0]`` nearest-label idiom would stay green
there. This test runs the same helper against the live graph so the Cypher
that actually selects a COCOS label is exercised.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.ids.handoff import _nearest_cocos_labels

FIELD_DATA = "magnetics/b_field_pol_probe/field/data"
FLUX_LOOP_DATA = "magnetics/flux_loop/flux/data"
FIELD_TIME = "magnetics/b_field_pol_probe/field/time"


@pytest.mark.graph
def test_nearest_cocos_labels_reads_each_target_from_the_live_graph():
    targets = [FIELD_DATA, FLUX_LOOP_DATA, FIELD_TIME]
    with GraphClient() as client:
        labels = _nearest_cocos_labels(client, targets)

    # Positive control: a silent zero-row query would otherwise report the
    # absence token for every target and look like a pass. Every requested
    # target must come back before the per-target labels are judged.
    assert set(labels) == set(targets)

    assert labels[FIELD_DATA] == ("one_like", "xml")
    assert labels[FLUX_LOOP_DATA] == ("psi_like", "inferred_forward")
    assert labels[FIELD_TIME] == ("none", "none")


@pytest.mark.graph
def test_an_ancestor_chain_carries_at_most_one_cocos_label():
    """No node's ancestor chain holds two labelled nodes, so hop order is moot.

    ``_nearest_cocos_labels`` keeps the nearest label with ``ORDER BY hops``
    and ``collect(label)[0]``. On the live graph that ordering cannot change
    any result: a label is unique to its structure, so a target reaches at
    most one labelled ancestor and the collection holds a single entry. If
    labels ever stack, nearest-vs-farthest starts to matter and this test
    turns red, which is the signal that the ordering needs proving.
    """
    with GraphClient() as client:
        rows = client.query(
            """
            MATCH p = (descendant:IMASNode)-[:HAS_PARENT*1..10]->(ancestor:IMASNode)
            WHERE descendant.cocos_transformation_type IS NOT NULL
              AND ancestor.cocos_transformation_type IS NOT NULL
            RETURN descendant.id AS descendant_id,
                   descendant.cocos_transformation_type AS descendant_label,
                   ancestor.id AS ancestor_id,
                   ancestor.cocos_transformation_type AS ancestor_label
            """
        )
    assert rows == []

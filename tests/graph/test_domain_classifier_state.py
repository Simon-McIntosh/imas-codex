"""The classifier's judgement context must carry what the path actually says.

A summary leaf such as ``summary/boundary/elongation/value`` carries its
meaning in the LLM-enriched description, while its DD documentation clause is
only the placeholder ``Value``. The classifier's decision therefore depends on
the enriched description, the parent's description and the siblings, reaching
both the Choice tree (through ``tree_state``) and the chat fallback (through
``_format_batch_user_prompt``). These tests hold that state together, and hold
the reverse case: a leaf whose enriched description is missing carries no
subject of its own, so it must not be classified on the placeholder alone but
left to inherit its parent's domain.

All graph access is stubbed; no Neo4j connection and no LLM call is made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.graph.dd_domain_classifier import (
    _format_batch_user_prompt,
    _query_unclassified_paths,
    carries_own_subject,
    gather_classification_context,
    tree_state,
)

LEAF_ID = "summary/boundary/elongation/value"
ENRICHED = "Elongation of the plasma boundary."
PARENT_DESCRIPTION = "Elongation of the plasma boundary."
SIBLINGS = ["error_upper", "error_lower", "index"]


def _leaf_row(**overrides):
    """One graph row as ``gather_classification_context`` would read it.

    The enriched description lives in ``description``; ``documentation`` is the
    DD clause, which for a container's value leaf is the placeholder ``Value``.
    """
    row = {
        "id": LEAF_ID,
        "description": ENRICHED,
        "documentation": "Value",
        "units": "",
        "parent_path": "summary/boundary/elongation",
        "parent_description": PARENT_DESCRIPTION,
        "siblings": list(SIBLINGS),
        "ids_name": "summary",
        "node_category": "quantity",
    }
    row.update(overrides)
    return row


def _stub_gc(rows):
    gc = MagicMock()
    gc.query.return_value = rows
    return gc


def test_state_carries_the_description_parent_description_and_siblings():
    """A value leaf reaches the tree with the context that identifies it.

    The documentation clause alone is the placeholder ``Value``, so a state
    without the enriched description, the parent's description and the siblings
    would ask the tree to name a domain from a word that names no physics.
    """
    context = gather_classification_context(_stub_gc([_leaf_row()]), [LEAF_ID])
    assert len(context) == 1, "stubbed graph returned no context"
    state = tree_state(context[0])

    assert state["description"] == ENRICHED
    assert state["documentation"] == "Value"
    assert state["parent_description"] == PARENT_DESCRIPTION
    assert state["siblings"] == SIBLINGS


def test_chat_fallback_prompt_renders_the_same_state():
    """The chat re-judgement reads the same fields the tree was shown."""
    context = gather_classification_context(_stub_gc([_leaf_row()]), [LEAF_ID])
    prompt = _format_batch_user_prompt([tree_state(context[0])])

    assert ENRICHED in prompt
    assert PARENT_DESCRIPTION in prompt
    for sibling in SIBLINGS:
        assert sibling in prompt, f"sibling missing from the prompt: {sibling}"


def test_a_leaf_without_an_enriched_description_is_not_judged_on_its_placeholder():
    """With no enriched description, ``Value`` is not a subject.

    The leaf states nothing of its own, so it is left to the inheritance pass
    rather than classified on the container placeholder.
    """
    assert carries_own_subject(_leaf_row(description="")) is False
    assert carries_own_subject(_leaf_row(description=ENRICHED)) is True


def test_the_placeholder_leaf_is_not_a_tier_one_candidate():
    """The selection the classifier runs must apply the same rule.

    Both leaves are returned by the query, and only the one carrying a subject
    reaches Tier 1 — so the other falls through to inheritance instead of being
    judged on ``Value``.
    """
    described = _leaf_row(id="summary/boundary/minor_radius/value")
    placeholder = _leaf_row(id=LEAF_ID, description="")
    rows = [described, placeholder]

    selected = _query_unclassified_paths(_stub_gc(rows))

    ids = [row["id"] for row in selected]
    assert described["id"] in ids
    assert LEAF_ID not in ids, "a leaf with no subject of its own reached Tier 1"

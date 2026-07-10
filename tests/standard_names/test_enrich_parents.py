"""enrich_parents pool — derived-parent description synthesis (coverage break).

A derived parent materialised with the deterministic placeholder description is
deadlocked: REVIEW_NAME drops it (placeholder ≠ real description), so it never
earns a ``reviewer_score_name``, so generate_docs drops it too. The
``enrich_parents`` pool synthesises a real description GENERALISED over the
parent's accepted children, embeds it locally, and routes the parent to
``name_stage='drafted'`` so it flows review → accept → docs.

Covers:
- claim eligibility predicate (origin=derived AND placeholder AND has-child)
- persist routes to ``drafted`` (no score) vs ``accepted`` (already scored)
- persist no-op when the node no longer matches
- worker: childless item is released (not enriched); normal item is enriched
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER as PLACEHOLDER,
)


def _mock_budget_manager() -> MagicMock:
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    mgr.run_id = "run-test"
    return mgr


def _verify_gc(rows: list[dict]) -> MagicMock:
    """Mock GraphClient whose single query returns the given winner rows."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=rows)
    return gc


# ---------------------------------------------------------------------------
# Claim-race verification (two-step claim_token verify)
# ---------------------------------------------------------------------------


class TestVerifyEnrichParentsClaimWinners:
    """A losing-race replica must spend zero enrichment LLM calls: the verifier
    re-reads committed state and drops nodes whose token was overwritten."""

    def test_drops_items_whose_token_lost(self):
        import imas_codex.standard_names.graph_ops as g

        items = [
            {"id": "won_a", "claim_token": "tok-1"},
            {"id": "lost_b", "claim_token": "tok-1"},  # overwritten by a racer
            {"id": "won_c", "claim_token": "tok-1"},
        ]
        gc = _verify_gc([{"id": "won_a"}, {"id": "won_c"}])
        with patch.object(g, "GraphClient", return_value=gc):
            survivors = g._verify_enrich_parents_claim_winners(items)

        assert [it["id"] for it in survivors] == ["won_a", "won_c"]
        cypher = gc.query.call_args_list[0][0][0]
        assert "sn.claim_token = $token" in cypher
        assert "sn.origin = 'derived'" in cypher
        assert "sn.description = $placeholder" in cypher
        assert gc.query.call_args_list[0][1]["token"] == "tok-1"

    def test_all_survive_returns_unchanged(self):
        import imas_codex.standard_names.graph_ops as g

        items = [{"id": "a", "claim_token": "t"}, {"id": "b", "claim_token": "t"}]
        gc = _verify_gc([{"id": "a"}, {"id": "b"}])
        with patch.object(g, "GraphClient", return_value=gc):
            assert g._verify_enrich_parents_claim_winners(items) == items

    def test_empty_items_short_circuit(self):
        import imas_codex.standard_names.graph_ops as g

        gc = _verify_gc([])
        with patch.object(g, "GraphClient", return_value=gc):
            assert g._verify_enrich_parents_claim_winners([]) == []
        gc.query.assert_not_called()


# ---------------------------------------------------------------------------
# Claim eligibility
# ---------------------------------------------------------------------------


def test_claim_eligibility_predicate():
    """The claim WHERE gates on derived-origin + placeholder + a live child."""
    import imas_codex.standard_names.graph_ops as g

    captured: dict[str, Any] = {}

    def _fake_claim(*, eligibility_where, query_params, **kwargs):
        captured["where"] = eligibility_where
        captured["params"] = query_params
        captured["extra"] = kwargs.get("extra_return_fields", "")
        return []

    with patch.object(g, "_claim_sn_atomic", side_effect=_fake_claim):
        g.claim_enrich_parents_batch(batch_size=5)

    where = captured["where"]
    assert "sn.origin = 'derived'" in where
    assert "sn.description = $parent_desc_placeholder" in where
    # must require at least one LIVE child (the childless-skip guarantee)
    assert "HAS_PARENT" in where
    assert "superseded" in where and "exhausted" in where
    assert captured["params"]["parent_desc_placeholder"] == PLACEHOLDER
    # claim does NOT transition stage (no stage_field/to_stage)
    assert "stage_field" not in captured or captured.get("stage_field") is None


# ---------------------------------------------------------------------------
# Persist routing
# ---------------------------------------------------------------------------


def _persist_gc(name_stage_returned: str | None):
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    rows = [{"name_stage": name_stage_returned}] if name_stage_returned else []
    # call 1: the SET; call 2: the source-mirror write (only if call 1 matched)
    gc.query = MagicMock(side_effect=[rows, []])
    return gc


def test_persist_accepts_structurally():
    """A derived parent skips REVIEW_NAME — accepted directly with the
    structural-inheritance marker and NO fabricated name score (the name is a
    deterministic peel, never reviewed, so scoring it is meaningless)."""
    import imas_codex.standard_names.graph_ops as g

    gc = _persist_gc("accepted")
    with patch.object(g, "GraphClient", return_value=gc):
        stage = g.persist_enriched_parent(
            sn_id="magnetic_field",
            claim_token="tok",
            description="The vector magnetic field.",
            embedding=[0.1, 0.2, 0.3],
            model="test/model",
        )
    assert stage == "accepted"
    set_cypher, set_params = (
        gc.query.call_args_list[0][0][0],
        gc.query.call_args_list[0][1],
    )
    # Accepted directly (no review routing), marked structural, NO name score.
    assert "sn.name_stage = 'accepted'" in set_cypher
    assert "structural-inheritance" in set_cypher
    # No fabricated name score is SET (the comment may mention the field).
    assert "sn.reviewer_score_name =" not in set_cypher
    assert "name_stage = CASE" not in set_cypher
    assert set_params["embedding"] == [0.1, 0.2, 0.3]


def test_persist_noop_when_node_unmatched():
    import imas_codex.standard_names.graph_ops as g

    gc = _persist_gc(None)  # SET matched nothing
    with patch.object(g, "GraphClient", return_value=gc):
        stage = g.persist_enriched_parent(
            sn_id="ghost",
            claim_token="tok",
            description="x" * 20,
            embedding=None,
            model="test/model",
        )
    assert stage == ""


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_skips_and_releases_childless_parent():
    """A claimed parent with no live children is released, never enriched."""
    from imas_codex.standard_names import workers as w

    mgr = _mock_budget_manager()
    stop = asyncio.Event()
    items = [
        {"id": "orphan_parent", "claim_token": "tok", "unit": "m", "kind": "scalar"}
    ]

    released: dict[str, Any] = {}

    def _release(*, sn_ids, claim_token):
        released["ids"] = sn_ids
        return len(sn_ids)

    with (
        patch(
            "imas_codex.standard_names.graph_ops.fetch_derived_parent_children",
            return_value={"orphan_parent": []},
        ),
        patch(
            "imas_codex.standard_names.graph_ops.release_enrich_parents_claims",
            side_effect=_release,
        ),
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
        patch("imas_codex.standard_names.graph_ops.persist_enriched_parent") as persist,
    ):
        count = await w.process_enrich_parents_batch(items, mgr, stop)

    assert count == 0
    assert released.get("ids") == ["orphan_parent"]
    persist.assert_not_called()


@pytest.mark.asyncio
async def test_worker_enriches_parent_with_children():
    """Happy path: children grounded → LLM desc → embed → persist (drafted)."""
    from imas_codex.standard_names import workers as w

    mgr = _mock_budget_manager()
    stop = asyncio.Event()
    items = [
        {
            "id": "magnetic_field",
            "claim_token": "tok",
            "unit": "T",
            "kind": "vector",
            "physics_domain": "magnetics",
        }
    ]
    children = {
        "magnetic_field": [
            {
                "name": "radial_magnetic_field",
                "description": "Radial component of B.",
                "unit": "T",
                "physics_domain": "magnetics",
                "kind": "scalar",
            },
            {
                "name": "toroidal_magnetic_field",
                "description": "Toroidal component of B.",
                "unit": "T",
                "physics_domain": "magnetics",
                "kind": "scalar",
            },
        ]
    }

    llm_result = SimpleNamespace(description="The vector magnetic field B.")
    persisted: dict[str, Any] = {}

    def _persist(*, sn_id, claim_token, description, embedding, model, run_id=None):
        persisted.update(
            sn_id=sn_id, description=description, embedding=embedding, model=model
        )
        return "accepted"  # derived parents accept structurally (skip review)

    with (
        patch(
            "imas_codex.standard_names.graph_ops.fetch_derived_parent_children",
            return_value=children,
        ),
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new=AsyncMock(return_value=(llm_result, 0.001, 120)),
        ),
        patch(
            "imas_codex.embeddings.description.embed_description",
            return_value=[0.1, 0.2, 0.3],
        ),
        patch(
            "imas_codex.standard_names.graph_ops.persist_enriched_parent",
            side_effect=_persist,
        ),
    ):
        count = await w.process_enrich_parents_batch(items, mgr, stop)

    assert count == 1
    assert persisted["sn_id"] == "magnetic_field"
    assert persisted["embedding"] == [0.1, 0.2, 0.3]
    assert "magnetic field" in persisted["description"].lower()

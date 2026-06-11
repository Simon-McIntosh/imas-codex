"""Tests for per-pool cost attribution.

Validates that:
1. ``write_standard_names`` accumulates ``llm_cost_generate_name`` with ``+=``
2. Repeated writes (regeneration) accumulate, not overwrite
3. ``write_reviews`` propagates ``llm_cost_review_name`` / ``llm_cost_review_docs``
4. ``llm_cost`` aggregate tracks the sum of per-pool costs
5. ``sn clear`` (DETACH DELETE) wipes all cost fields
6. ``write_reviews`` passes ``llm_tokens_cached_read/write`` through
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_queries(mock_gc: MagicMock) -> list[tuple[str, dict]]:
    """Return ``[(cypher, kwargs), ...]`` from a mock GraphClient.query."""
    calls = []
    for call in mock_gc.query.call_args_list:
        cypher = call.args[0] if call.args else ""
        kwargs = dict(call.kwargs)
        calls.append((cypher, kwargs))
    return calls


def _make_gc_context(mock_gc: MagicMock):
    """Build a context-manager mock that returns *mock_gc*."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_gc)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Phase 2b — write_standard_names compose cost accumulation
# ---------------------------------------------------------------------------


class TestComposeCostAccumulation:
    """``write_standard_names`` must use ``+=`` for cost fields."""

    def _cypher_from_write(self, names: list[dict]) -> str:
        """Run write_standard_names under a mock and return the MERGE Cypher."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=_make_gc_context(mock_gc),
        ):
            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        # Find the MERGE Cypher (the first query with 'MERGE (sn:StandardName')
        for cypher, _ in _capture_queries(mock_gc):
            if "MERGE (sn:StandardName" in cypher:
                return cypher
        pytest.fail("No MERGE StandardName Cypher found")

    def test_compose_cost_uses_accumulation(self):
        """llm_cost_generate_name uses += (not coalesce-set)."""
        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "llm_cost": 0.005,
            }
        ]
        cypher = self._cypher_from_write(names)

        # Must contain accumulation pattern for llm_cost_generate_name
        assert "sn.llm_cost_generate_name" in cypher
        assert "coalesce(sn.llm_cost_generate_name, 0.0)" in cypher

    def test_compose_count_increments(self):
        """generate_name_count increments by 1 on each write."""
        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "llm_cost": 0.005,
            }
        ]
        cypher = self._cypher_from_write(names)

        assert "sn.generate_name_count" in cypher
        assert "coalesce(sn.generate_name_count, 0) + 1" in cypher

    def test_regen_detected_via_generate_name_count(self):
        """llm_cost_refine_name only accumulates when generate_name_count > 0."""
        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "llm_cost": 0.005,
            }
        ]
        cypher = self._cypher_from_write(names)

        # Regen detection: CASE WHEN sn.generate_name_count > 0
        assert "sn.llm_cost_refine_name" in cypher
        assert "sn.generate_name_count" in cypher
        # Should have a CASE expression checking generate_name_count > 0
        assert re.search(
            r"sn\.generate_name_count\s+IS\s+NOT\s+NULL.*sn\.generate_name_count\s*>\s*0",
            cypher,
            re.DOTALL,
        )

    def test_aggregate_llm_cost_also_accumulated(self):
        """The aggregate llm_cost field must also use += pattern."""
        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "llm_cost": 0.005,
            }
        ]
        cypher = self._cypher_from_write(names)

        # Must use accumulation, not coalesce-set
        assert "coalesce(sn.llm_cost, 0.0) + b.llm_cost" in cypher

    def test_null_cost_does_not_mutate(self):
        """When llm_cost is None, cost fields should not change."""
        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "llm_cost": None,
            }
        ]
        cypher = self._cypher_from_write(names)

        # CASE WHEN b.llm_cost IS NOT NULL pattern — NULL cost → no mutation
        assert "b.llm_cost IS NOT NULL" in cypher


# ---------------------------------------------------------------------------
# Phase 2b — write_reviews review cost propagation
# ---------------------------------------------------------------------------


class TestReviewCostPropagation:
    """``write_reviews`` must propagate review cost to StandardName."""

    def test_review_cost_accumulated_on_standard_name(self):
        """After writing Reviews, StandardName.llm_cost_review_name gets +=."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        records = [
            {
                "id": "electron_temperature:names:grp1:0",
                "standard_name_id": "electron_temperature",
                "model": "test/model",
                "model_family": "test",
                "is_canonical": True,
                "score": 0.8,
                "scores_json": "{}",
                "tier": "good",
                "reviewed_at": "2024-01-01T00:00:00Z",
                "review_axis": "names",
                "cycle_index": 0,
                "review_group_id": "grp1",
                "resolution_role": "primary",
                "resolution_method": "single_review",
                "llm_cost": 0.002,
            }
        ]

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=_make_gc_context(mock_gc),
        ):
            from imas_codex.standard_names.graph_ops import write_reviews

            write_reviews(records)

        queries = _capture_queries(mock_gc)

        # Should have a query that updates StandardName.llm_cost_review_name
        review_cost_queries = [
            (c, k) for c, k in queries if "llm_cost_review_name" in c
        ]
        assert len(review_cost_queries) >= 1, (
            "Expected a Cypher query updating llm_cost_review_name on StandardName"
        )
        cypher = review_cost_queries[0][0]
        assert "coalesce(sn.llm_cost_review_name, 0.0)" in cypher

    def test_review_cached_tokens_written(self):
        """write_reviews passes llm_tokens_cached_read/write to Review node."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        records = [
            {
                "id": "electron_temperature:names:grp1:0",
                "standard_name_id": "electron_temperature",
                "model": "test/model",
                "model_family": "test",
                "is_canonical": True,
                "score": 0.8,
                "scores_json": "{}",
                "tier": "good",
                "reviewed_at": "2024-01-01T00:00:00Z",
                "review_axis": "names",
                "cycle_index": 0,
                "review_group_id": "grp1",
                "resolution_role": "primary",
                "llm_cost": 0.002,
                "llm_tokens_cached_read": 1500,
                "llm_tokens_cached_write": 500,
            }
        ]

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=_make_gc_context(mock_gc),
        ):
            from imas_codex.standard_names.graph_ops import write_reviews

            write_reviews(records)

        queries = _capture_queries(mock_gc)

        # Find the MERGE StandardNameReview Cypher
        review_cypher = [c for c, _ in queries if "MERGE (r:StandardNameReview" in c]
        assert review_cypher, "No MERGE StandardNameReview Cypher found"
        cypher = review_cypher[0]
        assert "r.llm_tokens_cached_read" in cypher
        assert "r.llm_tokens_cached_write" in cypher

        # Verify batch data includes cached token values
        merge_call = [k for c, k in queries if "MERGE (r:StandardNameReview" in c][0]
        batch = merge_call.get("batch", [])
        assert batch[0]["llm_tokens_cached_read"] == 1500
        assert batch[0]["llm_tokens_cached_write"] == 500


# ---------------------------------------------------------------------------
# Phase 2d — sn clear wipes cost fields
# ---------------------------------------------------------------------------


class TestClearWipesCosts:
    """``clear_sn_subsystem`` deletes nodes entirely (DETACH DELETE).

    Since clear uses DETACH DELETE (not SET null), all fields including
    cost fields are wiped. This test verifies that clear_sn_subsystem
    issues DETACH DELETE on StandardName and StandardNameReview nodes.
    """

    def test_clear_detach_deletes_standard_names(self):
        """clear_sn_subsystem deletes StandardName nodes entirely."""
        mock_gc = MagicMock()
        # _count returns 5 for each label
        mock_gc.query = MagicMock(
            side_effect=lambda q, **kw: [{"n": 5}] if "count" in q else []
        )

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=_make_gc_context(mock_gc),
        ):
            from imas_codex.standard_names.graph_ops import clear_sn_subsystem

            clear_sn_subsystem(dry_run=False)

        queries = _capture_queries(mock_gc)
        delete_queries = [c for c, _ in queries if "DETACH DELETE" in c]

        # Must have DETACH DELETE for StandardName
        sn_deletes = [q for q in delete_queries if "StandardName" in q]
        assert sn_deletes, "Expected DETACH DELETE on StandardName"

        # Must have DETACH DELETE for StandardNameReview
        review_deletes = [q for q in delete_queries if "StandardNameReview" in q]
        assert review_deletes, "Expected DETACH DELETE on StandardNameReview"


# ---------------------------------------------------------------------------
# Schema validation — fields exist in YAML
# ---------------------------------------------------------------------------


class TestCostSchemaFields:
    """Verify per-phase cost fields exist in the LinkML schema."""

    @pytest.fixture(autouse=True)
    def _load_schema(self):
        """Load the standard_name schema YAML."""
        import pathlib

        import yaml

        schema_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "imas_codex"
            / "schemas"
            / "standard_name.yaml"
        )
        with open(schema_path) as f:
            self.schema = yaml.safe_load(f)

    def _sn_attrs(self) -> dict:
        return self.schema["classes"]["StandardName"]["attributes"]

    def test_llm_cost_generate_name_exists(self):
        assert "llm_cost_generate_name" in self._sn_attrs()
        assert self._sn_attrs()["llm_cost_generate_name"]["range"] == "float"

    def test_llm_cost_review_name_exists(self):
        assert "llm_cost_review_name" in self._sn_attrs()
        assert self._sn_attrs()["llm_cost_review_name"]["range"] == "float"

    def test_llm_cost_refine_name_exists(self):
        assert "llm_cost_refine_name" in self._sn_attrs()
        assert self._sn_attrs()["llm_cost_refine_name"]["range"] == "float"

    def test_llm_cost_generate_docs_exists(self):
        assert "llm_cost_generate_docs" in self._sn_attrs()
        assert self._sn_attrs()["llm_cost_generate_docs"]["range"] == "float"

    def test_llm_cost_review_docs_exists(self):
        assert "llm_cost_review_docs" in self._sn_attrs()
        assert self._sn_attrs()["llm_cost_review_docs"]["range"] == "float"

    def test_llm_cost_refine_docs_exists(self):
        assert "llm_cost_refine_docs" in self._sn_attrs()
        assert self._sn_attrs()["llm_cost_refine_docs"]["range"] == "float"

    def test_generate_name_count_exists(self):
        assert "generate_name_count" in self._sn_attrs()
        assert self._sn_attrs()["generate_name_count"]["range"] == "integer"

    def test_review_name_count_exists(self):
        assert "review_name_count" in self._sn_attrs()
        assert self._sn_attrs()["review_name_count"]["range"] == "integer"

    def test_refine_name_count_exists(self):
        assert "refine_name_count" in self._sn_attrs()
        assert self._sn_attrs()["refine_name_count"]["range"] == "integer"

    def test_generate_docs_count_exists(self):
        assert "generate_docs_count" in self._sn_attrs()
        assert self._sn_attrs()["generate_docs_count"]["range"] == "integer"

    def test_review_docs_count_exists(self):
        assert "review_docs_count" in self._sn_attrs()
        assert self._sn_attrs()["review_docs_count"]["range"] == "integer"

    def test_refine_docs_count_exists(self):
        assert "refine_docs_count" in self._sn_attrs()
        assert self._sn_attrs()["refine_docs_count"]["range"] == "integer"

    def test_all_pools_have_cost_and_count(self):
        """Every pool name has llm_cost_<pool> and <pool>_count fields."""
        attrs = self._sn_attrs()
        pools = [
            "generate_name",
            "review_name",
            "refine_name",
            "generate_docs",
            "review_docs",
            "refine_docs",
        ]
        for pool in pools:
            cost_field = f"llm_cost_{pool}"
            count_field = f"{pool}_count"
            assert cost_field in attrs, f"Missing cost field: {cost_field}"
            assert attrs[cost_field]["range"] == "float"
            assert count_field in attrs, f"Missing count field: {count_field}"
            assert attrs[count_field]["range"] == "integer"


# ---------------------------------------------------------------------------
# Phase 1a — drop_params config
# ---------------------------------------------------------------------------


class TestDropParamsConfig:
    """Verify LiteLLM config has drop_params: false."""

    def test_litellm_config_drop_params_false(self):
        """litellm_config.yaml must have drop_params: false."""
        import pathlib

        import yaml

        config_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "imas_codex"
            / "config"
            / "litellm_config.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["litellm_settings"]["drop_params"] is False

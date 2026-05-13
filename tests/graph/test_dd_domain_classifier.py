"""Tests for the physics domain classifier module.

All graph and LLM calls are mocked — no Neo4j connection required.
Tests cover:
  - CLASSIFIABLE_CATEGORIES constant
  - compute_domain_input_hash determinism and sensitivity
  - batch_by_subtree grouping and overflow behaviour
  - _format_batch_user_prompt content
  - classify_tier3_none (Tier 3) — infrastructure paths
  - classify_tier2_inherit (Tier 2) — error and metadata inheritance
  - classify_tier1_llm (Tier 1) — LLM path including retry of "general"
  - classify_domains orchestrator — stats keys, dry_run, force
  - Gold set structural integrity (no Neo4j needed)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.graph.dd_domain_classifier import (
    CLASSIFIABLE_CATEGORIES,
    DEFAULT_BATCH_SIZE,
    SERVICE_TAG,
    DomainBatchResult,
    DomainClassification,
    _format_batch_user_prompt,
    _ids_filter_clause,
    batch_by_subtree,
    classify_domains,
    classify_tier2_inherit,
    classify_tier3_none,
    compute_domain_input_hash,
    gather_classification_context,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOLD_SET_PATH = (
    Path(__file__).parent.parent.parent
    / "imas_codex"
    / "definitions"
    / "physics"
    / "domain_gold_set.json"
)

EXPECTED_GOLD_SET_SIZE = 209

# Domains that must NOT appear in the gold set (catch-all / unknown)
FORBIDDEN_DOMAINS = {"general", "unknown", "other"}


@pytest.fixture
def mock_gc():
    """GraphClient mock that returns an empty list by default."""
    gc = MagicMock()
    gc.query.return_value = []
    return gc


@pytest.fixture
def sample_paths():
    """Minimal list of path dicts for batching tests."""
    return [
        {
            "id": "equilibrium/time_slice/profiles_1d/psi",
            "description": "Poloidal flux",
            "units": "Wb",
            "parent_path": "equilibrium/time_slice/profiles_1d",
            "node_category": "quantity",
        },
        {
            "id": "equilibrium/time_slice/profiles_1d/q",
            "description": "Safety factor",
            "units": "",
            "parent_path": "equilibrium/time_slice/profiles_1d",
            "node_category": "quantity",
        },
        {
            "id": "core_profiles/profiles_1d/electrons/temperature",
            "description": "Electron temperature",
            "units": "eV",
            "parent_path": "core_profiles/profiles_1d/electrons",
            "node_category": "quantity",
        },
    ]


# ===========================================================================
# 1. CLASSIFIABLE_CATEGORIES constant
# ===========================================================================


class TestClassifiableCategories:
    """CLASSIFIABLE_CATEGORIES defines which node_categories reach Tier 1."""

    def test_is_frozenset(self):
        assert isinstance(CLASSIFIABLE_CATEGORIES, frozenset)

    @pytest.mark.parametrize(
        "category",
        [
            "quantity",
            "structural",
            "representation",
            "geometry",
            "coordinate",
            "fit_artifact",
            "identifier",
        ],
    )
    def test_expected_categories_present(self, category):
        assert category in CLASSIFIABLE_CATEGORIES

    def test_metadata_not_classifiable(self):
        """'metadata' goes through Tier 2 (inheritance) not Tier 1 LLM."""
        assert "metadata" not in CLASSIFIABLE_CATEGORIES

    def test_service_tag_constant(self):
        assert SERVICE_TAG == "data-dictionary"

    def test_default_batch_size(self):
        assert DEFAULT_BATCH_SIZE == 30


# ===========================================================================
# 2. compute_domain_input_hash
# ===========================================================================


class TestComputeDomainInputHash:
    """Hash function must be deterministic and sensitive to relevant inputs."""

    def _make_ctx(self, description="desc", units="eV", parent_path="eq/ts"):
        return {"description": description, "units": units, "parent_path": parent_path}

    def test_deterministic(self):
        ctx = self._make_ctx()
        assert compute_domain_input_hash(ctx) == compute_domain_input_hash(ctx)

    def test_returns_hex_string(self):
        h = compute_domain_input_hash(self._make_ctx())
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_changes_with_description(self):
        h1 = compute_domain_input_hash(self._make_ctx(description="A"))
        h2 = compute_domain_input_hash(self._make_ctx(description="B"))
        assert h1 != h2

    def test_changes_with_units(self):
        h1 = compute_domain_input_hash(self._make_ctx(units="eV"))
        h2 = compute_domain_input_hash(self._make_ctx(units="keV"))
        assert h1 != h2

    def test_changes_with_parent_path(self):
        h1 = compute_domain_input_hash(self._make_ctx(parent_path="a/b"))
        h2 = compute_domain_input_hash(self._make_ctx(parent_path="a/c"))
        assert h1 != h2

    def test_handles_missing_keys_gracefully(self):
        """Missing keys default to empty string — should not raise."""
        h = compute_domain_input_hash({})
        # empty|empty|empty → valid SHA-256
        expected = hashlib.sha256(b"||").hexdigest()
        assert h == expected

    def test_none_values_treated_as_empty(self):
        ctx_none = {"description": None, "units": None, "parent_path": None}
        ctx_empty = {"description": "", "units": "", "parent_path": ""}
        assert compute_domain_input_hash(ctx_none) == compute_domain_input_hash(
            ctx_empty
        )

    def test_order_of_fields_matters(self):
        """Hash encodes fields in a fixed order — swapping values changes hash."""
        h1 = compute_domain_input_hash(
            {"description": "X", "units": "Y", "parent_path": "Z"}
        )
        # Same tokens, different positions
        h2 = compute_domain_input_hash(
            {"description": "Y", "units": "X", "parent_path": "Z"}
        )
        assert h1 != h2


# ===========================================================================
# 3. batch_by_subtree
# ===========================================================================


class TestBatchBySubtree:
    """batch_by_subtree groups paths by parent and respects batch_size."""

    def _make_path(self, path_id, parent):
        return {"id": path_id, "parent_path": parent}

    def test_empty_input_returns_empty(self):
        assert batch_by_subtree([]) == []

    def test_single_path_single_batch(self):
        paths = [self._make_path("a/b", "a")]
        batches = batch_by_subtree(paths)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_same_parent_grouped_together(self):
        paths = [
            self._make_path("eq/ts/a", "eq/ts"),
            self._make_path("eq/ts/b", "eq/ts"),
            self._make_path("eq/ts/c", "eq/ts"),
        ]
        batches = batch_by_subtree(paths, batch_size=10)
        # All three should be in one batch
        all_ids = {p["id"] for batch in batches for p in batch}
        assert all_ids == {"eq/ts/a", "eq/ts/b", "eq/ts/c"}
        assert len(batches) == 1

    def test_overflow_splits_large_group(self):
        """A group larger than batch_size must be split into multiple batches."""
        paths = [self._make_path(f"a/b/{i}", "a/b") for i in range(10)]
        batches = batch_by_subtree(paths, batch_size=3)
        # 10 paths / 3 = 4 batches (3+3+3+1)
        assert len(batches) == 4
        total = sum(len(b) for b in batches)
        assert total == 10

    def test_different_parents_separate_batches(self):
        """Paths from different parents may end up in different batches."""
        paths = [
            self._make_path("a/x", "a"),
        ] * 10 + [
            self._make_path("b/y", "b"),
        ] * 10
        # batch_size=12 → each group of 10 fits, but together they overflow
        batches = batch_by_subtree(paths, batch_size=12)
        total = sum(len(b) for b in batches)
        assert total == 20

    def test_no_path_duplicated(self, sample_paths):
        batches = batch_by_subtree(sample_paths, batch_size=2)
        all_ids = [p["id"] for batch in batches for p in batch]
        assert len(all_ids) == len(sample_paths)

    def test_batch_size_respected(self):
        paths = [self._make_path(f"p/{i}", "p") for i in range(100)]
        batches = batch_by_subtree(paths, batch_size=25)
        for batch in batches:
            assert len(batch) <= 25

    def test_paths_without_parent_path_use_id_prefix(self):
        """Paths without parent_path fall back to splitting the id."""
        paths = [{"id": "eq/ts/psi"}]  # no parent_path key
        batches = batch_by_subtree(paths)
        assert len(batches) == 1
        assert batches[0][0]["id"] == "eq/ts/psi"


# ===========================================================================
# 4. _format_batch_user_prompt
# ===========================================================================


class TestFormatBatchUserPrompt:
    """_format_batch_user_prompt produces numbered, human-readable text."""

    def test_numbering_starts_at_one(self, sample_paths):
        text = _format_batch_user_prompt(sample_paths)
        assert "1. path:" in text
        assert "2. path:" in text
        assert "3. path:" in text

    def test_includes_description(self, sample_paths):
        text = _format_batch_user_prompt(sample_paths)
        assert "Poloidal flux" in text

    def test_includes_units(self, sample_paths):
        text = _format_batch_user_prompt(sample_paths)
        assert "Wb" in text

    def test_includes_parent_path(self, sample_paths):
        text = _format_batch_user_prompt(sample_paths)
        assert "equilibrium/time_slice/profiles_1d" in text

    def test_empty_input_returns_empty_string(self):
        assert _format_batch_user_prompt([]) == ""

    def test_missing_fields_show_na(self):
        text = _format_batch_user_prompt([{"id": "a/b"}])
        assert "N/A" in text


# ===========================================================================
# 5. _ids_filter_clause
# ===========================================================================


class TestIdsFilterClause:
    def test_none_returns_empty_string(self):
        assert _ids_filter_clause(None, "n") == ""

    def test_empty_set_returns_empty_string(self):
        # An empty set means no filter
        result = _ids_filter_clause(set(), "n")
        assert result == ""

    def test_single_ids_produces_where_fragment(self):
        result = _ids_filter_clause({"equilibrium"}, "n")
        assert "equilibrium" in result
        assert "n.id" in result

    def test_multiple_ids_included(self):
        result = _ids_filter_clause({"equilibrium", "core_profiles"}, "n")
        assert "equilibrium" in result
        assert "core_profiles" in result

    def test_uses_specified_node_var(self):
        result = _ids_filter_clause({"magnetics"}, "parent")
        assert "parent.id" in result
        assert "n.id" not in result


# ===========================================================================
# 6. classify_tier3_none  (Tier 3 — infrastructure → None)
# ===========================================================================


class TestClassifyTier3None:
    """Tier 3: infrastructure paths (ids_properties/*, code/*) get None domain."""

    def test_calls_graph_query(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 5}]
        result = classify_tier3_none(mock_gc)
        assert mock_gc.query.called
        assert result == 5

    def test_returns_zero_on_empty_result(self, mock_gc):
        mock_gc.query.return_value = []
        assert classify_tier3_none(mock_gc) == 0

    def test_dry_run_reads_count_not_update(self, mock_gc):
        mock_gc.query.return_value = [{"cnt": 42}]
        result = classify_tier3_none(mock_gc, dry_run=True)
        cypher = mock_gc.query.call_args[0][0]
        # dry_run queries RETURN count, NOT SET
        assert "RETURN count" in cypher or "cnt" in cypher
        assert "SET" not in cypher
        assert result == 42

    def test_live_run_uses_set_clause(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 3}]
        classify_tier3_none(mock_gc, dry_run=False)
        cypher = mock_gc.query.call_args[0][0]
        assert "SET" in cypher

    def test_ids_filter_narrows_query(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 1}]
        classify_tier3_none(mock_gc, ids_filter={"equilibrium"})
        cypher = mock_gc.query.call_args[0][0]
        assert "equilibrium" in cypher

    def test_targets_ids_properties_and_code_subtrees(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 0}]
        classify_tier3_none(mock_gc)
        cypher = mock_gc.query.call_args[0][0]
        assert "ids_properties" in cypher
        assert "/code/" in cypher

    def test_sets_domain_source_none_metadata(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 1}]
        classify_tier3_none(mock_gc, dry_run=False)
        cypher = mock_gc.query.call_args[0][0]
        assert "none_metadata" in cypher


# ===========================================================================
# 7. classify_tier2_inherit  (Tier 2 — inheritance)
# ===========================================================================


class TestClassifyTier2Inherit:
    """Tier 2: error paths and non-infrastructure metadata inherit from parent."""

    def test_returns_sum_of_two_passes(self, mock_gc):
        # Two query calls: error parent + metadata parent — each returns a list
        mock_gc.query.side_effect = [[{"updated": 10}], [{"updated": 5}]]
        total = classify_tier2_inherit(mock_gc)
        assert total == 15

    def test_empty_results_return_zero(self, mock_gc):
        mock_gc.query.return_value = []
        assert classify_tier2_inherit(mock_gc) == 0

    def test_error_path_query_uses_has_error_relation(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 0}]
        classify_tier2_inherit(mock_gc)
        calls = [str(c) for c in mock_gc.query.call_args_list]
        assert any("HAS_ERROR" in c for c in calls)

    def test_metadata_path_query_uses_has_parent_relation(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 0}]
        classify_tier2_inherit(mock_gc)
        calls = [str(c) for c in mock_gc.query.call_args_list]
        assert any("HAS_PARENT" in c for c in calls)

    def test_dry_run_does_not_use_set(self, mock_gc):
        mock_gc.query.return_value = [{"cnt": 3}]
        classify_tier2_inherit(mock_gc, dry_run=True)
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            assert "SET" not in cypher

    def test_force_skips_domain_check(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 0}]
        classify_tier2_inherit(mock_gc, force=True)
        calls = [str(c) for c in mock_gc.query.call_args_list]
        # When force=True the "needs_clause" (which mentions 'general') is omitted
        for c in calls:
            assert "physics_domain IS NULL" not in c

    def test_no_force_filters_already_classified(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 0}]
        classify_tier2_inherit(mock_gc, force=False)
        calls = [str(c) for c in mock_gc.query.call_args_list]
        # At least one query should check for unclassified nodes
        assert any("physics_domain IS NULL" in c or "general" in c for c in calls)

    def test_inherited_domain_source_label(self, mock_gc):
        mock_gc.query.return_value = [{"updated": 5}]
        classify_tier2_inherit(mock_gc, dry_run=False)
        calls = [str(c) for c in mock_gc.query.call_args_list]
        assert any("inherited_from_parent" in c for c in calls)

    def test_fallback_on_missing_result_key(self, mock_gc):
        """If query returns no rows, should return 0, not raise."""
        mock_gc.query.return_value = []
        result = classify_tier2_inherit(mock_gc)
        assert result == 0


# ===========================================================================
# 8. gather_classification_context
# ===========================================================================


class TestGatherClassificationContext:
    """gather_classification_context fetches rich context from the graph."""

    def test_empty_ids_returns_empty_list(self, mock_gc):
        result = gather_classification_context(mock_gc, [])
        assert result == []
        mock_gc.query.assert_not_called()

    def test_returns_list_of_dicts(self, mock_gc):
        mock_gc.query.return_value = [
            {
                "id": "eq/ts/psi",
                "description": "Poloidal flux",
                "units": "Wb",
                "parent_path": "eq/ts",
                "parent_description": "Time slice",
                "siblings": ["q", "pprime"],
                "ids_name": "equilibrium",
                "node_category": "quantity",
            }
        ]
        result = gather_classification_context(mock_gc, ["eq/ts/psi"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_context_keys_present(self, mock_gc):
        mock_gc.query.return_value = [
            {
                "id": "eq/ts/psi",
                "description": "Poloidal flux",
                "units": "Wb",
                "parent_path": "eq/ts",
                "parent_description": "Time slice",
                "siblings": [],
                "ids_name": "equilibrium",
                "node_category": "quantity",
            }
        ]
        ctx = gather_classification_context(mock_gc, ["eq/ts/psi"])[0]
        for key in (
            "id",
            "description",
            "units",
            "parent_path",
            "parent_description",
            "siblings",
            "ids_name",
            "node_category",
        ):
            assert key in ctx, f"Missing key: {key}"

    def test_none_values_normalised_to_empty(self, mock_gc):
        mock_gc.query.return_value = [
            {
                "id": "a/b",
                "description": None,
                "units": None,
                "parent_path": None,
                "parent_description": None,
                "siblings": None,
                "ids_name": None,
                "node_category": None,
            }
        ]
        ctx = gather_classification_context(mock_gc, ["a/b"])[0]
        assert ctx["description"] == ""
        assert ctx["units"] == ""
        assert ctx["parent_path"] == ""
        assert ctx["siblings"] == []


# ===========================================================================
# 9. DomainBatchResult / DomainClassification Pydantic models
# ===========================================================================


class TestPydanticModels:
    """Pydantic models must validate correctly."""

    def test_domain_classification_valid(self):
        dc = DomainClassification(path_index=1, physics_domain="equilibrium")
        assert dc.path_index == 1
        assert dc.physics_domain == "equilibrium"

    def test_domain_batch_result_empty(self):
        dbr = DomainBatchResult(classifications=[])
        assert dbr.classifications == []

    def test_domain_batch_result_with_items(self):
        dc = DomainClassification(path_index=2, physics_domain="transport")
        dbr = DomainBatchResult(classifications=[dc])
        assert len(dbr.classifications) == 1
        assert dbr.classifications[0].physics_domain == "transport"


# ===========================================================================
# 10. classify_tier1_llm — async, with mocked LLM
# ===========================================================================


class TestClassifyTier1Llm:
    """Tier 1 LLM classification with a fully mocked LLM call."""

    def _make_llm_result(self, path_index: int, domain: str, model: str = "test-model"):
        """Build a fake (result_obj, cost, tokens) triple."""
        result_obj = DomainBatchResult(
            classifications=[
                DomainClassification(path_index=path_index, physics_domain=domain)
            ]
        )
        return result_obj, 0.001, {"prompt": 10, "completion": 5}

    @pytest.mark.asyncio
    async def test_classifies_single_path(self, mock_gc):
        from imas_codex.graph.dd_domain_classifier import classify_tier1_llm

        mock_gc.query.return_value = [
            {
                "id": "equilibrium/time_slice/profiles_1d/psi",
                "description": "Poloidal flux",
                "units": "Wb",
                "parent_path": "equilibrium/time_slice/profiles_1d",
                "parent_description": "",
                "siblings": [],
                "ids_name": "equilibrium",
                "node_category": "quantity",
            }
        ]

        llm_return = self._make_llm_result(1, "equilibrium")

        with patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=llm_return,
        ):
            with patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="system-prompt",
            ):
                paths = [
                    {
                        "id": "equilibrium/time_slice/profiles_1d/psi",
                        "node_category": "quantity",
                        "description": "Poloidal flux",
                        "units": "Wb",
                    }
                ]
                results = await classify_tier1_llm(mock_gc, paths, model="test-model")

        assert len(results) >= 1
        assert results[0]["physics_domain"] == "equilibrium"

    @pytest.mark.asyncio
    async def test_invalid_domain_falls_back_to_general(self, mock_gc):
        from imas_codex.graph.dd_domain_classifier import classify_tier1_llm

        mock_gc.query.return_value = [
            {
                "id": "eq/ts/psi",
                "description": "",
                "units": "",
                "parent_path": "",
                "parent_description": "",
                "siblings": [],
                "ids_name": "equilibrium",
                "node_category": "quantity",
            }
        ]

        bad_result = DomainBatchResult(
            classifications=[
                DomainClassification(
                    path_index=1, physics_domain="not_a_real_domain_xyz"
                )
            ]
        )

        with patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=(bad_result, 0.001, {}),
        ):
            with patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="system-prompt",
            ):
                paths = [{"id": "eq/ts/psi", "node_category": "quantity"}]
                results = await classify_tier1_llm(mock_gc, paths, model="test-model")

        # Should fall back to general for any retry target
        assert any(r["physics_domain"] == "general" for r in results)

    @pytest.mark.asyncio
    async def test_general_paths_trigger_retry(self, mock_gc):
        """Paths that get 'general' on first pass should trigger a retry batch."""
        from imas_codex.graph.dd_domain_classifier import classify_tier1_llm

        context_row = {
            "id": "eq/ts/psi",
            "description": "",
            "units": "",
            "parent_path": "",
            "parent_description": "",
            "siblings": [],
            "ids_name": "equilibrium",
            "node_category": "quantity",
            "cluster_peers": [],
        }
        mock_gc.query.return_value = [context_row]

        general_result = DomainBatchResult(
            classifications=[
                DomainClassification(path_index=1, physics_domain="general")
            ]
        )
        retry_result = DomainBatchResult(
            classifications=[
                DomainClassification(path_index=1, physics_domain="equilibrium")
            ]
        )

        call_count = {"n": 0}

        async def fake_llm(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return general_result, 0.001, {}
            return retry_result, 0.001, {}

        with patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=fake_llm,
        ):
            with patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="system-prompt",
            ):
                paths = [{"id": "eq/ts/psi", "node_category": "quantity"}]
                results = await classify_tier1_llm(mock_gc, paths, model="test-model")

        # LLM was called at least twice (initial + retry)
        assert call_count["n"] >= 2
        # Final result should NOT be "general" after the successful retry
        final_domain = results[-1]["physics_domain"]
        assert final_domain == "equilibrium"

    @pytest.mark.asyncio
    async def test_on_cost_callback_called(self, mock_gc):
        from imas_codex.graph.dd_domain_classifier import classify_tier1_llm

        mock_gc.query.return_value = [
            {
                "id": "eq/ts/psi",
                "description": "",
                "units": "",
                "parent_path": "",
                "parent_description": "",
                "siblings": [],
                "ids_name": "equilibrium",
                "node_category": "quantity",
            }
        ]

        llm_return = self._make_llm_result(1, "equilibrium")

        costs: list[float] = []

        with patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=llm_return,
        ):
            with patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="system-prompt",
            ):
                paths = [{"id": "eq/ts/psi", "node_category": "quantity"}]
                await classify_tier1_llm(
                    mock_gc,
                    paths,
                    model="test-model",
                    on_cost=costs.append,
                )

        assert len(costs) >= 1


# ===========================================================================
# 11. classify_domains orchestrator  (async)
# ===========================================================================


class TestClassifyDomains:
    """Test the top-level orchestrator for correct stats and control flow."""

    def _patch_settings(self, model="test-model"):
        return patch(
            "imas_codex.settings.get_model",
            return_value=model,
        )

    @pytest.mark.asyncio
    async def test_stats_keys_present(self, mock_gc):
        with self._patch_settings():
            with patch(
                "imas_codex.graph.dd_domain_classifier.classify_tier3_none",
                return_value=10,
            ):
                with patch(
                    "imas_codex.graph.dd_domain_classifier.classify_tier2_inherit",
                    return_value=20,
                ):
                    with patch(
                        "imas_codex.graph.dd_domain_classifier._query_unclassified_paths",
                        return_value=[],
                    ):
                        stats = await classify_domains(mock_gc)

        assert "tier3_none" in stats
        assert "tier2_inherited" in stats
        assert "tier1_llm" in stats
        assert "tier1_general" in stats
        assert "tier1_retried" in stats
        assert "total_cost" in stats
        assert "model" in stats

    @pytest.mark.asyncio
    async def test_stats_counts_correct(self, mock_gc):
        with self._patch_settings():
            with patch(
                "imas_codex.graph.dd_domain_classifier.classify_tier3_none",
                return_value=7,
            ):
                with patch(
                    "imas_codex.graph.dd_domain_classifier.classify_tier2_inherit",
                    return_value=13,
                ):
                    with patch(
                        "imas_codex.graph.dd_domain_classifier._query_unclassified_paths",
                        return_value=[],
                    ):
                        stats = await classify_domains(mock_gc)

        assert stats["tier3_none"] == 7
        assert stats["tier2_inherited"] == 13

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_tier1_llm(self, mock_gc):
        """dry_run=True skips the LLM tier entirely."""
        with self._patch_settings():
            with patch(
                "imas_codex.graph.dd_domain_classifier.classify_tier3_none",
                return_value=0,
            ):
                with patch(
                    "imas_codex.graph.dd_domain_classifier.classify_tier2_inherit",
                    return_value=0,
                ):
                    with patch(
                        "imas_codex.graph.dd_domain_classifier._query_unclassified_paths",
                        return_value=[{"id": "eq/psi", "node_category": "quantity"}],
                    ):
                        llm_mock = AsyncMock()
                        with patch(
                            "imas_codex.graph.dd_domain_classifier.classify_tier1_llm",
                            llm_mock,
                        ):
                            await classify_domains(mock_gc, dry_run=True)

        # When dry_run=True and paths exist, LLM should NOT be called
        llm_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_default_used_from_settings(self, mock_gc):
        with patch(
            "imas_codex.settings.get_model",
            return_value="settings-model",
        ) as mock_get:
            with patch(
                "imas_codex.graph.dd_domain_classifier.classify_tier3_none",
                return_value=0,
            ):
                with patch(
                    "imas_codex.graph.dd_domain_classifier.classify_tier2_inherit",
                    return_value=0,
                ):
                    with patch(
                        "imas_codex.graph.dd_domain_classifier._query_unclassified_paths",
                        return_value=[],
                    ):
                        stats = await classify_domains(mock_gc)

        mock_get.assert_called_once_with("language")
        assert stats["model"] == "settings-model"

    @pytest.mark.asyncio
    async def test_model_override_bypasses_settings(self, mock_gc):
        with patch("imas_codex.settings.get_model") as mock_get:
            with patch(
                "imas_codex.graph.dd_domain_classifier.classify_tier3_none",
                return_value=0,
            ):
                with patch(
                    "imas_codex.graph.dd_domain_classifier.classify_tier2_inherit",
                    return_value=0,
                ):
                    with patch(
                        "imas_codex.graph.dd_domain_classifier._query_unclassified_paths",
                        return_value=[],
                    ):
                        stats = await classify_domains(mock_gc, model="custom-model")

        mock_get.assert_not_called()
        assert stats["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_on_items_callback_called(self, mock_gc):
        counts: list[int] = []
        with self._patch_settings():
            with patch(
                "imas_codex.graph.dd_domain_classifier.classify_tier3_none",
                return_value=3,
            ):
                with patch(
                    "imas_codex.graph.dd_domain_classifier.classify_tier2_inherit",
                    return_value=5,
                ):
                    with patch(
                        "imas_codex.graph.dd_domain_classifier._query_unclassified_paths",
                        return_value=[],
                    ):
                        await classify_domains(mock_gc, on_items=counts.append)

        assert counts  # At least one call
        # Final call reports tier3 + tier2 + tier1
        assert 8 in counts  # 3 + 5 + 0

    @pytest.mark.asyncio
    async def test_force_passed_to_tier2(self, mock_gc):
        """force=True must be forwarded to classify_tier2_inherit."""
        with self._patch_settings():
            with patch(
                "imas_codex.graph.dd_domain_classifier.classify_tier3_none",
                return_value=0,
            ):
                tier2_mock = MagicMock(return_value=0)
                with patch(
                    "imas_codex.graph.dd_domain_classifier.classify_tier2_inherit",
                    tier2_mock,
                ):
                    with patch(
                        "imas_codex.graph.dd_domain_classifier._query_unclassified_paths",
                        return_value=[],
                    ):
                        await classify_domains(mock_gc, force=True)

        call_kwargs = tier2_mock.call_args[1]
        assert call_kwargs.get("force") is True

    @pytest.mark.asyncio
    async def test_ids_filter_passed_through(self, mock_gc):
        """ids_filter must reach all tier functions."""
        with self._patch_settings():
            tier3_mock = MagicMock(return_value=0)
            tier2_mock = MagicMock(return_value=0)
            query_mock = MagicMock(return_value=[])
            with patch(
                "imas_codex.graph.dd_domain_classifier.classify_tier3_none",
                tier3_mock,
            ):
                with patch(
                    "imas_codex.graph.dd_domain_classifier.classify_tier2_inherit",
                    tier2_mock,
                ):
                    with patch(
                        "imas_codex.graph.dd_domain_classifier._query_unclassified_paths",
                        query_mock,
                    ):
                        await classify_domains(mock_gc, ids_filter={"equilibrium"})

        assert tier3_mock.call_args[1]["ids_filter"] == {"equilibrium"}
        assert tier2_mock.call_args[1]["ids_filter"] == {"equilibrium"}
        assert query_mock.call_args[1]["ids_filter"] == {"equilibrium"}


# ===========================================================================
# 12. Gold set structural integrity  (no Neo4j required)
# ===========================================================================


class TestGoldSet:
    """domain_gold_set.json must satisfy structural invariants."""

    @pytest.fixture(scope="class")
    def gold_set(self):
        assert GOLD_SET_PATH.exists(), f"Gold set not found: {GOLD_SET_PATH}"
        return json.loads(GOLD_SET_PATH.read_text())

    def test_gold_set_is_list(self, gold_set):
        assert isinstance(gold_set, list)

    def test_gold_set_size(self, gold_set):
        assert len(gold_set) == EXPECTED_GOLD_SET_SIZE, (
            f"Expected {EXPECTED_GOLD_SET_SIZE} entries, got {len(gold_set)}"
        )

    def test_every_entry_has_required_keys(self, gold_set):
        required_keys = {"path", "expected_domain"}
        for entry in gold_set:
            missing = required_keys - entry.keys()
            assert not missing, f"Entry {entry} missing keys: {missing}"

    def test_no_general_domain(self, gold_set):
        """Gold set must not contain the 'general' catch-all domain."""
        general_entries = [e for e in gold_set if e["expected_domain"] == "general"]
        assert not general_entries, (
            f"Found {len(general_entries)} 'general' entries in gold set"
        )

    def test_no_forbidden_domains(self, gold_set):
        for entry in gold_set:
            assert entry["expected_domain"] not in FORBIDDEN_DOMAINS, (
                f"Entry {entry['path']} has forbidden domain {entry['expected_domain']!r}"
            )

    def test_all_domains_are_valid_physics_domains(self, gold_set):
        """Every expected_domain must be a valid PhysicsDomain value."""
        # Use the same domains discovered from gold set (avoids import dependency)
        known_domains = {
            "auxiliary_heating",
            "computational_workflow",
            "data_management",
            "divertor_physics",
            "edge_plasma_physics",
            "electromagnetic_wave_diagnostics",
            "equilibrium",
            "machine_operations",
            "magnetic_field_diagnostics",
            "magnetic_field_systems",
            "magnetohydrodynamics",
            "plant_systems",
            "plasma_control",
            "plasma_wall_interactions",
            "transport",
            "turbulence",
        }
        for entry in gold_set:
            assert entry["expected_domain"] in known_domains, (
                f"Unknown domain {entry['expected_domain']!r} for path {entry['path']}"
            )

    def test_transport_is_largest_domain(self, gold_set):
        """Transport is the most represented domain (physics workload check)."""
        from collections import Counter

        counts = Counter(e["expected_domain"] for e in gold_set)
        assert counts["transport"] == max(counts.values()), (
            f"Expected 'transport' to have the most entries, "
            f"got distribution: {counts.most_common(3)}"
        )

    def test_paths_are_non_empty_strings(self, gold_set):
        for entry in gold_set:
            assert isinstance(entry["path"], str)
            assert entry["path"].strip(), f"Empty path in entry: {entry}"

    def test_no_duplicate_paths(self, gold_set):
        paths = [e["path"] for e in gold_set]
        duplicates = [p for p in paths if paths.count(p) > 1]
        assert not duplicates, f"Duplicate paths in gold set: {set(duplicates)}"

    def test_multiple_domains_represented(self, gold_set):
        """Gold set must span at least 10 distinct domains for good coverage."""
        domains = {e["expected_domain"] for e in gold_set}
        assert len(domains) >= 10, f"Only {len(domains)} domains in gold set"

    def test_equilibrium_well_represented(self, gold_set):
        """Equilibrium is a core physics domain with a minimum number of entries."""
        eq_count = sum(1 for e in gold_set if e["expected_domain"] == "equilibrium")
        assert eq_count >= 10, f"Too few equilibrium entries: {eq_count}"

    @pytest.mark.parametrize(
        "path_prefix, expected_domain",
        [
            ("summary/boundary/elongation", "equilibrium"),
            ("summary/boundary/geometric_axis_r", "equilibrium"),
        ],
    )
    def test_known_path_domain_assignment(self, gold_set, path_prefix, expected_domain):
        """Spot-check that specific paths map to the expected domain."""
        by_path = {e["path"]: e["expected_domain"] for e in gold_set}
        assert path_prefix in by_path, f"Path {path_prefix!r} not in gold set"
        assert by_path[path_prefix] == expected_domain
